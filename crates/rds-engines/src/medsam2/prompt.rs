//! The prompt encoder: clicks, boxes and masks into tokens.
//!
//! Sparse prompts (clicks and box corners) become one token each, positioned
//! by a **random Fourier** encoding whose Gaussian matrix is a buffer in the
//! checkpoint - the one non-parameter tensor in the file. Dense prompts (an
//! existing mask) are convolved down to the embedding grid instead; when
//! there is no mask, a single learned `no_mask` embedding is broadcast over
//! it.
//!
//! Two token-count details decide whether the decoder sees what the reference
//! sees. The video predictor never passes a box through the `boxes` argument:
//! it turns the box into two points labelled 2 and 3, which makes
//! `pad = (boxes is None)` true and appends a `not_a_point` token. So a box
//! prompt is **three** sparse tokens, and a propagated slice with no prompt at
//! all is **two** - a synthesized `(0, 0)` point with label -1, plus the same
//! padding token.

use anyhow::Result;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config;
use super::layers::{Conv, Norm};
use super::ops;

/// SAM's point labels.
pub const LABEL_PAD: i32 = -1;
pub const LABEL_NEGATIVE: i32 = 0;
pub const LABEL_POSITIVE: i32 = 1;
pub const LABEL_BOX_MIN: i32 = 2;
pub const LABEL_BOX_MAX: i32 = 3;

/// One sparse prompt, in pixels of the 512 x 512 input the network sees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub label: i32,
}

impl Point {
    pub fn positive(x: f32, y: f32) -> Point {
        Point {
            x,
            y,
            label: LABEL_POSITIVE,
        }
    }

    pub fn negative(x: f32, y: f32) -> Point {
        Point {
            x,
            y,
            label: LABEL_NEGATIVE,
        }
    }

    /// A box, as the two corner points SAM 2 encodes it as.
    pub fn box_corners(x0: f32, y0: f32, x1: f32, y1: f32) -> [Point; 2] {
        [
            Point {
                x: x0,
                y: y0,
                label: LABEL_BOX_MIN,
            },
            Point {
                x: x1,
                y: y1,
                label: LABEL_BOX_MAX,
            },
        ]
    }

    /// What a slice with no prompt sends: one padding point at the origin.
    pub fn none() -> [Point; 1] {
        [Point {
            x: 0.0,
            y: 0.0,
            label: LABEL_PAD,
        }]
    }
}

/// Sparse and dense prompt embeddings.
pub struct Prompts<B: Backend> {
    /// `[1, tokens, 256]`.
    pub sparse: Tensor<B, 3>,
    /// `[1, 256, 32, 32]`.
    pub dense: Tensor<B, 4>,
}

pub struct PromptEncoder<B: Backend> {
    /// `[2, 128]`, kept on the host: it multiplies at most a handful of
    /// points per slice, and doing it here keeps the tiny matmul off the
    /// device.
    gaussian: Vec<f32>,
    /// Negative, positive, box-min, box-max.
    point_embeddings: Vec<Tensor<B, 3>>,
    not_a_point: Tensor<B, 3>,
    no_mask: Tensor<B, 3>,
    mask_conv0: Conv<B>,
    mask_norm0: Norm<B>,
    mask_conv1: Conv<B>,
    mask_norm1: Norm<B>,
    mask_conv2: Conv<B>,
    /// `get_dense_pe()`, built once: it depends only on the buffer and the
    /// grid.
    dense_pe: Tensor<B, 4>,
}

impl<B: Backend> PromptEncoder<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<PromptEncoder<B>> {
        let e = "sam_prompt_encoder";
        let d = config::D_MODEL;
        let gaussian = p
            .get(
                &format!("{e}.pe_layer.positional_encoding_gaussian_matrix"),
                &[2, config::PE_GAUSSIAN],
            )?
            .to_vec();
        let embed = |key: &str| -> Result<Tensor<B, 3>> {
            Ok(ops::from_slice(p.get(key, &[1, d])?, [1, 1, d], dev))
        };
        let mut point_embeddings = Vec::with_capacity(4);
        for i in 0..4 {
            point_embeddings.push(embed(&format!("{e}.point_embeddings.{i}.weight"))?);
        }
        let c = config::MASK_IN_CHANS;
        let encoder = PromptEncoder {
            dense_pe: Self::build_dense_pe(&gaussian, dev),
            gaussian,
            point_embeddings,
            not_a_point: embed(&format!("{e}.not_a_point_embed.weight"))?,
            no_mask: embed(&format!("{e}.no_mask_embed.weight"))?,
            mask_conv0: Conv::load(
                p,
                &format!("{e}.mask_downscaling.0"),
                c / 4,
                1,
                2,
                2,
                0,
                1,
                dev,
            )?,
            mask_norm0: Norm::load6(p, &format!("{e}.mask_downscaling.1"), c / 4, dev)?,
            mask_conv1: Conv::load(
                p,
                &format!("{e}.mask_downscaling.3"),
                c,
                c / 4,
                2,
                2,
                0,
                1,
                dev,
            )?,
            mask_norm1: Norm::load6(p, &format!("{e}.mask_downscaling.4"), c, dev)?,
            mask_conv2: Conv::load_1x1(p, &format!("{e}.mask_downscaling.6"), d, c, dev)?,
        };
        Ok(encoder)
    }

    /// `_pe_encoding`: `[sin, cos]` of `2π (2c - 1) G`, concatenated.
    fn fourier(&self, x_norm: f32, y_norm: f32) -> Vec<f32> {
        let half = config::PE_GAUSSIAN;
        let (cx, cy) = (2.0 * f64::from(x_norm) - 1.0, 2.0 * f64::from(y_norm) - 1.0);
        let mut out = vec![0f32; 2 * half];
        for j in 0..half {
            let g0 = f64::from(self.gaussian[j]);
            let g1 = f64::from(self.gaussian[half + j]);
            let v = 2.0 * std::f64::consts::PI * (cx * g0 + cy * g1);
            out[j] = v.sin() as f32;
            out[half + j] = v.cos() as f32;
        }
        out
    }

    /// The dense positional encoding of the embedding grid, `[1, 256, 32, 32]`.
    fn build_dense_pe(gaussian: &[f32], dev: &B::Device) -> Tensor<B, 4> {
        let (g, half) = (config::EMBED_GRID, config::PE_GAUSSIAN);
        let mut data = vec![0f32; 2 * half * g * g];
        let plane = g * g;
        for y in 0..g {
            let cy = 2.0 * ((y as f64 + 0.5) / g as f64) - 1.0;
            for x in 0..g {
                let cx = 2.0 * ((x as f64 + 0.5) / g as f64) - 1.0;
                let at = y * g + x;
                for j in 0..half {
                    let v = 2.0
                        * std::f64::consts::PI
                        * (cx * f64::from(gaussian[j]) + cy * f64::from(gaussian[half + j]));
                    data[j * plane + at] = v.sin() as f32;
                    data[(half + j) * plane + at] = v.cos() as f32;
                }
            }
        }
        ops::from_slice(&data, [1, 2 * half, g, g], dev)
    }

    /// `get_dense_pe()`.
    pub fn dense_pe(&self) -> Tensor<B, 4> {
        self.dense_pe.clone()
    }

    /// `mask_downscaling`, for a `[1, 1, 128, 128]` mask prompt.
    fn embed_mask(&self, mask: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = gelu(self.mask_norm0.apply_2d(self.mask_conv0.apply(mask)));
        let x = gelu(self.mask_norm1.apply_2d(self.mask_conv1.apply(x)));
        self.mask_conv2.apply(x)
    }

    /// Encode a prompt. A padding point is always appended, exactly as
    /// `pad = (boxes is None)` does in the reference.
    pub fn encode(&self, points: &[Point], mask: Option<Tensor<B, 4>>) -> Prompts<B> {
        let dev = self.not_a_point.device();
        let d = config::D_MODEL;
        let size = config::IMAGE_SIZE as f32;

        let mut padded: Vec<Point> = points.to_vec();
        // The shift to pixel centres happens *before* the padding point is
        // appended, so the padding point stays at exactly (0, 0).
        for p in padded.iter_mut() {
            p.x += 0.5;
            p.y += 0.5;
        }
        padded.push(Point {
            x: 0.0,
            y: 0.0,
            label: LABEL_PAD,
        });

        let mut tokens: Vec<Tensor<B, 3>> = Vec::with_capacity(padded.len());
        for p in &padded {
            let pe: Tensor<B, 3> =
                ops::from_slice(&self.fourier(p.x / size, p.y / size), [1, 1, d], &dev);
            tokens.push(match p.label {
                LABEL_PAD => self.not_a_point.clone(),
                l if (0..4).contains(&l) => pe + self.point_embeddings[l as usize].clone(),
                other => panic!("unknown point label {other}"),
            });
        }

        let dense = match mask {
            Some(m) => self.embed_mask(m),
            None => {
                let g = config::EMBED_GRID;
                self.no_mask
                    .clone()
                    .reshape([1, d, 1, 1])
                    .repeat_dim(2, g)
                    .repeat_dim(3, g)
            }
        };
        Prompts {
            sparse: Tensor::cat(tokens, 1),
            dense,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_is_two_labelled_corners() {
        let [a, b] = Point::box_corners(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.label, LABEL_BOX_MIN);
        assert_eq!(b.label, LABEL_BOX_MAX);
        assert_eq!((a.x, a.y, b.x, b.y), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(Point::none()[0].label, LABEL_PAD);
        assert_eq!(Point::positive(1.0, 1.0).label, LABEL_POSITIVE);
        assert_eq!(Point::negative(1.0, 1.0).label, LABEL_NEGATIVE);
    }
}
