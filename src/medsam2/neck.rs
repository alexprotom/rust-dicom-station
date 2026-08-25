//! The image encoder's FPN neck, and the sine positional encoding it emits.
//!
//! Four 1 x 1 convolutions bring the trunk's stages to a common width of 256.
//! Two traps: the convolutions are indexed in **reverse** relative to the
//! level they serve (`convs[3 - level]`), and only level 2 receives the
//! top-down addition, so the 128² and 64² maps are pure laterals. The 16²
//! level is computed, contributes to level 2, and is then discarded by the
//! image encoder's `scalp`.
//!
//! What the rest of the network consumes is therefore:
//!
//! | level | stride | size | role |
//! |---|---|---|---|
//! | 0 | 4 | 128² x 256 | high-resolution feature, projected to 32 channels |
//! | 1 | 8 | 64² x 256 | high-resolution feature, projected to 64 channels |
//! | 2 | 16 | 32² x 256 | the image embedding everything else works on |

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config;
use super::ops;

/// The sine positional encoding SAM 2 uses for image features and memories.
///
/// Deterministic in `(h, w, channels)`, so callers build it once. Three
/// details are worth spelling out, because each of them is a plausible-looking
/// mistake: the grid is **1-indexed**, the normalization divides by the last
/// index **plus 1e-6** rather than by the extent, and the **y** half comes
/// before the x half.
pub fn sine_pos_embed<B: Backend>(
    h: usize,
    w: usize,
    channels: usize,
    dev: &B::Device,
) -> Tensor<B, 4> {
    assert!(channels % 4 == 0, "channels must be a multiple of four");
    let half = channels / 2;
    let scale = 2.0 * std::f64::consts::PI;
    let dim_t: Vec<f64> = (0..half)
        .map(|i| {
            f64::from(config::PE_TEMPERATURE).powf(2.0 * ((i / 2) as f64) / half as f64)
        })
        .collect();

    let mut data = vec![0f32; channels * h * w];
    let plane = h * w;
    for y in 0..h {
        let ye = (y as f64 + 1.0) / (h as f64 + 1e-6) * scale;
        for x in 0..w {
            let xe = (x as f64 + 1.0) / (w as f64 + 1e-6) * scale;
            let at = y * w + x;
            for j in 0..half / 2 {
                let (lo, hi) = (2 * j, 2 * j + 1);
                data[lo * plane + at] = (ye / dim_t[lo]).sin() as f32;
                data[hi * plane + at] = (ye / dim_t[hi]).cos() as f32;
                data[(half + lo) * plane + at] = (xe / dim_t[lo]).sin() as f32;
                data[(half + hi) * plane + at] = (xe / dim_t[hi]).cos() as f32;
            }
        }
    }
    ops::from_slice(&data, [1, channels, h, w], dev)
}

/// The four lateral convolutions.
pub struct Neck<B: Backend> {
    convs: Vec<(Tensor<B, 4>, Tensor<B, 1>)>,
}

impl<B: Backend> Neck<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<Neck<B>> {
        let mut convs = Vec::with_capacity(config::FPN_LEVELS);
        for (i, ch) in config::BACKBONE_CHANNELS.iter().enumerate() {
            let (w, b) = p.conv2d(
                &format!("image_encoder.neck.convs.{i}.conv"),
                config::D_MODEL,
                *ch,
                1,
                1,
            )?;
            convs.push((
                ops::from_slice(w, [config::D_MODEL, *ch, 1, 1], dev),
                ops::from_slice(b, [config::D_MODEL], dev),
            ));
        }
        Ok(Neck { convs })
    }

    /// `xs` are the trunk's stage outputs, highest resolution first. The
    /// result is the same length and the same order.
    pub fn forward(&self, xs: &[Tensor<B, 4>]) -> Vec<Tensor<B, 4>> {
        let n = config::FPN_LEVELS - 1;
        let mut out: Vec<Option<Tensor<B, 4>>> = vec![None; config::FPN_LEVELS];
        let mut prev: Option<Tensor<B, 4>> = None;
        for i in (0..=n).rev() {
            let (w, b) = &self.convs[n - i];
            let lateral = ops::conv2d(xs[i].clone(), w, Some(b), 1, 0, 1);
            let current = match (&prev, config::FPN_TOP_DOWN_LEVELS.contains(&i)) {
                (Some(p), true) => lateral + ops::upsample_nearest_2x(p.clone()),
                _ => lateral,
            };
            prev = Some(current.clone());
            out[i] = Some(current);
        }
        out.into_iter().map(|o| o.expect("every level set")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::load_safetensors;
    use std::path::Path;

    type Bk = burn::backend::NdArray;

    #[test]
    fn the_sine_encoding_matches_the_reference() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        let f = load_safetensors(Path::new("tests/data/medsam2-ops.safetensors")).unwrap();
        let want = f.get("pe_sine.y").expect("fixture");
        // `PositionEmbeddingSine(num_pos_feats=8)` emits eight channels in
        // total — four for y and four for x — over a 3 x 4 grid.
        assert_eq!(want.shape, vec![1, 8, 3, 4]);
        let got = ops::to_vec(sine_pos_embed::<Bk>(3, 4, 8, &dev));
        assert_eq!(got.len(), want.data.len());
        for (i, (a, b)) in got.iter().zip(want.data.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "channel-major index {i}: {a} vs {b}");
        }
    }

    #[test]
    fn the_two_halves_are_y_then_x() {
        let dev: burn::tensor::Device<Bk> = Default::default();
        // On a 1 x N strip every row index is the same, so the y half is
        // constant along the row and the x half is not.
        let pe = ops::to_vec(sine_pos_embed::<Bk>(1, 4, 8, &dev));
        let plane = 4;
        for c in 0..4 {
            let row = &pe[c * plane..(c + 1) * plane];
            assert!(
                row.iter().all(|v| (v - row[0]).abs() < 1e-9),
                "y channel {c} varies along x: {row:?}"
            );
        }
        let x_channel = &pe[4 * plane..5 * plane];
        assert!(x_channel.iter().any(|v| (v - x_channel[0]).abs() > 1e-6));
    }
}
