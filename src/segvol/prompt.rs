//! The prompt encoder: SAM's, made three-dimensional.
//!
//! A prompt becomes a small set of 768-wide *sparse* tokens (one per point,
//! two per box, one for text) plus a *dense* embedding laid over the whole
//! token grid. Positions are encoded with random Fourier features drawn from
//! a `[3, 384]` matrix that is a **registered buffer**, not a parameter: it
//! is in the checkpoint and must be loaded. Regenerating it randomly produces
//! a network that runs, reports confident IoU, and segments the wrong place.
//!
//! # The axis convention, and why it looks wrong
//!
//! SAM is two-dimensional and takes points as `(x, y)` — column first — while
//! `image_size` is `(h, w)`. Its normalization is therefore
//! `coords[0] /= image_size[1]; coords[1] /= image_size[0]`, which is correct
//! for that ordering.
//!
//! SegVol inherits those lines verbatim, adds a third, and then feeds them
//! coordinates in **array-axis order**: prompts are built with
//! `torch.nonzero`, so a point is `(axis0, axis1, axis2)` and a box is
//! `(axis0_min, axis1_min, axis2_min, axis0_max, axis1_max, axis2_max)`.
//! With `image_size = [32, 256, 256]` the result is that axis 0 (range 0..32)
//! is divided by 256 and axis 1 (range 0..256) is divided by 32 — so the
//! first coordinate lands in `[0, 0.125)` and the second in `[0, 8)`.
//!
//! The dense encoding, meanwhile, normalizes each axis by its own length and
//! stacks the channels as `(axis1, axis0, axis2)`. Sparse and dense prompts
//! therefore disagree about both scale and channel order.
//!
//! This is reproduced exactly, because the weights were trained through it.
//! "Fixing" it moves every prompt.

use anyhow::{Context, Result};

use crate::nn::linalg::linear;
use crate::nn::tensor::{Act, Mat};

use super::config::*;
use crate::nn::params::Params;

/// Number of Fourier feature pairs; the encoding is `[sin; cos]`, so the
/// output width is twice this.
const POS_FEATS: usize = EMBED / 2;

/// A point prompt in the resized `ROI` grid, in array-axis order, with a
/// label: `1` foreground, `0` background, `-1` padding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub coord: [f32; 3],
    pub label: i8,
}

impl Point {
    pub fn foreground(coord: [f32; 3]) -> Point {
        Point { coord, label: 1 }
    }
    pub fn background(coord: [f32; 3]) -> Point {
        Point { coord, label: 0 }
    }
}

/// A box prompt in the resized `ROI` grid: `[min0, min1, min2, max0, max1,
/// max2]`, in array-axis order.
pub type BBox = [f32; 6];

/// What the prompt encoder produces for one forward pass.
pub struct Prompts {
    /// One row per sparse token, `[n, EMBED]`. May have zero rows.
    pub sparse: Mat,
    /// `[EMBED, GRID]` — the no-mask embedding broadcast over the grid.
    pub dense: Act,
}

pub struct PromptEncoder {
    /// The Fourier matrix, transposed to `[POS_FEATS, 3]` so it can go
    /// through the same `x @ wᵀ` path as every other projection.
    gauss_t: Vec<f32>,
    /// Background, foreground, box-min corner, box-max corner.
    point_embeddings: [Vec<f32>; 4],
    not_a_point: Vec<f32>,
    no_mask: Vec<f32>,
}

impl PromptEncoder {
    pub fn build(p: &Params) -> Result<PromptEncoder> {
        let g = p
            .get(
                "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
                &[3, POS_FEATS],
            )
            .context("the Fourier buffer is a registered buffer and must be in the checkpoint")?;
        let mut gauss_t = vec![0f32; 3 * POS_FEATS];
        for f in 0..POS_FEATS {
            for c in 0..3 {
                gauss_t[f * 3 + c] = g[c * POS_FEATS + f];
            }
        }
        let mut point_embeddings: [Vec<f32>; 4] = Default::default();
        for (i, e) in point_embeddings.iter_mut().enumerate() {
            *e = p
                .get(
                    &format!("prompt_encoder.point_embeddings.{i}.weight"),
                    &[1, EMBED],
                )?
                .to_vec();
        }
        Ok(PromptEncoder {
            gauss_t,
            point_embeddings,
            not_a_point: p
                .get("prompt_encoder.not_a_point_embed.weight", &[1, EMBED])?
                .to_vec(),
            no_mask: p
                .get("prompt_encoder.no_mask_embed.weight", &[1, EMBED])?
                .to_vec(),
        })
    }

    /// Random-Fourier encoding of coordinates already normalized to `[0, 1]`:
    /// `[sin(2π (2c-1) G); cos(2π (2c-1) G)]`.
    fn pe_encoding(&self, coords: &[[f32; 3]]) -> Mat {
        let x = Mat::from_vec(
            coords.len(),
            3,
            coords
                .iter()
                .flat_map(|c| c.iter().map(|v| 2.0 * v - 1.0))
                .collect(),
        );
        let proj = linear(&x, &self.gauss_t, POS_FEATS, None);
        let mut out = Mat::zeros(coords.len(), EMBED);
        for r in 0..coords.len() {
            for f in 0..POS_FEATS {
                let a = std::f32::consts::TAU * proj.data[r * POS_FEATS + f];
                out.data[r * EMBED + f] = a.sin();
                out.data[r * EMBED + POS_FEATS + f] = a.cos();
            }
        }
        out
    }

    /// The dense positional encoding of the token grid, `[TOKENS, EMBED]`.
    ///
    /// Each axis is normalized by its own length, and the three channels are
    /// stacked `(axis1, axis0, axis2)` — the order the reference's
    /// `torch.stack([x_embed, y_embed, z_embed])` produces.
    pub fn dense_pe(&self) -> Mat {
        let [g0, g1, g2] = GRID;
        let mut coords = Vec::with_capacity(TOKENS);
        for i0 in 0..g0 {
            for i1 in 0..g1 {
                for i2 in 0..g2 {
                    // cumsum over a grid of ones, minus 0.5, over the axis length
                    let a0 = (i0 as f32 + 0.5) / g0 as f32;
                    let a1 = (i1 as f32 + 0.5) / g1 as f32;
                    let a2 = (i2 as f32 + 0.5) / g2 as f32;
                    coords.push([a1, a0, a2]);
                }
            }
        }
        self.pe_encoding(&coords)
    }

    /// Encode sparse coordinates the way `forward_with_coords` does —
    /// including the inherited axis-swapped normalization documented above.
    fn encode_coords(&self, coords: &[[f32; 3]]) -> Mat {
        let scaled: Vec<[f32; 3]> = coords
            .iter()
            .map(|c| {
                [
                    c[0] / ROI[1] as f32,
                    c[1] / ROI[0] as f32,
                    c[2] / ROI[2] as f32,
                ]
            })
            .collect();
        self.pe_encoding(&scaled)
    }

    /// Assemble the sparse and dense embeddings for one prompt.
    ///
    /// `text` is the already-aligned 768-wide embedding, appended last. A
    /// padding point is added only when no box is given, matching the
    /// reference's `pad=(boxes is None)`.
    pub fn encode(&self, points: &[Point], boxes: &[BBox], text: Option<&[f32]>) -> Prompts {
        let mut sparse = Mat::zeros(0, EMBED);

        if !points.is_empty() || (boxes.is_empty() && !points.is_empty()) {
            sparse = sparse.vcat(&self.embed_points(points, boxes.is_empty()));
        }
        if !boxes.is_empty() {
            sparse = sparse.vcat(&self.embed_boxes(boxes));
        }
        if let Some(t) = text {
            assert_eq!(t.len(), EMBED, "the text embedding must be {EMBED} wide");
            sparse = sparse.vcat(&Mat::row_vec(t));
        }

        // No mask prompt is ever supplied, so the dense embedding is the
        // learned no-mask vector broadcast over every grid position.
        let mut dense = Act::zeros(EMBED, GRID[0], GRID[1], GRID[2]);
        let sp = dense.spatial();
        for c in 0..EMBED {
            dense.data[c * sp..(c + 1) * sp].fill(self.no_mask[c]);
        }
        Prompts { sparse, dense }
    }

    fn embed_points(&self, points: &[Point], pad: bool) -> Mat {
        let mut coords: Vec<[f32; 3]> = points
            .iter()
            .map(|p| [p.coord[0] + 0.5, p.coord[1] + 0.5, p.coord[2] + 0.5])
            .collect();
        let mut labels: Vec<i8> = points.iter().map(|p| p.label).collect();
        if pad {
            coords.push([0.5, 0.5, 0.5]); // the padding point is (0,0,0) + 0.5
            labels.push(-1);
        }
        let mut emb = self.encode_coords(&coords);
        for (i, l) in labels.iter().enumerate() {
            let row = emb.row_mut(i);
            match l {
                -1 => {
                    // zeroed first, then marked as "not a point"
                    row.fill(0.0);
                    for (v, e) in row.iter_mut().zip(self.not_a_point.iter()) {
                        *v += e;
                    }
                }
                0 => {
                    for (v, e) in row.iter_mut().zip(self.point_embeddings[0].iter()) {
                        *v += e;
                    }
                }
                _ => {
                    for (v, e) in row.iter_mut().zip(self.point_embeddings[1].iter()) {
                        *v += e;
                    }
                }
            }
        }
        emb
    }

    fn embed_boxes(&self, boxes: &[BBox]) -> Mat {
        // Each box becomes two corner points, min then max.
        let coords: Vec<[f32; 3]> = boxes
            .iter()
            .flat_map(|b| {
                [
                    [b[0] + 0.5, b[1] + 0.5, b[2] + 0.5],
                    [b[3] + 0.5, b[4] + 0.5, b[5] + 0.5],
                ]
            })
            .collect();
        let mut emb = self.encode_coords(&coords);
        for i in 0..emb.rows {
            let corner = i % 2; // 0 = min, 1 = max
            let e = &self.point_embeddings[2 + corner];
            for (v, ev) in emb.row_mut(i).iter_mut().zip(e.iter()) {
                *v += ev;
            }
        }
        emb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::WTensor;
    use std::collections::HashMap;

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    }

    fn encoder() -> PromptEncoder {
        let mut s = 4242u64;
        let mut m = HashMap::new();
        let mut put = |k: &str, shape: Vec<usize>, s: &mut u64| {
            let n: usize = shape.iter().product();
            m.insert(
                k.to_string(),
                WTensor {
                    shape,
                    data: (0..n).map(|_| rnd(s)).collect(),
                },
            );
        };
        put(
            "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
            vec![3, POS_FEATS],
            &mut s,
        );
        for i in 0..4 {
            put(
                &format!("prompt_encoder.point_embeddings.{i}.weight"),
                vec![1, EMBED],
                &mut s,
            );
        }
        put(
            "prompt_encoder.not_a_point_embed.weight",
            vec![1, EMBED],
            &mut s,
        );
        put(
            "prompt_encoder.no_mask_embed.weight",
            vec![1, EMBED],
            &mut s,
        );
        PromptEncoder::build(&Params::new(m)).unwrap()
    }

    #[test]
    fn the_fourier_buffer_must_be_present() {
        let e = PromptEncoder::build(&Params::new(HashMap::new()))
            .map(|_| ())
            .unwrap_err()
            .to_string();
        assert!(e.contains("registered buffer"), "{e}");
    }

    #[test]
    fn the_encoding_is_sin_and_cos_halves_and_stays_bounded() {
        let enc = encoder();
        let out = enc.pe_encoding(&[[0.0, 0.5, 1.0], [0.25, 0.25, 0.25]]);
        assert_eq!((out.rows, out.cols), (2, EMBED));
        for v in &out.data {
            assert!(v.abs() <= 1.0 + 1e-6, "{v} is not a sine or cosine");
        }
        // sin^2 + cos^2 = 1 pairwise across the two halves
        for r in 0..2 {
            for f in 0..POS_FEATS {
                let s = out.data[r * EMBED + f];
                let c = out.data[r * EMBED + POS_FEATS + f];
                assert!((s * s + c * c - 1.0).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn sparse_normalization_reproduces_the_inherited_axis_swap() {
        // This is trap #1. A point at (a, b, c) must be encoded exactly as
        // (a/ROI[1], b/ROI[0], c/ROI[2]) — axis 0 over 256, axis 1 over 32.
        // The check is behavioural: two coordinates that the swapped rule
        // maps to the same normalized value must encode identically, and a
        // "corrected" rule would separate them.
        let enc = encoder();
        // (8, 0, 0) -> (8/256, 0, 0);  (0, 1, 0) -> (0, 1/32, 0) = (0, 0.03125, 0)
        // 8/256 = 0.03125 as well, so these two land on the same magnitude in
        // different channels — only possible under the swapped divisors.
        let a = enc.encode_coords(&[[8.0, 0.0, 0.0]]);
        let b = enc.encode_coords(&[[0.0, 1.0, 0.0]]);
        assert!((8.0 / ROI[1] as f32 - 1.0 / ROI[0] as f32).abs() < 1e-9);
        // and directly: encoding (a,b,c) equals encoding the pre-divided form
        let direct = enc.pe_encoding(&[[
            5.0 / ROI[1] as f32,
            7.0 / ROI[0] as f32,
            9.0 / ROI[2] as f32,
        ]]);
        let viaswap = enc.encode_coords(&[[5.0, 7.0, 9.0]]);
        for (x, y) in direct.data.iter().zip(viaswap.data.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
        assert_ne!(a.data, b.data); // different channels, so still distinct
    }

    #[test]
    fn dense_pe_normalizes_each_axis_by_its_own_length() {
        let enc = encoder();
        let dense = enc.dense_pe();
        assert_eq!((dense.rows, dense.cols), (TOKENS, EMBED));
        // token (i0,i1,i2) must equal the direct encoding of the stacked
        // (axis1, axis0, axis2) triple
        let [g0, g1, g2] = GRID;
        for (i0, i1, i2) in [(0, 0, 0), (3, 5, 7), (g0 - 1, g1 - 1, g2 - 1)] {
            let t = (i0 * g1 + i1) * g2 + i2;
            let want = enc.pe_encoding(&[[
                (i1 as f32 + 0.5) / g1 as f32,
                (i0 as f32 + 0.5) / g0 as f32,
                (i2 as f32 + 0.5) / g2 as f32,
            ]]);
            for c in 0..EMBED {
                assert!(
                    (dense.data[t * EMBED + c] - want.data[c]).abs() < 1e-6,
                    "token {t} channel {c}"
                );
            }
        }
    }

    #[test]
    fn a_box_becomes_two_corner_tokens_with_distinct_embeddings() {
        let enc = encoder();
        let p = enc.encode(&[], &[[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]], None);
        assert_eq!(p.sparse.rows, 2, "a box is two corner tokens");
        // the two corners carry different learned embeddings
        assert_ne!(p.sparse.row(0), p.sparse.row(1));
        // two boxes give four tokens, in min/max order
        let p2 = enc.encode(
            &[],
            &[
                [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            None,
        );
        assert_eq!(p2.sparse.rows, 4);
        assert_eq!(p2.sparse.row(0), p.sparse.row(0));
        assert_eq!(p2.sparse.row(1), p.sparse.row(1));
    }

    #[test]
    fn a_padding_point_is_added_only_when_there_is_no_box() {
        let enc = encoder();
        // points alone: one padding point is appended
        let a = enc.encode(&[Point::foreground([1.0, 2.0, 3.0])], &[], None);
        assert_eq!(a.sparse.rows, 2);
        // points with a box: no padding point, plus the two corners
        let b = enc.encode(
            &[Point::foreground([1.0, 2.0, 3.0])],
            &[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
            None,
        );
        assert_eq!(b.sparse.rows, 3);
        // the point token itself is the same either way
        assert_eq!(a.sparse.row(0), b.sparse.row(0));
    }

    #[test]
    fn a_padding_point_is_the_not_a_point_embedding_alone() {
        let enc = encoder();
        let a = enc.encode(&[Point::foreground([1.0, 2.0, 3.0])], &[], None);
        // row 1 is the pad: its positional part is zeroed, leaving exactly
        // the learned not-a-point vector
        for (v, e) in a.sparse.row(1).iter().zip(enc.not_a_point.iter()) {
            assert!((v - e).abs() < 1e-6);
        }
    }

    #[test]
    fn foreground_and_background_points_differ_only_by_their_embedding() {
        let enc = encoder();
        let f = enc.encode(&[Point::foreground([4.0, 5.0, 6.0])], &[], None);
        let b = enc.encode(&[Point::background([4.0, 5.0, 6.0])], &[], None);
        for c in 0..EMBED {
            let d = f.sparse.row(0)[c] - b.sparse.row(0)[c];
            let want = enc.point_embeddings[1][c] - enc.point_embeddings[0][c];
            assert!((d - want).abs() < 1e-5, "channel {c}");
        }
    }

    #[test]
    fn text_is_appended_last_and_dense_is_the_no_mask_vector() {
        let enc = encoder();
        let text: Vec<f32> = (0..EMBED).map(|i| i as f32 * 0.001).collect();
        let p = enc.encode(&[], &[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], Some(&text));
        assert_eq!(p.sparse.rows, 3, "two corners then the text token");
        assert_eq!(p.sparse.row(2), &text[..]);
        // dense is the no-mask embedding broadcast over every grid position
        assert_eq!(
            (p.dense.c, p.dense.d, p.dense.h, p.dense.w),
            (EMBED, GRID[0], GRID[1], GRID[2])
        );
        let sp = p.dense.spatial();
        for c in 0..EMBED {
            for v in &p.dense.data[c * sp..(c + 1) * sp] {
                assert_eq!(*v, enc.no_mask[c]);
            }
        }
    }

    #[test]
    fn an_empty_prompt_produces_no_sparse_tokens() {
        let enc = encoder();
        let p = enc.encode(&[], &[], None);
        assert_eq!(p.sparse.rows, 0);
        assert_eq!(p.dense.c, EMBED);
    }
}
