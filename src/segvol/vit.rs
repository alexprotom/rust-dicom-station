//! The image encoder: MONAI's 3-D `ViT`.
//!
//! Not SAM's `ImageEncoderViT` and not SwinUNETR — both are imported by the
//! reference builder and neither is ever instantiated. What runs is a plain
//! pre-norm vision transformer, twelve blocks wide 768, with **global**
//! attention over all 2048 tokens: no windowing, no shifting, no relative
//! position bias, and no class token.
//!
//! Two details differ from most ViT implementations and both are load-bearing:
//!
//! * the patch embedding is a `Linear` over flattened `(4,16,16)` patches,
//!   stored `[768, 1024]`. It is numerically a strided `Conv3d`, but the
//!   weight layout is a matrix, so it is applied as one;
//! * the fused qkv projection has **no bias** (MONAI's `SABlock` hardcodes
//!   `bias=False`), while the output projection does. Adding a qkv bias for
//!   symmetry silently changes every activation downstream.

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::nn::attention::{attention, Mask};
use crate::nn::linalg::{gelu, layer_norm, linear, LAYER_NORM_EPS};
use crate::nn::tensor::Mat;

use super::config::*;
use crate::nn::params::Params;

/// One pre-norm transformer block.
struct Block {
    norm1_w: Vec<f32>,
    norm1_b: Vec<f32>,
    qkv_w: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
    norm2_w: Vec<f32>,
    norm2_b: Vec<f32>,
    lin1_w: Vec<f32>,
    lin1_b: Vec<f32>,
    lin2_w: Vec<f32>,
    lin2_b: Vec<f32>,
}

/// The assembled image encoder.
pub struct Vit {
    patch_w: Vec<f32>,
    patch_b: Vec<f32>,
    /// Learned absolute position embedding, `[TOKENS, EMBED]`.
    pos: Mat,
    blocks: Vec<Block>,
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
}

impl Vit {
    pub fn build(p: &Params) -> Result<Vit> {
        let pe = "image_encoder.patch_embedding";
        let (patch_w, patch_b) = p
            .linear_opt(&format!("{pe}.patch_embeddings.1"), EMBED, PATCH_FEATURES)
            .context("patch embedding")?;
        let pos = p.get(&format!("{pe}.position_embeddings"), &[1, TOKENS, EMBED])?;
        let mut blocks = Vec::with_capacity(VIT_BLOCKS);
        for i in 0..VIT_BLOCKS {
            let b = format!("image_encoder.blocks.{i}");
            let (norm1_w, norm1_b) = p.norm(&format!("{b}.norm1"), EMBED)?;
            let (norm2_w, norm2_b) = p.norm(&format!("{b}.norm2"), EMBED)?;
            // Fused qkv: [3 * EMBED, EMBED], deliberately without a bias.
            let (qkv_w, qkv_b) = p.linear_opt(&format!("{b}.attn.qkv"), 3 * EMBED, EMBED)?;
            if qkv_b.is_some() {
                anyhow::bail!(
                    "{b}.attn.qkv has a bias; MONAI's SABlock builds it with bias=False, \
                     so this checkpoint is not the network this port implements"
                );
            }
            let (out_w, out_b) = p.linear_opt(&format!("{b}.attn.out_proj"), EMBED, EMBED)?;
            let (lin1_w, lin1_b) = p.linear_opt(&format!("{b}.mlp.linear1"), VIT_MLP, EMBED)?;
            let (lin2_w, lin2_b) = p.linear_opt(&format!("{b}.mlp.linear2"), EMBED, VIT_MLP)?;
            blocks.push(Block {
                norm1_w: norm1_w.to_vec(),
                norm1_b: norm1_b.to_vec(),
                qkv_w: qkv_w.to_vec(),
                out_w: out_w.to_vec(),
                out_b: out_b.context("attn.out_proj needs a bias")?.to_vec(),
                norm2_w: norm2_w.to_vec(),
                norm2_b: norm2_b.to_vec(),
                lin1_w: lin1_w.to_vec(),
                lin1_b: lin1_b.context("mlp.linear1 needs a bias")?.to_vec(),
                lin2_w: lin2_w.to_vec(),
                lin2_b: lin2_b.context("mlp.linear2 needs a bias")?.to_vec(),
            });
        }
        let (norm_w, norm_b) = p.norm("image_encoder.norm", EMBED)?;
        Ok(Vit {
            patch_w: patch_w.to_vec(),
            patch_b: patch_b.context("patch embedding needs a bias")?.to_vec(),
            pos: Mat::from_vec(TOKENS, EMBED, pos.to_vec()),
            blocks,
            norm_w: norm_w.to_vec(),
            norm_b: norm_b.to_vec(),
        })
    }

    /// Cut a `ROI`-shaped volume into the token matrix `[TOKENS,
    /// PATCH_FEATURES]`.
    ///
    /// Token order is C-order over the patch grid, and within a patch the
    /// values run C-order over `PATCH` — the layout MONAI's
    /// `Rearrange("b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)")`
    /// produces for a single-channel input.
    pub fn patchify(volume: &[f32]) -> Mat {
        assert_eq!(
            volume.len(),
            ROI[0] * ROI[1] * ROI[2],
            "the image encoder only accepts a {ROI:?} volume"
        );
        let mut out = Mat::zeros(TOKENS, PATCH_FEATURES);
        let [_, g1, g2] = GRID;
        let [p0, p1, p2] = PATCH;
        out.data
            .par_chunks_mut(PATCH_FEATURES)
            .enumerate()
            .for_each(|(t, dst)| {
                let b0 = t / (g1 * g2);
                let b1 = (t / g2) % g1;
                let b2 = t % g2;
                for i0 in 0..p0 {
                    let a0 = b0 * p0 + i0;
                    for i1 in 0..p1 {
                        let a1 = b1 * p1 + i1;
                        let src = (a0 * ROI[1] + a1) * ROI[2] + b2 * p2;
                        let d = (i0 * p1 + i1) * p2;
                        dst[d..d + p2].copy_from_slice(&volume[src..src + p2]);
                    }
                }
            });
        out
    }

    /// Encode one `ROI`-shaped volume into `[TOKENS, EMBED]`.
    pub fn forward(&self, volume: &[f32]) -> Mat {
        let patches = Self::patchify(volume);
        let mut x = linear(&patches, &self.patch_w, EMBED, Some(&self.patch_b));
        x.add_assign(&self.pos);
        for b in &self.blocks {
            self.block(b, &mut x);
        }
        layer_norm(
            &mut x.data,
            EMBED,
            &self.norm_w,
            &self.norm_b,
            LAYER_NORM_EPS,
        );
        x
    }

    fn block(&self, b: &Block, x: &mut Mat) {
        // x = x + attn(norm1(x))
        let mut h = x.clone();
        layer_norm(&mut h.data, EMBED, &b.norm1_w, &b.norm1_b, LAYER_NORM_EPS);
        let qkv = linear(&h, &b.qkv_w, 3 * EMBED, None);
        // MONAI packs the 2304 columns as (qkv, head, head_dim) in C order,
        // so q, k and v are simply the three contiguous thirds.
        let q = split_cols(&qkv, 0, EMBED);
        let k = split_cols(&qkv, EMBED, EMBED);
        let v = split_cols(&qkv, 2 * EMBED, EMBED);
        let a = attention(&q, &k, &v, VIT_HEADS, Mask::None);
        let a = linear(&a, &b.out_w, EMBED, Some(&b.out_b));
        x.add_assign(&a);

        // x = x + mlp(norm2(x))
        let mut h = x.clone();
        layer_norm(&mut h.data, EMBED, &b.norm2_w, &b.norm2_b, LAYER_NORM_EPS);
        let mut h = linear(&h, &b.lin1_w, VIT_MLP, Some(&b.lin1_b));
        gelu(&mut h.data);
        let h = linear(&h, &b.lin2_w, EMBED, Some(&b.lin2_b));
        x.add_assign(&h);
    }
}

/// Columns `[start, start + len)` of `m` as a new matrix.
pub(super) fn split_cols(m: &Mat, start: usize, len: usize) -> Mat {
    let mut out = Mat::zeros(m.rows, len);
    out.data
        .par_chunks_mut(len)
        .enumerate()
        .for_each(|(r, dst)| {
            dst.copy_from_slice(&m.data[r * m.cols + start..r * m.cols + start + len])
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patchify_places_every_voxel_exactly_once() {
        // A volume whose value is its own flat index: every value must appear
        // exactly once in the token matrix, so the gather is a permutation.
        let n = ROI[0] * ROI[1] * ROI[2];
        let vol: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let m = Vit::patchify(&vol);
        assert_eq!((m.rows, m.cols), (TOKENS, PATCH_FEATURES));
        let mut seen = vec![false; n];
        for v in &m.data {
            let i = *v as usize;
            assert!(!seen[i], "value {i} appeared twice");
            seen[i] = true;
        }
        assert!(seen.into_iter().all(|s| s));
    }

    #[test]
    fn patchify_uses_c_order_within_and_across_patches() {
        let n = ROI[0] * ROI[1] * ROI[2];
        let vol: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let m = Vit::patchify(&vol);
        let at = |a0: usize, a1: usize, a2: usize| ((a0 * ROI[1] + a1) * ROI[2] + a2) as f32;
        // token 0 is the origin patch
        assert_eq!(m.data[0], at(0, 0, 0));
        assert_eq!(m.data[1], at(0, 0, 1)); // fastest axis inside the patch
        assert_eq!(m.data[PATCH[2]], at(0, 1, 0));
        assert_eq!(m.data[PATCH[1] * PATCH[2]], at(1, 0, 0));
        // token 1 is the next block along axis 2
        assert_eq!(m.data[PATCH_FEATURES], at(0, 0, PATCH[2]));
        // token GRID[2] is the next block along axis 1
        assert_eq!(m.data[GRID[2] * PATCH_FEATURES], at(0, PATCH[1], 0));
        // token GRID[1]*GRID[2] is the next block along axis 0
        assert_eq!(
            m.data[GRID[1] * GRID[2] * PATCH_FEATURES],
            at(PATCH[0], 0, 0)
        );
    }

    #[test]
    #[should_panic(expected = "only accepts")]
    fn patchify_rejects_a_wrong_shape() {
        Vit::patchify(&[0.0; 10]);
    }

    #[test]
    fn split_cols_takes_the_right_thirds() {
        let m = Mat::from_vec(2, 6, vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]);
        assert_eq!(split_cols(&m, 0, 2).data, vec![0., 1., 6., 7.]);
        assert_eq!(split_cols(&m, 2, 2).data, vec![2., 3., 8., 9.]);
        assert_eq!(split_cols(&m, 4, 2).data, vec![4., 5., 10., 11.]);
    }
}
