//! Efficient MedSAM2's image encoder: EfficientTAM's plain ViT (ViTDet)
//! and its one-level neck.
//!
//! [EfficientTAM](https://github.com/yformer/EfficientTAM) (Xiong et al.,
//! *Efficient Track Anything*, 2024) is SAM 2 with the hierarchical Hiera
//! trunk replaced by a plain, non-hierarchical ViT - ViT-Tiny (192 wide,
//! three heads) or ViT-Small (384, six) on 16 x 16 patches - and with the
//! decoder's high-resolution features dropped, since a plain ViT has only
//! one scale. Everything after the image encoder is SAM 2's. MedSAM2's
//! authors fine-tuned both sizes for the FLARE 2025 RECIST task
//! (`eff_medsam2_{tiny,small}_FLARE25_RECIST_baseline.pt`), to run on a
//! CPU.
//!
//! The trunk (`efficient_track_anything/modeling/backbones/vitdet.py`):
//!
//! 1. a 16 x 16, stride-16 patch embedding: 512 pixels become a 32 x 32
//!    token grid;
//! 2. an absolute position embedding learned at 224 pixels with a class
//!    token (`[1, 1 + 14 * 14, dim]`): the class row is dropped and the
//!    14 x 14 grid bicubically resized to the token grid (PyTorch's
//!    `interpolate`, `align_corners=False`) - once, at load time;
//! 3. twelve pre-norm blocks (LayerNorm with eps 1e-6, multi-head attention
//!    without relative positions, a GELU MLP four times as wide), blocks
//!    2, 5, 8 and 11 attending globally and the others inside 14 x 14
//!    windows. The 32-token grid does not divide by 14: it is padded to 42
//!    **after** the norm with zeros, and the padded tokens take part in the
//!    attention as keys - there is no mask - before being cropped away;
//! 4. the output of the last global block.
//!
//! The neck (`ViTDetNeck` with `neck_norm: LN`): a bias-free 1 x 1
//! convolution to 256, `LayerNorm2d`, a bias-free 3 x 3 convolution,
//! `LayerNorm2d`. Its sine position encoding is the one SAM 2's neck
//! computes for the same grid, which [`super::model`] already holds.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config::D_MODEL;
use super::layers::{GeluMlp, Lin, Norm};
use super::ops;

/// The geometry of one EfficientTAM ViT.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VitSpec {
    pub dim: usize,
    pub heads: usize,
    pub depth: usize,
    pub patch: usize,
    /// Window of the windowed blocks.
    pub window: usize,
    /// Blocks with global attention; the last one's output is the trunk's.
    pub global: Vec<usize>,
    /// Edge of the grid the position embedding was learned at
    /// (`pretrain_img_size / patch_size`), with a class token in front.
    pub pretrain_grid: usize,
}

impl VitSpec {
    /// The published configurations: `efficienttam_ti_512x512.yaml` and
    /// `efficienttam_s_512x512.yaml` differ only in width and heads.
    pub fn published(dim: usize) -> Result<VitSpec> {
        let heads = match dim {
            192 => 3,
            384 => 6,
            _ => bail!("no EfficientTAM ViT is {dim} wide"),
        };
        Ok(VitSpec {
            dim,
            heads,
            depth: 12,
            patch: 16,
            window: 14,
            global: vec![2, 5, 8, 11],
            pretrain_grid: 14,
        })
    }

    /// The width a checkpoint's patch embedding says it has, if it is an
    /// EfficientTAM checkpoint at all.
    pub fn width_of(p: &Params) -> Option<usize> {
        let s = p.shape("image_encoder.trunk.patch_embed.proj.weight")?;
        (s.len() == 4 && s[1] == 3 && p.shape("image_encoder.trunk.pos_embed_window").is_none())
            .then_some(s[0])
    }
}

struct VitBlock<B: Backend> {
    norm1: Norm<B>,
    qkv: Lin<B>,
    proj: Lin<B>,
    norm2: Norm<B>,
    mlp: GeluMlp<B>,
    heads: usize,
    window: usize,
}

impl<B: Backend> VitBlock<B> {
    fn load(p: &Params, i: usize, spec: &VitSpec, dev: &B::Device) -> Result<VitBlock<B>> {
        let base = format!("image_encoder.trunk.blocks.{i}");
        let d = spec.dim;
        Ok(VitBlock {
            norm1: Norm::load6(p, &format!("{base}.norm1"), d, dev)?,
            qkv: Lin::load(p, &format!("{base}.attn.qkv"), 3 * d, d, dev)?,
            proj: Lin::load(p, &format!("{base}.attn.proj"), d, d, dev)?,
            norm2: Norm::load6(p, &format!("{base}.norm2"), d, dev)?,
            mlp: GeluMlp::load(
                p,
                &format!("{base}.mlp.layers.0"),
                &format!("{base}.mlp.layers.1"),
                d,
                4 * d,
                dev,
            )?,
            heads: spec.heads,
            window: if spec.global.contains(&i) {
                0
            } else {
                spec.window
            },
        })
    }

    /// `Attention.forward` on `[b, h, w, c]` tokens.
    fn attention(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, w, c] = x.dims();
        let (heads, n) = (self.heads, h * w);
        let hd = c / heads;
        let qkv = self
            .qkv
            .apply(x.reshape([b, n, c]))
            .reshape([b, n, 3, heads, hd]);
        let take = |i: usize| {
            qkv.clone()
                .slice([0..b, 0..n, i..i + 1, 0..heads, 0..hd])
                .reshape([b, n, heads, hd])
                .swap_dims(1, 2)
        };
        let out = ops::sdpa(take(0), take(1), take(2));
        self.proj.apply(out.swap_dims(1, 2).reshape([b, h, w, c]))
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, h, w, _] = x.dims();
        let normed = self.norm1.apply(x.clone());
        let attended = if self.window > 0 {
            let (tiles, pad) = ops::window_partition(normed, self.window);
            ops::window_unpartition(self.attention(tiles), self.window, pad, [h, w])
        } else {
            self.attention(normed)
        };
        let x = x + attended;
        let mlp = self.mlp.apply(self.norm2.apply(x.clone()));
        x + mlp
    }
}

/// The image encoder, weights resident.
pub struct VitDet<B: Backend> {
    spec: VitSpec,
    patch_weight: Tensor<B, 4>,
    patch_bias: Tensor<B, 1>,
    /// The resized position embedding, `[1, grid, grid, dim]`, for the grid
    /// it was built for.
    pos_embed: Tensor<B, 4>,
    grid: usize,
    blocks: Vec<VitBlock<B>>,
    conv_1x1: Tensor<B, 4>,
    norm_0: Norm<B>,
    conv_3x3: Tensor<B, 4>,
    norm_1: Norm<B>,
}

impl<B: Backend> VitDet<B> {
    /// Load the published geometry the checkpoint's width names, for
    /// `image_size`-pixel slices.
    pub fn load(p: &Params, image_size: usize, dev: &B::Device) -> Result<VitDet<B>> {
        let Some(dim) = VitSpec::width_of(p) else {
            bail!("not an EfficientTAM checkpoint");
        };
        Self::load_spec(p, VitSpec::published(dim)?, image_size, dev)
    }

    pub fn load_spec(
        p: &Params,
        spec: VitSpec,
        image_size: usize,
        dev: &B::Device,
    ) -> Result<VitDet<B>> {
        let t = "image_encoder.trunk";
        let (d, k) = (spec.dim, spec.patch);
        let (pw, pb) = p.conv2d(&format!("{t}.patch_embed.proj"), d, 3, k, 1)?;
        let grid = image_size / k;
        let g0 = spec.pretrain_grid;
        let raw = p.get(&format!("{t}.pos_embed"), &[1, 1 + g0 * g0, d])?;
        // Drop the class token, `[g0 * g0, d]` -> `[1, d, g0, g0]`.
        let mut chw = vec![0f32; d * g0 * g0];
        for t_ in 0..g0 * g0 {
            for c in 0..d {
                chw[c * g0 * g0 + t_] = raw[(1 + t_) * d + c];
            }
        }
        let pos: Tensor<B, 4> = ops::from_slice(&chw, [1, d, g0, g0], dev);
        let pos = if g0 == grid {
            pos
        } else {
            ops::resize_bicubic(pos, [grid, grid])
        };
        let blocks = (0..spec.depth)
            .map(|i| VitBlock::load(p, i, &spec, dev))
            .collect::<Result<Vec<_>>>()?;
        let n = "image_encoder.neck.convs.0";
        Ok(VitDet {
            patch_weight: ops::from_slice(pw, [d, 3, k, k], dev),
            patch_bias: ops::from_slice(pb, [d], dev),
            pos_embed: pos.permute([0, 2, 3, 1]),
            grid,
            blocks,
            conv_1x1: ops::from_slice(
                p.get(&format!("{n}.conv_1x1.weight"), &[D_MODEL, d, 1, 1])?,
                [D_MODEL, d, 1, 1],
                dev,
            ),
            norm_0: Norm::load6(p, &format!("{n}.norm_0"), D_MODEL, dev)?,
            conv_3x3: ops::from_slice(
                p.get(&format!("{n}.conv_3x3.weight"), &[D_MODEL, D_MODEL, 3, 3])?,
                [D_MODEL, D_MODEL, 3, 3],
                dev,
            ),
            norm_1: Norm::load6(p, &format!("{n}.norm_1"), D_MODEL, dev)?,
            spec,
        })
    }

    pub fn spec(&self) -> &VitSpec {
        &self.spec
    }

    /// The trunk's output, `[n, dim, grid, grid]`.
    pub fn trunk(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = ops::conv2d(
            image,
            &self.patch_weight,
            Some(&self.patch_bias),
            self.spec.patch,
            0,
            1,
        );
        let [_, _, h, w] = x.dims();
        assert_eq!(
            [h, w],
            [self.grid, self.grid],
            "the slice is not the size the encoder was built for"
        );
        let mut x = x.permute([0, 2, 3, 1]) + self.pos_embed.clone();
        let last = *self.spec.global.last().expect("a global block");
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(x);
            if i == last {
                break;
            }
        }
        x.permute([0, 3, 1, 2])
    }

    /// The image embedding, `[n, 256, grid, grid]`.
    pub fn forward(&self, image: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.trunk(image);
        let x = self
            .norm_0
            .apply_2d(ops::conv2d(x, &self.conv_1x1, None, 1, 0, 1));
        self.norm_1
            .apply_2d(ops::conv2d(x, &self.conv_3x3, None, 1, 1, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::{load_safetensors, WTensor};
    use std::collections::{BTreeMap, HashMap};

    type Bk = burn::backend::NdArray;

    /// The 64-bit LCG `gen_vitdet_fixture.py` drew the weights and the input
    /// from: values in [-0.5, 0.5) times `scale`, rounded to f32 last.
    struct Lcg(u64);

    impl Lcg {
        fn take(&mut self, n: usize, scale: f64) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((self.0 >> 40) as f64 / 16777216.0 - 0.5) * scale) as f32
                })
                .collect()
        }
    }

    /// The fixture's network: 32 wide, two heads, four blocks, global
    /// attention in blocks 1 and 3.
    fn mini() -> (VitSpec, Params, Vec<f32>) {
        let spec = VitSpec {
            dim: 32,
            heads: 2,
            depth: 4,
            patch: 16,
            window: 14,
            global: vec![1, 3],
            pretrain_grid: 14,
        };
        let d = spec.dim;
        let mut shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let t = "image_encoder.trunk";
        shapes.insert(format!("{t}.pos_embed"), vec![1, 197, d]);
        shapes.insert(format!("{t}.patch_embed.proj.weight"), vec![d, 3, 16, 16]);
        shapes.insert(format!("{t}.patch_embed.proj.bias"), vec![d]);
        for i in 0..spec.depth {
            let b = format!("{t}.blocks.{i}");
            for (k, s) in [
                ("attn.qkv.weight", vec![3 * d, d]),
                ("attn.qkv.bias", vec![3 * d]),
                ("attn.proj.weight", vec![d, d]),
                ("attn.proj.bias", vec![d]),
                ("mlp.layers.0.weight", vec![4 * d, d]),
                ("mlp.layers.0.bias", vec![4 * d]),
                ("mlp.layers.1.weight", vec![d, 4 * d]),
                ("mlp.layers.1.bias", vec![d]),
                ("norm1.weight", vec![d]),
                ("norm1.bias", vec![d]),
                ("norm2.weight", vec![d]),
                ("norm2.bias", vec![d]),
            ] {
                shapes.insert(format!("{b}.{k}"), s);
            }
        }
        let n = "image_encoder.neck.convs.0";
        shapes.insert(format!("{n}.conv_1x1.weight"), vec![D_MODEL, d, 1, 1]);
        shapes.insert(format!("{n}.conv_3x3.weight"), vec![D_MODEL, D_MODEL, 3, 3]);
        for k in ["norm_0", "norm_1"] {
            shapes.insert(format!("{n}.{k}.weight"), vec![D_MODEL]);
            shapes.insert(format!("{n}.{k}.bias"), vec![D_MODEL]);
        }
        let mut rng = Lcg(0x5EED_EFF1);
        let mut tensors = HashMap::new();
        for (k, shape) in shapes {
            let numel: usize = shape.iter().product();
            let scale = if shape.len() == 1 {
                0.4
            } else if k.ends_with("pos_embed") {
                1.0
            } else {
                12f64.sqrt() / (shape[1..].iter().product::<usize>() as f64).sqrt()
            };
            let mut data = rng.take(numel, scale);
            if shape.len() == 1 && k.contains("norm") && k.ends_with("weight") {
                data.iter_mut().for_each(|v| *v += 1.0);
            }
            tensors.insert(k, WTensor { shape, data });
        }
        let input = rng.take(3 * 256 * 256, 4.0);
        (spec, Params::new(tensors), input)
    }

    #[test]
    fn the_vit_and_its_neck_match_efficienttam() {
        let (spec, p, input) = mini();
        let dev: burn::tensor::Device<Bk> = Default::default();
        let vit = VitDet::<Bk>::load_spec(&p, spec, 256, &dev).unwrap();
        let want = load_safetensors(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/data/medsam2-vitdet.safetensors"),
        )
        .expect("tests/data/medsam2-vitdet.safetensors");
        let x: Tensor<Bk, 4> = ops::from_slice(&input, [1, 3, 256, 256], &dev);
        let trunk = ops::to_vec(vit.trunk(x.clone()));
        let neck = ops::to_vec(vit.forward(x));
        for (got, key) in [(trunk, "vitdet.trunk"), (neck, "vitdet.neck")] {
            let w = &want[key];
            assert_eq!(got.len(), w.data.len(), "{key}");
            let range = w.data.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got
                .iter()
                .zip(&w.data)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                worst < 1e-4 * range.max(1.0),
                "{key}: worst {worst}, range {range}"
            );
        }
    }

    #[test]
    fn the_published_sizes_have_their_heads() {
        let t = VitSpec::published(192).unwrap();
        assert_eq!(
            (t.heads, t.depth, t.window, t.global.clone()),
            (3, 12, 14, vec![2, 5, 8, 11])
        );
        assert_eq!(VitSpec::published(384).unwrap().heads, 6);
        assert!(VitSpec::published(256).is_err());
    }
}
