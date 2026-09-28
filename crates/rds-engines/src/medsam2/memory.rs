//! The memory encoder: a mask and its slice, compressed into a memory.
//!
//! What gets stored per segmented slice is not the mask but a 64-channel
//! 32 x 32 tensor: the mask, downsampled by four strided convolutions to the
//! embedding grid, added to a projection of that slice's **raw** backbone
//! feature (not the memory-conditioned one), fused by two ConvNeXt-style
//! blocks and projected down. At 256 KB per slice a full seven-slice bank is
//! negligible next to the encoder's own activations.
//!
//! One numeric quirk matters: the mask handed in is not a probability. On a
//! prompted slice it is hard-binarized at zero, and either way it is scaled
//! to `[-10, +10]` (`x * 20 - 10`) before the downsampler, whose internal
//! sigmoid is then skipped.

use anyhow::Result;
use burn::tensor::activation::{gelu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config::{self, D_MODEL, MEM_DIM};
use super::layers::{Conv, Lin, Norm};
use super::neck::sine_pos_embed;
use super::ops;

/// One `CXBlock`: depthwise 7 x 7, channel-wise norm, a pointwise MLP and a
/// learned per-channel scale, added back to the input.
struct CxBlock<B: Backend> {
    dwconv: Conv<B>,
    norm: Norm<B>,
    pwconv1: Lin<B>,
    pwconv2: Lin<B>,
    gamma: Tensor<B, 4>,
}

impl<B: Backend> CxBlock<B> {
    fn load(p: &Params, prefix: &str, dev: &B::Device) -> Result<CxBlock<B>> {
        Ok(CxBlock {
            dwconv: Conv::load(
                p,
                &format!("{prefix}.dwconv"),
                D_MODEL,
                D_MODEL,
                config::CX_KERNEL,
                1,
                config::CX_KERNEL / 2,
                D_MODEL,
                dev,
            )?,
            norm: Norm::load6(p, &format!("{prefix}.norm"), D_MODEL, dev)?,
            pwconv1: Lin::load(
                p,
                &format!("{prefix}.pwconv1"),
                config::CX_MLP,
                D_MODEL,
                dev,
            )?,
            pwconv2: Lin::load(
                p,
                &format!("{prefix}.pwconv2"),
                D_MODEL,
                config::CX_MLP,
                dev,
            )?,
            gamma: ops::from_slice(
                p.vec(&format!("{prefix}.gamma"), D_MODEL)?,
                [1, 1, 1, D_MODEL],
                dev,
            ),
        })
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        let y = self.norm.apply_2d(self.dwconv.apply(x));
        // the pointwise part runs channels-last
        let y = y.permute([0, 2, 3, 1]);
        let y = self.pwconv2.apply(gelu(self.pwconv1.apply(y)));
        let y = (y * self.gamma.clone()).permute([0, 3, 1, 2]);
        residual + y
    }
}

/// One slice's memory: what later slices attend to.
pub struct Memory<B: Backend> {
    /// `[1, 64, 32, 32]`.
    pub features: Tensor<B, 4>,
    /// `[1, 64, 32, 32]`, the sine encoding of the same grid.
    pub pos: Tensor<B, 4>,
}

pub struct MemoryEncoder<B: Backend> {
    mask_convs: Vec<Conv<B>>,
    mask_norms: Vec<Norm<B>>,
    mask_final: Conv<B>,
    pix_feat_proj: Conv<B>,
    fuser: Vec<CxBlock<B>>,
    out_proj: Conv<B>,
    /// Built once; it depends only on the grid.
    pos: Tensor<B, 4>,
    /// A model-root parameter, but this is the only place it is used: it
    /// marks a memory as "the object was not here".
    no_obj_embed_spatial: Tensor<B, 4>,
}

impl<B: Backend> MemoryEncoder<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<MemoryEncoder<B>> {
        let e = "memory_encoder.mask_downsampler.encoder";
        let mut mask_convs = Vec::with_capacity(config::MASK_DOWN_LAYERS);
        let mut mask_norms = Vec::with_capacity(config::MASK_DOWN_LAYERS);
        let mut ch = 1;
        for i in 0..config::MASK_DOWN_LAYERS {
            let out = ch * config::MASK_DOWN_STRIDE * config::MASK_DOWN_STRIDE;
            mask_convs.push(Conv::load(
                p,
                &format!("{e}.{}", 3 * i),
                out,
                ch,
                3,
                config::MASK_DOWN_STRIDE,
                1,
                1,
                dev,
            )?);
            mask_norms.push(Norm::load6(p, &format!("{e}.{}", 3 * i + 1), out, dev)?);
            ch = out;
        }
        let mut fuser = Vec::with_capacity(config::FUSER_LAYERS);
        for i in 0..config::FUSER_LAYERS {
            fuser.push(CxBlock::load(
                p,
                &format!("memory_encoder.fuser.layers.{i}"),
                dev,
            )?);
        }
        Ok(MemoryEncoder {
            mask_convs,
            mask_norms,
            mask_final: Conv::load_1x1(
                p,
                &format!("{e}.{}", 3 * config::MASK_DOWN_LAYERS),
                D_MODEL,
                D_MODEL,
                dev,
            )?,
            pix_feat_proj: Conv::load_1x1(
                p,
                "memory_encoder.pix_feat_proj",
                D_MODEL,
                D_MODEL,
                dev,
            )?,
            fuser,
            out_proj: Conv::load_1x1(p, "memory_encoder.out_proj", MEM_DIM, D_MODEL, dev)?,
            pos: sine_pos_embed(config::EMBED_GRID, config::EMBED_GRID, MEM_DIM, dev),
            no_obj_embed_spatial: ops::from_slice(
                p.get("no_obj_embed_spatial", &[1, MEM_DIM])?,
                [1, MEM_DIM, 1, 1],
                dev,
            ),
        })
    }

    /// `MaskDownSampler`: 512 -> 32 in four strided steps.
    fn downsample_mask(&self, mask: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = mask;
        for (conv, norm) in self.mask_convs.iter().zip(self.mask_norms.iter()) {
            x = gelu(norm.apply_2d(conv.apply(x)));
        }
        self.mask_final.apply(x)
    }

    /// `_encode_new_memory`.
    ///
    /// `pix_feat` is the **raw** neck level-2 feature of this slice, not the
    /// memory-conditioned one. `from_prompt` selects the hard binarization
    /// the reference uses on prompted slices; `object_present` is the
    /// object-score head's verdict.
    pub fn encode(
        &self,
        pix_feat: Tensor<B, 4>,
        high_res_masks: Tensor<B, 4>,
        from_prompt: bool,
        object_present: bool,
    ) -> Memory<B> {
        let mask = if from_prompt {
            high_res_masks.greater_elem(0.0).float()
        } else {
            sigmoid(high_res_masks)
        };
        let mask = mask
            .mul_scalar(config::SIGMOID_SCALE)
            .add_scalar(config::SIGMOID_BIAS);

        let x = self.pix_feat_proj.apply(pix_feat) + self.downsample_mask(mask);
        let mut x = x;
        for block in &self.fuser {
            x = block.forward(x);
        }
        let mut features = self.out_proj.apply(x);
        if !object_present {
            let [_, _, h, w] = features.dims();
            features = features
                + self
                    .no_obj_embed_spatial
                    .clone()
                    .repeat_dim(2, h)
                    .repeat_dim(3, w);
        }
        Memory {
            features,
            pos: self.pos.clone(),
        }
    }
}
