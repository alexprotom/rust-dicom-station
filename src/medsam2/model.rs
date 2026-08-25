//! The assembled network, and the two ways a slice can be conditioned.
//!
//! Everything above this module is a piece; this is the piece that owns them,
//! plus the handful of model-root parameters that belong to no submodule:
//! `no_mem_embed`, the seven temporal encodings of the memory bank, and the
//! projection that gives an object pointer its temporal position.
//!
//! The split that matters for performance is [`Medsam2::encode_slice`] versus
//! everything else. Encoding is ~22 G multiply-accumulates and depends only on
//! the image, so it can be done once per slice and reused across prompts and
//! across both propagation directions; the rest — memory attention, decoder,
//! memory encoder — is another ~26 G but it is strictly sequential, because
//! slice *n* needs slice *n-1*'s memory.

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config::{self, D_MODEL, MEM_DIM};
use super::hiera::Hiera;
use super::layers::Lin;
use super::memattn::MemoryAttention;
use super::memory::MemoryEncoder;
use super::neck::{self, Neck};
use super::ops;
use super::resample;
use super::sam::{SamHead, SamOutput};

/// One slice, encoded. Independent of any prompt, so worth caching.
#[derive(Clone)]
pub struct SliceFeatures<B: Backend> {
    /// The neck's level-2 map, `[1, 256, 32, 32]` — what the memory attention
    /// and the SAM head consume, and what the memory encoder pairs with a
    /// mask.
    pub pix_feat: Tensor<B, 4>,
    /// `conv_s0` and `conv_s1` of the two high-resolution levels.
    pub high_res: [Tensor<B, 4>; 2],
}

pub struct Medsam2<B: Backend> {
    pub trunk: Hiera<B>,
    pub neck: Neck<B>,
    pub head: SamHead<B>,
    pub memory_encoder: MemoryEncoder<B>,
    pub memory_attention: MemoryAttention<B>,
    /// Added to a conditioning slice's features in place of any memory.
    no_mem_embed: Tensor<B, 4>,
    /// `[NUM_MASKMEM, MEM_DIM]`; row 0 is the most recent tracked slice and
    /// row 6 the conditioning slices.
    maskmem_tpos_enc: Tensor<B, 2>,
    obj_ptr_tpos_proj: Lin<B>,
    /// The sine encoding of the image tokens, `[1, tokens, 256]`.
    image_pos: Tensor<B, 3>,
    device: B::Device,
}

impl<B: Backend> Medsam2<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<Medsam2<B>> {
        let grid = config::EMBED_GRID;
        let image_pos = neck::sine_pos_embed::<B>(grid, grid, D_MODEL, dev)
            .reshape([1, D_MODEL, grid * grid])
            .swap_dims(1, 2);
        Ok(Medsam2 {
            trunk: Hiera::load(p, dev)?,
            neck: Neck::load(p, dev)?,
            head: SamHead::load(p, dev)?,
            memory_encoder: MemoryEncoder::load(p, dev)?,
            memory_attention: MemoryAttention::load(p, dev)?,
            no_mem_embed: ops::from_slice(
                p.get("no_mem_embed", &[1, 1, D_MODEL])?,
                [1, D_MODEL, 1, 1],
                dev,
            ),
            maskmem_tpos_enc: ops::from_slice(
                p.get(
                    "maskmem_tpos_enc",
                    &[config::NUM_MASKMEM, 1, 1, MEM_DIM],
                )?,
                [config::NUM_MASKMEM, MEM_DIM],
                dev,
            ),
            obj_ptr_tpos_proj: Lin::load(p, "obj_ptr_tpos_proj", MEM_DIM, D_MODEL, dev)?,
            image_pos,
            device: dev.clone(),
        })
    }

    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// The image encoder: trunk, neck, and the two high-resolution
    /// projections the decoder needs.
    pub fn encode_slice(&self, image: Tensor<B, 4>) -> SliceFeatures<B> {
        let stages = self.trunk.forward(image);
        let levels = self.neck.forward(&stages);
        let high_res = self
            .head
            .decoder
            .project_high_res(levels[0].clone(), levels[1].clone());
        SliceFeatures {
            pix_feat: levels[2].clone(),
            high_res,
        }
    }

    /// A conditioning slice: `directly_add_no_mem_embed` short-circuits the
    /// memory attention entirely and adds one learned vector instead.
    pub fn without_memory(&self, feats: &SliceFeatures<B>) -> Tensor<B, 4> {
        feats.pix_feat.clone() + self.no_mem_embed.clone()
    }

    /// A tracked slice, conditioned on the assembled memory.
    pub fn with_memory(
        &self,
        feats: &SliceFeatures<B>,
        memory: Tensor<B, 3>,
        memory_pos: Tensor<B, 3>,
        num_obj_ptr_tokens: usize,
    ) -> Tensor<B, 4> {
        let grid = config::EMBED_GRID;
        let curr = self.flatten(feats.pix_feat.clone());
        let out = self.memory_attention.forward(
            curr,
            self.image_pos.clone(),
            memory,
            memory_pos,
            num_obj_ptr_tokens,
        );
        out.swap_dims(1, 2).reshape([1, D_MODEL, grid, grid])
    }

    /// `[1, c, h, w]` -> `[1, h * w, c]`.
    pub fn flatten(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, c, h, w] = x.dims();
        x.reshape([b, c, h * w]).swap_dims(1, 2)
    }

    /// One row of the temporal encoding table, `[1, 1, MEM_DIM]`.
    ///
    /// Row 0 belongs to the slice immediately before the one being tracked and
    /// row `NUM_MASKMEM - 1` to the prompted slices — the reference indexes it
    /// as `maskmem_tpos_enc[num_maskmem - t_pos - 1]`, and callers here pass
    /// that row directly.
    pub fn tpos_row(&self, row: usize) -> Tensor<B, 3> {
        self.maskmem_tpos_enc
            .clone()
            .slice([row..row + 1, 0..MEM_DIM])
            .reshape([1, 1, MEM_DIM])
    }

    /// Temporal encodings for a set of object pointers, `[1, p, MEM_DIM]`.
    ///
    /// `offsets` are how far each pointer is from the current slice — signed
    /// for conditioning slices (`use_signed_tpos_enc_to_obj_ptrs`) and
    /// positive for tracked ones — normalized by `t_diff_max` and turned into
    /// a 256-wide sine encoding before the projection.
    pub fn pointer_pos(&self, offsets: &[f32], t_diff_max: f32) -> Tensor<B, 3> {
        let mut data = Vec::with_capacity(offsets.len() * D_MODEL);
        let half = D_MODEL / 2;
        for off in offsets {
            let p = f64::from(*off) / f64::from(t_diff_max);
            let mut sin = Vec::with_capacity(half);
            let mut cos = Vec::with_capacity(half);
            for i in 0..half {
                let dim_t = f64::from(config::PE_TEMPERATURE)
                    .powf(2.0 * ((i / 2) as f64) / half as f64);
                sin.push((p / dim_t).sin() as f32);
                cos.push((p / dim_t).cos() as f32);
            }
            data.extend(sin);
            data.extend(cos);
        }
        let pe: Tensor<B, 3> =
            ops::from_slice(&data, [1, offsets.len(), D_MODEL], &self.device);
        self.obj_ptr_tpos_proj.apply(pe)
    }

    /// `_use_mask_as_output`: a mask prompt bypasses the decoder entirely.
    ///
    /// `use_mask_input_as_output_without_sam` means the mask the user supplies
    /// *is* the answer for that slice — scaled to the network's logit range
    /// rather than predicted. The decoder still runs, but only to produce an
    /// object pointer for the slices that follow.
    pub fn mask_as_output(
        &self,
        pix_feat: &Tensor<B, 4>,
        feats: &SliceFeatures<B>,
        mask: Tensor<B, 4>,
    ) -> SamOutput<B> {
        let size = config::IMAGE_SIZE;
        let low = config::LOW_RES;
        let present = ops::to_vec(mask.clone()).iter().any(|v| *v > 0.0);
        let high_res_masks = mask
            .clone()
            .mul_scalar(config::SIGMOID_SCALE)
            .add_scalar(config::SIGMOID_BIAS);
        // The reference shrinks this one with an antialiased filter, which is
        // not what `interpolate` does by default.
        let low_res_masks: Tensor<B, 4> = ops::from_slice(
            &resample::resize(
                &ops::to_vec(high_res_masks.clone()),
                [size, size],
                [low, low],
                resample::Filter::Triangle,
                true,
            ),
            [1, 1, low, low],
            &self.device,
        );

        // The head runs with the mask as its only prompt, for the pointer.
        let head_out = self
            .head
            .forward(pix_feat.clone(), &feats.high_res, &[], Some(mask), false);
        let obj_ptr = if present {
            head_out.obj_ptr
        } else {
            self.head.no_obj_ptr()
        };
        let score = config::SIGMOID_SCALE * f32::from(present) + config::SIGMOID_BIAS;

        SamOutput {
            low_res_multimasks: low_res_masks.clone(),
            low_res_masks,
            high_res_masks,
            ious: Tensor::ones([1, 1], &self.device),
            obj_ptr,
            object_score_logits: Tensor::full([1, 1], score, &self.device),
        }
    }
}
