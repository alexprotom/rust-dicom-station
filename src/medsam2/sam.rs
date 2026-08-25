//! The SAM head as the tracker calls it — `SAM2Base._forward_sam_heads`.
//!
//! This is where the prompt encoder, the mask decoder and three of the
//! model's root-level parameters meet: a prompt goes in, and what comes out
//! is the pair a tracked slice needs — a mask, and a 256-dimensional
//! **object pointer** summarizing what was segmented, which later slices
//! attend to.
//!
//! The object-presence head is wired in here too. When it says the object has
//! left the slice, the logits are replaced wholesale by
//! [`config::NO_OBJ_SCORE`] and the pointer is swapped for the learned
//! `no_obj_ptr`, so an absent object propagates as a definite absence rather
//! than as a weak mask.

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config::{self, D_MODEL};
use super::decoder::{self, MaskDecoder};
use super::layers::{Conv, Mlp};
use super::ops;
use super::prompt::{Point, PromptEncoder};

/// What one prompt against one encoded slice produces.
pub struct SamOutput<B: Backend> {
    /// The candidate masks the decoder kept, `[1, m, 128, 128]`.
    pub low_res_multimasks: Tensor<B, 4>,
    /// The chosen mask at the network's own resolution, `[1, 1, 128, 128]`.
    pub low_res_masks: Tensor<B, 4>,
    /// The same, bilinearly resized to the input, `[1, 1, 512, 512]`.
    pub high_res_masks: Tensor<B, 4>,
    /// Predicted IoU of each candidate.
    pub ious: Tensor<B, 2>,
    /// `[1, 256]`, the memory of *what* was segmented.
    pub obj_ptr: Tensor<B, 2>,
    /// `[1, 1]`; positive means the object is present.
    pub object_score_logits: Tensor<B, 2>,
}

impl<B: Backend> SamOutput<B> {
    /// Whether the object-score head thinks the object is in this slice.
    pub fn object_present(&self) -> bool {
        ops::to_vec(self.object_score_logits.clone())[0] > 0.0
    }
}

pub struct SamHead<B: Backend> {
    pub prompt: PromptEncoder<B>,
    pub decoder: MaskDecoder<B>,
    obj_ptr_proj: Mlp<B>,
    no_obj_ptr: Tensor<B, 2>,
    /// `Conv2d(1, 1, 4, 4)`: brings a full-resolution mask prompt down to the
    /// 128 x 128 the prompt encoder expects.
    mask_downsample: Conv<B>,
}

impl<B: Backend> SamHead<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<SamHead<B>> {
        Ok(SamHead {
            prompt: PromptEncoder::load(p, dev)?,
            decoder: MaskDecoder::load(p, dev)?,
            obj_ptr_proj: Mlp::load(
                p,
                "obj_ptr_proj",
                &[D_MODEL, D_MODEL, D_MODEL, D_MODEL],
                dev,
            )?,
            no_obj_ptr: ops::from_slice(
                p.get("no_obj_ptr", &[1, D_MODEL])?,
                [1, D_MODEL],
                dev,
            ),
            mask_downsample: Conv::load(p, "mask_downsample", 1, 1, 4, 4, 0, 1, dev)?,
        })
    }

    /// `_use_multimask`. Note that a box counts as **two** points and so does
    /// *not* qualify, while a single click and an unprompted tracked slice
    /// both do.
    pub fn use_multimask(prompt_points: usize) -> bool {
        (config::MULTIMASK_MIN_PT..=config::MULTIMASK_MAX_PT).contains(&prompt_points)
    }

    /// `obj_ptr_proj` on its own, for tests and for the tracker's own use.
    pub fn project_obj_ptr(&self, token: Tensor<B, 2>) -> Tensor<B, 2> {
        self.obj_ptr_proj.apply(token)
    }

    /// The learned pointer used when the object is absent.
    pub fn no_obj_ptr(&self) -> Tensor<B, 2> {
        self.no_obj_ptr.clone()
    }

    /// Bring a mask prompt to the prompt encoder's expected size.
    pub fn downsample_mask(&self, mask: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, _, h, w] = mask.dims();
        if h == config::MASK_PROMPT_SIZE && w == config::MASK_PROMPT_SIZE {
            mask
        } else {
            self.mask_downsample.apply(mask)
        }
    }

    /// One prompt against one encoded slice.
    ///
    /// `pix_feat` is the neck's level-2 map `[1, 256, 32, 32]`; `high_res` are
    /// the two projected high-resolution features. `points` are in pixels of
    /// the 512 x 512 input — empty means a tracked slice, which sends a single
    /// padding point.
    pub fn forward(
        &self,
        pix_feat: Tensor<B, 4>,
        high_res: &[Tensor<B, 4>; 2],
        points: &[Point],
        mask_input: Option<Tensor<B, 4>>,
        multimask: bool,
    ) -> SamOutput<B> {
        let no_prompt = Point::none();
        let points = if points.is_empty() {
            &no_prompt[..]
        } else {
            points
        };
        let prompts = self
            .prompt
            .encode(points, mask_input.map(|m| self.downsample_mask(m)));
        let decoded = self.decoder.forward(
            pix_feat,
            self.prompt.dense_pe(),
            prompts.sparse,
            prompts.dense,
            high_res,
        );
        let selected = self.decoder.select(decoded, multimask);

        let present = ops::to_vec(selected.object_score_logits.clone())[0] > 0.0;
        let [b, m, h, w] = selected.masks.dims();
        let low_res_multimasks = if present {
            selected.masks
        } else {
            Tensor::full([b, m, h, w], config::NO_OBJ_SCORE, &selected.masks.device())
        };
        let size = config::IMAGE_SIZE;
        let high_res_multimasks = ops::resize_bilinear(low_res_multimasks.clone(), [size, size]);

        // Which candidate wins, and which token the pointer comes from. The
        // *predicted IoUs of every candidate* are reported as they are — the
        // reference returns the whole vector, and the tracker uses it.
        let (low_res_masks, high_res_masks, token_index) = if multimask {
            let best = decoder::argmax_iou(selected.ious.clone());
            (
                low_res_multimasks
                    .clone()
                    .slice([0..b, best..best + 1, 0..h, 0..w]),
                high_res_multimasks.slice([0..b, best..best + 1, 0..size, 0..size]),
                best.min(selected.sam_tokens.dims()[1] - 1),
            )
        } else {
            (low_res_multimasks.clone(), high_res_multimasks, 0)
        };

        let token = selected
            .sam_tokens
            .slice([0..b, token_index..token_index + 1, 0..D_MODEL])
            .reshape([b, D_MODEL]);
        let projected = self.obj_ptr_proj.apply(token);
        // `fixed_no_obj_ptr`: an absent object contributes the learned
        // pointer instead of a projection of whatever the decoder produced.
        let obj_ptr = if present {
            projected
        } else {
            self.no_obj_ptr.clone()
        };

        SamOutput {
            low_res_multimasks,
            low_res_masks,
            high_res_masks,
            ious: selected.ious,
            obj_ptr,
            object_score_logits: selected.object_score_logits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_does_not_use_multimask_but_a_click_does() {
        assert!(!SamHead::<burn::backend::NdArray>::use_multimask(2));
        assert!(SamHead::<burn::backend::NdArray>::use_multimask(1));
        // a tracked slice sends no points at all
        assert!(SamHead::<burn::backend::NdArray>::use_multimask(0));
        assert!(!SamHead::<burn::backend::NdArray>::use_multimask(3));
    }
}
