//! The mask decoder: a two-way transformer, then a hypernetwork.
//!
//! Six output tokens are prepended to the prompt tokens — one object-score
//! token, one IoU token and four mask tokens — and run through two blocks
//! that attend tokens to image, image to tokens, and tokens to themselves.
//! The image side is then upscaled twice, fused with the encoder's two
//! high-resolution feature maps, and dotted with per-token filters produced
//! by four small MLPs. What comes out is four `128 x 128` logit maps, four
//! predicted IoUs and one object-presence logit.
//!
//! Which of the four masks is used is not a detail: a **box** prompt takes
//! the single-mask token (with a stability fallback to the best of the other
//! three), while a **propagated** slice takes the best of the three
//! multi-mask outputs by predicted IoU. See
//! [`super::config::MULTIMASK_MAX_PT`] and `use_multimask` in
//! [`super::sam`].

use anyhow::Result;
use burn::tensor::activation::gelu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::config::{self, D_MODEL, NUM_MASK_TOKENS};
use super::layers::{Conv, ConvT2x, Mlp, Norm, SamAttention};
use super::ops;

/// `[D_MODEL] * IOU_HEAD_DEPTH + [NUM_MASK_TOKENS]`: the IoU head's widths.
fn iou_head_dims() -> Vec<usize> {
    let mut dims = vec![D_MODEL; config::IOU_HEAD_DEPTH];
    dims.push(NUM_MASK_TOKENS);
    dims
}

/// One `TwoWayAttentionBlock`.
struct TwoWayBlock<B: Backend> {
    self_attn: SamAttention<B>,
    norm1: Norm<B>,
    cross_token_to_image: SamAttention<B>,
    norm2: Norm<B>,
    mlp: Mlp<B>,
    norm3: Norm<B>,
    cross_image_to_token: SamAttention<B>,
    norm4: Norm<B>,
    /// True only on layer 0: its self-attention sees no positional encoding
    /// and has no residual.
    skip_first_layer_pe: bool,
}

impl<B: Backend> TwoWayBlock<B> {
    fn load(p: &Params, prefix: &str, first: bool, dev: &B::Device) -> Result<TwoWayBlock<B>> {
        let (h, ds) = (config::DEC_HEADS, config::DEC_DOWNSAMPLE);
        Ok(TwoWayBlock {
            self_attn: SamAttention::load(
                p,
                &format!("{prefix}.self_attn"),
                D_MODEL,
                h,
                1,
                D_MODEL,
                dev,
            )?,
            norm1: Norm::load(p, &format!("{prefix}.norm1"), D_MODEL, dev)?,
            cross_token_to_image: SamAttention::load(
                p,
                &format!("{prefix}.cross_attn_token_to_image"),
                D_MODEL,
                h,
                ds,
                D_MODEL,
                dev,
            )?,
            norm2: Norm::load(p, &format!("{prefix}.norm2"), D_MODEL, dev)?,
            mlp: Mlp::load(
                p,
                &format!("{prefix}.mlp"),
                &[D_MODEL, config::DEC_MLP, D_MODEL],
                dev,
            )?,
            norm3: Norm::load(p, &format!("{prefix}.norm3"), D_MODEL, dev)?,
            cross_image_to_token: SamAttention::load(
                p,
                &format!("{prefix}.cross_attn_image_to_token"),
                D_MODEL,
                h,
                ds,
                D_MODEL,
                dev,
            )?,
            norm4: Norm::load(p, &format!("{prefix}.norm4"), D_MODEL, dev)?,
            skip_first_layer_pe: first,
        })
    }

    fn forward(
        &self,
        queries: Tensor<B, 3>,
        keys: Tensor<B, 3>,
        query_pe: &Tensor<B, 3>,
        key_pe: &Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // self-attention
        let queries = if self.skip_first_layer_pe {
            self.self_attn
                .forward(queries.clone(), queries.clone(), queries)
        } else {
            let q = queries.clone() + query_pe.clone();
            queries.clone() + self.self_attn.forward(q.clone(), q, queries)
        };
        let queries = self.norm1.apply(queries);

        // tokens attending to the image
        let q = queries.clone() + query_pe.clone();
        let k = keys.clone() + key_pe.clone();
        let queries = queries.clone()
            + self
                .cross_token_to_image
                .forward(q, k.clone(), keys.clone());
        let queries = self.norm2.apply(queries);

        // MLP
        let queries = queries.clone() + self.mlp.apply(queries);
        let queries = self.norm3.apply(queries);

        // the image attending to the tokens: note that `k` is the query here
        let q = queries.clone() + query_pe.clone();
        let k = keys.clone() + key_pe.clone();
        let keys = keys + self.cross_image_to_token.forward(k, q, queries.clone());
        let keys = self.norm4.apply(keys);

        (queries, keys)
    }
}

struct TwoWayTransformer<B: Backend> {
    layers: Vec<TwoWayBlock<B>>,
    final_attn: SamAttention<B>,
    norm_final: Norm<B>,
}

impl<B: Backend> TwoWayTransformer<B> {
    fn load(p: &Params, prefix: &str, dev: &B::Device) -> Result<TwoWayTransformer<B>> {
        let mut layers = Vec::with_capacity(config::DEC_LAYERS);
        for i in 0..config::DEC_LAYERS {
            layers.push(TwoWayBlock::load(
                p,
                &format!("{prefix}.layers.{i}"),
                i == 0,
                dev,
            )?);
        }
        Ok(TwoWayTransformer {
            layers,
            final_attn: SamAttention::load(
                p,
                &format!("{prefix}.final_attn_token_to_image"),
                D_MODEL,
                config::DEC_HEADS,
                config::DEC_DOWNSAMPLE,
                D_MODEL,
                dev,
            )?,
            norm_final: Norm::load(p, &format!("{prefix}.norm_final_attn"), D_MODEL, dev)?,
        })
    }

    /// `image` and `image_pe` are `[1, 256, h, w]`; `tokens` is `[1, n, 256]`.
    fn forward(
        &self,
        image: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        tokens: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [b, c, h, w] = image.dims();
        let flatten = |t: Tensor<B, 4>| t.reshape([b, c, h * w]).swap_dims(1, 2);
        let key_pe = flatten(image_pe);
        let mut keys = flatten(image);
        let mut queries = tokens.clone();
        for layer in &self.layers {
            let (q, k) = layer.forward(queries, keys, &tokens, &key_pe);
            queries = q;
            keys = k;
        }
        // one last pass from the tokens to the image
        let q = queries.clone() + tokens;
        let k = keys.clone() + key_pe;
        let queries = queries + self.final_attn.forward(q, k, keys.clone());
        (self.norm_final.apply(queries), keys)
    }
}

/// Everything the decoder produces before a mask is chosen.
pub struct Decoded<B: Backend> {
    /// `[1, 4, 128, 128]`.
    pub masks: Tensor<B, 4>,
    /// `[1, 4]`.
    pub ious: Tensor<B, 2>,
    /// `[1, 4, 256]`.
    pub mask_tokens: Tensor<B, 3>,
    /// `[1, 1]`.
    pub object_score_logits: Tensor<B, 2>,
}

/// The decoder's output after the multi-mask choice has been made.
pub struct Selected<B: Backend> {
    /// `[1, m, 128, 128]`, `m` = 3 for a tracked slice, 1 otherwise.
    pub masks: Tensor<B, 4>,
    pub ious: Tensor<B, 2>,
    /// The token(s) the object pointer is projected from.
    pub sam_tokens: Tensor<B, 3>,
    pub object_score_logits: Tensor<B, 2>,
}

pub struct MaskDecoder<B: Backend> {
    transformer: TwoWayTransformer<B>,
    obj_score_token: Tensor<B, 3>,
    iou_token: Tensor<B, 3>,
    mask_tokens: Tensor<B, 3>,
    up0: ConvT2x<B>,
    up_norm: Norm<B>,
    up1: ConvT2x<B>,
    hypernetworks: Vec<Mlp<B>>,
    iou_head: Mlp<B>,
    obj_score_head: Mlp<B>,
    conv_s0: Conv<B>,
    conv_s1: Conv<B>,
}

impl<B: Backend> MaskDecoder<B> {
    pub fn load(p: &Params, dev: &B::Device) -> Result<MaskDecoder<B>> {
        let d = "sam_mask_decoder";
        let token = |key: &str, n: usize| -> Result<Tensor<B, 3>> {
            Ok(ops::from_slice(
                p.get(key, &[n, D_MODEL])?,
                [1, n, D_MODEL],
                dev,
            ))
        };
        let mut hypernetworks = Vec::with_capacity(NUM_MASK_TOKENS);
        for i in 0..NUM_MASK_TOKENS {
            hypernetworks.push(Mlp::load(
                p,
                &format!("{d}.output_hypernetworks_mlps.{i}"),
                &[D_MODEL, D_MODEL, D_MODEL, config::HYPER_DIM],
                dev,
            )?);
        }
        Ok(MaskDecoder {
            transformer: TwoWayTransformer::load(p, &format!("{d}.transformer"), dev)?,
            obj_score_token: token(&format!("{d}.obj_score_token.weight"), 1)?,
            iou_token: token(&format!("{d}.iou_token.weight"), 1)?,
            mask_tokens: token(&format!("{d}.mask_tokens.weight"), NUM_MASK_TOKENS)?,
            up0: ConvT2x::load(
                p,
                &format!("{d}.output_upscaling.0"),
                D_MODEL / 4,
                D_MODEL,
                dev,
            )?,
            up_norm: Norm::load6(p, &format!("{d}.output_upscaling.1"), D_MODEL / 4, dev)?,
            up1: ConvT2x::load(
                p,
                &format!("{d}.output_upscaling.3"),
                config::UPSCALED_CH,
                D_MODEL / 4,
                dev,
            )?,
            hypernetworks,
            iou_head: Mlp::load_with(
                p,
                &format!("{d}.iou_prediction_head"),
                &iou_head_dims(),
                true,
                dev,
            )?,
            obj_score_head: Mlp::load(
                p,
                &format!("{d}.pred_obj_score_head"),
                &[D_MODEL, D_MODEL, D_MODEL, 1],
                dev,
            )?,
            conv_s0: Conv::load_1x1(
                p,
                &format!("{d}.conv_s0"),
                config::HIGH_RES_S0_CH,
                D_MODEL,
                dev,
            )?,
            conv_s1: Conv::load_1x1(
                p,
                &format!("{d}.conv_s1"),
                config::HIGH_RES_S1_CH,
                D_MODEL,
                dev,
            )?,
        })
    }

    /// Project the neck's two high-resolution levels for the upscaling path.
    /// This happens in `forward_image`, once per slice, not per prompt.
    pub fn project_high_res(
        &self,
        level0: Tensor<B, 4>,
        level1: Tensor<B, 4>,
    ) -> [Tensor<B, 4>; 2] {
        [self.conv_s0.apply(level0), self.conv_s1.apply(level1)]
    }

    pub fn forward(
        &self,
        image: Tensor<B, 4>,
        image_pe: Tensor<B, 4>,
        sparse: Tensor<B, 3>,
        dense: Tensor<B, 4>,
        high_res: &[Tensor<B, 4>; 2],
    ) -> Decoded<B> {
        let tokens = Tensor::cat(
            vec![
                self.obj_score_token.clone(),
                self.iou_token.clone(),
                self.mask_tokens.clone(),
                sparse,
            ],
            1,
        );
        let src = image + dense;
        let [b, c, h, w] = src.dims();

        let (hs, keys) = self.transformer.forward(src, image_pe, tokens);
        // token 0 is the object score, 1 the IoU, 2..6 the masks
        let iou_token_out = hs.clone().slice([0..b, 1..2, 0..D_MODEL]);
        let mask_tokens_out = hs.clone().slice([0..b, 2..2 + NUM_MASK_TOKENS, 0..D_MODEL]);
        let obj_token_out = hs.slice([0..b, 0..1, 0..D_MODEL]);

        // upscale, fusing the high-resolution features in between
        let src = keys.swap_dims(1, 2).reshape([b, c, h, w]);
        let up = gelu(
            self.up_norm
                .apply_2d(self.up0.apply(src) + high_res[1].clone()),
        );
        let up = gelu(self.up1.apply(up) + high_res[0].clone());
        let [_, uc, uh, uw] = up.dims();

        // one filter per mask token, applied as a matrix product
        let filters: Vec<Tensor<B, 3>> = (0..NUM_MASK_TOKENS)
            .map(|i| {
                self.hypernetworks[i].apply(mask_tokens_out.clone().slice([
                    0..b,
                    i..i + 1,
                    0..D_MODEL,
                ]))
            })
            .collect();
        let hyper = Tensor::cat(filters, 1);
        let masks =
            hyper
                .matmul(up.reshape([b, uc, uh * uw]))
                .reshape([b, NUM_MASK_TOKENS, uh, uw]);

        Decoded {
            masks,
            ious: self
                .iou_head
                .apply(iou_token_out)
                .reshape([b, NUM_MASK_TOKENS]),
            mask_tokens: mask_tokens_out,
            object_score_logits: self.obj_score_head.apply(obj_token_out).reshape([b, 1]),
        }
    }

    /// `MaskDecoder.forward`'s output selection.
    ///
    /// With `multimask` the three multi-mask outputs are kept and the caller
    /// picks by IoU; without it the single-mask output is used unless it is
    /// unstable, in which case the best multi-mask output replaces it.
    pub fn select(&self, d: Decoded<B>, multimask: bool) -> Selected<B> {
        let b = d.masks.dims()[0];
        let [_, _, h, w] = d.masks.dims();
        if multimask {
            return Selected {
                masks: d.masks.slice([0..b, 1..NUM_MASK_TOKENS, 0..h, 0..w]),
                ious: d.ious.slice([0..b, 1..NUM_MASK_TOKENS]),
                sam_tokens: d.mask_tokens.slice([0..b, 1..NUM_MASK_TOKENS, 0..D_MODEL]),
                object_score_logits: d.object_score_logits,
            };
        }

        let single = d.masks.clone().slice([0..b, 0..1, 0..h, 0..w]);
        let single_iou = d.ious.clone().slice([0..b, 0..1]);
        let stable = stability_score(single.clone()) >= config::STABILITY_THRESH;
        let (masks, ious) = if stable {
            (single, single_iou)
        } else {
            let best = 1 + argmax_iou(d.ious.clone().slice([0..b, 1..NUM_MASK_TOKENS]));
            (
                d.masks.slice([0..b, best..best + 1, 0..h, 0..w]),
                d.ious.slice([0..b, best..best + 1]),
            )
        };
        Selected {
            masks,
            ious,
            // always the single-mask token when not doing multi-mask output
            sam_tokens: d.mask_tokens.slice([0..b, 0..1, 0..D_MODEL]),
            object_score_logits: d.object_score_logits,
        }
    }
}

/// `_get_stability_scores` for a single `[1, 1, h, w]` logit map: the ratio
/// between the areas above `+delta` and above `-delta`. A mask whose boundary
/// is a knife edge scores near 1; one that drifts across the threshold over a
/// wide band scores low, and the decoder then prefers a multi-mask output.
pub fn stability_score<B: Backend>(logits: Tensor<B, 4>) -> f32 {
    let delta = config::STABILITY_DELTA;
    let count = |t: Tensor<B, 4>, above: f32| -> f32 {
        ops::to_vec(t.greater_elem(above).float().sum())[0]
    };
    let area_i = count(logits.clone(), delta);
    let area_u = count(logits, -delta);
    if area_u > 0.0 {
        area_i / area_u
    } else {
        1.0
    }
}

/// Index of the largest predicted IoU.
pub fn argmax_iou<B: Backend>(ious: Tensor<B, 2>) -> usize {
    let v = ops::to_vec(ious);
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}
