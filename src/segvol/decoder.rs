//! The mask decoder: SAM's two-way transformer and upscaling path, in 3-D.
//!
//! Prompt tokens and image tokens attend to each other for two rounds, then a
//! per-mask "hypernetwork" MLP turns each mask token into a 96-wide filter
//! that is dotted against the upscaled feature volume to produce logits.
//!
//! Details that are easy to get wrong, all of them confirmed against the
//! published checkpoint:
//!
//! * the MLPs here use **ReLU**, not the GELU of the image encoder; the
//!   upscaling path uses GELU. Three activations coexist in this network;
//! * `output_upscaling.1` is a `LayerNorm` over the whole `(192,16,32,32)`
//!   activation — 3.1 M affine values in each of weight and bias, a fifth of
//!   the decoder's parameters, and a channel-wise norm in its place compiles
//!   and produces nonsense;
//! * the first layer skips the query positional encoding entirely and does
//!   **not** add a residual around its self-attention;
//! * text enters twice — once as a prompt token, and again here as an
//!   additive similarity map;
//! * inference keeps mask channel 0 of the four.

use anyhow::{Context, Result};

use crate::nn::attention::{attention, Mask};
use crate::nn::linalg::{gelu, layer_norm, linear, matmul, relu, LAYER_NORM_EPS};
use crate::nn::tensor::{conv_transpose3d_2x, Act, Mat};

use super::config::*;
use crate::nn::params::Params;

/// A `q`/`k`/`v`/`out` projection group running at `internal` width.
struct Attn {
    q: (Vec<f32>, Vec<f32>),
    k: (Vec<f32>, Vec<f32>),
    v: (Vec<f32>, Vec<f32>),
    out: (Vec<f32>, Vec<f32>),
    internal: usize,
}

impl Attn {
    fn build(p: &Params, prefix: &str, downsample: usize) -> Result<Attn> {
        let internal = EMBED / downsample;
        let proj = |name: &str, out: usize, inp: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            let (w, b) = p.linear_opt(&format!("{prefix}.{name}"), out, inp)?;
            Ok((
                w.to_vec(),
                b.with_context(|| format!("{prefix}.{name} needs a bias"))?
                    .to_vec(),
            ))
        };
        Ok(Attn {
            q: proj("q_proj", internal, EMBED)?,
            k: proj("k_proj", internal, EMBED)?,
            v: proj("v_proj", internal, EMBED)?,
            out: proj("out_proj", EMBED, internal)?,
            internal,
        })
    }

    fn forward(&self, q: &Mat, k: &Mat, v: &Mat) -> Mat {
        let qp = linear(q, &self.q.0, self.internal, Some(&self.q.1));
        let kp = linear(k, &self.k.0, self.internal, Some(&self.k.1));
        let vp = linear(v, &self.v.0, self.internal, Some(&self.v.1));
        let a = attention(&qp, &kp, &vp, DEC_HEADS, Mask::None);
        linear(&a, &self.out.0, EMBED, Some(&self.out.1))
    }
}

/// A multi-layer perceptron with ReLU between layers and none at the end.
struct Mlp {
    layers: Vec<(Vec<f32>, Vec<f32>, usize)>,
}

impl Mlp {
    fn build(p: &Params, prefix: &str, dims: &[usize]) -> Result<Mlp> {
        let mut layers = Vec::new();
        for (i, pair) in dims.windows(2).enumerate() {
            let (w, b) = p.linear_opt(&format!("{prefix}.layers.{i}"), pair[1], pair[0])?;
            layers.push((
                w.to_vec(),
                b.with_context(|| format!("{prefix}.layers.{i} needs a bias"))?
                    .to_vec(),
                pair[1],
            ));
        }
        Ok(Mlp { layers })
    }

    fn forward(&self, x: &Mat) -> Mat {
        let mut h = x.clone();
        let last = self.layers.len() - 1;
        for (i, (w, b, out)) in self.layers.iter().enumerate() {
            h = linear(&h, w, *out, Some(b));
            if i != last {
                relu(&mut h.data);
            }
        }
        h
    }
}

struct Layer {
    self_attn: Attn,
    norm1: (Vec<f32>, Vec<f32>),
    cross_token_to_image: Attn,
    norm2: (Vec<f32>, Vec<f32>),
    mlp_lin1: (Vec<f32>, Vec<f32>),
    mlp_lin2: (Vec<f32>, Vec<f32>),
    norm3: (Vec<f32>, Vec<f32>),
    norm4: (Vec<f32>, Vec<f32>),
    cross_image_to_token: Attn,
    skip_first_layer_pe: bool,
}

fn norm_of(p: &Params, prefix: &str) -> Result<(Vec<f32>, Vec<f32>)> {
    let (w, b) = p.norm(prefix, EMBED)?;
    Ok((w.to_vec(), b.to_vec()))
}

fn ln(x: &mut Mat, n: &(Vec<f32>, Vec<f32>)) {
    layer_norm(&mut x.data, EMBED, &n.0, &n.1, LAYER_NORM_EPS);
}

fn add(a: &Mat, b: &Mat) -> Mat {
    let mut out = a.clone();
    out.add_assign(b);
    out
}

impl Layer {
    fn build(p: &Params, i: usize) -> Result<Layer> {
        let b = format!("mask_decoder.transformer.layers.{i}");
        Ok(Layer {
            self_attn: Attn::build(p, &format!("{b}.self_attn"), 1)?,
            norm1: norm_of(p, &format!("{b}.norm1"))?,
            cross_token_to_image: Attn::build(
                p,
                &format!("{b}.cross_attn_token_to_image"),
                DEC_ATTN_DOWNSAMPLE,
            )?,
            norm2: norm_of(p, &format!("{b}.norm2"))?,
            mlp_lin1: {
                let (w, bi) = p.linear_opt(&format!("{b}.mlp.lin1"), DEC_MLP, EMBED)?;
                (w.to_vec(), bi.context("mlp.lin1 needs a bias")?.to_vec())
            },
            mlp_lin2: {
                let (w, bi) = p.linear_opt(&format!("{b}.mlp.lin2"), EMBED, DEC_MLP)?;
                (w.to_vec(), bi.context("mlp.lin2 needs a bias")?.to_vec())
            },
            norm3: norm_of(p, &format!("{b}.norm3"))?,
            norm4: norm_of(p, &format!("{b}.norm4"))?,
            cross_image_to_token: Attn::build(
                p,
                &format!("{b}.cross_attn_image_to_token"),
                DEC_ATTN_DOWNSAMPLE,
            )?,
            skip_first_layer_pe: i == 0,
        })
    }

    fn forward(&self, queries: Mat, keys: Mat, query_pe: &Mat, key_pe: &Mat) -> (Mat, Mat) {
        // Self-attention over the prompt tokens. On the first layer the
        // positional encoding is skipped and the result *replaces* the
        // queries — there is no residual here, unlike every later layer.
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(&queries, &queries, &queries)
        } else {
            let q = add(&queries, query_pe);
            let a = self.self_attn.forward(&q, &q, &queries);
            add(&queries, &a)
        };
        ln(&mut queries, &self.norm1);

        // Tokens attend to the image.
        let q = add(&queries, query_pe);
        let k = add(&keys, key_pe);
        let a = self.cross_token_to_image.forward(&q, &k, &keys);
        let mut queries = add(&queries, &a);
        ln(&mut queries, &self.norm2);

        // Token MLP, ReLU.
        let mut h = linear(&queries, &self.mlp_lin1.0, DEC_MLP, Some(&self.mlp_lin1.1));
        relu(&mut h.data);
        let h = linear(&h, &self.mlp_lin2.0, EMBED, Some(&self.mlp_lin2.1));
        let mut queries = add(&queries, &h);
        ln(&mut queries, &self.norm3);

        // The image attends back to the tokens: note q and k are swapped.
        let q = add(&queries, query_pe);
        let k = add(&keys, key_pe);
        let a = self.cross_image_to_token.forward(&k, &q, &queries);
        let mut keys = add(&keys, &a);
        ln(&mut keys, &self.norm4);

        (queries, keys)
    }
}

/// Output of one decoder pass.
pub struct Decoded {
    /// Mask logits, `[NUM_MASK_TOKENS, MASK_SHAPE]`.
    pub masks: Act,
    /// Predicted IoU, one per mask token.
    pub iou: Vec<f32>,
}

impl Decoded {
    /// The single mask inference uses: channel 0 of the four.
    pub fn best(&self) -> Act {
        let sp = self.masks.spatial();
        Act {
            c: 1,
            d: self.masks.d,
            h: self.masks.h,
            w: self.masks.w,
            data: self.masks.data[..sp].to_vec(),
        }
    }
}

pub struct MaskDecoder {
    iou_token: Vec<f32>,
    mask_tokens: Mat,
    layers: Vec<Layer>,
    final_attn: Attn,
    norm_final: (Vec<f32>, Vec<f32>),
    up0_w: Vec<f32>,
    up0_b: Vec<f32>,
    up1_w: Vec<f32>,
    up1_b: Vec<f32>,
    up3_w: Vec<f32>,
    up3_b: Vec<f32>,
    hyper: Vec<Mlp>,
    iou_head: Mlp,
    txt_align: (Vec<f32>, Vec<f32>),
}

impl MaskDecoder {
    pub fn build(p: &Params) -> Result<MaskDecoder> {
        let up_shape = &[EMBED / 4, FEAT_SHAPE[0], FEAT_SHAPE[1], FEAT_SHAPE[2]][..];
        let mut layers = Vec::with_capacity(DEC_LAYERS);
        for i in 0..DEC_LAYERS {
            layers.push(Layer::build(p, i)?);
        }
        let mut hyper = Vec::with_capacity(NUM_MASK_TOKENS);
        for i in 0..NUM_MASK_TOKENS {
            hyper.push(Mlp::build(
                p,
                &format!("mask_decoder.output_hypernetworks_mlps.{i}"),
                &[EMBED, EMBED, EMBED, UPSCALED_CHANNELS],
            )?);
        }
        let (txt_w, txt_b) = p.linear_opt(
            "mask_decoder.txt_align_upscaled_embedding",
            UPSCALED_CHANNELS,
            EMBED,
        )?;
        Ok(MaskDecoder {
            iou_token: p
                .get("mask_decoder.iou_token.weight", &[1, EMBED])?
                .to_vec(),
            mask_tokens: Mat::from_vec(
                NUM_MASK_TOKENS,
                EMBED,
                p.get("mask_decoder.mask_tokens.weight", &[NUM_MASK_TOKENS, EMBED])?
                    .to_vec(),
            ),
            layers,
            final_attn: Attn::build(
                p,
                "mask_decoder.transformer.final_attn_token_to_image",
                DEC_ATTN_DOWNSAMPLE,
            )?,
            norm_final: norm_of(p, "mask_decoder.transformer.norm_final_attn")?,
            up0_w: p
                .get(
                    "mask_decoder.output_upscaling.0.weight",
                    &[EMBED, EMBED / 4, 2, 2, 2],
                )?
                .to_vec(),
            up0_b: p
                .vec("mask_decoder.output_upscaling.0.bias", EMBED / 4)?
                .to_vec(),
            up1_w: p
                .get("mask_decoder.output_upscaling.1.weight", up_shape)?
                .to_vec(),
            up1_b: p
                .get("mask_decoder.output_upscaling.1.bias", up_shape)?
                .to_vec(),
            up3_w: p
                .get(
                    "mask_decoder.output_upscaling.3.weight",
                    &[EMBED / 4, UPSCALED_CHANNELS, 2, 2, 2],
                )?
                .to_vec(),
            up3_b: p
                .vec("mask_decoder.output_upscaling.3.bias", UPSCALED_CHANNELS)?
                .to_vec(),
            hyper,
            iou_head: Mlp::build(
                p,
                "mask_decoder.iou_prediction_head",
                &[EMBED, 256, 256, NUM_MASK_TOKENS],
            )?,
            txt_align: (
                txt_w.to_vec(),
                txt_b.context("txt_align needs a bias")?.to_vec(),
            ),
        })
    }

    /// Run the decoder.
    ///
    /// `image` is `[TOKENS, EMBED]` from the image encoder, `image_pe` the
    /// dense positional encoding, `sparse` the prompt tokens, `dense` the
    /// no-mask embedding, and `text` the aligned text vector if there is one.
    pub fn forward(
        &self,
        image: &Mat,
        image_pe: &Mat,
        sparse: &Mat,
        dense: &Act,
        text: Option<&[f32]>,
    ) -> Decoded {
        // Output tokens first, then the prompt tokens.
        let tokens = Mat::row_vec(&self.iou_token)
            .vcat(&self.mask_tokens)
            .vcat(sparse);

        // The dense prompt is added to the image embedding, not concatenated.
        let mut src = image.clone();
        let dense_tokens = dense_as_tokens(dense);
        src.add_assign(&dense_tokens);

        let (hs, src) = self.transformer(src, image_pe, &tokens);
        let iou_token_out = Mat::row_vec(hs.row(0));
        let mask_tokens_out = hs.rows_slice(1, 1 + NUM_MASK_TOKENS);

        // [TOKENS, EMBED] -> [EMBED, GRID] -> upscale
        let volume = Act::from_tokens(&src, GRID[0], GRID[1], GRID[2]);
        let upscaled = self.upscale(&volume);
        let up_dims = [upscaled.d, upscaled.h, upscaled.w];
        // [UPSCALED_CHANNELS, spatial], the same storage — 50 MB per window
        // not copied.
        let up_mat = upscaled.into_mat();

        // One 96-wide filter per mask token.
        let mut hyper_in = Mat::zeros(NUM_MASK_TOKENS, UPSCALED_CHANNELS);
        for i in 0..NUM_MASK_TOKENS {
            let row = self.hyper[i].forward(&Mat::row_vec(mask_tokens_out.row(i)));
            hyper_in.row_mut(i).copy_from_slice(&row.data);
        }
        let mut masks = matmul(&hyper_in, &up_mat); // [NUM_MASK_TOKENS, spatial]

        // Text enters a second time, as an additive similarity map shared by
        // every mask channel.
        if let Some(t) = text {
            let td = linear(
                &Mat::row_vec(t),
                &self.txt_align.0,
                UPSCALED_CHANNELS,
                Some(&self.txt_align.1),
            );
            let sim = matmul(&td, &up_mat); // [1, spatial]
            for r in 0..masks.rows {
                for (v, s) in masks.row_mut(r).iter_mut().zip(sim.data.iter()) {
                    *v += s;
                }
            }
        }

        let iou = self.iou_head.forward(&iou_token_out).data;
        Decoded {
            masks: Act {
                c: NUM_MASK_TOKENS,
                d: up_dims[0],
                h: up_dims[1],
                w: up_dims[2],
                data: masks.data,
            },
            iou,
        }
    }

    fn transformer(&self, image: Mat, image_pe: &Mat, tokens: &Mat) -> (Mat, Mat) {
        let mut queries = tokens.clone();
        let mut keys = image;
        for layer in &self.layers {
            let (q, k) = layer.forward(queries, keys, tokens, image_pe);
            queries = q;
            keys = k;
        }
        let q = add(&queries, tokens);
        let k = add(&keys, image_pe);
        let a = self.final_attn.forward(&q, &k, &keys);
        let mut queries = add(&queries, &a);
        ln(&mut queries, &self.norm_final);
        (queries, keys)
    }

    /// `ConvTranspose3d -> LayerNorm(C,D,H,W) -> GELU -> ConvTranspose3d ->
    /// GELU`, taking `[EMBED, GRID]` to `[UPSCALED_CHANNELS, MASK_SHAPE]`.
    fn upscale(&self, x: &Act) -> Act {
        let mut h = conv_transpose3d_2x(x, &self.up0_w, &self.up0_b, EMBED / 4);
        debug_assert_eq!([h.d, h.h, h.w], FEAT_SHAPE);
        // One group spanning the entire activation — not a per-channel norm.
        let whole = h.data.len();
        layer_norm(&mut h.data, whole, &self.up1_w, &self.up1_b, LAYER_NORM_EPS);
        gelu(&mut h.data);
        let mut h = conv_transpose3d_2x(&h, &self.up3_w, &self.up3_b, UPSCALED_CHANNELS);
        gelu(&mut h.data);
        debug_assert_eq!([h.d, h.h, h.w], MASK_SHAPE);
        h
    }
}

/// `[EMBED, d, h, w]` -> `[d*h*w, EMBED]`, the transpose `Act::from_tokens`
/// undoes.
fn dense_as_tokens(dense: &Act) -> Mat {
    let sp = dense.spatial();
    let mut m = Mat::zeros(sp, dense.c);
    for c in 0..dense.c {
        for t in 0..sp {
            m.data[t * dense.c + c] = dense.data[c * sp + t];
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_tokens_round_trip_through_the_volume_layout() {
        let mut a = Act::zeros(3, 1, 2, 2);
        for (i, v) in a.data.iter_mut().enumerate() {
            *v = i as f32;
        }
        let m = dense_as_tokens(&a);
        assert_eq!((m.rows, m.cols), (4, 3));
        let back = Act::from_tokens(&m, 1, 2, 2);
        assert_eq!(back.data, a.data);
    }

    #[test]
    fn mlp_applies_relu_between_layers_and_not_after() {
        // Two layers, both identity-ish, chosen so an extra trailing ReLU
        // would clip a negative output.
        let mlp = Mlp {
            layers: vec![
                (vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2),
                (vec![-1.0, 0.0], vec![0.0], 1),
            ],
        };
        let out = mlp.forward(&Mat::from_vec(1, 2, vec![2.0, 3.0]));
        assert_eq!(out.data, vec![-2.0], "the last layer must not be rectified");
        // and the intermediate ReLU does fire
        let out = mlp.forward(&Mat::from_vec(1, 2, vec![-2.0, 3.0]));
        assert_eq!(out.data, vec![0.0]);
    }

    #[test]
    fn best_takes_mask_channel_zero() {
        let mut masks = Act::zeros(NUM_MASK_TOKENS, 1, 1, 2);
        masks.data = vec![1.0, 2.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let d = Decoded {
            masks,
            iou: vec![0.0; NUM_MASK_TOKENS],
        };
        let b = d.best();
        assert_eq!(b.c, 1);
        assert_eq!(b.data, vec![1.0, 2.0]);
    }
}
