//! VISTA-3D's point mode: a structure from clicks on it (and beside it).
//!
//! The network is the other half of `vista3d132`: the encoder's *point*
//! decoder (`up_layers`) and MONAI's `PointMappingSAM` head, a SAM-style
//! mask decoder in 3-D. The features are halved in resolution (a strided
//! convolution block), the clicks become tokens (a random-Fourier position
//! code plus a learned embedding for positive, negative or padding, and a
//! token saying "supported class"), a two-way transformer lets tokens and
//! image attend to each other twice, the image side is brought back to full
//! resolution (a transposed convolution block) and dotted with the mask
//! token run through a small MLP: one logit per voxel.
//!
//! The pipeline is the bundle's with `use_point_window`
//! (`monai.apps.vista3d.inferer.point_based_window_inferer`): one 128-cubed
//! window centred on each click (shifted inside the image), every click
//! that falls inside a window prompts it, overlapping windows' logits are
//! *summed* (the inferer counts a voxel as covered, not how often), voxels
//! no window covered are undecided; then `VistaPostTransformd` keeps the
//! 26-connected pieces of the positive region that hold a positive click.

use anyhow::{bail, Result};
use burn::tensor::activation::{gelu, relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::medsam2::ops;
use crate::nn::fastconv;
use crate::nn::params::Params;
use crate::segresnet::net::Conv;

/// MONAI's `NINF_VALUE`: the logit of a window no click prompts.
pub const NINF: f32 = -9999.0;

/// A click: a position in the window's voxels and whether it is on the
/// structure (1), beside it (0), or on a special class's (3 / 2); -1 is
/// padding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Click {
    pub at: [f32; 3],
    pub label: i32,
}

struct Linear<B: Backend> {
    w: Tensor<B, 2>,
    b: Tensor<B, 1>,
}

impl<B: Backend> Linear<B> {
    fn load(p: &Params, name: &str, out: usize, inp: usize, dev: &B::Device) -> Result<Self> {
        let (w, b) = p.linear(name, out, inp)?;
        Ok(Linear {
            w: ops::from_slice(w, [out, inp], dev),
            b: ops::from_slice(b, [out], dev),
        })
    }

    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.w.clone().transpose()) + self.b.clone().unsqueeze_dim(0)
    }
}

struct LayerNorm<B: Backend> {
    w: Tensor<B, 1>,
    b: Tensor<B, 1>,
}

impl<B: Backend> LayerNorm<B> {
    fn load(p: &Params, name: &str, n: usize, dev: &B::Device) -> Result<Self> {
        Ok(LayerNorm {
            w: ops::from_slice(p.vec(&format!("{name}.weight"), n)?, [n], dev),
            b: ops::from_slice(p.vec(&format!("{name}.bias"), n)?, [n], dev),
        })
    }

    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mean = x.clone().mean_dim(1);
        let c = x - mean;
        let var = c.clone().powf_scalar(2.0).mean_dim(1);
        c / (var + 1e-5).sqrt() * self.w.clone().unsqueeze_dim(0) + self.b.clone().unsqueeze_dim(0)
    }
}

/// SAM's `Attention`, with the projections narrowed by `downsample`.
struct Attention<B: Backend> {
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    heads: usize,
    inner: usize,
}

impl<B: Backend> Attention<B> {
    fn load(
        p: &Params,
        name: &str,
        dim: usize,
        heads: usize,
        downsample: usize,
        dev: &B::Device,
    ) -> Result<Self> {
        let inner = dim / downsample;
        if !inner.is_multiple_of(heads) {
            bail!("{name}: {heads} heads do not divide {inner}");
        }
        Ok(Attention {
            q: Linear::load(p, &format!("{name}.q_proj"), inner, dim, dev)?,
            k: Linear::load(p, &format!("{name}.k_proj"), inner, dim, dev)?,
            v: Linear::load(p, &format!("{name}.v_proj"), inner, dim, dev)?,
            out: Linear::load(p, &format!("{name}.out_proj"), dim, inner, dev)?,
            heads,
            inner,
        })
    }

    /// `[nq, dim]`, `[nk, dim]`, `[nk, dim]` → `[nq, dim]`.
    fn apply(&self, q: Tensor<B, 2>, k: Tensor<B, 2>, v: Tensor<B, 2>) -> Tensor<B, 2> {
        let (h, c) = (self.heads, self.inner / self.heads);
        let split = |t: Tensor<B, 2>| -> Tensor<B, 3> {
            let n = t.dims()[0];
            t.reshape([n, h, c]).swap_dims(0, 1)
        };
        let nq = q.dims()[0];
        let q = split(self.q.apply(q));
        let k = split(self.k.apply(k));
        let v = split(self.v.apply(v));
        let attn = q.matmul(k.swap_dims(1, 2)) / (c as f64).sqrt();
        let attn = softmax(attn, 2);
        let o = attn.matmul(v).swap_dims(0, 1).reshape([nq, self.inner]);
        self.out.apply(o)
    }
}

/// `TwoWayAttentionBlock`.
struct Block<B: Backend> {
    self_attn: Attention<B>,
    norm1: LayerNorm<B>,
    cross_t2i: Attention<B>,
    norm2: LayerNorm<B>,
    mlp1: Linear<B>,
    mlp2: Linear<B>,
    norm3: LayerNorm<B>,
    norm4: LayerNorm<B>,
    cross_i2t: Attention<B>,
    skip_first_pe: bool,
}

impl<B: Backend> Block<B> {
    fn apply(
        &self,
        queries: Tensor<B, 2>,
        keys: Tensor<B, 2>,
        query_pe: &Tensor<B, 2>,
        key_pe: &Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let queries = if self.skip_first_pe {
            self.self_attn
                .apply(queries.clone(), queries.clone(), queries)
        } else {
            let q = queries.clone() + query_pe.clone();
            queries.clone() + self.self_attn.apply(q.clone(), q, queries)
        };
        let queries = self.norm1.apply(queries);
        let q = queries.clone() + query_pe.clone();
        let k = keys.clone() + key_pe.clone();
        let queries = self
            .norm2
            .apply(queries + self.cross_t2i.apply(q, k, keys.clone()));
        let mlp = self.mlp2.apply(relu(self.mlp1.apply(queries.clone())));
        let queries = self.norm3.apply(queries + mlp);
        let q = queries.clone() + query_pe.clone();
        let k = keys.clone() + key_pe.clone();
        let keys = self
            .norm4
            .apply(keys + self.cross_i2t.apply(k, q, queries.clone()));
        (queries, keys)
    }
}

/// `PointMappingSAM`.
pub struct PointHead<B: Backend> {
    down1: Conv<B>,
    down2: Conv<B>,
    blocks: Vec<Block<B>>,
    final_attn: Attention<B>,
    norm_final: LayerNorm<B>,
    /// The random Fourier matrix, `[3, width / 2]`.
    gauss: Vec<f32>,
    embed_point: [Vec<f32>; 2],
    not_a_point: Vec<f32>,
    special: Vec<f32>,
    mask_token: Vec<f32>,
    supported: Vec<f32>,
    up_w: Tensor<B, 5>,
    up_b: Tensor<B, 1>,
    up_conv: Conv<B>,
    hyper: [Linear<B>; 3],
    width: usize,
    device: B::Device,
}

fn instance_norm<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    let mean = x.clone().mean_dim(2).mean_dim(3).mean_dim(4);
    let c = x - mean;
    let var = c
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .mean_dim(3)
        .mean_dim(4);
    c / (var + 1e-5).sqrt()
}

/// SAM's `PositionEmbeddingRandom` of positions normalized to [0, 1]:
/// `[sin, cos]` of `2 pi (2 c - 1) G`.
fn fourier(gauss: &[f32], half: usize, c: [f32; 3], out: &mut [f32]) {
    let c: [f32; 3] = std::array::from_fn(|a| 2.0 * c[a] - 1.0);
    for f in 0..half {
        let v = c[0] * gauss[f] + c[1] * gauss[half + f] + c[2] * gauss[2 * half + f];
        let v = (2.0 * std::f64::consts::PI) as f32 * v;
        out[f] = v.sin();
        out[half + f] = v.cos();
    }
}

impl<B: Backend> PointHead<B> {
    pub fn load(p: &Params, width: usize, dev: &B::Device) -> Result<Self> {
        let f = width;
        let half = f / 2;
        let block = |i: usize| -> Result<Block<B>> {
            let n = format!("point_head.transformer.layers.{i}");
            let mlp = p
                .shape(&format!("{n}.mlp.linear1.weight"))
                .and_then(|s| s.first().copied())
                .unwrap_or(512);
            Ok(Block {
                self_attn: Attention::load(p, &format!("{n}.self_attn"), f, 4, 1, dev)?,
                norm1: LayerNorm::load(p, &format!("{n}.norm1"), f, dev)?,
                cross_t2i: Attention::load(
                    p,
                    &format!("{n}.cross_attn_token_to_image"),
                    f,
                    4,
                    2,
                    dev,
                )?,
                norm2: LayerNorm::load(p, &format!("{n}.norm2"), f, dev)?,
                mlp1: Linear::load(p, &format!("{n}.mlp.linear1"), mlp, f, dev)?,
                mlp2: Linear::load(p, &format!("{n}.mlp.linear2"), f, mlp, dev)?,
                norm3: LayerNorm::load(p, &format!("{n}.norm3"), f, dev)?,
                norm4: LayerNorm::load(p, &format!("{n}.norm4"), f, dev)?,
                cross_i2t: Attention::load(
                    p,
                    &format!("{n}.cross_attn_image_to_token"),
                    f,
                    4,
                    2,
                    dev,
                )?,
                skip_first_pe: i == 0,
            })
        };
        let emb = |name: &str| -> Result<Vec<f32>> { Ok(p.get(name, &[1, f])?.to_vec()) };
        let up_w = p.get("point_head.output_upscaling.0.weight", &[f, f, 3, 3, 3])?;
        Ok(PointHead {
            down1: Conv::load(p, "point_head.feat_downsample.0", f, f, 3, 2, true, dev)?,
            down2: Conv::load(p, "point_head.feat_downsample.3", f, f, 3, 1, true, dev)?,
            blocks: vec![block(0)?, block(1)?],
            final_attn: Attention::load(
                p,
                "point_head.transformer.final_attn_token_to_image",
                f,
                4,
                2,
                dev,
            )?,
            norm_final: LayerNorm::load(p, "point_head.transformer.norm_final_attn", f, dev)?,
            gauss: p
                .get(
                    "point_head.pe_layer.positional_encoding_gaussian_matrix",
                    &[3, half],
                )?
                .to_vec(),
            embed_point: [
                emb("point_head.point_embeddings.0.weight")?,
                emb("point_head.point_embeddings.1.weight")?,
            ],
            not_a_point: emb("point_head.not_a_point_embed.weight")?,
            special: emb("point_head.special_class_embed.weight")?,
            mask_token: emb("point_head.mask_tokens.weight")?,
            supported: emb("point_head.supported_embed.weight")?,
            up_w: ops::from_slice(up_w, [f, f, 3, 3, 3], dev),
            up_b: ops::from_slice(p.vec("point_head.output_upscaling.0.bias", f)?, [f], dev),
            up_conv: Conv::load(p, "point_head.output_upscaling.3", f, f, 3, 1, true, dev)?,
            hyper: [
                Linear::load(
                    p,
                    "point_head.output_hypernetworks_mlps.layers.0",
                    f,
                    f,
                    dev,
                )?,
                Linear::load(
                    p,
                    "point_head.output_hypernetworks_mlps.layers.1",
                    f,
                    f,
                    dev,
                )?,
                Linear::load(
                    p,
                    "point_head.output_hypernetworks_mlps.layers.2",
                    f,
                    f,
                    dev,
                )?,
            ],
            width: f,
            device: dev.clone(),
        })
    }

    /// The tokens: the mask token, one per click, the supported-class
    /// token. `size` is the full-resolution window the clicks are in.
    fn tokens(&self, clicks: &[Click], size: [usize; 3]) -> Vec<f32> {
        let f = self.width;
        let mut t = self.mask_token.clone();
        for c in clicks {
            let mut e = vec![0f32; f];
            if c.label != -1 {
                let at: [f32; 3] = std::array::from_fn(|a| (c.at[a] + 0.5) / size[a] as f32);
                fourier(&self.gauss, f / 2, at, &mut e);
            }
            let add = |e: &mut Vec<f32>, v: &[f32]| e.iter_mut().zip(v).for_each(|(a, b)| *a += b);
            match c.label {
                -1 => add(&mut e, &self.not_a_point),
                0 => add(&mut e, &self.embed_point[0]),
                1 => add(&mut e, &self.embed_point[1]),
                2 => {
                    add(&mut e, &self.embed_point[0]);
                    add(&mut e, &self.special);
                }
                _ => {
                    add(&mut e, &self.embed_point[1]);
                    add(&mut e, &self.special);
                }
            }
            t.extend_from_slice(&e);
        }
        t.extend_from_slice(&self.supported);
        t
    }

    /// The position code of every voxel of a `[h, w, d]` grid, `[h*w*d, width]`.
    fn grid_pe(&self, g: [usize; 3]) -> Vec<f32> {
        let f = self.width;
        let mut out = vec![0f32; g[0] * g[1] * g[2] * f];
        for i in 0..g[0] {
            for j in 0..g[1] {
                for k in 0..g[2] {
                    let c = [
                        (i as f32 + 0.5) / g[0] as f32,
                        (j as f32 + 0.5) / g[1] as f32,
                        (k as f32 + 0.5) / g[2] as f32,
                    ];
                    let n = (i * g[1] + j) * g[2] + k;
                    fourier(&self.gauss, f / 2, c, &mut out[n * f..(n + 1) * f]);
                }
            }
        }
        out
    }

    /// Logits `[h, w, d]` for features `[1, width, h, w, d]` (the point
    /// decoder's) and clicks in the same window.
    pub fn apply(&self, features: Tensor<B, 5>, clicks: &[Click]) -> Tensor<B, 1> {
        let f = self.width;
        let size = {
            let d = features.dims();
            [d[2], d[3], d[4]]
        };
        let low = instance_norm(
            self.down2
                .apply(gelu(instance_norm(self.down1.apply(features)))),
        );
        let [_, _, h, w, d] = low.dims();
        let n = h * w * d;
        let keys = low.reshape([f, n]).transpose();
        let key_pe = ops::from_slice::<B, 2>(&self.grid_pe([h, w, d]), [n, f], &self.device);
        let tokens = self.tokens(clicks, size);
        let nt = tokens.len() / f;
        let pe = ops::from_slice::<B, 2>(&tokens, [nt, f], &self.device);
        let (mut queries, mut keys) = (pe.clone(), keys);
        for b in &self.blocks {
            let (q, k) = b.apply(queries, keys, &pe, &key_pe);
            queries = q;
            keys = k;
        }
        let q = queries.clone() + pe.clone();
        let k = keys.clone() + key_pe;
        let queries = self
            .norm_final
            .apply(queries + self.final_attn.apply(q, k, keys.clone()));
        let token = queries.slice([0..1, 0..f]);
        let hyper =
            self.hyper[2].apply(relu(self.hyper[1].apply(relu(self.hyper[0].apply(token)))));
        let src = keys.transpose().reshape([1, f, h, w, d]);
        let up = fastconv::conv_transpose3d_k3s2(src, &self.up_w, Some(&self.up_b));
        let up = self.up_conv.apply(gelu(instance_norm(up)));
        let full = size[0] * size[1] * size[2];
        hyper.matmul(up.reshape([f, full])).reshape([full])
    }
}

// ---------------------------------------------------------- the pipeline --

/// `_get_window_idx_c`: the window of `roi` voxels around click position
/// `p` on an axis of `s`, shifted inside.
fn window_at(p: f32, roi: usize, s: usize) -> (usize, usize) {
    let half = (roi / 2) as f32;
    if p - half < 0.0 {
        (0, roi)
    } else if p + half > s as f32 {
        (s - roi, s)
    } else {
        let c = p as usize;
        (c - roi / 2, c + roi / 2)
    }
}

/// What [`stitch`] runs per window: its lower corner, the padded image
/// under it and the clicks inside it, to the window's logits.
pub type WindowFn<'a> = dyn FnMut([usize; 3], &[f32], &[Click]) -> Result<Vec<f32>> + 'a;

/// `point_based_window_inferer` (centre-only windows, no previous mask)
/// over a prepared volume `data` (`dims`, C-order): `forward` runs the
/// network on one `roi` window with the clicks inside it, in the window's
/// coordinates. Returns logits on `dims`, NaN where no window reached.
///
/// `forward` also gets the window's lower corner in the padded volume, so a
/// caller can keep the answers of windows whose clicks did not change.
pub fn stitch(
    data: &[f32],
    dims: [usize; 3],
    roi: [usize; 3],
    clicks: &[Click],
    forward: &mut WindowFn,
) -> Result<Vec<f32>> {
    // Pad to at least the window, half before, the rest after.
    let pad: [usize; 3] = std::array::from_fn(|a| roi[a].saturating_sub(dims[a]) / 2);
    let pd: [usize; 3] = std::array::from_fn(|a| dims[a].max(roi[a]));
    let mut padded = vec![0f32; pd[0] * pd[1] * pd[2]];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let s = (i * dims[1] + j) * dims[2];
            let d = ((i + pad[0]) * pd[1] + j + pad[1]) * pd[2] + pad[2];
            padded[d..d + dims[2]].copy_from_slice(&data[s..s + dims[2]]);
        }
    }
    let shifted: Vec<Click> = clicks
        .iter()
        .map(|c| Click {
            at: std::array::from_fn(|a| c.at[a] + pad[a] as f32),
            label: c.label,
        })
        .collect();
    let mut sum = vec![0f32; padded.len()];
    let mut seen = vec![false; padded.len()];
    let n = roi[0] * roi[1] * roi[2];
    for c in &shifted {
        let w: [(usize, usize); 3] = std::array::from_fn(|a| window_at(c.at[a], roi[a], pd[a]));
        let mut patch = Vec::with_capacity(n);
        for i in w[0].0..w[0].1 {
            for j in w[1].0..w[1].1 {
                let s = (i * pd[1] + j) * pd[2] + w[2].0;
                patch.extend_from_slice(&padded[s..s + roi[2]]);
            }
        }
        // `update_point_to_patch`: the clicks strictly inside, moved into
        // the window.
        let inside: Vec<Click> = shifted
            .iter()
            .filter(|q| {
                (0..3).all(|a| q.at[a] - w[a].0 as f32 > 0.0 && (w[a].1 as f32) - q.at[a] > 0.0)
            })
            .map(|q| Click {
                at: std::array::from_fn(|a| q.at[a] - w[a].0 as f32),
                label: q.label,
            })
            .filter(|q| q.label != -1)
            .collect();
        let out = if inside.is_empty() {
            vec![NINF; n]
        } else {
            forward([w[0].0, w[1].0, w[2].0], &patch, &inside)?
        };
        for (pi, i) in (w[0].0..w[0].1).enumerate() {
            for (pj, j) in (w[1].0..w[1].1).enumerate() {
                let s = (pi * roi[1] + pj) * roi[2];
                let d = (i * pd[1] + j) * pd[2] + w[2].0;
                for k in 0..roi[2] {
                    sum[d + k] += out[s + k];
                    seen[d + k] = true;
                }
            }
        }
    }
    let mut logits = vec![f32::NAN; dims[0] * dims[1] * dims[2]];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let p = ((i + pad[0]) * pd[1] + j + pad[1]) * pd[2] + k + pad[2];
                if seen[p] {
                    logits[(i * dims[1] + j) * dims[2] + k] = sum[p];
                }
            }
        }
    }
    Ok(logits)
}

/// `keep_components_with_positive_points` and the threshold after it: the
/// 26-connected pieces of the positive logits that hold a positive click
/// (rounded half to even, as Python's `round`) stay; the other positive
/// voxels take the mean of everything else that was computed, which in
/// practice drops them. With no positive click nothing is filtered.
pub fn keep_with_positive(logits: &[f32], dims: [usize; 3], clicks: &[Click]) -> Vec<u8> {
    let positive: Vec<u8> = logits.iter().map(|&v| u8::from(v > 0.0)).collect();
    let pos_clicks: Vec<&Click> = clicks
        .iter()
        .filter(|c| c.label == 1 || c.label == 3)
        .collect();
    if pos_clicks.is_empty() {
        return positive;
    }
    let (labels, _) = crate::unet2d::post::label26(&positive, dims);
    let mut keep: Vec<u32> = Vec::new();
    for c in pos_clicks {
        let v: [i64; 3] = std::array::from_fn(|a| f64::from(c.at[a]).round_ties_even() as i64);
        if (0..3).any(|a| v[a] < 0 || v[a] >= dims[a] as i64) {
            continue;
        }
        let id = labels[(v[0] as usize * dims[1] + v[1] as usize) * dims[2] + v[2] as usize];
        if id != 0 && !keep.contains(&id) {
            keep.push(id);
        }
    }
    let kept: Vec<bool> = labels.iter().map(|l| *l != 0 && keep.contains(l)).collect();
    let (mut s, mut n) = (0f64, 0usize);
    for (v, k) in logits.iter().zip(&kept) {
        if !k && !v.is_nan() {
            s += f64::from(*v);
            n += 1;
        }
    }
    let fill = if n > 0 {
        (s / n as f64) as f32
    } else {
        f32::NAN
    };
    positive
        .iter()
        .zip(&kept)
        .map(|(&p, &k)| u8::from(k || (p != 0 && fill > 0.0)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::load_safetensors;
    use crate::segresnet::net::{NormKind, SegResNetDs};

    type B = burn::backend::NdArray;

    pub(crate) fn fixture() -> Params {
        let all = load_safetensors(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/vista-points.safetensors"
        )))
        .expect("tests/data/vista-points.safetensors");
        Params::new(
            all.into_iter()
                .map(|(k, v)| (k.strip_prefix("network.").unwrap_or(&k).to_string(), v))
                .collect(),
        )
    }

    #[test]
    fn the_point_head_matches_monai() {
        let p = fixture();
        let dev = Default::default();
        let enc = SegResNetDs::<B>::load(
            &p,
            "image_encoder.",
            8,
            &[1, 1, 1],
            8,
            NormKind::Instance,
            Some("up_layers"),
            None,
            true,
            &dev,
        )
        .unwrap();
        let head = PointHead::<B>::load(&p, 8, &dev).unwrap();
        let x = p.get("win.input", &[1, 1, 32, 32, 32]).unwrap();
        let feats = enc.forward_point(ops::from_slice(x, [1, 1, 32, 32, 32], &dev));
        let want_f = p.get("win.features", &[1, 8, 32, 32, 32]).unwrap();
        let got_f = ops::to_vec(feats.clone());
        // the reference features are stored in float16
        let wf = got_f
            .iter()
            .zip(want_f)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0, f32::max);
        assert!(wf < 2e-3, "features: relative error {wf:e}");
        let coords = p.get("win.coords", &[1, 3, 3]).unwrap();
        let labels = p.get("win.labels", &[1, 3]).unwrap();
        let clicks: Vec<Click> = (0..3)
            .map(|i| Click {
                at: [coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]],
                label: labels[i] as i32,
            })
            .collect();
        let got = ops::to_vec(head.apply(feats, &clicks));
        let want = p.get("win.output", &[1, 1, 32, 32, 32]).unwrap();
        let w = crate::unet2d::net::tests::worst(&got, want);
        assert!(w < 1e-3, "logits: relative error {w:e}");
        let agree = got
            .iter()
            .zip(want)
            .filter(|(a, b)| (**a > 0.0) == (**b > 0.0))
            .count();
        assert!(
            agree as f64 >= 0.9995 * want.len() as f64,
            "{agree} of {}",
            want.len()
        );
    }

    #[test]
    fn the_point_pipeline_matches_the_bundle() {
        let p = fixture();
        let dev = Default::default();
        let enc = SegResNetDs::<B>::load(
            &p,
            "image_encoder.",
            8,
            &[1, 1, 1],
            8,
            NormKind::Instance,
            Some("up_layers"),
            None,
            true,
            &dev,
        )
        .unwrap();
        let head = PointHead::<B>::load(&p, 8, &dev).unwrap();
        let dims = [40, 28, 44];
        let img = p.get("pipe.image", &[1, 1, 40, 28, 44]).unwrap();
        let coords = p.get("pipe.coords", &[1, 3, 3]).unwrap();
        let labels = p.get("pipe.labels", &[1, 3]).unwrap();
        let clicks: Vec<Click> = (0..3)
            .map(|i| Click {
                at: [coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]],
                label: labels[i] as i32,
            })
            .collect();
        let mut windows = 0;
        let logits = stitch(img, dims, [32; 3], &clicks, &mut |_, x, c| {
            windows += 1;
            let feats = enc.forward_point(ops::from_slice(x, [1, 1, 32, 32, 32], &dev));
            Ok(ops::to_vec(head.apply(feats, c)))
        })
        .unwrap();
        assert_eq!(windows, 3);
        let want = p.get("pipe.logits", &[1, 1, 40, 28, 44]).unwrap();
        for (g, w) in logits.iter().zip(want) {
            assert_eq!(g.is_nan(), w.is_nan());
            if !w.is_nan() {
                assert!((g - w).abs() <= 1e-3 * (1.0 + w.abs()), "{g} vs {w}");
            }
        }
        let mask = keep_with_positive(&logits, dims, &clicks);
        let want_mask = safetensors_u8("pipe.mask");
        let differ = mask.iter().zip(&want_mask).filter(|(a, b)| a != b).count();
        assert!(differ <= 2, "{differ} voxels differ");
    }

    fn safetensors_u8(name: &str) -> Vec<u8> {
        let bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/vista-points.safetensors"
        ))
        .unwrap();
        let t = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        t.tensor(name).unwrap().data().to_vec()
    }

    #[test]
    fn windows_shift_inside_the_image() {
        assert_eq!(window_at(10.0, 32, 100), (0, 32));
        assert_eq!(window_at(90.0, 32, 100), (68, 100));
        assert_eq!(window_at(50.7, 32, 100), (34, 66));
    }

    #[test]
    fn only_pieces_with_a_positive_click_survive() {
        let dims = [1, 1, 9];
        let logits = [1.0, 1.0, -5.0, 2.0, 2.0, -5.0, -5.0, 3.0, f32::NAN];
        let clicks = [Click {
            at: [0.0, 0.0, 3.4],
            label: 1,
        }];
        assert_eq!(
            keep_with_positive(&logits, dims, &clicks),
            vec![0, 0, 0, 1, 1, 0, 0, 0, 0]
        );
        // no positive click: nothing is filtered
        let neg = [Click {
            at: [0.0, 0.0, 3.0],
            label: 0,
        }];
        assert_eq!(
            keep_with_positive(&logits, dims, &neg),
            vec![1, 1, 0, 1, 1, 0, 0, 1, 0]
        );
    }
}
