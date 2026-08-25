//! GPU image encoder via `burn`'s wgpu backend (Vulkan / DX12 / Metal).
//!
//! Only the image encoder runs here, and that is a deliberate choice rather
//! than an unfinished one. The ViT is ~97% of a window's arithmetic — 2.5e11
//! multiply-accumulates against the decoder's low single-digit billions — so
//! moving it alone captures essentially all of the speedup. Keeping the
//! prompt encoder and mask decoder on the CPU also keeps the trap-dense part
//! of the port (the axis-swapped positional encoding, the `(C,D,H,W)`
//! LayerNorm, the two text-injection paths) in exactly one implementation,
//! which is worth more than the milliseconds a second copy would save.
//!
//! The weights are uploaded once per model; each window is transferred, run,
//! and read back as a `[TOKENS, EMBED]` embedding that the CPU decoder then
//! consumes unchanged.
//!
//! Compiled only with the `gpu` cargo feature (on by default).

use anyhow::{anyhow, bail, Context, Result};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::activation::softmax;
use burn::tensor::{Tensor, TensorData};

use crate::nn::device::{guarded, GpuContext};
use crate::nn::params::Params;
use crate::nn::tensor::Mat;

use super::config::*;
use super::vit::Vit;

type B = Wgpu;

/// One transformer block's weights, resident on the device.
struct GBlock {
    ln1_w: Tensor<B, 3>,
    ln1_b: Tensor<B, 3>,
    qkv_w: Tensor<B, 2>,
    out_w: Tensor<B, 2>,
    out_b: Tensor<B, 3>,
    ln2_w: Tensor<B, 3>,
    ln2_b: Tensor<B, 3>,
    lin1_w: Tensor<B, 2>,
    lin1_b: Tensor<B, 3>,
    lin2_w: Tensor<B, 2>,
    lin2_b: Tensor<B, 3>,
}

/// The image encoder with its weights on the GPU.
pub struct GpuVit {
    device: WgpuDevice,
    patch_w: Tensor<B, 2>,
    patch_b: Tensor<B, 3>,
    pos: Tensor<B, 3>,
    blocks: Vec<GBlock>,
    norm_w: Tensor<B, 3>,
    norm_b: Tensor<B, 3>,
}

/// Upload a `[out, in]` weight already transposed to `[in, out]`, so the
/// forward pass is a plain `x @ w` with no per-call transpose.
fn upload_weight_t(d: &WgpuDevice, w: &[f32], out: usize, inp: usize) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(transpose_weight(w, out, inp), [inp, out]),
        d,
    )
}

/// Upload a length-`n` vector shaped for broadcasting over `[1, tokens, n]`.
fn upload_row(d: &WgpuDevice, v: &[f32]) -> Tensor<B, 3> {
    Tensor::from_data(TensorData::new(v.to_vec(), [1, 1, v.len()]), d)
}

impl GpuVit {
    /// Build from the same checkpoint tensors the CPU encoder reads.
    pub fn new(ctx: &GpuContext, p: &Params) -> Result<GpuVit> {
        let d = ctx.device();
        let pe = "image_encoder.patch_embedding";
        let (patch_w, patch_b) =
            p.linear_opt(&format!("{pe}.patch_embeddings.1"), EMBED, PATCH_FEATURES)?;
        let pos = p.get(&format!("{pe}.position_embeddings"), &[1, TOKENS, EMBED])?;
        let mut blocks = Vec::with_capacity(VIT_BLOCKS);
        for i in 0..VIT_BLOCKS {
            let b = format!("image_encoder.blocks.{i}");
            let (ln1_w, ln1_b) = p.norm(&format!("{b}.norm1"), EMBED)?;
            let (ln2_w, ln2_b) = p.norm(&format!("{b}.norm2"), EMBED)?;
            let (qkv_w, qkv_bias) = p.linear_opt(&format!("{b}.attn.qkv"), 3 * EMBED, EMBED)?;
            if qkv_bias.is_some() {
                bail!("{b}.attn.qkv has a bias; this is not the network this port implements");
            }
            let (out_w, out_b) = p.linear_opt(&format!("{b}.attn.out_proj"), EMBED, EMBED)?;
            let (lin1_w, lin1_b) = p.linear_opt(&format!("{b}.mlp.linear1"), VIT_MLP, EMBED)?;
            let (lin2_w, lin2_b) = p.linear_opt(&format!("{b}.mlp.linear2"), EMBED, VIT_MLP)?;
            blocks.push(GBlock {
                ln1_w: upload_row(d, ln1_w),
                ln1_b: upload_row(d, ln1_b),
                qkv_w: upload_weight_t(d, qkv_w, 3 * EMBED, EMBED),
                out_w: upload_weight_t(d, out_w, EMBED, EMBED),
                out_b: upload_row(d, out_b.context("attn.out_proj needs a bias")?),
                ln2_w: upload_row(d, ln2_w),
                ln2_b: upload_row(d, ln2_b),
                lin1_w: upload_weight_t(d, lin1_w, VIT_MLP, EMBED),
                lin1_b: upload_row(d, lin1_b.context("mlp.linear1 needs a bias")?),
                lin2_w: upload_weight_t(d, lin2_w, EMBED, VIT_MLP),
                lin2_b: upload_row(d, lin2_b.context("mlp.linear2 needs a bias")?),
            });
        }
        let (norm_w, norm_b) = p.norm("image_encoder.norm", EMBED)?;
        Ok(GpuVit {
            device: d.clone(),
            patch_w: upload_weight_t(d, patch_w, EMBED, PATCH_FEATURES),
            patch_b: upload_row(d, patch_b.context("patch embedding needs a bias")?),
            pos: Tensor::from_data(TensorData::new(pos.to_vec(), [1, TOKENS, EMBED]), d),
            blocks,
            norm_w: upload_row(d, norm_w),
            norm_b: upload_row(d, norm_b),
        })
    }

    /// Encode one `ROI`-shaped volume into `[TOKENS, EMBED]`.
    pub fn forward(&self, volume: &[f32]) -> Result<Mat> {
        let run = || -> Result<Mat> {
            // Patchification is a pure gather; doing it on the host avoids a
            // kernel and keeps one definition of the layout.
            let patches = Vit::patchify(volume);
            let x = Tensor::<B, 3>::from_data(
                TensorData::new(patches.data, [1, TOKENS, PATCH_FEATURES]),
                &self.device,
            );
            let mut x = x.matmul(self.patch_w.clone().unsqueeze()) + self.patch_b.clone();
            x = x + self.pos.clone();
            for b in &self.blocks {
                x = self.block(b, x);
            }
            let x = layer_norm(x, &self.norm_w, &self.norm_b);
            let data = x
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow!("GPU readback failed: {e:?}"))?;
            if data.len() != TOKENS * EMBED {
                bail!(
                    "GPU returned {} values, expected {}",
                    data.len(),
                    TOKENS * EMBED
                );
            }
            Ok(Mat::from_vec(TOKENS, EMBED, data))
        };
        guarded(run).context("GPU image encoder")
    }

    fn block(&self, b: &GBlock, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x = x + attn(norm1(x))
        let h = layer_norm(x.clone(), &b.ln1_w, &b.ln1_b);
        let qkv = h.matmul(b.qkv_w.clone().unsqueeze());
        // MONAI packs the 2304 columns as (qkv, head, head_dim) in C order.
        let q = qkv.clone().slice([0..1, 0..TOKENS, 0..EMBED]);
        let k = qkv.clone().slice([0..1, 0..TOKENS, EMBED..2 * EMBED]);
        let v = qkv.slice([0..1, 0..TOKENS, 2 * EMBED..3 * EMBED]);
        let a = attention(q, k, v, VIT_HEADS);
        let a = a.matmul(b.out_w.clone().unsqueeze()) + b.out_b.clone();
        let x = x + a;

        // x = x + mlp(norm2(x))
        let h = layer_norm(x.clone(), &b.ln2_w, &b.ln2_b);
        let h = h.matmul(b.lin1_w.clone().unsqueeze()) + b.lin1_b.clone();
        let h = gelu_erf(h);
        let h = h.matmul(b.lin2_w.clone().unsqueeze()) + b.lin2_b.clone();
        x + h
    }
}

/// LayerNorm over the last dimension, biased variance, eps 1e-5.
fn layer_norm(x: Tensor<B, 3>, w: &Tensor<B, 3>, b: &Tensor<B, 3>) -> Tensor<B, 3> {
    let mean = x.clone().mean_dim(2);
    let centered = x - mean;
    let var = centered.clone().powf_scalar(2.0).mean_dim(2);
    let normed = centered / (var + crate::nn::linalg::LAYER_NORM_EPS as f64).sqrt();
    normed * w.clone() + b.clone()
}

/// Exact (erf-based) GELU, matching `nn.GELU()` — *not* the tanh
/// approximation, which some frameworks use by default and which would
/// diverge from the CPU path.
fn gelu_erf(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let inner = x.clone() * std::f64::consts::FRAC_1_SQRT_2;
    x * (inner.erf() + 1.0) * 0.5
}

/// Multi-head attention over `[1, tokens, embed]` tensors.
///
/// Heads are processed one at a time, as on the CPU, and for a harder reason
/// than working-set size: batching all twelve would need a
/// `heads x tokens x tokens` score buffer — 192 MB in `f32` for the image
/// encoder — and **WebGPU's default `maxStorageBufferBindingSize` is
/// 128 MiB**. A batched implementation therefore fails to allocate on any
/// adapter that only offers the guaranteed limits, which includes software
/// rasterizers and plenty of real hardware. Per head the score buffer is
/// 16.8 MB, comfortably inside the guarantee, and each head is still a large
/// enough matmul to keep the device busy.
fn attention(q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>, heads: usize) -> Tensor<B, 3> {
    let hd = EMBED / heads;
    let scale = (hd as f64).sqrt();
    let mut per_head = Vec::with_capacity(heads);
    for h in 0..heads {
        let cols = h * hd..(h + 1) * hd;
        let qh = q.clone().slice([0..1, 0..TOKENS, cols.clone()]);
        let kh = k.clone().slice([0..1, 0..TOKENS, cols.clone()]);
        let vh = v.clone().slice([0..1, 0..TOKENS, cols]);
        let scores = qh.matmul(kh.swap_dims(1, 2)) / scale;
        per_head.push(softmax(scores, 2).matmul(vh));
    }
    Tensor::cat(per_head, 2)
}

/// Transpose an `[out, in]` weight into `[in, out]`.
fn transpose_weight(w: &[f32], out: usize, inp: usize) -> Vec<f32> {
    let mut t = vec![0f32; out * inp];
    for o in 0..out {
        for i in 0..inp {
            t[i * out + o] = w[o * inp + i];
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_weight_transpose_is_correct() {
        // [out=2, in=3] row-major -> [in=3, out=2] row-major
        let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            transpose_weight(&w, 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        // transposing twice is the identity
        let back = transpose_weight(&transpose_weight(&w, 2, 3), 3, 2);
        assert_eq!(back, w.to_vec());
    }

    /// Agreement with the CPU encoder.
    ///
    /// Ignored by default. It needs a real GPU: `WgpuDevice::default()`
    /// happily returns a *software* adapter wherever Mesa's lavapipe or
    /// Windows' WARP is installed — which is the case on CI runners — and
    /// running twelve transformer blocks through a software rasterizer takes
    /// minutes and tests nothing about the backend. Run it where the hardware
    /// is:
    ///
    /// ```text
    /// cargo test --release --lib segvol::gpu -- --ignored
    /// ```
    #[test]
    #[ignore]
    fn gpu_agrees_with_the_cpu_encoder() {
        let Ok(ctx) = GpuContext::try_new() else {
            eprintln!("no GPU available; skipping the CPU/GPU agreement check");
            return;
        };
        // A miniature but correctly shaped encoder: the dimensions are fixed
        // by the checkpoint, so this is the real 87 M-parameter shape.
        let mut m = std::collections::HashMap::new();
        let mut s = 99u64;
        let mut rnd = move |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5) * 0.02
                })
                .collect()
        };
        let put = |m: &mut std::collections::HashMap<String, crate::nn::cache::WTensor>,
                   k: String,
                   shape: Vec<usize>,
                   data: Vec<f32>| {
            m.insert(k, crate::nn::cache::WTensor { shape, data });
        };
        let pe = "image_encoder.patch_embedding";
        put(
            &mut m,
            format!("{pe}.patch_embeddings.1.weight"),
            vec![EMBED, PATCH_FEATURES],
            rnd(EMBED * PATCH_FEATURES),
        );
        put(
            &mut m,
            format!("{pe}.patch_embeddings.1.bias"),
            vec![EMBED],
            rnd(EMBED),
        );
        put(
            &mut m,
            format!("{pe}.position_embeddings"),
            vec![1, TOKENS, EMBED],
            rnd(TOKENS * EMBED),
        );
        for i in 0..VIT_BLOCKS {
            let b = format!("image_encoder.blocks.{i}");
            put(
                &mut m,
                format!("{b}.attn.qkv.weight"),
                vec![3 * EMBED, EMBED],
                rnd(3 * EMBED * EMBED),
            );
            put(
                &mut m,
                format!("{b}.attn.out_proj.weight"),
                vec![EMBED, EMBED],
                rnd(EMBED * EMBED),
            );
            put(
                &mut m,
                format!("{b}.attn.out_proj.bias"),
                vec![EMBED],
                rnd(EMBED),
            );
            put(
                &mut m,
                format!("{b}.mlp.linear1.weight"),
                vec![VIT_MLP, EMBED],
                rnd(VIT_MLP * EMBED),
            );
            put(
                &mut m,
                format!("{b}.mlp.linear1.bias"),
                vec![VIT_MLP],
                rnd(VIT_MLP),
            );
            put(
                &mut m,
                format!("{b}.mlp.linear2.weight"),
                vec![EMBED, VIT_MLP],
                rnd(EMBED * VIT_MLP),
            );
            put(
                &mut m,
                format!("{b}.mlp.linear2.bias"),
                vec![EMBED],
                rnd(EMBED),
            );
            for n in ["norm1", "norm2"] {
                put(
                    &mut m,
                    format!("{b}.{n}.weight"),
                    vec![EMBED],
                    vec![1.0; EMBED],
                );
                put(
                    &mut m,
                    format!("{b}.{n}.bias"),
                    vec![EMBED],
                    vec![0.0; EMBED],
                );
            }
        }
        put(
            &mut m,
            "image_encoder.norm.weight".into(),
            vec![EMBED],
            vec![1.0; EMBED],
        );
        put(
            &mut m,
            "image_encoder.norm.bias".into(),
            vec![EMBED],
            vec![0.0; EMBED],
        );
        let params = Params::new(m);

        let volume: Vec<f32> = (0..ROI[0] * ROI[1] * ROI[2])
            .map(|i| ((i % 97) as f32) / 97.0)
            .collect();
        let cpu = Vit::build(&params).unwrap().forward(&volume);
        let gpu = GpuVit::new(&ctx, &params)
            .unwrap()
            .forward(&volume)
            .unwrap();
        assert_eq!((gpu.rows, gpu.cols), (cpu.rows, cpu.cols));
        let worst = cpu
            .data
            .iter()
            .zip(gpu.data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Twelve residual blocks of fp32 accumulation in a different order:
        // agreement to ~1e-3 is what equivalence looks like here, not bitwise
        // equality.
        assert!(worst < 2e-3, "worst CPU/GPU difference {worst}");
    }
}
