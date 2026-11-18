//! GPU inference backend via `burn`'s wgpu backend (Vulkan / DX12 / Metal).
//!
//! Works on any GPU wgpu can drive - NVIDIA, AMD, Intel, Apple - with no
//! vendor toolkit; kernels are generated and autotuned by burn/cubecl at
//! runtime. The network weights are uploaded once per model; each sliding-
//! window patch is transferred, run, and the logits read back.
//!
//! The convolutions are not burn's: through the WGSL compiler burn has only
//! its direct convolution (its implicit-GEMM ones need cooperative-matrix
//! instructions, which WGSL does not have, and fail their autotune), at
//! about 0.9 TFLOPS on a card that multiplies matrices several times
//! faster. Every convolution with a kernel wider than a voxel is instead
//! im2col and a matmul ([`conv3d_gemm`]: a few output slices at a time, so
//! its extra memory is bounded), and the transposed convolutions are one
//! matmul and a permutation ([`conv_transpose_gemm`]) - three times faster
//! through the network at 3 mm. The products are those of the direct
//! kernel, summed in another order, so the logits move in their last bits
//! (the labels of the bundled CT are those of the direct kernels, voxel for
//! voxel, at 3 and 1.5 mm); `RDS_GPU_CONV=direct` brings burn's kernels
//! back, for comparing. The head (1×1×1) stays burn's.
//!
//! Both encoders of `net.rs` are mirrored here: the plain conv stages, and
//! the residual stages whose strided skip path is an average pool (done as a
//! depthwise conv3d with a constant kernel, which is what an average pool
//! is) followed, when the width changes, by a 1×1×1 conv and a norm.
//!
//! The segmentation head runs [`HEAD_CLASSES`] classes at a time, each
//! piece read back before the next is computed ([`GpuNet::forward_classes`]):
//! the logits of a 118-class model at a 128-cubed patch are 990 MB, of the
//! residual-encoder `small` models at 192 cubed 3.3 GB, which as one buffer
//! is more than many graphics cards will allocate, and as one readback more
//! than a tablet wants to hold beside its accumulator.
//!
//! Only compiled with the `gpu` cargo feature (on by default).

use anyhow::{anyhow, bail, Context, Result};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::activation::leaky_relu;
use burn::tensor::module::{conv3d, conv_transpose3d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Tensor, TensorData};

use super::net::{ConvBlock, Skip, Stage, UNet};
use crate::nn::device::{guarded, GpuContext};

type B = Wgpu;

/// A convolution's weights, in the one form its convolution reads.
enum ConvW {
    /// `[cout, cin, kd, kh, kw]`, for burn's own kernel (a 1×1×1 kernel,
    /// or every convolution under `RDS_GPU_CONV=direct`).
    Direct(Tensor<B, 5>),
    /// `[cin * taps, cout]`, for [`conv3d_gemm`].
    Gemm(Tensor<B, 2>),
}

struct GBlock {
    w: ConvW,
    b: Tensor<B, 1>,
    gamma: Tensor<B, 5>,
    beta: Tensor<B, 5>,
    kernel: [usize; 3],
    stride: [usize; 3],
}

/// The skip path of a residual block, weights resident.
enum GSkip {
    Identity,
    /// The pooling kernel `[c, 1, k, k, k]` of a depthwise conv, 1/k³ each.
    Pool {
        w: Tensor<B, 5>,
        stride: [usize; 3],
        channels: usize,
    },
    Project(Box<GProject>),
}

/// The pool (when strided), then the 1×1×1 conv and its norm.
struct GProject {
    pool: Option<(Tensor<B, 5>, [usize; 3], usize)>,
    w: Tensor<B, 5>,
    gamma: Tensor<B, 5>,
    beta: Tensor<B, 5>,
}

struct GResBlock {
    conv1: GBlock,
    conv2: GBlock,
    skip: GSkip,
}

enum GStage {
    Plain(Vec<GBlock>),
    Residual(Vec<GResBlock>),
}

struct GTransp {
    /// `[cin, cout, s0, s1, s2]` for burn's kernel, or the same values as
    /// `[cin, cout * taps]` for [`conv_transpose_gemm`].
    w: ConvW,
    b: Tensor<B, 1>,
    stride: [usize; 3],
}

/// Classes of the segmentation head computed and read back at once.
pub const HEAD_CLASSES: usize = 16;

/// The network with weights resident on the GPU.
pub struct GpuNet {
    device: WgpuDevice,
    stem: Option<GBlock>,
    enc: Vec<GStage>,
    transp: Vec<GTransp>,
    dec: Vec<Vec<GBlock>>,
    /// The head's weights in pieces of [`HEAD_CLASSES`] classes: the first
    /// class of the piece, its weights and biases.
    heads: Vec<(usize, Tensor<B, 5>, Tensor<B, 1>)>,
    classes: usize,
}

fn upload5(device: &WgpuDevice, data: &[f32], shape: [usize; 5]) -> Tensor<B, 5> {
    Tensor::from_data(TensorData::new(data.to_vec(), shape), device)
}

fn upload1(device: &WgpuDevice, data: &[f32]) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
}

/// The constant depthwise kernel that averages a `k` window per channel.
fn pool_kernel(device: &WgpuDevice, channels: usize, k: [usize; 3]) -> Tensor<B, 5> {
    let n = k[0] * k[1] * k[2];
    let data = vec![1.0f32 / n as f32; channels * n];
    upload5(device, &data, [channels, 1, k[0], k[1], k[2]])
}

impl GpuNet {
    pub fn new(ctx: &GpuContext, unet: &UNet) -> Result<GpuNet> {
        let d = ctx.device();
        let up_block = |blk: &ConvBlock| -> GBlock {
            let w = if gemm_conv() && blk.kernel != [1, 1, 1] {
                // [cout, k] → [k, cout] on the host, once.
                let k = blk.w.len() / blk.cout;
                let mut t = vec![0f32; blk.w.len()];
                for co in 0..blk.cout {
                    for i in 0..k {
                        t[i * blk.cout + co] = blk.w[co * k + i];
                    }
                }
                ConvW::Gemm(Tensor::from_data(TensorData::new(t, [k, blk.cout]), d))
            } else {
                ConvW::Direct(upload5(
                    d,
                    &blk.w,
                    [
                        blk.cout,
                        blk.cin,
                        blk.kernel[0],
                        blk.kernel[1],
                        blk.kernel[2],
                    ],
                ))
            };
            GBlock {
                w,
                b: upload1(d, &blk.b),
                gamma: upload5(d, &blk.gamma, [1, blk.cout, 1, 1, 1]),
                beta: upload5(d, &blk.beta, [1, blk.cout, 1, 1, 1]),
                kernel: blk.kernel,
                stride: blk.stride,
            }
        };
        let stem = unet.stem.as_ref().map(up_block);
        let enc = unet
            .enc
            .iter()
            .map(|stage| match stage {
                Stage::Plain(blocks) => GStage::Plain(blocks.iter().map(up_block).collect()),
                Stage::Residual(blocks) => GStage::Residual(
                    blocks
                        .iter()
                        .map(|rb| GResBlock {
                            conv1: up_block(&rb.conv1),
                            conv2: up_block(&rb.conv2),
                            skip: match &rb.skip {
                                Skip::Identity => GSkip::Identity,
                                Skip::Pool { kernel } => GSkip::Pool {
                                    w: pool_kernel(d, rb.conv1.cin, *kernel),
                                    stride: *kernel,
                                    channels: rb.conv1.cin,
                                },
                                Skip::Project {
                                    kernel,
                                    w,
                                    gamma,
                                    beta,
                                    cin,
                                    cout,
                                } => GSkip::Project(Box::new(GProject {
                                    pool: (*kernel != [1, 1, 1])
                                        .then(|| (pool_kernel(d, *cin, *kernel), *kernel, *cin)),
                                    w: upload5(d, w, [*cout, *cin, 1, 1, 1]),
                                    gamma: upload5(d, gamma, [1, *cout, 1, 1, 1]),
                                    beta: upload5(d, beta, [1, *cout, 1, 1, 1]),
                                })),
                            },
                        })
                        .collect(),
                ),
            })
            .collect();
        let dec = unet
            .dec
            .iter()
            .map(|stage| stage.iter().map(up_block).collect())
            .collect();
        let transp = unet
            .transp
            .iter()
            .map(|t| GTransp {
                w: if gemm_conv() {
                    // `[cin, cout, s0, s1, s2]` is `[cin, cout * taps]` as it is.
                    ConvW::Gemm(Tensor::from_data(
                        TensorData::new(t.w.clone(), [t.cin, t.w.len() / t.cin]),
                        d,
                    ))
                } else {
                    ConvW::Direct(upload5(
                        d,
                        &t.w,
                        [t.cin, t.cout, t.stride[0], t.stride[1], t.stride[2]],
                    ))
                },
                b: upload1(d, &t.b),
                stride: t.stride,
            })
            .collect();
        let (classes, cin) = (unet.head.classes, unet.head.cin);
        let heads = (0..classes)
            .step_by(HEAD_CLASSES)
            .map(|c0| {
                let n = HEAD_CLASSES.min(classes - c0);
                (
                    c0,
                    upload5(d, &unet.head.w[c0 * cin..(c0 + n) * cin], [n, cin, 1, 1, 1]),
                    upload1(d, &unet.head.b[c0..c0 + n]),
                )
            })
            .collect();
        Ok(GpuNet {
            device: d.clone(),
            stem,
            enc,
            transp,
            dec,
            heads,
            classes,
        })
    }

    pub fn num_classes(&self) -> usize {
        self.classes
    }

    /// Forward one normalized patch `[channels * p0*p1*p2]` → logits
    /// `[classes * p0*p1*p2]`, both flattened C-order. `channels` is 1 for
    /// a segmentation model and follows from the length.
    pub fn forward(&self, patch: &[f32], p: [usize; 3]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.classes * p[0] * p[1] * p[2]];
        let n = p[0] * p[1] * p[2];
        self.forward_classes(patch, p, &mut |c0, piece| {
            out[c0 * n..c0 * n + piece.len()].copy_from_slice(&piece);
            Ok(())
        })?;
        Ok(out)
    }

    /// [`forward`](Self::forward), the logits handed to `sink` a piece of
    /// [`HEAD_CLASSES`] classes at a time, in class order: the first class
    /// of the piece and its logits `[n, p0, p1, p2]`. Only one piece exists
    /// at a time, on the device and here.
    pub fn forward_classes(
        &self,
        patch: &[f32],
        p: [usize; 3],
        sink: &mut dyn FnMut(usize, Vec<f32>) -> Result<()>,
    ) -> Result<()> {
        let channels = patch.len() / (p[0] * p[1] * p[2]).max(1);
        let t_features = std::time::Instant::now();
        let mut sink_ns = 0u64;
        let features = guarded(|| -> Result<Tensor<B, 5>> {
            let x = Tensor::<B, 5>::from_data(
                TensorData::new(patch.to_vec(), [1, channels, p[0], p[1], p[2]]),
                &self.device,
            );
            let mut skips: Vec<Tensor<B, 5>> = Vec::with_capacity(self.enc.len());
            let mut h = match &self.stem {
                Some(stem) => run_block(stem, x),
                None => x,
            };
            for stage in &self.enc {
                match stage {
                    GStage::Plain(blocks) => {
                        for blk in blocks {
                            h = run_block(blk, h);
                        }
                    }
                    GStage::Residual(blocks) => {
                        for blk in blocks {
                            h = run_res_block(blk, h);
                        }
                    }
                }
                skips.push(h.clone());
            }
            let mut cur = skips.pop().unwrap();
            for (t, tc) in self.transp.iter().enumerate() {
                cur = match &tc.w {
                    ConvW::Gemm(wm) => conv_transpose_gemm(cur, tc, wm),
                    ConvW::Direct(w) => conv_transpose3d(
                        cur,
                        w.clone(),
                        Some(tc.b.clone()),
                        ConvTransposeOptions::new(tc.stride, [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
                    ),
                };
                let skip = skips.pop().unwrap();
                cur = Tensor::cat(vec![cur, skip], 1);
                for blk in &self.dec[t] {
                    cur = run_block(blk, cur);
                }
            }
            Ok(cur)
        })
        .context("GPU forward")?;
        if super::timing::enabled() {
            // Timing only: wait for the features by reading one value.
            guarded(|| {
                let _ = features
                    .clone()
                    .slice([0..1, 0..1, 0..1, 0..1, 0..1])
                    .into_data();
                Ok(())
            })?;
            super::timing::add(super::timing::Stage::GpuFeatures, t_features);
        }
        let t_heads = std::time::Instant::now();
        for (c0, w, b) in &self.heads {
            let piece = guarded(|| -> Result<Vec<f32>> {
                let logits = conv3d(
                    features.clone(),
                    w.clone(),
                    Some(b.clone()),
                    ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
                );
                // `into_vec` keeps the read-back bytes as the vector where
                // their alignment allows; `to_vec` always copied them.
                logits
                    .into_data()
                    .into_vec::<f32>()
                    .map_err(|e| anyhow!("GPU readback failed: {e:?}"))
            })
            .context("GPU forward")?;
            let n = HEAD_CLASSES.min(self.classes - c0);
            if piece.len() != n * p[0] * p[1] * p[2] {
                bail!("GPU returned unexpected logits size {}", piece.len());
            }
            let t_sink = std::time::Instant::now();
            sink(*c0, piece)?;
            sink_ns += t_sink.elapsed().as_nanos() as u64;
        }
        super::timing::add_ns(
            super::timing::Stage::GpuHeads,
            (t_heads.elapsed().as_nanos() as u64).saturating_sub(sink_ns),
        );
        Ok(())
    }
}

/// Conv → InstanceNorm3d, without the activation.
fn conv_norm(blk: &GBlock, x: Tensor<B, 5>) -> Tensor<B, 5> {
    let y = match &blk.w {
        ConvW::Gemm(wt) => conv3d_gemm(x, wt, &blk.b, blk.kernel, blk.stride),
        ConvW::Direct(w) => {
            let pad = [blk.kernel[0] / 2, blk.kernel[1] / 2, blk.kernel[2] / 2];
            conv3d(
                x,
                w.clone(),
                Some(blk.b.clone()),
                ConvOptions::new(blk.stride, pad, [1, 1, 1], 1),
            )
        }
    };
    instance_norm(y, &blk.gamma, &blk.beta)
}

/// Whether the convolutions and transposed convolutions go through
/// [`conv3d_gemm`] and [`conv_transpose_gemm`] (the default) rather than
/// burn's own kernels, which `RDS_GPU_CONV=direct` asks for: on the WGSL
/// compiler burn has only its direct convolution (the implicit-GEMM kernels
/// need cooperative-matrix instructions WGSL does not have), and that runs
/// at under a tenth of what the matmul does on the same card. Both sum the
/// same products in another order, so the logits differ in the last bits;
/// kept for comparing the two.
pub fn gemm_conv() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RDS_GPU_CONV").map_or(true, |v| v != "direct"))
}

/// Floats of im2col one chunk of output slices builds at most (256 MB;
/// the taps cut out for it and the matmul's output come on top, so a few
/// hundred megabytes in all): a few slices of a full-resolution layer at a
/// time, so the memory a window needs on the device grows by a bounded
/// amount, not by the layer's whole `taps x` copy of its input.
const GEMM_COL_FLOATS: usize = 64 << 20;

/// A convolution with `kernel / 2` padding and any stride as im2col and a
/// matmul. For a few output slices at a time the taps of the zero-padded
/// input are cut out (stepped by the stride) and stacked into the columns
/// `[voxels, cin * taps]`, in the order of the weights' own flattening
/// (`[cout, cin, kd, kh, kw]` is `[cout, cin * taps]`), and multiplied by
/// the weights transposed, `wt` = `[cin * taps, cout]`; the products,
/// `[voxels, cout]`, are turned back to channels first and the chunks
/// joined along the depth. Voxels as rows is the orientation burn's tiled
/// matmul runs fastest in here (a tenth faster than `[cout, voxels]`).
fn conv3d_gemm(
    x: Tensor<B, 5>,
    wt: &Tensor<B, 2>,
    b: &Tensor<B, 1>,
    kernel: [usize; 3],
    stride: [usize; 3],
) -> Tensor<B, 5> {
    use burn::tensor::ops::PadMode;
    use burn::tensor::Slice;
    let [_, cin, d, h, w] = x.dims();
    let [k, cout] = wt.dims();
    let p: [usize; 3] = std::array::from_fn(|a| kernel[a] / 2);
    let out_len = |len: usize, a: usize| (len + 2 * p[a] - kernel[a]) / stride[a] + 1;
    let (od, oh, ow) = (out_len(d, 0), out_len(h, 1), out_len(w, 2));
    let bias: Tensor<B, 2> = b.clone().reshape([1, cout]);
    let xp: Tensor<B, 4> = x.reshape([cin, d, h, w]).pad(
        [(p[0], p[0]), (p[1], p[1]), (p[2], p[2])],
        PadMode::Constant(0.0),
    );
    let plane = oh * ow;
    let cz = (GEMM_COL_FLOATS / (k * plane).max(1)).clamp(1, od);
    // `n` positions from `start`, `step` apart.
    let span = |start: usize, n: usize, step: usize| {
        Slice::new(
            start as isize,
            Some((start + (n - 1) * step + 1) as isize),
            step as isize,
        )
    };
    let mut outs = Vec::with_capacity(od.div_ceil(cz));
    for z0 in (0..od).step_by(cz) {
        let n = cz.min(od - z0);
        let mut taps: Vec<Tensor<B, 2>> = Vec::with_capacity(kernel.iter().product());
        for kz in 0..kernel[0] {
            for ky in 0..kernel[1] {
                for kx in 0..kernel[2] {
                    taps.push(
                        xp.clone()
                            .slice([
                                Slice::new(0, Some(cin as isize), 1),
                                span(z0 * stride[0] + kz, n, stride[0]),
                                span(ky, oh, stride[1]),
                                span(kx, ow, stride[2]),
                            ])
                            .reshape([cin, n * plane]),
                    );
                }
            }
        }
        // [cin, taps, voxels] → [cin * taps, voxels] → [voxels, cin * taps]
        let col: Tensor<B, 3> = Tensor::stack(taps, 1);
        let col = col.reshape([k, n * plane]).transpose();
        let y = col.matmul(wt.clone()) + bias.clone();
        outs.push(y.transpose().reshape([cout, n, oh, ow]));
    }
    Tensor::cat(outs, 1).reshape([1, cout, od, oh, ow])
}

/// A transposed convolution with kernel = stride (nnU-Net's upsampling) as
/// one matmul: every input voxel times the weights `[cin, cout * taps]`
/// gives its `taps` output voxels of every channel, which a permutation
/// interleaves into the larger grid (the outputs of two input voxels never
/// overlap, so nothing is added up across them).
fn conv_transpose_gemm(x: Tensor<B, 5>, t: &GTransp, wm: &Tensor<B, 2>) -> Tensor<B, 5> {
    let [_, cin, d, h, w] = x.dims();
    let [s0, s1, s2] = t.stride;
    let cout = wm.dims()[1] / (s0 * s1 * s2);
    let y: Tensor<B, 7> = x
        .reshape([cin, d * h * w])
        .transpose()
        .matmul(wm.clone())
        .reshape([d, h, w, cout, s0, s1, s2]);
    // [d, h, w, cout, s0, s1, s2] → [cout, d, s0, h, s1, w, s2]
    let y: Tensor<B, 5> =
        y.permute([3, 0, 4, 1, 5, 2, 6])
            .reshape([1, cout, d * s0, h * s1, w * s2]);
    y + t.b.clone().reshape([1, cout, 1, 1, 1])
}

/// InstanceNorm3d (biased variance, eps 1e-5) with affine weights.
fn instance_norm(y: Tensor<B, 5>, gamma: &Tensor<B, 5>, beta: &Tensor<B, 5>) -> Tensor<B, 5> {
    let mean = y.clone().mean_dim(2).mean_dim(3).mean_dim(4);
    let centered = y - mean;
    let var = centered
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .mean_dim(3)
        .mean_dim(4);
    let norm = centered / (var + 1e-5).sqrt();
    norm * gamma.clone() + beta.clone()
}

/// Conv → InstanceNorm3d → LeakyReLU(0.01).
fn run_block(blk: &GBlock, x: Tensor<B, 5>) -> Tensor<B, 5> {
    leaky_relu(conv_norm(blk, x), 0.01)
}

/// An average pool with kernel = stride as a depthwise convolution.
fn avg_pool(
    x: Tensor<B, 5>,
    w: &Tensor<B, 5>,
    stride: [usize; 3],
    channels: usize,
) -> Tensor<B, 5> {
    conv3d(
        x,
        w.clone(),
        None,
        ConvOptions::new(stride, [0, 0, 0], [1, 1, 1], channels),
    )
}

/// One residual block: conv1 with its activation, conv2 normalized, the
/// skip path added, LeakyReLU.
fn run_res_block(blk: &GResBlock, x: Tensor<B, 5>) -> Tensor<B, 5> {
    let residual = match &blk.skip {
        GSkip::Identity => x.clone(),
        GSkip::Pool {
            w,
            stride,
            channels,
        } => avg_pool(x.clone(), w, *stride, *channels),
        GSkip::Project(p) => {
            let src = match &p.pool {
                Some((pw, stride, channels)) => avg_pool(x.clone(), pw, *stride, *channels),
                None => x.clone(),
            };
            let projected = conv3d(
                src,
                p.w.clone(),
                None,
                ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
            );
            instance_norm(projected, &p.gamma, &p.beta)
        }
    };
    let h = run_block(&blk.conv1, x);
    let y = conv_norm(&blk.conv2, h);
    leaky_relu(y + residual, 0.01)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autoseg::net::tests::{patch, tiny_with};

    /// The GPU network against the CPU one, its head in pieces (20 classes:
    /// a piece of 16 and one of 4). Needs a GPU (a software one will do):
    /// `RDS_GPU_TESTS=1 cargo test -p rds-engines -- --ignored gpu_head`.
    #[test]
    #[ignore]
    fn gpu_head_in_pieces_matches_the_cpu() {
        if std::env::var_os("RDS_GPU_TESTS").is_none() {
            eprintln!("RDS_GPU_TESTS is not set");
            return;
        }
        let ctx = GpuContext::try_new().expect("a GPU");
        for residual in [false, true] {
            let (cfg, t) = tiny_with(residual, 20);
            let p = cfg.patch_size;
            let unet = UNet::build(cfg, &t).unwrap();
            let g = GpuNet::new(&ctx, &unet).unwrap();
            let x = patch(5, p);
            let want = unet.forward_cpu(&x).data;
            let mut pieces = Vec::new();
            g.forward_classes(&x.data, p, &mut |c0, piece| {
                pieces.push((c0, piece.len()));
                Ok(())
            })
            .unwrap();
            let n = p[0] * p[1] * p[2];
            assert_eq!(pieces, vec![(0, 16 * n), (16, 4 * n)]);
            let got = g.forward(&x.data, p).unwrap();
            let range = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            let worst = got
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("residual {residual}: worst {worst:e} of {range}");
            assert!(worst <= 1e-3 * range.max(1.0), "{worst} vs range {range}");
        }
    }
}
