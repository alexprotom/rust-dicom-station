//! GPU inference backend via `burn`'s wgpu backend (Vulkan / DX12 / Metal).
//!
//! Works on any GPU wgpu can drive - NVIDIA, AMD, Intel, Apple - with no
//! vendor toolkit; kernels are generated and autotuned by burn/cubecl at
//! runtime. The network weights are uploaded once per model; each sliding-
//! window patch is transferred, run, and the logits read back.
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

struct GBlock {
    w: Tensor<B, 5>,
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
    w: Tensor<B, 5>,
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
            GBlock {
                w: upload5(
                    d,
                    &blk.w,
                    [
                        blk.cout,
                        blk.cin,
                        blk.kernel[0],
                        blk.kernel[1],
                        blk.kernel[2],
                    ],
                ),
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
                w: upload5(
                    d,
                    &t.w,
                    [t.cin, t.cout, t.stride[0], t.stride[1], t.stride[2]],
                ),
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
                cur = conv_transpose3d(
                    cur,
                    tc.w.clone(),
                    Some(tc.b.clone()),
                    ConvTransposeOptions::new(tc.stride, [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
                );
                let skip = skips.pop().unwrap();
                cur = Tensor::cat(vec![cur, skip], 1);
                for blk in &self.dec[t] {
                    cur = run_block(blk, cur);
                }
            }
            Ok(cur)
        })
        .context("GPU forward")?;
        for (c0, w, b) in &self.heads {
            let piece = guarded(|| -> Result<Vec<f32>> {
                let logits = conv3d(
                    features.clone(),
                    w.clone(),
                    Some(b.clone()),
                    ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
                );
                logits
                    .into_data()
                    .to_vec::<f32>()
                    .map_err(|e| anyhow!("GPU readback failed: {e:?}"))
            })
            .context("GPU forward")?;
            let n = HEAD_CLASSES.min(self.classes - c0);
            if piece.len() != n * p[0] * p[1] * p[2] {
                bail!("GPU returned unexpected logits size {}", piece.len());
            }
            sink(*c0, piece)?;
        }
        Ok(())
    }
}

/// Conv → InstanceNorm3d, without the activation.
fn conv_norm(blk: &GBlock, x: Tensor<B, 5>) -> Tensor<B, 5> {
    let pad = [blk.kernel[0] / 2, blk.kernel[1] / 2, blk.kernel[2] / 2];
    let y = conv3d(
        x,
        blk.w.clone(),
        Some(blk.b.clone()),
        ConvOptions::new(blk.stride, pad, [1, 1, 1], 1),
    );
    instance_norm(y, &blk.gamma, &blk.beta)
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
