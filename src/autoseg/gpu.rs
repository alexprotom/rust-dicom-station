//! GPU inference backend via `burn`'s wgpu backend (Vulkan / DX12 / Metal).
//!
//! Works on any GPU wgpu can drive — NVIDIA, AMD, Intel, Apple — with no
//! vendor toolkit; kernels are generated and autotuned by burn/cubecl at
//! runtime. The network weights are uploaded once per model; each sliding-
//! window patch is transferred, run, and the logits read back.
//!
//! Only compiled with the `gpu` cargo feature (on by default).

use anyhow::{anyhow, bail, Context, Result};

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::tensor::activation::leaky_relu;
use burn::tensor::module::{conv3d, conv_transpose3d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Tensor, TensorData};

use super::net::UNet;
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

struct GTransp {
    w: Tensor<B, 5>,
    b: Tensor<B, 1>,
}

/// The network with weights resident on the GPU.
pub struct GpuNet {
    device: WgpuDevice,
    enc: Vec<Vec<GBlock>>,
    transp: Vec<GTransp>,
    dec: Vec<Vec<GBlock>>,
    head_w: Tensor<B, 5>,
    head_b: Tensor<B, 1>,
    classes: usize,
}

fn upload5(device: &WgpuDevice, data: &[f32], shape: [usize; 5]) -> Tensor<B, 5> {
    Tensor::from_data(TensorData::new(data.to_vec(), shape), device)
}

fn upload1(device: &WgpuDevice, data: &[f32]) -> Tensor<B, 1> {
    Tensor::from_data(TensorData::new(data.to_vec(), [data.len()]), device)
}

impl GpuNet {
    pub fn new(ctx: &GpuContext, unet: &UNet) -> Result<GpuNet> {
        let d = ctx.device();
        let up_block = |blk: &super::net::ConvBlock| -> GBlock {
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
        let enc = unet
            .enc
            .iter()
            .map(|stage| stage.iter().map(up_block).collect())
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
                w: upload5(d, &t.w, [t.cin, t.cout, 2, 2, 2]),
                b: upload1(d, &t.b),
            })
            .collect();
        let head_w = upload5(d, &unet.head.w, [unet.head.classes, unet.head.cin, 1, 1, 1]);
        let head_b = upload1(d, &unet.head.b);
        Ok(GpuNet {
            device: d.clone(),
            enc,
            transp,
            dec,
            head_w,
            head_b,
            classes: unet.head.classes,
        })
    }

    /// Forward one normalized patch `[p0*p1*p2]` → logits
    /// `[classes * p0*p1*p2]`, both flattened C-order.
    pub fn forward(&self, patch: &[f32], p: [usize; 3]) -> Result<Vec<f32>> {
        let run = || -> Result<Vec<f32>> {
            let x = Tensor::<B, 5>::from_data(
                TensorData::new(patch.to_vec(), [1, 1, p[0], p[1], p[2]]),
                &self.device,
            );
            let mut skips: Vec<Tensor<B, 5>> = Vec::with_capacity(self.enc.len());
            let mut h = x;
            for stage in &self.enc {
                for blk in stage {
                    h = run_block(blk, h);
                }
                skips.push(h.clone());
            }
            let mut cur = skips.pop().unwrap();
            for (t, tc) in self.transp.iter().enumerate() {
                cur = conv_transpose3d(
                    cur,
                    tc.w.clone(),
                    Some(tc.b.clone()),
                    ConvTransposeOptions::new([2, 2, 2], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1),
                );
                let skip = skips.pop().unwrap();
                cur = Tensor::cat(vec![cur, skip], 1);
                for blk in &self.dec[t] {
                    cur = run_block(blk, cur);
                }
            }
            let logits = conv3d(
                cur,
                self.head_w.clone(),
                Some(self.head_b.clone()),
                ConvOptions::new([1, 1, 1], [0, 0, 0], [1, 1, 1], 1),
            );
            let data = logits
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| anyhow!("GPU readback failed: {e:?}"))?;
            if data.len() != self.classes * p[0] * p[1] * p[2] {
                bail!("GPU returned unexpected logits size {}", data.len());
            }
            Ok(data)
        };
        guarded(run).context("GPU forward")
    }
}

fn run_block(blk: &GBlock, x: Tensor<B, 5>) -> Tensor<B, 5> {
    let pad = [blk.kernel[0] / 2, blk.kernel[1] / 2, blk.kernel[2] / 2];
    let y = conv3d(
        x,
        blk.w.clone(),
        Some(blk.b.clone()),
        ConvOptions::new(blk.stride, pad, [1, 1, 1], 1),
    );
    // InstanceNorm3d (biased variance, eps 1e-5) + LeakyReLU(0.01)
    let mean = y.clone().mean_dim(2).mean_dim(3).mean_dim(4);
    let centered = y - mean;
    let var = centered
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .mean_dim(3)
        .mean_dim(4);
    let norm = centered / (var + 1e-5).sqrt();
    let scaled = norm * blk.gamma.clone() + blk.beta.clone();
    leaky_relu(scaled, 0.01)
}
