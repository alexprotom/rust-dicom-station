//! nnU-Net v2 `PlainConvUNet` — architecture assembly and CPU forward pass.
//!
//! The network is rebuilt from `plans.json` (see `config`) and the checkpoint
//! tensors (see `weights`): an encoder of N stages (each: `n_conv` blocks of
//! Conv3d → InstanceNorm → LeakyReLU, the first conv of a stage carrying the
//! downsampling stride), and a decoder of N−1 stages (ConvTranspose3d
//! kernel=stride=2 → concat skip → conv blocks), finished by a 1×1×1
//! segmentation head. Deep-supervision heads exist in the checkpoint for
//! every decoder stage; inference uses only the full-resolution one.
//!
//! Checkpoint key layout (verified against the shipped weights):
//! `encoder.stages.{s}.0.convs.{i}.conv.{weight,bias}` /
//! `…convs.{i}.norm.{weight,bias}`, `decoder.transpconvs.{t}.{weight,bias}`,
//! `decoder.stages.{t}.convs.{i}.…`, `decoder.seg_layers.{t}.{weight,bias}`.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use super::config::ModelConfig;
use super::cpu::{concat, conv3d, conv_transpose3d_stride, instance_norm_lrelu, Act};
use crate::nn::cache::WTensor;

/// One Conv3d → InstanceNorm → LeakyReLU block.
pub struct ConvBlock {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub kernel: [usize; 3],
    pub stride: [usize; 3],
}

pub struct TranspConv {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    /// Kernel = stride of this upsampling step — the encoder stride it
    /// undoes. `[2, 2, 2]` for every isotropic model; the MR models plan
    /// `[1, 2, 2]` where the through-plane spacing is already coarse.
    pub stride: [usize; 3],
}

pub struct SegHead {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub classes: usize,
}

/// The assembled network (plain data — both the CPU and the GPU forward
/// passes read from this).
pub struct UNet {
    pub cfg: ModelConfig,
    pub enc: Vec<Vec<ConvBlock>>,
    pub transp: Vec<TranspConv>,
    pub dec: Vec<Vec<ConvBlock>>,
    pub head: SegHead,
}

fn take<'a>(map: &'a HashMap<String, WTensor>, key: &str) -> Result<&'a WTensor> {
    map.get(key)
        .with_context(|| format!("checkpoint tensor missing: {key}"))
}

impl UNet {
    pub fn build(cfg: ModelConfig, tensors: &HashMap<String, WTensor>) -> Result<UNet> {
        let n = cfg.n_stages();
        let mut enc = Vec::with_capacity(n);
        for s in 0..n {
            let mut blocks = Vec::new();
            let cin_stage = if s == 0 { 1 } else { cfg.features[s - 1] };
            for i in 0..cfg.n_conv_per_stage[s] {
                let cin = if i == 0 { cin_stage } else { cfg.features[s] };
                let cout = cfg.features[s];
                let stride = if i == 0 { cfg.strides[s] } else { [1, 1, 1] };
                let prefix = format!("encoder.stages.{s}.0.convs.{i}");
                blocks.push(load_block(
                    tensors,
                    &prefix,
                    cin,
                    cout,
                    cfg.kernels[s],
                    stride,
                )?);
            }
            enc.push(blocks);
        }
        let mut transp = Vec::with_capacity(n - 1);
        let mut dec = Vec::with_capacity(n - 1);
        for t in 0..n - 1 {
            let c_below = cfg.features[n - 1 - t];
            let c_skip = cfg.features[n - 2 - t];
            // Decoder step `t` undoes the stride of encoder stage n-1-t.
            let stride = cfg.strides[n - 1 - t];
            let tw = take(tensors, &format!("decoder.transpconvs.{t}.weight"))?;
            let tb = take(tensors, &format!("decoder.transpconvs.{t}.bias"))?;
            let want = [c_below, c_skip, stride[0], stride[1], stride[2]];
            if tw.shape != want {
                bail!(
                    "decoder.transpconvs.{t}.weight has shape {:?}, expected {:?}",
                    tw.shape,
                    want
                );
            }
            transp.push(TranspConv {
                w: tw.data.clone(),
                b: tb.data.clone(),
                cin: c_below,
                cout: c_skip,
                stride,
            });
            let mut blocks = Vec::new();
            let stage_kernel = cfg.kernels[n - 2 - t];
            for i in 0..cfg.n_conv_per_stage_decoder[t] {
                let cin = if i == 0 { 2 * c_skip } else { c_skip };
                let prefix = format!("decoder.stages.{t}.convs.{i}");
                blocks.push(load_block(
                    tensors,
                    &prefix,
                    cin,
                    c_skip,
                    stage_kernel,
                    [1, 1, 1],
                )?);
            }
            dec.push(blocks);
        }
        // full-resolution segmentation head = last seg layer
        let head_idx = n - 2;
        let hw = take(tensors, &format!("decoder.seg_layers.{head_idx}.weight"))?;
        let hb = take(tensors, &format!("decoder.seg_layers.{head_idx}.bias"))?;
        let classes = hw.shape[0];
        if hw.shape != [classes, cfg.features[0], 1, 1, 1] {
            bail!(
                "seg head has shape {:?}, expected [classes, {}, 1, 1, 1]",
                hw.shape,
                cfg.features[0]
            );
        }
        Ok(UNet {
            cfg,
            enc,
            transp,
            dec,
            head: SegHead {
                w: hw.data.clone(),
                b: hb.data.clone(),
                cin: cfg_features0(&hw.shape),
                classes,
            },
        })
    }

    pub fn num_classes(&self) -> usize {
        self.head.classes
    }

    /// CPU forward pass for one patch `[1, D, H, W]` → logits
    /// `[classes, D, H, W]`.
    pub fn forward_cpu(&self, x: &Act) -> Act {
        let n = self.enc.len();
        // Every stage's output is a skip connection, and the next stage reads
        // it straight out of `skips` — nothing is copied (stage 0's output is
        // 180 MB at 112³).
        let mut skips: Vec<Act> = Vec::with_capacity(n);
        for (i, stage) in self.enc.iter().enumerate() {
            let src: &Act = if i == 0 { x } else { &skips[i - 1] };
            let mut blocks = stage.iter();
            let first = blocks.next().expect("a stage has at least one block");
            let mut h = run_block(first, src);
            for blk in blocks {
                h = run_block(blk, &h);
            }
            skips.push(h);
        }
        let mut cur = skips.pop().unwrap();
        for (t, tc) in self.transp.iter().enumerate() {
            let up = conv_transpose3d_stride(&cur, &tc.w, &tc.b, tc.cout, tc.stride);
            let skip = skips.pop().unwrap();
            cur = concat(&up, &skip);
            for blk in &self.dec[t] {
                cur = run_block(blk, &cur);
            }
        }
        conv3d(
            &cur,
            &self.head.w,
            &self.head.b,
            self.head.classes,
            [1, 1, 1],
            [1, 1, 1],
        )
    }
}

fn cfg_features0(head_shape: &[usize]) -> usize {
    head_shape[1]
}

fn run_block(blk: &ConvBlock, x: &Act) -> Act {
    let mut y = conv3d(x, &blk.w, &blk.b, blk.cout, blk.kernel, blk.stride);
    instance_norm_lrelu(&mut y, &blk.gamma, &blk.beta);
    y
}

fn load_block(
    tensors: &HashMap<String, WTensor>,
    prefix: &str,
    cin: usize,
    cout: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
) -> Result<ConvBlock> {
    let w = take(tensors, &format!("{prefix}.conv.weight"))?;
    let b = take(tensors, &format!("{prefix}.conv.bias"))?;
    let g = take(tensors, &format!("{prefix}.norm.weight"))?;
    let be = take(tensors, &format!("{prefix}.norm.bias"))?;
    let expect = [cout, cin, kernel[0], kernel[1], kernel[2]];
    if w.shape != expect {
        bail!(
            "{prefix}.conv.weight has shape {:?}, expected {:?}",
            w.shape,
            expect
        );
    }
    if b.data.len() != cout || g.data.len() != cout || be.data.len() != cout {
        bail!("{prefix}: bias/norm length mismatch");
    }
    Ok(ConvBlock {
        w: w.data.clone(),
        b: b.data.clone(),
        gamma: g.data.clone(),
        beta: be.data.clone(),
        cin,
        cout,
        kernel,
        stride,
    })
}
