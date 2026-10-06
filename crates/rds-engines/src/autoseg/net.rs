//! nnU-Net v2 networks - architecture assembly and CPU forward pass.
//!
//! Two encoders share one decoder:
//!
//! * `PlainConvUNet`: an encoder of N stages (each: `n_conv` blocks of
//!   Conv3d → InstanceNorm → LeakyReLU, the first conv of a stage carrying
//!   the downsampling stride). Every TotalSegmentator model up to v2 and
//!   the `big` v3 models.
//! * `ResidualEncoderUNet` (the `nnUNetResEncUNet*Plans`, the `small` v3
//!   models): a one-block stem, then N stages of `n_blocks` residual blocks.
//!   A block is Conv3d → InstanceNorm → LeakyReLU, Conv3d → InstanceNorm,
//!   plus the input added back and one more LeakyReLU; the first block of a
//!   stage carries the stride, and its skip path is then an average pool
//!   with kernel = stride, followed - when the width changes too - by a
//!   bias-free 1×1×1 Conv3d and an InstanceNorm.
//!
//! The decoder is the same for both: N−1 stages of ConvTranspose3d
//! (kernel = stride) → concat skip → conv blocks, finished by a 1×1×1
//! segmentation head. Deep-supervision heads exist in the checkpoint for
//! every decoder stage; inference uses only the full-resolution one.
//!
//! Checkpoint key layout (verified against the shipped weights):
//! `encoder.stages.{s}.0.convs.{i}.conv.{weight,bias}` /
//! `…convs.{i}.norm.{weight,bias}` for the plain encoder;
//! `encoder.stem.convs.0.{conv,norm}.*`,
//! `encoder.stages.{s}.blocks.{b}.{conv1,conv2}.{conv,norm}.*` and
//! `encoder.stages.{s}.blocks.{b}.skip.1.{conv.weight,norm.weight,norm.bias}`
//! for the residual one (`skip.0` is the parameter-free pool);
//! `decoder.transpconvs.{t}.{weight,bias}`, `decoder.stages.{t}.convs.{i}.…`,
//! `decoder.seg_layers.{t}.{weight,bias}` for both.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

use super::config::{Arch, ModelConfig};
use super::cpu::{
    add_lrelu, avg_pool3d, concat, conv3d, conv_transpose3d_stride, instance_norm,
    instance_norm_lrelu, Act,
};
use crate::nn::cache::WTensor;

/// One Conv3d → InstanceNorm block; the LeakyReLU after it is the
/// caller's choice (every plain block has it, a residual block's second
/// conv does not).
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

/// The skip path of a residual block.
pub enum Skip {
    /// Same width, stride 1: the input itself.
    Identity,
    /// Stride without a change of width: an average pool, kernel = stride.
    Pool { kernel: [usize; 3] },
    /// Stride and a change of width: the pool, then a bias-free 1×1×1 conv
    /// and an InstanceNorm. (A change of width without a stride never
    /// occurs in the released plans; it would be the conv and norm alone,
    /// and is assembled as a pool with kernel 1.)
    Project {
        kernel: [usize; 3],
        w: Vec<f32>,
        gamma: Vec<f32>,
        beta: Vec<f32>,
        cin: usize,
        cout: usize,
    },
}

/// One `BasicBlockD` of the residual encoder.
pub struct ResBlock {
    pub conv1: ConvBlock,
    pub conv2: ConvBlock,
    pub skip: Skip,
}

/// One encoder stage, of either family.
pub enum Stage {
    Plain(Vec<ConvBlock>),
    Residual(Vec<ResBlock>),
}

pub struct TranspConv {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    /// Kernel = stride of this upsampling step - the encoder stride it
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

/// The assembled network (plain data - both the CPU and the GPU forward
/// passes read from this).
pub struct UNet {
    pub cfg: ModelConfig,
    /// Input channels of the first convolution: 1 for every segmentation
    /// model, the image plus the prompt channels for nnInteractive.
    pub in_channels: usize,
    /// The residual encoder's stem; `None` for the plain network.
    pub stem: Option<ConvBlock>,
    pub enc: Vec<Stage>,
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
        // The checkpoint says how many channels the first convolution reads.
        let first = match &cfg.arch {
            Arch::PlainConv => "encoder.stages.0.0.convs.0.conv.weight",
            Arch::ResidualEncoder { .. } => "encoder.stem.convs.0.conv.weight",
        };
        let in_channels = take(tensors, first)?.shape.get(1).copied().unwrap_or(1);
        let (stem, enc) = match &cfg.arch {
            Arch::PlainConv => (None, build_plain_encoder(&cfg, in_channels, tensors)?),
            Arch::ResidualEncoder { blocks_per_stage } => {
                let stem = load_block(
                    tensors,
                    "encoder.stem.convs.0",
                    in_channels,
                    cfg.features[0],
                    cfg.kernels[0],
                    [1, 1, 1],
                )?;
                (
                    Some(stem),
                    build_residual_encoder(&cfg, blocks_per_stage, tensors)?,
                )
            }
        };
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
            let stage_kernel = cfg
                .decoder_kernels
                .get(t)
                .copied()
                .unwrap_or(cfg.kernels[n - 2 - t]);
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
            in_channels,
            stem,
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

    /// CPU forward pass for one patch `[in_channels, D, H, W]` → logits
    /// `[classes, D, H, W]`.
    pub fn forward_cpu(&self, x: &Act) -> Act {
        let n = self.enc.len();
        // Every stage's output is a skip connection, and the next stage reads
        // it straight out of `skips` - nothing is copied (stage 0's output is
        // 180 MB at 112³).
        let mut skips: Vec<Act> = Vec::with_capacity(n);
        let stem_out = self.stem.as_ref().map(|blk| run_block(blk, x));
        for (i, stage) in self.enc.iter().enumerate() {
            let src: &Act = if i == 0 {
                stem_out.as_ref().unwrap_or(x)
            } else {
                &skips[i - 1]
            };
            let h = match stage {
                Stage::Plain(blocks) => {
                    let mut blocks = blocks.iter();
                    let first = blocks.next().expect("a stage has at least one block");
                    let mut h = run_block(first, src);
                    for blk in blocks {
                        h = run_block(blk, &h);
                    }
                    h
                }
                Stage::Residual(blocks) => {
                    let mut blocks = blocks.iter();
                    let first = blocks.next().expect("a stage has at least one block");
                    let mut h = run_res_block(first, src);
                    for blk in blocks {
                        h = run_res_block(blk, &h);
                    }
                    h
                }
            };
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

fn build_plain_encoder(
    cfg: &ModelConfig,
    in_channels: usize,
    tensors: &HashMap<String, WTensor>,
) -> Result<Vec<Stage>> {
    let n = cfg.n_stages();
    let mut enc = Vec::with_capacity(n);
    for s in 0..n {
        let mut blocks = Vec::new();
        let cin_stage = if s == 0 {
            in_channels
        } else {
            cfg.features[s - 1]
        };
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
        enc.push(Stage::Plain(blocks));
    }
    Ok(enc)
}

fn build_residual_encoder(
    cfg: &ModelConfig,
    blocks_per_stage: &[usize],
    tensors: &HashMap<String, WTensor>,
) -> Result<Vec<Stage>> {
    let mut enc = Vec::with_capacity(cfg.n_stages());
    for (s, &n_blocks) in blocks_per_stage.iter().enumerate() {
        let mut blocks = Vec::new();
        // The stem already brought the input to the first stage's width.
        let cin_stage = if s == 0 {
            cfg.features[0]
        } else {
            cfg.features[s - 1]
        };
        let cout = cfg.features[s];
        for b in 0..n_blocks {
            let cin = if b == 0 { cin_stage } else { cout };
            let stride = if b == 0 { cfg.strides[s] } else { [1, 1, 1] };
            let prefix = format!("encoder.stages.{s}.blocks.{b}");
            let conv1 = load_block(
                tensors,
                &format!("{prefix}.conv1"),
                cin,
                cout,
                cfg.kernels[s],
                stride,
            )?;
            let conv2 = load_block(
                tensors,
                &format!("{prefix}.conv2"),
                cout,
                cout,
                cfg.kernels[s],
                [1, 1, 1],
            )?;
            let has_stride = stride != [1, 1, 1];
            let skip = match (has_stride, cin != cout) {
                (false, false) => Skip::Identity,
                (true, false) => Skip::Pool { kernel: stride },
                (_, true) => {
                    let w = take(tensors, &format!("{prefix}.skip.1.conv.weight"))?;
                    let g = take(tensors, &format!("{prefix}.skip.1.norm.weight"))?;
                    let be = take(tensors, &format!("{prefix}.skip.1.norm.bias"))?;
                    if w.shape != [cout, cin, 1, 1, 1] {
                        bail!(
                            "{prefix}.skip.1.conv.weight has shape {:?}, expected {:?}",
                            w.shape,
                            [cout, cin, 1, 1, 1]
                        );
                    }
                    if g.data.len() != cout || be.data.len() != cout {
                        bail!("{prefix}.skip.1: norm length mismatch");
                    }
                    Skip::Project {
                        kernel: stride,
                        w: w.data.clone(),
                        gamma: g.data.clone(),
                        beta: be.data.clone(),
                        cin,
                        cout,
                    }
                }
            };
            blocks.push(ResBlock { conv1, conv2, skip });
        }
        enc.push(Stage::Residual(blocks));
    }
    Ok(enc)
}

fn cfg_features0(head_shape: &[usize]) -> usize {
    head_shape[1]
}

/// Conv → InstanceNorm → LeakyReLU.
fn run_block(blk: &ConvBlock, x: &Act) -> Act {
    let mut y = conv3d(x, &blk.w, &blk.b, blk.cout, blk.kernel, blk.stride);
    instance_norm_lrelu(&mut y, &blk.gamma, &blk.beta);
    y
}

/// One residual block: conv1 (with its LeakyReLU), conv2 (norm only), the
/// skip path added, LeakyReLU.
fn run_res_block(blk: &ResBlock, x: &Act) -> Act {
    let h = run_block(&blk.conv1, x);
    let mut y = conv3d(
        &h,
        &blk.conv2.w,
        &blk.conv2.b,
        blk.conv2.cout,
        blk.conv2.kernel,
        blk.conv2.stride,
    );
    instance_norm(&mut y, &blk.conv2.gamma, &blk.conv2.beta);
    let residual: Act;
    let r: &Act = match &blk.skip {
        Skip::Identity => x,
        Skip::Pool { kernel } => {
            residual = avg_pool3d(x, *kernel);
            &residual
        }
        Skip::Project {
            kernel,
            w,
            gamma,
            beta,
            cin: _,
            cout,
        } => {
            let pooled = if *kernel == [1, 1, 1] {
                None
            } else {
                Some(avg_pool3d(x, *kernel))
            };
            let src = pooled.as_ref().unwrap_or(x);
            let zero_bias = vec![0.0f32; *cout];
            let mut p = conv3d(src, w, &zero_bias, *cout, [1, 1, 1], [1, 1, 1]);
            instance_norm(&mut p, gamma, beta);
            residual = p;
            &residual
        }
    };
    add_lrelu(&mut y, r);
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
