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
    add_lrelu, avg_pool3d_in, concat_in, conv3d_in, conv_transpose3d_stride_into, instance_norm,
    instance_norm_lrelu, Act, Arena,
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

/// Where [`UNet`] gets its tensors: a borrowed map, each tensor copied out
/// of it, or one handed over, each tensor moved out (no copy: assembling a
/// network then costs no memory beyond the weights already loaded).
enum Tensors<'a> {
    Borrowed(&'a HashMap<String, WTensor>),
    Owned(HashMap<String, WTensor>),
}

impl Tensors<'_> {
    fn shape(&self, key: &str) -> Result<&[usize]> {
        let t = match self {
            Tensors::Borrowed(m) => m.get(key),
            Tensors::Owned(m) => m.get(key),
        };
        t.map(|t| t.shape.as_slice())
            .with_context(|| format!("checkpoint tensor missing: {key}"))
    }

    fn take(&mut self, key: &str) -> Result<WTensor> {
        let t = match self {
            Tensors::Borrowed(m) => m.get(key).cloned(),
            Tensors::Owned(m) => m.remove(key),
        };
        t.with_context(|| format!("checkpoint tensor missing: {key}"))
    }
}

impl UNet {
    pub fn build(cfg: ModelConfig, tensors: &HashMap<String, WTensor>) -> Result<UNet> {
        Self::assemble(cfg, &mut Tensors::Borrowed(tensors))
    }

    /// [`build`](Self::build) from a map the network may keep: every tensor
    /// is moved into it rather than copied.
    pub fn build_owned(cfg: ModelConfig, tensors: HashMap<String, WTensor>) -> Result<UNet> {
        Self::assemble(cfg, &mut Tensors::Owned(tensors))
    }

    fn assemble(cfg: ModelConfig, tensors: &mut Tensors) -> Result<UNet> {
        let n = cfg.n_stages();
        // The checkpoint says how many channels the first convolution reads.
        let first = match &cfg.arch {
            Arch::PlainConv => "encoder.stages.0.0.convs.0.conv.weight",
            Arch::ResidualEncoder { .. } => "encoder.stem.convs.0.conv.weight",
        };
        let in_channels = tensors.shape(first)?.get(1).copied().unwrap_or(1);
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
            let tw = tensors.take(&format!("decoder.transpconvs.{t}.weight"))?;
            let tb = tensors.take(&format!("decoder.transpconvs.{t}.bias"))?;
            let want = [c_below, c_skip, stride[0], stride[1], stride[2]];
            if tw.shape != want {
                bail!(
                    "decoder.transpconvs.{t}.weight has shape {:?}, expected {:?}",
                    tw.shape,
                    want
                );
            }
            transp.push(TranspConv {
                w: tw.data,
                b: tb.data,
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
        let hw = tensors.take(&format!("decoder.seg_layers.{head_idx}.weight"))?;
        let hb = tensors.take(&format!("decoder.seg_layers.{head_idx}.bias"))?;
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
                cin: cfg_features0(&hw.shape),
                w: hw.data,
                b: hb.data,
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
        self.forward_cpu_in(x, &mut Arena::default())
    }

    /// [`forward_cpu`](Self::forward_cpu) with every activation taken from
    /// `arena` and given back as soon as the next layer has read it, so a
    /// run reuses one patch's buffers for the next. The logits come from the
    /// arena too; give them back once read. The arithmetic is the same to
    /// the bit.
    pub fn forward_cpu_in(&self, x: &Act, arena: &mut Arena) -> Act {
        let n = self.enc.len();
        // Every stage's output is a skip connection, and the next stage reads
        // it straight out of `skips` - nothing is copied (stage 0's output is
        // 180 MB at 112³).
        let mut skips: Vec<Act> = Vec::with_capacity(n);
        let stem_out = self.stem.as_ref().map(|blk| run_block(blk, x, arena));
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
                    let mut h = run_block(first, src, arena);
                    for blk in blocks {
                        let next = run_block(blk, &h, arena);
                        arena.give(std::mem::replace(&mut h, next).data);
                    }
                    h
                }
                Stage::Residual(blocks) => {
                    let mut blocks = blocks.iter();
                    let first = blocks.next().expect("a stage has at least one block");
                    let mut h = run_res_block(first, src, arena);
                    for blk in blocks {
                        let next = run_res_block(blk, &h, arena);
                        arena.give(std::mem::replace(&mut h, next).data);
                    }
                    h
                }
            };
            skips.push(h);
        }
        if let Some(s) = stem_out {
            arena.give(s.data);
        }
        let mut cur = skips.pop().unwrap();
        for (t, tc) in self.transp.iter().enumerate() {
            let up_shape = arena.act(
                tc.cout,
                cur.d * tc.stride[0],
                cur.h * tc.stride[1],
                cur.w * tc.stride[2],
            );
            let up = conv_transpose3d_stride_into(&cur, &tc.w, &tc.b, tc.stride, up_shape);
            arena.give(cur.data);
            let skip = skips.pop().unwrap();
            cur = concat_in(&up, &skip, arena);
            arena.give(up.data);
            arena.give(skip.data);
            for blk in &self.dec[t] {
                let next = run_block(blk, &cur, arena);
                arena.give(std::mem::replace(&mut cur, next).data);
            }
        }
        let logits = conv3d_in(
            &cur,
            &self.head.w,
            &self.head.b,
            self.head.classes,
            [1, 1, 1],
            [1, 1, 1],
            arena,
        );
        arena.give(cur.data);
        logits
    }
}

fn build_plain_encoder(
    cfg: &ModelConfig,
    in_channels: usize,
    tensors: &mut Tensors,
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
    tensors: &mut Tensors,
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
                    let w = tensors.take(&format!("{prefix}.skip.1.conv.weight"))?;
                    let g = tensors.take(&format!("{prefix}.skip.1.norm.weight"))?;
                    let be = tensors.take(&format!("{prefix}.skip.1.norm.bias"))?;
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
                        w: w.data,
                        gamma: g.data,
                        beta: be.data,
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
fn run_block(blk: &ConvBlock, x: &Act, arena: &mut Arena) -> Act {
    let mut y = conv3d_in(x, &blk.w, &blk.b, blk.cout, blk.kernel, blk.stride, arena);
    instance_norm_lrelu(&mut y, &blk.gamma, &blk.beta);
    y
}

/// One residual block: conv1 (with its LeakyReLU), conv2 (norm only), the
/// skip path added, LeakyReLU.
fn run_res_block(blk: &ResBlock, x: &Act, arena: &mut Arena) -> Act {
    let h = run_block(&blk.conv1, x, arena);
    let mut y = conv3d_in(
        &h,
        &blk.conv2.w,
        &blk.conv2.b,
        blk.conv2.cout,
        blk.conv2.kernel,
        blk.conv2.stride,
        arena,
    );
    arena.give(h.data);
    instance_norm(&mut y, &blk.conv2.gamma, &blk.conv2.beta);
    let residual: Option<Act> = match &blk.skip {
        Skip::Identity => None,
        Skip::Pool { kernel } => Some(avg_pool3d_in(x, *kernel, arena)),
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
                Some(avg_pool3d_in(x, *kernel, arena))
            };
            let src = pooled.as_ref().unwrap_or(x);
            let zero_bias = vec![0.0f32; *cout];
            let mut p = conv3d_in(src, w, &zero_bias, *cout, [1, 1, 1], [1, 1, 1], arena);
            if let Some(pooled) = pooled {
                arena.give(pooled.data);
            }
            instance_norm(&mut p, gamma, beta);
            Some(p)
        }
    };
    add_lrelu(&mut y, residual.as_ref().unwrap_or(x));
    if let Some(r) = residual {
        arena.give(r.data);
    }
    y
}

fn load_block(
    tensors: &mut Tensors,
    prefix: &str,
    cin: usize,
    cout: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
) -> Result<ConvBlock> {
    let w = tensors.take(&format!("{prefix}.conv.weight"))?;
    let b = tensors.take(&format!("{prefix}.conv.bias"))?;
    let g = tensors.take(&format!("{prefix}.norm.weight"))?;
    let be = tensors.take(&format!("{prefix}.norm.bias"))?;
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
        w: w.data,
        b: b.data,
        gamma: g.data,
        beta: be.data,
        cin,
        cout,
        kernel,
        stride,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::autoseg::config::Norm;

    fn rngf(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 * 0.2 - 0.1
    }

    fn tensor(seed: &mut u64, shape: &[usize]) -> WTensor {
        let n: usize = shape.iter().product();
        WTensor {
            shape: shape.to_vec(),
            data: (0..n).map(|_| rngf(seed)).collect(),
        }
    }

    fn conv_norm(t: &mut HashMap<String, WTensor>, s: &mut u64, p: &str, cout: usize, cin: usize) {
        t.insert(format!("{p}.conv.weight"), tensor(s, &[cout, cin, 3, 3, 3]));
        t.insert(format!("{p}.conv.bias"), tensor(s, &[cout]));
        t.insert(format!("{p}.norm.weight"), tensor(s, &[cout]));
        t.insert(format!("{p}.norm.bias"), tensor(s, &[cout]));
    }

    /// A miniature network of either family in the checkpoint's naming;
    /// the last encoder stride is `[1, 2, 2]`, so the decoder's first
    /// upsampling takes the general transposed convolution.
    pub(crate) fn tiny(residual: bool) -> (ModelConfig, HashMap<String, WTensor>) {
        tiny_with(residual, 3)
    }

    /// [`tiny`] with `classes` classes.
    pub(crate) fn tiny_with(
        residual: bool,
        classes: usize,
    ) -> (ModelConfig, HashMap<String, WTensor>) {
        let features = vec![4usize, 8, 8];
        let blocks = vec![1usize, 2, 1];
        let strides = vec![[1, 1, 1], [2, 2, 2], [1, 2, 2]];
        let cfg = ModelConfig {
            arch: if residual {
                Arch::ResidualEncoder {
                    blocks_per_stage: blocks.clone(),
                }
            } else {
                Arch::PlainConv
            },
            conv_bias: true,
            patch_size: [8, 12, 12],
            spacing: [3.0, 3.0, 3.0],
            features: features.clone(),
            kernels: vec![[3, 3, 3]; 3],
            strides,
            n_conv_per_stage: vec![2, 2, 2],
            n_conv_per_stage_decoder: vec![2, 1],
            decoder_kernels: vec![],
            norm: Norm::Ct,
            clip_lo: -1.0,
            clip_hi: 1.0,
            mean: 0.0,
            std: 1.0,
        };
        let mut s = if residual { 11u64 } else { 5u64 };
        let mut t = HashMap::new();
        if residual {
            conv_norm(&mut t, &mut s, "encoder.stem.convs.0", features[0], 1);
        }
        for (st, &f) in features.iter().enumerate() {
            if residual {
                let cin_stage = if st == 0 { f } else { features[st - 1] };
                for b in 0..blocks[st] {
                    let cin = if b == 0 { cin_stage } else { f };
                    let p = format!("encoder.stages.{st}.blocks.{b}");
                    conv_norm(&mut t, &mut s, &format!("{p}.conv1"), f, cin);
                    conv_norm(&mut t, &mut s, &format!("{p}.conv2"), f, f);
                    if cin != f {
                        t.insert(
                            format!("{p}.skip.1.conv.weight"),
                            tensor(&mut s, &[f, cin, 1, 1, 1]),
                        );
                        t.insert(format!("{p}.skip.1.norm.weight"), tensor(&mut s, &[f]));
                        t.insert(format!("{p}.skip.1.norm.bias"), tensor(&mut s, &[f]));
                    }
                }
            } else {
                let cin_stage = if st == 0 { 1 } else { features[st - 1] };
                for i in 0..2 {
                    let cin = if i == 0 { cin_stage } else { f };
                    conv_norm(
                        &mut t,
                        &mut s,
                        &format!("encoder.stages.{st}.0.convs.{i}"),
                        f,
                        cin,
                    );
                }
            }
        }
        for tr in 0..2 {
            let c_below = features[2 - tr];
            let c_skip = features[1 - tr];
            let st = cfg.strides[2 - tr];
            t.insert(
                format!("decoder.transpconvs.{tr}.weight"),
                tensor(&mut s, &[c_below, c_skip, st[0], st[1], st[2]]),
            );
            t.insert(
                format!("decoder.transpconvs.{tr}.bias"),
                tensor(&mut s, &[c_skip]),
            );
            for i in 0..cfg.n_conv_per_stage_decoder[tr] {
                let cin = if i == 0 { 2 * c_skip } else { c_skip };
                conv_norm(
                    &mut t,
                    &mut s,
                    &format!("decoder.stages.{tr}.convs.{i}"),
                    c_skip,
                    cin,
                );
            }
            t.insert(
                format!("decoder.seg_layers.{tr}.weight"),
                tensor(&mut s, &[classes, c_skip, 1, 1, 1]),
            );
            t.insert(
                format!("decoder.seg_layers.{tr}.bias"),
                tensor(&mut s, &[classes]),
            );
        }
        (cfg, t)
    }

    pub(crate) fn patch(seed: u64, p: [usize; 3]) -> Act {
        let mut s = seed;
        let mut x = Act::zeros(1, p[0], p[1], p[2]);
        for v in &mut x.data {
            *v = rngf(&mut s) * 10.0;
        }
        x
    }

    fn bits(a: &Act) -> Vec<u32> {
        a.data.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn moved_weights_and_reused_buffers_give_the_same_logits_to_the_bit() {
        for residual in [false, true] {
            let (cfg, t) = tiny(residual);
            let p = cfg.patch_size;
            let copied = UNet::build(cfg.clone(), &t).unwrap();
            let moved = UNet::build_owned(cfg, t).unwrap();
            let mut arena = Arena::default();
            for seed in [3u64, 9, 27] {
                let x = patch(seed, p);
                let fresh = copied.forward_cpu(&x);
                // The arena carries the last patch's buffers, stale values
                // and all, into this one.
                let reused = moved.forward_cpu_in(&x, &mut arena);
                assert_eq!(
                    (reused.c, reused.d, reused.h, reused.w),
                    (3, p[0], p[1], p[2])
                );
                assert_eq!(
                    bits(&fresh),
                    bits(&reused),
                    "residual {residual}, seed {seed}"
                );
                arena.give(reused.data);
            }
        }
    }

    #[test]
    fn a_tensor_taken_twice_from_an_owned_map_is_reported_missing() {
        let (cfg, t) = tiny(false);
        let mut src = Tensors::Owned(t);
        assert!(src.take("decoder.seg_layers.1.bias").is_ok());
        let err = match src.take("decoder.seg_layers.1.bias") {
            Ok(_) => panic!("taken twice"),
            Err(e) => e,
        };
        assert!(format!("{err:#}").contains("missing"), "{err:#}");
        assert!(UNet::build_owned(cfg, HashMap::new()).is_err());
    }
}
