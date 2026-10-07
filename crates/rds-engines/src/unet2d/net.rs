//! lungmask's U-Net (`lungmask/resunet.py`), as `mask.py` builds it:
//! depth 5, 64 base filters, padded 3 x 3 convolutions, batch norm *after*
//! the ReLU, average pooling on the way down, bilinear upsampling plus a
//! 1 x 1 convolution on the way up, no residual paths. The checkpoint also
//! carries the residual branch's weights (the module creates them whether
//! they are used or not); they are not read.
//!
//! Written once against `burn`, so it runs on the GPU through wgpu and on
//! the pure-Rust CPU backend alike (its convolutions through
//! [`crate::nn::fastconv`]).

use anyhow::{Context, Result};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::module::avg_pool2d;
use burn::tensor::Tensor;

use crate::medsam2::ops;
use crate::nn::fastconv;
use crate::nn::params::Params;

/// Depth of the network and the width of its first stage (`2 ** wf`) as
/// lungmask builds it; [`Unet2d::load`] reads both from the checkpoint.
pub const DEPTH: usize = 5;
pub const BASE: usize = 64;
/// `nn.BatchNorm2d`'s default eps.
const BN_EPS: f32 = 1e-5;

/// One Conv 3 x 3 (padded) → ReLU → BatchNorm layer as plain data, the
/// norm folded into a per-channel scale and shift.
pub struct ConvBnW {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub scale: Vec<f32>,
    pub shift: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub k: usize,
}

impl ConvBnW {
    fn load(p: &Params, conv: &str, bn: &str, cin: usize, cout: usize, k: usize) -> Result<Self> {
        let (w, b) = p.conv2d(conv, cout, cin, k, 1)?;
        let gamma = p.vec(&format!("{bn}.weight"), cout)?;
        let beta = p.vec(&format!("{bn}.bias"), cout)?;
        let mean = p.vec(&format!("{bn}.running_mean"), cout)?;
        let var = p.vec(&format!("{bn}.running_var"), cout)?;
        let scale: Vec<f32> = gamma
            .iter()
            .zip(var)
            .map(|(g, v)| g / (v + BN_EPS).sqrt())
            .collect();
        let shift: Vec<f32> = beta
            .iter()
            .zip(mean)
            .zip(&scale)
            .map(|((b, m), s)| b - m * s)
            .collect();
        Ok(ConvBnW {
            w: w.to_vec(),
            b: b.to_vec(),
            scale,
            shift,
            cin,
            cout,
            k,
        })
    }
}

/// `UNetConvBlock`: two layers.
pub type BlockW = [ConvBnW; 2];

fn load_block(p: &Params, prefix: &str, cin: usize, cout: usize) -> Result<BlockW> {
    Ok([
        ConvBnW::load(
            p,
            &format!("{prefix}.block.0"),
            &format!("{prefix}.block.2"),
            cin,
            cout,
            3,
        )?,
        ConvBnW::load(
            p,
            &format!("{prefix}.block.3"),
            &format!("{prefix}.block.5"),
            cout,
            cout,
            3,
        )?,
    ])
}

/// `UNetUpBlock` in `upsample` mode: bilinear x2, a 1 x 1 conv
/// (`cin` → `cout`), the skip concatenated after it, then a block.
pub struct UpW {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
    pub block: BlockW,
}

/// The whole network as plain data, read once from the state dict; the
/// burn network and the CPU one ([`super::cpu`]) are both built from it.
pub struct Weights {
    pub down: Vec<BlockW>,
    pub up: Vec<UpW>,
    pub last_w: Vec<f32>,
    pub last_b: Vec<f32>,
    pub last_cin: usize,
    pub classes: usize,
}

impl Weights {
    /// The class count is read from the last layer, as `mask.get_model`
    /// does; the width and the depth from the down path.
    pub fn load(p: &Params) -> Result<Weights> {
        let classes = p
            .shape("last.bias")
            .and_then(|s| s.first().copied())
            .context("checkpoint has no last.bias")?;
        let base = p
            .shape("down_path.0.block.0.weight")
            .and_then(|s| s.first().copied())
            .context("checkpoint has no down_path.0")?;
        let depth = (0..)
            .take_while(|i| p.contains(&format!("down_path.{i}.block.0.weight")))
            .count();
        let mut down = Vec::with_capacity(depth);
        let mut cin = 1;
        for i in 0..depth {
            let cout = base << i;
            down.push(load_block(p, &format!("down_path.{i}"), cin, cout)?);
            cin = cout;
        }
        let mut up = Vec::with_capacity(depth - 1);
        for (j, i) in (0..depth - 1).rev().enumerate() {
            let cout = base << i;
            let (w, b) = p.conv2d(&format!("up_path.{j}.up.1"), cout, cin, 1, 1)?;
            up.push(UpW {
                w: w.to_vec(),
                b: b.to_vec(),
                cin,
                cout,
                block: load_block(p, &format!("up_path.{j}.conv_block"), cin, cout)?,
            });
            cin = cout;
        }
        let (w, b) = p.conv2d("last", classes, cin, 1, 1)?;
        Ok(Weights {
            down,
            up,
            last_w: w.to_vec(),
            last_b: b.to_vec(),
            last_cin: cin,
            classes,
        })
    }
}

/// Conv 3 x 3 (padded) → ReLU → BatchNorm, on a device.
struct ConvBn<B: Backend> {
    w: Tensor<B, 4>,
    b: Tensor<B, 1>,
    scale: Tensor<B, 4>,
    shift: Tensor<B, 4>,
}

impl<B: Backend> ConvBn<B> {
    fn upload(l: &ConvBnW, dev: &B::Device) -> ConvBn<B> {
        ConvBn {
            w: ops::from_slice(&l.w, [l.cout, l.cin, l.k, l.k], dev),
            b: ops::from_slice(&l.b, [l.cout], dev),
            scale: ops::from_slice(&l.scale, [1, l.cout, 1, 1], dev),
            shift: ops::from_slice(&l.shift, [1, l.cout, 1, 1], dev),
        }
    }

    fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let pad = self.w.dims()[2] / 2;
        let y = relu(fastconv::conv2d(x, &self.w, Some(&self.b), 1, pad));
        y * self.scale.clone() + self.shift.clone()
    }
}

/// `UNetConvBlock`: two [`ConvBn`]s.
struct Block<B: Backend> {
    a: ConvBn<B>,
    b: ConvBn<B>,
}

impl<B: Backend> Block<B> {
    fn upload(w: &BlockW, dev: &B::Device) -> Self {
        Block {
            a: ConvBn::upload(&w[0], dev),
            b: ConvBn::upload(&w[1], dev),
        }
    }

    fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.b.apply(self.a.apply(x))
    }
}

/// An up step on a device.
struct Up<B: Backend> {
    w: Tensor<B, 4>,
    b: Tensor<B, 1>,
    block: Block<B>,
}

/// The network with its weights on a device.
pub struct Unet2d<B: Backend> {
    down: Vec<Block<B>>,
    up: Vec<Up<B>>,
    last_w: Tensor<B, 4>,
    last_b: Tensor<B, 1>,
    classes: usize,
    device: B::Device,
}

impl<B: Backend> Unet2d<B> {
    /// Assemble from a lungmask state dict.
    pub fn load(p: &Params, dev: &B::Device) -> Result<Unet2d<B>> {
        Ok(Self::upload(&Weights::load(p)?, dev))
    }

    pub fn upload(w: &Weights, dev: &B::Device) -> Unet2d<B> {
        Unet2d {
            down: w.down.iter().map(|b| Block::upload(b, dev)).collect(),
            up: w
                .up
                .iter()
                .map(|u| Up {
                    w: ops::from_slice(&u.w, [u.cout, u.cin, 1, 1], dev),
                    b: ops::from_slice(&u.b, [u.cout], dev),
                    block: Block::upload(&u.block, dev),
                })
                .collect(),
            last_w: ops::from_slice(&w.last_w, [w.classes, w.last_cin, 1, 1], dev),
            last_b: ops::from_slice(&w.last_b, [w.classes], dev),
            classes: w.classes,
            device: dev.clone(),
        }
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    /// Logits `[n, classes, h, w]` for slices `[n, 1, h, w]` (h and w
    /// multiples of 16). The reference ends in a `LogSoftmax`, which the
    /// argmax the caller takes does not need.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut skips = Vec::with_capacity(self.down.len());
        let mut x = x;
        for (i, d) in self.down.iter().enumerate() {
            x = d.apply(x);
            if i + 1 != self.down.len() {
                skips.push(x.clone());
                x = avg_pool2d(x, [2, 2], [2, 2], [0, 0], true, false);
            }
        }
        for u in &self.up {
            let [_, _, h, w] = x.dims();
            let up = ops::resize_bilinear(x, [h * 2, w * 2]);
            let up = fastconv::conv2d(up, &u.w, Some(&u.b), 1, 0);
            let skip = skips.pop().expect("one skip per up block");
            x = u.block.apply(Tensor::cat(vec![up, skip], 1));
        }
        fastconv::conv2d(x, &self.last_w, Some(&self.last_b), 1, 0)
    }

    /// The label (argmax over classes, first maximum on a tie) of every
    /// pixel of `slices`, `n * h * w` values of `[n][h][w]`.
    pub fn labels(&self, slices: &[f32], n: usize, h: usize, w: usize) -> Vec<u8> {
        let x = ops::from_slice::<B, 4>(slices, [n, 1, h, w], &self.device);
        let logits = ops::to_vec(self.forward(x));
        let plane = h * w;
        let mut out = vec![0u8; n * plane];
        for s in 0..n {
            let base = s * self.classes * plane;
            for (v, o) in out[s * plane..(s + 1) * plane].iter_mut().enumerate() {
                let mut best = 0;
                let mut best_v = f32::NEG_INFINITY;
                for c in 0..self.classes {
                    let l = logits[base + c * plane + v];
                    if l > best_v {
                        best_v = l;
                        best = c;
                    }
                }
                *o = best as u8;
            }
        }
        out
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::nn::cache::load_safetensors;

    type B = burn::backend::NdArray;

    /// One family's tensors out of the PyTorch fixture, with the family
    /// prefix dropped.
    pub(crate) fn fixture(prefix: &str) -> Params {
        let all = load_safetensors(std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/data/zoo-nets.safetensors"
        )))
        .expect("tests/data/zoo-nets.safetensors");
        let p = format!("{prefix}.");
        Params::new(
            all.into_iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|k| (k.to_string(), v)))
                .collect(),
        )
    }

    /// Worst relative difference.
    pub(crate) fn worst(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0, f32::max)
    }

    #[test]
    fn the_unet_matches_pytorch() {
        let p = fixture("unet2d");
        let dev = Default::default();
        let net = Unet2d::<B>::load(&p, &dev).unwrap();
        assert_eq!(net.classes(), 3);
        let x = p.get("__input", &[2, 1, 32, 32]).unwrap();
        let want = p.get("__output", &[2, 3, 32, 32]).unwrap();
        let got = ops::to_vec(net.forward(ops::from_slice(x, [2, 1, 32, 32], &dev)));
        // The reference ends in LogSoftmax: compare after the same shift.
        let mut g = got.clone();
        for s in 0..2 {
            for v in 0..1024 {
                let ls: Vec<f32> = (0..3).map(|c| got[(s * 3 + c) * 1024 + v]).collect();
                let m = ls.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = m + ls.iter().map(|l| (l - m).exp()).sum::<f32>().ln();
                for c in 0..3 {
                    g[(s * 3 + c) * 1024 + v] = ls[c] - lse;
                }
            }
        }
        let w = worst(&g, want);
        assert!(w < 1e-4, "relative error {w:e}");
        // And the labels are the argmax.
        let labels = net.labels(x, 2, 32, 32);
        let mut agree = 0;
        for (v, &label) in labels.iter().enumerate().take(2048) {
            let (s, i) = (v / 1024, v % 1024);
            let best = (0..3)
                .max_by(|a, b| {
                    want[(s * 3 + a) * 1024 + i].total_cmp(&want[(s * 3 + b) * 1024 + i])
                })
                .unwrap();
            agree += (label as usize == best) as usize;
        }
        assert!(agree >= 2046, "{agree} of 2048 labels agree");
    }
}
