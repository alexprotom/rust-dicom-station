//! MONAI's SegResNet family (Myronenko, *3D MRI brain tumor segmentation
//! using autoencoder regularization*, 2018), as `monai.networks.nets`
//! defines it, written once against `burn`:
//!
//! * [`SegResNet`] - `SegResNet`: a 3 x 3 x 3 stem, residual blocks
//!   (norm, ReLU, conv, norm, ReLU, conv, plus the input) at doubling
//!   widths with strided-conv downsampling, a decoder of 1 x 1 x 1 conv,
//!   trilinear x2 upsampling, the skip added and residual blocks, and a
//!   final norm, ReLU and 1 x 1 x 1 conv. GroupNorm with 8 groups; the
//!   convolutions carry no bias but the last. The MONAI whole-body bundle.
//! * [`SegResNetDs`] - `SegResNetDS`: the same blocks with BatchNorm (or
//!   InstanceNorm), a transposed-conv decoder and a 1 x 1 x 1 head per
//!   level, of which inference uses the finest. CT-FM's whole-body model,
//!   and with a second decoder (`SegResNetDS2`) VISTA-3D's image encoder.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::medsam2::ops;
use crate::nn::fastconv::{self, Activation};
use crate::nn::params::Params;

/// A normalization layer of the family, for a backend: its parameters per
/// channel, applied by [`fastconv::group_norm_act`] and
/// [`fastconv::affine_act`].
pub struct Norm<B: Backend> {
    params: NormParams,
    _backend: std::marker::PhantomData<B>,
}

/// The normalization layers the family uses.
enum NormParams {
    /// `GroupNorm(groups, C)`, affine.
    Group {
        groups: usize,
        gamma: Vec<f32>,
        beta: Vec<f32>,
    },
    /// `InstanceNorm3d(C, affine=True)`.
    Instance { gamma: Vec<f32>, beta: Vec<f32> },
    /// `BatchNorm3d` in eval mode, folded into a per-channel affine map.
    Batch { scale: Vec<f32>, shift: Vec<f32> },
}

/// Which normalization a network was built with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormKind {
    Group(usize),
    Instance,
    Batch,
}

const EPS: f64 = 1e-5;

impl<B: Backend> Norm<B> {
    pub fn load(
        p: &Params,
        prefix: &str,
        c: usize,
        kind: NormKind,
        dev: &B::Device,
    ) -> Result<Self> {
        let w = p.vec(&format!("{prefix}.weight"), c)?;
        let b = p.vec(&format!("{prefix}.bias"), c)?;
        let _ = dev;
        let t = |v: &[f32]| v.to_vec();
        let params = match kind {
            NormKind::Group(groups) => {
                if !c.is_multiple_of(groups) {
                    bail!("{prefix}: {c} channels do not split into {groups} groups");
                }
                NormParams::Group {
                    groups,
                    gamma: t(w),
                    beta: t(b),
                }
            }
            NormKind::Instance => NormParams::Instance {
                gamma: t(w),
                beta: t(b),
            },
            NormKind::Batch => {
                let mean = p.vec(&format!("{prefix}.running_mean"), c)?;
                let var = p.vec(&format!("{prefix}.running_var"), c)?;
                let scale: Vec<f32> = w
                    .iter()
                    .zip(var)
                    .map(|(g, v)| g / (v + EPS as f32).sqrt())
                    .collect();
                let shift: Vec<f32> = b
                    .iter()
                    .zip(mean)
                    .zip(&scale)
                    .map(|((b, m), s)| b - m * s)
                    .collect();
                NormParams::Batch {
                    scale: t(&scale),
                    shift: t(&shift),
                }
            }
        };
        Ok(Norm {
            params,
            _backend: std::marker::PhantomData,
        })
    }

    pub fn apply(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        self.apply_act(x, Activation::None)
    }

    /// The normalization followed by `act`, fused on the CPU backend.
    pub fn apply_act(&self, x: Tensor<B, 5>, act: Activation) -> Tensor<B, 5> {
        match &self.params {
            NormParams::Group {
                groups,
                gamma,
                beta,
            } => fastconv::group_norm_act(x, *groups, Some(gamma), Some(beta), EPS, act),
            NormParams::Instance { gamma, beta } => {
                let c = x.dims()[1];
                fastconv::group_norm_act(x, c, Some(gamma), Some(beta), EPS, act)
            }
            NormParams::Batch { scale, shift } => fastconv::affine_act(x, scale, shift, act),
        }
    }
}

/// A 3-D convolution, padded `kernel / 2`.
pub struct Conv<B: Backend> {
    pub w: Tensor<B, 5>,
    pub b: Option<Tensor<B, 1>>,
    pub stride: usize,
    pub k: usize,
}

impl<B: Backend> Conv<B> {
    /// `{prefix}.weight` `[cout, cin, k, k, k]`, and `{prefix}.bias` when
    /// `bias`.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        p: &Params,
        prefix: &str,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        bias: bool,
        dev: &B::Device,
    ) -> Result<Self> {
        let w = p.get(&format!("{prefix}.weight"), &[cout, cin, k, k, k])?;
        let b = if bias {
            Some(ops::from_slice(
                p.vec(&format!("{prefix}.bias"), cout)?,
                [cout],
                dev,
            ))
        } else {
            None
        };
        Ok(Conv {
            w: ops::from_slice(w, [cout, cin, k, k, k], dev),
            b,
            stride,
            k,
        })
    }

    pub fn apply(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        fastconv::conv3d(x, &self.w, self.b.as_ref(), self.stride, self.k / 2)
    }
}

/// `ResBlock` / `SegResBlock`: norm, ReLU, conv, norm, ReLU, conv, plus
/// the input.
pub struct ResBlock<B: Backend> {
    norm1: Norm<B>,
    conv1: Conv<B>,
    norm2: Norm<B>,
    conv2: Conv<B>,
}

impl<B: Backend> ResBlock<B> {
    /// `conv` is the infix of the convolutions' keys: `.conv` for
    /// `SegResNet` (its convolutions are MONAI `Convolution` blocks),
    /// empty for `SegResNetDS` (plain `Conv3d`).
    pub fn load(
        p: &Params,
        prefix: &str,
        c: usize,
        norm: NormKind,
        conv: &str,
        dev: &B::Device,
    ) -> Result<Self> {
        Ok(ResBlock {
            norm1: Norm::load(p, &format!("{prefix}.norm1"), c, norm, dev)?,
            conv1: Conv::load(p, &format!("{prefix}.conv1{conv}"), c, c, 3, 1, false, dev)?,
            norm2: Norm::load(p, &format!("{prefix}.norm2"), c, norm, dev)?,
            conv2: Conv::load(p, &format!("{prefix}.conv2{conv}"), c, c, 3, 1, false, dev)?,
        })
    }

    pub fn apply(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let y = self
            .conv1
            .apply(self.norm1.apply_act(x.clone(), Activation::Relu));
        let y = self.conv2.apply(self.norm2.apply_act(y, Activation::Relu));
        y + x
    }
}

/// Linear (trilinear) x2 upsampling with `align_corners=False`: every
/// axis doubled, the new samples a quarter and three quarters of the way
/// between neighbours, the ends clamped - PyTorch's `interpolate`. Direct
/// loops on the CPU backend, burn's operations elsewhere.
pub fn upsample_linear_2x<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    match fastconv::upsample_linear_2x_cpu(&x) {
        Some(y) => y,
        None => upsample_linear_2x_burn(x),
    }
}

/// [`upsample_linear_2x`] in burn's operations, on any backend.
pub fn upsample_linear_2x_burn<B: Backend>(x: Tensor<B, 5>) -> Tensor<B, 5> {
    let mut x = x;
    for axis in 2..5 {
        let dims = x.dims();
        let n = dims[axis];
        let prev = if n > 1 {
            Tensor::cat(
                vec![
                    x.clone().narrow(axis, 0, 1),
                    x.clone().narrow(axis, 0, n - 1),
                ],
                axis,
            )
        } else {
            x.clone()
        };
        let next = if n > 1 {
            Tensor::cat(
                vec![
                    x.clone().narrow(axis, 1, n - 1),
                    x.clone().narrow(axis, n - 1, 1),
                ],
                axis,
            )
        } else {
            x.clone()
        };
        let even = x.clone() * 0.75 + prev * 0.25;
        let odd = x * 0.75 + next * 0.25;
        let both: Tensor<B, 6> = Tensor::stack(vec![even, odd], axis + 1);
        let mut out = dims;
        out[axis] = 2 * n;
        x = both.reshape(out);
    }
    x
}

/// One encoder level of [`SegResNet`]: the strided conv into it (none at
/// level 0) and its blocks.
type Level<B> = (Option<Conv<B>>, Vec<ResBlock<B>>);

/// One encoder level of [`SegResNetDs`]: its blocks and the strided conv to
/// the next (none at the last).
type DsLevel<B> = (Vec<ResBlock<B>>, Option<Conv<B>>);

/// MONAI `SegResNet` (`blocks_up` one per level, `upsample_mode`
/// non-trainable).
pub struct SegResNet<B: Backend> {
    conv_init: Conv<B>,
    /// Per level: the strided conv (none at level 0) and the blocks.
    down: Vec<Level<B>>,
    /// Per decoder level: the 1 x 1 x 1 conv before the upsampling, and the
    /// blocks after the skip.
    up: Vec<(Conv<B>, Vec<ResBlock<B>>)>,
    final_norm: Norm<B>,
    final_conv: Conv<B>,
    classes: usize,
}

impl<B: Backend> SegResNet<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        p: &Params,
        init_filters: usize,
        blocks_down: &[usize],
        blocks_up: &[usize],
        classes: usize,
        norm: NormKind,
        dev: &B::Device,
    ) -> Result<Self> {
        let f = init_filters;
        let conv_init = Conv::load(p, "convInit.conv", 1, f, 3, 1, false, dev)?;
        let mut down = Vec::new();
        for (i, &n) in blocks_down.iter().enumerate() {
            let c = f << i;
            let pre = if i > 0 {
                Some(Conv::load(
                    p,
                    &format!("down_layers.{i}.0.conv"),
                    c / 2,
                    c,
                    3,
                    2,
                    false,
                    dev,
                )?)
            } else {
                None
            };
            let blocks = (0..n)
                .map(|b| {
                    ResBlock::load(
                        p,
                        &format!("down_layers.{i}.{}", b + 1),
                        c,
                        norm,
                        ".conv",
                        dev,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            down.push((pre, blocks));
        }
        let n_up = blocks_up.len();
        let mut up = Vec::new();
        for (i, &n) in blocks_up.iter().enumerate() {
            let cin = f << (n_up - i);
            let conv = Conv::load(
                p,
                &format!("up_samples.{i}.0.conv"),
                cin,
                cin / 2,
                1,
                1,
                false,
                dev,
            )?;
            let blocks = (0..n)
                .map(|b| {
                    ResBlock::load(
                        p,
                        &format!("up_layers.{i}.{b}"),
                        cin / 2,
                        norm,
                        ".conv",
                        dev,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            up.push((conv, blocks));
        }
        Ok(SegResNet {
            conv_init,
            down,
            up,
            final_norm: Norm::load(p, "conv_final.0", f, norm, dev)?,
            final_conv: Conv::load(p, "conv_final.2.conv", f, classes, 1, 1, true, dev)?,
            classes,
        })
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    /// Logits `[1, classes, d, h, w]` for an input `[1, 1, d, h, w]`.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let mut x = self.conv_init.apply(x);
        let mut skips = Vec::with_capacity(self.down.len());
        for (pre, blocks) in &self.down {
            if let Some(c) = pre {
                x = c.apply(x);
            }
            for b in blocks {
                x = b.apply(x);
            }
            skips.push(x.clone());
        }
        skips.pop();
        for (conv, blocks) in &self.up {
            x = upsample_linear_2x(conv.apply(x)) + skips.pop().expect("a skip per level");
            for b in blocks {
                x = b.apply(x);
            }
        }
        self.final_conv
            .apply(self.final_norm.apply_act(x, Activation::Relu))
    }
}

/// One decoder level of `SegResNetDS`.
struct DsUp<B: Backend> {
    /// `ConvTranspose3d(2c, c, 3, stride 2, padding 1, output_padding 1)`,
    /// no bias.
    w: Tensor<B, 5>,
    blocks: Vec<ResBlock<B>>,
    head: Option<Conv<B>>,
}

/// MONAI `SegResNetDS` (isotropic: no `resolution`), and the decoders of
/// `SegResNetDS2`.
pub struct SegResNetDs<B: Backend> {
    conv_init: Conv<B>,
    /// Per level: blocks, then the strided conv to the next (none at the
    /// last).
    levels: Vec<DsLevel<B>>,
    up: Vec<DsUp<B>>,
    /// A second decoder (`up_layers_auto` of `SegResNetDS2`).
    up_auto: Vec<DsUp<B>>,
    classes: usize,
}

impl<B: Backend> SegResNetDs<B> {
    /// Load an encoder and one or both decoders. `prefix` is prepended to
    /// every key (`image_encoder.` for VISTA-3D); `point` and `auto` name
    /// the decoder lists to read (`up_layers`, and `up_layers_auto` of
    /// `SegResNetDS2`); `head` whether the finest level of each carries its
    /// 1 x 1 x 1 head.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        p: &Params,
        prefix: &str,
        init_filters: usize,
        blocks_down: &[usize],
        classes: usize,
        norm: NormKind,
        point: Option<&str>,
        auto: Option<&str>,
        head: bool,
        dev: &B::Device,
    ) -> Result<Self> {
        let f = init_filters;
        let conv_init = Conv::load(
            p,
            &format!("{prefix}encoder.conv_init"),
            1,
            f,
            3,
            1,
            false,
            dev,
        )?;
        let mut levels = Vec::new();
        for (i, &n) in blocks_down.iter().enumerate() {
            let c = f << i;
            let blocks = (0..n)
                .map(|b| {
                    ResBlock::load(
                        p,
                        &format!("{prefix}encoder.layers.{i}.blocks.{b}"),
                        c,
                        norm,
                        "",
                        dev,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let down = if i + 1 < blocks_down.len() {
                Some(Conv::load(
                    p,
                    &format!("{prefix}encoder.layers.{i}.downsample"),
                    c,
                    2 * c,
                    3,
                    2,
                    false,
                    dev,
                )?)
            } else {
                None
            };
            levels.push((blocks, down));
        }
        let n_up = blocks_down.len() - 1;
        let decoder = |name: &str| -> Result<Vec<DsUp<B>>> {
            let mut out = Vec::new();
            let mut c = f << n_up;
            for i in 0..n_up {
                c /= 2;
                let w = p.get(
                    &format!("{prefix}{name}.{i}.upsample.deconv.weight"),
                    &[2 * c, c, 3, 3, 3],
                )?;
                let blocks = vec![ResBlock::load(
                    p,
                    &format!("{prefix}{name}.{i}.blocks.0"),
                    c,
                    norm,
                    "",
                    dev,
                )?];
                let head = if head && i + 1 == n_up {
                    Some(Conv::load(
                        p,
                        &format!("{prefix}{name}.{i}.head"),
                        c,
                        classes,
                        1,
                        1,
                        true,
                        dev,
                    )?)
                } else {
                    None
                };
                out.push(DsUp {
                    w: ops::from_slice(w, [2 * c, c, 3, 3, 3], dev),
                    blocks,
                    head,
                });
            }
            Ok(out)
        };
        let up = match point {
            Some(name) => decoder(name)?,
            None => Vec::new(),
        };
        let up_auto = match auto {
            Some(name) => decoder(name)?,
            None => Vec::new(),
        };
        Ok(SegResNetDs {
            conv_init,
            levels,
            up,
            up_auto,
            classes,
        })
    }

    pub fn classes(&self) -> usize {
        self.classes
    }

    /// The encoder's outputs, coarsest first.
    pub fn encode(&self, x: Tensor<B, 5>) -> Vec<Tensor<B, 5>> {
        let mut x = self.conv_init.apply(x);
        let mut outs = Vec::with_capacity(self.levels.len());
        for (blocks, down) in &self.levels {
            for b in blocks {
                x = b.apply(x);
            }
            outs.push(x.clone());
            if let Some(d) = down {
                x = d.apply(x);
            }
        }
        outs.reverse();
        outs
    }

    /// Run one decoder over the encoder's outputs; the finest level's
    /// features, and its head's logits when it has one.
    fn decode(up: &[DsUp<B>], skips: &[Tensor<B, 5>]) -> (Tensor<B, 5>, Option<Tensor<B, 5>>) {
        let mut x = skips[0].clone();
        let mut logits = None;
        for (i, level) in up.iter().enumerate() {
            x = fastconv::conv_transpose3d_k3s2(x, &level.w, None) + skips[i + 1].clone();
            for b in &level.blocks {
                x = b.apply(x);
            }
            if let Some(h) = &level.head {
                logits = Some(h.apply(x.clone()));
            }
        }
        (x, logits)
    }

    /// `SegResNetDS.forward` in eval mode: the finest head's logits.
    pub fn forward(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let skips = self.encode(x);
        Self::decode(&self.up, &skips)
            .1
            .expect("a SegResNetDS loaded with its head")
    }

    /// `SegResNetDS2.forward` with both branches: the finest outputs of
    /// the point branch (`up_layers`) and of the class branch
    /// (`up_layers_auto`) - logits when the decoders were loaded with
    /// heads, features otherwise.
    pub fn forward_both(&self, x: Tensor<B, 5>) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let skips = self.encode(x);
        let (p, ph) = Self::decode(&self.up, &skips);
        let (a, ah) = Self::decode(&self.up_auto, &skips);
        (ph.unwrap_or(p), ah.unwrap_or(a))
    }

    /// The point branch alone (VISTA-3D's point mode).
    pub fn forward_point(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let skips = self.encode(x);
        let (p, ph) = Self::decode(&self.up, &skips);
        ph.unwrap_or(p)
    }

    /// The class branch alone (VISTA-3D's automatic mode).
    pub fn forward_auto(&self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        let skips = self.encode(x);
        let (a, ah) = Self::decode(&self.up_auto, &skips);
        ah.unwrap_or(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unet2d::net::tests::{fixture, worst};

    type B = burn::backend::NdArray;

    #[test]
    fn segresnet_matches_monai() {
        let p = fixture("segresnet");
        let dev = Default::default();
        let net = SegResNet::<B>::load(
            &p,
            2,
            &[1, 2, 2, 4],
            &[1, 1, 1],
            3,
            NormKind::Group(2),
            &dev,
        )
        .unwrap();
        let x = p.get("__input", &[1, 1, 16, 16, 16]).unwrap();
        let want = p.get("__output", &[1, 3, 16, 16, 16]).unwrap();
        let got = ops::to_vec(net.forward(ops::from_slice(x, [1, 1, 16, 16, 16], &dev)));
        let w = worst(&got, want);
        assert!(w < 1e-4, "relative error {w:e}");
    }

    #[test]
    fn segresnet_ds_matches_monai() {
        let p = fixture("segresnetds");
        let dev = Default::default();
        let net = SegResNetDs::<B>::load(
            &p,
            "",
            2,
            &[1, 1, 1, 1, 1],
            3,
            NormKind::Batch,
            Some("up_layers"),
            None,
            true,
            &dev,
        )
        .unwrap();
        let x = p.get("__input", &[1, 1, 16, 16, 32]).unwrap();
        let want = p.get("__output", &[1, 3, 16, 16, 32]).unwrap();
        let got = ops::to_vec(net.forward(ops::from_slice(x, [1, 1, 16, 16, 32], &dev)));
        let w = worst(&got, want);
        assert!(w < 1e-4, "relative error {w:e}");
    }

    #[test]
    fn trilinear_doubling_is_pytorchs() {
        // F.interpolate([0, 1, 2, 3], scale_factor=2, mode="linear",
        // align_corners=False) = [0, .25, .75, 1.25, 1.75, 2.25, 2.75, 3]
        // on every axis; a ramp along the last axis shows it.
        let dev = Default::default();
        let x = ops::from_slice::<B, 5>(&[0.0, 1.0, 2.0, 3.0], [1, 1, 1, 1, 4], &dev);
        let y = ops::to_vec(upsample_linear_2x(x));
        assert_eq!(y.len(), 4 * 8);
        assert_eq!(&y[..8], &[0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0]);
    }
}
