//! The small parameterized pieces every part of SAM 2 is built from.
//!
//! Each one loads itself from the checkpoint by key prefix and asserts its
//! shapes on the way in, so a drifted checkpoint fails while the network is
//! being assembled rather than producing a plausible mask from a mis-shaped
//! tensor.

use anyhow::Result;
use burn::tensor::activation::{gelu, relu, sigmoid};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::nn::params::Params;

use super::ops::{self, EPS, EPS_6};

/// A `LayerNorm`, applied over the last axis or over the channel axis of an
/// `[n, c, h, w]` tensor depending on which method is called.
pub struct Norm<B: Backend> {
    pub weight: Tensor<B, 1>,
    pub bias: Tensor<B, 1>,
    pub eps: f64,
}

impl<B: Backend> Norm<B> {
    /// PyTorch's default eps, used everywhere SAM 2 writes `nn.LayerNorm`.
    pub fn load(p: &Params, prefix: &str, n: usize, dev: &B::Device) -> Result<Norm<B>> {
        Self::load_eps(p, prefix, n, EPS, dev)
    }

    /// The 1e-6 variant: the Hiera blocks, `LayerNorm2d` and the CXBlocks.
    pub fn load6(p: &Params, prefix: &str, n: usize, dev: &B::Device) -> Result<Norm<B>> {
        Self::load_eps(p, prefix, n, EPS_6, dev)
    }

    fn load_eps(p: &Params, prefix: &str, n: usize, eps: f64, dev: &B::Device) -> Result<Norm<B>> {
        let (w, b) = p.norm(prefix, n)?;
        Ok(Norm {
            weight: ops::from_slice(w, [n], dev),
            bias: ops::from_slice(b, [n], dev),
            eps,
        })
    }

    pub fn apply<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        ops::layer_norm(x, &self.weight, &self.bias, self.eps)
    }

    /// SAM 2's `LayerNorm2d` — statistics over the channel axis.
    pub fn apply_2d(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        ops::layer_norm_2d(x, &self.weight, &self.bias, self.eps)
    }
}

/// An `nn.Linear`.
pub struct Lin<B: Backend> {
    pub weight: Tensor<B, 2>,
    pub bias: Tensor<B, 1>,
}

impl<B: Backend> Lin<B> {
    pub fn load(
        p: &Params,
        prefix: &str,
        out: usize,
        inp: usize,
        dev: &B::Device,
    ) -> Result<Lin<B>> {
        let (w, b) = p.linear(prefix, out, inp)?;
        Ok(Lin {
            weight: ops::from_slice(w, [out, inp], dev),
            bias: ops::from_slice(b, [out], dev),
        })
    }

    pub fn apply<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        ops::linear(x, &self.weight, Some(&self.bias))
    }
}

/// SAM 2's `MLP` helper: `num_layers` linear layers, ReLU between them, and
/// optionally a sigmoid on the output (the IoU head).
pub struct Mlp<B: Backend> {
    layers: Vec<Lin<B>>,
    sigmoid_output: bool,
}

impl<B: Backend> Mlp<B> {
    pub fn load(p: &Params, prefix: &str, dims: &[usize], dev: &B::Device) -> Result<Mlp<B>> {
        Self::load_with(p, prefix, dims, false, dev)
    }

    pub fn load_with(
        p: &Params,
        prefix: &str,
        dims: &[usize],
        sigmoid_output: bool,
        dev: &B::Device,
    ) -> Result<Mlp<B>> {
        let mut layers = Vec::with_capacity(dims.len() - 1);
        for i in 0..dims.len() - 1 {
            layers.push(Lin::load(
                p,
                &format!("{prefix}.layers.{i}"),
                dims[i + 1],
                dims[i],
                dev,
            )?);
        }
        Ok(Mlp {
            layers,
            sigmoid_output,
        })
    }

    pub fn apply<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let last = self.layers.len() - 1;
        let mut x = x;
        for (i, l) in self.layers.iter().enumerate() {
            x = l.apply(x);
            if i < last {
                x = relu(x);
            }
        }
        if self.sigmoid_output {
            x = sigmoid(x);
        }
        x
    }
}

/// A two-layer MLP with GELU between — the Hiera blocks and the CXBlocks,
/// which use `Linear` layers rather than SAM 2's `MLP` helper.
pub struct GeluMlp<B: Backend> {
    pub up: Lin<B>,
    pub down: Lin<B>,
}

impl<B: Backend> GeluMlp<B> {
    pub fn load(
        p: &Params,
        up: &str,
        down: &str,
        dim: usize,
        hidden: usize,
        dev: &B::Device,
    ) -> Result<GeluMlp<B>> {
        Ok(GeluMlp {
            up: Lin::load(p, up, hidden, dim, dev)?,
            down: Lin::load(p, down, dim, hidden, dev)?,
        })
    }

    pub fn apply<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.down.apply(gelu(self.up.apply(x)))
    }
}

/// An `nn.Conv2d`.
pub struct Conv<B: Backend> {
    pub weight: Tensor<B, 4>,
    pub bias: Tensor<B, 1>,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl<B: Backend> Conv<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        p: &Params,
        prefix: &str,
        out: usize,
        inp: usize,
        k: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        dev: &B::Device,
    ) -> Result<Conv<B>> {
        let (w, b) = p.conv2d(prefix, out, inp, k, groups)?;
        Ok(Conv {
            weight: ops::from_slice(w, [out, inp / groups, k, k], dev),
            bias: ops::from_slice(b, [out], dev),
            stride,
            padding,
            groups,
        })
    }

    /// A 1 x 1 projection, which is most of them.
    pub fn load_1x1(
        p: &Params,
        prefix: &str,
        out: usize,
        inp: usize,
        dev: &B::Device,
    ) -> Result<Conv<B>> {
        Self::load(p, prefix, out, inp, 1, 1, 0, 1, dev)
    }

    pub fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        ops::conv2d(
            x,
            &self.weight,
            Some(&self.bias),
            self.stride,
            self.padding,
            self.groups,
        )
    }
}

/// An `nn.ConvTranspose2d` with kernel = stride = 2.
pub struct ConvT2x<B: Backend> {
    pub weight: Tensor<B, 4>,
    pub bias: Tensor<B, 1>,
}

impl<B: Backend> ConvT2x<B> {
    pub fn load(
        p: &Params,
        prefix: &str,
        out: usize,
        inp: usize,
        dev: &B::Device,
    ) -> Result<ConvT2x<B>> {
        let (w, b) = p.conv_transpose2d(prefix, out, inp, 2)?;
        Ok(ConvT2x {
            weight: ops::from_slice(w, [inp, out, 2, 2], dev),
            bias: ops::from_slice(b, [out], dev),
        })
    }

    pub fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        ops::conv_transpose2d_2x(x, &self.weight, Some(&self.bias))
    }
}

/// `sam/transformer.py`'s `Attention`: four projections around one
/// scaled-dot-product attention, with an internal width that may be narrower
/// than the embedding and keys that may be narrower still.
pub struct SamAttention<B: Backend> {
    pub q: Lin<B>,
    pub k: Lin<B>,
    pub v: Lin<B>,
    pub out: Lin<B>,
    pub heads: usize,
    pub internal: usize,
}

impl<B: Backend> SamAttention<B> {
    pub fn load(
        p: &Params,
        prefix: &str,
        dim: usize,
        heads: usize,
        downsample: usize,
        kv_in: usize,
        dev: &B::Device,
    ) -> Result<SamAttention<B>> {
        let internal = dim / downsample;
        Ok(SamAttention {
            q: Lin::load(p, &format!("{prefix}.q_proj"), internal, dim, dev)?,
            k: Lin::load(p, &format!("{prefix}.k_proj"), internal, kv_in, dev)?,
            v: Lin::load(p, &format!("{prefix}.v_proj"), internal, kv_in, dev)?,
            out: Lin::load(p, &format!("{prefix}.out_proj"), dim, internal, dev)?,
            heads,
            internal,
        })
    }

    /// Split `[b, n, internal]` into `[b, heads, n, head_dim]`.
    pub fn split(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [b, n, _] = x.dims();
        x.reshape([b, n, self.heads, self.internal / self.heads])
            .swap_dims(1, 2)
    }

    /// The inverse.
    pub fn merge(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [b, _, n, _] = x.dims();
        x.swap_dims(1, 2).reshape([b, n, self.internal])
    }

    pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.split(self.q.apply(q));
        let k = self.split(self.k.apply(k));
        let v = self.split(self.v.apply(v));
        self.out.apply(self.merge(ops::sdpa(q, k, v)))
    }
}
