//! The TotalSynth UNet++ written once against `burn`, for the GPU through
//! wgpu (the CPU runs [`super::cpu`], the same layers without `burn`).
//!
//! `segmentation_models_pytorch.UnetPlusPlus` with a ResNet-34 encoder: a
//! 7 x 7 stride-2 stem, a 3 x 3 stride-2 max pool, four stages of
//! torchvision `BasicBlock`s, the nested decoder grid of
//! [`super::layout::NODES`] (nearest 2x upsampling, concatenation, two
//! conv 3 x 3 → ReLU), a conv 3 x 3 head and `tanh`. The batch norms are
//! folded into the convolutions ([`super::layout`]).

use burn::tensor::activation::{relu, tanh};
use burn::tensor::backend::Backend;
use burn::tensor::module::max_pool2d;
use burn::tensor::Tensor;

use super::layout::{node_inputs, ConvW, Input, Weights, NODES};
use crate::medsam2::ops;
use crate::nn::fastconv;

/// A convolution on a device.
struct Conv<B: Backend> {
    w: Tensor<B, 4>,
    b: Tensor<B, 1>,
    stride: usize,
    pad: usize,
}

impl<B: Backend> Conv<B> {
    fn upload(c: &ConvW, dev: &B::Device) -> Self {
        Conv {
            w: ops::from_slice(&c.w, [c.cout, c.cin, c.k, c.k], dev),
            b: ops::from_slice(&c.b, [c.cout], dev),
            stride: c.stride,
            pad: c.pad,
        }
    }

    fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        fastconv::conv2d(x, &self.w, Some(&self.b), self.stride, self.pad)
    }
}

struct Block<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
    skip: Option<Conv<B>>,
}

struct Node<B: Backend> {
    conv1: Conv<B>,
    conv2: Conv<B>,
}

/// The network with its weights on a device.
pub struct UnetPp<B: Backend> {
    stem: Conv<B>,
    stages: Vec<Vec<Block<B>>>,
    nodes: Vec<Node<B>>,
    head: Conv<B>,
    in_ch: usize,
    device: B::Device,
}

impl<B: Backend> UnetPp<B> {
    pub fn upload(w: &Weights, dev: &B::Device) -> Self {
        UnetPp {
            stem: Conv::upload(&w.stem, dev),
            stages: w
                .stages
                .iter()
                .map(|s| {
                    s.iter()
                        .map(|b| Block {
                            conv1: Conv::upload(&b.conv1, dev),
                            conv2: Conv::upload(&b.conv2, dev),
                            skip: b.skip.as_ref().map(|c| Conv::upload(c, dev)),
                        })
                        .collect()
                })
                .collect(),
            nodes: w
                .nodes
                .iter()
                .map(|n| Node {
                    conv1: Conv::upload(&n.conv1, dev),
                    conv2: Conv::upload(&n.conv2, dev),
                })
                .collect(),
            head: Conv::upload(&w.head, dev),
            in_ch: w.in_ch,
            device: dev.clone(),
        }
    }

    pub fn in_channels(&self) -> usize {
        self.in_ch
    }

    /// The head's output (before `tanh`) for stacks `[n, in_ch, h, w]` (h
    /// and w multiples of 32): `[n, 1, h, w]`.
    pub fn logits(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let stem = relu(self.stem.apply(x));
        let mut feat = max_pool2d(stem.clone(), [3, 3], [2, 2], [1, 1], [1, 1], false);
        let mut enc = vec![stem];
        for stage in &self.stages {
            for b in stage {
                let y = b.conv2.apply(relu(b.conv1.apply(feat.clone())));
                let r = match &b.skip {
                    Some(s) => s.apply(feat),
                    None => feat,
                };
                feat = relu(y + r);
            }
            enc.push(feat.clone());
        }
        let mut done: Vec<((usize, usize), Tensor<B, 4>)> = Vec::with_capacity(NODES.len());
        let get = |done: &[((usize, usize), Tensor<B, 4>)], i: Input| -> Tensor<B, 4> {
            match i {
                Input::Enc(e) => enc[e].clone(),
                Input::Node(d, l) => done
                    .iter()
                    .find(|(k, _)| *k == (d, l))
                    .map(|(_, t)| t.clone())
                    .expect("a node is computed before it is read"),
            }
        };
        for (k, &(d, l)) in NODES.iter().enumerate() {
            let (up, skips) = node_inputs(d, l);
            let mut parts = vec![ops::upsample_nearest_2x(get(&done, up))];
            parts.extend(skips.into_iter().map(|s| get(&done, s)));
            let n = &self.nodes[k];
            let y = relu(n.conv2.apply(relu(n.conv1.apply(Tensor::cat(parts, 1)))));
            done.push(((d, l), y));
        }
        let last = done.pop().expect("x_0_4").1;
        self.head.apply(last)
    }

    /// The network's output after `tanh`, `[n][h][w]`, for `n` stacks of
    /// `in_ch x h x w` given as `[n][in_ch][h][w]`.
    pub fn forward(&self, stacks: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
        let x = ops::from_slice::<B, 4>(stacks, [n, self.in_ch, h, w], &self.device);
        ops::to_vec(tanh(self.logits(x)))
    }
}
