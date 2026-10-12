//! The TotalSynth UNet++ on the CPU without `burn`: the layers of
//! [`super::net`] on the GEMM kernels of [`crate::autoseg::cpu`], with the
//! per-channel steps (ReLU, the residual sum, max pooling, nearest
//! upsampling, the final `tanh`) as plain loops.
//!
//! A batch of stacks is one activation volume whose depth axis is the
//! batch - `[C, stacks, H, W]` - and every convolution has a kernel one deep,
//! so the stacks never mix. The batch norms were folded into the
//! convolutions at conversion ([`super::layout`]).

use rayon::prelude::*;

use super::layout::{node_inputs, BasicBlockW, ConvW, Input, Weights, NODES};
use crate::autoseg::cpu::{conv3d_padded, Act};

/// One convolution of every stack, then ReLU when asked.
fn conv(x: &Act, l: &ConvW, relu: bool) -> Act {
    let oh = (x.h + 2 * l.pad - l.k) / l.stride + 1;
    let ow = (x.w + 2 * l.pad - l.k) / l.stride + 1;
    let mut y = conv3d_padded(
        x,
        &l.w,
        &l.b,
        l.cout,
        [1, l.k, l.k],
        [1, l.stride, l.stride],
        [0, l.pad, l.pad],
        [x.d, oh, ow],
    );
    if relu {
        y.data.par_iter_mut().for_each(|v| *v = v.max(0.0));
    }
    y
}

/// `BasicBlock`: ReLU(conv2(ReLU(conv1(x))) + skip(x)).
fn basic_block(x: &Act, b: &BasicBlockW) -> Act {
    let mut y = conv(&conv(x, &b.conv1, true), &b.conv2, false);
    match &b.skip {
        Some(s) => {
            let r = conv(x, s, false);
            y.data
                .par_iter_mut()
                .zip(r.data.par_iter())
                .for_each(|(a, b)| *a = (*a + b).max(0.0));
        }
        None => y
            .data
            .par_iter_mut()
            .zip(x.data.par_iter())
            .for_each(|(a, b)| *a = (*a + b).max(0.0)),
    }
    y
}

/// 3 x 3 max pooling, stride 2, padding 1, of every slice (`nn.MaxPool2d`:
/// the padding never wins).
fn max_pool3s2(x: &Act) -> Act {
    let (oh, ow) = ((x.h + 2 - 3) / 2 + 1, (x.w + 2 - 3) / 2 + 1);
    let mut out = Act::zeros(x.c, x.d, oh, ow);
    let (ip, op) = (x.h * x.w, oh * ow);
    out.data
        .par_chunks_mut(op)
        .zip(x.data.par_chunks(ip))
        .for_each(|(o, i)| {
            for oy in 0..oh {
                let y0 = (2 * oy).saturating_sub(1);
                let y1 = (2 * oy + 2).min(x.h);
                for ox in 0..ow {
                    let x0 = (2 * ox).saturating_sub(1);
                    let x1 = (2 * ox + 2).min(x.w);
                    let mut m = f32::NEG_INFINITY;
                    for yy in y0..y1 {
                        for v in &i[yy * x.w + x0..yy * x.w + x1] {
                            m = m.max(*v);
                        }
                    }
                    o[oy * ow + ox] = m;
                }
            }
        });
    out
}

/// Nearest-neighbour upsampling by 2 of every slice.
fn up2(x: &Act) -> Act {
    let (h, w) = (x.h * 2, x.w * 2);
    let mut out = Act::zeros(x.c, x.d, h, w);
    let (ip, op) = (x.h * x.w, h * w);
    out.data
        .par_chunks_mut(op)
        .zip(x.data.par_chunks(ip))
        .for_each(|(o, i)| {
            for y in 0..h {
                let src = &i[(y / 2) * x.w..(y / 2 + 1) * x.w];
                for (xx, v) in o[y * w..(y + 1) * w].iter_mut().enumerate() {
                    *v = src[xx / 2];
                }
            }
        });
    out
}

/// Channel concatenation of activations of one spatial size.
fn concat(parts: &[&Act]) -> Act {
    let (d, h, w) = (parts[0].d, parts[0].h, parts[0].w);
    let c = parts.iter().map(|p| p.c).sum();
    let mut out = Act::zeros(c, d, h, w);
    let mut at = 0;
    for p in parts {
        debug_assert!(p.d == d && p.h == h && p.w == w);
        out.data[at..at + p.data.len()].copy_from_slice(&p.data);
        at += p.data.len();
    }
    out
}

/// An encoder feature or a decoder node already computed.
fn input<'a>(enc: &'a [Act], nodes: &'a [((usize, usize), Act)], i: Input) -> &'a Act {
    match i {
        Input::Enc(e) => &enc[e],
        Input::Node(d, l) => nodes
            .iter()
            .find(|(k, _)| *k == (d, l))
            .map(|(_, a)| a)
            .expect("a node is computed before it is read"),
    }
}

/// The network's output for `n` stacks of `in_ch x h x w` (h and w
/// multiples of 32), `[n][h][w]`, after the final `tanh`.
pub fn forward(wt: &Weights, stacks: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
    let logits = logits(wt, stacks, n, h, w);
    logits.into_par_iter().map(f32::tanh).collect()
}

/// The head's raw output (before `tanh`), `[n][h][w]`.
pub fn logits(wt: &Weights, stacks: &[f32], n: usize, h: usize, w: usize) -> Vec<f32> {
    // [n, C, h, w] → [C, n, h, w]
    let plane = h * w;
    let c = wt.in_ch;
    let mut x = Act::zeros(c, n, h, w);
    for s in 0..n {
        for ch in 0..c {
            x.data[(ch * n + s) * plane..(ch * n + s + 1) * plane]
                .copy_from_slice(&stacks[(s * c + ch) * plane..(s * c + ch + 1) * plane]);
        }
    }
    // Encoder: enc[0] the stem (1/2), enc[1..=4] the stages.
    let stem = conv(&x, &wt.stem, true);
    let mut feat = max_pool3s2(&stem);
    let mut enc = vec![stem];
    for stage in &wt.stages {
        for b in stage {
            feat = basic_block(&feat, b);
        }
        enc.push(feat.clone());
    }
    drop(feat);
    // Decoder, in smp's order.
    let mut nodes: Vec<((usize, usize), Act)> = Vec::with_capacity(NODES.len());
    for (k, &(d, l)) in NODES.iter().enumerate() {
        let (up, skips) = node_inputs(d, l);
        let cat = {
            let up = up2(input(&enc, &nodes, up));
            let mut parts = vec![&up];
            parts.extend(skips.into_iter().map(|s| input(&enc, &nodes, s)));
            concat(&parts)
        };
        let node = &wt.nodes[k];
        let y = conv(&conv(&cat, &node.conv1, true), &node.conv2, true);
        nodes.push(((d, l), y));
    }
    let last = nodes.pop().expect("x_0_4").1;
    let y = conv(&last, &wt.head, false);
    // [1, n, h, w] is already [n][h][w]
    y.data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_and_upsampling_follow_pytorch() {
        // 1 channel, 1 slice, 4 x 4: MaxPool2d(3, 2, 1) takes the max of
        // the windows rows/cols {0,1}, {1,2,3}.
        let x = Act {
            c: 1,
            d: 1,
            h: 4,
            w: 4,
            data: (0..16).map(|v| ((v * 7) % 11) as f32).collect(),
        };
        let p = max_pool3s2(&x);
        assert_eq!((p.h, p.w), (2, 2));
        let at = |y: usize, x0: usize| x.data[y * 4 + x0];
        let want00 = [at(0, 0), at(0, 1), at(1, 0), at(1, 1)]
            .into_iter()
            .fold(f32::MIN, f32::max);
        assert_eq!(p.data[0], want00);
        let mut want11 = f32::MIN;
        for y in 1..4 {
            for xx in 1..4 {
                want11 = want11.max(at(y, xx));
            }
        }
        assert_eq!(p.data[3], want11);
        let u = up2(&p);
        assert_eq!((u.h, u.w), (4, 4));
        assert_eq!(u.data[5], p.data[0]);
        assert_eq!(u.data[15], p.data[3]);
    }
}
