//! lungmask's U-Net on the CPU without `burn`: the same layers as
//! [`super::net`], on the GEMM kernels of [`crate::autoseg::cpu`] and with
//! the per-channel steps (ReLU and the folded batch norm, pooling,
//! bilinear upsampling) as plain loops.
//!
//! A batch of slices is laid out as one activation volume whose depth axis
//! is the batch - `[C, slices, H, W]` - and every convolution has a kernel
//! one deep, so the slices never mix. burn's CPU backend spent most of its
//! time outside the convolutions here (its bilinear upsampling alone took
//! seconds per batch).

use rayon::prelude::*;

use super::net::{BlockW, ConvBnW, Weights};
use crate::autoseg::cpu::{concat, conv3d_padded, Act};

/// Conv → ReLU → scale and shift.
fn conv_bn(x: &Act, l: &ConvBnW) -> Act {
    let p = l.k / 2;
    let mut y = conv3d_padded(
        x,
        &l.w,
        &l.b,
        l.cout,
        [1, l.k, l.k],
        [1, 1, 1],
        [0, p, p],
        [x.d, x.h, x.w],
    );
    let n = y.d * y.h * y.w;
    y.data.par_chunks_mut(n).enumerate().for_each(|(c, ch)| {
        let (s, t) = (l.scale[c], l.shift[c]);
        for v in ch {
            *v = v.max(0.0) * s + t;
        }
    });
    y
}

fn block(x: &Act, b: &BlockW) -> Act {
    conv_bn(&conv_bn(x, &b[0]), &b[1])
}

/// 2 x 2 average pooling of every slice.
fn avg_pool2(x: &Act) -> Act {
    let (h, w) = (x.h / 2, x.w / 2);
    let mut out = Act::zeros(x.c, x.d, h, w);
    let (ip, op) = (x.h * x.w, h * w);
    out.data
        .par_chunks_mut(op)
        .zip(x.data.par_chunks(ip))
        .for_each(|(o, i)| {
            for y in 0..h {
                for xx in 0..w {
                    let a = (2 * y) * x.w + 2 * xx;
                    let b = a + x.w;
                    o[y * w + xx] = (i[a] + i[a + 1] + i[b] + i[b + 1]) / 4.0;
                }
            }
        });
    out
}

/// Bilinear x2 of every slice, `align_corners=False`: an even output takes
/// 1/4 of the previous input and 3/4 of its own, an odd one 3/4 of its own
/// and 1/4 of the next, clamped at the edges.
fn bilinear2(x: &Act) -> Act {
    let (h, w) = (x.h * 2, x.w * 2);
    let mut out = Act::zeros(x.c, x.d, h, w);
    let taps = |o: usize, n: usize| -> (usize, usize, f32, f32) {
        let src = (0.5 * (o as f32 + 0.5) - 0.5).max(0.0);
        let i0 = (src.floor() as usize).min(n - 1);
        let l1 = src - i0 as f32;
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, 1.0 - l1, l1)
    };
    let ty: Vec<_> = (0..h).map(|o| taps(o, x.h)).collect();
    let tx: Vec<_> = (0..w).map(|o| taps(o, x.w)).collect();
    let (ip, op) = (x.h * x.w, h * w);
    out.data
        .par_chunks_mut(op)
        .zip(x.data.par_chunks(ip))
        .for_each(|(o, i)| {
            for (oy, &(y0, y1, h0, h1)) in ty.iter().enumerate() {
                for (ox, &(x0, x1, w0, w1)) in tx.iter().enumerate() {
                    let top = i[y0 * x.w + x0] * w0 + i[y0 * x.w + x1] * w1;
                    let bot = i[y1 * x.w + x0] * w0 + i[y1 * x.w + x1] * w1;
                    o[oy * w + ox] = top * h0 + bot * h1;
                }
            }
        });
    out
}

/// A 1 x 1 convolution.
fn conv1(x: &Act, w: &[f32], b: &[f32], cout: usize) -> Act {
    conv3d_padded(
        x,
        w,
        b,
        cout,
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        [x.d, x.h, x.w],
    )
}

/// Logits `[classes, n, h, w]` for slices `[n][h][w]` (h and w multiples
/// of 16).
pub fn forward(wt: &Weights, slices: &[f32], n: usize, h: usize, w: usize) -> Act {
    let mut x = Act {
        c: 1,
        d: n,
        h,
        w,
        data: slices.to_vec(),
    };
    let mut skips = Vec::with_capacity(wt.down.len());
    for (i, d) in wt.down.iter().enumerate() {
        x = block(&x, d);
        if i + 1 != wt.down.len() {
            let pooled = avg_pool2(&x);
            skips.push(x);
            x = pooled;
        }
    }
    for u in &wt.up {
        let up = conv1(&bilinear2(&x), &u.w, &u.b, u.cout);
        let skip = skips.pop().expect("one skip per up block");
        x = block(&concat(&up, &skip), &u.block);
    }
    conv1(&x, &wt.last_w, &wt.last_b, wt.classes)
}

/// The label (argmax, first maximum on a tie) of every pixel, `[n][h][w]`.
pub fn labels(wt: &Weights, slices: &[f32], n: usize, h: usize, w: usize) -> Vec<u8> {
    let logits = forward(wt, slices, n, h, w);
    let per = n * h * w;
    (0..per)
        .map(|v| {
            let mut best = 0;
            let mut best_v = f32::NEG_INFINITY;
            for c in 0..wt.classes {
                let l = logits.data[c * per + v];
                if l > best_v {
                    best_v = l;
                    best = c;
                }
            }
            best as u8
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::net::tests::{fixture, worst};
    use super::*;

    #[test]
    fn the_cpu_network_matches_pytorch() {
        let p = fixture("unet2d");
        let wt = Weights::load(&p).unwrap();
        let x = p.get("__input", &[2, 1, 32, 32]).unwrap();
        let want = p.get("__output", &[2, 3, 32, 32]).unwrap();
        let y = forward(&wt, x, 2, 32, 32);
        // [classes, n, ...] → [n, classes, ...], then the reference's
        // LogSoftmax
        let mut g = vec![0f32; 2 * 3 * 1024];
        for s in 0..2 {
            for v in 0..1024 {
                let ls: Vec<f32> = (0..3).map(|c| y.data[(c * 2 + s) * 1024 + v]).collect();
                let m = ls.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let lse = m + ls.iter().map(|l| (l - m).exp()).sum::<f32>().ln();
                for c in 0..3 {
                    g[(s * 3 + c) * 1024 + v] = ls[c] - lse;
                }
            }
        }
        let w = worst(&g, want);
        assert!(w < 1e-4, "relative error {w:e}");
    }
}
