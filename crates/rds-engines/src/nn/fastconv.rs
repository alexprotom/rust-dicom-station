//! Convolutions for the networks written against `burn`, with the CPU
//! backend routed through the im2col + SIMD GEMM kernels of
//! [`crate::autoseg::cpu`].
//!
//! burn's pure-Rust `ndarray` backend convolves an order of magnitude more
//! slowly than those kernels (lungmask's 2-D U-Net ran at about 1.5
//! GFLOP/s through it, minutes per slice stack instead of seconds). The
//! networks stay generic over the backend; only these calls look at which
//! one they were given, and on the CPU backend move the tensors' data
//! through the GEMM kernels and back. On the GPU they are burn's own.

use std::any::TypeId;

use burn::tensor::backend::Backend;
use burn::tensor::module::{conv2d as burn_conv2d, conv3d as burn_conv3d, conv_transpose3d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions};
use burn::tensor::{Tensor, TensorData};

use crate::autoseg::cpu::{self, conv3d_padded, Act};

/// Is `B` the CPU backend the engines use?
fn is_cpu<B: Backend>() -> bool {
    TypeId::of::<B>() == TypeId::of::<crate::medsam2::engine::Cpu>()
}

pub(crate) fn data<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("float tensor")
}

pub(crate) fn tensor<B: Backend, const D: usize>(
    v: Vec<f32>,
    shape: [usize; D],
    dev: &B::Device,
) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(v, shape), dev)
}

/// `Conv2d(kernel k, stride, padding)` on `[n, c, h, w]`.
pub fn conv2d<B: Backend>(
    x: Tensor<B, 4>,
    w: &Tensor<B, 4>,
    b: Option<&Tensor<B, 1>>,
    stride: usize,
    padding: usize,
) -> Tensor<B, 4> {
    if !is_cpu::<B>() {
        return burn_conv2d(
            x,
            w.clone(),
            b.cloned(),
            ConvOptions::new([stride, stride], [padding, padding], [1, 1], 1),
        );
    }
    let dev = x.device();
    let [n, c, h, wd] = x.dims();
    let [cout, _, kh, kw] = w.dims();
    let oh = (h + 2 * padding - kh) / stride + 1;
    let ow = (wd + 2 * padding - kw) / stride + 1;
    let xs = data(x);
    let ws = data(w.clone());
    let bs = b
        .map(|b| data(b.clone()))
        .unwrap_or_else(|| vec![0.0; cout]);
    // The batch becomes the depth axis of a 3-D convolution with a kernel
    // one deep, so every slice is convolved on its own in one call.
    let plane = h * wd;
    let mut act = Act::zeros(c, n, h, wd);
    for s in 0..n {
        for ch in 0..c {
            act.data[(ch * n + s) * plane..(ch * n + s + 1) * plane]
                .copy_from_slice(&xs[(s * c + ch) * plane..(s * c + ch + 1) * plane]);
        }
    }
    let y = conv3d_padded(
        &act,
        &ws,
        &bs,
        cout,
        [1, kh, kw],
        [1, stride, stride],
        [0, padding, padding],
        [n, oh, ow],
    );
    let oplane = oh * ow;
    let mut out = vec![0f32; n * cout * oplane];
    for s in 0..n {
        for ch in 0..cout {
            out[(s * cout + ch) * oplane..(s * cout + ch + 1) * oplane]
                .copy_from_slice(&y.data[(ch * n + s) * oplane..(ch * n + s + 1) * oplane]);
        }
    }
    tensor(out, [n, cout, oh, ow], &dev)
}

/// `Conv3d(kernel k, stride, padding)` on `[n, c, d, h, w]`.
pub fn conv3d<B: Backend>(
    x: Tensor<B, 5>,
    w: &Tensor<B, 5>,
    b: Option<&Tensor<B, 1>>,
    stride: usize,
    padding: usize,
) -> Tensor<B, 5> {
    if !is_cpu::<B>() {
        return burn_conv3d(
            x,
            w.clone(),
            b.cloned(),
            ConvOptions::new([stride; 3], [padding; 3], [1, 1, 1], 1),
        );
    }
    let dev = x.device();
    let [n, c, d, h, wd] = x.dims();
    let [cout, _, kd, kh, kw] = w.dims();
    let o = |len: usize, k: usize| (len + 2 * padding - k) / stride + 1;
    let od = [o(d, kd), o(h, kh), o(wd, kw)];
    let xs = data(x);
    let ws = data(w.clone());
    let bs = b
        .map(|b| data(b.clone()))
        .unwrap_or_else(|| vec![0.0; cout]);
    let vol = c * d * h * wd;
    let mut out = Vec::with_capacity(n * cout * od[0] * od[1] * od[2]);
    for s in 0..n {
        let act = Act {
            c,
            d,
            h,
            w: wd,
            data: xs[s * vol..(s + 1) * vol].to_vec(),
        };
        let y = conv3d_padded(
            &act,
            &ws,
            &bs,
            cout,
            [kd, kh, kw],
            [stride; 3],
            [padding; 3],
            od,
        );
        out.extend_from_slice(&y.data);
    }
    tensor(out, [n, cout, od[0], od[1], od[2]], &dev)
}

/// `ConvTranspose3d(kernel 3, stride 2, padding 1, output_padding 1)` on
/// `[n, c, d, h, w]`; `w` is `[cin, cout, 3, 3, 3]`.
pub fn conv_transpose3d_k3s2<B: Backend>(
    x: Tensor<B, 5>,
    w: &Tensor<B, 5>,
    b: Option<&Tensor<B, 1>>,
) -> Tensor<B, 5> {
    if !is_cpu::<B>() {
        return conv_transpose3d(
            x,
            w.clone(),
            b.cloned(),
            ConvTransposeOptions::new([2; 3], [1; 3], [1; 3], [1; 3], 1),
        );
    }
    let dev = x.device();
    let [n, c, d, h, wd] = x.dims();
    let [_, cout, _, _, _] = w.dims();
    let xs = data(x);
    let ws = data(w.clone());
    let bs = b
        .map(|b| data(b.clone()))
        .unwrap_or_else(|| vec![0.0; cout]);
    let vol = c * d * h * wd;
    let mut out = Vec::with_capacity(n * cout * 8 * d * h * wd);
    for s in 0..n {
        let act = Act {
            c,
            d,
            h,
            w: wd,
            data: xs[s * vol..(s + 1) * vol].to_vec(),
        };
        out.extend_from_slice(&cpu::conv_transpose3d_k3s2(&act, &ws, &bs, cout).data);
    }
    tensor(out, [n, cout, 2 * d, 2 * h, 2 * wd], &dev)
}

/// What follows a normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    None,
    Relu,
    LeakyRelu(f32),
}

impl Activation {
    #[inline]
    fn apply(self, v: f32) -> f32 {
        match self {
            Activation::None => v,
            Activation::Relu => v.max(0.0),
            Activation::LeakyRelu(s) => {
                if v < 0.0 {
                    v * s
                } else {
                    v
                }
            }
        }
    }

    fn burn<B: Backend>(self, x: Tensor<B, 5>) -> Tensor<B, 5> {
        match self {
            Activation::None => x,
            Activation::Relu => burn::tensor::activation::relu(x),
            Activation::LeakyRelu(s) => burn::tensor::activation::leaky_relu(x, s as f64),
        }
    }
}

/// `GroupNorm(groups)` on `[n, c, d, h, w]` (`InstanceNorm` is `groups =
/// c`), each channel then scaled by `gamma` and shifted by `beta` (the
/// identity when `None`), then the activation.
///
/// On the CPU backend this is two passes over the data instead of the
/// dozen burn's elementwise operations make: the statistics of each group
/// (in `f64`, mean first, then the centred variance, as PyTorch), then one
/// pass applying normalization, affine map and activation together. The
/// whole-body SegResNet spent most of its time in them.
pub fn group_norm_act<B: Backend>(
    x: Tensor<B, 5>,
    groups: usize,
    gamma: Option<&[f32]>,
    beta: Option<&[f32]>,
    eps: f64,
    act: Activation,
) -> Tensor<B, 5> {
    let [n, c, d, h, w] = x.dims();
    if !is_cpu::<B>() {
        let dev = x.device();
        let g = x.reshape([n, groups, (c / groups) * d * h * w]);
        let mean = g.clone().mean_dim(2);
        let centred = g - mean;
        let var = centred.clone().powf_scalar(2.0).mean_dim(2);
        let mut y = (centred / (var + eps).sqrt()).reshape([n, c, d, h, w]);
        if let Some(gm) = gamma {
            y = y * tensor::<B, 5>(gm.to_vec(), [1, c, 1, 1, 1], &dev);
        }
        if let Some(bt) = beta {
            y = y + tensor::<B, 5>(bt.to_vec(), [1, c, 1, 1, 1], &dev);
        }
        return act.burn(y);
    }
    use rayon::prelude::*;
    let dev = x.device();
    let plane = d * h * w;
    let cg = c / groups;
    let mut v = data(x);
    v.par_chunks_mut(cg * plane)
        .enumerate()
        .for_each(|(gi, chunk)| {
            let g = gi % groups;
            let len = chunk.len() as f64;
            let mean = chunk.iter().map(|&x| x as f64).sum::<f64>() / len;
            let var = chunk
                .iter()
                .map(|&x| {
                    let t = x as f64 - mean;
                    t * t
                })
                .sum::<f64>()
                / len;
            let inv = 1.0 / (var + eps).sqrt();
            for (ci, ch) in chunk.chunks_mut(plane).enumerate() {
                let idx = g * cg + ci;
                let a = gamma.map_or(1.0, |gm| gm[idx] as f64) * inv;
                let b = beta.map_or(0.0, |bt| bt[idx] as f64) - mean * a;
                let (a, b) = (a as f32, b as f32);
                for x in ch.iter_mut() {
                    *x = act.apply(*x * a + b);
                }
            }
        });
    tensor(v, [n, c, d, h, w], &dev)
}

/// A per-channel affine map (`BatchNorm` in eval mode, folded) and the
/// activation, in one pass on the CPU backend.
pub fn affine_act<B: Backend>(
    x: Tensor<B, 5>,
    scale: &[f32],
    shift: &[f32],
    act: Activation,
) -> Tensor<B, 5> {
    let [n, c, d, h, w] = x.dims();
    let dev = x.device();
    if !is_cpu::<B>() {
        let y = x * tensor::<B, 5>(scale.to_vec(), [1, c, 1, 1, 1], &dev)
            + tensor::<B, 5>(shift.to_vec(), [1, c, 1, 1, 1], &dev);
        return act.burn(y);
    }
    use rayon::prelude::*;
    let plane = d * h * w;
    let mut v = data(x);
    v.par_chunks_mut(plane).enumerate().for_each(|(i, ch)| {
        let (a, b) = (scale[i % c], shift[i % c]);
        for x in ch.iter_mut() {
            *x = act.apply(*x * a + b);
        }
    });
    tensor(v, [n, c, d, h, w], &dev)
}

/// Linear (trilinear) x2 upsampling with `align_corners=False` on the CPU
/// backend: along each axis the new samples sit a quarter and three
/// quarters of the way between neighbours, the ends clamped (PyTorch's
/// `interpolate`). `None` on another backend.
pub fn upsample_linear_2x_cpu<B: Backend>(x: &Tensor<B, 5>) -> Option<Tensor<B, 5>> {
    if !is_cpu::<B>() {
        return None;
    }
    use rayon::prelude::*;
    let [n, c, d, h, w] = x.dims();
    let dev = x.device();
    let src = data(x.clone());
    let (od, oh, ow) = (2 * d, 2 * h, 2 * w);
    let mut out = vec![0.0f32; n * c * od * oh * ow];
    out.par_chunks_mut(od * oh * ow)
        .zip(src.par_chunks(d * h * w))
        .for_each(|(dst, s)| {
            // Along w, then h, then d.
            let mut a = vec![0.0f32; d * h * ow];
            for (row, out_row) in s.chunks(w).zip(a.chunks_mut(ow)) {
                up_line(row, out_row, 1);
            }
            let mut b = vec![0.0f32; d * oh * ow];
            for z in 0..d {
                let src_plane = &a[z * h * ow..(z + 1) * h * ow];
                let dst_plane = &mut b[z * oh * ow..(z + 1) * oh * ow];
                for col in 0..ow {
                    up_strided(src_plane, dst_plane, col, ow, h);
                }
            }
            for i in 0..oh * ow {
                up_strided(&b, dst, i, oh * ow, d);
            }
        });
    Some(tensor(out, [n, c, od, oh, ow], &dev))
}

/// One contiguous line of `n` samples doubled into `out`.
fn up_line(src: &[f32], out: &mut [f32], _stride: usize) {
    let n = src.len();
    for i in 0..n {
        let prev = src[i.saturating_sub(1)];
        let next = src[(i + 1).min(n - 1)];
        out[2 * i] = 0.75 * src[i] + 0.25 * prev;
        out[2 * i + 1] = 0.75 * src[i] + 0.25 * next;
    }
}

/// The line of `n` samples at `start`, `start + stride`, ... of `src`,
/// doubled into the same positions of `dst` (whose lines are twice as
/// long).
fn up_strided(src: &[f32], dst: &mut [f32], start: usize, stride: usize, n: usize) {
    let at = |i: usize| src[start + i * stride];
    for i in 0..n {
        let prev = at(i.saturating_sub(1));
        let next = at((i + 1).min(n - 1));
        let cur = at(i);
        dst[start + 2 * i * stride] = 0.75 * cur + 0.25 * prev;
        dst[start + (2 * i + 1) * stride] = 0.75 * cur + 0.25 * next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medsam2::engine::Cpu;

    /// The CPU paths of the norms and the upsampling against burn's own
    /// operations on the same backend's data (the GPU path is those).
    #[test]
    fn fused_norms_and_upsampling_match_the_plain_operations() {
        type B = Cpu;
        let dev = Default::default();
        let (n, c, d, h, w) = (1usize, 4usize, 3usize, 5usize, 6usize);
        let len = n * c * d * h * w;
        let vals: Vec<f32> = (0..len)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0)
            .collect();
        let x = || tensor::<B, 5>(vals.clone(), [n, c, d, h, w], &dev);
        let gamma = [1.5f32, -0.5, 0.25, 2.0];
        let beta = [0.1f32, 0.2, -0.3, 0.0];
        for (groups, act) in [
            (2, Activation::Relu),
            (4, Activation::LeakyRelu(0.01)),
            (1, Activation::None),
        ] {
            let got = data(group_norm_act(
                x(),
                groups,
                Some(&gamma),
                Some(&beta),
                1e-5,
                act,
            ));
            // The plain operations, written out.
            let cg = c / groups;
            let plane = d * h * w;
            let mut want = vals.clone();
            for g in 0..groups {
                let chunk = &vals[g * cg * plane..(g + 1) * cg * plane];
                let m = chunk.iter().map(|&v| v as f64).sum::<f64>() / chunk.len() as f64;
                let var =
                    chunk.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / chunk.len() as f64;
                for (i, o) in want[g * cg * plane..(g + 1) * cg * plane]
                    .iter_mut()
                    .enumerate()
                {
                    let ch = g * cg + i / plane;
                    let y = ((*o as f64 - m) / (var + 1e-5).sqrt()) as f32 * gamma[ch] + beta[ch];
                    *o = act.apply(y);
                }
            }
            let worst = got
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(worst < 1e-5, "groups {groups}: {worst}");
        }
        let got = data(affine_act(x(), &gamma, &beta, Activation::Relu));
        for (i, (g, v)) in got.iter().zip(&vals).enumerate() {
            let ch = (i / (d * h * w)) % c;
            assert!((g - (v * gamma[ch] + beta[ch]).max(0.0)).abs() < 1e-6);
        }
        // The upsampling against burn's narrow / cat / stack version.
        let fast = data(upsample_linear_2x_cpu(&x()).unwrap());
        let slow = data(crate::segresnet::net::upsample_linear_2x_burn(x()));
        assert_eq!(fast.len(), 8 * len);
        let worst = fast
            .iter()
            .zip(&slow)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-5, "upsampling: {worst}");
    }

    /// The reference: a direct loop over the definition.
    fn naive_conv3d(
        x: &[f32],
        [c, d, h, w]: [usize; 4],
        wt: &[f32],
        cout: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> (Vec<f32>, [usize; 3]) {
        let o = |len: usize| (len + 2 * pad - k) / stride + 1;
        let od = [o(d), o(h), o(w)];
        let mut out = vec![0f32; cout * od[0] * od[1] * od[2]];
        for co in 0..cout {
            for z in 0..od[0] {
                for y in 0..od[1] {
                    for xx in 0..od[2] {
                        let mut s = 0f64;
                        for ci in 0..c {
                            for kz in 0..k {
                                for ky in 0..k {
                                    for kx in 0..k {
                                        let iz = (z * stride + kz) as isize - pad as isize;
                                        let iy = (y * stride + ky) as isize - pad as isize;
                                        let ix = (xx * stride + kx) as isize - pad as isize;
                                        if iz < 0
                                            || iy < 0
                                            || ix < 0
                                            || iz >= d as isize
                                            || iy >= h as isize
                                            || ix >= w as isize
                                        {
                                            continue;
                                        }
                                        let xv = x[((ci * d + iz as usize) * h + iy as usize) * w
                                            + ix as usize];
                                        let wv = wt[(((co * c + ci) * k + kz) * k + ky) * k + kx];
                                        s += f64::from(xv) * f64::from(wv);
                                    }
                                }
                            }
                        }
                        out[((co * od[0] + z) * od[1] + y) * od[2] + xx] = s as f32;
                    }
                }
            }
        }
        (out, od)
    }

    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let v = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                (v % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn the_cpu_path_matches_the_definition() {
        let dev = Default::default();
        for (stride, pad, k) in [(1, 1, 3), (2, 1, 3), (1, 0, 1)] {
            let dims = [3usize, 5, 6, 7];
            let x = values(dims.iter().product(), 1);
            let w = values(4 * 3 * k * k * k, 7);
            let (want, od) = naive_conv3d(&x, dims, &w, 4, k, stride, pad);
            let xt = tensor::<Cpu, 5>(x, [1, 3, 5, 6, 7], &dev);
            let wt = tensor::<Cpu, 5>(w, [4, 3, k, k, k], &dev);
            let got = conv3d(xt, &wt, None, stride, pad);
            assert_eq!(got.dims(), [1, 4, od[0], od[1], od[2]]);
            let got = data(got);
            for (a, b) in got.iter().zip(&want) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn two_d_slices_are_convolved_separately() {
        let dev = Default::default();
        let x = values(2 * 3 * 6 * 5, 3);
        let w = values(4 * 3 * 9, 9);
        let b = vec![0.5f32, -0.25, 0.0, 1.0];
        let xt = tensor::<Cpu, 4>(x.clone(), [2, 3, 6, 5], &dev);
        let wt = tensor::<Cpu, 4>(w.clone(), [4, 3, 3, 3], &dev);
        let bt = tensor::<Cpu, 1>(b.clone(), [4], &dev);
        let fast = data(conv2d(xt.clone(), &wt, Some(&bt), 1, 1));
        let slow = data(burn_conv2d(
            xt,
            wt,
            Some(bt),
            ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
        ));
        for (a, b) in fast.iter().zip(&slow) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn the_transposed_convolution_matches_burns() {
        let dev = Default::default();
        let x = values(3 * 3 * 4 * 5, 5);
        let w = values(3 * 2 * 27, 11);
        let b = vec![0.1f32, -0.2];
        let xt = tensor::<Cpu, 5>(x, [1, 3, 3, 4, 5], &dev);
        let wt = tensor::<Cpu, 5>(w, [3, 2, 3, 3, 3], &dev);
        let bt = tensor::<Cpu, 1>(b, [2], &dev);
        let fast = conv_transpose3d_k3s2(xt.clone(), &wt, Some(&bt));
        assert_eq!(fast.dims(), [1, 2, 6, 8, 10]);
        let slow = conv_transpose3d(
            xt,
            wt,
            Some(bt),
            ConvTransposeOptions::new([2; 3], [1; 3], [1; 3], [1; 3], 1),
        );
        for (a, b) in data(fast).iter().zip(&data(slow)) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::medsam2::engine::Cpu;
    use burn::tensor::activation::relu;

    #[test]
    #[ignore]
    fn bench_lungmask_shapes() {
        let dev = Default::default();
        let x = tensor::<Cpu, 4>(vec![0.5; 2 * 64 * 256 * 256], [2, 64, 256, 256], &dev);
        let w = tensor::<Cpu, 4>(vec![0.01; 64 * 64 * 9], [64, 64, 3, 3], &dev);
        let b = tensor::<Cpu, 1>(vec![0.0; 64], [64], &dev);
        let s = tensor::<Cpu, 4>(vec![1.0; 64], [1, 64, 1, 1], &dev);
        let t0 = std::time::Instant::now();
        let y = conv2d(x.clone(), &w, Some(&b), 1, 1);
        let t1 = t0.elapsed().as_secs_f64();
        let y2 = relu(y.clone());
        let t2 = t0.elapsed().as_secs_f64();
        let y3 = y2 * s.clone() + s.clone();
        let _ = data(y3.clone());
        let t3 = t0.elapsed().as_secs_f64();
        let p = burn::tensor::module::avg_pool2d(y3.clone(), [2, 2], [2, 2], [0, 0], true, false);
        let _ = data(p.clone());
        let t4 = t0.elapsed().as_secs_f64();
        let u = crate::medsam2::ops::resize_bilinear(p, [256, 256]);
        let _ = data(u);
        let t5 = t0.elapsed().as_secs_f64();
        eprintln!(
            "conv {t1:.3}s ({:.1} GFLOP/s) relu {:.3} affine {:.3} pool {:.3} bilinear {:.3}",
            2.0 * 2.0 * 65536.0 * 64.0 * 64.0 * 9.0 / t1 / 1e9,
            t2 - t1,
            t3 - t2,
            t4 - t3,
            t5 - t4
        );
    }
}
