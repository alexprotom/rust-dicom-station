//! Naive reference kernels for the MedSAM2 op fixtures.
//!
//! Every function here is the textbook definition of one operation, written
//! as plain nested loops in `f64` with no shortcut, no blocking and no shared
//! code with the engine in `src/medsam2` - a second, independent
//! transcription of what PyTorch, PIL and SAM 2 compute. They exist for two
//! things: `examples/gen_ops_fixtures.rs` writes `tests/data/medsam2-ops.safetensors`
//! with them, and `tests/ops_fixtures.rs` proves them against the copy of
//! that file PyTorch itself produced, so the fixture can be regenerated
//! without Python and still stand for PyTorch's semantics.
//!
//! The conventions each kernel reproduces are written next to it; the
//! coordinate rules are the ones that separate the frameworks (PyTorch's
//! `align_corners=False`, PIL's widened support when shrinking, the 8-bit
//! fixed-point path of a `uint8` resize).

#![allow(dead_code)]

use rust_dicom_station::nn::cache::WTensor;

/// A tensor in row-major (PyTorch) order.
pub fn t(shape: &[usize], data: Vec<f32>) -> WTensor {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    WTensor {
        shape: shape.to_vec(),
        data,
    }
}

fn d(x: &WTensor) -> Vec<f64> {
    x.data.iter().map(|&v| v as f64).collect()
}

fn out(shape: &[usize], v: Vec<f64>) -> WTensor {
    t(shape, v.into_iter().map(|x| x as f32).collect())
}

// ---- convolutions ------------------------------------------------------------

/// `F.conv2d(x, w, b, stride, padding, groups)`: cross-correlation, zero
/// padding, `Ho = (H + 2p - k) / s + 1`.
pub fn conv2d(
    x: &WTensor,
    w: &WTensor,
    b: &WTensor,
    stride: usize,
    pad: usize,
    groups: usize,
) -> WTensor {
    let [n, c, h, wd] = dims4(x);
    let [o, cg, kh, kw] = dims4(w);
    assert_eq!(c, cg * groups);
    let (ho, wo) = (
        (h + 2 * pad - kh) / stride + 1,
        (wd + 2 * pad - kw) / stride + 1,
    );
    let (xd, wv, bv) = (d(x), d(w), d(b));
    let og = o / groups;
    let mut y = vec![0.0; n * o * ho * wo];
    for ni in 0..n {
        for oi in 0..o {
            let g = oi / og;
            for yo in 0..ho {
                for xo in 0..wo {
                    let mut acc = bv[oi];
                    for ci in 0..cg {
                        let cin = g * cg + ci;
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let yi = (yo * stride + ki) as isize - pad as isize;
                                let xi = (xo * stride + kj) as isize - pad as isize;
                                if yi < 0 || xi < 0 || yi >= h as isize || xi >= wd as isize {
                                    continue;
                                }
                                acc += xd[((ni * c + cin) * h + yi as usize) * wd + xi as usize]
                                    * wv[((oi * cg + ci) * kh + ki) * kw + kj];
                            }
                        }
                    }
                    y[((ni * o + oi) * ho + yo) * wo + xo] = acc;
                }
            }
        }
    }
    out(&[n, o, ho, wo], y)
}

/// `F.conv_transpose2d(x, w, b, stride)`, no padding: every input pixel
/// scatters its `k x k` stamp at `(i * s, j * s)`; `w` is `[Cin, Cout, k, k]`.
pub fn conv_transpose2d(x: &WTensor, w: &WTensor, b: &WTensor, stride: usize) -> WTensor {
    let [n, c, h, wd] = dims4(x);
    let [cin, o, kh, kw] = dims4(w);
    assert_eq!(c, cin);
    let (ho, wo) = ((h - 1) * stride + kh, (wd - 1) * stride + kw);
    let (xd, wv, bv) = (d(x), d(w), d(b));
    let mut y = vec![0.0; n * o * ho * wo];
    for ni in 0..n {
        for oi in 0..o {
            for v in y[(ni * o + oi) * ho * wo..(ni * o + oi + 1) * ho * wo].iter_mut() {
                *v = bv[oi];
            }
            for ci in 0..c {
                for i in 0..h {
                    for j in 0..wd {
                        let xv = xd[((ni * c + ci) * h + i) * wd + j];
                        for ki in 0..kh {
                            for kj in 0..kw {
                                y[((ni * o + oi) * ho + i * stride + ki) * wo + j * stride + kj] +=
                                    xv * wv[((ci * o + oi) * kh + ki) * kw + kj];
                            }
                        }
                    }
                }
            }
        }
    }
    out(&[n, o, ho, wo], y)
}

/// `F.max_pool2d(x, 2, 2, ceil_mode=False)`: an odd trailing row or column
/// is dropped.
pub fn maxpool2x2(x: &WTensor) -> WTensor {
    let [n, c, h, w] = dims4(x);
    let (ho, wo) = (h / 2, w / 2);
    let xd = d(x);
    let mut y = vec![0.0; n * c * ho * wo];
    for p in 0..n * c {
        for i in 0..ho {
            for j in 0..wo {
                let mut m = f64::NEG_INFINITY;
                for di in 0..2 {
                    for dj in 0..2 {
                        m = m.max(xd[(p * h + 2 * i + di) * w + 2 * j + dj]);
                    }
                }
                y[(p * ho + i) * wo + j] = m;
            }
        }
    }
    out(&[n, c, ho, wo], y)
}

// ---- PyTorch interpolation ------------------------------------------------

/// `F.interpolate(x, size, mode="bilinear", align_corners=False)`: the
/// source coordinate is `scale * (dst + 0.5) - 0.5`, clamped at zero,
/// with `scale = in / out`.
pub fn interp_bilinear(x: &WTensor, oh: usize, ow: usize) -> WTensor {
    let [n, c, h, w] = dims4(x);
    let xd = d(x);
    let src = |dst: usize, inn: usize, outn: usize| -> (usize, usize, f64) {
        let s = (inn as f64 / outn as f64 * (dst as f64 + 0.5) - 0.5).max(0.0);
        let i0 = (s.floor() as usize).min(inn - 1);
        (i0, (i0 + 1).min(inn - 1), s - i0 as f64)
    };
    let mut y = vec![0.0; n * c * oh * ow];
    for p in 0..n * c {
        for i in 0..oh {
            let (y0, y1, ly) = src(i, h, oh);
            for j in 0..ow {
                let (x0, x1, lx) = src(j, w, ow);
                let at = |yy: usize, xx: usize| xd[(p * h + yy) * w + xx];
                y[(p * oh + i) * ow + j] = (1.0 - ly) * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                    + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
            }
        }
    }
    out(&[n, c, oh, ow], y)
}

/// `F.interpolate(x, scale_factor=2.0, mode="nearest")`: output `i` reads
/// input `i / 2`.
pub fn interp_nearest2x(x: &WTensor) -> WTensor {
    let [n, c, h, w] = dims4(x);
    let (oh, ow) = (2 * h, 2 * w);
    let mut y = vec![0.0; n * c * oh * ow];
    for p in 0..n * c {
        for i in 0..oh {
            for j in 0..ow {
                y[(p * oh + i) * ow + j] = x.data[(p * h + i / 2) * w + j / 2] as f64;
            }
        }
    }
    out(&[n, c, oh, ow], y)
}

/// `F.interpolate(x, size, mode="bicubic", align_corners=False)`: Keys'
/// cubic with `A = -0.75`, the source coordinate `scale * (dst + 0.5) - 0.5`
/// left unclamped, and every tap index clamped into the image.
pub fn interp_bicubic(x: &WTensor, oh: usize, ow: usize) -> WTensor {
    let [n, c, h, w] = dims4(x);
    let xd = d(x);
    const A: f64 = -0.75;
    let cc1 = |x: f64| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let cc2 = |x: f64| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    let taps = |dst: usize, inn: usize, outn: usize| -> ([usize; 4], [f64; 4]) {
        let real = inn as f64 / outn as f64 * (dst as f64 + 0.5) - 0.5;
        let i0 = real.floor();
        let tt = real - i0;
        let idx = |k: isize| ((i0 as isize + k).clamp(0, inn as isize - 1)) as usize;
        (
            [idx(-1), idx(0), idx(1), idx(2)],
            [cc2(tt + 1.0), cc1(tt), cc1(1.0 - tt), cc2(2.0 - tt)],
        )
    };
    let mut y = vec![0.0; n * c * oh * ow];
    for p in 0..n * c {
        for i in 0..oh {
            let (yi, wy) = taps(i, h, oh);
            for j in 0..ow {
                let (xi, wx) = taps(j, w, ow);
                let mut acc = 0.0;
                for a in 0..4 {
                    for b in 0..4 {
                        acc += wy[a] * wx[b] * xd[(p * h + yi[a]) * w + xi[b]];
                    }
                }
                y[(p * oh + i) * ow + j] = acc;
            }
        }
    }
    out(&[n, c, oh, ow], y)
}

/// `F.interpolate(x, size, mode="bilinear", align_corners=False,
/// antialias=True)`: PIL's algorithm with a triangle filter - the support
/// widens by the shrink factor and the taps are normalized - one axis at a
/// time, width first.
pub fn interp_bilinear_aa(x: &WTensor, oh: usize, ow: usize) -> WTensor {
    let [n, c, h, w] = dims4(x);
    let tri = |x: f64| if x.abs() < 1.0 { 1.0 - x.abs() } else { 0.0 };
    let mut y = Vec::with_capacity(n * c * oh * ow);
    for p in 0..n * c {
        let plane: Vec<f64> = x.data[p * h * w..(p + 1) * h * w]
            .iter()
            .map(|&v| v as f64)
            .collect();
        let rows = resample_axis(&plane, w, h, ow, &tri, 1.0, true, false);
        let cols = resample_axis(&rows, ow, h, oh, &tri, 1.0, false, false);
        y.extend(cols);
    }
    out(&[n, c, oh, ow], y)
}

// ---- PIL -------------------------------------------------------------------

/// PIL's bicubic kernel: `a = -0.5`, support 2.
fn pil_bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// One separable pass of PIL's `precompute_coeffs` + resample, along the
/// width (`horizontal`) or the height. `support` is the filter's own half
/// width; when shrinking it is multiplied by the shrink factor and the
/// filter argument divided by it. `fixed8` reproduces the `uint8` path:
/// the normalized weights are rounded to 22-bit fixed point and each output
/// pixel is rounded and clipped to a byte before the next pass.
#[allow(clippy::too_many_arguments)]
fn resample_axis(
    src: &[f64],
    w: usize,
    h: usize,
    outn: usize,
    filter: &dyn Fn(f64) -> f64,
    support: f64,
    horizontal: bool,
    fixed8: bool,
) -> Vec<f64> {
    let inn = if horizontal { w } else { h };
    let scale = inn as f64 / outn as f64;
    let filterscale = scale.max(1.0);
    let sup = support * filterscale;
    let ss = 1.0 / filterscale;
    let ksize = sup.ceil() as usize * 2 + 1;
    // The taps of every output position, as PIL computes them: doubles,
    // normalized to one.
    let mut taps: Vec<(usize, Vec<f64>)> = Vec::with_capacity(outn);
    for xx in 0..outn {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - sup + 0.5) as isize).max(0) as usize;
        let xmax = ((center + sup + 0.5) as usize).min(inn);
        let mut k: Vec<f64> = (xmin..xmax)
            .map(|x| filter((x as f64 - center + 0.5) * ss))
            .collect();
        let ww: f64 = k.iter().sum();
        if ww != 0.0 {
            for v in &mut k {
                *v /= ww;
            }
        }
        k.resize(ksize, 0.0);
        if fixed8 {
            const PRECISION: f64 = (1u32 << 22) as f64;
            for v in &mut k {
                *v = if *v < 0.0 {
                    (-0.5 + *v * PRECISION).trunc()
                } else {
                    (0.5 + *v * PRECISION).trunc()
                };
            }
        }
        taps.push((xmin, k));
    }
    let (ow, oh) = if horizontal { (outn, h) } else { (w, outn) };
    let mut dst = vec![0.0; ow * oh];
    for oy in 0..oh {
        for ox in 0..ow {
            let (xmin, k) = if horizontal { &taps[ox] } else { &taps[oy] };
            let mut acc = if fixed8 { (1i64 << 21) as f64 } else { 0.0 };
            for (i, kv) in k.iter().enumerate() {
                let sv = if horizontal {
                    src.get(oy * w + xmin + i)
                } else {
                    src.get((xmin + i) * w + ox)
                };
                let Some(sv) = sv else { break };
                if xmin + i >= inn {
                    break;
                }
                acc += sv * kv;
            }
            dst[oy * ow + ox] = if fixed8 {
                ((acc as i64) >> 22).clamp(0, 255) as f64
            } else {
                acc as f32 as f64
            };
        }
    }
    dst
}

/// `PIL.Image.resize((ow, oh))` on a mode-`F` image: bicubic, doubles for
/// the taps and the sums, `float32` between the two passes.
pub fn pil_resize_f32(x: &WTensor, oh: usize, ow: usize) -> WTensor {
    let [h, w] = dims2(x);
    let rows = resample_axis(&d(x), w, h, ow, &pil_bicubic, 2.0, true, false);
    let cols = resample_axis(&rows, ow, h, oh, &pil_bicubic, 2.0, false, false);
    out(&[oh, ow], cols)
}

/// `PIL.Image.resize((ow, oh))` on an 8-bit image: the same taps in 22-bit
/// fixed point, every pass rounded and clipped to bytes.
pub fn pil_resize_u8(x: &WTensor, oh: usize, ow: usize) -> WTensor {
    let [h, w] = dims2(x);
    let rows = resample_axis(&d(x), w, h, ow, &pil_bicubic, 2.0, true, true);
    let cols = resample_axis(&rows, ow, h, oh, &pil_bicubic, 2.0, false, true);
    out(&[oh, ow], cols)
}

/// MedSAM2's slice preprocessing at `target` pixels: the byte image resized
/// as an RGB PIL image, divided by 255 and normalized with the ImageNet
/// statistics; returns the resized bytes and the `[1, 3, t, t]` input.
pub fn preprocess(u8: &WTensor, target: usize) -> (WTensor, WTensor) {
    let pil = pil_resize_u8(u8, target, target);
    const MEAN: [f64; 3] = [0.485, 0.456, 0.406];
    const STD: [f64; 3] = [0.229, 0.224, 0.225];
    let mut y = Vec::with_capacity(3 * target * target);
    for ch in 0..3 {
        for &v in &pil.data {
            y.push((v as f64 / 255.0 - MEAN[ch]) / STD[ch]);
        }
    }
    (pil, out(&[1, 3, target, target], y))
}

// ---- pointwise and normalization ---------------------------------------------

/// The error function to double precision: a Taylor series near zero, a
/// continued fraction for the complement beyond it.
pub fn erf(x: f64) -> f64 {
    let ax = x.abs();
    let r = if ax < 2.5 {
        let mut term = ax;
        let mut sum = ax;
        let x2 = ax * ax;
        let mut n = 0.0;
        while term.abs() > 1e-18 * sum.abs().max(1e-300) {
            n += 1.0;
            term *= -x2 / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc(x) = exp(-x²) / (x √π) · 1 / (1 + 1/(2x²) / (1 + 2/(2x²) / (1 + ...)))
        // evaluated as a Lentz continued fraction.
        let x2 = ax * ax;
        let mut f = ax;
        let mut c = ax;
        let mut dd = 0.0;
        for k in 1..200 {
            let a = k as f64 * 0.5;
            dd = ax + a * dd;
            dd = if dd == 0.0 { 1e-300 } else { 1.0 / dd };
            c = ax + a / c;
            if c == 0.0 {
                c = 1e-300;
            }
            let delta = c * dd;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (-x2).exp() / (f * std::f64::consts::PI.sqrt())
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

/// `F.gelu(x)`, the exact erf form.
pub fn gelu(x: &WTensor) -> WTensor {
    map(x, |v| 0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2)))
}

pub fn relu(x: &WTensor) -> WTensor {
    map(x, |v| v.max(0.0))
}

pub fn sigmoid(x: &WTensor) -> WTensor {
    map(x, |v| 1.0 / (1.0 + (-v).exp()))
}

fn map(x: &WTensor, f: impl Fn(f64) -> f64) -> WTensor {
    out(&x.shape, d(x).into_iter().map(f).collect())
}

/// `F.layer_norm(x, (last,), w, b, eps)`: biased variance over the last axis.
pub fn layer_norm_last(x: &WTensor, w: &WTensor, b: &WTensor, eps: f64) -> WTensor {
    let last = *x.shape.last().unwrap();
    let xd = d(x);
    let mut y = vec![0.0; xd.len()];
    for (row, o) in xd.chunks(last).zip(y.chunks_mut(last)) {
        let mean = row.iter().sum::<f64>() / last as f64;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / last as f64;
        for i in 0..last {
            o[i] = (row[i] - mean) / (var + eps).sqrt() * w.data[i] as f64 + b.data[i] as f64;
        }
    }
    out(&x.shape, y)
}

/// SAM's `LayerNorm2d`: mean and biased variance over the channel axis of an
/// `[N, C, H, W]` tensor, one pair per pixel.
pub fn layer_norm_2d(x: &WTensor, w: &WTensor, b: &WTensor, eps: f64) -> WTensor {
    let [n, c, h, wd] = dims4(x);
    let xd = d(x);
    let mut y = vec![0.0; xd.len()];
    for ni in 0..n {
        for p in 0..h * wd {
            let at = |ci: usize| (ni * c + ci) * h * wd + p;
            let mean = (0..c).map(|ci| xd[at(ci)]).sum::<f64>() / c as f64;
            let var = (0..c).map(|ci| (xd[at(ci)] - mean).powi(2)).sum::<f64>() / c as f64;
            for ci in 0..c {
                y[at(ci)] = (xd[at(ci)] - mean) / (var + eps).sqrt() * w.data[ci] as f64
                    + b.data[ci] as f64;
            }
        }
    }
    out(&x.shape, y)
}

/// `F.softmax(x, dim=-1)`.
pub fn softmax_last(x: &WTensor) -> WTensor {
    let last = *x.shape.last().unwrap();
    let xd = d(x);
    let mut y = vec![0.0; xd.len()];
    for (row, o) in xd.chunks(last).zip(y.chunks_mut(last)) {
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = row.iter().map(|v| (v - m).exp()).collect();
        let s: f64 = e.iter().sum();
        for i in 0..last {
            o[i] = e[i] / s;
        }
    }
    out(&x.shape, y)
}

/// `a @ b` for two matrices.
pub fn matmul(a: &WTensor, b: &WTensor) -> WTensor {
    let [m, k] = dims2(a);
    let [k2, n] = dims2(b);
    assert_eq!(k, k2);
    let (ad, bd) = (d(a), d(b));
    let mut y = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] = (0..k).map(|l| ad[i * k + l] * bd[l * n + j]).sum();
        }
    }
    out(&[m, n], y)
}

/// `F.scaled_dot_product_attention(q, k, v)` on `[B, heads, seq, dim]`:
/// `softmax(q kᵀ / √dim) v`, no mask.
pub fn sdpa(q: &WTensor, k: &WTensor, v: &WTensor) -> WTensor {
    let [b, hh, sq, dim] = dims4(q);
    let [_, _, sk, _] = dims4(k);
    let (qd, kd, vd) = (d(q), d(k), d(v));
    let mut y = vec![0.0; b * hh * sq * dim];
    for p in 0..b * hh {
        for i in 0..sq {
            let mut scores: Vec<f64> = (0..sk)
                .map(|j| {
                    (0..dim)
                        .map(|e| qd[(p * sq + i) * dim + e] * kd[(p * sk + j) * dim + e])
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            for s in &mut scores {
                *s = (*s - m).exp();
            }
            let z: f64 = scores.iter().sum();
            for e in 0..dim {
                y[(p * sq + i) * dim + e] = (0..sk)
                    .map(|j| scores[j] / z * vd[(p * sk + j) * dim + e])
                    .sum();
            }
        }
    }
    out(&[b, hh, sq, dim], y)
}

// ---- SAM 2's positional encodings -------------------------------------------

/// `PositionEmbeddingSine(num_pos_feats, temperature=10000, normalize=True)`
/// on an `h x w` grid: one-based row and column indices scaled to `2π`,
/// `num_pos_feats / 2` frequencies each, sines in the even and cosines in
/// the odd channels, the row channels before the column channels.
pub fn pe_sine(num_pos_feats: usize, temperature: f64, h: usize, w: usize) -> WTensor {
    let feats = num_pos_feats / 2;
    let scale = 2.0 * std::f64::consts::PI;
    let eps = 1e-6;
    let dim_t: Vec<f64> = (0..feats)
        .map(|i| temperature.powf(2.0 * (i / 2) as f64 / feats as f64))
        .collect();
    let encode = |embed: f64| -> Vec<f64> {
        (0..feats)
            .map(|c| {
                let p = embed / dim_t[c];
                if c % 2 == 0 {
                    p.sin()
                } else {
                    p.cos()
                }
            })
            .collect()
    };
    let mut y = vec![0.0; 2 * feats * h * w];
    for i in 0..h {
        for j in 0..w {
            let ye = (i + 1) as f64 / (h as f64 + eps) * scale;
            let xe = (j + 1) as f64 / (w as f64 + eps) * scale;
            for (c, v) in encode(ye).into_iter().chain(encode(xe)).enumerate() {
                y[(c * h + i) * w + j] = v;
            }
        }
    }
    out(&[1, 2 * feats, h, w], y)
}

/// `PositionEmbeddingRandom._pe_encoding` with a given `[2, feats]` Gaussian
/// matrix: coordinates in `[0, 1]` mapped to `[-1, 1]`, projected, scaled by
/// `2π`, then `[sin, cos]`.
fn pe_random_encode(gaussian: &WTensor, x: f64, y: f64) -> Vec<f64> {
    let feats = gaussian.shape[1];
    let (cx, cy) = (2.0 * x - 1.0, 2.0 * y - 1.0);
    let proj: Vec<f64> = (0..feats)
        .map(|f| {
            2.0 * std::f64::consts::PI
                * (cx * gaussian.data[f] as f64 + cy * gaussian.data[feats + f] as f64)
        })
        .collect();
    proj.iter()
        .map(|v| v.sin())
        .chain(proj.iter().map(|v| v.cos()))
        .collect()
}

/// `PositionEmbeddingRandom.forward((h, w))`: pixel centres `(j + 0.5) / w`,
/// `(i + 0.5) / h`, encoded and laid out `[2 * feats, h, w]`.
pub fn pe_random_dense(gaussian: &WTensor, h: usize, w: usize) -> WTensor {
    let c = 2 * gaussian.shape[1];
    let mut y = vec![0.0; c * h * w];
    for i in 0..h {
        for j in 0..w {
            let e = pe_random_encode(
                gaussian,
                (j as f64 + 0.5) / w as f64,
                (i as f64 + 0.5) / h as f64,
            );
            for (ch, v) in e.into_iter().enumerate() {
                y[(ch * h + i) * w + j] = v;
            }
        }
    }
    out(&[c, h, w], y)
}

/// `PositionEmbeddingRandom.forward_with_coords(coords, image_size)`:
/// `[B, n, 2]` pixel coordinates divided by the image size, encoded.
pub fn pe_random_coords(gaussian: &WTensor, coords: &WTensor, image: (usize, usize)) -> WTensor {
    let [b, n, two] = dims3(coords);
    assert_eq!(two, 2);
    let c = 2 * gaussian.shape[1];
    let mut y = Vec::with_capacity(b * n * c);
    for p in 0..b * n {
        let x = coords.data[p * 2] as f64 / image.1 as f64;
        let yy = coords.data[p * 2 + 1] as f64 / image.0 as f64;
        y.extend(pe_random_encode(gaussian, x, yy));
    }
    out(&[b, n, c], y)
}

/// `compute_axial_cis(dim, end_x, end_y, theta)`: `dim / 4` frequencies
/// `theta^(-4i/dim)`, token `t` at `(t % end_x, t / end_x)`, the x rotations
/// before the y rotations; returned as `[tokens, dim / 2]` (real, imaginary).
pub fn axial_cis(dim: usize, end_x: usize, end_y: usize, theta: f64) -> (WTensor, WTensor) {
    let nf = dim / 4;
    let freqs: Vec<f64> = (0..nf)
        .map(|i| 1.0 / theta.powf((4 * i) as f64 / dim as f64))
        .collect();
    let tokens = end_x * end_y;
    let (mut re, mut im) = (Vec::new(), Vec::new());
    for tk in 0..tokens {
        let (tx, ty) = ((tk % end_x) as f64, (tk / end_x) as f64);
        for f in freqs
            .iter()
            .map(|f| tx * f)
            .chain(freqs.iter().map(|f| ty * f))
        {
            re.push(f.cos());
            im.push(f.sin());
        }
    }
    (out(&[tokens, dim / 2], re), out(&[tokens, dim / 2], im))
}

/// `apply_rotary_enc` on one `[B, heads, seq, dim]` tensor: adjacent pairs
/// as complex numbers, multiplied by the rotation of their token; with
/// `repeat` the `[tokens, dim / 2]` rotations tile along a longer sequence.
pub fn rope(x: &WTensor, re: &WTensor, im: &WTensor) -> WTensor {
    let [b, hh, seq, dim] = dims4(x);
    let tokens = re.shape[0];
    assert_eq!(seq % tokens, 0, "repeat_freqs_k tiles whole copies");
    let half = dim / 2;
    let xd = d(x);
    let mut y = vec![0.0; xd.len()];
    for p in 0..b * hh {
        for s in 0..seq {
            for c in 0..half {
                let (a, bb) = (
                    xd[(p * seq + s) * dim + 2 * c],
                    xd[(p * seq + s) * dim + 2 * c + 1],
                );
                let (fr, fi) = (
                    re.data[(s % tokens) * half + c] as f64,
                    im.data[(s % tokens) * half + c] as f64,
                );
                y[(p * seq + s) * dim + 2 * c] = a * fr - bb * fi;
                y[(p * seq + s) * dim + 2 * c + 1] = a * fi + bb * fr;
            }
        }
    }
    out(&x.shape, y)
}

// ---- shapes ------------------------------------------------------------------

fn dims4(x: &WTensor) -> [usize; 4] {
    assert_eq!(x.shape.len(), 4, "{:?}", x.shape);
    [x.shape[0], x.shape[1], x.shape[2], x.shape[3]]
}

fn dims3(x: &WTensor) -> [usize; 3] {
    assert_eq!(x.shape.len(), 3, "{:?}", x.shape);
    [x.shape[0], x.shape[1], x.shape[2]]
}

fn dims2(x: &WTensor) -> [usize; 2] {
    assert_eq!(x.shape.len(), 2, "{:?}", x.shape);
    [x.shape[0], x.shape[1]]
}

/// The largest absolute difference between two tensors of one shape.
pub fn max_abs_diff(a: &WTensor, b: &WTensor) -> f64 {
    assert_eq!(a.shape, b.shape, "shape");
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .fold(0.0, f64::max)
}

// ---- the fixture itself -------------------------------------------------------

/// A small deterministic normal generator (xorshift64* and Box-Muller), so
/// the fixture is reproducible from its seed without a random-number crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `(0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Standard normal.
    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// A tensor of standard normals.
    pub fn randn(&mut self, shape: &[usize]) -> WTensor {
        let n: usize = shape.iter().product();
        t(shape, (0..n).map(|_| self.normal() as f32).collect())
    }
}

/// Every tensor of the op fixture, computed from fresh random inputs: the
/// same names, shapes and semantics as the file PyTorch wrote, so the tests
/// in `src/medsam2` read either without knowing which they got.
pub fn generate(seed: u64) -> std::collections::HashMap<String, WTensor> {
    let mut rng = Rng::new(seed);
    let mut out = std::collections::HashMap::new();
    let mut put = |name: &str, tensors: Vec<(&str, WTensor)>| {
        for (k, v) in tensors {
            out.insert(format!("{name}.{k}"), v);
        }
    };

    // convolutions: the patch embed, the mask downsampler, a neck
    // projection, a CXBlock depthwise, the decoder's upscaling
    for (name, cin, cout, k, stride, pad, groups, size) in [
        ("conv_k7s4p3", 3, 8, 7, 4, 3, 1, 32),
        ("conv_k3s2p1", 4, 6, 3, 2, 1, 1, 16),
        ("conv_k1", 6, 5, 1, 1, 0, 1, 7),
        ("conv_dw", 6, 6, 7, 1, 3, 6, 12),
    ] {
        let x = rng.randn(&[1, cin, size, size]);
        let w = rng.randn(&[cout, cin / groups, k, k]);
        let b = rng.randn(&[cout]);
        let y = conv2d(&x, &w, &b, stride, pad, groups);
        put(name, vec![("x", x), ("w", w), ("b", b), ("y", y)]);
    }
    let (x, w, b) = (
        rng.randn(&[1, 6, 8, 8]),
        rng.randn(&[6, 4, 2, 2]),
        rng.randn(&[4]),
    );
    let y = conv_transpose2d(&x, &w, &b, 2);
    put("convt_k2s2", vec![("x", x), ("w", w), ("b", b), ("y", y)]);

    // pooling, on an odd grid so ceil_mode=False actually drops a row
    let x = rng.randn(&[1, 3, 9, 9]);
    let y = maxpool2x2(&x);
    put("maxpool2x2", vec![("x", x), ("y", y)]);

    // interpolation
    let x = rng.randn(&[1, 2, 8, 8]);
    let y = interp_bilinear(&x, 20, 20);
    put("interp_bilinear", vec![("x", x), ("y", y)]);
    let x = rng.randn(&[1, 2, 5, 5]);
    let y = interp_nearest2x(&x);
    put("interp_nearest", vec![("x", x), ("y", y)]);
    let x = rng.randn(&[1, 3, 7, 7]);
    let y = interp_bicubic(&x, 32, 32);
    put("interp_bicubic", vec![("x", x), ("y", y)]);

    // pointwise and normalization
    let x = t(
        &[64],
        rng.randn(&[64]).data.iter().map(|v| v * 3.0).collect(),
    );
    put("gelu", vec![("x", x.clone()), ("y", gelu(&x))]);
    put("relu", vec![("x", x.clone()), ("y", relu(&x))]);
    put("sigmoid", vec![("x", x.clone()), ("y", sigmoid(&x))]);
    let (x, w, b) = (rng.randn(&[5, 12]), rng.randn(&[12]), rng.randn(&[12]));
    let y = layer_norm_last(&x, &w, &b, 1e-6);
    put(
        "layernorm_last",
        vec![("x", x), ("w", w), ("b", b), ("y", y)],
    );
    let (x, w, b) = (rng.randn(&[1, 6, 4, 5]), rng.randn(&[6]), rng.randn(&[6]));
    let y = layer_norm_2d(&x, &w, &b, 1e-6);
    put("layernorm2d", vec![("x", x), ("w", w), ("b", b), ("y", y)]);
    let x = rng.randn(&[4, 9]);
    let y = softmax_last(&x);
    put("softmax", vec![("x", x), ("y", y)]);
    let (a, b) = (rng.randn(&[6, 5]), rng.randn(&[5, 7]));
    let y = matmul(&a, &b);
    put("matmul", vec![("a", a), ("b", b), ("y", y)]);

    // attention: two heads of width 4, the queries shorter than the keys
    let (q, k, v) = (
        rng.randn(&[1, 2, 6, 4]),
        rng.randn(&[1, 2, 10, 4]),
        rng.randn(&[1, 2, 10, 4]),
    );
    let y = sdpa(&q, &k, &v);
    put("sdpa", vec![("q", q), ("k", k), ("v", v), ("y", y)]);

    // resampling: PIL's kernel both ways, and PyTorch's antialiased one
    let x = rng.randn(&[7, 5]);
    let y = pil_resize_f32(&x, 16, 13);
    put("pil_up", vec![("x", x), ("y", y)]);
    let x = rng.randn(&[32, 32]);
    let y = pil_resize_f32(&x, 12, 12);
    put("pil_down", vec![("x", x), ("y", y)]);
    let x = rng.randn(&[1, 1, 16, 16]);
    let y = interp_bilinear_aa(&x, 4, 4);
    put("torch_bilinear_aa", vec![("x", x), ("y", y)]);

    // the preprocessing pipeline on one windowed slice, at a size that fits
    let u8 = t(
        &[40, 36],
        (0..40 * 36)
            .map(|_| (rng.uniform() * 255.0).floor() as f32)
            .collect(),
    );
    let (pil, y) = preprocess(&u8, 64);
    put("preprocess", vec![("u8", u8), ("pil_u8", pil), ("y", y)]);

    // SAM 2's positional encodings
    put("pe_sine", vec![("y", pe_sine(8, 10000.0, 3, 4))]);
    let gaussian = rng.randn(&[2, 4]);
    let coords = t(&[1, 2, 2], vec![10.0, 20.0, 30.0, 40.0]);
    let dense = pe_random_dense(&gaussian, 3, 4);
    let y = pe_random_coords(&gaussian, &coords, (64, 64));
    put(
        "pe_random",
        vec![
            ("gaussian", gaussian),
            ("dense", dense),
            ("coords", coords),
            ("y", y),
        ],
    );
    // 2-D axial RoPE: head dim 16 over a 3 x 4 grid, then keys three times
    // as long, which tiles the rotations across memory frames
    let (re, im) = axial_cis(16, 4, 3, 10000.0);
    let (q, k) = (rng.randn(&[1, 2, 12, 16]), rng.randn(&[1, 2, 12, 16]));
    let (q_out, k_out) = (rope(&q, &re, &im), rope(&k, &re, &im));
    let k_long = rng.randn(&[1, 2, 36, 16]);
    let k_long_out = rope(&k_long, &re, &im);
    put(
        "rope_repeat",
        vec![
            ("k", k_long),
            ("q_out", q_out.clone()),
            ("k_out", k_long_out),
        ],
    );
    put(
        "rope",
        vec![
            ("freqs_real", re),
            ("freqs_imag", im),
            ("q", q),
            ("k", k),
            ("q_out", q_out),
            ("k_out", k_out),
        ],
    );
    out
}
