//! Pure-Rust CPU inference engine for small 3D CNNs.
//!
//! The only heavy primitive a (Plain-Conv) nnU-Net needs is 3D convolution;
//! it is implemented as per-output-slice im2col + SIMD GEMM (`gemm` crate),
//! parallelized over output slices with rayon - 15-50× faster than a direct
//! scalar convolution loop. The remaining ops (transposed conv with
//! kernel = stride = 2, instance norm, leaky ReLU, channel concat) are
//! memory-bound and hand-rolled.
//!
//! Tensor layout: `[C, D, H, W]`, C-contiguous, batch size fixed at 1.

use rayon::prelude::*;

// The activation volume and the transposed convolution are shared with the
// SegVol mask decoder, so they live in `nn`; re-exported here because this
// is where the nnU-Net code has always reached for them.
use crate::nn::tensor::SendPtr;
pub use crate::nn::tensor::{conv_transpose3d_2x, conv_transpose3d_stride, Act};

#[inline]
fn conv_out(len: usize, k: usize, s: usize) -> usize {
    // padding = k / 2 on both sides (nnU-Net convention)
    (len + 2 * (k / 2) - k) / s + 1
}

/// A slice whose whole im2col block is at most this many floats (4 MB) is
/// one work item, as it always was: small enough to stay in cache, and then
/// the arithmetic is exactly the per-slice GEMM's.
const UNTILED_MAX_FLOATS: usize = 1 << 20;

/// Floats of im2col one tile builds when a slice is split: about 1 MB, so
/// the block a GEMM reads stays in the core's L2 instead of streaming
/// through memory.
const TILE_FLOATS: usize = 1 << 18;

/// Fewest output columns a tile multiplies at once, so the GEMM computes
/// rather than re-reads its weights.
const MIN_TILE_COLS: usize = 96;

thread_local! {
    /// Per-thread im2col scratch, reused across tiles, layers and patches.
    /// Bounded by [`UNTILED_MAX_FLOATS`] or one tile, so it costs a few
    /// megabytes per worker for the life of the pool, against a fresh
    /// allocation - tens of megabytes at full resolution, and their page
    /// faults - for every output slice of every layer.
    static COL: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 3D convolution, padding `k/2`, arbitrary stride, bias included.
/// `weight`: `[cout, cin, kd, kh, kw]` C-contiguous.
///
/// One output slice is one GEMM over its im2col block, in parallel over the
/// slices; the GEMM writes straight into the output (no staging buffer) and
/// the block lives in a per-thread buffer that is reused. At full
/// resolution one slice's block is tens of megabytes (32 channels x 27 taps
/// x 112 x 128 values), which streams through memory twice; such a slice is
/// split into tiles of a few output rows whose blocks fit in cache, which is
/// about twice as fast there. The GEMM of a tile blocks its sums over `k`
/// its own way, so a tiled layer can differ from the untiled one in the last
/// bits (float rounding, deterministic either way); every smaller layer is
/// computed exactly as before.
pub fn conv3d(
    x: &Act,
    weight: &[f32],
    bias: &[f32],
    cout: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
) -> Act {
    let out = [
        conv_out(x.d, kernel[0], stride[0]),
        conv_out(x.h, kernel[1], stride[1]),
        conv_out(x.w, kernel[2], stride[2]),
    ];
    let pad = [kernel[0] / 2, kernel[1] / 2, kernel[2] / 2];
    conv3d_padded(x, weight, bias, cout, kernel, stride, pad, out)
}

/// [`conv3d`] with the padding before each axis and the output size given:
/// output voxel `o` reads input `o * stride - pad + k` for each tap `k`,
/// zero outside the input. Any kernel size, odd or even.
#[allow(clippy::too_many_arguments)]
pub fn conv3d_padded(
    x: &Act,
    weight: &[f32],
    bias: &[f32],
    cout: usize,
    kernel: [usize; 3],
    stride: [usize; 3],
    pad: [usize; 3],
    out_dims: [usize; 3],
) -> Act {
    let (cin, d, h, w) = (x.c, x.d, x.h, x.w);
    let [kd, kh, kw] = kernel;
    let [sd, sh, sw] = stride;
    debug_assert_eq!(weight.len(), cout * cin * kd * kh * kw);
    let [od, oh, ow] = out_dims;
    let ohw = oh * ow;
    let k = cin * kd * kh * kw;
    let (pd, ph, pw) = (pad[0] as isize, pad[1] as isize, pad[2] as isize);
    let mut out = Act::zeros(cout, od, oh, ow);
    if out.data.is_empty() || k == 0 {
        return out;
    }
    let out_ptr = SendPtr(out.data.as_mut_ptr());
    let od_stride = od * ohw; // per-channel stride in the output
    let rows_per_tile = if k * ohw <= UNTILED_MAX_FLOATS {
        oh
    } else {
        (TILE_FLOATS / k)
            .max(MIN_TILE_COLS)
            .div_ceil(ow)
            .clamp(1, oh)
    };
    let tiles_per_slice = oh.div_ceil(rows_per_tile);
    (0..od * tiles_per_slice).into_par_iter().for_each(|t| {
        let oz = t / tiles_per_slice;
        let y0 = (t % tiles_per_slice) * rows_per_tile;
        let y1 = (y0 + rows_per_tile).min(oh);
        let n = (y1 - y0) * ow;
        COL.with_borrow_mut(|col| {
            // im2col for output rows y0..y1 of slice oz: [k, n]. Zeroed
            // first: what the loops below skip is the padding.
            col.clear();
            col.resize(k * n, 0.0);
            let mut row = 0usize;
            for c in 0..cin {
                let cbase = c * d * h * w;
                for kz in 0..kd {
                    let iz = (oz * sd) as isize + kz as isize - pd;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let dst = &mut col[row * n..(row + 1) * n];
                            row += 1;
                            if iz < 0 || iz >= d as isize {
                                continue;
                            }
                            let zbase = cbase + iz as usize * h * w;
                            let dy = ky as isize - ph;
                            let dx = kx as isize - pw;
                            for oy in y0..y1 {
                                let iy = (oy * sh) as isize + dy;
                                if iy < 0 || iy >= h as isize {
                                    continue;
                                }
                                let src_row =
                                    &x.data[zbase + iy as usize * w..zbase + iy as usize * w + w];
                                let drow = &mut dst[(oy - y0) * ow..(oy - y0 + 1) * ow];
                                if sw == 1 {
                                    // contiguous copy with edge clipping
                                    let (o0, o1) = if dx < 0 {
                                        (
                                            ((-dx) as usize).min(ow),
                                            ow.min((w as isize - dx) as usize),
                                        )
                                    } else {
                                        (0, ow.min(w.saturating_sub(dx as usize)))
                                    };
                                    if o0 < o1 {
                                        let s0 = (o0 as isize + dx) as usize;
                                        drow[o0..o1].copy_from_slice(&src_row[s0..s0 + (o1 - o0)]);
                                    }
                                } else {
                                    for (oxi, dv) in drow.iter_mut().enumerate() {
                                        let ixp = (oxi * sw) as isize + dx;
                                        if ixp >= 0 && (ixp as usize) < w {
                                            *dv = src_row[ixp as usize];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // GEMM: [cout, k] × [k, n] → the output itself, channel-major:
            // row c of the product is channel c's rows y0..y1 of slice oz.
            // Tiles cover disjoint output ranges, so the parallel writes
            // never meet.
            let dst0 = oz * ohw + y0 * ow;
            unsafe {
                gemm::gemm(
                    cout,
                    n,
                    k,
                    out_ptr.get().add(dst0),
                    1,                  // dst col stride
                    od_stride as isize, // dst row stride: one channel
                    false,
                    weight.as_ptr(),
                    1,
                    k as isize,
                    col.as_ptr(),
                    1,
                    n as isize,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            for (c, &bv) in bias.iter().enumerate().take(cout) {
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.get().add(c * od_stride + dst0), n)
                };
                for v in dst.iter_mut() {
                    *v += bv;
                }
            }
        });
    });
    out
}

/// `ConvTranspose3d(kernel 3, stride 2, padding 1, output_padding 1)` -
/// SegResNetDS's upsampling - as eight small convolutions, one per parity
/// of the output index: along an axis, an even output `2m` takes tap 1 of
/// input `m`, an odd one `2m + 1` taps 2 and 0 of inputs `m` and `m + 1`.
/// `weight` is PyTorch's `[cin, cout, 3, 3, 3]`. The output is twice the
/// input along every axis.
pub fn conv_transpose3d_k3s2(x: &Act, weight: &[f32], bias: &[f32], cout: usize) -> Act {
    let cin = x.c;
    debug_assert_eq!(weight.len(), cin * cout * 27);
    let n = [x.d, x.h, x.w];
    let mut out = Act::zeros(cout, 2 * n[0], 2 * n[1], 2 * n[2]);
    let zero = vec![0f32; cout];
    // taps along one axis for a parity, in correlation order (input offset 0, 1)
    let taps = |parity: usize| -> &'static [usize] {
        if parity == 0 {
            &[1]
        } else {
            &[2, 0]
        }
    };
    for pz in 0..2 {
        for py in 0..2 {
            for px in 0..2 {
                let (tz, ty, tx) = (taps(pz), taps(py), taps(px));
                let kernel = [tz.len(), ty.len(), tx.len()];
                let mut w = Vec::with_capacity(cout * cin * kernel.iter().product::<usize>());
                for co in 0..cout {
                    for ci in 0..cin {
                        for &kz in tz {
                            for &ky in ty {
                                for &kx in tx {
                                    w.push(weight[(((ci * cout + co) * 3 + kz) * 3 + ky) * 3 + kx]);
                                }
                            }
                        }
                    }
                }
                let part = conv3d_padded(x, &w, &zero, cout, kernel, [1, 1, 1], [0, 0, 0], n);
                let o = &mut out;
                let (od, oh, ow) = (o.d, o.h, o.w);
                o.data
                    .par_chunks_mut(od * oh * ow)
                    .zip(part.data.par_chunks(n[0] * n[1] * n[2]))
                    .enumerate()
                    .for_each(|(c, (dst, src))| {
                        let b = bias.get(c).copied().unwrap_or(0.0);
                        for z in 0..n[0] {
                            for y in 0..n[1] {
                                let s = (z * n[1] + y) * n[2];
                                let d0 = ((2 * z + pz) * oh + 2 * y + py) * ow + px;
                                for xx in 0..n[2] {
                                    dst[d0 + 2 * xx] = src[s + xx] + b;
                                }
                            }
                        }
                    });
            }
        }
    }
    out
}

/// InstanceNorm3d (affine, eps 1e-5, biased variance) fused with
/// LeakyReLU(0.01) - the pairing every nnU-Net conv block uses.
pub fn instance_norm_lrelu(x: &mut Act, gamma: &[f32], beta: &[f32]) {
    instance_norm_impl(x, gamma, beta, true);
}

/// InstanceNorm3d alone - the second conv of a residual block is
/// normalized, added to its skip path, and only then activated
/// ([`add_lrelu`]).
pub fn instance_norm(x: &mut Act, gamma: &[f32], beta: &[f32]) {
    instance_norm_impl(x, gamma, beta, false);
}

fn instance_norm_impl(x: &mut Act, gamma: &[f32], beta: &[f32], lrelu: bool) {
    let n = x.spatial();
    let inv_n = 1.0 / n as f64;
    x.data.par_chunks_mut(n).enumerate().for_each(|(c, ch)| {
        let mut sum = 0f64;
        let mut sq = 0f64;
        for v in ch.iter() {
            let v = *v as f64;
            sum += v;
            sq += v * v;
        }
        let mean = sum * inv_n;
        let var = (sq * inv_n - mean * mean).max(0.0);
        let scale = (gamma[c] as f64 / (var + 1e-5).sqrt()) as f32;
        let shift = beta[c] - mean as f32 * scale;
        if lrelu {
            for v in ch.iter_mut() {
                let y = *v * scale + shift;
                *v = if y >= 0.0 { y } else { 0.01 * y };
            }
        } else {
            for v in ch.iter_mut() {
                *v = *v * scale + shift;
            }
        }
    });
}

/// `x = LeakyReLU(x + r)`, the end of a residual block.
pub fn add_lrelu(x: &mut Act, r: &Act) {
    debug_assert!(x.c == r.c && x.d == r.d && x.h == r.h && x.w == r.w);
    x.data
        .par_chunks_mut(1 << 14)
        .zip(r.data.par_chunks(1 << 14))
        .for_each(|(xs, rs)| {
            for (v, a) in xs.iter_mut().zip(rs) {
                let y = *v + *a;
                *v = if y >= 0.0 { y } else { 0.01 * y };
            }
        });
}

/// AvgPool3d with kernel = stride = `k` and no padding (PyTorch's
/// `AvgPool3d(k, k)` with `count_include_pad` immaterial because nothing is
/// padded): the skip path of a strided residual block. Trailing voxels that
/// do not fill a window are dropped, as PyTorch drops them.
pub fn avg_pool3d(x: &Act, k: [usize; 3]) -> Act {
    let [kd, kh, kw] = k;
    debug_assert!(kd >= 1 && kh >= 1 && kw >= 1);
    let (od, oh, ow) = (x.d / kd, x.h / kh, x.w / kw);
    let mut out = Act::zeros(x.c, od, oh, ow);
    let inv = 1.0 / (kd * kh * kw) as f32;
    let (d, h, w) = (x.d, x.h, x.w);
    let in_plane = h * w;
    let out_plane = oh * ow;
    out.data
        .par_chunks_mut(od * out_plane)
        .enumerate()
        .for_each(|(c, oc)| {
            let ic = &x.data[c * d * in_plane..(c + 1) * d * in_plane];
            for oz in 0..od {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut s = 0f32;
                        for kz in 0..kd {
                            let z = oz * kd + kz;
                            for ky in 0..kh {
                                let base = z * in_plane + (oy * kh + ky) * w + ox * kw;
                                for v in &ic[base..base + kw] {
                                    s += *v;
                                }
                            }
                        }
                        oc[oz * out_plane + oy * ow + ox] = s * inv;
                    }
                }
            }
        });
    out
}

/// Channel-wise concatenation `[a; b]`.
pub fn concat(a: &Act, b: &Act) -> Act {
    debug_assert!(a.d == b.d && a.h == b.h && a.w == b.w);
    let mut out = Act::zeros(a.c + b.c, a.d, a.h, a.w);
    let (sa, sb) = (a.c * a.spatial(), b.c * b.spatial());
    out.data[..sa].copy_from_slice(&a.data);
    out.data[sa..sa + sb].copy_from_slice(&b.data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct (naive) conv3d for verification.
    fn conv3d_naive(
        x: &Act,
        weight: &[f32],
        bias: &[f32],
        cout: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
    ) -> Act {
        let (cin, d, h, w) = (x.c, x.d, x.h, x.w);
        let [kd, kh, kw] = kernel;
        let [sd, sh, sw] = stride;
        let (od, oh, ow) = (
            conv_out(d, kd, sd),
            conv_out(h, kh, sh),
            conv_out(w, kw, sw),
        );
        let (pd, ph, pw) = ((kd / 2) as isize, (kh / 2) as isize, (kw / 2) as isize);
        let mut out = Act::zeros(cout, od, oh, ow);
        for co in 0..cout {
            for oz in 0..od {
                for oy in 0..oh {
                    for ox in 0..ow {
                        let mut acc = bias[co];
                        for ci in 0..cin {
                            for kz in 0..kd {
                                let iz = (oz * sd) as isize + kz as isize - pd;
                                if iz < 0 || iz >= d as isize {
                                    continue;
                                }
                                for ky in 0..kh {
                                    let iy = (oy * sh) as isize + ky as isize - ph;
                                    if iy < 0 || iy >= h as isize {
                                        continue;
                                    }
                                    for kx in 0..kw {
                                        let ix = (ox * sw) as isize + kx as isize - pw;
                                        if ix < 0 || ix >= w as isize {
                                            continue;
                                        }
                                        let xv = x.data[((ci * d + iz as usize) * h + iy as usize)
                                            * w
                                            + ix as usize];
                                        let wv = weight
                                            [(((co * cin + ci) * kd + kz) * kh + ky) * kw + kx];
                                        acc += xv * wv;
                                    }
                                }
                            }
                        }
                        out.data[((co * od + oz) * oh + oy) * ow + ox] = acc;
                    }
                }
            }
        }
        out
    }

    /// The per-slice implementation this module used before the scratch
    /// reuse and the tiling, kept as the reference both paths are held to.
    fn conv3d_per_slice(
        x: &Act,
        weight: &[f32],
        bias: &[f32],
        cout: usize,
        kernel: [usize; 3],
        stride: [usize; 3],
    ) -> Act {
        let (cin, d, h, w) = (x.c, x.d, x.h, x.w);
        let [kd, kh, kw] = kernel;
        let [sd, sh, sw] = stride;
        debug_assert_eq!(weight.len(), cout * cin * kd * kh * kw);
        let (od, oh, ow) = (
            conv_out(d, kd, sd),
            conv_out(h, kh, sh),
            conv_out(w, kw, sw),
        );
        let ohw = oh * ow;
        let k = cin * kd * kh * kw;
        let (pd, ph, pw) = ((kd / 2) as isize, (kh / 2) as isize, (kw / 2) as isize);
        let mut out = Act::zeros(cout, od, oh, ow);
        let out_ptr = SendPtr(out.data.as_mut_ptr());
        let od_stride = od * ohw; // per-channel stride in the output
        (0..od).into_par_iter().for_each(|oz| {
            // im2col for this output slice: [k, ohw]
            let mut col = vec![0f32; k * ohw];
            let mut row = 0usize;
            for c in 0..cin {
                let cbase = c * d * h * w;
                for kz in 0..kd {
                    let iz = (oz * sd) as isize + kz as isize - pd;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let dst = &mut col[row * ohw..(row + 1) * ohw];
                            row += 1;
                            if iz < 0 || iz >= d as isize {
                                continue;
                            }
                            let zbase = cbase + iz as usize * h * w;
                            let dy = ky as isize - ph;
                            let dx = kx as isize - pw;
                            for oy in 0..oh {
                                let iy = (oy * sh) as isize + dy;
                                if iy < 0 || iy >= h as isize {
                                    continue;
                                }
                                let src_row =
                                    &x.data[zbase + iy as usize * w..zbase + iy as usize * w + w];
                                let drow = &mut dst[oy * ow..(oy + 1) * ow];
                                if sw == 1 {
                                    // contiguous copy with edge clipping
                                    let (o0, o1) = if dx < 0 {
                                        (
                                            ((-dx) as usize).min(ow),
                                            ow.min((w as isize - dx) as usize),
                                        )
                                    } else {
                                        (0, ow.min(w.saturating_sub(dx as usize)))
                                    };
                                    if o0 < o1 {
                                        let s0 = (o0 as isize + dx) as usize;
                                        drow[o0..o1].copy_from_slice(&src_row[s0..s0 + (o1 - o0)]);
                                    }
                                } else {
                                    for (oxi, dv) in drow.iter_mut().enumerate() {
                                        let ixp = (oxi * sw) as isize + dx;
                                        if ixp >= 0 && (ixp as usize) < w {
                                            *dv = src_row[ixp as usize];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // GEMM: [cout, k] × [k, ohw] → [cout, ohw]
            let mut tmp = vec![0f32; cout * ohw];
            unsafe {
                gemm::gemm(
                    cout,
                    ohw,
                    k,
                    tmp.as_mut_ptr(),
                    1,            // dst col stride
                    ohw as isize, // dst row stride
                    false,
                    weight.as_ptr(),
                    1,
                    k as isize,
                    col.as_ptr(),
                    1,
                    ohw as isize,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            // scatter + bias into the channel-major output; slices are disjoint
            // across the parallel oz loop.
            for c in 0..cout {
                let bv = bias[c];
                let src = &tmp[c * ohw..(c + 1) * ohw];
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.get().add(c * od_stride + oz * ohw), ohw)
                };
                for (dv, sv) in dst.iter_mut().zip(src.iter()) {
                    *dv = sv + bv;
                }
            }
        });
        out
    }

    fn rngf(seed: &mut u64) -> f32 {
        // xorshift
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    }

    fn rand_act(c: usize, d: usize, h: usize, w: usize, seed: u64) -> Act {
        let mut s = seed | 1;
        let mut a = Act::zeros(c, d, h, w);
        for v in &mut a.data {
            *v = rngf(&mut s);
        }
        a
    }

    #[test]
    fn conv3d_matches_naive() {
        for (stride, dims) in [
            ([1, 1, 1], (5, 7, 6)),
            ([2, 2, 2], (6, 8, 7)),
            ([2, 2, 2], (5, 7, 9)), // odd sizes
        ] {
            let x = rand_act(3, dims.0, dims.1, dims.2, 42);
            let mut s = 7u64;
            let w: Vec<f32> = (0..4 * 3 * 27).map(|_| rngf(&mut s)).collect();
            let b: Vec<f32> = (0..4).map(|_| rngf(&mut s)).collect();
            let fast = conv3d(&x, &w, &b, 4, [3, 3, 3], stride);
            let slow = conv3d_naive(&x, &w, &b, 4, [3, 3, 3], stride);
            assert_eq!(fast.data.len(), slow.data.len());
            for (a, b) in fast.data.iter().zip(slow.data.iter()) {
                assert!((a - b).abs() < 1e-4, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn conv1x1_matches_naive() {
        let x = rand_act(6, 4, 5, 3, 11);
        let mut s = 3u64;
        let w: Vec<f32> = (0..5 * 6).map(|_| rngf(&mut s)).collect();
        let b: Vec<f32> = (0..5).map(|_| rngf(&mut s)).collect();
        let fast = conv3d(&x, &w, &b, 5, [1, 1, 1], [1, 1, 1]);
        let slow = conv3d_naive(&x, &w, &b, 5, [1, 1, 1], [1, 1, 1]);
        for (a, b) in fast.data.iter().zip(slow.data.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn tiled_and_whole_slice_convolutions_agree() {
        // Shapes on both sides of the tiling threshold: a slice that splits
        // into several tiles (with a ragged last one), strided and 1x1 ones,
        // and deep, small ones that stay whole.
        for (cin, cout, kernel, stride, dims) in [
            (32, 6, [3, 3, 3], [1, 1, 1], (3, 61, 147)),
            (16, 6, [3, 3, 3], [2, 2, 2], (5, 181, 301)),
            (8, 6, [3, 3, 3], [1, 1, 1], (5, 61, 47)),
            (3, 5, [1, 1, 1], [1, 1, 1], (4, 90, 70)),
            (64, 32, [3, 3, 3], [1, 1, 1], (3, 9, 7)),
            (4, 4, [1, 3, 3], [1, 2, 2], (3, 101, 57)),
        ] {
            let x = rand_act(cin, dims.0, dims.1, dims.2, 5);
            let mut s = 9u64;
            let kv = kernel[0] * kernel[1] * kernel[2];
            let w: Vec<f32> = (0..cout * cin * kv).map(|_| rngf(&mut s)).collect();
            let b: Vec<f32> = (0..cout).map(|_| rngf(&mut s)).collect();
            let got = conv3d(&x, &w, &b, cout, kernel, stride);
            let reference = conv3d_per_slice(&x, &w, &b, cout, kernel, stride);
            assert_eq!(got.data.len(), reference.data.len());
            let ohw = got.h * got.w;
            let tiled = cin * kv * ohw > UNTILED_MAX_FLOATS;
            for (a, r) in got.data.iter().zip(&reference.data) {
                if tiled {
                    assert!(
                        (a - r).abs() <= 1e-5 * (1.0 + r.abs()),
                        "{cin}->{cout} {kernel:?}/{stride:?} on {dims:?}: {a} vs {r}"
                    );
                } else {
                    assert_eq!(
                        a.to_bits(),
                        r.to_bits(),
                        "{cin}->{cout} {kernel:?}/{stride:?} on {dims:?}: {a} vs {r}"
                    );
                }
            }
        }
    }

    #[test]
    fn instance_norm_normalizes() {
        let mut x = rand_act(2, 8, 8, 8, 77);
        for v in &mut x.data {
            *v = *v * 3.0 + 1.0;
        }
        let gamma = vec![1.0f32; 2];
        let beta = vec![0.0f32; 2];
        instance_norm_lrelu(&mut x, &gamma, &beta);
        // after norm+lrelu, positive part should have mean≈0 pre-lrelu;
        // verify with an analytic re-check on channel 0 statistics instead:
        // reconstruct pre-lrelu values (invertible: y>=0 → y, y<0 → y/0.01)
        let n = x.spatial();
        let pre: Vec<f64> = x.data[..n]
            .iter()
            .map(|&v| if v >= 0.0 { v as f64 } else { v as f64 / 0.01 })
            .collect();
        let mean = pre.iter().sum::<f64>() / n as f64;
        let var = pre.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-3, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "var {var}");
    }

    #[test]
    fn instance_norm_without_activation_keeps_the_negative_half() {
        let mut x = rand_act(2, 6, 6, 6, 5);
        let gamma = vec![2.0f32, 0.5];
        let beta = vec![0.25f32, -1.0];
        instance_norm(&mut x, &gamma, &beta);
        let n = x.spatial();
        for c in 0..2 {
            let ch = &x.data[c * n..(c + 1) * n];
            let mean = ch.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
            let var = ch
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / n as f64;
            assert!(
                (mean - beta[c] as f64).abs() < 1e-3,
                "channel {c} mean {mean}"
            );
            assert!(
                (var.sqrt() - gamma[c] as f64).abs() < 1e-2,
                "channel {c} std {}",
                var.sqrt()
            );
            assert!(ch.iter().any(|v| *v < 0.0), "nothing was rectified");
        }
    }

    #[test]
    fn add_lrelu_adds_then_rectifies() {
        let mut x = Act {
            c: 1,
            d: 1,
            h: 1,
            w: 4,
            data: vec![1.0, -1.0, 0.5, -3.0],
        };
        let r = Act {
            c: 1,
            d: 1,
            h: 1,
            w: 4,
            data: vec![1.0, -1.0, -1.0, 1.0],
        };
        add_lrelu(&mut x, &r);
        let want = [2.0f32, -0.02, -0.005, -0.02];
        for (a, b) in x.data.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{:?}", x.data);
        }
    }

    #[test]
    fn avg_pool_matches_the_window_means() {
        // 2 channels, 4 x 4 x 6 voxels counted 0, 1, 2, ...
        let (c, d, h, w) = (2usize, 4usize, 4usize, 6usize);
        let data: Vec<f32> = (0..c * d * h * w).map(|i| i as f32).collect();
        let x = Act { c, d, h, w, data };
        let y = avg_pool3d(&x, [2, 2, 2]);
        assert_eq!((y.c, y.d, y.h, y.w), (2, 2, 2, 3));
        for ch in 0..c {
            for oz in 0..2 {
                for oy in 0..2 {
                    for ox in 0..3 {
                        let mut s = 0f32;
                        for kz in 0..2 {
                            for ky in 0..2 {
                                for kx in 0..2 {
                                    let i = ((ch * d + oz * 2 + kz) * h + oy * 2 + ky) * w
                                        + ox * 2
                                        + kx;
                                    s += x.data[i];
                                }
                            }
                        }
                        let got = y.data[((ch * 2 + oz) * 2 + oy) * 3 + ox];
                        assert!((got - s / 8.0).abs() < 1e-5, "{got} vs {}", s / 8.0);
                    }
                }
            }
        }
        // An odd extent drops the trailing voxels, as PyTorch does.
        let x = rand_act(1, 5, 5, 5, 3);
        let y = avg_pool3d(&x, [2, 2, 2]);
        assert_eq!((y.d, y.h, y.w), (2, 2, 2));
        // Kernel 1 is the identity.
        let y = avg_pool3d(&x, [1, 1, 1]);
        assert_eq!(y.data, x.data);
    }
}
