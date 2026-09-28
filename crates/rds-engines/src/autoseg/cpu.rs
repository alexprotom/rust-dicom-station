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
                                        ((-dx) as usize, ow)
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

/// InstanceNorm3d (affine, eps 1e-5, biased variance) fused with
/// LeakyReLU(0.01) - the pairing every nnU-Net conv block uses.
pub fn instance_norm_lrelu(x: &mut Act, gamma: &[f32], beta: &[f32]) {
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
        for v in ch.iter_mut() {
            let y = *v * scale + shift;
            *v = if y >= 0.0 { y } else { 0.01 * y };
        }
    });
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
                                        ((-dx) as usize, ow)
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
}
