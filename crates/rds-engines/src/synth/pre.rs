//! The arithmetic of the TotalSynth recipe around the network, as KonfAI
//! (2026-10) runs it for the ImpactSynth apps - each step here is one stage
//! of their `Prediction.yml`, to the rounding:
//!
//! 1. `TensorCast float16` → `Clip [-1024, 5000]` → `TensorCast int16`:
//!    [`cast_input`];
//! 2. `Resample spacing [1, 1, 3]` (linear; the grid keeps the scan's field
//!    of view, `align: extent`): [`working_dims`], [`resample_linear`];
//! 3. `Clip min .. percentile 99.5` and `Normalize [-1, 1]`: [`clip_normalize`];
//! 4. the 2.5-D stacks (`extend_slice 4`, reflected at the ends) on 512 x 512
//!    patches padded with -1 ([`tiles`], [`stack`]);
//! 5. the network's output kept in float16, `UnNormalize [-1024, 3071]`,
//!    `TensorCast int16`, and `Mask` outside the body: [`member_hu`];
//! 6. the mean over the checkpoints and test-time flips, back to int16,
//!    then resampled back onto the scan's grid ([`resample_linear`] again).
//!
//! Volumes are `[z][y][x]` in C order over `dims = [nz, ny, nx]`, the
//! array order KonfAI holds an image in; the caller hands them over in the
//! scan's LPS axis order.

use rayon::prelude::*;

use crate::nn::half::{f16_to_f32, f32_to_f16};

/// What the recipe works at, `[z, y, x]` in mm.
pub const SPACING: [f64; 3] = [3.0, 1.0, 1.0];
/// The in-plane patch the network sees.
pub const PATCH: usize = 512;
/// Slices of context on each side of a stack's centre (`extend_slice 4`).
pub const HALF_STACK: usize = 2;
/// The value the patches are padded with in plane.
pub const PAD: f32 = -1.0;
/// `Clip` before the resampling.
pub const CLIP_IN: (f32, f32) = (-1024.0, 5000.0);
/// The percentile the volume is clipped at before the normalisation.
pub const PERCENTILE: f64 = 99.5;
/// `UnNormalize`: the network's [-1, 1] onto these HU.
pub const OUT_RANGE: (f32, f32) = (-1024.0, 3071.0);
/// Value outside the body.
pub const AIR: i16 = -1024;
/// `Dilate` of the body mask, in working-grid voxels (a box).
pub const DILATE: usize = 5;

/// Round to the nearest binary16 value (ties to even), as `tensor.half()`.
#[inline]
pub fn f16(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// `float32 → int16` as torch casts it: toward zero, saturating.
#[inline]
pub fn to_i16(v: f32) -> i16 {
    v.trunc().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

/// Step 1, per voxel: `TensorCast float16`, `Clip [-1024, 5000]`,
/// `TensorCast int16`.
pub fn cast_input(v: i16) -> i16 {
    let h = f16(v as f32);
    to_i16(h.clamp(CLIP_IN.0, CLIP_IN.1))
}

/// The working grid's extent along one axis: as many voxels of `want` as
/// fit in `n` of `spacing`, rounded down (`floor(n * s / want + 1e-6)`), at
/// least one.
pub fn working_len(n: usize, spacing: f64, want: f64) -> usize {
    ((n as f64 * spacing / want + 1e-6).floor() as usize).max(1)
}

/// [`working_len`] per axis.
pub fn working_dims(dims: [usize; 3], spacing: [f64; 3]) -> [usize; 3] {
    std::array::from_fn(|a| working_len(dims[a], spacing[a], SPACING[a]))
}

/// The continuous index on a grid of `n_from` voxels of `s_from` that
/// voxel `t` of a grid of `n_to` voxels covering the same extent sits at.
/// The two grids share their outer faces, so the covering spacing is
/// `s_from * n_from / n_to` and the first centres sit half a voxel in from
/// the shared face.
pub fn source_coord(t: usize, n_to: usize, n_from: usize, s_from: f64) -> f64 {
    let cover = s_from * n_from as f64 / n_to as f64;
    // world (from the shared face's first centre, as KonfAI places the
    // grid): 0.5 * (cover - s_from) + t * cover; index: / s_from
    (0.5 * (cover - s_from) + t as f64 * cover) / s_from
}

/// One axis of an ITK linear interpolation: the base tap, the next one,
/// and the fraction - or `None` for the fraction when ITK takes the base
/// tap alone (at and past the last centre, and before the first).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tap {
    i0: usize,
    i1: usize,
    frac: Option<f64>,
}

fn taps(n_to: usize, n_from: usize, s_from: f64) -> Vec<Tap> {
    (0..n_to)
        .map(|t| {
            let c = source_coord(t, n_to, n_from, s_from);
            let base = c.floor().max(0.0);
            let i0 = (base as usize).min(n_from - 1);
            let d = c - base;
            if d > 0.0 && i0 + 1 < n_from {
                Tap {
                    i0,
                    i1: i0 + 1,
                    frac: Some(d),
                }
            } else {
                Tap {
                    i0,
                    i1: i0,
                    frac: None,
                }
            }
        })
        .collect()
}

#[inline]
fn blend(t: &Tap, lo: f64, hi: f64) -> f64 {
    match t.frac {
        Some(d) => (hi - lo) * d + lo,
        None => lo,
    }
}

/// Linear resampling of `src` (`dims_from`, spacing `s_from`) onto the grid
/// of `dims_to` voxels covering the same extent, as `sitk.Resample` (and
/// KonfAI's exact path) computes it: the taps held at the first voxel, the
/// blends `lo + (hi - lo) * d` in double precision folded along x, then y,
/// then z, the result rounded to `f32`.
pub fn resample_linear(
    src: &[f32],
    dims_from: [usize; 3],
    s_from: [f64; 3],
    dims_to: [usize; 3],
) -> Vec<f32> {
    let [fz, fy, fx] = dims_from;
    let [tz, ty, tx] = dims_to;
    let (az, ay, ax) = (
        taps(tz, fz, s_from[0]),
        taps(ty, fy, s_from[1]),
        taps(tx, fx, s_from[2]),
    );
    // along x for every source row
    let mut rx = vec![0f64; fz * fy * tx];
    rx.par_chunks_mut(tx).enumerate().for_each(|(row, out)| {
        let r = &src[row * fx..(row + 1) * fx];
        for (o, t) in out.iter_mut().zip(&ax) {
            *o = blend(t, r[t.i0] as f64, r[t.i1] as f64);
        }
    });
    // along y
    let mut ry = vec![0f64; fz * ty * tx];
    ry.par_chunks_mut(ty * tx).enumerate().for_each(|(z, out)| {
        for (y, t) in ay.iter().enumerate() {
            let lo = &rx[(z * fy + t.i0) * tx..(z * fy + t.i0 + 1) * tx];
            let hi = &rx[(z * fy + t.i1) * tx..(z * fy + t.i1 + 1) * tx];
            for x in 0..tx {
                out[y * tx + x] = blend(t, lo[x], hi[x]);
            }
        }
    });
    drop(rx);
    // along z
    let plane = ty * tx;
    let mut out = vec![0f32; tz * plane];
    out.par_chunks_mut(plane).enumerate().for_each(|(z, o)| {
        let t = &az[z];
        let lo = &ry[t.i0 * plane..(t.i0 + 1) * plane];
        let hi = &ry[t.i1 * plane..(t.i1 + 1) * plane];
        for v in 0..plane {
            o[v] = blend(t, lo[v], hi[v]) as f32;
        }
    });
    out
}

/// Nearest-neighbour resampling as ITK picks: the source voxel whose
/// continuous index rounds half up, `floor(c + 0.5)`.
pub fn resample_nearest(
    src: &[u8],
    dims_from: [usize; 3],
    s_from: [f64; 3],
    dims_to: [usize; 3],
) -> Vec<u8> {
    let pick = |n_to: usize, n_from: usize, s: f64| -> Vec<usize> {
        (0..n_to)
            .map(|t| {
                let c = source_coord(t, n_to, n_from, s);
                ((c + 0.5).floor().max(0.0) as usize).min(n_from - 1)
            })
            .collect()
    };
    let [fz, fy, fx] = dims_from;
    let [tz, ty, tx] = dims_to;
    let (pz, py, px) = (
        pick(tz, fz, s_from[0]),
        pick(ty, fy, s_from[1]),
        pick(tx, fx, s_from[2]),
    );
    let mut out = vec![0u8; tz * ty * tx];
    out.par_chunks_mut(ty * tx).enumerate().for_each(|(z, o)| {
        for (y, &sy) in py.iter().enumerate() {
            let row = &src[(pz[z] * fy + sy) * fx..(pz[z] * fy + sy + 1) * fx];
            for (x, &sx) in px.iter().enumerate() {
                o[y * tx + x] = row[sx];
            }
        }
    });
    out
}

/// `Dilate`: a box of radius `r` voxels on every axis, as three 1-D max
/// filters (KonfAI's separable max-pools).
pub fn dilate(mask: &[u8], dims: [usize; 3], r: usize) -> Vec<u8> {
    let [nz, ny, nx] = dims;
    let mut a: Vec<u8> = mask.iter().map(|&v| u8::from(v > 0)).collect();
    if r == 0 {
        return a;
    }
    // `count` lines of `len` voxels `stride` apart; line `l` starts at
    // `(l / stride) * len * stride + l % stride`.
    let pass = |a: &[u8], len: usize, stride: usize, count: usize| -> Vec<u8> {
        let mut out = vec![0u8; a.len()];
        for l in 0..count {
            let start = (l / stride) * len * stride + l % stride;
            for i in 0..len {
                let (lo, hi) = (i.saturating_sub(r), (i + r).min(len - 1));
                out[start + i * stride] = (lo..=hi).fold(0, |m, j| m | a[start + j * stride]);
            }
        }
        out
    };
    a = pass(&a, nz, ny * nx, ny * nx);
    a = pass(&a, ny, nx, nz * nx);
    pass(&a, nx, 1, nz * ny)
}

/// The value at the `q`-th percentile of `values` as `np.percentile`
/// computes it by default (linear interpolation between the order
/// statistics, numpy's `_lerp`).
pub fn percentile(values: &[i16], q: f64) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let h = (q / 100.0) * (n - 1) as f64;
    let lo = (h.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    let t = h - lo as f64;
    let mut v = values.to_vec();
    let (_, a, rest) = v.select_nth_unstable(lo);
    let a = *a as f64;
    let b = if hi == lo {
        a
    } else {
        // the smallest of what lies above the lo-th
        *rest
            .iter()
            .min()
            .expect("hi > lo means something lies above") as f64
    };
    let d = b - a;
    if t >= 0.5 {
        b - d * (1.0 - t)
    } else {
        a + d * t
    }
}

/// Step 3: `Clip min .. percentile 99.5` (a non-whole bound fills int16
/// voxels above it with the bound truncated, as torch's `masked_fill_`
/// does), then `Normalize [-1, 1]` over the clipped volume's own range, in
/// `f32` as torch computes `((x - min) * 2) / (max - min) - 1`.
pub fn clip_normalize(v: &[i16]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let p = percentile(v, PERCENTILE);
    let lo = *v.iter().min().expect("non-empty");
    let fill = to_i16(p as f32);
    let p32 = p as f32;
    let clipped: Vec<i16> = v
        .par_iter()
        .map(|&x| if (x as f32) > p32 { fill } else { x })
        .collect();
    let hi = *clipped.iter().max().expect("non-empty");
    let (mn, norm) = (lo as f32, (hi as f64 - lo as f64) as f32);
    if norm == 0.0 {
        return vec![-1.0; v.len()];
    }
    clipped
        .par_iter()
        .map(|&x| ((x as f32 - mn) * 2.0) / norm + -1.0)
        .collect()
}

/// The start of every patch along an axis of `n` voxels, as KonfAI cuts a
/// fixed-size grid with no overlap declared: as many patches as needed,
/// the remainder of the last spread evenly as overlap, the last cut short
/// at the end (and padded up to the patch for the network).
pub fn tiles(n: usize, patch: usize) -> Vec<(usize, usize)> {
    let count = n.div_ceil(patch).max(1);
    let overlap = if count > 1 {
        ((patch - n % patch) % patch) / (count - 1)
    } else {
        0
    };
    let mut out = Vec::new();
    let mut index = 0;
    loop {
        let start = (patch - overlap) * index;
        let end = start + patch;
        if end >= n {
            out.push((start, n));
            break;
        }
        out.push((start, end));
        index += 1;
    }
    out
}

/// The slice index of channel `c` of the stack centred on `z`, reflected
/// at the ends of a volume of `nz` slices (`F.pad(..., "reflect")`), and
/// folded again when the volume is too thin to reflect once.
pub fn stack_slice(z: usize, c: usize, nz: usize) -> usize {
    let mut i = z as isize + c as isize - HALF_STACK as isize;
    let n = nz as isize;
    if n == 1 {
        return 0;
    }
    loop {
        if i < 0 {
            i = -i;
        } else if i >= n {
            i = 2 * (n - 1) - i;
        } else {
            return i as usize;
        }
    }
}

/// One network input: the `channels`-slice stack centred on `z`, the
/// in-plane window `y` x `x` of the normalised volume, padded with -1 up to
/// `PATCH` x `PATCH` (after the window, as KonfAI pads).
pub fn stack(
    vol: &[f32],
    dims: [usize; 3],
    z: usize,
    channels: usize,
    y: (usize, usize),
    x: (usize, usize),
    out: &mut [f32],
) {
    let [nz, ny, nx] = dims;
    let plane = PATCH * PATCH;
    debug_assert_eq!(out.len(), channels * plane);
    out.fill(PAD);
    for c in 0..channels {
        let s = stack_slice(z, c, nz);
        for (r, yy) in (y.0..y.1).enumerate() {
            let src = &vol[(s * ny + yy) * nx + x.0..(s * ny + yy) * nx + x.1];
            out[c * plane + r * PATCH..c * plane + r * PATCH + src.len()].copy_from_slice(src);
        }
    }
}

/// Step 5 for one voxel of one member: the network's output kept in
/// float16 (KonfAI accumulates predictions at half precision), `UnNormalize`
/// computed in float16 - `((y + 1) / 2) * 4095 + (-1024)`, each operation
/// rounded - then cast to int16 toward zero.
pub fn member_hu(y: f32) -> i16 {
    let span = OUT_RANGE.1 - OUT_RANGE.0;
    let h = f16(y);
    let t = f16(h + 1.0);
    let t = f16(t / 2.0);
    let t = f16(t * span);
    let t = f16(t + OUT_RANGE.0);
    to_i16(t)
}

/// Flip a `[z][y][x]` volume along y and/or x.
pub fn flip<T: Copy + Send + Sync>(v: &[T], dims: [usize; 3], fy: bool, fx: bool) -> Vec<T> {
    let [_, ny, nx] = dims;
    let mut out = v.to_vec();
    if !fy && !fx {
        return out;
    }
    out.par_chunks_mut(ny * nx)
        .zip(v.par_chunks(ny * nx))
        .for_each(|(o, i)| {
            for y in 0..ny {
                let sy = if fy { ny - 1 - y } else { y };
                for x in 0..nx {
                    let sx = if fx { nx - 1 - x } else { x };
                    o[y * nx + x] = i[sy * nx + sx];
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_working_grid_keeps_the_extent() {
        // 56 voxels of 1.3 mm → 72 of 1.0111 mm; 12 of 2 mm → 8 of 3 mm
        assert_eq!(working_dims([12, 50, 56], [2.0, 1.1, 1.3]), [8, 55, 72]);
        // the first target centre is half a target voxel in from the face
        let c = source_coord(0, 8, 12, 2.0);
        assert!((c - (1.5 * 0.5 - 0.5)).abs() < 1e-12);
        // and the grids are each other's inverse at the faces
        let back = source_coord(11, 12, 8, 3.0);
        assert!((back - ((11.5) * 8.0 / 12.0 - 0.5)).abs() < 1e-12);
    }

    #[test]
    fn linear_resampling_holds_the_border_and_blends_inside() {
        // 1-D along x: [0, 10, 20, 30] onto 8 voxels (half the spacing)
        let src = [0f32, 10.0, 20.0, 30.0];
        let out = resample_linear(&src, [1, 1, 4], [1.0, 1.0, 2.0], [1, 1, 8]);
        // centres at -0.25 (held), 0.25, 0.75, ..., 3.25 (held)
        let want = [0.0, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30.0];
        for (g, w) in out.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{out:?}");
        }
    }

    #[test]
    fn percentiles_follow_numpy() {
        let v: Vec<i16> = vec![1, 2, 3, 4, 5, 10];
        assert!((percentile(&v, 99.5) - 9.875).abs() < 1e-12);
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 100.0), 10.0);
        // 50th of 1..=4: index 1.5 → 2.5
        assert_eq!(percentile(&[4, 1, 3, 2], 50.0), 2.5);
    }

    #[test]
    fn the_clip_truncates_a_fractional_bound() {
        let v: Vec<i16> = vec![1, 2, 3, 4, 5, 10];
        let n = clip_normalize(&v);
        // p = 9.875: 10 becomes 9, the range is 1..9
        assert_eq!(n[0], -1.0);
        assert_eq!(n[5], 1.0);
        assert!((n[2] - ((3.0 - 1.0) * 2.0 / 8.0 - 1.0)).abs() < 1e-7);
    }

    #[test]
    fn the_output_is_unnormalised_in_half_precision() {
        // tanh = 1 lands on 3072, not 3071: 4095 is not a float16
        assert_eq!(member_hu(1.0), 3072);
        assert_eq!(member_hu(-1.0), -1024);
        // soft tissue: 0.5 * 4095 = 2047.5 → float16 2048 → 1024
        assert_eq!(member_hu(0.0), 1024);
    }

    #[test]
    fn tiles_cover_the_axis_as_konfai_cuts_it() {
        assert_eq!(tiles(300, 512), vec![(0, 300)]);
        assert_eq!(tiles(512, 512), vec![(0, 512)]);
        // 600: two patches, the remainder 424 spread: overlap 424
        assert_eq!(tiles(600, 512), vec![(0, 512), (88, 600)]);
    }

    #[test]
    fn stacks_reflect_at_the_ends() {
        let s: Vec<usize> = (0..5).map(|c| stack_slice(0, c, 8)).collect();
        assert_eq!(s, [2, 1, 0, 1, 2]);
        let s: Vec<usize> = (0..5).map(|c| stack_slice(7, c, 8)).collect();
        assert_eq!(s, [5, 6, 7, 6, 5]);
        let s: Vec<usize> = (0..5).map(|c| stack_slice(0, c, 2)).collect();
        assert_eq!(s, [0, 1, 0, 1, 0]);
    }

    #[test]
    fn dilation_is_a_box() {
        let dims = [3, 5, 5];
        let mut m = vec![0u8; 75];
        m[25 + 2 * 5 + 2] = 1; // centre of the middle slice
        let d = dilate(&m, dims, 1);
        assert_eq!(d.iter().filter(|&&v| v == 1).count(), 27);
        assert_eq!(d[0], 0);
        assert_eq!(d[5 + 1], 1);
    }
}
