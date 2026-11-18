//! Volume preparation, and the resampling the two-pass pipeline needs.
//!
//! SegVol's preprocessing looks nothing like the auto-segmentation module's.
//! There is **no HU window and no resample to a target spacing**: intensities
//! are normalized from the volume's own statistics, and geometry is handled
//! by squashing whatever is left to a fixed 32x256x256. Reaching for the
//! familiar nnU-Net path here would quietly change the input distribution the
//! weights were trained on.
//!
//! In order (following the self-contained Hugging Face implementation, which
//! is what the published demo runs):
//!
//! 1. **Foreground normalization.** Threshold at the volume mean; take the
//!    0.05 and 99.95 percentiles, mean and standard deviation *of the voxels
//!    above that threshold*; clip to those percentiles and z-score with those
//!    statistics.
//! 2. **Canonical orientation.** The reference reaches `[S, A, R]` by
//!    `Orientationd(axcodes="RAS")` then a transpose swapping the first and
//!    last spatial axes. We get there directly from the DICOM direction
//!    cosines with [`Volume::canonical_axes`], which is
//!    the same target the nnU-Net engine already uses.
//! 3. **Min-max to [0, 1]**, which puts the global minimum at exactly zero.
//! 4. **Foreground crop** of everything `> 0` - that zero floor is what makes
//!    the crop remove the clipped air rim.
//!
//! What comes out is full-resolution and canonically oriented; the resize to
//! the network's input shape happens per pass in [`super::infer`].
//!
//! The reference does each step as a pass over a float copy of the volume.
//! Here the statistics come from a histogram of the `i16` voxels (65536
//! bins: the percentiles are read off it, the means are exact integer sums,
//! and only the variance - a sum of rounded squares - is a sequential pass
//! in voxel order), and since steps 1 to 3 depend on a voxel's value alone
//! they are one table over those 65536 values, applied while the crop is
//! written in canonical order. Nothing volume-sized is allocated but the
//! crop itself, and the result is that of the pass-by-pass version to the
//! bit (the test module keeps that version to hold it to it).

use rayon::prelude::*;

use crate::volume::Volume;

/// A volume ready for the network: canonically oriented `[S, A, R]`,
/// normalized to `[0, 1]`, and cropped to its foreground.
pub struct Prepared {
    pub data: Vec<f32>,
    /// Dimensions after cropping, in `[S, A, R]` order.
    pub dims: [usize; 3],
    /// Canonical axis -> volume axis.
    pub perm: [usize; 3],
    /// Canonical axis runs opposite to the volume axis.
    pub flip: [bool; 3],
    /// Volume dimensions along the mapped axes, before cropping.
    pub oriented_dims: [usize; 3],
    /// Index of the crop's origin within the oriented volume.
    pub crop_lo: [usize; 3],
}

/// Statistics of the above-mean voxels, as `ForegroundNormalization` computes
/// them.
#[derive(Clone, Copy, Debug)]
pub struct ForegroundStats {
    pub lower: f32,
    pub upper: f32,
    pub mean: f32,
    pub std: f32,
}

/// One histogram bin per `i16` value.
const BINS: usize = 1 << 16;

fn bin(x: i16) -> usize {
    (x as i32 - i16::MIN as i32) as usize
}

fn value(b: usize) -> i16 {
    (b as i32 + i16::MIN as i32) as i16
}

/// How many voxels hold each value. Counting is integer arithmetic, so the
/// parallel split cannot change the result.
fn histogram(v: &[i16]) -> Vec<u64> {
    v.par_chunks(1 << 20)
        .fold(
            || vec![0u64; BINS],
            |mut h, chunk| {
                for x in chunk {
                    h[bin(*x)] += 1;
                }
                h
            },
        )
        .reduce(
            || vec![0u64; BINS],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(&b) {
                    *x += y;
                }
                a
            },
        )
}

/// The `k`-th smallest (0-based) of the values counted in `h[from..]`.
fn order_statistic(h: &[u64], from: usize, k: u64) -> f32 {
    let mut seen = 0u64;
    for (b, c) in h.iter().enumerate().skip(from) {
        seen += c;
        if seen > k {
            return value(b) as f32;
        }
    }
    0.0
}

/// numpy's `percentile` with linear interpolation over the `n` values
/// counted in `h[from..]`. The two order statistics it interpolates between
/// are read off the histogram rather than selected out of a copy of the
/// (up to 10⁸) values, and interpolated in `f32` exactly as before.
fn percentile(h: &[u64], from: usize, n: u64, q: f64) -> f32 {
    if n == 0 {
        return 0.0;
    }
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as u64;
    let hi = pos.ceil() as u64;
    let at_lo = order_statistic(h, from, lo);
    if lo == hi {
        return at_lo;
    }
    let at_hi = order_statistic(h, from, lo + 1);
    let frac = (pos - lo as f64) as f32;
    at_lo * (1.0 - frac) + at_hi * frac
}

/// The statistics, and the histogram they were read from.
fn stats_and_histogram(v: &[i16]) -> (ForegroundStats, Vec<u64>) {
    let h = histogram(v);
    // The two means are sums of integers, kept as integers. That is not a
    // way around the determinism rule: a sequential `f64` sum over the
    // voxels is exact too - every partial sum is an integer below
    // n * 2^15, under 2^53 for any volume short of 2^38 voxels - so both
    // give the same bits, whatever the order.
    let count_and_sum = |from: usize| -> (u64, i64) {
        h.iter()
            .enumerate()
            .skip(from)
            .fold((0u64, 0i64), |(n, s), (b, c)| {
                (n + c, s + *c as i64 * value(b) as i64)
            })
    };
    let (_, total) = count_and_sum(0);
    let mean_all = total as f64 / v.len().max(1) as f64;
    // The bins are in value order, so the voxels above the mean are those
    // of every bin from the first one above it. None above it - a constant
    // volume - falls back to every voxel.
    let mut from = (0..BINS)
        .find(|b| value(*b) as f64 > mean_all)
        .unwrap_or(BINS);
    let (mut n, mut s) = count_and_sum(from);
    if n == 0 {
        from = 0;
        (n, s) = count_and_sum(0);
    }
    let nf = n as f64;
    let m = s as f64 / nf;
    // numpy's std: population, ddof = 0. This one is a sum of rounded
    // squares: it stays sequential, in voxel order, so its last bits
    // reproduce.
    let var = v
        .iter()
        .filter(|x| bin(**x) >= from)
        .map(|x| (*x as f64 - m) * (*x as f64 - m))
        .sum::<f64>()
        / nf;
    let st = ForegroundStats {
        lower: percentile(&h, from, n, 0.05),
        upper: percentile(&h, from, n, 99.95),
        mean: m as f32,
        std: var.sqrt() as f32,
    };
    (st, h)
}

/// Threshold at the mean, then summarize what is above it.
pub fn foreground_stats(v: &[i16]) -> ForegroundStats {
    stats_and_histogram(v).0
}

/// Prepare a DICOM volume for the network.
///
/// Everything after the statistics depends on a voxel's value alone, so the
/// whole normalization (clip, z-score, min-max) is one table over the 65536
/// values a voxel can hold, and the voxels go from `i16` straight into the
/// cropped, oriented image: no `f32` copy of the volume and no full-size
/// oriented image, only the crop that is handed back.
pub fn prepare(vol: &Volume) -> Prepared {
    let (perm, flip) = vol.canonical_axes();
    let dims = [vol.dims[0], vol.dims[1], vol.dims[2]];
    let oriented_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];

    // 1. foreground normalization, computed on the raw voxels
    let (st, h) = stats_and_histogram(&vol.data);
    let inv = 1.0 / st.std.max(1e-8);
    let z = |x: i16| ((x as f32).clamp(st.lower, st.upper) - st.mean) * inv;

    // 3. min-max to [0, 1]; the minimum becomes exactly zero. `z` never
    // decreases with the value (a clamp, a rounded subtraction and a
    // rounded product with a positive factor are all monotone), so its
    // extremes over the image are its values at the smallest and the
    // largest voxel - the very numbers a pass over the z-scored image finds.
    let present = || {
        h.iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(b, _)| b)
    };
    let (lo, hi) = match (present().next(), present().next_back()) {
        (Some(a), Some(b)) => (z(value(a)), z(value(b))),
        _ => (f32::INFINITY, f32::NEG_INFINITY),
    };
    let span = (hi - lo).max(1e-8);
    let mut table = vec![0f32; BINS];
    for (b, t) in table.iter_mut().enumerate() {
        if h[b] > 0 {
            *t = (z(value(b)) - lo) / span;
        }
    }
    let norm = |x: i16| table[bin(x)];

    // 4. crop to everything strictly above that zero floor. Which voxels
    // those are does not depend on the orientation, so the box is found in
    // the volume's own order and mapped onto the canonical axes after.
    let (crop_lo, crop_hi) = foreground_box(vol, &norm, perm, flip, oriented_dims);
    let dims_c = [
        crop_hi[0] - crop_lo[0],
        crop_hi[1] - crop_lo[1],
        crop_hi[2] - crop_lo[2],
    ];

    // 2. reorient, writing the crop only
    let [s0, s1, s2] = dims_c;
    let (nx, ny) = (vol.dims[0], vol.dims[1]);
    let mut data = vec![0f32; s0 * s1 * s2];
    data.par_chunks_mut((s1 * s2).max(1))
        .enumerate()
        .for_each(|(c0, slab)| {
            // crop index -> oriented index -> volume index, along one axis
            let along = |a: usize, c: usize| {
                let o = crop_lo[a] + c;
                if flip[a] {
                    oriented_dims[a] - 1 - o
                } else {
                    o
                }
            };
            let mut idx = [0usize; 3];
            idx[perm[0]] = along(0, c0);
            for c1 in 0..s1 {
                idx[perm[1]] = along(1, c1);
                for c2 in 0..s2 {
                    idx[perm[2]] = along(2, c2);
                    let src = idx[2] * nx * ny + idx[1] * nx + idx[0];
                    slab[c1 * s2 + c2] = norm(vol.data[src]);
                }
            }
        });
    Prepared {
        data,
        dims: dims_c,
        perm,
        flip,
        oriented_dims,
        crop_lo,
    }
}

/// Inclusive-exclusive box, on the canonical axes, of the voxels whose
/// normalized value is `> 0`. An image with none yields the whole extent
/// rather than an empty crop. Per-slice boxes are combined by integer
/// minimum and maximum, so the parallel split cannot change the result.
fn foreground_box(
    vol: &Volume,
    norm: &(impl Fn(i16) -> f32 + Sync),
    perm: [usize; 3],
    flip: [bool; 3],
    oriented_dims: [usize; 3],
) -> ([usize; 3], [usize; 3]) {
    let (nx, ny) = (vol.dims[0], vol.dims[1]);
    let found = vol
        .data
        .par_chunks((nx * ny).max(1))
        .enumerate()
        .filter_map(|(k, slice)| {
            let mut lo = [usize::MAX, usize::MAX, k];
            let mut hi = [0, 0, k + 1];
            for (j, row) in slice.chunks(nx.max(1)).enumerate() {
                let Some(first) = row.iter().position(|x| norm(*x) > 0.0) else {
                    continue;
                };
                let last = row.iter().rposition(|x| norm(*x) > 0.0).unwrap_or(first);
                lo[0] = lo[0].min(first);
                hi[0] = hi[0].max(last + 1);
                lo[1] = lo[1].min(j);
                hi[1] = hi[1].max(j + 1);
            }
            (lo[0] != usize::MAX).then_some((lo, hi))
        })
        .reduce_with(|(alo, ahi), (blo, bhi)| {
            (
                [alo[0].min(blo[0]), alo[1].min(blo[1]), alo[2].min(blo[2])],
                [ahi[0].max(bhi[0]), ahi[1].max(bhi[1]), ahi[2].max(bhi[2])],
            )
        });
    let Some((vlo, vhi)) = found else {
        return ([0, 0, 0], oriented_dims);
    };
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for a in 0..3 {
        let p = perm[a];
        (lo[a], hi[a]) = if flip[a] {
            (oriented_dims[a] - vhi[p], oriented_dims[a] - vlo[p])
        } else {
            (vlo[p], vhi[p])
        };
    }
    (lo, hi)
}

/// Copy a sub-box out of a volume.
pub fn crop(v: &[f32], dims: [usize; 3], lo: [usize; 3], size: [usize; 3]) -> Vec<f32> {
    let [_, d1, d2] = dims;
    let [s0, s1, s2] = size;
    let mut out = vec![0f32; s0 * s1 * s2];
    out.par_chunks_mut(s1 * s2)
        .enumerate()
        .for_each(|(i0, slab)| {
            for i1 in 0..s1 {
                let src = ((lo[0] + i0) * d1 + lo[1] + i1) * d2 + lo[2];
                slab[i1 * s2..i1 * s2 + s2].copy_from_slice(&v[src..src + s2]);
            }
        });
    out
}

/// Nearest-neighbour resize using PyTorch's `nearest-exact` rule:
/// output index `i` samples input index `floor((i + 0.5) * in / out)`.
///
/// This is not the same as plain `nearest`, which uses `floor(i * in / out)`
/// and is biased toward the origin. MONAI resizes the volume with
/// `nearest-exact`, so the network never sees an interpolated voxel.
pub fn resize_nearest_exact(v: &[f32], src: [usize; 3], dst: [usize; 3]) -> Vec<f32> {
    let idx = |i: usize, s: usize, d: usize| -> usize {
        let j = ((i as f64 + 0.5) * s as f64 / d as f64).floor() as isize;
        j.clamp(0, s as isize - 1) as usize
    };
    let map: Vec<Vec<usize>> = (0..3)
        .map(|a| (0..dst[a]).map(|i| idx(i, src[a], dst[a])).collect())
        .collect();
    let mut out = vec![0f32; dst[0] * dst[1] * dst[2]];
    out.par_chunks_mut(dst[1] * dst[2])
        .enumerate()
        .for_each(|(i0, slab)| {
            let s0 = map[0][i0];
            for i1 in 0..dst[1] {
                let base = (s0 * src[1] + map[1][i1]) * src[2];
                for i2 in 0..dst[2] {
                    slab[i1 * dst[2] + i2] = v[base + map[2][i2]];
                }
            }
        });
    out
}

/// Trilinear resize with `align_corners=false`, PyTorch's default for
/// `F.interpolate(mode='trilinear')` - used on the decoder's logits.
pub fn resize_trilinear(v: &[f32], src: [usize; 3], dst: [usize; 3]) -> Vec<f32> {
    let coord = |i: usize, s: usize, d: usize| -> f32 {
        // align_corners=false: (i + 0.5) * scale - 0.5
        (((i as f64 + 0.5) * s as f64 / d as f64) - 0.5).max(0.0) as f32
    };
    let mut out = vec![0f32; dst[0] * dst[1] * dst[2]];
    let c1: Vec<f32> = (0..dst[1]).map(|i| coord(i, src[1], dst[1])).collect();
    let c2: Vec<f32> = (0..dst[2]).map(|i| coord(i, src[2], dst[2])).collect();
    out.par_chunks_mut(dst[1] * dst[2])
        .enumerate()
        .for_each(|(i0, slab)| {
            let z = coord(i0, src[0], dst[0]);
            let (z0, fz) = (z.floor() as usize, z - z.floor());
            let z1 = (z0 + 1).min(src[0] - 1);
            for i1 in 0..dst[1] {
                let y = c1[i1];
                let (y0, fy) = (y.floor() as usize, y - y.floor());
                let y1 = (y0 + 1).min(src[1] - 1);
                for i2 in 0..dst[2] {
                    let x = c2[i2];
                    let (x0, fx) = (x.floor() as usize, x - x.floor());
                    let x1 = (x0 + 1).min(src[2] - 1);
                    let at = |a: usize, b: usize, c: usize| v[(a * src[1] + b) * src[2] + c];
                    let c00 = at(z0, y0, x0) * (1.0 - fx) + at(z0, y0, x1) * fx;
                    let c01 = at(z0, y1, x0) * (1.0 - fx) + at(z0, y1, x1) * fx;
                    let c10 = at(z1, y0, x0) * (1.0 - fx) + at(z1, y0, x1) * fx;
                    let c11 = at(z1, y1, x0) * (1.0 - fx) + at(z1, y1, x1) * fx;
                    let c0 = c00 * (1.0 - fy) + c01 * fy;
                    let c1v = c10 * (1.0 - fy) + c11 * fy;
                    slab[i1 * dst[2] + i2] = c0 * (1.0 - fz) + c1v * fz;
                }
            }
        });
    out
}

impl Prepared {
    /// The image's size in memory, what keeping it between prompts costs.
    pub fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Map a mask defined on the prepared (cropped, oriented) grid back onto
    /// the original volume's index order, `k*nx*ny + j*nx + i`.
    pub fn mask_to_volume_grid(&self, mask: &[u8], vol: &Volume) -> Vec<u8> {
        let (nx, ny, nz) = (vol.dims[0], vol.dims[1], vol.dims[2]);
        // volume axis -> canonical axis
        let mut canon_of = [0usize; 3];
        for (a, p) in self.perm.iter().enumerate() {
            canon_of[*p] = a;
        }
        let mut out = vec![0u8; nx * ny * nz];
        out.par_chunks_mut(nx * ny)
            .enumerate()
            .for_each(|(k, slab)| {
                for j in 0..ny {
                    for (i, dst) in slab[j * nx..(j + 1) * nx].iter_mut().enumerate() {
                        let vidx = [i, j, k];
                        let mut c = [0usize; 3];
                        let mut inside = true;
                        for a in 0..3 {
                            let v = vidx[self.perm[a]];
                            let oriented = if self.flip[a] {
                                self.oriented_dims[a] - 1 - v
                            } else {
                                v
                            };
                            if oriented < self.crop_lo[a]
                                || oriented >= self.crop_lo[a] + self.dims[a]
                            {
                                inside = false;
                                break;
                            }
                            c[a] = oriented - self.crop_lo[a];
                        }
                        if inside {
                            *dst = mask[(c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2]];
                        }
                    }
                }
            });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    // ------------------------------------------------- the pass-by-pass path --
    // What `prepare` was before it worked from a histogram and a table: each
    // step a pass over a float copy of the volume, as the reference does it.
    // Kept here, unchanged, as what the fast path must match to the bit.

    /// numpy's `percentile` with linear interpolation, by selection.
    fn percentile_by_selection(values: &mut [f32], q: f64) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        let pos = q / 100.0 * (values.len() - 1) as f64;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let (_, at_lo, above) = values.select_nth_unstable_by(lo, f32::total_cmp);
        let at_lo = *at_lo;
        if lo == hi {
            return at_lo;
        }
        let at_hi = above
            .iter()
            .copied()
            .min_by(f32::total_cmp)
            .unwrap_or(at_lo);
        let frac = (pos - lo as f64) as f32;
        at_lo * (1.0 - frac) + at_hi * frac
    }

    fn foreground_stats_by_passes(v: &[f32]) -> ForegroundStats {
        let mean_all = v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64;
        let mut fg: Vec<f32> = v
            .par_iter()
            .copied()
            .filter(|x| *x as f64 > mean_all)
            .collect();
        if fg.is_empty() {
            fg = v.to_vec();
        }
        let n = fg.len() as f64;
        let m = fg.iter().map(|x| *x as f64).sum::<f64>() / n;
        let var = fg
            .iter()
            .map(|x| (*x as f64 - m) * (*x as f64 - m))
            .sum::<f64>()
            / n;
        ForegroundStats {
            lower: percentile_by_selection(&mut fg, 0.05),
            upper: percentile_by_selection(&mut fg, 99.95),
            mean: m as f32,
            std: var.sqrt() as f32,
        }
    }

    fn foreground_bbox(v: &[f32], dims: [usize; 3]) -> ([usize; 3], [usize; 3]) {
        let [d0, d1, d2] = dims;
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        let mut any = false;
        for i0 in 0..d0 {
            for i1 in 0..d1 {
                let base = (i0 * d1 + i1) * d2;
                for i2 in 0..d2 {
                    if v[base + i2] > 0.0 {
                        any = true;
                        for (a, i) in [i0, i1, i2].into_iter().enumerate() {
                            lo[a] = lo[a].min(i);
                            hi[a] = hi[a].max(i + 1);
                        }
                    }
                }
            }
        }
        if !any {
            ([0, 0, 0], dims)
        } else {
            (lo, hi)
        }
    }

    /// The pass-by-pass preparation, and how many bytes its volume-sized
    /// buffers came to.
    fn prepare_by_passes(vol: &Volume) -> (Prepared, usize) {
        let (perm, flip) = vol.canonical_axes();
        let dims = [vol.dims[0], vol.dims[1], vol.dims[2]];
        let oriented_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];
        let raw: Vec<f32> = vol.data.iter().map(|v| *v as f32).collect();
        let st = foreground_stats_by_passes(&raw);
        let inv = 1.0 / st.std.max(1e-8);
        let [d0, d1, d2] = oriented_dims;
        let (nx, ny) = (vol.dims[0], vol.dims[1]);
        let mut oriented = vec![0f32; d0 * d1 * d2];
        oriented
            .par_chunks_mut(d1 * d2)
            .enumerate()
            .for_each(|(a0, slab)| {
                let mut idx = [0usize; 3];
                idx[perm[0]] = if flip[0] {
                    oriented_dims[0] - 1 - a0
                } else {
                    a0
                };
                for a1 in 0..d1 {
                    idx[perm[1]] = if flip[1] {
                        oriented_dims[1] - 1 - a1
                    } else {
                        a1
                    };
                    for a2 in 0..d2 {
                        idx[perm[2]] = if flip[2] {
                            oriented_dims[2] - 1 - a2
                        } else {
                            a2
                        };
                        let src = idx[2] * nx * ny + idx[1] * nx + idx[0];
                        let v = raw[src].clamp(st.lower, st.upper);
                        slab[a1 * d2 + a2] = (v - st.mean) * inv;
                    }
                }
            });
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &oriented {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let span = (hi - lo).max(1e-8);
        oriented.par_iter_mut().for_each(|v| *v = (*v - lo) / span);
        let (crop_lo, crop_hi) = foreground_bbox(&oriented, oriented_dims);
        let dims_c = [
            crop_hi[0] - crop_lo[0],
            crop_hi[1] - crop_lo[1],
            crop_hi[2] - crop_lo[2],
        ];
        let data = crop(&oriented, oriented_dims, crop_lo, dims_c);
        // the f32 copy, the above-mean values, the oriented image, the crop
        let mean_all = raw.iter().map(|x| *x as f64).sum::<f64>() / raw.len().max(1) as f64;
        let above = raw.iter().filter(|x| **x as f64 > mean_all).count();
        let fg_len = if above == 0 { raw.len() } else { above };
        let bytes = 4 * (raw.len() + fg_len + oriented.len() + data.len());
        (
            Prepared {
                data,
                dims: dims_c,
                perm,
                flip,
                oriented_dims,
                crop_lo,
            },
            bytes,
        )
    }

    /// Same image, same geometry, to the bit.
    fn assert_same(a: &Prepared, b: &Prepared, what: &str) {
        assert_eq!(a.dims, b.dims, "{what}: dims");
        assert_eq!(a.perm, b.perm, "{what}: perm");
        assert_eq!(a.flip, b.flip, "{what}: flip");
        assert_eq!(a.oriented_dims, b.oriented_dims, "{what}: oriented dims");
        assert_eq!(a.crop_lo, b.crop_lo, "{what}: crop");
        assert_eq!(a.data.len(), b.data.len(), "{what}: length");
        let first = a
            .data
            .iter()
            .zip(&b.data)
            .position(|(x, y)| x.to_bits() != y.to_bits());
        assert!(
            first.is_none(),
            "{what}: first differing value at {first:?}"
        );
    }

    /// A body-like phantom: an air rim, an ellipsoid of soft tissue with a
    /// bright core and noise from a fixed generator, so the statistics have
    /// a real tail, ties, negative values and a foreground crop to find.
    fn phantom(vol: &mut Volume, seed: u64) {
        let [nx, ny, nz] = vol.dims;
        let mut s = seed;
        let mut noise = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % 121) as i16 - 60
        };
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let d = |c: usize, n: usize| (c as f64 + 0.5) / n as f64 * 2.0 - 1.0;
                    let r2 = (d(i, nx) / 0.8).powi(2)
                        + (d(j, ny) / 0.6).powi(2)
                        + (d(k, nz) / 0.9).powi(2);
                    let v = if r2 > 1.0 {
                        -1000
                    } else if r2 < 0.1 {
                        1200 + noise()
                    } else {
                        40 + noise()
                    };
                    vol.data[k * nx * ny + j * nx + i] = v;
                }
            }
        }
    }

    fn axial(dims: [usize; 3]) -> Volume {
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [1.0, 1.0, 1.0],
            origin: Vec3::ZERO,
            row_dir: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            col_dir: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            normal: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        }
    }

    #[test]
    fn percentiles_interpolate_the_way_numpy_does() {
        // Shuffled on purpose: selection must not depend on the order.
        let raw: [i16; 11] = [7, 2, 9, 0, 5, 3, 10, 1, 8, 4, 6];
        let v: Vec<f32> = raw.iter().map(|i| *i as f32).collect();
        let h = histogram(&raw);
        let p = |q: f64| percentile(&h, 0, raw.len() as u64, q);
        assert_eq!(p(0.0), 0.0);
        assert_eq!(p(100.0), 10.0);
        assert_eq!(p(50.0), 5.0);
        // 25th percentile of 0..10 is 2.5 under linear interpolation
        assert!((p(25.0) - 2.5).abs() < 1e-6);
        assert!(percentile(&h, 0, 0, 50.0) == 0.0);
        // and the histogram reads what selection over the values finds,
        // to the bit, between order statistics as much as on them
        for q in [0.0, 0.05, 12.5, 25.0, 33.3, 50.0, 99.95, 100.0] {
            let by_selection = percentile_by_selection(&mut v.clone(), q);
            assert_eq!(p(q).to_bits(), by_selection.to_bits(), "q = {q}");
        }
        assert!(percentile_by_selection(&mut [], 50.0) == 0.0);
    }

    #[test]
    fn foreground_stats_use_only_above_mean_voxels() {
        // mean is 1.0; only the 3s are above it
        let v = vec![0i16, 0, 1, 3, 3, 3, 0, 0, 0, 0];
        let st = foreground_stats(&v);
        assert!((st.mean - 3.0).abs() < 1e-6, "{st:?}");
        assert!(st.std.abs() < 1e-6);
        assert!((st.lower - 3.0).abs() < 1e-6 && (st.upper - 3.0).abs() < 1e-6);
    }

    #[test]
    fn nearest_exact_is_not_plain_nearest() {
        // Upsampling 2 -> 4: nearest-exact gives 0,0,1,1; plain nearest gives
        // 0,0,1,1 too, but downsampling 4 -> 2 separates them: nearest-exact
        // samples 1 and 3, plain nearest samples 0 and 2.
        let v = vec![10.0f32, 20.0, 30.0, 40.0];
        let out = resize_nearest_exact(&v, [4, 1, 1], [2, 1, 1]);
        assert_eq!(out, vec![20.0, 40.0], "must sample the later of each pair");
        let up = resize_nearest_exact(&v, [4, 1, 1], [8, 1, 1]);
        assert_eq!(up, vec![10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0]);
        // identity
        assert_eq!(resize_nearest_exact(&v, [4, 1, 1], [4, 1, 1]), v);
    }

    #[test]
    fn trilinear_is_exact_at_identity_and_averages_when_halving() {
        let v: Vec<f32> = (0..8).map(|i| i as f32).collect();
        assert_eq!(resize_trilinear(&v, [2, 2, 2], [2, 2, 2]), v);
        // halving a 2x1x1 ramp lands on the midpoint
        let r = resize_trilinear(&[0.0, 4.0], [2, 1, 1], [1, 1, 1]);
        assert!((r[0] - 2.0).abs() < 1e-5, "{r:?}");
        // upsampling stays within the input range
        let up = resize_trilinear(&v, [2, 2, 2], [4, 4, 4]);
        assert!(up.iter().all(|x| (0.0..=7.0).contains(x)));
    }

    #[test]
    fn the_foreground_crop_removes_the_zero_rim() {
        // the pass-by-pass box, which the fast path is held to below
        let dims = [4, 4, 4];
        let mut v = vec![0f32; 64];
        let at = |i0: usize, i1: usize, i2: usize| (i0 * 4 + i1) * 4 + i2;
        v[at(1, 1, 2)] = 0.5;
        v[at(2, 2, 3)] = 0.7;
        let (lo, hi) = foreground_bbox(&v, dims);
        assert_eq!(lo, [1, 1, 2]);
        assert_eq!(hi, [3, 3, 4]);
        let c = crop(&v, dims, lo, [2, 2, 2]);
        assert_eq!(c.len(), 8);
        assert_eq!(c[0], 0.5);
        assert_eq!(c[7], 0.7);
        // an all-zero volume keeps its whole extent rather than vanishing
        let (lo, hi) = foreground_bbox(&vec![0f32; 64], dims);
        assert_eq!((lo, hi), ([0, 0, 0], dims));
    }

    #[test]
    fn preparation_normalizes_to_the_unit_range_and_orients_canonically() {
        let mut vol = axial([8, 6, 4]);
        for (i, v) in vol.data.iter_mut().enumerate() {
            *v = (i % 50) as i16;
        }
        let p = prepare(&vol);
        // standard axial LPS: S = +z (no flip), A = -y (flip), R = -x (flip)
        assert_eq!(p.perm, [2, 1, 0]);
        assert_eq!(p.flip, [false, true, true]);
        assert_eq!(p.oriented_dims, [4, 6, 8]);
        assert!(p.data.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(p.data.contains(&0.0), "the minimum must be exactly zero");
    }

    #[test]
    fn a_mask_round_trips_back_onto_the_original_grid() {
        // Mark a block in the volume, prepare, mark the same block in the
        // prepared grid, and check it lands back where it started.
        let mut vol = axial([10, 8, 6]);
        for k in 1..4 {
            for j in 2..6 {
                for i in 3..8 {
                    vol.data[k * 80 + j * 10 + i] = 1000;
                }
            }
        }
        let p = prepare(&vol);
        // everything inside the crop, marked
        let mask = vec![1u8; p.dims[0] * p.dims[1] * p.dims[2]];
        let back = p.mask_to_volume_grid(&mask, &vol);
        // every voxel of the original block must be covered
        for k in 1..4 {
            for j in 2..6 {
                for i in 3..8 {
                    assert_eq!(back[k * 80 + j * 10 + i], 1, "at {i},{j},{k}");
                }
            }
        }
        assert!(back.contains(&0) || p.dims == p.oriented_dims);
    }

    fn v3(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// The fast preparation is the pass-by-pass one to the bit: statistics,
    /// prepared image, crop and the masks mapped back - over orientations
    /// that permute and flip every axis, a constant volume (the fallback to
    /// every voxel), an all-zero one, a two-valued one, the ends of the
    /// `i16` range, and noise wide enough that the percentiles fall between
    /// distinct values.
    #[test]
    fn the_histogram_path_matches_the_pass_by_pass_one_to_the_bit() {
        let orientations = [
            // standard axial: S = +z, A = -y, R = -x
            (v3(1.0, 0.0, 0.0), v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)),
            // sagittal acquisition: rows along +y, columns down, slices along +x
            (v3(0.0, 1.0, 0.0), v3(0.0, 0.0, -1.0), v3(1.0, 0.0, 0.0)),
            // coronal, oblique, two axes reversed
            (
                v3(-0.99, 0.141, 0.0),
                v3(0.0, 0.0, -1.0),
                v3(0.141, 0.99, 0.0),
            ),
        ];
        let mut cases: Vec<(String, Volume)> = Vec::new();
        for (o, (r, c, n)) in orientations.into_iter().enumerate() {
            let mut base = axial([37, 29, 23]);
            (base.row_dir, base.col_dir, base.normal) = (r, c, n);
            let mut vol = base.clone();
            phantom(&mut vol, 7 + o as u64);
            cases.push((format!("phantom, orientation {o}"), vol));
            let mut vol = base.clone();
            for (i, v) in vol.data.iter_mut().enumerate() {
                *v = (i % 50) as i16;
            }
            cases.push((format!("ramp, orientation {o}"), vol));
            let mut vol = base.clone();
            let mut s = 0x9e3779b97f4a7c15u64 + o as u64;
            for v in vol.data.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *v = (s >> 48) as u16 as i16;
            }
            cases.push((format!("full-range noise, orientation {o}"), vol));
        }
        let mut constant = axial([12, 10, 6]);
        constant.data.fill(100);
        cases.push(("constant".into(), constant));
        cases.push(("all zero".into(), axial([12, 10, 6])));
        let mut block = axial([10, 8, 6]);
        for k in 1..4 {
            for j in 2..6 {
                for i in 3..8 {
                    block.data[k * 80 + j * 10 + i] = 1000;
                }
            }
        }
        cases.push(("block".into(), block));
        let mut ends = axial([9, 7, 5]);
        for (i, v) in ends.data.iter_mut().enumerate() {
            *v = match i % 3 {
                0 => i16::MIN,
                1 => i16::MAX,
                _ => 0,
            };
        }
        cases.push(("ends of the range".into(), ends));

        for (what, vol) in &cases {
            let raw: Vec<f32> = vol.data.iter().map(|v| *v as f32).collect();
            let (a, b) = (
                foreground_stats(&vol.data),
                foreground_stats_by_passes(&raw),
            );
            for (x, y, name) in [
                (a.lower, b.lower, "lower"),
                (a.upper, b.upper, "upper"),
                (a.mean, b.mean, "mean"),
                (a.std, b.std, "std"),
            ] {
                assert_eq!(x.to_bits(), y.to_bits(), "{what}: {name} {x} vs {y}");
            }
            let fast = prepare(vol);
            let (slow, _) = prepare_by_passes(vol);
            assert_same(&fast, &slow, what);
            // what a run hands back: a mask on the prepared grid, mapped
            // onto the volume's own
            let mask: Vec<u8> = fast.data.iter().map(|x| u8::from(*x > 0.5)).collect();
            assert_eq!(
                fast.mask_to_volume_grid(&mask, vol),
                slow.mask_to_volume_grid(&mask, vol),
                "{what}: mask"
            );
        }
    }

    /// Cold preparation of a CT-sized volume, old against new, and the
    /// volume-sized buffers each allocates. A measurement, not a check:
    ///
    /// ```text
    /// cargo test -p rds-engines --lib segvol::preprocess -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn preparation_time_and_memory_on_a_ct_sized_volume() {
        let mut vol = axial([512, 512, 300]);
        (vol.spacing, vol.normal) = ([0.98, 0.98, 2.0], v3(0.0, 0.0, 1.0));
        phantom(&mut vol, 1);
        let mib = |b: usize| b as f64 / (1 << 20) as f64;
        let t = std::time::Instant::now();
        let (slow, slow_bytes) = prepare_by_passes(&vol);
        let slow_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let fast = prepare(&vol);
        let fast_s = t.elapsed().as_secs_f64();
        assert_same(&fast, &slow, "CT-sized phantom");
        let threads = rayon::current_num_threads();
        println!(
            "pass by pass: {slow_s:.3} s, {:.0} MiB of volume-sized buffers\n\
             histogram:    {fast_s:.3} s, {:.0} MiB crop + {:.2} MiB of histograms and table \
             (at most {threads} histograms alive at once)\n\
             crop {:?} of {:?}",
            mib(slow_bytes),
            mib(fast.bytes()),
            mib(BINS * 4 + BINS * 8 * threads),
            fast.dims,
            fast.oriented_dims,
        );
    }
}
