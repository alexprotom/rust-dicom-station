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
//!    cosines with [`crate::autoseg::preprocess::canonical_axes`], which is
//!    the same target the nnU-Net engine already uses.
//! 3. **Min-max to [0, 1]**, which puts the global minimum at exactly zero.
//! 4. **Foreground crop** of everything `> 0` — that zero floor is what makes
//!    the crop remove the clipped air rim.
//!
//! What comes out is full-resolution and canonically oriented; the resize to
//! the network's input shape happens per pass in [`super::infer`].

use rayon::prelude::*;

use crate::autoseg::preprocess::canonical_axes;
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

/// numpy's `percentile` with linear interpolation, on already-sorted data.
fn percentile_sorted(sorted: &[f32], q: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = (pos - lo as f64) as f32;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Threshold at the mean, then summarize what is above it.
pub fn foreground_stats(v: &[f32]) -> ForegroundStats {
    let mean_all = v.iter().map(|x| *x as f64).sum::<f64>() / v.len().max(1) as f64;
    let mut fg: Vec<f32> = v.iter().copied().filter(|x| *x as f64 > mean_all).collect();
    if fg.is_empty() {
        fg = v.to_vec();
    }
    fg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = fg.len() as f64;
    let m = fg.iter().map(|x| *x as f64).sum::<f64>() / n;
    // numpy's std: population, ddof = 0
    let var = fg
        .iter()
        .map(|x| (*x as f64 - m) * (*x as f64 - m))
        .sum::<f64>()
        / n;
    ForegroundStats {
        lower: percentile_sorted(&fg, 0.05),
        upper: percentile_sorted(&fg, 99.95),
        mean: m as f32,
        std: var.sqrt() as f32,
    }
}

/// Prepare a DICOM volume for the network.
pub fn prepare(vol: &Volume) -> Prepared {
    let (perm, flip) = canonical_axes(vol);
    let dims = [vol.dims[0], vol.dims[1], vol.dims[2]];
    let oriented_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];

    // 1. foreground normalization, computed on the raw voxels
    let raw: Vec<f32> = vol.data.iter().map(|v| *v as f32).collect();
    let st = foreground_stats(&raw);
    let inv = 1.0 / st.std.max(1e-8);

    // 2. reorient while applying the normalization
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

    // 3. min-max to [0, 1]; the minimum becomes exactly zero
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for v in &oriented {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    let span = (hi - lo).max(1e-8);
    oriented.par_iter_mut().for_each(|v| *v = (*v - lo) / span);

    // 4. crop to everything strictly above that zero floor
    let (crop_lo, crop_hi) = foreground_bbox(&oriented, oriented_dims);
    let dims_c = [
        crop_hi[0] - crop_lo[0],
        crop_hi[1] - crop_lo[1],
        crop_hi[2] - crop_lo[2],
    ];
    let data = crop(&oriented, oriented_dims, crop_lo, dims_c);
    Prepared {
        data,
        dims: dims_c,
        perm,
        flip,
        oriented_dims,
        crop_lo,
    }
}

/// Inclusive-exclusive bounding box of everything `> 0`. An all-zero volume
/// yields the whole extent rather than an empty crop.
pub fn foreground_bbox(v: &[f32], dims: [usize; 3]) -> ([usize; 3], [usize; 3]) {
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
/// `F.interpolate(mode='trilinear')` — used on the decoder's logits.
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
        let v: Vec<f32> = (0..=10).map(|i| i as f32).collect();
        assert_eq!(percentile_sorted(&v, 0.0), 0.0);
        assert_eq!(percentile_sorted(&v, 100.0), 10.0);
        assert_eq!(percentile_sorted(&v, 50.0), 5.0);
        // 25th percentile of 0..10 is 2.5 under linear interpolation
        assert!((percentile_sorted(&v, 25.0) - 2.5).abs() < 1e-6);
        assert!(percentile_sorted(&[], 50.0) == 0.0);
    }

    #[test]
    fn foreground_stats_use_only_above_mean_voxels() {
        // mean is 1.0; only the 3s are above it
        let v = vec![0.0f32, 0.0, 1.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0];
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
}
