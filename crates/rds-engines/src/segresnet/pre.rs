//! The MONAI transforms the SegResNet and VISTA-3D pipelines are made of,
//! on a volume held in the model's own axis order.
//!
//! * `Orientation(axcodes)` is a permutation and flips of the scan's axes
//!   ([`Oriented::new`]).
//! * `Spacing(pixdim, mode="bilinear")` keeps the first voxel centre where
//!   it was and lays samples `pixdim` apart from it: output length
//!   `round((n - 1) * s_in / s_out + 1)` (numpy's half-to-even), sample `i`
//!   read at input coordinate `i * s_out / s_in`, linearly, clamped to the
//!   edge ([`resample_linear`]). Its inverse, for the labels, reads the
//!   nearest sample with PyTorch's half-to-even rounding
//!   ([`resample_nearest`]).
//! * The intensity transforms: `NormalizeIntensity(nonzero=True)`,
//!   `ScaleIntensity(minv, maxv)`, `ScaleIntensityRange(..., clip=True)`.
//! * `CropForeground` (the bounding box of positive voxels) and
//!   `KeepLargestConnectedComponent` (26-connected, per label).
//!
//! Each was checked against MONAI 1.6 on small arrays; the module's tests
//! keep the cases.

use rayon::prelude::*;

use crate::volume::{AxisOrder, Volume};

/// A scan in a model's axis order.
pub struct Oriented {
    /// Model axis → volume axis.
    pub perm: [usize; 3],
    pub flip: [bool; 3],
    /// Dimensions and spacing along the model axes.
    pub dims: [usize; 3],
    pub spacing: [f64; 3],
}

impl Oriented {
    pub fn new(vol: &Volume, order: AxisOrder) -> Oriented {
        let (perm, flip) = vol.axes_toward(order);
        Oriented {
            perm,
            flip,
            dims: std::array::from_fn(|a| vol.dims[perm[a]]),
            spacing: std::array::from_fn(|a| vol.spacing[perm[a]]),
        }
    }

    /// The `Volume::data` index of model-axis voxel `m`.
    pub fn volume_index(&self, vol_dims: [usize; 3], m: [usize; 3]) -> usize {
        let mut c = [0usize; 3];
        for a in 0..3 {
            c[self.perm[a]] = if self.flip[a] {
                self.dims[a] - 1 - m[a]
            } else {
                m[a]
            };
        }
        (c[2] * vol_dims[1] + c[1]) * vol_dims[0] + c[0]
    }

    /// The scan's values in model axis order (C-order over `dims`).
    pub fn read(&self, vol: &Volume) -> Vec<f32> {
        let [d0, d1, d2] = self.dims;
        let mut out = vec![0f32; d0 * d1 * d2];
        out.par_chunks_mut(d1 * d2)
            .enumerate()
            .for_each(|(a, slab)| {
                for b in 0..d1 {
                    for c in 0..d2 {
                        slab[b * d2 + c] = vol.data[self.volume_index(vol.dims, [a, b, c])] as f32;
                    }
                }
            });
        out
    }

    /// A fractional voxel position of the volume (`[x, y, z]` indices) in
    /// model axes.
    pub fn point_to_model(&self, v: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| {
            let x = v[self.perm[a]];
            if self.flip[a] {
                (self.dims[a] - 1) as f64 - x
            } else {
                x
            }
        })
    }

    /// A label map on the volume's own grid, in model axis order.
    pub fn read_labels(&self, labels: &[u8], vol_dims: [usize; 3]) -> Vec<u8> {
        let [d0, d1, d2] = self.dims;
        let mut out = vec![0u8; d0 * d1 * d2];
        out.par_chunks_mut(d1 * d2)
            .enumerate()
            .for_each(|(a, slab)| {
                for b in 0..d1 {
                    for c in 0..d2 {
                        slab[b * d2 + c] = labels[self.volume_index(vol_dims, [a, b, c])];
                    }
                }
            });
        out
    }

    /// Labels in model axis order back onto the volume's own grid.
    pub fn write_back(&self, labels: &[u8], vol: &Volume) -> Vec<u8> {
        let [d0, d1, d2] = self.dims;
        let mut out = vec![0u8; vol.data.len()];
        for a in 0..d0 {
            for b in 0..d1 {
                for c in 0..d2 {
                    out[self.volume_index(vol.dims, [a, b, c])] = labels[(a * d1 + b) * d2 + c];
                }
            }
        }
        out
    }
}

/// `Spacing`'s output length along one axis.
pub fn spacing_len(n: usize, s_in: f64, s_out: f64) -> usize {
    ((n as f64 - 1.0) * s_in / s_out + 1.0)
        .round_ties_even()
        .max(1.0) as usize
}

/// `Spacing(pixdim, mode="bilinear")`: the volume `data` (`dims`, spacing
/// `s_in`) on a grid of spacing `s_out` from the same first voxel.
pub fn resample_linear(
    data: &[f32],
    dims: [usize; 3],
    s_in: [f64; 3],
    s_out: [f64; 3],
) -> (Vec<f32>, [usize; 3]) {
    let out_dims: [usize; 3] = std::array::from_fn(|a| spacing_len(dims[a], s_in[a], s_out[a]));
    let taps = |a: usize| -> Vec<(usize, usize, f32)> {
        (0..out_dims[a])
            .map(|i| {
                let c = (i as f64 * s_out[a] / s_in[a]).clamp(0.0, (dims[a] - 1) as f64);
                let i0 = c.floor() as usize;
                let i1 = (i0 + 1).min(dims[a] - 1);
                (i0, i1, (c - i0 as f64) as f32)
            })
            .collect()
    };
    let (t0, t1, t2) = (taps(0), taps(1), taps(2));
    let [_, n1, n2] = dims;
    let [o0, o1, o2] = out_dims;
    let mut out = vec![0f32; o0 * o1 * o2];
    out.par_chunks_mut(o1 * o2)
        .enumerate()
        .for_each(|(i, slab)| {
            let (a0, a1, fa) = t0[i];
            for (j, &(b0, b1, fb)) in t1.iter().enumerate() {
                for (k, &(c0, c1, fc)) in t2.iter().enumerate() {
                    let at = |a: usize, b: usize, c: usize| data[(a * n1 + b) * n2 + c];
                    let lerp = |x: f32, y: f32, t: f32| x + (y - x) * t;
                    let c00 = lerp(at(a0, b0, c0), at(a0, b0, c1), fc);
                    let c01 = lerp(at(a0, b1, c0), at(a0, b1, c1), fc);
                    let c10 = lerp(at(a1, b0, c0), at(a1, b0, c1), fc);
                    let c11 = lerp(at(a1, b1, c0), at(a1, b1, c1), fc);
                    slab[j * o2 + k] = lerp(lerp(c00, c01, fb), lerp(c10, c11, fb), fa);
                }
            }
        });
    (out, out_dims)
}

/// The inverse `Spacing` for labels (mode `nearest`): the labels on a grid
/// of `dims_out` voxels spaced `s_out`, from labels on `dims_in` spaced
/// `s_in`, both grids starting at the same first voxel.
pub fn resample_nearest(
    labels: &[u8],
    dims_in: [usize; 3],
    s_in: [f64; 3],
    dims_out: [usize; 3],
    s_out: [f64; 3],
) -> Vec<u8> {
    let idx = |a: usize| -> Vec<usize> {
        (0..dims_out[a])
            .map(|i| {
                let c = (i as f64 * s_out[a] / s_in[a]).round_ties_even();
                (c.max(0.0) as usize).min(dims_in[a] - 1)
            })
            .collect()
    };
    let (i0, i1, i2) = (idx(0), idx(1), idx(2));
    let [_, n1, n2] = dims_in;
    let [_, o1, o2] = dims_out;
    let mut out = vec![0u8; dims_out.iter().product()];
    out.par_chunks_mut(o1 * o2)
        .enumerate()
        .for_each(|(i, slab)| {
            for (j, &b) in i1.iter().enumerate() {
                for (k, &c) in i2.iter().enumerate() {
                    slab[j * o2 + k] = labels[(i0[i] * n1 + b) * n2 + c];
                }
            }
        });
    out
}

/// `NormalizeIntensity(nonzero=True)`: the non-zero voxels shifted and
/// scaled by their own mean and (population) standard deviation; the
/// zeros stay zero.
pub fn normalize_nonzero(data: &mut [f32]) {
    let (n, sum) = data
        .iter()
        .filter(|v| **v != 0.0)
        .fold((0usize, 0f64), |(n, s), &v| (n + 1, s + v as f64));
    if n == 0 {
        return;
    }
    let mean = sum / n as f64;
    let var = data
        .iter()
        .filter(|v| **v != 0.0)
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let std = if var > 0.0 { var.sqrt() } else { 1.0 };
    let (mean, std) = (mean as f32, std as f32);
    data.par_iter_mut().for_each(|v| {
        if *v != 0.0 {
            *v = (*v - mean) / std;
        }
    });
}

/// `ScaleIntensity(minv, maxv)`: the volume's own range onto `[minv, maxv]`.
pub fn scale_minmax(data: &mut [f32], minv: f32, maxv: f32) {
    let (lo, hi) = data
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    // Also when the data hold a NaN.
    if hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        data.par_iter_mut().for_each(|v| *v *= minv);
        return;
    }
    data.par_iter_mut()
        .for_each(|v| *v = (*v - lo) / (hi - lo) * (maxv - minv) + minv);
}

/// `ScaleIntensityRange(a_min, a_max, b_min, b_max, clip=True)`.
pub fn scale_range_clip(data: &mut [f32], a_min: f32, a_max: f32, b_min: f32, b_max: f32) {
    data.par_iter_mut().for_each(|v| {
        let t = (*v - a_min) / (a_max - a_min);
        *v = (t * (b_max - b_min) + b_min).clamp(b_min.min(b_max), b_max.max(b_min));
    });
}

/// `CropForeground()`: the box `[lo, hi)` of the voxels above zero; the
/// whole volume when there are none.
pub fn foreground_box(data: &[f32], dims: [usize; 3]) -> ([usize; 3], [usize; 3]) {
    let [_, d1, d2] = dims;
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for (v, &x) in data.iter().enumerate() {
        if x > 0.0 {
            let c = [v / (d1 * d2), (v / d2) % d1, v % d2];
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a] + 1);
            }
        }
    }
    if lo[0] == usize::MAX {
        ([0; 3], dims)
    } else {
        (lo, hi)
    }
}

/// Cut `[lo, hi)` out of a C-order volume.
pub fn crop<T: Copy + Send + Sync>(
    data: &[T],
    dims: [usize; 3],
    lo: [usize; 3],
    hi: [usize; 3],
) -> Vec<T> {
    let [_, d1, d2] = dims;
    let mut out = Vec::with_capacity((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
    for a in lo[0]..hi[0] {
        for b in lo[1]..hi[1] {
            let row = (a * d1 + b) * d2;
            out.extend_from_slice(&data[row + lo[2]..row + hi[2]]);
        }
    }
    out
}

/// Paste a cropped label volume back into zeros of `dims`.
pub fn uncrop(part: &[u8], dims: [usize; 3], lo: [usize; 3], hi: [usize; 3]) -> Vec<u8> {
    let [_, d1, d2] = dims;
    let w = hi[2] - lo[2];
    let h = hi[1] - lo[1];
    let mut out = vec![0u8; dims.iter().product()];
    for (aa, a) in (lo[0]..hi[0]).enumerate() {
        for (bb, b) in (lo[1]..hi[1]).enumerate() {
            let src = (aa * h + bb) * w;
            let dst = (a * d1 + b) * d2 + lo[2];
            out[dst..dst + w].copy_from_slice(&part[src..src + w]);
        }
    }
    out
}

/// `KeepLargestConnectedComponent()` on a label volume: each label keeps
/// its largest 26-connected piece (the first of equal size).
pub fn keep_largest_per_label(labels: &mut [u8], dims: [usize; 3]) {
    let (id, n) = crate::unet2d::post::label26(labels, dims);
    if n == 0 {
        return;
    }
    let mut size = vec![0usize; n + 1];
    let mut value = vec![0u8; n + 1];
    for (k, &l) in id.iter().zip(labels.iter()) {
        size[*k as usize] += 1;
        value[*k as usize] = l;
    }
    let mut best = [0usize; 256];
    for k in 1..=n {
        let l = value[k] as usize;
        if best[l] == 0 || size[k] > size[best[l]] {
            best[l] = k;
        }
    }
    for (l, &k) in labels.iter_mut().zip(&id) {
        if k != 0 && best[*l as usize] != k as usize {
            *l = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacing_matches_monai() {
        // monai.transforms.Spacing on a 10 x 4 x 3 ramp at 2 mm → 3 mm
        // (bilinear): 7 samples, 0, 15, 30, ...
        let mut data = vec![0f32; 10 * 4 * 3];
        for i in 0..10 {
            for j in 0..12 {
                data[i * 12 + j] = i as f32 * 10.0;
            }
        }
        let (out, d) = resample_linear(&data, [10, 4, 3], [2.0, 1.0, 1.0], [3.0, 1.0, 1.0]);
        assert_eq!(d, [7, 4, 3]);
        let col: Vec<f32> = (0..7).map(|i| out[i * 12]).collect();
        assert_eq!(col, [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]);
        // ... and back with nearest: 3 mm → 2 mm, 7 → 10, the thirds
        // rounding to the nearer sample.
        let labels: Vec<u8> = (0..7u8).flat_map(|i| std::iter::repeat_n(i, 12)).collect();
        let back = resample_nearest(
            &labels,
            [7, 4, 3],
            [3.0, 1.0, 1.0],
            [10, 4, 3],
            [2.0, 1.0, 1.0],
        );
        let col: Vec<u8> = (0..10).map(|i| back[i * 12]).collect();
        assert_eq!(col, [0, 1, 1, 2, 3, 3, 4, 5, 5, 6]);
        // Exact halves round to even, as torch's grid_sample does:
        // 1.5 mm → 0.75 mm of 0..5.
        let lab: Vec<u8> = (0..5u8).collect();
        let n = spacing_len(5, 1.5, 0.75);
        let up = resample_nearest(
            &lab,
            [5, 1, 1],
            [1.5, 1.0, 1.0],
            [n, 1, 1],
            [0.75, 1.0, 1.0],
        );
        assert_eq!(up, [0, 0, 1, 2, 2, 2, 3, 4, 4]);
    }

    #[test]
    fn intensity_transforms() {
        let mut v = vec![0.0, 1.0, 3.0];
        normalize_nonzero(&mut v);
        assert_eq!(v, [0.0, -1.0, 1.0]);
        scale_minmax(&mut v, -1.0, 1.0);
        assert_eq!(v, [0.0, -1.0, 1.0]);
        let mut h = vec![-2000.0, -1024.0, 512.0, 3000.0];
        scale_range_clip(&mut h, -1024.0, 2048.0, 0.0, 1.0);
        assert_eq!(h, [0.0, 0.0, 0.5, 1.0]);
    }

    #[test]
    fn foreground_crops_and_largest_pieces_survive() {
        let dims = [2, 3, 4];
        let mut d = vec![0f32; 24];
        d[5] = 1.0; // (0,1,1)
        d[19] = 2.0; // (1,1,3)
        let (lo, hi) = foreground_box(&d, dims);
        assert_eq!((lo, hi), ([0, 1, 1], [2, 2, 4]));
        let part = crop(&d, dims, lo, hi);
        assert_eq!(part.len(), 2 * 3, "2 x 1 x 3");
        let lab: Vec<u8> = part.iter().map(|&x| x as u8).collect();
        let back = uncrop(&lab, dims, lo, hi);
        assert_eq!((back[5], back[19]), (1, 2));
        let mut l = vec![1u8, 0, 1, 1, 2, 0];
        keep_largest_per_label(&mut l, [1, 1, 6]);
        assert_eq!(l, [0, 0, 1, 1, 2, 0]);
    }
}
