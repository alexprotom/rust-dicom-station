//! Sliding-window inference with importance weighting, for every 3-D model
//! family: nnU-Net's predictor and MONAI's `SlidingWindowInferer` are the
//! same loop with different constants, and both are a [`WindowPlan`] here.
//!
//! * nnU-Net (`nnunetv2.inference`, as TotalSegmentator invokes it): tile
//!   step = `step_frac` x patch, Gaussian weight (sigma = patch/8, centre
//!   at patch/2), the volume zero-padded (in normalized space) to at least
//!   the patch, centred - [`nnunet_plan`].
//! * MONAI: interval `int(patch * (1 - overlap))`, windows from
//!   `dense_patch_slices`, Gaussian centred at (patch-1)/2 with
//!   sigma = 0.125 x patch and a floor of `max(min, 1e-3)`, padding
//!   constant or edge-replicated - [`monai_plan`].
//!
//! The argmax of the weighted logit sum is the answer (normalizing by the
//! weight sum, as both upstreams do, does not change it). The accumulator
//! is a ring buffer over the leading spatial axis - rows are finalized as
//! soon as no later window can touch them - and, when even that would
//! exceed the memory budget (a hundred classes at native resolution), the
//! volume is processed in strips along the second axis, each strip re-running
//! the windows that cross it. Peak memory stays near
//! `classes x patch0 x strip x D2` floats whatever the scan's size.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::config::ModelConfig;

/// nnU-Net `compute_steps_for_sliding_window`.
pub fn compute_steps(image_size: usize, tile_size: usize, step_frac: f64) -> Vec<usize> {
    debug_assert!(image_size >= tile_size);
    let target = tile_size as f64 * step_frac;
    let num = if image_size > tile_size {
        ((image_size - tile_size) as f64 / target).ceil() as usize + 1
    } else {
        1
    };
    if num == 1 {
        return vec![0];
    }
    let max_step = (image_size - tile_size) as f64 / (num - 1) as f64;
    (0..num)
        .map(|i| (max_step * i as f64).round() as usize)
        .collect()
}

/// 1-D Gaussian importance profile for one patch axis (sigma = len/8,
/// centre at len/2 - nnU-Net's `compute_gaussian`; the x10 scaling and
/// per-axis kernel normalizations cancel in the argmax and are omitted).
pub fn gauss_profile(len: usize) -> Vec<f32> {
    let sigma = len as f64 / 8.0;
    let center = (len / 2) as f64;
    (0..len)
        .map(|i| {
            let t = (i as f64 - center) / sigma;
            (-0.5 * t * t).exp() as f32
        })
        .collect()
}

/// MONAI `_get_scan_interval` + `dense_patch_slices` for one axis of a
/// (padded) image at least as long as the patch.
pub fn monai_starts(image: usize, patch: usize, overlap: f64) -> Vec<usize> {
    let interval = if patch == image {
        patch
    } else {
        ((patch as f64 * (1.0 - overlap)) as usize).max(1)
    };
    let num = image.div_ceil(interval);
    let scan = (0..num)
        .find(|d| d * interval + patch >= image)
        .map_or(1, |d| d + 1);
    (0..scan)
        .map(|i| {
            let s = i * interval;
            s - (s + patch).saturating_sub(image)
        })
        .collect()
}

/// MONAI's 1-D Gaussian (`compute_importance_map`, mode `gaussian`): centred
/// at (len-1)/2, sigma = `sigma_scale` x len.
pub fn monai_profile(len: usize, sigma_scale: f64) -> Vec<f32> {
    let sigma = sigma_scale * len as f64;
    (0..len)
        .map(|i| {
            // torch.arange in f32, then exp in f32.
            let x = (-(len as f32 - 1.0) / 2.0) + i as f32;
            (x * x / (-2.0 * (sigma as f32) * (sigma as f32))).exp()
        })
        .collect()
}

/// What a patch sees outside the volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// Zeros (nnU-Net's padding, MONAI's `constant`).
    Zero,
    /// The nearest edge voxel (MONAI's `replicate`).
    Replicate,
}

/// The windows, weights and padding of one sliding-window pass.
#[derive(Clone, Debug)]
pub struct WindowPlan {
    pub patch: [usize; 3],
    /// Front padding per axis: the volume starts at this index of the
    /// padded grid the window starts refer to.
    pub off: [usize; 3],
    /// Window start positions per axis, on the padded grid.
    pub starts: [Vec<usize>; 3],
    /// Weight per patch position, per axis; the weight of a voxel is the
    /// product, raised to at least `floor`.
    pub profiles: [Vec<f32>; 3],
    pub floor: f32,
    pub fill: Fill,
}

/// nnU-Net's plan: pad centred to the patch, steps of `step_frac`.
pub fn nnunet_plan(dims: [usize; 3], patch: [usize; 3], step_frac: f64) -> WindowPlan {
    let padded: [usize; 3] = std::array::from_fn(|a| dims[a].max(patch[a]));
    WindowPlan {
        patch,
        off: std::array::from_fn(|a| (padded[a] - dims[a]) / 2),
        starts: std::array::from_fn(|a| compute_steps(padded[a], patch[a], step_frac)),
        profiles: std::array::from_fn(|a| gauss_profile(patch[a])),
        floor: 0.0,
        fill: Fill::Zero,
    }
}

/// MONAI's `SlidingWindowInferer` plan (gaussian mode, sigma 0.125).
pub fn monai_plan(dims: [usize; 3], patch: [usize; 3], overlap: f64, fill: Fill) -> WindowPlan {
    let padded: [usize; 3] = std::array::from_fn(|a| dims[a].max(patch[a]));
    let profiles: [Vec<f32>; 3] = std::array::from_fn(|a| monai_profile(patch[a], 0.125));
    // The 3-D map's minimum is the product of the 1-D minima; MONAI clamps
    // the map to at least that or 1e-3, whichever is larger.
    let min: f32 = profiles
        .iter()
        .map(|p| p.iter().copied().fold(f32::INFINITY, f32::min))
        .product();
    WindowPlan {
        patch,
        off: std::array::from_fn(|a| (padded[a] - dims[a]) / 2),
        starts: std::array::from_fn(|a| monai_starts(padded[a], patch[a], overlap)),
        profiles,
        floor: min.max(1e-3),
        fill,
    }
}

/// Callbacks the sliding window needs from the caller.
pub trait InferHooks: Sync {
    /// Forward one patch `[1, p0, p1, p2]` (flattened, C-order) → logits
    /// `[classes, p0, p1, p2]` (flattened, C-order).
    fn forward(&self, patch: &[f32]) -> Result<Vec<f32>>;
    /// Called after each tile: `done` of `total`. Return false to cancel.
    fn tile_done(&self, done: usize, total: usize) -> bool;
}

/// Memory the logit accumulator may take before the volume is split into
/// strips (bytes).
pub const ACC_BUDGET: usize = 3 << 30;

/// Run nnU-Net sliding-window inference over a resampled volume.
///
/// * `vol` - raw values on the model grid, layout `[d0][d1][d2]`, normalized
///   here with `cfg`'s constants.
/// * returns per-voxel argmax labels (local class indices) on the same grid.
pub fn predict(
    vol: &[f32],
    dims: [usize; 3],
    classes: usize,
    cfg: &ModelConfig,
    step_frac: f64,
    hooks: &dyn InferHooks,
) -> Result<Vec<u8>> {
    let inv_std = 1.0 / cfg.std.max(1e-8);
    let normalized: Vec<f32> = vol
        .par_iter()
        .map(|&v| (v.clamp(cfg.clip_lo, cfg.clip_hi) - cfg.mean) * inv_std)
        .collect();
    let plan = nnunet_plan(dims, cfg.patch_size, step_frac);
    predict_plan(&normalized, dims, classes, &plan, ACC_BUDGET, hooks)
}

/// Run a sliding-window pass described by `plan` over an already
/// normalized volume; per-voxel argmax labels on the same grid.
pub fn predict_plan(
    vol: &[f32],
    dims: [usize; 3],
    classes: usize,
    plan: &WindowPlan,
    budget: usize,
    hooks: &dyn InferHooks,
) -> Result<Vec<u8>> {
    let [d0, d1, d2] = dims;
    let [p0, p1, p2] = plan.patch;
    if classes > u8::MAX as usize + 1 {
        bail!("too many classes for u8 labels");
    }
    // Strip height along axis 1 that keeps the ring within the budget.
    let per_row = classes * p0 * d2 * std::mem::size_of::<f32>();
    let strip = (budget / per_row.max(1)).clamp(1, d1.max(1));
    let strips: Vec<(usize, usize)> = (0..d1)
        .step_by(strip)
        .map(|lo| (lo, (lo + strip).min(d1)))
        .collect();
    // Which windows cross each strip (by their axis-1 start).
    let crosses = |s1: usize, (lo, hi): (usize, usize)| -> bool {
        let v_lo = s1.saturating_sub(plan.off[1]);
        let v_hi = (s1 + p1).saturating_sub(plan.off[1]).min(d1);
        v_lo < hi && v_hi > lo
    };
    let total_tiles: usize = strips
        .iter()
        .map(|&st| {
            plan.starts[0].len()
                * plan.starts[1].iter().filter(|&&s1| crosses(s1, st)).count()
                * plan.starts[2].len()
        })
        .sum();

    let mut labels = vec![0u8; d0 * d1 * d2];
    let mut patch = vec![0f32; p0 * p1 * p2];
    let mut done_tiles = 0usize;
    for &(y_lo, y_hi) in &strips {
        let h = y_hi - y_lo;
        let plane = h * d2;
        let win = p0;
        let mut acc = vec![0f32; win * classes * plane];
        let mut finalized = 0usize;
        let finalize_rows = |acc: &mut [f32], labels: &mut [u8], from: usize, to: usize| {
            // argmax per voxel of rows [from, to), then zero the slots.
            let rows: Vec<(usize, Vec<u8>)> = (from..to)
                .into_par_iter()
                .map(|r| {
                    let base = (r % win) * classes * plane;
                    let mut out = vec![0u8; plane];
                    for (v, lab) in out.iter_mut().enumerate() {
                        let mut best = 0usize;
                        let mut best_v = f32::NEG_INFINITY;
                        for c in 0..classes {
                            let a = acc[base + c * plane + v];
                            if a > best_v {
                                best_v = a;
                                best = c;
                            }
                        }
                        *lab = best as u8;
                    }
                    (r, out)
                })
                .collect();
            for (r, out) in rows {
                for y in 0..h {
                    let dst = (r * d1 + y_lo + y) * d2;
                    labels[dst..dst + d2].copy_from_slice(&out[y * d2..(y + 1) * d2]);
                }
                let slot = r % win;
                acc[slot * classes * plane..(slot + 1) * classes * plane].fill(0.0);
            }
        };
        for &s0 in &plan.starts[0] {
            // Rows strictly below this tile row's start receive no more writes.
            let tile_lo = s0.saturating_sub(plan.off[0]);
            if tile_lo > finalized {
                let to = tile_lo.min(d0);
                finalize_rows(&mut acc, &mut labels, finalized, to);
                finalized = to;
            }
            for &s1 in plan.starts[1]
                .iter()
                .filter(|&&s1| crosses(s1, (y_lo, y_hi)))
            {
                for &s2 in &plan.starts[2] {
                    extract_patch(vol, dims, plan, [s0, s1, s2], &mut patch);
                    let logits = hooks.forward(&patch)?;
                    if logits.len() != classes * p0 * p1 * p2 {
                        bail!(
                            "model returned {} logits, expected {}",
                            logits.len(),
                            classes * p0 * p1 * p2
                        );
                    }
                    let tile = Tile {
                        origin: [s0, s1, s2],
                        off: plan.off,
                        dims,
                        patch: plan.patch,
                        classes,
                        rows: (y_lo, y_hi),
                    };
                    let acc_ptr = SendPtr(acc.as_mut_ptr());
                    (0..p0).into_par_iter().for_each(|pz| {
                        let z = s0 + pz;
                        if z < plan.off[0] || z >= plan.off[0] + d0 {
                            return;
                        }
                        let slot = (z - plan.off[0]) % win;
                        // SAFETY: each pz maps to its own ring slot (the
                        // patch is at most `win` rows tall), so the slices
                        // written in parallel are disjoint.
                        let acc = unsafe {
                            std::slice::from_raw_parts_mut(
                                acc_ptr.get().add(slot * classes * plane),
                                classes * plane,
                            )
                        };
                        tile.accumulate_row(
                            acc,
                            &logits,
                            pz,
                            plan.profiles[0][pz],
                            &plan.profiles[1],
                            &plan.profiles[2],
                            plan.floor,
                        );
                    });
                    done_tiles += 1;
                    if !hooks.tile_done(done_tiles, total_tiles) {
                        bail!("cancelled");
                    }
                }
            }
        }
        finalize_rows(&mut acc, &mut labels, finalized, d0);
    }
    Ok(labels)
}

/// Copy the patch at padded-grid `origin` out of the volume, filling
/// what lies outside it as the plan says.
fn extract_patch(
    vol: &[f32],
    dims: [usize; 3],
    plan: &WindowPlan,
    origin: [usize; 3],
    patch: &mut [f32],
) {
    let [d0, d1, d2] = dims;
    let [_, p1, p2] = plan.patch;
    let off = plan.off;
    let plane = d1 * d2;
    let at = |idx: usize, a: usize, n: usize| -> Option<usize> {
        let v = idx as isize - off[a] as isize;
        if v >= 0 && (v as usize) < n {
            Some(v as usize)
        } else {
            match plan.fill {
                Fill::Zero => None,
                Fill::Replicate => Some(v.clamp(0, n as isize - 1) as usize),
            }
        }
    };
    patch
        .par_chunks_mut(p1 * p2)
        .enumerate()
        .for_each(|(pz, prow)| {
            let Some(z) = at(origin[0] + pz, 0, d0) else {
                prow.fill(0.0);
                return;
            };
            for py in 0..p1 {
                let dst = &mut prow[py * p2..(py + 1) * p2];
                let Some(y) = at(origin[1] + py, 1, d1) else {
                    dst.fill(0.0);
                    continue;
                };
                let row = z * plane + y * d2;
                for (px, d) in dst.iter_mut().enumerate() {
                    *d = match at(origin[2] + px, 2, d2) {
                        Some(x) => vol[row + x],
                        None => 0.0,
                    };
                }
            }
        });
}

/// Where one tile sits: its corner on the padded grid, the padding, the
/// volume, the patch, and the rows of axis 1 the current strip keeps.
struct Tile {
    origin: [usize; 3],
    off: [usize; 3],
    dims: [usize; 3],
    patch: [usize; 3],
    classes: usize,
    rows: (usize, usize),
}

impl Tile {
    /// The patch indices along `axis` that land inside the volume (and,
    /// on axis 1, inside the strip).
    fn inside(&self, axis: usize) -> std::ops::Range<usize> {
        let (vol_lo, vol_hi) = if axis == 1 {
            self.rows
        } else {
            (0, self.dims[axis])
        };
        let lo = (self.off[axis] + vol_lo).saturating_sub(self.origin[axis]);
        let hi = (self.off[axis] + vol_hi)
            .saturating_sub(self.origin[axis])
            .min(self.patch[axis]);
        lo..hi.max(lo)
    }

    /// Add patch row `pz` of the tile's logits, weighted, into its
    /// accumulator row `acc` (`[classes][strip rows * d2]`).
    ///
    /// The part of each patch row inside the volume is worked out once, not
    /// tested voxel by voxel, which leaves the inner loop a plain
    /// multiply-add over three slices that the compiler vectorizes. With no
    /// floor every voxel gets exactly `acc += logit * (w0 * g1) * g2`, in
    /// that order, so the sums are the ones the per-voxel loop made to the
    /// bit; with one, the weight `max(w0 * g1 * g2, floor)` is formed first.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_row(
        &self,
        acc: &mut [f32],
        logits: &[f32],
        pz: usize,
        w0: f32,
        g1: &[f32],
        g2: &[f32],
        floor: f32,
    ) {
        let [p0, p1, p2] = self.patch;
        let [_, _, d2] = self.dims;
        let strip_lo = self.rows.0;
        let plane = (self.rows.1 - self.rows.0) * d2;
        let (ys, xs) = (self.inside(1), self.inside(2));
        if xs.is_empty() {
            return;
        }
        let g2 = &g2[xs.clone()];
        // The first volume column the patch row covers.
        let x0 = self.origin[2] + xs.start - self.off[2];
        let mut w = vec![0f32; xs.len()];
        for c in 0..self.classes {
            let lbase = ((c * p0) + pz) * p1 * p2;
            let abase = c * plane;
            for py in ys.clone() {
                let wy = w0 * g1[py];
                let y = self.origin[1] + py - self.off[1] - strip_lo;
                let arow = &mut acc[abase + y * d2 + x0..][..xs.len()];
                let lrow = &logits[lbase + py * p2 + xs.start..][..xs.len()];
                if floor > 0.0 {
                    for (wv, &g) in w.iter_mut().zip(g2) {
                        *wv = (wy * g).max(floor);
                    }
                    for ((a, &l), &wv) in arow.iter_mut().zip(lrow).zip(&w) {
                        *a += l * wv;
                    }
                } else {
                    for ((a, &l), &g) in arow.iter_mut().zip(lrow).zip(g2) {
                        *a += l * wy * g;
                    }
                }
            }
        }
    }
}

/// Wrapper making a raw pointer Sync for the disjoint-row parallel loop.
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// Method (not field) access, so closures capture the whole wrapper.
    fn get(&self) -> *mut f32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_match_nnunet() {
        // Reference values computed with nnunetv2's
        // compute_steps_for_sliding_window.
        assert_eq!(compute_steps(112, 112, 0.8), vec![0]);
        assert_eq!(compute_steps(167, 112, 0.8), vec![0, 55]);
        assert_eq!(compute_steps(300, 112, 0.8), vec![0, 63, 125, 188]);
        assert_eq!(compute_steps(128, 128, 0.5), vec![0]);
        assert_eq!(compute_steps(200, 128, 0.5), vec![0, 36, 72]);
    }

    #[test]
    fn starts_match_monai() {
        // monai.data.utils.dense_patch_slices with _get_scan_interval.
        assert_eq!(monai_starts(96, 96, 0.25), vec![0]);
        assert_eq!(monai_starts(200, 96, 0.25), vec![0, 72, 104]);
        assert_eq!(monai_starts(133, 96, 0.625), vec![0, 36, 37]);
        assert_eq!(
            monai_starts(512, 160, 0.625),
            vec![0, 60, 120, 180, 240, 300, 352]
        );
        let g = monai_profile(4, 0.125);
        assert!((g[0] - g[3]).abs() < 1e-7 && (g[1] - g[2]).abs() < 1e-7 && g[1] > g[0]);
    }

    /// The accumulate loop as it was: every voxel of the patch tested
    /// against the volume on its own.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_row_per_voxel(
        acc: &mut [f32],
        logits: &[f32],
        t: &Tile,
        pz: usize,
        w0: f32,
        g1: &[f32],
        g2: &[f32],
    ) {
        let [_, s1, s2] = t.origin;
        let [p0, p1, p2] = t.patch;
        let [_, d1, d2] = t.dims;
        let off = t.off;
        let plane = d1 * d2;
        for c in 0..t.classes {
            let lbase = ((c * p0) + pz) * p1 * p2;
            let abase = c * plane;
            for (py, g1v) in g1.iter().enumerate() {
                let y = s1 + py;
                if y < off[1] || y >= off[1] + d1 {
                    continue;
                }
                let wy = w0 * g1v;
                let arow = abase + (y - off[1]) * d2;
                let lrow = lbase + py * p2;
                for px in 0..p2 {
                    let x = s2 + px;
                    if x < off[2] || x >= off[2] + d2 {
                        continue;
                    }
                    acc[arow + (x - off[2])] += logits[lrow + px] * wy * g2[px];
                }
            }
        }
    }

    #[test]
    fn the_accumulate_row_adds_what_the_per_voxel_loop_added() {
        // A patch wider than the volume on one axis (padding on both sides)
        // and narrower on the other, at several tile positions.
        let classes = 3;
        let patch = [4, 7, 9];
        let g0 = gauss_profile(patch[0]);
        let g1 = gauss_profile(patch[1]);
        let g2 = gauss_profile(patch[2]);
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let logits: Vec<f32> = (0..classes * patch[0] * patch[1] * patch[2])
            .map(|_| rand())
            .collect();
        for (dims, off, origin) in [
            ([4, 12, 5], [0, 0, 2], [0, 0, 0]),
            ([4, 12, 5], [0, 0, 2], [0, 5, 0]),
            ([4, 20, 30], [0, 0, 0], [0, 13, 21]),
            ([4, 5, 30], [0, 1, 0], [0, 0, 11]),
        ] {
            let t = Tile {
                origin,
                off,
                dims,
                patch,
                classes,
                rows: (0, dims[1]),
            };
            let plane = dims[1] * dims[2];
            let start: Vec<f32> = (0..classes * plane).map(|_| rand()).collect();
            for (pz, &w0) in g0.iter().enumerate() {
                let mut a = start.clone();
                let mut b = start.clone();
                t.accumulate_row(&mut a, &logits, pz, w0, &g1, &g2, 0.0);
                accumulate_row_per_voxel(&mut b, &logits, &t, pz, w0, &g1, &g2);
                let (a, b): (Vec<u32>, Vec<u32>) = (
                    a.iter().map(|v| v.to_bits()).collect(),
                    b.iter().map(|v| v.to_bits()).collect(),
                );
                assert_eq!(a, b, "dims {dims:?} origin {origin:?} row {pz}");
            }
        }
    }

    #[test]
    fn gaussian_profile_shape() {
        let g = gauss_profile(112);
        assert!((g[56] - 1.0).abs() < 1e-6); // center at len/2
        assert!(g[0] < g[28] && g[28] < g[56]);
        assert!(g[111] < g[56]);
    }

    /// A fake model: class 1 wherever the patch value > 0.5.
    struct Threshold;
    impl InferHooks for Threshold {
        fn forward(&self, patch: &[f32]) -> Result<Vec<f32>> {
            let mut out = vec![0f32; 2 * patch.len()];
            for (i, v) in patch.iter().enumerate() {
                out[i] = 1.0 - v; // class 0 logit
                out[patch.len() + i] = *v; // class 1 logit
            }
            Ok(out)
        }
        fn tile_done(&self, _d: usize, _t: usize) -> bool {
            true
        }
    }

    fn block_volume() -> (Vec<f32>, [usize; 3]) {
        // bigger than the patch in axis 0, smaller in axis 2 (padding)
        let dims = [20, 8, 6];
        let mut vol = vec![0f32; 20 * 8 * 6];
        for z in 5..15 {
            for y in 2..6 {
                for x in 1..5 {
                    vol[(z * 8 + y) * 6 + x] = 1.0;
                }
            }
        }
        (vol, dims)
    }

    fn expect_block(labels: &[u8]) {
        for z in 0..20 {
            for y in 0..8 {
                for x in 0..6 {
                    let expect =
                        ((5..15).contains(&z) && (2..6).contains(&y) && (1..5).contains(&x)) as u8;
                    assert_eq!(labels[(z * 8 + y) * 6 + x], expect, "at {z},{y},{x}");
                }
            }
        }
    }

    /// The sliding window covers every voxel and the argmax label lands
    /// where the "model" put it - under nnU-Net's plan, MONAI's, and with
    /// the volume cut into strips.
    #[test]
    fn sliding_window_covers_and_labels() {
        let cfg = ModelConfig {
            arch: crate::autoseg::config::Arch::PlainConv,
            conv_bias: true,
            norm: crate::autoseg::config::Norm::Ct,
            patch_size: [8, 8, 8],
            spacing: [3.0, 3.0, 3.0],
            features: vec![],
            kernels: vec![],
            strides: vec![],
            n_conv_per_stage: vec![],
            n_conv_per_stage_decoder: vec![],
            decoder_kernels: vec![],
            clip_lo: -1.0,
            clip_hi: 1.0,
            mean: 0.0,
            std: 1.0,
        };
        let (vol, dims) = block_volume();
        expect_block(&predict(&vol, dims, 2, &cfg, 0.5, &Threshold).unwrap());
        let plan = monai_plan(dims, [8, 8, 8], 0.25, Fill::Replicate);
        expect_block(&predict_plan(&vol, dims, 2, &plan, ACC_BUDGET, &Threshold).unwrap());
        // A budget of one row of axis 1 per strip: eight strips.
        let tiny = 2 * 8 * dims[2] * 4;
        let plan = nnunet_plan(dims, [8, 8, 8], 0.5);
        expect_block(&predict_plan(&vol, dims, 2, &plan, tiny, &Threshold).unwrap());
    }
}
