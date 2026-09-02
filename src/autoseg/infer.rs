//! nnU-Net sliding-window inference with Gaussian importance weighting.
//!
//! Faithful to `nnunetv2.inference` as TotalSegmentator invokes it for the
//! "total" task: tile step 0.8 × patch, Gaussian weight map (σ = patch/8,
//! scaled to 10 at the center), no mirroring TTA, argmax on the weighted
//! logit sum. The logit accumulator is a ring buffer over the leading
//! spatial axis: rows are finalized (argmax → u8 label) as soon as no future
//! tile can touch them, so peak memory stays ≈ `classes × patch₀ × D₁ × D₂`
//! floats regardless of scan length.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::config::ModelConfig;

/// nnU-Net `compute_steps_for_sliding_window`.
fn compute_steps(image_size: usize, tile_size: usize, step_frac: f64) -> Vec<usize> {
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

/// 1-D Gaussian importance profile for one patch axis (σ = len/8, center at
/// len/2 - nnU-Net's `compute_gaussian`; the ×10 scaling and per-axis kernel
/// normalizations cancel in the argmax and are omitted).
fn gauss_profile(len: usize) -> Vec<f32> {
    let sigma = len as f64 / 8.0;
    let center = (len / 2) as f64;
    (0..len)
        .map(|i| {
            let t = (i as f64 - center) / sigma;
            (-0.5 * t * t).exp() as f32
        })
        .collect()
}

/// Callbacks the sliding window needs from the caller.
pub trait InferHooks: Sync {
    /// Forward one normalized patch `[1, p0, p1, p2]` (flattened, C-order)
    /// → logits `[classes, p0, p1, p2]` (flattened, C-order).
    fn forward(&self, patch: &[f32]) -> Result<Vec<f32>>;
    /// Called after each tile: `done` of `total`. Return false to cancel.
    fn tile_done(&self, done: usize, total: usize) -> bool;
}

/// Run sliding-window inference over a resampled volume.
///
/// * `vol` - raw HU values on the model grid, layout `[d0][d1][d2]`.
/// * returns per-voxel argmax labels (local class indices) on the same grid.
pub fn predict(
    vol: &[f32],
    dims: [usize; 3],
    classes: usize,
    cfg: &ModelConfig,
    step_frac: f64,
    hooks: &dyn InferHooks,
) -> Result<Vec<u8>> {
    let [d0, d1, d2] = dims;
    let [p0, p1, p2] = cfg.patch_size;
    if classes > u8::MAX as usize {
        bail!("too many classes for u8 labels");
    }
    // pad up to the patch size (nnU-Net pads centered with zeros)
    let (pd0, pd1, pd2) = (d0.max(p0), d1.max(p1), d2.max(p2));
    let off = [(pd0 - d0) / 2, (pd1 - d1) / 2, (pd2 - d2) / 2];
    let steps0 = compute_steps(pd0, p0, step_frac);
    let steps1 = compute_steps(pd1, p1, step_frac);
    let steps2 = compute_steps(pd2, p2, step_frac);
    let total_tiles = steps0.len() * steps1.len() * steps2.len();

    let g0 = gauss_profile(p0);
    let g1 = gauss_profile(p1);
    let g2 = gauss_profile(p2);

    // ring accumulator over axis 0: W rows of [classes][d1*d2]
    let win = p0;
    let plane = d1 * d2;
    let mut acc = vec![0f32; win * classes * plane];
    let mut labels = vec![0u8; d0 * plane];
    let mut finalized = 0usize; // orig rows < finalized are done

    let inv_std = 1.0 / cfg.std.max(1e-8);
    let normalize = |v: f32| -> f32 { (v.clamp(cfg.clip_lo, cfg.clip_hi) - cfg.mean) * inv_std };

    let mut patch = vec![0f32; p0 * p1 * p2];
    let mut done_tiles = 0usize;

    let finalize_rows = |acc: &mut [f32], labels: &mut [u8], from: usize, to: usize| {
        // argmax per voxel of rows [from, to), then zero the slots for reuse
        labels[from * plane..to * plane]
            .par_chunks_mut(plane)
            .zip(from..to)
            .for_each(|(lrow, r)| {
                let slot = r % win;
                let base = slot * classes * plane;
                for (v, lab) in lrow.iter_mut().enumerate() {
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
            });
        for r in from..to {
            let slot = r % win;
            acc[slot * classes * plane..(slot + 1) * classes * plane].fill(0.0);
        }
    };

    for &s0 in &steps0 {
        // rows strictly below this tile row's start receive no further writes
        let tile_lo = s0.saturating_sub(off[0]);
        if tile_lo > finalized {
            let to = tile_lo.min(d0);
            finalize_rows(&mut acc, &mut labels, finalized, to);
            finalized = to;
        }
        for &s1 in &steps1 {
            for &s2 in &steps2 {
                // ---- extract + normalize the patch (0 outside the volume) --
                patch
                    .par_chunks_mut(p1 * p2)
                    .enumerate()
                    .for_each(|(pz, prow)| {
                        let z = s0 + pz;
                        if z < off[0] || z >= off[0] + d0 {
                            prow.fill(0.0);
                            return;
                        }
                        let vz = (z - off[0]) * plane;
                        for py in 0..p1 {
                            let y = s1 + py;
                            let dst = &mut prow[py * p2..(py + 1) * p2];
                            if y < off[1] || y >= off[1] + d1 {
                                dst.fill(0.0);
                                continue;
                            }
                            let vy = vz + (y - off[1]) * d2;
                            for (px, d) in dst.iter_mut().enumerate() {
                                let x = s2 + px;
                                *d = if x < off[2] || x >= off[2] + d2 {
                                    0.0
                                } else {
                                    normalize(vol[vy + (x - off[2])])
                                };
                            }
                        }
                    });
                // ---- forward ----------------------------------------------
                let logits = hooks.forward(&patch)?;
                if logits.len() != classes * p0 * p1 * p2 {
                    bail!(
                        "model returned {} logits, expected {}",
                        logits.len(),
                        classes * p0 * p1 * p2
                    );
                }
                // ---- weighted accumulate ----------------------------------
                // parallel over patch rows pz (distinct accumulator rows)
                let acc_ptr = SendPtr(acc.as_mut_ptr());
                (0..p0).into_par_iter().for_each(|pz| {
                    let z = s0 + pz;
                    if z < off[0] || z >= off[0] + d0 {
                        return;
                    }
                    let r = z - off[0];
                    let slot = r % win;
                    let w0 = g0[pz];
                    let acc = unsafe {
                        std::slice::from_raw_parts_mut(
                            acc_ptr.get().add(slot * classes * plane),
                            classes * plane,
                        )
                    };
                    for c in 0..classes {
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
                });
                done_tiles += 1;
                if !hooks.tile_done(done_tiles, total_tiles) {
                    bail!("cancelled");
                }
            }
        }
    }
    finalize_rows(&mut acc, &mut labels, finalized, d0);
    Ok(labels)
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
    fn gaussian_profile_shape() {
        let g = gauss_profile(112);
        assert!((g[56] - 1.0).abs() < 1e-6); // center at len/2
        assert!(g[0] < g[28] && g[28] < g[56]);
        assert!(g[111] < g[56]);
    }

    /// A fake single-class-per-position model: verify the sliding window
    /// covers every voxel and the argmax label lands where the "model" put it.
    #[test]
    fn sliding_window_covers_and_labels() {
        struct Hooks;
        impl InferHooks for Hooks {
            fn forward(&self, patch: &[f32]) -> Result<Vec<f32>> {
                // 2 classes: class 1 wherever patch value > 0.5 else class 0
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
        let cfg = ModelConfig {
            norm: crate::autoseg::config::Norm::Ct,
            patch_size: [8, 8, 8],
            spacing: [3.0, 3.0, 3.0],
            features: vec![],
            kernels: vec![],
            strides: vec![],
            n_conv_per_stage: vec![],
            n_conv_per_stage_decoder: vec![],
            clip_lo: -1.0,
            clip_hi: 1.0,
            mean: 0.0,
            std: 1.0,
        };
        // volume bigger than the patch in axis0, smaller in axis2 (padding)
        let dims = [20, 8, 6];
        let mut vol = vec![0f32; 20 * 8 * 6];
        // mark a block with value 1 (→ class 1)
        for z in 5..15 {
            for y in 2..6 {
                for x in 1..5 {
                    vol[(z * 8 + y) * 6 + x] = 1.0;
                }
            }
        }
        let labels = predict(&vol, dims, 2, &cfg, 0.5, &Hooks).unwrap();
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
}
