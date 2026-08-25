//! Zoom-out / zoom-in inference.
//!
//! The network only ever accepts a 32x256x256 volume, so a study is segmented
//! by running that same fixed graph twice: once over the whole volume crushed
//! down to the input shape, and then again as a sliding window over a crop
//! around whatever the first pass found. The paper's own measurements make
//! the case — resize-only is fast and coarse, sliding-window-only is accurate
//! and slow, and the two-stage scheme gets most of the accuracy for a small
//! multiple of the cost.
//!
//! Two deliberate divergences from the reference, both optional and both off
//! by default:
//!
//! * [`Config::skip_coarse_with_box`] — when the user has drawn a box, the
//!   coarse pass is only being asked to *find* a region that has already been
//!   pointed at. Cropping straight to the box halves the compute and stops
//!   small lesions being lost to the 32x256x256 downsample, which is where
//!   the coarse pass is weakest.
//! * the coarse logits are kept at the network's own output resolution rather
//!   than being upsampled to the full volume first. The only things read out
//!   of them are a threshold and a bounding box, and on a 512x512x300 study
//!   the full-resolution copy would cost 314 MB for nothing.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::config::*;
use super::net::SegVolNet;
use super::preprocess::{self, Prepared};
use super::prompt::{BBox, Point};
use crate::progress::{ProgressSink, CANCELLED};

/// How to run inference.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Run the sliding-window refinement pass at all.
    pub use_zoom_in: bool,
    /// Window overlap; the reference passes 0.5, giving a (16, 128, 128)
    /// stride.
    pub overlap: f32,
    /// Probability threshold applied to the sigmoid of the logits.
    pub threshold: f32,
    /// Crop straight to a user-supplied box instead of locating the region
    /// with a coarse pass first. Off by default so the faithful behaviour is
    /// what runs unless asked otherwise.
    pub skip_coarse_with_box: bool,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            use_zoom_in: true,
            overlap: 0.5,
            threshold: 0.5,
            skip_coarse_with_box: false,
        }
    }
}

/// The result of one segmentation.
pub struct Segmentation {
    /// One byte per voxel of the **prepared** grid — canonically oriented and
    /// cropped. Callers map it onto the original volume with
    /// [`Prepared::mask_to_volume_grid`].
    pub mask: Vec<u8>,
    pub voxels: u64,
    /// Windows run in the refinement pass; 0 when it was skipped.
    pub windows: usize,
    /// Whether a coarse pass ran.
    pub coarse: bool,
}

/// MONAI's `dense_patch_slices`: window start positions along one axis.
///
/// The count is `ceil(image / interval)` and each start is pulled back so the
/// window fits inside the image, which means the last few windows can overlap
/// more than the nominal stride. This is *not* nnU-Net's rule — that one
/// spreads the windows evenly — so the two engines cannot share it.
pub fn window_starts(image: usize, patch: usize, interval: usize) -> Vec<usize> {
    if image <= patch {
        return vec![0];
    }
    let interval = interval.max(1);
    let n = image.div_ceil(interval);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let start = i * interval;
        let start = start.saturating_sub((start + patch).saturating_sub(image));
        if out.last() != Some(&start) {
            out.push(start);
        }
        if start + patch >= image {
            break;
        }
    }
    out
}

/// Every window origin for a crop of `dims`.
pub fn window_grid(dims: [usize; 3], overlap: f32) -> Vec<[usize; 3]> {
    let interval: Vec<usize> = (0..3)
        .map(|a| ((ROI[a] as f32 * (1.0 - overlap)) as usize).max(1))
        .collect();
    let starts: Vec<Vec<usize>> = (0..3)
        .map(|a| window_starts(dims[a], ROI[a], interval[a]))
        .collect();
    let mut out = Vec::new();
    for &a in &starts[0] {
        for &b in &starts[1] {
            for &c in &starts[2] {
                out.push([a, b, c]);
            }
        }
    }
    out
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Bounding box of everything above `threshold`, padded so each axis is at
/// least `ROI` long, clamped to the volume.
fn roi_from_logits(
    logits: &[f32],
    dims: [usize; 3],
    threshold: f32,
) -> Option<([usize; 3], [usize; 3])> {
    let t = -((1.0 / threshold - 1.0).ln()); // the logit of `threshold`
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    let mut any = false;
    for i0 in 0..dims[0] {
        for i1 in 0..dims[1] {
            let base = (i0 * dims[1] + i1) * dims[2];
            for i2 in 0..dims[2] {
                if logits[base + i2] > t {
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
        return None;
    }
    let mut out_lo = [0usize; 3];
    let mut out_hi = [0usize; 3];
    for (a, (ol, oh)) in out_lo.iter_mut().zip(out_hi.iter_mut()).enumerate() {
        let pad = ROI[a].saturating_sub(hi[a] - lo[a]) / 2;
        *ol = lo[a].saturating_sub(pad);
        *oh = (hi[a] + pad).min(dims[a]);
    }
    Some((out_lo, out_hi))
}

/// Scale a box from one grid to another.
fn rescale_box(
    lo: [usize; 3],
    hi: [usize; 3],
    from: [usize; 3],
    to: [usize; 3],
) -> ([usize; 3], [usize; 3]) {
    let mut l = [0usize; 3];
    let mut h = [0usize; 3];
    for (a, (lv, hv)) in l.iter_mut().zip(h.iter_mut()).enumerate() {
        let s = to[a] as f64 / from[a] as f64;
        *lv = ((lo[a] as f64 * s).floor() as usize).min(to[a].saturating_sub(1));
        *hv = ((hi[a] as f64 * s).ceil() as usize).clamp(*lv + 1, to[a]);
    }
    (l, h)
}

/// Map a box given in the prepared volume's grid into a sub-grid.
fn box_to_grid(b: &BBox, from: [usize; 3], to: [usize; 3]) -> BBox {
    let mut out = [0f32; 6];
    for (a, (t, f)) in to.iter().zip(from.iter()).enumerate() {
        let s = *t as f32 / *f as f32;
        let cap = *t as f32 - 1.0;
        out[a] = (b[a] * s).clamp(0.0, cap);
        out[a + 3] = (b[a + 3] * s).clamp(0.0, cap);
    }
    out
}

/// Run one window and return the logits at `ROI` resolution.
fn run_window(
    net: &SegVolNet,
    window: &[f32],
    points: &[Point],
    boxes: &[BBox],
    text: Option<&[f32]>,
) -> Vec<f32> {
    let decoded = net.forward(window, points, boxes, text);
    let best = decoded.best();
    // The decoder produces MASK_SHAPE; the reference lifts it to ROI with a
    // trilinear interpolation before anything else looks at it.
    preprocess::resize_trilinear(&best.data, MASK_SHAPE, ROI)
}

/// Segment a prepared volume.
///
/// `points` and `boxes` are given in the prepared volume's own index space,
/// in `[S, A, R]` order.
pub fn segment(
    net: &SegVolNet,
    prep: &Prepared,
    points: &[Point],
    boxes: &[BBox],
    text: Option<&[f32]>,
    cfg: Config,
    hooks: &dyn ProgressSink,
) -> Result<Segmentation> {
    if points.is_empty() && boxes.is_empty() && text.is_none() {
        bail!("a prompt is required: a box, at least one point, or a text embedding");
    }
    let dims = prep.dims;

    // ---- decide the region to refine ------------------------------------
    let mut coarse_ran = false;
    let (lo, hi) = if cfg.skip_coarse_with_box && !boxes.is_empty() {
        // Straight to the user's box, padded to at least one window.
        let b = &boxes[0];
        let lo = [b[0] as usize, b[1] as usize, b[2] as usize];
        let hi = [
            (b[3] as usize + 1).min(dims[0]),
            (b[4] as usize + 1).min(dims[1]),
            (b[5] as usize + 1).min(dims[2]),
        ];
        let mut l = [0usize; 3];
        let mut h = [0usize; 3];
        for (a, (lv, hv)) in l.iter_mut().zip(h.iter_mut()).enumerate() {
            let pad = ROI[a].saturating_sub(hi[a] - lo[a]) / 2;
            *lv = lo[a].saturating_sub(pad);
            *hv = (hi[a] + pad).min(dims[a]);
        }
        (l, h)
    } else {
        hooks.report(0.0, "Locating the structure…");
        coarse_ran = true;
        let small = preprocess::resize_nearest_exact(&prep.data, dims, ROI);
        let coarse_boxes: Vec<BBox> = boxes.iter().map(|b| box_to_grid(b, dims, ROI)).collect();
        let coarse_points: Vec<Point> = points
            .iter()
            .map(|p| {
                let mut q = *p;
                for (a, c) in q.coord.iter_mut().enumerate() {
                    *c = p.coord[a] * ROI[a] as f32 / dims[a] as f32;
                }
                q
            })
            .collect();
        let logits = run_window(net, &small, &coarse_points, &coarse_boxes, text);
        if hooks.cancelled() {
            bail!(CANCELLED);
        }
        if !cfg.use_zoom_in {
            let mask: Vec<u8> = logits
                .iter()
                .map(|v| (sigmoid(*v) > cfg.threshold) as u8)
                .collect();
            let full = upsample_mask(&mask, ROI, dims);
            let voxels = full.iter().filter(|v| **v != 0).count() as u64;
            return Ok(Segmentation {
                mask: full,
                voxels,
                windows: 0,
                coarse: true,
            });
        }
        match roi_from_logits(&logits, ROI, cfg.threshold) {
            Some((l, h)) => rescale_box(l, h, ROI, dims),
            None => bail!("no foreground found: try a different prompt"),
        }
    };

    // ---- refine -----------------------------------------------------------
    let crop_dims = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let cropped = preprocess::crop(&prep.data, dims, lo, crop_dims);
    let padded_dims = [
        crop_dims[0].max(ROI[0]),
        crop_dims[1].max(ROI[1]),
        crop_dims[2].max(ROI[2]),
    ];
    let padded = if padded_dims == crop_dims {
        cropped
    } else {
        let mut p = vec![0f32; padded_dims[0] * padded_dims[1] * padded_dims[2]];
        for i0 in 0..crop_dims[0] {
            for i1 in 0..crop_dims[1] {
                let src = (i0 * crop_dims[1] + i1) * crop_dims[2];
                let dst = (i0 * padded_dims[1] + i1) * padded_dims[2];
                p[dst..dst + crop_dims[2]].copy_from_slice(&cropped[src..src + crop_dims[2]]);
            }
        }
        p
    };

    let windows = window_grid(padded_dims, cfg.overlap);
    let n = padded_dims[0] * padded_dims[1] * padded_dims[2];
    let mut acc = vec![0f32; n];
    let mut count = vec![0u16; n];
    let boxes_local: Vec<BBox> = boxes
        .iter()
        .map(|b| {
            let mut q = *b;
            for (a, l) in lo.iter().enumerate() {
                q[a] = (b[a] - *l as f32).max(0.0);
                q[a + 3] = (b[a + 3] - *l as f32).max(0.0);
            }
            q
        })
        .collect();
    let points_local: Vec<Point> = points
        .iter()
        .map(|p| {
            let mut q = *p;
            for (c, l) in q.coord.iter_mut().zip(lo.iter()) {
                *c -= *l as f32;
            }
            q
        })
        .collect();

    for (wi, w) in windows.iter().enumerate() {
        if hooks.cancelled() {
            bail!(CANCELLED);
        }
        hooks.report(
            wi as f32 / windows.len() as f32,
            &format!("Refining: window {}/{}", wi + 1, windows.len()),
        );
        let win = preprocess::crop(&padded, padded_dims, *w, ROI);
        // Prompts are expressed relative to this window.
        let wb: Vec<BBox> = boxes_local
            .iter()
            .map(|b| {
                let mut q = *b;
                for (a, wa) in w.iter().enumerate() {
                    let cap = ROI[a] as f32 - 1.0;
                    q[a] = (b[a] - *wa as f32).clamp(0.0, cap);
                    q[a + 3] = (b[a + 3] - *wa as f32).clamp(0.0, cap);
                }
                q
            })
            .collect();
        let wp: Vec<Point> = points_local
            .iter()
            .filter(|p| {
                (0..3).all(|a| p.coord[a] >= w[a] as f32 && p.coord[a] < (w[a] + ROI[a]) as f32)
            })
            .map(|p| {
                let mut q = *p;
                for (c, wa) in q.coord.iter_mut().zip(w.iter()) {
                    *c -= *wa as f32;
                }
                q
            })
            .collect();
        let logits = run_window(net, &win, &wp, &wb, text);
        // Uniform (constant-mode) blending: a plain average over how many
        // windows covered each voxel. Not the Gaussian weighting nnU-Net uses.
        for i0 in 0..ROI[0] {
            for i1 in 0..ROI[1] {
                let src = (i0 * ROI[1] + i1) * ROI[2];
                let dst = ((w[0] + i0) * padded_dims[1] + w[1] + i1) * padded_dims[2] + w[2];
                for i2 in 0..ROI[2] {
                    acc[dst + i2] += logits[src + i2];
                    count[dst + i2] += 1;
                }
            }
        }
    }

    hooks.report(1.0, "Assembling the mask…");
    let mut mask = vec![0u8; crop_dims[0] * crop_dims[1] * crop_dims[2]];
    mask.par_chunks_mut(crop_dims[1] * crop_dims[2])
        .enumerate()
        .for_each(|(i0, slab)| {
            for i1 in 0..crop_dims[1] {
                let src = (i0 * padded_dims[1] + i1) * padded_dims[2];
                for i2 in 0..crop_dims[2] {
                    let c = count[src + i2].max(1) as f32;
                    slab[i1 * crop_dims[2] + i2] =
                        (sigmoid(acc[src + i2] / c) > cfg.threshold) as u8;
                }
            }
        });

    // paste the crop back into the prepared grid
    let mut full = vec![0u8; dims[0] * dims[1] * dims[2]];
    for i0 in 0..crop_dims[0] {
        for i1 in 0..crop_dims[1] {
            let src = (i0 * crop_dims[1] + i1) * crop_dims[2];
            let dst = ((lo[0] + i0) * dims[1] + lo[1] + i1) * dims[2] + lo[2];
            full[dst..dst + crop_dims[2]].copy_from_slice(&mask[src..src + crop_dims[2]]);
        }
    }
    let voxels = full.iter().filter(|v| **v != 0).count() as u64;
    Ok(Segmentation {
        mask: full,
        voxels,
        windows: windows.len(),
        coarse: coarse_ran,
    })
}

/// Nearest-neighbour upsample of a byte mask.
fn upsample_mask(mask: &[u8], src: [usize; 3], dst: [usize; 3]) -> Vec<u8> {
    let f: Vec<f32> = mask.iter().map(|v| *v as f32).collect();
    preprocess::resize_nearest_exact(&f, src, dst)
        .into_iter()
        .map(|v| (v > 0.5) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_starts_follow_monais_rule() {
        // A single window when the image is no larger than the patch.
        assert_eq!(window_starts(32, 32, 16), vec![0]);
        assert_eq!(window_starts(20, 32, 16), vec![0]);
        // 64 long, 32 patch, 16 interval: ceil(64/16) = 4 candidate starts at
        // 0,16,32,48, with the last pulled back to 32 and deduplicated.
        assert_eq!(window_starts(64, 32, 16), vec![0, 16, 32]);
        // 100 long: starts 0,16,32,48,64,80,96 -> 96 pulled back to 68
        let s = window_starts(100, 32, 16);
        assert_eq!(*s.first().unwrap(), 0);
        assert!(s.last().unwrap() + 32 <= 100);
        assert!(s.windows(2).all(|w| w[1] > w[0]), "starts must increase");
    }

    #[test]
    fn every_voxel_is_covered_by_at_least_one_window() {
        for dims in [[32, 256, 256], [40, 300, 260], [64, 512, 512]] {
            let ws = window_grid(dims, 0.5);
            let mut covered = vec![false; dims[0]];
            for w in &ws {
                covered[w[0]..w[0] + ROI[0]].fill(true);
            }
            assert!(covered.into_iter().all(|c| c), "axis 0 gap for {dims:?}");
            // and no window runs off the end
            for w in &ws {
                for (a, wa) in w.iter().enumerate() {
                    assert!(wa + ROI[a] <= dims[a], "{w:?} overruns {dims:?}");
                }
            }
        }
    }

    #[test]
    fn the_nominal_stride_is_16_128_128_at_half_overlap() {
        let ws = window_grid([64, 512, 512], 0.5);
        let a0: Vec<usize> = {
            let mut v: Vec<usize> = ws.iter().map(|w| w[0]).collect();
            v.dedup();
            v
        };
        assert_eq!(a0[1] - a0[0], 16);
        let a1: Vec<usize> = {
            let mut v: Vec<usize> = ws.iter().map(|w| w[1]).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        assert_eq!(a1[1] - a1[0], 128);
    }

    #[test]
    fn the_roi_is_padded_to_at_least_one_window() {
        // A single positive voxel must still yield a window-sized region.
        let mut logits = vec![-10.0f32; ROI[0] * ROI[1] * ROI[2]];
        logits[(4 * ROI[1] + 100) * ROI[2] + 100] = 10.0;
        let (lo, hi) = roi_from_logits(&logits, ROI, 0.5).unwrap();
        for a in 0..3 {
            assert!(hi[a] > lo[a]);
            assert!(hi[a] <= ROI[a]);
        }
        // nothing above threshold is reported as such
        assert!(roi_from_logits(&vec![-10.0; ROI[0] * ROI[1] * ROI[2]], ROI, 0.5).is_none());
    }

    #[test]
    fn boxes_rescale_between_grids() {
        let b: BBox = [0.0, 0.0, 0.0, 10.0, 20.0, 30.0];
        let out = box_to_grid(&b, [20, 40, 60], [10, 20, 30]);
        assert_eq!(out, [0.0, 0.0, 0.0, 5.0, 10.0, 15.0]);
        // clamped to the target grid
        let out = box_to_grid(
            &[0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
            [20, 40, 60],
            [10, 20, 30],
        );
        assert_eq!(out, [0.0, 0.0, 0.0, 9.0, 19.0, 29.0]);
    }
}
