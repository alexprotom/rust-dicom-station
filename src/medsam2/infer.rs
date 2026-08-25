//! Propagating one prompt through a stack of slices.
//!
//! The reference does this as two independent passes: prompt the chosen
//! slice, track to the end of the volume, throw the memory away, prompt the
//! same slice again, track to the beginning, and OR the two results. Nothing
//! here departs from that except by choice:
//!
//! * **the range is bounded** ([`Config::max_slices`]). The reference always
//!   runs to both ends of the volume, which on a 300-slice CT means 300
//!   sequential steps for a lesion that spans twenty. Drift makes the far end
//!   worthless anyway, so the default is a slab around the prompt.
//! * **the largest-component cleanup is per segmentation**, not per volume.
//!   The reference accumulates every lesion of a study into one array and
//!   then keeps the largest connected component of the union, which silently
//!   deletes all but one lesion.
//!
//! One thing that is *not* a choice: hole filling. MedSAM2 enables
//! `fill_hole_area = 8`, but its implementation is a CUDA extension and falls
//! back to a no-op on CPU — so the reference itself does not fill holes
//! unless it is running on a GPU, and neither does this.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::model::{Medsam2, SliceFeatures};
use super::ops;
use super::track::{Prompt, SliceOutput, Tracker};

/// How to propagate.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// How many slices to track on each side of the prompt. `None` runs to
    /// both ends of the volume, as the reference does.
    pub max_slices: Option<usize>,
    /// Explicit inclusive bounds in stack indices, which win over
    /// `max_slices` when set. The prompted slice is always included.
    pub range: Option<(usize, usize)>,
    /// Track towards lower indices as well as higher ones.
    pub reverse_pass: bool,
    /// Logit threshold. The reference uses 0, which is the probability 0.5
    /// the network was trained against.
    pub threshold: f32,
    /// Keep only the largest 26-connected component.
    pub largest_component: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            max_slices: Some(64),
            range: None,
            reverse_pass: true,
            threshold: 0.0,
            largest_component: true,
        }
    }
}

/// The stack being segmented, already prepared for the network.
pub trait Slices<B: Backend> {
    /// Number of slices.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One slice as the network wants it: `[1, 3, 512, 512]`, windowed and
    /// normalized.
    fn slice(&self, index: usize) -> Tensor<B, 4>;

    /// In-plane size the masks are wanted at — the slice's own size, so the
    /// result lands on the study's grid rather than the network's.
    fn out_size(&self) -> [usize; 2];
}

/// Progress reporting and cancellation.
pub trait Hooks: Sync {
    fn report(&self, _frac: f32, _msg: &str) {}
    fn cancelled(&self) -> bool {
        false
    }
}

pub struct Quiet;
impl Hooks for Quiet {}

/// One propagated structure.
pub struct Segmentation {
    /// One byte per pixel per slice, `slices` long, each `size` in extent.
    /// Slices outside the propagated range are empty.
    pub masks: Vec<Vec<u8>>,
    pub size: [usize; 2],
    /// Slices the network actually ran on.
    pub slices_visited: usize,
    pub voxels: u64,
}

impl Segmentation {
    /// Index of the first and last non-empty slice, if any.
    pub fn extent(&self) -> Option<(usize, usize)> {
        let first = self.masks.iter().position(|m| m.iter().any(|v| *v != 0))?;
        let last = self.masks.iter().rposition(|m| m.iter().any(|v| *v != 0))?;
        Some((first, last))
    }
}

/// Run one prompt through the stack.
pub fn propagate<B: Backend>(
    model: &Medsam2<B>,
    slices: &dyn Slices<B>,
    prompt_slice: usize,
    prompt: &Prompt<B>,
    config: &Config,
    hooks: &dyn Hooks,
) -> Result<Segmentation> {
    if prompt_slice >= slices.len() {
        bail!(
            "slice {prompt_slice} is outside a stack of {}",
            slices.len()
        );
    }
    hooks.report(0.0, "Encoding the prompted slice");
    let anchor = model.encode_slice(slices.slice(prompt_slice));
    propagate_from(model, slices, prompt_slice, &anchor, prompt, config, hooks)
}

/// The same, with the prompted slice already encoded — which is what makes
/// re-running an adjusted prompt on the same slice cheap.
pub fn propagate_from<B: Backend>(
    model: &Medsam2<B>,
    slices: &dyn Slices<B>,
    prompt_slice: usize,
    anchor: &SliceFeatures<B>,
    prompt: &Prompt<B>,
    config: &Config,
    hooks: &dyn Hooks,
) -> Result<Segmentation> {
    let n = slices.len();
    if n == 0 {
        bail!("the stack is empty");
    }
    if prompt_slice >= n {
        bail!("slice {prompt_slice} is outside a stack of {n}");
    }
    let (first, last) = match config.range {
        // Explicit bounds, as the Slicer extension's start/end slices: the
        // prompted slice is always inside them.
        Some((a, b)) => (
            if config.reverse_pass {
                a.min(prompt_slice)
            } else {
                prompt_slice
            },
            b.max(prompt_slice).min(n - 1),
        ),
        None => {
            let reach = config.max_slices.unwrap_or(n);
            (
                if config.reverse_pass {
                    prompt_slice.saturating_sub(reach)
                } else {
                    prompt_slice
                },
                (prompt_slice + reach).min(n - 1),
            )
        }
    };
    let total = last - first + 1;
    let size = slices.out_size();
    let mut masks: Vec<Vec<u8>> = (0..n).map(|_| Vec::new()).collect();
    let mut visited = 0usize;

    let store = |index: usize, out: &SliceOutput<B>, masks: &mut Vec<Vec<u8>>| {
        masks[index] = threshold_mask(out.low_res_masks.clone(), size, config.threshold);
    };

    // ---- the prompted slice ------------------------------------------------
    let mut forward = Tracker::new(model, n);
    let out = forward.prompt(prompt_slice, anchor, prompt);
    store(prompt_slice, &out, &mut masks);
    visited += 1;

    // ---- forwards ----------------------------------------------------------
    for index in prompt_slice + 1..=last {
        if hooks.cancelled() {
            bail!("cancelled");
        }
        hooks.report(
            visited as f32 / total as f32,
            &format!("Slice {index} of {n}"),
        );
        let feats = model.encode_slice(slices.slice(index));
        let out = forward.track(index, &feats, false);
        store(index, &out, &mut masks);
        visited += 1;
    }
    drop(forward);

    // ---- backwards, from a fresh memory ------------------------------------
    if config.reverse_pass && first < prompt_slice {
        let mut reverse = Tracker::new(model, n);
        // The same prompt on the same slice: the reference re-prompts after
        // resetting rather than reusing the forward pass's memory.
        reverse.prompt(prompt_slice, anchor, prompt);
        for index in (first..prompt_slice).rev() {
            if hooks.cancelled() {
                bail!("cancelled");
            }
            hooks.report(
                visited as f32 / total as f32,
                &format!("Slice {index} of {n}"),
            );
            let feats = model.encode_slice(slices.slice(index));
            let out = reverse.track(index, &feats, true);
            store(index, &out, &mut masks);
            visited += 1;
        }
    }

    if config.largest_component {
        hooks.report(1.0, "Cleaning up");
        keep_largest_component(&mut masks, size);
    }
    let voxels = masks
        .iter()
        .map(|m| m.iter().filter(|v| **v != 0).count() as u64)
        .sum();
    Ok(Segmentation {
        masks,
        size,
        slices_visited: visited,
        voxels,
    })
}

/// Segment **only** the prompted slice.
///
/// This is the interactive half of the Slicer-style workflow: draw a box, look
/// at what the network makes of it on that one slice, adjust, and only then
/// pay for the propagation. No memory is involved, so it costs one encode —
/// and none at all when the caller already has the slice's features.
pub fn preview<B: Backend>(
    model: &Medsam2<B>,
    anchor: &SliceFeatures<B>,
    prompt: &Prompt<B>,
    size: [usize; 2],
    threshold: f32,
) -> Vec<u8> {
    // The prompted slice's own computation, with none of the bookkeeping that
    // only a later slice would need: no memory is encoded and no bank is
    // built, so this is exactly `Tracker::prompt` minus the parts a
    // single-slice answer never reads.
    let pix_feat = model.without_memory(anchor);
    let low_res = match prompt {
        Prompt::Points(points) => {
            let multimask = super::sam::SamHead::<B>::use_multimask(points.len());
            model
                .head
                .forward(pix_feat, &anchor.high_res, points, None, multimask)
                .low_res_masks
        }
        Prompt::Mask(mask) => {
            model
                .mask_as_output(&pix_feat, anchor, mask.clone())
                .low_res_masks
        }
    };
    threshold_mask(low_res, size, threshold)
}

/// Resize the network's `128 x 128` logits onto the slice's own grid and cut
/// them at `threshold`.
///
/// This is the reference's `_get_orig_video_res_output`: the low-resolution
/// logits go to the output size in **one** bilinear step, not via 512.
pub fn threshold_mask<B: Backend>(
    low_res: Tensor<B, 4>,
    size: [usize; 2],
    threshold: f32,
) -> Vec<u8> {
    let logits = ops::to_vec(ops::resize_bilinear(low_res, size));
    logits
        .into_iter()
        .map(|v| u8::from(v > threshold))
        .collect()
}

/// Keep only the largest 26-connected component, in place.
///
/// `skimage.measure.label` defaults to full connectivity, which in 3-D means
/// 26 neighbours, so that is what the reference's `getLargestCC` uses.
pub fn keep_largest_component(masks: &mut [Vec<u8>], size: [usize; 2]) {
    let [h, w] = size;
    let plane = h * w;
    let depth = masks.len();
    let mut label = vec![0u32; depth * plane];
    let mut sizes: Vec<u32> = vec![0];
    let at = |z: usize, y: usize, x: usize| z * plane + y * w + x;
    let filled = |masks: &[Vec<u8>], z: usize, y: usize, x: usize| {
        !masks[z].is_empty() && masks[z][y * w + x] != 0
    };

    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    for z in 0..depth {
        if masks[z].is_empty() {
            continue;
        }
        for y in 0..h {
            for x in 0..w {
                if !filled(masks, z, y, x) || label[at(z, y, x)] != 0 {
                    continue;
                }
                let current = sizes.len() as u32;
                sizes.push(0);
                label[at(z, y, x)] = current;
                stack.push((z, y, x));
                while let Some((cz, cy, cx)) = stack.pop() {
                    sizes[current as usize] += 1;
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                if dz == 0 && dy == 0 && dx == 0 {
                                    continue;
                                }
                                let (nz, ny, nx) = (
                                    cz as i32 + dz,
                                    cy as i32 + dy,
                                    cx as i32 + dx,
                                );
                                if nz < 0
                                    || ny < 0
                                    || nx < 0
                                    || nz >= depth as i32
                                    || ny >= h as i32
                                    || nx >= w as i32
                                {
                                    continue;
                                }
                                let (nz, ny, nx) = (nz as usize, ny as usize, nx as usize);
                                if filled(masks, nz, ny, nx) && label[at(nz, ny, nx)] == 0 {
                                    label[at(nz, ny, nx)] = current;
                                    stack.push((nz, ny, nx));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let Some(best) = (1..sizes.len())
        .max_by_key(|i| sizes[*i])
        .map(|i| i as u32)
    else {
        return;
    };
    for z in 0..depth {
        if masks[z].is_empty() {
            continue;
        }
        for i in 0..plane {
            if label[z * plane + i] != best {
                masks[z][i] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(depth: usize, size: [usize; 2], set: &[(usize, usize, usize)]) -> Vec<Vec<u8>> {
        let mut m: Vec<Vec<u8>> = (0..depth).map(|_| vec![0u8; size[0] * size[1]]).collect();
        for (z, y, x) in set {
            m[*z][y * size[1] + x] = 1;
        }
        m
    }

    #[test]
    fn the_largest_component_survives_and_the_rest_do_not() {
        let size = [5, 5];
        // a three-voxel blob at the front, a single voxel at the back
        let mut m = volume(
            3,
            size,
            &[(0, 1, 1), (0, 1, 2), (1, 1, 1), (2, 4, 4)],
        );
        keep_largest_component(&mut m, size);
        assert_eq!(m[0].iter().filter(|v| **v != 0).count(), 2);
        assert_eq!(m[1].iter().filter(|v| **v != 0).count(), 1);
        assert_eq!(m[2].iter().filter(|v| **v != 0).count(), 0);
    }

    #[test]
    fn connectivity_is_the_full_twenty_six() {
        let size = [3, 3];
        // two voxels touching only at a corner across slices
        let mut m = volume(2, size, &[(0, 0, 0), (1, 1, 1)]);
        keep_largest_component(&mut m, size);
        assert_eq!(
            m.iter().map(|s| s.iter().filter(|v| **v != 0).count()).sum::<usize>(),
            2,
            "a diagonal neighbour is still connected"
        );
    }

    #[test]
    fn an_empty_volume_is_left_alone() {
        let size = [4, 4];
        let mut m = volume(2, size, &[]);
        keep_largest_component(&mut m, size);
        assert!(m.iter().all(|s| s.iter().all(|v| *v == 0)));
    }

    #[test]
    fn the_default_config_is_a_bounded_two_sided_run() {
        let c = Config::default();
        assert_eq!(c.max_slices, Some(64));
        assert!(c.reverse_pass && c.largest_component);
        assert_eq!(c.threshold, 0.0);
    }
}
