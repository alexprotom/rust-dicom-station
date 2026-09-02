//! The memory bank, and the state machine that walks a stack of slices.
//!
//! SAM 2 segments a volume the way it segments a video: one slice is
//! prompted, and every other slice is conditioned on what the model already
//! decided about the slices near it. What "near" means is the whole of this
//! module.
//!
//! For a slice at `i`, tracking forwards, the memory is
//!
//! * every **prompted** slice, at temporal index 6;
//! * the six tracked slices at `i-6 … i-1`, at temporal indices 0 … 5, with
//!   index 0 being `i-1` - missing ones are simply skipped;
//! * up to sixteen **object pointers**, one per slice already decided, each
//!   256-wide vector split into four 64-wide tokens that share one projected
//!   sine encoding of how far away that slice is.
//!
//! Tracking backwards mirrors all three: `i+1 … i+6`, pointers from later
//! slices, and the sign of the conditioning slices' offsets flips.
//!
//! The pointer tokens go **last** in the sequence, because the memory
//! attention excludes exactly that tail from its rotary encoding.

use std::collections::BTreeMap;

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use super::config::{self, D_MODEL, MEM_DIM};
use super::memory::Memory;
use super::model::{Medsam2, SliceFeatures};
use super::ops;
use super::prompt::Point;
use super::sam::SamHead;

/// What one decided slice contributes to later slices.
#[derive(Clone)]
pub struct SliceMemory<B: Backend> {
    /// `[1, tokens, MEM_DIM]`, flattened from the memory encoder's map.
    pub features: Tensor<B, 3>,
    /// `[1, tokens, MEM_DIM]`, the spatial sine encoding - the temporal one
    /// is added at assembly time, because it depends on the distance to the
    /// slice being tracked.
    pub pos: Tensor<B, 3>,
    /// `[1, D_MODEL]`.
    pub obj_ptr: Tensor<B, 2>,
}

impl<B: Backend> SliceMemory<B> {
    fn from_memory(memory: Memory<B>, obj_ptr: Tensor<B, 2>) -> SliceMemory<B> {
        let flat = |t: Tensor<B, 4>| {
            let [b, c, h, w] = t.dims();
            t.reshape([b, c, h * w]).swap_dims(1, 2)
        };
        SliceMemory {
            features: flat(memory.features),
            pos: flat(memory.pos),
            obj_ptr,
        }
    }
}

/// The decided slices, split by whether the user prompted them.
pub struct MemoryBank<B: Backend> {
    conditioning: BTreeMap<usize, SliceMemory<B>>,
    tracked: BTreeMap<usize, SliceMemory<B>>,
    num_frames: usize,
}

/// The memory sequence for one slice.
pub struct Conditioning<B: Backend> {
    /// `[1, entries, MEM_DIM]`.
    pub memory: Tensor<B, 3>,
    pub memory_pos: Tensor<B, 3>,
    /// How many entries at the end are object-pointer tokens.
    pub num_obj_ptr_tokens: usize,
}

impl<B: Backend> MemoryBank<B> {
    pub fn new(num_frames: usize) -> MemoryBank<B> {
        MemoryBank {
            conditioning: BTreeMap::new(),
            tracked: BTreeMap::new(),
            num_frames,
        }
    }

    pub fn insert(&mut self, index: usize, memory: SliceMemory<B>, prompted: bool) {
        if prompted {
            self.conditioning.insert(index, memory);
        } else {
            self.tracked.insert(index, memory);
        }
    }

    pub fn has_prompt(&self) -> bool {
        !self.conditioning.is_empty()
    }

    /// Which slices contribute spatial memory to `frame_idx`, as
    /// `(temporal index, slice)` pairs in the order the reference concatenates
    /// them. Split out from [`Self::assemble`] so it can be asserted directly.
    pub fn spatial_entries(&self, frame_idx: usize, reverse: bool) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = self
            .conditioning
            .keys()
            .map(|index| (config::NUM_MASKMEM - 1, *index))
            .collect();
        for t_pos in 1..config::NUM_MASKMEM {
            let t_rel = config::NUM_MASKMEM - t_pos;
            let prev = if reverse {
                frame_idx.checked_add(t_rel)
            } else {
                frame_idx.checked_sub(t_rel)
            };
            let Some(prev) = prev else { continue };
            if self.tracked.contains_key(&prev) {
                out.push((config::NUM_MASKMEM - t_pos - 1, prev));
            }
        }
        out
    }

    /// Which pointers contribute, as `(offset, slice)` pairs. The offset is
    /// what the temporal encoding is computed from: signed distance for
    /// prompted slices, positive distance for tracked ones.
    pub fn pointer_entries(&self, frame_idx: usize, reverse: bool) -> Vec<(f32, usize)> {
        let max_ptrs = self.num_frames.min(config::MAX_OBJ_PTRS);
        let sign = if reverse { -1.0 } else { 1.0 };
        let here = frame_idx as isize;
        let mut out: Vec<(f32, usize)> = self
            .conditioning
            .keys()
            .filter(|t| {
                if reverse {
                    **t >= frame_idx
                } else {
                    **t <= frame_idx
                }
            })
            .map(|t| ((here - *t as isize) as f32 * sign, *t))
            .collect();
        for t_diff in 1..max_ptrs {
            let t = if reverse {
                here + t_diff as isize
            } else {
                here - t_diff as isize
            };
            if t < 0 || t >= self.num_frames as isize {
                break;
            }
            let t = t as usize;
            if self.tracked.contains_key(&t) {
                out.push((t_diff as f32, t));
            }
        }
        out
    }

    /// Build the memory sequence for `frame_idx`.
    pub fn assemble(
        &self,
        model: &Medsam2<B>,
        frame_idx: usize,
        reverse: bool,
    ) -> Option<Conditioning<B>> {
        if !self.has_prompt() {
            return None;
        }
        let get = |index: usize| -> &SliceMemory<B> {
            self.conditioning
                .get(&index)
                .or_else(|| self.tracked.get(&index))
                .expect("entry was listed")
        };

        let mut memory = Vec::new();
        let mut memory_pos = Vec::new();
        for (t_index, slice) in self.spatial_entries(frame_idx, reverse) {
            let m = get(slice);
            memory.push(m.features.clone());
            memory_pos.push(m.pos.clone() + model.tpos_row(t_index));
        }

        let pointers = self.pointer_entries(frame_idx, reverse);
        let mut num_obj_ptr_tokens = 0;
        if !pointers.is_empty() {
            let t_diff_max = (self.num_frames.min(config::MAX_OBJ_PTRS) - 1).max(1) as f32;
            let offsets: Vec<f32> = pointers.iter().map(|(o, _)| *o).collect();
            let pos = model.pointer_pos(&offsets, t_diff_max);
            let ptrs: Vec<Tensor<B, 3>> = pointers
                .iter()
                .map(|(_, index)| get(*index).obj_ptr.clone().reshape([1, 1, D_MODEL]))
                .collect();
            let count = ptrs.len();
            // Each 256-wide pointer becomes four 64-wide tokens, contiguous,
            // and its temporal encoding is repeated across the four.
            let ptrs = Tensor::cat(ptrs, 1).reshape([1, count * config::PTR_TOKENS, MEM_DIM]);
            let pos = pos
                .reshape([1, count, 1, MEM_DIM])
                .repeat_dim(2, config::PTR_TOKENS)
                .reshape([1, count * config::PTR_TOKENS, MEM_DIM]);
            num_obj_ptr_tokens = count * config::PTR_TOKENS;
            memory.push(ptrs);
            memory_pos.push(pos);
        }

        Some(Conditioning {
            memory: Tensor::cat(memory, 1),
            memory_pos: Tensor::cat(memory_pos, 1),
            num_obj_ptr_tokens,
        })
    }
}

/// What one slice's segmentation produced.
pub struct SliceOutput<B: Backend> {
    /// `[1, 1, 128, 128]` - the network's own resolution, and what a caller
    /// should resize to the slice's real size.
    pub low_res_masks: Tensor<B, 4>,
    /// `[1, 1, 512, 512]`, as the memory encoder saw it.
    pub high_res_masks: Tensor<B, 4>,
    pub object_present: bool,
}

/// A prompt on one slice.
pub enum Prompt<B: Backend> {
    /// Clicks, or the two corners of a box.
    Points(Vec<Point>),
    /// An existing binary mask at the network's resolution, `[1, 1, 512, 512]` -
    /// a contour the user already has, propagated.
    Mask(Tensor<B, 4>),
}

/// One propagation: a bank, and the slices decided so far.
pub struct Tracker<'a, B: Backend> {
    model: &'a Medsam2<B>,
    bank: MemoryBank<B>,
}

impl<'a, B: Backend> Tracker<'a, B> {
    pub fn new(model: &'a Medsam2<B>, num_frames: usize) -> Tracker<'a, B> {
        Tracker {
            model,
            bank: MemoryBank::new(num_frames),
        }
    }

    /// Segment a prompted slice. Memory attention is skipped entirely.
    pub fn prompt(
        &mut self,
        frame_idx: usize,
        feats: &SliceFeatures<B>,
        prompt: &Prompt<B>,
    ) -> SliceOutput<B> {
        let pix_feat = self.model.without_memory(feats);
        let out = match prompt {
            Prompt::Points(points) => {
                let multimask = SamHead::<B>::use_multimask(points.len());
                self.model
                    .head
                    .forward(pix_feat, &feats.high_res, points, None, multimask)
            }
            Prompt::Mask(mask) => self.model.mask_as_output(&pix_feat, feats, mask.clone()),
        };
        self.finish(frame_idx, feats, out, true)
    }

    /// Segment a slice from memory alone.
    pub fn track(
        &mut self,
        frame_idx: usize,
        feats: &SliceFeatures<B>,
        reverse: bool,
    ) -> SliceOutput<B> {
        let pix_feat = match self.bank.assemble(self.model, frame_idx, reverse) {
            Some(c) => self
                .model
                .with_memory(feats, c.memory, c.memory_pos, c.num_obj_ptr_tokens),
            // No prompt yet: nothing to condition on.
            None => self.model.without_memory(feats),
        };
        // A slice with no prompt has zero points, which does qualify for
        // multi-mask output - the reference picks the best of three by
        // predicted IoU on every tracked slice.
        let out = self
            .model
            .head
            .forward(pix_feat, &feats.high_res, &[], None, true);
        self.finish(frame_idx, feats, out, false)
    }

    /// Encode the answer into the bank and hand it back.
    fn finish(
        &mut self,
        frame_idx: usize,
        feats: &SliceFeatures<B>,
        out: super::sam::SamOutput<B>,
        prompted: bool,
    ) -> SliceOutput<B> {
        let object_present = out.object_present();
        // The reference re-derives the 512 mask from the 128 one before
        // encoding memory, on prompted and tracked slices alike.
        let size = config::IMAGE_SIZE;
        let high_res_masks = ops::resize_bilinear(out.low_res_masks.clone(), [size, size]);
        let memory = self.model.memory_encoder.encode(
            feats.pix_feat.clone(),
            high_res_masks.clone(),
            prompted,
            object_present,
        );
        self.bank.insert(
            frame_idx,
            SliceMemory::from_memory(memory, out.obj_ptr),
            prompted,
        );
        SliceOutput {
            low_res_masks: out.low_res_masks,
            high_res_masks,
            object_present,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Bk = burn::backend::NdArray;

    fn bank(cond: &[usize], tracked: &[usize], num_frames: usize) -> MemoryBank<Bk> {
        let dev: burn::tensor::Device<Bk> = Default::default();
        let dummy = || SliceMemory::<Bk> {
            features: Tensor::zeros([1, 4, MEM_DIM], &dev),
            pos: Tensor::zeros([1, 4, MEM_DIM], &dev),
            obj_ptr: Tensor::zeros([1, D_MODEL], &dev),
        };
        let mut b = MemoryBank::new(num_frames);
        for c in cond {
            b.insert(*c, dummy(), true);
        }
        for t in tracked {
            b.insert(*t, dummy(), false);
        }
        b
    }

    #[test]
    fn the_bank_takes_the_six_nearest_tracked_slices_and_every_prompt() {
        // prompted at 1, tracked 2..9, now at 9 going forwards
        let b = bank(&[1], &[2, 3, 4, 5, 6, 7, 8], 10);
        let e = b.spatial_entries(9, false);
        // conditioning first, at temporal index 6, then 3..8 with the most
        // recent at index 0
        assert_eq!(
            e,
            vec![(6, 1), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7), (0, 8)]
        );
        assert_eq!(e.len(), config::NUM_MASKMEM);
    }

    #[test]
    fn missing_slices_are_skipped_rather_than_padded() {
        let b = bank(&[4], &[5], 10);
        // at slice 6 only 5 is available; 0..4 are not tracked
        assert_eq!(b.spatial_entries(6, false), vec![(6, 4), (0, 5)]);
    }

    #[test]
    fn tracking_backwards_looks_the_other_way() {
        let b = bank(&[8], &[5, 6, 7], 10);
        assert_eq!(
            b.spatial_entries(4, true),
            vec![(6, 8), (2, 7), (1, 6), (0, 5)]
        );
        // and forwards from the same bank sees nothing but the prompt
        assert_eq!(b.spatial_entries(4, false), vec![(6, 8)]);
    }

    #[test]
    fn pointers_run_from_the_prompt_and_the_nearest_tracked_slices() {
        let b = bank(&[1], &[2, 3, 4], 10);
        let p = b.pointer_entries(5, false);
        // the prompt at distance 4, then 4, 3, 2 at distances 1, 2, 3
        assert_eq!(p, vec![(4.0, 1), (1.0, 4), (2.0, 3), (3.0, 2)]);
    }

    #[test]
    fn a_prompt_ahead_of_the_slice_is_dropped_when_tracking_forwards() {
        let b = bank(&[8], &[6, 7], 10);
        // forwards from 5: the prompt at 8 is in the future, so no pointer
        assert_eq!(b.pointer_entries(5, false), vec![]);
        // backwards it counts, and the sign flip makes its offset positive
        // again - `use_signed_tpos_enc_to_obj_ptrs` measures distance in the
        // direction of travel, not in slice order.
        assert_eq!(
            b.pointer_entries(5, true),
            vec![(3.0, 8), (1.0, 6), (2.0, 7)]
        );
    }

    #[test]
    fn the_pointer_scan_stops_at_the_first_gap_in_the_volume() {
        let b = bank(&[0], &[1, 2, 3], 4);
        // at slice 3, `max_ptrs` is min(4, 16) = 4, so t_diff runs 1..3
        let p = b.pointer_entries(3, false);
        assert_eq!(p, vec![(3.0, 0), (1.0, 2), (2.0, 1)]);
    }
}
