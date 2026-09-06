//! The live-wire tool: the [`crate::livewire`] cost machinery, wired to the
//! views.
//!
//! The interaction is the polygon tool's, with one difference: between two
//! clicks the curve is not a straight line but the cheapest path along the
//! image edge. Clicking anchors what is on screen; the next segment starts
//! from there.
//!
//! What is cached here is what is expensive: the slice's cost image, built
//! once per slice and window, and the shortest-path tree of the current
//! anchor, built once per click. Following the pointer is then a walk up
//! parent pointers, which is free enough to do every frame.

use crate::livewire::{Costs, Tree};
use crate::volume::ViewPlane;

use super::*;

/// Everything the tool keeps between frames.
pub(super) struct WireState {
    pub slot: usize,
    pub plane: ViewPlane,
    pub slice: usize,
    /// The display window the costs were built from; a changed window means
    /// a different picture, and the tool follows what is on screen.
    pub window: (f32, f32),
    pub costs: Costs,
    /// The tree of the current anchor, and the anchor itself.
    pub tree: Option<Tree>,
    pub anchor: Option<[usize; 2]>,
    /// Learn from every accepted segment.
    pub training: bool,
}

impl ViewerApp {
    /// Build (or reuse) the cost image for this slice. `false` when there
    /// is nothing to build it from.
    fn ensure_wire(&mut self, slot: usize, plane: ViewPlane, slice: usize) -> bool {
        let window = (self.window_center, self.window_width);
        let fresh = self.wire.as_ref().is_some_and(|w| {
            w.slot == slot && w.plane == plane && w.slice == slice && w.window == window
        });
        if fresh {
            return true;
        }
        let Some(study) = self.slots[slot].study.as_ref() else {
            return false;
        };
        if !study.has_volume() {
            return false;
        }
        let vol = &study.volume;
        let [w, h] = vol.plane_dims(plane);
        if w < 3 || h < 3 || slice >= vol.plane_slice_count(plane) {
            return false;
        }
        let mut buf: Vec<i16> = Vec::new();
        vol.extract_slice(plane, slice, &mut buf);
        let img: Vec<f32> = buf.iter().map(|&v| v as f32).collect();
        let costs = Costs::new(&img, w, h, window);
        // A new slice is a new curve: the old anchor pointed at pixels of a
        // picture that is no longer on screen.
        let training = self.wire.as_ref().is_some_and(|s| s.training);
        self.wire = Some(WireState {
            slot,
            plane,
            slice,
            window,
            costs,
            tree: None,
            anchor: None,
            training,
        });
        true
    }

    /// Pixel coordinates of a fractional voxel position in this plane.
    fn wire_pixel(&self, slot: usize, plane: ViewPlane, v: [f64; 3]) -> Option<[usize; 2]> {
        let vol = &self.slots[slot].study.as_ref()?.volume;
        let [w, h] = vol.plane_dims(plane);
        let p = vol.voxel_to_plane_pixel(plane, v);
        let (x, y) = (p[0].round(), p[1].round());
        (x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h)
            .then_some([x as usize, y as usize])
    }

    /// A pixel chain back in fractional voxel coordinates.
    fn wire_voxels(
        &self,
        slot: usize,
        plane: ViewPlane,
        slice: usize,
        path: &[[usize; 2]],
    ) -> Vec<[f64; 3]> {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return Vec::new();
        };
        let vol = &study.volume;
        path.iter()
            .map(|p| vol.plane_pixel_to_voxel(plane, slice, p[0] as f64, p[1] as f64))
            .collect()
    }

    /// One click of the live-wire: accept the segment from the last anchor,
    /// and start the next one here.
    pub(super) fn livewire_click(
        &mut self,
        slot: usize,
        plane: ViewPlane,
        slice: usize,
        v: [f64; 3],
    ) {
        if !self.ensure_wire(slot, plane, slice) {
            return;
        }
        let Some(px) = self.wire_pixel(slot, plane, v) else {
            return;
        };
        // The accepted segment, if there was an anchor to run from.
        let accepted: Vec<[usize; 2]> = self
            .wire
            .as_ref()
            .and_then(|w| w.tree.as_ref().map(|t| t.path_to(px)))
            .unwrap_or_default();
        // A drawing state that belongs to another slice starts again: the
        // same rule the polygon tool follows.
        let restart = !matches!(&self.draw, Some(d) if d.slot == slot && d.plane == plane && d.slice == slice);
        if restart {
            self.draw = Some(DrawState {
                slot,
                plane,
                slice,
                pts: Vec::new(),
            });
        }
        let pts = if accepted.len() >= 2 {
            // The anchor itself is already in the chain.
            self.wire_voxels(slot, plane, slice, &accepted[1..])
        } else {
            vec![v]
        };
        if let Some(d) = &mut self.draw {
            d.pts.extend(pts);
        }
        if let Some(w) = &mut self.wire {
            if w.training && accepted.len() >= 2 {
                w.costs.train(&accepted);
            }
            w.anchor = Some(px);
            w.tree = Some(w.costs.tree(px));
        }
    }

    /// The path the pointer is currently proposing, in fractional voxel
    /// coordinates: what the view draws as the rubber band.
    pub(super) fn livewire_preview(
        &self,
        slot: usize,
        plane: ViewPlane,
        slice: usize,
        v: [f64; 3],
    ) -> Option<Vec<[f64; 3]>> {
        let w = self.wire.as_ref()?;
        if w.slot != slot || w.plane != plane || w.slice != slice {
            return None;
        }
        let tree = w.tree.as_ref()?;
        let px = self.wire_pixel(slot, plane, v)?;
        let path = tree.path_to(px);
        (path.len() >= 2).then(|| self.wire_voxels(slot, plane, slice, &path))
    }

    /// Forget the anchor (not the training): a committed or abandoned
    /// contour starts the next one from scratch.
    pub(super) fn livewire_reset(&mut self) {
        if let Some(w) = &mut self.wire {
            w.anchor = None;
            w.tree = None;
        }
    }

    /// Whether the tool has learned anything, for the toolbar's tick.
    pub(super) fn livewire_trained(&self) -> bool {
        self.wire.as_ref().is_some_and(|w| w.costs.is_trained())
    }

    pub(super) fn livewire_untrain(&mut self) {
        if let Some(w) = &mut self.wire {
            w.costs.untrain();
        }
    }
}
