//! The contour tools: drawing and editing RT structures as contours.
//!
//! The brush and the region grower edit voxels; everything here edits the
//! polygons an RTSTRUCT actually stores, through [`crate::contours`]. The
//! difference is what a planner sees on a 3 mm CT: a drawn curve that stays
//! the curve it was drawn as, instead of the outline of the voxels it
//! happened to cover.
//!
//! One ROI at a time is "the ROI the tools edit" (`StudySlot::active_roi`),
//! picked in the structure list, by clicking a contour with a drawing tool
//! held, or created on the spot by drawing when there is none.

use super::*;
use crate::contours::{self, Poly, Stack};
use crate::rtstruct::{Roi, StructureSet};
use crate::volume::ViewPlane;

/// How much contour undo is kept, per the same reasoning as the brush's
/// stroke journal: enough to rescue a mistake, not enough to notice.
const ROI_UNDO_DEPTH: usize = 32;

/// The working geometry of the ROI under the contour tools.
///
/// It exists for one reason: a structure is *stored* as axial contours, so
/// drawing in a sagittal or coronal view would otherwise convert the whole
/// geometry there and back on every stroke, and each round trip through the
/// lattice would cost a little more of the shape. With the working stack
/// cached, the conversion happens once on the way in, and every commit
/// converts the same source afresh.
pub(super) struct EditStack {
    pub slot: usize,
    pub set: usize,
    pub roi: usize,
    pub stack: Stack,
    /// `settings_gen` when the cache was filled: anything else that touches
    /// structures bumps it, which is exactly when this must be re-read.
    pub gen: u64,
}

/// The in-plane coordinates of a voxel point, for a stack sliced on `axis`.
fn uv(axis: usize, v: [f64; 3]) -> [f64; 2] {
    let [a, b] = contours::plane_axes(axis);
    [v[a], v[b]]
}

/// A closed centripetal Catmull-Rom spline through the given points - the
/// curve the spline tool draws, and the only reason it differs from the
/// polygon tool.
fn spline_ring(pts: &[[f64; 2]], per_segment: usize) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(n * per_segment);
    for i in 0..n {
        let p0 = pts[(i + n - 1) % n];
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let p3 = pts[(i + 2) % n];
        // Centripetal parameterisation: no cusps, no self-intersection from
        // a doubled click.
        let t = |a: [f64; 2], b: [f64; 2], t0: f64| -> f64 {
            let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            t0 + d.sqrt().max(1e-6)
        };
        let (t0, t1) = (0.0, t(p0, p1, 0.0));
        let (t2, t3) = (t(p1, p2, t1), 0.0);
        let t3 = t(p2, p3, t2).max(t3);
        for s in 0..per_segment {
            let tt = t1 + (t2 - t1) * (s as f64 / per_segment as f64);
            let a1 = lerp2(p0, p1, (t1 - tt) / (t1 - t0), (tt - t0) / (t1 - t0));
            let a2 = lerp2(p1, p2, (t2 - tt) / (t2 - t1), (tt - t1) / (t2 - t1));
            let a3 = lerp2(p2, p3, (t3 - tt) / (t3 - t2), (tt - t2) / (t3 - t2));
            let b1 = lerp2(a1, a2, (t2 - tt) / (t2 - t0), (tt - t0) / (t2 - t0));
            let b2 = lerp2(a2, a3, (t3 - tt) / (t3 - t1), (tt - t1) / (t3 - t1));
            out.push(lerp2(b1, b2, (t2 - tt) / (t2 - t1), (tt - t1) / (t2 - t1)));
        }
    }
    out
}

#[inline]
fn lerp2(a: [f64; 2], b: [f64; 2], wa: f64, wb: f64) -> [f64; 2] {
    [a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb]
}

impl ViewerApp {
    // -- the ROI under the tools -----------------------------------------

    /// The (set, ROI) the contour tools edit right now, if the choice is
    /// still valid.
    pub(super) fn edit_target(&self, slot: usize) -> Option<(usize, usize)> {
        let s = &self.slots[slot];
        let ss = s.active_structures()?;
        (s.active_roi < ss.rois.len()).then_some((s.active_structs, s.active_roi))
    }

    /// The name of the ROI being edited, for the toolbar.
    pub(super) fn edit_roi_name(&self, slot: usize) -> Option<(&str, [u8; 3])> {
        let (_, roi) = self.edit_target(slot)?;
        let r = self.slots[slot].active_structures()?.rois.get(roi)?;
        Some((&r.name, r.color))
    }

    /// The (set, ROI) to draw into, creating one if nothing is chosen -
    /// drawing on a study with no structures at all should just work.
    ///
    /// It creates rather than adopts on purpose. Picking the set's first
    /// structure would be convenient exactly once and wrong every time
    /// after: a stroke must never land in a structure the user did not
    /// point at, and there is no undo for surprise.
    pub(super) fn ensure_edit_roi(&mut self, slot: usize) -> Option<(usize, usize)> {
        if let Some(t) = self.edit_target(slot) {
            return Some(t);
        }
        let roi = self.new_roi(slot, None, "ORGAN")?;
        Some((self.slots[slot].active_structs, roi))
    }

    /// Append an ROI to the slot's active structure set, creating an
    /// in-memory set when the study has none, and make it the one the tools
    /// edit. Returns its index.
    pub(super) fn new_roi(
        &mut self,
        slot: usize,
        name: Option<String>,
        roi_type: &str,
    ) -> Option<usize> {
        let vol = self.slots[slot].study.as_ref().map(|s| s.volume.clone())?;
        self.roi_counter += 1;
        let name = name.unwrap_or_else(|| format!("ROI {}", self.roi_counter));
        let color = crate::segmentation::SEG_PALETTE
            [(self.roi_counter - 1) % crate::segmentation::SEG_PALETTE.len()];
        let number = self.slots[slot]
            .active_structures()
            .map(|ss| ss.rois.iter().map(|r| r.number).max().unwrap_or(0) + 1)
            .unwrap_or(1);
        let roi = Roi {
            number,
            name,
            color,
            roi_type: roi_type.to_string(),
            contours: Vec::new(),
        };
        let StudySlot {
            study,
            active_structs,
            roi_visible,
            active_roi,
            ..
        } = &mut self.slots[slot];
        let study = study.as_mut()?;
        let idx = if let Some(ss) = study.structure_sets.get_mut(*active_structs) {
            ss.rois.push(roi);
            roi_visible.resize(ss.rois.len(), true);
            ss.rois.len() - 1
        } else {
            let active_series = study.series.get(study.active_series);
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            study.structure_sets.push(StructureSet {
                label: "Contours".into(),
                frame_of_reference_uid: vol.frame_of_reference_uid.clone(),
                sop_instance_uid: format!("2.25.{stamp}"),
                series_instance_uid: format!("2.25.{stamp}.1"),
                study_uid: active_series
                    .map(|s| s.study_uid.clone())
                    .unwrap_or_default(),
                referenced_series_uid: active_series.map(|s| s.uid.clone()).unwrap_or_default(),
                file_name: "drawn-contours".into(),
                rois: vec![roi],
            });
            *active_structs = study.structure_sets.len() - 1;
            *roi_visible = vec![true];
            0
        };
        *active_roi = idx;
        self.slots[slot].structs_shown = true;
        self.settings_gen += 1;
        self.edit = None;
        Some(idx)
    }

    /// Make the ROI whose contour the point falls on (or in) the one the
    /// tools edit. Returns whether anything was picked.
    pub(super) fn pick_roi_at(&mut self, slot: usize, plane: ViewPlane, v: [f64; 3]) -> bool {
        let axis = contours::axis_of_plane(plane);
        let level = v[axis].round();
        let Some(study) = self.slots[slot].study.as_ref() else {
            return false;
        };
        let grid = study.volume.grid();
        if level < 0.0 || level >= grid.dims[axis] as f64 {
            return false;
        }
        let Some(ss) = self.slots[slot].active_structures() else {
            return false;
        };
        let p = uv(axis, v);
        let hit = ss
            .rois
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                self.slots[slot]
                    .roi_visible
                    .get(*i)
                    .copied()
                    .unwrap_or(true)
            })
            .filter_map(|(i, roi)| {
                let st = Stack::from_roi(roi, &grid);
                let st = if st.axis == axis {
                    st
                } else {
                    st.to_axis(grid.dims, axis)
                };
                let r = st.region_at(level as usize)?;
                r.contains(p).then_some((i, r.area()))
            })
            // The smallest structure under the pointer is the one meant.
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        if let Some(i) = hit {
            self.slots[slot].active_roi = i;
            self.edit = None;
            return true;
        }
        false
    }

    // -- undo -------------------------------------------------------------

    pub(super) fn push_roi_undo(&mut self, slot: usize, set: usize, roi: usize) {
        let Some(contours) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.structure_sets.get(set))
            .and_then(|ss| ss.rois.get(roi))
            .map(|r| r.contours.clone())
        else {
            return;
        };
        self.roi_undo.push(RoiSnapshot {
            slot,
            set,
            roi,
            contours,
        });
        if self.roi_undo.len() > ROI_UNDO_DEPTH {
            self.roi_undo.remove(0);
        }
    }

    /// Undo the last contour edit of this slot.
    pub(super) fn undo_roi_edit(&mut self, slot: usize) {
        let Some(pos) = self.roi_undo.iter().rposition(|s| s.slot == slot) else {
            return;
        };
        let snap = self.roi_undo.remove(pos);
        if let Some(roi) = self.slots[snap.slot]
            .study
            .as_mut()
            .and_then(|st| st.structure_sets.get_mut(snap.set))
            .and_then(|ss| ss.rois.get_mut(snap.roi))
        {
            roi.contours = snap.contours;
        }
        self.edit = None;
        self.settings_gen += 1;
    }

    // -- the working stack ------------------------------------------------

    /// The ROI's geometry sliced along `axis`, read once and kept until
    /// something else touches structures.
    fn working_stack(&mut self, slot: usize, set: usize, roi: usize, axis: usize) -> Option<()> {
        let fresh = match &self.edit {
            Some(e) => {
                e.slot != slot
                    || e.set != set
                    || e.roi != roi
                    || e.stack.axis != axis
                    || e.gen != self.settings_gen
            }
            None => true,
        };
        if !fresh {
            return Some(());
        }
        let study = self.slots[slot].study.as_ref()?;
        let grid = study.volume.grid();
        let r = study.structure_sets.get(set)?.rois.get(roi)?;
        let st = Stack::from_roi(r, &grid);
        let src_axis = st.axis;
        let converted = src_axis != axis && !st.is_empty();
        let name = r.name.clone();
        let stack = if src_axis == axis {
            st
        } else {
            st.to_axis(grid.dims, axis)
        };
        if converted {
            self.notice = Some(format!(
                "'{name}' was drawn on {} slices. Editing it here re-cuts the geometry \
                 {}, which is what a planning system does when you change drawing \
                 plane - the shape is preserved to the lattice, its exact vertices \
                 are not.",
                contours::axis_name(src_axis),
                contours::axis_name(axis),
            ));
        }
        self.edit = Some(EditStack {
            slot,
            set,
            roi,
            stack,
            gen: self.settings_gen,
        });
        Some(())
    }

    /// Write the working stack back into the ROI (as axial contours) and
    /// keep the cache valid.
    fn flush_edit(&mut self) {
        let Some(e) = self.edit.take() else { return };
        let Some(grid) = self.slots[e.slot].study.as_ref().map(|st| st.volume.grid()) else {
            return;
        };
        if let Some(roi) = self.slots[e.slot]
            .study
            .as_mut()
            .and_then(|st| st.structure_sets.get_mut(e.set))
            .and_then(|ss| ss.rois.get_mut(e.roi))
        {
            e.stack.apply_to_roi(roi, &grid);
        }
        self.settings_gen += 1;
        self.edit = Some(EditStack {
            gen: self.settings_gen,
            ..e
        });
    }

    // -- drawing ----------------------------------------------------------

    /// Start, or continue, a drawing on this slice of this view.
    pub(super) fn draw_point(&mut self, slot: usize, plane: ViewPlane, slice: usize, v: [f64; 3]) {
        match &mut self.draw {
            Some(d) if d.slot == slot && d.plane == plane && d.slice == slice => {
                // A click on top of the last point is a request to close, not
                // a zero-length edge.
                if let Some(last) = d.pts.last() {
                    let d2 = (last[0] - v[0]).powi(2) + (last[1] - v[1]).powi(2);
                    if d2 < 1e-6 {
                        return;
                    }
                }
                d.pts.push(v);
            }
            _ => {
                self.draw = Some(DrawState {
                    slot,
                    plane,
                    slice,
                    pts: vec![v],
                });
            }
        }
    }

    /// Close the contour under construction and apply it.
    pub(super) fn commit_draw(&mut self) {
        let Some(d) = self.draw.take() else { return };
        if d.pts.len() < 3 {
            return;
        }
        let axis = contours::axis_of_plane(d.plane);
        let uvs: Vec<[f64; 2]> = d.pts.iter().map(|p| uv(axis, *p)).collect();
        let ring = match self.seg_tool {
            SegTool::Spline => Poly::new(spline_ring(&uvs, 12)),
            // A freehand drag arrives with a point per mouse event; thinning
            // it costs nothing visible and keeps the stored contour sane.
            SegTool::Freehand => Poly::new(uvs).simplify(0.08),
            _ => Poly::new(uvs),
        };
        self.apply_ring(d.slot, d.plane, d.slice, ring);
    }

    /// Apply one closed ring to the ROI being edited, in the current mode.
    pub(super) fn apply_ring(&mut self, slot: usize, plane: ViewPlane, level: usize, ring: Poly) {
        if ring.is_empty() {
            return;
        }
        let Some((set, roi)) = self.ensure_edit_roi(slot) else {
            return;
        };
        let axis = contours::axis_of_plane(plane);
        if self.working_stack(slot, set, roi, axis).is_none() {
            return;
        }
        self.push_roi_undo(slot, set, roi);
        let mode = self.draw_mode;
        let Some(e) = self.edit.as_mut() else { return };
        let region = e.stack.region_mut(level);
        let cut = match mode {
            DrawMode::Extend => false,
            DrawMode::Subtract => true,
            // RayStation's auto rule, verbatim: a drawing that crosses an
            // existing contour and starts outside it cuts; anything else adds.
            DrawMode::Auto => region.crosses(&ring) && !region.contains(ring.pts[0]),
        };
        if cut {
            region.subtract_ring(&ring);
        } else {
            region.add_ring(&ring);
        }
        e.stack.prune();
        self.flush_edit();
    }

    /// Push the contour under the pointer: every vertex within `radius_mm`
    /// moves with the drag, by a cosine falloff, so the outline deforms
    /// smoothly instead of developing a corner.
    pub(super) fn nudge_contour(
        &mut self,
        slot: usize,
        plane: ViewPlane,
        level: usize,
        from: [f64; 3],
        to: [f64; 3],
        radius_mm: f64,
    ) {
        let Some((set, roi)) = self.edit_target(slot) else {
            return;
        };
        let axis = contours::axis_of_plane(plane);
        let Some(spacing) = self.slots[slot].study.as_ref().map(|s| s.volume.spacing) else {
            return;
        };
        if self.working_stack(slot, set, roi, axis).is_none() {
            return;
        }
        let [ua, va] = contours::plane_axes(axis);
        // The radius is a distance in millimetres; the ring lives on the
        // lattice, so the geometry is told how long a unit of each axis is.
        let mm = [spacing[ua], spacing[va]];
        let (a, b) = (uv(axis, from), uv(axis, to));
        let Some(e) = self.edit.as_mut() else { return };
        let Some(region) = e
            .stack
            .slices
            .iter_mut()
            .find(|s| s.level == level)
            .map(|s| &mut s.region)
        else {
            return;
        };
        if !region.nudge(a, b, radius_mm, mm) {
            return;
        }
        self.push_roi_undo(slot, set, roi);
        self.flush_edit();
    }

    // -- what the views draw ----------------------------------------------

    /// The contour under construction, in the in-plane voxel coordinates of
    /// `plane`, plus whether it is a spline (drawn as its curve, not as the
    /// clicked chain).
    pub(super) fn draw_preview(
        &self,
        slot: usize,
        plane: ViewPlane,
        slice: usize,
    ) -> Option<Vec<[f64; 3]>> {
        let d = self.draw.as_ref()?;
        if d.slot != slot || d.plane != plane || d.slice != slice {
            return None;
        }
        if self.seg_tool == SegTool::Spline && d.pts.len() >= 3 {
            let axis = contours::axis_of_plane(plane);
            let [ua, va] = contours::plane_axes(axis);
            let uvs: Vec<[f64; 2]> = d.pts.iter().map(|p| uv(axis, *p)).collect();
            return Some(
                spline_ring(&uvs, 12)
                    .into_iter()
                    .map(|p| {
                        let mut v = [0.0f64; 3];
                        v[ua] = p[0];
                        v[va] = p[1];
                        v[axis] = slice as f64;
                        v
                    })
                    .collect(),
            );
        }
        Some(d.pts.clone())
    }
}

/// The rings of one interpolated slice in a view's pixel coordinates, and
/// the colour to draw them in.
pub(super) type InterpRings = (Vec<Vec<[f32; 2]>>, [u8; 3]);

/// The interpolated slices of the edited structure, kept while the contour
/// window shows them: they are a *preview*, and are only written into the
/// structure when accepted - the rule RayStation states and the reason the
/// preview is not simply added to the stack.
pub(super) struct InterpPreview {
    pub slot: usize,
    pub set: usize,
    pub roi: usize,
    pub axis: usize,
    pub stack: Stack,
    pub gen: u64,
}

impl ViewerApp {
    /// The lattice axis the contour tools currently work along: whatever the
    /// working stack was cut on, and the axial stack when there is none.
    pub(super) fn edit_axis(&self, slot: usize) -> usize {
        self.edit
            .as_ref()
            .filter(|e| e.slot == slot)
            .map(|e| e.stack.axis)
            .unwrap_or(2)
    }

    /// "The current slice" for every per-slice operation: the slice *shown*
    /// in the view that looks along the drawing axis, which is the one the
    /// strokes land on. The crosshair is the fallback - the wheel moves a
    /// view's slice without moving it.
    pub(super) fn edit_level(&self, slot: usize) -> usize {
        let axis = self.edit_axis(slot);
        let n = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.volume.dims[axis])
            .unwrap_or(1);
        let shown = self.slots[slot]
            .views
            .iter()
            .find(|v| contours::axis_of_plane(v.plane) == axis)
            .map(|v| v.slice);
        let level = match shown {
            Some(s) => s,
            None => self.slots[slot].cursor[axis].round().max(0.0) as usize,
        };
        level.min(n.saturating_sub(1))
    }

    /// Run `f` on the working stack of the edited ROI and write the result
    /// back, with one undo step. Returns false when there is nothing to edit.
    pub(super) fn with_edit_stack(
        &mut self,
        slot: usize,
        f: impl FnOnce(&mut Stack, [usize; 3]),
    ) -> bool {
        let Some((set, roi)) = self.edit_target(slot) else {
            return false;
        };
        let axis = self.edit_axis(slot);
        if self.working_stack(slot, set, roi, axis).is_none() {
            return false;
        }
        let Some(dims) = self.slots[slot].study.as_ref().map(|s| s.volume.dims) else {
            return false;
        };
        self.push_roi_undo(slot, set, roi);
        if let Some(e) = self.edit.as_mut() {
            f(&mut e.stack, dims);
            e.stack.prune();
        }
        self.flush_edit();
        self.interp = None;
        true
    }

    /// What the contour window reports about the structure being edited.
    pub(super) fn edit_summary(&self, slot: usize) -> Option<(String, String, f64, usize, usize)> {
        let (set, roi) = self.edit_target(slot)?;
        let study = self.slots[slot].study.as_ref()?;
        let r = study.structure_sets.get(set)?.rois.get(roi)?;
        let grid = study.volume.grid();
        let st = Stack::from_roi(r, &grid);
        let points: usize = r.contours.iter().map(|c| c.points.len()).sum();
        Some((
            r.name.clone(),
            r.roi_type.clone(),
            st.volume_cm3(grid.spacing),
            st.occupied(),
            points,
        ))
    }

    /// Set the RT ROI Interpreted Type of the edited structure - the tag a
    /// planning system branches on.
    pub(super) fn set_edit_roi_type(&mut self, slot: usize, roi_type: &str) {
        let Some((set, roi)) = self.edit_target(slot) else {
            return;
        };
        if let Some(r) = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.structure_sets.get_mut(set))
            .and_then(|ss| ss.rois.get_mut(roi))
        {
            r.roi_type = roi_type.to_string();
        }
        self.settings_gen += 1;
    }

    // -- per-slice operations ---------------------------------------------

    /// Copy this slice's contours to the clipboard.
    pub(super) fn copy_slice_contours(&mut self, slot: usize) -> bool {
        let Some((set, roi)) = self.edit_target(slot) else {
            return false;
        };
        let axis = self.edit_axis(slot);
        let level = self.edit_level(slot);
        if self.working_stack(slot, set, roi, axis).is_none() {
            return false;
        }
        let region = self
            .edit
            .as_ref()
            .and_then(|e| e.stack.region_at(level))
            .cloned();
        match region {
            Some(r) if !r.is_empty() => {
                self.contour_clip = Some((axis, r));
                true
            }
            _ => false,
        }
    }

    /// Paste the clipboard onto the current slice, in the current draw mode.
    pub(super) fn paste_slice_contours(&mut self, slot: usize) -> bool {
        let Some((axis, region)) = self.contour_clip.clone() else {
            return false;
        };
        if axis != self.edit_axis(slot) {
            self.notice = Some(
                "The copied contours were taken on another plane; switch back to it to \
                 paste them."
                    .into(),
            );
            return false;
        }
        let level = self.edit_level(slot);
        let mode = self.draw_mode;
        self.with_edit_stack(slot, |st, _| {
            let dst = st.region_mut(level);
            for ring in &region.rings {
                if mode == DrawMode::Subtract {
                    dst.subtract_ring(ring);
                } else {
                    dst.add_ring(ring);
                }
            }
        })
    }

    /// Delete every contour of the current slice.
    pub(super) fn clear_slice_contours(&mut self, slot: usize) -> bool {
        let level = self.edit_level(slot);
        self.with_edit_stack(slot, |st, _| st.remove_level(level))
    }

    /// Thin the structure out: keep one slice in `keep`, drop the rest,
    /// between `from` and `to` inclusive. RayStation's "delete multiple
    /// contours", which exists because interpolation can put them back.
    pub(super) fn thin_slices(&mut self, slot: usize, keep: usize, from: usize, to: usize) -> bool {
        let keep = keep.max(2);
        self.with_edit_stack(slot, |st, _| {
            st.slices.retain(|s| {
                s.level < from || s.level > to || (s.level - from).is_multiple_of(keep)
            });
        })
    }

    /// Keep, or delete, the connected piece the crosshair sits in - the
    /// "keep component" and "delete component" of a planning system, picked
    /// with the crosshair instead of a click because that is the pointer
    /// this window has.
    pub(super) fn component_at_cursor(&mut self, slot: usize, keep: bool) -> bool {
        let Some(dims) = self.slots[slot].study.as_ref().map(|s| s.volume.dims) else {
            return false;
        };
        let c = self.slots[slot].cursor;
        let v = [
            (c[0].round().max(0.0) as usize).min(dims[0].saturating_sub(1)),
            (c[1].round().max(0.0) as usize).min(dims[1].saturating_sub(1)),
            (c[2].round().max(0.0) as usize).min(dims[2].saturating_sub(1)),
        ];
        let mut hit = false;
        let ok = self.with_edit_stack(slot, |st, dims| {
            let axis = st.axis;
            let [ua, va] = contours::plane_axes(axis);
            hit = st
                .region_at(v[axis])
                .is_some_and(|r| r.contains([v[ua] as f64, v[va] as f64]));
            if hit {
                let out = st.component_at(dims, v, keep);
                *st = out;
            }
        });
        if ok && !hit {
            self.notice = Some(
                "The crosshair is not inside this structure - put it in the piece to \
                 keep or delete first."
                    .into(),
            );
        }
        ok
    }

    /// Move the whole structure so that its centroid lands under the
    /// crosshair, in the plane it is drawn on.
    pub(super) fn move_to_crosshair(&mut self, slot: usize) -> bool {
        let axis = self.edit_axis(slot);
        let [ua, va] = contours::plane_axes(axis);
        let c = self.slots[slot].cursor;
        let target = [c[ua], c[va]];
        self.with_edit_stack(slot, |st, _| {
            let cur = st.centroid();
            st.translate([target[0] - cur[0], target[1] - cur[1]]);
        })
    }

    // -- interpolation -----------------------------------------------------

    /// Recompute the interpolation preview if it is stale.
    pub(super) fn refresh_interp(&mut self, slot: usize) {
        let Some((set, roi)) = self.edit_target(slot) else {
            self.interp = None;
            return;
        };
        let axis = self.edit_axis(slot);
        let fresh = self.interp.as_ref().is_some_and(|p| {
            p.slot == slot
                && p.set == set
                && p.roi == roi
                && p.axis == axis
                && p.gen == self.settings_gen
        });
        if fresh {
            return;
        }
        if self.working_stack(slot, set, roi, axis).is_none() {
            self.interp = None;
            return;
        }
        let Some(dims) = self.slots[slot].study.as_ref().map(|s| s.volume.dims) else {
            return;
        };
        let Some(e) = self.edit.as_ref() else { return };
        let stack = e.stack.interpolated(dims);
        self.interp = Some(InterpPreview {
            slot,
            set,
            roi,
            axis,
            stack,
            gen: self.settings_gen,
        });
    }

    /// How many slices the preview holds.
    pub(super) fn interp_count(&self, slot: usize) -> usize {
        self.interp
            .as_ref()
            .filter(|p| p.slot == slot)
            .map(|p| p.stack.occupied())
            .unwrap_or(0)
    }

    /// Accept the interpolated contours: one slice (the current one) or all.
    pub(super) fn accept_interp(&mut self, slot: usize, all: bool) -> bool {
        let level = self.edit_level(slot);
        let Some(p) = self.interp.as_ref().filter(|p| p.slot == slot) else {
            return false;
        };
        let take: Vec<(usize, crate::contours::Region)> = p
            .stack
            .slices
            .iter()
            .filter(|s| all || s.level == level)
            .map(|s| (s.level, s.region.clone()))
            .collect();
        if take.is_empty() {
            return false;
        }
        self.with_edit_stack(slot, |st, _| {
            for (level, region) in take {
                *st.region_mut(level) = region;
            }
        })
    }

    /// The interpolated rings to draw on one view, in that view's pixel
    /// coordinates. Only the plane the interpolation runs along can show
    /// them: on the other two the preview is a cross-section of contours
    /// that do not exist yet.
    pub(super) fn interp_preview_at(
        &self,
        slot: usize,
        plane: ViewPlane,
        level: usize,
    ) -> Option<InterpRings> {
        let p = self.interp.as_ref()?;
        if p.slot != slot || p.axis != contours::axis_of_plane(plane) {
            return None;
        }
        let region = p.stack.region_at(level)?;
        let study = self.slots[slot].study.as_ref()?;
        let color = study.structure_sets.get(p.set)?.rois.get(p.roi)?.color;
        let [ua, va] = contours::plane_axes(p.axis);
        let rings = region
            .rings
            .iter()
            .map(|r| {
                r.pts
                    .iter()
                    .map(|q| {
                        let mut v = [0.0f64; 3];
                        v[ua] = q[0];
                        v[va] = q[1];
                        v[p.axis] = level as f64;
                        let pp = study.volume.voxel_to_plane_pixel(plane, v);
                        [pp[0] as f32, pp[1] as f32]
                    })
                    .collect()
            })
            .collect();
        Some((rings, color))
    }
}
