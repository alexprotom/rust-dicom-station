//! Interactive segmentation: the brush and eraser, geodesic region growing
//! with its live preview, per-stroke undo, and the conversion out to
//! RTSTRUCT. The voxel algorithms themselves live in `crate::segmentation`;
//! this is the UI-side state machine around them.
//!
//! The drawing strip's tools are one set for structures and segments
//! (`struct_tools.rs`), so the contour tools land here too when a segment
//! is their subject: a closed curve is filled into (or cut out of) the
//! slice by the drawing mode's rule ([`ViewerApp::seg_apply_ring`]), and a
//! nudge pushes the slice's outline and fills it back
//! ([`ViewerApp::seg_nudge`]). Every one of them goes through the
//! segment's own stroke journal, so Ctrl+Z takes a curve or a drag back in
//! one step, like a brush stroke. Grow, the other way round, lands in a
//! structure's contours when a structure is the subject
//! ([`ViewerApp::grow_into_structure`]).

use super::*;
use crate::contours::{self, Poly};

/// A Nudge stroke on a segment: which slice of which segment, and the
/// outline the stroke pushes (kept for the whole drag).
pub(super) struct SegNudge {
    slot: usize,
    seg: usize,
    axis: usize,
    level: usize,
    outline: segmentation::OutlineNudge,
}

impl ViewerApp {
    // -- Interactive segmentation ------------------------------------------
    /// Everything the 2D segmentation overlays of a slot depend on.
    pub(super) fn seg_overlay_hash(&self, slot: usize) -> u64 {
        let mut h = mix(0xA07A_5E65u64, slot as u64 + 1);
        for (i, s) in self.slots[slot].segs().iter().enumerate() {
            h = mix(h, i as u64 + 1);
            h = mix(h, s.gen);
            h = mix(h, s.visible as u64);
            h = mix(
                h,
                u64::from_le_bytes([s.color[0], s.color[1], s.color[2], 0, 0, 0, 0, 0]),
            );
        }
        if self.grow.as_ref().is_some_and(|g| g.slot == slot) {
            h = mix(h, self.grow_gen.wrapping_add(1));
        }
        h
    }

    /// Everything the 3D segmentation meshes of a slot depend on
    /// (geometry only - color and visibility are applied at draw time).
    pub(super) fn seg_mesh_hash(&self, slot: usize) -> u64 {
        let mut h = mix(0x5E6_3E54u64, slot as u64 + 1);
        for s in self.slots[slot].segs() {
            h = mix(h, s.gen.wrapping_add(1));
        }
        h
    }

    /// The segmentation series the tools write into, creating one bound to
    /// the displayed image series when the study holds none that fits it.
    ///
    /// Segmentation series live in the study, not in the view state, so a
    /// series switch no longer throws painted work away; the price is that
    /// the tools have to be told which of them they are editing.
    pub(super) fn ensure_seg_series(&mut self, slot: usize) -> Option<usize> {
        let s = &mut self.slots[slot];
        let study = s.study.as_mut()?;
        let dims = study.volume.dims;
        if let Some(i) = (!study.seg_series.is_empty())
            .then(|| s.active_seg_series.min(study.seg_series.len() - 1))
            .filter(|&i| study.seg_series[i].grid.dims == dims)
        {
            s.active_seg_series = i;
            return Some(i);
        }
        let se = study.series.get(study.active_series);
        let uid = se.map(|x| x.uid.clone()).unwrap_or_default();
        let study_uid = se.map(|x| x.study_uid.clone()).unwrap_or_default();
        if let Some(i) = study
            .seg_series
            .iter()
            .position(|sr| sr.referenced_series_uid == uid && sr.grid.dims == dims)
        {
            s.active_seg_series = i;
            return Some(i);
        }
        let label = format!("Segmentations {}", study.seg_series.len() + 1);
        study.seg_series.push(crate::dicomseg::SegSeries::new(
            label,
            study.volume.grid(),
            uid,
            study_uid,
        ));
        s.active_seg_series = study.seg_series.len() - 1;
        Some(s.active_seg_series)
    }

    /// Resample the segmentation series drawn on the displayed image series
    /// onto its lattice, so overlays, brush and meshes can index them with
    /// the volume's dimensions. Series belonging to another image series are
    /// left exactly as they are.
    pub(super) fn rebind_seg_series(&mut self, slot: usize) {
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        let vol = study.volume.clone();
        let (uid, for_uid) = (
            study
                .series
                .get(study.active_series)
                .map(|s| s.uid.clone())
                .unwrap_or_default(),
            vol.frame_of_reference_uid.clone(),
        );
        let mut changed = false;
        for ser in &mut study.seg_series {
            let mine = ser.referenced_series_uid == uid
                || (ser.referenced_series_uid.is_empty()
                    && !for_uid.is_empty()
                    && ser.grid.frame_of_reference_uid == for_uid);
            if mine {
                changed |= ser.rebind(&vol);
            }
        }
        if changed {
            self.settings_gen += 1;
        }
    }

    /// Create a new segment in the slot's active series and make it active.
    pub(super) fn create_seg(&mut self, slot: usize) {
        if self.ensure_seg_series(slot).is_none() {
            return;
        }
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let dims = study.volume.dims;
        let color = segmentation::SEG_PALETTE[self.seg_counter % segmentation::SEG_PALETTE.len()];
        self.seg_counter += 1;
        let name = format!("Seg {}", self.seg_counter);
        let s = &mut self.slots[slot];
        let Some(segs) = s.segs_mut() else { return };
        segs.push(Segmentation::new(name, color, dims));
        let n = segs.len();
        s.active_seg = n - 1;
    }

    /// Apply one brush sample: paints a capsule from the previous sample of
    /// this stroke to `vxl` (creating a segmentation first if none exists).
    pub(super) fn apply_brush(
        &mut self,
        slot: usize,
        plane: ViewPlane,
        slice: usize,
        vxl: [f64; 3],
        erase: bool,
    ) {
        if self.slots[slot].segs().is_empty() {
            if erase {
                return;
            }
            self.create_seg(slot);
        }
        let radius = self.brush_radius_mm as f64;
        let plane2d = if self.brush_3d {
            None
        } else {
            Some((plane, slice))
        };
        let from = match self.paint_last {
            Some((s, p)) if s == slot => p,
            _ => vxl,
        };
        let Some(vol) = self.slots[slot].study.as_ref().map(|s| s.volume.clone()) else {
            return;
        };
        // The edge band, as the contour brush has it: the values it lets
        // the stroke cover, from the display window.
        let values = (self.brush_band != super::contour_edit::EdgeBand::None).then(|| {
            let (wc, ww) = self.window_of(slot);
            self.brush_band.limits(wc, ww, self.brush_sensitivity)
        });
        let s = &mut self.slots[slot];
        let active = s.active_seg;
        let Some(segs) = s.segs_mut() else { return };
        if segs.is_empty() {
            return;
        }
        let idx = active.min(segs.len() - 1);
        segs[idx].paint_capsule_within(&vol, from, vxl, radius, erase, plane2d, values);
        self.paint_last = Some((slot, vxl));
    }

    /// Close the brush stroke in progress (one undo step).
    pub(super) fn end_paint_stroke(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        let active = s.active_seg;
        if let Some(seg) = s.segs_mut().and_then(|segs| segs.get_mut(active)) {
            seg.end_stroke();
        }
        self.paint_last = None;
    }

    pub(super) fn begin_grow(&mut self, slot: usize, seed: [f64; 3], y: f32) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if !study.has_volume() {
            return;
        }
        let dims = study.volume.dims;
        let seed = [
            (seed[0].round().max(0.0) as usize).min(dims[0] - 1),
            (seed[1].round().max(0.0) as usize).min(dims[1] - 1),
            (seed[2].round().max(0.0) as usize).min(dims[2] - 1),
        ];
        self.cancel_grow();
        self.grow = Some(GrowDrag {
            slot,
            level: 1.0,
            y0: y,
            capped: false,
        });
        // The limiting structure, when one is chosen and it really covers
        // the seed: a limit the seed is outside of would grow a single
        // voxel, which reads as a broken tool rather than as a limit.
        let limit = self.grow_limit_mask(slot);
        let idx = seed[2] * dims[0] * dims[1] + seed[1] * dims[0] + seed[0];
        let limit = match limit {
            Some((m, _)) if m.get(idx).copied().unwrap_or(0) != 0 => Some(std::sync::Arc::new(m)),
            Some((_, name)) => {
                self.notice = Some(format!(
                    "The seed is outside '{name}' - growing without the limit."
                ));
                None
            }
            None => None,
        };
        let vol = &self.slots[slot].study.as_ref().unwrap().volume;
        self.grow_state.set_limit(limit);
        self.grow_state.seed(vol, seed);
        self.sync_grow_preview();
    }

    /// The limiting structure's mask on the displayed lattice, with its
    /// name; `None` when none is chosen or it no longer resolves.
    fn grow_limit_mask(&self, slot: usize) -> Option<(Vec<u8>, String)> {
        let item = self.grow_limit?;
        let (mask, grid, name, _) = self.item_mask_grid(slot, item)?;
        let target = self.slots[slot].study.as_ref()?.volume.grid();
        let mask = if grid.matches(&target) {
            mask
        } else {
            crate::dicomseg::resample_mask(&mask, &grid, &target)
        };
        Some((mask, name))
    }

    pub(super) fn update_grow(&mut self, y: f32) {
        let Some(g) = &mut self.grow else { return };
        // Dragging up extends the geodesic reach exponentially.
        let level = ((g.y0 - y) * 0.015).exp().clamp(0.02, 1000.0);
        if (level / g.level - 1.0).abs() < 0.01 {
            return;
        }
        g.level = level;
        let slot = g.slot;
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        self.grow_state.set_level(&study.volume, level);
        self.sync_grow_preview();
    }

    /// Bring the preview mask in line with the grow state's current
    /// selection (only previously marked voxels are cleared, so the
    /// full-volume scratch buffer never needs a full wipe).
    pub(super) fn sync_grow_preview(&mut self) {
        let Some(g) = &mut self.grow else { return };
        g.capped = self.grow_state.capped;
        let n = self.slots[g.slot]
            .study
            .as_ref()
            .map(|st| st.volume.dims.iter().product())
            .unwrap_or(0);
        if self.grow_preview.len() != n {
            self.grow_preview.clear();
            self.grow_preview.resize(n, 0);
        }
        for &v in &self.grow_marked {
            if let Some(p) = self.grow_preview.get_mut(v as usize) {
                *p = 0;
            }
        }
        self.grow_marked.clear();
        self.grow_marked.extend_from_slice(&self.grow_state.voxels);
        for &v in &self.grow_marked {
            self.grow_preview[v as usize] = 1;
        }
        self.grow_gen += 1;
    }

    /// Commit the previewed region into what the tools act on, filling
    /// slice-enclosed holes (vessels etc.) so the organ is solid: the
    /// active segment's voxels, or - a structure being the subject - the
    /// structure's contours on every slice the region reached.
    pub(super) fn commit_grow(&mut self) {
        let Some(g) = self.grow.take() else { return };
        let slot = g.slot;
        let into_structure = self.edit_subject(slot) == super::seg_edit::EditSubject::Structure;
        if !self.grow_state.voxels.is_empty() && into_structure {
            let dims = self.slots[slot].study.as_ref().map(|s| s.volume.dims);
            if let Some(dims) = dims {
                let mut voxels = self.grow_state.voxels.clone();
                segmentation::fill_holes_slicewise(&mut voxels, dims);
                self.grow_into_structure(slot, &voxels);
            }
        } else if !self.grow_state.voxels.is_empty() {
            if self.slots[slot].segs().is_empty() {
                self.create_seg(slot);
            }
            let dims = self.slots[slot].study.as_ref().unwrap().volume.dims;
            let mut voxels = self.grow_state.voxels.clone();
            segmentation::fill_holes_slicewise(&mut voxels, dims);
            let s = &mut self.slots[slot];
            let active = s.active_seg;
            if let Some(segs) = s.segs_mut() {
                if !segs.is_empty() {
                    let idx = active.min(segs.len() - 1);
                    segs[idx].add_voxels(&voxels);
                    segs[idx].end_stroke();
                }
            }
        }
        self.clear_grow_preview();
        self.grow_state.release();
    }

    /// Abandon any region-growing drag and its preview.
    pub(super) fn cancel_grow(&mut self) {
        self.grow = None;
        self.clear_grow_preview();
        self.grow_state.release();
    }

    pub(super) fn clear_grow_preview(&mut self) {
        if self.grow_marked.is_empty() {
            return;
        }
        for &v in &self.grow_marked {
            if let Some(p) = self.grow_preview.get_mut(v as usize) {
                *p = 0;
            }
        }
        self.grow_marked.clear();
        self.grow_gen += 1;
    }

    /// One closed curve of the drawing tools onto the active segment: the
    /// voxels of slice `level` along `plane` whose centre it encloses are
    /// added or cut by the drawing mode's rule, decided on the slice's own
    /// outline as it is on a structure's contours. One undo step. A curve
    /// that adds on a workspace with no segment makes one first.
    pub(super) fn seg_apply_ring(
        &mut self,
        slot: usize,
        plane: ViewPlane,
        level: usize,
        ring: Poly,
    ) {
        if ring.is_empty() {
            return;
        }
        if !self.has_editable_seg(slot) {
            if self.draw_mode == DrawMode::Subtract {
                return;
            }
            self.create_seg(slot);
        }
        let axis = contours::axis_of_plane(plane);
        let mode = self.draw_mode;
        let s = &mut self.slots[slot];
        let active = s.active_seg;
        let Some(seg) = s.segs_mut().and_then(|v| v.get_mut(active)) else {
            return;
        };
        let [w, h] = seg.slice_dims(axis);
        let mut slice = seg.slice(axis, level);
        if slice.is_empty() {
            return;
        }
        let cut = match mode {
            DrawMode::Extend => false,
            DrawMode::Subtract => true,
            DrawMode::Auto => segmentation::slice_auto_cuts(&slice, w, h, &ring),
        };
        if segmentation::ring_into_slice(&mut slice, w, h, &ring, cut) {
            // Whatever stroke was open is its own step; this curve is one.
            seg.end_stroke();
            seg.write_slice(axis, level, &slice);
            seg.end_stroke();
        }
    }

    /// One sample of a Nudge drag on the active segment: the outline of
    /// slice `level` along `plane`, read when the stroke starts, is pushed
    /// from `from` to `to` (voxel coordinates) and filled back. A stroke
    /// that moves to another slice or segment starts again there.
    pub(super) fn seg_nudge(
        &mut self,
        slot: usize,
        plane: ViewPlane,
        level: usize,
        from: [f64; 3],
        to: [f64; 3],
        radius_mm: f64,
    ) {
        let axis = contours::axis_of_plane(plane);
        let active = self.slots[slot].active_seg;
        let same = self.seg_nudge.as_ref().is_some_and(|n| {
            n.slot == slot && n.seg == active && n.axis == axis && n.level == level
        });
        if !same {
            self.end_seg_nudge();
            let Some(seg) = self.slots[slot].segs().get(active) else {
                return;
            };
            let [w, h] = seg.slice_dims(axis);
            let Some(outline) = segmentation::OutlineNudge::new(&seg.slice(axis, level), w, h)
            else {
                return;
            };
            self.seg_nudge = Some(SegNudge {
                slot,
                seg: active,
                axis,
                level,
                outline,
            });
        }
        let Some(spacing) = self.slots[slot].study.as_ref().map(|s| s.volume.spacing) else {
            return;
        };
        let [ua, va] = contours::plane_axes(axis);
        let mm = [spacing[ua], spacing[va]];
        let (a, b) = ([from[ua], from[va]], [to[ua], to[va]]);
        let Some(n) = self.seg_nudge.as_mut() else {
            return;
        };
        let Some(next) = n.outline.push(a, b, radius_mm, mm) else {
            return;
        };
        if let Some(seg) = self.slots[slot].segs_mut().and_then(|v| v.get_mut(active)) {
            seg.write_slice(axis, level, &next);
        }
    }

    /// The Nudge drag on a segment let go: its samples are one undo step.
    pub(super) fn end_seg_nudge(&mut self) {
        let Some(n) = self.seg_nudge.take() else {
            return;
        };
        if let Some(seg) = self
            .slots
            .get_mut(n.slot)
            .and_then(|s| s.segs_mut())
            .and_then(|v| v.get_mut(n.seg))
        {
            seg.end_stroke();
        }
    }

    /// Make the visible segment holding the voxel at `v` the active one:
    /// the smallest, when several overlap there, as with structures.
    pub(super) fn pick_seg_at(&mut self, slot: usize, v: [f64; 3]) -> bool {
        let s = &self.slots[slot];
        let Some(dims) = s.study.as_ref().map(|st| st.volume.dims) else {
            return false;
        };
        let mut idx = 0usize;
        for a in (0..3).rev() {
            let c = v[a].round();
            if c < 0.0 || c >= dims[a] as f64 {
                return false;
            }
            idx = idx * dims[a] + c as usize;
        }
        let hit = s
            .segs()
            .iter()
            .enumerate()
            .filter(|(_, g)| g.visible && g.mask.get(idx).is_some_and(|&m| m != 0))
            .min_by_key(|(_, g)| g.count)
            .map(|(i, _)| i);
        match hit {
            Some(i) => {
                self.slots[slot].active_seg = i;
                true
            }
            None => false,
        }
    }

    /// Undo the last stroke of a slot's active segmentation.
    pub(super) fn undo_active_seg(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        let active = s.active_seg;
        if let Some(seg) = s.segs_mut().and_then(|segs| segs.get_mut(active)) {
            seg.undo_last();
        }
    }

    /// Convert a segmentation into RTSTRUCT contours: appends a ROI to the
    /// slot's active structure set (creating an in-memory set if the study
    /// has none), so it displays like any ROI and rides the DICOM export.
    ///
    /// `roi_type` is the RT ROI Interpreted Type the ROI is filed under.
    /// Everything painted by hand is an `ORGAN`; the body contour is the
    /// one thing that has to be an `EXTERNAL`, because that is the tag a
    /// planning system looks for to find the patient surface.
    pub(super) fn seg_to_rtstruct(&mut self, slot: usize, seg_idx: usize, roi_type: &str) {
        // The contours are built first, under a shared borrow, because the
        // segments now live inside the very study the ROI is appended to.
        let s = &self.slots[slot];
        let Some(study) = s.study.as_ref() else {
            return;
        };
        let Some(seg) = s.segs().get(seg_idx) else {
            return;
        };
        let vol = study.volume.clone();
        let number = study
            .structure_sets
            .get(s.active_structs)
            .map(|ss| ss.rois.iter().map(|r| r.number).max().unwrap_or(0) + 1)
            .unwrap_or(1);
        let mut roi = segmentation::mask_to_roi(seg, &vol.grid(), number);
        roi.roi_type = roi_type.to_string();

        let StudySlot {
            study,
            active_structs,
            roi_visible,
            ..
        } = &mut self.slots[slot];
        let Some(study) = study.as_mut() else { return };
        if let Some(ss) = study.structure_sets.get_mut(*active_structs) {
            ss.rois.push(roi);
            roi_visible.push(true);
        } else {
            let active_series = study.series.get(study.active_series);
            // Pseudo-UID for the in-memory set (rewritten on DICOM export).
            study.structure_sets.push(crate::rtstruct::StructureSet {
                label: "Segmentations".into(),
                frame_of_reference_uid: vol.frame_of_reference_uid.clone(),
                sop_instance_uid: crate::dicom_export::new_uid(),
                series_instance_uid: crate::dicom_export::new_uid(),
                study_uid: active_series
                    .map(|s| s.study_uid.clone())
                    .unwrap_or_default(),
                referenced_series_uid: active_series.map(|s| s.uid.clone()).unwrap_or_default(),
                file_name: crate::rtstruct::StructureSet::MADE_HERE[1].into(),
                locked: false,
                rois: vec![roi],
            });
            *active_structs = study.structure_sets.len() - 1;
            *roi_visible = vec![true];
        }
        self.settings_gen += 1;
    }

    /// Materialize the chosen organs the way the results window says: as
    /// segments, as contours in a structure set, or both - on the displayed
    /// series, or on every phase the run covered.
    ///
    /// The masks are extracted and their contours traced on a worker
    /// ([`prepare_autoseg`]), which takes the run's label maps by value and
    /// lets each go once its phase is done; what it made is filed by
    /// [`Self::on_autoseg_applied`] once it lands, after the workspace has
    /// been checked again. Cancelling it files nothing.
    pub(super) fn apply_autoseg_selection(&mut self) {
        if self.autoseg_apply_job.is_some() {
            return;
        }
        let Some(p) = self.autoseg_pending.take() else {
            return;
        };
        let slot = p.slot;
        let classes: Vec<(u8, String, [u8; 3])> = p
            .organs
            .iter()
            .zip(p.selected.iter())
            .filter(|(_, sel)| **sel)
            .map(|(organ, _)| {
                let name = if p.tg263 {
                    zoo::tg263_name(organ.name).unwrap_or_else(|| organ.name.to_string())
                } else {
                    organ.name.to_string()
                };
                (organ.label, name, organ.color)
            })
            .collect();
        if classes.is_empty() {
            return;
        }
        let Some((first, _)) = p.phases.first() else {
            return;
        };
        let group = first.group.clone();
        let mut output = p.output.clone();
        if group.is_none() && !self.slot_still_shows_image(slot, first) {
            self.error = Some("Workspace changed - auto-segmentation result discarded.".into());
            return;
        }
        // The structure set chosen for the contours, by identity: the sets
        // are indexed, and the index may name another set by the time the
        // worker is done. One that is not there now means a new set, as it
        // did when the contours were filed at once.
        let set_uid = match (group.is_none() && output.kind.structures(), output.set) {
            (true, Some(i)) => self.slots[slot]
                .study
                .as_ref()
                .and_then(|st| st.structure_sets.get(i))
                .map(|ss| ss.sop_instance_uid.clone()),
            _ => None,
        };
        if set_uid.is_none() {
            output.set = None;
        }
        let kind = output.kind;
        let phases = p.phases;
        let progress = Arc::new(Progress::default());
        progress.set("Adding the structures");
        self.autoseg_apply_slot = slot;
        self.autoseg_apply_job = Some(Job::spawn(progress, move |pr| {
            let made = prepare_autoseg(phases, &classes, group, kind, pr);
            (
                slot,
                made.map(|made| AutosegApply {
                    output,
                    set_uid,
                    made,
                }),
            )
        }));
    }

    /// File what [`Self::apply_autoseg_selection`]'s worker made, if the
    /// workspace still holds what it was made for: on a run over the
    /// phases, each phase that is still in the workspace (and, if it is on
    /// screen, on the same lattice); on the displayed series, that same
    /// series on that same lattice - a same-sized phase stepped to
    /// meanwhile is not it - and the structure set chosen, wherever it now
    /// sits in the list and only if it is not locked.
    pub(super) fn on_autoseg_applied(&mut self, slot: usize, applied: AutosegApply) {
        let AutosegApply {
            mut output,
            set_uid,
            made,
        } = applied;
        match made {
            AutosegMade::Phases { group, phases } => {
                self.land_prepared_phases(slot, &group, output.kind, phases);
            }
            AutosegMade::Displayed { info, rois, segs } => {
                if !self.slot_still_shows_image(slot, &info) {
                    self.error =
                        Some("Workspace changed - auto-segmentation result discarded.".into());
                    return;
                }
                if let Some(uid) = set_uid {
                    let at = self.slots[slot].study.as_ref().and_then(|st| {
                        st.structure_sets
                            .iter()
                            .position(|ss| ss.sop_instance_uid == uid)
                    });
                    match at {
                        Some(i) => output.set = Some(i),
                        None => {
                            self.error = Some(
                                "The structure set chosen for the auto-segmentation result \
                                 is gone - the result was discarded."
                                    .into(),
                            );
                            return;
                        }
                    }
                }
                self.land_prepared_masks(slot, &output, "Auto-segmentation", rois, segs);
            }
        }
    }
}

/// What *Add* of the auto-segmentation results made on its worker, filed
/// by [`ViewerApp::on_autoseg_applied`] once checked.
pub(super) struct AutosegApply {
    output: ToolOutput,
    /// The SOP Instance UID of the structure set `output.set` named when
    /// *Add* was clicked; `None` for a new set (or no contours).
    set_uid: Option<String>,
    made: AutosegMade,
}

/// The masks and contours made from a run's label maps.
pub(super) enum AutosegMade {
    /// A run on the displayed series: the contours (when structures are
    /// filed), traced on the lattice of `info`, and the segments (when
    /// segments are).
    Displayed {
        /// Boxed, so the two variants are of one size.
        info: Box<PhaseInfo>,
        rois: Vec<crate::rtstruct::Roi>,
        segs: Vec<Segmentation>,
    },
    /// A run over the phases of `group`, in temporal order.
    Phases {
        group: String,
        phases: Vec<workflow::group::PreparedPhase>,
    },
}

/// Extract the chosen classes from each label map of a run and trace their
/// contours, on the calling (worker) thread. The results are consumed one
/// at a time and each label map goes as soon as its masks exist; when only
/// contours are filed, a phase's masks go too once traced, so a run over
/// ten phases holds one phase's masks at a time rather than all of them.
/// Every structure gets the RT ROI Interpreted Type `ORGAN`.
fn prepare_autoseg(
    phases: Vec<(PhaseInfo, autoseg::AutosegResult)>,
    classes: &[(u8, String, [u8; 3])],
    group: Option<String>,
    kind: OutputKind,
    p: &Progress,
) -> anyhow::Result<AutosegMade> {
    let n = phases.len().max(1);
    let Some(group) = group else {
        let (info, result) = phases
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("the run left no result"))?;
        p.set("Extracting the structures");
        // One scan of the label map fills the masks of all the chosen
        // structures.
        let segs = Segmentation::from_label_map_many(result.dims, &result.labels, classes);
        drop(result);
        if p.cancelled() {
            anyhow::bail!(crate::progress::CANCELLED);
        }
        let rois = if kind.structures() {
            p.set("Tracing the contours");
            segmentation::masks_to_rois(&segs, &info.grid, "ORGAN")
        } else {
            Vec::new()
        };
        let segs = if kind.segments() { segs } else { Vec::new() };
        return Ok(AutosegMade::Displayed {
            info: Box::new(info),
            rois,
            segs,
        });
    };
    let mut out = Vec::with_capacity(phases.len());
    for (i, (info, result)) in phases.into_iter().enumerate() {
        if p.cancelled() {
            anyhow::bail!(crate::progress::CANCELLED);
        }
        p.set_phase(i as f32 / n as f32, 1.0 / n as f32);
        p.set(format!(
            "Phase {} ({}/{n}): adding the structures",
            info.label,
            i + 1
        ));
        // One scan of each phase's label map fills the masks of all the
        // chosen classes; a class a phase did not find lands nowhere on
        // that phase.
        let mine: Vec<(u8, String, [u8; 3])> = classes
            .iter()
            .filter(|(label, _, _)| result.organs.iter().any(|o| o.label == *label))
            .cloned()
            .collect();
        let segs = Segmentation::from_label_map_many(result.dims, &result.labels, &mine);
        drop(result);
        out.push(info.copy(segs, Vec::new()).prepare(kind.filing(), "ORGAN"));
    }
    if p.cancelled() {
        anyhow::bail!(crate::progress::CANCELLED);
    }
    Ok(AutosegMade::Phases { group, phases: out })
}
