//! Interactive segmentation: the brush and eraser, geodesic region growing
//! with its live preview, per-stroke undo, and the conversion out to
//! RTSTRUCT. The voxel algorithms themselves live in `crate::segmentation`;
//! this is the UI-side state machine around them.

use super::*;

impl ViewerApp {
    // -- Interactive segmentation ------------------------------------------
    /// Everything the 2D segmentation overlays of a slot depend on.
    pub(super) fn seg_overlay_hash(&self, slot: usize) -> u64 {
        let mut h = mix(0xA07A_5E65u64, slot as u64 + 1);
        for (i, s) in self.slots[slot].segs.iter().enumerate() {
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
    /// (geometry only — color and visibility are applied at draw time).
    pub(super) fn seg_mesh_hash(&self, slot: usize) -> u64 {
        let mut h = mix(0x5E6_3E54u64, slot as u64 + 1);
        for s in &self.slots[slot].segs {
            h = mix(h, s.gen.wrapping_add(1));
        }
        h
    }

    /// Create a new segmentation on a slot's volume and make it active.
    pub(super) fn create_seg(&mut self, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let dims = study.volume.dims;
        let color = segmentation::SEG_PALETTE[self.seg_counter % segmentation::SEG_PALETTE.len()];
        self.seg_counter += 1;
        let name = format!("Seg {}", self.seg_counter);
        let s = &mut self.slots[slot];
        s.segs.push(Segmentation::new(name, color, dims));
        s.active_seg = s.segs.len() - 1;
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
        if self.slots[slot].segs.is_empty() {
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
        let StudySlot {
            study,
            segs,
            active_seg,
            ..
        } = &mut self.slots[slot];
        let vol = &study.as_ref().unwrap().volume;
        let idx = (*active_seg).min(segs.len() - 1);
        segs[idx].paint_capsule(vol, from, vxl, radius, erase, plane2d);
        self.paint_last = Some((slot, vxl));
    }

    /// Close the brush stroke in progress (one undo step).
    pub(super) fn end_paint_stroke(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        if let Some(seg) = s.segs.get_mut(s.active_seg) {
            seg.end_stroke();
        }
        self.paint_last = None;
    }

    pub(super) fn begin_grow(&mut self, slot: usize, seed: [f64; 3], y: f32) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
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
        let vol = &self.slots[slot].study.as_ref().unwrap().volume;
        self.grow_state.seed(vol, seed);
        self.sync_grow_preview();
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

    /// Commit the previewed region into the slot's active segmentation,
    /// filling slice-enclosed holes (vessels etc.) so the organ is solid.
    pub(super) fn commit_grow(&mut self) {
        let Some(g) = self.grow.take() else { return };
        let slot = g.slot;
        if !self.grow_state.voxels.is_empty() {
            if self.slots[slot].segs.is_empty() {
                self.create_seg(slot);
            }
            let dims = self.slots[slot].study.as_ref().unwrap().volume.dims;
            let mut voxels = self.grow_state.voxels.clone();
            segmentation::fill_holes_slicewise(&mut voxels, dims);
            let s = &mut self.slots[slot];
            let idx = s.active_seg.min(s.segs.len() - 1);
            s.segs[idx].add_voxels(&voxels);
            s.segs[idx].end_stroke();
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

    /// Undo the last stroke of a slot's active segmentation.
    pub(super) fn undo_active_seg(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        if let Some(seg) = s.segs.get_mut(s.active_seg) {
            seg.undo_last();
        }
    }

    /// Convert a segmentation into RTSTRUCT contours: appends a ROI to the
    /// slot's active structure set (creating an in-memory set if the study
    /// has none), so it displays like any ROI and rides the DICOM export.
    pub(super) fn seg_to_rtstruct(&mut self, slot: usize, seg_idx: usize) {
        let StudySlot {
            study,
            segs,
            active_structs,
            roi_visible,
            ..
        } = &mut self.slots[slot];
        let Some(study) = study.as_mut() else { return };
        let Some(seg) = segs.get(seg_idx) else { return };
        let vol = &study.volume;
        if let Some(ss) = study.structure_sets.get_mut(*active_structs) {
            let number = ss.rois.iter().map(|r| r.number).max().unwrap_or(0) + 1;
            ss.rois.push(segmentation::mask_to_roi(seg, vol, number));
            roi_visible.push(true);
        } else {
            let active_series = study.series.get(study.active_series);
            // Pseudo-UID for the in-memory set (rewritten on DICOM export).
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            study.structure_sets.push(crate::rtstruct::StructureSet {
                label: "Segmentations".into(),
                frame_of_reference_uid: vol.frame_of_reference_uid.clone(),
                sop_instance_uid: format!("2.25.{stamp}"),
                study_uid: active_series
                    .map(|s| s.study_uid.clone())
                    .unwrap_or_default(),
                referenced_series_uid: active_series.map(|s| s.uid.clone()).unwrap_or_default(),
                file_name: "painted-segmentation".into(),
                rois: vec![segmentation::mask_to_roi(seg, vol, 1)],
            });
            *active_structs = study.structure_sets.len() - 1;
            *roi_visible = vec![true];
        }
        self.settings_gen += 1;
    }

    /// Materialize the chosen organs as editable segmentations (and
    /// optionally RTSTRUCT contours).
    pub(super) fn apply_autoseg_selection(&mut self) {
        let Some(p) = self.autoseg_pending.take() else {
            return;
        };
        let slot = p.slot;
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        if study.volume.dims != p.result.volume_dims {
            self.error = Some("Dataset changed — auto-segmentation result discarded.".into());
            return;
        }
        let dims = study.volume.dims;
        let first_new = self.slots[slot].segs.len();
        let mut added = 0usize;
        for (organ, sel) in p.result.organs.iter().zip(p.selected.iter()) {
            if !sel {
                continue;
            }
            let seg = Segmentation::from_label_map(
                organ.name.to_string(),
                organ.color,
                dims,
                &p.result.labels,
                organ.label,
            );
            self.slots[slot].segs.push(seg);
            added += 1;
        }
        if added == 0 {
            return;
        }
        self.slots[slot].active_seg = first_new;
        if p.also_rs {
            for i in first_new..first_new + added {
                self.seg_to_rtstruct(slot, i);
            }
        }
    }
}
