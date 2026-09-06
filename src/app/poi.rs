//! Points of interest: markers, reference points and the localization
//! point.
//!
//! RTSTRUCT has no separate object for these - a POI is an ROI whose
//! geometry is a single `POINT` contour ([`crate::rtstruct::Roi::is_poi`]),
//! which is why they are read, drawn and exported by the same code as
//! everything else and this module is only what a planner *does* with one:
//! put it somewhere, find it again, move it, and say which one the patient
//! is lined up on.

use crate::geometry::Vec3;

use super::*;

impl ViewerApp {
    /// A new point of interest at the crosshair. Returns its index in the
    /// active structure set.
    pub(super) fn new_poi(&mut self, slot: usize, name: Option<String>) -> Option<usize> {
        let p = self.crosshair_patient(slot)?;
        let name = name.unwrap_or_else(|| format!("POI {}", self.poi_counter + 1));
        self.poi_counter += 1;
        let idx = self.new_roi(slot, Some(name), "MARKER")?;
        self.set_poi_at(slot, idx, p);
        Some(idx)
    }

    /// A point of interest at the centre of gravity of a structure - the
    /// one POI nobody wants to place by hand.
    pub(super) fn poi_at_structure(&mut self, slot: usize, roi: usize) -> Option<usize> {
        let (mask, grid, name, _) = self.item_mask_grid(
            slot,
            combine_win::ItemRef {
                kind: SetKind::Structures,
                set: self.slots[slot].active_structs,
                idx: roi,
            },
        )?;
        let c = crate::motion::centroid_mm(&mask, &grid)?;
        self.poi_counter += 1;
        let idx = self.new_roi(slot, Some(format!("{name} centre")), "MARKER")?;
        self.set_poi_at(slot, idx, c);
        Some(idx)
    }

    /// Move (or place) a point of interest.
    pub(super) fn set_poi_at(&mut self, slot: usize, roi: usize, p: Vec3) -> bool {
        let set = self.slots[slot].active_structs;
        let ok = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.structure_sets.get_mut(set))
            .and_then(|ss| ss.rois.get_mut(roi))
            .map(|r| r.set_point(p))
            .unwrap_or(false);
        if ok {
            self.settings_gen += 1;
        }
        ok
    }

    /// Move the point of interest to the crosshair.
    pub(super) fn move_poi_to_crosshair(&mut self, slot: usize, roi: usize) -> bool {
        match self.crosshair_patient(slot) {
            Some(p) => self.set_poi_at(slot, roi, p),
            None => false,
        }
    }

    /// The same for a structure: centre the views on its centre of gravity.
    pub(super) fn localize_roi(&mut self, slot: usize, roi: usize) -> bool {
        let item = combine_win::ItemRef {
            kind: SetKind::Structures,
            set: self.slots[slot].active_structs,
            idx: roi,
        };
        // A POI is its own centre, and rasterizing one would find nothing.
        if let Some(p) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.structure_sets.get(item.set))
            .and_then(|ss| ss.rois.get(roi))
            .and_then(|r| r.point())
        {
            return self.set_cursor_patient(slot, p);
        }
        let Some((mask, grid, _, _)) = self.item_mask_grid(slot, item) else {
            return false;
        };
        match crate::motion::centroid_mm(&mask, &grid) {
            Some(c) => self.set_cursor_patient(slot, c),
            None => {
                self.notice = Some("That structure has no geometry to centre on.".into());
                false
            }
        }
    }

    /// Put the crosshair on a patient point of this dataset.
    fn set_cursor_patient(&mut self, slot: usize, p: Vec3) -> bool {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return false;
        };
        let v = study.volume.patient_to_voxel(p);
        let dims = study.volume.dims;
        if (0..3).any(|a| v[a] < -0.5 || v[a] > dims[a] as f64 - 0.5) {
            self.notice = Some("That point lies outside the displayed images.".into());
            return false;
        }
        // Through `set_cursor`, so the three views follow and a linked
        // dataset follows through the registration, exactly as a click in
        // a view would.
        self.set_cursor(slot, v, usize::MAX);
        true
    }

    /// Make this POI the structure set's localization point, and no other.
    ///
    /// A patient is lined up on one point, so this is a radio button
    /// spread over a list: the type of every other POI in the set falls
    /// back to `MARKER`.
    pub(super) fn set_localization(&mut self, slot: usize, roi: usize) {
        let set = self.slots[slot].active_structs;
        let Some(ss) = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.structure_sets.get_mut(set))
        else {
            return;
        };
        for (i, r) in ss.rois.iter_mut().enumerate() {
            if !r.is_poi() {
                continue;
            }
            r.roi_type = if i == roi {
                "ISOCENTER".to_string()
            } else if r.roi_type == "ISOCENTER" {
                "MARKER".to_string()
            } else {
                std::mem::take(&mut r.roi_type)
            };
        }
        self.settings_gen += 1;
    }

    /// The point behind a picked list entry, when it is a point of
    /// interest: what lets the comparison window measure two markers
    /// against each other instead of two empty masks.
    pub(super) fn poi_of_item(
        &self,
        slot: usize,
        item: combine_win::ItemRef,
    ) -> Option<(String, Vec3)> {
        if item.kind != SetKind::Structures {
            return None;
        }
        let r = self.slots[slot]
            .study
            .as_ref()?
            .structure_sets
            .get(item.set)?
            .rois
            .get(item.idx)?;
        r.point().map(|p| (r.name.clone(), p))
    }
}
