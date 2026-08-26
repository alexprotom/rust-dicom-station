//! Structure sets and segmentation series as data-tree nodes: creating them,
//! re-pointing them at an image series, moving whole series between datasets,
//! and moving individual structures / segments between any two of them.
//!
//! The two kinds are deliberately symmetric. An RT structure set stores
//! contours in patient coordinates; a segmentation series stores voxel masks
//! on a lattice. A transfer between them is therefore a conversion —
//! [`segmentation::rasterize_roi`] one way, [`segmentation::mask_to_roi`] the
//! other — and a transfer between two segmentation series on different
//! lattices is a resampling. Doing it here, once, is what lets the context
//! menus offer every series of both datasets as a destination without caring
//! which kind the user picked.

use super::*;

use crate::dicomseg::{self, SegSeries};
use crate::rtstruct::{Roi, StructureSet};

impl ViewerApp {
    // -- Series nodes ------------------------------------------------------

    /// Create an empty structure set / segmentation series bound to the
    /// slot's displayed image series, make it the active one and return its
    /// index.
    pub(super) fn new_set(&mut self, slot: usize, kind: SetKind) -> Option<usize> {
        let s = &mut self.slots[slot];
        let study = s.study.as_mut()?;
        let se = study.series.get(study.active_series);
        let uid = se.map(|x| x.uid.clone()).unwrap_or_default();
        let study_uid = se.map(|x| x.study_uid.clone()).unwrap_or_default();
        let for_uid = study.volume.frame_of_reference_uid.clone();
        let grid = study.volume.grid();
        match kind {
            SetKind::Structures => {
                study.structure_sets.push(StructureSet {
                    label: format!("Structure Set {}", study.structure_sets.len() + 1),
                    frame_of_reference_uid: for_uid,
                    sop_instance_uid: crate::dicom_export::new_uid(),
                    study_uid,
                    referenced_series_uid: uid,
                    file_name: String::new(),
                    rois: Vec::new(),
                });
                s.active_structs = study.structure_sets.len() - 1;
                s.roi_visible = Vec::new();
                Some(s.active_structs)
            }
            SetKind::Segmentations => {
                study.seg_series.push(SegSeries::new(
                    format!("Segmentations {}", study.seg_series.len() + 1),
                    grid,
                    uid,
                    study_uid,
                ));
                s.active_seg_series = study.seg_series.len() - 1;
                s.active_seg = 0;
                Some(s.active_seg_series)
            }
        }
    }

    pub(super) fn apply_set_action(&mut self, act: SetAction) {
        match act {
            SetAction::New(r) => {
                self.new_set(r.slot, r.kind);
                self.settings_gen += 1;
            }
            SetAction::Remove(r) => self.remove_set(r),
            SetAction::Rename(r) => self.rename_request = Some(RenameTarget::Set(r)),
            SetAction::Connect(r, uid) => self.connect_set(r, &uid),
            SetAction::Transfer { from, copy } => self.transfer_set(from, copy),
            SetAction::ExportSeg(r) => self.export_seg_series(r),
        }
    }

    /// Drop a whole series, keeping the slot's active selections valid.
    fn remove_set(&mut self, r: SetRef) {
        let StudySlot {
            study,
            roi_visible,
            active_structs,
            active_seg_series,
            active_seg,
            ..
        } = &mut self.slots[r.slot];
        let Some(study) = study.as_mut() else { return };
        match r.kind {
            SetKind::Structures => {
                if r.idx >= study.structure_sets.len() {
                    return;
                }
                study.structure_sets.remove(r.idx);
                if r.idx < *active_structs {
                    *active_structs -= 1;
                } else if r.idx == *active_structs {
                    *active_structs =
                        (*active_structs).min(study.structure_sets.len().saturating_sub(1));
                    let n = study
                        .structure_sets
                        .get(*active_structs)
                        .map(|ss| ss.rois.len())
                        .unwrap_or(0);
                    *roi_visible = vec![true; n];
                }
            }
            SetKind::Segmentations => {
                if r.idx >= study.seg_series.len() {
                    return;
                }
                study.seg_series.remove(r.idx);
                if r.idx < *active_seg_series {
                    *active_seg_series -= 1;
                } else if r.idx == *active_seg_series {
                    *active_seg_series =
                        (*active_seg_series).min(study.seg_series.len().saturating_sub(1));
                    *active_seg = 0;
                }
            }
        }
        self.settings_gen += 1;
    }

    /// Re-point a series at another image series of the same dataset.
    ///
    /// For contours this is bookkeeping — they are in patient coordinates
    /// either way. For a segmentation series it also decides which volume
    /// its masks are resampled onto, which is why the rebind follows.
    fn connect_set(&mut self, r: SetRef, uid: &str) {
        {
            let Some(study) = self.slots[r.slot].study.as_mut() else {
                return;
            };
            let Some(se) = study.series.iter().find(|se| se.uid == uid) else {
                return;
            };
            let study_uid = se.study_uid.clone();
            match r.kind {
                SetKind::Structures => {
                    let Some(ss) = study.structure_sets.get_mut(r.idx) else {
                        return;
                    };
                    ss.referenced_series_uid = uid.to_string();
                    if !study_uid.is_empty() {
                        ss.study_uid = study_uid;
                    }
                }
                SetKind::Segmentations => {
                    let Some(sr) = study.seg_series.get_mut(r.idx) else {
                        return;
                    };
                    sr.referenced_series_uid = uid.to_string();
                    if !study_uid.is_empty() {
                        sr.study_uid = study_uid;
                    }
                }
            }
        }
        if r.kind == SetKind::Segmentations {
            self.rebind_seg_series(r.slot);
        }
        self.settings_gen += 1;
    }

    /// Copy / move one whole series to the other dataset.
    fn transfer_set(&mut self, from: SetRef, copy: bool) {
        let to = 1 - from.slot;
        if self.slots[to].study.is_none() {
            self.error = Some(format!(
                "dataset {} is empty — load a study into it before moving series there",
                SLOT_NAMES[to]
            ));
            return;
        }
        let taken: Option<(Option<StructureSet>, Option<SegSeries>)> = {
            let Some(study) = self.slots[from.slot].study.as_ref() else {
                return;
            };
            match from.kind {
                SetKind::Structures => study
                    .structure_sets
                    .get(from.idx)
                    .map(|ss| (Some(ss.clone()), None)),
                SetKind::Segmentations => study
                    .seg_series
                    .get(from.idx)
                    .map(|sr| (None, Some(sr.clone()))),
            }
        };
        let Some((ss, sr)) = taken else { return };
        {
            let Some(dst) = self.slots[to].study.as_mut() else {
                return;
            };
            if let Some(ss) = ss {
                dst.structure_sets.push(ss);
            }
            if let Some(sr) = sr {
                dst.seg_series.push(sr);
            }
        }
        if !copy {
            self.remove_set(from);
        }
        self.comparison = true;
        self.rebind_seg_series(to);
        self.settings_gen += 1;
    }

    /// Write one segmentation series as a standalone DICOM SEG file.
    fn export_seg_series(&mut self, r: SetRef) {
        let Some(study) = self.slots[r.slot].study.as_ref() else {
            return;
        };
        let Some(ser) = study.seg_series.get(r.idx) else {
            return;
        };
        if ser.segs.iter().all(|s| s.count == 0) {
            self.error = Some("this segmentation series is empty — nothing to write".into());
            return;
        }
        let stem: String = ser
            .label
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save the segmentation series as DICOM SEG")
            .set_file_name(format!("SEG_{stem}.dcm"))
            .save_file()
        else {
            return;
        };
        let params = dicom_export::ExportParams::for_study(study);
        let (date, time) = dicom_export::today();
        let study_uid = if ser.study_uid.is_empty() {
            study
                .series
                .first()
                .map(|s| s.study_uid.clone())
                .unwrap_or_default()
        } else {
            ser.study_uid.clone()
        };
        let for_uid = if ser.grid.frame_of_reference_uid.is_empty() {
            study.volume.frame_of_reference_uid.clone()
        } else {
            ser.grid.frame_of_reference_uid.clone()
        };
        let ctx = dicomseg::SegWriteCtx {
            study_uid: &study_uid,
            for_uid: &for_uid,
            date: &date,
            time: &time,
            series_number: 20 + r.idx as i64,
            image_series_uid: &ser.referenced_series_uid,
            image_sop_uids: &[],
            params: &params,
        };
        match dicomseg::write(ser, &ctx, &path) {
            Ok(()) => {
                self.notice = Some(format!(
                    "✔ {} segment(s) written to {}",
                    ser.segs.len(),
                    path.display()
                ))
            }
            Err(e) => self.error = Some(format!("Writing the SEG file failed: {e:#}")),
        }
    }

    // -- Individual structures and segments --------------------------------

    pub(super) fn apply_item_action(&mut self, act: ItemAction) {
        match act {
            ItemAction::Remove { from, items } => self.remove_items(from, &items),
            ItemAction::Rename { from, idx } => {
                self.rename_request = Some(RenameTarget::Item { set: from, idx })
            }
            ItemAction::Transfer {
                from,
                items,
                to,
                copy,
            } => self.transfer_items(from, &items, to, copy),
        }
    }

    /// Delete structures / segments from their series.
    fn remove_items(&mut self, from: SetRef, items: &[usize]) {
        let mut items = items.to_vec();
        items.sort_unstable();
        items.dedup();
        let StudySlot {
            study,
            roi_visible,
            active_structs,
            active_seg,
            ..
        } = &mut self.slots[from.slot];
        let Some(study) = study.as_mut() else { return };
        match from.kind {
            SetKind::Structures => {
                let Some(ss) = study.structure_sets.get_mut(from.idx) else {
                    return;
                };
                for &i in items.iter().rev() {
                    if i < ss.rois.len() {
                        ss.rois.remove(i);
                        if from.idx == *active_structs && i < roi_visible.len() {
                            roi_visible.remove(i);
                        }
                    }
                }
                if from.idx == *active_structs {
                    roi_visible.resize(ss.rois.len(), true);
                }
            }
            SetKind::Segmentations => {
                let Some(sr) = study.seg_series.get_mut(from.idx) else {
                    return;
                };
                for &i in items.iter().rev() {
                    if i < sr.segs.len() {
                        sr.segs.remove(i);
                    }
                }
                if *active_seg >= sr.segs.len() {
                    *active_seg = sr.segs.len().saturating_sub(1);
                }
            }
        }
        self.settings_gen += 1;
    }

    /// Copy / move structures or segments into any series of either dataset,
    /// converting between contours and masks where the two kinds differ.
    fn transfer_items(&mut self, from: SetRef, items: &[usize], to: SetRef, copy: bool) {
        if from == to {
            return;
        }
        let mut items = items.to_vec();
        items.sort_unstable();
        items.dedup();
        if items.is_empty() {
            return;
        }

        // ---- read the source ------------------------------------------
        let (src_rois, src_segs, src_grid) = {
            let Some(study) = self.slots[from.slot].study.as_ref() else {
                return;
            };
            match from.kind {
                SetKind::Structures => {
                    let Some(ss) = study.structure_sets.get(from.idx) else {
                        return;
                    };
                    (
                        items
                            .iter()
                            .filter_map(|&i| ss.rois.get(i).cloned())
                            .collect::<Vec<Roi>>(),
                        Vec::new(),
                        study.volume.grid(),
                    )
                }
                SetKind::Segmentations => {
                    let Some(sr) = study.seg_series.get(from.idx) else {
                        return;
                    };
                    (
                        Vec::new(),
                        items
                            .iter()
                            .filter_map(|&i| sr.segs.get(i).cloned())
                            .collect::<Vec<Segmentation>>(),
                        sr.grid.clone(),
                    )
                }
            }
        };

        // ---- resolve (and possibly create) the destination -------------
        let mut to = to;
        if to.idx == SetRef::NEW {
            match self.new_set(to.slot, to.kind) {
                Some(i) => to.idx = i,
                None => {
                    self.error = Some(format!(
                        "dataset {} is empty — load a study into it first",
                        SLOT_NAMES[to.slot]
                    ));
                    return;
                }
            }
        }
        let dst_grid = {
            let Some(study) = self.slots[to.slot].study.as_ref() else {
                return;
            };
            match to.kind {
                SetKind::Structures => match study.structure_sets.get(to.idx) {
                    Some(_) => study.volume.grid(),
                    None => return,
                },
                SetKind::Segmentations => match study.seg_series.get(to.idx) {
                    Some(sr) => sr.grid.clone(),
                    None => return,
                },
            }
        };

        // ---- convert ----------------------------------------------------
        let mut notes: Vec<String> = Vec::new();
        let mut new_rois: Vec<Roi> = Vec::new();
        let mut new_segs: Vec<Segmentation> = Vec::new();
        match to.kind {
            SetKind::Structures => {
                new_rois.extend(src_rois.iter().cloned());
                for seg in &src_segs {
                    let roi = segmentation::mask_to_roi(seg, &src_grid, 0);
                    if roi.contours.is_empty() {
                        notes.push(format!("'{}' is empty — nothing was copied", seg.name));
                        continue;
                    }
                    new_rois.push(roi);
                }
            }
            SetKind::Segmentations => {
                for roi in &src_rois {
                    match segmentation::rasterize_roi(&dst_grid, roi) {
                        Some(mask) => new_segs.push(Segmentation::from_mask(
                            roi.name.clone(),
                            roi.color,
                            dst_grid.dims,
                            mask,
                        )),
                        None => notes.push(format!(
                            "'{}' has no contour inside the destination volume",
                            roi.name
                        )),
                    }
                }
                for seg in &src_segs {
                    let mask = if src_grid.matches(&dst_grid) {
                        seg.mask.clone()
                    } else {
                        dicomseg::resample_mask(&seg.mask, &src_grid, &dst_grid)
                    };
                    let made =
                        Segmentation::from_mask(seg.name.clone(), seg.color, dst_grid.dims, mask);
                    if made.count == 0 {
                        notes.push(format!(
                            "'{}' does not overlap the destination volume",
                            seg.name
                        ));
                        continue;
                    }
                    new_segs.push(made);
                }
            }
        }
        if new_rois.is_empty() && new_segs.is_empty() {
            self.error = Some(if notes.is_empty() {
                "nothing to transfer".into()
            } else {
                notes.join("\n")
            });
            return;
        }

        // ---- insert ------------------------------------------------------
        {
            let StudySlot {
                study,
                roi_visible,
                active_structs,
                ..
            } = &mut self.slots[to.slot];
            let Some(study) = study.as_mut() else { return };
            match to.kind {
                SetKind::Structures => {
                    let Some(ss) = study.structure_sets.get_mut(to.idx) else {
                        return;
                    };
                    let mut number = ss.rois.iter().map(|r| r.number).max().unwrap_or(0);
                    for mut roi in new_rois {
                        number += 1;
                        roi.number = number;
                        ss.rois.push(roi);
                    }
                    if to.idx == *active_structs {
                        roi_visible.resize(ss.rois.len(), true);
                    }
                }
                SetKind::Segmentations => {
                    let Some(sr) = study.seg_series.get_mut(to.idx) else {
                        return;
                    };
                    sr.segs.extend(new_segs);
                }
            }
            study.warnings.extend(notes);
        }

        if !copy {
            self.remove_items(from, &items);
        }
        if to.slot != from.slot {
            self.comparison = true;
        }
        self.settings_gen += 1;
    }
}
