//! Structure sets and segmentation series as data-tree nodes: creating them,
//! re-pointing them at an image series, moving whole series between workspaces,
//! and moving individual structures / segments between any two of them.
//!
//! The two kinds are deliberately symmetric. An RT structure set stores
//! contours in patient coordinates; a segmentation series stores voxel masks
//! on a lattice. A transfer between them is therefore a conversion -
//! [`segmentation::rasterize_roi`] one way, [`segmentation::mask_to_roi`] the
//! other - and a transfer between two segmentation series on different
//! lattices is a resampling. Doing it here, once, is what lets the context
//! menus offer every series of both workspaces as a destination without caring
//! which kind the user picked.

use super::*;

use crate::dicomseg::{self, SegSeries};
use crate::rtstruct::{Roi, StructureSet};
use crate::workflow::{self, group::Landing, select};

/// Which of a workspace's two kinds of segmented series an action addresses.
///
/// The data tree treats them alike - both are series drawn on an image
/// series, both hold named, coloured items - even though one stores contours
/// and the other voxel masks. Conversions between the two happen on
/// transfer (`ViewerApp::apply_item_action`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SetKind {
    /// RT Structure Set: contours.
    Structures,
    /// DICOM Segmentation series: binary voxel masks.
    Segmentations,
}

impl SetKind {
    /// "structure" / "segment", pluralized for `n`.
    pub(super) fn item_name(self, n: usize) -> &'static str {
        match (self, n) {
            (SetKind::Structures, 1) => "structure",
            (SetKind::Structures, _) => "structures",
            (SetKind::Segmentations, 1) => "segment",
            (SetKind::Segmentations, _) => "segments",
        }
    }
    pub(super) fn series_name(self) -> &'static str {
        match self {
            SetKind::Structures => "RT structure set",
            SetKind::Segmentations => "segmentation series",
        }
    }
}

/// One structure set / segmentation series of one workspace.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct SetRef {
    pub(super) slot: usize,
    pub(super) kind: SetKind,
    /// Index into that workspace's list, or [`SetRef::NEW`] for a series that
    /// does not exist yet - what the *New …* transfer destinations mean.
    pub(super) idx: usize,
}

impl SetRef {
    pub(super) const NEW: usize = usize::MAX;
}

/// Deferred right-click action on a whole series node of the data tree.
pub(super) enum SetAction {
    New(SetRef),
    Remove(SetRef),
    /// Open the rename dialog on the series.
    Rename(SetRef),
    /// Re-point the series at the image series with this Series Instance UID.
    Connect(SetRef, String),
    /// Copy (`copy`) or move the whole series to another workspace.
    Transfer {
        from: SetRef,
        to: usize,
        copy: bool,
    },
    /// Write a segmentation series as a standalone DICOM SEG file. An empty
    /// `items` means the whole series; otherwise only those segments.
    ExportSeg {
        set: SetRef,
        items: Vec<usize>,
    },
}

/// Deferred right-click action on individual structures / segments.
/// What the structure list's point-of-interest entries do.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PoiAct {
    /// Centre the views on the item.
    Localize,
    /// Move the point to where the views cross.
    MoveHere,
    /// Make this the structure set's localization point.
    Localization,
    /// A new point at the centre of gravity of this structure.
    AtCentre,
}

pub(super) enum ItemAction {
    /// Copy (`copy`) or move `items` of `from` into the series `to`.
    Transfer {
        from: SetRef,
        items: Vec<usize>,
        to: SetRef,
        copy: bool,
    },
    Remove {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Open the rename dialog on the clicked item alone.
    Rename {
        from: SetRef,
        idx: usize,
    },
    /// Open the structure-algebra window with these items as its operands.
    Combine {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Plot these items' dose-volume histograms.
    Dvh {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Carry these items into the other workspace through the active
    /// registration: open the propagation module aimed at them.
    Map {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Write these segments as a DICOM SEG file of their own.
    ExportSeg {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Copy (`copy`) or move `items` of `from` onto every phase of a 4D
    /// group, as they are, filed per phase the way `landing` says.
    ToPhases {
        from: SetRef,
        items: Vec<usize>,
        slot: usize,
        group: usize,
        landing: workflow::group::Landing,
        copy: bool,
    },
}

/// A destination of *Copy to* / *Move to*: one series, or every phase of a
/// 4D group at once.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Destination {
    Set(SetRef),
    Phases {
        slot: usize,
        group: usize,
        landing: workflow::group::Landing,
    },
}

impl Destination {
    /// The transfer of `items` of `from` here.
    pub(super) fn transfer(self, from: SetRef, items: Vec<usize>, copy: bool) -> ItemAction {
        match self {
            Destination::Set(to) => ItemAction::Transfer {
                from,
                items,
                to,
                copy,
            },
            Destination::Phases {
                slot,
                group,
                landing,
            } => ItemAction::ToPhases {
                from,
                items,
                slot,
                group,
                landing,
                copy,
            },
        }
    }
}

/// A *Copy to each phase* in flight: what was copied, where it goes and how
/// it is filed, kept until the phases are loaded and the copies made.
pub(super) struct PhasesCopy {
    pub(super) from: SetRef,
    pub(super) items: Vec<usize>,
    pub(super) slot: usize,
    pub(super) group_name: String,
    pub(super) landing: workflow::group::Landing,
    pub(super) copy: bool,
}

impl ViewerApp {
    // -- Series nodes ------------------------------------------------------

    /// Create an empty structure set / segmentation series bound to the
    /// slot's displayed image series, make it the active one and return its
    /// index.
    pub(super) fn new_set(&mut self, slot: usize, kind: SetKind) -> Option<usize> {
        // A new set is bound to the displayed image series' lattice; with
        // no volume there is nothing to bind it to.
        if !self.slots[slot].has_volume() {
            return None;
        }
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
                    series_instance_uid: crate::dicom_export::new_uid(),
                    study_uid,
                    referenced_series_uid: uid,
                    file_name: String::new(),
                    locked: false,
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
            SetAction::Transfer { from, to, copy } => self.transfer_set(from, to, copy),
            SetAction::ExportSeg { set, items } => self.export_seg_series(set, &items),
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

    /// Re-point a series at another image series of the same workspace.
    ///
    /// For contours this is bookkeeping - they are in patient coordinates
    /// either way. For a segmentation series it also decides which volume
    /// its masks are resampled onto, which is why the rebind follows.
    fn connect_set(&mut self, r: SetRef, uid: &str) {
        {
            let Some(study) = self.slots[r.slot].study.as_mut() else {
                return;
            };
            // An empty uid ties a segmentation series to no image series:
            // it then follows any image of its frame of reference.
            let study_uid = if uid.is_empty() {
                if r.kind != SetKind::Segmentations {
                    return;
                }
                String::new()
            } else {
                let Some(se) = study.series.iter().find(|se| se.uid == uid) else {
                    return;
                };
                se.study_uid.clone()
            };
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

    /// Copy / move one whole series to another workspace.
    fn transfer_set(&mut self, from: SetRef, to: usize, copy: bool) {
        if self.slots[to].study.is_none() {
            self.error = Some(format!(
                "workspace {} is empty - load a study into it before moving series there",
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
        self.show_workspace(to);
        self.rebind_seg_series(to);
        self.settings_gen += 1;
    }

    /// Write a segmentation series - or just the chosen segments of it - as a
    /// standalone DICOM SEG file.
    ///
    /// A subset is written as a series in its own right: same lattice, same
    /// referenced image series, its own SOP Instance UID, and only the
    /// segments asked for. That is exactly what a SEG file is, so exporting
    /// three organs out of twelve needs no special case in the writer.
    fn export_seg_series(&mut self, r: SetRef, items: &[usize]) {
        let Some(study) = self.slots[r.slot].study.as_ref() else {
            return;
        };
        let Some(full) = study.seg_series.get(r.idx) else {
            return;
        };
        // Only the name and the emptiness check are needed before the
        // dialog; the series itself is derived when there is a path.
        let mut chosen: Vec<usize> = if items.is_empty() {
            (0..full.segs.len()).collect()
        } else {
            items.to_vec()
        };
        chosen.sort_unstable();
        chosen.dedup();
        chosen.retain(|&i| i < full.segs.len());
        if chosen.iter().all(|&i| full.segs[i].count == 0) {
            self.error =
                Some("nothing to write - the chosen segmentation(s) have no voxels".into());
            return;
        }
        let stem: String = full
            .label
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { '_' })
            .collect();
        let suffix = if items.is_empty() {
            String::new()
        } else {
            format!("_{}_of_{}", chosen.len(), full.segs.len())
        };
        let items = items.to_vec();
        self.ask_save(
            if items.is_empty() {
                "Save the segmentation series as DICOM SEG"
            } else {
                "Save the selected segmentation(s) as DICOM SEG"
            },
            format!("SEG_{stem}{suffix}.dcm"),
            None,
            None,
            move |app, path| app.write_seg_series(r, &items, &path),
        );
    }

    /// The second half of [`Self::export_seg_series`]: the series is derived
    /// again from the study, so nothing has to be held across the dialog.
    fn write_seg_series(&mut self, r: SetRef, items: &[usize], path: &std::path::Path) {
        let Some(study) = self.slots[r.slot].study.as_ref() else {
            return;
        };
        let Some(full) = study.seg_series.get(r.idx) else {
            return;
        };
        let (ser, subset) = seg_to_write(full, items);
        let ser = subset.as_ref().unwrap_or(ser);
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
        match dicomseg::write(ser, &ctx, path) {
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
            ItemAction::ExportSeg { from, items } => self.export_seg_series(from, &items),
            ItemAction::Combine { from, items } => {
                // The tree already knows which items were ticked; the window
                // only has to be told, in the order they were listed.
                let seed = items
                    .iter()
                    .map(|&idx| combine::ItemRef {
                        kind: from.kind,
                        set: from.idx,
                        idx,
                    })
                    .collect();
                self.open_combine_dialog(from.slot, seed);
            }
            ItemAction::Dvh { from, items } => {
                let seed = items
                    .iter()
                    .map(|&idx| combine::ItemRef {
                        kind: from.kind,
                        set: from.idx,
                        idx,
                    })
                    .collect();
                self.open_dvh_dialog(from.slot, seed);
            }
            ItemAction::Map { from, items } => self.map_items_across(from, &items),
            ItemAction::Transfer {
                from,
                items,
                to,
                copy,
            } => self.transfer_items(from, &items, to, copy),
            ItemAction::ToPhases {
                from,
                items,
                slot,
                group,
                landing,
                copy,
            } => self.transfer_to_phases(from, &items, slot, group, landing, copy),
        }
    }

    /// Start copying structures / segments onto every phase of a 4D group:
    /// the phases are loaded on a worker for their lattices and the
    /// structures resampled onto each; [`Self::on_phases_copied`] files
    /// the copies.
    fn transfer_to_phases(
        &mut self,
        from: SetRef,
        items: &[usize],
        slot: usize,
        group: usize,
        landing: Landing,
        copy: bool,
    ) {
        if self.phases_job.is_some() {
            self.error = Some("a copy onto the phases of a 4D group is still running".into());
            return;
        }
        let mut items = items.to_vec();
        items.sort_unstable();
        items.dedup();
        let structures: Vec<select::Structure> = {
            let Some(study) = self.slots[from.slot].study.as_ref() else {
                return;
            };
            match from.kind {
                SetKind::Structures => match study.structure_sets.get(from.idx) {
                    Some(ss) => items
                        .iter()
                        .filter_map(|&i| ss.rois.get(i))
                        .map(select::Structure::from_roi)
                        .collect(),
                    None => return,
                },
                SetKind::Segmentations => match study.seg_series.get(from.idx) {
                    Some(sr) => items
                        .iter()
                        .filter_map(|&i| select::Structure::from_segment(sr, i))
                        .collect(),
                    None => return,
                },
            }
        };
        if structures.is_empty() {
            return;
        }
        let (group_name, phases) = {
            let Some(study) = self.slots[slot].study.as_ref() else {
                return;
            };
            let Some(g) = study.fourd_groups.get(group) else {
                return;
            };
            match workflow::phases_of(g, &study.series) {
                Ok(p) => (g.name.clone(), p),
                Err(e) => {
                    self.error = Some(format!("{e:#}"));
                    return;
                }
            }
        };
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let job = Job::spawn(progress, move |p| {
            workflow::group::copy_to_phases(&structures, &phases, p)
        });
        self.phases_job = Some((
            PhasesCopy {
                from,
                items,
                slot,
                group_name,
                landing,
                copy,
            },
            job,
        ));
    }

    /// File the copies a [`Self::transfer_to_phases`] made: into the
    /// segmentation series already bound to each phase (or a new one), or
    /// as contours into each phase's own RT structure set - through
    /// [`Self::land_phases`], the same landing an engine run over the
    /// phases uses.
    pub(super) fn on_phases_copied(
        &mut self,
        what: PhasesCopy,
        phases: Vec<workflow::group::PhaseCopy>,
    ) {
        let kind = match what.landing {
            Landing::Segmentation => OutputKind::Segments,
            Landing::StructureSet => OutputKind::Structures,
        };
        let landed = self.land_phases(what.slot, &what.group_name, kind, "GTV", phases);
        if landed == 0 {
            return;
        }
        if !what.copy {
            self.remove_items(what.from, &what.items);
        }
    }

    /// Delete structures / segments from their series.
    fn remove_items(&mut self, from: SetRef, items: &[usize]) {
        if from.kind == SetKind::Structures && self.set_locked(from.slot, from.idx) {
            self.locked_notice(from.slot);
            return;
        }
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

    /// Copy / move structures or segments into any series of either workspace,
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
                        "workspace {} is empty - load a study into it first",
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
                        notes.push(format!("'{}' is empty - nothing was copied", seg.name));
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
            self.show_workspace(to.slot);
        }
        self.settings_gen += 1;
    }
}

/// What the structure list's *Template* menu asks for.
#[derive(Clone, PartialEq)]
pub(super) enum TemplateAct {
    /// Save the structure set at this index as a template of its own name.
    Save(usize),
    /// Create the structures of this template that the set at this index
    /// does not have yet.
    Apply(usize, String),
}

impl ViewerApp {
    pub(super) fn apply_template_act(&mut self, slot: usize, act: TemplateAct) {
        match act {
            TemplateAct::Save(set) => {
                let Some(ss) = self.slots[slot]
                    .study
                    .as_ref()
                    .and_then(|st| st.structure_sets.get(set))
                else {
                    return;
                };
                let t = crate::templates::of_set(&ss.label, ss);
                let n = t.rois.len();
                match crate::templates::save(&t) {
                    Ok(()) => {
                        self.notice = Some(format!(
                            "Saved '{}' as a template of {n} structure(s), in {}.",
                            t.name,
                            crate::templates::dir().display()
                        ))
                    }
                    Err(e) => self.error = Some(format!("Could not save the template: {e:#}")),
                }
            }
            TemplateAct::Apply(set, name) => {
                if self.set_locked(slot, set) {
                    self.locked_notice(slot);
                    return;
                }
                let t = match crate::templates::load(&name) {
                    Ok(t) => t,
                    Err(e) => {
                        self.error = Some(format!("Could not read the template: {e:#}"));
                        return;
                    }
                };
                let Some(ss) = self.slots[slot]
                    .study
                    .as_mut()
                    .and_then(|st| st.structure_sets.get_mut(set))
                else {
                    return;
                };
                let missing = crate::templates::missing_rois(&t, ss);
                let n = missing.len();
                let skipped = t.rois.len() - n;
                ss.rois.extend(missing);
                let total = ss.rois.len();
                self.slots[slot].roi_visible.resize(total, true);
                self.settings_gen += 1;
                self.notice = Some(match skipped {
                    0 => format!("'{}' added {n} structure(s).", t.name),
                    _ => format!(
                        "'{}' added {n} structure(s); {skipped} were already here.",
                        t.name
                    ),
                });
            }
        }
    }
}

/// The series to write for `items`: the whole of `full`, or a subset that is
/// a series in its own right - same lattice, same referenced image series,
/// its own SOP Instance UID, and only the segments asked for. The subset is
/// returned owned beside the reference so a caller can use either.
fn seg_to_write<'a>(full: &'a SegSeries, items: &[usize]) -> (&'a SegSeries, Option<SegSeries>) {
    if items.is_empty() {
        return (full, None);
    }
    let mut picked = items.to_vec();
    picked.sort_unstable();
    picked.dedup();
    let mut s = full.clone();
    s.segs = picked
        .iter()
        .filter_map(|&i| full.segs.get(i).cloned())
        .collect();
    s.sop_instance_uid = dicom_export::new_uid();
    (full, Some(s))
}
