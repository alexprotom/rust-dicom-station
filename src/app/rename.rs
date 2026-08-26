//! Renaming anything the data tree shows: patients, studies, image series,
//! structure sets and segmentation series, individual structures and
//! segments, dose grids, plans, planar images, registrations and records.
//!
//! Every target addresses something inside one dataset's [`LoadedStudy`], so
//! the rename itself is a pure function of that study — which is what makes
//! it testable without a running UI, and what keeps the dialog down to a text
//! field and two buttons.
//!
//! Renames are in-memory: they change what the tree, the overlays and the 3D
//! view call things, and they are what a DICOM export writes out. The files
//! a study was loaded from are never touched.

use super::*;

/// What the rename dialog is editing.
#[derive(Clone, PartialEq)]
pub(super) enum RenameTarget {
    /// PatientName of every series of one patient (keyed by `patient_key`).
    Patient { slot: usize, key: String },
    /// StudyDescription of every series of one study.
    Study { slot: usize, uid: String },
    /// SeriesDescription of one image series.
    Series { slot: usize, idx: usize },
    /// Label of one structure set / segmentation series.
    Set(SetRef),
    /// Name of one structure / segment.
    Item { set: SetRef, idx: usize },
    /// Label of one dose grid.
    Dose { slot: usize, idx: usize },
    /// Label of one plan.
    Plan { slot: usize, idx: usize },
    /// Label of one planar image.
    Planar { slot: usize, idx: usize },
    /// Label of one REG spatial registration object.
    Registration { slot: usize, idx: usize },
    /// Label of one treatment record.
    Record { slot: usize, idx: usize },
}

impl RenameTarget {
    pub(super) fn slot(&self) -> usize {
        match self {
            RenameTarget::Patient { slot, .. }
            | RenameTarget::Study { slot, .. }
            | RenameTarget::Series { slot, .. }
            | RenameTarget::Dose { slot, .. }
            | RenameTarget::Plan { slot, .. }
            | RenameTarget::Planar { slot, .. }
            | RenameTarget::Registration { slot, .. }
            | RenameTarget::Record { slot, .. } => *slot,
            RenameTarget::Set(r) => r.slot,
            RenameTarget::Item { set, .. } => set.slot,
        }
    }

    /// What the dialog calls the thing being renamed.
    fn what(&self) -> &'static str {
        match self {
            RenameTarget::Patient { .. } => "patient",
            RenameTarget::Study { .. } => "study",
            RenameTarget::Series { .. } => "series",
            RenameTarget::Set(r) => r.kind.series_name(),
            RenameTarget::Item { set, .. } => set.kind.item_name(1),
            RenameTarget::Dose { .. } => "dose",
            RenameTarget::Plan { .. } => "plan",
            RenameTarget::Planar { .. } => "planar image",
            RenameTarget::Registration { .. } => "spatial registration",
            RenameTarget::Record { .. } => "treatment record",
        }
    }

    /// Which DICOM attribute the new text lands in — worth saying, because
    /// renaming a patient touches every series filed under them.
    fn attribute(&self) -> &'static str {
        match self {
            RenameTarget::Patient { .. } => "PatientName of every series of this patient",
            RenameTarget::Study { .. } => "StudyDescription of every series of this study",
            RenameTarget::Series { .. } => "SeriesDescription",
            RenameTarget::Set(SetRef {
                kind: SetKind::Structures,
                ..
            }) => "StructureSetLabel",
            RenameTarget::Set(_) => "SeriesDescription of the segmentation series",
            RenameTarget::Item { set, .. } => match set.kind {
                SetKind::Structures => "ROIName",
                SetKind::Segmentations => "SegmentLabel",
            },
            RenameTarget::Dose { .. } => "the dose label",
            RenameTarget::Plan { .. } => "RTPlanLabel",
            RenameTarget::Planar { .. } => "the image label",
            RenameTarget::Registration { .. } => "the registration label",
            RenameTarget::Record { .. } => "the record label",
        }
    }
}

/// The rename dialog: one text field over one target.
pub(super) struct RenameDialog {
    target: RenameTarget,
    text: String,
    /// Focus the text field on the first frame it is shown.
    focus: bool,
}

impl ViewerApp {
    /// What the target is currently called (`None` when it no longer exists).
    fn rename_current(study: &LoadedStudy, t: &RenameTarget) -> Option<String> {
        Some(match t {
            RenameTarget::Patient { key, .. } => study
                .series
                .iter()
                .find(|se| se.patient_key() == key)?
                .patient_name
                .clone(),
            RenameTarget::Study { uid, .. } => study
                .series
                .iter()
                .find(|se| se.study_uid == *uid)?
                .study_description
                .clone(),
            RenameTarget::Series { idx, .. } => study.series.get(*idx)?.description.clone(),
            RenameTarget::Set(r) => match r.kind {
                SetKind::Structures => study.structure_sets.get(r.idx)?.label.clone(),
                SetKind::Segmentations => study.seg_series.get(r.idx)?.label.clone(),
            },
            RenameTarget::Item { set, idx } => match set.kind {
                SetKind::Structures => study
                    .structure_sets
                    .get(set.idx)?
                    .rois
                    .get(*idx)?
                    .name
                    .clone(),
                SetKind::Segmentations => {
                    study.seg_series.get(set.idx)?.segs.get(*idx)?.name.clone()
                }
            },
            RenameTarget::Dose { idx, .. } => study.doses.get(*idx)?.label.clone(),
            RenameTarget::Plan { idx, .. } => study.plans.get(*idx)?.label.clone(),
            RenameTarget::Planar { idx, .. } => study.planar_images.get(*idx)?.label.clone(),
            RenameTarget::Registration { idx, .. } => study.registrations.get(*idx)?.label.clone(),
            RenameTarget::Record { idx, .. } => study.treat_records.get(*idx)?.label.clone(),
        })
    }

    /// Write `text` into the target. Returns whether anything was renamed.
    pub(super) fn rename_in_study(study: &mut LoadedStudy, t: &RenameTarget, text: &str) -> bool {
        let set_opt = |slot: Option<&mut String>| match slot {
            Some(s) => {
                *s = text.to_string();
                true
            }
            None => false,
        };
        match t {
            RenameTarget::Patient { key, .. } => {
                // A patient is a grouping over series, not an object — every
                // series filed under them has to follow, or the tree splits
                // into an old and a new patient node.
                let mut hit = false;
                for se in study.series.iter_mut() {
                    if se.patient_key() == key {
                        se.patient_name = text.to_string();
                        hit = true;
                    }
                }
                if study.meta.patient_id == *key || study.meta.patient_name == *key {
                    study.meta.patient_name = text.to_string();
                }
                hit
            }
            RenameTarget::Study { uid, .. } => {
                let mut hit = false;
                for se in study.series.iter_mut() {
                    if se.study_uid == *uid {
                        se.study_description = text.to_string();
                        hit = true;
                    }
                }
                // The study-level copy is what the export dialog pre-fills.
                if study
                    .series
                    .get(study.active_series)
                    .map(|se| se.study_uid.as_str())
                    == Some(uid.as_str())
                {
                    study.meta.study_description = text.to_string();
                }
                hit
            }
            RenameTarget::Series { idx, .. } => {
                set_opt(study.series.get_mut(*idx).map(|se| &mut se.description))
            }
            RenameTarget::Set(r) => match r.kind {
                SetKind::Structures => {
                    set_opt(study.structure_sets.get_mut(r.idx).map(|ss| &mut ss.label))
                }
                SetKind::Segmentations => {
                    set_opt(study.seg_series.get_mut(r.idx).map(|sr| &mut sr.label))
                }
            },
            RenameTarget::Item { set, idx } => match set.kind {
                SetKind::Structures => set_opt(
                    study
                        .structure_sets
                        .get_mut(set.idx)
                        .and_then(|ss| ss.rois.get_mut(*idx))
                        .map(|roi| &mut roi.name),
                ),
                SetKind::Segmentations => set_opt(
                    study
                        .seg_series
                        .get_mut(set.idx)
                        .and_then(|sr| sr.segs.get_mut(*idx))
                        .map(|seg| &mut seg.name),
                ),
            },
            RenameTarget::Dose { idx, .. } => {
                set_opt(study.doses.get_mut(*idx).map(|d| &mut d.label))
            }
            RenameTarget::Plan { idx, .. } => {
                set_opt(study.plans.get_mut(*idx).map(|p| &mut p.label))
            }
            RenameTarget::Planar { idx, .. } => {
                set_opt(study.planar_images.get_mut(*idx).map(|i| &mut i.label))
            }
            RenameTarget::Registration { idx, .. } => {
                set_opt(study.registrations.get_mut(*idx).map(|r| &mut r.label))
            }
            RenameTarget::Record { idx, .. } => {
                set_opt(study.treat_records.get_mut(*idx).map(|r| &mut r.label))
            }
        }
    }

    /// Open the dialog on a target, pre-filled with its current name.
    pub(super) fn open_rename(&mut self, target: RenameTarget) {
        let Some(study) = self.slots[target.slot()].study.as_ref() else {
            return;
        };
        let Some(text) = Self::rename_current(study, &target) else {
            return;
        };
        self.rename = Some(RenameDialog {
            target,
            text,
            focus: true,
        });
    }

    /// The rename dialog. Enter applies, Esc and ✕ cancel, and an empty name
    /// is simply not accepted — every one of these labels is what something
    /// is called somewhere else in the UI.
    pub(super) fn rename_window(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.rename.take() else {
            return;
        };
        let mut open = true;
        let mut apply = false;
        let mut cancel = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        egui::Window::new(format!("✎ Rename {}", d.target.what()))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(340.0);
                let te = ui.add(
                    egui::TextEdit::singleline(&mut d.text)
                        .desired_width(320.0)
                        .hint_text("new name"),
                );
                if d.focus {
                    te.request_focus();
                    d.focus = false;
                }
                if te.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    apply = true;
                }
                ui.weak(format!("Writes {}.", d.target.attribute()));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!d.text.trim().is_empty(), egui::Button::new("Rename"))
                        .clicked()
                    {
                        apply = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        if apply {
            let text = d.text.trim().to_string();
            if text.is_empty() {
                self.rename = Some(d);
                return;
            }
            let slot = d.target.slot();
            if let Some(study) = self.slots[slot].study.as_mut() {
                Self::rename_in_study(study, &d.target, &text);
            }
            self.settings_gen += 1;
            return;
        }
        if open && !cancel {
            self.rename = Some(d);
        }
    }
}

#[cfg(test)]
mod rename_tests {
    use super::*;
    use crate::dicomseg::SegSeries;
    use crate::geometry::Vec3;
    use crate::rtstruct::{Roi, StructureSet};

    fn series(uid: &str, patient: &str, study: &str) -> loader::SeriesInfo {
        loader::SeriesInfo {
            uid: uid.into(),
            modality: "CT".into(),
            description: "raw".into(),
            patient_id: patient.into(),
            patient_name: format!("{patient}^Name"),
            study_uid: study.into(),
            study_date: "20260826".into(),
            study_description: "before".into(),
            files: Vec::new(),
        }
    }

    fn study() -> LoadedStudy {
        let vol = Arc::new(Volume {
            data: vec![0],
            dims: [1, 1, 1],
            spacing: [1.0; 3],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        });
        let mut seg = SegSeries::new("Segs".into(), vol.grid(), "se1".into(), "st1".into());
        seg.segs
            .push(Segmentation::new("blob".into(), [1, 2, 3], vol.dims));
        LoadedStudy {
            meta: loader::PatientMeta {
                patient_name: "P1^Name".into(),
                patient_id: "P1".into(),
                study_date: "20260826".into(),
                study_description: "before".into(),
            },
            series: vec![
                series("se1", "P1", "st1"),
                series("se2", "P1", "st1"),
                series("se3", "P2", "st2"),
            ],
            active_series: 0,
            volume: vol,
            structure_sets: vec![StructureSet {
                label: "Set".into(),
                frame_of_reference_uid: String::new(),
                sop_instance_uid: "ss1".into(),
                study_uid: "st1".into(),
                referenced_series_uid: "se1".into(),
                file_name: "RS.dcm".into(),
                rois: vec![Roi {
                    number: 1,
                    name: "Liver".into(),
                    color: [1, 2, 3],
                    roi_type: "ORGAN".into(),
                    contours: Vec::new(),
                }],
            }],
            seg_series: vec![seg],
            doses: Vec::new(),
            plans: Vec::new(),
            planar_images: Vec::new(),
            registrations: Vec::new(),
            treat_records: Vec::new(),
            warnings: Vec::new(),
            default_window: (40.0, 400.0),
        }
    }

    /// A patient is a grouping, so the rename has to reach every series of
    /// that patient — and nobody else's.
    #[test]
    fn renaming_a_patient_moves_all_of_their_series() {
        let mut st = study();
        let t = RenameTarget::Patient {
            slot: 0,
            key: "P1".into(),
        };
        assert_eq!(
            ViewerApp::rename_current(&st, &t).as_deref(),
            Some("P1^Name")
        );
        assert!(ViewerApp::rename_in_study(&mut st, &t, "Doe^Jane"));
        assert_eq!(st.series[0].patient_name, "Doe^Jane");
        assert_eq!(st.series[1].patient_name, "Doe^Jane");
        assert_eq!(
            st.series[2].patient_name, "P2^Name",
            "other patient untouched"
        );
        assert_eq!(st.meta.patient_name, "Doe^Jane");
        // The grouping key is the ID, so both series stay one node.
        assert_eq!(st.series[0].patient_key(), st.series[1].patient_key());
    }

    #[test]
    fn renaming_a_study_reaches_its_series_and_the_export_defaults() {
        let mut st = study();
        let t = RenameTarget::Study {
            slot: 0,
            uid: "st1".into(),
        };
        assert!(ViewerApp::rename_in_study(&mut st, &t, "Planning CT"));
        assert_eq!(st.series[0].study_description, "Planning CT");
        assert_eq!(st.series[1].study_description, "Planning CT");
        assert_eq!(st.series[2].study_description, "before");
        assert_eq!(st.meta.study_description, "Planning CT");
    }

    #[test]
    fn series_sets_and_items_rename_in_place() {
        let mut st = study();
        let cases: Vec<(RenameTarget, &str)> = vec![
            (RenameTarget::Series { slot: 0, idx: 1 }, "CT thorax"),
            (
                RenameTarget::Set(SetRef {
                    slot: 0,
                    kind: SetKind::Structures,
                    idx: 0,
                }),
                "Approved",
            ),
            (
                RenameTarget::Item {
                    set: SetRef {
                        slot: 0,
                        kind: SetKind::Structures,
                        idx: 0,
                    },
                    idx: 0,
                },
                "Liver_edited",
            ),
            (
                RenameTarget::Set(SetRef {
                    slot: 0,
                    kind: SetKind::Segmentations,
                    idx: 0,
                }),
                "TotalSeg",
            ),
            (
                RenameTarget::Item {
                    set: SetRef {
                        slot: 0,
                        kind: SetKind::Segmentations,
                        idx: 0,
                    },
                    idx: 0,
                },
                "spleen",
            ),
        ];
        for (t, name) in &cases {
            assert!(ViewerApp::rename_in_study(&mut st, t, name), "{name}");
            assert_eq!(
                ViewerApp::rename_current(&st, t).as_deref(),
                Some(*name),
                "{name} reads back"
            );
        }
        assert_eq!(st.series[1].description, "CT thorax");
        assert_eq!(st.structure_sets[0].rois[0].name, "Liver_edited");
        assert_eq!(st.seg_series[0].segs[0].name, "spleen");
        // The first series was not the one renamed.
        assert_eq!(st.series[0].description, "raw");
    }

    /// A target that no longer exists must be reported, not panic.
    #[test]
    fn a_stale_target_is_refused() {
        let mut st = study();
        let gone = RenameTarget::Item {
            set: SetRef {
                slot: 0,
                kind: SetKind::Structures,
                idx: 0,
            },
            idx: 9,
        };
        assert!(ViewerApp::rename_current(&st, &gone).is_none());
        assert!(!ViewerApp::rename_in_study(&mut st, &gone, "x"));
        let no_dose = RenameTarget::Dose { slot: 0, idx: 0 };
        assert!(!ViewerApp::rename_in_study(&mut st, &no_dose, "x"));
    }
}
