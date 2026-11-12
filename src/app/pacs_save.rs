//! *Save to the PACS*: the structure sets and segmentation series of a
//! workspace written into this station's PACS, beside the study they were
//! drawn on, in the PACS's own folder layout (the patient's folder in its
//! cohort folder, whichever storage folder that is in).
//!
//! The window lists the workspace's objects, each with what the PACS knows
//! of it:
//!
//! - **new**: made here (drawn, painted, filled by a tool or a workflow) and
//!   never saved - ticked;
//! - **not in the PACS**: read from a file outside it - ticked;
//! - **in the PACS**: the PACS holds an object with its SOP Instance UID,
//!   as anything loaded from it and left alone does - not ticked; ticking
//!   it saves it again as a new object beside the old one (an edited set
//!   is saved that way, the old version staying);
//! - **empty**, and objects of a study the PACS does not hold: nothing to
//!   save, or nowhere to put it (import the study first).
//!
//! The writing is [`crate::archive::save::save_derived`] on a job. When it
//! lands, a new object takes on the identity of its copy in the PACS (SOP
//! and Series Instance UIDs, file name), so it is *in the PACS* from then
//! on and the window does not offer it a second time; an object saved again
//! from the PACS keeps the identity it was loaded with.
//! [`ViewerApp::open_pacs_save`] is the one way in: *File ▶ Save to the
//! PACS*, a set's menu in the data tree (that set ticked alone), and the
//! PACS window's *Send workspace*.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use crate::archive::save::Saved;
use crate::archive::Tree;
use crate::dicom_export::{self, DerivedKind};

use super::*;

/// What the PACS knows of one object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowState {
    New,
    Outside,
    InPacs,
    Empty,
    /// Its study is not in the PACS.
    NoStudy,
}

impl RowState {
    fn text(self) -> &'static str {
        match self {
            RowState::New => "new",
            RowState::Outside => "not in the PACS",
            RowState::InPacs => "in the PACS",
            RowState::Empty => "empty",
            RowState::NoStudy => "study not in the PACS",
        }
    }

    fn tip(self) -> &'static str {
        match self {
            RowState::New => "Made here and not saved yet",
            RowState::Outside => "Read from a file the PACS does not hold",
            RowState::InPacs => {
                "The PACS holds it already. Ticked, it is saved again as a new object \
                 beside the old one: how an edited set is kept, the old version staying"
            }
            RowState::Empty => "Nothing drawn in it: nothing to save",
            RowState::NoStudy => {
                "The PACS does not hold the study it belongs to; import the study first \
                 (PACS window, Import folder)"
            }
        }
    }

    fn savable(self) -> bool {
        !matches!(self, RowState::Empty | RowState::NoStudy)
    }
}

/// One structure set or segmentation series of the workspace.
#[derive(Clone, Debug)]
pub(super) struct SaveRow {
    pub kind: DerivedKind,
    pub index: usize,
    pub label: String,
    /// `12 structures`, `3 segments`.
    pub detail: String,
    pub study_uid: String,
    pub state: RowState,
    pub tick: bool,
}

/// The window's state.
pub(super) struct PacsSave {
    pub slot: usize,
    pub rows: Vec<SaveRow>,
    /// Where each study's objects go; `None`: the PACS does not hold it.
    pub homes: BTreeMap<String, Option<PathBuf>>,
    pub status: Option<String>,
}

/// The rows of `study`: what each object is, and what the PACS knows of it
/// (`held`: the SOP Instance UIDs it holds, `homes`: its folder for each
/// study). Ticked: what is not in the PACS yet.
pub(super) fn rows_of(
    study: &LoadedStudy,
    held: &HashSet<String>,
    homes: &BTreeMap<String, Option<PathBuf>>,
) -> Vec<SaveRow> {
    let state = |empty: bool, study_uid: &str, sop: &str, made_here: bool| {
        if empty {
            RowState::Empty
        } else if !homes.get(study_uid).is_some_and(Option::is_some) {
            RowState::NoStudy
        } else if !sop.is_empty() && held.contains(sop) {
            RowState::InPacs
        } else if made_here {
            RowState::New
        } else {
            RowState::Outside
        }
    };
    let mut rows = Vec::new();
    for (i, ss) in study.structure_sets.iter().enumerate() {
        let study_uid =
            dicom_export::derived_study_uid(study, &ss.study_uid, &ss.referenced_series_uid);
        let st = state(
            ss.rois.is_empty(),
            &study_uid,
            &ss.sop_instance_uid,
            !ss.read_from_file(),
        );
        rows.push(SaveRow {
            kind: DerivedKind::StructureSet,
            index: i,
            label: if ss.label.is_empty() {
                ss.file_name.clone()
            } else {
                ss.label.clone()
            },
            detail: plural(ss.rois.len(), "structure", "structures"),
            study_uid,
            state: st,
            tick: matches!(st, RowState::New | RowState::Outside),
        });
    }
    for (i, ser) in study.seg_series.iter().enumerate() {
        let study_uid =
            dicom_export::derived_study_uid(study, &ser.study_uid, &ser.referenced_series_uid);
        let filled = ser.segs.iter().filter(|s| s.count > 0).count();
        let st = state(
            filled == 0,
            &study_uid,
            &ser.sop_instance_uid,
            ser.file_name.is_empty(),
        );
        rows.push(SaveRow {
            kind: DerivedKind::Segmentation,
            index: i,
            label: ser.label.clone(),
            detail: plural(filled, "segment", "segments"),
            study_uid,
            state: st,
            tick: matches!(st, RowState::New | RowState::Outside),
        });
    }
    rows
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// What a finished save hands back: the workspace, what each saved object
/// was before (to know whose identity to take over), what was saved, and
/// the PACS listed afresh.
pub(super) type SaveDone = (usize, anyhow::Result<(Vec<SaveRow>, Saved, Tree)>);

impl ViewerApp {
    /// Open *Save to the PACS* on workspace `slot`; with `only`, that one
    /// object alone is ticked.
    pub(super) fn open_pacs_save(&mut self, slot: usize, only: Option<(DerivedKind, usize)>) {
        let Some(study) = self.slots.get(slot).and_then(|s| s.study.as_ref()) else {
            return;
        };
        let archive = self.station_pacs();
        let mut homes: BTreeMap<String, Option<PathBuf>> = BTreeMap::new();
        let uids = study
            .structure_sets
            .iter()
            .map(|s| dicom_export::derived_study_uid(study, &s.study_uid, &s.referenced_series_uid))
            .chain(study.seg_series.iter().map(|s| {
                dicom_export::derived_study_uid(study, &s.study_uid, &s.referenced_series_uid)
            }));
        for uid in uids {
            if let std::collections::btree_map::Entry::Vacant(e) = homes.entry(uid) {
                let home = archive.study_home(e.key());
                e.insert(home);
            }
        }
        let sops = study
            .structure_sets
            .iter()
            .map(|s| s.sop_instance_uid.as_str())
            .chain(study.seg_series.iter().map(|s| s.sop_instance_uid.as_str()))
            .filter(|s| !s.is_empty());
        let held = archive.held_sops(sops);
        let mut rows = rows_of(study, &held, &homes);
        if let Some((kind, index)) = only {
            for r in &mut rows {
                r.tick = r.kind == kind && r.index == index && r.state.savable();
            }
        }
        self.pacs_save = Some(PacsSave {
            slot,
            rows,
            homes,
            status: None,
        });
        self.raise_window("pacs_save");
    }

    pub(super) fn pacs_save_window(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.pacs_save.take() else {
            return;
        };
        let busy = self.pacs_save_job.is_some();
        let progress = self.pacs_save_job.as_ref().map(|j| j.progress.get());
        let mut open = true;
        let mut save = false;
        let summary = self.workspace_summary(d.slot);
        detach::tool_window(
            ctx,
            "pacs_save",
            "📤 Save to the PACS",
            &mut open,
            detach::WinOpts::width(520.0),
            |ui| {
                ui.label(
                    egui::RichText::new(format!("Workspace {} · {summary}", SLOT_NAMES[d.slot]))
                        .strong(),
                );
                ui.weak(
                    "Structure sets and segmentations, written beside the study they belong \
                     to in this station's PACS: new UIDs, the study's own Study and Frame of \
                     Reference UIDs.",
                );
                ui.separator();
                if d.rows.is_empty() {
                    ui.weak("This workspace has no structure sets or segmentations.");
                }
                egui::Grid::new("pacs_save_rows")
                    .num_columns(3)
                    .spacing([10.0, 4.0])
                    .show(ui, |ui| {
                        for r in &mut d.rows {
                            let glyph = match r.kind {
                                DerivedKind::StructureSet => "▣",
                                DerivedKind::Segmentation => "◉",
                            };
                            ui.add_enabled_ui(r.state.savable() && !busy, |ui| {
                                ui.checkbox(&mut r.tick, format!("{glyph} {}", r.label));
                            });
                            ui.weak(&r.detail);
                            let state = egui::RichText::new(r.state.text());
                            let state = if r.tick && r.state == RowState::InPacs {
                                state.color(ui.visuals().warn_fg_color)
                            } else if r.state.savable() {
                                state
                            } else {
                                state.weak()
                            };
                            ui.label(state).on_hover_text(r.state.tip());
                            ui.end_row();
                        }
                    });
                // Where they go, a line per study.
                let studies: Vec<&String> = d
                    .rows
                    .iter()
                    .filter(|r| r.tick)
                    .map(|r| &r.study_uid)
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                for uid in studies {
                    if let Some(Some(home)) = d.homes.get(uid) {
                        ui.horizontal_wrapped(|ui| {
                            ui.weak("Into");
                            ui.label(home.display().to_string());
                        });
                    }
                }
                if !d.rows.is_empty() && d.rows.iter().all(|r| r.state == RowState::NoStudy) {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "This station's PACS does not hold the study these belong to: \
                         import it first (PACS window, Import folder), then save.",
                    );
                }
                ui.separator();
                let n = d.rows.iter().filter(|r| r.tick).count();
                ui.horizontal(|ui| {
                    let label = format!("📤 Save {}", plural(n, "object", "objects"));
                    if ui
                        .add_enabled(n > 0 && !busy, egui::Button::new(label))
                        .on_hover_text("Write the ticked objects into the PACS")
                        .clicked()
                    {
                        save = true;
                    }
                    if let Some(p) = &progress {
                        ui.spinner();
                        ui.weak(p);
                    }
                });
                if let Some(s) = &d.status {
                    ui.label(s);
                }
            },
        );
        if save {
            self.start_pacs_save(&d);
        }
        if open {
            self.pacs_save = Some(d);
        }
    }

    fn start_pacs_save(&mut self, d: &PacsSave) {
        let Some(study) = self.slots[d.slot].study.as_ref() else {
            return;
        };
        let rows: Vec<SaveRow> = d.rows.iter().filter(|r| r.tick).cloned().collect();
        let keep: HashSet<(DerivedKind, usize)> = rows.iter().map(|r| (r.kind, r.index)).collect();
        let study = study.clone();
        let params = dicom_export::ExportParams::for_study(&study);
        let archive = self.station_pacs();
        let slot = d.slot;
        let progress = Arc::new(Progress::default());
        progress.set("Writing");
        self.pacs_save_job = Some(Job::spawn(progress, move |p| {
            let out = crate::archive::save::save_derived(&archive, &study, &params, p, &|k, i| {
                keep.contains(&(k, i))
            })
            .map(|saved| (rows, saved, archive.tree()));
            (slot, out)
        }));
    }

    /// A save landed: new objects take on their PACS identity, the window
    /// and the PACS window show what is there now.
    pub(super) fn on_pacs_saved(
        &mut self,
        slot: usize,
        rows: Vec<SaveRow>,
        saved: Saved,
        tree: Tree,
    ) {
        if let Some(study) = self.slots[slot].study.as_mut() {
            for o in &saved.objects {
                let Some(was) = rows.iter().find(|r| r.kind == o.kind && r.index == o.index) else {
                    continue;
                };
                // Only what was not in the PACS: one loaded from it keeps
                // the identity a plan or a registration may refer to.
                if was.state == RowState::InPacs {
                    continue;
                }
                let name = o
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                match o.kind {
                    DerivedKind::StructureSet => {
                        if let Some(ss) = study
                            .structure_sets
                            .get_mut(o.index)
                            .filter(|s| s.label == was.label || s.file_name == was.label)
                        {
                            ss.sop_instance_uid = o.sop_instance_uid.clone();
                            ss.series_instance_uid = o.series_instance_uid.clone();
                            ss.file_name = name;
                        }
                    }
                    DerivedKind::Segmentation => {
                        if let Some(ser) = study
                            .seg_series
                            .get_mut(o.index)
                            .filter(|s| s.label == was.label)
                        {
                            ser.sop_instance_uid = o.sop_instance_uid.clone();
                            ser.series_instance_uid = o.series_instance_uid.clone();
                            ser.file_name = name;
                        }
                    }
                }
            }
        }
        let n = saved.objects.len();
        let msg = if saved.summary.too_long > 0 {
            format!(
                "⚠ {} saved; {} left out: the path would pass the limit",
                plural(n, "object", "objects"),
                saved.summary.too_long
            )
        } else {
            format!("✔ {} saved into the PACS", plural(n, "object", "objects"))
        };
        if let Some(w) = self.pacs.as_mut() {
            w.status = Some(msg.clone());
            w.set_tree(tree);
        }
        if self.pacs_save.as_ref().is_some_and(|d| d.slot == slot) {
            self.open_pacs_save(slot, None);
            // Nothing new is left ticked; what was saved reads "in the PACS".
            if let Some(d) = self.pacs_save.as_mut() {
                d.status = Some(msg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dicomseg::SegSeries;
    use crate::rtstruct::StructureSet;

    #[test]
    fn what_is_new_is_ticked_and_what_the_pacs_holds_is_not() {
        let mut study = LoadedStudy::default();
        let set = |label: &str, file: &str, sop: &str, study: &str| StructureSet {
            label: label.into(),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: sop.into(),
            series_instance_uid: String::new(),
            study_uid: study.into(),
            referenced_series_uid: String::new(),
            file_name: file.into(),
            locked: false,
            rois: vec![crate::rtstruct::Roi {
                number: 1,
                name: "PTV".into(),
                color: [255, 0, 0],
                roi_type: String::new(),
                description: String::new(),
                contours: Vec::new(),
            }],
        };
        study.structure_sets = vec![
            set("Clinical", "RS1.dcm", "1.1", "S"),
            set("Drawn", "drawn-contours", "", "S"),
            set("Outside", "RS9.dcm", "9.9", "S"),
            set("Elsewhere", "drawn-contours", "", "T"),
        ];
        let mut empty = set("Empty", "drawn-contours", "", "S");
        empty.rois.clear();
        study.structure_sets.push(empty);
        let grid = crate::volume::Grid {
            dims: [1, 1, 1],
            spacing: [1.0, 1.0, 1.0],
            origin: crate::geometry::Vec3::ZERO,
            row_dir: crate::geometry::Vec3::new(1.0, 0.0, 0.0),
            col_dir: crate::geometry::Vec3::new(0.0, 1.0, 0.0),
            normal: crate::geometry::Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        };
        let mut seg = SegSeries::new("Painted".into(), grid, String::new(), "S".into());
        seg.segs.push(crate::segmentation::Segmentation::from_mask(
            "Ball".into(),
            [1, 2, 3],
            [1, 1, 1],
            vec![1],
        ));
        study.seg_series.push(seg);

        let held: HashSet<String> = ["1.1".to_string()].into();
        let homes: BTreeMap<String, Option<PathBuf>> = [
            ("S".to_string(), Some(PathBuf::from("pacs/Trial/P01"))),
            ("T".to_string(), None),
        ]
        .into();
        let rows = rows_of(&study, &held, &homes);
        let got: Vec<(&str, RowState, bool)> = rows
            .iter()
            .map(|r| (r.label.as_str(), r.state, r.tick))
            .collect();
        assert_eq!(
            got,
            vec![
                ("Clinical", RowState::InPacs, false),
                ("Drawn", RowState::New, true),
                ("Outside", RowState::Outside, true),
                ("Elsewhere", RowState::NoStudy, false),
                ("Empty", RowState::Empty, false),
                ("Painted", RowState::New, true),
            ]
        );
        assert_eq!(rows[5].kind, DerivedKind::Segmentation);
        assert_eq!(rows[5].detail, "1 segment");
    }
}
