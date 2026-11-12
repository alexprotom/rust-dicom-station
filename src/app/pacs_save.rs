//! *Save to the PACS*: the structure sets and segmentation series of a
//! workspace written into this station's PACS, beside the study they were
//! drawn on, in the PACS's own folder layout (beside the image series an
//! object references, in the patient's folder of its cohort folder,
//! whichever storage folder that is in).
//!
//! The window lists the workspace's objects, each with what the PACS knows
//! of it:
//!
//! - **new**: made here (drawn, painted, filled by a tool or a workflow) and
//!   never saved - ticked;
//! - **not in the PACS**: read from a file outside it - ticked;
//! - **changed**: the PACS holds an object with its SOP Instance UID, and it
//!   was edited here since it was loaded or last saved - ticked; saved as a
//!   new object beside the old one, the old version staying;
//! - **in the PACS**: the PACS holds it, and it is as it was loaded or
//!   saved - not ticked; ticking it saves it again all the same;
//! - **empty**, and objects of a study the PACS does not hold: nothing to
//!   save, or nowhere to put it (import the study first);
//! - **reading the PACS**, **PACS not read yet**: while the catalog is
//!   looked up, and when there is none (a PACS never scanned).
//!
//! *Changed* is told by an edit fingerprint kept on the workspace
//! ([`super::slot::StudySlot::saved_marks`]): of every object as it was
//! loaded, and of every object saved, as it was saved. A structure set's is
//! a hash of its label, references and ROIs (names, numbers, colours,
//! types, every contour point), so a rename, a removal and any tool's edit
//! move it alike; a segmentation series' takes each segment's generation,
//! which every edit of a mask moves.
//!
//! What the PACS holds is read from its catalog on a job
//! ([`crate::archive::Archive::save_targets`]: one load of the catalog, one
//! pass; never a scan), so the window opens at once and fills in. The
//! writing is [`crate::archive::save::save_derived`] on another job, which
//! reads the targets afresh after writing. When it lands, a new object
//! takes on the identity of its copy in the PACS (SOP and Series Instance
//! UIDs, file name), so it is *in the PACS* from then on and the window
//! does not offer it a second time; an object saved again from the PACS
//! keeps the identity it was loaded with (a plan or a registration may
//! refer to it), and its fingerprint moves, so it reads *in the PACS* until
//! the next edit. A save that stops half way keeps what it filed: those
//! objects take on their identity, the window says what stopped the rest,
//! and the rest stay ticked. [`ViewerApp::open_pacs_save`] is the one way
//! in: *File ▶ Save to the PACS*, a set's menu in the data tree (that set
//! ticked alone), and the PACS window's *Send workspace*.

use std::collections::{BTreeSet, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use crate::archive::save::Saved;
use crate::archive::{SaveTargets, Tree};
use crate::dicom_export::{self, DerivedKind};
use crate::dicomseg::SegSeries;
use crate::rtstruct::StructureSet;

use super::*;

/// What the PACS knows of one object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RowState {
    New,
    Outside,
    /// In the PACS, changed here since it was loaded or saved.
    Changed,
    InPacs,
    Empty,
    /// Its study is not in the PACS.
    NoStudy,
    /// The catalog is being looked up.
    Reading,
    /// The PACS has no catalog yet.
    NotRead,
}

impl RowState {
    fn text(self) -> &'static str {
        match self {
            RowState::New => "new",
            RowState::Outside => "not in the PACS",
            RowState::Changed => "changed",
            RowState::InPacs => "in the PACS",
            RowState::Empty => "empty",
            RowState::NoStudy => "study not in the PACS",
            RowState::Reading => "reading the PACS",
            RowState::NotRead => "PACS not read yet",
        }
    }

    fn tip(self) -> &'static str {
        match self {
            RowState::New => "Made here and not saved yet",
            RowState::Outside => "Not in the PACS",
            RowState::Changed => "Changed here; saved as a new object beside the old one",
            RowState::InPacs => "Already in the PACS; saved again as a new object",
            RowState::Empty => "Empty: nothing to save",
            RowState::NoStudy => "Import its study into the PACS first",
            RowState::Reading => "Looking up the PACS",
            RowState::NotRead => "Rescan the PACS in the PACS window first",
        }
    }

    fn savable(self) -> bool {
        matches!(
            self,
            RowState::New | RowState::Outside | RowState::Changed | RowState::InPacs
        )
    }

    /// Ticked when the window opens.
    fn ticked(self) -> bool {
        matches!(self, RowState::New | RowState::Outside | RowState::Changed)
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
    /// The image series it was drawn on: where it is filed.
    pub series_uid: String,
    pub state: RowState,
    pub tick: bool,
}

/// The window's state.
pub(super) struct PacsSave {
    pub slot: usize,
    pub rows: Vec<SaveRow>,
    /// What the PACS holds; `None` while it is looked up.
    pub targets: Option<SaveTargets>,
    /// The one object to tick when the lookup lands, as opened from the tree.
    pub only: Option<(DerivedKind, usize)>,
    /// The last save's outcome, and whether it is a warning.
    pub status: Option<(String, bool)>,
}

/// An edit fingerprint of a structure set: its label, references and ROIs
/// to the last contour point.
pub(super) fn mark_of_set(ss: &StructureSet) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ss.label.hash(&mut h);
    ss.study_uid.hash(&mut h);
    ss.frame_of_reference_uid.hash(&mut h);
    ss.referenced_series_uid.hash(&mut h);
    ss.rois.len().hash(&mut h);
    for r in &ss.rois {
        r.number.hash(&mut h);
        r.name.hash(&mut h);
        r.color.hash(&mut h);
        r.roi_type.hash(&mut h);
        r.description.hash(&mut h);
        r.contours.len().hash(&mut h);
        for c in &r.contours {
            c.geometric_type.hash(&mut h);
            c.points.len().hash(&mut h);
            for p in &c.points {
                p.x.to_bits().hash(&mut h);
                p.y.to_bits().hash(&mut h);
                p.z.to_bits().hash(&mut h);
            }
        }
    }
    h.finish()
}

/// An edit fingerprint of a segmentation series: its label and references,
/// and per segment its name, colour and generation (which moves on every
/// edit of the mask, and is unique in the process).
pub(super) fn mark_of_seg(ser: &SegSeries) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ser.label.hash(&mut h);
    ser.study_uid.hash(&mut h);
    ser.referenced_series_uid.hash(&mut h);
    ser.segs.len().hash(&mut h);
    for s in &ser.segs {
        s.name.hash(&mut h);
        s.color.hash(&mut h);
        s.gen.hash(&mut h);
    }
    h.finish()
}

/// The edit fingerprint of object `index` of `kind`, with its SOP Instance
/// UID.
fn mark_of(study: &LoadedStudy, kind: DerivedKind, index: usize) -> Option<(String, u64)> {
    match kind {
        DerivedKind::StructureSet => study
            .structure_sets
            .get(index)
            .map(|s| (s.sop_instance_uid.clone(), mark_of_set(s))),
        DerivedKind::Segmentation => study
            .seg_series
            .get(index)
            .map(|s| (s.sop_instance_uid.clone(), mark_of_seg(s))),
    }
}

/// The study and referenced series of every object of `study`, and their
/// SOP Instance UIDs: what [`crate::archive::Archive::save_targets`] is
/// asked about.
fn lookup_keys(study: &LoadedStudy) -> (Vec<(String, String)>, Vec<String>) {
    let mut keys = Vec::new();
    let mut sops = Vec::new();
    for s in &study.structure_sets {
        keys.push((
            dicom_export::derived_study_uid(study, &s.study_uid, &s.referenced_series_uid),
            s.referenced_series_uid.clone(),
        ));
        sops.push(s.sop_instance_uid.clone());
    }
    for s in &study.seg_series {
        keys.push((
            dicom_export::derived_study_uid(study, &s.study_uid, &s.referenced_series_uid),
            s.referenced_series_uid.clone(),
        ));
        sops.push(s.sop_instance_uid.clone());
    }
    sops.retain(|s| !s.is_empty());
    (keys, sops)
}

/// The rows of `study`: what each object is, and what the PACS knows of it
/// (`targets`: what it holds and where each study's objects go; `None`
/// while it is looked up). `changed(kind, index)`: whether an object was
/// edited since it was loaded or saved. Ticked: what is not in the PACS
/// yet, and what was changed.
pub(super) fn rows_of(
    study: &LoadedStudy,
    targets: Option<&SaveTargets>,
    changed: &dyn Fn(DerivedKind, usize) -> bool,
) -> Vec<SaveRow> {
    let state = |kind: DerivedKind,
                 index: usize,
                 empty: bool,
                 key: &(String, String),
                 sop: &str,
                 made_here: bool| {
        let Some(t) = targets else {
            return if empty {
                RowState::Empty
            } else {
                RowState::Reading
            };
        };
        if empty {
            RowState::Empty
        } else if !t.catalog {
            RowState::NotRead
        } else if !t.homes.get(key).is_some_and(Option::is_some) {
            RowState::NoStudy
        } else if !sop.is_empty() && t.held.contains(sop) {
            if changed(kind, index) {
                RowState::Changed
            } else {
                RowState::InPacs
            }
        } else if made_here {
            RowState::New
        } else {
            RowState::Outside
        }
    };
    let mut rows = Vec::new();
    for (i, ss) in study.structure_sets.iter().enumerate() {
        let key = (
            dicom_export::derived_study_uid(study, &ss.study_uid, &ss.referenced_series_uid),
            ss.referenced_series_uid.clone(),
        );
        let st = state(
            DerivedKind::StructureSet,
            i,
            ss.rois.is_empty(),
            &key,
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
            study_uid: key.0,
            series_uid: key.1,
            state: st,
            tick: st.ticked(),
        });
    }
    for (i, ser) in study.seg_series.iter().enumerate() {
        let key = (
            dicom_export::derived_study_uid(study, &ser.study_uid, &ser.referenced_series_uid),
            ser.referenced_series_uid.clone(),
        );
        let filled = ser.segs.iter().filter(|s| s.count > 0).count();
        let st = state(
            DerivedKind::Segmentation,
            i,
            filled == 0,
            &key,
            &ser.sop_instance_uid,
            ser.file_name.is_empty(),
        );
        rows.push(SaveRow {
            kind: DerivedKind::Segmentation,
            index: i,
            label: ser.label.clone(),
            detail: plural(filled, "segment", "segments"),
            study_uid: key.0,
            series_uid: key.1,
            state: st,
            tick: st.ticked(),
        });
    }
    rows
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// What a finished save hands back.
pub(super) struct SaveLanded {
    /// What each saved object was before (to know whose identity to take
    /// over).
    pub rows: Vec<SaveRow>,
    pub saved: Saved,
    /// The PACS listed afresh, and what it holds now.
    pub tree: Tree,
    pub targets: SaveTargets,
    /// The edit fingerprint of each object as it was saved, by kind and
    /// index.
    pub marks: Vec<(DerivedKind, usize, u64)>,
}

pub(super) type SaveDone = (usize, anyhow::Result<SaveLanded>);

/// What a finished lookup hands back: the workspace, and what the PACS holds.
pub(super) type LookupDone = (usize, SaveTargets);

impl ViewerApp {
    /// Take every object of workspace `slot` as it is now as unchanged:
    /// what a study that has just landed is.
    pub(super) fn mark_all_saved(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        s.saved_marks.clear();
        let Some(study) = s.study.as_ref() else {
            return;
        };
        for ss in study.structure_sets.iter() {
            if !ss.sop_instance_uid.is_empty() {
                s.saved_marks.insert(
                    (DerivedKind::StructureSet, ss.sop_instance_uid.clone()),
                    mark_of_set(ss),
                );
            }
        }
        for ser in study.seg_series.iter() {
            if !ser.sop_instance_uid.is_empty() {
                s.saved_marks.insert(
                    (DerivedKind::Segmentation, ser.sop_instance_uid.clone()),
                    mark_of_seg(ser),
                );
            }
        }
    }

    /// The rows of workspace `slot` against `targets`.
    fn pacs_rows(&self, slot: usize, targets: Option<&SaveTargets>) -> Vec<SaveRow> {
        let s = &self.slots[slot];
        let Some(study) = s.study.as_ref() else {
            return Vec::new();
        };
        let changed = |kind: DerivedKind, index: usize| {
            mark_of(study, kind, index).is_some_and(|(sop, now)| {
                s.saved_marks
                    .get(&(kind, sop))
                    .is_some_and(|then| *then != now)
            })
        };
        rows_of(study, targets, &changed)
    }

    /// Open *Save to the PACS* on workspace `slot`; with `only`, that one
    /// object alone is ticked. The window comes up at once; what the PACS
    /// holds is looked up on a job ([`Self::on_pacs_lookup`]).
    pub(super) fn open_pacs_save(&mut self, slot: usize, only: Option<(DerivedKind, usize)>) {
        if self
            .slots
            .get(slot)
            .and_then(|s| s.study.as_ref())
            .is_none()
        {
            return;
        }
        let rows = self.pacs_rows(slot, None);
        self.pacs_save = Some(PacsSave {
            slot,
            rows,
            targets: None,
            only,
            status: None,
        });
        self.start_pacs_lookup(slot);
        self.raise_window("pacs_save");
    }

    /// Look up what the PACS holds of workspace `slot`'s objects.
    fn start_pacs_lookup(&mut self, slot: usize) {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let (keys, sops) = lookup_keys(study);
        let archive = self.station_pacs();
        let progress = Arc::new(Progress::default());
        progress.set("Reading the PACS");
        self.pacs_save_lookup_job = Some(Job::spawn(progress, move |_| {
            (slot, archive.save_targets(&keys, &sops))
        }));
    }

    /// The lookup landed: the rows say what the PACS holds. Dropped when
    /// the window was closed or moved to another workspace meanwhile;
    /// looked up again when the workspace holds objects it was not asked
    /// about (a study replaced, a set added).
    pub(super) fn on_pacs_lookup(&mut self, slot: usize, targets: SaveTargets) {
        if !self.pacs_save.as_ref().is_some_and(|d| d.slot == slot) {
            return;
        }
        let Some(study) = self.slots[slot].study.as_ref() else {
            self.pacs_save = None;
            return;
        };
        if !self.targets_cover(study, &targets) {
            self.start_pacs_lookup(slot);
            return;
        }
        let mut rows = self.pacs_rows(slot, Some(&targets));
        let Some(d) = self.pacs_save.as_mut() else {
            return;
        };
        if let Some((kind, index)) = d.only.take() {
            for r in &mut rows {
                r.tick = r.kind == kind && r.index == index && r.state.savable();
            }
        }
        d.rows = rows;
        d.targets = Some(targets);
    }

    /// Whether `targets` were asked about every object `study` holds now.
    fn targets_cover(&self, study: &LoadedStudy, targets: &SaveTargets) -> bool {
        !targets.catalog
            || lookup_keys(study)
                .0
                .iter()
                .all(|k| targets.homes.contains_key(k))
    }

    pub(super) fn pacs_save_window(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.pacs_save.take() else {
            return;
        };
        let busy = self.pacs_save_job.is_some();
        let progress = self.pacs_save_job.as_ref().map(|j| j.progress.get());
        let reading = self.pacs_save_lookup_job.is_some() || d.targets.is_none();
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
                // Where they go, a line per folder.
                if let Some(t) = &d.targets {
                    let homes: BTreeSet<&PathBuf> = d
                        .rows
                        .iter()
                        .filter(|r| r.tick)
                        .filter_map(|r| {
                            t.homes
                                .get(&(r.study_uid.clone(), r.series_uid.clone()))?
                                .as_ref()
                        })
                        .collect();
                    for home in homes {
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
                if d.rows.iter().any(|r| r.state == RowState::NotRead) {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "This station's PACS has not been scanned yet: Rescan in the PACS \
                         window, then open Save to the PACS again.",
                    );
                }
                ui.separator();
                let n = d.rows.iter().filter(|r| r.tick).count();
                ui.horizontal(|ui| {
                    let label = format!("📤 Save {}", plural(n, "object", "objects"));
                    if ui
                        .add_enabled(n > 0 && !busy && !reading, egui::Button::new(label))
                        .on_hover_text("Save the ticked objects")
                        .clicked()
                    {
                        save = true;
                    }
                    if let Some(p) = &progress {
                        ui.spinner();
                        ui.weak(p);
                    } else if reading {
                        ui.spinner();
                        ui.weak("Reading the PACS");
                    }
                });
                if let Some((s, warn)) = &d.status {
                    if *warn {
                        ui.colored_label(ui.visuals().warn_fg_color, s);
                    } else {
                        ui.label(s);
                    }
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
            .map(|saved| {
                // The fingerprints of what was written, from the copy it
                // was written from: an edit made while the job ran is not
                // in the PACS.
                let marks = saved
                    .objects
                    .iter()
                    .filter_map(|o| {
                        mark_of(&study, o.kind, o.index).map(|(_, m)| (o.kind, o.index, m))
                    })
                    .collect();
                // What the PACS holds now, the saved copies included.
                let (keys, mut sops) = lookup_keys(&study);
                sops.extend(saved.objects.iter().map(|o| o.sop_instance_uid.clone()));
                SaveLanded {
                    targets: archive.save_targets(&keys, &sops),
                    tree: archive.tree(),
                    rows,
                    saved,
                    marks,
                }
            });
            (slot, out)
        }));
    }

    /// A save landed: new objects take on their PACS identity, every saved
    /// object's fingerprint moves to what was saved, and the window and the
    /// PACS window show what is there now. Nothing here reads the catalog:
    /// the job brought the targets with it.
    pub(super) fn on_pacs_saved(&mut self, slot: usize, landed: SaveLanded) {
        let SaveLanded {
            rows,
            saved,
            tree,
            targets,
            marks,
        } = landed;
        let s = &mut self.slots[slot];
        if let Some(study) = s.study.as_mut() {
            for o in &saved.objects {
                let Some(was) = rows.iter().find(|r| r.kind == o.kind && r.index == o.index) else {
                    continue;
                };
                let mark = marks
                    .iter()
                    .find(|(k, i, _)| *k == o.kind && *i == o.index)
                    .map(|(_, _, m)| *m);
                // Only what was not in the PACS takes the saved copy's
                // identity: one loaded from it (changed or not) keeps the
                // identity a plan or a registration may refer to.
                let keeps = matches!(was.state, RowState::InPacs | RowState::Changed);
                let name = o
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let sop = match o.kind {
                    DerivedKind::StructureSet => study
                        .structure_sets
                        .get_mut(o.index)
                        .filter(|s| s.label == was.label || s.file_name == was.label)
                        .map(|ss| {
                            if !keeps {
                                ss.sop_instance_uid = o.sop_instance_uid.clone();
                                ss.series_instance_uid = o.series_instance_uid.clone();
                                ss.file_name = name.clone();
                            }
                            ss.sop_instance_uid.clone()
                        }),
                    DerivedKind::Segmentation => study
                        .seg_series
                        .get_mut(o.index)
                        .filter(|s| s.label == was.label)
                        .map(|ser| {
                            if !keeps {
                                ser.sop_instance_uid = o.sop_instance_uid.clone();
                                ser.series_instance_uid = o.series_instance_uid.clone();
                                ser.file_name = name.clone();
                            }
                            ser.sop_instance_uid.clone()
                        }),
                };
                if let (Some(sop), Some(mark)) = (sop, mark) {
                    if !sop.is_empty() {
                        s.saved_marks.insert((o.kind, sop), mark);
                    }
                }
            }
        }
        let n = saved.objects.len();
        let (msg, warn) = if let Some(why) = &saved.failed {
            (
                format!(
                    "⚠ {} saved; the rest failed: {why}",
                    plural(n, "object", "objects")
                ),
                true,
            )
        } else if saved.summary.too_long > 0 {
            (
                format!(
                    "⚠ {} saved; {} left out: the path would pass the limit",
                    plural(n, "object", "objects"),
                    saved.summary.too_long
                ),
                true,
            )
        } else {
            (
                format!("✔ {} saved into the PACS", plural(n, "object", "objects")),
                false,
            )
        };
        if let Some(w) = self.pacs.as_mut() {
            w.status = Some(msg.clone());
            w.set_tree(tree);
        }
        if self.pacs_save.as_ref().is_some_and(|d| d.slot == slot) {
            // What was saved reads "in the PACS"; what failed is still
            // ticked.
            let cover = self.slots[slot]
                .study
                .as_ref()
                .is_some_and(|st| self.targets_cover(st, &targets));
            let rows = self.pacs_rows(slot, cover.then_some(&targets));
            if let Some(d) = self.pacs_save.as_mut() {
                d.rows = rows;
                d.only = None;
                d.targets = cover.then_some(targets);
                d.status = Some((msg, warn));
            }
            if !cover {
                self.start_pacs_lookup(slot);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dicomseg::SegSeries;
    use crate::rtstruct::StructureSet;
    use std::collections::BTreeMap;

    #[test]
    fn what_is_new_or_changed_is_ticked_and_what_the_pacs_holds_is_not() {
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
            set("Edited", "RS2.dcm", "2.2", "S"),
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

        let targets = SaveTargets {
            catalog: true,
            held: ["1.1".to_string(), "2.2".to_string()].into(),
            homes: [
                (
                    ("S".to_string(), String::new()),
                    Some(PathBuf::from("pacs/Trial/P01")),
                ),
                (("T".to_string(), String::new()), None),
            ]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
        };
        // The fifth set was edited since it was loaded.
        let changed = |k: DerivedKind, i: usize| k == DerivedKind::StructureSet && i == 4;
        let rows = rows_of(&study, Some(&targets), &changed);
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
                ("Edited", RowState::Changed, true),
                ("Empty", RowState::Empty, false),
                ("Painted", RowState::New, true),
            ]
        );
        assert_eq!(rows[6].kind, DerivedKind::Segmentation);
        assert_eq!(rows[6].detail, "1 segment");

        // Changed only counts for what the PACS holds.
        let all = |_: DerivedKind, _: usize| true;
        let rows = rows_of(&study, Some(&targets), &all);
        assert_eq!(rows[0].state, RowState::Changed);
        assert_eq!(rows[2].state, RowState::Outside);

        // While the PACS is read, and before it ever was: nothing to tick.
        let reading = rows_of(&study, None, &changed);
        assert!(reading
            .iter()
            .all(|r| !r.tick && matches!(r.state, RowState::Reading | RowState::Empty)));
        let unread = rows_of(&study, Some(&SaveTargets::default()), &changed);
        assert!(unread
            .iter()
            .all(|r| !r.tick && matches!(r.state, RowState::NotRead | RowState::Empty)));
    }

    #[test]
    fn an_edit_moves_the_fingerprint_and_nothing_else_does() {
        let mut ss = StructureSet {
            label: "S".into(),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: "1.1".into(),
            series_instance_uid: String::new(),
            study_uid: String::new(),
            referenced_series_uid: String::new(),
            file_name: "RS.dcm".into(),
            locked: false,
            rois: vec![crate::rtstruct::Roi {
                number: 1,
                name: "PTV".into(),
                color: [255, 0, 0],
                roi_type: String::new(),
                description: String::new(),
                contours: vec![crate::rtstruct::Contour {
                    points: vec![crate::geometry::Vec3::new(1.0, 2.0, 3.0)],
                    geometric_type: "CLOSED_PLANAR".into(),
                }],
            }],
        };
        let before = mark_of_set(&ss);
        ss.locked = true;
        assert_eq!(before, mark_of_set(&ss), "locking is no edit");
        ss.rois[0].contours[0].points[0].x += 0.5;
        assert_ne!(before, mark_of_set(&ss), "a point moved");
        ss.rois[0].contours[0].points[0].x -= 0.5;
        assert_eq!(before, mark_of_set(&ss));
        ss.rois[0].name = "CTV".into();
        assert_ne!(before, mark_of_set(&ss), "a rename");
    }
}
