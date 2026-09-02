//! *Tools ▶ 🏥 PACS*: the local patient archive as a window.
//!
//! Everything the archive can do reduces to three gestures - file a folder
//! into it, take a patient or a study out of it into a viewer dataset, and
//! give back what was drawn on one - and each is one button here. The
//! archive itself ([`crate::archive`]) knows nothing about the UI; this
//! window is the part that knows which dataset the user meant.
//!
//! Loading needs no special path: a study folder in the archive *is* a DICOM
//! folder, so it goes through the same `loader::load_directory` as *File ▶
//! Add DICOM folder*, with the same merging and the same progress.

use crate::archive::{Archive, ImportSummary, PatientEntry};

use super::*;

/// What the PACS window is showing and doing.
pub(super) struct PacsWindow {
    /// Archive root as edited in the window.
    pub dir: String,
    /// The last scan; `None` until one has been made.
    pub patients: Option<Vec<PatientEntry>>,
    /// Which patient row is expanded, by index into `patients`.
    pub expanded: Option<usize>,
    /// The selected study, as (patient index, study index). A patient with
    /// no study selected means "the whole patient".
    pub selected: Option<(usize, Option<usize>)>,
    pub status: Option<String>,
}

impl PacsWindow {
    fn new(dir: String) -> PacsWindow {
        PacsWindow {
            dir,
            patients: None,
            expanded: None,
            selected: None,
            status: None,
        }
    }
}

/// What a background archive job answers with.
pub(super) enum PacsOutcome {
    Scanned(Vec<PatientEntry>),
    Imported(ImportSummary),
    /// Objects written back into the archive: how many, and where.
    Uploaded(usize, String),
    Removed,
}

impl ViewerApp {
    pub(super) fn open_pacs_window(&mut self) {
        if self.pacs.is_none() {
            let dir = if self.archive_dir.trim().is_empty() {
                crate::archive::default_root().display().to_string()
            } else {
                self.archive_dir.clone()
            };
            self.pacs = Some(PacsWindow::new(dir));
            self.start_pacs_scan();
        }
    }

    /// The archive root the window is pointed at.
    fn pacs_root(&self) -> std::path::PathBuf {
        crate::archive::root_from_setting(self.pacs.as_ref().map(|p| p.dir.as_str()).unwrap_or(""))
    }

    pub(super) fn start_pacs_scan(&mut self) {
        if self.pacs_job.is_some() {
            return;
        }
        let root = self.pacs_root();
        let progress = Arc::new(Progress::default());
        progress.set("Reading the archive");
        self.pacs_job = Some(Job::spawn(progress, move |_| {
            Archive::new(root).scan().map(PacsOutcome::Scanned)
        }));
    }

    fn start_pacs_import(&mut self, src: std::path::PathBuf) {
        if self.pacs_job.is_some() {
            return;
        }
        let root = self.pacs_root();
        let progress = Arc::new(Progress::default());
        self.pacs_job = Some(Job::spawn(progress, move |p| {
            Archive::new(root)
                .import(&src, p)
                .map(PacsOutcome::Imported)
        }));
    }

    /// Write the structure sets and segmentation series of a dataset back
    /// into the archive.
    ///
    /// They are written to a scratch folder first and then imported, so the
    /// filing rule lives in exactly one place - the archive - and an upload
    /// that fails half way leaves the archive untouched rather than
    /// half-written.
    fn start_pacs_upload(&mut self, slot: usize) {
        if self.pacs_job.is_some() {
            return;
        }
        let Some(study) = self.slots[slot].study.as_ref() else {
            self.error = Some(format!("dataset {} is not loaded", SLOT_NAMES[slot]));
            return;
        };
        let derived = study.structure_sets.iter().any(|ss| !ss.rois.is_empty())
            || study
                .seg_series
                .iter()
                .any(|sr| sr.segs.iter().any(|s| s.count > 0));
        if !derived {
            self.error = Some(format!(
                "dataset {} has no structure sets or segmentations to send",
                SLOT_NAMES[slot]
            ));
            return;
        }
        let study = study.clone();
        let params = dicom_export::ExportParams::for_study(&study);
        let root = self.pacs_root();
        let scratch = std::env::temp_dir().join(format!(
            "rds_upload_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        let progress = Arc::new(Progress::default());
        self.pacs_job = Some(Job::spawn(progress, move |p| {
            p.set("Writing the derived objects");
            let n = dicom_export::export_derived(&study, &scratch, &params, p)?;
            let archive = Archive::new(root);
            let sum = archive.import(&scratch, p)?;
            let _ = std::fs::remove_dir_all(&scratch);
            Ok(PacsOutcome::Uploaded(n, sum.describe()))
        }));
    }

    fn start_pacs_remove(&mut self, dir: std::path::PathBuf) {
        if self.pacs_job.is_some() {
            return;
        }
        let root = self.pacs_root();
        let progress = Arc::new(Progress::default());
        progress.set("Removing");
        self.pacs_job = Some(Job::spawn(progress, move |_| {
            Archive::new(root)
                .remove(&dir)
                .map(|()| PacsOutcome::Removed)
        }));
    }

    /// An archive job finished.
    pub(super) fn on_pacs_done(&mut self, outcome: PacsOutcome) {
        let mut rescan = false;
        if let Some(w) = &mut self.pacs {
            match outcome {
                PacsOutcome::Scanned(patients) => {
                    w.status = Some(format!(
                        "{} patient(s), {} study(ies)",
                        patients.len(),
                        patients.iter().map(|p| p.studies.len()).sum::<usize>()
                    ));
                    // A selection that the rescan invalidated must go, or the
                    // next click would act on a row that has moved.
                    if w.selected
                        .map(|(pi, si)| match si {
                            Some(si) => patients
                                .get(pi)
                                .map(|p| si >= p.studies.len())
                                .unwrap_or(true),
                            None => pi >= patients.len(),
                        })
                        .unwrap_or(false)
                    {
                        w.selected = None;
                    }
                    w.expanded = w.expanded.filter(|i| *i < patients.len());
                    w.patients = Some(patients);
                }
                PacsOutcome::Imported(sum) => {
                    w.status = Some(format!("✔ imported: {}", sum.describe()));
                    rescan = true;
                }
                PacsOutcome::Uploaded(n, filed) => {
                    w.status = Some(format!("✔ {n} object(s) sent - {filed}"));
                    rescan = true;
                }
                PacsOutcome::Removed => {
                    w.status = Some("✔ removed".into());
                    rescan = true;
                }
            }
        }
        if rescan {
            self.start_pacs_scan();
        }
    }

    pub(super) fn pacs_window(&mut self, ctx: &egui::Context) {
        if self.pacs.is_none() {
            return;
        }
        let mut open = true;
        let mut close = false;
        let mut rescan = false;
        let mut browse = false;
        let mut import = false;
        let mut load: Option<(usize, std::path::PathBuf)> = None;
        let mut upload: Option<usize> = None;
        let mut remove: Option<std::path::PathBuf> = None;
        let mut expand: Option<Option<usize>> = None;
        let mut select: Option<(usize, Option<usize>)> = None;
        let mut commit_dir = false;

        let busy = self.pacs_job.is_some();
        let loaded: [bool; 2] = [self.slots[0].study.is_some(), self.slots[1].study.is_some()];
        let mut w = self.pacs.take().expect("checked above");

        detach::tool_window(
            ctx,
            "pacs",
            "🏥 PACS - patient archive",
            &mut open,
            detach::WinOpts::size(720.0, 520.0),
            |ui| {
                ui.label(
                    "The local archive: every study filed here, ready to be taken into a \
                     dataset and given back the structures and segmentations drawn on it.",
                );
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Archive").strong());
                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut w.dir)
                            .desired_width(360.0)
                            .hint_text("archive folder"),
                    );
                    // Committed when the field is left, not on every
                    // keystroke: a half-typed path is not a folder to scan.
                    if resp.lost_focus() {
                        commit_dir = true;
                    }
                    if ui.button("📂 Browse").clicked() {
                        browse = true;
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("⟲ Rescan"))
                        .clicked()
                    {
                        rescan = true;
                    }
                });
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!busy, egui::Button::new("📥 Import folder"))
                        .on_hover_text("Copy every DICOM file of a folder into the archive")
                        .clicked()
                    {
                        import = true;
                    }
                    for (slot, name) in SLOT_NAMES.iter().enumerate() {
                        if ui
                            .add_enabled(
                                !busy && loaded[slot],
                                egui::Button::new(format!("📤 Send dataset {name}")),
                            )
                            .on_hover_text(
                                "Write this dataset's structure sets and segmentation \
                                 series back into the archive, attached to the study they \
                                 belong to (new SOP Instance UIDs, original Study and \
                                 Frame of Reference UIDs). Images are never re-sent.",
                            )
                            .clicked()
                        {
                            upload = Some(slot);
                        }
                    }
                });

                if let Some(job) = &self.pacs_job {
                    ui.separator();
                    progress_row(ui, &job.progress);
                }
                if let Some(s) = &w.status {
                    ui.weak(s);
                }
                ui.separator();

                let Some(patients) = &w.patients else {
                    ui.weak("reading");
                    return;
                };
                if patients.is_empty() {
                    ui.weak("The archive is empty. 📥 Import folder files a study into it.");
                    return;
                }
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for (pi, p) in patients.iter().enumerate() {
                            let is_open = w.expanded == Some(pi);
                            ui.horizontal(|ui| {
                                if ui.small_button(if is_open { "▼" } else { "▶" }).clicked() {
                                    expand = Some(if is_open { None } else { Some(pi) });
                                }
                                let resp = ui.add(
                                    egui::Button::selectable(
                                        w.selected == Some((pi, None)),
                                        format!(
                                            "{}   {} study(ies) · {} file(s)",
                                            p.title(),
                                            p.studies.len(),
                                            p.files()
                                        ),
                                    )
                                    .wrap(),
                                );
                                if resp.clicked() {
                                    select = Some((pi, None));
                                    expand = Some(Some(pi));
                                }
                                resp.context_menu(|ui| {
                                    if ui.button("🗑 Remove this patient").clicked() {
                                        remove = Some(p.dir.clone());
                                        ui.close();
                                    }
                                });
                            });
                            if !is_open {
                                continue;
                            }
                            for (si, st) in p.studies.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    ui.add_space(24.0);
                                    let resp = ui.add(
                                        egui::Button::selectable(
                                            w.selected == Some((pi, Some(si))),
                                            st.describe(),
                                        )
                                        .wrap(),
                                    );
                                    if resp.clicked() {
                                        select = Some((pi, Some(si)));
                                    }
                                    let resp = resp.on_hover_text(format!(
                                        "Study UID …{}\n{}",
                                        tail(&st.study_uid),
                                        st.dir.display()
                                    ));
                                    resp.context_menu(|ui| {
                                        if ui.button("🗑 Remove this study").clicked() {
                                            remove = Some(st.dir.clone());
                                            ui.close();
                                        }
                                    });
                                });
                            }
                        }
                    });

                ui.separator();
                let picked = w.selected.and_then(|(pi, si)| {
                    let p = patients.get(pi)?;
                    Some(match si {
                        Some(si) => (p.studies.get(si)?.dir.clone(), st_label(p, si)),
                        None => (p.dir.clone(), p.title()),
                    })
                });
                ui.horizontal(|ui| {
                    for (slot, name) in SLOT_NAMES.iter().enumerate() {
                        if ui
                            .add_enabled(
                                !busy && picked.is_some(),
                                egui::Button::new(format!("📩 Load into dataset {name}")),
                            )
                            .on_hover_text(
                                "Read the selection into this dataset, merging it with \
                                 whatever is already there - the same as adding its folder",
                            )
                            .clicked()
                        {
                            if let Some((dir, _)) = &picked {
                                load = Some((slot, dir.clone()));
                            }
                        }
                    }
                    match &picked {
                        Some((_, label)) => ui.weak(label.clone()),
                        None => ui.weak("select a patient or a study"),
                    };
                });
                ui.add_space(4.0);
                if ui.button("Close").clicked() {
                    close = true;
                }
            },
        );

        if let Some(e) = expand {
            w.expanded = e;
        }
        if let Some(s) = select {
            w.selected = Some(s);
        }
        if browse {
            if let Some(dir) = Self::pick_folder("Archive folder") {
                w.dir = dir.display().to_string();
                w.patients = None;
                commit_dir = true;
                rescan = true;
            }
        }
        // The window closing counts as leaving the field.
        let leaving = close || !open;
        let dir_changed = (commit_dir || leaving) && w.dir != self.archive_dir;
        if dir_changed {
            self.archive_dir = w.dir.clone();
            if !rescan {
                w.patients = None;
                rescan = true;
            }
        }
        if !leaving {
            self.pacs = Some(w);
        }
        if dir_changed {
            self.persist_settings();
        }
        if rescan && self.pacs.is_some() {
            self.start_pacs_scan();
        }
        if import {
            if let Some(dir) = Self::pick_folder("Folder to file into the archive") {
                self.start_pacs_import(dir);
            }
        }
        if let Some(slot) = upload {
            self.start_pacs_upload(slot);
        }
        if let Some(dir) = remove {
            self.start_pacs_remove(dir);
        }
        if let Some((slot, dir)) = load {
            self.start_load(slot, dir);
        }
    }
}

/// A study row's label, for the "what is selected" line.
fn st_label(p: &PatientEntry, si: usize) -> String {
    match p.studies.get(si) {
        Some(st) => format!("{} · {}", p.title(), st.describe()),
        None => p.title(),
    }
}
