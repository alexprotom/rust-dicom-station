//! *Modules ▶ Image synthesis*: a synthetic CT from the displayed MR or
//! CBCT ([`crate::synth`], TotalSynth), landed as a new CT series of the
//! study ([`crate::sct`]), and its evaluation against a CT of the same
//! patient ([`crate::sct::eval`]).
//!
//! The section has the bones of the auto tools: the workspace row, a
//! description, the inputs, an `Options` fold with the compute device and
//! the model folder, the licence line in the warning colour (the published
//! weights are CC BY-NC), and a button row that becomes the progress row.
//! The result is a series of files that joins the study, so it can be
//! shown, segmented, dosed, exported and archived like any CT; it is shown
//! in another workspace by default, on the same frame of reference as its
//! source, which is the review a physicist does first.

use crate::loader::SeriesInfo;
use crate::models::Engine as ModelsEngine;
use crate::nn::device::DevicePref;
use crate::registration::{resample_through, Direction, Transform3};
use crate::sct::{self, eval, BodySource, SctOutcome, SctRequest};
use crate::synth::{self, App};

use super::*;

/// The section's glyph and name.
pub(super) const SYNTHESIS: ToolInfo = ToolInfo {
    glyph: "🔮",
    name: "Image synthesis",
};

/// Which checkpoints a run averages.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Members {
    /// CV_0 alone: the quick look.
    Quick,
    /// Every checkpoint of the app: the published recipe.
    All,
    /// The ticked ones.
    Chosen,
}

/// Where the air outside the patient comes from, as the section offers it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum BodyChoice {
    Network,
    Classical,
    /// A structure of the workspace's active set, by name.
    Structure(String),
    None,
}

/// The reference CT of an evaluation.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum EvalRef {
    /// The series another workspace displays.
    Workspace(usize),
    /// A CT series of the same study, by UID (read from its files).
    Series(String),
}

/// Where an evaluation compares.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(super) enum EvalMask {
    /// The body contour of the reference (threshold and morphology).
    Body,
    /// A structure of the workspace's active set, by name.
    Structure(String),
    /// Wherever the reference has a value.
    Covered,
}

/// The section's state.
pub(super) struct SynthState {
    pub slot: usize,
    pub app: String,
    /// The modality the model was last chosen for: a series of another
    /// modality re-seeds it.
    pub seeded_for: String,
    pub members: Members,
    pub chosen: Vec<bool>,
    pub tta: usize,
    pub uncertainty: bool,
    pub body: BodyChoice,
    pub device: DevicePref,
    /// Show the result in this workspace (`None`: in the source's own).
    pub show_in: Option<usize>,
    pub status: Option<String>,
    /// Where the last run wrote its files (shown on hover: a path does not
    /// wrap).
    pub status_dir: Option<String>,
    pub eval_ref: Option<EvalRef>,
    pub eval_mask: EvalMask,
    pub eval_wet: bool,
    pub hlut: String,
    pub eval_dice: bool,
    pub eval_diff: bool,
    pub eval_status: Option<String>,
    pub eval_result: Option<EvalView>,
}

impl Default for SynthState {
    fn default() -> Self {
        SynthState {
            slot: 0,
            app: App::Mr.key().to_string(),
            seeded_for: String::new(),
            members: Members::Quick,
            chosen: vec![true; synth::CHECKPOINTS],
            tta: 0,
            uncertainty: false,
            body: BodyChoice::Network,
            device: DevicePref::Auto,
            show_in: Some(1),
            status: None,
            status_dir: None,
            eval_ref: None,
            eval_mask: EvalMask::Body,
            eval_wet: false,
            hlut: eval::Hlut::default().text(),
            eval_dice: false,
            eval_diff: true,
            eval_status: None,
            eval_result: None,
        }
    }
}

/// What a finished run hands to the interface thread.
pub(super) struct SynthDone {
    outcome: SctOutcome,
    source_uid: String,
    show_in: Option<usize>,
}

/// An evaluation's numbers as the section shows them.
pub(super) struct EvalView {
    pub what: String,
    pub result: eval::EvalResult,
    pub csv: String,
}

/// What a finished evaluation hands to the interface thread.
pub(super) struct EvalDone {
    view: EvalView,
    /// The difference series, written and read back, with its volume.
    difference: Option<(LoadedStudy, Arc<Volume>, String)>,
    sct_uid: String,
}

/// The app a key names, or the MR app.
fn app_of(key: &str) -> App {
    App::from_key(key).unwrap_or(App::Mr)
}

/// The app that suits a series of this modality.
fn default_app(modality: &str) -> App {
    if modality.eq_ignore_ascii_case("MR") {
        App::Mr
    } else {
        App::Cbct
    }
}

impl ViewerApp {
    /// The names of the volume structures of a workspace's active set.
    fn active_roi_names(&self, slot: usize) -> Vec<String> {
        let s = &self.slots[slot];
        s.study
            .as_ref()
            .and_then(|st| st.structure_sets.get(s.active_structs))
            .map(|ss| {
                ss.rois
                    .iter()
                    .filter(|r| !r.is_poi() && !r.contours.is_empty())
                    .map(|r| r.name.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// One structure of a workspace's active set as a mask on `grid`.
    fn roi_mask(&self, slot: usize, name: &str, grid: &crate::volume::Grid) -> Option<Vec<u8>> {
        let s = &self.slots[slot];
        let ss = s.study.as_ref()?.structure_sets.get(s.active_structs)?;
        let roi = ss.rois.iter().find(|r| r.name == name)?;
        crate::segmentation::rasterize_roi(grid, roi)
    }

    /// Snapshot the section and run the synthesis on a worker thread.
    pub(super) fn start_synthesis(&mut self) {
        if self.synth_job.is_some() {
            return;
        }
        let slot = self.synth.slot;
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(source) = study.series.get(study.active_series).cloned() else {
            return;
        };
        let volume = study.volume.clone();
        let app = app_of(&self.synth.app);
        let checkpoints: Vec<usize> = match self.synth.members {
            Members::Quick => vec![0],
            Members::All => (0..app.checkpoint_count()).collect(),
            Members::Chosen => (0..app.checkpoint_count())
                .filter(|&k| self.synth.chosen.get(k).copied().unwrap_or(false))
                .collect(),
        };
        if checkpoints.is_empty() {
            self.error = Some("Tick at least one checkpoint.".into());
            return;
        }
        let body = match &self.synth.body {
            BodyChoice::Network => BodySource::Network,
            BodyChoice::Classical => BodySource::Classical,
            BodyChoice::None => BodySource::None,
            BodyChoice::Structure(name) => match self.roi_mask(slot, name, &volume.grid()) {
                Some(mask) => BodySource::Structure {
                    name: name.clone(),
                    mask,
                },
                None => {
                    self.error = Some(format!("Structure {name} is not on this series."));
                    return;
                }
            },
        };
        let req = SctRequest {
            app,
            opts: synth::Options {
                checkpoints,
                tta: self.synth.tta.min(synth::TTA_MAX),
                uncertainty: self.synth.uncertainty,
                device: self.synth.device,
            },
            body,
            series_number: sct::next_series_number(study),
        };
        let root = self.models_root();
        let out_root = sct::default_out_root();
        let show_in = self.synth.show_in;
        self.persist_settings();
        let progress = Arc::new(Progress::default());
        progress.set("Preparing");
        self.synth_slot = slot;
        self.synth_job = Some(Job::spawn(progress, move |p| {
            let r = sct::synthesize(&volume, &source, &req, &root, &out_root, p).map(|outcome| {
                SynthDone {
                    outcome,
                    source_uid: source.uid.clone(),
                    show_in,
                }
            });
            (slot, r)
        }));
    }

    /// A run finished: the series joins the source's study and is shown
    /// where the section said.
    pub(super) fn on_synthesis_done(&mut self, slot: usize, done: SynthDone) {
        let SynthDone {
            outcome,
            source_uid,
            show_in,
        } = done;
        let uid = outcome.series.series_uid.clone();
        let dir = outcome
            .series
            .files
            .first()
            .and_then(|f| f.parent())
            .map(|d| d.display().to_string())
            .unwrap_or_default();
        let has_source = self.slots[slot]
            .study
            .as_ref()
            .is_some_and(|st| st.series.iter().any(|s| s.uid == source_uid));
        if !has_source {
            // The workspace moved on; the files are written all the same.
            self.notice = Some(format!(
                "The synthetic CT was written to {dir}, but its workspace no longer holds \
                 the source series: add the folder with File ▶ Add DICOM folder."
            ));
            return;
        }
        let summary = sct::summary(&outcome);
        let mut sub = outcome.study.clone();
        let idx_in_sub = sub.series.iter().position(|s| s.uid == uid).unwrap_or(0);
        sub.active_series = idx_in_sub;
        sub.volume = outcome.volume.clone();
        sub.default_window = (40.0, 400.0);
        match show_in.filter(|&t| t != slot && t < MAX_WORKSPACES) {
            Some(target) => {
                // Into the source's tree as well, so the series is listed
                // with the study it belongs to.
                if let Some(dest) = self.slots[slot].study.as_mut() {
                    let notes = loader::merge_study(dest, outcome.study.clone());
                    dest.warnings.extend(notes);
                }
                self.tree_insert(
                    target,
                    sub,
                    &uid,
                    Some((outcome.volume.clone(), (40.0, 400.0))),
                );
            }
            None => {
                let idx = {
                    let Some(dest) = self.slots[slot].study.as_mut() else {
                        return;
                    };
                    let notes = loader::merge_study(dest, outcome.study.clone());
                    dest.warnings.extend(notes);
                    dest.series.iter().position(|s| s.uid == uid)
                };
                if let Some(idx) = idx {
                    self.apply_new_volume(slot, outcome.volume.clone(), (40.0, 400.0), idx);
                }
            }
        }
        self.settings_gen += 1;
        self.synth.status = Some(format!("✔ {summary}"));
        self.synth.status_dir = Some(dir);
    }

    /// The reference and its alignment for an evaluation, read on the
    /// interface thread: the volume when it is in memory, how to carry the
    /// sCT's points into it, and its name.
    #[allow(clippy::type_complexity)]
    fn eval_reference(
        &self,
        sct_slot: usize,
        r: &EvalRef,
    ) -> Result<
        (
            Option<Arc<Volume>>,
            Option<SeriesInfo>,
            Option<(Arc<Transform3>, Direction)>,
            String,
        ),
        String,
    > {
        let sct_uid = self.slots[sct_slot]
            .displayed_uid()
            .unwrap_or("")
            .to_string();
        let sct_for = self.slots[sct_slot]
            .study
            .as_ref()
            .map(|st| st.volume.frame_of_reference_uid.clone())
            .unwrap_or_default();
        match r {
            EvalRef::Workspace(other) => {
                let st = self.slots[*other]
                    .study
                    .as_ref()
                    .filter(|st| st.has_volume())
                    .ok_or_else(|| format!("Workspace {} shows no image.", SLOT_NAMES[*other]))?;
                let ref_uid = self.slots[*other].displayed_uid().unwrap_or("").to_string();
                let name = format!(
                    "workspace {} ({})",
                    SLOT_NAMES[*other],
                    st.series
                        .get(st.active_series)
                        .map(|s| s.description.clone())
                        .unwrap_or_default()
                );
                // A registration between exactly these two images.
                if let Some(reg) = &self.registration {
                    let t = reg.result.transform.clone();
                    if reg.fixed_slot == sct_slot
                        && reg.fixed_uid == sct_uid
                        && reg.moving_slot == *other
                        && reg.moving_uid == ref_uid
                    {
                        return Ok((
                            Some(st.volume.clone()),
                            None,
                            Some((t, Direction::FixedToMoving)),
                            format!("{name}, through the active registration"),
                        ));
                    }
                    if reg.moving_slot == sct_slot
                        && reg.moving_uid == sct_uid
                        && reg.fixed_slot == *other
                        && reg.fixed_uid == ref_uid
                    {
                        return Ok((
                            Some(st.volume.clone()),
                            None,
                            Some((t, Direction::MovingToFixed)),
                            format!("{name}, through the active registration"),
                        ));
                    }
                }
                if !sct_for.is_empty() && st.volume.frame_of_reference_uid == sct_for {
                    return Ok((
                        Some(st.volume.clone()),
                        None,
                        None,
                        format!("{name}, same frame of reference"),
                    ));
                }
                Err(format!(
                    "Workspace {} is on another frame of reference: register it to the \
                     synthetic CT first (Image registration; across MR and CT, by \
                     structures).",
                    SLOT_NAMES[*other]
                ))
            }
            EvalRef::Series(uid) => {
                let st = self.slots[sct_slot]
                    .study
                    .as_ref()
                    .ok_or_else(|| "No study.".to_string())?;
                let se = st
                    .series
                    .iter()
                    .find(|s| &s.uid == uid)
                    .cloned()
                    .ok_or_else(|| "That series is no longer in the workspace.".to_string())?;
                let name = format!("series {} of this study", se.description);
                Ok((None, Some(se), None, name))
            }
        }
    }

    /// Snapshot the evaluation and run it on a worker thread.
    pub(super) fn start_sct_eval(&mut self) {
        if self.synth_eval_job.is_some() {
            return;
        }
        let slot = self.synth.slot;
        let Some(r) = self.synth.eval_ref.clone() else {
            self.error = Some("Choose the reference CT.".into());
            return;
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(sct_series) = study.series.get(study.active_series).cloned() else {
            return;
        };
        let sct_vol = study.volume.clone();
        let (ref_vol, ref_series, align, ref_name) = match self.eval_reference(slot, &r) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(e);
                return;
            }
        };
        let mask_roi = match &self.synth.eval_mask {
            EvalMask::Structure(name) => match self.roi_mask(slot, name, &sct_vol.grid()) {
                Some(m) => Some((name.clone(), m)),
                None => {
                    self.error = Some(format!("Structure {name} is not on this series."));
                    return;
                }
            },
            _ => None,
        };
        let mask_kind = self.synth.eval_mask.clone();
        let wet = if self.synth.eval_wet {
            match eval::Hlut::parse(&self.synth.hlut) {
                Ok(h) => Some(h),
                Err(e) => {
                    self.error = Some(format!("The HU to RSP table: {e:#}"));
                    return;
                }
            }
        } else {
            None
        };
        let opts = eval::EvalOptions {
            wet,
            structures: self.synth.eval_dice,
            device: self.synth.device,
        };
        let land_diff = self.synth.eval_diff;
        let root = self.models_root();
        let out_root = sct::default_out_root();
        let next_number = sct::next_series_number(study);
        let progress = Arc::new(Progress::default());
        progress.set("Preparing");
        self.synth_slot = slot;
        self.synth_eval_job = Some(Job::spawn(progress, move |p| {
            let run = || -> anyhow::Result<EvalDone> {
                let grid = sct_vol.grid();
                let reference = match (ref_vol, ref_series) {
                    (Some(v), _) => {
                        p.set("Resampling the reference onto the synthetic CT");
                        resample_through(&v, &grid, align.as_ref().map(|(t, d)| (t.as_ref(), *d)))
                    }
                    (None, Some(se)) => {
                        let (v, _, _) = loader::load_series_volume(&se, p)?;
                        if v.frame_of_reference_uid != sct_vol.frame_of_reference_uid {
                            anyhow::bail!(
                                "the reference series is on another frame of reference: show \
                                 it in another workspace and register it first"
                            );
                        }
                        p.set("Resampling the reference onto the synthetic CT");
                        resample_through(&v, &grid, None)
                    }
                    (None, None) => anyhow::bail!("no reference"),
                };
                let (mask, mask_name) = match mask_kind {
                    EvalMask::Covered => (None, "where the reference has a value".to_string()),
                    EvalMask::Structure(_) => {
                        let (name, m) = mask_roi.expect("rasterized before the run");
                        (Some(m), format!("structure {name}"))
                    }
                    EvalMask::Body => {
                        p.set("Finding the body on the reference");
                        let ct: Vec<i16> = reference
                            .iter()
                            .map(|&v| {
                                if v.is_nan() {
                                    -1024
                                } else {
                                    v.round().clamp(-32768.0, 32767.0) as i16
                                }
                            })
                            .collect();
                        let ct = crate::sct::on_grid(&sct_vol, ct);
                        let b = crate::bodymask::contour_body(
                            &ct,
                            &crate::bodymask::BodyParams {
                                make_external: false,
                                ..crate::bodymask::BodyParams::for_modality("CT")
                            },
                            &root,
                            p,
                        )?;
                        (Some(b.mask), "body contour of the reference".to_string())
                    }
                };
                let result =
                    eval::evaluate(&sct_vol, &reference, mask.as_deref(), &opts, &root, p)?;
                let difference = if land_diff {
                    p.set("Writing the difference series");
                    let vol = Arc::new(crate::sct::on_grid(&sct_vol, result.difference.clone()));
                    let uid = crate::dicom_export::new_uid();
                    let desc = sct::derived_description("minus CT", &sct_series.description);
                    let written = crate::dicom_export::write_derived_image_series(
                        &vol,
                        &sct_series,
                        &sct::series_dir(&out_root, &sct_series.study_uid, &uid),
                        &crate::dicom_export::DerivedImage {
                            series_uid: &uid,
                            description: &desc,
                            derivation: &format!(
                                "Synthetic CT '{}' minus the reference CT ({ref_name}), HU, \
                                 on the synthetic CT's grid; 0 where the reference has no value.",
                                sct_series.description
                            ),
                            series_number: next_number,
                            window: (0.0, 400.0),
                            made_by: "Synthetic CT evaluation",
                        },
                        p,
                    )?;
                    let study = loader::load_files_to_merge(&written.files, "difference", p)?;
                    Some((study, vol, uid))
                } else {
                    None
                };
                let csv = eval::csv(&result);
                Ok(EvalDone {
                    view: EvalView {
                        what: format!(
                            "{} against {ref_name}, inside the {mask_name}",
                            sct_series.description
                        ),
                        result,
                        csv,
                    },
                    difference,
                    sct_uid: sct_series.uid.clone(),
                })
            };
            (slot, run())
        }));
    }

    /// An evaluation finished: its numbers go to the section, the
    /// difference series to the study.
    pub(super) fn on_sct_eval_done(&mut self, slot: usize, done: EvalDone) {
        let EvalDone {
            view,
            difference,
            sct_uid,
        } = done;
        if let Some((study, _vol, _uid)) = difference {
            if let Some(dest) = self.slots[slot]
                .study
                .as_mut()
                .filter(|st| st.series.iter().any(|s| s.uid == sct_uid))
            {
                let notes = loader::merge_study(dest, study);
                dest.warnings.extend(notes);
                self.settings_gen += 1;
            }
        }
        let r = &view.result;
        self.synth.eval_status = Some(format!(
            "✔ MAE {:.1} HU over {} voxels in {:.0} s",
            r.overall.mae, r.overall.voxels, r.elapsed_secs
        ));
        self.synth.eval_result = Some(view);
    }

    /// The *Image synthesis* module.
    pub(super) fn synthesis_section(&mut self, ui: &mut egui::Ui) {
        let title = egui::RichText::new(SYNTHESIS.name).strong();
        if !self.any_volume() {
            egui::CollapsingHeader::new(title)
                .default_open(false)
                .show(ui, |ui| {
                    ui.weak("Load a workspace with an MR or CBCT series to synthesize a CT from");
                });
            ui.separator();
            return;
        }
        if !self.slots[self.synth.slot].has_volume() {
            self.synth.slot = self.first_volume_slot();
        }
        let busy = self.synth_job.is_some() || self.synth_eval_job.is_some();
        let mut new_slot = None;
        let mut actions = SectionActions::default();
        egui::CollapsingHeader::new(title)
            .default_open(false)
            .show(ui, |ui| {
                new_slot = workspace_row(ui, self.synth.slot, self.volume_slots(), !busy);
                self.synthesis_body(ui, &mut actions);
                ui.separator();
                egui::CollapsingHeader::new("Evaluate against a CT")
                    .default_open(false)
                    .show(ui, |ui| self.sct_eval_body(ui, &mut actions));
            });
        ui.separator();
        if let Some(s) = new_slot {
            self.synth.slot = s;
            self.synth.eval_ref = None;
        }
        if actions.browse {
            self.ask_folder("Model folder", |app, dir| {
                app.models_dir = dir.display().to_string();
            });
        }
        if actions.cancel {
            cancel_if(true, &self.synth_job);
            cancel_if(true, &self.synth_eval_job);
        }
        if actions.run {
            self.start_synthesis();
        }
        if actions.evaluate {
            self.start_sct_eval();
        }
        if actions.save_csv {
            if let Some(v) = &self.synth.eval_result {
                let text = v.csv.clone();
                self.ask_save(
                    "Save the evaluation",
                    "sct_evaluation.csv",
                    None,
                    Some(CSV_FILES),
                    move |app, path| match std::fs::write(&path, text) {
                        Ok(()) => app.notice = Some(format!("Written to {}", path.display())),
                        Err(e) => app.error = Some(format!("Could not write the file: {e}")),
                    },
                );
            }
        }
    }

    /// The synthesis part of the section.
    fn synthesis_body(&mut self, ui: &mut egui::Ui, actions: &mut SectionActions) {
        let slot = self.synth.slot;
        let modality = self.slot_modality(slot);
        let rois = self.active_roi_names(slot);
        let targets = self.copy_targets(slot);
        let root = self.models_root();
        let passes = self.slots[slot]
            .study
            .as_ref()
            .map(|st| synth::passes(&st.volume))
            .unwrap_or(0);
        let running = self.synth_job.as_ref().filter(|_| self.synth_slot == slot);
        let s = &mut self.synth;
        ui.label(
            "Synthesizes a CT from the displayed MR or CBCT with TotalSynth (a 2.5-D UNet++ \
             trained on SynthRAD 2023 and 2025, brain to pelvis) and adds it to the study as \
             a new CT series, on the same frame of reference.",
        );
        ui.separator();
        // ---- the app ---------------------------------------------------
        // A model that no longer exists, or one for the other modality when
        // the displayed series changed, gives way to the one for this series.
        let mut app = App::from_key(&s.app).unwrap_or_else(|| default_app(&modality));
        if s.seeded_for != modality {
            if !modality.is_empty() && !app.modality().accepts(&modality) {
                app = default_app(&modality);
            }
            s.seeded_for = modality.clone();
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Model:");
            egui::ComboBox::from_id_salt("sct_app")
                .selected_text(app.label())
                .width(220.0)
                .show_ui(ui, |ui| {
                    for a in App::all() {
                        ui.selectable_value(&mut app, a, a.label())
                            .on_hover_text(a.detail());
                    }
                });
        });
        s.app = app.key().to_string();
        ui.weak(app.detail());
        if !app.modality().accepts(&modality) {
            ui.label(
                egui::RichText::new(format!(
                    "This model turns {} into a CT; the displayed series is {}.",
                    app.modality().label(),
                    if modality.is_empty() {
                        "of no modality"
                    } else {
                        &modality
                    }
                ))
                .small()
                .color(warn_color(ui.visuals())),
            );
        }
        // ---- members ---------------------------------------------------
        let n = app.checkpoint_count();
        s.chosen.resize(n, true);
        ui.horizontal_wrapped(|ui| {
            ui.label("Checkpoints:");
            ui.radio_value(&mut s.members, Members::Quick, "CV_0 (quick look)");
            ui.radio_value(&mut s.members, Members::All, format!("all {n}"))
                .on_hover_text("The published recipe: the mean of every cross-validation model");
            ui.radio_value(&mut s.members, Members::Chosen, "chosen");
        });
        if s.members == Members::Chosen {
            ui.horizontal_wrapped(|ui| {
                for (k, c) in s.chosen.iter_mut().enumerate() {
                    ui.checkbox(c, format!("CV_{k}"));
                }
            });
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Mirrors:");
            for t in 0..=synth::TTA_MAX {
                ui.radio_value(&mut s.tta, t, t.to_string())
                    .on_hover_text(match t {
                        0 => "No test-time augmentation",
                        1 => "Also the slice flipped top to bottom",
                        _ => {
                            "Also flipped top to bottom, then left to right (the published recipe)"
                        }
                    });
            }
        });
        let ckpts = match s.members {
            Members::Quick => 1,
            Members::All => n,
            Members::Chosen => s.chosen.iter().filter(|c| **c).count(),
        };
        let members = ckpts * (1 + s.tta);
        ui.checkbox(
            &mut s.uncertainty,
            "Spread of the members as a second series",
        )
        .on_hover_text(
            "The voxel-wise standard deviation (HU) of the averaged predictions, written \
                 as its own series to fuse as an overlay. Needs two members or more.",
        );
        // ---- body ----------------------------------------------------------
        ui.horizontal_wrapped(|ui| {
            ui.label("Air outside:");
            let label = match &s.body {
                BodyChoice::Network => "body network (6 mm)".to_string(),
                BodyChoice::Classical => "body contour (threshold)".to_string(),
                BodyChoice::Structure(n) => format!("structure {n}"),
                BodyChoice::None => "nothing set to air".to_string(),
            };
            egui::ComboBox::from_id_salt("sct_body")
                .selected_text(label)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.body, BodyChoice::Network, "body network (6 mm)")
                        .on_hover_text(
                            "The body contour guided by TotalSegmentator's body network (body \
                             on CT and CBCT, body_mr on MR, both at 6 mm), as upstream masks",
                        );
                    ui.selectable_value(
                        &mut s.body,
                        BodyChoice::Classical,
                        "body contour (threshold)",
                    )
                    .on_hover_text("Threshold and morphology: nothing to download");
                    for r in &rois {
                        ui.selectable_value(
                            &mut s.body,
                            BodyChoice::Structure(r.clone()),
                            format!("structure {r}"),
                        );
                    }
                    ui.selectable_value(&mut s.body, BodyChoice::None, "nothing set to air");
                });
        });
        if let BodyChoice::Structure(n) = &s.body {
            if !rois.contains(n) {
                s.body = BodyChoice::Network;
            }
        }
        // ---- where it goes -----------------------------------------------
        if s.show_in.is_some_and(|t| !targets.contains(&t)) {
            s.show_in = targets.first().copied();
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Show it in:");
            for t in &targets {
                ui.radio_value(
                    &mut s.show_in,
                    Some(*t),
                    format!("workspace {}", SLOT_NAMES[*t]),
                )
                .on_hover_text(
                    "Beside its source, on the same frame of reference: fused with no registration",
                );
            }
            ui.radio_value(&mut s.show_in, None, "this workspace");
        });
        // ---- cost ---------------------------------------------------------
        let need = (0..n)
            .filter(|k| match s.members {
                Members::Quick => *k == 0,
                Members::All => true,
                Members::Chosen => s.chosen[*k],
            })
            .collect::<Vec<_>>();
        let bytes = app.download_needed(&need, &root);
        ui.weak(format!(
            "{members} member{} of {passes} network passes each (about 148 GFLOP a pass): \
             minutes per member on a CPU, far less on a GPU.{}",
            if members == 1 { "" } else { "s" },
            if bytes > 0 {
                format!(" {} to download once.", models::human_bytes(bytes))
            } else if app.is_custom() {
                String::new()
            } else {
                " Weights cached ✔".to_string()
            }
        ));
        ui.separator();
        ui.collapsing("Options", |ui| {
            device_row(ui, &mut s.device);
            actions.browse |= models_dir_row(ui, &mut self.models_dir, ModelsEngine::Synth);
        });
        ui.separator();
        let (note, warn) = weights_licence(ModelsEngine::Synth);
        licence_line(ui, note, warn);
        let s = &mut self.synth;
        ui.separator();
        match running {
            Some(job) => actions.cancel |= progress_row(ui, &job.progress),
            None => {
                ui.add_enabled_ui(self.synth_eval_job.is_none(), |ui| {
                    if tip_button(
                        ui,
                        "▶ Synthesize",
                        "Make the synthetic CT of the displayed series",
                    ) {
                        actions.run = true;
                    }
                });
            }
        }
        if let Some(status) = &s.status {
            let r = ui.weak(status);
            if let Some(dir) = &s.status_dir {
                r.on_hover_text(format!("Written to {dir}"));
            }
        }
    }

    /// The evaluation fold.
    fn sct_eval_body(&mut self, ui: &mut egui::Ui, actions: &mut SectionActions) {
        let slot = self.synth.slot;
        let rois = self.active_roi_names(slot);
        // References: what the other workspaces show, and the study's other
        // CT series.
        let mut refs: Vec<(EvalRef, String)> = Vec::new();
        for o in self.volume_slot_list() {
            if o == slot {
                continue;
            }
            let desc = self.slots[o]
                .study
                .as_ref()
                .and_then(|st| st.series.get(st.active_series))
                .map(|se| format!("{} {}", se.modality, se.description))
                .unwrap_or_default();
            refs.push((
                EvalRef::Workspace(o),
                format!("workspace {}: {desc}", SLOT_NAMES[o]),
            ));
        }
        if let Some(st) = self.slots[slot].study.as_ref() {
            let shown = st.series.get(st.active_series).map(|s| s.uid.clone());
            for se in &st.series {
                if se.modality.eq_ignore_ascii_case("CT")
                    && Some(&se.uid) != shown.as_ref()
                    && !se.description.starts_with("sCT")
                    && !se.files.is_empty()
                {
                    refs.push((
                        EvalRef::Series(se.uid.clone()),
                        format!("series: {}", se.description),
                    ));
                }
            }
        }
        let align_note = self
            .synth
            .eval_ref
            .as_ref()
            .map(|r| self.eval_reference(slot, r).map(|(_, _, _, n)| n));
        let running = self
            .synth_eval_job
            .as_ref()
            .filter(|_| self.synth_slot == slot);
        let s = &mut self.synth;
        ui.label(
            "Compares the displayed synthetic CT with a CT of the same patient: MAE, mean \
             error, PSNR and SSIM inside a mask, per HU class, and optionally the \
             water-equivalent thickness and the Dice of TotalSegmentator's structures.",
        );
        if refs.is_empty() {
            ui.weak("Show the reference CT in another workspace, or load it into this study.");
            return;
        }
        if s.eval_ref
            .as_ref()
            .is_none_or(|r| !refs.iter().any(|(x, _)| x == r))
        {
            s.eval_ref = refs.first().map(|(r, _)| r.clone());
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Reference CT:");
            let sel = refs
                .iter()
                .find(|(r, _)| Some(r) == s.eval_ref.as_ref())
                .map(|(_, l)| l.clone())
                .unwrap_or_default();
            egui::ComboBox::from_id_salt("sct_eval_ref")
                .selected_text(sel)
                .width(220.0)
                .show_ui(ui, |ui| {
                    for (r, l) in &refs {
                        ui.selectable_value(&mut s.eval_ref, Some(r.clone()), l);
                    }
                });
        });
        match &align_note {
            Some(Ok(n)) => {
                ui.weak(format!("Aligned: {n}"));
            }
            Some(Err(e)) => {
                ui.label(
                    egui::RichText::new(e)
                        .small()
                        .color(warn_color(ui.visuals())),
                );
            }
            None => {}
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Inside:");
            let label = match &s.eval_mask {
                EvalMask::Body => "the reference's body".to_string(),
                EvalMask::Structure(n) => format!("structure {n}"),
                EvalMask::Covered => "everything the reference covers".to_string(),
            };
            egui::ComboBox::from_id_salt("sct_eval_mask")
                .selected_text(label)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut s.eval_mask, EvalMask::Body, "the reference's body");
                    for r in &rois {
                        ui.selectable_value(
                            &mut s.eval_mask,
                            EvalMask::Structure(r.clone()),
                            format!("structure {r}"),
                        );
                    }
                    ui.selectable_value(
                        &mut s.eval_mask,
                        EvalMask::Covered,
                        "everything the reference covers",
                    );
                });
        });
        if let EvalMask::Structure(n) = &s.eval_mask {
            if !rois.contains(n) {
                s.eval_mask = EvalMask::Body;
            }
        }
        ui.checkbox(&mut s.eval_wet, "Water-equivalent thickness (protons)")
            .on_hover_text(
                "Ray by ray through the mask along the three axes of the grid, through an HU \
                 to relative stopping power table: the range a proton beam would see on each.",
            );
        if s.eval_wet {
            ui.horizontal_wrapped(|ui| {
                ui.label("HU:RSP");
                ui.add(egui::TextEdit::singleline(&mut s.hlut).desired_width(260.0))
                    .on_hover_text(
                        "Pairs HU:RSP separated by commas. The default is a generic curve, not \
                         a scanner's calibration: paste your department's.",
                    );
                if small_tip_button(ui, "↺", "Back to the generic curve") {
                    s.hlut = eval::Hlut::default().text();
                }
            });
        }
        ui.checkbox(&mut s.eval_dice, "Dice of TotalSegmentator's structures")
            .on_hover_text(
                "Runs TotalSegmentator (total, 3 mm) on both images and compares what it \
                 finds: the model card's third metric. Minutes on a CPU, 140 MB to download \
                 once.",
            );
        ui.checkbox(
            &mut s.eval_diff,
            "Add the difference (sCT minus CT) as a series",
        );
        ui.separator();
        match running {
            Some(job) => actions.cancel |= progress_row(ui, &job.progress),
            None => {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            self.synth_job.is_none() && matches!(align_note, Some(Ok(_))),
                            egui::Button::new("▶ Evaluate"),
                        )
                        .on_hover_text("Compare the displayed series with the reference")
                        .clicked()
                    {
                        actions.evaluate = true;
                    }
                    if s.eval_result.is_some()
                        && small_tip_button(ui, "💾 Export CSV", "Save the numbers as a table")
                    {
                        actions.save_csv = true;
                    }
                });
            }
        }
        if let Some(st) = &s.eval_status {
            ui.weak(st);
        }
        if let Some(v) = &s.eval_result {
            eval_table(ui, v);
        }
    }
}

/// What the section's buttons asked for this frame.
#[derive(Default)]
struct SectionActions {
    run: bool,
    evaluate: bool,
    cancel: bool,
    browse: bool,
    save_csv: bool,
}

/// The evaluation's numbers.
fn eval_table(ui: &mut egui::Ui, v: &EvalView) {
    let r = &v.result;
    ui.weak(&v.what);
    egui::Grid::new("sct_eval_grid")
        .num_columns(5)
        .striped(true)
        .spacing([10.0, 2.0])
        .show(ui, |ui| {
            for h in ["", "voxels", "MAE", "ME", "RMSE"] {
                ui.label(egui::RichText::new(h).strong());
            }
            ui.end_row();
            let row = |ui: &mut egui::Ui, name: &str, m: &eval::Metrics| {
                ui.label(name);
                ui.label(m.voxels.to_string());
                ui.label(format!("{:.1} HU", m.mae));
                ui.label(format!("{:+.1} HU", m.me));
                ui.label(format!("{:.1} HU", m.rmse));
                ui.end_row();
            };
            row(ui, "all", &r.overall);
            for c in &r.classes {
                if c.m.voxels > 0 {
                    row(ui, c.name, &c.m);
                }
            }
        });
    ui.label(format!(
        "PSNR {:.2} dB, SSIM {:.4} (range -1024 to 3071 HU)",
        r.psnr_db, r.ssim
    ));
    if !r.wet.is_empty() {
        egui::Grid::new("sct_wet_grid")
            .num_columns(4)
            .striped(true)
            .spacing([10.0, 2.0])
            .show(ui, |ui| {
                for (h, tip) in [
                    ("ΔWET", "along the axis of the grid"),
                    ("mean", "mean of sCT minus CT"),
                    ("mean |Δ|", "and relative to the CT's WET"),
                    ("95% |Δ|", "95th percentile of the absolute difference"),
                ] {
                    ui.label(egui::RichText::new(h).strong()).on_hover_text(tip);
                }
                ui.end_row();
                for w in &r.wet {
                    // `i (columns), left-right`: the direction says enough here.
                    let dir = w.axis.rsplit(", ").next().unwrap_or(&w.axis);
                    ui.label(dir).on_hover_text(&w.axis);
                    ui.label(format!("{:+.1} mm", w.mean_diff_mm));
                    ui.label(format!(
                        "{:.1} mm ({:.1}%)",
                        w.mean_abs_diff_mm, w.mean_abs_rel_pct
                    ));
                    ui.label(format!("{:.1} mm", w.p95_abs_diff_mm));
                    ui.end_row();
                }
            });
    }
    if let Some(m) = r.mean_dice {
        ui.label(format!(
            "Mean Dice {:.3} over {} structures",
            m,
            r.dice.len()
        ));
        egui::CollapsingHeader::new("Per structure")
            .id_salt("sct_dice_rows")
            .show(ui, |ui| {
                for d in &r.dice {
                    ui.weak(format!(
                        "{}: {:.3} ({:.1} / {:.1} cm³)",
                        d.name, d.dice, d.cm3_ct, d.cm3_sct
                    ));
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_section_names_itself_like_the_tools() {
        assert_eq!(SYNTHESIS.menu_entry(), "🔮 Image synthesis");
        assert_eq!(default_app("MR"), App::Mr);
        assert_eq!(default_app("CT"), App::Cbct);
        assert_eq!(app_of("nope"), App::Mr);
        assert_eq!(app_of("sct_mr_cbct"), App::MrCbct);
        let s = SynthState::default();
        assert_eq!((s.members, s.tta), (Members::Quick, 0));
    }
}
