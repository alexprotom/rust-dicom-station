//! *Tools ▶ 4D motion / ITV analysis*: the automated per-phase pipeline.
//!
//! One run reproduces the whole 4DCT motion workflow on a recognised 4D
//! group: the reference phase is registered to every other phase (rigidly,
//! and deformably on top of the rigid result), the chosen targets are
//! propagated through each transform, and what comes back is measured -
//! centroid trajectories, peak-to-peak amplitudes, drift against a
//! reference structure (typically the heart) with direction-wise
//! correlation, per-phase registration quality, and motion-encompassing
//! ITVs stored as segmentations on the reference phase.
//!
//! The dialog's settings survive as a *recipe*: the same targets (matched
//! by name), models and options can be re-applied to the other dataset or
//! to the next study with two clicks, which is what makes the workflow
//! practical over a cohort rather than a single case.

use crate::motion::MotionModel;
use crate::registration::RegParams;
pub(super) use crate::workflow::motion::{run as run_motion, MotionOutcome, MotionRequest};
use crate::workflow::select::Structure;
use crate::workflow::{self};

use super::combine_win::ItemRef;
use super::*;

/// The tool window's state; it stays open across runs.
pub(super) struct MotionDialog {
    pub slot: usize,
    /// Index into the study's `fourd_groups`.
    pub group: usize,
    /// Member position of the reference phase within the group.
    pub reference: usize,
    /// Ticks parallel to [`ViewerApp::combine_candidates`].
    pub targets: Vec<bool>,
    /// Candidate index of the reference structure (drift / correlation).
    pub ref_struct: Option<usize>,
    pub rigid: bool,
    pub deformable: bool,
    pub build_itv: bool,
    /// Uniform margin added to each ITV, mm.
    pub itv_margin_mm: f64,
    /// Also keep every propagated per-phase mask as a segmentation series
    /// on its phase.
    pub keep_phase_segs: bool,
    // Registration settings for the per-phase runs.
    pub levels: usize,
    pub iterations: usize,
    pub samples: usize,
    pub grid_mm: f64,
    pub threshold: f32,
    pub status: Option<String>,
}

/// The dialog's transferable part: what to analyse and how, with targets
/// remembered by *name* so the same recipe applies to another dataset.
#[derive(Clone)]
pub(super) struct MotionRecipe {
    pub targets: Vec<String>,
    pub ref_struct: Option<String>,
    pub rigid: bool,
    pub deformable: bool,
    pub build_itv: bool,
    pub itv_margin_mm: f64,
    pub keep_phase_segs: bool,
    pub levels: usize,
    pub iterations: usize,
    pub samples: usize,
    pub grid_mm: f64,
    pub threshold: f32,
}

impl ViewerApp {
    /// Open the tool for `slot`, optionally pre-selecting a 4D group.
    pub(super) fn open_motion_dialog(&mut self, slot: usize, group: Option<usize>) {
        let n_cand = self.combine_candidates(slot).len();
        let (levels, iterations, samples, grid_mm, threshold) = (
            self.reg_levels,
            self.reg_iterations,
            self.reg_samples,
            self.reg_grid_mm,
            self.reg_threshold,
        );
        let mut d = MotionDialog {
            slot,
            group: group.unwrap_or(0),
            reference: 0,
            targets: vec![false; n_cand],
            ref_struct: None,
            rigid: true,
            deformable: true,
            build_itv: true,
            itv_margin_mm: 0.0,
            keep_phase_segs: false,
            levels,
            iterations,
            samples,
            grid_mm,
            threshold,
            status: None,
        };
        if let Some(study) = self.slots[slot].study.as_ref() {
            if let Some(g) = study.fourd_groups.get(d.group) {
                d.reference = g.default_reference().unwrap_or(0);
            }
        }
        self.motion_dialog = Some(d);
    }

    /// The name of one candidate item (without its set), for recipes.
    pub(super) fn item_name(&self, slot: usize, item: ItemRef) -> Option<String> {
        let study = self.slots[slot].study.as_ref()?;
        match item.kind {
            SetKind::Structures => Some(
                study
                    .structure_sets
                    .get(item.set)?
                    .rois
                    .get(item.idx)?
                    .name
                    .clone(),
            ),
            SetKind::Segmentations => Some(
                study
                    .seg_series
                    .get(item.set)?
                    .segs
                    .get(item.idx)?
                    .name
                    .clone(),
            ),
        }
    }

    /// Freeze one candidate for the worker thread.
    fn snapshot(&self, slot: usize, item: ItemRef) -> Option<Structure> {
        let study = self.slots[slot].study.as_ref()?;
        match item.kind {
            SetKind::Structures => Some(Structure::from_roi(
                study.structure_sets.get(item.set)?.rois.get(item.idx)?,
            )),
            SetKind::Segmentations => {
                Structure::from_segment(study.seg_series.get(item.set)?, item.idx)
            }
        }
    }

    /// The current dialog as a name-based recipe.
    fn motion_recipe_of(&self, d: &MotionDialog) -> MotionRecipe {
        let cands = self.combine_candidates(d.slot);
        let name_of = |i: usize| {
            cands
                .get(i)
                .and_then(|(r, _)| self.item_name(d.slot, *r))
                .unwrap_or_default()
        };
        MotionRecipe {
            targets: d
                .targets
                .iter()
                .enumerate()
                .filter(|(_, &on)| on)
                .map(|(i, _)| name_of(i))
                .filter(|n| !n.is_empty())
                .collect(),
            ref_struct: d.ref_struct.map(name_of).filter(|n| !n.is_empty()),
            rigid: d.rigid,
            deformable: d.deformable,
            build_itv: d.build_itv,
            itv_margin_mm: d.itv_margin_mm,
            keep_phase_segs: d.keep_phase_segs,
            levels: d.levels,
            iterations: d.iterations,
            samples: d.samples,
            grid_mm: d.grid_mm,
            threshold: d.threshold,
        }
    }

    /// Tick the dialog's lists from a recipe, matching items by name.
    fn apply_motion_recipe(&self, d: &mut MotionDialog, r: &MotionRecipe) {
        let cands = self.combine_candidates(d.slot);
        d.targets = vec![false; cands.len()];
        d.ref_struct = None;
        for (i, (item, _)) in cands.iter().enumerate() {
            let Some(name) = self.item_name(d.slot, *item) else {
                continue;
            };
            if r.targets.contains(&name) {
                d.targets[i] = true;
            }
            if d.ref_struct.is_none() && r.ref_struct.as_deref() == Some(name.as_str()) {
                d.ref_struct = Some(i);
            }
        }
        d.rigid = r.rigid;
        d.deformable = r.deformable;
        d.build_itv = r.build_itv;
        d.itv_margin_mm = r.itv_margin_mm;
        d.keep_phase_segs = r.keep_phase_segs;
        d.levels = r.levels;
        d.iterations = r.iterations;
        d.samples = r.samples;
        d.grid_mm = r.grid_mm;
        d.threshold = r.threshold;
    }

    /// Start the pipeline on a worker thread.
    fn start_motion_run(&mut self) {
        if self.motion_job.is_some() {
            return;
        }
        let Some(d) = &self.motion_dialog else {
            return;
        };
        let slot = d.slot;
        let req = match self.build_motion_request(d) {
            Ok(r) => r,
            Err(e) => {
                if let Some(d) = &mut self.motion_dialog {
                    d.status = Some(format!("{e:#}"));
                }
                return;
            }
        };
        self.motion_recipe = Some(self.motion_recipe_of(d));
        self.motion_slot = slot;
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.motion_job = Some(Job::spawn(progress, move |p| (slot, run_motion(req, p))));
    }

    /// Snapshot everything the worker needs, or say what is missing.
    fn build_motion_request(&self, d: &MotionDialog) -> anyhow::Result<MotionRequest> {
        use anyhow::{bail, Context};
        let slot = d.slot;
        let Some(study) = self.slots[slot].study.as_ref() else {
            bail!("dataset {} is not loaded", SLOT_NAMES[slot]);
        };
        let Some(group) = study.fourd_groups.get(d.group) else {
            bail!("no 4D group selected");
        };
        let phases = workflow::phases_of(group, &study.series)?;
        // `d.reference` is a member position; the phases list skips the
        // reconstructions, so find where that member landed in it.
        let reference = group
            .phase_members()
            .iter()
            .position(|&mi| mi == d.reference)
            .context("the reference must be one of the phases")?;

        let cands = self.combine_candidates(slot);
        let mut targets = Vec::new();
        for (i, &on) in d.targets.iter().enumerate() {
            if !on {
                continue;
            }
            let (item, label) = &cands[i];
            targets.push(
                self.snapshot(slot, *item)
                    .with_context(|| format!("'{label}' is gone"))?,
            );
        }
        if targets.is_empty() {
            bail!("tick at least one target structure");
        }
        let ref_struct = match d.ref_struct {
            Some(i) => {
                let (item, label) = cands
                    .get(i)
                    .context("the reference-structure choice is stale - pick it again")?;
                let s = self
                    .snapshot(slot, *item)
                    .with_context(|| format!("'{label}' is gone"))?;
                if targets.iter().any(|t| t.name == s.name) {
                    bail!("'{}' cannot be both target and reference", s.name);
                }
                Some(s)
            }
            None => None,
        };
        let mut models = Vec::new();
        if d.rigid {
            models.push(MotionModel::Rigid);
        }
        if d.deformable {
            models.push(MotionModel::Deformable);
        }
        if models.is_empty() {
            bail!("choose at least one model (rigid / deformable)");
        }
        let params = RegParams {
            method: RegMethod::ElastixRigid,
            levels: d.levels,
            iterations: d.iterations,
            samples: d.samples,
            grid_spacing_mm: d.grid_mm,
            fixed_threshold: d.threshold,
            ..RegParams::default()
        };
        Ok(MotionRequest {
            // Numbered, so two runs on the same group stay distinguishable
            // in the results window's pick lists.
            run_name: format!(
                "#{} {} · {} · ref {}",
                self.motion_reports.len() + 1,
                SLOT_NAMES[slot],
                group.name,
                phases[reference].0
            ),
            slot_name: SLOT_NAMES[slot].to_string(),
            patient: study.meta.patient_name.replace('^', " "),
            group_name: group.name.clone(),
            study_uid: group.study_uid.clone(),
            phases,
            reference,
            targets,
            ref_struct,
            models,
            build_itv: d.build_itv,
            itv_margin_mm: d.itv_margin_mm,
            keep_phase_segs: d.keep_phase_segs,
            params,
        })
    }

    /// Land a finished run: file its segmentations, keep its report, open
    /// the results.
    pub(super) fn on_motion_done(&mut self, slot: usize, outcome: MotionOutcome) {
        let mut lines = vec![format!(
            "Motion analysis finished: {}",
            outcome.report.run_name
        )];
        // The report is kept whatever happened to the dataset meanwhile -
        // it is self-contained - but segmentations only land in the study
        // the run analysed.
        let still_there = self.slots[slot]
            .study
            .as_ref()
            .is_some_and(|st| st.series.iter().any(|se| se.study_uid == outcome.study_uid));
        if !still_there {
            lines.push("The dataset changed while it ran - segmentations were discarded.".into());
            self.motion_reports.push(outcome.report);
            self.motion_sel = self.motion_reports.len() - 1;
            self.motion_results_open = true;
            if let Some(d) = &mut self.motion_dialog {
                d.status = Some(lines.join(" "));
            }
            return;
        }
        if let Some(study) = self.slots[slot].study.as_mut() {
            let mut add = |o: crate::workflow::motion::OutSeries| {
                study.seg_series.push(o.into_seg_series(&outcome.study_uid));
            };
            if let Some(itv) = outcome.itv_series {
                lines.push(format!(
                    "ITVs stored as segmentation series '{}' on the reference phase.",
                    itv.label
                ));
                add(itv);
            }
            let n_phase = outcome.phase_series.len();
            for o in outcome.phase_series {
                add(o);
            }
            if n_phase > 0 {
                lines.push(format!("{n_phase} per-phase series kept."));
            }
        }
        self.rebind_seg_series(slot);
        self.motion_reports.push(outcome.report);
        self.motion_sel = self.motion_reports.len() - 1;
        self.motion_results_open = true;
        if let Some(d) = &mut self.motion_dialog {
            d.status = Some(lines.join(" "));
        }
    }

    // ---- the window --------------------------------------------------------

    pub(super) fn motion_window(&mut self, ctx: &egui::Context) {
        let Some(d) = &self.motion_dialog else {
            return;
        };
        let slot = d.slot;
        if !self.slots[slot].has_volume() {
            self.motion_dialog = None;
            return;
        }
        let cands = self.combine_candidates(slot);
        // (group index, group name, [(member position, phase label)]).
        type GroupRow = (usize, String, Vec<(usize, String)>);
        let groups: Vec<GroupRow> = self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                st.fourd_groups
                    .iter()
                    .enumerate()
                    .map(|(gi, g)| {
                        let phase_positions = g
                            .members
                            .iter()
                            .enumerate()
                            .filter(|(_, m)| m.role == crate::fourd::Role::Phase)
                            .map(|(mi, m)| (mi, m.label.clone()))
                            .collect();
                        (gi, g.name.clone(), phase_positions)
                    })
                    .collect()
            })
            .unwrap_or_default();

        let running = self
            .motion_job
            .as_ref()
            .filter(|_| self.motion_slot == slot);
        let progress = running.map(|j| j.progress.clone());
        let has_recipe = self.motion_recipe.is_some();

        let mut run = false;
        let mut cancel = false;
        let mut close = false;
        let mut apply_recipe = false;
        let has = [self.slots[0].has_volume(), self.slots[1].has_volume()];
        let mut switch: Option<usize> = None;
        let mut open = true;

        let d = self.motion_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "motion",
            MOTION.title(slot),
            &mut open,
            detach::WinOpts::default(),
            |ui| {
                switch = dataset_row(ui, slot, has, running.is_none());
                ui.label(
                    "Register the reference phase to every phase of a 4D group, carry the \
                     targets across, and measure their motion - trajectories, amplitudes, \
                     drift against a reference structure, and the ITV.",
                );
                ui.add_space(4.0);
                if groups.is_empty() {
                    ui.colored_label(
                        warn_color(ui.visuals()),
                        "No 4D group in this dataset. Phases are recognised from the series \
                         descriptions (e.g. \"… 30%\"); series can also be grouped by hand \
                         from the data tree (right-click a series > 4D group).",
                    );
                    return;
                }
                d.group = d.group.min(groups.len() - 1);
                let (_, gname, _) = &groups[d.group];
                ui.horizontal(|ui| {
                    ui.label("4D group:");
                    egui::ComboBox::from_id_salt("motion_group")
                        .width(280.0)
                        .selected_text(gname.clone())
                        .show_ui(ui, |ui| {
                            for (gi, name, _) in &groups {
                                ui.selectable_value(&mut d.group, *gi, name);
                            }
                        });
                });
                let (_, _, phases) = &groups[d.group];
                if !phases.iter().any(|(mi, _)| *mi == d.reference) {
                    d.reference = phases.first().map(|(mi, _)| *mi).unwrap_or(0);
                }
                ui.horizontal(|ui| {
                    ui.label("Reference phase:");
                    let sel = phases
                        .iter()
                        .find(|(mi, _)| *mi == d.reference)
                        .map(|(_, l)| l.clone())
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("motion_ref")
                        .selected_text(sel)
                        .show_ui(ui, |ui| {
                            for (mi, label) in phases {
                                ui.selectable_value(&mut d.reference, *mi, label);
                            }
                        });
                    ui.label("·").on_hover_text(
                        "Targets are defined on this phase and carried to the others",
                    );
                });
                ui.separator();

                ui.label("Targets (defined on / resampled to the reference phase):");
                d.targets.resize(cands.len(), false);
                // The candidate list shrinks when sets are removed while
                // the window is open; a stale pick must not survive it.
                if d.ref_struct.is_some_and(|i| i >= cands.len()) {
                    d.ref_struct = None;
                }
                egui::ScrollArea::vertical()
                    .id_salt("motion_targets")
                    .max_height(120.0)
                    .show(ui, |ui| {
                        for (i, (_, label)) in cands.iter().enumerate() {
                            ui.checkbox(&mut d.targets[i], label);
                        }
                        if cands.is_empty() {
                            ui.weak("no structures or segmentations in this dataset");
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("Reference structure:");
                    let sel = d
                        .ref_struct
                        .and_then(|i| cands.get(i).map(|(_, l)| l.clone()))
                        .unwrap_or_else(|| "(none)".into());
                    egui::ComboBox::from_id_salt("motion_refstruct")
                        .width(260.0)
                        .selected_text(sel)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut d.ref_struct, None, "(none)");
                            for (i, (_, label)) in cands.iter().enumerate() {
                                ui.selectable_value(&mut d.ref_struct, Some(i), label);
                            }
                        });
                })
                .response
                .on_hover_text(
                    "Carried along for target-reference drift and direction-wise \
                     correlation - typically the heart for cardiac targets",
                );
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Models:");
                    ui.checkbox(&mut d.rigid, "rigid");
                    ui.checkbox(&mut d.deformable, "deformable")
                        .on_hover_text("B-spline refinement on top of the rigid result");
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut d.build_itv, "Build ITV").on_hover_text(
                        "Union of the target over all phases, on the reference phase",
                    );
                    ui.add_enabled(
                        d.build_itv,
                        egui::DragValue::new(&mut d.itv_margin_mm)
                            .speed(0.5)
                            .range(0.0..=30.0)
                            .prefix("+ ")
                            .suffix(" mm"),
                    )
                    .on_hover_text("Uniform margin added to the union");
                });
                ui.checkbox(&mut d.keep_phase_segs, "Keep per-phase segmentations")
                    .on_hover_text(
                        "Store every propagated mask as a segmentation series on its phase \
                         - one series per phase",
                    );
                egui::CollapsingHeader::new("Registration settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::Grid::new("motion_reg").num_columns(2).show(ui, |ui| {
                            ui.label("Resolution levels:");
                            ui.add(egui::DragValue::new(&mut d.levels).range(1..=5));
                            ui.end_row();
                            ui.label("Iterations / level:");
                            ui.add(egui::DragValue::new(&mut d.iterations).range(50..=2000));
                            ui.end_row();
                            ui.label("Samples / iteration:");
                            ui.add(egui::DragValue::new(&mut d.samples).range(500..=20000));
                            ui.end_row();
                            ui.label("B-spline grid (mm):");
                            ui.add(
                                egui::DragValue::new(&mut d.grid_mm)
                                    .speed(1.0)
                                    .range(8.0..=100.0),
                            );
                            ui.end_row();
                            ui.label("Sampling threshold (HU):");
                            ui.add(egui::DragValue::new(&mut d.threshold).speed(10.0));
                            ui.end_row();
                        });
                        ui.weak(
                            "Elastix rigid, then B-spline refinement - the same engines \
                                 as the Registration panel.",
                        );
                    });
                ui.add_space(4.0);
                if let Some(status) = &d.status {
                    ui.label(status.clone());
                }
                match &progress {
                    Some(p) => {
                        if seg_engines::progress_row(ui, p) {
                            cancel = true;
                        }
                    }
                    None => {
                        ui.horizontal(|ui| {
                            if ui.button("▶ Analyse").clicked() {
                                run = true;
                            }
                            if ui
                                .add_enabled(has_recipe, egui::Button::new("Apply last recipe"))
                                .on_hover_text(
                                    "Tick the same targets (matched by name) and re-use the \
                                     options of the previous run - for the other dataset or \
                                     the next study",
                                )
                                .clicked()
                            {
                                apply_recipe = true;
                            }
                            if ui.button("Close").clicked() {
                                close = true;
                            }
                        });
                    }
                }
            },
        );
        if cancel {
            if let Some(job) = &self.motion_job {
                job.progress.cancel();
            }
        }
        if let Some(s) = switch {
            self.open_motion_dialog(s, None);
            return;
        }
        if apply_recipe {
            if let (Some(mut d), Some(r)) = (self.motion_dialog.take(), self.motion_recipe.clone())
            {
                self.apply_motion_recipe(&mut d, &r);
                self.motion_dialog = Some(d);
            }
        }
        if run {
            self.start_motion_run();
        }
        if close || !open {
            self.motion_dialog = None;
        }
    }
}
