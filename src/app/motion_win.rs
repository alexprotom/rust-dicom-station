//! *Tools ▶ 4D motion / ITV analysis*: the automated per-phase pipeline.
//!
//! One run reproduces the whole 4DCT motion workflow on a recognised 4D
//! group: the reference phase is registered to every other phase (rigidly,
//! and deformably on top of the rigid result), the chosen targets are
//! propagated through each transform, and what comes back is measured —
//! centroid trajectories, peak-to-peak amplitudes, drift against a
//! reference structure (typically the heart) with direction-wise
//! correlation, per-phase registration quality, and motion-encompassing
//! ITVs stored as segmentations on the reference phase.
//!
//! The dialog's settings survive as a *recipe*: the same targets (matched
//! by name), models and options can be re-applied to the other dataset or
//! to the next study with two clicks, which is what makes the workflow
//! practical over a cohort rather than a single case.

use crate::dicomseg::{resample_mask, SegSeries};
use crate::loader::SeriesInfo;
use crate::morphology;
use crate::motion::{
    self, AxisCorrelation, ItvResult, MotionModel, MotionReport, PhaseSample, RegQa, Track,
};
use crate::propagate::{self, Subject};
use crate::registration::RegParams;
use crate::rtstruct::Roi;
use crate::volume::Grid;

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

/// One structure frozen for the worker thread.
struct Snapshot {
    name: String,
    color: [u8; 3],
    src: Src,
}

/// Where the structure's geometry comes from.
enum Src {
    Contours(Roi),
    Mask { mask: Vec<u8>, grid: Grid },
}

/// Everything a run needs, snapshotted when it starts.
struct MotionRequest {
    run_name: String,
    slot_name: String,
    patient: String,
    group_name: String,
    study_uid: String,
    /// The phase members, in temporal order: label + the series to load.
    phases: Vec<(String, SeriesInfo)>,
    /// Index of the reference phase within `phases`.
    reference: usize,
    targets: Vec<Snapshot>,
    ref_struct: Option<Snapshot>,
    models: Vec<MotionModel>,
    build_itv: bool,
    itv_margin_mm: f64,
    keep_phase_segs: bool,
    params: RegParams,
}

/// One finished segmentation series to add to the study.
pub(super) struct OutSeries {
    pub label: String,
    pub grid: Grid,
    pub referenced_series_uid: String,
    pub segs: Vec<(String, [u8; 3], Vec<u8>)>,
}

/// What a finished run hands back.
pub(super) struct MotionOutcome {
    pub report: MotionReport,
    /// The ITVs, on the reference phase's lattice.
    pub itv_series: Option<OutSeries>,
    /// Per-phase propagated masks, when the run kept them.
    pub phase_series: Vec<OutSeries>,
    pub study_uid: String,
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
    fn snapshot(&self, slot: usize, item: ItemRef) -> Option<Snapshot> {
        let study = self.slots[slot].study.as_ref()?;
        match item.kind {
            SetKind::Structures => {
                let roi = study.structure_sets.get(item.set)?.rois.get(item.idx)?;
                Some(Snapshot {
                    name: roi.name.clone(),
                    color: roi.color,
                    src: Src::Contours(roi.clone()),
                })
            }
            SetKind::Segmentations => {
                let ser = study.seg_series.get(item.set)?;
                let seg = ser.segs.get(item.idx)?;
                Some(Snapshot {
                    name: seg.name.clone(),
                    color: seg.color,
                    src: Src::Mask {
                        mask: seg.mask.clone(),
                        grid: ser.grid.clone(),
                    },
                })
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
        progress.set("starting…");
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
        let resolved = group.resolve(&study.series);
        let mut phases = Vec::new();
        let mut reference = None;
        for (mi, m) in group.members.iter().enumerate() {
            if m.role != crate::fourd::Role::Phase {
                continue;
            }
            let Some(si) = resolved[mi] else {
                bail!("phase '{}' has no series any more", m.label);
            };
            if mi == d.reference {
                reference = Some(phases.len());
            }
            phases.push((m.label.clone(), study.series[si].clone()));
        }
        if phases.len() < 2 {
            bail!("the group needs at least two phases");
        }
        let reference = reference.context("the reference must be one of the phases")?;

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
                    .context("the reference-structure choice is stale — pick it again")?;
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
        // The report is kept whatever happened to the dataset meanwhile —
        // it is self-contained — but segmentations only land in the study
        // the run analysed.
        let still_there = self.slots[slot]
            .study
            .as_ref()
            .is_some_and(|st| st.series.iter().any(|se| se.study_uid == outcome.study_uid));
        if !still_there {
            lines.push("The dataset changed while it ran — segmentations were discarded.".into());
            self.motion_reports.push(outcome.report);
            self.motion_sel = self.motion_reports.len() - 1;
            self.motion_results_open = true;
            if let Some(d) = &mut self.motion_dialog {
                d.status = Some(lines.join(" "));
            }
            return;
        }
        if let Some(study) = self.slots[slot].study.as_mut() {
            let mut add = |o: OutSeries| {
                let mut ser = SegSeries::new(
                    o.label,
                    o.grid,
                    o.referenced_series_uid,
                    outcome.study_uid.clone(),
                );
                for (name, color, mask) in o.segs {
                    ser.segs.push(Segmentation::from_label_map(
                        name,
                        color,
                        ser.grid.dims,
                        &mask,
                        1,
                    ));
                }
                study.seg_series.push(ser);
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
        if self.slots[slot].study.is_none() {
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
        let mut open = true;

        let d = self.motion_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "motion",
            MOTION.title(slot),
            &mut open,
            detach::WinOpts::default(),
            |ui| {
                ui.label(
                    "Register the reference phase to every phase of a 4D group, carry the \
                     targets across, and measure their motion — trajectories, amplitudes, \
                     drift against a reference structure, and the ITV.",
                );
                ui.add_space(4.0);
                if groups.is_empty() {
                    ui.colored_label(
                        warn_color(ui.visuals()),
                        "No 4D group in this dataset. Phases are recognised from the series \
                         descriptions (e.g. \"… 30%\"); series can also be grouped by hand \
                         from the data tree (right-click a series ▸ 4D group).",
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
                    "Carried along for target–reference drift and direction-wise \
                     correlation — typically the heart for cardiac targets",
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
                         — one series per phase",
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
                            "Elastix rigid, then B-spline refinement — the same engines \
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
                                     options of the previous run — for the other dataset or \
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

// ---- the pipeline itself ---------------------------------------------------

/// Rasterize / resample a snapshot onto `grid`.
fn mask_on(s: &Snapshot, grid: &Grid) -> anyhow::Result<Vec<u8>> {
    use anyhow::{bail, Context};
    let mask = match &s.src {
        Src::Contours(roi) => segmentation::rasterize_roi(grid, roi)
            .with_context(|| format!("'{}' has no contour inside the reference phase", s.name))?,
        Src::Mask { mask, grid: from } => {
            if from.matches(grid) {
                mask.clone()
            } else {
                resample_mask(mask, from, grid)
            }
        }
    };
    if mask.iter().all(|&v| v == 0) {
        bail!("'{}' is empty on the reference phase", s.name);
    }
    Ok(mask)
}

/// The whole per-phase pipeline, on the worker thread.
fn run_motion(req: MotionRequest, p: &Progress) -> anyhow::Result<MotionOutcome> {
    use anyhow::{anyhow, bail};
    let n = req.phases.len();
    let n_targets = req.targets.len();
    let cancelled = || anyhow!(progress::CANCELLED);

    // The reference phase.
    p.set_phase(0.0, 0.04);
    let (ref_vol, _, _) = loader::load_series_volume(&req.phases[req.reference].1, p)?;
    let ref_grid = ref_vol.grid();

    // All structures on the reference lattice: targets first, then the
    // reference structure.
    let mut subjects: Vec<Subject> = Vec::new();
    for s in req.targets.iter().chain(req.ref_struct.iter()) {
        subjects.push(Subject {
            name: s.name.clone(),
            color: s.color,
            mask: mask_on(s, &ref_grid)?,
        });
    }
    let n_subjects = subjects.len();

    // samples[model][subject][phase] — filled as the phases are processed.
    let mut samples: Vec<Vec<Vec<Option<PhaseSample>>>> =
        vec![vec![vec![None; n]; n_subjects]; req.models.len()];
    // Union accumulators on the reference grid, [model][target].
    let ref_n = ref_grid.dims[0] * ref_grid.dims[1] * ref_grid.dims[2];
    let mut unions: Vec<Vec<Vec<u8>>> = if req.build_itv {
        vec![vec![vec![0u8; ref_n]; n_targets]; req.models.len()]
    } else {
        Vec::new()
    };
    let mut qa: Vec<RegQa> = Vec::new();
    let mut phase_series: Vec<OutSeries> = Vec::new();

    // The reference phase's own samples (and its contribution to the ITV).
    for (mi, _) in req.models.iter().enumerate() {
        for (si, subject) in subjects.iter().enumerate() {
            let c = motion::centroid_mm(&subject.mask, &ref_grid)
                .ok_or_else(|| anyhow!("'{}' is empty", subject.name))?;
            samples[mi][si][req.reference] = Some(PhaseSample {
                phase: req.phases[req.reference].0.clone(),
                centroid: c,
                volume_cm3: motion::volume_cm3(&subject.mask, &ref_grid),
            });
            if req.build_itv && si < n_targets {
                motion::union_into(&mut unions[mi][si], &subject.mask);
            }
        }
    }

    // Every other phase: register, propagate, measure.
    let others: Vec<usize> = (0..n).filter(|&i| i != req.reference).collect();
    for (oi, &pi) in others.iter().enumerate() {
        if p.cancelled() {
            return Err(cancelled());
        }
        let base = 0.05 + 0.9 * oi as f32 / others.len() as f32;
        let span = 0.9 / others.len() as f32;
        let (label, series) = &req.phases[pi];
        p.set_phase(base, span * 0.15);
        p.set(format!("Phase {label}: loading…"));
        let (vol, _, _) = loader::load_series_volume(series, p)?;
        let phase_grid = vol.grid();

        p.set_phase(base + span * 0.15, span * 0.35);
        p.set(format!("Phase {label}: rigid registration…"));
        let mut params = req.params.clone();
        params.method = RegMethod::ElastixRigid;
        let rigid = registration::register(&ref_vol, &vol, &params, p)?;
        qa.push(RegQa {
            phase: label.clone(),
            model: MotionModel::Rigid,
            metric_line: rigid.metric_line(),
            folding_pct: 100.0 * rigid.analysis.jacobian.folded,
            disp_p95_mm: rigid.analysis.displacement.p95,
        });

        let deformable = if req.models.contains(&MotionModel::Deformable) {
            p.set_phase(base + span * 0.5, span * 0.35);
            p.set(format!("Phase {label}: deformable refinement…"));
            let mut params = req.params.clone();
            params.method = RegMethod::ElastixBSpline;
            params.start = Some(rigid.transform.clone());
            let def = registration::register(&ref_vol, &vol, &params, p)?;
            qa.push(RegQa {
                phase: label.clone(),
                model: MotionModel::Deformable,
                metric_line: def.metric_line(),
                folding_pct: 100.0 * def.analysis.jacobian.folded,
                disp_p95_mm: def.analysis.displacement.p95,
            });
            Some(def)
        } else {
            None
        };

        p.set_phase(base + span * 0.85, span * 0.15);
        let mut phase_out: Vec<(String, [u8; 3], Vec<u8>)> = Vec::new();
        for (mi, model) in req.models.iter().enumerate() {
            let transform = match model {
                MotionModel::Rigid => &rigid.transform,
                MotionModel::Deformable => &deformable.as_ref().expect("built above").transform,
            };
            p.set(format!("Phase {label}: propagating ({})…", model.label()));
            // The transform maps reference → phase; landing on the phase
            // lattice therefore samples through the inverse.
            let props = propagate::propagate(&ref_vol, &vol, transform, true, &subjects, p)?;
            for (si, prop) in props.iter().enumerate() {
                let c = motion::centroid_mm(&prop.mask, &phase_grid).ok_or_else(|| {
                    anyhow!(
                        "'{}' vanished on phase {label} ({})",
                        prop.name,
                        model.label()
                    )
                })?;
                samples[mi][si][pi] = Some(PhaseSample {
                    phase: label.clone(),
                    centroid: c,
                    volume_cm3: prop.result_cm3,
                });
                if req.build_itv && si < n_targets {
                    let on_ref = resample_mask(&prop.mask, &phase_grid, &ref_grid);
                    motion::union_into(&mut unions[mi][si], &on_ref);
                }
                if req.keep_phase_segs {
                    phase_out.push((
                        format!("{} ({label}, {})", prop.name, model.label()),
                        prop.color,
                        prop.mask.clone(),
                    ));
                }
            }
        }
        if !phase_out.is_empty() {
            phase_series.push(OutSeries {
                label: format!("4D {label} — {}", req.group_name),
                grid: phase_grid,
                referenced_series_uid: series.uid.clone(),
                segs: phase_out,
            });
        }
    }
    if p.cancelled() {
        return Err(cancelled());
    }
    p.set_phase(0.95, 0.05);
    p.set("Assembling the report…");

    // Tracks in phase order.
    let mut tracks = Vec::new();
    let mut reference_tracks = Vec::new();
    for (mi, model) in req.models.iter().enumerate() {
        for (si, subject) in subjects.iter().enumerate() {
            let track = Track {
                target: subject.name.clone(),
                model: *model,
                samples: samples[mi][si]
                    .iter()
                    .map(|s| s.clone().expect("every phase was filled"))
                    .collect(),
                reference: req.reference,
            };
            if si < n_targets {
                tracks.push(track);
            } else {
                reference_tracks.push(track);
            }
        }
    }

    // Correlation of every target against the reference structure.
    let mut correlations = Vec::new();
    for t in &tracks {
        let Some(rt) = reference_tracks.iter().find(|r| r.model == t.model) else {
            continue;
        };
        let td = t.displacements();
        let rd = rt.displacements();
        let comp = |v: &[crate::geometry::Vec3], a: usize| -> Vec<f64> {
            v.iter().map(|p| [p.x, p.y, p.z][a]).collect()
        };
        let mut axes = Vec::new();
        for (a, name) in motion::AXES.iter().enumerate() {
            if let Some((r, pv)) = motion::pearson(&comp(&td, a), &comp(&rd, a)) {
                axes.push(AxisCorrelation {
                    axis: name,
                    r,
                    p: pv,
                });
            }
        }
        if !axes.is_empty() {
            correlations.push((t.target.clone(), t.model, axes));
        }
    }

    // The ITVs.
    let mut itvs = Vec::new();
    let mut itv_segs: Vec<(String, [u8; 3], Vec<u8>)> = Vec::new();
    if req.build_itv {
        for (mi, model) in req.models.iter().enumerate() {
            for (si, target) in req.targets.iter().enumerate() {
                let mut mask = std::mem::take(&mut unions[mi][si]);
                if req.itv_margin_mm > 0.0 {
                    mask = morphology::dilate_mm(
                        &mask,
                        ref_grid.dims,
                        ref_grid.spacing,
                        req.itv_margin_mm,
                    );
                }
                let name = if req.itv_margin_mm > 0.0 {
                    format!(
                        "ITV {} +{:.0}mm ({})",
                        target.name,
                        req.itv_margin_mm,
                        model.label()
                    )
                } else {
                    format!("ITV {} ({})", target.name, model.label())
                };
                itvs.push(ItvResult {
                    target: target.name.clone(),
                    model: *model,
                    margin_mm: req.itv_margin_mm,
                    volume_cm3: motion::volume_cm3(&mask, &ref_grid),
                    seg_name: name.clone(),
                });
                itv_segs.push((name, target.color, mask));
            }
        }
    }
    if subjects.is_empty() {
        bail!("nothing to analyse");
    }

    let report = MotionReport {
        run_name: req.run_name,
        slot_name: req.slot_name,
        patient: req.patient,
        phases: req.phases.iter().map(|(l, _)| l.clone()).collect(),
        reference: req.phases[req.reference].0.clone(),
        tracks,
        reference_tracks,
        reference_structure: req.ref_struct.as_ref().map(|s| s.name.clone()),
        correlations,
        qa,
        itvs,
    };
    Ok(MotionOutcome {
        report,
        itv_series: (!itv_segs.is_empty()).then(|| OutSeries {
            label: format!("4D ITV — {}", req.group_name),
            grid: ref_grid,
            referenced_series_uid: req.phases[req.reference].1.uid.clone(),
            segs: itv_segs,
        }),
        phase_series,
        study_uid: req.study_uid,
    })
}
