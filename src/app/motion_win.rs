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
    /// Ticks parallel to [`ViewerApp::combine_candidates`]: the structures
    /// that exist on some phases only (or outside the group).
    pub targets: Vec<bool>,
    /// Ticks parallel to [`ViewerApp::motion_groups`]: the structures every
    /// phase of the group carries under one name.
    pub group_targets: Vec<bool>,
    /// The reference structure (drift / correlation): a candidate, or a
    /// name every phase carries.
    pub ref_struct: Option<RefPick>,
    pub rigid: bool,
    /// The rigid model fitted on each structure's own neighbourhood, with
    /// this margin; off is one global rigid body per phase.
    pub local_rigid: bool,
    pub local_rigid_margin_mm: f64,
    pub deformable: bool,
    /// The `as contoured` model: no registration, the per-phase contours
    /// themselves (grouped targets only).
    pub contoured: bool,
    /// Which phases of the group take part (parallel to the group's phase
    /// members); the reference is always in.
    pub phases_on: Vec<bool>,
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

/// What the reference structure is picked from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RefPick {
    /// A candidate (one set's structure).
    Item(usize),
    /// A name every phase carries (index into [`ViewerApp::motion_groups`]).
    Group(usize),
}

/// A structure every phase of the group carries under one name: the
/// per-phase instances, in the group's phase order.
pub(super) struct MotionGroup {
    pub name: String,
    pub color: [u8; 3],
    /// One candidate per phase member, where the phase has one.
    pub per_phase: Vec<Option<ItemRef>>,
}

/// The dialog's transferable part: what to analyse and how, with targets
/// remembered by *name* so the same recipe applies to another dataset.
#[derive(Clone)]
pub(super) struct MotionRecipe {
    pub targets: Vec<String>,
    pub ref_struct: Option<String>,
    pub rigid: bool,
    pub local_rigid: bool,
    pub local_rigid_margin_mm: f64,
    pub contoured: bool,
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
            group_targets: Vec::new(),
            ref_struct: None,
            rigid: true,
            local_rigid: true,
            local_rigid_margin_mm: 15.0,
            deformable: true,
            contoured: true,
            phases_on: Vec::new(),
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

    /// The structures the phases of a 4D group carry under one name, each
    /// with its per-phase instances: a candidate whose set references a
    /// phase's series belongs to that phase. Names present on every phase
    /// come first, then those on some phases, both in name order.
    pub(super) fn motion_groups(&self, slot: usize, group: usize) -> Vec<MotionGroup> {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return Vec::new();
        };
        let Some(g) = study.fourd_groups.get(group) else {
            return Vec::new();
        };
        let phase_uids: Vec<&str> = g
            .phase_members()
            .iter()
            .map(|&mi| g.members[mi].series_uid.as_str())
            .collect();
        let series_of = |item: ItemRef| -> Option<&str> {
            match item.kind {
                SetKind::Structures => study
                    .structure_sets
                    .get(item.set)
                    .map(|ss| ss.referenced_series_uid.as_str()),
                SetKind::Segmentations => study
                    .seg_series
                    .get(item.set)
                    .map(|sr| sr.referenced_series_uid.as_str()),
            }
        };
        let color_of = |item: ItemRef| -> [u8; 3] {
            match item.kind {
                SetKind::Structures => study.structure_sets[item.set].rois[item.idx].color,
                SetKind::Segmentations => study.seg_series[item.set].segs[item.idx].color,
            }
        };
        let mut groups: Vec<MotionGroup> = Vec::new();
        for (item, _) in self.combine_candidates(slot) {
            let Some(name) = self.item_name(slot, item) else {
                continue;
            };
            let Some(uid) = series_of(item) else {
                continue;
            };
            let Some(pi) = phase_uids.iter().position(|u| *u == uid) else {
                continue;
            };
            let entry = match groups.iter_mut().find(|g| g.name == name) {
                Some(e) => e,
                None => {
                    groups.push(MotionGroup {
                        name: name.clone(),
                        color: color_of(item),
                        per_phase: vec![None; phase_uids.len()],
                    });
                    groups.last_mut().expect("just pushed")
                }
            };
            // The last instance on a phase wins: the most recent landing.
            entry.per_phase[pi] = Some(item);
        }
        groups.sort_by(|a, b| {
            let ca = a.per_phase.iter().all(|x| x.is_some());
            let cb = b.per_phase.iter().all(|x| x.is_some());
            cb.cmp(&ca).then_with(|| a.name.cmp(&b.name))
        });
        groups
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
        let groups = self.motion_groups(d.slot, d.group);
        let mut targets: Vec<String> = d
            .group_targets
            .iter()
            .enumerate()
            .filter(|(_, &on)| on)
            .filter_map(|(i, _)| groups.get(i).map(|g| g.name.clone()))
            .collect();
        targets.extend(
            d.targets
                .iter()
                .enumerate()
                .filter(|(_, &on)| on)
                .map(|(i, _)| name_of(i))
                .filter(|n| !n.is_empty()),
        );
        MotionRecipe {
            targets,
            ref_struct: match d.ref_struct {
                Some(RefPick::Item(i)) => Some(name_of(i)).filter(|n| !n.is_empty()),
                Some(RefPick::Group(i)) => groups.get(i).map(|g| g.name.clone()),
                None => None,
            },
            rigid: d.rigid,
            local_rigid: d.local_rigid,
            local_rigid_margin_mm: d.local_rigid_margin_mm,
            contoured: d.contoured,
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
        let groups = self.motion_groups(d.slot, d.group);
        d.targets = vec![false; cands.len()];
        d.group_targets = vec![false; groups.len()];
        d.ref_struct = None;
        // A name every phase carries is the grouped target; anything else
        // is matched among the single candidates.
        for (i, g) in groups.iter().enumerate() {
            if r.targets.contains(&g.name) {
                d.group_targets[i] = true;
            }
            if d.ref_struct.is_none() && r.ref_struct.as_deref() == Some(g.name.as_str()) {
                d.ref_struct = Some(RefPick::Group(i));
            }
        }
        for (i, (item, _)) in cands.iter().enumerate() {
            let Some(name) = self.item_name(d.slot, *item) else {
                continue;
            };
            let grouped = groups.iter().any(|g| g.name == name);
            if r.targets.contains(&name) && !grouped {
                d.targets[i] = true;
            }
            if d.ref_struct.is_none() && r.ref_struct.as_deref() == Some(name.as_str()) {
                d.ref_struct = Some(RefPick::Item(i));
            }
        }
        d.rigid = r.rigid;
        d.local_rigid = r.local_rigid;
        d.local_rigid_margin_mm = r.local_rigid_margin_mm;
        d.contoured = r.contoured;
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
        let all_phases = workflow::phases_of(group, &study.series)?;
        // `d.reference` is a member position; the phases list skips the
        // reconstructions, so find where that member landed in it.
        let members = group.phase_members();
        let reference_all = members
            .iter()
            .position(|&mi| mi == d.reference)
            .context("the reference must be one of the phases")?;
        // The phases taking part: the ticked ones, the reference always.
        let on = |i: usize| d.phases_on.get(i).copied().unwrap_or(true) || i == reference_all;
        let kept: Vec<usize> = (0..all_phases.len()).filter(|&i| on(i)).collect();
        if kept.len() < 2 {
            bail!("select at least one phase besides the reference");
        }
        let phases: Vec<(String, loader::SeriesInfo)> =
            kept.iter().map(|&i| all_phases[i].clone()).collect();
        let reference = kept
            .iter()
            .position(|&i| i == reference_all)
            .expect("the reference is kept");

        let cands = self.combine_candidates(slot);
        let groups = self.motion_groups(slot, d.group);
        let mut targets = Vec::new();
        let mut contoured = Vec::new();
        // A grouped target: its reference-phase instance is the target the
        // registration models carry; every phase's instance feeds the
        // `as contoured` model.
        for (gi, &on) in d.group_targets.iter().enumerate() {
            if !on {
                continue;
            }
            let g = groups
                .get(gi)
                .context("the target list changed - tick again")?;
            let on_ref = g.per_phase[reference_all]
                .and_then(|item| self.snapshot(slot, item))
                .with_context(|| format!("'{}' is missing on the reference phase", g.name))?;
            targets.push(on_ref);
            contoured.push(crate::workflow::motion::ContouredTarget {
                name: g.name.clone(),
                color: g.color,
                phases: kept
                    .iter()
                    .map(|&i| g.per_phase[i].and_then(|item| self.snapshot(slot, item)))
                    .collect(),
            });
        }
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
            Some(RefPick::Item(i)) => {
                let (item, label) = cands
                    .get(i)
                    .context("the reference-structure choice is stale - pick it again")?;
                Some(
                    self.snapshot(slot, *item)
                        .with_context(|| format!("'{label}' is gone"))?,
                )
            }
            Some(RefPick::Group(gi)) => {
                let g = groups
                    .get(gi)
                    .context("the reference-structure choice is stale - pick it again")?;
                let on_ref = g.per_phase[reference_all]
                    .and_then(|item| self.snapshot(slot, item))
                    .with_context(|| format!("'{}' is missing on the reference phase", g.name))?;
                contoured.push(crate::workflow::motion::ContouredTarget {
                    name: g.name.clone(),
                    color: g.color,
                    phases: kept
                        .iter()
                        .map(|&i| g.per_phase[i].and_then(|item| self.snapshot(slot, item)))
                        .collect(),
                });
                Some(on_ref)
            }
            None => None,
        };
        if let Some(r) = &ref_struct {
            if targets.iter().any(|t| t.name == r.name) {
                bail!("'{}' cannot be both target and reference", r.name);
            }
        }
        let mut models = Vec::new();
        if d.rigid {
            models.push(MotionModel::Rigid);
        }
        if d.deformable {
            models.push(MotionModel::Deformable);
        }
        if d.contoured
            && contoured
                .iter()
                .any(|c| c.phases.iter().all(|s| s.is_some()))
        {
            models.push(MotionModel::Contoured);
        }
        if models.is_empty() {
            bail!("choose at least one model (rigid / deformable / as contoured)");
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
            contoured,
            models,
            local_rigid_margin_mm: d.local_rigid.then_some(d.local_rigid_margin_mm),
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

        let groups_here = self.motion_groups(slot, d.group);
        let cand_names: Vec<String> = cands
            .iter()
            .map(|(item, _)| self.item_name(slot, *item).unwrap_or_default())
            .collect();
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

                d.targets.resize(cands.len(), false);
                d.group_targets.resize(groups_here.len(), false);
                d.phases_on.resize(phases.len(), true);
                // The lists shrink when sets are removed while the window
                // is open; a stale pick must not survive it.
                match d.ref_struct {
                    Some(RefPick::Item(i)) if i >= cands.len() => d.ref_struct = None,
                    Some(RefPick::Group(i)) if i >= groups_here.len() => d.ref_struct = None,
                    _ => {}
                }
                // Candidates that belong to a grouped name are listed once,
                // under the group; the rest are single structures.
                let grouped_names: Vec<&str> = groups_here
                    .iter()
                    .filter(|g| g.per_phase.iter().all(|x| x.is_some()))
                    .map(|g| g.name.as_str())
                    .collect();
                let single: Vec<(usize, &String)> = cands
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| {
                        !cand_names
                            .get(*i)
                            .is_some_and(|n| grouped_names.contains(&n.as_str()))
                    })
                    .map(|(i, (_, l))| (i, l))
                    .collect();
                ui.label("Targets:");
                ui.columns(2, |cols| {
                    cols[0].label(egui::RichText::new("On every phase").strong());
                    egui::ScrollArea::vertical()
                        .id_salt("motion_targets_all")
                        .max_height(120.0)
                        .show(&mut cols[0], |ui| {
                            let mut any = false;
                            for (i, g) in groups_here.iter().enumerate() {
                                if !grouped_names.contains(&g.name.as_str()) {
                                    continue;
                                }
                                any = true;
                                ui.checkbox(&mut d.group_targets[i], &g.name).on_hover_text(
                                    "Contoured on every phase of the group: the reference \
                                     phase's instance is carried by the registration models, \
                                     and the per-phase instances are read as contoured",
                                );
                            }
                            if !any {
                                ui.weak("none");
                            }
                        });
                    cols[1].label(egui::RichText::new("On some phases").strong());
                    egui::ScrollArea::vertical()
                        .id_salt("motion_targets_some")
                        .max_height(120.0)
                        .show(&mut cols[1], |ui| {
                            for (i, label) in &single {
                                ui.checkbox(&mut d.targets[*i], *label).on_hover_text(
                                    "Defined on (or resampled to) the reference phase and \
                                     carried to the others by the registration models",
                                );
                            }
                            if single.is_empty() {
                                ui.weak("none");
                            }
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Reference structure:");
                    let sel = match d.ref_struct {
                        Some(RefPick::Item(i)) => cands.get(i).map(|(_, l)| l.clone()),
                        Some(RefPick::Group(i)) => groups_here.get(i).map(|g| g.name.clone()),
                        None => None,
                    }
                    .unwrap_or_else(|| "(none)".into());
                    egui::ComboBox::from_id_salt("motion_refstruct")
                        .width(260.0)
                        .selected_text(sel)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut d.ref_struct, None, "(none)");
                            for (i, g) in groups_here.iter().enumerate() {
                                if grouped_names.contains(&g.name.as_str()) {
                                    ui.selectable_value(
                                        &mut d.ref_struct,
                                        Some(RefPick::Group(i)),
                                        format!("{} (every phase)", g.name),
                                    );
                                }
                            }
                            for (i, label) in &single {
                                ui.selectable_value(
                                    &mut d.ref_struct,
                                    Some(RefPick::Item(*i)),
                                    *label,
                                );
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
                    ui.add_enabled_ui(d.rigid, |ui| {
                        ui.checkbox(&mut d.local_rigid, "local").on_hover_text(
                            "One rigid body per structure, fitted on the structure \
                                 dilated by the margin. A whole breathing patient is not a \
                                 rigid body: the global fit finds the couch and reports no \
                                 motion.",
                        );
                        ui.add_enabled(
                            d.local_rigid,
                            egui::DragValue::new(&mut d.local_rigid_margin_mm)
                                .speed(1.0)
                                .range(3.0..=60.0)
                                .suffix(" mm"),
                        );
                    });
                    ui.checkbox(&mut d.deformable, "deformable")
                        .on_hover_text("B-spline refinement on top of the global rigid result");
                    let any_group = d.group_targets.iter().any(|&on| on);
                    ui.add_enabled(
                        any_group,
                        egui::Checkbox::new(&mut d.contoured, "as contoured"),
                    )
                    .on_hover_text(
                        "No registration: the target as it is contoured on each phase \
                             (targets from the left column only)",
                    );
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut d.build_itv, "Build ITV").on_hover_text(
                        "Union of the target over the selected phases, on the reference \
                         phase; one per target and model",
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
                    ui.separator();
                    ui.label("Phases:");
                    if ui.small_button("All").clicked() {
                        d.phases_on.iter_mut().for_each(|v| *v = true);
                    }
                    if ui.small_button("None").clicked() {
                        d.phases_on.iter_mut().for_each(|v| *v = false);
                    }
                    let n_on = d
                        .phases_on
                        .iter()
                        .enumerate()
                        .filter(|(i, &v)| {
                            v || phases.get(*i).is_some_and(|(mi, _)| *mi == d.reference)
                        })
                        .count();
                    ui.weak(format!("{n_on} of {}", phases.len()));
                });
                egui::CollapsingHeader::new("Select phases")
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for (i, (mi, label)) in phases.iter().enumerate() {
                                let is_ref = *mi == d.reference;
                                if is_ref {
                                    let mut on = true;
                                    ui.add_enabled(false, egui::Checkbox::new(&mut on, label))
                                        .on_hover_text("The reference phase is always in");
                                } else {
                                    ui.checkbox(&mut d.phases_on[i], label);
                                }
                            }
                        });
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
