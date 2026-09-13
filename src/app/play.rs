//! Playback: running through the slices of a view, and through the phases
//! of a 4D group.
//!
//! Two different things wear the same button. *Play 3D* steps the slice of
//! one viewport, which is free: the volume is already in memory and only a
//! texture is rebuilt. *Play 4D* steps the whole dataset from one phase of
//! a 4D group to the next, which is not free at all: a phase switch means a
//! different image series, and reading one from disk takes long enough that
//! playing straight off the disk would be a slideshow rather than a cine.
//!
//! So the phases are read once into [`PhaseCache`] and played from memory.
//! That costs real RAM - ten phases of a 512 x 512 x 150 CT are about
//! 800 MB - so the cache is filled by a background job with a progress bar,
//! it is refused when the estimate exceeds the budget the *Playback* module
//! carries, and it is dropped the moment the study it belongs to changes.
//!
//! Stepping a phase deliberately does *not* go through
//! [`ViewerApp::apply_new_volume`]: that resets the crosshair, the zoom, the
//! pan and every view's slice and throws away the registration, which is
//! right for "the user picked another series" and quite wrong for "the next
//! frame of a cine". [`ViewerApp::install_phase_volume`] keeps all of it and
//! changes only what belongs to the phase.

use std::sync::Arc;

use anyhow::{bail, Result};

use super::*;
use crate::fourd::Role;

/// What a run steps through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PlayTarget {
    /// The slices of one view of one dataset (*Play 3D*).
    Slices { slot: usize, view: usize },
    /// The phases of the 4D group a dataset is showing (*Play 4D*).
    Phases { slot: usize },
}

impl PlayTarget {
    pub(super) fn slot(self) -> usize {
        match self {
            PlayTarget::Slices { slot, .. } | PlayTarget::Phases { slot } => slot,
        }
    }
}

/// What a run does when it reaches the end of the range.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum PlayMode {
    /// Start again from the other end.
    #[default]
    Loop,
    /// Turn around and come back: how a breathing cycle is usually watched,
    /// because the jump from the last phase to the first is a jump the
    /// patient never made.
    Bounce,
    /// Stop at the end.
    Once,
}

impl PlayMode {
    pub(super) const ALL: [PlayMode; 3] = [PlayMode::Loop, PlayMode::Bounce, PlayMode::Once];

    pub(super) fn label(self) -> &'static str {
        match self {
            PlayMode::Loop => "Loop",
            PlayMode::Bounce => "Bounce",
            PlayMode::Once => "Once",
        }
    }

    pub(super) fn hint(self) -> &'static str {
        match self {
            PlayMode::Loop => "At the end, start again from the beginning",
            PlayMode::Bounce => {
                "At the end, turn around and run back. For a breathing cycle this is the \
                 honest one: the jump from the last phase to the first is a jump the \
                 patient never made."
            }
            PlayMode::Once => "Stop at the end",
        }
    }
}

/// The run in flight. Only one exists at a time: two animations at once
/// would fight over the same repaint clock and neither would keep its rate.
#[derive(Clone, Copy, Debug)]
pub(super) struct Running {
    pub(super) target: PlayTarget,
    /// Which way the next frame goes; *Bounce* flips it at the ends.
    pub(super) dir: i32,
    /// egui's own clock (`input.time`, seconds) at the last frame shown.
    pub(super) last: f64,
}

/// Every phase of one 4D group, read into memory so that playing them is a
/// texture upload rather than a disk read.
pub(super) struct PhaseCache {
    /// The group as the data tree names it.
    pub(super) group: String,
    /// Series UID of each phase, in temporal order. The cache is matched
    /// against the study by these rather than by index, because the tree
    /// can be edited under it.
    pub(super) uids: Vec<String>,
    /// `0%`, `50%` and so on, for the scrubber and the dose match.
    pub(super) labels: Vec<String>,
    pub(super) vols: Vec<Arc<Volume>>,
    pub(super) windows: Vec<(f32, f32)>,
}

impl PhaseCache {
    pub(super) fn len(&self) -> usize {
        self.vols.len()
    }

    /// What the cache is holding, in bytes.
    pub(super) fn bytes(&self) -> usize {
        self.vols
            .iter()
            .map(|v| v.data.len() * std::mem::size_of::<i16>())
            .sum()
    }
}

/// Settings and state of playback. Everything here is what the *Playback*
/// module edits; the buttons on the viewports only start and stop.
pub(super) struct PlayState {
    /// The dataset the module's own controls act on.
    pub(super) slot: usize,
    /// The view the module's slice transport runs, as an index into the
    /// three panes.
    pub(super) view: usize,
    pub(super) running: Option<Running>,
    /// Frames per second, one rate for slices and one for phases: a stack
    /// of 200 slices wants to move faster than a 10-phase breathing cycle.
    pub(super) slice_fps: f32,
    pub(super) phase_fps: f32,
    pub(super) mode: PlayMode,
    /// Show every `slice_step`-th slice: a way through a long stack that
    /// does not take a minute.
    pub(super) slice_step: usize,
    /// What a phase change carries with it.
    pub(super) follow_structs: bool,
    pub(super) follow_segs: bool,
    pub(super) follow_dose: bool,
    /// Refuse to read a group whose phases would need more than this.
    pub(super) budget_mb: usize,
    pub(super) cache: [Option<PhaseCache>; 2],
    pub(super) job: Option<Job<(usize, Result<PhaseCache>)>>,
    /// Start this run as soon as the phases are in memory: what *Play 4D*
    /// asked for while the cache was still being read.
    pub(super) start_after_load: Option<PlayTarget>,
}

impl Default for PlayState {
    fn default() -> Self {
        PlayState {
            slot: 0,
            view: 0,
            running: None,
            // Ten slices a second reads as a scroll rather than a flicker;
            // four phases a second is about one breathing cycle every two
            // seconds for a ten-phase 4DCT.
            slice_fps: 10.0,
            phase_fps: 4.0,
            mode: PlayMode::default(),
            slice_step: 1,
            follow_structs: true,
            follow_segs: true,
            follow_dose: true,
            budget_mb: 4096,
            cache: [None, None],
            job: None,
            start_after_load: None,
        }
    }
}

/// The frame after `cur` in a run of `n` frames, and the direction the run
/// carries on with. `None` ends the run.
///
/// Pulled out of the tick because this is the whole behaviour of the three
/// modes, and it is worth being able to test it without a window.
pub(super) fn advance(
    cur: usize,
    n: usize,
    step: usize,
    dir: i32,
    mode: PlayMode,
) -> Option<(usize, i32)> {
    if n == 0 {
        return None;
    }
    if n == 1 {
        // Nowhere to go. Looping on a single frame would spin the repaint
        // clock for nothing.
        return match mode {
            PlayMode::Once => None,
            _ => Some((0, dir)),
        };
    }
    let step = step.max(1) as i64;
    let dir = if dir >= 0 { 1 } else { -1 };
    let next = cur as i64 + dir as i64 * step;
    if (0..n as i64).contains(&next) {
        return Some((next as usize, dir));
    }
    match mode {
        PlayMode::Loop => Some((next.rem_euclid(n as i64) as usize, dir)),
        PlayMode::Bounce => {
            let back = (cur as i64 - dir as i64 * step).clamp(0, n as i64 - 1);
            Some((back as usize, -dir))
        }
        PlayMode::Once => None,
    }
}

impl ViewerApp {
    // -- what can play -----------------------------------------------------

    /// The 4D group the dataset is showing, as (name, phase series indices,
    /// phase labels) in temporal order. `None` unless the displayed series
    /// belongs to a group that still resolves to at least two phases -
    /// which is exactly when a *Play 4D* button should exist.
    pub(super) fn fourd_phases(&self, slot: usize) -> Option<(String, Vec<usize>, Vec<String>)> {
        let study = self.slots[slot].study.as_ref()?;
        if !study.has_volume() {
            return None;
        }
        let active = study.active_series;
        for g in &study.fourd_groups {
            if g.dissolved {
                continue;
            }
            let resolved = g.resolve(&study.series);
            if !resolved.contains(&Some(active)) {
                continue;
            }
            let mut idxs = Vec::new();
            let mut labels = Vec::new();
            for (m, member) in g.members.iter().enumerate() {
                if member.role != Role::Phase {
                    continue;
                }
                if let Some(i) = resolved[m] {
                    idxs.push(i);
                    labels.push(member.label.clone());
                }
            }
            if idxs.len() >= 2 {
                return Some((g.name.clone(), idxs, labels));
            }
        }
        None
    }

    /// Which phase of its group the dataset is showing, when it is showing
    /// one at all (the AVG or MIP member of a group is not a phase).
    pub(super) fn current_phase(&self, slot: usize) -> Option<(usize, usize)> {
        let (_, idxs, _) = self.fourd_phases(slot)?;
        let study = self.slots[slot].study.as_ref()?;
        let at = idxs.iter().position(|i| *i == study.active_series)?;
        Some((at, idxs.len()))
    }

    /// Is this run the one in flight?
    pub(super) fn is_playing(&self, target: PlayTarget) -> bool {
        self.play.running.is_some_and(|r| r.target == target)
    }

    /// The cache holds exactly the phases the dataset's group has now.
    pub(super) fn phase_cache_ready(&self, slot: usize) -> bool {
        let Some(cache) = self.play.cache[slot].as_ref() else {
            return false;
        };
        let Some((_, idxs, _)) = self.fourd_phases(slot) else {
            return false;
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return false;
        };
        idxs.len() == cache.len()
            && idxs
                .iter()
                .zip(&cache.uids)
                .all(|(i, uid)| study.series.get(*i).is_some_and(|s| &s.uid == uid))
    }

    /// What reading every phase of the dataset's group would cost, in bytes,
    /// estimated from the phase on display: the phases of one acquisition
    /// share a matrix, so one of them sizes them all.
    pub(super) fn phase_cache_estimate(&self, slot: usize) -> Option<usize> {
        let (_, idxs, _) = self.fourd_phases(slot)?;
        let study = self.slots[slot].study.as_ref()?;
        Some(study.volume.data.len() * std::mem::size_of::<i16>() * idxs.len())
    }

    // -- starting and stopping --------------------------------------------

    /// Start `target`, or stop it when it is the run already in flight.
    ///
    /// *Play 4D* with no phases in memory starts reading them and plays as
    /// soon as they land, so the button means the same thing either way.
    pub(super) fn toggle_play(&mut self, target: PlayTarget, now: f64) {
        if self.is_playing(target) {
            self.stop_play();
            return;
        }
        if let PlayTarget::Phases { slot } = target {
            if !self.phase_cache_ready(slot) {
                self.play.start_after_load = Some(target);
                self.start_phase_cache(slot);
                return;
            }
        }
        self.play.running = Some(Running {
            target,
            dir: 1,
            last: now,
        });
    }

    pub(super) fn stop_play(&mut self) {
        self.play.running = None;
        self.play.start_after_load = None;
    }

    /// Stop a run whose subject is gone: a closed dataset, a group that was
    /// dissolved, a view that no longer has slices.
    fn stop_if_stale(&mut self) {
        let Some(r) = self.play.running else {
            return;
        };
        let alive = match r.target {
            PlayTarget::Slices { slot, view } => {
                self.slots[slot].has_volume() && view < 3 && self.view_slice_count(slot, view) > 1
            }
            PlayTarget::Phases { slot } => self.phase_cache_ready(slot),
        };
        if !alive {
            self.stop_play();
        }
    }

    /// How many slices the view runs through.
    pub(super) fn view_slice_count(&self, slot: usize, view: usize) -> usize {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return 0;
        };
        let Some(v) = self.slots[slot].views.get(view) else {
            return 0;
        };
        study.volume.plane_slice_count(v.plane)
    }

    // -- the clock ---------------------------------------------------------

    /// One frame of the run in flight, if it is due.
    ///
    /// The app is otherwise reactive - it repaints on input and on a job
    /// landing - so playing means asking for the next repaint ourselves,
    /// the way [`super::poll_job`] does while a worker is running.
    pub(super) fn play_tick(&mut self, ctx: &egui::Context) {
        self.stop_if_stale();
        let Some(run) = self.play.running else {
            return;
        };
        let fps = match run.target {
            PlayTarget::Slices { .. } => self.play.slice_fps,
            PlayTarget::Phases { .. } => self.play.phase_fps,
        };
        let period = 1.0 / f64::from(fps.clamp(0.25, 60.0));
        let now = ctx.input(|i| i.time);
        let due = now - run.last;
        if due < period {
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(period - due));
            return;
        }
        let step = self.play.slice_step;
        let mode = self.play.mode;
        let next = match run.target {
            PlayTarget::Slices { slot, view } => {
                let n = self.view_slice_count(slot, view);
                let cur = self.slots[slot].views[view].slice.min(n.saturating_sub(1));
                match advance(cur, n, step, run.dir, mode) {
                    Some((s, dir)) => {
                        self.slots[slot].views[view].slice = s;
                        Some(dir)
                    }
                    None => None,
                }
            }
            PlayTarget::Phases { slot } => {
                let n = self.play.cache[slot].as_ref().map_or(0, PhaseCache::len);
                let cur = self.current_phase(slot).map_or(0, |(at, _)| at);
                // One phase at a time: a 4D group is short, and skipping
                // phases of a breathing cycle hides the motion it is there
                // to show.
                match advance(cur, n, 1, run.dir, mode) {
                    Some((p, dir)) => {
                        self.show_phase(slot, p);
                        Some(dir)
                    }
                    None => None,
                }
            }
        };
        match next {
            Some(dir) => {
                // The frame is timed from when it was due rather than from
                // now, so a slow frame does not stretch the whole run.
                let last = if due < 2.0 * period {
                    run.last + period
                } else {
                    now
                };
                self.play.running = Some(Running { dir, last, ..run });
                ctx.request_repaint_after(std::time::Duration::from_secs_f64(period));
            }
            None => self.stop_play(),
        }
    }

    // -- showing one phase -------------------------------------------------

    /// Show phase `i` of the cached group on `slot`.
    ///
    /// Everything the user set up stays: the crosshair, the zoom, the pan,
    /// the slice of every view and the active registration. Only what
    /// belongs to the phase changes - the image, and the structure set,
    /// segmentation series and dose drawn on it.
    pub(super) fn show_phase(&mut self, slot: usize, i: usize) -> bool {
        let Some((_, idxs, _)) = self.fourd_phases(slot) else {
            return false;
        };
        let Some(cache) = self.play.cache[slot].as_ref() else {
            return false;
        };
        let (Some(vol), Some(window), Some(idx)) = (
            cache.vols.get(i).cloned(),
            cache.windows.get(i).copied(),
            idxs.get(i).copied(),
        ) else {
            return false;
        };
        let label = cache.labels.get(i).cloned().unwrap_or_default();
        self.install_phase_volume(slot, vol, window, idx, &label);
        true
    }

    /// Put a phase's volume on a dataset without disturbing the view.
    ///
    /// The counterpart of [`ViewerApp::apply_new_volume`], which is what
    /// picking another series in the tree does: that one starts the dataset
    /// again from the middle slice with no registration, because the user
    /// asked for a different image. Stepping a phase is the same patient one
    /// moment later, so the view has to hold still or the motion the cine is
    /// there to show is lost in the camera moving with it.
    pub(super) fn install_phase_volume(
        &mut self,
        slot: usize,
        vol: Arc<Volume>,
        window: (f32, f32),
        idx: usize,
        phase_label: &str,
    ) {
        let (follow_structs, follow_segs, follow_dose) = (
            self.play.follow_structs,
            self.play.follow_segs,
            self.play.follow_dose,
        );
        let other_loaded = self.slots[1 - slot].study.is_some();
        let s = &mut self.slots[slot];
        let Some(study) = &mut s.study else {
            return;
        };
        let dims_before = study.volume.dims;
        study.volume = vol;
        study.active_series = idx;
        let dims = study.volume.dims;
        if let Some(uid) = study.series.get(idx).map(|se| se.uid.clone()) {
            if follow_structs {
                if let Some(i) = study
                    .structure_sets
                    .iter()
                    .position(|ss| ss.referenced_series_uid == uid)
                {
                    if i != s.active_structs {
                        s.active_structs = i;
                        s.roi_visible = vec![true; study.structure_sets[i].rois.len()];
                    }
                }
            }
            if follow_segs {
                if let Some(i) = study
                    .seg_series
                    .iter()
                    .position(|sr| sr.referenced_series_uid == uid)
                {
                    s.active_seg_series = i;
                }
            }
        }
        // A dose carries no reference to an image series, so the only thing
        // that can tie one to a phase is what it is called. A group whose
        // doses are not named after its phases simply keeps the dose it has.
        if follow_dose && !phase_label.is_empty() {
            if let Some(i) = study
                .doses
                .iter()
                .position(|d| dose_is_for_phase(&d.label, phase_label))
            {
                s.active_dose = i;
            }
        }
        // The phases of one acquisition share a matrix, so this normally
        // changes nothing; a group put together by hand out of unequal
        // series still must not leave the crosshair outside the volume.
        if dims != dims_before {
            for (c, d) in s.cursor.iter_mut().zip(dims) {
                *c = c.min(d.saturating_sub(1) as f64);
            }
        }
        for v in &mut s.views {
            let n = study.volume.plane_slice_count(v.plane);
            v.slice = v.slice.min(n.saturating_sub(1));
            v.invalidate();
        }
        // The window follows the phase only when nothing else could have
        // set it, the same rule a series switch uses.
        if !other_loaded {
            self.window_center = window.0;
            self.window_width = window.1;
        }
        self.rebind_seg_series(slot);
    }

    // -- reading the phases into memory ------------------------------------

    /// Read every phase of the dataset's group in the background.
    pub(super) fn start_phase_cache(&mut self, slot: usize) {
        if self.play.job.is_some() {
            return;
        }
        let Some((group, idxs, labels)) = self.fourd_phases(slot) else {
            self.error = Some("This dataset is not showing a 4D group.".into());
            self.play.start_after_load = None;
            return;
        };
        let budget = self.play.budget_mb.saturating_mul(1024 * 1024);
        if let Some(est) = self.phase_cache_estimate(slot) {
            if est > budget {
                self.error = Some(format!(
                    "Reading the {} phases of {group} would take about {}, over the {} the \
                     Playback module allows. Raise the budget there, or play the slices \
                     instead.",
                    idxs.len(),
                    human_bytes(est),
                    human_bytes(budget)
                ));
                self.play.start_after_load = None;
                return;
            }
        }
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let series: Vec<crate::loader::SeriesInfo> = idxs
            .iter()
            .filter_map(|i| study.series.get(*i).cloned())
            .collect();
        if series.len() != idxs.len() {
            return;
        }
        let progress = Arc::new(Progress::default());
        self.play.job = Some(Job::spawn(progress, move |p| {
            (slot, read_phases(&group, &series, &labels, p))
        }));
    }

    /// Poll the reader and, when it lands, start the run that asked for it.
    pub(super) fn poll_phase_cache(&mut self, ctx: &egui::Context) {
        let Some((slot, res)) = poll_tool_job(
            &mut self.play.job,
            ctx,
            "Reading the phases",
            &mut self.error,
        ) else {
            return;
        };
        self.play.cache[slot] = Some(res);
        if let Some(target) = self.play.start_after_load.take() {
            if target.slot() == slot {
                let now = ctx.input(|i| i.time);
                self.play.running = Some(Running {
                    target,
                    dir: 1,
                    last: now,
                });
            }
        }
    }

    /// Forget the phases of one dataset: its study changed, or the user
    /// asked for the memory back.
    pub(super) fn drop_phase_cache(&mut self, slot: usize) {
        self.play.cache[slot] = None;
        if self
            .play
            .running
            .is_some_and(|r| matches!(r.target, PlayTarget::Phases { slot: s } if s == slot))
        {
            self.stop_play();
        }
    }
}

/// Does this dose belong to that phase, going by what it is called?
///
/// The match is deliberately narrow: a label has to carry the phase's own
/// token (`50%`, `T3`) as a whole word, so that `50%` does not also claim
/// the dose called `150%` and a group with no per-phase doses matches
/// nothing at all rather than the wrong thing.
fn dose_is_for_phase(dose_label: &str, phase_label: &str) -> bool {
    if phase_label.is_empty() {
        return false;
    }
    let phase = phase_label.trim().to_ascii_lowercase();
    let hay = dose_label.to_ascii_lowercase();
    let mut from = 0;
    while let Some(at) = hay[from..].find(&phase) {
        let start = from + at;
        let end = start + phase.len();
        let before_ok = start == 0
            || !hay[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '.');
        let after_ok = end == hay.len()
            || !hay[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// `1.2 GB`, `830 MB`: what the module and the refusal message say.
pub(super) fn human_bytes(n: usize) -> String {
    const MB: f64 = 1024.0 * 1024.0;
    let mb = n as f64 / MB;
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{mb:.0} MB")
    }
}

/// Read every phase, in order, reporting where it is.
fn read_phases(
    group: &str,
    series: &[crate::loader::SeriesInfo],
    labels: &[String],
    p: &Progress,
) -> Result<PhaseCache> {
    let mut cache = PhaseCache {
        group: group.to_string(),
        uids: Vec::with_capacity(series.len()),
        labels: labels.to_vec(),
        vols: Vec::with_capacity(series.len()),
        windows: Vec::with_capacity(series.len()),
    };
    for (i, se) in series.iter().enumerate() {
        if p.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        p.set_phase(i as f32 / series.len() as f32, 1.0 / series.len() as f32);
        p.set(format!(
            "Reading phase {} of {} ({group})",
            i + 1,
            series.len()
        ));
        let (vol, window, _) = crate::loader::load_series_volume(se, p)?;
        cache.uids.push(se.uid.clone());
        cache.vols.push(Arc::new(vol));
        cache.windows.push(window);
    }
    Ok(cache)
}

// ---------------------------------------------------------------------------
// The Playback module
// ---------------------------------------------------------------------------

impl ViewerApp {
    pub(super) fn playback_section(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("Playback");
        let state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        let header = state.show_header(ui, |ui| {
            ui.label(egui::RichText::new("Playback").strong());
            // The one run in flight is stoppable from the header, so a cine
            // started on a viewport can be stopped without hunting for it.
            if self.play.running.is_some() && small_tip_button(ui, "⏸", "Stop what is playing") {
                self.stop_play();
            }
        });
        header.body(|ui| self.playback_body(ui));
        ui.separator();
    }

    fn playback_body(&mut self, ui: &mut egui::Ui) {
        if !self.any_volume() {
            ui.weak("Load a dataset with an image volume");
            return;
        }
        if !self.slots[self.play.slot].has_volume() {
            self.play.slot = self.first_volume_slot();
        }
        if let Some(s) = seg_engines::dataset_row(ui, self.play.slot, self.volume_slots(), true) {
            self.play.slot = s;
        }
        let slot = self.play.slot;
        let now = ui.input(|i| i.time);

        self.slice_transport(ui, slot, now);
        ui.add_space(2.0);
        self.phase_transport(ui, slot, now);

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Speed");
            ui.add(
                egui::DragValue::new(&mut self.play.slice_fps)
                    .speed(0.5)
                    .range(0.25..=60.0)
                    .suffix(" slices/s"),
            )
            .on_hover_text("How fast ▶3 runs through the slices of a view");
        });
        ui.horizontal(|ui| {
            ui.add_space(38.0);
            ui.add(
                egui::DragValue::new(&mut self.play.phase_fps)
                    .speed(0.25)
                    .range(0.25..=30.0)
                    .suffix(" phases/s"),
            )
            .on_hover_text(
                "How fast ▶4 runs through the phases of a 4D group. A ten-phase \
                 breathing cycle at 5 phases/s is one breath every two seconds.",
            );
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("At the end");
            for m in PlayMode::ALL {
                ui.selectable_value(&mut self.play.mode, m, m.label())
                    .on_hover_text(m.hint());
            }
        });
        ui.horizontal(|ui| {
            ui.label("Slice step");
            ui.add(
                egui::DragValue::new(&mut self.play.slice_step)
                    .speed(1)
                    .range(1..=20),
            )
            .on_hover_text(
                "Show every n-th slice, so that a long stack can be watched in one pass. \
                 Phases always step one at a time: skipping phases of a breathing cycle \
                 hides the motion the cine is there to show.",
            );
        });

        ui.separator();
        ui.label("A phase change carries with it:");
        ui.checkbox(&mut self.play.follow_structs, "Structures")
            .on_hover_text(
                "Switch to the structure set drawn on the new phase, when the study has \
                 one per phase (matched by the series the set references)",
            );
        ui.checkbox(&mut self.play.follow_segs, "Segmentations")
            .on_hover_text("Switch to the segmentation series that belongs to the new phase");
        ui.checkbox(&mut self.play.follow_dose, "Dose")
            .on_hover_text(
                "Switch to the dose named after the new phase. A dose carries no reference \
                 to an image series, so the only thing that can tie one to a phase is what \
                 it is called; a study whose doses are not named after its phases keeps the \
                 dose it has.",
            );

        ui.separator();
        self.phase_memory_row(ui, slot);
        ui.weak(
            "The 3D window has its own ▶4, and a Prepare that meshes every phase so the \
             surfaces keep up.",
        );
    }

    /// The slice transport: which view, and the frame controls for it.
    fn slice_transport(&mut self, ui: &mut egui::Ui, slot: usize, now: f64) {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(egui::RichText::new("Slices").strong());
            for (i, name) in ["Axial", "Sagittal", "Coronal"].iter().enumerate() {
                ui.selectable_value(&mut self.play.view, i, *name);
            }
        });
        let view = self.play.view;
        let n = self.view_slice_count(slot, view);
        let target = PlayTarget::Slices { slot, view };
        let mut step: i32 = 0;
        let mut toggle = false;
        let playing = self.is_playing(target);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if tip_widget(
                ui,
                n > 1,
                egui::Button::new("⏮").small(),
                "The slice before this one",
            ) {
                step = -1;
            }
            if tip_widget(
                ui,
                n > 1,
                egui::Button::new(if playing { "⏸" } else { "▶" }).small(),
                "Play through the slices of this view",
            ) {
                toggle = true;
            }
            if tip_widget(
                ui,
                n > 1,
                egui::Button::new("⏭").small(),
                "The slice after this one",
            ) {
                step = 1;
            }
            let cur = self.slots[slot].views.get(view).map_or(0, |v| v.slice);
            if n > 1 {
                let mut at = cur.min(n - 1);
                let resp = ui.add(
                    egui::Slider::new(&mut at, 0..=n - 1)
                        .show_value(false)
                        .custom_formatter(|v, _| format!("{}", v as usize + 1)),
                );
                if resp.changed() {
                    self.slots[slot].views[view].slice = at;
                }
                ui.weak(format!("{} / {n}", at + 1));
            } else {
                ui.weak("one slice");
            }
        });
        if toggle {
            self.toggle_play(target, now);
        }
        if step != 0 && n > 0 {
            let cur = self.slots[slot].views[view].slice;
            let next = (cur as i64 + step as i64).clamp(0, n as i64 - 1) as usize;
            self.slots[slot].views[view].slice = next;
        }
    }

    /// The phase transport, and what it says when the dataset has no group.
    fn phase_transport(&mut self, ui: &mut egui::Ui, slot: usize, now: f64) {
        let Some((group, idxs, labels)) = self.fourd_phases(slot) else {
            ui.horizontal_wrapped(|ui| {
                ui.label(egui::RichText::new("Phases").strong());
                ui.weak("this dataset is not showing a 4D group");
            });
            return;
        };
        let n = idxs.len();
        let at = self.current_phase(slot).map(|(a, _)| a);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.label(egui::RichText::new("Phases").strong());
            ui.weak(group);
        });
        let target = PlayTarget::Phases { slot };
        let playing = self.is_playing(target);
        let ready = self.phase_cache_ready(slot);
        let busy = self.play.job.is_some();
        let mut goto: Option<usize> = None;
        let mut toggle = false;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if tip_widget(
                ui,
                at.is_some_and(|a| a > 0) || (at.is_none() && n > 0),
                egui::Button::new("⏮").small(),
                "The phase before this one",
            ) {
                goto = Some(at.map_or(0, |a| a.saturating_sub(1)));
            }
            if tip_widget(
                ui,
                !busy,
                egui::Button::new(if playing { "⏸" } else { "▶" }).small(),
                "Play through the phases. The first press reads them into memory.",
            ) {
                toggle = true;
            }
            if tip_widget(
                ui,
                at.is_some_and(|a| a + 1 < n) || at.is_none(),
                egui::Button::new("⏭").small(),
                "The phase after this one",
            ) {
                goto = Some(at.map_or(0, |a| (a + 1).min(n - 1)));
            }
            match at {
                Some(a) => {
                    let mut sel = a;
                    let resp = ui.add(
                        egui::Slider::new(&mut sel, 0..=n - 1)
                            .show_value(false)
                            .custom_formatter(|v, _| format!("{}", v as usize + 1)),
                    );
                    if resp.changed() {
                        goto = Some(sel);
                    }
                    ui.weak(format!(
                        "{} ({} / {n})",
                        labels.get(a).cloned().unwrap_or_default(),
                        a + 1
                    ));
                }
                // The average or the MIP of the group is a member of it but
                // not a phase of it, so there is no position to scrub from.
                None => {
                    ui.weak("showing a member that is not a phase");
                }
            }
        });
        if !ready && !busy {
            ui.weak("Stepping reads one phase from disk; ▶ reads them all once, then plays.");
        }
        if toggle {
            self.toggle_play(target, now);
        }
        if let Some(p) = goto {
            self.goto_phase(slot, p);
        }
    }

    /// Step to one phase by hand: out of the cache when it is there, off the
    /// disk when it is not.
    pub(super) fn goto_phase(&mut self, slot: usize, phase: usize) {
        if self.phase_cache_ready(slot) && self.show_phase(slot, phase) {
            return;
        }
        let Some((_, idxs, _)) = self.fourd_phases(slot) else {
            return;
        };
        let Some(idx) = idxs.get(phase).copied() else {
            return;
        };
        self.start_phase_switch(slot, idx);
    }

    /// The phases-in-memory row: what is held, what reading them would
    /// cost, and the budget that decides whether it is allowed.
    fn phase_memory_row(&mut self, ui: &mut egui::Ui, slot: usize) {
        let held = self.play.cache[slot]
            .as_ref()
            .map(|c| (c.len(), c.bytes(), c.group.clone()));
        let ready = self.phase_cache_ready(slot);
        let est = self.phase_cache_estimate(slot);
        let busy = self.play.job.is_some();
        ui.label("Phases in memory:");
        let anything_held = held.is_some();
        match (held, ready) {
            (Some((n, bytes, _)), true) => {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.colored_label(theme::good_color(ui.visuals()), "✔");
                    ui.weak(format!("{n} phases, {}", human_bytes(bytes)));
                });
            }
            (Some((_, _, group)), false) => {
                ui.weak(format!("held for {group}; ▶ reads this group instead"));
            }
            (None, _) => match est {
                Some(e) => {
                    ui.weak(format!(
                        "none; reading them would take about {}",
                        human_bytes(e)
                    ));
                }
                None => {
                    ui.weak("none, and this dataset has no 4D group");
                }
            },
        }
        if busy {
            let (msg, frac) = self
                .play
                .job
                .as_ref()
                .map(|j| (j.progress.get(), j.progress.frac()))
                .unwrap_or_default();
            ui.add(egui::ProgressBar::new(frac).show_percentage());
            ui.weak(msg);
            if tip_button(ui, "Cancel", "Stop reading the phases") {
                if let Some(job) = &self.play.job {
                    job.progress.cancel();
                }
            }
        } else {
            ui.horizontal(|ui| {
                if enabled_tip_button(
                    ui,
                    est.is_some() && !ready,
                    "Read phases",
                    "Read every phase of this dataset's 4D group into memory, so that \
                     playing them is smooth",
                ) {
                    self.start_phase_cache(slot);
                }
                if enabled_tip_button(
                    ui,
                    anything_held,
                    "Free",
                    "Give the memory back. Playing reads the phases again.",
                ) {
                    self.drop_phase_cache(slot);
                }
            });
        }
        ui.horizontal(|ui| {
            ui.label("Budget");
            ui.add(
                egui::DragValue::new(&mut self.play.budget_mb)
                    .speed(256)
                    .range(256..=65536)
                    .suffix(" MB"),
            )
            .on_hover_text(
                "A group whose phases would need more than this is refused rather than \
                 read: ten phases of a large CT run to about a gigabyte, and the machine \
                 has other work to do.",
            );
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk a whole run and collect the frames it shows, so that a mode is
    /// judged by what the viewer actually sees rather than by one step.
    fn run(start: usize, n: usize, step: usize, mode: PlayMode, frames: usize) -> Vec<usize> {
        let mut out = vec![start];
        let (mut cur, mut dir) = (start, 1);
        for _ in 0..frames {
            match advance(cur, n, step, dir, mode) {
                Some((next, d)) => {
                    cur = next;
                    dir = d;
                    out.push(cur);
                }
                None => break,
            }
        }
        out
    }

    #[test]
    fn looping_starts_again_from_the_other_end() {
        assert_eq!(run(0, 4, 1, PlayMode::Loop, 6), [0, 1, 2, 3, 0, 1, 2]);
        // A step that overshoots wraps by the remainder, so the run keeps
        // its rhythm instead of landing on the last frame every cycle.
        assert_eq!(run(0, 5, 2, PlayMode::Loop, 4), [0, 2, 4, 1, 3]);
    }

    #[test]
    fn bouncing_turns_around_at_both_ends() {
        // The point of Bounce: a breathing cycle never jumps from the last
        // phase back to the first.
        assert_eq!(
            run(0, 4, 1, PlayMode::Bounce, 8),
            [0, 1, 2, 3, 2, 1, 0, 1, 2]
        );
    }

    #[test]
    fn once_stops_at_the_end() {
        assert_eq!(run(0, 4, 1, PlayMode::Once, 10), [0, 1, 2, 3]);
        assert_eq!(advance(3, 4, 1, 1, PlayMode::Once), None);
        // Starting in the middle still stops at the end, not after n frames.
        assert_eq!(run(2, 4, 1, PlayMode::Once, 10), [2, 3]);
    }

    #[test]
    fn a_single_frame_has_nowhere_to_go() {
        assert_eq!(advance(0, 1, 1, 1, PlayMode::Loop), Some((0, 1)));
        assert_eq!(advance(0, 1, 1, 1, PlayMode::Once), None);
        assert_eq!(advance(0, 0, 1, 1, PlayMode::Loop), None);
    }

    #[test]
    fn a_step_never_leaves_the_range() {
        // Whatever the mode and the step, the frame shown is always one
        // that exists: this is what indexes a slice and a phase.
        for mode in PlayMode::ALL {
            for n in 1..8usize {
                for step in 1..5usize {
                    for start in 0..n {
                        for dir in [1, -1] {
                            if let Some((next, d)) = advance(start, n, step, dir, mode) {
                                assert!(next < n, "{mode:?} n={n} step={step} gave {next}");
                                assert!(d == 1 || d == -1);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_zero_step_still_moves() {
        // The module clamps the setting, but nothing else should have to.
        assert_eq!(advance(0, 3, 0, 1, PlayMode::Loop), Some((1, 1)));
    }

    #[test]
    fn a_dose_belongs_to_the_phase_it_is_named_after() {
        assert!(dose_is_for_phase("Dose 50%", "50%"));
        assert!(dose_is_for_phase("50% dose", "50%"));
        assert!(dose_is_for_phase("PLAN_4DCT_T3_dose", "T3"));
        // Narrow on purpose: 50% must not claim the dose of 150%, and a
        // group whose doses are not named after its phases matches nothing
        // rather than the wrong thing.
        assert!(!dose_is_for_phase("Dose 150%", "50%"));
        assert!(!dose_is_for_phase("Plan dose", "50%"));
        assert!(!dose_is_for_phase("Dose 50%", ""));
        assert!(dose_is_for_phase("dose 0%", "0%"));
        assert!(!dose_is_for_phase("dose 10%", "0%"));
    }

    #[test]
    fn byte_counts_read_the_way_a_person_would_say_them() {
        assert_eq!(human_bytes(820 * 1024 * 1024), "820 MB");
        assert_eq!(human_bytes(2 * 1024 * 1024 * 1024), "2.0 GB");
        assert_eq!(human_bytes(0), "0 MB");
    }
}
