//! The operations that spawn a background job.
//!
//! Every long operation follows the same shape: snapshot the inputs, spawn a
//! thread, poll the channel each frame (`poll_job`), and land the result --
//! validating it first where the underlying data could have changed
//! meanwhile. The `Job` type, `poll_job` and the per-frame poll of every
//! job (`poll_all_jobs`) are here too; see `docs/architecture.md` for the
//! pattern.

use super::*;

use std::path::Path;

// Background loading

/// A freshly reconstructed volume: pixels, its default window/level and any
/// non-fatal notes raised while reading the series.
pub(super) type LoadedVolume = (Arc<Volume>, (f32, f32), Vec<String>);

pub(super) enum LoadResult {
    /// A whole folder, for the given slot.
    Study(Box<anyhow::Result<LoadedStudy>>, usize),
    /// A single series switched into (slot, series index).
    Volume(Box<anyhow::Result<LoadedVolume>>, usize, usize),
    /// One phase of a 4D group stepped into (slot, series index). Lands
    /// through [`ViewerApp::install_phase_volume`] instead, so that the
    /// crosshair, the zoom and the registration stay where they are.
    Phase(Box<anyhow::Result<LoadedVolume>>, usize, usize),
}

/// A unit of work running on a background thread: a shared progress handle
/// plus the channel its result arrives on. Every background feature in the
/// app has this shape, and [`poll_job`] drives them all identically.
pub(super) struct Job<T> {
    pub(super) progress: Arc<Progress>,
    pub(super) rx: mpsc::Receiver<T>,
}

impl<T> Job<T> {
    /// Run `work` on a new thread and return the handle to poll for its
    /// result. The worker gets the progress handle; the caller keeps a clone.
    pub(super) fn spawn(
        progress: Arc<Progress>,
        work: impl FnOnce(&Progress) -> T + Send + 'static,
    ) -> Job<T>
    where
        T: Send + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let p = progress.clone();
        std::thread::spawn(move || {
            let _ = tx.send(work(&p));
        });
        Job { progress, rx }
    }
}

/// Ask a running job to stop when its window's Cancel button was pressed;
/// nothing happens without a job.
pub(super) fn cancel_if<T>(cancel: bool, job: &Option<Job<T>>) {
    if cancel {
        if let Some(job) = job {
            job.progress.cancel();
        }
    }
}

/// Poll a background job. Returns its result once, clearing the slot; reports
/// a worker that died without answering into `error`; otherwise schedules the
/// next poll and returns `None`.
pub(super) fn poll_job<T>(
    slot: &mut Option<Job<T>>,
    ctx: &egui::Context,
    what: &str,
    error: &mut Option<String>,
) -> Option<T> {
    let job = slot.as_ref()?;
    match job.rx.try_recv() {
        Ok(v) => {
            *slot = None;
            Some(v)
        }
        Err(mpsc::TryRecvError::Empty) => {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            None
        }
        Err(mpsc::TryRecvError::Disconnected) => {
            *slot = None;
            *error = Some(format!("{what} thread terminated unexpectedly"));
            None
        }
    }
}

/// [`poll_job`] for the jobs that answer with `(slot, Result)`: a failure is
/// reported as `"{what} failed: "`, except a cancellation, which is what
/// the user asked for and needs no dialog.
pub(super) fn poll_tool_job<T>(
    slot: &mut Option<Job<(usize, anyhow::Result<T>)>>,
    ctx: &egui::Context,
    what: &str,
    error: &mut Option<String>,
) -> Option<(usize, T)> {
    match poll_job(slot, ctx, what, error)? {
        (s, Ok(v)) => Some((s, v)),
        (_, Err(e)) => {
            if !progress::is_cancellation(&e) {
                *error = Some(format!("{what} failed: {e:#}"));
            }
            None
        }
    }
}

impl ViewerApp {
    /// Note that a workspace was loaded from here, so *Restore the last
    /// session* can put it back on the next run. Repeated loads of the same
    /// folder are one entry, and the list is written out as it changes: the
    /// program is not always closed politely.
    pub(super) fn remember_source(&mut self, slot: usize, path: &Path) {
        let slot = slot.min(MAX_WORKSPACES - 1);
        // Absolute, because the next run will not have this working
        // directory: a relative path from the command line would restore
        // nothing (or, worse, something else).
        let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if self.session[slot].contains(&path) {
            return;
        }
        self.session[slot].push(path);
        self.persist_settings();
    }

    /// [`ViewerApp::remember_source`] for many paths at once, the settings
    /// written once.
    pub(super) fn remember_sources(&mut self, slot: usize, paths: &[PathBuf]) {
        let slot = slot.min(MAX_WORKSPACES - 1);
        let mut known: std::collections::HashSet<PathBuf> =
            self.session[slot].iter().cloned().collect();
        let mut changed = false;
        for p in paths {
            let p = p.canonicalize().unwrap_or_else(|_| p.clone());
            if known.insert(p.clone()) {
                self.session[slot].push(p);
                changed = true;
            }
        }
        if changed {
            self.persist_settings();
        }
    }

    /// A workspace was emptied: it is no longer part of the session.
    pub(super) fn forget_sources(&mut self, slot: usize) {
        if self.session[slot.min(MAX_WORKSPACES - 1)].is_empty() {
            return;
        }
        self.session[slot.min(MAX_WORKSPACES - 1)].clear();
        self.persist_settings();
    }

    pub(super) fn start_load(&mut self, slot: usize, path: PathBuf) {
        if self.loading.is_some() {
            self.pending_load = Some((slot, path));
            return;
        }
        self.remember_source(slot, &path);
        if let Some(r) = &mut self.wf_run {
            r.forget_parked(slot);
        }
        // Into an open workspace it is a merge, which keeps the workspace's
        // own volume: no point decoding the folder's.
        let merge = self.slots[slot].study.is_some();
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = if merge {
                loader::load_directory_to_merge(&path, &p2)
            } else {
                loader::load_directory(&path, &p2)
            };
            let _ = tx.send(LoadResult::Study(Box::new(res), slot));
        });
        self.loading = Some(Job { progress, rx });
    }

    /// *File ▶ Add DICOM file(s)*: load an explicit selection of files.
    ///
    /// It lands through the same [`LoadResult::Study`] as a folder, so the
    /// files merge into the workspace and appear in the tree exactly as a
    /// folder's contents would. Nothing downstream knows the difference.
    pub(super) fn start_load_files(&mut self, slot: usize, paths: Vec<PathBuf>) {
        self.start_load_files_noting(slot, paths, None);
    }

    /// [`ViewerApp::start_load_files`], the session remembering `sources`
    /// (the folders a selection of the PACS covers) instead of every file.
    pub(super) fn start_load_files_noting(
        &mut self,
        slot: usize,
        paths: Vec<PathBuf>,
        sources: Option<Vec<PathBuf>>,
    ) {
        if paths.is_empty() {
            return;
        }
        if self.loading.is_some() {
            self.pending_load_files = Some((slot, paths, sources));
            return;
        }
        match &sources {
            Some(src) => self.remember_sources(slot, src),
            None => self.remember_sources(slot, &paths),
        }
        let origin = match paths.len() {
            1 => paths[0].display().to_string(),
            n => format!("{n} selected files"),
        };
        let merge = self.slots[slot].study.is_some();
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = if merge {
                loader::load_files_to_merge(&paths, &origin, &p2)
            } else {
                loader::load_files(&paths, &origin, &p2)
            };
            let _ = tx.send(LoadResult::Study(Box::new(res), slot));
        });
        self.loading = Some(Job { progress, rx });
    }

    /// Read series `idx` of `slot` off the disk and put it on display.
    ///
    /// Asked while another read is running, it waits its turn rather than
    /// being dropped: a workspace that was handed its series list ahead of
    /// the volume (a series moved in from another workspace) would
    /// otherwise be left with nothing to show.
    pub(super) fn start_series_switch(&mut self, slot: usize, idx: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let Some(series) = study.series.get(idx).cloned() else {
            return;
        };
        if self.loading.is_some() {
            self.pending_switch = Some((slot, series.uid.clone()));
            return;
        }
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = loader::load_series_volume(&series, &p2)
                .map(|(vol, window, warnings)| (Arc::new(vol), window, warnings));
            let _ = tx.send(LoadResult::Volume(Box::new(res), slot, idx));
        });
        self.loading = Some(Job { progress, rx });
    }

    /// Step to one phase of a 4D group off the disk.
    ///
    /// The same read as [`Self::start_series_switch`], landing differently:
    /// a phase is the same patient a moment later, so the view holds still
    /// (see [`Self::install_phase_volume`]).
    pub(super) fn start_phase_switch(&mut self, slot: usize, idx: usize) {
        if self.loading.is_some() {
            return;
        }
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let Some(series) = study.series.get(idx).cloned() else {
            return;
        };
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = loader::load_series_volume(&series, &p2)
                .map(|(vol, window, warnings)| (Arc::new(vol), window, warnings));
            let _ = tx.send(LoadResult::Phase(Box::new(res), slot, idx));
        });
        self.loading = Some(Job { progress, rx });
    }

    /// What the 4D group calls the phase this series is, for the dose match.
    pub(super) fn phase_label_of(&self, slot: usize, idx: usize) -> String {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return String::new();
        };
        let Some(uid) = study.series.get(idx).map(|s| s.uid.as_str()) else {
            return String::new();
        };
        study
            .fourd_groups
            .iter()
            .filter(|g| !g.dissolved)
            .find_map(|g| {
                g.members
                    .iter()
                    .find(|m| m.series_uid == uid)
                    .map(|m| m.label.clone())
            })
            .unwrap_or_default()
    }

    /// A folder finished loading (*File ▶ Add DICOM folder*): merge it into
    /// an occupied slot, or install it into an empty one. Merging leaves the
    /// displayed volume and all selections untouched - the new patients /
    /// studies / series simply appear in the data tree.
    pub(super) fn absorb_loaded_study(&mut self, slot: usize, study: LoadedStudy) {
        let Some(dest) = self.slots[slot].study.as_mut() else {
            // Loaded to be merged, but the workspace was closed meanwhile:
            // it is installed instead, and its volume is read now.
            let unread = !study.has_volume() && !study.series.is_empty();
            let series = study.active_series;
            self.on_study_loaded(slot, study);
            if unread {
                self.start_series_switch(slot, series);
            }
            return;
        };
        let was_empty = !dest.has_volume();
        let notes = loader::merge_study(dest, study);
        dest.warnings.extend(notes);
        self.settings_gen += 1;
        // A workspace that held only RT images or RT objects has just been
        // given an image series: display it. Merging normally leaves the
        // shown volume alone, but here there was none, and leaving the views
        // empty beside a tree that now lists a CT would be a puzzle rather
        // than a policy.
        if was_empty
            && self.slots[slot]
                .study
                .as_ref()
                .is_some_and(|s| !s.series.is_empty())
        {
            self.start_series_switch(slot, 0);
        }
    }

    pub(super) fn on_study_loaded(&mut self, slot: usize, study: LoadedStudy) {
        // Whatever was played from this workspace belongs to the study that
        // is being replaced.
        self.drop_phase_cache(slot);
        // The window: another workspace's showing the same modality, so a
        // comparison stays comparable, else the series' own.
        let modality = study
            .series
            .get(study.active_series)
            .map(|s| s.modality.clone())
            .unwrap_or_default();
        self.adopt_window(slot, &modality, study.default_window);
        self.series_window[slot] = study.default_window;
        if !study.doses.is_empty() && self.dose_mode == DoseMode::Off {
            self.dose_mode = DoseMode::Both;
        }
        self.slots[slot].dose_shown = true;
        let s = &mut self.slots[slot];
        // Default to the structure set drawn on the active image series
        // (matters for e.g. 4DCT patients with one RTSTRUCT per phase).
        let active_uid = study
            .series
            .get(study.active_series)
            .map(|se| se.uid.clone());
        s.active_structs = active_uid
            .as_deref()
            .and_then(|uid| {
                study
                    .structure_sets
                    .iter()
                    .position(|ss| ss.referenced_series_uid == uid)
            })
            .unwrap_or(0);
        s.roi_visible = study
            .structure_sets
            .get(s.active_structs)
            .map(|ss| vec![true; ss.rois.len()])
            .unwrap_or_default();
        s.active_dose = 0;
        s.dose_reference = study
            .plans
            .iter()
            .find_map(|p| p.target_prescription_dose)
            .map(|d| d as f32)
            .or_else(|| study.doses.first().map(|d| d.max_dose))
            .unwrap_or(1.0);
        let dims = study.volume.dims;
        s.cursor = [
            dims[0] as f64 * 0.5,
            dims[1] as f64 * 0.5,
            dims[2] as f64 * 0.5,
        ];
        for v in &mut s.views {
            v.slice = match v.plane {
                ViewPlane::Axial => dims[2] / 2,
                ViewPlane::Sagittal => dims[0] / 2,
                ViewPlane::Coronal => dims[1] / 2,
            };
            v.zoom = 0.0;
            v.pan = Vec2::ZERO;
            v.invalidate();
        }
        // Default to the segmentation series of the active image series,
        // the same rule the structure sets follow above.
        s.active_seg = 0;
        s.active_seg_series = active_uid
            .as_deref()
            .and_then(|uid| {
                study
                    .seg_series
                    .iter()
                    .position(|sr| sr.referenced_series_uid == uid)
            })
            .unwrap_or(0);
        s.study = Some(study);
        self.rebind_seg_series(slot);
        self.cancel_grow();
        self.paint_last = None;
        self.show_workspace(slot);
        // Any previous registration no longer matches the loaded volumes,
        // and open viewers for this slot reference stale data.
        // The planar panes name images by index, and those were another
        // study's.
        self.slots[slot].planar = Default::default();
        self.d3_windows.retain(|w| w.slot != slot);
        if self.maximized.map(|(s, _)| s == slot).unwrap_or(false) {
            self.maximized = None;
        }
        self.clear_registration();
        self.settings_gen += 1;
        self.reapply_preset(slot);
    }

    pub(super) fn apply_new_volume(
        &mut self,
        slot: usize,
        vol: Arc<Volume>,
        window: (f32, f32),
        idx: usize,
    ) {
        let modality = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.series.get(idx))
            .map(|s| s.modality.clone())
            .unwrap_or_default();
        if modality != self.shown_modality(slot) {
            self.adopt_window(slot, &modality, window);
        }
        self.series_window[slot] = window;
        let s = &mut self.slots[slot];
        if let Some(study) = &mut s.study {
            study.volume = vol;
            study.active_series = idx;
            // Follow the series switch with the matching structure set,
            // if one references the newly active series.
            if let Some(uid) = study.series.get(idx).map(|se| se.uid.clone()) {
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
                // Same for the segmentations drawn on the new series.
                if let Some(i) = study
                    .seg_series
                    .iter()
                    .position(|sr| sr.referenced_series_uid == uid)
                {
                    s.active_seg_series = i;
                }
            }
            let dims = study.volume.dims;
            s.cursor = [
                dims[0] as f64 * 0.5,
                dims[1] as f64 * 0.5,
                dims[2] as f64 * 0.5,
            ];
            for v in &mut s.views {
                v.slice = match v.plane {
                    ViewPlane::Axial => dims[2] / 2,
                    ViewPlane::Sagittal => dims[0] / 2,
                    ViewPlane::Coronal => dims[1] / 2,
                };
                v.zoom = 0.0;
                v.pan = Vec2::ZERO;
                v.invalidate();
            }
            s.active_seg = 0;
            self.cancel_grow();
            self.rebind_seg_series(slot);
            self.paint_last = None;
            self.clear_registration();
            self.settings_gen += 1;
        }
        // A preset picked before goes on this series too.
        self.reapply_preset(slot);
    }

    /// Generate a transformed copy of the source study into the other slot
    /// (background thread; the applied parameters are the ground truth).
    pub(super) fn start_simulation(&mut self) {
        if self.sim_job.is_some() || self.loading.is_some() {
            return;
        }
        let source = self.sim_source.min(MAX_WORKSPACES - 1);
        let target = self.sim_target.min(MAX_WORKSPACES - 1);
        if target == source {
            return;
        }
        let Some(study) = &self.slots[source].study else {
            self.error = Some(format!(
                "Load something into workspace {} first",
                SLOT_NAMES[source]
            ));
            return;
        };
        // Bump centered at the source study's crosshair.
        let c = self.slots[source].cursor;
        let p = study.volume.voxel_to_patient(c[0], c[1], c[2]);
        let mut params = self.sim_params;
        params.bump_center = [p.x, p.y, p.z];

        self.last_sim = Some(format!(
            "{} ▶ {}: {}",
            SLOT_NAMES[source],
            SLOT_NAMES[target],
            params.describe()
        ));

        let src = study.clone();
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let out = simulate::generate_transformed_study(&src, &params, &p2);
            let _ = tx.send((target, out));
        });
        self.sim_job = Some(Job { progress, rx });
    }

    /// Run the export plan into its output folder.
    ///
    /// Both workspaces are handed to the worker, because one run can write out
    /// series from either - the plan decides which.
    pub(super) fn start_export(&mut self) {
        if self.export_job.is_some() {
            return;
        }
        let Some(plan) = self.export_plan.clone() else {
            return;
        };
        let dir = PathBuf::from(self.export_dir.trim());
        if dir.as_os_str().is_empty() {
            self.error = Some("Choose an output folder for the export".into());
            return;
        }
        if plan.is_empty() {
            self.error = Some("Nothing is selected for export".into());
            return;
        }
        let studies: [Option<LoadedStudy>; MAX_WORKSPACES] =
            std::array::from_fn(|s| self.slots[s].study.clone());
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let refs: [Option<&LoadedStudy>; MAX_WORKSPACES] =
                std::array::from_fn(|s| studies[s].as_ref());
            let _ = tx.send(export::run(&plan, refs, &dir, &p2));
        });
        self.export_result = None;
        self.export_warnings.clear();
        self.export_job = Some(Job { progress, rx });
    }

    /// Write the built-in synthetic RT test study into the configured folder
    /// (background thread; the folder is created if it does not exist).
    pub(super) fn start_generate(&mut self) {
        if self.gen_job.is_some() {
            return;
        }
        let dir = PathBuf::from(self.gen_dir.trim());
        if dir.as_os_str().is_empty() {
            self.error = Some("Choose an output folder for the test data".into());
            return;
        }
        let params = self.gen_params.clone();
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = gen_test_data::generate(&dir, &params, &p2).map(|n| (n, dir));
            let _ = tx.send(res);
        });
        self.gen_result = None;
        self.gen_job = Some(Job { progress, rx });
    }

    /// Snapshot the volume and run the segmentation on a worker thread, with
    /// the parameters of the open tool window.
    pub(super) fn start_autoseg(&mut self) {
        if self.autoseg_job.is_some() {
            return;
        }
        let Some(d) = &self.autoseg_dialog else {
            return;
        };
        let Some(study) = self.slots[d.slot].study.as_ref() else {
            return;
        };
        let volume = study.volume.clone();
        let displayed = PhaseInfo::displayed(study);
        let root = self.models_root();
        let (slot, model) = (d.slot, d.model);
        let opts = zoo::RunOptions {
            device: d.device,
            parts: (!d.parts.is_empty()).then(|| d.parts.clone()),
        };
        let phases = if d.output.phases {
            match self.phase_inputs(slot) {
                Ok(p) => Some(p),
                Err(e) => {
                    self.error = Some(format!("{e:#}"));
                    return;
                }
            }
        } else {
            None
        };
        self.persist_settings();
        let progress = Arc::new(Progress::default());
        progress.set("Starting auto-segmentation");
        self.autoseg_slot = slot;
        self.autoseg_job = Some(Job::spawn(progress, move |p| {
            let run = |vol: &Volume, p: &Progress| model.run(vol, &opts, &root, p);
            let out = match phases {
                Some((group, inputs)) => run_on_phases(&group, inputs, p, run),
                None => run(&volume, p).map(|r| vec![(displayed, r)]),
            };
            (slot, out)
        }));
    }

    /// A run finished: verify the slot still shows the same volume (or,
    /// for a run over the phases, still holds the study), then open the
    /// organ-selection dialog over everything any phase found.
    pub(super) fn on_autoseg_done(
        &mut self,
        slot: usize,
        results: Vec<(PhaseInfo, autoseg::AutosegResult)>,
    ) {
        let Some((first, result)) = results.first() else {
            return;
        };
        let over_phases = first.group.is_some();
        if !over_phases
            && !self.slot_still_shows(slot, result.volume_dims, &result.frame_of_reference_uid)
        {
            self.error = Some(stale_result(&AUTOSEG));
            return;
        }
        if over_phases
            && !self.slots[slot].study.as_ref().is_some_and(|st| {
                results
                    .iter()
                    .any(|(info, _)| st.series.iter().any(|se| se.uid == info.series_uid))
            })
        {
            self.error = Some(stale_result(&AUTOSEG));
            return;
        }
        // Every class any phase found, once, with its mean volume over the
        // phases that have it; ordered by that volume like a single run's
        // list is.
        let mut organs: Vec<autoseg::OrganHit> = Vec::new();
        let mut hits: Vec<usize> = Vec::new();
        for (_, r) in &results {
            for o in &r.organs {
                match organs.iter().position(|k| k.label == o.label) {
                    Some(i) => {
                        organs[i].voxels += o.voxels;
                        organs[i].cm3 += o.cm3;
                        hits[i] += 1;
                    }
                    None => {
                        organs.push(o.clone());
                        hits.push(1);
                    }
                }
            }
        }
        for (o, n) in organs.iter_mut().zip(&hits) {
            o.voxels /= *n as u64;
            o.cm3 /= *n as f64;
        }
        organs.sort_by_key(|o| std::cmp::Reverse(o.voxels));
        if organs.is_empty() {
            self.error = Some(if over_phases {
                "Auto-segmentation found no organs on any phase.".into()
            } else {
                "Auto-segmentation found no organs in this volume.".into()
            });
            return;
        }
        let selected = vec![true; organs.len()];
        let output = self
            .autoseg_dialog
            .as_ref()
            .map(|d| d.output.clone())
            .unwrap_or_default();
        let tg263 = self.autoseg_pending_tg263;
        self.autoseg_pending = Some(AutosegPending {
            slot,
            phases: results,
            organs,
            selected,
            output,
            tg263,
        });
        // The structures are picked in the section's *Results* fold, which
        // is unfolded on the run's workspace.
        self.auto.slot = slot;
        self.auto.reveal_results = true;
        self.show_module(Module::AutoTools);
    }

    // -- Tools ▶ Anonymize DICOM folder ------------------------------------
    pub(super) fn anon_start_scan(&mut self) {
        let dir = PathBuf::from(self.anon_dir.trim());
        if dir.as_os_str().is_empty() {
            self.error = Some("Choose a folder with DICOM data first".into());
            return;
        }
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(anonymize::scan(&dir, &p2));
        });
        self.anon_scan = None;
        self.anon_result = None;
        self.anon_scan_job = Some(Job { progress, rx });
    }

    pub(super) fn anon_start_apply(&mut self) {
        let Some(scan) = &self.anon_scan else { return };
        let params = anonymize::ApplyParams {
            replacements: scan
                .findings
                .iter()
                .filter(|f| f.enabled)
                .map(|f| (f.tag, f.vr, f.replacement.trim().to_string()))
                .collect(),
            remove_private: self.anon_remove_private,
            remap_uids: self.anon_remap_uids,
            mark_deidentified: self.anon_mark,
            out_dir: if self.anon_in_place {
                None
            } else {
                let out = PathBuf::from(self.anon_out.trim());
                if out.as_os_str().is_empty() {
                    self.error =
                        Some("Choose an output folder, or tick “overwrite in place”".into());
                    return;
                }
                Some(out)
            },
        };
        let files = scan.files.clone();
        let root = scan.root.clone();
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(anonymize::apply(&files, &root, &params, &p2));
        });
        self.anon_result = None;
        self.anon_apply_job = Some(Job { progress, rx });
    }
}

impl ViewerApp {
    /// Once a frame, before anything is drawn: every background job is
    /// polled and what finished lands, in this order; then the 4D cine
    /// clock, the recorder and the snapshot take the frame egui delivered.
    pub(super) fn poll_all_jobs(&mut self, ctx: &egui::Context) {
        // Poll background loading.
        match poll_job(&mut self.loading, ctx, "Loading", &mut self.error) {
            Some(LoadResult::Study(res, slot)) => match *res {
                Ok(study) => self.absorb_loaded_study(slot, study),
                Err(e) => self.error = Some(format!("{e:#}")),
            },
            Some(LoadResult::Volume(res, slot, idx)) => match *res {
                Ok((vol, window, warnings)) => {
                    self.apply_new_volume(slot, vol, window, idx);
                    if let Some(study) = &mut self.slots[slot].study {
                        study.warnings.extend(warnings);
                    }
                }
                Err(e) => self.error = Some(format!("{e:#}")),
            },
            Some(LoadResult::Phase(res, slot, idx)) => match *res {
                Ok((vol, window, _)) => {
                    let label = self.phase_label_of(slot, idx);
                    self.install_phase_volume(slot, vol, window, idx, &label);
                }
                Err(e) => self.error = Some(format!("{e:#}")),
            },
            None => {}
        }
        // Kick a queued load once the current one finished.
        if self.loading.is_none() {
            if let Some((slot, uid)) = self.pending_switch.take() {
                let idx = self.slots[slot]
                    .study
                    .as_ref()
                    .and_then(|st| st.series.iter().position(|se| se.uid == uid));
                if let Some(idx) = idx {
                    self.start_series_switch(slot, idx);
                }
            } else if let Some((slot, path)) = self.pending_load.take() {
                self.start_load(slot, path);
            } else if let Some((slot, paths, sources)) = self.pending_load_files.take() {
                self.start_load_files_noting(slot, paths, sources);
            } else if !self.restore_queue.is_empty() {
                // One source of the session being restored at a time.
                let (slot, path) = self.restore_queue.remove(0);
                if path.is_dir() {
                    self.start_load(slot, path);
                } else {
                    self.start_load_files(slot, vec![path]);
                }
            }
        }

        self.poll_pending_phase_field();

        // Poll background simulation.
        if let Some((target, study)) =
            poll_job(&mut self.sim_job, ctx, "Simulation", &mut self.error)
        {
            self.on_study_loaded(target, study);
            self.show_workspace(target);
        }

        // Poll background export.
        match poll_job(&mut self.export_job, ctx, "Export", &mut self.error) {
            Some(Ok(sum)) => {
                self.export_result = Some(sum.message());
                self.export_warnings = sum.warnings;
            }
            Some(Err(e)) => self.error = Some(format!("Export failed: {e:#}")),
            None => {}
        }

        // Poll background test-data generation.
        match poll_job(
            &mut self.gen_job,
            ctx,
            "Test data generation",
            &mut self.error,
        ) {
            Some(Ok((n, dir))) => {
                self.gen_result = Some(format!("✔ {n} DICOM file(s) written to {}", dir.display()));
                if self.gen_load_after {
                    self.gen_open = false;
                    self.start_load(0, dir);
                }
            }
            Some(Err(e)) => self.error = Some(format!("Test data generation failed: {e:#}")),
            None => {}
        }

        // Poll the test-data download.
        self.poll_testdata_job(ctx);

        // Poll a model download / update batch.
        self.poll_models_job(ctx);

        // Poll the tool windows' workers.
        if let Some((slot, result)) =
            poll_tool_job(&mut self.autoseg_job, ctx, AUTOSEG.name, &mut self.error)
        {
            self.on_autoseg_done(slot, result);
        }
        if let Some((slot, result)) =
            poll_tool_job(&mut self.medsam2_job, ctx, SLICE_PROP.name, &mut self.error)
        {
            self.on_medsam2_done(slot, result);
        }
        if let Some((slot, result)) =
            poll_tool_job(&mut self.segvol_job, ctx, PROMPT_SEG.name, &mut self.error)
        {
            self.on_segvol_done(slot, result);
        }
        let nni_busy = self.nni_job.is_some();
        if let Some((slot, result)) = poll_tool_job(
            &mut self.nni_job,
            ctx,
            INTERACTIVE_SEG.name,
            &mut self.error,
        ) {
            self.on_nni_done(slot, result);
        } else if nni_busy && self.nni_job.is_none() {
            // Failed or cancelled: the prompt's mark goes, the rest wait on.
            if self.nni.inflight_mark {
                self.nni.marks.pop();
            }
            self.nni_next();
        }
        if let Some((slot, result)) =
            poll_tool_job(&mut self.body_job, ctx, BODY_CONTOUR.name, &mut self.error)
        {
            self.on_body_done(slot, result);
        }
        if let Some((slot, done)) = poll_tool_job(
            &mut self.synth_job,
            ctx,
            synth_win::SYNTHESIS.name,
            &mut self.error,
        ) {
            self.on_synthesis_done(slot, done);
        }
        if let Some((slot, done)) = poll_tool_job(
            &mut self.synth_eval_job,
            ctx,
            "Synthetic CT evaluation",
            &mut self.error,
        ) {
            self.on_sct_eval_done(slot, done);
        }
        if let Some((slot, result)) =
            poll_tool_job(&mut self.combine_job, ctx, COMBINE.name, &mut self.error)
        {
            self.on_combine_done(slot, result);
        }
        if let Some((slot, result)) = poll_tool_job(
            &mut self.derived_job,
            ctx,
            "Derived structure",
            &mut self.error,
        ) {
            self.on_derived_done(slot, result);
        }
        // One at a time: the queue is what "update all" leaves behind.
        if self.derived_job.is_none() && !self.derived_queue.is_empty() {
            let slot = self.derived_slot;
            self.next_derived_in_queue(slot);
        }
        if let Some(rows) = poll_job(
            &mut self.dose_est_job,
            ctx,
            "Dose analysis",
            &mut self.error,
        ) {
            self.on_dose_est_done(rows);
        }
        match poll_job(&mut self.dvh_job, ctx, "DVH", &mut self.error) {
            Some(Ok(done)) => self.on_dvh_done(done),
            Some(Err(e)) if !progress::is_cancellation(&e) => {
                self.error = Some(format!("DVH failed: {e:#}"));
            }
            _ => {}
        }
        if let Some((slot, outcome)) =
            poll_tool_job(&mut self.motion_job, ctx, MOTION.name, &mut self.error)
        {
            self.on_motion_done(slot, outcome);
        }

        // Poll background registration.
        if let Some((fixed_slot, out)) =
            poll_tool_job(&mut self.reg_job, ctx, "Registration", &mut self.error)
        {
            self.install_registration(out);
            self.fusion_on = true;
            // Re-propagate the crosshair through the new transform.
            let cursor = self.slots[fixed_slot].cursor;
            self.set_cursor(fixed_slot, cursor, usize::MAX);
        }

        // Poll an archive job - a scan, an import, an upload or a removal.
        match poll_job(&mut self.pacs_job, ctx, "Archive", &mut self.error) {
            Some(Ok(outcome)) => self.on_pacs_done(outcome),
            Some(Err(e)) => self.error = Some(format!("Archive: {e:#}")),
            None => {}
        }
        // A call to a paired PACS server, and the queue it keeps current.
        if let Some(r) = poll_job(&mut self.pacs_remote_job, ctx, "PACS", &mut self.error) {
            self.on_remote_done(r);
        }
        self.poll_remote_tasks(ctx.input(|i| i.time));
        // The PACS server on this machine.
        if let Some(r) = poll_job(
            &mut self.pacs_server_job,
            ctx,
            "PACS server",
            &mut self.error,
        ) {
            self.on_server_done(r);
        }
        self.poll_server_status(ctx.input(|i| i.time));

        // Poll a 2D/3D registration (or its DRRs, its nudge, its copy).
        if let Some(out) = poll_job(
            &mut self.r2d_job,
            ctx,
            "2D/3D registration",
            &mut self.error,
        ) {
            self.on_reg2d3d_done(out);
        }

        // Poll a DRR rendering.
        match poll_job(&mut self.drr_job, ctx, "DRR", &mut self.error) {
            Some(Ok(run)) => self.on_drr_done(run),
            // A cancelled render is what the user asked for, not a failure.
            Some(Err(e)) if !progress::is_cancellation(&e) => {
                self.error = Some(format!("DRR failed: {e:#}"));
            }
            _ => {}
        }

        // Poll a structure propagation.
        if let Some((dst_slot, out)) =
            poll_tool_job(&mut self.propagate_job, ctx, "Propagation", &mut self.error)
        {
            self.on_propagation_done(dst_slot, out);
        }

        // Poll a copy onto every phase of a 4D group.
        if let Some((what, job)) = self.phases_job.take() {
            let mut slot = Some(job);
            match poll_job(&mut slot, ctx, "Copy to phases", &mut self.error) {
                Some(Ok(phases)) => self.on_phases_copied(what, phases),
                Some(Err(e)) if !progress::is_cancellation(&e) => {
                    self.error = Some(format!("Copy to phases failed: {e:#}"));
                }
                Some(Err(_)) => {}
                None => {
                    if let Some(job) = slot {
                        self.phases_job = Some((what, job));
                    }
                }
            }
        }

        // Poll the 4D phase reader, run the cine clock, and hand the frame
        // egui has just delivered to whoever asked for it - a recording of a
        // run, or a still of the views.
        self.poll_phase_cache(ctx);
        self.play_tick(ctx);
        let shot = ctx.input(|i| {
            i.events.iter().rev().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        self.record_tick(ctx, shot.as_ref());
        self.snapshot_tick(ctx, shot.as_ref());

        // Poll a workflow run: its steps, what it shows, its end.
        self.poll_workflow_run(ctx);

        // Poll a vector-field re-sampling.
        if let Some(field) = poll_job(&mut self.field_job, ctx, "Vector field", &mut self.error) {
            if let Some(reg) = &mut self.registration {
                reg.field = Arc::new(field);
            }
        }
    }
}
