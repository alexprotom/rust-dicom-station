//! The operations that spawn a background job.
//!
//! Every long operation follows the same shape: snapshot the inputs, spawn a
//! thread, poll the channel each frame (`poll_job`), and land the result --
//! validating it first where the underlying data could have changed
//! meanwhile. The `Job` type and `poll_job` itself live in the parent
//! module; see `docs/architecture.md` for the pattern.

use super::*;

impl ViewerApp {
    pub(super) fn start_load(&mut self, slot: usize, path: PathBuf) {
        if self.loading.is_some() {
            self.pending_load = Some((slot, path));
            return;
        }
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = loader::load_directory(&path, &p2);
            let _ = tx.send(LoadResult::Study(Box::new(res), slot));
        });
        self.loading = Some(Job { progress, rx });
    }

    pub(super) fn start_series_switch(&mut self, slot: usize, idx: usize) {
        if self.loading.is_some() {
            return;
        }
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let series = study.series[idx].clone();
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

    /// A folder finished loading (*File ▶ Add DICOM folder*): merge it into
    /// an occupied slot, or install it into an empty one. Merging leaves the
    /// displayed volume and all selections untouched — the new patients /
    /// studies / series simply appear in the data tree.
    pub(super) fn absorb_loaded_study(&mut self, slot: usize, study: LoadedStudy) {
        if self.slots[slot].study.is_some() {
            let dest = self.slots[slot].study.as_mut().unwrap();
            let notes = loader::merge_study(dest, study);
            dest.warnings.extend(notes);
            self.settings_gen += 1;
        } else {
            self.on_study_loaded(slot, study);
        }
    }

    pub(super) fn on_study_loaded(&mut self, slot: usize, study: LoadedStudy) {
        let other_loaded = self.slots[1 - slot].study.is_some();
        // Shared W/L: adopt the study default unless another study is already up.
        if !other_loaded {
            self.window_center = study.default_window.0;
            self.window_width = study.default_window.1;
        }
        if !study.doses.is_empty() && self.dose_mode == DoseMode::Off {
            self.dose_mode = DoseMode::Both;
        }
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
        if slot == 1 {
            self.comparison = true;
        }
        // Any previous registration no longer matches the loaded volumes,
        // and open viewers for this slot reference stale data.
        self.planar_windows.retain(|w| w.slot != slot);
        self.d3_windows.retain(|w| w.slot != slot);
        if self.maximized.map(|(s, _)| s == slot).unwrap_or(false) {
            self.maximized = None;
        }
        self.clear_registration();
        self.settings_gen += 1;
    }

    pub(super) fn apply_new_volume(
        &mut self,
        slot: usize,
        vol: Arc<Volume>,
        window: (f32, f32),
        idx: usize,
    ) {
        let other_loaded = self.slots[1 - slot].study.is_some();
        if !other_loaded {
            self.window_center = window.0;
            self.window_width = window.1;
        }
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
    }

    /// Generate a transformed copy of the source study into the other slot
    /// (background thread; the applied parameters are the ground truth).
    pub(super) fn start_simulation(&mut self) {
        if self.sim_job.is_some() || self.loading.is_some() {
            return;
        }
        let source = self.sim_source.min(1);
        let target = 1 - source;
        let Some(study) = &self.slots[source].study else {
            self.error = Some(format!(
                "Load a dataset into slot {} first",
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
        progress.set("starting…");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let out = simulate::generate_transformed_study(&src, &params, &p2);
            let _ = tx.send((target, out));
        });
        self.sim_job = Some(Job { progress, rx });
    }

    /// Export the dialog's dataset as DICOM files into its output folder.
    pub(super) fn start_export(&mut self) {
        if self.export_job.is_some() {
            return;
        }
        let slot = self.export_slot.min(1);
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let Some(params) = self.export_params.clone() else {
            return;
        };
        let dir = PathBuf::from(self.export_dir.trim());
        if dir.as_os_str().is_empty() {
            self.error = Some("Choose an output folder for the export".into());
            return;
        }
        let src = study.clone();
        let progress = Arc::new(Progress::default());
        progress.set("starting…");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = dicom_export::export_study(&src, &dir, &params, &p2)
                .map(|n| (n, dir.display().to_string()));
            let _ = tx.send(res);
        });
        self.export_result = None;
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
        progress.set("starting…");
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
        let models_dir = self.engine_models_dir(models::Engine::TotalSegmentator);
        let (slot, variant, device, parts) = (d.slot, d.variant, d.device, d.parts);
        self.persist_settings();
        let progress = Arc::new(Progress::default());
        progress.set("Starting auto-segmentation…");
        self.autoseg_slot = slot;
        self.autoseg_job = Some(Job::spawn(progress, move |p| {
            (
                slot,
                autoseg::run(&volume, variant, device, parts, &models_dir, p),
            )
        }));
    }

    /// A run finished: verify the slot still shows the same volume, then
    /// open the organ-selection dialog in place of the tool window.
    pub(super) fn on_autoseg_done(&mut self, slot: usize, result: autoseg::AutosegResult) {
        self.autoseg_dialog = None;
        if !self.slot_still_shows(slot, result.volume_dims, &result.frame_of_reference_uid) {
            self.error = Some(stale_result(&AUTOSEG));
            return;
        }
        if result.organs.is_empty() {
            self.error = Some("Auto-segmentation found no organs in this volume.".into());
            return;
        }
        let selected = vec![true; result.organs.len()];
        self.autoseg_pending = Some(AutosegPending {
            slot,
            result,
            selected,
            also_rs: false,
        });
    }

    // -- Tools ▶ Anonymize DICOM folder ------------------------------------
    pub(super) fn anon_start_scan(&mut self) {
        let dir = PathBuf::from(self.anon_dir.trim());
        if dir.as_os_str().is_empty() {
            self.error = Some("Choose a folder with DICOM data first".into());
            return;
        }
        let progress = Arc::new(Progress::default());
        progress.set("starting…");
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
        progress.set("starting…");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(anonymize::apply(&files, &root, &params, &p2));
        });
        self.anon_result = None;
        self.anon_apply_job = Some(Job { progress, rx });
    }
}
