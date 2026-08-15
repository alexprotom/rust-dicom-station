//! The egui application: menu bar, toolbar, side panel, and one or two rows
//! (comparison mode) of three linked MPR views.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use egui::{
    Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, TextureHandle,
    TextureOptions, Vec2,
};

use rayon::prelude::*;

use crate::dicom_export;
use crate::loader::{self, LoadedStudy, Progress};
use crate::registration::{
    self, RegKind, RegParams, RegProgress, RegistrationResult, Transform3,
};
use crate::render;
use crate::simulate::{self, SimParams};
use crate::volume::{ViewPlane, Volume};

const SLOT_NAMES: [&str; 2] = ["A", "B"];

// ---------------------------------------------------------------------------
// Dose display settings
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DoseMode {
    Off,
    Colorwash,
    Isodose,
    Both,
}

impl DoseMode {
    fn label(self) -> &'static str {
        match self {
            DoseMode::Off => "Off",
            DoseMode::Colorwash => "Colorwash",
            DoseMode::Isodose => "Isodose lines",
            DoseMode::Both => "Colorwash + isodose",
        }
    }
    fn wash(self) -> bool {
        matches!(self, DoseMode::Colorwash | DoseMode::Both)
    }
    fn iso(self) -> bool {
        matches!(self, DoseMode::Isodose | DoseMode::Both)
    }
}

struct IsoLevel {
    pct: f32,
    color: Color32,
    on: bool,
}

fn default_iso_levels() -> Vec<IsoLevel> {
    vec![
        IsoLevel { pct: 107.0, color: Color32::from_rgb(255, 0, 255), on: true },
        IsoLevel { pct: 100.0, color: Color32::from_rgb(255, 0, 0), on: true },
        IsoLevel { pct: 95.0, color: Color32::from_rgb(255, 128, 0), on: true },
        IsoLevel { pct: 90.0, color: Color32::from_rgb(255, 255, 0), on: true },
        IsoLevel { pct: 80.0, color: Color32::from_rgb(0, 220, 0), on: true },
        IsoLevel { pct: 70.0, color: Color32::from_rgb(0, 255, 255), on: true },
        IsoLevel { pct: 50.0, color: Color32::from_rgb(0, 128, 255), on: true },
        IsoLevel { pct: 30.0, color: Color32::from_rgb(0, 0, 255), on: true },
    ]
}

/// Common CT window presets: (name, center, width) in HU.
const WL_PRESETS: &[(&str, f32, f32)] = &[
    ("Brain", 40.0, 80.0),
    ("Subdural", 75.0, 215.0),
    ("Stroke", 32.0, 8.0),
    ("Head/Neck soft tissue", 50.0, 350.0),
    ("Temporal bone", 600.0, 2800.0),
    ("Lungs", -600.0, 1500.0),
    ("Mediastinum", 50.0, 350.0),
    ("Abdomen", 50.0, 400.0),
    ("Liver", 30.0, 150.0),
    ("Spine soft tissue", 50.0, 250.0),
    ("Bone", 400.0, 1800.0),
    ("Angio (CTA)", 170.0, 600.0),
];

// ---------------------------------------------------------------------------
// Per-viewport state and caches
// ---------------------------------------------------------------------------

struct ViewState {
    plane: ViewPlane,
    slice: usize,
    /// Screen pixels per mm; 0 means auto-fit.
    zoom: f32,
    /// Pan offset of the image center relative to the viewport center, mm.
    pan: Vec2,
    /// Fractional mouse-wheel line accumulator for slice stepping.
    scroll_accum: f32,
    tex: Option<TextureHandle>,
    dose_tex: Option<TextureHandle>,
    img_key: Option<(usize, u32, u32)>,
    dose_key: Option<u64>,
    contour_key: Option<u64>,
    slice_buf: Vec<i16>,
    gray_buf: Vec<Color32>,
    dose_plane: Vec<f32>,
    dose_rgba: Vec<Color32>,
    iso_segs: Vec<(usize, render::Segment)>,
    contours: Vec<(usize, render::RoiPlaneGraphics)>,
    fusion_tex: Option<TextureHandle>,
    fusion_key: Option<u64>,
}

impl ViewState {
    fn new(plane: ViewPlane) -> Self {
        ViewState {
            plane,
            slice: 0,
            zoom: 0.0,
            pan: Vec2::ZERO,
            scroll_accum: 0.0,
            tex: None,
            dose_tex: None,
            img_key: None,
            dose_key: None,
            contour_key: None,
            slice_buf: Vec::new(),
            gray_buf: Vec::new(),
            dose_plane: Vec::new(),
            dose_rgba: Vec::new(),
            iso_segs: Vec::new(),
            contours: Vec::new(),
            fusion_tex: None,
            fusion_key: None,
        }
    }

    fn invalidate(&mut self) {
        self.img_key = None;
        self.dose_key = None;
        self.contour_key = None;
        self.fusion_key = None;
    }
}

fn fresh_views() -> [ViewState; 3] {
    [
        ViewState::new(ViewPlane::Axial),
        ViewState::new(ViewPlane::Sagittal),
        ViewState::new(ViewPlane::Coronal),
    ]
}

// ---------------------------------------------------------------------------
// A loaded study with its own display state ("A" = primary, "B" = comparison)
// ---------------------------------------------------------------------------

struct StudySlot {
    study: Option<LoadedStudy>,
    views: [ViewState; 3],
    /// Fractional voxel coords of the linked crosshair (in this slot's volume).
    cursor: [f64; 3],
    roi_visible: Vec<bool>,
    active_dose: usize,
    dose_reference: f32,
}

impl StudySlot {
    fn empty() -> Self {
        StudySlot {
            study: None,
            views: fresh_views(),
            cursor: [0.0; 3],
            roi_visible: Vec::new(),
            active_dose: 0,
            dose_reference: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Background loading
// ---------------------------------------------------------------------------

enum LoadResult {
    Study(Box<anyhow::Result<LoadedStudy>>, usize),
    Volume(Box<anyhow::Result<(Volume, (f32, f32), Vec<String>)>>, usize, usize),
}

struct LoadJob {
    progress: Arc<Progress>,
    rx: mpsc::Receiver<LoadResult>,
}

struct RegJob {
    progress: Arc<RegProgress>,
    rx: mpsc::Receiver<anyhow::Result<RegistrationResult>>,
    /// Slot used as the fixed image for this run.
    fixed_slot: usize,
}

struct SimJob {
    progress: Arc<Progress>,
    rx: mpsc::Receiver<(usize, LoadedStudy)>,
}

struct ExportJob {
    progress: Arc<Progress>,
    rx: mpsc::Receiver<anyhow::Result<(usize, String)>>,
}

/// A completed registration plus the direction it was run in.
struct ActiveRegistration {
    result: RegistrationResult,
    /// The fixed image's slot; the transform maps this slot's patient
    /// coordinates into the other (moving) slot's. The fusion overlay is
    /// drawn on this slot's views.
    fixed_slot: usize,
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

pub struct ViewerApp {
    slots: [StudySlot; 2],
    /// Comparison mode: study B shown in a second row of three views.
    comparison: bool,
    /// Propagate the crosshair between studies via patient coordinates.
    link_studies: bool,
    /// Slot whose readout is expanded in the status bar.
    hovered_slot: usize,

    loading: Option<LoadJob>,
    /// A load queued behind the one in flight (slot, directory).
    pending_load: Option<(usize, PathBuf)>,
    error: Option<String>,

    // Registration (direction selectable: either study can be the fixed one).
    registration: Option<ActiveRegistration>,
    reg_job: Option<RegJob>,
    /// Fixed-image slot for the *next* registration run (0 = A, 1 = B).
    reg_fixed_slot: usize,
    fusion_on: bool,
    fusion_weight: f32,
    /// Bumped when the registration result changes → fusion cache rebuild.
    reg_gen: u64,
    reg_iterations: usize,
    reg_samples: usize,
    reg_grid_mm: f64,

    // Study transform simulator (registration QA).
    sim_source: usize,
    sim_params: SimParams,
    sim_job: Option<SimJob>,
    last_sim: Option<String>,
    export_job: Option<ExportJob>,
    export_result: Option<String>,

    window_center: f32,
    window_width: f32,

    show_contours: bool,
    show_crosshair: bool,
    show_labels: bool,
    show_isocenters: bool,

    dose_mode: DoseMode,
    dose_opacity: f32,
    dose_threshold_pct: f32,
    iso_levels: Vec<IsoLevel>,

    /// Bumped whenever ROI visibility / dose settings change → cache rebuild.
    settings_gen: u64,
}

impl ViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial_a: Option<PathBuf>,
        initial_b: Option<PathBuf>,
    ) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut app = ViewerApp {
            slots: [StudySlot::empty(), StudySlot::empty()],
            comparison: initial_b.is_some(),
            link_studies: true,
            hovered_slot: 0,
            loading: None,
            pending_load: None,
            error: None,
            registration: None,
            reg_job: None,
            reg_fixed_slot: 0,
            fusion_on: false,
            fusion_weight: 1.0,
            reg_gen: 0,
            reg_iterations: 300,
            reg_samples: 3000,
            reg_grid_mm: 32.0,
            sim_source: 0,
            sim_params: SimParams::default(),
            sim_job: None,
            last_sim: None,
            export_job: None,
            export_result: None,
            window_center: 40.0,
            window_width: 400.0,
            show_contours: true,
            show_crosshair: true,
            show_labels: true,
            show_isocenters: true,
            dose_mode: DoseMode::Off,
            dose_opacity: 0.45,
            dose_threshold_pct: 15.0,
            iso_levels: default_iso_levels(),
            settings_gen: 0,
        };
        if let Some(p) = initial_a {
            app.start_load(0, p);
        }
        if let Some(p) = initial_b {
            app.pending_load = Some((1, p));
        }
        app
    }

    fn start_load(&mut self, slot: usize, path: PathBuf) {
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
        self.loading = Some(LoadJob { progress, rx });
    }

    fn start_series_switch(&mut self, slot: usize, idx: usize) {
        if self.loading.is_some() {
            return;
        }
        let Some(study) = &self.slots[slot].study else { return };
        let series = study.series[idx].clone();
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = loader::load_series_volume(&series, &p2);
            let _ = tx.send(LoadResult::Volume(Box::new(res), slot, idx));
        });
        self.loading = Some(LoadJob { progress, rx });
    }

    fn on_study_loaded(&mut self, slot: usize, study: LoadedStudy) {
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
        s.roi_visible = study
            .structures
            .as_ref()
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
        s.study = Some(study);
        if slot == 1 {
            self.comparison = true;
        }
        // Any previous registration no longer matches the loaded volumes.
        self.clear_registration();
        self.settings_gen += 1;
    }

    fn apply_new_volume(&mut self, slot: usize, vol: Volume, window: (f32, f32), idx: usize) {
        let other_loaded = self.slots[1 - slot].study.is_some();
        if !other_loaded {
            self.window_center = window.0;
            self.window_width = window.1;
        }
        let s = &mut self.slots[slot];
        if let Some(study) = &mut s.study {
            study.volume = vol;
            study.active_series = idx;
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
            self.clear_registration();
            self.settings_gen += 1;
        }
    }

    fn close_comparison(&mut self) {
        self.slots[1] = StudySlot::empty();
        self.comparison = false;
        self.hovered_slot = 0;
        self.clear_registration();
    }

    fn clear_registration(&mut self) {
        if let Some(job) = &self.reg_job {
            job.progress.cancel();
        }
        self.registration = None;
        self.fusion_on = false;
        self.reg_gen += 1;
    }

    fn start_registration(&mut self, kind: RegKind) {
        if self.reg_job.is_some() {
            return;
        }
        let fixed_slot = self.reg_fixed_slot.min(1);
        let moving_slot = 1 - fixed_slot;
        let (Some(f), Some(m)) = (
            &self.slots[fixed_slot].study,
            &self.slots[moving_slot].study,
        ) else {
            self.error =
                Some("Registration needs two loaded studies (comparison mode)".into());
            return;
        };
        let fixed = f.volume.clone();
        let moving = m.volume.clone();
        let params = RegParams {
            kind,
            levels: 3,
            iterations: self.reg_iterations,
            samples: self.reg_samples,
            grid_spacing_mm: self.reg_grid_mm,
            fixed_threshold: -500.0,
        };
        let progress = Arc::new(RegProgress::default());
        progress.set("starting…");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = registration::register(&fixed, &moving, &params, &p2);
            let _ = tx.send(res);
        });
        self.reg_job = Some(RegJob { progress, rx, fixed_slot });
    }

    /// Generate a transformed copy of the source study into the other slot
    /// (background thread; the applied parameters are the ground truth).
    fn start_simulation(&mut self) {
        if self.sim_job.is_some() || self.loading.is_some() {
            return;
        }
        let source = self.sim_source.min(1);
        let target = 1 - source;
        let Some(study) = &self.slots[source].study else {
            self.error = Some(format!(
                "Load a study into slot {} first",
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
        self.sim_job = Some(SimJob { progress, rx });
    }

    /// Export a loaded study as DICOM files into a user-chosen folder.
    fn start_export(&mut self, slot: usize) {
        if self.export_job.is_some() {
            return;
        }
        let Some(study) = &self.slots[slot].study else { return };
        let Some(dir) = rfd::FileDialog::new()
            .set_title(&format!("Export study {} as DICOM — choose folder", SLOT_NAMES[slot]))
            .pick_folder()
        else {
            return;
        };
        let src = study.clone();
        let progress = Arc::new(Progress::default());
        progress.set("starting…");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = dicom_export::export_study(&src, &dir, &p2)
                .map(|n| (n, dir.display().to_string()));
            let _ = tx.send(res);
        });
        self.export_job = Some(ExportJob { progress, rx });
    }

    fn reset_all_views(&mut self) {
        for s in &mut self.slots {
            for v in &mut s.views {
                v.zoom = 0.0;
                v.pan = Vec2::ZERO;
            }
        }
    }

    /// Combined hash of everything that affects dose overlays of a slot.
    fn dose_settings_hash(&self, slot: usize) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        };
        mix(slot as u64 + 1);
        mix(self.slots[slot].active_dose as u64);
        mix(self.dose_mode as u64);
        mix(self.dose_opacity.to_bits() as u64);
        mix(self.dose_threshold_pct.to_bits() as u64);
        mix(self.slots[slot].dose_reference.to_bits() as u64);
        for l in &self.iso_levels {
            mix(l.pct.to_bits() as u64 | ((l.on as u64) << 40));
        }
        mix(self.settings_gen);
        h
    }

    fn contour_settings_hash(&self, slot: usize) -> u64 {
        let mut h: u64 = 0x9e3779b97f4a7c15 ^ (slot as u64).wrapping_mul(0xff51afd7ed558ccd);
        for (i, v) in self.slots[slot].roi_visible.iter().enumerate() {
            if *v {
                h = h.rotate_left(7) ^ (i as u64 + 1);
            }
        }
        h ^ self.settings_gen.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn pick_folder(title: &str) -> Option<PathBuf> {
        rfd::FileDialog::new().set_title(title).pick_folder()
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Poll background loading.
        if let Some(job) = &self.loading {
            match job.rx.try_recv() {
                Ok(LoadResult::Study(res, slot)) => {
                    self.loading = None;
                    match *res {
                        Ok(study) => self.on_study_loaded(slot, study),
                        Err(e) => self.error = Some(format!("{e:#}")),
                    }
                }
                Ok(LoadResult::Volume(res, slot, idx)) => {
                    self.loading = None;
                    match *res {
                        Ok((vol, window, warnings)) => {
                            self.apply_new_volume(slot, vol, window, idx);
                            if let Some(study) = &mut self.slots[slot].study {
                                study.warnings.extend(warnings);
                            }
                        }
                        Err(e) => self.error = Some(format!("{e:#}")),
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(80));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.loading = None;
                    self.error = Some("Loading thread terminated unexpectedly".into());
                }
            }
        }
        // Kick a queued load once the current one finished.
        if self.loading.is_none() {
            if let Some((slot, path)) = self.pending_load.take() {
                self.start_load(slot, path);
            }
        }

        // Poll background simulation.
        if let Some(job) = &self.sim_job {
            match job.rx.try_recv() {
                Ok((target, study)) => {
                    self.sim_job = None;
                    self.on_study_loaded(target, study);
                    self.comparison = true;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.sim_job = None;
                    self.error = Some("Simulation thread terminated unexpectedly".into());
                }
            }
        }

        // Poll background export.
        if let Some(job) = &self.export_job {
            match job.rx.try_recv() {
                Ok(Ok((n, dir))) => {
                    self.export_job = None;
                    self.export_result = Some(format!("✔ {n} DICOM file(s) written to {dir}"));
                }
                Ok(Err(e)) => {
                    self.export_job = None;
                    self.error = Some(format!("Export failed: {e:#}"));
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.export_job = None;
                    self.error = Some("Export thread terminated unexpectedly".into());
                }
            }
        }

        // Poll background registration.
        if let Some(job) = &self.reg_job {
            match job.rx.try_recv() {
                Ok(Ok(result)) => {
                    let fixed_slot = self.reg_job.as_ref().map(|j| j.fixed_slot).unwrap_or(0);
                    self.reg_job = None;
                    self.registration = Some(ActiveRegistration { result, fixed_slot });
                    self.fusion_on = true;
                    self.reg_gen += 1;
                    // Re-propagate the crosshair through the new transform.
                    let cursor = self.slots[fixed_slot].cursor;
                    self.set_cursor(fixed_slot, cursor, usize::MAX);
                }
                Ok(Err(e)) => {
                    self.reg_job = None;
                    let msg = format!("{e:#}");
                    if !msg.contains("cancelled") {
                        self.error = Some(format!("Registration failed: {msg}"));
                    }
                }
                Err(mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.reg_job = None;
                    self.error = Some("Registration thread terminated unexpectedly".into());
                }
            }
        }

        self.menu_bar(ui, &ctx);
        self.top_bar(ui);
        self.side_panel(ui);
        self.status_bar(ui);
        self.central_views(ui);
        self.modals(&ctx);
    }
}

impl ViewerApp {
    // -- Menu bar ---------------------------------------------------------

    fn menu_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut open_a = false;
        let mut open_b = false;
        let mut close_b = false;
        let mut reset_views = false;
        let mut do_reg: Option<RegKind> = None;
        let mut cancel_reg = false;
        let mut clear_reg = false;

        egui::Panel::top(egui::Id::new("menu_bar")).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("📂 Open study…").clicked() {
                        open_a = true;
                        ui.close();
                    }
                    if ui.button("📂 Open comparison study (B)…").clicked() {
                        open_b = true;
                        ui.close();
                    }
                    let has_b = self.slots[1].study.is_some();
                    if ui
                        .add_enabled(has_b, egui::Button::new("Close comparison study"))
                        .clicked()
                    {
                        close_b = true;
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui
                        .checkbox(&mut self.comparison, "Comparison mode (2 × 3 views)")
                        .clicked()
                    {
                        ui.close();
                    }
                    ui.checkbox(&mut self.link_studies, "Link crosshairs between studies");
                    ui.separator();
                    ui.checkbox(&mut self.show_contours, "Contours");
                    ui.checkbox(&mut self.show_crosshair, "Crosshair");
                    ui.checkbox(&mut self.show_labels, "Orientation labels");
                    ui.checkbox(&mut self.show_isocenters, "Isocenters");
                    ui.separator();
                    if ui.button("Reset all views").clicked() {
                        reset_views = true;
                        ui.close();
                    }
                });
                ui.menu_button("Registration", |ui| {
                    let both =
                        self.slots[0].study.is_some() && self.slots[1].study.is_some();
                    let running = self.reg_job.is_some();
                    ui.label("Direction (moving ▶ fixed):");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.reg_fixed_slot, 0, "B ▶ A");
                        ui.selectable_value(&mut self.reg_fixed_slot, 1, "A ▶ B");
                    });
                    ui.separator();
                    let moving = SLOT_NAMES[1 - self.reg_fixed_slot.min(1)];
                    let fixed = SLOT_NAMES[self.reg_fixed_slot.min(1)];
                    if ui
                        .add_enabled(
                            both && !running,
                            egui::Button::new(format!("Rigid: register {moving} ▶ {fixed}")),
                        )
                        .on_hover_text(
                            "6-DOF Euler transform, ASGD optimizer (elastix-style), \
                             3 resolution levels",
                        )
                        .clicked()
                    {
                        do_reg = Some(RegKind::Rigid);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            both && !running,
                            egui::Button::new(format!(
                                "Deformable: register {moving} ▶ {fixed} (B-spline)"
                            )),
                        )
                        .on_hover_text(
                            "Rigid pre-alignment + cubic B-spline free-form deformation, \
                             ASGD optimizer (elastix-style)",
                        )
                        .clicked()
                    {
                        do_reg = Some(RegKind::Deformable);
                        ui.close();
                    }
                    ui.separator();
                    let fusion_label = match &self.registration {
                        Some(reg) => format!("Fusion overlay on {}", SLOT_NAMES[reg.fixed_slot]),
                        None => "Fusion overlay".to_string(),
                    };
                    ui.add_enabled(
                        self.registration.is_some(),
                        egui::Checkbox::new(&mut self.fusion_on, fusion_label),
                    );
                    if ui
                        .add_enabled(running, egui::Button::new("Cancel registration"))
                        .clicked()
                    {
                        cancel_reg = true;
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.registration.is_some(),
                            egui::Button::new("Clear registration"),
                        )
                        .clicked()
                    {
                        clear_reg = true;
                        ui.close();
                    }
                    if !both {
                        ui.weak("Load two studies (comparison mode) first");
                    }
                });
                ui.menu_button("Help", |ui| {
                    ui.label("Mouse bindings:");
                    ui.weak("Left click / drag — move linked crosshair");
                    ui.weak("Mouse wheel — scroll slices");
                    ui.weak("Ctrl + wheel — zoom at cursor");
                    ui.weak("Middle drag — pan");
                    ui.weak("Right drag — window / level");
                    ui.weak("Double click — reset view");
                });
            });
        });

        if open_a {
            if let Some(dir) = Self::pick_folder("Select DICOM directory (study A)") {
                self.start_load(0, dir);
            }
        }
        if open_b {
            if let Some(dir) = Self::pick_folder("Select DICOM directory (study B)") {
                self.comparison = true;
                self.start_load(1, dir);
            }
        }
        if close_b {
            self.close_comparison();
        }
        if reset_views {
            self.reset_all_views();
        }
        if let Some(kind) = do_reg {
            self.start_registration(kind);
        }
        if cancel_reg {
            if let Some(job) = &self.reg_job {
                job.progress.cancel();
            }
        }
        if clear_reg {
            self.clear_registration();
        }
    }

    // -- Toolbar ----------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top(egui::Id::new("top_bar")).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 Open folder…").clicked() {
                    if let Some(dir) = Self::pick_folder("Select a DICOM directory") {
                        self.start_load(0, dir);
                    }
                }

                if self.slots[0].study.is_some() || self.slots[1].study.is_some() {
                    ui.separator();
                    ui.label("W/L:");
                    ui.add(
                        egui::DragValue::new(&mut self.window_center)
                            .speed(2.0)
                            .prefix("C "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.window_width)
                            .speed(4.0)
                            .range(1.0..=20000.0)
                            .prefix("W "),
                    );
                    let mut full_range = false;
                    egui::ComboBox::from_id_salt("wl_preset")
                        .selected_text("CT presets")
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for (name, c, w) in WL_PRESETS {
                                if ui
                                    .button(format!("{name}  (C {c:.0} / W {w:.0})"))
                                    .clicked()
                                {
                                    self.window_center = *c;
                                    self.window_width = *w;
                                }
                            }
                            ui.separator();
                            if ui.button("Full range").clicked() {
                                full_range = true;
                            }
                        });
                    if full_range {
                        if let Some(study) = &self.slots[self.hovered_slot.min(1)].study {
                            let v = &study.volume;
                            self.window_center = (v.min_value as f32 + v.max_value as f32) * 0.5;
                            self.window_width = (v.max_value as f32 - v.min_value as f32).max(1.0);
                        }
                    }

                    ui.separator();
                    ui.checkbox(&mut self.show_contours, "Contours");
                    ui.checkbox(&mut self.show_crosshair, "Crosshair");
                    ui.checkbox(&mut self.show_labels, "Labels");

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let mut parts = Vec::new();
                        for (i, s) in self.slots.iter().enumerate() {
                            if let Some(study) = &s.study {
                                let m = &study.meta;
                                parts.push(format!(
                                    "{}: {} {}",
                                    SLOT_NAMES[i],
                                    m.patient_name.replace('^', " "),
                                    m.study_date
                                ));
                            }
                        }
                        ui.label(egui::RichText::new(parts.join("   ")).weak());
                    });
                }
            });
        });
    }

    // -- Side panel -------------------------------------------------------

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
            return;
        }
        egui::Panel::left(egui::Id::new("side"))
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.registration_section(ui);
                    self.simulation_section(ui);
                    for slot in 0..2 {
                        if self.slots[slot].study.is_none() {
                            continue;
                        }
                        self.study_section(ui, slot);
                    }
                });
            });
    }

    fn registration_section(&mut self, ui: &mut egui::Ui) {
        let both = self.slots[0].study.is_some() && self.slots[1].study.is_some();
        if !both && self.registration.is_none() && self.reg_job.is_none() {
            return;
        }
        let mut do_reg: Option<RegKind> = None;
        let mut cancel_reg = false;
        let mut clear_reg = false;
        egui::CollapsingHeader::new(egui::RichText::new("Registration").strong())
            .default_open(true)
            .show(ui, |ui| {
                if let Some(job) = &self.reg_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                    if ui.button("Cancel").clicked() {
                        cancel_reg = true;
                    }
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Direction");
                    ui.selectable_value(&mut self.reg_fixed_slot, 0, "B ▶ A")
                        .on_hover_text("B is deformed/moved onto A; fusion shown on A");
                    ui.selectable_value(&mut self.reg_fixed_slot, 1, "A ▶ B")
                        .on_hover_text("A is deformed/moved onto B; fusion shown on B");
                });

                ui.horizontal(|ui| {
                    ui.label("Iterations/level");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_iterations)
                            .speed(10)
                            .range(50..=5000),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Samples/iter");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_samples)
                            .speed(100)
                            .range(500..=50000),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("B-spline grid");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_grid_mm)
                            .speed(1.0)
                            .range(8.0..=128.0)
                            .suffix(" mm"),
                    );
                });
                ui.horizontal(|ui| {
                    if ui.add_enabled(both, egui::Button::new("▶ Rigid")).clicked() {
                        do_reg = Some(RegKind::Rigid);
                    }
                    if ui
                        .add_enabled(both, egui::Button::new("▶ Deformable"))
                        .clicked()
                    {
                        do_reg = Some(RegKind::Deformable);
                    }
                });

                if let Some(reg) = &self.registration {
                    ui.separator();
                    let res = &reg.result;
                    let kind = match res.kind {
                        RegKind::Rigid => "Rigid (Euler 6-DOF)",
                        RegKind::Deformable => "Rigid + B-spline FFD",
                    };
                    let moving = SLOT_NAMES[1 - reg.fixed_slot];
                    let fixed = SLOT_NAMES[reg.fixed_slot];
                    ui.label(format!("✔ {kind}  ({moving} ▶ {fixed})"));
                    ui.weak(format!(
                        "MSD {:.1} ▶ {:.1}  ({} iters, {:.1} s)",
                        res.initial_metric,
                        res.final_metric,
                        res.iterations_run,
                        res.elapsed_secs
                    ));
                    let t = &res.transform.rigid;
                    ui.weak(format!(
                        "t = ({:.1}, {:.1}, {:.1}) mm  r = ({:.2}, {:.2}, {:.2})°",
                        t.params[3],
                        t.params[4],
                        t.params[5],
                        t.params[0].to_degrees(),
                        t.params[1].to_degrees(),
                        t.params[2].to_degrees()
                    ));
                    ui.checkbox(
                        &mut self.fusion_on,
                        format!("Fusion overlay on {fixed}"),
                    );
                    let resp = ui.add(
                        egui::Slider::new(&mut self.fusion_weight, 0.0..=1.0)
                            .text("Fusion blend"),
                    );
                    let _ = resp;
                    if ui.button("Clear registration").clicked() {
                        clear_reg = true;
                    }
                }
            });
        ui.separator();
        if let Some(kind) = do_reg {
            self.start_registration(kind);
        }
        if cancel_reg {
            if let Some(job) = &self.reg_job {
                job.progress.cancel();
            }
        }
        if clear_reg {
            self.clear_registration();
        }
    }

    /// Study transform simulator: apply a known rigid motion + optional
    /// Gaussian deformation to a study, generate the result into the other
    /// slot, and export any study as DICOM files.
    fn simulation_section(&mut self, ui: &mut egui::Ui) {
        if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
            return;
        }
        let mut do_generate = false;
        let mut do_export: Option<usize> = None;
        egui::CollapsingHeader::new(egui::RichText::new("Simulation (registration QA)").strong())
            .default_open(false)
            .show(ui, |ui| {
                if let Some(job) = &self.sim_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Source");
                    ui.selectable_value(&mut self.sim_source, 0, "A");
                    ui.selectable_value(&mut self.sim_source, 1, "B");
                    ui.weak(format!(
                        "▶ generates study {}",
                        SLOT_NAMES[1 - self.sim_source.min(1)]
                    ));
                });

                ui.label("Rigid motion:");
                ui.horizontal(|ui| {
                    ui.label("t (mm)");
                    for v in &mut self.sim_params.translation {
                        ui.add(egui::DragValue::new(v).speed(0.5).range(-200.0..=200.0));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("r (°)");
                    for v in &mut self.sim_params.rotation_deg {
                        ui.add(egui::DragValue::new(v).speed(0.2).range(-45.0..=45.0));
                    }
                });

                ui.label("Gaussian deformation (0 = off):");
                ui.horizontal(|ui| {
                    ui.label("amp (mm)");
                    for v in &mut self.sim_params.bump_amp {
                        ui.add(egui::DragValue::new(v).speed(0.5).range(-40.0..=40.0));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("σ (mm)");
                    ui.add(
                        egui::DragValue::new(&mut self.sim_params.bump_sigma)
                            .speed(1.0)
                            .range(5.0..=200.0),
                    );
                    ui.weak("centered at the crosshair");
                });

                let src_ok = self.slots[self.sim_source.min(1)].study.is_some();
                if ui
                    .add_enabled(
                        src_ok && self.loading.is_none(),
                        egui::Button::new(format!(
                            "⚙ Generate transformed study ▶ {}",
                            SLOT_NAMES[1 - self.sim_source.min(1)]
                        )),
                    )
                    .clicked()
                {
                    do_generate = true;
                }
                if let Some(s) = &self.last_sim {
                    ui.weak(format!("Ground truth {s}"));
                }

                ui.separator();
                ui.label("Export as DICOM files:");
                if let Some(job) = &self.export_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                } else {
                    ui.horizontal(|ui| {
                        for slot in 0..2 {
                            if ui
                                .add_enabled(
                                    self.slots[slot].study.is_some(),
                                    egui::Button::new(format!(
                                        "💾 Export {}…",
                                        SLOT_NAMES[slot]
                                    )),
                                )
                                .clicked()
                            {
                                do_export = Some(slot);
                            }
                        }
                    });
                }
                if let Some(msg) = &self.export_result {
                    ui.weak(msg);
                }
            });
        ui.separator();
        if do_generate {
            self.start_simulation();
        }
        if let Some(slot) = do_export {
            self.export_result = None;
            self.start_export(slot);
        }
    }

    fn study_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let header = {
            let study = self.slots[slot].study.as_ref().unwrap();
            let m = &study.meta;
            let both = self.slots[0].study.is_some() && self.slots[1].study.is_some();
            if both || slot == 1 {
                format!(
                    "Study {} — {} {}",
                    SLOT_NAMES[slot],
                    m.patient_name.replace('^', " "),
                    m.study_date
                )
            } else {
                format!("{} {}", m.patient_name.replace('^', " "), m.study_date)
            }
        };
        egui::CollapsingHeader::new(egui::RichText::new(header).strong())
            .id_salt(("study_hdr", slot))
            .default_open(true)
            .show(ui, |ui| {
                self.series_selector(ui, slot);
                self.structures_section(ui, slot);
                self.dose_section(ui, slot);
                self.plan_section(ui, slot);
                self.warnings_section(ui, slot);
            });
        ui.separator();
    }

    fn series_selector(&mut self, ui: &mut egui::Ui, slot: usize) {
        let mut switch_to = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            if study.series.len() > 1 {
                let active = study.active_series;
                let mut selected = active;
                let label = |s: &loader::SeriesInfo| {
                    format!(
                        "{} {} ({} sl.)",
                        s.modality,
                        if s.description.is_empty() { "series" } else { &s.description },
                        s.files.len()
                    )
                };
                egui::ComboBox::from_id_salt(("series_sel", slot))
                    .width(230.0)
                    .selected_text(label(&study.series[active]))
                    .show_ui(ui, |ui| {
                        for (i, s) in study.series.iter().enumerate() {
                            ui.selectable_value(&mut selected, i, label(s));
                        }
                    });
                if selected != active {
                    switch_to = Some(selected);
                }
            } else {
                let s = &study.series[study.active_series];
                ui.weak(format!(
                    "{} {} ({} sl.)",
                    s.modality,
                    if s.description.is_empty() { "series" } else { &s.description },
                    s.files.len()
                ));
            }
        }
        if let Some(i) = switch_to {
            self.start_series_switch(slot, i);
        }
    }

    fn structures_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let has = self.slots[slot]
            .study
            .as_ref()
            .is_some_and(|s| s.structures.is_some());
        if !has {
            // No RTSTRUCT in this study — show nothing.
            return;
        }
        let mut changed = false;
        {
            let StudySlot { study, roi_visible, .. } = &mut self.slots[slot];
            let ss = study.as_ref().unwrap().structures.as_ref().unwrap();
            let n_vis = roi_visible.iter().filter(|v| **v).count();
            egui::CollapsingHeader::new(format!("Structures ({}/{})", n_vis, ss.rois.len()))
                .id_salt(("structs", slot))
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.small_button("All").clicked() {
                            roi_visible.iter_mut().for_each(|v| *v = true);
                            changed = true;
                        }
                        if ui.small_button("None").clicked() {
                            roi_visible.iter_mut().for_each(|v| *v = false);
                            changed = true;
                        }
                        ui.weak(&ss.label);
                    });
                    for (i, roi) in ss.rois.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                            ui.painter().rect_filled(
                                rect,
                                2.0,
                                Color32::from_rgb(roi.color[0], roi.color[1], roi.color[2]),
                            );
                            let resp = ui.checkbox(
                                &mut roi_visible[i],
                                format!(
                                    "{}{}",
                                    roi.name,
                                    if roi.roi_type.is_empty() {
                                        String::new()
                                    } else {
                                        format!("  [{}]", roi.roi_type)
                                    }
                                ),
                            );
                            if resp.changed() {
                                changed = true;
                            }
                            resp.on_hover_text(format!(
                                "ROI {} · {} contour(s)",
                                roi.number,
                                roi.contours.len()
                            ));
                        });
                    }
                });
        }
        if changed {
            self.settings_gen += 1;
        }
    }

    fn dose_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n_doses = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.doses.len())
            .unwrap_or(0);
        if n_doses == 0 {
            // No RTDOSE in this study — show nothing.
            return;
        }
        let mut mode = self.dose_mode;
        let mut opacity = self.dose_opacity;
        let mut threshold = self.dose_threshold_pct;
        {
            let StudySlot { study, active_dose, dose_reference, .. } = &mut self.slots[slot];
            let doses = &study.as_ref().unwrap().doses;
            egui::CollapsingHeader::new("Dose")
                .id_salt(("dose", slot))
                .default_open(true)
                .show(ui, |ui| {
                    if doses.len() > 1 {
                        let mut sel = (*active_dose).min(doses.len() - 1);
                        egui::ComboBox::from_id_salt(("dose_sel", slot))
                            .width(230.0)
                            .selected_text(&doses[sel].label)
                            .show_ui(ui, |ui| {
                                for (i, d) in doses.iter().enumerate() {
                                    ui.selectable_value(&mut sel, i, &d.label);
                                }
                            });
                        *active_dose = sel;
                    }
                    let d = &doses[(*active_dose).min(doses.len() - 1)];
                    ui.weak(format!(
                        "{}  max {:.2} {}",
                        d.summation_type,
                        d.max_dose,
                        d.units.to_lowercase()
                    ));

                    egui::ComboBox::from_id_salt(("dose_mode", slot))
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            for m in
                                [DoseMode::Off, DoseMode::Colorwash, DoseMode::Isodose, DoseMode::Both]
                            {
                                ui.selectable_value(&mut mode, m, m.label());
                            }
                        });

                    ui.horizontal(|ui| {
                        ui.label("Reference");
                        ui.add(
                            egui::DragValue::new(dose_reference)
                                .speed(0.05)
                                .range(0.01..=1000.0)
                                .suffix(" Gy"),
                        );
                        if ui.small_button("max").clicked() {
                            *dose_reference = d.max_dose;
                        }
                    });
                    ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(
                        egui::Slider::new(&mut threshold, 0.0..=100.0).text("Threshold %"),
                    );
                });
        }
        self.dose_mode = mode;
        self.dose_opacity = opacity;
        self.dose_threshold_pct = threshold;

        // Isodose levels are shared; show them once (under the first slot
        // that has dose).
        let first_dose_slot = (0..2).find(|&s| {
            self.slots[s]
                .study
                .as_ref()
                .is_some_and(|st| !st.doses.is_empty())
        });
        if first_dose_slot == Some(slot) {
            egui::CollapsingHeader::new("Isodose levels (% of reference)")
                .id_salt("iso_levels")
                .default_open(true)
                .show(ui, |ui| {
                    for l in &mut self.iso_levels {
                        ui.horizontal(|ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                            ui.painter().rect_filled(rect, 2.0, l.color);
                            ui.checkbox(&mut l.on, format!("{:.0}%", l.pct));
                        });
                    }
                });
        }
    }

    fn plan_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else { return };
        if study.plans.is_empty() {
            // No RTPLAN in this study — show nothing.
            return;
        }
        for (pi, plan) in study.plans.iter().enumerate() {
            egui::CollapsingHeader::new(format!(
                "Plan: {}",
                if plan.label.is_empty() { "unnamed" } else { &plan.label }
            ))
            .id_salt(("plan", slot, pi))
            .default_open(pi == 0)
            .show(ui, |ui| {
                if !plan.name.is_empty() && plan.name != plan.label {
                    ui.weak(format!("Name: {}", plan.name));
                }
                if !plan.plan_kind.is_empty() {
                    ui.weak(format!("Type: {}", plan.plan_kind));
                }
                if let Some(fx) = plan.n_fractions {
                    ui.weak(format!("Fractions: {fx}"));
                }
                if let Some(rx) = plan.target_prescription_dose {
                    ui.weak(format!("Prescription: {rx:.2} Gy"));
                }
                if !plan.date.is_empty() {
                    ui.weak(format!("Date: {}", plan.date));
                }
                if !plan.beams.is_empty() {
                    egui::Grid::new(("beam_grid", slot, pi))
                        .striped(true)
                        .min_col_width(10.0)
                        .show(ui, |ui| {
                            ui.strong("Beam");
                            ui.strong("Type");
                            ui.strong("G°");
                            ui.strong("C°");
                            ui.strong("E (MeV)");
                            ui.strong("MU");
                            ui.strong("CPs");
                            ui.end_row();
                            for b in &plan.beams {
                                ui.label(&b.name).on_hover_text(format!(
                                    "Beam {} · {} · dose/fx {}",
                                    b.number,
                                    if b.delivery_type.is_empty() {
                                        "TREATMENT"
                                    } else {
                                        &b.delivery_type
                                    },
                                    b.beam_dose
                                        .map(|d| format!("{d:.2} Gy"))
                                        .unwrap_or_else(|| "n/a".into()),
                                ));
                                ui.label(format!(
                                    "{}{}",
                                    b.radiation_type,
                                    if b.scan_mode.is_empty() {
                                        String::new()
                                    } else {
                                        format!("/{}", b.scan_mode)
                                    }
                                ));
                                ui.label(
                                    b.gantry_angle
                                        .map(|g| format!("{g:.0}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(
                                    b.couch_angle
                                        .map(|c| format!("{c:.0}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(match (b.energy_min, b.energy_max) {
                                    (Some(a), Some(bb)) if (a - bb).abs() > 0.01 => {
                                        format!("{a:.0}–{bb:.0}")
                                    }
                                    (Some(a), _) => format!("{a:.0}"),
                                    _ => "–".into(),
                                });
                                ui.label(
                                    b.meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(format!("{}", b.n_control_points));
                                ui.end_row();
                            }
                        });
                }
            });
        }
    }

    fn warnings_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else { return };
        if study.warnings.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(
            egui::RichText::new(format!("⚠ Warnings ({})", study.warnings.len()))
                .color(Color32::from_rgb(240, 190, 60)),
        )
        .id_salt(("warn", slot))
        .show(ui, |ui| {
            for w in &study.warnings {
                ui.label(egui::RichText::new(w).small());
            }
        });
    }

    // -- Status bar -------------------------------------------------------

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom(egui::Id::new("status")).show(ui, |ui| {
            ui.horizontal(|ui| {
                let any = self.slots.iter().any(|s| s.study.is_some());
                if !any {
                    ui.weak("No data loaded");
                    return;
                }
                for slot in 0..2 {
                    if slot == 1 && !self.comparison {
                        // Study B is hidden while comparison mode is off.
                        continue;
                    }
                    let s = &self.slots[slot];
                    let Some(study) = &s.study else { continue };
                    let v = &study.volume;
                    let c = s.cursor;
                    let p = v.voxel_to_patient(c[0], c[1], c[2]);
                    let both = self.comparison && self.slots[1].study.is_some();
                    let prefix = if both {
                        format!("{}: ", SLOT_NAMES[slot])
                    } else {
                        String::new()
                    };
                    if slot == self.hovered_slot || !both {
                        ui.monospace(format!(
                            "{}({:6.1},{:6.1},{:6.1})mm ijk({:3},{:3},{:3})",
                            prefix,
                            p.x,
                            p.y,
                            p.z,
                            c[0].round() as i64,
                            c[1].round() as i64,
                            c[2].round() as i64
                        ));
                    } else {
                        ui.monospace(prefix.trim_end().to_string());
                    }
                    if let Some(hu) =
                        v.get(c[0].round() as i64, c[1].round() as i64, c[2].round() as i64)
                    {
                        ui.monospace(format!("{hu:5} HU"));
                    }
                    if let Some(d) = study
                        .doses
                        .get(s.active_dose)
                        .and_then(|d| d.sample(p))
                    {
                        ui.monospace(format!(
                            "{:.2} Gy ({:.0}%)",
                            d,
                            100.0 * d / s.dose_reference.max(1e-6)
                        ));
                    }
                    if both && slot == 0 {
                        ui.separator();
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak("LMB crosshair · RMB W/L · MMB pan · wheel slice · Ctrl+wheel zoom · double-click reset");
                });
            });
        });
    }

    // -- Central: one or two rows of three views --------------------------

    fn central_views(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default_margins()
            .frame(egui::Frame::NONE.fill(Color32::from_gray(10)))
            .show(ui, |ui| {
                if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
                    self.empty_state(ui);
                    return;
                }
                let two_rows = self.comparison;
                let full = ui.available_rect_before_wrap();
                let row_gap = 6.0;
                let n_rows = if two_rows { 2.0 } else { 1.0 };
                let row_h = (full.height() - (n_rows - 1.0) * row_gap) / n_rows;

                for row in 0..(n_rows as usize) {
                    let y0 = full.top() + row as f32 * (row_h + row_gap);
                    let row_rect = Rect::from_min_size(
                        Pos2::new(full.left(), y0),
                        Vec2::new(full.width(), row_h),
                    );
                    if self.slots[row].study.is_some() {
                        self.study_row(ui, row, row_rect);
                    } else {
                        self.empty_row(ui, row, row_rect);
                    }
                }
            });
    }

    fn study_row(&mut self, ui: &mut egui::Ui, slot: usize, row_rect: Rect) {
        let gap = 4.0;
        let slider_h = 26.0;
        let col_w = (row_rect.width() - 2.0 * gap) / 3.0;
        for idx in 0..3 {
            let x0 = row_rect.left() + idx as f32 * (col_w + gap);
            let col = Rect::from_min_size(
                Pos2::new(x0, row_rect.top()),
                Vec2::new(col_w, row_rect.height()),
            );
            let view_rect =
                Rect::from_min_max(col.min, Pos2::new(col.max.x, col.max.y - slider_h));
            let slider_rect = Rect::from_min_max(
                Pos2::new(col.min.x + 6.0, col.max.y - slider_h + 2.0),
                Pos2::new(col.max.x - 6.0, col.max.y - 2.0),
            );
            self.one_view(ui, slot, idx, view_rect);
            let max_slice = self.slots[slot]
                .study
                .as_ref()
                .map(|s| {
                    s.volume
                        .plane_slice_count(self.slots[slot].views[idx].plane)
                        .saturating_sub(1)
                })
                .unwrap_or(0);
            if max_slice > 0 {
                let mut slice = self.slots[slot].views[idx].slice.min(max_slice);
                let resp = ui.put(
                    slider_rect,
                    egui::Slider::new(&mut slice, 0..=max_slice).show_value(false),
                );
                if resp.changed() {
                    self.slots[slot].views[idx].slice = slice;
                }
            }
        }
    }

    fn empty_row(&mut self, ui: &mut egui::Ui, slot: usize, rect: Rect) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(14));
        painter.text(
            rect.center() - Vec2::new(0.0, 24.0),
            Align2::CENTER_CENTER,
            format!("No comparison study ({})", SLOT_NAMES[slot]),
            FontId::proportional(15.0),
            Color32::GRAY,
        );
        let btn_rect = Rect::from_center_size(
            rect.center() + Vec2::new(0.0, 10.0),
            Vec2::new(220.0, 28.0),
        );
        if ui
            .put(btn_rect, egui::Button::new("📂 Open comparison study…"))
            .clicked()
        {
            if let Some(dir) = Self::pick_folder("Select DICOM directory (study B)") {
                self.start_load(slot, dir);
            }
        }
        if self.loading.is_some() {
            if let Some(job) = &self.loading {
                painter.text(
                    rect.center() + Vec2::new(0.0, 44.0),
                    Align2::CENTER_CENTER,
                    format!("⏳ {}", job.progress.get()),
                    FontId::proportional(13.0),
                    Color32::WHITE,
                );
            }
        }
    }

    fn empty_state(&mut self, ui: &mut egui::Ui) {
        ui.centered_and_justified(|ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.35);
                ui.heading("Rust DICOM / RT viewer");
                ui.add_space(8.0);
                if self.loading.is_some() {
                    ui.spinner();
                    if let Some(job) = &self.loading {
                        ui.label(job.progress.get());
                    }
                } else {
                    ui.label("Open a folder containing a DICOM study");
                    ui.add_space(8.0);
                    if ui.button("📂 Open folder…").clicked() {
                        if let Some(dir) = Self::pick_folder("Select a DICOM directory") {
                            self.start_load(0, dir);
                        }
                    }
                }
            });
        });
    }

    // -- One viewport -----------------------------------------------------

    fn one_view(&mut self, ui: &mut egui::Ui, slot: usize, idx: usize, rect: Rect) {
        let ctx = ui.ctx().clone();
        let plane = self.slots[slot].views[idx].plane;

        // ---- cache refresh (image, dose, contours) ----
        self.refresh_view_caches(&ctx, slot, idx);

        let slot_state = &self.slots[slot];
        let Some(study) = &slot_state.study else { return };
        let vol = &study.volume;
        let view = &slot_state.views[idx];

        let [w_px, h_px] = vol.plane_dims(plane);
        let [sx, sy] = vol.plane_spacing(plane);
        let w_mm = (w_px as f64 * sx) as f32;
        let h_mm = (h_px as f64 * sy) as f32;

        let fit_zoom = ((rect.width() / w_mm).min(rect.height() / h_mm) * 0.97).max(0.01);
        let zoom = if view.zoom > 0.0 { view.zoom } else { fit_zoom };
        let center = rect.center() + view.pan * zoom;
        let img_rect = Rect::from_center_size(center, Vec2::new(w_mm * zoom, h_mm * zoom));

        let px_to_screen = |p: [f32; 2]| -> Pos2 {
            Pos2::new(
                img_rect.left() + (p[0] + 0.5) * sx as f32 * zoom,
                img_rect.top() + (p[1] + 0.5) * sy as f32 * zoom,
            )
        };
        let screen_to_px = |s: Pos2| -> [f32; 2] {
            [
                (s.x - img_rect.left()) / (sx as f32 * zoom) - 0.5,
                (s.y - img_rect.top()) / (sy as f32 * zoom) - 0.5,
            ]
        };

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        let fusion_active = self.fusion_on
            && self
                .registration
                .as_ref()
                .is_some_and(|r| r.fixed_slot == slot)
            && view.fusion_tex.is_some();
        if fusion_active {
            if let Some(tex) = &view.fusion_tex {
                painter.image(
                    tex.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        } else if let Some(tex) = &view.tex {
            painter.image(
                tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        if self.dose_mode.wash() {
            if let Some(tex) = &view.dose_tex {
                painter.image(
                    tex.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        }

        // Isodose lines.
        if self.dose_mode.iso() {
            for (li, seg) in &view.iso_segs {
                if let Some(level) = self.iso_levels.get(*li) {
                    painter.line_segment(
                        [px_to_screen(seg.0), px_to_screen(seg.1)],
                        Stroke::new(1.6, level.color),
                    );
                }
            }
        }

        // Contours.
        if self.show_contours {
            if let Some(ss) = &study.structures {
                for (ri, gfx) in &view.contours {
                    let Some(roi) = ss.rois.get(*ri) else { continue };
                    let color = Color32::from_rgb(roi.color[0], roi.color[1], roi.color[2]);
                    let stroke = Stroke::new(1.8, color);
                    for pl in &gfx.polylines {
                        let pts: Vec<Pos2> = pl.iter().map(|p| px_to_screen(*p)).collect();
                        painter.add(egui::Shape::closed_line(pts, stroke));
                    }
                    for (a, b) in &gfx.segments {
                        painter.line_segment([px_to_screen(*a), px_to_screen(*b)], stroke);
                    }
                    for p in &gfx.points {
                        let c = px_to_screen(*p);
                        painter.line_segment(
                            [c + Vec2::new(-4.0, 0.0), c + Vec2::new(4.0, 0.0)],
                            stroke,
                        );
                        painter.line_segment(
                            [c + Vec2::new(0.0, -4.0), c + Vec2::new(0.0, 4.0)],
                            stroke,
                        );
                    }
                }
            }
        }

        // Isocenter markers.
        if self.show_isocenters {
            let mut seen: Vec<[i64; 3]> = Vec::new();
            for plan in &study.plans {
                for b in &plan.beams {
                    let Some(iso) = b.isocenter else { continue };
                    let key = [
                        (iso.x * 10.0) as i64,
                        (iso.y * 10.0) as i64,
                        (iso.z * 10.0) as i64,
                    ];
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.push(key);
                    let (pp, dz) =
                        render::patient_to_plane_pixel(vol, plane, view.slice, iso);
                    let on_plane = dz.abs() <= 1.0;
                    let alpha = if on_plane { 255 } else { 80 };
                    let col = Color32::from_rgba_unmultiplied(255, 230, 40, alpha);
                    let c = px_to_screen(pp);
                    if rect.expand(20.0).contains(c) {
                        let s = Stroke::new(1.5, col);
                        painter.circle_stroke(c, 6.0, s);
                        painter.line_segment([c + Vec2::new(-9.0, 0.0), c + Vec2::new(9.0, 0.0)], s);
                        painter.line_segment([c + Vec2::new(0.0, -9.0), c + Vec2::new(0.0, 9.0)], s);
                    }
                }
            }
        }

        // Crosshair.
        if self.show_crosshair {
            let cp = vol.voxel_to_plane_pixel(plane, slot_state.cursor);
            let c = px_to_screen([cp[0] as f32, cp[1] as f32]);
            let col = Color32::from_rgba_unmultiplied(120, 255, 120, 110);
            let s = Stroke::new(1.0, col);
            if rect.contains(Pos2::new(c.x, rect.center().y)) {
                painter.line_segment(
                    [Pos2::new(c.x, rect.top()), Pos2::new(c.x, rect.bottom())],
                    s,
                );
            }
            if rect.contains(Pos2::new(rect.center().x, c.y)) {
                painter.line_segment(
                    [Pos2::new(rect.left(), c.y), Pos2::new(rect.right(), c.y)],
                    s,
                );
            }
        }

        // Annotations.
        if self.show_labels {
            let n_slices = vol.plane_slice_count(plane);
            let both = self.comparison;
            let title = if both {
                format!("{} · {}", plane.title(), SLOT_NAMES[slot])
            } else {
                plane.title().to_string()
            };
            painter.text(
                rect.left_top() + Vec2::new(6.0, 4.0),
                Align2::LEFT_TOP,
                title,
                FontId::proportional(14.0),
                if slot == 0 {
                    Color32::from_rgb(255, 170, 60)
                } else {
                    Color32::from_rgb(120, 200, 255)
                },
            );
            painter.text(
                rect.right_top() + Vec2::new(-6.0, 4.0),
                Align2::RIGHT_TOP,
                format!("{}/{}", view.slice + 1, n_slices),
                FontId::proportional(12.0),
                Color32::LIGHT_GRAY,
            );
            // Anatomical edge labels.
            let (dx, dy) = vol.plane_screen_dirs(plane);
            let lbl = |v| crate::geometry::direction_label(v);
            let f = FontId::proportional(12.0);
            let lc = Color32::from_rgb(120, 200, 255);
            painter.text(
                Pos2::new(rect.right() - 8.0, rect.center().y),
                Align2::RIGHT_CENTER,
                lbl(dx),
                f.clone(),
                lc,
            );
            painter.text(
                Pos2::new(rect.left() + 8.0, rect.center().y),
                Align2::LEFT_CENTER,
                lbl(dx * -1.0),
                f.clone(),
                lc,
            );
            painter.text(
                Pos2::new(rect.center().x, rect.bottom() - 6.0),
                Align2::CENTER_BOTTOM,
                lbl(dy),
                f.clone(),
                lc,
            );
            painter.text(
                Pos2::new(rect.center().x, rect.top() + 4.0),
                Align2::CENTER_TOP,
                lbl(dy * -1.0),
                f,
                lc,
            );
        }

        // Loading overlay.
        if self.loading.is_some() {
            painter.rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 140));
            if idx == 1 && slot == 0 {
                if let Some(job) = &self.loading {
                    painter.text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        format!("⏳ {}", job.progress.get()),
                        FontId::proportional(15.0),
                        Color32::WHITE,
                    );
                }
            }
        }

        // ---- interaction ----
        let resp = ui.interact(
            rect,
            egui::Id::new(("viewport", slot, idx)),
            Sense::click_and_drag(),
        );
        let n_slices = vol.plane_slice_count(plane);

        let mut new_slice = None;
        let mut new_zoom = None;
        let mut new_pan = None;
        let mut new_cursor = None;
        let mut wl_delta = None;
        let mut reset_view = false;
        let mut new_accum = None;

        if resp.hovered() {
            let (wheel_lines, zoom_delta, pointer) = ui.input(|i| {
                let mut lines = 0.0f32;
                for e in &i.events {
                    if let egui::Event::MouseWheel { unit, delta, modifiers, .. } = e {
                        if !(modifiers.ctrl || modifiers.command) {
                            lines += match unit {
                                egui::MouseWheelUnit::Line => delta.y,
                                egui::MouseWheelUnit::Point => delta.y / 40.0,
                                egui::MouseWheelUnit::Page => delta.y * 10.0,
                            };
                        }
                    }
                }
                (lines, i.zoom_delta(), i.pointer.hover_pos())
            });
            if (zoom_delta - 1.0).abs() > 1e-4 {
                // Keep the point under the cursor fixed while zooming.
                let z0 = zoom;
                let z1 = (z0 * zoom_delta).clamp(fit_zoom * 0.2, fit_zoom * 40.0);
                if let Some(mp) = pointer {
                    let rel = (mp - rect.center()) / z0 - view.pan;
                    let pan1 = (mp - rect.center()) / z1 - rel;
                    new_pan = Some(pan1);
                }
                new_zoom = Some(z1);
            }
            if wheel_lines.abs() > 0.0 {
                let acc = view.scroll_accum + wheel_lines;
                let steps = acc.trunc() as i64;
                new_accum = Some(acc - steps as f32);
                if steps != 0 {
                    let s = (view.slice as i64 - steps).clamp(0, n_slices as i64 - 1) as usize;
                    new_slice = Some(s);
                }
            }
        }

        if resp.dragged_by(egui::PointerButton::Primary) || resp.clicked() {
            if let Some(mp) = resp.interact_pointer_pos() {
                let px = screen_to_px(mp);
                let vxl = vol.plane_pixel_to_voxel(plane, view.slice, px[0] as f64, px[1] as f64);
                new_cursor = Some(vxl);
            }
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            let d = resp.drag_delta();
            wl_delta = Some((d.x, d.y));
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            new_pan = Some(view.pan + d / zoom);
        }
        if resp.double_clicked() {
            reset_view = true;
        }
        let hovered = resp.hovered();

        // Apply interactions (mutable phase).
        if hovered {
            self.hovered_slot = slot;
        }
        if let Some(a) = new_accum {
            self.slots[slot].views[idx].scroll_accum = a;
        }
        if let Some(s) = new_slice {
            self.slots[slot].views[idx].slice = s;
        }
        if let Some(z) = new_zoom {
            self.slots[slot].views[idx].zoom = z;
        }
        if let Some(p) = new_pan {
            self.slots[slot].views[idx].pan = p;
        }
        if reset_view {
            self.slots[slot].views[idx].zoom = 0.0;
            self.slots[slot].views[idx].pan = Vec2::ZERO;
        }
        if let Some((dx, dy)) = wl_delta {
            self.window_width = (self.window_width * (1.0 + dx * 0.005)).clamp(1.0, 30000.0);
            self.window_center += dy * self.window_width * 0.002;
        }
        if let Some(c) = new_cursor {
            self.set_cursor(slot, c, idx);
        }
    }

    /// Set the crosshair of `slot` (voxel coords), sync its other two views,
    /// and — when study linking is on — propagate the same patient-space
    /// point to the other study.
    fn set_cursor(&mut self, slot: usize, c: [f64; 3], source_view: usize) {
        let Some(study) = &self.slots[slot].study else { return };
        let dims = study.volume.dims;
        let clamped = [
            c[0].clamp(0.0, dims[0] as f64 - 1.0),
            c[1].clamp(0.0, dims[1] as f64 - 1.0),
            c[2].clamp(0.0, dims[2] as f64 - 1.0),
        ];
        let patient = study
            .volume
            .voxel_to_patient(clamped[0], clamped[1], clamped[2]);
        self.slots[slot].cursor = clamped;
        self.sync_views_to_cursor(slot, Some(source_view));

        if self.link_studies {
            let other = 1 - slot;
            let Some(ostudy) = &self.slots[other].study else { return };
            // Map through the registration transform when one exists.
            // The transform maps fixed-slot patient coordinates into the
            // moving slot; clicks on the moving study use the inverse.
            let target = match &self.registration {
                Some(reg) if slot == reg.fixed_slot => reg.result.transform.map(patient),
                Some(reg) => reg.result.transform.unmap(patient),
                None => patient,
            };
            let odims = ostudy.volume.dims;
            let oc = ostudy.volume.patient_to_voxel(target);
            self.slots[other].cursor = [
                oc[0].clamp(0.0, odims[0] as f64 - 1.0),
                oc[1].clamp(0.0, odims[1] as f64 - 1.0),
                oc[2].clamp(0.0, odims[2] as f64 - 1.0),
            ];
            self.sync_views_to_cursor(other, None);
        }
    }

    /// Update slice indices of a slot's views to follow its cursor
    /// (skipping the view the user is interacting with, if any).
    fn sync_views_to_cursor(&mut self, slot: usize, skip_view: Option<usize>) {
        let Some(study) = &self.slots[slot].study else { return };
        let cursor = self.slots[slot].cursor;
        let mut new_slices = [None; 3];
        for i in 0..3 {
            if skip_view == Some(i) {
                continue;
            }
            let pl = self.slots[slot].views[i].plane;
            let sc = match pl {
                ViewPlane::Axial => cursor[2],
                ViewPlane::Sagittal => cursor[0],
                ViewPlane::Coronal => cursor[1],
            };
            let max = study.volume.plane_slice_count(pl).saturating_sub(1);
            new_slices[i] = Some((sc.round().max(0.0) as usize).min(max));
        }
        for i in 0..3 {
            if let Some(s) = new_slices[i] {
                self.slots[slot].views[i].slice = s;
            }
        }
    }

    /// Rebuild per-view textures & cached geometry when their inputs changed.
    fn refresh_view_caches(&mut self, ctx: &egui::Context, slot: usize, idx: usize) {
        if self.slots[slot].study.is_none() {
            return;
        }
        // Pre-compute hashes that need `&self` before borrowing mutably.
        let dose_hash = self.dose_settings_hash(slot);
        let contour_hash = self.contour_settings_hash(slot);
        let wc = self.window_center;
        let ww = self.window_width;
        let dose_on = self.dose_mode != DoseMode::Off;
        let contours_on = self.show_contours;

        let StudySlot { study, views, roi_visible, active_dose, dose_reference, .. } =
            &mut self.slots[slot];
        let study = study.as_ref().unwrap();
        let vol = &study.volume;
        let plane = views[idx].plane;
        let n_slices = vol.plane_slice_count(plane);
        if views[idx].slice >= n_slices {
            views[idx].slice = n_slices.saturating_sub(1);
        }
        let slice = views[idx].slice;
        let [w, h] = vol.plane_dims(plane);

        // Grayscale image.
        let img_key = (slice, wc.to_bits(), ww.to_bits());
        if views[idx].img_key != Some(img_key) {
            let view = &mut views[idx];
            let mut slice_buf = std::mem::take(&mut view.slice_buf);
            let mut gray_buf = std::mem::take(&mut view.gray_buf);
            vol.extract_slice(plane, slice, &mut slice_buf);
            render::slice_to_gray(&slice_buf, wc, ww, &mut gray_buf);
            let img = ColorImage::new([w, h], gray_buf.clone());
            match &mut view.tex {
                Some(t) => t.set(img, TextureOptions::LINEAR),
                None => {
                    view.tex = Some(ctx.load_texture(
                        format!("img{slot}_{idx}"),
                        img,
                        TextureOptions::LINEAR,
                    ))
                }
            }
            view.slice_buf = slice_buf;
            view.gray_buf = gray_buf;
            view.img_key = Some(img_key);
        }

        // Dose overlay + isodose segments.
        if dose_on && !study.doses.is_empty() {
            let dose_key =
                dose_hash.wrapping_add((slice as u64).wrapping_mul(0x9E3779B97F4A7C15));
            if views[idx].dose_key != Some(dose_key) {
                let dose = &study.doses[(*active_dose).min(study.doses.len() - 1)];
                let reference = *dose_reference;
                let view = &mut views[idx];
                let mut dose_plane = std::mem::take(&mut view.dose_plane);
                let mut dose_rgba = std::mem::take(&mut view.dose_rgba);
                render::sample_dose_plane(vol, dose, plane, slice, &mut dose_plane);
                render::dose_colorwash(
                    &dose_plane,
                    reference,
                    self.dose_threshold_pct / 100.0,
                    self.dose_opacity,
                    &mut dose_rgba,
                );
                let img = ColorImage::new([w, h], dose_rgba.clone());
                match &mut view.dose_tex {
                    Some(t) => t.set(img, TextureOptions::LINEAR),
                    None => {
                        view.dose_tex = Some(ctx.load_texture(
                            format!("dose{slot}_{idx}"),
                            img,
                            TextureOptions::LINEAR,
                        ))
                    }
                }
                // Isodose segments.
                view.iso_segs.clear();
                for (li, level) in self.iso_levels.iter().enumerate() {
                    if !level.on {
                        continue;
                    }
                    let abs = level.pct / 100.0 * reference;
                    if abs <= 0.0 {
                        continue;
                    }
                    for seg in render::marching_squares(&dose_plane, w, h, abs) {
                        view.iso_segs.push((li, seg));
                    }
                }
                view.dose_plane = dose_plane;
                view.dose_rgba = dose_rgba;
                view.dose_key = Some(dose_key);
            }
        }

        // Contours.
        if contours_on {
            if let Some(ss) = &study.structures {
                let ckey =
                    contour_hash.wrapping_add((slice as u64).wrapping_mul(0x517CC1B727220A95));
                if views[idx].contour_key != Some(ckey) {
                    let mut contours = Vec::new();
                    for (ri, roi) in ss.rois.iter().enumerate() {
                        if !roi_visible.get(ri).copied().unwrap_or(false) {
                            continue;
                        }
                        let gfx = render::roi_on_plane(vol, roi, plane, slice);
                        if !gfx.polylines.is_empty()
                            || !gfx.segments.is_empty()
                            || !gfx.points.is_empty()
                        {
                            contours.push((ri, gfx));
                        }
                    }
                    views[idx].contours = contours;
                    views[idx].contour_key = Some(ckey);
                }
            }
        }

        self.refresh_fusion_cache(ctx, slot, idx);
    }

    /// Rebuild the magenta/green fusion texture of a fixed-study view: the
    /// fixed image stays gray in R/B, the transformed moving image is blended
    /// into the green channel. Aligned anatomy therefore reads gray,
    /// mismatches magenta/green. Drawn on whichever slot was the fixed image
    /// of the active registration.
    fn refresh_fusion_cache(&mut self, ctx: &egui::Context, slot: usize, idx: usize) {
        if !self.fusion_on {
            return;
        }
        let Some(reg) = &self.registration else { return };
        if reg.fixed_slot != slot {
            return;
        }
        if self.slots[0].study.is_none() || self.slots[1].study.is_none() {
            return;
        }
        let transform: Arc<Transform3> = reg.result.transform.clone();
        let fixed_slot = reg.fixed_slot;
        let wc = self.window_center;
        let ww = self.window_width.max(1.0);
        let weight = self.fusion_weight.clamp(0.0, 1.0);

        let (left, right) = self.slots.split_at_mut(1);
        let (a, bvol) = if fixed_slot == 0 {
            let bvol = &right[0].study.as_ref().unwrap().volume;
            (&mut left[0], bvol)
        } else {
            let bvol = &left[0].study.as_ref().unwrap().volume;
            (&mut right[0], bvol)
        };
        let avol = &a.study.as_ref().unwrap().volume;
        let view = &mut a.views[idx];
        let plane = view.plane;
        let slice = view.slice;
        let [w, h] = avol.plane_dims(plane);

        let mut key: u64 = 0x243F6A8885A308D3 ^ self.reg_gen.wrapping_mul(0x9E3779B97F4A7C15);
        for v in [
            slice as u64,
            wc.to_bits() as u64,
            ww.to_bits() as u64,
            weight.to_bits() as u64,
            self.settings_gen,
        ] {
            key ^= v;
            key = key.wrapping_mul(0x100000001b3);
        }
        if view.fusion_key == Some(key) {
            return;
        }

        // Ensure the A slice buffer matches the current slice.
        if view.slice_buf.len() != w * h {
            avol.extract_slice(plane, slice, &mut view.slice_buf);
        }
        let slice_buf = &view.slice_buf;

        let lo = wc - ww * 0.5;
        let scale = 255.0 / ww;
        let wl = |v: f32| -> f32 { ((v - lo) * scale).clamp(0.0, 255.0) };

        let mut pixels = vec![Color32::BLACK; w * h];
        pixels.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
            for (px, out) in row.iter_mut().enumerate() {
                let a_gray = wl(slice_buf[py * w + px] as f32);
                let vxl = avol.plane_pixel_to_voxel(plane, slice, px as f64, py as f64);
                let p_fixed = avol.voxel_to_patient(vxl[0], vxl[1], vxl[2]);
                let q = transform.map(p_fixed);
                let b_gray = bvol.sample_patient(q).map(&wl).unwrap_or(0.0);
                let g = a_gray + (b_gray - a_gray) * weight;
                *out = Color32::from_rgb(a_gray as u8, g as u8, a_gray as u8);
            }
        });

        let img = ColorImage::new([w, h], pixels);
        match &mut view.fusion_tex {
            Some(t) => t.set(img, TextureOptions::LINEAR),
            None => {
                view.fusion_tex = Some(ctx.load_texture(
                    format!("fusion{fixed_slot}_{idx}"),
                    img,
                    TextureOptions::LINEAR,
                ))
            }
        }
        view.fusion_key = Some(key);
    }

    // -- Modals -----------------------------------------------------------

    fn modals(&mut self, ctx: &egui::Context) {
        if let Some(err) = self.error.clone() {
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(&err);
                    ui.add_space(8.0);
                    if ui.button("OK").clicked() {
                        self.error = None;
                    }
                });
        }
    }
}
