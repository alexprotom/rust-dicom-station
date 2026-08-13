//! The egui application: three linked MPR views in a row, structure/dose/plan
//! panels, and all interaction logic.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use egui::{
    Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, TextureHandle,
    TextureOptions, Vec2,
};

use crate::loader::{self, LoadedStudy, Progress};
use crate::render;
use crate::volume::{ViewPlane, Volume};

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

const WL_PRESETS: &[(&str, f32, f32)] = &[
    ("Soft tissue", 40.0, 400.0),
    ("Lung", -600.0, 1500.0),
    ("Bone", 300.0, 1500.0),
    ("Brain", 40.0, 80.0),
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
        }
    }

    fn invalidate(&mut self) {
        self.img_key = None;
        self.dose_key = None;
        self.contour_key = None;
    }
}

// ---------------------------------------------------------------------------
// Background loading
// ---------------------------------------------------------------------------

enum LoadResult {
    Study(Box<anyhow::Result<LoadedStudy>>),
    Volume(Box<anyhow::Result<(Volume, (f32, f32), Vec<String>)>>, usize),
}

struct LoadJob {
    progress: Arc<Progress>,
    rx: mpsc::Receiver<LoadResult>,
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

pub struct ViewerApp {
    study: Option<LoadedStudy>,
    loading: Option<LoadJob>,
    error: Option<String>,

    views: [ViewState; 3],
    cursor: [f64; 3], // fractional voxel coords of the linked crosshair

    window_center: f32,
    window_width: f32,

    roi_visible: Vec<bool>,
    show_contours: bool,
    show_crosshair: bool,
    show_labels: bool,
    show_isocenters: bool,

    active_dose: usize,
    dose_mode: DoseMode,
    dose_opacity: f32,
    dose_threshold_pct: f32,
    dose_reference: f32,
    iso_levels: Vec<IsoLevel>,

    /// Bumped whenever ROI visibility / dose settings change → cache rebuild.
    settings_gen: u64,
}

impl ViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial_path: Option<PathBuf>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let mut app = ViewerApp {
            study: None,
            loading: None,
            error: None,
            views: [
                ViewState::new(ViewPlane::Axial),
                ViewState::new(ViewPlane::Sagittal),
                ViewState::new(ViewPlane::Coronal),
            ],
            cursor: [0.0; 3],
            window_center: 40.0,
            window_width: 400.0,
            roi_visible: Vec::new(),
            show_contours: true,
            show_crosshair: true,
            show_labels: true,
            show_isocenters: true,
            active_dose: 0,
            dose_mode: DoseMode::Off,
            dose_opacity: 0.45,
            dose_threshold_pct: 15.0,
            dose_reference: 1.0,
            iso_levels: default_iso_levels(),
            settings_gen: 0,
        };
        if let Some(p) = initial_path {
            app.start_load(p);
        }
        app
    }

    fn start_load(&mut self, path: PathBuf) {
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = loader::load_directory(&path, &p2);
            let _ = tx.send(LoadResult::Study(Box::new(res)));
        });
        self.loading = Some(LoadJob { progress, rx });
    }

    fn start_series_switch(&mut self, idx: usize) {
        let Some(study) = &self.study else { return };
        let series = study.series[idx].clone();
        let progress = Arc::new(Progress::default());
        let (tx, rx) = mpsc::channel();
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let res = loader::load_series_volume(&series, &p2);
            let _ = tx.send(LoadResult::Volume(Box::new(res), idx));
        });
        self.loading = Some(LoadJob { progress, rx });
    }

    fn on_study_loaded(&mut self, study: LoadedStudy) {
        self.window_center = study.default_window.0;
        self.window_width = study.default_window.1;
        self.roi_visible = study
            .structures
            .as_ref()
            .map(|s| vec![true; s.rois.len()])
            .unwrap_or_default();
        self.active_dose = 0;
        self.dose_mode = if study.doses.is_empty() { DoseMode::Off } else { DoseMode::Both };
        self.dose_reference = study
            .plans
            .iter()
            .find_map(|p| p.target_prescription_dose)
            .map(|d| d as f32)
            .or_else(|| study.doses.first().map(|d| d.max_dose))
            .unwrap_or(1.0);
        let dims = study.volume.dims;
        self.cursor = [
            dims[0] as f64 * 0.5,
            dims[1] as f64 * 0.5,
            dims[2] as f64 * 0.5,
        ];
        for v in &mut self.views {
            v.slice = match v.plane {
                ViewPlane::Axial => dims[2] / 2,
                ViewPlane::Sagittal => dims[0] / 2,
                ViewPlane::Coronal => dims[1] / 2,
            };
            v.zoom = 0.0;
            v.pan = Vec2::ZERO;
            v.invalidate();
        }
        self.study = Some(study);
        self.settings_gen += 1;
    }

    fn apply_new_volume(&mut self, vol: Volume, window: (f32, f32), idx: usize) {
        if let Some(study) = &mut self.study {
            study.volume = vol;
            study.active_series = idx;
            self.window_center = window.0;
            self.window_width = window.1;
            let dims = study.volume.dims;
            self.cursor = [
                dims[0] as f64 * 0.5,
                dims[1] as f64 * 0.5,
                dims[2] as f64 * 0.5,
            ];
            for v in &mut self.views {
                v.slice = study.volume.plane_slice_count(v.plane) / 2;
                v.zoom = 0.0;
                v.pan = Vec2::ZERO;
                v.invalidate();
            }
            self.settings_gen += 1;
        }
    }

    /// Combined hash of everything that affects dose overlays.
    fn dose_settings_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        };
        mix(self.active_dose as u64);
        mix(self.dose_mode as u64);
        mix(self.dose_opacity.to_bits() as u64);
        mix(self.dose_threshold_pct.to_bits() as u64);
        mix(self.dose_reference.to_bits() as u64);
        for l in &self.iso_levels {
            mix(l.pct.to_bits() as u64 | ((l.on as u64) << 40));
        }
        mix(self.settings_gen);
        h
    }

    fn contour_settings_hash(&self) -> u64 {
        let mut h: u64 = 0x9e3779b97f4a7c15;
        for (i, v) in self.roi_visible.iter().enumerate() {
            if *v {
                h = h.rotate_left(7) ^ (i as u64 + 1);
            }
        }
        h ^ self.settings_gen.wrapping_mul(0x2545F4914F6CDD1D)
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
                Ok(LoadResult::Study(res)) => {
                    self.loading = None;
                    match *res {
                        Ok(study) => self.on_study_loaded(study),
                        Err(e) => self.error = Some(format!("{e:#}")),
                    }
                }
                Ok(LoadResult::Volume(res, idx)) => {
                    self.loading = None;
                    match *res {
                        Ok((vol, window, warnings)) => {
                            self.apply_new_volume(vol, window, idx);
                            if let Some(study) = &mut self.study {
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

        self.top_bar(ui);
        self.side_panel(ui);
        self.status_bar(ui);
        self.central_views(ui);
        self.modals(&ctx);
    }
}

impl ViewerApp {
    // -- Top bar ----------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let mut switch_to: Option<usize> = None;
        egui::Panel::top(egui::Id::new("top_bar")).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 Open folder…").clicked() {
                    if let Some(dir) = rfd::FileDialog::new()
                        .set_title("Select a DICOM directory")
                        .pick_folder()
                    {
                        self.start_load(dir);
                    }
                }

                if let Some(study) = &self.study {
                    ui.separator();
                    // Series selector.
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
                    egui::ComboBox::from_id_salt("series_sel")
                        .width(220.0)
                        .selected_text(label(&study.series[active]))
                        .show_ui(ui, |ui| {
                            for (i, s) in study.series.iter().enumerate() {
                                ui.selectable_value(&mut selected, i, label(s));
                            }
                        });
                    if selected != active {
                        switch_to = Some(selected);
                    }

                    ui.separator();
                    ui.label("W/L:");
                    let mut changed = false;
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.window_center)
                                .speed(2.0)
                                .prefix("C "),
                        )
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut self.window_width)
                                .speed(4.0)
                                .range(1.0..=20000.0)
                                .prefix("W "),
                        )
                        .changed();
                    egui::ComboBox::from_id_salt("wl_preset")
                        .selected_text("Presets")
                        .width(90.0)
                        .show_ui(ui, |ui| {
                            for (name, c, w) in WL_PRESETS {
                                if ui.button(*name).clicked() {
                                    self.window_center = *c;
                                    self.window_width = *w;
                                    changed = true;
                                }
                            }
                            if ui.button("Full range").clicked() {
                                let v = &self.study.as_ref().unwrap().volume;
                                self.window_center =
                                    (v.min_value as f32 + v.max_value as f32) * 0.5;
                                self.window_width =
                                    (v.max_value as f32 - v.min_value as f32).max(1.0);
                                changed = true;
                            }
                        });
                    let _ = changed;

                    ui.separator();
                    ui.checkbox(&mut self.show_contours, "Contours");
                    ui.checkbox(&mut self.show_crosshair, "Crosshair");
                    ui.checkbox(&mut self.show_labels, "Labels");

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let m = &study.meta;
                        ui.label(
                            egui::RichText::new(format!(
                                "{}  {}  {}",
                                m.patient_name.replace('^', " "),
                                m.patient_id,
                                m.study_date
                            ))
                            .weak(),
                        )
                        .on_hover_text(if m.study_description.is_empty() {
                            "No study description".to_string()
                        } else {
                            m.study_description.clone()
                        });
                    });
                }
            });
        });
        if let Some(i) = switch_to {
            if self.loading.is_none() {
                self.start_series_switch(i);
            }
        }
    }

    // -- Side panel -------------------------------------------------------

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        if self.study.is_none() {
            return;
        }
        egui::Panel::left(egui::Id::new("side"))
            .resizable(true)
            .default_size(270.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.structures_section(ui);
                    self.dose_section(ui);
                    self.plan_section(ui);
                    self.warnings_section(ui);
                });
            });
    }

    fn structures_section(&mut self, ui: &mut egui::Ui) {
        let Some(study) = &self.study else { return };
        let Some(ss) = &study.structures else {
            ui.collapsing("Structures", |ui| {
                ui.weak("No RTSTRUCT loaded");
            });
            return;
        };
        let n_vis = self.roi_visible.iter().filter(|v| **v).count();
        let mut changed = false;
        egui::CollapsingHeader::new(format!("Structures ({}/{})", n_vis, ss.rois.len()))
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.small_button("All").clicked() {
                        self.roi_visible.iter_mut().for_each(|v| *v = true);
                        changed = true;
                    }
                    if ui.small_button("None").clicked() {
                        self.roi_visible.iter_mut().for_each(|v| *v = false);
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
                            &mut self.roi_visible[i],
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
        if changed {
            self.settings_gen += 1;
        }
    }

    fn dose_section(&mut self, ui: &mut egui::Ui) {
        let Some(study) = &self.study else { return };
        if study.doses.is_empty() {
            ui.collapsing("Dose", |ui| {
                ui.weak("No RTDOSE loaded");
            });
            return;
        }
        let doses = &study.doses;
        egui::CollapsingHeader::new("Dose")
            .default_open(true)
            .show(ui, |ui| {
                if doses.len() > 1 {
                    let mut sel = self.active_dose;
                    egui::ComboBox::from_id_salt("dose_sel")
                        .width(230.0)
                        .selected_text(&doses[sel.min(doses.len() - 1)].label)
                        .show_ui(ui, |ui| {
                            for (i, d) in doses.iter().enumerate() {
                                ui.selectable_value(&mut sel, i, &d.label);
                            }
                        });
                    if sel != self.active_dose {
                        self.active_dose = sel;
                    }
                }
                let d = &doses[self.active_dose.min(doses.len() - 1)];
                ui.weak(format!(
                    "{}  max {:.2} {}",
                    d.summation_type,
                    d.max_dose,
                    d.units.to_lowercase()
                ));

                let mut mode = self.dose_mode;
                egui::ComboBox::from_id_salt("dose_mode")
                    .selected_text(mode.label())
                    .show_ui(ui, |ui| {
                        for m in [DoseMode::Off, DoseMode::Colorwash, DoseMode::Isodose, DoseMode::Both]
                        {
                            ui.selectable_value(&mut mode, m, m.label());
                        }
                    });
                self.dose_mode = mode;

                ui.horizontal(|ui| {
                    ui.label("Reference");
                    ui.add(
                        egui::DragValue::new(&mut self.dose_reference)
                            .speed(0.05)
                            .range(0.01..=1000.0)
                            .suffix(" Gy"),
                    );
                    if ui.small_button("max").clicked() {
                        self.dose_reference = d.max_dose;
                    }
                });
                ui.add(
                    egui::Slider::new(&mut self.dose_opacity, 0.0..=1.0).text("Opacity"),
                );
                ui.add(
                    egui::Slider::new(&mut self.dose_threshold_pct, 0.0..=100.0)
                        .text("Threshold %"),
                );

                ui.label("Isodose levels (% of reference):");
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

    fn plan_section(&mut self, ui: &mut egui::Ui) {
        let Some(study) = &self.study else { return };
        if study.plans.is_empty() {
            ui.collapsing("Plan", |ui| {
                ui.weak("No RTPLAN loaded");
            });
            return;
        }
        for (pi, plan) in study.plans.iter().enumerate() {
            egui::CollapsingHeader::new(format!(
                "Plan: {}",
                if plan.label.is_empty() { "unnamed" } else { &plan.label }
            ))
            .id_salt(("plan", pi))
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
                ui.checkbox(&mut self.show_isocenters, "Show isocenters");
                if !plan.beams.is_empty() {
                    egui::Grid::new(("beam_grid", pi))
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

    fn warnings_section(&mut self, ui: &mut egui::Ui) {
        let Some(study) = &self.study else { return };
        if study.warnings.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(
            egui::RichText::new(format!("⚠ Warnings ({})", study.warnings.len()))
                .color(Color32::from_rgb(240, 190, 60)),
        )
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
                if let Some(study) = &self.study {
                    let v = &study.volume;
                    let c = self.cursor;
                    let p = v.voxel_to_patient(c[0], c[1], c[2]);
                    ui.monospace(format!(
                        "xyz: ({:7.1}, {:7.1}, {:7.1}) mm   ijk: ({:4}, {:4}, {:4})",
                        p.x,
                        p.y,
                        p.z,
                        c[0].round() as i64,
                        c[1].round() as i64,
                        c[2].round() as i64
                    ));
                    if let Some(hu) =
                        v.get(c[0].round() as i64, c[1].round() as i64, c[2].round() as i64)
                    {
                        ui.monospace(format!("value: {hu:5}"));
                    }
                    if let Some(d) = study
                        .doses
                        .get(self.active_dose)
                        .and_then(|d| d.sample(p))
                    {
                        ui.monospace(format!(
                            "dose: {:.2} Gy ({:.0}%)",
                            d,
                            100.0 * d / self.dose_reference.max(1e-6)
                        ));
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak("LMB crosshair · RMB W/L · MMB pan · wheel slice · Ctrl+wheel zoom · double-click reset");
                    });
                } else {
                    ui.weak("No data loaded");
                }
            });
        });
    }

    // -- Central: three views in a row ------------------------------------

    fn central_views(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default_margins()
            .frame(egui::Frame::NONE.fill(Color32::from_gray(10)))
            .show(ui, |ui| {
                if self.study.is_none() {
                    self.empty_state(ui);
                    return;
                }
                let full = ui.available_rect_before_wrap();
                let gap = 4.0;
                let slider_h = 26.0;
                let col_w = (full.width() - 2.0 * gap) / 3.0;
                for idx in 0..3 {
                    let x0 = full.left() + idx as f32 * (col_w + gap);
                    let col =
                        Rect::from_min_size(Pos2::new(x0, full.top()), Vec2::new(col_w, full.height()));
                    let view_rect = Rect::from_min_max(
                        col.min,
                        Pos2::new(col.max.x, col.max.y - slider_h),
                    );
                    let slider_rect = Rect::from_min_max(
                        Pos2::new(col.min.x + 6.0, col.max.y - slider_h + 2.0),
                        Pos2::new(col.max.x - 6.0, col.max.y - 2.0),
                    );
                    self.one_view(ui, idx, view_rect);
                    // Slice slider under the view.
                    let max_slice = self
                        .study
                        .as_ref()
                        .map(|s| s.volume.plane_slice_count(self.views[idx].plane).saturating_sub(1))
                        .unwrap_or(0);
                    if max_slice > 0 {
                        let mut slice = self.views[idx].slice.min(max_slice);
                        let resp = ui.put(
                            slider_rect,
                            egui::Slider::new(&mut slice, 0..=max_slice).show_value(false),
                        );
                        if resp.changed() {
                            self.views[idx].slice = slice;
                        }
                    }
                }
            });
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
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            self.start_load(dir);
                        }
                    }
                }
            });
        });
    }

    // -- One viewport -----------------------------------------------------

    fn one_view(&mut self, ui: &mut egui::Ui, idx: usize, rect: Rect) {
        let ctx = ui.ctx().clone();
        let plane = self.views[idx].plane;

        // ---- cache refresh (image, dose, contours) ----
        self.refresh_view_caches(&ctx, idx);

        let Some(study) = &self.study else { return };
        let vol = &study.volume;
        let view = &self.views[idx];

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

        if let Some(tex) = &view.tex {
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
            let cp = vol.voxel_to_plane_pixel(plane, self.cursor);
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
            painter.text(
                rect.left_top() + Vec2::new(6.0, 4.0),
                Align2::LEFT_TOP,
                plane.title(),
                FontId::proportional(14.0),
                Color32::from_rgb(255, 170, 60),
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
            if idx == 1 {
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
        let resp = ui.interact(rect, egui::Id::new(("viewport", idx)), Sense::click_and_drag());
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

        // Apply interactions (mutable phase).
        if let Some(a) = new_accum {
            self.views[idx].scroll_accum = a;
        }
        if let Some(s) = new_slice {
            self.views[idx].slice = s;
        }
        if let Some(z) = new_zoom {
            self.views[idx].zoom = z;
        }
        if let Some(p) = new_pan {
            self.views[idx].pan = p;
        }
        if reset_view {
            self.views[idx].zoom = 0.0;
            self.views[idx].pan = Vec2::ZERO;
        }
        if let Some((dx, dy)) = wl_delta {
            self.window_width = (self.window_width * (1.0 + dx * 0.005)).clamp(1.0, 30000.0);
            self.window_center += dy * self.window_width * 0.002;
        }
        if let Some(c) = new_cursor {
            let dims = self.study.as_ref().unwrap().volume.dims;
            self.cursor = [
                c[0].clamp(0.0, dims[0] as f64 - 1.0),
                c[1].clamp(0.0, dims[1] as f64 - 1.0),
                c[2].clamp(0.0, dims[2] as f64 - 1.0),
            ];
            // Link the other two views to the crosshair.
            for i in 0..3 {
                if i == idx {
                    continue;
                }
                let pl = self.views[i].plane;
                let sc = match pl {
                    ViewPlane::Axial => self.cursor[2],
                    ViewPlane::Sagittal => self.cursor[0],
                    ViewPlane::Coronal => self.cursor[1],
                };
                let max = self
                    .study
                    .as_ref()
                    .unwrap()
                    .volume
                    .plane_slice_count(pl)
                    .saturating_sub(1);
                self.views[i].slice = (sc.round().max(0.0) as usize).min(max);
            }
        }
    }

    /// Rebuild per-view textures & cached geometry when their inputs changed.
    fn refresh_view_caches(&mut self, ctx: &egui::Context, idx: usize) {
        let Some(study) = &self.study else { return };
        let vol = &study.volume;
        let plane = self.views[idx].plane;
        let n_slices = vol.plane_slice_count(plane);
        if self.views[idx].slice >= n_slices {
            self.views[idx].slice = n_slices.saturating_sub(1);
        }
        let slice = self.views[idx].slice;
        let [w, h] = vol.plane_dims(plane);

        // Grayscale image.
        let img_key = (
            slice,
            self.window_center.to_bits(),
            self.window_width.to_bits(),
        );
        if self.views[idx].img_key != Some(img_key) {
            let view = &mut self.views[idx];
            let mut slice_buf = std::mem::take(&mut view.slice_buf);
            let mut gray_buf = std::mem::take(&mut view.gray_buf);
            vol.extract_slice(plane, slice, &mut slice_buf);
            render::slice_to_gray(&slice_buf, self.window_center, self.window_width, &mut gray_buf);
            let img = ColorImage::new([w, h], gray_buf.clone());
            match &mut view.tex {
                Some(t) => t.set(img, TextureOptions::LINEAR),
                None => {
                    view.tex = Some(ctx.load_texture(
                        format!("img{idx}"),
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
        if self.dose_mode != DoseMode::Off && !study.doses.is_empty() {
            let dose_key = self
                .dose_settings_hash()
                .wrapping_add((slice as u64).wrapping_mul(0x9E3779B97F4A7C15));
            if self.views[idx].dose_key != Some(dose_key) {
                let dose = &study.doses[self.active_dose.min(study.doses.len() - 1)];
                let view = &mut self.views[idx];
                let mut dose_plane = std::mem::take(&mut view.dose_plane);
                let mut dose_rgba = std::mem::take(&mut view.dose_rgba);
                render::sample_dose_plane(vol, dose, plane, slice, &mut dose_plane);
                render::dose_colorwash(
                    &dose_plane,
                    self.dose_reference,
                    self.dose_threshold_pct / 100.0,
                    self.dose_opacity,
                    &mut dose_rgba,
                );
                let img = ColorImage::new([w, h], dose_rgba.clone());
                match &mut view.dose_tex {
                    Some(t) => t.set(img, TextureOptions::LINEAR),
                    None => {
                        view.dose_tex = Some(ctx.load_texture(
                            format!("dose{idx}"),
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
                    let abs = level.pct / 100.0 * self.dose_reference;
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
        if self.show_contours {
            if let Some(ss) = &study.structures {
                let ckey = self
                    .contour_settings_hash()
                    .wrapping_add((slice as u64).wrapping_mul(0x517CC1B727220A95));
                if self.views[idx].contour_key != Some(ckey) {
                    let mut contours = Vec::new();
                    for (ri, roi) in ss.rois.iter().enumerate() {
                        if !self.roi_visible.get(ri).copied().unwrap_or(false) {
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
                    let view = &mut self.views[idx];
                    view.contours = contours;
                    view.contour_key = Some(ckey);
                }
            }
        }
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
