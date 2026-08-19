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

use crate::anonymize;
use crate::dicom_export;
use crate::extras;
use crate::gen_test_data::{self, GenParams};
use crate::loader::{self, LoadedStudy, Progress};
use crate::mesh3d::{self, GridGeom, RoiMesh};
use crate::registration::{
    self, RegKind, RegParams, RegProgress, RegistrationResult, Transform3,
};
use crate::render;
use crate::segmentation::{self, GrowState, Segmentation};
use crate::settings::{self, Settings};
use crate::simulate::{self, SimParams};
use crate::volume::{ViewPlane, Volume};

const SLOT_NAMES: [&str; 2] = ["A", "B"];

// ---------------------------------------------------------------------------
// Theme-dependent colors
//
// The image viewports stay black in both themes — that is the convention in
// clinical viewers, keeps grayscale windowing and the dose colorwash reading
// correctly, and lets the overlay annotations use one fixed palette. Only the
// surrounding chrome and the few hand-painted accents follow the theme.
// ---------------------------------------------------------------------------

/// Fill of the area around and between the viewports.
fn backdrop_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_gray(10)
    } else {
        Color32::from_gray(190)
    }
}

/// Fill of an empty study row (slightly lifted off the backdrop).
fn empty_row_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_gray(14)
    } else {
        Color32::from_gray(205)
    }
}

/// Amber accent for warnings — darkened in light mode, where pale yellow on
/// white is unreadable.
fn warn_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(240, 190, 60)
    } else {
        Color32::from_rgb(146, 98, 0)
    }
}

/// Red-orange accent for values needing attention (e.g. an abnormal
/// treatment termination status).
fn alert_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(240, 120, 60)
    } else {
        Color32::from_rgb(176, 56, 8)
    }
}

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
// Interactive segmentation tools
// ---------------------------------------------------------------------------

/// Active viewport tool. `None` keeps the classic behavior (LMB navigates
/// the crosshair); the segmentation tools take over the left mouse button.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SegTool {
    None,
    /// Paint into the active segmentation (Alt temporarily erases).
    Brush,
    /// Erase from the active segmentation.
    Erase,
    /// Seeded region growing: press to seed, drag up/down to widen/narrow
    /// the intensity tolerance, release to commit (Esc cancels).
    Grow,
}

/// An in-progress region-growing drag.
struct GrowDrag {
    slot: usize,
    /// Drag level: multiplier on the base geodesic reach (1.0 at press).
    level: f32,
    /// Screen y at drag start (level 1.0).
    y0: f32,
    /// The last computed region hit the voxel cap.
    capped: bool,
}

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
    dose_plane: Vec<f32>,
    iso_segs: Vec<(usize, render::Segment)>,
    contours: Vec<(usize, render::RoiPlaneGraphics)>,
    fusion_tex: Option<TextureHandle>,
    fusion_key: Option<u64>,
    seg_tex: Option<TextureHandle>,
    seg_key: Option<u64>,
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
            dose_plane: Vec::new(),
            iso_segs: Vec::new(),
            contours: Vec::new(),
            fusion_tex: None,
            fusion_key: None,
            seg_tex: None,
            seg_key: None,
        }
    }

    fn invalidate(&mut self) {
        self.img_key = None;
        self.dose_key = None;
        self.contour_key = None;
        self.fusion_key = None;
        self.seg_key = None;
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
    /// Index of the active structure set within `study.structure_sets`.
    active_structs: usize,
    active_dose: usize,
    dose_reference: f32,
    /// Painted segmentations on this slot's volume.
    segs: Vec<Segmentation>,
    /// Index of the segmentation the tools edit.
    active_seg: usize,
}

impl StudySlot {
    /// The currently selected structure set of this slot, if any.
    fn active_structures(&self) -> Option<&crate::rtstruct::StructureSet> {
        self.study
            .as_ref()
            .and_then(|s| s.structure_sets.get(self.active_structs))
    }
}

impl StudySlot {
    fn empty() -> Self {
        StudySlot {
            study: None,
            views: fresh_views(),
            cursor: [0.0; 3],
            roi_visible: Vec::new(),
            active_structs: 0,
            active_dose: 0,
            dose_reference: 1.0,
            segs: Vec::new(),
            active_seg: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Background loading
// ---------------------------------------------------------------------------

/// A freshly reconstructed volume: pixels, its default window/level and any
/// non-fatal notes raised while reading the series.
type LoadedVolume = (Volume, (f32, f32), Vec<String>);

enum LoadResult {
    /// A whole folder, for the given slot.
    Study(Box<anyhow::Result<LoadedStudy>>, usize),
    /// A single series switched into (slot, series index).
    Volume(Box<anyhow::Result<LoadedVolume>>, usize, usize),
}

/// A unit of work running on a background thread: a shared progress handle
/// plus the channel its result arrives on. Every background feature in the
/// app has this shape, and [`poll_job`] drives them all identically.
struct Job<T, P = Progress> {
    progress: Arc<P>,
    rx: mpsc::Receiver<T>,
}

/// Poll a background job. Returns its result once, clearing the slot; reports
/// a worker that died without answering into `error`; otherwise schedules the
/// next poll and returns `None`.
fn poll_job<T, P>(
    slot: &mut Option<Job<T, P>>,
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

/// A floating 3D structure-rendering window (one per study slot).
struct D3Window {
    slot: usize,
    open: bool,
    yaw: f32,
    pitch: f32,
    /// Zoom multiplier on the auto-fit scale.
    zoom: f32,
    pan: Vec2,
    opacity: f32,
    meshes: Option<Arc<Vec<RoiMesh>>>,
    /// Scene bounding-sphere (patient mm) for auto-fit.
    center: [f32; 3],
    radius: f32,
    /// Identity of the structure set the meshes were built from.
    key: u64,
    job: Option<Job<Vec<RoiMesh>>>,
    /// Live meshes of the painted segmentations (`roi_index` = seg index).
    seg_meshes: Option<Arc<Vec<RoiMesh>>>,
    seg_job: Option<Job<Vec<RoiMesh>>>,
    /// Hash of the segmentation state `seg_meshes` was built from.
    seg_built: u64,
    /// Cached projected geometry for the current camera.
    frame: D3Frame,
}


/// Cached triangle soup of a 3D window.
///
/// egui repaints on every pointer move, and projecting + depth-sorting a few
/// hundred thousand triangles takes several milliseconds, so the soup is
/// rebuilt only when something it actually depends on changes. The draw
/// order depends on orientation and visibility alone, so panning and zooming
/// reuse the existing sort.
#[derive(Default)]
struct D3Frame {
    /// Identity of the current depth sort (orientation + visibility + meshes).
    order_key: Option<u64>,
    /// Identity of the current projected vertices (also zoom / pan / size).
    vertex_key: Option<u64>,
    mesh: Arc<egui::epaint::Mesh>,
    /// Triangles in scene order, indexed by the sort below.
    tris: Vec<[u32; 3]>,
    /// View-space depth per vertex.
    depth: Vec<f32>,
    /// `(monotone depth key) << 32 | triangle slot`, sorted far-to-near.
    order: Vec<u64>,
}

/// Map an `f32` to a `u32` that sorts in the same order, so the painter's
/// algorithm can use a primitive sort instead of a float comparator.
#[inline]
fn depth_key(d: f32) -> u32 {
    let b = d.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b ^ 0x8000_0000
    }
}

#[inline]
fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x100000001b3)
}

/// What a right-click action on the data tree selects.
#[derive(Clone)]
enum TreeSel {
    /// All series of one patient (grouped by `SeriesInfo::patient_key`).
    Patient(String),
    /// All series of one study (StudyInstanceUID).
    Study(String),
    /// A single series (index into `LoadedStudy::series`).
    Series(usize),
}

/// What to do with the selection.
#[derive(Clone, Copy, PartialEq)]
enum TreeOp {
    Copy,
    Move,
    Remove,
}

/// Right-click action on the data tree.
#[derive(Clone)]
struct TreeAction {
    from: usize,
    sel: TreeSel,
    op: TreeOp,
}

/// Which parts of a `LoadedStudy` a tree selection covers: the selected
/// series plus the RT objects linked to them via the DICOM reference chain
/// (RTSTRUCT ▶ series, RTPLAN ▶ RTSTRUCT, RTDOSE ▶ RTPLAN).
struct SubsetMasks {
    series: Vec<bool>,
    structs: Vec<bool>,
    doses: Vec<bool>,
    plans: Vec<bool>,
    /// Planar images / REG objects / treatment records are only carried when
    /// the selection covers the whole slot content.
    take_extras: bool,
}

/// A floating viewer window for a planar image (DX / CR / RTIMAGE).
struct PlanarWindow {
    slot: usize,
    idx: usize,
    open: bool,
    wl: (f32, f32),
    tex: Option<TextureHandle>,
    tex_wl: (f32, f32),
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

    loading: Option<Job<LoadResult>>,
    /// A load queued behind the one in flight (slot, directory).
    pending_load: Option<(usize, PathBuf)>,
    error: Option<String>,

    // Registration (direction selectable: either study can be the fixed one).
    registration: Option<ActiveRegistration>,
    /// The payload carries the slot that was used as the fixed image.
    reg_job: Option<Job<(usize, anyhow::Result<RegistrationResult>), RegProgress>>,
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
    sim_job: Option<Job<(usize, LoadedStudy)>>,
    last_sim: Option<String>,
    export_job: Option<Job<anyhow::Result<(usize, String)>>>,
    export_result: Option<String>,

    // Built-in synthetic test-data generator.
    /// Dialog visibility.
    gen_open: bool,
    gen_params: GenParams,
    /// Output folder as edited in the dialog (defaults to the app folder).
    gen_dir: String,
    gen_job: Option<Job<anyhow::Result<(usize, PathBuf)>>>,
    gen_result: Option<String>,
    /// Load the generated study into slot A once it has been written.
    gen_load_after: bool,

    // Tools ▶ Anonymize DICOM folder.
    anon_open: bool,
    /// Input folder as edited in the dialog.
    anon_dir: String,
    /// Output folder (ignored when `anon_in_place`).
    anon_out: String,
    anon_in_place: bool,
    anon_remove_private: bool,
    anon_remap_uids: bool,
    anon_mark: bool,
    /// Last scan result; findings are edited in place by the table.
    anon_scan: Option<anonymize::ScanResult>,
    anon_scan_job: Option<Job<anyhow::Result<anonymize::ScanResult>>>,
    anon_apply_job: Option<Job<anyhow::Result<usize>>>,
    anon_result: Option<String>,

    /// Open floating viewers for planar images.
    planar_windows: Vec<PlanarWindow>,
    /// Open 3D structure-rendering windows.
    d3_windows: Vec<D3Window>,
    /// Deferred right-click action from the study tree.
    tree_action: Option<TreeAction>,
    /// When set, this single (slot, view) fills the whole central area.
    maximized: Option<(usize, usize)>,
    /// Invert REG matrices before applying them as the active registration.
    reg_apply_invert: bool,

    window_center: f32,
    window_width: f32,

    show_contours: bool,
    show_crosshair: bool,
    show_labels: bool,
    show_isocenters: bool,

    // Interactive segmentation.
    seg_tool: SegTool,
    /// Brush radius in mm (shared by paint and erase).
    brush_radius_mm: f32,
    /// Spherical 3D brush (paints across slices) vs. in-plane 2D circle.
    brush_3d: bool,
    /// Last brush sample of the stroke in progress: (slot, voxel coords).
    paint_last: Option<(usize, [f64; 3])>,
    /// Region-growing drag in progress.
    grow: Option<GrowDrag>,
    grow_state: GrowState,
    /// Full-volume scratch mask holding the region-growing preview.
    grow_preview: Vec<u8>,
    /// Voxels currently marked in `grow_preview` (for cheap clearing).
    grow_marked: Vec<u32>,
    /// Bumped whenever the preview changes → overlay rebuild.
    grow_gen: u64,
    /// Counter for naming newly created segmentations.
    seg_counter: usize,

    dose_mode: DoseMode,
    dose_opacity: f32,
    dose_threshold_pct: f32,
    iso_levels: Vec<IsoLevel>,

    /// Bumped whenever ROI visibility / dose settings change → cache rebuild.
    settings_gen: u64,

    /// Light / dark / follow-the-system appearance, persisted between runs.
    theme: egui::ThemePreference,
    /// Non-fatal note shown in the View menu if the settings file could not
    /// be written (e.g. a read-only installation folder).
    settings_error: Option<String>,
}

impl ViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial_a: Option<PathBuf>,
        initial_b: Option<PathBuf>,
    ) -> Self {
        let prefs = settings::load();
        cc.egui_ctx.set_theme(prefs.theme);
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
            gen_open: false,
            gen_params: GenParams::default(),
            gen_dir: gen_test_data::default_output_dir().display().to_string(),
            gen_job: None,
            gen_result: None,
            gen_load_after: true,
            anon_open: false,
            anon_dir: String::new(),
            anon_out: String::new(),
            anon_in_place: false,
            anon_remove_private: true,
            anon_remap_uids: true,
            anon_mark: true,
            anon_scan: None,
            anon_scan_job: None,
            anon_apply_job: None,
            anon_result: None,
            planar_windows: Vec::new(),
            d3_windows: Vec::new(),
            tree_action: None,
            maximized: None,
            reg_apply_invert: false,
            window_center: 40.0,
            window_width: 400.0,
            show_contours: true,
            show_crosshair: true,
            show_labels: true,
            show_isocenters: true,
            seg_tool: SegTool::None,
            brush_radius_mm: 5.0,
            brush_3d: true,
            paint_last: None,
            grow: None,
            grow_state: GrowState::default(),
            grow_preview: Vec::new(),
            grow_marked: Vec::new(),
            grow_gen: 0,
            seg_counter: 0,
            dose_mode: DoseMode::Off,
            dose_opacity: 0.45,
            dose_threshold_pct: 15.0,
            iso_levels: default_iso_levels(),
            settings_gen: 0,
            theme: prefs.theme,
            settings_error: None,
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
        self.loading = Some(Job { progress, rx });
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
        self.loading = Some(Job { progress, rx });
    }

    /// A folder finished loading (*File ▶ Add DICOM folder*): merge it into
    /// an occupied slot, or install it into an empty one. Merging leaves the
    /// displayed volume and all selections untouched — the new patients /
    /// studies / series simply appear in the data tree.
    fn absorb_loaded_study(&mut self, slot: usize, study: LoadedStudy) {
        if self.slots[slot].study.is_some() {
            let dest = self.slots[slot].study.as_mut().unwrap();
            let notes = loader::merge_study(dest, study);
            dest.warnings.extend(notes);
            self.settings_gen += 1;
        } else {
            self.on_study_loaded(slot, study);
        }
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
        // Default to the structure set drawn on the active image series
        // (matters for e.g. 4DCT patients with one RTSTRUCT per phase).
        let active_uid = study.series.get(study.active_series).map(|se| se.uid.clone());
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
        // Segmentations were painted on the previous volume grid.
        s.segs.clear();
        s.active_seg = 0;
        s.study = Some(study);
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
                        s.roi_visible =
                            vec![true; study.structure_sets[i].rois.len()];
                    }
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
            // Segmentations were painted on the previous volume grid.
            s.segs.clear();
            s.active_seg = 0;
            self.cancel_grow();
            self.paint_last = None;
            self.clear_registration();
            self.settings_gen += 1;
        }
    }

    fn close_comparison(&mut self) {
        self.slots[1] = StudySlot::empty();
        self.comparison = false;
        self.hovered_slot = 0;
        self.planar_windows.retain(|w| w.slot != 1);
        self.d3_windows.retain(|w| w.slot != 1);
        if self.maximized.map(|(s, _)| s == 1).unwrap_or(false) {
            self.maximized = None;
        }
        self.clear_registration();
    }

    /// Empty a study slot completely (used by tree "move" actions).
    fn tree_clear_slot(&mut self, slot: usize) {
        self.slots[slot] = StudySlot::empty();
        self.planar_windows.retain(|w| w.slot != slot);
        self.d3_windows.retain(|w| w.slot != slot);
        if self.maximized.map(|(s, _)| s == slot).unwrap_or(false) {
            self.maximized = None;
        }
        self.clear_registration();
        self.hovered_slot = 0;
    }

    /// Install a rigid transform (e.g. from a DICOM REG object) as the
    /// active registration, exactly as if it had been computed.
    fn apply_external_rigid(&mut self, rigid: registration::RigidTransform, fixed_slot: usize) {
        self.registration = Some(ActiveRegistration {
            result: RegistrationResult {
                transform: Arc::new(Transform3 { rigid, bspline: None }),
                kind: RegKind::Rigid,
                initial_metric: 0.0,
                final_metric: 0.0,
                iterations_run: 0,
                elapsed_secs: 0.0,
            },
            fixed_slot,
        });
        self.fusion_on = self.slots[0].study.is_some() && self.slots[1].study.is_some();
        self.reg_gen += 1;
        let cursor = self.slots[fixed_slot].cursor;
        self.set_cursor(fixed_slot, cursor, usize::MAX);
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
            let _ = tx.send((fixed_slot, res));
        });
        self.reg_job = Some(Job { progress, rx });
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

    /// Export a loaded study as DICOM files into a user-chosen folder.
    fn start_export(&mut self, slot: usize) {
        if self.export_job.is_some() {
            return;
        }
        let Some(study) = &self.slots[slot].study else { return };
        let Some(dir) = rfd::FileDialog::new()
            .set_title(format!("Export dataset {} as DICOM — choose folder", SLOT_NAMES[slot]))
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
        self.export_job = Some(Job { progress, rx });
    }

    /// Apply an appearance preference and remember it for the next run.
    fn set_theme(&mut self, ctx: &egui::Context, theme: egui::ThemePreference) {
        self.theme = theme;
        ctx.set_theme(theme);
        match settings::save(&Settings { theme }) {
            Ok(()) => self.settings_error = None,
            Err(e) => {
                self.settings_error = Some(format!("⚠ settings not saved: {e:#}"));
            }
        }
    }

    /// Write the built-in synthetic RT test study into the configured folder
    /// (background thread; the folder is created if it does not exist).
    fn start_generate(&mut self) {
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

    /// Reset zoom, pan and slice (back to the volume center) of every view.
    fn reset_all_views(&mut self) {
        for s in &mut self.slots {
            let dims = s.study.as_ref().map(|st| st.volume.dims);
            for v in &mut s.views {
                v.zoom = 0.0;
                v.pan = Vec2::ZERO;
                if let Some(dims) = dims {
                    v.slice = match v.plane {
                        ViewPlane::Axial => dims[2] / 2,
                        ViewPlane::Sagittal => dims[0] / 2,
                        ViewPlane::Coronal => dims[1] / 2,
                    };
                    v.invalidate();
                }
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
        h = h.rotate_left(11) ^ (self.slots[slot].active_structs as u64 + 1);
        for (i, v) in self.slots[slot].roi_visible.iter().enumerate() {
            if *v {
                h = h.rotate_left(7) ^ (i as u64 + 1);
            }
        }
        h ^ self.settings_gen.wrapping_mul(0x2545F4914F6CDD1D)
    }

    // -- Interactive segmentation ------------------------------------------

    /// Everything the 2D segmentation overlays of a slot depend on.
    fn seg_overlay_hash(&self, slot: usize) -> u64 {
        let mut h = mix(0xA07A_5E65u64, slot as u64 + 1);
        for (i, s) in self.slots[slot].segs.iter().enumerate() {
            h = mix(h, i as u64 + 1);
            h = mix(h, s.gen);
            h = mix(h, s.visible as u64);
            h = mix(h, u64::from_le_bytes([s.color[0], s.color[1], s.color[2], 0, 0, 0, 0, 0]));
        }
        if self.grow.as_ref().is_some_and(|g| g.slot == slot) {
            h = mix(h, self.grow_gen.wrapping_add(1));
        }
        h
    }

    /// Everything the 3D segmentation meshes of a slot depend on
    /// (geometry only — color and visibility are applied at draw time).
    fn seg_mesh_hash(&self, slot: usize) -> u64 {
        let mut h = mix(0x5E6_3E54u64, slot as u64 + 1);
        for s in &self.slots[slot].segs {
            h = mix(h, s.gen.wrapping_add(1));
        }
        h
    }

    /// Create a new segmentation on a slot's volume and make it active.
    fn create_seg(&mut self, slot: usize) {
        let Some(study) = &self.slots[slot].study else { return };
        let dims = study.volume.dims;
        let color = segmentation::SEG_PALETTE[self.seg_counter % segmentation::SEG_PALETTE.len()];
        self.seg_counter += 1;
        let name = format!("Seg {}", self.seg_counter);
        let s = &mut self.slots[slot];
        s.segs.push(Segmentation::new(name, color, dims));
        s.active_seg = s.segs.len() - 1;
    }

    /// Apply one brush sample: paints a capsule from the previous sample of
    /// this stroke to `vxl` (creating a segmentation first if none exists).
    fn apply_brush(&mut self, slot: usize, plane: ViewPlane, slice: usize, vxl: [f64; 3], erase: bool) {
        if self.slots[slot].segs.is_empty() {
            if erase {
                return;
            }
            self.create_seg(slot);
        }
        let radius = self.brush_radius_mm as f64;
        let plane2d = if self.brush_3d { None } else { Some((plane, slice)) };
        let from = match self.paint_last {
            Some((s, p)) if s == slot => p,
            _ => vxl,
        };
        let StudySlot { study, segs, active_seg, .. } = &mut self.slots[slot];
        let vol = &study.as_ref().unwrap().volume;
        let idx = (*active_seg).min(segs.len() - 1);
        segs[idx].paint_capsule(vol, from, vxl, radius, erase, plane2d);
        self.paint_last = Some((slot, vxl));
    }

    /// Close the brush stroke in progress (one undo step).
    fn end_paint_stroke(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        if let Some(seg) = s.segs.get_mut(s.active_seg) {
            seg.end_stroke();
        }
        self.paint_last = None;
    }

    fn begin_grow(&mut self, slot: usize, seed: [f64; 3], y: f32) {
        let Some(study) = &self.slots[slot].study else { return };
        let dims = study.volume.dims;
        let seed = [
            (seed[0].round().max(0.0) as usize).min(dims[0] - 1),
            (seed[1].round().max(0.0) as usize).min(dims[1] - 1),
            (seed[2].round().max(0.0) as usize).min(dims[2] - 1),
        ];
        self.cancel_grow();
        self.grow = Some(GrowDrag { slot, level: 1.0, y0: y, capped: false });
        let vol = &self.slots[slot].study.as_ref().unwrap().volume;
        self.grow_state.seed(vol, seed);
        self.sync_grow_preview();
    }

    fn update_grow(&mut self, y: f32) {
        let Some(g) = &mut self.grow else { return };
        // Dragging up extends the geodesic reach exponentially.
        let level = ((g.y0 - y) * 0.015).exp().clamp(0.02, 1000.0);
        if (level / g.level - 1.0).abs() < 0.01 {
            return;
        }
        g.level = level;
        let slot = g.slot;
        let Some(study) = &self.slots[slot].study else { return };
        self.grow_state.set_level(&study.volume, level);
        self.sync_grow_preview();
    }

    /// Bring the preview mask in line with the grow state's current
    /// selection (only previously marked voxels are cleared, so the
    /// full-volume scratch buffer never needs a full wipe).
    fn sync_grow_preview(&mut self) {
        let Some(g) = &mut self.grow else { return };
        g.capped = self.grow_state.capped;
        let n = self.slots[g.slot]
            .study
            .as_ref()
            .map(|st| st.volume.dims.iter().product())
            .unwrap_or(0);
        if self.grow_preview.len() != n {
            self.grow_preview.clear();
            self.grow_preview.resize(n, 0);
        }
        for &v in &self.grow_marked {
            if let Some(p) = self.grow_preview.get_mut(v as usize) {
                *p = 0;
            }
        }
        self.grow_marked.clear();
        self.grow_marked.extend_from_slice(&self.grow_state.voxels);
        for &v in &self.grow_marked {
            self.grow_preview[v as usize] = 1;
        }
        self.grow_gen += 1;
    }

    /// Commit the previewed region into the slot's active segmentation,
    /// filling slice-enclosed holes (vessels etc.) so the organ is solid.
    fn commit_grow(&mut self) {
        let Some(g) = self.grow.take() else { return };
        let slot = g.slot;
        if !self.grow_state.voxels.is_empty() {
            if self.slots[slot].segs.is_empty() {
                self.create_seg(slot);
            }
            let dims = self.slots[slot].study.as_ref().unwrap().volume.dims;
            let mut voxels = self.grow_state.voxels.clone();
            segmentation::fill_holes_slicewise(&mut voxels, dims);
            let s = &mut self.slots[slot];
            let idx = s.active_seg.min(s.segs.len() - 1);
            s.segs[idx].add_voxels(&voxels);
            s.segs[idx].end_stroke();
        }
        self.clear_grow_preview();
        self.grow_state.release();
    }

    /// Abandon any region-growing drag and its preview.
    fn cancel_grow(&mut self) {
        self.grow = None;
        self.clear_grow_preview();
        self.grow_state.release();
    }

    fn clear_grow_preview(&mut self) {
        if self.grow_marked.is_empty() {
            return;
        }
        for &v in &self.grow_marked {
            if let Some(p) = self.grow_preview.get_mut(v as usize) {
                *p = 0;
            }
        }
        self.grow_marked.clear();
        self.grow_gen += 1;
    }

    /// Undo the last stroke of a slot's active segmentation.
    fn undo_active_seg(&mut self, slot: usize) {
        let s = &mut self.slots[slot];
        if let Some(seg) = s.segs.get_mut(s.active_seg) {
            seg.undo_last();
        }
    }

    /// Convert a segmentation into RTSTRUCT contours: appends a ROI to the
    /// slot's active structure set (creating an in-memory set if the study
    /// has none), so it displays like any ROI and rides the DICOM export.
    fn seg_to_rtstruct(&mut self, slot: usize, seg_idx: usize) {
        let StudySlot { study, segs, active_structs, roi_visible, .. } = &mut self.slots[slot];
        let Some(study) = study.as_mut() else { return };
        let Some(seg) = segs.get(seg_idx) else { return };
        let vol = &study.volume;
        if let Some(ss) = study.structure_sets.get_mut(*active_structs) {
            let number = ss.rois.iter().map(|r| r.number).max().unwrap_or(0) + 1;
            ss.rois.push(segmentation::mask_to_roi(seg, vol, number));
            roi_visible.push(true);
        } else {
            let active_series = study.series.get(study.active_series);
            // Pseudo-UID for the in-memory set (rewritten on DICOM export).
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);
            study.structure_sets.push(crate::rtstruct::StructureSet {
                label: "Segmentations".into(),
                frame_of_reference_uid: vol.frame_of_reference_uid.clone(),
                sop_instance_uid: format!("2.25.{stamp}"),
                study_uid: active_series.map(|s| s.study_uid.clone()).unwrap_or_default(),
                referenced_series_uid: active_series.map(|s| s.uid.clone()).unwrap_or_default(),
                file_name: "painted-segmentation".into(),
                rois: vec![segmentation::mask_to_roi(seg, vol, 1)],
            });
            *active_structs = study.structure_sets.len() - 1;
            *roi_visible = vec![true];
        }
        self.settings_gen += 1;
    }

    fn pick_folder(title: &str) -> Option<PathBuf> {
        rfd::FileDialog::new().set_title(title).pick_folder()
    }

    // -- Data tree copy / move / remove actions ----------------------------

    fn apply_tree_action(&mut self, action: TreeAction) {
        self.tree_transfer(action.from, &action.sel, action.op);
    }

    /// Series selection mask for a tree selection.
    fn tree_sel_mask(study: &LoadedStudy, sel: &TreeSel) -> Vec<bool> {
        match sel {
            TreeSel::Patient(pid) => study
                .series
                .iter()
                .map(|s| s.patient_key() == pid)
                .collect(),
            TreeSel::Study(uid) => study
                .series
                .iter()
                .map(|s| s.study_uid == *uid)
                .collect(),
            TreeSel::Series(i) => (0..study.series.len()).map(|k| k == *i).collect(),
        }
    }

    /// Which RT objects the selected series carry along. A single series
    /// takes only its reference chain (RTSTRUCT drawn on it, plans made on
    /// those structure sets, doses computed for those plans); study/patient
    /// selections additionally take objects filed under the same studies.
    fn subset_masks(
        study: &LoadedStudy,
        sel: &[bool],
        study_scope: bool,
        take_extras: bool,
    ) -> SubsetMasks {
        if take_extras {
            // Whole slot content: everything goes.
            return SubsetMasks {
                series: sel.to_vec(),
                structs: vec![true; study.structure_sets.len()],
                doses: vec![true; study.doses.len()],
                plans: vec![true; study.plans.len()],
                take_extras,
            };
        }
        let suids: Vec<&str> = study
            .series
            .iter()
            .zip(sel)
            .filter(|(_, k)| **k)
            .map(|(s, _)| s.uid.as_str())
            .collect();
        let stuids: Vec<&str> = study
            .series
            .iter()
            .zip(sel)
            .filter(|(_, k)| **k)
            .map(|(s, _)| s.study_uid.as_str())
            .filter(|u| !u.is_empty())
            .collect();
        let structs: Vec<bool> = study
            .structure_sets
            .iter()
            .map(|ss| {
                suids.contains(&ss.referenced_series_uid.as_str())
                    || (study_scope
                        && !ss.study_uid.is_empty()
                        && stuids.contains(&ss.study_uid.as_str()))
            })
            .collect();
        let struct_sops: Vec<&str> = study
            .structure_sets
            .iter()
            .zip(&structs)
            .filter(|(_, k)| **k)
            .map(|(s, _)| s.sop_instance_uid.as_str())
            .filter(|u| !u.is_empty())
            .collect();
        let plans: Vec<bool> = study
            .plans
            .iter()
            .map(|p| {
                struct_sops.contains(&p.referenced_structset_uid.as_str())
                    || (study_scope
                        && !p.study_uid.is_empty()
                        && stuids.contains(&p.study_uid.as_str()))
            })
            .collect();
        let plan_sops: Vec<&str> = study
            .plans
            .iter()
            .zip(&plans)
            .filter(|(_, k)| **k)
            .map(|(p, _)| p.sop_instance_uid.as_str())
            .filter(|u| !u.is_empty())
            .collect();
        let doses: Vec<bool> = study
            .doses
            .iter()
            .map(|d| {
                plan_sops.contains(&d.referenced_plan_uid.as_str())
                    || (study_scope
                        && !d.study_uid.is_empty()
                        && stuids.contains(&d.study_uid.as_str()))
            })
            .collect();
        SubsetMasks {
            series: sel.to_vec(),
            structs,
            doses,
            plans,
            take_extras,
        }
    }

    /// Standalone copy of the selected subset. `activate` is the source
    /// series index to display; the volume is a placeholder (the source's
    /// current volume) that is correct exactly when `activate` is the
    /// source's active series.
    fn build_subset(study: &LoadedStudy, masks: &SubsetMasks, activate: usize) -> LoadedStudy {
        let pick = |sel: &[bool], n: usize| -> Vec<usize> {
            (0..n).filter(|&i| sel.get(i).copied().unwrap_or(false)).collect()
        };
        let series: Vec<loader::SeriesInfo> = pick(&masks.series, study.series.len())
            .iter()
            .map(|&i| study.series[i].clone())
            .collect();
        let sub_active = pick(&masks.series, study.series.len())
            .iter()
            .position(|&i| i == activate)
            .unwrap_or(0);
        let se = &study.series[activate];
        let meta = loader::PatientMeta {
            patient_name: if se.patient_name.is_empty() {
                study.meta.patient_name.clone()
            } else {
                se.patient_name.clone()
            },
            patient_id: if se.patient_id.is_empty() {
                study.meta.patient_id.clone()
            } else {
                se.patient_id.clone()
            },
            study_date: se.study_date.clone(),
            study_description: se.study_description.clone(),
        };
        LoadedStudy {
            meta,
            series,
            active_series: sub_active,
            volume: study.volume.clone(),
            structure_sets: pick(&masks.structs, study.structure_sets.len())
                .iter()
                .map(|&i| study.structure_sets[i].clone())
                .collect(),
            doses: pick(&masks.doses, study.doses.len())
                .iter()
                .map(|&i| study.doses[i].clone())
                .collect(),
            plans: pick(&masks.plans, study.plans.len())
                .iter()
                .map(|&i| study.plans[i].clone())
                .collect(),
            planar_images: if masks.take_extras { study.planar_images.clone() } else { Vec::new() },
            registrations: if masks.take_extras { study.registrations.clone() } else { Vec::new() },
            treat_records: if masks.take_extras { study.treat_records.clone() } else { Vec::new() },
            warnings: Vec::new(),
            default_window: study.default_window,
        }
    }

    /// Copy / move / remove a tree selection. Copy and move merge the
    /// selection (plus its linked RT objects) into the other dataset slot;
    /// move and remove then delete it from the source.
    fn tree_transfer(&mut self, from: usize, sel: &TreeSel, op: TreeOp) {
        let Some(study) = self.slots[from].study.as_ref() else { return };
        let sel_mask = Self::tree_sel_mask(study, sel);
        if !sel_mask.iter().any(|b| *b) {
            return;
        }
        let all_selected = sel_mask.iter().all(|b| *b);
        let study_scope = !matches!(sel, TreeSel::Series(_));
        let masks = Self::subset_masks(study, &sel_mask, study_scope, study_scope && all_selected);

        if op != TreeOp::Remove {
            // Choose the series the destination will display.
            let active = study.active_series;
            let activate = if sel_mask.get(active).copied().unwrap_or(false) {
                active
            } else {
                match (0..study.series.len())
                    .find(|&i| sel_mask[i] && !study.series[i].files.is_empty())
                {
                    Some(i) => i,
                    None => {
                        self.error = Some(
                            "The selected series exist only in memory (no source files) — \
                             they cannot be loaded as the displayed volume of the other slot"
                                .into(),
                        );
                        return;
                    }
                }
            };
            let sub = Self::build_subset(study, &masks, activate);
            let direct = (activate == active)
                .then(|| (study.volume.clone(), study.default_window));
            let uid = study.series[activate].uid.clone();
            self.tree_insert(1 - from, sub, &uid, direct);
        }
        if op != TreeOp::Copy {
            if all_selected {
                self.tree_clear_slot(from);
            } else {
                self.remove_subset(from, &masks);
            }
        }
    }

    /// Merge a subset into a slot (or load it into an empty slot) and show
    /// the series with UID `activate_uid` there. `direct` carries the
    /// volume when it is already in memory (no file reload needed).
    fn tree_insert(
        &mut self,
        to: usize,
        sub: LoadedStudy,
        activate_uid: &str,
        direct: Option<(Volume, (f32, f32))>,
    ) {
        self.comparison = true;
        if self.slots[to].study.is_none() {
            let need_switch = direct.is_none();
            let idx = sub.active_series;
            self.on_study_loaded(to, sub);
            if need_switch {
                self.start_series_switch(to, idx);
            }
            return;
        }
        let idx = {
            let dest = self.slots[to].study.as_mut().unwrap();
            let notes = loader::merge_study(dest, sub);
            dest.warnings.extend(notes);
            dest.series.iter().position(|s| s.uid == activate_uid)
        };
        if let Some(idx) = idx {
            match direct {
                Some((vol, win)) => self.apply_new_volume(to, vol, win, idx),
                None => self.start_series_switch(to, idx),
            }
        }
        self.settings_gen += 1;
    }

    /// Delete the masked subset from a slot, keeping the displayed volume
    /// valid (switching to another file-backed series if the active one was
    /// removed, clearing the slot if nothing is left).
    fn remove_subset(&mut self, slot: usize, masks: &SubsetMasks) {
        let mut reload: Option<usize> = None;
        let mut empty = false;
        {
            let s = &mut self.slots[slot];
            let Some(st) = s.study.as_mut() else { return };
            let active_uid = st.series.get(st.active_series).map(|se| se.uid.clone());
            let mut i = 0;
            st.series.retain(|_| {
                let k = !masks.series.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            let mut i = 0;
            st.structure_sets.retain(|_| {
                let k = !masks.structs.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            let mut i = 0;
            st.doses.retain(|_| {
                let k = !masks.doses.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            let mut i = 0;
            st.plans.retain(|_| {
                let k = !masks.plans.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            if masks.take_extras {
                st.planar_images.clear();
                st.registrations.clear();
                st.treat_records.clear();
            }
            if st.series.is_empty() {
                empty = true;
            } else {
                match active_uid
                    .as_deref()
                    .and_then(|uid| st.series.iter().position(|se| se.uid == uid))
                {
                    Some(i) => st.active_series = i,
                    None => {
                        if let Some(i) = st.series.iter().position(|se| !se.files.is_empty()) {
                            st.active_series = i;
                            reload = Some(i);
                        } else {
                            st.active_series = 0;
                        }
                    }
                }
                // Clamp structure / dose selections after pruning.
                let n_sets = st.structure_sets.len();
                if s.active_structs >= n_sets {
                    s.active_structs = 0;
                    let n = st.structure_sets.first().map(|ss| ss.rois.len()).unwrap_or(0);
                    s.roi_visible = vec![true; n];
                }
                if s.active_dose >= st.doses.len() {
                    s.active_dose = 0;
                }
            }
        }
        if empty {
            self.tree_clear_slot(slot);
            return;
        }
        if let Some(i) = reload {
            self.start_series_switch(slot, i);
        }
        self.settings_gen += 1;
    }

    // -- 3D structure windows ----------------------------------------------

    /// Identity of the structure set a 3D window would be built from.
    fn d3_key(&self, slot: usize) -> u64 {
        let mut h: u64 = 0x9E3779B97F4A7C15 ^ (slot as u64);
        if let Some(ss) = self.slots[slot].active_structures() {
            for b in ss.sop_instance_uid.bytes().chain(ss.file_name.bytes()) {
                h = h.wrapping_mul(31).wrapping_add(b as u64);
            }
            h ^= (self.slots[slot].active_structs as u64) << 40;
            h ^= ss.rois.len() as u64;
        }
        h
    }

    fn open_d3_window(&mut self, slot: usize) {
        let key = self.d3_key(slot);
        if let Some(w) = self.d3_windows.iter_mut().find(|w| w.slot == slot) {
            if w.key == key {
                w.open = true;
                return;
            }
        }
        let ss = self.slots[slot].active_structures().cloned();
        if ss.is_none() && self.slots[slot].segs.is_empty() {
            return;
        }
        // Initial auto-fit from the volume extents; replaced by the meshes'
        // own bounding sphere once structure meshes arrive. Keeps the camera
        // stable for segmentation-only scenes that rebuild while painting.
        let (center, radius) = self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                let v = &st.volume;
                let d = v.dims;
                let a = v.voxel_to_patient(0.0, 0.0, 0.0);
                let b = v.voxel_to_patient(
                    d[0] as f64 - 1.0,
                    d[1] as f64 - 1.0,
                    d[2] as f64 - 1.0,
                );
                let c = (a + b) * 0.5;
                let r = ((b - a).length() * 0.5).max(10.0);
                ([c.x as f32, c.y as f32, c.z as f32], r as f32)
            })
            .unwrap_or(([0.0; 3], 100.0));
        self.d3_windows.retain(|w| w.slot != slot);
        let job = ss.map(|ss| {
            let progress = Arc::new(Progress::default());
            progress.set("starting…");
            let p2 = progress.clone();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let meshes = mesh3d::build_meshes(&ss, &p2);
                let _ = tx.send(meshes);
            });
            Job { progress, rx }
        });
        let no_structs = job.is_none();
        self.d3_windows.push(D3Window {
            slot,
            open: true,
            yaw: 0.7,
            pitch: -0.5,
            zoom: 1.0,
            pan: Vec2::ZERO,
            opacity: 1.0,
            meshes: no_structs.then(|| Arc::new(Vec::new())),
            seg_meshes: None,
            seg_job: None,
            seg_built: 0,
            frame: D3Frame::default(),
            center,
            radius,
            key,
            job,
        });
    }
}

/// Last few characters of a UID for compact display.
fn tail(uid: &str) -> String {
    let n = uid.chars().count();
    if n <= 10 {
        uid.to_string()
    } else {
        uid.chars().skip(n - 10).collect()
    }
}

// ---------------------------------------------------------------------------
// eframe::App
// ---------------------------------------------------------------------------

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Poll background loading.
        match poll_job(&mut self.loading, &ctx, "Loading", &mut self.error) {
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
            None => {}
        }
        // Kick a queued load once the current one finished.
        if self.loading.is_none() {
            if let Some((slot, path)) = self.pending_load.take() {
                self.start_load(slot, path);
            }
        }

        // Poll background simulation.
        if let Some((target, study)) = poll_job(&mut self.sim_job, &ctx, "Simulation", &mut self.error) {
            self.on_study_loaded(target, study);
            self.comparison = true;
        }

        // Poll background export.
        match poll_job(&mut self.export_job, &ctx, "Export", &mut self.error) {
            Some(Ok((n, dir))) => {
                self.export_result = Some(format!("✔ {n} DICOM file(s) written to {dir}"));
            }
            Some(Err(e)) => self.error = Some(format!("Export failed: {e:#}")),
            None => {}
        }

        // Poll background test-data generation.
        match poll_job(&mut self.gen_job, &ctx, "Test data generation", &mut self.error) {
            Some(Ok((n, dir))) => {
                self.gen_result =
                    Some(format!("✔ {n} DICOM file(s) written to {}", dir.display()));
                if self.gen_load_after {
                    self.gen_open = false;
                    self.start_load(0, dir);
                }
            }
            Some(Err(e)) => self.error = Some(format!("Test data generation failed: {e:#}")),
            None => {}
        }

        // Poll background registration.
        match poll_job(&mut self.reg_job, &ctx, "Registration", &mut self.error) {
            Some((fixed_slot, Ok(result))) => {
                self.registration = Some(ActiveRegistration { result, fixed_slot });
                self.fusion_on = true;
                self.reg_gen += 1;
                // Re-propagate the crosshair through the new transform.
                let cursor = self.slots[fixed_slot].cursor;
                self.set_cursor(fixed_slot, cursor, usize::MAX);
            }
            Some((_, Err(e))) => {
                let msg = format!("{e:#}");
                if !msg.contains("cancelled") {
                    self.error = Some(format!("Registration failed: {msg}"));
                }
            }
            None => {}
        }

        // Global segmentation shortcuts (skipped while a text field is
        // focused): Ctrl+Z undo, Esc cancels a region-grow drag, [ ] resize
        // the brush.
        if !ctx.egui_wants_keyboard_input() {
            let (undo, esc, smaller, bigger) = ctx.input(|i| {
                (
                    i.modifiers.command && i.key_pressed(egui::Key::Z),
                    i.key_pressed(egui::Key::Escape),
                    i.key_pressed(egui::Key::OpenBracket),
                    i.key_pressed(egui::Key::CloseBracket),
                )
            });
            if undo {
                let slot = self.hovered_slot.min(1);
                self.undo_active_seg(slot);
            }
            if esc && self.grow.is_some() {
                self.cancel_grow();
            }
            if self.seg_tool != SegTool::None {
                if smaller {
                    self.brush_radius_mm = (self.brush_radius_mm / 1.2).max(0.5);
                }
                if bigger {
                    self.brush_radius_mm = (self.brush_radius_mm * 1.2).min(80.0);
                }
            }
        }

        self.menu_bar(ui, &ctx);
        self.top_bar(ui);
        self.side_panel(ui);
        self.status_bar(ui);
        self.central_views(ui);
        self.planar_windows_ui(&ctx);
        self.d3_windows_ui(&ctx);
        if let Some(action) = self.tree_action.take() {
            self.apply_tree_action(action);
        }
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
        let mut open_gen = false;
        let mut new_theme: Option<egui::ThemePreference> = None;

        egui::Panel::top(egui::Id::new("menu_bar")).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui
                        .button("📂 Add DICOM folder to A…")
                        .on_hover_text(
                            "Scan a folder and add its patients / studies / series to \
                             dataset A (existing content stays loaded)",
                        )
                        .clicked()
                    {
                        open_a = true;
                        ui.close();
                    }
                    if ui
                        .button("📂 Add DICOM folder to B…")
                        .on_hover_text(
                            "Scan a folder and add its patients / studies / series to \
                             dataset B (existing content stays loaded)",
                        )
                        .clicked()
                    {
                        open_b = true;
                        ui.close();
                    }
                    let has_a = self.slots[0].study.is_some();
                    if ui
                        .add_enabled(has_a, egui::Button::new("Clear dataset A"))
                        .clicked()
                    {
                        self.tree_clear_slot(0);
                        ui.close();
                    }
                    let has_b = self.slots[1].study.is_some();
                    if ui
                        .add_enabled(has_b, egui::Button::new("Close dataset B"))
                        .clicked()
                    {
                        close_b = true;
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .button("🧪 Generate test data…")
                        .on_hover_text(
                            "Write a complete synthetic RT study (CT, RTSTRUCT, RTPLAN, \
                             RTDOSE, DX, RTIMAGE, REG, RTRECORD) into the application folder",
                        )
                        .clicked()
                    {
                        open_gen = true;
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
                    ui.checkbox(&mut self.link_studies, "Link crosshairs between datasets");
                    ui.separator();
                    ui.checkbox(&mut self.show_contours, "Contours");
                    ui.checkbox(&mut self.show_crosshair, "Crosshair");
                    ui.checkbox(&mut self.show_labels, "Orientation labels");
                    ui.checkbox(&mut self.show_isocenters, "Isocenters");
                    ui.separator();
                    ui.label("Appearance:");
                    let before = self.theme;
                    self.theme.radio_buttons(ui);
                    if self.theme != before {
                        new_theme = Some(self.theme);
                    }
                    if let Some(msg) = &self.settings_error {
                        ui.weak(msg);
                    }
                    ui.separator();
                    if ui.button("Reset all views").clicked() {
                        reset_views = true;
                        ui.close();
                    }
                });
                ui.menu_button("Registration", |ui| {
                    // Quick actions only — direction, parameters, fusion and
                    // cancel/clear live in the sidebar Registration section.
                    let both =
                        self.slots[0].study.is_some() && self.slots[1].study.is_some();
                    let running = self.reg_job.is_some();
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
                    if !both {
                        ui.weak("Load two datasets (comparison mode) first");
                    }
                });
                ui.menu_button("Tools", |ui| {
                    if ui
                        .button("🔏 Anonymize DICOM folder…")
                        .on_hover_text(
                            "Scan a folder, review every identifying tag with its current \
                             and proposed values, then rewrite the files (in place or into \
                             a new folder) with consistently regenerated UIDs",
                        )
                        .clicked()
                    {
                        self.anon_open = true;
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    ui.label("MPR views — mouse:");
                    ui.weak("Left click / drag — move linked crosshair");
                    ui.weak("Mouse wheel — scroll slices");
                    ui.weak("Ctrl + wheel / pinch — zoom at cursor");
                    ui.weak("Middle drag — pan");
                    ui.weak("Right drag — window / level (x = width, y = center)");
                    ui.separator();
                    ui.label("Segmentation (🖌 ◻ ✨ take over the left button):");
                    ui.weak("Left drag — paint / erase");
                    ui.weak("Left press + drag ↑↓ — grow / shrink the region (✨)");
                    ui.weak("Alt — erase while painting");
                    ui.weak("Shift + wheel, or [ ] — brush radius");
                    ui.weak("Ctrl + Z — undo the last stroke");
                    ui.weak("Esc — cancel the running region grow");
                    ui.separator();
                    ui.label("Buttons:");
                    ui.weak("⟲ (view corner) — reset that view's zoom, pan and slice");
                    ui.weak("⛶ / ❐ — maximize that view / restore the layout");
                    ui.weak("⟲ (toolbar) — reset every view of both datasets");
                    ui.weak(
                        "⌖ — show / hide the crosshair; hidden, left click no \
                         longer navigates",
                    );
                    ui.separator();
                    ui.weak(format!(
                        "rust-dicom-viewer {} — research / QA viewer, not a medical device",
                        env!("CARGO_PKG_VERSION")
                    ));
                });
            });
        });

        if open_a {
            if let Some(dir) = Self::pick_folder("Select DICOM folder to add to dataset A") {
                self.start_load(0, dir);
            }
        }
        if open_b {
            if let Some(dir) = Self::pick_folder("Select DICOM folder to add to dataset B") {
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
        if open_gen {
            self.gen_open = true;
        }
        if let Some(theme) = new_theme {
            self.set_theme(ctx, theme);
        }
    }

    // -- Toolbar ----------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        // Only the primary reading controls live here (window/level);
        // file actions, display toggles and appearance are in the menus.
        let any_study = self.slots[0].study.is_some() || self.slots[1].study.is_some();
        if !any_study {
            return;
        }
        egui::Panel::top(egui::Id::new("top_bar")).show(ui, |ui| {
            ui.horizontal(|ui| {
                {
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
                    // 3D structure rendering windows.
                    #[allow(clippy::needless_range_loop)] // `slot` also indexes `self.slots`
                    for slot in 0..2 {
                        let has_3d = self.slots[slot]
                            .study
                            .as_ref()
                            .map(|s| !s.structure_sets.is_empty())
                            .unwrap_or(false)
                            || !self.slots[slot].segs.is_empty();
                        if slot == 1 && self.slots[1].study.is_none() {
                            continue;
                        }
                        if ui
                            .add_enabled(
                                has_3d,
                                egui::Button::new(format!("3D {}", SLOT_NAMES[slot])),
                            )
                            .on_hover_text(format!(
                                "Open a 3D surface rendering of dataset {}'s structures \
                                 and segmentations",
                                SLOT_NAMES[slot]
                            ))
                            .clicked()
                        {
                            self.open_d3_window(slot);
                        }
                    }

                    // Slice-intersection (crosshair) toggle. With the
                    // crosshair hidden, left-click navigation is disabled.
                    if ui
                        .selectable_label(self.show_crosshair, "⌖")
                        .on_hover_text(
                            "Show / hide the slice intersection (crosshair).\n\
                             Hidden: left click does not navigate — slices change \
                             only by scrolling each view",
                        )
                        .clicked()
                    {
                        self.show_crosshair = !self.show_crosshair;
                    }

                    // Reset every view of both datasets.
                    if ui
                        .button("⟲")
                        .on_hover_text("Reset all views (zoom, pan and slice)")
                        .clicked()
                    {
                        self.reset_all_views();
                    }

                    // Segmentation tools. Selecting a tool takes over the
                    // left mouse button in the MPR views.
                    ui.separator();
                    let mut pick = |ui: &mut egui::Ui, tool: SegTool, label: &str, tip: &str| {
                        if ui
                            .selectable_label(self.seg_tool == tool, label)
                            .on_hover_text(tip)
                            .clicked()
                        {
                            self.seg_tool = if self.seg_tool == tool { SegTool::None } else { tool };
                            if self.seg_tool != SegTool::Grow {
                                self.cancel_grow();
                            }
                        }
                    };
                    pick(
                        ui,
                        SegTool::Brush,
                        "🖌 Paint",
                        "Paint the active segmentation (LMB drag).\n\
                         Hold Alt to erase · Shift+wheel or [ ] resize the brush · Ctrl+Z undo",
                    );
                    pick(
                        ui,
                        SegTool::Erase,
                        "◻ Erase",
                        "Erase from the active segmentation (LMB drag)",
                    );
                    pick(
                        ui,
                        SegTool::Grow,
                        "✨ Grow",
                        "Interactive organ segmentation (geodesic fast marching): press \
                         to place a seed, drag up/down to grow/shrink the region with a \
                         live preview. Intensity changes and edges act as barriers, so \
                         the organ under the seed is suggested before anything leaks. \
                         Release commits (enclosed holes are filled), Esc cancels",
                    );
                    if matches!(self.seg_tool, SegTool::Brush | SegTool::Erase) {
                        ui.add(
                            egui::DragValue::new(&mut self.brush_radius_mm)
                                .speed(0.5)
                                .range(0.5..=80.0)
                                .suffix(" mm"),
                        )
                        .on_hover_text("Brush radius");
                        if ui
                            .selectable_label(self.brush_3d, "3D")
                            .on_hover_text(
                                "Spherical 3D brush: paints through neighboring slices.\n\
                                 Off: flat 2D circle on the displayed slice only",
                            )
                            .clicked()
                        {
                            self.brush_3d = !self.brush_3d;
                        }
                    }
                }

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
                    let t = res.transform.rigid.params();
                    ui.weak(format!(
                        "t = ({:.1}, {:.1}, {:.1}) mm  r = ({:.2}, {:.2}, {:.2})°",
                        t[3],
                        t[4],
                        t[5],
                        t[0].to_degrees(),
                        t[1].to_degrees(),
                        t[2].to_degrees()
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
                        "▶ generates dataset {}",
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
                            "⚙ Generate transformed dataset ▶ {}",
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
                        #[allow(clippy::needless_range_loop)] // `slot` also indexes `self.slots`
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
        // Plain header — the patient(s) always appear as tree nodes below.
        let header = format!("Dataset {}", SLOT_NAMES[slot]);
        let ch = egui::CollapsingHeader::new(egui::RichText::new(header).strong())
            .id_salt(("study_hdr", slot))
            .default_open(true)
            .show(ui, |ui| {
                self.series_selector(ui, slot);
                self.structures_section(ui, slot);
                self.segmentation_section(ui, slot);
                self.dose_section(ui, slot);
                self.plan_section(ui, slot);
                self.planar_section(ui, slot);
                self.reg_objects_section(ui, slot);
                self.records_section(ui, slot);
                self.warnings_section(ui, slot);
            });
        // Right-click on the dataset header: clear the slot.
        let mut clear = false;
        ch.header_response.context_menu(|ui| {
            if ui
                .button(format!("Clear dataset {}", SLOT_NAMES[slot]))
                .clicked()
            {
                clear = true;
                ui.close();
            }
        });
        if clear {
            if slot == 1 {
                self.close_comparison();
            } else {
                self.tree_clear_slot(slot);
            }
        }
        ui.separator();
    }

    /// DICOM data tree: patient ▶ study ▶ series, all visible at once. The
    /// active series (the displayed volume) is marked; clicking another
    /// series loads it. Right-click any level to copy / move it to the
    /// other dataset or remove it.
    fn series_selector(&mut self, ui: &mut egui::Ui, slot: usize) {
        let mut switch_to = None;
        let mut act_series: Option<TreeAction> = None;
        let mut act_study: Option<TreeAction> = None;
        let mut act_patient: Option<TreeAction> = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            let active = study.active_series;
            let other = SLOT_NAMES[1 - slot];
            let label = |s: &loader::SeriesInfo| {
                format!(
                    "{} {} ({} sl.)",
                    s.modality,
                    if s.description.is_empty() { "series" } else { &s.description },
                    s.files.len()
                )
            };
            // Distinct patients, in first-seen order.
            let mut patients: Vec<&str> = Vec::new();
            for s in &study.series {
                let k = s.patient_key();
                if !patients.contains(&k) {
                    patients.push(k);
                }
            }
            for (pi, pkey) in patients.iter().enumerate() {
                let pinfo = study
                    .series
                    .iter()
                    .find(|s| s.patient_key() == *pkey)
                    .unwrap();
                let pname = pinfo.patient_name.replace('^', " ");
                let ptitle = if pname.is_empty() && pinfo.patient_id.is_empty() {
                    "Unknown patient".to_string()
                } else if pname.is_empty() {
                    format!("Patient {}", pinfo.patient_id)
                } else if pinfo.patient_id.is_empty() {
                    pname.clone()
                } else {
                    format!("{} ({})", pname, pinfo.patient_id)
                };
                let pch = egui::CollapsingHeader::new(ptitle)
                    .id_salt(("pat_hdr", slot, pi))
                    .default_open(true)
                    .show(ui, |ui| {
                        // Studies of this patient, in first-seen order.
                        let mut studies: Vec<&str> = Vec::new();
                        for s in &study.series {
                            if s.patient_key() == *pkey
                                && !studies.contains(&s.study_uid.as_str())
                            {
                                studies.push(&s.study_uid);
                            }
                        }
                        for (si, study_uid) in studies.iter().enumerate() {
                            let info = study
                                .series
                                .iter()
                                .find(|s| {
                                    s.study_uid == *study_uid && s.patient_key() == *pkey
                                })
                                .unwrap();
                            let title = format!(
                                "Study {}{}",
                                if info.study_date.is_empty() {
                                    format!("{}", si + 1)
                                } else {
                                    info.study_date.clone()
                                },
                                if info.study_description.is_empty() {
                                    String::new()
                                } else {
                                    format!(" — {}", info.study_description)
                                }
                            );
                            let sch = egui::CollapsingHeader::new(title)
                                .id_salt(("study_tree", slot, pi, si))
                                .default_open(true)
                                .show(ui, |ui| {
                                    for (i, s) in study.series.iter().enumerate() {
                                        if s.study_uid != *study_uid
                                            || s.patient_key() != *pkey
                                        {
                                            continue;
                                        }
                                        let resp =
                                            ui.selectable_label(i == active, label(s));
                                        if resp.clicked() && i != active {
                                            switch_to = Some(i);
                                        }
                                        resp.context_menu(|ui| {
                                            if ui
                                                .button(format!(
                                                    "Copy series to dataset {other}"
                                                ))
                                                .clicked()
                                            {
                                                act_series = Some(TreeAction {
                                                    from: slot,
                                                    sel: TreeSel::Series(i),
                                                    op: TreeOp::Copy,
                                                });
                                                ui.close();
                                            }
                                            if ui
                                                .button(format!(
                                                    "Move series to dataset {other}"
                                                ))
                                                .clicked()
                                            {
                                                act_series = Some(TreeAction {
                                                    from: slot,
                                                    sel: TreeSel::Series(i),
                                                    op: TreeOp::Move,
                                                });
                                                ui.close();
                                            }
                                            ui.separator();
                                            if ui.button("Remove series").clicked() {
                                                act_series = Some(TreeAction {
                                                    from: slot,
                                                    sel: TreeSel::Series(i),
                                                    op: TreeOp::Remove,
                                                });
                                                ui.close();
                                            }
                                        });
                                        resp.on_hover_text(format!(
                                            "Series UID …{}\nright-click: copy / move to \
                                             dataset {other}, or remove",
                                            tail(&s.uid)
                                        ));
                                    }
                                });
                            sch.header_response.context_menu(|ui| {
                                if ui
                                    .button(format!("Copy study to dataset {other}"))
                                    .clicked()
                                {
                                    act_study = Some(TreeAction {
                                        from: slot,
                                        sel: TreeSel::Study(study_uid.to_string()),
                                        op: TreeOp::Copy,
                                    });
                                    ui.close();
                                }
                                if ui
                                    .button(format!("Move study to dataset {other}"))
                                    .clicked()
                                {
                                    act_study = Some(TreeAction {
                                        from: slot,
                                        sel: TreeSel::Study(study_uid.to_string()),
                                        op: TreeOp::Move,
                                    });
                                    ui.close();
                                }
                                ui.separator();
                                if ui.button("Remove study").clicked() {
                                    act_study = Some(TreeAction {
                                        from: slot,
                                        sel: TreeSel::Study(study_uid.to_string()),
                                        op: TreeOp::Remove,
                                    });
                                    ui.close();
                                }
                            });
                        }
                    });
                pch.header_response.context_menu(|ui| {
                    if ui
                        .button(format!("Copy patient to dataset {other}"))
                        .clicked()
                    {
                        act_patient = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Patient(pkey.to_string()),
                            op: TreeOp::Copy,
                        });
                        ui.close();
                    }
                    if ui
                        .button(format!("Move patient to dataset {other}"))
                        .clicked()
                    {
                        act_patient = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Patient(pkey.to_string()),
                            op: TreeOp::Move,
                        });
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Remove patient").clicked() {
                        act_patient = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Patient(pkey.to_string()),
                            op: TreeOp::Remove,
                        });
                        ui.close();
                    }
                });
            }
        }
        if let Some(a) = act_series.or(act_study).or(act_patient) {
            self.tree_action = Some(a);
        }
        if let Some(i) = switch_to {
            self.start_series_switch(slot, i);
        }
    }

    fn structures_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n_sets = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.structure_sets.len())
            .unwrap_or(0);
        if n_sets == 0 {
            // No RTSTRUCT in this study — show nothing.
            return;
        }
        let mut changed = false;
        let mut new_active: Option<usize> = None;
        {
            let StudySlot { study, roi_visible, active_structs, .. } = &mut self.slots[slot];
            let study = study.as_ref().unwrap();
            let active_set = (*active_structs).min(n_sets - 1);
            let ss = &study.structure_sets[active_set];
            let n_vis = roi_visible.iter().filter(|v| **v).count();
            egui::CollapsingHeader::new(format!("Structures ({}/{})", n_vis, ss.rois.len()))
                .id_salt(("structs", slot))
                .default_open(true)
                .show(ui, |ui| {
                    // Structure-set selector with links to the referenced
                    // image series (one set per 4DCT phase, replans, …).
                    if n_sets > 1 {
                        for (i, set) in study.structure_sets.iter().enumerate() {
                            let series_ref = study
                                .series
                                .iter()
                                .find(|se| se.uid == set.referenced_series_uid)
                                .map(|se| {
                                    format!(
                                        " ▶ {} {}",
                                        se.modality,
                                        if se.description.is_empty() { "series" } else { &se.description }
                                    )
                                })
                                .unwrap_or_default();
                            let resp = ui.selectable_label(
                                i == active_set,
                                format!(
                                    "{} ({} ROIs){}",
                                    if set.label.is_empty() { &set.file_name } else { &set.label },
                                    set.rois.len(),
                                    series_ref
                                ),
                            );
                            if resp.clicked() && i != active_set {
                                new_active = Some(i);
                            }
                            resp.on_hover_text(format!(
                                "{}\nreferences series …{}",
                                set.file_name,
                                tail(&set.referenced_series_uid)
                            ));
                        }
                        ui.separator();
                    }
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
        if let Some(i) = new_active {
            let s = &mut self.slots[slot];
            s.active_structs = i;
            let n = s
                .study
                .as_ref()
                .map(|st| st.structure_sets[i].rois.len())
                .unwrap_or(0);
            s.roi_visible = vec![true; n];
            changed = true;
        }
        if changed {
            self.settings_gen += 1;
        }
    }

    fn segmentation_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let mut make_new = false;
        let mut delete: Option<usize> = None;
        let mut to_struct: Option<usize> = None;
        {
            let StudySlot { study, segs, active_seg, .. } = &mut self.slots[slot];
            let Some(study) = study.as_ref() else { return };
            let spacing = study.volume.spacing;
            egui::CollapsingHeader::new(format!("Segmentations ({})", segs.len()))
                .id_salt(("segs", slot))
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.small_button("➕ New").clicked() {
                            make_new = true;
                        }
                        ui.weak("drawn with 🖌 / ✨ in the views");
                    });
                    for (i, seg) in segs.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.color_edit_button_srgb(&mut seg.color);
                            ui.checkbox(&mut seg.visible, "")
                                .on_hover_text("Show / hide this segmentation");
                            let resp = ui
                                .selectable_label(i == *active_seg, &seg.name)
                                .on_hover_text(
                                    "Click to make this the segmentation the tools edit",
                                );
                            if resp.clicked() {
                                *active_seg = i;
                            }
                            ui.weak(format!("{:.1} cm³", seg.volume_cm3(spacing)));
                            if ui
                                .add_enabled(seg.can_undo(), egui::Button::new("↶").small())
                                .on_hover_text("Undo the last stroke (Ctrl+Z)")
                                .clicked()
                            {
                                seg.undo_last();
                            }
                            if ui
                                .small_button("→RS")
                                .on_hover_text(
                                    "Convert to RTSTRUCT contours: adds a ROI to the \
                                     structure set, so it exports with \
                                     Simulation ▶ 💾 Export",
                                )
                                .clicked()
                            {
                                to_struct = Some(i);
                            }
                            if ui
                                .small_button("🗑")
                                .on_hover_text("Delete this segmentation")
                                .clicked()
                            {
                                delete = Some(i);
                            }
                        });
                    }
                });
        }
        if make_new {
            self.create_seg(slot);
        }
        if let Some(i) = delete {
            let s = &mut self.slots[slot];
            if i < s.segs.len() {
                s.segs.remove(i);
                if s.active_seg >= s.segs.len() {
                    s.active_seg = s.segs.len().saturating_sub(1);
                }
            }
        }
        if let Some(i) = to_struct {
            self.seg_to_rtstruct(slot, i);
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
            let plans = &study.as_ref().unwrap().plans;
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
                    // DICOM cross-reference: which plan this dose belongs to.
                    if !d.referenced_plan_uid.is_empty() {
                        if let Some(p) = plans
                            .iter()
                            .find(|p| p.sop_instance_uid == d.referenced_plan_uid)
                        {
                            ui.weak(format!(
                                "▶ plan {}",
                                if p.label.is_empty() { "unnamed" } else { &p.label }
                            ));
                        }
                    }

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
                // DICOM cross-reference: the structure set the plan was
                // created on.
                if !plan.referenced_structset_uid.is_empty() {
                    if let Some(ss) = study
                        .structure_sets
                        .iter()
                        .find(|s| s.sop_instance_uid == plan.referenced_structset_uid)
                    {
                        ui.weak(format!(
                            "▶ structures {}",
                            if ss.label.is_empty() { &ss.file_name } else { &ss.label }
                        ));
                    }
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

    /// DX / CR / RTIMAGE planar images: list with per-image viewer windows.
    fn planar_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.planar_images.len())
            .unwrap_or(0);
        if n == 0 {
            return;
        }
        let mut open_idx = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            egui::CollapsingHeader::new(format!("Planar images ({n})"))
                .id_salt(("planar", slot))
                .default_open(false)
                .show(ui, |ui| {
                    for (i, img) in study.planar_images.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("[{}]", img.modality)).weak(),
                            );
                            ui.label(&img.label);
                            if ui.small_button("View").clicked() {
                                open_idx = Some(i);
                            }
                        });
                    }
                });
        }
        if let Some(i) = open_idx {
            if let Some(w) = self
                .planar_windows
                .iter_mut()
                .find(|w| w.slot == slot && w.idx == i)
            {
                w.open = true;
            } else {
                let wl = self.slots[slot].study.as_ref().unwrap().planar_images[i].window;
                self.planar_windows.push(PlanarWindow {
                    slot,
                    idx: i,
                    open: true,
                    wl,
                    tex: None,
                    tex_wl: (f32::NAN, f32::NAN),
                });
            }
        }
    }

    /// REG spatial registration objects: matrices + apply as active
    /// registration.
    fn reg_objects_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.registrations.len())
            .unwrap_or(0);
        if n == 0 {
            return;
        }
        let both = self.slots[0].study.is_some() && self.slots[1].study.is_some();
        let mut apply: Option<(registration::RigidTransform, usize)> = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            // Frame-of-reference UIDs of the loaded volumes for hints.
            let for_a = self.slots[0]
                .study
                .as_ref()
                .map(|s| s.volume.frame_of_reference_uid.clone())
                .unwrap_or_default();
            let for_b = self.slots[1]
                .study
                .as_ref()
                .map(|s| s.volume.frame_of_reference_uid.clone())
                .unwrap_or_default();
            let mut invert = self.reg_apply_invert;
            egui::CollapsingHeader::new(format!("Spatial registrations ({n})"))
                .id_salt(("regobj", slot))
                .default_open(false)
                .show(ui, |ui| {
                    for (ri, reg) in study.registrations.iter().enumerate() {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}{}",
                                reg.label,
                                if reg.deformable { "  [deformable: matrices only]" } else { "" }
                            ))
                            .strong(),
                        );
                        for (ii, item) in reg.items.iter().enumerate() {
                            if item.is_identity {
                                ui.weak(format!("· item {}: identity ({})", ii + 1, item.matrix_type));
                                continue;
                            }
                            let m = &item.matrix;
                            ui.weak(format!("· item {}: {}", ii + 1, item.matrix_type));
                            for r in 0..3 {
                                ui.monospace(format!(
                                    "  [{:7.3} {:7.3} {:7.3} {:8.2}]",
                                    m[r * 4],
                                    m[r * 4 + 1],
                                    m[r * 4 + 2],
                                    m[r * 4 + 3]
                                ));
                            }
                            // FoR hints against loaded studies.
                            let src_hint = if !for_a.is_empty() && item.for_uid == for_a {
                                " (= A)"
                            } else if !for_b.is_empty() && item.for_uid == for_b {
                                " (= B)"
                            } else {
                                ""
                            };
                            let dst_hint = if !for_a.is_empty()
                                && reg.frame_of_reference_uid == for_a
                            {
                                " (= A)"
                            } else if !for_b.is_empty() && reg.frame_of_reference_uid == for_b {
                                " (= B)"
                            } else {
                                ""
                            };
                            ui.weak(format!(
                                "  maps FoR …{}{} ▶ …{}{}",
                                tail(&item.for_uid),
                                src_hint,
                                tail(&reg.frame_of_reference_uid),
                                dst_hint
                            ));
                            match extras::matrix_to_rigid(m, invert) {
                                Some(rigid) => {
                                    let rp = rigid.params();
                                    ui.weak(format!(
                                        "  t = ({:.1}, {:.1}, {:.1}) mm  r = ({:.2}, {:.2}, {:.2})°{}",
                                        rp[3],
                                        rp[4],
                                        rp[5],
                                        rp[0].to_degrees(),
                                        rp[1].to_degrees(),
                                        rp[2].to_degrees(),
                                        if invert { "  (inverted)" } else { "" }
                                    ));
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut invert, "Invert")
                                            .on_hover_text(
                                                "Invert the matrix before applying (flip the mapping direction)",
                                            );
                                        if ui
                                            .add_enabled(
                                                both,
                                                egui::Button::new("Apply as B ▶ A"),
                                            )
                                            .on_hover_text(
                                                "Use this matrix as the transform mapping A (fixed) coordinates into B (moving)",
                                            )
                                            .clicked()
                                        {
                                            if let Some(r2) =
                                                extras::matrix_to_rigid(m, invert)
                                            {
                                                apply = Some((r2, 0));
                                            }
                                        }
                                        if ui
                                            .add_enabled(
                                                both,
                                                egui::Button::new("Apply as A ▶ B"),
                                            )
                                            .on_hover_text(
                                                "Use this matrix as the transform mapping B (fixed) coordinates into A (moving)",
                                            )
                                            .clicked()
                                        {
                                            if let Some(r2) =
                                                extras::matrix_to_rigid(m, invert)
                                            {
                                                apply = Some((r2, 1));
                                            }
                                        }
                                    });
                                }
                                None => {
                                    ui.weak("  (matrix is not a pure rigid transform — cannot apply)");
                                }
                            }
                        }
                        if ri + 1 < study.registrations.len() {
                            ui.separator();
                        }
                    }
                });
            self.reg_apply_invert = invert;
        }
        if let Some((rigid, fixed_slot)) = apply {
            self.apply_external_rigid(rigid, fixed_slot);
        }
    }

    /// RT (Ion) Beams Treatment Records: per-beam delivered metersets.
    fn records_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else { return };
        if study.treat_records.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(format!("Treatment records ({})", study.treat_records.len()))
            .id_salt(("records", slot))
            .default_open(false)
            .show(ui, |ui| {
                for (ri, rec) in study.treat_records.iter().enumerate() {
                    ui.label(
                        egui::RichText::new(format!(
                            "{}{}{}{}",
                            rec.label,
                            if rec.ion { "  [ion]" } else { "" },
                            rec.fraction
                                .map(|f| format!("  fx {f}"))
                                .unwrap_or_default(),
                            if rec.date.is_empty() {
                                String::new()
                            } else {
                                format!("  {}", rec.date)
                            }
                        ))
                        .strong(),
                    );
                    if !rec.machine.is_empty() {
                        ui.weak(format!("Machine: {}", rec.machine));
                    }
                    egui::Grid::new(("rec_grid", slot, ri))
                        .striped(true)
                        .min_col_width(10.0)
                        .show(ui, |ui| {
                            ui.strong("Beam");
                            ui.strong("MU spec");
                            ui.strong("MU del");
                            ui.strong("Δ%");
                            ui.strong("Status");
                            ui.end_row();
                            for b in &rec.beams {
                                ui.label(&b.name).on_hover_text(format!(
                                    "Beam {} · verification: {}",
                                    b.number,
                                    if b.verification_status.is_empty() {
                                        "n/a"
                                    } else {
                                        &b.verification_status
                                    }
                                ));
                                ui.label(
                                    b.specified_meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(
                                    b.delivered_meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(match (b.specified_meterset, b.delivered_meterset) {
                                    (Some(s), Some(d)) if s > 1e-9 => {
                                        format!("{:+.1}", 100.0 * (d - s) / s)
                                    }
                                    _ => "–".into(),
                                });
                                let status = if b.termination_status.is_empty() {
                                    "–"
                                } else {
                                    &b.termination_status
                                };
                                if status == "NORMAL" || status == "–" {
                                    ui.label(status);
                                } else {
                                    let c = alert_color(ui.visuals());
                                    ui.label(egui::RichText::new(status).color(c));
                                }
                                ui.end_row();
                            }
                        });
                }
            });
    }

    fn warnings_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else { return };
        if study.warnings.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(
            egui::RichText::new(format!("⚠ Warnings ({})", study.warnings.len()))
                .color(warn_color(ui.visuals())),
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
                #[allow(clippy::needless_range_loop)] // `slot` also indexes `self.slots`
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
                    ui.weak(match self.seg_tool {
                        SegTool::None => {
                            "LMB crosshair · RMB W/L · MMB pan · wheel slice · Ctrl+wheel zoom"
                        }
                        SegTool::Brush => {
                            "LMB paint · Alt erase · Shift+wheel / [ ] brush size · Ctrl+Z undo · wheel slice"
                        }
                        SegTool::Erase => {
                            "LMB erase · Shift+wheel / [ ] brush size · Ctrl+Z undo · wheel slice"
                        }
                        SegTool::Grow => {
                            "LMB press seed · drag up/down = grow/shrink · release commit · Esc cancel · Ctrl+Z undo"
                        }
                    });
                });
            });
        });
    }

    // -- Central: one or two rows of three views --------------------------

    fn central_views(&mut self, ui: &mut egui::Ui) {
        let backdrop = backdrop_color(ui.visuals());
        egui::CentralPanel::default_margins()
            .frame(egui::Frame::NONE.fill(backdrop))
            .show(ui, |ui| {
                if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
                    self.empty_state(ui);
                    return;
                }
                // Maximized single-view layout: one view fills the window.
                if let Some((mslot, midx)) = self.maximized {
                    if self.slots[mslot.min(1)].study.is_some() && midx < 3 {
                        let full = ui.available_rect_before_wrap();
                        self.view_cell(ui, mslot.min(1), midx, full);
                        return;
                    }
                    self.maximized = None;
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
        let col_w = (row_rect.width() - 2.0 * gap) / 3.0;
        for idx in 0..3 {
            let x0 = row_rect.left() + idx as f32 * (col_w + gap);
            let col = Rect::from_min_size(
                Pos2::new(x0, row_rect.top()),
                Vec2::new(col_w, row_rect.height()),
            );
            self.view_cell(ui, slot, idx, col);
        }
    }

    /// One viewport plus its slice slider inside `cell` (used both by the
    /// three-in-a-row layout and by the maximized single-view layout).
    fn view_cell(&mut self, ui: &mut egui::Ui, slot: usize, idx: usize, cell: Rect) {
        let slider_h = 26.0;
        let view_rect =
            Rect::from_min_max(cell.min, Pos2::new(cell.max.x, cell.max.y - slider_h));
        let slider_rect = Rect::from_min_max(
            Pos2::new(cell.min.x + 6.0, cell.max.y - slider_h + 2.0),
            Pos2::new(cell.max.x - 6.0, cell.max.y - 2.0),
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

    fn empty_row(&mut self, ui: &mut egui::Ui, slot: usize, rect: Rect) {
        // `text_color`, not `weak_text_color`: the dimmed variant drops below a
        // readable contrast on the dark row fill (see `theme_tests`).
        let (fill, hint, strong) = {
            let v = ui.visuals();
            (empty_row_color(v), v.text_color(), v.strong_text_color())
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, fill);
        painter.text(
            rect.center() - Vec2::new(0.0, 24.0),
            Align2::CENTER_CENTER,
            format!("No dataset {}", SLOT_NAMES[slot]),
            FontId::proportional(15.0),
            hint,
        );
        let btn_rect = Rect::from_center_size(
            rect.center() + Vec2::new(0.0, 10.0),
            Vec2::new(220.0, 28.0),
        );
        if ui
            .put(btn_rect, egui::Button::new("📂 Add DICOM folder…"))
            .clicked()
        {
            if let Some(dir) = Self::pick_folder("Select DICOM folder to add to dataset B") {
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
                    strong,
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
                } else if let Some(job) = &self.gen_job {
                    ui.spinner();
                    ui.label(format!("Generating test data — {}", job.progress.get()));
                } else {
                    ui.label("Add a folder containing DICOM data");
                    ui.add_space(8.0);
                    if ui.button("📂 Add DICOM folder…").clicked() {
                        if let Some(dir) = Self::pick_folder("Select a DICOM folder") {
                            self.start_load(0, dir);
                        }
                    }
                    ui.add_space(12.0);
                    ui.weak("…or create a synthetic RT study to try the viewer on");
                    ui.add_space(4.0);
                    if ui
                        .button("🧪 Generate test data…")
                        .on_hover_text(
                            "Writes a synthetic CT + RTSTRUCT + RTPLAN + RTDOSE study \
                             into the application folder",
                        )
                        .clicked()
                    {
                        self.gen_open = true;
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

        // Painted segmentations (and the live region-growing preview).
        if let Some(tex) = &view.seg_tex {
            painter.image(
                tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
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
            if let Some(ss) = slot_state.active_structures() {
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

        // Brush cursor and region-growing readout.
        if self.seg_tool != SegTool::None {
            let hover = ui.input(|i| i.pointer.hover_pos()).filter(|p| rect.contains(*p));
            match self.seg_tool {
                SegTool::Brush | SegTool::Erase => {
                    if let Some(mp) = hover {
                        let erase = self.seg_tool == SegTool::Erase
                            || ui.input(|i| i.modifiers.alt);
                        let col = if erase {
                            Color32::from_rgba_unmultiplied(255, 90, 90, 200)
                        } else {
                            let c = slot_state
                                .segs
                                .get(slot_state.active_seg)
                                .map(|s| s.color)
                                .unwrap_or([140, 255, 140]);
                            Color32::from_rgba_unmultiplied(c[0], c[1], c[2], 220)
                        };
                        painter.circle_stroke(
                            mp,
                            self.brush_radius_mm * zoom,
                            Stroke::new(1.5, col),
                        );
                    }
                }
                SegTool::Grow => {
                    if let Some(mp) = hover {
                        let s = Stroke::new(
                            1.5,
                            Color32::from_rgba_unmultiplied(255, 220, 0, 220),
                        );
                        painter.line_segment(
                            [mp + Vec2::new(-6.0, 0.0), mp + Vec2::new(6.0, 0.0)],
                            s,
                        );
                        painter.line_segment(
                            [mp + Vec2::new(0.0, -6.0), mp + Vec2::new(0.0, 6.0)],
                            s,
                        );
                    }
                    if let Some(g) = self.grow.as_ref().filter(|g| g.slot == slot) {
                        let vv = vol.spacing[0] * vol.spacing[1] * vol.spacing[2] / 1000.0;
                        painter.text(
                            rect.left_bottom() + Vec2::new(6.0, -22.0),
                            Align2::LEFT_BOTTOM,
                            format!(
                                "grow {:.1} cm³ · reach ×{:.2}{}",
                                self.grow_state.voxels.len() as f64 * vv,
                                g.level,
                                if g.capped { " · CAPPED" } else { "" }
                            ),
                            FontId::proportional(13.0),
                            Color32::from_rgb(255, 220, 0),
                        );
                    }
                }
                SegTool::None => {}
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

        // ---- corner buttons: reset view & maximize / restore layout ----
        // Registered before the viewport interaction; the viewport handlers
        // below additionally ignore any pointer activity over the buttons.
        let is_max = self.maximized == Some((slot, idx));
        let bsize = egui::vec2(24.0, 20.0);
        let by = rect.top() + 22.0; // below the slice counter
        let max_rect =
            Rect::from_min_size(Pos2::new(rect.right() - bsize.x - 4.0, by), bsize);
        let fit_rect =
            Rect::from_min_size(Pos2::new(max_rect.left() - bsize.x - 4.0, by), bsize);
        let max_resp = ui
            .put(
                max_rect,
                egui::Button::new(if is_max { "❐" } else { "⛶" }).small(),
            )
            .on_hover_text(if is_max {
                "Restore the view layout"
            } else {
                "Maximize this view (whole window)"
            });
        let fit_resp = ui
            .put(fit_rect, egui::Button::new("⟲").small())
            .on_hover_text("Reset view (fit zoom, center)");
        let (pointer_pos, any_click) = ui.input(|i| (i.pointer.interact_pos(), i.pointer.any_click()));
        let over_buttons = pointer_pos
            .map(|p| max_rect.contains(p) || fit_rect.contains(p))
            .unwrap_or(false);
        let clicked_max = max_resp.clicked()
            || (any_click && pointer_pos.map(|p| max_rect.contains(p)).unwrap_or(false));
        let clicked_fit = fit_resp.clicked()
            || (any_click && pointer_pos.map(|p| fit_rect.contains(p)).unwrap_or(false));
        // (applied below, in the mutable phase)

        // ---- interaction ----
        let resp = ui.interact(
            rect,
            egui::Id::new(("viewport", slot, idx)),
            Sense::click_and_drag(),
        );
        let n_slices = vol.plane_slice_count(plane);
        let seg_active = self.seg_tool != SegTool::None;
        let cur_slice = view.slice;

        let mut new_slice = None;
        let mut new_zoom = None;
        let mut new_pan = None;
        let mut new_cursor = None;
        let mut wl_delta = None;
        let mut reset_view = false;
        let mut new_accum = None;
        let mut new_brush = None;

        if resp.hovered() {
            let (wheel_lines, brush_lines, zoom_delta, pointer) = ui.input(|i| {
                let mut lines = 0.0f32;
                let mut brush = 0.0f32;
                for e in &i.events {
                    if let egui::Event::MouseWheel { unit, delta, modifiers, .. } = e {
                        let scale = match unit {
                            egui::MouseWheelUnit::Line => 1.0,
                            egui::MouseWheelUnit::Point => 1.0 / 40.0,
                            egui::MouseWheelUnit::Page => 10.0,
                        };
                        if modifiers.shift && seg_active {
                            // Some platforms report shift+wheel horizontally.
                            brush += (delta.y + delta.x) * scale;
                        } else if !(modifiers.ctrl || modifiers.command) {
                            lines += delta.y * scale;
                        }
                    }
                }
                (lines, brush, i.zoom_delta(), i.pointer.hover_pos())
            });
            if brush_lines.abs() > 0.0 {
                new_brush = Some(
                    (self.brush_radius_mm * (brush_lines * 0.12).exp()).clamp(0.5, 80.0),
                );
            }
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

        // Left-click crosshair navigation only while the crosshair is shown
        // and no segmentation tool holds the left button; with ⌖ off, slices
        // change only by scrolling the hovered view.
        if self.show_crosshair
            && !seg_active
            && (resp.dragged_by(egui::PointerButton::Primary) || resp.clicked())
            && !over_buttons
        {
            if let Some(mp) = resp.interact_pointer_pos() {
                let px = screen_to_px(mp);
                let vxl = vol.plane_pixel_to_voxel(plane, view.slice, px[0] as f64, px[1] as f64);
                new_cursor = Some(vxl);
            }
        }

        // Segmentation tools on the left button.
        let mut paint_to: Option<([f64; 3], bool)> = None;
        let mut paint_done = false;
        let mut grow_start: Option<([f64; 3], f32)> = None;
        let mut grow_move: Option<f32> = None;
        let mut grow_done = false;
        if seg_active && !over_buttons {
            let to_voxel = |mp: Pos2| {
                let px = screen_to_px(mp);
                vol.plane_pixel_to_voxel(plane, cur_slice, px[0] as f64, px[1] as f64)
            };
            match self.seg_tool {
                SegTool::Brush | SegTool::Erase => {
                    if resp.dragged_by(egui::PointerButton::Primary) || resp.clicked() {
                        if let Some(mp) = resp.interact_pointer_pos() {
                            let erase = self.seg_tool == SegTool::Erase
                                || ui.input(|i| i.modifiers.alt);
                            paint_to = Some((to_voxel(mp), erase));
                        }
                    }
                    if resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked() {
                        paint_done = true;
                    }
                }
                SegTool::Grow => {
                    if resp.drag_started_by(egui::PointerButton::Primary)
                        || (resp.clicked() && self.grow.is_none())
                    {
                        if let Some(mp) = resp.interact_pointer_pos() {
                            grow_start = Some((to_voxel(mp), mp.y));
                        }
                    } else if resp.dragged_by(egui::PointerButton::Primary) {
                        if let Some(mp) = resp.interact_pointer_pos() {
                            grow_move = Some(mp.y);
                        }
                    }
                    if resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked() {
                        grow_done = true;
                    }
                }
                SegTool::None => {}
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
        let hovered = resp.hovered();

        // Apply interactions (mutable phase).
        if clicked_max {
            self.maximized = if is_max { None } else { Some((slot, idx)) };
        }
        if clicked_fit {
            reset_view = true;
        }
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
            // Also return to the default (central) slice.
            self.slots[slot].views[idx].slice = n_slices / 2;
        }
        if let Some((dx, dy)) = wl_delta {
            self.window_width = (self.window_width * (1.0 + dx * 0.005)).clamp(1.0, 30000.0);
            self.window_center += dy * self.window_width * 0.002;
        }
        if let Some(c) = new_cursor {
            self.set_cursor(slot, c, idx);
        }
        if let Some(r) = new_brush {
            self.brush_radius_mm = r;
        }
        if let Some((vxl, erase)) = paint_to {
            self.apply_brush(slot, plane, cur_slice, vxl, erase);
        }
        if paint_done {
            self.end_paint_stroke(slot);
        }
        if let Some((seed, y)) = grow_start {
            self.begin_grow(slot, seed, y);
        }
        if let Some(y) = grow_move {
            self.update_grow(y);
        }
        if grow_done {
            self.commit_grow();
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
        for (i, out) in new_slices.iter_mut().enumerate() {
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
            *out = Some((sc.round().max(0.0) as usize).min(max));
        }
        for (view, s) in self.slots[slot].views.iter_mut().zip(new_slices) {
            if let Some(s) = s {
                view.slice = s;
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
        let seg_hash = self.seg_overlay_hash(slot);
        let grow_here = self.grow.as_ref().is_some_and(|g| g.slot == slot);
        let wc = self.window_center;
        let ww = self.window_width;
        let dose_on = self.dose_mode != DoseMode::Off;
        let contours_on = self.show_contours;

        let StudySlot {
            study,
            views,
            roi_visible,
            active_structs,
            active_dose,
            dose_reference,
            segs,
            ..
        } = &mut self.slots[slot];
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
            let mut gray = Vec::new();
            vol.extract_slice(plane, slice, &mut slice_buf);
            render::slice_to_gray(&slice_buf, wc, ww, w, &mut gray);
            // Moved straight into the texture: keeping a second copy around
            // as a scratch buffer only bought an extra full-image memcpy.
            let img = ColorImage::new([w, h], gray);
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
                let mut dose_rgba = Vec::new();
                render::sample_dose_plane(vol, dose, plane, slice, &mut dose_plane);
                render::dose_colorwash(
                    &dose_plane,
                    reference,
                    self.dose_threshold_pct / 100.0,
                    self.dose_opacity,
                    &mut dose_rgba,
                );
                let img = ColorImage::new([w, h], dose_rgba);
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
                // Isodose segments — one marching-squares pass per enabled
                // level, and the levels are independent.
                let levels: Vec<(usize, f32)> = self
                    .iso_levels
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.on)
                    .map(|(li, l)| (li, l.pct / 100.0 * reference))
                    .filter(|&(_, abs)| abs > 0.0)
                    .collect();
                let per_level: Vec<Vec<(usize, render::Segment)>> = levels
                    .par_iter()
                    .map(|&(li, abs)| {
                        render::marching_squares(&dose_plane, w, h, abs)
                            .into_iter()
                            .map(|seg| (li, seg))
                            .collect()
                    })
                    .collect();
                view.iso_segs.clear();
                view.iso_segs.extend(per_level.into_iter().flatten());
                view.dose_plane = dose_plane;
                view.dose_key = Some(dose_key);
            }
        }

        // Contours.
        if contours_on {
            if let Some(ss) = study.structure_sets.get(*active_structs) {
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

        // Segmentation overlay: all visible masks plus the live
        // region-growing preview, blended into one RGBA texture. NEAREST
        // filtering keeps the voxel raster crisp while editing.
        if !segs.is_empty() || grow_here {
            let skey =
                seg_hash.wrapping_add((slice as u64).wrapping_mul(0x2545F4914F6CDD1D));
            if views[idx].seg_key != Some(skey) {
                let mut rgba = vec![Color32::TRANSPARENT; w * h];
                for seg in segs.iter().filter(|s| s.visible) {
                    segmentation::overlay_slice(
                        &seg.mask, vol.dims, plane, slice, seg.color, 90, &mut rgba,
                    );
                }
                let n = vol.dims[0] * vol.dims[1] * vol.dims[2];
                if grow_here && self.grow_preview.len() == n {
                    segmentation::overlay_slice(
                        &self.grow_preview,
                        vol.dims,
                        plane,
                        slice,
                        [255, 220, 0],
                        150,
                        &mut rgba,
                    );
                }
                let img = ColorImage::new([w, h], rgba);
                let view = &mut views[idx];
                match &mut view.seg_tex {
                    Some(t) => t.set(img, TextureOptions::NEAREST),
                    None => {
                        view.seg_tex = Some(ctx.load_texture(
                            format!("seg{slot}_{idx}"),
                            img,
                            TextureOptions::NEAREST,
                        ))
                    }
                }
                view.seg_key = Some(skey);
            }
        } else if views[idx].seg_tex.is_some() {
            views[idx].seg_tex = None;
            views[idx].seg_key = None;
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

    // -- Floating planar image viewers -------------------------------------

    fn planar_windows_ui(&mut self, ctx: &egui::Context) {
        let mut windows = std::mem::take(&mut self.planar_windows);
        for w in &mut windows {
            let Some(study) = &self.slots[w.slot].study else {
                w.open = false;
                continue;
            };
            let Some(img) = study.planar_images.get(w.idx) else {
                w.open = false;
                continue;
            };
            if !w.open {
                continue;
            }

            // Rebuild the texture when W/L changed.
            if w.tex.is_none() || w.tex_wl != w.wl {
                let lo = w.wl.0 - w.wl.1.max(1.0) * 0.5;
                let scale = 255.0 / w.wl.1.max(1.0);
                let pixels: Vec<Color32> = img
                    .data
                    .iter()
                    .map(|&v| {
                        let g = ((v - lo) * scale).clamp(0.0, 255.0) as u8;
                        Color32::from_gray(g)
                    })
                    .collect();
                let ci = ColorImage::new([img.cols, img.rows], pixels);
                match &mut w.tex {
                    Some(t) => t.set(ci, TextureOptions::LINEAR),
                    None => {
                        w.tex = Some(ctx.load_texture(
                            format!("planar{}_{}", w.slot, w.idx),
                            ci,
                            TextureOptions::LINEAR,
                        ))
                    }
                }
                w.tex_wl = w.wl;
            }

            let title = format!(
                "{}: {} [{}]",
                SLOT_NAMES[w.slot], img.label, img.modality
            );
            let mut open = w.open;
            egui::Window::new(title)
                .id(egui::Id::new(("planar_win", w.slot, w.idx)))
                .open(&mut open)
                .default_size([560.0, 640.0])
                .resizable(true)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("W/L:");
                        ui.add(
                            egui::DragValue::new(&mut w.wl.0).speed(4.0).prefix("C "),
                        );
                        ui.add(
                            egui::DragValue::new(&mut w.wl.1)
                                .speed(8.0)
                                .range(1.0..=1.0e6)
                                .prefix("W "),
                        );
                        if ui.small_button("Auto").clicked() {
                            w.wl = (
                                (img.min_value + img.max_value) * 0.5,
                                (img.max_value - img.min_value).max(1.0),
                            );
                        }
                    });
                    // Physical aspect ratio, fitted to the available width.
                    let w_mm = (img.cols as f64 * img.spacing[0]) as f32;
                    let h_mm = (img.rows as f64 * img.spacing[1]) as f32;
                    let avail = ui.available_width().max(64.0);
                    let scale = (avail / w_mm).min(520.0 / h_mm.max(1.0));
                    if let Some(tex) = &w.tex {
                        // Same interactive window/level as the CT views:
                        // right-drag, x = width, y = center.
                        let resp = ui.add(
                            egui::Image::new(egui::load::SizedTexture::new(
                                tex.id(),
                                egui::vec2(w_mm * scale, h_mm * scale),
                            ))
                            .sense(Sense::click_and_drag()),
                        );
                        if resp.dragged_by(egui::PointerButton::Secondary) {
                            let d = resp.drag_delta();
                            w.wl.1 = (w.wl.1 * (1.0 + d.x * 0.005)).clamp(1.0, 1.0e6);
                            w.wl.0 += d.y * w.wl.1 * 0.002;
                        }
                        resp.on_hover_text("Right-drag: window/level (x = width, y = center)");
                    }
                    for (k, v) in &img.info {
                        ui.weak(format!("{k}: {v}"));
                    }
                });
            w.open = open;
        }
        windows.retain(|w| w.open);
        self.planar_windows = windows;
    }

    // -- 3D structure windows (render) --------------------------------------

    fn d3_windows_ui(&mut self, ctx: &egui::Context) {
        let mut windows = std::mem::take(&mut self.d3_windows);
        for w in &mut windows {
            if !w.open {
                continue;
            }
            // Poll mesh building.
            {
                let mut err = None;
                if let Some(meshes) = poll_job(&mut w.job, ctx, "Meshing", &mut err) {
                    {
                        // Scene bounding sphere for auto-fit.
                        let (mut mn, mut mx) = ([f32::MAX; 3], [f32::MIN; 3]);
                        for m in &meshes {
                            for v in &m.verts {
                                for a in 0..3 {
                                    mn[a] = mn[a].min(v[a]);
                                    mx[a] = mx[a].max(v[a]);
                                }
                            }
                        }
                        if mn[0] < mx[0] {
                            w.center = [
                                (mn[0] + mx[0]) * 0.5,
                                (mn[1] + mx[1]) * 0.5,
                                (mn[2] + mx[2]) * 0.5,
                            ];
                            w.radius = (0..3)
                                .map(|a| (mx[a] - mn[a]) * 0.5)
                                .fold(0.0f32, |acc, v| (acc * acc + v * v).sqrt())
                                .max(10.0);
                        }
                        w.meshes = Some(Arc::new(meshes));
                    }
                }
                self.error = self.error.take().or(err);
            }

            // Live segmentation meshes: rebuilt in the background whenever a
            // mask changes (one build in flight; a newer state simply spawns
            // the next build once the current one lands), so painting shows
            // up in 3D essentially in real time.
            {
                let mut err = None;
                if let Some(m) = poll_job(&mut w.seg_job, ctx, "Segmentation meshing", &mut err)
                {
                    w.seg_meshes = Some(Arc::new(m));
                }
                self.error = self.error.take().or(err);
                let hash = self.seg_mesh_hash(w.slot);
                if w.seg_job.is_none() && w.seg_built != hash {
                    w.seg_built = hash;
                    if let Some(study) = &self.slots[w.slot].study {
                        let geom = GridGeom::of(&study.volume);
                        let snaps: Vec<_> = self.slots[w.slot]
                            .segs
                            .iter()
                            .enumerate()
                            .filter_map(|(i, s)| {
                                s.mesh_grid().map(|g| (i, s.name.clone(), s.color, g))
                            })
                            .collect();
                        if snaps.is_empty() {
                            w.seg_meshes = Some(Arc::new(Vec::new()));
                        } else {
                            let progress = Arc::new(Progress::default());
                            let (tx, rx) = mpsc::channel();
                            std::thread::spawn(move || {
                                let meshes: Vec<RoiMesh> = snaps
                                    .into_par_iter()
                                    .filter_map(|(i, name, color, (grid, gdims, lo, stride))| {
                                        mesh3d::mesh_from_mask(&grid, gdims, lo, stride, &geom)
                                            .map(|(verts, normals, tris)| RoiMesh {
                                                roi_index: i,
                                                name,
                                                color,
                                                external: false,
                                                verts,
                                                normals,
                                                tris,
                                            })
                                    })
                                    .collect();
                                let _ = tx.send(meshes);
                            });
                            w.seg_job = Some(Job { progress, rx });
                        }
                    }
                }
            }

            let visible: &[bool] = &self.slots[w.slot].roi_visible;
            // Snapshot of segmentation display state (visibility + live color).
            let seg_disp: Vec<(bool, [u8; 3])> = self.slots[w.slot]
                .segs
                .iter()
                .map(|s| (s.visible, s.color))
                .collect();
            let title = format!("3D structures — dataset {}", SLOT_NAMES[w.slot]);
            let mut open = w.open;
            egui::Window::new(title)
                .id(egui::Id::new(("d3_win", w.slot)))
                .open(&mut open)
                .default_size([640.0, 700.0])
                .resizable(true)
                .show(ctx, |ui| {
                    if let Some(job) = &w.job {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(job.progress.get());
                        });
                        return;
                    }
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Slider::new(&mut w.opacity, 0.2..=1.0).text("Opacity"),
                        );
                        if ui.small_button("⟲ Reset view").clicked() {
                            w.yaw = 0.7;
                            w.pitch = -0.5;
                            w.zoom = 1.0;
                            w.pan = Vec2::ZERO;
                        }
                        ui.weak("drag rotate · wheel zoom · middle-drag pan");
                    });

                    let avail = ui.available_size();
                    let size = Vec2::new(avail.x.max(240.0), avail.y.max(240.0));
                    let (resp, painter) =
                        ui.allocate_painter(size, Sense::click_and_drag());
                    let rect = resp.rect;
                    painter.rect_filled(rect, 0.0, Color32::BLACK);

                    // Interaction.
                    if resp.dragged_by(egui::PointerButton::Primary) {
                        let d = resp.drag_delta();
                        w.yaw += d.x * 0.01;
                        w.pitch = (w.pitch + d.y * 0.01).clamp(-1.55, 1.55);
                    }
                    if resp.dragged_by(egui::PointerButton::Middle) {
                        w.pan += resp.drag_delta();
                    }
                    if resp.hovered() {
                        let (lines, zd) = ui.input(|i| {
                            let mut l = 0.0f32;
                            for e in &i.events {
                                if let egui::Event::MouseWheel { unit, delta, .. } = e {
                                    l += match unit {
                                        egui::MouseWheelUnit::Line => delta.y,
                                        egui::MouseWheelUnit::Point => delta.y / 40.0,
                                        egui::MouseWheelUnit::Page => delta.y * 10.0,
                                    };
                                }
                            }
                            (l, i.zoom_delta())
                        });
                        w.zoom =
                            (w.zoom * (lines * 0.12).exp() * zd).clamp(0.1, 40.0);
                    }

                    // Render.
                    let Some(meshes) = &w.meshes else { return };
                    let seg_meshes = w.seg_meshes.clone();
                    let n_seg = seg_meshes.as_ref().map(|m| m.len()).unwrap_or(0);
                    if meshes.is_empty() && n_seg == 0 {
                        painter.text(
                            rect.center(),
                            Align2::CENTER_CENTER,
                            if w.seg_job.is_some() {
                                "Meshing segmentation…"
                            } else {
                                "No meshable structures"
                            },
                            FontId::proportional(14.0),
                            Color32::GRAY,
                        );
                        return;
                    }
                    let scale = 0.45 * rect.width().min(rect.height()) / w.radius * w.zoom;
                    let cx = rect.center().x + w.pan.x;
                    let cyc = rect.center().y + w.pan.y;
                    let alpha = (w.opacity * 255.0) as u8;

                    // What the cached geometry depends on. Orientation and
                    // visibility fix the draw order; the rest only moves the
                    // already-ordered triangles around on screen.
                    let mut order_key = mix(0x243F6A8885A308D3, Arc::as_ptr(meshes) as u64);
                    order_key = mix(order_key, w.yaw.to_bits() as u64);
                    order_key = mix(order_key, w.pitch.to_bits() as u64);
                    for m in meshes.iter() {
                        let on = visible.get(m.roi_index).copied().unwrap_or(true);
                        order_key = mix(order_key, on as u64);
                    }
                    if let Some(sm) = &seg_meshes {
                        order_key = mix(order_key, Arc::as_ptr(sm) as u64);
                        for m in sm.iter() {
                            let on = seg_disp.get(m.roi_index).map(|d| d.0).unwrap_or(false);
                            order_key = mix(order_key, on as u64);
                        }
                    }
                    let mut vertex_key = mix(order_key, scale.to_bits() as u64);
                    vertex_key = mix(vertex_key, cx.to_bits() as u64);
                    vertex_key = mix(vertex_key, cyc.to_bits() as u64);
                    vertex_key = mix(vertex_key, alpha as u64);
                    // Segmentation colors are applied live at draw time.
                    for (_, c) in &seg_disp {
                        vertex_key = mix(
                            vertex_key,
                            (c[0] as u64) | ((c[1] as u64) << 8) | ((c[2] as u64) << 16),
                        );
                    }

                    if w.frame.vertex_key != Some(vertex_key) {
                        let (sy, cy) = w.yaw.sin_cos();
                        let (sp, cp) = w.pitch.sin_cos();
                        let c = w.center;
                        // Yaw about patient z, then pitch about the screen x.
                        let rot = |p: [f32; 3], centered: bool| -> [f32; 3] {
                            let (x, y, z) = if centered {
                                (p[0] - c[0], p[1] - c[1], p[2] - c[2])
                            } else {
                                (p[0], p[1], p[2])
                            };
                            let x1 = cy * x - sy * y;
                            let y1 = sy * x + cy * y;
                            let y2 = cp * y1 - sp * z;
                            let z2 = sp * y1 + cp * z;
                            [x1, y2, z2]
                        };
                        let reorder = w.frame.order_key != Some(order_key);
                        let f = &mut w.frame;
                        // Buffers are reused across frames; `make_mut` hands
                        // back the previous allocation because the painter has
                        // already dropped last frame's reference.
                        let mesh = Arc::make_mut(&mut f.mesh);
                        mesh.vertices.clear();
                        f.depth.clear();
                        if reorder {
                            f.tris.clear();
                        }
                        let entries = meshes
                            .iter()
                            .map(|m| {
                                (
                                    m,
                                    visible.get(m.roi_index).copied().unwrap_or(true),
                                    m.color,
                                    m.external,
                                )
                            })
                            .chain(seg_meshes.iter().flat_map(|a| a.iter()).map(|m| {
                                let (on, c) =
                                    seg_disp.get(m.roi_index).copied().unwrap_or((false, m.color));
                                (m, on, c, false)
                            }));
                        for (m, on, color, external) in entries {
                            if !on {
                                continue;
                            }
                            let base = mesh.vertices.len() as u32;
                            // External/body contours render translucent so the
                            // interior structures remain visible.
                            let roi_alpha = if external {
                                (alpha as f32 * 0.22) as u8
                            } else {
                                alpha
                            };
                            for (v, n) in m.verts.iter().zip(m.normals.iter()) {
                                let t = rot(*v, true);
                                let nn = rot(*n, false);
                                // Headlight along the view axis, two-sided.
                                let inten = 0.30 + 0.70 * nn[1].abs();
                                let col = Color32::from_rgba_unmultiplied(
                                    (color[0] as f32 * inten) as u8,
                                    (color[1] as f32 * inten) as u8,
                                    (color[2] as f32 * inten) as u8,
                                    roi_alpha,
                                );
                                mesh.vertices.push(egui::epaint::Vertex {
                                    pos: Pos2::new(cx + t[0] * scale, cyc - t[2] * scale),
                                    uv: egui::epaint::WHITE_UV,
                                    color: col,
                                });
                                f.depth.push(t[1]);
                            }
                            if reorder {
                                f.tris.extend(
                                    m.tris
                                        .iter()
                                        .map(|t| [base + t[0], base + t[1], base + t[2]]),
                                );
                            }
                        }

                        if reorder {
                            // Painter's algorithm: far triangles first (viewer
                            // at -y). Packing the depth into the high half of a
                            // u64 lets this be a primitive sort rather than a
                            // float comparator over a tuple.
                            f.order.clear();
                            f.order.extend(f.tris.iter().enumerate().map(|(i, t)| {
                                let d = (f.depth[t[0] as usize]
                                    + f.depth[t[1] as usize]
                                    + f.depth[t[2] as usize])
                                    / 3.0;
                                ((!depth_key(d) as u64) << 32) | i as u64
                            }));
                            f.order.par_sort_unstable();
                            mesh.indices.clear();
                            mesh.indices.reserve(f.order.len() * 3);
                            for &o in &f.order {
                                mesh.indices.extend_from_slice(&f.tris[(o & 0xFFFF_FFFF) as usize]);
                            }
                            f.order_key = Some(order_key);
                        }
                        f.vertex_key = Some(vertex_key);
                    }

                    painter.add(egui::Shape::Mesh(w.frame.mesh.clone()));
                    painter.text(
                        rect.left_bottom() + Vec2::new(6.0, -6.0),
                        Align2::LEFT_BOTTOM,
                        format!(
                            "{} structure(s), {} triangles",
                            meshes.len() + n_seg,
                            w.frame.tris.len()
                        ),
                        FontId::proportional(11.0),
                        Color32::GRAY,
                    );
                });
            w.open = open;
        }
        windows.retain(|w| w.open);
        self.d3_windows = windows;
    }

    // -- Modals -----------------------------------------------------------

    fn modals(&mut self, ctx: &egui::Context) {
        self.generator_window(ctx);
        self.anonymize_window(ctx);
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

    /// Built-in synthetic test-data generator (the Rust replacement for the
    /// old `tools/generate_test_data.py`): pick an output folder, tweak the
    /// phantom parameters and write a complete RT study.
    fn generator_window(&mut self, ctx: &egui::Context) {
        if !self.gen_open {
            return;
        }
        let running = self.gen_job.is_some();
        let mut open = true;
        let mut do_generate = false;
        let mut browse = false;
        let mut reset_dir = false;
        let mut reset_params = false;

        egui::Window::new("🧪 Generate synthetic RT test study")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_max_width(560.0);
                ui.label(
                    "Writes a self-contained test study: 40-slice CT water phantom with a \
                     spherical target and a cord, matching RTSTRUCT contours, a Gaussian \
                     RTDOSE and a two-beam proton RTPLAN.",
                );
                ui.add_space(6.0);

                ui.label(egui::RichText::new("Output folder").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.gen_dir)
                            .desired_width(360.0)
                            .hint_text("folder to write the DICOM files into"),
                    );
                    if ui.button("📂 Browse…").clicked() {
                        browse = true;
                    }
                    if ui
                        .button("↺")
                        .on_hover_text("Reset to the application folder")
                        .clicked()
                    {
                        reset_dir = true;
                    }
                });
                ui.weak(format!("Files: {}", gen_test_data::output_summary(&self.gen_params)));

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Phantom").strong());
                egui::Grid::new("gen_params_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Dose peak (Gy)")
                            .on_hover_text("Also written as the plan's prescription dose");
                        ui.add(
                            egui::DragValue::new(&mut self.gen_params.peak)
                                .speed(0.5)
                                .range(0.1..=200.0),
                        );
                        ui.end_row();

                        ui.label("Target Y shift (mm)").on_hover_text(
                            "Moves the target sphere and the dose peak inside the phantom",
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.gen_params.target_shift_y)
                                .speed(0.5)
                                .range(-60.0..=60.0),
                        );
                        ui.end_row();

                        ui.label("Phantom shift X / Y (mm)")
                            .on_hover_text("Shifts the whole phantom — for registration tests");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.gen_params.shift_x)
                                    .speed(0.5)
                                    .range(-60.0..=60.0),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.gen_params.shift_y)
                                    .speed(0.5)
                                    .range(-60.0..=60.0),
                            );
                        });
                        ui.end_row();

                        ui.label("Plan label");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.gen_params.plan_label)
                                .desired_width(160.0)
                                .char_limit(16),
                        );
                        ui.end_row();

                        ui.label("REG translation (mm)").on_hover_text(
                            "Translation written into the REG object's second matrix",
                        );
                        ui.horizontal(|ui| {
                            for v in &mut self.gen_params.reg_shift {
                                ui.add(
                                    egui::DragValue::new(v).speed(0.5).range(-100.0..=100.0),
                                );
                            }
                        });
                        ui.end_row();
                    });
                ui.checkbox(
                    &mut self.gen_params.extras,
                    "Also write DX, RTIMAGE, REG and RTRECORD objects",
                );
                ui.checkbox(&mut self.gen_load_after, "Load the study into slot A when done");

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if running {
                        ui.spinner();
                        if let Some(job) = &self.gen_job {
                            ui.label(job.progress.get());
                        }
                    } else {
                        if ui
                            .add(egui::Button::new("⚙ Generate"))
                            .on_hover_text("Existing files with the same names are overwritten")
                            .clicked()
                        {
                            do_generate = true;
                        }
                        if ui.button("Defaults").clicked() {
                            reset_params = true;
                        }
                    }
                });
                if let Some(msg) = &self.gen_result {
                    ui.add_space(4.0);
                    ui.label(msg);
                }
            });

        self.gen_open = open;
        if browse {
            if let Some(dir) = Self::pick_folder("Select an output folder for the test data") {
                self.gen_dir = dir.display().to_string();
            }
        }
        if reset_dir {
            self.gen_dir = gen_test_data::default_output_dir().display().to_string();
        }
        if reset_params {
            self.gen_params = GenParams::default();
        }
        if do_generate {
            self.start_generate();
        }
    }

    // -- Tools ▶ Anonymize DICOM folder ------------------------------------

    fn anon_start_scan(&mut self) {
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

    fn anon_start_apply(&mut self) {
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
                    self.error = Some(
                        "Choose an output folder, or tick “overwrite in place”".into(),
                    );
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

    /// The anonymizer tool window: pick a folder, scan it, review every
    /// identifying tag (current values, proposed replacement — editable),
    /// then rewrite the files.
    fn anonymize_window(&mut self, ctx: &egui::Context) {
        if !self.anon_open {
            return;
        }
        // Poll background jobs.
        match poll_job(&mut self.anon_scan_job, ctx, "Scan", &mut self.error) {
            Some(Ok(mut scan)) => {
                if self.anon_out.trim().is_empty() {
                    self.anon_out = format!("{}_anon", scan.root.display());
                }
                scan.warnings.truncate(8);
                self.anon_scan = Some(scan);
            }
            Some(Err(e)) => self.error = Some(format!("{e:#}")),
            None => {}
        }
        match poll_job(&mut self.anon_apply_job, ctx, "Anonymize", &mut self.error) {
            Some(Ok(n)) => {
                let dest = if self.anon_in_place {
                    "in place".to_string()
                } else {
                    format!("into {}", self.anon_out.trim())
                };
                self.anon_result = Some(format!("✔ {n} file(s) anonymized {dest}"));
            }
            Some(Err(e)) => self.error = Some(format!("{e:#}")),
            None => {}
        }

        let busy = self.anon_scan_job.is_some() || self.anon_apply_job.is_some();
        let mut open = true;
        let mut browse_in = false;
        let mut browse_out = false;
        let mut do_scan = false;
        let mut do_apply = false;

        egui::Window::new("🔏 Anonymize DICOM folder")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([780.0, 560.0])
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(
                    "Scans a folder, shows every identifying tag with its current values \
                     and a proposed replacement (editable), then rewrites the files. \
                     UIDs are regenerated consistently across all files, so series, \
                     structure-set, plan and dose references stay linked.",
                );
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("DICOM folder").strong());
                    ui.add(
                        egui::TextEdit::singleline(&mut self.anon_dir)
                            .desired_width(420.0)
                            .hint_text("folder to scan (recursively)"),
                    );
                    if ui.button("📂 Browse…").clicked() {
                        browse_in = true;
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("🔍 Scan"))
                        .clicked()
                    {
                        do_scan = true;
                    }
                });

                if let Some(job) = &self.anon_scan_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                }

                if let Some(scan) = &mut self.anon_scan {
                    let mods = scan
                        .modalities
                        .iter()
                        .map(|(m, n)| format!("{m} {n}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.weak(format!(
                        "{} DICOM file(s) ({mods}) — {} unique UID(s), {} private element(s)",
                        scan.files.len(),
                        scan.uid_count,
                        scan.private_count
                    ));
                    for w in &scan.warnings {
                        ui.colored_label(warn_color(ui.visuals()), format!("⚠ {w}"));
                    }
                    ui.add_space(4.0);

                    // ---- findings table ----
                    egui::ScrollArea::vertical()
                        .max_height(ui.available_height() - 130.0)
                        .show(ui, |ui| {
                            egui::Grid::new("anon_grid")
                                .num_columns(3)
                                .striped(true)
                                .spacing([10.0, 3.0])
                                .show(ui, |ui| {
                                    ui.label(egui::RichText::new("Tag").strong());
                                    ui.label(egui::RichText::new("Current value(s)").strong());
                                    ui.label(egui::RichText::new("New value").strong());
                                    ui.end_row();
                                    for f in &mut scan.findings {
                                        let label = format!(
                                            "({:04X},{:04X}) {}",
                                            f.tag.group(),
                                            f.tag.element(),
                                            f.name
                                        );
                                        ui.checkbox(&mut f.enabled, label)
                                            .on_hover_text(format!(
                                                "present in {} file(s); unchecked rows are \
                                                 left unchanged",
                                                f.n_files
                                            ));
                                        let shown = f
                                            .values
                                            .iter()
                                            .map(|v| {
                                                if v.is_empty() { "«empty»" } else { v.as_str() }
                                            })
                                            .collect::<Vec<_>>()
                                            .join("  |  ");
                                        let extra = f.n_values.saturating_sub(f.values.len());
                                        let txt = if extra > 0 {
                                            format!("{shown}  (+{extra} more)")
                                        } else {
                                            shown
                                        };
                                        ui.add(
                                            egui::Label::new(egui::RichText::new(txt).weak())
                                                .truncate(),
                                        )
                                        .on_hover_text(f.values.join("\n"));
                                        ui.horizontal(|ui| {
                                            ui.add_enabled(
                                                f.enabled,
                                                egui::TextEdit::singleline(&mut f.replacement)
                                                    .desired_width(170.0)
                                                    .hint_text("(clear)"),
                                            );
                                            if f.replacement != f.suggested
                                                && ui
                                                    .small_button("↺")
                                                    .on_hover_text(format!(
                                                        "Back to the suggestion: “{}”",
                                                        if f.suggested.is_empty() {
                                                            "(clear)"
                                                        } else {
                                                            &f.suggested
                                                        }
                                                    ))
                                                    .clicked()
                                            {
                                                f.replacement = f.suggested.clone();
                                            }
                                        });
                                        ui.end_row();
                                    }
                                });
                        });

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.checkbox(
                            &mut self.anon_remap_uids,
                            format!("Regenerate {} UID(s)", scan.uid_count),
                        )
                        .on_hover_text(
                            "Every study / series / SOP instance / frame-of-reference UID is \
                             replaced by a fresh one — the same original always maps to the \
                             same new UID, so cross-references stay valid",
                        );
                        ui.checkbox(
                            &mut self.anon_remove_private,
                            format!("Remove {} private element(s)", scan.private_count),
                        )
                        .on_hover_text("Drops all odd-group (vendor private) elements");
                        ui.checkbox(&mut self.anon_mark, "Mark as de-identified")
                            .on_hover_text(
                                "Writes PatientIdentityRemoved=YES and DeidentificationMethod",
                            );
                    });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Write to").strong());
                        ui.add_enabled(
                            !self.anon_in_place,
                            egui::TextEdit::singleline(&mut self.anon_out)
                                .desired_width(380.0)
                                .hint_text("output folder (files keep their relative paths)"),
                        );
                        if ui
                            .add_enabled(!self.anon_in_place, egui::Button::new("📂 Browse…"))
                            .clicked()
                        {
                            browse_out = true;
                        }
                        ui.checkbox(&mut self.anon_in_place, "overwrite in place")
                            .on_hover_text("Rewrites the original files — no copy is kept");
                    });

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if let Some(job) = &self.anon_apply_job {
                            ui.spinner();
                            ui.label(job.progress.get());
                        } else if ui
                            .add_enabled(!busy, egui::Button::new("🔏 Anonymize"))
                            .on_hover_text("Applies the checked replacements to every file")
                            .clicked()
                        {
                            do_apply = true;
                        }
                        if let Some(msg) = &self.anon_result {
                            ui.label(msg);
                        }
                    });
                }
            });

        if !open {
            // Closing the window forgets everything that was scanned — the
            // findings (they contain patient identity!), any running scan,
            // the result line and the derived output path. Only the folder
            // field is kept for convenience.
            self.anon_scan = None;
            self.anon_scan_job = None;
            self.anon_result = None;
            self.anon_out.clear();
            self.anon_in_place = false;
            // A running rewrite is not aborted — the background thread
            // finishes writing; only its completion message is dropped.
            self.anon_apply_job = None;
        }
        self.anon_open = open;
        if browse_in {
            if let Some(dir) = Self::pick_folder("Select a DICOM folder to anonymize") {
                self.anon_dir = dir.display().to_string();
            }
        }
        if browse_out {
            if let Some(dir) = Self::pick_folder("Select the output folder") {
                self.anon_out = dir.display().to_string();
            }
        }
        if do_scan {
            self.anon_start_scan();
        }
        if do_apply {
            self.anon_start_apply();
        }
    }
}

#[cfg(test)]
mod theme_tests {
    use super::*;

    /// Relative luminance per WCAG 2.1.
    fn luminance(c: Color32) -> f64 {
        let ch = |v: u8| {
            let s = v as f64 / 255.0;
            if s <= 0.03928 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
        };
        0.2126 * ch(c.r()) + 0.7152 * ch(c.g()) + 0.0722 * ch(c.b())
    }

    /// WCAG contrast ratio between two opaque colors (1.0 … 21.0).
    fn contrast(a: Color32, b: Color32) -> f64 {
        let (la, lb) = (luminance(a), luminance(b));
        (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
    }

    /// The hand-picked accents must stay legible on the panel background of
    /// *both* themes — a pale amber that works on near-black is unreadable on
    /// egui's light `panel_fill` (gray 248), which is exactly the regression
    /// this guards against. 4.5 is the WCAG AA threshold for body text.
    #[test]
    fn accent_colors_are_legible_in_both_themes() {
        for visuals in [egui::Visuals::dark(), egui::Visuals::light()] {
            let bg = visuals.panel_fill;
            let name = if visuals.dark_mode { "dark" } else { "light" };
            for (label, color) in [
                ("warn", warn_color(&visuals)),
                ("alert", alert_color(&visuals)),
            ] {
                let ratio = contrast(color, bg);
                assert!(
                    ratio >= 4.5,
                    "{name} theme: {label} accent {color:?} on {bg:?} has contrast {ratio:.2}"
                );
            }
        }
    }

    /// The viewport gutter and an empty study row must be distinguishable from
    /// each other and from the panels, in both themes.
    #[test]
    fn backdrops_stay_distinguishable() {
        for visuals in [egui::Visuals::dark(), egui::Visuals::light()] {
            let name = if visuals.dark_mode { "dark" } else { "light" };
            let (backdrop, row) = (backdrop_color(&visuals), empty_row_color(&visuals));
            assert_ne!(backdrop, row, "{name} theme: gutter equals empty row");
            // An empty row shows hint text; it needs real contrast against it.
            let ratio = contrast(visuals.text_color(), row);
            assert!(
                ratio >= 3.0,
                "{name} theme: hint text on an empty row has contrast {ratio:.2}"
            );
            // The gutter must read as a frame around the black viewports, not
            // blend into them.
            if !visuals.dark_mode {
                assert!(
                    contrast(backdrop, Color32::BLACK) >= 4.5,
                    "{name} theme: gutter is too dark to frame the viewports"
                );
            }
        }
    }
}

#[cfg(test)]
mod tree_tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::rtdose::DoseGrid;
    use crate::rtplan::PlanInfo;
    use crate::rtstruct::StructureSet;

    fn series(uid: &str, patient: &str, study: &str) -> loader::SeriesInfo {
        loader::SeriesInfo {
            uid: uid.into(),
            modality: "CT".into(),
            description: format!("{uid} desc"),
            patient_id: patient.into(),
            patient_name: format!("{patient}^Name"),
            study_uid: study.into(),
            study_date: "20260818".into(),
            study_description: String::new(),
            files: vec![std::path::PathBuf::from(format!("{uid}.dcm"))],
        }
    }

    fn structset(sop: &str, series_uid: &str, study: &str) -> StructureSet {
        StructureSet {
            label: sop.into(),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: sop.into(),
            study_uid: study.into(),
            referenced_series_uid: series_uid.into(),
            file_name: format!("{sop}.dcm"),
            rois: Vec::new(),
        }
    }

    fn plan(sop: &str, structset_sop: &str, study: &str) -> PlanInfo {
        PlanInfo {
            label: sop.into(),
            name: String::new(),
            date: String::new(),
            plan_kind: "Ion".into(),
            n_fractions: None,
            target_prescription_dose: None,
            sop_instance_uid: sop.into(),
            study_uid: study.into(),
            referenced_structset_uid: structset_sop.into(),
            beams: Vec::new(),
        }
    }

    fn dose(plan_sop: &str, study: &str) -> DoseGrid {
        DoseGrid {
            data: vec![0.0],
            dims: [1, 1, 1],
            spacing: [1.0, 1.0],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            offsets: vec![0.0],
            units: "GY".into(),
            summation_type: "PLAN".into(),
            max_dose: 1.0,
            frame_of_reference_uid: String::new(),
            study_uid: study.into(),
            referenced_plan_uid: plan_sop.into(),
            label: plan_sop.into(),
        }
    }

    /// Two series, each with its own RTSTRUCT ▶ RTPLAN ▶ RTDOSE chain.
    fn two_chain_study() -> LoadedStudy {
        LoadedStudy {
            meta: loader::PatientMeta::default(),
            series: vec![series("se1", "P1", "st1"), series("se2", "P1", "st2")],
            active_series: 0,
            volume: Volume {
                data: vec![0],
                dims: [1, 1, 1],
                spacing: [1.0, 1.0, 1.0],
                origin: Vec3::new(0.0, 0.0, 0.0),
                row_dir: Vec3::new(1.0, 0.0, 0.0),
                col_dir: Vec3::new(0.0, 1.0, 0.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                frame_of_reference_uid: String::new(),
                min_value: 0,
                max_value: 0,
            },
            structure_sets: vec![
                structset("ss1", "se1", "st1"),
                structset("ss2", "se2", "st2"),
            ],
            doses: vec![dose("pl1", "st1"), dose("pl2", "st2")],
            plans: vec![plan("pl1", "ss1", "st1"), plan("pl2", "ss2", "st2")],
            planar_images: Vec::new(),
            registrations: Vec::new(),
            treat_records: Vec::new(),
            warnings: Vec::new(),
            default_window: (40.0, 400.0),
        }
    }

    /// Selecting one series must take exactly its reference chain — the bug
    /// this guards against is "move series moved every series and RT object".
    #[test]
    fn series_selection_takes_only_linked_objects() {
        let study = two_chain_study();
        let sel = ViewerApp::tree_sel_mask(&study, &TreeSel::Series(0));
        assert_eq!(sel, vec![true, false]);
        let masks = ViewerApp::subset_masks(&study, &sel, false, false);
        assert_eq!(masks.series, vec![true, false]);
        assert_eq!(masks.structs, vec![true, false]);
        assert_eq!(masks.plans, vec![true, false]);
        assert_eq!(masks.doses, vec![true, false]);

        let sub = ViewerApp::build_subset(&study, &masks, 0);
        assert_eq!(sub.series.len(), 1);
        assert_eq!(sub.series[0].uid, "se1");
        assert_eq!(sub.structure_sets.len(), 1);
        assert_eq!(sub.structure_sets[0].sop_instance_uid, "ss1");
        assert_eq!(sub.plans.len(), 1);
        assert_eq!(sub.doses.len(), 1);
        assert_eq!(sub.doses[0].referenced_plan_uid, "pl1");
    }

    /// Study selection takes the chain plus same-study objects.
    #[test]
    fn study_selection_takes_study_objects() {
        let study = two_chain_study();
        let sel = ViewerApp::tree_sel_mask(&study, &TreeSel::Study("st2".into()));
        assert_eq!(sel, vec![false, true]);
        let masks = ViewerApp::subset_masks(&study, &sel, true, false);
        assert_eq!(masks.structs, vec![false, true]);
        assert_eq!(masks.plans, vec![false, true]);
        assert_eq!(masks.doses, vec![false, true]);
    }

    /// Patient selection over all series covers everything.
    #[test]
    fn patient_selection_covers_all() {
        let study = two_chain_study();
        let sel = ViewerApp::tree_sel_mask(&study, &TreeSel::Patient("P1".into()));
        assert_eq!(sel, vec![true, true]);
        let masks = ViewerApp::subset_masks(&study, &sel, true, true);
        assert!(masks.structs.iter().all(|b| *b));
        assert!(masks.take_extras);
    }

    /// merge_study skips series and RT objects that are already present.
    #[test]
    fn merge_dedupes_by_uid() {
        let mut dest = two_chain_study();
        let masks = ViewerApp::subset_masks(
            &dest,
            &[true, false],
            false,
            false,
        );
        let sub = ViewerApp::build_subset(&dest, &masks, 0);
        let notes = loader::merge_study(&mut dest, sub);
        assert_eq!(dest.series.len(), 2, "duplicate series must not be added");
        assert_eq!(dest.structure_sets.len(), 2);
        assert_eq!(dest.plans.len(), 2);
        assert_eq!(dest.doses.len(), 2);
        assert!(!notes.is_empty(), "skipping duplicates should be reported");

        // A genuinely new series does get merged.
        let mut extra = two_chain_study();
        extra.series[0].uid = "se3".into();
        extra.series[0].study_uid = "st3".into();
        extra.structure_sets.clear();
        extra.plans.clear();
        extra.doses.clear();
        extra.series.truncate(1);
        loader::merge_study(&mut dest, extra);
        assert_eq!(dest.series.len(), 3);
    }
}
