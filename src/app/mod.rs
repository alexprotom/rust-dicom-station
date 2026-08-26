//! The egui application: menu bar, toolbar, side panel, and one or two rows
//! (comparison mode) of three linked MPR views.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use egui::{
    Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions,
    Vec2,
};

use rayon::prelude::*;

use crate::anonymize;
use crate::autoseg;
use crate::dicom_export;
use crate::extras;
use crate::gen_test_data::{self, GenParams};
use crate::geometry::Vec3;
use crate::loader::{self, LoadedStudy};
use crate::mesh3d::{self, GridGeom, RoiMesh};
use crate::models;
use crate::progress::{self, Progress};
use crate::registration::{
    self, dvf, FieldStyle, LandmarkPair, LandmarkParams, Metric, RegMethod, RegionMask,
    RegistrationResult, Transform3, VectorField,
};
use crate::render;
use crate::segmentation::{self, GrowState, Segmentation};
use crate::settings::{self, Settings};
use crate::simulate::{self, SimParams};
use crate::volume::{ViewPlane, Volume};

mod box_seg;
mod chrome;
mod d3;
mod dialogs;
mod drr_win;
mod jobs;
mod models_win;
mod panels;
mod planar;
mod prompt_seg;
mod propagate_win;
mod reg_panel;
mod rename;
mod seg;
mod seg_engines;
mod sets;
mod theme;
mod tree;
mod views;

use drr_win::DrrDialog;
use propagate_win::{PropOutcome, PropagateDialog};
use reg_panel::{RegOutcome, RegRoi};
use rename::{RenameDialog, RenameTarget};
use seg_engines::*;
use theme::*;

const SLOT_NAMES: [&str; 2] = ["A", "B"];

/// The auto-segmentation window: its parameters, and the run they start.
struct AutosegDialog {
    slot: usize,
    variant: autoseg::Variant,
    device: autoseg::DevicePref,
    /// Sub-model selection for the 1.5 mm variant
    /// (organs, vertebrae, cardiac, muscles, ribs).
    parts: [bool; 5],
}

/// A finished auto-segmentation waiting for the user to choose organs.
struct AutosegPending {
    slot: usize,
    result: autoseg::AutosegResult,
    selected: Vec<bool>,
    also_rs: bool,
}

// Dose display settings

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
        IsoLevel {
            pct: 107.0,
            color: Color32::from_rgb(255, 0, 255),
            on: true,
        },
        IsoLevel {
            pct: 100.0,
            color: Color32::from_rgb(255, 0, 0),
            on: true,
        },
        IsoLevel {
            pct: 95.0,
            color: Color32::from_rgb(255, 128, 0),
            on: true,
        },
        IsoLevel {
            pct: 90.0,
            color: Color32::from_rgb(255, 255, 0),
            on: true,
        },
        IsoLevel {
            pct: 80.0,
            color: Color32::from_rgb(0, 220, 0),
            on: true,
        },
        IsoLevel {
            pct: 70.0,
            color: Color32::from_rgb(0, 255, 255),
            on: true,
        },
        IsoLevel {
            pct: 50.0,
            color: Color32::from_rgb(0, 128, 255),
            on: true,
        },
        IsoLevel {
            pct: 30.0,
            color: Color32::from_rgb(0, 0, 255),
            on: true,
        },
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

// Interactive segmentation tools

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

// Per-viewport state and caches

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
    /// Identity of the vector-field geometry cached below.
    field_key: Option<u64>,
    /// Arrows of the deformation field on this slice, display-pixel space.
    field_arrows: Vec<registration::dvf::Glyph>,
    /// The deformed lattice of this slice, display-pixel space.
    field_lines: Vec<Vec<[f32; 2]>>,
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
            field_key: None,
            field_arrows: Vec::new(),
            field_lines: Vec::new(),
        }
    }

    fn invalidate(&mut self) {
        self.img_key = None;
        self.dose_key = None;
        self.contour_key = None;
        self.fusion_key = None;
        self.seg_key = None;
        self.field_key = None;
    }
}

/// Which of the three view slots shows a plane — the order `fresh_views`
/// builds them in.
fn plane_index(plane: ViewPlane) -> usize {
    match plane {
        ViewPlane::Axial => 0,
        ViewPlane::Sagittal => 1,
        ViewPlane::Coronal => 2,
    }
}

fn fresh_views() -> [ViewState; 3] {
    [
        ViewState::new(ViewPlane::Axial),
        ViewState::new(ViewPlane::Sagittal),
        ViewState::new(ViewPlane::Coronal),
    ]
}

// A loaded study with its own display state ("A" = primary, "B" = comparison)

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
    /// Index of the active segmentation series within `study.seg_series`.
    active_seg_series: usize,
    /// Index of the segment the tools edit, within that series.
    active_seg: usize,
}

impl StudySlot {
    /// The currently selected structure set of this slot, if any.
    fn active_structures(&self) -> Option<&crate::rtstruct::StructureSet> {
        self.study
            .as_ref()
            .and_then(|s| s.structure_sets.get(self.active_structs))
    }

    /// Index of the segmentation series the tools edit, clamped to what the
    /// study actually holds.
    fn seg_series_idx(&self) -> Option<usize> {
        let st = self.study.as_ref()?;
        (!st.seg_series.is_empty()).then(|| self.active_seg_series.min(st.seg_series.len() - 1))
    }

    /// Segments of the active segmentation series — empty unless they live
    /// on the displayed volume's lattice, because every overlay, brush
    /// stroke and mesh indexes them with that volume's dimensions. A series
    /// drawn on another image series simply has nothing to show here.
    fn segs(&self) -> &[Segmentation] {
        match (self.study.as_ref(), self.seg_series_idx()) {
            (Some(st), Some(i)) if st.seg_series[i].grid.dims == st.volume.dims => {
                &st.seg_series[i].segs
            }
            _ => &[],
        }
    }

    /// [`Self::segs`] for editing; `None` when there is nothing editable.
    fn segs_mut(&mut self) -> Option<&mut Vec<Segmentation>> {
        let i = self.seg_series_idx()?;
        let st = self.study.as_mut()?;
        let dims = st.volume.dims;
        let ser = &mut st.seg_series[i];
        (ser.grid.dims == dims).then_some(&mut ser.segs)
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
            active_seg_series: 0,
            active_seg: 0,
        }
    }
}

// Background loading

/// A freshly reconstructed volume: pixels, its default window/level and any
/// non-fatal notes raised while reading the series.
type LoadedVolume = (Arc<Volume>, (f32, f32), Vec<String>);

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

impl<T, P> Job<T, P> {
    /// Run `work` on a new thread and return the handle to poll for its
    /// result. The worker gets the progress handle; the caller keeps a clone.
    fn spawn(progress: Arc<P>, work: impl FnOnce(&P) -> T + Send + 'static) -> Job<T, P>
    where
        T: Send + 'static,
        P: Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel();
        let p = progress.clone();
        std::thread::spawn(move || {
            let _ = tx.send(work(&p));
        });
        Job { progress, rx }
    }
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

/// [`poll_job`] for the jobs that answer with `(slot, Result)`: a failure is
/// reported as `"{what} failed: …"`, except a cancellation, which is what
/// the user asked for and needs no dialog.
fn poll_tool_job<T, P>(
    slot: &mut Option<Job<(usize, anyhow::Result<T>), P>>,
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
    /// Also draw the *other* dataset's structures, mapped through the active
    /// registration — the two anatomies in one scene is what makes a
    /// deformable result readable at all.
    show_other: bool,
    /// Opacity of that second dataset, independent of this one's.
    other_opacity: f32,
    other_meshes: Option<Arc<Vec<RoiMesh>>>,
    other_job: Option<Job<Vec<RoiMesh>>>,
    /// Identity of the (structure set, registration) `other_meshes` were
    /// built from.
    other_key: u64,
    /// Draw the deformation field as arrows in the scene.
    show_field: bool,
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

/// Which of a dataset's two kinds of segmented series an action addresses.
///
/// The data tree treats them alike — both are series drawn on an image
/// series, both hold named, coloured items — even though one stores contours
/// and the other voxel masks. Conversions between the two happen on
/// transfer (`ViewerApp::apply_item_action`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum SetKind {
    /// RT Structure Set: contours.
    Structures,
    /// DICOM Segmentation series: binary voxel masks.
    Segmentations,
}

impl SetKind {
    /// "structure" / "segment", pluralized for `n`.
    fn item_name(self, n: usize) -> &'static str {
        match (self, n) {
            (SetKind::Structures, 1) => "structure",
            (SetKind::Structures, _) => "structures",
            (SetKind::Segmentations, 1) => "segment",
            (SetKind::Segmentations, _) => "segments",
        }
    }
    fn series_name(self) -> &'static str {
        match self {
            SetKind::Structures => "RT structure set",
            SetKind::Segmentations => "segmentation series",
        }
    }
}

/// One structure set / segmentation series of one dataset.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SetRef {
    slot: usize,
    kind: SetKind,
    /// Index into that dataset's list, or [`SetRef::NEW`] for a series that
    /// does not exist yet — what the *New …* transfer destinations mean.
    idx: usize,
}

impl SetRef {
    const NEW: usize = usize::MAX;
}

/// Deferred right-click action on a whole series node of the data tree.
enum SetAction {
    New(SetRef),
    Remove(SetRef),
    /// Open the rename dialog on the series.
    Rename(SetRef),
    /// Re-point the series at the image series with this Series Instance UID.
    Connect(SetRef, String),
    /// Copy (`copy`) or move the whole series to the other dataset.
    Transfer {
        from: SetRef,
        copy: bool,
    },
    /// Write one segmentation series as a standalone DICOM SEG file.
    ExportSeg(SetRef),
}

/// Deferred right-click action on individual structures / segments.
enum ItemAction {
    /// Copy (`copy`) or move `items` of `from` into the series `to`.
    Transfer {
        from: SetRef,
        items: Vec<usize>,
        to: SetRef,
        copy: bool,
    },
    Remove {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Open the rename dialog on the clicked item alone.
    Rename {
        from: SetRef,
        idx: usize,
    },
}

/// Which parts of a `LoadedStudy` a tree selection covers: the selected
/// series plus the RT objects linked to them via the DICOM reference chain
/// (RTSTRUCT ▶ series, RTPLAN ▶ RTSTRUCT, RTDOSE ▶ RTPLAN).
struct SubsetMasks {
    series: Vec<bool>,
    structs: Vec<bool>,
    /// Segmentation series drawn on the selected image series.
    seg_series: Vec<bool>,
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
    /// The displacement field sampled from the transform, so the views draw
    /// a lattice lookup instead of evaluating the transform per pixel.
    field: Arc<VectorField>,
    /// The region the run was restricted to, kept so the field can be
    /// re-sampled at a different lattice without rebuilding the mask.
    region: Option<Arc<RegionMask>>,
}

// Application

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
    /// A one-line confirmation shown in a small modal (e.g. a written file).
    notice: Option<String>,

    // Registration (direction selectable: either study can be the fixed one).
    registration: Option<ActiveRegistration>,
    /// The payload carries the slot that was used as the fixed image.
    reg_job: Option<SegJob<RegOutcome>>,
    /// Fixed-image slot for the *next* registration run (0 = A, 1 = B).
    reg_fixed_slot: usize,
    fusion_on: bool,
    fusion_weight: f32,
    /// Bumped when the registration result changes → fusion cache rebuild.
    reg_gen: u64,
    /// Which algorithm the next run uses.
    reg_method: RegMethod,
    /// What the plastimatch engine minimizes.
    reg_metric: Metric,
    reg_levels: usize,
    reg_iterations: usize,
    reg_samples: usize,
    reg_grid_mm: f64,
    /// Sampling threshold (a crude body mask), HU.
    reg_threshold: f32,
    /// plastimatch bending-energy weight.
    reg_regularization: f64,
    /// Kernel, stiffness and reach of the landmark warp.
    reg_landmark: LandmarkParams,
    /// The paired points the landmark warp interpolates.
    reg_landmarks: Vec<LandmarkPair>,
    /// Which structure of the fixed dataset restricts the next run.
    reg_roi: RegRoi,
    /// Margin the region is grown by, mm.
    reg_margin_mm: f64,

    // The deformation vector field of the active registration.
    field_on: bool,
    field_style: FieldStyle,
    field_step_mm: f64,
    /// Arrows are drawn this many times their true length.
    field_scale: f32,
    field_color: bool,
    /// A re-sampling of the field after the lattice step changed.
    field_job: Option<Job<VectorField>>,

    // Tools ▶ DRR.
    drr_dialog: Option<DrrDialog>,
    drr_job: Option<Job<anyhow::Result<Vec<crate::drr::DrrImage>>>>,

    // Tools ▶ Propagate structures.
    /// The window, when open.
    propagate_dialog: Option<PropagateDialog>,
    /// The payload carries the destination slot.
    propagate_job: Option<SegJob<PropOutcome>>,

    // Study transform simulator (registration QA).
    sim_source: usize,
    sim_params: SimParams,
    sim_job: Option<Job<(usize, LoadedStudy)>>,
    last_sim: Option<String>,
    // DICOM export (File ▶ Export dataset …).
    /// Dialog visibility.
    export_open: bool,
    /// Dataset the dialog exports.
    export_slot: usize,
    /// Output folder as edited in the dialog.
    export_dir: String,
    /// Editable DICOM attributes, filled from the study when the dialog opens.
    export_params: Option<dicom_export::ExportParams>,
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
    /// Deferred right-click action on a structure set / segmentation series.
    set_action: Option<SetAction>,
    /// Deferred right-click action on structures / segments.
    item_action: Option<ItemAction>,
    /// The rename dialog, when open.
    rename: Option<RenameDialog>,
    /// A rename requested from a context menu, opened after the frame's
    /// borrows are released.
    rename_request: Option<RenameTarget>,
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

    /// Root folder of the downloaded network weights, shared by the three
    /// engines (persisted in the settings file; blank = the default).
    models_dir: String,

    // Tools ▶ Downloaded models: the inventory window.
    models_open: bool,
    /// The inventory with each model's state, re-read at most twice a second.
    models_scan: Vec<(models::ModelAsset, models::AssetStatus)>,
    /// `ctx` time the scan above was taken at.
    models_scan_at: f64,
    /// A download / update batch in flight; its payload is the summary line.
    models_job: Option<Job<anyhow::Result<String>>>,
    models_result: Option<String>,

    // Auto-segmentation (TotalSegmentator re-implementation, see `autoseg`).
    /// The payload carries the slot the volume came from.
    autoseg_job: Option<SegJob<autoseg::AutosegResult>>,
    /// Slot currently being segmented (progress shown in its sidebar section).
    autoseg_slot: usize,
    /// The tool window, when open; it stays open while a run is in flight.
    autoseg_dialog: Option<AutosegDialog>,
    /// Finished result awaiting organ selection.
    autoseg_pending: Option<AutosegPending>,

    // Prompt-driven segmentation (SegVol re-implementation, see `segvol`).
    segvol_job: Option<SegJob<prompt_seg::SegVolResult>>,
    segvol_slot: usize,
    /// The tool window, when open; it stays open across runs.
    segvol_dialog: Option<prompt_seg::SegVolDialog>,

    // Slice-propagating segmentation (MedSAM2 re-implementation): the drawn
    // box, the loaded engine and the prepared stack all live in one struct.
    medsam2_job: Option<SegJob<box_seg::Medsam2Done>>,
    medsam2: box_seg::Medsam2State,

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

/// Last few characters of a UID for compact display.
fn tail(uid: &str) -> String {
    let n = uid.chars().count();
    if n <= 10 {
        uid.to_string()
    } else {
        uid.chars().skip(n - 10).collect()
    }
}

impl ViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial_a: Option<PathBuf>,
        initial_b: Option<PathBuf>,
    ) -> Self {
        let prefs = settings::load();
        cc.egui_ctx.set_theme(prefs.theme);
        let models_dir = prefs
            .models_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        // Installations that predate the single `models/` root keep their
        // downloads; the folders are moved into place, never re-fetched.
        let moved = models::migrate_legacy_layout(&models::root_from_setting(&models_dir));
        for engine in &moved {
            eprintln!(
                "moved the {} weights into {}",
                engine.subdir(),
                models::root_from_setting(&models_dir).display()
            );
        }
        let mut app = ViewerApp {
            slots: [StudySlot::empty(), StudySlot::empty()],
            comparison: initial_b.is_some(),
            link_studies: true,
            hovered_slot: 0,
            loading: None,
            pending_load: None,
            error: None,
            notice: None,
            registration: None,
            reg_job: None,
            reg_fixed_slot: 0,
            fusion_on: false,
            fusion_weight: 1.0,
            reg_gen: 0,
            reg_method: RegMethod::ElastixRigid,
            reg_metric: Metric::MeanSquares,
            reg_levels: 3,
            reg_iterations: 300,
            reg_samples: 3000,
            reg_grid_mm: 32.0,
            reg_threshold: -500.0,
            reg_regularization: 0.02,
            reg_landmark: LandmarkParams::default(),
            reg_landmarks: Vec::new(),
            reg_roi: RegRoi::Whole,
            reg_margin_mm: 10.0,
            field_on: false,
            field_style: FieldStyle::Arrows,
            field_step_mm: 12.0,
            field_scale: 3.0,
            field_color: true,
            field_job: None,
            drr_dialog: None,
            drr_job: None,
            propagate_dialog: None,
            propagate_job: None,
            sim_source: 0,
            sim_params: SimParams::default(),
            sim_job: None,
            last_sim: None,
            export_open: false,
            export_slot: 0,
            export_dir: String::new(),
            export_params: None,
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
            set_action: None,
            item_action: None,
            rename: None,
            rename_request: None,
            maximized: None,
            reg_apply_invert: false,
            window_center: 40.0,
            window_width: 400.0,
            show_contours: true,
            show_crosshair: true,
            show_labels: true,
            show_isocenters: true,
            models_dir,
            models_open: false,
            models_scan: Vec::new(),
            models_scan_at: f64::NEG_INFINITY,
            models_job: None,
            models_result: None,
            autoseg_job: None,
            autoseg_slot: 0,
            autoseg_dialog: None,
            autoseg_pending: None,
            segvol_job: None,
            segvol_slot: 0,
            segvol_dialog: None,
            medsam2_job: None,
            medsam2: Default::default(),
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

    /// Apply an appearance preference and remember it for the next run.
    pub(super) fn set_theme(&mut self, ctx: &egui::Context, theme: egui::ThemePreference) {
        self.theme = theme;
        ctx.set_theme(theme);
        self.persist_settings();
    }

    /// Write all persisted preferences (best-effort, see `settings::save`).
    pub(super) fn persist_settings(&mut self) {
        let default_dir = models::default_root().display().to_string();
        let models_dir =
            if self.models_dir.trim().is_empty() || self.models_dir.trim() == default_dir {
                None
            } else {
                Some(PathBuf::from(self.models_dir.trim()))
            };
        match settings::save(&Settings {
            theme: self.theme,
            models_dir,
        }) {
            Ok(()) => self.settings_error = None,
            Err(e) => {
                self.settings_error = Some(format!("⚠ settings not saved: {e:#}"));
            }
        }
    }

    /// Reset zoom, pan, crosshair and slice (all back to the volume center)
    /// of every view of both datasets.
    pub(super) fn reset_all_views(&mut self) {
        for s in &mut self.slots {
            for v in &mut s.views {
                v.zoom = 0.0;
                v.pan = Vec2::ZERO;
                v.invalidate();
            }
        }
        for slot in 0..self.slots.len() {
            self.center_cursor(slot);
        }
    }

    /// Put the crosshair of `slot` back at its volume center and follow it
    /// with that slot's three slices. The other dataset is left alone even
    /// when crosshair linking is on — a reset is per-dataset, and "Reset all
    /// views" recenters both anyway.
    pub(super) fn center_cursor(&mut self, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let d = study.volume.dims;
        self.slots[slot].cursor = [
            (d[0] as f64 - 1.0).max(0.0) / 2.0,
            (d[1] as f64 - 1.0).max(0.0) / 2.0,
            (d[2] as f64 - 1.0).max(0.0) / 2.0,
        ];
        self.sync_views_to_cursor(slot, None);
    }

    pub(super) fn close_comparison(&mut self) {
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

    pub(super) fn pick_folder(title: &str) -> Option<PathBuf> {
        rfd::FileDialog::new().set_title(title).pick_folder()
    }
}

// eframe::App

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
        if let Some((target, study)) =
            poll_job(&mut self.sim_job, &ctx, "Simulation", &mut self.error)
        {
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
        match poll_job(
            &mut self.gen_job,
            &ctx,
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

        // Poll a model download / update batch.
        self.poll_models_job(&ctx);

        // Poll the three segmentation engines.
        if let Some((slot, result)) =
            poll_tool_job(&mut self.autoseg_job, &ctx, AUTOSEG.name, &mut self.error)
        {
            self.on_autoseg_done(slot, result);
        }
        if let Some((slot, result)) = poll_tool_job(
            &mut self.medsam2_job,
            &ctx,
            SLICE_PROP.name,
            &mut self.error,
        ) {
            self.on_medsam2_done(slot, result);
        }
        if let Some((slot, result)) =
            poll_tool_job(&mut self.segvol_job, &ctx, PROMPT_SEG.name, &mut self.error)
        {
            self.on_segvol_done(slot, result);
        }

        // Poll background registration.
        if let Some((fixed_slot, out)) =
            poll_tool_job(&mut self.reg_job, &ctx, "Registration", &mut self.error)
        {
            self.registration = Some(ActiveRegistration {
                result: out.result,
                fixed_slot,
                field: Arc::new(out.field),
                region: out.region,
            });
            self.fusion_on = true;
            self.reg_gen += 1;
            // Re-propagate the crosshair through the new transform.
            let cursor = self.slots[fixed_slot].cursor;
            self.set_cursor(fixed_slot, cursor, usize::MAX);
        }

        // Poll a DRR rendering.
        match poll_job(&mut self.drr_job, &ctx, "DRR", &mut self.error) {
            Some(Ok(images)) => self.on_drr_done(images),
            // A cancelled render is what the user asked for, not a failure.
            Some(Err(e)) if !progress::is_cancellation(&e) => {
                self.error = Some(format!("DRR failed: {e:#}"));
            }
            _ => {}
        }

        // Poll a structure propagation.
        if let Some((dst_slot, out)) = poll_tool_job(
            &mut self.propagate_job,
            &ctx,
            "Propagation",
            &mut self.error,
        ) {
            self.on_propagation_done(dst_slot, out);
        }

        // Poll a vector-field re-sampling.
        if let Some(field) = poll_job(&mut self.field_job, &ctx, "Vector field", &mut self.error) {
            if let Some(reg) = &mut self.registration {
                reg.field = Arc::new(field);
            }
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
        if let Some(action) = self.set_action.take() {
            self.apply_set_action(action);
        }
        if let Some(action) = self.item_action.take() {
            self.apply_item_action(action);
        }
        if let Some(target) = self.rename_request.take() {
            self.open_rename(target);
        }
        self.modals(&ctx);
    }
}
