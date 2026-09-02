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
use crate::bodymask;
use crate::dicom_export;
use crate::extras;
use crate::fourd;
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

mod body_win;
mod box_seg;
mod chrome;
mod combine_win;
mod compare_win;
mod d3;
mod detach;
mod dialogs;
mod drr_win;
mod dvh_win;
mod glyphs;
mod jobs;
mod models_win;
mod motion_results;
mod motion_win;
mod pacs_win;
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
mod transfer_win;
mod tree;
mod views;

use drr_win::DrrDialog;
use pacs_win::{PacsOutcome, PacsWindow};
use propagate_win::{GroupRegistration, PropOutcome, PropagateDialog};
use reg_panel::{RegOutcome, RegRoi};
use rename::{RenameDialog, RenameTarget};
use seg_engines::*;
use theme::*;

const SLOT_NAMES: [&str; 2] = ["A", "B"];

/// A 4D-group edit requested from the data tree's context menus, applied
/// after the frame's borrows are released (the tree renders behind a shared
/// borrow of the study).
enum FourDAction {
    /// Add a series to an existing group, as a phase.
    Add {
        slot: usize,
        group: usize,
        series: usize,
    },
    /// Start a new custom group from one series.
    New { slot: usize, series: usize },
    /// Remove one member from a group.
    RemoveMember {
        slot: usize,
        group: usize,
        member: usize,
    },
    /// Move a member one place up (−1) or down (+1).
    Shift {
        slot: usize,
        group: usize,
        member: usize,
        delta: isize,
    },
    /// Cycle a member's role (phase ▸ AVG ▸ MIP ▸ MinIP).
    SetRole {
        slot: usize,
        group: usize,
        member: usize,
        role: fourd::Role,
    },
    /// Dissolve the whole group (the series stay).
    Dissolve { slot: usize, group: usize },
    /// Re-run automatic detection, keeping custom groups.
    Redetect { slot: usize },
    /// Open the 4D motion tool on this group.
    Analyse { slot: usize, group: usize },
}

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
    /// Which plans are drawn in the views (their isocenters), one flag per
    /// entry of `study.plans`. Missing entries count as visible, so a plan
    /// that arrives later is shown.
    plan_visible: Vec<bool>,
    /// Index of the active structure set within `study.structure_sets`.
    active_structs: usize,
    /// The tick box on the RT structures series row: whether the active set
    /// is drawn at all. Unticking it clears the contours from the image views
    /// while the set stays selected, so the ROI list, the drawing tools and a
    /// 3D scene opened on it all keep working.
    structs_shown: bool,
    /// The same tick box on the segmentation series row.
    segs_shown: bool,
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

    /// Whether this slot shows an image volume.
    ///
    /// A slot can hold a perfectly good dataset with none — RT images, a
    /// structure set, a plan. Every feature that needs voxels (the MPR views,
    /// the brush, registration, the segmentation engines, the DRR) asks this
    /// rather than `study.is_some()`.
    fn has_volume(&self) -> bool {
        self.study.as_ref().is_some_and(|st| st.has_volume())
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
            (Some(st), Some(i))
                if st.has_volume() && st.seg_series[i].grid.dims == st.volume.dims =>
            {
                &st.seg_series[i].segs
            }
            _ => &[],
        }
    }

    /// [`Self::segs`] for editing; `None` when there is nothing editable.
    fn segs_mut(&mut self) -> Option<&mut Vec<Segmentation>> {
        let i = self.seg_series_idx()?;
        let st = self.study.as_mut()?;
        if !st.has_volume() {
            return None;
        }
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
            plan_visible: Vec::new(),
            active_structs: 0,
            structs_shown: true,
            segs_shown: true,
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
struct Job<T> {
    progress: Arc<Progress>,
    rx: mpsc::Receiver<T>,
}

impl<T> Job<T> {
    /// Run `work` on a new thread and return the handle to poll for its
    /// result. The worker gets the progress handle; the caller keeps a clone.
    fn spawn(progress: Arc<Progress>, work: impl FnOnce(&Progress) -> T + Send + 'static) -> Job<T>
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

/// Poll a background job. Returns its result once, clearing the slot; reports
/// a worker that died without answering into `error`; otherwise schedules the
/// next poll and returns `None`.
fn poll_job<T>(
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
fn poll_tool_job<T>(
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

/// One of the study-level objects that are neither image, structure nor
/// segmentation series: they hang off the study, are drawn in the views (or
/// not), and can be taken out of the dataset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ObjKind {
    Dose,
    Plan,
    Planar,
    Registration,
    Record,
}

/// Which object, in which dataset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ObjRef {
    slot: usize,
    kind: ObjKind,
    idx: usize,
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
    /// Write a segmentation series as a standalone DICOM SEG file. An empty
    /// `items` means the whole series; otherwise only those segments.
    ExportSeg {
        set: SetRef,
        items: Vec<usize>,
    },
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
    /// Open the structure-algebra window with these items as its operands.
    Combine {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Plot these items' dose–volume histograms.
    Dvh {
        from: SetRef,
        items: Vec<usize>,
    },
    /// Write these segments as a DICOM SEG file of their own.
    ExportSeg {
        from: SetRef,
        items: Vec<usize>,
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
    /// What each dataset has been loaded from this run, in order: the
    /// *Restore the last session* button on the start screen replays it, and
    /// it is written to the settings file as it changes.
    session: [Vec<PathBuf>; 2],
    /// Sources still to load for a restore, drained one at a time as each
    /// load finishes (the loader takes one folder at a time).
    restore_queue: Vec<(usize, PathBuf)>,
    /// The session the previous run ended with, as read from the settings
    /// file. Emptied when it turns out its data is gone.
    last_session: [Vec<PathBuf>; 2],
    /// Whether the archive holds anything, and when that was last looked at:
    /// the start screen offers *Load data from PACS* only when it does, and
    /// asking the file system every frame would be silly.
    archive_has_data: bool,
    archive_checked_at: f64,
    pending_load: Option<(usize, PathBuf)>,
    /// The same, for an explicit file selection (slot, files).
    pending_load_files: Option<(usize, Vec<PathBuf>)>,
    error: Option<String>,
    /// A one-line confirmation shown in a small modal (e.g. a written file).
    notice: Option<String>,

    // Registration (direction selectable: either study can be the fixed one).
    registration: Option<ActiveRegistration>,
    /// The last 4D group registered phase by phase, so propagating onto it
    /// does not repeat the registrations. Cleared when the registration is.
    group_registration: Option<GroupRegistration>,
    /// Which 4D group the registration module runs against, when it runs
    /// against one rather than the other dataset's displayed volume.
    reg_group: Option<(usize, usize)>,
    /// Dataset whose displayed volume is the moving image of a group run.
    /// Its own dataset is a normal choice: a planning CT and the 4DCT of the
    /// same patient usually arrive together.
    reg_group_moving: usize,
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
    /// A study-level object the tree asked to remove, applied after the
    /// panel has been drawn (it borrows the study while drawing).
    obj_remove: Option<ObjRef>,
    /// Deferred right-click action on structures / segments.
    item_action: Option<ItemAction>,
    /// The rename dialog, when open.
    rename: Option<RenameDialog>,
    /// A rename requested from a context menu, opened after the frame's
    /// borrows are released.
    rename_request: Option<RenameTarget>,
    /// Anchor of the last check-box click in a structure / segment list, so
    /// Shift-click can extend a range from it.
    tick_anchor: Option<(SetRef, usize)>,
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

    /// Root of the local patient archive (persisted; blank = the default).
    archive_dir: String,

    // Tools ▶ PACS: the patient archive window.
    /// The window, when open.
    pacs: Option<PacsWindow>,
    /// The archive job in flight — a scan, an import, an upload or a removal.
    pacs_job: Option<Job<anyhow::Result<PacsOutcome>>>,

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

    // Body / External contouring (see `bodymask`) — the one tool that can
    // answer with no network at all.
    body_job: Option<SegJob<bodymask::BodyResult>>,
    body_slot: usize,
    /// The tool window, when open; it stays open across runs.
    body_dialog: Option<body_win::BodyDialog>,

    // Dose–volume histograms (see `dvh`), in a window of their own.
    dvh_open: bool,
    dvh_dialog: Option<dvh_win::DvhDialog>,
    dvh_job: Option<Job<anyhow::Result<dvh_win::DvhDone>>>,

    // Structure algebra (see `structops`): combining contours and segments.
    combine_job: Option<SegJob<combine_win::CombineResult>>,
    combine_slot: usize,
    combine_dialog: Option<combine_win::CombineDialog>,

    // 4D motion / ITV analysis (see `motion` and `fourd`).
    motion_job: Option<SegJob<motion_win::MotionOutcome>>,
    motion_slot: usize,
    motion_dialog: Option<motion_win::MotionDialog>,
    /// The last run's settings, re-applicable to another dataset / study.
    motion_recipe: Option<motion_win::MotionRecipe>,
    /// Every finished run of this session, newest last.
    motion_reports: Vec<crate::motion::MotionReport>,
    /// The results window: visibility, selected run, comparison run.
    motion_results_open: bool,
    motion_sel: usize,
    motion_cmp: Option<usize>,

    // Tools ▶ Transfer by relationship.
    transfer_dialog: Option<transfer_win::TransferDialog>,

    // Tools ▶ Compare structures.
    compare_dialog: Option<compare_win::CompareDialog>,

    /// Deferred 4D-group edit from the data tree's context menus.
    fourd_action: Option<FourDAction>,

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

    /// The CT window preset last picked from the toolbar list (an index into
    /// [`WL_PRESETS`]), so the closed combo can name it instead of reading
    /// "CT presets". Dropped as soon as the window no longer matches it.
    wl_preset: Option<usize>,

    /// *Modules ▶ Image registration*: the registration section is part of
    /// the modules panel. Persisted between runs.
    module_registration: bool,
    /// *Modules ▶ Image simulation*: the simulation section is part of the
    /// modules panel. Persisted between runs.
    module_simulation: bool,
    /// *Modules ▶ Structures propagation*: the propagation section is part
    /// of the modules panel. Persisted between runs.
    module_propagation: bool,
    /// The left panel is expanded (View ▶ Data tree, F9, or the arrow on the
    /// panel edge). It holds the data tree and nothing else.
    side_open: bool,
    /// The right panel is expanded (View ▶ Modules, F10, or the arrow on the
    /// panel edge). It holds the module sections. Collapse both and the
    /// views have the whole window.
    right_open: bool,

    /// Light / dark / follow-the-system appearance, persisted between runs.
    theme: egui::ThemePreference,
    /// Which graphics backend the *next* run will use. Read at startup by
    /// `main`, before the window exists, so changing it here only takes
    /// effect after a restart — which the menu says out loud.
    graphics_backend: crate::gfx::Backend,
    /// What the graphics library actually started with, which is not always
    /// what was asked for: `Settings > Graphics backend` reports this one.
    active_backend: Option<crate::gfx::Backend>,
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
        // Before anything is drawn: the font stack that makes every glyph in
        // the interface render (see `glyphs`).
        glyphs::install(&cc.egui_ctx);
        cc.egui_ctx.set_theme(prefs.theme);
        let models_dir = prefs
            .models_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let archive_dir = prefs
            .archive_dir
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
            // What the last run ended with, ready for the start screen's
            // *Restore the last session*; it becomes this run's session as
            // soon as anything is loaded.
            session: [Vec::new(), Vec::new()],
            last_session: prefs.session.clone(),
            restore_queue: Vec::new(),
            archive_has_data: false,
            archive_checked_at: f64::NEG_INFINITY,
            pending_load: None,
            pending_load_files: None,
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
            group_registration: None,
            reg_group: None,
            reg_group_moving: 0,
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
            obj_remove: None,
            item_action: None,
            rename: None,
            rename_request: None,
            tick_anchor: None,
            maximized: None,
            reg_apply_invert: false,
            window_center: 40.0,
            window_width: 400.0,
            show_contours: true,
            show_crosshair: true,
            show_labels: true,
            show_isocenters: true,
            models_dir,
            archive_dir,
            pacs: None,
            pacs_job: None,
            models_open: false,
            models_scan: Vec::new(),
            models_scan_at: f64::NEG_INFINITY,
            models_job: None,
            models_result: None,
            autoseg_job: None,
            autoseg_slot: 0,
            autoseg_dialog: None,
            autoseg_pending: None,
            body_job: None,
            body_slot: 0,
            body_dialog: None,

            dvh_open: false,
            dvh_dialog: None,
            dvh_job: None,

            combine_job: None,
            combine_slot: 0,
            combine_dialog: None,
            motion_job: None,
            motion_slot: 0,
            motion_dialog: None,
            motion_recipe: None,
            motion_reports: Vec::new(),
            motion_results_open: false,
            motion_sel: 0,
            motion_cmp: None,
            transfer_dialog: None,
            compare_dialog: None,
            fourd_action: None,

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
            wl_preset: None,
            module_registration: prefs.module_registration,
            module_simulation: prefs.module_simulation,
            module_propagation: prefs.module_propagation,
            side_open: true,
            right_open: true,
            theme: prefs.theme,
            graphics_backend: prefs.graphics_backend,
            active_backend: cc
                .wgpu_render_state
                .as_ref()
                .map(|r| crate::gfx::Backend::from_wgpu(r.adapter.get_info().backend)),
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
        let default_archive = crate::archive::default_root().display().to_string();
        let archive_dir =
            if self.archive_dir.trim().is_empty() || self.archive_dir.trim() == default_archive {
                None
            } else {
                Some(PathBuf::from(self.archive_dir.trim()))
            };
        match settings::save(&Settings {
            theme: self.theme,
            models_dir,
            archive_dir,
            module_registration: self.module_registration,
            module_simulation: self.module_simulation,
            module_propagation: self.module_propagation,
            session: self.session.clone(),
            graphics_backend: self.graphics_backend,
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

    /// Does the local archive hold any patient? Re-read at most twice a
    /// second, since the start screen asks every frame.
    pub(super) fn archive_has_data(&mut self, now: f64) -> bool {
        if now - self.archive_checked_at < 0.5 {
            return self.archive_has_data;
        }
        self.archive_checked_at = now;
        let root = crate::archive::root_from_setting(&self.archive_dir);
        self.archive_has_data = crate::archive::Archive::new(root).has_patients();
        self.archive_has_data
    }

    /// Is there a session from the last run to offer?
    pub(super) fn has_last_session(&self) -> bool {
        self.last_session.iter().any(|paths| !paths.is_empty())
    }

    /// Load again what the program was showing when it was last closed.
    ///
    /// Folders move and get cleaned up between sessions, so the sources are
    /// checked first: if any of them is gone the session is dropped, with a
    /// message saying so, and the start screen no longer offers it.
    pub(super) fn restore_last_session(&mut self) {
        let missing: Vec<String> = self
            .last_session
            .iter()
            .flatten()
            .filter(|p| !p.exists())
            .map(|p| p.display().to_string())
            .collect();
        if !missing.is_empty() {
            self.error = Some(format!(
                "The last session cannot be restored. This is no longer on \
                 disk:\n{}",
                missing.join("\n")
            ));
            self.last_session = [Vec::new(), Vec::new()];
            self.session = [Vec::new(), Vec::new()];
            self.persist_settings();
            return;
        }
        if !self.last_session[1].is_empty() {
            self.comparison = true;
        }
        for (slot, paths) in self.last_session.clone().iter().enumerate() {
            for path in paths {
                self.restore_queue.push((slot, path.clone()));
            }
        }
    }

    pub(super) fn close_comparison(&mut self) {
        self.slots[1] = StudySlot::empty();
        self.forget_sources(1);
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

    /// Pick one or more DICOM files.
    ///
    /// "All files" comes first because DICOM files very often have no
    /// extension at all; the `.dcm` filter is the convenience, not the rule.
    pub(super) fn pick_files(title: &str) -> Option<Vec<PathBuf>> {
        rfd::FileDialog::new()
            .set_title(title)
            .add_filter("All files", &["*"])
            .add_filter("DICOM", &["dcm", "DCM", "ima", "IMA", "dic", "img"])
            .pick_files()
            .filter(|v| !v.is_empty())
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
            } else if let Some((slot, paths)) = self.pending_load_files.take() {
                self.start_load_files(slot, paths);
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

        // Poll the tool windows' workers.
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
        if let Some((slot, result)) =
            poll_tool_job(&mut self.body_job, &ctx, BODY_CONTOUR.name, &mut self.error)
        {
            self.on_body_done(slot, result);
        }
        if let Some((slot, result)) =
            poll_tool_job(&mut self.combine_job, &ctx, COMBINE.name, &mut self.error)
        {
            self.on_combine_done(slot, result);
        }
        match poll_job(&mut self.dvh_job, &ctx, "DVH", &mut self.error) {
            Some(Ok(done)) => self.on_dvh_done(done),
            Some(Err(e)) if !progress::is_cancellation(&e) => {
                self.error = Some(format!("DVH failed: {e:#}"));
            }
            _ => {}
        }
        if let Some((slot, outcome)) =
            poll_tool_job(&mut self.motion_job, &ctx, MOTION.name, &mut self.error)
        {
            self.on_motion_done(slot, outcome);
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

        // Poll an archive job — a scan, an import, an upload or a removal.
        match poll_job(&mut self.pacs_job, &ctx, "Archive", &mut self.error) {
            Some(Ok(outcome)) => self.on_pacs_done(outcome),
            Some(Err(e)) => self.error = Some(format!("Archive: {e:#}")),
            None => {}
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
            let (undo, esc, smaller, bigger, toggle_side, toggle_right) = ctx.input(|i| {
                (
                    i.modifiers.command && i.key_pressed(egui::Key::Z),
                    i.key_pressed(egui::Key::Escape),
                    i.key_pressed(egui::Key::OpenBracket),
                    i.key_pressed(egui::Key::CloseBracket),
                    i.key_pressed(egui::Key::F9),
                    i.key_pressed(egui::Key::F10),
                )
            });
            if toggle_side {
                self.side_open = !self.side_open;
            }
            if toggle_right {
                self.right_open = !self.right_open;
            }
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
        self.modules_panel(ui);
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
        if let Some(obj) = self.obj_remove.take() {
            self.remove_object(obj);
        }
        if let Some(action) = self.item_action.take() {
            self.apply_item_action(action);
        }
        if let Some(action) = self.fourd_action.take() {
            self.apply_fourd_action(action);
        }
        if let Some(target) = self.rename_request.take() {
            self.open_rename(target);
        }
        self.modals(&ctx);
    }
}
