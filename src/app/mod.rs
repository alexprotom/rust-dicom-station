//! The egui application: menu bar, toolbar, side panel, and one row per
//! workspace (a second in comparison mode) of up to three linked panes each -
//! the three MPR planes and the 3D surface scene, chosen under
//! *Settings ▸ View layout*.

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
use crate::export;
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
use crate::settings::{self, PaneKind, Settings};
use crate::simulate::{self, SimParams};
use crate::testdata;
use crate::volume::{ViewPlane, Volume};
use crate::workflow;
use crate::zoo;

use pick::{CSV_FILES, PROTOCOL_FILES};

mod auto_tools;
mod body_win;
mod box_seg;
mod chrome;
mod combine;
mod compare_win;
mod contour_edit;
mod d3;
mod data_tree;
mod derived_app;
mod detach;
mod dialogs;
mod dose_est;
mod drr_win;
mod dvh_win;
mod export_win;
mod form;
mod fourd_tree;
mod glyphs;
mod home;
mod image_gen;
mod img_info;
mod interactive_seg;
mod jobs;
mod livewire_app;
mod matrix_edit;
mod models_win;
mod motion_results;
mod motion_win;
mod pacs_remote;
mod pacs_server_win;
mod pacs_tree;
mod pacs_win;
mod panels;
mod pick;
mod planar;
mod play;
mod poi;
mod prompt_seg;
mod propagate_win;
mod record;
mod reg2d3d;
mod reg_panel;
mod reg_shape;
mod rename;
mod report_view;
mod run_report;
mod seg;
mod seg_edit;
mod seg_engines;
mod sets;
mod slot;
mod snapshot;
mod stats_win;
mod struct_tools;
mod synth_win;
mod testdata_win;
mod theme;
mod transfer_win;
mod tree;
mod tree_dose;
mod tree_menus;
mod tree_model;
mod tree_planar;
mod tree_plans;
mod tree_records;
mod tree_registrations;
mod tree_reports;
mod tree_segmentations;
mod tree_structures;
mod views;
mod widgets;
mod workflow_edit;
mod workflow_run;
mod workspace_pick;

use d3::D3Window;
use drr_win::{DrrRun, DrrState};
use jobs::{cancel_if, poll_job, poll_tool_job, Job, LoadResult};
use pacs_win::{PacsOutcome, PacsWindow};
use propagate_win::{GroupRegistration, PropOutcome, PropagateDialog};
use reg_panel::{ActiveRegistration, FusionSide};
use reg_panel::{RegImage, RegInit, RegOutcome, RegRoi};
use rename::{RenameDialog, RenameTarget};
use seg_engines::*;
use sets::{Destination, ItemAction, PhasesCopy, PoiAct, SetAction, SetKind, SetRef};
use slot::{plane_index, StudySlot};
use theme::*;
use tree::{ObjKind, ObjRef, TreeAction, TreeOp, TreeSel};
use widgets::*;

/// How many workspaces the program will hold at once, from
/// [`crate::settings::MAX_WORKSPACES`].
///
/// Four is the number a comparison ever really needs - a plan, a repeat, a
/// phase and the thing being checked against - and it is also where the
/// screen runs out: each open workspace takes a row of the central area,
/// and a fifth row would be a strip of pixels rather than an image.
use crate::settings::MAX_WORKSPACES;

const SLOT_NAMES: [&str; MAX_WORKSPACES] = ["A", "B", "C", "D"];

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
    /// Start a new custom group from several series at once - the ones
    /// ticked in the tree, or every series of a modality node - ordered by
    /// [`fourd::group_from`].
    NewFrom { slot: usize, series: Vec<usize> },
    /// Add several series to an existing group, as phases.
    AddMany {
        slot: usize,
        group: usize,
        series: Vec<usize>,
    },
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
    /// The model to run.
    model: zoo::AutoModel,
    /// The last choice within TotalSegmentator's 117-class CT family, kept
    /// while another model is picked.
    total: autoseg::Variant,
    device: autoseg::DevicePref,
    /// Which sub-models of a model that has several to run (the `total`
    /// 1.5 mm parts, the MR parts), one per [`zoo::AutoModel::part_names`].
    parts: Vec<bool>,
    /// Where the result goes, and whether the run covers every phase of
    /// the displayed series' 4D group. The output part is asked again in
    /// the results window, where the organs are picked; the scope has to
    /// be known before the run.
    output: ToolOutput,
}

/// A finished auto-segmentation waiting for the user to choose organs: one
/// label map per volume it ran on - the displayed series, or every phase
/// of its 4D group.
struct AutosegPending {
    slot: usize,
    phases: Vec<(PhaseInfo, autoseg::AutosegResult)>,
    /// Every class any phase found, in the order of the list; the volumes
    /// shown are the mean over the phases that have the class.
    organs: Vec<autoseg::OrganHit>,
    selected: Vec<bool>,
    output: ToolOutput,
    /// Name the structures by AAPM TG-263 where a class has a TG-263 name.
    tg263: bool,
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

/// A volume (held weakly, so the cache does not keep it alive) and the
/// windows of [`RELATIVE_PRESETS`] worked out for it.
type RelativeWindows = (std::sync::Weak<Volume>, Vec<(f32, f32)>);

/// The window presets of a modality without a fixed scale (MR, PET, ...):
/// (name, low, high percentile of the image's values, the padding value
/// left out). The first is the series' own window, not a percentile pair.
const RELATIVE_PRESETS: &[(&str, f32, f32)] = &[
    ("Series default", 0.0, 0.0),
    ("Auto, 1 - 99 %", 1.0, 99.0),
    ("Wide, 0.5 - 99.9 %", 0.5, 99.9),
    ("Narrow, 5 - 95 %", 5.0, 95.0),
    ("Dark tissue, 0 - 60 %", 0.0, 60.0),
    ("Bright tissue, 30 - 99.9 %", 30.0, 99.9),
];

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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SegTool {
    None,
    /// Paint into the active segmentation (Alt temporarily erases).
    Brush,
    /// Erase from the active segmentation.
    Erase,
    /// Seeded region growing: press to seed, drag up/down to widen/narrow
    /// the intensity tolerance, release to commit (Esc cancels).
    Grow,
    /// Draw a polygon into the active RT structure, click by click.
    Polygon,
    /// The same, through a closed spline fitted to the clicked points.
    Spline,
    /// Draw a contour freehand, in one drag.
    Freehand,
    /// Push the contour under the pointer around, like a finger in clay.
    Nudge,
    /// Paint into the active RT structure: a round brush on the patient that
    /// pushes the contour lines, rather than the voxel brush's mask.
    ContourBrush,
    /// Draw along the edge under the pointer: the curve between two clicks
    /// is the cheapest path through the image gradient, not a straight line.
    LiveWire,
}

impl SegTool {
    /// Does this tool edit RT structure contours rather than voxel masks?
    fn draws_contours(self) -> bool {
        matches!(
            self,
            SegTool::Polygon
                | SegTool::Spline
                | SegTool::Freehand
                | SegTool::Nudge
                | SegTool::ContourBrush
                | SegTool::LiveWire
        )
    }
}

/// What a drawn contour does to the ones already on the slice.
///
/// The names and the *auto* rule are the ones planning systems have
/// settled on, because a planner should not have to learn a second set of
/// habits to draw a contour here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DrawMode {
    /// Crosses nothing: a new contour. Crosses something, starting inside
    /// it: extend. Crosses something, starting outside: cut.
    Auto,
    /// Always add.
    Extend,
    /// Always cut.
    Subtract,
}

impl DrawMode {
    const ALL: [DrawMode; 3] = [DrawMode::Auto, DrawMode::Extend, DrawMode::Subtract];
    fn label(self) -> &'static str {
        match self {
            DrawMode::Auto => "Auto",
            DrawMode::Extend => "Extend",
            DrawMode::Subtract => "Subtract",
        }
    }
    fn hint(self) -> &'static str {
        match self {
            DrawMode::Auto => {
                "New contour when the drawing crosses nothing; extend when it starts \
                 inside an existing one; cut when it starts outside"
            }
            DrawMode::Extend => "Always add to the structure",
            DrawMode::Subtract => "Always cut away from the structure",
        }
    }
}

/// A contour being drawn: the points collected so far, in this slot's voxel
/// coordinates, on one slice of one view plane.
struct DrawState {
    slot: usize,
    plane: crate::volume::ViewPlane,
    slice: usize,
    /// Fractional voxel coordinates of the clicked (or dragged) points.
    pts: Vec<[f64; 3]>,
}

/// One step of contour undo: the whole geometry of one ROI before an edit.
/// A structure is a few hundred points a slice, so keeping the lot is both
/// simpler and cheaper than a per-vertex journal.
struct RoiSnapshot {
    slot: usize,
    set: usize,
    roi: usize,
    contours: Vec<crate::rtstruct::Contour>,
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

// Planar panes

/// One planar pane of a row ([`PaneKind::Planar`]): which of the
/// workspace's planar images it shows and how.
///
/// A pane shows one image or several, stacked: the scrubber along its
/// bottom edge moves through the stack, and the two images either side of
/// it are compared the way [`Compare`] says - faded into each other, in a
/// checkerboard, as a difference or one's edges over the other. Images are
/// named by their index in `study.planar_images`; removing one renumbers the
/// panes ([`ViewerApp::planar_removed`]). Zoom and pan are in the units of an
/// MPR view (screen pixels per millimetre, millimetres off the centre) and
/// every image of the stack is drawn at that scale about the same centre, so
/// two renderings of one geometry lie exactly on each other.
#[derive(Default)]
struct PlanarView {
    /// The stack, bottom first. Empty: nothing given to the pane yet.
    layers: Vec<PlanarLayer>,
    /// Where the scrubber stands in the stack: 0 is the first image, 1 the
    /// second, and a fraction is between the two.
    blend: f32,
    compare: Compare,
    /// Screen pixels per mm; 0 means auto-fit.
    zoom: f32,
    pan: Vec2,
    /// Fractional mouse-wheel line accumulator for stepping through the
    /// stack.
    scroll_accum: f32,
    /// The difference or edge image of the pair being compared, and what it
    /// was made from.
    cmp_tex: Option<TextureHandle>,
    cmp_key: Option<u64>,
    /// The report the pane shows instead of a stack (the additional pane
    /// only): index into `study.reports`.
    report: Option<usize>,
    /// Whether the report's page is shown rather than its tables.
    report_page: bool,
    /// The texture of that page, and the page it was made from.
    page_tex: Option<TextureHandle>,
    page_key: Option<PlanarKey>,
}

/// One image of a planar pane's stack, with the window it is shown with.
///
/// The window is the image's own in this pane: two planar images of one
/// workspace - a kV image and its DRR - have nothing like a shared modality
/// window.
struct PlanarLayer {
    /// Index into `study.planar_images`.
    image: usize,
    /// (centre, width), and the identity of the image it was set for: an
    /// image replaced under the same index starts from its own window.
    wl: (f32, f32),
    wl_for: Option<PlanarKey>,
    tex: Option<TextureHandle>,
    /// What `tex` was made from: the image and the window.
    tex_key: Option<(PlanarKey, u32, u32)>,
}

impl PlanarLayer {
    fn new(image: usize) -> Self {
        PlanarLayer {
            image,
            wl: (0.0, 1.0),
            wl_for: None,
            tex: None,
            tex_key: None,
        }
    }
}

/// How a planar pane compares the two images either side of its scrubber.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Compare {
    /// The upper one over the lower, as opaque as the scrubber is far
    /// towards it.
    #[default]
    Fade,
    /// Alternate squares of the two.
    Checker,
    /// The signed difference of what the two show, mid-grey where they
    /// agree (same raster only).
    Difference,
    /// The lower one, with the edges of the upper drawn over it.
    Edges,
    /// The two in turn, twice a second.
    Flicker,
}

/// Identity of one planar image's pixels: the buffer and its shape. The
/// label can be renamed and the index shifts when an earlier image is
/// removed; neither changes what the texture has to show.
type PlanarKey = (usize, usize, usize);

#[inline]
fn mix(h: u64, v: u64) -> u64 {
    (h ^ v).wrapping_mul(0x100000001b3)
}

/// One image series of one workspace, as the registration module names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct RegPick {
    pub slot: usize,
    /// Index into the study's series list.
    pub series: usize,
}

// Application

/// What the edit section's volume summary was computed for: slot, set, ROI,
/// `settings_gen`, the volume's identity, the contour and point counts.
type EditSummaryKey = (usize, usize, usize, u64, usize, usize, usize);

pub struct ViewerApp {
    slots: [StudySlot; MAX_WORKSPACES],
    /// Which workspaces are on screen, A first.
    ///
    /// A workspace holding data is always shown; the flag is what lets one
    /// be opened before anything is loaded into it (the row says so and
    /// offers to load), and what *View ▸ Workspaces* switches to put one
    /// aside without emptying it.
    shown: [bool; MAX_WORKSPACES],
    /// What each row shows, left to right - up to four panes, chosen under
    /// *Settings ▸ View layout* and remembered between runs. The panes of a
    /// row split its width evenly, so a row of one is one large image.
    view_rows: [Vec<PaneKind>; MAX_WORKSPACES],
    /// Propagate the crosshair between studies via patient coordinates.
    link_studies: bool,
    /// Slot whose readout is expanded in the status bar.
    hovered_slot: usize,
    /// Where each pane was drawn this pass, so the recorder knows what part
    /// of the window a run is playing in.
    pane_rects: Vec<(usize, PaneKind, Rect)>,
    /// The run being recorded, if any, and what the next one will be
    /// written as.
    rec: Option<record::Recording>,
    rec_format: record::RecFormat,
    /// Stop a recording after this many frames, whatever the run does: an
    /// endless loop would otherwise fill memory with pictures.
    rec_max: usize,
    /// What the last recording did, for the module to say.
    rec_status: Option<String>,
    /// Where each workspace's row of the central area was drawn this pass,
    /// which is what *File ▸ Save image* cuts out of the window.
    row_rects: [Option<Rect>; MAX_WORKSPACES],
    /// The hand-edited transform matrix of the registration panel, and
    /// whether it is the one to use.
    reg_matrix: matrix_edit::ManualMatrix,
    /// Which of the module's transforms its matrix grid is showing - the
    /// active registration, or one phase of a group run.
    reg_matrix_pick: usize,
    /// What the propagation in flight was asked to do to the landed masks,
    /// recorded when it started so its table reports the run rather than
    /// whatever the ticks say by the time it finishes.
    propagate_finish: crate::propagate::Finish,
    /// The same, for structure propagation and for transfer by relationship.
    prop_matrix: matrix_edit::ManualMatrix,
    transfer_matrix: matrix_edit::ManualMatrix,
    /// The buttons on every pane are folded away behind the ◀ on its bar.
    /// One switch for all of them, remembered between runs.
    pane_buttons_hidden: bool,
    /// The *Save image* dialog, and the picture it has asked for.
    save_img: Option<snapshot::SaveImgDialog>,
    snap: Option<snapshot::PendingShot>,
    snap_status: Option<String>,
    /// The hand on every viewport's bar: while it is on, dragging with the
    /// left button moves the image instead of placing the crosshair. One
    /// switch for all the views, so the pointer means the same thing
    /// wherever it happens to be.
    hand_pan: bool,

    loading: Option<Job<LoadResult>>,
    /// A load queued behind the one in flight (slot, directory).
    /// What each workspace has been loaded from this run, in order: the
    /// *Restore the last session* button on the start screen replays it, and
    /// it is written to the settings file as it changes.
    session: [Vec<PathBuf>; MAX_WORKSPACES],
    /// Sources still to load for a restore, drained one at a time as each
    /// load finishes (the loader takes one folder at a time).
    restore_queue: Vec<(usize, PathBuf)>,
    /// The session the previous run ended with, as read from the settings
    /// file. Emptied when it turns out its data is gone.
    last_session: [Vec<PathBuf>; MAX_WORKSPACES],
    pending_load: Option<(usize, PathBuf)>,
    /// The same, for an explicit file selection (slot, files).
    pending_load_files: Option<(usize, Vec<PathBuf>, Option<Vec<PathBuf>>)>,
    /// A series to put on display once the current read finishes (slot,
    /// series UID): asked for while something else was loading. By UID,
    /// not index, since the study may gain series in between.
    pending_switch: Option<(usize, String)>,
    error: Option<String>,
    /// A one-line confirmation shown in a small modal (e.g. a written file).
    notice: Option<String>,

    // Registration (direction selectable: either study can be the fixed one).
    registration: Option<ActiveRegistration>,
    /// The last 4D group registered phase by phase, so propagating onto it
    /// does not repeat the registrations. Cleared when the registration is.
    group_registration: Option<GroupRegistration>,
    /// Which 4D group the registration module runs against, when it runs
    /// against one rather than the other workspace's displayed volume.
    reg_group: Option<(usize, usize)>,
    /// The payload carries the slot that was used as the fixed image.
    reg_job: Option<SegJob<RegOutcome>>,
    /// Fixed-image slot for the *next* registration run (0 = A, 1 = B).
    fusion_on: bool,
    fusion_weight: f32,
    /// Bumped when the registration result changes → fusion cache rebuild.
    reg_gen: u64,
    /// The fixed and the moving image of the next run.
    reg_fixed: RegPick,
    reg_moving: RegPick,
    /// Which image the fusion overlay is drawn on.
    fusion_side: FusionSide,
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
    /// Which structure of the fixed workspace restricts the next run.
    reg_roi: RegRoi,
    /// Margin the region is grown by, mm.
    reg_margin_mm: f64,
    /// Where the next run starts its search.
    reg_init: RegInit,
    /// *Align by structures*: the pairs ticked and how they are fitted.
    reg_shape: reg_shape::ShapeRegState,

    // The deformation vector field of the active registration.
    field_on: bool,
    /// Draw only what the warp adds to the rigid alignment
    /// ([`Transform3::warp_only`]).
    field_warp_only: bool,
    /// A phase of the group registration asked to be shown, waiting for its
    /// series to be put on display: (workspace, series UID, phase index).
    pending_phase_field: Option<(usize, String, usize)>,
    field_style: FieldStyle,
    field_step_mm: f64,
    /// Arrows are drawn this many times their true length.
    field_scale: f32,
    field_color: bool,
    /// A re-sampling of the field after the lattice step changed.
    field_job: Option<Job<VectorField>>,

    // Modules ▶ Image generation: the module's workspace, and the DRR
    // section's state (made the first time the section is drawn) and run.
    gen: image_gen::ImageGen,
    drr: Option<DrrState>,
    drr_job: Option<Job<anyhow::Result<DrrRun>>>,
    /// *Image registration ▶ 2D/3D registration*: its state, and its run.
    r2d: Option<reg2d3d::Reg2d3dState>,
    r2d_job: Option<Job<reg2d3d::Reg2d3dJobOut>>,
    /// The 2D/3D section's remembered choices (persisted).
    r2d_prefs: settings::Reg2d3dPrefs,

    // Tools ▶ Propagate structures.
    /// The window, when open.
    propagate_dialog: Option<PropagateDialog>,
    /// The payload carries the destination slot.
    propagate_job: Option<SegJob<PropOutcome>>,
    /// A copy of structures onto every phase of a 4D group, with what to do
    /// with the copies once they are made.
    phases_job: Option<(
        PhasesCopy,
        Job<anyhow::Result<Vec<workflow::group::PhaseCopy>>>,
    )>,

    // Study transform simulator (registration QA).
    sim_source: usize,
    /// Where *Image simulation* writes its transformed copy.
    sim_target: usize,
    sim_params: SimParams,
    sim_job: Option<Job<(usize, LoadedStudy)>>,
    last_sim: Option<String>,
    // DICOM export (File ▶ Export DICOM). One window for both workspaces: what
    // goes out is chosen inside it, not in the menu.
    /// Window visibility.
    export_open: bool,
    /// Output folder as edited in the window.
    export_dir: String,
    /// The selection tree and every identifier it carries, built when the
    /// window opens.
    export_plan: Option<export::ExportPlan>,
    export_job: Option<Job<anyhow::Result<export::ExportSummary>>>,
    export_result: Option<String>,
    /// What the finished run could not do exactly as asked.
    export_warnings: Vec<String>,

    // Built-in synthetic test-data generator.
    /// Dialog visibility.
    gen_open: bool,
    gen_params: GenParams,
    /// Output folder as edited in the dialog (defaults to the app folder).
    gen_dir: String,
    gen_job: Option<Job<anyhow::Result<(usize, PathBuf)>>>,
    gen_result: Option<String>,
    /// Load the generated study into workspace A once it has been written.
    gen_load_after: bool,

    // Tools ▶ Download test data: the repository's data-test/ from GitHub.
    /// Dialog visibility.
    testdata_open: bool,
    /// Destination folder as edited in the dialog (defaults to
    /// `<data folder>/data-test`).
    testdata_dir: String,
    testdata_job: Option<Job<anyhow::Result<testdata::Summary>>>,
    testdata_result: Option<String>,
    /// Load the first study into workspace A once everything is there.
    testdata_load_after: bool,

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
    /// The data-tree row the keyboard acts on: the one last clicked, in
    /// either button. F2 renames it, Delete removes it. The same address a
    /// rename takes, because every row the tree shows can be renamed, and
    /// so every row has one.
    tree_focus: Option<RenameTarget>,
    /// Image series ticked with Ctrl-click in the data tree, as (workspace,
    /// series UID): the selection *New 4D group from the selected series*
    /// and the other multi-series actions work on. UIDs, because series are
    /// renamed, removed and moved while the selection stands.
    tree_picks: Vec<(usize, String)>,
    /// Anchor of the last check-box click in a structure / segment list, so
    /// Shift-click can extend a range from it.
    tick_anchor: Option<(SetRef, usize)>,
    /// When set, this single (slot, view) fills the whole central area.
    maximized: Option<(usize, PaneKind)>,
    /// Invert REG matrices before applying them as the active registration.
    reg_apply_invert: bool,

    /// Each workspace's display window (centre, width). Workspaces that
    /// show the same modality move together ([`ViewerApp::set_window`]).
    window_center: [f32; MAX_WORKSPACES],
    window_width: [f32; MAX_WORKSPACES],
    /// The window each workspace's series came with (its DICOM window, or
    /// one worked out from its values), the *Series default* preset.
    series_window: [(f32, f32); MAX_WORKSPACES],
    /// The relative presets of a volume (MR, PET...): its percentile
    /// windows, worked out once per volume.
    wl_relative: Option<RelativeWindows>,

    show_contours: bool,
    show_crosshair: bool,
    show_labels: bool,
    show_isocenters: bool,

    // Interactive segmentation.
    seg_tool: SegTool,
    /// Brush radius in mm (shared by paint and erase).
    brush_radius_mm: f32,
    /// Which tissue the contour brush is allowed to paint into: the smart
    /// brush's edge setting.
    brush_band: contour_edit::EdgeBand,
    /// Pull the interpolated contours onto the edges the image shows:
    /// smart interpolation.
    interp_snap: bool,
    /// Whether the contour brush's current stroke is cutting: decided once,
    /// when an Auto stroke starts, from whether it started inside.
    brush_auto_cut: bool,
    /// The structure the ✨ Grow tool is not allowed to leave, if any.
    grow_limit: Option<combine::ItemRef>,
    /// How far past the band's own threshold the brush may still reach,
    /// as a fraction of the display window. 0 is the bare threshold.
    brush_sensitivity: f32,
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
    /// The contour being drawn, if any.
    draw: Option<DrawState>,
    /// What the next drawn contour does to the ones already there.
    draw_mode: DrawMode,
    /// Undo stack of the contour tools (see [`RoiSnapshot`]).
    roi_undo: Vec<RoiSnapshot>,
    /// Counter for naming newly created ROIs.
    roi_counter: usize,
    /// The same for points of interest, so their default names do not
    /// collide with the structures'.
    poi_counter: usize,
    /// The working geometry of the ROI under the contour tools.
    edit: Option<contour_edit::EditStack>,
    /// When the working stack was last written into its structure (see
    /// `contour_edit::OFF_AXIS_FLUSH`).
    edit_flushed: Option<std::time::Instant>,
    /// The interpolated slices shown as a preview, until accepted.
    interp: Option<contour_edit::InterpPreview>,
    /// Contours copied from one slice, and the axis they were cut on.
    contour_clip: Option<(usize, crate::contours::Region)>,
    /// *Modules ▶ Structure editor*: which workspace it works on and the
    /// numbers its buttons apply.
    tools: struct_tools::StructTools,
    /// The live-wire's cost image and current anchor, kept between frames.
    wire: Option<livewire_app::WireState>,
    /// Whether the live wire learns from every accepted segment. Kept here,
    /// outside the per-slice wire state, so the choice survives a change of
    /// slice and can be made before the first click.
    wire_learn: bool,
    /// The structure-details table.
    stats_dialog: Option<stats_win::StatsDialog>,

    /// Root folder of the downloaded network weights, shared by the three
    /// engines (persisted in the settings file; blank = the default).
    models_dir: String,

    /// The main PACS storage folder (persisted; blank = the default).
    archive_dir: String,
    /// The PACS's other storage folders (persisted).
    archive_more: Vec<PathBuf>,

    // Tools ▶ PACS: the Picture Archiving and Communication System window.
    /// The window, when open.
    pacs: Option<PacsWindow>,
    /// The PACS job in flight - a scan, an import, an upload, a name, a
    /// cohort, a move or a removal.
    pacs_job: Option<Job<anyhow::Result<PacsOutcome>>>,
    /// A call to a paired PACS server in flight (`pacs_remote.rs`).
    pacs_remote_job: Option<Job<anyhow::Result<pacs_remote::RemoteOutcome>>>,
    // PACS ▶ Settings ▶ PACS server: the server on this machine
    // (`pacs_server_win.rs`), while that page is shown.
    pacs_server: Option<pacs_server_win::ServerWindow>,
    pacs_server_job: Option<Job<anyhow::Result<pacs_server_win::ServerOutcome>>>,

    // Tools ▶ Downloaded models: the inventory window.
    models_open: bool,
    /// The inventory with each model's state, re-read at most twice a second.
    models_scan: Vec<(models::ModelAsset, models::AssetStatus)>,
    /// `ctx` time the scan above was taken at.
    models_scan_at: f64,
    /// A download / update batch in flight; its payload is the summary line.
    models_job: Option<Job<anyhow::Result<String>>>,
    models_result: Option<String>,
    /// The TotalSegmentator licence number (persisted in the user's own
    /// settings file; blank = none).
    ts_licence: String,
    /// The licence field shows its text.
    ts_licence_shown: bool,
    /// nnU-Net model folders the user added (persisted).
    nnunet_folders: Vec<PathBuf>,
    /// Folders of TotalSynth checkpoints the user added (persisted).
    synth_folders: Vec<PathBuf>,
    /// Why a model folder (nnU-Net or TotalSynth) could not be used, by
    /// folder.
    nnunet_folder_errors: Vec<(PathBuf, String)>,

    // Auto-segmentation (TotalSegmentator re-implementation, see `autoseg`).
    /// The payload carries the slot the volume came from.
    autoseg_job: Option<SegJob<Vec<(PhaseInfo, autoseg::AutosegResult)>>>,
    /// Slot currently being segmented (progress shown in its sidebar section).
    autoseg_slot: usize,
    /// The tool window, when open; it stays open while a run is in flight.
    autoseg_dialog: Option<AutosegDialog>,
    /// Finished result awaiting organ selection.
    autoseg_pending: Option<AutosegPending>,
    /// The results window's last TG-263 naming choice, carried to the next
    /// run.
    autoseg_pending_tg263: bool,

    // Synthetic CT (see `sct`, `synth`): the run and the evaluation, one at
    // a time, on the workspace of the Image synthesis module.
    synth: synth_win::SynthState,
    synth_job: Option<SegJob<synth_win::SynthDone>>,
    synth_eval_job: Option<SegJob<synth_win::EvalDone>>,
    synth_slot: usize,

    // Body / External contouring (see `bodymask`) - the one tool that can
    // answer with no network at all.
    body_job: Option<SegJob<Vec<(PhaseInfo, bodymask::BodyResult)>>>,
    body_slot: usize,
    /// The tool window, when open; it stays open across runs.
    body_dialog: Option<body_win::BodyDialog>,

    // Dose-volume histograms (see `dvh`), in a window of their own.
    dvh_open: bool,
    dvh_dialog: Option<dvh_win::DvhDialog>,
    dvh_job: Option<Job<anyhow::Result<dvh_win::DvhDone>>>,

    // Structure algebra (see `structops`): combining contours and segments.
    combine_job: Option<SegJob<combine::CombineResult>>,
    /// Re-evaluating one derived structure, and the slot it belongs to.
    derived_job: Option<SegJob<derived_app::DerivedResult>>,
    derived_slot: usize,
    /// Derived structures still waiting for their turn (set, ROI).
    derived_queue: Vec<(usize, usize)>,
    /// Per slot: the derived statuses of the active structure set.
    derived: [derived_app::DerivedCache; MAX_WORKSPACES],
    combine_slot: usize,
    combine_dialog: Option<combine::CombineDialog>,

    // 4D motion / ITV analysis (see `motion` and `fourd`).
    motion_job: Option<SegJob<motion_win::MotionOutcome>>,
    motion_slot: usize,
    motion_dialog: Option<motion_win::MotionDialog>,
    /// The last run's settings, re-applicable to another workspace / study.
    motion_recipe: Option<motion_win::MotionRecipe>,
    /// Every finished run of this session, newest last.
    motion_reports: Vec<crate::motion::MotionReport>,
    /// The results window: visibility, selected run, comparison run.
    motion_results_open: bool,
    motion_sel: usize,
    motion_cmp: Option<usize>,

    // Tools ▶ Transfer by relationship.
    transfer_dialog: Option<transfer_win::TransferDialog>,

    // Tools ▶ Structure comparison.
    compare_dialog: Option<compare_win::CompareDialog>,

    /// Deferred 4D-group edit from the data tree's context menus.
    fourd_action: Option<FourDAction>,

    // Prompt-driven segmentation (SegVol re-implementation, see `segvol`).
    segvol_job: Option<SegJob<Vec<(PhaseInfo, prompt_seg::SegVolResult)>>>,
    segvol_slot: usize,
    /// The tool window, when open; it stays open across runs.
    segvol_dialog: Option<prompt_seg::SegVolDialog>,
    /// The SegVol network, kept from one prompt to the next.
    segvol_model: Arc<KeptModel<crate::segvol::model::Model>>,

    // Slice-propagating segmentation (MedSAM2 re-implementation): the drawn
    // box, the loaded engine and the prepared stack all live in one struct.
    medsam2_job: Option<SegJob<box_seg::Medsam2Done>>,
    medsam2: box_seg::Medsam2State,

    // Interactive segmentation (nnInteractive): the prompts, the kept
    // network and session, and the prompt in flight.
    nni_job: Option<SegJob<interactive_seg::NniDone>>,
    nni: interactive_seg::NniState,

    dose_mode: DoseMode,
    dose_opacity: f32,
    dose_threshold_pct: f32,
    iso_levels: Vec<IsoLevel>,

    /// Bumped whenever ROI visibility / dose settings change → cache rebuild.
    settings_gen: u64,
    /// Each workspace's data tree as last laid out, and what for
    /// (see `tree_model::tree_layout_cached`).
    tree_cache: [Option<(u64, Arc<Vec<tree_model::PatientNode>>)>; MAX_WORKSPACES],
    /// Per workspace: bumped by every sample of a contour edit (brush,
    /// nudge, hand drag, drawing), which used to bump `settings_gen` and so
    /// rebuilt every cache of every workspace - dose washes, fusion, the
    /// other workspace's contours and meshes - on each mouse move. Caches of
    /// structures read [`ViewerApp::structures_gen`], which combines the
    /// two; the rest keep reading `settings_gen` alone.
    struct_gen: [u64; MAX_WORKSPACES],
    /// The edited structure's volume and occupied-slice count, and what they
    /// were computed for: rebuilding its slice stack every frame the edit
    /// section is open cost milliseconds for a body outline.
    edit_summary_cache: Option<(EditSummaryKey, (f64, usize))>,

    /// The window preset last picked from the toolbar list (an index into
    /// [`WL_PRESETS`] for a CT, into [`RELATIVE_PRESETS`] otherwise), so
    /// the closed combo can name it instead of reading "CT presets".
    /// Dropped as soon as the window no longer matches it.
    wl_preset: Option<usize>,

    /// *Modules ▶ Image registration*: the registration section is part of
    /// the modules panel. Persisted between runs.
    module_registration: bool,
    /// *Modules ▶ Image simulation*: the simulation section is part of the
    /// modules panel. Persisted between runs.
    module_simulation: bool,
    /// *Modules ▶ Image generation*: the synthetic CT and the DRR, two
    /// sections of one module. Persisted.
    module_generation: bool,
    /// *Modules ▶ Structure propagation*: the propagation section is part
    /// of the modules panel. Persisted between runs.
    module_propagation: bool,
    /// *Modules ▶ Structure editor*: inserting, editing and combining
    /// structures is a section of the modules panel. Persisted between
    /// runs; on by default.
    module_structures: bool,
    /// *Modules ▶ Structure auto tools*: body contour and the three engines
    /// are a section of the modules panel. Persisted; on by default.
    module_auto: bool,
    /// *Modules ▶ Dose estimation*: the dose metrics table.
    module_dose: bool,
    /// *Modules ▶ Image information*: voxel spacing, slice thickness and
    /// the rest of what the displayed series is, read from its headers.
    module_info: bool,
    /// *Modules ▶ Playback*: the speed, the loop mode and what a phase
    /// change carries with it, plus the transport controls themselves.
    module_play: bool,
    /// Which module sections are unfolded (settings key `modules_open`),
    /// and whether the next panel draw has to apply that list - set by
    /// *Restore the last session*.
    modules_open: Vec<String>,
    apply_modules_open: bool,
    /// The list follows the headers only once one was unfolded in this
    /// run or the last run's list was applied; before that the settings
    /// keep the last run's list.
    modules_tracked: bool,
    dose_est: dose_est::DoseEst,
    /// Playing through slices and through 4D phases: the settings the
    /// *Playback* module carries, and the run in flight.
    play: play::PlayState,
    /// State of the *Image information* module.
    info: img_info::InfoState,
    dose_est_job: Option<Job<Vec<dose_est::DoseRow>>>,
    /// The auto tools module's state: its workspace and the section to unfold.
    auto: auto_tools::AutoTools,
    /// The toolbar's *✏ Draw structure* is unfolded: the drawing tools and
    /// the options of the one in hand are on the toolbar.
    draw_open: bool,
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
    /// effect after a restart - which the menu says out loud.
    graphics_backend: crate::gfx::Backend,
    /// What the graphics library actually started with, which is not always
    /// what was asked for: `Settings > Graphics backend` reports this one.
    active_backend: Option<crate::gfx::Backend>,
    /// Non-fatal note shown in the View menu if the settings file could not
    /// be written (e.g. a read-only installation folder).
    settings_error: Option<String>,
    home_content_height: f32,
    /// The file dialog in flight, where the platform has no native one
    /// (see `pick.rs`); never anything on the desktop.
    picker: pick::Picker,

    // Workflows (the Workflows menu; see `workflow::graph`).
    /// The workflow being edited, and its window.
    wf_editor: Option<workflow_edit::WorkflowEditor>,
    /// A New / Load that would drop unsaved changes, waiting for a yes.
    wf_confirm: Option<workflow_edit::WfPending>,
    /// The run window, and the run going on in it.
    wf_run: Option<workflow_run::WorkflowRun>,
    /// The workflow files last saved, newest first (persisted).
    recent_workflows: Vec<PathBuf>,
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
    /// The first open workspace that shows an image volume. Where a tool has
    /// to start somewhere and nobody pointed at a workspace.
    pub(super) fn first_volume_slot(&self) -> usize {
        self.volume_slot_list().first().copied().unwrap_or(0)
    }

    /// The workspace under the pointer when it shows a volume, else
    /// [`Self::first_volume_slot`] - what the toolbar and the keyboard act on.
    pub(super) fn preferred_volume_slot(&self) -> usize {
        let hovered = self.hovered_slot.min(MAX_WORKSPACES - 1);
        if self.slots[hovered].has_volume() {
            hovered
        } else {
            self.first_volume_slot()
        }
    }

    /// Is any section of the modules panel switched on?
    pub(super) fn any_module(&self) -> bool {
        self.module_registration
            || self.module_simulation
            || self.module_generation
            || self.module_propagation
            || self.module_structures
            || self.module_auto
            || self.module_dose
            || self.module_info
            || self.module_play
    }

    /// Both workspaces show an image volume.
    pub(super) fn both_volumes(&self) -> bool {
        self.volume_slot_list().len() > 1
    }

    /// The open workspaces that show an image volume.
    pub(super) fn volume_slot_list(&self) -> Vec<usize> {
        self.open_slots()
            .into_iter()
            .filter(|s| self.slots[*s].has_volume())
            .collect()
    }

    /// At least one workspace shows an image volume.
    pub(super) fn any_volume(&self) -> bool {
        !self.volume_slot_list().is_empty()
    }

    /// Which workspaces show an image volume, in the form the tool windows'
    /// workspace row takes: one flag per workspace, open or not.
    pub(super) fn volume_slots(&self) -> [bool; MAX_WORKSPACES] {
        std::array::from_fn(|s| self.is_open(s) && self.slots[s].has_volume())
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
        let archive_more = prefs.archive_more.clone();
        // Installations that predate the single `models/` root keep their
        // downloads; the folders are moved into place, never re-fetched.
        let moved = models::migrate_legacy_layout(&models::root_from_setting(&models_dir));
        let folder_errors = models::apply_settings(&prefs);
        for engine in &moved {
            eprintln!(
                "moved the {} weights into {}",
                engine.subdir(),
                models::root_from_setting(&models_dir).display()
            );
        }
        let mut app = ViewerApp {
            slots: std::array::from_fn(|_| StudySlot::empty()),
            shown: std::array::from_fn(|s| s == 0 || (s == 1 && initial_b.is_some())),
            view_rows: prefs.view_rows.clone(),
            link_studies: true,
            hovered_slot: 0,
            pane_rects: Vec::new(),
            rec: None,
            rec_format: record::RecFormat::default(),
            rec_max: 600,
            rec_status: None,
            row_rects: [None; MAX_WORKSPACES],
            save_img: None,
            snap: None,
            snap_status: None,
            hand_pan: false,
            loading: None,
            home_content_height: 0.0,
            // What the last run ended with, ready for the start screen's
            // *Restore the last session*; it becomes this run's session as
            // soon as anything is loaded.
            session: std::array::from_fn(|_| Vec::new()),
            last_session: prefs.session.clone(),
            restore_queue: Vec::new(),
            pending_load: None,
            pending_load_files: None,
            pending_switch: None,
            error: None,
            notice: None,
            registration: None,
            reg_job: None,
            reg_fixed: RegPick { slot: 0, series: 0 },
            reg_moving: RegPick { slot: 1, series: 0 },
            fusion_side: FusionSide::Fixed,
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
            reg_init: RegInit::Auto,
            reg_shape: reg_shape::ShapeRegState::default(),
            field_on: false,
            field_warp_only: false,
            pending_phase_field: None,
            field_style: FieldStyle::Arrows,
            field_step_mm: 12.0,
            field_scale: 3.0,
            field_color: true,
            field_job: None,
            gen: Default::default(),
            drr: None,
            drr_job: None,
            r2d: None,
            r2d_job: None,
            r2d_prefs: prefs.reg2d3d.clone(),
            propagate_dialog: None,
            propagate_job: None,
            phases_job: None,
            group_registration: None,
            reg_group: None,
            sim_source: 0,
            sim_target: 1,
            sim_params: SimParams::default(),
            sim_job: None,
            last_sim: None,
            export_open: false,
            export_dir: String::new(),
            export_plan: None,
            export_job: None,
            export_result: None,
            export_warnings: Vec::new(),
            gen_open: false,
            gen_params: GenParams::default(),
            gen_dir: gen_test_data::default_output_dir().display().to_string(),
            gen_job: None,
            gen_result: None,
            gen_load_after: true,
            testdata_open: false,
            testdata_dir: testdata::default_output_dir().display().to_string(),
            testdata_job: None,
            testdata_result: None,
            testdata_load_after: true,
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
            d3_windows: Vec::new(),
            tree_action: None,
            set_action: None,
            obj_remove: None,
            item_action: None,
            rename: None,
            rename_request: None,
            tree_focus: None,
            tree_picks: Vec::new(),
            tick_anchor: None,
            maximized: None,
            reg_apply_invert: false,
            window_center: [40.0; MAX_WORKSPACES],
            window_width: [400.0; MAX_WORKSPACES],
            series_window: [(40.0, 400.0); MAX_WORKSPACES],
            wl_relative: None,
            show_contours: true,
            show_crosshair: true,
            show_labels: true,
            show_isocenters: true,
            models_dir,
            archive_dir,
            archive_more,
            pacs: None,
            pacs_job: None,
            pacs_remote_job: None,
            pacs_server: None,
            pacs_server_job: None,
            models_open: false,
            models_scan: Vec::new(),
            models_scan_at: f64::NEG_INFINITY,
            models_job: None,
            models_result: None,
            ts_licence: prefs.ts_licence.clone().unwrap_or_default(),
            ts_licence_shown: false,
            nnunet_folders: prefs.nnunet_folders.clone(),
            synth_folders: prefs.synth_folders.clone(),
            nnunet_folder_errors: folder_errors,
            synth: synth_win::SynthState::default(),
            synth_job: None,
            synth_eval_job: None,
            synth_slot: 0,
            autoseg_job: None,
            autoseg_slot: 0,
            autoseg_dialog: None,
            autoseg_pending: None,
            autoseg_pending_tg263: false,
            body_job: None,
            body_slot: 0,
            body_dialog: None,

            dvh_open: false,
            dvh_dialog: None,
            dvh_job: None,

            combine_job: None,
            derived_job: None,
            derived_slot: 0,
            derived_queue: Vec::new(),
            derived: Default::default(),
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
            segvol_model: Arc::default(),
            medsam2_job: None,
            medsam2: Default::default(),
            nni_job: None,
            nni: Default::default(),
            seg_tool: SegTool::None,
            brush_radius_mm: 5.0,
            brush_band: contour_edit::EdgeBand::None,
            interp_snap: false,
            brush_auto_cut: false,
            grow_limit: None,
            brush_sensitivity: 0.15,
            brush_3d: true,
            paint_last: None,
            grow: None,
            grow_state: GrowState::default(),
            grow_preview: Vec::new(),
            grow_marked: Vec::new(),
            grow_gen: 0,
            seg_counter: 0,
            draw: None,
            draw_mode: DrawMode::Auto,
            roi_undo: Vec::new(),
            roi_counter: 0,
            poi_counter: 0,
            edit: None,
            edit_flushed: None,
            interp: None,
            contour_clip: None,
            tools: struct_tools::StructTools::default(),
            wire: None,
            wire_learn: true,
            stats_dialog: None,
            dose_mode: DoseMode::Off,
            dose_opacity: 0.45,
            dose_threshold_pct: 15.0,
            iso_levels: default_iso_levels(),
            settings_gen: 0,
            tree_cache: Default::default(),
            struct_gen: [0; MAX_WORKSPACES],
            edit_summary_cache: None,
            wl_preset: None,
            module_registration: prefs.module_registration,
            module_simulation: prefs.module_simulation,
            module_generation: prefs.module_generation,
            module_propagation: prefs.module_propagation,
            module_structures: prefs.module_structures,
            module_auto: prefs.module_auto,
            module_dose: prefs.module_dose,
            module_info: prefs.module_info,
            module_play: prefs.module_play,
            pane_buttons_hidden: prefs.pane_buttons_hidden,
            reg_matrix: matrix_edit::ManualMatrix::default(),
            reg_matrix_pick: 0,
            propagate_finish: crate::propagate::Finish::default(),
            prop_matrix: matrix_edit::ManualMatrix::default(),
            transfer_matrix: matrix_edit::ManualMatrix::default(),
            modules_open: prefs.modules_open.clone(),
            apply_modules_open: false,
            modules_tracked: false,
            dose_est: dose_est::DoseEst::default(),
            play: play::PlayState::default(),
            info: img_info::InfoState::default(),
            dose_est_job: None,
            auto: auto_tools::AutoTools::default(),
            draw_open: false,
            side_open: true,
            right_open: true,
            theme: prefs.theme,
            graphics_backend: prefs.graphics_backend,
            active_backend: cc
                .wgpu_render_state
                .as_ref()
                .map(|r| crate::gfx_wgpu::from_wgpu(r.adapter.get_info().backend)),
            settings_error: None,
            picker: pick::Picker::default(),
            wf_editor: None,
            wf_confirm: None,
            wf_run: None,
            recent_workflows: prefs.recent_workflows.clone(),
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
            archive_more: self.archive_more.clone(),
            module_registration: self.module_registration,
            module_simulation: self.module_simulation,
            module_generation: self.module_generation,
            module_propagation: self.module_propagation,
            module_structures: self.module_structures,
            module_auto: self.module_auto,
            module_dose: self.module_dose,
            module_info: self.module_info,
            module_play: self.module_play,
            pane_buttons_hidden: self.pane_buttons_hidden,
            modules_open: self.modules_open.clone(),
            session: self.session.clone(),
            view_rows: self.view_rows.clone(),
            graphics_backend: self.graphics_backend,
            recent_workflows: self.recent_workflows.clone(),
            ts_licence: {
                let l = self.ts_licence.trim();
                (!l.is_empty()).then(|| l.to_string())
            },
            nnunet_folders: self.nnunet_folders.clone(),
            synth_folders: self.synth_folders.clone(),
            reg2d3d: self.r2d_prefs.clone(),
        }) {
            Ok(()) => self.settings_error = None,
            Err(e) => {
                self.settings_error = Some(format!("⚠ settings not saved: {e:#}"));
            }
        }
    }

    /// Reset zoom, pan, crosshair and slice (all back to the volume center)
    /// of every view of both workspaces.
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
    /// with that slot's three slices. The other workspace is left alone even
    /// when crosshair linking is on - a reset is per-workspace, and "Reset all
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

    /// The modality of the series a workspace shows; empty when none.
    pub(super) fn shown_modality(&self, slot: usize) -> String {
        self.slots
            .get(slot)
            .and_then(|s| s.study.as_ref())
            .and_then(|st| st.series.get(st.active_series))
            .map(|s| s.modality.trim().to_uppercase())
            .unwrap_or_default()
    }

    /// The window (centre, width) a workspace is drawn with.
    pub(super) fn window_of(&self, slot: usize) -> (f32, f32) {
        let s = slot.min(MAX_WORKSPACES - 1);
        (self.window_center[s], self.window_width[s])
    }

    /// Set a workspace's window, and that of every other workspace showing
    /// the same modality: two CTs side by side are read with one window, a
    /// CT beside an MR each with its own.
    pub(super) fn set_window(&mut self, slot: usize, centre: f32, width: f32) {
        let slot = slot.min(MAX_WORKSPACES - 1);
        let m = self.shown_modality(slot);
        for s in 0..MAX_WORKSPACES {
            if s == slot || (!m.is_empty() && self.shown_modality(s) == m) {
                self.window_center[s] = centre;
                self.window_width[s] = width.max(1.0);
            }
        }
    }

    /// The window a workspace takes when a series of `modality` comes up in
    /// it: another workspace's showing that modality, so a comparison stays
    /// comparable, otherwise `default` (the series' own).
    pub(super) fn adopt_window(&mut self, slot: usize, modality: &str, default: (f32, f32)) {
        let slot = slot.min(MAX_WORKSPACES - 1);
        let m = modality.trim().to_uppercase();
        let other = (0..MAX_WORKSPACES)
            .find(|s| *s != slot && !m.is_empty() && self.shown_modality(*s) == m);
        let (c, w) = match other {
            Some(o) => self.window_of(o),
            None => default,
        };
        self.window_center[slot] = c;
        self.window_width[slot] = w.max(1.0);
    }

    /// The windows of [`RELATIVE_PRESETS`] for a workspace's volume: the
    /// series' own first, then the percentile pairs, worked out once per
    /// volume.
    pub(super) fn relative_windows(&mut self, slot: usize) -> Vec<(f32, f32)> {
        let slot = slot.min(MAX_WORKSPACES - 1);
        let own = self.series_window[slot];
        let Some(vol) = self.slots[slot].study.as_ref().map(|s| s.volume.clone()) else {
            return vec![own; RELATIVE_PRESETS.len()];
        };
        let cached = self.wl_relative.as_ref().and_then(|(w, v)| {
            w.upgrade()
                .filter(|x| Arc::ptr_eq(x, &vol))
                .map(|_| v.clone())
        });
        let mut out = match cached {
            Some(v) => v,
            None => {
                let pairs: Vec<(f32, f32)> = RELATIVE_PRESETS
                    .iter()
                    .map(|(_, lo, hi)| (*lo, *hi))
                    .collect();
                let v = loader::percentile_windows(&vol.data, &pairs);
                self.wl_relative = Some((Arc::downgrade(&vol), v.clone()));
                v
            }
        };
        if let Some(first) = out.first_mut() {
            *first = own;
        }
        out
    }

    /// The workspace the toolbar's window acts on: the one last pointed
    /// at, or the first that shows something.
    pub(super) fn wl_slot(&self) -> usize {
        let here = self.hovered_slot.min(MAX_WORKSPACES - 1);
        if self.slots[here].has_volume() {
            return here;
        }
        (0..MAX_WORKSPACES)
            .find(|s| self.slots[*s].has_volume())
            .unwrap_or(here)
    }

    /// This station's PACS: the storage folders the settings name.
    pub(super) fn station_pacs(&self) -> crate::archive::Archive {
        crate::archive::Archive::with_roots(crate::archive::station_roots(
            &self.archive_dir,
            &self.archive_more,
        ))
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
            self.last_session = std::array::from_fn(|_| Vec::new());
            self.session = std::array::from_fn(|_| Vec::new());
            self.persist_settings();
            return;
        }
        for slot in 1..MAX_WORKSPACES {
            if !self.last_session[slot].is_empty() {
                self.shown[slot] = true;
            }
        }
        self.apply_modules_open = true;
        for (slot, paths) in self.last_session.clone().iter().enumerate() {
            for path in paths {
                self.restore_queue.push((slot, path.clone()));
            }
        }
    }

    /// Empty a workspace past the first and take it off the screen.
    ///
    /// The letters do not shift up: closing B leaves C where it is, because
    /// everything on screen - a registration, a propagation, a window
    /// title - names workspaces by letter, and renaming them under the user
    /// would be worse than a gap.
    pub(super) fn close_workspace(&mut self, slot: usize) {
        if slot == 0 || slot >= MAX_WORKSPACES {
            return;
        }
        self.slots[slot] = StudySlot::empty();
        self.forget_sources(slot);
        self.shown[slot] = false;
        if self.hovered_slot == slot {
            self.hovered_slot = 0;
        }
        self.d3_windows.retain(|w| w.slot != slot);
        if self.maximized.map(|(s, _)| s == slot).unwrap_or(false) {
            self.maximized = None;
        }
        // A registration is between two images of two workspaces; one of
        // them has just gone.
        if self
            .registration
            .as_ref()
            .is_some_and(|r| r.fixed_slot == slot || r.moving_slot == slot)
        {
            self.clear_registration();
        }
    }

    /// The workspaces on screen, in order.
    pub(super) fn open_slots(&self) -> Vec<usize> {
        (0..MAX_WORKSPACES).filter(|s| self.is_open(*s)).collect()
    }

    /// Is this workspace on screen?
    ///
    /// A is always. The others are the ones something has been loaded into
    /// (loading switches this on) or that were opened to load into, minus
    /// any the user has put aside under *View ▸ Workspaces*.
    pub(super) fn is_open(&self, slot: usize) -> bool {
        slot == 0 || (slot < MAX_WORKSPACES && self.shown[slot])
    }

    /// How many workspaces are on screen. One is the ordinary case; more
    /// than one is what used to be called comparison mode.
    pub(super) fn open_count(&self) -> usize {
        self.open_slots().len()
    }

    /// More than one workspace on screen: the toolbar's Sync, the status
    /// bar's side-by-side readout, the letters in window titles.
    pub(super) fn comparing(&self) -> bool {
        self.open_count() > 1
    }

    /// The workspace a two-sided tool should pair `slot` with by default:
    /// the next one round the ring that is open, so the answer with two
    /// workspaces is the same one the program always gave.
    pub(super) fn other_open(&self, slot: usize) -> Option<usize> {
        (1..MAX_WORKSPACES)
            .map(|d| (slot + d) % MAX_WORKSPACES)
            .find(|s| self.is_open(*s))
    }

    /// Where a copy, a move or a generated study may go from `from`: every
    /// other open workspace, and one new letter if there is room.
    ///
    /// This is the rule the whole program follows for offering workspaces:
    /// the ones that exist, plus one more - never four letters when only A
    /// is open, and never a fifth.
    pub(super) fn copy_targets(&self, from: usize) -> Vec<usize> {
        workspace_pick::transfer_targets(&std::array::from_fn(|s| self.is_open(s)), from)
    }

    /// Where data can be loaded: every open workspace, plus one new letter
    /// while there is room. The same rule as [`Self::copy_targets`], with
    /// nothing to exclude.
    pub(super) fn open_plus_new(&self) -> Vec<usize> {
        self.copy_targets(usize::MAX)
    }

    /// Put a workspace on screen, loaded or not.
    pub(super) fn show_workspace(&mut self, slot: usize) {
        if slot < MAX_WORKSPACES {
            self.shown[slot] = true;
        }
    }
}

// eframe::App

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // A sagittal or coronal stroke let go where the view did not see the
        // release (over a corner button, outside the view) still writes what
        // it left waiting, before anything landing below can move `gen`.
        if self.edit.as_ref().is_some_and(|e| e.pending) && !ctx.input(|i| i.pointer.primary_down())
        {
            self.end_contour_stroke();
        }
        self.poll_all_jobs(&ctx);

        // Global segmentation shortcuts (skipped while a text field is
        // focused): Ctrl+Z undo, Esc cancels a region-grow drag, [ ] resize
        // the brush.
        if !ctx.egui_wants_keyboard_input() {
            let (undo, esc, smaller, bigger, toggle_side, toggle_right, rename, remove) = ctx
                .input(|i| {
                    (
                        i.modifiers.command && i.key_pressed(egui::Key::Z),
                        i.key_pressed(egui::Key::Escape),
                        i.key_pressed(egui::Key::OpenBracket),
                        i.key_pressed(egui::Key::CloseBracket),
                        i.key_pressed(egui::Key::F9),
                        i.key_pressed(egui::Key::F10),
                        i.key_pressed(egui::Key::F2),
                        i.key_pressed(egui::Key::Delete),
                    )
                });
            // ↑ / ↓ walk the data tree at the level of the focused row -
            // CT to CT within a study, phase to phase within a 4D group -
            // and show what they land on. Repeat is honoured, so the key
            // can be held down to run through the phases of a 4DCT.
            let (up, down) = ctx.input(|i| {
                (
                    i.key_pressed(egui::Key::ArrowUp),
                    i.key_pressed(egui::Key::ArrowDown),
                )
            });
            if up != down {
                self.tree_key_step(if down { 1 } else { -1 });
            }
            if toggle_side {
                self.side_open = !self.side_open;
            }
            if toggle_right {
                self.right_open = !self.right_open;
            }
            // F2 and Delete act on the data-tree row last clicked, the way
            // they do in a file manager.
            if rename {
                self.tree_key_rename();
            }
            if remove {
                self.tree_key_remove();
            }
            if undo {
                let slot = self.preferred_volume_slot();
                // A contour tool undoes contours, a voxel tool voxels: the
                // key follows the tool in hand rather than a global history.
                if self.seg_tool.draws_contours() {
                    self.undo_roi_edit(slot);
                } else {
                    self.undo_active_seg(slot);
                }
            }
            if esc && self.grow.is_some() {
                self.cancel_grow();
            }
            if esc && self.draw.is_some() {
                self.draw = None;
                self.livewire_reset();
            }
            // Enter closes the polygon under construction, for anyone who
            // would rather not reach for the right button.
            if self.draw.is_some() && ctx.input(|i| i.key_pressed(egui::Key::Enter)) {
                self.commit_draw();
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
        // A PACS call started this frame - by a click, or by the result of
        // the one before it (a pull is followed by a refresh) - has not been
        // polled yet, so nothing else asks for the next frame.
        if self.pacs_remote_job.is_some() || self.pacs_server_job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        } else if self.pacs_server.is_some() || self.remote_tasks_running() {
            // The server window's status and a running task's progress are
            // polled; frames have to come for that to happen.
            ctx.request_repaint_after(std::time::Duration::from_secs(1));
        }
    }
}
