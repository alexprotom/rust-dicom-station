//! Box-driven segmentation: drawing the prompt, and the loop around it.
//!
//! The engine below ([`crate::medsam2`]) segments one slice from a prompt and
//! follows it through the stack. This module is the part the user touches, and
//! it is deliberately shaped like the MedSAM2 extension for 3D Slicer, which
//! is the interface people who use this model already know:
//!
//! 1. scroll to a slice where the structure is clear and **drag a box** around
//!    it, directly in the image — the box stays, with handles, and can be
//!    resized or moved;
//! 2. **preview that one slice**, which costs a single encoder pass and is
//!    re-run for free while the box is adjusted;
//! 3. add **include / exclude clicks** if the preview is not quite right;
//! 4. set the **slice range** to propagate through, and run it.
//!
//! Nothing is anchored to the crosshair, and the crosshair does not move while
//! the box tool has the left button. The expensive things — the weights and
//! the prepared stack — are built once and kept in [`Medsam2State`], and the
//! engine keeps the *prompted slice's* encoder output, so steps 2 and 3 loop
//! at the cost of the prompt path alone.
//!
//! One conversion is central: the box lives in the drawing view's own pixel
//! coordinates, and [`Medsam2State::engine_prompt`] is the only place it
//! becomes network coordinates.

use crate::medsam2::engine::{Engine, EnginePrompt, PixelPrompt};
use crate::medsam2::infer::Config;
use crate::medsam2::preprocess::{self, Prepared, Window};
use crate::medsam2::weights::{self, Variant};
use crate::models::Engine as ModelsEngine;
use crate::nn::device::DevicePref;

use super::*;

/// How close to a corner the pointer has to be, in screen pixels, to grab it.
pub(super) const HANDLE_GRAB: f32 = 9.0;

// ---------------------------------------------------------------- the box --

/// What a left-drag in the drawing view does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum BoxTool {
    /// Draw, resize or move the box.
    Draw,
    /// Add a click that says "this is the structure".
    Include,
    /// Add a click that says "this is not".
    Exclude,
}

/// What the current drag is doing to the box.
#[derive(Clone, Copy, Debug)]
enum Grab {
    /// Drawing a new box: `min` is fixed, the pointer is `max`.
    Fresh,
    /// Dragging one corner; the opposite one is fixed.
    Corner,
    /// Moving the whole box: where it was grabbed, and where it started.
    Move {
        at: [f32; 2],
        min: [f32; 2],
        max: [f32; 2],
    },
}

/// The prompt as drawn: a rectangle and any refinement clicks, in the drawing
/// plane's pixel coordinates, on one slice.
pub(super) struct BoxPrompt {
    pub plane: ViewPlane,
    pub slice: usize,
    /// The two dragged corners, in drawing order — not necessarily sorted.
    a: [f32; 2],
    b: [f32; 2],
    /// Refinement clicks and whether each says "include".
    pub points: Vec<([f32; 2], bool)>,
    grab: Option<Grab>,
}

impl BoxPrompt {
    fn new(plane: ViewPlane, slice: usize, at: [f32; 2]) -> BoxPrompt {
        BoxPrompt {
            plane,
            slice,
            a: at,
            b: at,
            points: Vec::new(),
            grab: Some(Grab::Fresh),
        }
    }

    /// The rectangle, sorted.
    pub fn rect(&self) -> ([f32; 2], [f32; 2]) {
        (
            [self.a[0].min(self.b[0]), self.a[1].min(self.b[1])],
            [self.a[0].max(self.b[0]), self.a[1].max(self.b[1])],
        )
    }

    /// A box too small to mean anything — a stray click rather than a drag.
    pub fn is_degenerate(&self) -> bool {
        let (lo, hi) = self.rect();
        hi[0] - lo[0] < 2.0 || hi[1] - lo[1] < 2.0
    }

    /// The four corners, in the order `handle` indexes them.
    pub fn corners(&self) -> [[f32; 2]; 4] {
        let (lo, hi) = self.rect();
        [
            [lo[0], lo[1]],
            [hi[0], lo[1]],
            [hi[0], hi[1]],
            [lo[0], hi[1]],
        ]
    }

    /// Which corner is within `tol` pixels of `p`, if any.
    fn handle(&self, p: [f32; 2], tol: f32) -> Option<usize> {
        self.corners()
            .iter()
            .enumerate()
            .map(|(i, c)| (i, (c[0] - p[0]).hypot(c[1] - p[1])))
            .filter(|(_, d)| *d <= tol)
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(i, _)| i)
    }

    fn contains(&self, p: [f32; 2]) -> bool {
        let (lo, hi) = self.rect();
        p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1]
    }

    /// Begin a drag at `p`: grab a corner, move the box, or start a new one.
    fn press(&mut self, p: [f32; 2], tol: f32) {
        self.grab = if let Some(i) = self.handle(p, tol) {
            // Dragging a corner keeps the opposite one fixed.
            let c = self.corners();
            self.a = c[(i + 2) % 4];
            self.b = p;
            Some(Grab::Corner)
        } else if self.contains(p) {
            let (lo, hi) = self.rect();
            Some(Grab::Move {
                at: p,
                min: lo,
                max: hi,
            })
        } else {
            self.a = p;
            self.b = p;
            Some(Grab::Fresh)
        };
    }

    fn drag(&mut self, p: [f32; 2]) {
        match self.grab {
            Some(Grab::Fresh) | Some(Grab::Corner) => self.b = p,
            Some(Grab::Move { at, min, max }) => {
                let d = [p[0] - at[0], p[1] - at[1]];
                self.a = [min[0] + d[0], min[1] + d[1]];
                self.b = [max[0] + d[0], max[1] + d[1]];
            }
            None => {}
        }
    }

    fn release(&mut self) {
        self.grab = None;
    }

    /// Clamp to the slice's extent, so a box dragged past the edge still means
    /// something.
    fn clamp(&mut self, size: [usize; 2]) {
        let (w, h) = (size[0] as f32 - 1.0, size[1] as f32 - 1.0);
        for p in [&mut self.a, &mut self.b] {
            p[0] = p[0].clamp(0.0, w);
            p[1] = p[1].clamp(0.0, h);
        }
        self.points
            .retain(|(p, _)| p[0] >= 0.0 && p[0] <= w && p[1] >= 0.0 && p[1] <= h);
    }
}

// ------------------------------------------------------------- background --

/// Identifies the prepared stack: rebuild it when any of this changes.
#[derive(Clone, PartialEq, Debug)]
pub struct PrepKey {
    pub slot: usize,
    pub dims: [usize; 3],
    pub uid: String,
    pub window: [f32; 2],
}

/// What the panel asked for.
pub enum Request {
    /// Segment the prompted slice only.
    Preview,
    /// Propagate through the range.
    Propagate,
}

/// One finished run.
pub struct Medsam2Result {
    /// On the volume's own grid, one byte per voxel.
    pub mask: Vec<u8>,
    pub voxels: u64,
    pub slices_visited: usize,
    pub extent: Option<(usize, usize)>,
    pub elapsed_secs: f64,
}

/// What came back, together with everything worth keeping for the next run.
pub struct Medsam2Done {
    pub engine: Arc<Engine>,
    pub prepared: Arc<Prepared>,
    pub volume: Arc<Volume>,
    pub key: PrepKey,
    pub device: String,
    pub request: Request,
    pub result: Medsam2Result,
}

/// Everything a run needs, snapshotted from the panel when it starts.
struct Medsam2Request {
    /// The loaded network, when a previous run left one.
    engine: Option<Arc<Engine>>,
    /// The prepared stack, when it is still the one this study and window need.
    cached: Option<(Arc<Prepared>, Arc<Volume>)>,
    /// The voxels to prepare a stack from, when `cached` is `None`.
    fresh: Option<Arc<Volume>>,
    key: PrepKey,
    variant: Variant,
    window: Window,
    device: DevicePref,
    models_dir: PathBuf,
    request: Request,
    slice: usize,
    prompt: EnginePrompt,
    cfg: Config,
}

/// The background half: build whatever is missing, then run the request.
fn run_job(req: Medsam2Request, progress: &Progress) -> anyhow::Result<Medsam2Done> {
    let prepared_is_new = req.cached.is_none();
    let (prepared, volume) = match req.cached {
        Some(pair) => pair,
        None => {
            progress.set("Preparing the study");
            let volume = req
                .fresh
                .ok_or_else(|| anyhow::anyhow!("no volume to prepare"))?;
            let prepared = Arc::new(Prepared::prepare(&volume, req.window));
            (prepared, volume)
        }
    };
    let engine = match req.engine {
        Some(e) => {
            if prepared_is_new {
                // The engine keeps the last encoded slice; a stack built with
                // a different window would make that a lie.
                e.clear_cache();
            }
            e
        }
        None => {
            progress.set("Loading the weights");
            let params = weights::load(req.variant, &req.models_dir, progress)?;
            progress.set("Choosing the compute device");
            Arc::new(Engine::load(&params, req.device)?)
        }
    };
    progress.set_device(engine.device().to_string());

    let started = std::time::Instant::now();
    let result = match req.request {
        Request::Preview => {
            progress.set("Segmenting this slice");
            let slice_mask = engine.preview(&prepared, req.slice, &req.prompt, &req.cfg)?;
            let voxels = slice_mask.iter().filter(|v| **v != 0).count() as u64;
            // One slice, on the volume's grid, so the viewer can draw it with
            // everything else.
            let mut masks: Vec<Vec<u8>> = (0..prepared.dims[0]).map(|_| Vec::new()).collect();
            masks[req.slice] = slice_mask;
            Medsam2Result {
                mask: prepared.mask_to_volume_grid(&masks, &volume),
                voxels,
                slices_visited: 1,
                extent: Some((req.slice, req.slice)),
                elapsed_secs: started.elapsed().as_secs_f64(),
            }
        }
        Request::Propagate => {
            let (mask, seg) = engine.propagate_to_volume(
                &prepared,
                &volume,
                req.slice,
                &req.prompt,
                &req.cfg,
                progress,
            )?;
            Medsam2Result {
                mask,
                voxels: seg.voxels,
                slices_visited: seg.slices_visited,
                extent: seg.extent(),
                elapsed_secs: started.elapsed().as_secs_f64(),
            }
        }
    };
    Ok(Medsam2Done {
        device: engine.device().to_string(),
        engine,
        prepared,
        volume,
        key: req.key,
        request: req.request,
        result,
    })
}

// ------------------------------------------------------------- the state --

/// Where the intensity window comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum WindowSource {
    /// What the viewport is showing — what you see is what the model sees.
    Viewport,
    /// One of the windows the MedSAM2 paper trained with.
    Preset(usize),
}

/// Everything the panel owns.
pub(super) struct Medsam2State {
    pub open: bool,
    pub slot: usize,
    pub tool: BoxTool,
    pub prompt: Option<BoxPrompt>,
    /// The loaded network, kept across runs and studies.
    pub engine: Option<Arc<Engine>>,
    /// The prepared stack, kept until the study or the window changes.
    pub prep: Option<(PrepKey, Arc<Prepared>, Arc<Volume>)>,
    /// Index of the segmentation the preview and the result are written to.
    pub target_seg: Option<usize>,
    pub auto_preview: bool,
    /// Slice range to propagate through, in the drawing view's slice numbers.
    pub range: Option<(usize, usize)>,
    pub variant: Variant,
    pub window: WindowSource,
    pub cfg: Config,
    pub device: DevicePref,
    pub name: String,
    pub status: Option<String>,
    /// Set when the box changed and an automatic preview is due.
    pub dirty: bool,
    /// True once the user has chosen a range themselves, after which it stops
    /// following the box.
    pub range_pinned: bool,
    /// Add each run to what is already there, instead of replacing it — how a
    /// slice that drifted gets corrected: draw a fresh box on it and run
    /// again.
    pub merge: bool,
    /// The mask as of the last committed propagation, which previews are
    /// shown on top of.
    pub base_mask: Option<Vec<u8>>,
}

impl Default for Medsam2State {
    fn default() -> Medsam2State {
        Medsam2State {
            open: false,
            slot: 0,
            tool: BoxTool::Draw,
            prompt: None,
            engine: None,
            prep: None,
            target_seg: None,
            auto_preview: true,
            range: None,
            variant: Variant::default(),
            window: WindowSource::Viewport,
            cfg: Config {
                max_slices: None,
                ..Config::default()
            },
            device: DevicePref::Auto,
            name: "Propagated".to_string(),
            status: None,
            dirty: false,
            range_pinned: false,
            merge: true,
            base_mask: None,
        }
    }
}

impl Medsam2State {
    /// The prepared stack, if it is the one this study and window need.
    fn cached(&self, key: &PrepKey) -> Option<(Arc<Prepared>, Arc<Volume>)> {
        self.prep
            .as_ref()
            .filter(|(k, _, _)| k == key)
            .map(|(_, p, v)| (p.clone(), v.clone()))
    }

    /// Turn the drawn box into the engine's prompt, and say which prepared
    /// slice it sits on.
    ///
    /// This is the whole coordinate story: view pixels -> volume voxel ->
    /// prepared (slice, row, column). Both corners land on the same prepared
    /// slice because the box is drawn in the plane the stack is sliced along.
    fn engine_prompt(&self, vol: &Volume) -> Option<(usize, EnginePrompt)> {
        let bx = self.prompt.as_ref()?;
        if bx.is_degenerate() {
            return None;
        }
        let to_prepared = |p: [f32; 2]| -> [usize; 3] {
            let v = vol.plane_pixel_to_voxel(bx.plane, bx.slice, f64::from(p[0]), f64::from(p[1]));
            let idx = [
                (v[0].round().max(0.0) as usize).min(vol.dims[0] - 1),
                (v[1].round().max(0.0) as usize).min(vol.dims[1] - 1),
                (v[2].round().max(0.0) as usize).min(vol.dims[2] - 1),
            ];
            preprocess::volume_index_to_prepared(vol, idx)
        };
        let (lo, hi) = bx.rect();
        let a = to_prepared(lo);
        let b = to_prepared(hi);
        let mut points =
            PixelPrompt::box_corners(a[1] as f32, a[2] as f32, b[1] as f32, b[2] as f32);
        for (p, include) in &bx.points {
            let q = to_prepared(*p);
            points.push(if *include {
                PixelPrompt::positive(q[1] as f32, q[2] as f32)
            } else {
                PixelPrompt::negative(q[1] as f32, q[2] as f32)
            });
        }
        Some((a[0], EnginePrompt::Points(points)))
    }

    /// The propagation range, converted from view slice numbers to the
    /// prepared stack's own indices.
    fn prepared_range(&self, vol: &Volume, plane: ViewPlane) -> Option<(usize, usize)> {
        let (a, b) = self.range?;
        let to_index = |s: usize| -> usize {
            let v = vol.plane_pixel_to_voxel(plane, s, 0.0, 0.0);
            preprocess::volume_index_to_prepared(
                vol,
                [
                    (v[0].max(0.0) as usize).min(vol.dims[0] - 1),
                    (v[1].max(0.0) as usize).min(vol.dims[1] - 1),
                    (v[2].max(0.0) as usize).min(vol.dims[2] - 1),
                ],
            )[0]
        };
        let (x, y) = (to_index(a), to_index(b));
        Some((x.min(y), x.max(y)))
    }
}

/// Which of the three views the stack is sliced along — the one a prompt has
/// to be drawn in.
pub(super) fn drawing_plane(vol: &Volume) -> ViewPlane {
    let (perm, _) = preprocess::axial_axes(vol);
    match perm[0] {
        0 => ViewPlane::Sagittal,
        1 => ViewPlane::Coronal,
        _ => ViewPlane::Axial,
    }
}

pub(super) fn plane_name(plane: ViewPlane) -> &'static str {
    match plane {
        ViewPlane::Axial => "axial",
        ViewPlane::Sagittal => "sagittal",
        ViewPlane::Coronal => "coronal",
    }
}

// -------------------------------------------------------- the application --

impl ViewerApp {
    /// Tools ▶ slice propagation: open the tool window for `slot`.
    pub(super) fn open_medsam2_panel(&mut self, slot: usize) {
        if !self.slots[slot].has_volume() {
            return;
        }
        if self.medsam2.slot != slot {
            // Not while a run on the other dataset is still in flight: the
            // box, the target segmentation and the range belong to it.
            if self.medsam2_job.is_some() {
                return;
            }
            self.medsam2.prompt = None;
            self.medsam2.target_seg = None;
            self.medsam2.base_mask = None;
            self.medsam2.range = None;
            self.medsam2.range_pinned = false;
        }
        self.medsam2.slot = slot;
        self.medsam2.open = true;
    }

    /// Is this the viewport the prompt belongs to? The box is drawn on screen
    /// whenever the panel is open, including while a run is in flight.
    pub(super) fn medsam2_showing_in(&self, slot: usize, plane: ViewPlane) -> bool {
        self.medsam2.open
            && self.medsam2.slot == slot
            && self.slots[slot].has_volume()
            && self.slots[slot]
                .study
                .as_ref()
                .is_some_and(|s| drawing_plane(&s.volume) == plane)
    }

    /// ...and can it be edited? Not while the network is busy with it.
    pub(super) fn medsam2_drawing_in(&self, slot: usize, plane: ViewPlane) -> bool {
        self.medsam2_showing_in(slot, plane) && self.medsam2_job.is_none()
    }

    /// The pointer went down in the drawing view.
    pub(super) fn medsam2_press(&mut self, plane: ViewPlane, slice: usize, p: [f32; 2], tol: f32) {
        match self.medsam2.tool {
            BoxTool::Draw => {
                let fresh = match &self.medsam2.prompt {
                    // A box on another slice is replaced rather than moved:
                    // the prompt belongs to the slice it was drawn on.
                    Some(b) => b.slice != slice || b.plane != plane,
                    None => true,
                };
                if fresh {
                    self.medsam2.prompt = Some(BoxPrompt::new(plane, slice, p));
                } else if let Some(b) = self.medsam2.prompt.as_mut() {
                    b.press(p, tol);
                }
            }
            BoxTool::Include | BoxTool::Exclude => {
                let include = self.medsam2.tool == BoxTool::Include;
                if let Some(b) = self.medsam2.prompt.as_mut() {
                    if b.slice == slice && b.plane == plane {
                        b.points.push((p, include));
                        self.medsam2.dirty = true;
                    }
                }
            }
        }
    }

    pub(super) fn medsam2_drag(&mut self, p: [f32; 2]) {
        if self.medsam2.tool == BoxTool::Draw {
            if let Some(b) = self.medsam2.prompt.as_mut() {
                b.drag(p);
            }
        }
    }

    pub(super) fn medsam2_release(&mut self, size: [usize; 2]) {
        if self.medsam2.tool != BoxTool::Draw {
            return;
        }
        if let Some(b) = self.medsam2.prompt.as_mut() {
            b.release();
            b.clamp(size);
            if b.is_degenerate() {
                // A click with no drag: not a box.
                self.medsam2.prompt = None;
            } else {
                let slice = b.slice;
                self.medsam2.dirty = true;
                if !self.medsam2.range_pinned {
                    // Until the range is set by hand it follows the box, which
                    // is nearly always what a first run wants.
                    self.medsam2.range = Some((slice.saturating_sub(32), slice + 32));
                }
            }
        }
    }

    /// Start a run, building whatever is missing on the way.
    pub(super) fn start_medsam2(&mut self, request: Request) {
        if self.medsam2_job.is_some() {
            return;
        }
        let slot = self.medsam2.slot;
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let window = match self.medsam2.window {
            WindowSource::Viewport => {
                Window::from_width_level(self.window_width, self.window_center)
            }
            WindowSource::Preset(i) => {
                let (_, w, l) = Window::PRESETS[i];
                Window::from_width_level(w, l)
            }
        };
        let key = PrepKey {
            slot,
            dims: study.volume.dims,
            uid: study.volume.frame_of_reference_uid.clone(),
            window: [window.lower, window.upper],
        };
        let cached = self.medsam2.cached(&key);
        let plane = drawing_plane(&study.volume);
        // The box maps to slice and pixel numbers without a prepared stack,
        // so nothing expensive happens on this thread.
        let Some((slice, prompt)) = self.medsam2.engine_prompt(&study.volume) else {
            self.error = Some(
                format!("Draw a box in the {} view first - drag around the structure on a slice where it is clear.", plane_name(plane)),
            );
            return;
        };
        let mut cfg = self.medsam2.cfg;
        cfg.range = self.medsam2.prepared_range(&study.volume, plane);
        if matches!(request, Request::Preview) {
            cfg.largest_component = false;
        }
        // Preparing the stack is the worker's job; it only needs the voxels.
        let fresh = if cached.is_some() {
            None
        } else {
            Some(study.volume.clone())
        };
        let progress = Arc::new(Progress::default());
        progress.set(match request {
            Request::Preview => "Segmenting this slice",
            Request::Propagate => "Propagating",
        });
        let req = Medsam2Request {
            engine: self.medsam2.engine.clone(),
            cached,
            fresh,
            key,
            variant: self.medsam2.variant,
            window,
            device: self.medsam2.device,
            models_dir: self.engine_models_dir(ModelsEngine::MedSam2),
            request,
            slice,
            prompt,
            cfg,
        };
        self.medsam2.dirty = false;
        self.medsam2_job = Some(Job::spawn(progress, move |p| (slot, run_job(req, p))));
    }

    /// A run finished.
    pub(super) fn on_medsam2_done(&mut self, slot: usize, done: Medsam2Done) {
        let valid = self.slot_still_shows(slot, done.key.dims, &done.key.uid);
        // Keep the expensive parts whatever happened to the result.
        self.medsam2.engine = Some(done.engine);
        self.medsam2.prep = Some((done.key.clone(), done.prepared, done.volume));
        if !valid {
            self.error = Some(stale_result(&SLICE_PROP));
            return;
        }
        let preview = matches!(done.request, Request::Preview);
        let mut r = done.result;
        // A correction run adds to what is there; a preview is shown on top of
        // it without ever being committed, so a box that turns out wrong is
        // replaced by the next preview rather than accumulating.
        if self.medsam2.merge {
            if let Some(base) = &self.medsam2.base_mask {
                if base.len() == r.mask.len() {
                    for (m, b) in r.mask.iter_mut().zip(base) {
                        *m |= *b;
                    }
                    r.voxels = r.mask.iter().filter(|v| **v != 0).count() as u64;
                }
            }
        }
        let r = &r;
        if r.voxels == 0 {
            self.medsam2.status = Some(
                "Nothing found inside the box. Try a tighter box, an include click, \
                 or a different window."
                    .to_string(),
            );
            return;
        }
        let name = if preview {
            format!("{} (this slice)", self.medsam2.name.trim())
        } else {
            self.medsam2.name.trim().to_string()
        };
        let dims = done.key.dims;
        let index = match self.medsam2.target_seg {
            // Replace the previous preview or result in place, so the list
            // does not fill up with attempts.
            Some(i) if i < self.slots[slot].segs().len() => {
                let color = self.slots[slot].segs()[i].color;
                let made = Segmentation::from_label_map(name, color, dims, &r.mask, 1);
                if let Some(segs) = self.slots[slot].segs_mut() {
                    segs[i] = made;
                }
                i
            }
            _ => self.add_segmentation(slot, name, dims, &r.mask),
        };
        self.medsam2.target_seg = Some(index);
        self.slots[slot].active_seg = index;
        if !preview {
            // What the next preview will be shown on top of.
            self.medsam2.base_mask = Some(r.mask.clone());
        }
        let spacing = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.volume.spacing)
            .unwrap_or([1.0; 3]);
        let cm3 = r.voxels as f64 * spacing[0] * spacing[1] * spacing[2] / 1000.0;
        self.medsam2.status = Some(if preview {
            format!(
                "This slice: {} pixels in {:.1} s on {}",
                r.voxels, r.elapsed_secs, done.device
            )
        } else {
            let span = match r.extent {
                Some((a, b)) => format!("{} slice(s)", b - a + 1),
                None => "nothing".to_string(),
            };
            format!(
                "✔ {}: {} voxels ({cm3:.1} cm³) over {span} in {:.1} s on {} - {} slice(s) tracked",
                self.medsam2.name.trim(),
                r.voxels,
                r.elapsed_secs,
                done.device,
                r.slices_visited
            )
        });
    }

    /// The tool window; while a run is in flight its buttons become the
    /// progress row.
    pub(super) fn medsam2_window(&mut self, ctx: &egui::Context) {
        if !self.medsam2.open {
            return;
        }
        let slot = self.medsam2.slot;
        let Some(study) = self.slots[slot].study.as_ref() else {
            self.medsam2.open = false;
            return;
        };
        let plane = drawing_plane(&study.volume);
        let n_slices = study.volume.plane_slice_count(plane);
        let current = self.slots[slot].views[super::plane_index(plane)].slice;
        let running = self.medsam2_job.is_some();
        let models_dir = self.engine_models_dir(ModelsEngine::MedSam2);

        let mut request: Option<Request> = None;
        let mut open = true;
        let mut close = false;
        let mut cancel = false;
        let mut clear = false;
        let mut browse = false;

        detach::tool_window(
            ctx,
            "medsam2",
            SLICE_PROP.title(slot),
            &mut open,
            detach::WinOpts::width(380.0),
            |ui| {
                ui.label(format!(
                    "Follows a structure boxed on one slice through the stack with MedSAM2, \
                     re-implemented natively in Rust. Drag a box around it in the {} view, on a \
                     slice where it is clear; the box stays - drag its corners to resize, its \
                     middle to move it.",
                    plane_name(plane)
                ));
                ui.separator();

                // ---- the prompt ------------------------------------------
                ui.horizontal(|ui| {
                    ui.label("Draw:");
                    ui.selectable_value(&mut self.medsam2.tool, BoxTool::Draw, "⬚ Box")
                        .on_hover_text("Drag a new box, or move and resize the one that is there");
                    ui.selectable_value(&mut self.medsam2.tool, BoxTool::Include, "➕ Include")
                        .on_hover_text("Click a spot the box got wrong - this is the structure");
                    ui.selectable_value(&mut self.medsam2.tool, BoxTool::Exclude, "➖ Exclude")
                        .on_hover_text("Click a spot that must stay out");
                    if ui.button("Clear").clicked() {
                        clear = true;
                    }
                });
                match &self.medsam2.prompt {
                    Some(b) => {
                        let (lo, hi) = b.rect();
                        ui.weak(format!(
                            "Box on slice {} - {:.0} x {:.0} px, {} click(s)",
                            b.slice + 1,
                            hi[0] - lo[0],
                            hi[1] - lo[1],
                            b.points.len()
                        ));
                    }
                    None => {
                        ui.weak("No box yet.");
                    }
                }

                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !running && self.medsam2.prompt.is_some(),
                            egui::Button::new("👁 Preview this slice"),
                        )
                        .on_hover_text(
                            "Segment only the slice the box is on. The slice is encoded \
                             once and kept, so adjusting the box after that is quick.",
                        )
                        .clicked()
                    {
                        request = Some(Request::Preview);
                    }
                    ui.checkbox(&mut self.medsam2.auto_preview, "automatically")
                        .on_hover_text("Preview again whenever the box or the clicks change");
                });

                // ---- the range -------------------------------------------
                ui.separator();
                ui.label("Propagate through:");
                let last = n_slices.saturating_sub(1);
                let mut range = self.medsam2.range.unwrap_or((0, last));
                let before = range;
                // Slice numbers are shown 1-based; the values stay 0-based.
                fn one_based<'a>(
                    v: &'a mut usize,
                    last: usize,
                    prefix: &str,
                ) -> egui::DragValue<'a> {
                    egui::DragValue::new(v)
                        .range(0..=last)
                        .custom_formatter(|v, _| format!("{}", v as usize + 1))
                        .custom_parser(|s| s.parse::<f64>().ok().map(|v| v - 1.0))
                        .prefix(prefix)
                }
                ui.horizontal(|ui| {
                    ui.add(one_based(&mut range.0, last, "from "));
                    if ui.button("⇤ this slice").clicked() {
                        range.0 = current;
                    }
                    ui.add(one_based(&mut range.1, last, "to "));
                    if ui.button("⇥ this slice").clicked() {
                        range.1 = current;
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Whole study").clicked() {
                        range = (0, last);
                    }
                    if let Some(b) = &self.medsam2.prompt {
                        if ui.button("± 32 slices").clicked() {
                            range = (b.slice.saturating_sub(32), (b.slice + 32).min(last));
                        }
                    }
                    ui.weak(format!("{} of {} slices", range.1 - range.0 + 1, n_slices));
                });
                if range != before {
                    // Touched by hand: stop following the box.
                    self.medsam2.range_pinned = true;
                }
                self.medsam2.range = Some((
                    range.0.min(range.1).min(last),
                    range.0.max(range.1).min(last),
                ));

                if self.medsam2.base_mask.is_some() {
                    ui.checkbox(&mut self.medsam2.merge, "Add to what is already there")
                        .on_hover_text(
                            "Correcting a slice that drifted: scroll to it, draw a fresh \
                             box, and propagate again - the new run is added to the \
                             segmentation instead of replacing it.",
                        );
                }
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.add(egui::TextEdit::singleline(&mut self.medsam2.name).desired_width(160.0));
                });

                // ---- options ---------------------------------------------
                ui.separator();
                ui.collapsing("Options", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Window:");
                        ui.selectable_value(
                            &mut self.medsam2.window,
                            WindowSource::Viewport,
                            "Viewport",
                        );
                        for (i, (name, _, _)) in Window::PRESETS.iter().enumerate() {
                            ui.selectable_value(
                                &mut self.medsam2.window,
                                WindowSource::Preset(i),
                                *name,
                            );
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Model:");
                        egui::ComboBox::from_id_salt("medsam2_variant")
                            .selected_text(self.medsam2.variant.label())
                            .show_ui(ui, |ui| {
                                for v in Variant::ALL {
                                    ui.selectable_value(&mut self.medsam2.variant, v, v.label());
                                }
                            });
                    });
                    ui.checkbox(&mut self.medsam2.cfg.reverse_pass, "Both directions");
                    ui.checkbox(
                        &mut self.medsam2.cfg.largest_component,
                        "Keep only the largest connected component",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Threshold:");
                        ui.add(egui::Slider::new(
                            &mut self.medsam2.cfg.threshold,
                            -4.0..=4.0,
                        ));
                    });
                    device_row(ui, &mut self.medsam2.device);
                    browse = models_dir_row(ui, &mut self.models_dir, ModelsEngine::MedSam2);
                });
                ui.separator();
                let need = weights::download_needed(self.medsam2.variant, &models_dir);
                let weights_note = if need == 0 {
                    "Weights: MedSAM2 (research and education only) - cached ✔.".to_string()
                } else {
                    format!(
                        "Weights: MedSAM2 (research and education only) - {} MB downloaded \
                         once from Hugging Face, at your request, never redistributed.",
                        need / 1_000_000
                    )
                };
                licence_line(ui, &weights_note, true);

                // ---- run -------------------------------------------------
                ui.separator();
                match &self.medsam2_job {
                    Some(job) => cancel = progress_row(ui, &job.progress),
                    None => {
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    self.medsam2.prompt.is_some(),
                                    egui::Button::new("▶ Propagate"),
                                )
                                .on_hover_text("Follow the structure through the slice range")
                                .clicked()
                            {
                                request = Some(Request::Propagate);
                            }
                            if ui.button("Close").clicked() {
                                close = true;
                            }
                        });
                    }
                }
                if let Some(status) = &self.medsam2.status {
                    ui.separator();
                    ui.weak(status);
                }
            },
        );

        if browse {
            if let Some(dir) = Self::pick_folder("Model folder") {
                self.models_dir = dir.display().to_string();
            }
        }
        if clear {
            self.medsam2.prompt = None;
            self.medsam2.dirty = false;
            // The next run is a new structure, not a revision of this one.
            self.medsam2.target_seg = None;
            self.medsam2.base_mask = None;
        }
        if cancel {
            if let Some(job) = &self.medsam2_job {
                job.progress.cancel();
            }
        }
        if !open || close {
            self.medsam2.open = false;
            // The weights stay; the study-sized buffers do not. A run in
            // flight carries on and lands as usual.
            self.medsam2.prep = None;
            if let Some(e) = &self.medsam2.engine {
                e.clear_cache();
            }
            self.persist_settings();
        }
        // An automatic preview waits for the pointer to be released, and for
        // whatever is running to finish.
        if request.is_none()
            && self.medsam2.auto_preview
            && self.medsam2.dirty
            && !running
            && !ctx.input(|i| i.pointer.any_down())
        {
            request = Some(Request::Preview);
        }
        if let Some(r) = request {
            self.start_medsam2(r);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed() -> BoxPrompt {
        let mut b = BoxPrompt::new(ViewPlane::Axial, 4, [10.0, 20.0]);
        b.drag([40.0, 60.0]);
        b.release();
        b
    }

    #[test]
    fn a_drawn_box_sorts_its_corners() {
        let mut b = BoxPrompt::new(ViewPlane::Axial, 0, [40.0, 60.0]);
        b.drag([10.0, 20.0]);
        let (lo, hi) = b.rect();
        assert_eq!(lo, [10.0, 20.0]);
        assert_eq!(hi, [40.0, 60.0]);
        assert!(!b.is_degenerate());
    }

    #[test]
    fn a_click_without_a_drag_is_not_a_box() {
        let mut b = BoxPrompt::new(ViewPlane::Axial, 0, [10.0, 10.0]);
        b.drag([10.5, 10.5]);
        assert!(b.is_degenerate());
    }

    #[test]
    fn a_corner_drag_keeps_the_opposite_corner() {
        let mut b = boxed();
        // grab the top-left and pull it out
        b.press([10.0, 20.0], HANDLE_GRAB);
        b.drag([0.0, 0.0]);
        let (lo, hi) = b.rect();
        assert_eq!(lo, [0.0, 0.0]);
        assert_eq!(hi, [40.0, 60.0], "the opposite corner stayed put");
    }

    #[test]
    fn a_grab_inside_moves_the_whole_box() {
        let mut b = boxed();
        b.press([25.0, 40.0], HANDLE_GRAB);
        b.drag([35.0, 30.0]);
        let (lo, hi) = b.rect();
        assert_eq!(lo, [20.0, 10.0]);
        assert_eq!(hi, [50.0, 50.0]);
    }

    #[test]
    fn a_press_outside_starts_a_new_box() {
        let mut b = boxed();
        b.press([200.0, 200.0], HANDLE_GRAB);
        b.drag([220.0, 230.0]);
        let (lo, hi) = b.rect();
        assert_eq!(lo, [200.0, 200.0]);
        assert_eq!(hi, [220.0, 230.0]);
    }

    #[test]
    fn clamping_keeps_the_box_and_its_clicks_inside_the_slice() {
        let mut b = boxed();
        b.points.push(([500.0, 5.0], true));
        b.points.push(([30.0, 30.0], false));
        // grab the far corner and pull it well past the edge
        b.press([40.0, 60.0], HANDLE_GRAB);
        b.drag([500.0, 500.0]);
        b.release();
        b.clamp([64, 64]);
        let (_, hi) = b.rect();
        assert_eq!(hi, [63.0, 63.0]);
        assert_eq!(b.points.len(), 1, "the click outside the slice was dropped");
    }

    /// A study with the usual head-first-supine geometry, or flipped along
    /// the slice axis.
    fn study(flip_z: bool) -> Volume {
        use crate::geometry::Vec3;
        let dims = [64, 48, 10];
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [1.0, 1.0, 2.0],
            origin: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            row_dir: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            col_dir: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            normal: Vec3 {
                x: 0.0,
                y: 0.0,
                z: if flip_z { -1.0 } else { 1.0 },
            },
            frame_of_reference_uid: "1.2.3".into(),
            min_value: 0,
            max_value: 1,
        }
    }

    fn drawn(slice: usize) -> BoxPrompt {
        let mut b = BoxPrompt::new(ViewPlane::Axial, slice, [10.0, 12.0]);
        b.drag([30.0, 36.0]);
        b.release();
        // a click in the middle of it
        b.points.push(([20.0, 24.0], true));
        b
    }

    #[test]
    fn a_box_on_screen_becomes_a_prompt_in_the_networks_own_grid() {
        use crate::medsam2::prompt::{LABEL_BOX_MAX, LABEL_BOX_MIN, LABEL_POSITIVE};

        let vol = study(false);
        let state = Medsam2State {
            prompt: Some(drawn(4)),
            ..Default::default()
        };
        let (slice, prompt) = state.engine_prompt(&vol).expect("a prompt");
        let EnginePrompt::Points(points) = prompt else {
            panic!("a box is a point prompt");
        };
        assert_eq!(points.len(), 3, "two corners and the click");
        assert_eq!(points[0].label, LABEL_BOX_MIN);
        assert_eq!(points[1].label, LABEL_BOX_MAX);
        assert_eq!(points[2].label, LABEL_POSITIVE);

        // The prompted slice is the one the box was drawn on, mapped through
        // the same reorientation the engine will use.
        let prepared = Prepared::prepare(&vol, Window::new(-100.0, 300.0));
        assert_eq!(slice, prepared.from_volume_index([10, 12, 4])[0]);

        // The corners come back sorted, they enclose the click, and they keep
        // the drawn extent — whichever way the axes were permuted.
        assert!(points[0].row <= points[1].row && points[0].column <= points[1].column);
        assert!(points[2].row >= points[0].row && points[2].row <= points[1].row);
        assert!(points[2].column >= points[0].column && points[2].column <= points[1].column);
        let mut extent = [
            points[1].row - points[0].row,
            points[1].column - points[0].column,
        ];
        extent.sort_by(f32::total_cmp);
        assert_eq!(extent, [20.0, 24.0], "the drawn 20 x 24 box, reoriented");
    }

    #[test]
    fn a_flipped_study_flips_the_prompted_slice() {
        let vol = study(true);
        let mut state = Medsam2State {
            prompt: Some(drawn(4)),
            ..Default::default()
        };
        let (slice, _) = state.engine_prompt(&vol).expect("a prompt");
        // 10 slices, drawn on index 4, and the stack reads the other way.
        assert_eq!(slice, 5);

        let straight = study(false);
        state.prompt = Some(drawn(4));
        let (unflipped, _) = state.engine_prompt(&straight).expect("a prompt");
        assert_eq!(unflipped, 4);
    }

    #[test]
    fn the_range_is_converted_the_same_way() {
        let vol = study(true);
        let state = Medsam2State {
            range: Some((2, 6)),
            ..Default::default()
        };
        let (a, b) = state
            .prepared_range(&vol, ViewPlane::Axial)
            .expect("a range");
        // Reversed, and still ordered.
        assert_eq!((a, b), (3, 7));
    }

    #[test]
    fn a_degenerate_box_is_not_a_prompt() {
        let vol = study(false);
        let mut b = BoxPrompt::new(ViewPlane::Axial, 1, [10.0, 10.0]);
        b.drag([10.5, 10.5]);
        b.release();
        let state = Medsam2State {
            prompt: Some(b),
            ..Default::default()
        };
        assert!(state.engine_prompt(&vol).is_none());
    }
}
