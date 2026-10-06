//! Interactive segmentation: nnInteractive's and VISTA-3D's point mode's
//! user interface.
//!
//! The loop nnInteractive is built for, in the image itself: click on the
//! structure (or drag a box around it on one slice, or scribble on it, or
//! circle it with a lasso), see the 3-D answer, then click again where it
//! is wrong - an *include* prompt where it is missing, an *exclude* prompt
//! where it overreaches. Every prompt goes to the network with all the
//! earlier ones and the current answer, so a few clicks refine rather than
//! restart. Prompts work in all three views.
//!
//! The expensive parts are kept between prompts: the loaded network
//! ([`NniState::model`], for as long as the device and the model folder
//! stay the same) and the session ([`NniState::session`]: the normalized
//! image, the prompts, the current answer), which belongs to one volume.
//! Prompts given while the network is busy wait in a queue and run in
//! order. The answer is written into one segmentation of the workspace,
//! replaced in place after every prompt; *New object* leaves it as it is
//! and starts the next.
//!
//! Coordinates: a prompt is drawn in a view's pixel coordinates on one
//! slice ([`Mark`]); the worker turns it into volume voxels and those into
//! the session's own axes ([`prompt_to_session`]), the only place that
//! happens.
//!
//! VISTA-3D's point mode is the second engine of the section: clicks only
//! (no boxes, strokes or outlines), each answered by a 128-cubed window
//! around it; the windows a new click does not touch are kept.

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::models::Engine as ModelsEngine;
use crate::nn::device::DevicePref;
use crate::nninteractive::{self, Kind, Model, VolumeSession};
use crate::vista3d;
use crate::vista3d::session::{PointModel, PointSession};

/// Which network answers the prompts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NniEngine {
    NnInteractive,
    Vista3d,
}

impl NniEngine {
    fn label(self) -> &'static str {
        match self {
            NniEngine::NnInteractive => "nnInteractive",
            NniEngine::Vista3d => "VISTA-3D points",
        }
    }

    fn takes(self, tool: NniTool) -> bool {
        match self {
            NniEngine::NnInteractive => true,
            NniEngine::Vista3d => matches!(tool, NniTool::Point | NniTool::Navigate),
        }
    }
}

/// The session of either engine.
pub(super) enum AnySession {
    Nni(VolumeSession),
    Vista(PointSession),
}

use super::*;

/// What a left press in a view does while the section is open.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NniTool {
    /// The left button moves the crosshair, as everywhere else.
    Navigate,
    Point,
    /// A box on one slice.
    Box,
    Scribble,
    Lasso,
}

impl NniTool {
    fn label(self) -> &'static str {
        match self {
            NniTool::Navigate => "⌖ Navigate",
            NniTool::Point => "● Point",
            NniTool::Box => "⬚ Box",
            NniTool::Scribble => "✏ Scribble",
            NniTool::Lasso => "◌ Lasso",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            NniTool::Navigate => "The left button moves the crosshair; prompts are off",
            NniTool::Point => "Click on the structure (include) or beside it (exclude)",
            NniTool::Box => "Drag a box around the structure on one slice",
            NniTool::Scribble => "Draw a stroke over the structure",
            NniTool::Lasso => "Draw a closed outline around the structure on one slice",
        }
    }
}

/// One prompt as drawn: the view, the slice, and the pointer positions in
/// the view's pixel coordinates (one point, two box corners, or a stroke).
#[derive(Clone, Debug)]
pub(super) struct Mark {
    pub plane: ViewPlane,
    pub slice: usize,
    pub tool: NniTool,
    pub include: bool,
    pub pts: Vec<[f32; 2]>,
}

impl Mark {
    /// Is this worth sending? A box needs an area, a lasso a shape.
    fn is_usable(&self) -> bool {
        match self.tool {
            NniTool::Point | NniTool::Scribble => !self.pts.is_empty(),
            NniTool::Box => {
                self.pts.len() == 2
                    && (self.pts[0][0] - self.pts[1][0]).abs() >= 1.0
                    && (self.pts[0][1] - self.pts[1][1]).abs() >= 1.0
            }
            NniTool::Lasso => self.pts.len() >= 3,
            NniTool::Navigate => false,
        }
    }
}

/// The volume a session was built on: rebuild when this changes.
#[derive(Clone, PartialEq, Debug)]
pub(super) struct NniKey {
    pub engine: NniEngine,
    pub slot: usize,
    pub dims: [usize; 3],
    pub uid: String,
    /// The voxels themselves change when a 4D group steps to another phase
    /// on the same lattice; the volume's allocation tells them apart.
    pub data: usize,
}

/// Everything the section owns.
pub(super) struct NniState {
    pub open: bool,
    pub slot: usize,
    pub engine: NniEngine,
    pub tool: NniTool,
    /// Whether the next prompt says "this is it" or "this is not".
    pub include: bool,
    pub device: DevicePref,
    pub autozoom: bool,
    pub name: String,
    pub status: Option<String>,
    /// The loaded networks.
    pub model: Arc<KeptModel<Model>>,
    pub vista_model: Arc<KeptModel<PointModel>>,
    pub session: Option<(NniKey, Arc<Mutex<AnySession>>)>,
    /// The segmentation the answer is written to.
    pub target_seg: Option<usize>,
    /// Prompts given for the current object, for drawing.
    pub marks: Vec<Mark>,
    /// The box, stroke or outline being drawn.
    pub drawing: Option<Mark>,
    /// Prompts waiting for the network.
    pub queue: VecDeque<Mark>,
    /// Whether the run in flight adds a prompt (rather than taking one
    /// back), so a failure knows which mark to drop.
    pub inflight_mark: bool,
}

impl Default for NniState {
    fn default() -> NniState {
        NniState {
            open: false,
            slot: 0,
            engine: NniEngine::NnInteractive,
            tool: NniTool::Point,
            include: true,
            device: DevicePref::Auto,
            autozoom: true,
            name: "Interactive".to_string(),
            status: None,
            model: Arc::default(),
            vista_model: Arc::default(),
            session: None,
            target_seg: None,
            marks: Vec::new(),
            drawing: None,
            queue: VecDeque::new(),
            inflight_mark: false,
        }
    }
}

/// What one prompt's run hands back.
pub(super) struct NniDone {
    pub key: NniKey,
    pub session: Arc<Mutex<AnySession>>,
    /// The answer on the volume's own grid.
    pub mask: Vec<u8>,
    pub voxels: u64,
    pub passes: usize,
    pub zoom: f64,
    pub elapsed_secs: f64,
    pub device: String,
}

/// Everything a run needs, snapshotted when it starts.
struct NniRequest {
    model: Arc<KeptModel<Model>>,
    vista_model: Arc<KeptModel<PointModel>>,
    session: Option<Arc<Mutex<AnySession>>>,
    volume: Arc<Volume>,
    key: NniKey,
    device: DevicePref,
    root: PathBuf,
    autozoom: bool,
    /// A prompt to add; `None` takes the last one back (VISTA-3D, whose
    /// answer has to be recomputed).
    mark: Option<Mark>,
}

/// The voxels of the view's pixels `px`, on the mark's slice.
fn voxels_of(vol: &Volume, m: &Mark, px: &[[i64; 2]]) -> Vec<[i64; 3]> {
    px.iter()
        .map(|p| {
            let v = vol.plane_pixel_to_voxel(m.plane, m.slice, p[0] as f64, p[1] as f64);
            [
                v[0].round() as i64,
                v[1].round() as i64,
                v[2].round() as i64,
            ]
        })
        .filter(|v| (0..3).all(|a| v[a] >= 0 && (v[a] as usize) < vol.dims[a]))
        .collect()
}

/// The pixels within `r` of the polyline through `pts`.
fn stroke_pixels(pts: &[[f32; 2]], r: f32) -> Vec<[i64; 2]> {
    let mut out = std::collections::BTreeSet::new();
    let mut stamp = |c: [f32; 2]| {
        let reach = r.ceil() as i64 + 1;
        let (cx, cy) = (c[0].round() as i64, c[1].round() as i64);
        for y in cy - reach..=cy + reach {
            for x in cx - reach..=cx + reach {
                let (dx, dy) = (x as f32 - c[0], y as f32 - c[1]);
                if dx * dx + dy * dy <= r * r {
                    out.insert([x, y]);
                }
            }
        }
    };
    stamp(pts[0]);
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let len = (b[0] - a[0]).hypot(b[1] - a[1]);
        let steps = (len * 4.0).ceil().max(1.0) as usize;
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            stamp([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
        }
    }
    out.into_iter().collect()
}

/// The pixels whose centres lie inside the closed polygon `pts`
/// (even-odd), with the outline itself.
fn lasso_pixels(pts: &[[f32; 2]]) -> Vec<[i64; 2]> {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in pts {
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let mut out: std::collections::BTreeSet<[i64; 2]> =
        stroke_pixels(pts, 0.5).into_iter().collect();
    for y in lo[1].floor() as i64..=hi[1].ceil() as i64 {
        for x in lo[0].floor() as i64..=hi[0].ceil() as i64 {
            let (px, py) = (x as f32, y as f32);
            let mut inside = false;
            let n = pts.len();
            for i in 0..n {
                let (a, b) = (pts[i], pts[(i + 1) % n]);
                if (a[1] > py) != (b[1] > py) {
                    let xc = a[0] + (py - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                    if px < xc {
                        inside = !inside;
                    }
                }
            }
            if inside {
                out.insert([x, y]);
            }
        }
    }
    out.into_iter().collect()
}

/// Hand one drawn prompt to the session, in its own axes.
fn prompt_to_session(
    vs: &mut VolumeSession,
    vol: &Volume,
    m: &Mark,
    scribble_width: usize,
) -> anyhow::Result<()> {
    use anyhow::bail;
    let ax = &vs.axes;
    // A volume voxel index in the session's axes.
    let to_session = |v: [i64; 3]| -> [usize; 3] {
        std::array::from_fn(|a| {
            let x = v[ax.perm[a]] as usize;
            if ax.flip[a] {
                ax.dims[a] - 1 - x
            } else {
                x
            }
        })
    };
    match m.tool {
        NniTool::Point => {
            let p = m.pts[0];
            let v = vol.plane_pixel_to_voxel(m.plane, m.slice, f64::from(p[0]), f64::from(p[1]));
            let at = vs.to_session(v);
            vs.session.add_point(at, m.include)
        }
        NniTool::Box => {
            let corners = voxels_of(
                vol,
                m,
                &[
                    [m.pts[0][0].round() as i64, m.pts[0][1].round() as i64],
                    [m.pts[1][0].round() as i64, m.pts[1][1].round() as i64],
                ],
            );
            if corners.len() != 2 {
                bail!("the box reaches outside the image");
            }
            let s0 = to_session(corners[0]);
            let s1 = to_session(corners[1]);
            let lo: [f64; 3] = std::array::from_fn(|a| s0[a].min(s1[a]) as f64);
            let hi: [f64; 3] = std::array::from_fn(|a| (s0[a].max(s1[a]) + 1) as f64);
            vs.session.add_box(lo, hi, m.include)
        }
        NniTool::Scribble | NniTool::Lasso => {
            let px = if m.tool == NniTool::Scribble {
                stroke_pixels(&m.pts, scribble_width as f32 / 2.0)
            } else {
                lasso_pixels(&m.pts)
            };
            let vox: Vec<[usize; 3]> = voxels_of(vol, m, &px).into_iter().map(to_session).collect();
            if vox.is_empty() {
                bail!("the prompt lies outside the image");
            }
            let lo: [usize; 3] =
                std::array::from_fn(|a| vox.iter().map(|v| v[a]).min().unwrap_or(0));
            let hi: [usize; 3] =
                std::array::from_fn(|a| vox.iter().map(|v| v[a]).max().unwrap_or(0) + 1);
            let d: [usize; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
            let mut bits = vec![0u8; d[0] * d[1] * d[2]];
            for v in &vox {
                bits[((v[0] - lo[0]) * d[1] + (v[1] - lo[1])) * d[2] + (v[2] - lo[2])] = 1;
            }
            let kind = if m.tool == NniTool::Scribble {
                Kind::Scribble
            } else {
                Kind::Lasso
            };
            vs.session.add_mask(kind, lo, d, bits, m.include)
        }
        NniTool::Navigate => Ok(()),
    }
}

/// The background half of one prompt: load what is missing, apply the
/// prompt, predict.
fn run_job(req: NniRequest, progress: &Progress) -> anyhow::Result<NniDone> {
    match req.key.engine {
        NniEngine::NnInteractive => run_nni(req, progress),
        NniEngine::Vista3d => run_vista(req, progress),
    }
}

fn run_nni(req: NniRequest, progress: &Progress) -> anyhow::Result<NniDone> {
    let t0 = std::time::Instant::now();
    let key = ModelKey {
        models_dir: req.root.clone(),
        device: req.device,
        variant: "nnInteractive",
    };
    let model = req.model.get_or_load(&key, || {
        progress.set("Loading the network");
        Model::load(&req.root, req.device, progress)
    })?;
    progress.set_device(&model.device);
    let session = match req.session {
        Some(s) => s,
        None => {
            progress.set("Preparing the image");
            let vs = VolumeSession::new(&req.volume, model.settings.clone())?;
            Arc::new(Mutex::new(AnySession::Nni(vs)))
        }
    };
    let (out, mask) = {
        let mut guard = session.lock().unwrap_or_else(|p| p.into_inner());
        let AnySession::Nni(vs) = &mut *guard else {
            anyhow::bail!("the session belongs to the other engine");
        };
        vs.session.settings.autozoom = req.autozoom;
        let Some(mark) = &req.mark else {
            anyhow::bail!("nothing to do");
        };
        prompt_to_session(vs, &req.volume, mark, model.scribble_thickness)?;
        let out = vs.session.predict(model.as_ref(), false, progress);
        if out.is_err() {
            // A cancelled or failed prediction must not leave its prompt in
            // the session, where the next one would carry it.
            vs.session.undo();
        }
        (out?, vs.mask_on_volume())
    };
    let voxels = mask.iter().filter(|&&v| v != 0).count() as u64;
    Ok(NniDone {
        key: req.key,
        session,
        mask,
        voxels,
        passes: out.passes,
        zoom: out.zoom,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        device: model.device.clone(),
    })
}

fn run_vista(req: NniRequest, progress: &Progress) -> anyhow::Result<NniDone> {
    let t0 = std::time::Instant::now();
    let key = ModelKey {
        models_dir: req.root.clone(),
        device: req.device,
        variant: "VISTA-3D points",
    };
    let model = req.vista_model.get_or_load(&key, || {
        progress.set("Loading the network");
        PointModel::load(&req.root, req.device, progress)
    })?;
    progress.set_device(&model.device);
    let session = match req.session {
        Some(s) => s,
        None => {
            progress.set("Preparing the volume");
            Arc::new(Mutex::new(AnySession::Vista(PointSession::new(
                &req.volume,
            ))))
        }
    };
    let mask = {
        let mut guard = session.lock().unwrap_or_else(|p| p.into_inner());
        let AnySession::Vista(vs) = &mut *guard else {
            anyhow::bail!("the session belongs to the other engine");
        };
        match &req.mark {
            Some(m) => {
                let p = m.pts[0];
                let v = req.volume.plane_pixel_to_voxel(
                    m.plane,
                    m.slice,
                    f64::from(p[0]),
                    f64::from(p[1]),
                );
                vs.add(v, m.include)?;
            }
            None => {
                vs.undo();
            }
        }
        let mask = vs.segment(model.as_ref(), &req.volume, progress);
        if mask.is_err() && req.mark.is_some() {
            vs.undo();
        }
        mask?
    };
    let voxels = mask.iter().filter(|&&v| v != 0).count() as u64;
    Ok(NniDone {
        key: req.key,
        session,
        mask,
        voxels,
        passes: 0,
        zoom: 1.0,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        device: model.device.clone(),
    })
}

impl ViewerApp {
    /// Open the section on `slot`.
    pub(super) fn open_nni_panel(&mut self, slot: usize) {
        if !self.slots[slot].has_volume() {
            return;
        }
        if self.nni.slot != slot {
            if self.nni_job.is_some() {
                return;
            }
            self.nni_new_object();
            self.nni.session = None;
        }
        self.nni.slot = slot;
        self.nni.open = true;
    }

    /// The key of the volume `slot` shows, for the engine picked.
    fn nni_key(&self, slot: usize) -> Option<NniKey> {
        let study = self.slots[slot].study.as_ref()?;
        Some(NniKey {
            engine: self.nni.engine,
            slot,
            dims: study.volume.dims,
            uid: study.volume.frame_of_reference_uid.clone(),
            data: study.volume.data.as_ptr() as usize,
        })
    }

    /// Does the left button belong to the section in this workspace?
    pub(super) fn nni_drawing_in(&self, slot: usize) -> bool {
        self.nni.open
            && self.nni.slot == slot
            && self.nni.tool != NniTool::Navigate
            && self.nni.engine.takes(self.nni.tool)
            && self.slots[slot].has_volume()
    }

    /// Are its prompts drawn in this workspace?
    pub(super) fn nni_showing_in(&self, slot: usize) -> bool {
        self.nni.open && self.nni.slot == slot && self.slots[slot].has_volume()
    }

    pub(super) fn nni_press(&mut self, plane: ViewPlane, slice: usize, p: [f32; 2]) {
        let tool = self.nni.tool;
        self.nni.drawing = Some(Mark {
            plane,
            slice,
            tool,
            include: self.nni.include,
            pts: vec![p, p],
        });
        if matches!(tool, NniTool::Point | NniTool::Scribble | NniTool::Lasso) {
            if let Some(m) = self.nni.drawing.as_mut() {
                m.pts.truncate(1);
            }
        }
    }

    pub(super) fn nni_drag(&mut self, p: [f32; 2]) {
        let Some(m) = self.nni.drawing.as_mut() else {
            return;
        };
        match m.tool {
            NniTool::Box => m.pts[1] = p,
            NniTool::Scribble | NniTool::Lasso => {
                let last = *m.pts.last().unwrap_or(&p);
                // A new vertex per half pixel of motion is plenty.
                if (p[0] - last[0]).hypot(p[1] - last[1]) >= 0.5 {
                    m.pts.push(p);
                }
            }
            _ => {}
        }
    }

    /// The pointer went up: the drawn prompt is queued.
    pub(super) fn nni_release(&mut self) {
        let Some(m) = self.nni.drawing.take() else {
            return;
        };
        if !m.is_usable() || !self.nni.engine.takes(m.tool) {
            return;
        }
        self.nni.queue.push_back(m);
        self.nni_next();
    }

    /// Hand the worker one request: a prompt, or (VISTA-3D) taking the
    /// last one back.
    fn nni_spawn(&mut self, mark: Option<Mark>) {
        let slot = self.nni.slot;
        let Some(key) = self.nni_key(slot) else {
            self.nni.queue.clear();
            return;
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let volume = study.volume.clone();
        let session = self
            .nni
            .session
            .as_ref()
            .filter(|(k, _)| *k == key)
            .map(|(_, s)| s.clone());
        if session.is_none() {
            // A new volume: whatever was being segmented belongs to the old one.
            self.nni.target_seg = None;
            self.nni.marks.clear();
        }
        if let Some(m) = &mark {
            self.nni.marks.push(m.clone());
        }
        self.nni.inflight_mark = mark.is_some();
        let req = NniRequest {
            model: self.nni.model.clone(),
            vista_model: self.nni.vista_model.clone(),
            session,
            volume,
            key,
            device: self.nni.device,
            root: self.models_root(),
            autozoom: self.nni.autozoom,
            mark,
        };
        self.persist_settings();
        let progress = Arc::new(Progress::default());
        progress.set("Segmenting");
        self.nni_job = Some(Job::spawn(progress, move |p| (slot, run_job(req, p))));
    }

    /// Start the next queued prompt when nothing is running.
    pub(super) fn nni_next(&mut self) {
        if self.nni_job.is_some() {
            return;
        }
        if let Some(mark) = self.nni.queue.pop_front() {
            self.nni_spawn(Some(mark));
        }
    }

    /// A prompt's run finished: write the answer into the target
    /// segmentation and report.
    pub(super) fn on_nni_done(&mut self, slot: usize, done: NniDone) {
        self.nni.session = Some((done.key.clone(), done.session.clone()));
        if !self.slot_still_shows(slot, done.key.dims, &done.key.uid) {
            self.error = Some(stale_result(&INTERACTIVE_SEG));
            self.nni.queue.clear();
            return;
        }
        self.nni_land(slot, done.key.dims, done.mask);
        let passes = match done.key.engine {
            NniEngine::NnInteractive => format!(
                " - {} network pass(es){}",
                done.passes,
                if done.zoom > 1.0 {
                    format!(", zoomed out x{:.2}", done.zoom)
                } else {
                    String::new()
                }
            ),
            NniEngine::Vista3d => String::new(),
        };
        self.nni.status = Some(format!(
            "{} prompt(s): {} voxels ({:.2} cm³){passes} in {:.1} s on {}",
            self.nni.marks.len(),
            done.voxels,
            self.slots[slot].voxels_cm3(done.voxels),
            done.elapsed_secs,
            done.device
        ));
        self.nni_next();
    }

    /// Write `mask` into the target segmentation, making one when there is
    /// none.
    fn nni_land(&mut self, slot: usize, dims: [usize; 3], mask: Vec<u8>) {
        let name = self.nni.name.trim().to_string();
        match self.nni.target_seg {
            Some(i)
                if i < self.slots[slot].segs().len() && self.slots[slot].segs()[i].dims == dims =>
            {
                if let Some(segs) = self.slots[slot].segs_mut() {
                    segs[i].replace_mask(mask);
                }
                self.slots[slot].active_seg = i;
            }
            _ => {
                let i = self.add_segmentation(slot, name, dims, &mask);
                self.nni.target_seg = Some(i);
            }
        }
    }

    /// Keep the current answer as it is and start a new object.
    pub(super) fn nni_new_object(&mut self) {
        self.nni.target_seg = None;
        self.nni.marks.clear();
        self.nni.drawing = None;
        self.nni.queue.clear();
        if let Some((_, s)) = &self.nni.session {
            match &mut *s.lock().unwrap_or_else(|p| p.into_inner()) {
                AnySession::Nni(vs) => vs.session.reset(),
                AnySession::Vista(vs) => vs.reset(),
            }
        }
    }

    /// The section.
    pub(super) fn nni_section(&mut self, ui: &mut egui::Ui) {
        self.open_nni_panel(self.auto.slot);
        let slot = self.nni.slot;
        if !self.slots[slot].has_volume() {
            return;
        }
        let root = self.models_root();
        let running = self.nni_job.is_some();
        let mut cancel = false;
        let mut browse = false;
        let mut new_object = false;
        let mut undo = false;
        let mut start_from: Option<Vec<u8>> = None;
        let engine_before = self.nni.engine;

        ui.label(
            "Segments whatever you point at, with nnInteractive or VISTA-3D's point mode, \
             re-implemented natively in Rust: click on the structure in any view, then click \
             again where the answer is wrong. Every prompt refines the same object.",
        );
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label("Engine:");
            for (e, name, hint) in [
                (
                    NniEngine::NnInteractive,
                    "nnInteractive",
                    "Points, boxes, scribbles and lassos, on CT, MR or PET; zooms out by \
                     itself for large structures",
                ),
                (
                    NniEngine::Vista3d,
                    "VISTA-3D points",
                    "Clicks only, on CT; NVIDIA's VISTA-3D, the same weights as its \
                     automatic mode",
                ),
            ] {
                if ui
                    .add_enabled(
                        !running,
                        egui::Button::selectable(self.nni.engine == e, name),
                    )
                    .on_hover_text(hint)
                    .clicked()
                {
                    self.nni.engine = e;
                }
            }
        });
        let engine = self.nni.engine;
        ui.horizontal_wrapped(|ui| {
            ui.label("Prompt:");
            for t in [
                NniTool::Point,
                NniTool::Box,
                NniTool::Scribble,
                NniTool::Lasso,
                NniTool::Navigate,
            ] {
                // Added straight into the wrapping row: a child Ui per
                // button would take the rest of the row's width and wrap
                // the last label letter by letter.
                let resp = ui
                    .add_enabled(
                        engine.takes(t),
                        egui::Button::selectable(self.nni.tool == t, t.label()),
                    )
                    .on_hover_text(t.hint())
                    .on_disabled_hover_text(format!("{} takes clicks only", engine.label()));
                if resp.clicked() {
                    self.nni.tool = t;
                }
            }
        });
        if !engine.takes(self.nni.tool) {
            self.nni.tool = NniTool::Point;
        }
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.nni.include, true, "➕ Include")
                .on_hover_text("The prompt marks the structure");
            ui.selectable_value(&mut self.nni.include, false, "➖ Exclude")
                .on_hover_text("The prompt marks what must stay out");
        });
        let n_marks = self.nni.marks.len();
        let queued = self.nni.queue.len();
        ui.weak(if n_marks == 0 {
            "No prompt yet.".to_string()
        } else if queued > 0 {
            format!("{n_marks} prompt(s) given, {queued} waiting")
        } else {
            format!("{n_marks} prompt(s) given")
        });
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!running && n_marks > 0, egui::Button::new("↺ Undo prompt"))
                .on_hover_text("Take back the last prompt and its answer")
                .clicked()
            {
                undo = true;
            }
            if ui
                .add_enabled(!running, egui::Button::new("➕ New object"))
                .on_hover_text("Keep this segmentation as it is and start the next one")
                .clicked()
            {
                new_object = true;
            }
        });
        let active = self.slots[slot].active_seg;
        if engine == NniEngine::NnInteractive {
            let active_mask = self.slots[slot]
                .segs()
                .get(active)
                .filter(|s| s.count > 0)
                .map(|s| (s.name.clone(), s.mask.clone()));
            if let Some((seg_name, mask)) = active_mask {
                if ui
                    .add_enabled(
                        !running && self.nni.target_seg != Some(active),
                        egui::Button::new(format!("Refine \"{seg_name}\"")),
                    )
                    .on_hover_text(
                        "Start from the active segmentation - an auto-segmented organ, a drawn \
                         outline - and correct it with prompts",
                    )
                    .clicked()
                {
                    start_from = Some(mask);
                }
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Name:");
            ui.add(egui::TextEdit::singleline(&mut self.nni.name).desired_width(160.0));
        });
        ui.separator();
        ui.collapsing("Options", |ui| {
            if engine == NniEngine::NnInteractive {
                ui.checkbox(
                    &mut self.nni.autozoom,
                    "Zoom out when the answer reaches the border",
                )
                .on_hover_text(
                    "nnInteractive's AutoZoom: when the answer touches the edge of the \
                         192-voxel window, look again at up to 4x the field of view, then \
                         refine at full resolution. Off: one pass per prompt, limited to the \
                         window.",
                );
            }
            device_row(ui, &mut self.nni.device);
            let models_engine = match engine {
                NniEngine::NnInteractive => ModelsEngine::NnInteractive,
                NniEngine::Vista3d => ModelsEngine::Vista3d,
            };
            browse = models_dir_row(ui, &mut self.models_dir, models_engine);
        });
        ui.separator();
        let note = match engine {
            NniEngine::NnInteractive => {
                let need = nninteractive::download_needed(&root);
                if need == 0 {
                    "Weights: nnInteractive v1.0 (DKFZ), CC BY-NC-SA 4.0 - non-commercial use \
                     only - cached ✔."
                        .to_string()
                } else {
                    format!(
                        "Weights: nnInteractive v1.0 (DKFZ), CC BY-NC-SA 4.0 - non-commercial \
                         use only; {} MB downloaded once from Hugging Face at your request, \
                         never redistributed.",
                        need / 1_000_000
                    )
                }
            }
            NniEngine::Vista3d => {
                let need = vista3d::download_needed(&root);
                if need == 0 {
                    "Weights: NVIDIA NV-Segment-CT (VISTA-3D), NVIDIA Open Model License - \
                     cached ✔."
                        .to_string()
                } else {
                    format!(
                        "Weights: NVIDIA NV-Segment-CT (VISTA-3D), NVIDIA Open Model License - \
                         commercial use allowed with attribution and its conditions; {} MB \
                         downloaded once from Hugging Face.",
                        need / 1_000_000
                    )
                }
            }
        };
        licence_line(ui, &note, true);
        if self.nni.device != DevicePref::Gpu {
            ui.weak(match engine {
                NniEngine::NnInteractive => {
                    "On the CPU one prompt takes a minute or more (each pass is a 192-cubed \
                     residual U-Net); the authors recommend a GPU."
                }
                NniEngine::Vista3d => {
                    "On the CPU every click costs a minute or more (a 128-cubed window through \
                     the whole network); a GPU is recommended."
                }
            });
        }
        ui.separator();
        if let Some(job) = &self.nni_job {
            cancel = progress_row(ui, &job.progress);
        } else if n_marks == 0 {
            ui.weak("Pick a prompt and use it in a view.");
        }
        if let Some(status) = &self.nni.status {
            ui.separator();
            ui.weak(status);
        }

        if browse {
            self.ask_folder("Model folder", |app, dir| {
                app.models_dir = dir.display().to_string();
            });
        }
        if self.nni.engine != engine_before {
            // Another engine is another object.
            let now = self.nni.engine;
            self.nni.engine = engine_before;
            self.nni_new_object();
            self.nni.engine = now;
            self.nni.session = None;
            self.nni.status = None;
        }
        if cancel {
            self.nni.queue.clear();
        }
        cancel_if(cancel, &self.nni_job);
        if undo {
            self.nni_undo(slot);
        }
        if new_object {
            self.nni_new_object();
            self.nni.status = None;
        }
        if let Some(mask) = start_from {
            self.nni_start_from(slot, active, mask);
        }
    }

    fn nni_undo(&mut self, slot: usize) {
        let Some(key) = self.nni_key(slot) else {
            return;
        };
        let Some((k, s)) = &self.nni.session else {
            return;
        };
        if *k != key {
            return;
        }
        let s = s.clone();
        let mut guard = s.lock().unwrap_or_else(|p| p.into_inner());
        match &mut *guard {
            AnySession::Nni(vs) => {
                if !vs.session.undo() {
                    return;
                }
                let mask = vs.mask_on_volume();
                drop(guard);
                self.nni.marks.pop();
                self.nni_land(slot, key.dims, mask);
                self.nni.status = Some("Took back the last prompt.".to_string());
            }
            AnySession::Vista(_) => {
                // The answer has to be recomputed (from the windows kept).
                drop(guard);
                self.nni.marks.pop();
                self.nni_spawn(None);
            }
        }
    }

    /// Make the active segmentation the starting point (nnInteractive):
    /// with no session yet, the next prompt builds one and is applied on
    /// top of it.
    fn nni_start_from(&mut self, slot: usize, index: usize, mask: Vec<u8>) {
        let Some(key) = self.nni_key(slot) else {
            return;
        };
        let session = match &self.nni.session {
            Some((k, s)) if *k == key => s.clone(),
            _ => {
                // Building the session needs the network's settings only;
                // the weights are not needed until the first prompt.
                let Some(study) = self.slots[slot].study.as_ref() else {
                    return;
                };
                let settings = nninteractive::Settings::v1([192, 192, 192]);
                match VolumeSession::new(&study.volume, settings) {
                    Ok(vs) => Arc::new(Mutex::new(AnySession::Nni(vs))),
                    Err(e) => {
                        self.error = Some(format!("{e:#}"));
                        return;
                    }
                }
            }
        };
        {
            let mut guard = session.lock().unwrap_or_else(|p| p.into_inner());
            let AnySession::Nni(vs) = &mut *guard else {
                return;
            };
            let m = vs.from_volume(&mask);
            if let Err(e) = vs.session.set_initial(&m, false) {
                self.error = Some(format!("{e:#}"));
                return;
            }
        }
        self.nni.session = Some((key, session));
        self.nni.target_seg = Some(index);
        self.nni.marks.clear();
        self.nni.queue.clear();
        if let Some(s) = self.slots[slot].segs().get(index) {
            self.nni.name = s.name.clone();
        }
        self.nni.status =
            Some("Starting from the active segmentation: prompts now correct it.".to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stroke_covers_its_path_and_a_lasso_its_inside() {
        let s = stroke_pixels(&[[0.0, 0.0], [4.0, 0.0]], 1.0);
        for x in 0..=4 {
            assert!(s.contains(&[x, 0]), "{x}");
        }
        assert!(!s.contains(&[2, 2]));
        let l = lasso_pixels(&[[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]]);
        assert!(l.contains(&[3, 3]));
        assert!(!l.contains(&[8, 3]));
    }

    #[test]
    fn marks_need_a_shape_to_be_sent() {
        let m = |tool, pts: Vec<[f32; 2]>| Mark {
            plane: ViewPlane::Axial,
            slice: 0,
            tool,
            include: true,
            pts,
        };
        assert!(m(NniTool::Point, vec![[1.0, 1.0]]).is_usable());
        assert!(!m(NniTool::Box, vec![[1.0, 1.0], [1.2, 9.0]]).is_usable());
        assert!(m(NniTool::Box, vec![[1.0, 1.0], [5.0, 9.0]]).is_usable());
        assert!(!m(NniTool::Lasso, vec![[1.0, 1.0], [5.0, 9.0]]).is_usable());
        assert!(!m(NniTool::Navigate, vec![[1.0, 1.0]]).is_usable());
    }
}
