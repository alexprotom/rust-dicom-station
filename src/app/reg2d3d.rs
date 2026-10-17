//! *Modules ▶ Image registration ▶ 2D/3D registration*: the pose of a CT
//! that makes its DRRs match the workspace's planar images - kV radiographs,
//! MV portal images - by their intensities, by implanted markers, or the
//! markers first and then the intensities.
//!
//! The section chooses the radiographs and where each one's geometry comes
//! from (its own RT Image attributes, a beam of the plan, a calibrated room
//! view, IEC numbers typed in), the isocentre and the patient's position,
//! the region and the method, and runs [`crate::setup::run`] in the
//! background. What comes back is filed with the study (*Setups* in the data
//! tree) and put on screen: each radiograph stacked with its DRR at the
//! answer in the row's planar panes, where the pane's compare modes show how
//! well they agree, with the isocentre, the markers and the outlines of
//! chosen structures drawn over them. The result rows give the correction
//! in patient coordinates and as couch moves; the nudge buttons move the
//! pose by hand and draw the DRRs again; the apply row makes the positioned
//! copy (the study as the patient was found, registered to the CT), saves
//! the report and writes the registration and the DRRs as DICOM.

use crate::drr::Geometry;
use crate::iec::{PatientFrame, PatientPosition};
use crate::nn::device::DevicePref;
use crate::registration::metric2d::Metric2d;
use crate::registration::projective::{ClipBox, Dof2d3d, Init2d3d, Optimizer2d};
use crate::settings::Reg2d3dPrefs;
use crate::setup::{
    self, GeometrySource, MarkerSource, SetupMethod, SetupOptions, SetupResult, ViewChoice,
};

use super::*;

/// The section's glyph and name.
pub(super) const REG2D3D_TOOL: ToolInfo = ToolInfo {
    glyph: "⌖",
    name: "2D/3D registration",
};

/// One planar image of the workspace, as the section offers it.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct ViewPick {
    pub on: bool,
    pub source: GeometrySource,
    /// The region compared: polygons in the image's pixels.
    pub region: Vec<Vec<[f64; 2]>>,
    /// Markers placed by hand: name and pixel.
    pub picks: Vec<(String, [f64; 2])>,
}

/// What the left button does on a planar pane showing a radiograph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(super) enum PaneTool {
    #[default]
    None,
    Rectangle,
    Polygon,
    /// Place the marker chosen in the section.
    Pick,
}

/// A change a planar pane asks of the regions or the picks.
pub(super) enum PaneEdit {
    /// The rectangle being dragged, corner to corner in pixels; `done`
    /// when the button is let go.
    Rect {
        image: usize,
        a: [f64; 2],
        b: [f64; 2],
        done: bool,
    },
    Vertex {
        image: usize,
        p: [f64; 2],
    },
    Close {
        image: usize,
    },
    Pick {
        image: usize,
        p: [f64; 2],
    },
}

/// An outline drawn over the radiographs: segments in each view's raster,
/// `0..1` across and down.
pub(super) struct Outline {
    pub color: Color32,
    /// Per view of the current setup.
    pub segments: Vec<Vec<([f32; 2], [f32; 2])>>,
}

/// The section's state.
pub(super) struct Reg2d3dState {
    /// The workspace whose CT and planar images are registered.
    pub slot: usize,
    /// One per planar image of the workspace, in order.
    pub views: Vec<ViewPick>,
    pub opts: SetupOptions,
    pub clip_on: bool,
    pub clip_half: f64,
    /// The isocentre typed or taken from the crosshair; the images' or the
    /// plan's while `None`.
    pub iso: Option<Vec3>,
    pub status: Option<String>,
    /// The setup on display: an index into the study's setups.
    pub current: Option<usize>,
    /// The planar images made for the overlay of the current setup.
    pub drrs: Vec<usize>,
    /// Nudge steps: millimetres and degrees.
    pub step_mm: f64,
    pub step_deg: f64,
    pub show_iso: bool,
    pub show_markers: bool,
    /// Structures whose outlines are drawn, by name.
    pub outlined: Vec<String>,
    pub outlines: Vec<Outline>,
    /// What the outlines were made for: (setup index, its pose's
    /// parameters, the names).
    pub outlines_for: Option<(usize, [u64; 6], Vec<String>)>,
    pub rooms: Option<(crate::rooms::RoomsFile, Option<String>)>,
    /// The clip box about a structure, by name, with a margin, rather than
    /// a cube about the isocentre.
    pub clip_structure: Option<String>,
    pub clip_margin: f64,
    /// Draw the clip box in the CT's views, and the box as last worked out
    /// for what the section says (`None` inside when it cannot be).
    pub show_clip: bool,
    pub clip_shown: Option<(String, Option<ClipBox>)>,
    /// The correction beyond these is coloured: mm and degrees.
    pub tolerance: [f64; 2],
    pub tool: PaneTool,
    /// A region being drawn: the image and its points so far.
    pub drawing: Option<(usize, Vec<[f64; 2]>)>,
    /// The marker the next click on a radiograph places.
    pub pick_marker: Option<String>,
    /// The names of the markers the run would use, for the source they
    /// were listed for.
    pub marker_names: Option<(String, Vec<String>)>,
    /// The room last chosen for an image.
    pub room: String,
}

impl Reg2d3dState {
    fn new(slot: usize, prefs: &Reg2d3dPrefs) -> Self {
        Reg2d3dState {
            slot,
            views: Vec::new(),
            opts: SetupOptions {
                metric: Metric2d::from_key(&prefs.metric).unwrap_or(Metric2d::GradientCorrelation),
                optimizer: Optimizer2d::from_key(&prefs.optimizer).unwrap_or_default(),
                ..SetupOptions::default()
            },
            clip_on: false,
            clip_half: 100.0,
            iso: None,
            status: None,
            current: None,
            drrs: Vec::new(),
            step_mm: 1.0,
            step_deg: 0.5,
            show_iso: true,
            show_markers: true,
            outlined: Vec::new(),
            outlines: Vec::new(),
            outlines_for: None,
            rooms: None,
            clip_structure: None,
            clip_margin: 10.0,
            show_clip: false,
            clip_shown: None,
            tolerance: [
                prefs.tolerance_tenths[0] as f64 / 10.0,
                prefs.tolerance_tenths[1] as f64 / 10.0,
            ],
            tool: PaneTool::None,
            drawing: None,
            pick_marker: None,
            marker_names: None,
            room: prefs.room.clone(),
        }
    }

    /// What the section's choices make of the remembered ones.
    fn prefs(&self) -> Reg2d3dPrefs {
        let tenths = |v: f64| (v * 10.0).round().clamp(1.0, 1000.0) as u16;
        Reg2d3dPrefs {
            metric: self.opts.metric.key().into(),
            optimizer: self.opts.optimizer.key().into(),
            tolerance_tenths: [tenths(self.tolerance[0]), tenths(self.tolerance[1])],
            room: self.room.clone(),
        }
    }

    /// The choices the run is made of.
    fn choices(&self) -> Vec<ViewChoice> {
        self.views
            .iter()
            .enumerate()
            .filter(|(_, v)| v.on)
            .map(|(i, v)| ViewChoice {
                region: v.region.clone(),
                picks: v.picks.clone(),
                ..ViewChoice::new(i, v.source.clone())
            })
            .collect()
    }

    fn options(&self) -> SetupOptions {
        SetupOptions {
            clip_half_mm: self.clip_on.then_some(self.clip_half),
            clip_structure: self.clip_structure.clone().filter(|_| self.clip_on),
            clip_margin_mm: self.clip_margin,
            isocenter: self.iso,
            ..self.opts.clone()
        }
    }
}

/// What a background job of the section brings back: one message per job,
/// so the size of the larger variants costs nothing.
#[allow(clippy::large_enum_variant)]
pub(super) enum Reg2d3dDone {
    /// A registration.
    Run(SetupResult),
    /// The DRRs at the plan.
    Preview(Vec<crate::extras::PlanarImage>),
    /// The DRRs at a pose moved by hand, for setup `idx`.
    Nudged {
        idx: usize,
        pose: registration::RigidTransform,
        drrs: Vec<crate::registration::metric2d::Image2d>,
    },
    /// The study as the patient was found, for workspace `target`, with the
    /// transform from it onto the CT.
    Copy {
        target: usize,
        study: Box<LoadedStudy>,
        transform: registration::Transform3,
    },
}

/// A finished job, with the workspace it was for.
pub(super) struct Reg2d3dJobOut {
    pub slot: usize,
    pub dims: [usize; 3],
    pub for_uid: String,
    pub result: anyhow::Result<Reg2d3dDone>,
}

impl ViewerApp {
    /// Make the state, or keep the views in step with the workspace's
    /// planar images.
    fn reg2d3d_follow(&mut self, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let prefs = &self.r2d_prefs;
        let st = self
            .r2d
            .get_or_insert_with(|| Reg2d3dState::new(slot, prefs));
        if st.slot != slot && self.r2d_job.is_none() {
            *st = Reg2d3dState::new(slot, prefs);
        }
        if st.rooms.is_none() {
            st.rooms = Some(crate::rooms::load_default());
        }
        // An image that does not place itself takes the views of the room
        // last chosen, in order, when there is one.
        let room_views: Vec<String> = st
            .rooms
            .as_ref()
            .and_then(|(f, _)| f.rooms.iter().find(|r| r.name == st.room))
            .map(|r| r.views.iter().map(|v| v.name.clone()).collect())
            .unwrap_or_default();
        let n = study.planar_images.len();
        if st.views.len() != n {
            st.views.truncate(n);
            for img in &study.planar_images[st.views.len()..] {
                let placed = img.camera.is_some();
                let unplaced = st
                    .views
                    .iter()
                    .filter(|v| matches!(v.source, GeometrySource::Room { .. }))
                    .count();
                let source = if placed {
                    GeometrySource::Image
                } else if let Some(view) = room_views.get(unplaced) {
                    GeometrySource::Room {
                        room: st.room.clone(),
                        view: view.clone(),
                    }
                } else {
                    GeometrySource::Typed(Geometry::default())
                };
                st.views.push(ViewPick {
                    // The radiographs that place themselves are ticked; a
                    // DRR made here is not a radiograph.
                    on: placed && !img.label.starts_with("DRR"),
                    source,
                    region: Vec::new(),
                    picks: Vec::new(),
                });
            }
        }
        if st.opts.patient.is_none() {
            st.opts.patient = Some(setup::default_patient(study));
        }
        if st.current.is_some_and(|c| c >= study.setups.len()) {
            st.current = None;
        }
    }

    /// Whether the section has anything to work on: a workspace with a CT
    /// and planar images, or a setup to show.
    pub(super) fn reg2d3d_relevant(&self) -> bool {
        self.r2d_job.is_some()
            || self.open_slots().into_iter().any(|s| {
                self.slots[s].study.as_ref().is_some_and(|st| {
                    st.has_volume() && (!st.planar_images.is_empty() || !st.setups.is_empty())
                })
            })
    }

    /// The workspaces the section can work on.
    fn reg2d3d_slots(&self) -> [bool; MAX_WORKSPACES] {
        std::array::from_fn(|s| {
            self.slots[s]
                .study
                .as_ref()
                .is_some_and(|st| st.has_volume() && !st.planar_images.is_empty())
        })
    }

    fn start_reg2d3d(&mut self, what: &'static str) {
        if self.r2d_job.is_some() {
            return;
        }
        let Some(st) = &self.r2d else { return };
        let slot = st.slot;
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let views = st.choices();
        if views.is_empty() {
            self.error = Some("Tick at least one radiograph".into());
            return;
        }
        let opts = st.options();
        let rooms = st.rooms.as_ref().map(|(f, _)| f.clone());
        let study = study.clone();
        let (dims, for_uid) = (
            study.volume.dims,
            study.volume.frame_of_reference_uid.clone(),
        );
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.r2d_job = Some(Job::spawn(progress, move |p| {
            let result = match what {
                "preview" => setup::preview(&study, &views, &opts, rooms.as_ref(), p)
                    .map(Reg2d3dDone::Preview),
                _ => setup::run(&study, &views, &opts, rooms.as_ref(), p).map(Reg2d3dDone::Run),
            };
            Reg2d3dJobOut {
                slot,
                dims,
                for_uid,
                result,
            }
        }));
    }

    /// Move the current setup by `delta` (rotations in degrees, then mm)
    /// and draw its DRRs again.
    fn nudge_reg2d3d(&mut self, delta: [f64; 6]) {
        if self.r2d_job.is_some() {
            return;
        }
        let Some(st) = &self.r2d else { return };
        let (slot, Some(idx)) = (st.slot, st.current) else {
            return;
        };
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let Some(s) = study.setups.get(idx) else {
            return;
        };
        let t = &s.result.transform;
        let mut p = t.recentered(s.isocenter).params();
        for k in 0..3 {
            p[k] += delta[k].to_radians();
            p[k + 3] += delta[k + 3];
        }
        let pose = registration::RigidTransform::new(p, s.isocenter);
        let (study, s) = (study.clone(), s.clone());
        let (dims, for_uid) = (
            study.volume.dims,
            study.volume.frame_of_reference_uid.clone(),
        );
        let progress = Arc::new(Progress::default());
        self.r2d_job =
            Some(Job::spawn(progress, move |p| {
                let result = setup::render_at(&study, &s, &pose, p)
                    .map(|drrs| Reg2d3dDone::Nudged { idx, pose, drrs });
                Reg2d3dJobOut {
                    slot,
                    dims,
                    for_uid,
                    result,
                }
            }));
    }

    /// The positioned copy of the current setup into the next free
    /// workspace.
    fn reg2d3d_copy(&mut self) {
        if self.r2d_job.is_some() {
            return;
        }
        let Some(st) = &self.r2d else { return };
        let (slot, Some(idx)) = (st.slot, st.current) else {
            return;
        };
        let Some(target) = self
            .copy_targets(slot)
            .into_iter()
            .find(|t| self.slots[*t].study.is_none())
        else {
            self.error =
                Some("Every workspace is in use: close one for the positioned copy".into());
            return;
        };
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let Some(s) = study.setups.get(idx) else {
            return;
        };
        let (study, s) = (study.clone(), s.clone());
        let (dims, for_uid) = (
            study.volume.dims,
            study.volume.frame_of_reference_uid.clone(),
        );
        let progress = Arc::new(Progress::default());
        progress.set("moving the study");
        self.r2d_job = Some(Job::spawn(progress, move |p| {
            let copy = setup::positioned_copy(&study, &s, p);
            Reg2d3dJobOut {
                slot,
                dims,
                for_uid,
                result: Ok(Reg2d3dDone::Copy {
                    target,
                    study: Box::new(copy),
                    transform: registration::Transform3::rigid_only(s.result.transform.clone()),
                }),
            }
        }));
    }

    /// A job landed.
    pub(super) fn on_reg2d3d_done(&mut self, out: Reg2d3dJobOut) {
        let done = match out.result {
            Ok(d) => d,
            Err(e) => {
                if !progress::is_cancellation(&e) {
                    self.error = Some(format!("2D/3D registration failed: {e:#}"));
                }
                return;
            }
        };
        if !self.slot_still_shows(out.slot, out.dims, &out.for_uid) {
            self.error = Some(stale_result(&REG2D3D_TOOL));
            return;
        }
        let slot = out.slot;
        match done {
            Reg2d3dDone::Run(s) => {
                let line = s.result.correction_line();
                let Some(study) = self.slots[slot].study.as_mut() else {
                    return;
                };
                study.setups.push(s);
                let idx = study.setups.len() - 1;
                if let Some(st) = &mut self.r2d {
                    st.status = Some(format!("✔ {line}"));
                }
                self.show_setup(slot, idx);
            }
            Reg2d3dDone::Preview(imgs) => {
                let n = imgs.len();
                let Some(study) = self.slots[slot].study.as_mut() else {
                    return;
                };
                let first = study.planar_images.len();
                study.planar_images.extend(imgs);
                // Each radiograph with its DRR at the plan over it.
                let views: Vec<usize> = self
                    .r2d
                    .as_ref()
                    .map(|st| st.choices().iter().map(|v| v.image).collect())
                    .unwrap_or_default();
                for (k, img) in views.iter().take(settings::PLANAR_PANES).enumerate() {
                    self.show_planar(slot, *img, Some(k));
                    self.stack_planar(slot, first + k, Some(k));
                }
                if let Some(st) = &mut self.r2d {
                    st.status = Some(format!(
                        "✔ {n} {} at the plan filed under Planar images",
                        if n == 1 { "DRR" } else { "DRRs" }
                    ));
                }
                self.shown[slot] = true;
                self.settings_gen += 1;
            }
            Reg2d3dDone::Nudged { idx, pose, drrs } => {
                let Some(study) = self.slots[slot].study.as_mut() else {
                    return;
                };
                let Some(s) = study.setups.get_mut(idx) else {
                    return;
                };
                setup::adjust(s, pose, drrs);
                let line = s.result.correction_line();
                self.refresh_setup_drrs(slot, idx);
                if let Some(st) = &mut self.r2d {
                    st.status = Some(format!("✔ moved by hand: {line}"));
                }
            }
            Reg2d3dDone::Copy {
                target,
                study,
                transform,
            } => {
                self.on_study_loaded(target, *study);
                self.show_workspace(target);
                self.apply_external_transform_between(transform, RegMethod::Given, target, slot);
                if let Some(st) = &mut self.r2d {
                    st.status = Some(format!(
                        "✔ positioned copy in workspace {}, registered to {}",
                        SLOT_NAMES[target], SLOT_NAMES[slot]
                    ));
                }
            }
        }
    }

    /// Put setup `idx` of `slot` on screen: its DRRs filed (or refreshed)
    /// and each radiograph stacked with its DRR in a planar pane.
    pub(super) fn show_setup(&mut self, slot: usize, idx: usize) {
        self.reg2d3d_follow(slot);
        if let Some(st) = &mut self.r2d {
            st.slot = slot;
            st.current = Some(idx);
            st.drrs.clear();
            st.outlines_for = None;
        }
        self.refresh_setup_drrs(slot, idx);
        let (images, drrs) = match (&self.slots[slot].study, &self.r2d) {
            (Some(study), Some(st)) => (
                study
                    .setups
                    .get(idx)
                    .map(|s| s.images.clone())
                    .unwrap_or_default(),
                st.drrs.clone(),
            ),
            _ => return,
        };
        let n = self.planar_count(slot);
        for (k, (img, drr)) in images
            .iter()
            .zip(&drrs)
            .take(settings::PLANAR_PANES)
            .enumerate()
        {
            if *img < n {
                self.show_planar(slot, *img, Some(k));
                self.stack_planar(slot, *drr, Some(k));
            } else {
                self.show_planar(slot, *drr, Some(k));
            }
        }
        self.shown[slot] = true;
    }

    /// File the DRRs of setup `idx` at its pose as planar images, or put
    /// the new pixels into the ones filed for it already.
    fn refresh_setup_drrs(&mut self, slot: usize, idx: usize) {
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        let Some(s) = study.setups.get(idx) else {
            return;
        };
        let made: Vec<crate::extras::PlanarImage> = (0..s.result.views.len())
            .filter_map(|k| {
                let radiograph = s.images.get(k).and_then(|i| study.planar_images.get(*i));
                s.drr_planar(k, radiograph)
            })
            .collect();
        let Some(st) = &mut self.r2d else { return };
        let fresh =
            st.drrs.len() != made.len() || st.drrs.iter().any(|d| *d >= study.planar_images.len());
        if fresh {
            st.drrs.clear();
            for img in made {
                study.planar_images.push(img);
                st.drrs.push(study.planar_images.len() - 1);
            }
            self.settings_gen += 1;
        } else {
            for (d, img) in st.drrs.iter().zip(made) {
                // A new allocation: the panes key their textures on it.
                let label = study.planar_images[*d].label.clone();
                study.planar_images[*d] = crate::extras::PlanarImage { label, ..img };
            }
        }
        st.outlines_for = None;
    }

    /// The overlay of the current setup over a planar image of `slot`, when
    /// the image is one of its radiographs or DRRs: the isocentre, the
    /// markers (observed ○, predicted ×) and the outlines.
    pub(super) fn reg2d3d_overlay(
        &self,
        painter: &egui::Painter,
        slot: usize,
        image: usize,
        on: Rect,
    ) {
        let Some(st) = &self.r2d else { return };
        if st.slot != slot {
            return;
        }
        let (Some(idx), Some(study)) = (st.current, &self.slots[slot].study) else {
            return;
        };
        let Some(s) = study.setups.get(idx) else {
            return;
        };
        let Some(k) = s
            .images
            .iter()
            .position(|i| *i == image)
            .or_else(|| st.drrs.iter().position(|d| *d == image))
        else {
            return;
        };
        let Some(view) = s.result.views.get(k) else {
            return;
        };
        let at = |u: f32, v: f32| Pos2::new(on.left() + u * on.width(), on.top() + v * on.height());
        let [w, h] = view.camera.dims;
        let norm = |p: [f64; 2]| {
            (
                (p[0] as f32 + 0.5) / w as f32,
                (p[1] as f32 + 0.5) / h as f32,
            )
        };
        // The planned aperture of a portal image.
        for f in s.fields.iter().filter(|f| f.view == k) {
            let stroke = egui::Stroke::new(1.5, Color32::from_rgb(0, 200, 230));
            for (a, b) in &f.outline {
                painter.line_segment([at(a[0], a[1]), at(b[0], b[1])], stroke);
            }
        }
        // The outlines.
        for o in &st.outlines {
            if let Some(segs) = o.segments.get(k) {
                for (a, b) in segs {
                    painter.line_segment(
                        [at(a[0], a[1]), at(b[0], b[1])],
                        egui::Stroke::new(1.5, o.color),
                    );
                }
            }
        }
        // The isocentre, where the plan puts it in the room.
        if st.show_iso {
            if let Some(p) = view.camera.project(s.isocenter) {
                let (u, v) = norm(p);
                let c = at(u, v);
                let r = 8.0;
                let stroke = egui::Stroke::new(1.5, Color32::from_rgb(255, 210, 0));
                painter.line_segment([c - Vec2::new(r, 0.0), c + Vec2::new(r, 0.0)], stroke);
                painter.line_segment([c - Vec2::new(0.0, r), c + Vec2::new(0.0, r)], stroke);
            }
        }
        // The markers, in the pixels of the image as given.
        if st.show_markers {
            if let Some(m) = &s.result.markers {
                let full = s
                    .images
                    .get(k)
                    .and_then(|i| study.planar_images.get(*i))
                    .map(|p| (p.cols as f32, p.rows as f32))
                    .unwrap_or((w as f32, h as f32));
                let fnorm =
                    |p: [f64; 2]| at((p[0] as f32 + 0.5) / full.0, (p[1] as f32 + 0.5) / full.1);
                for r in m.residuals.iter().filter(|r| r.view == view.label) {
                    let (o, q) = (fnorm(r.observed), fnorm(r.predicted));
                    let moved = m
                        .triangulated
                        .iter()
                        .any(|t| t.migrated && t.marker == r.marker);
                    let col = if moved {
                        Color32::from_rgb(255, 80, 80)
                    } else {
                        Color32::from_rgb(80, 220, 120)
                    };
                    painter.circle_stroke(o, 5.0, egui::Stroke::new(1.5, col));
                    let d = 4.0;
                    let x = egui::Stroke::new(1.5, Color32::from_rgb(90, 170, 255));
                    painter.line_segment([q - Vec2::splat(d), q + Vec2::splat(d)], x);
                    painter.line_segment([q + Vec2::new(-d, d), q + Vec2::new(d, -d)], x);
                    painter.line_segment([o, q], egui::Stroke::new(1.0, col));
                }
            }
        }
    }

    /// The radiograph of a planar pane's stack the regions and picks
    /// belong to: the first of `images` that is one of the section's.
    pub(super) fn reg2d3d_pane_image(&self, slot: usize, images: &[usize]) -> Option<usize> {
        let st = self.r2d.as_ref().filter(|st| st.slot == slot)?;
        images.iter().copied().find(|&i| {
            st.views
                .get(i)
                .is_some_and(|v| v.on || !v.region.is_empty() || !v.picks.is_empty())
        })
    }

    /// Whether the left button on planar image `image` of `slot` works a
    /// tool of the section rather than moving the image.
    pub(super) fn reg2d3d_pane_tool(&self, slot: usize, image: usize) -> bool {
        self.r2d.as_ref().is_some_and(|st| {
            st.slot == slot
                && image < st.views.len()
                && match st.tool {
                    PaneTool::None => false,
                    PaneTool::Pick => st.pick_marker.is_some(),
                    _ => true,
                }
        })
    }

    /// What the pointer does on planar image `image` of `slot`, drawn in
    /// `on`, with the section's tool.
    pub(super) fn reg2d3d_pane_input(
        &self,
        slot: usize,
        image: usize,
        on: Rect,
        resp: &egui::Response,
    ) -> Option<PaneEdit> {
        if !self.reg2d3d_pane_tool(slot, image) {
            return None;
        }
        let st = self.r2d.as_ref()?;
        let img = self.slots[slot].study.as_ref()?.planar_images.get(image)?;
        let to_px = |p: Pos2| {
            [
                ((p.x - on.left()) / on.width() * img.cols as f32 - 0.5) as f64,
                ((p.y - on.top()) / on.height() * img.rows as f32 - 0.5) as f64,
            ]
        };
        let here = resp
            .interact_pointer_pos()
            .or_else(|| resp.hover_pos())
            .map(to_px);
        let primary = egui::PointerButton::Primary;
        match st.tool {
            PaneTool::Rectangle => {
                if resp.drag_started_by(primary) {
                    // From where the button went down, not where the drag
                    // was first noticed.
                    let p = resp
                        .ctx
                        .input(|i| i.pointer.press_origin())
                        .map(to_px)
                        .or(here)?;
                    return Some(PaneEdit::Rect {
                        image,
                        a: p,
                        b: p,
                        done: false,
                    });
                }
                let (a, last) = match &st.drawing {
                    Some((i, pts)) if *i == image && pts.len() == 2 => (pts[0], pts[1]),
                    _ => return None,
                };
                if resp.drag_stopped_by(primary) {
                    Some(PaneEdit::Rect {
                        image,
                        a,
                        b: here.unwrap_or(last),
                        done: true,
                    })
                } else if resp.dragged_by(primary) {
                    Some(PaneEdit::Rect {
                        image,
                        a,
                        b: here?,
                        done: false,
                    })
                } else {
                    None
                }
            }
            PaneTool::Polygon => {
                if resp.double_clicked() {
                    Some(PaneEdit::Close { image })
                } else if resp.clicked() {
                    Some(PaneEdit::Vertex { image, p: here? })
                } else {
                    None
                }
            }
            PaneTool::Pick => resp
                .clicked()
                .then_some(())
                .and(here)
                .map(|p| PaneEdit::Pick { image, p }),
            PaneTool::None => None,
        }
    }

    /// Apply what a planar pane asked.
    pub(super) fn reg2d3d_pane_apply(&mut self, e: PaneEdit) {
        let Some(st) = &mut self.r2d else { return };
        match e {
            PaneEdit::Rect { image, a, b, done } => {
                if !done {
                    st.drawing = Some((image, vec![a, b]));
                    return;
                }
                st.drawing = None;
                let (x0, x1) = (a[0].min(b[0]), a[0].max(b[0]));
                let (y0, y1) = (a[1].min(b[1]), a[1].max(b[1]));
                if x1 - x0 >= 2.0 && y1 - y0 >= 2.0 {
                    if let Some(v) = st.views.get_mut(image) {
                        v.region.push(vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
                    }
                }
            }
            PaneEdit::Vertex { image, p } => match &mut st.drawing {
                Some((i, pts)) if *i == image => pts.push(p),
                _ => st.drawing = Some((image, vec![p])),
            },
            PaneEdit::Close { image } => {
                if let Some((i, mut pts)) = st.drawing.take() {
                    pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 0.5 && (a[1] - b[1]).abs() < 0.5);
                    if i == image && pts.len() >= 3 {
                        if let Some(v) = st.views.get_mut(image) {
                            v.region.push(pts);
                        }
                    }
                }
            }
            PaneEdit::Pick { image, p } => {
                if let (Some(name), Some(v)) = (st.pick_marker.clone(), st.views.get_mut(image)) {
                    v.picks.retain(|(n, _)| *n != name);
                    v.picks.push((name, p));
                }
            }
        }
    }

    /// The regions, the one being drawn and the markers placed by hand,
    /// over planar image `image` of `slot` drawn in `on`.
    pub(super) fn reg2d3d_marks(
        &self,
        painter: &egui::Painter,
        slot: usize,
        image: usize,
        on: Rect,
    ) {
        let Some(st) = self.r2d.as_ref().filter(|st| st.slot == slot) else {
            return;
        };
        let (Some(v), Some(img)) = (
            st.views.get(image),
            self.slots[slot]
                .study
                .as_ref()
                .and_then(|s| s.planar_images.get(image)),
        ) else {
            return;
        };
        let at = |p: [f64; 2]| {
            Pos2::new(
                on.left() + (p[0] as f32 + 0.5) / img.cols as f32 * on.width(),
                on.top() + (p[1] as f32 + 0.5) / img.rows as f32 * on.height(),
            )
        };
        let col = Color32::from_rgb(255, 170, 60);
        let stroke = egui::Stroke::new(1.5, col);
        for poly in &v.region {
            painter.add(egui::Shape::closed_line(
                poly.iter().map(|p| at(*p)).collect(),
                stroke,
            ));
        }
        if let Some((_, pts)) = st.drawing.as_ref().filter(|(i, _)| *i == image) {
            if st.tool == PaneTool::Rectangle && pts.len() == 2 {
                painter.rect_stroke(
                    Rect::from_two_pos(at(pts[0]), at(pts[1])),
                    0.0,
                    stroke,
                    egui::StrokeKind::Middle,
                );
            } else {
                let scr: Vec<Pos2> = pts.iter().map(|p| at(*p)).collect();
                for p in &scr {
                    painter.circle_filled(*p, 2.5, col);
                }
                painter.add(egui::Shape::line(scr, stroke));
            }
        }
        let pick = Color32::from_rgb(255, 120, 200);
        for (name, p) in &v.picks {
            let c = at(*p);
            let r = 6.0;
            let s = egui::Stroke::new(1.5, pick);
            painter.line_segment([c - Vec2::new(r, 0.0), c + Vec2::new(r, 0.0)], s);
            painter.line_segment([c - Vec2::new(0.0, r), c + Vec2::new(0.0, r)], s);
            painter.text(
                c + Vec2::new(5.0, -5.0),
                egui::Align2::LEFT_BOTTOM,
                name,
                egui::FontId::proportional(11.0),
                pick,
            );
        }
    }

    /// The clip box to draw in the views of `slot`, when the section shows
    /// it.
    pub(super) fn reg2d3d_clip_shown(&self, slot: usize) -> Option<ClipBox> {
        let st = self.r2d.as_ref().filter(|st| st.slot == slot)?;
        if !(st.clip_on && st.show_clip) {
            return None;
        }
        st.clip_shown.as_ref().and_then(|(_, b)| *b)
    }

    /// Make the outlines again when the setup, its pose or the structures
    /// changed.
    fn reg2d3d_outlines(&mut self) {
        let Some(st) = &self.r2d else { return };
        let (slot, Some(idx)) = (st.slot, st.current) else {
            return;
        };
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let Some(s) = study.setups.get(idx) else {
            return;
        };
        let key = (
            idx,
            s.result.transform.params().map(f64::to_bits),
            st.outlined.clone(),
        );
        if st.outlines_for.as_ref() == Some(&key) {
            return;
        }
        let grid = study.volume.grid();
        let all = crate::workflow::select::list(study);
        let mut outlines = Vec::new();
        for name in &st.outlined {
            let Some(e) = all.iter().rev().find(|e| &e.name == name) else {
                continue;
            };
            let Some(mask) =
                crate::workflow::select::structure(study, e).and_then(|x| x.mask_on(&grid).ok())
            else {
                continue;
            };
            let segments = s
                .result
                .views
                .iter()
                .map(|v| setup::silhouette(&study.volume, &mask, &v.camera, &s.result.transform))
                .collect();
            outlines.push(Outline {
                color: Color32::from_rgb(e.color[0], e.color[1], e.color[2]),
                segments,
            });
        }
        if let Some(st) = &mut self.r2d {
            st.outlines = outlines;
            st.outlines_for = Some(key);
        }
    }

    /// Save the current setup's report as JSON.
    fn reg2d3d_save_report(&mut self) {
        let Some((slot, idx)) = self
            .r2d
            .as_ref()
            .and_then(|st| st.current.map(|c| (st.slot, c)))
        else {
            return;
        };
        let Some(s) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.setups.get(idx))
        else {
            return;
        };
        let text = serde_json::to_string_pretty(&s.json()).unwrap_or_default();
        const JSON: pick::Filter = pick::Filter {
            name: "JSON",
            exts: &["json"],
        };
        self.ask_save(
            "Save the setup report",
            "setup_report.json",
            None,
            Some(JSON),
            move |app, path| match std::fs::write(&path, text) {
                Ok(()) => {
                    if let Some(st) = &mut app.r2d {
                        st.status = Some(format!("✔ report written to {}", path.display()));
                    }
                }
                Err(e) => app.error = Some(format!("Could not write {}: {e}", path.display())),
            },
        );
    }

    /// Write the current setup's registration object, DRRs and report into
    /// a folder.
    fn reg2d3d_save_dicom(&mut self) {
        let Some((slot, idx)) = self
            .r2d
            .as_ref()
            .and_then(|st| st.current.map(|c| (st.slot, c)))
        else {
            return;
        };
        if self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.setups.get(idx))
            .is_none()
        {
            return;
        }
        self.ask_folder("Write the setup as DICOM into", move |app, dir| {
            let Some(study) = app.slots[slot].study.as_ref() else {
                return;
            };
            let Some(s) = study.setups.get(idx) else {
                return;
            };
            match setup::write_files(study, s, &dir) {
                Ok(files) => {
                    if let Some(st) = &mut app.r2d {
                        st.status = Some(format!(
                            "✔ {} files written to {}",
                            files.len(),
                            dir.display()
                        ));
                    }
                }
                Err(e) => app.error = Some(format!("Writing the setup failed: {e:#}")),
            }
        });
    }

    /// The section, inside the Image registration module.
    pub(super) fn reg2d3d_section(&mut self, ui: &mut egui::Ui) {
        let busy = self.r2d_job.is_some();
        let slots = self.reg2d3d_slots();
        let slot = match &self.r2d {
            Some(st)
                if busy
                    || slots[st.slot]
                    || self.slots[st.slot]
                        .study
                        .as_ref()
                        .is_some_and(|s| !s.setups.is_empty()) =>
            {
                st.slot
            }
            _ => (0..MAX_WORKSPACES).find(|s| slots[*s]).unwrap_or(0),
        };
        self.reg2d3d_follow(slot);
        if let Some(st) = &mut self.r2d {
            if st.rooms.is_none() {
                st.rooms = Some(crate::rooms::load_default());
            }
        }
        self.reg2d3d_outlines();
        let Some(mut st) = self.r2d.take() else {
            ui.weak("Load a CT with radiographs");
            return;
        };
        let Some(study) = self.slots[st.slot].study.as_ref() else {
            self.r2d = Some(st);
            return;
        };

        // Everything the rows below show of the study, taken before the
        // closures borrow the state.
        let planar: Vec<(String, bool, Option<String>)> = study
            .planar_images
            .iter()
            .map(|p| {
                (
                    p.label.clone(),
                    p.camera.is_some(),
                    p.geometry_missing.clone(),
                )
            })
            .collect();
        let beams: Vec<(usize, usize, String)> = study
            .plans
            .iter()
            .enumerate()
            .flat_map(|(pi, plan)| {
                plan.beams
                    .iter()
                    .enumerate()
                    .map(move |(bi, b)| (pi, bi, format!("{} · {}", plan.label, b.name)))
            })
            .collect();
        let room_views: Vec<(String, String)> = st
            .rooms
            .as_ref()
            .map(|(f, _)| {
                f.rooms
                    .iter()
                    .flat_map(|r| {
                        r.views
                            .iter()
                            .map(move |v| (r.name.clone(), v.name.clone()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let structures: Vec<String> = {
            let mut v: Vec<String> = crate::workflow::select::list(study)
                .into_iter()
                .map(|e| e.name)
                .collect();
            v.dedup();
            v
        };
        let default_iso = setup::default_isocenter(study, &st.choices());
        let crosshair = {
            let c = self.slots[st.slot].cursor;
            study.volume.voxel_to_patient(c[0], c[1], c[2])
        };
        // The markers a run would use, listed again only when their source
        // changes: finding them in the CT takes a moment.
        let marker_key = format!("{:?}", st.opts.markers);
        if st
            .marker_names
            .as_ref()
            .is_none_or(|(k, _)| *k != marker_key)
        {
            let names = setup::study_markers(study, &st.options(), None)
                .into_iter()
                .map(|m| m.name)
                .collect();
            st.marker_names = Some((marker_key, names));
        }
        let marker_names: Vec<String> = st
            .marker_names
            .as_ref()
            .map(|(_, n)| n.clone())
            .unwrap_or_default();
        let n_markers = marker_names.len();
        // The clip box as the views draw it, worked out again when what
        // makes it changes.
        let iso_now = st.iso.unwrap_or(default_iso);
        let clip_key = format!(
            "{} {:?} {} {} {:.2} {:.2} {:.2}",
            st.clip_on,
            st.clip_structure,
            st.clip_half,
            st.clip_margin,
            iso_now.x,
            iso_now.y,
            iso_now.z
        );
        if st.clip_on && st.clip_shown.as_ref().is_none_or(|(k, _)| *k != clip_key) {
            let b = match &st.clip_structure {
                Some(n) => setup::structure_box(study, n, st.clip_margin).ok(),
                None => Some(ClipBox::around(iso_now, st.clip_half)),
            };
            st.clip_shown = Some((clip_key, b));
        }
        let current = st.current.and_then(|i| study.setups.get(i)).cloned();
        let job_line = self
            .r2d_job
            .as_ref()
            .map(|j| (j.progress.get(), j.progress.frac()));

        let mut new_slot = None;
        let mut act_run = false;
        let mut act_preview = false;
        let mut act_cancel = false;
        let mut nudge: Option<[f64; 6]> = None;
        let mut act_copy = false;
        let mut act_report = false;
        let mut act_dicom = false;
        let mut act_clip = false;
        let mut chosen_room: Option<String> = None;
        let id = ui.make_persistent_id("reg2d3d");

        new_slot = workspace_row(ui, st.slot, slots, !busy).or(new_slot);
        egui::CollapsingHeader::new("Images")
            .id_salt((id, "images"))
            .default_open(true)
            .show(ui, |ui| {
                if planar.is_empty() {
                    ui.weak("No planar images in this workspace");
                }
                for (i, (label, placed, missing)) in planar.iter().enumerate() {
                    let Some(v) = st.views.get_mut(i) else {
                        continue;
                    };
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut v.on, label.as_str());
                        let text = match &v.source {
                            GeometrySource::Image => "RT image".to_string(),
                            GeometrySource::Beam { plan, beam, .. } => beams
                                .iter()
                                .find(|(p, b, _)| p == plan && b == beam)
                                .map(|(_, _, l)| l.clone())
                                .unwrap_or_else(|| "beam".into()),
                            GeometrySource::Room { view, .. } => format!("room: {view}"),
                            GeometrySource::Typed(_) => "typed".into(),
                        };
                        egui::ComboBox::from_id_salt((id, "src", i))
                            .selected_text(text)
                            .width(110.0)
                            .show_ui(ui, |ui| {
                                if *placed
                                    && ui
                                        .selectable_label(
                                            v.source == GeometrySource::Image,
                                            "RT image",
                                        )
                                        .clicked()
                                {
                                    v.source = GeometrySource::Image;
                                }
                                for (p, b, l) in &beams {
                                    if ui.selectable_label(false, l.as_str()).clicked() {
                                        v.source = GeometrySource::Beam {
                                            plan: *p,
                                            beam: *b,
                                            base: Geometry::default(),
                                        };
                                    }
                                }
                                for (r, w) in &room_views {
                                    if ui.selectable_label(false, format!("room: {w}")).clicked() {
                                        v.source = GeometrySource::Room {
                                            room: r.clone(),
                                            view: w.clone(),
                                        };
                                        chosen_room = Some(r.clone());
                                    }
                                }
                                if ui
                                    .selectable_label(
                                        matches!(v.source, GeometrySource::Typed(_)),
                                        "typed",
                                    )
                                    .clicked()
                                {
                                    v.source = GeometrySource::Typed(Geometry::default());
                                }
                            });
                        if v.source == GeometrySource::Image && !placed {
                            ui.colored_label(
                                ui.visuals().warn_fg_color,
                                format!("missing: {}", missing.as_deref().unwrap_or("geometry")),
                            );
                        }
                    });
                    let typed = matches!(v.source, GeometrySource::Typed(_));
                    match &mut v.source {
                        GeometrySource::Typed(g) | GeometrySource::Beam { base: g, .. } => {
                            ui.horizontal(|ui| {
                                ui.add_space(24.0);
                                if typed {
                                    ui.label("G");
                                    ui.add(
                                        egui::DragValue::new(&mut g.gantry_deg)
                                            .range(-360.0..=360.0)
                                            .suffix("°"),
                                    );
                                    ui.label("C");
                                    ui.add(
                                        egui::DragValue::new(&mut g.couch_deg)
                                            .range(-180.0..=180.0)
                                            .suffix("°"),
                                    );
                                }
                                ui.label("SAD");
                                ui.add(
                                    egui::DragValue::new(&mut g.sad)
                                        .speed(5.0)
                                        .range(100.0..=3000.0),
                                );
                                ui.label("SID");
                                ui.add(
                                    egui::DragValue::new(&mut g.sid)
                                        .speed(5.0)
                                        .range(100.0..=4000.0),
                                );
                            });
                        }
                        _ => {}
                    }
                }
                if let Some((_, Some(e))) = &st.rooms {
                    ui.colored_label(ui.visuals().warn_fg_color, "imaging-rooms.json")
                        .on_hover_text(e.as_str());
                }
            });

        egui::CollapsingHeader::new("Setup")
            .id_salt((id, "setup"))
            .default_open(true)
            .show(ui, |ui| {
                form::form(ui, "reg2d3d_setup", |f| {
                    f.row_tip(
                        "Isocentre, mm",
                        "x, y, z in patient coordinates (LPS)",
                        |ui| {
                            let mut p = st.iso.unwrap_or(default_iso);
                            let mut changed = false;
                            for v in [&mut p.x, &mut p.y, &mut p.z] {
                                changed |= ui
                                    .add(egui::DragValue::new(v).speed(1.0).max_decimals(1))
                                    .changed();
                            }
                            if changed {
                                st.iso = Some(p);
                            }
                            if small_tip_button(ui, "⌖", "Take the isocentre from the crosshair")
                            {
                                st.iso = Some(crosshair);
                            }
                            if st.iso.is_some()
                                && small_tip_button(ui, "↺", "The images' or the plan's isocentre")
                            {
                                st.iso = None;
                            }
                        },
                    );
                    f.row("Patient", |ui| {
                        let mut frame = st.opts.patient.unwrap_or_default();
                        egui::ComboBox::from_id_salt((id, "patient"))
                            .selected_text(frame.describe())
                            .width(120.0)
                            .show_ui(ui, |ui| {
                                for p in PatientPosition::ALL {
                                    if ui
                                        .selectable_label(
                                            frame == PatientFrame::Position(p),
                                            p.code(),
                                        )
                                        .clicked()
                                    {
                                        frame = PatientFrame::Position(p);
                                    }
                                }
                                let up = matches!(frame, PatientFrame::Upright { .. });
                                if ui.selectable_label(up, "Upright").clicked() && !up {
                                    frame = PatientFrame::Upright { facing_deg: 0.0 };
                                }
                            });
                        if let PatientFrame::Upright { facing_deg } = &mut frame {
                            ui.add(
                                egui::DragValue::new(facing_deg)
                                    .range(-360.0..=360.0)
                                    .suffix("°"),
                            )
                            .on_hover_text("The direction the patient faces, from the nozzle");
                        }
                        st.opts.patient = Some(frame);
                    });
                    f.row_tip("Box", "The region that matters: its shadow on each radiograph is what is compared", |ui| {
                        ui.checkbox(&mut st.clip_on, "");
                        ui.add_enabled_ui(st.clip_on, |ui| {
                            egui::ComboBox::from_id_salt((id, "clipof"))
                                .selected_text(st.clip_structure.as_deref().unwrap_or("Cube"))
                                .width(90.0)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(st.clip_structure.is_none(), "Cube")
                                        .clicked()
                                    {
                                        st.clip_structure = None;
                                    }
                                    for n in &structures {
                                        if ui
                                            .selectable_label(
                                                st.clip_structure.as_ref() == Some(n),
                                                n.as_str(),
                                            )
                                            .clicked()
                                        {
                                            st.clip_structure = Some(n.clone());
                                        }
                                    }
                                });
                            if st.clip_structure.is_none() {
                                ui.add(
                                    egui::DragValue::new(&mut st.clip_half)
                                        .range(10.0..=400.0)
                                        .suffix(" mm"),
                                )
                                .on_hover_text("Half its size, about the isocentre");
                            } else {
                                ui.add(
                                    egui::DragValue::new(&mut st.clip_margin)
                                        .range(0.0..=100.0)
                                        .prefix("+")
                                        .suffix(" mm"),
                                )
                                .on_hover_text("Around the structure");
                            }
                        });
                        if st.clip_on
                            && ui
                                .add(egui::Button::selectable(st.show_clip, "👁").small())
                                .on_hover_text("Show the box in the views")
                                .clicked()
                        {
                            st.show_clip = !st.show_clip;
                            act_clip = st.show_clip;
                        }
                        if st.clip_on
                            && ui
                                .add(egui::Button::selectable(st.opts.clip_rays, "Rays inside").small())
                                .on_hover_text("The DRRs see only what is inside the box")
                                .clicked()
                        {
                            st.opts.clip_rays = !st.opts.clip_rays;
                        }
                    });
                    f.row_tip(
                        "Region",
                        "Drawn on the radiographs in the planar panes: only what is inside \
                         is compared",
                        |ui| {
                            for (t, name, tip) in [
                                (PaneTool::Rectangle, "Rectangle", "Drag across a radiograph"),
                                (
                                    PaneTool::Polygon,
                                    "Polygon",
                                    "Click the corners on a radiograph; double-click closes",
                                ),
                            ] {
                                if ui
                                    .selectable_label(st.tool == t, name)
                                    .on_hover_text(tip)
                                    .clicked()
                                {
                                    st.tool = if st.tool == t { PaneTool::None } else { t };
                                    st.drawing = None;
                                }
                            }
                            let n: usize = st.views.iter().map(|v| v.region.len()).sum();
                            if n > 0 {
                                ui.weak(format!("{n}"));
                                if ui.small_button("Clear").clicked() {
                                    st.views.iter_mut().for_each(|v| v.region.clear());
                                    st.drawing = None;
                                }
                            }
                        },
                    );
                    f.row("Threshold", |ui| {
                        ui.add(
                            egui::DragValue::new(&mut st.opts.threshold_hu)
                                .speed(10.0)
                                .range(-1000.0..=2000.0)
                                .suffix(" HU"),
                        );
                        let bone = st.opts.threshold_hu >= 100.0;
                        if ui.selectable_label(bone, "Bone only").clicked() {
                            st.opts.threshold_hu = if bone { -800.0 } else { 150.0 };
                        }
                    });
                    f.row("Method", |ui| {
                        egui::ComboBox::from_id_salt((id, "method"))
                            .selected_text(st.opts.method.label())
                            .width(160.0)
                            .show_ui(ui, |ui| {
                                for m in SetupMethod::ALL {
                                    ui.selectable_value(&mut st.opts.method, m, m.label());
                                }
                            });
                    });
                    if st.opts.method != SetupMethod::Markers {
                        f.row("Measure", |ui| {
                            egui::ComboBox::from_id_salt((id, "metric"))
                                .selected_text(st.opts.metric.label())
                                .width(160.0)
                                .show_ui(ui, |ui| {
                                    for m in Metric2d::ALL {
                                        ui.selectable_value(&mut st.opts.metric, m, m.label());
                                    }
                                });
                        });
                        f.row("Search", |ui| {
                            egui::ComboBox::from_id_salt((id, "opt"))
                                .selected_text(st.opts.optimizer.label())
                                .width(100.0)
                                .show_ui(ui, |ui| {
                                    for o in Optimizer2d::ALL {
                                        ui.selectable_value(&mut st.opts.optimizer, o, o.label());
                                    }
                                });
                            ui.checkbox(&mut st.opts.refine, "Refine");
                        });
                        f.row("Start", |ui| {
                            egui::ComboBox::from_id_salt((id, "init"))
                                .selected_text(st.opts.init.label())
                                .width(120.0)
                                .show_ui(ui, |ui| {
                                    for i in Init2d3d::ALL {
                                        ui.selectable_value(&mut st.opts.init, i, i.label());
                                    }
                                });
                        });
                        f.row("Levels · pixels", |ui| {
                            ui.add(egui::DragValue::new(&mut st.opts.levels).range(1..=5));
                            ui.add(
                                egui::DragValue::new(&mut st.opts.max_pixels)
                                    .range(64..=1024)
                                    .suffix(" px"),
                            );
                        });
                        f.row("Compute", |ui| {
                            for p in DevicePref::ALL {
                                ui.radio_value(&mut st.opts.device, p, p.label());
                            }
                        });
                    }
                    if st.opts.method != SetupMethod::Intensity {
                        f.row("Markers", |ui| {
                            let detect = st.opts.markers == MarkerSource::Detect;
                            egui::ComboBox::from_id_salt((id, "markers"))
                                .selected_text(if detect {
                                    "Found in the CT"
                                } else {
                                    "MARKER points"
                                })
                                .width(130.0)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(!detect, "MARKER points").clicked() {
                                        st.opts.markers = MarkerSource::Pois;
                                    }
                                    if ui.selectable_label(detect, "Found in the CT").clicked() {
                                        st.opts.markers = MarkerSource::Detect;
                                    }
                                });
                            ui.weak(format!("{n_markers}"));
                        });
                        f.row_tip(
                            "Place",
                            "Click where the chosen marker is on a radiograph: it stands for \
                             what detection would find there",
                            |ui| {
                                egui::ComboBox::from_id_salt((id, "pick"))
                                    .selected_text(st.pick_marker.as_deref().unwrap_or("Off"))
                                    .width(110.0)
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_label(st.pick_marker.is_none(), "Off")
                                            .clicked()
                                        {
                                            st.pick_marker = None;
                                            if st.tool == PaneTool::Pick {
                                                st.tool = PaneTool::None;
                                            }
                                        }
                                        for n in &marker_names {
                                            if ui
                                                .selectable_label(
                                                    st.pick_marker.as_ref() == Some(n),
                                                    n.as_str(),
                                                )
                                                .clicked()
                                            {
                                                st.pick_marker = Some(n.clone());
                                                st.tool = PaneTool::Pick;
                                                st.drawing = None;
                                            }
                                        }
                                    });
                                let n: usize = st.views.iter().map(|v| v.picks.len()).sum();
                                if n > 0 {
                                    ui.weak(format!("{n}"));
                                    if ui.small_button("Clear").clicked() {
                                        st.views.iter_mut().for_each(|v| v.picks.clear());
                                    }
                                }
                            },
                        );
                        f.row_tip(
                            "Search · size",
                            "Around each marker's projection; the markers' diameter",
                            |ui| {
                                ui.add(
                                    egui::DragValue::new(&mut st.opts.search_mm)
                                        .range(2.0..=60.0)
                                        .suffix(" mm"),
                                );
                                ui.add(
                                    egui::DragValue::new(&mut st.opts.marker_mm)
                                        .speed(0.1)
                                        .range(0.3..=10.0)
                                        .suffix(" mm"),
                                );
                            },
                        );
                    }
                    f.row("Moves", |ui| {
                        let mut six = st.opts.dof == Dof2d3d::Six;
                        if ui.selectable_label(six, "Six").clicked() {
                            six = true;
                        }
                        if ui.selectable_label(!six, "Translations").clicked() {
                            six = false;
                        }
                        st.opts.dof = if six {
                            Dof2d3d::Six
                        } else {
                            Dof2d3d::Translations
                        };
                    });
                    f.row_tip("Tolerance", "A correction beyond these is coloured", |ui| {
                        ui.add(
                            egui::DragValue::new(&mut st.tolerance[0])
                                .speed(0.1)
                                .range(0.1..=100.0)
                                .suffix(" mm"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut st.tolerance[1])
                                .speed(0.1)
                                .range(0.1..=100.0)
                                .suffix("°"),
                        );
                    });
                });
            });

        ui.horizontal(|ui| {
            if busy {
                ui.spinner();
                if ui.button("Cancel").clicked() {
                    act_cancel = true;
                }
            } else {
                let any = st.views.iter().any(|v| v.on);
                if ui
                    .add_enabled(any, egui::Button::new("▶ Register"))
                    .clicked()
                {
                    act_run = true;
                }
                if ui
                    .add_enabled(any, egui::Button::new("👁 DRR at the plan"))
                    .on_hover_text("Each ticked radiograph with the CT's DRR at the plan over it")
                    .clicked()
                {
                    act_preview = true;
                }
            }
        });
        if let Some((msg, frac)) = job_line {
            ui.label(msg);
            ui.add(egui::ProgressBar::new(frac).show_percentage());
        }
        if let Some(s) = &st.status {
            ui.label(s.as_str());
        }

        if let Some(s) = &current {
            ui.separator();
            let r = &s.result;
            ui.label(egui::RichText::new(&s.label).strong());
            if s.adjusted {
                ui.weak("moved by hand");
            }
            let c = r.correction();
            // Three numbers in a row, each coloured when beyond `limit`.
            let three = |ui: &mut egui::Ui, items: [(&str, f64); 3], unit: &str, limit: f64| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    for (name, v) in items {
                        let text = egui::RichText::new(format!("{name} {v:+6.2}")).monospace();
                        if v.abs() > limit {
                            ui.colored_label(ui.visuals().warn_fg_color, text);
                        } else {
                            ui.label(text);
                        }
                    }
                    ui.label(egui::RichText::new(unit).monospace());
                });
            };
            let [tol_mm, tol_deg] = st.tolerance;
            ui.weak("Correction, patient (LPS)");
            three(ui, [("x", c[3]), ("y", c[4]), ("z", c[5])], "mm", tol_mm);
            three(ui, [("rx", c[0]), ("ry", c[1]), ("rz", c[2])], "°", tol_deg);
            ui.weak("Couch");
            three(
                ui,
                [
                    ("lat", r.couch.lateral),
                    ("lng", r.couch.longitudinal),
                    ("vrt", r.couch.vertical),
                ],
                "mm",
                tol_mm,
            );
            three(
                ui,
                [
                    ("yaw", r.couch.yaw),
                    ("pitch", r.couch.pitch),
                    ("roll", r.couch.roll),
                ],
                "°",
                tol_deg,
            );
            for v in &r.views {
                let weak = matches!(r.metric, Metric2d::GradientCorrelation | Metric2d::Ncc)
                    && v.after < 0.3;
                let text = format!(
                    "{}: {} {:.3} ▶ {:.3}{}",
                    v.label,
                    r.metric.tag(),
                    v.before,
                    v.after,
                    if v.inverted { " (inverted)" } else { "" }
                );
                if weak {
                    ui.colored_label(ui.visuals().warn_fg_color, text);
                } else {
                    ui.label(text);
                }
            }
            if let Some(m) = &r.markers {
                ui.label(format!(
                    "Markers: RMS {:.2} mm, {} observations",
                    m.rms_mm,
                    m.residuals.len()
                ));
            }
            for f in &s.fields {
                ui.label(f.line());
            }
            if let Some(w) = &r.warning {
                ui.colored_label(ui.visuals().warn_fg_color, format!("⚠ {w}"));
            }
            let mut done = format!("{} · {}", s.when, r.metric_line());
            if !r.device.is_empty() {
                done.push_str(&format!(" · {}", r.device));
            }
            ui.weak(done);

            // The nudge: the pose moved by hand, the DRRs drawn again.
            egui::Grid::new((id, "nudge"))
                .num_columns(4)
                .show(ui, |ui| {
                    let axes = ["x", "y", "z"];
                    for (k, a) in axes.iter().enumerate() {
                        ui.label(*a);
                        if ui
                            .add_enabled(!busy, egui::Button::new("➖").small())
                            .clicked()
                        {
                            let mut d = [0.0; 6];
                            d[k + 3] = -st.step_mm;
                            nudge = Some(d);
                        }
                        if ui
                            .add_enabled(!busy, egui::Button::new("➕").small())
                            .clicked()
                        {
                            let mut d = [0.0; 6];
                            d[k + 3] = st.step_mm;
                            nudge = Some(d);
                        }
                        ui.label(format!("r{a}"));
                        if ui
                            .add_enabled(!busy, egui::Button::new("➖").small())
                            .clicked()
                        {
                            let mut d = [0.0; 6];
                            d[k] = -st.step_deg;
                            nudge = Some(d);
                        }
                        if ui
                            .add_enabled(!busy, egui::Button::new("➕").small())
                            .clicked()
                        {
                            let mut d = [0.0; 6];
                            d[k] = st.step_deg;
                            nudge = Some(d);
                        }
                        ui.end_row();
                    }
                    ui.label("step");
                    ui.add(
                        egui::DragValue::new(&mut st.step_mm)
                            .speed(0.1)
                            .range(0.1..=10.0)
                            .suffix(" mm"),
                    );
                    ui.label("");
                    ui.add(
                        egui::DragValue::new(&mut st.step_deg)
                            .speed(0.1)
                            .range(0.1..=5.0)
                            .suffix("°"),
                    );
                    ui.end_row();
                });

            ui.horizontal(|ui| {
                ui.checkbox(&mut st.show_iso, "Isocentre");
                if r.markers.is_some() {
                    ui.checkbox(&mut st.show_markers, "Markers");
                }
            });
            if !structures.is_empty() {
                egui::CollapsingHeader::new("Outlines")
                    .id_salt((id, "outlines"))
                    .default_open(false)
                    .show(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .max_height(140.0)
                            .show(ui, |ui| {
                                for name in &structures {
                                    let mut on = st.outlined.contains(name);
                                    if ui.checkbox(&mut on, name.as_str()).changed() {
                                        if on {
                                            st.outlined.push(name.clone());
                                        } else {
                                            st.outlined.retain(|n| n != name);
                                        }
                                    }
                                }
                            });
                    });
            }
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(!busy, egui::Button::new("⇥ Positioned copy"))
                    .on_hover_text("The study as the patient was found, in a new workspace registered to this one")
                    .clicked()
                {
                    act_copy = true;
                }
                if ui.button("💾 Report").clicked() {
                    act_report = true;
                }
                if ui
                    .button("💾 DICOM")
                    .on_hover_text("The registration object, the DRRs as RT Images and the report, into a folder")
                    .clicked()
                {
                    act_dicom = true;
                }
                if ui.button("📋 Copy").on_hover_text("The report as text").clicked() {
                    ui.ctx().copy_text(s.report());
                }
                if ui
                    .button("📋 Table")
                    .on_hover_text("The numbers, tab-separated")
                    .clicked()
                {
                    ui.ctx().copy_text(s.table());
                }
            });
        }

        let target_slot = st.slot;
        let clip_centre = st
            .clip_shown
            .as_ref()
            .and_then(|(_, b)| *b)
            .map(|b| b.center);
        if let Some(r) = chosen_room {
            st.room = r;
        }
        let prefs = st.prefs();
        self.r2d = Some(st);
        if prefs != self.r2d_prefs {
            self.r2d_prefs = prefs;
            self.persist_settings();
        }
        if let Some(s) = new_slot {
            if let Some(st) = &mut self.r2d {
                *st = Reg2d3dState::new(s, &self.r2d_prefs);
            }
            self.reg2d3d_follow(s);
        }
        if act_cancel {
            cancel_if(true, &self.r2d_job);
        }
        if act_run {
            self.start_reg2d3d("run");
        }
        if act_preview {
            self.start_reg2d3d("preview");
        }
        if let Some(d) = nudge {
            self.nudge_reg2d3d(d);
        }
        if act_copy {
            self.reg2d3d_copy();
        }
        if act_report {
            self.reg2d3d_save_report();
        }
        if act_dicom {
            self.reg2d3d_save_dicom();
        }
        if act_clip {
            // The box's centre under the crosshair, so its slices show it.
            if let (Some(c), Some(study)) = (clip_centre, &self.slots[target_slot].study) {
                let v = study.volume.patient_to_voxel(c);
                self.slots[target_slot].cursor = v;
                self.sync_views_to_cursor(target_slot, None);
            }
        }
    }
}

impl ViewerApp {
    /// Setup `idx` of `slot` was taken out of the study.
    pub(super) fn setup_removed(&mut self, slot: usize, idx: usize) {
        if let Some(st) = self.r2d.as_mut().filter(|st| st.slot == slot) {
            st.current = match st.current {
                Some(c) if c == idx => None,
                Some(c) if c > idx => Some(c - 1),
                c => c,
            };
            if st.current.is_none() {
                st.drrs.clear();
                st.outlines.clear();
                st.outlines_for = None;
            }
        }
    }

    /// *Setups* in the data tree: the 2D/3D registrations made for this
    /// workspace's CT.
    pub(super) fn setups_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let n = study.setups.len();
        if n == 0 {
            return;
        }
        let rows: Vec<(String, String)> = study
            .setups
            .iter()
            .map(|s| {
                let c = s.result.correction();
                (
                    s.label.clone(),
                    format!(
                        "{} · {:+.1}, {:+.1}, {:+.1} mm{}",
                        s.when,
                        c[3],
                        c[4],
                        c[5],
                        if s.adjusted { " · moved by hand" } else { "" }
                    ),
                )
            })
            .collect();
        let current = self
            .r2d
            .as_ref()
            .filter(|st| st.slot == slot)
            .and_then(|st| st.current);
        let focused = self.tree_focus.clone();
        let mut show = None;
        let mut rename = None;
        let mut remove = None;
        let mut focus = None;
        egui::CollapsingHeader::new(format!("Setups ({n})"))
            .id_salt(("setups", slot))
            .default_open(true)
            .show(ui, |ui| {
                for (i, (label, line)) in rows.iter().enumerate() {
                    let text = egui::RichText::new(format!("{} {label}", REG2D3D_TOOL.glyph));
                    let resp = ui.selectable_label(current == Some(i), text);
                    ui.weak(line.as_str());
                    if resp.clicked() {
                        show = Some(i);
                    }
                    Self::focus_row(
                        ui,
                        &resp,
                        RenameTarget::Setup { slot, idx: i },
                        &focused,
                        &mut focus,
                    );
                    resp.context_menu(|ui| {
                        if ui.button("👁 Show").clicked() {
                            show = Some(i);
                            ui.close();
                        }
                        if ui.button("✏ Rename").clicked() {
                            rename = Some(RenameTarget::Setup { slot, idx: i });
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("🗑 Remove").clicked() {
                            remove = Some(ObjRef {
                                slot,
                                kind: ObjKind::Setup,
                                idx: i,
                            });
                            ui.close();
                        }
                    });
                }
            });
        if focus.is_some() {
            self.tree_focus = focus;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if remove.is_some() {
            self.obj_remove = remove;
        }
        if let Some(i) = show {
            // Where its controls are: the registration module, unfolded.
            self.module_registration = true;
            self.show_setup(slot, i);
        }
    }
}
