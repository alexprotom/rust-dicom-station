//! *Tools ▸ Posture model*: the way from one position of a patient to
//! another, played and measured ([`crate::posture`]).
//!
//! The window builds the model between studies already open in the
//! workspaces (each a **state**: its displayed series and the structure set
//! drawn on it) through [`crate::workflow::posture::build`] on a background
//! job, and then drives a workspace of its own - the **model workspace** -
//! with it. That workspace is one of the four letters (the first free one
//! past A, since every per-workspace table of the viewer is four long), kept
//! off the main screen while the window has it ([`ViewerApp::posture_owns`]):
//! its three planes, its 3D scene and its chart pane are drawn here by the
//! same `pane` function the main rows use, so the crosshair, zoom, window,
//! structure visibility and the 3D camera are the ordinary ones.
//!
//! **Time.** The slider runs over `[0, n]` for `n + 1` states, with a tick
//! at each state; with a posture parameter typed for every state the time
//! reads in its units too. Playing is one more [`super::play::PlayTarget`]
//! (`Model`), so the clock, the end modes (loop, bounce, once) and the stop
//! button are the Image playback module's. Two ways to play:
//!
//! * **Live slices** (the default): at each step only the three planes on
//!   screen are computed, on the full grid, at the exact time - on the
//!   inference GPU when there is one ([`crate::posture::gpu`], the states'
//!   images uploaded once when the model is built), else on the CPU
//!   ([`crate::posture::apply::render_planes`]); the path at the time is
//!   computed on the lattice in the same background job. The clock waits
//!   for each step to land, so a run shows every step (and a recording
//!   films every one). The rest of the volume holds whatever was there
//!   before; nothing reads it while the slider moves.
//! * **Key frames**: images rendered at evenly spaced times on a working
//!   grid that fits the playback budget
//!   ([`crate::posture::apply::working_grid`]), held in memory as a 4D
//!   group's phases are, and played from there.
//!
//! Either way, when the slider rests the exact frame at its time is
//! rendered on the full grid in the background and put in their place.
//!
//! **A frame on screen** is three things swapped under the views without
//! moving them: the volume (key frame or exact), the first state's
//! structure set with every contour point carried to the time, and the
//! first state's meshes with every vertex carried ([`crate::posture::apply`]);
//! with *Overlay* on, the forward displacement as the registration
//! module's glyphs or deformed grid. Only the volume is precomputed: the
//! structures and meshes are carried when a frame is shown.
//!
//! **What is shown** ([`Show`]): the first state of the segment moved
//! forward, the second moved back, their cross-fade (exactly each scan at
//! its end), their difference, or the Jacobian of the path, in a window of
//! its own. Over any of them, the *Jacobian wash* colours the local volume
//! change since the first state where the tissue now is
//! ([`crate::posture::apply::wash_plane`]), and the first state's dose, when
//! it has one, is drawn carried along the path - for display only: it is
//! the planned dose moved with the tissue, not a dose recalculated on the
//! new anatomy, and no number is derived from it (the workspace is the
//! window's, so no DVH or dose readout reaches it). The window says, under
//! the bar, what the model is: an interpolation between scans, for looking
//! and measuring along, not for planning; every export says it too.
//!
//! **A state lacking structures.** When one state has structures another
//! lacks, the bar says which, and when an automatic model of the registry
//! makes some of them ([`crate::workflow::posture::model_for_missing`]),
//! offers to run it on that state; its results land through the Structure
//! auto tools module as any run's do, and the model is then built again.
//!
//! **The decision to keep this a window.** Every other tool that works on
//! the images on screen is a module of the right panel. This one is not,
//! and stays a tool window: it runs a workspace of its own whose planes,
//! scene and chart it draws in itself, with a time slider and a table wider
//! than the panel; put in the panel, its views would have to replace one
//! of the main row's workspaces, which is what holding the model workspace
//! off the main screen avoids.
//!
//! **Writes**: the tables as CSV, the model as a `.rdsposture` file (read
//! back with *Open model* onto the same states, nothing refitted), the end
//! point as a DICOM Deformable Spatial Registration, the image and the
//! structures at the slider's time as a derived CT series and an RT
//! Structure Set marked as interpolated, the film of a pane as GIF or
//! PNG frames (the recorder of [`super::record`], run on this window's own
//! screenshots), and *Save GIF*: the path rendered frame by frame from the
//! model, never from the screen ([`crate::posture::film`]), all three
//! planes in one or the **selected view** only - the pane last clicked in
//! the window (any button), which is also the pane the recorder films and
//! the one its list beside ⏺ shows.
//!
//! The tables are `posture_table.rs`.

use super::detach::{tool_window, WinOpts};
use super::*;
use crate::posture::apply::{self as papply, Extras, FrameExtras, KeyFrame, Plane, Show};
use crate::posture::film as pfilm;
use crate::posture::gpu::{self as pgpu, GpuSlicer};
use crate::posture::measure::Tables;
use crate::posture::svf::Field;
use crate::posture::{self, file as pfile, Anchor, Mode, PostureModel};
use crate::registration::FieldStyle;
use crate::rtdose::DoseGrid;
use crate::rtstruct::StructureSet;
use crate::workflow::posture as wp;

/// What the anchor picker holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(super) enum AnchorKind {
    /// A structure on every state (the name beside it).
    #[default]
    Structure,
    Image,
    Scanner,
}

/// A row of the table whose curves the chart pane shows.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum ChartRow {
    Structure(String),
    Body(String),
    Joint(String),
}

/// A finished build: the model, its tables, and what drawing it needs.
pub(super) struct Built {
    pub(super) model: Arc<PostureModel>,
    pub(super) tables: Tables,
    pub(super) occupancy: posture::file::Occupancies,
    /// The states' own images, in their own coordinates.
    pub(super) images: Vec<Arc<Volume>>,
    /// The first state, lean: its series, volume and the structure set used.
    pub(super) first: LoadedStudy,
    /// The last state's frame of reference, for the registration export.
    pub(super) last_for: String,
    /// The first state's structures and their meshes, carried each frame.
    pub(super) base_set: Option<StructureSet>,
    pub(super) base_meshes: Arc<Vec<RoiMesh>>,
    pub(super) warnings: Vec<String>,
    /// The states' images on the GPU, for the live slices; `None`: the CPU.
    pub(super) slicer: Option<Arc<GpuSlicer>>,
    /// Where the live slices are computed, as the user should read it.
    pub(super) slicer_note: String,
    /// The first state's dose, drawn along the path (display only).
    pub(super) dose: Option<Arc<DoseGrid>>,
}

/// A key-frame build's answer.
type Keys = (Show, bool, Vec<KeyFrame>);
/// An exact frame's answer: time, what it shows, the image, the field,
/// the extras.
type Exact = (f64, Show, Arc<Volume>, Arc<Field>, FrameExtras);

/// A live step's answer: the three planes on screen at a time.
pub(super) struct Live {
    t: f64,
    show: Show,
    planes: Vec<Plane>,
    values: Vec<Vec<f32>>,
    forward: Arc<Field>,
    extras: FrameExtras,
    /// Computed on the GPU; why not, when it was asked and failed.
    gpu: bool,
    why: Option<String>,
}

/// A state that lacks structures another has, and what would make them.
#[derive(Clone)]
pub(super) struct Offer {
    slot: usize,
    lacking: Vec<String>,
    model: Option<(crate::zoo::AutoModel, Vec<String>)>,
}

/// The window's state.
pub(super) struct PostureWin {
    pub(super) open: bool,
    /// The states, as workspaces, in order.
    pub(super) states: Vec<usize>,
    /// The posture parameter typed for each state.
    pub(super) posture_text: Vec<String>,
    pub(super) anchor_kind: AnchorKind,
    pub(super) anchor_name: String,
    pub(super) mode: Mode,
    /// The rigid bodies by name; `None`: the bones every state has.
    pub(super) bodies: Option<Vec<String>>,
    pub(super) residual: bool,
    pub(super) lattice_mm: f32,
    pub(super) reach_mm: f32,
    /// Key frames to play from.
    pub(super) frames: usize,
    pub(super) show: Show,
    pub(super) overlay: FieldStyle,
    pub(super) job: Option<Job<anyhow::Result<Built>>>,
    pub(super) built: Option<Built>,
    /// The model workspace.
    pub(super) slot: Option<usize>,
    /// The slider's time.
    pub(super) t: f64,
    /// What is on screen: time, show, exact.
    pub(super) shown: Option<(f64, Show, bool)>,
    pub(super) keys: Vec<KeyFrame>,
    /// What the key frames show, and whether they are on the full grid.
    pub(super) keys_show: Show,
    pub(super) keys_full: bool,
    pub(super) key_job: Option<Job<anyhow::Result<Keys>>>,
    pub(super) exact_job: Option<Job<anyhow::Result<Exact>>>,
    /// egui's clock when the slider last moved: the exact frame waits for
    /// it to rest.
    pub(super) moved_at: f64,
    /// Key frames a second.
    pub(super) fps: f32,
    pub(super) focus: Option<String>,
    pub(super) chart: Option<ChartRow>,
    /// The two bodies of a joint being added.
    pub(super) joint_pick: [String; 2],
    /// Joints added by hand (name, first, second).
    pub(super) joints: Vec<(String, String, String)>,
    /// The pane a recording films, and the one *Save GIF* takes alone: the
    /// pane last clicked in the window, or the one picked in its list.
    pub(super) film: PaneKind,
    pub(super) rec: Option<record::Recording>,
    pub(super) rec_status: Option<String>,
    /// *Save GIF*: the three planes (else the selected pane alone), the
    /// film's width in pixels, there and back, the job writing it.
    pub(super) gif_all: bool,
    pub(super) gif_width: usize,
    pub(super) gif_bounce: bool,
    /// What the last GIF came to, or why it did not.
    pub(super) gif_status: Option<String>,
    pub(super) gif_job: Option<Job<anyhow::Result<(std::path::PathBuf, pfilm::FilmSummary)>>>,
    /// Width over height of a plane's pane as last drawn: the film's cells.
    pub(super) pane_aspect: Option<f32>,
    /// The forward displacement on screen, for the overlay, and its count.
    pub(super) field: Option<Arc<VectorField>>,
    pub(super) field_gen: u64,
    pub(super) export_job: Option<Job<anyhow::Result<String>>>,
    pub(super) status: Option<String>,
    /// Play the three slices on screen (else the key frames).
    pub(super) live: bool,
    pub(super) live_job: Option<Job<anyhow::Result<Live>>>,
    /// Where the last live step ran, and why not on the GPU.
    pub(super) live_note: Option<String>,
    /// The Jacobian as a colour wash over the views.
    pub(super) wash: bool,
    /// The wash of the frame on screen, per lattice node, and its count.
    pub(super) wash_vals: Option<Arc<Vec<f32>>>,
    pub(super) wash_gen: u64,
    /// Draw the first state's dose along the path.
    pub(super) dose_on: bool,
    /// The WEPL column, and its beam's IEC angles.
    pub(super) wepl: bool,
    pub(super) gantry_deg: f32,
    pub(super) couch_deg: f32,
    /// Bodies on curves through a chain's states.
    pub(super) smooth: bool,
    /// What the missing-structures line offers, for the states as they are
    /// (the key: every state's names).
    pub(super) offer_key: String,
    pub(super) offers: Vec<Offer>,
}

impl Default for PostureWin {
    fn default() -> Self {
        PostureWin {
            open: false,
            states: vec![0, 1],
            posture_text: vec![String::new(); 2],
            anchor_kind: AnchorKind::Structure,
            anchor_name: String::new(),
            mode: Mode::Diffeomorphic,
            bodies: None,
            residual: false,
            lattice_mm: 6.0,
            reach_mm: 40.0,
            frames: 21,
            show: Show::StartForward,
            overlay: FieldStyle::None,
            job: None,
            built: None,
            slot: None,
            t: 0.0,
            shown: None,
            keys: Vec::new(),
            keys_show: Show::StartForward,
            keys_full: false,
            key_job: None,
            exact_job: None,
            moved_at: 0.0,
            fps: 6.0,
            focus: None,
            chart: None,
            joint_pick: [String::new(), String::new()],
            joints: Vec::new(),
            film: PaneKind::Plane(ViewPlane::Sagittal),
            rec: None,
            rec_status: None,
            gif_all: true,
            gif_width: 800,
            gif_bounce: false,
            gif_status: None,
            gif_job: None,
            pane_aspect: None,
            field: None,
            field_gen: 0,
            export_job: None,
            status: None,
            live: true,
            live_job: None,
            live_note: None,
            wash: false,
            wash_vals: None,
            wash_gen: 0,
            dose_on: true,
            wepl: true,
            gantry_deg: 90.0,
            couch_deg: 0.0,
            smooth: true,
            offer_key: String::new(),
            offers: Vec::new(),
        }
    }
}

/// The GPU slicer for the live slices, when the inference device is a real
/// one and takes the states' images; else why the CPU does them.
fn make_slicer(images: &[Arc<Volume>]) -> (Option<Arc<GpuSlicer>>, String) {
    match crate::nn::device::DevicePref::Auto.resolve() {
        Ok(Some(ctx)) if !ctx.is_software() => match GpuSlicer::new(&ctx, images) {
            Ok(s) => {
                let note = format!("Live slices on the {}", s.describe());
                (Some(Arc::new(s)), note)
            }
            Err(e) => (
                None,
                format!("Live slices on the CPU (the GPU refused them: {e:#})"),
            ),
        },
        Ok(Some(_)) => (
            None,
            "Live slices on the CPU (the GPU is a renderer in software)".into(),
        ),
        _ => (
            None,
            format!("Live slices on the {}", crate::nn::device::describe_cpu()),
        ),
    }
}

/// The names of a study's volume structures, in the set the posture model
/// would use (the one drawn on the displayed series, else the first).
fn structure_names(study: &LoadedStudy) -> Vec<String> {
    let uid = study
        .series
        .get(study.active_series)
        .map(|s| s.uid.as_str());
    let set = study
        .structure_sets
        .iter()
        .position(|ss| uid.is_some_and(|u| ss.referenced_series_uid == u))
        .or_else(|| (!study.structure_sets.is_empty()).then_some(0));
    set.map(|i| {
        study.structure_sets[i]
            .rois
            .iter()
            .filter(|r| !r.is_poi() && !r.contours.is_empty())
            .map(|r| r.name.clone())
            .collect()
    })
    .unwrap_or_default()
}

/// A lean copy of a study for a background job: its displayed series, its
/// volume (shared) and the structure set that counts.
fn lean(study: &LoadedStudy) -> LoadedStudy {
    let uid = study.series.get(study.active_series).map(|s| s.uid.clone());
    let set = study
        .structure_sets
        .iter()
        .position(|ss| {
            uid.as_deref()
                .is_some_and(|u| ss.referenced_series_uid == u)
        })
        .or_else(|| (!study.structure_sets.is_empty()).then_some(0));
    LoadedStudy {
        meta: study.meta.clone(),
        series: study
            .series
            .get(study.active_series)
            .cloned()
            .into_iter()
            .collect(),
        active_series: 0,
        volume: study.volume.clone(),
        structure_sets: set
            .map(|i| study.structure_sets[i].clone())
            .into_iter()
            .collect(),
        default_window: study.default_window,
        ..LoadedStudy::default()
    }
}

impl ViewerApp {
    /// Does the posture window hold this workspace as its model workspace?
    pub(super) fn posture_owns(&self, slot: usize) -> bool {
        self.posture.slot == Some(slot)
    }

    /// The forward displacement to draw over the model workspace's views,
    /// with its style and a counter that moves when it changes.
    pub(super) fn posture_field(&self, slot: usize) -> Option<(Arc<VectorField>, FieldStyle, u64)> {
        if !self.posture_owns(slot) || self.posture.overlay == FieldStyle::None {
            return None;
        }
        self.posture
            .field
            .clone()
            .map(|f| (f, self.posture.overlay, self.posture.field_gen))
    }

    /// The Jacobian wash over the model workspace's views: the model's
    /// lattice, the values per node, and a counter that moves with them.
    pub(super) fn posture_wash(
        &self,
        slot: usize,
    ) -> Option<(posture::svf::Lattice, Arc<Vec<f32>>, u64, bool)> {
        if !self.posture_owns(slot) || !self.posture.wash {
            return None;
        }
        let b = self.posture.built.as_ref()?;
        let v = self.posture.wash_vals.clone()?;
        // Over a scan, tissue only; over the difference or the Jacobian
        // image there is no air to leave out.
        let scan = matches!(
            self.posture.show,
            Show::StartForward | Show::EndBackward | Show::CrossFade
        );
        Some((b.model.lattice.clone(), v, self.posture.wash_gen, scan))
    }

    /// The structure the table points at, as an ROI of the model
    /// workspace's set: drawn thicker, like a module's focus.
    pub(super) fn posture_focus(&self, slot: usize) -> Option<usize> {
        if !self.posture_owns(slot) {
            return None;
        }
        let name = self.posture.focus.as_ref()?;
        self.slots[slot]
            .active_structures()?
            .rois
            .iter()
            .position(|r| r.name.eq_ignore_ascii_case(name))
    }

    /// Open the window (*Tools ▸ Posture model*): the first two workspaces
    /// with an image as the states.
    pub(super) fn open_posture(&mut self) {
        let vols: Vec<usize> = self
            .volume_slot_list()
            .into_iter()
            .filter(|s| !self.posture_owns(*s))
            .collect();
        let st = &mut self.posture;
        if !st.open && st.built.is_none() && vols.len() >= 2 {
            st.states = vols.iter().take(2).copied().collect();
            st.posture_text = vec![String::new(); st.states.len()];
        }
        st.open = true;
        self.raise_window("posture");
        // A model built before the window was closed comes back.
        if self.posture.built.is_some() && self.posture.slot.is_none() {
            self.posture_make_workspace();
        }
    }

    /// The structures every chosen state has, in the first one's order.
    fn posture_common(&self) -> Vec<String> {
        let mut lists = self
            .posture
            .states
            .iter()
            .filter_map(|&s| self.slots.get(s).and_then(|x| x.study.as_ref()))
            .map(structure_names);
        let Some(first) = lists.next() else {
            return Vec::new();
        };
        let rest: Vec<Vec<String>> = lists.collect();
        first
            .into_iter()
            .filter(|n| {
                rest.iter()
                    .all(|l| l.iter().any(|m| m.trim().eq_ignore_ascii_case(n.trim())))
            })
            .collect()
    }

    /// The request the window's controls make.
    fn posture_request(&self) -> wp::PostureRequest {
        let st = &self.posture;
        let anchor = match st.anchor_kind {
            AnchorKind::Structure if !st.anchor_name.trim().is_empty() => {
                Anchor::Structure(st.anchor_name.trim().to_string())
            }
            AnchorKind::Structure | AnchorKind::Image => Anchor::Image,
            AnchorKind::Scanner => Anchor::Scanner,
        };
        // With two states and a registration between exactly their
        // workspaces, the linear mode walks that.
        let linear = match (&self.registration, st.mode, st.states.as_slice()) {
            (Some(r), Mode::Linear, [a, b]) if r.fixed_slot == *a && r.moving_slot == *b => {
                Some(r.result.transform.clone())
            }
            _ => None,
        };
        wp::PostureRequest {
            anchor,
            bodies: match &st.bodies {
                None => wp::Bodies::Bones,
                Some(v) if v.is_empty() => wp::Bodies::None,
                Some(v) => wp::Bodies::Named(v.clone()),
            },
            mode: st.mode,
            lattice_mm: st.lattice_mm as f64,
            reach_mm: st.reach_mm as f64,
            residual: st.residual.then(wp::default_residual),
            linear,
            joints: st.joints.clone(),
            wepl: st.wepl.then_some(wp::WeplBeam {
                gantry_deg: st.gantry_deg as f64,
                couch_deg: st.couch_deg as f64,
                hlut: None,
            }),
            smooth_chain: st.smooth,
            ..wp::PostureRequest::default()
        }
    }

    /// The extras every frame job is asked for.
    fn posture_extras(&self) -> Extras {
        Extras {
            dose: self
                .posture
                .built
                .as_ref()
                .and_then(|b| b.dose.clone())
                .filter(|_| self.posture.dose_on),
            jacobian: self.posture.wash,
        }
    }

    /// Start building the model in the background.
    fn posture_build(&mut self, from_file: Option<std::path::PathBuf>) {
        let st = &self.posture;
        if st.job.is_some() {
            return;
        }
        let mut studies = Vec::new();
        for (i, &s) in st.states.iter().enumerate() {
            match self.slots.get(s).and_then(|x| x.study.as_ref()) {
                Some(study) if study.has_volume() => studies.push((
                    format!("{}{}", SLOT_NAMES[s], if i == 0 { " (start)" } else { "" }),
                    lean(study),
                )),
                _ => {
                    self.error = Some(format!(
                        "Workspace {} has no image to be a state of the model",
                        SLOT_NAMES[s.min(MAX_WORKSPACES - 1)]
                    ));
                    return;
                }
            }
        }
        let postures: Vec<Option<f64>> = st
            .posture_text
            .iter()
            .map(|t| t.trim().replace(',', ".").parse::<f64>().ok())
            .collect();
        let req = self.posture_request();
        // The first state's dose, as its workspace shows it.
        let dose: Option<Arc<DoseGrid>> = st.states.first().and_then(|&s| {
            let slot = self.slots.get(s)?;
            let study = slot.study.as_ref()?;
            let d = study
                .doses
                .get(slot.active_dose.min(study.doses.len().saturating_sub(1)))?;
            Some(Arc::new(d.clone()))
        });
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.posture.status = None;
        self.posture.job = Some(Job::spawn(progress, move |p| -> anyhow::Result<Built> {
            let inputs: Vec<wp::StateInput> = studies
                .iter()
                .enumerate()
                .map(|(i, (label, s))| wp::StateInput {
                    label: label.clone(),
                    study: s,
                    structure_set: (!s.structure_sets.is_empty()).then_some(0),
                    posture: postures.get(i).copied().flatten(),
                })
                .collect();
            let o = match &from_file {
                Some(path) => {
                    let model = pfile::read(path)?;
                    wp::reopen(model, &inputs, &req, p)?
                }
                None => wp::build(&inputs, &req, p)?,
            };
            let first = studies[0].1.clone();
            let base_set = first.structure_sets.first().cloned();
            p.set("Meshing the structures");
            let base_meshes = base_set
                .as_ref()
                .map(|ss| mesh3d::build_meshes(ss, &Progress::default()))
                .unwrap_or_default();
            p.set("Uploading the states for the live slices");
            let images: Vec<Arc<Volume>> = studies.iter().map(|(_, s)| s.volume.clone()).collect();
            let (slicer, slicer_note) = make_slicer(&images);
            Ok(Built {
                model: Arc::new(o.model),
                tables: o.tables,
                occupancy: o.occupancy,
                images: studies.iter().map(|(_, s)| s.volume.clone()).collect(),
                last_for: studies
                    .last()
                    .map(|(_, s)| s.volume.frame_of_reference_uid.clone())
                    .unwrap_or_default(),
                first,
                base_set,
                base_meshes: Arc::new(base_meshes),
                warnings: o.warnings,
                slicer,
                slicer_note,
                dose,
            })
        }));
    }

    /// A free workspace for the model: the one it had, else the first letter
    /// past A that is neither on screen nor holding anything.
    fn posture_free_slot(&self) -> Option<usize> {
        self.posture.slot.or_else(|| {
            (1..MAX_WORKSPACES).find(|&s| !self.is_open(s) && self.slots[s].study.is_none())
        })
    }

    /// Put the built model into its workspace, at the first state.
    fn posture_make_workspace(&mut self) {
        let Some(slot) = self.posture_free_slot() else {
            self.error = Some(
                "The posture model needs a workspace of its own and all four are in use: \
                 close one (B to D) and press Build again."
                    .into(),
            );
            return;
        };
        let Some(b) = self.posture.built.as_ref() else {
            return;
        };
        let mut series = b.first.series.first().cloned().unwrap_or_default_series();
        series.uid = format!("posture.{}", series.uid);
        series.description = format!(
            "Posture model: {}",
            b.model
                .states
                .iter()
                .map(|s| s.label.as_str())
                .collect::<Vec<_>>()
                .join(" to ")
        );
        let set = b.base_set.clone().map(|mut ss| {
            ss.label = "Posture model".into();
            ss.referenced_series_uid = series.uid.clone();
            ss.locked = true;
            ss
        });
        let study = LoadedStudy {
            meta: b.first.meta.clone(),
            series: vec![series],
            active_series: 0,
            volume: b.images[0].clone(),
            structure_sets: set.into_iter().collect(),
            default_window: b.first.default_window,
            warnings: b.warnings.clone(),
            ..LoadedStudy::default()
        };
        let first = b.images[0].clone();
        let zero = Field::zero(&b.model.lattice);
        let extras = FrameExtras {
            dose: b.dose.clone().filter(|_| self.posture.dose_on),
            jacobian: None,
        };
        self.posture.slot = Some(slot);
        self.install_model_study(slot, study);
        // The first frame is the first state as it is, its structures,
        // meshes and dose where they are.
        self.posture.t = 0.0;
        self.posture.shown = None;
        self.posture.keys.clear();
        self.posture.field = None;
        self.posture.wash_vals = None;
        self.posture_install(first, &zero, &extras, 0.0, Show::StartForward, true);
        if !self.posture.live {
            self.posture_start_keys();
        }
    }

    /// Fill a workspace with the model's study without what loading a study
    /// does to the rest of the viewer (no registration cleared, the
    /// workspace not put on the main screen).
    fn install_model_study(&mut self, slot: usize, study: LoadedStudy) {
        self.drop_phase_cache(slot);
        self.d3_windows.retain(|w| w.slot != slot);
        self.slots[slot] = StudySlot::empty();
        let window = study.default_window;
        let dims = study.volume.dims;
        let s = &mut self.slots[slot];
        s.roi_visible = study
            .structure_sets
            .first()
            .map(|ss| vec![true; ss.rois.len()])
            .unwrap_or_default();
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
        self.series_window[slot] = window;
        self.window_center[slot] = window.0;
        self.window_width[slot] = window.1.max(1.0);
        self.chart[slot] = None;
        self.mark_all_saved(slot);
        self.shown[slot] = false;
        self.settings_gen += 1;
    }

    /// Render the key frames in the background, for what is shown now.
    fn posture_start_keys(&mut self) {
        let Some(b) = self.posture.built.as_ref() else {
            return;
        };
        if let Some(j) = &self.posture.key_job {
            j.progress.cancel();
        }
        let model = b.model.clone();
        let images = b.images.clone();
        let show = self.posture.show;
        let n = self.posture.frames.clamp(2, 200);
        let budget = self.play.budget_mb.max(64) * (1 << 20);
        let grid = papply::working_grid(&model.grid, n, budget);
        let full = grid.dims == model.grid.dims;
        let extras = self.posture_extras();
        let progress = Arc::new(Progress::default());
        progress.set("Key frames");
        self.posture.key_job = Some(Job::spawn(progress, move |p| {
            papply::key_frames(&model, &images, n, show, &grid, &extras, p).map(|k| (show, full, k))
        }));
    }

    /// The three planes the model workspace's views show now.
    fn posture_planes(&self, slot: usize) -> Vec<Plane> {
        let mut out: Vec<Plane> = self.slots[slot]
            .views
            .iter()
            .take(3)
            .map(|v| Plane::of_view(v.plane, v.slice))
            .collect();
        out.dedup();
        out
    }

    /// One live step: the planes on screen at the slider's time.
    fn posture_start_live(&mut self) {
        let (Some(b), Some(slot)) = (self.posture.built.as_ref(), self.posture.slot) else {
            return;
        };
        let model = b.model.clone();
        let images = b.images.clone();
        let slicer = b.slicer.clone();
        let (t, show) = (self.posture.t, self.posture.show);
        let planes = self.posture_planes(slot);
        let extras = self.posture_extras();
        let progress = Arc::new(Progress::default());
        self.posture.live_job = Some(Job::spawn(progress, move |_p| {
            let frame = model.frame(t);
            let (values, gpu, why) = pgpu::render_planes_with(
                slicer.as_deref(),
                &model,
                &images,
                &frame,
                show,
                &model.grid,
                &planes,
            )?;
            let extras = extras.of(&frame);
            Ok(Live {
                t,
                show,
                planes,
                values,
                forward: Arc::new(frame.forward),
                extras,
                gpu,
                why,
            })
        }));
    }

    /// Render the exact frame at the slider's time on the full grid.
    fn posture_start_exact(&mut self) {
        let Some(b) = self.posture.built.as_ref() else {
            return;
        };
        let model = b.model.clone();
        let images = b.images.clone();
        let (t, show) = (self.posture.t, self.posture.show);
        let extras = self.posture_extras();
        let progress = Arc::new(Progress::default());
        self.posture.exact_job = Some(Job::spawn(progress, move |_p| {
            let frame = model.frame(t);
            let vol = papply::render(&model, &images, &frame, show, &model.grid)?;
            let extras = extras.of(&frame);
            Ok((t, show, Arc::new(vol), Arc::new(frame.forward), extras))
        }));
    }

    /// Put a live step on the model workspace: its planes written into the
    /// volume on screen (on the full grid; a key frame's coarser volume is
    /// replaced by the first state's image first), then the rest as any
    /// frame.
    fn posture_install_planes(&mut self, live: Live) {
        let (Some(slot), Some(b)) = (self.posture.slot, self.posture.built.as_ref()) else {
            return;
        };
        let grid_dims = b.model.grid.dims;
        let base = b.images[0].clone();
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        let mut vol = std::mem::replace(&mut study.volume, base.clone());
        if vol.dims != grid_dims {
            vol = base;
        }
        papply::write_planes(Arc::make_mut(&mut vol), &live.planes, &live.values);
        self.posture.live_note = Some(match (&live.why, live.gpu) {
            (Some(why), _) => format!("Live slices on the CPU: the GPU failed ({why})"),
            (None, true) => "GPU".into(),
            (None, false) => "CPU".into(),
        });
        self.posture_install(vol, &live.forward, &live.extras, live.t, live.show, false);
    }

    /// Put a frame on the model workspace: the volume, the structures and
    /// the meshes carried to its time, the view held still.
    fn posture_install(
        &mut self,
        vol: Arc<Volume>,
        forward: &Field,
        extras: &FrameExtras,
        t: f64,
        show: Show,
        exact: bool,
    ) {
        let Some(slot) = self.posture.slot else {
            return;
        };
        let Some(b) = self.posture.built.as_ref() else {
            return;
        };
        let when = match b.model.param_at(t) {
            Some(v) => format!("t = {t:.2} ({v:.1})"),
            None => format!("t = {t:.2}"),
        };
        let carried = b.base_set.as_ref().map(|ss| {
            let mut c = papply::carry_structure_set(ss, forward, format!("Posture model, {when}"));
            if let Some(study) = self.slots[slot].study.as_ref() {
                if let Some(se) = study.series.first() {
                    c.referenced_series_uid = se.uid.clone();
                }
            }
            c
        });
        let meshes = Arc::new(papply::carry_meshes(&b.base_meshes, forward));
        let field = (self.posture.overlay != FieldStyle::None)
            .then(|| Arc::new(forward.to_vector_field(Some("posture model".into()))));
        let window_changed = self.posture.shown.is_none_or(|(_, s, _)| s != show);
        let dose_on = self.posture.dose_on;
        let s = &mut self.slots[slot];
        let Some(study) = s.study.as_mut() else {
            return;
        };
        // Key frames and the exact frame may be on two lattices of one
        // extent: the views keep their place in millimetres.
        let old = study.volume.dims;
        let new = vol.dims;
        if old != new {
            let scale = |v: f64, a: usize| {
                if old[a] > 1 {
                    v * (new[a].max(1) - 1) as f64 / (old[a] - 1) as f64
                } else {
                    v
                }
            };
            for (a, c) in s.cursor.iter_mut().enumerate() {
                *c = scale(*c, a).clamp(0.0, new[a].saturating_sub(1) as f64);
            }
            for v in &mut s.views {
                let a = match v.plane {
                    ViewPlane::Axial => 2,
                    ViewPlane::Sagittal => 0,
                    ViewPlane::Coronal => 1,
                };
                v.slice = (scale(v.slice as f64, a).round() as usize).min(new[a].saturating_sub(1));
            }
        }
        study.volume = vol;
        if let Some(c) = carried {
            study.structure_sets = vec![c];
        }
        // The dose drawn along the path (display only); none when it is
        // switched off.
        match (&extras.dose, dose_on) {
            (Some(d), true) => {
                study.doses = vec![(**d).clone()];
                s.active_dose = 0;
                s.dose_shown = true;
            }
            (_, false) => {
                study.doses.clear();
                s.dose_shown = false;
            }
            (None, true) => {}
        }
        for v in &mut s.views {
            v.invalidate();
        }
        self.struct_gen[slot] += 1;
        if window_changed {
            let w = show.window().unwrap_or(b.first.default_window);
            self.series_window[slot] = w;
            self.window_center[slot] = w.0;
            self.window_width[slot] = w.1.max(1.0);
        }
        self.install_scene_meshes(slot, meshes);
        self.posture.field = field;
        self.posture.field_gen += 1;
        match &extras.jacobian {
            Some(j) => {
                self.posture.wash_vals = Some(j.clone());
                self.posture.wash_gen += 1;
            }
            None if !self.posture.wash => self.posture.wash_vals = None,
            None => {}
        }
        self.posture.shown = Some((t, show, exact));
    }

    /// The times a run steps through: the key frames', or with live
    /// slices as many evenly spaced times as there would be key frames.
    pub(super) fn posture_times(&self) -> Vec<f64> {
        if !self.posture.live {
            return self.posture.keys.iter().map(|k| k.t).collect();
        }
        let Some(b) = self.posture.built.as_ref() else {
            return Vec::new();
        };
        let n = self.posture.frames.clamp(2, 200);
        let span = b.model.span();
        (0..n).map(|k| span * k as f64 / (n - 1) as f64).collect()
    }

    /// The step of a run nearest the slider.
    fn posture_nearest_key(&self) -> Option<usize> {
        let times = self.posture_times();
        (0..times.len()).min_by(|&a, &b| {
            (times[a] - self.posture.t)
                .abs()
                .total_cmp(&(times[b] - self.posture.t).abs())
        })
    }

    /// Move the slider to `t` (a click, a drag, the clock).
    pub(super) fn posture_set_t(&mut self, t: f64, now: f64) {
        let span = self.posture.built.as_ref().map_or(1.0, |b| b.model.span());
        self.posture.t = t.clamp(0.0, span);
        self.posture.moved_at = now;
        if let Some(j) = self.posture.exact_job.take() {
            j.progress.cancel();
        }
    }

    /// One step of a run through the key frames; `None` ends it.
    pub(super) fn posture_play_step(
        &mut self,
        dir: i32,
        mode: play::PlayMode,
        now: f64,
    ) -> Option<i32> {
        let times = self.posture_times();
        let cur = self.posture_nearest_key()?;
        let (next, dir) = play::advance(cur, times.len(), 1, dir, mode)?;
        self.posture_set_t(times[next], now);
        Some(dir)
    }

    /// A live step is being computed: the clock waits for it.
    pub(super) fn posture_step_pending(&self) -> bool {
        self.posture.live && self.posture.live_job.is_some()
    }

    /// Where a run through the key frames stands (for the recorder).
    pub(super) fn posture_position(&self) -> Option<usize> {
        self.posture_nearest_key()
    }

    /// Is a model run in flight?
    fn posture_playing(&self) -> bool {
        self.play
            .running
            .is_some_and(|r| matches!(r.target, play::PlayTarget::Model { .. }))
    }

    /// Poll the jobs and keep the frame on screen the one the slider asks
    /// for. Runs every pass, open or not.
    pub(super) fn poll_posture(&mut self, ctx: &egui::Context) {
        if let Some(res) = poll_job(&mut self.posture.job, ctx, "Posture model", &mut self.error) {
            match res {
                Ok(b) => {
                    self.posture.status = Some(format!(
                        "Built: {} state(s), {} bod{}, {} structure row(s)",
                        b.model.states.len(),
                        b.model.body_names().len(),
                        if b.model.body_names().len() == 1 {
                            "y"
                        } else {
                            "ies"
                        },
                        b.tables.structures.len()
                    ));
                    if let Some(slot) = self.posture.slot {
                        self.close_posture_workspace(slot);
                    }
                    self.posture.built = Some(b);
                    self.posture.chart = None;
                    self.posture_make_workspace();
                }
                Err(e) if !progress::is_cancellation(&e) => {
                    self.error = Some(format!("Posture model failed: {e:#}"));
                }
                Err(_) => self.posture.status = Some("Cancelled".into()),
            }
        }
        if let Some(res) = poll_job(
            &mut self.posture.key_job,
            ctx,
            "Key frames",
            &mut self.error,
        ) {
            match res {
                Ok((show, full, keys)) if show == self.posture.show => {
                    self.posture.keys = keys;
                    self.posture.keys_show = show;
                    self.posture.keys_full = full;
                }
                Ok(_) => {}
                Err(e) if !progress::is_cancellation(&e) => {
                    self.error = Some(format!("Key frames failed: {e:#}"));
                }
                Err(_) => {}
            }
        }
        if let Some(res) = poll_job(
            &mut self.posture.exact_job,
            ctx,
            "Posture frame",
            &mut self.error,
        ) {
            match res {
                Ok((t, show, vol, fwd, extras))
                    if (t - self.posture.t).abs() < 1e-9 && show == self.posture.show =>
                {
                    self.posture_install(vol, &fwd, &extras, t, show, true);
                }
                Ok(_) => {}
                Err(e) if !progress::is_cancellation(&e) => {
                    self.error = Some(format!("Posture frame failed: {e:#}"));
                }
                Err(_) => {}
            }
        }
        if let Some(res) = poll_job(&mut self.posture.export_job, ctx, "Export", &mut self.error) {
            match res {
                Ok(msg) => self.notice = Some(msg),
                Err(e) => self.error = Some(format!("Export failed: {e:#}")),
            }
        }
        if let Some(res) = poll_job(&mut self.posture.gif_job, ctx, "GIF", &mut self.error) {
            self.posture.gif_status = match res {
                Ok((dest, s)) => Some(format!(
                    "GIF: {} frames, {} x {} px, {} KB, on the {}, written to {}",
                    s.frames,
                    s.width,
                    s.height,
                    s.bytes.div_ceil(1024),
                    if s.gpu { "GPU" } else { "CPU" },
                    dest.display()
                )),
                Err(e) if progress::is_cancellation(&e) => Some("GIF cancelled".into()),
                Err(e) => {
                    self.error = Some(format!("Writing the GIF failed: {e:#}"));
                    None
                }
            };
        }
        if let Some(res) = poll_job(
            &mut self.posture.live_job,
            ctx,
            "Posture slices",
            &mut self.error,
        ) {
            match res {
                Ok(live) => {
                    // Never over the exact frame of the slider's time.
                    let exact_here = self.posture.shown.is_some_and(|(t, s, exact)| {
                        exact && s == self.posture.show && (t - self.posture.t).abs() < 1e-9
                    });
                    if live.show == self.posture.show && !exact_here {
                        self.posture_install_planes(live);
                    }
                }
                Err(e) if !progress::is_cancellation(&e) => {
                    self.error = Some(format!("Posture slices failed: {e:#}"));
                }
                Err(_) => {}
            }
        }
        if self.posture.slot.is_none() || self.posture.built.is_none() {
            return;
        }
        let show = self.posture.show;
        // Live: the planes at the slider's time, one step at a time.
        if self.posture.live && self.posture.live_job.is_none() {
            let here = self
                .posture
                .shown
                .is_some_and(|(t, s, _)| s == show && (t - self.posture.t).abs() < 1e-9);
            if !here {
                self.posture_start_live();
            }
        }
        // The key frame nearest the slider, when it is not on screen.
        if !self.posture.live && self.posture.keys_show == show {
            if let Some(k) = self.posture_nearest_key() {
                let key_t = self.posture.keys[k].t;
                let on_screen = self.posture.shown.is_some_and(|(t, s, exact)| {
                    s == show
                        && (t - key_t).abs() < 1e-9
                        && (!exact || (t - self.posture.t).abs() < 1e-9)
                });
                let exact_here = self.posture.shown.is_some_and(|(t, s, exact)| {
                    exact && s == show && (t - self.posture.t).abs() < 1e-9
                });
                if !on_screen && !exact_here {
                    let f = self.posture.keys[k].clone();
                    self.posture_install(f.volume, &f.forward, &f.extras, f.t, show, false);
                }
            }
        }
        // The exact frame once the slider rests.
        let now = ctx.input(|i| i.time);
        let rested = now - self.posture.moved_at > 0.35;
        let exact_shown = self
            .posture
            .shown
            .is_some_and(|(t, s, exact)| exact && s == show && (t - self.posture.t).abs() < 1e-9);
        let key_is_exact = !self.posture.live
            && self.posture.keys_full
            && self.posture.keys_show == show
            && self
                .posture
                .keys
                .iter()
                .any(|k| (k.t - self.posture.t).abs() < 1e-9);
        if !self.posture_playing()
            && self.posture.exact_job.is_none()
            && !exact_shown
            && !key_is_exact
        {
            if rested {
                self.posture_start_exact();
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(120));
            }
        }
    }

    /// Empty the model workspace and let its letter go.
    fn close_posture_workspace(&mut self, slot: usize) {
        if self.play.running.is_some_and(|r| r.target.slot() == slot) {
            self.stop_play();
        }
        self.posture.slot = None;
        self.posture.shown = None;
        self.posture.field = None;
        self.close_workspace(slot);
    }

    /// The window closed: the jobs stop and the workspace goes; the model
    /// stays, and comes back with the window.
    fn close_posture(&mut self) {
        if let Some(j) = &self.posture.key_job {
            j.progress.cancel();
        }
        if let Some(j) = &self.posture.exact_job {
            j.progress.cancel();
        }
        if let Some(j) = &self.posture.live_job {
            j.progress.cancel();
        }
        if let Some(j) = &self.posture.gif_job {
            j.progress.cancel();
        }
        self.posture.key_job = None;
        self.posture.exact_job = None;
        self.posture.live_job = None;
        self.posture.rec = None;
        if let Some(slot) = self.posture.slot {
            self.close_posture_workspace(slot);
        }
        self.posture.open = false;
    }

    /// Draw the window when it is open.
    pub(super) fn posture_window(&mut self, ctx: &egui::Context) {
        if !self.posture.open {
            return;
        }
        let mut open = true;
        tool_window(
            ctx,
            "posture",
            "🎞 Posture model",
            &mut open,
            WinOpts::size(1500.0, 900.0).no_scroll(),
            |ui| self.posture_body(ui),
        );
        if !open {
            self.close_posture();
        }
    }

    fn posture_body(&mut self, ui: &mut egui::Ui) {
        let full = ui.available_rect_before_wrap();
        ui.expand_to_include_rect(full);
        let top_h = 64.0;
        let bottom_h = 40.0;
        let right_w = (full.width() * 0.34).clamp(320.0, 520.0);
        let top = Rect::from_min_max(full.min, Pos2::new(full.right(), full.top() + top_h));
        let bottom = Rect::from_min_max(Pos2::new(full.left(), full.bottom() - bottom_h), full.max);
        let right = Rect::from_min_max(
            Pos2::new(full.right() - right_w, top.bottom()),
            Pos2::new(full.right(), bottom.top()),
        );
        let centre = Rect::from_min_max(
            Pos2::new(full.left(), top.bottom()),
            Pos2::new(right.left() - 4.0, bottom.top()),
        );
        let child = |ui: &mut egui::Ui, r: Rect, dir: egui::Layout| {
            ui.new_child(egui::UiBuilder::new().max_rect(r.shrink(2.0)).layout(dir))
        };
        let mut top_ui = child(ui, top, egui::Layout::top_down(egui::Align::Min));
        self.posture_top_bar(&mut top_ui);
        let mut right_ui = child(ui, right, egui::Layout::top_down(egui::Align::Min));
        self.posture_side(&mut right_ui);
        let mut bottom_ui = child(ui, bottom, egui::Layout::left_to_right(egui::Align::Center));
        self.posture_bottom_bar(&mut bottom_ui);
        self.posture_panes(ui, centre);
    }

    /// States, anchor, mode, bodies, Build.
    fn posture_top_bar(&mut self, ui: &mut egui::Ui) {
        let busy = self.posture.job.is_some();
        let vols: Vec<usize> = self
            .volume_slot_list()
            .into_iter()
            .filter(|s| !self.posture_owns(*s))
            .collect();
        let common = self.posture_common();
        let summaries: Vec<(usize, String)> = vols
            .iter()
            .map(|&s| {
                (
                    s,
                    format!("{}: {}", SLOT_NAMES[s], self.workspace_summary(s)),
                )
            })
            .collect();
        let mut remove: Option<usize> = None;
        let mut add = false;
        let mut build = false;
        let mut open_file = false;
        let mut cancel = false;
        ui.horizontal_wrapped(|ui| {
            ui.add_enabled_ui(!busy, |ui| {
                ui.label("States:");
                let n = self.posture.states.len();
                self.posture.posture_text.resize(n, String::new());
                for i in 0..n {
                    let cur = self.posture.states[i];
                    let text = summaries
                        .iter()
                        .find(|(s, _)| *s == cur)
                        .map(|(_, t)| t.clone())
                        .unwrap_or_else(|| format!("{}: (no image)", SLOT_NAMES[cur.min(MAX_WORKSPACES - 1)]));
                    let label = ["Start", "Next", "Then", "Then"][i.min(3)];
                    egui::ComboBox::from_id_salt(("posture_state", i))
                        .selected_text(text)
                        .width(180.0)
                        .show_ui(ui, |ui| {
                            for (s, t) in &summaries {
                                ui.selectable_value(&mut self.posture.states[i], *s, t);
                            }
                        })
                        .response
                        .on_hover_text(format!("{label} state: the workspace whose displayed series and structures it is"));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.posture.posture_text[i])
                            .desired_width(44.0)
                            .hint_text("param"),
                    )
                    .tip("Posture parameter of this state (an angle, a day, a filling): the time axis reads in it when every state has one");
                    if n > 2 && small_tip_button(ui, "✖", "Remove this state") {
                        remove = Some(i);
                    }
                }
                if self.posture.states.len() < vols.len() && small_tip_button(ui, "➕", "Add a state: the model becomes a chain") {
                    add = true;
                }
                ui.separator();
                ui.label("Anchor:");
                let anchor_text = match self.posture.anchor_kind {
                    AnchorKind::Structure if !self.posture.anchor_name.is_empty() => self.posture.anchor_name.clone(),
                    AnchorKind::Structure => "(pick a structure)".into(),
                    AnchorKind::Image => "Whole image".into(),
                    AnchorKind::Scanner => "Scanner frame".into(),
                };
                egui::ComboBox::from_id_salt("posture_anchor")
                    .selected_text(anchor_text)
                    .width(140.0)
                    .show_ui(ui, |ui| {
                        for name in &common {
                            let on = self.posture.anchor_kind == AnchorKind::Structure
                                && self.posture.anchor_name.eq_ignore_ascii_case(name);
                            if ui.selectable_label(on, name).clicked() {
                                self.posture.anchor_kind = AnchorKind::Structure;
                                self.posture.anchor_name = name.clone();
                            }
                        }
                        ui.separator();
                        ui.selectable_value(&mut self.posture.anchor_kind, AnchorKind::Image, "Whole image")
                            .on_hover_text("A rigid intensity registration of each state onto the first");
                        ui.selectable_value(&mut self.posture.anchor_kind, AnchorKind::Scanner, "Scanner frame")
                            .on_hover_text("Nothing held still: the scans' own coordinates (for scans that share a frame)");
                    })
                    .response
                    .on_hover_text("What is held still: every number is in its frame, and the animation shows it standing still");
                ui.label("Mode:");
                egui::ComboBox::from_id_salt("posture_mode")
                    .selected_text(self.posture.mode.label())
                    .show_ui(ui, |ui| {
                        for m in Mode::ALL {
                            ui.selectable_value(&mut self.posture.mode, m, m.label())
                                .on_hover_text(m.hint());
                        }
                    })
                    .response
                    .on_hover_text(self.posture.mode.hint());
                let bodies_text = match &self.posture.bodies {
                    None => "Rigid: the bones".to_string(),
                    Some(v) if v.is_empty() => "Rigid: none".to_string(),
                    Some(v) => format!("Rigid: {}", v.len()),
                };
                ui.menu_button(bodies_text, |ui| {
                    ui.weak("Structures moved as rigid bodies");
                    let mut picked: Vec<String> = match &self.posture.bodies {
                        Some(v) => v.clone(),
                        None => common.iter().filter(|n| posture::fit::is_bone(n)).cloned().collect(),
                    };
                    let mut changed = false;
                    egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                        for name in &common {
                            let mut on = picked.iter().any(|p| p.eq_ignore_ascii_case(name));
                            if ui.checkbox(&mut on, name).changed() {
                                changed = true;
                                if on {
                                    picked.push(name.clone());
                                } else {
                                    picked.retain(|p| !p.eq_ignore_ascii_case(name));
                                }
                            }
                        }
                    });
                    if changed {
                        self.posture.bodies = Some(picked);
                    }
                    if ui.button("The bones (default)").clicked() {
                        self.posture.bodies = None;
                    }
                });
                ui.checkbox(&mut self.posture.residual, "Residual")
                    .on_hover_text("A B-spline registration on top of the bodies, for what they leave: organ filling, soft tissue");
                ui.add(egui::DragValue::new(&mut self.posture.lattice_mm).range(2.0..=20.0).suffix(" mm"))
                    .tip("Lattice of the velocity fields");
                ui.add(egui::DragValue::new(&mut self.posture.reach_mm).range(5.0..=150.0).suffix(" mm"))
                    .tip("Reach of a rigid body's influence on the tissue around it");
                if self.posture.states.len() >= 3 {
                    ui.checkbox(&mut self.posture.smooth, "Smooth chain")
                        .on_hover_text("Every rigid body on a curve through the states, its velocity continuous at each, rather than a constant turn per pair");
                }
                ui.checkbox(&mut self.posture.wepl, "WEPL")
                    .on_hover_text("The water-equivalent path length to each structure's centroid along a beam, in the tables");
                if self.posture.wepl {
                    ui.add(egui::DragValue::new(&mut self.posture.gantry_deg).range(-360.0..=360.0).suffix(" deg"))
                        .tip("Gantry angle of the WEPL beam (IEC 61217, in the first state's patient position)");
                    ui.add(egui::DragValue::new(&mut self.posture.couch_deg).range(-360.0..=360.0).suffix(" deg"))
                        .tip("Patient support angle of the WEPL beam");
                }
            });
            ui.separator();
            if busy {
                if let Some(job) = &self.posture.job {
                    ui.spinner();
                    ui.add(egui::ProgressBar::new(job.progress.frac()).desired_width(160.0).show_percentage());
                    ui.label(job.progress.get());
                }
                if ui.button("Cancel").clicked() {
                    cancel = true;
                }
            } else {
                let ready = self.posture.states.len() >= 2
                    && self.posture.states.iter().all(|s| vols.contains(s))
                    && self.posture.states.windows(2).all(|w| w[0] != w[1]);
                if enabled_tip_button(ui, ready, "▶ Build", "Fit the model between the states") {
                    build = true;
                }
                if enabled_tip_button(ui, ready, "📂 Open model", "Read a saved model back onto these states, nothing fitted again") {
                    open_file = true;
                }
            }
        });
        ui.weak(
            "An interpolation between what was scanned: for looking at and measuring along the \
             path, not for planning a treatment on.",
        );
        if !busy {
            self.posture_offers_ui(ui);
        }
        if let Some(i) = remove {
            self.posture.states.remove(i);
            self.posture.posture_text.remove(i);
        }
        if add {
            if let Some(next) = vols.iter().find(|s| !self.posture.states.contains(s)) {
                self.posture.states.push(*next);
                self.posture.posture_text.push(String::new());
            }
        }
        if cancel {
            if let Some(j) = &self.posture.job {
                j.progress.cancel();
            }
        }
        if build {
            self.posture_build(None);
        }
        if open_file {
            self.ask_file(
                "Open a posture model",
                None,
                Some(pick::Filter {
                    name: "Posture model",
                    exts: &[pfile::EXTENSION],
                }),
                |app, path| app.posture_build(Some(path)),
            );
        }
    }

    /// The three planes, the 3D scene and the chart of the model workspace.
    fn posture_panes(&mut self, ui: &mut egui::Ui, area: Rect) {
        let painter = ui.painter_at(area);
        painter.rect_filled(area, 0.0, backdrop_color(ui.visuals()));
        let Some(slot) = self.posture.slot.filter(|s| self.slots[*s].study.is_some()) else {
            let msg = if self.posture.job.is_some() {
                "Building the model..."
            } else {
                "Pick the states (workspaces with an image of the same patient in two positions), \
                 the anchor and the mode, and press Build."
            };
            painter.text(
                area.center(),
                Align2::CENTER_CENTER,
                msg,
                FontId::proportional(14.0),
                ui.visuals().weak_text_color(),
            );
            return;
        };
        let gap = 2.0;
        let h_top = (area.height() * 0.5).floor();
        let w3 = (area.width() - 2.0 * gap) / 3.0;
        let planes = [ViewPlane::Axial, ViewPlane::Sagittal, ViewPlane::Coronal];
        let mut rects: Vec<(PaneKind, Rect)> = Vec::new();
        for (i, p) in planes.iter().enumerate() {
            let x = area.left() + i as f32 * (w3 + gap);
            rects.push((
                PaneKind::Plane(*p),
                Rect::from_min_size(Pos2::new(x, area.top()), Vec2::new(w3, h_top - gap)),
            ));
        }
        let lower = Rect::from_min_max(Pos2::new(area.left(), area.top() + h_top), area.max);
        let chart = self.posture.chart.is_some() && self.chart_source(slot).is_some();
        if chart {
            let w = (lower.width() - gap) * 0.5;
            rects.push((
                PaneKind::Scene3d,
                Rect::from_min_size(lower.min, Vec2::new(w, lower.height())),
            ));
            rects.push((
                PaneKind::Chart,
                Rect::from_min_max(Pos2::new(lower.left() + w + gap, lower.top()), lower.max),
            ));
        } else {
            rects.push((PaneKind::Scene3d, lower));
        }
        for (i, (kind, r)) in rects.iter().enumerate() {
            self.pane(ui, slot, *kind, *r, i == 0);
        }
        // The selected view (what a recording films, and *Save GIF* takes
        // alone) is the pane last clicked here, with any button.
        let pressed = ui.input(|i| {
            i.pointer
                .any_pressed()
                .then(|| i.pointer.interact_pos())
                .flatten()
        });
        if let Some(at) = pressed {
            if let Some((k, _)) = rects.iter().find(|(k, r)| {
                r.contains(at) && matches!(k, PaneKind::Plane(_) | PaneKind::Scene3d)
            }) {
                self.posture.film = *k;
            }
        }
        if let Some((_, r)) = rects.first() {
            self.posture.pane_aspect = Some(r.width() / r.height().max(1.0));
        }
        let film = rects
            .iter()
            .find(|(k, _)| *k == self.posture.film)
            .map(|(_, r)| *r)
            .unwrap_or(area);
        self.posture_record_tick(ui, film);
    }

    /// Transport, time, show, overlay, recording.
    fn posture_bottom_bar(&mut self, ui: &mut egui::Ui) {
        let Some((span, ticks, param)) = self.posture.built.as_ref().map(|b| {
            (
                b.model.span(),
                b.model.states.len(),
                b.model.param_at(self.posture.t),
            )
        }) else {
            ui.weak("No model yet.");
            return;
        };
        let slot = self.posture.slot.unwrap_or(0);
        let now = ui.input(|i| i.time);
        let target = play::PlayTarget::Model { slot };
        let playing = self.is_playing(target);
        if small_tip_button(ui, "⏮", "To the first state") {
            self.posture_set_t(0.0, now);
        }
        let ready = self.posture_times().len() >= 2;
        if tip_widget(
            ui,
            ready,
            egui::Button::new(if playing { "⏸" } else { "▶" }),
            if playing { "Pause" } else { "Play the path" },
        ) {
            self.play.from_pane = None;
            self.toggle_play(target, now);
        }
        if small_tip_button(ui, "⏭", "To the last state") {
            self.posture_set_t(span, now);
        }
        let mut t = self.posture.t;
        let saved = ui.spacing().slider_width;
        ui.spacing_mut().slider_width = (ui.available_width() - 640.0).max(160.0);
        let resp = ui
            .add(egui::Slider::new(&mut t, 0.0..=span).show_value(false))
            .named("Time along the path");
        ui.spacing_mut().slider_width = saved;
        // A tick at each state, under the slider.
        let r = resp.rect;
        let stroke = Stroke::new(1.5, ui.visuals().strong_text_color());
        for k in 0..ticks {
            let x = r.left() + 6.0 + (r.width() - 12.0) * k as f32 / (ticks - 1).max(1) as f32;
            ui.painter().line_segment(
                [
                    Pos2::new(x, r.bottom() - 3.0),
                    Pos2::new(x, r.bottom() + 2.0),
                ],
                stroke,
            );
        }
        if resp.changed() {
            if playing {
                self.stop_play();
            }
            self.posture_set_t(t, now);
        }
        ui.label(match param {
            Some(p) => format!("t {:.2} ({p:.1})", self.posture.t),
            None => format!("t {:.2}", self.posture.t),
        });
        ui.add(
            egui::DragValue::new(&mut self.posture.fps)
                .range(0.5..=30.0)
                .speed(0.1)
                .suffix(" fps"),
        )
        .tip("Key frames a second");
        egui::ComboBox::from_id_salt("posture_end")
            .selected_text(self.play.mode.label())
            .width(70.0)
            .show_ui(ui, |ui| {
                for m in play::PlayMode::ALL {
                    ui.selectable_value(&mut self.play.mode, m, m.label())
                        .on_hover_text(m.hint());
                }
            })
            .response
            .on_hover_text("What a run does at the end");
        let before = self.posture.show;
        egui::ComboBox::from_id_salt("posture_show")
            .selected_text(self.posture.show.label())
            .width(130.0)
            .show_ui(ui, |ui| {
                for s in Show::ALL {
                    ui.selectable_value(&mut self.posture.show, s, s.label())
                        .on_hover_text(s.hint());
                }
            })
            .response
            .on_hover_text(self.posture.show.hint());
        if self.posture.show != before {
            self.posture.keys.clear();
            if !self.posture.live {
                self.posture_start_keys();
            }
            self.posture.moved_at = now;
        }
        let before = self.posture.overlay;
        egui::ComboBox::from_id_salt("posture_overlay")
            .selected_text(match self.posture.overlay {
                FieldStyle::None => "No overlay",
                FieldStyle::Arrows => "Arrows",
                FieldStyle::Grid => "Deformed grid",
            })
            .width(110.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.posture.overlay, FieldStyle::None, "No overlay");
                ui.selectable_value(&mut self.posture.overlay, FieldStyle::Arrows, "Arrows");
                ui.selectable_value(&mut self.posture.overlay, FieldStyle::Grid, "Deformed grid");
            })
            .response
            .on_hover_text("The displacement from the first state, over the views");
        if self.posture.overlay != before {
            // Shown again with the field (or without it).
            self.posture.shown = None;
        }
        // The extras: the wash and the dose; a change renders again.
        let (wash, dose) = (self.posture.wash, self.posture.dose_on);
        ui.checkbox(&mut self.posture.wash, "Jacobian wash")
            .on_hover_text(
                "The local volume change since the first state as a colour over the image: blue \
             compressed, red expanded (full colour at half or twice the volume), magenta folded",
            );
        let has_dose = self
            .posture
            .built
            .as_ref()
            .is_some_and(|b| b.dose.is_some());
        if has_dose {
            ui.checkbox(&mut self.posture.dose_on, "Dose (display only)").on_hover_text(
                "The first state's dose drawn where its tissue has moved along the path. Display \
                 only: it is not recalculated on the new anatomy, and no dose number or DVH comes \
                 from it.",
            );
        }
        if (wash, dose) != (self.posture.wash, self.posture.dose_on) {
            self.posture.shown = None;
            self.posture.keys.clear();
            if !self.posture.live {
                self.posture_start_keys();
            }
        }
        ui.separator();
        let was_live = self.posture.live;
        egui::ComboBox::from_id_salt("posture_play_from")
            .selected_text(if self.posture.live {
                "Live slices"
            } else {
                "Key frames"
            })
            .width(90.0)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.posture.live, true, "Live slices")
                    .on_hover_text("Each step computes the three planes on screen at its exact time, on the full grid (on the GPU when there is one)");
                ui.selectable_value(&mut self.posture.live, false, "Key frames")
                    .on_hover_text("Whole volumes at evenly spaced times on a working grid that fits the playback budget, rendered once and played from memory");
            })
            .response
            .on_hover_text(
                self.posture
                    .built
                    .as_ref()
                    .map_or_else(String::new, |b| b.slicer_note.clone()),
            );
        if self.posture.live != was_live {
            self.posture.shown = None;
            if !self.posture.live && self.posture.keys.is_empty() {
                self.posture_start_keys();
            }
        }
        if self.posture.live {
            if let Some(n) = &self.posture.live_note {
                ui.weak(n.as_str());
            }
        }
        ui.add(
            egui::DragValue::new(&mut self.posture.frames)
                .range(2..=200)
                .suffix(" frames"),
        )
        .tip("Steps along the path a run takes (key frames: rendered once, played from memory)");
        if !self.posture.live {
            if ui
                .small_button("↺")
                .tip("Render the key frames again at this count")
                .clicked()
            {
                self.posture.keys.clear();
                self.posture_start_keys();
            }
            if let Some(j) = &self.posture.key_job {
                ui.add(egui::ProgressBar::new(j.progress.frac()).desired_width(60.0))
                    .tip("Rendering the key frames");
            }
        }
        ui.separator();
        let films = [
            (PaneKind::Plane(ViewPlane::Axial), "Axial"),
            (PaneKind::Plane(ViewPlane::Sagittal), "Sagittal"),
            (PaneKind::Plane(ViewPlane::Coronal), "Coronal"),
            (PaneKind::Scene3d, "3D"),
        ];
        egui::ComboBox::from_id_salt("posture_film")
            .selected_text(
                films
                    .iter()
                    .find(|f| f.0 == self.posture.film)
                    .map_or("Axial", |f| f.1),
            )
            .width(70.0)
            .show_ui(ui, |ui| {
                for (k, l) in films {
                    ui.selectable_value(&mut self.posture.film, k, l);
                }
            })
            .response
            .on_hover_text(
                "The selected view: the pane a recording films and Save GIF takes alone. A click \
                 in a pane selects it too.",
            );
        let armed = self.posture.rec.is_some();
        if ui
            .add(egui::Button::selectable(armed, if armed { "⏹" } else { "⏺" }))
            .tip(if armed {
                "Stop recording and write what was taken"
            } else {
                "Record: the next run is filmed, one cycle, as GIF or PNG frames (the Image playback module's format)"
            })
            .clicked()
        {
            if armed {
                self.posture_finish_recording(false);
            } else {
                self.posture_arm_recording();
            }
        }
        if let Some(s) = &self.posture.rec_status {
            ui.weak(s.as_str());
        }
    }

    /// The plane the selected view shows, if it is a plane.
    fn posture_film_plane(&self) -> Option<ViewPlane> {
        match self.posture.film {
            PaneKind::Plane(p) => Some(p),
            _ => None,
        }
    }

    /// *Save GIF*: its choices in a menu, or the job writing it.
    fn posture_gif_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(j) = &self.posture.gif_job {
            ui.add(
                egui::ProgressBar::new(j.progress.frac())
                    .desired_width(70.0)
                    .show_percentage(),
            )
            .tip("Writing the GIF, frame by frame");
            if ui.small_button("Cancel").clicked() {
                j.progress.cancel();
            }
            return;
        }
        let selected = self.posture_film_plane();
        let mut save = false;
        // Choices, like the View menu: setting one leaves the menu open; it
        // closes on a click outside it, or on *Save...*.
        let (menu, _) = egui::containers::menu::MenuButton::new("Save GIF")
            .config(
                egui::containers::menu::MenuConfig::new()
                    .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
            )
            .ui(ui, |ui| {
                ui.weak("The path as an animated GIF, every frame rendered from the model");
                ui.radio_value(&mut self.posture.gif_all, true, "All views")
                    .on_hover_text(
                        "Axial, sagittal and coronal side by side, as the window has them",
                    );
                let name = selected.map_or("3D", |p| p.title());
                ui.add_enabled_ui(selected.is_some(), |ui| {
                    ui.radio_value(
                    &mut self.posture.gif_all,
                    false,
                    format!("Selected view only ({name})"),
                )
                .on_hover_text(
                    "The pane last clicked in this window (or picked in the selected-view list \
                     beside the record button)",
                )
                .on_disabled_hover_text(
                    "The 3D scene is not rendered into a GIF: click a plane, or record the 3D \
                     pane from the screen",
                );
                });
                ui.horizontal(|ui| {
                    ui.label("Width");
                    ui.add(
                        egui::DragValue::new(&mut self.posture.gif_width)
                            .range(pfilm::MIN_WIDTH..=pfilm::MAX_WIDTH)
                            .speed(4.0)
                            .suffix(" px"),
                    )
                    .tip("Width of the GIF in pixels; the height follows the panes' shape");
                });
                ui.checkbox(&mut self.posture.gif_bounce, "There and back")
                    .on_hover_text("The way back added after the way there, so the loop is smooth");
                ui.weak(format!(
                    "{} frames at {:.1} fps (the bar's), {}, the structures outlined{}",
                    self.posture.frames.clamp(2, 200),
                    self.posture.fps,
                    self.posture.show.label(),
                    if self.posture.wash {
                        ", with the Jacobian wash"
                    } else {
                        ""
                    }
                ));
                let ready = self.posture.gif_all || selected.is_some();
                if ui
                    .add_enabled(ready, egui::Button::new("Save..."))
                    .on_hover_text("Choose the file and write the GIF in the background")
                    .clicked()
                {
                    save = true;
                    ui.close();
                }
            });
        menu.on_hover_text(
            "Save the animation as a GIF: all views in one, or the selected view only",
        );
        if let Some(s) = &self.posture.gif_status {
            ui.weak(s.as_str());
        }
        if save {
            self.ask_save(
                "Save the animation as a GIF",
                "posture.gif",
                None,
                Some(pick::Filter {
                    name: "GIF",
                    exts: &["gif"],
                }),
                |app, path| app.posture_start_gif(path),
            );
        }
    }

    /// Write the GIF to `dest` on a background job: the times of a run,
    /// each frame rendered exactly on the full grid ([`pfilm::write_gif`]).
    pub(super) fn posture_start_gif(&mut self, dest: std::path::PathBuf) {
        let (Some(b), Some(slot)) = (self.posture.built.as_ref(), self.posture.slot) else {
            return;
        };
        let views = match (self.posture.gif_all, self.posture_film_plane()) {
            (false, Some(p)) => pfilm::FilmViews::One(p),
            _ => pfilm::FilmViews::All,
        };
        // The views' slices on the full grid (a key frame's volume may be
        // on a coarser one of the same extent).
        let full = b.model.grid.dims;
        let cur = self.slots[slot]
            .study
            .as_ref()
            .map_or(full, |s| s.volume.dims);
        let mut slices = [full[0] / 2, full[1] / 2, full[2] / 2];
        for v in self.slots[slot].views.iter().take(3) {
            let axis = Plane::of_view(v.plane, v.slice).axis;
            let s = if cur[axis] > 1 && cur[axis] != full[axis] {
                (v.slice as f64 * (full[axis] - 1) as f64 / (cur[axis] - 1) as f64).round() as usize
            } else {
                v.slice
            };
            slices[axis] = s.min(full[axis].saturating_sub(1));
        }
        // The structures ticked visible, from the first state's set.
        let visible = &self.slots[slot].roi_visible;
        let structures = b.base_set.as_ref().map(|ss| {
            let mut ss = ss.clone();
            let mut i = 0;
            ss.rois.retain(|_| {
                let keep = visible.get(i).copied().unwrap_or(true);
                i += 1;
                keep
            });
            ss
        });
        let params = pfilm::FilmParams {
            views,
            slices,
            frames: self.posture.frames.clamp(2, 200),
            fps: self.posture.fps,
            bounce: self.posture.gif_bounce,
            show: self.posture.show,
            window: (self.window_center[slot], self.window_width[slot]),
            width: self.posture.gif_width,
            cell_aspect: self.posture.pane_aspect,
            outlines: true,
            wash: self.posture.wash,
        };
        let model = b.model.clone();
        let images = b.images.clone();
        let slicer = b.slicer.clone();
        self.posture.gif_status = None;
        let progress = Arc::new(Progress::default());
        progress.set("Writing the GIF");
        self.posture.gif_job = Some(Job::spawn(progress, move |p| {
            pfilm::write_gif(
                &dest,
                &model,
                &images,
                slicer.as_deref(),
                structures.as_ref(),
                &params,
                p,
            )
            .map(|s| (dest, s))
        }));
    }

    /// The states that lack structures another has, and the automatic run
    /// that would make them (see the module doc).
    fn posture_offers_ui(&mut self, ui: &mut egui::Ui) {
        let names: Vec<Vec<String>> = self
            .posture
            .states
            .iter()
            .map(|&s| {
                self.slots
                    .get(s)
                    .and_then(|x| x.study.as_ref())
                    .map(structure_names)
                    .unwrap_or_default()
            })
            .collect();
        let key = names
            .iter()
            .zip(&self.posture.states)
            .map(|(n, s)| format!("{s}:{}", n.join("|")))
            .collect::<Vec<_>>()
            .join("/");
        if key != self.posture.offer_key {
            let lacking = wp::lacking(&names);
            self.posture.offers = self
                .posture
                .states
                .iter()
                .zip(lacking)
                .filter(|(_, l)| !l.is_empty())
                .map(|(&slot, l)| {
                    let modality = self.slot_modality(slot);
                    Offer {
                        slot,
                        model: wp::model_for_missing(&l, &modality),
                        lacking: l,
                    }
                })
                .collect();
            self.posture.offer_key = key;
        }
        let busy = self.autoseg_job.is_some();
        let mut run: Option<(usize, zoo::AutoModel)> = None;
        for o in &self.posture.offers {
            let letter = SLOT_NAMES[o.slot.min(MAX_WORKSPACES - 1)];
            ui.horizontal_wrapped(|ui| {
                let shown: Vec<&str> = o.lacking.iter().take(6).map(String::as_str).collect();
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!(
                        "{letter} lacks {} structure{} the other states have: {}{}",
                        o.lacking.len(),
                        if o.lacking.len() == 1 { "" } else { "s" },
                        shown.join(", "),
                        if o.lacking.len() > shown.len() { ", ..." } else { "" }
                    ),
                );
                match &o.model {
                    Some((m, hits)) => {
                        if enabled_tip_button(
                            ui,
                            !busy,
                            format!("Contour {letter} with {}", m.label()),
                            format!(
                                "Run {} on workspace {letter}: it makes {} of them ({}). Its results come \
                                 up in the Structure auto tools module; add them there, then press Build again.",
                                m.label(),
                                hits.len(),
                                hits.join(", ")
                            ),
                        ) {
                            run = Some((o.slot, *m));
                        }
                    }
                    None => {
                        ui.weak("No automatic model makes these: contour them, or the model goes without them.");
                    }
                }
            });
        }
        if let Some((slot, m)) = run {
            self.posture_run_model(slot, m);
        }
    }

    /// Run an automatic model on a state's workspace, through the
    /// Structure auto tools module.
    fn posture_run_model(&mut self, slot: usize, model: zoo::AutoModel) {
        self.auto.slot = slot;
        self.open_autoseg_dialog(slot);
        let ready = match self.autoseg_dialog.as_mut() {
            Some(d) if d.slot == slot => {
                d.model = model;
                d.parts = vec![true; model.part_names().len()];
                if let Some(v) = model.total_variant() {
                    d.total = v;
                }
                true
            }
            _ => false,
        };
        self.show_module(Module::AutoTools);
        if ready {
            self.start_autoseg();
            self.notice = Some(format!(
                "Running {} on workspace {}: its results come up in the Structure auto tools \
                 module; add them there, then press Build again.",
                model.label(),
                SLOT_NAMES[slot.min(MAX_WORKSPACES - 1)]
            ));
        }
    }

    /// Arm a recording of the next model run.
    fn posture_arm_recording(&mut self) {
        let arm = |app: &mut ViewerApp, dest: std::path::PathBuf| {
            app.posture.rec_status = Some("Armed: press ▶".into());
            app.posture.rec = Some(record::Recording {
                format: app.rec_format,
                dest,
                bound: None,
                frames: Vec::new(),
                max_frames: app.rec_max.max(1),
                pending: false,
                at: u64::MAX,
            });
        };
        match self.rec_format {
            record::RecFormat::Gif => {
                self.ask_save("Save the recording", "posture.gif", None, None, arm)
            }
            record::RecFormat::Pngs => self.ask_folder("Folder for the recorded frames", arm),
        }
    }

    /// The recorder on this window's own pictures: latch onto a model run,
    /// take one picture per key frame, stop after one cycle.
    fn posture_record_tick(&mut self, ui: &egui::Ui, film: Rect) {
        if self.posture.rec.is_none() {
            return;
        }
        let shot = ui.input(|i| {
            i.events.iter().rev().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        let ppp = ui.ctx().pixels_per_point();
        let running = self.posture_playing();
        let pos = self.posture_position();
        let frame_no = self.play.frame_no;
        let fps = self.posture.fps;
        let mut finish: Option<bool> = None;
        let mut want = false;
        if let Some(rec) = self.posture.rec.as_mut() {
            if let (Some(shot), true) = (shot, rec.pending) {
                rec.pending = false;
                if let Some(b) = rec.bound.as_mut() {
                    if let Some(f) = record::cut(&shot, b.crop) {
                        rec.frames.push(f);
                        match (b.start, pos) {
                            (None, p) => b.start = p,
                            (Some(s), Some(p)) if p != s => b.left = true,
                            (Some(s), Some(p)) if p == s && b.left => {
                                rec.frames.pop();
                                finish = Some(false);
                            }
                            _ => {}
                        }
                    }
                }
            }
            match rec.bound {
                None if running => {
                    rec.bound = Some(record::Bound {
                        target: play::PlayTarget::Model {
                            slot: self.posture.slot.unwrap_or(0),
                        },
                        crop: [
                            (film.left() * ppp).round().max(0.0) as usize,
                            (film.top() * ppp).round().max(0.0) as usize,
                            (film.width() * ppp).round().max(1.0) as usize,
                            (film.height() * ppp).round().max(1.0) as usize,
                        ],
                        fps,
                        start: None,
                        left: false,
                    });
                    rec.at = u64::MAX;
                }
                Some(_) if !running => finish = finish.or(Some(false)),
                Some(_) if rec.frames.len() >= rec.max_frames => finish = Some(true),
                _ => {}
            }
            if finish.is_none() && rec.bound.is_some() && !rec.pending && rec.at != frame_no {
                rec.at = frame_no;
                rec.pending = true;
                want = true;
            }
        }
        if let Some(capped) = finish {
            self.posture_finish_recording(capped);
            return;
        }
        if want {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
            ui.ctx().request_repaint();
        }
    }

    fn posture_finish_recording(&mut self, capped: bool) {
        let Some(rec) = self.posture.rec.take() else {
            return;
        };
        if rec.frames.is_empty() {
            self.posture.rec_status = Some("Nothing was recorded".into());
            return;
        }
        let fps = rec.bound.as_ref().map_or(self.posture.fps, |b| b.fps);
        let n = rec.frames.len();
        let res = match rec.format {
            record::RecFormat::Gif => record::write_gif(&rec.dest, &rec.frames, fps).map(|()| n),
            record::RecFormat::Pngs => record::write_pngs(&rec.dest, &rec.frames),
        };
        self.posture.rec_status = Some(match res {
            Ok(n) => format!(
                "{n} frames written to {}{}",
                rec.dest.display(),
                if capped { " (frame limit)" } else { "" }
            ),
            Err(e) => format!("Could not write the recording: {e:#}"),
        });
    }

    /// The right side: what was built, the exports and the table.
    fn posture_side(&mut self, ui: &mut egui::Ui) {
        if let Some(s) = &self.posture.status {
            ui.weak(s.as_str());
        }
        let Some(b) = self.posture.built.as_ref() else {
            ui.label("No model yet.");
            return;
        };
        for w in &b.warnings {
            ui.colored_label(ui.visuals().warn_fg_color, w);
        }
        let busy = self.posture.export_job.is_some();
        let mut act: Option<u8> = None;
        ui.horizontal_wrapped(|ui| {
            if enabled_tip_button(ui, !busy, "💾 CSV", "The four tables as one CSV file") {
                act = Some(0);
            }
            if enabled_tip_button(ui, !busy, "💾 Model", "The model as a file, to reopen without fitting again") {
                act = Some(1);
            }
            if enabled_tip_button(
                ui,
                !busy,
                "📤 REG",
                "The whole path, first state to last, as a DICOM Deformable Spatial Registration",
            ) {
                act = Some(2);
            }
            if enabled_tip_button(
                ui,
                !busy,
                "📤 Image at t",
                format!(
                    "The image and the structures at the slider's time as a derived CT series and an \
                     RT Structure Set, marked as {}",
                    wp::CAVEAT
                ),
            ) {
                act = Some(3);
            }
        });
        match act {
            Some(0) => {
                let text = b.tables.csv();
                self.ask_save(
                    "Save the posture tables",
                    "posture_tables.csv",
                    None,
                    Some(CSV_FILES),
                    move |app, path| match std::fs::write(
                        &path,
                        crate::motion::csv_file_bytes(&text),
                    ) {
                        Ok(()) => app.notice = Some(format!("Written: {}", path.display())),
                        Err(e) => {
                            app.error = Some(format!("Could not write {}: {e}", path.display()))
                        }
                    },
                );
            }
            Some(1) => {
                let model = b.model.clone();
                let occ = b.occupancy.clone();
                self.ask_save(
                    "Save the posture model",
                    format!("posture.{}", pfile::EXTENSION),
                    None,
                    Some(pick::Filter {
                        name: "Posture model",
                        exts: &[pfile::EXTENSION],
                    }),
                    move |app, path| match pfile::write(&path, &model, &occ) {
                        Ok(()) => app.notice = Some(format!("Written: {}", path.display())),
                        Err(e) => app.error = Some(format!("{e:#}")),
                    },
                );
            }
            Some(2) => {
                let model = b.model.clone();
                let first = b.first.clone();
                let last_for = b.last_for.clone();
                self.ask_save(
                    "Save the deformation as DICOM",
                    "posture_reg.dcm",
                    None,
                    None,
                    move |app, path| match wp::export_registration(&model, &first, &last_for, &path)
                    {
                        Ok(()) => app.notice = Some(format!("Written: {}", path.display())),
                        Err(e) => app.error = Some(format!("{e:#}")),
                    },
                );
            }
            Some(3) => {
                let model = b.model.clone();
                let images = b.images.clone();
                let first = b.first.clone();
                let (t, show) = (self.posture.t, self.posture.show);
                self.ask_folder("Folder for the image at this time", move |app, dir| {
                    let progress = Arc::new(Progress::default());
                    app.posture.export_job = Some(Job::spawn(progress, move |p| {
                        let e =
                            wp::export_frame(&model, &images, &first, Some(0), t, show, &dir, p)?;
                        Ok(format!(
                            "t = {t:.2}: {} images and {} structure set(s) written to {}",
                            e.image_files.len(),
                            e.structure_files.len(),
                            dir.display()
                        ))
                    }));
                });
            }
            _ => {}
        }
        // The animation as a GIF: all views in one, or the selected one.
        ui.horizontal_wrapped(|ui| self.posture_gif_ui(ui));
        ui.separator();
        egui::ScrollArea::both()
            .id_salt("posture_table")
            .auto_shrink([false, false])
            .show(ui, |ui| self.posture_table(ui));
    }
}

/// A series to stand in when a study has none (never the case for a state,
/// which has an image).
trait OrDefaultSeries {
    fn unwrap_or_default_series(self) -> loader::SeriesInfo;
}

impl OrDefaultSeries for Option<loader::SeriesInfo> {
    fn unwrap_or_default_series(self) -> loader::SeriesInfo {
        self.unwrap_or_else(|| loader::SeriesInfo {
            uid: crate::dicom_export::new_uid(),
            modality: "CT".into(),
            description: String::new(),
            patient_id: String::new(),
            patient_name: String::new(),
            study_uid: String::new(),
            study_date: String::new(),
            study_description: String::new(),
            study_id: String::new(),
            series_number: None,
            temporal_id: None,
            suv_bw: None,
            frame_of_reference_uid: String::new(),
            series_datetime: String::new(),
            files: Vec::new(),
            fourd_head: None,
            z_range: None,
        })
    }
}
