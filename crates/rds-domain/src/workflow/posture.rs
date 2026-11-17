//! The posture model built from studies: the one entry the posture window,
//! `examples/posture_cli.rs`, the MCP tool and the workflow step call.
//!
//! [`build`] takes two or more states (a study each, with the structure
//! set whose contours count) and a [`PostureRequest`], and does what
//! [`crate::posture`] describes, in this order:
//!
//! 1. the structures every state has, by name (case and surrounding spaces
//!    ignored); each rasterized on its state's own grid, coarsened to the
//!    analysis spacing ([`crate::posture::coarsen`]), and counted on the full
//!    grid for its volume;
//! 2. the anchor: each later state laid onto the first by the anchor
//!    structure's surfaces, or by a rigid intensity registration, or not at
//!    all;
//! 3. every state's image and structures carried into the anchor frame, on
//!    the first state's analysis grid, with the part of it each scan covers;
//! 4. a segment per neighbouring pair ([`crate::posture::fit::fit_segment`]),
//!    the bodies the named ones or the bones every state has, the anchor
//!    among them standing still;
//! 5. a chain of three or more states put on curves through its states
//!    ([`crate::posture::chain`]), unless asked not to;
//! 6. the tables ([`crate::posture::measure`]), with the WEPL column along
//!    a beam when one is given.
//!
//! [`lacking`] and [`model_for_missing`] are what the window offers when a
//! state lacks structures another has: the automatic model of the registry
//! that makes most of them, to run on that state before building.
//!
//! Masks are kept as their boxes throughout ([`BoxedMask`]): 49 structures
//! on two states of a 512 x 512 x 274 study would otherwise be gigabytes.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::loader::LoadedStudy;
use crate::posture::fit::{self, BodyInput, FitOptions};
use crate::posture::measure::{self, StructureInput, Tables};
use crate::posture::se3::Rigid;
use crate::posture::svf::{Field, Lattice};
use crate::posture::{self, apply, coarsen, Anchor, Mode, PostureModel, StateInfo};
use crate::progress::{self, Progress, ProgressSink};
use crate::propagate::BoxedMask;
use crate::registration::{self, RegMethod, RegParams, Transform3};
use crate::rtstruct::Roi;
use crate::segmentation;
use crate::volume::{Grid, Volume};

/// One state: a study and the structure set that counts.
pub struct StateInput<'a> {
    pub label: String,
    pub study: &'a LoadedStudy,
    /// Index into `study.structure_sets`; `None` for the one drawn on the
    /// displayed series, else the first.
    pub structure_set: Option<usize>,
    pub posture: Option<f64>,
}

/// Which structures are rigid bodies.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum Bodies {
    /// The bones every state has ([`fit::is_bone`]), and the anchor.
    #[default]
    Bones,
    /// These, where every state has them.
    Named(Vec<String>),
    /// None: the residual (or nothing) carries everything.
    None,
}

/// How the model is built.
#[derive(Clone)]
pub struct PostureRequest {
    pub anchor: Anchor,
    pub bodies: Bodies,
    pub mode: Mode,
    /// Velocity lattice spacing, mm.
    pub lattice_mm: f64,
    /// Analysis grid: the first state's coarsened to at least this, mm.
    pub analysis_mm: f64,
    /// Reach of a body in the blend, mm.
    pub reach_mm: f64,
    /// The residual registration; `None` for none.
    pub residual: Option<RegParams>,
    /// For the linear mode with two states: the first state's patient
    /// coordinates to the second's (a registration the station holds).
    pub linear: Option<Arc<Transform3>>,
    /// Joints as (name, first body, second body); empty for the presets.
    pub joints: Vec<(String, String, String)>,
    /// Time samples per segment in the tables.
    pub samples_per_segment: usize,
    /// The beam the tables' WEPL column is along; `None` for no column.
    pub wepl: Option<WeplBeam>,
    /// A chain of three or more states: bodies on curves through the
    /// states rather than a constant turn per segment.
    pub smooth_chain: bool,
}

/// The WEPL column's beam, in IEC 61217 angles read in the first state's
/// patient position.
#[derive(Clone, Debug)]
pub struct WeplBeam {
    pub gantry_deg: f64,
    pub couch_deg: f64,
    /// HU to relative stopping power; `None`, the generic curve.
    pub hlut: Option<crate::sct::eval::Hlut>,
}

impl WeplBeam {
    fn beam(&self, position: &str) -> measure::Beam {
        let mut b = measure::Beam::iec(self.gantry_deg, self.couch_deg, position);
        if let Some(h) = &self.hlut {
            b.hlut = h.clone();
        }
        b
    }
}

impl Default for PostureRequest {
    fn default() -> Self {
        PostureRequest {
            anchor: Anchor::Image,
            bodies: Bodies::Bones,
            mode: Mode::Diffeomorphic,
            lattice_mm: 6.0,
            analysis_mm: 2.0,
            reach_mm: 40.0,
            residual: None,
            linear: None,
            joints: Vec::new(),
            samples_per_segment: 10,
            wepl: None,
            smooth_chain: true,
        }
    }
}

/// Per state, the structures another state has and it lacks, by name (case
/// and surrounding spaces ignored), in the order they are first met.
pub fn lacking(names: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut all: Vec<String> = Vec::new();
    for list in names {
        for n in list {
            if !all.iter().any(|a| key(a) == key(n)) {
                all.push(n.clone());
            }
        }
    }
    names
        .iter()
        .map(|list| {
            all.iter()
                .filter(|a| !list.iter().any(|n| key(n) == key(a)))
                .cloned()
                .collect()
        })
        .collect()
}

/// The automatic model of the registry ([`crate::zoo::AutoModel`]) for a
/// series of `modality` whose classes are the most of `missing` by name,
/// and those names: what to run on a state that lacks them. Ties go to the
/// model with fewer classes (the smaller run). `None` when no model makes
/// any of them.
pub fn model_for_missing(
    missing: &[String],
    modality: &str,
) -> Option<(crate::zoo::AutoModel, Vec<String>)> {
    let mut best: Option<(crate::zoo::AutoModel, Vec<String>)> = None;
    for m in crate::zoo::AutoModel::all() {
        if !m.modality().accepts(modality) {
            continue;
        }
        let classes = m.classes();
        let hits: Vec<String> = missing
            .iter()
            .filter(|n| classes.iter().any(|c| key(c) == key(n)))
            .cloned()
            .collect();
        if hits.is_empty() {
            continue;
        }
        let better = match &best {
            None => true,
            Some((b, h)) => {
                hits.len() > h.len() || (hits.len() == h.len() && classes.len() < b.classes().len())
            }
        };
        if better {
            best = Some((m, hits));
        }
    }
    best
}

/// A state from the command line or a test's environment: a folder the
/// viewer would open, or `DIR::FILE[::FILE...]` - the files of one folder
/// (a series as an archive files it) with other files beside them (the
/// structure set of that series, filed at the study's level), so one
/// series and one set are loaded without copying anything.
pub fn load_state(spec: &str, p: &Progress) -> Result<LoadedStudy> {
    let mut parts = spec.split("::");
    let dir = std::path::PathBuf::from(parts.next().unwrap_or_default());
    let extra: Vec<std::path::PathBuf> = parts.map(std::path::PathBuf::from).collect();
    if extra.is_empty() {
        return crate::loader::load_directory(&dir, p)
            .with_context(|| format!("load {}", dir.display()));
    }
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| format!("read {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|f| f.is_file())
        .collect();
    files.sort();
    files.extend(extra);
    crate::loader::load_files(&files, &dir.to_string_lossy(), p)
        .with_context(|| format!("load {spec}"))
}

/// A rigid body's contoured volume differs between the states by at most
/// this fraction of the larger.
pub const MAX_BODY_VOLUME_CHANGE: f64 = 0.25;

/// A rigid body spans at least this many analysis voxels along every axis.
pub const MIN_BODY_VOXELS: usize = 3;

/// The residual registration a request asks for by default when it wants
/// one: plastimatch's B-spline at a 30 mm lattice, MSD.
pub fn default_residual() -> RegParams {
    RegParams {
        method: RegMethod::PlastimatchBSpline,
        levels: 2,
        iterations: 40,
        grid_spacing_mm: 30.0,
        stride: 2,
        ..RegParams::default()
    }
}

/// What [`build`] hands back.
pub struct PostureOutcome {
    pub model: PostureModel,
    pub tables: Tables,
    /// The structures every state has, in the first state's order.
    pub structures: Vec<String>,
    /// Per segment, per body: the lattice nodes it covers at the start
    /// (what [`crate::posture::file::write`] keeps).
    pub occupancy: posture::file::Occupancies,
    pub warnings: Vec<String>,
}

fn key(name: &str) -> String {
    name.trim().to_lowercase()
}

/// The structure set of a state.
fn structure_set(s: &StateInput) -> Option<usize> {
    let st = s.study;
    if let Some(i) = s.structure_set {
        return (i < st.structure_sets.len()).then_some(i);
    }
    let uid = st.series.get(st.active_series).map(|se| se.uid.as_str());
    st.structure_sets
        .iter()
        .position(|ss| uid.is_some_and(|u| ss.referenced_series_uid == u))
        .or_else(|| (!st.structure_sets.is_empty()).then_some(0))
}

/// The volume ROIs of a state by their key.
fn rois<'a>(s: &'a StateInput) -> BTreeMap<String, &'a Roi> {
    let mut out = BTreeMap::new();
    if let Some(i) = structure_set(s) {
        for r in &s.study.structure_sets[i].rois {
            if r.is_poi() || r.contours.is_empty() {
                continue;
            }
            out.entry(key(&r.name)).or_insert(r);
        }
    }
    out
}

/// The joints the names suggest: each hip, the lumbosacral junction and
/// every pair of neighbouring vertebrae, by TotalSegmentator's names
/// (`hip_left`, `femur_left`, `vertebrae_L5`) or a clinical scheme's
/// (`Bone_Pelvic_L`, `Femur_L`, `VB_L5`); two bodies and nothing else
/// found, the two.
pub fn preset_joints(bodies: &[String]) -> Vec<(String, String, String)> {
    let find = |n: &str| bodies.iter().find(|b| key(b) == n).cloned();
    let find_any = |names: &[String]| names.iter().find_map(|n| find(n));
    let mut out = Vec::new();
    for (side, s) in [("left", "l"), ("right", "r")] {
        let hip = find_any(&[format!("hip_{side}"), format!("bone_pelvic_{s}")]);
        let femur = find_any(&[format!("femur_{side}"), format!("femur_{s}")]);
        if let (Some(h), Some(f)) = (hip, femur) {
            out.push((
                format!("{} hip", if side == "left" { "Left" } else { "Right" }),
                h,
                f,
            ));
        }
    }
    // Vertebrae by level, from the head down: C1..C7, T1..T12, L1..L5.
    let level = |n: &str| -> Option<usize> {
        let k = key(n);
        let k = k
            .strip_prefix("vertebrae_")
            .or_else(|| k.strip_prefix("vb_"))
            .unwrap_or(&k);
        let (c, num) = k.split_at(1.min(k.len()));
        let num: usize = num.parse().ok()?;
        match c {
            "c" if (1..=7).contains(&num) => Some(num),
            "t" if (1..=12).contains(&num) => Some(7 + num),
            "l" if (1..=6).contains(&num) => Some(19 + num),
            _ => None,
        }
    };
    let mut vert: Vec<(usize, String)> = bodies
        .iter()
        .filter_map(|b| level(b).map(|l| (l, b.clone())))
        .collect();
    vert.sort();
    for w in vert.windows(2) {
        if w[1].0 == w[0].0 + 1 {
            out.push((
                format!("{} / {}", w[0].1, w[1].1),
                w[0].1.clone(),
                w[1].1.clone(),
            ));
        }
    }
    if let (Some(last), Some(s)) = (vert.last(), find("sacrum")) {
        out.push(("Lumbosacral".into(), s, last.1.clone()));
    }
    if out.is_empty() && bodies.len() == 2 {
        out.push((
            format!("{} / {}", bodies[0], bodies[1]),
            bodies[0].clone(),
            bodies[1].clone(),
        ));
    }
    out
}

/// A mask of `full` on `coarse` (the same extent, coarser voxels): each
/// coarse voxel takes the full voxel nearest its centre.
fn sample_down(mask: &[u8], full: &Grid, coarse: &Grid) -> Vec<u8> {
    use rayon::prelude::*;
    let [nx, ny, nz] = coarse.dims;
    let [fx, fy, fz] = full.dims;
    (0..nx * ny * nz)
        .into_par_iter()
        .map(|n| {
            let (i, j, k) = (n % nx, (n / nx) % ny, n / (nx * ny));
            let v = full.patient_to_voxel(coarse.voxel_to_patient(i as f64, j as f64, k as f64));
            let a = (v[0].round().max(0.0) as usize).min(fx - 1);
            let b = (v[1].round().max(0.0) as usize).min(fy - 1);
            let c = (v[2].round().max(0.0) as usize).min(fz - 1);
            mask[(c * fy + b) * fx + a]
        })
        .collect()
}

/// Steps 1 to 3 of [`build`]: the common structures, the anchor (`given`
/// transforms stand in for the fit when a model is reopened), and every
/// state in the anchor frame on the analysis grid.
struct Prepared {
    /// Keys and display names of the common structures.
    common: Vec<String>,
    display: Vec<String>,
    colors: Vec<[u8; 3]>,
    external: Vec<bool>,
    own_cm3: Vec<Vec<f64>>,
    agrid: Grid,
    vols: Vec<Arc<Volume>>,
    coverage: Vec<BoxedMask>,
    /// Per structure, per state.
    masks: Vec<Vec<BoxedMask>>,
    anchor_idx: Option<usize>,
    to_state: Vec<Rigid>,
    anchor_rms: Vec<Option<f64>>,
    provenance: Vec<String>,
    warnings: Vec<String>,
}

fn check(p: &Progress) -> Result<()> {
    if p.cancelled() {
        bail!("posture model {}", progress::CANCELLED);
    }
    Ok(())
}

fn validate(states: &[StateInput]) -> Result<()> {
    if states.len() < 2 {
        bail!("a posture model needs at least two states");
    }
    for s in states {
        if !s.study.has_volume() {
            bail!("'{}' has no image volume", s.label);
        }
    }
    Ok(())
}

fn prepare(
    states: &[StateInput],
    req: &PostureRequest,
    given: Option<&[Rigid]>,
    p: &Progress,
) -> Result<Prepared> {
    let mut warnings = Vec::new();
    let mut provenance = Vec::new();
    let grid0 = states[0].study.volume.grid();
    // ---- 1. the structures every state has ----
    p.report(0.0, "Matching the structures");
    let per_state: Vec<BTreeMap<String, &Roi>> = states.iter().map(rois).collect();
    let first_order: Vec<String> = structure_set(&states[0])
        .map(|i| {
            states[0].study.structure_sets[i]
                .rois
                .iter()
                .map(|r| key(&r.name))
                .collect()
        })
        .unwrap_or_default();
    let mut common: Vec<String> = Vec::new();
    for k in first_order {
        if !common.contains(&k) && per_state.iter().all(|m| m.contains_key(&k)) {
            common.push(k);
        }
    }
    let display: Vec<String> = common
        .iter()
        .map(|k| per_state[0][k].name.clone())
        .collect();
    if common.is_empty() {
        warnings.push(
            "No structure has the same name on every state: the tables have no structure rows, \
             and only an image anchor can be used."
                .into(),
        );
    }

    // ---- grids ----
    let agrid = coarsen(&grid0, req.analysis_mm);
    let own_grids: Vec<Grid> = states
        .iter()
        .map(|s| coarsen(&s.study.volume.grid(), req.analysis_mm))
        .collect();
    provenance.push(format!(
        "analysis grid {}x{}x{} at {:.2}x{:.2}x{:.2} mm",
        agrid.dims[0],
        agrid.dims[1],
        agrid.dims[2],
        agrid.spacing[0],
        agrid.spacing[1],
        agrid.spacing[2]
    ));

    // Each state's image on its own analysis grid (for the anchor fit).
    let own_vols: Vec<Arc<Volume>> = states
        .iter()
        .zip(&own_grids)
        .map(|(s, g)| Arc::new(apply::resample(&s.study.volume, g, |x| x)))
        .collect();

    // Masks on each state's own analysis grid, and the full-grid volumes.
    let mut own_masks: Vec<Vec<BoxedMask>> = vec![Vec::new(); states.len()];
    let mut own_cm3: Vec<Vec<f64>> = vec![Vec::new(); common.len()];
    for (si, k) in common.iter().enumerate() {
        check(p)?;
        p.report(
            0.02 + 0.08 * si as f32 / common.len().max(1) as f32,
            &format!("Rasterizing {}", display[si]),
        );
        for (st, s) in states.iter().enumerate() {
            // On the full grid, where every contour has a slice of its own:
            // rasterized straight onto a grid coarsened along the slices,
            // two contours would share a slice and the even-odd fill would
            // cancel them. The mask is then sampled down.
            let roi = per_state[st][k];
            let full = s.study.volume.grid();
            let g = &own_grids[st];
            match segmentation::rasterize_roi(&full, roi) {
                Some(m) => {
                    own_cm3[si].push(crate::morphology::count_set(&m) as f64 * full.voxel_cm3());
                    let coarse = if full.matches(g) {
                        m
                    } else {
                        sample_down(&m, &full, g)
                    };
                    own_masks[st].push(BoxedMask::pack(&coarse, g.dims));
                }
                None => {
                    own_cm3[si].push(0.0);
                    own_masks[st].push(BoxedMask::empty(g.dims));
                }
            }
        }
    }

    // ---- 2. the anchor ----
    p.report(0.1, "Fitting the anchor");
    let anchor_idx = match &req.anchor {
        Anchor::Structure(name) => Some(
            common
                .iter()
                .position(|k| *k == key(name))
                .with_context(|| format!("the anchor '{name}' is not contoured on every state"))?,
        ),
        _ => None,
    };
    let mut to_state = vec![Rigid::IDENTITY];
    let mut anchor_rms = vec![None];
    for st in 1..states.len() {
        check(p)?;
        if let Some(g) = given.and_then(|g| g.get(st)) {
            // A model read back: its anchor transforms stand.
            to_state.push(*g);
            anchor_rms.push(None);
            continue;
        }
        let g = match (&req.anchor, anchor_idx) {
            (Anchor::Structure(_), Some(ai)) => {
                let body = BodyInput {
                    name: display[ai].clone(),
                    color: per_state[0][&common[ai]].color,
                    fixed: own_masks[0][ai].clone(),
                    moving: own_masks[st][ai].clone(),
                };
                let f = fit::fit_pair(
                    &own_vols[0],
                    &own_vols[st],
                    &body,
                    crate::registration::shape::POINTS_PER_STRUCTURE,
                    p,
                )
                .with_context(|| format!("the anchor fit to '{}'", states[st].label))?;
                provenance.push(format!(
                    "anchor {} onto {}: {:.1} deg, surfaces {:.2} -> {:.2} mm{}",
                    display[ai],
                    states[st].label,
                    f.motion.angle_deg(),
                    f.mean_before_mm,
                    f.mean_after_mm,
                    if f.seeded {
                        " (principal-axes start)"
                    } else {
                        ""
                    }
                ));
                anchor_rms.push(Some(f.rms_after_mm));
                f.motion
            }
            (Anchor::Image, _) => {
                let params = RegParams {
                    method: RegMethod::ElastixRigid,
                    ..RegParams::default()
                };
                let r = registration::register(
                    &own_vols[0],
                    &own_vols[st],
                    &params,
                    &Progress::default(),
                )
                .with_context(|| format!("the rigid image fit to '{}'", states[st].label))?;
                provenance.push(format!(
                    "anchor: rigid image registration onto {}: {}",
                    states[st].label,
                    r.metric_line()
                ));
                anchor_rms.push(None);
                Rigid::from_transform(&r.transform.rigid)
            }
            _ => {
                anchor_rms.push(None);
                Rigid::IDENTITY
            }
        };
        to_state.push(g);
    }

    // ---- 3. every state in the anchor frame ----
    p.report(0.2, "Carrying the states into the anchor frame");
    let vols: Vec<Arc<Volume>> = states
        .iter()
        .zip(&to_state)
        .map(|(s, g)| Arc::new(apply::resample(&s.study.volume, &agrid, |y| g.map(y))))
        .collect();
    let coverage: Vec<BoxedMask> = states
        .iter()
        .zip(&to_state)
        .map(|(s, g)| {
            let v = &s.study.volume;
            let cells: Vec<u32> = (0..agrid.dims.iter().product::<usize>())
                .filter(|&n| {
                    let [nx, ny, _] = agrid.dims;
                    let y = agrid.voxel_to_patient(
                        (n % nx) as f64,
                        ((n / nx) % ny) as f64,
                        (n / (nx * ny)) as f64,
                    );
                    let c = v.patient_to_voxel(g.map(y));
                    (0..3).all(|a| c[a] >= -0.5 && c[a] <= v.dims[a] as f64 - 0.5)
                })
                .map(|n| n as u32)
                .collect();
            BoxedMask::from_cells(&cells, agrid.dims)
        })
        .collect();
    let mut masks: Vec<Vec<BoxedMask>> = vec![Vec::new(); common.len()];
    for (si, _) in common.iter().enumerate() {
        check(p)?;
        for st in 0..states.len() {
            let src = &own_masks[st][si];
            let m = if st == 0 && agrid.matches(&own_grids[0]) {
                src.clone()
            } else {
                let inv = to_state[st].inverse();
                let t = Transform3::rigid_only(inv.to_transform(crate::geometry::Vec3::ZERO));
                let carried = crate::propagate::carry_mask(&src.mask(), &own_grids[st], &agrid, &t);
                BoxedMask::pack(&carried, agrid.dims)
            };
            masks[si].push(m);
        }
    }
    let colors = common.iter().map(|k| per_state[0][k].color).collect();
    let external = common
        .iter()
        .map(|k| {
            per_state[0][k].roi_type.eq_ignore_ascii_case("EXTERNAL")
                || matches!(k.as_str(), "body" | "external" | "skin" | "outline")
        })
        .collect();
    Ok(Prepared {
        common,
        display,
        colors,
        external,
        own_cm3,
        agrid,
        vols,
        coverage,
        masks,
        anchor_idx,
        to_state,
        anchor_rms,
        provenance,
        warnings,
    })
}

/// Build the model and its tables.
pub fn build(states: &[StateInput], req: &PostureRequest, p: &Progress) -> Result<PostureOutcome> {
    validate(states)?;
    if req.mode == Mode::Linear && req.linear.is_some() && states.len() != 2 {
        bail!("a given transform walks one pair of states, not a chain");
    }
    let Prepared {
        common,
        display,
        colors,
        external,
        own_cm3,
        agrid,
        vols,
        coverage,
        masks,
        anchor_idx,
        to_state,
        anchor_rms,
        mut provenance,
        mut warnings,
    } = prepare(states, req, None, p)?;
    let grid0 = states[0].study.volume.grid();
    let lattice = Lattice::over(&grid0, req.lattice_mm);
    provenance.push(format!(
        "lattice {}x{}x{} at {:.1} mm, reach {:.0} mm",
        lattice.dims[0],
        lattice.dims[1],
        lattice.dims[2],
        lattice.min_spacing(),
        req.reach_mm
    ));

    // ---- 4. the segments ----
    let body_keys: Vec<usize> = match &req.bodies {
        Bodies::None => Vec::new(),
        Bodies::Bones => (0..common.len())
            .filter(|&i| fit::is_bone(&common[i]) || Some(i) == anchor_idx)
            .collect(),
        Bodies::Named(names) => {
            let mut out: Vec<usize> = Vec::new();
            for n in names {
                match common.iter().position(|k| *k == key(n)) {
                    Some(i) => out.push(i),
                    None => {
                        warnings.push(format!("'{n}' is not contoured on every state; left out"))
                    }
                }
            }
            if let Some(a) = anchor_idx {
                if !out.contains(&a) {
                    out.push(a);
                }
            }
            out
        }
    };
    // A rigid fit needs a body with extent along every axis: one contoured
    // on a slice or two (a marker, a carina point drawn as a contour) turns
    // nowhere in the fit, and would read as a body that stood still.
    let body_keys: Vec<usize> = body_keys
        .into_iter()
        .filter(|&i| {
            let thin = (0..states.len()).any(|st| {
                masks[i][st]
                    .bbox()
                    .is_none_or(|(lo, hi)| (0..3).any(|a| hi[a] - lo[a] + 1 < MIN_BODY_VOXELS))
            });
            if thin && Some(i) != anchor_idx {
                warnings.push(format!(
                    "'{}' is under {MIN_BODY_VOXELS} voxels thick along an axis on a state: too thin \
                     to fit as a rigid body, left out of the bodies",
                    display[i]
                ));
            }
            if thin {
                return Some(i) == anchor_idx;
            }
            // A bone keeps its volume: one whose contoured volume differs
            // between the states by more than a quarter was seen in part -
            // cut by a field of view, or drawn differently - and a rigid fit
            // of two different parts reads a turn that never happened.
            let v = &own_cm3[i];
            let (lo, hi) = v
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
            let unequal = hi > 0.0 && (hi - lo) / hi > MAX_BODY_VOLUME_CHANGE;
            if unequal && Some(i) != anchor_idx {
                warnings.push(format!(
                    "'{}' is contoured with {:.0} % less volume on one state than on another: \
                     not the same part of the bone on both, left out of the bodies",
                    display[i],
                    100.0 * (hi - lo) / hi
                ));
                return false;
            }
            true
        })
        .collect();
    let anchor_name = anchor_idx.map(|i| display[i].clone());
    let mut segments = Vec::with_capacity(states.len() - 1);
    let mut occupancy: posture::file::Occupancies = Vec::new();
    for k in 0..states.len() - 1 {
        check(p)?;
        let base = 0.25 + 0.5 * k as f32 / (states.len() - 1) as f32;
        let span = 0.5 / (states.len() - 1) as f32;
        p.set_phase(base, span);
        p.set_prefix(format!("{} to {}: ", states[k].label, states[k + 1].label));
        let bodies: Vec<BodyInput> = body_keys
            .iter()
            .map(|&i| BodyInput {
                name: display[i].clone(),
                color: colors[i],
                fixed: masks[i][k].clone(),
                moving: masks[i][k + 1].clone(),
            })
            .collect();
        let linear = match (req.mode, &req.linear) {
            (Mode::Linear, Some(t)) => {
                // The given transform is between the states' own frames.
                let into = to_state[1].inverse();
                Some(Field::from_map(&lattice, |x| into.map(t.map(x))))
            }
            _ => None,
        };
        let opts = FitOptions {
            mode: req.mode,
            reach_mm: req.reach_mm,
            residual: req.residual.clone(),
            linear,
            keep_weights: req.smooth_chain && states.len() >= 3,
            ..FitOptions::default()
        };
        let seg = fit::fit_segment(
            &vols[k],
            &vols[k + 1],
            &bodies,
            anchor_name.as_deref(),
            &lattice,
            &opts,
            p,
        )?;
        occupancy.push(
            seg.bodies
                .iter()
                .map(|b| {
                    let src = bodies
                        .iter()
                        .find(|x| x.name == b.name)
                        .expect("a fitted body was asked for");
                    posture::polyrigid::occupancy(&src.fixed.mask(), &agrid, &lattice)
                })
                .collect(),
        );
        for n in &seg.notes {
            if n.contains("left out") {
                warnings.push(format!(
                    "{} to {}: {n}",
                    states[k].label,
                    states[k + 1].label
                ));
            }
            provenance.push(format!(
                "{} to {}: {n}",
                states[k].label,
                states[k + 1].label
            ));
        }
        segments.push(seg);
    }
    p.set_prefix("");
    p.set_phase(0.0, 1.0);

    let infos: Vec<StateInfo> = states
        .iter()
        .map(|s| {
            let se = s.study.series.get(s.study.active_series);
            StateInfo {
                label: s.label.clone(),
                date: se.map(|x| x.study_date.clone()).unwrap_or_default(),
                series: se.map(|x| x.description.clone()).unwrap_or_default(),
                position: se
                    .and_then(crate::loader::series_patient_position)
                    .map(|p| p.code().to_string())
                    .unwrap_or_default(),
                posture: s.posture,
                grid: Some(s.study.volume.grid()),
            }
        })
        .collect();
    let mut model = PostureModel {
        lattice,
        grid: grid0,
        analysis: agrid.clone(),
        anchor: req.anchor.clone(),
        to_state,
        states: infos,
        segments,
        ends: Vec::new(),
        provenance,
        reach_mm: req.reach_mm,
    };
    if posture::chain::smooth(&mut model) {
        model.provenance.push(
            "chain: each body on a cubic curve through the states, its velocity continuous \
             at every state"
                .into(),
        );
    }
    p.report(0.76, "The path's end points");
    model.finish();
    check(p)?;

    // ---- 5. the tables ----
    // Used: a body of the model (a poor fit was left out of it), or the
    // anchor.
    let fitted = model.body_names();
    let used: Vec<bool> = (0..common.len())
        .map(|i| Some(i) == anchor_idx || fitted.iter().any(|b| key(b) == common[i]))
        .collect();
    let structures = structure_inputs(&display, &colors, &masks, &own_cm3, &used, &external);
    let joints = if req.joints.is_empty() {
        preset_joints(&model.body_names())
    } else {
        req.joints.clone()
    };
    p.set_phase(0.78, 0.22);
    let tables = measure::tables(
        &model,
        &measure::Inputs {
            grid: &agrid,
            volumes: &vols,
            coverage: &coverage,
            structures: &structures,
            joints: &joints,
            anchor_rms_mm: &anchor_rms,
            samples_per_segment: req.samples_per_segment,
            beam: req.wepl.as_ref().map(|w| w.beam(&model.states[0].position)),
        },
        p,
    );
    p.set_phase(0.0, 1.0);
    p.report(1.0, "Posture model built");
    Ok(PostureOutcome {
        model,
        tables,
        structures: display,
        occupancy,
        warnings,
    })
}

fn structure_inputs(
    display: &[String],
    colors: &[[u8; 3]],
    masks: &[Vec<BoxedMask>],
    own_cm3: &[Vec<f64>],
    used: &[bool],
    external: &[bool],
) -> Vec<StructureInput> {
    (0..display.len())
        .map(|i| StructureInput {
            name: display[i].clone(),
            color: colors[i],
            masks: masks[i].clone(),
            own_cm3: own_cm3[i].clone(),
            used: used[i],
            external: external[i],
        })
        .collect()
}

/// The tables of a model read back from its file ([`crate::posture::file`])
/// for the states it was built from: the structures and the states carried
/// into the anchor frame by the model's own transforms, nothing fitted
/// again. `joints` and `samples_per_segment` are taken from `req`.
pub fn reopen(
    model: PostureModel,
    states: &[StateInput],
    req: &PostureRequest,
    p: &Progress,
) -> Result<PostureOutcome> {
    validate(states)?;
    if states.len() != model.states.len() {
        bail!(
            "the model is of {} states, {} were given",
            model.states.len(),
            states.len()
        );
    }
    let mut req = req.clone();
    req.anchor = model.anchor.clone();
    let prep = prepare(states, &req, Some(&model.to_state), p)?;
    let bodies = model.body_names();
    let used: Vec<bool> = prep
        .display
        .iter()
        .map(|d| bodies.iter().any(|b| key(b) == key(d)))
        .collect();
    let structures = structure_inputs(
        &prep.display,
        &prep.colors,
        &prep.masks,
        &prep.own_cm3,
        &used,
        &prep.external,
    );
    let joints = if req.joints.is_empty() {
        preset_joints(&bodies)
    } else {
        req.joints.clone()
    };
    let tables = measure::tables(
        &model,
        &measure::Inputs {
            grid: &prep.agrid,
            volumes: &prep.vols,
            coverage: &prep.coverage,
            structures: &structures,
            joints: &joints,
            anchor_rms_mm: &prep.anchor_rms,
            samples_per_segment: req.samples_per_segment,
            beam: req.wepl.as_ref().map(|w| w.beam(&model.states[0].position)),
        },
        p,
    );
    let mut warnings = prep.warnings;
    for (k, s) in states.iter().enumerate() {
        let want = model.states[k].grid.as_ref().map(|g| g.dims);
        if want.is_some_and(|d| d != s.study.volume.dims) {
            warnings.push(format!(
                "'{}' is not on the grid the model was built on",
                s.label
            ));
        }
    }
    Ok(PostureOutcome {
        model,
        tables,
        structures: prep.display,
        occupancy: Vec::new(),
        warnings,
    })
}

/// The caveat every interpolated export carries, in its series description
/// and its derivation: what the image is, and what it is not.
pub const CAVEAT: &str =
    "interpolated by the posture model between scans, not a scan: for looking at and measuring along, not for planning";

/// What [`export_frame`] wrote.
pub struct FrameExport {
    pub series_uid: String,
    pub image_files: Vec<std::path::PathBuf>,
    pub structure_files: Vec<std::path::PathBuf>,
}

/// The image at time `t` as a derived CT series of the first state's study
/// on its grid, and the first state's structures carried to `t` as an RT
/// Structure Set on it, into `dir`. `images` are the states' own volumes;
/// `first` is the first state's study and `structure_set` the set to carry.
#[allow(clippy::too_many_arguments)]
pub fn export_frame(
    model: &PostureModel,
    images: &[Arc<Volume>],
    first: &LoadedStudy,
    structure_set: Option<usize>,
    t: f64,
    show: apply::Show,
    dir: &std::path::Path,
    p: &Progress,
) -> Result<FrameExport> {
    use crate::dicom_export::{self, DerivedImage};
    let source = first
        .series
        .get(first.active_series)
        .context("the first state has no image series")?;
    let frame = model.frame(t);
    p.set(format!("Rendering t = {t:.2}"));
    let vol = apply::render(model, images, &frame, show, &model.grid)?;
    let series_uid = dicom_export::new_uid();
    let when = match model.param_at(t) {
        Some(v) => format!("t = {t:.2} ({v:.1})"),
        None => format!("t = {t:.2}"),
    };
    let description = format!("Posture model {when} [{}]", show.label());
    let derivation = format!(
        "{} {} of the way along the {} path from {}; {CAVEAT}",
        show.label(),
        when,
        model.mode().label().to_lowercase(),
        model
            .states
            .iter()
            .map(|s| s.label.as_str())
            .collect::<Vec<_>>()
            .join(" to ")
    );
    let images_dir = dir.join("image");
    let written = dicom_export::write_derived_image_series(
        &vol,
        source,
        &images_dir,
        &DerivedImage {
            series_uid: &series_uid,
            description: &description,
            derivation: &derivation,
            series_number: 900,
            window: show.window().unwrap_or(first.default_window),
            made_by: "Posture model",
        },
        p,
    )?;
    let set = structure_set
        .or_else(|| (!first.structure_sets.is_empty()).then_some(0))
        .and_then(|i| first.structure_sets.get(i));
    let mut structure_files = Vec::new();
    if let Some(ss) = set {
        let mut carried = apply::carry_structure_set(ss, &frame.forward, format!("Posture {when}"));
        carried.referenced_series_uid = written.series_uid.clone();
        carried.file_name = String::new();
        carried.locked = false;
        carried.study_uid = source.study_uid.clone();
        carried.frame_of_reference_uid = model.grid.frame_of_reference_uid.clone();
        for r in &mut carried.rois {
            r.description = format!("carried to {when}; {CAVEAT}");
        }
        let holder = LoadedStudy {
            meta: first.meta.clone(),
            series: first.series.clone(),
            active_series: first.active_series,
            volume: first.volume.clone(),
            structure_sets: vec![carried],
            ..LoadedStudy::default()
        };
        let params = dicom_export::ExportParams::for_study(first);
        let files = dicom_export::export_derived_some(
            &holder,
            &dir.join("structures"),
            &params,
            p,
            &|_, _| true,
        )?;
        structure_files = files.into_iter().map(|f| f.path).collect();
    }
    Ok(FrameExport {
        series_uid: written.series_uid,
        image_files: written.files,
        structure_files,
    })
}

/// The whole path, first state to last, as a DICOM Deformable Spatial
/// Registration from the first state's frame of reference into the last
/// state's own patient coordinates, so the model's end point is usable
/// wherever a registration is.
pub fn export_registration(
    model: &PostureModel,
    first: &LoadedStudy,
    last_for_uid: &str,
    path: &std::path::Path,
) -> Result<()> {
    use crate::dicom_export::{self, DvfExport};
    let end = model.frame(model.span());
    let into = *model.to_state.last().context("no states")?;
    let field = Field::from_map(&model.lattice, |p| into.map(end.forward.map(p)));
    let vf = field.to_vector_field(None);
    let description = format!(
        "Posture model {} ({}), anchor {}",
        model
            .states
            .iter()
            .map(|s| s.label.as_str())
            .collect::<Vec<_>>()
            .join(" to "),
        model.mode().label().to_lowercase(),
        model.anchor.label()
    );
    dicom_export::write_deformable_registration(
        path,
        &vf,
        &DvfExport {
            source_for_uid: &model.grid.frame_of_reference_uid,
            target_for_uid: last_for_uid,
            study_uid: first
                .series
                .get(first.active_series)
                .map(|s| s.study_uid.as_str())
                .unwrap_or_default(),
            patient_name: &first.meta.patient_name,
            patient_id: &first.meta.patient_id,
            label: "POSTURE",
            description: &description,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_names_suggest_the_joints() {
        let names: Vec<String> = [
            "hip_left",
            "femur_left",
            "sacrum",
            "vertebrae_L5",
            "vertebrae_L4",
            "femur_right",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let j = preset_joints(&names);
        let labels: Vec<&str> = j.iter().map(|x| x.0.as_str()).collect();
        assert!(labels.contains(&"Left hip"));
        assert!(
            !labels.iter().any(|l| l.contains("Right")),
            "no right hip bone"
        );
        assert!(labels.contains(&"vertebrae_L4 / vertebrae_L5"));
        assert!(labels.contains(&"Lumbosacral"));
        let clinical: Vec<String> = ["Bone_Pelvic_L", "Femur_L", "VB_L4", "VB_L5", "Sacrum"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let j = preset_joints(&clinical);
        let labels: Vec<&str> = j.iter().map(|x| x.0.as_str()).collect();
        assert!(labels.contains(&"Left hip"), "{labels:?}");
        assert!(labels.contains(&"VB_L4 / VB_L5"), "{labels:?}");
        assert!(labels.contains(&"Lumbosacral"), "{labels:?}");
        let two = preset_joints(&["BONE_UPPER".into(), "BONE_LOWER".into()]);
        assert_eq!(two.len(), 1);
        assert_eq!(two[0].1, "BONE_UPPER");
    }

    #[test]
    fn a_state_lacking_organs_is_offered_the_model_that_makes_them() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let names = vec![
            s(&["BODY", "liver", "Spleen", "kidney_left", "PTV"]),
            s(&["body", "PTV"]),
        ];
        let l = lacking(&names);
        assert!(l[0].is_empty(), "{:?}", l[0]);
        assert_eq!(l[1], s(&["liver", "Spleen", "kidney_left"]));
        let (m, hits) = model_for_missing(&l[1], "CT").expect("a CT model makes organs");
        assert_eq!(hits.len(), 3, "{} makes {hits:?}", m.key());
        assert!(m.classes().contains(&"liver"));
        // A target is no model's class.
        assert!(model_for_missing(&s(&["PTV", "CTV_70"]), "CT").is_none());
    }
}
