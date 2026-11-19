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
use crate::posture::template::{self, Pose, Sample, SampleBody, SampleOrgan, Template};
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
    /// Fit and draw each state's patient only ([`patient_only`]): what lies
    /// outside its EXTERNAL structure - the couch, the chair - set to air.
    /// A state with no EXTERNAL structure keeps its image as scanned.
    pub patient_only: bool,
    /// Keep what a posture template needs of a pair's model
    /// ([`PostureOutcome::sample`], [`crate::posture::template`]).
    pub sample: bool,
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
            patient_only: true,
            sample: false,
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

/// The anchor [`Anchor::Auto`] takes keeps its contoured volume between
/// the states within this fraction of the larger: a bone the scans saw in
/// part is no frame to hold still.
pub const MAX_ANCHOR_VOLUME_CHANGE: f64 = 0.10;

/// The smallest bone [`Anchor::Auto`] takes, cm3: a vertebral body is
/// thirty and more, a rib's tip or a coccyx a few.
pub const MIN_ANCHOR_CM3: f64 = 10.0;

/// One structure as [`auto_anchor`] weighs it.
#[derive(Clone, Debug)]
pub struct AnchorCandidate {
    pub name: String,
    /// Every scan sees it whole: on no state does it reach the edge of the
    /// scan, and its contoured volumes agree within
    /// [`MAX_ANCHOR_VOLUME_CHANGE`].
    pub whole: bool,
    /// Its smallest contoured volume over the states, cm3.
    pub cm3: f64,
    /// Its centroid's height on the first state (patient z, mm).
    pub si_mm: f64,
}

/// The vertebral level of a name (`VB_L5`, `vertebrae_T12`, `L5`) from the
/// head down: C1 is 1, T1 8, L1 20.
fn vertebra_level(name: &str) -> Option<usize> {
    let k = key(name);
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
}

/// What [`Anchor::Auto`] holds still, by its index in `candidates`: among
/// the bones ([`fit::is_bone`]) every scan sees whole and of at least
/// [`MIN_ANCHOR_CM3`], the sacrum; else the middle one, by height, of the
/// vertebrae; else the largest. `None`: no bone qualifies, and the whole
/// image is the anchor.
///
/// Why this order: the frame every number is read in should be the bone
/// the anatomy pivots about and the scans agree on. The sacrum is the
/// pelvis's keystone and the usual reference of a pelvic site; on a
/// thoraco-abdominal pair it is often cut by one scan (an upright scan
/// framed on the chest ends at S1), and a cut bone fitted to its whole self
/// reads a turn of tens of degrees that never happened. A vertebra in the
/// middle of the ones both scans see keeps every other level's motion small
/// and both ends of the spine in view.
pub fn auto_anchor(candidates: &[AnchorCandidate]) -> Option<usize> {
    let ok: Vec<usize> = (0..candidates.len())
        .filter(|&i| {
            let c = &candidates[i];
            c.whole && c.cm3 >= MIN_ANCHOR_CM3 && fit::is_bone(&c.name)
        })
        .collect();
    if let Some(&i) = ok.iter().find(|&&i| key(&candidates[i].name) == "sacrum") {
        return Some(i);
    }
    let mut vert: Vec<usize> = ok
        .iter()
        .copied()
        .filter(|&i| vertebra_level(&candidates[i].name).is_some())
        .collect();
    if !vert.is_empty() {
        // Head down by height (patient z falls towards the feet).
        vert.sort_by(|&a, &b| candidates[b].si_mm.total_cmp(&candidates[a].si_mm));
        return Some(vert[(vert.len() - 1) / 2]);
    }
    ok.into_iter()
        .max_by(|&a, &b| candidates[a].cm3.total_cmp(&candidates[b].cm3))
}

/// The residual registration a request asks for by default when it wants
/// one: plastimatch's B-spline at a 30 mm lattice, by mutual information,
/// with a bending penalty of one.
///
/// Why those: on three supine and upright pairs (pelvis, chest and
/// abdomen, 2026-11) mean squares drove the lattice into folds wherever the
/// content changed - a bladder with contrast on one scan and none on the
/// other, gas in the bowel - and had to be scaled to an eighth to stay
/// invertible; mutual information with the penalty raised from 0.02 to 1
/// moved the organs the furthest (model-check Dice of the organs up by
/// 0.01 to 0.03 over the bones alone) without folding or scaling. A coarser
/// lattice (50 to 80 mm) or a stiffer one (5) moved them less.
pub fn default_residual() -> RegParams {
    RegParams {
        method: RegMethod::PlastimatchBSpline,
        levels: 2,
        iterations: 40,
        grid_spacing_mm: 30.0,
        stride: 2,
        regularization: 1.0,
        metric: crate::registration::Metric::MutualInformation,
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
    /// The states' images as the model was fitted on them and draws them,
    /// each in its own coordinates: the patient only when the request asked
    /// and the state has an EXTERNAL structure, else the image as scanned.
    /// A predicted model's second image is the first moved along it.
    pub images: Vec<Arc<Volume>>,
    /// What a posture template keeps of a pair's model, when the request
    /// asked for it ([`PostureRequest::sample`]); its label is left for the
    /// caller to give.
    pub sample: Option<template::Sample>,
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

/// How far a state's EXTERNAL structure is grown before what lies outside
/// it is cleared, mm: the skin stays, and a contour drawn a voxel inside it
/// does not shave it.
pub const PATIENT_MARGIN_MM: f64 = 3.0;

/// Whether a structure is the patient's outline: its type says EXTERNAL, or
/// its name is BODY, EXTERNAL, SKIN or OUTLINE.
pub fn is_external(roi: &Roi) -> bool {
    roi.roi_type.eq_ignore_ascii_case("EXTERNAL")
        || matches!(
            key(&roi.name).as_str(),
            "body" | "external" | "skin" | "outline"
        )
}

/// A study's image with what is not the patient set to air: every voxel
/// outside its EXTERNAL structure ([`is_external`], in `structure_set` -
/// `None` for the set drawn on the displayed series, else the first - grown
/// by [`PATIENT_MARGIN_MM`]) takes the value of air, -1000 HU, or the
/// image's lowest value when that is higher (an MR). The couch of a lying
/// scan and the chair of a seated one are not anatomy: carried along the
/// path they bend with the tissue around them, and the generated image of
/// the other position should hold neither. `None` when the study has no
/// EXTERNAL structure.
pub fn patient_only(study: &LoadedStudy, structure_set: Option<usize>) -> Option<Volume> {
    let probe = StateInput {
        label: String::new(),
        study,
        structure_set,
        posture: None,
    };
    let set = &study.structure_sets[self::structure_set(&probe)?];
    let roi = set
        .rois
        .iter()
        .find(|r| is_external(r) && !r.contours.is_empty())?;
    let v = &study.volume;
    let grid = v.grid();
    let mask = segmentation::rasterize_roi(&grid, roi)?;
    let mask = crate::morphology::dilate_mm(&mask, grid.dims, grid.spacing, PATIENT_MARGIN_MM);
    let air = (-1000i16).max(v.min_value);
    let mut out = (**v).clone();
    for (x, &m) in out.data.iter_mut().zip(&mask) {
        if m == 0 {
            *x = air;
        }
    }
    out.min_value = out.data.iter().copied().min().unwrap_or(air);
    out.max_value = out.data.iter().copied().max().unwrap_or(air);
    Some(out)
}

/// The states' images as a request has the model fit and draw them: each
/// [`patient_only`] when asked and possible, else the image as scanned.
pub fn state_images(states: &[StateInput], patient: bool) -> Vec<Arc<Volume>> {
    use rayon::prelude::*;
    states
        .par_iter()
        .map(|s| {
            patient
                .then(|| patient_only(s.study, s.structure_set))
                .flatten()
                .map(Arc::new)
                .unwrap_or_else(|| s.study.volume.clone())
        })
        .collect()
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
    let mut vert: Vec<(usize, String)> = bodies
        .iter()
        .filter_map(|b| vertebra_level(b).map(|l| (l, b.clone())))
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
    /// The states' images as fitted ([`state_images`]).
    images: Vec<Arc<Volume>>,
    /// The anchor as resolved ([`Anchor::Auto`] chosen).
    anchor: Anchor,
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
    let images = state_images(states, req.patient_only);
    if req.patient_only {
        let without: Vec<&str> = states
            .iter()
            .zip(&images)
            .filter(|(s, im)| Arc::ptr_eq(&s.study.volume, im))
            .map(|(s, _)| s.label.as_str())
            .collect();
        if without.is_empty() {
            provenance
                .push("patient only: outside each state's EXTERNAL structure set to air".into());
        } else {
            provenance.push(format!(
                "patient only: {} without an EXTERNAL structure, kept as scanned (couch and all)",
                without.join(", ")
            ));
        }
    }
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
    let own_vols: Vec<Arc<Volume>> = images
        .iter()
        .zip(&own_grids)
        .map(|(im, g)| Arc::new(apply::resample(im, g, |x| x)))
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
    // Every scan sees a structure whole when on no state it reaches the
    // edge of that state's own grid (the scan's field of view) and its
    // contoured volumes agree.
    let whole: Vec<bool> = (0..common.len())
        .map(|si| {
            let edge = (0..states.len()).any(|st| {
                let d = own_grids[st].dims;
                own_masks[st][si]
                    .bbox()
                    .is_none_or(|(lo, hi)| (0..3).any(|a| lo[a] == 0 || hi[a] + 1 >= d[a]))
            });
            let v = &own_cm3[si];
            let (lo, hi) = v
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
            !edge && hi > 0.0 && (hi - lo) / hi <= MAX_ANCHOR_VOLUME_CHANGE
        })
        .collect();
    let anchor = match &req.anchor {
        Anchor::Auto => {
            let candidates: Vec<AnchorCandidate> = (0..common.len())
                .map(|si| {
                    let m = &own_masks[0][si];
                    let mut z = 0.0;
                    m.for_each_set(|i, j, k| {
                        z += own_grids[0]
                            .voxel_to_patient(i as f64, j as f64, k as f64)
                            .z
                    });
                    AnchorCandidate {
                        name: display[si].clone(),
                        whole: whole[si],
                        cm3: own_cm3[si].iter().copied().fold(f64::INFINITY, f64::min),
                        si_mm: z / m.count().max(1) as f64,
                    }
                })
                .collect();
            match auto_anchor(&candidates) {
                Some(i) => {
                    provenance.push(format!(
                        "anchor chosen: {} (every scan sees it whole)",
                        display[i]
                    ));
                    Anchor::Structure(display[i].clone())
                }
                None => {
                    provenance.push(
                        "anchor chosen: the whole image (no bone every scan sees whole)".into(),
                    );
                    Anchor::Image
                }
            }
        }
        a => a.clone(),
    };
    let anchor_idx = match &anchor {
        Anchor::Structure(name) => Some(
            common
                .iter()
                .position(|k| *k == key(name))
                .with_context(|| format!("the anchor '{name}' is not contoured on every state"))?,
        ),
        _ => None,
    };
    if let Some(ai) = anchor_idx.filter(|&ai| !whole[ai] && given.is_none()) {
        warnings.push(format!(
            "The anchor '{}' is not seen whole by every scan (it reaches the edge of a scan, or \
             its contoured volume differs between the states): the frame it gives may be turned. \
             A bone every scan sees whole is a better anchor (Auto picks one).",
            display[ai]
        ));
    }
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
        let g = match (&anchor, anchor_idx) {
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
    let vols: Vec<Arc<Volume>> = images
        .iter()
        .zip(&to_state)
        .map(|(im, g)| Arc::new(apply::resample(im, &agrid, |y| g.map(y))))
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
        .map(|k| is_external(per_state[0][k]))
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
        images,
        anchor,
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
        images,
        anchor,
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
        // The residual is fitted only where both scans have tissue: the
        // end scan's coverage, and the patient's outline at the start.
        let body = external
            .iter()
            .position(|&e| e)
            .map(|i| masks[i][k].clone())
            .filter(|m| !m.is_empty());
        let opts = FitOptions {
            mode: req.mode,
            reach_mm: req.reach_mm,
            residual: req.residual.clone(),
            linear,
            keep_weights: req.smooth_chain && states.len() >= 3,
            cover: req.residual.is_some().then(|| coverage[k + 1].clone()),
            body: body.filter(|_| req.residual.is_some()),
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
        anchor,
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
    let sample = (req.sample && states.len() == 2)
        .then(|| make_sample(&model, &tables, &display, &masks, &external, &agrid));
    p.report(1.0, "Posture model built");
    Ok(PostureOutcome {
        model,
        tables,
        structures: display,
        occupancy,
        warnings,
        images,
        sample,
    })
}

/// What a template keeps of a pair's model ([`template::Sample`]): the
/// bodies with their points and motions, the joints between them at the
/// first state, and the organs not cut by either scan with where the bones
/// alone carry them.
fn make_sample(
    model: &PostureModel,
    tables: &Tables,
    display: &[String],
    masks: &[Vec<BoxedMask>],
    external: &[bool],
    agrid: &Grid,
) -> Sample {
    let seg = &model.segments[0];
    let index = |name: &str| display.iter().position(|d| key(d) == key(name));
    let mut bodies = Vec::new();
    let mut names = Vec::new();
    let mut body_masks: Vec<&BoxedMask> = Vec::new();
    for b in &seg.bodies {
        let Some(i) = index(&b.name) else {
            continue;
        };
        let Some((c, pts)) = template::body_points(&masks[i][0], agrid) else {
            continue;
        };
        bodies.push(SampleBody {
            name: b.name.clone(),
            centroid: [c.x, c.y, c.z],
            points: pts.iter().map(|q| [q.x, q.y, q.z]).collect(),
            motion: Pose::of(&b.motion),
            fit_mm: if b.anchor { 0.0 } else { b.mean_after_mm },
            anchor: b.anchor,
        });
        names.push(b.name.clone());
        body_masks.push(&masks[i][0]);
    }
    let joints = template::contacts(&names, &body_masks, agrid, template::CONTACT_MM);
    // The organs as the bones alone carry them: the blend, no residual.
    let blend = seg.velocity.exp(1.0);
    let h = model.lattice.min_spacing();
    let organs = tables
        .structures
        .iter()
        .filter(|r| {
            !r.used
                && !fit::is_bone(&r.name)
                && !r.is_cut()
                && !index(&r.name).is_some_and(|i| external[i])
                && r.volume_cm3.iter().all(|&v| v >= template::MIN_ORGAN_CM3)
        })
        .filter_map(|r| {
            let start = (*r.centroid.first()?)?;
            let end = (*r.centroid.last()?)?;
            let carried = blend.map(start);
            Some(SampleOrgan {
                name: r.name.clone(),
                start: [start.x, start.y, start.z],
                end: [end.x, end.y, end.z],
                carried: [carried.x, carried.y, carried.z],
                gradient: blend.gradient_at_point(start, h),
                cm3: [
                    r.volume_cm3.first().copied().unwrap_or(0.0),
                    r.volume_cm3.last().copied().unwrap_or(0.0),
                ],
            })
        })
        .collect();
    Sample {
        label: String::new(),
        anchor: model.anchor.label(),
        scanner: Pose::of(model.to_state.get(1).unwrap_or(&Rigid::IDENTITY)),
        positions: model.states.iter().map(|s| s.position.clone()).collect(),
        bodies,
        joints,
        organs,
    }
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
        images: prep.images,
        sample: None,
    })
}

/// The label of a predicted state.
pub const PREDICTED: &str = "predicted";

/// The second state of one scan predicted from a posture template
/// ([`crate::posture::template`]): a model of two states whose second is
/// not a scan but the first moved as the cohort moved - its bones walked
/// out from a root along the joints the template knows, the organs it has
/// offsets for moved by them before the bones carry them (`organs`), the
/// rest blended. The outcome is a model like any: it plays, measures and
/// exports; its second image is the first carried to the end of the path,
/// and nothing in its tables is checked against a scan (no Dice, no
/// verdict on fits). Of `req`, the lattice, the analysis grid, the reach,
/// the joints and the samples, the WEPL beam and *patient only* are read;
/// the anchor is the root ([`auto_anchor`] on this one scan).
pub fn predict(
    state: &StateInput,
    tmpl: &Template,
    req: &PostureRequest,
    organs: bool,
    p: &Progress,
) -> Result<PostureOutcome> {
    use crate::posture::polyrigid::{self, BodyOnLattice};
    let st = state.study;
    if !st.has_volume() {
        bail!("'{}' has no image volume", state.label);
    }
    let edges = tmpl.edges(None);
    let known = tmpl.bones();
    if known.is_empty() {
        bail!(
            "the template has no joint seen on {} patients or more",
            template::MIN_EDGE_SAMPLES
        );
    }
    let offsets: BTreeMap<String, template::OrganStats> = if organs {
        tmpl.organs(None)
            .into_iter()
            .filter(|(_, o)| o.n >= template::MIN_ORGAN_SAMPLES)
            .collect()
    } else {
        BTreeMap::new()
    };
    let si = structure_set(state)
        .context("a prediction needs the scan's structure set, with its bones contoured")?;
    let ss = &st.structure_sets[si];
    let mut provenance = vec![format!(
        "predicted from a posture template of {} patient(s): not fitted to a second scan",
        tmpl.samples.len()
    )];
    let mut warnings = Vec::new();
    let images = state_images(std::slice::from_ref(state), req.patient_only);
    let image = images[0].clone();
    let grid0 = st.volume.grid();
    let agrid = coarsen(&grid0, req.analysis_mm);

    // ---- every structure of the scan, on the analysis grid ----
    let mut display: Vec<String> = Vec::new();
    let mut colors = Vec::new();
    let mut masks0: Vec<BoxedMask> = Vec::new();
    let mut cm3 = Vec::new();
    let mut external = Vec::new();
    for (i, roi) in ss.rois.iter().enumerate() {
        check(p)?;
        if display.iter().any(|d| key(d) == key(&roi.name)) {
            continue;
        }
        p.report(
            0.3 * i as f32 / ss.rois.len().max(1) as f32,
            &format!("Rasterizing {}", roi.name),
        );
        let Some(m) = segmentation::rasterize_roi(&grid0, roi) else {
            continue;
        };
        cm3.push(crate::morphology::count_set(&m) as f64 * grid0.voxel_cm3());
        let coarse = if grid0.matches(&agrid) {
            m
        } else {
            sample_down(&m, &grid0, &agrid)
        };
        masks0.push(BoxedMask::pack(&coarse, agrid.dims));
        display.push(roi.name.clone());
        colors.push(roi.color);
        external.push(is_external(roi));
    }

    // ---- the bones the template knows, and the root ----
    let edge_of = |m: &BoxedMask| {
        m.bbox()
            .is_none_or(|(lo, hi)| (0..3).any(|a| lo[a] == 0 || hi[a] + 1 >= agrid.dims[a]))
    };
    let bones: Vec<usize> = (0..display.len())
        .filter(|&i| fit::is_bone(&display[i]) && known.contains(&key(&display[i])))
        .filter(|&i| {
            masks0[i]
                .bbox()
                .is_some_and(|(lo, hi)| (0..3).all(|a| hi[a] - lo[a] + 1 >= MIN_BODY_VOXELS))
        })
        .collect();
    if bones.is_empty() {
        bail!("no bone of this scan is one the template knows by name");
    }
    let candidates: Vec<AnchorCandidate> = bones
        .iter()
        .map(|&i| {
            let m = &masks0[i];
            let mut z = 0.0;
            m.for_each_set(|a, b, c| z += agrid.voxel_to_patient(a as f64, b as f64, c as f64).z);
            AnchorCandidate {
                name: display[i].clone(),
                whole: !edge_of(m),
                cm3: cm3[i],
                si_mm: z / m.count().max(1) as f64,
            }
        })
        .collect();
    let names: Vec<String> = bones.iter().map(|&i| display[i].clone()).collect();
    let root = auto_anchor(&candidates)
        .or_else(|| template::root_of(&names))
        .unwrap_or(0);
    provenance.push(format!("root (held still): {}", names[root]));

    // ---- the joints, and the motions walked out along them ----
    p.report(0.32, "Finding the joints");
    let body_masks: Vec<&BoxedMask> = bones.iter().map(|&i| &masks0[i]).collect();
    let joints = template::contacts(&names, &body_masks, &agrid, template::CONTACT_MM);
    let cents: Vec<crate::geometry::Vec3> = body_masks
        .iter()
        .map(|m| {
            template::body_points(m, &agrid)
                .map(|x| x.0)
                .unwrap_or_default()
        })
        .collect();
    let pred = template::predict_bodies(&edges, &names, &cents, &joints, root);
    let unreached: Vec<&str> = names
        .iter()
        .zip(&pred)
        .filter(|(_, q)| q.motion.is_none())
        .map(|(n, _)| n.as_str())
        .collect();
    if !unreached.is_empty() {
        warnings.push(format!(
            "No joint the template knows reaches {}: left out of the bodies, they follow their \
             neighbours",
            unreached.join(", ")
        ));
    }

    // ---- the blend ----
    p.report(0.4, "Blending the bodies");
    let lattice = Lattice::over(&grid0, req.lattice_mm);
    let mut fitted = Vec::new();
    let mut on_lattice = Vec::new();
    let mut occupancy = Vec::new();
    for (k, q) in pred.iter().enumerate() {
        let Some(m) = q.motion else {
            continue;
        };
        let i = bones[k];
        let xi = posture::se3::log(&m);
        let occ = polyrigid::occupancy(&masks0[i].mask(), &agrid, &lattice);
        on_lattice.push(BodyOnLattice::new(display[i].clone(), xi, &occ, &lattice));
        occupancy.push(occ);
        fitted.push(fit::FittedBody {
            name: display[i].clone(),
            color: colors[i],
            motion: m,
            xi,
            centroid: cents[k],
            mean_before_mm: 0.0,
            mean_after_mm: 0.0,
            rms_after_mm: 0.0,
            dice_after: None,
            seeded: false,
            anchor: k == root,
        });
    }
    polyrigid::separate(&mut on_lattice, &lattice);
    let velocity = polyrigid::blend(&lattice, &on_lattice, req.reach_mm);
    let mut notes = vec![format!(
        "{} bones moved as the template's joints move them, {} not reached",
        fitted.len(),
        unreached.len()
    )];

    // ---- the organs' offsets, before the bones ----
    let mut residual = None;
    let organ_idx: Vec<(usize, &template::OrganStats)> = (0..display.len())
        .filter(|&i| !external[i] && !fit::is_bone(&display[i]))
        .filter_map(|i| offsets.get(&key(&display[i])).map(|o| (i, o)))
        .collect();
    if !organ_idx.is_empty() {
        p.report(0.5, "Moving the organs");
        let step = lattice.min_spacing();
        let mut sum = vec![crate::geometry::Vec3::ZERO; lattice.len()];
        let mut wsum = vec![0.0f64; lattice.len()];
        for (i, o) in &organ_idx {
            check(p)?;
            // The organ as a whole, faded over a lattice step at its edge.
            let f: Vec<f32> = masks0[*i]
                .mask()
                .iter()
                .map(|&v| f32::from(v.min(1)))
                .collect();
            let w = crate::morphology::blur_mm(&f, agrid.dims, agrid.spacing, step);
            let [nx, ny, _] = agrid.dims;
            for (n, (s, ws)) in sum.iter_mut().zip(wsum.iter_mut()).enumerate() {
                let c = agrid.patient_to_voxel(lattice.node(n));
                if (0..3).any(|a| c[a] < -0.5 || c[a] > agrid.dims[a] as f64 - 0.5) {
                    continue;
                }
                let [a, b, cc] =
                    [0, 1, 2].map(|x| (c[x].round().max(0.0) as usize).min(agrid.dims[x] - 1));
                let wv = w[(cc * ny + b) * nx + a] as f64;
                if wv > 1e-3 {
                    *s = *s + o.offset * wv;
                    *ws += wv;
                }
            }
        }
        let mut d = Field::zero(&lattice);
        for (n, v) in d.d.iter_mut().enumerate() {
            *v = sum[n] * (1.0 / wsum[n].max(1.0));
        }
        let rho = polyrigid::rigidity(&lattice, &on_lattice);
        polyrigid::fade(&mut d.d, &rho);
        let (mut u, _) = posture::svf::log(&d, 12);
        let mut scale = 1.0;
        for _ in 0..6 {
            let jmin = u
                .exp(1.0)
                .jacobians()
                .into_iter()
                .fold(f64::INFINITY, f64::min);
            if jmin.is_nan() || jmin >= fit::MIN_RESIDUAL_JACOBIAN {
                break;
            }
            u = u.scaled(0.5);
            scale *= 0.5;
        }
        notes.push(format!(
            "organs moved by the template's offsets before the bones: {}{}",
            organ_idx
                .iter()
                .map(|(i, o)| format!("{} {:.0} mm", display[*i], o.offset.length()))
                .collect::<Vec<_>>()
                .join(", "),
            if scale < 1.0 {
                format!(" (scaled by {scale} so they do not fold)")
            } else {
                String::new()
            }
        ));
        residual = (!u.is_zero()).then_some(u);
    }
    for n in &notes {
        provenance.push(format!("A to {PREDICTED}: {n}"));
    }
    let seg = posture::Segment {
        mode: Mode::Diffeomorphic,
        velocity,
        residual,
        linear: None,
        bodies: fitted,
        curves: None,
        notes,
    };
    let se = st.series.get(st.active_series);
    let info0 = StateInfo {
        label: state.label.clone(),
        date: se.map(|x| x.study_date.clone()).unwrap_or_default(),
        series: se.map(|x| x.description.clone()).unwrap_or_default(),
        position: se
            .and_then(crate::loader::series_patient_position)
            .map(|p| p.code().to_string())
            .unwrap_or_default(),
        posture: state.posture,
        grid: Some(grid0.clone()),
    };
    // The cohort's second position, as its headers say it most often.
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for s in &tmpl.samples {
        if let Some(pos) = s.positions.get(1).filter(|x| !x.is_empty()) {
            *seen.entry(pos.as_str()).or_default() += 1;
        }
    }
    let info1 = StateInfo {
        label: PREDICTED.into(),
        series: format!("predicted from {} patients", tmpl.samples.len()),
        position: seen
            .into_iter()
            .max_by_key(|x| x.1)
            .map(|x| x.0.to_string())
            .unwrap_or_default(),
        grid: Some(grid0.clone()),
        ..StateInfo::default()
    };
    let mut model = PostureModel {
        lattice,
        grid: grid0.clone(),
        analysis: agrid.clone(),
        anchor: Anchor::Structure(names[root].clone()),
        to_state: vec![Rigid::IDENTITY, Rigid::IDENTITY],
        states: vec![info0, info1],
        segments: vec![seg],
        ends: Vec::new(),
        provenance,
        reach_mm: req.reach_mm,
    };
    p.report(0.6, "The path's end points");
    model.finish();
    check(p)?;

    // ---- the second state: the first carried to the end ----
    p.report(0.65, "The predicted image");
    let (fwd, back) = (&model.ends[0].0, &model.ends[0].1);
    let predicted = Arc::new(apply::resample(&image, &grid0, |y| back.map(y)));
    let vols = vec![
        Arc::new(apply::resample(&image, &agrid, |x| x)),
        Arc::new(apply::resample(&predicted, &agrid, |x| x)),
    ];
    let full = BoxedMask::pack(&vec![1u8; agrid.dims.iter().product()], agrid.dims);
    let coverage = vec![full.clone(), full];
    let masks: Vec<Vec<BoxedMask>> = masks0
        .iter()
        .map(|m| vec![m.clone(), apply::carry_mask(m, &agrid, fwd, back)])
        .collect();
    let own_cm3: Vec<Vec<f64>> = masks
        .iter()
        .zip(&cm3)
        .map(|(m, &v0)| {
            // The carried volume, scaled from the analysis grid's count.
            let ratio = m[1].count() as f64 / m[0].count().max(1) as f64;
            vec![v0, v0 * ratio]
        })
        .collect();
    let body_names = model.body_names();
    let used: Vec<bool> = display
        .iter()
        .map(|d| body_names.iter().any(|b| key(b) == key(d)))
        .collect();
    let structures = structure_inputs(&display, &colors, &masks, &own_cm3, &used, &external);
    let joints = if req.joints.is_empty() {
        preset_joints(&body_names)
    } else {
        req.joints.clone()
    };
    p.set_phase(0.7, 0.3);
    let mut tables = measure::tables(
        &model,
        &measure::Inputs {
            grid: &agrid,
            volumes: &vols,
            coverage: &coverage,
            structures: &structures,
            joints: &joints,
            anchor_rms_mm: &[None, None],
            samples_per_segment: req.samples_per_segment,
            beam: req.wepl.as_ref().map(|w| w.beam(&model.states[0].position)),
        },
        p,
    );
    p.set_phase(0.0, 1.0);
    // Nothing to check a prediction against: no scan of the second state.
    for r in &mut tables.structures {
        r.check = None;
        r.deformation = None;
    }
    tables.model.check_used_dice = None;
    tables.model.check_unused_dice = None;
    tables.model.predicted = Some(format!(
        "{} patient(s), {} of this scan's bones moved by the template's joints",
        tmpl.samples.len(),
        model.body_names().len()
    ));
    p.report(1.0, "Posture predicted");
    Ok(PostureOutcome {
        model,
        tables,
        structures: display,
        occupancy: vec![occupancy],
        warnings,
        images: vec![image, predicted],
        sample: None,
    })
}

/// How another image of the first state's position is laid onto the first
/// state, for [`carry_image`].
#[derive(Clone)]
pub enum CarryAlign {
    /// The two share a frame of reference (a scanner's two series).
    Scanner,
    /// A rigid registration by mutual information, on images coarsened to
    /// the analysis spacing: an MR and a CT of one position.
    Rigid,
    /// A transform the station holds: the first state's patient coordinates
    /// to the image's (a registration with the first state fixed).
    Given(Arc<Transform3>),
}

/// Another image taken in the first state's position - an MR, a PET, a
/// contrast series - put on the first state's grid, so the model carries it
/// along the path exactly as it carries the first state's own image: the
/// way to an image of a modality only one position was scanned in (a supine
/// MR and a supine and an upright CT make an upright MR). Pass the result
/// in place of the first of the model's images to
/// [`apply::render`] / [`apply::key_frames`] / [`crate::posture::film`] with
/// [`apply::Show::StartForward`]. Returns the image and a line saying how it
/// was laid. Only the geometry is carried: what the other position does to
/// the grey values (an MR's signal, a filling's contrast) is not modelled.
pub fn carry_image(
    first: &Volume,
    other: &Volume,
    align: &CarryAlign,
    analysis_mm: f64,
    p: &Progress,
) -> Result<(Volume, String)> {
    if other.is_empty() {
        bail!("the image to carry has no voxels");
    }
    let grid = first.grid();
    let (map, line): (
        Box<dyn Fn(crate::geometry::Vec3) -> crate::geometry::Vec3 + Sync>,
        String,
    ) = match align {
        CarryAlign::Scanner => {
            if !first.frame_of_reference_uid.is_empty()
                && !other.frame_of_reference_uid.is_empty()
                && first.frame_of_reference_uid != other.frame_of_reference_uid
            {
                bail!(
                    "the image to carry is in another frame of reference than the first \
                         state: register it (rigid) instead"
                );
            }
            (Box::new(|x| x), "laid in the scanner's frame".into())
        }
        CarryAlign::Given(t) => {
            let t = t.clone();
            (
                Box::new(move |x| t.map(x)),
                "laid by the given registration".into(),
            )
        }
        CarryAlign::Rigid => {
            p.set("Registering the image to carry onto the first state");
            let a = apply::resample(first, &coarsen(&grid, analysis_mm), |x| x);
            let b = apply::resample(other, &coarsen(&other.grid(), analysis_mm), |x| x);
            let params = RegParams {
                method: RegMethod::ElastixRigid,
                metric: crate::registration::Metric::MutualInformation,
                ..RegParams::default()
            };
            let r = registration::register(&a, &b, &params, &Progress::default())
                .context("the rigid registration of the image to carry")?;
            let line = format!("laid by a rigid registration: {}", r.metric_line());
            let t = r.transform.clone();
            (Box::new(move |x| t.map(x)), line)
        }
    };
    p.set("Resampling the image to carry onto the first state");
    let mut v = apply::resample(other, &grid, map);
    v.frame_of_reference_uid = first.frame_of_reference_uid.clone();
    Ok((v, line))
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

    fn cand(name: &str, whole: bool, cm3: f64, si_mm: f64) -> AnchorCandidate {
        AnchorCandidate {
            name: name.into(),
            whole,
            cm3,
            si_mm,
        }
    }

    /// The sacrum when every scan sees it whole; cut, the middle vertebra
    /// of those seen whole, by height; no vertebra, the largest bone; no
    /// bone (organs, a cut bone, a rib tip under the size), none.
    #[test]
    fn the_auto_anchor_prefers_the_sacrum_then_the_middle_vertebra() {
        let spine = [
            cand("VB_T12", true, 40.0, 60.0),
            cand("VB_L1", true, 45.0, 30.0),
            cand("VB_L2", true, 50.0, 0.0),
            cand("VB_L3", true, 52.0, -30.0),
            cand("Liver", true, 1500.0, 80.0),
            cand("Femur_L", true, 300.0, -250.0),
        ];
        let mut c = spine.to_vec();
        c.push(cand("Sacrum", true, 200.0, -120.0));
        assert_eq!(auto_anchor(&c).map(|i| c[i].name.as_str()), Some("Sacrum"));
        c.last_mut().unwrap().whole = false;
        // Four vertebrae head down: T12, L1, L2, L3; the lower middle is L1.
        assert_eq!(auto_anchor(&c).map(|i| c[i].name.as_str()), Some("VB_L1"));
        let bones = [
            cand("Femur_L", true, 300.0, -250.0),
            cand("Bone_Pelvic_L", true, 380.0, -150.0),
            cand("Liver", true, 1500.0, 80.0),
        ];
        assert_eq!(
            auto_anchor(&bones).map(|i| bones[i].name.as_str()),
            Some("Bone_Pelvic_L")
        );
        let none = [
            cand("Liver", true, 1500.0, 80.0),
            cand("Femur_L", false, 300.0, -250.0),
            cand("Rib12_L", true, 3.0, 40.0),
        ];
        assert_eq!(auto_anchor(&none), None);
    }

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
