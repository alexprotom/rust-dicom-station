//! The three-layer fit of one segment: the rigid bodies, their poly-rigid
//! blend, and the residual deformation.
//!
//! Both images of a segment are already in the anchor frame, on one grid
//! ([`crate::workflow::posture`] puts them there), so every transform here
//! maps the anchor frame to itself: a point of the start state to where it
//! is on the end state.
//!
//! **1. Rigid bodies.** Each body is registered on its own, surface to
//! surface, by [`crate::registration::shape::register`] with that one
//! structure (signed distance maps, Gauss-Newton, no grey values). The
//! search is local and starts from the centroids; a bone that turned
//! through sixty degrees between lying and sitting is out of its reach. So
//! when the fit from the centroids leaves the surfaces further apart than
//! [`GOOD_FIT_MM`], it is run again from each of the four proper rotations
//! that lay the body's principal axes (the eigenvectors of its voxels'
//! covariance) onto its partner's, and the best of the five is kept - a
//! principal-axes start only when it leaves the surfaces clearly closer
//! ([`SEED_GAIN`]) than the local fit. A body whose best fit is still poor
//! ([`POOR_FIT_MM`], [`POOR_FIT_DICE`]) is left out of the bodies. A body
//! that is the anchor stands still by definition and is not fitted.
//!
//! **2. Blend.** The motions as twists, mixed by [`super::polyrigid`] on
//! the lattice.
//!
//! **3. Residual** (optional): the end image pulled back through the blend,
//! `M(x) = B(exp(v)(x))`, registered onto the start image by a B-spline
//! engine of [`crate::registration`] from the identity; its displacement on
//! the lattice is faded to nothing inside the bodies (`1 - rho`) and turned
//! into a velocity field by [`super::svf::log`]. The path is then
//! `exp(t v) o exp(t u)`.
//!
//! **Linear mode** walks a given end-to-end transform (`d = T(x) - x` on
//! the lattice) or, with none given, the diffeomorphic model's own `Phi_1`,
//! so the two modes share their end points and differ only in the path.

use std::sync::Arc;

use anyhow::{bail, Context, Result};

use super::polyrigid::{self, BodyOnLattice};
use super::se3::{self, Rigid, Twist};
use super::svf::{self, Field, Lattice, Svf};
use super::{Mode, Segment};
use crate::geometry::Vec3;
use crate::progress::{self, Progress, ProgressSink};
use crate::propagate::BoxedMask;
use crate::registration::{self, shape, RegParams};
use crate::volume::{Grid, Volume};

/// A body fit whose mean surface distance is below this needs no second
/// start, mm.
pub const GOOD_FIT_MM: f64 = 1.5;

/// A principal-axes start replaces the local fit only when its RMS surface
/// distance is below this fraction of the local fit's.
pub const SEED_GAIN: f64 = 0.8;

/// A body whose best fit leaves the surfaces further apart than this, mm,
/// is no rigid body of this pair of scans - a rib cut differently by the
/// two fields of view, a bone the segmentation drew differently - and is
/// left out of the blend (the tissue there follows its neighbours), with a
/// note, rather than driving the field with a motion it does not have.
pub const POOR_FIT_MM: f64 = 3.0;

/// The same for the overlap of the moved body with its partner: a thin
/// curved bone (a rib) laid on itself backwards still lies within a couple
/// of millimetres of its own surface, but no longer overlaps it.
pub const POOR_FIT_DICE: f64 = 0.6;

/// The structures taken for bones when nothing else is said: the names
/// TotalSegmentator gives the skeleton, and any name with *bone*, *femur*,
/// *pelvis*, *vertebra* or a spine level in it - also as clinical naming
/// schemes write them (`VB_L5`, `Rib09_L`, `Bone_Pelvic_L`).
pub fn is_bone(name: &str) -> bool {
    let n = name.trim().to_ascii_lowercase();
    const WORDS: [&str; 21] = [
        "bone",
        "femur",
        "pelvis",
        "vertebra",
        "sacrum",
        "coccyx",
        "hip_",
        "rib_",
        "humerus",
        "scapula",
        "clavicula",
        "clavicle",
        "sternum",
        "skull",
        "tibia",
        "fibula",
        "patella",
        "radius",
        "ulna",
        "mandible",
        "spine",
    ];
    if WORDS.iter().any(|w| n.contains(w)) {
        return true;
    }
    // VB_L5, Rib09_L: a vertebral body by level, a rib by number.
    if n.starts_with("vb_") {
        return true;
    }
    if let Some(rest) = n.strip_prefix("rib") {
        if rest.starts_with(|c: char| c.is_ascii_digit()) {
            return true;
        }
    }
    // C1..C7, T1..T12, L1..L5, S1 as names of their own.
    let b = n.as_bytes();
    b.len() >= 2
        && b.len() <= 3
        && matches!(b[0], b'c' | b't' | b'l' | b's')
        && n[1..].parse::<u32>().is_ok()
}

/// One rigid body as both images of a segment have it.
#[derive(Clone)]
pub struct BodyInput {
    pub name: String,
    pub color: [u8; 3],
    /// On the segment's grid, at the start state.
    pub fixed: BoxedMask,
    /// On the same grid, at the end state.
    pub moving: BoxedMask,
}

/// One body's motion and how well it fitted.
#[derive(Clone, Debug)]
pub struct FittedBody {
    pub name: String,
    pub color: [u8; 3],
    /// A point of the start state to where it is on the end state.
    pub motion: Rigid,
    pub xi: Twist,
    /// Centroid at the start state.
    pub centroid: Vec3,
    /// Mean surface distance before and after the fit, mm.
    pub mean_before_mm: f64,
    pub mean_after_mm: f64,
    /// RMS of the surface distances after, mm.
    pub rms_after_mm: f64,
    pub dice_after: Option<f64>,
    /// The principal-axes start won.
    pub seeded: bool,
    /// The body is the anchor: still, not fitted.
    pub anchor: bool,
}

/// The segment's settings.
#[derive(Clone)]
pub struct FitOptions {
    pub mode: Mode,
    /// Reach of a body's influence in the blend, mm.
    pub reach_mm: f64,
    /// The residual registration (a B-spline method); `None` for none.
    pub residual: Option<RegParams>,
    /// The linear mode's end-to-end displacement on the lattice (anchor
    /// frame to anchor frame); `None` to walk the diffeomorphic model's
    /// `Phi_1`.
    pub linear: Option<Field>,
    /// Surface points per side of a body fit.
    pub points_per_structure: usize,
    /// Keep the blend's weights, so a chain can put the bodies on curves
    /// through its states ([`super::chain`]).
    pub keep_weights: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            mode: Mode::Diffeomorphic,
            reach_mm: 40.0,
            residual: None,
            linear: None,
            points_per_structure: shape::POINTS_PER_STRUCTURE,
            keep_weights: false,
        }
    }
}

/// Centroid and covariance of a mask's voxels, patient mm, in scan order.
fn moments(mask: &BoxedMask, grid: &Grid) -> Option<(Vec3, se3::M3)> {
    let n = mask.count();
    if n == 0 {
        return None;
    }
    let mut s = Vec3::ZERO;
    mask.for_each_set(|i, j, k| s = s + grid.voxel_to_patient(i as f64, j as f64, k as f64));
    let c = s * (1.0 / n as f64);
    let mut m = [[0.0; 3]; 3];
    mask.for_each_set(|i, j, k| {
        let d = grid.voxel_to_patient(i as f64, j as f64, k as f64) - c;
        let v = [d.x, d.y, d.z];
        for a in 0..3 {
            for b in 0..3 {
                m[a][b] += v[a] * v[b];
            }
        }
    });
    for row in m.iter_mut() {
        for v in row.iter_mut() {
            *v /= n as f64;
        }
    }
    Some((c, m))
}

/// The four proper rotations that lay `a`'s principal axes on `b`'s, as
/// motions taking `a`'s centroid to `b`'s (each mask on its own grid).
fn axis_seeds(a: &BoxedMask, ga: &Grid, b: &BoxedMask, gb: &Grid) -> Vec<Rigid> {
    let (Some((ca, ma)), Some((cb, mb))) = (moments(a, ga), moments(b, gb)) else {
        return Vec::new();
    };
    let (_, mut ea) = se3::sym_eigen(&ma);
    let (_, mut eb) = se3::sym_eigen(&mb);
    // Right-handed bases, so every candidate is a rotation.
    for e in [&mut ea, &mut eb] {
        if se3::det(e) < 0.0 {
            for row in e.iter_mut() {
                row[2] = -row[2];
            }
        }
    }
    let signs = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    signs
        .iter()
        .map(|sg| {
            // R = E_b S E_a^T.
            let mut r = [[0.0; 3]; 3];
            for (i, row) in r.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell = (0..3).map(|k| eb[i][k] * sg[k] * ea[j][k]).sum();
                }
            }
            Rigid {
                r,
                t: cb - se3::mat_vec(&r, ca),
            }
        })
        .collect()
}

/// Register one body between the two images (see the module doc).
pub fn fit_body(
    fixed: &Volume,
    moving: &Volume,
    body: &BodyInput,
    opts: &FitOptions,
    p: &Progress,
) -> Result<FittedBody> {
    fit_pair(fixed, moving, body, opts.points_per_structure, p)
}

/// [`fit_body`] with the two masks each on its own image's grid: what the
/// anchor's fit between two states in two frames is.
pub fn fit_pair(
    fixed: &Volume,
    moving: &Volume,
    body: &BodyInput,
    points_per_structure: usize,
    p: &Progress,
) -> Result<FittedBody> {
    let grid = fixed.grid();
    let mgrid = moving.grid();
    let pair = shape::ShapePair {
        name: body.name.clone(),
        color: body.color,
        weight: 1.0,
        fixed: body.fixed.mask(),
        moving: body.moving.mask(),
    };
    let base = shape::ShapeRequest {
        pairs: vec![pair],
        points_per_structure,
        ..shape::ShapeRequest::default()
    };
    let quiet = Progress::default();
    let run = |seed: Option<Rigid>| -> Result<shape::ShapeOutcome> {
        if p.cancelled() {
            bail!("posture model {}", progress::CANCELLED);
        }
        let mut req = base.clone();
        req.seed = seed.map(|g| g.to_transform(Vec3::ZERO));
        shape::register(fixed, moving, &req, &quiet)
    };
    let mut best = run(None).with_context(|| format!("fitting '{}'", body.name))?;
    let mut seeded = false;
    let score = |o: &shape::ShapeOutcome| {
        o.report
            .lines
            .first()
            .map_or(f64::INFINITY, |l| l.rms_after_mm)
    };
    if best
        .report
        .lines
        .first()
        .is_none_or(|l| l.mean_after_mm > GOOD_FIT_MM)
    {
        // A start from the principal axes must beat the local fit clearly:
        // on a nearly symmetric body (a rib, a short piece of a long bone
        // cut by the scan) a flipped placement can lie a hair closer to the
        // surface by noise, and win a turn the anatomy never made.
        let local = score(&best);
        for seed in axis_seeds(&body.fixed, &grid, &body.moving, &mgrid) {
            if let Ok(o) = run(Some(seed)) {
                if score(&o) < SEED_GAIN * local && score(&o) < score(&best) - 1e-6 {
                    best = o;
                    seeded = true;
                }
            }
        }
    }
    let line = best.report.lines.first().cloned();
    let motion = Rigid::from_transform(&best.result.transform.rigid);
    let centroid = moments(&body.fixed, &grid).map(|m| m.0).unwrap_or_default();
    Ok(FittedBody {
        name: body.name.clone(),
        color: body.color,
        xi: se3::log(&motion),
        motion,
        centroid,
        mean_before_mm: line.as_ref().map_or(f64::NAN, |l| l.mean_before_mm),
        mean_after_mm: line.as_ref().map_or(f64::NAN, |l| l.mean_after_mm),
        rms_after_mm: line.as_ref().map_or(f64::NAN, |l| l.rms_after_mm),
        dice_after: line.and_then(|l| l.dice_after),
        seeded,
        anchor: false,
    })
}

/// The anchor as a body: still, by definition of the frame.
pub fn anchor_body(body: &BodyInput, grid: &Grid) -> FittedBody {
    FittedBody {
        name: body.name.clone(),
        color: body.color,
        motion: Rigid::IDENTITY,
        xi: Twist::ZERO,
        centroid: moments(&body.fixed, grid).map(|m| m.0).unwrap_or_default(),
        mean_before_mm: 0.0,
        mean_after_mm: 0.0,
        rms_after_mm: 0.0,
        dice_after: None,
        seeded: false,
        anchor: true,
    }
}

/// Fit the segment from `fixed` (the start state) to `moving` (the end
/// state), both on one grid in the anchor frame.
#[allow(clippy::too_many_arguments)]
pub fn fit_segment(
    fixed: &Arc<Volume>,
    moving: &Arc<Volume>,
    bodies: &[BodyInput],
    anchor: Option<&str>,
    lattice: &Lattice,
    opts: &FitOptions,
    p: &Progress,
) -> Result<Segment> {
    let grid = fixed.grid();
    if !moving.grid().matches(&grid) {
        bail!("the two states are not on one grid");
    }
    let mut notes = Vec::new();

    // ---- layer 1: the bodies ----
    let mut fitted = Vec::with_capacity(bodies.len());
    let n_b = bodies.len().max(1);
    for (i, b) in bodies.iter().enumerate() {
        p.report(
            0.6 * i as f32 / n_b as f32,
            &format!("Rigid fit of {} ({}/{})", b.name, i + 1, bodies.len()),
        );
        if b.fixed.is_empty() || b.moving.is_empty() {
            notes.push(format!("{}: empty on one state, left out", b.name));
            continue;
        }
        let f = if anchor.is_some_and(|a| a.eq_ignore_ascii_case(&b.name)) {
            anchor_body(b, &grid)
        } else {
            fit_body(fixed, moving, b, opts, p)?
        };
        let poor_dice = f.dice_after.is_some_and(|d| d < POOR_FIT_DICE);
        // A fit with no number (NaN) is no better than a poor one.
        let poor_fit = f.mean_after_mm.is_nan() || f.mean_after_mm > POOR_FIT_MM;
        if !f.anchor && (poor_fit || poor_dice) {
            notes.push(format!(
                "{}: left out of the rigid bodies - its best fit leaves the surfaces {:.1} mm \
                 apart with a Dice of {} (a rigid body fits within {POOR_FIT_MM} mm and a Dice \
                 of {POOR_FIT_DICE}: not one rigid body on these two scans)",
                f.name,
                f.mean_after_mm,
                f.dice_after.map_or("-".to_string(), |d| format!("{d:.2}"))
            ));
            continue;
        }
        if !f.anchor {
            notes.push(format!(
                "{}: {:.1} deg, surfaces {:.2} -> {:.2} mm{}",
                f.name,
                f.motion.angle_deg(),
                f.mean_before_mm,
                f.mean_after_mm,
                if f.seeded {
                    " (principal-axes start)"
                } else {
                    ""
                }
            ));
        }
        fitted.push(f);
    }

    // ---- layer 2: the blend ----
    p.report(0.62, "Blending the bodies");
    let on_lattice: Vec<BodyOnLattice> = fitted
        .iter()
        .map(|f| {
            let b = bodies
                .iter()
                .find(|b| b.name == f.name)
                .expect("a fitted body was given");
            let occ = polyrigid::occupancy(&b.fixed.mask(), &grid, lattice);
            BodyOnLattice::new(f.name.clone(), f.xi, &occ, lattice)
        })
        .collect();
    let velocity = polyrigid::blend(lattice, &on_lattice, opts.reach_mm);
    let curves = (opts.keep_weights && opts.mode == Mode::Diffeomorphic).then(|| {
        let xis: Vec<Twist> = fitted.iter().map(|f| f.xi).collect();
        Arc::new(super::chain::SegmentCurves::stationary(
            &xis,
            super::chain::SparseWeights::of(lattice, &on_lattice, opts.reach_mm),
        ))
    });
    if p.cancelled() {
        bail!("posture model {}", progress::CANCELLED);
    }

    // ---- layer 3: the residual ----
    let mut residual = None;
    if let Some(params) = &opts.residual {
        p.report(0.7, "Residual deformation");
        let blend_fwd = velocity.exp(1.0);
        let pulled = super::apply::resample(moving, &grid, |x| blend_fwd.map(x));
        let mut rp = params.clone();
        rp.init = registration::Init::Identity;
        rp.start = None;
        rp.region = None;
        p.set_phase(0.7, 0.25);
        let res =
            registration::register(fixed, &pulled, &rp, p).context("the residual registration")?;
        p.set_phase(0.0, 1.0);
        let t = res.transform.clone();
        let mut d = Field::from_map(lattice, |x| t.map(x));
        let rho = polyrigid::rigidity(lattice, &on_lattice);
        polyrigid::fade(&mut d.d, &rho);
        let (u, fit) = svf::log(&d, 12);
        notes.push(format!(
            "residual: {} ({}), logarithm {} rounds, {:.3} mm",
            rp.method.label(),
            res.metric_line(),
            fit.rounds,
            fit.residual_mm
        ));
        residual = (!u.is_zero()).then_some(u);
    }

    // ---- the linear mode's end point ----
    let linear = match opts.mode {
        Mode::Diffeomorphic => None,
        Mode::Linear => Some(match &opts.linear {
            Some(d) => {
                notes.push("linear: the given transform, walked in a straight line".into());
                d.clone()
            }
            None => {
                notes.push("linear: the diffeomorphic end point, walked in a straight line".into());
                let seg = Segment {
                    mode: Mode::Diffeomorphic,
                    velocity: velocity.clone(),
                    residual: residual.clone(),
                    linear: None,
                    bodies: Vec::new(),
                    curves: None,
                    notes: Vec::new(),
                };
                seg.forward(1.0)
            }
        }),
    };
    p.report(1.0, "Segment fitted");
    Ok(Segment {
        mode: opts.mode,
        velocity: velocity_or_zero(velocity, lattice),
        residual,
        linear,
        bodies: fitted,
        curves,
        notes,
    })
}

fn velocity_or_zero(v: Svf, lattice: &Lattice) -> Svf {
    if v.lattice == *lattice {
        v
    } else {
        Svf::zero(lattice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bones_are_told_from_organs_by_name() {
        for b in [
            "femur_left",
            "hip_right",
            "vertebrae_L4",
            "sacrum",
            "BONE_UPPER",
            "L5",
            "T12",
            "rib_left_3",
            "VB_L5",
            "Rib09_L",
            "Bone_Pelvic_R",
            "Femur_L",
        ] {
            assert!(is_bone(b), "{b}");
        }
        for o in [
            "bladder",
            "liver",
            "PTV",
            "BODY",
            "CTV_70",
            "lung_left",
            "Lung",
            "Ribbon",
            "Bowel_Small",
            "costal_cartilages",
        ] {
            assert!(!is_bone(o), "{o}");
        }
    }
}
