//! What the path did: the four tables of the posture window, the checks of
//! the model, and their CSV.
//!
//! Every number is in the anchor frame (the first state's patient
//! coordinates), in patient directions RL, AP, SI ([`crate::motion::AXES`]),
//! and every time-dependent column is sampled at evenly spaced times along
//! the whole path ([`Tables::times`]); a reading at any other time is the
//! linear interpolation of its two neighbours ([`lerp_at`]). At time `t` a
//! column reads "from the first state to here"; at the end it reads the
//! whole change.
//!
//! **States**: label, date, series and patient position from the header,
//! the posture parameter, the grid, the rigid transform that brought the
//! state into the anchor frame (its six numbers about the anchor's
//! centroid), and the body volume (the BODY / EXTERNAL structure when every
//! state has one, else the tissue above -300 HU inside the overlap of the
//! scans) with its change.
//!
//! **Structures** (one row per structure present on every state): the
//! volumes as contoured on each state (on each state's own grid), and -
//! since two scans of two positions rarely cover the same length of the
//! patient - whether the structure is **cut by a scan** (it reaches the
//! edge of what that state's scan covers) and its volume inside the part
//! of the anchor frame every scan covers, whose change is the one to read
//! for a cut structure; the
//! centroid displacement as contoured and as the model carries it, the
//! length of the centroid's path (longer than the displacement when it
//! curves), the rotation of the best rigid fit of the structure's own
//! motion (orthogonal Procrustes of its surface points against where the
//! model puts them), what is left after that fit (Dice, HD95 and mean
//! surface distance between the rigidly moved first contour and the last,
//! [`crate::motion::overlap`]), the local volume change (mean, 5th and 95th
//! percentile of the Jacobian determinant inside), the mean HU on each
//! state ([`crate::motion::grey_stats`]: a change of content rather than of
//! shape), and the model check: the first contour carried by the model to
//! the last state against the last state's own contour, and whether the
//! structure took part in the fit. With a beam direction given ([`Beam`]), also the water-equivalent
//! path length (WEPL) from where the beam enters the image to the
//! structure's centroid at each time: the image at that time (the
//! segment's two scans carried to it and cross-faded, as [`Show::CrossFade`]
//! shows it) read along the ray from the centroid back towards the source,
//! through an HU to relative stopping power table
//! ([`crate::sct::eval::Hlut`], the same generic curve the synthetic CT's
//! WET rows use unless one is given), by the midpoint rule in steps of half
//! the analysis spacing. The beam is fixed in the anchor frame (its gantry
//! and couch angles read in the first state's patient position): what it
//! measures is how far the target's depth changes as the patient moves
//! while the beam stays where it was set up. At the states it is the scans
//! themselves; in between it is the interpolated image, so the curve is the
//! model's, not a measurement.
//!
//! [`Show::CrossFade`]: super::apply::Show::CrossFade
//!
//! **Bones and joints**: each rigid body's screw over the whole path (axis,
//! angle, travel along the axis), its fit (surface distance, and the
//! largest stretch inside it along the path: about zero for a bone that
//! keeps its shape, a quarter for the linear mode's shrinking femur; and
//! the 95th percentile over its points a lattice step or more from every
//! other body, since the points of a bone within a cell of a close
//! neighbour - the femoral head in its socket, the sacrum at the
//! sacroiliac joint - read the blend between the two motions, a thin
//! shearing layer the largest value finds), and
//! each joint - a pair of bodies - as the rotation of the second relative
//! to the first, split about the patient axes ([`super::se3::patient_axis_angles`]).
//!
//! **Model**: mode, anchor, bodies, lattice, the checks - inverse
//! consistency (`Phi_t^-1 o Phi_t`, maximum and 95th percentile of what is
//! left, mm, at half way and at the end, over the lattice nodes the path
//! keeps on the lattice), the Jacobian over the path (its
//! minimum and the fraction of the lattice at or below zero: none in the
//! diffeomorphic mode), the mean Dice of the model check on structures used
//! in the fit and on those left out - and the provenance of every piece.
//!
//! The sums here report rather than decide, and are taken sequentially in
//! scan order; the per-point work is parallel and collected in order.

use std::sync::Arc;

use rayon::prelude::*;

use super::se3::{self, Rigid, Screw};
use super::svf::Field;
use super::{Mode, PostureModel};
use crate::geometry::Vec3;
use crate::motion::{self, Csv};
use crate::progress::ProgressSink;
use crate::propagate::BoxedMask;
use crate::registration::analysis::Dof6;
use crate::sct::eval::Hlut;
use crate::volume::{Grid, Volume};

/// Points of a structure's inside followed along the path.
const INTERIOR_POINTS: usize = 6000;
/// Points of its surface.
const SURFACE_POINTS: usize = 2000;
/// Half-step of the central differences a deformation gradient is read
/// with at a point, mm.
const STEP_MM: f64 = 0.5;

/// One structure as every state has it, in the anchor frame.
#[derive(Clone)]
pub struct StructureInput {
    pub name: String,
    pub color: [u8; 3],
    /// Per state, on the analysis grid.
    pub masks: Vec<BoxedMask>,
    /// Per state, as contoured on its own grid, cm3.
    pub own_cm3: Vec<f64>,
    /// A rigid body of the fit, or the anchor.
    pub used: bool,
    /// The patient outline (EXTERNAL / BODY).
    pub external: bool,
}

/// What the tables are computed from besides the model.
pub struct Inputs<'a> {
    /// The analysis grid, in the anchor frame.
    pub grid: &'a Grid,
    /// Each state's image on it.
    pub volumes: &'a [Arc<Volume>],
    /// Where each state's scan covers it.
    pub coverage: &'a [BoxedMask],
    pub structures: &'a [StructureInput],
    /// Pairs of bodies to report as joints, with a name.
    pub joints: &'a [(String, String, String)],
    /// The anchor's fit to each state (surface RMS, mm), when it was fitted.
    pub anchor_rms_mm: &'a [Option<f64>],
    /// Time samples per segment.
    pub samples_per_segment: usize,
    /// The beam the WEPL column is along; `None` leaves it out.
    pub beam: Option<Beam>,
}

/// A beam direction for the WEPL column.
#[derive(Clone, Debug)]
pub struct Beam {
    /// Unit direction the beam travels, source to patient, anchor frame.
    pub direction: Vec3,
    /// How it was given ("gantry 90, couch 0 (HFS)").
    pub label: String,
    pub hlut: Hlut,
}

impl Beam {
    /// The beam at IEC 61217 gantry and patient support angles, for a
    /// patient lying as the DICOM *Patient Position* `position` says (head
    /// first supine when it says nothing the chain knows).
    pub fn iec(gantry_deg: f64, couch_deg: f64, position: &str) -> Beam {
        use crate::iec::{self, PatientFrame, PatientPosition};
        let (sg, cg) = gantry_deg.to_radians().sin_cos();
        // The source sits at +Zg = (sin g, 0, cos g) in the fixed frame;
        // the beam travels from it to the isocentre.
        let travel = Vec3::new(-sg, 0.0, -cg);
        let pos = PatientPosition::from_code(position);
        let frame = pos.map(PatientFrame::Position).unwrap_or_default();
        let m = iec::mul(
            &iec::table_top(couch_deg, 0.0, 0.0),
            &frame.patient_to_fixed(),
        );
        Beam {
            direction: iec::mul_vec(&iec::transpose(&m), travel).normalized(),
            label: format!(
                "gantry {gantry_deg:.0}, couch {couch_deg:.0} ({})",
                pos.map_or("HFS assumed", |p| p.code())
            ),
            hlut: Hlut::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StateRow {
    pub label: String,
    pub date: String,
    pub series: String,
    pub position: String,
    pub posture: Option<f64>,
    pub dims: [usize; 3],
    pub spacing: [f64; 3],
    /// SI extent of the scan in the anchor frame, mm.
    pub si_range: (f64, f64),
    /// Into the anchor frame, about the anchor's centroid.
    pub to_anchor: Dof6,
    pub body_cm3: Option<f64>,
    pub body_change_cm3: Option<f64>,
}

/// A structure's motion at one time sample.
#[derive(Clone, Copy, Debug, Default)]
pub struct StructureAt {
    /// Centroid as the model carries it.
    pub centroid: Vec3,
    /// Length of the centroid's path so far, mm.
    pub path_mm: f64,
    /// Rotation of the best rigid fit of its motion so far, degrees, and
    /// its axis.
    pub rotation_deg: f64,
    pub axis: Vec3,
    /// Jacobian determinant inside: mean, 5th and 95th percentile.
    pub jacobian: [f64; 3],
}

/// Dice, HD95, mean surface distance and centroid error of two contours.
#[derive(Clone, Copy, Debug, Default)]
pub struct Agreement {
    pub dice: f64,
    pub hd95_mm: f64,
    pub msd_mm: f64,
    pub centroid_mm: f64,
}

#[derive(Clone, Debug)]
pub struct StructureRow {
    pub name: String,
    pub color: [u8; 3],
    pub used: bool,
    /// Per state.
    pub volume_cm3: Vec<f64>,
    /// Per state: inside the part every scan covers, cm3.
    pub common_cm3: Vec<f64>,
    /// Per state: it reaches the edge of that state's scan.
    pub cut: Vec<bool>,
    /// Per state, from the contour, in the anchor frame.
    pub centroid: Vec<Option<Vec3>>,
    /// Per state.
    pub mean_hu: Vec<Option<f64>>,
    /// Per time sample.
    pub at: Vec<StructureAt>,
    /// After the structure's own rigid fit, against the last state.
    pub deformation: Option<Agreement>,
    /// The model's carried first contour against the last state's.
    pub check: Option<Agreement>,
    /// Per time sample: the WEPL to its centroid along the beam, mm; empty
    /// without a beam.
    pub wepl_mm: Vec<f64>,
}

impl StructureRow {
    /// Volume change from the first state to the last, cm3 and percent.
    pub fn volume_change(&self) -> (f64, f64) {
        let (a, b) = (
            self.volume_cm3.first().copied().unwrap_or(0.0),
            self.volume_cm3.last().copied().unwrap_or(0.0),
        );
        (
            b - a,
            if a > 0.0 {
                100.0 * (b - a) / a
            } else {
                f64::NAN
            },
        )
    }

    /// Centroid displacement as contoured, first state to last.
    pub fn contoured_shift(&self) -> Option<Vec3> {
        Some((*self.centroid.last()?)? - (*self.centroid.first()?)?)
    }

    /// Centroid displacement as the model carries it, at sample `i`.
    pub fn model_shift(&self, i: usize) -> Option<Vec3> {
        Some(self.at.get(i)?.centroid - self.at.first()?.centroid)
    }

    /// Volume change inside the part every scan covers, first state to
    /// last, cm3.
    pub fn common_change(&self) -> f64 {
        match (self.common_cm3.first(), self.common_cm3.last()) {
            (Some(a), Some(b)) => b - a,
            _ => f64::NAN,
        }
    }

    /// Is it cut by any state's scan?
    pub fn is_cut(&self) -> bool {
        self.cut.iter().any(|&c| c)
    }

    /// The WEPL's change from the first state to sample `i`, mm.
    pub fn wepl_change(&self, i: usize) -> Option<f64> {
        Some(self.wepl_mm.get(i)? - self.wepl_mm.first()?)
    }
}

/// A body's pose at one time sample.
#[derive(Clone, Copy, Debug, Default)]
pub struct BodyAt {
    pub angle_deg: f64,
    pub shift: Vec3,
    /// Largest stretch inside so far.
    pub stretch: f64,
}

#[derive(Clone, Debug)]
pub struct BodyRow {
    pub name: String,
    pub color: [u8; 3],
    pub anchor: bool,
    /// The whole path, its axis point nearest the body's centroid.
    pub screw: Screw,
    pub at: Vec<BodyAt>,
    /// Surface distance of the rigid fit (mean, RMS), mm, worst segment.
    pub fit_mean_mm: f64,
    pub fit_rms_mm: f64,
    pub seeded: bool,
    /// Largest stretch inside along the whole path.
    pub max_stretch: f64,
    /// The 95th percentile of the stretch over its points a lattice step or
    /// more from every other body (where the model claims its rigid
    /// motion), at its worst time.
    pub stretch_p95: f64,
}

#[derive(Clone, Debug)]
pub struct JointRow {
    pub name: String,
    pub first: String,
    pub second: String,
    /// Per time sample: the second's rotation relative to the first, about
    /// RL, AP, SI, degrees.
    pub at: Vec<[f64; 3]>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelBlock {
    pub mode: Mode,
    pub anchor: String,
    pub bodies: Vec<String>,
    pub lattice_mm: f64,
    pub lattice_dims: [usize; 3],
    pub analysis_mm: [f64; 3],
    pub reach_mm: f64,
    /// `Phi_t^-1 o Phi_t` left over at half way and at the end: max, p95.
    pub inverse_mm: [[f64; 2]; 2],
    /// Smallest Jacobian determinant over the sampled path, and the
    /// fraction of the lattice at or below zero at its worst sample.
    pub jacobian_min: f64,
    pub folded_fraction: f64,
    /// Mean Dice of the model check, used in the fit and not.
    pub check_used_dice: Option<f64>,
    pub check_unused_dice: Option<f64>,
    /// Fraction of the first state's scan the other scans also cover, and
    /// the SI extent of that overlap, mm.
    pub overlap_fraction: f64,
    pub overlap_si: (f64, f64),
    /// The WEPL column's beam, as given.
    pub beam: Option<String>,
    pub provenance: Vec<String>,
}

/// The four tables.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    /// The sampled times, on `[0, n]`.
    pub times: Vec<f64>,
    /// The posture parameter at each, when every state has one.
    pub params: Vec<Option<f64>>,
    pub states: Vec<StateRow>,
    pub structures: Vec<StructureRow>,
    pub bodies: Vec<BodyRow>,
    pub joints: Vec<JointRow>,
    pub model: ModelBlock,
}

/// The value of a sampled column at time `t`: linear between the two
/// samples around it.
pub fn lerp_at(times: &[f64], values: &[f64], t: f64) -> f64 {
    if times.is_empty() || values.is_empty() {
        return f64::NAN;
    }
    let n = times.len().min(values.len());
    if t <= times[0] {
        return values[0];
    }
    for i in 1..n {
        if t <= times[i] {
            let w = (t - times[i - 1]) / (times[i] - times[i - 1]).max(1e-12);
            return values[i - 1] + (values[i] - values[i - 1]) * w;
        }
    }
    values[n - 1]
}

impl Tables {
    /// The sample nearest `t`.
    pub fn nearest(&self, t: f64) -> usize {
        self.times
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - t).abs().total_cmp(&(b.1 - t).abs()))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// How far the model can be trusted, in a few lines ([`Verdict`]).
    pub fn verdict(&self) -> Verdict {
        let m = &self.model;
        let mut v = Verdict::default();
        let worst = |level: Level, line: String, v: &mut Verdict| {
            v.level = v.level.max(level);
            v.lines.push((level, line));
        };
        if m.folded_fraction > 0.0 || m.jacobian_min <= 0.0 {
            worst(
                Level::Poor,
                format!(
                    "The path folds tissue ({:.2} % of it, Jacobian down to {:.2}): not a \
                     deformation of anatomy. Build it without the residual, or with a coarser \
                     lattice.",
                    100.0 * m.folded_fraction,
                    m.jacobian_min
                ),
                &mut v,
            );
        }
        let inv = m.inverse_mm[1][1];
        if inv > VERDICT_INVERSE_MM {
            worst(
                Level::Check,
                format!("The way back misses the way there by {inv:.1} mm (95th percentile)."),
                &mut v,
            );
        }
        for s in self.states.iter().skip(1) {
            if s.to_anchor.residual_mm > VERDICT_ANCHOR_RMS_MM {
                worst(
                    Level::Check,
                    format!(
                        "The anchor fits {} only to {:.1} mm: is it cut by a scan?",
                        s.label, s.to_anchor.residual_mm
                    ),
                    &mut v,
                );
            }
        }
        let loose: Vec<String> = self
            .bodies
            .iter()
            .filter(|b| !b.anchor && b.fit_mean_mm > VERDICT_BODY_FIT_MM)
            .map(|b| format!("{} {:.1} mm", b.name, b.fit_mean_mm))
            .collect();
        if !loose.is_empty() {
            worst(
                Level::Check,
                format!(
                    "Bones that fit loosely (mean surface distance): {}.",
                    loose.join(", ")
                ),
                &mut v,
            );
        }
        let bodies: Vec<&BodyRow> = self.bodies.iter().filter(|b| !b.anchor).collect();
        if !bodies.is_empty() {
            let fit = bodies.iter().map(|b| b.fit_mean_mm).sum::<f64>() / bodies.len() as f64;
            v.lines.push((
                Level::Good,
                format!(
                    "{} bones on screws, fitted to {fit:.1} mm on average; anchor {}.",
                    bodies.len(),
                    m.anchor
                ),
            ));
        }
        if let Some(d) = m.check_unused_dice {
            let level = if d < VERDICT_SOFT_DICE {
                Level::Check
            } else {
                Level::Good
            };
            let residual = m.provenance.iter().any(|l| l.contains("residual:"));
            let line = if d < VERDICT_SOFT_DICE {
                format!(
                    "Organs overlap their own contours on the last state by a Dice of {d:.2} \
                     only: they moved on their own (gravity, filling), which the bones do not \
                     carry{}",
                    if residual {
                        ", and the residual follows only in part."
                    } else {
                        ". The residual follows them some way."
                    }
                )
            } else {
                format!("Organs land on their own contours with a Dice of {d:.2}.")
            };
            worst(level, line, &mut v);
        }
        if m.overlap_fraction < VERDICT_OVERLAP {
            worst(
                Level::Check,
                format!(
                    "The scans overlap on {:.0} % of the first: the model holds only there.",
                    100.0 * m.overlap_fraction
                ),
                &mut v,
            );
        }
        // The worst first, then the rest in order.
        v.lines.sort_by_key(|l| std::cmp::Reverse(l.0));
        v
    }
}

/// [`Tables::verdict`]: the way back may miss the way there by this much at
/// the 95th percentile, mm, before it is pointed out.
pub const VERDICT_INVERSE_MM: f64 = 1.0;
/// The anchor's surface fit RMS above which the anchor is suspected, mm.
pub const VERDICT_ANCHOR_RMS_MM: f64 = 2.5;
/// A bone's mean surface distance above which it is named, mm.
pub const VERDICT_BODY_FIT_MM: f64 = 2.5;
/// The organs' mean model-check Dice below which they are said to have
/// moved on their own.
pub const VERDICT_SOFT_DICE: f64 = 0.6;
/// The scans' overlap below which the model's reach is pointed out.
pub const VERDICT_OVERLAP: f64 = 0.5;

/// How far a model can be trusted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    #[default]
    Good,
    /// Usable, with something to look at.
    Check,
    /// Not a deformation of anatomy.
    Poor,
}

impl Level {
    pub fn label(self) -> &'static str {
        match self {
            Level::Good => "good",
            Level::Check => "check",
            Level::Poor => "poor",
        }
    }
}

/// A model's quality in a word and a few lines: the worst level met, and
/// each finding with its level, the worst first.
#[derive(Clone, Debug, Default)]
pub struct Verdict {
    pub level: Level,
    pub lines: Vec<(Level, String)>,
}

/// Points of a mask on `grid`: every n-th set voxel, at most `max`.
fn mask_points(mask: &BoxedMask, grid: &Grid, max: usize) -> Vec<Vec3> {
    let step = mask.count().div_ceil(max.max(1)).max(1);
    let mut out = Vec::with_capacity(mask.count() / step + 1);
    let mut i = 0usize;
    mask.for_each_set(|a, b, c| {
        if i.is_multiple_of(step) {
            out.push(grid.voxel_to_patient(a as f64, b as f64, c as f64));
        }
        i += 1;
    });
    out
}

fn mean(points: &[Vec3]) -> Option<Vec3> {
    if points.is_empty() {
        return None;
    }
    Some(points.iter().fold(Vec3::ZERO, |a, b| a + *b) * (1.0 / points.len() as f64))
}

/// The rigid motion that best carries `from` onto `to` (orthogonal
/// Procrustes, through the polar factor of the cross-covariance).
pub fn procrustes(from: &[Vec3], to: &[Vec3]) -> Rigid {
    let (Some(pc), Some(qc)) = (mean(from), mean(to)) else {
        return Rigid::IDENTITY;
    };
    let mut h = [[0.0; 3]; 3];
    for (p, q) in from.iter().zip(to) {
        let a = *p - pc;
        let b = *q - qc;
        let (av, bv) = ([a.x, a.y, a.z], [b.x, b.y, b.z]);
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += bv[i] * av[j];
            }
        }
    }
    let r = se3::nearest_rotation(&h).unwrap_or(se3::I3);
    Rigid {
        r,
        t: qc - se3::mat_vec(&r, pc),
    }
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let k = ((v.len() - 1) as f64 * q).round() as usize;
    v[k.min(v.len() - 1)]
}

/// Two masks of `grid` compared in the box around both (padded), so a
/// small structure on a large image costs its own size.
pub fn agreement(a: &BoxedMask, b: &BoxedMask, grid: &Grid) -> Option<Agreement> {
    let (alo, ahi) = a.bbox()?;
    let (blo, bhi) = b.bbox()?;
    let pad = 3usize;
    let mut lo = [0usize; 3];
    let mut hi = [0usize; 3];
    for k in 0..3 {
        lo[k] = alo[k].min(blo[k]).saturating_sub(pad);
        hi[k] = (ahi[k].max(bhi[k]) + pad).min(grid.dims[k] - 1);
    }
    let size = [0, 1, 2].map(|k| hi[k] - lo[k] + 1);
    let crop = |m: &BoxedMask| -> Vec<u8> {
        let mut out = vec![0u8; size[0] * size[1] * size[2]];
        m.for_each_set(|i, j, k| {
            if (lo[0]..=hi[0]).contains(&i)
                && (lo[1]..=hi[1]).contains(&j)
                && (lo[2]..=hi[2]).contains(&k)
            {
                out[((k - lo[2]) * size[1] + j - lo[1]) * size[0] + i - lo[0]] = 1;
            }
        });
        out
    };
    let sub = Grid {
        dims: size,
        origin: grid.voxel_to_patient(lo[0] as f64, lo[1] as f64, lo[2] as f64),
        ..grid.clone()
    };
    let o = motion::overlap(&crop(a), &crop(b), &sub)?;
    Some(Agreement {
        dice: o.dice,
        hd95_mm: o.hd95_mm,
        msd_mm: o.msd_mm,
        centroid_mm: o.centroid_shift().map_or(f64::NAN, |v| v.length()),
    })
}

/// Is `p` inside the box of `grid`'s voxels (half a voxel beyond the
/// outermost centres)?
fn in_box(grid: &Grid, p: Vec3) -> bool {
    let c = grid.patient_to_voxel(p);
    (0..3).all(|a| c[a] >= -0.5 && c[a] <= grid.dims[a] as f64 - 0.5)
}

/// The WEPL from where the ray towards the source leaves `grid` to
/// `target`, at the time of `fr`, through the cross-faded image of the
/// segment's two states (`vols`, on `grid` in the anchor frame); outside
/// both scans the tissue is air.
pub fn wepl_at(
    fr: &super::Frame,
    vols: &[Arc<Volume>],
    grid: &Grid,
    target: Vec3,
    beam: &Beam,
) -> f64 {
    let i = fr.segment;
    let (Some(a), Some(b)) = (vols.get(i), vols.get(i + 1)) else {
        return f64::NAN;
    };
    let s = fr.s as f32;
    let step = (0.5 * grid.spacing.iter().cloned().fold(f64::INFINITY, f64::min)).max(0.25);
    let reach: f64 = (0..3)
        .map(|k| (grid.dims[k] as f64 * grid.spacing[k]).powi(2))
        .sum::<f64>()
        .sqrt();
    if !in_box(grid, target) {
        // The target has left the image: no depth to read.
        return f64::NAN;
    }
    let back = beam.direction * -1.0;
    let mut acc = 0.0;
    let mut l = 0.5 * step;
    // In order along the ray: the same sum on any machine.
    while l < reach {
        let x = target + back * l;
        if !in_box(grid, x) {
            break;
        }
        let hu = match (
            super::apply::sample(a, fr.to_prev.map(x)),
            super::apply::sample(b, fr.to_next.map(x)),
        ) {
            (Some(p), Some(q)) => (1.0 - s) * p + s * q,
            (Some(p), None) => p,
            (None, Some(q)) => q,
            (None, None) => -1000.0,
        };
        acc += beam.hlut.rsp(hu as f64) * step;
        l += step;
    }
    acc
}

/// A mask moved by a rigid motion on its grid.
fn rigid_mask(mask: &BoxedMask, grid: &Grid, g: &Rigid) -> BoxedMask {
    let inv = g.inverse();
    super::apply::carry_mask_by(mask, grid, |p| g.map(p), |p| inv.map(p))
}

/// The tables of `model`.
pub fn tables(model: &PostureModel, inp: &Inputs, p: &crate::progress::Progress) -> Tables {
    let grid = inp.grid;
    let span = model.span();
    let per = inp.samples_per_segment.max(2);
    let n_t = model.segments.len() * per + 1;
    let times: Vec<f64> = (0..n_t)
        .map(|k| span * k as f64 / (n_t - 1) as f64)
        .collect();
    let params: Vec<Option<f64>> = times.iter().map(|&t| model.param_at(t)).collect();
    let n_states = model.states.len();
    let last = n_states - 1;

    // ---- the path, sampled; the WEPL on the way, which needs the whole
    // frame and not only its forward field ----
    p.set("Sampling the path");
    let interiors: Vec<Vec<Vec3>> = match &inp.beam {
        Some(_) => inp
            .structures
            .iter()
            .map(|s| mask_points(&s.masks[0], grid, INTERIOR_POINTS))
            .collect(),
        None => Vec::new(),
    };
    let mut wepl: Vec<Vec<f64>> = vec![Vec::new(); interiors.len()];
    let frames: Vec<Field> = times
        .iter()
        .map(|&t| {
            let fr = model.frame(t);
            if let Some(beam) = &inp.beam {
                for (si, pts) in interiors.iter().enumerate() {
                    let moved: Vec<Vec3> = pts.par_iter().map(|&q| fr.forward.map(q)).collect();
                    let v = match mean(&moved) {
                        Some(c) => wepl_at(&fr, inp.volumes, grid, c, beam),
                        None => f64::NAN,
                    };
                    wepl[si].push(v);
                }
            }
            fr.forward
        })
        .collect();

    // ---- the part every scan covers ----
    let overlap: Option<BoxedMask> = intersect(inp.coverage, grid.dims);

    // ---- structures ----
    let mut structures = Vec::with_capacity(inp.structures.len());
    for (si, s) in inp.structures.iter().enumerate() {
        p.report(
            0.1 + 0.6 * si as f32 / inp.structures.len().max(1) as f32,
            &format!("Measuring {}", s.name),
        );
        let first = &s.masks[0];
        let interior = mask_points(first, grid, INTERIOR_POINTS);
        let surface = motion::surface_points(&first.mask(), grid, SURFACE_POINTS);
        let c0 = mean(&interior).unwrap_or_default();
        let mut at = Vec::with_capacity(n_t);
        let mut path = 0.0;
        let mut prev: Option<Vec3> = None;
        for f in &frames {
            let moved: Vec<Vec3> = interior.par_iter().map(|&q| f.map(q)).collect();
            let c = mean(&moved).unwrap_or(c0);
            if let Some(pc) = prev {
                path += (c - pc).length();
            }
            prev = Some(c);
            let msurf: Vec<Vec3> = surface.par_iter().map(|&q| f.map(q)).collect();
            let g = procrustes(&surface, &msurf);
            let w = se3::log_rotation(&g.r);
            let mut jac: Vec<f64> = interior
                .par_iter()
                .map(|&q| se3::det(&f.gradient_at_point(q, STEP_MM)))
                .collect();
            let jm = if jac.is_empty() {
                f64::NAN
            } else {
                jac.iter().sum::<f64>() / jac.len() as f64
            };
            let p5 = percentile(&mut jac, 0.05);
            let p95 = percentile(&mut jac, 0.95);
            at.push(StructureAt {
                centroid: c,
                path_mm: path,
                rotation_deg: w.length().to_degrees(),
                axis: w.normalized(),
                jacobian: [jm, p5, p95],
            });
        }
        // The end: rigid fit of the structure's own motion, and the model.
        let end = &model.ends;
        let (fwd_end, back_end) = if end.len() == 1 {
            (end[0].0.clone(), end[0].1.clone())
        } else {
            let fr = model.frame(span);
            (fr.forward, fr.to_first)
        };
        let target = &s.masks[last];
        let msurf: Vec<Vec3> = surface.par_iter().map(|&q| fwd_end.map(q)).collect();
        let g = procrustes(&surface, &msurf);
        let deformation = agreement(&rigid_mask(first, grid, &g), target, grid);
        let carried = super::apply::carry_mask(first, grid, &fwd_end, &back_end);
        let check = agreement(&carried, target, grid);
        let common_cm3: Vec<f64> = s
            .masks
            .iter()
            .map(|m| {
                let mut n = 0usize;
                m.for_each_set(|i, j, k| {
                    if overlap.as_ref().is_some_and(|o| o.get(i, j, k)) {
                        n += 1;
                    }
                });
                n as f64 * grid.voxel_cm3()
            })
            .collect();
        let cut: Vec<bool> = s
            .masks
            .iter()
            .zip(inp.coverage)
            .map(|(m, cov)| reaches_edge(m, cov, grid.dims))
            .collect();
        structures.push(StructureRow {
            name: s.name.clone(),
            color: s.color,
            used: s.used,
            volume_cm3: s.own_cm3.clone(),
            common_cm3,
            cut,
            centroid: s
                .masks
                .iter()
                .map(|m| motion::centroid_mm_boxed(m, grid))
                .collect(),
            mean_hu: s
                .masks
                .iter()
                .zip(inp.volumes)
                .map(|(m, v)| motion::grey_stats_boxed(m, v).map(|g| g[1]))
                .collect(),
            at,
            deformation,
            check,
            wepl_mm: wepl.get(si).cloned().unwrap_or_default(),
        });
    }

    // ---- bodies ----
    p.report(0.75, "Measuring the bodies");
    let body_names = model.body_names();
    let mut bodies = Vec::new();
    let mut poses: Vec<(String, Vec<Rigid>)> = Vec::new();
    for name in &body_names {
        let Some(s) = inp.structures.iter().find(|s| &s.name == name) else {
            continue;
        };
        let first = &s.masks[0];
        let interior = mask_points(first, grid, INTERIOR_POINTS);
        let surface = motion::surface_points(&first.mask(), grid, SURFACE_POINTS);
        let c0 = mean(&interior).unwrap_or_default();
        let mut at = Vec::with_capacity(n_t);
        let mut pose = Vec::with_capacity(n_t);
        // Clear of the joints: points a lattice step or more from every
        // other body (probed along the patient axes), where the model claims
        // the body's own motion. Nearer, a point's cell has corners in the
        // blend between this body and its neighbour - at a joint narrower
        // than a lattice step, the thin layer where the two slide. A body
        // with no such point is read whole.
        let step = model.lattice.min_spacing();
        let axes = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let others: Vec<&BoxedMask> = body_names
            .iter()
            .filter(|b| *b != name)
            .filter_map(|b| inp.structures.iter().find(|s| &s.name == b))
            .map(|s| &s.masks[0])
            .collect();
        let in_other = |q: Vec3| -> bool {
            let v = grid.patient_to_voxel(q);
            let (i, j, k) = (v[0].round(), v[1].round(), v[2].round());
            i >= 0.0
                && j >= 0.0
                && k >= 0.0
                && others
                    .iter()
                    .any(|m| m.get(i as usize, j as usize, k as usize))
        };
        let core: Vec<bool> = interior
            .par_iter()
            .map(|&q| {
                !in_other(q)
                    && axes
                        .iter()
                        .all(|&a| !in_other(q + a * step) && !in_other(q - a * step))
            })
            .collect();
        let any_core = core.iter().any(|&c| c);
        let mut worst: f64 = 0.0;
        let mut worst_p95: f64 = 0.0;
        for f in &frames {
            let msurf: Vec<Vec3> = surface.par_iter().map(|&q| f.map(q)).collect();
            let g = procrustes(&surface, &msurf);
            let each: Vec<f64> = interior
                .par_iter()
                .map(|&q| se3::stretch(&f.gradient_at_point(q, STEP_MM)))
                .collect();
            let st = each.iter().cloned().fold(0.0, f64::max);
            let mut core_each: Vec<f64> = each
                .iter()
                .zip(&core)
                .filter(|(_, &c)| c || !any_core)
                .map(|(v, _)| *v)
                .collect();
            worst_p95 = worst_p95.max(percentile(&mut core_each, 0.95));
            worst = worst.max(st);
            at.push(BodyAt {
                angle_deg: g.angle_deg(),
                shift: g.map(c0) - c0,
                stretch: worst,
            });
            pose.push(g);
        }
        let whole = pose.last().copied().unwrap_or_default();
        let fits: Vec<&super::fit::FittedBody> = model
            .segments
            .iter()
            .flat_map(|seg| seg.bodies.iter().filter(|b| &b.name == name))
            .collect();
        bodies.push(BodyRow {
            name: name.clone(),
            color: s.color,
            anchor: fits.iter().all(|b| b.anchor),
            screw: se3::screw(&whole, c0),
            at,
            fit_mean_mm: fits.iter().map(|b| b.mean_after_mm).fold(0.0, f64::max),
            fit_rms_mm: fits.iter().map(|b| b.rms_after_mm).fold(0.0, f64::max),
            seeded: fits.iter().any(|b| b.seeded),
            max_stretch: worst,
            stretch_p95: if worst_p95.is_finite() {
                worst_p95
            } else {
                0.0
            },
        });
        poses.push((name.clone(), pose));
    }
    let joints = inp
        .joints
        .iter()
        .filter_map(|(label, a, b)| {
            let pa = &poses.iter().find(|(n, _)| n == a)?.1;
            let pb = &poses.iter().find(|(n, _)| n == b)?.1;
            Some(JointRow {
                name: label.clone(),
                first: a.clone(),
                second: b.clone(),
                at: pa
                    .iter()
                    .zip(pb)
                    .map(|(ga, gb)| {
                        let rel = se3::mat_mul(&se3::transpose(&ga.r), &gb.r);
                        se3::patient_axis_angles(&rel)
                    })
                    .collect(),
            })
        })
        .collect();

    // ---- states ----
    p.report(0.85, "The states");
    let anchor_c = inp
        .structures
        .iter()
        .find(|s| matches!(&model.anchor, super::Anchor::Structure(a) if a.eq_ignore_ascii_case(&s.name)))
        .and_then(|s| motion::centroid_mm_boxed(&s.masks[0], grid))
        .unwrap_or_else(|| {
            grid.voxel_to_patient(
                (grid.dims[0] as f64 - 1.0) * 0.5,
                (grid.dims[1] as f64 - 1.0) * 0.5,
                (grid.dims[2] as f64 - 1.0) * 0.5,
            )
        });
    let external = inp.structures.iter().find(|s| s.external);
    let tissue: Vec<Option<f64>> = inp
        .volumes
        .iter()
        .map(|v| {
            let ov = overlap.as_ref()?;
            let mut n = 0usize;
            ov.for_each_set(|i, j, k| {
                if v.index(i, j, k) > -300 {
                    n += 1;
                }
            });
            Some(n as f64 * grid.voxel_cm3())
        })
        .collect();
    let body_cm3: Vec<Option<f64>> = (0..n_states)
        .map(|k| match external {
            Some(e) => e.own_cm3.get(k).copied(),
            None => tissue.get(k).copied().flatten(),
        })
        .collect();
    let states = model
        .states
        .iter()
        .enumerate()
        .map(|(k, st)| {
            let g = &model.to_state[k];
            let e = se3::patient_axis_angles(&g.r);
            let gr = st.grid.clone().unwrap_or_else(|| grid.clone());
            // The scan's SI extent, through the anchor frame.
            let inv = g.inverse();
            let corners: Vec<f64> = [0.0, (gr.dims[0].max(1) - 1) as f64]
                .iter()
                .flat_map(|&i| {
                    let gr = gr.clone();
                    [0.0, (gr.dims[1].max(1) - 1) as f64]
                        .into_iter()
                        .flat_map(move |j| {
                            let gr = gr.clone();
                            [0.0, (gr.dims[2].max(1) - 1) as f64]
                                .into_iter()
                                .map(move |kk| inv.map(gr.voxel_to_patient(i, j, kk)).z)
                        })
                })
                .collect();
            let lo = corners.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = corners.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            StateRow {
                label: st.label.clone(),
                date: st.date.clone(),
                series: st.series.clone(),
                position: st.position.clone(),
                posture: st.posture,
                dims: gr.dims,
                spacing: gr.spacing,
                si_range: (lo, hi),
                to_anchor: Dof6 {
                    translation: g.map(anchor_c) - anchor_c,
                    rotation_deg: e,
                    residual_mm: inp.anchor_rms_mm.get(k).copied().flatten().unwrap_or(0.0),
                },
                body_cm3: body_cm3[k],
                body_change_cm3: match (body_cm3[k], body_cm3[0]) {
                    (Some(a), Some(b)) => Some(a - b),
                    _ => None,
                },
            }
        })
        .collect();

    // ---- the model's checks ----
    p.report(0.92, "Checking the model");
    let lat = &model.lattice;
    let interior_nodes: Vec<usize> = (0..lat.len())
        .filter(|&n| {
            let [i, j, k] = lat.ijk(n);
            i > 0
                && j > 0
                && k > 0
                && i + 1 < lat.dims[0]
                && j + 1 < lat.dims[1]
                && k + 1 < lat.dims[2]
        })
        .collect();
    let inverse_at = |t: f64| -> [f64; 2] {
        let fr = model.frame(t);
        let round = Field::compose(&fr.to_first, &fr.forward);
        // Where the path keeps a node on the lattice: past its faces the
        // fields are extended, not known, and a node a large turn carries
        // out there says nothing about the model.
        let mut left: Vec<f64> = interior_nodes
            .iter()
            .filter(|&&n| lat.contains(fr.forward.map(lat.node(n))))
            .map(|&n| round.d[n].length())
            .collect();
        let max = left.iter().cloned().fold(0.0, f64::max);
        [max, percentile(&mut left, 0.95)]
    };
    let inverse_mm = [inverse_at(span * 0.5), inverse_at(span)];
    let (mut jmin, mut folded) = (f64::INFINITY, 0.0f64);
    for f in &frames {
        let j = f.jacobians();
        let inside: Vec<f64> = interior_nodes.iter().map(|&n| j[n]).collect();
        let m = inside.iter().cloned().fold(f64::INFINITY, f64::min);
        jmin = jmin.min(m);
        let fr = inside.iter().filter(|&&x| x <= 0.0).count() as f64 / inside.len().max(1) as f64;
        folded = folded.max(fr);
    }
    let mean_dice = |used: bool| -> Option<f64> {
        let d: Vec<f64> = structures
            .iter()
            .filter(|s| s.used == used)
            .filter_map(|s| s.check.map(|c| c.dice))
            .collect();
        (!d.is_empty()).then(|| d.iter().sum::<f64>() / d.len() as f64)
    };
    let cov0 = inp.coverage.first().map_or(0, BoxedMask::count);
    let overlap_si = overlap
        .as_ref()
        .and_then(BoxedMask::bbox)
        .map(|(lo, hi)| {
            let a = grid
                .voxel_to_patient(lo[0] as f64, lo[1] as f64, lo[2] as f64)
                .z;
            let b = grid
                .voxel_to_patient(hi[0] as f64, hi[1] as f64, hi[2] as f64)
                .z;
            (a.min(b), a.max(b))
        })
        .unwrap_or((f64::NAN, f64::NAN));
    let model_block = ModelBlock {
        mode: model.mode(),
        anchor: model.anchor.label(),
        bodies: body_names.clone(),
        lattice_mm: lat.min_spacing(),
        lattice_dims: lat.dims,
        analysis_mm: grid.spacing,
        reach_mm: model.reach_mm,
        inverse_mm,
        jacobian_min: jmin,
        folded_fraction: folded,
        check_used_dice: mean_dice(true),
        check_unused_dice: mean_dice(false),
        overlap_fraction: overlap
            .as_ref()
            .map_or(0.0, |o| o.count() as f64 / cov0.max(1) as f64),
        overlap_si,
        beam: inp.beam.as_ref().map(|b| b.label.clone()),
        provenance: model.provenance.clone(),
    };
    p.report(1.0, "Tables done");
    Tables {
        times,
        params,
        states,
        structures,
        bodies,
        joints,
        model: model_block,
    }
}

/// Does `mask` reach the edge of `coverage` (a set voxel with a neighbour
/// the scan does not cover, or on the grid's face)?
fn reaches_edge(mask: &BoxedMask, coverage: &BoxedMask, dims: [usize; 3]) -> bool {
    let mut cut = false;
    mask.for_each_set(|i, j, k| {
        if cut {
            return;
        }
        let at = [i, j, k];
        for a in 0..3 {
            if at[a] == 0 || at[a] + 1 >= dims[a] {
                cut = true;
                return;
            }
            for d in [-1i64, 1] {
                let mut n = at;
                n[a] = (at[a] as i64 + d) as usize;
                if !coverage.get(n[0], n[1], n[2]) {
                    cut = true;
                    return;
                }
            }
        }
    });
    cut
}

/// The voxels every mask has.
fn intersect(masks: &[BoxedMask], dims: [usize; 3]) -> Option<BoxedMask> {
    let first = masks.first()?;
    let mut cells: Vec<u32> = Vec::new();
    first.for_each_set(|i, j, k| {
        if masks[1..].iter().all(|m| m.get(i, j, k)) {
            cells.push((i + dims[0] * (j + dims[1] * k)) as u32);
        }
    });
    Some(BoxedMask::from_cells(&cells, dims))
}

fn n(v: f64, d: usize) -> String {
    motion::num(v, d)
}

fn opt(v: Option<f64>, d: usize) -> String {
    v.map(|x| n(x, d)).unwrap_or_default()
}

impl Tables {
    /// The four tables as one CSV file: a section each, a blank line
    /// between, time-dependent columns one per sampled time.
    pub fn csv(&self) -> String {
        let mut w = Csv::default();
        let time_cells: Vec<String> = self
            .times
            .iter()
            .zip(&self.params)
            .map(|(t, p)| match p {
                Some(p) => format!("t={t:.2} ({p:.1})"),
                None => format!("t={t:.2}"),
            })
            .collect();
        w.row(["Posture model", self.model.mode.label()]);
        w.row(["Anchor", &self.model.anchor]);
        w.blank();

        w.row(["States"]);
        w.row([
            "State",
            "Date",
            "Series",
            "Position",
            "Posture",
            "Grid",
            "Spacing mm",
            "SI from mm",
            "SI to mm",
            "Into anchor RL mm",
            "AP mm",
            "SI mm",
            "rot RL deg",
            "rot AP deg",
            "rot SI deg",
            "Anchor fit RMS mm",
            "Body cm3",
            "Body change cm3",
        ]);
        for s in &self.states {
            let cells = [
                s.label.clone(),
                s.date.clone(),
                s.series.clone(),
                s.position.clone(),
                opt(s.posture, 2),
                format!("{}x{}x{}", s.dims[0], s.dims[1], s.dims[2]),
                format!(
                    "{:.2}x{:.2}x{:.2}",
                    s.spacing[0], s.spacing[1], s.spacing[2]
                ),
                n(s.si_range.0, 1),
                n(s.si_range.1, 1),
                n(s.to_anchor.translation.x, 2),
                n(s.to_anchor.translation.y, 2),
                n(s.to_anchor.translation.z, 2),
                n(s.to_anchor.rotation_deg[0], 2),
                n(s.to_anchor.rotation_deg[1], 2),
                n(s.to_anchor.rotation_deg[2], 2),
                n(s.to_anchor.residual_mm, 2),
                opt(s.body_cm3, 1),
                opt(s.body_change_cm3, 1),
            ];
            w.row(cells.iter().map(String::as_str));
        }
        w.blank();

        w.row(["Structures"]);
        let mut head: Vec<String> = vec![
            "Structure".into(),
            "Used in fit".into(),
            "Volume first cm3".into(),
            "Volume last cm3".into(),
            "Change cm3".into(),
            "Change %".into(),
            "Cut by a scan".into(),
            "Change in common FOV cm3".into(),
            "Contoured RL mm".into(),
            "AP mm".into(),
            "SI mm".into(),
            "3D mm".into(),
            "Model 3D mm (end)".into(),
            "Path mm".into(),
            "Rotation deg".into(),
            "After rigid Dice".into(),
            "HD95 mm".into(),
            "MSD mm".into(),
            "Jacobian mean".into(),
            "p5".into(),
            "p95".into(),
            "HU first".into(),
            "HU last".into(),
            "Check Dice".into(),
            "Check HD95 mm".into(),
            "WEPL first mm".into(),
            "WEPL last mm".into(),
            "WEPL change mm".into(),
        ];
        w.row(head.iter().map(String::as_str));
        for s in &self.structures {
            let (dv, dp) = s.volume_change();
            let sh = s.contoured_shift();
            let end = s.at.last().copied().unwrap_or_default();
            let model = s.model_shift(s.at.len().saturating_sub(1));
            let cells = [
                s.name.clone(),
                if s.used { "yes" } else { "no" }.to_string(),
                opt(s.volume_cm3.first().copied(), 2),
                opt(s.volume_cm3.last().copied(), 2),
                n(dv, 2),
                n(dp, 1),
                if s.is_cut() { "yes" } else { "no" }.to_string(),
                n(s.common_change(), 2),
                opt(sh.map(|v| v.x), 2),
                opt(sh.map(|v| v.y), 2),
                opt(sh.map(|v| v.z), 2),
                opt(sh.map(|v| v.length()), 2),
                opt(model.map(|v| v.length()), 2),
                n(end.path_mm, 2),
                n(end.rotation_deg, 2),
                opt(s.deformation.map(|a| a.dice), 3),
                opt(s.deformation.map(|a| a.hd95_mm), 2),
                opt(s.deformation.map(|a| a.msd_mm), 2),
                n(end.jacobian[0], 3),
                n(end.jacobian[1], 3),
                n(end.jacobian[2], 3),
                opt(s.mean_hu.first().copied().flatten(), 1),
                opt(s.mean_hu.last().copied().flatten(), 1),
                opt(s.check.map(|a| a.dice), 3),
                opt(s.check.map(|a| a.hd95_mm), 2),
                opt(s.wepl_mm.first().copied(), 1),
                opt(s.wepl_mm.last().copied(), 1),
                opt(s.wepl_change(s.wepl_mm.len().saturating_sub(1)), 1),
            ];
            w.row(cells.iter().map(String::as_str));
        }
        w.blank();
        if self.model.beam.is_some() {
            head = vec!["Structure".into(), "WEPL mm".into()];
            head.extend(time_cells.iter().cloned());
            w.row(head.iter().map(String::as_str));
            for s in &self.structures {
                let mut cells = vec![s.name.clone(), String::new()];
                cells.extend(s.wepl_mm.iter().map(|v| n(*v, 1)));
                w.row(cells.iter().map(String::as_str));
            }
            w.blank();
        }
        head = vec!["Structure".into(), "Centroid displacement 3D mm".into()];
        head.extend(time_cells.iter().cloned());
        w.row(head.iter().map(String::as_str));
        for s in &self.structures {
            let mut cells = vec![s.name.clone(), String::new()];
            for i in 0..s.at.len() {
                cells.push(opt(s.model_shift(i).map(|v| v.length()), 2));
            }
            w.row(cells.iter().map(String::as_str));
        }
        w.blank();

        w.row(["Bones and joints"]);
        w.row([
            "Body",
            "Anchor",
            "Angle deg",
            "Axis RL",
            "Axis AP",
            "Axis SI",
            "Pivot RL mm",
            "Pivot AP mm",
            "Pivot SI mm",
            "Travel mm",
            "Centroid shift mm",
            "Fit mean mm",
            "Fit RMS mm",
            "Principal-axes start",
            "Largest stretch",
            "Stretch p95",
        ]);
        for b in &self.bodies {
            let cells = [
                b.name.clone(),
                if b.anchor { "yes" } else { "no" }.to_string(),
                n(b.screw.angle_deg, 2),
                n(b.screw.axis.x, 3),
                n(b.screw.axis.y, 3),
                n(b.screw.axis.z, 3),
                n(b.screw.point.x, 1),
                n(b.screw.point.y, 1),
                n(b.screw.point.z, 1),
                n(b.screw.travel_mm, 2),
                n(b.screw.reference_shift_mm, 2),
                n(b.fit_mean_mm, 2),
                n(b.fit_rms_mm, 2),
                if b.seeded { "yes" } else { "no" }.to_string(),
                n(b.max_stretch, 4),
                n(b.stretch_p95, 4),
            ];
            w.row(cells.iter().map(String::as_str));
        }
        if !self.joints.is_empty() {
            w.blank();
            let mut head = vec!["Joint".to_string(), "About".to_string()];
            head.extend(time_cells.iter().cloned());
            w.row(head.iter().map(String::as_str));
            for j in &self.joints {
                for (a, axis) in motion::AXES.iter().enumerate() {
                    let mut cells = vec![
                        format!("{} ({} -> {})", j.name, j.first, j.second),
                        axis.to_string(),
                    ];
                    cells.extend(j.at.iter().map(|v| n(v[a], 2)));
                    w.row(cells.iter().map(String::as_str));
                }
            }
        }
        w.blank();

        w.row(["Model"]);
        let m = &self.model;
        let lines = [
            ("Mode", m.mode.label().to_string()),
            ("Anchor", m.anchor.clone()),
            ("Bodies", m.bodies.join(" ")),
            (
                "Lattice",
                format!(
                    "{}x{}x{} at {:.1} mm",
                    m.lattice_dims[0], m.lattice_dims[1], m.lattice_dims[2], m.lattice_mm
                ),
            ),
            (
                "Analysis grid mm",
                format!(
                    "{:.2}x{:.2}x{:.2}",
                    m.analysis_mm[0], m.analysis_mm[1], m.analysis_mm[2]
                ),
            ),
            ("Reach mm", n(m.reach_mm, 1)),
            (
                "Inverse consistency half way max mm",
                n(m.inverse_mm[0][0], 3),
            ),
            (
                "Inverse consistency half way p95 mm",
                n(m.inverse_mm[0][1], 3),
            ),
            ("Inverse consistency end max mm", n(m.inverse_mm[1][0], 3)),
            ("Inverse consistency end p95 mm", n(m.inverse_mm[1][1], 3)),
            ("Jacobian minimum", n(m.jacobian_min, 4)),
            ("Folded fraction", n(m.folded_fraction, 6)),
            ("Check Dice used in fit", opt(m.check_used_dice, 3)),
            ("Check Dice not used", opt(m.check_unused_dice, 3)),
            ("Overlap of the scans", n(m.overlap_fraction, 3)),
            ("Overlap SI from mm", n(m.overlap_si.0, 1)),
            ("Overlap SI to mm", n(m.overlap_si.1, 1)),
            ("WEPL beam", m.beam.clone().unwrap_or_else(|| "none".into())),
        ];
        for (k, v) in lines {
            w.row([k, v.as_str()]);
        }
        for line in &m.provenance {
            w.row(["Provenance", line.as_str()]);
        }
        w.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sampled_column_is_read_between_its_samples() {
        let t = [0.0, 0.5, 1.0];
        let v = [0.0, 10.0, 30.0];
        assert_eq!(lerp_at(&t, &v, -1.0), 0.0);
        assert!((lerp_at(&t, &v, 0.25) - 5.0).abs() < 1e-12);
        assert!((lerp_at(&t, &v, 0.75) - 20.0).abs() < 1e-12);
        assert_eq!(lerp_at(&t, &v, 2.0), 30.0);
    }

    #[test]
    fn the_beam_travels_as_iec_61217_says() {
        // Head first supine: gantry 0 shoots down, anterior to posterior
        // (+y in LPS); gantry 90 from the patient's left (+x) to the right.
        let b = Beam::iec(0.0, 0.0, "HFS");
        assert!((b.direction - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-12);
        let b = Beam::iec(90.0, 0.0, "HFS");
        assert!((b.direction - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-12);
        // Prone, the same gantry 0 beam enters the back.
        let b = Beam::iec(0.0, 0.0, "HFP");
        assert!((b.direction - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-12);
        // The couch turned 90 degrees: the lateral beam runs head to foot.
        let b = Beam::iec(90.0, 90.0, "HFS");
        assert!(b.direction.z.abs() > 0.999, "{:?}", b.direction);
        assert!(Beam::iec(0.0, 0.0, "").label.contains("HFS assumed"));
    }

    #[test]
    fn procrustes_recovers_a_rigid_motion() {
        let g = Rigid {
            r: se3::exp(&se3::Twist {
                w: Vec3::new(0.1, 0.5, -0.2),
                v: Vec3::ZERO,
            })
            .r,
            t: Vec3::new(3.0, -4.0, 5.0),
        };
        let pts: Vec<Vec3> = (0..50)
            .map(|i| {
                let f = i as f64;
                Vec3::new((f * 0.7).sin() * 20.0, (f * 1.3).cos() * 15.0, f - 25.0)
            })
            .collect();
        let moved: Vec<Vec3> = pts.iter().map(|&p| g.map(p)).collect();
        let back = procrustes(&pts, &moved);
        for p in &pts {
            assert!((back.map(*p) - g.map(*p)).length() < 1e-9);
        }
    }
}
