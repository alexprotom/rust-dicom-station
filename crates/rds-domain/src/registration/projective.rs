//! 2D/3D registration: the rigid pose of the planning CT whose projections
//! match radiographs taken at treatment.
//!
//! Each radiograph comes with its camera ([`ProjectionGeometry`]) in the
//! planned patient frame - where the beam would see the patient if he lay
//! exactly as planned. The pose `T` maps that frame onto the CT (the
//! registration's fixed → moving convention): the DRR at pose `T` is the CT
//! rendered through the camera moved by `T`, so no voxel is ever resampled.
//! `T` is the correction - applied to the patient it puts him back where he
//! was planned - and `T⁻¹` is how he was found displaced.
//!
//! The run, per [`register_2d3d`]:
//!
//! 1. The CT as attenuation ([`MuVolume`]), cropped to the clip box, with
//!    coarser copies; each radiograph (as line integrals) and its mask
//!    halved to at most [`Request2d3d::max_pixels`] and then into a pyramid,
//!    each level with the matching coarser camera.
//! 2. **Polarity**: the NCC of each radiograph with its DRR at the start;
//!    a radiograph stored bone-dark is turned so it reads as the DRR does.
//! 3. **Start**: the plan (or a given pose), then optionally the
//!    *in-plane* search - per view the 2D shift of the DRR that best matches
//!    the radiograph, the shifts of all views solved for one 3D translation
//!    by least squares through the projection's Jacobian at the isocentre -
//!    or a *grid* over the three translations.
//! 4. **Levels**, coarse to fine: Powell (or Nelder-Mead) over the six
//!    parameters in millimetre-like units (rotations times the clip box's
//!    half diagonal), inside bounds about the start, every evaluation a
//!    rendering of every view and the weighted similarity. What a measure
//!    needs of a radiograph alone (its gradients, its window sums, its
//!    histogram bins) is computed once per level and view
//!    ([`Metric2d::fixed`]), so an evaluation computes the rendering's half.
//! 5. Optionally a finite-difference L-BFGS refinement at the finest
//!    level; then the curvature of the cost at the answer, whose weak
//!    directions are reported when they are much weaker than the strongest
//!    of their kind (one view says little about depth).
//!
//! The renderings run on the CPU, or on the GPU when the request asks for it
//! ([`Request2d3d::device`]): the same ray-cast as a compute shader
//! ([`GpuProjector`]), each mu-volume a level reads uploaded once, the views
//! of an evaluation rendered in one submission, the similarity still
//! computed on the CPU. When the GPU was only preferred
//! (`Auto`) and fails, the run goes on with the CPU and says so.

use std::cell::{OnceCell, RefCell};
use std::time::Instant;

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::metric2d::{border_mask, halve_mask, ncc, Fixed, Image2d, Mask2d, Metric2d};
use super::optim::{self, Bounds, Limits, Optimum};
use super::RigidTransform;
use crate::drr::HuMode;
use crate::geometry::Vec3;
use crate::iec::{self, CouchCorrection, PatientFrame, M3};
use crate::nn::device::{describe_cpu, DevicePref};
use crate::progress::{ProgressSink, CANCELLED};
use crate::projection::gpu::GpuProjector;
use crate::projection::{render_posed, render_posed_in, MuVolume, ProjectionGeometry};
use crate::volume::Volume;

/// One radiograph and where it was taken from.
#[derive(Clone, Debug)]
pub struct View2d {
    pub label: String,
    /// The pixels as line integrals (bright where much attenuates), or as
    /// detector signal when `transmission` is set.
    pub image: Image2d,
    /// In the planned patient frame; `dims` equal to the image's.
    pub camera: ProjectionGeometry,
    /// Pixels that count; the whole image less a border when `None`.
    pub mask: Option<Mask2d>,
    pub weight: f64,
    /// The values are transmitted signal: `−ln(I / I₀)` is applied first.
    pub transmission: bool,
}

/// The search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Optimizer2d {
    #[default]
    Powell,
    NelderMead,
}

impl Optimizer2d {
    pub const ALL: [Optimizer2d; 2] = [Optimizer2d::Powell, Optimizer2d::NelderMead];

    pub fn label(self) -> &'static str {
        match self {
            Optimizer2d::Powell => "Powell",
            Optimizer2d::NelderMead => "Nelder-Mead",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Optimizer2d::Powell => "powell",
            Optimizer2d::NelderMead => "nelder-mead",
        }
    }

    pub fn from_key(s: &str) -> Option<Optimizer2d> {
        let s = s.trim().to_ascii_lowercase().replace(['_', ' '], "-");
        Optimizer2d::ALL
            .into_iter()
            .find(|o| o.key() == s || (s == "nm" && *o == Optimizer2d::NelderMead))
    }
}

/// Where the search starts, beyond the plan or the given pose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Init2d3d {
    /// The plan, or [`Request2d3d::start`].
    Plan,
    /// The plan, moved by the translation the 2D shifts of the views say.
    #[default]
    InPlane,
    /// The best of a grid over the three translations.
    Grid,
}

impl Init2d3d {
    pub const ALL: [Init2d3d; 3] = [Init2d3d::Plan, Init2d3d::InPlane, Init2d3d::Grid];

    pub fn label(self) -> &'static str {
        match self {
            Init2d3d::Plan => "Plan",
            Init2d3d::InPlane => "In-plane search",
            Init2d3d::Grid => "Translation grid",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Init2d3d::Plan => "plan",
            Init2d3d::InPlane => "in-plane",
            Init2d3d::Grid => "grid",
        }
    }

    pub fn from_key(s: &str) -> Option<Init2d3d> {
        let s = s.trim().to_ascii_lowercase().replace(['_', ' '], "-");
        Init2d3d::ALL.into_iter().find(|o| o.key() == s)
    }
}

/// Which parameters move.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Dof2d3d {
    #[default]
    Six,
    /// The rotations stay at the start's.
    Translations,
}

/// An axis-aligned box in patient millimetres: rays integrate inside it
/// only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipBox {
    pub center: Vec3,
    /// Half the size along x, y, z.
    pub half: Vec3,
}

impl ClipBox {
    /// A cube of half-size `half_mm` about `center`.
    pub fn around(center: Vec3, half_mm: f64) -> ClipBox {
        ClipBox {
            center,
            half: Vec3::new(half_mm, half_mm, half_mm),
        }
    }

    /// Min and max corners.
    pub fn corners(&self) -> (Vec3, Vec3) {
        (self.center - self.half, self.center + self.half)
    }

    pub fn half_diagonal(&self) -> f64 {
        self.half.length()
    }
}

/// Everything a run needs beyond the CT.
#[derive(Clone, Debug)]
pub struct Request2d3d {
    pub views: Vec<View2d>,
    /// The centre of rotation, and the point the readout is about.
    pub isocenter: Vec3,
    /// How the patient lies on the table top, for the couch readout.
    pub patient: PatientFrame,
    /// The region that matters: its shadow on each image is what the
    /// measure compares.
    pub clip: Option<ClipBox>,
    /// The rays integrate inside the clip box only. Otherwise (the default)
    /// the DRRs see the whole CT, as a radiograph sees the whole patient,
    /// and the box says only which pixels count: a box that cuts the rays
    /// leaves out of the DRR what the radiograph still shows.
    pub clip_rays: bool,
    pub threshold_hu: f32,
    pub hu: HuMode,
    pub metric: Metric2d,
    pub optimizer: Optimizer2d,
    /// Pyramid levels, coarsest first in the run.
    pub levels: usize,
    pub init: Init2d3d,
    pub dof: Dof2d3d,
    /// Finite-difference L-BFGS after the search at the finest level.
    pub refine_gradient: bool,
    /// The curvature at the answer, and the warning when one direction is
    /// weak. 1 + 2n + 2n(n − 1) renderings at the second-finest level.
    pub conditioning: bool,
    /// Start here rather than at the plan (any centre; recentred on the
    /// isocentre).
    pub start: Option<RigidTransform>,
    /// The finest level's longest side, pixels.
    pub max_pixels: usize,
    /// Ray step at the finest level, mm; doubled per coarser level.
    pub step_mm: f64,
    /// Bounds about the start: translation, mm, and rotation, degrees.
    pub range_mm: f64,
    pub range_deg: f64,
    /// [`Metric2d::LocalNcc`]'s window half-width at the finest level.
    pub local_radius: usize,
    /// Per level.
    pub max_evaluations: usize,
    /// Where the renderings run.
    pub device: DevicePref,
}

impl Request2d3d {
    pub fn new(views: Vec<View2d>, isocenter: Vec3) -> Self {
        Request2d3d {
            views,
            isocenter,
            patient: PatientFrame::default(),
            clip: None,
            clip_rays: false,
            threshold_hu: -800.0,
            hu: HuMode::Water,
            metric: Metric2d::Ncc,
            optimizer: Optimizer2d::Powell,
            levels: 3,
            init: Init2d3d::InPlane,
            dof: Dof2d3d::Six,
            refine_gradient: false,
            conditioning: true,
            start: None,
            max_pixels: 256,
            step_mm: 1.0,
            range_mm: 30.0,
            range_deg: 15.0,
            local_radius: 7,
            max_evaluations: 1500,
            device: DevicePref::Cpu,
        }
    }
}

/// What became of one view.
#[derive(Clone, Debug)]
pub struct ViewOutcome {
    pub label: String,
    /// The similarity at the start and at the answer, finest level.
    pub before: f64,
    pub after: f64,
    /// The radiograph was turned to read as the DRR does.
    pub inverted: bool,
    /// At the finest level: the radiograph as compared, the DRR at the
    /// answer, their camera and mask.
    pub image: Image2d,
    pub drr: Image2d,
    pub camera: ProjectionGeometry,
    pub mask: Mask2d,
}

/// One line of the log per level.
#[derive(Clone, Debug)]
pub struct LevelLine {
    /// 1 is the coarsest.
    pub level: usize,
    /// Of the first view.
    pub pixels: [usize; 2],
    pub step_mm: f64,
    pub evaluations: usize,
    pub before: f64,
    pub after: f64,
    pub secs: f64,
}

/// The answer.
#[derive(Clone, Debug)]
pub struct Result2d3d {
    /// The correction, planned frame → CT, about the isocentre.
    pub transform: RigidTransform,
    /// The correction as moves of the patient support.
    pub couch: CouchCorrection,
    /// `T⁻¹`, how the patient was found: rotations about x, y, z in
    /// degrees, then translations in mm, patient LPS.
    pub displacement: [f64; 6],
    pub metric: Metric2d,
    pub views: Vec<ViewOutcome>,
    pub levels: Vec<LevelLine>,
    /// Where the search began, in words.
    pub start: String,
    pub evaluations: usize,
    pub iterations: usize,
    pub elapsed_secs: f64,
    /// Eigenvalues of the cost's curvature at the answer, in the search's
    /// units, strongest first; empty when not computed.
    pub curvature: Vec<f64>,
    pub warning: Option<String>,
    /// What the markers said, when they took part.
    pub markers: Option<super::markers::MarkerReport>,
    /// Where the renderings ran, as the user reads it.
    pub device: String,
}

const AXES: [&str; 3] = ["L-R", "A-P", "S-I"];

/// The weight of the pull of the translations towards where a level starts,
/// per square millimetre: about a hundredth of what a millimetre in the
/// plane of a view costs.
const PRIOR: f64 = 1e-5;

/// A direction whose curvature is under this fraction of the strongest of
/// its kind is reported as weakly constrained.
const WEAK: f64 = 0.02;

impl Result2d3d {
    /// The correction's parameters: rotations about x, y, z in degrees,
    /// then translations in mm, patient LPS.
    pub fn correction(&self) -> [f64; 6] {
        let p = self.transform.params();
        [
            p[0].to_degrees(),
            p[1].to_degrees(),
            p[2].to_degrees(),
            p[3],
            p[4],
            p[5],
        ]
    }

    /// `T  x +1.20 · y −0.40 · z +3.10 mm · rx +0.50 · ry −0.20 · rz +0.10°`
    pub fn correction_line(&self) -> String {
        six_line(&self.correction())
    }

    pub fn displacement_line(&self) -> String {
        six_line(&self.displacement)
    }

    /// The similarity before and after, averaged over the views with their
    /// weights as the cost had them.
    pub fn metric_line(&self) -> String {
        let n = self.views.len().max(1) as f64;
        let b: f64 = self.views.iter().map(|v| v.before).sum::<f64>() / n;
        let a: f64 = self.views.iter().map(|v| v.after).sum::<f64>() / n;
        format!(
            "{} {:.3} ▶ {:.3}  ({} evaluations, {:.1} s)",
            self.metric.tag(),
            b,
            a,
            self.evaluations,
            self.elapsed_secs
        )
    }

    /// The whole result as text, one fact per line.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Correction (LPS): {}\n", self.correction_line()));
        s.push_str(&format!("Couch: {}\n", self.couch.line()));
        s.push_str(&format!(
            "Displacement (LPS): {}\n",
            self.displacement_line()
        ));
        s.push_str(&format!("Start: {}\n", self.start));
        s.push_str(&format!("{}\n", self.metric_line()));
        if !self.device.is_empty() {
            s.push_str(&format!("Projector: {}\n", self.device));
        }
        for v in &self.views {
            s.push_str(&format!(
                "{}: {} {:.3} ▶ {:.3}{}\n",
                v.label,
                self.metric.tag(),
                v.before,
                v.after,
                if v.inverted { ", inverted" } else { "" }
            ));
        }
        for l in &self.levels {
            s.push_str(&format!(
                "Level {}: {}×{} px, step {:.1} mm, {} evaluations, {:.3} ▶ {:.3}, {:.2} s\n",
                l.level,
                l.pixels[0],
                l.pixels[1],
                l.step_mm,
                l.evaluations,
                l.before,
                l.after,
                l.secs
            ));
        }
        if let Some(m) = &self.markers {
            s.push_str(&m.report());
        }
        if let Some(w) = &self.warning {
            s.push_str(&format!("⚠ {w}\n"));
        }
        s
    }

    /// The same answer as a 3D registration result, for everything that
    /// shows, compares or exports one.
    pub fn to_registration(&self, vol: &Volume) -> super::RegistrationResult {
        let n = self.views.len().max(1) as f64;
        let t = std::sync::Arc::new(super::Transform3::rigid_only(self.transform.clone()));
        super::RegistrationResult {
            analysis: super::analysis::analyse(vol, &t, None),
            transform: t,
            method: super::RegMethod::Projective,
            metric: super::Metric::Projective(self.metric),
            initial_metric: self.views.iter().map(|v| v.before).sum::<f64>() / n,
            final_metric: self.views.iter().map(|v| v.after).sum::<f64>() / n,
            iterations_run: self.iterations,
            elapsed_secs: self.elapsed_secs,
            region: None,
        }
    }
}

fn six_line(p: &[f64; 6]) -> String {
    format!(
        "x {:+.2} · y {:+.2} · z {:+.2} mm · rx {:+.2} · ry {:+.2} · rz {:+.2}°",
        p[3], p[4], p[5], p[0], p[1], p[2]
    )
}

fn to_m3(m: [f64; 9]) -> M3 {
    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
}

/// `R = Rz(c) Ry(b) Rx(a)` → `(a, b, c)`, radians.
fn euler_zyx(m: &M3) -> [f64; 3] {
    let b = -m[2][0].clamp(-1.0, 1.0).asin();
    let a = m[2][1].atan2(m[2][2]);
    let c = m[1][0].atan2(m[0][0]);
    [a, b, c]
}

/// One view at one level.
struct Level {
    image: Image2d,
    mask: Mask2d,
    camera: ProjectionGeometry,
    /// The measure's half of the radiograph over the mask, made on the
    /// first evaluation and kept: every later one computes only the
    /// rendering's half.
    fixed: OnceCell<Fixed>,
}

/// The pixel size at the isocentre, mm.
pub(crate) fn pixel_at(cam: &ProjectionGeometry, p: Vec3) -> f64 {
    match cam.project_with_jacobian(p) {
        Some((_, [di, dj])) => {
            let g = 0.5 * (di.length() + dj.length());
            if g > 1e-12 {
                1.0 / g
            } else {
                cam.spacing()[0]
            }
        }
        None => cam.spacing()[0],
    }
}

/// The best integer shift `s` (pixels) of `drr` against `image` - the one
/// with `image(u) ≈ drr(u − s)` - within `±r`, refined to a fraction of a
/// pixel by a parabola through its neighbours; and its NCC.
fn best_shift(
    image: &[f32],
    drr: &[f32],
    dims: [usize; 2],
    mask: &[u8],
    r: isize,
) -> ([f64; 2], f64) {
    let [w, h] = dims;
    let side = (2 * r + 1) as usize;
    let shifts: Vec<(isize, isize)> = (-r..=r)
        .flat_map(|sy| (-r..=r).map(move |sx| (sx, sy)))
        .collect();
    let scores: Vec<f64> = shifts
        .par_iter()
        .map(|&(sx, sy)| {
            let (mut n, mut sa, mut sb, mut saa, mut sbb, mut sab) =
                (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
            for j in 0..h as isize {
                let jj = j - sy;
                if jj < 0 || jj >= h as isize {
                    continue;
                }
                for i in 0..w as isize {
                    let ii = i - sx;
                    if ii < 0 || ii >= w as isize {
                        continue;
                    }
                    let u = j as usize * w + i as usize;
                    if mask[u] == 0 {
                        continue;
                    }
                    let a = image[u] as f64;
                    let b = drr[jj as usize * w + ii as usize] as f64;
                    n += 1.0;
                    sa += a;
                    sb += b;
                    saa += a * a;
                    sbb += b * b;
                    sab += a * b;
                }
            }
            if n < 16.0 {
                return f64::NEG_INFINITY;
            }
            let va = saa - sa * sa / n;
            let vb = sbb - sb * sb / n;
            if va <= 1e-12 || vb <= 1e-12 {
                return f64::NEG_INFINITY;
            }
            (sab - sa * sb / n) / (va * vb).sqrt()
        })
        .collect();
    let mut best = 0usize;
    for (k, &s) in scores.iter().enumerate() {
        if s > scores[best] {
            best = k;
        }
    }
    let (bx, by) = ((best % side) as isize, (best / side) as isize);
    let at = |x: isize, y: isize| -> Option<f64> {
        if x < 0 || y < 0 || x >= side as isize || y >= side as isize {
            return None;
        }
        let v = scores[y as usize * side + x as usize];
        v.is_finite().then_some(v)
    };
    let refine = |m: Option<f64>, c: f64, p: Option<f64>| -> f64 {
        match (m, p) {
            (Some(m), Some(p)) => {
                let den = m - 2.0 * c + p;
                if den < -1e-12 {
                    (0.5 * (m - p) / den).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    };
    let c = scores[best];
    let fx = refine(at(bx - 1, by), c, at(bx + 1, by));
    let fy = refine(at(bx, by - 1), c, at(bx, by + 1));
    ([shifts[best].0 as f64 + fx, shifts[best].1 as f64 + fy], c)
}

/// Solve the 3 × 3 system `a x = b`; `None` when singular.
fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&a);
    if d.abs() < 1e-300 {
        return None;
    }
    let mut x = [0.0; 3];
    for (c, xc) in x.iter_mut().enumerate() {
        let mut m = a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        *xc = det(&m) / d;
    }
    Some(x)
}

/// Eigenvalues and eigenvectors of a symmetric matrix (cyclic Jacobi),
/// eigenvalues descending, vectors as rows.
fn symmetric_eigen(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| (i == j) as u8 as f64).collect())
        .collect();
    for _ in 0..64 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|(i, j)| i != j)
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                let (rp, rq) = (a[p].clone(), a[q].clone());
                for (k, (apk, aqk)) in rp.into_iter().zip(rq).enumerate() {
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let vals = order.iter().map(|&i| a[i][i]).collect();
    let vecs = order
        .iter()
        .map(|&i| (0..n).map(|k| v[k][i]).collect())
        .collect();
    (vals, vecs)
}

/// The engine's state over a run: the pyramids, the pose parameterisation
/// and the evaluation count.
struct Engine<'a> {
    req: &'a Request2d3d,
    sink: &'a dyn ProgressSink,
    /// `levels[l][v]`, 0 the finest.
    levels: Vec<Vec<Level>>,
    /// Per level and view, the pixels the cost renders: the mask grown by
    /// what the measure's kernels reach beyond it.
    reach: Vec<Vec<Mask2d>>,
    mu: Vec<MuVolume>,
    /// Mu-volume index and ray step per level.
    mu_of_level: Vec<usize>,
    step_of_level: Vec<f64>,
    /// Millimetres per radian of the rotation parameters.
    scale: f64,
    /// The start's parameters, radians and mm, about the isocentre.
    base: [f64; 6],
    evaluations: usize,
    cancelled: bool,
    weights: f64,
    /// A projector on the GPU per mu-volume a level reads, when the run
    /// uses one.
    gpu: Vec<Option<GpuProjector>>,
    /// Why the GPU was left, when it failed in a run that only preferred it.
    gpu_failed: RefCell<Option<String>>,
}

impl Engine<'_> {
    fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// The search vector to a pose.
    fn pose(&self, x: &[f64]) -> RigidTransform {
        let mut p = self.base;
        match self.req.dof {
            Dof2d3d::Six => {
                for k in 0..3 {
                    p[k] += x[k] / self.scale;
                    p[k + 3] += x[k + 3];
                }
            }
            Dof2d3d::Translations => {
                for k in 0..3 {
                    p[k + 3] += x[k];
                }
            }
        }
        RigidTransform::new(p, self.req.isocenter)
    }

    fn dims(&self) -> usize {
        match self.req.dof {
            Dof2d3d::Six => 6,
            Dof2d3d::Translations => 3,
        }
    }

    /// The whole DRR of `view` at `level` and pose `t`, for display and
    /// the searches that shift it.
    fn render(&self, level: usize, view: usize, t: &RigidTransform) -> Result<Vec<f32>> {
        let lv = &self.levels[level][view];
        let m = self.mu_of_level[level];
        let cam = lv.camera.transformed(t);
        let step = self.step_of_level[level];
        if self.gpu_failed.borrow().is_none() {
            if let Some(Some(gpu)) = self.gpu.get(m) {
                match gpu.render(&cam, step) {
                    Ok(px) => return Ok(px),
                    Err(e) if self.req.device == DevicePref::Gpu => return Err(e),
                    Err(e) => *self.gpu_failed.borrow_mut() = Some(e.to_string()),
                }
            }
        }
        render_posed(&self.mu[m], &cam, step, self.sink)
    }

    /// Where the renderings ran.
    fn device_line(&self, gpu: &str) -> String {
        match (
            &*self.gpu_failed.borrow(),
            self.gpu.iter().any(Option::is_some),
        ) {
            (Some(why), _) => format!("{} (the GPU failed: {why})", describe_cpu()),
            (None, true) => gpu.to_string(),
            (None, false) => describe_cpu(),
        }
    }

    fn radius(&self, level: usize) -> usize {
        (self.req.local_radius >> level).max(2)
    }

    /// The DRR of every view at `level` and pose `t`, on the CPU only the
    /// pixels the measure reads; on the GPU all of them in one submission.
    fn render_all(&self, level: usize, t: &RigidTransform) -> Result<Vec<Vec<f32>>> {
        let m = self.mu_of_level[level];
        let step = self.step_of_level[level];
        let cams: Vec<ProjectionGeometry> = self.levels[level]
            .iter()
            .map(|lv| lv.camera.transformed(t))
            .collect();
        if self.gpu_failed.borrow().is_none() {
            if let Some(Some(gpu)) = self.gpu.get(m) {
                match gpu.render_views(&cams, step) {
                    Ok(px) => return Ok(px),
                    Err(e) if self.req.device == DevicePref::Gpu => return Err(e),
                    Err(e) => *self.gpu_failed.borrow_mut() = Some(e.to_string()),
                }
            }
        }
        cams.iter()
            .enumerate()
            .map(|(v, cam)| {
                let mask = Some(self.reach[level][v].as_slice());
                render_posed_in(&self.mu[m], cam, step, mask, self.sink)
            })
            .collect()
    }

    /// Per view similarity at `level` for pose `t`.
    fn similarities(&self, level: usize, t: &RigidTransform) -> Result<Vec<f64>> {
        let drrs = self.render_all(level, t)?;
        let mut out = Vec::with_capacity(drrs.len());
        for (lv, drr) in self.levels[level].iter().zip(&drrs) {
            let fixed = lv.fixed.get_or_init(|| {
                self.req
                    .metric
                    .fixed(&lv.image.data, lv.image.dims, Some(&lv.mask))
            });
            out.push(fixed.similarity(&lv.image.data, drr, Some(&lv.mask), self.radius(level)));
        }
        Ok(out)
    }

    /// The cost: minus the weighted mean similarity. A cancelled run
    /// answers infinity until the optimizer gives up.
    fn cost(&mut self, level: usize, x: &[f64]) -> f64 {
        if self.cancelled || self.sink.cancelled() {
            self.cancelled = true;
            return f64::INFINITY;
        }
        let t = self.pose(x);
        let sims = match self.similarities(level, &t) {
            Ok(s) => s,
            Err(_) => {
                self.cancelled = true;
                return f64::INFINITY;
            }
        };
        self.evaluations += 1;
        let s: f64 = sims
            .iter()
            .zip(&self.req.views)
            .map(|(s, v)| s * v.weight)
            .sum();
        -s / self.weights
    }

    fn report(&self, level: usize, evals: usize, value: f64) {
        let n = self.n_levels();
        let done = (n - 1 - level) as f32;
        let frac =
            (done + (evals as f32 / self.req.max_evaluations.max(1) as f32).min(1.0)) / n as f32;
        self.sink.report(
            frac.min(0.99),
            &format!(
                "level {} of {}, {} {:.3}",
                n - level,
                n,
                self.req.metric.tag(),
                -value
            ),
        );
    }
}

/// Register `req.views` to `vol`.
pub fn register_2d3d(
    vol: &Volume,
    req: &Request2d3d,
    sink: &dyn ProgressSink,
) -> Result<Result2d3d> {
    let started = Instant::now();
    if req.views.is_empty() {
        bail!("no radiograph to register");
    }
    for v in &req.views {
        if v.image.dims != v.camera.dims {
            bail!(
                "{}: the image is {}×{} but its geometry {}×{}",
                v.label,
                v.image.dims[0],
                v.image.dims[1],
                v.camera.dims[0],
                v.camera.dims[1]
            );
        }
        if v.image.len() != v.image.dims[0] * v.image.dims[1] || v.image.is_empty() {
            bail!("{}: the image has no pixels", v.label);
        }
    }
    let weights: f64 = req.views.iter().map(|v| v.weight.max(0.0)).sum();
    if weights <= 0.0 {
        bail!("every view has weight zero");
    }
    let n_levels = req.levels.clamp(1, 6);
    let iso = req.isocenter;

    let start = req
        .start
        .as_ref()
        .map(|s| s.recentered(iso))
        .unwrap_or_else(|| RigidTransform::identity(iso));

    // Pyramids.
    let mut levels: Vec<Vec<Level>> = (0..n_levels).map(|_| Vec::new()).collect();
    for v in &req.views {
        let (mut img, mut mask, mut cam) =
            working_view(v, req.max_pixels, req.clip.as_ref().map(|c| (c, &start)))?;
        for lv in levels.iter_mut() {
            lv.push(Level {
                image: img.clone(),
                mask: mask.clone(),
                camera: cam,
                fixed: OnceCell::new(),
            });
            let (m, _) = halve_mask(&mask, img.dims);
            img = img.halved();
            mask = m;
            cam = cam.coarser(2);
        }
    }

    // The CT as attenuation, and which copy each level reads.
    sink.report(0.0, "preparing the CT");
    let clip = req.clip.filter(|_| req.clip_rays).map(|c| c.corners());
    let mu = MuVolume::prepare(vol, req.hu, req.threshold_hu, clip).levels(n_levels + 1);
    // Each level reads the coarsest copy whose finest spacing is still no
    // larger than its pixels at the isocentre: what a pixel cannot resolve
    // need not be read.
    let mut mu_of_level = Vec::with_capacity(n_levels);
    let mut step_of_level = Vec::with_capacity(n_levels);
    for (l, lv) in levels.iter().enumerate() {
        let px = pixel_at(&lv[0].camera, iso);
        let pick = (0..mu.len())
            .rev()
            .find(|&m| mu[m].finest_spacing() <= 1.01 * px)
            .unwrap_or(0);
        mu_of_level.push(pick);
        step_of_level.push(req.step_mm.max(0.05) * (1u64 << l) as f64);
    }

    // The GPU, when asked for: each copy a level reads, uploaded once. A
    // device that was only preferred and fails here leaves the CPU to it.
    let mut gpu: Vec<Option<GpuProjector>> = (0..mu.len()).map(|_| None).collect();
    let mut gpu_failed = None;
    let mut gpu_name = String::new();
    // A renderer in software is slower at this than the CPU path, so it is
    // used only when asked for by name.
    let ctx = req
        .device
        .resolve()?
        .filter(|c| req.device == DevicePref::Gpu || !c.is_software());
    if let Some(ctx) = ctx {
        sink.report(0.0, "uploading the CT to the GPU");
        gpu_name = ctx.describe();
        for &m in &mu_of_level {
            if gpu[m].is_some() {
                continue;
            }
            match GpuProjector::new(&ctx, &mu[m]) {
                Ok(p) => gpu[m] = Some(p),
                Err(e) if req.device == DevicePref::Gpu => return Err(e),
                Err(e) => {
                    gpu_failed = Some(e.to_string());
                    gpu.iter_mut().for_each(|g| *g = None);
                    break;
                }
            }
        }
    }

    let scale = req
        .clip
        .map(|c| c.half_diagonal())
        .unwrap_or(100.0)
        .max(10.0);
    // A gradient, a window or a pattern read pixels next to the ones they
    // are for: those are rendered too.
    let reach: Vec<Vec<Mask2d>> = levels
        .iter()
        .enumerate()
        .map(|(l, lv)| {
            let r = (req.local_radius >> l).max(2).max(3) + 1;
            lv.iter().map(|v| grown(&v.mask, v.image.dims, r)).collect()
        })
        .collect();
    let mut eng = Engine {
        req,
        sink,
        levels,
        reach,
        mu,
        mu_of_level,
        step_of_level,
        scale,
        base: start.params(),
        evaluations: 0,
        cancelled: false,
        weights,
        gpu,
        gpu_failed: RefCell::new(gpu_failed),
    };
    let coarsest = n_levels - 1;

    // Polarity, at the start.
    let mut inverted = vec![false; req.views.len()];
    for (v, inv) in inverted.iter_mut().enumerate() {
        let drr = eng.render(coarsest, v, &start)?;
        let lv = &eng.levels[coarsest][v];
        if ncc(&lv.image.data, &drr, Some(&lv.mask)) < 0.0 {
            *inv = true;
            for level in eng.levels.iter_mut() {
                level[v].image.negate();
                level[v].fixed = OnceCell::new();
            }
        }
    }

    // The start.
    let n = eng.dims();
    let mut x = vec![0.0; n];
    let mut start_line = match &req.start {
        Some(_) => "the given pose".to_string(),
        None => "the plan".to_string(),
    };
    match req.init {
        Init2d3d::Plan => {}
        Init2d3d::InPlane => {
            if let Some(t) = in_plane(&mut eng, &start)? {
                let off = |x: &mut [f64], t: [f64; 3]| {
                    let k = x.len() - 3;
                    x[k..].copy_from_slice(&t);
                };
                let mut trial = x.clone();
                off(&mut trial, t);
                let (c0, c1) = (eng.cost(coarsest, &x), eng.cost(coarsest, &trial));
                if c1 < c0 {
                    x = trial;
                    start_line = format!(
                        "{} with the in-plane search, {:+.1}, {:+.1}, {:+.1} mm",
                        start_line, t[0], t[1], t[2]
                    );
                }
            }
        }
        Init2d3d::Grid => {
            let steps = 5i32;
            let d = req.range_mm / steps as f64;
            let mut best = (eng.cost(coarsest, &x), [0.0; 3]);
            for a in -steps..=steps {
                for b in -steps..=steps {
                    for c in -steps..=steps {
                        let t = [a as f64 * d, b as f64 * d, c as f64 * d];
                        let mut trial = x.clone();
                        let k = trial.len() - 3;
                        trial[k..].copy_from_slice(&t);
                        let f = eng.cost(coarsest, &trial);
                        if f < best.0 {
                            best = (f, t);
                        }
                    }
                }
                if eng.cancelled {
                    bail!(CANCELLED);
                }
                eng.report(coarsest, 0, best.0);
            }
            let k = x.len() - 3;
            x[k..].copy_from_slice(&best.1);
            start_line = format!(
                "{} with the translation grid, {:+.1}, {:+.1}, {:+.1} mm",
                start_line, best.1[0], best.1[1], best.1[2]
            );
        }
    }
    if eng.cancelled {
        bail!(CANCELLED);
    }

    // Bounds about the start, in the search's units.
    let rot = req.range_deg.to_radians() * scale;
    let bounds: Bounds = match req.dof {
        Dof2d3d::Six => vec![
            (-rot, rot),
            (-rot, rot),
            (-rot, rot),
            (-req.range_mm, req.range_mm),
            (-req.range_mm, req.range_mm),
            (-req.range_mm, req.range_mm),
        ],
        Dof2d3d::Translations => vec![(-req.range_mm, req.range_mm); 3],
    };
    for (v, (lo, hi)) in x.iter_mut().zip(&bounds) {
        *v = v.clamp(*lo, *hi);
    }

    // Coarse to fine.
    let mut lines = Vec::new();
    let mut iterations = 0;
    for level in (0..n_levels).rev() {
        let t_level = Instant::now();
        let before_evals = eng.evaluations;
        let before = eng.cost(level, &x);
        let f = (1u64 << level) as f64;
        let lim = Limits {
            max_evaluations: req.max_evaluations.max(20),
            ftol: 1e-6,
            xtol: 0.05 * f,
            initial_step: (2.0 * f).min(0.5 * req.range_mm.max(1.0)),
        };
        let opt: Optimum = {
            let mut evals = 0usize;
            let centre = x.clone();
            let mut cost = |p: &[f64]| {
                // A faint pull towards where this level starts: a direction
                // the images do not constrain (the depth of one view) stays
                // where it was instead of drifting to a bound; one they do
                // is not moved by it.
                // Translations only: a rotation's unit moves the anatomy
                // less, and its curvature is of the size of this pull.
                let k = p.len() - 3;
                let pull: f64 = p[k..]
                    .iter()
                    .zip(&centre[k..])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum();
                let c = eng.cost(level, p) + PRIOR * pull;
                evals += 1;
                if evals.is_multiple_of(10) {
                    eng.report(level, evals, c);
                }
                c
            };
            match req.optimizer {
                Optimizer2d::Powell => optim::powell(&mut cost, &x, &bounds, lim),
                Optimizer2d::NelderMead => optim::nelder_mead(&mut cost, &x, &bounds, lim),
            }
        };
        if eng.cancelled {
            bail!(CANCELLED);
        }
        iterations += opt.iterations;
        if opt.value <= before {
            x = opt.x;
        }
        lines.push(LevelLine {
            level: n_levels - level,
            pixels: eng.levels[level][0].image.dims,
            step_mm: eng.step_of_level[level],
            evaluations: eng.evaluations - before_evals,
            before: -before,
            after: -opt.value.min(before),
            secs: t_level.elapsed().as_secs_f64(),
        });
    }

    // Gradient refinement.
    if req.refine_gradient {
        let h = 0.5;
        let before = eng.cost(0, &x);
        let fg = |p: &[f64], g: &mut [f64]| -> f64 {
            let f0 = eng.cost(0, p);
            let mut q = p.to_vec();
            for k in 0..p.len() {
                q[k] = p[k] + h;
                let fp = eng.cost(0, &q);
                q[k] = p[k] - h;
                let fm = eng.cost(0, &q);
                q[k] = p[k];
                g[k] = (fp - fm) / (2.0 * h);
            }
            f0
        };
        let opt = optim::lbfgs(fg, &x, &bounds, 10, 1.0);
        if eng.cancelled {
            bail!(CANCELLED);
        }
        if opt.value < before {
            x = opt.x;
        }
        iterations += opt.iterations;
    }

    // Curvature at the answer.
    let (curvature, warning) = if req.conditioning {
        conditioning(&mut eng, &x)?
    } else {
        (Vec::new(), None)
    };

    // The outcome.
    let solution = eng.pose(&x);
    let mut views = Vec::with_capacity(req.views.len());
    let before = eng.similarities(0, &start)?;
    for (v, view) in req.views.iter().enumerate() {
        let drr = eng.render(0, v, &solution)?;
        let lv = &eng.levels[0][v];
        let after = req.metric.similarity(
            &lv.image.data,
            &drr,
            lv.image.dims,
            Some(&lv.mask),
            eng.radius(0),
        );
        views.push(ViewOutcome {
            label: view.label.clone(),
            before: before[v],
            after,
            inverted: inverted[v],
            image: lv.image.clone(),
            drr: Image2d::new(drr, lv.image.dims),
            camera: lv.camera,
            mask: lv.mask.clone(),
        });
    }
    let (couch, displacement) = readout(&solution, &req.patient);
    let device = eng.device_line(&gpu_name);
    sink.report(1.0, "done");
    Ok(Result2d3d {
        transform: solution,
        couch,
        displacement,
        metric: req.metric,
        views,
        levels: lines,
        start: start_line,
        evaluations: eng.evaluations,
        iterations,
        elapsed_secs: started.elapsed().as_secs_f64(),
        curvature,
        warning,
        markers: None,
        device,
    })
}

/// A view as the finest level compares it: line integrals, the mask (the
/// image less a border when none is given), halved until its longest side
/// is at most `max_pixels`, with the matching camera.
pub(crate) fn working_view(
    v: &View2d,
    max_pixels: usize,
    region: Option<(&ClipBox, &RigidTransform)>,
) -> Result<(Image2d, Mask2d, ProjectionGeometry)> {
    let mut img = v.image.clone();
    if v.transmission {
        img.transmission_to_line_integral();
    }
    let border = 16usize.min(img.dims[0].min(img.dims[1]) / 8);
    let mut mask = v
        .mask
        .clone()
        .unwrap_or_else(|| border_mask(img.dims, border));
    if mask.len() != img.len() {
        bail!("{}: the mask and the image differ in size", v.label);
    }
    // Where the clip box projects at the start: what the DRR can show.
    if let Some((clip, pose)) = region {
        let shadow = box_shadow(&v.camera.transformed(pose), clip);
        for (a, b) in mask.iter_mut().zip(&shadow) {
            *a &= *b;
        }
    }
    let mut cam = v.camera;
    while img.dims[0].max(img.dims[1]) > max_pixels.max(16) {
        let (m, _) = halve_mask(&mask, img.dims);
        img = img.halved();
        mask = m;
        cam = cam.coarser(2);
    }
    Ok((img, mask, cam))
}

/// `mask` grown by `r` pixels (a square neighbourhood).
fn grown(mask: &[u8], dims: [usize; 2], r: usize) -> Mask2d {
    let [w, h] = dims;
    let mut rows = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            let (a, b) = (i.saturating_sub(r), (i + r).min(w - 1));
            rows[j * w + i] = mask[j * w + a..=j * w + b].iter().any(|&m| m != 0) as u8;
        }
    }
    let mut out = vec![0u8; w * h];
    for j in 0..h {
        let (a, b) = (j.saturating_sub(r), (j + r).min(h - 1));
        for i in 0..w {
            out[j * w + i] = (a..=b).any(|k| rows[k * w + i] != 0) as u8;
        }
    }
    out
}

/// The pixels of `cam` the box projects onto: the convex hull of its eight
/// corners' projections.
pub fn box_shadow(cam: &ProjectionGeometry, clip: &ClipBox) -> Mask2d {
    let [w, h] = cam.dims;
    let (lo, hi) = clip.corners();
    let mut pts: Vec<[f64; 2]> = (0..8)
        .filter_map(|c| {
            cam.project(Vec3::new(
                if c & 1 == 0 { lo.x } else { hi.x },
                if c & 2 == 0 { lo.y } else { hi.y },
                if c & 4 == 0 { lo.z } else { hi.z },
            ))
        })
        .collect();
    if pts.len() < 8 {
        // A corner behind the source: the box surrounds it, and covers all.
        return vec![1; w * h];
    }
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f64; 2]> = Vec::new();
    for pass in 0..2 {
        let start = hull.len();
        let it: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 {
            Box::new(pts.iter())
        } else {
            Box::new(pts.iter().rev())
        };
        for &p in it {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    let mut m = vec![0u8; w * h];
    if hull.len() < 3 {
        return m;
    }
    for j in 0..h {
        for i in 0..w {
            let p = [i as f64, j as f64];
            let inside =
                (0..hull.len()).all(|k| cross(hull[k], hull[(k + 1) % hull.len()], p) >= 0.0);
            m[j * w + i] = inside as u8;
        }
    }
    m
}

/// The correction as couch moves, and the displacement `T⁻¹` as rotations
/// (degrees) and translations (mm) in patient LPS.
pub(crate) fn readout(t: &RigidTransform, patient: &PatientFrame) -> (CouchCorrection, [f64; 6]) {
    let p = t.params();
    let r = to_m3(t.matrix());
    let tr = Vec3::new(p[3], p[4], p[5]);
    let couch = iec::couch_readout(&r, tr, patient);
    let rt = iec::transpose(&r);
    let back = iec::mul_vec(&rt, tr) * -1.0;
    let e = euler_zyx(&rt);
    (
        couch,
        [
            e[0].to_degrees(),
            e[1].to_degrees(),
            e[2].to_degrees(),
            back.x,
            back.y,
            back.z,
        ],
    )
}

/// Whether `image` reads opposite to the CT's projection through `cam` at
/// `pose` (bone dark where the DRR has it bright), judged at a resolution
/// of at most 64 pixels.
pub(crate) fn inverted(
    mu: &MuVolume,
    image: &Image2d,
    mask: &[u8],
    cam: &ProjectionGeometry,
    pose: &RigidTransform,
    sink: &dyn ProgressSink,
) -> Result<bool> {
    let (mut img, mut m, mut c) = (image.clone(), mask.to_vec(), *cam);
    while img.dims[0].max(img.dims[1]) > 64 {
        let (h, _) = halve_mask(&m, img.dims);
        img = img.halved();
        m = h;
        c = c.coarser(2);
    }
    let step = mu
        .axes
        .iter()
        .map(|a| a.length())
        .fold(0.0, f64::max)
        .max(1.0);
    let drr = render_posed(mu, &c.transformed(pose), step, sink)?;
    Ok(ncc(&img.data, &drr, Some(&m)) < 0.0)
}

/// The translation the views' 2D shifts say, at the coarsest level.
fn in_plane(eng: &mut Engine, start: &RigidTransform) -> Result<Option<[f64; 3]>> {
    let coarsest = eng.n_levels() - 1;
    let req = eng.req;
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    let target = start.map(req.isocenter);
    let mut used = 0;
    for v in 0..req.views.len() {
        let drr = eng.render(coarsest, v, start)?;
        let lv = &eng.levels[coarsest][v];
        let cam0 = lv.camera.transformed(start);
        let Some((_, [di, dj])) = cam0.project_with_jacobian(target) else {
            continue;
        };
        let px = pixel_at(&cam0, target);
        let r = ((req.range_mm / px).ceil() as isize).clamp(
            1,
            (lv.image.dims[0].min(lv.image.dims[1]) / 3).max(1) as isize,
        );
        let (s, score) = best_shift(&lv.image.data, &drr, lv.image.dims, &lv.mask, r);
        if !score.is_finite() {
            continue;
        }
        // J t = −s, every view weighted as the cost weighs it.
        let w = req.views[v].weight.max(0.0);
        let rows = [(di, s[0]), (dj, s[1])];
        for (g, sv) in rows {
            let g = [g.x, g.y, g.z];
            for r in 0..3 {
                for c in 0..3 {
                    a[r][c] += w * g[r] * g[c];
                }
                b[r] -= w * g[r] * sv;
            }
        }
        used += 1;
    }
    if used == 0 {
        return Ok(None);
    }
    // A single view leaves the depth free: a small ridge picks the
    // translation with no component along it.
    let tr = a[0][0] + a[1][1] + a[2][2];
    for (k, row) in a.iter_mut().enumerate() {
        row[k] += 1e-6 * tr.max(1e-12);
    }
    let Some(t) = solve3(a, b) else {
        return Ok(None);
    };
    // In the parameters' frame: the extra translation is added after the
    // start, which the parameters' translation is.
    let lim = req.range_mm;
    Ok(Some([
        t[0].clamp(-lim, lim),
        t[1].clamp(-lim, lim),
        t[2].clamp(-lim, lim),
    ]))
}

/// The curvature of the cost at `x` (finite differences one millimetre
/// wide at the second-finest level), its eigenvalues, and a warning naming
/// the translations under [`WEAK`] of the strongest translation and the
/// rotations under it of the strongest rotation, weakest first.
fn conditioning(eng: &mut Engine, x: &[f64]) -> Result<(Vec<f64>, Option<String>)> {
    let level = 1.min(eng.n_levels() - 1);
    let n = x.len();
    let h = 1.0;
    let f0 = eng.cost(level, x);
    let mut hm = vec![vec![0.0; n]; n];
    let mut q = x.to_vec();
    let mut eval = |eng: &mut Engine, d: &[(usize, f64)]| {
        q.copy_from_slice(x);
        for &(k, s) in d {
            q[k] += s;
        }
        eng.cost(level, &q)
    };
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let fp = eval(eng, &[(i, h)]);
        let fm = eval(eng, &[(i, -h)]);
        hm[i][i] = (fp - 2.0 * f0 + fm) / (h * h);
        for j in 0..i {
            let fpp = eval(eng, &[(i, h), (j, h)]);
            let fpm = eval(eng, &[(i, h), (j, -h)]);
            let fmp = eval(eng, &[(i, -h), (j, h)]);
            let fmm = eval(eng, &[(i, -h), (j, -h)]);
            let v = (fpp - fpm - fmp + fmm) / (4.0 * h * h);
            hm[i][j] = v;
            hm[j][i] = v;
        }
    }
    if eng.cancelled {
        bail!(CANCELLED);
    }
    // The translations among themselves and the rotations among
    // themselves: a rotation moves the anatomy less than its millimetre-
    // like unit says, so comparing across the two would flag every
    // rotation.
    let block = |ks: &[usize]| -> Vec<(f64, String)> {
        let sub: Vec<Vec<f64>> = ks
            .iter()
            .map(|&i| ks.iter().map(|&j| hm[i][j]).collect())
            .collect();
        let (vals, vecs) = symmetric_eigen(sub);
        let strongest = vals.first().copied().unwrap_or(0.0);
        let mut out = Vec::new();
        if strongest <= 0.0 {
            return out;
        }
        for (val, v) in vals.iter().zip(&vecs).rev() {
            if *val >= WEAK * strongest {
                break;
            }
            let m = (0..ks.len())
                .max_by(|&a, &b| v[a].abs().total_cmp(&v[b].abs()))
                .unwrap_or(0);
            let k = ks[m];
            let name = match (eng.req.dof, k) {
                (Dof2d3d::Six, 0..=2) => format!("rotation about {}", AXES[k]),
                (Dof2d3d::Six, _) => format!("translation {}", AXES[k - 3]),
                (Dof2d3d::Translations, _) => format!("translation {}", AXES[k]),
            };
            out.push((*val / strongest, name));
        }
        out
    };
    let mut weak = match eng.req.dof {
        Dof2d3d::Six => [block(&[3, 4, 5]), block(&[0, 1, 2])].concat(),
        Dof2d3d::Translations => block(&[0, 1, 2]),
    };
    weak.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut names: Vec<String> = Vec::new();
    for (_, n) in weak {
        if !names.contains(&n) {
            names.push(n);
        }
    }
    let warning = (!names.is_empty()).then(|| format!("Weakly constrained: {}", names.join(", ")));
    let (vals, _) = symmetric_eigen(hm);
    Ok((vals, warning))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_of_a_known_matrix() {
        let (vals, vecs) = symmetric_eigen(vec![
            vec![2.0, 1.0, 0.0],
            vec![1.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.5],
        ]);
        assert!(
            (vals[0] - 3.0).abs() < 1e-9
                && (vals[1] - 1.0).abs() < 1e-9
                && (vals[2] - 0.5).abs() < 1e-9
        );
        assert!((vecs[2][2].abs() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn euler_reads_back() {
        let t = RigidTransform::new([0.1, -0.2, 0.3, 0.0, 0.0, 0.0], Vec3::ZERO);
        let e = euler_zyx(&to_m3(t.matrix()));
        assert!(
            (e[0] - 0.1).abs() < 1e-12 && (e[1] + 0.2).abs() < 1e-12 && (e[2] - 0.3).abs() < 1e-12
        );
    }

    #[test]
    fn a_shifted_image_gives_its_shift() {
        let dims = [40, 30];
        let f = |x: f64, y: f64| {
            (-((x - 18.0).powi(2) + (y - 14.0).powi(2)) / 30.0).exp()
                + 0.3 * (-((x - 8.0).powi(2) + (y - 22.0).powi(2)) / 10.0).exp()
        };
        let drr: Vec<f32> = (0..1200)
            .map(|p| f((p % 40) as f64, (p / 40) as f64) as f32)
            .collect();
        // image(u) = drr(u − s), s = (3, −2).
        let image: Vec<f32> = (0..1200)
            .map(|p| f((p % 40) as f64 - 3.0, (p / 40) as f64 + 2.0) as f32)
            .collect();
        let mask = vec![1u8; 1200];
        let (s, c) = best_shift(&image, &drr, dims, &mask, 6);
        assert!(
            (s[0] - 3.0).abs() < 0.2 && (s[1] + 2.0).abs() < 0.2,
            "{s:?} {c}"
        );
    }

    #[test]
    fn keys_read_back() {
        for o in Optimizer2d::ALL {
            assert_eq!(Optimizer2d::from_key(o.key()), Some(o));
        }
        assert_eq!(Optimizer2d::from_key("NM"), Some(Optimizer2d::NelderMead));
        for i in Init2d3d::ALL {
            assert_eq!(Init2d3d::from_key(i.key()), Some(i));
        }
    }
}
