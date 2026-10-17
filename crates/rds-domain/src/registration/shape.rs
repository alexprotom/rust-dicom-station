//! Registration by structures: two images aligned on the surfaces of
//! structures contoured on both, with the voxel values left out of it.
//!
//! Every other engine of [`crate::registration`] looks at the images. That
//! is what a registration usually wants, and exactly what it does not want
//! when the contours are trusted more than the grey values: a contrast CT
//! against a plain one, a CT against an MR or a CBCT, two scans whose
//! anatomy agrees but whose intensities never will, or an alignment that
//! must follow one set of organs and nothing else. Here the two images are
//! reduced to their structures, paired by name (the heart on one with the
//! heart on the other), and the transform is the one that lays each
//! structure's surface onto its partner's.
//!
//! **Distance maps.** Each structure becomes a signed distance map on its
//! own image's lattice: millimetres to the surface, negative inside,
//! clamped at [`DISTANCE_REACH_MM`] so a point far from the surface has no
//! gradient to follow and cannot drag the fit. The map is computed in a box
//! around the structure (its bounding box grown by the reach), which is all
//! that is ever sampled and a hundredth of a whole CT. On the lattice, the
//! surface is where the map's linear interpolation crosses zero: halfway
//! between a voxel inside and its neighbour outside.
//!
//! **Surface points.** The points that are laid onto the other side's map
//! are exactly those zero crossings: the midpoint of every face between a
//! voxel inside the structure and a neighbour outside it. A face on the
//! border of the lattice is not surface (it is a structure cut by the field
//! of view, not anatomy). Each side keeps at most
//! [`POINTS_PER_STRUCTURE`] of them, drawn by a fixed seed, and each
//! structure's sum is divided by its own count, so a large organ does not
//! outvote a small one; the per-structure weight is then what the user
//! says it should be.
//!
//! **The rigid stage** is classical distance-transform (chamfer) matching,
//! made symmetric:
//!
//! ```text
//! E(θ) = Σ_k w_k [ 1/n_k Σ_i ρ(D_k^moving(T_θ x_i)) + 1/m_k Σ_j ρ(D_k^fixed(T_θ⁻¹ y_j)) ]
//! ```
//!
//! with `x_i` the fixed surface points of structure `k`, `y_j` its moving
//! surface points and `T_θ` the Euler transform of
//! [`RigidTransform`] about the centre of the fixed surfaces. The second
//! term is exact for a rigid body (`unmap`) and is what keeps a structure
//! contoured over a shorter length on one side from sliding along its
//! partner. `ρ` is the square, or Huber's function of a given width (the
//! *robust* option), so a slice contoured differently on one side does not
//! steer the whole fit. The six parameters (three with *translation only*)
//! are found by Gauss-Newton with Levenberg-Marquardt damping on the exact
//! derivative of the trilinear maps: deterministic, a few dozen iterations,
//! well under a second. Every sum over points goes through
//! [`crate::par::ordered_fold_by`], so a run reproduces itself on any
//! thread count. The search starts from the centroids of the structures
//! matched (or the identity, or two given points: [`Init`]).
//!
//! **The deformable stage** (optional) is the existing B-spline engines run
//! on the distance maps instead of the images, one structure at a time from
//! the largest to the smallest: the region is the structure plus a margin,
//! the start is the transform so far, so each refinement is a local
//! correction composed onto it ([`super::Warp::combined`]) and the rest of
//! the patient keeps the rigid result. This is what the anchored
//! propagation's contour mode ([`crate::workflow::anchored`]) has done since
//! 2026-09, and the quantised maps it registers are the ones
//! [`distance_volume`] makes here.
//!
//! **What comes back** is an ordinary [`RegistrationResult`] (fixed patient
//! coordinates to moving ones, like every engine) whose metric is the RMS
//! signed surface distance in millimetres before and after, plus a
//! [`ShapeReport`] with one line per structure: mean surface distance and
//! Dice before and after, and what each refinement did.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use super::{
    analysis, Init, Mat3, Metric, RegMethod, RegParams, RegionMask, RegistrationResult,
    RigidTransform, Transform3,
};
use crate::geometry::Vec3;
use crate::morphology;
use crate::progress::{self, Progress, ProgressSink};
use crate::volume::{Grid, Volume};

/// How far a distance map reaches, mm: beyond this the value is clamped,
/// so a sample far from the surface has no gradient to follow and cannot
/// drag the fit.
pub const DISTANCE_REACH_MM: f64 = 40.0;
/// Units per millimetre of the quantised maps [`distance_volume`] makes
/// (stored as `i16`, so 0.01 mm).
pub const DISTANCE_SCALE: f64 = 100.0;
/// Surface points kept per structure and side.
pub const POINTS_PER_STRUCTURE: usize = 4000;
/// Iterations of the rigid stage, per round.
const MAX_ITERATIONS: usize = 100;
/// The rigid stage stops when a step moves less than this, mm ...
const STEP_MM: f64 = 1e-3;
/// ... and turns less than this, rad.
const STEP_RAD: f64 = 1e-5;
/// Points per piece of the ordered sums: a few thousand points still
/// spread over every core.
const PIECE: usize = 256;
/// How much farther than the margin the moving side of a refinement is
/// cut out, mm: room for what the rigid stage left unaligned.
const REFINE_SLACK_MM: f64 = 20.0;

// ---------------------------------------------------------------------------
// The request and what comes back
// ---------------------------------------------------------------------------

/// How many degrees of freedom the rigid stage recovers.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ShapeDof {
    /// Three rotations and three translations.
    #[default]
    Rigid,
    /// The translations alone: a couch shift between two scans of one
    /// patient.
    Translation,
}

impl ShapeDof {
    pub const ALL: [ShapeDof; 2] = [ShapeDof::Rigid, ShapeDof::Translation];

    pub fn label(self) -> &'static str {
        match self {
            ShapeDof::Rigid => "rigid 6-DOF",
            ShapeDof::Translation => "translation only",
        }
    }

    /// Which of `[rx, ry, rz, tx, ty, tz]` the search moves.
    fn active(self) -> &'static [usize] {
        match self {
            ShapeDof::Rigid => &[0, 1, 2, 3, 4, 5],
            ShapeDof::Translation => &[3, 4, 5],
        }
    }
}

/// One structure as it is on the two images.
#[derive(Clone)]
pub struct ShapePair {
    /// What the report calls it: the name, or `fixed / moving` when the two
    /// sides name it differently.
    pub name: String,
    pub color: [u8; 3],
    /// How much it counts against the others (1 is the default).
    pub weight: f64,
    /// On the fixed volume's lattice, one byte per voxel, 1 inside.
    pub fixed: Vec<u8>,
    /// On the moving volume's lattice.
    pub moving: Vec<u8>,
}

/// Everything a registration by structures needs besides the two volumes.
#[derive(Clone)]
pub struct ShapeRequest {
    pub pairs: Vec<ShapePair>,
    pub dof: ShapeDof,
    /// Also lay the moving surfaces onto the fixed maps (see the module
    /// doc). Off, only the fixed surfaces are laid onto the moving maps.
    pub symmetric: bool,
    /// Huber width, mm; `None` is plain least squares.
    pub robust_mm: Option<f64>,
    /// Where the rigid search starts. `Auto` and `CenterOfGravity` both
    /// match the centroids of the structures (the contour analogue of the
    /// centres of gravity); `Points` matches two given points.
    pub init: Init,
    /// An alignment to refine instead of a rigid stage: the deformable
    /// stage then starts from it. Needs `refine`.
    pub start: Option<Arc<Transform3>>,
    /// The deformable stage: a B-spline method with its effort; `region`,
    /// `start`, `init`, `metric` and the threshold are set here. `None`
    /// keeps the result rigid.
    pub refine: Option<RegParams>,
    /// Dilation of each structure that bounds its refinement and the
    /// analysis of the result, mm.
    pub margin_mm: f64,
    /// Surface points kept per structure and side.
    pub points_per_structure: usize,
}

impl Default for ShapeRequest {
    fn default() -> Self {
        ShapeRequest {
            pairs: Vec::new(),
            dof: ShapeDof::Rigid,
            symmetric: true,
            robust_mm: None,
            init: Init::Auto,
            start: None,
            refine: None,
            margin_mm: 10.0,
            points_per_structure: POINTS_PER_STRUCTURE,
        }
    }
}

/// How one structure came out.
#[derive(Clone, Debug)]
pub struct ShapeLine {
    pub name: String,
    pub color: [u8; 3],
    pub weight: f64,
    /// Surface points that took part (both sides).
    pub points: usize,
    /// Mean absolute surface distance, mm, where the search started and
    /// where it ended.
    pub mean_before_mm: f64,
    pub mean_after_mm: f64,
    /// RMS of the same distances, mm.
    pub rms_before_mm: f64,
    pub rms_after_mm: f64,
    /// Dice of the moving structure carried onto the fixed lattice against
    /// the fixed one; `None` when either is empty there.
    pub dice_before: Option<f64>,
    pub dice_after: Option<f64>,
    /// The refinement on this structure: `MSD a ▶ b (n iters, t s)`.
    pub refine_line: Option<String>,
    /// Of that refinement: 95th-percentile displacement inside its region,
    /// mm, and the fraction of it that folds (Jacobian <= 0).
    pub displacement_p95_mm: Option<f64>,
    pub folded_fraction: Option<f64>,
}

impl ShapeLine {
    /// `heart: 4.2 ▶ 0.6 mm, Dice 0.71 ▶ 0.94`.
    pub fn line(&self) -> String {
        let dice = match (self.dice_before, self.dice_after) {
            (Some(a), Some(b)) => format!(", Dice {a:.2} ▶ {b:.2}"),
            (None, Some(b)) => format!(", Dice {b:.2}"),
            _ => String::new(),
        };
        format!(
            "{}: {:.2} ▶ {:.2} mm{dice}",
            self.name, self.mean_before_mm, self.mean_after_mm
        )
    }
}

/// What a registration by structures says besides the transform.
#[derive(Clone, Debug, Default)]
pub struct ShapeReport {
    /// One per pair, in the order they were given.
    pub lines: Vec<ShapeLine>,
    pub dof: ShapeDof,
    pub symmetric: bool,
    pub robust_mm: Option<f64>,
    /// Iterations of the rigid stage (both rounds); 0 when it was skipped.
    pub rigid_iterations: usize,
    /// `[rx, ry, rz (rad), tx, ty, tz (mm)]` of the rigid stage about
    /// `rigid_center`, when it ran.
    pub rigid: Option<[f64; 6]>,
    pub rigid_center: Option<Vec3>,
    /// The deformable method, when the stage ran.
    pub refine: Option<RegMethod>,
}

/// A registration by structures: the result as every engine hands it back,
/// and the report.
pub struct ShapeOutcome {
    pub result: RegistrationResult,
    pub report: ShapeReport,
    /// The union of the fixed structures grown by the margin: where the
    /// analysis looked, and where a vector field of the result is worth
    /// drawing.
    pub region: Option<Arc<RegionMask>>,
}

// ---------------------------------------------------------------------------
// Distance maps
// ---------------------------------------------------------------------------

/// `√outside² − √inside²`: millimetres to the surface, negative inside.
#[inline]
fn signed(outside2: f32, inside2: f32) -> f64 {
    (outside2.max(0.0) as f64).sqrt() - (inside2.max(0.0) as f64).sqrt()
}

/// The signed distance map of a mask on its own lattice, mm, negative
/// inside, clamped at `±reach_mm`.
pub fn signed_distance(
    mask: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    reach_mm: f64,
) -> Vec<f32> {
    let outside = morphology::dist2_to_foreground(mask, dims, spacing);
    let inverted: Vec<u8> = mask.par_iter().map(|&v| (v == 0) as u8).collect();
    let inside = morphology::dist2_to_foreground(&inverted, dims, spacing);
    outside
        .par_iter()
        .zip(inside.par_iter())
        .map(|(&o, &i)| signed(o, i).clamp(-reach_mm, reach_mm) as f32)
        .collect()
}

/// The signed distance map of a mask on the volume's lattice, as a volume
/// the intensity engines can register: millimetres to the surface,
/// negative inside, clamped at [`DISTANCE_REACH_MM`] and scaled by
/// [`DISTANCE_SCALE`]. What the anchored propagation's contour mode and the
/// deformable stage here register in place of the images.
pub fn distance_volume(on: &Volume, mask: &[u8]) -> Volume {
    distance_volume_on(&on.grid(), mask)
}

/// [`distance_volume`] on a lattice given as a [`Grid`] (a box cut out of
/// an image, which has no voxels of its own to carry).
pub fn distance_volume_on(grid: &Grid, mask: &[u8]) -> Volume {
    let dims = grid.dims;
    let outside = morphology::dist2_to_foreground(mask, dims, grid.spacing);
    let inverted: Vec<u8> = mask.iter().map(|&v| (v == 0) as u8).collect();
    let inside = morphology::dist2_to_foreground(&inverted, dims, grid.spacing);
    let data: Vec<i16> = outside
        .iter()
        .zip(&inside)
        .map(|(&o, &i)| {
            let d = signed(o, i);
            (d.clamp(-DISTANCE_REACH_MM, DISTANCE_REACH_MM) * DISTANCE_SCALE).round() as i16
        })
        .collect();
    let reach = (DISTANCE_REACH_MM * DISTANCE_SCALE) as i16;
    Volume {
        data,
        dims,
        spacing: grid.spacing,
        origin: grid.origin,
        row_dir: grid.row_dir,
        col_dir: grid.col_dir,
        normal: grid.normal,
        frame_of_reference_uid: grid.frame_of_reference_uid.clone(),
        min_value: -reach,
        max_value: reach,
    }
}

/// A box of a lattice, inclusive voxel bounds.
#[derive(Clone, Copy, Debug)]
struct Crop {
    lo: [usize; 3],
    hi: [usize; 3],
}

impl Crop {
    /// The mask's bounding box grown by `pad_mm`, clipped to the lattice;
    /// `None` when the mask is empty.
    fn around(mask: &[u8], grid: &Grid, pad_mm: f64) -> Option<Crop> {
        let (lo, hi) = morphology::mask_bbox(mask, grid.dims)?;
        let mut c = Crop { lo, hi };
        for a in 0..3 {
            let r = (pad_mm.max(0.0) / grid.spacing[a]).ceil() as usize;
            c.lo[a] = lo[a].saturating_sub(r);
            c.hi[a] = (hi[a] + r).min(grid.dims[a] - 1);
        }
        Some(c)
    }

    fn dims(&self) -> [usize; 3] {
        [
            self.hi[0] - self.lo[0] + 1,
            self.hi[1] - self.lo[1] + 1,
            self.hi[2] - self.lo[2] + 1,
        ]
    }

    /// The box as a lattice of its own.
    fn grid(&self, g: &Grid) -> Grid {
        Grid {
            dims: self.dims(),
            origin: g.voxel_to_patient(self.lo[0] as f64, self.lo[1] as f64, self.lo[2] as f64),
            ..g.clone()
        }
    }

    /// The part of a mask of the full lattice inside the box.
    fn cut(&self, mask: &[u8], full: [usize; 3]) -> Vec<u8> {
        let [cx, cy, cz] = self.dims();
        let mut out = Vec::with_capacity(cx * cy * cz);
        for k in 0..cz {
            for j in 0..cy {
                let at = (self.lo[2] + k) * full[0] * full[1] + (self.lo[1] + j) * full[0];
                out.extend_from_slice(&mask[at + self.lo[0]..at + self.lo[0] + cx]);
            }
        }
        out
    }
}

/// A signed distance map, in `f32` millimetres, on a box around one
/// structure, sampled trilinearly with its derivative.
struct DistMap {
    d: Vec<f32>,
    dims: [usize; 3],
    origin: Vec3,
    axes: [Vec3; 3],
    spacing: [f64; 3],
}

impl DistMap {
    /// The map of `mask` (on `grid`), in its bounding box grown by the
    /// reach and two voxels: everything beyond is at the clamp anyway.
    fn of(mask: &[u8], grid: &Grid, reach_mm: f64) -> Option<DistMap> {
        let pad = reach_mm + 2.0 * grid.spacing.iter().cloned().fold(0.0, f64::max);
        let crop = Crop::around(mask, grid, pad)?;
        let g = crop.grid(grid);
        let m = crop.cut(mask, grid.dims);
        Some(DistMap {
            d: signed_distance(&m, g.dims, g.spacing, reach_mm),
            dims: g.dims,
            origin: g.origin,
            axes: [g.row_dir, g.col_dir, g.normal],
            spacing: g.spacing,
        })
    }

    /// Value (mm) and gradient (per mm, patient axes) at `p`.
    ///
    /// Outside the box the value of its border is carried on and the
    /// gradient across that border is zero. The box reaches past the clamp
    /// on every side the lattice allows, so this only extrapolates where the
    /// image itself ends: a structure cut by the field of view keeps the
    /// distances its last slice has.
    #[inline]
    fn sample(&self, p: Vec3) -> (f64, Vec3) {
        let d = p - self.origin;
        let mut base = [0usize; 3];
        let mut f = [0.0f64; 3];
        let mut live = [false; 3];
        let mut step = [0usize; 3];
        let strides = [1, self.dims[0], self.dims[0] * self.dims[1]];
        for a in 0..3 {
            let n = self.dims[a];
            if n < 2 {
                continue;
            }
            step[a] = strides[a];
            let u = d.dot(self.axes[a]) / self.spacing[a];
            let top = (n - 1) as f64;
            if u <= 0.0 {
                base[a] = 0;
                f[a] = 0.0;
            } else if u >= top {
                base[a] = n - 2;
                f[a] = 1.0;
            } else {
                let b = (u.floor() as usize).min(n - 2);
                base[a] = b;
                f[a] = u - b as f64;
                live[a] = true;
            }
        }
        let at = base[0] + strides[1] * base[1] + strides[2] * base[2];
        let c = |dx: usize, dy: usize, dz: usize| -> f64 {
            self.d[at + dx * step[0] + dy * step[1] + dz * step[2]] as f64
        };
        let (c000, c100, c010, c110) = (c(0, 0, 0), c(1, 0, 0), c(0, 1, 0), c(1, 1, 0));
        let (c001, c101, c011, c111) = (c(0, 0, 1), c(1, 0, 1), c(0, 1, 1), c(1, 1, 1));
        let [fx, fy, fz] = f;
        let c00 = c000 + (c100 - c000) * fx;
        let c10 = c010 + (c110 - c010) * fx;
        let c01 = c001 + (c101 - c001) * fx;
        let c11 = c011 + (c111 - c011) * fx;
        let c0 = c00 + (c10 - c00) * fy;
        let c1 = c01 + (c11 - c01) * fy;
        let value = c0 + (c1 - c0) * fz;
        let du = [
            ((c100 - c000) * (1.0 - fy) + (c110 - c010) * fy) * (1.0 - fz)
                + ((c101 - c001) * (1.0 - fy) + (c111 - c011) * fy) * fz,
            (c10 - c00) * (1.0 - fz) + (c11 - c01) * fz,
            c1 - c0,
        ];
        let mut g = Vec3::ZERO;
        for a in 0..3 {
            if live[a] {
                g = g + self.axes[a] * (du[a] / self.spacing[a]);
            }
        }
        (value, g)
    }
}

// ---------------------------------------------------------------------------
// Surface points
// ---------------------------------------------------------------------------

/// The six face neighbours.
const FACES: [[isize; 3]; 6] = [
    [-1, 0, 0],
    [1, 0, 0],
    [0, -1, 0],
    [0, 1, 0],
    [0, 0, -1],
    [0, 0, 1],
];

/// The midpoint of every face between a voxel inside `mask` and a face
/// neighbour outside it, in patient coordinates, in scan order. A face on
/// the border of the lattice is not surface: a structure cut by the field
/// of view ends there, its anatomy does not.
pub fn surface_points(mask: &[u8], grid: &Grid) -> Vec<Vec3> {
    let [nx, ny, nz] = grid.dims;
    let Some((lo, hi)) = morphology::mask_bbox(mask, grid.dims) else {
        return Vec::new();
    };
    let slices: Vec<Vec<Vec3>> = (lo[2]..=hi[2])
        .into_par_iter()
        .map(|k| {
            let mut out = Vec::new();
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    if mask[k * nx * ny + j * nx + i] == 0 {
                        continue;
                    }
                    for f in FACES {
                        let (ii, jj, kk) =
                            (i as isize + f[0], j as isize + f[1], k as isize + f[2]);
                        if ii < 0
                            || jj < 0
                            || kk < 0
                            || ii >= nx as isize
                            || jj >= ny as isize
                            || kk >= nz as isize
                        {
                            continue;
                        }
                        if mask[kk as usize * nx * ny + jj as usize * nx + ii as usize] == 0 {
                            out.push(grid.voxel_to_patient(
                                i as f64 + 0.5 * f[0] as f64,
                                j as f64 + 0.5 * f[1] as f64,
                                k as f64 + 0.5 * f[2] as f64,
                            ));
                        }
                    }
                }
            }
            out
        })
        .collect();
    slices.concat()
}

/// SplitMix64: the fixed-seed stream the points are drawn with.
fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// At most `n` of the points, drawn without replacement by a fixed seed
/// (a partial Fisher-Yates shuffle): the same points on every run.
fn thin(mut pts: Vec<Vec3>, n: usize, seed: u64) -> Vec<Vec3> {
    if pts.len() <= n {
        return pts;
    }
    let mut s = seed;
    for i in 0..n {
        let j = i + (splitmix(&mut s) % (pts.len() - i) as u64) as usize;
        pts.swap(i, j);
    }
    pts.truncate(n);
    pts
}

// ---------------------------------------------------------------------------
// The cost and its derivative
// ---------------------------------------------------------------------------

/// One surface point and how it counts.
#[derive(Clone, Copy)]
struct Sample {
    p: Vec3,
    pair: usize,
    /// A moving surface point, laid onto the fixed map through `T⁻¹`.
    back: bool,
    /// The pair's weight over its count on this side.
    w: f64,
}

/// Cost, gradient and Gauss-Newton matrix, summed over points.
#[derive(Clone, Copy, Default)]
struct Acc {
    cost: f64,
    g: [f64; 6],
    h: [[f64; 6]; 6],
}

impl Acc {
    fn add(mut self, o: Acc) -> Acc {
        self.cost += o.cost;
        for a in 0..6 {
            self.g[a] += o.g[a];
            for b in 0..6 {
                self.h[a][b] += o.h[a][b];
            }
        }
        self
    }
}

/// `ρ(r)` and the weight `ψ(r)/r` iteratively reweighted least squares
/// uses: the square, or Huber's function of width `delta`.
#[inline]
fn huber(r: f64, delta: Option<f64>) -> (f64, f64) {
    match delta {
        Some(d) if r.abs() > d => (d * (r.abs() - 0.5 * d), d / r.abs()),
        _ => (0.5 * r * r, 1.0),
    }
}

/// The two lattices' maps, one per pair and side.
struct Maps {
    fixed: Vec<DistMap>,
    moving: Vec<DistMap>,
}

/// What a rigid transform needs per point, computed once per evaluation.
struct RigidJets {
    /// `(∂R/∂r_i)ᵀ`.
    drot_t: [Mat3; 3],
}

impl RigidJets {
    fn of(t: &RigidTransform) -> RigidJets {
        RigidJets {
            drot_t: [
                t.drot[0].transpose(),
                t.drot[1].transpose(),
                t.drot[2].transpose(),
            ],
        }
    }
}

/// The residual of one point under a rigid transform and its derivative
/// with respect to `[rx, ry, rz, tx, ty, tz]`.
#[inline]
fn residual_rigid(
    t: &RigidTransform,
    jets: &RigidJets,
    s: &Sample,
    maps: &Maps,
) -> (f64, [f64; 6]) {
    if !s.back {
        let q = t.map(s.p);
        let (d, g) = maps.moving[s.pair].sample(q);
        let jac = t.jacobian(s.p);
        (d, std::array::from_fn(|j| g.dot(jac[j])))
    } else {
        // T⁻¹(y) = Rᵀ(y − c − t) + c: ∂/∂r_i = (∂R/∂r_i)ᵀ (y − c − t),
        // ∂/∂t_a = −Rᵀ e_a, and g · (−Rᵀ e_a) = −(R g)_a.
        let q = t.unmap(s.p);
        let (d, g) = maps.fixed[s.pair].sample(q);
        let v = s.p - t.center - t.t;
        let rg = t.rot.mul_vec(g);
        (
            d,
            [
                g.dot(jets.drot_t[0].mul_vec(v)),
                g.dot(jets.drot_t[1].mul_vec(v)),
                g.dot(jets.drot_t[2].mul_vec(v)),
                -rg.x,
                -rg.y,
                -rg.z,
            ],
        )
    }
}

/// The cost, its gradient and the Gauss-Newton matrix at `t`, summed in
/// fixed pieces and in order.
fn evaluate(t: &RigidTransform, samples: &[Sample], maps: &Maps, robust: Option<f64>) -> Acc {
    let jets = RigidJets::of(t);
    crate::par::ordered_fold_by(
        samples,
        PIECE,
        |part, _| {
            let mut acc = Acc::default();
            for s in part {
                let (r, j) = residual_rigid(t, &jets, s, maps);
                let (rho, wt) = huber(r, robust);
                acc.cost += s.w * rho;
                let k = s.w * wt;
                for a in 0..6 {
                    acc.g[a] += k * r * j[a];
                    for b in a..6 {
                        acc.h[a][b] += k * j[a] * j[b];
                    }
                }
            }
            acc
        },
        Acc::add,
        Acc::default(),
    )
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting; a
/// vanishing pivot leaves its unknown at zero.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap_or(col);
        a.swap(col, piv);
        b.swap(col, piv);
        let p = a[col][col];
        if p.abs() < 1e-300 {
            continue;
        }
        for r in col + 1..n {
            let f = a[r][col] / p;
            if f == 0.0 {
                continue;
            }
            // Row `r` minus `f` times the pivot row (which sits above it).
            let (above, below) = a.split_at_mut(r);
            for (x, &y) in below[0][col..].iter_mut().zip(&above[col][col..]) {
                *x -= f * y;
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        if a[r][r].abs() < 1e-300 {
            continue;
        }
        let mut s = b[r];
        for c in r + 1..n {
            s -= a[r][c] * x[c];
        }
        x[r] = s / a[r][r];
    }
    x
}

/// What the rigid stage minimises over: the points, the maps they are laid
/// onto, the centre of rotation and how the residuals count.
struct Problem<'a> {
    center: Vec3,
    samples: &'a [Sample],
    maps: &'a Maps,
    robust: Option<f64>,
    dof: ShapeDof,
}

/// Levenberg-Marquardt on the rigid parameters from `theta`, returning the
/// parameters and the accepted iterations.
fn fit_rigid(
    pb: &Problem,
    mut theta: [f64; 6],
    p: &Progress,
    label: &str,
) -> Result<([f64; 6], usize)> {
    let (center, samples, maps, robust) = (pb.center, pb.samples, pb.maps, pb.robust);
    let active = pb.dof.active();
    let n = active.len();
    let mut lambda = 1e-3;
    let mut iters = 0;
    let mut cur = evaluate(&RigidTransform::new(theta, center), samples, maps, robust);
    for _ in 0..MAX_ITERATIONS {
        if p.cancelled() {
            bail!("registration {}", progress::CANCELLED);
        }
        // The matrix is accumulated as its upper triangle.
        let h = |i: usize, j: usize| {
            if i <= j {
                cur.h[i][j]
            } else {
                cur.h[j][i]
            }
        };
        let scale = active
            .iter()
            .map(|&i| h(i, i))
            .fold(0.0, f64::max)
            .max(1e-12);
        let mut accepted = None;
        for _ in 0..16 {
            let mut a = vec![vec![0.0; n]; n];
            let mut b = vec![0.0; n];
            for (r, &i) in active.iter().enumerate() {
                b[r] = -cur.g[i];
                for (c, &j) in active.iter().enumerate() {
                    a[r][c] = h(i, j);
                }
                a[r][r] += lambda * h(i, i) + 1e-9 * scale;
            }
            let delta = solve(a, b);
            let mut next = theta;
            for (r, &i) in active.iter().enumerate() {
                next[i] += delta[r];
            }
            let trial = evaluate(&RigidTransform::new(next, center), samples, maps, robust);
            if trial.cost <= cur.cost {
                let small = active
                    .iter()
                    .zip(&delta)
                    .all(|(&i, d)| d.abs() < if i < 3 { STEP_RAD } else { STEP_MM });
                theta = next;
                cur = trial;
                lambda = (lambda / 3.0).max(1e-9);
                accepted = Some(small);
                break;
            }
            lambda *= 4.0;
        }
        // No step lowers the cost any more: that is the minimum.
        let Some(small) = accepted else {
            break;
        };
        iters += 1;
        p.report(
            iters as f32 / MAX_ITERATIONS as f32,
            &format!("{label}: iteration {iters}"),
        );
        if small {
            break;
        }
    }
    Ok((theta, iters))
}

/// The residual of every point under any transform (a warp included):
/// what the before / after numbers are made of.
fn residuals(t: &Transform3, samples: &[Sample], maps: &Maps) -> Vec<f64> {
    samples
        .par_iter()
        .map(|s| {
            if s.back {
                maps.fixed[s.pair].sample(t.unmap(s.p)).0
            } else {
                maps.moving[s.pair].sample(t.map(s.p)).0
            }
        })
        .collect()
}

/// Per pair, the mean absolute and the RMS surface distance; overall, the
/// weighted RMS. Summed sequentially in point order.
fn surface_stats(r: &[f64], samples: &[Sample], pairs: usize) -> (Vec<(f64, f64)>, f64) {
    let mut per = vec![(0.0f64, 0.0f64, 0usize); pairs];
    let (mut wsum, mut wsq) = (0.0, 0.0);
    for (s, &d) in samples.iter().zip(r) {
        let e = &mut per[s.pair];
        e.0 += d.abs();
        e.1 += d * d;
        e.2 += 1;
        wsum += s.w;
        wsq += s.w * d * d;
    }
    let per = per
        .into_iter()
        .map(|(a, q, n)| {
            let n = n.max(1) as f64;
            (a / n, (q / n).sqrt())
        })
        .collect();
    (per, (wsq / wsum.max(1e-300)).sqrt())
}

/// Dice of the moving structure carried onto the fixed lattice through `t`
/// against the fixed one: carried by the propagation itself (its mapping
/// lattice keeps a deformable transform affordable on a whole organ), the
/// way the registration module scores structures, so the two agree.
fn dice_through(t: &Transform3, pair: &ShapePair, fixed: &Volume, moving: &Volume) -> Option<f64> {
    let subject = crate::propagate::Subject {
        name: pair.name.clone(),
        color: pair.color,
        mask: pair.moving.clone(),
        surface_cm3: None,
        keep_shape: false,
    };
    // The transform maps fixed coordinates to moving ones: arriving on the
    // fixed lattice uses it as it is.
    let carried = crate::propagate::propagate(
        moving,
        fixed,
        t,
        false,
        std::slice::from_ref(&subject),
        &progress::Quiet,
    )
    .ok()?
    .pop()?
    .mask;
    crate::motion::dice(&pair.fixed, &carried)
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// Register `moving` onto `fixed` by the structures of `req`. The transform
/// maps fixed patient coordinates to moving ones, like every engine's. The
/// volumes' voxel values are used for nothing but the image overlap the
/// analysis reports.
pub fn register(
    fixed: &Volume,
    moving: &Volume,
    req: &ShapeRequest,
    p: &Progress,
) -> Result<ShapeOutcome> {
    let t_start = std::time::Instant::now();
    if req.pairs.is_empty() {
        bail!("pick at least one structure contoured on both images");
    }
    if fixed.is_empty() || moving.is_empty() {
        bail!("both images need voxels");
    }
    if req.start.is_some() && req.refine.is_none() {
        bail!("a refinement needs a deformable stage");
    }
    if let Some(r) = &req.refine {
        if !matches!(
            r.method,
            RegMethod::ElastixBSpline | RegMethod::PlastimatchBSpline
        ) {
            bail!("the deformable stage needs a B-spline method");
        }
    }
    let fgrid = fixed.grid();
    let mgrid = moving.grid();
    let (nf, nm) = (fixed.data.len(), moving.data.len());
    for pair in &req.pairs {
        if pair.fixed.len() != nf || pair.moving.len() != nm {
            bail!("'{}' is not on the lattices of the two images", pair.name);
        }
        if !(pair.weight.is_finite() && pair.weight > 0.0) {
            bail!("the weight of '{}' must be positive", pair.name);
        }
    }
    let n_pairs = req.pairs.len();

    // ---- maps and points ----
    p.set_phase(0.0, 0.15);
    let mut maps = Maps {
        fixed: Vec::with_capacity(n_pairs),
        moving: Vec::with_capacity(n_pairs),
    };
    let mut samples: Vec<Sample> = Vec::new();
    let mut fixed_points: Vec<Vec3> = Vec::new();
    let mut counts = Vec::with_capacity(n_pairs);
    let mut centroids = (Vec3::ZERO, Vec3::ZERO, 0.0f64, 0.0f64);
    let keep = req.points_per_structure.max(16);
    for (k, pair) in req.pairs.iter().enumerate() {
        if p.cancelled() {
            bail!("registration {}", progress::CANCELLED);
        }
        p.report(
            k as f32 / n_pairs as f32,
            &format!("Distance maps of {} ({}/{n_pairs})", pair.name, k + 1),
        );
        let fmap = DistMap::of(&pair.fixed, &fgrid, DISTANCE_REACH_MM)
            .with_context(|| format!("'{}' is empty on the fixed image", pair.name))?;
        let mmap = DistMap::of(&pair.moving, &mgrid, DISTANCE_REACH_MM)
            .with_context(|| format!("'{}' is empty on the moving image", pair.name))?;
        maps.fixed.push(fmap);
        maps.moving.push(mmap);
        let fpts = thin(
            surface_points(&pair.fixed, &fgrid),
            keep,
            0x5EED + 2 * k as u64,
        );
        let mpts = thin(
            surface_points(&pair.moving, &mgrid),
            keep,
            0x5EED + 2 * k as u64 + 1,
        );
        if fpts.is_empty() || mpts.is_empty() {
            bail!(
                "'{}' has no surface inside the image (it fills the field of view)",
                pair.name
            );
        }
        let wf = pair.weight / fpts.len() as f64;
        samples.extend(fpts.iter().map(|&p| Sample {
            p,
            pair: k,
            back: false,
            w: wf,
        }));
        if req.symmetric {
            let wm = pair.weight / mpts.len() as f64;
            samples.extend(mpts.iter().map(|&p| Sample {
                p,
                pair: k,
                back: true,
                w: wm,
            }));
        }
        fixed_points.extend_from_slice(&fpts);
        // The structures' centroids, weighted by their volumes, are where
        // the search starts unless told otherwise.
        let fc = crate::motion::centroid_mm(&pair.fixed, &fgrid);
        let mc = crate::motion::centroid_mm(&pair.moving, &mgrid);
        let fv = morphology::count_set(&pair.fixed) as f64 * fgrid.voxel_cm3();
        let mv = morphology::count_set(&pair.moving) as f64 * mgrid.voxel_cm3();
        if let (Some(fc), Some(mc)) = (fc, mc) {
            centroids.0 = centroids.0 + fc * fv;
            centroids.1 = centroids.1 + mc * mv;
            centroids.2 += fv;
            centroids.3 += mv;
        }
        counts.push(fv);
    }
    // Rotations are taken about the centre of the fixed surfaces: about
    // the patient's centre, a small angle would be paid for in translation.
    let center = {
        let s = fixed_points.iter().fold(Vec3::ZERO, |a, &b| a + b);
        s * (1.0 / fixed_points.len().max(1) as f64)
    };
    let shift = match req.init {
        Init::Identity => Vec3::ZERO,
        Init::Points { fixed, moving } => moving - fixed,
        Init::Auto | Init::CenterOfGravity => {
            if centroids.2 > 0.0 && centroids.3 > 0.0 {
                centroids.1 * (1.0 / centroids.3) - centroids.0 * (1.0 / centroids.2)
            } else {
                Vec3::ZERO
            }
        }
    };
    let initial = match &req.start {
        Some(s) => (**s).clone(),
        None => Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, shift.x, shift.y, shift.z],
            center,
        )),
    };
    let before = residuals(&initial, &samples, &maps);
    let (per_before, rms_before) = surface_stats(&before, &samples, n_pairs);

    // ---- the rigid stage ----
    let mut report = ShapeReport {
        dof: req.dof,
        symmetric: req.symmetric,
        robust_mm: req.robust_mm,
        refine: req.refine.as_ref().map(|r| r.method),
        ..ShapeReport::default()
    };
    let mut transform = initial.clone();
    if req.start.is_none() {
        let span = if req.refine.is_some() { 0.3 } else { 0.75 };
        // A first round on every eighth point finds the basin cheaply; the
        // second polishes on all of them.
        let mut theta = initial.rigid.params();
        let mut iters = 0;
        if samples.len() >= 8 * 500 {
            p.set_phase(0.15, span * 0.4);
            let coarse: Vec<Sample> = samples.iter().step_by(8).copied().collect();
            let pb = Problem {
                center,
                samples: &coarse,
                maps: &maps,
                robust: req.robust_mm,
                dof: req.dof,
            };
            let (t, n) = fit_rigid(&pb, theta, p, "Rigid on the structures, coarse")?;
            theta = t;
            iters += n;
        }
        p.set_phase(0.15 + span * 0.4, span * 0.6);
        let pb = Problem {
            center,
            samples: &samples,
            maps: &maps,
            robust: req.robust_mm,
            dof: req.dof,
        };
        let (t, n) = fit_rigid(&pb, theta, p, "Rigid on the structures")?;
        theta = t;
        iters += n;
        report.rigid_iterations = iters;
        report.rigid = Some(theta);
        report.rigid_center = Some(center);
        transform = Transform3::rigid_only(RigidTransform::new(theta, center));
    }

    // ---- the deformable stage ----
    let mut lines: Vec<ShapeLine> = req
        .pairs
        .iter()
        .enumerate()
        .map(|(k, pair)| ShapeLine {
            name: pair.name.clone(),
            color: pair.color,
            weight: pair.weight,
            points: samples.iter().filter(|s| s.pair == k).count(),
            mean_before_mm: per_before[k].0,
            mean_after_mm: per_before[k].0,
            rms_before_mm: per_before[k].1,
            rms_after_mm: per_before[k].1,
            dice_before: None,
            dice_after: None,
            refine_line: None,
            displacement_p95_mm: None,
            folded_fraction: None,
        })
        .collect();
    let mut refine_iters = 0;
    if let Some(base) = &req.refine {
        // The largest first: where two regions overlap, the smaller
        // structure's correction comes last and keeps its own surface.
        let mut order: Vec<usize> = (0..n_pairs).collect();
        order.sort_by(|&a, &b| counts[b].total_cmp(&counts[a]));
        let base_frac = if req.start.is_none() { 0.45 } else { 0.15 };
        let span = (0.85 - base_frac) / n_pairs as f32;
        let max_sp = |g: &Grid| g.spacing.iter().cloned().fold(0.0, f64::max);
        for (step, &k) in order.iter().enumerate() {
            let pair = &req.pairs[k];
            p.set_phase(base_frac + span * step as f32, span);
            p.set(format!(
                "Refining on {} ({}/{n_pairs})",
                pair.name,
                step + 1
            ));
            // Both sides cut out around the structure: the fixed one as far
            // as its region reaches, the moving one with room for what is
            // still out of place.
            let fcrop = Crop::around(&pair.fixed, &fgrid, req.margin_mm + 2.0 * max_sp(&fgrid))
                .with_context(|| format!("'{}' is empty on the fixed image", pair.name))?;
            let mcrop = Crop::around(&pair.moving, &mgrid, req.margin_mm + REFINE_SLACK_MM)
                .with_context(|| format!("'{}' is empty on the moving image", pair.name))?;
            let fg = fcrop.grid(&fgrid);
            let fm = fcrop.cut(&pair.fixed, fgrid.dims);
            let mg = mcrop.grid(&mgrid);
            let mm = mcrop.cut(&pair.moving, mgrid.dims);
            let fixed_map = distance_volume_on(&fg, &fm);
            let moving_map = distance_volume_on(&mg, &mm);
            let region =
                RegionMask::from_mask(&fixed_map, &fm, pair.name.clone(), req.margin_mm.max(0.0))
                    .with_context(|| format!("'{}' is empty on the fixed image", pair.name))?;
            let mut params = base.clone();
            params.region = Some(Arc::new(region));
            params.start = Some(Arc::new(transform.clone()));
            params.init = Init::Auto;
            // Every voxel of a distance map is informative; the body
            // threshold is for images, and so is mutual information.
            params.fixed_threshold = f32::MIN;
            params.metric = Metric::MeanSquares;
            params.landmarks.clear();
            let r = super::register(&fixed_map, &moving_map, &params, p)
                .with_context(|| format!("refining on '{}'", pair.name))?;
            refine_iters += r.iterations_run;
            let line = &mut lines[k];
            line.refine_line = Some(r.metric_line());
            line.displacement_p95_mm = Some(r.analysis.displacement.p95);
            line.folded_fraction = Some(r.analysis.jacobian.folded);
            transform = (*r.transform).clone();
        }
    }

    // ---- what it did ----
    p.set_phase(0.85, 0.15);
    p.set("Measuring the structures");
    let after = residuals(&transform, &samples, &maps);
    let (per_after, rms_after) = surface_stats(&after, &samples, n_pairs);
    for (k, line) in lines.iter_mut().enumerate() {
        line.mean_after_mm = per_after[k].0;
        line.rms_after_mm = per_after[k].1;
        line.dice_before = dice_through(&initial, &req.pairs[k], fixed, moving);
        line.dice_after = dice_through(&transform, &req.pairs[k], fixed, moving);
    }
    report.lines = lines;

    // The analysis looks where the structures are: their union on the fixed
    // image, grown by the margin.
    let mut union = vec![0u8; nf];
    for pair in &req.pairs {
        union
            .par_iter_mut()
            .zip(pair.fixed.par_iter())
            .for_each(|(u, &v)| *u |= (v != 0) as u8);
    }
    let names = req
        .pairs
        .iter()
        .map(|p| p.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let region =
        RegionMask::from_mask(fixed, &union, names.clone(), req.margin_mm.max(0.0)).map(Arc::new);
    let transform = Arc::new(transform);
    p.set("Measuring the deformation");
    let mut analysis = analysis::analyse(fixed, &transform, region.as_deref());
    analysis.overlap = analysis::overlap(fixed, moving, &transform, region.as_deref());
    p.set("done");
    let method = if req.refine.is_some() {
        RegMethod::ShapeDeformable
    } else {
        RegMethod::ShapeRigid
    };
    Ok(ShapeOutcome {
        result: RegistrationResult {
            transform,
            method,
            metric: Metric::SurfaceDistance,
            initial_metric: rms_before,
            final_metric: rms_after,
            iterations_run: report.rigid_iterations + refine_iters,
            elapsed_secs: t_start.elapsed().as_secs_f64(),
            region: Some(names),
            analysis,
        },
        report,
        region,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lattice of `n³` voxels at `s` mm, the identity orientation.
    fn grid(n: usize, s: f64, origin: Vec3) -> Grid {
        Grid {
            dims: [n, n, n],
            spacing: [s, s, s],
            origin,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        }
    }

    fn mask_of(g: &Grid, inside: impl Fn(Vec3) -> bool) -> Vec<u8> {
        let [nx, ny, nz] = g.dims;
        let mut m = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    m[k * nx * ny + j * nx + i] =
                        inside(g.voxel_to_patient(i as f64, j as f64, k as f64)) as u8;
                }
            }
        }
        m
    }

    #[test]
    fn the_signed_distance_of_a_sphere_is_its_radius_difference() {
        let g = grid(40, 1.0, Vec3::ZERO);
        let c = Vec3::new(19.5, 19.5, 19.5);
        let r = 10.0;
        let m = mask_of(&g, |p| (p - c).length() <= r);
        let d = signed_distance(&m, g.dims, g.spacing, 8.0);
        let [nx, ny, _] = g.dims;
        let mut worst = 0.0f64;
        for k in 0..40 {
            for j in 0..40 {
                for i in 0..40 {
                    let p = g.voxel_to_patient(i as f64, j as f64, k as f64);
                    let truth = ((p - c).length() - r).clamp(-8.0, 8.0);
                    let v = d[k * nx * ny + j * nx + i] as f64;
                    // A voxel-wise map sits within a voxel of the analytic
                    // surface (it measures to voxel centres, not to the
                    // surface between them).
                    worst = worst.max((v - truth).abs());
                    if m[k * nx * ny + j * nx + i] != 0 {
                        assert!(v < 0.0, "negative inside");
                    } else {
                        assert!(v > 0.0, "positive outside");
                    }
                    assert!(v.abs() <= 8.0 + 1e-6, "clamped at the reach");
                }
            }
        }
        assert!(
            worst <= 1.0,
            "within a voxel of the analytic distance: {worst}"
        );
    }

    #[test]
    fn the_cropped_map_samples_what_the_whole_map_holds() {
        let g = grid(48, 1.5, Vec3::new(-30.0, -20.0, 5.0));
        let c = g.voxel_to_patient(20.0, 25.0, 22.0);
        let m = mask_of(&g, |p| {
            let d = p - c;
            (d.x / 9.0).powi(2) + (d.y / 6.0).powi(2) + (d.z / 7.0).powi(2) <= 1.0
        });
        let map = DistMap::of(&m, &g, 10.0).unwrap();
        let whole = signed_distance(&m, g.dims, g.spacing, 10.0);
        let [nx, ny, _] = g.dims;
        for &(i, j, k) in &[
            (20, 25, 22),
            (5, 5, 5),
            (30, 25, 22),
            (20, 40, 10),
            (47, 47, 47),
        ] {
            let (v, _) = map.sample(g.voxel_to_patient(i as f64, j as f64, k as f64));
            let w = whole[k * nx * ny + j * nx + i] as f64;
            assert!((v - w).abs() < 1e-5, "({i},{j},{k}): {v} against {w}");
        }
    }

    #[test]
    fn surface_points_lie_on_the_zero_level_and_skip_the_cut_faces() {
        // A slab five slices thick that runs through the whole lattice in
        // x: its x faces are cut by the field of view, not surface.
        let g = grid(20, 2.0, Vec3::ZERO);
        let m = mask_of(&g, |p| {
            (6.0..=14.0).contains(&p.y) && (10.0..=18.0).contains(&p.z)
        });
        let pts = surface_points(&m, &g);
        assert!(!pts.is_empty());
        for p in &pts {
            assert!(p.x > -0.5 && p.x < 38.5, "no point on the cut faces: {p:?}");
        }
        // Every slice of the slab has points, and the top and bottom faces
        // sit half a voxel beyond the last slices inside.
        for z in [10.0, 12.0, 14.0, 16.0, 18.0] {
            assert!(pts.iter().any(|p| (p.z - z).abs() < 1e-9), "slice z = {z}");
        }
        assert!(pts.iter().any(|p| (p.z - 9.0).abs() < 1e-9));
        assert!(pts.iter().any(|p| (p.z - 19.0).abs() < 1e-9));
        // On the map of the same mask, each point reads zero.
        let map = DistMap::of(&m, &g, 10.0).unwrap();
        for p in pts.iter().step_by(7) {
            let (v, _) = map.sample(*p);
            assert!(v.abs() < 1e-6, "zero on the surface: {v} at {p:?}");
        }
    }

    /// Two different shapes on two lattices, and samples on both sides.
    fn two_sided() -> (Maps, Vec<Sample>) {
        let gf = grid(40, 1.5, Vec3::ZERO);
        let gm = grid(36, 1.7, Vec3::new(3.0, -2.0, 1.0));
        let cf = Vec3::new(30.0, 28.0, 31.0);
        let cm = Vec3::new(32.0, 27.0, 30.0);
        let fm = mask_of(&gf, |p| {
            let d = p - cf;
            (d.x / 12.0).powi(2) + (d.y / 8.0).powi(2) + (d.z / 10.0).powi(2) <= 1.0
        });
        let mm = mask_of(&gm, |p| {
            let d = p - cm;
            (d.x / 11.0).powi(2) + (d.y / 9.0).powi(2) + (d.z / 10.0).powi(2) <= 1.0
        });
        let maps = Maps {
            fixed: vec![DistMap::of(&fm, &gf, 20.0).unwrap()],
            moving: vec![DistMap::of(&mm, &gm, 20.0).unwrap()],
        };
        let mut samples: Vec<Sample> = thin(surface_points(&fm, &gf), 300, 1)
            .into_iter()
            .map(|p| Sample {
                p,
                pair: 0,
                back: false,
                w: 1.0 / 300.0,
            })
            .collect();
        samples.extend(
            thin(surface_points(&mm, &gm), 300, 2)
                .into_iter()
                .map(|p| Sample {
                    p,
                    pair: 0,
                    back: true,
                    w: 1.0 / 300.0,
                }),
        );
        (maps, samples)
    }

    #[test]
    fn the_gradient_of_the_cost_matches_finite_differences() {
        let (maps, samples) = two_sided();
        let center = Vec3::new(30.0, 28.0, 31.0);
        let theta = [0.03, -0.02, 0.05, 1.2, -0.7, 0.4];
        for robust in [None, Some(1.0)] {
            let acc = evaluate(&RigidTransform::new(theta, center), &samples, &maps, robust);
            for a in 0..6 {
                let h = if a < 3 { 1e-6 } else { 1e-5 };
                let mut up = theta;
                let mut dn = theta;
                up[a] += h;
                dn[a] -= h;
                let cu = evaluate(&RigidTransform::new(up, center), &samples, &maps, robust).cost;
                let cd = evaluate(&RigidTransform::new(dn, center), &samples, &maps, robust).cost;
                let fd = (cu - cd) / (2.0 * h);
                let an = acc.g[a];
                // Trilinear maps are piecewise smooth; a point crossing a
                // cell face inside the stencil costs a little agreement.
                assert!(
                    (fd - an).abs() <= 1e-3 * an.abs().max(fd.abs()) + 1e-6,
                    "parameter {a} ({robust:?}): analytic {an}, finite difference {fd}"
                );
            }
        }
    }

    #[test]
    fn huber_is_the_square_inside_and_linear_outside() {
        assert_eq!(huber(0.5, Some(1.0)), (0.125, 1.0));
        let (rho, w) = huber(3.0, Some(1.0));
        assert!((rho - 2.5).abs() < 1e-12);
        assert!((w - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(huber(3.0, None), (4.5, 1.0));
    }

    #[test]
    fn the_quantised_map_is_the_one_the_anchored_run_always_registered() {
        let g = grid(24, 2.0, Vec3::new(-5.0, 4.0, 0.0));
        let c = g.voxel_to_patient(11.0, 12.0, 10.0);
        let m = mask_of(&g, |p| (p - c).length() <= 9.0);
        let v = distance_volume_on(&g, &m);
        // The formula as anchored.rs had it, written out.
        let outside = morphology::dist2_to_foreground(&m, g.dims, g.spacing);
        let inverted: Vec<u8> = m.iter().map(|&v| (v == 0) as u8).collect();
        let inside = morphology::dist2_to_foreground(&inverted, g.dims, g.spacing);
        for (n, (&o, &i)) in outside.iter().zip(&inside).enumerate() {
            let d = (o.max(0.0) as f64).sqrt() - (i.max(0.0) as f64).sqrt();
            let want = (d.clamp(-40.0, 40.0) * 100.0).round() as i16;
            assert_eq!(v.data[n], want);
        }
        assert_eq!((v.min_value, v.max_value), (-4000, 4000));
        assert_eq!(v.origin, g.origin);
    }

    #[test]
    fn a_crop_is_the_same_lattice_shifted() {
        let g = grid(30, 1.25, Vec3::new(10.0, -3.0, 7.0));
        let m = mask_of(&g, |p| {
            (p - g.voxel_to_patient(12.0, 14.0, 9.0)).length() < 4.0
        });
        let c = Crop::around(&m, &g, 3.0).unwrap();
        let cg = c.grid(&g);
        let cut = c.cut(&m, g.dims);
        let [cx, cy, _] = cg.dims;
        assert_eq!(cut.len(), cx * cy * cg.dims[2]);
        for k in 0..cg.dims[2] {
            for j in 0..cy {
                for i in 0..cx {
                    let p = cg.voxel_to_patient(i as f64, j as f64, k as f64);
                    let q = g.voxel_to_patient(
                        (c.lo[0] + i) as f64,
                        (c.lo[1] + j) as f64,
                        (c.lo[2] + k) as f64,
                    );
                    assert!((p - q).length() < 1e-9);
                    let at = (c.lo[2] + k) * 900 + (c.lo[1] + j) * 30 + c.lo[0] + i;
                    assert_eq!(cut[k * cx * cy + j * cx + i], m[at]);
                }
            }
        }
    }
}
