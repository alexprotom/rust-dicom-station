//! Pure-Rust 3-D image registration.
//!
//! Two independent intensity-based engines and one geometric one, all
//! re-implemented natively so the application keeps a single-language,
//! dependency-light build:
//!
//! * [`elastix`] — the [elastix](https://elastix.dev) framework: multi-
//!   resolution Gaussian pyramids, *random coordinate* sampling, a mean-
//!   squared-difference metric, an Euler rigid transform and a cubic
//!   B-spline free-form deformation, all driven by the Adaptive Stochastic
//!   Gradient Descent optimizer of Klein et al. (IJCV 2009) — elastix's
//!   default. Stochastic, fast, tolerant of a poor starting point.
//! * [`plastimatch`] — the [plastimatch](https://plastimatch.org) B-spline
//!   registration of Shackleford et al.: centre-of-gravity alignment, then a
//!   *dense* cost over every eligible fixed voxel with the exact analytic
//!   gradient scattered onto the control lattice, a bending-energy
//!   regularizer, mean-squared error **or** Mattes mutual information, and a
//!   quasi-Newton (L-BFGS) optimizer with a line search. Deterministic,
//!   smoother, and the multi-modal option.
//! * [`landmark`] — plastimatch's `landmark_warp`: a radial-basis
//!   deformation interpolating paired points, with the thin-plate spline,
//!   Gaussian and Wendland kernels.
//!
//! elastix and plastimatch are C++ / ITK toolboxes; nothing of either is
//! linked here. Parameter names mirror their vocabularies
//! (`NumberOfResolutions`, `MaximumNumberOfIterations`,
//! `NumberOfSpatialSamples`, `FinalGridSpacingInPhysicalUnits`;
//! `grid_spacing`, `young_modulus`, `max_its`) so a parameter file from
//! either toolbox reads across.
//!
//! Any of them can be restricted to a [`RegionMask`] — a structure or a
//! segmentation with a margin — which is what "register this tumour, not
//! the whole patient" means; see [`analysis`] for what comes back and
//! [`dvf`] for the vector field.
//!
//! Convention: the recovered transform maps **fixed-image patient
//! coordinates → moving-image patient coordinates** (the resampling
//! convention used by elastix, ITK and plastimatch alike).

pub mod analysis;
pub mod dvf;
pub mod elastix;
pub mod landmark;
pub mod plastimatch;

use std::sync::Arc;

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::progress::Progress;
use crate::volume::Volume;

pub use analysis::{Dof6, JacobianStats, RegAnalysis, VectorStats};
pub use dvf::{FieldStyle, VectorField};
pub use landmark::{LandmarkKernel, LandmarkPair, LandmarkParams, RbfWarp};

// ---------------------------------------------------------------------------
// Methods, metrics and parameters
// ---------------------------------------------------------------------------

/// Which algorithm recovers the transform.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegMethod {
    /// elastix Euler 6-DOF rigid, ASGD over stochastic samples.
    ElastixRigid,
    /// elastix rigid pre-alignment + cubic B-spline FFD, ASGD.
    ElastixBSpline,
    /// plastimatch B-spline: dense analytic gradient, bending-energy
    /// regularization, L-BFGS.
    PlastimatchBSpline,
    /// plastimatch `landmark_warp`: a radial-basis warp through paired
    /// points — no image intensities involved.
    PlastimatchLandmark,
}

impl RegMethod {
    pub const ALL: [RegMethod; 4] = [
        RegMethod::ElastixRigid,
        RegMethod::ElastixBSpline,
        RegMethod::PlastimatchBSpline,
        RegMethod::PlastimatchLandmark,
    ];

    /// Full name, as the result panel writes it.
    pub fn label(self) -> &'static str {
        match self {
            RegMethod::ElastixRigid => "Rigid — Euler 6-DOF (elastix, ASGD)",
            RegMethod::ElastixBSpline => "Deformable — rigid + B-spline FFD (elastix, ASGD)",
            RegMethod::PlastimatchBSpline => "Deformable — B-spline (plastimatch, L-BFGS)",
            RegMethod::PlastimatchLandmark => "Deformable — landmark warp (plastimatch, RBF)",
        }
    }

    /// Name for a button or a menu entry.
    pub fn short(self) -> &'static str {
        match self {
            RegMethod::ElastixRigid => "Rigid (elastix)",
            RegMethod::ElastixBSpline => "B-spline (elastix)",
            RegMethod::PlastimatchBSpline => "B-spline (plastimatch)",
            RegMethod::PlastimatchLandmark => "Landmarks (plastimatch)",
        }
    }

    /// The tooltip that explains when to reach for it.
    pub fn hint(self) -> &'static str {
        match self {
            RegMethod::ElastixRigid => {
                "6-DOF Euler transform about the fixed-image centre. Stochastic \
                 sampling and the ASGD optimizer — seconds, and tolerant of a poor \
                 starting alignment."
            }
            RegMethod::ElastixBSpline => {
                "Rigid pre-alignment, then a cubic B-spline free-form deformation, \
                 both optimized by ASGD on random samples. Fast; the displacement \
                 inside uniform regions is interpolated from the lattice."
            }
            RegMethod::PlastimatchBSpline => {
                "Centre-of-gravity alignment, then a B-spline deformation optimized \
                 over every eligible voxel with the exact analytic gradient and a \
                 bending-energy penalty (L-BFGS). Deterministic and smoother than the \
                 stochastic engine, and the only one with mutual information — so also \
                 the CT–MR option. Slower."
            }
            RegMethod::PlastimatchLandmark => {
                "A deformation interpolating the landmark pairs you place — thin-plate \
                 spline, Gaussian or Wendland kernel. Image intensities are not used at \
                 all, so it works across modalities and where an intensity metric has \
                 nothing to lock onto."
            }
        }
    }

    /// Which toolbox the algorithm comes from.
    pub fn family(self) -> &'static str {
        match self {
            RegMethod::ElastixRigid | RegMethod::ElastixBSpline => "elastix",
            RegMethod::PlastimatchBSpline | RegMethod::PlastimatchLandmark => "plastimatch",
        }
    }

    /// True when the result carries a deformation, not just a rigid body.
    pub fn is_deformable(self) -> bool {
        self != RegMethod::ElastixRigid
    }

    /// True when image intensities drive the result.
    pub fn is_intensity_based(self) -> bool {
        self != RegMethod::PlastimatchLandmark
    }
}

/// What the plastimatch engine minimizes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Metric {
    /// Mean squared intensity difference — mono-modal (CT–CT).
    MeanSquares,
    /// Mattes mutual information — multi-modal (CT–MR, CT–CBCT).
    MutualInformation,
}

impl Metric {
    pub const ALL: [Metric; 2] = [Metric::MeanSquares, Metric::MutualInformation];

    pub fn label(self) -> &'static str {
        match self {
            Metric::MeanSquares => "Mean squares",
            Metric::MutualInformation => "Mutual information",
        }
    }

    /// Short name used in the metric readout ("MSD 9700 ▶ 1800").
    pub fn tag(self) -> &'static str {
        match self {
            Metric::MeanSquares => "MSD",
            Metric::MutualInformation => "−MI",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Metric::MeanSquares => {
                "Mean squared HU difference. Right when the two images measure the same \
                 thing (CT–CT); meaningless when they do not."
            }
            Metric::MutualInformation => {
                "Mattes mutual information over a 32 × 32 joint histogram with cubic \
                 B-spline Parzen windows. Needs no intensity correspondence, so it is \
                 what CT–MR and CT–CBCT need. Slower and less sharply peaked."
            }
        }
    }
}

/// Everything a run needs beyond the two volumes.
#[derive(Clone)]
pub struct RegParams {
    pub method: RegMethod,
    /// elastix: NumberOfResolutions / plastimatch: number of stages.
    pub levels: usize,
    /// elastix: MaximumNumberOfIterations (per resolution level).
    pub iterations: usize,
    /// elastix: NumberOfSpatialSamples (new samples every iteration).
    /// Unused by the plastimatch engine, which is dense.
    pub samples: usize,
    /// elastix: FinalGridSpacingInPhysicalUnits / plastimatch: grid_spacing
    /// (B-spline control lattice, mm).
    pub grid_spacing_mm: f64,
    /// Sample only fixed-image voxels above this value (crude body mask; use
    /// a very low value to disable). Comparable to a fixed-image mask.
    pub fixed_threshold: f32,
    /// plastimatch: `young_modulus`, the weight of the bending-energy
    /// penalty on the control lattice (0 = off).
    pub regularization: f64,
    /// plastimatch: which metric to minimize.
    pub metric: Metric,
    /// plastimatch: keep every `stride`-th eligible voxel (1 = all of them).
    /// The cost is dense either way; this bounds it on large volumes.
    pub stride: usize,
    /// Kernel and stiffness of the landmark warp.
    pub landmark: LandmarkParams,
    /// The paired points the landmark method interpolates.
    pub landmarks: Vec<LandmarkPair>,
    /// Restrict the fixed-image samples and the control lattice to a region —
    /// what makes a registration *local*.
    pub region: Option<Arc<RegionMask>>,
    /// An alignment to start from and refine rather than replace.
    ///
    /// A deformable run with a start recovers a *correction*: the moving
    /// image is sampled through `start` plus the new deformation, and the
    /// result is the two composed. That is what makes a local refinement
    /// behave the way a physicist expects — the structure is realigned while
    /// the rest of the patient keeps the global result, because a lattice
    /// covering only the structure is exactly zero outside it.
    pub start: Option<Arc<Transform3>>,
}

impl Default for RegParams {
    fn default() -> Self {
        RegParams {
            method: RegMethod::ElastixRigid,
            levels: 3,
            iterations: 300,
            samples: 3000,
            grid_spacing_mm: 32.0,
            fixed_threshold: -500.0,
            regularization: 0.02,
            metric: Metric::MeanSquares,
            stride: 2,
            landmark: LandmarkParams::default(),
            landmarks: Vec::new(),
            region: None,
            start: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Region of interest (local registration)
// ---------------------------------------------------------------------------

/// The part of the fixed image a registration is restricted to.
///
/// Local registration — "align this tumour, not the whole patient" — needs
/// two things from a structure: the samples must come from inside it, and
/// the B-spline lattice must cover it rather than the entire volume. Both
/// come from this mask, which is the structure's own voxel mask dilated by a
/// margin, so the deformation is driven by the structure *and the tissue
/// immediately around it* — without that margin nothing constrains the
/// boundary.
pub struct RegionMask {
    /// What the region is, for the result summary.
    pub name: String,
    /// One byte per fixed-volume voxel, 1 inside.
    mask: Vec<u8>,
    dims: [usize; 3],
    spacing: [f64; 3],
    origin: Vec3,
    axes: [Vec3; 3],
    /// Inclusive voxel bounding box of the dilated mask.
    lo: [usize; 3],
    hi: [usize; 3],
    /// Voxels inside.
    count: usize,
}

impl RegionMask {
    /// Build a region from a voxel mask on the fixed volume's own grid,
    /// dilated by `margin_mm`. Returns `None` when the mask is empty or does
    /// not match the volume.
    pub fn from_mask(vol: &Volume, mask: &[u8], name: String, margin_mm: f64) -> Option<Self> {
        let dims = vol.dims;
        if mask.len() != dims[0] * dims[1] * dims[2] || mask.is_empty() {
            return None;
        }
        let mut m = mask.to_vec();
        // Separable box dilation: one pass per axis, radius from that axis'
        // own spacing so a millimetre margin is a millimetre on every axis.
        for axis in 0..3 {
            let r = (margin_mm / vol.spacing[axis]).round() as usize;
            if r > 0 {
                dilate_axis(&mut m, dims, axis, r);
            }
        }
        let [nx, ny, _] = dims;
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        let mut count = 0usize;
        for (o, &v) in m.iter().enumerate() {
            if v == 0 {
                continue;
            }
            count += 1;
            let k = o / (nx * ny);
            let rem = o - k * nx * ny;
            for (a, c) in [rem % nx, rem / nx, k].into_iter().enumerate() {
                lo[a] = lo[a].min(c);
                hi[a] = hi[a].max(c);
            }
        }
        if count == 0 {
            return None;
        }
        Some(RegionMask {
            name,
            mask: m,
            dims,
            spacing: vol.spacing,
            origin: vol.origin,
            axes: [vol.row_dir, vol.col_dir, vol.normal],
            lo,
            hi,
            count,
        })
    }

    /// Voxels inside the dilated region.
    pub fn voxels(&self) -> usize {
        self.count
    }

    /// Volume of the region in cm³.
    pub fn cm3(&self) -> f64 {
        self.count as f64 * self.spacing[0] * self.spacing[1] * self.spacing[2] / 1000.0
    }

    /// Is this patient-space point inside (nearest-neighbour lookup)?
    #[inline]
    pub fn contains(&self, p: Vec3) -> bool {
        let d = p - self.origin;
        let mut idx = [0usize; 3];
        for (a, slot) in idx.iter_mut().enumerate() {
            let u = (d.dot(self.axes[a]) / self.spacing[a]).round();
            if u < 0.0 || u >= self.dims[a] as f64 {
                return false;
            }
            *slot = u as usize;
        }
        self.mask[idx[2] * self.dims[0] * self.dims[1] + idx[1] * self.dims[0] + idx[0]] != 0
    }

    /// Inclusive voxel bounding box of the dilated region.
    pub fn bbox(&self) -> ([usize; 3], [usize; 3]) {
        (self.lo, self.hi)
    }
}

/// In-place box dilation of a 0/1 mask along one axis.
fn dilate_axis(mask: &mut [u8], dims: [usize; 3], axis: usize, radius: usize) {
    let [nx, ny, nz] = dims;
    let (n, stride) = match axis {
        0 => (nx, 1usize),
        1 => (ny, nx),
        _ => (nz, nx * ny),
    };
    if n == 0 {
        return;
    }
    // Start index of every line along `axis`.
    let lines: Vec<usize> = match axis {
        0 => (0..ny * nz).map(|l| l * nx).collect(),
        1 => (0..nx * nz)
            .map(|l| (l / nx) * nx * ny + (l % nx))
            .collect(),
        _ => (0..nx * ny).collect(),
    };
    let src = mask.to_vec();
    // Each line is an independent 1-D dilation over a running count of set
    // voxels in the window, so the whole pass is O(voxels) whatever radius.
    let done: Vec<(usize, Vec<u8>)> = lines
        .par_iter()
        .map(|&base| {
            let mut line = vec![0u8; n];
            let mut acc = 0usize;
            for u in 0..=radius.min(n - 1) {
                acc += (src[base + u * stride] != 0) as usize;
            }
            for (t, slot) in line.iter_mut().enumerate() {
                if t > 0 {
                    if t + radius < n {
                        acc += (src[base + (t + radius) * stride] != 0) as usize;
                    }
                    if t > radius {
                        acc -= (src[base + (t - radius - 1) * stride] != 0) as usize;
                    }
                }
                *slot = (acc > 0) as u8;
            }
            (base, line)
        })
        .collect();
    for (base, line) in done {
        for (t, v) in line.into_iter().enumerate() {
            mask[base + t * stride] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

/// 3×3 rotation matrix (row-major).
#[derive(Clone, Copy)]
pub(crate) struct Mat3(pub(crate) [f64; 9]);

impl Mat3 {
    pub(crate) fn mul_vec(&self, v: Vec3) -> Vec3 {
        let m = &self.0;
        Vec3::new(
            m[0] * v.x + m[1] * v.y + m[2] * v.z,
            m[3] * v.x + m[4] * v.y + m[5] * v.z,
            m[6] * v.x + m[7] * v.y + m[8] * v.z,
        )
    }
    fn mul(&self, o: &Mat3) -> Mat3 {
        let a = &self.0;
        let b = &o.0;
        let mut r = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
            }
        }
        Mat3(r)
    }
    fn transpose(&self) -> Mat3 {
        let m = &self.0;
        Mat3([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]])
    }
}

fn rot_x(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    Mat3([1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c])
}
fn rot_y(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    Mat3([c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c])
}
fn rot_z(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    Mat3([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
}
fn drot_x(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    Mat3([0.0, 0.0, 0.0, 0.0, -s, -c, 0.0, c, -s])
}
fn drot_y(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    Mat3([-s, 0.0, c, 0.0, 0.0, 0.0, -c, 0.0, -s])
}
fn drot_z(a: f64) -> Mat3 {
    let (s, c) = a.sin_cos();
    Mat3([-s, -c, 0.0, c, -s, 0.0, 0.0, 0.0, 0.0])
}

/// Euler rigid transform (elastix EulerTransform):
/// `T(x) = R_z R_y R_x (x - c) + c + t`, parameters `[rx, ry, rz, tx, ty, tz]`.
///
/// The rotation matrix, its transpose and the three rotation derivatives are
/// computed once in [`RigidTransform::new`] and cached. `map`, `unmap` and
/// `jacobian` run on millions of points per registration and per fusion
/// frame, so none of them may touch trigonometry. The parameters are private
/// precisely so the cache cannot go stale — build a new transform to change
/// them.
#[derive(Clone)]
pub struct RigidTransform {
    params: [f64; 6],
    center: Vec3,
    /// `R_z R_y R_x`.
    rot: Mat3,
    /// `rot` transposed = the exact inverse rotation.
    rot_t: Mat3,
    /// `∂R/∂rx`, `∂R/∂ry`, `∂R/∂rz`.
    drot: [Mat3; 3],
    /// Translation part `[tx, ty, tz]`.
    t: Vec3,
}

impl RigidTransform {
    pub fn new(params: [f64; 6], center: Vec3) -> Self {
        let (rx, ry, rz) = (params[0], params[1], params[2]);
        let rot = rot_z(rz).mul(&rot_y(ry)).mul(&rot_x(rx));
        RigidTransform {
            params,
            center,
            rot,
            rot_t: rot.transpose(),
            drot: [
                rot_z(rz).mul(&rot_y(ry)).mul(&drot_x(rx)),
                rot_z(rz).mul(&drot_y(ry)).mul(&rot_x(rx)),
                drot_z(rz).mul(&rot_y(ry)).mul(&rot_x(rx)),
            ],
            t: Vec3::new(params[3], params[4], params[5]),
        }
    }

    pub fn identity(center: Vec3) -> Self {
        Self::new([0.0; 6], center)
    }

    /// `[rx, ry, rz, tx, ty, tz]` (radians / mm).
    pub fn params(&self) -> [f64; 6] {
        self.params
    }

    /// The point rotations are taken about.
    pub fn center(&self) -> Vec3 {
        self.center
    }

    /// The rotation matrix, row-major.
    pub fn matrix(&self) -> [f64; 9] {
        self.rot.0
    }

    #[inline]
    pub fn map(&self, p: Vec3) -> Vec3 {
        self.rot.mul_vec(p - self.center) + self.center + self.t
    }

    /// Exact inverse (rotation transposed).
    #[inline]
    pub fn unmap(&self, q: Vec3) -> Vec3 {
        self.rot_t.mul_vec(q - self.center - self.t) + self.center
    }

    /// The same mapping, expressed about a different centre of rotation.
    ///
    /// `R(p − c₁) + c₁ + t₁ ≡ R(p − c₂) + c₂ + t₂` with
    /// `t₂ = R(c₂ − c₁) + c₁ + t₁ − c₂`, so a transform recovered globally
    /// can seed a local run about the structure's own centre without moving
    /// a single voxel.
    pub fn recentered(&self, center: Vec3) -> RigidTransform {
        let t2 = self.rot.mul_vec(center - self.center) + self.center + self.t - center;
        RigidTransform::new(
            [
                self.params[0],
                self.params[1],
                self.params[2],
                t2.x,
                t2.y,
                t2.z,
            ],
            center,
        )
    }

    /// ∂T/∂param_i evaluated at fixed point `p` (3-vectors per parameter).
    #[inline]
    fn jacobian(&self, p: Vec3) -> [Vec3; 6] {
        let v = p - self.center;
        [
            self.drot[0].mul_vec(v),
            self.drot[1].mul_vec(v),
            self.drot[2].mul_vec(v),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }
}

/// Cubic B-spline free-form deformation on a regular control-point grid
/// aligned with the fixed image axes. Displacements are patient-space
/// vectors; the grid covers the fixed image domain (or the region) plus a
/// one-cell margin.
#[derive(Clone)]
pub struct BSplineTransform {
    /// Control-point displacement coefficients, `[3 * (ix + nx*(iy + ny*iz))]`.
    pub coeffs: Vec<f64>,
    pub grid_dims: [usize; 3],
    pub grid_origin: Vec3,
    pub spacing: f64,
    /// Fixed-image axes the grid is aligned with.
    pub axes: [Vec3; 3],
}

#[inline]
fn bspline_weights(f: f64) -> [f64; 4] {
    let f2 = f * f;
    let f3 = f2 * f;
    [
        (1.0 - f).powi(3) / 6.0,
        (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0,
        (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0,
        f3 / 6.0,
    ]
}

impl BSplineTransform {
    /// Create an identity (zero displacement) grid covering the fixed volume.
    pub fn new(fixed: &Volume, spacing_mm: f64) -> Self {
        let extent = [
            fixed.dims[0] as f64 * fixed.spacing[0],
            fixed.dims[1] as f64 * fixed.spacing[1],
            fixed.dims[2] as f64 * fixed.spacing[2],
        ];
        let mut grid_dims = [0usize; 3];
        for a in 0..3 {
            // +3 support points for the cubic kernel, +2 margin cells.
            grid_dims[a] = (extent[a] / spacing_mm).ceil() as usize + 4;
        }
        let axes = [fixed.row_dir, fixed.col_dir, fixed.normal];
        // One cell + half a voxel before the first voxel center.
        let grid_origin = fixed.origin
            - axes[0] * (1.5 * spacing_mm)
            - axes[1] * (1.5 * spacing_mm)
            - axes[2] * (1.5 * spacing_mm);
        BSplineTransform {
            coeffs: vec![0.0; 3 * grid_dims[0] * grid_dims[1] * grid_dims[2]],
            grid_dims,
            grid_origin,
            spacing: spacing_mm,
            axes,
        }
    }

    /// The lattice a run should use: the whole volume, or just the region's
    /// bounding box when the registration is local. A local lattice is what
    /// makes a small structure affordable at a fine spacing — it covers the
    /// structure, not the patient.
    pub fn for_region(fixed: &Volume, region: Option<&RegionMask>, spacing_mm: f64) -> Self {
        let Some(r) = region else {
            return Self::new(fixed, spacing_mm);
        };
        let (lo, hi) = r.bbox();
        let axes = [fixed.row_dir, fixed.col_dir, fixed.normal];
        let corner = fixed.voxel_to_patient(lo[0] as f64, lo[1] as f64, lo[2] as f64);
        let mut grid_dims = [0usize; 3];
        for a in 0..3 {
            let extent = (hi[a] - lo[a]) as f64 * fixed.spacing[a];
            grid_dims[a] = (extent / spacing_mm).ceil() as usize + 4;
        }
        let grid_origin = corner
            - axes[0] * (1.5 * spacing_mm)
            - axes[1] * (1.5 * spacing_mm)
            - axes[2] * (1.5 * spacing_mm);
        BSplineTransform {
            coeffs: vec![0.0; 3 * grid_dims[0] * grid_dims[1] * grid_dims[2]],
            grid_dims,
            grid_origin,
            spacing: spacing_mm,
            axes,
        }
    }

    /// Control points on the lattice.
    pub fn control_points(&self) -> usize {
        self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
    }

    /// A copy carrying only the grid geometry, with no coefficients.
    fn geometry(&self) -> BSplineTransform {
        BSplineTransform {
            coeffs: Vec::new(),
            ..self.clone()
        }
    }

    /// Grid support of a point: base indices, per-axis weights, and validity.
    #[inline]
    fn support(&self, p: Vec3) -> Option<([i64; 3], [[f64; 4]; 3])> {
        let d = p - self.grid_origin;
        let mut base = [0i64; 3];
        let mut w = [[0.0; 4]; 3];
        for a in 0..3 {
            let u = d.dot(self.axes[a]) / self.spacing;
            let iu = u.floor();
            base[a] = iu as i64 - 1;
            if base[a] < 0 || base[a] + 3 >= self.grid_dims[a] as i64 {
                return None;
            }
            w[a] = bspline_weights(u - iu);
        }
        Some((base, w))
    }

    /// Displacement at a fixed-image point.
    pub fn displacement(&self, p: Vec3) -> Vec3 {
        let Some((base, w)) = self.support(p) else {
            return Vec3::ZERO;
        };
        let [nx, ny, _] = self.grid_dims;
        let mut disp = Vec3::ZERO;
        for kz in 0..4 {
            let iz = (base[2] + kz as i64) as usize;
            for ky in 0..4 {
                let iy = (base[1] + ky as i64) as usize;
                let wyz = w[1][ky] * w[2][kz];
                let row = 3 * (base[0] as usize + nx * (iy + ny * iz));
                for (kx, wx) in w[0].iter().enumerate() {
                    let wt = wx * wyz;
                    let o = row + 3 * kx;
                    disp.x += wt * self.coeffs[o];
                    disp.y += wt * self.coeffs[o + 1];
                    disp.z += wt * self.coeffs[o + 2];
                }
            }
        }
        disp
    }
}

/// The deformable part of a recovered mapping.
#[derive(Clone)]
pub enum Warp {
    /// Rigid body only.
    None,
    /// Cubic B-spline free-form deformation on a regular lattice.
    BSpline(BSplineTransform),
    /// Radial-basis warp through paired landmarks.
    Rbf(RbfWarp),
    /// A displacement field on a regular lattice, trilinearly interpolated
    /// — what a DICOM Deformable Spatial Registration carries, and what a
    /// result read back from one becomes.
    Field(Arc<VectorField>),
    /// Several warps added together — what a refinement produces: the
    /// deformation that was already there, plus the correction just
    /// recovered on top of it.
    Composite(Vec<Warp>),
}

impl Warp {
    /// Displacement at a fixed-image point (zero when there is no warp).
    #[inline]
    pub fn displacement(&self, p: Vec3) -> Vec3 {
        match self {
            Warp::None => Vec3::ZERO,
            Warp::BSpline(b) => b.displacement(p),
            Warp::Rbf(r) => r.displacement(p),
            Warp::Field(f) => f.sample_patient(p),
            Warp::Composite(parts) => parts.iter().fold(Vec3::ZERO, |a, w| a + w.displacement(p)),
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            Warp::None => true,
            Warp::Composite(parts) => parts.iter().all(Warp::is_none),
            _ => false,
        }
    }

    /// `a` then `b`, flattened so a chain of refinements stays one list.
    pub fn combined(a: Warp, b: Warp) -> Warp {
        let mut parts = Vec::new();
        for w in [a, b] {
            match w {
                Warp::None => {}
                Warp::Composite(inner) => parts.extend(inner),
                other => parts.push(other),
            }
        }
        match parts.len() {
            0 => Warp::None,
            1 => parts.pop().unwrap(),
            _ => Warp::Composite(parts),
        }
    }

    /// One line describing the deformation model, for the result panel.
    pub fn describe(&self) -> String {
        match self {
            Warp::None => "rigid body only".to_string(),
            Warp::BSpline(b) => format!(
                "B-spline lattice {}×{}×{} at {:.0} mm ({} control points)",
                b.grid_dims[0],
                b.grid_dims[1],
                b.grid_dims[2],
                b.spacing,
                b.control_points()
            ),
            Warp::Rbf(r) => r.describe(),
            Warp::Field(f) => format!("displacement field — {}", f.describe()),
            Warp::Composite(parts) => parts
                .iter()
                .map(Warp::describe)
                .collect::<Vec<_>>()
                .join(" + "),
        }
    }
}

/// The full recovered mapping: fixed patient point → moving patient point.
/// Deformable results compose as `T(p) = T_rigid(p) + d_warp(p)`
/// (displacement parameterized on the fixed domain, elastix "compose" style).
#[derive(Clone)]
pub struct Transform3 {
    pub rigid: RigidTransform,
    pub warp: Warp,
}

impl Transform3 {
    /// A rigid-body-only mapping.
    pub fn rigid_only(rigid: RigidTransform) -> Self {
        Transform3 {
            rigid,
            warp: Warp::None,
        }
    }

    #[inline]
    pub fn map(&self, p: Vec3) -> Vec3 {
        let q = self.rigid.map(p);
        match &self.warp {
            Warp::None => q,
            w => q + w.displacement(p),
        }
    }

    /// Displacement `T(p) − p` at a fixed-image point: what the vector field
    /// draws and what the analytics measure.
    #[inline]
    pub fn displacement(&self, p: Vec3) -> Vec3 {
        self.map(p) - p
    }

    /// Inverse mapping (moving → fixed). Exact for rigid; fixed-point
    /// iteration for the deformable part (adequate for the smooth, moderate
    /// deformations these models produce).
    pub fn unmap(&self, q: Vec3) -> Vec3 {
        match &self.warp {
            Warp::None => self.rigid.unmap(q),
            _ => {
                let mut x = self.rigid.unmap(q);
                for _ in 0..12 {
                    let err = q - self.map(x);
                    if err.length() < 1e-3 {
                        break;
                    }
                    // Newton-like step through the (near-rigid) linear part.
                    let corr = self.rigid.unmap(self.rigid.map(x) + err) - x;
                    x = x + corr;
                }
                x
            }
        }
    }
}

/// Registration output with quality statistics.
pub struct RegistrationResult {
    pub transform: Arc<Transform3>,
    pub method: RegMethod,
    /// Which metric the numbers below are in.
    pub metric: Metric,
    pub initial_metric: f64,
    pub final_metric: f64,
    pub iterations_run: usize,
    pub elapsed_secs: f64,
    /// Name of the region a *local* registration was restricted to.
    pub region: Option<String>,
    /// Displacement, rotation and Jacobian statistics of the result.
    pub analysis: RegAnalysis,
}

impl RegistrationResult {
    /// `MSD 9700 ▶ 1800  (900 iters, 20.1 s)`.
    pub fn metric_line(&self) -> String {
        format!(
            "{} {:.1} ▶ {:.1}  ({} iters, {:.1} s)",
            self.metric.tag(),
            self.initial_metric,
            self.final_metric,
            self.iterations_run,
            self.elapsed_secs
        )
    }
}

// ---------------------------------------------------------------------------
// Registration image (f32 + geometry) and Gaussian pyramid
// ---------------------------------------------------------------------------

pub struct RegImage {
    data: Vec<f32>,
    dims: [usize; 3],
    spacing: [f64; 3],
    origin: Vec3,
    axes: [Vec3; 3],
    /// Flat indices of voxels eligible for sampling (value above the
    /// fixed-image threshold and, when the run is local, inside the region),
    /// built once per pyramid level by [`RegImage::prepare_sampling`].
    /// Random sampling draws from this list instead of rejecting random
    /// draws, which keeps every draw a hit; the dense engine walks it.
    eligible: Vec<u32>,
}

impl RegImage {
    pub fn from_volume(v: &Volume) -> Self {
        RegImage {
            data: v.data.iter().map(|&x| x as f32).collect(),
            dims: v.dims,
            spacing: v.spacing,
            origin: v.origin,
            axes: [v.row_dir, v.col_dir, v.normal],
            eligible: Vec::new(),
        }
    }

    /// Build the eligible-voxel list. Voxels on the far boundary are
    /// excluded so a jittered sample always has a full interpolation
    /// neighbourhood.
    fn prepare_sampling(&mut self, threshold: f32, region: Option<&RegionMask>) {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 {
            self.eligible.clear();
            return;
        }
        self.eligible = (0..nz - 1)
            .into_par_iter()
            .flat_map_iter(|k| {
                let plane = k * nx * ny;
                (0..ny - 1).flat_map(move |j| {
                    let row = plane + j * nx;
                    (0..nx - 1).map(move |i| (row + i) as u32)
                })
            })
            .filter(|&o| self.data[o as usize] >= threshold)
            .filter(|&o| match region {
                None => true,
                Some(r) => {
                    let o = o as usize;
                    let k = o / (nx * ny);
                    let rem = o - k * nx * ny;
                    r.contains(self.index_to_patient(
                        (rem % nx) as f64,
                        (rem / nx) as f64,
                        k as f64,
                    ))
                }
            })
            .collect();
    }

    /// Fixed-image points of every eligible voxel, thinned by `stride`, with
    /// the image value there — the sample set the dense (plastimatch) engine
    /// works on. Deterministic: the same volume always yields the same set.
    fn dense_samples(&self, stride: usize) -> Vec<(Vec3, f32)> {
        let stride = stride.max(1);
        let [nx, ny, _] = self.dims;
        self.eligible
            .par_iter()
            .step_by(stride)
            .map(|&o| {
                let o = o as usize;
                let k = o / (nx * ny);
                let rem = o - k * nx * ny;
                (
                    self.index_to_patient((rem % nx) as f64, (rem / nx) as f64, k as f64),
                    self.data[o],
                )
            })
            .collect()
    }

    #[inline]
    fn at(&self, i: usize, j: usize, k: usize) -> f32 {
        self.data[k * self.dims[0] * self.dims[1] + j * self.dims[0] + i]
    }

    /// The largest voxel dimension of this level, mm.
    fn max_spacing(&self) -> f64 {
        self.spacing.iter().cloned().fold(0.0f64, f64::max)
    }

    /// [1 2 1]/4 smoothing + factor-2 decimation along axes with ≥ 8 voxels.
    fn downsample(&self) -> RegImage {
        let f = |n: usize| if n >= 8 { n / 2 } else { n };
        let (nx, ny, nz) = (self.dims[0], self.dims[1], self.dims[2]);
        let (mx, my, mz) = (f(nx), f(ny), f(nz));
        let (dx, dy, dz) = (nx / mx.max(1), ny / my.max(1), nz / mz.max(1));
        let mut out = vec![0.0f32; mx * my * mz];
        let smooth = |c: i64, n: usize, half: bool| -> [usize; 3] {
            let c = if half { c * 2 } else { c };
            let cl = (c - 1).clamp(0, n as i64 - 1) as usize;
            let cc = c.clamp(0, n as i64 - 1) as usize;
            let cr = (c + 1).clamp(0, n as i64 - 1) as usize;
            [cl, cc, cr]
        };
        out.par_chunks_mut(mx * my)
            .enumerate()
            .for_each(|(k, plane)| {
                let ks = smooth(k as i64, nz, dz == 2);
                for j in 0..my {
                    let js = smooth(j as i64, ny, dy == 2);
                    for i in 0..mx {
                        let is = smooth(i as i64, nx, dx == 2);
                        let mut acc = 0.0f32;
                        let wz = [0.25f32, 0.5, 0.25];
                        let wy = [0.25f32, 0.5, 0.25];
                        let wx = [0.25f32, 0.5, 0.25];
                        for (kz, &kk) in ks.iter().enumerate() {
                            for (jy, &jj) in js.iter().enumerate() {
                                for (ix, &ii) in is.iter().enumerate() {
                                    acc += wz[kz] * wy[jy] * wx[ix] * self.at(ii, jj, kk);
                                }
                            }
                        }
                        plane[j * mx + i] = acc;
                    }
                }
            });
        RegImage {
            data: out,
            dims: [mx, my, mz],
            spacing: [
                self.spacing[0] * dx as f64,
                self.spacing[1] * dy as f64,
                self.spacing[2] * dz as f64,
            ],
            origin: self.origin
                + self.axes[0] * (0.5 * self.spacing[0] * (dx as f64 - 1.0))
                + self.axes[1] * (0.5 * self.spacing[1] * (dy as f64 - 1.0))
                + self.axes[2] * (0.5 * self.spacing[2] * (dz as f64 - 1.0)),
            axes: self.axes,
            eligible: Vec::new(),
        }
    }

    fn patient_to_index(&self, p: Vec3) -> [f64; 3] {
        let d = p - self.origin;
        [
            d.dot(self.axes[0]) / self.spacing[0],
            d.dot(self.axes[1]) / self.spacing[1],
            d.dot(self.axes[2]) / self.spacing[2],
        ]
    }

    fn index_to_patient(&self, i: f64, j: f64, k: f64) -> Vec3 {
        self.origin
            + self.axes[0] * (i * self.spacing[0])
            + self.axes[1] * (j * self.spacing[1])
            + self.axes[2] * (k * self.spacing[2])
    }

    /// Trilinear sample at fractional voxel indices that the caller has
    /// already constrained to `[0, dim - 1)`. No gradient, no bounds check —
    /// this is the sampler used when drawing fixed-image samples, which is
    /// the hottest loop of the whole registration.
    #[inline]
    fn value_at_index(&self, u: f64, v: f64, w: f64) -> f32 {
        let [nx, ny, _] = self.dims;
        let i0 = u as usize;
        let j0 = v as usize;
        let k0 = w as usize;
        let fu = (u - i0 as f64) as f32;
        let fv = (v - j0 as f64) as f32;
        let fw = (w - k0 as f64) as f32;
        let o = k0 * nx * ny + j0 * nx + i0;
        let d = &self.data;
        let c00 = d[o] + (d[o + 1] - d[o]) * fu;
        let c10 = d[o + nx] + (d[o + nx + 1] - d[o + nx]) * fu;
        let p = o + nx * ny;
        let c01 = d[p] + (d[p + 1] - d[p]) * fu;
        let c11 = d[p + nx] + (d[p + nx + 1] - d[p + nx]) * fu;
        let c0 = c00 + (c10 - c00) * fv;
        let c1 = c01 + (c11 - c01) * fv;
        c0 + (c1 - c0) * fw
    }

    /// Trilinear sample + analytic gradient of the interpolant, in patient
    /// coordinates. `None` outside the volume.
    fn sample_grad(&self, p: Vec3) -> Option<(f32, Vec3)> {
        let [u, v, w] = self.patient_to_index(p);
        let [nx, ny, nz] = self.dims;
        if u < 0.0 || v < 0.0 || w < 0.0 {
            return None;
        }
        let i0 = u.floor() as usize;
        let j0 = v.floor() as usize;
        let k0 = w.floor() as usize;
        if i0 + 1 >= nx || j0 + 1 >= ny || k0 + 1 >= nz {
            return None;
        }
        let fu = (u - i0 as f64) as f32;
        let fv = (v - j0 as f64) as f32;
        let fw = (w - k0 as f64) as f32;

        let c000 = self.at(i0, j0, k0);
        let c100 = self.at(i0 + 1, j0, k0);
        let c010 = self.at(i0, j0 + 1, k0);
        let c110 = self.at(i0 + 1, j0 + 1, k0);
        let c001 = self.at(i0, j0, k0 + 1);
        let c101 = self.at(i0 + 1, j0, k0 + 1);
        let c011 = self.at(i0, j0 + 1, k0 + 1);
        let c111 = self.at(i0 + 1, j0 + 1, k0 + 1);

        let c00 = c000 + (c100 - c000) * fu;
        let c10 = c010 + (c110 - c010) * fu;
        let c01 = c001 + (c101 - c001) * fu;
        let c11 = c011 + (c111 - c011) * fu;
        let c0 = c00 + (c10 - c00) * fv;
        let c1 = c01 + (c11 - c01) * fv;
        let val = c0 + (c1 - c0) * fw;

        // Analytic derivatives of the trilinear interpolant (index space).
        let du = ((c100 - c000) * (1.0 - fv) + (c110 - c010) * fv) * (1.0 - fw)
            + ((c101 - c001) * (1.0 - fv) + (c111 - c011) * fv) * fw;
        let dv = ((c010 - c000) * (1.0 - fu) + (c110 - c100) * fu) * (1.0 - fw)
            + ((c011 - c001) * (1.0 - fu) + (c111 - c101) * fu) * fw;
        let dw = (c001 - c000) * (1.0 - fu) * (1.0 - fv)
            + (c101 - c100) * fu * (1.0 - fv)
            + (c011 - c010) * (1.0 - fu) * fv
            + (c111 - c110) * fu * fv;

        let grad = self.axes[0] * (du as f64 / self.spacing[0])
            + self.axes[1] * (dv as f64 / self.spacing[1])
            + self.axes[2] * (dw as f64 / self.spacing[2]);
        Some((val, grad))
    }

    /// Intensity range over the eligible voxels — what the mutual-
    /// information histogram is binned over.
    fn eligible_range(&self) -> (f32, f32) {
        if self.eligible.is_empty() {
            return (0.0, 1.0);
        }
        self.eligible
            .par_iter()
            .map(|&o| {
                let v = self.data[o as usize];
                (v, v)
            })
            .reduce(|| (f32::MAX, f32::MIN), |a, b| (a.0.min(b.0), a.1.max(b.1)))
    }
}

// ---------------------------------------------------------------------------
// Random coordinate sampler (xorshift; deterministic per level for
// reproducibility, fresh samples every iteration — elastix RandomCoordinate
// with NewSamplesEveryIteration=true)
// ---------------------------------------------------------------------------

pub(crate) struct XorShift(u64);

impl XorShift {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Deterministic per-sample stream derived from a base seed, so the parallel
/// draw below produces the same sample set as a serial one would.
#[inline]
fn stream(seed: u64, i: usize) -> XorShift {
    let mut x = seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    XorShift(x | 1)
}

/// One iteration's sample set: fixed-space points with fixed image values.
///
/// Draws uniformly from the pre-built eligible-voxel list (see
/// [`RegImage::prepare_sampling`]) and jitters within the cell, which keeps
/// elastix's *RandomCoordinate* continuous sampling while removing the
/// rejection loop entirely — every draw is a hit, so the work is exactly `n`
/// interpolations and can run in parallel.
fn draw_samples(fixed: &RegImage, n: usize, rng: &mut XorShift) -> Vec<(Vec3, f32)> {
    let m = fixed.eligible.len();
    if m == 0 || n == 0 {
        return Vec::new();
    }
    let [nx, ny, _] = fixed.dims;
    let seed = rng.next_u64();
    (0..n)
        .into_par_iter()
        .map(|s| {
            let mut r = stream(seed, s);
            let o = fixed.eligible[(r.next_u64() % m as u64) as usize] as usize;
            let k = o / (nx * ny);
            let rem = o - k * nx * ny;
            let (j, i) = (rem / nx, rem % nx);
            // Sub-voxel jitter inside the cell (indices stay in range because
            // `prepare_sampling` excluded the far boundary).
            let u = i as f64 + r.next_f64();
            let v = j as f64 + r.next_f64();
            let w = k as f64 + r.next_f64();
            (
                fixed.index_to_patient(u, v, w),
                fixed.value_at_index(u, v, w),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Metric evaluation (mean squared difference, elastix AdvancedMeanSquares)
// ---------------------------------------------------------------------------

/// MSD metric only (no gradient) — used for reporting.
fn msd_value(fixed: &RegImage, moving: &RegImage, t: &Transform3, n: usize) -> f64 {
    let mut rng = XorShift(0xD1B54A32D192ED03);
    let samples = draw_samples(fixed, n, &mut rng);
    let (sum, cnt) = samples
        .par_iter()
        .map(|&(p, f)| match moving.sample_grad(t.map(p)) {
            Some((m, _)) => ((m - f) as f64 * (m - f) as f64, 1usize),
            None => (0.0, 0),
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    if cnt == 0 {
        f64::MAX
    } else {
        sum / cnt as f64
    }
}

// ---------------------------------------------------------------------------
// Top-level: build the pyramids and dispatch to an engine
// ---------------------------------------------------------------------------

fn build_pyramid(img: RegImage, levels: usize) -> Vec<RegImage> {
    // pyramid[0] = coarsest … pyramid[levels-1] = full resolution.
    let mut pyr = vec![img];
    for _ in 1..levels {
        let next = pyr.last().unwrap().downsample();
        pyr.push(next);
    }
    pyr.reverse();
    pyr
}

/// Everything an engine gets: the two pyramids, the rotation centre, the
/// fixed volume and the run's parameters.
pub(crate) struct RegSetup<'a> {
    pub fixed: Vec<RegImage>,
    pub moving: Vec<RegImage>,
    /// Fixed-image (or region) centre — the point rotations are taken about
    /// and the anchor of the parameter scaling.
    pub center: Vec3,
    pub fixed_vol: &'a Volume,
    pub params: &'a RegParams,
}

impl RegSetup<'_> {
    /// Index of the full-resolution level.
    fn finest(&self) -> usize {
        self.fixed.len() - 1
    }
}

/// What an engine hands back before the analytics are computed.
pub(crate) struct EngineOutput {
    pub transform: Transform3,
    pub iterations: usize,
    /// The engine's own final cost, in whatever units it minimizes.
    pub final_metric: f64,
}

/// Register `moving` onto `fixed`. Returns the transform mapping fixed
/// patient coordinates to moving patient coordinates.
pub fn register(
    fixed_vol: &Volume,
    moving_vol: &Volume,
    params: &RegParams,
    progress: &Progress,
) -> Result<RegistrationResult> {
    let t_start = std::time::Instant::now();

    // The landmark warp never looks at a voxel, so it skips the pyramids
    // entirely — building them for a geometric interpolation would be
    // several seconds of pure waste on a 512³ study.
    if params.method == RegMethod::PlastimatchLandmark {
        progress.set("Solving the landmark system…");
        let out = landmark::run(params)?;
        let transform = Arc::new(out.transform);
        progress.set("Measuring the deformation…");
        let analysis = analysis::analyse(fixed_vol, &transform, params.region.as_deref());
        progress.set("done");
        return Ok(RegistrationResult {
            transform,
            method: params.method,
            metric: params.metric,
            initial_metric: out.final_metric,
            final_metric: out.final_metric,
            iterations_run: out.iterations,
            elapsed_secs: t_start.elapsed().as_secs_f64(),
            region: params.region.as_ref().map(|r| r.name.clone()),
            analysis,
        });
    }

    progress.set("Building image pyramids…");
    let fixed_full = RegImage::from_volume(fixed_vol);
    let moving_full = RegImage::from_volume(moving_vol);

    // Centre of rotation: the fixed image's geometric centre, or the
    // region's when the run is local — rotating a tumour about the patient's
    // centre would put the whole recovered angle into the translation.
    let center = match params.region.as_deref() {
        None => fixed_full.index_to_patient(
            (fixed_full.dims[0] as f64 - 1.0) * 0.5,
            (fixed_full.dims[1] as f64 - 1.0) * 0.5,
            (fixed_full.dims[2] as f64 - 1.0) * 0.5,
        ),
        Some(r) => {
            let (lo, hi) = r.bbox();
            fixed_vol.voxel_to_patient(
                (lo[0] + hi[0]) as f64 * 0.5,
                (lo[1] + hi[1]) as f64 * 0.5,
                (lo[2] + hi[2]) as f64 * 0.5,
            )
        }
    };

    let mut fixed_pyr = build_pyramid(fixed_full, params.levels);
    let moving_pyr = build_pyramid(moving_full, params.levels);

    // Eligible-voxel lists (the fixed-image mask, intersected with the region
    // when the run is local) are built once per level instead of being
    // re-derived by rejection on every iteration.
    for img in &mut fixed_pyr {
        img.prepare_sampling(params.fixed_threshold, params.region.as_deref());
    }
    if fixed_pyr.iter().all(|i| i.eligible.is_empty()) {
        match &params.region {
            Some(r) => bail!(
                "no fixed-image voxels inside '{}' above the sampling threshold — \
                 lower the threshold or widen the margin",
                r.name
            ),
            None => {
                bail!("no fixed-image voxels above the sampling threshold — lower it and retry")
            }
        }
    }

    let setup = RegSetup {
        fixed: fixed_pyr,
        moving: moving_pyr,
        center,
        fixed_vol,
        params,
    };

    let identity = Transform3::rigid_only(RigidTransform::identity(center));
    let finest = setup.finest();
    let initial_msd = msd_value(
        &setup.fixed[finest],
        &setup.moving[finest],
        &identity,
        params.samples.max(2000),
    );

    let out = match params.method {
        RegMethod::ElastixRigid | RegMethod::ElastixBSpline => elastix::run(&setup, progress)?,
        RegMethod::PlastimatchBSpline => plastimatch::run(&setup, progress)?,
        RegMethod::PlastimatchLandmark => unreachable!("handled above"),
    };

    // Whatever the engine minimized, the reported before/after pair is in
    // the same units — otherwise the two numbers cannot be compared.
    let (initial_metric, final_metric) = match params.metric {
        Metric::MeanSquares => (
            initial_msd,
            msd_value(
                &setup.fixed[finest],
                &setup.moving[finest],
                &out.transform,
                params.samples.max(2000),
            ),
        ),
        Metric::MutualInformation => (
            plastimatch::mi_value(&setup, &identity),
            plastimatch::mi_value(&setup, &out.transform),
        ),
    };

    let transform = Arc::new(out.transform);
    progress.set("Measuring the deformation…");
    let analysis = analysis::analyse(fixed_vol, &transform, params.region.as_deref());

    progress.set("done");
    Ok(RegistrationResult {
        transform,
        method: params.method,
        metric: params.metric,
        initial_metric,
        final_metric,
        iterations_run: out.iterations,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
        region: params.region.as_ref().map(|r| r.name.clone()),
        analysis,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube(dims: [usize; 3]) -> Volume {
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [2.0, 2.0, 2.0],
            origin: Vec3::ZERO,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 1,
        }
    }

    #[test]
    fn a_region_dilates_by_the_requested_margin() {
        let dims = [20, 20, 20];
        let vol = cube(dims);
        let mut mask = vec![0u8; dims[0] * dims[1] * dims[2]];
        mask[10 * dims[0] * dims[1] + 10 * dims[0] + 10] = 1;
        let bare = RegionMask::from_mask(&vol, &mask, "seed".into(), 0.0).unwrap();
        assert_eq!(bare.voxels(), 1);
        assert_eq!(bare.bbox(), ([10, 10, 10], [10, 10, 10]));
        assert!(bare.contains(vol.voxel_to_patient(10.0, 10.0, 10.0)));
        assert!(!bare.contains(vol.voxel_to_patient(11.0, 10.0, 10.0)));

        // 4 mm at 2 mm voxels = two voxels each way: a 5×5×5 block.
        let grown = RegionMask::from_mask(&vol, &mask, "grown".into(), 4.0).unwrap();
        assert_eq!(grown.voxels(), 125);
        assert_eq!(grown.bbox(), ([8, 8, 8], [12, 12, 12]));
        assert!(grown.contains(vol.voxel_to_patient(12.0, 10.0, 10.0)));
        assert!(!grown.contains(vol.voxel_to_patient(13.0, 10.0, 10.0)));
        assert!((grown.cm3() - 125.0 * 8.0 / 1000.0).abs() < 1e-9);
    }

    #[test]
    fn an_empty_or_mismatched_mask_is_not_a_region() {
        let vol = cube([8, 8, 8]);
        assert!(RegionMask::from_mask(&vol, &vec![0u8; 8 * 8 * 8], "empty".into(), 5.0).is_none());
        assert!(RegionMask::from_mask(&vol, &[1u8; 4], "wrong size".into(), 0.0).is_none());
    }

    #[test]
    fn a_local_control_lattice_covers_the_region_not_the_volume() {
        let dims = [64, 64, 64];
        let vol = cube(dims);
        let mut mask = vec![0u8; dims[0] * dims[1] * dims[2]];
        for k in 30..34 {
            for j in 30..34 {
                for i in 30..34 {
                    mask[k * dims[0] * dims[1] + j * dims[0] + i] = 1;
                }
            }
        }
        let region = RegionMask::from_mask(&vol, &mask, "roi".into(), 0.0).unwrap();
        let global = BSplineTransform::for_region(&vol, None, 8.0);
        let local = BSplineTransform::for_region(&vol, Some(&region), 8.0);
        assert!(
            local.control_points() < global.control_points() / 8,
            "{} vs {}",
            local.control_points(),
            global.control_points()
        );
        // The lattice must still support every point of the region.
        for (i, j, k) in [(30, 30, 30), (33, 33, 33), (31, 32, 33)] {
            let p = vol.voxel_to_patient(i as f64, j as f64, k as f64);
            assert!(local.support(p).is_some(), "{i},{j},{k} unsupported");
        }
    }

    #[test]
    fn methods_are_labelled_and_grouped_by_toolbox() {
        assert_eq!(RegMethod::ALL.len(), 4);
        for m in RegMethod::ALL {
            assert!(!m.label().is_empty() && !m.short().is_empty() && !m.hint().is_empty());
            assert!(matches!(m.family(), "elastix" | "plastimatch"));
        }
        assert!(!RegMethod::ElastixRigid.is_deformable());
        assert!(RegMethod::PlastimatchBSpline.is_deformable());
        assert!(!RegMethod::PlastimatchLandmark.is_intensity_based());
        assert_eq!(Metric::MeanSquares.tag(), "MSD");
    }

    #[test]
    fn a_rigid_transform_round_trips_and_reports_its_displacement() {
        let c = Vec3::new(10.0, 20.0, 30.0);
        let t = Transform3::rigid_only(RigidTransform::new([0.1, -0.05, 0.2, 3.0, -2.0, 1.0], c));
        let p = Vec3::new(15.0, 25.0, 35.0);
        let q = t.map(p);
        assert!((t.unmap(q) - p).length() < 1e-9);
        assert!((t.displacement(p) - (q - p)).length() < 1e-12);
        assert!(t.warp.is_none());
        assert_eq!(t.warp.describe(), "rigid body only");
    }
}
