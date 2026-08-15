//! Pure-Rust 3D image registration following the elastix framework
//! (<https://elastix.dev>): multi-resolution Gaussian pyramids, random
//! coordinate sampling, a mean-squared-difference metric, an Euler rigid
//! transform and a cubic B-spline free-form deformation, all driven by the
//! Adaptive Stochastic Gradient Descent (ASGD) optimizer of Klein et al.
//! (IJCV 2009) — elastix's default optimizer.
//!
//! elastix itself is a C++ / ITK toolbox; this module re-implements its core
//! algorithms natively in Rust so the application keeps a single-language,
//! dependency-light build. Parameter names in [`RegParams`] mirror the
//! elastix parameter file vocabulary (NumberOfResolutions,
//! MaximumNumberOfIterations, NumberOfSpatialSamples,
//! FinalGridSpacingInPhysicalUnits).
//!
//! Convention: the recovered transform maps **fixed-image patient
//! coordinates → moving-image patient coordinates** (the resampling
//! convention used by elastix/ITK).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::volume::Volume;

// ---------------------------------------------------------------------------
// Parameters & progress
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RegKind {
    Rigid,
    /// Rigid pre-alignment followed by cubic B-spline FFD.
    Deformable,
}

#[derive(Clone, Copy)]
pub struct RegParams {
    pub kind: RegKind,
    /// elastix: NumberOfResolutions.
    pub levels: usize,
    /// elastix: MaximumNumberOfIterations (per resolution level).
    pub iterations: usize,
    /// elastix: NumberOfSpatialSamples (new samples every iteration).
    pub samples: usize,
    /// elastix: FinalGridSpacingInPhysicalUnits (B-spline control grid, mm).
    pub grid_spacing_mm: f64,
    /// Sample only fixed-image voxels above this value (crude body mask; use
    /// a very low value to disable). Comparable to a fixed-image mask.
    pub fixed_threshold: f32,
}

impl Default for RegParams {
    fn default() -> Self {
        RegParams {
            kind: RegKind::Rigid,
            levels: 3,
            iterations: 300,
            samples: 3000,
            grid_spacing_mm: 32.0,
            fixed_threshold: -500.0,
        }
    }
}

/// Shared progress/cancel handle for the background registration thread.
#[derive(Default)]
pub struct RegProgress {
    msg: Mutex<String>,
    cancel: AtomicBool,
}

impl RegProgress {
    pub fn set(&self, s: impl Into<String>) {
        *self.msg.lock().unwrap() = s.into();
    }
    pub fn get(&self) -> String {
        self.msg.lock().unwrap().clone()
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

/// 3×3 rotation matrix (row-major).
#[derive(Clone, Copy)]
struct Mat3([f64; 9]);

impl Mat3 {
    fn mul_vec(&self, v: Vec3) -> Vec3 {
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
                r[i * 3 + j] =
                    a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
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
#[derive(Clone)]
pub struct RigidTransform {
    pub params: [f64; 6],
    pub center: Vec3,
}

impl RigidTransform {
    pub fn identity(center: Vec3) -> Self {
        RigidTransform { params: [0.0; 6], center }
    }

    fn rotation(&self) -> Mat3 {
        rot_z(self.params[2])
            .mul(&rot_y(self.params[1]))
            .mul(&rot_x(self.params[0]))
    }

    pub fn map(&self, p: Vec3) -> Vec3 {
        let r = self.rotation();
        let t = Vec3::new(self.params[3], self.params[4], self.params[5]);
        r.mul_vec(p - self.center) + self.center + t
    }

    /// Exact inverse (rotation transposed).
    pub fn unmap(&self, q: Vec3) -> Vec3 {
        let r = self.rotation().transpose();
        let t = Vec3::new(self.params[3], self.params[4], self.params[5]);
        r.mul_vec(q - self.center - t) + self.center
    }

    /// ∂T/∂param_i evaluated at fixed point `p` (3-vectors per parameter).
    fn jacobian(&self, p: Vec3) -> [Vec3; 6] {
        let v = p - self.center;
        let (rx, ry, rz) = (self.params[0], self.params[1], self.params[2]);
        let d_rx = rot_z(rz).mul(&rot_y(ry)).mul(&drot_x(rx)).mul_vec(v);
        let d_ry = rot_z(rz).mul(&drot_y(ry)).mul(&rot_x(rx)).mul_vec(v);
        let d_rz = drot_z(rz).mul(&rot_y(ry)).mul(&rot_x(rx)).mul_vec(v);
        [
            d_rx,
            d_ry,
            d_rz,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }
}

/// Cubic B-spline free-form deformation on a regular control-point grid
/// aligned with the fixed image axes. Displacements are patient-space
/// vectors; the grid covers the fixed image domain plus a one-cell margin.
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
                for kx in 0..4 {
                    let wt = w[0][kx] * wyz;
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

/// The full recovered mapping: fixed patient point → moving patient point.
/// Deformable results compose as `T(p) = T_rigid(p) + d_bspline(p)`
/// (displacement parameterized on the fixed domain, elastix "compose" style).
#[derive(Clone)]
pub struct Transform3 {
    pub rigid: RigidTransform,
    pub bspline: Option<BSplineTransform>,
}

impl Transform3 {
    pub fn map(&self, p: Vec3) -> Vec3 {
        let q = self.rigid.map(p);
        match &self.bspline {
            Some(b) => q + b.displacement(p),
            None => q,
        }
    }

    /// Inverse mapping (moving → fixed). Exact for rigid; fixed-point
    /// iteration for the deformable part (adequate for the smooth,
    /// moderate deformations B-spline grids produce).
    pub fn unmap(&self, q: Vec3) -> Vec3 {
        match &self.bspline {
            None => self.rigid.unmap(q),
            Some(_) => {
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
    pub kind: RegKind,
    pub initial_metric: f64,
    pub final_metric: f64,
    pub iterations_run: usize,
    pub elapsed_secs: f64,
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
}

impl RegImage {
    pub fn from_volume(v: &Volume) -> Self {
        RegImage {
            data: v.data.iter().map(|&x| x as f32).collect(),
            dims: v.dims,
            spacing: v.spacing,
            origin: v.origin,
            axes: [v.row_dir, v.col_dir, v.normal],
        }
    }

    #[inline]
    fn at(&self, i: usize, j: usize, k: usize) -> f32 {
        self.data[k * self.dims[0] * self.dims[1] + j * self.dims[0] + i]
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
        out.par_chunks_mut(mx * my).enumerate().for_each(|(k, plane)| {
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
}

// ---------------------------------------------------------------------------
// Random coordinate sampler (xorshift; deterministic per level for
// reproducibility, fresh samples every iteration — elastix RandomCoordinate
// with NewSamplesEveryIteration=true)
// ---------------------------------------------------------------------------

struct XorShift(u64);

impl XorShift {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One iteration's sample set: fixed-space points with fixed image values.
fn draw_samples(
    fixed: &RegImage,
    n: usize,
    threshold: f32,
    rng: &mut XorShift,
) -> Vec<(Vec3, f32)> {
    let mut out = Vec::with_capacity(n);
    let mut attempts = 0usize;
    let max_attempts = n * 20;
    while out.len() < n && attempts < max_attempts {
        attempts += 1;
        let i = rng.next_f64() * (fixed.dims[0] as f64 - 1.0);
        let j = rng.next_f64() * (fixed.dims[1] as f64 - 1.0);
        let k = rng.next_f64() * (fixed.dims[2] as f64 - 1.0);
        let p = fixed.index_to_patient(i, j, k);
        if let Some((v, _)) = fixed.sample_grad(p) {
            if v >= threshold {
                out.push((p, v));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// ASGD optimizer (Klein et al. 2009) over a generic parametric problem
// ---------------------------------------------------------------------------

/// Metric value + gradient callback: given parameters, fill `grad` (scaled
/// space) and return (metric, valid_sample_fraction).
type GradFn<'a> = dyn Fn(&[f64], &mut [f64], &mut XorShift) -> (f64, f64) + Sync + 'a;

struct AsgdConfig {
    iterations: usize,
    /// elastix SP_A.
    big_a: f64,
    /// Target initial step in scaled-parameter units (≈ mm).
    delta: f64,
}

/// Run ASGD; returns (final params, last metric, iterations done) or None if
/// cancelled.
fn asgd(
    mut params: Vec<f64>,
    eval: &GradFn,
    cfg: &AsgdConfig,
    progress: &RegProgress,
    label: &str,
    metric_out: &mut f64,
) -> Option<(Vec<f64>, usize)> {
    let n = params.len();
    let mut rng = XorShift(0x9E3779B97F4A7C15 ^ (n as u64));
    let mut grad = vec![0.0; n];
    let mut prev_grad = vec![0.0; n];

    // Estimate the gain factor `a` so the first steps are ~delta: use the
    // median gradient norm of three independent sample draws (a lightweight
    // stand-in for elastix's AutomaticParameterEstimation, robust against a
    // single unlucky near-zero draw).
    let mut norms = [0.0f64; 3];
    let mut m0 = 0.0;
    for norm in &mut norms {
        let (m, _) = eval(&params, &mut grad, &mut rng);
        m0 = m;
        *norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    }
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let g0 = norms[1];
    if g0 < 1e-20 {
        *metric_out = m0;
        return Some((params, 0));
    }
    let a = cfg.delta * (cfg.big_a + 1.0) / g0;
    // Trust region: no single step may move the (scaled) parameter vector by
    // more than 2·delta, whatever the current gradient magnitude is. This
    // guards against gain over-estimation when the initial gradient is small.
    let step_cap = 2.0 * cfg.delta;
    let mut t = 0.0f64;
    let mut metric = m0;

    // Track the best parameters seen (stochastic metric, but effective as a
    // divergence guard).
    let mut best_params = params.clone();
    let mut best_metric = m0;

    for it in 0..cfg.iterations {
        if progress.cancelled() {
            return None;
        }
        let mut gamma = a / (t + cfg.big_a);
        let gnorm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm * gamma > step_cap {
            gamma = step_cap / gnorm;
        }
        for i in 0..n {
            params[i] -= gamma * grad[i];
        }
        prev_grad.copy_from_slice(&grad);
        let (m, valid) = eval(&params, &mut grad, &mut rng);
        metric = m;
        if valid < 0.25 {
            // Too few samples map into the moving image — undo the step and
            // damp (comparable to elastix's RequiredRatioOfValidSamples).
            for i in 0..n {
                params[i] += gamma * prev_grad[i];
            }
            grad.copy_from_slice(&prev_grad);
            t += 2.0;
            continue;
        }
        if m < best_metric {
            best_metric = m;
            best_params.copy_from_slice(&params);
        }
        // Klein et al. time update: t += sigmoid(-<g_k, g_{k-1}>);
        // near-step sigmoid with f_max = 1, f_min = -0.8.
        let dot: f64 = grad.iter().zip(prev_grad.iter()).map(|(a, b)| a * b).sum();
        t = (t + if dot > 0.0 { -0.8 } else { 1.0 }).max(0.0);

        if it % 25 == 0 {
            progress.set(format!("{label}: iter {}/{}  MSD {:.1}", it, cfg.iterations, m));
        }
    }
    // Return the best parameters rather than the last ones if the last
    // iterations drifted (stochastic gradients on noisy problems).
    if best_metric < metric {
        params = best_params;
        metric = best_metric;
    }
    *metric_out = metric;
    Some((params, cfg.iterations))
}

// ---------------------------------------------------------------------------
// Metric evaluation (mean squared difference, elastix AdvancedMeanSquares)
// ---------------------------------------------------------------------------

/// MSD metric only (no gradient) — used for reporting.
fn msd_value(fixed: &RegImage, moving: &RegImage, t: &Transform3, n: usize, thr: f32) -> f64 {
    let mut rng = XorShift(0xD1B54A32D192ED03);
    let samples = draw_samples(fixed, n, thr, &mut rng);
    let (sum, cnt) = samples
        .par_iter()
        .map(|&(p, f)| match moving.sample_grad(t.map(p)) {
            Some((m, _)) => ((m - f) as f64 * (m - f) as f64, 1usize),
            None => (0.0, 0),
        })
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    if cnt == 0 { f64::MAX } else { sum / cnt as f64 }
}

// ---------------------------------------------------------------------------
// Top-level: rigid and deformable registration
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

/// Register `moving` onto `fixed`. Returns the transform mapping fixed
/// patient coordinates to moving patient coordinates.
pub fn register(
    fixed_vol: &Volume,
    moving_vol: &Volume,
    params: &RegParams,
    progress: &RegProgress,
) -> Result<RegistrationResult> {
    let t_start = std::time::Instant::now();
    progress.set("Building image pyramids…");

    let fixed_full = RegImage::from_volume(fixed_vol);
    let moving_full = RegImage::from_volume(moving_vol);

    // Fixed-image center of rotation (elastix AutomaticTransformInitialization
    // with CenterOfGravity would be similar; geometric center is robust here).
    let center = fixed_full.index_to_patient(
        (fixed_full.dims[0] as f64 - 1.0) * 0.5,
        (fixed_full.dims[1] as f64 - 1.0) * 0.5,
        (fixed_full.dims[2] as f64 - 1.0) * 0.5,
    );

    let fixed_pyr = build_pyramid(fixed_full, params.levels);
    let moving_pyr = build_pyramid(moving_full, params.levels);

    let initial_metric = msd_value(
        &fixed_pyr[params.levels - 1],
        &moving_pyr[params.levels - 1],
        &Transform3 { rigid: RigidTransform::identity(center), bspline: None },
        params.samples.max(2000),
        params.fixed_threshold,
    );

    // Rotation parameter scale (elastix AutomaticScalesEstimation analogue):
    // 1 rad of rotation moves a typical point by ~r mm.
    let ext = [
        fixed_vol.dims[0] as f64 * fixed_vol.spacing[0],
        fixed_vol.dims[1] as f64 * fixed_vol.spacing[1],
        fixed_vol.dims[2] as f64 * fixed_vol.spacing[2],
    ];
    let rot_scale = 0.25 * (ext[0] + ext[1] + ext[2]) / 3.0 * 2.0; // ≈ half mean extent

    // ---------------- Rigid stage (always runs) ----------------
    let mut rigid = RigidTransform::identity(center);
    let mut total_iters = 0usize;
    let mut last_metric = initial_metric;

    for level in 0..params.levels {
        if progress.cancelled() {
            bail!("registration cancelled");
        }
        let fixed = &fixed_pyr[level];
        let moving = &moving_pyr[level];
        let delta = fixed.spacing.iter().cloned().fold(0.0f64, f64::max);
        let n_samples = params.samples;
        let thr = params.fixed_threshold;
        let label = format!("Rigid L{}/{}", level + 1, params.levels);

        let center_l = center;
        let eval = move |p: &[f64], grad: &mut [f64], rng: &mut XorShift| -> (f64, f64) {
            let tr = RigidTransform {
                params: [
                    p[0] / rot_scale,
                    p[1] / rot_scale,
                    p[2] / rot_scale,
                    p[3],
                    p[4],
                    p[5],
                ],
                center: center_l,
            };
            let samples = draw_samples(fixed, n_samples, thr, rng);
            let n_total = samples.len().max(1);
            let (g, sum, cnt) = samples
                .par_iter()
                .map(|&(x, fval)| {
                    let mut gl = [0.0f64; 6];
                    match moving.sample_grad(tr.map(x)) {
                        Some((mval, mg)) => {
                            let diff = (mval - fval) as f64;
                            let jac = tr.jacobian(x);
                            for (pi, j) in jac.iter().enumerate() {
                                gl[pi] = 2.0 * diff * mg.dot(*j);
                            }
                            // chain rule into scaled space
                            gl[0] /= rot_scale;
                            gl[1] /= rot_scale;
                            gl[2] /= rot_scale;
                            (gl, diff * diff, 1usize)
                        }
                        None => (gl, 0.0, 0usize),
                    }
                })
                .reduce(
                    || ([0.0; 6], 0.0, 0),
                    |a, b| {
                        let mut g = a.0;
                        for i in 0..6 {
                            g[i] += b.0[i];
                        }
                        (g, a.1 + b.1, a.2 + b.2)
                    },
                );
            let cntf = cnt.max(1) as f64;
            for i in 0..6 {
                grad[i] = g[i] / cntf;
            }
            (sum / cntf, cnt as f64 / n_total as f64)
        };

        let scaled0 = vec![
            rigid.params[0] * rot_scale,
            rigid.params[1] * rot_scale,
            rigid.params[2] * rot_scale,
            rigid.params[3],
            rigid.params[4],
            rigid.params[5],
        ];
        let mut mlast = last_metric;
        let Some((scaled, iters)) = asgd(
            scaled0,
            &eval,
            &AsgdConfig { iterations: params.iterations, big_a: 20.0, delta },
            progress,
            &label,
            &mut mlast,
        ) else {
            bail!("registration cancelled");
        };
        rigid.params = [
            scaled[0] / rot_scale,
            scaled[1] / rot_scale,
            scaled[2] / rot_scale,
            scaled[3],
            scaled[4],
            scaled[5],
        ];
        total_iters += iters;
        last_metric = mlast;
    }

    let mut transform = Transform3 { rigid: rigid.clone(), bspline: None };

    // ---------------- B-spline stage (deformable only) ----------------
    if params.kind == RegKind::Deformable {
        let mut bspline = BSplineTransform::new(fixed_vol, params.grid_spacing_mm);
        let n_coeffs = bspline.coeffs.len();
        let [gnx, gny, _] = bspline.grid_dims;

        for level in 0..params.levels {
            if progress.cancelled() {
                bail!("registration cancelled");
            }
            let fixed = &fixed_pyr[level];
            let moving = &moving_pyr[level];
            let delta = 0.5 * fixed.spacing.iter().cloned().fold(0.0f64, f64::max);
            let n_samples = params.samples;
            let thr = params.fixed_threshold;
            let label = format!("B-spline L{}/{}", level + 1, params.levels);
            let rigid_l = rigid.clone();
            let grid = bspline.clone(); // geometry only; coeffs come from p

            let eval = move |p: &[f64], grad: &mut [f64], rng: &mut XorShift| -> (f64, f64) {
                let samples = draw_samples(fixed, n_samples, thr, rng);
                let n_total = samples.len().max(1);
                let (gsum, sum, cnt) = samples
                    .par_iter()
                    .map(|&(x, fval)| {
                        let mut sparse: Vec<(usize, f64)> = Vec::new();
                        let (val, ok) = {
                            let Some((base, w)) = grid.support(x) else {
                                return (sparse, 0.0, 0usize);
                            };
                            // displacement from p
                            let mut disp = Vec3::ZERO;
                            let mut touched = [(0usize, 0.0f64); 64];
                            let mut ti = 0;
                            for kz in 0..4 {
                                let iz = (base[2] + kz as i64) as usize;
                                for ky in 0..4 {
                                    let iy = (base[1] + ky as i64) as usize;
                                    let wyz = w[1][ky] * w[2][kz];
                                    let row =
                                        3 * (base[0] as usize + gnx * (iy + gny * iz));
                                    for kx in 0..4 {
                                        let wt = w[0][kx] * wyz;
                                        let o = row + 3 * kx;
                                        disp.x += wt * p[o];
                                        disp.y += wt * p[o + 1];
                                        disp.z += wt * p[o + 2];
                                        touched[ti] = (o, wt);
                                        ti += 1;
                                    }
                                }
                            }
                            let q = rigid_l.map(x) + disp;
                            match moving.sample_grad(q) {
                                Some((mval, mg)) => {
                                    let diff = (mval - fval) as f64;
                                    for &(o, wt) in touched.iter() {
                                        let c = 2.0 * diff * wt;
                                        sparse.push((o, c * mg.x));
                                        sparse.push((o + 1, c * mg.y));
                                        sparse.push((o + 2, c * mg.z));
                                    }
                                    (diff * diff, 1usize)
                                }
                                None => (0.0, 0usize),
                            }
                        };
                        (sparse, val, ok)
                    })
                    .fold(
                        || (vec![0.0f64; n_coeffs], 0.0f64, 0usize),
                        |mut acc, item| {
                            for (o, v) in item.0 {
                                acc.0[o] += v;
                            }
                            (acc.0, acc.1 + item.1, acc.2 + item.2)
                        },
                    )
                    .reduce(
                        || (vec![0.0f64; n_coeffs], 0.0f64, 0usize),
                        |mut a, b| {
                            for i in 0..n_coeffs {
                                a.0[i] += b.0[i];
                            }
                            (a.0, a.1 + b.1, a.2 + b.2)
                        },
                    );
                let cntf = cnt.max(1) as f64;
                for i in 0..n_coeffs {
                    grad[i] = gsum[i] / cntf;
                }
                (sum / cntf, cnt as f64 / n_total as f64)
            };

            let mut mlast = last_metric;
            let Some((coeffs, iters)) = asgd(
                bspline.coeffs.clone(),
                &eval,
                &AsgdConfig { iterations: params.iterations, big_a: 20.0, delta },
                progress,
                &label,
                &mut mlast,
            ) else {
                bail!("registration cancelled");
            };
            bspline.coeffs = coeffs;
            total_iters += iters;
            last_metric = mlast;
        }
        transform.bspline = Some(bspline);
    }

    // Final metric on the full-resolution images.
    let final_metric = msd_value(
        &fixed_pyr[params.levels - 1],
        &moving_pyr[params.levels - 1],
        &transform,
        params.samples.max(2000),
        params.fixed_threshold,
    );

    progress.set("done");
    Ok(RegistrationResult {
        transform: Arc::new(transform),
        kind: params.kind,
        initial_metric,
        final_metric,
        iterations_run: total_iters,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    })
}
