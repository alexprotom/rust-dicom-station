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

    pub fn center(&self) -> Vec3 {
        self.center
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

    /// A copy carrying only the grid geometry, with no coefficients.
    fn geometry(&self) -> BSplineTransform {
        BSplineTransform { coeffs: Vec::new(), ..self.clone() }
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
    /// Flat indices of voxels eligible for random sampling (value above the
    /// fixed-image threshold), built once per pyramid level by
    /// [`RegImage::prepare_sampling`]. Sampling draws from this list instead
    /// of rejecting random draws, which keeps every draw a hit.
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

    /// Build the eligible-voxel list for random sampling. Voxels on the far
    /// boundary are excluded so a jittered sample always has a full
    /// interpolation neighbourhood.
    fn prepare_sampling(&mut self, threshold: f32) {
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
            .collect();
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
}

// ---------------------------------------------------------------------------
// Random coordinate sampler (xorshift; deterministic per level for
// reproducibility, fresh samples every iteration — elastix RandomCoordinate
// with NewSamplesEveryIteration=true)
// ---------------------------------------------------------------------------

struct XorShift(u64);

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

    let mut fixed_pyr = build_pyramid(fixed_full, params.levels);
    let moving_pyr = build_pyramid(moving_full, params.levels);

    // Eligible-voxel lists (the fixed-image mask) are built once per level
    // instead of being re-derived by rejection on every iteration.
    for img in &mut fixed_pyr {
        img.prepare_sampling(params.fixed_threshold);
    }
    if fixed_pyr.iter().all(|i| i.eligible.is_empty()) {
        bail!("no fixed-image voxels above the sampling threshold — lower it and retry");
    }

    let initial_metric = msd_value(
        &fixed_pyr[params.levels - 1],
        &moving_pyr[params.levels - 1],
        &Transform3 { rigid: RigidTransform::identity(center), bspline: None },
        params.samples.max(2000),
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
        let label = format!("Rigid L{}/{}", level + 1, params.levels);

        let center_l = center;
        let eval = move |p: &[f64], grad: &mut [f64], rng: &mut XorShift| -> (f64, f64) {
            let tr = RigidTransform::new(
                [p[0] / rot_scale, p[1] / rot_scale, p[2] / rot_scale, p[3], p[4], p[5]],
                center_l,
            );
            let samples = draw_samples(fixed, n_samples, rng);
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
                        for (x, y) in g.iter_mut().zip(b.0) {
                            *x += y;
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

        let rp = rigid.params();
        let scaled0 = vec![
            rp[0] * rot_scale,
            rp[1] * rot_scale,
            rp[2] * rot_scale,
            rp[3],
            rp[4],
            rp[5],
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
        rigid = RigidTransform::new(
            [
                scaled[0] / rot_scale,
                scaled[1] / rot_scale,
                scaled[2] / rot_scale,
                scaled[3],
                scaled[4],
                scaled[5],
            ],
            center,
        );
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
            let label = format!("B-spline L{}/{}", level + 1, params.levels);
            let rigid_l = rigid.clone();
            // Grid geometry only - the coefficients come from `p` on every
            // call, so the coefficient vector itself is not carried along.
            let grid = bspline.geometry();

            let eval = move |p: &[f64], grad: &mut [f64], rng: &mut XorShift| -> (f64, f64) {
                let samples = draw_samples(fixed, n_samples, rng);
                let n_total = samples.len().max(1);
                // One dense gradient accumulator per worker chunk, scattered
                // into directly. The previous shape built a 192-entry sparse
                // Vec for every sample (one heap allocation per sample) and
                // let rayon allocate an `n_coeffs` accumulator per split,
                // so the gradient reduction cost far more than the metric.
                let chunk = samples.len().div_ceil(rayon::current_num_threads().max(1)).max(1);
                let (gsum, sum, cnt) = samples
                    .par_chunks(chunk)
                    .fold(
                        || (vec![0.0f64; n_coeffs], 0.0f64, 0usize),
                        |acc, part| {
                            part.iter().fold(acc, |mut acc, &(x, fval)| {
                                let Some((base, w)) = grid.support(x) else {
                                    return acc;
                                };
                                let mut disp = Vec3::ZERO;
                                let mut touched = [(0usize, 0.0f64); 64];
                                let mut ti = 0;
                                for kz in 0..4 {
                                    let iz = (base[2] + kz as i64) as usize;
                                    for (ky, wy) in w[1].iter().enumerate() {
                                        let iy = (base[1] + ky as i64) as usize;
                                        let wyz = wy * w[2][kz];
                                        let row = 3 * (base[0] as usize + gnx * (iy + gny * iz));
                                        for (kx, wx) in w[0].iter().enumerate() {
                                            let wt = wx * wyz;
                                            let o = row + 3 * kx;
                                            disp.x += wt * p[o];
                                            disp.y += wt * p[o + 1];
                                            disp.z += wt * p[o + 2];
                                            touched[ti] = (o, wt);
                                            ti += 1;
                                        }
                                    }
                                }
                                let Some((mval, mg)) = moving.sample_grad(rigid_l.map(x) + disp)
                                else {
                                    return acc;
                                };
                                let diff = (mval - fval) as f64;
                                for &(o, wt) in &touched[..ti] {
                                    let c = 2.0 * diff * wt;
                                    acc.0[o] += c * mg.x;
                                    acc.0[o + 1] += c * mg.y;
                                    acc.0[o + 2] += c * mg.z;
                                }
                                acc.1 += diff * diff;
                                acc.2 += 1;
                                acc
                            })
                        },
                    )
                    .reduce(
                        || (vec![0.0f64; n_coeffs], 0.0f64, 0usize),
                        |mut a, b| {
                            for (x, y) in a.0.iter_mut().zip(b.0.iter()) {
                                *x += y;
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
