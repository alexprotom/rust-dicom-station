//! The plastimatch engine: a dense B-spline registration with an exact
//! analytic gradient, a bending-energy regularizer and a quasi-Newton
//! optimizer.
//!
//! Where [`super::elastix`] estimates the metric from a few thousand fresh
//! random samples per iteration and takes many cheap noisy steps, this
//! engine follows [plastimatch](https://plastimatch.org)'s `bspline`
//! (Shackleford et al., *High performance deformable image registration
//! algorithms for manycore processors*): it evaluates the cost and its exact
//! gradient over **every** eligible fixed voxel, scatters each voxel's
//! contribution onto the 64 control points that support it, adds a
//! smoothness penalty on the control lattice, and hands the result to a
//! quasi-Newton optimizer with a line search. Each iteration costs orders of
//! magnitude more; far fewer are needed, the result is deterministic, and
//! the field is smoother.
//!
//! The stages follow plastimatch's own vocabulary:
//!
//! 1. `xform=align_center` — a translation matching the centres of gravity
//!    of the two thresholded images. Cheap, and it removes the gross offset
//!    a deformable model should never have to represent.
//! 2. `xform=bspline` per resolution level, coarse to fine, with
//!    `grid_spacing`, `young_modulus` (the regularizer) and `max_its`.
//!
//! **Metric.** `mse` is the mean squared difference; `mi` is Mattes mutual
//! information over a 32 × 32 joint histogram with a zero-order Parzen
//! window on the fixed image and a cubic B-spline window on the moving one
//! (Mattes et al., IEEE TMI 2003) — the only metric here that survives two
//! modalities. Both are dimensionless in the cost below: the squared
//! difference is divided by the fixed image's variance, so the regularizer's
//! weight means the same thing whatever the images contain.
//!
//! **Regularizer.** The discrete bending energy of the control lattice,
//! `Σ (∂²c/∂x² )² + … + 2(∂²c/∂x∂y)² + …` evaluated by second differences and
//! made dimensionless by the lattice spacing. It is quadratic in the
//! coefficients, so its gradient is exact rather than approximated.
//!
//! **Optimizer.** L-BFGS (limited-memory BFGS, two-loop recursion, history
//! 6) with an Armijo backtracking line search. plastimatch's default is
//! L-BFGS-B; the bounded variant differs only in handling box constraints on
//! the parameters, and B-spline coefficients have none.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::*;

/// How many samples one level may use before the eligible list is thinned.
/// "Dense" means every eligible voxel; on a 512³ study that is tens of
/// millions, and an exact gradient over all of them is not what anybody
/// wants to wait for. The cap keeps the engine's character — the same
/// deterministic sample set every iteration — while bounding the cost.
const MAX_DENSE_SAMPLES: usize = 400_000;

/// Joint-histogram bins per axis for Mattes mutual information.
const MI_BINS: usize = 32;

/// Cubic B-spline Parzen window and its derivative.
#[inline]
fn beta3(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        2.0 / 3.0 - a * a + a * a * a * 0.5
    } else if a < 2.0 {
        let d = 2.0 - a;
        d * d * d / 6.0
    } else {
        0.0
    }
}

#[inline]
fn dbeta3(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        -2.0 * t + 1.5 * t * a
    } else if a < 2.0 {
        let d = 2.0 - a;
        -t.signum() * d * d * 0.5
    } else {
        0.0
    }
}

/// Centre of gravity of the voxels at or above `threshold`.
fn center_of_gravity(img: &RegImage, threshold: f32) -> Option<Vec3> {
    let [nx, ny, _] = img.dims;
    let (sum, n) = img
        .data
        .par_iter()
        .enumerate()
        .filter(|(_, &v)| v >= threshold)
        .map(|(o, _)| {
            let k = o / (nx * ny);
            let rem = o - k * nx * ny;
            (
                img.index_to_patient((rem % nx) as f64, (rem / nx) as f64, k as f64),
                1usize,
            )
        })
        .reduce(|| (Vec3::ZERO, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    (n > 0).then(|| sum * (1.0 / n as f64))
}

/// Centre of gravity of the eligible voxels of a prepared fixed image.
fn eligible_center_of_gravity(img: &RegImage) -> Option<Vec3> {
    let [nx, ny, _] = img.dims;
    let (sum, n) = img
        .eligible
        .par_iter()
        .map(|&o| {
            let o = o as usize;
            let k = o / (nx * ny);
            let rem = o - k * nx * ny;
            (
                img.index_to_patient((rem % nx) as f64, (rem / nx) as f64, k as f64),
                1usize,
            )
        })
        .reduce(|| (Vec3::ZERO, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    (n > 0).then(|| sum * (1.0 / n as f64))
}

/// The bins and scales one level's mutual information is computed over.
#[derive(Clone, Copy)]
struct MiScale {
    f_min: f64,
    f_step: f64,
    m_min: f64,
    m_step: f64,
}

impl MiScale {
    fn of(fixed: &RegImage, moving: &RegImage) -> MiScale {
        let (fa, fb) = fixed.eligible_range();
        let (ma, mb) = (
            moving
                .data
                .par_iter()
                .cloned()
                .reduce(|| f32::MAX, f32::min),
            moving
                .data
                .par_iter()
                .cloned()
                .reduce(|| f32::MIN, f32::max),
        );
        MiScale {
            f_min: fa as f64,
            f_step: ((fb - fa) as f64 / (MI_BINS - 1) as f64).max(1e-6),
            m_min: ma as f64,
            // Two padding bins at each end so the cubic window never reaches
            // outside the histogram.
            m_step: ((mb - ma) as f64 / (MI_BINS - 5) as f64).max(1e-6),
        }
    }

    /// Zero-order bin of a fixed-image value.
    #[inline]
    fn fixed_bin(&self, v: f32) -> usize {
        (((v as f64 - self.f_min) / self.f_step).round() as i64).clamp(0, MI_BINS as i64 - 1)
            as usize
    }

    /// Continuous bin coordinate of a moving-image value.
    #[inline]
    fn moving_coord(&self, v: f32) -> f64 {
        ((v as f64 - self.m_min) / self.m_step + 2.0).clamp(2.0, MI_BINS as f64 - 3.0)
    }
}

/// One resolution level's problem: the sample set, the lattice geometry and
/// everything the cost function needs.
struct Level<'a> {
    moving: &'a RegImage,
    /// Fixed-image points with their image values.
    samples: Vec<(Vec3, f32)>,
    /// Where each sample already maps to before this level's correction.
    base: Vec<Vec3>,
    grid: BSplineTransform,
    n_coeffs: usize,
    /// Variance of the fixed values, so the data term is dimensionless.
    variance: f64,
    metric: Metric,
    mi: MiScale,
    /// Bending-energy weight.
    lambda: f64,
}

impl Level<'_> {
    /// Displacement of every sample under `c`, plus the support each sample
    /// touches. Recomputed per evaluation; this is the engine's hot loop.
    fn cost_and_gradient(&self, c: &[f64], grad: Option<&mut [f64]>) -> f64 {
        let [gnx, gny, _] = self.grid.grid_dims;
        let grid = &self.grid;

        // ---- pass 1: where does every sample land, and is it inside? ----
        let mapped: Vec<Option<(f32, Vec3)>> = self
            .samples
            .par_iter()
            .zip(self.base.par_iter())
            .map(|(&(x, _), &b)| {
                let d = match grid.support(x) {
                    None => Vec3::ZERO,
                    Some((base, w)) => {
                        let mut disp = Vec3::ZERO;
                        for kz in 0..4 {
                            let iz = (base[2] + kz as i64) as usize;
                            for (ky, wy) in w[1].iter().enumerate() {
                                let iy = (base[1] + ky as i64) as usize;
                                let wyz = wy * w[2][kz];
                                let row = 3 * (base[0] as usize + gnx * (iy + gny * iz));
                                for (kx, wx) in w[0].iter().enumerate() {
                                    let wt = wx * wyz;
                                    let o = row + 3 * kx;
                                    disp.x += wt * c[o];
                                    disp.y += wt * c[o + 1];
                                    disp.z += wt * c[o + 2];
                                }
                            }
                        }
                        disp
                    }
                };
                self.moving.sample_grad(b + d)
            })
            .collect();

        // ---- the per-sample scalar the gradient scatters ----
        // For both metrics the gradient has the same shape,
        //   ∂C/∂c = Σ_samples s_k · ∇I_moving(y_k) · ∂y_k/∂c,
        // and only the scalar s_k differs. That is what lets one scatter
        // loop serve the mean-squared and the mutual-information cost alike.
        let (cost, scalars) = match self.metric {
            Metric::MeanSquares => {
                let (sum, cnt) = mapped
                    .par_iter()
                    .zip(self.samples.par_iter())
                    .map(|(m, &(_, f))| match m {
                        Some((v, _)) => ((v - f) as f64 * (v - f) as f64, 1usize),
                        None => (0.0, 0),
                    })
                    .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
                let denom = (cnt.max(1) as f64) * self.variance;
                let s: Vec<f64> = mapped
                    .par_iter()
                    .zip(self.samples.par_iter())
                    .map(|(m, &(_, f))| match m {
                        Some((v, _)) => 2.0 * (v - f) as f64 / denom,
                        None => 0.0,
                    })
                    .collect();
                (sum / denom, s)
            }
            Metric::MutualInformation => self.mi_cost_and_scalars(&mapped),
        };

        let mut total = cost;

        if let Some(g) = grad {
            g.iter_mut().for_each(|v| *v = 0.0);
            let chunk = self
                .samples
                .len()
                .div_ceil(rayon::current_num_threads().max(1))
                .max(1);
            let acc = self
                .samples
                .par_chunks(chunk)
                .zip(mapped.par_chunks(chunk))
                .zip(scalars.par_chunks(chunk))
                .fold(
                    || vec![0.0f64; self.n_coeffs],
                    |mut acc, ((part, mp), sc)| {
                        for ((&(x, _), m), &s) in part.iter().zip(mp).zip(sc) {
                            let Some((_, mg)) = m else { continue };
                            if s == 0.0 {
                                continue;
                            }
                            let Some((base, w)) = grid.support(x) else {
                                continue;
                            };
                            let sx = s * mg.x;
                            let sy = s * mg.y;
                            let sz = s * mg.z;
                            for kz in 0..4 {
                                let iz = (base[2] + kz as i64) as usize;
                                for (ky, wy) in w[1].iter().enumerate() {
                                    let iy = (base[1] + ky as i64) as usize;
                                    let wyz = wy * w[2][kz];
                                    let row = 3 * (base[0] as usize + gnx * (iy + gny * iz));
                                    for (kx, wx) in w[0].iter().enumerate() {
                                        let wt = wx * wyz;
                                        let o = row + 3 * kx;
                                        acc[o] += wt * sx;
                                        acc[o + 1] += wt * sy;
                                        acc[o + 2] += wt * sz;
                                    }
                                }
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || vec![0.0f64; self.n_coeffs],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b.iter()) {
                            *x += y;
                        }
                        a
                    },
                );
            g.copy_from_slice(&acc);
            if self.lambda > 0.0 {
                total += bending_energy(c, self.grid.grid_dims, self.grid.spacing, self.lambda, g);
            }
        } else if self.lambda > 0.0 {
            total += bending_energy_value(c, self.grid.grid_dims, self.grid.spacing, self.lambda);
        }
        total
    }

    /// Mattes mutual information: build the joint histogram, then the
    /// per-sample scalar `∂(−MI)/∂I_moving`.
    fn mi_cost_and_scalars(&self, mapped: &[Option<(f32, Vec3)>]) -> (f64, Vec<f64>) {
        let nb = MI_BINS;
        // The histogram is a reduction over a few hundred thousand samples
        // into a 1024-entry table; per-thread tables and one merge.
        let chunk = mapped
            .len()
            .div_ceil(rayon::current_num_threads().max(1))
            .max(1);
        let (mut joint, valid) = mapped
            .par_chunks(chunk)
            .zip(self.samples.par_chunks(chunk))
            .map(|(mp, part)| {
                let mut h = vec![0.0f64; nb * nb];
                let mut c = 0usize;
                for (m, &(_, f)) in mp.iter().zip(part) {
                    let Some((mv, _)) = m else { continue };
                    let fb = self.mi.fixed_bin(f);
                    let u = self.mi.moving_coord(*mv);
                    let b0 = u.floor() as i64;
                    for d in -1i64..=2 {
                        let b = (b0 + d).clamp(0, nb as i64 - 1) as usize;
                        h[fb * nb + b] += beta3(u - (b0 + d) as f64);
                    }
                    c += 1;
                }
                (h, c)
            })
            .reduce(
                || (vec![0.0f64; nb * nb], 0usize),
                |mut a, b| {
                    for (x, y) in a.0.iter_mut().zip(b.0.iter()) {
                        *x += y;
                    }
                    (a.0, a.1 + b.1)
                },
            );
        if valid == 0 {
            return (0.0, vec![0.0; mapped.len()]);
        }
        let inv = 1.0 / valid as f64;
        for p in joint.iter_mut() {
            *p *= inv;
        }
        let mut p_f = vec![0.0f64; nb];
        let mut p_m = vec![0.0f64; nb];
        for f in 0..nb {
            for m in 0..nb {
                p_f[f] += joint[f * nb + m];
                p_m[m] += joint[f * nb + m];
            }
        }
        let mut mi = 0.0;
        for f in 0..nb {
            for m in 0..nb {
                let p = joint[f * nb + m];
                if p > 1e-12 && p_f[f] > 1e-12 && p_m[m] > 1e-12 {
                    mi += p * (p / (p_f[f] * p_m[m])).ln();
                }
            }
        }
        // ∂(−MI)/∂I_moving(y_k): only the joint and the moving marginal
        // depend on the transform — the fixed marginal cannot.
        let scale = -inv / self.mi.m_step;
        let scalars: Vec<f64> = mapped
            .par_iter()
            .zip(self.samples.par_iter())
            .map(|(m, &(_, f))| {
                let Some((mv, _)) = m else { return 0.0 };
                let fb = self.mi.fixed_bin(f);
                let u = self.mi.moving_coord(*mv);
                let b0 = u.floor() as i64;
                let mut s = 0.0;
                for d in -1i64..=2 {
                    let b = (b0 + d).clamp(0, nb as i64 - 1) as usize;
                    let p = joint[fb * nb + b];
                    let pm = p_m[b];
                    if p > 1e-12 && pm > 1e-12 {
                        s += dbeta3(u - (b0 + d) as f64) * (p / pm).ln();
                    }
                }
                // −MI is what is minimized, hence the sign in `scale`.
                s * scale
            })
            .collect();
        (-mi, scalars)
    }
}

/// Discrete bending energy of the control lattice and its exact gradient.
///
/// `E = (λ / N) Σ_interior Σ_components [ (∂²c/∂x²)² + (∂²c/∂y²)² +
/// (∂²c/∂z²)² + 2(∂²c/∂x∂y)² + 2(∂²c/∂x∂z)² + 2(∂²c/∂y∂z)² ]`, evaluated by
/// second differences on the lattice and divided by the lattice spacing
/// squared so the whole term is dimensionless — the same λ then means the
/// same amount of smoothing whatever the grid spacing and the image size.
///
/// The energy is quadratic in the coefficients, so what is added to `grad`
/// below is the derivative itself and not an approximation of one; the unit
/// test checks it against a central difference.
fn bending_energy(c: &[f64], dims: [usize; 3], spacing: f64, lambda: f64, grad: &mut [f64]) -> f64 {
    let [nx, ny, nz] = dims;
    if nx < 3 || ny < 3 || nz < 3 || lambda <= 0.0 {
        return 0.0;
    }
    let n_interior = ((nx - 2) * (ny - 2) * (nz - 2)) as f64;
    let w = lambda / (n_interior.max(1.0) * spacing * spacing);
    let at = |i: usize, j: usize, k: usize, comp: usize| 3 * (i + nx * (j + ny * k)) + comp;
    // (axis a, axis b, weight): a == b is a pure second derivative, a != b a
    // mixed one, which the thin-plate energy counts twice.
    const TERMS: [(usize, usize, f64); 6] = [
        (0, 0, 1.0),
        (1, 1, 1.0),
        (2, 2, 1.0),
        (0, 1, 2.0),
        (0, 2, 2.0),
        (1, 2, 2.0),
    ];
    let mut energy = 0.0;
    for k in 1..nz - 1 {
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                for comp in 0..3 {
                    for (a, b, wt) in TERMS {
                        let step = |axis: usize, s: i64| -> [i64; 3] {
                            let mut d = [0i64; 3];
                            d[axis] = s;
                            d
                        };
                        let idx = |d: [i64; 3]| {
                            at(
                                (i as i64 + d[0]) as usize,
                                (j as i64 + d[1]) as usize,
                                (k as i64 + d[2]) as usize,
                                comp,
                            )
                        };
                        if a == b {
                            let p = idx(step(a, 1));
                            let m = idx([0, 0, 0]);
                            let q = idx(step(a, -1));
                            let d2 = c[p] - 2.0 * c[m] + c[q];
                            energy += w * wt * d2 * d2;
                            let g = 2.0 * w * wt * d2;
                            grad[p] += g;
                            grad[m] -= 2.0 * g;
                            grad[q] += g;
                        } else {
                            let mix = |sa: i64, sb: i64| {
                                let mut d = [0i64; 3];
                                d[a] = sa;
                                d[b] = sb;
                                idx(d)
                            };
                            let (pp, pm, mp, mm) = (mix(1, 1), mix(1, -1), mix(-1, 1), mix(-1, -1));
                            let d2 = 0.25 * (c[pp] - c[pm] - c[mp] + c[mm]);
                            energy += w * wt * d2 * d2;
                            let g = 2.0 * w * wt * d2 * 0.25;
                            grad[pp] += g;
                            grad[pm] -= g;
                            grad[mp] -= g;
                            grad[mm] += g;
                        }
                    }
                }
            }
        }
    }
    energy
}

/// The bending energy alone (line-search evaluations need no gradient).
fn bending_energy_value(c: &[f64], dims: [usize; 3], spacing: f64, lambda: f64) -> f64 {
    let mut throwaway = vec![0.0; c.len()];
    bending_energy(c, dims, spacing, lambda, &mut throwaway)
}

/// L-BFGS with an Armijo backtracking line search.
///
/// Returns the optimized coefficients, the number of cost evaluations and
/// the final cost, or `None` if the run was cancelled.
fn lbfgs(
    mut x: Vec<f64>,
    level: &Level,
    max_iters: usize,
    first_step: f64,
    progress: &Progress,
    label: &str,
) -> Option<(Vec<f64>, usize, f64)> {
    const HISTORY: usize = 6;
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut f = level.cost_and_gradient(&x, Some(&mut g));
    let mut evals = 1usize;
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho: Vec<f64> = Vec::new();
    let mut stalled = 0usize;

    for it in 0..max_iters {
        if progress.cancelled() {
            return None;
        }
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !gnorm.is_finite() || gnorm < 1e-14 {
            break;
        }
        // Two-loop recursion for the search direction.
        let mut q = g.clone();
        let mut alpha = vec![0.0; s_hist.len()];
        for i in (0..s_hist.len()).rev() {
            let a = rho[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            axpy(-a, &y_hist[i], &mut q);
        }
        let gamma = if let (Some(s), Some(y)) = (s_hist.last(), y_hist.last()) {
            dot(s, y) / dot(y, y).max(1e-30)
        } else {
            // First step: scale so the largest coefficient moves about one
            // voxel, which is what plastimatch's first line search finds too.
            first_step / gnorm.max(1e-30)
        };
        q.iter_mut().for_each(|v| *v *= gamma);
        for i in 0..s_hist.len() {
            let b = rho[i] * dot(&y_hist[i], &q);
            axpy(alpha[i] - b, &s_hist[i], &mut q);
        }
        // `q` is now the quasi-Newton direction; descend along −q.
        let dir: Vec<f64> = q.iter().map(|v| -v).collect();
        let slope = dot(&g, &dir);
        if slope >= 0.0 {
            // Not a descent direction (a bad curvature pair): restart.
            s_hist.clear();
            y_hist.clear();
            rho.clear();
            continue;
        }

        // Armijo backtracking.
        let mut step = 1.0f64;
        let mut ok = false;
        let mut x_new = x.clone();
        for _ in 0..24 {
            if progress.cancelled() {
                return None;
            }
            x_new.copy_from_slice(&x);
            axpy(step, &dir, &mut x_new);
            let trial = level.cost_and_gradient(&x_new, None);
            evals += 1;
            if trial.is_finite() && trial <= f + 1e-4 * step * slope {
                ok = true;
                break;
            }
            step *= 0.5;
        }
        if !ok {
            break;
        }

        let mut g_new = vec![0.0; n];
        let f_new = level.cost_and_gradient(&x_new, Some(&mut g_new));
        evals += 1;
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if s_hist.len() == HISTORY {
                s_hist.remove(0);
                y_hist.remove(0);
                rho.remove(0);
            }
            s_hist.push(s);
            y_hist.push(y);
            rho.push(1.0 / sy);
        }
        let improvement = (f - f_new).abs() / f.abs().max(1e-12);
        x = x_new;
        g = g_new;
        f = f_new;
        if it % 2 == 0 {
            progress.set(format!("{label}: iter {}/{max_iters}  cost {f:.5}", it + 1));
        }
        stalled = if improvement < 1e-6 { stalled + 1 } else { 0 };
        if stalled >= 3 {
            break;
        }
    }
    Some((x, evals, f))
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (o, v) in y.iter_mut().zip(x) {
        *o += a * v;
    }
}

/// Variance of the fixed-image values of a sample set (the denominator that
/// makes the squared-difference term dimensionless).
fn variance(samples: &[(Vec3, f32)]) -> f64 {
    if samples.is_empty() {
        return 1.0;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().map(|s| s.1 as f64).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|s| {
            let d = s.1 as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    var.max(1.0)
}

/// The effective stride for one level: the user's, or whatever keeps the
/// sample count under [`MAX_DENSE_SAMPLES`], whichever thins more.
fn effective_stride(eligible: usize, requested: usize) -> usize {
    let by_cap = eligible.div_ceil(MAX_DENSE_SAMPLES).max(1);
    requested.max(1).max(by_cap)
}

/// −MI of a transform at the finest level, for the before/after readout.
pub(super) fn mi_value(setup: &RegSetup, t: &Transform3) -> f64 {
    let finest = setup.finest();
    let fixed = &setup.fixed[finest];
    let moving = &setup.moving[finest];
    let stride = effective_stride(fixed.eligible.len(), setup.params.stride);
    let samples = fixed.dense_samples(stride);
    if samples.is_empty() {
        return 0.0;
    }
    let base: Vec<Vec3> = samples.par_iter().map(|&(x, _)| t.map(x)).collect();
    let level = Level {
        moving,
        variance: variance(&samples),
        mi: MiScale::of(fixed, moving),
        metric: Metric::MutualInformation,
        grid: BSplineTransform::for_region(setup.fixed_vol, None, 1e9),
        n_coeffs: 0,
        lambda: 0.0,
        samples,
        base,
    };
    let mapped: Vec<Option<(f32, Vec3)>> = level
        .base
        .par_iter()
        .map(|&y| level.moving.sample_grad(y))
        .collect();
    level.mi_cost_and_scalars(&mapped).0
}

/// Run the plastimatch engine.
pub(super) fn run(setup: &RegSetup, progress: &Progress) -> Result<EngineOutput> {
    let params = setup.params;
    let levels = setup.fixed.len();

    // ---- stage 1: align_center -----------------------------------------
    // Skipped for a local run and for a refinement — both already start from
    // an alignment, and matching the centres of gravity of a structure
    // against the whole moving image would undo it.
    let base_transform = match params.start.as_deref() {
        Some(s) => s.clone(),
        None => {
            let mut rigid = RigidTransform::identity(setup.center);
            if params.region.is_none() {
                progress.set("align_center: matching the centres of gravity");
                let cf = eligible_center_of_gravity(&setup.fixed[0]);
                let cm = center_of_gravity(&setup.moving[0], params.fixed_threshold);
                if let (Some(cf), Some(cm)) = (cf, cm) {
                    let t = cm - cf;
                    rigid = RigidTransform::new([0.0, 0.0, 0.0, t.x, t.y, t.z], setup.center);
                }
            }
            Transform3::rigid_only(rigid)
        }
    };
    let rigid = base_transform.rigid.clone();

    // ---- stage 2: B-spline, coarse to fine ------------------------------
    let mut bspline = BSplineTransform::for_region(
        setup.fixed_vol,
        params.region.as_deref(),
        params.grid_spacing_mm,
    );
    if bspline.coeffs.is_empty() {
        bail!("the B-spline lattice is empty - the grid spacing is larger than the image");
    }
    let mut total_evals = 0usize;
    let mut final_cost = f64::MAX;

    for level in 0..levels {
        if progress.cancelled() {
            bail!("registration cancelled");
        }
        let fixed = &setup.fixed[level];
        let moving = &setup.moving[level];
        if fixed.eligible.is_empty() {
            continue;
        }
        let stride = effective_stride(fixed.eligible.len(), params.stride);
        let samples = fixed.dense_samples(stride);
        if samples.is_empty() {
            continue;
        }
        let label = format!(
            "plastimatch L{}/{} ({} samples)",
            level + 1,
            levels,
            samples.len()
        );
        progress.set(format!("{label}: preparing"));
        let base: Vec<Vec3> = samples
            .par_iter()
            .map(|&(x, _)| base_transform.map(x))
            .collect();
        let lvl = Level {
            moving,
            variance: variance(&samples),
            mi: MiScale::of(fixed, moving),
            metric: params.metric,
            grid: bspline.geometry(),
            n_coeffs: bspline.coeffs.len(),
            lambda: params.regularization.max(0.0),
            samples,
            base,
        };
        let first_step = fixed.max_spacing();
        let Some((coeffs, evals, cost)) = lbfgs(
            bspline.coeffs.clone(),
            &lvl,
            params.iterations,
            first_step,
            progress,
            &label,
        ) else {
            bail!("registration cancelled");
        };
        bspline.coeffs = coeffs;
        total_evals += evals;
        final_cost = cost;
    }

    Ok(EngineOutput {
        transform: Transform3 {
            rigid,
            warp: Warp::combined(base_transform.warp.clone(), Warp::BSpline(bspline)),
        },
        iterations: total_evals,
        final_metric: final_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_parzen_window_is_a_partition_of_unity_and_its_derivative_sums_to_zero() {
        for t0 in [0.0, 0.17, 0.5, 0.83] {
            let s: f64 = (-1..=2).map(|d| beta3(t0 - d as f64)).sum();
            assert!((s - 1.0).abs() < 1e-12, "β³ at {t0} sums to {s}");
            let ds: f64 = (-1..=2).map(|d| dbeta3(t0 - d as f64)).sum();
            assert!(ds.abs() < 1e-12, "dβ³ at {t0} sums to {ds}");
        }
        assert_eq!(beta3(2.5), 0.0);
        assert_eq!(dbeta3(-3.0), 0.0);
    }

    #[test]
    fn the_parzen_derivative_matches_a_finite_difference() {
        let h = 1e-6;
        for t in [-1.7, -0.9, -0.3, 0.25, 1.1, 1.9] {
            let fd = (beta3(t + h) - beta3(t - h)) / (2.0 * h);
            assert!(
                (dbeta3(t) - fd).abs() < 1e-6,
                "at {t}: {} vs {fd}",
                dbeta3(t)
            );
        }
    }

    #[test]
    fn the_bending_energy_vanishes_on_an_affine_field_and_its_gradient_is_exact() {
        let dims = [6usize, 6, 6];
        let n = 3 * dims[0] * dims[1] * dims[2];
        // An affine coefficient field has zero second derivative everywhere.
        let mut c = vec![0.0f64; n];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let o = 3 * (i + dims[0] * (j + dims[1] * k));
                    c[o] = 1.0 + 2.0 * i as f64 - 0.5 * j as f64;
                    c[o + 1] = 3.0 * k as f64;
                    c[o + 2] = -1.0 + j as f64;
                }
            }
        }
        let mut g = vec![0.0; n];
        let e = bending_energy(&c, dims, 4.0, 1.0, &mut g);
        assert!(e < 1e-18, "affine field has bending energy {e}");
        assert!(g.iter().all(|v| v.abs() < 1e-12));

        // A curved field: check the analytic gradient against a difference.
        let mut c = vec![0.0f64; n];
        for (idx, v) in c.iter_mut().enumerate() {
            *v = ((idx % 7) as f64 - 3.0) * 0.31;
        }
        let mut g = vec![0.0; n];
        let e0 = bending_energy(&c, dims, 4.0, 1.0, &mut g);
        assert!(e0 > 0.0);
        let h = 1e-5;
        for probe in [0usize, 137, 401, n - 5] {
            let mut cp = c.clone();
            cp[probe] += h;
            let ep = bending_energy_value(&cp, dims, 4.0, 1.0);
            let mut cm = c.clone();
            cm[probe] -= h;
            let em = bending_energy_value(&cm, dims, 4.0, 1.0);
            let fd = (ep - em) / (2.0 * h);
            assert!(
                (g[probe] - fd).abs() < 1e-6 * (1.0 + fd.abs()),
                "coefficient {probe}: analytic {} vs finite difference {fd}",
                g[probe]
            );
        }
    }

    #[test]
    fn the_dense_stride_honours_both_the_request_and_the_cap() {
        assert_eq!(effective_stride(1000, 1), 1);
        assert_eq!(effective_stride(1000, 4), 4);
        // Ten million eligible voxels must be thinned whatever was asked.
        let s = effective_stride(10_000_000, 1);
        assert!(s >= 25, "{s}");
        assert!(10_000_000 / s <= MAX_DENSE_SAMPLES);
    }
}
