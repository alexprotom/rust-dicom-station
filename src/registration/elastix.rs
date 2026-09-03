//! The elastix engine: stochastic sampling + Adaptive Stochastic Gradient
//! Descent, for a rigid Euler transform and a cubic B-spline FFD.
//!
//! This is a native re-implementation of what an elastix parameter file with
//! `Optimizer AdaptiveStochasticGradientDescent`, `ImageSampler
//! RandomCoordinate`, `NewSamplesEveryIteration true` and
//! `Metric AdvancedMeanSquares` asks for - the toolbox's own defaults. The
//! defining property is that the metric and its gradient are estimated from
//! a few thousand fresh random samples per iteration rather than from the
//! whole image: an iteration costs almost nothing, so thousands of them are
//! affordable, and the noise in the estimate is what carries the search past
//! small local minima.
//!
//! Contrast [`super::plastimatch`], which spends far more per iteration on
//! an exact gradient and needs far fewer of them.

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::*;

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

/// Run ASGD; returns (final params, iterations done) or None if cancelled.
fn asgd(
    mut params: Vec<f64>,
    eval: &GradFn,
    cfg: &AsgdConfig,
    progress: &Progress,
    label: &str,
    metric_out: &mut f64,
) -> Result<Option<(Vec<f64>, usize)>> {
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
    let mut valid0 = 0.0;
    for norm in &mut norms {
        let (m, valid) = eval(&params, &mut grad, &mut rng);
        m0 = m;
        valid0 = valid;
        *norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    }
    if valid0 < 0.25 {
        // The images barely overlap where the search starts: there is no
        // gradient to follow, and returning the start unchanged would look
        // like a result. Say so instead.
        bail!(
            "only {:.0} % of the fixed-image samples land inside the moving image at the \
             starting alignment - the two images do not overlap there. Initialise the \
             registration (centres of gravity, or a structure contoured on both) before \
             running it",
            100.0 * valid0
        );
    }
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let g0 = norms[1];
    if g0 < 1e-20 {
        *metric_out = m0;
        return Ok(Some((params, 0)));
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
            return Ok(None);
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
            // Too few samples map into the moving image - undo the step and
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
            progress.set(format!(
                "{label}: iter {}/{}  MSD {:.1}",
                it, cfg.iterations, m
            ));
        }
    }
    // Return the best parameters rather than the last ones if the last
    // iterations drifted (stochastic gradients on noisy problems).
    if best_metric < metric {
        params = best_params;
        metric = best_metric;
    }
    *metric_out = metric;
    Ok(Some((params, cfg.iterations)))
}

/// The extent the rotation scaling is derived from: the whole fixed volume,
/// or the region's bounding box for a local run. Getting this wrong on a
/// small structure makes one radian worth hundreds of millimetres and the
/// optimizer never turns it.
fn scale_extent(setup: &RegSetup) -> f64 {
    let v = setup.fixed_vol;
    let ext = match setup.params.region.as_deref() {
        None => [
            v.dims[0] as f64 * v.spacing[0],
            v.dims[1] as f64 * v.spacing[1],
            v.dims[2] as f64 * v.spacing[2],
        ],
        Some(r) => {
            let (lo, hi) = r.bbox();
            [
                (hi[0] - lo[0] + 1) as f64 * v.spacing[0],
                (hi[1] - lo[1] + 1) as f64 * v.spacing[1],
                (hi[2] - lo[2] + 1) as f64 * v.spacing[2],
            ]
        }
    };
    (ext[0] + ext[1] + ext[2]) / 3.0
}

/// Run the elastix engine: the rigid stage always, the B-spline stage for
/// [`RegMethod::ElastixBSpline`].
pub(super) fn run(setup: &RegSetup, progress: &Progress) -> Result<EngineOutput> {
    let params = setup.params;
    let levels = setup.fixed.len();
    let center = setup.center;

    // Rotation parameter scale (elastix AutomaticScalesEstimation analogue):
    // 1 rad of rotation moves a typical point by ~r mm.
    let rot_scale = 0.25 * scale_extent(setup) * 2.0; // ≈ half mean extent

    // A refinement starts from the alignment it was handed. For a rigid run
    // that means seeding the six parameters (about this run's own centre);
    // for a deformable one the alignment is already there, so the rigid
    // stage is skipped altogether and only the correction is recovered.
    //
    // A *local* deformable run skips it for a different reason: a rigid body
    // fitted to one structure would be applied to the whole volume, moving
    // anatomy nobody asked about. Confined to the lattice, the correction
    // stays where the structure is - which is what "local" has to mean.
    let start = params.start.as_deref();
    let refining =
        params.method == RegMethod::ElastixBSpline && (start.is_some() || params.region.is_some());

    // ---------------- Rigid stage ----------------
    let mut rigid = match start {
        Some(s) => s.rigid.recentered(center),
        None => setup.init.clone(),
    };
    let mut total_iters = 0usize;
    let mut last_metric = f64::MAX;

    for level in (0..levels).take(if refining { 0 } else { levels }) {
        if progress.cancelled() {
            bail!("registration cancelled");
        }
        let fixed = &setup.fixed[level];
        let moving = &setup.moving[level];
        if fixed.eligible.is_empty() {
            continue;
        }
        let delta = fixed.max_spacing();
        let n_samples = params.samples;
        let label = format!("Rigid L{}/{}", level + 1, levels);

        let center_l = center;
        let eval = move |p: &[f64], grad: &mut [f64], rng: &mut XorShift| -> (f64, f64) {
            let tr = RigidTransform::new(
                [
                    p[0] / rot_scale,
                    p[1] / rot_scale,
                    p[2] / rot_scale,
                    p[3],
                    p[4],
                    p[5],
                ],
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
            &AsgdConfig {
                iterations: params.iterations,
                big_a: 20.0,
                delta,
            },
            progress,
            &label,
            &mut mlast,
        )?
        else {
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

    // What the B-spline correction is measured on top of.
    let base = match start {
        Some(s) if refining => s.clone(),
        _ => Transform3::rigid_only(rigid.clone()),
    };
    let mut transform = Transform3::rigid_only(rigid.clone());

    // ---------------- B-spline stage (deformable only) ----------------
    if params.method == RegMethod::ElastixBSpline {
        let mut bspline = BSplineTransform::for_region(
            setup.fixed_vol,
            params.region.as_deref(),
            params.grid_spacing_mm,
        );
        let n_coeffs = bspline.coeffs.len();
        let [gnx, gny, _] = bspline.grid_dims;

        for level in 0..levels {
            if progress.cancelled() {
                bail!("registration cancelled");
            }
            let fixed = &setup.fixed[level];
            let moving = &setup.moving[level];
            if fixed.eligible.is_empty() {
                continue;
            }
            let delta = 0.5 * fixed.max_spacing();
            let n_samples = params.samples;
            let label = format!("B-spline L{}/{}", level + 1, levels);
            let base_l = base.clone();
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
                let chunk = samples
                    .len()
                    .div_ceil(rayon::current_num_threads().max(1))
                    .max(1);
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
                                let Some((mval, mg)) = moving.sample_grad(base_l.map(x) + disp)
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
                &AsgdConfig {
                    iterations: params.iterations,
                    big_a: 20.0,
                    delta,
                },
                progress,
                &label,
                &mut mlast,
            )?
            else {
                bail!("registration cancelled");
            };
            bspline.coeffs = coeffs;
            total_iters += iters;
            last_metric = mlast;
        }
        transform.rigid = base.rigid.clone();
        transform.warp = Warp::combined(base.warp.clone(), Warp::BSpline(bspline));
    }

    Ok(EngineOutput {
        transform,
        iterations: total_iters,
        final_metric: last_metric,
    })
}
