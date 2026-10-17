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

/// Samples per piece of the rigid gradient sum. An iteration draws a few
/// thousand samples, so a fixed piece this short still spreads them over
/// every core; being fixed, it keeps the sum's order (and so the result)
/// the same on every machine (see `crate::par`).
const RIGID_PIECE: usize = 256;

/// Samples per piece of the B-spline gradient scatter (at most 16 pieces).
const BSPLINE_PIECE: usize = 512;

/// Where one piece of the B-spline gradient sum puts its contributions.
trait Scatter {
    fn add(&mut self, o: usize, v: f64);
}

/// The dense accumulator: one running sum per coefficient.
impl Scatter for Vec<f64> {
    #[inline]
    fn add(&mut self, o: usize, v: f64) {
        self[o] += v;
    }
}

/// The sparse accumulator: running sums in a dense scratch that is zero
/// between pieces, and the list of the coefficients a piece touched. A
/// piece's few thousand samples reach a few hundred thousand coefficient
/// slots at most; the scratch comes from a [`ScratchPool`] that lives as
/// long as one level, not zeroed whole on every iteration and added up
/// whole per piece.
struct Touched<'a> {
    sum: &'a mut [f64],
    mark: &'a mut [bool],
    list: Vec<u32>,
}

impl Scatter for Touched<'_> {
    #[inline]
    fn add(&mut self, o: usize, v: f64) {
        if !self.mark[o] {
            self.mark[o] = true;
            self.list.push(o as u32);
        }
        self.sum[o] += v;
    }
}

impl Touched<'_> {
    /// The touched coefficients with their sums, the scratch left zero.
    fn drain(self) -> Vec<(u32, f64)> {
        let Touched { sum, mark, list } = self;
        list.into_iter()
            .map(|o| {
                let o_ = o as usize;
                let v = std::mem::take(&mut sum[o_]);
                mark[o_] = false;
                (o, v)
            })
            .collect()
    }
}

/// The [`Touched`] scratches of one level: a piece takes one (or makes one
/// when all are in use), leaves it zero and gives it back, so there are
/// never more than pieces running at once, and they go with the level
/// instead of staying with the threads for the life of the program. A
/// piece that panics takes its scratch with it rather than returning one
/// left dirty.
#[derive(Default)]
pub(crate) struct ScratchPool(std::sync::Mutex<Vec<(Vec<f64>, Vec<bool>)>>);

impl ScratchPool {
    fn take(&self, n: usize) -> (Vec<f64>, Vec<bool>) {
        let got = self.0.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let (mut sum, mut mark) = got.unwrap_or_default();
        if sum.len() < n {
            sum.resize(n, 0.0);
            mark.resize(n, false);
        }
        (sum, mark)
    }

    fn give(&self, scratch: (Vec<f64>, Vec<bool>)) {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(scratch);
    }
}

/// One piece of the B-spline stage's cost: for the samples of `part`, the
/// squared differences summed and counted, and every coefficient's share
/// of the gradient handed to `out` in sample order.
#[allow(clippy::too_many_arguments)]
fn bspline_piece(
    part: &[(Vec3, f32)],
    grid: &BSplineTransform,
    p: &[f64],
    base: &Transform3,
    moving: &RegImage,
    out: &mut impl Scatter,
) -> (f64, usize) {
    let [gnx, gny, _] = grid.grid_dims;
    let (mut ssd, mut count) = (0.0f64, 0usize);
    for &(x, fval) in part {
        let Some((corner, w)) = grid.support(x) else {
            continue;
        };
        let mut disp = Vec3::ZERO;
        let mut touched = [(0usize, 0.0f64); 64];
        let mut ti = 0;
        for kz in 0..4 {
            let iz = (corner[2] + kz as i64) as usize;
            for (ky, wy) in w[1].iter().enumerate() {
                let iy = (corner[1] + ky as i64) as usize;
                let wyz = wy * w[2][kz];
                let row = 3 * (corner[0] as usize + gnx * (iy + gny * iz));
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
        let Some((mval, mg)) = moving.sample_grad(base.map(x) + disp) else {
            continue;
        };
        let diff = (mval - fval) as f64;
        for &(o, wt) in &touched[..ti] {
            let c = 2.0 * diff * wt;
            out.add(o, c * mg.x);
            out.add(o + 1, c * mg.y);
            out.add(o + 2, c * mg.z);
        }
        ssd += diff * diff;
        count += 1;
    }
    (ssd, count)
}

/// One piece's result: the coefficients it touched with their sums, its
/// summed squared difference and its sample count.
type Piece = (Vec<(u32, f64)>, f64, usize);

/// The B-spline stage's estimate from `samples`: the pieces (their number
/// set by the sample count alone, never the thread count) summed in order
/// into `grad`, each coefficient's contributions added in sample order
/// within a piece and in piece order across them, so the step is the same
/// on every machine. Returns the summed squared difference and the count.
fn bspline_estimate(
    samples: &[(Vec3, f32)],
    grid: &BSplineTransform,
    p: &[f64],
    base: &Transform3,
    moving: &RegImage,
    grad: &mut [f64],
    pool: &ScratchPool,
) -> (f64, usize) {
    let n_coeffs = grad.len();
    let parts = samples.len().div_ceil(BSPLINE_PIECE).clamp(1, 16);
    let chunk = samples.len().div_ceil(parts).max(1);
    let pieces: Vec<Piece> = samples
        .par_chunks(chunk)
        .map(|part| {
            let (mut sum, mut mark) = pool.take(n_coeffs);
            let mut out = Touched {
                sum: &mut sum[..n_coeffs],
                mark: &mut mark[..n_coeffs],
                list: Vec::new(),
            };
            let (ssd, count) = bspline_piece(part, grid, p, base, moving, &mut out);
            let touched = out.drain();
            pool.give((sum, mark));
            (touched, ssd, count)
        })
        .collect();
    // A coefficient a piece did not touch would have added +0.0 there,
    // which changes no sum (none is ever -0.0: every sum starts at +0.0),
    // so adding only what each piece touched is the dense sum to the bit.
    grad.fill(0.0);
    let (mut ssd, mut count) = (0.0f64, 0usize);
    for (values, s, c) in &pieces {
        for &(o, v) in values {
            grad[o as usize] += v;
        }
        ssd += s;
        count += c;
    }
    (ssd, count)
}

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
            // Summed in fixed pieces and in order (`crate::par`), so the
            // step the optimizer takes does not depend on the thread count.
            let (g, sum, cnt) = crate::par::ordered_fold_by(
                &samples,
                RIGID_PIECE,
                |part, _| {
                    let mut acc = ([0.0f64; 6], 0.0f64, 0usize);
                    for &(x, fval) in part {
                        let Some((mval, mg)) = moving.sample_grad(tr.map(x)) else {
                            continue;
                        };
                        let diff = (mval - fval) as f64;
                        let jac = tr.jacobian(x);
                        let mut gl = [0.0f64; 6];
                        for (pi, j) in jac.iter().enumerate() {
                            gl[pi] = 2.0 * diff * mg.dot(*j);
                        }
                        // chain rule into scaled space
                        gl[0] /= rot_scale;
                        gl[1] /= rot_scale;
                        gl[2] /= rot_scale;
                        for (a, b) in acc.0.iter_mut().zip(gl) {
                            *a += b;
                        }
                        acc.1 += diff * diff;
                        acc.2 += 1;
                    }
                    acc
                },
                |a, b| {
                    let mut g = a.0;
                    for (x, y) in g.iter_mut().zip(b.0) {
                        *x += y;
                    }
                    (g, a.1 + b.1, a.2 + b.2)
                },
                ([0.0; 6], 0.0, 0),
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
            let pool = ScratchPool::default();

            let eval = move |p: &[f64], grad: &mut [f64], rng: &mut XorShift| -> (f64, f64) {
                let samples = draw_samples(fixed, n_samples, rng);
                let n_total = samples.len().max(1);
                let (sum, cnt) = bspline_estimate(&samples, &grid, p, &base_l, moving, grad, &pool);
                let cntf = cnt.max(1) as f64;
                for g in grad.iter_mut() {
                    *g /= cntf;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(shift: f64) -> Volume {
        let dims = [26, 22, 18];
        let mut data = vec![0i16; dims[0] * dims[1] * dims[2]];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let (x, y, z) = (i as f64 + shift, j as f64, k as f64);
                    let v = 300.0 * ((x * 0.37).sin() + (y * 0.29 + z * 0.11).cos())
                        + 40.0 * ((x * y * 0.05).sin());
                    data[(k * dims[1] + j) * dims[0] + i] = v as i16;
                }
            }
        }
        Volume {
            data,
            dims,
            spacing: [2.0, 2.2, 3.0],
            origin: Vec3::new(-20.0, -15.0, 4.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: -400,
            max_value: 400,
        }
    }

    /// The gradient scattered only into the coefficients each piece
    /// touched is the dense accumulators' sum, bit for bit, for one piece
    /// and for many.
    #[test]
    fn the_touched_scatter_sums_to_the_dense_one_bit_for_bit() {
        let fixed_vol = volume(0.0);
        let moving_vol = volume(1.3);
        let mut fixed = RegImage::from_volume(&fixed_vol);
        fixed.prepare_sampling(f32::MIN, None);
        let moving = RegImage::from_volume(&moving_vol);
        let grid = BSplineTransform::for_region(&fixed_vol, None, 9.0);
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let p: Vec<f64> = grid
            .coeffs
            .iter()
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x % 2001) as f64 - 1000.0) * 1e-3
            })
            .collect();
        let geom = grid.geometry();
        let base = Transform3::rigid_only(RigidTransform::new(
            [0.01, 0.02, -0.01, 0.5, -0.3, 0.2],
            Vec3::new(5.0, 7.0, 30.0),
        ));
        let mut rng = XorShift(12345);
        for n in [300, 3000, 9000] {
            let samples = draw_samples(&fixed, n, &mut rng);
            let mut grad = vec![0.0; p.len()];
            let (ssd, count) = bspline_estimate(
                &samples,
                &geom,
                &p,
                &base,
                &moving,
                &mut grad,
                &ScratchPool::default(),
            );
            // The dense way: an accumulator over every coefficient per
            // piece, added whole in piece order.
            let parts = samples.len().div_ceil(BSPLINE_PIECE).clamp(1, 16);
            let chunk = samples.len().div_ceil(parts).max(1);
            let mut dense = vec![0.0f64; p.len()];
            let (mut ssd2, mut count2) = (0.0f64, 0usize);
            for part in samples.chunks(chunk) {
                let mut acc = vec![0.0f64; p.len()];
                let (s, c) = bspline_piece(part, &geom, &p, &base, &moving, &mut acc);
                for (d, a) in dense.iter_mut().zip(&acc) {
                    *d += a;
                }
                ssd2 += s;
                count2 += c;
            }
            assert!(count > n / 4, "{count} of {n} samples landed");
            assert_eq!((ssd.to_bits(), count), (ssd2.to_bits(), count2));
            for (i, (a, b)) in grad.iter().zip(&dense).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "coefficient {i}: {a} vs {b}");
            }
            assert!(grad.iter().any(|g| *g != 0.0));
        }
    }
}
