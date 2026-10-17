//! Derivative-free optimizers for small, bounded parameter spaces - the six
//! parameters of a rigid pose - and an L-BFGS for when a gradient is at hand.
//!
//! The 2D/3D registration's cost is a similarity between a radiograph and a
//! rendering: each evaluation is a projection of the CT, and an analytic
//! gradient through the projector and the measure would be the most
//! complicated part of the engine for no better answer. The field's
//! systems (Penney 1998; van der Bom 2011; ITK's `PowellOptimizer`) use a
//! derivative-free search instead, and so does this one.
//!
//! * [`powell`]: Powell's conjugate directions (Powell 1964), each line
//!   minimised by Brent's method (bracketing by golden-section expansion,
//!   then parabolic interpolation with golden-section fall-back, *Numerical
//!   Recipes* 10.3 and 10.7). Directions start as the axes; after a sweep the
//!   direction of the whole sweep replaces the one along which the cost fell
//!   most, unless that would make the set degenerate.
//! * [`nelder_mead`]: the downhill simplex with the standard coefficients
//!   (reflection 1, expansion 2, contraction ½, shrink ½) and one restart
//!   from the best point, for a noisy cost.
//! * [`lbfgs`]: limited-memory BFGS (history 6) with an Armijo backtracking
//!   line search, for a cost whose gradient is supplied (by finite
//!   differences, in the 2D/3D engine's final refinement).
//!
//! The parameters are taken as given: the caller scales them so that one
//! unit of each moves the anatomy by about the same amount (millimetres
//! for a pose), and bounds are boxes in those units. Every one of them is
//! deterministic - no random restarts, no parallel evaluation in an order
//! that could change.

/// A cost to minimise.
pub trait Cost {
    fn eval(&mut self, x: &[f64]) -> f64;
}

impl<F: FnMut(&[f64]) -> f64> Cost for F {
    fn eval(&mut self, x: &[f64]) -> f64 {
        self(x)
    }
}

/// Where a run ended, and what it cost.
#[derive(Clone, Debug)]
pub struct Optimum {
    pub x: Vec<f64>,
    pub value: f64,
    pub evaluations: usize,
    /// Sweeps (Powell), iterations (Nelder-Mead, L-BFGS).
    pub iterations: usize,
}

/// Limits of a run.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Evaluations of the cost, at most.
    pub max_evaluations: usize,
    /// A Powell sweep that improves the cost by less than this, relatively,
    /// and moves no parameter by more than `xtol` ends the run; a simplex
    /// ends when either holds.
    pub ftol: f64,
    pub xtol: f64,
    /// The first step of a line search or the size of the first simplex.
    pub initial_step: f64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_evaluations: 2000,
            ftol: 1e-5,
            xtol: 0.05,
            initial_step: 2.0,
        }
    }
}

/// Bounds of a box: per parameter `(lo, hi)`.
pub type Bounds = Vec<(f64, f64)>;

fn clamp_into(x: &mut [f64], bounds: &Bounds) {
    for (v, (lo, hi)) in x.iter_mut().zip(bounds) {
        *v = v.clamp(*lo, *hi);
    }
}

/// Counts evaluations and stops answering past the budget.
struct Counted<'a, C: Cost + ?Sized> {
    cost: &'a mut C,
    n: usize,
    max: usize,
    best: f64,
    best_x: Vec<f64>,
}

impl<C: Cost + ?Sized> Counted<'_, C> {
    fn eval(&mut self, x: &[f64]) -> f64 {
        self.n += 1;
        let f = self.cost.eval(x);
        let f = if f.is_finite() { f } else { f64::INFINITY };
        if f < self.best {
            self.best = f;
            self.best_x = x.to_vec();
        }
        f
    }

    fn exhausted(&self) -> bool {
        self.n >= self.max
    }
}

const GOLD: f64 = std::f64::consts::GOLDEN_RATIO;
const CGOLD: f64 = 0.381_966_011_250_105;

/// The range of `λ` keeping `x + λ d` inside the box.
fn line_range(x: &[f64], d: &[f64], bounds: &Bounds) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::NEG_INFINITY, f64::INFINITY);
    for ((xi, di), (b0, b1)) in x.iter().zip(d).zip(bounds) {
        if di.abs() < 1e-15 {
            continue;
        }
        let (a, b) = ((b0 - xi) / di, (b1 - xi) / di);
        let (a, b) = if a < b { (a, b) } else { (b, a) };
        lo = lo.max(a);
        hi = hi.min(b);
    }
    (lo.min(0.0), hi.max(0.0))
}

/// Minimise `f(x + λ d)` over `λ` in the box: bracket from `step`, then
/// Brent. Returns `(λ, f)`; `f0` is the value at `λ = 0`.
fn line_minimise<C: Cost + ?Sized>(
    c: &mut Counted<'_, C>,
    x: &[f64],
    d: &[f64],
    f0: f64,
    step: f64,
    tol: f64,
    bounds: &Bounds,
) -> (f64, f64) {
    let (lmin, lmax) = line_range(x, d, bounds);
    if lmax - lmin < 1e-12 {
        return (0.0, f0);
    }
    let point = |l: f64| -> Vec<f64> { x.iter().zip(d).map(|(a, b)| a + l * b).collect() };
    let at = |c: &mut Counted<'_, C>, l: f64| c.eval(&point(l.clamp(lmin, lmax)));

    // ---- bracket: a < b < c with f(b) below both ends ----
    let (mut a, mut fa) = (0.0f64, f0);
    let mut b = step.clamp(lmin, lmax);
    if b == 0.0 {
        b = if lmax > 0.0 {
            lmax.min(step.abs())
        } else {
            lmin
        };
    }
    let mut fb = at(c, b);
    if fb > fa {
        // Downhill is the other way.
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    let mut cc = (b + GOLD * (b - a)).clamp(lmin, lmax);
    let mut fc = if cc == b { fb } else { at(c, cc) };
    let mut guard = 0;
    while fc < fb && guard < 40 && !c.exhausted() {
        // Still going down: step further out, to the bound at most.
        let next = (cc + GOLD * (cc - b)).clamp(lmin, lmax);
        a = b;
        fa = fb;
        b = cc;
        fb = fc;
        if next == cc {
            // At the bound: the minimum is there.
            return (b, fb);
        }
        cc = next;
        fc = at(c, cc);
        guard += 1;
    }
    let _ = fa;
    let (mut lo, mut hi) = if a < cc { (a, cc) } else { (cc, a) };

    // ---- Brent on [lo, hi] with b inside ----
    let (mut xm, mut w, mut v) = (b, b, b);
    let (mut fx, mut fw, mut fv) = (fb, fb, fb);
    let mut e = 0.0f64;
    let mut dd = 0.0f64;
    for _ in 0..60 {
        if c.exhausted() {
            break;
        }
        let mid = 0.5 * (lo + hi);
        let tol1 = tol * 0.5 + 1e-10;
        let tol2 = 2.0 * tol1;
        if (xm - mid).abs() <= tol2 - 0.5 * (hi - lo) {
            break;
        }
        let mut golden = true;
        if e.abs() > tol1 {
            // Parabola through x, w, v.
            let r = (xm - w) * (fx - fv);
            let mut q = (xm - v) * (fx - fw);
            let mut p = (xm - v) * q - (xm - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = dd;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (lo - xm) && p < q * (hi - xm) {
                dd = p / q;
                let u = xm + dd;
                if u - lo < tol2 || hi - u < tol2 {
                    dd = tol1.copysign(mid - xm);
                }
                golden = false;
            }
        }
        if golden {
            e = if xm >= mid { lo - xm } else { hi - xm };
            dd = CGOLD * e;
        }
        let u = if dd.abs() >= tol1 {
            xm + dd
        } else {
            xm + tol1.copysign(dd)
        };
        let fu = at(c, u);
        if fu <= fx {
            if u >= xm {
                lo = xm;
            } else {
                hi = xm;
            }
            v = w;
            fv = fw;
            w = xm;
            fw = fx;
            xm = u;
            fx = fu;
        } else {
            if u < xm {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || w == xm {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == xm || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    if fx < f0 {
        (xm.clamp(lmin, lmax), fx)
    } else {
        (0.0, f0)
    }
}

/// Powell's method from `x0` inside `bounds`.
pub fn powell<C: Cost + ?Sized>(cost: &mut C, x0: &[f64], bounds: &Bounds, lim: Limits) -> Optimum {
    let n = x0.len();
    let mut x = x0.to_vec();
    clamp_into(&mut x, bounds);
    let mut c = Counted {
        cost,
        n: 0,
        max: lim.max_evaluations,
        best: f64::INFINITY,
        best_x: x.clone(),
    };
    let mut f = c.eval(&x);
    let mut dirs: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let tol = (lim.xtol * 0.5).max(1e-6);
    let mut sweeps = 0;
    while !c.exhausted() && sweeps < 200 {
        sweeps += 1;
        let start = x.clone();
        let f_start = f;
        let mut biggest = (0usize, 0.0f64);
        for (i, d) in dirs.iter().enumerate() {
            if c.exhausted() {
                break;
            }
            let (l, fl) = line_minimise(&mut c, &x, d, f, lim.initial_step, tol, bounds);
            if fl < f {
                for (xi, di) in x.iter_mut().zip(d) {
                    *xi += l * di;
                }
                clamp_into(&mut x, bounds);
                if f - fl > biggest.1 {
                    biggest = (i, f - fl);
                }
                f = fl;
            }
        }
        let moved = x
            .iter()
            .zip(&start)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        let decrease = f_start - f;
        // Converged when a sweep neither improves the cost nor moves the
        // point by more than the tolerances; either alone can be a sweep
        // along a flat valley floor, or a sharp step that did not move far.
        let flat = 2.0 * decrease <= lim.ftol * (f_start.abs() + f.abs()) + 1e-14;
        if (flat && moved < lim.xtol) || decrease <= 0.0 {
            break;
        }
        // The sweep's own direction, tried once and kept if it helps.
        let dir: Vec<f64> = x.iter().zip(&start).map(|(a, b)| a - b).collect();
        let extrap: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a + d).collect();
        let mut ex = extrap.clone();
        clamp_into(&mut ex, bounds);
        let fe = c.eval(&ex);
        if fe < f_start {
            let t = 2.0 * (f_start - 2.0 * f + fe) * (f_start - f - biggest.1).powi(2)
                - biggest.1 * (f_start - fe).powi(2);
            if t < 0.0 {
                let (l, fl) = line_minimise(&mut c, &x, &dir, f, 1.0, tol, bounds);
                if fl < f {
                    for (xi, di) in x.iter_mut().zip(&dir) {
                        *xi += l * di;
                    }
                    clamp_into(&mut x, bounds);
                    f = fl;
                }
                let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    let last = dirs.len() - 1;
                    dirs[biggest.0] = dirs[last].clone();
                    dirs[last] = dir.iter().map(|v| v / norm).collect();
                }
            }
        }
    }
    // The best point seen is the answer, wherever the bookkeeping is.
    if c.best < f {
        x = c.best_x.clone();
        f = c.best;
    }
    Optimum {
        x,
        value: f,
        evaluations: c.n,
        iterations: sweeps,
    }
}

/// The Nelder-Mead simplex from `x0` inside `bounds`, restarted once from
/// its answer.
pub fn nelder_mead<C: Cost + ?Sized>(
    cost: &mut C,
    x0: &[f64],
    bounds: &Bounds,
    lim: Limits,
) -> Optimum {
    let n = x0.len();
    let mut c = Counted {
        cost,
        n: 0,
        max: lim.max_evaluations,
        best: f64::INFINITY,
        best_x: x0.to_vec(),
    };
    let mut start = x0.to_vec();
    clamp_into(&mut start, bounds);
    let mut iterations = 0;
    for _round in 0..2 {
        let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
        let f0 = c.eval(&start);
        simplex.push((start.clone(), f0));
        for i in 0..n {
            let mut p = start.clone();
            p[i] += lim.initial_step;
            if p[i] > bounds[i].1 {
                p[i] = start[i] - lim.initial_step;
            }
            clamp_into(&mut p, bounds);
            let f = c.eval(&p);
            simplex.push((p, f));
        }
        while !c.exhausted() {
            iterations += 1;
            simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
            let (best, worst) = (simplex[0].1, simplex[n].1);
            let size = simplex[1..]
                .iter()
                .map(|(p, _)| {
                    p.iter()
                        .zip(&simplex[0].0)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f64, f64::max)
                })
                .fold(0.0f64, f64::max);
            if size < lim.xtol
                || 2.0 * (worst - best) <= lim.ftol * (best.abs() + worst.abs()) + 1e-14
            {
                break;
            }
            let centroid: Vec<f64> = (0..n)
                .map(|j| simplex[..n].iter().map(|(p, _)| p[j]).sum::<f64>() / n as f64)
                .collect();
            let along = |t: f64| -> Vec<f64> {
                let mut p: Vec<f64> = centroid
                    .iter()
                    .zip(&simplex[n].0)
                    .map(|(c0, w)| c0 + t * (c0 - w))
                    .collect();
                clamp_into(&mut p, bounds);
                p
            };
            let xr = along(1.0);
            let fr = c.eval(&xr);
            if fr < simplex[0].1 {
                let xe = along(2.0);
                let fe = c.eval(&xe);
                simplex[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
            } else if fr < simplex[n - 1].1 {
                simplex[n] = (xr, fr);
            } else {
                let (xc, fc) = if fr < simplex[n].1 {
                    let p = along(0.5);
                    let f = c.eval(&p);
                    (p, f)
                } else {
                    let p = along(-0.5);
                    let f = c.eval(&p);
                    (p, f)
                };
                if fc < simplex[n].1.min(fr) {
                    simplex[n] = (xc, fc);
                } else {
                    // Shrink towards the best.
                    let b = simplex[0].0.clone();
                    for s in simplex.iter_mut().skip(1) {
                        let p: Vec<f64> =
                            s.0.iter()
                                .zip(&b)
                                .map(|(a, bb)| bb + 0.5 * (a - bb))
                                .collect();
                        let f = c.eval(&p);
                        *s = (p, f);
                    }
                }
            }
        }
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        start = simplex[0].0.clone();
    }
    Optimum {
        x: c.best_x.clone(),
        value: c.best,
        evaluations: c.n,
        iterations,
    }
}

/// L-BFGS with an Armijo backtracking line search, inside `bounds`
/// (a step is clamped into the box). `fg` returns the value and writes the
/// gradient.
pub fn lbfgs(
    mut fg: impl FnMut(&[f64], &mut [f64]) -> f64,
    x0: &[f64],
    bounds: &Bounds,
    max_iterations: usize,
    first_step: f64,
) -> Optimum {
    const HISTORY: usize = 6;
    let n = x0.len();
    let mut x = x0.to_vec();
    clamp_into(&mut x, bounds);
    let mut g = vec![0.0; n];
    let mut f = fg(&x, &mut g);
    let mut evals = 1;
    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut iterations = 0;
    for _ in 0..max_iterations {
        iterations += 1;
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !gnorm.is_finite() || gnorm < 1e-12 {
            break;
        }
        // Two-loop recursion.
        let mut q = g.clone();
        let mut alpha = vec![0.0; s_hist.len()];
        for i in (0..s_hist.len()).rev() {
            let rho = 1.0 / dot(&y_hist[i], &s_hist[i]);
            alpha[i] = rho * dot(&s_hist[i], &q);
            axpy(-alpha[i], &y_hist[i], &mut q);
        }
        let gamma = match (s_hist.last(), y_hist.last()) {
            (Some(s), Some(y)) => dot(s, y) / dot(y, y),
            _ => first_step / gnorm,
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..s_hist.len() {
            let rho = 1.0 / dot(&y_hist[i], &s_hist[i]);
            let beta = rho * dot(&y_hist[i], &q);
            axpy(alpha[i] - beta, &s_hist[i], &mut q);
        }
        let d: Vec<f64> = q.iter().map(|v| -v).collect();
        let slope = dot(&g, &d);
        let d = if slope >= 0.0 {
            g.iter().map(|v| -v * first_step / gnorm).collect()
        } else {
            d
        };
        let slope = dot(&g, &d);
        let mut t = 1.0;
        let mut accepted = None;
        for _ in 0..20 {
            let mut xn: Vec<f64> = x.iter().zip(&d).map(|(a, b)| a + t * b).collect();
            clamp_into(&mut xn, bounds);
            let mut gn = vec![0.0; n];
            let fnew = fg(&xn, &mut gn);
            evals += 1;
            if fnew <= f + 1e-4 * t * slope {
                accepted = Some((xn, fnew, gn));
                break;
            }
            t *= 0.5;
        }
        let Some((xn, fnew, gn)) = accepted else {
            break;
        };
        let s: Vec<f64> = xn.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = gn.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        let improved = f - fnew;
        x = xn;
        f = fnew;
        g = gn;
        if sy > 1e-12 {
            s_hist.push(s);
            y_hist.push(y);
            if s_hist.len() > HISTORY {
                s_hist.remove(0);
                y_hist.remove(0);
            }
        }
        if improved.abs() <= 1e-10 * (f.abs() + 1e-12) {
            break;
        }
    }
    Optimum {
        x,
        value: f,
        evaluations: evals,
        iterations,
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    /// A quadratic bowl whose axes are not the coordinate axes, centred off
    /// the origin: what a pose cost looks like near its minimum.
    fn tilted(x: &[f64]) -> f64 {
        let c = [1.5, -2.0, 0.7, 3.0, -1.0, 0.25];
        let mut s = 0.0;
        for i in 0..6 {
            let d = x[i] - c[i];
            s += (i + 1) as f64 * d * d;
            if i + 1 < 6 {
                s += 0.8 * d * (x[i + 1] - c[i + 1]);
            }
        }
        s
    }

    fn wide(n: usize) -> Bounds {
        vec![(-10.0, 10.0); n]
    }

    #[test]
    fn powell_finds_the_minimum_of_a_tilted_bowl() {
        let lim = Limits {
            xtol: 1e-7,
            ftol: 1e-14,
            initial_step: 1.0,
            max_evaluations: 20000,
        };
        let r = powell(&mut tilted, &[0.0; 6], &wide(6), lim);
        let c = [1.5, -2.0, 0.7, 3.0, -1.0, 0.25];
        for (a, b) in r.x.iter().zip(c) {
            assert!((a - b).abs() < 1e-5, "{:?}", r.x);
        }
        // And the same answer again: no randomness anywhere.
        let again = powell(&mut tilted, &[0.0; 6], &wide(6), lim);
        assert_eq!(r.x, again.x);
        assert_eq!(r.evaluations, again.evaluations);
    }

    #[test]
    fn powell_walks_the_rosenbrock_valley() {
        let lim = Limits {
            xtol: 1e-9,
            ftol: 1e-16,
            initial_step: 0.5,
            max_evaluations: 50000,
        };
        let r = powell(&mut rosenbrock, &[-1.2, 1.0], &wide(2), lim);
        assert!(
            (r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] - 1.0).abs() < 1e-4,
            "{:?}",
            r.x
        );
    }

    #[test]
    fn the_bounds_hold() {
        // The minimum is outside the box: the answer is on its edge.
        let mut f = |x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] + 5.0).powi(2);
        let b = vec![(-1.0, 2.0), (-3.0, 3.0)];
        let r = powell(&mut f, &[0.0, 0.0], &b, Limits::default());
        assert!(
            (r.x[0] - 2.0).abs() < 1e-3 && (r.x[1] + 3.0).abs() < 1e-3,
            "{:?}",
            r.x
        );
        let r = nelder_mead(&mut f, &[0.0, 0.0], &b, Limits::default());
        assert!(r.x[0] <= 2.0 + 1e-12 && r.x[1] >= -3.0 - 1e-12);
        assert!(
            (r.x[0] - 2.0).abs() < 0.05 && (r.x[1] + 3.0).abs() < 0.05,
            "{:?}",
            r.x
        );
    }

    #[test]
    fn nelder_mead_finds_the_minimum_of_a_tilted_bowl() {
        let lim = Limits {
            xtol: 1e-6,
            ftol: 1e-14,
            initial_step: 1.0,
            max_evaluations: 50000,
        };
        let r = nelder_mead(&mut tilted, &[0.0; 6], &wide(6), lim);
        let c = [1.5, -2.0, 0.7, 3.0, -1.0, 0.25];
        for (a, b) in r.x.iter().zip(c) {
            assert!((a - b).abs() < 1e-3, "{:?}", r.x);
        }
    }

    #[test]
    fn lbfgs_follows_a_gradient() {
        let fg = |x: &[f64], g: &mut [f64]| {
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
            rosenbrock(x)
        };
        let r = lbfgs(fg, &[-1.2, 1.0], &wide(2), 500, 0.1);
        assert!(
            (r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] - 1.0).abs() < 1e-4,
            "{:?}",
            r
        );
    }
}
