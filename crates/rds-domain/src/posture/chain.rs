//! Chains of states: the rigid bodies on curves through every state, so
//! their velocity does not jump where one segment hands over to the next.
//!
//! A segment on its own is a stationary flow: each body turns on its screw
//! at a constant rate, `h(s) = exp(s xi)`. Two segments of a chain have two
//! rates, and at the state between them the body stops one motion and
//! starts the other - a kink in every curve the tables draw, and a jolt in
//! the animation. Here each body's path within segment `i` is instead a
//! cubic curve in SE(3) that starts and ends where the stationary one does
//! and leaves and arrives with chosen velocities: the cumulative cubic
//! Hermite form of Kim, Kim and Shin (*A general construction scheme for
//! unit quaternion curves with simple high order derivatives*, SIGGRAPH
//! 1995), written for rigid motions acting on the left,
//!
//! ```text
//! h(s) = exp(b3(s) w_c) exp(b2(s) w_b) exp(b1(s) w_a)
//! b1 = 1 - (1 - s)^3,   b2 = 3 s^2 - 2 s^3,   b3 = s^3
//! w_a = m_i / 3,   w_c = m_{i+1} / 3,   exp(w_b) = exp(-w_c) g exp(-w_a)
//! ```
//!
//! with `g` the segment's fitted motion and `m_k` the body's spatial
//! velocity (a twist in the anchor frame) at state `k`. `h(0)` is the
//! identity, `h(1) = g` exactly, and the spatial velocity `h' h^-1` is `m_i`
//! at the start and `m_{i+1}` at the end, exactly (`b1'(0) = 3`,
//! `b3'(1) = 3`, every other derivative zero there). The velocities at the
//! states are Catmull-Rom's: the mean of the two neighbouring segments'
//! twists inside the chain, the one segment's own at its two ends. When the
//! two neighbours agree, `w_a = w_b = w_c = xi / 3` and the curve is the
//! stationary screw itself: a chain of equal steps is unchanged.
//!
//! **The field between the bodies.** At a time `s` of segment `i` the
//! deformation is the flow, for unit time, of the velocity field
//!
//! ```text
//! v(s) = s v_i + sum_b w_b (eta_b(s) - s xi_b),     eta_b(s) = log h_b(s)
//! ```
//!
//! the segment's own poly-rigid field scaled to `s`, corrected body by body
//! by how far each body's curve is from its screw, with the blend's weights.
//! At `s = 0` it is zero, at `s = 1` it is the segment's field exactly (the
//! correction vanishes), and where a body's weight is one it is that body's
//! twist `eta_b(s)`, whose flow is `h_b(s)` - the exact regions of
//! [`super::svf::Svf::exp`] keep it rigid there. So the path is invertible
//! at every time (the flow of a smooth field) and each body's velocity is
//! continuous across a state exactly; between the bodies it is continuous
//! to first order. The weights are kept per node for the four bodies that
//! weigh most there ([`SparseWeights`]), which is the blend to well under a
//! per cent everywhere the correction is not zero. The residual layer stays
//! stationary in each segment: it is the part of the change the bodies do
//! not explain, measured once per pair of scans, and has no velocity of its
//! own to match.
//!
//! **Not here: a statistical model over many states.** With dozens of
//! states of one patient (daily scans over a course) a principal component
//! analysis of the per-state logarithms would give the main modes of
//! posture change and a mean posture. Two to a handful of states, which is
//! what one patient's imaging gives, are too few to estimate any mode, and
//! modes across patients need every patient brought into one anatomy first
//! (an atlas registration the station does not make). That is future work.

use std::sync::Arc;

use rayon::prelude::*;

use super::polyrigid::{self, BodyOnLattice};
use super::se3::{self, Rigid, Twist};
use super::svf::{ExactRegion, Lattice, Svf};
use super::{Mode, PostureModel};

/// Bodies kept per node in [`SparseWeights`].
pub const TOP: usize = 4;

/// The blend's weights at every node, the [`TOP`] largest only.
#[derive(Clone, Debug)]
pub struct SparseWeights {
    /// Per node: (body index, weight), unused entries with weight zero.
    pub entries: Vec<[(u16, f32); TOP]>,
}

impl SparseWeights {
    /// The weights [`polyrigid::blend`] gives `bodies` over `lattice`.
    pub fn of(lattice: &Lattice, bodies: &[BodyOnLattice], reach_mm: f64) -> SparseWeights {
        let entries = (0..lattice.len())
            .into_par_iter()
            .map(|n| {
                let w = polyrigid::weights(bodies, n, reach_mm);
                let mut order: Vec<usize> = (0..w.len()).filter(|&i| w[i] > 0.0).collect();
                // The largest first; a tie to the lower body index.
                order.sort_by(|&a, &b| w[b].total_cmp(&w[a]).then(a.cmp(&b)));
                let mut out = [(0u16, 0f32); TOP];
                for (slot, &i) in out.iter_mut().zip(order.iter()) {
                    *slot = (i as u16, w[i] as f32);
                }
                out
            })
            .collect();
        SparseWeights { entries }
    }
}

/// One body's curve within a segment (see the module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyCurve {
    /// The segment's own twist of the body, `log g`.
    pub xi: Twist,
    /// `w_a`, `w_b`, `w_c`, applied in that order.
    pub omega: [Twist; 3],
}

impl BodyCurve {
    /// The stationary screw `exp(s xi)`.
    pub fn stationary(xi: Twist) -> BodyCurve {
        let third = xi.scaled(1.0 / 3.0);
        BodyCurve {
            xi,
            omega: [third; 3],
        }
    }

    /// The curve from the identity to `g` that leaves with spatial velocity
    /// `m0` and arrives with `m1`.
    pub fn hermite(g: &Rigid, m0: Twist, m1: Twist) -> BodyCurve {
        let a = m0.scaled(1.0 / 3.0);
        let c = m1.scaled(1.0 / 3.0);
        // exp(w_b) = exp(-w_c) g exp(-w_a): exp(-w_a) first.
        let mid = se3::exp(&a.scaled(-1.0))
            .then(g)
            .then(&se3::exp(&c.scaled(-1.0)));
        BodyCurve {
            xi: se3::log(g),
            omega: [a, se3::log(&mid), c],
        }
    }

    /// The motion from the segment's start to `s` in `[0, 1]`.
    pub fn pose(&self, s: f64) -> Rigid {
        let s = s.clamp(0.0, 1.0);
        let u = 1.0 - s;
        let b1 = 1.0 - u * u * u;
        let b2 = s * s * (3.0 - 2.0 * s);
        let b3 = s * s * s;
        se3::exp(&self.omega[0].scaled(b1))
            .then(&se3::exp(&self.omega[1].scaled(b2)))
            .then(&se3::exp(&self.omega[2].scaled(b3)))
    }

    /// `log h(s)`: the twist whose flow for unit time is [`Self::pose`].
    pub fn twist(&self, s: f64) -> Twist {
        se3::log(&self.pose(s))
    }

    /// Is it the stationary screw (to rounding)?
    pub fn is_stationary(&self) -> bool {
        let third = self.xi.scaled(1.0 / 3.0);
        self.omega
            .iter()
            .all(|w| (w.w - third.w).length() < 1e-12 && (w.v - third.v).length() < 1e-9)
    }
}

/// A segment's bodies on curves, and the weights that spread them.
#[derive(Clone, Debug)]
pub struct SegmentCurves {
    /// In the segment's body order.
    pub bodies: Vec<BodyCurve>,
    pub weights: Arc<SparseWeights>,
}

impl SegmentCurves {
    /// Every body on its stationary screw (what a segment starts with
    /// before [`smooth`] sees its neighbours).
    pub fn stationary(xis: &[Twist], weights: SparseWeights) -> SegmentCurves {
        SegmentCurves {
            bodies: xis.iter().map(|x| BodyCurve::stationary(*x)).collect(),
            weights: Arc::new(weights),
        }
    }

    /// Is every body on its stationary screw?
    pub fn is_stationary(&self) -> bool {
        self.bodies.iter().all(BodyCurve::is_stationary)
    }

    /// The velocity field whose flow for unit time is the segment's
    /// poly-rigid layer at `s` (see the module doc).
    pub fn velocity_at(&self, base: &Svf, s: f64) -> Svf {
        let eta: Vec<Twist> = self.bodies.iter().map(|c| c.twist(s)).collect();
        let corr: Vec<Twist> = self
            .bodies
            .iter()
            .zip(&eta)
            .map(|(c, e)| e.plus(&c.xi.scaled(-s)))
            .collect();
        let xi: Vec<Twist> = base
            .xi
            .par_iter()
            .zip(self.weights.entries.par_iter())
            .map(|(x, w)| {
                // In entry order, which is fixed: the same twist on any
                // thread count.
                w.iter()
                    .filter(|(_, wt)| *wt > 0.0)
                    .fold(x.scaled(s), |acc, &(b, wt)| match corr.get(b as usize) {
                        Some(c) => acc.plus(&c.scaled(wt as f64)),
                        None => acc,
                    })
            })
            .collect();
        // Each exact region belongs to the body whose twist it carries.
        let exact = base
            .exact
            .iter()
            .map(|r| {
                let body = self.bodies.iter().position(|c| c.xi == r.xi);
                ExactRegion {
                    xi: body.map_or(r.xi.scaled(s), |b| eta[b]),
                    nodes: r.nodes.clone(),
                }
            })
            .collect();
        Svf {
            lattice: base.lattice.clone(),
            xi,
            exact,
        }
    }
}

/// Velocities at the states: per state, the twist of each body named, from
/// the twists of the segments either side of it.
fn tangent(model: &PostureModel, name: &str, state: usize) -> Option<Twist> {
    let xi_of = |seg: usize| -> Option<Twist> {
        model
            .segments
            .get(seg)?
            .bodies
            .iter()
            .find(|b| b.name == name)
            .map(|b| b.xi)
    };
    let before = state.checked_sub(1).and_then(xi_of);
    let after = xi_of(state);
    match (before, after) {
        (Some(a), Some(b)) => Some(a.plus(&b).scaled(0.5)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// Put every body of a chain on its curve through the states. Segments
/// without weights (fitted without [`super::fit::FitOptions::keep_weights`])
/// and linear models are left alone. Returns whether anything changed; the
/// caller runs [`PostureModel::finish`] after.
pub fn smooth(model: &mut PostureModel) -> bool {
    if model.segments.len() < 2 || model.mode() == Mode::Linear {
        return false;
    }
    let mut changed = false;
    for i in 0..model.segments.len() {
        let Some(curves) = model.segments[i].curves.clone() else {
            continue;
        };
        let mut next = (*curves).clone();
        for (bi, b) in model.segments[i].bodies.iter().enumerate() {
            if b.anchor || bi >= next.bodies.len() {
                continue;
            }
            let (Some(m0), Some(m1)) = (tangent(model, &b.name, i), tangent(model, &b.name, i + 1))
            else {
                continue;
            };
            next.bodies[bi] = BodyCurve::hermite(&b.motion, m0, m1);
        }
        if !next.is_stationary() {
            changed = true;
        }
        model.segments[i].curves = Some(Arc::new(next));
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    fn twist(deg: f64, t: Vec3) -> Twist {
        Twist {
            w: Vec3::new(1.0, 0.3, -0.2).normalized() * deg.to_radians(),
            v: t,
        }
    }

    #[test]
    fn equal_neighbours_give_the_stationary_screw() {
        let xi = twist(35.0, Vec3::new(4.0, -2.0, 7.0));
        let g = se3::exp(&xi);
        let c = BodyCurve::hermite(&g, xi, xi);
        for s in [0.0, 0.2, 0.5, 0.9, 1.0] {
            let want = se3::exp(&xi.scaled(s));
            let got = c.pose(s);
            let p = Vec3::new(30.0, -10.0, 50.0);
            assert!((got.map(p) - want.map(p)).length() < 1e-9, "s {s}");
        }
        assert!(c.is_stationary());
    }

    #[test]
    fn a_hermite_curve_lands_on_its_end_and_leaves_and_arrives_as_asked() {
        let g = se3::exp(&twist(40.0, Vec3::new(5.0, 3.0, -8.0)));
        let m0 = twist(10.0, Vec3::new(1.0, 0.0, 0.0));
        let m1 = twist(60.0, Vec3::new(0.0, -6.0, 2.0));
        let c = BodyCurve::hermite(&g, m0, m1);
        let p = Vec3::new(20.0, 40.0, -15.0);
        assert!((c.pose(1.0).map(p) - g.map(p)).length() < 1e-9);
        assert!((c.pose(0.0).map(p) - p).length() < 1e-12);
        // One-sided differences at the ends: pose(s) h^-1 near 0 and 1.
        let h = 1e-5;
        let v0 = se3::log(&c.pose(h)).scaled(1.0 / h);
        let v1 = se3::log(&c.pose(1.0 - h).inverse().then(&c.pose(1.0))).scaled(1.0 / h);
        for (got, want) in [(v0, m0), (v1, m1)] {
            assert!((got.w - want.w).length() < 1e-4, "{got:?} vs {want:?}");
            assert!((got.v - want.v).length() < 1e-3, "{got:?} vs {want:?}");
        }
    }
}
