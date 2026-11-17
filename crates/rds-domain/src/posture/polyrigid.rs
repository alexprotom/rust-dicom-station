//! The poly-rigid blend: the velocity field that moves every rigid body by
//! its own screw and the space between them smoothly.
//!
//! Each body `i` has a rigid motion between two states, written as its
//! logarithm `xi_i` ([`super::se3::log`], a twist). The blended field is
//!
//! ```text
//! xi(x) = sum_i w_i(x) xi_i,        sum_i w_i(x) = 1
//! ```
//!
//! and the deformation is its flow, `exp(xi)` ([`super::svf::Svf::exp`]).
//! Twists add as vectors, so the blend is again a twist at every node; since
//! a twist acts on homogeneous points linearly, the mix does not depend on
//! where the patient origin is. This is the log-Euclidean poly-affine
//! framework of Arsigny, Commowick, Ayache and Pennec (*J Math Imaging Vis*
//! 2009;33:222-238) restricted to rigid components (Alexa, *Linear
//! combination of transformations*, SIGGRAPH 2002, mixes transforms the same
//! way).
//!
//! **Where a body's weight is one: its swept region.** The field is
//! stationary, so a point moves by body `i`'s screw exactly only while its
//! whole path stays where `w_i = 1`. A bone that turns 60 degrees about a
//! joint leaves the place it started from, and with weights read off its
//! first position its far end would slide into the blend half way and
//! bend. So the region of weight one is the volume the body *sweeps* on
//! its screw from `t = 0` to `t = 1` (the lattice nodes `y` for which
//! `exp(-t xi_i)(y)` falls in the body at one of 16 times), grown by one
//! node so the trilinear cell of every point of the path is in it too.
//! Inside it the velocity is the body's own twist, so the true flow moves
//! the body by exactly its rigid motion at every `t`.
//! Where the swept regions of two bodies meet, a node goes to the body
//! that is there at the start - judged first on the bodies as they are,
//! then on them grown by a node, so at a joint narrower than a lattice step
//! (the femoral head in its socket) the head's nodes stay the femur's and
//! the socket's the pelvis's; if both or neither are, the two share it.
//! The nodes that belong to one body alone are handed to the exponential as
//! an [`ExactRegion`], so a node whose path stays among them is given the
//! body's motion exactly rather than the interpolated one.
//!
//! **Weights elsewhere.** `d_i(x)` is the exact Euclidean distance from a
//! node to the swept region of body `i`
//! ([`crate::morphology::dist2_to_foreground`] on the lattice), and
//!
//! ```text
//! g_i(x) = exp(-d_i / reach) / d_i^2,        w_i = g_i / sum_j g_j
//! ```
//!
//! Shepard's inverse-distance weights, which interpolate exactly - `w_i`
//! tends to one at the surface of region `i` and the others to zero, so the
//! field is continuous there - with an exponential reach so a bone's
//! influence fades over a few centimetres rather than over the whole
//! patient. The ratio is taken relative to the nearest body, so nothing
//! under- or overflows however far a node is from all of them.
//!
//! **Rigidity.** The distances to the bodies where they are at the start
//! give a weight `rho(x)`: 1 inside a body, falling smoothly (a smoothstep
//! over two lattice steps) to 0 away from all of them. The residual layer
//! of [`super::fit`] acts before the blend, in the first state's
//! configuration, multiplied by `1 - rho`, so it never bends a bone.

use rayon::prelude::*;

use super::se3::{self, Twist};
use super::svf::{ExactRegion, Lattice, Svf};
use crate::geometry::Vec3;
use crate::morphology;
use crate::volume::Grid;

/// Times at which a body's sweep is sampled.
const SWEEP_STEPS: usize = 16;

/// One rigid body as the blend sees it.
#[derive(Clone, Debug)]
pub struct BodyOnLattice {
    pub name: String,
    /// Its motion between the two states, as a twist.
    pub xi: Twist,
    /// The nodes it covers at the start, as it is.
    pub occ: Vec<u8>,
    /// The same grown by one node.
    pub own: Vec<u8>,
    /// Distance from every node to its swept region, mm (0 inside).
    pub dist_mm: Vec<f32>,
    /// Distance from every node to the body at the start, mm.
    pub own_dist_mm: Vec<f32>,
}

impl BodyOnLattice {
    /// A body from its occupancy at the start and its motion.
    pub fn new(name: String, xi: Twist, occ: &[u8], lattice: &Lattice) -> BodyOnLattice {
        let own = grow(occ, lattice.dims);
        let swept = swept(&own, &xi, lattice);
        BodyOnLattice {
            name,
            xi,
            dist_mm: distances(&swept, lattice),
            own_dist_mm: distances(occ, lattice),
            occ: occ.to_vec(),
            own,
        }
    }
}

/// A body's mask on its image grid, put onto the lattice: the nodes it
/// covers, sampled at the nearest voxel; the node nearest its centroid
/// when it covers none.
pub fn occupancy(mask: &[u8], grid: &Grid, lattice: &Lattice) -> Vec<u8> {
    let [nx, ny, nz] = grid.dims;
    let mut occ: Vec<u8> = (0..lattice.len())
        .into_par_iter()
        .map(|n| {
            let v = grid.patient_to_voxel(lattice.node(n));
            let (i, j, k) = (v[0].round(), v[1].round(), v[2].round());
            if i < 0.0 || j < 0.0 || k < 0.0 {
                return 0;
            }
            let (i, j, k) = (i as usize, j as usize, k as usize);
            if i >= nx || j >= ny || k >= nz {
                return 0;
            }
            (mask[k * nx * ny + j * nx + i] != 0) as u8
        })
        .collect();
    if occ.iter().all(|&v| v == 0) {
        if let Some(c) = crate::motion::centroid_mm(mask, grid) {
            let u = lattice.coords(c);
            let mut ijk = [0usize; 3];
            for a in 0..3 {
                ijk[a] = u[a].round().clamp(0.0, (lattice.dims[a] - 1) as f64) as usize;
            }
            occ[lattice.index(ijk[0], ijk[1], ijk[2])] = 1;
        }
    }
    occ
}

/// An occupancy grown by one node in all 26 directions.
fn grow(occ: &[u8], dims: [usize; 3]) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    (0..occ.len())
        .into_par_iter()
        .map(|n| {
            let (i, j, k) = (n % nx, (n / nx) % ny, n / (nx * ny));
            for dk in -1i64..=1 {
                for dj in -1i64..=1 {
                    for di in -1i64..=1 {
                        let (a, b, c) = (i as i64 + di, j as i64 + dj, k as i64 + dk);
                        if a < 0 || b < 0 || c < 0 {
                            continue;
                        }
                        let (a, b, c) = (a as usize, b as usize, c as usize);
                        if a >= nx || b >= ny || c >= nz {
                            continue;
                        }
                        if occ[a + nx * (b + ny * c)] != 0 {
                            return 1;
                        }
                    }
                }
            }
            0
        })
        .collect()
}

/// The nodes a body passes over on its screw from `t = 0` to `t = 1`.
fn swept(own: &[u8], xi: &Twist, lattice: &Lattice) -> Vec<u8> {
    if *xi == Twist::ZERO {
        return own.to_vec();
    }
    let back: Vec<se3::Rigid> = (0..=SWEEP_STEPS)
        .map(|s| se3::exp(&xi.scaled(-(s as f64) / SWEEP_STEPS as f64)))
        .collect();
    let [nx, ny, nz] = lattice.dims;
    let hit = (0..lattice.len())
        .into_par_iter()
        .map(|n| {
            if own[n] != 0 {
                return 1;
            }
            let y = lattice.node(n);
            for g in &back {
                let u = lattice.coords(g.map(y));
                let (i, j, k) = (u[0].round(), u[1].round(), u[2].round());
                if i < 0.0 || j < 0.0 || k < 0.0 {
                    continue;
                }
                let (i, j, k) = (i as usize, j as usize, k as usize);
                if i < nx && j < ny && k < nz && own[i + nx * (j + ny * k)] != 0 {
                    return 1;
                }
            }
            0
        })
        .collect::<Vec<u8>>();
    grow(&hit, lattice.dims)
}

/// Distance from every node to the nearest occupied node, mm.
pub fn distances(occ: &[u8], lattice: &Lattice) -> Vec<f32> {
    if occ.iter().all(|&v| v == 0) {
        return vec![f32::INFINITY; occ.len()];
    }
    morphology::dist2_to_foreground(occ, lattice.dims, lattice.spacing)
        .into_par_iter()
        .map(|d2| d2.max(0.0).sqrt())
        .collect()
}

/// The normalised weights of every body at node `n`, written into `out`.
fn weights_at(bodies: &[BodyOnLattice], n: usize, reach_mm: f64, out: &mut [f64]) {
    out.iter_mut().for_each(|w| *w = 0.0);
    let inside: Vec<usize> = (0..bodies.len())
        .filter(|&i| bodies[i].dist_mm[n] <= 0.0)
        .collect();
    if !inside.is_empty() {
        // Where two swept regions meet, the body that is there first: as
        // it is, else grown by a node.
        let there = |grown: bool| -> Vec<usize> {
            inside
                .iter()
                .copied()
                .filter(|&i| {
                    if grown {
                        bodies[i].own[n] != 0
                    } else {
                        bodies[i].occ.get(n).is_some_and(|&v| v != 0)
                    }
                })
                .collect()
        };
        let exact = there(false);
        let first = if exact.is_empty() { there(true) } else { exact };
        let pick = if first.is_empty() { &inside } else { &first };
        let w = 1.0 / pick.len() as f64;
        for &i in pick {
            out[i] = w;
        }
        return;
    }
    // log g_i = -d_i / reach - 2 ln d_i, relative to the largest.
    let lg: Vec<f64> = bodies
        .iter()
        .map(|b| {
            let d = b.dist_mm[n] as f64;
            if d.is_finite() {
                -d / reach_mm - 2.0 * d.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let top = lg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !top.is_finite() {
        return;
    }
    let mut sum = 0.0;
    for (w, l) in out.iter_mut().zip(&lg) {
        *w = (l - top).exp();
        sum += *w;
    }
    for w in out.iter_mut() {
        *w /= sum;
    }
}

/// The blended velocity field over `lattice`. With no body it is zero.
pub fn blend(lattice: &Lattice, bodies: &[BodyOnLattice], reach_mm: f64) -> Svf {
    if bodies.is_empty() {
        return Svf::zero(lattice);
    }
    let reach = reach_mm.max(1.0);
    // The twist of every node, and the body it belongs to alone, if any.
    let mixed: Vec<(Twist, Option<u32>)> = (0..lattice.len())
        .into_par_iter()
        .map_init(
            || vec![0.0f64; bodies.len()],
            |w, n| {
                weights_at(bodies, n, reach, w);
                let only = w.iter().position(|&x| x == 1.0).map(|i| i as u32);
                // In body order: the same twist on any thread count.
                let xi = bodies
                    .iter()
                    .zip(w.iter())
                    .fold(Twist::ZERO, |acc, (b, &wi)| acc.plus(&b.xi.scaled(wi)));
                (xi, only)
            },
        )
        .collect();
    let mut exact: Vec<ExactRegion> = bodies
        .iter()
        .map(|b| ExactRegion {
            xi: b.xi,
            nodes: Vec::new(),
        })
        .collect();
    for (n, (_, only)) in mixed.iter().enumerate() {
        if let Some(i) = only {
            exact[*i as usize].nodes.push(n as u32);
        }
    }
    // A body standing still needs no correction: its nodes stay put.
    exact.retain(|r| r.xi != Twist::ZERO && !r.nodes.is_empty());
    Svf {
        lattice: lattice.clone(),
        xi: mixed.into_iter().map(|(x, _)| x).collect(),
        exact,
    }
}

/// The weights of every body at a node, for the tests and the readout.
pub fn weights(bodies: &[BodyOnLattice], n: usize, reach_mm: f64) -> Vec<f64> {
    let mut w = vec![0.0; bodies.len()];
    if !bodies.is_empty() {
        weights_at(bodies, n, reach_mm.max(1.0), &mut w);
    }
    w
}

/// `rho` per node: 1 inside any body at the start, 0 two lattice steps away
/// from all of them.
pub fn rigidity(lattice: &Lattice, bodies: &[BodyOnLattice]) -> Vec<f32> {
    let fall = 2.0 * lattice.min_spacing();
    (0..lattice.len())
        .into_par_iter()
        .map(|n| {
            let d = bodies
                .iter()
                .map(|b| b.own_dist_mm[n] as f64)
                .fold(f64::INFINITY, f64::min);
            if !d.is_finite() {
                return 0.0;
            }
            let x = (1.0 - d / fall).clamp(0.0, 1.0);
            (x * x * (3.0 - 2.0 * x)) as f32
        })
        .collect()
}

/// A displacement field faded by `1 - rho`: what is left of a residual
/// deformation once the bodies keep their own motion.
pub fn fade(d: &mut [Vec3], rho: &[f32]) {
    d.par_iter_mut()
        .zip(rho.par_iter())
        .for_each(|(v, &r)| *v = *v * (1.0 - r as f64));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posture::se3::Rigid;

    fn lattice() -> Lattice {
        Lattice {
            dims: [30, 12, 34],
            spacing: [4.0; 3],
            origin: Vec3::new(-60.0, -24.0, -68.0),
            axes: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        }
    }

    /// Two bars of nodes, one still, one turning 50 degrees about a hinge
    /// between them.
    fn two_bodies(lat: &Lattice) -> (Vec<BodyOnLattice>, Rigid) {
        let occ = |pred: &dyn Fn(Vec3) -> bool| -> Vec<u8> {
            (0..lat.len()).map(|n| pred(lat.node(n)) as u8).collect()
        };
        let upper = occ(&|p| p.z > 12.0 && p.z < 50.0 && p.x.abs() < 9.0 && p.y.abs() < 9.0);
        let lower = occ(&|p| p.z < -12.0 && p.z > -50.0 && p.x.abs() < 9.0 && p.y.abs() < 9.0);
        let r = se3::exp(&Twist {
            w: Vec3::new(0.0, 50f64.to_radians(), 0.0),
            v: Vec3::ZERO,
        })
        .r;
        let g = Rigid { r, t: Vec3::ZERO };
        let bodies = vec![
            BodyOnLattice::new("upper".into(), Twist::ZERO, &upper, lat),
            BodyOnLattice::new("lower".into(), se3::log(&g), &lower, lat),
        ];
        (bodies, g)
    }

    #[test]
    fn a_body_turning_through_fifty_degrees_moves_rigidly_at_every_time() {
        let lat = lattice();
        let (bodies, g) = two_bodies(&lat);
        let v = blend(&lat, &bodies, 30.0);
        for t in [0.3, 0.7, 1.0] {
            let f = v.exp(t);
            let gt = se3::exp(&bodies[1].xi.scaled(t));
            for n in 0..lat.len() {
                let p = lat.node(n);
                let in_lower = p.z < -12.0 && p.z > -50.0 && p.x.abs() < 9.0 && p.y.abs() < 9.0;
                if in_lower {
                    let err = (f.map(p) - gt.map(p)).length();
                    assert!(err < 1e-6, "t {t}, lower bar at {p:?}: {err} mm");
                }
                let in_upper = p.z > 12.0 && p.z < 50.0 && p.x.abs() < 9.0 && p.y.abs() < 9.0;
                if in_upper {
                    assert!(f.d[n].length() < 1e-9, "upper bar moved at {p:?}");
                }
            }
            assert!(f.jacobians().iter().all(|&j| j > 0.0), "t {t} folds");
        }
        let _ = g;
        // Between the two the weights are smooth and sum to one.
        let mid = lat.index(26, 6, 17);
        let w = weights(&bodies, mid, 30.0);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(w.iter().all(|&x| x > 0.0 && x < 1.0), "{w:?}");
    }

    #[test]
    fn the_rigidity_weight_is_one_inside_and_zero_far_away() {
        let lat = lattice();
        let (bodies, _) = two_bodies(&lat);
        let rho = rigidity(&lat, &bodies);
        for (n, &r) in rho.iter().enumerate() {
            let d = bodies[0].own_dist_mm[n].min(bodies[1].own_dist_mm[n]);
            if d == 0.0 {
                assert_eq!(r, 1.0);
            }
            if d as f64 >= 2.0 * lat.min_spacing() {
                assert_eq!(r, 0.0);
            }
        }
    }
}
