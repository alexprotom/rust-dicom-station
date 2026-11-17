//! Rigid transforms as elements of the group SE(3): the matrix logarithm
//! and exponential, screw parameters, and a rotation split about the
//! patient axes.
//!
//! A rigid motion `x -> R x + t` is a point of SE(3); its logarithm is a
//! *twist* `xi = (w, v)`, the velocity field `x -> w x x + v` whose flow for
//! unit time is that motion. Twists are vectors: they add and scale, and a
//! blend `sum_i a_i xi_i` of twists is again a twist, which is what the
//! poly-rigid layer of [`super`] mixes. Scaling a twist by `t` and taking
//! the exponential walks the motion on its *screw*: a constant axis, a
//! constant angular rate, a constant rate of travel along the axis, which
//! is what a joint does to first order.
//!
//! **Logarithm.** The angle comes from both halves of the rotation matrix,
//! `theta = atan2(|skew(R)|, (tr R - 1) / 2)`, which keeps its precision
//! at both ends of `[0, pi]` where `acos` alone loses it. The axis is the
//! skew part over `sin theta` while that is large, and the dominant column
//! of the symmetric part `(R + R^T) / 2 - cos(theta) I = (1 - cos theta) a a^T`
//! past a right angle, signed by the skew part: near half a turn the skew
//! part is all rounding and the symmetric part is exact. The translation is
//! `v = V^-1 t` with `V = I + (1 - cos) / theta^2 W + (theta - sin) / theta^3 W^2`,
//! `W` the cross-product matrix of `w`; every coefficient has its Taylor
//! series below `1e-4` rad, so the identity and a small turn are exact too.
//!
//! **Exponential.** Rodrigues' formula for `R`, the same `V` for `t`.
//! `exp(log(g)) = g` to rounding for any angle up to and including half a
//! turn (tested at `pi - 1e-9` and `pi`).
//!
//! **Screw.** A motion with angle `theta` about the unit axis `a` is a
//! rotation about the line through `q` along `a` and a travel `d = a . t`
//! along it; `q` solves `(I - R) q = t - d a` in the plane normal to `a`,
//! `q = (t_p + cot(theta / 2) a x t_p) / 2`, and is moved along the axis to
//! the point nearest a reference (the body's own centroid), so it reads as
//! "the pivot near this bone". This is Chasles' theorem; the formulas are
//! those of Murray, Li and Sastry, *A Mathematical Introduction to Robotic
//! Manipulation* (1994), ch. 2.
//!
//! **Joint angles.** A relative rotation is split as `Rz Ry Rx` about the
//! patient axes x (right-left), y (anterior-posterior), z (inferior-
//! superior), the convention of [`crate::registration::RigidTransform`] and
//! of the registration analysis: the x angle is flexion / extension of a
//! hip or a spine, the y angle abduction / adduction or side bending, the z
//! angle axial rotation. The split is about the axes of the first state in
//! the anchor frame, not about a joint's own anatomical frame (which the
//! station has no landmarks for).
//!
//! Everything is `f64` patient millimetres (LPS) and radians unless a name
//! says degrees.

use crate::geometry::Vec3;
use crate::registration::RigidTransform;

/// A 3 x 3 matrix, row-major: `m[row][col]`.
pub type M3 = [[f64; 3]; 3];

/// The identity matrix.
pub const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Below this angle (rad) the coefficients of the exponential and the
/// logarithm are taken from their series.
const SMALL: f64 = 1e-4;

#[inline]
pub fn mat_vec(m: &M3, v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

pub fn mat_mul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

pub fn transpose(a: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for (i, row) in a.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            r[j][i] = *v;
        }
    }
    r
}

pub fn det(a: &M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn add(a: &M3, b: &M3, sa: f64, sb: f64) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = sa * a[i][j] + sb * b[i][j];
        }
    }
    r
}

/// The cross-product matrix: `skew(w) v = w x v`.
pub fn skew(w: Vec3) -> M3 {
    [[0.0, -w.z, w.y], [w.z, 0.0, -w.x], [-w.y, w.x, 0.0]]
}

/// A rigid motion `x -> r x + t`, patient millimetres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid {
    pub r: M3,
    pub t: Vec3,
}

impl Default for Rigid {
    fn default() -> Self {
        Rigid::IDENTITY
    }
}

impl Rigid {
    pub const IDENTITY: Rigid = Rigid {
        r: I3,
        t: Vec3::ZERO,
    };

    #[inline]
    pub fn map(&self, p: Vec3) -> Vec3 {
        mat_vec(&self.r, p) + self.t
    }

    /// The exact inverse: `R^T (x - t)`.
    pub fn inverse(&self) -> Rigid {
        let rt = transpose(&self.r);
        Rigid {
            r: rt,
            t: mat_vec(&rt, self.t) * -1.0,
        }
    }

    /// `self` first, then `next`: `next o self`.
    pub fn then(&self, next: &Rigid) -> Rigid {
        Rigid {
            r: mat_mul(&next.r, &self.r),
            t: mat_vec(&next.r, self.t) + next.t,
        }
    }

    /// The mapping a registration's rigid transform performs (its centre
    /// folded into the translation).
    pub fn from_transform(rt: &RigidTransform) -> Rigid {
        let m = rt.matrix();
        let r = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
        // map(0) is the translation of `x -> R x + t`.
        Rigid {
            r,
            t: rt.map(Vec3::ZERO),
        }
    }

    /// The same mapping as a registration's rigid transform about `center`
    /// (Euler `Rz Ry Rx`, the convention of [`RigidTransform`]).
    pub fn to_transform(&self, center: Vec3) -> RigidTransform {
        let [rx, ry, rz] = euler_zyx(&self.r);
        let about_origin =
            RigidTransform::new([rx, ry, rz, self.t.x, self.t.y, self.t.z], Vec3::ZERO);
        about_origin.recentered(center)
    }

    /// The rotation angle, degrees, in `[0, 180]`.
    pub fn angle_deg(&self) -> f64 {
        log(self).w.length().to_degrees()
    }

    /// Distance between this motion and `other` at a point: how far the two
    /// put `p` apart.
    pub fn gap_at(&self, other: &Rigid, p: Vec3) -> f64 {
        (self.map(p) - other.map(p)).length()
    }
}

/// Euler angles `[rx, ry, rz]` (rad) of `r = Rz Ry Rx`; in gimbal lock the
/// whole of `rx +- rz` goes to `rx`.
pub fn euler_zyx(r: &M3) -> [f64; 3] {
    let sy = -r[2][0];
    let ry = sy.clamp(-1.0, 1.0).asin();
    if (1.0 - sy.abs()) < 1e-12 {
        [r[0][1].atan2(r[1][1]), ry, 0.0]
    } else {
        [r[2][1].atan2(r[2][2]), ry, r[1][0].atan2(r[0][0])]
    }
}

/// An element of se(3): the velocity field `x -> w x x + v`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Twist {
    /// Angular velocity, rad per unit time, about the patient origin.
    pub w: Vec3,
    /// Linear velocity of the patient origin, mm per unit time.
    pub v: Vec3,
}

impl Twist {
    pub const ZERO: Twist = Twist {
        w: Vec3::ZERO,
        v: Vec3::ZERO,
    };

    /// A pure velocity (no turning): what a plain vector field is made of.
    pub fn translation(v: Vec3) -> Twist {
        Twist { w: Vec3::ZERO, v }
    }

    #[inline]
    pub fn velocity(&self, p: Vec3) -> Vec3 {
        self.w.cross(p) + self.v
    }

    #[inline]
    pub fn scaled(&self, s: f64) -> Twist {
        Twist {
            w: self.w * s,
            v: self.v * s,
        }
    }

    #[inline]
    pub fn plus(&self, o: &Twist) -> Twist {
        Twist {
            w: self.w + o.w,
            v: self.v + o.v,
        }
    }
}

/// Below this angle (rad) the coefficients that are differences of nearly
/// equal numbers are taken from their series (to the sixth order, below
/// rounding here).
const SERIES: f64 = 0.1;

/// `(sin theta) / theta`, `(1 - cos theta) / theta^2`, `(theta - sin theta) / theta^3`.
///
/// `1 - cos` is written `2 sin^2(theta / 2)`, which has no cancellation;
/// `theta - sin` has nothing like it and goes to its series below
/// [`SERIES`].
fn coefficients(theta: f64) -> (f64, f64, f64) {
    let t2 = theta * theta;
    if theta < SMALL {
        return (1.0 - t2 / 6.0, 0.5 - t2 / 24.0, 1.0 / 6.0 - t2 / 120.0);
    }
    let s = theta.sin();
    let h = (theta * 0.5).sin();
    let c = if theta < SERIES {
        1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0 - t2 * t2 * t2 / 362_880.0
    } else {
        (theta - s) / (t2 * theta)
    };
    (s / theta, 2.0 * h * h / t2, c)
}

/// The rigid motion a twist flows to in unit time.
pub fn exp(xi: &Twist) -> Rigid {
    let theta = xi.w.length();
    let (a, b, c) = coefficients(theta);
    let w = skew(xi.w);
    let w2 = mat_mul(&w, &w);
    let r = add(&add(&I3, &w, 1.0, a), &w2, 1.0, b);
    let v = add(&add(&I3, &w, 1.0, b), &w2, 1.0, c);
    Rigid {
        r,
        t: mat_vec(&v, xi.v),
    }
}

/// The rotation vector `theta a` of a rotation matrix, `theta` in `[0, pi]`.
pub fn log_rotation(r: &M3) -> Vec3 {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let c = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0);
    // sin(theta) a, from the skew-symmetric part.
    let sv = Vec3::new(r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]) * 0.5;
    let s = sv.length();
    let theta = s.atan2(c);
    if theta < SMALL {
        // R ~ I + W: the skew part is the rotation vector to second order,
        // and the series of theta / sin(theta) takes it to third.
        return sv * (1.0 + theta * theta / 6.0);
    }
    if c > -0.5 {
        return sv * (theta / s);
    }
    // Past 120 degrees the skew part is the small difference of two large
    // numbers; the symmetric part (1 - c) a a^T is exact. Its largest
    // diagonal entry gives the best-conditioned column.
    let k = 1.0 - c;
    let b = |i: usize, j: usize| (r[i][j] + r[j][i]) * 0.5 - if i == j { c } else { 0.0 };
    let j = (0..3)
        .max_by(|&x, &y| b(x, x).total_cmp(&b(y, y)))
        .unwrap_or(0);
    let aj = (b(j, j) / k).max(0.0).sqrt();
    let mut a = Vec3::new(b(0, j), b(1, j), b(2, j)) * (1.0 / (aj * k).max(1e-300));
    a = a.normalized();
    // The sign is the skew part's; at exactly half a turn both are right.
    if a.dot(sv) < 0.0 {
        a = a * -1.0;
    }
    a * theta
}

/// The twist whose flow for unit time is `g`.
pub fn log(g: &Rigid) -> Twist {
    let w = log_rotation(&g.r);
    let theta = w.length();
    let wm = skew(w);
    let w2 = mat_mul(&wm, &wm);
    // V^-1 = I - W / 2 + c W^2, c = (1 - theta sin / (2 (1 - cos))) / theta^2
    //      = (1 - (theta / 2) cot(theta / 2)) / theta^2, a difference of
    // nearly equal numbers for a small turn: its series there.
    let t2 = theta * theta;
    let c = if theta < SERIES {
        1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30_240.0 + t2 * t2 * t2 / 1_209_600.0
    } else {
        let x = theta * 0.5;
        (1.0 - x / x.tan()) / t2
    };
    let vinv = add(&add(&I3, &wm, 1.0, -0.5), &w2, 1.0, c);
    Twist {
        w,
        v: mat_vec(&vinv, g.t),
    }
}

/// A rigid motion as a screw: an axis, a turn about it and a travel along
/// it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Screw {
    /// Unit direction of the axis (the translation's direction for a pure
    /// shift).
    pub axis: Vec3,
    /// The point of the axis nearest the reference it was asked about.
    pub point: Vec3,
    /// Turn about the axis, degrees, right-handed, `[0, 180]`.
    pub angle_deg: f64,
    /// Travel along the axis, mm.
    pub travel_mm: f64,
    /// How far the reference point itself moved, mm.
    pub reference_shift_mm: f64,
}

/// The screw of `g`, its axis point taken nearest `about`.
pub fn screw(g: &Rigid, about: Vec3) -> Screw {
    let w = log_rotation(&g.r);
    let theta = w.length();
    let shift = (g.map(about) - about).length();
    if theta < 1e-9 {
        let len = g.t.length();
        return Screw {
            axis: if len > 0.0 {
                g.t * (1.0 / len)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            },
            point: about,
            angle_deg: 0.0,
            travel_mm: len,
            reference_shift_mm: shift,
        };
    }
    let a = w * (1.0 / theta);
    let d = a.dot(g.t);
    let tp = g.t - a * d;
    let cot = 1.0 / (theta * 0.5).tan();
    let q0 = (tp + a.cross(tp) * cot) * 0.5;
    let q = q0 + a * (about - q0).dot(a);
    Screw {
        axis: a,
        point: q,
        angle_deg: theta.to_degrees(),
        travel_mm: d,
        reference_shift_mm: shift,
    }
}

/// A rotation split about the patient axes, degrees: `[x, y, z]` of
/// `R = Rz Ry Rx` (flexion, abduction, axial rotation for a joint; see the
/// module doc).
pub fn patient_axis_angles(r: &M3) -> [f64; 3] {
    let e = euler_zyx(r);
    [e[0].to_degrees(), e[1].to_degrees(), e[2].to_degrees()]
}

/// Eigenvalues (ascending) and unit eigenvectors (as columns `v[.][k]`) of a
/// symmetric 3 x 3 matrix, by cyclic Jacobi rotations: a few sweeps reach
/// rounding, and the vectors come out orthonormal. What the principal axes
/// of a structure and the stretches of a deformation are read from.
pub fn sym_eigen(a: &M3) -> ([f64; 3], M3) {
    let mut m = *a;
    let mut v = I3;
    for _ in 0..64 {
        let off = m[0][1].abs() + m[0][2].abs() + m[1][2].abs();
        if off < 1e-300 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if m[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // m <- J^T m J, v <- v J.
            for row in m.iter_mut() {
                let (mkp, mkq) = (row[p], row[q]);
                row[p] = c * mkp - s * mkq;
                row[q] = s * mkp + c * mkq;
            }
            let (rp, rq) = (m[p], m[q]);
            for k in 0..3 {
                m[p][k] = c * rp[k] - s * rq[k];
                m[q][k] = s * rp[k] + c * rq[k];
            }
            for row in v.iter_mut() {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&x, &y| m[x][x].total_cmp(&m[y][y]));
    let vals = [
        m[order[0]][order[0]],
        m[order[1]][order[1]],
        m[order[2]][order[2]],
    ];
    let mut vecs = [[0.0; 3]; 3];
    for (col, &k) in order.iter().enumerate() {
        for row in 0..3 {
            vecs[row][col] = v[row][k];
        }
    }
    (vals, vecs)
}

/// The largest departure from length-preservation of a deformation
/// gradient `f`: `max |sigma_i - 1|` over its singular values. Zero for a
/// rigid motion; a bone that keeps its shape stays near it.
pub fn stretch(f: &M3) -> f64 {
    let ftf = mat_mul(&transpose(f), f);
    let (vals, _) = sym_eigen(&ftf);
    vals.iter()
        .map(|l| (l.max(0.0).sqrt() - 1.0).abs())
        .fold(0.0, f64::max)
}

/// The rotation nearest a matrix (its polar factor), from the eigen
/// decomposition of `m^T m`; `None` for a matrix that maps space flat.
pub fn nearest_rotation(m: &M3) -> Option<M3> {
    let (vals, v) = sym_eigen(&mat_mul(&transpose(m), m));
    if vals[0] <= 1e-18 {
        return None;
    }
    // (m^T m)^(-1/2) = V diag(1 / sqrt(l)) V^T; R = m (m^T m)^(-1/2).
    let mut inv_sqrt = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            inv_sqrt[i][j] = (0..3).map(|k| v[i][k] * v[j][k] / vals[k].sqrt()).sum();
        }
    }
    let r = mat_mul(m, &inv_sqrt);
    (det(&r) > 0.0).then_some(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot(axis: Vec3, angle: f64) -> M3 {
        exp(&Twist {
            w: axis.normalized() * angle,
            v: Vec3::ZERO,
        })
        .r
    }

    fn close(a: &Rigid, b: &Rigid, tol: f64) -> bool {
        (0..3).all(|i| (0..3).all(|j| (a.r[i][j] - b.r[i][j]).abs() < tol))
            && (a.t - b.t).length() < tol * 100.0
    }

    #[test]
    fn exp_of_log_is_the_motion_for_every_angle_up_to_half_a_turn() {
        let axes = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, -0.8, 0.5),
            Vec3::new(-0.2, 0.1, 0.97),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let angles = [
            0.0,
            1e-12,
            1e-6,
            1e-4,
            0.3,
            1.0,
            2.0,
            std::f64::consts::PI - 1e-3,
            std::f64::consts::PI - 1e-9,
            std::f64::consts::PI,
        ];
        for a in axes {
            for th in angles {
                let g = Rigid {
                    r: rot(a, th),
                    t: Vec3::new(12.0, -40.0, 230.0),
                };
                let back = exp(&log(&g));
                assert!(close(&g, &back, 1e-12), "axis {a:?} angle {th}: {back:?}");
                // The angle comes back (the axis may flip at exactly pi).
                assert!((log(&g).w.length() - th).abs() < 1e-9, "{th}");
            }
        }
    }

    #[test]
    fn a_scaled_twist_walks_the_screw_and_halves_compose_to_the_whole() {
        let g = Rigid {
            r: rot(Vec3::new(0.2, 1.0, -0.1), 1.2),
            t: Vec3::new(30.0, 5.0, -12.0),
        };
        let xi = log(&g);
        let half = exp(&xi.scaled(0.5));
        assert!(close(&half.then(&half), &g, 1e-12));
        let third = exp(&xi.scaled(1.0 / 3.0));
        assert!(close(&third.then(&third).then(&third), &g, 1e-12));
        // The inverse is the twist backwards.
        assert!(close(&exp(&xi.scaled(-1.0)), &g.inverse(), 1e-12));
    }

    #[test]
    fn a_hinge_reads_back_as_its_screw() {
        // 40 degrees about the RL axis through (10, -20, 5), 3 mm along it.
        let q = Vec3::new(10.0, -20.0, 5.0);
        let a = Vec3::new(1.0, 0.0, 0.0);
        let r = rot(a, 40f64.to_radians());
        let t = q - mat_vec(&r, q) + a * 3.0;
        let g = Rigid { r, t };
        let s = screw(&g, Vec3::new(50.0, 0.0, 0.0));
        assert!((s.angle_deg - 40.0).abs() < 1e-9);
        assert!((s.axis - a).length() < 1e-9);
        assert!((s.travel_mm - 3.0).abs() < 1e-9);
        assert!((s.point - Vec3::new(50.0, -20.0, 5.0)).length() < 1e-9);
        let ang = patient_axis_angles(&r);
        assert!((ang[0] - 40.0).abs() < 1e-9 && ang[1].abs() < 1e-9 && ang[2].abs() < 1e-9);
    }

    #[test]
    fn registrations_rigid_transform_converts_both_ways() {
        let rt = RigidTransform::new([0.1, -0.2, 0.3, 4.0, 5.0, -6.0], Vec3::new(7.0, 8.0, 9.0));
        let g = Rigid::from_transform(&rt);
        let p = Vec3::new(-30.0, 22.0, 104.0);
        assert!((g.map(p) - rt.map(p)).length() < 1e-9);
        let back = g.to_transform(Vec3::new(1.0, 2.0, 3.0));
        assert!((back.map(p) - rt.map(p)).length() < 1e-9);
    }

    #[test]
    fn eigen_and_polar_factor_are_exact_on_known_matrices() {
        let r = rot(Vec3::new(1.0, 2.0, 3.0), 0.7);
        // F = R diag(1, 2, 3) R^T has eigenvalues 1, 2, 3.
        let d = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let f = mat_mul(&mat_mul(&r, &d), &transpose(&r));
        let (vals, vecs) = sym_eigen(&f);
        for (v, want) in vals.iter().zip([1.0, 2.0, 3.0]) {
            assert!((v - want).abs() < 1e-12, "{vals:?}");
        }
        assert!((det(&vecs).abs() - 1.0).abs() < 1e-12);
        // A rotation times a stretch: the polar factor is the rotation.
        let m = mat_mul(&r, &f);
        let near = nearest_rotation(&m).unwrap();
        assert!((0..3).all(|i| (0..3).all(|j| (near[i][j] - r[i][j]).abs() < 1e-10)));
        assert!(stretch(&r) < 1e-12);
        assert!((stretch(&d) - 2.0).abs() < 1e-12);
    }
}
