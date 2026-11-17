//! Planar contour geometry: the *editable* representation of an RTSTRUCT ROI.
//!
//! An RTSTRUCT stores an ROI as closed planar polygons in patient
//! coordinates, and that is what a planner draws, corrects and hands to the
//! next system. Everything else in this crate works on voxels, which is the
//! right representation for distance transforms, margins and learned
//! segmentation but the wrong one for drawing: a round trip through a 3 mm
//! lattice replaces a curve somebody spent a minute on with a staircase.
//!
//! So this module carries contours as contours:
//!
//! * [`Poly`] - one closed ring, in the in-plane coordinates of a slice
//!   (fractional voxel indices along the lattice's two in-plane axes).
//! * [`Region`] - the polygons of one slice, filled by the **even-odd** rule.
//!   That is what RTSTRUCT means (a hole is a second polygon inside the
//!   first) and what [`crate::segmentation::rasterize_roi`] implements.
//! * [`Stack`] - every slice of one ROI geometry, stacked along one lattice
//!   axis. Axis 2 is the axial stack every planning system expects; drawing
//!   in a sagittal or coronal view produces axis 0 or 1, and
//!   [`Stack::to_axis`] converts - the same "switching patient direction
//!   converts the contours" rule every planning system follows.
//!
//! ## Booleans
//!
//! Editing needs boolean geometry: a stroke in *extend* mode is a union, in
//! *subtract* mode a difference, and tidying overlapping polygons is a
//! self-union. Rather than a polygon clipper - whose failure mode on the
//! degenerate input real contours are full of is a wrong answer, silently -
//! the boolean here rasterizes the operands **locally and supersampled**
//! ([`BOOL_SS`] × per axis by default, over the bounding box of the operands
//! only) and re-extracts the outline with marching squares. The geometric
//! error is bounded by half a fine cell, `1/16` of a voxel at the default
//! factor: two orders of magnitude below what a hand draws, and rings the
//! operation does not touch are passed through *untouched*, with their
//! original vertices. It cannot fail, has no degenerate cases, and handles
//! holes, multiple components and self-intersecting freehand strokes alike.
//!
//! An edit next to a large outline - a brush stamp on a body's edge - goes
//! through [`Region::boolean_local`] instead, which rasterizes only a window
//! around what changes, on the same lattice of fine cells, and splices its
//! trace into the outline outside, which keeps its vertices; where the
//! splice cannot be made it is the boolean above. The 3-D brush merges its
//! footprint the same way ([`BallFootprint::merge`]).

use crate::geometry::Vec3;
use crate::render;
use crate::rtstruct::{Contour, Roi};
use crate::structops::BoolOp;
use crate::volume::{Grid, ViewPlane};

/// A point in the in-plane coordinates of a slice: fractional voxel indices
/// along the lattice's two in-plane axes, in the order given by
/// [`plane_axes`].
pub type Pt = [f64; 2];

/// Supersampling factor per axis of the raster boolean. The boundary of a
/// result lands within half a fine cell of the true one, i.e. `1/(2·SS)` of a
/// voxel.
pub const BOOL_SS: usize = 8;

/// Upper bound on the fine raster of one boolean, so a pathological bounding
/// box degrades the resolution instead of the memory.
const BOOL_MAX_CELLS: usize = 64 << 20;

/// Rings below this area (in voxel units) are dropped as numerical debris.
const MIN_RING_AREA: f64 = 1e-6;

/// A rigid motion in a lattice's own millimetre frame (index × spacing
/// along i / j / k): `p' = R (p - c) + c + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid {
    pub rot: [[f64; 3]; 3],
    pub centre: [f64; 3],
    pub shift: [f64; 3],
}

impl Rigid {
    pub const IDENTITY_ROT: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    pub fn translation(shift: [f64; 3]) -> Rigid {
        Rigid {
            rot: Self::IDENTITY_ROT,
            centre: [0.0; 3],
            shift,
        }
    }

    /// Rotation by `angle` (radians, right-handed) about the line through
    /// `centre` along `dir`. A zero direction gives the identity.
    pub fn rotation_about(dir: [f64; 3], centre: [f64; 3], angle: f64) -> Rigid {
        let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if n.is_nan() || n <= 1e-12 || !angle.is_finite() {
            return Rigid::translation([0.0; 3]);
        }
        let [x, y, z] = dir.map(|v| v / n);
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        Rigid {
            rot: [
                [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
            ],
            centre,
            shift: [0.0; 3],
        }
    }

    /// Rotations about the three lattice axes in turn (x first, then y,
    /// then z), all about `centre`.
    pub fn euler_about(centre: [f64; 3], angles: [f64; 3]) -> Rigid {
        let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut r = Rigid::translation([0.0; 3]);
        r.centre = centre;
        for (dir, a) in axes.into_iter().zip(angles) {
            let q = Rigid::rotation_about(dir, centre, a);
            r.rot = mat_mul(q.rot, r.rot);
        }
        r
    }

    pub fn is_identity(&self) -> bool {
        self.rot == Self::IDENTITY_ROT && self.shift == [0.0; 3]
    }

    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let d = [
            p[0] - self.centre[0],
            p[1] - self.centre[1],
            p[2] - self.centre[2],
        ];
        let r = mat_vec(self.rot, d);
        [
            r[0] + self.centre[0] + self.shift[0],
            r[1] + self.centre[1] + self.shift[1],
            r[2] + self.centre[2] + self.shift[2],
        ]
    }

    /// Where the point that lands on `q` came from.
    pub fn inverse_apply(&self, q: [f64; 3]) -> [f64; 3] {
        let d = [
            q[0] - self.centre[0] - self.shift[0],
            q[1] - self.centre[1] - self.shift[1],
            q[2] - self.centre[2] - self.shift[2],
        ];
        // The inverse of a rotation is its transpose.
        let r = self.rot;
        [
            r[0][0] * d[0] + r[1][0] * d[1] + r[2][0] * d[2] + self.centre[0],
            r[0][1] * d[0] + r[1][1] * d[1] + r[2][1] * d[2] + self.centre[1],
            r[0][2] * d[0] + r[1][2] * d[1] + r[2][2] * d[2] + self.centre[2],
        ]
    }
}

fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

/// The two in-plane lattice axes of a stack sliced along `axis`, in
/// ascending order: `[0, 1]` for the axial stack, `[1, 2]` for sagittal,
/// `[0, 2]` for coronal.
#[inline]
pub fn plane_axes(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

/// The stack axis a view plane draws on: axial slices stack along k,
/// sagittal along i, coronal along j.
#[inline]
pub fn axis_of_plane(plane: ViewPlane) -> usize {
    match plane {
        ViewPlane::Axial => 2,
        ViewPlane::Sagittal => 0,
        ViewPlane::Coronal => 1,
    }
}

/// Name of a stack axis, for the message that says a conversion happened.
pub fn axis_name(axis: usize) -> &'static str {
    match axis {
        0 => "sagittal",
        1 => "coronal",
        _ => "axial",
    }
}

// ---------------------------------------------------------------------------
// Poly - one closed ring
// ---------------------------------------------------------------------------

/// A closed polygon. The closing edge is implicit: the last point is *not* a
/// repeat of the first, which is also how RTSTRUCT stores a contour.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Poly {
    pub pts: Vec<Pt>,
}

impl Poly {
    pub fn new(pts: Vec<Pt>) -> Poly {
        Poly { pts }
    }

    pub fn len(&self) -> usize {
        self.pts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pts.len() < 3
    }

    /// Shoelace: positive counter-clockwise in a right-handed (u, v) frame.
    pub fn signed_area(&self) -> f64 {
        let n = self.pts.len();
        if n < 3 {
            return 0.0;
        }
        let mut s = 0.0;
        for i in 0..n {
            let a = self.pts[i];
            let b = self.pts[(i + 1) % n];
            s += a[0] * b[1] - b[0] * a[1];
        }
        s * 0.5
    }

    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    pub fn is_ccw(&self) -> bool {
        self.signed_area() > 0.0
    }

    pub fn reverse(&mut self) {
        self.pts.reverse();
    }

    /// Force an orientation, in place.
    pub fn orient(&mut self, ccw: bool) {
        if self.is_ccw() != ccw && self.pts.len() >= 3 {
            self.reverse();
        }
    }

    /// Area centroid, falling back to the vertex mean for a degenerate ring.
    pub fn centroid(&self) -> Pt {
        let n = self.pts.len();
        if n == 0 {
            return [0.0, 0.0];
        }
        let a = self.signed_area();
        if a.abs() < 1e-12 {
            let mut c = [0.0, 0.0];
            for p in &self.pts {
                c[0] += p[0];
                c[1] += p[1];
            }
            return [c[0] / n as f64, c[1] / n as f64];
        }
        let (mut cx, mut cy) = (0.0, 0.0);
        for i in 0..n {
            let p = self.pts[i];
            let q = self.pts[(i + 1) % n];
            let cross = p[0] * q[1] - q[0] * p[1];
            cx += (p[0] + q[0]) * cross;
            cy += (p[1] + q[1]) * cross;
        }
        [cx / (6.0 * a), cy / (6.0 * a)]
    }

    pub fn perimeter(&self) -> f64 {
        let n = self.pts.len();
        if n < 2 {
            return 0.0;
        }
        (0..n)
            .map(|i| {
                let p = self.pts[i];
                let q = self.pts[(i + 1) % n];
                ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt()
            })
            .sum()
    }

    /// `[x0, y0, x1, y1]`; an empty ring reports an inverted, empty box.
    pub fn bbox(&self) -> [f64; 4] {
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for p in &self.pts {
            b[0] = b[0].min(p[0]);
            b[1] = b[1].min(p[1]);
            b[2] = b[2].max(p[0]);
            b[3] = b[3].max(p[1]);
        }
        b
    }

    /// Even-odd point test (the rule RTSTRUCT contours are filled by).
    pub fn contains(&self, p: Pt) -> bool {
        let n = self.pts.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        for i in 0..n {
            let a = self.pts[i];
            let b = self.pts[(i + 1) % n];
            // Half-open in y so a vertex on the ray counts once.
            if (a[1] <= p[1]) != (b[1] <= p[1]) {
                let t = (p[1] - a[1]) / (b[1] - a[1]);
                if a[0] + t * (b[0] - a[0]) > p[0] {
                    inside = !inside;
                }
            }
        }
        inside
    }

    pub fn translate(&mut self, d: Pt) {
        for p in &mut self.pts {
            p[0] += d[0];
            p[1] += d[1];
        }
    }

    pub fn scale_about(&mut self, c: Pt, s: f64) {
        for p in &mut self.pts {
            p[0] = c[0] + (p[0] - c[0]) * s;
            p[1] = c[1] + (p[1] - c[1]) * s;
        }
    }

    pub fn rotate_about(&mut self, c: Pt, angle: f64) {
        let (sn, cs) = angle.sin_cos();
        for p in &mut self.pts {
            let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
            p[0] = c[0] + dx * cs - dy * sn;
            p[1] = c[1] + dx * sn + dy * cs;
        }
    }

    /// Douglas-Peucker on a *closed* ring: the ring is split at its two most
    /// distant vertices and each chain simplified, so the result is still
    /// closed and no vertex of the original is displaced.
    pub fn simplify(&self, tol: f64) -> Poly {
        let n = self.pts.len();
        if n < 4 || tol <= 0.0 {
            return self.clone();
        }
        // Anchor: the vertex farthest from pts[0], so the two chains are
        // roughly balanced and the split does not depend on the start index.
        let p0 = self.pts[0];
        let far = (1..n)
            .max_by(|&a, &b| {
                let da = (self.pts[a][0] - p0[0]).powi(2) + (self.pts[a][1] - p0[1]).powi(2);
                let db = (self.pts[b][0] - p0[0]).powi(2) + (self.pts[b][1] - p0[1]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(n - 1);
        let mut out: Vec<Pt> = Vec::with_capacity(n);
        let first: Vec<Pt> = self.pts[..=far].to_vec();
        let mut second: Vec<Pt> = self.pts[far..].to_vec();
        second.push(self.pts[0]);
        let a = dp(&first, tol);
        let b = dp(&second, tol);
        out.extend_from_slice(&a[..a.len() - 1]);
        out.extend_from_slice(&b[..b.len() - 1]);
        if out.len() < 3 {
            return self.clone();
        }
        Poly::new(out)
    }

    /// Even spacing along the ring - what the smoothing and nudge tools want
    /// so a dense freehand stroke and a four-point polygon behave alike.
    pub fn resample(&self, step: f64) -> Poly {
        let n = self.pts.len();
        if n < 3 || step <= 0.0 {
            return self.clone();
        }
        let total = self.perimeter();
        let count = (total / step).round().max(3.0) as usize;
        let step = total / count as f64;
        let mut out = Vec::with_capacity(count);
        let mut acc = 0.0;
        let mut want = 0.0;
        let mut i = 0;
        while i < n && out.len() < count {
            let a = self.pts[i];
            let b = self.pts[(i + 1) % n];
            let seg = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            if seg <= 1e-12 {
                i += 1;
                continue;
            }
            while want <= acc + seg && out.len() < count {
                let t = (want - acc) / seg;
                out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                want += step;
            }
            acc += seg;
            i += 1;
        }
        if out.len() < 3 {
            return self.clone();
        }
        Poly::new(out)
    }

    /// One pass of curvature smoothing that keeps the enclosed area roughly
    /// constant (a plain average shrinks a convex ring every pass).
    pub fn smooth(&self, passes: usize) -> Poly {
        let n = self.pts.len();
        if n < 5 || passes == 0 {
            return self.clone();
        }
        let mut pts = self.pts.clone();
        for _ in 0..passes {
            let src = pts.clone();
            for i in 0..n {
                let a = src[(i + n - 1) % n];
                let b = src[i];
                let c = src[(i + 1) % n];
                pts[i] = [
                    0.25 * a[0] + 0.5 * b[0] + 0.25 * c[0],
                    0.25 * a[1] + 0.5 * b[1] + 0.25 * c[1],
                ];
            }
        }
        let mut out = Poly::new(pts);
        // Restore the area lost to smoothing by scaling about the centroid.
        let (a0, a1) = (self.area(), out.area());
        if a0 > 1e-9 && a1 > 1e-9 {
            let c = out.centroid();
            out.scale_about(c, (a0 / a1).sqrt());
        }
        out
    }

    /// A circle - the brush stamp, and the shape every test starts from.
    pub fn circle(c: Pt, r: f64, n: usize) -> Poly {
        let n = n.max(8);
        Poly::new(
            (0..n)
                .map(|i| {
                    let a = i as f64 / n as f64 * std::f64::consts::TAU;
                    [c[0] + r * a.cos(), c[1] + r * a.sin()]
                })
                .collect(),
        )
    }

    /// The outline of a capsule: a stroke of radius `r` swept from `a` to
    /// `b`, so a fast drag stays gap-free the way the voxel brush does.
    pub fn capsule(a: Pt, b: Pt, r: f64, n: usize) -> Poly {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            return Poly::circle(a, r, n);
        }
        let n = n.max(8) / 2 * 2;
        let ang = dy.atan2(dx);
        let mut pts = Vec::with_capacity(n + 2);
        // Half circle around b, then around a.
        for i in 0..=n / 2 {
            let t = ang - std::f64::consts::FRAC_PI_2
                + i as f64 / (n as f64 / 2.0) * std::f64::consts::PI;
            pts.push([b[0] + r * t.cos(), b[1] + r * t.sin()]);
        }
        for i in 0..=n / 2 {
            let t = ang
                + std::f64::consts::FRAC_PI_2
                + i as f64 / (n as f64 / 2.0) * std::f64::consts::PI;
            pts.push([a[0] + r * t.cos(), a[1] + r * t.sin()]);
        }
        Poly::new(pts)
    }

    /// The same capsule, built in millimetres on a lattice whose two
    /// in-plane axes are `mm[0]` and `mm[1]` millimetres per unit.
    ///
    /// A brush is a disc *on the patient*, so on a sagittal slice of a 1 mm x
    /// 3 mm lattice it has to come out as an ellipse on the lattice. Building
    /// the shape in millimetres and dividing back is the whole trick.
    pub fn capsule_mm(a: Pt, b: Pt, r_mm: f64, mm: Pt, n: usize) -> Poly {
        let (su, sv) = (mm[0].max(1e-6), mm[1].max(1e-6));
        let mut p = Poly::capsule(
            [a[0] * su, a[1] * sv],
            [b[0] * su, b[1] * sv],
            r_mm.max(1e-3),
            n,
        );
        for q in &mut p.pts {
            q[0] /= su;
            q[1] /= sv;
        }
        p
    }

    /// Does the ring cross itself? A freehand stroke often does, and the
    /// answer decides whether a boolean has to run at all.
    pub fn self_intersects(&self) -> bool {
        let n = self.pts.len();
        if n < 4 {
            return false;
        }
        for i in 0..n {
            let a0 = self.pts[i];
            let a1 = self.pts[(i + 1) % n];
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 {
                    continue; // adjacent through the closing edge
                }
                let b0 = self.pts[j];
                let b1 = self.pts[(j + 1) % n];
                if segments_cross(a0, a1, b0, b1) {
                    return true;
                }
            }
        }
        false
    }
}

/// Douglas-Peucker on an open polyline.
fn dp(pts: &[Pt], tol: f64) -> Vec<Pt> {
    if pts.len() < 3 {
        return pts.to_vec();
    }
    let (first, last) = (pts[0], pts[pts.len() - 1]);
    let mut worst = 0.0;
    let mut idx = 0;
    for (i, p) in pts.iter().enumerate().take(pts.len() - 1).skip(1) {
        let d = point_line_distance(*p, first, last);
        if d > worst {
            worst = d;
            idx = i;
        }
    }
    if worst > tol {
        let mut left = dp(&pts[..=idx], tol);
        let right = dp(&pts[idx..], tol);
        left.pop();
        left.extend(right);
        left
    } else {
        vec![first, last]
    }
}

fn point_line_distance(p: Pt, a: Pt, b: Pt) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-18 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len2.sqrt()
}

fn segments_cross(a0: Pt, a1: Pt, b0: Pt, b1: Pt) -> bool {
    fn orient(p: Pt, q: Pt, r: Pt) -> f64 {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    }
    let d1 = orient(a0, a1, b0);
    let d2 = orient(a0, a1, b1);
    let d3 = orient(b0, b1, a0);
    let d4 = orient(b0, b1, a1);
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// A closed centripetal Catmull-Rom spline through the given points, with
/// `per_segment` points between two of them - the curve the spline tool
/// draws, and the only reason it differs from the polygon tool.
pub fn spline_points(pts: &[[f64; 2]], per_segment: usize) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(n * per_segment);
    for i in 0..n {
        let p0 = pts[(i + n - 1) % n];
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let p3 = pts[(i + 2) % n];
        // Centripetal parameterisation: no cusps, no self-intersection from
        // a doubled click.
        let t = |a: [f64; 2], b: [f64; 2], t0: f64| -> f64 {
            let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            t0 + d.sqrt().max(1e-6)
        };
        let (t0, t1) = (0.0, t(p0, p1, 0.0));
        let (t2, t3) = (t(p1, p2, t1), 0.0);
        let t3 = t(p2, p3, t2).max(t3);
        for s in 0..per_segment {
            let tt = t1 + (t2 - t1) * (s as f64 / per_segment as f64);
            let a1 = lerp2(p0, p1, (t1 - tt) / (t1 - t0), (tt - t0) / (t1 - t0));
            let a2 = lerp2(p1, p2, (t2 - tt) / (t2 - t1), (tt - t1) / (t2 - t1));
            let a3 = lerp2(p2, p3, (t3 - tt) / (t3 - t2), (tt - t2) / (t3 - t2));
            let b1 = lerp2(a1, a2, (t2 - tt) / (t2 - t0), (tt - t0) / (t2 - t0));
            let b2 = lerp2(a2, a3, (t3 - tt) / (t3 - t1), (tt - t1) / (t3 - t1));
            out.push(lerp2(b1, b2, (t2 - tt) / (t2 - t1), (tt - t1) / (t2 - t1)));
        }
    }
    out
}

#[inline]
fn lerp2(a: [f64; 2], b: [f64; 2], wa: f64, wb: f64) -> [f64; 2] {
    [a[0] * wa + b[0] * wb, a[1] * wa + b[1] * wb]
}

/// The *auto* rule of the drawing tools: a ring that crosses an existing
/// contour and starts outside it cuts; anything else adds.
pub fn auto_cuts(region: &Region, ring: &Poly) -> bool {
    !ring.is_empty() && region.crosses(ring) && !region.contains(ring.pts[0])
}

// ---------------------------------------------------------------------------
// Region - the polygons of one slice
// ---------------------------------------------------------------------------

/// The contours of one slice of one ROI, filled by the even-odd rule: a ring
/// inside another is a hole, a ring inside that is an island again.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Region {
    pub rings: Vec<Poly>,
}

impl Region {
    pub fn new() -> Region {
        Region::default()
    }

    pub fn from_ring(p: Poly) -> Region {
        Region { rings: vec![p] }
    }

    pub fn is_empty(&self) -> bool {
        self.rings.iter().all(|r| r.is_empty())
    }

    pub fn bbox(&self) -> [f64; 4] {
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for r in &self.rings {
            let rb = r.bbox();
            b[0] = b[0].min(rb[0]);
            b[1] = b[1].min(rb[1]);
            b[2] = b[2].max(rb[2]);
            b[3] = b[3].max(rb[3]);
        }
        b
    }

    /// Even-odd: inside an odd number of rings.
    pub fn contains(&self, p: Pt) -> bool {
        self.rings.iter().filter(|r| r.contains(p)).count() % 2 == 1
    }

    /// Enclosed area in square voxels. Exact for a set of rings that do not
    /// cross each other - which is what every operation here returns and what
    /// a structure set contains.
    pub fn area(&self) -> f64 {
        let depths = self.depths();
        self.rings
            .iter()
            .zip(&depths)
            .map(|(r, d)| if d % 2 == 0 { r.area() } else { -r.area() })
            .sum::<f64>()
            .max(0.0)
    }

    /// Nesting depth of every ring: how many other rings contain it.
    pub fn depths(&self) -> Vec<usize> {
        self.rings
            .iter()
            .enumerate()
            .map(|(i, r)| {
                self.rings
                    .iter()
                    .enumerate()
                    .filter(|(j, other)| *j != i && ring_inside(r, other))
                    .count()
            })
            .collect()
    }

    /// Drop degenerate rings and orient by nesting depth - outer rings
    /// counter-clockwise, holes clockwise. Cosmetic for the even-odd fill,
    /// but it is what makes [`Region::area`] exact and what other systems
    /// expect to read back.
    pub fn normalize(&mut self) {
        self.rings
            .retain(|r| !r.is_empty() && r.area() > MIN_RING_AREA);
        let depths = self.depths();
        for (r, d) in self.rings.iter_mut().zip(&depths) {
            r.orient(d % 2 == 0);
        }
    }

    pub fn simplify(&mut self, tol: f64) {
        for r in &mut self.rings {
            *r = r.simplify(tol);
        }
        self.rings.retain(|r| !r.is_empty());
    }

    pub fn smooth(&mut self, passes: usize) {
        for r in &mut self.rings {
            *r = r.smooth(passes);
        }
    }

    /// Remove every hole: keep only rings at even nesting depth.
    pub fn remove_holes(&mut self) {
        let depths = self.depths();
        let mut keep = depths.iter().map(|d| d % 2 == 0);
        self.rings.retain(|_| keep.next().unwrap_or(true));
    }

    /// Drop rings (and their holes) whose area is under `min_area`, in
    /// square voxels.
    pub fn drop_smaller_than(&mut self, min_area: f64) {
        let depths = self.depths();
        let small: Vec<bool> = self
            .rings
            .iter()
            .zip(&depths)
            .map(|(r, d)| d % 2 == 0 && r.area() < min_area)
            .collect();
        // A hole inside a dropped ring goes with it.
        let doomed: Vec<Poly> = self
            .rings
            .iter()
            .zip(&small)
            .filter(|(_, s)| **s)
            .map(|(r, _)| r.clone())
            .collect();
        let mut idx = 0;
        self.rings.retain(|r| {
            let s = small[idx];
            idx += 1;
            if s {
                return false;
            }
            !doomed.iter().any(|d| ring_inside(r, d))
        });
    }

    /// Keep only the largest outer ring (with its holes).
    pub fn keep_largest(&mut self) {
        let depths = self.depths();
        let best = self
            .rings
            .iter()
            .zip(&depths)
            .enumerate()
            .filter(|(_, (_, d))| **d == 0)
            .max_by(|a, b| {
                a.1 .0
                    .area()
                    .partial_cmp(&b.1 .0.area())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        let Some(best) = best else {
            self.rings.clear();
            return;
        };
        let outer = self.rings[best].clone();
        let mut i = 0;
        self.rings.retain(|r| {
            let keep = i == best || ring_inside(r, &outer);
            i += 1;
            keep
        });
    }

    /// Boolean against another region, by local supersampled rasterization
    /// (see the module documentation for why).
    pub fn boolean(&self, other: &Region, op: BoolOp) -> Region {
        raster_boolean(self, other, op, BOOL_SS)
    }

    /// [`Self::boolean`] for an edit: a union, difference or symmetric
    /// difference with something small next to `self`, which changes
    /// nothing outside `other`'s box. Only a window around that box is
    /// rasterized and traced, on the same cells, and spliced into the
    /// outline outside it, whose vertices are kept as they are (see *Booleans
    /// within a window* in the source); a stroke of the brush along a body
    /// outline costs the brush's size, not the body's. Falls back on
    /// [`Self::boolean`] when the window would not be much smaller, or the
    /// splice cannot be made.
    pub fn boolean_local(&self, other: &Region, op: BoolOp) -> Region {
        window_boolean(self, other, op, BOOL_SS).unwrap_or_else(|| self.boolean(other, op))
    }

    /// Add a ring to the region.
    ///
    /// A ring that crosses nothing and encloses no existing ring is handled
    /// *exactly*, with no boolean at all: under the even-odd rule adding it
    /// simply toggles what its interior does, so it is appended when the
    /// interior is currently empty (a new island, or a hole being filled) and
    /// ignored when the interior is already filled. Every other case goes
    /// through the boolean. The point of the distinction is that an ordinary
    /// stroke leaves the vertices of everything it did not reach untouched.
    pub fn add_ring(&mut self, ring: &Poly) {
        if ring.is_empty() {
            return;
        }
        if let Some(inside) = self.exact_toggle(ring) {
            if !inside {
                let mut r = ring.clone();
                r.orient(true);
                self.rings.push(r);
                self.normalize();
            }
            return;
        }
        *self = self.boolean_local(&Region::from_ring(ring.clone()), BoolOp::Union);
    }

    /// Cut a ring out of the region, with the same exact fast paths: a ring
    /// wholly inside the filled area becomes a hole, one over empty space
    /// changes nothing.
    pub fn subtract_ring(&mut self, ring: &Poly) {
        if ring.is_empty() || self.is_empty() {
            return;
        }
        if let Some(inside) = self.exact_toggle(ring) {
            if inside {
                let mut r = ring.clone();
                r.orient(true);
                self.rings.push(r);
                self.normalize();
            }
            return;
        }
        *self = self.boolean_local(&Region::from_ring(ring.clone()), BoolOp::Subtract);
    }

    /// `Some(filled)` when `ring` can be applied by even-odd toggling alone:
    /// it is simple, crosses no existing ring and encloses none, so its whole
    /// interior is on one side. `filled` says which.
    fn exact_toggle(&self, ring: &Poly) -> Option<bool> {
        if ring.self_intersects() || self.touches(ring) {
            return None;
        }
        for r in &self.rings {
            if ring_inside(r, ring) {
                return None; // it swallows an existing ring
            }
        }
        Some(self.contains(interior_point(ring)))
    }

    /// Does any edge of `ring` cross any edge of the region? This is the
    /// question the *auto* drawing mode asks: a stroke that crosses
    /// the outline modifies it, one that does not starts a new contour.
    pub fn crosses(&self, ring: &Poly) -> bool {
        self.touches(ring)
    }

    /// Does any edge of `ring` cross any edge of the region?
    fn touches(&self, ring: &Poly) -> bool {
        let rb = ring.bbox();
        for r in &self.rings {
            let b = r.bbox();
            if b[2] < rb[0] || b[0] > rb[2] || b[3] < rb[1] || b[1] > rb[3] {
                continue;
            }
            let (n, m) = (r.pts.len(), ring.pts.len());
            for i in 0..n {
                let a0 = r.pts[i];
                let a1 = r.pts[(i + 1) % n];
                for j in 0..m {
                    let b0 = ring.pts[j];
                    let b1 = ring.pts[(j + 1) % m];
                    if segments_cross(a0, a1, b0, b1) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Push the outline around: every vertex within `radius_mm` of `from`
    /// moves by the drag vector `from -> to`, scaled by a cosine falloff, so
    /// the curve deforms smoothly instead of developing a corner.
    ///
    /// `mm` gives the millimetres per unit of each in-plane axis, because the
    /// radius the user set is a distance on the patient and the ring lives on
    /// the lattice. Returns whether anything moved.
    pub fn nudge(&mut self, from: Pt, to: Pt, radius_mm: f64, mm: Pt) -> bool {
        let (du, dv) = (to[0] - from[0], to[1] - from[1]);
        if radius_mm <= 0.0 || (du.abs() < 1e-9 && dv.abs() < 1e-9) {
            return false;
        }
        let (su, sv) = (mm[0].max(1e-6), mm[1].max(1e-6));
        let mut moved = false;
        for r in &mut self.rings {
            for p in &mut r.pts {
                let d = (((p[0] - from[0]) * su).powi(2) + ((p[1] - from[1]) * sv).powi(2)).sqrt();
                if d >= radius_mm {
                    continue;
                }
                let w = 0.5 * (1.0 + (std::f64::consts::PI * d / radius_mm).cos());
                p[0] += du * w;
                p[1] += dv * w;
                moved = true;
            }
        }
        if moved {
            self.normalize();
        }
        moved
    }

    /// The ring the point falls in, as an index into `rings` - the pick a
    /// "delete this contour" click needs. Prefers the innermost ring.
    pub fn ring_at(&self, p: Pt) -> Option<usize> {
        self.rings
            .iter()
            .enumerate()
            .filter(|(_, r)| r.contains(p))
            .min_by(|a, b| {
                a.1.area()
                    .partial_cmp(&b.1.area())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// The cells of a `w` x `h` raster whose centre the region covers, one
    /// byte a cell (row-major, cell `(i, j)` centred on `(i, j)`) - the even-odd
    /// fill [`crate::segmentation::rasterize_roi`] and [`Stack::rasterize`]
    /// use, on one slice. This is how a curve drawn with the contour tools
    /// lands in a segmentation's mask: the voxels whose centre it encloses.
    pub fn rasterize(&self, w: usize, h: usize) -> Vec<u8> {
        let mut out = vec![0u8; w * h];
        if self.is_empty() || w == 0 || h == 0 {
            return out;
        }
        // Only the rows the rings can reach; the rest stay empty.
        let b = self.bbox();
        let j0 = (b[1].floor().max(0.0) as usize).min(h);
        let j1 = ((b[3].ceil() + 1.0).max(0.0) as usize).min(h);
        fill_region_rows(self, -0.5, -0.5, w, j0..j1, 1.0, |j, i| out[j * w + i] = 1);
        out
    }
}

/// The outline of a binary slice (`w` x `h`, row-major, cell `(i, j)`
/// centred on `(i, j)`) as a region: the marching-squares trace
/// [`Stack::from_mask`] makes of every slice, so a mask slice read here is
/// the contour a generator would have made of it. Filled back with
/// [`Region::rasterize`] it gives the same cells: the outline runs half a
/// cell from every centre, on the right side of each.
///
/// Only the box the filled cells occupy is traced, so an organ on a large
/// slice costs its own size.
pub fn region_of_mask(mask: &[u8], w: usize, h: usize) -> Region {
    let (mut lo, mut hi) = ([usize::MAX; 2], [0usize; 2]);
    for j in 0..h {
        let row = &mask[j * w..(j + 1) * w];
        if let (Some(a), Some(b)) = (
            row.iter().position(|&m| m != 0),
            row.iter().rposition(|&m| m != 0),
        ) {
            lo = [lo[0].min(a), lo[1].min(j)];
            hi = [hi[0].max(b), hi[1].max(j)];
        }
    }
    if lo[0] == usize::MAX {
        return Region::new();
    }
    let (bw, bh) = (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1);
    let mut field = PaddedField::new(bw, bh);
    for j in 0..bh {
        for i in 0..bw {
            if mask[(j + lo[1]) * w + i + lo[0]] != 0 {
                field.set(i, j);
            }
        }
    }
    field.trace_region(lo[0] as f64, lo[1] as f64)
}

/// Is `inner` nested inside `outer`? Valid whenever the two do not cross,
/// which is the invariant every region here maintains: then one boundary
/// point settles it, and a boundary point - unlike the centroid - is never
/// swallowed by a ring nested *inside* `inner`.
fn ring_inside(inner: &Poly, outer: &Poly) -> bool {
    !inner.is_empty() && !outer.is_empty() && outer.contains(inner.pts[0])
}

/// A point strictly inside a ring: the midpoint of the shortest interior
/// horizontal span at the ring's centroid height, so a crescent works too.
fn interior_point(r: &Poly) -> Pt {
    let c = r.centroid();
    if r.contains(c) {
        return c;
    }
    let n = r.pts.len();
    let y = c[1];
    let mut xs: Vec<f64> = Vec::new();
    for i in 0..n {
        let a = r.pts[i];
        let b = r.pts[(i + 1) % n];
        if (a[1] <= y) != (b[1] <= y) {
            let t = (y - a[1]) / (b[1] - a[1]);
            xs.push(a[0] + t * (b[0] - a[0]));
        }
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if xs.len() >= 2 {
        // The widest span is the safest one to sit in.
        let mut best = (xs[0] + xs[1]) * 0.5;
        let mut wide = xs[1] - xs[0];
        for pair in xs.as_chunks::<2>().0 {
            if pair[1] - pair[0] > wide {
                wide = pair[1] - pair[0];
                best = (pair[0] + pair[1]) * 0.5;
            }
        }
        return [best, y];
    }
    c
}

/// Fill a region into a boolean raster whose cell `(i, j)` has its centre at
/// `(x0 + (i + 0.5) * step, y0 + (j + 0.5) * step)`.
fn fill_region(region: &Region, x0: f64, y0: f64, w: usize, h: usize, step: f64, out: &mut [bool]) {
    fill_region_rows(region, x0, y0, w, 0..h, step, |j, i| out[j * w + i] = true);
}

/// [`fill_region`] over the rows `rows` of a raster `w` cells wide, handing
/// each filled cell `(row, column)` to `set`: the same arithmetic, so a
/// caller that only wants a box of the raster gets exactly the cells the
/// whole raster has there.
fn fill_region_rows(
    region: &Region,
    x0: f64,
    y0: f64,
    w: usize,
    rows: std::ops::Range<usize>,
    step: f64,
    mut set: impl FnMut(usize, usize),
) {
    let mut crossings: Vec<f64> = Vec::new();
    for j in rows {
        let y = y0 + (j as f64 + 0.5) * step;
        crossings.clear();
        for r in &region.rings {
            let n = r.pts.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let a = r.pts[i];
                let b = r.pts[(i + 1) % n];
                if (a[1] <= y) != (b[1] <= y) {
                    let t = (y - a[1]) / (b[1] - a[1]);
                    crossings.push(a[0] + t * (b[0] - a[0]));
                }
            }
        }
        if crossings.len() < 2 {
            continue;
        }
        crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for span in crossings.as_chunks::<2>().0 {
            let from = ((span[0] - x0) / step - 0.5).ceil().max(0.0) as i64;
            let to = (((span[1] - x0) / step - 0.5).floor() as i64).min(w as i64 - 1);
            for i in from..=to {
                if i >= 0 {
                    set(j, i as usize);
                }
            }
        }
    }
}

fn raster_boolean(a: &Region, b: &Region, op: BoolOp, ss: usize) -> Region {
    if a.is_empty() && b.is_empty() {
        return Region::new();
    }
    let (ba, bb) = (a.bbox(), b.bbox());
    // Which part of the plane can hold the answer.
    let win = match op {
        BoolOp::Subtract => ba,
        BoolOp::Intersect => [
            ba[0].max(bb[0]),
            ba[1].max(bb[1]),
            ba[2].min(bb[2]),
            ba[3].min(bb[3]),
        ],
        _ => [
            ba[0].min(bb[0]),
            ba[1].min(bb[1]),
            ba[2].max(bb[2]),
            ba[3].max(bb[3]),
        ],
    };
    if !(win[2] > win[0] && win[3] > win[1]) {
        return match op {
            BoolOp::Subtract => a.clone(),
            BoolOp::Union | BoolOp::Xor if b.is_empty() => a.clone(),
            BoolOp::Union | BoolOp::Xor if a.is_empty() => b.clone(),
            BoolOp::Intersect => Region::new(),
            _ => {
                let mut r = a.clone();
                r.rings.extend(b.rings.iter().cloned());
                r.normalize();
                r
            }
        };
    }
    let (x0, y0) = (win[0].floor() - 1.0, win[1].floor() - 1.0);
    let (x1, y1) = (win[2].ceil() + 1.0, win[3].ceil() + 1.0);
    let mut step = 1.0 / ss as f64;
    let mut w = ((x1 - x0) / step).ceil() as usize + 1;
    let mut h = ((y1 - y0) / step).ceil() as usize + 1;
    // A pathological bounding box coarsens the raster rather than the machine.
    while w.saturating_mul(h) > BOOL_MAX_CELLS && step < 1.0 {
        step *= 2.0;
        w = ((x1 - x0) / step).ceil() as usize + 1;
        h = ((y1 - y0) / step).ceil() as usize + 1;
    }
    let mut ma = vec![false; w * h];
    let mut mb = vec![false; w * h];
    fill_region(a, x0, y0, w, h, step, &mut ma);
    fill_region(b, x0, y0, w, h, step, &mut mb);

    // Padded scalar field for marching squares: cell (i, j) of the raster is
    // field cell (i + 1, j + 1), so a run touching the border still closes.
    let (pw, ph) = (w + 2, h + 2);
    let mut field = vec![0.0f32; pw * ph];
    let mut any = false;
    for j in 0..h {
        for i in 0..w {
            let (x, y) = (ma[j * w + i], mb[j * w + i]);
            let v = match op {
                BoolOp::Union => x || y,
                BoolOp::Intersect => x && y,
                BoolOp::Subtract => x && !y,
                BoolOp::Xor => x != y,
            };
            if v {
                field[(j + 1) * pw + i + 1] = 1.0;
                any = true;
            }
        }
    }
    if !any {
        return Region::new();
    }
    let tol = step * 0.75;
    let mut out = Region::new();
    for pts in stitch_loops(&render::marching_squares(&field, pw, ph, 0.5)) {
        let ring = Poly::new(
            pts.iter()
                .map(|p| {
                    [
                        x0 + (p[0] as f64 - 0.5) * step,
                        y0 + (p[1] as f64 - 0.5) * step,
                    ]
                })
                .collect(),
        );
        let ring = ring.simplify(tol);
        if !ring.is_empty() {
            out.rings.push(ring);
        }
    }
    out.normalize();
    out
}

// ---------------------------------------------------------------------------
// Booleans within a window
// ---------------------------------------------------------------------------
//
// A brush stroke changes a slice only where the brush went, but the raster
// boolean above re-rasterizes the bounding box of both operands: for a stroke
// along a body outline that is a few million fine cells per mouse sample,
// and the whole outline re-traced. The windowed boolean rasterizes only a
// box around what changes - the same lattice of fine cells, so the cells it
// fills are the ones the whole raster has there - and splices the trace of
// that box into the outline outside it, which it keeps vertex for vertex.
//
// The splice: the box's cells, traced by marching squares between their
// centres, give closed loops (wholly inside) and chains that start and end on
// the rectangle through the outermost centres. The outline as it was gives
// pieces that leave the rectangle and come back. The cells on the
// rectangle's border are filled from the outline as it was, so every place
// where a piece meets the rectangle has a chain end beside it, of the same
// kind (leaving or entering), in the same order around the rectangle; the
// two are matched in that order and joined. Whatever does not match - an
// outline that crosses the rectangle twice between two centres, a vertex
// grazing it - and any result whose area disagrees with the cells' is
// refused, and the caller falls back on the whole raster.

/// A box of the fine raster of the booleans: cell `(i, j)` is centred on
/// `(x0 + (i + 0.5) step, y0 + (j + 0.5) step)`, with `x0`, `y0` whole
/// numbers, so every window of every boolean, and [`raster_boolean`]'s own
/// box, sit on one lattice of cells.
#[derive(Clone, Copy, Debug)]
struct Window {
    x0: f64,
    y0: f64,
    w: usize,
    h: usize,
    step: f64,
}

/// A point of a window's trace, in half cells from its first centre: centre
/// `(i, j)` is `(2i, 2j)`, the midpoint between two neighbours odd in one.
type Key = (i64, i64);

impl Window {
    /// The cells over the box `b` (`[x0, y0, x1, y1]`, in-plane voxel
    /// coordinates) widened by `margin` voxels, at `ss` cells per voxel.
    /// `None` for an empty or non-finite box, or one past `max_cells`.
    fn around(b: [f64; 4], margin: f64, ss: usize, max_cells: usize) -> Option<Window> {
        if b.iter().any(|v| !v.is_finite()) || b[2] < b[0] || b[3] < b[1] {
            return None;
        }
        let (x0, y0) = ((b[0] - margin).floor(), (b[1] - margin).floor());
        let (x1, y1) = ((b[2] + margin).ceil(), (b[3] + margin).ceil());
        let w = ((x1 - x0) as usize) * ss;
        let h = ((y1 - y0) as usize) * ss;
        (w >= 4 && h >= 4 && w.saturating_mul(h) <= max_cells).then_some(Window {
            x0,
            y0,
            w,
            h,
            step: 1.0 / ss as f64,
        })
    }

    /// In-plane coordinates of a trace point.
    fn at(&self, k: Key) -> Pt {
        [
            self.x0 + (k.0 as f64 + 1.0) * 0.5 * self.step,
            self.y0 + (k.1 as f64 + 1.0) * 0.5 * self.step,
        ]
    }

    /// The rectangle through the outermost cell centres.
    fn rect(&self) -> [f64; 4] {
        let a = self.at((0, 0));
        let b = self.at((2 * (self.w as i64 - 1), 2 * (self.h as i64 - 1)));
        [a[0], a[1], b[0], b[1]]
    }

    /// The box in voxel coordinates, `[x0, y0, x1, y1]`.
    fn bounds(&self) -> [f64; 4] {
        [
            self.x0,
            self.y0,
            self.x0 + self.w as f64 * self.step,
            self.y0 + self.h as f64 * self.step,
        ]
    }

    /// The cells whose centre `region` covers, by the even-odd fill every
    /// raster here uses. Only the rings that reach the box take part: a ring
    /// wholly to one side adds its crossings in pairs, which changes no
    /// cell's parity.
    fn fill(&self, region: &Region) -> Vec<bool> {
        let b = self.bounds();
        let near = Region {
            rings: region
                .rings
                .iter()
                .filter(|r| {
                    let rb = r.bbox();
                    !(rb[2] < b[0] || rb[0] > b[2] || rb[3] < b[1] || rb[1] > b[3])
                })
                .cloned()
                .collect(),
        };
        let mut out = vec![false; self.w * self.h];
        fill_region(&near, self.x0, self.y0, self.w, self.h, self.step, &mut out);
        out
    }
}

/// The directed outline of a window's cells: marching squares between the
/// centres, filled cells on the left of every segment (as an outer ring
/// wound counter-clockwise has its inside), saddles split as
/// [`render::marching_squares`] splits them on a binary field (diagonal
/// neighbours apart). Chains run from the window's border to its border;
/// loops close inside it. `None` if the segments do not link up, which a
/// binary field never does.
#[allow(clippy::type_complexity)]
fn trace_window(cells: &[bool], w: usize, h: usize) -> Option<(Vec<Vec<Key>>, Vec<Vec<Key>>)> {
    use std::collections::HashMap;
    let c = |i: usize, j: usize| cells[j * w + i];
    let mut segs: Vec<(Key, Key)> = Vec::new();
    for j in 0..h - 1 {
        for i in 0..w - 1 {
            let (v00, v10, v11, v01) = (c(i, j), c(i + 1, j), c(i + 1, j + 1), c(i, j + 1));
            let case = v00 as u8 | (v10 as u8) << 1 | (v11 as u8) << 2 | (v01 as u8) << 3;
            if case == 0 || case == 15 {
                continue;
            }
            let (x, y) = (2 * i as i64, 2 * j as i64);
            let top = (x + 1, y);
            let bottom = (x + 1, y + 2);
            let left = (x, y + 1);
            let right = (x + 2, y + 1);
            let (c00, c10, c11, c01) = ((x, y), (x + 2, y), (x + 2, y + 2), (x, y + 2));
            // Each segment with a corner on one side of it, to orient by.
            let mut push = |p: Key, q: Key, corner: Key, filled: bool| {
                let n = (-(q.1 - p.1), q.0 - p.0);
                let d = (2 * corner.0 - p.0 - q.0, 2 * corner.1 - p.1 - q.1);
                if (d.0 * n.0 + d.1 * n.1 > 0) == filled {
                    segs.push((p, q));
                } else {
                    segs.push((q, p));
                }
            };
            match case {
                1 | 14 => push(left, top, c00, v00),
                2 | 13 => push(top, right, c10, v10),
                4 | 11 => push(right, bottom, c11, v11),
                8 | 7 => push(bottom, left, c01, v01),
                3 | 12 => push(left, right, c00, v00),
                6 | 9 => push(top, bottom, c00, v00),
                // The saddle: on a binary field the centre average is the
                // level, and `marching_squares` cuts off the filled corners.
                5 => {
                    push(left, top, c00, v00);
                    push(right, bottom, c11, v11);
                }
                _ => {
                    push(top, right, c10, v10);
                    push(bottom, left, c01, v01);
                }
            }
        }
    }
    let (bx, by) = (2 * (w as i64 - 1), 2 * (h as i64 - 1));
    let on_border = |k: Key| k.0 == 0 || k.1 == 0 || k.0 == bx || k.1 == by;
    let mut from: HashMap<Key, usize> = HashMap::with_capacity(segs.len());
    for (si, s) in segs.iter().enumerate() {
        if from.insert(s.0, si).is_some() {
            return None;
        }
    }
    let mut used = vec![false; segs.len()];
    let mut chains = Vec::new();
    for si in 0..segs.len() {
        if used[si] || !on_border(segs[si].0) {
            continue;
        }
        let mut pts = vec![segs[si].0];
        let mut cur = si;
        loop {
            used[cur] = true;
            let end = segs[cur].1;
            pts.push(end);
            match from.get(&end) {
                Some(&n) if !used[n] => cur = n,
                Some(_) => return None,
                None if on_border(end) => break,
                None => return None,
            }
        }
        chains.push(pts);
    }
    let mut loops = Vec::new();
    for si in 0..segs.len() {
        if used[si] {
            continue;
        }
        let start = segs[si].0;
        let mut pts = Vec::new();
        let mut cur = si;
        loop {
            used[cur] = true;
            pts.push(segs[cur].0);
            let end = segs[cur].1;
            if end == start {
                break;
            }
            match from.get(&end) {
                Some(&n) if !used[n] => cur = n,
                _ => return None,
            }
        }
        loops.push(pts);
    }
    Some((chains, loops))
}

/// Where a ring meets a rectangle's border: on edge `edge` of the ring,
/// `enter` going in, at `pt` on the border.
#[derive(Clone, Copy, Debug)]
struct Crossing {
    edge: usize,
    enter: bool,
    pt: Pt,
}

/// Every place `ring` passes from outside the open rectangle `r` to inside
/// it or back, in ring order. A vertex on the border counts as outside, and
/// an edge that only runs along the border or touches a corner crosses
/// nothing.
fn rect_crossings(ring: &Poly, r: [f64; 4]) -> Vec<Crossing> {
    let inside = |p: Pt| p[0] > r[0] && p[0] < r[2] && p[1] > r[1] && p[1] < r[3];
    let n = ring.pts.len();
    let mut out = Vec::new();
    for k in 0..n {
        let (p, q) = (ring.pts[k], ring.pts[(k + 1) % n]);
        let d = [q[0] - p[0], q[1] - p[1]];
        // Liang-Barsky against the closed rectangle.
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        let mut empty = false;
        for (num, den) in [
            (p[0] - r[0], -d[0]),
            (r[2] - p[0], d[0]),
            (p[1] - r[1], -d[1]),
            (r[3] - p[1], d[1]),
        ] {
            if den == 0.0 {
                if num < 0.0 {
                    empty = true;
                    break;
                }
            } else {
                let t = num / den;
                if den < 0.0 {
                    t0 = t0.max(t);
                } else {
                    t1 = t1.min(t);
                }
            }
        }
        if empty || t1 <= t0 {
            continue;
        }
        let mid = (t0 + t1) * 0.5;
        if !inside([p[0] + d[0] * mid, p[1] + d[1] * mid]) {
            continue;
        }
        // On the border, snapped onto the side it is closest to.
        let snap = |t: f64| -> Pt {
            let mut x = [p[0] + d[0] * t, p[1] + d[1] * t];
            let ds = [
                (x[0] - r[0]).abs(),
                (x[0] - r[2]).abs(),
                (x[1] - r[1]).abs(),
                (x[1] - r[3]).abs(),
            ];
            let side = (0..4)
                .min_by(|&a, &b| {
                    ds[a]
                        .partial_cmp(&ds[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            match side {
                0 => x[0] = r[0],
                1 => x[0] = r[2],
                2 => x[1] = r[1],
                _ => x[1] = r[3],
            }
            x[0] = x[0].clamp(r[0], r[2]);
            x[1] = x[1].clamp(r[1], r[3]);
            x
        };
        if !inside(p) {
            out.push(Crossing {
                edge: k,
                enter: true,
                pt: snap(t0),
            });
        }
        if !inside(q) {
            out.push(Crossing {
                edge: k,
                enter: false,
                pt: snap(t1),
            });
        }
    }
    out
}

/// How far round the border of rectangle `r` the point `p` on it lies:
/// along the top from its left end, down the right, back along the bottom,
/// up the left.
fn perimeter_at(r: [f64; 4], p: Pt) -> f64 {
    let (w, h) = (r[2] - r[0], r[3] - r[1]);
    let ds = [
        (p[1] - r[1]).abs(),
        (p[0] - r[2]).abs(),
        (p[1] - r[3]).abs(),
        (p[0] - r[0]).abs(),
    ];
    let side = (0..4)
        .min_by(|&a, &b| {
            ds[a]
                .partial_cmp(&ds[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    match side {
        0 => p[0] - r[0],
        1 => w + (p[1] - r[1]),
        2 => w + h + (r[2] - p[0]),
        _ => 2.0 * w + h + (r[3] - p[1]),
    }
}

/// Pair two lists of positions round a border of length `len`, each sorted:
/// the same count, in the same cyclic order, each pair within `tol`. Returns
/// for every entry of `a` the index into `b`.
fn match_round(a: &[f64], b: &[f64], len: f64, tol: f64) -> Option<Vec<usize>> {
    let n = a.len();
    if n != b.len() {
        return None;
    }
    if n == 0 {
        return Some(Vec::new());
    }
    let dist = |x: f64, y: f64| {
        let d = (x - y).rem_euclid(len);
        d.min(len - d)
    };
    // The first of either list may sit just past the corner the other's
    // last one is before: a turn of one either way is allowed.
    for shift in [0, 1, n - 1] {
        let ok = (0..n).all(|i| dist(a[i], b[(i + shift) % n]) <= tol);
        if ok {
            return Some((0..n).map(|i| (i + shift) % n).collect());
        }
    }
    None
}

/// `current` with its part inside window `win` replaced by the outline of
/// `cells` (the window's raster of the answer, `cur` the window's raster of
/// `current`): the splice described above. `None` when the two cannot be
/// matched, or the result's area is not the cells' answer.
fn splice_window(current: &Region, win: &Window, cells: &[bool], cur: &[bool]) -> Option<Region> {
    let (w, h) = (win.w, win.h);
    // The border comes from the outline as it was, so the chains end where
    // its pieces do.
    let mut cells = cells.to_vec();
    for i in 0..w {
        cells[i] = cur[i];
        cells[(h - 1) * w + i] = cur[(h - 1) * w + i];
    }
    for j in 0..h {
        cells[j * w] = cur[j * w];
        cells[j * w + w - 1] = cur[j * w + w - 1];
    }
    let (chains, loops) = trace_window(&cells, w, h)?;
    let rect = win.rect();
    let len = 2.0 * ((rect[2] - rect[0]) + (rect[3] - rect[1]));
    let tol = 0.75 * win.step;

    // The rings as they were: untouched, dropped (wholly inside), or cut
    // into the pieces outside.
    let depths = current.depths();
    let mut out = Region::new();
    // (points, where it starts on the border, where it ends)
    let mut pieces: Vec<(Vec<Pt>, f64, f64)> = Vec::new();
    for (ring, d) in current.rings.iter().zip(&depths) {
        if ring.is_empty() {
            continue;
        }
        let rb = ring.bbox();
        if rb[2] < rect[0] || rb[0] > rect[2] || rb[3] < rect[1] || rb[1] > rect[3] {
            out.rings.push(ring.clone());
            continue;
        }
        let mut ring = ring.clone();
        ring.orient(d % 2 == 0);
        let xs = rect_crossings(&ring, rect);
        if xs.is_empty() {
            let p = ring.pts[0];
            let inside = p[0] > rect[0] && p[0] < rect[2] && p[1] > rect[1] && p[1] < rect[3];
            if !inside {
                out.rings.push(ring);
            }
            continue;
        }
        // Leaving and entering take turns round the ring.
        if !xs.len().is_multiple_of(2) {
            return None;
        }
        let n = ring.pts.len();
        for (a, x) in xs.iter().enumerate() {
            if x.enter {
                continue;
            }
            let y = xs[(a + 1) % xs.len()];
            if !y.enter {
                return None;
            }
            let mut pts = vec![x.pt];
            let mut steps = (y.edge + n - x.edge) % n;
            // On one edge entering comes before leaving, so an entry on the
            // edge it left by is the whole way round.
            if steps == 0 {
                steps = n;
            }
            for s in 1..=steps {
                pts.push(ring.pts[(x.edge + s) % n]);
            }
            pts.push(y.pt);
            pieces.push((pts, perimeter_at(rect, x.pt), perimeter_at(rect, y.pt)));
        }
    }

    // A piece ends where a chain starts; a chain ends where a piece starts.
    let mut enters: Vec<(f64, usize)> = pieces.iter().enumerate().map(|(i, p)| (p.2, i)).collect();
    let mut leaves: Vec<(f64, usize)> = pieces.iter().enumerate().map(|(i, p)| (p.1, i)).collect();
    let mut starts: Vec<(f64, usize)> = chains
        .iter()
        .enumerate()
        .map(|(i, c)| (perimeter_at(rect, win.at(c[0])), i))
        .collect();
    let mut ends: Vec<(f64, usize)> = chains
        .iter()
        .enumerate()
        .map(|(i, c)| (perimeter_at(rect, win.at(*c.last().unwrap())), i))
        .collect();
    let by_t = |a: &(f64, usize), b: &(f64, usize)| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    };
    for v in [&mut enters, &mut leaves, &mut starts, &mut ends] {
        v.sort_by(by_t);
    }
    let pos = |v: &[(f64, usize)]| v.iter().map(|x| x.0).collect::<Vec<f64>>();
    let near = 2.5 * win.step;
    let m_in = match_round(&pos(&enters), &pos(&starts), len, near)?;
    let m_out = match_round(&pos(&ends), &pos(&leaves), len, near)?;
    let mut chain_after = vec![usize::MAX; pieces.len()];
    for (a, &b) in m_in.iter().enumerate() {
        chain_after[enters[a].1] = starts[b].1;
    }
    let mut piece_after = vec![usize::MAX; chains.len()];
    for (a, &b) in m_out.iter().enumerate() {
        piece_after[ends[a].1] = leaves[b].1;
    }

    // Walk piece, chain, piece, ... back to the first.
    let mut piece_used = vec![false; pieces.len()];
    let mut chain_used = vec![false; chains.len()];
    for first in 0..pieces.len() {
        if piece_used[first] {
            continue;
        }
        let mut pts: Vec<Pt> = Vec::new();
        let mut p = first;
        loop {
            if piece_used[p] {
                if p != first {
                    return None;
                }
                break;
            }
            piece_used[p] = true;
            pts.extend_from_slice(&pieces[p].0);
            let c = *chain_after.get(p)?;
            if c == usize::MAX || chain_used[c] {
                return None;
            }
            chain_used[c] = true;
            let cp: Vec<Pt> = chains[c].iter().map(|k| win.at(*k)).collect();
            pts.extend(dp(&cp, tol));
            p = piece_after[c];
            if p == usize::MAX {
                return None;
            }
        }
        pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12);
        out.rings.push(Poly::new(pts));
    }
    // A chain that met no piece: the outline as it was never reached the
    // border there, so the trace cannot have either.
    if chain_used.iter().any(|u| !u) {
        return None;
    }
    for l in loops {
        let ring = Poly::new(l.into_iter().map(|k| win.at(k)).collect()).simplify(tol);
        if !ring.is_empty() {
            out.rings.push(ring);
        }
    }
    out.normalize();

    // The area the cells say, against the one the outline has.
    let count = |v: &[bool]| v.iter().filter(|&&b| b).count() as f64;
    let cell = win.step * win.step;
    let want = current.area() + (count(&cells) - count(cur)) * cell;
    let slack = 1.0 + 0.02 * (w * h) as f64 * cell;
    ((out.area() - want).abs() <= slack).then_some(out)
}

/// The windowed boolean for an operation that changes nothing outside
/// `b`'s box (union, difference, symmetric difference): `None` when the
/// window would not be much smaller than the whole raster, or no splice
/// was found.
fn window_boolean(a: &Region, b: &Region, op: BoolOp, ss: usize) -> Option<Region> {
    if matches!(op, BoolOp::Intersect) || a.is_empty() || b.is_empty() {
        return None;
    }
    let (ba, bb) = (a.bbox(), b.bbox());
    let whole =
        (ba[2].max(bb[2]) - ba[0].min(bb[0]) + 2.0) * (ba[3].max(bb[3]) - ba[1].min(bb[1]) + 2.0);
    // Two tries, the second with the border elsewhere, before the caller
    // falls back on the whole raster.
    for margin in [1.0, 2.5] {
        let win = Window::around(bb, margin, ss, BOOL_MAX_CELLS)?;
        let wb = win.bounds();
        if (wb[2] - wb[0]) * (wb[3] - wb[1]) > 0.5 * whole {
            return None;
        }
        let cur = win.fill(a);
        let other = win.fill(b);
        let cells: Vec<bool> = cur
            .iter()
            .zip(&other)
            .map(|(&x, &y)| match op {
                BoolOp::Union => x || y,
                BoolOp::Subtract => x && !y,
                _ => x != y,
            })
            .collect();
        if let Some(r) = splice_window(a, &win, &cells, &cur) {
            return Some(r);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Marching-squares plumbing, shared with segmentation.rs
// ---------------------------------------------------------------------------

/// A binary slice padded by one cell on every side, ready to be traced: cell
/// `(i, j)` of the raster is field cell `(i + 1, j + 1)`, so a run touching
/// the border still closes. One buffer serves every slice of a stack.
pub struct PaddedField {
    w: usize,
    h: usize,
    buf: Vec<f32>,
    any: bool,
}

impl PaddedField {
    pub fn new(w: usize, h: usize) -> PaddedField {
        PaddedField {
            w,
            h,
            buf: vec![0.0; (w + 2) * (h + 2)],
            any: false,
        }
    }

    /// Empty the field for the next slice.
    pub fn clear(&mut self) {
        if self.any {
            self.buf.fill(0.0);
            self.any = false;
        }
    }

    /// Mark raster cell `(i, j)` as filled.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize) {
        self.buf[(j + 1) * (self.w + 2) + i + 1] = 1.0;
        self.any = true;
    }

    /// Was anything marked since the last [`Self::clear`]?
    pub fn is_empty(&self) -> bool {
        !self.any
    }

    /// The closed loops around the filled cells, as marching squares
    /// finds them, stitched and stripped of collinear runs, in raster
    /// coordinates (the padding already taken off). Loops of fewer than
    /// three points are dropped.
    pub fn trace(&self) -> Vec<Vec<[f32; 2]>> {
        let (pw, ph) = (self.w + 2, self.h + 2);
        stitch_loops(&render::marching_squares(&self.buf, pw, ph, 0.5))
            .into_iter()
            .map(drop_collinear)
            .filter(|pts| pts.len() >= 3)
            .map(|mut pts| {
                for p in &mut pts {
                    p[0] -= 1.0;
                    p[1] -= 1.0;
                }
                pts
            })
            .collect()
    }

    /// The closed loops along the edges of the filled cells: the exact
    /// outline of the voxels, in raster coordinates like [`Self::trace`].
    ///
    /// Marching squares cuts every corner, which is what makes an outline
    /// look drawn rather than stepped - and costs area: an eighth of a cell
    /// at each outer corner, half a cell for a cell on its own. For an organ
    /// that is nothing; for a target that lands as a cloud of single voxels
    /// it is a tenth of its volume gone between the mask and its contours.
    /// These loops enclose exactly the filled cells, so a contour made from
    /// them has the planimetric area of the voxels it came from and
    /// rasterizes back (centre inside) to the same voxels.
    ///
    /// Cells that touch only at a corner are kept apart - the walk turns
    /// around the cell it is on - so each loop is simple. Holes come out as
    /// loops of their own, wound the other way.
    pub fn trace_voxels(&self) -> Vec<Vec<[f32; 2]>> {
        let (pw, ph) = (self.w + 2, self.h + 2);
        let filled = |i: usize, j: usize| self.buf[j * pw + i] > 0.5;
        // Corner (u, v) sits at the low corner of padded cell (u, v); cell
        // (i, j) spans corners (i, j) to (i + 1, j + 1). Each boundary edge
        // runs with the filled cell on its left (y up): directions as bits,
        // 1 = +x, 2 = +y, 4 = -x, 8 = -y, stored at the corner they leave.
        let cw = pw + 1;
        let mut out = vec![0u8; cw * (ph + 1)];
        for j in 1..=self.h {
            for i in 1..=self.w {
                if !filled(i, j) {
                    continue;
                }
                if !filled(i, j - 1) {
                    out[j * cw + i] |= 1;
                }
                if !filled(i + 1, j) {
                    out[j * cw + i + 1] |= 2;
                }
                if !filled(i, j + 1) {
                    out[(j + 1) * cw + i + 1] |= 4;
                }
                if !filled(i - 1, j) {
                    out[(j + 1) * cw + i] |= 8;
                }
            }
        }
        let step = |d: u8| -> (isize, isize) {
            match d {
                1 => (1, 0),
                2 => (0, 1),
                4 => (-1, 0),
                _ => (0, -1),
            }
        };
        let left = |d: u8| -> u8 {
            if d == 8 {
                1
            } else {
                d << 1
            }
        };
        let right = |d: u8| -> u8 {
            if d == 1 {
                8
            } else {
                d >> 1
            }
        };
        let mut used = vec![0u8; out.len()];
        let mut loops = Vec::new();
        for v0 in 0..out.len() {
            loop {
                let free = out[v0] & !used[v0];
                if free == 0 {
                    break;
                }
                // The lowest free direction (`free` is not zero here).
                let d0 = 1u8 << free.trailing_zeros();
                let (mut v, mut d) = (v0, d0);
                let mut pts = Vec::new();
                loop {
                    used[v] |= d;
                    let (u, w) = (v % cw, v / cw);
                    pts.push([u as f32 - 1.5, w as f32 - 1.5]);
                    let (dx, dy) = step(d);
                    v = ((w as isize + dy) as usize) * cw + (u as isize + dx) as usize;
                    // Where two edges leave a corner (cells touching only
                    // there), the left turn keeps to the cell being walked.
                    let avail = out[v];
                    d = [left(d), d, right(d)]
                        .into_iter()
                        .find(|c| avail & c != 0)
                        .expect("an outline of cells always closes");
                    if v == v0 && d == d0 {
                        break;
                    }
                }
                let pts = drop_collinear(pts);
                if pts.len() >= 3 {
                    loops.push(pts);
                }
            }
        }
        loops
    }

    /// [`Self::trace`] as a normalized [`Region`], shifted by `(x0, y0)`.
    pub fn trace_region(&self, x0: f64, y0: f64) -> Region {
        let mut region = Region::new();
        for pts in self.trace() {
            region.rings.push(Poly::new(
                pts.iter()
                    .map(|p| [x0 + p[0] as f64, y0 + p[1] as f64])
                    .collect(),
            ));
        }
        region.normalize();
        region
    }
}

/// A contour path cut into the simple loops it is made of, wherever it
/// comes back to a vertex it has already passed through.
///
/// Some systems write several touching pieces as one path: two voxels one
/// above the other come out as the first square, then the second, drawn the
/// other way round, through the corner they share - nine points, one
/// contour. Its shoelace area is zero (the two halves cancel), so read as
/// one ring it vanishes from the planimetry, while a scanline fill counts
/// both squares. Cut at the shared vertex it is two squares, as it should
/// be. A keyhole contour (an outline and its hole joined by a slit) comes
/// apart the same way into the outline and the hole, with the slit - two
/// points - dropped. A path that never revisits a vertex is returned whole;
/// a closing point that repeats the first is not a second loop.
pub fn simple_loops(pts: &[[f64; 3]]) -> Vec<Vec<[f64; 3]>> {
    use std::collections::HashMap;
    // Bit-exact: the repeated vertices are the same numbers written twice.
    let key = |p: &[f64; 3]| (p[0].to_bits(), p[1].to_bits(), p[2].to_bits());
    let mut path: Vec<[f64; 3]> = Vec::with_capacity(pts.len());
    let mut seen: HashMap<(u64, u64, u64), usize> = HashMap::with_capacity(pts.len());
    let mut out = Vec::new();
    for p in pts {
        match seen.get(&key(p)) {
            Some(&i) => {
                // A loop closes here: from the first visit to this one. The
                // vertex itself stays on the path, which carries on from it.
                let tail: Vec<[f64; 3]> = path.drain(i + 1..).collect();
                for q in &tail {
                    seen.remove(&key(q));
                }
                if tail.len() >= 2 {
                    let mut ring = Vec::with_capacity(tail.len() + 1);
                    ring.push(path[i]);
                    ring.extend(tail);
                    out.push(ring);
                }
            }
            None => {
                seen.insert(key(p), path.len());
                path.push(*p);
            }
        }
    }
    if path.len() >= 3 {
        out.push(path);
    }
    out
}

/// Endpoint key for loop stitching. On a binary field every marching-squares
/// endpoint lies exactly on a half-integer, so doubling is lossless.
#[inline]
fn ep_key(p: [f32; 2]) -> (i64, i64) {
    ((p[0] * 2.0).round() as i64, (p[1] * 2.0).round() as i64)
}

/// Chain unordered marching-squares segments into closed loops.
pub fn stitch_loops(segs: &[render::Segment]) -> Vec<Vec<[f32; 2]>> {
    use std::collections::HashMap;
    let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (si, s) in segs.iter().enumerate() {
        adj.entry(ep_key(s.0)).or_default().push(si);
        adj.entry(ep_key(s.1)).or_default().push(si);
    }
    let mut used = vec![false; segs.len()];
    let mut out = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let start_key = ep_key(segs[start].0);
        let mut pts = vec![segs[start].0, segs[start].1];
        let mut cur = ep_key(segs[start].1);
        let mut closed = cur == start_key;
        while !closed {
            let next = adj
                .get(&cur)
                .and_then(|c| c.iter().copied().find(|&si| !used[si]));
            let Some(nxt) = next else { break };
            used[nxt] = true;
            let s = &segs[nxt];
            let np = if ep_key(s.0) == cur { s.1 } else { s.0 };
            cur = ep_key(np);
            if cur == start_key {
                closed = true;
            } else {
                pts.push(np);
            }
        }
        if closed && pts.len() >= 3 {
            out.push(pts);
        }
    }
    out
}

/// Remove points that lie on the straight line between their neighbors -
/// marching squares on a binary mask produces long collinear runs.
pub fn drop_collinear(pts: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    let n = pts.len();
    if n < 4 {
        return pts;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = pts[(i + n - 1) % n];
        let c = pts[i];
        let q = pts[(i + 1) % n];
        let cross = (c[0] - p[0]) * (q[1] - c[1]) - (c[1] - p[1]) * (q[0] - c[0]);
        if cross.abs() > 1e-4 {
            out.push(c);
        }
    }
    if out.len() >= 3 {
        out
    } else {
        pts
    }
}

// ---------------------------------------------------------------------------
// Stack - one ROI geometry
// ---------------------------------------------------------------------------

/// The contours of one slice, at a lattice index along the stack axis.
#[derive(Clone, Debug, Default)]
pub struct SlicePolys {
    pub level: usize,
    pub region: Region,
}

/// Every slice of one ROI geometry, on one lattice, stacked along `axis`.
#[derive(Clone, Debug)]
pub struct Stack {
    /// Lattice axis the slices are stacked along: 2 for the axial stack
    /// every planning system expects, 0 sagittal, 1 coronal.
    pub axis: usize,
    /// Sorted by `level`, never two entries for the same level.
    pub slices: Vec<SlicePolys>,
    /// Slices that lie off the lattice along `axis` (a structure moved past
    /// the first or last image slice), by their signed level. They are
    /// carried along by the in-plane operations and the level shifts, come
    /// back onto `slices` when a shift brings them in, and are written out
    /// with everything else - a structure pushed out of the field of view
    /// is not cut. Nothing that goes through a voxel mask sees them.
    pub beyond: Vec<(i64, Region)>,
}

impl Stack {
    pub fn empty(axis: usize) -> Stack {
        Stack {
            axis,
            slices: Vec::new(),
            beyond: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slices.iter().all(|s| s.region.is_empty())
    }

    /// Number of slices carrying at least one contour.
    pub fn occupied(&self) -> usize {
        self.slices.iter().filter(|s| !s.region.is_empty()).count()
    }

    pub fn region_at(&self, level: usize) -> Option<&Region> {
        self.slices
            .iter()
            .find(|s| s.level == level)
            .map(|s| &s.region)
    }

    /// The slice's region, created empty if this is the first contour on it.
    pub fn region_mut(&mut self, level: usize) -> &mut Region {
        let pos = self.slices.binary_search_by_key(&level, |s| s.level);
        let i = match pos {
            Ok(i) => i,
            Err(i) => {
                self.slices.insert(
                    i,
                    SlicePolys {
                        level,
                        region: Region::new(),
                    },
                );
                i
            }
        };
        &mut self.slices[i].region
    }

    pub fn remove_level(&mut self, level: usize) {
        self.slices.retain(|s| s.level != level);
    }

    /// Drop empty slices - run after any edit, so `occupied` and the level
    /// range mean what they say.
    pub fn prune(&mut self) {
        self.slices.retain(|s| !s.region.is_empty());
        self.beyond.retain(|(_, r)| !r.is_empty());
    }

    /// First and last occupied level.
    pub fn level_range(&self) -> Option<(usize, usize)> {
        let mut it = self.slices.iter().filter(|s| !s.region.is_empty());
        let first = it.next()?.level;
        let last = it.next_back().map(|s| s.level).unwrap_or(first);
        Some((first, last))
    }

    /// Enclosed volume in cm³, by planimetry on the contours themselves -
    /// the area of every slice times the slice thickness, which is what a
    /// planning system reports and is finer than counting voxels.
    pub fn volume_cm3(&self, spacing: [f64; 3]) -> f64 {
        let [u, v] = plane_axes(self.axis);
        let cell = spacing[u] * spacing[v] * spacing[self.axis];
        self.slices.iter().map(|s| s.region.area()).sum::<f64>() * cell / 1000.0
    }

    // -- conversions ------------------------------------------------------

    /// Read an ROI's contours onto a lattice.
    ///
    /// The stack axis is decided by the contours themselves: a planar contour
    /// has (almost) no spread along the axis it was drawn on, so the axis
    /// with the smallest total spread wins, and ties go to axial.
    pub fn from_roi(roi: &Roi, grid: &Grid) -> Stack {
        let mut vox: Vec<Vec<[f64; 3]>> = Vec::new();
        for c in &roi.contours {
            if c.geometric_type == "POINT" || c.points.len() < 3 {
                continue;
            }
            let pts: Vec<[f64; 3]> = c.points.iter().map(|p| grid.patient_to_voxel(*p)).collect();
            vox.extend(simple_loops(&pts));
        }
        let mut spread = [0.0f64; 3];
        for pts in &vox {
            for a in 0..3 {
                let (mut lo, mut hi) = (f64::MAX, f64::MIN);
                for p in pts {
                    lo = lo.min(p[a]);
                    hi = hi.max(p[a]);
                }
                spread[a] += hi - lo;
            }
        }
        // Ties (and no contours at all) go to the axial stack.
        let axis = if vox.is_empty() {
            2
        } else {
            let mut best = 2usize;
            for a in [0usize, 1] {
                if spread[a] < spread[best] * 0.999 {
                    best = a;
                }
            }
            best
        };
        let [ua, va] = plane_axes(axis);
        let mut st = Stack::empty(axis);
        for pts in &vox {
            let mean = pts.iter().map(|p| p[axis]).sum::<f64>() / pts.len() as f64;
            let lvl = mean.round();
            let ring = Poly::new(pts.iter().map(|p| [p[ua], p[va]]).collect());
            if ring.is_empty() {
                continue;
            }
            if lvl < 0.0 || lvl >= grid.dims[axis] as f64 {
                st.beyond_mut(lvl as i64).rings.push(ring);
                continue;
            }
            st.region_mut(lvl as usize).rings.push(ring);
        }
        for s in &mut st.slices {
            s.region.normalize();
        }
        st.prune();
        st
    }

    /// Write the stack back out as RTSTRUCT contours in patient coordinates.
    pub fn to_contours(&self, grid: &Grid) -> Vec<Contour> {
        let [ua, va] = plane_axes(self.axis);
        let mut out = Vec::new();
        let on = self.slices.iter().map(|s| (s.level as f64, &s.region));
        let off = self.beyond.iter().map(|(l, r)| (*l as f64, r));
        for (level, region) in on.chain(off) {
            for ring in &region.rings {
                if ring.is_empty() {
                    continue;
                }
                let points: Vec<Vec3> = ring
                    .pts
                    .iter()
                    .map(|p| {
                        let mut v = [0.0f64; 3];
                        v[ua] = p[0];
                        v[va] = p[1];
                        v[self.axis] = level;
                        grid.voxel_to_patient(v[0], v[1], v[2])
                    })
                    .collect();
                out.push(Contour {
                    points,
                    geometric_type: "CLOSED_PLANAR".into(),
                });
            }
        }
        out
    }

    /// Replace an ROI's geometry with this stack, converting to the axial
    /// stack first when it was drawn on another plane - RTSTRUCT readers
    /// expect axial contours, and a mixed set is nobody's friend.
    pub fn apply_to_roi(&self, roi: &mut Roi, grid: &Grid) {
        let axial;
        let st = if self.axis == 2 {
            self
        } else {
            axial = self.to_axis(grid.dims, 2);
            &axial
        };
        // Points (reference marks) are geometry too, and no business of ours.
        let points: Vec<Contour> = roi
            .contours
            .iter()
            .filter(|c| c.geometric_type == "POINT")
            .cloned()
            .collect();
        roi.contours = st.to_contours(grid);
        roi.contours.extend(points);
    }

    /// [`Self::apply_to_roi`] for a sagittal or coronal stack of which only
    /// the axial slices `levels[0]..=levels[1]` can have changed since the
    /// ROI was last written: those are re-cut ([`Self::to_axis_levels`]),
    /// every other axial contour of the ROI stays as it is. Re-cutting the
    /// whole structure was a rasterization and trace of all of it on every
    /// write of a stroke (130 ms on a body outline); this costs the slab. An
    /// ROI that is not stored as axial contours is written whole.
    pub fn apply_levels_to_roi(&self, roi: &mut Roi, grid: &Grid, levels: [usize; 2]) {
        if self.axis == 2 {
            self.apply_to_roi(roi, grid);
            return;
        }
        let mut axial = Stack::from_roi(roi, grid);
        if axial.axis != 2 {
            if !axial.is_empty() {
                self.apply_to_roi(roi, grid);
                return;
            }
            axial = Stack::empty(2);
        }
        let recut = self.to_axis_levels(grid.dims, 2, levels);
        axial
            .slices
            .retain(|s| s.level < levels[0] || s.level > levels[1]);
        axial.slices.extend(recut.slices);
        axial.slices.sort_by_key(|s| s.level);
        axial.prune();
        axial.apply_to_roi(roi, grid);
    }

    /// Fill the stack into a voxel mask in the volume's index order.
    /// The lattice box the geometry lives in, clamped to `dims`; `None`
    /// when the stack is empty. Any per-voxel pass over one structure wants
    /// this instead of the whole volume.
    pub fn bbox(&self, dims: [usize; 3]) -> Option<([usize; 3], [usize; 3])> {
        let [ua, va] = plane_axes(self.axis);
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        let mut any = false;
        for s in &self.slices {
            if s.level >= dims[self.axis] || s.region.is_empty() {
                continue;
            }
            for ring in &s.region.rings {
                for p in &ring.pts {
                    let u = p[0].floor().max(0.0) as usize;
                    let v = p[1].floor().max(0.0) as usize;
                    let uh = (p[0].ceil().max(0.0) as usize).min(dims[ua].saturating_sub(1));
                    let vh = (p[1].ceil().max(0.0) as usize).min(dims[va].saturating_sub(1));
                    lo[ua] = lo[ua].min(u.min(dims[ua].saturating_sub(1)));
                    lo[va] = lo[va].min(v.min(dims[va].saturating_sub(1)));
                    hi[ua] = hi[ua].max(uh);
                    hi[va] = hi[va].max(vh);
                    any = true;
                }
            }
            lo[self.axis] = lo[self.axis].min(s.level);
            hi[self.axis] = hi[self.axis].max(s.level);
        }
        if !any {
            return None;
        }
        Some((lo, hi))
    }

    pub fn rasterize(&self, dims: [usize; 3]) -> Vec<u8> {
        let [nx, ny, nz] = dims;
        let [ua, va] = plane_axes(self.axis);
        let (nu, nv) = (dims[ua], dims[va]);
        let mut mask = vec![0u8; nx * ny * nz];
        let mut plane = vec![false; nu * nv];
        for s in &self.slices {
            if s.level >= dims[self.axis] || s.region.is_empty() {
                continue;
            }
            plane.iter_mut().for_each(|v| *v = false);
            fill_region(&s.region, -0.5, -0.5, nu, nv, 1.0, &mut plane);
            for v in 0..nv {
                for u in 0..nu {
                    if !plane[v * nu + u] {
                        continue;
                    }
                    let mut idx = [0usize; 3];
                    idx[ua] = u;
                    idx[va] = v;
                    idx[self.axis] = s.level;
                    mask[idx[2] * nx * ny + idx[1] * nx + idx[0]] = 1;
                }
            }
        }
        mask
    }

    /// Trace a voxel mask into contours, slice by slice along `axis`.
    pub fn from_mask(mask: &[u8], dims: [usize; 3], axis: usize) -> Stack {
        Stack::from_mask_at(mask, dims, [0; 3], axis)
    }

    /// [`Self::from_mask`] of the mask that holds exactly `voxels` (linear
    /// indices on a lattice `dims`), tracing only the box they occupy: what a
    /// region grown into an RT structure is, without a pass over the whole
    /// volume for a few thousand voxels.
    pub fn from_voxels(voxels: &[u32], dims: [usize; 3], axis: usize) -> Stack {
        let [nx, ny, _] = dims;
        let sl = nx * ny;
        let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
        let at = |v: u32| {
            let v = v as usize;
            [v % nx, (v % sl) / nx, v / sl]
        };
        for &v in voxels {
            let c = at(v);
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a]);
            }
        }
        if lo[0] == usize::MAX || sl == 0 {
            return Stack::empty(axis);
        }
        let sub = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
        let mut mask = vec![0u8; sub[0] * sub[1] * sub[2]];
        for &v in voxels {
            let c = at(v);
            let (i, j, k) = (c[0] - lo[0], c[1] - lo[1], c[2] - lo[2]);
            mask[k * sub[0] * sub[1] + j * sub[0] + i] = 1;
        }
        Stack::from_mask_at(&mask, sub, lo, axis)
    }

    /// [`Self::from_mask`] of a mask that is the box at `lo` of a larger
    /// lattice, the contours put in that lattice's coordinates.
    fn from_mask_at(mask: &[u8], dims: [usize; 3], lo: [usize; 3], axis: usize) -> Stack {
        let [nx, ny, _nz] = dims;
        let [ua, va] = plane_axes(axis);
        let (nu, nv) = (dims[ua], dims[va]);
        let mut st = Stack::empty(axis);
        let mut field = PaddedField::new(nu, nv);
        for level in 0..dims[axis] {
            field.clear();
            for v in 0..nv {
                for u in 0..nu {
                    let mut idx = [0usize; 3];
                    idx[ua] = u;
                    idx[va] = v;
                    idx[axis] = level;
                    if mask[idx[2] * nx * ny + idx[1] * nx + idx[0]] != 0 {
                        field.set(u, v);
                    }
                }
            }
            if field.is_empty() {
                continue;
            }
            let region = field.trace_region(lo[ua] as f64, lo[va] as f64);
            if !region.is_empty() {
                st.slices.push(SlicePolys {
                    level: level + lo[axis],
                    region,
                });
            }
        }
        st
    }

    /// Re-slice along another lattice axis. This is the one conversion that
    /// necessarily goes through voxels - the contours of a sagittal stack say
    /// nothing about where the axial ones run - so it is explicit, and the
    /// interface says so when it happens.
    ///
    /// Only the lattice box the geometry occupies ([`Self::bbox`]) is
    /// rasterised and traced, which is what [`Self::rasterize`] and
    /// [`Self::from_mask`] give over the whole lattice, to the bit: every
    /// cell they fill lies in that box, the fill is computed with the whole
    /// lattice's coordinates, and the trace of a box is the trace of the
    /// lattice shifted by a whole number of cells, which half-integer
    /// coordinates carry exactly. A contour edit on a sagittal stack writes
    /// back through here on every commit, and the whole lattice was a
    /// full-volume pass each time.
    pub fn to_axis(&self, dims: [usize; 3], axis: usize) -> Stack {
        if axis == self.axis {
            return self.clone();
        }
        let Some((lo, hi)) = self.bbox(dims) else {
            return Stack::empty(axis);
        };
        let sub = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
        let mask = self.rasterize_box(dims, lo, sub);
        Stack::from_mask_at(&mask, sub, lo, axis)
    }

    /// [`Self::to_axis`] for the new stack's slices `levels[0]..=levels[1]`
    /// only: the same slices, to the bit (each is the trace of its own
    /// slice of the same mask), at the cost of the slab rather than of the
    /// whole box. What a stroke on a sagittal or coronal stack, which moved
    /// a few axial slices, writes back.
    pub fn to_axis_levels(&self, dims: [usize; 3], axis: usize, levels: [usize; 2]) -> Stack {
        if axis == self.axis {
            let mut st = self.clone();
            st.slices
                .retain(|s| s.level >= levels[0] && s.level <= levels[1]);
            st.beyond.clear();
            return st;
        }
        let Some((mut lo, mut hi)) = self.bbox(dims) else {
            return Stack::empty(axis);
        };
        lo[axis] = lo[axis].max(levels[0]);
        hi[axis] = hi[axis].min(levels[1]);
        if lo[axis] > hi[axis] {
            return Stack::empty(axis);
        }
        let sub = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
        let mask = self.rasterize_box(dims, lo, sub);
        Stack::from_mask_at(&mask, sub, lo, axis)
    }

    /// [`Self::rasterize`] of the box at `lo`, `sub` cells large, of a
    /// lattice `dims`: the box's own mask, in its own index order.
    fn rasterize_box(&self, dims: [usize; 3], lo: [usize; 3], sub: [usize; 3]) -> Vec<u8> {
        let [ua, va] = plane_axes(self.axis);
        let nu = dims[ua];
        let mut mask = vec![0u8; sub[0] * sub[1] * sub[2]];
        let (u0, v0) = (lo[ua], lo[va]);
        let (u1, v1) = (lo[ua] + sub[ua], lo[va] + sub[va]);
        for s in &self.slices {
            if s.level >= dims[self.axis] || s.region.is_empty() {
                continue;
            }
            fill_region_rows(&s.region, -0.5, -0.5, nu, v0..v1, 1.0, |v, u| {
                if u < u0 || u >= u1 {
                    return;
                }
                let mut idx = [0usize; 3];
                idx[ua] = u - u0;
                idx[va] = v - v0;
                idx[self.axis] = s.level - lo[self.axis];
                mask[idx[2] * sub[0] * sub[1] + idx[1] * sub[0] + idx[0]] = 1;
            });
        }
        mask
    }

    // -- editing ----------------------------------------------------------

    pub fn simplify(&mut self, tol: f64) {
        for s in &mut self.slices {
            s.region.simplify(tol);
        }
        self.prune();
    }

    pub fn smooth(&mut self, passes: usize) {
        for s in &mut self.slices {
            s.region.smooth(passes);
        }
    }

    pub fn remove_holes(&mut self) {
        for s in &mut self.slices {
            s.region.remove_holes();
        }
    }

    pub fn drop_smaller_than(&mut self, min_area: f64) {
        for s in &mut self.slices {
            s.region.drop_smaller_than(min_area);
        }
        self.prune();
    }

    /// Resolve rings of the same slice that overlap - the "simplify contours"
    /// conflict resolution: the filled area is unchanged, the outline is one
    /// clean set of nested rings.
    pub fn resolve_overlaps(&mut self) {
        for s in &mut self.slices {
            if s.region.rings.len() < 2 {
                continue;
            }
            let mut crossing = false;
            'outer: for i in 0..s.region.rings.len() {
                for j in (i + 1)..s.region.rings.len() {
                    let one = Region::from_ring(s.region.rings[j].clone());
                    if one.touches(&s.region.rings[i]) {
                        crossing = true;
                        break 'outer;
                    }
                }
            }
            if crossing {
                s.region = s.region.boolean(&Region::new(), BoolOp::Union);
            }
        }
        self.prune();
    }

    /// Cap the number of points of every ring, by simplifying with a
    /// tolerance found by bisection.
    pub fn limit_points(&mut self, max_points: usize) {
        if max_points < 3 {
            return;
        }
        for s in &mut self.slices {
            for r in &mut s.region.rings {
                if r.len() <= max_points {
                    continue;
                }
                let (mut lo, mut hi) = (0.0f64, 8.0f64);
                let mut best = r.simplify(hi);
                for _ in 0..24 {
                    let mid = 0.5 * (lo + hi);
                    let cand = r.simplify(mid);
                    if cand.len() > max_points {
                        lo = mid;
                    } else {
                        hi = mid;
                        best = cand;
                    }
                }
                // A ring too small to survive the tolerance at all (every
                // vertex within it of the closing chord) keeps its points
                // under simplification: even spacing is what caps it.
                if best.len() > max_points {
                    best = r.resample(r.perimeter() / max_points as f64);
                }
                *r = best;
            }
        }
    }

    pub fn translate(&mut self, d: Pt) {
        for r in self.rings_mut() {
            r.translate(d);
        }
    }

    /// Scale every slice about the stack's own in-plane centroid.
    pub fn scale(&mut self, factor: f64) {
        let c = self.centroid();
        for r in self.rings_mut() {
            r.scale_about(c, factor);
        }
    }

    pub fn rotate(&mut self, angle: f64) {
        let c = self.centroid();
        for r in self.rings_mut() {
            r.rotate_about(c, angle);
        }
    }

    /// Every ring, on and off the lattice.
    fn rings_mut(&mut self) -> impl Iterator<Item = &mut Poly> {
        self.slices
            .iter_mut()
            .flat_map(|s| s.region.rings.iter_mut())
            .chain(self.beyond.iter_mut().flat_map(|(_, r)| r.rings.iter_mut()))
    }

    /// Area-weighted centroid in lattice indices, all three axes.
    pub fn centroid3(&self) -> [f64; 3] {
        let [ua, va] = plane_axes(self.axis);
        let mut acc = [0.0f64; 3];
        let mut w = 0.0;
        for s in &self.slices {
            for r in &s.region.rings {
                let a = r.area();
                let c = r.centroid();
                acc[ua] += c[0] * a;
                acc[va] += c[1] * a;
                acc[self.axis] += s.level as f64 * a;
                w += a;
            }
        }
        if w > 0.0 {
            acc.map(|v| v / w)
        } else {
            [0.0; 3]
        }
    }

    /// The stack shifted by a whole number of levels along its axis;
    /// slices pushed off the lattice are dropped.
    pub fn shift_levels(&mut self, d: i64, dims: [usize; 3]) {
        if d == 0 {
            return;
        }
        let n = dims[self.axis] as i64;
        // Every slice by `d`, on or off the lattice, then sort them back
        // into the two lists by where they landed.
        let mut all: Vec<(i64, Region)> = self
            .slices
            .drain(..)
            .map(|s| (s.level as i64 + d, s.region))
            .chain(self.beyond.drain(..).map(|(l, r)| (l + d, r)))
            .collect();
        all.sort_by_key(|(l, _)| *l);
        for (l, region) in all {
            if (0..n).contains(&l) {
                self.slices.push(SlicePolys {
                    level: l as usize,
                    region,
                });
            } else {
                self.beyond.push((l, region));
            }
        }
    }

    /// The off-lattice region at a signed level, created empty if new.
    fn beyond_mut(&mut self, level: i64) -> &mut Region {
        let i = match self.beyond.iter().position(|(l, _)| *l >= level) {
            Some(i) if self.beyond[i].0 == level => i,
            Some(i) => {
                self.beyond.insert(i, (level, Region::new()));
                i
            }
            None => {
                self.beyond.push((level, Region::new()));
                self.beyond.len() - 1
            }
        };
        &mut self.beyond[i].1
    }

    /// Translate by millimetres along the lattice axes: exactly in the
    /// plane, and by the nearest whole number of slices along the stacking
    /// axis - the contours keep their shape, the way a planning system
    /// moves a structure.
    pub fn translate_mm(&mut self, shift: [f64; 3], spacing: [f64; 3], dims: [usize; 3]) {
        let [ua, va] = plane_axes(self.axis);
        let sp = spacing.map(|v| v.max(1e-9));
        self.translate([shift[ua] / sp[ua], shift[va] / sp[va]]);
        let d = (shift[self.axis] / sp[self.axis]).round();
        if d.is_finite() {
            self.shift_levels(d as i64, dims);
        }
    }

    /// The stack moved by a rigid motion in the lattice's millimetre frame.
    ///
    /// Goes through a voxel mask: the geometry is rasterized, resampled
    /// (trilinear, cut at one half) at the moved position and traced
    /// again, so a tilt lands on the slices it now crosses. The mask is
    /// finer than the lattice in the plane (up to four times, less for a
    /// structure too large for that) and only as large as the structure's
    /// box, so a small volume keeps its shape to a fraction of a pixel.
    /// Anything moved off the lattice is lost, and the result is empty
    /// when nothing is left.
    pub fn rigid_moved(&self, dims: [usize; 3], spacing: [f64; 3], m: &Rigid) -> Stack {
        use rayon::prelude::*;
        let axis = self.axis;
        let [ua, va] = plane_axes(axis);
        let Some((lo, hi)) = self.bbox(dims) else {
            return Stack::empty(axis);
        };
        let sp = spacing.map(|v| v.max(1e-9));
        // Output box in lattice indices: the moved corners of the input box.
        let mut olo = [f64::INFINITY; 3];
        let mut ohi = [f64::NEG_INFINITY; 3];
        for corner in 0..8 {
            let p = [
                if corner & 1 == 0 { lo[0] } else { hi[0] } as f64 * sp[0],
                if corner & 2 == 0 { lo[1] } else { hi[1] } as f64 * sp[1],
                if corner & 4 == 0 { lo[2] } else { hi[2] } as f64 * sp[2],
            ];
            let q = m.apply(p);
            for a in 0..3 {
                olo[a] = olo[a].min(q[a] / sp[a]);
                ohi[a] = ohi[a].max(q[a] / sp[a]);
            }
        }
        let mut b0 = [0usize; 3];
        let mut b1 = [0usize; 3];
        for a in 0..3 {
            if !olo[a].is_finite() || !ohi[a].is_finite() || dims[a] == 0 {
                return Stack::empty(axis);
            }
            let lo_i = (olo[a].floor() - 1.0).max(0.0);
            let hi_i = (ohi[a].ceil() + 1.0).min(dims[a] as f64 - 1.0);
            if lo_i > hi_i {
                return Stack::empty(axis);
            }
            b0[a] = lo_i as usize;
            b1[a] = hi_i as usize;
        }
        // The in-plane refinement, capped so the two work masks stay small.
        let voxels = |a: [usize; 3], b: [usize; 3]| -> f64 {
            (0..3).map(|k| (b[k] - a[k] + 3) as f64).product()
        };
        let biggest = voxels(lo, hi).max(voxels(b0, b1));
        let f = ((6.0e7 / biggest).sqrt().floor() as usize).clamp(1, 4);
        let ff = f as f64;
        // Input work lattice: the structure's box, a voxel of margin, the
        // plane refined `f` times. Local fine index u' = (u - lo + 1 + 0.5) f - 0.5.
        let mut in_dims = [0usize; 3];
        for a in 0..3 {
            in_dims[a] = hi[a] - lo[a] + 3;
        }
        in_dims[ua] *= f;
        in_dims[va] *= f;
        let mut local = self.clone();
        for s in &mut local.slices {
            s.level = s.level + 1 - lo[axis];
            for r in &mut s.region.rings {
                r.translate([-(lo[ua] as f64) + 1.0, -(lo[va] as f64) + 1.0]);
                r.scale_about([-0.5, -0.5], ff);
            }
        }
        let mask = local.rasterize(in_dims);
        let [nx, ny, nz] = in_dims;
        let at = |i: isize, j: isize, k: isize| -> f64 {
            if i < 0 || j < 0 || k < 0 || i >= nx as isize || j >= ny as isize || k >= nz as isize {
                0.0
            } else {
                mask[k as usize * nx * ny + j as usize * nx + i as usize] as f64
            }
        };
        // Global millimetres → local fine input index.
        let to_local = |src: [f64; 3]| -> [f64; 3] {
            let mut r = [0.0; 3];
            for a in 0..3 {
                let idx = src[a] / sp[a] - lo[a] as f64 + 1.0;
                r[a] = if a == axis {
                    idx
                } else {
                    (idx + 0.5) * ff - 0.5
                };
            }
            r
        };
        let inside = |p_mm: [f64; 3]| -> bool {
            let f3 = to_local(m.inverse_apply(p_mm));
            let base = f3.map(|v| v.floor());
            let t = [f3[0] - base[0], f3[1] - base[1], f3[2] - base[2]];
            let (i, j, k) = (base[0] as isize, base[1] as isize, base[2] as isize);
            let mut v = 0.0;
            for c in 0..8 {
                let (di, dj, dk) = (
                    (c & 1) as isize,
                    ((c >> 1) & 1) as isize,
                    ((c >> 2) & 1) as isize,
                );
                let w = (if di == 1 { t[0] } else { 1.0 - t[0] })
                    * (if dj == 1 { t[1] } else { 1.0 - t[1] })
                    * (if dk == 1 { t[2] } else { 1.0 - t[2] });
                if w > 0.0 {
                    v += w * at(i + di, j + dj, k + dk);
                }
            }
            v >= 0.5
        };
        // Output work lattice: the output box, refined the same way.
        let mut out_dims = [0usize; 3];
        for a in 0..3 {
            out_dims[a] = b1[a] - b0[a] + 1;
        }
        out_dims[ua] *= f;
        out_dims[va] *= f;
        let [ox, oy, _] = out_dims;
        let mut out = vec![0u8; out_dims[0] * out_dims[1] * out_dims[2]];
        // Local fine output index → global millimetres.
        let to_mm = |idx: [usize; 3]| -> [f64; 3] {
            let mut r = [0.0; 3];
            for a in 0..3 {
                let g = if a == axis {
                    idx[a] as f64
                } else {
                    (idx[a] as f64 + 0.5) / ff - 0.5
                };
                r[a] = (g + b0[a] as f64) * sp[a];
            }
            r
        };
        out.par_chunks_mut(ox).enumerate().for_each(|(r, line)| {
            let (k, j) = (r / oy, r % oy);
            for (i, cell) in line.iter_mut().enumerate() {
                if inside(to_mm([i, j, k])) {
                    *cell = 1;
                }
            }
        });
        let mut st = Stack::from_mask(&out, out_dims, axis);
        for s in &mut st.slices {
            s.level += b0[axis];
            for r in &mut s.region.rings {
                r.scale_about([-0.5, -0.5], 1.0 / ff);
                r.translate([b0[ua] as f64, b0[va] as f64]);
            }
        }
        st
    }

    /// Area-weighted centroid over every slice, in in-plane coordinates.
    pub fn centroid(&self) -> Pt {
        let mut acc = [0.0f64, 0.0];
        let mut w = 0.0;
        for s in &self.slices {
            for r in &s.region.rings {
                let a = r.area();
                let c = r.centroid();
                acc[0] += c[0] * a;
                acc[1] += c[1] * a;
                w += a;
            }
        }
        if w > 0.0 {
            [acc[0] / w, acc[1] / w]
        } else {
            [0.0, 0.0]
        }
    }

    /// Keep (or delete) the connected piece the given voxel falls in.
    ///
    /// Connectivity is a three-dimensional question, so this one goes through
    /// the mask: connected components of the voxelised stack, the chosen one
    /// kept or dropped, and the
    /// contours re-traced. Everything else in this module leaves untouched
    /// slices untouched; this does not, and says so.
    pub fn component_at(&self, dims: [usize; 3], voxel: [usize; 3], keep: bool) -> Stack {
        let mask = self.rasterize(dims);
        let [nx, ny, _] = dims;
        let hit = voxel[2] * nx * ny + voxel[1] * nx + voxel[0];
        if hit >= mask.len() || mask[hit] == 0 {
            return self.clone();
        }
        let comps = crate::morphology::components(&mask, dims);
        let Some(chosen) = comps.iter().find(|c| c.voxels.contains(&(hit as u32))) else {
            return self.clone();
        };
        let mut out = vec![0u8; mask.len()];
        if keep {
            for &v in &chosen.voxels {
                out[v as usize] = 1;
            }
        } else {
            out.copy_from_slice(&mask);
            for &v in &chosen.voxels {
                out[v as usize] = 0;
            }
        }
        Stack::from_mask(&out, dims, self.axis)
    }

    /// Linear interpolation onto the empty slices between drawn ones.
    ///
    /// Between two occupied levels the two regions are turned into signed
    /// distance fields over their common window, blended, and the zero level
    /// re-extracted - which handles the cases point matching gets wrong: a
    /// structure that splits in two, or one whose contours barely overlap.
    /// Returns only the *new* slices, so the caller can show them as a
    /// preview and accept them one at a time.
    pub fn interpolated(&self, dims: [usize; 3]) -> Stack {
        let mut out = Stack::empty(self.axis);
        let occ: Vec<usize> = self
            .slices
            .iter()
            .filter(|s| !s.region.is_empty())
            .map(|s| s.level)
            .collect();
        if occ.len() < 2 {
            return out;
        }
        let [ua, va] = plane_axes(self.axis);
        let (nu, nv) = (dims[ua], dims[va]);
        for pair in occ.windows(2) {
            let (k0, k1) = (pair[0], pair[1]);
            if k1 <= k0 + 1 {
                continue;
            }
            let (Some(r0), Some(r1)) = (self.region_at(k0), self.region_at(k1)) else {
                continue;
            };
            // A window around both, one voxel of margin, clipped to the slice.
            let (b0, b1) = (r0.bbox(), r1.bbox());
            let x0 = (b0[0].min(b1[0]).floor() - 2.0).max(0.0);
            let y0 = (b0[1].min(b1[1]).floor() - 2.0).max(0.0);
            let x1 = (b0[2].max(b1[2]).ceil() + 2.0).min(nu as f64 - 1.0);
            let y1 = (b0[3].max(b1[3]).ceil() + 2.0).min(nv as f64 - 1.0);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let (w, h) = ((x1 - x0) as usize + 1, (y1 - y0) as usize + 1);
            let d0 = sdf_of(r0, x0, y0, w, h);
            let d1 = sdf_of(r1, x0, y0, w, h);
            let mut field = PaddedField::new(w, h);
            for k in (k0 + 1)..k1 {
                let t = (k - k0) as f32 / (k1 - k0) as f32;
                field.clear();
                for j in 0..h {
                    for i in 0..w {
                        if (1.0 - t) * d0[j * w + i] + t * d1[j * w + i] <= 0.0 {
                            field.set(i, j);
                        }
                    }
                }
                if field.is_empty() {
                    continue;
                }
                let region = field.trace_region(x0, y0);
                if !region.is_empty() {
                    out.slices.push(SlicePolys { level: k, region });
                }
            }
        }
        out.slices.sort_by_key(|s| s.level);
        out
    }
}

/// Signed distance to a region's boundary over a window, negative inside, in
/// voxel units. Built from the exact Euclidean transform in `morphology`.
fn sdf_of(region: &Region, x0: f64, y0: f64, w: usize, h: usize) -> Vec<f32> {
    let mut inside = vec![false; w * h];
    fill_region(region, x0 - 0.5, y0 - 0.5, w, h, 1.0, &mut inside);
    let mask: Vec<u8> = inside.iter().map(|&b| b as u8).collect();
    let dims = [w, h, 1];
    let sp = [1.0, 1.0, 1.0];
    let d_out = crate::morphology::dist2_to_foreground(&mask, dims, sp);
    let d_in = crate::morphology::dist2_to_background(&mask, dims, sp);
    (0..w * h)
        .map(|i| {
            if mask[i] != 0 {
                -d_in[i].sqrt()
            } else {
                d_out[i].sqrt()
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The 3-D contour brush
// ---------------------------------------------------------------------------

/// The finest footprint cell, in millimetres: below this a finer cell
/// changes nothing a hand can aim at, and only costs cells.
const FOOT_MIN_CELL_MM: f64 = 0.2;
/// A brush radius is cut into at least this many cells, so a small brush
/// keeps its round outline while a large one does not paint millions of
/// cells per mouse sample.
const FOOT_CELLS_PER_RADIUS: f64 = 24.0;
/// The most cells per voxel along an in-plane axis.
const FOOT_MAX_SUB: usize = 8;

/// What one stroke of the 3-D contour brush has covered so far, slice by
/// slice of a [`Stack`] - a ball in patient millimetres swept along the
/// pointer, cut by every slice it reaches.
///
/// A ball through a structure's contours is, on each slice, a disc (a
/// capsule, swept) unioned with or cut out of that slice's region. Done as
/// polygon booleans on every mouse sample, that is a boolean per reached
/// slice per sample, which is what kept the contour brush on one slice.
/// This keeps the stroke as a raster instead: each sample marks the cells
/// the ball covers, which is a few thousand distance tests and nothing
/// else, and the owner turns a slice's cells into an outline
/// ([`Self::region`]) and merges it into the contours ([`Self::merged`])
/// only when it wants to show or keep the result. The merge always starts
/// from the slice as it was before the stroke, so however often it runs,
/// nothing compounds: the rings the stroke did not reach keep their
/// vertices, the boolean being exact on them ([`Region::boolean`]).
///
/// The geometry is the lattice's own: the distance from a cell centre to
/// the swept segment is measured in patient millimetres through the grid's
/// direction vectors and spacings, so a slice of a tilted acquisition, whose
/// normal is not its stacking direction, gets the disc where the sphere
/// really crosses it - the circle's radius from the distance along the
/// slice normal, its centre shifted by the tilt. Cells are finer than the
/// voxels (`sub` per voxel along each in-plane axis, from the radius), so a
/// disc stays round on a 3 mm sagittal lattice; the edge band, when the
/// owner gives one, is asked of the voxel each cell lies in, voxel by voxel
/// as the segment brush has it.
pub struct BallFootprint {
    axis: usize,
    dims: [usize; 3],
    /// Cells per voxel along the two in-plane axes, [`plane_axes`] order.
    sub: [usize; 2],
    /// `m[a]` is the patient vector (mm) of one step along lattice axis `a`.
    m: [[f64; 3]; 3],
    /// How many voxels along each lattice axis one millimetre of ball
    /// radius can reach: the norms of the rows of the inverse of `m`, which
    /// bound an ellipsoid's extent exactly. Along the stacking axis it is
    /// one over the spacing along the slice normal.
    reach: [f64; 3],
    slices: std::collections::BTreeMap<usize, FootSlice>,
}

/// The cells one slice of a footprint holds.
struct FootSlice {
    /// One row of cells per entry; an empty row was never touched.
    rows: Vec<Vec<u8>>,
    /// The box of the marked cells, inclusive.
    lo: [usize; 2],
    hi: [usize; 2],
    /// Marked since it was last taken (see [`BallFootprint::take_dirty`]),
    /// and the box of the cells marked since, inclusive.
    dirty: bool,
    dlo: [usize; 2],
    dhi: [usize; 2],
}

impl FootSlice {
    fn cell(&self, i: i64, j: i64) -> bool {
        if i < 0 || j < 0 {
            return false;
        }
        self.rows
            .get(j as usize)
            .and_then(|r| r.get(i as usize))
            .is_some_and(|&c| c != 0)
    }

    /// The box marked since it was last taken, `[u0, v0, u1, v1]` in cells,
    /// and the slice counted as taken.
    fn take(&mut self) -> Option<[usize; 4]> {
        if !std::mem::take(&mut self.dirty) {
            return None;
        }
        let b = [self.dlo[0], self.dlo[1], self.dhi[0], self.dhi[1]];
        self.dlo = [usize::MAX; 2];
        self.dhi = [0; 2];
        Some(b)
    }
}

impl BallFootprint {
    /// An empty footprint on `grid`'s lattice, for a stack along `axis`,
    /// with cells fine enough for a ball of `radius_mm`.
    pub fn new(grid: &Grid, axis: usize, radius_mm: f64) -> BallFootprint {
        let sp = grid.spacing;
        let dirs = [grid.row_dir, grid.col_dir, grid.normal];
        let m: [[f64; 3]; 3] =
            std::array::from_fn(|a| [dirs[a].x * sp[a], dirs[a].y * sp[a], dirs[a].z * sp[a]]);
        let v = |a: usize| Vec3::new(m[a][0], m[a][1], m[a][2]);
        let det = v(0).dot(v(1).cross(v(2)));
        let reach = if det.abs() > 1e-12 {
            [
                v(1).cross(v(2)).length() / det.abs(),
                v(2).cross(v(0)).length() / det.abs(),
                v(0).cross(v(1)).length() / det.abs(),
            ]
        } else {
            // A degenerate lattice: the axis-aligned answer, which at
            // least paints something.
            std::array::from_fn(|a| 1.0 / sp[a].max(1e-6))
        };
        let cell = (radius_mm.max(0.01) / FOOT_CELLS_PER_RADIUS).max(FOOT_MIN_CELL_MM);
        let [ua, va] = plane_axes(axis);
        let sub = [ua, va].map(|a| ((sp[a] / cell).ceil() as usize).clamp(1, FOOT_MAX_SUB));
        BallFootprint {
            axis,
            dims: grid.dims,
            sub,
            m,
            reach,
            slices: Default::default(),
        }
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Nothing marked yet.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// The levels the stroke has reached, in order.
    pub fn levels(&self) -> Vec<usize> {
        self.slices.keys().copied().collect()
    }

    /// Mark the ball of `radius_mm` swept from `a` to `b` (fractional voxel
    /// indices, `[i, j, k]`), on every slice it reaches. `keep` is asked of
    /// the voxel (`[i, j, k]`) under each cell the ball covers; a cell whose
    /// voxel it refuses stays unmarked. Returns whether any cell was newly
    /// marked.
    pub fn sweep(
        &mut self,
        a: [f64; 3],
        b: [f64; 3],
        radius_mm: f64,
        keep: impl Fn([usize; 3]) -> bool,
    ) -> bool {
        let r = radius_mm.max(0.01);
        let axis = self.axis;
        let [ua, va] = plane_axes(axis);
        let n = self.dims;
        if n.contains(&0) {
            return false;
        }
        // The box of the swept ball along each lattice axis, in voxels.
        let lo: [f64; 3] = std::array::from_fn(|x| a[x].min(b[x]) - r * self.reach[x]);
        let hi: [f64; 3] = std::array::from_fn(|x| a[x].max(b[x]) + r * self.reach[x]);
        let l0 = lo[axis].ceil().max(0.0);
        let l1 = hi[axis].floor().min(n[axis] as f64 - 1.0);
        if l1 < l0 {
            return false;
        }
        // Voxel coordinate x of cell c, with s cells per voxel: the cells
        // tile the voxels exactly, from -0.5 to n - 0.5.
        let cells = |x: usize| n[[ua, va][x]] * self.sub[x];
        let range = |x: usize| -> Option<(usize, usize)> {
            let s = self.sub[x] as f64;
            let ax = [ua, va][x];
            let c0 = ((lo[ax] + 0.5) * s - 0.5).ceil().max(0.0);
            let c1 = ((hi[ax] + 0.5) * s - 0.5)
                .floor()
                .min(cells(x) as f64 - 1.0);
            (c1 >= c0).then_some((c0 as usize, c1 as usize))
        };
        let (Some((cu0, cu1)), Some((cv0, cv1))) = (range(0), range(1)) else {
            return false;
        };
        let m = self.m;
        let to_mm = |d: [f64; 3]| -> [f64; 3] {
            std::array::from_fn(|c| m[0][c] * d[0] + m[1][c] * d[1] + m[2][c] * d[2])
        };
        let dot = |p: [f64; 3], q: [f64; 3]| p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
        let ab = to_mm([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
        let ab2 = dot(ab, ab);
        let r2 = r * r;
        let (su, sv) = (self.sub[0] as f64, self.sub[1] as f64);
        let w = cells(0);
        let h = cells(1);
        let mut changed = false;
        let mut hits: Vec<(usize, usize)> = Vec::new();
        for level in (l0 as usize)..=(l1 as usize) {
            hits.clear();
            for cv in cv0..=cv1 {
                let y = (cv as f64 + 0.5) / sv - 0.5;
                for cu in cu0..=cu1 {
                    let x = (cu as f64 + 0.5) / su - 0.5;
                    let mut q = [0.0f64; 3];
                    q[ua] = x;
                    q[va] = y;
                    q[axis] = level as f64;
                    let d = to_mm([q[0] - a[0], q[1] - a[1], q[2] - a[2]]);
                    let t = if ab2 > 1e-12 {
                        (dot(d, ab) / ab2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let e = [d[0] - ab[0] * t, d[1] - ab[1] * t, d[2] - ab[2] * t];
                    if dot(e, e) > r2 {
                        continue;
                    }
                    let mut vox = [0usize; 3];
                    vox[ua] = (x.round().max(0.0) as usize).min(n[ua] - 1);
                    vox[va] = (y.round().max(0.0) as usize).min(n[va] - 1);
                    vox[axis] = level;
                    if keep(vox) {
                        hits.push((cu, cv));
                    }
                }
            }
            if hits.is_empty() {
                continue;
            }
            let s = self.slices.entry(level).or_insert_with(|| FootSlice {
                rows: vec![Vec::new(); h],
                lo: [usize::MAX; 2],
                hi: [0; 2],
                dirty: false,
                dlo: [usize::MAX; 2],
                dhi: [0; 2],
            });
            for &(cu, cv) in &hits {
                let row = &mut s.rows[cv];
                if row.is_empty() {
                    row.resize(w, 0);
                }
                if row[cu] == 0 {
                    row[cu] = 1;
                    s.lo = [s.lo[0].min(cu), s.lo[1].min(cv)];
                    s.hi = [s.hi[0].max(cu), s.hi[1].max(cv)];
                    s.dlo = [s.dlo[0].min(cu), s.dlo[1].min(cv)];
                    s.dhi = [s.dhi[0].max(cu), s.dhi[1].max(cv)];
                    s.dirty = true;
                    changed = true;
                }
            }
        }
        changed
    }

    /// The levels marked since the last call, in order; they count as
    /// taken afterwards.
    pub fn take_dirty(&mut self) -> Vec<usize> {
        self.take_dirty_boxes()
            .into_iter()
            .map(|(l, _)| l)
            .collect()
    }

    /// Whether `level` was marked since it was last taken; it counts as
    /// taken afterwards.
    pub fn take_dirty_at(&mut self, level: usize) -> bool {
        self.take_dirty_box_at(level).is_some()
    }

    /// [`Self::take_dirty`] with the box of the cells each level had marked
    /// since (`[u0, v0, u1, v1]`, inclusive), for [`Self::merge`].
    pub fn take_dirty_boxes(&mut self) -> Vec<(usize, [usize; 4])> {
        self.slices
            .iter_mut()
            .filter_map(|(l, s)| s.take().map(|b| (*l, b)))
            .collect()
    }

    /// [`Self::take_dirty_at`] with the box of what was marked since.
    pub fn take_dirty_box_at(&mut self, level: usize) -> Option<[usize; 4]> {
        self.slices.get_mut(&level).and_then(FootSlice::take)
    }

    /// Whether the point `(x, y)` (in-plane voxel coordinates) of a slice
    /// is inside the stroke's outline there ([`Self::region`]), asked of the
    /// four cells around it: the marching-squares outline between their
    /// centres, case by case, saddles split as the trace splits them.
    fn inside(&self, s: &FootSlice, x: f64, y: f64) -> bool {
        let cx = (x + 0.5) * self.sub[0] as f64 - 0.5;
        let cy = (y + 0.5) * self.sub[1] as f64 - 0.5;
        let (i, j) = (cx.floor(), cy.floor());
        let (u, v) = (cx - i, cy - j);
        let (i, j) = (i as i64, j as i64);
        let case = s.cell(i, j) as u8
            | (s.cell(i + 1, j) as u8) << 1
            | (s.cell(i + 1, j + 1) as u8) << 2
            | (s.cell(i, j + 1) as u8) << 3;
        let c00 = u + v < 0.5;
        let c10 = (1.0 - u) + v < 0.5;
        let c11 = (1.0 - u) + (1.0 - v) < 0.5;
        let c01 = u + (1.0 - v) < 0.5;
        match case {
            0 => false,
            15 => true,
            1 => c00,
            2 => c10,
            4 => c11,
            8 => c01,
            14 => !c00,
            13 => !c10,
            11 => !c11,
            7 => !c01,
            3 => v < 0.5,
            12 => v >= 0.5,
            9 => u < 0.5,
            6 => u >= 0.5,
            5 => c00 || c11,
            _ => c10 || c01,
        }
    }

    /// The slice `current` - this stroke's earlier merges into `base`, the
    /// slice as the stroke found it - with what the stroke marked on
    /// `level` inside `dirty` (a box from [`Self::take_dirty_boxes`]) merged
    /// in as well: the answer of [`Self::merged`] from `base`, made over a
    /// window around the box only. Outside it the new cells change nothing,
    /// so the outline there is `current`'s, vertex for vertex; inside, the
    /// cells are those of `base` and of the stroke (never of `current`'s
    /// outline of them), so however often a slice is merged nothing
    /// compounds. Where the splice cannot be made this is [`Self::merged`].
    pub fn merge(
        &self,
        level: usize,
        dirty: [usize; 4],
        base: &Region,
        current: &Region,
        erase: bool,
    ) -> Region {
        let Some(s) = self.slices.get(&level) else {
            return current.clone();
        };
        if dirty[0] > dirty[2] || dirty[1] > dirty[3] {
            return current.clone();
        }
        let (su, sv) = (self.sub[0] as f64, self.sub[1] as f64);
        // The new cells' extent in voxels, a cell wider each way: the
        // outline between a new cell and its neighbours moves too.
        let b = [
            (dirty[0] as f64 - 1.0) / su - 0.5,
            (dirty[1] as f64 - 1.0) / sv - 0.5,
            (dirty[2] as f64 + 2.0) / su - 0.5,
            (dirty[3] as f64 + 2.0) / sv - 0.5,
        ];
        for margin in [1.0, 2.5] {
            let Some(win) = Window::around(b, margin, BOOL_SS, BOOL_MAX_CELLS) else {
                break;
            };
            let cur = win.fill(current);
            let was = win.fill(base);
            let mut cells = vec![false; win.w * win.h];
            for j in 0..win.h {
                let y = win.y0 + (j as f64 + 0.5) * win.step;
                for i in 0..win.w {
                    let x = win.x0 + (i as f64 + 0.5) * win.step;
                    let k = j * win.w + i;
                    cells[k] = if erase {
                        was[k] && !self.inside(s, x, y)
                    } else {
                        was[k] || self.inside(s, x, y)
                    };
                }
            }
            if let Some(r) = splice_window(current, &win, &cells, &cur) {
                return r;
            }
        }
        self.merged(level, base, erase)
    }

    /// The stroke's outline on `level`, in the stack's in-plane voxel
    /// coordinates: the marching-squares trace of its cells, as a mask
    /// slice is traced everywhere else ([`region_of_mask`]), scaled back from
    /// cells to voxels. Only the box of the marked cells is traced.
    pub fn region(&self, level: usize) -> Region {
        let Some(s) = self.slices.get(&level).filter(|s| s.lo[0] != usize::MAX) else {
            return Region::new();
        };
        let (bw, bh) = (s.hi[0] - s.lo[0] + 1, s.hi[1] - s.lo[1] + 1);
        let mut field = PaddedField::new(bw, bh);
        for j in 0..bh {
            let row = &s.rows[j + s.lo[1]];
            if row.is_empty() {
                continue;
            }
            for i in 0..bw {
                if row[i + s.lo[0]] != 0 {
                    field.set(i, j);
                }
            }
        }
        let (su, sv) = (self.sub[0] as f64, self.sub[1] as f64);
        let (x0, y0) = (s.lo[0] as f64, s.lo[1] as f64);
        let mut region = Region::new();
        for pts in field.trace() {
            region.rings.push(Poly::new(
                pts.iter()
                    .map(|p| {
                        [
                            (x0 + p[0] as f64 + 0.5) / su - 0.5,
                            (y0 + p[1] as f64 + 0.5) / sv - 0.5,
                        ]
                    })
                    .collect(),
            ));
        }
        region.normalize();
        region
    }

    /// `base` - a slice as it was before the stroke - with the stroke's
    /// outline on `level` added (paint) or cut out (`erase`).
    pub fn merged(&self, level: usize, base: &Region, erase: bool) -> Region {
        let foot = self.region(level);
        if foot.is_empty() {
            return base.clone();
        }
        match (erase, base.is_empty()) {
            (true, true) => Region::new(),
            (false, true) => foot,
            (true, false) => base.boolean(&foot, BoolOp::Subtract),
            (false, false) => base.boolean(&foot, BoolOp::Union),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn rigid_moved_matches_the_exact_in_plane_shift_and_turns_a_box() {
        let dims = [40, 40, 20];
        let sp = [1.0, 1.0, 2.0];
        let mut st = Stack::empty(2);
        for level in 5..10 {
            let region = Region {
                rings: vec![Poly::new(vec![
                    [10.0, 10.0],
                    [20.0, 10.0],
                    [20.0, 14.0],
                    [10.0, 14.0],
                ])],
            };
            st.slices.push(SlicePolys { level, region });
        }
        let base = st.rasterize(dims);
        let n0 = base.iter().filter(|&&v| v != 0).count();
        // Two voxels along z: every slice moves up by one level.
        let moved = st.rigid_moved(dims, sp, &Rigid::translation([0.0, 0.0, 4.0]));
        let levels: Vec<usize> = moved.slices.iter().map(|s| s.level).collect();
        assert_eq!(levels, vec![7, 8, 9, 10, 11]);
        let n1 = moved.rasterize(dims).iter().filter(|&&v| v != 0).count();
        assert!(
            (n1 as f64 - n0 as f64).abs() <= n0 as f64 * 0.1,
            "{n0} vs {n1}"
        );
        // A quarter turn about z swaps the box's extent.
        let c = st.centroid3();
        let turned = st.rigid_moved(
            dims,
            sp,
            &Rigid::rotation_about(
                [0.0, 0.0, 1.0],
                [c[0], c[1], c[2] * sp[2]],
                std::f64::consts::FRAC_PI_2,
            ),
        );
        let (lo, hi) = turned.bbox(dims).unwrap();
        assert!(hi[1] - lo[1] > hi[0] - lo[0], "{lo:?} {hi:?}");
        // Off the lattice: nothing is left, and nothing panics.
        let gone = st.rigid_moved(dims, sp, &Rigid::translation([1000.0, 0.0, 0.0]));
        assert!(gone.slices.is_empty());
    }

    #[test]
    fn a_structure_pushed_out_of_the_field_of_view_is_not_cut() {
        let dims = [40, 40, 20];
        let sp = [1.0, 1.0, 2.0];
        let mut st = Stack::empty(2);
        for level in 5..10 {
            let region = Region {
                rings: vec![Poly::new(vec![
                    [10.0, 10.0],
                    [20.0, 10.0],
                    [20.0, 14.0],
                    [10.0, 14.0],
                ])],
            };
            st.slices.push(SlicePolys { level, region });
        }
        let grid = Grid {
            dims,
            spacing: sp,
            origin: Vec3::ZERO,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        };
        // Past the last slice: two stay, three wait off the lattice, and
        // the contours written out carry all five.
        st.translate_mm([0.0, 0.0, 26.0], sp, dims);
        assert_eq!(st.slices.len(), 2);
        assert_eq!(
            st.beyond.iter().map(|(l, _)| *l).collect::<Vec<_>>(),
            vec![20, 21, 22]
        );
        assert_eq!(st.to_contours(&grid).len(), 5);
        // In-plane moves carry the off-lattice slices along.
        st.translate([3.0, 0.0]);
        assert!((st.beyond[0].1.rings[0].centroid()[0] - 18.0).abs() < 1e-9);
        // Back in: everything on the lattice again.
        st.translate_mm([0.0, 0.0, -26.0], sp, dims);
        assert!(st.beyond.is_empty());
        assert_eq!(
            st.slices.iter().map(|s| s.level).collect::<Vec<_>>(),
            vec![5, 6, 7, 8, 9]
        );
        // A re-read of the written contours keeps the off-lattice ones.
        st.translate_mm([0.0, 0.0, 26.0], sp, dims);
        let mut roi = Roi {
            number: 1,
            name: "box".into(),
            color: [0; 3],
            roi_type: String::new(),
            description: String::new(),
            contours: Vec::new(),
        };
        st.apply_to_roi(&mut roi, &grid);
        let again = Stack::from_roi(&roi, &grid);
        assert_eq!((again.slices.len(), again.beyond.len()), (2, 3));
    }

    use super::*;

    /// Traced along their edges, the filled cells come back exactly: a
    /// rectangle as its four corners, a lone cell as a whole square (not
    /// the half a marching-squares diamond keeps), two cells touching at a
    /// corner as two squares, and a ring as its outline and its hole.
    #[test]
    fn voxel_outlines_enclose_exactly_the_cells() {
        let region = |f: &PaddedField| {
            let mut r = Region::new();
            for pts in f.trace_voxels() {
                r.rings.push(Poly::new(
                    pts.iter().map(|p| [p[0] as f64, p[1] as f64]).collect(),
                ));
            }
            r.normalize();
            r
        };
        let mut f = PaddedField::new(8, 8);
        for j in 1..4 {
            for i in 2..5 {
                f.set(i, j);
            }
        }
        let rings = f.trace_voxels();
        assert_eq!(rings.len(), 1);
        assert_eq!(rings[0].len(), 4, "a rectangle is its corners");
        let r = region(&f);
        assert!((r.area() - 9.0).abs() < 1e-9, "{}", r.area());
        assert_eq!(r.bbox(), [1.5, 0.5, 4.5, 3.5]);

        f.clear();
        f.set(3, 3);
        assert!((region(&f).area() - 1.0).abs() < 1e-9);

        f.clear();
        f.set(2, 2);
        f.set(3, 3);
        assert_eq!(
            f.trace_voxels().len(),
            2,
            "corner-touching cells stay apart"
        );
        assert!((region(&f).area() - 2.0).abs() < 1e-9);
        f.clear();
        f.set(3, 2);
        f.set(2, 3);
        assert_eq!(f.trace_voxels().len(), 2);
        assert!((region(&f).area() - 2.0).abs() < 1e-9);

        f.clear();
        for j in 1..4 {
            for i in 1..4 {
                if (i, j) != (2, 2) {
                    f.set(i, j);
                }
            }
        }
        assert_eq!(f.trace_voxels().len(), 2, "outline and hole");
        assert!((region(&f).area() - 8.0).abs() < 1e-9);
    }

    /// A contour that passes through a vertex twice is the loops it makes:
    /// two squares written as one path through their shared corner count
    /// both (the path's own shoelace area is zero); a closing point equal
    /// to the first is not a loop of its own; a plain ring stays whole.
    #[test]
    fn a_path_through_a_vertex_twice_is_split_into_its_loops() {
        let p = |x: f64, y: f64| [x, y, 0.0];
        let two = [
            p(0.0, 1.0),
            p(0.0, 0.0),
            p(1.0, 0.0),
            p(1.0, 1.0),
            p(0.0, 1.0),
            p(0.0, 2.0),
            p(1.0, 2.0),
            p(1.0, 1.0),
            p(0.0, 1.0),
        ];
        let loops = simple_loops(&two);
        assert_eq!(loops.len(), 2, "{loops:?}");
        assert!(loops.iter().all(|l| l.len() == 4));
        let closed = [
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(2.0, 2.0),
            p(0.0, 2.0),
            p(0.0, 0.0),
        ];
        assert_eq!(simple_loops(&closed).len(), 1);
        assert_eq!(simple_loops(&closed)[0].len(), 4);
        let open = [p(0.0, 0.0), p(2.0, 0.0), p(2.0, 2.0)];
        assert_eq!(simple_loops(&open), vec![open.to_vec()]);
    }

    /// A padded field traces a filled rectangle as one ring of four
    /// corners, shifted by the origin it is asked for, and an empty one as
    /// nothing; clearing it makes it reusable.
    #[test]
    fn a_padded_field_traces_a_rectangle_and_forgets_it() {
        let mut f = PaddedField::new(6, 5);
        assert!(f.is_empty());
        for j in 1..4 {
            for i in 2..5 {
                f.set(i, j);
            }
        }
        assert!(!f.is_empty());
        let rings = f.trace();
        assert_eq!(rings.len(), 1);
        // Marching squares chamfers the four corners: eight vertices, the
        // straight runs between them collapsed.
        assert_eq!(rings[0].len(), 8, "collinear runs are dropped");
        let region = f.trace_region(10.0, 20.0);
        assert!(
            (region.area() - 8.5).abs() < 1e-6,
            "3 x 3 cells less four corners"
        );
        let b = region.bbox();
        assert!(
            (b[0] - 11.5).abs() < 1e-6 && (b[1] - 20.5).abs() < 1e-6,
            "{b:?}"
        );
        f.clear();
        assert!(f.is_empty());
        assert!(f.trace().is_empty());
    }

    fn grid(dims: [usize; 3], spacing: [f64; 3]) -> Grid {
        Grid {
            dims,
            spacing,
            origin: Vec3::new(-100.0, -50.0, 20.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: "1.2.3".into(),
        }
    }

    /// A ball of radius `r` voxels centred in `dims`.
    fn ball(dims: [usize; 3], r: f64) -> Vec<u8> {
        let [nx, ny, nz] = dims;
        let c = [
            (nx - 1) as f64 / 2.0,
            (ny - 1) as f64 / 2.0,
            (nz - 1) as f64 / 2.0,
        ];
        let mut m = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let d = (i as f64 - c[0]).powi(2)
                        + (j as f64 - c[1]).powi(2)
                        + (k as f64 - c[2]).powi(2);
                    if d <= r * r {
                        m[k * nx * ny + j * nx + i] = 1;
                    }
                }
            }
        }
        m
    }

    fn square(x: f64, y: f64, s: f64) -> Poly {
        Poly::new(vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]])
    }

    #[test]
    fn a_square_has_the_area_orientation_and_centroid_it_should() {
        let p = square(2.0, 3.0, 4.0);
        assert!((p.area() - 16.0).abs() < 1e-9);
        assert!(p.is_ccw());
        let c = p.centroid();
        assert!((c[0] - 4.0).abs() < 1e-9 && (c[1] - 5.0).abs() < 1e-9);
        let mut q = p.clone();
        q.orient(false);
        assert!(!q.is_ccw());
        assert!((q.area() - 16.0).abs() < 1e-9);
        assert!((p.perimeter() - 16.0).abs() < 1e-9);
    }

    #[test]
    fn even_odd_makes_the_inner_ring_a_hole() {
        let mut r = Region::new();
        r.rings.push(square(0.0, 0.0, 10.0));
        r.rings.push(square(3.0, 3.0, 4.0));
        r.normalize();
        assert!(r.contains([1.0, 1.0]));
        assert!(!r.contains([5.0, 5.0]), "the inner ring is a hole");
        assert!((r.area() - (100.0 - 16.0)).abs() < 1e-9);
        // Orientation follows nesting: outer counter-clockwise, hole not.
        assert!(r.rings[0].is_ccw());
        assert!(!r.rings[1].is_ccw());
    }

    #[test]
    fn douglas_peucker_keeps_the_corners_and_drops_the_rest() {
        // A square walked in 1/10 steps: 40 points for 4 corners.
        let mut pts = Vec::new();
        for i in 0..10 {
            pts.push([i as f64, 0.0]);
        }
        for i in 0..10 {
            pts.push([10.0, i as f64]);
        }
        for i in 0..10 {
            pts.push([10.0 - i as f64, 10.0]);
        }
        for i in 0..10 {
            pts.push([0.0, 10.0 - i as f64]);
        }
        let p = Poly::new(pts);
        let s = p.simplify(0.01);
        assert_eq!(s.len(), 4, "only the corners survive");
        assert!((s.area() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn the_boolean_agrees_with_the_analytic_answer() {
        // Two circles of radius 10 whose centres are 10 apart. The lens has
        // area 2 r^2 (acos(d/2r) - (d/4r) sqrt(4r^2 - d^2) / r) ...
        let (r, d) = (10.0f64, 10.0f64);
        let lens = 2.0 * r * r * (d / (2.0 * r)).acos() - 0.5 * d * (4.0 * r * r - d * d).sqrt();
        let a = Region::from_ring(Poly::circle([0.0, 0.0], r, 256));
        let b = Region::from_ring(Poly::circle([d, 0.0], r, 256));
        let disc = std::f64::consts::PI * r * r;

        let inter = a.boolean(&b, BoolOp::Intersect);
        assert!(
            (inter.area() - lens).abs() < 0.02 * lens,
            "intersection {} vs analytic {lens}",
            inter.area()
        );
        let union = a.boolean(&b, BoolOp::Union);
        assert!(
            (union.area() - (2.0 * disc - lens)).abs() < 0.01 * disc,
            "union {}",
            union.area()
        );
        let diff = a.boolean(&b, BoolOp::Subtract);
        assert!(
            (diff.area() - (disc - lens)).abs() < 0.02 * lens,
            "difference {}",
            diff.area()
        );
        let xor = a.boolean(&b, BoolOp::Xor);
        assert!(
            (xor.area() - 2.0 * (disc - lens)).abs() < 0.03 * disc,
            "xor {}",
            xor.area()
        );
        // A union of two overlapping discs is one ring.
        assert_eq!(union.rings.len(), 1);
    }

    #[test]
    fn a_subtraction_that_cuts_a_hole_produces_a_hole() {
        let a = Region::from_ring(Poly::circle([0.0, 0.0], 10.0, 256));
        let b = Region::from_ring(Poly::circle([0.0, 0.0], 3.0, 128));
        let out = a.boolean(&b, BoolOp::Subtract);
        assert_eq!(out.rings.len(), 2, "outline plus hole");
        assert!(!out.contains([0.0, 0.0]));
        assert!(out.contains([6.0, 0.0]));
        let want = std::f64::consts::PI * (100.0 - 9.0);
        assert!((out.area() - want).abs() < 0.02 * want, "{}", out.area());
    }

    #[test]
    fn the_exact_paths_leave_untouched_rings_untouched() {
        let mut r = Region::from_ring(Poly::circle([0.0, 0.0], 10.0, 64));
        let before = r.rings[0].clone();

        // Disjoint: appended, the first ring is not even re-sampled.
        r.add_ring(&Poly::circle([40.0, 0.0], 5.0, 64));
        assert_eq!(r.rings.len(), 2);
        assert_eq!(r.rings[0], before);

        // Wholly inside the filled area: nothing changes at all.
        r.add_ring(&Poly::circle([0.0, 0.0], 3.0, 64));
        assert_eq!(r.rings.len(), 2);
        assert_eq!(r.rings[0], before);

        // Wholly inside, subtracted: becomes a hole, exactly.
        r.subtract_ring(&Poly::circle([0.0, 0.0], 3.0, 64));
        assert_eq!(r.rings.len(), 3);
        assert_eq!(r.rings[0], before);
        assert!(!r.contains([0.0, 0.0]));

        // ... and filling the hole again removes it, still exactly.
        r.add_ring(&Poly::circle([0.0, 0.0], 3.0, 64));
        assert!(r.contains([0.0, 0.0]));
        assert_eq!(r.rings[0], before);
    }

    #[test]
    fn a_stroke_that_crosses_the_outline_goes_through_the_boolean() {
        let mut r = Region::from_ring(Poly::circle([0.0, 0.0], 10.0, 256));
        let before = r.area();
        r.add_ring(&Poly::circle([10.0, 0.0], 4.0, 128));
        assert_eq!(r.rings.len(), 1, "one merged outline");
        assert!(r.area() > before, "the stroke added area");
        assert!(r.contains([12.0, 0.0]));
    }

    #[test]
    fn tidying_does_what_it_says() {
        let mut r = Region::new();
        r.rings.push(square(0.0, 0.0, 10.0));
        r.rings.push(square(3.0, 3.0, 2.0)); // hole
        r.rings.push(square(40.0, 0.0, 1.0)); // speck
        r.normalize();

        let mut a = r.clone();
        a.remove_holes();
        assert_eq!(a.rings.len(), 2);
        assert!(a.contains([4.0, 4.0]));

        let mut b = r.clone();
        b.drop_smaller_than(2.0);
        assert_eq!(b.rings.len(), 2, "the speck goes, the hole stays");
        assert!(!b.contains([4.0, 4.0]));

        let mut c = r.clone();
        c.keep_largest();
        assert_eq!(c.rings.len(), 2, "the big square with its hole");
        assert!(!c.contains([40.5, 0.5]));
    }

    #[test]
    fn a_mask_round_trips_through_contours() {
        let dims = [40, 40, 20];
        let mask = ball(dims, 12.0);
        let set: usize = mask.iter().filter(|&&v| v != 0).count();
        let st = Stack::from_mask(&mask, dims, 2);
        assert!(st.occupied() > 10);
        let back = st.rasterize(dims);
        let same = back.iter().zip(&mask).filter(|(a, b)| a == b).count();
        assert_eq!(same, mask.len(), "every voxel comes back");
        assert_eq!(back.iter().filter(|&&v| v != 0).count(), set);
    }

    #[test]
    fn contours_survive_the_trip_through_patient_coordinates() {
        let dims = [40, 40, 20];
        let g = grid(dims, [1.5, 1.5, 3.0]);
        let st = Stack::from_mask(&ball(dims, 12.0), dims, 2);
        let vol = st.volume_cm3(g.spacing);
        let mut roi = Roi {
            number: 1,
            name: "ball".into(),
            color: [255, 0, 0],
            roi_type: "ORGAN".into(),
            description: String::new(),
            contours: Vec::new(),
        };
        st.apply_to_roi(&mut roi, &g);
        assert!(!roi.contours.is_empty());
        let back = Stack::from_roi(&roi, &g);
        assert_eq!(back.axis, 2);
        assert_eq!(back.occupied(), st.occupied());
        assert!(
            (back.volume_cm3(g.spacing) - vol).abs() < 1e-6 * vol.max(1.0),
            "{} vs {vol}",
            back.volume_cm3(g.spacing)
        );
    }

    /// `to_axis` as it was: the whole lattice rasterised, then traced.
    fn to_axis_whole(st: &Stack, dims: [usize; 3], axis: usize) -> Stack {
        Stack::from_mask(&st.rasterize(dims), dims, axis)
    }

    fn same_stacks(a: &Stack, b: &Stack) {
        assert_eq!(a.axis, b.axis);
        assert_eq!(a.slices.len(), b.slices.len());
        for (x, y) in a.slices.iter().zip(&b.slices) {
            assert_eq!(x.level, y.level);
            assert_eq!(x.region.rings.len(), y.region.rings.len());
            for (r, q) in x.region.rings.iter().zip(&y.region.rings) {
                assert_eq!(r.pts, q.pts, "level {}", x.level);
            }
        }
    }

    #[test]
    fn re_slicing_the_occupied_box_is_re_slicing_the_whole_lattice() {
        let dims = [57, 43, 31];
        // Off-centre geometry with a hole, a second island and vertices off
        // the lattice, on every stacking axis, to every other axis.
        for axis in 0..3 {
            let [ua, va] = plane_axes(axis);
            let (nu, nv) = (dims[ua] as f64, dims[va] as f64);
            let mut st = Stack::empty(axis);
            for level in 3..dims[axis] - 5 {
                let t = level as f64 * 0.37;
                let mut region = Region::new();
                region.add_ring(&Poly::circle(
                    [nu * 0.55 + t.sin(), nv * 0.4],
                    7.3 + t.cos(),
                    40,
                ));
                region.subtract_ring(&Poly::circle([nu * 0.55, nv * 0.4], 2.6, 24));
                if level % 3 == 0 {
                    region.add_ring(&Poly::capsule([3.2, nv - 6.1], [9.7, nv - 3.4], 1.9, 16));
                }
                *st.region_mut(level) = region;
            }
            st.prune();
            for to in 0..3 {
                if to == axis {
                    continue;
                }
                same_stacks(&st.to_axis(dims, to), &to_axis_whole(&st, dims, to));
            }
        }
        // Nothing to re-slice.
        assert!(Stack::empty(0).to_axis(dims, 2).slices.is_empty());
    }

    #[test]
    fn a_sagittal_stack_converts_to_axial_with_its_volume() {
        let dims = [40, 40, 24];
        let mask = ball(dims, 11.0);
        let sag = Stack::from_mask(&mask, dims, 0);
        assert_eq!(sag.axis, 0);
        let sp = [1.0, 1.0, 1.0];
        let ax = sag.to_axis(dims, 2);
        assert_eq!(ax.axis, 2);
        let (a, b) = (sag.volume_cm3(sp), ax.volume_cm3(sp));
        assert!((a - b).abs() < 0.05 * a, "{a} vs {b}");
        // And a stack read back from a sagittal ROI is recognised as sagittal.
        let g = grid(dims, sp);
        let mut roi = Roi {
            number: 1,
            name: "s".into(),
            color: [0, 255, 0],
            roi_type: "ORGAN".into(),
            description: String::new(),
            contours: sag.to_contours(&g),
        };
        assert_eq!(Stack::from_roi(&roi, &g).axis, 0);
        // apply_to_roi converts on the way out, as the writer requires.
        sag.apply_to_roi(&mut roi, &g);
        assert_eq!(Stack::from_roi(&roi, &g).axis, 2);
    }

    #[test]
    fn interpolation_fills_the_gaps_it_is_given() {
        let dims = [40, 40, 21];
        let full = Stack::from_mask(&ball(dims, 12.0), dims, 2);
        let mut sparse = Stack::empty(2);
        for s in &full.slices {
            if s.level % 4 == 0 {
                sparse.slices.push(s.clone());
            }
        }
        assert!(sparse.occupied() >= 3);
        let filled = sparse.interpolated(dims);
        assert!(!filled.is_empty());
        // Every new slice is one the sparse stack did not have.
        for s in &filled.slices {
            assert!(sparse.region_at(s.level).is_none());
        }
        let mut merged = sparse.clone();
        for s in &filled.slices {
            *merged.region_mut(s.level) = s.region.clone();
        }
        let sp = [1.0, 1.0, 1.0];
        let (want, got) = (full.volume_cm3(sp), merged.volume_cm3(sp));
        assert!(
            (got - want).abs() < 0.06 * want,
            "interpolated {got} vs drawn {want}"
        );
    }

    #[test]
    fn components_can_be_kept_or_deleted() {
        let dims = [40, 20, 12];
        let [nx, ny, nz] = dims;
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 2..8 {
            for j in 6..14 {
                for i in 3..9 {
                    mask[k * nx * ny + j * nx + i] = 1;
                }
                for i in 25..35 {
                    mask[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
        let st = Stack::from_mask(&mask, dims, 2);
        let big = st.component_at(dims, [30, 10, 5], true);
        assert!(big.volume_cm3([1.0; 3]) > 0.0);
        let sp = [1.0, 1.0, 1.0];
        let kept = big.volume_cm3(sp);
        let whole = st.volume_cm3(sp);
        assert!(
            kept < whole * 0.75 && kept > whole * 0.4,
            "{kept} of {whole}"
        );
        let cut = st.component_at(dims, [30, 10, 5], false);
        assert!((cut.volume_cm3(sp) + kept - whole).abs() < 0.02 * whole);
    }

    #[test]
    fn limit_points_hits_the_cap_without_wrecking_the_shape() {
        let dims = [60, 60, 3];
        let mut st = Stack::from_mask(&ball(dims, 25.0), dims, 2);
        let before = st.volume_cm3([1.0; 3]);
        assert!(st.slices[1].region.rings[0].len() > 40);
        st.limit_points(24);
        for s in &st.slices {
            for r in &s.region.rings {
                assert!(r.len() <= 24, "{} points", r.len());
            }
        }
        let after = st.volume_cm3([1.0; 3]);
        assert!(
            (after - before).abs() < 0.05 * before,
            "{after} vs {before}"
        );
    }

    #[test]
    fn a_brush_stamp_is_round_on_the_patient_not_on_the_lattice() {
        // 1 mm across, 3 mm through: the ring has to be three times wider in
        // v units than in u units to be a disc of 6 mm on the patient.
        let mm = [1.0, 3.0];
        let r = 6.0;
        let p = Poly::capsule_mm([10.0, 4.0], [10.0, 4.0], r, mm, 64);
        let b = p.bbox();
        assert!(((b[2] - b[0]) - 2.0 * r / mm[0]).abs() < 0.05, "{b:?}");
        assert!(((b[3] - b[1]) - 2.0 * r / mm[1]).abs() < 0.05, "{b:?}");
        // Every vertex is on the circle, measured in millimetres.
        for q in &p.pts {
            let d = (((q[0] - 10.0) * mm[0]).powi(2) + ((q[1] - 4.0) * mm[1]).powi(2)).sqrt();
            assert!((d - r).abs() < 1e-6, "{d}");
        }
        // A swept stroke is the disc plus the rectangle between the ends.
        let sweep = Poly::capsule_mm([10.0, 4.0], [20.0, 4.0], r, mm, 64);
        let area_mm2 = sweep.area() * mm[0] * mm[1];
        let want = std::f64::consts::PI * r * r + 2.0 * r * 10.0 * mm[0];
        assert!(
            (area_mm2 - want).abs() < 0.02 * want,
            "{area_mm2} vs {want}"
        );
    }

    #[test]
    fn a_nudge_pushes_the_outline_and_leaves_the_far_side_alone() {
        let mut r = Region::from_ring(Poly::circle([0.0, 0.0], 20.0, 240));
        let before = r.rings[0].clone();
        let area0 = r.area();
        // Push the rightmost point 3 units further right, over a 6 mm reach
        // on a lattice of 1 mm in u and 2 mm in v.
        let moved = r.nudge([20.0, 0.0], [23.0, 0.0], 6.0, [1.0, 2.0]);
        assert!(moved);
        assert!(r.area() > area0, "the bulge adds area");

        let after = &r.rings[0];
        assert_eq!(after.pts.len(), before.pts.len(), "no vertex is added");
        // Every vertex out of reach is untouched, bit for bit.
        let mut far = 0;
        for (j, p) in before.pts.iter().enumerate() {
            let d = ((p[0] - 20.0).powi(2) + (p[1] * 2.0).powi(2)).sqrt();
            if d >= 6.0 {
                assert_eq!(after.pts[j], *p, "vertex {j} was out of reach");
                far += 1;
            }
        }
        assert!(far > 200, "most of the ring is out of reach");
        // The vertex under the pointer follows the drag in full.
        let i = (0..before.pts.len())
            .min_by(|&a, &b| {
                let da = (before.pts[a][0] - 20.0).abs() + before.pts[a][1].abs();
                let db = (before.pts[b][0] - 20.0).abs() + before.pts[b][1].abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        assert!(
            (after.pts[i][0] - before.pts[i][0] - 3.0).abs() < 0.02,
            "moved by {}",
            after.pts[i][0] - before.pts[i][0]
        );
        // A drag of nothing does nothing.
        let mut q = r.clone();
        assert!(!q.nudge([0.0, 0.0], [0.0, 0.0], 6.0, [1.0, 1.0]));
    }

    #[test]
    fn the_transforms_move_what_they_say() {
        let dims = [40, 40, 6];
        let mut st = Stack::from_mask(&ball(dims, 10.0), dims, 2);
        let c0 = st.centroid();
        let v0 = st.volume_cm3([1.0; 3]);
        st.translate([3.0, -2.0]);
        let c1 = st.centroid();
        assert!((c1[0] - c0[0] - 3.0).abs() < 1e-9 && (c1[1] - c0[1] + 2.0).abs() < 1e-9);
        assert!((st.volume_cm3([1.0; 3]) - v0).abs() < 1e-9);
        st.scale(2.0);
        assert!((st.volume_cm3([1.0; 3]) - 4.0 * v0).abs() < 1e-6 * v0);
        let c2 = st.centroid();
        assert!((c2[0] - c1[0]).abs() < 1e-9, "scaling keeps the centroid");
        st.rotate(std::f64::consts::FRAC_PI_2);
        assert!((st.volume_cm3([1.0; 3]) - 4.0 * v0).abs() < 1e-6 * v0);
    }

    /// A mask slice read as an outline and filled back is the same slice,
    /// to the cell: islands, a hole, a single cell, cells touching only at
    /// a corner, and cells on the border of the raster.
    #[test]
    fn a_mask_slice_round_trips_through_its_outline() {
        let (w, h) = (24usize, 18usize);
        let mut m = vec![0u8; w * h];
        let mut set = |i: usize, j: usize| m[j * w + i] = 1;
        // A ring with a hole.
        for j in 3..12 {
            for i in 3..12 {
                if !(6..9).contains(&i) || !(6..9).contains(&j) {
                    set(i, j);
                }
            }
        }
        // A cell on its own, a diagonal pair, and the raster's corners.
        set(16, 4);
        set(18, 8);
        set(19, 9);
        set(0, 0);
        set(w - 1, h - 1);
        set(w - 1, 0);
        let region = region_of_mask(&m, w, h);
        assert!(!region.is_empty());
        assert_eq!(region.rasterize(w, h), m);
        // Nothing in, nothing out.
        assert!(region_of_mask(&vec![0u8; w * h], w, h).is_empty());
        assert!(Region::new().rasterize(w, h).iter().all(|&v| v == 0));
    }

    /// A drawn ring fills the cells whose centre it encloses, and a ring
    /// partly off the raster fills what is on it without a panic.
    #[test]
    fn a_ring_rasterizes_to_the_cells_whose_centre_it_holds() {
        let (w, h) = (10usize, 8usize);
        // From 1.5 to 4.5: the centres 2, 3 and 4 on each axis.
        let got = Region::from_ring(square(1.5, 1.5, 3.0)).rasterize(w, h);
        let n = got.iter().filter(|&&v| v != 0).count();
        assert_eq!(n, 9);
        assert_eq!(got[2 * w + 2], 1);
        assert_eq!(got[4 * w + 4], 1);
        assert_eq!(got[w + 1], 0);
        let off = Region::from_ring(square(-5.0, -5.0, 6.5)).rasterize(w, h);
        assert_eq!(off.iter().filter(|&&v| v != 0).count(), 4);
    }

    /// Contours of a set of voxels traced in their own box are the
    /// contours of the whole mask.
    #[test]
    fn contours_of_voxels_are_those_of_their_mask() {
        let dims = [30, 26, 9];
        let mask = ball(dims, 6.0);
        let voxels: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0)
            .map(|(i, _)| i as u32)
            .collect();
        for axis in 0..3 {
            let a = Stack::from_mask(&mask, dims, axis);
            let b = Stack::from_voxels(&voxels, dims, axis);
            let la: Vec<usize> = a.slices.iter().map(|s| s.level).collect();
            let lb: Vec<usize> = b.slices.iter().map(|s| s.level).collect();
            assert_eq!(la, lb, "axis {axis}");
            assert_eq!(a.rasterize(dims), b.rasterize(dims), "axis {axis}");
            assert_eq!(b.rasterize(dims), mask, "axis {axis}");
        }
        assert!(Stack::from_voxels(&[], dims, 2).is_empty());
    }

    /// The area of a footprint slice in mm², from its outline.
    fn foot_area(f: &BallFootprint, level: usize, sp: [f64; 3]) -> f64 {
        let [ua, va] = plane_axes(f.axis());
        f.region(level).area() * sp[ua] * sp[va]
    }

    fn near(got: f64, want: f64, rel: f64) -> bool {
        (got - want).abs() <= rel * want
    }

    /// A ball on an isotropic lattice reaches the slices within its radius
    /// and cuts each in the circle where the sphere crosses it.
    #[test]
    fn a_ball_cuts_each_slice_it_reaches_in_its_circle() {
        let sp = [1.0, 1.0, 1.0];
        let g = grid([40, 40, 30], sp);
        let mut f = BallFootprint::new(&g, 2, 5.5);
        assert!(f.is_empty());
        let c = [20.0, 20.0, 15.0];
        assert!(f.sweep(c, c, 5.5, |_| true));
        assert_eq!(f.levels(), (10..=20).collect::<Vec<_>>());
        let disc = |h: f64| std::f64::consts::PI * (5.5f64 * 5.5 - h * h);
        for (level, h) in [(15, 0.0), (18, 3.0), (12, 3.0), (10, 5.0)] {
            let a = foot_area(&f, level, sp);
            assert!(near(a, disc(h), 0.04), "level {level}: {a} vs {}", disc(h));
        }
        // The same sample again marks nothing new.
        assert_eq!(f.take_dirty().len(), 11);
        assert!(f.take_dirty().is_empty());
        assert!(!f.sweep(c, c, 5.5, |_| true));
        assert!(!f.take_dirty_at(15));
    }

    /// On a 3 mm lattice the ball reaches one slice either side, and its
    /// disc there is the sphere's circle in millimetres, not in voxels; a
    /// sagittal stack of the same lattice gets an ellipse on the lattice.
    #[test]
    fn a_ball_follows_the_spacing_along_the_slice_normal() {
        let sp = [0.8, 0.8, 3.0];
        let g = grid([60, 60, 30], sp);
        let c = [30.0, 30.0, 15.0];
        let mut f = BallFootprint::new(&g, 2, 5.5);
        f.sweep(c, c, 5.5, |_| true);
        assert_eq!(f.levels(), vec![14, 15, 16]);
        let a = foot_area(&f, 16, sp);
        let want = std::f64::consts::PI * (5.5f64 * 5.5 - 9.0);
        assert!(near(a, want, 0.04), "{a} vs {want}");
        // Sideways: sagittal slices are 0.8 mm apart, and each is a disc in
        // millimetres - an ellipse three and three quarters as wide in j as
        // in k on the lattice.
        let mut s = BallFootprint::new(&g, 0, 5.5);
        s.sweep(c, c, 5.5, |_| true);
        assert_eq!(s.levels().len(), 2 * 6 + 1);
        let a = foot_area(&s, 30, sp);
        let want = std::f64::consts::PI * 5.5 * 5.5;
        assert!(near(a, want, 0.04), "{a} vs {want}");
        let b = s.region(30).bbox();
        let (wj, wk) = ((b[2] - b[0]) * sp[1], (b[3] - b[1]) * sp[2]);
        assert!(
            (wj - 11.0).abs() < 0.8 && (wk - 11.0).abs() < 1.6,
            "{wj} x {wk}"
        );
    }

    /// A tilted acquisition: the slices are stacked along a direction that
    /// is not their normal. The ball reaches as far as the spacing along the
    /// normal allows, and the circle on a neighbouring slice is shifted by
    /// the tilt, as the sphere really crosses it.
    #[test]
    fn a_ball_on_a_tilted_lattice_is_cut_where_the_sphere_crosses() {
        let t = 20f64.to_radians();
        let sp = [1.0, 1.0, 2.0];
        let mut g = grid([40, 40, 30], sp);
        g.normal = Vec3::new(0.0, t.sin(), t.cos());
        let c = [20.0, 20.0, 15.0];
        let mut f = BallFootprint::new(&g, 2, 5.5);
        f.sweep(c, c, 5.5, |_| true);
        // 2 mm along the stack is 1.88 mm along the normal: two slices out.
        assert_eq!(f.levels(), (13..=17).collect::<Vec<_>>());
        let h = sp[2] * t.cos();
        let shift = sp[2] * t.sin();
        let want = std::f64::consts::PI * (5.5f64 * 5.5 - h * h);
        let a = foot_area(&f, 16, sp);
        assert!(near(a, want, 0.04), "{a} vs {want}");
        let b = f.region(16).bbox();
        let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
        assert!((cx - 20.0).abs() < 0.1, "{cx}");
        assert!(
            (cy - (20.0 - shift)).abs() < 0.1,
            "{cy} vs {}",
            20.0 - shift
        );
        let b = f.region(14).bbox();
        assert!(((b[1] + b[3]) / 2.0 - (20.0 + shift)).abs() < 0.1);
    }

    /// The edge band is asked voxel by voxel, and a swept ball is a
    /// capsule on the slice it was drawn on.
    #[test]
    fn a_swept_ball_is_a_capsule_and_keeps_only_what_the_band_allows() {
        let sp = [1.0, 1.0, 1.0];
        let g = grid([60, 40, 20], sp);
        let (a, b) = ([15.0, 20.0, 10.0], [35.0, 20.0, 10.0]);
        let mut f = BallFootprint::new(&g, 2, 4.0);
        f.sweep(a, b, 4.0, |_| true);
        let want = 20.0 * 8.0 + std::f64::consts::PI * 16.0;
        let got = foot_area(&f, 10, sp);
        assert!(near(got, want, 0.03), "{got} vs {want}");
        // Only the voxels left of i = 25 are allowed.
        let mut half = BallFootprint::new(&g, 2, 4.0);
        half.sweep(a, b, 4.0, |v| v[0] < 25);
        let got = foot_area(&half, 10, sp);
        let r = half.region(10).bbox();
        assert!(near(got, want / 2.0, 0.05), "{got} vs {}", want / 2.0);
        assert!(r[2] <= 24.6, "{r:?}");
        // A band that allows nothing marks nothing.
        let mut none = BallFootprint::new(&g, 2, 4.0);
        assert!(!none.sweep(a, b, 4.0, |_| false));
        assert!(none.is_empty());
    }

    /// Merging adds the stroke to a slice or cuts it out, always from the
    /// slice as it was, and leaves a slice the stroke never reached as it
    /// is, vertices and all.
    #[test]
    fn a_footprint_merges_into_the_slice_it_started_from() {
        let sp = [1.0, 1.0, 1.0];
        let g = grid([60, 60, 20], sp);
        let base = Region::from_ring(square(10.0, 10.0, 30.0));
        let mut f = BallFootprint::new(&g, 2, 5.0);
        // Inside the square: erasing makes a hole of the disc.
        f.sweep([25.0, 25.0, 10.0], [25.0, 25.0, 10.0], 5.0, |_| true);
        let disc = std::f64::consts::PI * 25.0;
        let cut = f.merged(10, &base, true);
        assert!(near(cut.area(), 900.0 - disc, 0.01), "{}", cut.area());
        let inside = f.merged(10, &base, false).area();
        assert!(near(inside, 900.0, 0.005), "{inside}");
        // On an empty slice, paint is the disc and erase is nothing.
        assert!(near(f.merged(10, &Region::new(), false).area(), disc, 0.03));
        assert!(f.merged(10, &Region::new(), true).is_empty());
        // Off the ball, the slice comes back untouched.
        let far = f.merged(2, &base, true);
        assert_eq!(far.rings[0].pts, base.rings[0].pts);
        // Half over the edge: painting pushes the outline out by the half
        // disc outside it, and does it the same however often it merges.
        let mut edge = BallFootprint::new(&g, 2, 5.0);
        edge.sweep([40.0, 25.0, 10.0], [40.0, 25.0, 10.0], 5.0, |_| true);
        let once = edge.merged(10, &base, false);
        assert!(
            near(once.area(), 900.0 + disc / 2.0, 0.02),
            "{}",
            once.area()
        );
        assert!((edge.merged(10, &base, false).area() - once.area()).abs() < 1e-9);
    }

    /// A many-vertex circle, the outline of something body-sized.
    fn big_circle(c: Pt, r: f64, n: usize) -> Region {
        let mut p = Poly::circle(c, r, n);
        p.orient(true);
        Region::from_ring(p)
    }

    /// The traced outline of a window's cells is wound with the cells on
    /// its left, a hole the other way, and an edge of the window ends a
    /// chain rather than closing a loop.
    #[test]
    fn a_window_traces_its_cells_with_the_filled_side_on_the_left() {
        let (w, h) = (8, 8);
        let mut cells = vec![false; w * h];
        for j in 2..6 {
            for i in 2..6 {
                cells[j * w + i] = !(i == 3 && j == 3);
            }
        }
        let (chains, loops) = trace_window(&cells, w, h).unwrap();
        assert!(chains.is_empty());
        assert_eq!(loops.len(), 2);
        let area = |l: &Vec<Key>| {
            Poly::new(l.iter().map(|k| [k.0 as f64, k.1 as f64]).collect()).signed_area()
        };
        let mut a: Vec<f64> = loops.iter().map(area).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!(a[0] < 0.0 && a[1] > 0.0, "{a:?}");
        // A block running off the right edge: one chain, border to border.
        let mut cells = vec![false; w * h];
        for j in 2..6 {
            for i in 4..8 {
                cells[j * w + i] = true;
            }
        }
        let (chains, loops) = trace_window(&cells, w, h).unwrap();
        assert!(loops.is_empty());
        assert_eq!(chains.len(), 1);
        let (s, e) = (chains[0][0], *chains[0].last().unwrap());
        assert!(s.0 == 14 && e.0 == 14, "{s:?} {e:?}");
    }

    /// The windowed boolean gives the whole raster's answer - the same
    /// area, the same voxels - and keeps every vertex of the outline it did
    /// not reach exactly where it was: a capsule painted over and erased
    /// from the edge of a large circle, across a long edge of a square (an
    /// edge that enters and leaves the window without a vertex in it), and
    /// into a hole.
    #[test]
    fn a_windowed_boolean_matches_the_whole_raster_and_keeps_the_rest() {
        let body = big_circle([200.0, 160.0], 150.0, 1500);
        // Off the voxel centres, so their rasters do not hang on a tie.
        let mut square_hole = Region::from_ring(square(20.3, 20.3, 300.0));
        let mut hole = square(150.3, 150.3, 40.0);
        hole.orient(false);
        square_hole.rings.push(hole);
        square_hole.normalize();
        let cases = [
            (
                &body,
                Poly::capsule([345.0, 160.0], [352.0, 175.0], 6.0, 40),
            ),
            (&body, Poly::capsule([200.0, 8.0], [215.0, 12.0], 9.0, 40)),
            (
                &square_hole,
                Poly::capsule([170.0, 15.0], [175.0, 25.0], 7.0, 40),
            ),
            (
                &square_hole,
                Poly::capsule([150.0, 170.0], [160.0, 175.0], 5.0, 40),
            ),
        ];
        for (n, (region, stamp)) in cases.iter().enumerate() {
            let b = Region::from_ring(stamp.clone());
            for op in [BoolOp::Union, BoolOp::Subtract] {
                let whole = region.boolean(&b, op);
                let local = window_boolean(region, &b, op, BOOL_SS)
                    .unwrap_or_else(|| panic!("case {n} {op:?}: no splice"));
                // The change in area is the stamp's less what it shares with
                // the region (a boolean over the stamp's box alone, whose
                // raster is exact to a fraction of a cell); the whole
                // raster re-traces everything and is off by its own bias.
                let common = region.boolean(&b, BoolOp::Intersect).area();
                let want = match op {
                    BoolOp::Union => b.area() - common,
                    _ => -common,
                };
                let got = local.area() - region.area();
                assert!(got.abs() > 1.0, "case {n} {op:?}: nothing changed");
                // Each raster puts an outline within half a cell (1/16 of a
                // voxel) of the true one.
                let slack = 0.5 + stamp.perimeter() / 16.0;
                assert!(
                    (got - want).abs() < slack,
                    "case {n} {op:?}: {got} vs {want}"
                );
                // Around the stamp both are the trace of the same cells.
                let sb = stamp.bbox();
                let (w, h) = (360, 360);
                let (rw, rl) = (whole.rasterize(w, h), local.rasterize(w, h));
                let differ = (0..w * h)
                    .filter(|&k| {
                        let (x, y) = ((k % w) as f64, (k / w) as f64);
                        x > sb[0] - 2.0 && x < sb[2] + 2.0 && y > sb[1] - 2.0 && y < sb[3] + 2.0
                    })
                    .filter(|&k| rw[k] != rl[k])
                    .count();
                assert!(differ <= 4, "case {n} {op:?}: {differ} voxels differ");
                // Far from the stamp, the vertices are the original ones.
                let far = |p: &Pt| {
                    p[0] < sb[0] - 4.0
                        || p[0] > sb[2] + 4.0
                        || p[1] < sb[1] - 4.0
                        || p[1] > sb[3] + 4.0
                };
                for ring in &region.rings {
                    for p in ring.pts.iter().filter(|p| far(p)) {
                        assert!(
                            local.rings.iter().any(|r| r.pts.contains(p)),
                            "case {n} {op:?}: {p:?} lost"
                        );
                    }
                }
            }
        }
        // A ring on its own, painted next to: no splice needed, nothing
        // inside the window to keep.
        let lone = Region::from_ring(square(10.0, 10.0, 5.0));
        let other = Region::from_ring(square(100.0, 100.0, 5.0));
        let got = lone.boolean_local(&other, BoolOp::Union);
        // The new square is the raster's trace of it, as from the whole
        // raster; the old one is kept as it is.
        let traced = other.boolean(&Region::new(), BoolOp::Union).area();
        assert!(
            (got.area() - 25.0 - traced).abs() < 1e-6,
            "{} vs {}",
            got.area(),
            25.0 + traced
        );
        assert!(got.rings.iter().any(|r| r.pts == lone.rings[0].pts));
    }

    /// The brush on a structure, sample by sample: a slice merged after
    /// every sample from what the last merge left, and one merged once from
    /// the slice as the stroke found it, come out the same; merging again
    /// changes nothing.
    #[test]
    fn a_footprint_merged_sample_by_sample_is_the_one_merged_at_once() {
        let sp = [0.9, 0.9, 3.0];
        let g = grid([400, 400, 20], sp);
        let base = big_circle([200.0, 200.0], 150.0, 1200);
        for erase in [false, true] {
            let mut f = BallFootprint::new(&g, 2, 12.0);
            let mut cur = base.clone();
            let mut prev = [348.0, 200.0, 10.0];
            let mut path = vec![prev];
            for s in 0..40 {
                let a = s as f64 * 0.05;
                let p = [200.0 + 148.0 * a.cos(), 200.0 + 148.0 * a.sin(), 10.0];
                f.sweep(prev, p, 12.0, |_| true);
                prev = p;
                path.push(p);
                if let Some(b) = f.take_dirty_box_at(10) {
                    cur = f.merge(10, b, &base, &cur, erase);
                }
            }
            // Merged at once re-traces the whole slice, which moves its area
            // a little by itself: the change is compared.
            let once = f.merged(10, &base, erase);
            let retraced = base.boolean(&Region::new(), BoolOp::Union).area();
            let (d_cur, d_once) = (cur.area() - base.area(), once.area() - retraced);
            assert!(d_cur.abs() > 500.0, "erase {erase}: {d_cur}");
            assert!(
                (d_cur - d_once).abs() < 0.01 * d_once.abs() + 2.0,
                "erase {erase}: {d_cur} vs {d_once}"
            );
            // Where the ball went, both are the trace of the same cells.
            let reach = 12.0 / 0.9 + 0.5;
            let near = |x: f64, y: f64| {
                path.windows(2).any(|s| {
                    let (a, b) = ([s[0][0], s[0][1]], [s[1][0], s[1][1]]);
                    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                    let l2 = (dx * dx + dy * dy).max(1e-12);
                    let t = (((x - a[0]) * dx + (y - a[1]) * dy) / l2).clamp(0.0, 1.0);
                    (x - a[0] - t * dx).hypot(y - a[1] - t * dy) < reach
                })
            };
            let (rc, ro) = (cur.rasterize(400, 400), once.rasterize(400, 400));
            let differ = (0..400 * 400)
                .filter(|&k| rc[k] != ro[k] && near((k % 400) as f64, (k / 400) as f64))
                .count();
            assert!(differ <= 8, "erase {erase}: {differ} voxels differ");
            let a = cur.area();
            // The whole stroke again at once: the same outline.
            let s = &f.slices[&10];
            let all = [s.lo[0], s.lo[1], s.hi[0], s.hi[1]];
            let again = f.merge(10, all, &base, &cur, erase);
            assert!((again.area() - a).abs() < 2.0, "{} vs {a}", again.area());
            // The far side of the circle was never reached: its vertices
            // are the base's own.
            let p = base.rings[0].pts[600];
            assert!(cur.rings.iter().any(|r| r.pts.contains(&p)));
        }
    }

    /// A sagittal stack re-cut onto a slab of axial slices gives those
    /// slices exactly as re-cutting the whole does; written into an ROI, the
    /// slab is re-cut and every other axial contour stays the ROI's own.
    #[test]
    fn a_slab_of_axial_slices_is_re_cut_as_the_whole_is() {
        let dims = [40, 40, 20];
        let g = grid(dims, [1.5, 1.5, 3.0]);
        let axial = Stack::from_mask(&ball(dims, 12.0), dims, 2);
        let mut sag = axial.to_axis(dims, 0);
        // An edit on the sagittal stack, so the re-cut is not the original.
        sag.region_mut(20).subtract_ring(&square(5.0, 8.0, 30.0));
        let whole = sag.to_axis(dims, 2);
        let slab = sag.to_axis_levels(dims, 2, [8, 11]);
        let inside = |l: usize| (8..=11).contains(&l);
        assert!(!slab.slices.is_empty());
        assert!(slab.slices.iter().all(|s| inside(s.level)));
        assert_eq!(
            slab.slices.len(),
            whole.slices.iter().filter(|s| inside(s.level)).count()
        );
        for s in &slab.slices {
            assert_eq!(
                Some(&s.region),
                whole.region_at(s.level),
                "level {}",
                s.level
            );
        }
        let near = |a: &Region, b: &Region| {
            a.rings.len() == b.rings.len()
                && a.rings.iter().zip(&b.rings).all(|(x, y)| {
                    x.pts.len() == y.pts.len()
                        && x.pts
                            .iter()
                            .zip(&y.pts)
                            .all(|(p, q)| (p[0] - q[0]).abs() < 1e-6 && (p[1] - q[1]).abs() < 1e-6)
                })
        };
        let mut roi = Roi {
            number: 1,
            name: "ball".into(),
            color: [255, 0, 0],
            roi_type: "ORGAN".into(),
            description: String::new(),
            contours: Vec::new(),
        };
        axial.apply_to_roi(&mut roi, &g);
        sag.apply_levels_to_roi(&mut roi, &g, [8, 11]);
        let after = Stack::from_roi(&roi, &g);
        assert_eq!(after.axis, 2);
        assert_eq!(after.occupied(), axial.occupied());
        for s in &after.slices {
            let want = if inside(s.level) {
                whole.region_at(s.level)
            } else {
                axial.region_at(s.level)
            };
            assert!(
                want.is_some_and(|w| near(w, &s.region)),
                "level {}",
                s.level
            );
        }
    }
}
