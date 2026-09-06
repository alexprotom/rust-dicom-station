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
        *self = self.boolean(&Region::from_ring(ring.clone()), BoolOp::Union);
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
        *self = self.boolean(&Region::from_ring(ring.clone()), BoolOp::Subtract);
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
    let mut crossings: Vec<f64> = Vec::new();
    for j in 0..h {
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
        let row = j * w;
        for span in crossings.as_chunks::<2>().0 {
            let from = ((span[0] - x0) / step - 0.5).ceil().max(0.0) as i64;
            let to = (((span[1] - x0) / step - 0.5).floor() as i64).min(w as i64 - 1);
            for i in from..=to {
                if i >= 0 {
                    out[row + i as usize] = true;
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
// Marching-squares plumbing, shared with segmentation.rs
// ---------------------------------------------------------------------------

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
}

impl Stack {
    pub fn empty(axis: usize) -> Stack {
        Stack {
            axis,
            slices: Vec::new(),
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
            vox.push(c.points.iter().map(|p| grid.patient_to_voxel(*p)).collect());
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
            if lvl < 0.0 || lvl >= grid.dims[axis] as f64 {
                continue;
            }
            let ring = Poly::new(pts.iter().map(|p| [p[ua], p[va]]).collect());
            if ring.is_empty() {
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
        for s in &self.slices {
            for ring in &s.region.rings {
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
                        v[self.axis] = s.level as f64;
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
        let [nx, ny, _nz] = dims;
        let [ua, va] = plane_axes(axis);
        let (nu, nv) = (dims[ua], dims[va]);
        let (pw, ph) = (nu + 2, nv + 2);
        let mut st = Stack::empty(axis);
        let mut field = vec![0.0f32; pw * ph];
        for level in 0..dims[axis] {
            field.iter_mut().for_each(|v| *v = 0.0);
            let mut any = false;
            for v in 0..nv {
                for u in 0..nu {
                    let mut idx = [0usize; 3];
                    idx[ua] = u;
                    idx[va] = v;
                    idx[axis] = level;
                    if mask[idx[2] * nx * ny + idx[1] * nx + idx[0]] != 0 {
                        field[(v + 1) * pw + u + 1] = 1.0;
                        any = true;
                    }
                }
            }
            if !any {
                continue;
            }
            let mut region = Region::new();
            for pts in stitch_loops(&render::marching_squares(&field, pw, ph, 0.5)) {
                let pts = drop_collinear(pts);
                if pts.len() < 3 {
                    continue;
                }
                region.rings.push(Poly::new(
                    pts.iter()
                        .map(|p| [p[0] as f64 - 1.0, p[1] as f64 - 1.0])
                        .collect(),
                ));
            }
            region.normalize();
            if !region.is_empty() {
                st.slices.push(SlicePolys { level, region });
            }
        }
        st
    }

    /// Re-slice along another lattice axis. This is the one conversion that
    /// necessarily goes through voxels - the contours of a sagittal stack say
    /// nothing about where the axial ones run - so it is explicit, and the
    /// interface says so when it happens.
    pub fn to_axis(&self, dims: [usize; 3], axis: usize) -> Stack {
        if axis == self.axis {
            return self.clone();
        }
        Stack::from_mask(&self.rasterize(dims), dims, axis)
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
        for s in &mut self.slices {
            for r in &mut s.region.rings {
                r.translate(d);
            }
        }
    }

    /// Scale every slice about the stack's own in-plane centroid.
    pub fn scale(&mut self, factor: f64) {
        let c = self.centroid();
        for s in &mut self.slices {
            for r in &mut s.region.rings {
                r.scale_about(c, factor);
            }
        }
    }

    pub fn rotate(&mut self, angle: f64) {
        let c = self.centroid();
        for s in &mut self.slices {
            for r in &mut s.region.rings {
                r.rotate_about(c, angle);
            }
        }
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
            for k in (k0 + 1)..k1 {
                let t = (k - k0) as f32 / (k1 - k0) as f32;
                let (pw, ph) = (w + 2, h + 2);
                let mut field = vec![0.0f32; pw * ph];
                let mut any = false;
                for j in 0..h {
                    for i in 0..w {
                        let d = (1.0 - t) * d0[j * w + i] + t * d1[j * w + i];
                        if d <= 0.0 {
                            field[(j + 1) * pw + i + 1] = 1.0;
                            any = true;
                        }
                    }
                }
                if !any {
                    continue;
                }
                let mut region = Region::new();
                for pts in stitch_loops(&render::marching_squares(&field, pw, ph, 0.5)) {
                    let pts = drop_collinear(pts);
                    if pts.len() < 3 {
                        continue;
                    }
                    region.rings.push(Poly::new(
                        pts.iter()
                            .map(|p| [x0 + p[0] as f64 - 1.0, y0 + p[1] as f64 - 1.0])
                            .collect(),
                    ));
                }
                region.normalize();
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
