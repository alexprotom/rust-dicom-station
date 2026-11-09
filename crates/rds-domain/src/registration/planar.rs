//! 2D/2D registration of planar images: one radiograph, RT image or DRR
//! onto another.
//!
//! **Frames.** Every transform here is written in millimetres of an image's
//! own raster, origin at the image centre: pixel `[i, j]` (centre of the
//! first pixel at `(0, 0)`) is `x = (i + 0.5 − cols / 2) · sx`, `y = (j +
//! 0.5 − rows / 2) · sy` ([`px_to_mm`]). A shift then reads in millimetres at
//! the detector; divided by the magnification (`SID / SAD`, when the image
//! has a camera) it reads at the isocentre.
//!
//! **Direction.** As every registration of the program, the transform maps
//! the **fixed** image's frame onto the **moving** image's: the moving image
//! resampled onto the fixed raster is `moving(T(x))`. The fixed image is the
//! one the result is shown on (the radiograph of the day), the moving one
//! the reference (an approved image of day one, a DRR). A contour drawn on
//! the moving image is carried onto the fixed one through `T⁻¹`
//! ([`Transform2d::inverse_map`]: exact for the rigid part, by fixed-point
//! iteration for a B-spline, as `propagate` inverts its 3D fields).
//!
//! **Models.** [`Rigid2d`] is a similarity about a centre (shift, rotation,
//! optionally an isotropic scale); [`BSpline2d`] is a free-form deformation
//! on a uniform grid of control points (Rueckert 1999), two coefficients per
//! node, cubic B-spline weights with a 4 × 4 support - the 2D reduction of
//! [`crate::registration::BSplineTransform`]. A [`Transform2d`] is the rigid
//! part followed by the warp: `T(x) = R(x) + d(x)` with `d` evaluated in the
//! fixed frame.

use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::geometry::Vec3;
use crate::progress::{ProgressSink, CANCELLED};
use crate::projection::ProjectionGeometry;
use crate::registration::metric2d::{border_mask, halve_mask, ncc, Image2d, Mask2d, Metric2d};
use crate::registration::optim::{lbfgs, nelder_mead, powell, Bounds, Limits};
pub use crate::registration::projective::Optimizer2d;

/// The millimetre frame of a raster of `dims = [cols, rows]` and `spacing =
/// [sx, sy]`: pixel coordinates to millimetres about the image centre.
#[inline]
pub fn px_to_mm(dims: [usize; 2], spacing: [f64; 2], p: [f64; 2]) -> [f64; 2] {
    [
        (p[0] + 0.5 - dims[0] as f64 * 0.5) * spacing[0],
        (p[1] + 0.5 - dims[1] as f64 * 0.5) * spacing[1],
    ]
}

/// The inverse of [`px_to_mm`].
#[inline]
pub fn mm_to_px(dims: [usize; 2], spacing: [f64; 2], q: [f64; 2]) -> [f64; 2] {
    [
        q[0] / spacing[0].max(1e-12) - 0.5 + dims[0] as f64 * 0.5,
        q[1] / spacing[1].max(1e-12) - 0.5 + dims[1] as f64 * 0.5,
    ]
}

/// A similarity in the millimetre frame: `T(p) = c + s · R(θ) · (p − c) +
/// t`, fixed → moving.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rigid2d {
    pub tx: f64,
    pub ty: f64,
    /// Radians, counter-clockwise in `(x, y)` (on screen, with `y` down,
    /// clockwise).
    pub theta: f64,
    pub scale: f64,
    pub center: [f64; 2],
}

impl Default for Rigid2d {
    fn default() -> Self {
        Rigid2d::identity([0.0, 0.0])
    }
}

impl Rigid2d {
    pub fn identity(center: [f64; 2]) -> Rigid2d {
        Rigid2d {
            tx: 0.0,
            ty: 0.0,
            theta: 0.0,
            scale: 1.0,
            center,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.tx == 0.0 && self.ty == 0.0 && self.theta == 0.0 && self.scale == 1.0
    }

    #[inline]
    pub fn map(&self, p: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.theta.sin_cos();
        let (dx, dy) = (p[0] - self.center[0], p[1] - self.center[1]);
        [
            self.center[0] + self.scale * (c * dx - s * dy) + self.tx,
            self.center[1] + self.scale * (s * dx + c * dy) + self.ty,
        ]
    }

    /// `T⁻¹(q)`, exact.
    #[inline]
    pub fn inverse_map(&self, q: [f64; 2]) -> [f64; 2] {
        let (s, c) = self.theta.sin_cos();
        let k = 1.0 / self.scale.max(1e-12);
        let (dx, dy) = (
            q[0] - self.tx - self.center[0],
            q[1] - self.ty - self.center[1],
        );
        [
            self.center[0] + k * (c * dx + s * dy),
            self.center[1] + k * (-s * dx + c * dy),
        ]
    }

    /// The inverse transform as a similarity of its own.
    pub fn inverse(&self) -> Rigid2d {
        // T⁻¹(q) = c + R(−θ)/s · (q − c) − R(−θ)/s · t.
        let mut inv = Rigid2d {
            tx: 0.0,
            ty: 0.0,
            theta: -self.theta,
            scale: 1.0 / self.scale.max(1e-12),
            center: self.center,
        };
        let o = inv.map(self.center);
        let want = self.inverse_map(self.center);
        inv.tx = want[0] - o[0];
        inv.ty = want[1] - o[1];
        inv
    }

    /// The same transform written about another centre.
    pub fn recentered(&self, center: [f64; 2]) -> Rigid2d {
        let mut r = Rigid2d {
            center,
            tx: 0.0,
            ty: 0.0,
            ..*self
        };
        let want = self.map(center);
        r.tx = want[0] - center[0];
        r.ty = want[1] - center[1];
        r
    }

    /// The parameter vector the optimisers move: `[tx, ty, θ·R, ln(s)·R]`,
    /// `R` the radius that makes a radian and a unit of log-scale read like
    /// millimetres.
    pub fn to_params(&self, radius: f64) -> [f64; 4] {
        [
            self.tx,
            self.ty,
            self.theta * radius,
            self.scale.max(1e-12).ln() * radius,
        ]
    }

    pub fn from_params(p: &[f64], radius: f64, center: [f64; 2]) -> Rigid2d {
        let r = radius.max(1e-9);
        Rigid2d {
            tx: p.first().copied().unwrap_or(0.0),
            ty: p.get(1).copied().unwrap_or(0.0),
            theta: p.get(2).copied().unwrap_or(0.0) / r,
            scale: (p.get(3).copied().unwrap_or(0.0) / r).exp(),
            center,
        }
    }
}

/// A cubic B-spline free-form deformation over the fixed image's
/// millimetre frame: control point `(a, b)` sits at `origin + (a, b) ·
/// spacing`, its two coefficients at `coeffs[2 (b · nx + a)]` (x) and `+ 1`
/// (y). The grid reaches one node beyond the image on every side, so every
/// point of the image has its full 4 × 4 support.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BSpline2d {
    pub coeffs: Vec<f64>,
    pub grid_dims: [usize; 2],
    pub origin: [f64; 2],
    pub spacing: f64,
}

/// The four cubic B-spline weights at fraction `t` (the node `⌊u⌋ − 1` gets
/// the first), as [`crate::registration::BSplineTransform`] has them.
#[inline]
pub fn bspline_weights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    [
        mt * mt * mt / 6.0,
        (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0,
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0,
        t3 / 6.0,
    ]
}

/// Their derivatives with respect to `t`.
#[inline]
pub fn bspline_dweights(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let mt = 1.0 - t;
    [
        -mt * mt / 2.0,
        (9.0 * t2 - 12.0 * t) / 6.0,
        (-9.0 * t2 + 6.0 * t + 3.0) / 6.0,
        t2 / 2.0,
    ]
}

impl BSpline2d {
    /// A zero warp covering `[lo, hi]` (mm) with nodes `spacing` apart.
    pub fn zero(lo: [f64; 2], hi: [f64; 2], spacing: f64) -> BSpline2d {
        let sp = spacing.max(1e-3);
        let n = |a: usize| ((hi[a] - lo[a]) / sp).ceil().max(0.0) as usize + 4;
        let grid_dims = [n(0), n(1)];
        BSpline2d {
            coeffs: vec![0.0; 2 * grid_dims[0] * grid_dims[1]],
            grid_dims,
            origin: [lo[0] - sp, lo[1] - sp],
            spacing: sp,
        }
    }

    /// Continuous grid coordinates of a point.
    #[inline]
    pub fn grid_coords(&self, p: [f64; 2]) -> [f64; 2] {
        [
            (p[0] - self.origin[0]) / self.spacing,
            (p[1] - self.origin[1]) / self.spacing,
        ]
    }

    /// The support of `p`: the first node `(a0, b0)` and the weights along
    /// each axis, or `None` outside the grid.
    #[inline]
    pub fn support(&self, p: [f64; 2]) -> Option<([usize; 2], [f64; 4], [f64; 4])> {
        let u = self.grid_coords(p);
        let (fa, fb) = (u[0].floor(), u[1].floor());
        if fa < 1.0 || fb < 1.0 {
            return None;
        }
        let (a0, b0) = (fa as usize - 1, fb as usize - 1);
        if a0 + 3 >= self.grid_dims[0] || b0 + 3 >= self.grid_dims[1] {
            return None;
        }
        Some((
            [a0, b0],
            bspline_weights(u[0] - fa),
            bspline_weights(u[1] - fb),
        ))
    }

    /// The displacement at `p` (mm), zero outside the grid.
    #[inline]
    pub fn displacement(&self, p: [f64; 2]) -> [f64; 2] {
        let Some(([a0, b0], wx, wy)) = self.support(p) else {
            return [0.0, 0.0];
        };
        let nx = self.grid_dims[0];
        let (mut dx, mut dy) = (0.0, 0.0);
        for (m, wym) in wy.iter().enumerate() {
            let row = (b0 + m) * nx + a0;
            for (l, wxl) in wx.iter().enumerate() {
                let w = wxl * wym;
                let o = 2 * (row + l);
                dx += w * self.coeffs[o];
                dy += w * self.coeffs[o + 1];
            }
        }
        [dx, dy]
    }

    /// The Jacobian of the displacement `[[∂dx/∂x, ∂dx/∂y], [∂dy/∂x,
    /// ∂dy/∂y]]` at `p`.
    pub fn displacement_jacobian(&self, p: [f64; 2]) -> [[f64; 2]; 2] {
        let u = self.grid_coords(p);
        let (fa, fb) = (u[0].floor(), u[1].floor());
        if fa < 1.0 || fb < 1.0 {
            return [[0.0; 2]; 2];
        }
        let (a0, b0) = (fa as usize - 1, fb as usize - 1);
        if a0 + 3 >= self.grid_dims[0] || b0 + 3 >= self.grid_dims[1] {
            return [[0.0; 2]; 2];
        }
        let (wx, wy) = (bspline_weights(u[0] - fa), bspline_weights(u[1] - fb));
        let (dwx, dwy) = (bspline_dweights(u[0] - fa), bspline_dweights(u[1] - fb));
        let nx = self.grid_dims[0];
        let mut j = [[0.0; 2]; 2];
        let inv = 1.0 / self.spacing;
        for m in 0..4 {
            for l in 0..4 {
                let o = 2 * ((b0 + m) * nx + a0 + l);
                let gx = dwx[l] * wy[m] * inv;
                let gy = wx[l] * dwy[m] * inv;
                j[0][0] += gx * self.coeffs[o];
                j[0][1] += gy * self.coeffs[o];
                j[1][0] += gx * self.coeffs[o + 1];
                j[1][1] += gy * self.coeffs[o + 1];
            }
        }
        j
    }
}

/// A 2D/2D registration's answer: the similarity, then the warp.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Transform2d {
    pub rigid: Rigid2d,
    pub warp: Option<BSpline2d>,
}

impl Transform2d {
    pub fn rigid(rigid: Rigid2d) -> Transform2d {
        Transform2d { rigid, warp: None }
    }

    /// Fixed frame → moving frame, mm.
    #[inline]
    pub fn map(&self, p: [f64; 2]) -> [f64; 2] {
        let q = self.rigid.map(p);
        match &self.warp {
            Some(w) => {
                let d = w.displacement(p);
                [q[0] + d[0], q[1] + d[1]]
            }
            None => q,
        }
    }

    /// Moving frame → fixed frame: exact without a warp, else the fixed
    /// point `p = R⁻¹(q − d(p))` iterated from `R⁻¹(q)` (20 steps or until
    /// it moves less than `tol` mm). `None` when it does not settle - inside
    /// a fold, where the warp has no inverse.
    pub fn inverse_map(&self, q: [f64; 2], tol: f64) -> Option<[f64; 2]> {
        let Some(w) = &self.warp else {
            return Some(self.rigid.inverse_map(q));
        };
        let mut p = self.rigid.inverse_map(q);
        for _ in 0..20 {
            let d = w.displacement(p);
            let next = self.rigid.inverse_map([q[0] - d[0], q[1] - d[1]]);
            let moved = ((next[0] - p[0]).powi(2) + (next[1] - p[1]).powi(2)).sqrt();
            p = next;
            if moved < tol {
                return Some(p);
            }
        }
        let back = self.map(p);
        let err = ((back[0] - q[0]).powi(2) + (back[1] - q[1]).powi(2)).sqrt();
        (err < tol * 10.0).then_some(p)
    }
}

// ---------------------------------------------------------------------------
// The images and the request
// ---------------------------------------------------------------------------

/// One image of a 2D/2D registration: its pixels, their spacing, and what
/// places it in the room when it is known.
#[derive(Clone, Debug)]
pub struct Planar2d {
    pub image: Image2d,
    pub spacing: [f64; 2],
    /// The image's camera, when it has one.
    pub camera: Option<ProjectionGeometry>,
    /// The isocentre its geometry names, when it names one: the rotation
    /// centre (its projection) and the magnification a shift at the
    /// detector is read at.
    pub isocenter: Option<Vec3>,
    /// A transmission image (bright where little attenuates).
    pub transmission: bool,
}

impl Planar2d {
    /// The registration's view of a planar image.
    pub fn of(img: &crate::extras::PlanarImage) -> Planar2d {
        Planar2d {
            image: Image2d::new(img.data.clone(), [img.cols, img.rows]),
            spacing: img.spacing,
            camera: img
                .camera
                .map(|c| crate::planar_structs::rescaled(&c, [img.cols, img.rows])),
            isocenter: img.iec.map(|g| g.isocenter),
            transmission: false,
        }
    }

    pub fn dims(&self) -> [usize; 2] {
        self.image.dims
    }

    /// Where the isocentre projects, in the millimetre frame, and the
    /// magnification there (detector distance over isocentre distance).
    pub fn iso_frame(&self) -> Option<([f64; 2], f64)> {
        let (cam, iso) = (self.camera?, self.isocenter?);
        let p = cam.project(iso)?;
        let on_detector = cam.pixel(p[0], p[1]);
        let mag = (on_detector - cam.source).length() / (iso - cam.source).length().max(1e-9);
        Some((px_to_mm(self.dims(), self.spacing, p), mag))
    }
}

/// What a 2D/2D run does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Method2d {
    /// The images' intensities, coarse to fine.
    #[default]
    Intensity,
    /// Pairs of corresponding points (placed, or markers found).
    Points,
    /// The points' fit, then the intensities from it.
    PointsThenIntensity,
    /// The intensities with a B-spline warp after the rigid start.
    Deformable,
}

impl Method2d {
    pub const ALL: [Method2d; 4] = [
        Method2d::Intensity,
        Method2d::Points,
        Method2d::PointsThenIntensity,
        Method2d::Deformable,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Method2d::Intensity => "Intensity",
            Method2d::Points => "Points",
            Method2d::PointsThenIntensity => "Points, then intensity",
            Method2d::Deformable => "Deformable",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Method2d::Intensity => "intensity",
            Method2d::Points => "points",
            Method2d::PointsThenIntensity => "points_then_intensity",
            Method2d::Deformable => "deformable",
        }
    }

    pub fn from_key(s: &str) -> Option<Method2d> {
        let s = s.trim().to_ascii_lowercase();
        Method2d::ALL.into_iter().find(|m| m.key() == s)
    }
}

/// The degrees of freedom of the rigid part.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dof2d {
    Shift,
    #[default]
    ShiftRotation,
    /// Shift, rotation and an isotropic scale (two images at another SID,
    /// an imager whose pixel spacing is not trusted).
    Similarity,
}

impl Dof2d {
    pub const ALL: [Dof2d; 3] = [Dof2d::Shift, Dof2d::ShiftRotation, Dof2d::Similarity];

    pub fn count(self) -> usize {
        match self {
            Dof2d::Shift => 2,
            Dof2d::ShiftRotation => 3,
            Dof2d::Similarity => 4,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Dof2d::Shift => "Shift",
            Dof2d::ShiftRotation => "Shift and rotation",
            Dof2d::Similarity => "Similarity",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Dof2d::Shift => "shift",
            Dof2d::ShiftRotation => "shift_rotation",
            Dof2d::Similarity => "similarity",
        }
    }

    pub fn from_key(s: &str) -> Option<Dof2d> {
        let s = s.trim().to_ascii_lowercase();
        Dof2d::ALL.into_iter().find(|m| m.key() == s)
    }
}

/// The measure of the deformable method.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeformMetric {
    /// Mean squared difference: two images of one kind.
    #[default]
    Msd,
    /// Mattes mutual information: a radiograph against a DRR.
    Mi,
}

impl DeformMetric {
    pub const ALL: [DeformMetric; 2] = [DeformMetric::Msd, DeformMetric::Mi];

    pub fn label(self) -> &'static str {
        match self {
            DeformMetric::Msd => "Mean squared difference",
            DeformMetric::Mi => "Mutual information",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            DeformMetric::Msd => "msd",
            DeformMetric::Mi => "mi",
        }
    }

    pub fn from_key(s: &str) -> Option<DeformMetric> {
        let s = s.trim().to_ascii_lowercase();
        DeformMetric::ALL.into_iter().find(|m| m.key() == s)
    }

    /// The bending energy's weight a run of this measure starts from. The
    /// mean squared difference of normalised intensities moves about thirty
    /// times as much per millimetre of misfit as the mutual information
    /// does, so the same smoothing takes a weight thirty times larger
    /// against the latter.
    pub fn default_lambda(self) -> f64 {
        match self {
            DeformMetric::Msd => 0.01,
            DeformMetric::Mi => 0.3,
        }
    }
}

/// Everything a 2D/2D run is asked.
#[derive(Clone, Debug)]
pub struct Request2d {
    pub fixed: Planar2d,
    pub moving: Planar2d,
    pub method: Method2d,
    pub metric: Metric2d,
    pub optimizer: Optimizer2d,
    pub dof: Dof2d,
    /// Pyramid levels, and the largest side of the finest one.
    pub levels: usize,
    pub max_pixels: usize,
    /// The search range: shift (mm, at the detector) and rotation (deg).
    pub range_mm: f64,
    pub range_deg: f64,
    /// L-BFGS with finite differences after the search, at the finest
    /// level.
    pub refine: bool,
    /// The pixels of the fixed image that count (its region), full raster.
    pub mask: Option<Mask2d>,
    /// Corresponding points, `(fixed px, moving px)`.
    pub pairs: Vec<([f64; 2], [f64; 2])>,
    /// Find markers of this size (mm) on both images and pair them.
    pub detect_markers: Option<f64>,
    /// The B-spline's node spacing at the first level and at the last,
    /// mm, and the weight of its bending energy.
    pub grid_mm: f64,
    pub grid_min_mm: f64,
    pub lambda: f64,
    pub deformable_metric: DeformMetric,
    /// Start from this transform rather than the identity.
    pub start: Option<Rigid2d>,
    pub max_evaluations: usize,
}

impl Request2d {
    pub fn new(fixed: Planar2d, moving: Planar2d) -> Request2d {
        Request2d {
            fixed,
            moving,
            method: Method2d::Intensity,
            metric: Metric2d::Ncc,
            optimizer: Optimizer2d::Powell,
            dof: Dof2d::ShiftRotation,
            levels: 3,
            max_pixels: 256,
            range_mm: 30.0,
            range_deg: 10.0,
            refine: false,
            mask: None,
            pairs: Vec::new(),
            detect_markers: None,
            grid_mm: 32.0,
            grid_min_mm: 8.0,
            lambda: 0.01,
            deformable_metric: DeformMetric::Msd,
            start: None,
            max_evaluations: 3000,
        }
    }
}

/// One level of a run: its raster and the measure before and after.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelLine {
    pub level: usize,
    pub dims: [usize; 2],
    pub before: f64,
    pub after: f64,
    pub evaluations: usize,
}

/// One pair of points after the fit.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairResidual {
    pub fixed: [f64; 2],
    pub moving: [f64; 2],
    /// `|T(fixed) − moving|`, mm in the moving image's frame.
    pub residual_mm: f64,
    /// Kept in the fit (a pair beyond three times the median is left out).
    pub used: bool,
}

/// What the warp does, over the fixed image.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WarpStats2d {
    pub mean_mm: f64,
    pub max_mm: f64,
    pub jacobian_min: f64,
    pub jacobian_max: f64,
    /// Pixels where the warp folds (a Jacobian determinant at or below
    /// zero): there the carry of a contour is refused.
    pub folds: usize,
}

/// A dense displacement per pixel of the fixed image (mm in its frame):
/// `T(x) − x`, for display and for carrying contours.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Field2d {
    pub dims: [usize; 2],
    /// Full pixels of the fixed image per field pixel (a side).
    pub factor: usize,
    pub dx: Vec<f32>,
    pub dy: Vec<f32>,
}

/// A finished 2D/2D run.
#[derive(Clone, Debug)]
pub struct Result2d {
    pub transform: Transform2d,
    pub method: Method2d,
    pub metric: Metric2d,
    /// The measure at the start and at the answer, at the finest level.
    pub before: f64,
    pub after: f64,
    pub levels: Vec<LevelLine>,
    pub pairs: Vec<PairResidual>,
    /// Indices into `pairs` of the pairs left out.
    pub left_out: Vec<usize>,
    pub field: Option<Field2d>,
    pub warp_stats: Option<WarpStats2d>,
    /// The shift at the isocentre (mm), when the fixed image is placed.
    pub shift_iso_mm: Option<[f64; 2]>,
    /// The moving image was inverted to match the fixed one's polarity.
    pub inverted: bool,
    pub evaluations: usize,
    pub elapsed_secs: f64,
    pub warning: Option<String>,
}

impl Result2d {
    /// One line: the shift, the rotation and the scale.
    pub fn line(&self) -> String {
        let r = &self.transform.rigid;
        let mut s = format!(
            "shift ({:+.2}, {:+.2}) mm, rotation {:+.2} deg",
            r.tx,
            r.ty,
            r.theta.to_degrees()
        );
        if (r.scale - 1.0).abs() > 1e-9 {
            s.push_str(&format!(", scale {:.4}", r.scale));
        }
        if let Some(i) = self.shift_iso_mm {
            s.push_str(&format!(
                " · at the isocentre ({:+.2}, {:+.2}) mm",
                i[0], i[1]
            ));
        }
        if self.transform.warp.is_some() {
            s.push_str(" · deformable");
        }
        s
    }

    pub fn json(&self) -> serde_json::Value {
        let r = &self.transform.rigid;
        serde_json::json!({
            "method": self.method.key(),
            "metric": self.metric.key(),
            "shift_mm": [r.tx, r.ty],
            "rotation_deg": r.theta.to_degrees(),
            "scale": r.scale,
            "center_mm": r.center,
            "shift_isocenter_mm": self.shift_iso_mm,
            "before": self.before,
            "after": self.after,
            "levels": self.levels,
            "pairs": self.pairs,
            "left_out": self.left_out,
            "warp": self.warp_stats,
            "inverted": self.inverted,
            "evaluations": self.evaluations,
            "seconds": self.elapsed_secs,
            "warning": self.warning,
        })
    }
}

/// A 2D/2D registration as the study keeps it: its images by SOP Instance
/// UID (never by index), the answer, and the moved copy filed with it.
#[derive(Clone, Debug)]
pub struct PlanarReg {
    pub label: String,
    pub fixed_uid: String,
    pub moving_uid: String,
    pub result: Result2d,
    /// The moving image resampled onto the fixed one, when it was filed;
    /// removing the copy removes the registration with it.
    pub moved_copy_uid: Option<String>,
    /// `YYYY-MM-DD HH:MM` (UTC).
    pub when: String,
}

pub use crate::registration::projective::{rotation_from_rolls, translation_from_shifts};

// ---------------------------------------------------------------------------
// Sampling between the two frames
// ---------------------------------------------------------------------------

/// One level of an image: its pixels and how they sit on the full raster
/// (`factor` full pixels a side, coarse pixel `k` centred on full pixel
/// `k f + (f − 1) / 2`).
#[derive(Clone, Debug)]
struct LevelImage {
    image: Image2d,
    factor: f64,
    /// The full raster and its spacing: the millimetre frame.
    full: [usize; 2],
    spacing: [f64; 2],
}

impl LevelImage {
    #[inline]
    fn to_mm(&self, p: [f64; 2]) -> [f64; 2] {
        let f = self.factor;
        px_to_mm(
            self.full,
            self.spacing,
            [p[0] * f + (f - 1.0) * 0.5, p[1] * f + (f - 1.0) * 0.5],
        )
    }

    #[inline]
    fn to_px(&self, q: [f64; 2]) -> [f64; 2] {
        let f = self.factor;
        let p = mm_to_px(self.full, self.spacing, q);
        [(p[0] - (f - 1.0) * 0.5) / f, (p[1] - (f - 1.0) * 0.5) / f]
    }

    /// The level's own pixel size, mm.
    fn pixel_mm(&self) -> f64 {
        self.spacing[0].min(self.spacing[1]) * self.factor
    }
}

/// `levels` levels of `img` (finest first), the finest halved until no side
/// exceeds `max_pixels`.
fn levels_of(img: &Planar2d, levels: usize, max_pixels: usize) -> Vec<LevelImage> {
    let full = img.dims();
    let mut cur = img.image.clone();
    let mut f = 1.0;
    while cur.dims[0].max(cur.dims[1]) > max_pixels.max(16) {
        cur = cur.halved();
        f *= 2.0;
    }
    let mut out = Vec::new();
    for _ in 0..levels.max(1) {
        out.push(LevelImage {
            image: cur.clone(),
            factor: f,
            full,
            spacing: img.spacing,
        });
        if cur.dims[0].min(cur.dims[1]) < 24 {
            break;
        }
        cur = cur.halved();
        f *= 2.0;
    }
    out
}

/// The fixed mask at a level: `mask` (full raster) halved as the image was.
fn mask_at(mask: Option<&[u8]>, full: [usize; 2], level: &LevelImage) -> Option<Mask2d> {
    let m = mask?;
    if m.len() != full[0] * full[1] {
        return None;
    }
    let mut cur = m.to_vec();
    let mut dims = full;
    let mut f = 1.0;
    while f < level.factor {
        let (h, d) = halve_mask(&cur, dims);
        cur = h;
        dims = d;
        f *= 2.0;
    }
    (dims == level.image.dims).then_some(cur)
}

/// The moving level resampled onto the fixed level through `t`: the
/// pixels, and which of them fell inside the moving image (outside, the
/// nearest edge pixel stands in, so a gradient there is no step).
fn resample_level(moving: &LevelImage, t: &Transform2d, fixed: &LevelImage) -> (Vec<f32>, Vec<u8>) {
    let [w, h] = fixed.image.dims;
    let mut px = vec![0.0f32; w * h];
    let mut ok = vec![0u8; w * h];
    px.par_chunks_mut(w)
        .zip(ok.par_chunks_mut(w))
        .enumerate()
        .for_each(|(j, (row, okrow))| {
            for i in 0..w {
                let q = moving.to_px(t.map(fixed.to_mm([i as f64, j as f64])));
                let inside = moving.image.sample(q[0], q[1]);
                row[i] = match inside {
                    Some(v) => {
                        okrow[i] = 1;
                        v
                    }
                    None => moving.image.sample_clamped(q[0], q[1]),
                };
            }
        });
    (px, ok)
}

/// `moving` resampled onto `onto`'s raster through `t` (bilinear; outside
/// the moving image, its smallest value): the moved copy a pane stacks over
/// the fixed image.
pub fn resample(moving: &Planar2d, t: &Transform2d, onto: &Planar2d) -> Image2d {
    let m = LevelImage {
        image: moving.image.clone(),
        factor: 1.0,
        full: moving.dims(),
        spacing: moving.spacing,
    };
    let f = LevelImage {
        image: Image2d::new(vec![0.0; onto.dims()[0] * onto.dims()[1]], onto.dims()),
        factor: 1.0,
        full: onto.dims(),
        spacing: onto.spacing,
    };
    let lo = moving.image.data.iter().copied().fold(f32::MAX, f32::min);
    let (mut px, ok) = resample_level(&m, t, &f);
    for (v, o) in px.iter_mut().zip(&ok) {
        if *o == 0 {
            *v = lo;
        }
    }
    Image2d::new(px, onto.dims())
}

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------

/// The weighted least-squares similarity taking the `fixed` points onto the
/// `moving` ones (both mm), the 2D case of Umeyama (1991): the rotation from
/// the cross-covariance's angle, the scale (when `scale`) from its norm over
/// the fixed points' spread, the shift from the centroids. About the frame's
/// origin (the image centre). One pair gives the shift alone. Returns the
/// transform and each pair's residual `|T(p) − q|`.
pub fn fit_similarity(
    pairs: &[([f64; 2], [f64; 2])],
    weights: &[f64],
    scale: bool,
) -> (Rigid2d, Vec<f64>) {
    fit_dof(
        pairs,
        weights,
        if scale {
            Dof2d::Similarity
        } else {
            Dof2d::ShiftRotation
        },
    )
}

fn fit_dof(pairs: &[([f64; 2], [f64; 2])], weights: &[f64], dof: Dof2d) -> (Rigid2d, Vec<f64>) {
    let w = |k: usize| weights.get(k).copied().unwrap_or(1.0).max(0.0);
    let total: f64 = (0..pairs.len()).map(w).sum();
    let mut r = Rigid2d::identity([0.0, 0.0]);
    if pairs.is_empty() || total <= 0.0 {
        return (r, vec![0.0; pairs.len()]);
    }
    let (mut mp, mut mq) = ([0.0; 2], [0.0; 2]);
    for (k, (p, q)) in pairs.iter().enumerate() {
        for a in 0..2 {
            mp[a] += w(k) * p[a] / total;
            mq[a] += w(k) * q[a] / total;
        }
    }
    let (mut sxx, mut sxy, mut spp) = (0.0, 0.0, 0.0);
    for (k, (p, q)) in pairs.iter().enumerate() {
        let (px, py) = (p[0] - mp[0], p[1] - mp[1]);
        let (qx, qy) = (q[0] - mq[0], q[1] - mq[1]);
        sxx += w(k) * (px * qx + py * qy);
        sxy += w(k) * (px * qy - py * qx);
        spp += w(k) * (px * px + py * py);
    }
    if pairs.len() >= 2 && dof != Dof2d::Shift && spp > 1e-12 {
        r.theta = sxy.atan2(sxx);
        if dof == Dof2d::Similarity {
            r.scale = (sxx * sxx + sxy * sxy).sqrt() / spp;
        }
    }
    // The shift: b = μq − s R μp, about the origin.
    let (s, c) = r.theta.sin_cos();
    r.tx = mq[0] - r.scale * (c * mp[0] - s * mp[1]);
    r.ty = mq[1] - r.scale * (s * mp[0] + c * mp[1]);
    let res = pairs
        .iter()
        .map(|(p, q)| {
            let m = r.map(*p);
            ((m[0] - q[0]).powi(2) + (m[1] - q[1]).powi(2)).sqrt()
        })
        .collect();
    (r, res)
}

/// Markers of `size_mm` found in both images and paired: the fixed image's
/// strongest spots carried through `start` onto the moving image, laid on
/// its spots by the common shift that fits the most of them
/// ([`crate::registration::markers`]), then paired one to one within the
/// search radius. Pixels of the full rasters.
pub fn marker_pairs(
    fixed: &Planar2d,
    moving: &Planar2d,
    mask: Option<&[u8]>,
    size_mm: f64,
    start: &Rigid2d,
    reach_mm: f64,
) -> Vec<([f64; 2], [f64; 2])> {
    use crate::registration::markers::{assign, constellation_shift, detect_2d};
    let sigma = |p: &Planar2d| (size_mm * 0.5 / p.spacing[0].min(p.spacing[1]).max(1e-6)).max(0.6);
    let mut fc = detect_2d(&fixed.image, mask, sigma(fixed), 5.0);
    fc.truncate(12);
    let mc = detect_2d(&moving.image, None, sigma(moving), 5.0);
    if fc.is_empty() || mc.is_empty() {
        return Vec::new();
    }
    let predicted: Vec<Option<[f64; 2]>> = fc
        .iter()
        .map(|c| {
            let q = start.map(px_to_mm(fixed.dims(), fixed.spacing, c.pixel));
            Some(mm_to_px(moving.dims(), moving.spacing, q))
        })
        .collect();
    let mpx = moving.spacing[0].min(moving.spacing[1]).max(1e-6);
    let reach = reach_mm / mpx;
    let tol = (size_mm / mpx).max(2.0);
    let shift = constellation_shift(&predicted, &mc, reach, tol);
    let shifted: Vec<Option<[f64; 2]>> = predicted
        .iter()
        .map(|p| p.map(|p| [p[0] + shift[0], p[1] + shift[1]]))
        .collect();
    // A loose first pairing (the shift leaves the turn), then the fit of
    // what paired carries every spot again and the pairing is made tight.
    let mut a = assign(&shifted, &mc, 3.0 * tol);
    for _ in 0..3 {
        let mm: Vec<([f64; 2], [f64; 2])> = fc
            .iter()
            .zip(&a)
            .filter_map(|(f, m)| {
                m.map(|m| {
                    (
                        px_to_mm(fixed.dims(), fixed.spacing, f.pixel),
                        px_to_mm(moving.dims(), moving.spacing, mc[m].pixel),
                    )
                })
            })
            .collect();
        if mm.len() < 2 {
            break;
        }
        let (fit, _) = fit_dof(&mm, &[], Dof2d::ShiftRotation);
        let again: Vec<Option<[f64; 2]>> = fc
            .iter()
            .map(|c| {
                let q = fit.map(px_to_mm(fixed.dims(), fixed.spacing, c.pixel));
                Some(mm_to_px(moving.dims(), moving.spacing, q))
            })
            .collect();
        a = assign(&again, &mc, tol);
    }
    fc.iter()
        .zip(a)
        .filter_map(|(f, m)| m.map(|m| (f.pixel, mc[m].pixel)))
        .collect()
}

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// The parameter vector of a dof, and back.
fn params_of(r: &Rigid2d, dof: Dof2d, radius: f64) -> Vec<f64> {
    r.to_params(radius)[..dof.count()].to_vec()
}

fn rigid_of(x: &[f64], base: &Rigid2d, dof: Dof2d, radius: f64) -> Rigid2d {
    let mut p = base.to_params(radius);
    p[..dof.count()].copy_from_slice(&x[..dof.count()]);
    Rigid2d::from_params(&p, radius, base.center)
}

/// The fixed mask at a level, with the moving image's footprint and a
/// border of two pixels.
fn combined(base: &[u8], ok: &[u8]) -> Mask2d {
    base.iter()
        .zip(ok)
        .map(|(a, b)| (*a != 0 && *b != 0) as u8)
        .collect()
}

/// Register `req.moving` onto `req.fixed`.
pub fn register(req: &Request2d, sink: &dyn ProgressSink) -> Result<Result2d> {
    let t0 = std::time::Instant::now();
    let (fd, md) = (req.fixed.dims(), req.moving.dims());
    if fd[0] < 8 || fd[1] < 8 || md[0] < 8 || md[1] < 8 {
        bail!("an image is too small to register");
    }
    if req.mask.as_ref().is_some_and(|m| m.len() != fd[0] * fd[1]) {
        bail!("the region does not fit the fixed image");
    }
    // The rotation centre: where the isocentre projects, else the centre.
    let iso = req.fixed.iso_frame();
    let center = iso.map(|(c, _)| c).unwrap_or([0.0, 0.0]);
    let mut start = req
        .start
        .map(|s| s.recentered(center))
        .unwrap_or_else(|| Rigid2d::identity(center));
    let mut warning: Option<String> = None;
    let mut evaluations = 0usize;

    // Points: placed and found.
    let mut pair_list: Vec<([f64; 2], [f64; 2])> = req.pairs.clone();
    if let Some(size) = req.detect_markers {
        let found = marker_pairs(
            &req.fixed,
            &req.moving,
            req.mask.as_deref(),
            size,
            &start,
            req.range_mm,
        );
        if found.is_empty() {
            warning = Some("no markers were paired between the two images".into());
        }
        pair_list.extend(found);
    }
    let wants_points = matches!(req.method, Method2d::Points | Method2d::PointsThenIntensity);
    let mut pairs_out: Vec<PairResidual> = Vec::new();
    let mut left_out: Vec<usize> = Vec::new();
    if wants_points {
        if pair_list.is_empty() {
            bail!("the points method needs at least one pair of points");
        }
        let mm: Vec<([f64; 2], [f64; 2])> = pair_list
            .iter()
            .map(|(p, q)| {
                (
                    px_to_mm(fd, req.fixed.spacing, *p),
                    px_to_mm(md, req.moving.spacing, *q),
                )
            })
            .collect();
        let dof = if mm.len() == 1 { Dof2d::Shift } else { req.dof };
        let (mut fit, mut res) = fit_dof(&mm, &[], dof);
        // A pair beyond three times the median residual is left out and the
        // fit made again, as a migrated seed is in 3D.
        if mm.len() >= 3 {
            let mut sorted = res.clone();
            sorted.sort_by(f64::total_cmp);
            let median = sorted[sorted.len() / 2].max(0.5);
            left_out = (0..mm.len()).filter(|&k| res[k] > 3.0 * median).collect();
            if !left_out.is_empty() && mm.len() - left_out.len() >= 2 {
                let w: Vec<f64> = (0..mm.len())
                    .map(|k| if left_out.contains(&k) { 0.0 } else { 1.0 })
                    .collect();
                let (f2, _) = fit_dof(&mm, &w, dof);
                fit = f2;
                res = mm
                    .iter()
                    .map(|(p, q)| {
                        let m = fit.map(*p);
                        ((m[0] - q[0]).powi(2) + (m[1] - q[1]).powi(2)).sqrt()
                    })
                    .collect();
            } else {
                left_out.clear();
            }
        }
        pairs_out = pair_list
            .iter()
            .zip(&res)
            .enumerate()
            .map(|(k, ((p, q), r))| PairResidual {
                fixed: *p,
                moving: *q,
                residual_mm: *r,
                used: !left_out.contains(&k),
            })
            .collect();
        start = fit.recentered(center);
    }

    let fixed_levels = levels_of(&req.fixed, req.levels, req.max_pixels);
    let moving_levels = levels_of(&req.moving, fixed_levels.len(), req.max_pixels);
    let nl = fixed_levels.len().min(moving_levels.len());
    let radius = {
        let [w, h] = fd;
        0.5 * ((w as f64 * req.fixed.spacing[0]).powi(2)
            + (h as f64 * req.fixed.spacing[1]).powi(2))
        .sqrt()
    };
    // The level masks: the region, less a border of two pixels.
    let masks: Vec<Mask2d> = (0..nl)
        .map(|l| {
            let lv = &fixed_levels[l];
            let border = border_mask(lv.image.dims, 2);
            match mask_at(req.mask.as_deref(), fd, lv) {
                Some(m) => combined(&m, &border),
                None => border,
            }
        })
        .collect();
    let lncc_radius = 7;
    let measure = |l: usize, t: &Transform2d, moving: &[LevelImage]| -> f64 {
        let (px, ok) = resample_level(&moving[l], t, &fixed_levels[l]);
        let m = combined(&masks[l], &ok);
        if m.iter().filter(|&&v| v != 0).count() < 32 {
            return f64::NEG_INFINITY;
        }
        req.metric.similarity(
            &fixed_levels[l].image.data,
            &px,
            fixed_levels[l].image.dims,
            Some(&m),
            lncc_radius,
        )
    };

    // Polarity: the moving image turned when it correlates negatively with
    // the fixed one at the start (a radiograph stored bone dark against a
    // DRR stored bone bright).
    let mut moving_levels = moving_levels;
    let coarse = nl - 1;
    let inverted = {
        let (px, ok) = resample_level(
            &moving_levels[coarse],
            &Transform2d::rigid(start),
            &fixed_levels[coarse],
        );
        let m = combined(&masks[coarse], &ok);
        ncc(&fixed_levels[coarse].image.data, &px, Some(&m)) < -0.05
    };
    if inverted {
        for l in &mut moving_levels {
            l.image.negate();
        }
    }

    let full_t = |r: Rigid2d| Transform2d::rigid(r);
    let before = measure(0, &full_t(start), &moving_levels);
    let mut cur = start;
    let mut lines = Vec::new();
    let intensity = !matches!(req.method, Method2d::Points);
    if intensity {
        let per_level = (req.max_evaluations / nl.max(1)).max(50);
        for l in (0..nl).rev() {
            if sink.cancelled() {
                bail!("{CANCELLED}");
            }
            sink.report(
                (nl - 1 - l) as f32 / nl as f32,
                &format!("2D/2D level {} of {nl}", nl - l),
            );
            let lv_before = measure(l, &full_t(cur), &moving_levels);
            // The coarsest level of a run with no start of its own begins
            // from the best whole-pixel shift in range.
            if l == coarse && req.start.is_none() && !wants_points {
                let (px, ok) = resample_level(&moving_levels[l], &full_t(cur), &fixed_levels[l]);
                let m = combined(&masks[l], &ok);
                let pix = fixed_levels[l].pixel_mm();
                let r = (req.range_mm / pix).ceil().clamp(1.0, 24.0) as isize;
                let (s, _) = crate::registration::projective::best_shift(
                    &fixed_levels[l].image.data,
                    &px,
                    fixed_levels[l].image.dims,
                    &m,
                    r,
                );
                // fixed(u) ≈ moved(u − s): the new transform is T(x − s).
                let sx = s[0] * fixed_levels[l].factor * req.fixed.spacing[0];
                let sy = s[1] * fixed_levels[l].factor * req.fixed.spacing[1];
                let shifted = cur.map([center[0] - sx, center[1] - sy]);
                let mut c2 = cur;
                c2.tx = shifted[0] - center[0];
                c2.ty = shifted[1] - center[1];
                evaluations += ((2 * r + 1) * (2 * r + 1)) as usize;
                if measure(l, &full_t(c2), &moving_levels) > lv_before {
                    cur = c2;
                }
            }
            let mut evals = 0usize;
            let x0 = params_of(&cur, req.dof, radius);
            let base = start;
            let mut bounds: Bounds = vec![
                (base.tx - req.range_mm, base.tx + req.range_mm),
                (base.ty - req.range_mm, base.ty + req.range_mm),
                (
                    (base.theta - req.range_deg.to_radians()) * radius,
                    (base.theta + req.range_deg.to_radians()) * radius,
                ),
                (
                    (base.scale.ln() - 0.1) * radius,
                    (base.scale.ln() + 0.1) * radius,
                ),
            ];
            bounds.truncate(req.dof.count());
            let pix = fixed_levels[l].pixel_mm();
            let lim = Limits {
                max_evaluations: per_level,
                ftol: 1e-6,
                xtol: 0.02 * pix,
                initial_step: 2.0 * pix,
            };
            let mut cost = |x: &[f64]| -> f64 {
                evals += 1;
                let v = measure(
                    l,
                    &full_t(rigid_of(x, &cur, req.dof, radius)),
                    &moving_levels,
                );
                if v.is_finite() {
                    -v
                } else {
                    1e9
                }
            };
            let opt = match req.optimizer {
                Optimizer2d::Powell => powell(&mut cost, &x0, &bounds, lim),
                Optimizer2d::NelderMead => nelder_mead(&mut cost, &x0, &bounds, lim),
            };
            let mut next = rigid_of(&opt.x, &cur, req.dof, radius);
            if l == 0 && req.refine {
                let h = 0.05 * pix;
                let fg = |x: &[f64], g: &mut [f64]| -> f64 {
                    let f = |y: &[f64]| -> f64 {
                        let v = measure(
                            0,
                            &full_t(rigid_of(y, &cur, req.dof, radius)),
                            &moving_levels,
                        );
                        if v.is_finite() {
                            -v
                        } else {
                            1e9
                        }
                    };
                    let f0 = f(x);
                    let mut y = x.to_vec();
                    for k in 0..x.len() {
                        y[k] = x[k] + h;
                        let fp = f(&y);
                        y[k] = x[k] - h;
                        let fm = f(&y);
                        y[k] = x[k];
                        g[k] = (fp - fm) / (2.0 * h);
                    }
                    f0
                };
                let o2 = lbfgs(fg, &opt.x, &bounds, 20, 0.5 * pix);
                evals += o2.evaluations * (1 + 2 * req.dof.count());
                if o2.value < opt.value {
                    next = rigid_of(&o2.x, &cur, req.dof, radius);
                }
            }
            cur = next;
            evaluations += evals;
            lines.push(LevelLine {
                level: l,
                dims: fixed_levels[l].image.dims,
                before: lv_before,
                after: measure(l, &full_t(cur), &moving_levels),
                evaluations: evals,
            });
        }
    }
    let mut transform = Transform2d::rigid(cur);
    let mut field = None;
    let mut warp_stats = None;
    if req.method == Method2d::Deformable {
        let (w, e) = deformable(req, &transform, &fixed_levels, &moving_levels, &masks, sink)?;
        evaluations += e;
        transform.warp = Some(w);
        let (f, s) = field_and_stats(&transform, &req.fixed);
        field = Some(f);
        if s.folds > 0 {
            warning = Some(format!(
                "the warp folds at {} pixel(s): contours are not carried there",
                s.folds
            ));
        }
        warp_stats = Some(s);
    }
    let after = if req.method == Method2d::Deformable {
        measure(0, &transform, &moving_levels)
    } else {
        measure(0, &full_t(cur), &moving_levels)
    };
    let shift_iso_mm = iso.map(|(c, mag)| {
        // The shift of the isocentre's projection, read at the isocentre.
        let m = transform.rigid.map(c);
        [(m[0] - c[0]) / mag, (m[1] - c[1]) / mag]
    });
    sink.report(1.0, "2D/2D registration done");
    Ok(Result2d {
        transform,
        method: req.method,
        metric: req.metric,
        before,
        after,
        levels: lines,
        pairs: pairs_out,
        left_out,
        field,
        warp_stats,
        shift_iso_mm,
        inverted,
        evaluations,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        warning,
    })
}

// ---------------------------------------------------------------------------
// The deformable method: a B-spline over the rigid start
// ---------------------------------------------------------------------------

/// The samples of one level: fixed pixels in the mask, in row-major order,
/// with their position (mm) and value.
struct Samples {
    x: Vec<[f64; 2]>,
    f: Vec<f64>,
}

fn samples_of(fixed: &LevelImage, mask: &[u8]) -> Samples {
    let [w, h] = fixed.image.dims;
    let (mut x, mut f) = (Vec::new(), Vec::new());
    for j in 0..h {
        for i in 0..w {
            let o = j * w + i;
            if mask[o] != 0 {
                x.push(fixed.to_mm([i as f64, j as f64]));
                f.push(fixed.image.data[o] as f64);
            }
        }
    }
    Samples { x, f }
}

/// Mean and standard deviation of `v`, summed in order.
fn mean_sd(v: impl Iterator<Item = f64> + Clone) -> (f64, f64) {
    let (mut n, mut s, mut ss) = (0.0f64, 0.0f64, 0.0f64);
    for a in v {
        n += 1.0;
        s += a;
        ss += a * a;
    }
    if n < 1.0 {
        return (0.0, 1.0);
    }
    let m = s / n;
    (m, ((ss / n - m * m).max(0.0)).sqrt().max(1e-12))
}

/// What one evaluation of the deformable cost reads: the samples, the
/// moving level and how its intensities are normalised.
struct DeformCost<'a> {
    samples: &'a Samples,
    moving: &'a LevelImage,
    rigid: Rigid2d,
    metric: DeformMetric,
    lambda: f64,
    /// Normalisation of the fixed and the moving intensities (MSD), or
    /// the bin mapping (MI): `(v − offset) · scale`.
    f_norm: (f64, f64),
    m_norm: (f64, f64),
}

/// Bins of the 2D Mattes histogram.
const MI_BINS: usize = 32;

impl DeformCost<'_> {
    /// The cost and its gradient with respect to the coefficients of `w`
    /// (written into `grad`).
    fn eval(&self, w: &BSpline2d, grad: &mut [f64]) -> f64 {
        grad.iter_mut().for_each(|g| *g = 0.0);
        let n = self.samples.x.len();
        let nx = w.grid_dims[0];
        let inv_px = [
            1.0 / (self.moving.spacing[0] * self.moving.factor),
            1.0 / (self.moving.spacing[1] * self.moving.factor),
        ];
        // Per sample: the warped moving value and its gradient (per mm of
        // the moving frame), and the support.
        struct At {
            m: f64,
            g: [f64; 2],
            support: Option<([usize; 2], [f64; 4], [f64; 4])>,
        }
        let at: Vec<At> = self
            .samples
            .x
            .par_iter()
            .map(|&x| {
                let q = self.rigid.map(x);
                let d = w.displacement(x);
                let y = [q[0] + d[0], q[1] + d[1]];
                let p = self.moving.to_px(y);
                let (v, gx, gy) = self.moving.image.sample_with_gradient(p[0], p[1]);
                At {
                    m: (v as f64 - self.m_norm.0) * self.m_norm.1,
                    g: [
                        gx as f64 * inv_px[0] * self.m_norm.1,
                        gy as f64 * inv_px[1] * self.m_norm.1,
                    ],
                    support: w.support(x),
                }
            })
            .collect();
        // dCost/dm per sample, and the cost.
        let (cost, dm): (f64, Vec<f64>) = match self.metric {
            DeformMetric::Msd => {
                let mut c = 0.0;
                let mut dm = Vec::with_capacity(n);
                for (a, f) in at.iter().zip(&self.samples.f) {
                    let r = a.m - (f - self.f_norm.0) * self.f_norm.1;
                    c += r * r;
                    dm.push(2.0 * r / n as f64);
                }
                (c / n.max(1) as f64, dm)
            }
            DeformMetric::Mi => self.mi(&at.iter().map(|a| a.m).collect::<Vec<_>>()),
        };
        // Scattered onto the 16 nodes of each sample's support, in order.
        for (a, d) in at.iter().zip(&dm) {
            let Some(([a0, b0], wx, wy)) = a.support else {
                continue;
            };
            let (gx, gy) = (d * a.g[0], d * a.g[1]);
            for (mm, wym) in wy.iter().enumerate() {
                for (l, wxl) in wx.iter().enumerate() {
                    let o = 2 * ((b0 + mm) * nx + a0 + l);
                    let ww = wxl * wym;
                    grad[o] += gx * ww;
                    grad[o + 1] += gy * ww;
                }
            }
        }
        cost + self.lambda * bending_energy(w, Some(grad), self.lambda)
    }

    /// Mattes mutual information (negated, a cost) of the fixed samples'
    /// bins and the warped moving values (already in bin units), with its
    /// derivative per sample.
    fn mi(&self, m: &[f64]) -> (f64, Vec<f64>) {
        let b = MI_BINS;
        let n = m.len().max(1) as f64;
        let fbin: Vec<usize> = self
            .samples
            .f
            .iter()
            .map(|f| {
                (((f - self.f_norm.0) * self.f_norm.1).round() as isize).clamp(0, b as isize - 1)
                    as usize
            })
            .collect();
        let mut joint = vec![0.0f64; b * b];
        for (k, &mv) in m.iter().enumerate() {
            let c = mv.floor() as isize;
            for d in -1..=2 {
                let kb = c + d;
                if kb < 0 || kb >= b as isize {
                    continue;
                }
                joint[fbin[k] * b + kb as usize] += super::parzen::beta3(kb as f64 - mv) / n;
            }
        }
        let mut pm = vec![0.0f64; b];
        let mut pf = vec![0.0f64; b];
        for i in 0..b {
            for k in 0..b {
                pm[k] += joint[i * b + k];
                pf[i] += joint[i * b + k];
            }
        }
        let mut mi = 0.0;
        let mut lg = vec![0.0f64; b * b];
        for i in 0..b {
            for k in 0..b {
                let p = joint[i * b + k];
                if p > 1e-12 && pm[k] > 1e-12 && pf[i] > 1e-12 {
                    mi += p * (p / (pf[i] * pm[k])).ln();
                    lg[i * b + k] = (p / pm[k]).ln();
                }
            }
        }
        // ∂MI/∂m(x) = −(1/N) Σ_κ β³'(κ − m) log(p(ι, κ) / p_m(κ)).
        let dm = m
            .iter()
            .enumerate()
            .map(|(k, &mv)| {
                let c = mv.floor() as isize;
                let mut s = 0.0;
                for d in -1..=2 {
                    let kb = c + d;
                    if kb < 0 || kb >= b as isize {
                        continue;
                    }
                    s += super::parzen::dbeta3(kb as f64 - mv) * lg[fbin[k] * b + kb as usize];
                }
                // MI is maximised: the cost is −MI, d(−MI)/dm = +s / N.
                s / n
            })
            .collect();
        (-mi, dm)
    }
}

/// The bending energy of the warp's coefficients: the second differences
/// along x, along y and across (twice), per node spacing, squared and
/// averaged over the nodes - zero for any affine field. With `grad`, its exact
/// gradient times `lambda` is added in.
pub fn bending_energy(w: &BSpline2d, grad: Option<&mut [f64]>, lambda: f64) -> f64 {
    let [nx, ny] = w.grid_dims;
    if nx < 3 || ny < 3 {
        return 0.0;
    }
    // Second differences per node spacing: dimensionless, so lambda weighs
    // the same against the image measure whatever the grid.
    let s2 = w.spacing;
    let count = (nx * ny) as f64;
    let c = |a: usize, b: usize, comp: usize| w.coeffs[2 * (b * nx + a) + comp];
    let mut e = 0.0;
    let mut g = grad;
    for comp in 0..2 {
        for b in 1..ny - 1 {
            for a in 1..nx - 1 {
                let dxx = (c(a + 1, b, comp) - 2.0 * c(a, b, comp) + c(a - 1, b, comp)) / s2;
                let dyy = (c(a, b + 1, comp) - 2.0 * c(a, b, comp) + c(a, b - 1, comp)) / s2;
                let dxy = (c(a + 1, b + 1, comp) - c(a + 1, b - 1, comp) - c(a - 1, b + 1, comp)
                    + c(a - 1, b - 1, comp))
                    / (4.0 * s2);
                e += (dxx * dxx + dyy * dyy + 2.0 * dxy * dxy) / count;
                if let Some(g) = g.as_deref_mut() {
                    let k = lambda * 2.0 / count;
                    let o = |aa: usize, bb: usize| 2 * (bb * nx + aa) + comp;
                    g[o(a + 1, b)] += k * dxx / s2;
                    g[o(a, b)] += k * dxx * -2.0 / s2;
                    g[o(a - 1, b)] += k * dxx / s2;
                    g[o(a, b + 1)] += k * dyy / s2;
                    g[o(a, b)] += k * dyy * -2.0 / s2;
                    g[o(a, b - 1)] += k * dyy / s2;
                    let kx = k * 2.0 * dxy / (4.0 * s2);
                    g[o(a + 1, b + 1)] += kx;
                    g[o(a + 1, b - 1)] -= kx;
                    g[o(a - 1, b + 1)] -= kx;
                    g[o(a - 1, b - 1)] += kx;
                }
            }
        }
    }
    e
}

/// The warp on a grid of half the spacing, exactly the same displacement
/// inside (the dyadic refinement of the cubic B-spline, axis by axis).
pub fn refine(w: &BSpline2d, lo: [f64; 2], hi: [f64; 2]) -> BSpline2d {
    let mut out = BSpline2d::zero(lo, hi, w.spacing * 0.5);
    let [nx, ny] = w.grid_dims;
    let [mx, my] = out.grid_dims;
    // Old node a sits at lo + (a − 1) s; new node b at lo + (b − 1) s / 2,
    // which is old index (b + 1) / 2.
    let old = |a: isize, b: isize, comp: usize| -> f64 {
        let a = a.clamp(0, nx as isize - 1) as usize;
        let b = b.clamp(0, ny as isize - 1) as usize;
        w.coeffs[2 * (b * nx + a) + comp]
    };
    let rule = |b: usize| -> Vec<(isize, f64)> {
        if b % 2 == 1 {
            let i = (b as isize + 1) / 2;
            vec![(i - 1, 0.125), (i, 0.75), (i + 1, 0.125)]
        } else {
            let i = b as isize / 2;
            vec![(i, 0.5), (i + 1, 0.5)]
        }
    };
    for by in 0..my {
        let ry = rule(by);
        for bx in 0..mx {
            let rx = rule(bx);
            for comp in 0..2 {
                let mut v = 0.0;
                for &(iy, wy) in &ry {
                    for &(ix, wx) in &rx {
                        v += wx * wy * old(ix, iy, comp);
                    }
                }
                out.coeffs[2 * (by * mx + bx) + comp] = v;
            }
        }
    }
    out
}

/// The fixed image's extent in its millimetre frame (pixel edges).
fn extent(p: &Planar2d) -> ([f64; 2], [f64; 2]) {
    let d = p.dims();
    (
        px_to_mm(d, p.spacing, [-0.5, -0.5]),
        px_to_mm(d, p.spacing, [d[0] as f64 - 0.5, d[1] as f64 - 0.5]),
    )
}

/// The warp of the deformable method over the rigid part of `t`, coarse
/// to fine; the evaluations spent.
fn deformable(
    req: &Request2d,
    t: &Transform2d,
    fixed_levels: &[LevelImage],
    moving_levels: &[LevelImage],
    masks: &[Mask2d],
    sink: &dyn ProgressSink,
) -> Result<(BSpline2d, usize)> {
    let (lo, hi) = extent(&req.fixed);
    let mut spacings = vec![req.grid_mm.max(req.grid_min_mm).max(1.0)];
    while spacings.last().copied().unwrap_or(0.0) * 0.5 >= req.grid_min_mm.max(1.0) - 1e-9 {
        let next = spacings.last().unwrap() * 0.5;
        spacings.push(next);
    }
    let nl = fixed_levels.len().min(moving_levels.len());
    let mut w = BSpline2d::zero(lo, hi, spacings[0]);
    let mut evaluations = 0;
    for (g, &sp) in spacings.iter().enumerate() {
        if sink.cancelled() {
            bail!("{CANCELLED}");
        }
        sink.report(
            g as f32 / spacings.len() as f32,
            &format!("deformable, grid {sp:.0} mm"),
        );
        if g > 0 {
            w = refine(&w, lo, hi);
        }
        // The finer the grid, the finer the level it is fitted on.
        let l = (nl - 1).saturating_sub(g).min(nl - 1);
        let samples = samples_of(&fixed_levels[l], &masks[l]);
        if samples.x.len() < 64 {
            continue;
        }
        let cost = match req.deformable_metric {
            DeformMetric::Msd => {
                // One scale for both: the measure is for two images of one
                // kind, whose values mean the same.
                let (fm, fs) = mean_sd(samples.f.iter().copied());
                DeformCost {
                    samples: &samples,
                    moving: &moving_levels[l],
                    rigid: t.rigid,
                    metric: DeformMetric::Msd,
                    lambda: req.lambda,
                    f_norm: (fm, 1.0 / fs),
                    m_norm: (fm, 1.0 / fs),
                }
            }
            DeformMetric::Mi => {
                // Intensities onto bins 1 .. B − 2, the window's reach inside.
                let range = |v: &mut dyn Iterator<Item = f64>| {
                    v.fold((f64::MAX, f64::MIN), |(a, b), x| (a.min(x), b.max(x)))
                };
                let (f0, f1) = range(&mut samples.f.iter().copied());
                let (m0, m1) = range(&mut moving_levels[l].image.data.iter().map(|v| *v as f64));
                let span = (MI_BINS - 3) as f64;
                DeformCost {
                    samples: &samples,
                    moving: &moving_levels[l],
                    rigid: t.rigid,
                    metric: DeformMetric::Mi,
                    lambda: req.lambda,
                    f_norm: (
                        f0 - (f1 - f0).max(1e-12) / span,
                        span / (f1 - f0).max(1e-12),
                    ),
                    m_norm: (
                        m0 - (m1 - m0).max(1e-12) / span,
                        span / (m1 - m0).max(1e-12),
                    ),
                }
            }
        };
        let bounds: Bounds = vec![(-sp, sp); w.coeffs.len()];
        let mut warp = w.clone();
        let opt = lbfgs(
            |x: &[f64], gr: &mut [f64]| {
                warp.coeffs.copy_from_slice(x);
                cost.eval(&warp, gr)
            },
            &w.coeffs,
            &bounds,
            100,
            0.25 * sp,
        );
        evaluations += opt.evaluations;
        w.coeffs = opt.x;
    }
    Ok((w, evaluations))
}

/// The dense field of a transform over the fixed image (every `factor`
/// full pixels, at most 512 a side), and what the warp does there.
pub fn field_and_stats(t: &Transform2d, fixed: &Planar2d) -> (Field2d, WarpStats2d) {
    let full = fixed.dims();
    let factor = (full[0].max(full[1]).div_ceil(512)).max(1);
    let dims = [full[0].div_ceil(factor), full[1].div_ceil(factor)];
    let mut dx = vec![0.0f32; dims[0] * dims[1]];
    let mut dy = vec![0.0f32; dims[0] * dims[1]];
    let mut jac = vec![1.0f64; dims[0] * dims[1]];
    let (s, c) = t.rigid.theta.sin_cos();
    let sr = [
        [t.rigid.scale * c, -t.rigid.scale * s],
        [t.rigid.scale * s, t.rigid.scale * c],
    ];
    for j in 0..dims[1] {
        for i in 0..dims[0] {
            let p = [(i * factor) as f64, (j * factor) as f64];
            let x = px_to_mm(full, fixed.spacing, p);
            let y = t.map(x);
            let o = j * dims[0] + i;
            dx[o] = (y[0] - x[0]) as f32;
            dy[o] = (y[1] - x[1]) as f32;
            let jd = t
                .warp
                .as_ref()
                .map(|w| w.displacement_jacobian(x))
                .unwrap_or([[0.0; 2]; 2]);
            let a = [
                [sr[0][0] + jd[0][0], sr[0][1] + jd[0][1]],
                [sr[1][0] + jd[1][0], sr[1][1] + jd[1][1]],
            ];
            jac[o] = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        }
    }
    // The warp's own displacement (the rigid part left out), for the stats.
    let (mut sum, mut max) = (0.0f64, 0.0f64);
    let mut n = 0.0f64;
    for j in 0..dims[1] {
        for i in 0..dims[0] {
            let x = px_to_mm(
                full,
                fixed.spacing,
                [(i * factor) as f64, (j * factor) as f64],
            );
            let d = t
                .warp
                .as_ref()
                .map(|w| w.displacement(x))
                .unwrap_or([0.0, 0.0]);
            let m = (d[0] * d[0] + d[1] * d[1]).sqrt();
            sum += m;
            max = max.max(m);
            n += 1.0;
        }
    }
    let (jmin, jmax) = jac
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let stats = WarpStats2d {
        mean_mm: sum / n.max(1.0),
        max_mm: max,
        jacobian_min: jmin,
        jacobian_max: jmax,
        folds: jac.iter().filter(|&&v| v <= 0.0).count(),
    };
    (
        Field2d {
            dims,
            factor,
            dx,
            dy,
        },
        stats,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_similarity_and_its_inverse_undo_each_other() {
        let t = Rigid2d {
            tx: 7.0,
            ty: -3.0,
            theta: 0.05,
            scale: 1.04,
            center: [10.0, -20.0],
        };
        let inv = t.inverse();
        for p in [[0.0, 0.0], [55.0, -12.0], [-80.0, 31.0]] {
            let q = t.map(p);
            let a = t.inverse_map(q);
            let b = inv.map(q);
            for k in 0..2 {
                assert!((a[k] - p[k]).abs() < 1e-9);
                assert!((b[k] - p[k]).abs() < 1e-9);
            }
        }
        let r = t.recentered([0.0, 0.0]);
        for p in [[3.0, 4.0], [-50.0, 9.0]] {
            let (a, b) = (t.map(p), r.map(p));
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        }
        let back = Rigid2d::from_params(&t.to_params(150.0), 150.0, t.center);
        assert!((back.theta - t.theta).abs() < 1e-12 && (back.scale - t.scale).abs() < 1e-12);
    }

    #[test]
    fn the_millimetre_frame_is_centred() {
        let dims = [4, 2];
        let sp = [0.5, 2.0];
        assert_eq!(px_to_mm(dims, sp, [1.5, 0.5]), [0.0, 0.0]);
        let q = px_to_mm(dims, sp, [0.0, 0.0]);
        assert_eq!(q, [-0.75, -1.0]);
        let p = mm_to_px(dims, sp, q);
        assert!((p[0]).abs() < 1e-12 && (p[1]).abs() < 1e-12);
    }

    #[test]
    fn the_bspline_weights_sum_to_one_and_their_derivatives_to_zero() {
        for t in [0.0, 0.25, 0.6, 0.999] {
            let w = bspline_weights(t);
            let d = bspline_dweights(t);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(d.iter().sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    fn a_warp_inverts_by_iteration() {
        let mut w = BSpline2d::zero([-50.0, -50.0], [50.0, 50.0], 20.0);
        for (k, c) in w.coeffs.iter_mut().enumerate() {
            *c = ((k as f64) * 0.7).sin() * 2.0;
        }
        let t = Transform2d {
            rigid: Rigid2d {
                tx: 2.0,
                theta: 0.02,
                ..Rigid2d::identity([0.0, 0.0])
            },
            warp: Some(w),
        };
        for p in [[0.0, 0.0], [20.0, -13.0], [-31.0, 40.0]] {
            let q = t.map(p);
            let back = t.inverse_map(q, 1e-6).expect("settles");
            assert!((back[0] - p[0]).abs() < 1e-4 && (back[1] - p[1]).abs() < 1e-4);
        }
    }

    /// A smooth test image of `n × n` pixels of 1.5 mm.
    fn smooth(n: usize, shift: f64) -> LevelImage {
        let data: Vec<f32> = (0..n * n)
            .map(|o| {
                let (x, y) = ((o % n) as f64 + shift, (o / n) as f64);
                (50.0 + 20.0 * (x * 0.21).sin() * (y * 0.13).cos() + 10.0 * ((x + y) * 0.07).sin())
                    as f32
            })
            .collect();
        LevelImage {
            image: Image2d::new(data, [n, n]),
            factor: 1.0,
            full: [n, n],
            spacing: [1.5, 1.5],
        }
    }

    fn check_gradient(metric: DeformMetric) {
        let n = 48;
        let fixed = smooth(n, 0.0);
        let moving = smooth(n, 1.3);
        let mask = border_mask([n, n], 3);
        let samples = samples_of(&fixed, &mask);
        let lo = [-36.0, -36.0];
        let mut w = BSpline2d::zero(lo, [36.0, 36.0], 18.0);
        for (k, c) in w.coeffs.iter_mut().enumerate() {
            *c = ((k as f64) * 1.7).sin() * 1.5;
        }
        let (f_norm, m_norm) = match metric {
            DeformMetric::Msd => ((50.0, 1.0 / 15.0), (50.0, 1.0 / 15.0)),
            DeformMetric::Mi => ((15.0, 29.0 / 70.0), (15.0, 29.0 / 70.0)),
        };
        let cost = DeformCost {
            samples: &samples,
            moving: &moving,
            rigid: Rigid2d {
                tx: 0.7,
                theta: 0.02,
                ..Rigid2d::identity([0.0, 0.0])
            },
            metric,
            lambda: 0.05,
            f_norm,
            m_norm,
        };
        let mut g = vec![0.0; w.coeffs.len()];
        cost.eval(&w, &mut g);
        let mut scratch = vec![0.0; w.coeffs.len()];
        // Bilinear sampling in f32 is piecewise linear: a step of a
        // thousandth of a millimetre keeps the difference quotient clear of
        // both the rounding and the kinks.
        let h = 1e-3;
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for k in (0..w.coeffs.len()).step_by(7) {
            let mut wp = w.clone();
            wp.coeffs[k] += h;
            let fp = cost.eval(&wp, &mut scratch);
            wp.coeffs[k] -= 2.0 * h;
            let fm = cost.eval(&wp, &mut scratch);
            let fd = (fp - fm) / (2.0 * h);
            worst = worst.max((fd - g[k]).abs());
            scale = scale.max(fd.abs()).max(g[k].abs());
        }
        assert!(
            scale > 0.0,
            "{metric:?}: the cost moves with the coefficients"
        );
        assert!(
            worst <= 1e-2 * scale + 1e-9,
            "{metric:?}: {worst} against {scale}"
        );
    }

    #[test]
    fn the_msd_gradient_is_the_finite_differences() {
        check_gradient(DeformMetric::Msd);
    }

    #[test]
    fn the_mi_gradient_is_the_finite_differences() {
        check_gradient(DeformMetric::Mi);
    }

    #[test]
    fn the_bending_energy_of_an_affine_field_is_zero() {
        let mut w = BSpline2d::zero([-50.0, -40.0], [50.0, 40.0], 10.0);
        let [nx, _] = w.grid_dims;
        for k in 0..w.coeffs.len() / 2 {
            let (a, b) = ((k % nx) as f64, (k / nx) as f64);
            w.coeffs[2 * k] = 0.3 * a - 0.2 * b + 1.0;
            w.coeffs[2 * k + 1] = -0.1 * a + 0.4 * b - 2.0;
        }
        assert!(bending_energy(&w, None, 1.0).abs() < 1e-20);
        // And the gradient of a curved field is the finite differences.
        for (k, c) in w.coeffs.iter_mut().enumerate() {
            *c += ((k as f64) * 0.9).sin();
        }
        let mut g = vec![0.0; w.coeffs.len()];
        let e = bending_energy(&w, Some(&mut g), 1.0);
        assert!(e > 0.0);
        let h = 1e-5;
        for k in (0..w.coeffs.len()).step_by(5) {
            let mut wp = w.clone();
            wp.coeffs[k] += h;
            let ep = bending_energy(&wp, None, 1.0);
            wp.coeffs[k] -= 2.0 * h;
            let em = bending_energy(&wp, None, 1.0);
            let fd = (ep - em) / (2.0 * h);
            assert!(
                (fd - g[k]).abs() < 1e-8 + 1e-5 * fd.abs(),
                "{k}: {fd} vs {}",
                g[k]
            );
        }
    }

    #[test]
    fn refining_a_warp_keeps_its_displacement() {
        let (lo, hi) = ([-60.0, -40.0], [60.0, 40.0]);
        let mut w = BSpline2d::zero(lo, hi, 20.0);
        for (k, c) in w.coeffs.iter_mut().enumerate() {
            *c = ((k as f64) * 0.37).cos() * 3.0;
        }
        let r = refine(&w, lo, hi);
        assert!((r.spacing - 10.0).abs() < 1e-12);
        for p in [[0.0, 0.0], [-55.0, 33.0], [41.0, -12.5], [59.0, 39.0]] {
            let (a, b) = (w.displacement(p), r.displacement(p));
            assert!(
                (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9,
                "{p:?}: {a:?} {b:?}"
            );
        }
    }

    #[test]
    fn a_pixel_samples_with_its_gradient() {
        let img = Image2d::new((0..20).map(|v| v as f32 * 2.0).collect(), [5, 4]);
        // v(x, y) = 2 (5 y + x): ∂x 2, ∂y 10.
        let (v, gx, gy) = img.sample_with_gradient(1.5, 2.25);
        assert!((v - 2.0 * (5.0 * 2.25 + 1.5)).abs() < 1e-5);
        assert!((gx - 2.0).abs() < 1e-5 && (gy - 10.0).abs() < 1e-5);
        assert!(img.sample(-0.1, 1.0).is_none() && img.sample(4.0, 3.0).is_some());
        let (_, gx, _) = img.sample_with_gradient(-3.0, 1.0);
        assert_eq!(gx, 0.0, "outside, no gradient across the edge");
    }
}
