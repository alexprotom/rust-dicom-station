//! What a recovered transform actually *did*: six degrees of freedom,
//! displacement statistics and the Jacobian of the deformation.
//!
//! A registration result is otherwise two numbers (a metric before and
//! after) and a black box. Everything here is measured on the transform
//! itself, on a regular lattice over the fixed image or over the region a
//! local run was restricted to, so it applies to any method - the numbers
//! for a landmark warp are computed exactly the same way as for a B-spline.
//!
//! * **Six degrees of freedom.** Even a deformable result has a best-fitting
//!   rigid body, and it is usually the number a physicist wants first: how
//!   far did the patient move, and how far did they turn? It is the
//!   orthogonal Procrustes fit of the mapping over the sampled points -
//!   translation, three Euler angles in the same `Rz Ry Rx` convention as
//!   [`RigidTransform`], and the RMS residual, which says how much of the
//!   transform those six numbers do *not* explain (zero for a rigid result,
//!   by construction).
//! * **Displacements.** Magnitude statistics of `T(p) − p` in millimetres,
//!   plus the mean vector, which separates a systematic shift from
//!   scattered local motion.
//! * **Jacobian determinant.** `det(I + ∂d/∂x)` by central differences:
//!   above 1 the tissue expanded, below 1 it compressed, and at or below
//!   zero the deformation folded onto itself - which is not anatomy, it is
//!   an artefact, and the folded fraction is the standard way to say so.
//! * **Overlap.** Everything above describes the transform; none of it says
//!   whether the images ended up on top of each other. [`OverlapStats`] does:
//!   a tissue mask of the fixed image against the same mask of the moving
//!   image pulled through the transform, as a Dice coefficient - before and
//!   after, because a Dice of 0.94 means nothing until you know it was 0.71
//!   to begin with.

use super::*;

/// Magnitude statistics of a set of vectors, in millimetres.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorStats {
    pub mean: f64,
    /// 95th percentile - where the bulk of the motion ends.
    pub p95: f64,
    pub max: f64,
    pub rms: f64,
}

impl VectorStats {
    /// Statistics of a set of displacement vectors.
    pub fn of(vectors: &[Vec3]) -> VectorStats {
        let mut mags: Vec<f64> = vectors.iter().map(|v| v.length()).collect();
        if mags.is_empty() {
            return VectorStats::default();
        }
        let n = mags.len() as f64;
        let mean = mags.iter().sum::<f64>() / n;
        let rms = (mags.iter().map(|m| m * m).sum::<f64>() / n).sqrt();
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((mags.len() as f64 * 0.95).ceil() as usize).min(mags.len()) - 1;
        VectorStats {
            mean,
            p95: mags[idx],
            max: *mags.last().unwrap(),
            rms,
        }
    }

    /// `mean 3.1 · p95 7.8 · max 11.2 mm`.
    pub fn line(&self) -> String {
        format!(
            "mean {:.2} · p95 {:.2} · max {:.2} mm",
            self.mean, self.p95, self.max
        )
    }
}

/// How much the deformation expands or compresses tissue.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct JacobianStats {
    pub min: f64,
    pub mean: f64,
    pub max: f64,
    /// Fraction of sample points where the determinant is ≤ 0 - the
    /// deformation folded, which is never anatomy.
    pub folded: f64,
}

impl JacobianStats {
    /// `det J 0.82 - 1.24 (mean 1.00), no folding`.
    pub fn line(&self) -> String {
        format!(
            "det J {:.2} - {:.2} (mean {:.2}), {}",
            self.min,
            self.max,
            self.mean,
            if self.folded <= 0.0 {
                "no folding".to_string()
            } else {
                format!("folded at {:.2} % of points", 100.0 * self.folded)
            }
        )
    }
}

/// How well the two images actually overlap, as a Dice coefficient.
///
/// The transform statistics above are all measured on the transform alone,
/// which is to say they describe what was *done*, never whether it was right.
/// This is the other half: threshold both images into a tissue mask, pull the
/// moving one through the transform, and count.
///
/// `before` is the same measurement with the identity in place of the
/// transform, so the pair reads as "the images overlapped this much, and now
/// they overlap this much". A registration that leaves Dice where it found it
/// did nothing, however small its final metric.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OverlapStats {
    /// Dice of the two tissue masks with no transform applied.
    pub before: f64,
    /// Dice of the two tissue masks through the transform.
    pub after: f64,
    /// The value at or above which a voxel counted as tissue, in the image's
    /// own units (Hounsfield units for CT).
    pub threshold: f64,
    /// Sample points that fell inside the moving image. A registration that
    /// pushes the fixed image off the end of the moving one scores well on
    /// the little that is left, so this is reported with the number.
    pub samples: usize,
}

impl OverlapStats {
    /// `Dice 0.71 ▶ 0.96`.
    pub fn line(&self) -> String {
        format!("Dice {:.3} ▶ {:.3}", self.before, self.after)
    }

    /// What the score gained. Negative means the registration made the
    /// overlap worse than it was.
    pub fn gain(&self) -> f64 {
        self.after - self.before
    }
}

/// The rigid body that best explains a mapping.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dof6 {
    /// Translation of the fit, mm.
    pub translation: Vec3,
    /// Euler angles `[rx, ry, rz]` in degrees, `Rz Ry Rx` - the same
    /// convention as [`RigidTransform`].
    pub rotation_deg: [f64; 3],
    /// RMS distance between the fit and the real mapping, mm: how much of
    /// the transform the six numbers do not account for.
    pub residual_mm: f64,
}

impl Dof6 {
    /// `t = (1.2, −0.4, 3.0) mm   r = (0.51, −0.10, 0.03)°`.
    pub fn line(&self) -> String {
        format!(
            "t = ({:.2}, {:.2}, {:.2}) mm   r = ({:.2}, {:.2}, {:.2})°",
            self.translation.x,
            self.translation.y,
            self.translation.z,
            self.rotation_deg[0],
            self.rotation_deg[1],
            self.rotation_deg[2]
        )
    }
}

/// Everything measured about one registration result.
#[derive(Clone, Debug, Default)]
pub struct RegAnalysis {
    pub dof: Dof6,
    pub displacement: VectorStats,
    /// Mean displacement vector (LPS), mm - a systematic shift shows here
    /// while the magnitude statistics cannot tell it from random motion.
    pub mean_vector: Vec3,
    pub jacobian: JacobianStats,
    /// Points the statistics were measured over.
    pub samples: usize,
    /// Lattice step of the sampling, mm.
    pub step_mm: f64,
    /// Image overlap before and after, when both images were available to
    /// measure it (a landmark warp solved from points alone has none).
    pub overlap: Option<OverlapStats>,
}

// ---------------------------------------------------------------------------
// 3 × 3 helpers (Procrustes needs a polar decomposition, nothing more)
// ---------------------------------------------------------------------------

type M3 = [[f64; 3]; 3];

fn m3_transpose(a: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for (i, row) in a.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            r[j][i] = *v;
        }
    }
    r
}

fn m3_det(a: &M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn m3_inverse(a: &M3) -> Option<M3> {
    let det = m3_det(a);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            // Cofactor of (j, i) - the adjugate is the transposed cofactor
            // matrix, which is what the inverse needs.
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            *v = (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) * inv;
        }
    }
    Some(r)
}

/// Nearest rotation to `a`, by Higham's polar-decomposition iteration
/// `R ← ½(R + R⁻ᵀ)` - quadratically convergent and free of any eigen
/// solver, which is why the whole analysis needs no linear-algebra
/// dependency.
fn nearest_rotation(a: &M3) -> M3 {
    let mut r = *a;
    if m3_det(&r).abs() < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    for _ in 0..32 {
        let Some(inv) = m3_inverse(&r) else { break };
        let it = m3_transpose(&inv);
        let mut next = [[0.0; 3]; 3];
        let mut delta = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                next[i][j] = 0.5 * (r[i][j] + it[i][j]);
                delta += (next[i][j] - r[i][j]).abs();
            }
        }
        r = next;
        if delta < 1e-14 {
            break;
        }
    }
    if m3_det(&r) < 0.0 {
        // A reflection is not a rotation: flip the least-significant column.
        for row in r.iter_mut() {
            row[2] = -row[2];
        }
    }
    r
}

/// Euler angles of `R = Rz(rz) · Ry(ry) · Rx(rx)`, radians.
fn euler_zyx(r: &M3) -> [f64; 3] {
    let sy = -r[2][0];
    let ry = sy.clamp(-1.0, 1.0).asin();
    // Gimbal lock: with cos(ry) ≈ 0 only the sum rx ± rz is determined;
    // putting it all in rx is the usual convention and keeps the fit exact.
    if (1.0 - sy.abs()) < 1e-9 {
        [r[0][1].atan2(r[1][1]), ry, 0.0]
    } else {
        [r[2][1].atan2(r[2][2]), ry, r[1][0].atan2(r[0][0])]
    }
}

/// The rigid body that best explains `p → q` (orthogonal Procrustes).
pub fn fit_rigid(from: &[Vec3], to: &[Vec3]) -> Dof6 {
    let n = from.len().min(to.len());
    if n == 0 {
        return Dof6::default();
    }
    let inv = 1.0 / n as f64;
    let pc = from.iter().take(n).fold(Vec3::ZERO, |a, b| a + *b) * inv;
    let qc = to.iter().take(n).fold(Vec3::ZERO, |a, b| a + *b) * inv;
    let mut h: M3 = [[0.0; 3]; 3];
    for (p, q) in from.iter().take(n).zip(to.iter().take(n)) {
        let a = *p - pc;
        let b = *q - qc;
        let av = [a.x, a.y, a.z];
        let bv = [b.x, b.y, b.z];
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] += bv[i] * av[j];
            }
        }
    }
    let r = nearest_rotation(&h);
    let rot = |v: Vec3| {
        Vec3::new(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z,
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z,
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z,
        )
    };
    // T̂(p) = R(p − c) + c + t with c = the sample centroid.
    let t = qc - pc;
    let e = euler_zyx(&r);
    let mut sq = 0.0;
    for (p, q) in from.iter().take(n).zip(to.iter().take(n)) {
        let fitted = rot(*p - pc) + pc + t;
        sq += (fitted - *q).dot(fitted - *q);
    }
    Dof6 {
        translation: t,
        rotation_deg: [e[0].to_degrees(), e[1].to_degrees(), e[2].to_degrees()],
        residual_mm: (sq * inv).sqrt(),
    }
}

/// Displacement statistics of a transform over a set of points - what the
/// per-structure readout uses, with the structure's own contour points.
pub fn stats_over_points(t: &Transform3, points: &[Vec3]) -> (VectorStats, Vec3) {
    let d: Vec<Vec3> = points.iter().map(|p| t.displacement(*p)).collect();
    let mean = if d.is_empty() {
        Vec3::ZERO
    } else {
        d.iter().fold(Vec3::ZERO, |a, b| a + *b) * (1.0 / d.len() as f64)
    };
    (VectorStats::of(&d), mean)
}

/// The lattice step that keeps the sample count near `target`.
fn analysis_step(dims: [usize; 3], target: usize) -> usize {
    let total = dims[0] * dims[1] * dims[2];
    if total <= target {
        return 1;
    }
    ((total as f64 / target as f64).cbrt().ceil() as usize).max(1)
}

/// Measure a transform over the fixed image, or over a region.
pub fn analyse(vol: &Volume, t: &Transform3, region: Option<&RegionMask>) -> RegAnalysis {
    // A few hundred thousand probes is plenty for millimetre statistics and
    // keeps this well under a second even on a 512³ study.
    const TARGET: usize = 120_000;
    let (lo, hi) = match region {
        Some(r) => r.bbox(),
        None => (
            [0, 0, 0],
            [vol.dims[0] - 1, vol.dims[1] - 1, vol.dims[2] - 1],
        ),
    };
    let span = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    let step = analysis_step(span, TARGET);
    let step_mm = step as f64 * vol.spacing.iter().sum::<f64>() / 3.0;

    let ks: Vec<usize> = (lo[2]..=hi[2]).step_by(step).collect();
    let rows: Vec<(Vec<Vec3>, Vec<Vec3>, Vec<f64>)> = ks
        .par_iter()
        .map(|&k| {
            let mut from = Vec::new();
            let mut to = Vec::new();
            let mut dets = Vec::new();
            let mut j = lo[1];
            while j <= hi[1] {
                let mut i = lo[0];
                while i <= hi[0] {
                    let p = vol.voxel_to_patient(i as f64, j as f64, k as f64);
                    if region.map(|r| r.contains(p)).unwrap_or(true) {
                        from.push(p);
                        to.push(t.map(p));
                        dets.push(jacobian_det(vol, t, p));
                    }
                    i += step;
                }
                j += step;
            }
            (from, to, dets)
        })
        .collect();

    let mut from = Vec::new();
    let mut to = Vec::new();
    let mut dets = Vec::new();
    for (f, q, d) in rows {
        from.extend(f);
        to.extend(q);
        dets.extend(d);
    }
    if from.is_empty() {
        return RegAnalysis::default();
    }
    let disp: Vec<Vec3> = from.iter().zip(&to).map(|(p, q)| *q - *p).collect();
    let mean_vector = disp.iter().fold(Vec3::ZERO, |a, b| a + *b) * (1.0 / disp.len() as f64);
    let folded = dets.iter().filter(|d| **d <= 0.0).count() as f64 / dets.len() as f64;
    let jac = JacobianStats {
        min: dets.iter().cloned().fold(f64::MAX, f64::min),
        max: dets.iter().cloned().fold(f64::MIN, f64::max),
        mean: dets.iter().sum::<f64>() / dets.len() as f64,
        folded,
    };
    RegAnalysis {
        dof: fit_rigid(&from, &to),
        displacement: VectorStats::of(&disp),
        mean_vector,
        jacobian: jac,
        samples: from.len(),
        step_mm,
        overlap: None,
    }
}

// ---------------------------------------------------------------------------
// Image overlap
// ---------------------------------------------------------------------------

/// The value at or above which a voxel of `vol` counts as tissue.
///
/// For CT that is a Hounsfield number and -300 HU is the usual place to cut:
/// well above lung parenchyma and air, well below anything solid, and it puts
/// the boundary on the patient's outline rather than on any one organ. An
/// image with no air in it is not CT (or is a cropped field), and no fixed
/// number means anything there, so a quarter of the way up its own value
/// range is used instead.
fn tissue_threshold(vol: &Volume) -> f64 {
    const AIR: i16 = -500;
    const CT_TISSUE_HU: f64 = -300.0;
    if vol.min_value <= AIR {
        return CT_TISSUE_HU;
    }
    let (lo, hi) = (vol.min_value as f64, vol.max_value as f64);
    lo + 0.25 * (hi - lo)
}

/// Dice of the fixed image's tissue mask against the moving image's, before
/// and after the transform.
///
/// Measured on the same lattice as [`analyse`] and by the same rule: walk the
/// fixed image, ask the moving image what is at the mapped point. A sample
/// that lands outside the moving image is dropped from both masks rather than
/// counted as background - it is missing data, not empty space, and counting
/// it as empty would reward a transform for pushing the images apart.
///
/// `None` when nothing overlaps at all, which is not a Dice of zero but an
/// absence of a measurement.
pub fn overlap(
    fixed: &Volume,
    moving: &Volume,
    t: &Transform3,
    region: Option<&RegionMask>,
) -> Option<OverlapStats> {
    const TARGET: usize = 120_000;
    if fixed.is_empty() || moving.is_empty() {
        return None;
    }
    let (lo, hi) = match region {
        Some(r) => r.bbox(),
        None => (
            [0, 0, 0],
            [fixed.dims[0] - 1, fixed.dims[1] - 1, fixed.dims[2] - 1],
        ),
    };
    let span = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    let step = analysis_step(span, TARGET);
    let thr_fixed = tissue_threshold(fixed);
    let thr_moving = tissue_threshold(moving);

    // (intersection, fixed count, moving count) for each of the two
    // mappings, and the number of samples that had a moving value at all.
    let ks: Vec<usize> = (lo[2]..=hi[2]).step_by(step).collect();
    let counts = ks
        .par_iter()
        .map(|&k| {
            let mut c = [0usize; 7];
            let mut j = lo[1];
            while j <= hi[1] {
                let mut i = lo[0];
                while i <= hi[0] {
                    let p = fixed.voxel_to_patient(i as f64, j as f64, k as f64);
                    i += step;
                    if !region.map(|r| r.contains(p)).unwrap_or(true) {
                        continue;
                    }
                    let Some(fv) = fixed.sample_patient(p) else {
                        continue;
                    };
                    let f = fv as f64 >= thr_fixed;
                    // Identity and transform are counted over the samples
                    // each of them can see, so neither is charged for the
                    // other's misses.
                    if let Some(mv) = moving.sample_patient(p) {
                        let m = mv as f64 >= thr_moving;
                        c[0] += usize::from(f && m);
                        c[1] += usize::from(f);
                        c[2] += usize::from(m);
                    }
                    if let Some(mv) = moving.sample_patient(t.map(p)) {
                        let m = mv as f64 >= thr_moving;
                        c[3] += usize::from(f && m);
                        c[4] += usize::from(f);
                        c[5] += usize::from(m);
                        c[6] += 1;
                    }
                }
                j += step;
            }
            c
        })
        .reduce(
            || [0usize; 7],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );

    let dice = |inter: usize, a: usize, b: usize| -> f64 {
        if a + b == 0 {
            0.0
        } else {
            2.0 * inter as f64 / (a + b) as f64
        }
    };
    if counts[6] == 0 || (counts[4] + counts[5]) == 0 {
        return None;
    }
    Some(OverlapStats {
        before: dice(counts[0], counts[1], counts[2]),
        after: dice(counts[3], counts[4], counts[5]),
        threshold: thr_fixed,
        samples: counts[6],
    })
}

/// Determinant of the deformation Jacobian at a point, by central
/// differences one voxel wide along the volume's own axes.
fn jacobian_det(vol: &Volume, t: &Transform3, p: Vec3) -> f64 {
    let axes = [vol.row_dir, vol.col_dir, vol.normal];
    let mut j: M3 = [[0.0; 3]; 3];
    for (b, axis) in axes.iter().enumerate() {
        let h = vol.spacing[b];
        let plus = t.map(p + *axis * h);
        let minus = t.map(p - *axis * h);
        let d = (plus - minus) * (1.0 / (2.0 * h));
        // ∂T/∂(patient axis b) expressed in patient components.
        let col = [d.x, d.y, d.z];
        let ax = [axis.x, axis.y, axis.z];
        for (a, row) in j.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v += col[a] * ax[c];
            }
        }
    }
    m3_det(&j)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol(dims: [usize; 3]) -> Volume {
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [2.0, 2.0, 2.5],
            origin: Vec3::new(-100.0, -100.0, -50.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 1,
        }
    }

    #[test]
    fn a_rigid_transform_is_recovered_exactly_by_the_six_dof_fit() {
        let v = vol([40, 40, 30]);
        let center = v.voxel_to_patient(20.0, 20.0, 15.0);
        let truth = [0.03_f64, -0.07, 0.11, 4.0, -3.0, 2.0];
        let t = Transform3::rigid_only(RigidTransform::new(truth, center));
        let a = analyse(&v, &t, None);
        for (got, want) in a
            .dof
            .rotation_deg
            .iter()
            .zip(truth[..3].iter().map(|r| r.to_degrees()))
        {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
        // Six numbers explain a rigid body completely.
        assert!(a.dof.residual_mm < 1e-6, "{}", a.dof.residual_mm);
        // …and a rigid body neither expands nor compresses anything.
        assert!((a.jacobian.min - 1.0).abs() < 1e-6);
        assert!((a.jacobian.max - 1.0).abs() < 1e-6);
        assert_eq!(a.jacobian.folded, 0.0);
        assert!(a.samples > 1000);
        assert!(!a.dof.line().is_empty());
        assert!(a.jacobian.line().contains("no folding"));
    }

    #[test]
    fn a_pure_translation_shows_up_as_the_mean_vector_and_nothing_else() {
        let v = vol([32, 32, 24]);
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, 5.0, 0.0, -1.0],
            Vec3::ZERO,
        ));
        let a = analyse(&v, &t, None);
        assert!((a.mean_vector - Vec3::new(5.0, 0.0, -1.0)).length() < 1e-9);
        let d = (26.0f64).sqrt();
        assert!((a.displacement.mean - d).abs() < 1e-9);
        assert!((a.displacement.max - d).abs() < 1e-9);
        assert!(a.displacement.line().contains("mean"));
        for r in a.dof.rotation_deg {
            assert!(r.abs() < 1e-6);
        }
    }

    #[test]
    fn a_uniform_expansion_shows_up_in_the_jacobian() {
        // A B-spline lattice whose coefficients grow linearly with x is a
        // uniform stretch along x: det J = 1 + rate.
        let v = vol([40, 40, 30]);
        let mut b = BSplineTransform::new(&v, 20.0);
        let [nx, ny, _] = b.grid_dims;
        let rate = 0.1;
        for k in 0..b.grid_dims[2] {
            for j in 0..ny {
                for i in 0..nx {
                    let o = 3 * (i + nx * (j + ny * k));
                    let x = b.grid_origin.x + i as f64 * b.spacing;
                    b.coeffs[o] = rate * x;
                }
            }
        }
        let t = Transform3 {
            rigid: RigidTransform::identity(Vec3::ZERO),
            warp: Warp::BSpline(b),
        };
        let a = analyse(&v, &t, None);
        assert!(
            (a.jacobian.mean - (1.0 + rate)).abs() < 0.02,
            "mean det {} vs {}",
            a.jacobian.mean,
            1.0 + rate
        );
        assert_eq!(a.jacobian.folded, 0.0);
        // A stretch is not a rigid body, so the fit leaves a residual.
        assert!(a.dof.residual_mm > 0.5, "{}", a.dof.residual_mm);
    }

    #[test]
    fn the_procrustes_fit_ignores_a_reflection() {
        // Points mirrored through a plane are not reachable by a rotation;
        // the fit must return a proper rotation anyway, never a reflection.
        let from: Vec<Vec3> = (0..20)
            .map(|i| Vec3::new(i as f64, (i * i % 7) as f64, (i % 5) as f64))
            .collect();
        let to: Vec<Vec3> = from.iter().map(|p| Vec3::new(p.x, p.y, -p.z)).collect();
        let d = fit_rigid(&from, &to);
        assert!(d.residual_mm > 0.0);
        assert!(d.rotation_deg.iter().all(|r| r.is_finite()));
    }

    /// A volume of air with one solid box in it, in voxel index ranges.
    fn boxed(dims: [usize; 3], lo: [usize; 3], hi: [usize; 3]) -> Volume {
        let mut v = vol(dims);
        v.data.fill(-1000);
        for k in lo[2]..hi[2] {
            for j in lo[1]..hi[1] {
                for i in lo[0]..hi[0] {
                    v.data[i + dims[0] * (j + dims[1] * k)] = 200;
                }
            }
        }
        v.min_value = -1000;
        v.max_value = 200;
        v
    }

    #[test]
    fn the_overlap_rises_when_the_transform_undoes_the_shift() {
        // The moving image holds the same box five voxels (10 mm) further
        // along x. Untransformed the two boxes only partly cover each
        // other; the transform that carries a fixed point to where the
        // moving image put it should land them on top of one another.
        let dims = [40, 40, 30];
        let fixed = boxed(dims, [10, 10, 8], [30, 30, 22]);
        let moving = boxed(dims, [15, 10, 8], [35, 30, 22]);
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            Vec3::ZERO,
        ));
        let ov = overlap(&fixed, &moving, &t, None).expect("both volumes have tissue");
        // CT-like data, so the cut is the fixed Hounsfield threshold.
        assert!((ov.threshold + 300.0).abs() < 1e-9, "{}", ov.threshold);
        assert!(ov.samples > 1000, "{}", ov.samples);
        assert!(ov.after > 0.98, "after {}", ov.after);
        assert!(ov.before < 0.8, "before {}", ov.before);
        assert!(ov.gain() > 0.2, "gain {}", ov.gain());
        assert!(ov.line().contains("Dice"));
    }

    #[test]
    fn the_overlap_of_an_image_with_itself_is_one() {
        let dims = [30, 30, 20];
        let v = boxed(dims, [8, 8, 5], [22, 22, 15]);
        let ov = overlap(
            &v,
            &v,
            &Transform3::rigid_only(RigidTransform::identity(Vec3::ZERO)),
            None,
        )
        .expect("tissue");
        assert!((ov.after - 1.0).abs() < 1e-9, "{}", ov.after);
        assert!((ov.before - 1.0).abs() < 1e-9, "{}", ov.before);
        assert_eq!(ov.gain(), 0.0);
    }

    #[test]
    fn an_empty_volume_has_no_overlap_to_measure() {
        let v = boxed([20, 20, 10], [4, 4, 2], [16, 16, 8]);
        let e = Volume::empty();
        assert!(overlap(
            &e,
            &v,
            &Transform3::rigid_only(RigidTransform::identity(Vec3::ZERO)),
            None
        )
        .is_none());
        assert!(overlap(
            &v,
            &e,
            &Transform3::rigid_only(RigidTransform::identity(Vec3::ZERO)),
            None
        )
        .is_none());
    }

    #[test]
    fn an_image_without_air_is_cut_a_quarter_of_the_way_up_its_own_range() {
        // MR and PET carry no Hounsfield numbers, so no fixed HU means
        // anything: the threshold follows the data instead.
        let mut v = vol([20, 20, 10]);
        v.data.fill(0);
        v.min_value = 0;
        v.max_value = 400;
        let ov = overlap(
            &v,
            &v,
            &Transform3::rigid_only(RigidTransform::identity(Vec3::ZERO)),
            None,
        );
        assert!(
            ov.is_none(),
            "nothing is above the cut, so there is no Dice"
        );
        let mut w = v.clone();
        w.data.fill(300);
        let ov = overlap(
            &w,
            &w,
            &Transform3::rigid_only(RigidTransform::identity(Vec3::ZERO)),
            None,
        )
        .expect("all tissue");
        assert!((ov.threshold - 100.0).abs() < 1e-9, "{}", ov.threshold);
        assert!((ov.after - 1.0).abs() < 1e-9);
    }

    #[test]
    fn statistics_over_a_point_set_match_the_transform() {
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, 3.0, 4.0, 0.0],
            Vec3::ZERO,
        ));
        let pts = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let (s, mean) = stats_over_points(&t, &pts);
        assert!((s.mean - 5.0).abs() < 1e-12);
        assert!((mean - Vec3::new(3.0, 4.0, 0.0)).length() < 1e-12);
        assert_eq!(VectorStats::of(&[]), VectorStats::default());
    }
}
