//! What a recovered transform actually *did*: six degrees of freedom,
//! displacement statistics and the Jacobian of the deformation.
//!
//! A registration result is otherwise two numbers (a metric before and
//! after) and a black box. Everything here is measured on the transform
//! itself, on a regular lattice over the fixed image or over the region a
//! local run was restricted to, so it applies to any method — the numbers
//! for a landmark warp are computed exactly the same way as for a B-spline.
//!
//! * **Six degrees of freedom.** Even a deformable result has a best-fitting
//!   rigid body, and it is usually the number a physicist wants first: how
//!   far did the patient move, and how far did they turn? It is the
//!   orthogonal Procrustes fit of the mapping over the sampled points —
//!   translation, three Euler angles in the same `Rz Ry Rx` convention as
//!   [`RigidTransform`], and the RMS residual, which says how much of the
//!   transform those six numbers do *not* explain (zero for a rigid result,
//!   by construction).
//! * **Displacements.** Magnitude statistics of `T(p) − p` in millimetres,
//!   plus the mean vector, which separates a systematic shift from
//!   scattered local motion.
//! * **Jacobian determinant.** `det(I + ∂d/∂x)` by central differences:
//!   above 1 the tissue expanded, below 1 it compressed, and at or below
//!   zero the deformation folded onto itself — which is not anatomy, it is
//!   an artefact, and the folded fraction is the standard way to say so.

use super::*;

/// Magnitude statistics of a set of vectors, in millimetres.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VectorStats {
    pub min: f64,
    pub mean: f64,
    /// 95th percentile — where the bulk of the motion ends.
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
            min: mags[0],
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
    /// Fraction of sample points where the determinant is ≤ 0 — the
    /// deformation folded, which is never anatomy.
    pub folded: f64,
}

impl JacobianStats {
    /// `det J 0.82 – 1.24 (mean 1.00), no folding`.
    pub fn line(&self) -> String {
        format!(
            "det J {:.2} – {:.2} (mean {:.2}), {}",
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

/// The rigid body that best explains a mapping.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dof6 {
    /// Translation of the fit, mm.
    pub translation: Vec3,
    /// Euler angles `[rx, ry, rz]` in degrees, `Rz Ry Rx` — the same
    /// convention as [`RigidTransform`].
    pub rotation_deg: [f64; 3],
    /// The point the rotation is taken about (the sample centroid).
    pub center: Vec3,
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
    /// Mean displacement vector (LPS), mm — a systematic shift shows here
    /// while the magnitude statistics cannot tell it from random motion.
    pub mean_vector: Vec3,
    pub jacobian: JacobianStats,
    /// Points the statistics were measured over.
    pub samples: usize,
    /// Lattice step of the sampling, mm.
    pub step_mm: f64,
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
            // Cofactor of (j, i) — the adjugate is the transposed cofactor
            // matrix, which is what the inverse needs.
            let (r0, r1) = ((j + 1) % 3, (j + 2) % 3);
            let (c0, c1) = ((i + 1) % 3, (i + 2) % 3);
            *v = (a[r0][c0] * a[r1][c1] - a[r0][c1] * a[r1][c0]) * inv;
        }
    }
    Some(r)
}

/// Nearest rotation to `a`, by Higham's polar-decomposition iteration
/// `R ← ½(R + R⁻ᵀ)` — quadratically convergent and free of any eigen
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
        center: pc,
        residual_mm: (sq * inv).sqrt(),
    }
}

/// Displacement statistics of a transform over a set of points — what the
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
    }
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
