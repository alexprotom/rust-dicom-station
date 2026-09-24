//! Geometric motion analysis over 4D phase series.
//!
//! Everything in here is arithmetic on masks and centroids - no UI, no
//! DICOM, no registration engine. The 4D motion tool feeds it the per-phase
//! masks its registrations produced; this module turns them into the
//! numbers a physicist reports: centroid trajectories, displacement
//! magnitudes, peak-to-peak amplitudes, target-reference drift, direction-
//! wise correlation with significance, ITV volumes, and structure-overlap
//! measures (Dice, HD95, mean surface distance).
//!
//! Conventions: coordinates are patient LPS in millimetres, so the
//! anatomical directions are x = right-left (RL), y = anterior-posterior
//! (AP), z = inferior-superior (SI). Volumes are cm³. Peak-to-peak of a
//! trajectory is the largest pairwise distance between its points - the
//! amplitude of the motion, independent of which phase is the reference.

use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::morphology;
pub use crate::registration::RunMetrics;
use crate::volume::Grid;

/// The anatomical direction names of the patient axes, in x/y/z order.
pub const AXES: [&str; 3] = ["RL", "AP", "SI"];

/// How a structure was carried across the phases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MotionModel {
    /// Rigid registration: translation + rotation, shape preserved. Local
    /// to the structure's neighbourhood when the run asks for it.
    Rigid,
    /// Rigid followed by a B-spline refinement: shape follows the anatomy.
    Deformable,
    /// No registration: the structure as it is contoured on every phase
    /// (a target propagated there earlier, or drawn per phase).
    Contoured,
}

impl MotionModel {
    pub fn label(self) -> &'static str {
        match self {
            MotionModel::Rigid => "rigid",
            MotionModel::Deformable => "deformable",
            MotionModel::Contoured => "as contoured",
        }
    }

    /// Whether the model needs a registration of the phases.
    pub fn registers(self) -> bool {
        !matches!(self, MotionModel::Contoured)
    }
}

/// Minimum, mean and maximum image value under a mask; `None` when the mask
/// is empty or does not fit the volume.
pub fn grey_stats(mask: &[u8], vol: &crate::volume::Volume) -> Option<[f64; 3]> {
    let n = vol.dims[0] * vol.dims[1] * vol.dims[2];
    if mask.len() != n {
        return None;
    }
    let (mut lo, mut hi, mut sum, mut count) = (f64::MAX, f64::MIN, 0.0f64, 0usize);
    for (&m, &v) in mask.iter().zip(vol.data.iter()) {
        if m == 0 {
            continue;
        }
        let v = v as f64;
        lo = lo.min(v);
        hi = hi.max(v);
        sum += v;
        count += 1;
    }
    (count > 0).then(|| [lo, sum / count as f64, hi])
}

/// Centroid of a mask in patient coordinates (mm); `None` for an empty mask.
///
/// The mean voxel index is mapped through the grid's affine, which is the
/// centroid exactly because the mapping is affine.
pub fn centroid_mm(mask: &[u8], grid: &Grid) -> Option<Vec3> {
    let [nx, ny, nz] = grid.dims;
    debug_assert_eq!(mask.len(), nx * ny * nz);
    let (mut si, mut sj, mut sk, mut n) = (0.0f64, 0.0f64, 0.0f64, 0u64);
    for k in 0..nz {
        for j in 0..ny {
            let row = k * nx * ny + j * nx;
            for (i, &v) in mask[row..row + nx].iter().enumerate() {
                if v != 0 {
                    si += i as f64;
                    sj += j as f64;
                    sk += k as f64;
                    n += 1;
                }
            }
        }
    }
    (n > 0).then(|| {
        let n = n as f64;
        grid.voxel_to_patient(si / n, sj / n, sk / n)
    })
}

/// Volume of a mask on `grid`, cm³.
pub fn volume_cm3(mask: &[u8], grid: &Grid) -> f64 {
    let vox = grid.voxel_cm3();
    crate::morphology::count_set(mask) as f64 * vox
}

/// Largest pairwise distance between the points - the peak-to-peak
/// amplitude of a trajectory.
pub fn peak_to_peak(points: &[Vec3]) -> f64 {
    let mut best = 0.0f64;
    for (i, a) in points.iter().enumerate() {
        for b in &points[i + 1..] {
            best = best.max((*a - *b).length());
        }
    }
    best
}

/// In-place union of masks (all on one grid).
pub fn union_into(acc: &mut [u8], mask: &[u8]) {
    debug_assert_eq!(acc.len(), mask.len());
    for (a, &m) in acc.iter_mut().zip(mask) {
        if m != 0 {
            *a = 1;
        }
    }
}

// ---- correlation -----------------------------------------------------------

/// Pearson correlation of two equally long series with its two-tailed
/// p-value (t-test with n − 2 degrees of freedom). `None` when fewer than
/// three points or either series is constant.
pub fn pearson(x: &[f64], y: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n != y.len() || n < 3 {
        return None;
    }
    let nf = n as f64;
    let (mx, my) = (x.iter().sum::<f64>() / nf, y.iter().sum::<f64>() / nf);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (&a, &b) in x.iter().zip(y) {
        let (dx, dy) = (a - mx, b - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    let r = (sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0);
    // t = r √(n−2) / √(1−r²); two-tailed p from the t-distribution equals
    // the regularized incomplete beta I_{ν/(ν+t²)}(ν/2, 1/2).
    let df = nf - 2.0;
    let p = if r.abs() >= 1.0 {
        0.0
    } else {
        let t2 = r * r * df / (1.0 - r * r);
        betai(df / 2.0, 0.5, df / (df + t2))
    };
    Some((r, p.clamp(0.0, 1.0)))
}

/// The wording the report uses for a synchrony level, from |r|.
pub fn synchrony_level(r: f64) -> &'static str {
    match r.abs() {
        v if v >= 0.9 => "very high",
        v if v >= 0.7 => "high",
        v if v >= 0.5 => "moderate",
        v if v >= 0.3 => "low",
        _ => "negligible",
    }
}

/// Significance stars for a p-value: `***` < 0.001, `**` < 0.01, `*` < 0.05.
pub fn stars(p: f64) -> &'static str {
    match p {
        v if v < 0.001 => "***",
        v if v < 0.01 => "**",
        v if v < 0.05 => "*",
        _ => "",
    }
}

/// Regularized incomplete beta function I_x(a, b), by the standard
/// continued fraction (Lentz's method).
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = ln_front.exp();
    // The continued fraction converges fast for x < (a+1)/(a+b+2); use the
    // symmetry I_x(a,b) = 1 − I_{1−x}(b,a) on the other side.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - betai_reflected(a, b, x)
    }
}

/// The reflected branch of [`betai`], kept out of line for clarity.
fn betai_reflected(a: f64, b: f64, x: f64) -> f64 {
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + b * (1.0 - x).ln() + a * x.ln();
    ln_front.exp() * betacf(b, a, 1.0 - x) / b
}

/// Continued fraction for the incomplete beta (Numerical Recipes `betacf`).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Lanczos approximation of ln Γ(x), x > 0.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.120_865_097_386_617_5e-2,
        -0.539_523_938_495_3e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000_000_000_190_015;
    for g in G {
        y += 1.0;
        ser += g / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

// ---- overlap ---------------------------------------------------------------

/// Overlap and surface-distance measures of two masks on one grid.
#[derive(Clone, Debug)]
pub struct Overlap {
    pub vol_a_cm3: f64,
    pub vol_b_cm3: f64,
    /// Dice similarity coefficient, 0-1.
    pub dice: f64,
    /// 95th-percentile symmetric Hausdorff distance, mm.
    pub hd95_mm: f64,
    /// Mean symmetric surface distance, mm.
    pub msd_mm: f64,
    /// Standard deviation of the same surface distances, mm.
    pub sd_mm: f64,
    /// Largest of them, mm - the plain Hausdorff distance.
    pub max_mm: f64,
    pub centroid_a: Option<Vec3>,
    pub centroid_b: Option<Vec3>,
}

impl Overlap {
    /// Distance between the two centroids, when both exist.
    pub fn centroid_shift(&self) -> Option<Vec3> {
        Some(self.centroid_b? - self.centroid_a?)
    }
}

/// The Dice coefficient of two masks on one lattice, `2|A ∩ B| / (|A| + |B|)`;
/// `None` when they differ in size or either is empty.
pub fn dice(a: &[u8], b: &[u8]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let (na, nb, nab) = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| {
            let (x, y) = (x != 0, y != 0);
            (x as u64, y as u64, (x && y) as u64)
        })
        .reduce(|| (0, 0, 0), |p, q| (p.0 + q.0, p.1 + q.1, p.2 + q.2));
    (na > 0 && nb > 0).then(|| 2.0 * nab as f64 / (na + nb) as f64)
}

/// Compare two masks on the same grid. `None` when either mask is empty.
pub fn overlap(a: &[u8], b: &[u8], grid: &Grid) -> Option<Overlap> {
    let n = grid.dims[0] * grid.dims[1] * grid.dims[2];
    if a.len() != n || b.len() != n {
        return None;
    }
    let (mut na, mut nb, mut nab) = (0u64, 0u64, 0u64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x != 0, y != 0);
        na += x as u64;
        nb += y as u64;
        nab += (x && y) as u64;
    }
    if na == 0 || nb == 0 {
        return None;
    }
    let dice = 2.0 * nab as f64 / (na + nb) as f64;

    // Surface distances: for every surface voxel of A, the distance to the
    // nearest voxel of B (via the exact EDT of B), and vice versa. HD95 is
    // the 95th percentile of both directed sets pooled; MSD their mean.
    let db = morphology::dist2_to_foreground(b, grid.dims, grid.spacing);
    let da = morphology::dist2_to_foreground(a, grid.dims, grid.spacing);
    let mut dists: Vec<f32> = Vec::new();
    collect_surface_distances(a, &db, grid.dims, &mut dists);
    collect_surface_distances(b, &da, grid.dims, &mut dists);
    if dists.is_empty() {
        return None;
    }
    let msd = dists.iter().map(|&d| d as f64).sum::<f64>() / dists.len() as f64;
    let var = dists
        .iter()
        .map(|&d| (d as f64 - msd) * (d as f64 - msd))
        .sum::<f64>()
        / dists.len() as f64;
    let max = dists.iter().fold(0.0f32, |a, &b| a.max(b)) as f64;
    let k = (((dists.len() - 1) as f64 * 0.95).round() as usize).min(dists.len() - 1);
    let (_, p95, _) = dists.select_nth_unstable_by(k, |x, y| x.total_cmp(y));
    let hd95 = *p95 as f64;

    let vox = grid.voxel_cm3();
    Some(Overlap {
        vol_a_cm3: na as f64 * vox,
        vol_b_cm3: nb as f64 * vox,
        dice,
        hd95_mm: hd95,
        msd_mm: msd,
        sd_mm: var.sqrt(),
        max_mm: max,
        centroid_a: centroid_mm(a, grid),
        centroid_b: centroid_mm(b, grid),
    })
}

// ---- the rigid offset between two structures ------------------------------

/// The surface voxels of a mask as patient points (mm), thinned to at most
/// `max_points` by taking every n-th so the sample stays spread over the
/// whole surface rather than over its first slices.
pub fn surface_points(mask: &[u8], grid: &Grid, max_points: usize) -> Vec<Vec3> {
    let [nx, ny, nz] = grid.dims;
    if mask.len() != nx * ny * nz {
        return Vec::new();
    }
    let mut all: Vec<[usize; 3]> = Vec::new();
    crate::morphology::for_each_surface_voxel(mask, grid.dims, |_, ijk| all.push(ijk));
    let step = if max_points == 0 || all.len() <= max_points {
        1
    } else {
        all.len().div_ceil(max_points)
    };
    all.iter()
        .step_by(step)
        .map(|&[i, j, k]| grid.voxel_to_patient(i as f64, j as f64, k as f64))
        .collect()
}

/// The rigid body that best carries structure A onto structure B, and how
/// much of the difference it fails to explain.
#[derive(Clone, Debug)]
pub struct SurfaceFit {
    /// Translation and rotation of the fit, about A's own centroid.
    pub dof: crate::registration::analysis::Dof6,
    /// Surface points used from each structure.
    pub points: [usize; 2],
    /// Distance from each of A's surface points to the nearest of B's,
    /// after the fit: mean, standard deviation, largest (mm).
    pub residual_mm: [f64; 3],
    /// The same three before the fit, so the fit can be seen to have done
    /// something.
    pub before_mm: [f64; 3],
}

/// Least-squares rigid offset between two structures, by closest-point
/// iteration over their surfaces.
///
/// The two surfaces carry no point correspondence, so one is invented and
/// then improved: pair every point of A with the nearest point of B, fit
/// the rigid body that best explains those pairs (orthogonal Procrustes,
/// [`crate::registration::analysis::fit_rigid`]), move A, pair again. The
/// fit is always taken from A's *original* points, so the iteration
/// refines one global transform instead of accumulating a chain of small
/// ones.
///
/// Both masks must live on `grid`. `None` when either surface is empty.
pub fn surface_fit(a: &[u8], b: &[u8], grid: &Grid) -> Option<SurfaceFit> {
    // A few thousand points per surface put the fit well inside a tenth of
    // a degree and keep the closest-point search under a second.
    const MAX_POINTS: usize = 3000;
    const ITERATIONS: usize = 25;
    let pa = surface_points(a, grid, MAX_POINTS);
    let pb = surface_points(b, grid, MAX_POINTS);
    if pa.is_empty() || pb.is_empty() {
        return None;
    }
    let nearest = |pts: &[Vec3]| -> Vec<Vec3> {
        pts.par_iter()
            .map(|p| {
                let mut best = pb[0];
                let mut bd = f64::MAX;
                for q in &pb {
                    let d = *q - *p;
                    let d2 = d.dot(d);
                    if d2 < bd {
                        bd = d2;
                        best = *q;
                    }
                }
                best
            })
            .collect()
    };
    let spread = |pts: &[Vec3], to: &[Vec3]| -> [f64; 3] {
        let d: Vec<f64> = pts
            .iter()
            .zip(to)
            .map(|(p, q)| (*q - *p).length())
            .collect();
        let n = d.len().max(1) as f64;
        let mean = d.iter().sum::<f64>() / n;
        let sd = (d.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
        let max = d.iter().fold(0.0f64, |x, &y| x.max(y));
        [mean, sd, max]
    };

    let before = spread(&pa, &nearest(&pa));
    let centre = pa.iter().fold(Vec3::ZERO, |x, y| x + *y) * (1.0 / pa.len() as f64);
    let mut cur = pa.clone();
    let mut dof = crate::registration::analysis::Dof6::default();
    let mut last = f64::MAX;
    for _ in 0..ITERATIONS {
        let matched = nearest(&cur);
        dof = crate::registration::analysis::fit_rigid(&pa, &matched);
        let t = crate::registration::RigidTransform::new(
            [
                dof.rotation_deg[0].to_radians(),
                dof.rotation_deg[1].to_radians(),
                dof.rotation_deg[2].to_radians(),
                dof.translation.x,
                dof.translation.y,
                dof.translation.z,
            ],
            centre,
        );
        cur = pa.iter().map(|p| t.map(*p)).collect();
        let rms = spread(&cur, &nearest(&cur))[0];
        if (last - rms).abs() < 1e-4 {
            break;
        }
        last = rms;
    }
    let residual = spread(&cur, &nearest(&cur));
    Some(SurfaceFit {
        dof,
        points: [pa.len(), pb.len()],
        residual_mm: residual,
        before_mm: before,
    })
}

/// Push the distance (mm) to the other structure for every surface voxel of
/// `mask` - a set voxel with an unset 6-neighbour (volume faces count as
/// boundary).
fn collect_surface_distances(
    mask: &[u8],
    dist2_other: &[f32],
    dims: [usize; 3],
    out: &mut Vec<f32>,
) {
    crate::morphology::for_each_surface_voxel(mask, dims, |c, _| {
        out.push(dist2_other[c].max(0.0).sqrt());
    });
}

// ---- the report ------------------------------------------------------------

/// One structure at one phase: where it is and how big it is.
#[derive(Clone, Debug)]
pub struct PhaseSample {
    /// Phase label, e.g. "0%".
    pub phase: String,
    pub centroid: Vec3,
    pub volume_cm3: f64,
    /// Minimum, mean and maximum image value inside the structure on this
    /// phase; `None` when the phase's own images were not at hand. A target
    /// whose mean grey level walks across the phases is a target the
    /// propagation put somewhere it does not belong.
    pub grey: Option<[f64; 3]>,
}

/// One structure carried across all phases with one model.
#[derive(Clone, Debug)]
pub struct Track {
    pub target: String,
    pub model: MotionModel,
    /// One sample per phase, in the group's phase order.
    pub samples: Vec<PhaseSample>,
    /// Index of the reference phase within `samples`.
    pub reference: usize,
}

impl Track {
    /// Displacement of each phase's centroid from the reference phase's.
    pub fn displacements(&self) -> Vec<Vec3> {
        let r = self.samples[self.reference].centroid;
        self.samples.iter().map(|s| s.centroid - r).collect()
    }

    /// 3D displacement magnitude per phase.
    pub fn magnitudes(&self) -> Vec<f64> {
        self.displacements().iter().map(|d| d.length()).collect()
    }

    /// Peak-to-peak amplitude of the centroid trajectory, mm.
    pub fn peak_to_peak(&self) -> f64 {
        let pts: Vec<Vec3> = self.samples.iter().map(|s| s.centroid).collect();
        peak_to_peak(&pts)
    }

    /// The target − reference difference vector per phase, when the other
    /// track covers the same phases.
    pub fn drift_against(&self, reference: &Track) -> Option<Vec<Vec3>> {
        (reference.samples.len() == self.samples.len()).then(|| {
            self.samples
                .iter()
                .zip(&reference.samples)
                .map(|(a, b)| a.centroid - b.centroid)
                .collect()
        })
    }
}

/// Correlation of target vs. reference motion along one patient axis.
#[derive(Clone, Debug)]
pub struct AxisCorrelation {
    /// "RL", "AP" or "SI".
    pub axis: &'static str,
    pub r: f64,
    pub p: f64,
}

impl AxisCorrelation {
    /// `SI  r = 0.951  p < 0.001 ***  (very high)`.
    pub fn line(&self) -> String {
        let p = if self.p < 0.001 {
            "p < 0.001".to_string()
        } else {
            format!("p = {:.3}", self.p)
        };
        format!(
            "{}  r = {:.3}  {} {}  ({})",
            self.axis,
            self.r,
            p,
            stars(self.p),
            synchrony_level(self.r)
        )
    }
}

/// Registration quality of one phase.
#[derive(Clone, Debug)]
pub struct RegQa {
    pub phase: String,
    pub model: MotionModel,
    /// What the fit looked at: `None` for the whole image, or the name of
    /// the structure whose neighbourhood a local rigid fit was confined to.
    pub region: Option<String>,
    /// The engine's own `MSD 9700 ▶ 1800 (900 iters, 20.1 s)` line.
    pub metric_line: String,
    /// The numbers of `metric_line`, for a table.
    pub metrics: Option<RunMetrics>,
    /// Fraction of sampled voxels with a non-positive Jacobian, percent.
    pub folding_pct: f64,
    /// 95th-percentile displacement magnitude of the deformation, mm.
    pub disp_p95_mm: f64,
    /// Dice of the tissue of the two images after this registration and
    /// before it: whether the phase now sits on the reference at all.
    /// `None` when the engine could not measure it.
    pub image_dice: Option<(f64, f64)>,
    /// Dice of each structure this model carried onto the phase against the
    /// phase's own contour of it, where the clinic drew one. Empty when
    /// nothing on this phase is contoured independently.
    pub struct_dice: Vec<(String, f64)>,
}

impl RegQa {
    /// The Dice line this row shows, `Dice 0.912 (was 0.604)`, or nothing
    /// when the overlap was not measured.
    pub fn dice_line(&self) -> Option<String> {
        self.image_dice
            .map(|(after, before)| format!("Dice {after:.3} (was {before:.3})"))
    }
}

/// One ITV the run produced.
#[derive(Clone, Debug)]
pub struct ItvResult {
    pub target: String,
    pub model: MotionModel,
    /// Uniform margin added on top of the union, mm.
    pub margin_mm: f64,
    pub volume_cm3: f64,
    /// Name of the segmentation the ITV was stored as.
    pub seg_name: String,
}

/// Everything one 4D motion run measured.
#[derive(Clone, Debug)]
pub struct MotionReport {
    /// `#1 A · 4DCT - Thorax (10 phases) · ref 0%`, the run's identity in
    /// the UI: numbered, so two runs on the same group stay distinguishable.
    pub run_name: String,
    /// "A" or "B" - which workspace the run analysed.
    pub slot_name: String,
    pub patient: String,
    /// Phase labels in order, e.g. `["0%", "10%", …]`.
    pub phases: Vec<String>,
    /// Label of the reference phase.
    pub reference: String,
    /// Target trajectories, one per (target, model).
    pub tracks: Vec<Track>,
    /// The reference structure's trajectories (e.g. the heart), when one
    /// was chosen - same phases, one per model.
    pub reference_tracks: Vec<Track>,
    /// Name of the reference structure, when one was chosen.
    pub reference_structure: Option<String>,
    /// Per target and model: correlation with the reference structure's
    /// motion along each patient axis.
    pub correlations: Vec<(String, MotionModel, Vec<AxisCorrelation>)>,
    pub qa: Vec<RegQa>,
    pub itvs: Vec<ItvResult>,
}

impl MotionReport {
    /// The reference track matching `model`, if any.
    pub fn reference_track(&self, model: MotionModel) -> Option<&Track> {
        self.reference_tracks.iter().find(|t| t.model == model)
    }

    /// The report as a CSV a spreadsheet opens as it is: a short header
    /// block, then sections with the **phases as columns** and **one row per
    /// method** (rigid, deformable, as contoured) under every quantity, so
    /// the methods sit on consecutive rows and read against each other
    /// phase by phase. Sections are separated by a blank line:
    ///
    /// * *Per-phase values* - centroid position, displacement from the
    ///   reference phase (per axis and |d|), volume, grey levels and the
    ///   offset to the reference structure, per structure;
    /// * *Summary* - one row per structure and method: peak-to-peak
    ///   amplitude, largest |d|, target-reference drift, correlation with
    ///   the reference structure, ITV;
    /// * *Registration quality* - per phase: image Dice after and before,
    ///   structure Dice against the phase's own contour, the metric, the
    ///   95th-percentile displacement and the folding rate.
    ///
    /// Text is plain ASCII where the UI uses typographic symbols (see
    /// [`plain_text`]); write it with [`csv_file_bytes`].
    pub fn csv(&self) -> String {
        let mut w = Csv::default();
        w.row(["Motion report", &self.run_name]);
        w.row(["Workspace", &self.slot_name]);
        w.row(["Reference phase", &self.reference]);
        if let Some(s) = &self.reference_structure {
            w.row(["Reference structure", s]);
        }
        w.row([
            "Coordinates",
            "patient LPS in mm: RL = x (right to left), AP = y (anterior to posterior), \
             SI = z (inferior to superior)",
        ]);
        w.row([
            "Displacement",
            "centroid on the phase minus centroid on the reference phase",
        ]);
        w.blank();
        self.csv_phase_values(&mut w);
        w.blank();
        self.csv_summary(&mut w);
        if !self.qa.is_empty() {
            w.blank();
            self.csv_registration_quality(&mut w);
        }
        w.text
    }

    /// Distinct target names in the order the run lists them.
    fn target_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = Vec::new();
        for t in &self.tracks {
            if !names.contains(&t.target.as_str()) {
                names.push(&t.target);
            }
        }
        names
    }

    /// The per-phase section: per structure and quantity, one row per
    /// method, one column per phase.
    fn csv_phase_values(&self, w: &mut Csv) {
        w.row(["Per-phase values"]);
        w.phase_header(["Structure", "Quantity", "Unit", "Method"], &self.phases);
        let mut groups: Vec<(String, Vec<&Track>, bool)> = self
            .target_names()
            .into_iter()
            .map(|n| {
                let tracks = self.tracks.iter().filter(|t| t.target == n).collect();
                (n.to_string(), tracks, true)
            })
            .collect();
        if !self.reference_tracks.is_empty() {
            let name = self
                .reference_structure
                .clone()
                .unwrap_or_else(|| self.reference_tracks[0].target.clone());
            groups.push((
                format!("{name} (reference)"),
                self.reference_tracks.iter().collect(),
                false,
            ));
        }
        type Value = fn(&PhaseSample, Vec3) -> Option<f64>;
        let quantities: [(&str, &str, usize, Value); 11] = [
            ("Centroid RL (x)", "mm", 3, |s, _| Some(s.centroid.x)),
            ("Centroid AP (y)", "mm", 3, |s, _| Some(s.centroid.y)),
            ("Centroid SI (z)", "mm", 3, |s, _| Some(s.centroid.z)),
            ("Displacement RL", "mm", 3, |_, d| Some(d.x)),
            ("Displacement AP", "mm", 3, |_, d| Some(d.y)),
            ("Displacement SI", "mm", 3, |_, d| Some(d.z)),
            ("Displacement |d|", "mm", 3, |_, d| Some(d.length())),
            ("Volume", "cm3", 3, |s, _| Some(s.volume_cm3)),
            ("Grey level min", "image value", 1, |s, _| {
                s.grey.map(|g| g[0])
            }),
            ("Grey level mean", "image value", 1, |s, _| {
                s.grey.map(|g| g[1])
            }),
            ("Grey level max", "image value", 1, |s, _| {
                s.grey.map(|g| g[2])
            }),
        ];
        for (name, tracks, is_target) in &groups {
            let has_grey = tracks
                .iter()
                .any(|t| t.samples.iter().any(|s| s.grey.is_some()));
            for (quantity, unit, decimals, value) in quantities {
                if quantity.starts_with("Grey") && !has_grey {
                    continue;
                }
                for t in tracks {
                    let disp = t.displacements();
                    let cells = self.phases.iter().map(|ph| {
                        t.samples
                            .iter()
                            .position(|s| &s.phase == ph)
                            .and_then(|i| value(&t.samples[i], disp[i]))
                            .map(|v| num(v, decimals))
                    });
                    w.phase_row([name.as_str(), quantity, unit, t.model.label()], cells);
                }
            }
            // Where the target sits against the reference structure.
            if !is_target {
                continue;
            }
            let reference = self.reference_structure.as_deref().unwrap_or("reference");
            for (ai, axis) in AXES.iter().enumerate() {
                let quantity = format!("Offset to {reference} {axis}");
                for t in tracks {
                    let Some(drift) = self
                        .reference_track(t.model)
                        .and_then(|rt| t.drift_against(rt).map(|d| (rt, d)))
                    else {
                        continue;
                    };
                    let (rt, drift) = drift;
                    let cells = self.phases.iter().map(|ph| {
                        let a = t.samples.iter().position(|s| &s.phase == ph)?;
                        // The drift pairs samples by index; only report it
                        // where both tracks are at the same phase.
                        let d = drift[a];
                        (rt.samples.get(a).map(|s| &s.phase) == Some(ph))
                            .then(|| num([d.x, d.y, d.z][ai], 3))
                    });
                    w.phase_row([name.as_str(), &quantity, "mm", t.model.label()], cells);
                }
            }
        }
    }

    /// One row per structure and method: the numbers that do not depend on
    /// the phase.
    fn csv_summary(&self, w: &mut Csv) {
        w.row(["Summary"]);
        let mut header = vec![
            "Structure".to_string(),
            "Method".into(),
            "Peak-to-peak (mm)".into(),
            "Largest |d| (mm)".into(),
        ];
        let has_ref = !self.reference_tracks.is_empty();
        if has_ref {
            header.push("Target-reference drift peak-to-peak (mm)".into());
            for axis in AXES {
                header.push(format!("Correlation r {axis}"));
                header.push(format!("Correlation p {axis}"));
            }
        }
        if !self.itvs.is_empty() {
            header.extend([
                "ITV volume (cm3)".to_string(),
                "ITV margin (mm)".into(),
                "ITV structure".into(),
            ]);
        }
        w.row(header.iter().map(String::as_str));
        let largest = |t: &Track| t.magnitudes().into_iter().fold(0.0, f64::max);
        for name in self.target_names() {
            for t in self.tracks.iter().filter(|t| t.target == name) {
                let mut cells = vec![
                    t.target.clone(),
                    t.model.label().to_string(),
                    num(t.peak_to_peak(), 3),
                    num(largest(t), 3),
                ];
                if has_ref {
                    cells.push(
                        self.reference_track(t.model)
                            .and_then(|rt| t.drift_against(rt))
                            .map(|d| num(peak_to_peak(&d), 3))
                            .unwrap_or_default(),
                    );
                    let corr = self
                        .correlations
                        .iter()
                        .find(|(n, m, _)| n == name && *m == t.model)
                        .map(|c| &c.2);
                    for axis in AXES {
                        let c = corr.and_then(|c| c.iter().find(|c| c.axis == axis));
                        cells.push(c.map(|c| num(c.r, 4)).unwrap_or_default());
                        cells.push(c.map(|c| p_value(c.p)).unwrap_or_default());
                    }
                }
                if !self.itvs.is_empty() {
                    match self
                        .itvs
                        .iter()
                        .find(|i| i.target == name && i.model == t.model)
                    {
                        Some(i) => cells.extend([
                            num(i.volume_cm3, 3),
                            num(i.margin_mm, 1),
                            i.seg_name.clone(),
                        ]),
                        None => cells.extend([String::new(), String::new(), String::new()]),
                    }
                }
                w.row(cells.iter().map(String::as_str));
            }
        }
        if let Some(t0) = self.reference_tracks.first() {
            let name = self.reference_structure.as_deref().unwrap_or(&t0.target);
            for t in &self.reference_tracks {
                let label = format!("{name} (reference)");
                w.row([
                    label.as_str(),
                    t.model.label(),
                    &num(t.peak_to_peak(), 3),
                    &num(largest(t), 3),
                ]);
            }
        }
    }

    /// Registration quality per phase: per quantity, one row per method
    /// (and, for a local rigid fit, per structure it was fitted on).
    fn csv_registration_quality(&self, w: &mut Csv) {
        w.row(["Registration quality"]);
        w.phase_header(["Quantity", "Unit", "Method", "Fit"], &self.phases);
        // The fits, in the order the run made them.
        let mut fits: Vec<(MotionModel, Option<&str>)> = Vec::new();
        for q in &self.qa {
            let key = (q.model, q.region.as_deref());
            if !fits.contains(&key) {
                fits.push(key);
            }
        }
        let fit_label = |r: Option<&str>| match r {
            None => "whole image".to_string(),
            Some(n) => format!("{n} neighbourhood"),
        };
        let tag = self
            .qa
            .iter()
            .find_map(|q| q.metrics.map(|m| m.tag))
            .unwrap_or("metric");
        type Value = fn(&RegQa) -> Option<String>;
        let quantities: [(&str, &str, Value); 8] = [
            ("Image Dice after", "", |q| {
                q.image_dice.map(|d| num(d.0, 4))
            }),
            ("Image Dice before", "", |q| {
                q.image_dice.map(|d| num(d.1, 4))
            }),
            ("Metric at start", "metric", |q| {
                q.metrics.map(|m| metric(m.initial))
            }),
            ("Metric at end", "metric", |q| {
                q.metrics.map(|m| metric(m.final_value))
            }),
            ("Iterations", "", |q| {
                q.metrics.map(|m| m.iterations.to_string())
            }),
            ("Time", "s", |q| q.metrics.map(|m| num(m.secs, 1))),
            ("Displacement p95", "mm", |q| Some(num(q.disp_p95_mm, 3))),
            ("Folding", "%", |q| Some(num(q.folding_pct, 3))),
        ];
        let cell =
            |fit: (MotionModel, Option<&str>), ph: &str, v: &dyn Fn(&RegQa) -> Option<String>| {
                self.qa
                    .iter()
                    .find(|q| q.phase == ph && q.model == fit.0 && q.region.as_deref() == fit.1)
                    .and_then(v)
            };
        for (quantity, unit, value) in quantities {
            let unit = if unit == "metric" { tag } else { unit };
            for &fit in &fits {
                let cells: Vec<Option<String>> =
                    self.phases.iter().map(|ph| cell(fit, ph, &value)).collect();
                if cells.iter().all(Option::is_none) {
                    continue;
                }
                w.phase_row([quantity, unit, fit.0.label(), &fit_label(fit.1)], cells);
            }
        }
        // The propagated structure against the phase's own contour of it.
        let mut scored: Vec<&str> = Vec::new();
        for q in &self.qa {
            for (n, _) in &q.struct_dice {
                if !scored.contains(&n.as_str()) {
                    scored.push(n);
                }
            }
        }
        for name in scored {
            let quantity = format!("Dice {name} vs contoured");
            for &fit in &fits {
                let v = |q: &RegQa| {
                    q.struct_dice
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, d)| num(*d, 4))
                };
                let cells: Vec<Option<String>> =
                    self.phases.iter().map(|ph| cell(fit, ph, &v)).collect();
                if cells.iter().all(Option::is_none) {
                    continue;
                }
                w.phase_row([&quantity, "", fit.0.label(), &fit_label(fit.1)], cells);
            }
        }
    }
}

/// Two runs side by side, one row per structure and method: the matched
/// peak-to-peak amplitudes and ITVs of [`MotionReport::csv`], with the
/// change from the first run to the second. Structures and methods are
/// matched by name, as the results window matches them.
pub fn comparison_csv(a: &MotionReport, b: &MotionReport) -> String {
    let mut w = Csv::default();
    w.row(["Comparison"]);
    w.row(["First run", &a.run_name]);
    w.row(["Second run", &b.run_name]);
    w.row([
        "Structure",
        "Method",
        "Peak-to-peak first (mm)",
        "Peak-to-peak second (mm)",
        "Peak-to-peak change (mm)",
        "ITV first (cm3)",
        "ITV second (cm3)",
        "ITV change (%)",
    ]);
    let mut keys: Vec<(&str, MotionModel)> = Vec::new();
    for t in a.tracks.iter().chain(&b.tracks) {
        let k = (t.target.as_str(), t.model);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    let track = |r: &'_ MotionReport, (n, m): (&str, MotionModel)| {
        r.tracks
            .iter()
            .find(|t| t.target == n && t.model == m)
            .map(Track::peak_to_peak)
    };
    let itv = |r: &'_ MotionReport, (n, m): (&str, MotionModel)| {
        r.itvs
            .iter()
            .find(|i| i.target == n && i.model == m)
            .map(|i| i.volume_cm3)
    };
    let opt = |v: Option<f64>, d: usize| v.map(|v| num(v, d)).unwrap_or_default();
    for k in keys {
        let (pa, pb) = (track(a, k), track(b, k));
        let (ia, ib) = (itv(a, k), itv(b, k));
        let pp_change = pa.zip(pb).map(|(a, b)| b - a);
        let itv_change = ia
            .zip(ib)
            .filter(|(a, _)| *a > 1e-9)
            .map(|(a, b)| 100.0 * (b - a) / a);
        w.row([
            k.0,
            k.1.label(),
            &opt(pa, 3),
            &opt(pb, 3),
            &opt(pp_change, 3),
            &opt(ia, 3),
            &opt(ib, 3),
            &opt(itv_change, 1),
        ]);
    }
    w.text
}

/// Plain-ASCII spelling of the typographic symbols the program uses on
/// screen (`·`, `▶`, `→`, `³`, `–`, ...), so a CSV reads the same in any
/// program - a spreadsheet that takes a file for Latin-1 would otherwise
/// show `Â·` for a `·`. Line breaks and other control characters become
/// spaces, so a name can never break a row. Letters outside ASCII (a
/// structure named in Cyrillic, say) are kept; [`csv_file_bytes`] then
/// marks the file as UTF-8.
pub fn plain_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        // Written as escapes: the glyph check reads this file, and several
        // of these are the very characters it keeps off the screen.
        let plain = match c {
            // Middle dot, bullet, en dash, em dash, minus sign.
            '\u{b7}' | '\u{2022}' | '\u{2013}' | '\u{2014}' | '\u{2212}' => "-",
            // Right-pointing triangle and arrows; the leftwards arrow.
            '\u{25b6}' | '\u{2192}' | '\u{27f6}' | '\u{279c}' => "->",
            '\u{2190}' => "<-",
            // Ellipsis; superscript three and two.
            '\u{2026}' => "...",
            '\u{b3}' => "3",
            '\u{b2}' => "2",
            // Micro sign and Greek mu; plus-minus; multiplication sign.
            '\u{b5}' | '\u{3bc}' => "u",
            '\u{b1}' => "+/-",
            '\u{d7}' => "x",
            // Almost equal, less-or-equal, greater-or-equal.
            '\u{2248}' => "~",
            '\u{2264}' => "<=",
            '\u{2265}' => ">=",
            // Curly quotes; no-break space.
            '\u{2018}' | '\u{2019}' => "'",
            '\u{201c}' | '\u{201d}' => "\"",
            '\u{a0}' => " ",
            c if c.is_control() => " ",
            c => {
                out.push(c);
                continue;
            }
        };
        out.push_str(plain);
    }
    out
}

/// The bytes of a CSV file: the text as it is when it is plain ASCII, and
/// behind a UTF-8 byte-order mark when it is not, which is what makes
/// Excel read the file as UTF-8 instead of the system code page.
pub fn csv_file_bytes(text: &str) -> Vec<u8> {
    if text.is_ascii() {
        return text.as_bytes().to_vec();
    }
    let mut out = Vec::with_capacity(text.len() + 3);
    out.extend_from_slice(b"\xEF\xBB\xBF");
    out.extend_from_slice(text.as_bytes());
    out
}

/// A number with a fixed count of decimals, never `-0.000`.
fn num(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return String::new();
    }
    let s = format!("{v:.decimals$}");
    if s.starts_with('-') && s[1..].chars().all(|c| c == '0' || c == '.') {
        s[1..].to_string()
    } else {
        s
    }
}

/// A registration metric: enough digits for an MSD in the thousands and a
/// mutual information below one alike.
fn metric(v: f64) -> String {
    let decimals = match v.abs() {
        a if a >= 100.0 => 1,
        a if a >= 10.0 => 2,
        _ => 4,
    };
    num(v, decimals)
}

/// A p-value: four decimals, or scientific notation below 0.0001, which a
/// spreadsheet reads as a number either way.
fn p_value(p: f64) -> String {
    if p > 0.0 && p < 1e-4 {
        format!("{p:.2e}")
    } else {
        num(p, 4)
    }
}

/// CSV text being built: cells made plain (see [`plain_text`]) and quoted
/// where they need it.
#[derive(Default)]
struct Csv {
    text: String,
}

impl Csv {
    fn cell(&mut self, v: &str) {
        let v = plain_text(v);
        if v.contains([',', '"']) || v.starts_with(' ') || v.ends_with(' ') {
            self.text.push('"');
            self.text.push_str(&v.replace('"', "\"\""));
            self.text.push('"');
        } else {
            self.text.push_str(&v);
        }
    }

    fn row<'a>(&mut self, cells: impl IntoIterator<Item = &'a str>) {
        for (i, c) in cells.into_iter().enumerate() {
            if i > 0 {
                self.text.push(',');
            }
            self.cell(c);
        }
        self.text.push('\n');
    }

    fn blank(&mut self) {
        self.text.push('\n');
    }

    /// The label columns, then one column per phase.
    fn phase_header(&mut self, labels: [&str; 4], phases: &[String]) {
        self.row(labels.into_iter().chain(phases.iter().map(String::as_str)));
    }

    /// The label cells, then one cell per phase (empty where there is no
    /// value).
    fn phase_row(&mut self, labels: [&str; 4], cells: impl IntoIterator<Item = Option<String>>) {
        let cells: Vec<String> = cells.into_iter().map(Option::unwrap_or_default).collect();
        self.row(labels.into_iter().chain(cells.iter().map(String::as_str)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(dims: [usize; 3], spacing: [f64; 3]) -> Grid {
        Grid {
            dims,
            spacing,
            origin: Vec3::new(10.0, -5.0, 20.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        }
    }

    fn ball(g: &Grid, c: [f64; 3], r: f64) -> Vec<u8> {
        let [nx, ny, nz] = g.dims;
        let mut m = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let d = [
                        (i as f64 - c[0]) * g.spacing[0],
                        (j as f64 - c[1]) * g.spacing[1],
                        (k as f64 - c[2]) * g.spacing[2],
                    ];
                    if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r * r {
                        m[k * nx * ny + j * nx + i] = 1;
                    }
                }
            }
        }
        m
    }

    /// An ellipsoid rotated by `deg` about the patient z axis and shifted.
    fn tilted_ellipsoid(g: &Grid, c: [f64; 3], half: [f64; 3], deg: f64) -> Vec<u8> {
        let [nx, ny, nz] = g.dims;
        let mut m = vec![0u8; nx * ny * nz];
        let (s, co) = deg.to_radians().sin_cos();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let d = [
                        (i as f64 - c[0]) * g.spacing[0],
                        (j as f64 - c[1]) * g.spacing[1],
                        (k as f64 - c[2]) * g.spacing[2],
                    ];
                    // Rotate the sample back into the ellipsoid's frame.
                    let x = co * d[0] + s * d[1];
                    let y = -s * d[0] + co * d[1];
                    let v =
                        (x / half[0]).powi(2) + (y / half[1]).powi(2) + (d[2] / half[2]).powi(2);
                    if v <= 1.0 {
                        m[k * nx * ny + j * nx + i] = 1;
                    }
                }
            }
        }
        m
    }

    #[test]
    fn the_surface_fit_recovers_a_shift_and_a_rotation() {
        let g = grid([64, 64, 32], [1.0, 1.0, 1.0]);
        let half = [18.0, 9.0, 12.0];
        let a = tilted_ellipsoid(&g, [32.0, 32.0, 16.0], half, 0.0);
        // The same body, turned 12° about z and moved 3 mm to the left.
        let b = tilted_ellipsoid(&g, [35.0, 32.0, 16.0], half, 12.0);
        let f = surface_fit(&a, &b, &g).expect("both surfaces exist");
        assert!(
            (f.dof.translation.x - 3.0).abs() < 0.6,
            "translation {:?}",
            f.dof.translation
        );
        assert!(f.dof.translation.y.abs() < 0.6 && f.dof.translation.z.abs() < 0.6);
        assert!(
            (f.dof.rotation_deg[2] - 12.0).abs() < 1.5,
            "rotation {:?}",
            f.dof.rotation_deg
        );
        // And the fit has to leave the surfaces closer than it found them.
        assert!(
            f.residual_mm[0] < 0.5 * f.before_mm[0],
            "residual {:?} vs {:?}",
            f.residual_mm,
            f.before_mm
        );
    }

    #[test]
    fn a_structure_fitted_to_itself_does_not_move() {
        let g = grid([48, 48, 24], [1.0, 1.0, 2.0]);
        let m = ball(&g, [24.0, 24.0, 12.0], 9.0);
        let f = surface_fit(&m, &m, &g).expect("non-empty");
        assert!(f.dof.translation.length() < 0.2);
        assert!(f.residual_mm[2] < 0.2);
        assert_eq!(f.points[0], f.points[1]);
    }

    #[test]
    fn centroid_is_the_sphere_center_in_patient_coordinates() {
        let g = grid([32, 32, 16], [1.0, 1.0, 2.0]);
        let m = ball(&g, [16.0, 12.0, 8.0], 6.0);
        let c = centroid_mm(&m, &g).unwrap();
        let want = g.voxel_to_patient(16.0, 12.0, 8.0);
        assert!((c - want).length() < 0.05, "{c:?} vs {want:?}");
        assert!(centroid_mm(&vec![0u8; 32 * 32 * 16], &g).is_none());
    }

    #[test]
    fn peak_to_peak_is_the_largest_pairwise_distance() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -3.0),
            Vec3::new(0.0, 4.0, 0.0),
        ];
        assert!((peak_to_peak(&pts) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_matches_reference_values() {
        // Perfect correlation.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let (r, p) = pearson(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
        assert!(p < 1e-9);
        // A reference pair (scipy.stats.pearsonr: r=0.919145, p=0.027262).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 3.0, 2.0, 5.0, 7.0];
        let (r, p) = pearson(&x, &y).unwrap();
        assert!((r - 0.919145).abs() < 1e-6, "r = {r}");
        assert!((p - 0.027262).abs() < 1e-4, "p = {p}");
        // Constant series: undefined.
        assert!(pearson(&x, &[1.0; 5]).is_none());
    }

    #[test]
    fn significance_wording_matches_the_manuscript_convention() {
        assert_eq!(stars(0.0005), "***");
        assert_eq!(stars(0.005), "**");
        assert_eq!(stars(0.04), "*");
        assert_eq!(stars(0.2), "");
        assert_eq!(synchrony_level(0.951), "very high");
        assert_eq!(synchrony_level(-0.839), "high");
        assert_eq!(synchrony_level(0.503), "moderate");
    }

    #[test]
    fn dice_counts_the_overlap_and_refuses_empty_or_mismatched_masks() {
        let a = [1u8, 1, 1, 0, 0, 0];
        let b = [0u8, 1, 1, 1, 0, 0];
        assert!((dice(&a, &b).unwrap() - 4.0 / 6.0).abs() < 1e-12);
        assert!((dice(&a, &a).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(dice(&a, &[0u8; 6]), None, "an empty mask has no Dice");
        assert_eq!(dice(&a, &b[..5]), None, "different lattices");
    }

    #[test]
    fn identical_masks_have_dice_one_and_zero_distances() {
        let g = grid([24, 24, 12], [1.0, 1.0, 2.0]);
        let m = ball(&g, [12.0, 12.0, 6.0], 5.0);
        let o = overlap(&m, &m, &g).unwrap();
        assert!((o.dice - 1.0).abs() < 1e-12);
        assert!(o.hd95_mm < 1e-6);
        assert!(o.msd_mm < 1e-6);
        assert_eq!(o.centroid_shift().unwrap().length(), 0.0);
    }

    #[test]
    fn a_pure_shift_shows_up_in_hd_and_centroid_shift() {
        let g = grid([40, 24, 12], [1.0, 1.0, 1.0]);
        let a = ball(&g, [12.0, 12.0, 6.0], 5.0);
        let b = ball(&g, [18.0, 12.0, 6.0], 5.0);
        let o = overlap(&a, &b, &g).unwrap();
        let shift = o.centroid_shift().unwrap();
        assert!((shift.x - 6.0).abs() < 0.05, "{shift:?}");
        assert!(o.dice < 0.6);
        // The farthest surface points are ~6 mm apart; HD95 a bit below.
        assert!(o.hd95_mm > 3.0 && o.hd95_mm <= 6.5, "hd95 = {}", o.hd95_mm);
        assert!((o.vol_a_cm3 - o.vol_b_cm3).abs() < 1e-9);
    }

    #[test]
    fn tracks_report_displacements_drift_and_peak_to_peak() {
        let mk = |offsets: &[f64]| Track {
            target: "TV".into(),
            model: MotionModel::Rigid,
            samples: offsets
                .iter()
                .enumerate()
                .map(|(i, &z)| PhaseSample {
                    phase: format!("{}%", i * 10),
                    centroid: Vec3::new(0.0, 0.0, z),
                    volume_cm3: 1.0,
                    grey: None,
                })
                .collect(),
            reference: 0,
        };
        let tv = mk(&[0.0, 3.0, 8.0, 3.0]);
        let heart = mk(&[0.0, 1.0, 4.0, 1.0]);
        assert_eq!(tv.magnitudes(), vec![0.0, 3.0, 8.0, 3.0]);
        assert!((tv.peak_to_peak() - 8.0).abs() < 1e-12);
        let drift = tv.drift_against(&heart).unwrap();
        let pp_drift = peak_to_peak(&drift);
        assert!((pp_drift - 4.0).abs() < 1e-12);
    }

    /// A track of `target` under `model` over the given phases, moving
    /// along x by `xs`.
    fn track(target: &str, model: MotionModel, phases: &[&str], xs: &[f64]) -> Track {
        Track {
            target: target.into(),
            model,
            samples: phases
                .iter()
                .zip(xs)
                .map(|(p, &x)| PhaseSample {
                    phase: (*p).into(),
                    centroid: Vec3::new(x, 10.0, -20.0),
                    volume_cm3: 2.0,
                    grey: Some([-980.0, -120.5, 340.0]),
                })
                .collect(),
            reference: 0,
        }
    }

    fn report(tracks: Vec<Track>, qa: Vec<RegQa>) -> MotionReport {
        MotionReport {
            run_name: "#1 A · 4DCT (3 phases) · ref 0%".into(),
            slot_name: "A".into(),
            patient: "P".into(),
            phases: vec!["0%".into(), "50%".into(), "80%".into()],
            reference: "0%".into(),
            tracks,
            reference_tracks: Vec::new(),
            reference_structure: None,
            correlations: Vec::new(),
            qa,
            itvs: Vec::new(),
        }
    }

    /// The cells of the first row whose leading cells are `lead`.
    fn row<'a>(csv: &'a str, lead: &[&str]) -> Vec<&'a str> {
        csv.lines()
            .map(|l| l.split(',').collect::<Vec<_>>())
            .find(|cells| cells.len() >= lead.len() && cells[..lead.len()] == *lead)
            .unwrap_or_else(|| panic!("no row {lead:?} in\n{csv}"))
    }

    #[test]
    fn the_csv_has_the_phases_as_columns_and_a_row_per_method() {
        let phases = ["0%", "50%", "80%"];
        let mut rep = report(
            vec![
                track("GTV", MotionModel::Rigid, &phases, &[0.0, 1.0, 0.5]),
                track("GTV", MotionModel::Deformable, &phases, &[0.0, 2.0, 1.0]),
                track("GTV", MotionModel::Contoured, &phases, &[0.0, 3.0, 1.5]),
            ],
            Vec::new(),
        );
        rep.reference_tracks = vec![track(
            "Heart",
            MotionModel::Deformable,
            &phases,
            &[0.0, 0.5, 0.25],
        )];
        rep.reference_structure = Some("Heart".into());
        rep.correlations = vec![(
            "GTV".into(),
            MotionModel::Deformable,
            vec![AxisCorrelation {
                axis: "RL",
                r: 0.9,
                p: 0.00001,
            }],
        )];
        rep.itvs = vec![ItvResult {
            target: "GTV".into(),
            model: MotionModel::Deformable,
            margin_mm: 0.0,
            volume_cm3: 12.5,
            seg_name: "ITV GTV".into(),
        }];
        let csv = rep.csv();

        // The phases head the columns after the four label columns.
        assert_eq!(
            row(&csv, &["Structure", "Quantity"]),
            [
                "Structure",
                "Quantity",
                "Unit",
                "Method",
                "0%",
                "50%",
                "80%"
            ]
        );
        // One row per method under each quantity, next to each other.
        let lines: Vec<&str> = csv.lines().collect();
        let at = |lead: &str| lines.iter().position(|l| l.starts_with(lead)).unwrap();
        let rigid = at("GTV,Centroid RL (x),mm,rigid,");
        assert!(lines[rigid + 1].starts_with("GTV,Centroid RL (x),mm,deformable,"));
        assert!(lines[rigid + 2].starts_with("GTV,Centroid RL (x),mm,as contoured,"));
        assert_eq!(
            row(&csv, &["GTV", "Centroid RL (x)", "mm", "as contoured"])[4..],
            ["0.000", "3.000", "1.500"]
        );
        assert_eq!(
            row(&csv, &["GTV", "Displacement |d|", "mm", "deformable"])[4..],
            ["0.000", "2.000", "1.000"]
        );
        assert_eq!(
            row(&csv, &["GTV", "Grey level mean", "image value", "rigid"])[4..],
            ["-120.5", "-120.5", "-120.5"]
        );
        // The target against the reference structure, where the model has one.
        assert_eq!(
            row(&csv, &["GTV", "Offset to Heart RL", "mm", "deformable"])[4..],
            ["0.000", "1.500", "0.750"]
        );
        assert!(!csv.contains("Offset to Heart RL,mm,rigid"));
        assert!(csv.contains("Heart (reference),Displacement |d|,mm,deformable,0.000,0.500,0.250"));

        // The summary: one row per method.
        let summary = row(&csv, &["GTV", "deformable"]);
        assert_eq!(summary[2], "2.000", "peak-to-peak");
        assert_eq!(summary[3], "2.000", "largest |d|");
        assert_eq!(summary[4], "1.500", "drift peak-to-peak");
        assert_eq!(summary[5], "0.9000", "r RL");
        assert_eq!(summary[6], "1.00e-5", "p RL");
        assert_eq!(summary[summary.len() - 3..], ["12.500", "0.0", "ITV GTV"]);
        assert_eq!(row(&csv, &["GTV", "rigid"])[2], "1.000");

        // Readable anywhere: our symbols spelled in ASCII, so no byte-order
        // mark is needed.
        assert!(csv.is_ascii(), "{csv}");
        assert!(csv.starts_with("Motion report,#1 A - 4DCT (3 phases) - ref 0%\n"));
        assert_eq!(csv_file_bytes(&csv), csv.as_bytes());
    }

    #[test]
    fn names_outside_ascii_are_kept_and_the_file_says_it_is_utf8() {
        let phases = ["0%", "50%", "80%"];
        // A name with a letter outside ASCII (an a-umlaut) and a comma.
        let name = "L\u{e4}sion, links";
        let rep = report(
            vec![track(name, MotionModel::Rigid, &phases, &[0.0, 1.0, 2.0])],
            Vec::new(),
        );
        let csv = rep.csv();
        assert!(
            csv.contains(&format!("\"{name}\",Volume,cm3,rigid,2.000,2.000,2.000")),
            "quoted, not mangled: {csv}"
        );
        let bytes = csv_file_bytes(&csv);
        assert_eq!(&bytes[..3], b"\xEF\xBB\xBF");
        assert_eq!(&bytes[3..], csv.as_bytes());
        assert_eq!(
            plain_text("MSD 9.7 ▶ 1.8 · 3 cm³\nnext"),
            "MSD 9.7 -> 1.8 - 3 cm3 next"
        );
    }

    #[test]
    fn the_registration_quality_rows_carry_the_dice() {
        let metrics = |initial, final_value| {
            Some(RunMetrics {
                tag: "MSD",
                initial,
                final_value,
                iterations: 900,
                secs: 20.1,
            })
        };
        let qa = vec![
            RegQa {
                phase: "50%".into(),
                model: MotionModel::Rigid,
                region: None,
                metric_line: "global: MSD 9700.0 ▶ 4100.0  (900 iters, 20.1 s)".into(),
                metrics: metrics(9700.0, 4100.0),
                folding_pct: 0.0,
                disp_p95_mm: 3.1,
                image_dice: Some((0.85, 0.604)),
                struct_dice: Vec::new(),
            },
            RegQa {
                phase: "50%".into(),
                model: MotionModel::Rigid,
                region: Some("GTV".into()),
                metric_line: "GTV: MSD 9700.0 ▶ 3900.0  (900 iters, 20.1 s)".into(),
                metrics: metrics(9700.0, 3900.0),
                folding_pct: 0.0,
                disp_p95_mm: 3.3,
                image_dice: Some((0.86, 0.604)),
                struct_dice: vec![("GTV".into(), 0.801)],
            },
            RegQa {
                phase: "50%".into(),
                model: MotionModel::Deformable,
                region: None,
                metric_line: "MSD 9700.0 ▶ 1800.0  (900 iters, 20.1 s)".into(),
                metrics: metrics(9700.0, 1800.0),
                folding_pct: 0.02,
                disp_p95_mm: 6.4,
                image_dice: Some((0.912, 0.604)),
                struct_dice: vec![("GTV".into(), 0.845)],
            },
            RegQa {
                phase: "80%".into(),
                model: MotionModel::Deformable,
                region: None,
                metric_line: "MSD 9700.0 ▶ 2000.0  (900 iters, 20.1 s)".into(),
                metrics: metrics(9700.0, 2000.0),
                folding_pct: 0.0,
                disp_p95_mm: 3.1,
                image_dice: None,
                struct_dice: Vec::new(),
            },
        ];
        assert_eq!(qa[2].dice_line().as_deref(), Some("Dice 0.912 (was 0.604)"));
        assert_eq!(qa[3].dice_line(), None);

        let csv = report(Vec::new(), qa).csv();
        assert_eq!(
            row(&csv, &["Quantity", "Unit"]),
            ["Quantity", "Unit", "Method", "Fit", "0%", "50%", "80%"]
        );
        // The reference phase is not registered, and the phase that could
        // not be measured leaves its cell empty.
        assert_eq!(
            row(&csv, &["Image Dice after", "", "deformable", "whole image"])[4..],
            ["", "0.9120", ""]
        );
        assert_eq!(
            row(
                &csv,
                &["Image Dice after", "", "rigid", "GTV neighbourhood"]
            )[4..],
            ["", "0.8600", ""]
        );
        assert_eq!(
            row(&csv, &["Metric at end", "MSD", "deformable", "whole image"])[4..],
            ["", "1800.0", "2000.0"]
        );
        assert_eq!(
            row(
                &csv,
                &["Dice GTV vs contoured", "", "deformable", "whole image"]
            )[4..],
            ["", "0.8450", ""]
        );
        // The methods of one quantity sit on consecutive rows.
        let lines: Vec<&str> = csv.lines().collect();
        let first = lines
            .iter()
            .position(|l| l.starts_with("Image Dice after,"))
            .unwrap();
        assert!(lines[first..first + 3]
            .iter()
            .all(|l| l.starts_with("Image Dice after,")));
        assert!(csv.is_ascii(), "{csv}");
    }

    #[test]
    fn a_comparison_matches_structures_and_methods_by_name() {
        let phases = ["0%", "50%", "80%"];
        let mut a = report(
            vec![track(
                "GTV",
                MotionModel::Deformable,
                &phases,
                &[0.0, 4.0, 2.0],
            )],
            Vec::new(),
        );
        let mut b = a.clone();
        b.run_name = "#2 B".into();
        b.tracks[0] = track("GTV", MotionModel::Deformable, &phases, &[0.0, 6.0, 3.0]);
        for (r, v) in [(&mut a, 10.0), (&mut b, 12.5)] {
            r.itvs.push(ItvResult {
                target: "GTV".into(),
                model: MotionModel::Deformable,
                margin_mm: 0.0,
                volume_cm3: v,
                seg_name: "ITV GTV".into(),
            });
        }
        let csv = comparison_csv(&a, &b);
        assert_eq!(
            row(&csv, &["GTV", "deformable"])[2..],
            ["4.000", "6.000", "2.000", "10.000", "12.500", "25.0"]
        );
    }
}
