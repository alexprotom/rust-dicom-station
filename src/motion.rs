//! Geometric motion analysis over 4D phase series.
//!
//! Everything in here is arithmetic on masks and centroids — no UI, no
//! DICOM, no registration engine. The 4D motion tool feeds it the per-phase
//! masks its registrations produced; this module turns them into the
//! numbers a physicist reports: centroid trajectories, displacement
//! magnitudes, peak-to-peak amplitudes, target–reference drift, direction-
//! wise correlation with significance, ITV volumes, and structure-overlap
//! measures (Dice, HD95, mean surface distance).
//!
//! Conventions: coordinates are patient LPS in millimetres, so the
//! anatomical directions are x = right–left (RL), y = anterior–posterior
//! (AP), z = inferior–superior (SI). Volumes are cm³. Peak-to-peak of a
//! trajectory is the largest pairwise distance between its points — the
//! amplitude of the motion, independent of which phase is the reference.

use crate::geometry::Vec3;
use crate::morphology;
use crate::volume::Grid;

/// The anatomical direction names of the patient axes, in x/y/z order.
pub const AXES: [&str; 3] = ["RL", "AP", "SI"];

/// How a structure was carried across the phases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MotionModel {
    /// Rigid registration: translation + rotation, shape preserved.
    Rigid,
    /// Rigid followed by a B-spline refinement: shape follows the anatomy.
    Deformable,
}

impl MotionModel {
    pub fn label(self) -> &'static str {
        match self {
            MotionModel::Rigid => "rigid",
            MotionModel::Deformable => "deformable",
        }
    }
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
    let vox = grid.spacing[0] * grid.spacing[1] * grid.spacing[2] / 1000.0;
    mask.iter().filter(|&&v| v != 0).count() as f64 * vox
}

/// Largest pairwise distance between the points — the peak-to-peak
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

/// Max − min of each component over the points, in x/y/z order.
pub fn axis_ranges(points: &[Vec3]) -> [f64; 3] {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for (a, v) in [p.x, p.y, p.z].into_iter().enumerate() {
            lo[a] = lo[a].min(v);
            hi[a] = hi[a].max(v);
        }
    }
    if points.is_empty() {
        return [0.0; 3];
    }
    [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]
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
    /// Dice similarity coefficient, 0–1.
    pub dice: f64,
    /// 95th-percentile symmetric Hausdorff distance, mm.
    pub hd95_mm: f64,
    /// Mean symmetric surface distance, mm.
    pub msd_mm: f64,
    pub centroid_a: Option<Vec3>,
    pub centroid_b: Option<Vec3>,
}

impl Overlap {
    /// Distance between the two centroids, when both exist.
    pub fn centroid_shift(&self) -> Option<Vec3> {
        Some(self.centroid_b? - self.centroid_a?)
    }
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
    let k = (((dists.len() - 1) as f64 * 0.95).round() as usize).min(dists.len() - 1);
    let (_, p95, _) = dists.select_nth_unstable_by(k, |x, y| x.total_cmp(y));
    let hd95 = *p95 as f64;

    let vox = grid.spacing[0] * grid.spacing[1] * grid.spacing[2] / 1000.0;
    Some(Overlap {
        vol_a_cm3: na as f64 * vox,
        vol_b_cm3: nb as f64 * vox,
        dice,
        hd95_mm: hd95,
        msd_mm: msd,
        centroid_a: centroid_mm(a, grid),
        centroid_b: centroid_mm(b, grid),
    })
}

/// Push the distance (mm) to the other structure for every surface voxel of
/// `mask` — a set voxel with an unset 6-neighbour (volume faces count as
/// boundary).
fn collect_surface_distances(
    mask: &[u8],
    dist2_other: &[f32],
    dims: [usize; 3],
    out: &mut Vec<f32>,
) {
    let [nx, ny, nz] = dims;
    let idx = |i: usize, j: usize, k: usize| k * nx * ny + j * nx + i;
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = idx(i, j, k);
                if mask[c] == 0 {
                    continue;
                }
                let surface = i == 0
                    || j == 0
                    || k == 0
                    || i == nx - 1
                    || j == ny - 1
                    || k == nz - 1
                    || mask[idx(i - 1, j, k)] == 0
                    || mask[idx(i + 1, j, k)] == 0
                    || mask[idx(i, j - 1, k)] == 0
                    || mask[idx(i, j + 1, k)] == 0
                    || mask[idx(i, j, k - 1)] == 0
                    || mask[idx(i, j, k + 1)] == 0;
                if surface {
                    out.push(dist2_other[c].max(0.0).sqrt());
                }
            }
        }
    }
}

// ---- the report ------------------------------------------------------------

/// One structure at one phase: where it is and how big it is.
#[derive(Clone, Debug)]
pub struct PhaseSample {
    /// Phase label, e.g. "0%".
    pub phase: String,
    pub centroid: Vec3,
    pub volume_cm3: f64,
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
    /// The engine's own `MSD 9700 ▶ 1800 (900 iters, 20.1 s)` line.
    pub metric_line: String,
    /// Fraction of sampled voxels with a non-positive Jacobian, percent.
    pub folding_pct: f64,
    /// 95th-percentile displacement magnitude of the deformation, mm.
    pub disp_p95_mm: f64,
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
    /// `A · 4D CT — Thorax (10 phases)`, the run's identity in the UI.
    pub run_name: String,
    /// "A" or "B" — which dataset the run analysed.
    pub slot_name: String,
    pub patient: String,
    pub group: String,
    /// Phase labels in order, e.g. `["0%", "10%", …]`.
    pub phases: Vec<String>,
    /// Label of the reference phase.
    pub reference: String,
    /// Target trajectories, one per (target, model).
    pub tracks: Vec<Track>,
    /// The reference structure's trajectories (e.g. the heart), when one
    /// was chosen — same phases, one per model.
    pub reference_tracks: Vec<Track>,
    /// Name of the reference structure, when one was chosen.
    pub reference_structure: Option<String>,
    /// Per target and model: correlation with the reference structure's
    /// motion along each patient axis.
    pub correlations: Vec<(String, MotionModel, Vec<AxisCorrelation>)>,
    pub qa: Vec<RegQa>,
    pub itvs: Vec<ItvResult>,
    pub notes: Vec<String>,
}

impl MotionReport {
    /// The reference track matching `model`, if any.
    pub fn reference_track(&self, model: MotionModel) -> Option<&Track> {
        self.reference_tracks.iter().find(|t| t.model == model)
    }

    /// Long-format CSV of the whole report: one `table` column tells the
    /// sections apart so the file loads into any tool as a single sheet.
    pub fn csv(&self) -> String {
        let mut s = String::new();
        s.push_str("table,run,target,model,phase,axis,value1,value2,value3,value4\n");
        let esc = |v: &str| {
            if v.contains(',') || v.contains('"') {
                format!("\"{}\"", v.replace('"', "\"\""))
            } else {
                v.to_string()
            }
        };
        let run = esc(&self.run_name);
        let mut track_rows = |t: &Track, kind: &str| {
            let disp = t.displacements();
            for (s_i, d) in t.samples.iter().zip(&disp) {
                s.push_str(&format!(
                    "{kind},{run},{},{},{},centroid,{:.3},{:.3},{:.3},{:.4}\n",
                    esc(&t.target),
                    t.model.label(),
                    esc(&s_i.phase),
                    s_i.centroid.x,
                    s_i.centroid.y,
                    s_i.centroid.z,
                    s_i.volume_cm3
                ));
                s.push_str(&format!(
                    "{kind},{run},{},{},{},displacement,{:.3},{:.3},{:.3},{:.3}\n",
                    esc(&t.target),
                    t.model.label(),
                    esc(&s_i.phase),
                    d.x,
                    d.y,
                    d.z,
                    d.length()
                ));
            }
            s.push_str(&format!(
                "peak_to_peak,{run},{},{},,,{:.3},,,\n",
                esc(&t.target),
                t.model.label(),
                t.peak_to_peak()
            ));
        };
        for t in &self.tracks {
            track_rows(t, "track");
        }
        for t in &self.reference_tracks {
            track_rows(t, "reference");
        }
        for (target, model, axes) in &self.correlations {
            for c in axes {
                s.push_str(&format!(
                    "correlation,{run},{},{},,{},{:.4},{:.6},,\n",
                    esc(target),
                    model.label(),
                    c.axis,
                    c.r,
                    c.p
                ));
            }
        }
        for q in &self.qa {
            s.push_str(&format!(
                "registration_qa,{run},,{},{},,{:.3},{:.3},,{}\n",
                q.model.label(),
                esc(&q.phase),
                q.folding_pct,
                q.disp_p95_mm,
                esc(&q.metric_line)
            ));
        }
        for itv in &self.itvs {
            s.push_str(&format!(
                "itv,{run},{},{},,,{:.3},{:.2},,{}\n",
                esc(&itv.target),
                itv.model.label(),
                itv.volume_cm3,
                itv.margin_mm,
                esc(&itv.seg_name)
            ));
        }
        s
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
        assert_eq!(axis_ranges(&pts), [1.0, 4.0, 3.0]);
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

    #[test]
    fn csv_has_one_row_per_sample_and_section() {
        let t = Track {
            target: "TV,1".into(),
            model: MotionModel::Deformable,
            samples: vec![
                PhaseSample {
                    phase: "0%".into(),
                    centroid: Vec3::ZERO,
                    volume_cm3: 2.0,
                },
                PhaseSample {
                    phase: "50%".into(),
                    centroid: Vec3::new(1.0, 0.0, 0.0),
                    volume_cm3: 2.1,
                },
            ],
            reference: 0,
        };
        let rep = MotionReport {
            run_name: "A · test".into(),
            slot_name: "A".into(),
            patient: "P".into(),
            group: "G".into(),
            phases: vec!["0%".into(), "50%".into()],
            reference: "0%".into(),
            tracks: vec![t],
            reference_tracks: Vec::new(),
            reference_structure: None,
            correlations: vec![(
                "TV,1".into(),
                MotionModel::Deformable,
                vec![AxisCorrelation {
                    axis: "SI",
                    r: 0.9,
                    p: 0.01,
                }],
            )],
            qa: Vec::new(),
            itvs: vec![ItvResult {
                target: "TV,1".into(),
                model: MotionModel::Deformable,
                margin_mm: 0.0,
                volume_cm3: 12.5,
                seg_name: "ITV TV,1".into(),
            }],
            notes: Vec::new(),
        };
        let csv = rep.csv();
        assert_eq!(csv.lines().count(), 1 + 2 * 2 + 1 + 1 + 1);
        assert!(csv.contains("\"TV,1\""), "comma-escaped target name");
        assert!(csv
            .lines()
            .all(|l| l.split(',').count() >= 10 || l.contains('"')));
    }
}
