//! Markers: fiducials implanted in the target (gold seeds, coils) or
//! landmarks picked by hand, found in the CT and on the radiographs, and the
//! pose that lays the one on the other.
//!
//! * **In the CT** ([`detect_3d`]): inside the clip box, voxels above a
//!   threshold (1800 HU), 6-connected components of 1 to 150 mm³, each one's
//!   centroid weighted by how far its voxels rise above the threshold. Or the
//!   study's points of interest, named by the caller.
//! * **On a radiograph** ([`detect_2d`]): a difference of Gaussians at the
//!   marker's projected size, local maxima above `k` times the noise (the
//!   median absolute deviation of the response), each refined to the centroid
//!   of its 5 × 5 neighbourhood.
//! * **Correspondence**: each marker's projection at the start pose, the
//!   candidates within the search radius, the one-to-one assignment with the
//!   smallest total distance; repeated with the pose found, so a start that
//!   was far off does not leave a marker paired with the wrong spot. A
//!   position picked by hand stands for that marker and view.
//! * **Pose** ([`register_markers`]): Levenberg-Marquardt on the projection
//!   residuals, in millimetres at the isocentre, through the analytic
//!   Jacobian of the projection and of the pose; Huber-weighted when asked.
//!   Six parameters from three or more markers; the three translations from
//!   fewer, or when asked.
//! * **Triangulation**: a marker seen in two views is placed where its two
//!   rays pass closest, brought into the CT by the pose and compared with
//!   where the CT has it: a marker that moved since the CT is reported, not
//!   averaged into the pose.

use std::time::Instant;

use anyhow::{bail, Result};

use super::metric2d::{Image2d, Metric2d};
use super::projective::{self, Dof2d3d, Request2d3d, Result2d3d, View2d, ViewOutcome};
use super::RigidTransform;
use crate::drr::HuMode;
use crate::geometry::Vec3;
use crate::iec::PatientFrame;
use crate::progress::ProgressSink;
use crate::projection::{render_posed, MuVolume, ProjectionGeometry};
use crate::volume::Volume;

/// A marker in the CT, patient coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Marker3d {
    pub name: String,
    pub position: Vec3,
}

/// What [`detect_3d`] looks for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detect3d {
    pub threshold_hu: f32,
    pub min_mm3: f64,
    pub max_mm3: f64,
    /// Min and max corners of the box to look in.
    pub clip: Option<(Vec3, Vec3)>,
}

impl Default for Detect3d {
    fn default() -> Self {
        Detect3d {
            threshold_hu: 1800.0,
            min_mm3: 1.0,
            max_mm3: 150.0,
            clip: None,
        }
    }
}

/// Markers in the CT, superior first.
pub fn detect_3d(vol: &Volume, p: &Detect3d) -> Vec<Marker3d> {
    let [nx, ny, nz] = vol.dims;
    let inside = |q: Vec3| match p.clip {
        None => true,
        Some((lo, hi)) => {
            q.x >= lo.x && q.x <= hi.x && q.y >= lo.y && q.y <= hi.y && q.z >= lo.z && q.z <= hi.z
        }
    };
    let threshold = p.threshold_hu.round().clamp(-32768.0, 32767.0) as i16;
    let mut mask = vec![0u8; nx * ny * nz];
    for (o, &v) in vol.data.iter().enumerate() {
        if v >= threshold {
            let (i, j, k) = (o % nx, (o / nx) % ny, o / (nx * ny));
            if inside(vol.voxel_to_patient(i as f64, j as f64, k as f64)) {
                mask[o] = 1;
            }
        }
    }
    let voxel_mm3 = vol.voxel_cm3() * 1000.0;
    let mut found: Vec<Vec3> = Vec::new();
    for c in crate::morphology::components(&mask, vol.dims) {
        let mm3 = c.voxels.len() as f64 * voxel_mm3;
        if mm3 < p.min_mm3 || mm3 > p.max_mm3 {
            continue;
        }
        let (mut w, mut s) = (0.0f64, Vec3::ZERO);
        for &o in &c.voxels {
            let o = o as usize;
            let (i, j, k) = (o % nx, (o / nx) % ny, o / (nx * ny));
            let wv = (vol.data[o] as f64 - threshold as f64 + 1.0).max(1.0);
            s = s + vol.voxel_to_patient(i as f64, j as f64, k as f64) * wv;
            w += wv;
        }
        found.push(s * (1.0 / w));
    }
    found.sort_by(|a, b| {
        b.z.total_cmp(&a.z)
            .then(a.x.total_cmp(&b.x))
            .then(a.y.total_cmp(&b.y))
    });
    found
        .into_iter()
        .enumerate()
        .map(|(k, position)| Marker3d {
            name: format!("Marker {}", k + 1),
            position,
        })
        .collect()
}

/// A spot on a radiograph that may be a marker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    /// Column and row, fractional.
    pub pixel: [f64; 2],
    /// The response over the noise.
    pub strength: f64,
}

fn blur(data: &[f32], dims: [usize; 2], sigma: f64) -> Vec<f32> {
    let [w, h] = dims;
    let r = (3.0 * sigma).ceil().max(1.0) as isize;
    let k: Vec<f32> = (-r..=r)
        .map(|d| (-(d * d) as f64 / (2.0 * sigma * sigma)).exp() as f32)
        .collect();
    let ksum: f32 = k.iter().sum();
    let at = |n: usize, i: isize| i.clamp(0, n as isize - 1) as usize;
    let mut tmp = vec![0.0f32; w * h];
    for j in 0..h {
        for i in 0..w {
            let mut s = 0.0;
            for (d, kd) in k.iter().enumerate() {
                s += kd * data[j * w + at(w, i as isize + d as isize - r)];
            }
            tmp[j * w + i] = s / ksum;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for j in 0..h {
        for i in 0..w {
            let mut s = 0.0;
            for (d, kd) in k.iter().enumerate() {
                s += kd * tmp[at(h, j as isize + d as isize - r) * w + i];
            }
            out[j * w + i] = s / ksum;
        }
    }
    out
}

/// Bright spots of about `sigma_px` on `image` (line integrals: a marker
/// attenuates, so it is bright), strongest first.
pub fn detect_2d(image: &Image2d, mask: Option<&[u8]>, sigma_px: f64, k: f64) -> Vec<Candidate> {
    let [w, h] = image.dims;
    let sigma = sigma_px.max(0.6);
    let a = blur(&image.data, image.dims, sigma);
    let b = blur(&image.data, image.dims, 1.6 * sigma);
    let r: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let counts = |i: usize| mask.is_none_or(|m| m[i] != 0);
    let mut vals: Vec<f32> = (0..r.len()).filter(|&i| counts(i)).map(|i| r[i]).collect();
    if vals.len() < 9 {
        return Vec::new();
    }
    let mid = vals.len() / 2;
    vals.select_nth_unstable_by(mid, f32::total_cmp);
    let median = vals[mid];
    let mut dev: Vec<f32> = vals.iter().map(|v| (v - median).abs()).collect();
    dev.select_nth_unstable_by(mid, f32::total_cmp);
    let noise = (1.4826 * dev[mid] as f64).max(1e-12);
    let reach = sigma.ceil().max(1.0) as isize;
    let mut out = Vec::new();
    for j in 0..h as isize {
        for i in 0..w as isize {
            let p = j as usize * w + i as usize;
            if !counts(p) {
                continue;
            }
            let v = r[p];
            let strength = (v - median) as f64 / noise;
            if strength < k {
                continue;
            }
            let mut peak = true;
            'n: for dj in -reach..=reach {
                for di in -reach..=reach {
                    let (x, y) = (i + di, j + dj);
                    if (di, dj) == (0, 0) || x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                        continue;
                    }
                    let q = r[y as usize * w + x as usize];
                    // Ties go to the first in raster order.
                    if q > v || (q == v && (dj < 0 || (dj == 0 && di < 0))) {
                        peak = false;
                        break 'n;
                    }
                }
            }
            if !peak {
                continue;
            }
            let (mut sw, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
            for dj in -2..=2isize {
                for di in -2..=2isize {
                    let (x, y) = (i + di, j + dj);
                    if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                        continue;
                    }
                    let q = (r[y as usize * w + x as usize] - median).max(0.0) as f64;
                    sw += q;
                    sx += q * x as f64;
                    sy += q * y as f64;
                }
            }
            let pixel = if sw > 0.0 {
                [sx / sw, sy / sw]
            } else {
                [i as f64, j as f64]
            };
            out.push(Candidate { pixel, strength });
        }
    }
    out.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    out.truncate(200);
    out
}

/// The one-to-one assignment of markers to candidates within `radius`
/// with the smallest total distance, a marker left without one costing
/// `radius`.
fn assign(predicted: &[Option<[f64; 2]>], cands: &[Candidate], radius: f64) -> Vec<Option<usize>> {
    let near: Vec<Vec<(usize, f64)>> = predicted
        .iter()
        .map(|p| match p {
            None => Vec::new(),
            Some(p) => {
                let mut v: Vec<(usize, f64)> = cands
                    .iter()
                    .enumerate()
                    .map(|(c, q)| {
                        (
                            c,
                            ((q.pixel[0] - p[0]).powi(2) + (q.pixel[1] - p[1]).powi(2)).sqrt(),
                        )
                    })
                    .filter(|(_, d)| *d <= radius)
                    .collect();
                v.sort_by(|a, b| a.1.total_cmp(&b.1));
                v.truncate(6);
                v
            }
        })
        .collect();
    struct Search<'a> {
        near: &'a [Vec<(usize, f64)>],
        radius: f64,
        used: Vec<bool>,
        cur: Vec<Option<usize>>,
        best: (f64, Vec<Option<usize>>),
    }
    fn go(s: &mut Search, m: usize, cost: f64) {
        if cost >= s.best.0 {
            return;
        }
        if m == s.near.len() {
            s.best = (cost, s.cur.clone());
            return;
        }
        for k in 0..s.near[m].len() {
            let (c, d) = s.near[m][k];
            if !s.used[c] {
                s.used[c] = true;
                s.cur[m] = Some(c);
                go(s, m + 1, cost + d);
                s.used[c] = false;
                s.cur[m] = None;
            }
        }
        go(s, m + 1, cost + s.radius);
    }
    let mut s = Search {
        near: &near,
        radius,
        used: vec![false; cands.len()],
        cur: vec![None; predicted.len()],
        best: (f64::INFINITY, vec![None; predicted.len()]),
    };
    go(&mut s, 0, 0.0);
    s.best.1
}

/// The common 2D shift that lays most of the predicted markers on
/// candidates: every candidate within `reach` of a marker proposes the
/// shift that would put that marker on it, and the proposal that brings the
/// most markers within `tol` of a candidate wins (ties to the smaller total
/// distance). The markers' layout is what is matched, so a start that is
/// far off in the plane still pairs each marker with its own spot. Zero for
/// fewer than two markers.
fn constellation_shift(
    predicted: &[Option<[f64; 2]>],
    cands: &[Candidate],
    reach: f64,
    tol: f64,
) -> [f64; 2] {
    let pts: Vec<[f64; 2]> = predicted.iter().flatten().copied().collect();
    if pts.len() < 2 {
        return [0.0, 0.0];
    }
    let dist = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
    let mut best = (0usize, f64::INFINITY, [0.0, 0.0]);
    for p in &pts {
        for c in cands {
            if dist(*p, c.pixel) > reach {
                continue;
            }
            let s = [c.pixel[0] - p[0], c.pixel[1] - p[1]];
            let (mut n, mut total) = (0usize, 0.0f64);
            for q in &pts {
                let at = [q[0] + s[0], q[1] + s[1]];
                if let Some(d) = cands
                    .iter()
                    .map(|c| dist(at, c.pixel))
                    .min_by(f64::total_cmp)
                {
                    if d <= tol {
                        n += 1;
                        total += d;
                    }
                }
            }
            let shift_len = (s[0] * s[0] + s[1] * s[1]).sqrt();
            if n > best.0 || (n == best.0 && n > 0 && total + 1e-3 * shift_len < best.1) {
                best = (n, total + 1e-3 * shift_len, s);
            }
        }
    }
    if best.0 >= 2 {
        best.2
    } else {
        [0.0, 0.0]
    }
}

/// One marker seen in one view.
#[derive(Clone, Copy, Debug)]
struct Observation {
    marker: usize,
    view: usize,
    pixel: [f64; 2],
    picked: bool,
}

/// Solve `a x = b` (n × n, partial pivoting); `None` when singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-300 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        let pivot = a[c].clone();
        for r in c + 1..n {
            let f = a[r][c] / pivot[c];
            for (ark, pk) in a[r][c..].iter_mut().zip(&pivot[c..]) {
                *ark -= f * pk;
            }
            b[r] -= f * b[c];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// The pose (planned frame → CT) putting each marker's projection on its
/// observation, by Levenberg-Marquardt from `start`. Residuals are in
/// millimetres at the isocentre (`mm_per_px` per view).
#[allow(clippy::too_many_arguments)]
fn solve_pose(
    markers: &[Marker3d],
    cams: &[ProjectionGeometry],
    mm_per_px: &[f64],
    obs: &[Observation],
    start: &RigidTransform,
    iso: Vec3,
    dof: Dof2d3d,
    scale: f64,
    robust: bool,
) -> (RigidTransform, usize) {
    let base = start.recentered(iso).params();
    let n = match dof {
        Dof2d3d::Six => 6,
        Dof2d3d::Translations => 3,
    };
    let pose = |x: &[f64]| {
        let mut p = base;
        match dof {
            Dof2d3d::Six => {
                for k in 0..3 {
                    p[k] += x[k] / scale;
                    p[k + 3] += x[k + 3];
                }
            }
            Dof2d3d::Translations => {
                for k in 0..3 {
                    p[k + 3] += x[k];
                }
            }
        }
        RigidTransform::new(p, iso)
    };
    // Residuals (2 per observation) and their Jacobian rows.
    let eval = |x: &[f64], jac: bool| -> (Vec<f64>, Vec<Vec<f64>>) {
        let t = pose(x);
        let mut r = Vec::with_capacity(2 * obs.len());
        let mut j = Vec::new();
        for o in obs {
            let m = markers[o.marker].position;
            let q = t.unmap(m);
            let s = mm_per_px[o.view];
            match cams[o.view].project_with_jacobian(q) {
                Some((px, [di, dj])) => {
                    r.push((px[0] - o.pixel[0]) * s);
                    r.push((px[1] - o.pixel[1]) * s);
                    if jac {
                        let dq = t.unmap_jacobian(m);
                        for g in [di, dj] {
                            let row: Vec<f64> = match dof {
                                Dof2d3d::Six => (0..6)
                                    .map(|k| g.dot(dq[k]) * s / if k < 3 { scale } else { 1.0 })
                                    .collect(),
                                Dof2d3d::Translations => (3..6).map(|k| g.dot(dq[k]) * s).collect(),
                            };
                            j.push(row);
                        }
                    }
                }
                None => {
                    r.push(1e3);
                    r.push(1e3);
                    if jac {
                        j.push(vec![0.0; n]);
                        j.push(vec![0.0; n]);
                    }
                }
            }
        }
        (r, j)
    };
    let weights = |r: &[f64]| -> Vec<f64> {
        (0..r.len() / 2)
            .flat_map(|k| {
                let e = (r[2 * k].powi(2) + r[2 * k + 1].powi(2)).sqrt();
                // Huber at one millimetre.
                let w = if robust && e > 1.0 { 1.0 / e } else { 1.0 };
                [w, w]
            })
            .collect()
    };
    // A faint pull towards the start: a direction the observations do not
    // constrain (the depth of one view) stays where it was rather than
    // wandering off, and nothing the observations do say is moved by it.
    const PRIOR: f64 = 1e-4;
    let cost = |r: &[f64], w: &[f64], x: &[f64]| {
        r.iter().zip(w).map(|(a, b)| a * a * b).sum::<f64>()
            + PRIOR * x.iter().map(|v| v * v).sum::<f64>()
    };
    let mut x = vec![0.0; n];
    let mut lambda = 1e-3;
    let mut iterations = 0;
    for _ in 0..100 {
        iterations += 1;
        let (r, j) = eval(&x, true);
        let w = weights(&r);
        let f0 = cost(&r, &w, &x);
        let mut jtj = vec![vec![0.0; n]; n];
        let mut jtr = vec![0.0; n];
        for (row, (ri, wi)) in j.iter().zip(r.iter().zip(&w)) {
            for a in 0..n {
                jtr[a] += wi * row[a] * ri;
                for b in 0..n {
                    jtj[a][b] += wi * row[a] * row[b];
                }
            }
        }
        for a in 0..n {
            jtj[a][a] += PRIOR;
            jtr[a] += PRIOR * x[a];
        }
        let mut improved = false;
        for _ in 0..12 {
            let mut a = jtj.clone();
            for k in 0..n {
                a[k][k] += lambda * jtj[k][k].max(1e-9);
            }
            let Some(dx) = solve(a, jtr.iter().map(|v| -v).collect()) else {
                lambda *= 10.0;
                continue;
            };
            let trial: Vec<f64> = x.iter().zip(&dx).map(|(a, b)| a + b).collect();
            let (rt, _) = eval(&trial, false);
            let f1 = cost(&rt, &weights(&rt), &trial);
            if f1 < f0 {
                let step = dx.iter().map(|v| v.abs()).fold(0.0, f64::max);
                x = trial;
                lambda = (lambda * 0.3).max(1e-9);
                improved = true;
                if step < 1e-7 || f0 - f1 < 1e-14 * (1.0 + f0) {
                    return (pose(&x), iterations);
                }
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    (pose(&x), iterations)
}

/// Where two or more rays pass closest: the least-squares point.
fn triangulate(rays: &[(Vec3, Vec3)]) -> Option<Vec3> {
    // Σ (I − d dᵀ) p = Σ (I − d dᵀ) a, d unit.
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    for &(o, d) in rays {
        let l = d.length();
        if l < 1e-12 {
            continue;
        }
        let d = d * (1.0 / l);
        let dv = [d.x, d.y, d.z];
        let ov = [o.x, o.y, o.z];
        for r in 0..3 {
            for c in 0..3 {
                let m = (r == c) as u8 as f64 - dv[r] * dv[c];
                a[r][c] += m;
                b[r] += m * ov[c];
            }
        }
    }
    let x = solve(a.iter().map(|r| r.to_vec()).collect(), b.to_vec())?;
    Some(Vec3::new(x[0], x[1], x[2]))
}

/// One marker in one view at the answer.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerResidual {
    pub marker: String,
    pub view: String,
    /// Observed and predicted, pixels of the view.
    pub observed: [f64; 2],
    pub predicted: [f64; 2],
    /// At the isocentre, mm.
    pub residual_mm: f64,
    /// Picked by hand rather than detected.
    pub picked: bool,
}

/// A marker seen in two or more views, placed by its rays.
#[derive(Clone, Debug, PartialEq)]
pub struct Triangulated {
    pub marker: String,
    /// Where the rays put it, in the CT's coordinates at the answer.
    pub position: Vec3,
    /// From where the CT has it, mm.
    pub offset_mm: f64,
    /// Further than the others explain: moved since the CT.
    pub migrated: bool,
}

/// What the markers said.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MarkerReport {
    pub residuals: Vec<MarkerResidual>,
    /// Over the markers that did not move.
    pub rms_mm: f64,
    pub triangulated: Vec<Triangulated>,
    /// The parameters the markers could determine.
    pub dof: Dof2d3d,
    /// Markers not found in any view.
    pub missing: Vec<String>,
}

impl MarkerReport {
    pub fn report(&self) -> String {
        let mut s = format!(
            "Markers: RMS {:.2} mm, {} observations, {}\n",
            self.rms_mm,
            self.residuals.len(),
            match self.dof {
                Dof2d3d::Six => "six parameters",
                Dof2d3d::Translations => "translations only",
            }
        );
        for r in &self.residuals {
            s.push_str(&format!(
                "  {} in {}: {:.2} mm{}\n",
                r.marker,
                r.view,
                r.residual_mm,
                if r.picked { " (picked)" } else { "" }
            ));
        }
        for t in &self.triangulated {
            s.push_str(&format!(
                "  {} placed {:.2} mm from the CT's{}\n",
                t.marker,
                t.offset_mm,
                if t.migrated { ", moved" } else { "" }
            ));
        }
        if !self.missing.is_empty() {
            s.push_str(&format!("  Not found: {}\n", self.missing.join(", ")));
        }
        s
    }
}

/// A marker registration.
#[derive(Clone, Debug)]
pub struct MarkerRequest {
    pub markers: Vec<Marker3d>,
    pub views: Vec<View2d>,
    /// Per view, per marker: a position picked by hand (pixels of the
    /// view's image), which stands for that marker in that view.
    pub picks: Vec<Vec<Option<[f64; 2]>>>,
    pub isocenter: Vec3,
    pub patient: PatientFrame,
    pub start: Option<RigidTransform>,
    /// Asked for; three translations are all fewer than three markers give.
    pub dof: Dof2d3d,
    /// Around each marker's projection at the start, at the isocentre, mm.
    pub search_mm: f64,
    /// How far the markers' layout as a whole may lie from where the start
    /// puts it, at the isocentre, mm.
    pub shift_mm: f64,
    /// The markers' diameter, mm.
    pub marker_mm: f64,
    /// Detection threshold, times the noise.
    pub k: f64,
    pub robust: bool,
    /// For the polarity of the views and the DRRs of the result.
    pub hu: HuMode,
    pub threshold_hu: f32,
    pub max_pixels: usize,
    /// Rotations in millimetre-like units: the half diagonal of the region
    /// the markers are in, mm per radian.
    pub scale_mm: f64,
}

impl MarkerRequest {
    pub fn new(markers: Vec<Marker3d>, views: Vec<View2d>, isocenter: Vec3) -> Self {
        MarkerRequest {
            markers,
            views,
            picks: Vec::new(),
            isocenter,
            patient: PatientFrame::default(),
            start: None,
            dof: Dof2d3d::Six,
            search_mm: 15.0,
            shift_mm: 40.0,
            marker_mm: 1.5,
            k: 4.0,
            robust: true,
            hu: HuMode::Water,
            threshold_hu: -800.0,
            max_pixels: 512,
            scale_mm: 100.0,
        }
    }
}

/// The pose from the markers, as a 2D/3D result (NCC of the images before
/// and after for reference) with its [`MarkerReport`].
pub fn register_markers(
    vol: &Volume,
    req: &MarkerRequest,
    sink: &dyn ProgressSink,
) -> Result<Result2d3d> {
    let started = Instant::now();
    if req.markers.is_empty() {
        bail!("no markers in the CT");
    }
    if req.views.is_empty() {
        bail!("no radiograph");
    }
    let iso = req.isocenter;
    let start = req
        .start
        .as_ref()
        .map(|s| s.recentered(iso))
        .unwrap_or_else(|| RigidTransform::identity(iso));
    sink.report(0.05, "preparing the CT");
    let mu = MuVolume::prepare(vol, req.hu, req.threshold_hu, None);

    // Each view as line integrals, bright where a marker is.
    let mut work = Vec::with_capacity(req.views.len());
    for v in &req.views {
        let (mut img, mask, cam) = projective::working_view(v, req.max_pixels, None)?;
        let inv = projective::inverted(&mu, &img, &mask, &cam, &start, sink)?;
        if inv {
            img.negate();
        }
        // Picks are in the pixels of the image as given.
        let f = v.image.dims[0] as f64 / img.dims[0] as f64;
        let picks: Vec<Option<[f64; 2]>> = (0..req.markers.len())
            .map(|m| {
                req.picks
                    .get(work.len())
                    .and_then(|p| p.get(m).copied().flatten())
                    .map(|p| [(p[0] + 0.5) / f - 0.5, (p[1] + 0.5) / f - 0.5])
            })
            .collect();
        work.push((img, mask, cam, inv, picks));
    }
    let cams: Vec<ProjectionGeometry> = work.iter().map(|w| w.2).collect();
    let mm_per_px: Vec<f64> = cams.iter().map(|c| projective::pixel_at(c, iso)).collect();

    sink.report(0.2, "finding the markers on the images");
    let cands: Vec<Vec<Candidate>> = work
        .iter()
        .zip(&mm_per_px)
        .map(|((img, mask, ..), px)| {
            let sigma = (0.5 * req.marker_mm / px / std::f64::consts::SQRT_2).max(0.6);
            detect_2d(img, Some(mask), sigma, req.k)
        })
        .collect();

    // Assign, solve, assign again with the pose found.
    let mut pose = start.clone();
    let mut obs: Vec<Observation> = Vec::new();
    let mut dof = req.dof;
    let mut iterations = 0;
    for round in 0..4 {
        let mut next = Vec::new();
        for (v, (_, _, cam, _, picks)) in work.iter().enumerate() {
            let predicted: Vec<Option<[f64; 2]>> = req
                .markers
                .iter()
                .map(|m| cam.project(pose.unmap(m.position)))
                .collect();
            let radius = req.search_mm / mm_per_px[v];
            let mut free: Vec<Option<[f64; 2]>> = predicted
                .iter()
                .zip(picks)
                .map(|(p, pick)| if pick.is_some() { None } else { *p })
                .collect();
            if round == 0 {
                let s = constellation_shift(
                    &free,
                    &cands[v],
                    req.shift_mm.max(req.search_mm) / mm_per_px[v],
                    (req.search_mm / 3.0).max(2.0) / mm_per_px[v],
                );
                for p in free.iter_mut().flatten() {
                    p[0] += s[0];
                    p[1] += s[1];
                }
            }
            let a = assign(&free, &cands[v], radius);
            for (m, pick) in picks.iter().enumerate() {
                if let Some(p) = pick {
                    next.push(Observation {
                        marker: m,
                        view: v,
                        pixel: *p,
                        picked: true,
                    });
                } else if let Some(c) = a[m] {
                    next.push(Observation {
                        marker: m,
                        view: v,
                        pixel: cands[v][c].pixel,
                        picked: false,
                    });
                }
            }
        }
        let distinct = {
            let mut seen: Vec<usize> = next.iter().map(|o| o.marker).collect();
            seen.sort_unstable();
            seen.dedup();
            seen.len()
        };
        if next.is_empty() {
            bail!(
                "no marker found on any image within {:.0} mm of where the plan puts it",
                req.search_mm
            );
        }
        dof = if distinct >= 3 {
            req.dof
        } else {
            Dof2d3d::Translations
        };
        let unchanged = round > 0
            && next.len() == obs.len()
            && next
                .iter()
                .zip(&obs)
                .all(|(a, b)| a.marker == b.marker && a.view == b.view && a.pixel == b.pixel);
        obs = next;
        let (p, it) = solve_pose(
            &req.markers,
            &cams,
            &mm_per_px,
            &obs,
            &start,
            iso,
            dof,
            req.scale_mm.max(10.0),
            req.robust,
        );
        pose = p;
        iterations += it;
        if unchanged {
            break;
        }
    }

    // Where each marker seen twice is, by its rays, at a pose.
    let place = |pose: &RigidTransform| -> Vec<(usize, Vec3)> {
        (0..req.markers.len())
            .filter_map(|m| {
                let rays: Vec<(Vec3, Vec3)> = obs
                    .iter()
                    .filter(|o| o.marker == m)
                    .map(|o| {
                        let c = &cams[o.view];
                        (c.source, c.pixel(o.pixel[0], o.pixel[1]) - c.source)
                    })
                    .collect();
                (rays.len() >= 2)
                    .then(|| triangulate(&rays))
                    .flatten()
                    .map(|p| (m, pose.map(p)))
            })
            .collect()
    };
    // Moved: further than 2 mm and three times what the others show. A
    // marker that moved is left out and the pose solved again from the
    // others, when they are enough to solve it.
    let placed = place(&pose);
    let mut offs: Vec<f64> = placed
        .iter()
        .map(|(m, p)| (*p - req.markers[*m].position).length())
        .collect();
    offs.sort_by(f64::total_cmp);
    let typical = offs.get(offs.len() / 2).copied().unwrap_or(0.0);
    let moved: Vec<usize> = placed
        .iter()
        .filter(|(m, p)| (*p - req.markers[*m].position).length() > 2.0f64.max(3.0 * typical))
        .map(|(m, _)| *m)
        .collect();
    if !moved.is_empty() {
        let kept: Vec<Observation> = obs
            .iter()
            .filter(|o| !moved.contains(&o.marker))
            .copied()
            .collect();
        let mut distinct: Vec<usize> = kept.iter().map(|o| o.marker).collect();
        distinct.sort_unstable();
        distinct.dedup();
        if !kept.is_empty() {
            dof = if distinct.len() >= 3 {
                req.dof
            } else {
                Dof2d3d::Translations
            };
            let (p, it) = solve_pose(
                &req.markers,
                &cams,
                &mm_per_px,
                &kept,
                &start,
                iso,
                dof,
                req.scale_mm.max(10.0),
                req.robust,
            );
            pose = p;
            iterations += it;
        }
    }

    // The report.
    let mut residuals = Vec::new();
    let mut sq = 0.0;
    let mut counted = 0usize;
    for o in &obs {
        let pred = cams[o.view]
            .project(pose.unmap(req.markers[o.marker].position))
            .unwrap_or([f64::NAN; 2]);
        let e = ((pred[0] - o.pixel[0]).powi(2) + (pred[1] - o.pixel[1]).powi(2)).sqrt()
            * mm_per_px[o.view];
        if !moved.contains(&o.marker) {
            sq += e * e;
            counted += 1;
        }
        let f = req.views[o.view].image.dims[0] as f64 / work[o.view].0.dims[0] as f64;
        let up = |p: [f64; 2]| [(p[0] + 0.5) * f - 0.5, (p[1] + 0.5) * f - 0.5];
        residuals.push(MarkerResidual {
            marker: req.markers[o.marker].name.clone(),
            view: req.views[o.view].label.clone(),
            observed: up(o.pixel),
            predicted: up(pred),
            residual_mm: e,
            picked: o.picked,
        });
    }
    let rms = (sq / counted.max(1) as f64).sqrt();
    let triangulated: Vec<Triangulated> = place(&pose)
        .into_iter()
        .map(|(m, p)| Triangulated {
            marker: req.markers[m].name.clone(),
            position: p,
            offset_mm: (p - req.markers[m].position).length(),
            migrated: moved.contains(&m),
        })
        .collect();
    let missing: Vec<String> = req
        .markers
        .iter()
        .enumerate()
        .filter(|(m, _)| !obs.iter().any(|o| o.marker == *m))
        .map(|(_, mk)| mk.name.clone())
        .collect();

    // The views at the answer, for the overlay and the numbers.
    sink.report(0.8, "rendering the DRRs");
    let mut views = Vec::with_capacity(work.len());
    for (v, (img, mask, cam, inv, _)) in work.iter().enumerate() {
        let step = mu
            .axes
            .iter()
            .map(|a| a.length())
            .fold(f64::INFINITY, f64::min)
            .max(0.5);
        let before = render_posed(&mu, &cam.transformed(&start), step, sink)?;
        let after = render_posed(&mu, &cam.transformed(&pose), step, sink)?;
        views.push(ViewOutcome {
            label: req.views[v].label.clone(),
            before: Metric2d::Ncc.similarity(&img.data, &before, img.dims, Some(mask), 1),
            after: Metric2d::Ncc.similarity(&img.data, &after, img.dims, Some(mask), 1),
            inverted: *inv,
            image: img.clone(),
            drr: Image2d::new(after, img.dims),
            camera: *cam,
            mask: mask.clone(),
        });
    }
    let (couch, displacement) = projective::readout(&pose, &req.patient);
    let warning = triangulated
        .iter()
        .filter(|t| t.migrated)
        .map(|t| t.marker.clone())
        .reduce(|a, b| format!("{a}, {b}"))
        .map(|m| format!("Moved since the CT: {m}"));
    sink.report(1.0, "done");
    Ok(Result2d3d {
        transform: pose,
        couch,
        displacement,
        metric: Metric2d::Ncc,
        views,
        levels: Vec::new(),
        start: match &req.start {
            Some(_) => "the given pose".into(),
            None => "the plan".into(),
        },
        evaluations: iterations,
        iterations,
        elapsed_secs: started.elapsed().as_secs_f64(),
        curvature: Vec::new(),
        warning,
        markers: Some(MarkerReport {
            residuals,
            rms_mm: rms,
            triangulated,
            dof,
            missing,
        }),
        device: String::new(),
    })
}

/// The markers' pose as the start of an intensity registration: the usual
/// clinical sequence. The result carries the marker report.
pub fn register_hybrid(
    vol: &Volume,
    markers: &MarkerRequest,
    intensity: &Request2d3d,
    sink: &dyn ProgressSink,
) -> Result<Result2d3d> {
    let m = register_markers(vol, markers, sink)?;
    let mut req = intensity.clone();
    req.start = Some(m.transform.clone());
    let mut res = projective::register_2d3d(vol, &req, sink)?;
    res.start = "the markers' pose".into();
    res.markers = m.markers;
    res.elapsed_secs += m.elapsed_secs;
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spot_is_found_where_it_is() {
        let dims = [64, 48];
        let mut data = vec![0.0f32; 64 * 48];
        for j in 0..48 {
            for i in 0..64 {
                let d2 = (i as f64 - 20.3).powi(2) + (j as f64 - 30.6).powi(2);
                data[j * 64 + i] =
                    (5.0 * (-d2 / 4.0).exp()) as f32 + 0.01 * ((i * 7 + j * 13) % 5) as f32;
            }
        }
        let c = detect_2d(&Image2d::new(data, dims), None, 1.4, 4.0);
        assert!(!c.is_empty());
        assert!(
            (c[0].pixel[0] - 20.3).abs() < 0.3 && (c[0].pixel[1] - 30.6).abs() < 0.3,
            "{:?}",
            c[0]
        );
    }

    #[test]
    fn the_assignment_is_one_to_one_and_nearest_overall() {
        let cands = [
            Candidate {
                pixel: [10.0, 10.0],
                strength: 9.0,
            },
            Candidate {
                pixel: [13.0, 10.0],
                strength: 9.0,
            },
        ];
        // Marker 0 is nearest candidate 1 but marker 1 needs it more.
        let a = assign(&[Some([12.0, 10.0]), Some([14.0, 10.0])], &cands, 5.0);
        assert_eq!(a, vec![Some(0), Some(1)]);
        let a = assign(&[Some([12.0, 10.0]), Some([40.0, 10.0])], &cands, 5.0);
        assert_eq!(a, vec![Some(1), None]);
    }

    #[test]
    fn two_rays_meet_where_they_cross() {
        let p = triangulate(&[
            (Vec3::new(-100.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(5.0, -100.0, 3.0), Vec3::new(0.0, 1.0, 0.0)),
        ])
        .unwrap();
        // Closest points (5, 0, 0) and (5, 0, 3): the midpoint.
        assert!((p - Vec3::new(5.0, 0.0, 1.5)).length() < 1e-9, "{p:?}");
    }

    #[test]
    fn the_inverse_jacobian_matches_finite_differences() {
        let p = [0.1, -0.2, 0.15, 3.0, -2.0, 5.0];
        let c = Vec3::new(1.0, 2.0, 3.0);
        let q = Vec3::new(20.0, -10.0, 7.0);
        let j = RigidTransform::new(p, c).unmap_jacobian(q);
        for k in 0..6 {
            let mut a = p;
            let mut b = p;
            a[k] += 1e-6;
            b[k] -= 1e-6;
            let fd = (RigidTransform::new(a, c).unmap(q) - RigidTransform::new(b, c).unmap(q))
                * (1.0 / 2e-6);
            assert!((fd - j[k]).length() < 1e-6, "{k}: {fd:?} vs {:?}", j[k]);
        }
    }
}
