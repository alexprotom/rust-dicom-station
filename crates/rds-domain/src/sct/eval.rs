//! How close a synthetic CT is to a CT of the same patient (*Image
//! synthesis ▶ Evaluate*, the MCP `evaluate_sct` tool, the `EvaluateSct`
//! workflow node).
//!
//! The reference is resampled onto the sCT's grid first
//! ([`crate::registration::resample_through`]: through the active
//! registration, or directly when the two share a frame), then compared
//! inside a mask, as the model card and the SynthRAD challenges compare:
//!
//! * MAE and mean error (sCT minus CT) in HU, overall and per class of the
//!   reference (air below -200 HU, soft tissue -200 to 200, bone above);
//! * PSNR and SSIM over the dynamic range the apps output, [-1024, 3071]
//!   (4095 HU): both images clipped to it, outside the mask set to -1024,
//!   SSIM as scikit-image computes it (7 x 7 x 7 uniform window, sample
//!   covariance, K1 0.01, K2 0.03, reflected at the edges) and averaged
//!   inside the mask;
//! * optionally the water-equivalent thickness through the mask along the
//!   three axes of the grid, through an HU to relative stopping power
//!   table, for proton work;
//! * optionally the Dice of the structures TotalSegmentator (`total`, 3 mm)
//!   finds on each, the model card's third metric.
//!
//! The numbers say how two images differ, not whether a dose computed on
//! the sCT is acceptable: that is the department's acceptance test.

use std::path::Path;

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::nn::device::DevicePref;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::volume::Volume;
use crate::zoo::{AutoModel, RunOptions};

/// The dynamic range the apps write: [-1024, 3071] HU.
pub const RANGE: (f64, f64) = (-1024.0, 3071.0);

/// An HU to relative stopping power (RSP) table: points `(HU, RSP)`,
/// linear between them and constant beyond the ends.
#[derive(Clone, Debug, PartialEq)]
pub struct Hlut(pub Vec<(f64, f64)>);

impl Default for Hlut {
    /// A generic piecewise-linear curve of the usual shape (air, lung,
    /// fat, water, soft tissue, bone), not any scanner's calibration: the
    /// WET *difference* between two images of one patient depends little on
    /// it, the WET itself does. Replace it with the department's own.
    fn default() -> Self {
        Hlut(vec![
            (-1024.0, 0.001),
            (-1000.0, 0.001),
            (-800.0, 0.19),
            (-120.0, 0.93),
            (0.0, 1.0),
            (60.0, 1.05),
            (200.0, 1.10),
            (1000.0, 1.55),
            (3071.0, 2.60),
        ])
    }
}

impl Hlut {
    /// Read `HU:RSP` pairs separated by commas, semicolons or line breaks
    /// (`-1000:0.001, 0:1, 1000:1.55`). The HU must increase.
    pub fn parse(text: &str) -> Result<Hlut> {
        let mut pts = Vec::new();
        for item in text.split([',', ';', '\n']) {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            let Some((h, r)) = item.split_once(':') else {
                bail!("{item:?} is not HU:RSP");
            };
            let h: f64 = h
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("{h:?} is not a number"))?;
            let r: f64 = r
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("{r:?} is not a number"))?;
            if !h.is_finite() || !r.is_finite() || r < 0.0 {
                bail!("{item:?}: the RSP must be a number of zero or more");
            }
            pts.push((h, r));
        }
        if pts.len() < 2 {
            bail!("an HU to RSP table needs two points at least");
        }
        if pts.windows(2).any(|w| w[1].0 <= w[0].0) {
            bail!("the HU of the table must increase");
        }
        Ok(Hlut(pts))
    }

    /// The table as [`Hlut::parse`] reads it.
    pub fn text(&self) -> String {
        self.0
            .iter()
            .map(|(h, r)| format!("{h}:{r}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn rsp(&self, hu: f64) -> f64 {
        let p = &self.0;
        if hu <= p[0].0 {
            return p[0].1;
        }
        for w in p.windows(2) {
            if hu <= w[1].0 {
                let t = (hu - w[0].0) / (w[1].0 - w[0].0);
                return w[0].1 + t * (w[1].1 - w[0].1);
            }
        }
        p[p.len() - 1].1
    }
}

/// What to compute beyond the image metrics.
#[derive(Clone, Debug, Default)]
pub struct EvalOptions {
    /// The table for the WET rows; `None` leaves them out.
    pub wet: Option<Hlut>,
    /// Run TotalSegmentator (`total`, 3 mm) on both and compare its
    /// structures (minutes on a CPU, a download of 140 MB the first time).
    pub structures: bool,
    pub device: DevicePref,
}

/// Agreement inside one set of voxels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    pub voxels: usize,
    /// Mean absolute error, HU.
    pub mae: f64,
    /// Mean error (sCT minus CT), HU.
    pub me: f64,
    pub rmse: f64,
}

/// One class of the reference's values.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassRow {
    pub name: &'static str,
    /// The reference's HU range of the class.
    pub lo: f64,
    pub hi: f64,
    pub m: Metrics,
}

/// Water-equivalent thickness along one axis of the grid, ray by ray
/// through the mask.
#[derive(Clone, Debug, PartialEq)]
pub struct WetRow {
    /// `i (columns)`, `j (rows)`, `k (slices)` with the patient direction
    /// of each.
    pub axis: String,
    pub rays: usize,
    /// Mean WET of the reference over the rays, mm.
    pub mean_wet_mm: f64,
    /// Mean of sCT minus CT, mm.
    pub mean_diff_mm: f64,
    pub mean_abs_diff_mm: f64,
    /// 95th percentile of the absolute difference, mm.
    pub p95_abs_diff_mm: f64,
    /// Mean absolute difference relative to the reference's WET, over the
    /// rays with 10 mm or more of it, %.
    pub mean_abs_rel_pct: f64,
}

/// One structure TotalSegmentator found.
#[derive(Clone, Debug, PartialEq)]
pub struct DiceRow {
    pub name: String,
    pub dice: f64,
    pub cm3_ct: f64,
    pub cm3_sct: f64,
}

pub struct EvalResult {
    /// Voxels compared: inside the mask, with a reference value.
    pub overall: Metrics,
    pub classes: Vec<ClassRow>,
    pub psnr_db: f64,
    pub ssim: f64,
    pub wet: Vec<WetRow>,
    pub dice: Vec<DiceRow>,
    /// Mean Dice over [`EvalResult::dice`].
    pub mean_dice: Option<f64>,
    /// sCT minus CT on the sCT's grid, HU; 0 where there is no reference.
    pub difference: Vec<i16>,
    /// Voxels of the sCT's grid the reference covers.
    pub covered: usize,
    pub elapsed_secs: f64,
}

/// The classes of the per-class rows.
pub const CLASSES: [(&str, f64, f64); 3] = [
    ("air", f64::NEG_INFINITY, -200.0),
    ("soft tissue", -200.0, 200.0),
    ("bone", 200.0, f64::INFINITY),
];

fn metrics(pairs: impl Iterator<Item = (f64, f64)>) -> Metrics {
    let (mut n, mut sa, mut s, mut sq) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    for (a, b) in pairs {
        let d = a - b;
        n += 1;
        sa += d.abs();
        s += d;
        sq += d * d;
    }
    if n == 0 {
        return Metrics::default();
    }
    let n_f = n as f64;
    Metrics {
        voxels: n,
        mae: sa / n_f,
        me: s / n_f,
        rmse: (sq / n_f).sqrt(),
    }
}

/// Half-sample-symmetric reflection (`scipy.ndimage`'s `reflect`).
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    loop {
        if i < 0 {
            i = -i - 1;
        } else if i >= n {
            i = 2 * n - i - 1;
        } else {
            return i as usize;
        }
    }
}

/// Mean over a window of `2r + 1` along one axis of a `[d, h, w]` block,
/// reflected at the ends (the slab axis 0 has a halo of `r` the caller
/// gave it, so its reflection never reaches a slice that is kept).
fn box_axis(src: &[f64], dims: [usize; 3], axis: usize, r: usize) -> Vec<f64> {
    let [d, h, w] = dims;
    let k = (2 * r + 1) as f64;
    let ri = r as isize;
    let mut out = vec![0f64; src.len()];
    match axis {
        2 => out
            .par_chunks_mut(w)
            .zip(src.par_chunks(w))
            .for_each(|(o, s)| {
                for (i, v) in o.iter_mut().enumerate() {
                    let c = i as isize;
                    *v = (c - ri..=c + ri).map(|t| s[reflect(t, w)]).sum::<f64>() / k;
                }
            }),
        1 => out
            .par_chunks_mut(h * w)
            .zip(src.par_chunks(h * w))
            .for_each(|(o, s)| {
                for j in 0..h {
                    let c = j as isize;
                    for x in 0..w {
                        o[j * w + x] = (c - ri..=c + ri)
                            .map(|t| s[reflect(t, h) * w + x])
                            .sum::<f64>()
                            / k;
                    }
                }
            }),
        _ => {
            let plane = h * w;
            out.par_chunks_mut(plane).enumerate().for_each(|(z, o)| {
                let c = z as isize;
                for (p, v) in o.iter_mut().enumerate() {
                    *v = (c - ri..=c + ri)
                        .map(|t| src[reflect(t, d) * plane + p])
                        .sum::<f64>()
                        / k;
                }
            });
        }
    }
    out
}

/// SSIM of `a` against `b` (both already clipped, masked and shifted to
/// [0, range]) averaged over `mask`, slab by slab to bound the memory.
fn ssim(a: &[f32], b: &[f32], mask: &[bool], dims: [usize; 3], range: f64) -> f64 {
    const R: usize = 3;
    const SLAB: usize = 8;
    let [nx, ny, nz] = dims;
    let plane = nx * ny;
    let np = ((2 * R + 1) as f64).powi(3);
    let cov = np / (np - 1.0);
    let c1 = (0.01 * range).powi(2);
    let c2 = (0.03 * range).powi(2);
    let (mut sum, mut n) = (0.0f64, 0usize);
    let mut z0 = 0;
    while z0 < nz {
        let z1 = (z0 + SLAB).min(nz);
        if !mask[z0 * plane..z1 * plane].iter().any(|&m| m) {
            z0 = z1;
            continue;
        }
        // The slab with a reflected halo of R slices on each side.
        let zs: Vec<usize> = (z0 as isize - R as isize..z1 as isize + R as isize)
            .map(|z| reflect(z, nz))
            .collect();
        let d = zs.len();
        let take = |v: &[f32], f: &(dyn Fn(f64, f64) -> f64 + Sync), other: &[f32]| -> Vec<f64> {
            let mut out = vec![0f64; d * plane];
            out.par_chunks_mut(plane).enumerate().for_each(|(s, o)| {
                let src = &v[zs[s] * plane..(zs[s] + 1) * plane];
                let oth = &other[zs[s] * plane..(zs[s] + 1) * plane];
                for ((x, &p), &q) in o.iter_mut().zip(src).zip(oth) {
                    *x = f(p as f64, q as f64);
                }
            });
            out
        };
        let filt = |v: Vec<f64>| -> Vec<f64> {
            let dims3 = [d, ny, nx];
            let v = box_axis(&v, dims3, 2, R);
            let v = box_axis(&v, dims3, 1, R);
            box_axis(&v, dims3, 0, R)
        };
        let ux = filt(take(a, &|p, _| p, b));
        let uy = filt(take(b, &|p, _| p, a));
        let uxx = filt(take(a, &|p, _| p * p, b));
        let uyy = filt(take(b, &|p, _| p * p, a));
        let uxy = filt(take(a, &|p, q| p * q, b));
        let (s, c) = (z0..z1)
            .into_par_iter()
            .map(|z| {
                let zi = z - z0 + R;
                let (mut s, mut c) = (0.0f64, 0usize);
                for p in 0..plane {
                    if !mask[z * plane + p] {
                        continue;
                    }
                    let i = zi * plane + p;
                    let (mx, my) = (ux[i], uy[i]);
                    let vx = cov * (uxx[i] - mx * mx);
                    let vy = cov * (uyy[i] - my * my);
                    let vxy = cov * (uxy[i] - mx * my);
                    let num = (2.0 * mx * my + c1) * (2.0 * vxy + c2);
                    let den = (mx * mx + my * my + c1) * (vx + vy + c2);
                    s += num / den;
                    c += 1;
                }
                (s, c)
            })
            .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        sum += s;
        n += c;
        z0 = z1;
    }
    if n == 0 {
        f64::NAN
    } else {
        sum / n as f64
    }
}

/// The patient direction a grid axis points to, for the WET rows.
fn axis_name(v: crate::geometry::Vec3, axis: usize) -> String {
    let (ax, comp) = [(v.x.abs(), 0), (v.y.abs(), 1), (v.z.abs(), 2)]
        .into_iter()
        .fold((0.0, 0), |m, (a, c)| if a > m.0 { (a, c) } else { m });
    let _ = ax;
    let dir = match comp {
        0 => "left-right",
        1 => "anterior-posterior",
        _ => "inferior-superior",
    };
    let which = ["i (columns)", "j (rows)", "k (slices)"][axis];
    format!("{which}, {dir}")
}

/// WET along each axis of the grid, ray by ray through `mask`.
fn wet_rows(sct: &Volume, reference: &[f32], mask: &[bool], hlut: &Hlut) -> Vec<WetRow> {
    let [nx, ny, nz] = sct.dims;
    let idx = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let dirs = [sct.row_dir, sct.col_dir, sct.normal];
    (0..3)
        .map(|axis| {
            let (n_rays, len) = match axis {
                0 => (ny * nz, nx),
                1 => (nx * nz, ny),
                _ => (nx * ny, nz),
            };
            let step = sct.spacing[axis];
            let rays: Vec<(f64, f64)> = (0..n_rays)
                .into_par_iter()
                .filter_map(|r| {
                    let (mut ws, mut wc, mut any) = (0.0f64, 0.0f64, false);
                    for t in 0..len {
                        let v = match axis {
                            0 => idx(t, r % ny, r / ny),
                            1 => idx(r % nx, t, r / nx),
                            _ => idx(r % nx, r / nx, t),
                        };
                        if !mask[v] {
                            continue;
                        }
                        any = true;
                        ws += hlut.rsp(sct.data[v] as f64) * step;
                        wc += hlut.rsp(reference[v] as f64) * step;
                    }
                    any.then_some((ws, wc))
                })
                .collect();
            let n = rays.len();
            let mut abs: Vec<f64> = rays.iter().map(|(s, c)| (s - c).abs()).collect();
            abs.sort_by(|a, b| a.total_cmp(b));
            let rel: Vec<f64> = rays
                .iter()
                .filter(|(_, c)| *c >= 10.0)
                .map(|(s, c)| (s - c).abs() / c * 100.0)
                .collect();
            let mean = |v: &mut dyn Iterator<Item = f64>, n: usize| {
                if n == 0 {
                    0.0
                } else {
                    v.sum::<f64>() / n as f64
                }
            };
            WetRow {
                axis: axis_name(dirs[axis], axis),
                rays: n,
                mean_wet_mm: mean(&mut rays.iter().map(|r| r.1), n),
                mean_diff_mm: mean(&mut rays.iter().map(|(s, c)| s - c), n),
                mean_abs_diff_mm: mean(&mut abs.iter().copied(), n),
                p95_abs_diff_mm: if n == 0 {
                    0.0
                } else {
                    abs[((n as f64 * 0.95).ceil() as usize).clamp(1, n) - 1]
                },
                mean_abs_rel_pct: mean(&mut rel.iter().copied(), rel.len()),
            }
        })
        .collect()
}

/// A volume on `like`'s grid holding `data`.
fn like(like: &Volume, data: Vec<i16>) -> Volume {
    let (lo, hi) = data
        .iter()
        .fold((i16::MAX, i16::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    Volume {
        min_value: lo,
        max_value: hi,
        data,
        dims: like.dims,
        spacing: like.spacing,
        origin: like.origin,
        row_dir: like.row_dir,
        col_dir: like.col_dir,
        normal: like.normal,
        frame_of_reference_uid: like.frame_of_reference_uid.clone(),
    }
}

/// TotalSegmentator on both images, Dice per structure it finds on the
/// reference (1 cm³ or more inside the evaluated region).
fn dice_rows(
    sct: &Volume,
    reference: &[f32],
    opts: &EvalOptions,
    root: &Path,
    progress: &Progress,
) -> Result<Vec<DiceRow>> {
    let model = AutoModel::Nn(&crate::autoseg::total::TOTAL_FAST);
    // Both images go through the same network: built once.
    let _keep = crate::autoseg::KeepNetworks::new();
    let ro = RunOptions {
        device: opts.device,
        parts: None,
    };
    let ct = like(
        sct,
        reference
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    -1024
                } else {
                    v.round().clamp(-32768.0, 32767.0) as i16
                }
            })
            .collect(),
    );
    progress.set_outer(0.45, 0.25);
    let a = model.run(sct, &ro, root, progress)?;
    progress.set_outer(0.70, 0.25);
    let b = model.run(&ct, &ro, root, progress)?;
    let voxel_cm3 = crate::volume::voxel_cm3(sct.spacing);
    let n_classes = model.classes().len();
    let mut inter = vec![0u64; n_classes + 1];
    let mut na = vec![0u64; n_classes + 1];
    let mut nb = vec![0u64; n_classes + 1];
    for (v, (&la, &lb)) in a.labels.iter().zip(&b.labels).enumerate() {
        if reference[v].is_nan() {
            continue;
        }
        na[la as usize] += 1;
        nb[lb as usize] += 1;
        if la == lb {
            inter[la as usize] += 1;
        }
    }
    let mut rows: Vec<DiceRow> = (1..=n_classes)
        .filter(|&l| nb[l] as f64 * voxel_cm3 >= 1.0)
        .map(|l| DiceRow {
            name: model.classes()[l - 1].to_string(),
            dice: 2.0 * inter[l] as f64 / (na[l] + nb[l]) as f64,
            cm3_ct: nb[l] as f64 * voxel_cm3,
            cm3_sct: na[l] as f64 * voxel_cm3,
        })
        .collect();
    rows.sort_by(|x, y| y.cm3_ct.total_cmp(&x.cm3_ct));
    Ok(rows)
}

/// Compare `sct` with `reference` (on `sct`'s grid, `NaN` where it has no
/// value; [`crate::registration::resample_through`] makes one) inside
/// `mask` (`sct`'s grid; `None`: wherever the reference has a value).
/// Blocking. `root` is the model folder, for the structure Dice.
pub fn evaluate(
    sct: &Volume,
    reference: &[f32],
    mask: Option<&[u8]>,
    opts: &EvalOptions,
    root: &Path,
    progress: &Progress,
) -> Result<EvalResult> {
    let start = std::time::Instant::now();
    let n = sct.data.len();
    if reference.len() != n {
        bail!("the reference is not on the synthetic CT's grid");
    }
    if let Some(m) = mask {
        if m.len() != n {
            bail!("the mask is not on the synthetic CT's grid");
        }
    }
    progress.set_outer(0.0, 1.0);
    progress.report(0.0, "Comparing the images");
    let covered = reference.iter().filter(|v| !v.is_nan()).count();
    if covered == 0 {
        bail!("the reference does not overlap the synthetic CT");
    }
    let inside: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|v| !reference[v].is_nan() && mask.is_none_or(|m| m[v] != 0))
        .collect();
    if !inside.iter().any(|&b| b) {
        bail!("the mask and the reference do not overlap");
    }
    let pairs = || {
        (0..n)
            .filter(|&v| inside[v])
            .map(|v| (sct.data[v] as f64, reference[v] as f64))
    };
    let overall = metrics(pairs());
    let classes = CLASSES
        .iter()
        .map(|&(name, lo, hi)| ClassRow {
            name,
            lo,
            hi,
            m: metrics(pairs().filter(|&(_, c)| c >= lo && c < hi)),
        })
        .collect();

    // PSNR and SSIM over the apps' range, outside the mask at its bottom.
    let (lo, hi) = RANGE;
    let span = hi - lo;
    let clip = |v: f64, keep: bool| -> f32 {
        if keep {
            (v.clamp(lo, hi) - lo) as f32
        } else {
            0.0
        }
    };
    let a: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|v| clip(sct.data[v] as f64, inside[v]))
        .collect();
    let b: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|v| clip(reference[v] as f64, inside[v]))
        .collect();
    let mse = (0..n)
        .filter(|&v| inside[v])
        .map(|v| ((a[v] - b[v]) as f64).powi(2))
        .sum::<f64>()
        / overall.voxels as f64;
    let psnr_db = if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (span * span / mse).log10()
    };
    progress.report(0.1, "Structural similarity");
    let ssim = ssim(&a, &b, &inside, sct.dims, span);
    drop((a, b));
    if progress.cancelled() {
        bail!(CANCELLED);
    }
    progress.report(0.4, "Water-equivalent thickness");
    let wet = match &opts.wet {
        Some(h) => wet_rows(sct, reference, &inside, h),
        None => Vec::new(),
    };
    let dice = if opts.structures {
        dice_rows(sct, reference, opts, root, progress)?
    } else {
        Vec::new()
    };
    progress.set_outer(0.0, 1.0);
    let mean_dice =
        (!dice.is_empty()).then(|| dice.iter().map(|d| d.dice).sum::<f64>() / dice.len() as f64);
    let difference: Vec<i16> = (0..n)
        .into_par_iter()
        .map(|v| {
            if reference[v].is_nan() {
                0
            } else {
                (sct.data[v] as f32 - reference[v])
                    .round()
                    .clamp(-32768.0, 32767.0) as i16
            }
        })
        .collect();
    progress.report(1.0, "Evaluated");
    Ok(EvalResult {
        overall,
        classes,
        psnr_db,
        ssim,
        wet,
        dice,
        mean_dice,
        difference,
        covered,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}

/// The numbers as a CSV table: `section,item,metric,value`.
pub fn csv(r: &EvalResult) -> String {
    let mut s = String::from("section,item,metric,value\n");
    let mut row = |a: &str, b: &str, c: &str, v: f64| {
        s.push_str(&format!("{a},{b},{c},{v}\n"));
    };
    for (name, m) in
        std::iter::once(("all", &r.overall)).chain(r.classes.iter().map(|c| (c.name, &c.m)))
    {
        row("hu", name, "voxels", m.voxels as f64);
        row("hu", name, "mae_hu", m.mae);
        row("hu", name, "me_hu", m.me);
        row("hu", name, "rmse_hu", m.rmse);
    }
    row("image", "all", "psnr_db", r.psnr_db);
    row("image", "all", "ssim", r.ssim);
    for w in &r.wet {
        let axis = w.axis.replace(',', ";");
        row("wet", &axis, "rays", w.rays as f64);
        row("wet", &axis, "mean_wet_mm", w.mean_wet_mm);
        row("wet", &axis, "mean_diff_mm", w.mean_diff_mm);
        row("wet", &axis, "mean_abs_diff_mm", w.mean_abs_diff_mm);
        row("wet", &axis, "p95_abs_diff_mm", w.p95_abs_diff_mm);
        row("wet", &axis, "mean_abs_rel_pct", w.mean_abs_rel_pct);
    }
    for d in &r.dice {
        row("dice", &d.name, "dice", d.dice);
        row("dice", &d.name, "cm3_ct", d.cm3_ct);
        row("dice", &d.name, "cm3_sct", d.cm3_sct);
    }
    if let Some(m) = r.mean_dice {
        row("dice", "mean", "dice", m);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    fn vol(dims: [usize; 3], f: impl Fn(usize, usize, usize) -> i16) -> Volume {
        let mut data = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    data.push(f(i, j, k));
                }
            }
        }
        Volume {
            min_value: *data.iter().min().unwrap(),
            max_value: *data.iter().max().unwrap(),
            data,
            dims,
            spacing: [1.0, 1.0, 2.0],
            origin: Vec3::ZERO,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        }
    }

    #[test]
    fn the_table_reads_and_interpolates() {
        let h = Hlut::parse("-1000:0.001; 0:1\n1000 : 1.5").unwrap();
        assert_eq!(h.rsp(-2000.0), 0.001);
        assert_eq!(h.rsp(0.0), 1.0);
        assert!((h.rsp(500.0) - 1.25).abs() < 1e-12);
        assert_eq!(h.rsp(5000.0), 1.5);
        assert_eq!(Hlut::parse(&h.text()).unwrap(), h);
        assert!(Hlut::parse("0:1").is_err());
        assert!(Hlut::parse("0:1, 0:2").is_err());
        assert!(Hlut::parse("0:x, 1:2").is_err());
        let d = Hlut::default();
        assert!((d.rsp(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn identical_images_agree_perfectly() {
        let ct = vol([20, 18, 12], |i, j, k| {
            ((i * 37 + j * 11 + k * 5) % 700) as i16 - 300
        });
        let reference: Vec<f32> = ct.data.iter().map(|&v| v as f32).collect();
        let opts = EvalOptions {
            wet: Some(Hlut::default()),
            ..EvalOptions::default()
        };
        let r = evaluate(
            &ct,
            &reference,
            None,
            &opts,
            Path::new("."),
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(r.overall.voxels, ct.data.len());
        assert_eq!((r.overall.mae, r.overall.me), (0.0, 0.0));
        assert!(r.psnr_db.is_infinite());
        assert!((r.ssim - 1.0).abs() < 1e-9, "ssim {}", r.ssim);
        assert!(r.difference.iter().all(|&d| d == 0));
        assert_eq!(r.wet.len(), 3);
        for w in &r.wet {
            assert!(w.rays > 0 && w.mean_wet_mm > 0.0);
            assert_eq!((w.mean_abs_diff_mm, w.p95_abs_diff_mm), (0.0, 0.0));
        }
        assert!(r.wet[2].axis.contains("inferior-superior"));
        let text = csv(&r);
        assert!(text.starts_with("section,item,metric,value\n"));
        assert!(text.contains("hu,all,mae_hu,0\n"));
    }

    #[test]
    fn an_offset_shows_in_the_error_and_the_wet() {
        let ct = vol([16, 16, 8], |_, _, _| 0);
        let sct = vol([16, 16, 8], |_, _, _| 100);
        let mut reference: Vec<f32> = ct.data.iter().map(|&v| v as f32).collect();
        // The reference does not cover the first slice.
        for v in &mut reference[..256] {
            *v = f32::NAN;
        }
        let opts = EvalOptions {
            wet: Some(Hlut::parse("0:1, 100:1.1").unwrap()),
            ..EvalOptions::default()
        };
        let r = evaluate(
            &sct,
            &reference,
            None,
            &opts,
            Path::new("."),
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(r.covered, 16 * 16 * 7);
        assert_eq!(r.overall.voxels, 16 * 16 * 7);
        assert_eq!(
            (r.overall.mae, r.overall.me, r.overall.rmse),
            (100.0, 100.0, 100.0)
        );
        // All of it is soft tissue by the reference.
        assert_eq!(r.classes[1].m.voxels, r.overall.voxels);
        assert_eq!(r.classes[0].m.voxels + r.classes[2].m.voxels, 0);
        let want_psnr = 10.0 * (4095.0f64 * 4095.0 / 10000.0).log10();
        assert!((r.psnr_db - want_psnr).abs() < 1e-9);
        assert!(r.ssim < 1.0);
        // Along the columns: 16 voxels of 1 mm, RSP 1.1 against 1.0.
        assert!((r.wet[0].mean_diff_mm - 1.6).abs() < 1e-9, "{:?}", r.wet[0]);
        assert!((r.wet[0].mean_abs_rel_pct - 10.0).abs() < 1e-9);
        assert_eq!(r.difference[0], 0);
        assert_eq!(r.difference[256], 100);
        // A mask narrows it.
        let mut mask = vec![0u8; ct.data.len()];
        mask[300] = 1;
        let r = evaluate(
            &sct,
            &reference,
            Some(&mask),
            &EvalOptions::default(),
            Path::new("."),
            &Progress::default(),
        )
        .unwrap();
        assert_eq!(r.overall.voxels, 1);
        assert!(r.wet.is_empty());
    }

    #[test]
    fn ssim_follows_the_reference_formula() {
        // Two small random-looking images; SSIM by brute force over the
        // reflected 7-voxel window, as scikit-image computes it.
        let dims = [9, 8, 7];
        let a: Vec<f32> = (0..dims.iter().product::<usize>())
            .map(|i| ((i * 7919) % 4096) as f32)
            .collect();
        let b: Vec<f32> = a
            .iter()
            .enumerate()
            .map(|(i, v)| (v + (i % 13) as f32 * 30.0).min(4095.0))
            .collect();
        let mask = vec![true; a.len()];
        let got = ssim(&a, &b, &mask, dims, 4095.0);
        let [nx, ny, nz] = dims;
        let at = |v: &[f32], i: isize, j: isize, k: isize| -> f64 {
            v[(reflect(k, nz) * ny + reflect(j, ny)) * nx + reflect(i, nx)] as f64
        };
        let (c1, c2) = ((0.01 * 4095.0f64).powi(2), (0.03 * 4095.0f64).powi(2));
        let np = 343.0;
        let mut total = 0.0;
        for k in 0..nz as isize {
            for j in 0..ny as isize {
                for i in 0..nx as isize {
                    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                    for dk in -3..=3 {
                        for dj in -3..=3 {
                            for di in -3..=3 {
                                let x = at(&a, i + di, j + dj, k + dk);
                                let y = at(&b, i + di, j + dj, k + dk);
                                sx += x;
                                sy += y;
                                sxx += x * x;
                                syy += y * y;
                                sxy += x * y;
                            }
                        }
                    }
                    let (mx, my) = (sx / np, sy / np);
                    let cov = np / (np - 1.0);
                    let vx = cov * (sxx / np - mx * mx);
                    let vy = cov * (syy / np - my * my);
                    let vxy = cov * (sxy / np - mx * my);
                    total += ((2.0 * mx * my + c1) * (2.0 * vxy + c2))
                        / ((mx * mx + my * my + c1) * (vx + vy + c2));
                }
            }
        }
        let want = total / a.len() as f64;
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
}
