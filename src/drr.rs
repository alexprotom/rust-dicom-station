//! Digitally reconstructed radiographs — two independent forward projectors.
//!
//! A DRR is a line integral of attenuation from a point source through the
//! CT to a flat detector: the simulated radiograph a treatment beam would
//! produce, and the image every image-guidance workflow compares a portal or
//! kV image against. There is more than one honest way to compute it, and
//! the two implemented here come from different lineages and disagree in
//! interesting ways:
//!
//! * [`Engine::Siddon`] — **plastimatch**'s exact ray tracer (`drr -i
//!   exact`), after Siddon (Med. Phys. 1985) with Jacobs' incremental
//!   formulation. The ray is intersected with the three families of voxel
//!   planes and each voxel contributes exactly the length of ray inside it.
//!   No interpolation, no sampling step: the integral is *exact* for a
//!   piecewise-constant volume, and it is the reference the other one is
//!   checked against.
//! * [`Engine::RayCast`] — the **ITK / elastix-stack**
//!   `RayCastInterpolateImageFunction` used by `itkImageToImageMetric`-based
//!   2-D/3-D registration: march the ray at a fixed step and accumulate
//!   trilinearly interpolated values. The volume is treated as a smooth
//!   field rather than a set of boxes, so edges come out softer, and the
//!   step size is a real accuracy/speed knob rather than a formality.
//!
//! Running both on the same geometry and subtracting is the point: the
//! difference image is a direct measure of the interpolation error one
//! accepts by choosing either.
//!
//! ## Geometry
//!
//! [`Geometry`] is a cone-beam geometry in IEC 61217 terms — source-to-axis
//! and source-to-imager distances, gantry and couch angles about an
//! isocentre — because that is how a linac states it and how an RTPLAN beam
//! stores it ([`Geometry::from_beam`]). The IEC fixed frame is mapped to the
//! DICOM patient frame for a head-first supine patient: `Xf` (patient left)
//! = `+x`, `Yf` (gantry rotation axis, towards the head) = `+z`, `Zf`
//! (vertical, up) = `−y`. Gantry 0° puts the source directly above the
//! patient; 90° puts it on the patient's left.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::extras::PlanarImage;

use crate::geometry::Vec3;
use crate::progress::{ProgressSink, CANCELLED};
use crate::rtplan::BeamInfo;
use crate::volume::Volume;

/// Which forward projector to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Engine {
    /// plastimatch: exact voxel-intersection ray tracing (Siddon/Jacobs).
    #[default]
    Siddon,
    /// ITK: fixed-step ray marching with trilinear interpolation.
    RayCast,
}

impl Engine {
    pub const ALL: [Engine; 2] = [Engine::Siddon, Engine::RayCast];

    pub fn label(self) -> &'static str {
        match self {
            Engine::Siddon => "Siddon (plastimatch)",
            Engine::RayCast => "Ray-cast (ITK)",
        }
    }

    /// The engine's name without its lineage, for compact labels.
    pub fn short(self) -> &'static str {
        match self {
            Engine::Siddon => "Siddon",
            Engine::RayCast => "Ray-cast",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Engine::Siddon => {
                "Exact voxel-intersection ray tracing: every voxel contributes the true \
                 length of ray inside it. No sampling step and no interpolation, so the \
                 result is exact for the voxel model - and slightly harder-edged."
            }
            Engine::RayCast => {
                "Fixed-step marching with trilinear interpolation, as ITK's \
                 RayCastInterpolateImageFunction does it. Treats the volume as a smooth \
                 field, so edges are softer; the step size trades accuracy for speed."
            }
        }
    }
}

/// How voxel values become attenuation.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum HuMode {
    /// `μ = μ_water · (1 + HU/1000)`, clamped at zero — plastimatch's
    /// `preprocess` conversion. Physically meaningful: the integral is an
    /// optical depth and the image is a real radiograph.
    #[default]
    Water,
    /// Integrate the raw values (plastimatch `-h none`). No physics, but it
    /// is what one wants when comparing against another tool's raw output.
    Raw,
}

impl HuMode {
    pub const ALL: [HuMode; 2] = [HuMode::Water, HuMode::Raw];

    pub fn label(self) -> &'static str {
        match self {
            HuMode::Water => "Attenuation (μ from HU)",
            HuMode::Raw => "Raw line integral",
        }
    }
}

/// Linear attenuation coefficient of water at ~60 keV, mm⁻¹ — the effective
/// energy plastimatch's DRR preprocessing assumes.
pub const MU_WATER: f64 = 0.0206;

/// Cone-beam geometry, IEC 61217.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Source-to-axis distance, mm.
    pub sad: f64,
    /// Source-to-imager distance, mm.
    pub sid: f64,
    /// Gantry angle, degrees (0 = source above the patient).
    pub gantry_deg: f64,
    /// Patient-support (couch) angle, degrees.
    pub couch_deg: f64,
    /// Isocentre in patient coordinates.
    pub isocenter: Vec3,
    /// Detector size at the imager, mm (width × height).
    pub panel_mm: [f64; 2],
    /// Output image size, pixels.
    pub dims: [usize; 2],
}

impl Default for Geometry {
    fn default() -> Self {
        Geometry {
            sad: 1000.0,
            sid: 1500.0,
            gantry_deg: 0.0,
            couch_deg: 0.0,
            isocenter: Vec3::ZERO,
            panel_mm: [400.0, 400.0],
            dims: [512, 512],
        }
    }
}

/// The source, the detector centre and the detector's in-plane axes.
pub struct BeamFrame {
    pub source: Vec3,
    /// Unit vector from the source towards the isocentre.
    pub axis: Vec3,
    pub panel_center: Vec3,
    /// Unit vectors of the detector's +x and +y pixel directions.
    pub panel_x: Vec3,
    pub panel_y: Vec3,
}

impl Geometry {
    /// Take the angles and the isocentre from a plan beam; the distances,
    /// panel and resolution stay as they are.
    pub fn from_beam(&self, beam: &BeamInfo) -> Geometry {
        Geometry {
            gantry_deg: beam.gantry_angle.unwrap_or(self.gantry_deg),
            couch_deg: beam.couch_angle.unwrap_or(self.couch_deg),
            isocenter: beam.isocenter.unwrap_or(self.isocenter),
            ..*self
        }
    }

    /// The isocentre of the volume's own centre — a sane default when there
    /// is no plan to take one from.
    pub fn centered_on(vol: &Volume) -> Geometry {
        let d = vol.dims;
        Geometry {
            isocenter: vol.voxel_to_patient(
                (d[0] as f64 - 1.0) * 0.5,
                (d[1] as f64 - 1.0) * 0.5,
                (d[2] as f64 - 1.0) * 0.5,
            ),
            ..Geometry::default()
        }
    }

    /// Where everything is, in patient coordinates.
    pub fn frame(&self) -> BeamFrame {
        // IEC fixed frame for a head-first supine patient, in DICOM LPS.
        let xf = Vec3::new(1.0, 0.0, 0.0); // patient left
        let yf = Vec3::new(0.0, 0.0, 1.0); // towards the head, the gantry axis
        let zf = Vec3::new(0.0, -1.0, 0.0); // vertical, up (anterior)

        let g = self.gantry_deg.to_radians();
        // Source direction from the isocentre: up at 0°, patient-left at 90°.
        let mut to_source = xf * g.sin() + zf * g.cos();
        // The detector's +x runs with the gantry; +y stays along the axis.
        let mut panel_x = xf * g.cos() - zf * g.sin();
        let mut panel_y = yf;

        // Rotating the couch by C is, from the beam's point of view, rotating
        // the patient by −C about the vertical axis.
        let c = -self.couch_deg.to_radians();
        if c != 0.0 {
            let rot = |v: Vec3| {
                // Rotation about zf (= −y in LPS) by `c`.
                let (s, co) = c.sin_cos();
                // Components in the (xf, yf) plane rotate; zf is fixed.
                let a = v.dot(xf);
                let b = v.dot(yf);
                let h = v.dot(zf);
                xf * (a * co - b * s) + yf * (a * s + b * co) + zf * h
            };
            to_source = rot(to_source);
            panel_x = rot(panel_x);
            panel_y = rot(panel_y);
        }

        BeamFrame {
            source: self.isocenter + to_source * self.sad,
            axis: to_source * -1.0,
            panel_center: self.isocenter + to_source * -(self.sid - self.sad),
            panel_x,
            panel_y,
        }
    }

    /// Detector pixel spacing, mm.
    pub fn pixel_mm(&self) -> [f64; 2] {
        [
            self.panel_mm[0] / self.dims[0].max(1) as f64,
            self.panel_mm[1] / self.dims[1].max(1) as f64,
        ]
    }

    /// Pixel spacing projected back to the isocentre plane — what a
    /// millimetre on the DRR is worth on the patient.
    pub fn pixel_mm_at_isocenter(&self) -> [f64; 2] {
        let m = self.sad / self.sid.max(1e-6);
        let p = self.pixel_mm();
        [p[0] * m, p[1] * m]
    }
}

/// What a run needs beyond the volume.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrrParams {
    pub geometry: Geometry,
    pub engine: Engine,
    pub hu: HuMode,
    /// Ray-cast step, mm (ignored by the exact tracer).
    pub step_mm: f64,
    /// Voxels below this value contribute nothing — the standard way to keep
    /// couch and air out of a DRR.
    pub threshold_hu: f32,
}

impl DrrParams {
    pub fn for_volume(vol: &Volume) -> DrrParams {
        DrrParams {
            geometry: Geometry::centered_on(vol),
            engine: Engine::Siddon,
            hu: HuMode::Water,
            step_mm: 1.0,
            threshold_hu: -800.0,
        }
    }
}

/// A rendered radiograph.
#[derive(Clone)]
pub struct DrrImage {
    pub pixels: Vec<f32>,
    pub dims: [usize; 2],
    /// Detector pixel spacing, mm.
    pub spacing: [f64; 2],
    pub min: f32,
    pub max: f32,
    pub engine: Engine,
    pub elapsed_secs: f64,
}

impl DrrImage {
    /// Mean of the image.
    pub fn mean(&self) -> f32 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        self.pixels.iter().sum::<f32>() / self.pixels.len() as f32
    }

    /// `512 × 512 · 0.78 mm/px · range 0.00 – 24.31`.
    pub fn describe(&self) -> String {
        format!(
            "{} × {} · {:.2} mm/px · range {:.2} - {:.2} · {:.2} s",
            self.dims[0], self.dims[1], self.spacing[0], self.min, self.max, self.elapsed_secs
        )
    }

    /// File this rendering as a planar image, so it can live in the data tree
    /// beside the DX / CR / RTIMAGE the study came with — a DRR *is* an RT
    /// Image, and once it is one it inherits everything the tree already
    /// does: its own viewer window, renaming, and travelling with the study
    /// when it is copied or moved.
    ///
    /// `invert` stores the greyscale the way the DRR window is showing it.
    /// A line integral is large where attenuation is large, which on a
    /// radiograph reads as *dark*; inverting keeps what lands in the tree
    /// looking like what was on screen. The geometry that produced the image
    /// is carried along as the info rows the planar viewer lists.
    pub fn to_planar(&self, params: &DrrParams, invert: bool) -> PlanarImage {
        let g = &params.geometry;
        // The inversion maps [min, max] onto itself, so the range and the
        // default window are the same either way.
        let data: Vec<f32> = if invert {
            self.pixels
                .iter()
                .map(|v| self.min + self.max - v)
                .collect()
        } else {
            self.pixels.clone()
        };
        let mut info = vec![
            ("Source".into(), format!("DRR - {}", self.engine.label())),
            (
                "Geometry".into(),
                format!(
                    "SAD {:.0} mm · SID {:.0} mm · gantry {:.1}° · couch {:.1}°",
                    g.sad, g.sid, g.gantry_deg, g.couch_deg
                ),
            ),
            (
                "Isocentre".into(),
                format!(
                    "({:.1}, {:.1}, {:.1}) mm",
                    g.isocenter.x, g.isocenter.y, g.isocenter.z
                ),
            ),
            (
                "Panel".into(),
                format!("{:.0} × {:.0} mm", g.panel_mm[0], g.panel_mm[1]),
            ),
            ("HU model".into(), params.hu.label().into()),
            ("Threshold".into(), format!("{:.0} HU", params.threshold_hu)),
        ];
        if self.engine == Engine::RayCast {
            info.push(("Step".into(), format!("{:.2} mm", params.step_mm)));
        }
        info.push(("Render time".into(), format!("{:.2} s", self.elapsed_secs)));
        info.push((
            "Greyscale".into(),
            if invert {
                "inverted - dark is high attenuation, as on a radiograph".into()
            } else {
                "line integral - bright is high attenuation".into()
            },
        ));
        PlanarImage {
            label: format!(
                "DRR {} · G {:.0}° C {:.0}°",
                self.engine.short(),
                g.gantry_deg,
                g.couch_deg
            ),
            modality: "RTIMAGE".into(),
            rows: self.dims[1],
            cols: self.dims[0],
            spacing: self.spacing,
            data,
            min_value: self.min,
            max_value: self.max,
            window: ((self.min + self.max) * 0.5, (self.max - self.min).max(1.0)),
            info,
        }
    }
}

/// How two renderings of the same geometry differ — the reason for having
/// two implementations at all.
#[derive(Clone, Copy, Debug, Default)]
pub struct DrrComparison {
    pub max_abs: f32,
    pub mean_abs: f32,
    /// Mean absolute difference as a fraction of the mean image value.
    pub relative: f32,
    /// Pearson correlation of the two images.
    pub correlation: f32,
}

impl DrrComparison {
    /// Compare two renderings pixel for pixel.
    pub fn of(a: &DrrImage, b: &DrrImage) -> Option<DrrComparison> {
        if a.dims != b.dims || a.pixels.is_empty() {
            return None;
        }
        let n = a.pixels.len() as f64;
        let (mut sa, mut sb) = (0.0f64, 0.0f64);
        for (x, y) in a.pixels.iter().zip(&b.pixels) {
            sa += *x as f64;
            sb += *y as f64;
        }
        let (ma, mb) = (sa / n, sb / n);
        let (mut max_abs, mut sum_abs) = (0.0f64, 0.0f64);
        let (mut caa, mut cbb, mut cab) = (0.0f64, 0.0f64, 0.0f64);
        for (x, y) in a.pixels.iter().zip(&b.pixels) {
            let d = (*x - *y) as f64;
            max_abs = max_abs.max(d.abs());
            sum_abs += d.abs();
            let (da, db) = (*x as f64 - ma, *y as f64 - mb);
            caa += da * da;
            cbb += db * db;
            cab += da * db;
        }
        let mean_abs = sum_abs / n;
        Some(DrrComparison {
            max_abs: max_abs as f32,
            mean_abs: mean_abs as f32,
            relative: if ma.abs() > 1e-9 {
                (mean_abs / ma.abs()) as f32
            } else {
                0.0
            },
            correlation: if caa > 0.0 && cbb > 0.0 {
                (cab / (caa * cbb).sqrt()) as f32
            } else {
                0.0
            },
        })
    }

    /// `max 0.14 · mean 0.007 (0.05 %) · r = 0.99998`.
    pub fn line(&self) -> String {
        format!(
            "max {:.4} · mean {:.5} ({:.3} %) · r = {:.6}",
            self.max_abs,
            self.mean_abs,
            100.0 * self.relative,
            self.correlation
        )
    }
}

/// Voxel value → attenuation per millimetre.
#[inline]
fn attenuation(v: f32, mode: HuMode, threshold: f32) -> f32 {
    if v < threshold {
        return 0.0;
    }
    match mode {
        HuMode::Water => {
            let mu = MU_WATER as f32 * (1.0 + v / 1000.0);
            if mu > 0.0 {
                mu
            } else {
                0.0
            }
        }
        HuMode::Raw => v,
    }
}

/// The parametric range `[t0, t1]` over which a ray stays inside the volume,
/// in the ray's own `p = a + t·(b − a)` parameterization, with everything
/// expressed in continuous voxel indices.
///
/// Voxel `i` covers `[i − ½, i + ½]`, so the volume occupies `[−½, n − ½]`.
fn clip_to_volume(a: [f64; 3], d: [f64; 3], dims: [usize; 3]) -> Option<(f64, f64)> {
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    for axis in 0..3 {
        let lo = -0.5;
        let hi = dims[axis] as f64 - 0.5;
        if d[axis].abs() < 1e-12 {
            if a[axis] < lo || a[axis] > hi {
                return None;
            }
            continue;
        }
        let mut ta = (lo - a[axis]) / d[axis];
        let mut tb = (hi - a[axis]) / d[axis];
        if ta > tb {
            std::mem::swap(&mut ta, &mut tb);
        }
        t0 = t0.max(ta);
        t1 = t1.min(tb);
        if t0 >= t1 {
            return None;
        }
    }
    Some((t0, t1))
}

/// Exact line integral along one ray (Siddon / Jacobs).
///
/// Walks voxel to voxel, always crossing whichever of the three plane
/// families comes next, and adds `length × attenuation` for the voxel just
/// left behind. Nothing is interpolated and nothing is sampled: for a
/// piecewise-constant volume this *is* the integral.
fn siddon(vol: &Volume, src: Vec3, dst: Vec3, mode: HuMode, threshold: f32) -> f32 {
    let a = vol.patient_to_voxel(src);
    let b = vol.patient_to_voxel(dst);
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let Some((t0, t1)) = clip_to_volume(a, d, vol.dims) else {
        return 0.0;
    };
    // Physical length of the whole parametric interval.
    let total_mm = (dst - src).length();

    // Voxel the ray is in just after entering.
    let mut idx = [0i64; 3];
    let mut step = [0i64; 3];
    // Parametric distance to the next plane crossing per axis, and the
    // parametric length of one voxel step per axis.
    let mut t_next = [f64::INFINITY; 3];
    let mut dt = [f64::INFINITY; 3];
    let eps = 1e-9;
    for axis in 0..3 {
        let p = a[axis] + (t0 + eps) * d[axis];
        idx[axis] = p.round() as i64;
        idx[axis] = idx[axis].clamp(0, vol.dims[axis] as i64 - 1);
        if d[axis].abs() < 1e-12 {
            continue;
        }
        step[axis] = if d[axis] > 0.0 { 1 } else { -1 };
        dt[axis] = (1.0 / d[axis]).abs();
        // The boundary of the current voxel in the direction of travel.
        let boundary = idx[axis] as f64 + 0.5 * step[axis] as f64;
        t_next[axis] = (boundary - a[axis]) / d[axis];
    }

    let mut t = t0;
    let mut acc = 0.0f32;
    // At most one step per voxel along each axis, plus a margin.
    let limit = 4 * (vol.dims[0] + vol.dims[1] + vol.dims[2]) + 8;
    for _ in 0..limit {
        if t >= t1 {
            break;
        }
        let axis = if t_next[0] <= t_next[1] && t_next[0] <= t_next[2] {
            0
        } else if t_next[1] <= t_next[2] {
            1
        } else {
            2
        };
        let t_end = t_next[axis].min(t1);
        let seg = (t_end - t) * total_mm;
        if seg > 0.0
            && idx[0] >= 0
            && idx[1] >= 0
            && idx[2] >= 0
            && idx[0] < vol.dims[0] as i64
            && idx[1] < vol.dims[1] as i64
            && idx[2] < vol.dims[2] as i64
        {
            let v = vol.index(idx[0] as usize, idx[1] as usize, idx[2] as usize) as f32;
            acc += seg as f32 * attenuation(v, mode, threshold);
        }
        t = t_end;
        idx[axis] += step[axis];
        t_next[axis] += dt[axis];
        if idx[axis] < 0 || idx[axis] >= vol.dims[axis] as i64 {
            break;
        }
    }
    acc
}

/// Fixed-step line integral with trilinear interpolation (ITK-style).
fn raycast(vol: &Volume, src: Vec3, dst: Vec3, mode: HuMode, threshold: f32, step_mm: f64) -> f32 {
    let a = vol.patient_to_voxel(src);
    let b = vol.patient_to_voxel(dst);
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let Some((t0, t1)) = clip_to_volume(a, d, vol.dims) else {
        return 0.0;
    };
    let dir = dst - src;
    let total_mm = dir.length();
    let span_mm = (t1 - t0) * total_mm;
    if span_mm <= 0.0 {
        return 0.0;
    }
    let n = ((span_mm / step_mm.max(1e-3)).ceil() as usize).max(1);
    let h = span_mm / n as f64;
    let mut acc = 0.0f32;
    for s in 0..n {
        // Midpoint of the step: the ITK ray-cast integrates the interpolated
        // field, and a midpoint rule is what makes that second-order.
        let t = t0 + (s as f64 + 0.5) * (t1 - t0) / n as f64;
        let p = src + dir * t;
        if let Some(v) = vol.sample_patient(p) {
            acc += h as f32 * attenuation(v, mode, threshold);
        }
    }
    acc
}

/// Render one radiograph.
pub fn render(vol: &Volume, params: &DrrParams, sink: &dyn ProgressSink) -> Result<DrrImage> {
    let t_start = std::time::Instant::now();
    let g = &params.geometry;
    let [w, h] = g.dims;
    if w == 0 || h == 0 {
        bail!("the detector has no pixels");
    }
    if g.sid <= 0.0 || g.sad <= 0.0 {
        bail!("source-to-axis and source-to-imager distances must be positive");
    }
    let frame = g.frame();
    let px = g.pixel_mm();
    let engine = params.engine;
    let mode = params.hu;
    let threshold = params.threshold_hu;
    let step = params.step_mm;

    let rows: Vec<(usize, Vec<f32>)> = (0..h)
        .into_par_iter()
        .map(|j| {
            let mut row = vec![0.0f32; w];
            if sink.cancelled() {
                return (j, row);
            }
            // Pixel centres, +y running down the image as a detector row does.
            let dy = (j as f64 + 0.5 - h as f64 * 0.5) * px[1];
            for (i, out) in row.iter_mut().enumerate() {
                let dx = (i as f64 + 0.5 - w as f64 * 0.5) * px[0];
                let target = frame.panel_center + frame.panel_x * dx - frame.panel_y * dy;
                *out = match engine {
                    Engine::Siddon => siddon(vol, frame.source, target, mode, threshold),
                    Engine::RayCast => raycast(vol, frame.source, target, mode, threshold, step),
                };
            }
            (j, row)
        })
        .collect();
    if sink.cancelled() {
        bail!(CANCELLED);
    }

    let mut pixels = vec![0.0f32; w * h];
    for (j, row) in rows {
        pixels[j * w..(j + 1) * w].copy_from_slice(&row);
    }
    let min = pixels.iter().cloned().fold(f32::MAX, f32::min);
    let max = pixels.iter().cloned().fold(f32::MIN, f32::max);
    sink.report(1.0, "done");
    Ok(DrrImage {
        pixels,
        dims: [w, h],
        spacing: px,
        min,
        max,
        engine,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::Quiet;

    /// A uniform cube of `value` HU inside a `n³` volume of air.
    fn slab(n: usize, spacing: f64, inner: std::ops::Range<usize>, value: i16) -> Volume {
        let mut data = vec![-1000i16; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    if inner.contains(&i) && inner.contains(&j) && inner.contains(&k) {
                        data[k * n * n + j * n + i] = value;
                    }
                }
            }
        }
        let half = (n as f64 - 1.0) * 0.5 * spacing;
        Volume {
            data,
            dims: [n, n, n],
            spacing: [spacing; 3],
            origin: Vec3::new(-half, -half, -half),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: -1000,
            max_value: value,
        }
    }

    /// A hand-built rendering, to check the tree hand-off without paying
    /// for a projection.
    fn fake_image(engine: Engine) -> DrrImage {
        DrrImage {
            pixels: vec![0.0, 1.0, 3.0, 4.0, 2.0, 0.0],
            dims: [3, 2],
            spacing: [0.5, 0.25],
            min: 0.0,
            max: 4.0,
            engine,
            elapsed_secs: 1.5,
        }
    }

    /// The planar image a DRR becomes must describe the same picture: same
    /// raster, same physical size, same value range — and the greyscale the
    /// window was showing.
    #[test]
    fn a_rendering_becomes_a_planar_image() {
        let params = DrrParams {
            geometry: Geometry {
                gantry_deg: 90.0,
                couch_deg: 0.0,
                ..Geometry::default()
            },
            engine: Engine::Siddon,
            ..DrrParams::for_volume(&slab(8, 1.0, 3..5, 0))
        };
        let im = fake_image(Engine::Siddon);

        let plain = im.to_planar(&params, false);
        assert_eq!((plain.cols, plain.rows), (3, 2), "columns × rows");
        assert_eq!(plain.spacing, [0.5, 0.25]);
        assert_eq!(plain.modality, "RTIMAGE");
        assert_eq!(plain.data, im.pixels, "values are the line integral itself");
        assert_eq!((plain.min_value, plain.max_value), (0.0, 4.0));
        assert_eq!(plain.window, (2.0, 4.0), "centre and width of the range");
        assert!(plain.label.contains("Siddon") && plain.label.contains("G 90°"));

        // Inverting mirrors the values about the middle of the range and
        // leaves the range itself alone.
        let flipped = im.to_planar(&params, true);
        assert_eq!(flipped.data, vec![4.0, 3.0, 1.0, 0.0, 2.0, 4.0]);
        assert_eq!(
            (flipped.min_value, flipped.max_value),
            (im.min, im.max),
            "the range is unchanged by the inversion"
        );

        // The geometry rides along, and the sampling step is only meaningful
        // for the engine that has one.
        let keys: Vec<&str> = plain.info.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"Geometry") && keys.contains(&"Isocentre"));
        assert!(!keys.contains(&"Step"), "the exact tracer has no step");
        let cast = fake_image(Engine::RayCast).to_planar(
            &DrrParams {
                engine: Engine::RayCast,
                ..params
            },
            true,
        );
        assert!(cast.info.iter().any(|(k, _)| k == "Step"));
    }

    #[test]
    fn the_beam_frame_follows_the_iec_convention() {
        let g = Geometry {
            isocenter: Vec3::ZERO,
            ..Geometry::default()
        };
        // Gantry 0: the source is above a supine patient, i.e. anterior (−y).
        let f = g.frame();
        assert!((f.source - Vec3::new(0.0, -1000.0, 0.0)).length() < 1e-9);
        assert!((f.axis - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-9);
        // The imager is 500 mm beyond the isocentre, on the far side.
        assert!((f.panel_center - Vec3::new(0.0, 500.0, 0.0)).length() < 1e-9);
        // Gantry 90: the source is at the patient's left (+x).
        let f = Geometry {
            gantry_deg: 90.0,
            ..g
        }
        .frame();
        assert!((f.source - Vec3::new(1000.0, 0.0, 0.0)).length() < 1e-6);
        // The detector axes stay orthonormal and perpendicular to the beam.
        for g in [0.0, 37.0, 90.0, 180.0, 270.0] {
            for c in [0.0, 45.0, -90.0] {
                let f = Geometry {
                    gantry_deg: g,
                    couch_deg: c,
                    ..Geometry::default()
                }
                .frame();
                assert!((f.panel_x.length() - 1.0).abs() < 1e-9);
                assert!((f.panel_y.length() - 1.0).abs() < 1e-9);
                assert!(f.panel_x.dot(f.panel_y).abs() < 1e-9);
                assert!(f.panel_x.dot(f.axis).abs() < 1e-9);
                assert!(f.panel_y.dot(f.axis).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn a_central_ray_through_water_integrates_to_the_known_optical_depth() {
        // 40 mm of water on the central axis: μ_water × 40 mm.
        let n = 41;
        let spacing = 2.0;
        let vol = slab(n, spacing, 10..30, 0);
        let g = Geometry {
            dims: [3, 3],
            panel_mm: [3.0, 3.0],
            ..Geometry::centered_on(&vol)
        };
        let params = DrrParams {
            geometry: g,
            hu: HuMode::Water,
            threshold_hu: -500.0,
            step_mm: 0.25,
            engine: Engine::Siddon,
        };
        let exact = render(&vol, &params, &Quiet).unwrap();
        // The centre pixel of a 3 × 3 detector.
        let centre = exact.pixels[4];
        let want = (MU_WATER * 40.0) as f32;
        assert!(
            (centre - want).abs() < 0.02 * want,
            "Siddon centre {centre} vs {want}"
        );
        let marched = render(
            &vol,
            &DrrParams {
                engine: Engine::RayCast,
                ..params
            },
            &Quiet,
        )
        .unwrap();
        let centre2 = marched.pixels[4];
        assert!(
            (centre2 - want).abs() < 0.03 * want,
            "ray-cast centre {centre2} vs {want}"
        );
    }

    #[test]
    fn the_two_engines_agree_on_the_same_geometry() {
        let vol = slab(48, 3.0, 12..36, 300);
        let params = DrrParams {
            geometry: Geometry {
                dims: [48, 48],
                panel_mm: [300.0, 300.0],
                gantry_deg: 35.0,
                ..Geometry::centered_on(&vol)
            },
            engine: Engine::Siddon,
            hu: HuMode::Water,
            step_mm: 0.5,
            threshold_hu: -800.0,
        };
        let a = render(&vol, &params, &Quiet).unwrap();
        let b = render(
            &vol,
            &DrrParams {
                engine: Engine::RayCast,
                ..params
            },
            &Quiet,
        )
        .unwrap();
        let c = DrrComparison::of(&a, &b).expect("same size");
        eprintln!("drr: {} | {} | {}", a.describe(), b.describe(), c.line());
        assert!(
            c.correlation > 0.999,
            "the two engines disagree: {}",
            c.line()
        );
        assert!(c.relative < 0.03, "{}", c.line());
        assert!(a.max > 0.0 && b.max > 0.0);
        assert!(DrrComparison::of(
            &a,
            &render(
                &vol,
                &DrrParams {
                    geometry: Geometry {
                        dims: [8, 8],
                        ..params.geometry
                    },
                    ..params
                },
                &Quiet
            )
            .unwrap()
        )
        .is_none());
    }

    #[test]
    fn a_ray_that_misses_the_volume_integrates_to_nothing() {
        let vol = slab(24, 2.0, 8..16, 0);
        let params = DrrParams {
            geometry: Geometry {
                // An odd pixel count so one ray really is the central one,
                // and a panel far wider than the volume so the corners miss.
                dims: [33, 33],
                panel_mm: [2000.0, 2000.0],
                ..Geometry::centered_on(&vol)
            },
            ..DrrParams::for_volume(&vol)
        };
        let img = render(&vol, &params, &Quiet).unwrap();
        assert_eq!(img.pixels[0], 0.0, "the corner ray hit something");
        assert!(img.max > 0.0, "the central ray hit nothing");
        assert!(img.min == 0.0);
        assert!(img.mean() > 0.0);
        assert!(img.describe().contains("33 × 33"));
    }

    #[test]
    fn zero_sized_detectors_and_distances_are_refused() {
        let vol = slab(8, 2.0, 2..6, 0);
        let mut p = DrrParams::for_volume(&vol);
        p.geometry.dims = [0, 8];
        assert!(render(&vol, &p, &Quiet).is_err());
        let mut p = DrrParams::for_volume(&vol);
        p.geometry.sad = 0.0;
        assert!(render(&vol, &p, &Quiet).is_err());
    }
}
