//! Dose-volume histograms: the numbers a plan is actually judged on.
//!
//! A DVH answers one question - how much of this structure receives at least
//! this dose - and every constraint in every protocol is a reading off that
//! curve. The arithmetic is elementary; what makes an implementation right or
//! wrong is the bookkeeping around it, and this module is deliberate about
//! four things that are easy to get quietly wrong:
//!
//! * **Where it samples.** The structure's own lattice, not the dose grid.
//!   A CT mask is 1 mm and a dose grid is 2-3 mm, so walking the mask and
//!   interpolating the dose gives a curve with the structure's resolution
//!   rather than the dose's. The walk is affine, so the dose grid
//!   coordinates are stepped rather than recomputed per voxel.
//!
//! * **What falls outside the dose grid.** Counted, kept, and reported.
//!   Those voxels enter the histogram at zero dose - which is the honest
//!   reading of "this part of the structure was not irradiated by *this*
//!   dose object" - but a DVH silently computed over 60 % of a structure is
//!   a wrong DVH, so [`Dvh::outside_fraction`] exists and the interface
//!   shows it.
//!
//! * **Statistics from the samples, not the bins.** Minimum, mean and
//!   maximum are accumulated during the walk. Reading them off a binned
//!   histogram costs half a bin width of accuracy for no reason.
//!
//! * **Interpolation inside a bin.** D95 % is almost never exactly at a bin
//!   edge. The cumulative curve is interpolated linearly between edges, so a
//!   finer bin width changes the answer by less than the bin width rather
//!   than by a whole one.
//!
//! Nothing here knows about the interface: a [`Dvh`] is computed from a mask,
//! a lattice and a [`DoseGrid`], and everything else - metrics, constraints,
//! CSV - is a pure function of it.

use std::fmt::Write as _;

use anyhow::{bail, Result};

use crate::rtdose::DoseGrid;
use crate::volume::Grid;

/// How finely the histogram is binned, as a fraction of the dose maximum.
/// 2000 bins over a 70 Gy plan is 3.5 cGy - finer than any constraint is
/// quoted to, and small enough that interpolation inside a bin is a
/// formality.
const DEFAULT_BINS: usize = 2000;

/// One structure's dose-volume histogram.
#[derive(Clone, Debug)]
pub struct Dvh {
    pub name: String,
    pub color: [u8; 3],
    /// Which dose object this was computed against - shown in the legend,
    /// because overlaying two plans is the whole point of allowing more than
    /// one.
    pub dose_label: String,
    /// Dose units of the source grid, "GY" or "RELATIVE".
    pub units: String,
    /// Volume in cm³ per bin; bin `b` covers dose `[b·w, (b+1)·w)`.
    pub bins: Vec<f64>,
    pub bin_width: f64,
    /// Volume of the whole structure, including anything outside the dose
    /// grid, in cm³.
    pub volume_cm3: f64,
    /// Volume that fell outside the dose grid, in cm³. It is in `bins[0]`.
    pub outside_cm3: f64,
    /// Over the voxels that *were* inside the dose grid.
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

impl Dvh {
    /// Fraction of the structure that lay outside the dose grid, 0‥1.
    pub fn outside_fraction(&self) -> f64 {
        if self.volume_cm3 <= 0.0 {
            0.0
        } else {
            self.outside_cm3 / self.volume_cm3
        }
    }

    /// Dose at the upper edge of the last non-empty bin - where the curve
    /// stops being worth drawing.
    pub fn dose_extent(&self) -> f64 {
        let last = self.bins.iter().rposition(|v| *v > 0.0).unwrap_or(0);
        (last + 1) as f64 * self.bin_width
    }

    /// The cumulative curve: `(dose, volume ≥ that dose in cm³)` at every bin
    /// edge, starting at `(0, whole structure)`.
    pub fn cumulative(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(self.bins.len() + 1);
        let mut remaining = self.bins.iter().sum::<f64>();
        for (b, v) in self.bins.iter().enumerate() {
            out.push((b as f64 * self.bin_width, remaining));
            remaining -= *v;
        }
        out.push((self.bins.len() as f64 * self.bin_width, remaining.max(0.0)));
        out
    }

    /// The differential curve: `(bin centre, volume in that bin in cm³)`.
    pub fn differential(&self) -> Vec<(f64, f64)> {
        self.bins
            .iter()
            .enumerate()
            .map(|(b, v)| ((b as f64 + 0.5) * self.bin_width, *v))
            .collect()
    }

    /// Volume receiving at least `dose`, in cm³. Linear inside a bin.
    pub fn volume_at_dose(&self, dose: f64) -> f64 {
        if dose <= 0.0 {
            return self.bins.iter().sum();
        }
        let x = dose / self.bin_width;
        let b = x.floor() as usize;
        if b >= self.bins.len() {
            return 0.0;
        }
        // Volume at or above the bin's lower edge, less the part of this
        // bin's own volume that lies below `dose`.
        let above: f64 = self.bins[b..].iter().sum();
        let frac_into_bin = x - b as f64;
        (above - self.bins[b] * frac_into_bin).max(0.0)
    }

    /// Volume receiving at least `dose`, as a fraction of the structure.
    pub fn volume_fraction_at_dose(&self, dose: f64) -> f64 {
        if self.volume_cm3 <= 0.0 {
            0.0
        } else {
            self.volume_at_dose(dose) / self.volume_cm3
        }
    }

    /// The dose that `volume_cm3` of the structure receives at least - D2cc
    /// and friends. Interpolated on the cumulative curve.
    ///
    /// Returns 0 for a volume larger than the structure, which is the honest
    /// answer: every part of it receives at least nothing.
    pub fn dose_at_volume(&self, volume_cm3: f64) -> f64 {
        let total: f64 = self.bins.iter().sum();
        if volume_cm3 <= 0.0 {
            return self.dose_extent();
        }
        if volume_cm3 >= total {
            return 0.0;
        }
        // Walk down from the top until the accumulated volume reaches the
        // target, then interpolate inside the bin it happened in.
        let mut acc = 0.0;
        for b in (0..self.bins.len()).rev() {
            let v = self.bins[b];
            if acc + v >= volume_cm3 {
                if b == 0 {
                    // The lowest bin holds exact zeros - voxels outside the
                    // dose grid, and anything the plan genuinely misses.
                    // Interpolating across it would report a few hundredths
                    // of a Gy for a structure half of which is unirradiated,
                    // which reads as a real dose and is not one.
                    return 0.0;
                }
                let need = volume_cm3 - acc;
                let frac = if v > 0.0 { need / v } else { 0.0 };
                // `frac` of this bin, measured down from its upper edge.
                return (b as f64 + 1.0 - frac) * self.bin_width;
            }
            acc += v;
        }
        0.0
    }

    /// The dose that `fraction` (0‥1) of the structure receives at least -
    /// D95 %, D2 %.
    pub fn dose_at_volume_fraction(&self, fraction: f64) -> f64 {
        self.dose_at_volume(fraction * self.volume_cm3)
    }
}

/// Everything one DVH run is told about how to bin.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DvhParams {
    /// Bin width in dose units. `None` derives it from the dose maximum.
    pub bin_width: Option<f64>,
}

/// Compute one structure's DVH.
///
/// `mask` is a 0/1 mask on `grid`; `dose` is sampled at the patient position
/// of every set voxel's centre.
pub fn compute(
    name: &str,
    color: [u8; 3],
    mask: &[u8],
    grid: &Grid,
    dose: &DoseGrid,
    params: DvhParams,
) -> Result<Dvh> {
    let [nx, ny, nz] = grid.dims;
    if mask.len() != nx * ny * nz {
        bail!("the mask does not match its lattice");
    }
    let voxel_cm3 = grid.spacing[0] * grid.spacing[1] * grid.spacing[2] / 1000.0;
    let top = (dose.max_dose as f64).max(1e-6);
    let bin_width = params.bin_width.unwrap_or(top / DEFAULT_BINS as f64);
    if !bin_width.is_finite() || bin_width <= 0.0 {
        bail!("the bin width must be a positive number of dose units");
    }
    // One bin past the maximum, so the hottest voxel has somewhere to land.
    let n_bins = ((top / bin_width).ceil() as usize + 1).max(2);
    let mut bins = vec![0.0f64; n_bins];

    // The dose grid's coordinates are an affine function of position, so the
    // walk steps them instead of projecting each voxel: three adds per voxel
    // rather than three dot products.
    let base = dose.grid_coords(grid.voxel_to_patient(0.0, 0.0, 0.0));
    let step_i = delta(dose, grid, 0, base);
    let step_j = delta(dose, grid, 1, base);
    let step_k = delta(dose, grid, 2, base);

    let (mut n_in, mut n_out) = (0u64, 0u64);
    let (mut lo, mut hi, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
    for k in 0..nz {
        for j in 0..ny {
            let row = k * nx * ny + j * nx;
            let start = [
                base[0] + step_j[0] * j as f64 + step_k[0] * k as f64,
                base[1] + step_j[1] * j as f64 + step_k[1] * k as f64,
                base[2] + step_j[2] * j as f64 + step_k[2] * k as f64,
            ];
            for i in 0..nx {
                if mask[row + i] == 0 {
                    continue;
                }
                let uvw = [
                    start[0] + step_i[0] * i as f64,
                    start[1] + step_i[1] * i as f64,
                    start[2] + step_i[2] * i as f64,
                ];
                match dose.sample_uvw(uvw) {
                    Some(d) => {
                        let d = d as f64;
                        n_in += 1;
                        sum += d;
                        lo = lo.min(d);
                        hi = hi.max(d);
                        let b = ((d / bin_width) as usize).min(n_bins - 1);
                        bins[b] += voxel_cm3;
                    }
                    None => {
                        n_out += 1;
                        bins[0] += voxel_cm3;
                    }
                }
            }
        }
    }
    if n_in + n_out == 0 {
        bail!("'{name}' has no voxels on this lattice");
    }
    Ok(Dvh {
        name: name.to_string(),
        color,
        dose_label: dose.label.clone(),
        units: dose.units.clone(),
        bins,
        bin_width,
        volume_cm3: (n_in + n_out) as f64 * voxel_cm3,
        outside_cm3: n_out as f64 * voxel_cm3,
        min: if n_in > 0 { lo } else { 0.0 },
        max: if n_in > 0 { hi } else { 0.0 },
        mean: if n_in > 0 { sum / n_in as f64 } else { 0.0 },
    })
}

/// How the dose grid coordinates change per step of one lattice axis.
fn delta(dose: &DoseGrid, grid: &Grid, axis: usize, base: [f64; 3]) -> [f64; 3] {
    let one = match axis {
        0 => grid.voxel_to_patient(1.0, 0.0, 0.0),
        1 => grid.voxel_to_patient(0.0, 1.0, 0.0),
        _ => grid.voxel_to_patient(0.0, 0.0, 1.0),
    };
    let c = dose.grid_coords(one);
    [c[0] - base[0], c[1] - base[1], c[2] - base[2]]
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// One readable number off a curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    Volume,
    Min,
    Mean,
    Max,
    /// Dose to at least this percentage of the structure - `D95%`.
    DoseAtPct(f64),
    /// Dose to at least this absolute volume - `D2cc`.
    DoseAtCc(f64),
    /// Percentage of the structure at or above this dose - `V20Gy`.
    VolumePctAtDose(f64),
    /// Absolute volume at or above this dose - `V20Gy[cc]`.
    VolumeCcAtDose(f64),
}

impl Metric {
    /// `D95%`, `V20`, `Dmean` - the column heading, and what the protocol
    /// file writes (`V20Gy` is accepted on input).
    pub fn label(&self) -> String {
        match self {
            Metric::Volume => "Volume".into(),
            Metric::Min => "Dmin".into(),
            Metric::Mean => "Dmean".into(),
            Metric::Max => "Dmax".into(),
            Metric::DoseAtPct(p) => format!("D{}%", trim(*p)),
            Metric::DoseAtCc(v) => format!("D{}cc", trim(*v)),
            Metric::VolumePctAtDose(d) => format!("V{}", trim(*d)),
            Metric::VolumeCcAtDose(d) => format!("V{}cc", trim(*d)),
        }
    }

    /// What the number is in: dose units, cm³ or per cent.
    pub fn unit(&self, dose_units: &str) -> String {
        match self {
            Metric::Volume | Metric::VolumeCcAtDose(_) => "cm³".into(),
            Metric::VolumePctAtDose(_) => "%".into(),
            _ => nice_units(dose_units),
        }
    }

    /// True when the value is a dose, and so follows the dose axis when the
    /// window is showing per cent of the prescription.
    pub fn is_dose(&self) -> bool {
        matches!(
            self,
            Metric::Min | Metric::Mean | Metric::Max | Metric::DoseAtPct(_) | Metric::DoseAtCc(_)
        )
    }

    pub fn evaluate(&self, dvh: &Dvh) -> f64 {
        match *self {
            Metric::Volume => dvh.volume_cm3,
            Metric::Min => dvh.min,
            Metric::Mean => dvh.mean,
            Metric::Max => dvh.max,
            Metric::DoseAtPct(p) => dvh.dose_at_volume_fraction(p / 100.0),
            Metric::DoseAtCc(v) => dvh.dose_at_volume(v),
            Metric::VolumePctAtDose(d) => dvh.volume_fraction_at_dose(d) * 100.0,
            Metric::VolumeCcAtDose(d) => dvh.volume_at_dose(d),
        }
    }

    /// Parse a column or constraint name: `Dmean`, `D95%`, `D2cc`, `V20Gy`,
    /// `V20Gy[cc]`. Case-insensitive.
    pub fn parse(s: &str) -> Option<Metric> {
        let t = s.trim();
        let lower = t.to_lowercase();
        match lower.as_str() {
            "volume" | "vol" => return Some(Metric::Volume),
            "dmin" | "min" => return Some(Metric::Min),
            "dmean" | "mean" => return Some(Metric::Mean),
            "dmax" | "max" => return Some(Metric::Max),
            _ => {}
        }
        // Not `split_at(1)`: that panics on an empty string and on any
        // first character wider than one byte, and this is fed straight from
        // a text box the user is still typing in.
        let mut chars = lower.chars();
        let head = chars.next()?;
        let rest = chars.as_str();
        match head {
            'd' => {
                if let Some(v) = rest.strip_suffix("cc") {
                    v.trim().parse().ok().map(Metric::DoseAtCc)
                } else {
                    rest.trim_end_matches('%')
                        .trim()
                        .parse()
                        .ok()
                        .map(Metric::DoseAtPct)
                }
            }
            'v' => {
                // Both the label form (`V20cc`) and the explicit one
                // (`V20Gy[cc]`); `label()` emits the first, so the two have
                // to agree or a protocol cannot survive being saved.
                let (body, absolute) = match rest.strip_suffix("[cc]") {
                    Some(b) => (b, true),
                    None => match rest.strip_suffix("cc") {
                        Some(b) => (b, true),
                        None => (rest, false),
                    },
                };
                let d: f64 = body
                    .trim_end_matches("gy")
                    .trim_end_matches('%')
                    .trim()
                    .parse()
                    .ok()?;
                Some(if absolute {
                    Metric::VolumeCcAtDose(d)
                } else {
                    Metric::VolumePctAtDose(d)
                })
            }
            _ => None,
        }
    }
}

/// The columns every table starts with.
pub fn default_metrics() -> Vec<Metric> {
    vec![
        Metric::Volume,
        Metric::Min,
        Metric::Mean,
        Metric::Max,
        Metric::DoseAtPct(95.0),
        Metric::DoseAtPct(2.0),
    ]
}

fn trim(v: f64) -> String {
    let s = format!("{v:.2}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() {
        "0".into()
    } else {
        s
    }
}

/// "GY" as it should be printed.
pub fn nice_units(units: &str) -> String {
    match units.to_uppercase().as_str() {
        "GY" => "Gy".into(),
        "RELATIVE" => "%".into(),
        "" => "Gy".into(),
        other => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Constraints
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    AtMost,
    AtLeast,
}

impl Cmp {
    pub fn symbol(&self) -> &'static str {
        match self {
            Cmp::AtMost => "<=",
            Cmp::AtLeast => ">=",
        }
    }
    fn holds(&self, value: f64, limit: f64) -> bool {
        match self {
            Cmp::AtMost => value <= limit,
            Cmp::AtLeast => value >= limit,
        }
    }
}

/// One line of a protocol: *this structure's* `metric` must be at most or at
/// least `limit`.
#[derive(Clone, Debug, PartialEq)]
pub struct Constraint {
    /// Matched against the structure name, case-insensitively; a leading or
    /// trailing `*` matches loosely, so `PTV*` catches `PTV_5400`.
    pub structure: String,
    pub metric: Metric,
    pub cmp: Cmp,
    pub limit: f64,
}

impl Constraint {
    pub fn matches(&self, name: &str) -> bool {
        let (pat, name) = (self.structure.to_lowercase(), name.to_lowercase());
        match (pat.strip_prefix('*'), pat.strip_suffix('*')) {
            (Some(p), Some(_)) => {
                let inner = p.strip_suffix('*').unwrap_or(p);
                name.contains(inner)
            }
            (Some(p), None) => name.ends_with(p),
            (None, Some(p)) => name.starts_with(p),
            (None, None) => name == pat,
        }
    }

    /// `Cord Dmax <= 45` - the line a protocol file holds. A name with a
    /// space in it is quoted, because that is how it is read back.
    pub fn to_line(&self) -> String {
        let name = if self.structure.contains(char::is_whitespace) {
            format!("\"{}\"", self.structure)
        } else {
            self.structure.clone()
        };
        format!(
            "{name} {} {} {}",
            self.metric.label(),
            self.cmp.symbol(),
            trim(self.limit)
        )
    }
}

/// How one constraint came out.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub constraint: Constraint,
    /// The structure it was matched against; empty when nothing matched.
    pub structure: String,
    pub value: Option<f64>,
    pub pass: bool,
}

/// Evaluate a protocol against the curves on screen.
///
/// A constraint that matches nothing is reported with no value and does not
/// pass - a protocol line that silently evaluates to "fine" because the
/// structure was never contoured is the worst possible failure mode.
pub fn check(constraints: &[Constraint], curves: &[Dvh]) -> Vec<Verdict> {
    constraints
        .iter()
        .map(|c| match curves.iter().find(|d| c.matches(&d.name)) {
            Some(d) => {
                let value = c.metric.evaluate(d);
                Verdict {
                    constraint: c.clone(),
                    structure: d.name.clone(),
                    value: Some(value),
                    pass: c.cmp.holds(value, c.limit),
                }
            }
            None => Verdict {
                constraint: c.clone(),
                structure: String::new(),
                value: None,
                pass: false,
            },
        })
        .collect()
}

/// Read a protocol: one constraint per line, `#` comments, blank lines
/// ignored. Structure names may be quoted when they contain spaces.
pub fn parse_protocol(text: &str) -> Vec<Constraint> {
    text.lines()
        .filter_map(|line| {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                return None;
            }
            let (structure, rest) = if let Some(r) = line.strip_prefix('"') {
                let (name, rest) = r.split_once('"')?;
                (name.to_string(), rest)
            } else {
                let (name, rest) = line.split_once(char::is_whitespace)?;
                (name.to_string(), rest)
            };
            let mut it = rest.split_whitespace();
            let metric = Metric::parse(it.next()?)?;
            let cmp = match it.next()? {
                "<=" | "<" => Cmp::AtMost,
                ">=" | ">" => Cmp::AtLeast,
                _ => return None,
            };
            let limit = it.next()?.parse().ok()?;
            Some(Constraint {
                structure,
                metric,
                cmp,
                limit,
            })
        })
        .collect()
}

/// Write a protocol back out, so what was loaded can be edited and saved.
pub fn write_protocol(constraints: &[Constraint]) -> String {
    let mut s = String::from("# One constraint per line: STRUCTURE METRIC <=|>= LIMIT\n");
    for c in constraints {
        let _ = writeln!(s, "{}", c.to_line());
    }
    s
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// The cumulative curves as CSV: one dose column, then one volume column per
/// structure. Curves may have different bin widths (different dose grids), so
/// they are resampled onto one dose axis rather than assumed to share one.
pub fn curves_csv(curves: &[Dvh], relative_volume: bool) -> String {
    let mut out = String::new();
    if curves.is_empty() {
        return out;
    }
    let width = curves
        .iter()
        .map(|c| c.bin_width)
        .fold(f64::INFINITY, f64::min);
    let extent = curves.iter().map(|c| c.dose_extent()).fold(0.0, f64::max);
    let steps = ((extent / width).ceil() as usize).clamp(1, 100_000);
    let units = nice_units(&curves[0].units);
    out.push_str(&format!("Dose [{units}]"));
    for c in curves {
        let _ = write!(
            out,
            ",{} [{}]{}",
            c.name,
            if relative_volume { "%" } else { "cm³" },
            if c.dose_label.is_empty() {
                String::new()
            } else {
                format!(" ({})", c.dose_label)
            }
        );
    }
    out.push('\n');
    for s in 0..=steps {
        let d = s as f64 * width;
        let _ = write!(out, "{d:.4}");
        for c in curves {
            let v = if relative_volume {
                c.volume_fraction_at_dose(d) * 100.0
            } else {
                c.volume_at_dose(d)
            };
            let _ = write!(out, ",{v:.4}");
        }
        out.push('\n');
    }
    out
}

/// The metrics table as CSV.
pub fn metrics_csv(curves: &[Dvh], metrics: &[Metric]) -> String {
    let mut out = String::from("Structure,Dose");
    for m in metrics {
        let _ = write!(
            out,
            ",{} [{}]",
            m.label(),
            m.unit(curves.first().map(|c| c.units.as_str()).unwrap_or("GY"))
        );
    }
    out.push('\n');
    for c in curves {
        let _ = write!(out, "{},{}", c.name, c.dose_label);
        for m in metrics {
            let _ = write!(out, ",{:.4}", m.evaluate(c));
        }
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    /// A dose grid that ramps linearly from 0 to `top` along +x, on a 1 mm
    /// lattice - every DVH over it can be worked out on paper.
    fn ramp(top: f32, nx: usize, ny: usize, nz: usize) -> DoseGrid {
        let mut data = vec![0f32; nx * ny * nz];
        for f in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    data[f * nx * ny + j * nx + i] = top * i as f32 / (nx - 1) as f32;
                }
            }
        }
        DoseGrid {
            data,
            dims: [nx, ny, nz],
            spacing: [1.0, 1.0],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            offsets: (0..nz).map(|f| f as f64).collect(),
            units: "GY".into(),
            summation_type: "PLAN".into(),
            max_dose: top,
            frame_of_reference_uid: "1.2.3".into(),
            study_uid: String::new(),
            referenced_plan_uid: String::new(),
            label: "test".into(),
        }
    }

    fn grid(dims: [usize; 3], spacing: [f64; 3]) -> Grid {
        Grid {
            dims,
            spacing,
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: "1.2.3".into(),
        }
    }

    /// A block covering `i` in `0..nx`, on the same lattice as the dose.
    fn block(g: &Grid) -> Vec<u8> {
        vec![1u8; g.dims[0] * g.dims[1] * g.dims[2]]
    }

    #[test]
    fn a_uniform_dose_gives_a_step_and_exact_statistics() {
        let mut dose = ramp(10.0, 11, 4, 3);
        dose.data.iter_mut().for_each(|v| *v = 7.0);
        dose.max_dose = 7.0;
        let g = grid([11, 4, 3], [1.0; 3]);
        let d = compute(
            "block",
            [255, 0, 0],
            &block(&g),
            &g,
            &dose,
            DvhParams::default(),
        )
        .expect("a curve");
        assert!((d.min - 7.0).abs() < 1e-6);
        assert!((d.max - 7.0).abs() < 1e-6);
        assert!((d.mean - 7.0).abs() < 1e-6);
        // 11 × 4 × 3 mm³ = 0.132 cm³.
        assert!((d.volume_cm3 - 0.132).abs() < 1e-9, "{}", d.volume_cm3);
        // Everything is above 6.9 Gy and nothing above 7.1.
        assert!((d.volume_at_dose(6.9) - d.volume_cm3).abs() < 1e-9);
        assert!(d.volume_at_dose(7.1) < 1e-9);
        assert!((d.dose_at_volume_fraction(0.5) - 7.0).abs() < 0.05);
    }

    #[test]
    fn a_linear_ramp_gives_a_linear_dvh() {
        // Dose runs 0‥10 Gy over 101 columns; the structure is all of them.
        // The fraction receiving ≥ D is then (10 − D)/10, exactly.
        let dose = ramp(10.0, 101, 3, 3);
        let g = grid([101, 3, 3], [1.0; 3]);
        let d = compute(
            "ramp",
            [0, 255, 0],
            &block(&g),
            &g,
            &dose,
            DvhParams::default(),
        )
        .expect("a curve");
        assert!((d.min - 0.0).abs() < 1e-6);
        assert!((d.max - 10.0).abs() < 1e-6);
        assert!((d.mean - 5.0).abs() < 1e-6);
        for probe in [1.0, 2.5, 5.0, 7.5, 9.0] {
            let want = (10.0 - probe) / 10.0;
            let got = d.volume_fraction_at_dose(probe);
            assert!(
                (got - want).abs() < 0.02,
                "V{probe}Gy = {got:.4}, expected {want:.4}"
            );
        }
        // …and the inverse reading agrees.
        for frac in [0.1, 0.5, 0.9] {
            let want = 10.0 * (1.0 - frac);
            let got = d.dose_at_volume_fraction(frac);
            assert!(
                (got - want).abs() < 0.15,
                "D{}% = {got:.3}, expected {want:.3}",
                frac * 100.0
            );
        }
    }

    #[test]
    fn voxels_outside_the_dose_grid_are_counted_and_reported() {
        // The dose covers 11 mm in x; the structure is twice as wide.
        let dose = ramp(10.0, 11, 4, 3);
        let g = grid([22, 4, 3], [1.0; 3]);
        let d = compute(
            "wide",
            [0, 0, 255],
            &block(&g),
            &g,
            &dose,
            DvhParams::default(),
        )
        .expect("a curve");
        assert!(
            (d.outside_fraction() - 0.5).abs() < 0.05,
            "outside {:.3}",
            d.outside_fraction()
        );
        // Half the structure is at zero dose, so D60% must be zero…
        assert!(d.dose_at_volume_fraction(0.6) < 1e-6);
        // …while the statistics describe only what was irradiated.
        assert!((d.max - 10.0).abs() < 1e-6);
        assert!((d.mean - 5.0).abs() < 0.2, "mean {}", d.mean);
    }

    #[test]
    fn dose_at_volume_reads_the_curve_from_the_top() {
        let dose = ramp(10.0, 101, 3, 3);
        let g = grid([101, 3, 3], [1.0; 3]);
        let d =
            compute("ramp", [0; 3], &block(&g), &g, &dose, DvhParams::default()).expect("a curve");
        // D0 is the hottest dose; a volume beyond the structure is 0.
        assert!(d.dose_at_volume(0.0) >= 9.9);
        assert_eq!(d.dose_at_volume(d.volume_cm3 * 2.0), 0.0);
        // D2cc on a 0.909 cm³ structure is beyond it, so also zero.
        assert_eq!(d.dose_at_volume(2.0), 0.0);
    }

    #[test]
    fn metric_names_round_trip() {
        for (text, metric) in [
            ("Dmean", Metric::Mean),
            ("Dmax", Metric::Max),
            ("D95%", Metric::DoseAtPct(95.0)),
            ("D2cc", Metric::DoseAtCc(2.0)),
            ("V20", Metric::VolumePctAtDose(20.0)),
            ("V20cc", Metric::VolumeCcAtDose(20.0)),
        ] {
            let parsed = Metric::parse(text).unwrap_or_else(|| panic!("parse {text}"));
            assert_eq!(parsed, metric, "{text}");
        }
        assert_eq!(Metric::parse("V20Gy"), Some(Metric::VolumePctAtDose(20.0)));
        assert_eq!(Metric::DoseAtPct(95.0).label(), "D95%");
        assert_eq!(Metric::VolumeCcAtDose(20.0).label(), "V20cc");
        assert_eq!(Metric::parse("nonsense"), None);
    }

    #[test]
    fn a_half_typed_metric_is_refused_and_never_panics() {
        // The column box is parsed on every keystroke and on every click of
        // the + button, so anything that can be in a text field has to come
        // back as None rather than take the program with it.
        // The wide characters are written as escapes on purpose: the glyph
        // guard in `app::glyphs` reads every literal in `src` as something
        // the interface might draw, and these are input, not interface text.
        for text in [
            "",
            " ",
            "\t",
            "D",
            "V",
            "%",
            "cc",
            "Gy",
            "-",
            "\u{b5}",         // micro sign, two bytes
            "\u{d8}",         // capital O with stroke
            "\u{432}\u{43e}", // Cyrillic, two bytes each
            "\u{1f642}",      // an emoji, four bytes
            "D%",
            "Vcc",
            "D cc",
            "V[cc]",
        ] {
            assert_eq!(Metric::parse(text), None, "{text:?} is not a metric");
        }
    }

    #[test]
    fn a_protocol_round_trips_and_a_missing_structure_fails_loudly() {
        let text = "\
# head and neck
Cord      Dmax  <= 45
PTV*      D95%  >= 57
\"Parotid L\" Dmean <= 26
Missing   Dmean <= 10
";
        let cs = parse_protocol(text);
        assert_eq!(cs.len(), 4);
        assert_eq!(cs[2].structure, "Parotid L");
        assert!(cs[1].matches("PTV_5400"), "prefix wildcard");
        assert!(!cs[0].matches("Cord_PRV"));
        let round = parse_protocol(&write_protocol(&cs));
        assert_eq!(round, cs);

        let dose = ramp(60.0, 61, 3, 3);
        let g = grid([61, 3, 3], [1.0; 3]);
        let curves = vec![
            compute("Cord", [0; 3], &block(&g), &g, &dose, DvhParams::default()).unwrap(),
            compute(
                "PTV_5400",
                [0; 3],
                &block(&g),
                &g,
                &dose,
                DvhParams::default(),
            )
            .unwrap(),
            compute(
                "Parotid L",
                [0; 3],
                &block(&g),
                &g,
                &dose,
                DvhParams::default(),
            )
            .unwrap(),
        ];
        let v = check(&cs, &curves);
        assert_eq!(v[0].structure, "Cord");
        assert!(!v[0].pass, "Dmax 60 > 45");
        assert_eq!(v[1].structure, "PTV_5400");
        assert!(v[3].value.is_none(), "nothing matched 'Missing'");
        assert!(!v[3].pass, "an unmatched constraint must not pass");
    }

    #[test]
    fn csv_carries_one_dose_axis_for_curves_that_do_not_share_bins() {
        let g = grid([21, 2, 2], [1.0; 3]);
        let a = compute(
            "a",
            [0; 3],
            &block(&g),
            &g,
            &ramp(20.0, 21, 2, 2),
            DvhParams {
                bin_width: Some(0.5),
            },
        )
        .unwrap();
        let b = compute(
            "b",
            [0; 3],
            &block(&g),
            &g,
            &ramp(20.0, 21, 2, 2),
            DvhParams {
                bin_width: Some(0.1),
            },
        )
        .unwrap();
        let csv = curves_csv(&[a, b], true);
        let mut lines = csv.lines();
        assert!(lines.next().unwrap().starts_with("Dose [Gy],a [%]"));
        let first = lines.next().unwrap();
        assert!(first.starts_with("0.0000,100.0000,100.0000"), "{first}");
        // The finer of the two bin widths sets the step.
        let second = lines.next().unwrap();
        assert!(second.starts_with("0.1000"), "{second}");
    }
}
