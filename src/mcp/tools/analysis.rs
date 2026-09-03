//! Comparing structures, dose-volume histograms, and reports.

use anyhow::{bail, Context, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::Core;
use super::fourd::report_json;
use super::session::{round1, round2, round3, vec3};
use crate::dvh;
use crate::motion;
use crate::progress::Progress;

/// A structure of a dataset, and the lattice it is measured on.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StructureRef {
    pub dataset: String,
    pub structure: String,
    #[serde(default)]
    pub set: Option<String>,
    /// Series whose lattice the structure is rasterized on; the displayed
    /// series when omitted.
    #[serde(default)]
    pub series: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CompareArgs {
    pub a: StructureRef,
    /// Measured on `a`'s lattice: the two need one frame of reference.
    pub b: StructureRef,
}

pub fn compare_structures(core: &mut Core, args: CompareArgs, p: &Progress) -> Result<Value> {
    let a_idx = core
        .session
        .series_index(core.session.dataset(&args.a.dataset)?, args.a.series)?;
    let b_idx = core
        .session
        .series_index(core.session.dataset(&args.b.dataset)?, args.b.series)?;
    let vol_a = core.session.volume(&args.a.dataset, a_idx, p)?;
    let vol_b = core.session.volume(&args.b.dataset, b_idx, p)?;
    let grid = vol_a.grid();
    let sa = core
        .session
        .structure(&args.a.dataset, &args.a.structure, args.a.set.as_deref())?;
    let sb = core
        .session
        .structure(&args.b.dataset, &args.b.structure, args.b.set.as_deref())?;
    let ma = sa.mask_on(&grid)?;
    let mb = sb.mask_on(&grid).context("b, rasterized on a's lattice")?;
    let same_for = vol_a.frame_of_reference_uid == vol_b.frame_of_reference_uid;
    let ov = motion::overlap(&ma, &mb, &grid).context("one of the masks is empty")?;
    let mut out = json!({
        "a": { "dataset": args.a.dataset, "structure": clean_text(&sa.name), "volume_cm3": round1(ov.vol_a_cm3) },
        "b": { "dataset": args.b.dataset, "structure": clean_text(&sb.name), "volume_cm3": round1(ov.vol_b_cm3) },
        "dice": round3(ov.dice),
        "hd95_mm": round2(ov.hd95_mm),
        "mean_surface_distance_mm": round2(ov.msd_mm),
        "centroid_shift_mm": ov.centroid_shift().map(vec3),
        "centroid_shift_norm_mm": ov.centroid_shift().map(|d| round2((d.x * d.x + d.y * d.y + d.z * d.z).sqrt())),
        "same_frame_of_reference": same_for,
    });
    if !same_for {
        out["warning"] = json!(
            "the two series are in different frames of reference; the comparison assumes their \
             patient coordinates coincide, which they do only after a registration"
        );
    }
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DvhArgs {
    pub dataset: String,
    /// Dose number (from describe_dataset); the first dose when omitted.
    #[serde(default)]
    pub dose: Option<u32>,
    /// Structures to evaluate. Empty: every structure of the dataset.
    #[serde(default)]
    pub structures: Vec<super::register::StructureArg>,
    /// Series whose lattice the structures are sampled on; the displayed
    /// series when omitted.
    #[serde(default)]
    pub series: Option<u32>,
    /// Metrics such as `D95%`, `D2cc`, `V20Gy`, `V20Gy[cc]`, `Dmean`, `Dmax`,
    /// `Dmin`. Empty: the default set.
    #[serde(default)]
    pub metrics: Vec<String>,
    /// A protocol, one constraint per line: `Heart Dmean < 5`, `PTV D95% >= 25`,
    /// wildcards allowed in the structure name (`Lung*`).
    #[serde(default)]
    pub protocol: Option<String>,
    /// Bin width in dose units; derived from the dose maximum when omitted.
    #[serde(default)]
    pub bin_width: Option<f64>,
    /// Include the cumulative curves as CSV text.
    #[serde(default)]
    pub include_curves: bool,
}

pub fn compute_dvh(core: &mut Core, a: DvhArgs, p: &Progress) -> Result<Value> {
    let (dose_idx, series_idx, names) = {
        let ds = core.session.dataset(&a.dataset)?;
        if ds.study.doses.is_empty() {
            bail!("{} holds no dose grid", ds.id);
        }
        let dose_idx = match a.dose {
            None => 0,
            Some(n) => {
                let i = (n as usize)
                    .checked_sub(1)
                    .context("dose numbers start at 1")?;
                if i >= ds.study.doses.len() {
                    bail!(
                        "{} has {} dose grids; there is no dose {n}",
                        ds.id,
                        ds.study.doses.len()
                    );
                }
                i
            }
        };
        let series_idx = core.session.series_index(ds, a.series)?;
        let names: Vec<(String, Option<String>)> = if a.structures.is_empty() {
            crate::workflow::select::list(&ds.study)
                .into_iter()
                .map(|e| (e.name, Some(e.set_label)))
                .collect()
        } else {
            a.structures
                .iter()
                .map(|s| (s.structure.clone(), s.set.clone()))
                .collect()
        };
        (dose_idx, series_idx, names)
    };
    let mut metrics = Vec::new();
    for m in &a.metrics {
        metrics.push(
            dvh::Metric::parse(m).with_context(|| format!("metric '{m}' is not understood"))?,
        );
    }
    if metrics.is_empty() {
        metrics = dvh::default_metrics();
    }
    let grid = core.session.grid(&a.dataset, series_idx, p)?;
    let params = dvh::DvhParams {
        bin_width: a.bin_width,
    };
    let mut curves = Vec::new();
    let mut failed = Vec::new();
    for (i, (name, set)) in names.iter().enumerate() {
        p.set(format!("DVH of {name} ({}/{})", i + 1, names.len()));
        let s = core.session.structure(&a.dataset, name, set.as_deref())?;
        let mask = match s.mask_on(&grid) {
            Ok(m) => m,
            Err(e) => {
                failed.push(json!({"structure": clean_text(name), "reason": format!("{e:#}")}));
                continue;
            }
        };
        let ds = core.session.dataset(&a.dataset)?;
        let dose = &ds.study.doses[dose_idx];
        match dvh::compute(&s.name, s.color, &mask, &grid, dose, params) {
            Ok(c) => curves.push(c),
            Err(e) => {
                failed.push(json!({"structure": clean_text(name), "reason": format!("{e:#}")}))
            }
        }
    }
    if curves.is_empty() {
        bail!("no structure could be evaluated: {failed:?}");
    }
    let units = curves[0].units.clone();
    let rows: Vec<Value> = curves
        .iter()
        .map(|c| {
            let vals: serde_json::Map<String, Value> = metrics
                .iter()
                .map(|m| (m.label(), json!(round2(m.evaluate(c)))))
                .collect();
            json!({
                "structure": clean_text(&c.name),
                "volume_cm3": round1(c.volume_cm3),
                "outside_dose_grid_cm3": round1(c.outside_cm3),
                "outside_fraction": round3(c.outside_fraction()),
                "min": round2(c.min),
                "mean": round2(c.mean),
                "max": round2(c.max),
                "metrics": vals,
            })
        })
        .collect();
    let mut out = json!({
        "dataset": a.dataset,
        "dose": dose_idx + 1,
        "units": units,
        "metrics": metrics.iter().map(|m| m.label()).collect::<Vec<_>>(),
        "structures": rows,
        "skipped": failed,
        "metrics_csv": dvh::metrics_csv(&curves, &metrics),
    });
    if let Some(text) = &a.protocol {
        let constraints = dvh::parse_protocol(text);
        if constraints.is_empty() {
            bail!("the protocol text holds no constraint that could be parsed");
        }
        let verdicts = dvh::check(&constraints, &curves);
        out["protocol"] = json!({
            "constraints": constraints.len(),
            "passed": verdicts.iter().filter(|v| v.pass).count(),
            "verdicts": verdicts.iter().map(|v| json!({
                "constraint": v.constraint.to_line(),
                "structure": clean_text(&v.structure),
                "value": v.value.map(round2),
                "pass": v.pass,
            })).collect::<Vec<_>>(),
        });
    }
    if a.include_curves {
        out["curves_csv"] = json!(dvh::curves_csv(&curves, true));
    }
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunArgs {
    /// A motion run handle such as `run1`.
    pub run: String,
    /// Include the long-format CSV of the report.
    #[serde(default = "yes")]
    pub include_csv: bool,
}

fn yes() -> bool {
    true
}

pub fn motion_report(core: &mut Core, a: RunArgs, _p: &Progress) -> Result<Value> {
    let r = core.session.run(&a.run)?;
    let mut out = json!({
        "run": r.id,
        "dataset": r.dataset,
        "report": report_json(&r.report),
    });
    if a.include_csv {
        out["csv"] = json!(r.report.csv());
    }
    Ok(out)
}
