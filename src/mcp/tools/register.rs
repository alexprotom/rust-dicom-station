//! Registration of one pair, and propagation through it.

use std::sync::Arc;

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::session::Registration;
use super::super::Core;
use super::session::{round1, round2, round3, vec3};
use crate::motion;
use crate::progress::Progress;
use crate::propagate;
use crate::registration::{self, Init, Metric, RegMethod, RegParams, RegionMask};

/// One side of a registration: a dataset and, optionally, a series of it.
#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Side {
    pub dataset: String,
    /// Series number; the displayed series when omitted.
    #[serde(default)]
    pub series: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegionArg {
    /// A structure of the *fixed* dataset.
    pub structure: String,
    #[serde(default)]
    pub set: Option<String>,
    /// Dilation of the structure that defines the region, mm.
    #[serde(default = "default_region_margin")]
    pub margin_mm: f64,
}

fn default_region_margin() -> f64 {
    10.0
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegisterArgs {
    pub fixed: Side,
    pub moving: Side,
    /// `elastix_rigid`, `elastix_bspline` or `plastimatch_bspline`.
    #[serde(default = "default_method")]
    pub method: String,
    /// Resolution levels (elastix) or stages (plastimatch).
    #[serde(default)]
    pub levels: Option<usize>,
    /// Iterations per level.
    #[serde(default)]
    pub iterations: Option<usize>,
    /// Spatial samples per iteration (elastix).
    #[serde(default)]
    pub samples: Option<usize>,
    /// B-spline control-point spacing, mm.
    #[serde(default)]
    pub grid_spacing_mm: Option<f64>,
    /// Fixed-image voxels below this value are not sampled (a crude body
    /// mask; CT default -500).
    #[serde(default)]
    pub fixed_threshold: Option<f32>,
    /// plastimatch: `mean_squares` or `mutual_information`.
    #[serde(default)]
    pub metric: Option<String>,
    /// plastimatch: bending-energy weight.
    #[serde(default)]
    pub regularization: Option<f64>,
    /// Restrict the run to a structure of the fixed dataset: a local
    /// registration (a rigid fit of one organ, or a deformable refinement).
    #[serde(default)]
    pub region: Option<RegionArg>,
    /// A `reg` handle of the same pair to refine rather than replace.
    #[serde(default)]
    pub start: Option<String>,
    /// Where the search starts: `auto` (the identity when the images
    /// overlap, otherwise their centres of gravity), `identity`,
    /// `center_of_gravity`, or a structure name contoured on both datasets
    /// (`init_moving` names it on the moving side when it differs) whose
    /// centroids are matched. Two images of one patient in different frames
    /// of reference do not overlap at the identity and need one of the
    /// latter.
    #[serde(default)]
    pub init: Option<String>,
    #[serde(default)]
    pub init_moving: Option<String>,
}

fn default_method() -> String {
    "elastix_rigid".into()
}

pub fn parse_method(s: &str) -> Result<RegMethod> {
    Ok(match s {
        "elastix_rigid" => RegMethod::ElastixRigid,
        "elastix_bspline" => RegMethod::ElastixBSpline,
        "plastimatch_bspline" => RegMethod::PlastimatchBSpline,
        other => bail!(
            "method must be elastix_rigid, elastix_bspline or plastimatch_bspline (got '{other}')"
        ),
    })
}

/// The registration analysis as JSON: displacement, rotation, Jacobian.
pub fn analysis_json(r: &registration::RegistrationResult) -> Value {
    let a = &r.analysis;
    json!({
        "method": r.method.label(),
        "metric": r.metric.tag(),
        "initial_metric": round2(r.initial_metric),
        "final_metric": round2(r.final_metric),
        "iterations": r.iterations_run,
        "elapsed_s": round1(r.elapsed_secs),
        "region": r.region.as_deref().map(clean_text),
        "translation_mm": vec3(a.dof.translation),
        "rotation_deg": a.dof.rotation_deg.map(round2),
        "rigid_residual_mm": round2(a.dof.residual_mm),
        "displacement_mm": {
            "mean": round2(a.displacement.mean),
            "p95": round2(a.displacement.p95),
            "max": round2(a.displacement.max),
            "rms": round2(a.displacement.rms),
        },
        "jacobian": {
            "min": round3(a.jacobian.min),
            "mean": round3(a.jacobian.mean),
            "max": round3(a.jacobian.max),
            "folded_fraction": round3(a.jacobian.folded),
        },
        "summary": r.metric_line(),
    })
}

pub fn register(core: &mut Core, a: RegisterArgs, p: &Progress) -> Result<Value> {
    let method = parse_method(&a.method)?;
    let fds = core.session.dataset(&a.fixed.dataset)?;
    let fixed_idx = core.session.series_index(fds, a.fixed.series)?;
    let mds = core.session.dataset(&a.moving.dataset)?;
    let moving_idx = core.session.series_index(mds, a.moving.series)?;
    if a.fixed.dataset == a.moving.dataset && fixed_idx == moving_idx {
        bail!("fixed and moving are the same series");
    }
    let fixed = core.session.volume(&a.fixed.dataset, fixed_idx, p)?;
    let moving = core.session.volume(&a.moving.dataset, moving_idx, p)?;

    let mut params = RegParams {
        method,
        ..RegParams::default()
    };
    if let Some(v) = a.levels {
        params.levels = v.clamp(1, 6);
    }
    if let Some(v) = a.iterations {
        params.iterations = v.clamp(1, 5000);
    }
    if let Some(v) = a.samples {
        params.samples = v.clamp(100, 200_000);
    }
    if let Some(v) = a.grid_spacing_mm {
        params.grid_spacing_mm = v.clamp(4.0, 200.0);
    }
    if let Some(v) = a.fixed_threshold {
        params.fixed_threshold = v;
    }
    if let Some(v) = a.regularization {
        params.regularization = v.max(0.0);
    }
    if let Some(m) = &a.metric {
        params.metric = match m.as_str() {
            "mean_squares" => Metric::MeanSquares,
            "mutual_information" => Metric::MutualInformation,
            other => bail!("metric must be mean_squares or mutual_information (got '{other}')"),
        };
    }
    if let Some(init) = a.init.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        params.init = match init {
            "auto" => Init::Auto,
            "identity" => Init::Identity,
            "center_of_gravity" | "centre_of_gravity" | "cog" => Init::CenterOfGravity,
            name => {
                let fname = name;
                let mname = a.init_moving.as_deref().unwrap_or(name);
                let fs = core.session.structure(&a.fixed.dataset, fname, None)?;
                let ms = core.session.structure(&a.moving.dataset, mname, None)?;
                let fc = motion::centroid_mm(&fs.mask_on(&fixed.grid())?, &fixed.grid())
                    .ok_or_else(|| anyhow::anyhow!("'{fname}' is empty on the fixed volume"))?;
                let mc = motion::centroid_mm(&ms.mask_on(&moving.grid())?, &moving.grid())
                    .ok_or_else(|| anyhow::anyhow!("'{mname}' is empty on the moving volume"))?;
                Init::Points {
                    fixed: fc,
                    moving: mc,
                }
            }
        };
    }
    if let Some(r) = &a.region {
        let s = core
            .session
            .structure(&a.fixed.dataset, &r.structure, r.set.as_deref())?;
        let mask = s.mask_on(&fixed.grid())?;
        let region = RegionMask::from_mask(&fixed, &mask, s.name.clone(), r.margin_mm.max(0.0))
            .ok_or_else(|| anyhow::anyhow!("'{}' is empty on the fixed volume", s.name))?;
        params.region = Some(Arc::new(region));
    }
    if let Some(id) = &a.start {
        let prev = core.session.registration(id)?;
        if prev.fixed != (a.fixed.dataset.clone(), fixed_idx)
            || prev.moving != (a.moving.dataset.clone(), moving_idx)
        {
            bail!("{id} was made for another pair of series; a refinement needs the same pair");
        }
        if !method.is_deformable() {
            bail!("a refinement needs a deformable method");
        }
        params.start = Some(prev.result.transform.clone());
    }

    p.set("Registering");
    let result = registration::register(&fixed, &moving, &params, p)?;
    let summary = analysis_json(&result);
    let id = core.session.mint("reg");
    core.session.registrations.push(Registration {
        id: id.clone(),
        fixed: (a.fixed.dataset.clone(), fixed_idx),
        moving: (a.moving.dataset.clone(), moving_idx),
        result,
    });
    Ok(json!({
        "reg": id,
        "fixed": { "dataset": a.fixed.dataset, "series": fixed_idx + 1 },
        "moving": { "dataset": a.moving.dataset, "series": moving_idx + 1 },
        "analysis": summary,
        "note": "the transform maps fixed patient coordinates to moving ones",
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegArgs {
    /// A registration handle such as `reg1`.
    pub reg: String,
}

pub fn describe_registration(core: &mut Core, a: RegArgs, _p: &Progress) -> Result<Value> {
    let r = core.session.registration(&a.reg)?;
    Ok(json!({
        "reg": r.id,
        "fixed": { "dataset": r.fixed.0, "series": r.fixed.1 + 1 },
        "moving": { "dataset": r.moving.0, "series": r.moving.1 + 1 },
        "analysis": analysis_json(&r.result),
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StructureArg {
    pub structure: String,
    #[serde(default)]
    pub set: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PropagateArgs {
    pub reg: String,
    /// Structures of the *source* side (the side `to` is not).
    pub structures: Vec<StructureArg>,
    /// `fixed` or `moving`: where the structures land.
    pub to: String,
    /// Suffix added to each landed name; default `(from dsN)`.
    #[serde(default)]
    pub suffix: Option<String>,
}

pub fn propagate(core: &mut Core, a: PropagateArgs, p: &Progress) -> Result<Value> {
    if a.structures.is_empty() {
        bail!("nothing selected to propagate");
    }
    let (fixed, moving, transform) = {
        let r = core.session.registration(&a.reg)?;
        (
            r.fixed.clone(),
            r.moving.clone(),
            r.result.transform.clone(),
        )
    };
    // The transform maps fixed → moving; landing on the moving side runs
    // through the inverse.
    let (src, dst, use_inverse) = match a.to.as_str() {
        "fixed" => (moving, fixed, false),
        "moving" => (fixed, moving, true),
        other => bail!("to must be fixed or moving (got '{other}')"),
    };
    let src_vol = core.session.volume(&src.0, src.1, p)?;
    let dst_vol = core.session.volume(&dst.0, dst.1, p)?;
    let src_grid = src_vol.grid();
    let mut subjects = Vec::new();
    for s in &a.structures {
        let st = core
            .session
            .structure(&src.0, &s.structure, s.set.as_deref())?;
        subjects.push(st.subject_on(&src_grid)?);
    }
    let items = propagate::propagate(&src_vol, &dst_vol, &transform, use_inverse, &subjects, p)?;
    let suffix = a
        .suffix
        .clone()
        .unwrap_or_else(|| format!("(from {})", src.0));
    let mut report = Vec::new();
    let mut masks = Vec::new();
    for it in items {
        report.push(json!({
            "structure": clean_text(&it.name),
            "landed_as": clean_text(&format!("{} {suffix}", it.name)),
            "source_cm3": round1(it.source_cm3),
            "result_cm3": round1(it.result_cm3),
            "voxels": it.voxels,
            "summary": it.summary(),
        }));
        if it.voxels > 0 {
            masks.push((format!("{} {suffix}", it.name), it.color, it.mask));
        }
    }
    let dst_grid = dst_vol.grid();
    let set = if masks.is_empty() {
        String::new()
    } else {
        core.session.land_masks(&dst.0, dst.1, &dst_grid, masks)?
    };
    Ok(json!({
        "reg": a.reg,
        "from": { "dataset": src.0, "series": src.1 + 1 },
        "to": { "dataset": dst.0, "series": dst.1 + 1 },
        "set": clean_text(&set),
        "structures": report,
    }))
}
