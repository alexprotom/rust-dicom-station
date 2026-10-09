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

/// A method as the client names it - the names the workflow file uses too
/// ([`crate::workflow::params::RegMethodChoice`]).
pub fn parse_method(s: &str) -> Result<RegMethod> {
    Ok(crate::workflow::params::RegMethodChoice::from_name(s)?.method())
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

// ---- registration by structures --------------------------------------------

/// One structure of a registration by structures.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShapeStructureArg {
    /// The structure on the fixed dataset.
    pub fixed: String,
    /// Its name on the moving dataset, when it differs.
    #[serde(default)]
    pub moving: Option<String>,
    /// The structure set (or segmentation series) on the fixed dataset,
    /// when the name is in several.
    #[serde(default)]
    pub set: Option<String>,
    /// The same on the moving dataset.
    #[serde(default)]
    pub moving_set: Option<String>,
    /// How much it counts against the others (default 1; each structure is
    /// already normalised by its own size).
    #[serde(default)]
    pub weight: Option<f64>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShapeRegisterArgs {
    pub fixed: Side,
    pub moving: Side,
    /// The structures to align on, each contoured on both datasets.
    pub structures: Vec<ShapeStructureArg>,
    /// `rigid` (default) or `translation`.
    #[serde(default)]
    pub dof: Option<String>,
    /// `none` (default), `elastix_bspline` or `plastimatch_bspline`: a local
    /// B-spline refinement on each structure's distance maps after the
    /// rigid fit.
    #[serde(default)]
    pub refine: Option<String>,
    /// Dilation of each structure that bounds its refinement and the
    /// analysis, mm (default 10).
    #[serde(default)]
    pub margin_mm: Option<f64>,
    /// Huber width, mm: distances beyond it count linearly, so a slice
    /// contoured differently does not steer the fit. Omitted or 0: least
    /// squares.
    #[serde(default)]
    pub robust_mm: Option<f64>,
    /// Also lay the moving surfaces onto the fixed structures (default
    /// true).
    #[serde(default)]
    pub symmetric: Option<bool>,
    /// The refinement's resolution levels, iterations, samples and B-spline
    /// grid spacing (mm).
    #[serde(default)]
    pub levels: Option<usize>,
    #[serde(default)]
    pub iterations: Option<usize>,
    #[serde(default)]
    pub samples: Option<usize>,
    #[serde(default)]
    pub grid_spacing_mm: Option<f64>,
    /// Where the rigid search starts: `auto` (the structures' centroids
    /// matched), `identity`, or a structure name contoured on both
    /// datasets whose own centroids are matched.
    #[serde(default)]
    pub init: Option<String>,
    /// A `reg` handle of the same pair to refine (skips the rigid fit;
    /// needs `refine`).
    #[serde(default)]
    pub start: Option<String>,
}

/// A structure of one side: by set when one is named, else the one drawn
/// on the series (a structure set or segmentation that references it),
/// else any of that name in the dataset.
fn structure_on(
    core: &Core,
    dataset: &str,
    series: usize,
    name: &str,
    set: Option<&str>,
) -> Result<crate::workflow::select::Structure> {
    if set.is_some() {
        return core.session.structure(dataset, name, set);
    }
    let ds = core.session.dataset(dataset)?;
    let uid = ds
        .study
        .series
        .get(series)
        .map(|s| s.uid.clone())
        .unwrap_or_default();
    match crate::workflow::select::find_on_series(&ds.study, name, &uid, "") {
        Some(s) => Ok(s),
        // The session's own lookup says what there is instead.
        None => core.session.structure(dataset, name, None),
    }
}

pub fn register_structures(core: &mut Core, a: ShapeRegisterArgs, p: &Progress) -> Result<Value> {
    use crate::workflow::params::{ShapeDofChoice, ShapeRefine, ShapeRegParams};
    if a.structures.is_empty() {
        bail!("structures: name at least one structure contoured on both datasets");
    }
    let fds = core.session.dataset(&a.fixed.dataset)?;
    let fixed_idx = core.session.series_index(fds, a.fixed.series)?;
    let mds = core.session.dataset(&a.moving.dataset)?;
    let moving_idx = core.session.series_index(mds, a.moving.series)?;
    if a.fixed.dataset == a.moving.dataset && fixed_idx == moving_idx {
        bail!("fixed and moving are the same series");
    }
    let mut prm = ShapeRegParams::default();
    if let Some(d) = &a.dof {
        prm.dof = ShapeDofChoice::from_name(d)?;
    }
    if let Some(r) = &a.refine {
        prm.refine = ShapeRefine::from_name(r)?;
    }
    if let Some(v) = a.margin_mm {
        prm.margin_mm = v.clamp(0.0, 60.0);
    }
    if let Some(v) = a.robust_mm {
        prm.robust_mm = v.max(0.0);
    }
    if let Some(v) = a.symmetric {
        prm.symmetric = v;
    }
    if let Some(v) = a.levels {
        prm.effort.levels = v;
    }
    if let Some(v) = a.iterations {
        prm.effort.iterations = v;
    }
    if let Some(v) = a.samples {
        prm.effort.samples = v;
    }
    if let Some(v) = a.grid_spacing_mm {
        prm.effort.grid_spacing_mm = v;
    }
    let fixed = core.session.volume(&a.fixed.dataset, fixed_idx, p)?;
    let moving = core.session.volume(&a.moving.dataset, moving_idx, p)?;
    let (fgrid, mgrid) = (fixed.grid(), moving.grid());

    p.set("Rasterizing the structures");
    let mut pairs = Vec::with_capacity(a.structures.len());
    for s in &a.structures {
        let mname = s.moving.as_deref().unwrap_or(&s.fixed);
        let fs = structure_on(
            core,
            &a.fixed.dataset,
            fixed_idx,
            &s.fixed,
            s.set.as_deref(),
        )?;
        let ms = structure_on(
            core,
            &a.moving.dataset,
            moving_idx,
            mname,
            s.moving_set.as_deref(),
        )?;
        let weight = s.weight.unwrap_or(1.0);
        if !(weight.is_finite() && weight > 0.0) {
            bail!("the weight of '{}' must be positive", s.fixed);
        }
        let name = if mname.eq_ignore_ascii_case(&s.fixed) {
            fs.name.clone()
        } else {
            format!("{} / {}", fs.name, ms.name)
        };
        pairs.push(registration::ShapePair {
            name,
            color: fs.color,
            weight,
            fixed: fs
                .mask_on(&fgrid)
                .map_err(|_| anyhow::anyhow!("'{}' is empty on the fixed series", s.fixed))?,
            moving: ms
                .mask_on(&mgrid)
                .map_err(|_| anyhow::anyhow!("'{mname}' is empty on the moving series"))?,
        });
    }
    let init = match a.init.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None | Some("auto") | Some("center_of_gravity") | Some("centre_of_gravity") => Init::Auto,
        Some("identity") => Init::Identity,
        Some(name) => {
            let fs = structure_on(core, &a.fixed.dataset, fixed_idx, name, None)?;
            let ms = structure_on(core, &a.moving.dataset, moving_idx, name, None)?;
            let fc = motion::centroid_mm(&fs.mask_on(&fgrid)?, &fgrid)
                .ok_or_else(|| anyhow::anyhow!("'{name}' is empty on the fixed series"))?;
            let mc = motion::centroid_mm(&ms.mask_on(&mgrid)?, &mgrid)
                .ok_or_else(|| anyhow::anyhow!("'{name}' is empty on the moving series"))?;
            Init::Points {
                fixed: fc,
                moving: mc,
            }
        }
    };
    let mut req = prm.request(pairs, init);
    if let Some(id) = &a.start {
        let prev = core.session.registration(id)?;
        if prev.fixed != (a.fixed.dataset.clone(), fixed_idx)
            || prev.moving != (a.moving.dataset.clone(), moving_idx)
        {
            bail!("{id} was made for another pair of series; a refinement needs the same pair");
        }
        if req.refine.is_none() {
            bail!("a refinement needs refine: elastix_bspline or plastimatch_bspline");
        }
        req.start = Some(prev.result.transform.clone());
    }

    p.set("Registering on the structures");
    let out = registration::shape::register(&fixed, &moving, &req, p)?;
    let summary = analysis_json(&out.result);
    let structures: Vec<Value> = out
        .report
        .lines
        .iter()
        .map(|l| {
            json!({
                "structure": clean_text(&l.name),
                "weight": round2(l.weight),
                "surface_points": l.points,
                "mean_distance_mm": { "before": round2(l.mean_before_mm), "after": round2(l.mean_after_mm) },
                "rms_distance_mm": { "before": round2(l.rms_before_mm), "after": round2(l.rms_after_mm) },
                "dice": { "before": l.dice_before.map(round3), "after": l.dice_after.map(round3) },
                "refinement": l.refine_line.as_deref().map(clean_text),
                "refinement_displacement_p95_mm": l.displacement_p95_mm.map(round2),
                "refinement_folded_fraction": l.folded_fraction.map(round3),
            })
        })
        .collect();
    let id = core.session.mint("reg");
    core.session.registrations.push(Registration {
        id: id.clone(),
        fixed: (a.fixed.dataset.clone(), fixed_idx),
        moving: (a.moving.dataset.clone(), moving_idx),
        result: out.result,
    });
    Ok(json!({
        "reg": id,
        "fixed": { "dataset": a.fixed.dataset, "series": fixed_idx + 1 },
        "moving": { "dataset": a.moving.dataset, "series": moving_idx + 1 },
        "analysis": summary,
        "structures": structures,
        "rigid_iterations": out.report.rigid_iterations,
        "note": "the transform maps fixed patient coordinates to moving ones; the image \
                 intensities took no part in it",
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
    /// After landing: morphological closing radius, mm (gaps narrower than
    /// twice this are bridged; a cloud of pieces becomes one surface).
    #[serde(default)]
    pub close_mm: f64,
    /// After landing (and closing): fill the interior, so a surface becomes
    /// a solid.
    #[serde(default)]
    pub fill: bool,
    /// Carry each structure as a rigid body - the transform's best rigid
    /// fit over the structure itself - so it keeps its shape and volume and
    /// follows the deformation's local position and turn.
    #[serde(default)]
    pub keep_shape: bool,
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
    let finish = propagate::Finish {
        close_mm: a.close_mm.clamp(0.0, 50.0),
        fill: a.fill,
        keep_shape: a.keep_shape,
    };
    finish.carry(&mut subjects);
    let mut items =
        propagate::propagate(&src_vol, &dst_vol, &transform, use_inverse, &subjects, p)?;
    finish.apply_all(&mut items, &dst_vol.grid(), p);
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
            "source_cm3": round2(it.source_cm3),
            "mapped_cm3": round2(it.mapped_cm3),
            "result_cm3": round2(it.result_cm3),
            "voxels": it.voxels,
            "rigid_residual_mm": it.rigid_residual_mm.map(round2),
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
