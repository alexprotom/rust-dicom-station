//! Segmentation: organs, the body outline, and structure algebra.

use std::path::PathBuf;

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::Core;
use super::session::round2;
use crate::bodymask;
use crate::models;
use crate::progress::Progress;
use crate::structops::{self, BoolOp, Cleanup, Margin, Operand, Recipe};
use crate::zoo::{self, AutoModel};

/// The model folder, and the check that stops a run from turning into a
/// download when downloads are not allowed.
fn models_root(core: &Core) -> PathBuf {
    core.session.config.models_dir()
}

fn refuse_download(core: &Core, bytes: u64, what: &str) -> Result<()> {
    if bytes > 0 && !core.session.config.allow_model_download {
        bail!(
            "the {what} weights are not present ({} to download) and allow_model_download is off; \
             fetch them once through the viewer's model manager (Tools > Models), or allow \
             downloads in mcp.toml",
            models::human_bytes(bytes)
        );
    }
    Ok(())
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OrgansArgs {
    /// Dataset handle.
    pub dataset: String,
    /// Series number (from describe_dataset); the displayed series when
    /// omitted.
    #[serde(default)]
    pub series: Option<u32>,
    /// The model to run, by its key from list_models (`total_fast`,
    /// `total_mr`, `mrsegmentator`, `lung_vessels`, `lungmask_lobes`,
    /// `monai_wholebody`, `ctfm_wholebody`, `vista3d`, ...).
    /// When omitted, `variant` chooses among the TotalSegmentator CT
    /// models.
    #[serde(default)]
    pub model: Option<String>,
    /// Used when `model` is omitted: `fast` (3 mm, all 117 classes),
    /// `high` (1.5 mm, choose `parts`) or `preview` (6 mm) run the v2
    /// weights; `fast_v3`, `high_v3`, `preview_v3` the v3 weights, and
    /// `small_v3` / `small_high_v3` the v3 residual-encoder ("small")
    /// network at 3 mm / 1.5 mm. v3 names label 26 `vertebrae_L6` where v2
    /// says `vertebrae_S1`.
    #[serde(default = "default_variant")]
    pub variant: String,
    /// For a model with sub-models (list_models shows them; for `total`
    /// at 1.5 mm: organs, vertebrae, cardiac, muscles, ribs): which to run.
    /// Empty means all.
    #[serde(default)]
    pub parts: Vec<String>,
    /// Keep only these structures (the model's class names, such as
    /// `heart`, `aorta`, `lung_upper_lobe_left`, or their TG-263 names).
    /// Empty keeps everything found.
    #[serde(default)]
    pub keep: Vec<String>,
    /// Name the structures by AAPM TG-263 where a class has a TG-263 name
    /// (Kidney_L, Lung_RUL, VB_T07); the others keep the model's name.
    #[serde(default)]
    pub tg263: bool,
}

fn default_variant() -> String {
    "fast".into()
}

/// The model `a` names.
fn chosen_model(a: &OrgansArgs) -> Result<AutoModel> {
    match &a.model {
        Some(key) => AutoModel::from_key(key)
            .ok_or_else(|| anyhow::anyhow!("unknown model '{key}'; list_models names every model")),
        None => Ok(AutoModel::Nn(
            crate::workflow::params::AutosegVariant::from_name(&a.variant)?
                .variant()
                .task(),
        )),
    }
}

pub fn segment_organs(core: &mut Core, a: OrgansArgs, p: &Progress) -> Result<Value> {
    let model = chosen_model(&a)?;
    let names = model.part_names();
    let parts = if a.parts.is_empty() || names.is_empty() {
        None
    } else {
        let mut on = vec![false; names.len()];
        for part in &a.parts {
            let Some(i) = names
                .iter()
                .position(|n| n.eq_ignore_ascii_case(part.trim()))
            else {
                bail!(
                    "unknown part '{part}'; the parts of {} are {}",
                    model.key(),
                    names.join(", ")
                );
            };
            on[i] = true;
        }
        Some(on)
    };
    let root = models_root(core);
    refuse_download(
        core,
        model.download_needed(parts.as_deref(), &root),
        model.label(),
    )?;
    let ds = core.session.dataset(&a.dataset)?;
    let series = core.session.series_index(ds, a.series)?;
    let modality = ds.study.series[series].modality.clone();
    let volume = core.session.volume(&a.dataset, series, p)?;
    let opts = zoo::RunOptions {
        device: core.session.config.device_pref(),
        parts,
    };
    let result = model.run(&volume, &opts, &root, p)?;
    if result.organs.is_empty() {
        bail!("no structures were found in this volume");
    }
    let keep_lower: Vec<String> = a.keep.iter().map(|k| k.to_lowercase()).collect();
    let named = |o: &crate::autoseg::OrganHit| -> String {
        if a.tg263 {
            zoo::tg263_name(o.name).unwrap_or_else(|| o.name.to_string())
        } else {
            o.name.to_string()
        }
    };
    let classes: Vec<(u8, String, [u8; 3])> = result
        .organs
        .iter()
        .filter(|o| {
            keep_lower.is_empty()
                || keep_lower.contains(&o.name.to_lowercase())
                || zoo::tg263_name(o.name).is_some_and(|t| keep_lower.contains(&t.to_lowercase()))
        })
        .map(|o| (o.label, named(o), o.color))
        .collect();
    if classes.is_empty() {
        bail!(
            "none of the requested structures was found; found: {}",
            result
                .organs
                .iter()
                .map(|o| o.name)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let made = crate::segmentation::Segmentation::from_label_map_many(
        result.dims,
        &result.labels,
        &classes,
    );
    let grid = volume.grid();
    let spacing = volume.spacing;
    let organs: Vec<Value> = made
        .iter()
        .map(|s| json!({ "name": s.name, "volume_cm3": round2(s.volume_cm3(spacing)) }))
        .collect();
    let masks: Vec<(String, [u8; 3], Vec<u8>)> = made
        .into_iter()
        .map(|s| (s.name.clone(), s.color, s.mask))
        .collect();
    let set = core.session.land_masks(&a.dataset, series, &grid, masks)?;
    let mut notes = result.notes.clone();
    if !model.modality().accepts(&modality) {
        notes.push(format!(
            "{} was trained on {}; this series is {}",
            model.label(),
            model.modality().label(),
            clean_text(&modality)
        ));
    }
    Ok(json!({
        "dataset": a.dataset,
        "series": series + 1,
        "set": clean_text(&set),
        "model": model.key(),
        "model_label": model.label(),
        "licence": model.licence().name(),
        "device": result.device,
        "elapsed_s": (result.elapsed_secs * 10.0).round() / 10.0,
        "structures": organs,
        "notes": notes,
    }))
}

/// Every automatic model, with what it needs.
pub fn list_models(core: &mut Core, _: super::NoArgs, _: &Progress) -> Result<Value> {
    let root = models_root(core);
    let models: Vec<Value> = AutoModel::all()
        .into_iter()
        .map(|m| {
            let need = m.download_needed(None, &root);
            json!({
                "key": m.key(),
                "label": m.label(),
                "group": m.group(),
                "family": m.family().label(),
                "modality": m.modality().label(),
                "licence": m.licence().name(),
                "research_only": m.licence().research_only(),
                "classes": m.classes().len(),
                "parts": m.part_names(),
                "ready": need == 0,
                "download_bytes": need,
                "detail": m.detail(),
            })
        })
        .collect();
    Ok(
        json!({ "models": models, "allow_model_download": core.session.config.allow_model_download }),
    )
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BodyArgs {
    pub dataset: String,
    #[serde(default)]
    pub series: Option<u32>,
    /// `classical` (threshold and morphology, no model) or `model_assisted`.
    #[serde(default = "default_method")]
    pub method: String,
    /// Name of the resulting structure.
    #[serde(default = "default_body_name")]
    pub name: String,
}

fn default_method() -> String {
    "classical".into()
}
fn default_body_name() -> String {
    "BODY".into()
}

pub fn segment_body(core: &mut Core, a: BodyArgs, p: &Progress) -> Result<Value> {
    let ds = core.session.dataset(&a.dataset)?;
    let series = core.session.series_index(ds, a.series)?;
    let modality = ds.study.series[series].modality.clone();
    let mut params = bodymask::BodyParams::for_modality(&modality);
    params.method = crate::workflow::params::BodyMethod::from_name(&a.method)?.method();
    params.device = core.session.config.device_pref();
    let dir = models_root(core);
    if params.method == bodymask::Method::ModelAssisted {
        refuse_download(
            core,
            bodymask::download_needed(params.model, &dir),
            "body-outline",
        )?;
    }
    let volume = core.session.volume(&a.dataset, series, p)?;
    let r = bodymask::contour_body(&volume, &params, &dir, p)?;
    let grid = volume.grid();
    let set = core.session.land_masks(
        &a.dataset,
        series,
        &grid,
        vec![(a.name.clone(), [0, 200, 0], r.mask)],
    )?;
    Ok(json!({
        "dataset": a.dataset,
        "series": series + 1,
        "set": clean_text(&set),
        "structure": a.name,
        "volume_cm3": round2(r.cm3),
        "pieces": r.pieces.iter().map(|x| round2(x.cm3)).collect::<Vec<_>>(),
        "removed_voxels": r.removed_voxels,
    }))
}

/// A margin as the client writes it: one number for all directions, or
/// six by patient direction (mm; negative shrinks).
#[derive(Deserialize, JsonSchema, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct MarginArg {
    #[serde(default)]
    pub uniform_mm: Option<f64>,
    #[serde(default)]
    pub right: Option<f64>,
    #[serde(default)]
    pub left: Option<f64>,
    #[serde(default)]
    pub anterior: Option<f64>,
    #[serde(default)]
    pub posterior: Option<f64>,
    #[serde(default)]
    pub superior: Option<f64>,
    #[serde(default)]
    pub inferior: Option<f64>,
}

impl MarginArg {
    pub fn to_margin(&self) -> Margin {
        let base = Margin::uniform(self.uniform_mm.unwrap_or(0.0));
        Margin {
            right: self.right.unwrap_or(base.right),
            left: self.left.unwrap_or(base.left),
            anterior: self.anterior.unwrap_or(base.anterior),
            posterior: self.posterior.unwrap_or(base.posterior),
            superior: self.superior.unwrap_or(base.superior),
            inferior: self.inferior.unwrap_or(base.inferior),
        }
    }
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OperandArg {
    /// Structure name (see list_structures).
    pub structure: String,
    /// The structure set or segmentation series it is in, when the name is
    /// not unique.
    #[serde(default)]
    pub set: Option<String>,
    #[serde(default)]
    pub margin: MarginArg,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CombineArgs {
    pub dataset: String,
    /// The lattice the result lives on: a series number; the displayed
    /// series when omitted.
    #[serde(default)]
    pub series: Option<u32>,
    /// `union`, `intersect` or `subtract` (the first operand minus the rest).
    pub op: String,
    pub operands: Vec<OperandArg>,
    /// Name of the result.
    pub name: String,
    /// Margin applied to the combined result.
    #[serde(default)]
    pub margin: MarginArg,
    #[serde(default)]
    pub fill_holes: bool,
    #[serde(default)]
    pub close_mm: f64,
    #[serde(default)]
    pub keep_largest: bool,
    #[serde(default)]
    pub min_volume_cm3: f64,
}

pub fn combine_structures(core: &mut Core, a: CombineArgs, p: &Progress) -> Result<Value> {
    let op = match a.op.as_str() {
        "union" => BoolOp::Union,
        "intersect" => BoolOp::Intersect,
        "subtract" => BoolOp::Subtract,
        other => bail!("op must be union, intersect or subtract (got '{other}')"),
    };
    let ds = core.session.dataset(&a.dataset)?;
    let series = core.session.series_index(ds, a.series)?;
    let grid = core.session.grid(&a.dataset, series, p)?;
    let mut operands = Vec::new();
    for o in &a.operands {
        let s = core
            .session
            .structure(&a.dataset, &o.structure, o.set.as_deref())?;
        operands.push(Operand {
            name: s.name.clone(),
            mask: s.mask_on(&grid)?,
            margin: o.margin.to_margin(),
        });
    }
    let recipe = Recipe {
        op,
        operands,
        margin: a.margin.to_margin(),
        cleanup: Cleanup {
            fill_holes: a.fill_holes,
            close_mm: a.close_mm,
            keep_largest: a.keep_largest,
            min_volume_cm3: a.min_volume_cm3,
        },
    };
    let out = structops::combine(&recipe, &grid, p)?;
    let set = core.session.land_masks(
        &a.dataset,
        series,
        &grid,
        vec![(a.name.clone(), [255, 200, 0], out.mask)],
    )?;
    Ok(json!({
        "dataset": a.dataset,
        "set": clean_text(&set),
        "structure": a.name,
        "op": op.label(),
        "volume_cm3": round2(out.cm3),
        "voxels": out.voxels,
        "pieces": out.pieces,
    }))
}
