//! Segmentation: organs, the body outline, and structure algebra.

use std::path::PathBuf;

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::Core;
use super::session::round1;
use crate::autoseg;
use crate::bodymask;
use crate::models::{self, Engine};
use crate::progress::Progress;
use crate::structops::{self, BoolOp, Cleanup, Margin, Operand, Recipe};

/// The engine's model folder, and the check that stops a run from turning
/// into a download when downloads are not allowed.
fn models_dir(core: &Core, engine: Engine) -> PathBuf {
    models::engine_dir(&core.session.config.models_dir(), engine)
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
    /// `fast` (3 mm, all 117 classes), `high` (1.5 mm, choose `parts`) or
    /// `preview` (6 mm).
    #[serde(default = "default_variant")]
    pub variant: String,
    /// For `high`: any of organs, vertebrae, cardiac, muscles, ribs. Empty
    /// means all five.
    #[serde(default)]
    pub parts: Vec<String>,
    /// Keep only these organs (TotalSegmentator class names such as
    /// `heart`, `aorta`, `lung_upper_lobe_left`). Empty keeps everything
    /// found.
    #[serde(default)]
    pub keep: Vec<String>,
}

fn default_variant() -> String {
    "fast".into()
}

pub fn segment_organs(core: &mut Core, a: OrgansArgs, p: &Progress) -> Result<Value> {
    let variant = match a.variant.as_str() {
        "fast" => autoseg::Variant::Fast3mm,
        "high" => autoseg::Variant::HighRes15mm,
        "preview" => autoseg::Variant::Preview6mm,
        other => bail!("variant must be fast, high or preview (got '{other}')"),
    };
    let mut parts = [a.parts.is_empty(); 5];
    for part in &a.parts {
        let Some(i) = autoseg::classes::PART_NAMES
            .iter()
            .position(|n| n.eq_ignore_ascii_case(part))
        else {
            bail!(
                "unknown part '{part}'; the parts are {}",
                autoseg::classes::PART_NAMES.join(", ")
            );
        };
        parts[i] = true;
    }
    let dir = models_dir(core, Engine::TotalSegmentator);
    refuse_download(
        core,
        autoseg::download_needed(variant, parts, &dir),
        "TotalSegmentator",
    )?;
    let ds = core.session.dataset(&a.dataset)?;
    let series = core.session.series_index(ds, a.series)?;
    let volume = core.session.volume(&a.dataset, series, p)?;
    let device = core.session.config.device_pref();

    let result = autoseg::run(&volume, variant, device, parts, &dir, p)?;
    if result.organs.is_empty() {
        bail!("no organs were found in this volume");
    }
    let keep_lower: Vec<String> = a.keep.iter().map(|k| k.to_lowercase()).collect();
    let classes: Vec<(u8, String, [u8; 3])> = result
        .organs
        .iter()
        .filter(|o| keep_lower.is_empty() || keep_lower.contains(&o.name.to_lowercase()))
        .map(|o| (o.label, o.name.to_string(), o.color))
        .collect();
    if classes.is_empty() {
        bail!(
            "none of the requested organs was found; found: {}",
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
        .map(|s| json!({ "name": s.name, "volume_cm3": round1(s.volume_cm3(spacing)) }))
        .collect();
    let masks: Vec<(String, [u8; 3], Vec<u8>)> = made
        .into_iter()
        .map(|s| (s.name.clone(), s.color, s.mask))
        .collect();
    let set = core.session.land_masks(&a.dataset, series, &grid, masks)?;
    Ok(json!({
        "dataset": a.dataset,
        "series": series + 1,
        "set": clean_text(&set),
        "variant": variant.label(),
        "device": result.device,
        "elapsed_s": (result.elapsed_secs * 10.0).round() / 10.0,
        "structures": organs,
    }))
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
    params.method = match a.method.as_str() {
        "classical" => bodymask::Method::Classical,
        "model_assisted" => bodymask::Method::ModelAssisted,
        other => bail!("method must be classical or model_assisted (got '{other}')"),
    };
    params.device = core.session.config.device_pref();
    let dir = models_dir(core, Engine::TotalSegmentator);
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
        "volume_cm3": round1(r.cm3),
        "pieces": r.pieces.iter().map(|x| round1(x.cm3)).collect::<Vec<_>>(),
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
        "volume_cm3": round1(out.cm3),
        "voxels": out.voxels,
        "pieces": out.pieces,
    }))
}
