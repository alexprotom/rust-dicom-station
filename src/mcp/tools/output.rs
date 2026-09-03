//! Writing results out, and getting a person to look at them.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::config::PhiPolicy;
use super::super::phi::clean_text;
use super::super::Core;
use crate::anonymize;
use crate::archive;
use crate::dicom_export::{self, ExportParams};
use crate::export::{self, ExportPlan, Layout, ObjKind, StructFormat, UidMode};
use crate::progress::Progress;
use crate::registration::VectorField;
use crate::settings;

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportArgs {
    pub dataset: String,
    /// `seg` (Segmentation Storage, one mask per structure) or `rtstruct`
    /// (contours). Applies to structure sets and segmentation series alike.
    #[serde(default = "default_format")]
    pub format: String,
    /// `keep` the study's identifiers or mint `new` ones.
    #[serde(default = "default_uid_mode")]
    pub uid_mode: String,
    /// Also write the image series (all of them).
    #[serde(default)]
    pub include_images: bool,
    #[serde(default = "yes")]
    pub include_structures: bool,
    #[serde(default = "yes")]
    pub include_doses: bool,
    #[serde(default = "yes")]
    pub include_plans: bool,
    /// Only structure sets / segmentation series whose label contains this
    /// text (case-insensitive). Empty: all of them.
    #[serde(default)]
    pub sets_matching: Option<String>,
    /// Name of the folder under the session's output folder.
    #[serde(default)]
    pub folder: Option<String>,
}

fn default_format() -> String {
    "seg".into()
}
fn default_uid_mode() -> String {
    "keep".into()
}
fn yes() -> bool {
    true
}

pub fn export(core: &mut Core, a: ExportArgs, p: &Progress) -> Result<Value> {
    let format = match a.format.as_str() {
        "seg" => StructFormat::Seg,
        "rtstruct" => StructFormat::RtStruct,
        other => bail!("format must be seg or rtstruct (got '{other}')"),
    };
    let uid_mode = match a.uid_mode.as_str() {
        "keep" => UidMode::Keep,
        "new" => UidMode::New,
        other => bail!("uid_mode must be keep or new (got '{other}')"),
    };
    let out = core.session.fresh_out_subdir(
        a.folder
            .as_deref()
            .unwrap_or(&format!("{}-export", a.dataset)),
    )?;
    let ds = core.session.dataset(&a.dataset)?;
    let study = &ds.study;
    let mut plan = ExportPlan::build([Some(study), None], ExportParams::for_study(study));
    plan.layout = Layout::StudyFolders;
    plan.set_uid_mode(uid_mode);
    plan.set_all_formats(format);
    let needle = a.sets_matching.as_deref().map(str::to_lowercase);
    for st in plan.studies_mut() {
        for se in &mut st.series {
            se.selected = a.include_images;
        }
        for ob in &mut st.objects {
            ob.selected = match ob.kind {
                ObjKind::Structures | ObjKind::Segmentation => {
                    a.include_structures
                        && needle
                            .as_ref()
                            .is_none_or(|n| ob.label.to_lowercase().contains(n))
                }
                ObjKind::Dose => a.include_doses,
                ObjKind::Plan => a.include_plans,
            };
        }
    }
    if plan.is_empty() {
        bail!("nothing is selected for export");
    }
    let summary = export::run(&plan, [Some(study), None], &out, p)?;
    Ok(json!({
        "dataset": a.dataset,
        "folder": out.to_string_lossy(),
        "files": summary.files,
        "format": format.label(),
        "uid_mode": a.uid_mode,
        "identifiers": match core.session.config.phi_policy {
            PhiPolicy::Allow if !ds.phi_tags.is_empty() => "the study's own (policy allow)",
            _ => "as loaded (anonymized or aliased)",
        },
        "warnings": summary.warnings.iter().map(|w| clean_text(w)).collect::<Vec<_>>(),
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportRegArgs {
    pub reg: String,
    /// Lattice step of the written displacement grid, mm.
    #[serde(default = "default_step")]
    pub step_mm: f64,
    #[serde(default)]
    pub folder: Option<String>,
}

fn default_step() -> f64 {
    5.0
}

pub fn export_registration(core: &mut Core, a: ExportRegArgs, p: &Progress) -> Result<Value> {
    let (fixed, moving, transform, region, label) = {
        let r = core.session.registration(&a.reg)?;
        (
            r.fixed.clone(),
            r.moving.clone(),
            r.result.transform.clone(),
            r.result.region.clone(),
            r.result.method.label().to_string(),
        )
    };
    let fixed_vol = core.session.volume(&fixed.0, fixed.1, p)?;
    let moving_vol = core.session.volume(&moving.0, moving.1, p)?;
    p.set("Sampling the vector field");
    let field = VectorField::sample(&fixed_vol, &transform, None, a.step_mm.clamp(1.0, 50.0));
    let dir = core.session.fresh_out_subdir(
        a.folder
            .as_deref()
            .unwrap_or(&format!("{}-registration", a.reg)),
    )?;
    std::fs::create_dir_all(&dir).with_context(|| "create the output folder".to_string())?;
    let path = dir.join(format!("{}.dcm", a.reg));
    let ds = core.session.dataset(&fixed.0)?;
    let study_uid = ds.study.series[fixed.1].study_uid.clone();
    let meta = dicom_export::DvfExport {
        source_for_uid: &fixed_vol.frame_of_reference_uid,
        target_for_uid: &moving_vol.frame_of_reference_uid,
        study_uid: &study_uid,
        patient_name: &ds.study.meta.patient_name,
        patient_id: &ds.study.meta.patient_id,
        label: &label,
        description: &format!(
            "rds-mcp {} {} -> {}{}",
            a.reg,
            fixed.0,
            moving.0,
            region
                .as_deref()
                .map(|r| format!(" (local: {r})"))
                .unwrap_or_default()
        ),
    };
    dicom_export::write_deformable_registration(&path, &field, &meta)?;
    Ok(json!({
        "reg": a.reg,
        "file": path.to_string_lossy(),
        "grid_dims": field.dims,
        "step_mm": field.spacing,
    }))
}

/// A folder the client may hand to the archive or the viewer: under a root
/// or under the output folder.
fn readable_folder(core: &Core, given: &Path) -> Result<PathBuf> {
    let (real, _) = core.session.config.resolve_input(given)?;
    if !real.is_dir() {
        bail!("'{}' is not a folder", given.display());
    }
    Ok(real)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportArgs {
    /// A folder under a root or under the output folder.
    pub path: PathBuf,
}

pub fn import_to_archive(core: &mut Core, a: ImportArgs, p: &Progress) -> Result<Value> {
    let src = readable_folder(core, &a.path)?;
    let settings = settings::load();
    let root = archive::root_from_setting(
        &settings
            .archive_dir
            .as_deref()
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_default(),
    );
    let ar = archive::Archive::new(root.clone());
    let sum = ar.import(&src, p)?;
    core.session.add_root(&root, "archive");
    Ok(json!({
        "imported_from": src.to_string_lossy(),
        "summary": clean_text(&sum.describe()),
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenViewerArgs {
    /// Folder shown as dataset A.
    pub path: PathBuf,
    /// Folder shown as dataset B (comparison mode).
    #[serde(default)]
    pub path_b: Option<PathBuf>,
}

/// The viewer executable: configured, or beside this one.
fn viewer_exe(core: &Core) -> Result<PathBuf> {
    if let Some(p) = &core.session.config.viewer_exe {
        if p.is_file() {
            return Ok(p.clone());
        }
        bail!("viewer_exe in mcp.toml does not exist");
    }
    let here = std::env::current_exe().context("locate this executable")?;
    let dir = here.parent().context("this executable has no folder")?;
    for name in ["rust-dicom-station.exe", "rust-dicom-station"] {
        let c = dir.join(name);
        if c.is_file() {
            return Ok(c);
        }
    }
    bail!("the viewer executable was not found beside rds-mcp; set viewer_exe in mcp.toml")
}

pub fn open_in_viewer(core: &mut Core, a: OpenViewerArgs, _p: &Progress) -> Result<Value> {
    let a_dir = readable_folder(core, &a.path)?;
    let b_dir = match &a.path_b {
        Some(b) => Some(readable_folder(core, b)?),
        None => None,
    };
    let exe = viewer_exe(core)?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg(&a_dir);
    if let Some(b) = &b_dir {
        cmd.arg(b);
    }
    // The viewer outlives this call; its output is not ours to read.
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    let child = cmd.spawn().context("start the viewer")?;
    Ok(json!({
        "launched": true,
        "pid": child.id(),
        "a": a_dir.to_string_lossy(),
        "b": b_dir.map(|b| b.to_string_lossy().to_string()),
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnonymizeArgs {
    /// A folder under a root.
    pub path: PathBuf,
    /// Also clear the study and series descriptions (they sometimes carry
    /// names); off by default because 4D phase detection reads them.
    #[serde(default)]
    pub clear_descriptions: bool,
    #[serde(default = "yes")]
    pub remove_private: bool,
    #[serde(default = "yes")]
    pub remap_uids: bool,
    #[serde(default)]
    pub folder: Option<String>,
}

pub fn anonymize(core: &mut Core, a: AnonymizeArgs, p: &Progress) -> Result<Value> {
    let src = readable_folder(core, &a.path)?;
    let scan = anonymize::scan(&src, p)?;
    // Everything the scan found goes to the redactor: the values are in
    // the findings, and the findings never leave this function.
    let mut values = Vec::new();
    for f in &scan.findings {
        values.extend(f.values.iter().cloned());
    }
    core.session.add_values(values);
    let replacements: Vec<_> = scan
        .findings
        .iter()
        .filter(|f| {
            f.enabled
                || (a.clear_descriptions
                    && matches!(f.name.as_str(), "StudyDescription" | "SeriesDescription"))
        })
        .map(|f| {
            let value = if a.clear_descriptions
                && matches!(f.name.as_str(), "StudyDescription" | "SeriesDescription")
            {
                String::new()
            } else {
                f.replacement.trim().to_string()
            };
            (f.tag, f.vr, value)
        })
        .collect();
    let tags_changed: Vec<String> = scan
        .findings
        .iter()
        .filter(|f| f.enabled)
        .map(|f| f.name.clone())
        .collect();
    let out = core
        .session
        .fresh_out_subdir(a.folder.as_deref().unwrap_or("anonymized"))?;
    let params = anonymize::ApplyParams {
        replacements,
        remove_private: a.remove_private,
        remap_uids: a.remap_uids,
        mark_deidentified: true,
        out_dir: Some(out.clone()),
    };
    let n = anonymize::apply(&scan.files, &scan.root, &params, p)?;
    Ok(json!({
        "folder": out.to_string_lossy(),
        "files": n,
        "tags_replaced": tags_changed,
        "uids_remapped": if a.remap_uids { scan.uid_count } else { 0 },
        "private_elements_removed": if a.remove_private { scan.private_count } else { 0 },
        "warnings": scan.warnings.iter().map(|w| clean_text(w)).collect::<Vec<_>>(),
        "note": "open the folder with open_dataset; it passes the PHI gate",
    }))
}
