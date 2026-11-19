//! The posture model between datasets ([`crate::workflow::posture`]): the
//! tables, and on request the files, never an image or an identifier.
//!
//! The states are open datasets, each by its handle, in order; each stands
//! for its displayed series and the structure set drawn on it. What comes
//! back is the model's verdict ([`crate::posture::measure::Tables::verdict`])
//! and the four tables of the posture window (states, structures, bones and
//! joints, the model and its checks) as numbers - by default a summary of
//! them (`detail: "full"` for every column), every name through
//! [`clean_text`] and the whole answer through the redactor like every other
//! tool's; the states are named by their handles. With `write`, the tables
//! (CSV), the model (`.rdsposture`) and the whole path as a DICOM Deformable
//! Spatial Registration go into a folder of the session's output folder,
//! and with `export_t` the image and structures at that time as a derived
//! CT series and an RT Structure Set of the first state's study: at the
//! last state, the first state's image in the last one's position.
//!
//! **Templates** ([`crate::posture::template`]). A pair's model built with
//! `sample` also leaves its sample (`sample.json`: per bone its motion, the
//! joints, the organs' offsets from the bones, under the label the caller
//! gives and nothing read from the files). `build_posture_template` gathers
//! samples (files, folders searched for them, templates) into a template in
//! the output folder and reports its leave-one-out - each patient predicted
//! from the others against its own fitted bones, beside no motion at all -
//! and `predict_posture` moves one scan as the cohort moved: a model like
//! any other, whose second state is a prediction, not a scan. Templates are
//! found by name in the station's templates folder (the posture window's;
//! `posture_templates_dir` in `mcp.toml`), which is only read, or by path.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::Core;
use super::session::{round2, round3, vec3};
use super::NoArgs;
use crate::posture::template::{self, Template};
use crate::posture::{file as pfile, Anchor, Mode};
use crate::progress::Progress;
use crate::workflow::posture::{self as wp, Bodies, PostureOutcome, PostureRequest, StateInput};

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PostureArgs {
    /// The states, in order: dataset handles (each its displayed series
    /// and the structure set drawn on it). Two or more; more make a chain.
    pub datasets: Vec<String>,
    /// What is held still: `auto` (the default: the sacrum when every
    /// scan sees all of it, else the middle vertebra every scan sees whole,
    /// else the largest such bone, else the image), a structure contoured
    /// on every state (its name), `image` (a rigid registration of the
    /// images) or `scanner` (nothing: for scans that share a frame of
    /// reference).
    #[serde(default)]
    pub anchor: Option<String>,
    /// The rigid bodies by name. Omitted: the bones every state has (and
    /// the anchor); an empty list: none.
    #[serde(default)]
    pub bodies: Option<Vec<String>>,
    /// `diffeomorphic` (rigid bodies on screws, invertible) or `linear`
    /// (straight lines on the end-to-end displacement).
    #[serde(default)]
    pub mode: Option<String>,
    /// Add a B-spline residual deformation on top of the bodies.
    #[serde(default)]
    pub residual: bool,
    /// Velocity lattice, mm (6).
    #[serde(default)]
    pub lattice_mm: Option<f64>,
    /// Analysis grid, mm (2).
    #[serde(default)]
    pub analysis_mm: Option<f64>,
    /// Reach of a body in the blend, mm (40).
    #[serde(default)]
    pub reach_mm: Option<f64>,
    /// A posture parameter per state (an angle, a day, a filling).
    #[serde(default)]
    pub posture: Option<Vec<f64>>,
    /// Joints to report: [name, first body, second body]. Empty: the
    /// presets (hips, lumbosacral, neighbouring vertebrae; two bodies and
    /// nothing else, the two).
    #[serde(default)]
    pub joints: Vec<[String; 3]>,
    /// The water-equivalent path length to each structure's centroid along
    /// a beam at this IEC 61217 gantry angle, degrees (with
    /// `wepl_couch_deg`, read in the first state's patient position):
    /// first, last and its change per structure. Omitted with the couch
    /// angle: no WEPL.
    #[serde(default)]
    pub wepl_gantry_deg: Option<f64>,
    /// The patient support angle of that beam, degrees (0).
    #[serde(default)]
    pub wepl_couch_deg: Option<f64>,
    /// A chain of three or more: rigid bodies on curves through the
    /// states, velocities continuous at each (true), or a constant turn per
    /// segment (false).
    #[serde(default)]
    pub smooth_chain: Option<bool>,
    /// Fit and draw the scans as they are, couch and chair included. By
    /// default each state is its patient only: outside its EXTERNAL
    /// structure set to air, when it has one.
    #[serde(default)]
    pub whole_scan: bool,
    /// `summary` (the default: the verdict, the bones, the joints, the
    /// checks and a line per structure) or `full` (every column of the four
    /// tables).
    #[serde(default)]
    pub detail: Option<String>,
    /// Also write tables.csv, the model file and the deformation as DICOM.
    #[serde(default)]
    pub write: bool,
    /// With `write`: the image at this time (0 to the number of states
    /// less one; the last state's for the first image in the last
    /// position) as a derived CT series and the first state's structures
    /// carried to it, in `frame/` of the folder.
    #[serde(default)]
    pub export_t: Option<f64>,
    /// Folder under the session's output folder for `write`.
    #[serde(default)]
    pub folder: Option<String>,
    /// A pair only: also write the model's sample for a posture template
    /// (`sample.json` in the folder; implies `write`): per bone its motion,
    /// the joints where bones touch, per organ its offset from where the
    /// bones carry it. Bones contoured on both scans are what it is made of.
    #[serde(default)]
    pub sample: bool,
    /// The sample's label in the template (a pseudonymised code; never
    /// read from the files). Omitted: the folder's name.
    #[serde(default)]
    pub label: Option<String>,
}

fn full_detail(d: Option<&str>) -> Result<bool> {
    match d {
        None | Some("summary") => Ok(false),
        Some("full") => Ok(true),
        Some(d) => bail!("detail must be summary or full (got '{d}')"),
    }
}

pub fn build_posture_model(core: &mut Core, a: PostureArgs, p: &Progress) -> Result<Value> {
    if a.datasets.len() < 2 {
        bail!("a posture model needs at least two datasets");
    }
    if a.sample && a.datasets.len() != 2 {
        bail!("a sample is of a pair: two datasets");
    }
    let mode = match a.mode.as_deref() {
        None => Mode::Diffeomorphic,
        Some(m) => Mode::from_key(m)
            .ok_or_else(|| anyhow::anyhow!("mode must be diffeomorphic or linear (got '{m}')"))?,
    };
    let req = PostureRequest {
        anchor: a
            .anchor
            .as_deref()
            .map(Anchor::parse)
            .unwrap_or(Anchor::Auto),
        bodies: match &a.bodies {
            None => Bodies::Bones,
            Some(v) if v.is_empty() => Bodies::None,
            Some(v) => Bodies::Named(v.clone()),
        },
        mode,
        lattice_mm: a.lattice_mm.unwrap_or(6.0).clamp(2.0, 30.0),
        analysis_mm: a.analysis_mm.unwrap_or(2.0).clamp(0.5, 10.0),
        reach_mm: a.reach_mm.unwrap_or(40.0).clamp(5.0, 200.0),
        residual: a.residual.then(wp::default_residual),
        joints: joints(&a.joints),
        wepl: wepl(a.wepl_gantry_deg, a.wepl_couch_deg),
        smooth_chain: a.smooth_chain.unwrap_or(true),
        patient_only: !a.whole_scan,
        sample: a.sample,
        ..PostureRequest::default()
    };
    let full = full_detail(a.detail.as_deref())?;
    let outcome = {
        let mut states = Vec::with_capacity(a.datasets.len());
        for (i, id) in a.datasets.iter().enumerate() {
            let ds = core.session.dataset(id)?;
            states.push(StateInput {
                label: id.clone(),
                study: &ds.study,
                structure_set: None,
                posture: a.posture.as_ref().and_then(|v| v.get(i).copied()),
            });
        }
        wp::build(&states, &req, p)?
    };
    let mut out = report(&outcome, full);
    if a.export_t.is_some() && !a.write {
        bail!("export_t writes the image at that time: ask for write too");
    }
    if a.write || a.sample {
        let name = a
            .folder
            .clone()
            .unwrap_or_else(|| format!("posture-{}", a.datasets.join("-")));
        let last_for = core
            .session
            .dataset(a.datasets.last().expect("two or more"))?
            .study
            .volume
            .frame_of_reference_uid
            .clone();
        let (dir, mut files, frame) = write_files(
            core,
            &outcome,
            &name,
            &a.datasets[0],
            &last_for,
            a.export_t,
            p,
        )?;
        if let Some(f) = frame {
            out["frame"] = f;
        }
        if a.sample {
            let mut s = outcome
                .sample
                .clone()
                .ok_or_else(|| anyhow::anyhow!("the model left no sample"))?;
            s.label = a
                .label
                .as_deref()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| {
                    dir.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default()
                });
            s.write(&dir.join(template::SAMPLE_FILE))?;
            files.push(template::SAMPLE_FILE.into());
            out["sample"] = json!({
                "label": clean_text(&s.label),
                "file": template::SAMPLE_FILE,
                "bones": s.bodies.len(),
                "joints": s.joints.len(),
                "organs": s.organs.len(),
            });
        }
        out["folder"] = json!(dir.to_string_lossy());
        out["files"] = json!(files);
    }
    Ok(out)
}

fn joints(j: &[[String; 3]]) -> Vec<(String, String, String)> {
    j.iter()
        .map(|[n, x, y]| (n.clone(), x.clone(), y.clone()))
        .collect()
}

fn wepl(gantry: Option<f64>, couch: Option<f64>) -> Option<wp::WeplBeam> {
    (gantry.is_some() || couch.is_some()).then(|| wp::WeplBeam {
        gantry_deg: gantry.unwrap_or(0.0),
        couch_deg: couch.unwrap_or(0.0),
        hlut: None,
    })
}

/// The tables of a model as the client reads them: a summary, or every
/// column.
fn report(outcome: &PostureOutcome, full: bool) -> Value {
    let t = &outcome.tables;
    let verdict = t.verdict();
    let verdict_json = json!({
        "level": verdict.level.label(),
        "findings": verdict
            .lines
            .iter()
            .map(|(l, line)| json!({"level": l.label(), "text": clean_text(line)}))
            .collect::<Vec<_>>(),
    });
    if !full {
        return json!({
            "verdict": verdict_json,
            "states": t.states.iter().map(|s| json!({
                "dataset": s.label,
                "posture": s.posture,
                "into_anchor": {
                    "translation_mm": vec3(s.to_anchor.translation),
                    "rotation_deg": s.to_anchor.rotation_deg.map(round2),
                    "anchor_fit_rms_mm": round2(s.to_anchor.residual_mm),
                },
            })).collect::<Vec<_>>(),
            "bodies": t.bodies.iter().filter(|b| !b.anchor).map(|b| json!({
                "body": clean_text(&b.name),
                "angle_deg": round2(b.screw.angle_deg),
                "fit_mean_mm": round2(b.fit_mean_mm),
                "stretch_p95": round3(b.stretch_p95),
            })).collect::<Vec<_>>(),
            "joints": t.joints.iter().map(|j| json!({
                "joint": clean_text(&j.name),
                "angles_deg_rl_ap_si": j.at.last().copied().unwrap_or_default().map(round2),
            })).collect::<Vec<_>>(),
            "structures": t.structures.iter().map(|s| json!({
                "structure": clean_text(&s.name),
                "used_in_fit": s.used,
                "cut_by_a_scan": s.is_cut(),
                "check_dice": s.check.map(|d| round3(d.dice)),
                "check_centroid_mm": s.check.map(|d| round2(d.centroid_mm)),
                "contoured_shift_mm": s.contoured_shift().map(vec3),
            })).collect::<Vec<_>>(),
            "model": {
                "mode": t.model.mode.key(),
                "anchor": clean_text(&t.model.anchor),
                "jacobian_min": round3(t.model.jacobian_min),
                "folded_fraction": t.model.folded_fraction,
                "inverse_consistency_end_p95_mm": round3(t.model.inverse_mm[1][1]),
                "check_dice_used": t.model.check_used_dice.map(round3),
                "check_dice_not_used": t.model.check_unused_dice.map(round3),
                "overlap_fraction": round3(t.model.overlap_fraction),
            },
            "warnings": outcome.warnings.iter().map(|w| clean_text(w)).collect::<Vec<_>>(),
            "caveat": wp::CAVEAT,
        });
    }
    json!({
    "verdict": verdict_json,
    "states": t.states.iter().map(|s| json!({
        "dataset": s.label,
        "posture": s.posture,
        "position": s.position,
        "grid": s.dims,
        "into_anchor": {
            "translation_mm": vec3(s.to_anchor.translation),
            "rotation_deg": s.to_anchor.rotation_deg.map(round2),
            "anchor_fit_rms_mm": round2(s.to_anchor.residual_mm),
        },
        "body_cm3": s.body_cm3.map(round2),
        "body_change_cm3": s.body_change_cm3.map(round2),
    })).collect::<Vec<_>>(),
    "structures": t.structures.iter().map(|s| {
        let (dv, pct) = s.volume_change();
        let end = s.at.last().copied().unwrap_or_default();
        json!({
            "structure": clean_text(&s.name),
            "used_in_fit": s.used,
            "volume_cm3": s.volume_cm3.iter().map(|v| round2(*v)).collect::<Vec<_>>(),
            "volume_change_cm3": round2(dv),
            "volume_change_pct": if pct.is_finite() { Some(round2(pct)) } else { None },
            "cut_by_a_scan": s.is_cut(),
            "volume_change_common_fov_cm3": round2(s.common_change()),
            "contoured_shift_mm": s.contoured_shift().map(vec3),
            "model_shift_mm": s.model_shift(s.at.len().saturating_sub(1)).map(vec3),
            "path_mm": round2(end.path_mm),
            "rotation_deg": round2(end.rotation_deg),
            "jacobian": end.jacobian.map(round3),
            "after_rigid": s.deformation.map(|d| json!({"dice": round3(d.dice), "hd95_mm": round2(d.hd95_mm), "msd_mm": round2(d.msd_mm)})),
            "model_check": s.check.map(|d| json!({"dice": round3(d.dice), "hd95_mm": round2(d.hd95_mm), "msd_mm": round2(d.msd_mm), "centroid_mm": round2(d.centroid_mm)})),
            "mean_hu": s.mean_hu.iter().map(|v| v.map(round2)).collect::<Vec<_>>(),
            "wepl_mm": (!s.wepl_mm.is_empty()).then(|| json!({
                "first": s.wepl_mm.first().map(|v| round2(*v)),
                "last": s.wepl_mm.last().map(|v| round2(*v)),
                "change": s.wepl_change(s.wepl_mm.len() - 1).map(round2),
            })),
        })
    }).collect::<Vec<_>>(),
    "bodies": t.bodies.iter().map(|b| json!({
        "body": clean_text(&b.name),
        "anchor": b.anchor,
        "angle_deg": round2(b.screw.angle_deg),
        "axis": vec3(b.screw.axis),
        "pivot_mm": vec3(b.screw.point),
        "travel_mm": round2(b.screw.travel_mm),
        "fit_mean_mm": round2(b.fit_mean_mm),
        "principal_axes_start": b.seeded,
        "largest_stretch": round3(b.max_stretch),
        "stretch_p95": round3(b.stretch_p95),
    })).collect::<Vec<_>>(),
    "joints": t.joints.iter().map(|j| json!({
        "joint": clean_text(&j.name),
        "first": clean_text(&j.first),
        "second": clean_text(&j.second),
        "angles_deg_rl_ap_si": j.at.last().copied().unwrap_or_default().map(round2),
    })).collect::<Vec<_>>(),
    "model": {
        "mode": t.model.mode.key(),
        "anchor": clean_text(&t.model.anchor),
        "lattice_mm": round2(t.model.lattice_mm),
        "inverse_consistency_mm": {
            "half_way_max": round3(t.model.inverse_mm[0][0]),
            "half_way_p95": round3(t.model.inverse_mm[0][1]),
            "end_max": round3(t.model.inverse_mm[1][0]),
            "end_p95": round3(t.model.inverse_mm[1][1]),
        },
        "jacobian_min": round3(t.model.jacobian_min),
        "folded_fraction": t.model.folded_fraction,
        "check_dice_used": t.model.check_used_dice.map(round3),
        "check_dice_not_used": t.model.check_unused_dice.map(round3),
        "overlap_fraction": round3(t.model.overlap_fraction),
        "wepl_beam": t.model.beam,
        "provenance": t.model.provenance.iter().map(|l| clean_text(l)).collect::<Vec<_>>(),
    },
    "warnings": outcome.warnings.iter().map(|w| clean_text(w)).collect::<Vec<_>>(),
    "caveat": wp::CAVEAT,
    })
}

/// A model's files in a new folder `name` of the output folder: the
/// tables, the model, the path as a DICOM deformable registration from the
/// first dataset's frame onto `last_for` and, at `export_t`, the image and
/// structures at that time. Returns the folder, the files and what the
/// frame holds.
fn write_files(
    core: &mut Core,
    outcome: &PostureOutcome,
    name: &str,
    first: &str,
    last_for: &str,
    export_t: Option<f64>,
    p: &Progress,
) -> Result<(PathBuf, Vec<String>, Option<Value>)> {
    let dir = core.session.fresh_out_subdir(name)?;
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("tables.csv"),
        crate::motion::csv_file_bytes(&outcome.tables.csv()),
    )?;
    pfile::write(
        &dir.join(format!("model.{}", pfile::EXTENSION)),
        &outcome.model,
        &outcome.occupancy,
    )?;
    let first = core.session.dataset(first)?;
    wp::export_registration(
        &outcome.model,
        &first.study,
        last_for,
        &dir.join("posture_reg.dcm"),
    )?;
    let mut files = vec![
        "tables.csv".to_string(),
        format!("model.{}", pfile::EXTENSION),
        "posture_reg.dcm".to_string(),
    ];
    let mut frame = None;
    if let Some(at) = export_t {
        let span = outcome.model.span();
        if !(0.0..=span).contains(&at) {
            bail!("export_t must be within 0 to {span}");
        }
        let e = wp::export_frame(
            &outcome.model,
            &outcome.images,
            &first.study,
            None,
            at,
            crate::posture::apply::Show::StartForward,
            &dir.join("frame"),
            p,
        )?;
        frame = Some(json!({
            "t": at,
            "images": e.image_files.len(),
            "structure_sets": e.structure_files.len(),
            "series_description_says": "interpolated by the posture model, not a scan",
        }));
        files.push("frame/".into());
    }
    Ok((dir, files, frame))
}

// ---- templates ------------------------------------------------------------

fn templates_dir(core: &Core) -> PathBuf {
    core.session
        .config
        .posture_templates_dir
        .clone()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(template::templates_dir)
}

fn stem(p: &Path) -> String {
    p.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

pub fn list_posture_templates(core: &mut Core, _a: NoArgs, _p: &Progress) -> Result<Value> {
    let listed: Vec<Value> = template::list(&templates_dir(core))
        .iter()
        .map(|f| match Template::read(f) {
            Ok(t) => json!({
                "name": clean_text(&stem(f)),
                "patients": t.samples.len(),
                "bones": t.bones().len(),
                "summary": clean_text(t.summary().first().map(String::as_str).unwrap_or("")),
            }),
            Err(_) => json!({"name": clean_text(&stem(f)), "error": "not a posture template this station reads"}),
        })
        .collect();
    Ok(json!({ "templates": listed }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TemplateArgs {
    /// Where the samples are: `sample.json` files build_posture_model wrote
    /// (with `sample`), folders (every sample.json and template in them and
    /// below: the session's output folder, say) or templates (all their
    /// patients). Paths as the server reports them (`output/session/...`,
    /// `output/...`) or under a root.
    pub samples: Vec<String>,
    /// The template's name: written as `<name>.rdsposture-template` into a
    /// folder of the output folder, for predict_posture (by its path) and
    /// the viewer's posture window (*Open a template...*). Omitted: nothing
    /// is written, only the report (an existing template's, say).
    #[serde(default)]
    pub name: Option<String>,
    /// `summary` (the default: the error by group of bones and by organ, a
    /// line per patient) or `full` (also every bone of every patient).
    #[serde(default)]
    pub detail: Option<String>,
}

pub fn build_posture_template(core: &mut Core, a: TemplateArgs, _p: &Progress) -> Result<Value> {
    let full = full_detail(a.detail.as_deref())?;
    if a.samples.is_empty() {
        bail!("samples: give the sample files, folders or templates of the cohort");
    }
    let mut files: Vec<PathBuf> = Vec::new();
    for given in &a.samples {
        let (path, _) = core.session.resolve_input(Path::new(given))?;
        if path.is_dir() {
            files.extend(template::find_samples(&path));
        } else {
            files.push(path);
        }
    }
    files.sort();
    files.dedup();
    let mut samples = Vec::new();
    let mut skipped = 0usize;
    for f in &files {
        match template::read_samples(f) {
            Ok(v) => samples.extend(v),
            Err(_) => skipped += 1,
        }
    }
    if samples.is_empty() {
        bail!(
            "no sample among the {} file(s) found: build_posture_model with sample writes one per pair",
            files.len()
        );
    }
    let mut warnings = Vec::new();
    if skipped > 0 {
        warnings.push(format!(
            "{skipped} file(s) were neither a sample nor a template"
        ));
    }
    let mut labels: Vec<&str> = samples.iter().map(|s| s.label.as_str()).collect();
    labels.sort_unstable();
    let twice: Vec<&str> = labels
        .windows(2)
        .filter(|w| w[0] == w[1])
        .map(|w| w[0])
        .collect();
    if !twice.is_empty() {
        warnings.push(format!(
            "label(s) given more than once ({}): a patient twice counts twice",
            twice.join(", ")
        ));
    }
    let t = Template::new(samples);
    let mut out = template_report(&t, full);
    if !warnings.is_empty() {
        out["warnings"] = json!(warnings.iter().map(|w| clean_text(w)).collect::<Vec<_>>());
    }
    if let Some(name) = a.name.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        let dir = core
            .session
            .fresh_out_subdir(&format!("posture-template-{name}"))?;
        std::fs::create_dir_all(&dir)?;
        let safe: String = name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let file = dir.join(format!("{safe}.{}", template::EXTENSION));
        t.write(&file)?;
        out["file"] = json!(file.to_string_lossy());
    }
    Ok(out)
}

/// A template as the client reads it: what it holds and its leave-one-out.
fn template_report(t: &Template, full: bool) -> Value {
    let edges = t.edges(None);
    let used = |joint: bool| {
        edges
            .values()
            .filter(|e| e.joint == joint && e.n >= template::MIN_EDGE_SAMPLES)
            .count()
    };
    let organs = t
        .organs(None)
        .values()
        .filter(|o| o.n >= template::MIN_ORGAN_SAMPLES)
        .count();
    let loo = template::leave_one_out(t);
    let spread = |s: template::Spread| json!({"mean": round2(s.mean), "median": round2(s.median), "p90": round2(s.p90)});
    let patients: Vec<Value> = loo
        .iter()
        .map(|p| {
            let reached: Vec<&template::LooBody> =
                p.bodies.iter().filter(|b| b.error_mm.is_some()).collect();
            let mean = |f: &dyn Fn(&template::LooBody) -> f64| {
                (!reached.is_empty()).then(|| {
                    round2(reached.iter().map(|b| f(b)).sum::<f64>() / reached.len() as f64)
                })
            };
            let mut v = json!({
                "label": clean_text(&p.label),
                "root": clean_text(&p.root),
                "bones": reached.len(),
                "unreached": p.bodies.len() - reached.len(),
                "error_mm": mean(&|b| b.error_mm.unwrap_or(0.0)),
                "no_motion_mm": mean(&|b| b.still_mm),
            });
            if full {
                v["by_bone"] = json!(p
                    .bodies
                    .iter()
                    .map(|b| json!({
                        "bone": clean_text(&b.name),
                        "group": b.group,
                        "hops": b.hops,
                        "error_mm": b.error_mm.map(round2),
                        "no_motion_mm": round2(b.still_mm),
                    }))
                    .collect::<Vec<_>>());
                v["organs"] = json!(p
                    .organs
                    .iter()
                    .map(|o| json!({
                        "organ": clean_text(&o.name),
                        "no_motion_mm": round2(o.still_mm),
                        "own_bones_mm": round2(o.bones_mm),
                        "template_mm": round2(o.template_mm),
                    }))
                    .collect::<Vec<_>>());
            }
            v
        })
        .collect();
    json!({
        "patients": t.samples.len(),
        "labels": t.samples.iter().map(|s| clean_text(&s.label)).collect::<Vec<_>>(),
        "summary": t.summary().iter().map(|l| clean_text(l)).collect::<Vec<_>>(),
        "relations": {
            "joints": used(true),
            "direct": used(false),
            "organ_offsets": organs,
            "seen_on_at_least": template::MIN_EDGE_SAMPLES,
        },
        "bones_known": t.bones().len(),
        "leave_one_out": {
            "what": "each patient predicted from the others: the mean distance at each bone's seven points between the predicted and the patient's own fitted motion relative to the root, mm, beside no motion at all; organs by their centroid against where they were contoured on the second scan",
            "groups": template::loo_groups(&loo).iter().map(|g| json!({
                "group": g.group,
                "bones": g.n,
                "template_mm": spread(g.template),
                "no_motion_mm": spread(g.still),
            })).collect::<Vec<_>>(),
            "organs": template::loo_organs(&loo).iter().map(|o| json!({
                "organ": clean_text(&o.name),
                "patients": o.n,
                "no_motion_mm": round2(o.still_mm),
                "own_bones_mm": round2(o.bones_mm),
                "template_mm": round2(o.template_mm),
            })).collect::<Vec<_>>(),
            "patients": patients,
        },
        "caveat": "a template predicts how a cohort moved, not how this patient will: check a prediction against a scan before relying on it",
    })
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PredictArgs {
    /// The scan: a dataset handle (its displayed series and the structure
    /// set drawn on it, its bones contoured under the names the cohort's
    /// were: TotalSegmentator's, say).
    pub dataset: String,
    /// The template: a name list_posture_templates gives (the station's
    /// posture templates, as the posture window lists them), or a template
    /// file (`output/session/...` as build_posture_template reported it, or
    /// under a root).
    pub template: String,
    /// Move the organs the template has offsets for before the bones carry
    /// them (true); false: the bones alone.
    #[serde(default)]
    pub organs: Option<bool>,
    /// Velocity lattice, mm (6).
    #[serde(default)]
    pub lattice_mm: Option<f64>,
    /// Analysis grid, mm (2).
    #[serde(default)]
    pub analysis_mm: Option<f64>,
    /// Reach of a body in the blend, mm (40).
    #[serde(default)]
    pub reach_mm: Option<f64>,
    /// Joints to report: [name, first body, second body]; empty: the
    /// presets.
    #[serde(default)]
    pub joints: Vec<[String; 3]>,
    /// The WEPL to each structure's centroid along a beam at this IEC 61217
    /// gantry angle, degrees (with `wepl_couch_deg`): before and after.
    #[serde(default)]
    pub wepl_gantry_deg: Option<f64>,
    /// The patient support angle of that beam, degrees (0).
    #[serde(default)]
    pub wepl_couch_deg: Option<f64>,
    /// Move and draw the scan as it is, couch included; by default the
    /// patient only (outside its EXTERNAL to air).
    #[serde(default)]
    pub whole_scan: bool,
    /// `summary` (the default) or `full`, as build_posture_model.
    #[serde(default)]
    pub detail: Option<String>,
    /// Also write tables.csv, the model file and the deformation as DICOM.
    #[serde(default)]
    pub write: bool,
    /// With `write`: the image at this time (0 to 1; 1 for the predicted
    /// position) as a derived CT series and the structures carried to it,
    /// in `frame/` of the folder.
    #[serde(default)]
    pub export_t: Option<f64>,
    /// Folder under the session's output folder for `write`.
    #[serde(default)]
    pub folder: Option<String>,
}

pub fn predict_posture(core: &mut Core, a: PredictArgs, p: &Progress) -> Result<Value> {
    let full = full_detail(a.detail.as_deref())?;
    if a.export_t.is_some() && !a.write {
        bail!("export_t writes the image at that time: ask for write too");
    }
    let path = template_path(core, &a.template)?;
    let t = Template::read(&path)?;
    let req = PostureRequest {
        lattice_mm: a.lattice_mm.unwrap_or(6.0).clamp(2.0, 30.0),
        analysis_mm: a.analysis_mm.unwrap_or(2.0).clamp(0.5, 10.0),
        reach_mm: a.reach_mm.unwrap_or(40.0).clamp(5.0, 200.0),
        joints: joints(&a.joints),
        wepl: wepl(a.wepl_gantry_deg, a.wepl_couch_deg),
        patient_only: !a.whole_scan,
        ..PostureRequest::default()
    };
    let outcome = {
        let ds = core.session.dataset(&a.dataset)?;
        let state = StateInput {
            label: a.dataset.clone(),
            study: &ds.study,
            structure_set: None,
            posture: None,
        };
        wp::predict(&state, &t, &req, a.organs.unwrap_or(true), p)?
    };
    let mut out = report(&outcome, full);
    out["template"] = json!({
        "name": clean_text(&stem(&path)),
        "patients": t.samples.len(),
    });
    if a.write {
        let name = a
            .folder
            .clone()
            .unwrap_or_else(|| format!("posture-predicted-{}", a.dataset));
        // The predicted state is the scan moved: it keeps the scan's frame.
        let own_for = core
            .session
            .dataset(&a.dataset)?
            .study
            .volume
            .frame_of_reference_uid
            .clone();
        let (dir, files, frame) =
            write_files(core, &outcome, &name, &a.dataset, &own_for, a.export_t, p)?;
        if let Some(mut f) = frame {
            f["series_description_says"] = json!("predicted by a posture template, not a scan");
            out["frame"] = f;
        }
        out["folder"] = json!(dir.to_string_lossy());
        out["files"] = json!(files);
    }
    Ok(out)
}

/// A template by name (in the templates folder, the file's stem, case
/// aside) or by path.
fn template_path(core: &Core, given: &str) -> Result<PathBuf> {
    let listed = template::list(&templates_dir(core));
    let g = given.trim();
    if let Some(f) = listed.iter().find(|f| stem(f).eq_ignore_ascii_case(g)) {
        return Ok(f.clone());
    }
    match core.session.resolve_input(Path::new(g)) {
        Ok((p, _)) => Ok(p),
        Err(e) => {
            let names: Vec<String> = listed.iter().map(|f| stem(f)).collect();
            bail!(
                "no posture template '{g}' among the station's ({}) and not a file: {e}",
                if names.is_empty() {
                    "none".to_string()
                } else {
                    names.join(", ")
                }
            )
        }
    }
}
