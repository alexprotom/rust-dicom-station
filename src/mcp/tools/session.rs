//! Opening, describing and closing datasets; the session itself.

use std::path::PathBuf;

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::config::PhiPolicy;
use super::super::phi::{self, clean_text};
use super::super::session::Dataset;
use super::super::Core;
use super::NoArgs;
use crate::anonymize;
use crate::loader::{self, LoadedStudy};
use crate::progress::Progress;
use crate::workflow::select;

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenArgs {
    /// A folder under one of the configured roots. Everything DICOM inside
    /// it (recursively) becomes one dataset.
    pub path: PathBuf,
    /// Explicit files instead of a whole folder (each under a root).
    #[serde(default)]
    pub files: Vec<PathBuf>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DatasetArgs {
    /// A dataset handle such as `ds1`.
    pub dataset: String,
}

/// Replace every identifying value inside the loaded study by the alias,
/// so that under `redact` and `allow` nothing downstream (a report header,
/// an export field, a series description) still carries it.
fn scrub_in_memory(study: &mut LoadedStudy, values: &[String], alias: &str) {
    let scrub = |s: &mut String| {
        for v in values {
            if v.len() >= 2 && s.to_ascii_lowercase().contains(&v.to_ascii_lowercase()) {
                *s = phi::Redactor::scrub_with(s, v, alias);
            }
        }
    };
    study.meta.patient_name = alias.to_string();
    study.meta.patient_id = alias.to_string();
    scrub(&mut study.meta.study_description);
    for s in &mut study.series {
        s.patient_name = alias.to_string();
        s.patient_id = alias.to_string();
        scrub(&mut s.description);
        scrub(&mut s.study_description);
    }
    for w in &mut study.warnings {
        scrub(w);
    }
}

pub fn open_dataset(core: &mut Core, a: OpenArgs, p: &Progress) -> Result<Value> {
    let (real, root_label) = core.session.resolve_input(&a.path)?;
    let mut study = if a.files.is_empty() {
        if !real.is_dir() {
            bail!(
                "'{}' is not a folder; pass files for single files",
                real.display()
            );
        }
        loader::load_directory(&real, p)?
    } else {
        let mut files = Vec::with_capacity(a.files.len());
        for f in &a.files {
            let (rf, rl) = core.session.resolve_input(f)?;
            if rl != root_label {
                bail!("all files of one dataset must be under the same root");
            }
            files.push(rf);
        }
        loader::load_files(&files, "selected files", p)?
    };
    if study.series.is_empty() && study.structure_sets.is_empty() && study.doses.is_empty() {
        bail!("no DICOM data found under '{}'", real.display());
    }

    // The gate.
    p.set("Checking the headers for identifying data");
    let sample = phi::sample_files(&study, real.is_dir().then_some(real.as_path()));
    let verdict = phi::classify(&sample)?;
    let tags = verdict.tags.clone();
    let description = verdict.describe();
    let anonymized = verdict.is_anonymized();
    // The values go to the redactor first, whatever the policy: the
    // refusal below, and every later message, must not carry them either.
    let ids: Vec<String> = study
        .series
        .iter()
        .map(|s| s.patient_key().to_string())
        .chain(std::iter::once(study.meta.patient_id.clone()))
        .filter(|s| !s.is_empty() && s != "?")
        .collect();
    let alias = anonymize::patient_alias(&ids);
    let values = verdict.values().to_vec();
    core.session.add_values(values.clone());
    if !anonymized {
        match core.session.config.phi_policy {
            PhiPolicy::Refuse => bail!(
                "the dataset {description}. The policy is 'refuse': anonymize it first \
                 (Tools > Anonymize DICOM folder in the viewer, or the anonymize tool) and open \
                 the copy"
            ),
            // `allow` keeps the study as it is so an export can carry the
            // original identifiers; the redactor already holds the values,
            // so nothing that leaves the process does.
            PhiPolicy::Redact => scrub_in_memory(&mut study, &values, &alias),
            PhiPolicy::Allow => {}
        }
    }

    let ds = core.session.add_dataset(study, real, root_label, tags)?;
    Ok(describe(ds))
}

pub fn describe_dataset(core: &mut Core, a: DatasetArgs, _p: &Progress) -> Result<Value> {
    Ok(describe(core.session.dataset(&a.dataset)?))
}

/// The dataset as the client sees it. No patient tags: the study's own
/// `PatientMeta` is not read here at all.
pub fn describe(ds: &Dataset) -> Value {
    let st = &ds.study;
    let in_group = |uid: &str| -> Option<String> {
        st.fourd_groups
            .iter()
            .filter(|g| !g.dissolved)
            .find(|g| g.members.iter().any(|m| m.series_uid == uid))
            .map(|g| g.name.clone())
    };
    let series: Vec<Value> = st
        .series
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let mut v = json!({
                "number": i + 1,
                "modality": s.modality,
                "description": clean_text(&s.description),
                "files": s.files.len(),
                "active": i == st.active_series,
            });
            if let Some(n) = s.series_number {
                v["series_number"] = json!(n);
            }
            if let Some(g) = in_group(&s.uid) {
                v["fourd_group"] = json!(clean_text(&g));
            }
            if i == st.active_series && st.has_volume() {
                v["dims"] = json!(st.volume.dims);
                v["spacing_mm"] = json!(st.volume.spacing);
            }
            v
        })
        .collect();
    let groups: Vec<Value> = st
        .fourd_groups
        .iter()
        .enumerate()
        .filter(|(_, g)| !g.dissolved)
        .map(|(i, g)| {
            json!({
                "index": i + 1,
                "name": clean_text(&g.name),
                "phases": g.phase_members().iter().map(|&m| clean_text(&g.members[m].label)).collect::<Vec<_>>(),
                "reconstructions": g.members.iter().filter(|m| m.role != crate::fourd::Role::Phase).map(|m| clean_text(&m.label)).collect::<Vec<_>>(),
            })
        })
        .collect();
    let structure_sets: Vec<Value> = st
        .structure_sets
        .iter()
        .map(|ss| {
            json!({
                "label": clean_text(&ss.label),
                "rois": ss.rois.iter().map(|r| clean_text(&r.name)).collect::<Vec<_>>(),
            })
        })
        .collect();
    let seg_series: Vec<Value> = st
        .seg_series
        .iter()
        .map(|s| {
            json!({
                "label": clean_text(&s.label),
                "segments": s.segs.iter().map(|g| clean_text(&g.name)).collect::<Vec<_>>(),
                "dims": s.grid.dims,
            })
        })
        .collect();
    let doses: Vec<Value> = st
        .doses
        .iter()
        .enumerate()
        .map(|(i, d)| {
            json!({
                "number": i + 1,
                "label": clean_text(&d.label),
                "units": d.units,
                "max": d.max_dose,
                "dims": d.dims,
            })
        })
        .collect();
    let plans: Vec<Value> = st
        .plans
        .iter()
        .map(|pl| {
            json!({
                "label": clean_text(&pl.label),
                "kind": pl.plan_kind,
                "fractions": pl.n_fractions,
                "prescription_gy": pl.target_prescription_dose,
            })
        })
        .collect();
    json!({
        "dataset": ds.id,
        "origin": ds.origin.to_string_lossy(),
        "root": ds.root_label,
        "phi": {
            "status": if ds.phi_tags.is_empty() { "anonymized" } else { "identifying (redacted in memory)" },
            "tags": ds.phi_tags,
        },
        "series": series,
        "fourd_groups": groups,
        "structure_sets": structure_sets,
        "segmentation_series": seg_series,
        "doses": doses,
        "plans": plans,
        "warnings": st.warnings.len(),
    })
}

pub fn list_structures(core: &mut Core, a: DatasetArgs, _p: &Progress) -> Result<Value> {
    let ds = core.session.dataset(&a.dataset)?;
    let spacing = ds.study.volume.spacing;
    let items: Vec<Value> = select::list(&ds.study)
        .into_iter()
        .map(|e| {
            let mut v = json!({
                "name": clean_text(&e.name),
                "kind": match e.kind { select::Kind::Roi => "roi", select::Kind::Segment => "segment" },
                "set": clean_text(&e.set_label),
                "color": e.color,
            });
            if e.kind == select::Kind::Segment {
                if let Some(seg) = ds.study.seg_series[e.set].segs.get(e.idx) {
                    let ser = &ds.study.seg_series[e.set];
                    let sp = if ser.grid.dims == ds.study.volume.dims { spacing } else { ser.grid.spacing };
                    v["volume_cm3"] = json!(round1(seg.volume_cm3(sp)));
                }
            } else if let Some(roi) = ds
                .study
                .structure_sets
                .get(e.set)
                .and_then(|ss| ss.rois.get(e.idx))
            {
                v["roi_type"] = json!(clean_text(&roi.roi_type));
                v["contours"] = json!(roi.contours.len());
            }
            v
        })
        .collect();
    Ok(json!({ "dataset": ds.id, "structures": items }))
}

pub fn close_dataset(core: &mut Core, a: DatasetArgs, _p: &Progress) -> Result<Value> {
    core.session.close_dataset(&a.dataset)?;
    Ok(
        json!({ "closed": a.dataset, "open": core.session.datasets.iter().map(|d| d.id.clone()).collect::<Vec<_>>() }),
    )
}

pub fn describe_session(core: &mut Core, _a: NoArgs, _p: &Progress) -> Result<Value> {
    let s = &core.session;
    let bytes: usize = s
        .datasets
        .iter()
        .map(|d| {
            d.study.volume.data.len() * 2
                + d.study
                    .seg_series
                    .iter()
                    .map(|x| x.segs.iter().map(|g| g.mask.len()).sum::<usize>())
                    .sum::<usize>()
        })
        .sum();
    Ok(json!({
        "datasets": s.datasets.iter().map(|d| json!({"dataset": d.id, "root": d.root_label, "series": d.study.series.len()})).collect::<Vec<_>>(),
        "registrations": s.registrations.iter().map(|r| json!({"reg": r.id, "fixed": r.fixed.0, "moving": r.moving.0, "method": r.result.method.label()})).collect::<Vec<_>>(),
        "group_registrations": s.group_registrations.iter().map(|g| json!({"greg": g.id, "dataset": g.dataset, "group": clean_text(&g.group_name), "phases": g.phases.len()})).collect::<Vec<_>>(),
        "runs": s.runs.iter().map(|r| json!({"run": r.id, "dataset": r.dataset})).collect::<Vec<_>>(),
        "phi_policy": s.config.phi_policy.label(),
        "roots": s.config.roots.len(),
        "output_configured": s.config.output_dir.is_some(),
        "model_download_allowed": s.config.allow_model_download,
        "device": s.config.device,
        "resident_mb": bytes / (1024 * 1024),
        "identifying_values_known": s.redactor.read().expect("redactor lock").known_values(),
    }))
}

/// One decimal, for volumes in cm³.
pub fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// Two decimals, for millimetres.
pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Three decimals, for Dice and correlations.
pub fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

pub fn vec3(v: crate::geometry::Vec3) -> Value {
    json!([round2(v.x), round2(v.y), round2(v.z)])
}
