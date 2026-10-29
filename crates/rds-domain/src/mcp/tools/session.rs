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
        *s = phi::Redactor::scrub_with(s, values, alias);
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
    let study = if a.files.is_empty() {
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
    let dir = real.is_dir().then(|| real.clone());
    admit(core, study, dir.as_deref(), real, root_label, p)
}

/// The gate: check the headers of a loaded study for identifying data,
/// apply the policy, and file the study as a dataset. `dir` is the folder
/// it was read from when it was a whole folder (its loose files are
/// checked too); `origin` and `root_label` are what the dataset reports.
pub fn admit(
    core: &mut Core,
    mut study: LoadedStudy,
    dir: Option<&std::path::Path>,
    origin: PathBuf,
    root_label: String,
    p: &Progress,
) -> Result<Value> {
    p.set("Checking the headers for identifying data");
    let sample = phi::sample_files(&study, dir);
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

    let ds = core.session.add_dataset(study, origin, root_label, tags)?;
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
    let planar: Vec<Value> = st
        .planar_images
        .iter()
        .enumerate()
        .map(|(i, im)| {
            let mut v = json!({
                "number": i + 1,
                "label": clean_text(&im.label),
                "kind": match im.kind {
                    crate::extras::PlanarKind::Projection => "projection",
                    crate::extras::PlanarKind::Additional => "additional",
                },
                "modality": im.modality,
                "pixels": [im.cols, im.rows],
                "spacing_mm": im.spacing,
                "geometry": if im.camera.is_some() {
                    "rt_image".to_string()
                } else {
                    match &im.geometry_missing {
                        Some(m) => format!("missing: {m}"),
                        None => "none".to_string(),
                    }
                },
            });
            if let Some(b) = im.referenced_beam {
                v["beam"] = json!(b);
            }
            if im.megavoltage {
                v["megavoltage"] = json!(true);
            }
            v
        })
        .collect();
    // The reports as their tables: never the SR tree's person names, and
    // never a page's free lines (a physician's name sits there).
    let reports: Vec<Value> = st
        .reports
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let tables: Vec<Value> = r
                .all_tables(true)
                .into_iter()
                .map(|t| {
                    let truncated = t.rows.len() > REPORT_ROWS_CAP;
                    let mut v = json!({
                        "title": clean_text(&t.title),
                        "columns": t.columns.iter().map(|c| clean_text(c)).collect::<Vec<_>>(),
                        "rows": t.rows.iter().take(REPORT_ROWS_CAP).map(|row| {
                            row.iter().map(|c| clean_text(c)).collect::<Vec<_>>()
                        }).collect::<Vec<_>>(),
                    });
                    if truncated {
                        v["rows_total"] = json!(t.rows.len());
                    }
                    v
                })
                .collect();
            let mut v = json!({
                "number": i + 1,
                "label": clean_text(&r.label),
                "kind": match r.kind {
                    crate::report::ReportKind::Structured => "structured_report",
                    crate::report::ReportKind::DoseReport => "dose_report",
                    crate::report::ReportKind::ProtocolPage => "protocol_page",
                    crate::report::ReportKind::Document => "document",
                },
                "modality": r.modality,
                "facts": r.info.iter().map(|(k, v)| json!([clean_text(k), clean_text(v)])).collect::<Vec<_>>(),
                "tables": tables,
            });
            if !r.date.is_empty() {
                v["date"] = json!(r.date);
            }
            if r.page.is_some() {
                v["page"] = json!(true);
            }
            v
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
        "planar_images": planar,
        "reports": reports,
        "sessions": sessions_summary(st),
        "warnings": st.warnings.len(),
    })
}

/// The treatment sessions of a dataset ([`crate::sessions`]), as counts and
/// outcomes. No date and no time of day: a session is its number in the
/// course and its day counted from the first session's (`day` 0), stricter
/// than a report's `date` and in step with the archive listing, which says
/// no more of a date than its year. Beam names come from the plan and the
/// records and pass the redactor like every label.
pub fn sessions_summary(st: &LoadedStudy) -> Vec<Value> {
    use crate::sessions;
    let list = sessions::sessions(st);
    let first = list
        .first()
        .and_then(|s| sessions::seconds(&format!("{}000000", s.date)));
    let status = |r: usize| -> String {
        st.treat_records
            .get(r)
            .map(|rec| clean_text(sessions::record_status(rec)))
            .unwrap_or_default()
    };
    let n_reports =
        |pairs: &[sessions::Pair]| -> usize { pairs.iter().map(|p| p.reports.len()).sum() };
    list.iter()
        .map(|s| {
            let day = first
                .zip(sessions::seconds(&format!("{}000000", s.date)))
                .map(|(a, b)| (b - a) / 86_400);
            let images: Vec<String> = s
                .series
                .iter()
                .filter_map(|&i| st.series.get(i))
                .map(|se| se.modality.clone())
                .collect();
            let steps: Vec<Value> = s
                .steps
                .iter()
                .map(|step| {
                    let mut v = json!({
                        "beam": clean_text(&step.name),
                        "setup_pairs": step.setup.len(),
                        "records": step.records.iter().map(|&r| status(r)).collect::<Vec<_>>(),
                        "all_normal": step.all_normal,
                        "verification_pairs": step.verification.len(),
                        "dose_reports": n_reports(&step.setup) + n_reports(&step.verification),
                    });
                    if let Some(c) = step
                        .treatment_reg
                        .and_then(|r| st.registrations.get(r))
                        .and_then(sessions::correction)
                    {
                        let r = |x: f64, d: f64| (x * d).round() / d;
                        v["correction"] = json!({
                            "translation_mm": [r(c[0], 10.0), r(c[1], 10.0), r(c[2], 10.0)],
                            "rotation_deg": [r(c[3], 100.0), r(c[4], 100.0), r(c[5], 100.0)],
                        });
                    }
                    v
                })
                .collect();
            json!({
                "session": s.number,
                "of": s.of,
                "day": day,
                "images": images,
                "imaging_registrations": s.image_regs.len(),
                "ct_to_ct_registrations": s.ct_regs.len(),
                "steps": steps,
                "records_without_step": s.loose_records.iter().map(|&r| status(r)).collect::<Vec<_>>(),
                "pairs_without_step": s.loose_pairs.len(),
                "reports": s.reports().count(),
            })
        })
        .collect()
}

/// Rows of one report table a description carries at most: an SR tree
/// can run to hundreds, and the dose table is what matters.
const REPORT_ROWS_CAP: usize = 200;

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
                    v["volume_cm3"] = json!(round2(seg.volume_cm3(sp)));
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
        "setups": s.setups.iter().map(|x| json!({"setup": x.id, "dataset": x.dataset, "method": x.result.method.key()})).collect::<Vec<_>>(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extras::{
        PlanarImage, PlanarKind, RecordBeam, RegMatrixItem, SpatialReg, TreatRecord,
    };
    use crate::rtplan::{BeamInfo, PlanInfo};

    /// Two treatment days a week apart, each one beam with a setup pair,
    /// a correction and a record.
    fn course() -> LoadedStudy {
        let mut st = LoadedStudy {
            plans: vec![PlanInfo {
                sop_instance_uid: "9.9.1".into(),
                beams: vec![BeamInfo {
                    number: 2,
                    name: "LPO".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        #[rustfmt::skip]
        let shifted = [
            1.0, 0.0, 0.0, -0.8,
            0.0, 1.0, 0.0, 4.9,
            0.0, 0.0, 1.0, -4.4,
            0.0, 0.0, 0.0, 1.0,
        ];
        for day in ["20240917", "20240924"] {
            st.registrations.push(SpatialReg {
                datetime: format!("{day}105403"),
                references: vec!["9.9.1".into()],
                items: vec![RegMatrixItem {
                    matrix: shifted,
                    ..Default::default()
                }],
                ..Default::default()
            });
            st.treat_records.push(TreatRecord {
                datetime: format!("{day}105418"),
                beams: vec![RecordBeam {
                    name: "LPO".into(),
                    termination_status: "NORMAL".into(),
                    ..Default::default()
                }],
                ..Default::default()
            });
            for _ in 0..2 {
                st.planar_images.push(PlanarImage {
                    kind: PlanarKind::Projection,
                    series_number: Some(212),
                    instance_number: Some(3),
                    acquired: format!("{day}105145"),
                    reviewed: format!("{day}105402"),
                    ..Default::default()
                });
            }
        }
        st
    }

    #[test]
    fn the_sessions_summary_counts_days_and_says_no_date_or_time() {
        let v = json!(sessions_summary(&course()));
        let s = v.as_array().unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(
            (s[0]["day"].as_i64(), s[1]["day"].as_i64()),
            (Some(0), Some(7))
        );
        assert_eq!(s[1]["session"], 2);
        let step = &s[0]["steps"][0];
        assert_eq!(step["beam"], "LPO");
        assert_eq!(step["setup_pairs"], 1);
        assert_eq!(step["records"], json!(["NORMAL"]));
        assert_eq!(
            step["correction"]["translation_mm"],
            json!([-0.8, 4.9, -4.4])
        );
        let text = v.to_string();
        for leak in ["2024", "0917", "0924", "1054", "10:54"] {
            assert!(!text.contains(leak), "the summary carries '{leak}': {text}");
        }
    }
}
