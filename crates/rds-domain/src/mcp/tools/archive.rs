//! The station's archive (its PACS) from the assistant's side: what it
//! holds, what each series of a study would become when loaded, and a study
//! opened as a dataset through the same gate as `open_dataset`.
//!
//! The archive is the one place on a station that holds patients under
//! their own names, so these tools name nobody: a patient is an alias
//! made from the key the catalog files them under, a study is its Study
//! Instance UID and the year it was made, and every name and ID the
//! catalog knows goes to the redactor before the first answer leaves.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use dicom_dictionary_std::tags;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::config::PhiPolicy;
use super::super::phi::clean_text;
use super::super::Core;
use super::session;
use crate::anonymize;
use crate::archive::{self, Archive, PatientEntry, Pick, StudyEntry};
use crate::dicomfile;
use crate::loader::{self, i32_of, str_of, FileClass};
use crate::progress::Progress;

/// The archive the configuration names: the station's own unless
/// `archive` in mcp.toml says otherwise. Its folders go to the redactor,
/// so a path under them is reported as `archive/...`.
fn archive_of(core: &Core) -> Archive {
    let ar = Archive::from_setting(&core.session.config.archive);
    for root in ar.roots() {
        core.session.add_root(root, "archive");
    }
    ar
}

/// The alias a patient of the archive is listed under: a hash of the key
/// the catalog files them by, the same in every call.
fn alias_of(p: &PatientEntry) -> String {
    anonymize::patient_alias(&[p.key()])
}

/// What a patient is listed as: their code, under the `pseudonymized`
/// policy, when the key the catalog files them by is one (a code is no
/// PHI, and the curation tools speak of `P01-039`, not of a hash); the
/// alias otherwise.
fn listed_as(core: &Core, p: &PatientEntry) -> String {
    let key = p.key();
    if is_code(core, &key) {
        key
    } else {
        alias_of(p)
    }
}

/// Whether `value` is a patient code the `pseudonymized` policy lets
/// through as it is.
fn is_code(core: &Core, value: &str) -> bool {
    core.session.config.phi_policy == PhiPolicy::Pseudonymized
        && core.session.config.pacs.codes().is_code(value)
}

/// Everything about the patients the catalog holds that could name one,
/// handed to the redactor: the name as filed and as shown, the ID, the key,
/// and the labels typed into the PACS window (a person may type a name
/// there).
fn hide_identities(core: &Core, patients: &[PatientEntry]) {
    let mut values: Vec<String> = Vec::new();
    for p in patients {
        let name = p.name.trim();
        if !name.is_empty() {
            values.push(name.to_string());
            values.push(name.replace('^', " ").trim().to_string());
            // The parts of a `Family^Given` name on their own.
            values.extend(
                name.split('^')
                    .map(str::trim)
                    .filter(|s| s.len() >= 3)
                    .map(String::from),
            );
        }
        for v in [p.id.trim(), p.key.trim(), p.label.trim()] {
            if !v.is_empty() && v != "unknown" {
                values.push(v.to_string());
            }
        }
        for s in &p.studies {
            if !s.label.trim().is_empty() {
                values.push(s.label.trim().to_string());
            }
        }
    }
    values.sort();
    values.dedup();
    // A code stays readable under the `pseudonymized` policy; the parts of
    // a name are hidden whatever the policy.
    values.retain(|v| !is_code(core, v));
    core.session.add_values(values);
}

/// The archive's tree, catalog brought up to date, with every patient's
/// identity already hidden.
fn tree_of(core: &Core, ar: &Archive, p: &Progress) -> Result<archive::Tree> {
    p.set("Reading the archive's catalog");
    let tree = ar.scan_tree(p)?;
    hide_identities(core, &tree.patients());
    Ok(tree)
}

/// The patients of the archive, as [`tree_of`] reads them.
fn patients_of(core: &Core, ar: &Archive, p: &Progress) -> Result<Vec<PatientEntry>> {
    Ok(tree_of(core, ar, p)?.patients())
}

/// `20240517` as `2024`; anything else as nothing. The year is all a
/// listing says of a date.
fn year_of(date: &str) -> Option<String> {
    let d = date.trim();
    (d.len() >= 4 && d[..4].bytes().all(|b| b.is_ascii_digit())).then(|| d[..4].to_string())
}

fn study_json(s: &StudyEntry) -> Value {
    let mut v = json!({
        "study": s.study_uid,
        "modalities": s.modalities,
        "description": clean_text(&s.description),
        "files": s.files,
        "series": s.series.iter().map(|se| json!({
            "modality": se.modality,
            "number": se.number.trim(),
            "description": clean_text(&se.description),
            "files": se.files,
        })).collect::<Vec<_>>(),
    });
    if let Some(y) = year_of(&s.date) {
        v["year"] = json!(y);
    }
    v
}

/// Patients a listing returns when the call names no `limit`: enough to
/// page through, few enough that one page stays well under what a tool
/// result may carry (the whole of a large archive was close to a million
/// characters).
pub const LIST_LIMIT: usize = 25;

#[derive(Deserialize, JsonSchema, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ListArchiveArgs {
    /// A patient as an earlier listing named them (an alias, or their
    /// code under the pseudonymized policy): only their studies. Empty:
    /// every patient.
    pub patient: Option<String>,
    /// Only the patients of this cohort (its name, as the summary lists
    /// it).
    pub cohort: Option<String>,
    /// Counts only: per cohort the patients, studies, series and files, and
    /// the modalities. Small whatever the archive's size.
    pub summary: bool,
    /// Skip this many patients (paging; the answer says the next offset).
    pub offset: usize,
    /// At most this many patients; default 25.
    pub limit: Option<usize>,
}

pub fn list_archive(core: &mut Core, a: ListArchiveArgs, p: &Progress) -> Result<Value> {
    let ar = archive_of(core);
    let tree = tree_of(core, &ar, p)?;
    let cohort = a.cohort.as_deref().map(str::trim).filter(|s| !s.is_empty());
    let cohorts: Vec<&archive::CohortEntry> = tree
        .cohorts
        .iter()
        .filter(|c| cohort.is_none_or(|w| c.name.trim().eq_ignore_ascii_case(w)))
        .collect();
    if let Some(w) = cohort {
        if cohorts.is_empty() {
            bail!("the archive has no cohort '{}'", clean_text(w));
        }
    }
    if a.summary {
        let per: Vec<Value> = cohorts
            .iter()
            .map(|c| {
                let studies: usize = c.patients.iter().map(|p| p.studies.len()).sum();
                let series: usize = c
                    .patients
                    .iter()
                    .flat_map(|p| &p.studies)
                    .map(|s| s.series.len())
                    .sum();
                let mut modalities: BTreeMap<String, usize> = BTreeMap::new();
                for s in c
                    .patients
                    .iter()
                    .flat_map(|p| &p.studies)
                    .flat_map(|s| &s.series)
                {
                    *modalities.entry(s.modality.clone()).or_default() += 1;
                }
                json!({
                    "cohort": clean_text(&c.name),
                    "patients": c.patients.len(),
                    "studies": studies,
                    "series": series,
                    "files": c.files(),
                    "series_by_modality": modalities,
                })
            })
            .collect();
        return Ok(json!({ "archive": "archive", "cohorts": per }));
    }
    // Every patient once, in name order, as `Tree::patients` does, within
    // the cohorts asked for.
    let mut seen = std::collections::HashSet::new();
    let mut patients: Vec<&PatientEntry> = cohorts
        .iter()
        .flat_map(|c| c.patients.iter())
        .filter(|p| seen.insert(p.key.clone()))
        .collect();
    patients.sort_by_key(|p| p.title().to_lowercase());
    let want = a
        .patient
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let matching: Vec<&PatientEntry> = patients
        .into_iter()
        .filter(|pt| want.is_none_or(|w| alias_of(pt) == w || listed_as(core, pt) == w))
        .collect();
    if let Some(w) = want {
        if matching.is_empty() {
            bail!("no patient of the archive is listed as '{}'", clean_text(w));
        }
    }
    let total = matching.len();
    let limit = a.limit.unwrap_or(LIST_LIMIT).max(1);
    let page: Vec<Value> = matching
        .iter()
        .skip(a.offset)
        .take(limit)
        .map(|pt| {
            json!({
                "patient": listed_as(core, pt),
                "studies": pt.studies.iter().map(study_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    let mut out = json!({
        "archive": "archive",
        "patients": total,
        "offset": a.offset,
        "listed": page.len(),
        "studies": page.iter().map(|p| p["studies"].as_array().map_or(0, Vec::len)).sum::<usize>(),
        "listing": page,
    });
    if a.offset + limit < total {
        out["next_offset"] = json!(a.offset + limit);
    }
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArchiveStudyArgs {
    /// The Study Instance UID, as list_archive gives it.
    pub study: String,
}

/// The study `uid` names, with its patient, from a listing whose
/// identities are hidden already.
fn find_study(patients: &[PatientEntry], uid: &str) -> Result<(PatientEntry, StudyEntry)> {
    let uid = uid.trim();
    if !archive::is_uid(uid) {
        bail!("study must be a Study Instance UID, as list_archive gives it");
    }
    patients
        .iter()
        .find_map(|pt| {
            pt.studies
                .iter()
                .find(|s| s.study_uid == uid)
                .map(|s| (pt.clone(), s.clone()))
        })
        .ok_or_else(|| anyhow::anyhow!("the archive holds no study {uid}"))
}

/// What the header of one file says, for the series listing.
struct HeadFacts {
    modality: String,
    sop_class: String,
    has_geometry: bool,
    image_type: Option<String>,
    rows: Option<i32>,
    cols: Option<i32>,
    frames: Option<i32>,
    samples: Option<i32>,
    photometric: Option<String>,
    /// A scanner's page whose text reads back: a report, not an image.
    page_text: bool,
}

fn head_facts(path: &Path) -> Option<HeadFacts> {
    let obj = dicomfile::open_scan(path).ok()?;
    Some(HeadFacts {
        page_text: crate::report::siemens::oog_text(&obj).is_some(),
        modality: str_of(&obj, tags::MODALITY).unwrap_or_default(),
        sop_class: str_of(&obj, tags::SOP_CLASS_UID).unwrap_or_default(),
        has_geometry: loader::f64s_of(&obj, tags::IMAGE_POSITION_PATIENT)
            .is_some_and(|v| v.len() == 3),
        image_type: str_of(&obj, tags::IMAGE_TYPE).map(|t| t.replace('\\', " / ")),
        rows: i32_of(&obj, tags::ROWS),
        cols: i32_of(&obj, tags::COLUMNS),
        frames: i32_of(&obj, tags::NUMBER_OF_FRAMES),
        samples: i32_of(&obj, tags::SAMPLES_PER_PIXEL),
        photometric: str_of(&obj, tags::PHOTOMETRIC_INTERPRETATION),
    })
}

/// Up to `n` of `files`, spread along the series.
fn spread(files: &[PathBuf], n: usize) -> Vec<&PathBuf> {
    if files.len() <= n {
        return files.iter().collect();
    }
    (0..n)
        .map(|i| &files[i * (files.len() - 1) / (n - 1)])
        .collect()
}

pub fn describe_archive_study(core: &mut Core, a: ArchiveStudyArgs, p: &Progress) -> Result<Value> {
    let ar = archive_of(core);
    let patients = patients_of(core, &ar, p)?;
    let (pt, st) = find_study(&patients, &a.study)?;
    p.set("Reading the headers of the study");
    // The files of every series, from the catalog.
    let mut by_series: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for inst in ar.files_of(&[Pick::Study(st.study_uid.clone())]) {
        by_series
            .entry(inst.series_uid)
            .or_default()
            .push(inst.path);
    }
    let series: Vec<Value> = st
        .series
        .iter()
        .map(|se| {
            let files = by_series.get(&se.series_uid).cloned().unwrap_or_default();
            let first = files.first().and_then(|f| head_facts(f));
            let mut v = json!({
                "modality": se.modality,
                "number": se.number.trim(),
                "description": clean_text(&se.description),
                "files": se.files,
            });
            let Some(h) = first else {
                v["class"] = json!("unreadable");
                return v;
            };
            // Positioned anywhere along the series, from a few of its files.
            let positioned = h.has_geometry
                || spread(&files, 8)
                    .into_iter()
                    .skip(1)
                    .any(|f| head_facts(f).is_some_and(|h| h.has_geometry));
            let mut class = loader::classify(&h.modality, &h.sop_class, h.has_geometry, positioned);
            if class == FileClass::Additional && h.page_text {
                class = FileClass::Report;
                v["report"] = json!("protocol page");
            }
            let pdf = h.sop_class == crate::sop::SOP_ENCAPSULATED_PDF;
            if pdf {
                v["report"] = json!("document");
            }
            v["class"] = json!(class.label());
            v["sop_class"] = json!(crate::sop::name_or_uid(&h.sop_class));
            v["positioned"] = json!(positioned);
            if let Some(t) = h.image_type {
                v["image_type"] = json!(clean_text(&t));
            }
            if let (Some(c), Some(r)) = (h.cols, h.rows) {
                v["pixels"] = json!([c, r]);
            }
            if let Some(f) = h.frames.filter(|f| *f > 1) {
                v["frames"] = json!(f);
            }
            if let Some(s) = h.samples.filter(|s| *s > 1) {
                v["samples_per_pixel"] = json!(s);
            }
            if let Some(ph) = h.photometric {
                v["photometric"] = json!(clean_text(&ph));
            }
            if matches!(
                class,
                FileClass::Projection
                    | FileClass::Additional
                    | FileClass::Report
                    | FileClass::Other
            ) {
                v["shown_in"] = json!(match class {
                    FileClass::Projection => "a planar pane",
                    FileClass::Report if pdf => "the Additional pane, as its pages",
                    FileClass::Report => "the Additional pane, as a table",
                    FileClass::Other => "nowhere: it is left out when the study is loaded",
                    _ => "the Additional pane",
                });
            }
            v
        })
        .collect();
    let mut out = study_json(&st);
    out["patient"] = json!(listed_as(core, &pt));
    out["series"] = json!(series);
    Ok(out)
}

pub fn open_archive_study(core: &mut Core, a: ArchiveStudyArgs, p: &Progress) -> Result<Value> {
    let ar = archive_of(core);
    let patients = patients_of(core, &ar, p)?;
    let (_, st) = find_study(&patients, &a.study)?;
    let files: Vec<PathBuf> = ar
        .files_of(&[Pick::Study(st.study_uid.clone())])
        .into_iter()
        .map(|i| i.path)
        .collect();
    if files.is_empty() {
        bail!("the archive holds no file of study {}", st.study_uid);
    }
    let origin = ar
        .find_study(&st.study_uid)
        .unwrap_or_else(|| st.dir.clone());
    let study = loader::load_files(&files, "archive study", p)?;
    session::admit(core, study, None, origin, "archive".to_string(), p)
}
