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
    core.session.add_values(values);
}

/// The patients of the archive, catalog brought up to date, with their
/// identities already hidden.
fn patients_of(core: &Core, ar: &Archive, p: &Progress) -> Result<Vec<PatientEntry>> {
    p.set("Reading the archive's catalog");
    let tree = ar.scan_tree(p)?;
    let patients = tree.patients();
    hide_identities(core, &patients);
    Ok(patients)
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

#[derive(Deserialize, JsonSchema, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ListArchiveArgs {
    /// A patient alias from an earlier listing (`anon_...`): only their
    /// studies. Empty: every patient.
    pub patient: Option<String>,
}

pub fn list_archive(core: &mut Core, a: ListArchiveArgs, p: &Progress) -> Result<Value> {
    let ar = archive_of(core);
    let patients = patients_of(core, &ar, p)?;
    let want = a
        .patient
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let listed: Vec<Value> = patients
        .iter()
        .filter(|pt| want.is_none_or(|w| alias_of(pt) == w))
        .map(|pt| {
            json!({
                "patient": alias_of(pt),
                "studies": pt.studies.iter().map(study_json).collect::<Vec<_>>(),
            })
        })
        .collect();
    if let Some(w) = want {
        if listed.is_empty() {
            bail!("no patient of the archive is listed as '{}'", clean_text(w));
        }
    }
    Ok(json!({
        "archive": "archive",
        "patients": listed.len(),
        "studies": listed.iter().map(|p| p["studies"].as_array().map_or(0, Vec::len)).sum::<usize>(),
        "listing": listed,
    }))
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
    out["patient"] = json!(alias_of(&pt));
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
