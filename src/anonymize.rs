//! Interactive DICOM anonymizer (*Tools ▶ Anonymize DICOM folder…*) — a
//! generalized Rust port of the one-off `tools/anonymize_dicom.py` used to
//! prepare `example_data/`.
//!
//! Two phases, both on background threads:
//!
//! * [`scan`] walks a folder, finds every identifying tag that is present,
//!   collects its distinct current values and proposes a replacement
//!   (fixed anonymization date/time, cleared physician/institution fields,
//!   a deterministic `anon_xxxxxx` patient alias derived from the original
//!   PatientID). The proposals are shown in the UI and can be edited.
//! * [`apply`] rewrites the files with the (possibly edited) replacements,
//!   optionally removing private (odd-group) elements and remapping every
//!   non-standard UID — consistently across all files, so the DICOM
//!   reference chains (RTSTRUCT ▶ series, RTDOSE ▶ RTPLAN ▶ RTSTRUCT,
//!   ReferencedSOPInstanceUID lists, frames of reference) stay intact.
//!   Pixel data is copied through untouched.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use dicom_core::value::{PrimitiveValue, Value};
use dicom_core::{DataElement, Length, Tag, VR};
use dicom_dictionary_std::tags;
use dicom_object::meta::FileMetaTableBuilder;
use dicom_object::{InMemDicomObject, OpenFileOptions};
use rayon::prelude::*;

use crate::progress::Progress;

// ---------------------------------------------------------------------------
// Rules: which tags are treated as identifying, and what to suggest.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Suggest {
    /// Deterministic patient alias derived from the original PatientID.
    Alias,
    /// The fixed anonymization date / time.
    Date,
    Time,
    /// Clear the value (write a zero-length element).
    Clear,
    /// A fixed literal.
    Fixed(&'static str),
    /// Keep the current value (row starts disabled; opt-in edit).
    Keep,
}

struct Rule {
    tag: Tag,
    name: &'static str,
    vr: VR,
    suggest: Suggest,
    /// Rows for these tags start unchecked (descriptions are not PHI per se,
    /// but often contain it — the user opts in).
    default_on: bool,
}

const ANON_DATE: &str = "20000101";
const ANON_TIME: &str = "000000";

#[allow(deprecated)] // retired tags (OtherPatientIDs, EthnicGroup) still occur in old files
fn rules() -> Vec<Rule> {
    use Suggest::*;
    let r = |tag, name, vr, suggest| Rule {
        tag,
        name,
        vr,
        suggest,
        default_on: true,
    };
    let opt = |tag, name, vr| Rule {
        tag,
        name,
        vr,
        suggest: Keep,
        default_on: false,
    };
    vec![
        r(tags::PATIENT_NAME, "PatientName", VR::PN, Alias),
        r(tags::PATIENT_ID, "PatientID", VR::LO, Alias),
        r(tags::PATIENT_BIRTH_DATE, "PatientBirthDate", VR::DA, Clear),
        r(tags::PATIENT_BIRTH_TIME, "PatientBirthTime", VR::TM, Clear),
        r(tags::PATIENT_SEX, "PatientSex", VR::CS, Clear),
        r(tags::PATIENT_AGE, "PatientAge", VR::AS, Clear),
        r(tags::PATIENT_WEIGHT, "PatientWeight", VR::DS, Clear),
        r(tags::PATIENT_SIZE, "PatientSize", VR::DS, Clear),
        r(tags::OTHER_PATIENT_I_DS, "OtherPatientIDs", VR::LO, Clear),
        r(tags::PATIENT_ADDRESS, "PatientAddress", VR::LO, Clear),
        r(
            tags::PATIENT_TELEPHONE_NUMBERS,
            "PatientTelephoneNumbers",
            VR::SH,
            Clear,
        ),
        r(tags::PATIENT_COMMENTS, "PatientComments", VR::LT, Clear),
        r(tags::ETHNIC_GROUP, "EthnicGroup", VR::SH, Clear),
        r(tags::STUDY_DATE, "StudyDate", VR::DA, Date),
        r(tags::SERIES_DATE, "SeriesDate", VR::DA, Date),
        r(tags::ACQUISITION_DATE, "AcquisitionDate", VR::DA, Date),
        r(tags::CONTENT_DATE, "ContentDate", VR::DA, Date),
        r(tags::STUDY_TIME, "StudyTime", VR::TM, Time),
        r(tags::SERIES_TIME, "SeriesTime", VR::TM, Time),
        r(tags::ACQUISITION_TIME, "AcquisitionTime", VR::TM, Time),
        r(tags::CONTENT_TIME, "ContentTime", VR::TM, Time),
        r(tags::ACCESSION_NUMBER, "AccessionNumber", VR::SH, Clear),
        r(
            tags::REFERRING_PHYSICIAN_NAME,
            "ReferringPhysicianName",
            VR::PN,
            Clear,
        ),
        r(
            tags::PERFORMING_PHYSICIAN_NAME,
            "PerformingPhysicianName",
            VR::PN,
            Clear,
        ),
        r(
            tags::PHYSICIANS_OF_RECORD,
            "PhysiciansOfRecord",
            VR::PN,
            Clear,
        ),
        r(tags::OPERATORS_NAME, "OperatorsName", VR::PN, Clear),
        r(tags::INSTITUTION_NAME, "InstitutionName", VR::LO, Clear),
        r(
            tags::INSTITUTION_ADDRESS,
            "InstitutionAddress",
            VR::ST,
            Clear,
        ),
        r(
            tags::INSTITUTIONAL_DEPARTMENT_NAME,
            "InstitutionalDepartmentName",
            VR::LO,
            Clear,
        ),
        r(tags::STATION_NAME, "StationName", VR::SH, Clear),
        r(
            tags::DEVICE_SERIAL_NUMBER,
            "DeviceSerialNumber",
            VR::LO,
            Clear,
        ),
        r(tags::STUDY_ID, "StudyID", VR::SH, Fixed("1")),
        opt(tags::STUDY_DESCRIPTION, "StudyDescription", VR::LO),
        opt(tags::SERIES_DESCRIPTION, "SeriesDescription", VR::LO),
        opt(tags::MANUFACTURER, "Manufacturer", VR::LO),
        opt(
            tags::MANUFACTURER_MODEL_NAME,
            "ManufacturerModelName",
            VR::LO,
        ),
    ]
}

// ---------------------------------------------------------------------------
// Scan
// ---------------------------------------------------------------------------

/// One identifying tag found in the folder, with its current values and the
/// (editable) proposed replacement.
#[derive(Clone)]
pub struct TagFinding {
    pub tag: Tag,
    pub name: String,
    pub vr: VR,
    /// Distinct current values (at most [`MAX_SHOWN_VALUES`] kept).
    pub values: Vec<String>,
    /// Total number of distinct values.
    pub n_values: usize,
    /// In how many of the scanned files the tag is present.
    pub n_files: usize,
    /// Algorithm proposal (kept for display / reset).
    pub suggested: String,
    /// Editable replacement, initialized to `suggested`. Empty = clear.
    pub replacement: String,
    /// Whether the row is applied.
    pub enabled: bool,
}

pub const MAX_SHOWN_VALUES: usize = 6;

pub struct ScanResult {
    pub root: PathBuf,
    pub files: Vec<PathBuf>,
    /// Modality → file count.
    pub modalities: BTreeMap<String, usize>,
    pub findings: Vec<TagFinding>,
    /// Distinct non-standard UIDs that would be regenerated.
    pub uid_count: usize,
    /// Total private (odd-group) elements found, including inside sequences.
    pub private_count: usize,
    pub warnings: Vec<String>,
}

/// Deterministic alias for the dataset's patient(s): `anon_` + 6 hex digits
/// derived from the sorted original PatientIDs (falls back to names).
fn patient_alias(ids: &[String]) -> String {
    let mut h = DefaultHasher::new();
    for id in ids {
        id.hash(&mut h);
    }
    format!("anon_{:06x}", h.finish() & 0xFF_FFFF)
}

fn is_standard_uid(uid: &str) -> bool {
    // DICOM-registered UIDs (SOP classes, transfer syntaxes, well-known
    // frames of reference…) must never be remapped.
    uid.starts_with("1.2.840.10008")
}

/// Recursively collect non-standard UIDs and count private elements.
fn walk_stats(obj: &InMemDicomObject, uids: &mut HashSet<String>, private: &mut usize) {
    for el in obj.iter() {
        if el.header().tag.group() % 2 == 1 {
            *private += 1;
        }
        match el.value() {
            Value::Sequence(seq) => {
                for item in seq.items() {
                    walk_stats(item, uids, private);
                }
            }
            Value::Primitive(p) if el.vr() == VR::UI => {
                for s in p.to_multi_str().iter() {
                    let s = s.trim_end_matches('\0').trim();
                    if !s.is_empty() && !is_standard_uid(s) {
                        uids.insert(s.to_string());
                    }
                }
            }
            _ => {}
        }
    }
}

pub fn scan(dir: &Path, progress: &Progress) -> Result<ScanResult> {
    progress.set("Scanning folder…");
    let files: Vec<PathBuf> = walkdir::WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    anyhow::ensure!(!files.is_empty(), "No files found in {}", dir.display());

    let rule_list = rules();

    /// What one file contributes to the scan totals.
    struct FileScan {
        modality: String,
        /// Present identifying tags with their current value.
        found: Vec<(Tag, String)>,
        uids: HashSet<String>,
        private: usize,
    }

    // Header parsing dominates the scan and the files are independent, so it
    // runs in parallel; only the (cheap) merge into the totals is serial.
    let done = AtomicUsize::new(0);
    let scanned: Vec<Option<FileScan>> = files
        .par_iter()
        .map(|path| {
            let n = done.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(64) {
                progress.set(format!("Reading headers… {}/{}", n + 1, files.len()));
            }
            // Header-only read: identifying tags and reference sequences all
            // sit before Pixel Data.
            let obj = OpenFileOptions::new()
                .read_until(tags::PIXEL_DATA)
                .open_file(path)
                .ok()?;
            let mut uids = HashSet::new();
            let mut private = 0usize;
            walk_stats(&obj, &mut uids, &mut private);
            Some(FileScan {
                modality: crate::loader::str_of(&obj, tags::MODALITY).unwrap_or_else(|| "?".into()),
                found: rule_list
                    .iter()
                    .filter_map(|rule| {
                        let el = obj.element(rule.tag).ok()?;
                        let v = el
                            .to_str()
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default();
                        Some((rule.tag, v))
                    })
                    .collect(),
                uids,
                private,
            })
        })
        .collect();

    let mut per_tag: HashMap<Tag, (HashSet<String>, usize)> = HashMap::new();
    let mut modalities: BTreeMap<String, usize> = BTreeMap::new();
    let mut uids = HashSet::new();
    let mut private = 0usize;
    let mut dicom_files = Vec::new();
    let mut unreadable = 0usize;

    for (path, fs) in files.iter().zip(scanned) {
        let Some(fs) = fs else {
            unreadable += 1;
            continue;
        };
        dicom_files.push(path.clone());
        *modalities.entry(fs.modality).or_insert(0) += 1;
        for (tag, v) in fs.found {
            let e = per_tag.entry(tag).or_default();
            e.0.insert(v);
            e.1 += 1;
        }
        uids.extend(fs.uids);
        private += fs.private;
    }
    anyhow::ensure!(
        !dicom_files.is_empty(),
        "No DICOM files found in {}",
        dir.display()
    );

    // Patient alias from the original IDs (names as fallback).
    let mut ids: Vec<String> = per_tag
        .get(&tags::PATIENT_ID)
        .or_else(|| per_tag.get(&tags::PATIENT_NAME))
        .map(|(vals, _)| vals.iter().cloned().collect())
        .unwrap_or_default();
    ids.sort();
    let alias = patient_alias(&ids);

    let mut warnings = Vec::new();
    if unreadable > 0 {
        warnings.push(format!(
            "{unreadable} non-DICOM file(s) will be left untouched"
        ));
    }
    if ids.len() > 1 {
        warnings.push(format!(
            "{} distinct patients in this folder — they would all receive the same alias; \
             edit the PatientName/PatientID replacements if that is not intended",
            ids.len()
        ));
    }

    let mut findings = Vec::new();
    for rule in &rule_list {
        let Some((vals, n_files)) = per_tag.get(&rule.tag) else {
            continue;
        };
        let mut values: Vec<String> = vals.iter().cloned().collect();
        values.sort();
        let n_values = values.len();
        values.truncate(MAX_SHOWN_VALUES);
        let suggested = match rule.suggest {
            Suggest::Alias => alias.clone(),
            Suggest::Date => ANON_DATE.into(),
            Suggest::Time => ANON_TIME.into(),
            Suggest::Clear => String::new(),
            Suggest::Fixed(s) => s.into(),
            Suggest::Keep => values.first().cloned().unwrap_or_default(),
        };
        findings.push(TagFinding {
            tag: rule.tag,
            name: rule.name.into(),
            vr: rule.vr,
            values,
            n_values,
            n_files: *n_files,
            replacement: suggested.clone(),
            suggested,
            enabled: rule.default_on,
        });
    }

    progress.set("done");
    Ok(ScanResult {
        root: dir.to_path_buf(),
        files: dicom_files,
        modalities,
        findings,
        uid_count: uids.len(),
        private_count: private,
        warnings,
    })
}

// ---------------------------------------------------------------------------
// Apply
// ---------------------------------------------------------------------------

pub struct ApplyParams {
    /// Enabled replacements: tag, VR, new value (empty = clear).
    pub replacements: Vec<(Tag, VR, String)>,
    pub remove_private: bool,
    pub remap_uids: bool,
    /// Write the Patient Identity Removed / De-identification Method tags.
    pub mark_deidentified: bool,
    /// Output folder (files mirror their relative paths); `None` = in place.
    pub out_dir: Option<PathBuf>,
}

/// Fresh UID under the 2.25 (UUID-derived) root — deterministic within one
/// run (salted hash of the original), collision-checked against the map.
fn new_uid(old: &str, salt: u64, taken: &HashSet<String>) -> String {
    let mut n = 0u64;
    loop {
        let mut h = DefaultHasher::new();
        salt.hash(&mut h);
        old.hash(&mut h);
        n.hash(&mut h);
        let a = h.finish();
        let mut h2 = DefaultHasher::new();
        (salt ^ 0x9E3779B97F4A7C15).hash(&mut h2);
        old.hash(&mut h2);
        n.hash(&mut h2);
        let uid = format!("2.25.{}{:019}", a, h2.finish() % 10_000_000_000_000_000_000);
        if uid.len() <= 64 && !taken.contains(&uid) {
            return uid;
        }
        n += 1;
    }
}

/// Recursively rewrite an object: drop private elements, remap UIDs,
/// descend into sequences.
fn transform(obj: &mut InMemDicomObject, p: &ApplyParams, uid_map: &HashMap<String, String>) {
    let tags_now: Vec<Tag> = obj.tags().collect();
    for tag in tags_now {
        if p.remove_private && tag.group() % 2 == 1 {
            obj.remove_element(tag);
            continue;
        }
        let Ok(el) = obj.element(tag) else { continue };
        match el.value() {
            Value::Sequence(_) => {
                let el = obj.take_element(tag).expect("tag just seen");
                let vr = el.vr();
                if let Value::Sequence(seq) = el.into_value() {
                    let mut items: Vec<InMemDicomObject> = seq.into_items().into_iter().collect();
                    for item in &mut items {
                        transform(item, p, uid_map);
                    }
                    obj.put(DataElement::new(
                        tag,
                        vr,
                        Value::new_sequence(items, Length::UNDEFINED),
                    ));
                }
            }
            Value::Primitive(prim) if el.vr() == VR::UI && p.remap_uids => {
                let vals: Vec<String> = prim
                    .to_multi_str()
                    .iter()
                    .map(|s| s.trim_end_matches('\0').trim().to_string())
                    .collect();
                if vals.iter().any(|v| uid_map.contains_key(v)) {
                    let new: Vec<String> = vals
                        .iter()
                        .map(|v| uid_map.get(v).cloned().unwrap_or_else(|| v.clone()))
                        .collect();
                    obj.put(DataElement::new(
                        tag,
                        VR::UI,
                        PrimitiveValue::from(new.join("\\")),
                    ));
                }
            }
            _ => {}
        }
    }
}

fn put_replacement(obj: &mut InMemDicomObject, tag: Tag, vr: VR, value: &str) {
    if value.is_empty() {
        obj.put(DataElement::new(
            tag,
            vr,
            Value::Primitive(PrimitiveValue::Empty),
        ));
    } else {
        obj.put(DataElement::new(tag, vr, PrimitiveValue::from(value)));
    }
}

/// Rewrite every scanned file. Returns the number of files written.
pub fn apply(
    files: &[PathBuf],
    root: &Path,
    p: &ApplyParams,
    progress: &Progress,
) -> Result<usize> {
    // Pass 1: collect every non-standard UID so the remapping is consistent
    // across files (referential integrity of the whole folder).
    let mut uid_map: HashMap<String, String> = HashMap::new();
    if p.remap_uids {
        progress.set("Collecting UIDs…");
        let per_file: Vec<Result<HashSet<String>>> = files
            .par_iter()
            .map(|path| {
                let obj = OpenFileOptions::new()
                    .read_until(tags::PIXEL_DATA)
                    .open_file(path)
                    .with_context(|| format!("re-open {}", path.display()))?;
                let mut u = HashSet::new();
                let mut private = 0usize;
                walk_stats(&obj, &mut u, &mut private);
                Ok(u)
            })
            .collect();
        let mut uids = HashSet::new();
        for r in per_file {
            uids.extend(r?);
        }
        let salt = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x5DEECE66D);
        let mut sorted: Vec<String> = uids.into_iter().collect();
        sorted.sort();
        let mut taken: HashSet<String> = HashSet::new();
        for old in sorted {
            let fresh = new_uid(&old, salt, &taken);
            taken.insert(fresh.clone());
            uid_map.insert(old, fresh);
        }
    }

    if let Some(out) = &p.out_dir {
        std::fs::create_dir_all(out)
            .with_context(|| format!("create output folder {}", out.display()))?;
    }

    let done = AtomicUsize::new(0);
    // One file in, one file out — parallel over files, like the scan pass.
    let results: Vec<Result<()>> = files
        .par_iter()
        .map(|path| -> Result<()> {
            let i = done.fetch_add(1, Ordering::Relaxed);
            progress.set(format!("Anonymizing… {}/{}", i + 1, files.len()));
            let file_obj = dicom_object::open_file(path)
                .with_context(|| format!("open {}", path.display()))?;
            let meta = file_obj.meta().clone();
            let mut obj = file_obj.into_inner();

            transform(&mut obj, p, &uid_map);
            for (tag, vr, value) in &p.replacements {
                put_replacement(&mut obj, *tag, *vr, value);
            }
            if p.mark_deidentified {
                put_replacement(&mut obj, tags::PATIENT_IDENTITY_REMOVED, VR::CS, "YES");
                put_replacement(
                    &mut obj,
                    tags::DEIDENTIFICATION_METHOD,
                    VR::LO,
                    "rust-dicom-station anonymizer",
                );
            }

            let sop_class = crate::loader::str_of(&obj, tags::SOP_CLASS_UID).unwrap_or_default();
            let sop_inst = crate::loader::str_of(&obj, tags::SOP_INSTANCE_UID).unwrap_or_default();
            let ts = meta.transfer_syntax().trim_end_matches('\0').to_string();
            let out_obj = obj
                .with_meta(
                    FileMetaTableBuilder::new()
                        .transfer_syntax(&ts)
                        .media_storage_sop_class_uid(&sop_class)
                        .media_storage_sop_instance_uid(&sop_inst),
                )
                .with_context(|| format!("rebuild file meta of {}", path.display()))?;

            let out_path = match &p.out_dir {
                Some(out) => {
                    let rel = path.strip_prefix(root).unwrap_or(path.as_path());
                    let dst = out.join(rel);
                    if let Some(parent) = dst.parent() {
                        std::fs::create_dir_all(parent)
                            .with_context(|| format!("create {}", parent.display()))?;
                    }
                    dst
                }
                None => path.clone(),
            };
            // Write via a temp file so an error never corrupts the original.
            let tmp = out_path.with_extension("tmp_anon");
            out_obj
                .write_to_file(&tmp)
                .with_context(|| format!("write {}", tmp.display()))?;
            std::fs::rename(&tmp, &out_path)
                .with_context(|| format!("replace {}", out_path.display()))?;
            Ok(())
        })
        .collect();
    let mut written = 0usize;
    for r in results {
        r?;
        written += 1;
    }
    progress.set("done");
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_is_deterministic_and_short() {
        let a = patient_alias(&["P102".into()]);
        let b = patient_alias(&["P102".into()]);
        assert_eq!(a, b);
        assert!(a.starts_with("anon_") && a.len() == 11, "{a}");
    }

    #[test]
    fn standard_uids_are_never_remapped() {
        assert!(is_standard_uid("1.2.840.10008.5.1.4.1.1.2"));
        assert!(!is_standard_uid("1.3.6.1.4.1.14519.5.2.1"));
        assert!(!is_standard_uid("2.25.1234"));
    }

    #[test]
    fn new_uids_are_valid_and_unique() {
        let mut taken = HashSet::new();
        for i in 0..100 {
            let u = new_uid(&format!("1.2.3.{i}"), 42, &taken);
            assert!(u.len() <= 64 && u.starts_with("2.25."));
            assert!(taken.insert(u));
        }
    }
}
