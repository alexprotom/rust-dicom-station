//! `phi_report`, `plan_pseudonymize`, `patient_demographics`,
//! `set_patient_demographics`: identified imports made into the archive's
//! pseudonymized form ([`crate::curation::anon`]). Nothing here returns a
//! name, an ID or a birth date: places by tag, counts, and patient codes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::*;
use crate::curation::anon;
use crate::curation::raw::RawFile;
use crate::curation::remap::{simulate, Changes, UidMaps};
use crate::curation::reorg::{self, Rules};

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhiArgs {
    /// A folder under the [pacs] roots or the quarantine; its first-level
    /// folders are the patients.
    pub path: PathBuf,
    /// One patient code for the whole folder.
    #[serde(default)]
    pub patient: Option<String>,
    /// The cohort whose patient_code_map applies (pacs-rules.toml).
    #[serde(default)]
    pub cohort: Option<String>,
}

/// A patient of a folder: how they are shown, their code (none when no
/// rule names the folder), their files.
type Patient = (String, Option<String>, Vec<PathBuf>);

/// What the dry run of one file gave: its rewrite (none when it stays as
/// it is) and the places a token is left in.
type Checked = (PathBuf, Result<(Option<journal::Op>, Vec<String>)>);

/// The files of `dir` by patient: (how the patient is shown, its code or
/// none when no rule names the folder, files).
fn by_patient(
    core: &Core,
    dir: &Path,
    patient: Option<&str>,
    cohort: &str,
) -> Result<Vec<Patient>> {
    let files = curation::files_under(dir);
    if let Some(code) = patient.map(str::trim).filter(|c| !c.is_empty()) {
        return Ok(vec![(code.to_string(), Some(code.to_string()), files)]);
    }
    let rules = Rules::load(&records(core)?)?;
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for f in files {
        let Ok(rel) = f.strip_prefix(dir) else {
            continue;
        };
        if rel.components().count() < 2 {
            continue;
        }
        let folder = rel
            .components()
            .next()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .unwrap_or_default();
        groups.entry(folder).or_default().push(f);
    }
    Ok(groups
        .into_iter()
        .enumerate()
        .map(|(i, (folder, files))| {
            let code = rules.patient_code(cohort, &folder);
            // A code only when a rule made one: the folder's own name may
            // be the patient's.
            let named =
                code != reorg::clean(&folder) || core.session.config.pacs.codes().is_code(&code);
            if named {
                (code.clone(), Some(code), files)
            } else {
                (format!("folder #{}", i + 1), None, files)
            }
        })
        .collect())
}

fn read_all(files: &[PathBuf]) -> Vec<RawFile> {
    files
        .par_iter()
        .filter_map(|f| RawFile::read(f).ok().filter(RawFile::is_dicom))
        .collect()
}

pub fn phi_report(core: &mut Core, a: PhiArgs, p: &Progress) -> Result<Value> {
    let dir = resolve(core, &a.path)?;
    let groups = by_patient(
        core,
        &dir,
        a.patient.as_deref(),
        a.cohort.as_deref().unwrap_or(""),
    )?;
    let mut out = Vec::new();
    for (shown, code, files) in groups {
        p.set(format!("Scanning {shown}"));
        // A patient at a time: the files' identities, then the scan.
        let raws = read_all(&files);
        let r = anon::report(&raws, code.as_deref().unwrap_or(""));
        out.push(json!({
            "patient": shown,
            "code_rule": code.is_some(),
            "report": r,
        }));
        if p.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
    }
    Ok(json!({
        "folder": dir.display().to_string(),
        "patients": out,
        "note": "places are tags (sequence>element) whose text holds a token of the patient's \
                 names or IDs, with the number of files; no value is shown",
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PseudonymizeArgs {
    pub path: PathBuf,
    #[serde(default)]
    pub patient: Option<String>,
    #[serde(default)]
    pub cohort: Option<String>,
    /// Keep private elements (default true: a setup system's data needs
    /// them).
    #[serde(default)]
    pub keep_private: Option<bool>,
    /// Move each file, once rewritten, to this folder (its path below
    /// `path` kept): pseudonymize duplicates as they go to quarantine.
    /// Default: in place.
    #[serde(default)]
    pub move_to: Option<PathBuf>,
    pub reason: String,
}

pub fn plan_pseudonymize(core: &mut Core, a: PseudonymizeArgs, p: &Progress) -> Result<Value> {
    let dir = resolve(core, &a.path)?;
    let move_to = match &a.move_to {
        Some(m) => {
            let m = core.session.config.pacs.resolve(m).map(|(r, _)| plain(r))?;
            Some(m)
        }
        None => None,
    };
    let keep_private = a.keep_private.unwrap_or(true);
    let groups = by_patient(
        core,
        &dir,
        a.patient.as_deref(),
        a.cohort.as_deref().unwrap_or(""),
    )?;
    let mut ops = Vec::new();
    let mut refused: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut unrewritable = 0usize;
    let mut summary = Vec::new();
    for (shown, code, files) in groups {
        let Some(code) = code else {
            bail!(
                "{shown} matches no patient rule of pacs-rules.toml; give patient, or a \
                 patient_folders pattern for it"
            );
        };
        p.set(format!("Pseudonymizing {shown}: reading the identities"));
        // Pass 1: every name and ID the patient's files carry.
        let ids: BTreeSet<String> = files
            .par_iter()
            .filter_map(|f| RawFile::read(f).ok().filter(RawFile::is_dicom))
            .flat_map_iter(|r| anon::identities(&r))
            .collect();
        let toks = anon::tokens(ids.iter().map(String::as_str), &code);
        // Pass 2: each file rewritten in memory, then scanned.
        p.set(format!(
            "Pseudonymizing {shown}: checking {} files",
            files.len()
        ));
        let results: Vec<Checked> = files
            .par_iter()
            .map(|f| {
                let r = (|| {
                    let raw = RawFile::read(f)?;
                    if !raw.is_dicom() {
                        return Ok((None, Vec::new()));
                    }
                    let changes = anon::changes(&raw, &code, keep_private);
                    let s = simulate(f, &changes, &UidMaps::default())?;
                    let g = RawFile::parse(s.bytes)?;
                    let hits = anon::scan(&g, &toks);
                    let op = (s.changed > 0).then(|| journal::Op::Rewrite {
                        path: f.clone(),
                        changes,
                        source: Some(s.source),
                        expect: Some(s.expect),
                    });
                    Ok((op, hits))
                })();
                (f.clone(), r)
            })
            .collect();
        let mut planned = 0;
        for (f, r) in results {
            match r {
                Ok((op, hits)) => {
                    for h in hits {
                        *refused
                            .entry(shown.clone())
                            .or_default()
                            .entry(h)
                            .or_default() += 1;
                    }
                    if let Some(op) = op {
                        ops.push(op);
                        planned += 1;
                    }
                    if let Some(m) = &move_to {
                        if let Ok(rel) = f.strip_prefix(&dir) {
                            ops.push(journal::Op::Move {
                                from: f.clone(),
                                to: m.join(rel),
                            });
                        }
                    }
                }
                Err(_) => unrewritable += 1,
            }
        }
        summary.push(json!({
            "patient": shown,
            "files": files.len(),
            "rewrites": planned,
            "tokens_checked": toks.len(),
        }));
    }
    if !refused.is_empty() {
        bail!(
            "the leftover scan found the patient's names or IDs in what would be written, \
             nothing is planned: {}",
            serde_json::to_string(&refused)?
        );
    }
    if unrewritable > 0 {
        bail!(
            "{unrewritable} file(s) cannot be rewritten safely (audit_folder says why); \
             nothing is planned"
        );
    }
    let mut out = save_manifest(core, "pseudonymize", &a.reason, ops)?;
    out["patients"] = json!(summary);
    out["kept"] = json!("dates, UIDs, OperatorsName/ReviewerName, and private elements unless keep_private is false");
    Ok(out)
}

// ---- demographics -----------------------------------------------------------

pub fn patient_demographics(core: &mut Core, a: PhiArgs, p: &Progress) -> Result<Value> {
    let dir = resolve(core, &a.path)?;
    let groups = by_patient(
        core,
        &dir,
        a.patient.as_deref(),
        a.cohort.as_deref().unwrap_or(""),
    )?;
    let mut out = Vec::new();
    for (shown, _, files) in groups {
        p.set(format!("Reading {shown}"));
        let raws = read_all(&files);
        let (dates, sexes, variants) = anon::demographics(&raws);
        if dates > 1 || sexes > 1 {
            out.push(json!({
                "patient": shown,
                "birth_dates": dates,
                "sexes": sexes,
                "files_per_variant": variants,
            }));
        }
    }
    Ok(json!({
        "folder": dir.display().to_string(),
        "disagreeing_patients": out,
        "note": "only the number of distinct values is shown; set_patient_demographics sets one",
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DemographicsArgs {
    /// Files or folders under the [pacs] roots (one patient's).
    pub paths: Vec<PathBuf>,
    /// The birth date: `YYYY0101`, or empty to clear it.
    pub birth_date: String,
    /// `M`, `F`, `O`, or empty.
    pub sex: String,
    pub reason: String,
}

pub fn set_patient_demographics(
    core: &mut Core,
    a: DemographicsArgs,
    p: &Progress,
) -> Result<Value> {
    let bd = a.birth_date.trim().to_string();
    if !(bd.is_empty()
        || (bd.len() == 8 && bd.ends_with("0101") && bd.chars().all(|c| c.is_ascii_digit())))
    {
        bail!("birth_date is YYYY0101 (the year only) or empty");
    }
    let sex = a.sex.trim().to_uppercase();
    if !matches!(sex.as_str(), "" | "M" | "F" | "O") {
        bail!("sex is M, F, O or empty");
    }
    let files = pick_files(core, &a.paths, &[], &[])?;
    let changes = Changes {
        set: vec![("00100030".into(), bd), ("00100040".into(), sex)],
        ..Default::default()
    };
    p.set("Checking the rewrites");
    let none = UidMaps::default();
    let results: Vec<Result<Option<journal::Op>>> = files
        .par_iter()
        .map(|f| {
            let s = simulate(f, &changes, &none)?;
            Ok((s.changed > 0).then(|| journal::Op::Rewrite {
                path: f.clone(),
                changes: changes.clone(),
                source: Some(s.source),
                expect: Some(s.expect),
            }))
        })
        .collect();
    let mut ops = Vec::new();
    for r in results {
        if let Some(op) = r? {
            ops.push(op);
        }
    }
    if ops.is_empty() {
        return Ok(json!({ "operations": 0, "note": "every file already says so" }));
    }
    save_manifest(core, "demographics", &a.reason, ops)
}
