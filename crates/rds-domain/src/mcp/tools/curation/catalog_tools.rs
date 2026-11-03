//! Catalog maintenance: `catalog_relink`, `set_label`, `fill_study_labels`,
//! `catalog_ignore`, and the read-only `check_patient_identity`
//! ([`crate::curation::catalog_fix`]). The catalog is the station's
//! `rds-pacs-catalog.json` in the first [pacs] root that has one (else the
//! archive's main folder); before each change a copy goes to
//! `records/catalog-backups/`, and the change is written the station's own
//! way (under its lock, one line per entry, one rename).

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::*;
use crate::curation::catalog_fix;
use crate::curation::reorg::{self, Rules};

/// The main folder of the catalog the tools change.
fn catalog_main(core: &Core) -> Result<PathBuf> {
    for r in &core.session.config.pacs.roots {
        if catalog::exists(r) {
            return Ok(r.clone());
        }
    }
    let roots = core.session.config.archive_roots();
    match roots.first() {
        Some(main) if catalog::exists(main) => Ok(main.clone()),
        _ => bail!("no PACS catalog in the [pacs] roots or the archive; list_archive makes one"),
    }
}

/// Keep a copy of the catalog before it changes; returns where.
fn backup(core: &Core, main: &Path) -> Result<PathBuf> {
    let dir = records(core)?.join("catalog-backups");
    std::fs::create_dir_all(&dir)?;
    let to = dir.join(format!("{}.json", journal::new_id("catalog")));
    std::fs::copy(catalog::path_in(main), &to)?;
    Ok(to)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CohortArgs {
    /// The cohort's folder name (directly in a storage folder).
    pub cohort: String,
}

pub fn catalog_relink(core: &mut Core, a: CohortArgs, _p: &Progress) -> Result<Value> {
    let main = catalog_main(core)?;
    let kept = backup(core, &main)?;
    let cohort = a.cohort.trim().to_string();
    let r = catalog::update(&main, |c| Ok(catalog_fix::relink(c, &cohort)))?;
    Ok(json!({
        "cohort": cohort,
        "result": r,
        "backup": kept.display().to_string(),
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IgnoreArgs {
    /// Folder name patterns to add (`*` any run of characters, case
    /// ignored; a name, never a path).
    #[serde(default)]
    pub add: Vec<String>,
    /// Patterns to take out again.
    #[serde(default)]
    pub remove: Vec<String>,
}

pub fn catalog_ignore(core: &mut Core, a: IgnoreArgs, _p: &Progress) -> Result<Value> {
    for p in &a.add {
        let p = p.trim();
        if p.is_empty() || p.len() > 64 || p.contains(['/', '\\']) || p.chars().all(|c| c == '*') {
            bail!("a pattern is a folder name of at most 64 characters, '*' for any run of characters: '{p}' is not");
        }
    }
    let main = catalog_main(core)?;
    let changed = !a.add.is_empty() || !a.remove.is_empty();
    let kept = if changed {
        Some(backup(core, &main)?)
    } else {
        None
    };
    let list = catalog::update(&main, |c| {
        for p in &a.remove {
            c.ignore.retain(|x| !x.eq_ignore_ascii_case(p.trim()));
        }
        for p in &a.add {
            let p = p.trim().to_string();
            if !c.ignore.iter().any(|x| x.eq_ignore_ascii_case(&p)) {
                c.ignore.push(p);
            }
        }
        Ok(c.ignore.clone())
    })?;
    Ok(json!({
        "built_in": catalog::IGNORED_FOLDERS,
        "ignore": list,
        "backup": kept.map(|k| k.display().to_string()),
        "note": "a scan (list_archive) looks into no folder of these names; the files in them \
                 leave the catalog with the next scan, and nothing on disk is touched",
    }))
}

pub fn check_patient_identity(
    core: &mut Core,
    _a: crate::mcp::tools::NoArgs,
    p: &Progress,
) -> Result<Value> {
    let main = catalog_main(core)?;
    let c = catalog::load(&main);
    p.set("Reading one file of each study filed in several cohorts");
    let codes = core.session.config.pacs.codes();
    let clashes = catalog_fix::identity_clashes(&c);
    let shown: Vec<Value> = clashes
        .iter()
        .enumerate()
        .map(|(i, k)| {
            // A code is shown with its folders; anything else might be a
            // name or a real ID, and stays a number.
            let code = codes.is_code(&k.patient);
            let parts: Vec<Value> = k
                .parts
                .iter()
                .map(|x| {
                    let folder = c.storage.get(x.storage as usize).map(|r| {
                        catalog::abs_of(Path::new(r), &x.folder)
                            .display()
                            .to_string()
                    });
                    json!({
                        "cohort": x.cohort,
                        "folder": if code { folder } else { None },
                        "studies": x.studies,
                        "birth_date_and_sex_variants": x.variants,
                    })
                })
                .collect();
            json!({
                "patient": if code { k.patient.clone() } else { format!("patient #{}", i + 1) },
                "cohorts": parts,
            })
        })
        .collect();
    Ok(json!({
        "clashes": shown,
        "note": "one Patient ID filed in cohort folders whose files share no birth date and sex: \
                 two people under one code, listed as one patient. Split them by giving one \
                 cohort's folder a code of its own: set_patient_identity with that folder as \
                 paths, then apply_manifest and list_archive. The values are never shown.",
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelArgs {
    /// A patient's key (their code) ...
    #[serde(default)]
    pub patient: Option<String>,
    /// ... or a Study Instance UID.
    #[serde(default)]
    pub study: Option<String>,
    /// The label; empty clears it.
    pub text: String,
}

pub fn set_label(core: &mut Core, a: LabelArgs, _p: &Progress) -> Result<Value> {
    let text: String = a
        .text
        .chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string();
    if text.chars().count() > 128 {
        bail!("a label is at most 128 characters");
    }
    let main = catalog_main(core)?;
    let kept = backup(core, &main)?;
    let what = match (&a.patient, &a.study) {
        (Some(p), None) => {
            let p = p.trim().to_string();
            catalog::update(&main, |c| match c.patients.get_mut(&p) {
                Some(rec) => {
                    rec.label = text.clone();
                    Ok(())
                }
                None => bail!("the catalog has no patient {p}"),
            })?;
            "patient"
        }
        (None, Some(s)) => {
            let s = s.trim().to_string();
            catalog::update(&main, |c| match c.studies.get_mut(&s) {
                Some(rec) => {
                    rec.label = text.clone();
                    Ok(())
                }
                None => bail!("the catalog has no study {s}"),
            })?;
            "study"
        }
        _ => bail!("name a patient or a study, one of them"),
    };
    Ok(json!({ "labelled": what, "backup": kept.display().to_string() }))
}

pub fn fill_study_labels(core: &mut Core, a: CohortArgs, p: &Progress) -> Result<Value> {
    let main = catalog_main(core)?;
    let rules = Rules::load(&records(core)?)?;
    let cohort = a.cohort.trim().to_string();
    // Positions read from the files, outside the lock.
    p.set("Reading a few headers per study");
    let c = catalog::load(&main);
    let folders = catalog_fix::study_folders(&c, &cohort);
    let mut positions: std::collections::HashMap<String, &'static str> = Default::default();
    for (uid, (_, _, folder)) in &folders {
        if c.studies
            .get(uid)
            .is_some_and(|s| !s.label.trim().is_empty())
        {
            continue;
        }
        if folder.contains("_Upright-") || folder.contains("_Supine-") {
            continue;
        }
        let files: Vec<PathBuf> = c
            .series
            .values()
            .filter(|s| s.study == *uid)
            .flat_map(|s| s.files.iter().take(3))
            .filter_map(|f| {
                let root = c.storage.get(f.0 as usize)?;
                Some(catalog::abs_of(Path::new(root), &f.1))
            })
            .collect();
        positions.insert(uid.clone(), reorg::position_of(&files, &rules));
    }
    let kept = backup(core, &main)?;
    let n = catalog::update(&main, |c| {
        Ok(catalog_fix::fill_labels(c, &cohort, &|uid| {
            positions.get(uid).copied()
        }))
    })?;
    Ok(json!({
        "cohort": cohort,
        "labelled": n,
        "studies": folders.len(),
        "backup": kept.display().to_string(),
        "note": "studies that had a label kept it",
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Set4dArgs {
    /// The Study Instance UID.
    pub study: String,
    /// The series of the group, phases in their order: each a Series
    /// Instance UID, or {"series": UID, "role": "phase|untagged|AVG|MIP|MinIP|trace",
    /// "phase": "30%"}. A bare UID is a phase, labelled t1, t2, ...
    pub members: Vec<Member4d>,
    /// What to call it; empty: `4DCT <date> (<n> phases)`.
    #[serde(default)]
    pub name: String,
}

#[derive(Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Member4d {
    Series(String),
    Full {
        series: String,
        #[serde(default)]
        role: String,
        #[serde(default)]
        phase: String,
    },
}

/// The catalog's archive, for the 4D tools.
fn fourd_archive(core: &Core) -> Result<(crate::archive::Archive, PathBuf)> {
    let main = catalog_main(core)?;
    let kept = backup(core, &main)?;
    Ok((crate::archive::Archive::new(&main), kept))
}

pub fn set_4d_group(core: &mut Core, a: Set4dArgs, _p: &Progress) -> Result<Value> {
    let study = a.study.trim().to_string();
    if !crate::archive::is_uid(&study) {
        bail!("study must be a Study Instance UID");
    }
    let members: Vec<(String, String, String)> = a
        .members
        .into_iter()
        .map(|m| match m {
            Member4d::Series(s) => (s.trim().to_string(), String::new(), String::new()),
            Member4d::Full {
                series,
                role,
                phase,
            } => (series.trim().to_string(), role, phase),
        })
        .collect();
    let name: String = a
        .name
        .chars()
        .filter(|c| !c.is_control())
        .take(128)
        .collect();
    let (ar, kept) = fourd_archive(core)?;
    let id = ar.set_4d_group(&study, &members, &name)?;
    let groups = ar.fourd_groups(&study);
    let g = groups.iter().find(|g| g.set.group == id);
    Ok(json!({
        "group": id,
        "name": g.map(|g| g.set.name.clone()),
        "members": g.map_or(0, |g| g.members.len()),
        "backup": kept.display().to_string(),
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Confirm4dArgs {
    /// The group's id, as list_archive or list_archive_4d_groups give it.
    pub group: String,
    /// true: it is a 4DCT (a probable one becomes confirmed); false: it is
    /// none (the group is dissolved and stays so through rescans).
    pub yes: bool,
    /// Instead: forget what was decided about the group, and let the rules
    /// decide again.
    #[serde(default)]
    pub reset: bool,
}

pub fn confirm_4d(core: &mut Core, a: Confirm4dArgs, _p: &Progress) -> Result<Value> {
    let group = a.group.trim().to_string();
    let (ar, kept) = fourd_archive(core)?;
    if a.reset {
        ar.reset_4d_group(&group)?;
        return Ok(json!({"group": group, "reset": true, "backup": kept.display().to_string()}));
    }
    let id = ar.confirm_4d(&group, a.yes)?;
    Ok(json!({
        "group": id,
        "confirmed": a.yes,
        "backup": kept.display().to_string(),
    }))
}
