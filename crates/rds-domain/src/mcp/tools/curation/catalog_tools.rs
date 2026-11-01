//! Catalog maintenance: `catalog_relink`, `set_label`, `fill_study_labels`
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
