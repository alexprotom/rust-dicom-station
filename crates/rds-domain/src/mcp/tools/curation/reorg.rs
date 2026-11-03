//! `plan_reorganize` / `apply_reorganize`: sorting an import (or a cohort
//! of the archive) into the archive's layout by `pacs-rules.toml`
//! ([`crate::curation::reorg`] names everything).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::*;
use crate::curation::reorg::{self, Attachments, Rules};

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReorganizeArgs {
    /// The folder to sort, under the [pacs] roots or the quarantine: its
    /// first-level folders are the patients.
    pub folder: PathBuf,
    /// The cohort folder to sort into (in the first [pacs] root, or `into`).
    pub cohort: String,
    /// The storage folder to sort into; default the first [pacs] root.
    #[serde(default)]
    pub into: Option<PathBuf>,
    /// One patient code for the whole folder (its files are one patient).
    #[serde(default)]
    pub patient: Option<String>,
    /// Files that are not DICOM: `report` (default), `keep` (beside the
    /// series they came with), `records`, `quarantine`.
    #[serde(default)]
    pub attachments: Option<String>,
    pub reason: String,
}

/// The archive's study folders by Study Instance UID, below `root`: from the
/// station's catalog, where it covers `root`.
fn known_studies(core: &Core, root: &Path) -> HashMap<String, PathBuf> {
    let mut out = HashMap::new();
    let roots = core.session.config.archive_roots();
    let Some(main) = roots.first().filter(|r| catalog::exists(r)) else {
        return out;
    };
    let c = catalog::load(main);
    let Some(idx) = c.storage.iter().position(|s| catalog::same_folder(root, s)) else {
        return out;
    };
    for s in c.series.values() {
        for f in s.files.iter().filter(|f| f.0 as usize == idx) {
            let parts: Vec<&str> = f.1.split('/').collect();
            if parts.len() >= 5 {
                out.entry(s.study.clone())
                    .or_insert_with(|| parts[..3].iter().collect::<PathBuf>());
            }
        }
    }
    out
}

pub fn plan_reorganize(core: &mut Core, a: ReorganizeArgs, p: &Progress) -> Result<Value> {
    let source = resolve(core, &a.folder)?;
    let root = match &a.into {
        Some(x) => resolve(core, x)?,
        None => first_root(core)?,
    };
    let pacs_roots: Vec<PathBuf> = core
        .session
        .config
        .pacs
        .roots
        .iter()
        .filter_map(|r| r.canonicalize().ok().map(plain))
        .collect();
    if !pacs_roots.contains(&root) {
        bail!("into must be one of the [pacs] roots");
    }
    let cohort = a.cohort.trim();
    if reorg::clean(cohort).is_empty() {
        bail!("name the cohort folder");
    }
    let attachments = match a.attachments.as_deref().map(str::trim) {
        None | Some("") | Some("report") => Attachments::Report,
        Some("keep") => Attachments::Keep,
        Some("records") => Attachments::Records,
        Some("quarantine") => Attachments::Quarantine,
        Some(other) => bail!("attachments is report, keep, records or quarantine, not {other}"),
    };
    let rec = records(core)?;
    let rules = Rules::load(&rec)?;
    let quarantine = core
        .session
        .config
        .pacs
        .quarantine
        .as_ref()
        .and_then(|q| q.canonicalize().ok())
        .map(plain);
    let known = known_studies(core, &root);
    let input = reorg::Input {
        source: &source,
        root: &root,
        cohort,
        patient: a
            .patient
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty()),
        quarantine: quarantine.as_deref(),
        records: &rec,
        attachments,
        known_studies: &known,
    };
    let plan = reorg::plan(&input, &rules, p)?;
    let summary = json!({
        "files": plan.files,
        "patients": plan.patients,
        "studies": plan.studies,
        "series": plan.series,
        "moves": plan.moves,
        "already_in_place": plan.in_place,
        "identity_rewrites": plan.identity_rewrites,
        "copies_to_quarantine": plan.to_quarantine,
        "merged_into_existing_studies": plan.merged_studies,
        "conflicts": plan.conflicts,
        "attachments": files_json(&plan.attachments.iter().map(PathBuf::from).collect::<Vec<_>>()),
        "attachment_count": plan.attachments.len(),
        "study_folders": plan.study_folders.iter().take(40).map(|(u, f)| json!({"study": u, "folder": f})).collect::<Vec<_>>(),
        "rules": if rec.join(reorg::RULES_FILE).is_file() { "records/pacs-rules.toml" } else { "the defaults (no pacs-rules.toml in the records folder)" },
    });
    if plan.ops.is_empty() {
        return Ok(json!({
            "operations": 0,
            "summary": summary,
            "warnings": plan.warnings,
            "note": "everything is where the rules put it",
        }));
    }
    let mut out = save_manifest_with(
        core,
        "reorganize",
        &a.reason,
        plan.ops,
        Default::default(),
        plan.warnings,
    )?;
    out["summary"] = summary;
    out["next"] = json!(
        "apply_reorganize with this id rewrites the patient tags and moves the files; \
         list_archive (refresh) then catalogs them"
    );
    Ok(out)
}

pub fn apply_reorganize(core: &mut Core, a: ManifestArgs, p: &Progress) -> Result<Value> {
    let m = journal::load(&records(core)?, a.manifest.trim())?;
    if m.tool != "reorganize" {
        bail!(
            "manifest {} was made by {}, not plan_reorganize: use apply_manifest",
            m.id,
            m.tool
        );
    }
    apply_manifest(core, a, p)
}
