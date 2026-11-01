//! The curation tools: checking, de-duplicating and correcting the archive
//! the operator named in `[pacs]` of `mcp.toml` ([`crate::curation`] does
//! the work).
//!
//! * Read-only: `audit_folder`, `find_duplicates`, `check_references`.
//! * Writing, always in two steps: `quarantine` and `set_patient_identity`
//!   only plan - they return a manifest id and what it would do, and write
//!   nothing but the manifest into the records folder - and
//!   `apply_manifest` does it, journaled and resumable; `undo_journal` takes
//!   it back. A manifest is read from the records folder by its id, never
//!   from the client, and every path in it is checked again when applied.
//!
//! Writes go only under the `[pacs]` roots and the quarantine folder, never
//! into a storage folder's top level (where the catalog would file a stray
//! file as "other"), and no path longer than 259 characters is written.
//! The tools read the patient's tags where they must (to set them) and
//! return none: findings name files, UIDs and counts, and every answer
//! passes the redactor like any other.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::Core;
use crate::archive::catalog;
use crate::curation::remap::UidMaps;
use crate::curation::{self, audit, dupes, journal, raw};
use crate::progress::Progress;

mod anon;
mod catalog_tools;
mod remap;
mod reorg;
pub use anon::*;
pub use catalog_tools::*;
pub use remap::*;
pub use reorg::*;

/// The most files a finding lists; the rest are counted.
const FILES_SHOWN: usize = 5;

/// `\\?\D:\x` as `D:\x`: canonical paths on Windows carry the prefix, and
/// it would count against the path limit and show in every answer.
fn plain(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if rest.as_bytes().get(1) == Some(&b':') => PathBuf::from(rest),
        _ => p,
    }
}

/// A folder or file a curation tool was given, under the `[pacs]` folders.
fn resolve(core: &Core, given: &Path) -> Result<PathBuf> {
    // Accept the labels answers report paths under (`pacs1/...`).
    let (real, _) = core.session.resolve_input(given)?;
    let (real, _) = core.session.config.pacs.resolve(&real)?;
    Ok(plain(real))
}

/// The canonical `[pacs]` roots and quarantine folder: where a write may go.
fn writable(core: &Core) -> Vec<PathBuf> {
    let pacs = &core.session.config.pacs;
    pacs.roots
        .iter()
        .chain(pacs.quarantine.iter())
        // Attachments moved to the records folder.
        .chain(pacs.records.iter())
        .filter_map(|r| r.canonicalize().ok().map(plain))
        .collect()
}

/// Whether `p` may be written: inside one of `roots`, below its top level.
fn may_write(roots: &[PathBuf], p: &Path) -> bool {
    roots.iter().any(|r| {
        p.strip_prefix(r)
            .is_ok_and(|rel| rel.components().count() >= 2)
    })
}

fn records(core: &Core) -> Result<PathBuf> {
    let r = core.session.config.pacs.records()?.to_path_buf();
    std::fs::create_dir_all(&r).with_context(|| format!("create {}", r.display()))?;
    Ok(plain(r.canonicalize()?))
}

/// What the station's archive holds, for references that leave a folder.
fn known(core: &Core) -> audit::Known {
    let mut k = audit::Known::default();
    let roots = core.session.config.archive_roots();
    if let Some(main) = roots.first().filter(|r| !r.as_os_str().is_empty()) {
        if catalog::exists(main) {
            let c = catalog::load(main);
            for (uid, s) in &c.series {
                k.series.insert(uid.clone());
                k.studies.insert(s.study.clone());
                k.sops.extend(s.files.iter().map(|f| f.4.clone()));
            }
        }
    }
    // The pixel index knows the [pacs] roots' files, frames of reference too.
    if let Ok(rec) = core.session.config.pacs.records() {
        for e in dupes::PixelIndex::load(rec).files.into_values() {
            k.sops.insert(e.sop);
            k.series.insert(e.series);
            if !e.study.is_empty() {
                k.studies.insert(e.study);
            }
            if !e.frame.is_empty() {
                k.frames.insert(e.frame);
            }
        }
    }
    k
}

fn files_json(files: &[PathBuf]) -> Value {
    let mut shown: Vec<Value> = files
        .iter()
        .take(FILES_SHOWN)
        .map(|f| json!(f.display().to_string()))
        .collect();
    if files.len() > FILES_SHOWN {
        shown.push(json!(format!("... and {} more", files.len() - FILES_SHOWN)));
    }
    Value::Array(shown)
}

fn report_json(report: &audit::Report, keep: impl Fn(&audit::Finding) -> bool) -> (Value, usize) {
    let mut n = 0;
    let studies: Vec<Value> = report
        .iter()
        .filter_map(|(study, list)| {
            let list: Vec<Value> = list
                .iter()
                .filter(|f| keep(f))
                .map(|f| {
                    n += 1;
                    json!({
                        "severity": f.severity,
                        "check": f.check,
                        "detail": f.detail,
                        "series": f.series,
                        "files": f.files.len(),
                        "examples": files_json(&f.files),
                    })
                })
                .collect();
            (!list.is_empty()).then(|| json!({ "study": study, "findings": list }))
        })
        .collect();
    (Value::Array(studies), n)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AuditArgs {
    /// A folder under the [pacs] roots, quarantine or records folder.
    pub path: PathBuf,
}

pub fn audit_folder(core: &mut Core, a: AuditArgs, p: &Progress) -> Result<Value> {
    let dir = resolve(core, &a.path)?;
    let files = curation::files_under(&dir);
    let (report, n) = audit::audit(&files, &known(core), false, p)?;
    let (studies, findings) = report_json(&report, |_| true);
    let count = |s: audit::Severity| {
        report
            .values()
            .flatten()
            .filter(|f| f.severity == s)
            .count()
    };
    Ok(json!({
        "folder": dir.display().to_string(),
        "files": n,
        "findings": findings,
        "broken": count(audit::Severity::Broken),
        "dangling_reference": count(audit::Severity::DanglingReference),
        "warning": count(audit::Severity::Warning),
        "by_study": studies,
    }))
}

pub fn check_references(core: &mut Core, a: AuditArgs, p: &Progress) -> Result<Value> {
    let dir = resolve(core, &a.path)?;
    let files = curation::files_under(&dir);
    let (report, n) = audit::audit(&files, &known(core), true, p)?;
    let (studies, findings) = report_json(&report, |f| {
        f.severity == audit::Severity::DanglingReference
    });
    Ok(json!({
        "folder": dir.display().to_string(),
        "files": n,
        "unresolved": findings,
        "by_study": studies,
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DuplicatesArgs {
    /// The folder whose files may be copies (an import).
    pub source: PathBuf,
    /// The folder to look in; default the first [pacs] root.
    #[serde(default)]
    pub against: Option<PathBuf>,
    /// Tags left out of the exact comparison, as `GGGGEEEE`; default
    /// PatientName and PatientID (group lengths are always left out).
    #[serde(default)]
    pub ignore_tags: Option<Vec<String>>,
    /// A pixel twin counts only between files of one patient folder (the
    /// source's first-level folder, through pacs-rules.toml, against the
    /// archive's cohort/patient folder). Default true.
    #[serde(default)]
    pub same_patient: Option<bool>,
    /// Which of the copies inside the source to keep, by words in their
    /// paths, best first; default full, case, plan.
    #[serde(default)]
    pub prefer: Option<Vec<String>>,
    /// Also plan moving the copies (series judged identical or pixel-twin)
    /// to the quarantine folder: returns a manifest for apply_manifest.
    #[serde(default)]
    pub quarantine_copies: bool,
    /// Why, for the manifest.
    #[serde(default)]
    pub reason: String,
}

pub fn find_duplicates(core: &mut Core, a: DuplicatesArgs, p: &Progress) -> Result<Value> {
    let source = resolve(core, &a.source)?;
    let against = match &a.against {
        Some(x) => resolve(core, x)?,
        None => {
            let first = core
                .session
                .config
                .pacs
                .roots
                .first()
                .context("no [pacs] roots are configured")?
                .clone();
            plain(first.canonicalize()?)
        }
    };
    if source.starts_with(&against) || against.starts_with(&source) {
        bail!("the source and the folder it is compared with must not contain each other");
    }
    let ignore = match &a.ignore_tags {
        Some(list) => list
            .iter()
            .map(|t| journal::parse_tag(t))
            .collect::<Result<Vec<_>>>()?,
        None => dupes::default_ignored(),
    };
    let rec = records(core)?;
    let mut index = dupes::PixelIndex::load(&rec);
    let target_files = curation::files_under(&against);
    let (indexed, read) = index.refresh(&target_files, p)?;
    index.save(&rec)?;
    let rules = curation::reorg::Rules::load(&rec)?;
    let src_files = curation::files_under(&source);
    let src = facts_of(&src_files, true, p)?;
    let patient_below = |base: &Path, f: &Path, depth: usize| -> Option<String> {
        let parts: Vec<String> = f
            .strip_prefix(base)
            .ok()?
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        (parts.len() > depth + 1).then(|| parts[depth].clone())
    };
    // The archive: cohort / patient / study / series / file.
    let same = |s: &Path, t: &Path| -> bool {
        let sp = patient_below(&source, s, 0).map(|f| rules.patient_code("", &f));
        let tp = patient_below(&against, t, 1).or_else(|| patient_below(&against, t, 0));
        match (sp, tp) {
            (Some(a), Some(b)) => curation::reorg::code_key(&a) == curation::reorg::code_key(&b),
            _ => true,
        }
    };
    let verdicts = if a.same_patient.unwrap_or(true) {
        dupes::find_in(&src, &indexed, &ignore, &same)
    } else {
        dupes::find_in(&src, &indexed, &ignore, &dupes::any_patient)
    };
    let prefer = a.prefer.clone().unwrap_or_else(dupes::default_prefer);
    let inside = dupes::inside(&src, &prefer);
    let (maps, splits) = dupes::twin_maps(&verdicts);
    let map_id = if maps.is_empty() {
        None
    } else {
        Some(save_map(core, "twins", &maps)?)
    };
    let count = |v: &str| verdicts.iter().filter(|x| x.verdict == v).count();
    let mut out = json!({
        "source": source.display().to_string(),
        "against": against.display().to_string(),
        "archive_files": target_files.len(),
        "hashed_now": read,
        "series": verdicts.len(),
        "identical": count("identical"),
        "pixel_twin": count("pixel-twin"),
        "partial": count("partial"),
        "ambiguous": count("ambiguous"),
        "new": count("new"),
        "copies": verdicts.iter().map(|v| v.exact + v.twins).sum::<usize>(),
        "internal_twins": verdicts.iter().map(|v| v.internal_twins).sum::<usize>(),
        "inside": {
            "same_sop_identical": inside.same_sop_identical.len(),
            "same_sop_versions": inside.same_sop_versions.len(),
            "versioned_sops": inside.versioned_sops,
            "twin_copies": inside.twin_copies.len(),
            "twin_series": inside.twin_series.iter().map(|(c, k)| json!({"copy": c, "kept": k})).collect::<Vec<_>>(),
        },
        "splits": splits,
        "uid_map": map_id,
        "uid_map_entries": maps.len(),
        "verdicts": verdicts.iter().map(|v| json!({
            "series": v.series,
            "modality": v.modality,
            "files": v.files,
            "verdict": v.verdict,
            "exact": v.exact,
            "pixel_twins": v.twins,
            "same_uid_other_content": v.same_uid_other_content,
            "internal_twins": v.internal_twins,
            "archive_series": v.matches,
        })).collect::<Vec<_>>(),
    });
    if a.quarantine_copies {
        let files: Vec<PathBuf> = verdicts
            .iter()
            .filter(|v| matches!(v.verdict, "identical" | "pixel-twin"))
            .flat_map(|v| v.copies.iter().cloned())
            .chain(inside.same_sop_identical.iter().cloned())
            .chain(inside.same_sop_versions.iter().cloned())
            .chain(inside.twin_copies.iter().cloned())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let reason = if a.reason.trim().is_empty() {
            format!("copies of files in {}", against.display())
        } else {
            a.reason.clone()
        };
        out["manifest"] = plan_quarantine(core, &files, &reason)?;
    }
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QuarantineArgs {
    /// Files or folders under the [pacs] roots.
    pub paths: Vec<PathBuf>,
    /// Only the files of these Series Instance UIDs.
    #[serde(default)]
    pub series: Vec<String>,
    /// Only the files of these Study Instance UIDs.
    #[serde(default)]
    pub studies: Vec<String>,
    /// Why, for the manifest and the journal.
    pub reason: String,
}

/// The dry run of a move into quarantine: `quarantine/<path below its
/// root>` (the first folder below a root being its cohort), saved as a
/// manifest.
fn plan_quarantine(core: &Core, files: &[PathBuf], reason: &str) -> Result<Value> {
    let q = plain(
        core.session
            .config
            .pacs
            .quarantine()?
            .canonicalize()
            .context("the [pacs] quarantine folder does not exist")?,
    );
    let roots: Vec<PathBuf> = core
        .session
        .config
        .pacs
        .roots
        .iter()
        .filter_map(|r| r.canonicalize().ok().map(plain))
        .collect();
    let mut ops = Vec::new();
    for f in files {
        let Some(root) = roots.iter().find(|r| f.starts_with(r)) else {
            bail!("{} is not under a [pacs] root", f.display());
        };
        let rel = f.strip_prefix(root).expect("under the root");
        ops.push(journal::Op::Move {
            from: f.clone(),
            to: q.join(rel),
        });
    }
    save_manifest(core, "quarantine", reason, ops)
}

fn save_manifest(core: &Core, tool: &str, reason: &str, ops: Vec<journal::Op>) -> Result<Value> {
    save_manifest_with(core, tool, reason, ops, UidMaps::default(), Vec::new())
}

/// [`save_manifest`] with the UID maps its rewrites apply and what the
/// planning noticed.
fn save_manifest_with(
    core: &Core,
    tool: &str,
    reason: &str,
    ops: Vec<journal::Op>,
    maps: UidMaps,
    noticed: Vec<String>,
) -> Result<Value> {
    let rec = records(core)?;
    let roots = writable(core);
    let mut warnings = journal::check_paths(&ops, &|p| may_write(&roots, p))?;
    warnings.extend(noticed);
    let (d, t) = crate::dicom_export::today();
    let m = journal::Manifest {
        id: journal::new_id(tool),
        tool: tool.into(),
        reason: super::super::phi::clean_text(reason),
        created: format!("{d}T{t}"),
        ops,
        warnings,
        maps,
    };
    journal::save(&rec, &m)?;
    let preview: Vec<Value> = m
        .ops
        .iter()
        .take(10)
        .map(|op| match op {
            journal::Op::Move { from, to } => json!({
                "move": from.display().to_string(),
                "to": to.display().to_string(),
            }),
            journal::Op::SetTags { path, tags } => json!({
                "rewrite": path.display().to_string(),
                "tags": tags.iter().map(|(t, _)| t).collect::<Vec<_>>(),
            }),
            journal::Op::Rewrite { path, changes, .. } => json!({
                "rewrite": path.display().to_string(),
                "set": changes.set.iter().map(|(t, _)| t).collect::<Vec<_>>(),
                "remove": changes.remove,
                "remove_private": changes.remove_private,
                "remap": changes.remap,
            }),
        })
        .collect();
    Ok(json!({
        "manifest": m.id,
        "operations": m.ops.len(),
        "warnings": m.warnings,
        "uid_map_entries": m.maps.len(),
        "first_operations": preview,
        "next": "apply_manifest with this id does it; nothing has been changed yet",
    }))
}

/// The files under `paths`, narrowed to `series` / `studies` when given
/// (their headers are read for that).
fn pick_files(
    core: &Core,
    paths: &[PathBuf],
    series: &[String],
    studies: &[String],
) -> Result<Vec<PathBuf>> {
    if paths.is_empty() {
        bail!("name at least one file or folder");
    }
    let mut files = Vec::new();
    for given in paths {
        let real = resolve(core, given)?;
        if real.is_dir() {
            files.extend(curation::files_under(&real));
        } else {
            files.push(real);
        }
    }
    files.sort();
    files.dedup();
    if series.is_empty() && studies.is_empty() {
        return Ok(files);
    }
    let series: HashSet<&str> = series.iter().map(|s| s.trim()).collect();
    let studies: HashSet<&str> = studies.iter().map(|s| s.trim()).collect();
    use rayon::prelude::*;
    Ok(files
        .into_par_iter()
        .filter(|f| {
            let Ok(r) = raw::RawFile::read(f) else {
                return false;
            };
            let t = |tag| r.text(tag).unwrap_or_default();
            series.contains(t(dicom_dictionary_std::tags::SERIES_INSTANCE_UID).as_str())
                || studies.contains(t(dicom_dictionary_std::tags::STUDY_INSTANCE_UID).as_str())
        })
        .collect())
}

pub fn quarantine(core: &mut Core, a: QuarantineArgs, _p: &Progress) -> Result<Value> {
    let files = pick_files(core, &a.paths, &a.series, &a.studies)?;
    if files.is_empty() {
        bail!("no file matches");
    }
    plan_quarantine(core, &files, &a.reason)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IdentityArgs {
    /// Files or folders under the [pacs] roots.
    pub paths: Vec<PathBuf>,
    /// Only the files of these Series / Study Instance UIDs.
    #[serde(default)]
    pub series: Vec<String>,
    #[serde(default)]
    pub studies: Vec<String>,
    /// The new PatientName (a patient code, say).
    pub name: String,
    /// The new PatientID.
    pub id: String,
    pub reason: String,
}

pub fn set_patient_identity(core: &mut Core, a: IdentityArgs, p: &Progress) -> Result<Value> {
    let (name, id) = (a.name.trim().to_string(), a.id.trim().to_string());
    if name.is_empty() || id.is_empty() {
        bail!("name and id must both be given");
    }
    if name.len() > 64 || id.len() > 64 {
        bail!("name and id are at most 64 characters");
    }
    let files = pick_files(core, &a.paths, &a.series, &a.studies)?;
    p.set("Reading the patient tags");
    use rayon::prelude::*;
    let name_tag = dicom_dictionary_std::tags::PATIENT_NAME;
    let id_tag = dicom_dictionary_std::tags::PATIENT_ID;
    // Only the files that would change, and only DICOM ones that rewrite.
    let planned: Vec<(PathBuf, Option<String>)> = files
        .par_iter()
        .map(|f| match raw::RawFile::read(f) {
            Ok(r) if r.is_dicom() => {
                if let Err(e) = r.rewritable() {
                    return (f.clone(), Some(format!("{e:#}")));
                }
                let same = r.text(name_tag).as_deref() == Some(name.as_str())
                    && r.text(id_tag).as_deref() == Some(id.as_str());
                if same {
                    (PathBuf::new(), None)
                } else {
                    (f.clone(), None)
                }
            }
            _ => (PathBuf::new(), None),
        })
        .collect();
    let refused: Vec<String> = planned
        .iter()
        .filter_map(|(f, e)| e.as_ref().map(|e| format!("{}: {e}", f.display())))
        .collect();
    if !refused.is_empty() {
        bail!(
            "{} file(s) cannot be rewritten safely, the first: {}",
            refused.len(),
            refused[0]
        );
    }
    let tags = vec![
        ("00100010".to_string(), name.clone()),
        ("00100020".to_string(), id.clone()),
    ];
    let ops: Vec<journal::Op> = planned
        .into_iter()
        .filter(|(f, _)| !f.as_os_str().is_empty())
        .map(|(path, _)| journal::Op::SetTags {
            path,
            tags: tags.clone(),
        })
        .collect();
    if ops.is_empty() {
        return Ok(
            json!({ "operations": 0, "note": "every file already carries this name and ID" }),
        );
    }
    let mut out = save_manifest(core, "identity", &a.reason, ops)?;
    out["unchanged_files"] = json!(files.len() - out["operations"].as_u64().unwrap_or(0) as usize);
    Ok(out)
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ManifestArgs {
    /// The id a planning tool returned.
    pub manifest: String,
}

pub fn apply_manifest(core: &mut Core, a: ManifestArgs, p: &Progress) -> Result<Value> {
    let rec = records(core)?;
    let roots = writable(core);
    let done = journal::apply(&rec, a.manifest.trim(), &|x| may_write(&roots, x), p)?;
    Ok(json!({
        "manifest": a.manifest.trim(),
        "done": done.done,
        "resumed": done.resumed,
        "unchanged": done.unchanged,
        "failed": done.failed,
        "note": if done.failed.is_empty() {
            "applied; undo_journal takes it back"
        } else {
            "stopped at the first failure; applying again resumes after the operations done"
        },
    }))
}

pub fn undo_journal(core: &mut Core, a: ManifestArgs, p: &Progress) -> Result<Value> {
    let rec = records(core)?;
    let roots = writable(core);
    let u = journal::undo(&rec, a.manifest.trim(), &|x| may_write(&roots, x), p)?;
    Ok(json!({
        "manifest": a.manifest.trim(),
        "undone": u.undone,
        "skipped": u.skipped,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_go_below_a_roots_top_level_only() {
        let roots = vec![PathBuf::from("/pacs")];
        assert!(may_write(&roots, Path::new("/pacs/cohort/p/s.dcm")));
        assert!(!may_write(&roots, Path::new("/pacs/stray.dcm")));
        assert!(!may_write(&roots, Path::new("/elsewhere/x/y.dcm")));
        assert_eq!(plain(PathBuf::from(r"\\?\D:\x")), PathBuf::from(r"D:\x"));
    }
}
