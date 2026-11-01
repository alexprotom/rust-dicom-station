//! The tools that change what objects refer to: `relink_structure_set`,
//! `remap_references`, `merge_studies`, `merge_frames` ([`crate::curation::remap`],
//! [`crate::curation::relink`] do the work). Each plans: it builds typed UID
//! maps, runs every rewrite dry, refuses when an old UID would stay, and
//! saves a manifest (with its maps) for `apply_manifest`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::*;
use crate::curation::facts::{self, Facts};
use crate::curation::relink::{self, Mapping};
use crate::curation::remap::{plan_rewrites, Changes, Planned, UidKind, UidMaps};

/// Facts of many files at once (with pixel hashes when `hash`).
pub(super) fn facts_of(files: &[PathBuf], hash: bool, p: &Progress) -> Result<Vec<Facts>> {
    let n = files.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<Facts> = files
        .par_iter()
        .map(|f| {
            let i = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i.is_multiple_of(256) {
                p.set(format!("Reading headers {i}/{n}"));
            }
            facts::read(f, hash)
        })
        .collect();
    if p.cancelled() {
        bail!(crate::progress::CANCELLED);
    }
    Ok(out)
}

// ---- saved maps -----------------------------------------------------------

fn map_path(records: &Path, id: &str) -> Result<PathBuf> {
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        bail!("'{id}' is not a map id");
    }
    Ok(records.join("maps").join(format!("{id}.json")))
}

/// Keep a UID map in the records folder; returns its id.
pub(super) fn save_map(core: &Core, tool: &str, maps: &UidMaps) -> Result<String> {
    let rec = records(core)?;
    let id = journal::new_id(tool);
    let path = map_path(&rec, &id)?;
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, serde_json::to_vec_pretty(maps)?)?;
    Ok(id)
}

pub(super) fn load_map(core: &Core, id: &str) -> Result<UidMaps> {
    let path = map_path(&records(core)?, id.trim())?;
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("no UID map {id} is saved"))?;
    Ok(serde_json::from_str(&text)?)
}

/// What `known` and the pixel index say exists, as a resolver.
fn resolver(core: &Core) -> impl Fn(UidKind, &str) -> bool + Sync {
    let k = known(core);
    move |kind, uid| match kind {
        UidKind::Object => k.sops.contains(uid) || k.series.contains(uid),
        UidKind::Frame => k.frames.contains(uid),
        // Studies: only the files' own (an object's study reference to a
        // study the archive has is found through its files' facts below).
        UidKind::Study => k.studies.contains(uid),
        _ => true,
    }
}

/// The planned rewrites as a manifest, or a refusal saying why.
fn save_planned(
    core: &Core,
    tool: &str,
    reason: &str,
    planned: Planned,
    maps: UidMaps,
    refuse_unresolved: bool,
) -> Result<Value> {
    if let Some((f, e)) = planned.refused.first() {
        bail!(
            "{} file(s) cannot be rewritten, the first: {}: {e}",
            planned.refused.len(),
            f.display()
        );
    }
    if let Some((f, k, uid)) = planned.leftovers.first() {
        bail!(
            "{} old UID(s) would stay, the first a {k:?} {uid} in {}; nothing is planned",
            planned.leftovers.len(),
            f.display()
        );
    }
    let unresolved: Vec<Value> = planned
        .unresolved
        .iter()
        .take(20)
        .map(|(k, u)| json!({ "kind": k, "uid": u }))
        .collect();
    if refuse_unresolved && !planned.unresolved.is_empty() {
        bail!(
            "{} reference(s) would lead nowhere after the rewrite: {}; nothing is planned",
            planned.unresolved.len(),
            serde_json::to_string(&unresolved)?
        );
    }
    if planned.ops.is_empty() {
        return Ok(json!({
            "operations": 0,
            "unchanged_files": planned.unchanged,
            "note": "nothing to rewrite: no file holds a UID of the map",
        }));
    }
    let entries = maps.len();
    let noticed = if planned.unresolved.is_empty() {
        Vec::new()
    } else {
        vec![format!(
            "{} reference(s) do not resolve in the folders or the archive (check_references lists them)",
            planned.unresolved.len()
        )]
    };
    let mut out = save_manifest_with(core, tool, reason, planned.ops, maps, noticed)?;
    out["unchanged_files"] = json!(planned.unchanged);
    out["uid_map_entries"] = json!(entries);
    out["unresolved"] = json!(unresolved);
    Ok(out)
}

// ---- relink_structure_set -------------------------------------------------

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RelinkArgs {
    /// The structure set file (under the [pacs] roots).
    pub rtstruct: PathBuf,
    /// The Series Instance UID of the CT series to point it at.
    pub target_series: String,
    /// A folder holding the target series; default: found through the
    /// station's catalog.
    #[serde(default)]
    pub target: Option<PathBuf>,
    /// The folder of the export the structure set came with (its images);
    /// default: the folder two levels up from it (its study's).
    #[serde(default)]
    pub originals: Option<PathBuf>,
    /// `pixel` (default): through pixel twins; `z`: through the slice
    /// position, after checking the series lie the same way.
    #[serde(default)]
    pub mapping: Option<String>,
    /// Series of the export (UIDs) that shared the structure set's frame of
    /// reference and move with it into the target study and frame.
    #[serde(default)]
    pub merge_series: Vec<String>,
    pub reason: String,
}

/// The files of a series in the station's catalog.
fn catalog_series_files(core: &Core, series: &str) -> Vec<PathBuf> {
    let roots = core.session.config.archive_roots();
    let Some(main) = roots.first().filter(|r| catalog::exists(r)) else {
        return Vec::new();
    };
    let c = catalog::load(main);
    let Some(s) = c.series.get(series) else {
        return Vec::new();
    };
    s.files
        .iter()
        .filter_map(|f| {
            let root = c.storage.get(f.0 as usize)?;
            Some(catalog::abs_of(Path::new(root), &f.1))
        })
        .collect()
}

pub fn relink_structure_set(core: &mut Core, a: RelinkArgs, p: &Progress) -> Result<Value> {
    let rs = resolve(core, &a.rtstruct)?;
    let mapping = match a.mapping.as_deref().map(str::trim) {
        None | Some("") | Some("pixel") => Mapping::Pixel,
        Some("z") => Mapping::Z,
        Some(other) => bail!("mapping is pixel or z, not {other}"),
    };
    let series = a.target_series.trim().to_string();
    let target_files: Vec<PathBuf> = match &a.target {
        Some(t) => curation::files_under(&resolve(core, t)?),
        None => catalog_series_files(core, &series),
    };
    if target_files.is_empty() {
        bail!("the target series was not found; name the folder that holds it (target)");
    }
    let hash = mapping == Mapping::Pixel;
    p.set("Reading the target series");
    let target: Vec<Facts> = facts_of(&target_files, hash, p)?
        .into_iter()
        .filter(|f| f.series == series)
        .collect();
    if target.is_empty() {
        bail!("no file of series {series} is in the target folder");
    }
    let originals_dir = match &a.originals {
        Some(o) => resolve(core, o)?,
        None => rs
            .parent()
            .and_then(Path::parent)
            .context("the structure set has no study folder")?
            .to_path_buf(),
    };
    p.set("Reading the export");
    let originals = facts_of(&curation::files_under(&originals_dir), hash, p)?;
    let plan = relink::plan(&rs, &target, &originals, mapping, &a.merge_series)?;
    // After the rewrite, references must lead to the target or to the
    // archive.
    let base = resolver(core);
    let t_sops: BTreeSet<String> = target.iter().map(|f| f.sop.clone()).collect();
    let (t_series, t_study, t_frame) = (
        target[0].series.clone(),
        target[0].study.clone(),
        target[0].frame.clone(),
    );
    let resolves = |k: UidKind, u: &str| match k {
        UidKind::Object => t_sops.contains(u) || u == t_series || base(k, u),
        UidKind::Study => u == t_study || base(k, u),
        UidKind::Frame => u == t_frame || base(k, u),
        _ => true,
    };
    let changes = Changes {
        remap: true,
        ..Default::default()
    };
    p.set("Checking the rewrites");
    let planned = plan_rewrites(&plan.files, &changes, &plan.maps, &resolves);
    let mut out = save_planned(core, "relink", &a.reason, planned, plan.maps, true)?;
    out["images_by_pixels"] = json!(plan.by_pixels);
    out["images_by_z"] = json!(plan.by_z);
    out["images_by_contour_z"] = json!(plan.by_contour_z);
    out["merged_series_files"] = json!(plan.files.len() - 1);
    Ok(out)
}

// ---- remap_references -----------------------------------------------------

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RemapArgs {
    /// The objects to rewrite: files or folders under the [pacs] roots.
    pub objects: Vec<PathBuf>,
    /// Build the maps from the pixel twins of this folder's image series
    /// in `against` (the import the objects came with).
    #[serde(default)]
    pub source: Option<PathBuf>,
    /// The folder the twins are in; default the first [pacs] root.
    #[serde(default)]
    pub against: Option<PathBuf>,
    /// A UID map saved earlier (remap_references, dedupe_rt_objects).
    #[serde(default)]
    pub map: Option<String>,
    /// Entries by hand: old UID → new, per kind.
    #[serde(default)]
    pub objects_map: BTreeMap<String, String>,
    #[serde(default)]
    pub studies_map: BTreeMap<String, String>,
    #[serde(default)]
    pub frames_map: BTreeMap<String, String>,
    pub reason: String,
}

/// The maps the pixel twins of `source` in `against` give, refusing a
/// split (one source study or frame mapping to two).
pub(super) fn maps_from_twins(
    core: &Core,
    source: &Path,
    against: Option<&Path>,
    p: &Progress,
) -> Result<(UidMaps, Vec<dupes::SeriesVerdict>)> {
    let against = match against {
        Some(x) => resolve(core, x)?,
        None => first_root(core)?,
    };
    let rec = records(core)?;
    let mut index = dupes::PixelIndex::load(&rec);
    let (indexed, _) = index.refresh(&curation::files_under(&against), p)?;
    index.save(&rec)?;
    let verdicts = dupes::find(
        &curation::files_under(source),
        &indexed,
        &dupes::default_ignored(),
        &dupes::any_patient,
        p,
    )?;
    let (maps, splits) = dupes::twin_maps(&verdicts);
    if !splits.is_empty() {
        bail!(
            "{} source stud(ies) or frame(s) of reference have copies in more than one of the \
             archive's: {}; merge those first (merge_studies / merge_frames) or map by hand",
            splits.len(),
            serde_json::to_string(&splits.iter().take(5).collect::<Vec<_>>())?
        );
    }
    Ok((maps, verdicts))
}

pub(super) fn first_root(core: &Core) -> Result<PathBuf> {
    let first = core
        .session
        .config
        .pacs
        .roots
        .first()
        .context("no [pacs] roots are configured")?
        .clone();
    Ok(plain(first.canonicalize()?))
}

pub fn remap_references(core: &mut Core, a: RemapArgs, p: &Progress) -> Result<Value> {
    let mut maps = UidMaps::default();
    if let Some(id) = &a.map {
        maps.merge(&load_map(core, id)?)?;
    }
    let by_hand = UidMaps {
        objects: a.objects_map.clone(),
        studies: a.studies_map.clone(),
        frames: a.frames_map.clone(),
    };
    maps.merge(&by_hand)?;
    if let Some(src) = &a.source {
        let src = resolve(core, src)?;
        let against = a.against.as_ref().map(|x| resolve(core, x)).transpose()?;
        let (twins, _) = maps_from_twins(core, &src, against.as_deref(), p)?;
        maps.merge(&twins)?;
    }
    maps.resolve()?;
    if maps.is_empty() {
        bail!("the map is empty: give source (twins), map, or entries by hand");
    }
    let files = pick_files(core, &a.objects, &[], &[])?;
    let changes = Changes {
        remap: true,
        ..Default::default()
    };
    // The targets of the map exist by construction (twins, a saved map,
    // the person's word).
    let base = resolver(core);
    let targets: BTreeSet<(UidKind, String)> = [
        (UidKind::Object, &maps.objects),
        (UidKind::Study, &maps.studies),
        (UidKind::Frame, &maps.frames),
    ]
    .into_iter()
    .flat_map(|(k, m)| m.values().map(move |v| (k, v.clone())))
    .collect();
    let resolves = |k: UidKind, u: &str| targets.contains(&(k, u.to_string())) || base(k, u);
    p.set("Checking the rewrites");
    let planned = plan_rewrites(&files, &changes, &maps, &resolves);
    let map_id = save_map(core, "remap", &maps)?;
    let mut out = save_planned(core, "remap", &a.reason, planned, maps, false)?;
    out["map"] = json!(map_id);
    Ok(out)
}

// ---- merge_studies / merge_frames -----------------------------------------

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MergeArgs {
    /// Where the files are: folders under the [pacs] roots (a patient's,
    /// say). Every file that holds one of `from` is rewritten.
    pub paths: Vec<PathBuf>,
    /// The Study Instance UIDs (merge_studies) or Frame of Reference UIDs
    /// (merge_frames) to merge away.
    pub from: Vec<String>,
    /// The one they become.
    pub into: String,
    pub reason: String,
}

fn merge(core: &mut Core, a: MergeArgs, kind: UidKind, p: &Progress) -> Result<Value> {
    let into = a.into.trim();
    if into.is_empty() || a.from.is_empty() {
        bail!("name what to merge (from) and into what (into)");
    }
    let mut maps = UidMaps::default();
    for f in &a.from {
        if maps.insert(kind, f.trim(), into).is_err() {
            bail!("{f} is named twice");
        }
    }
    let files = pick_files(core, &a.paths, &[], &[])?;
    let changes = Changes {
        remap: true,
        ..Default::default()
    };
    // Only the files that hold a merged UID somewhere change; the target
    // must be among the files (or the archive).
    let base = resolver(core);
    let resolves = |k: UidKind, u: &str| (k == kind && u == into) || base(k, u);
    p.set("Checking the rewrites");
    let planned = plan_rewrites(&files, &changes, &maps, &resolves);
    let tool = if kind == UidKind::Study {
        "merge-studies"
    } else {
        "merge-frames"
    };
    let mut out = save_planned(core, tool, &a.reason, planned, maps, false)?;
    out["next"] = json!(
        "apply_manifest with this id rewrites them; plan_reorganize then sorts the merged files"
    );
    Ok(out)
}

pub fn merge_studies(core: &mut Core, a: MergeArgs, p: &Progress) -> Result<Value> {
    merge(core, a, UidKind::Study, p)
}

pub fn merge_frames(core: &mut Core, a: MergeArgs, p: &Progress) -> Result<Value> {
    merge(core, a, UidKind::Frame, p)
}

// ---- dedupe_rt_objects ----------------------------------------------------

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DedupeArgs {
    /// The import whose RT objects may be copies.
    pub source: PathBuf,
    /// The folder to look in; default the first [pacs] root.
    #[serde(default)]
    pub against: Option<PathBuf>,
    /// Start from the UID map the source's image twins give (default true).
    #[serde(default)]
    pub use_twins: Option<bool>,
    /// And from a saved map.
    #[serde(default)]
    pub map: Option<String>,
    /// Also plan moving the copies to the quarantine folder.
    #[serde(default)]
    pub quarantine_copies: bool,
    #[serde(default)]
    pub reason: String,
}

pub fn dedupe_rt_objects(core: &mut Core, a: DedupeArgs, p: &Progress) -> Result<Value> {
    let source = resolve(core, &a.source)?;
    let against = match &a.against {
        Some(x) => resolve(core, x)?,
        None => first_root(core)?,
    };
    if source.starts_with(&against) || against.starts_with(&source) {
        bail!("the source and the folder it is compared with must not contain each other");
    }
    let rec = records(core)?;
    let mut index = dupes::PixelIndex::load(&rec);
    let (indexed, _) = index.refresh(&curation::files_under(&against), p)?;
    index.save(&rec)?;
    let src_files = curation::files_under(&source);
    let mut maps = UidMaps::default();
    if let Some(id) = &a.map {
        maps.merge(&load_map(core, id)?)?;
    }
    let mut splits = Vec::new();
    if a.use_twins.unwrap_or(true) {
        let verdicts = dupes::find(
            &src_files,
            &indexed,
            &dupes::default_ignored(),
            &dupes::any_patient,
            p,
        )?;
        let (twins, s) = dupes::twin_maps(&verdicts);
        splits = s;
        maps.merge(&twins)?;
    }
    let objects: Vec<PathBuf> = indexed
        .iter()
        .filter(|(_, e)| curation::objects::is_rt_object(&e.class))
        .map(|(f, _)| f.clone())
        .collect();
    let d = curation::objects::dedupe(&src_files, &objects, maps, p)?;
    let mut how: BTreeMap<&str, usize> = BTreeMap::new();
    for m in &d.matches {
        *how.entry(m.how).or_default() += 1;
    }
    let map_id = save_map(core, "objects", &d.maps)?;
    let mut out = json!({
        "source": source.display().to_string(),
        "against": against.display().to_string(),
        "archive_objects": objects.len(),
        "copies": d.matches.len(),
        "matched_by": how,
        "by_class": d.by_class.iter().map(|(k, (n, m))| json!({"class": k, "objects": n, "copies": m})).collect::<Vec<_>>(),
        "first_copies": d.matches.iter().take(10).map(|m| json!({
            "copy": m.path.display().to_string(),
            "kept": m.kept.display().to_string(),
            "how": m.how,
        })).collect::<Vec<_>>(),
        "splits": splits,
        "uid_map": map_id,
        "uid_map_entries": d.maps.len(),
        "next": "remap_references with this map points what remains at the kept objects",
    });
    if a.quarantine_copies && !d.matches.is_empty() {
        let files: Vec<PathBuf> = d.matches.iter().map(|m| m.path.clone()).collect();
        let reason = if a.reason.trim().is_empty() {
            "RT objects the archive holds already".to_string()
        } else {
            a.reason.clone()
        };
        out["manifest"] = plan_quarantine(core, &files, &reason)?;
    }
    Ok(out)
}
