//! What each kind of node does.
//!
//! Every step is the program's own code path, reached the way the MCP
//! server reaches it: the loader, `autoseg::run`, `bodymask::contour_body`,
//! `registration::register`, `propagate`, and the `workflow` pipelines
//! (`group`, `anchored`, `motion`). This file only turns wire values into
//! their arguments and files what comes back in the study it belongs to -
//! the way the viewer files it, so a study a workflow worked on looks in the
//! data tree exactly as if the same steps had been clicked.
//!
//! A step that files structures records which structure set or
//! segmentation series it touched, which is what *Export DICOM* with
//! *what the run changed* writes.

use std::collections::BTreeSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};

use crate::autoseg::classes;
use crate::bodymask;
use crate::dicom_export::ExportParams;
use crate::export::{self, ExportPlan, Layout, ObjKind, StructFormat, UidMode};
use crate::loader::{self, LoadedStudy, SeriesInfo};
use crate::models::{self, Engine};
use crate::motion::MotionModel;
use crate::progress::Progress;
use crate::propagate::{self, Propagated};
use crate::registration::{self, RegMethod};
use crate::segmentation::Segmentation;
use crate::volume::Grid;
use crate::workflow::session::{self, item, NameClash};
use crate::workflow::{self, anchored, group, motion, select};
use crate::zoo;

use super::catalog::{self as cat, name_matches, split_names, Op};
use super::exec::{Ctx, Done, RegEntry, Report, RunOptions, Scope, Table, Value};
use super::{safe_name, Node};

mod more;

/// Run one node on its inputs (one list of values per input port).
pub(super) fn run_node(
    ctx: &mut Ctx,
    node: &Node,
    inputs: &[Vec<Value>],
    p: &Progress,
) -> Result<Done> {
    let one = |k: usize| -> Option<&Value> { inputs.get(k).and_then(|v| v.first()) };
    let need = |k: usize| -> Result<&Value> {
        one(k).ok_or_else(|| {
            anyhow!(
                "nothing arrived on '{}'",
                node.op.inputs().get(k).map(|s| s.name).unwrap_or("?")
            )
        })
    };
    match &node.op {
        Op::LoadFolder(p_) => load_folder(ctx, node, p_, p),
        Op::SelectImage(p_) => select_image(ctx, need(0)?, p_),
        Op::SelectGroup(p_) => select_group(ctx, need(0)?, p_),
        Op::SelectStructures(p_) => select_structures(ctx, need(0)?, p_),
        Op::AutoSegment(p_) => auto_segment(ctx, node, need(0)?, p_, p),
        Op::BodyContour(p_) => body_contour(ctx, node, need(0)?, p_, p),
        Op::Register(p_) => register(ctx, node, need(0)?, need(1)?, p_, p),
        Op::Propagate(p_) => propagate_pair(ctx, node, need(0)?, need(1)?, p_, p),
        Op::PropagateToGroup(p_) => {
            propagate_to_group(ctx, node, need(0)?, one(1), need(2)?, p_, p)
        }
        Op::Motion(p_) => motion_node(ctx, node, need(0)?, one(1), one(2), p_, p),
        Op::ExportDicom(p_) => export_dicom(ctx, &inputs[0], p_, p),
        Op::SaveReport(p_) => save_report(ctx, node, &inputs[0], p_),
        // A batch is run by the runner, a case at a time, each as a plain
        // folder step; the node itself never runs.
        Op::LoadFolders(_) => bail!("a batch runs through the run dialog, one case at a time"),
        Op::LoadFromArchive(p_) => more::load_from_archive(ctx, node, p_, p),
        Op::Anonymize(p_) => more::anonymize(ctx, node, need(0)?, p_, p),
        Op::SegVolText(p_) => more::segvol_text(ctx, node, need(0)?, p_, p),
        Op::Combine(p_) => more::combine(ctx, node, need(0)?, &inputs[1], p_, p),
        Op::Rename(p_) => more::rename(ctx, need(0)?, p_),
        Op::Transfer(p_) => more::transfer(ctx, node, need(0)?, need(1)?, need(2)?, p_, p),
        Op::CopyToPhases(p_) => more::copy_to_phases(ctx, node, need(0)?, need(1)?, p_, p),
        Op::Dvh(p_) => more::dvh(ctx, node, &inputs[0], p_, p),
        Op::DoseMetrics(p_) => more::dose_metrics(ctx, node, &inputs[0], p_, p),
        Op::ArchiveImport(p_) => more::archive_import(&mut *ctx, &inputs[0], p_, p),
        Op::Drr(p_) => more::drr(ctx, node, need(0)?, p_, p),
    }
}

pub(super) fn done(outputs: Vec<Value>, lines: Vec<String>) -> Done {
    Done {
        outputs,
        lines,
        touched: Vec::new(),
        report: None,
    }
}

/// `1 structure set`, `3 structure sets`.
pub(super) fn count(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

pub(super) fn r2(v: f64) -> String {
    format!("{v:.2}")
}

pub(super) fn r1(v: f64) -> String {
    format!("{v:.1}")
}

// ---- input -----------------------------------------------------------------

/// A folder node's folder: the run's override, else the parameter, a
/// relative one looked for beside the workflow file, in the current folder
/// and beside the program.
pub(super) fn input_folder(opts: &RunOptions, node: &Node, given: &str) -> Result<PathBuf> {
    let raw = match opts.inputs.get(&node.id) {
        Some(p) => p.clone(),
        None => PathBuf::from(given.trim()),
    };
    if raw.as_os_str().is_empty() {
        bail!("no folder is given for '{}'", node.label());
    }
    if raw.is_absolute() {
        return Ok(raw);
    }
    let mut tried = Vec::new();
    let bases = [
        opts.base_dir.clone(),
        std::env::current_dir().ok(),
        Some(crate::settings::app_dir()),
    ];
    for b in bases.into_iter().flatten() {
        let c = b.join(&raw);
        if c.is_dir() {
            return Ok(c);
        }
        tried.push(c);
    }
    Ok(tried.into_iter().next().unwrap_or(raw))
}

/// What is on disk under `dir`: every file's path, size and time, hashed.
/// A step reading the folder is not taken from an earlier run when this
/// changed.
fn folder_signature(dir: &Path) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    dir.hash(&mut h);
    for e in walkdir::WalkDir::new(dir)
        .follow_links(true)
        .sort_by_file_name()
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
    {
        e.path().hash(&mut h);
        if let Ok(m) = e.metadata() {
            m.len().hash(&mut h);
            if let Ok(t) = m.modified() {
                t.hash(&mut h);
            }
        }
    }
    h.finish()
}

/// For a step that reads from disk, a fingerprint of what it would read
/// (see `exec`'s reruns); `None` for every other step.
pub(super) fn source_signature(opts: &RunOptions, node: &Node) -> Option<u64> {
    match &node.op {
        Op::LoadFolder(p) => {
            let dir = input_folder(opts, node, &p.path).ok()?;
            Some(folder_signature(&dir))
        }
        Op::LoadFromArchive(p) => {
            let dir = more::archive_study_dir(p).ok()?;
            Some(folder_signature(&dir))
        }
        // The protocol file is read when the step runs.
        Op::Dvh(p) if !p.protocol_file.trim().is_empty() => {
            let text = std::fs::read_to_string(p.protocol_file.trim()).unwrap_or_default();
            let mut h = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut h);
            Some(h.finish())
        }
        _ => None,
    }
}

fn load_folder(ctx: &mut Ctx, node: &Node, prm: &cat::LoadFolder, p: &Progress) -> Result<Done> {
    let path = input_folder(ctx.opts, node, &prm.path)?;
    if !path.is_dir() {
        bail!("the folder {} does not exist", path.display());
    }
    p.set(format!("Reading {}", path.display()));
    let study = loader::load_directory(&path, p)?;
    Ok(study_loaded(ctx, node, path, prm.workspace, study))
}

/// A study just read, as a dataset of the run and the step's lines.
pub(super) fn study_loaded(
    ctx: &mut Ctx,
    node: &Node,
    path: PathBuf,
    workspace: cat::Workspace,
    study: LoadedStudy,
) -> Done {
    // A group's name carries its phase count already ("4DCT (10 phases)").
    let groups: Vec<String> = study
        .fourd_groups
        .iter()
        .filter(|g| !g.dissolved)
        .map(|g| g.name.clone())
        .collect();
    let mut lines = vec![
        path.display().to_string(),
        format!(
            "{} image series, {}, {}",
            study.series.len(),
            count(
                study.structure_sets.len(),
                "structure set",
                "structure sets"
            ),
            count(
                study.seg_series.len(),
                "segmentation series",
                "segmentation series"
            )
        ),
    ];
    if !groups.is_empty() {
        lines.push(format!("4D: {}", groups.join(", ")));
    }
    let focus = study.series.get(study.active_series).map(|s| s.uid.clone());
    let ds = ctx.add_dataset(node.label(), node.id, path, workspace.slot(), study);
    let mut d = done(vec![Value::Study(ds)], lines);
    d.touched.push((ds, focus));
    d
}

// ---- finding things -------------------------------------------------------

pub(super) fn study_of(ctx: &Ctx, v: &Value) -> Result<usize> {
    let ds = v
        .dataset()
        .ok_or_else(|| anyhow!("this input needs something that belongs to a study"))?;
    ctx.ds(ds)?;
    Ok(ds)
}

/// Series that are members of a live 4D group.
fn in_any_group(study: &LoadedStudy) -> std::collections::BTreeSet<String> {
    study
        .fourd_groups
        .iter()
        .filter(|g| !g.dissolved)
        .flat_map(|g| g.members.iter().map(|m| m.series_uid.clone()))
        .collect()
}

fn select_image(ctx: &mut Ctx, v: &Value, prm: &cat::SelectImage) -> Result<Done> {
    let ds = study_of(ctx, v)?;
    let d = ctx.ds(ds)?;
    let st = &d.study;
    let grouped = in_any_group(st);
    let want_mod = prm.modality.trim();
    let want_desc = prm.description.trim().to_lowercase();
    let cands: Vec<&SeriesInfo> = st
        .series
        .iter()
        .filter(|s| want_mod.is_empty() || s.modality.eq_ignore_ascii_case(want_mod))
        .filter(|s| want_desc.is_empty() || s.description.to_lowercase().contains(&want_desc))
        .filter(|s| !(prm.outside_4d && grouped.contains(&s.uid)))
        .collect();
    let Some(pick) = (match prm.pick {
        cat::ImagePick::Largest => cands.iter().copied().rev().max_by_key(|s| s.files.len()),
        cat::ImagePick::First => cands.first().copied(),
        cat::ImagePick::Last => cands.last().copied(),
    }) else {
        let have: Vec<String> = st
            .series
            .iter()
            .map(|s| format!("{} '{}'", s.modality, s.description))
            .collect();
        bail!(
            "no {} series{} in '{}'; it holds: {}",
            if want_mod.is_empty() {
                "image"
            } else {
                want_mod
            },
            if want_desc.is_empty() {
                String::new()
            } else {
                format!(" with '{}' in its description", prm.description.trim())
            },
            d.label,
            if have.is_empty() {
                "no image series".into()
            } else {
                have.join(", ")
            }
        );
    };
    let lines = vec![format!(
        "{} '{}', {} slices",
        pick.modality,
        pick.description,
        pick.files.len()
    )];
    let uid = pick.uid.clone();
    let mut out = done(
        vec![Value::Image {
            ds,
            uid: uid.clone(),
        }],
        lines,
    );
    out.touched.push((ds, Some(uid)));
    Ok(out)
}

fn select_group(ctx: &mut Ctx, v: &Value, prm: &cat::SelectGroup) -> Result<Done> {
    let ds = study_of(ctx, v)?;
    let want = prm.name.trim().to_lowercase();
    let found = {
        let st = &ctx.ds(ds)?.study;
        st.fourd_groups.iter().position(|g| {
            !g.dissolved && (want.is_empty() || g.name.to_lowercase().contains(&want))
        })
    };
    let (gi, made) = match found {
        Some(gi) => (gi, false),
        None => {
            if prm.if_none == cat::IfNoGroup::Fail {
                let st = &ctx.ds(ds)?.study;
                let names: Vec<&str> = st
                    .fourd_groups
                    .iter()
                    .filter(|g| !g.dissolved)
                    .map(|g| g.name.as_str())
                    .collect();
                bail!(
                    "no 4D group{} in '{}'{}",
                    if want.is_empty() {
                        String::new()
                    } else {
                        format!(" named like '{}'", prm.name.trim())
                    },
                    ctx.ds(ds)?.label,
                    if names.is_empty() {
                        String::new()
                    } else {
                        format!("; there are: {}", names.join(", "))
                    }
                );
            }
            let modality = prm.modality.trim().to_string();
            let d = ctx.ds_mut(ds)?;
            let picked: Vec<usize> = d
                .study
                .series
                .iter()
                .enumerate()
                .filter(|(_, s)| modality.is_empty() || s.modality.eq_ignore_ascii_case(&modality))
                .map(|(i, _)| i)
                .collect();
            if picked.len() < 2 {
                bail!(
                    "'{}' has no 4D group and fewer than two {} series to make one of",
                    d.label,
                    if modality.is_empty() {
                        "image"
                    } else {
                        &modality
                    }
                );
            }
            let g = crate::fourd::group_from(&d.study.series, &picked).ok_or_else(|| {
                anyhow!(
                    "the {modality} series of '{}' do not make a 4D group",
                    d.label
                )
            })?;
            d.study.fourd_groups.push(g);
            (d.study.fourd_groups.len() - 1, true)
        }
    };
    let st = &ctx.ds(ds)?.study;
    let g = &st.fourd_groups[gi];
    let phases = workflow::phases_of(g, &st.series)?;
    let labels: Vec<&str> = phases.iter().map(|(l, _)| l.as_str()).collect();
    let mut lines = vec![format!("'{}': {} phases", g.name, phases.len())];
    lines.push(labels.join(", "));
    if made {
        lines.push("made from the series, as none was recognised".into());
    }
    let focus = phases.first().map(|(_, s)| s.uid.clone());
    let mut out = done(vec![Value::Group { ds, group: gi }], lines);
    out.touched.push((ds, focus));
    Ok(out)
}

/// The phases of group `gi` of a study as (label, series).
pub(super) fn phases(ctx: &Ctx, ds: usize, gi: usize) -> Result<Vec<(String, SeriesInfo)>> {
    let st = &ctx.ds(ds)?.study;
    let g = st
        .fourd_groups
        .get(gi)
        .ok_or_else(|| anyhow!("the 4D group is no longer there"))?;
    workflow::phases_of(g, &st.series)
}

/// Names of the structures drawn on one image series: in the structure sets
/// and segmentation series that reference it; failing any, every name of
/// the study (a set that names no series still counts).
pub(super) fn names_on_image(study: &LoadedStudy, uid: &str) -> Vec<String> {
    let mut bound = Vec::new();
    for ss in study
        .structure_sets
        .iter()
        .filter(|s| s.referenced_series_uid == uid)
    {
        bound.extend(ss.rois.iter().map(|r| r.name.clone()));
    }
    for sr in study
        .seg_series
        .iter()
        .filter(|s| s.referenced_series_uid == uid)
    {
        bound.extend(sr.segs.iter().map(|s| s.name.clone()));
    }
    if bound.is_empty() {
        bound = select::list(study).into_iter().map(|e| e.name).collect();
    }
    bound
}

fn select_structures(ctx: &mut Ctx, v: &Value, prm: &cat::SelectStructures) -> Result<Done> {
    let ds = study_of(ctx, v)?;
    let d = ctx.ds(ds)?;
    let st = &d.study;
    let (candidates, on, place) = match v {
        Value::Image { uid, .. } => {
            let desc = st
                .series
                .iter()
                .find(|s| &s.uid == uid)
                .map(|s| format!("'{}'", s.description))
                .unwrap_or_else(|| "the image".into());
            (names_on_image(st, uid), Scope::Image(uid.clone()), desc)
        }
        Value::Group { group, .. }
        | Value::Structures {
            on: Scope::Group(group),
            ..
        } => {
            // A name counts when every phase has it.
            let ph = phases(ctx, ds, *group)?;
            let mut common: Option<Vec<String>> = None;
            for (_, se) in &ph {
                let here = names_on_image(st, &se.uid);
                common = Some(match common {
                    None => here,
                    Some(c) => c
                        .into_iter()
                        .filter(|n| here.iter().any(|h| h.eq_ignore_ascii_case(n)))
                        .collect(),
                });
            }
            let gname = st.fourd_groups[*group].name.clone();
            (
                common.unwrap_or_default(),
                Scope::Group(*group),
                format!("every phase of '{gname}'"),
            )
        }
        _ => (
            select::list(st).into_iter().map(|e| e.name).collect(),
            Scope::Study,
            format!("'{}'", d.label),
        ),
    };
    let patterns = split_names(&prm.names);
    let mut found: Vec<String> = Vec::new();
    'outer: for pat in &patterns {
        for name in &candidates {
            if name_matches(pat, name) && !found.iter().any(|f| f.eq_ignore_ascii_case(name)) {
                found.push(name.clone());
                if prm.first_only {
                    break 'outer;
                }
            }
        }
    }
    if found.is_empty() && prm.required {
        let mut known: Vec<String> = candidates.clone();
        known.dedup();
        bail!(
            "no structure matching {} on {place}; there are: {}",
            patterns.join(", "),
            if known.is_empty() {
                "none".into()
            } else {
                known.join(", ")
            }
        );
    }
    let lines = vec![if found.is_empty() {
        "nothing matched".to_string()
    } else {
        found.join(", ")
    }];
    Ok(done(
        vec![Value::Structures {
            ds,
            names: found,
            on,
        }],
        lines,
    ))
}

// ---- filing ------------------------------------------------------------------

/// How a step files what it made: where, and what happens to a name that
/// is taken.
#[derive(Clone, Copy)]
pub(super) struct FileAs<'a> {
    pub kind: cat::OutputKind,
    pub set: cat::SetChoice,
    pub set_label: &'a str,
    pub clash: NameClash,
}

impl<'a> FileAs<'a> {
    pub fn new(
        kind: cat::OutputKind,
        set: cat::SetChoice,
        set_label: &'a str,
        clash: NameClash,
    ) -> Self {
        FileAs {
            kind,
            set,
            set_label,
            clash,
        }
    }

    /// Where a step that carries structures onto an image files them:
    /// that image's own set, or a segmentation series.
    pub fn landing(landing: cat::Landing, set_label: &'a str, clash: NameClash) -> Self {
        FileAs {
            kind: match landing {
                cat::Landing::StructureSet => cat::OutputKind::Structures,
                cat::Landing::Segmentation => cat::OutputKind::Segments,
            },
            set: cat::SetChoice::Own,
            set_label,
            clash,
        }
    }
}

/// Every name the filings of one step on these images would meet: the
/// same name on every phase of a 4D group comes from here.
pub(super) fn taken_on(
    ctx: &Ctx,
    ds: usize,
    series: &[&SeriesInfo],
    how: &FileAs,
) -> Result<BTreeSet<String>> {
    let st = &ctx.ds(ds)?.study;
    Ok(session::taken_names(
        st,
        series.iter().copied(),
        how.kind,
        how.set,
    ))
}

/// File masks made on one image series of study `ds` - as contours in its
/// structure set (or a new one), as segments of its segmentation series,
/// or both - the way the viewer's tools file them ([`session::file_items`]).
/// `taken`: the names every image of the step's filing meets (a 4D group's
/// phases); the name each item lands under is then the same on all of
/// them. Answers item by item, in the order of `items`: the name it landed
/// under, `None` where nothing was filed (an empty mask).
#[allow(clippy::too_many_arguments)]
pub(super) fn land(
    ctx: &mut Ctx,
    ds: usize,
    series: &SeriesInfo,
    grid: &Grid,
    items: Vec<Propagated>,
    how: &FileAs,
    roi_types: &[String],
    taken: Option<&BTreeSet<String>>,
) -> Result<Vec<Option<String>>> {
    let d = ctx.ds_mut(ds)?;
    let filed = session::file_items(
        &mut d.study,
        series,
        grid,
        &items,
        roi_types,
        &session::Filing {
            kind: how.kind,
            set: how.set,
            set_label: how.set_label,
            clash: Some(how.clash),
            taken,
        },
    )?;
    if let Some(i) = filed.set {
        d.touched_sets.insert(i);
    }
    if let Some(i) = filed.seg_series {
        d.touched_segs.insert(i);
    }
    Ok(filed.names)
}

/// The RT ROI Interpreted Type of a named structure on an image, when it is
/// an RT structure; empty otherwise.
pub(super) fn roi_type_of(study: &LoadedStudy, name: &str, uid: &str) -> String {
    let lower = name.to_lowercase();
    let sets = study
        .structure_sets
        .iter()
        .filter(|s| s.referenced_series_uid == uid)
        .chain(study.structure_sets.iter());
    for ss in sets {
        if let Some(r) = ss
            .rois
            .iter()
            .find(|r| r.name == name || r.name.to_lowercase() == lower)
        {
            return r.roi_type.clone();
        }
    }
    String::new()
}

// ---- segmentation --------------------------------------------------------------

/// The images a segmentation step runs on: one image series, or every phase
/// of a group; with the scope its result is on.
/// A study, the (label, series) it runs on, and the scope of the result.
pub(super) type Targets = (usize, Vec<(String, SeriesInfo)>, Scope);

pub(super) fn engine_targets(ctx: &Ctx, v: &Value) -> Result<Targets> {
    match v {
        Value::Image { ds, uid } => {
            let st = &ctx.ds(*ds)?.study;
            let se = st
                .series
                .iter()
                .find(|s| &s.uid == uid)
                .cloned()
                .ok_or_else(|| anyhow!("the image series is no longer in the study"))?;
            Ok((*ds, vec![(String::new(), se)], Scope::Image(uid.clone())))
        }
        Value::Group { ds, group } => Ok((*ds, phases(ctx, *ds, *group)?, Scope::Group(*group))),
        _ => bail!("this step runs on an image series or a 4D group"),
    }
}

/// One volume's masks, named and coloured, with their interpreted types.
pub(super) struct Made {
    pub items: Vec<Propagated>,
    pub roi_types: Vec<String>,
    /// Report rows: (structure, cm³).
    pub rows: Vec<(String, f64)>,
    pub notes: Vec<String>,
}

/// Run an engine on every target image and file what it makes; the shared
/// body of the two segmentation nodes.
pub(super) fn engine_node(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    how: FileAs,
    title: &str,
    p: &Progress,
    mut engine: impl FnMut(&crate::volume::Volume, &Progress) -> Result<Made>,
) -> Result<Done> {
    let (ds, targets, scope) = engine_targets(ctx, v)?;
    let n = targets.len().max(1);
    let series: Vec<&SeriesInfo> = targets.iter().map(|(_, s)| s).collect();
    let taken = taken_on(ctx, ds, &series, &how)?;
    let mut table = Table::new(
        "Structures",
        &["Image", "Structure", "Filed as", "Volume (cm3)"],
    );
    let mut first_names: Option<Vec<String>> = None;
    let mut notes = Vec::new();
    let mut last_uid = None;
    for (i, (label, series)) in targets.iter().enumerate() {
        if p.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        p.set_outer(i as f32 / n as f32, 1.0 / n as f32);
        if n > 1 {
            p.set_prefix(format!("Phase {label} ({}/{n}): ", i + 1));
        }
        p.set("loading");
        let vol = ctx.volume(ds, &series.uid, p)?;
        let made = engine(&vol, p).with_context(|| {
            if label.is_empty() {
                format!("'{}'", series.description)
            } else {
                format!("phase {label}")
            }
        })?;
        let grid = vol.grid();
        // A set made on a phase is named after the phase too, so the ten
        // sets of a 4D group are told apart in the tree and on disk.
        let label_here = if label.is_empty() {
            how.set_label.to_string()
        } else {
            format!("{} {label}", how.set_label)
        };
        let landed = land(
            ctx,
            ds,
            series,
            &grid,
            made.items,
            &FileAs {
                set_label: &label_here,
                ..how
            },
            &made.roi_types,
            Some(&taken),
        )?;
        let image = if label.is_empty() {
            series.description.clone()
        } else {
            label.clone()
        };
        for (k, (name, cm3)) in made.rows.iter().enumerate() {
            table.row(vec![
                image.clone(),
                name.clone(),
                landed
                    .get(k)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| "(empty)".into()),
                r2(*cm3),
            ]);
        }
        notes.extend(made.notes);
        if first_names.is_none() {
            first_names = Some(landed.iter().flatten().cloned().collect());
        }
        last_uid = Some(series.uid.clone());
        // Show each phase as it is done: the viewer steps through the
        // phases the way the run does.
        if n > 1 && i + 1 < n {
            ctx.show(node.id, &[(ds, Some(series.uid.clone()))], None, p)?;
        }
    }
    p.set_prefix("");
    p.set_outer(0.0, 1.0);
    // The names downstream steps look up: what they were filed as - the
    // same on every phase of a group, a counter included (see `land`).
    let names: Vec<String> = first_names.unwrap_or_default();
    let mut lines = vec![format!(
        "{} on {} image{}",
        if names.is_empty() {
            "nothing filed".to_string()
        } else {
            names.join(", ")
        },
        targets.len(),
        if targets.len() == 1 { "" } else { "s" }
    )];
    lines.extend(notes.iter().take(3).cloned());
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("{title}: {}", node.label()),
        notes,
        tables: vec![table],
        motion: None,
    });
    let mut out = done(
        vec![
            Value::Structures {
                ds,
                names,
                on: scope,
            },
            Value::Report(report),
        ],
        lines,
    );
    out.touched.push((ds, last_uid));
    out.report = Some(report);
    Ok(out)
}

/// An engine's folder under the run's model root.
pub(super) fn models_root(ctx: &Ctx, engine: Engine) -> PathBuf {
    models::engine_dir(&ctx.opts.models_dir, engine)
}

fn auto_segment(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::AutoSegment,
    p: &Progress,
) -> Result<Done> {
    let model = prm
        .auto_model()
        .ok_or_else(|| anyhow!("'{}' is not a model this program has", prm.model.trim()))?;
    let wanted: Vec<(u8, String)> = prm
        .organs
        .iter()
        .map(|o| {
            prm.label_of(&o.organ)
                .map(|l| {
                    let name = if o.name.trim().is_empty() {
                        prm.default_name(model.classes()[l as usize - 1])
                    } else {
                        o.landed_name()
                    };
                    (l, name)
                })
                .ok_or_else(|| anyhow!("'{}' is not a class of {}", o.organ.trim(), model.label()))
        })
        .collect::<Result<_>>()?;
    // A model in parts runs only the parts holding an organ asked for.
    let labels: Vec<u8> = wanted.iter().map(|(l, _)| *l).collect();
    let parts = model.parts_holding(&labels);
    let root = ctx.opts.models_dir.clone();
    let need = model.download_needed(parts.as_deref(), &root);
    if need > 0 && !ctx.opts.allow_download {
        bail!(
            "the {} weights are not present ({} to download) and downloads are off for this run",
            model.label(),
            models::human_bytes(need)
        );
    }
    let opts = zoo::RunOptions {
        device: prm.device.pref(),
        parts,
    };
    let tg263 = prm.tg263;
    let run = move |vol: &crate::volume::Volume, p: &Progress| -> Result<Made> {
        let r = model.run(vol, &opts, &root, p)?;
        let name_of = |name: &str| -> String {
            if tg263 {
                zoo::tg263_name(name).unwrap_or_else(|| name.to_string())
            } else {
                name.to_string()
            }
        };
        let classes: Vec<(u8, String, [u8; 3])> = if wanted.is_empty() {
            r.organs
                .iter()
                .map(|o| (o.label, name_of(o.name), o.color))
                .collect()
        } else {
            let missing: Vec<&str> = wanted
                .iter()
                .filter(|(l, _)| !r.organs.iter().any(|o| o.label == *l))
                .map(|(l, _)| r.class_name(*l))
                .collect();
            if !missing.is_empty() {
                bail!("{} was not found", missing.join(", "));
            }
            wanted
                .iter()
                .map(|(l, name)| (*l, name.clone(), classes::color_of(r.class_name(*l), *l)))
                .collect()
        };
        let segs = Segmentation::from_label_map_many(r.dims, &r.labels, &classes);
        let spacing = vol.spacing;
        let mut made = Made {
            items: Vec::new(),
            roi_types: Vec::new(),
            rows: Vec::new(),
            notes: Vec::new(),
        };
        for s in segs {
            made.rows.push((s.name.clone(), s.volume_cm3(spacing)));
            made.roi_types.push("ORGAN".into());
            made.items.push(item(s.name, s.color, s.mask));
        }
        Ok(made)
    };
    engine_node(
        ctx,
        node,
        v,
        FileAs::new(prm.output, prm.set, &prm.set_label, prm.names),
        "Auto-segmentation",
        p,
        run,
    )
}

fn body_contour(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::BodyContour,
    p: &Progress,
) -> Result<Done> {
    let dir = ctx.opts.models_dir.clone();
    let (_, targets, _) = engine_targets(ctx, v)?;
    let modality = targets
        .first()
        .map(|(_, s)| s.modality.clone())
        .unwrap_or_else(|| "CT".into());
    let mut params = bodymask::BodyParams::for_modality(&modality);
    params.method = prm.method.method();
    params.device = prm.device.pref();
    if params.method == bodymask::Method::ModelAssisted {
        let need = bodymask::download_needed(params.model, &dir);
        if need > 0 && !ctx.opts.allow_download {
            bail!(
                "the body-outline weights are not present ({} to download) and downloads are \
                 off for this run",
                models::human_bytes(need)
            );
        }
    }
    let name = prm.name.trim().to_string();
    let run = move |vol: &crate::volume::Volume, p: &Progress| -> Result<Made> {
        let r = bodymask::contour_body(vol, &params, &dir, p)?;
        Ok(Made {
            rows: vec![(name.clone(), r.cm3)],
            roi_types: vec!["EXTERNAL".into()],
            items: vec![item(name.clone(), [0, 200, 0], r.mask)],
            notes: Vec::new(),
        })
    };
    engine_node(
        ctx,
        node,
        v,
        FileAs::new(prm.output, prm.set, &prm.set_label, prm.names),
        "Body contour",
        p,
        run,
    )
}

// ---- registration --------------------------------------------------------------

pub(super) fn image_of(v: &Value) -> Result<(usize, String)> {
    match v {
        Value::Image { ds, uid } => Ok((*ds, uid.clone())),
        Value::Structures {
            ds,
            on: Scope::Image(uid),
            ..
        } => Ok((*ds, uid.clone())),
        _ => bail!("this input needs one image series"),
    }
}

fn register(
    ctx: &mut Ctx,
    node: &Node,
    fixed: &Value,
    moving: &Value,
    prm: &cat::Register,
    p: &Progress,
) -> Result<Done> {
    let f = image_of(fixed)?;
    let m = image_of(moving)?;
    if f == m {
        bail!("the fixed and the moving image are the same series");
    }
    let fv = ctx.volume(f.0, &f.1, p)?;
    let mv = ctx.volume(m.0, &m.1, p)?;
    let mut params = prm.effort.params(prm.method.method());
    params.init = prm.init.init();
    p.set("Registering");
    let r = registration::register(&fv, &mv, &params, p)?;
    let a = &r.analysis;
    let mut t = Table::new("Registration", &["Quantity", "Value"]);
    t.row(vec!["Method".into(), r.method.label().into()]);
    t.row(vec![
        format!("Metric ({})", r.metric.tag()),
        format!("{} to {}", r2(r.initial_metric), r2(r.final_metric)),
    ]);
    t.row(vec!["Iterations".into(), r.iterations_run.to_string()]);
    t.row(vec!["Time (s)".into(), r1(r.elapsed_secs)]);
    t.row(vec![
        "Translation (mm)".into(),
        format!(
            "{} {} {}",
            r2(a.dof.translation.x),
            r2(a.dof.translation.y),
            r2(a.dof.translation.z)
        ),
    ]);
    t.row(vec![
        "Rotation (deg)".into(),
        a.dof
            .rotation_deg
            .iter()
            .map(|v| r2(*v))
            .collect::<Vec<_>>()
            .join(" "),
    ]);
    t.row(vec![
        "Displacement p95 / max (mm)".into(),
        format!("{} / {}", r2(a.displacement.p95), r2(a.displacement.max)),
    ]);
    t.row(vec![
        "Folded fraction".into(),
        format!("{:.4}", a.jacobian.folded),
    ]);
    let line = r.metric_line();
    let reg = ctx.add_reg(RegEntry::Pair {
        fixed: f.clone(),
        moving: m,
        result: Arc::new(r),
    });
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Registration: {}", node.label()),
        notes: vec![line.clone()],
        tables: vec![t],
        motion: None,
    });
    let mut out = done(
        vec![Value::Registration(reg), Value::Report(report)],
        vec![line],
    );
    out.report = Some(report);
    out.touched.push((f.0, Some(f.1)));
    Ok(out)
}

/// The structures a value names, on the image `uid` of study `ds`, as
/// subjects on `grid`.
pub(super) fn subjects_on(
    study: &LoadedStudy,
    names: &[String],
    uid: &str,
    grid: &Grid,
) -> Result<Vec<propagate::Subject>> {
    let mut out = Vec::new();
    for n in names {
        let s = select::find_on_series(study, n, uid, &grid.frame_of_reference_uid)
            .ok_or_else(|| anyhow!("no structure '{n}' on the source image"))?;
        out.push(s.subject_on(grid)?);
    }
    Ok(out)
}

fn propagate_pair(
    ctx: &mut Ctx,
    node: &Node,
    reg: &Value,
    structs: &Value,
    prm: &cat::Propagate,
    p: &Progress,
) -> Result<Done> {
    let Value::Registration(ri) = reg else {
        bail!("the first input needs a registration");
    };
    let Some(RegEntry::Pair {
        fixed,
        moving,
        result,
    }) = ctx.regs.get(*ri)
    else {
        bail!("this step needs a registration of two images, not one per phase");
    };
    let (fixed, moving, transform) = (fixed.clone(), moving.clone(), result.transform.clone());
    let Value::Structures { ds, names, on } = structs else {
        bail!("the second input needs structures");
    };
    if names.is_empty() {
        bail!("no structures arrived to carry");
    }
    // Which side they are on decides the direction: the transform maps
    // fixed to moving, so landing on the moving side runs its inverse.
    let from_moving = match on {
        Scope::Image(uid) => {
            if (*ds, uid.clone()) == moving {
                true
            } else if (*ds, uid.clone()) == fixed {
                false
            } else {
                bail!("the structures are on neither image of the registration");
            }
        }
        _ => {
            if *ds == moving.0 && *ds != fixed.0 {
                true
            } else if *ds == fixed.0 && *ds != moving.0 {
                false
            } else {
                bail!("say which image the structures are on (find them on an image series)");
            }
        }
    };
    let (src, dst, use_inverse) = if from_moving {
        (moving, fixed, false)
    } else {
        (fixed, moving, true)
    };
    let src_vol = ctx.volume(src.0, &src.1, p)?;
    let dst_vol = ctx.volume(dst.0, &dst.1, p)?;
    let src_grid = src_vol.grid();
    let dst_grid = dst_vol.grid();
    let (mut subjects, types) = {
        let st = &ctx.ds(src.0)?.study;
        let subjects = subjects_on(st, names, &src.1, &src_grid)?;
        let types: Vec<String> = names.iter().map(|n| roi_type_of(st, n, &src.1)).collect();
        (subjects, types)
    };
    let finish = prm.finish.finish();
    finish.carry(&mut subjects);
    p.set("Propagating");
    let mut items =
        propagate::propagate(&src_vol, &dst_vol, &transform, use_inverse, &subjects, p)?;
    finish.apply_all(&mut items, &dst_grid, p);
    let mut t = Table::new(
        "Propagated structures",
        &["Structure", "Source (cm3)", "Result (cm3)", "Change (%)"],
    );
    for it in &mut items {
        t.row(vec![
            it.name.clone(),
            r2(it.source_cm3),
            r2(it.result_cm3),
            r1(if it.source_cm3 > 1e-9 {
                100.0 * (it.result_cm3 - it.source_cm3) / it.source_cm3
            } else {
                0.0
            }),
        ]);
        if !prm.suffix.trim().is_empty() {
            it.name = format!("{} {}", it.name, prm.suffix.trim());
        }
    }
    let dst_series = ctx
        .ds(dst.0)?
        .study
        .series
        .iter()
        .find(|s| s.uid == dst.1)
        .cloned()
        .ok_or_else(|| anyhow!("the destination series is gone"))?;
    let landed = land(
        ctx,
        dst.0,
        &dst_series,
        &dst_grid,
        items,
        &FileAs::landing(prm.landing, "Propagated", prm.names),
        &types,
        None,
    )?;
    let names: Vec<String> = landed.iter().flatten().cloned().collect();
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Propagation: {}", node.label()),
        notes: Vec::new(),
        tables: vec![t],
        motion: None,
    });
    let mut out = done(
        vec![
            Value::Structures {
                ds: dst.0,
                names: names.clone(),
                on: Scope::Image(dst.1.clone()),
            },
            Value::Report(report),
        ],
        vec![format!("{} landed", names.join(", "))],
    );
    out.report = Some(report);
    out.touched.push((dst.0, Some(dst.1)));
    Ok(out)
}

/// The 4D group a value points at: a group, or structures made on one.
pub(super) fn group_of(v: &Value) -> Result<(usize, usize)> {
    match v {
        Value::Group { ds, group } => Ok((*ds, *group)),
        Value::Structures {
            ds,
            on: Scope::Group(g),
            ..
        } => Ok((*ds, *g)),
        _ => bail!("this input needs a 4D group (or structures made on every phase of one)"),
    }
}

fn propagate_to_group(
    ctx: &mut Ctx,
    node: &Node,
    structs: &Value,
    anchor: Option<&Value>,
    onto: &Value,
    prm: &cat::PropagateToGroup,
    p: &Progress,
) -> Result<Done> {
    let Value::Structures {
        ds: src_ds,
        names,
        on,
    } = structs
    else {
        bail!("the first input needs structures");
    };
    let src_uid = match on {
        Scope::Image(u) => u.clone(),
        _ => bail!(
            "the structures to carry must be found on one image series (connect the image to \
             'Find structures')"
        ),
    };
    let anchor_name = match anchor {
        None => None,
        Some(Value::Structures {
            ds, names: an, on, ..
        }) => {
            if *ds != *src_ds || matches!(on, Scope::Image(u) if *u != src_uid) {
                bail!("the anchor must be on the same image series as the structures");
            }
            Some(
                an.first()
                    .cloned()
                    .ok_or_else(|| anyhow!("the anchor input names no structure"))?,
            )
        }
        Some(_) => bail!("the anchor input needs structures"),
    };
    let (dst_ds, gi) = group_of(onto)?;
    let ph = phases(ctx, dst_ds, gi)?;
    let group_name = ctx.ds(dst_ds)?.study.fourd_groups[gi].name.clone();
    let src_vol = ctx.volume(*src_ds, &src_uid, p)?;
    let src_grid = src_vol.grid();
    let (subjects, types, src_anchor, anchor_type) = {
        let st = &ctx.ds(*src_ds)?.study;
        let subjects = subjects_on(st, names, &src_uid, &src_grid)?;
        let types: Vec<String> = names.iter().map(|n| roi_type_of(st, n, &src_uid)).collect();
        let src_anchor = match &anchor_name {
            Some(a) => Some(
                select::find_on_series(st, a, &src_uid, &src_grid.frame_of_reference_uid)
                    .ok_or_else(|| anyhow!("no anchor '{a}' on the source image"))?,
            ),
            None => None,
        };
        let anchor_type = anchor_name
            .as_deref()
            .map(|a| roi_type_of(st, a, &src_uid))
            .unwrap_or_default();
        (subjects, types, src_anchor, anchor_type)
    };
    let base = prm.effort.params(prm.method.method());
    let finish = prm.finish.finish();
    let n_subjects = subjects.len();
    let (group_out, qa) = match (&anchor_name, src_anchor) {
        (Some(a), Some(src_anchor)) => {
            let mut anchored_phases = Vec::with_capacity(ph.len());
            {
                let st = &ctx.ds(dst_ds)?.study;
                for (label, series) in &ph {
                    let on_phase =
                        select::find_on_series(st, a, &series.uid, "").ok_or_else(|| {
                            anyhow!(
                            "no '{a}' on phase {label}; an anchored run needs it on every phase \
                             (segment it there first)"
                        )
                        })?;
                    anchored_phases.push(anchored::AnchoredPhase {
                        label: label.clone(),
                        series: series.clone(),
                        anchor: on_phase,
                    });
                }
            }
            let req = anchored::AnchoredRequest {
                src_vol: src_vol.clone(),
                src_anchor,
                anchor_landed_name: (!prm.anchor_landed_as.trim().is_empty())
                    .then(|| prm.anchor_landed_as.trim().to_string()),
                anchor_landed_color: None,
                subjects,
                phases: anchored_phases,
                margin_mm: prm.anchor_margin_mm.max(0.0),
                mode: match prm.anchor_by {
                    cat::AnchorBy::Contours => anchored::AnchorMode::Contours,
                    cat::AnchorBy::Intensity => anchored::AnchorMode::Intensity,
                },
                rigid: anchored::default_rigid(&base),
                deformable: (!prm.rigid_only).then(|| anchored::default_deformable(&base)),
                finish,
                group_name: group_name.clone(),
                group: gi,
                moving_slot: 0,
                moving_series_uid: src_uid.clone(),
                volumes: ctx.volumes.clone(),
            };
            let out = anchored::run(req, p)?;
            (out.group, Some(out.qa))
        }
        _ => {
            if subjects.is_empty() {
                bail!("nothing to carry, and no anchor");
            }
            let req = group::GroupRequest {
                src_vol: src_vol.clone(),
                subjects,
                cached: vec![None; ph.len()],
                phases: ph.clone(),
                params: base,
                finish,
                group_name: group_name.clone(),
                group: gi,
                moving_slot: 0,
                moving_series_uid: src_uid.clone(),
                volumes: ctx.volumes.clone(),
            };
            (group::run(req, p)?, None)
        }
    };

    // File every phase's structures on the phase.
    let mut reg_table = Table::new(
        "Registration per phase",
        &[
            "Phase",
            "Registration",
            "Anchor Dice",
            "Anchor HD95 (mm)",
            "Anchor centroids apart (mm)",
            "Verdict",
        ],
    );
    let mut vol_table = Table::new(
        "Structures per phase",
        &[
            "Phase",
            "Structure",
            "Filed as",
            "Source (cm3)",
            "Result (cm3)",
            "Change (%)",
        ],
    );
    let mut first_names: Option<Vec<String>> = None;
    let mut transforms = Vec::new();
    let how = FileAs::landing(prm.landing, "", prm.names);
    let taken = {
        let series: Vec<&SeriesInfo> = ph.iter().map(|(_, s)| s).collect();
        taken_on(ctx, dst_ds, &series, &how)?
    };
    for (k, phase) in group_out.phases.iter().enumerate() {
        let series = ph
            .iter()
            .find(|(_, s)| s.uid == phase.series_uid)
            .map(|(_, s)| s.clone())
            .ok_or_else(|| anyhow!("phase {} is gone", phase.label))?;
        let item_types: Vec<String> = phase
            .items
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i < n_subjects {
                    types.get(i).cloned().unwrap_or_default()
                } else {
                    anchor_type.clone()
                }
            })
            .collect();
        let set_label = format!("{group_name} {}", phase.label);
        let landed = land(
            ctx,
            dst_ds,
            &series,
            &phase.grid,
            phase.unpacked(),
            &FileAs {
                set_label: &set_label,
                ..how
            },
            &item_types,
            Some(&taken),
        )?;
        for (i, it) in phase.items.iter().enumerate() {
            vol_table.row(vec![
                phase.label.clone(),
                it.name.clone(),
                landed.get(i).cloned().flatten().unwrap_or_default(),
                r2(it.source_cm3),
                r2(it.result_cm3),
                r1(if it.source_cm3 > 1e-9 {
                    100.0 * (it.result_cm3 - it.source_cm3) / it.source_cm3
                } else {
                    0.0
                }),
            ]);
        }
        if first_names.is_none() {
            first_names = Some(landed.iter().take(n_subjects).flatten().cloned().collect());
        }
        let q = qa.as_ref().and_then(|q| q.get(k));
        let ov = q.and_then(|q| q.overlap.as_ref());
        reg_table.row(vec![
            phase.label.clone(),
            phase.metric_line.clone(),
            ov.map(|o| format!("{:.3}", o.dice)).unwrap_or_default(),
            ov.map(|o| r2(o.hd95_mm)).unwrap_or_default(),
            ov.and_then(|o| o.centroid_shift())
                .map(|d| r2(d.length()))
                .unwrap_or_default(),
            q.map(|q| q.verdict().to_string()).unwrap_or_default(),
        ]);
        transforms.push((
            phase.label.clone(),
            phase.series_uid.clone(),
            phase.transform.clone(),
        ));
    }
    let reg = ctx.add_reg(RegEntry::Group {
        ds: dst_ds,
        group: gi,
        moving: (*src_ds, src_uid.clone()),
        phases: transforms,
    });
    let worst = qa
        .as_ref()
        .map(|q| {
            q.iter()
                .filter_map(|x| x.overlap.as_ref().map(|o| o.dice))
                .fold(f64::NAN, f64::min)
        })
        .filter(|d| !d.is_nan());
    let mut notes = Vec::new();
    if let Some(a) = &anchor_name {
        notes.push(format!(
            "Anchored on '{a}' ({}), {}.",
            prm.anchor_by.label(),
            if prm.rigid_only {
                "rigid only"
            } else {
                "rigid then local deformable"
            }
        ));
    }
    if let Some(w) = worst {
        notes.push(format!("Worst anchor Dice over the phases: {w:.3}."));
    }
    let names: Vec<String> = first_names.unwrap_or_default();
    let mut lines = vec![format!(
        "{} onto {} phases of '{group_name}'",
        if names.is_empty() {
            "nothing".to_string()
        } else {
            names.join(", ")
        },
        group_out.phases.len()
    )];
    lines.extend(notes.iter().cloned());
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Propagation onto the 4D group: {}", node.label()),
        notes,
        tables: vec![reg_table, vol_table],
        motion: None,
    });
    let mut out = done(
        vec![
            Value::Structures {
                ds: dst_ds,
                names,
                on: Scope::Group(gi),
            },
            Value::Registration(reg),
            Value::Report(report),
        ],
        lines,
    );
    out.report = Some(report);
    let last = group_out.phases.last().map(|ph| ph.series_uid.clone());
    out.touched.push((dst_ds, last));
    Ok(out)
}

// ---- 4D motion -------------------------------------------------------------

fn motion_node(
    ctx: &mut Ctx,
    node: &Node,
    targets: &Value,
    reference: Option<&Value>,
    group_in: Option<&Value>,
    prm: &cat::Motion,
    p: &Progress,
) -> Result<Done> {
    let Value::Structures {
        ds,
        names: target_names,
        on,
    } = targets
    else {
        bail!("the first input needs structures");
    };
    if target_names.is_empty() {
        bail!("no target structures arrived");
    }
    let (ds, gi) = match group_in {
        Some(g) => group_of(g)?,
        None => match on {
            Scope::Group(g) => (*ds, *g),
            Scope::Image(uid) => {
                let st = &ctx.ds(*ds)?.study;
                let g = st
                    .fourd_groups
                    .iter()
                    .position(|g| !g.dissolved && g.members.iter().any(|m| &m.series_uid == uid))
                    .ok_or_else(|| {
                        anyhow!("the targets' image is in no 4D group; connect the group")
                    })?;
                (*ds, g)
            }
            Scope::Study => bail!("connect the 4D group, or find the targets on it"),
        },
    };
    let ref_name = match reference {
        None => None,
        Some(Value::Structures { names, .. }) => names.first().cloned(),
        Some(_) => bail!("the reference input needs structures"),
    };
    let (all_phases, group_name, study_uid, default_ref) = {
        let st = &ctx.ds(ds)?.study;
        let g = &st.fourd_groups[gi];
        let ph = workflow::phases_of(g, &st.series)?;
        let default_ref = g
            .default_reference()
            .and_then(|m| g.phase_members().iter().position(|&x| x == m))
            .unwrap_or(0);
        (ph, g.name.clone(), g.study_uid.clone(), default_ref)
    };
    let reference_idx = if prm.reference_phase.trim().is_empty() {
        default_ref
    } else {
        all_phases
            .iter()
            .position(|(l, _)| l.eq_ignore_ascii_case(prm.reference_phase.trim()))
            .ok_or_else(|| {
                anyhow!(
                    "no phase '{}' in '{group_name}'; the phases are {}",
                    prm.reference_phase.trim(),
                    all_phases
                        .iter()
                        .map(|(l, _)| l.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?
    };
    // The phase subset, the reference always in.
    let wanted = split_names(&prm.phases);
    let (phases_used, reference_idx) = if wanted.is_empty() {
        (all_phases.clone(), reference_idx)
    } else {
        let kept: Vec<usize> = (0..all_phases.len())
            .filter(|&i| {
                i == reference_idx
                    || wanted
                        .iter()
                        .any(|w| w.eq_ignore_ascii_case(&all_phases[i].0))
            })
            .collect();
        if kept.len() < 2 {
            bail!("select at least one phase besides the reference");
        }
        let r = kept.iter().position(|&i| i == reference_idx).unwrap_or(0);
        (kept.into_iter().map(|i| all_phases[i].clone()).collect(), r)
    };
    // Each structure: on every phase (then also read as contoured), else on
    // the reference phase alone.
    let ref_uid = phases_used[reference_idx].1.uid.clone();
    let mut contoured = Vec::new();
    let resolve =
        |name: &str, contoured: &mut Vec<motion::ContouredTarget>| -> Result<select::Structure> {
            let st = &ctx.ds(ds)?.study;
            let per_phase: Vec<Option<select::Structure>> = phases_used
                .iter()
                .map(|(_, se)| select::find_on_series(st, name, &se.uid, ""))
                .collect();
            if per_phase.iter().all(Option::is_some) {
                let mut per_phase = per_phase;
                let on_ref = per_phase[reference_idx].take().expect("checked");
                contoured.push(motion::ContouredTarget {
                    name: on_ref.name.clone(),
                    color: on_ref.color,
                    phases: phases_used
                        .iter()
                        .map(|(_, se)| select::find_on_series(st, name, &se.uid, ""))
                        .collect(),
                });
                Ok(on_ref)
            } else {
                select::find_on_series(st, name, &ref_uid, "")
                    .ok_or_else(|| anyhow!("no structure '{name}' on the reference phase"))
            }
        };
    let mut target_structs = Vec::new();
    for n in target_names {
        target_structs.push(resolve(n, &mut contoured)?);
    }
    let ref_struct = match &ref_name {
        Some(n) => Some(resolve(n, &mut contoured)?),
        None => None,
    };
    let mut models = Vec::new();
    if prm.rigid {
        models.push(MotionModel::Rigid);
    }
    if prm.deformable {
        models.push(MotionModel::Deformable);
    }
    if prm.contoured
        && contoured
            .iter()
            .any(|c| c.phases.iter().all(Option::is_some))
    {
        models.push(MotionModel::Contoured);
    }
    if models.is_empty() {
        bail!(
            "no motion model is left: turn on rigid or deformable, or have the targets \
             contoured on every phase"
        );
    }
    let (label, patient) = {
        let d = ctx.ds(ds)?;
        (d.label.clone(), d.study.meta.patient_name.clone())
    };
    let req = motion::MotionRequest {
        run_name: format!(
            "{} · {group_name} · ref {}",
            node.label(),
            phases_used[reference_idx].0
        ),
        slot_name: label,
        patient,
        group_name: group_name.clone(),
        study_uid: study_uid.clone(),
        phases: phases_used.clone(),
        reference: reference_idx,
        targets: target_structs,
        ref_struct,
        contoured,
        models,
        local_rigid_margin_mm: (prm.local_rigid_margin_mm > 0.0)
            .then_some(prm.local_rigid_margin_mm),
        build_itv: prm.build_itv,
        itv_margin_mm: prm.itv_margin_mm.max(0.0),
        keep_phase_segs: prm.keep_phase_segs,
        params: prm.effort.params(RegMethod::ElastixRigid),
        volumes: ctx.volumes.clone(),
    };
    let out = motion::run(req, p)?;

    // File the ITVs on the reference phase, and the per-phase copies.
    let mut itv_names = Vec::new();
    if let Some(itv) = out.itv_series {
        let series = phases_used[reference_idx].1.clone();
        let items: Vec<Propagated> = itv
            .segs
            .into_iter()
            .map(|(n, c, m)| item(n, c, m))
            .collect();
        let types = vec![String::new(); items.len()];
        let landed = land(
            ctx,
            ds,
            &series,
            &itv.grid,
            items,
            &FileAs::landing(prm.itv_landing, &itv.label, prm.names),
            &types,
            None,
        )?;
        itv_names = landed.iter().flatten().cloned().collect();
    }
    {
        let d = ctx.ds_mut(ds)?;
        for o in out.phase_series {
            d.study.seg_series.push(o.into_seg_series(&out.study_uid));
            let i = d.study.seg_series.len() - 1;
            d.touched_segs.insert(i);
        }
    }
    let r = out.report;
    let mut tracks = Table::new(
        "Motion",
        &[
            "Structure",
            "Model",
            "Peak-to-peak (mm)",
            "Largest |d| (mm)",
        ],
    );
    for t in r.tracks.iter().chain(&r.reference_tracks) {
        let largest = t
            .displacements()
            .iter()
            .map(|d| d.length())
            .fold(0.0, f64::max);
        tracks.row(vec![
            t.target.clone(),
            t.model.label().to_string(),
            r2(t.peak_to_peak()),
            r2(largest),
        ]);
    }
    let mut itvs = Table::new(
        "ITV",
        &["Target", "Model", "Margin (mm)", "Volume (cm3)", "Filed as"],
    );
    for i in &r.itvs {
        itvs.row(vec![
            i.target.clone(),
            i.model.label().to_string(),
            r1(i.margin_mm),
            r2(i.volume_cm3),
            i.seg_name.clone(),
        ]);
    }
    let mut corr = Table::new(
        "Correlation with the reference",
        &["Target", "Model", "Axis", "r", "p"],
    );
    for (target, model, axes) in &r.correlations {
        for a in axes {
            corr.row(vec![
                target.clone(),
                model.label().to_string(),
                a.axis.to_string(),
                format!("{:.3}", a.r),
                format!("{:.3}", a.p),
            ]);
        }
    }
    let mut tables = vec![tracks];
    if !corr.rows.is_empty() {
        tables.push(corr);
    }
    if !itvs.rows.is_empty() {
        tables.push(itvs);
    }
    let lines: Vec<String> = r
        .tracks
        .iter()
        .take(4)
        .map(|t| {
            format!(
                "{} ({}): {} mm peak-to-peak",
                t.target,
                t.model.label(),
                r1(t.peak_to_peak())
            )
        })
        .chain(r.itvs.iter().take(2).map(|i| {
            format!(
                "ITV {} ({}): {} cm3",
                i.target,
                i.model.label(),
                r1(i.volume_cm3)
            )
        }))
        .collect();
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Motion: {}", node.label()),
        notes: vec![format!(
            "{} phases of '{group_name}', reference {}.",
            phases_used.len(),
            phases_used[reference_idx].0
        )],
        tables,
        motion: Some(r),
    });
    let mut out = done(
        vec![
            Value::Structures {
                ds,
                names: itv_names,
                on: Scope::Image(ref_uid.clone()),
            },
            Value::Report(report),
        ],
        lines,
    );
    out.report = Some(report);
    out.touched.push((ds, Some(ref_uid)));
    Ok(out)
}

// ---- output ------------------------------------------------------------------

fn export_dicom(
    ctx: &mut Ctx,
    values: &[Value],
    prm: &cat::ExportDicom,
    p: &Progress,
) -> Result<Done> {
    let mut studies: Vec<usize> = Vec::new();
    for v in values {
        if let Some(ds) = v.dataset() {
            if !studies.contains(&ds) {
                studies.push(ds);
            }
        }
    }
    if studies.is_empty() {
        bail!("nothing arrived to export");
    }
    let per_input = !prm.folder.contains("{input}") && studies.len() > 1;
    let mut lines = Vec::new();
    for ds in studies {
        let d = ctx.ds(ds)?;
        let mut folder = ctx.out_path(&prm.folder, &d.label);
        if per_input {
            folder = folder.join(safe_name(&d.label));
        }
        let study = &d.study;
        let mut plan = ExportPlan::build(export::one_study(study), ExportParams::for_study(study));
        plan.layout = Layout::StudyFolders;
        plan.set_uid_mode(match prm.uids {
            cat::UidChoice::Keep => UidMode::Keep,
            cat::UidChoice::New => UidMode::New,
        });
        plan.set_all_formats(match prm.format {
            cat::StructFormatChoice::Rtstruct => StructFormat::RtStruct,
            cat::StructFormatChoice::Seg => StructFormat::Seg,
        });
        for st in plan.studies_mut() {
            for se in &mut st.series {
                se.selected = prm.images;
            }
            for ob in &mut st.objects {
                ob.selected = match ob.kind {
                    ObjKind::Structures => {
                        prm.sets == cat::WhichSets::All || d.touched_sets.contains(&ob.index)
                    }
                    ObjKind::Segmentation => {
                        prm.sets == cat::WhichSets::All || d.touched_segs.contains(&ob.index)
                    }
                    ObjKind::Dose | ObjKind::Plan => prm.doses_and_plans,
                };
            }
        }
        if plan.is_empty() {
            lines.push(format!("{}: nothing to write", d.label));
            continue;
        }
        std::fs::create_dir_all(&folder).with_context(|| format!("create {}", folder.display()))?;
        p.set(format!("Writing {}", d.label));
        let summary = export::run(&plan, export::one_study(study), &folder, p)?;
        ctx.ds_mut(ds)?.exported.push(folder.clone());
        let d = ctx.ds(ds)?;
        lines.push(format!(
            "{}: {} files to {}",
            d.label,
            summary.files,
            folder.display()
        ));
        for w in summary.warnings.iter().take(3) {
            lines.push(w.clone());
        }
        ctx.channel
            .log(format!("Exported {} to {}", d.label, folder.display()));
    }
    Ok(done(Vec::new(), lines))
}

fn save_report(
    ctx: &mut Ctx,
    node: &Node,
    values: &[Value],
    prm: &cat::SaveReport,
) -> Result<Done> {
    let idx: Vec<usize> = values
        .iter()
        .filter_map(|v| match v {
            Value::Report(i) => Some(*i),
            _ => None,
        })
        .collect();
    if idx.is_empty() {
        bail!("no report arrived");
    }
    let folder = ctx.out_path(&prm.folder, "");
    std::fs::create_dir_all(&folder).with_context(|| format!("create {}", folder.display()))?;
    let mut written: Vec<PathBuf> = Vec::new();
    let mut md = format!("# {}\n\n", node.label());
    for &i in &idx {
        let r = ctx
            .reports
            .get(i)
            .ok_or_else(|| anyhow!("internal: no report {i}"))?;
        // "Motion: heart" as a file name reads "Motion - heart".
        let stem = safe_name(&r.title.replace(": ", " - "));
        if prm.csv {
            for t in &r.tables {
                let path = folder.join(format!("{stem} - {}.csv", safe_name(&t.title)));
                write(&path, &t.csv())?;
                written.push(path);
            }
            if let Some(m) = &r.motion {
                let path = folder.join(format!("{stem} - full report.csv"));
                write(&path, &m.csv())?;
                written.push(path);
            }
        }
        md.push_str(&r.markdown());
    }
    if prm.text {
        let path = folder.join(format!("{}.md", safe_name(&node.label())));
        write(&path, &md)?;
        written.push(path);
    }
    let mut lines = vec![format!("{} files to {}", written.len(), folder.display())];
    lines.extend(
        written
            .iter()
            .take(4)
            .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string())),
    );
    Ok(done(Vec::new(), lines))
}

fn write(path: &Path, text: &str) -> Result<()> {
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_folder_signature_changes_with_what_is_in_it() {
        let dir = std::env::temp_dir().join(format!("rds-wf-sig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::write(dir.join("a/x.dcm"), b"one").unwrap();
        let before = folder_signature(&dir);
        assert_eq!(
            before,
            folder_signature(&dir),
            "the same files, the same signature"
        );
        std::fs::write(dir.join("a/y.dcm"), b"two").unwrap();
        assert_ne!(before, folder_signature(&dir), "a file added");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
