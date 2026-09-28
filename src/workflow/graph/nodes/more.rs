//! The steps added after the first twelve: another source (the archive), a
//! batch's folders, anonymizing, SegVol by name, editing structures
//! (combine, rename or delete), placing a structure by its relationship to
//! another, copying onto the phases, the dose measurements, filing into the
//! archive and DRRs.
//!
//! Like the rest of `nodes`, each is the program's own code path with the
//! wire values turned into its arguments: `structops::combine`,
//! `group::copy_to_phases`, `dvh`, `anonymize`, `archive`, `drr`, the SegVol
//! model the prompt window uses.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};

use super::super::catalog::{self as cat, name_matches, split_names};
use super::super::exec::{Ctx, Done, Report, Scope, Table, Value};
use super::super::safe_name;
use super::super::Node;
use super::{
    count, done, engine_node, group_of, land, models_root, phases, r1, r2, roi_type_of,
    study_loaded, study_of, taken_on, FileAs, Made,
};
use crate::loader::{self, LoadedStudy, SeriesInfo};
use crate::models::{self, Engine};
use crate::progress::Progress;
use crate::volume::Grid;
use crate::workflow::session::item;
use crate::workflow::{group, select};

// ---- helpers -----------------------------------------------------------------

/// The image series of `ds` with this UID.
fn series_of(ctx: &Ctx, ds: usize, uid: &str) -> Result<SeriesInfo> {
    ctx.ds(ds)?
        .study
        .series
        .iter()
        .find(|s| s.uid == uid)
        .cloned()
        .ok_or_else(|| anyhow!("the image series is no longer in the study"))
}

/// The lattice of a series without reading its pixels when it can be had
/// otherwise: the displayed volume's, a volume the run holds, or the slice
/// headers'.
fn grid_of(ctx: &mut Ctx, ds: usize, series: &SeriesInfo, p: &Progress) -> Result<Grid> {
    let st = &ctx.ds(ds)?.study;
    if st.has_volume() && st.series.get(st.active_series).map(|s| &s.uid) == Some(&series.uid) {
        return Ok(st.volume.grid());
    }
    if let Some(v) = ctx.volumes.get(series) {
        return Ok(v.grid());
    }
    match loader::series_grid(series) {
        Ok(g) => Ok(g),
        Err(_) => Ok(ctx.volume(ds, &series.uid, p)?.grid()),
    }
}

/// The images a structures value is on: one image, every phase of a group,
/// labelled; the study as a whole has none.
fn images_of(ctx: &Ctx, ds: usize, on: &Scope) -> Result<Vec<(String, SeriesInfo)>> {
    match on {
        Scope::Image(uid) => Ok(vec![(String::new(), series_of(ctx, ds, uid)?)]),
        Scope::Group(g) => phases(ctx, ds, *g),
        Scope::Study => bail!(
            "find the structures on an image series or a 4D group first (connect the image to \
             'Find structures')"
        ),
    }
}

/// A structure by name on one image, as a mask on `grid`.
fn mask_on(study: &LoadedStudy, name: &str, uid: &str, grid: &Grid) -> Result<(Vec<u8>, [u8; 3])> {
    let s = select::find_on_series(study, name, uid, &grid.frame_of_reference_uid)
        .ok_or_else(|| anyhow!("no structure '{name}' on the image"))?;
    Ok((s.mask_on(grid)?, s.color))
}

fn structures(v: &Value) -> Result<(usize, &[String], &Scope)> {
    match v {
        Value::Structures { ds, names, on } => Ok((*ds, names.as_slice(), on)),
        _ => bail!("this input needs structures"),
    }
}

// ---- input ---------------------------------------------------------------------

/// The archive a step names, or the station's.
fn archive_of(archive: &str) -> crate::archive::Archive {
    let root = if archive.trim().is_empty() {
        crate::archive::root_from_setting(
            &crate::settings::load()
                .archive_dir
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )
    } else {
        PathBuf::from(archive.trim())
    };
    crate::archive::Archive::new(root)
}

/// The study folder an archive step reads: its patient, and of that
/// patient's studies the one the step names, or the newest.
pub(super) fn archive_study_dir(prm: &cat::LoadFromArchive) -> Result<PathBuf> {
    let arch = archive_of(&prm.archive);
    let patients = arch.scan()?;
    let want = prm.patient.trim();
    let hits: Vec<_> = patients
        .iter()
        .filter(|pt| name_matches(want, &pt.id) || name_matches(want, &pt.name))
        .collect();
    let pt = match hits.as_slice() {
        [one] => *one,
        [] => bail!(
            "no patient '{want}' in the archive ({} patients)",
            patients.len()
        ),
        many => bail!(
            "'{want}' names {} patients in the archive; give the ID",
            many.len()
        ),
    };
    let study = prm.study.trim().to_lowercase();
    let pick = if study.is_empty() {
        pt.studies.iter().max_by(|a, b| a.date.cmp(&b.date))
    } else {
        pt.studies.iter().rev().find(|s| {
            s.date == study
                || s.description.to_lowercase().contains(&study)
                || name_matches(&study, &s.description)
        })
    };
    pick.map(|s| s.dir.clone())
        .ok_or_else(|| anyhow!("the patient has no study like '{}'", prm.study.trim()))
}

pub(super) fn load_from_archive(
    ctx: &mut Ctx,
    node: &Node,
    prm: &cat::LoadFromArchive,
    p: &Progress,
) -> Result<Done> {
    let dir = archive_study_dir(prm)?;
    p.set(format!("Reading {}", dir.display()));
    let study = loader::load_directory(&dir, p)?;
    Ok(study_loaded(ctx, node, dir, prm.workspace, study))
}

pub(super) fn anonymize(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::Anonymize,
    p: &Progress,
) -> Result<Done> {
    let ds = study_of(ctx, v)?;
    let (src, label, workspace) = {
        let d = ctx.ds(ds)?;
        (d.origin.clone(), d.label.clone(), d.workspace)
    };
    let out = ctx.out_path(&prm.folder, &label);
    let scan = crate::anonymize::scan(&src, p)?;
    let is_description = |f: &crate::anonymize::TagFinding| {
        matches!(f.name.as_str(), "StudyDescription" | "SeriesDescription")
    };
    let replacements: Vec<_> = scan
        .findings
        .iter()
        .filter(|f| f.enabled || (prm.clear_descriptions && is_description(f)))
        .map(|f| {
            let value = if prm.clear_descriptions && is_description(f) {
                String::new()
            } else {
                f.replacement.trim().to_string()
            };
            (f.tag, f.vr, value)
        })
        .collect();
    std::fs::create_dir_all(&out).with_context(|| format!("create {}", out.display()))?;
    let params = crate::anonymize::ApplyParams {
        replacements,
        remove_private: prm.remove_private,
        remap_uids: prm.remap_uids,
        mark_deidentified: true,
        out_dir: Some(out.clone()),
    };
    let n = crate::anonymize::apply(&scan.files, &scan.root, &params, p)?;
    p.set("Reading the anonymized copy");
    let study = loader::load_directory(&out, p)?;
    let ws = match workspace {
        Some(0) => cat::Workspace::A,
        Some(1) => cat::Workspace::B,
        Some(2) => cat::Workspace::C,
        Some(3) => cat::Workspace::D,
        _ => cat::Workspace::Auto,
    };
    let mut d = study_loaded(ctx, node, out.clone(), ws, study);
    if let Some(Value::Study(new)) = d.outputs.first() {
        ctx.ds_mut(*new)?.label = format!("{label} anonymized");
    }
    d.lines.insert(0, format!("{n} files to {}", out.display()));
    Ok(d)
}

// ---- SegVol by name -------------------------------------------------------------

pub(super) fn segvol_text(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::SegVolText,
    p: &Progress,
) -> Result<Done> {
    use crate::segvol::{infer, model::Model, preprocess, weights};
    let prompts: Vec<(String, String)> = prm
        .prompts
        .iter()
        .filter(|r| !r.structure.trim().is_empty())
        .map(|r| (r.structure.trim().to_string(), r.landed_name()))
        .collect();
    if prompts.is_empty() {
        bail!("no structure name to prompt with");
    }
    let dir = models_root(ctx, Engine::SegVol);
    let need = weights::download_needed(&dir, true);
    if need > 0 && !ctx.opts.allow_download {
        bail!(
            "the SegVol weights are not present ({} to download) and downloads are off for \
             this run",
            models::human_bytes(need)
        );
    }
    let cfg = infer::Config {
        use_zoom_in: prm.refine,
        threshold: prm.threshold.clamp(0.05, 0.95),
        ..infer::Config::default()
    };
    let device = prm.device.pref();
    // Loaded on the first image and kept for the rest: every phase of a
    // group is a prompt against the same network.
    let mut model: Option<Model> = None;
    let run = move |vol: &crate::volume::Volume, p: &Progress| -> Result<Made> {
        if model.is_none() {
            p.set("Loading SegVol");
            model = Some(Model::load(&dir, device, p)?);
        }
        let m = model.as_ref().expect("loaded above");
        p.set_device(&m.device);
        let prep = preprocess::prepare(vol);
        let mut made = Made {
            items: Vec::new(),
            roi_types: Vec::new(),
            rows: Vec::new(),
            notes: Vec::new(),
        };
        for (k, (structure, name)) in prompts.iter().enumerate() {
            p.set(format!("'{structure}' ({}/{})", k + 1, prompts.len()));
            let text = m.encode_text(structure, p)?;
            let seg = infer::segment(&m.net, &prep, &[], &[], Some(&text), cfg, p)?;
            let mask = prep.mask_to_volume_grid(&seg.mask, vol);
            let color = cat::organ_label(structure)
                .map(crate::autoseg::classes::class_color)
                .unwrap_or(
                    crate::segmentation::SEG_PALETTE[k % crate::segmentation::SEG_PALETTE.len()],
                );
            let it = item(name.clone(), color, mask);
            if it.voxels == 0 {
                made.notes.push(format!("'{structure}' found nothing"));
            }
            made.rows
                .push((name.clone(), it.voxels as f64 * vol.voxel_cm3()));
            made.roi_types.push(String::new());
            made.items.push(it);
        }
        Ok(made)
    };
    engine_node(
        ctx,
        node,
        v,
        FileAs::new(prm.output, prm.set, &prm.set_label, prm.names),
        "Prompt by name",
        p,
        run,
    )
}

// ---- editing structures ------------------------------------------------------------

/// The union of the named structures on one image, as a mask on `grid`.
fn union_of(study: &LoadedStudy, names: &[String], uid: &str, grid: &Grid) -> Result<Vec<u8>> {
    let mut acc: Option<Vec<u8>> = None;
    for n in names {
        let (m, _) = mask_on(study, n, uid, grid)?;
        acc = Some(match acc {
            None => m,
            Some(mut a) => {
                for (x, y) in a.iter_mut().zip(m) {
                    *x |= y;
                }
                a
            }
        });
    }
    acc.ok_or_else(|| anyhow!("no structure arrived"))
}

pub(super) fn combine(
    ctx: &mut Ctx,
    node: &Node,
    a: &Value,
    b: &[Value],
    prm: &cat::Combine,
    p: &Progress,
) -> Result<Done> {
    use crate::structops::{self, BoolOp, Cleanup, Operand, Recipe};
    let (ds, a_names, on) = structures(a)?;
    if a_names.is_empty() {
        bail!("no structure arrived on A");
    }
    let mut b_names: Vec<String> = Vec::new();
    for v in b {
        let (bds, names, _) = structures(v)?;
        if bds != ds {
            bail!("A and B must be structures of one study");
        }
        b_names.extend(names.iter().cloned());
    }
    let images = images_of(ctx, ds, on)?;
    let how = FileAs::new(prm.output, prm.set, &prm.set_label, prm.names);
    let taken = {
        let series: Vec<&SeriesInfo> = images.iter().map(|(_, s)| s).collect();
        taken_on(ctx, ds, &series, &how)?
    };
    let expr = if b_names.is_empty() {
        a_names.join(" ∪ ")
    } else {
        format!(
            "({}) {} ({})",
            a_names.join(" ∪ "),
            prm.op.bool_op().joiner(),
            b_names.join(" ∪ ")
        )
    };
    let mut table = Table::new(
        "Combined structure",
        &["Image", "Filed as", "Volume (cm3)", "Pieces"],
    );
    let mut first: Option<Vec<String>> = None;
    let n = images.len().max(1);
    for (i, (label, series)) in images.iter().enumerate() {
        if p.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        p.set_outer(i as f32 / n as f32, 1.0 / n as f32);
        let grid = grid_of(ctx, ds, series, p)?;
        let (am, bm) = {
            let st = &ctx.ds(ds)?.study;
            let am = union_of(st, a_names, &series.uid, &grid)?;
            let bm = if b_names.is_empty() {
                None
            } else {
                Some(union_of(st, &b_names, &series.uid, &grid)?)
            };
            (am, bm)
        };
        let mut operands = vec![Operand {
            name: "A".into(),
            mask: am,
            margin: prm.margin_a.margin(),
        }];
        if let Some(bm) = bm {
            operands.push(Operand {
                name: "B".into(),
                mask: bm,
                margin: prm.margin_b.margin(),
            });
        }
        let recipe = Recipe {
            op: if operands.len() == 1 {
                BoolOp::Union
            } else {
                prm.op.bool_op()
            },
            operands,
            margin: prm.margin.margin(),
            cleanup: Cleanup {
                fill_holes: prm.fill_holes,
                close_mm: prm.close_mm.max(0.0),
                keep_largest: prm.keep_largest,
                min_volume_cm3: prm.min_volume_cm3.max(0.0),
            },
        };
        let out = structops::combine(&recipe, &grid, p)?;
        let set_label = if label.is_empty() {
            prm.set_label.clone()
        } else {
            format!("{} {label}", prm.set_label)
        };
        let landed = land(
            ctx,
            ds,
            series,
            &grid,
            vec![item(prm.name.trim(), [255, 200, 0], out.mask)],
            &FileAs {
                set_label: &set_label,
                ..how
            },
            &[String::new()],
            Some(&taken),
        )?;
        table.row(vec![
            if label.is_empty() {
                series.description.clone()
            } else {
                label.clone()
            },
            landed
                .first()
                .cloned()
                .flatten()
                .unwrap_or_else(|| "(empty)".into()),
            r2(out.cm3),
            out.pieces.to_string(),
        ]);
        if first.is_none() {
            first = Some(landed.into_iter().flatten().collect());
        }
    }
    p.set_outer(0.0, 1.0);
    let names = first.unwrap_or_default();
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Combined: {}", node.label()),
        notes: vec![format!("{} = {expr}", prm.name.trim())],
        tables: vec![table],
        motion: None,
    });
    let last = images.last().map(|(_, s)| s.uid.clone());
    let mut d = done(
        vec![
            Value::Structures {
                ds,
                names: names.clone(),
                on: on.clone(),
            },
            Value::Report(report),
        ],
        vec![format!(
            "{} = {expr} on {}",
            names.join(", "),
            count(images.len(), "image", "images")
        )],
    );
    d.report = Some(report);
    d.touched.push((ds, last));
    Ok(d)
}

pub(super) fn rename(ctx: &mut Ctx, v: &Value, prm: &cat::Rename) -> Result<Done> {
    let (ds, names, on) = structures(v)?;
    let uids: Option<BTreeSet<String>> = match on {
        Scope::Study => None,
        _ => Some(
            images_of(ctx, ds, on)?
                .into_iter()
                .map(|(_, s)| s.uid)
                .collect(),
        ),
    };
    let rules: Vec<&cat::RenameRule> = prm
        .rules
        .iter()
        .filter(|r| !r.from.trim().is_empty())
        .collect();
    // Only the structures that arrived, when any did: a rule applies to
    // what the step before it found.
    let wanted = |name: &str| names.is_empty() || names.iter().any(|n| n == name);
    let rule_for = |name: &str| {
        rules
            .iter()
            .find(|r| name_matches(r.from.trim(), name))
            .copied()
    };
    let in_scope = |uid: &str| uids.as_ref().is_none_or(|u| u.contains(uid));
    let d = ctx.ds_mut(ds)?;
    let mut changed = 0usize;
    let mut out_names: Vec<String> = Vec::new();
    for (si, ss) in d.study.structure_sets.iter_mut().enumerate() {
        if !in_scope(&ss.referenced_series_uid) {
            continue;
        }
        let before = changed;
        match prm.action {
            cat::RenameAction::Delete => {
                let n0 = ss.rois.len();
                ss.rois
                    .retain(|r| !(wanted(&r.name) && rule_for(&r.name).is_some()));
                changed += n0 - ss.rois.len();
            }
            cat::RenameAction::Rename => {
                for k in 0..ss.rois.len() {
                    let name = ss.rois[k].name.clone();
                    let Some(rule) = rule_for(&name).filter(|_| wanted(&name)) else {
                        continue;
                    };
                    let taken: BTreeSet<String> = ss
                        .rois
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != k)
                        .map(|(_, r)| r.name.clone())
                        .collect();
                    let new = crate::workflow::session::chosen_name(
                        rule.to.trim(),
                        crate::workflow::session::NameClash::Counter,
                        &taken,
                    );
                    if !out_names.contains(&new) {
                        out_names.push(new.clone());
                    }
                    ss.rois[k].name = new;
                    changed += 1;
                }
            }
        }
        if changed > before {
            d.touched_sets.insert(si);
        }
    }
    for (si, sr) in d.study.seg_series.iter_mut().enumerate() {
        if !in_scope(&sr.referenced_series_uid) {
            continue;
        }
        let before = changed;
        match prm.action {
            cat::RenameAction::Delete => {
                let n0 = sr.segs.len();
                sr.segs
                    .retain(|s| !(wanted(&s.name) && rule_for(&s.name).is_some()));
                changed += n0 - sr.segs.len();
            }
            cat::RenameAction::Rename => {
                for s in sr.segs.iter_mut() {
                    if let Some(rule) = rule_for(&s.name).filter(|_| wanted(&s.name)) {
                        s.name = rule.to.trim().to_string();
                        if !out_names.contains(&s.name) {
                            out_names.push(s.name.clone());
                        }
                        changed += 1;
                    }
                }
            }
        }
        if changed > before {
            d.touched_segs.insert(si);
        }
    }
    if changed == 0 {
        bail!("no structure matched the rules");
    }
    let what = match prm.action {
        cat::RenameAction::Rename => "renamed",
        cat::RenameAction::Delete => "deleted",
    };
    Ok(done(
        vec![Value::Structures {
            ds,
            names: out_names.clone(),
            on: on.clone(),
        }],
        vec![format!(
            "{} {what}{}",
            count(changed, "structure", "structures"),
            if out_names.is_empty() {
                String::new()
            } else {
                format!(": {}", out_names.join(", "))
            }
        )],
    ))
}

// ---- placing and copying -----------------------------------------------------------

pub(super) fn transfer(
    ctx: &mut Ctx,
    node: &Node,
    target: &Value,
    reference: &Value,
    onto: &Value,
    prm: &cat::Transfer,
    p: &Progress,
) -> Result<Done> {
    let (src_ds, t_names, t_on) = structures(target)?;
    let Scope::Image(src_uid) = t_on else {
        bail!("the target must be found on one image series");
    };
    let (rds, r_names, _) = structures(reference)?;
    if rds != src_ds {
        bail!("the reference must be on the target's image");
    }
    let (dst_ds, d_names, d_on) = structures(onto)?;
    let (Some(tname), Some(rname), Some(dname)) =
        (t_names.first(), r_names.first(), d_names.first())
    else {
        bail!("a target, a reference and the reference on the destination are needed");
    };
    let src_series = series_of(ctx, src_ds, src_uid)?;
    let src_grid = grid_of(ctx, src_ds, &src_series, p)?;
    let (tm, tcolor, rm, rtype) = {
        let st = &ctx.ds(src_ds)?.study;
        let (tm, tcolor) = mask_on(st, tname, src_uid, &src_grid)?;
        let (rm, _) = mask_on(st, rname, src_uid, &src_grid)?;
        (tm, tcolor, rm, roi_type_of(st, tname, src_uid))
    };
    let (Some(c_target), Some(c_ref)) = (
        crate::motion::centroid_mm(&tm, &src_grid),
        crate::motion::centroid_mm(&rm, &src_grid),
    ) else {
        bail!("the target or the reference is empty on the source image");
    };
    let images = images_of(ctx, dst_ds, d_on)?;
    let how = FileAs::new(prm.output, prm.set, &prm.set_label, prm.names);
    let taken = {
        let series: Vec<&SeriesInfo> = images.iter().map(|(_, s)| s).collect();
        taken_on(ctx, dst_ds, &series, &how)?
    };
    let name = if prm.name.trim().is_empty() {
        tname.clone()
    } else {
        prm.name.trim().to_string()
    };
    let mut table = Table::new(
        "Placed",
        &[
            "Image",
            "Filed as",
            "Offset RL (mm)",
            "Offset AP (mm)",
            "Offset SI (mm)",
            "Volume (cm3)",
        ],
    );
    let off = c_target - c_ref;
    let mut first: Option<Vec<String>> = None;
    for (label, series) in &images {
        let grid = grid_of(ctx, dst_ds, series, p)?;
        let (dm, _) = {
            let st = &ctx.ds(dst_ds)?.study;
            mask_on(st, dname, &series.uid, &grid)?
        };
        let Some(c_dst) = crate::motion::centroid_mm(&dm, &grid) else {
            bail!("'{dname}' is empty on {}", series.description);
        };
        let t = crate::registration::Transform3::from_matrix(
            crate::registration::Mat4::translation(c_dst - c_ref),
            crate::geometry::Vec3::ZERO,
        );
        let mask = crate::propagate::carry_mask(&tm, &src_grid, &grid, &t);
        let cm3 = crate::motion::volume_cm3(&mask, &grid);
        let set_label = if label.is_empty() {
            prm.set_label.clone()
        } else {
            format!("{} {label}", prm.set_label)
        };
        let landed = land(
            ctx,
            dst_ds,
            series,
            &grid,
            vec![item(name.clone(), tcolor, mask)],
            &FileAs {
                set_label: &set_label,
                ..how
            },
            std::slice::from_ref(&rtype),
            Some(&taken),
        )?;
        table.row(vec![
            if label.is_empty() {
                series.description.clone()
            } else {
                label.clone()
            },
            landed
                .first()
                .cloned()
                .flatten()
                .unwrap_or_else(|| "(outside the image)".into()),
            r1(off.x),
            r1(off.y),
            r1(off.z),
            r2(cm3),
        ]);
        if first.is_none() {
            first = Some(landed.into_iter().flatten().collect());
        }
    }
    let names = first.unwrap_or_default();
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Transfer: {}", node.label()),
        notes: vec![format!(
            "'{tname}' placed at its offset from '{rname}' ({} {} {} mm RL/AP/SI), onto '{dname}'.",
            r1(off.x),
            r1(off.y),
            r1(off.z)
        )],
        tables: vec![table],
        motion: None,
    });
    let last = images.last().map(|(_, s)| s.uid.clone());
    let mut d = done(
        vec![
            Value::Structures {
                ds: dst_ds,
                names: names.clone(),
                on: d_on.clone(),
            },
            Value::Report(report),
        ],
        vec![format!(
            "{} placed on {}",
            if names.is_empty() {
                "nothing".to_string()
            } else {
                names.join(", ")
            },
            count(images.len(), "image", "images")
        )],
    );
    d.report = Some(report);
    d.touched.push((dst_ds, last));
    Ok(d)
}

pub(super) fn copy_to_phases(
    ctx: &mut Ctx,
    _node: &Node,
    structs: &Value,
    onto: &Value,
    prm: &cat::CopyToPhases,
    p: &Progress,
) -> Result<Done> {
    let (src_ds, names, on) = structures(structs)?;
    if names.is_empty() {
        bail!("no structures arrived to copy");
    }
    let (dst_ds, gi) = group_of(onto)?;
    let found: Vec<select::Structure> = {
        let st = &ctx.ds(src_ds)?.study;
        names
            .iter()
            .map(|n| {
                match on {
                    Scope::Image(uid) => select::find_on_series(st, n, uid, ""),
                    _ => select::find(st, n, None),
                }
                .ok_or_else(|| anyhow!("no structure '{n}'"))
            })
            .collect::<Result<_>>()?
    };
    let ph = phases(ctx, dst_ds, gi)?;
    let copies = group::copy_to_phases(&found, &ph, p)?;
    let how = FileAs::landing(prm.landing, "Copied", prm.names);
    let taken = {
        let series: Vec<&SeriesInfo> = ph.iter().map(|(_, s)| s).collect();
        taken_on(ctx, dst_ds, &series, &how)?
    };
    let mut first: Option<Vec<String>> = None;
    let mut notes = Vec::new();
    for c in copies {
        let series = ph
            .iter()
            .find(|(_, s)| s.uid == c.series_uid)
            .map(|(_, s)| s.clone())
            .ok_or_else(|| anyhow!("phase {} is gone", c.label))?;
        let items: Vec<_> = c
            .segs
            .into_iter()
            .map(|s| item(s.name, s.color, s.mask))
            .collect();
        let types: Vec<String> = items
            .iter()
            .map(|it| {
                let st = ctx.ds(src_ds).map(|d| &d.study);
                st.map(|st| roi_type_of(st, &it.name, ""))
                    .unwrap_or_default()
            })
            .collect();
        let set_label = format!("Copied {}", c.label);
        let landed = land(
            ctx,
            dst_ds,
            &series,
            &c.grid,
            items,
            &FileAs {
                set_label: &set_label,
                ..how
            },
            &types,
            Some(&taken),
        )?;
        if first.is_none() {
            first = Some(landed.into_iter().flatten().collect());
        }
        notes.extend(c.notes);
    }
    let names = first.unwrap_or_default();
    let mut lines = vec![format!(
        "{} on {} phases",
        if names.is_empty() {
            "nothing".to_string()
        } else {
            names.join(", ")
        },
        ph.len()
    )];
    lines.extend(notes.into_iter().take(3));
    let mut d = done(
        vec![Value::Structures {
            ds: dst_ds,
            names,
            on: Scope::Group(gi),
        }],
        lines,
    );
    d.touched
        .push((dst_ds, ph.last().map(|(_, s)| s.uid.clone())));
    Ok(d)
}

// ---- dose -----------------------------------------------------------------------------

/// The structures of the values, and the image they are measured on (the
/// one they were found on, else the study's displayed series).
fn measured(ctx: &Ctx, values: &[Value]) -> Result<(usize, Vec<String>, SeriesInfo)> {
    let mut ds = None;
    let mut uid = None;
    let mut names: Vec<String> = Vec::new();
    for v in values {
        let (d, n, on) = structures(v)?;
        if ds.is_some_and(|x| x != d) {
            bail!("the structures must be of one study");
        }
        ds = Some(d);
        if let Scope::Image(u) = on {
            uid = Some(u.clone());
        }
        for x in n {
            if !names.contains(x) {
                names.push(x.clone());
            }
        }
    }
    let ds = ds.ok_or_else(|| anyhow!("no structures arrived"))?;
    if names.is_empty() {
        bail!("no structures arrived");
    }
    let st = &ctx.ds(ds)?.study;
    let series = match uid {
        Some(u) => series_of(ctx, ds, &u)?,
        None => st
            .series
            .get(st.active_series)
            .cloned()
            .ok_or_else(|| anyhow!("the study has no image series to measure on"))?,
    };
    Ok((ds, names, series))
}

/// The dose a step measures against: of the kind asked for, with the words
/// asked for in its label; the first such.
fn pick_dose(study: &LoadedStudy, words: &str, kind: cat::DoseKindChoice) -> Result<usize> {
    if study.doses.is_empty() {
        bail!("the study holds no dose");
    }
    let words = words.trim().to_lowercase();
    study
        .doses
        .iter()
        .position(|d| {
            let effective = d.dose_type.eq_ignore_ascii_case("EFFECTIVE");
            let kind_ok = match kind {
                cat::DoseKindChoice::Any => true,
                cat::DoseKindChoice::Physical => !effective,
                cat::DoseKindChoice::Effective => effective,
            };
            kind_ok && (words.is_empty() || d.label.to_lowercase().contains(&words))
        })
        .ok_or_else(|| {
            anyhow!(
                "no {}{} among the study's doses ({})",
                kind.label(),
                if words.is_empty() {
                    String::new()
                } else {
                    format!(" labelled like '{words}'")
                },
                study
                    .doses
                    .iter()
                    .map(|d| d.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// Every structure's DVH against one dose.
fn curves(
    ctx: &mut Ctx,
    ds: usize,
    names: &[String],
    series: &SeriesInfo,
    dose: usize,
    bin_width: Option<f64>,
    p: &Progress,
) -> Result<(Vec<crate::dvh::Dvh>, Vec<String>)> {
    let grid = grid_of(ctx, ds, series, p)?;
    let st = &ctx.ds(ds)?.study;
    let d = &st.doses[dose];
    let mut out = Vec::new();
    let mut notes = Vec::new();
    for (i, n) in names.iter().enumerate() {
        if p.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        p.set(format!("DVH of {n} ({}/{})", i + 1, names.len()));
        let (mask, color) = match mask_on(st, n, &series.uid, &grid) {
            Ok(m) => m,
            Err(e) => {
                notes.push(format!("{n}: {e:#}"));
                continue;
            }
        };
        match crate::dvh::compute(
            n,
            color,
            &mask,
            &grid,
            d,
            crate::dvh::DvhParams { bin_width },
        ) {
            Ok(c) => out.push(c),
            Err(e) => notes.push(format!("{n}: {e:#}")),
        }
    }
    if out.is_empty() {
        bail!("no structure could be measured: {}", notes.join("; "));
    }
    Ok((out, notes))
}

fn metric_list(text: &str) -> Result<Vec<crate::dvh::Metric>> {
    let v: Vec<crate::dvh::Metric> = split_names(text)
        .iter()
        .map(|m| crate::dvh::Metric::parse(m).ok_or_else(|| anyhow!("'{m}' is not a metric")))
        .collect::<Result<_>>()?;
    Ok(if v.is_empty() {
        crate::dvh::default_metrics()
    } else {
        v
    })
}

/// One row per curve: the structure, its volume and the metrics.
fn metrics_table(title: &str, curves: &[crate::dvh::Dvh], metrics: &[crate::dvh::Metric]) -> Table {
    let units = curves
        .first()
        .map(|c| crate::dvh::nice_units(&c.units))
        .unwrap_or_default();
    let mut header = vec!["Structure".to_string(), "Volume (cm3)".to_string()];
    header.extend(
        metrics
            .iter()
            .map(|m| format!("{} ({})", m.label(), m.unit(&units))),
    );
    let mut t = Table {
        title: title.to_string(),
        header,
        rows: Vec::new(),
    };
    for c in curves {
        let mut row = vec![c.name.clone(), r2(c.volume_cm3)];
        row.extend(metrics.iter().map(|m| r2(m.evaluate(c))));
        t.row(row);
    }
    t
}

pub(super) fn dvh(
    ctx: &mut Ctx,
    node: &Node,
    values: &[Value],
    prm: &cat::Dvh,
    p: &Progress,
) -> Result<Done> {
    let (ds, names, series) = measured(ctx, values)?;
    let dose = pick_dose(&ctx.ds(ds)?.study, &prm.dose, cat::DoseKindChoice::Any)?;
    let metrics = metric_list(&prm.metrics)?;
    let mut protocol_text = prm.protocol.clone();
    if !prm.protocol_file.trim().is_empty() {
        let text = std::fs::read_to_string(prm.protocol_file.trim())
            .with_context(|| format!("read the protocol {}", prm.protocol_file.trim()))?;
        protocol_text.push('\n');
        protocol_text.push_str(&text);
    }
    let constraints = crate::dvh::parse_protocol(&protocol_text);
    let bin = (prm.bin_width > 0.0).then_some(prm.bin_width);
    let (curves, notes) = curves(ctx, ds, &names, &series, dose, bin, p)?;
    let dose_label = ctx.ds(ds)?.study.doses[dose].label.clone();
    let mut tables = vec![metrics_table("DVH metrics", &curves, &metrics)];
    let mut lines = vec![format!(
        "{} against '{dose_label}'",
        count(curves.len(), "structure", "structures")
    )];
    if !constraints.is_empty() {
        let verdicts = crate::dvh::check(&constraints, &curves);
        let passed = verdicts.iter().filter(|v| v.pass).count();
        let mut t = Table::new("Protocol", &["Constraint", "Structure", "Value", "Result"]);
        for v in &verdicts {
            t.row(vec![
                v.constraint.to_line(),
                v.structure.clone(),
                v.value.map(r2).unwrap_or_default(),
                if v.pass { "pass" } else { "FAIL" }.to_string(),
            ]);
        }
        tables.push(t);
        lines.push(format!("{passed} of {} constraints hold", verdicts.len()));
    }
    if prm.curves {
        let csv = crate::dvh::curves_csv(&curves, true);
        let mut rows = csv
            .lines()
            .map(|l| l.split(',').map(str::to_string).collect::<Vec<_>>());
        if let Some(header) = rows.next() {
            tables.push(Table {
                title: "Cumulative DVH".into(),
                header,
                rows: rows.collect(),
            });
        }
    }
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("DVH: {}", node.label()),
        notes: std::iter::once(format!(
            "Against '{dose_label}', on '{}'.",
            series.description
        ))
        .chain(notes)
        .collect(),
        tables,
        motion: None,
    });
    let mut d = done(vec![Value::Report(report)], lines);
    d.report = Some(report);
    Ok(d)
}

pub(super) fn dose_metrics(
    ctx: &mut Ctx,
    node: &Node,
    values: &[Value],
    prm: &cat::DoseMetrics,
    p: &Progress,
) -> Result<Done> {
    let (ds, names, series) = measured(ctx, values)?;
    let dose = pick_dose(&ctx.ds(ds)?.study, &prm.dose, prm.dose_kind)?;
    let metrics = metric_list(&prm.metrics)?;
    let (curves, notes) = curves(ctx, ds, &names, &series, dose, None, p)?;
    let dose_label = ctx.ds(ds)?.study.doses[dose].label.clone();
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Dose estimation: {}", node.label()),
        notes: std::iter::once(format!(
            "Against '{dose_label}' ({}).",
            prm.dose_kind.label()
        ))
        .chain(notes)
        .collect(),
        tables: vec![metrics_table("Dose metrics", &curves, &metrics)],
        motion: None,
    });
    let mut d = done(
        vec![Value::Report(report)],
        vec![format!(
            "{} against '{dose_label}'",
            count(curves.len(), "structure", "structures")
        )],
    );
    d.report = Some(report);
    Ok(d)
}

// ---- output -------------------------------------------------------------------------

pub(super) fn archive_import(
    ctx: &mut Ctx,
    values: &[Value],
    prm: &cat::ArchiveImport,
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
        bail!("nothing arrived to file");
    }
    let arch = archive_of(&prm.archive);
    let mut lines = Vec::new();
    for ds in studies {
        let d = ctx.ds(ds)?;
        let sources: Vec<PathBuf> = match prm.source {
            cat::ImportSource::Exported if !d.exported.is_empty() => d.exported.clone(),
            _ => vec![d.origin.clone()],
        };
        for src in sources {
            p.set(format!("Filing {}", src.display()));
            let s = arch.import(&src, p)?;
            lines.push(format!("{}: {}", d.label, s.describe()));
        }
    }
    Ok(done(Vec::new(), lines))
}

pub(super) fn drr(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::Drr,
    p: &Progress,
) -> Result<Done> {
    use crate::drr;
    let (ds, uid) = match v {
        Value::Image { ds, uid } => (*ds, uid.clone()),
        _ => bail!("this step needs one image series"),
    };
    let vol = ctx.volume(ds, &uid, p)?;
    let mut base = drr::DrrParams::for_volume(&vol);
    let size = prm.size_px.clamp(64, 2048);
    base.geometry.dims = [size, size];
    // (label, geometry) of every image to make.
    let views: Vec<(String, drr::Geometry)> = if prm.plan_beams {
        let st = &ctx.ds(ds)?.study;
        let plan = st
            .plans
            .iter()
            .find(|pl| !pl.beams.is_empty())
            .ok_or_else(|| anyhow!("the study has no plan with beams"))?;
        plan.beams
            .iter()
            .map(|b| {
                (
                    if b.name.trim().is_empty() {
                        format!("beam {}", b.number)
                    } else {
                        b.name.trim().to_string()
                    },
                    base.geometry.from_beam(b),
                )
            })
            .collect()
    } else {
        cat::parse_angles(&prm.angles)
            .ok_or_else(|| anyhow!("the angles are numbers separated by commas"))?
            .into_iter()
            .map(|a| {
                let mut g = base.geometry;
                g.gantry_deg = a;
                g.couch_deg = prm.couch_deg;
                (format!("G{a:.0}"), g)
            })
            .collect()
    };
    let label = ctx.ds(ds)?.label.clone();
    let folder = ctx.out_path(&prm.folder, &label);
    std::fs::create_dir_all(&folder).with_context(|| format!("create {}", folder.display()))?;
    let mut table = Table::new("DRR", &["Image", "Gantry (deg)", "Couch (deg)", "File"]);
    let mut planar = Vec::new();
    let n = views.len().max(1);
    for (i, (name, g)) in views.iter().enumerate() {
        if p.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        p.set_outer(i as f32 / n as f32, 1.0 / n as f32);
        let params = drr::DrrParams {
            geometry: *g,
            ..base
        };
        let img = drr::render(&vol, &params, p)?;
        let file = folder.join(format!("{}.png", safe_name(name)));
        write_png(&img, prm.invert, &file)?;
        table.row(vec![
            name.clone(),
            r1(g.gantry_deg),
            r1(g.couch_deg),
            file.display().to_string(),
        ]);
        if prm.file_into_study {
            let mut pl = img.to_planar(&params, prm.invert);
            pl.label = format!("DRR {name}");
            planar.push(pl);
        }
    }
    p.set_outer(0.0, 1.0);
    if !planar.is_empty() {
        ctx.ds_mut(ds)?.study.planar_images.extend(planar);
    }
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("DRR: {}", node.label()),
        notes: vec![format!("Written to {}.", folder.display())],
        tables: vec![table],
        motion: None,
    });
    let mut d = done(
        vec![Value::Report(report)],
        vec![format!(
            "{} to {}",
            count(views.len(), "image", "images"),
            folder.display()
        )],
    );
    d.report = Some(report);
    Ok(d)
}

/// A DRR as an 8-bit greyscale PNG over its own range.
fn write_png(img: &crate::drr::DrrImage, invert: bool, path: &std::path::Path) -> Result<()> {
    let [w, h] = img.dims;
    let span = (img.max - img.min).max(1e-6);
    let px: Vec<u8> = img
        .pixels
        .iter()
        .map(|v| {
            let t = ((v - img.min) / span).clamp(0.0, 1.0);
            let t = if invert { 1.0 - t } else { t };
            (t * 255.0).round() as u8
        })
        .collect();
    let buf = image::GrayImage::from_raw(w as u32, h as u32, px)
        .ok_or_else(|| anyhow!("the image has the wrong size"))?;
    buf.save(path)
        .with_context(|| format!("write {}", path.display()))
}
