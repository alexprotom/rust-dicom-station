//! Planar images: the *Project structures* and *2D/2D registration* steps.
//! Images are named by number or label from the study behind the Image
//! value, as the *2D/3D registration* step names them.

use anyhow::{anyhow, bail, Result};

use super::super::catalog as cat;
use super::super::exec::{Ctx, Done, Report, Table, Value};
use super::super::Node;
use super::{done, image_of, r1, r2};
use crate::loader::LoadedStudy;
use crate::planar_structs::{self, PoseChoice, ProjectRequest};
use crate::progress::Progress;
use crate::registration::metric2d::Metric2d;
use crate::registration::planar::{self as p2, Dof2d, Method2d, Planar2d, PlanarReg, Request2d};

/// A planar image by number (1-based) or label.
fn image_named(study: &LoadedStudy, t: &str) -> Result<usize> {
    let t = t.trim();
    match t.parse::<usize>() {
        Ok(n) if n >= 1 && n <= study.planar_images.len() => Ok(n - 1),
        Ok(n) => Err(anyhow!("there is no planar image {n}")),
        Err(_) => study
            .planar_images
            .iter()
            .position(|p| p.label == t)
            .ok_or_else(|| anyhow!("no planar image is labelled '{t}'")),
    }
}

pub(super) fn project_structures(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::ProjectStructures,
    p: &Progress,
) -> Result<Done> {
    let (ds, uid) = image_of(v)?;
    let vol = ctx.volume(ds, &uid, p)?;
    let mut study = planar_structs::projection_view(&ctx.ds(ds)?.study);
    if let Some(i) = study.series.iter().position(|s| s.uid == uid) {
        study.active_series = i;
    }
    study.volume = vol;
    let images: Vec<usize> = if prm.images.trim().is_empty() {
        study
            .radiographs()
            .filter(|(_, p)| p.camera.is_some())
            .map(|(i, _)| i)
            .collect()
    } else {
        prm.images
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| image_named(&study, t))
            .collect::<Result<_>>()?
    };
    if images.is_empty() {
        bail!("the study has no planar image that states its geometry");
    }
    let pose = if prm.at_setup {
        if study.setups.is_empty() {
            bail!("the study has no setup to project at");
        }
        PoseChoice::Setup(study.setups.len() - 1)
    } else {
        PoseChoice::AsIs
    };
    let req = ProjectRequest {
        structures: prm
            .structures
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        images,
        pose,
        label: prm.set.trim().to_string(),
    };
    let (set, lines) = planar_structs::project_many(&study, &req, p)?;
    let label = set.label.clone();
    let mut t = Table::new("Projected", &["Structure", "Image", "Rings", "Area (mm2)"]);
    for l in &lines {
        t.row(vec![
            l.structure.clone(),
            study
                .planar_images
                .get(l.image)
                .map(|i| i.label.clone())
                .unwrap_or_default(),
            l.rings.to_string(),
            r1(l.area_mm2),
        ]);
    }
    {
        let d = ctx.ds_mut(ds)?;
        planar_structs::file_set(&mut d.study, set);
    }
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Project structures: {}", node.label()),
        notes: vec![format!("filed as the planar structure set {label}")],
        tables: vec![t],
        motion: None,
    });
    let mut out = done(vec![Value::Report(report)], vec![label]);
    out.report = Some(report);
    out.touched.push((ds, Some(uid)));
    Ok(out)
}

pub(super) fn register_2d(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::Register2d,
    p: &Progress,
) -> Result<Done> {
    let (ds, uid) = image_of(v)?;
    let pairs =
        cat::parse_pairs(&prm.pairs).ok_or_else(|| anyhow!("the pairs are fixed:moving images"))?;
    let method = Method2d::from_key(&prm.method)
        .ok_or_else(|| anyhow!("unknown method '{}'", prm.method))?;
    let metric = Metric2d::from_key(&prm.metric)
        .ok_or_else(|| anyhow!("unknown measure '{}'", prm.metric))?;
    let dof = Dof2d::from_key(&prm.dof).unwrap_or_default();
    let mut t = Table::new(
        "2D/2D registration",
        &["Match", "Shift (mm)", "Rotation (deg)", "Scale", "Measure"],
    );
    let mut made = Vec::new();
    for (k, (fs, ms)) in pairs.iter().enumerate() {
        let study = &ctx.ds(ds)?.study;
        let (fi, mi) = (image_named(study, fs)?, image_named(study, ms)?);
        if fi == mi {
            bail!(
                "match {}: the fixed and the moving image are the same",
                k + 1
            );
        }
        let (f, m) = (&study.planar_images[fi], &study.planar_images[mi]);
        let mut req = Request2d::new(Planar2d::of(f), Planar2d::of(m));
        req.method = method;
        req.metric = metric;
        req.dof = dof;
        req.levels = prm.levels.clamp(1, 5);
        req.max_pixels = prm.pixels.clamp(32, 2048);
        req.detect_markers = (prm.markers_mm > 0.0).then_some(prm.markers_mm);
        p.set(format!("Registering match {} of {}", k + 1, pairs.len()));
        let result = p2::register(&req, p)?;
        let r = &result.transform.rigid;
        t.row(vec![
            format!("{} onto {}", m.label, f.label),
            format!("{} {}", r2(r.tx), r2(r.ty)),
            r2(r.theta.to_degrees()),
            format!("{:.4}", r.scale),
            format!("{} to {}", r2(result.before), r2(result.after)),
        ]);
        made.push(PlanarReg {
            label: format!("{} onto {}", m.label, f.label),
            fixed_uid: f.sop_instance_uid.clone(),
            moving_uid: m.sop_instance_uid.clone(),
            result,
            moved_copy_uid: None,
            when: crate::dicom_export::today().0,
        });
    }
    let mut notes = Vec::new();
    let mut setup = None;
    if prm.as_setup {
        let vol = ctx.volume(ds, &uid, p)?;
        let mut study = ctx.ds(ds)?.study.clone();
        if let Some(i) = study.series.iter().position(|s| s.uid == uid) {
            study.active_series = i;
        }
        study.volume = vol;
        let refs: Vec<&PlanarReg> = made.iter().collect();
        let s = crate::setup::from_planar(&study, &refs, &Default::default(), p)?;
        notes.push(s.result.correction_line());
        if let Some(w) = &s.result.warning {
            notes.push(w.clone());
        }
        setup = Some(s);
    }
    let lines = vec![format!("{} match(es)", made.len())];
    {
        let d = ctx.ds_mut(ds)?;
        d.study.planar_regs.extend(made);
        if let Some(s) = setup {
            d.study.setups.push(s);
        }
    }
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("2D/2D registration: {}", node.label()),
        notes,
        tables: vec![t],
        motion: None,
    });
    let mut out = done(vec![Value::Report(report)], lines);
    out.report = Some(report);
    out.touched.push((ds, Some(uid)));
    Ok(out)
}
