//! Patient setup from radiographs: the *2D/3D registration* and *Positioned
//! copy* steps.

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};

use super::super::catalog as cat;
use super::super::exec::{Ctx, Done, RegEntry, Report, Table, Value};
use super::super::Node;
use super::{done, image_of, r1, r2};
use crate::loader::LoadedStudy;
use crate::progress::Progress;
use crate::registration::metric2d::Metric2d;
use crate::registration::projective::Dof2d3d;
use crate::registration::{self, RegMethod, RigidTransform, Transform3};
use crate::setup::{self, GeometrySource, MarkerSource, SetupMethod, SetupOptions, ViewChoice};

/// The study with series `uid` as its volume.
fn study_on(ctx: &mut Ctx, ds: usize, uid: &str, p: &Progress) -> Result<LoadedStudy> {
    let vol = ctx.volume(ds, uid, p)?;
    let mut study = ctx.ds(ds)?.study.clone();
    if let Some(i) = study.series.iter().position(|s| s.uid == uid) {
        study.active_series = i;
    }
    study.volume = vol;
    Ok(study)
}

/// The planar images a parameter names: numbers or labels separated by
/// commas, or every image that states its geometry.
fn chosen_images(study: &LoadedStudy, text: &str) -> Result<Vec<usize>> {
    if text.trim().is_empty() {
        let all: Vec<usize> = study
            .radiographs()
            .filter(|(_, p)| p.camera.is_some())
            .map(|(i, _)| i)
            .collect();
        if all.is_empty() {
            bail!("the study has no radiograph that states its geometry");
        }
        return Ok(all);
    }
    text.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| match t.parse::<usize>() {
            Ok(n) if n >= 1 && n <= study.planar_images.len() => Ok(n - 1),
            Ok(n) => Err(anyhow!("there is no planar image {n}")),
            Err(_) => study
                .planar_images
                .iter()
                .position(|p| p.label == t)
                .ok_or_else(|| anyhow!("no planar image is labelled '{t}'")),
        })
        .collect()
}

pub(super) fn register_2d3d(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    prm: &cat::Register2d3d,
    p: &Progress,
) -> Result<Done> {
    let (ds, uid) = image_of(v)?;
    let study = study_on(ctx, ds, &uid, p)?;
    let images = chosen_images(&study, &prm.images)?;
    let views: Vec<ViewChoice> = images
        .iter()
        .map(|&i| ViewChoice::new(i, GeometrySource::Image))
        .collect();
    let opts = SetupOptions {
        method: SetupMethod::from_key(&prm.method)
            .ok_or_else(|| anyhow!("unknown method '{}'", prm.method))?,
        metric: Metric2d::from_key(&prm.metric)
            .ok_or_else(|| anyhow!("unknown measure '{}'", prm.metric))?,
        dof: if prm.translations_only {
            Dof2d3d::Translations
        } else {
            Dof2d3d::Six
        },
        clip_half_mm: (prm.clip_mm > 0.0).then_some(prm.clip_mm),
        clip_structure: Some(prm.clip_structure.trim().to_string()).filter(|s| !s.is_empty()),
        clip_margin_mm: prm.clip_margin_mm.clamp(0.0, 100.0),
        clip_rays: prm.clip_rays,
        threshold_hu: prm.threshold_hu,
        levels: prm.levels.clamp(1, 5),
        max_pixels: prm.pixels.clamp(32, 1024),
        markers: if prm.detect_markers {
            MarkerSource::Detect
        } else {
            MarkerSource::Pois
        },
        device: prm.device.pref(),
        ..SetupOptions::default()
    };
    let rooms = crate::rooms::load_default().0;
    p.set("Registering the radiographs");
    let s = setup::run(&study, &views, &opts, Some(&rooms), p)?;
    let r = &s.result;

    let mut t = Table::new("Setup", &["Quantity", "Value"]);
    let c = r.correction();
    t.row(vec!["Method".into(), s.method.label().into()]);
    t.row(vec!["Images".into(), s.image_labels.join(", ")]);
    t.row(vec![
        "Correction, translation (mm)".into(),
        format!("{} {} {}", r2(c[3]), r2(c[4]), r2(c[5])),
    ]);
    t.row(vec![
        "Correction, rotation (deg)".into(),
        format!("{} {} {}", r2(c[0]), r2(c[1]), r2(c[2])),
    ]);
    t.row(vec![
        "Couch lat / lng / vrt (mm)".into(),
        format!(
            "{} {} {}",
            r2(r.couch.lateral),
            r2(r.couch.longitudinal),
            r2(r.couch.vertical)
        ),
    ]);
    t.row(vec![
        "Couch yaw / pitch / roll (deg)".into(),
        format!(
            "{} {} {}",
            r2(r.couch.yaw),
            r2(r.couch.pitch),
            r2(r.couch.roll)
        ),
    ]);
    for v in &r.views {
        t.row(vec![
            format!("{} ({})", v.label, r.metric.tag()),
            format!("{} to {}", r2(v.before), r2(v.after)),
        ]);
    }
    if let Some(m) = &r.markers {
        t.row(vec!["Marker RMS (mm)".into(), r2(m.rms_mm)]);
    }
    t.row(vec!["Time (s)".into(), r1(r.elapsed_secs)]);
    let mut notes = vec![r.correction_line()];
    if let Some(w) = &r.warning {
        notes.push(w.clone());
    }

    let label = ctx.ds(ds)?.label.clone();
    let mut lines = vec![r.metric_line()];
    if !prm.folder.trim().is_empty() {
        let folder = ctx.out_path(&prm.folder, &label);
        let files = setup::write_files(&study, &s, &folder)?;
        notes.push(format!(
            "{} files written to {}.",
            files.len(),
            folder.display()
        ));
        lines.push(format!("to {}", folder.display()));
    }
    let registration = r.to_registration(&study.volume);
    {
        let d = ctx.ds_mut(ds)?;
        if prm.file_drrs {
            for k in 0..s.result.views.len() {
                let radiograph = s
                    .images
                    .get(k)
                    .and_then(|i| d.study.planar_images.get(*i))
                    .cloned();
                if let Some(img) = s.drr_planar(k, radiograph.as_ref()) {
                    d.study.planar_images.push(img);
                }
            }
        }
        d.study.setups.push(s.clone());
    }
    let reg = ctx.add_reg(RegEntry::Pair {
        fixed: (ds, uid.clone()),
        moving: (ds, uid.clone()),
        result: Arc::new(registration),
    });
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("2D/3D registration: {}", node.label()),
        notes,
        tables: vec![t],
        motion: None,
    });
    let mut out = done(vec![Value::Registration(reg), Value::Report(report)], lines);
    out.report = Some(report);
    out.touched.push((ds, Some(uid)));
    Ok(out)
}

pub(super) fn positioned_copy(
    ctx: &mut Ctx,
    node: &Node,
    v: &Value,
    pose: Option<&Value>,
    prm: &cat::PositionedCopy,
    p: &Progress,
) -> Result<Done> {
    let (ds, uid) = image_of(v)?;
    let study = study_on(ctx, ds, &uid, p)?;
    let vol = study.volume.clone();
    let centre = vol.voxel_to_patient(
        (vol.dims[0] as f64 - 1.0) * 0.5,
        (vol.dims[1] as f64 - 1.0) * 0.5,
        (vol.dims[2] as f64 - 1.0) * 0.5,
    );
    // Planned point to where it is found.
    let (displacement, from) = match pose {
        None => (
            RigidTransform::new(
                [
                    prm.rotation_deg[0].to_radians(),
                    prm.rotation_deg[1].to_radians(),
                    prm.rotation_deg[2].to_radians(),
                    prm.translation_mm[0],
                    prm.translation_mm[1],
                    prm.translation_mm[2],
                ],
                centre,
            ),
            "the displacement given".to_string(),
        ),
        Some(Value::Registration(r)) => match ctx.regs.get(*r) {
            Some(RegEntry::Pair { fixed, result, .. })
                if result.method == RegMethod::Projective =>
            {
                if fixed.0 != ds || fixed.1 != uid {
                    bail!("the pose is of another image series");
                }
                (
                    result.transform.rigid.inverse(),
                    "the 2D/3D registration".to_string(),
                )
            }
            _ => bail!("the pose must come from a 2D/3D registration"),
        },
        Some(_) => bail!("the pose input takes a registration"),
    };
    p.set("Moving the study");
    let copy = setup::moved_copy(&study, &displacement, p);
    let copy_uid = copy
        .series
        .get(copy.active_series)
        .map(|s| s.uid.clone())
        .unwrap_or_default();
    let copy_vol = copy.volume.clone();
    let (label, origin) = {
        let d = ctx.ds(ds)?;
        (format!("{} positioned", d.label), d.origin.clone())
    };
    let new_ds = ctx.add_dataset(label.clone(), node.id, origin, None, copy);
    // The copy onto the original: the inverse of the displacement.
    let t = Arc::new(Transform3::rigid_only(displacement.inverse()));
    let result = registration::RegistrationResult {
        analysis: registration::analysis::analyse(&copy_vol, &t, None),
        transform: t,
        method: RegMethod::Given,
        metric: registration::Metric::MeanSquares,
        initial_metric: 0.0,
        final_metric: 0.0,
        iterations_run: 0,
        elapsed_secs: 0.0,
        region: None,
    };
    let reg = ctx.add_reg(RegEntry::Pair {
        fixed: (new_ds, copy_uid.clone()),
        moving: (ds, uid),
        result: Arc::new(result),
    });
    let mut out = done(
        vec![
            Value::Image {
                ds: new_ds,
                uid: copy_uid.clone(),
            },
            Value::Registration(reg),
        ],
        vec![format!("{label}, moved by {from}")],
    );
    out.touched.push((new_ds, Some(copy_uid)));
    Ok(out)
}
