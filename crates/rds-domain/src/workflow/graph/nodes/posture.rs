//! The *Posture model* step: two or three image series of one patient in
//! different positions as the states of [`crate::workflow::posture::build`],
//! and its tables as the step's report.
//!
//! Each input is an image series of a study the run holds; the state is
//! that series with the structure set drawn on it (else the study's first).
//! The step changes no study: what it makes is the report - bodies and
//! their screws, joints, structures with their volume change, shift and
//! model check, and the model's own checks - which *Save report* writes
//! like any other.

use anyhow::{anyhow, bail, Result};

use super::super::catalog as cat;
use super::super::exec::{Ctx, Done, Report, Table, Value};
use super::super::Node;
use super::{done, image_of, r1, r2};
use crate::loader::LoadedStudy;
use crate::posture::{Anchor, Mode};
use crate::progress::Progress;
use crate::workflow::posture::{self as wp, Bodies, PostureRequest, StateInput};

/// The request a step's parameters make.
pub(in super::super) fn request(prm: &cat::Posture) -> Result<PostureRequest> {
    let mode = Mode::from_key(&prm.mode)
        .ok_or_else(|| anyhow!("mode must be diffeomorphic or linear (got '{}')", prm.mode))?;
    let names: Vec<String> = prm
        .bodies
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Ok(PostureRequest {
        anchor: Anchor::parse(if prm.anchor.trim().is_empty() {
            "image"
        } else {
            &prm.anchor
        }),
        bodies: match names.as_slice() {
            [] => Bodies::Bones,
            [one] if one.eq_ignore_ascii_case("none") => Bodies::None,
            _ => Bodies::Named(names),
        },
        mode,
        lattice_mm: prm.lattice_mm.clamp(2.0, 30.0),
        reach_mm: prm.reach_mm.clamp(5.0, 200.0),
        residual: prm.residual.then(wp::default_residual),
        wepl: prm.wepl.then_some(wp::WeplBeam {
            gantry_deg: prm.gantry_deg,
            couch_deg: prm.couch_deg,
            hlut: None,
        }),
        smooth_chain: prm.smooth_chain,
        ..PostureRequest::default()
    })
}

pub(in super::super) fn posture(
    ctx: &mut Ctx,
    node: &Node,
    images: &[&Value],
    prm: &cat::Posture,
    p: &Progress,
) -> Result<Done> {
    let req = request(prm)?;
    // A lean study per state: the series wired in, its volume, its set.
    let mut studies: Vec<(String, LoadedStudy)> = Vec::new();
    for v in images {
        let (ds, uid) = image_of(v)?;
        let vol = ctx.volume(ds, &uid, p)?;
        let d = ctx.ds(ds)?;
        let series = d
            .study
            .series
            .iter()
            .find(|s| s.uid == uid)
            .cloned()
            .ok_or_else(|| anyhow!("internal: no series {uid}"))?;
        let set = d
            .study
            .structure_sets
            .iter()
            .find(|ss| ss.referenced_series_uid == uid)
            .or_else(|| d.study.structure_sets.first())
            .cloned();
        studies.push((
            format!("{} {}", d.label, series.description)
                .trim()
                .to_string(),
            LoadedStudy {
                meta: d.study.meta.clone(),
                series: vec![series],
                active_series: 0,
                volume: vol,
                structure_sets: set.into_iter().collect(),
                default_window: d.study.default_window,
                ..LoadedStudy::default()
            },
        ));
    }
    if studies.len() < 2 {
        bail!("the step needs at least two images");
    }
    let states: Vec<StateInput> = studies
        .iter()
        .map(|(label, s)| StateInput {
            label: label.clone(),
            study: s,
            structure_set: None,
            posture: None,
        })
        .collect();
    let o = wp::build(&states, &req, p)?;
    let t = &o.tables;

    let mut bodies = Table::new(
        "Bones",
        &[
            "Body",
            "Angle (deg)",
            "Axis RL",
            "Axis AP",
            "Axis SI",
            "Travel (mm)",
            "Fit (mm)",
            "Stretch",
        ],
    );
    for b in &t.bodies {
        bodies.row(vec![
            b.name.clone(),
            r2(b.screw.angle_deg),
            r2(b.screw.axis.x),
            r2(b.screw.axis.y),
            r2(b.screw.axis.z),
            r2(b.screw.travel_mm),
            if b.anchor {
                "anchor".into()
            } else {
                r2(b.fit_mean_mm)
            },
            format!("{:.4}", b.max_stretch),
        ]);
    }
    let mut joints = Table::new("Joints", &["Joint", "RL (deg)", "AP (deg)", "SI (deg)"]);
    for j in &t.joints {
        let e = j.at.last().copied().unwrap_or_default();
        joints.row(vec![j.name.clone(), r1(e[0]), r1(e[1]), r1(e[2])]);
    }
    let mut structures = Table::new(
        "Structures",
        &[
            "Structure",
            "Used in the fit",
            "Volume change (cm3)",
            "Contoured shift (mm)",
            "Model shift (mm)",
            "Rotation (deg)",
            "Check Dice",
            "Check HD95 (mm)",
            "WEPL first (mm)",
            "WEPL change (mm)",
        ],
    );
    for s in &t.structures {
        let (dv, _) = s.volume_change();
        let end = s.at.last().copied().unwrap_or_default();
        structures.row(vec![
            s.name.clone(),
            if s.used { "yes" } else { "no" }.into(),
            r2(dv),
            s.contoured_shift()
                .map_or_else(String::new, |v| r2(v.length())),
            s.model_shift(s.at.len().saturating_sub(1))
                .map_or_else(String::new, |v| r2(v.length())),
            r2(end.rotation_deg),
            s.check
                .map_or_else(String::new, |c| format!("{:.3}", c.dice)),
            s.check.map_or_else(String::new, |c| r2(c.hd95_mm)),
            s.wepl_mm.first().map_or_else(String::new, |v| r1(*v)),
            s.wepl_change(s.wepl_mm.len().saturating_sub(1))
                .map_or_else(String::new, r1),
        ]);
    }
    let m = &t.model;
    let mut model = Table::new("Model", &["", ""]);
    for (k, v) in [
        ("Mode", m.mode.label().to_string()),
        ("Anchor", m.anchor.clone()),
        ("Lattice (mm)", r1(m.lattice_mm)),
        (
            "Inverse consistency at the end, p95 (mm)",
            r2(m.inverse_mm[1][1]),
        ),
        ("Smallest Jacobian", format!("{:.3}", m.jacobian_min)),
        ("Folded fraction", format!("{:.5}", m.folded_fraction)),
    ] {
        model.row(vec![k.to_string(), v]);
    }
    let mut notes = vec![
        format!(
            "The path from {}.",
            states
                .iter()
                .map(|s| s.label.as_str())
                .collect::<Vec<_>>()
                .join(" to ")
        ),
        wp::CAVEAT.to_string(),
    ];
    notes.extend(o.warnings.iter().cloned());
    notes.extend(m.provenance.iter().cloned());
    let report = ctx.add_report(Report {
        node: node.id,
        title: format!("Posture model: {}", node.label()),
        notes,
        tables: vec![bodies, joints, structures, model],
        motion: None,
    });
    let mut lines = Vec::new();
    for b in t.bodies.iter().filter(|b| !b.anchor) {
        lines.push(format!("{}: {:.1} deg", b.name, b.screw.angle_deg));
    }
    lines.push(format!(
        "{} structures, Jacobian min {:.2}",
        t.structures.len(),
        m.jacobian_min
    ));
    let mut d = done(vec![Value::Report(report)], lines);
    d.report = Some(report);
    Ok(d)
}
