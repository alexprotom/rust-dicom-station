//! The posture model between datasets ([`crate::workflow::posture`]): the
//! tables, and on request the files, never an image or an identifier.
//!
//! The states are open datasets, each by its handle, in order; each stands
//! for its displayed series and the structure set drawn on it. What comes
//! back is the four tables of the posture window (states, structures,
//! bones and joints, the model and its checks) as numbers, every name
//! through [`clean_text`] and the whole answer through the redactor like
//! every other tool's; the states are named by their handles. With `write`,
//! the tables (CSV), the model (`.rdsposture`) and the whole path as a
//! DICOM Deformable Spatial Registration go into a folder of the session's
//! output folder.

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::Core;
use super::session::{round2, round3, vec3};
use crate::posture::{file as pfile, Anchor, Mode};
use crate::progress::Progress;
use crate::workflow::posture::{self as wp, Bodies, PostureRequest, StateInput};

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PostureArgs {
    /// The states, in order: dataset handles (each its displayed series
    /// and the structure set drawn on it). Two or more; more make a chain.
    pub datasets: Vec<String>,
    /// What is held still: a structure contoured on every state (its
    /// name), `image` (a rigid registration of the images, the default) or
    /// `scanner` (nothing: for scans that share a frame of reference).
    #[serde(default)]
    pub anchor: Option<String>,
    /// The rigid bodies by name. Omitted: the bones every state has (and
    /// the anchor); an empty list: none.
    #[serde(default)]
    pub bodies: Option<Vec<String>>,
    /// `diffeomorphic` (rigid bodies on screws, invertible) or `linear`
    /// (straight lines on the end-to-end displacement).
    #[serde(default)]
    pub mode: Option<String>,
    /// Add a B-spline residual deformation on top of the bodies.
    #[serde(default)]
    pub residual: bool,
    /// Velocity lattice, mm (6).
    #[serde(default)]
    pub lattice_mm: Option<f64>,
    /// Analysis grid, mm (2).
    #[serde(default)]
    pub analysis_mm: Option<f64>,
    /// Reach of a body in the blend, mm (40).
    #[serde(default)]
    pub reach_mm: Option<f64>,
    /// A posture parameter per state (an angle, a day, a filling).
    #[serde(default)]
    pub posture: Option<Vec<f64>>,
    /// Joints to report: [name, first body, second body]. Empty: the
    /// presets (hips, lumbosacral, neighbouring vertebrae; two bodies and
    /// nothing else, the two).
    #[serde(default)]
    pub joints: Vec<[String; 3]>,
    /// The water-equivalent path length to each structure's centroid along
    /// a beam at this IEC 61217 gantry angle, degrees (with
    /// `wepl_couch_deg`, read in the first state's patient position):
    /// first, last and its change per structure. Omitted with the couch
    /// angle: no WEPL.
    #[serde(default)]
    pub wepl_gantry_deg: Option<f64>,
    /// The patient support angle of that beam, degrees (0).
    #[serde(default)]
    pub wepl_couch_deg: Option<f64>,
    /// A chain of three or more: rigid bodies on curves through the
    /// states, velocities continuous at each (true), or a constant turn per
    /// segment (false).
    #[serde(default)]
    pub smooth_chain: Option<bool>,
    /// Also write tables.csv, the model file and the deformation as DICOM.
    #[serde(default)]
    pub write: bool,
    /// Folder under the session's output folder for `write`.
    #[serde(default)]
    pub folder: Option<String>,
}

pub fn build_posture_model(core: &mut Core, a: PostureArgs, p: &Progress) -> Result<Value> {
    if a.datasets.len() < 2 {
        bail!("a posture model needs at least two datasets");
    }
    let mode = match a.mode.as_deref() {
        None => Mode::Diffeomorphic,
        Some(m) => Mode::from_key(m)
            .ok_or_else(|| anyhow::anyhow!("mode must be diffeomorphic or linear (got '{m}')"))?,
    };
    let req = PostureRequest {
        anchor: a
            .anchor
            .as_deref()
            .map(Anchor::parse)
            .unwrap_or(Anchor::Image),
        bodies: match &a.bodies {
            None => Bodies::Bones,
            Some(v) if v.is_empty() => Bodies::None,
            Some(v) => Bodies::Named(v.clone()),
        },
        mode,
        lattice_mm: a.lattice_mm.unwrap_or(6.0).clamp(2.0, 30.0),
        analysis_mm: a.analysis_mm.unwrap_or(2.0).clamp(0.5, 10.0),
        reach_mm: a.reach_mm.unwrap_or(40.0).clamp(5.0, 200.0),
        residual: a.residual.then(wp::default_residual),
        joints: a
            .joints
            .iter()
            .map(|[n, x, y]| (n.clone(), x.clone(), y.clone()))
            .collect(),
        wepl: (a.wepl_gantry_deg.is_some() || a.wepl_couch_deg.is_some()).then(|| wp::WeplBeam {
            gantry_deg: a.wepl_gantry_deg.unwrap_or(0.0),
            couch_deg: a.wepl_couch_deg.unwrap_or(0.0),
            hlut: None,
        }),
        smooth_chain: a.smooth_chain.unwrap_or(true),
        ..PostureRequest::default()
    };
    let outcome = {
        let mut states = Vec::with_capacity(a.datasets.len());
        for (i, id) in a.datasets.iter().enumerate() {
            let ds = core.session.dataset(id)?;
            states.push(StateInput {
                label: id.clone(),
                study: &ds.study,
                structure_set: None,
                posture: a.posture.as_ref().and_then(|v| v.get(i).copied()),
            });
        }
        wp::build(&states, &req, p)?
    };
    let t = &outcome.tables;
    let mut out = json!({
        "states": t.states.iter().map(|s| json!({
            "dataset": s.label,
            "posture": s.posture,
            "position": s.position,
            "grid": s.dims,
            "into_anchor": {
                "translation_mm": vec3(s.to_anchor.translation),
                "rotation_deg": s.to_anchor.rotation_deg.map(round2),
                "anchor_fit_rms_mm": round2(s.to_anchor.residual_mm),
            },
            "body_cm3": s.body_cm3.map(round2),
            "body_change_cm3": s.body_change_cm3.map(round2),
        })).collect::<Vec<_>>(),
        "structures": t.structures.iter().map(|s| {
            let (dv, pct) = s.volume_change();
            let end = s.at.last().copied().unwrap_or_default();
            json!({
                "structure": clean_text(&s.name),
                "used_in_fit": s.used,
                "volume_cm3": s.volume_cm3.iter().map(|v| round2(*v)).collect::<Vec<_>>(),
                "volume_change_cm3": round2(dv),
                "volume_change_pct": if pct.is_finite() { Some(round2(pct)) } else { None },
                "cut_by_a_scan": s.is_cut(),
                "volume_change_common_fov_cm3": round2(s.common_change()),
                "contoured_shift_mm": s.contoured_shift().map(vec3),
                "model_shift_mm": s.model_shift(s.at.len().saturating_sub(1)).map(vec3),
                "path_mm": round2(end.path_mm),
                "rotation_deg": round2(end.rotation_deg),
                "jacobian": end.jacobian.map(round3),
                "after_rigid": s.deformation.map(|d| json!({"dice": round3(d.dice), "hd95_mm": round2(d.hd95_mm), "msd_mm": round2(d.msd_mm)})),
                "model_check": s.check.map(|d| json!({"dice": round3(d.dice), "hd95_mm": round2(d.hd95_mm), "msd_mm": round2(d.msd_mm), "centroid_mm": round2(d.centroid_mm)})),
                "mean_hu": s.mean_hu.iter().map(|v| v.map(round2)).collect::<Vec<_>>(),
                "wepl_mm": (!s.wepl_mm.is_empty()).then(|| json!({
                    "first": s.wepl_mm.first().map(|v| round2(*v)),
                    "last": s.wepl_mm.last().map(|v| round2(*v)),
                    "change": s.wepl_change(s.wepl_mm.len() - 1).map(round2),
                })),
            })
        }).collect::<Vec<_>>(),
        "bodies": t.bodies.iter().map(|b| json!({
            "body": clean_text(&b.name),
            "anchor": b.anchor,
            "angle_deg": round2(b.screw.angle_deg),
            "axis": vec3(b.screw.axis),
            "pivot_mm": vec3(b.screw.point),
            "travel_mm": round2(b.screw.travel_mm),
            "fit_mean_mm": round2(b.fit_mean_mm),
            "principal_axes_start": b.seeded,
            "largest_stretch": round3(b.max_stretch),
            "stretch_p95": round3(b.stretch_p95),
        })).collect::<Vec<_>>(),
        "joints": t.joints.iter().map(|j| json!({
            "joint": clean_text(&j.name),
            "first": clean_text(&j.first),
            "second": clean_text(&j.second),
            "angles_deg_rl_ap_si": j.at.last().copied().unwrap_or_default().map(round2),
        })).collect::<Vec<_>>(),
        "model": {
            "mode": t.model.mode.key(),
            "anchor": clean_text(&t.model.anchor),
            "lattice_mm": round2(t.model.lattice_mm),
            "inverse_consistency_mm": {
                "half_way_max": round3(t.model.inverse_mm[0][0]),
                "half_way_p95": round3(t.model.inverse_mm[0][1]),
                "end_max": round3(t.model.inverse_mm[1][0]),
                "end_p95": round3(t.model.inverse_mm[1][1]),
            },
            "jacobian_min": round3(t.model.jacobian_min),
            "folded_fraction": t.model.folded_fraction,
            "check_dice_used": t.model.check_used_dice.map(round3),
            "check_dice_not_used": t.model.check_unused_dice.map(round3),
            "overlap_fraction": round3(t.model.overlap_fraction),
            "wepl_beam": t.model.beam,
            "provenance": t.model.provenance.iter().map(|l| clean_text(l)).collect::<Vec<_>>(),
        },
        "warnings": outcome.warnings.iter().map(|w| clean_text(w)).collect::<Vec<_>>(),
        "caveat": wp::CAVEAT,
    });
    if a.write {
        let name = a
            .folder
            .clone()
            .unwrap_or_else(|| format!("posture-{}", a.datasets.join("-")));
        let dir = core.session.fresh_out_subdir(&name)?;
        std::fs::create_dir_all(&dir)?;
        std::fs::write(
            dir.join("tables.csv"),
            crate::motion::csv_file_bytes(&t.csv()),
        )?;
        pfile::write(
            &dir.join(format!("model.{}", pfile::EXTENSION)),
            &outcome.model,
            &outcome.occupancy,
        )?;
        let first = core.session.dataset(&a.datasets[0])?;
        let last = core
            .session
            .dataset(a.datasets.last().expect("two or more"))?;
        let last_for = last.study.volume.frame_of_reference_uid.clone();
        wp::export_registration(
            &outcome.model,
            &first.study,
            &last_for,
            &dir.join("posture_reg.dcm"),
        )?;
        out["folder"] = json!(dir.to_string_lossy());
        out["files"] = json!([
            "tables.csv",
            format!("model.{}", pfile::EXTENSION),
            "posture_reg.dcm"
        ]);
    }
    Ok(out)
}
