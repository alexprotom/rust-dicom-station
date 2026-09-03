//! 4D: groups, one series onto every phase, and the motion pipeline.

use std::sync::Arc;

use anyhow::{bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::phi::clean_text;
use super::super::session::{GroupRegistration, PhaseReg, Run};
use super::super::Core;
use super::register::{parse_method, StructureArg};
use super::session::{round1, round2, round3, vec3, DatasetArgs};
use crate::fourd::Role;
use crate::motion::{MotionModel, MotionReport};
use crate::progress::Progress;
use crate::registration::{RegMethod, RegParams};
use crate::workflow::{self, anchored, group, motion, select};

pub fn list_4d_groups(core: &mut Core, a: DatasetArgs, _p: &Progress) -> Result<Value> {
    let ds = core.session.dataset(&a.dataset)?;
    let groups: Vec<Value> = ds
        .study
        .fourd_groups
        .iter()
        .enumerate()
        .filter(|(_, g)| !g.dissolved)
        .map(|(i, g)| {
            let series_no = |uid: &str| {
                ds.study
                    .series
                    .iter()
                    .position(|s| s.uid == uid)
                    .map(|x| x + 1)
            };
            json!({
                "index": i + 1,
                "name": clean_text(&g.name),
                "members": g.members.iter().map(|m| json!({
                    "label": clean_text(&m.label),
                    "role": match m.role { Role::Phase => "phase", _ => "reconstruction" },
                    "percent": m.percent,
                    "series": series_no(&m.series_uid),
                })).collect::<Vec<_>>(),
                "default_reference": g.default_reference().map(|m| clean_text(&g.members[m].label)),
            })
        })
        .collect();
    Ok(json!({ "dataset": ds.id, "groups": groups }))
}

/// The group a call names, by 1-based index or by name.
fn group_index(ds: &super::super::session::Dataset, group: &str) -> Result<usize> {
    let groups = &ds.study.fourd_groups;
    if let Ok(n) = group.parse::<usize>() {
        if n >= 1 && n <= groups.len() && !groups[n - 1].dissolved {
            return Ok(n - 1);
        }
    }
    groups
        .iter()
        .position(|g| !g.dissolved && g.name.eq_ignore_ascii_case(group))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no 4D group '{group}' in {}; there are {} (see list_4d_groups)",
                ds.id,
                groups.iter().filter(|g| !g.dissolved).count()
            )
        })
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GroupArgs {
    /// Dataset holding the 4D group.
    pub dataset: String,
    /// Group index (from list_4d_groups) or name.
    pub group: String,
    /// The series the structures were drawn on (the moving image). Defaults
    /// to the displayed series of `source_dataset`.
    #[serde(default)]
    pub source_dataset: Option<String>,
    #[serde(default)]
    pub source_series: Option<u32>,
    /// Structures of the source to carry onto every phase. Empty: register
    /// only.
    #[serde(default)]
    pub structures: Vec<StructureArg>,
    /// `elastix_bspline` (default) or `plastimatch_bspline`; a rigid method
    /// is refused, the phases differ by breathing. With an `anchor` this is
    /// the refinement stage's method.
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub levels: Option<usize>,
    #[serde(default)]
    pub iterations: Option<usize>,
    #[serde(default)]
    pub samples: Option<usize>,
    #[serde(default)]
    pub grid_spacing_mm: Option<f64>,
    /// A structure contoured on the source *and* on every phase (in each
    /// phase's own structure set), for example the heart. The run is then
    /// anchored on it: centroids matched first (the two images need not
    /// overlap at all), a rigid registration sampling only the structure
    /// plus `anchor_margin_mm`, then a local deformable refinement unless
    /// `rigid_only`. The structure travels with the others and its overlap
    /// with the phase's own contour is reported as the check. This is the
    /// way to bring a cardiac CT onto a 4DCT.
    #[serde(default)]
    pub anchor: Option<StructureArg>,
    /// Dilation of the anchor that bounds the registration, mm.
    #[serde(default = "default_anchor_margin")]
    pub anchor_margin_mm: f64,
    /// Anchored run: stop after the rigid stage.
    #[serde(default)]
    pub rigid_only: bool,
    /// Where the propagated structures are filed on each phase:
    /// `segmentation` (default; a new segmentation series bound to the
    /// phase) or `structure_set` (contours appended to the phase's own RT
    /// structure set, the one that references the phase's series; a new set
    /// bound to it when there is none).
    #[serde(default)]
    pub land: Option<String>,
    /// Anchored run: what the registration compares. `contours` (default)
    /// matches the anchor's surfaces through their signed distance maps,
    /// indifferent to contrast agent and cardiac phase; `intensity` matches
    /// the images inside the region.
    #[serde(default)]
    pub anchor_by: Option<String>,
}

fn default_anchor_margin() -> f64 {
    10.0
}

pub fn propagate_to_group(core: &mut Core, a: GroupArgs, p: &Progress) -> Result<Value> {
    let src_ds_id = a
        .source_dataset
        .clone()
        .unwrap_or_else(|| a.dataset.clone());
    let (gi, phases, group_name, moving_idx) = {
        let ds = core.session.dataset(&a.dataset)?;
        let gi = group_index(ds, &a.group)?;
        let g = &ds.study.fourd_groups[gi];
        let phases = workflow::phases_of(g, &ds.study.series)?;
        let sds = core.session.dataset(&src_ds_id)?;
        let moving_idx = core.session.series_index(sds, a.source_series)?;
        (gi, phases, g.name.clone(), moving_idx)
    };
    let src_vol = core.session.volume(&src_ds_id, moving_idx, p)?;
    let src_grid = src_vol.grid();
    let mut subjects = Vec::new();
    for s in &a.structures {
        let st = core
            .session
            .structure(&src_ds_id, &s.structure, s.set.as_deref())?;
        subjects.push(st.subject_on(&src_grid)?);
    }
    let land = parse_landing(a.land.as_deref())?;
    let method = match &a.method {
        Some(m) => parse_method(m)?,
        None => RegMethod::ElastixBSpline,
    };
    if !method.is_deformable() {
        bail!("a run onto a 4D group needs a deformable method: the phases differ by breathing");
    }
    let mut params = RegParams {
        method,
        ..RegParams::default()
    };
    if let Some(v) = a.levels {
        params.levels = v.clamp(1, 6);
    }
    if let Some(v) = a.iterations {
        params.iterations = v.clamp(1, 5000);
    }
    if let Some(v) = a.samples {
        params.samples = v.clamp(100, 200_000);
    }
    if let Some(v) = a.grid_spacing_mm {
        params.grid_spacing_mm = v.clamp(4.0, 200.0);
    }
    let moving_uid = core.session.dataset(&src_ds_id)?.study.series[moving_idx]
        .uid
        .clone();
    if let Some(anchor) = &a.anchor {
        return propagate_anchored(
            core, &a, anchor, src_ds_id, gi, phases, group_name, moving_idx, moving_uid, src_vol,
            subjects, params, p,
        );
    }
    // Transforms already made for this group from this moving series.
    let cached: Vec<Option<Arc<crate::registration::Transform3>>> =
        match core.session.group_registrations.iter().find(|g| {
            g.dataset == a.dataset && g.group == gi && g.moving == (src_ds_id.clone(), moving_idx)
        }) {
            Some(gr) => phases
                .iter()
                .map(|(_, se)| {
                    gr.phases
                        .iter()
                        .find(|ph| ph.series_uid == se.uid)
                        .map(|ph| ph.transform.clone())
                })
                .collect(),
            None => vec![None; phases.len()],
        };
    let reused = cached.iter().filter(|c| c.is_some()).count();
    let req = group::GroupRequest {
        src_vol,
        subjects,
        phases,
        cached,
        params,
        group_name: group_name.clone(),
        group: gi,
        moving_slot: 0,
        moving_series_uid: moving_uid,
    };
    let out = group::run(req, p)?;

    // File the results and keep the transforms.
    let mut phase_reports = Vec::new();
    let mut regs = Vec::new();
    for ph in &out.phases {
        let items: Vec<Value> = ph
            .items
            .iter()
            .map(|it| {
                json!({
                    "structure": clean_text(&it.name),
                    "source_cm3": round2(it.source_cm3),
                    "mapped_cm3": round2(it.mapped_cm3),
                    "result_cm3": round2(it.result_cm3),
                    "voxels": it.voxels,
                })
            })
            .collect();
        let set = file_phase(core, &a.dataset, &group_name, ph, land)?;
        phase_reports.push(json!({
            "phase": clean_text(&ph.label),
            "registration": ph.metric_line,
            "set": clean_text(&set),
            "landed_as": land.label(),
            "structures": items,
        }));
        regs.push(PhaseReg {
            label: ph.label.clone(),
            series_uid: ph.series_uid.clone(),
            transform: ph.transform.clone(),
            metric_line: ph.metric_line.clone(),
        });
    }
    core.session.group_registrations.retain(|g| {
        !(g.dataset == a.dataset && g.group == gi && g.moving == (src_ds_id.clone(), moving_idx))
    });
    let id = core.session.mint("greg");
    core.session.group_registrations.push(GroupRegistration {
        id: id.clone(),
        dataset: a.dataset.clone(),
        group: gi,
        group_name: group_name.clone(),
        moving: (src_ds_id.clone(), moving_idx),
        phases: regs,
    });
    Ok(json!({
        "greg": id,
        "dataset": a.dataset,
        "group": clean_text(&group_name),
        "source": { "dataset": src_ds_id, "series": moving_idx + 1 },
        "transforms_reused": reused,
        "phases": phase_reports,
    }))
}

fn parse_landing(s: Option<&str>) -> Result<group::Landing> {
    Ok(match s.map(str::trim) {
        None | Some("") | Some("segmentation") => group::Landing::Segmentation,
        Some("structure_set") | Some("rtstruct") => group::Landing::StructureSet,
        Some(other) => bail!("land must be segmentation or structure_set (got '{other}')"),
    })
}

/// File one phase's propagated structures on the dataset, the way `land`
/// asks; returns the label of the set they went into.
fn file_phase(
    core: &mut Core,
    dataset: &str,
    group_name: &str,
    ph: &group::PhaseOutcome,
    land: group::Landing,
) -> Result<String> {
    let ds = core.session.dataset_mut(dataset)?;
    Ok(match land {
        group::Landing::Segmentation => match ph.seg_series(group_name) {
            Some(series) => {
                let label = series.label.clone();
                ds.study.seg_series.push(series);
                label
            }
            None => String::new(),
        },
        group::Landing::StructureSet => group::land_in_structure_set(
            &mut ds.study,
            &ph.series_uid,
            &ph.study_uid,
            &ph.grid,
            &ph.items,
            &format!("{} {}", group_name, ph.label),
        )
        .map(|(label, _)| label)
        .unwrap_or_default(),
    })
}

/// The anchored variant of [`propagate_to_group`]: see
/// [`crate::workflow::anchored`].
#[allow(clippy::too_many_arguments)]
fn propagate_anchored(
    core: &mut Core,
    a: &GroupArgs,
    anchor: &StructureArg,
    src_ds_id: String,
    gi: usize,
    phases: Vec<(String, crate::loader::SeriesInfo)>,
    group_name: String,
    moving_idx: usize,
    moving_uid: String,
    src_vol: Arc<crate::volume::Volume>,
    subjects: Vec<crate::propagate::Subject>,
    params: RegParams,
    p: &Progress,
) -> Result<Value> {
    let src_anchor =
        core.session
            .structure(&src_ds_id, &anchor.structure, anchor.set.as_deref())?;
    // The anchor on every phase: the contour drawn on that phase's series.
    let mut anchored = Vec::with_capacity(phases.len());
    {
        let ds = core.session.dataset(&a.dataset)?;
        for (label, series) in phases {
            let Some(st) = select::find_on_series(&ds.study, &anchor.structure, &series.uid, "")
            else {
                bail!(
                    "no structure '{}' for phase '{}' of {}; an anchored run needs it contoured \
                     on every phase",
                    anchor.structure,
                    label,
                    a.dataset
                );
            };
            anchored.push(anchored::AnchoredPhase {
                label,
                series,
                anchor: st,
            });
        }
    }
    let land = parse_landing(a.land.as_deref())?;
    let mode = match a.anchor_by.as_deref().map(str::trim) {
        None | Some("") | Some("contours") => anchored::AnchorMode::Contours,
        Some("intensity") => anchored::AnchorMode::Intensity,
        Some(other) => bail!("anchor_by must be contours or intensity (got '{other}')"),
    };
    let rigid = anchored::default_rigid(&params);
    let deformable = (!a.rigid_only).then(|| anchored::default_deformable(&params));
    let req = anchored::AnchoredRequest {
        src_vol,
        src_anchor,
        subjects,
        phases: anchored,
        margin_mm: a.anchor_margin_mm,
        mode,
        rigid,
        deformable,
        group_name: group_name.clone(),
        group: gi,
        moving_slot: 0,
        moving_series_uid: moving_uid,
    };
    let out = anchored::run(req, p)?;

    let mut phase_reports = Vec::new();
    let mut regs = Vec::new();
    for (ph, qa) in out.group.phases.iter().zip(&out.qa) {
        let items: Vec<Value> = ph
            .items
            .iter()
            .map(|it| {
                json!({
                    "structure": clean_text(&it.name),
                    "source_cm3": round2(it.source_cm3),
                    "mapped_cm3": round2(it.mapped_cm3),
                    "result_cm3": round2(it.result_cm3),
                    "voxels": it.voxels,
                })
            })
            .collect();
        let set = file_phase(core, &a.dataset, &group_name, ph, land)?;
        phase_reports.push(json!({
            "phase": clean_text(&ph.label),
            "registration": ph.metric_line,
            "set": clean_text(&set),
            "landed_as": land.label(),
            "structures": items,
            "anchor_check": {
                "anchor": clean_text(&qa.anchor),
                "initial_shift_mm": round1(qa.initial_shift_mm),
                "rigid": qa.rigid_line,
                "deformable": qa.deformable_line,
                "displacement_p95_mm": qa.displacement_p95_mm.map(round2),
                "folded_fraction": qa.folded_fraction.map(round3),
                "dice": qa.overlap.as_ref().map(|o| round3(o.dice)),
                "hd95_mm": qa.overlap.as_ref().map(|o| round2(o.hd95_mm)),
                "mean_surface_distance_mm": qa.overlap.as_ref().map(|o| round2(o.msd_mm)),
                "centroid_shift_mm": qa.overlap.as_ref().and_then(|o| o.centroid_shift()).map(vec3),
                "verdict": qa.verdict(),
            },
        }));
        regs.push(PhaseReg {
            label: ph.label.clone(),
            series_uid: ph.series_uid.clone(),
            transform: ph.transform.clone(),
            metric_line: ph.metric_line.clone(),
        });
    }
    core.session.group_registrations.retain(|g| {
        !(g.dataset == a.dataset && g.group == gi && g.moving == (src_ds_id.clone(), moving_idx))
    });
    let id = core.session.mint("greg");
    core.session.group_registrations.push(GroupRegistration {
        id: id.clone(),
        dataset: a.dataset.clone(),
        group: gi,
        group_name: group_name.clone(),
        moving: (src_ds_id.clone(), moving_idx),
        phases: regs,
    });
    let worst = out
        .qa
        .iter()
        .filter_map(|q| q.overlap.as_ref().map(|o| o.dice))
        .fold(f64::NAN, f64::min);
    Ok(json!({
        "greg": id,
        "dataset": a.dataset,
        "group": clean_text(&group_name),
        "source": { "dataset": src_ds_id, "series": moving_idx + 1 },
        "anchor": clean_text(&anchor.structure),
        "anchor_by": mode.label(),
        "stages": if a.rigid_only { "centroids, rigid" } else { "centroids, rigid, deformable" },
        "worst_anchor_dice": if worst.is_nan() { None } else { Some(round3(worst)) },
        "phases": phase_reports,
        "note": "each phase's transform maps the phase onto the source; the anchor's Dice \
                 against the phase's own contour is the check on everything that travelled \
                 with it",
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MotionArgs {
    pub dataset: String,
    /// Group index or name.
    pub group: String,
    /// Reference phase label (for example `0%`); the group's default when
    /// omitted.
    #[serde(default)]
    pub reference_phase: Option<String>,
    /// Target structures, on the reference phase.
    pub targets: Vec<StructureArg>,
    /// Structure whose motion the targets are compared with (typically the
    /// heart).
    #[serde(default)]
    pub reference_structure: Option<StructureArg>,
    #[serde(default = "yes")]
    pub rigid: bool,
    #[serde(default = "yes")]
    pub deformable: bool,
    #[serde(default = "yes")]
    pub build_itv: bool,
    #[serde(default)]
    pub itv_margin_mm: f64,
    /// Also file every propagated per-phase mask on its phase.
    #[serde(default)]
    pub keep_phase_segs: bool,
    #[serde(default)]
    pub levels: Option<usize>,
    #[serde(default)]
    pub iterations: Option<usize>,
    #[serde(default)]
    pub samples: Option<usize>,
    #[serde(default)]
    pub grid_spacing_mm: Option<f64>,
    #[serde(default)]
    pub fixed_threshold: Option<f32>,
}

fn yes() -> bool {
    true
}

/// The report as JSON.
pub fn report_json(r: &MotionReport) -> Value {
    let track = |t: &crate::motion::Track| {
        json!({
            "target": clean_text(&t.target),
            "model": t.model.label(),
            "reference_phase": t.samples.get(t.reference).map(|s| clean_text(&s.phase)),
            "peak_to_peak_mm": round2(t.peak_to_peak()),
            "phases": t.samples.iter().map(|s| json!({
                "phase": clean_text(&s.phase),
                "centroid_mm": vec3(s.centroid),
                "volume_cm3": round1(s.volume_cm3),
            })).collect::<Vec<_>>(),
            "displacements_mm": t.displacements().into_iter().map(vec3).collect::<Vec<_>>(),
        })
    };
    json!({
        "run_name": clean_text(&r.run_name),
        "phases": r.phases.iter().map(|s| clean_text(s)).collect::<Vec<_>>(),
        "reference_phase": clean_text(&r.reference),
        "reference_structure": r.reference_structure.as_deref().map(clean_text),
        "tracks": r.tracks.iter().map(track).collect::<Vec<_>>(),
        "reference_tracks": r.reference_tracks.iter().map(track).collect::<Vec<_>>(),
        "correlations": r.correlations.iter().map(|(t, m, axes)| json!({
            "target": clean_text(t),
            "model": m.label(),
            "axes": axes.iter().map(|a| json!({
                "axis": a.axis,
                "r": round3(a.r),
                "p": round3(a.p),
                "synchrony": crate::motion::synchrony_level(a.r),
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "qa": r.qa.iter().map(|q| json!({
            "phase": clean_text(&q.phase),
            "model": q.model.label(),
            "registration": q.metric_line,
            "folding_pct": round2(q.folding_pct),
            "displacement_p95_mm": round2(q.disp_p95_mm),
        })).collect::<Vec<_>>(),
        "itvs": r.itvs.iter().map(|i| json!({
            "target": clean_text(&i.target),
            "model": i.model.label(),
            "margin_mm": i.margin_mm,
            "volume_cm3": round1(i.volume_cm3),
            "structure": clean_text(&i.seg_name),
        })).collect::<Vec<_>>(),
    })
}

pub fn analyse_motion(core: &mut Core, a: MotionArgs, p: &Progress) -> Result<Value> {
    if a.targets.is_empty() {
        bail!("name at least one target structure");
    }
    let (gi, phases, group_name, study_uid, reference) = {
        let ds = core.session.dataset(&a.dataset)?;
        let gi = group_index(ds, &a.group)?;
        let g = &ds.study.fourd_groups[gi];
        let phases = workflow::phases_of(g, &ds.study.series)?;
        let reference = match &a.reference_phase {
            Some(label) => phases
                .iter()
                .position(|(l, _)| l.eq_ignore_ascii_case(label))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "no phase '{label}' in the group; the phases are {}",
                        phases
                            .iter()
                            .map(|(l, _)| l.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?,
            None => {
                let m = g.default_reference().unwrap_or(0);
                g.phase_members().iter().position(|&x| x == m).unwrap_or(0)
            }
        };
        (gi, phases, g.name.clone(), g.study_uid.clone(), reference)
    };
    let _ = gi;
    let mut targets = Vec::new();
    for t in &a.targets {
        targets.push(
            core.session
                .structure(&a.dataset, &t.structure, t.set.as_deref())?,
        );
    }
    let ref_struct = match &a.reference_structure {
        Some(s) => Some(
            core.session
                .structure(&a.dataset, &s.structure, s.set.as_deref())?,
        ),
        None => None,
    };
    let mut models = Vec::new();
    if a.rigid {
        models.push(MotionModel::Rigid);
    }
    if a.deformable {
        models.push(MotionModel::Deformable);
    }
    let mut params = RegParams {
        method: RegMethod::ElastixRigid,
        ..RegParams::default()
    };
    if let Some(v) = a.levels {
        params.levels = v.clamp(1, 6);
    }
    if let Some(v) = a.iterations {
        params.iterations = v.clamp(1, 5000);
    }
    if let Some(v) = a.samples {
        params.samples = v.clamp(100, 200_000);
    }
    if let Some(v) = a.grid_spacing_mm {
        params.grid_spacing_mm = v.clamp(4.0, 200.0);
    }
    if let Some(v) = a.fixed_threshold {
        params.fixed_threshold = v;
    }
    let run_no = core.session.runs.len() + 1;
    let req = motion::MotionRequest {
        run_name: format!(
            "#{run_no} {} · {} · ref {}",
            a.dataset, group_name, phases[reference].0
        ),
        slot_name: a.dataset.clone(),
        // The handle, never the name: the report header is part of the CSV.
        patient: a.dataset.clone(),
        group_name: group_name.clone(),
        study_uid,
        phases,
        reference,
        targets,
        ref_struct,
        models,
        build_itv: a.build_itv,
        itv_margin_mm: a.itv_margin_mm.max(0.0),
        keep_phase_segs: a.keep_phase_segs,
        params,
    };
    let out = motion::run(req, p)?;

    // File the segmentations the way the viewer does.
    let mut filed = Vec::new();
    {
        let ds = core.session.dataset_mut(&a.dataset)?;
        if let Some(itv) = out.itv_series {
            filed.push(clean_text(&itv.label));
            ds.study
                .seg_series
                .push(itv.into_seg_series(&out.study_uid));
        }
        for ph in out.phase_series {
            filed.push(clean_text(&ph.label));
            ds.study.seg_series.push(ph.into_seg_series(&out.study_uid));
        }
    }
    let report = report_json(&out.report);
    let id = core.session.mint("run");
    core.session.runs.push(Run {
        id: id.clone(),
        dataset: a.dataset.clone(),
        report: out.report,
    });
    Ok(json!({
        "run": id,
        "dataset": a.dataset,
        "group": clean_text(&group_name),
        "filed_series": filed,
        "report": report,
    }))
}
