//! The per-phase 4D pipeline: register the reference phase to every other
//! phase (rigidly, and deformably on top of the rigid result), propagate the
//! targets through each transform, and measure what comes back - centroid
//! trajectories, amplitudes, drift against a reference structure, per-phase
//! registration quality, and motion-encompassing ITVs on the reference
//! phase. A target that every phase carries in its own right can be read
//! as contoured instead, with no registration at all; the rigid model can be
//! fitted locally, on each structure's own neighbourhood.
//!
//! Moved out of `app/motion.rs`; the viewer's
//! *Structure motion* section and the MCP server both build a
//! [`MotionRequest`] and call [`run`].

use crate::dicomseg::{resample_mask, SegSeries};
use crate::loader::SeriesInfo;
use crate::morphology;
use crate::motion::{
    self, AxisCorrelation, ItvResult, MotionModel, MotionReport, PhaseSample, RegQa, Track,
};
use crate::progress::{self, Progress};
use crate::propagate::{self, BoxedMask, PackedItem, Propagated, Subject};
use crate::registration::{self, RegMethod, RegParams};
use crate::segmentation::Segmentation;
use crate::volume::Grid;

use super::select::Structure;

use anyhow::{anyhow, bail, Result};

/// Everything a run needs, snapshotted when it starts.
pub struct MotionRequest {
    pub run_name: String,
    pub slot_name: String,
    /// Shown in the report's header. The viewer passes the patient name;
    /// the MCP server passes the workspace handle, never the name.
    pub patient: String,
    pub group_name: String,
    pub study_uid: String,
    /// The phase members, in temporal order: label + the series to load.
    pub phases: Vec<(String, SeriesInfo)>,
    /// Index of the reference phase within `phases`.
    pub reference: usize,
    /// The targets as defined on the reference phase (the registration
    /// models carry these across).
    pub targets: Vec<Structure>,
    pub ref_struct: Option<Structure>,
    /// Targets that exist on every phase in their own right: the same name
    /// contoured (or propagated) per phase. The `as contoured` model reads
    /// these instead of registering anything. Each is matched to `targets`
    /// (and the reference structure) by name.
    pub contoured: Vec<ContouredTarget>,
    pub models: Vec<MotionModel>,
    /// The rigid model fitted on the structure's own neighbourhood - the
    /// structure dilated by this margin - rather than on the whole image.
    /// `None` is one global rigid body per phase, which for a breathing
    /// patient is the couch: nothing moves rigidly as a whole.
    pub local_rigid_margin_mm: Option<f64>,
    pub build_itv: bool,
    pub itv_margin_mm: f64,
    pub keep_phase_segs: bool,
    /// Levels, iterations, samples, grid spacing and threshold of the
    /// per-phase runs; the method is set by the pipeline.
    pub params: RegParams,
    /// Where the phases are read from (see
    /// [`crate::workflow::group::GroupRequest::volumes`]).
    pub volumes: crate::workflow::session::Volumes,
}

/// One structure as it exists on every phase, in the phases' order.
pub struct ContouredTarget {
    pub name: String,
    pub color: [u8; 3],
    /// One entry per phase of the request; `None` where the phase has no
    /// contour of it (the model then skips the structure).
    pub phases: Vec<Option<Structure>>,
}

/// One finished segmentation series to add to the study, its masks kept
/// as their boxes until they are filed.
pub struct OutSeries {
    pub label: String,
    pub grid: Grid,
    pub referenced_series_uid: String,
    pub segs: Vec<(String, [u8; 3], BoxedMask)>,
}

impl OutSeries {
    /// File the series the way the viewer does: one `SegSeries` bound to the
    /// image series it belongs to, every mask a segment.
    pub fn into_seg_series(self, study_uid: &str) -> SegSeries {
        let mut ser = SegSeries::new(
            self.label,
            self.grid,
            self.referenced_series_uid,
            study_uid.to_string(),
        );
        for (name, color, mask) in self.segs {
            ser.segs.push(Segmentation::from_label_map(
                name,
                color,
                ser.grid.dims,
                &mask.mask(),
                1,
            ));
        }
        ser
    }
}

/// The same lattice to the bit: resampling a mask from one onto the other
/// takes every voxel to itself (the frame of reference aside, which moves
/// no voxel).
fn same_lattice(a: &Grid, b: &Grid) -> bool {
    a.dims == b.dims
        && a.spacing == b.spacing
        && a.origin == b.origin
        && a.row_dir == b.row_dir
        && a.col_dir == b.col_dir
        && a.normal == b.normal
}

/// What a finished run hands back.
pub struct MotionOutcome {
    pub report: MotionReport,
    /// The ITVs, on the reference phase's lattice.
    pub itv_series: Option<OutSeries>,
    /// Per-phase propagated masks, when the run kept them.
    pub phase_series: Vec<OutSeries>,
    pub study_uid: String,
}

/// The whole per-phase pipeline, on the calling thread.
pub fn run(req: MotionRequest, p: &Progress) -> Result<MotionOutcome> {
    let n = req.phases.len();
    let n_targets = req.targets.len();
    let cancelled = || anyhow!(progress::CANCELLED);
    if n < 2 {
        bail!("the group needs at least two phases");
    }
    if req.reference >= n {
        bail!("the reference must be one of the phases");
    }
    if req.targets.is_empty() {
        bail!("tick at least one target structure");
    }
    if req.models.is_empty() {
        bail!("choose at least one model (rigid / deformable / as contoured)");
    }
    if req.models.contains(&MotionModel::Contoured)
        && !req
            .contoured
            .iter()
            .any(|c| c.phases.iter().all(|s| s.is_some()))
    {
        bail!("the 'as contoured' model needs a target that every phase carries");
    }
    let registers = req.models.iter().any(|m| m.registers());
    if let Some(r) = &req.ref_struct {
        if req.targets.iter().any(|t| t.name == r.name) {
            bail!("'{}' cannot be both target and reference", r.name);
        }
    }

    // The reference phase.
    p.set_phase(0.0, 0.04);
    let ref_vol = req.volumes.load(&req.phases[req.reference].1, p)?;
    let ref_grid = ref_vol.grid();
    // Every registration below is of the reference and the phase in hand:
    // their pyramids are built once and shared (see `PyramidCache`).
    let mut pyramids = registration::PyramidCache::default();

    // All structures on the reference lattice: targets first, then the
    // reference structure.
    let mut subjects: Vec<Subject> = Vec::new();
    for s in req.targets.iter().chain(req.ref_struct.iter()) {
        subjects.push(s.subject_on(&ref_grid)?);
    }
    let n_subjects = subjects.len();

    // samples[model][subject][phase] - filled as the phases are processed.
    let mut samples: Vec<Vec<Vec<Option<PhaseSample>>>> =
        vec![vec![vec![None; n]; n_subjects]; req.models.len()];
    // Union accumulators on the reference grid, [model][target].
    let ref_n = ref_grid.dims[0] * ref_grid.dims[1] * ref_grid.dims[2];
    let mut unions: Vec<Vec<Vec<u8>>> = if req.build_itv {
        vec![vec![vec![0u8; ref_n]; n_targets]; req.models.len()]
    } else {
        Vec::new()
    };
    let mut qa: Vec<RegQa> = Vec::new();
    let mut phase_series: Vec<OutSeries> = Vec::new();

    // The reference phase's own samples (and its contribution to the ITV).
    for (mi, _) in req.models.iter().enumerate() {
        for (si, subject) in subjects.iter().enumerate() {
            let c = motion::centroid_mm(&subject.mask, &ref_grid)
                .ok_or_else(|| anyhow!("'{}' is empty", subject.name))?;
            samples[mi][si][req.reference] = Some(PhaseSample {
                phase: req.phases[req.reference].0.clone(),
                centroid: c,
                volume_cm3: motion::volume_cm3(&subject.mask, &ref_grid),
                grey: motion::grey_stats(&subject.mask, &ref_vol),
            });
            if req.build_itv && si < n_targets {
                motion::union_into(&mut unions[mi][si], &subject.mask);
            }
        }
    }

    // The local rigid model's neighbourhoods, one per subject, made once.
    let mut regions: Vec<Option<Option<std::sync::Arc<registration::RegionMask>>>> =
        vec![None; n_subjects];

    // Every other phase: register, propagate, measure.
    let others: Vec<usize> = (0..n).filter(|&i| i != req.reference).collect();
    for (oi, &pi) in others.iter().enumerate() {
        if p.cancelled() {
            return Err(cancelled());
        }
        let base = 0.05 + 0.9 * oi as f32 / others.len() as f32;
        let span = 0.9 / others.len() as f32;
        let (label, series) = &req.phases[pi];
        p.set_phase(base, span * 0.15);
        p.set(format!("Phase {label}: loading"));
        let vol = req.volumes.load(series, p)?;
        let phase_grid = vol.grid();

        // The global rigid body: the start of the deformable model, and the
        // rigid model itself when no local margin is asked for.
        let need_global = registers
            && (req.models.contains(&MotionModel::Deformable)
                || (req.models.contains(&MotionModel::Rigid)
                    && req.local_rigid_margin_mm.is_none()));
        // Which qa row each model of this phase owns, so the structure
        // scores measured after propagation land on the right one.
        let mut rigid_qa: Option<usize> = None;
        let mut local_qa: Vec<Option<usize>> = vec![None; n_subjects];
        let mut def_qa: Option<usize> = None;

        let rigid = if need_global {
            p.set_phase(base + span * 0.15, span * 0.35);
            p.set(format!("Phase {label}: rigid registration"));
            let mut params = req.params.clone();
            params.method = RegMethod::ElastixRigid;
            let rigid = registration::register_cached(&ref_vol, &vol, &params, &mut pyramids, p)?;
            qa.push(RegQa {
                phase: label.clone(),
                model: MotionModel::Rigid,
                region: None,
                metric_line: if req.local_rigid_margin_mm.is_some() {
                    format!("global: {}", rigid.metric_line())
                } else {
                    rigid.metric_line()
                },
                metrics: Some(rigid.metrics()),
                folding_pct: 100.0 * rigid.analysis.jacobian.folded,
                disp_p95_mm: rigid.analysis.displacement.p95,
                image_dice: rigid.analysis.overlap.map(|o| (o.after, o.before)),
                struct_dice: Vec::new(),
            });
            rigid_qa = Some(qa.len() - 1);
            Some(rigid)
        } else {
            None
        };

        // The local rigid model: one rigid body per structure, fitted on
        // the structure's neighbourhood alone.
        let mut local_rigid: Vec<Option<std::sync::Arc<registration::Transform3>>> =
            vec![None; n_subjects];
        if let (true, Some(margin)) = (
            req.models.contains(&MotionModel::Rigid),
            req.local_rigid_margin_mm,
        ) {
            for (si, subject) in subjects.iter().enumerate() {
                if p.cancelled() {
                    return Err(cancelled());
                }
                p.set(format!("Phase {label}: rigid fit of {}", subject.name));
                // The neighbourhood is the reference's, the same on every
                // phase: made on the first and kept.
                if regions[si].is_none() {
                    regions[si] = Some(
                        registration::RegionMask::from_mask(
                            &ref_vol,
                            &subject.mask,
                            subject.name.clone(),
                            margin.max(0.0),
                        )
                        .map(std::sync::Arc::new),
                    );
                }
                let Some(Some(region)) = &regions[si] else {
                    continue;
                };
                let mut params = req.params.clone();
                params.method = RegMethod::ElastixRigid;
                params.region = Some(region.clone());
                let r = registration::register_cached(&ref_vol, &vol, &params, &mut pyramids, p)?;
                qa.push(RegQa {
                    phase: label.clone(),
                    model: MotionModel::Rigid,
                    region: Some(subject.name.clone()),
                    metric_line: format!("{}: {}", subject.name, r.metric_line()),
                    metrics: Some(r.metrics()),
                    folding_pct: 0.0,
                    disp_p95_mm: r.analysis.displacement.p95,
                    image_dice: r.analysis.overlap.map(|o| (o.after, o.before)),
                    struct_dice: Vec::new(),
                });
                local_qa[si] = Some(qa.len() - 1);
                local_rigid[si] = Some(r.transform.clone());
            }
        }

        let deformable = if req.models.contains(&MotionModel::Deformable) {
            p.set_phase(base + span * 0.5, span * 0.35);
            p.set(format!("Phase {label}: deformable refinement"));
            let mut params = req.params.clone();
            params.method = RegMethod::ElastixBSpline;
            params.start = Some(rigid.as_ref().expect("built above").transform.clone());
            let def = registration::register_cached(&ref_vol, &vol, &params, &mut pyramids, p)?;
            qa.push(RegQa {
                phase: label.clone(),
                model: MotionModel::Deformable,
                region: None,
                metric_line: def.metric_line(),
                metrics: Some(def.metrics()),
                folding_pct: 100.0 * def.analysis.jacobian.folded,
                disp_p95_mm: def.analysis.displacement.p95,
                image_dice: def.analysis.overlap.map(|o| (o.after, o.before)),
                struct_dice: Vec::new(),
            });
            def_qa = Some(qa.len() - 1);
            Some(def)
        } else {
            None
        };

        p.set_phase(base + span * 0.85, span * 0.15);
        let mut phase_out: Vec<(String, [u8; 3], BoxedMask)> = Vec::new();
        // A structure contoured on this phase in its own right, on the
        // phase's lattice: read once, by whichever model asks first (the
        // contoured model, or the score of a registration model), and kept
        // as its box for the others.
        let mut contoured_here: Vec<Option<Option<BoxedMask>>> = vec![None; n_subjects];
        let mut contoured_on = |si: usize| -> Result<Option<BoxedMask>> {
            if let Some(done) = &contoured_here[si] {
                return Ok(done.clone());
            }
            let st = req
                .contoured
                .iter()
                .find(|c| c.name == subjects[si].name)
                .and_then(|c| c.phases.get(pi))
                .and_then(|s| s.as_ref());
            let boxed = match st {
                Some(st) => Some(BoxedMask::pack(&st.mask_on(&phase_grid)?, phase_grid.dims)),
                None => None,
            };
            contoured_here[si] = Some(boxed.clone());
            Ok(boxed)
        };
        // The phase's lattice is the reference's to the bit (as the phases
        // of one acquisition are): a mask needs no resampling to join the
        // ITV, it is already there.
        let ref_lattice = same_lattice(&phase_grid, &ref_grid);
        for (mi, model) in req.models.iter().enumerate() {
            p.set(format!("Phase {label}: {}", model.label()));
            // What each subject looks like on this phase under this model,
            // each kept as the box it landed in.
            let props: Vec<PackedItem> = match model {
                MotionModel::Contoured => {
                    let mut out = Vec::new();
                    for (si, subject) in subjects.iter().enumerate() {
                        let Some(boxed) = contoured_on(si)? else {
                            continue;
                        };
                        let cm3 = boxed.count() as f64 * phase_grid.voxel_cm3();
                        out.push(PackedItem::with_box(
                            Propagated {
                                name: subject.name.clone(),
                                color: subject.color,
                                mask: Vec::new(),
                                voxels: boxed.count(),
                                source_cm3: cm3,
                                result_cm3: cm3,
                                mapped_cm3: cm3,
                                source_surface_cm3: subject.surface_cm3,
                                rigid_residual_mm: None,
                            },
                            boxed,
                        ));
                    }
                    out
                }
                MotionModel::Rigid if req.local_rigid_margin_mm.is_some() => {
                    let mut out = Vec::new();
                    for (si, subject) in subjects.iter().enumerate() {
                        let Some(t) = &local_rigid[si] else {
                            continue;
                        };
                        let one = std::slice::from_ref(subject);
                        out.extend(propagate::propagate_packed(
                            &ref_vol, &vol, t, true, one, p,
                        )?);
                    }
                    out
                }
                MotionModel::Rigid => {
                    let t = &rigid.as_ref().expect("built above").transform;
                    propagate::propagate_packed(&ref_vol, &vol, t, true, &subjects, p)?
                }
                MotionModel::Deformable => {
                    let t = &deformable.as_ref().expect("built above").transform;
                    // The transform maps reference → phase; landing on the
                    // phase lattice therefore samples through the inverse.
                    propagate::propagate_packed(&ref_vol, &vol, t, true, &subjects, p)?
                }
            };
            for prop in props {
                let Some(si) = subjects.iter().position(|s| s.name == prop.name) else {
                    continue;
                };
                let boxed = prop.boxed();
                let c = motion::centroid_mm_boxed(boxed, &phase_grid).ok_or_else(|| {
                    anyhow!(
                        "'{}' vanished on phase {label} ({})",
                        prop.name,
                        model.label()
                    )
                })?;
                samples[mi][si][pi] = Some(PhaseSample {
                    phase: label.clone(),
                    centroid: c,
                    volume_cm3: prop.result_cm3,
                    grey: motion::grey_stats_boxed(boxed, &vol),
                });
                // Where the clinic contoured this structure on this phase
                // too, the propagation can be scored against it: the honest
                // measure of whether the model followed the anatomy.
                let owner = match model {
                    MotionModel::Rigid if req.local_rigid_margin_mm.is_some() => local_qa[si],
                    MotionModel::Rigid => rigid_qa,
                    MotionModel::Deformable => def_qa,
                    MotionModel::Contoured => None,
                };
                if let Some(qi) = owner {
                    // A contour that does not rasterise here scores nothing,
                    // as before.
                    if let Some(d) = contoured_on(si)
                        .ok()
                        .flatten()
                        .and_then(|m| motion::dice_boxed(boxed, &m))
                    {
                        qa[qi].struct_dice.push((prop.name.clone(), d));
                    }
                }
                if req.build_itv && si < n_targets {
                    if ref_lattice {
                        motion::union_into_boxed(&mut unions[mi][si], boxed);
                    } else {
                        let on_ref = resample_mask(&boxed.mask(), &phase_grid, &ref_grid);
                        motion::union_into(&mut unions[mi][si], &on_ref);
                    }
                }
                if req.keep_phase_segs {
                    let name = format!("{} ({label}, {})", prop.name, model.label());
                    let color = prop.color;
                    phase_out.push((name, color, prop.into_boxed()));
                }
            }
        }
        if !phase_out.is_empty() {
            phase_series.push(OutSeries {
                label: format!("4D {label} - {}", req.group_name),
                grid: phase_grid,
                referenced_series_uid: series.uid.clone(),
                segs: phase_out,
            });
        }
    }
    if p.cancelled() {
        return Err(cancelled());
    }
    p.set_phase(0.95, 0.05);
    p.set("Assembling the report");

    // Tracks in phase order.
    let mut tracks = Vec::new();
    let mut reference_tracks = Vec::new();
    for (mi, model) in req.models.iter().enumerate() {
        for (si, subject) in subjects.iter().enumerate() {
            // A structure the model could not follow on some phase (no
            // contour of it there, no local fit) has no track under it.
            let Some(filled) = samples[mi][si]
                .iter()
                .cloned()
                .collect::<Option<Vec<PhaseSample>>>()
            else {
                continue;
            };
            let track = Track {
                target: subject.name.clone(),
                model: *model,
                samples: filled,
                reference: req.reference,
            };
            if si < n_targets {
                tracks.push(track);
            } else {
                reference_tracks.push(track);
            }
        }
    }

    // Correlation of every target against the reference structure.
    let mut correlations = Vec::new();
    for t in &tracks {
        let Some(rt) = reference_tracks.iter().find(|r| r.model == t.model) else {
            continue;
        };
        let td = t.displacements();
        let rd = rt.displacements();
        let comp = |v: &[crate::geometry::Vec3], a: usize| -> Vec<f64> {
            v.iter().map(|p| [p.x, p.y, p.z][a]).collect()
        };
        let mut axes = Vec::new();
        for (a, name) in motion::AXES.iter().enumerate() {
            if let Some((r, pv)) = motion::pearson(&comp(&td, a), &comp(&rd, a)) {
                axes.push(AxisCorrelation {
                    axis: name,
                    r,
                    p: pv,
                });
            }
        }
        if !axes.is_empty() {
            correlations.push((t.target.clone(), t.model, axes));
        }
    }

    // The ITVs.
    let mut itvs = Vec::new();
    let mut itv_segs: Vec<(String, [u8; 3], BoxedMask)> = Vec::new();
    if req.build_itv {
        for (mi, model) in req.models.iter().enumerate() {
            for (si, target) in req.targets.iter().enumerate() {
                // No track under this model, no ITV either.
                if !tracks
                    .iter()
                    .any(|t| t.model == *model && t.target == target.name)
                {
                    continue;
                }
                let mut mask = std::mem::take(&mut unions[mi][si]);
                if req.itv_margin_mm > 0.0 {
                    mask = morphology::dilate_mm(
                        &mask,
                        ref_grid.dims,
                        ref_grid.spacing,
                        req.itv_margin_mm,
                    );
                }
                let name = if req.itv_margin_mm > 0.0 {
                    format!(
                        "ITV {} +{:.0}mm ({})",
                        target.name,
                        req.itv_margin_mm,
                        model.label()
                    )
                } else {
                    format!("ITV {} ({})", target.name, model.label())
                };
                itvs.push(ItvResult {
                    target: target.name.clone(),
                    model: *model,
                    margin_mm: req.itv_margin_mm,
                    volume_cm3: motion::volume_cm3(&mask, &ref_grid),
                    seg_name: name.clone(),
                });
                itv_segs.push((name, target.color, BoxedMask::pack(&mask, ref_grid.dims)));
            }
        }
    }

    let report = MotionReport {
        run_name: req.run_name,
        slot_name: req.slot_name,
        patient: req.patient,
        phases: req.phases.iter().map(|(l, _)| l.clone()).collect(),
        reference: req.phases[req.reference].0.clone(),
        tracks,
        reference_tracks,
        reference_structure: req.ref_struct.as_ref().map(|s| s.name.clone()),
        correlations,
        qa,
        itvs,
    };
    Ok(MotionOutcome {
        report,
        itv_series: (!itv_segs.is_empty()).then(|| OutSeries {
            label: format!("4D ITV - {}", req.group_name),
            grid: ref_grid,
            referenced_series_uid: req.phases[req.reference].1.uid.clone(),
            segs: itv_segs,
        }),
        phase_series,
        study_uid: req.study_uid,
    })
}
