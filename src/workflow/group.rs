//! One volume onto every phase of a 4D group: register the source volume
//! onto each phase and carry the structures across, one phase at a time.
//!
//! Each phase gets its own registration: a 4D acquisition is exactly the
//! case where one transform for the whole group would be wrong, since the
//! point of the phases is that the anatomy moves between them.
//!
//! Moved out of `app/propagate_win.rs`; the viewer's module and the MCP
//! server both build a [`GroupRequest`] and call [`run`].

use std::sync::Arc;

use crate::dicomseg::SegSeries;
use crate::loader::{self, LoadedStudy, SeriesInfo};
use crate::progress::Progress;
use crate::propagate::{self, Finish, PackedItem, Propagated, Subject};
use crate::registration::{self, RegParams, Transform3};

use crate::segmentation::Segmentation;
use crate::volume::{Grid, Volume};
use crate::workflow::session;

use anyhow::{Context, Result};

/// Where propagated structures are filed on their destination image.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Landing {
    /// A new segmentation series bound to the image series (editable masks).
    #[default]
    Segmentation,
    /// The image series' own RT structure set, as contours - the set that
    /// references the series, or a new one bound to it when there is none.
    /// A 4DCT with one structure set per phase gets the target next to that
    /// phase's heart, which is where a planning system expects it.
    StructureSet,
}

impl Landing {
    pub fn label(self) -> &'static str {
        match self {
            Landing::Segmentation => "segmentation series",
            Landing::StructureSet => "structure set",
        }
    }
}

/// Everything the worker needs for a run against a 4D group.
pub struct GroupRequest {
    /// The moving image: the volume the structures were drawn on.
    pub src_vol: Arc<Volume>,
    /// What to carry across. Empty means register and nothing else.
    pub subjects: Vec<Subject>,
    /// (phase label, the series to load) in temporal order.
    pub phases: Vec<(String, SeriesInfo)>,
    /// A transform already known for that phase, which is then not
    /// recomputed. Same length and order as `phases`.
    pub cached: Vec<Option<Arc<Transform3>>>,
    /// Must be deformable: phases of one acquisition differ by breathing.
    pub params: RegParams,
    /// What is done to each landed mask (closing, filling).
    pub finish: Finish,
    pub group_name: String,
    pub group: usize,
    pub moving_slot: usize,
    pub moving_series_uid: String,
    /// Where the phases are read from: a run's shared cache, or
    /// [`session::Volumes::none`] to read each from disk.
    pub volumes: session::Volumes,
}

/// What one phase of a 4D group came out with.
pub struct PhaseOutcome {
    /// The phase's name within the group: "0%", "50%", "t3".
    pub label: String,
    /// The image series the results belong to, and the study it is in.
    pub series_uid: String,
    pub study_uid: String,
    /// The lattice they are on.
    pub grid: Grid,
    /// Empty when the run was a registration and nothing else. Packed: a
    /// group's phases are all held until they are filed.
    pub items: Vec<PackedItem>,
    /// Phase → the moving image.
    pub transform: Arc<Transform3>,
    /// `MSD 9700 ▶ 1800  (900 iters, 20.1 s)` of that phase's registration,
    /// or what it says instead when the transform was reused.
    pub metric_line: String,
    /// The same numbers, for a table. `None` when nothing was run for this
    /// phase - a reused transform has no metric of its own.
    pub metrics: Option<crate::registration::RunMetrics>,
}

impl PhaseOutcome {
    /// The propagated structures with their whole-volume masks, for filing.
    pub fn unpacked(&self) -> Vec<Propagated> {
        self.items.iter().map(PackedItem::unpack).collect()
    }

    /// The propagated structures as one segmentation series bound to the
    /// phase's image series, so the tree files it under the right member.
    /// Empty items (nothing landed) are skipped; `None` when none landed.
    pub fn seg_series(&self, group_name: &str) -> Option<SegSeries> {
        let mut series = SegSeries::new(
            format!("{} {}", group_name, self.label),
            self.grid.clone(),
            self.series_uid.clone(),
            self.study_uid.clone(),
        );
        for item in &self.items {
            if item.voxels == 0 {
                continue;
            }
            series.segs.push(Segmentation::from_label_map(
                item.name.clone(),
                item.color,
                self.grid.dims,
                &item.mask(),
                1,
            ));
        }
        (!series.segs.is_empty()).then_some(series)
    }
}

/// File propagated masks as contours in the structure set of `series_uid`
/// within `study`: the set that references the series (the last such, the
/// most recent), or a new in-memory set bound to it. Empty items are
/// skipped. Returns the label of the set and the names filed, or `None`
/// when nothing landed.
///
/// Names that already exist in the set are suffixed with a counter, so a
/// second run adds `target (2)` rather than a second `target`.
pub fn land_in_structure_set(
    study: &mut LoadedStudy,
    series_uid: &str,
    study_uid: &str,
    grid: &Grid,
    items: &[Propagated],
    new_set_label: &str,
) -> Option<(String, Vec<String>)> {
    land_in_structure_set_as(
        study,
        series_uid,
        study_uid,
        grid,
        items,
        new_set_label,
        "GTV",
    )
}

/// [`land_in_structure_set`] with the RT ROI Interpreted Type spelled out:
/// a propagated target is a `GTV`, an organ an engine found is an `ORGAN`,
/// a body outline an `EXTERNAL`.
pub fn land_in_structure_set_as(
    study: &mut LoadedStudy,
    series_uid: &str,
    study_uid: &str,
    grid: &Grid,
    items: &[Propagated],
    new_set_label: &str,
    roi_type: &str,
) -> Option<(String, Vec<String>)> {
    land_items_as(
        study,
        series_uid,
        study_uid,
        grid,
        items,
        new_set_label,
        roi_type,
    )
    .map(|(label, filed)| (label, filed.into_iter().flatten().collect()))
}

/// [`land_in_structure_set_as`], answering item by item: the names come
/// back in the order of `items`, `None` where an item filed nothing (it was
/// empty, or traced to no contour). That is what a caller needs to find the
/// ROI a given item became - to measure it, say - when the name it was
/// filed under carries a counter.
pub fn land_items_as(
    study: &mut LoadedStudy,
    series_uid: &str,
    study_uid: &str,
    grid: &Grid,
    items: &[Propagated],
    new_set_label: &str,
    roi_type: &str,
) -> Option<(String, Vec<Option<String>>)> {
    if items.iter().all(|it| it.voxels == 0) {
        return None;
    }
    // The series the set is bound to: the study's own entry when it lists
    // it (a phase does), else one that carries just the identity.
    let series = study
        .series
        .iter()
        .find(|s| s.uid == series_uid)
        .cloned()
        .unwrap_or_else(|| SeriesInfo {
            uid: series_uid.to_string(),
            study_uid: study_uid.to_string(),
            ..SeriesInfo::default()
        });
    let types = vec![roi_type.to_string(); items.len()];
    // A set that is locked (approved) is not written into: the structures
    // go into a new set beside it.
    let locked = study
        .structure_sets
        .iter()
        .rposition(|s| s.referenced_series_uid == series_uid)
        .is_some_and(|i| study.structure_sets[i].locked);
    let filed = session::file_items(
        study,
        &series,
        grid,
        items,
        &types,
        &session::Filing {
            kind: session::OutputKind::Structures,
            set: if locked {
                session::SetChoice::New
            } else {
                session::SetChoice::Own
            },
            set_label: new_set_label,
            clash: Some(session::NameClash::Counter),
            taken: None,
        },
    )
    .ok()?;
    if filed.names.iter().all(Option::is_none) {
        return None;
    }
    let set = filed.set?;
    Some((study.structure_sets[set].label.clone(), filed.names))
}

/// What a run against a whole 4D group hands back.
pub struct GroupOutcome {
    pub group_name: String,
    /// Which group this was, so the transforms can be filed and found again.
    pub group: usize,
    pub moving_slot: usize,
    pub moving_series_uid: String,
    pub phases: Vec<PhaseOutcome>,
}

/// Register the source volume onto every phase of the group and carry the
/// structures across, on the calling thread.
pub fn run(mut req: GroupRequest, p: &Progress) -> Result<GroupOutcome> {
    let finish = req.finish;
    finish.carry(&mut req.subjects);
    let n = req.phases.len().max(1);
    let mut phases = Vec::with_capacity(req.phases.len());
    // The source volume is the moving image of every registration: its
    // pyramid is built once.
    let mut pyramids = registration::PyramidCache::default();
    for (i, (label, series)) in req.phases.iter().enumerate() {
        let base = i as f32 / n as f32;
        let span = 1.0 / n as f32;
        p.set_phase(base, span * 0.25);
        p.set(format!("Phase {label}: loading ({}/{n})", i + 1));
        let vol = req
            .volumes
            .load(series, p)
            .with_context(|| format!("phase '{label}'"))?;
        let cached = req.cached.get(i).and_then(|t| t.clone());
        let (transform, metric_line, metrics) = match cached {
            Some(t) => (t, "transform reused".to_string(), None),
            None => {
                p.set_phase(base + span * 0.25, span * 0.55);
                p.set(format!("Phase {label}: registering ({}/{n})", i + 1));
                // Fixed is the phase, moving is the source volume, so the
                // transform maps phase → source: exactly the destination →
                // source direction `propagate` pulls along, with no
                // inversion.
                let r = registration::register_cached(
                    &vol,
                    &req.src_vol,
                    &req.params,
                    &mut pyramids,
                    p,
                )
                .with_context(|| format!("phase '{label}'"))?;
                (r.transform.clone(), r.metric_line(), Some(r.metrics()))
            }
        };
        let items = if req.subjects.is_empty() {
            Vec::new()
        } else {
            p.set_phase(base + span * 0.8, span * 0.2);
            p.set(format!("Phase {label}: propagating ({}/{n})", i + 1));
            let mut items =
                propagate::propagate(&req.src_vol, &vol, &transform, false, &req.subjects, p)
                    .with_context(|| format!("phase '{label}'"))?;
            req.finish.apply_all(&mut items, &vol.grid(), p);
            items
                .into_iter()
                .map(|it| PackedItem::pack(it, vol.dims))
                .collect()
        };
        phases.push(PhaseOutcome {
            label: label.clone(),
            series_uid: series.uid.clone(),
            study_uid: series.study_uid.clone(),
            grid: vol.grid(),
            items,
            transform,
            metric_line,
            metrics,
        });
    }
    Ok(GroupOutcome {
        group_name: req.group_name,
        group: req.group,
        moving_slot: req.moving_slot,
        moving_series_uid: req.moving_series_uid,
        phases,
    })
}

/// One phase of a 4D group with structures copied onto its lattice as they
/// are, no registration involved: what *Copy to each phase of ...* hands
/// back for the viewer to file.
pub struct PhaseCopy {
    pub label: String,
    pub series_uid: String,
    pub study_uid: String,
    pub grid: Grid,
    pub segs: Vec<Segmentation>,
    /// Structures that did not reach this phase (outside its volume).
    pub notes: Vec<String>,
}

impl PhaseCopy {
    /// The copies as a fresh segmentation series bound to the phase.
    pub fn seg_series(&self, group_name: &str) -> SegSeries {
        let mut series = SegSeries::new(
            format!("{} {}", group_name, self.label),
            self.grid.clone(),
            self.series_uid.clone(),
            self.study_uid.clone(),
        );
        series.segs = self.segs.clone();
        series
    }
}

/// Carry `structures` onto every phase of a group without registering
/// anything: each phase's lattice is read from its slice headers (the
/// pixels play no part) and the structures are
/// rasterized (contours) or resampled (masks) onto it in patient
/// coordinates. This is a copy in the sense of the tree's *Copy to*, not a
/// propagation - the structure stays where it is while the anatomy under
/// it moves - which is what a fixed margin, a couch or an ITV wants.
pub fn copy_to_phases(
    structures: &[crate::workflow::select::Structure],
    phases: &[(String, SeriesInfo)],
    p: &Progress,
) -> Result<Vec<PhaseCopy>> {
    let n = phases.len().max(1);
    let mut out = Vec::with_capacity(phases.len());
    for (i, (label, series)) in phases.iter().enumerate() {
        p.set_phase(i as f32 / n as f32, 1.0 / n as f32);
        p.set(format!(
            "Phase {label}: reading the lattice ({}/{n})",
            i + 1
        ));
        if p.cancelled() {
            anyhow::bail!("cancelled");
        }
        let grid = loader::series_grid(series).with_context(|| format!("reading {label}"))?;
        let mut segs = Vec::new();
        let mut notes = Vec::new();
        for s in structures {
            match s.mask_on(&grid) {
                Ok(mask) => {
                    let seg = Segmentation::from_mask(s.name.clone(), s.color, grid.dims, mask);
                    if seg.count == 0 {
                        notes.push(format!("'{}' does not overlap {label}", s.name));
                    } else {
                        segs.push(seg);
                    }
                }
                Err(_) => notes.push(format!("'{}' has no contour inside {label}", s.name)),
            }
        }
        out.push(PhaseCopy {
            label: label.clone(),
            series_uid: series.uid.clone(),
            study_uid: series.study_uid.clone(),
            grid,
            segs,
            notes,
        });
    }
    Ok(out)
}
