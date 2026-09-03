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
use crate::loader::{self, SeriesInfo};
use crate::progress::Progress;
use crate::propagate::{self, Propagated, Subject};
use crate::registration::{self, RegParams, Transform3};
use crate::segmentation::Segmentation;
use crate::volume::{Grid, Volume};

use anyhow::{Context, Result};

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
    pub group_name: String,
    pub group: usize,
    pub moving_slot: usize,
    pub moving_series_uid: String,
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
    /// Empty when the run was a registration and nothing else.
    pub items: Vec<Propagated>,
    /// Phase → the moving image.
    pub transform: Arc<Transform3>,
    /// `MSD 9700 ▶ 1800  (900 iters, 20.1 s)` of that phase's registration,
    /// or what it says instead when the transform was reused.
    pub metric_line: String,
}

impl PhaseOutcome {
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
                &item.mask,
                1,
            ));
        }
        (!series.segs.is_empty()).then_some(series)
    }
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
pub fn run(req: GroupRequest, p: &Progress) -> Result<GroupOutcome> {
    let n = req.phases.len().max(1);
    let mut phases = Vec::with_capacity(req.phases.len());
    for (i, (label, series)) in req.phases.iter().enumerate() {
        let base = i as f32 / n as f32;
        let span = 1.0 / n as f32;
        p.set_phase(base, span * 0.25);
        p.set(format!("Phase {label}: loading ({}/{n})", i + 1));
        let (vol, _, _) =
            loader::load_series_volume(series, p).with_context(|| format!("phase '{label}'"))?;
        let cached = req.cached.get(i).and_then(|t| t.clone());
        let (transform, metric_line) = match cached {
            Some(t) => (t, "transform reused".to_string()),
            None => {
                p.set_phase(base + span * 0.25, span * 0.55);
                p.set(format!("Phase {label}: registering ({}/{n})", i + 1));
                // Fixed is the phase, moving is the source volume, so the
                // transform maps phase → source: exactly the destination →
                // source direction `propagate` pulls along, with no
                // inversion.
                let r = registration::register(&vol, &req.src_vol, &req.params, p)
                    .with_context(|| format!("phase '{label}'"))?;
                (r.transform.clone(), r.metric_line())
            }
        };
        let items = if req.subjects.is_empty() {
            Vec::new()
        } else {
            p.set_phase(base + span * 0.8, span * 0.2);
            p.set(format!("Phase {label}: propagating ({}/{n})", i + 1));
            propagate::propagate(&req.src_vol, &vol, &transform, false, &req.subjects, p)
                .with_context(|| format!("phase '{label}'"))?
        };
        phases.push(PhaseOutcome {
            label: label.clone(),
            series_uid: series.uid.clone(),
            study_uid: series.study_uid.clone(),
            grid: vol.grid(),
            items,
            transform,
            metric_line,
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
