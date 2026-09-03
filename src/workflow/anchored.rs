//! One volume onto every phase of a 4D group, anchored on a structure that
//! is contoured on both: the cardiac CT onto the 4DCT phases, aligned on
//! the heart.
//!
//! A cardiac CT and a 4DCT of the same patient are two acquisitions in two
//! frames of reference: the small, sharp, contrast-enhanced heart volume
//! sits hundreds of millimetres from the planning scan in patient
//! coordinates, and a global registration of the whole image would match
//! the wrong things (contrast against none, a breath-hold against ten
//! respiratory bins) even once it found the patient. What is actually wanted
//! is narrower and better posed: put the heart where the heart is on every
//! phase, and carry the target with it.
//!
//! So each phase gets its own three steps, all confined to the anchor:
//!
//! 1. the centroids of the anchor on the phase and on the source are
//!    matched (a translation - the registration's [`Init::Points`]);
//! 2. a rigid registration of the source onto the phase, sampling only the
//!    anchor plus a margin, finds the rotation and the residual shift;
//! 3. optionally a local B-spline refinement on the same region recovers
//!    what is not rigid (the heart at a different cardiac phase, the lungs
//!    pressing differently).
//!
//! What steps 2 and 3 look at is the choice of [`AnchorMode`]. By default
//! they match the anchor's *contours*: each side's mask becomes a signed
//! distance map (millimetres to the surface, negative inside), and the
//! engines minimise the squared difference of the two maps. That is a
//! surface-to-surface fit, indifferent to contrast agent, cardiac phase and
//! kernel, and it is what "align the heart contours" means. Matching the
//! intensities instead is offered for images that are alike; between a
//! contrast-enhanced cardiac CT and a plain 4DCT the mean-squares metric
//! has every incentive to push the bright blood pool out of correspondence,
//! and does.
//!
//! Then the structures travel through that phase's transform, and the anchor
//! itself travels with them: its overlap with the phase's own contour is the
//! quality check nobody has to run separately - a heart that lands on the
//! heart with a Dice of 0.9 says the target landed too.
//!
//! The viewer's propagation module and the MCP server both build an
//! [`AnchoredRequest`] and call [`run`]; the outcome files exactly like a
//! plain group run, plus one [`AnchorQa`] per phase.

use std::sync::Arc;

use crate::loader::{self, SeriesInfo};
use crate::morphology;
use crate::motion::{self, Overlap};
use crate::progress::Progress;
use crate::propagate::{self, Subject};
use crate::registration::{self, Init, RegMethod, RegParams, RegionMask};
use crate::volume::Volume;

use super::group::{GroupOutcome, PhaseOutcome};
use super::select::Structure;

use anyhow::{bail, Context, Result};

/// One phase of the group, with the anchor as it is contoured on it.
pub struct AnchoredPhase {
    pub label: String,
    pub series: SeriesInfo,
    /// The anchor on this phase (the fixed side).
    pub anchor: Structure,
}

/// What the registration stages of an anchored run compare.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum AnchorMode {
    /// The anchor's surfaces, as signed distance maps of the two masks.
    #[default]
    Contours,
    /// The images themselves inside the region.
    Intensity,
}

impl AnchorMode {
    pub fn label(self) -> &'static str {
        match self {
            AnchorMode::Contours => "contours",
            AnchorMode::Intensity => "intensity",
        }
    }
}

/// Everything a run anchored on a structure needs.
pub struct AnchoredRequest {
    /// The moving image: the volume the structures were drawn on.
    pub src_vol: Arc<Volume>,
    /// The anchor on the source (the moving side).
    pub src_anchor: Structure,
    /// What to carry across besides the anchor. Empty is fine: the anchor
    /// always travels, and the run is then a registration with its check.
    pub subjects: Vec<Subject>,
    pub phases: Vec<AnchoredPhase>,
    /// Dilation of the phase's anchor that bounds the registration, mm.
    pub margin_mm: f64,
    /// What the registration stages compare.
    pub mode: AnchorMode,
    /// The rigid stage; `method`, `region`, `init` and `start` are set here.
    pub rigid: RegParams,
    /// The local deformable refinement after the rigid stage; `None` keeps
    /// the alignment rigid.
    pub deformable: Option<RegParams>,
    pub group_name: String,
    pub group: usize,
    pub moving_slot: usize,
    pub moving_series_uid: String,
}

/// How well the anchor landed on one phase: the propagated source anchor
/// against the phase's own contour of it, on the phase's lattice.
pub struct AnchorQa {
    pub phase: String,
    pub anchor: String,
    /// Centroid distance the initialisation closed, mm.
    pub initial_shift_mm: f64,
    /// Rigid stage: `MSD a ▶ b (n iters, t s)`.
    pub rigid_line: String,
    /// Deformable stage, when run.
    pub deformable_line: Option<String>,
    /// Of the deformable stage: 95th-percentile displacement inside the
    /// region, mm, and the fraction of it that folds (Jacobian <= 0).
    pub displacement_p95_mm: Option<f64>,
    pub folded_fraction: Option<f64>,
    /// The overlap; `None` when the anchor did not land at all.
    pub overlap: Option<Overlap>,
}

impl AnchorQa {
    /// `60%: heart_total Dice 0.91, HD95 4.2 mm, centroids 1.3 mm apart`.
    pub fn line(&self) -> String {
        match &self.overlap {
            Some(o) => format!(
                "{}: {} Dice {:.2}, HD95 {:.1} mm, centroids {:.1} mm apart",
                self.phase,
                self.anchor,
                o.dice,
                o.hd95_mm,
                o.centroid_shift().map(|d| d.length()).unwrap_or(0.0)
            ),
            None => format!("{}: {} did not land on this phase", self.phase, self.anchor),
        }
    }

    /// A verdict a person can act on: 0.85 is where a heart contour carried
    /// between two CTs of one patient stops being a matter of taste.
    pub fn verdict(&self) -> &'static str {
        match self.overlap.as_ref().map(|o| o.dice) {
            Some(d) if d >= 0.85 => "good",
            Some(d) if d >= 0.7 => "check",
            Some(_) => "poor",
            None => "failed",
        }
    }
}

/// What an anchored run hands back: a group outcome that files like any
/// other, plus the per-phase check.
pub struct AnchoredOutcome {
    pub group: GroupOutcome,
    pub qa: Vec<AnchorQa>,
}

/// Register the source volume onto every phase, anchored on the structure,
/// and carry the structures across, on the calling thread.
pub fn run(req: AnchoredRequest, p: &Progress) -> Result<AnchoredOutcome> {
    if req.rigid.method.is_deformable() {
        bail!("the anchored rigid stage needs a rigid method");
    }
    if let Some(d) = &req.deformable {
        if !d.method.is_deformable() {
            bail!("the anchored refinement needs a deformable method");
        }
    }
    let src_grid = req.src_vol.grid();
    let src_anchor_mask = req.src_anchor.mask_on(&src_grid)?;
    let src_centroid = motion::centroid_mm(&src_anchor_mask, &src_grid)
        .with_context(|| format!("'{}' is empty on the source", req.src_anchor.name))?;
    // What the moving side of the registration is: the image, or the
    // anchor's distance map on the image's lattice.
    let moving_reg: Arc<Volume> = match req.mode {
        AnchorMode::Intensity => req.src_vol.clone(),
        AnchorMode::Contours => {
            p.set("Distance map of the source anchor");
            Arc::new(distance_volume(&req.src_vol, &src_anchor_mask))
        }
    };
    // The anchor travels too: it is the check.
    let mut subjects = req.subjects;
    subjects.push(Subject {
        name: req.src_anchor.name.clone(),
        color: req.src_anchor.color,
        mask: src_anchor_mask,
    });
    let anchor_idx = subjects.len() - 1;

    let n = req.phases.len().max(1);
    let mut phases = Vec::with_capacity(req.phases.len());
    let mut qa = Vec::with_capacity(req.phases.len());
    for (i, ph) in req.phases.iter().enumerate() {
        let label = &ph.label;
        let base = i as f32 / n as f32;
        let span = 1.0 / n as f32;
        p.set_phase(base, span * 0.15);
        p.set(format!("Phase {label}: loading ({}/{n})", i + 1));
        let (vol, _, _) = loader::load_series_volume(&ph.series, p)
            .with_context(|| format!("phase '{label}'"))?;
        let vol = Arc::new(vol);
        let grid = vol.grid();

        // The anchor on the phase: where the registration looks, and where
        // it starts.
        let fixed_mask = ph
            .anchor
            .mask_on(&grid)
            .with_context(|| format!("phase '{label}'"))?;
        let fixed_centroid = motion::centroid_mm(&fixed_mask, &grid)
            .with_context(|| format!("'{}' is empty on phase '{label}'", ph.anchor.name))?;
        let region = RegionMask::from_mask(
            &vol,
            &fixed_mask,
            ph.anchor.name.clone(),
            req.margin_mm.max(0.0),
        )
        .with_context(|| format!("'{}' is empty on phase '{label}'", ph.anchor.name))?;
        let region = Arc::new(region);
        let initial_shift_mm = (src_centroid - fixed_centroid).length();

        // Fixed is the phase, moving is the source, so the transform maps
        // phase → source: the destination → source direction `propagate`
        // pulls along, with no inversion.
        p.set_phase(
            base + span * 0.15,
            span * if req.deformable.is_some() { 0.3 } else { 0.6 },
        );
        p.set(format!(
            "Phase {label}: rigid on {} ({}/{n})",
            ph.anchor.name,
            i + 1
        ));
        let fixed_reg: Arc<Volume> = match req.mode {
            AnchorMode::Intensity => vol.clone(),
            AnchorMode::Contours => {
                p.set(format!("Phase {label}: distance map of {}", ph.anchor.name));
                Arc::new(distance_volume(&vol, &fixed_mask))
            }
        };
        let mut rigid = req.rigid.clone();
        rigid.region = Some(region.clone());
        rigid.start = None;
        rigid.init = Init::Points {
            fixed: fixed_centroid,
            moving: src_centroid,
        };
        if req.mode == AnchorMode::Contours {
            // Every voxel of a distance map is informative; the body
            // threshold is for images.
            rigid.fixed_threshold = f32::MIN;
        }
        let r = registration::register(&fixed_reg, &moving_reg, &rigid, p)
            .with_context(|| format!("phase '{label}', rigid stage"))?;
        let rigid_line = r.metric_line();
        let mut transform = r.transform.clone();
        let mut metric_line = format!("{} rigid {rigid_line}", req.mode.label());
        let mut deformable_line = None;
        let mut displacement_p95_mm = None;
        let mut folded_fraction = None;

        if let Some(d) = &req.deformable {
            p.set_phase(base + span * 0.45, span * 0.3);
            p.set(format!(
                "Phase {label}: refining on {} ({}/{n})",
                ph.anchor.name,
                i + 1
            ));
            let mut params = d.clone();
            params.region = Some(region.clone());
            params.start = Some(transform.clone());
            if req.mode == AnchorMode::Contours {
                params.fixed_threshold = f32::MIN;
            }
            let r = registration::register(&fixed_reg, &moving_reg, &params, p)
                .with_context(|| format!("phase '{label}', deformable stage"))?;
            let line = r.metric_line();
            metric_line.push_str(&format!(" · {} {line}", d.method.label()));
            deformable_line = Some(line);
            displacement_p95_mm = Some(r.analysis.displacement.p95);
            folded_fraction = Some(r.analysis.jacobian.folded);
            transform = r.transform.clone();
        }

        p.set_phase(base + span * 0.75, span * 0.25);
        p.set(format!("Phase {label}: propagating ({}/{n})", i + 1));
        let items = propagate::propagate(&req.src_vol, &vol, &transform, false, &subjects, p)
            .with_context(|| format!("phase '{label}'"))?;
        let overlap = items
            .get(anchor_idx)
            .filter(|it| it.voxels > 0)
            .and_then(|it| motion::overlap(&it.mask, &fixed_mask, &grid));
        let check = AnchorQa {
            phase: label.clone(),
            anchor: ph.anchor.name.clone(),
            initial_shift_mm,
            rigid_line,
            deformable_line,
            displacement_p95_mm,
            folded_fraction,
            overlap,
        };
        metric_line.push_str(&match &check.overlap {
            Some(o) => format!(" · {} Dice {:.2}", check.anchor, o.dice),
            None => format!(" · {} did not land", check.anchor),
        });
        qa.push(check);
        phases.push(PhaseOutcome {
            label: label.clone(),
            series_uid: ph.series.uid.clone(),
            study_uid: ph.series.study_uid.clone(),
            grid,
            items,
            transform,
            metric_line,
        });
    }
    Ok(AnchoredOutcome {
        group: GroupOutcome {
            group_name: req.group_name,
            group: req.group,
            moving_slot: req.moving_slot,
            moving_series_uid: req.moving_series_uid,
            phases,
        },
        qa,
    })
}

/// How far a distance map reaches, mm: beyond this the value is clamped,
/// so a sample far from the surface has no gradient to follow and cannot
/// drag the fit.
const DISTANCE_REACH_MM: f64 = 40.0;
/// Distance-map units per millimetre (the maps are stored as i16).
const DISTANCE_SCALE: f64 = 100.0;

/// The signed distance map of a mask on the volume's lattice, as a volume
/// the engines can register: millimetres to the surface, negative inside,
/// clamped at [`DISTANCE_REACH_MM`] and scaled by [`DISTANCE_SCALE`].
pub fn distance_volume(on: &Volume, mask: &[u8]) -> Volume {
    let dims = on.dims;
    let outside = morphology::dist2_to_foreground(mask, dims, on.spacing);
    let inverted: Vec<u8> = mask.iter().map(|&v| (v == 0) as u8).collect();
    let inside = morphology::dist2_to_foreground(&inverted, dims, on.spacing);
    let data: Vec<i16> = outside
        .iter()
        .zip(&inside)
        .map(|(&o, &i)| {
            let d = (o.max(0.0) as f64).sqrt() - (i.max(0.0) as f64).sqrt();
            (d.clamp(-DISTANCE_REACH_MM, DISTANCE_REACH_MM) * DISTANCE_SCALE).round() as i16
        })
        .collect();
    let reach = (DISTANCE_REACH_MM * DISTANCE_SCALE) as i16;
    Volume {
        data,
        dims,
        spacing: on.spacing,
        origin: on.origin,
        row_dir: on.row_dir,
        col_dir: on.col_dir,
        normal: on.normal,
        frame_of_reference_uid: on.frame_of_reference_uid.clone(),
        min_value: -reach,
        max_value: reach,
    }
}

/// The rigid parameters an anchored run uses unless told otherwise: the
/// elastix rigid engine at the caller's settings, sampling the region.
pub fn default_rigid(base: &RegParams) -> RegParams {
    RegParams {
        method: RegMethod::ElastixRigid,
        region: None,
        start: None,
        init: Init::Auto,
        ..base.clone()
    }
}

/// The deformable refinement an anchored run uses unless told otherwise.
pub fn default_deformable(base: &RegParams) -> RegParams {
    RegParams {
        method: if base.method.is_deformable() {
            base.method
        } else {
            RegMethod::ElastixBSpline
        },
        region: None,
        start: None,
        init: Init::Auto,
        ..base.clone()
    }
}
