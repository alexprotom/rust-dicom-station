//! *Modules ▶ Structures propagation*: carrying contours and segmentations
//! across the active registration.
//!
//! The module is deliberately thin: the hard part is the transform, and that
//! already exists. What it adds is the choice of *what* travels, the
//! direction, and one option that matters clinically: refining the
//! registration on an enclosing structure first, which is what makes a small
//! structure inside a larger one land where it belongs rather than where the
//! whole patient's average deformation puts it.
//!
//! It sits in the modules panel beside image registration, which is where
//! the transform it uses comes from, rather than in a window of its own.

use super::*;
use crate::propagate::{self, Propagated, Subject};
use crate::volume::Grid;
use crate::workflow::anchored::{self, AnchoredOutcome};
use crate::workflow::group::{land_in_structure_set, Landing};
pub(super) use crate::workflow::group::{run as run_group, GroupOutcome, GroupRequest};
use crate::workflow::select::Structure;

/// Where a propagation run puts its results.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PropTarget {
    /// The displayed volume of the other dataset, through the registration
    /// that is already active.
    Other,
    /// Every phase of one 4D group, registering the source volume onto each
    /// phase on the way. `slot` is the dataset the group belongs to, which
    /// may be the source's own: a planning CT and the 4DCT of the same
    /// patient often arrive together.
    Group { slot: usize, group: usize },
}

/// The propagation module's state.
pub(super) struct PropagateDialog {
    /// Dataset the structures come from; they land on the other one.
    pub src_slot: usize,
    /// What they land on.
    pub target: PropTarget,
    /// Selected ROIs of the source dataset's active structure set.
    pub structs: Vec<bool>,
    /// Selected segmentations of the source dataset.
    pub segs: Vec<bool>,
    /// Refine the registration on this region of the *fixed* dataset first.
    pub local: RegRoi,
    pub local_margin_mm: f64,
    /// Against a group: the structure of the *source* the run is anchored
    /// on (contoured on every phase too), or `Whole` for a plain run.
    pub anchor: RegRoi,
    pub anchor_margin_mm: f64,
    /// Anchored run: refine deformably after the rigid stage.
    pub anchor_deformable: bool,
    /// Anchored run: match the anchor's contours (distance maps) rather
    /// than the image intensities.
    pub anchor_contours: bool,
    /// Where the results are filed on the destination image.
    pub landing: Landing,
    /// Through the active registration: which of its two images the
    /// structures come from (they land on the other).
    pub from_side: FusionSide,
    /// What the last run produced.
    pub summary: Vec<String>,
}

impl Default for PropagateDialog {
    fn default() -> Self {
        PropagateDialog {
            src_slot: 0,
            target: PropTarget::Other,
            structs: Vec::new(),
            segs: Vec::new(),
            local: RegRoi::Whole,
            local_margin_mm: 10.0,
            anchor: RegRoi::Whole,
            anchor_margin_mm: 10.0,
            anchor_deformable: true,
            anchor_contours: true,
            landing: Landing::Segmentation,
            from_side: FusionSide::Moving,
            summary: Vec::new(),
        }
    }
}

/// What a propagation run hands back.
pub(super) enum PropOutcome {
    /// One destination volume, carried through the active registration.
    /// Boxed: a `RegOutcome` carries a whole vector field, which would make
    /// every one of these as large as the largest.
    One {
        items: Vec<Propagated>,
        /// The image they landed on.
        dst_uid: String,
        dst_grid: Grid,
        /// A local refinement run on the way, which becomes the active
        /// registration so the panel reports what was actually used.
        refined: Option<Box<RegOutcome>>,
    },
    /// One result per phase of a 4D group, each with the registration that
    /// put it there.
    Group(GroupOutcome),
    /// The same, anchored on a structure, with the per-phase check.
    Anchored(AnchoredOutcome),
}

/// The per-phase transforms of one 4D group, kept so a later propagation
/// onto the same group does not pay for the registrations again.
///
/// A registration is minutes; loading a phase and pulling a mask through a
/// transform is seconds. Once the group is registered, carrying another
/// structure set across should not repeat the expensive half.
pub(super) struct GroupRegistration {
    /// Dataset the moving image came from, and the series it was.
    pub moving_slot: usize,
    pub moving_series_uid: String,
    /// The dataset holding the group, and which group of it.
    pub slot: usize,
    pub group: usize,
    pub group_name: String,
    /// One entry per phase, in the group's temporal order.
    pub phases: Vec<GroupPhaseReg>,
}

/// One phase of a registered 4D group.
pub(super) struct GroupPhaseReg {
    pub label: String,
    pub series_uid: String,
    /// Phase → moving image: the destination → source direction a
    /// propagation pulls along, so it needs no inversion.
    pub transform: Arc<registration::Transform3>,
    pub metric_line: String,
}

impl ViewerApp {
    /// Show the propagation module and point it at `src_slot`.
    ///
    /// A running job owns the dialog (its results have to land somewhere), so
    /// this never replaces one that is in flight; it only aims it.
    pub(super) fn open_propagate_module(&mut self, src_slot: usize) {
        self.module_propagation = true;
        self.right_open = true;
        self.settings_gen += 1;
        match &mut self.propagate_dialog {
            Some(d) if self.propagate_job.is_some() => d.src_slot = src_slot,
            _ => {
                let mut d = PropagateDialog {
                    src_slot,
                    ..Default::default()
                };
                self.sync_propagate_lists(&mut d);
                self.propagate_dialog = Some(d);
            }
        }
    }

    /// Keep the check-box lists the same length as what the dataset holds.
    fn sync_propagate_lists(&self, d: &mut PropagateDialog) {
        let n_struct = self.slots[d.src_slot]
            .active_structures()
            .map(|ss| ss.rois.len())
            .unwrap_or(0);
        d.structs.resize(n_struct, false);
        d.segs.resize(self.slots[d.src_slot].segs().len(), false);
    }

    /// Everything ticked, frozen: contours as they are, painted
    /// segmentations with the lattice they are on. A run turns them into
    /// masks on whatever source lattice it maps from, which need not be
    /// loaded yet.
    fn propagate_structures(&self, d: &PropagateDialog) -> Result<Vec<Structure>, String> {
        use crate::workflow::select::Source;
        let slot = d.src_slot;
        let Some(study) = &self.slots[slot].study else {
            return Err(format!("dataset {} is not loaded", SLOT_NAMES[slot]));
        };
        let mut out = Vec::new();
        if let Some(ss) = self.slots[slot].active_structures() {
            for (i, roi) in ss.rois.iter().enumerate() {
                if d.structs.get(i).copied().unwrap_or(false) {
                    out.push(Structure::from_roi(roi));
                }
            }
        }
        let grid = study.volume.grid();
        for (i, seg) in self.slots[slot].segs().iter().enumerate() {
            if d.segs.get(i).copied().unwrap_or(false) && seg.count > 0 {
                out.push(Structure {
                    name: seg.name.clone(),
                    color: seg.color,
                    source: Source::Mask {
                        mask: seg.mask.clone(),
                        grid: grid.clone(),
                    },
                });
            }
        }
        if out.is_empty() {
            return Err("tick at least one structure or segmentation".into());
        }
        Ok(out)
    }

    /// Collect the masks of everything ticked.
    ///
    /// Contours are rasterized on `grid`, the lattice of the source image
    /// the run maps from; painted segmentations must already be on it.
    fn propagate_subjects(&self, d: &PropagateDialog, grid: &Grid) -> Result<Vec<Subject>, String> {
        let slot = d.src_slot;
        if self.slots[slot].study.is_none() {
            return Err(format!("dataset {} is not loaded", SLOT_NAMES[slot]));
        }
        let mut out = Vec::new();
        if let Some(ss) = self.slots[slot].active_structures() {
            for (i, roi) in ss.rois.iter().enumerate() {
                if !d.structs.get(i).copied().unwrap_or(false) {
                    continue;
                }
                match segmentation::rasterize_roi(grid, roi) {
                    Some(mask) => out.push(Subject {
                        name: roi.name.clone(),
                        color: roi.color,
                        mask,
                    }),
                    None => {
                        return Err(format!(
                            "'{}' has no planar contour inside the displayed volume",
                            roi.name
                        ))
                    }
                }
            }
        }
        for (i, seg) in self.slots[slot].segs().iter().enumerate() {
            if d.segs.get(i).copied().unwrap_or(false) && seg.count > 0 {
                if seg.dims != grid.dims {
                    return Err(format!(
                        "'{}' is painted on the displayed volume, not on the registered source \
                         image; display that image to carry painted segmentations",
                        seg.name
                    ));
                }
                out.push(Subject {
                    name: seg.name.clone(),
                    color: seg.color,
                    mask: seg.mask.clone(),
                });
            }
        }
        if out.is_empty() {
            return Err("tick at least one structure or segmentation".into());
        }
        Ok(out)
    }

    /// The 4D groups either dataset offers, as `(target, label)`.
    ///
    /// A group in the source's own dataset counts: a planning CT and the
    /// 4DCT of the same patient usually arrive together, and the structures
    /// still have to travel from the one to the phases of the other.
    pub(super) fn propagate_group_choices(&self) -> Vec<(PropTarget, String)> {
        let mut out = Vec::new();
        for (slot, name) in SLOT_NAMES.iter().enumerate() {
            let Some(study) = self.slots[slot].study.as_ref() else {
                continue;
            };
            for (gi, g) in study.fourd_groups.iter().enumerate() {
                let n = g
                    .members
                    .iter()
                    .filter(|m| m.role == crate::fourd::Role::Phase)
                    .count();
                if n < 2 {
                    continue;
                }
                out.push((
                    PropTarget::Group { slot, group: gi },
                    format!("{name}: {} ({n} phases)", g.name),
                ));
            }
        }
        out
    }

    /// How many phases one 4D group has, or 0 if it is gone.
    pub(super) fn group_phase_count(&self, slot: usize, group: usize) -> usize {
        self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.fourd_groups.get(group))
            .map(|g| {
                g.members
                    .iter()
                    .filter(|m| m.role == crate::fourd::Role::Phase)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Start a run against every phase of a 4D group: register the displayed
    /// volume of `moving_slot` onto each phase, and carry `subjects` across
    /// on the way.
    ///
    /// An empty `subjects` makes it a registration and nothing else, which is
    /// what the registration module asks for; the transforms it leaves behind
    /// are what a later propagation onto the same group reuses.
    pub(super) fn start_group_run(
        &mut self,
        moving: RegPick,
        slot: usize,
        group: usize,
        structures: Vec<Structure>,
    ) {
        if self.propagate_job.is_some() {
            return;
        }
        let moving_slot = moving.slot;
        let Some(src) = self.slots[moving_slot].study.as_ref() else {
            self.error = Some("This needs a loaded source dataset".into());
            return;
        };
        let Some(moving_series) = src.series.get(moving.series).cloned() else {
            self.error = Some("The moving series is gone - pick it again.".into());
            return;
        };
        // The moving volume: the displayed one, or loaded on the worker.
        let src_ready =
            (src.has_volume() && src.active_series == moving.series).then(|| src.volume.clone());
        let moving_series_uid = moving_series.uid.clone();
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(g) = study.fourd_groups.get(group) else {
            self.error = Some("That 4D group is gone - pick it again.".into());
            return;
        };
        let phases = match crate::workflow::phases_of(g, &study.series) {
            Ok(p) => p,
            Err(e) => {
                self.error = Some(format!("{e}"));
                return;
            }
        };
        // No region: a refinement belongs to one pair of images, and there
        // is one pair per phase here.
        let mut params = self.current_reg_params(None, true);
        // Phases of one acquisition differ by breathing, which is a
        // deformation; a rigid-only run would leave the anatomy where the
        // reference put it.
        if !params.method.is_deformable() {
            params.method = registration::RegMethod::PlastimatchBSpline;
        }
        // Transforms already made for exactly this group, from exactly this
        // moving image. Anything else about the pair having changed and they
        // would be answering a different question.
        let cached: Vec<Option<Arc<registration::Transform3>>> = match &self.group_registration {
            Some(gr)
                if gr.slot == slot
                    && gr.group == group
                    && gr.moving_slot == moving_slot
                    && gr.moving_series_uid == moving_series_uid =>
            {
                phases
                    .iter()
                    .map(|(_, se)| {
                        gr.phases
                            .iter()
                            .find(|ph| ph.series_uid == se.uid)
                            .map(|ph| ph.transform.clone())
                    })
                    .collect()
            }
            _ => vec![None; phases.len()],
        };
        let group_name = g.name.clone();
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.propagate_job = Some(Job::spawn(progress, move |p| {
            let src_vol = match src_ready {
                Some(v) => v,
                None => {
                    p.set("Loading the moving image");
                    match loader::load_series_volume(&moving_series, p) {
                        Ok((v, _, _)) => Arc::new(v),
                        Err(e) => return (slot, Err(e)),
                    }
                }
            };
            let grid = src_vol.grid();
            let subjects: Vec<Subject> =
                match structures.iter().map(|s| s.subject_on(&grid)).collect() {
                    Ok(v) => v,
                    Err(e) => return (slot, Err(e)),
                };
            let req = GroupRequest {
                src_vol,
                subjects,
                phases,
                cached,
                params,
                group_name,
                group,
                moving_slot,
                moving_series_uid,
            };
            (slot, run_group(req, p).map(PropOutcome::Group))
        }));
    }

    /// Start a run against every phase of a 4D group anchored on a
    /// structure: the source's `anchor` is matched to its namesake on each
    /// phase (centroids, then a rigid fit on the structure, then optionally a
    /// local deformable refinement), and `subjects` travel through that.
    ///
    /// This is how a cardiac CT meets a 4DCT: the two share no frame of
    /// reference, and only the heart is worth matching.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_anchored_run(
        &mut self,
        moving_slot: usize,
        slot: usize,
        group: usize,
        anchor: RegRoi,
        margin_mm: f64,
        deformable: bool,
        contours: bool,
        subjects: Vec<Subject>,
    ) {
        if self.propagate_job.is_some() {
            return;
        }
        let Some(src) = self.slots[moving_slot].study.as_ref() else {
            self.error = Some("This needs a loaded source dataset".into());
            return;
        };
        let src_vol = src.volume.clone();
        let moving_series_uid = src
            .series
            .get(src.active_series)
            .map(|s| s.uid.clone())
            .unwrap_or_default();
        // The anchor on the source, as a frozen structure.
        let src_anchor = match anchor {
            RegRoi::Structure(i) => self.slots[moving_slot]
                .active_structures()
                .and_then(|ss| ss.rois.get(i))
                .map(crate::workflow::select::Structure::from_roi),
            RegRoi::Segmentation(i) => {
                let segs = self.slots[moving_slot].segs();
                segs.get(i).map(|seg| crate::workflow::select::Structure {
                    name: seg.name.clone(),
                    color: seg.color,
                    source: crate::workflow::select::Source::Mask {
                        mask: seg.mask.clone(),
                        grid: src_vol.grid(),
                    },
                })
            }
            RegRoi::Whole => None,
        };
        let Some(src_anchor) = src_anchor else {
            self.error = Some("Pick the structure to anchor the run on.".into());
            return;
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(g) = study.fourd_groups.get(group) else {
            self.error = Some("That 4D group is gone - pick it again.".into());
            return;
        };
        let phases = match crate::workflow::phases_of(g, &study.series) {
            Ok(p) => p,
            Err(e) => {
                self.error = Some(format!("{e}"));
                return;
            }
        };
        // The anchor on every phase: the contour drawn on that phase.
        let mut anchored = Vec::with_capacity(phases.len());
        for (label, series) in phases {
            let Some(st) =
                crate::workflow::select::find_on_series(study, &src_anchor.name, &series.uid, "")
            else {
                self.error = Some(format!(
                    "Phase {label} has no structure '{}'; an anchored run needs it contoured \
                     on every phase.",
                    src_anchor.name
                ));
                return;
            };
            anchored.push(anchored::AnchoredPhase {
                label,
                series,
                anchor: st,
            });
        }
        let base = self.current_reg_params(None, false);
        let rigid = anchored::default_rigid(&base);
        let deformable = deformable.then(|| anchored::default_deformable(&base));
        let req = anchored::AnchoredRequest {
            src_vol,
            src_anchor,
            subjects,
            phases: anchored,
            margin_mm: margin_mm.max(0.0),
            mode: if contours {
                anchored::AnchorMode::Contours
            } else {
                anchored::AnchorMode::Intensity
            },
            rigid,
            deformable,
            group_name: g.name.clone(),
            group,
            moving_slot,
            moving_series_uid,
        };
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.propagate_job = Some(Job::spawn(progress, move |p| {
            (slot, anchored::run(req, p).map(PropOutcome::Anchored))
        }));
    }

    /// Start the propagation (with its optional local refinement) on a
    /// worker thread.
    fn start_propagation(&mut self) {
        if self.propagate_job.is_some() {
            return;
        }
        let Some(d) = &self.propagate_dialog else {
            return;
        };
        if let PropTarget::Group { slot, group } = d.target {
            let moving_slot = d.src_slot;
            let anchor = d.anchor;
            let (margin, deformable, contours) =
                (d.anchor_margin_mm, d.anchor_deformable, d.anchor_contours);
            if anchor != RegRoi::Whole {
                let Some(src_grid) = self.slots[moving_slot]
                    .study
                    .as_ref()
                    .filter(|st| st.has_volume())
                    .map(|st| st.volume.grid())
                else {
                    self.error = Some("The source dataset shows no image volume.".into());
                    return;
                };
                // An anchored run carries the anchor itself as its check, so
                // nothing else need be ticked.
                let subjects = self.propagate_subjects(d, &src_grid).unwrap_or_default();
                self.start_anchored_run(
                    moving_slot,
                    slot,
                    group,
                    anchor,
                    margin,
                    deformable,
                    contours,
                    subjects,
                );
            } else {
                let structures = match self.propagate_structures(d) {
                    Ok(s) => s,
                    Err(e) => {
                        self.error = Some(format!("Propagation: {e}"));
                        return;
                    }
                };
                // The moving image: the one the group is already registered
                // against when that is a series of this dataset (a cardiac CT
                // registered from the registration module, on display or
                // not), otherwise the displayed one.
                let Some(study) = self.slots[moving_slot].study.as_ref() else {
                    return;
                };
                let registered = self
                    .group_registration
                    .as_ref()
                    .filter(|gr| {
                        gr.slot == slot && gr.group == group && gr.moving_slot == moving_slot
                    })
                    .and_then(|gr| {
                        study
                            .series
                            .iter()
                            .position(|se| se.uid == gr.moving_series_uid)
                    });
                let series = match registered {
                    Some(i) => i,
                    None if study.has_volume() => study.active_series,
                    None => {
                        self.error = Some("The source dataset shows no image volume.".into());
                        return;
                    }
                };
                self.start_group_run(
                    RegPick {
                        slot: moving_slot,
                        series,
                    },
                    slot,
                    group,
                    structures,
                );
            }
            return;
        }
        let Some(reg) = &self.registration else {
            self.error = Some("Run a registration first - propagation needs one.".into());
            return;
        };
        // The structures come from one of the two registered images and
        // land on the other. The transform maps fixed → moving, so landing
        // on the moving image runs through the inverse.
        let (src_slot, src_vol, dst_slot, dst_vol, dst_uid, use_inverse) = match d.from_side {
            FusionSide::Moving => (
                reg.moving_slot,
                reg.moving_vol.clone(),
                reg.fixed_slot,
                reg.fixed_vol.clone(),
                reg.fixed_uid.clone(),
                false,
            ),
            FusionSide::Fixed => (
                reg.fixed_slot,
                reg.fixed_vol.clone(),
                reg.moving_slot,
                reg.moving_vol.clone(),
                reg.moving_uid.clone(),
                true,
            ),
        };
        let fixed_slot = reg.fixed_slot;
        let fixed_vol = reg.fixed_vol.clone();
        let moving_vol = reg.moving_vol.clone();
        let (fixed_img, moving_img) = (
            RegImage {
                slot: reg.fixed_slot,
                uid: reg.fixed_uid.clone(),
                vol: reg.fixed_vol.clone(),
            },
            RegImage {
                slot: reg.moving_slot,
                uid: reg.moving_uid.clone(),
                vol: reg.moving_vol.clone(),
            },
        );
        let transform = reg.result.transform.clone();
        let mut d2 = PropagateDialog {
            src_slot,
            ..PropagateDialog::default()
        };
        d2.structs = d.structs.clone();
        d2.segs = d.segs.clone();
        let subjects = match self.propagate_subjects(&d2, &src_vol.grid()) {
            Ok(s) => s,
            Err(e) => {
                self.error = Some(format!("Propagation: {e}"));
                return;
            }
        };
        let Some(d) = &self.propagate_dialog else {
            return;
        };

        // The optional local refinement, run before anything is carried.
        let local = d.local;
        let margin = d.local_margin_mm;
        let region = if local == RegRoi::Whole {
            None
        } else if !self
            .registration
            .as_ref()
            .is_some_and(|r| r.shows_fixed(fixed_slot, &self.slots))
        {
            self.error = Some(
                "Local refinement: display the fixed image (click its series in the data \
                 tree) - the region is drawn on it."
                    .into(),
            );
            return;
        } else {
            match self.region_for(fixed_slot, local, margin) {
                Ok(r) => r,
                Err(e) => {
                    self.error = Some(format!("Local refinement: {e:#}"));
                    return;
                }
            }
        };
        let mut params = self.current_reg_params(region.clone(), true);
        params.start = Some(transform.clone());
        // A refinement is a deformation on top of what is there; a rigid
        // method would replace the alignment rather than refine it.
        if !params.method.is_deformable() {
            params.method = registration::RegMethod::PlastimatchBSpline;
        }
        let field_step = self.field_step_mm;

        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.propagate_job = Some(Job::spawn(progress, move |p| {
            let mut refined = None;
            let transform = if region.is_some() {
                p.set_phase(0.0, 0.6);
                p.set("Refining the registration on the region");
                match registration::register(&fixed_vol, &moving_vol, &params, p) {
                    Ok(result) => {
                        let field = VectorField::sample(
                            &fixed_vol,
                            &result.transform,
                            region.as_deref(),
                            field_step,
                        );
                        let t = result.transform.clone();
                        refined = Some(Box::new(RegOutcome {
                            result,
                            field,
                            region,
                            fixed: fixed_img,
                            moving: moving_img,
                        }));
                        t
                    }
                    Err(e) => return (dst_slot, Err(e)),
                }
            } else {
                transform
            };
            p.set_phase(
                if refined.is_some() { 0.6 } else { 0.0 },
                if refined.is_some() { 0.4 } else { 1.0 },
            );
            let items =
                propagate::propagate(&src_vol, &dst_vol, &transform, use_inverse, &subjects, p);
            (
                dst_slot,
                items.map(|items| PropOutcome::One {
                    items,
                    dst_uid,
                    dst_grid: dst_vol.grid(),
                    refined,
                }),
            )
        }));
    }

    /// A propagation run landed: install the masks (and the refinement).
    pub(super) fn on_propagation_done(&mut self, dst_slot: usize, out: PropOutcome) {
        let lines = match out {
            PropOutcome::One {
                items,
                dst_uid,
                dst_grid,
                refined,
            } => self.install_one(dst_slot, &dst_uid, &dst_grid, items, refined),
            PropOutcome::Group(g) => self.install_group(dst_slot, g),
            PropOutcome::Anchored(a) => {
                let mut lines = self.install_group(dst_slot, a.group);
                lines.push(String::new());
                lines.push("Anchor check (propagated against the phase's own contour)".into());
                for q in &a.qa {
                    lines.push(format!("   {} [{}]", q.line(), q.verdict()));
                }
                lines
            }
        };
        if let Some(d) = &mut self.propagate_dialog {
            d.summary = lines;
        }
        self.settings_gen += 1;
    }

    /// Results carried onto one of the two registered images.
    fn install_one(
        &mut self,
        dst_slot: usize,
        dst_uid: &str,
        dst_grid: &Grid,
        items: Vec<Propagated>,
        refined: Option<Box<RegOutcome>>,
    ) -> Vec<String> {
        if let Some(refined) = refined {
            self.install_registration(*refined);
            // The refinement is now the active registration - show the
            // section that reports and clears it.
            self.module_registration = true;
        }
        let Some(study) = &self.slots[dst_slot].study else {
            return Vec::new();
        };
        let study_uid = study
            .series
            .iter()
            .find(|se| se.uid == dst_uid)
            .map(|se| se.study_uid.clone())
            .unwrap_or_default();
        let src = self
            .registration
            .as_ref()
            .map(|r| {
                if r.fixed_uid == dst_uid {
                    r.moving_slot
                } else {
                    r.fixed_slot
                }
            })
            .unwrap_or(1 - dst_slot);
        let landing = self
            .propagate_dialog
            .as_ref()
            .map(|d| d.landing)
            .unwrap_or_default();
        let mut lines: Vec<String> = items.iter().map(|it| it.summary()).collect();
        let named: Vec<Propagated> = items
            .into_iter()
            .map(|mut it| {
                it.name = format!("{} (from {})", it.name, SLOT_NAMES[src]);
                it
            })
            .collect();
        match landing {
            Landing::Segmentation => {
                if self.slots[dst_slot].displayed_uid() == Some(dst_uid) {
                    let dims = dst_grid.dims;
                    for item in &named {
                        if item.voxels == 0 {
                            continue;
                        }
                        self.add_colored_segmentation(
                            dst_slot,
                            item.name.clone(),
                            item.color,
                            dims,
                            &item.mask,
                        );
                    }
                } else if let Some(label) =
                    self.land_seg_series(dst_slot, dst_uid, &study_uid, dst_grid, &named)
                {
                    lines.push(format!("▸ {label}"));
                }
            }
            Landing::StructureSet => {
                if let Some((label, names)) = self.land_rois(
                    dst_slot,
                    dst_uid,
                    &study_uid,
                    dst_grid,
                    &named,
                    "Propagated",
                ) {
                    lines.push(format!("▸ {label}: {}", names.join(", ")));
                }
            }
        }
        lines
    }

    /// File propagated masks as a new segmentation series bound to an image
    /// series of `slot`'s study (the way a phase of a group gets its own),
    /// for a destination that is not on display. Returns its label.
    fn land_seg_series(
        &mut self,
        slot: usize,
        series_uid: &str,
        study_uid: &str,
        grid: &Grid,
        items: &[Propagated],
    ) -> Option<String> {
        let study = self.slots[slot].study.as_mut()?;
        let n = study.seg_series.len() + 1;
        let mut series = crate::dicomseg::SegSeries::new(
            format!("Propagated {n}"),
            grid.clone(),
            series_uid.to_string(),
            study_uid.to_string(),
        );
        for it in items.iter().filter(|it| it.voxels > 0) {
            series.segs.push(Segmentation::from_label_map(
                it.name.clone(),
                it.color,
                grid.dims,
                &it.mask,
                1,
            ));
        }
        if series.segs.is_empty() {
            return None;
        }
        let label = series.label.clone();
        study.seg_series.push(series);
        self.settings_gen += 1;
        Some(label)
    }

    /// File propagated masks as contours in the structure set of one image
    /// series of `slot`'s study, keeping the visibility list in step when
    /// that set is the one on show.
    fn land_rois(
        &mut self,
        slot: usize,
        series_uid: &str,
        study_uid: &str,
        grid: &Grid,
        items: &[Propagated],
        new_set_label: &str,
    ) -> Option<(String, Vec<String>)> {
        let s = &mut self.slots[slot];
        let study = s.study.as_mut()?;
        let landed =
            land_in_structure_set(study, series_uid, study_uid, grid, items, new_set_label)?;
        if let Some(ss) = study.structure_sets.get(s.active_structs) {
            if ss.referenced_series_uid == series_uid {
                s.roi_visible.resize(ss.rois.len(), true);
            }
        }
        self.settings_gen += 1;
        Some(landed)
    }

    /// Results carried onto every phase of a 4D group.
    ///
    /// One segmentation series per phase, bound to that phase's image series
    /// so the tree files it under the right member and the views show it when
    /// that phase is displayed.
    fn install_group(&mut self, dst_slot: usize, g: GroupOutcome) -> Vec<String> {
        if self.slots[dst_slot].study.is_none() {
            return Vec::new();
        }
        let landing = self
            .propagate_dialog
            .as_ref()
            .map(|d| d.landing)
            .unwrap_or_default();
        let mut lines = Vec::new();
        let mut regs = Vec::new();
        for phase in g.phases {
            lines.push(format!("{} - {}", phase.label, phase.metric_line));
            regs.push(GroupPhaseReg {
                label: phase.label.clone(),
                series_uid: phase.series_uid.clone(),
                transform: phase.transform.clone(),
                metric_line: phase.metric_line.clone(),
            });
            for item in &phase.items {
                lines.push(format!("   {}", item.summary()));
            }
            match landing {
                Landing::Segmentation => {
                    let Some(series) = phase.seg_series(&g.group_name) else {
                        continue;
                    };
                    if let Some(study) = self.slots[dst_slot].study.as_mut() {
                        study.seg_series.push(series);
                        self.slots[dst_slot].active_seg_series = study.seg_series.len() - 1;
                        self.slots[dst_slot].active_seg = 0;
                    }
                }
                Landing::StructureSet => {
                    if let Some((label, names)) = self.land_rois(
                        dst_slot,
                        &phase.series_uid,
                        &phase.study_uid,
                        &phase.grid,
                        &phase.items,
                        &format!("{} {}", g.group_name, phase.label),
                    ) {
                        lines.push(format!("   ▸ {label}: {}", names.join(", ")));
                    }
                }
            }
        }
        // Keep the transforms: propagating another structure set onto the
        // same group now costs one load and one pull per phase, not another
        // registration.
        self.group_registration = Some(GroupRegistration {
            moving_slot: g.moving_slot,
            moving_series_uid: g.moving_series_uid,
            slot: dst_slot,
            group: g.group,
            group_name: g.group_name,
            phases: regs,
        });
        self.reg_gen += 1;
        lines
    }

    // -- the window --------------------------------------------------------

    /// The propagation section of the modules panel.
    pub(super) fn propagate_section(&mut self, ui: &mut egui::Ui) {
        // The module is on, so the section exists whether or not anything has
        // aimed it yet.
        if self.propagate_dialog.is_none() {
            let mut d = PropagateDialog::default();
            self.sync_propagate_lists(&mut d);
            self.propagate_dialog = Some(d);
        }
        let mut run = false;
        let mut cancel = false;
        let mut set_all: Option<bool> = None;

        // Everything the closure needs to read while `self` is borrowed.
        let registered = self.registration.as_ref().map(|r| {
            let (fixed, moving) = r.describe(&self.slots);
            (
                r.fixed_slot,
                r.moving_slot,
                fixed,
                moving,
                r.result.method.label().to_string(),
                r.result.region.clone(),
            )
        });
        let mut d = self.propagate_dialog.take().unwrap();
        // Through a registration the source is one of its two images, and
        // the structure list is that image's dataset's.
        if let (Some((fs, ms, ..)), PropTarget::Other) = (&registered, d.target) {
            d.src_slot = match d.from_side {
                FusionSide::Fixed => *fs,
                FusionSide::Moving => *ms,
            };
        }
        self.sync_propagate_lists(&mut d);
        let src_slot = d.src_slot;
        let dst_slot = 1 - src_slot;
        let struct_rows: Vec<(String, [u8; 3])> = self.slots[src_slot]
            .active_structures()
            .map(|ss| ss.rois.iter().map(|r| (r.name.clone(), r.color)).collect())
            .unwrap_or_default();
        let seg_rows: Vec<(String, [u8; 3], f64)> = {
            let spacing = self.slots[src_slot]
                .study
                .as_ref()
                .map(|s| s.volume.spacing)
                .unwrap_or([1.0; 3]);
            self.slots[src_slot]
                .segs()
                .iter()
                .map(|s| (s.name.clone(), s.color, s.volume_cm3(spacing)))
                .collect()
        };
        let local_choices = registered
            .as_ref()
            .map(|(fixed, ..)| self.region_choices_for(*fixed))
            .unwrap_or_default();
        let group_choices = self.propagate_group_choices();
        // Anchors are structures of the *source*: that is where they must
        // exist, and every phase must carry one of the same name.
        let anchor_choices = self.region_choices_for(src_slot);
        // A group that was removed while the module sat open leaves a stale
        // choice behind; fall back rather than run against nothing.
        if !matches!(d.target, PropTarget::Other)
            && !group_choices.iter().any(|(t, _)| *t == d.target)
        {
            d.target = PropTarget::Other;
        }
        let to_group = matches!(d.target, PropTarget::Group { .. });
        if !anchor_choices.iter().any(|(c, _)| *c == d.anchor) {
            d.anchor = RegRoi::Whole;
        }
        let anchored = to_group && d.anchor != RegRoi::Whole;
        // Transforms already made for exactly this group from exactly this
        // moving image: the run then costs one load and one pull per phase.
        // An anchored run always registers afresh: it answers a different
        // question from the plain one.
        let reuse = !anchored
            && match (d.target, &self.group_registration) {
                (PropTarget::Group { slot, group }, Some(gr)) => {
                    gr.slot == slot
                        && gr.group == group
                        && gr.moving_slot == src_slot
                        && self.slots[src_slot]
                            .study
                            .as_ref()
                            .and_then(|st| st.series.get(st.active_series))
                            .is_some_and(|se| se.uid == gr.moving_series_uid)
                }
                _ => false,
            };
        let n_phases = match d.target {
            PropTarget::Group { slot, group } => self.group_phase_count(slot, group),
            PropTarget::Other => 0,
        };

        egui::CollapsingHeader::new(egui::RichText::new("⇄ Structures propagation").strong())
            .id_salt("module_propagate")
            .default_open(true)
            .show(ui, |ui| {
                ui.label(
                    "Carries structures and segmentations onto another image. Every \
                 destination voxel is asked where it comes from, so nothing is left \
                 with holes.",
                );
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("From");
                    match (&registered, to_group) {
                        (Some((_, _, fixed, moving, _, _)), false) => {
                            ui.selectable_value(
                                &mut d.from_side,
                                FusionSide::Moving,
                                format!("moving image ({moving})"),
                            )
                            .on_hover_text("Its structures land on the fixed image");
                            ui.selectable_value(
                                &mut d.from_side,
                                FusionSide::Fixed,
                                format!("fixed image ({fixed})"),
                            )
                            .on_hover_text("Its structures land on the moving image");
                        }
                        _ => {
                            ui.selectable_value(&mut d.src_slot, 0, "A");
                            ui.selectable_value(&mut d.src_slot, 1, "B");
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("To");
                    let current = match d.target {
                        PropTarget::Other => format!("dataset {}", SLOT_NAMES[dst_slot]),
                        PropTarget::Group { .. } => group_choices
                            .iter()
                            .find(|(t, _)| *t == d.target)
                            .map(|(_, l)| l.clone())
                            .unwrap_or_default(),
                    };
                    egui::ComboBox::from_id_salt("prop_target")
                        .selected_text(current)
                        .width(200.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut d.target,
                                PropTarget::Other,
                                format!("dataset {}", SLOT_NAMES[dst_slot]),
                            );
                            for (target, label) in &group_choices {
                                ui.selectable_value(&mut d.target, *target, label);
                            }
                        });
                });
                ui.separator();
                match (to_group, &registered) {
                    // Against a group the module registers as it goes, one
                    // run per phase, so it needs no registration in advance.
                    (true, _) if reuse => {
                        ui.weak(format!(
                            "This group is already registered against this image, phase by \
                             phase, so the {n_phases} transforms are reused and only the \
                             structures are carried across."
                        ));
                    }
                    (true, _) => {
                        ui.weak(format!(
                            "The source volume is registered onto each of the {n_phases} \
                             phases in turn, and the structures are carried across that \
                             phase's own transform. Breathing moves the anatomy between \
                             phases, so one transform for the whole group would put them \
                             all where the reference phase is."
                        ));
                        ui.weak(
                            "Method and parameters come from the image registration \
                             module; a rigid choice is run deformably here.",
                        );
                    }
                    (false, None) => {
                        ui.colored_label(
                            alert_color(ui.visuals()),
                            "No active registration - run one in the registration module \
                             first, or send these to a 4D group instead.",
                        );
                    }
                    (false, Some((_, _, fixed, moving, method, region))) => {
                        ui.weak(format!(
                            "Using: {method}{}",
                            match region {
                                Some(r) => format!(" · restricted to {r}"),
                                None => String::new(),
                            }
                        ));
                        ui.weak(format!(
                            "Fixed {fixed}, moving {moving} - the transform is inverted \
                             automatically for the other direction."
                        ));
                    }
                }
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.small_button("All").clicked() {
                        set_all = Some(true);
                    }
                    if ui.small_button("None").clicked() {
                        set_all = Some(false);
                    }
                    let n = d.structs.iter().filter(|v| **v).count()
                        + d.segs.iter().filter(|v| **v).count();
                    ui.weak(format!("{n} selected"));
                });

                egui::ScrollArea::vertical()
                    .max_height(260.0)
                    .show(ui, |ui| {
                        if !struct_rows.is_empty() {
                            ui.label(egui::RichText::new("Structures").strong());
                            for (i, (name, color)) in struct_rows.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    if let Some(on) = d.structs.get_mut(i) {
                                        ui.checkbox(on, "");
                                    }
                                    ui.colored_label(
                                        Color32::from_rgb(color[0], color[1], color[2]),
                                        "◼",
                                    );
                                    ui.label(name);
                                });
                            }
                        }
                        if !seg_rows.is_empty() {
                            ui.add_space(4.0);
                            ui.label(egui::RichText::new("Segmentations").strong());
                            for (i, (name, color, cm3)) in seg_rows.iter().enumerate() {
                                ui.horizontal(|ui| {
                                    if let Some(on) = d.segs.get_mut(i) {
                                        ui.checkbox(on, "");
                                    }
                                    ui.colored_label(
                                        Color32::from_rgb(color[0], color[1], color[2]),
                                        "◼",
                                    );
                                    ui.label(name);
                                    ui.weak(format!("{cm3:.1} cm³"));
                                });
                            }
                        }
                        if struct_rows.is_empty() && seg_rows.is_empty() {
                            ui.weak("This dataset has nothing to propagate.");
                        }
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Land as");
                    ui.selectable_value(
                        &mut d.landing,
                        Landing::Segmentation,
                        "segmentation series",
                    )
                    .on_hover_text(
                        "A new segmentation series bound to the destination image: \
                         editable masks, convertible to RTSTRUCT later",
                    );
                    ui.selectable_value(&mut d.landing, Landing::StructureSet, "structure set")
                        .on_hover_text(
                            "Contours appended to the destination image's own RT structure \
                             set (the one that references it; a new set when there is none). \
                             On a 4DCT with one set per phase the target goes next to that \
                             phase's heart.",
                        );
                });
                ui.separator();
                if to_group {
                    // Against a group the run may be anchored on a structure
                    // the source and every phase carry.
                    egui::CollapsingHeader::new("Anchor on a structure")
                        .id_salt("prop_anchor")
                        .default_open(d.anchor != RegRoi::Whole)
                        .show(ui, |ui| {
                            ui.label(
                                "A structure contoured on the source and on every phase (the \
                                 heart of a cardiac CT and of a 4DCT) is matched first: its \
                                 centroids are aligned, a rigid fit is made on it alone, then \
                                 a local deformable refinement. The two images need not share \
                                 a frame of reference, and the structure travels along as the \
                                 check: its overlap with each phase's own contour is reported.",
                            );
                            ui.horizontal(|ui| {
                                ui.label("Anchor");
                                let current = anchor_choices
                                    .iter()
                                    .find(|(c, _)| *c == d.anchor)
                                    .map(|(_, l)| l.clone())
                                    .unwrap_or_else(|| "None (plain run)".into());
                                egui::ComboBox::from_id_salt("prop_anchor_roi")
                                    .selected_text(current)
                                    .width(200.0)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut d.anchor,
                                            RegRoi::Whole,
                                            "None (plain run)",
                                        );
                                        for (choice, label) in &anchor_choices {
                                            if *choice == RegRoi::Whole {
                                                continue;
                                            }
                                            ui.selectable_value(&mut d.anchor, *choice, label);
                                        }
                                    });
                            });
                            if d.anchor != RegRoi::Whole {
                                ui.horizontal(|ui| {
                                    ui.label("Margin");
                                    ui.add(
                                        egui::DragValue::new(&mut d.anchor_margin_mm)
                                            .speed(1.0)
                                            .range(0.0..=60.0)
                                            .suffix(" mm"),
                                    )
                                    .on_hover_text(
                                        "The phase's structure is grown by this much to bound \
                                         the registration; the boundary is what aligns it.",
                                    );
                                    ui.checkbox(&mut d.anchor_deformable, "Refine deformably")
                                        .on_hover_text(
                                            "After the rigid fit, a local B-spline on the same \
                                             region takes up what is not rigid. Off keeps the \
                                             alignment rigid.",
                                        );
                                });
                                ui.checkbox(&mut d.anchor_contours, "Match the contours")
                                    .on_hover_text(
                                        "The registration compares the anchor's surfaces \
                                         (signed distance maps of the two contours) instead \
                                         of the image intensities: a contrast-enhanced \
                                         cardiac CT and a plain 4DCT cannot be matched by \
                                         intensity, but their heart contours can.",
                                    );
                            }
                        });
                    ui.separator();
                }
                // A local refinement refines *the* active registration.
                // Against a group there is one registration per phase, each
                // made on the spot, so there is nothing here to refine.
                egui::CollapsingHeader::new("Refine locally first")
                    .id_salt("prop_local")
                    .open(to_group.then_some(false))
                    .default_open(false)
                    .show(ui, |ui| {
                        ui.label(
                            "A structure inside a larger one lands where the *larger* one's \
                         deformation puts it. Refining the registration on the enclosing \
                         structure first is what fixes that - and it only changes the \
                         transform inside that structure.",
                        );
                        ui.horizontal(|ui| {
                            ui.label("Region");
                            let current = local_choices
                                .iter()
                                .find(|(c, _)| *c == d.local)
                                .map(|(_, l)| l.clone())
                                .unwrap_or_else(|| "No refinement".into());
                            egui::ComboBox::from_id_salt("prop_region")
                                .selected_text(current)
                                .width(200.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(
                                        &mut d.local,
                                        RegRoi::Whole,
                                        "No refinement",
                                    );
                                    for (choice, label) in &local_choices {
                                        if *choice == RegRoi::Whole {
                                            continue;
                                        }
                                        ui.selectable_value(&mut d.local, *choice, label);
                                    }
                                });
                        });
                        if d.local != RegRoi::Whole {
                            ui.horizontal(|ui| {
                                ui.label("Margin");
                                ui.add(
                                    egui::DragValue::new(&mut d.local_margin_mm)
                                        .speed(1.0)
                                        .range(0.0..=60.0)
                                        .suffix(" mm"),
                                );
                            });
                            ui.weak(
                                "The refinement replaces the active registration, so the \
                             sidebar reports exactly what the propagation used.",
                            );
                        }
                    });

                ui.separator();
                match &self.propagate_job {
                    Some(job) => cancel = progress_row(ui, &job.progress),
                    None => {
                        ui.horizontal(|ui| {
                            let (label, hint, ready) = if to_group && reuse {
                                (
                                    format!("▶ Propagate to {n_phases} phases"),
                                    "One segmentation series per phase, each bound to that \
                                     phase's image series, through the transforms the \
                                     registration module already made",
                                    n_phases >= 2,
                                )
                            } else if anchored {
                                (
                                    format!("▶ Anchor and propagate to {n_phases} phases"),
                                    "Per phase: centroids matched, a rigid fit on the anchor, \
                                     the refinement, then the structures (and the anchor, as \
                                     the check) carried across",
                                    n_phases >= 2,
                                )
                            } else if to_group {
                                (
                                    format!("▶ Register and propagate to {n_phases} phases"),
                                    "One registration and one segmentation series per \
                                     phase, each bound to that phase's image series",
                                    n_phases >= 2,
                                )
                            } else {
                                (
                                    "▶ Propagate".to_string(),
                                    "Results land as editable segmentations on the other \
                                     dataset, convertible to RTSTRUCT like any other",
                                    registered.is_some(),
                                )
                            };
                            if ui
                                .add_enabled(ready, egui::Button::new(label))
                                .on_hover_text(hint)
                                .clicked()
                            {
                                run = true;
                            }
                        });
                    }
                }
                if !d.summary.is_empty() {
                    ui.separator();
                    ui.label(egui::RichText::new("Last run").strong());
                    for line in &d.summary {
                        ui.monospace(line);
                    }
                    ui.weak(
                        "A volume change is the deformation's doing: it is exactly what the \
                     Jacobian in the registration panel reports.",
                    );
                }
            });
        ui.separator();

        if let Some(v) = set_all {
            d.structs.iter_mut().for_each(|s| *s = v);
            d.segs.iter_mut().for_each(|s| *s = v);
        }
        self.propagate_dialog = Some(d);
        if cancel {
            if let Some(job) = &self.propagate_job {
                job.progress.cancel();
            }
        }
        if run {
            self.start_propagation();
        }
    }
}
