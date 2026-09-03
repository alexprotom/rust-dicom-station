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
use crate::propagate::{self, Finish, Propagated, Subject};
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
    /// The image the structures were drawn on: any series of either
    /// dataset. Through the active registration it is one of the two
    /// registered images.
    pub src: RegPick,
    /// Which structure set or segmentation series of that dataset the
    /// structures are taken from.
    pub set: Option<SetPick>,
    /// Ticks over the entries of `set`.
    pub ticked: Vec<bool>,
    /// What they land on.
    pub target: PropTarget,
    /// Refine the registration on this region of the *fixed* dataset first.
    pub local: RegRoi,
    pub local_margin_mm: f64,
    /// Against a group: the entry of `set` the run is anchored on (a
    /// structure every phase carries too), or `None` for a plain run.
    pub anchor: Option<usize>,
    pub anchor_margin_mm: f64,
    /// Anchored run: refine deformably after the rigid stage.
    pub anchor_deformable: bool,
    /// Anchored run: match the anchor's contours (distance maps) rather
    /// than the image intensities.
    pub anchor_contours: bool,
    /// Where the results are filed on the destination image.
    pub landing: Landing,
    /// What is done to each landed mask: closing and filling.
    pub finish: Finish,
    /// What the last run produced.
    pub summary: Vec<String>,
}

/// A structure set or a segmentation series of one dataset.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum SetPick {
    Structures(usize),
    Segmentations(usize),
}

impl Default for PropagateDialog {
    fn default() -> Self {
        PropagateDialog {
            src: RegPick { slot: 0, series: 0 },
            set: None,
            ticked: Vec::new(),
            target: PropTarget::Other,
            local: RegRoi::Whole,
            local_margin_mm: 10.0,
            anchor: None,
            anchor_margin_mm: 10.0,
            anchor_deformable: true,
            anchor_contours: true,
            landing: Landing::Segmentation,
            finish: Finish::default(),
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
    /// Show the propagation module and point it at the displayed series of
    /// `src_slot`.
    ///
    /// A running job owns the dialog (its results have to land somewhere), so
    /// this never replaces one that is in flight; it only aims it.
    pub(super) fn open_propagate_module(&mut self, src_slot: usize) {
        self.module_propagation = true;
        self.right_open = true;
        self.settings_gen += 1;
        let src = self.displayed_pick(src_slot).unwrap_or(RegPick {
            slot: src_slot,
            series: 0,
        });
        if self.propagate_job.is_some() {
            return;
        }
        let mut d = match self.propagate_dialog.take() {
            Some(mut d) => {
                d.src = src;
                d.set = None;
                d
            }
            None => PropagateDialog {
                src,
                ..Default::default()
            },
        };
        self.settle_propagate_dialog_set(&mut d);
        self.propagate_dialog = Some(d);
    }

    /// The displayed series of a slot as a pick.
    fn displayed_pick(&self, slot: usize) -> Option<RegPick> {
        let st = self.slots[slot].study.as_ref()?;
        (st.has_volume() && !st.series.is_empty()).then(|| RegPick {
            slot,
            series: st.active_series.min(st.series.len() - 1),
        })
    }

    /// The sets a dataset offers, as `(pick, label)`: structure sets first,
    /// then segmentation series, each with what it references.
    fn set_choices(&self, slot: usize) -> Vec<(SetPick, String)> {
        let Some(st) = self.slots[slot].study.as_ref() else {
            return Vec::new();
        };
        let series_no = |uid: &str| {
            st.series
                .iter()
                .position(|se| se.uid == uid)
                .map(|i| format!(" (on series {})", i + 1))
                .unwrap_or_default()
        };
        let mut out = Vec::new();
        for (i, ss) in st.structure_sets.iter().enumerate() {
            out.push((
                SetPick::Structures(i),
                format!("▣ {}{}", ss.label, series_no(&ss.referenced_series_uid)),
            ));
        }
        for (i, sr) in st.seg_series.iter().enumerate() {
            out.push((
                SetPick::Segmentations(i),
                format!("✏ {}{}", sr.label, series_no(&sr.referenced_series_uid)),
            ));
        }
        out
    }

    /// The entries of a set: `(name, colour, volume in cm³ when known)`.
    fn set_entries(&self, slot: usize, set: SetPick) -> Vec<(String, [u8; 3], Option<f64>)> {
        let Some(st) = self.slots[slot].study.as_ref() else {
            return Vec::new();
        };
        match set {
            SetPick::Structures(i) => st
                .structure_sets
                .get(i)
                .map(|ss| {
                    ss.rois
                        .iter()
                        .map(|r| (r.name.clone(), r.color, None))
                        .collect()
                })
                .unwrap_or_default(),
            SetPick::Segmentations(i) => st
                .seg_series
                .get(i)
                .map(|sr| {
                    sr.segs
                        .iter()
                        .map(|s| (s.name.clone(), s.color, Some(s.volume_cm3(sr.grid.spacing))))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }

    /// One entry of a set, frozen for a worker thread.
    fn set_structure(&self, slot: usize, set: SetPick, idx: usize) -> Option<Structure> {
        let st = self.slots[slot].study.as_ref()?;
        match set {
            SetPick::Structures(i) => st
                .structure_sets
                .get(i)?
                .rois
                .get(idx)
                .map(Structure::from_roi),
            SetPick::Segmentations(i) => Structure::from_segment(st.seg_series.get(i)?, idx),
        }
    }

    /// Keep the dialog's set and ticks valid for its source: a set that is
    /// gone, or none chosen yet, becomes the set drawn on the source series
    /// (the structure set referencing it, else the slot's active set, else
    /// the first there is), and the tick list follows the set's length.
    fn settle_propagate_dialog_set(&self, d: &mut PropagateDialog) {
        let choices = self.set_choices(d.src.slot);
        if d.set.is_some_and(|s| !choices.iter().any(|(c, _)| *c == s)) {
            d.set = None;
        }
        if d.set.is_none() {
            let st = self.slots[d.src.slot].study.as_ref();
            let uid = st
                .and_then(|st| st.series.get(d.src.series))
                .map(|se| se.uid.clone())
                .unwrap_or_default();
            let on_series = st.and_then(|st| {
                st.structure_sets
                    .iter()
                    .rposition(|ss| !uid.is_empty() && ss.referenced_series_uid == uid)
                    .map(SetPick::Structures)
            });
            let active = st.and_then(|st| {
                (!st.structure_sets.is_empty())
                    .then(|| SetPick::Structures(self.slots[d.src.slot].active_structs))
                    .filter(|p| choices.iter().any(|(c, _)| c == p))
            });
            d.set = on_series
                .or(active)
                .or_else(|| choices.first().map(|(c, _)| *c));
            d.ticked.clear();
            d.anchor = None;
        }
        let n = d
            .set
            .map(|set| self.set_entries(d.src.slot, set).len())
            .unwrap_or(0);
        d.ticked.resize(n, false);
        if d.anchor.is_some_and(|a| a >= n) {
            d.anchor = None;
        }
    }

    /// Everything ticked, frozen: contours as they are, painted
    /// segmentations with the lattice they are on. A run turns them into
    /// masks on whatever source lattice it maps from, which need not be
    /// loaded yet.
    fn propagate_structures(&self, d: &PropagateDialog) -> Result<Vec<Structure>, String> {
        let Some(set) = d.set else {
            return Err("pick a structure set or a segmentation series".into());
        };
        let out: Vec<Structure> = d
            .ticked
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .filter_map(|(i, _)| self.set_structure(d.src.slot, set, i))
            .collect();
        if out.is_empty() {
            return Err("tick at least one structure".into());
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
        finish: Finish,
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
                finish,
                group_name,
                group,
                moving_slot,
                moving_series_uid,
            };
            (slot, run_group(req, p).map(PropOutcome::Group))
        }));
    }

    /// Start a run against every phase of a 4D group anchored on a
    /// structure: the source's anchor is matched to its namesake on each
    /// phase (centroids, then a rigid fit on the structure, then optionally a
    /// local deformable refinement), and `structures` travel through that.
    ///
    /// This is how a cardiac CT meets a 4DCT: the two share no frame of
    /// reference, and only the heart is worth matching.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn start_anchored_run(
        &mut self,
        moving: RegPick,
        slot: usize,
        group: usize,
        src_anchor: Structure,
        margin_mm: f64,
        deformable: bool,
        contours: bool,
        finish: Finish,
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
            self.error = Some("The source series is gone - pick it again.".into());
            return;
        };
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
        // The anchor on every phase: the contour drawn on that phase.
        let mut anchored = Vec::with_capacity(phases.len());
        for (label, series) in phases {
            let Some(st) =
                crate::workflow::select::find_on_series(study, &src_anchor.name, &series.uid, "")
            else {
                self.error = Some(format!(
                    "Phase {label} has no structure '{}'; an anchored run needs it on every \
                     phase.",
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
        let group_name = g.name.clone();
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.propagate_job = Some(Job::spawn(progress, move |p| {
            let src_vol = match src_ready {
                Some(v) => v,
                None => {
                    p.set("Loading the source image");
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
                finish,
                group_name,
                group,
                moving_slot,
                moving_series_uid,
            };
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
        let finish = d.finish;
        if let PropTarget::Group { slot, group } = d.target {
            let src = d.src;
            let (margin, deformable, contours) =
                (d.anchor_margin_mm, d.anchor_deformable, d.anchor_contours);
            let anchor = d
                .anchor
                .and_then(|i| d.set.and_then(|set| self.set_structure(src.slot, set, i)));
            if let Some(anchor) = anchor {
                // The anchor carries itself as the check, so nothing else
                // need be ticked; what is ticked travels with it.
                let structures = self.propagate_structures(d).unwrap_or_default();
                self.start_anchored_run(
                    src, slot, group, anchor, margin, deformable, contours, finish, structures,
                );
            } else {
                let structures = match self.propagate_structures(d) {
                    Ok(s) => s,
                    Err(e) => {
                        self.error = Some(format!("Propagation: {e}"));
                        return;
                    }
                };
                self.start_group_run(src, slot, group, structures, finish);
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
        let src_uid = self.slots[d.src.slot]
            .study
            .as_ref()
            .and_then(|st| st.series.get(d.src.series))
            .map(|se| se.uid.clone())
            .unwrap_or_default();
        let from_moving = d.src.slot == reg.moving_slot && src_uid == reg.moving_uid;
        let from_fixed = d.src.slot == reg.fixed_slot && src_uid == reg.fixed_uid;
        if !from_moving && !from_fixed {
            self.error = Some(
                "Through a registration the structures come from one of its two images: pick \
                 the fixed or the moving image as the source."
                    .into(),
            );
            return;
        }
        let (src_slot, src_vol, dst_slot, dst_vol, dst_uid, use_inverse) = if from_moving {
            (
                reg.moving_slot,
                reg.moving_vol.clone(),
                reg.fixed_slot,
                reg.fixed_vol.clone(),
                reg.fixed_uid.clone(),
                false,
            )
        } else {
            (
                reg.fixed_slot,
                reg.fixed_vol.clone(),
                reg.moving_slot,
                reg.moving_vol.clone(),
                reg.moving_uid.clone(),
                true,
            )
        };
        let _ = src_slot;
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
        let structures = match self.propagate_structures(d) {
            Ok(s) => s,
            Err(e) => {
                self.error = Some(format!("Propagation: {e}"));
                return;
            }
        };
        let src_grid = src_vol.grid();
        let subjects: Vec<Subject> =
            match structures.iter().map(|s| s.subject_on(&src_grid)).collect() {
                Ok(v) => v,
                Err(e) => {
                    self.error = Some(format!("Propagation: {e:#}"));
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
                items.map(|mut items| {
                    finish.apply_all(&mut items, &dst_vol.grid(), p);
                    PropOutcome::One {
                        items,
                        dst_uid,
                        dst_grid: dst_vol.grid(),
                        refined,
                    }
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
            let src = self
                .displayed_pick(0)
                .or_else(|| self.displayed_pick(1))
                .unwrap_or(RegPick { slot: 0, series: 0 });
            let mut d = PropagateDialog {
                src,
                ..Default::default()
            };
            self.settle_propagate_dialog_set(&mut d);
            self.propagate_dialog = Some(d);
        }
        let mut run = false;
        let mut cancel = false;
        let mut set_all: Option<bool> = None;

        // Everything the closure needs to read while `self` is borrowed.
        let registered = self.registration.as_ref().map(|r| {
            let (fixed, moving) = r.describe(&self.slots);
            (
                RegImageRef {
                    slot: r.fixed_slot,
                    uid: r.fixed_uid.clone(),
                    label: fixed,
                },
                RegImageRef {
                    slot: r.moving_slot,
                    uid: r.moving_uid.clone(),
                    label: moving,
                },
                r.result.method.label().to_string(),
                r.result.region.clone(),
            )
        });
        let mut d = self.propagate_dialog.take().unwrap();
        let group_choices = self.propagate_group_choices();
        // A group that was removed while the module sat open leaves a stale
        // choice behind; fall back rather than run against nothing.
        if !matches!(d.target, PropTarget::Other)
            && !group_choices.iter().any(|(t, _)| *t == d.target)
        {
            d.target = PropTarget::Other;
        }
        let to_group = matches!(d.target, PropTarget::Group { .. });

        // The images the structures may come from: any series of either
        // dataset against a group; through a registration, its two images.
        let all_choices = self.reg_choices();
        let pick_uid = |p: RegPick| -> String {
            self.slots[p.slot]
                .study
                .as_ref()
                .and_then(|st| st.series.get(p.series))
                .map(|se| se.uid.clone())
                .unwrap_or_default()
        };
        let src_choices: Vec<(RegPick, String)> = match (&registered, to_group) {
            (Some((fixed, moving, _, _)), false) => all_choices
                .iter()
                .filter(|c| {
                    let uid = pick_uid(c.pick);
                    (c.pick.slot == moving.slot && uid == moving.uid)
                        || (c.pick.slot == fixed.slot && uid == fixed.uid)
                })
                .map(|c| {
                    let role = if c.pick.slot == moving.slot && pick_uid(c.pick) == moving.uid {
                        "moving"
                    } else {
                        "fixed"
                    };
                    (c.pick, format!("{} · {role}", c.label))
                })
                .collect(),
            _ => all_choices
                .iter()
                .map(|c| (c.pick, c.label.clone()))
                .collect(),
        };
        if !src_choices.iter().any(|(p, _)| *p == d.src) {
            if let Some((p, _)) = src_choices.first() {
                d.src = *p;
                d.set = None;
            }
        }
        self.settle_propagate_dialog_set(&mut d);
        let src_slot = d.src.slot;
        let dst_slot = 1 - src_slot;
        let set_choices = self.set_choices(src_slot);
        let entries = d
            .set
            .map(|set| self.set_entries(src_slot, set))
            .unwrap_or_default();
        let local_choices = registered
            .as_ref()
            .map(|(fixed, ..)| self.region_choices_for(fixed.slot))
            .unwrap_or_default();
        let anchored = to_group && d.anchor.is_some();
        // Transforms already made for exactly this group from exactly this
        // source image: the run then costs one load and one pull per phase.
        // An anchored run always registers afresh: it answers a different
        // question from the plain one.
        let src_uid = pick_uid(d.src);
        let reuse = !anchored
            && match (d.target, &self.group_registration) {
                (PropTarget::Group { slot, group }, Some(gr)) => {
                    gr.slot == slot
                        && gr.group == group
                        && gr.moving_slot == src_slot
                        && gr.moving_series_uid == src_uid
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
                    "Carries structures onto another image through a registration. Every \
                     destination voxel is asked where it comes from and how much of it is \
                     inside, so the volume is kept.",
                );
                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    ui.label("From image");
                    let current = src_choices
                        .iter()
                        .find(|(p, _)| *p == d.src)
                        .map(|(_, l)| l.clone())
                        .unwrap_or_else(|| "(none)".into());
                    egui::ComboBox::from_id_salt("prop_src")
                        .selected_text(current)
                        .width(230.0)
                        .show_ui(ui, |ui| {
                            for (pick, label) in &src_choices {
                                if ui.selectable_label(d.src == *pick, label).clicked() {
                                    d.src = *pick;
                                    d.set = None;
                                }
                            }
                        })
                        .response
                        .on_hover_text(
                            "The image the structures were drawn on. It is loaded for the run \
                             when it is not on display.",
                        );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Structures of");
                    let current = set_choices
                        .iter()
                        .find(|(c, _)| Some(*c) == d.set)
                        .map(|(_, l)| l.clone())
                        .unwrap_or_else(|| "(none)".into());
                    egui::ComboBox::from_id_salt("prop_set")
                        .selected_text(current)
                        .width(230.0)
                        .show_ui(ui, |ui| {
                            for (choice, label) in &set_choices {
                                if ui.selectable_label(d.set == Some(*choice), label).clicked() {
                                    d.set = Some(*choice);
                                    d.ticked.clear();
                                    d.anchor = None;
                                }
                            }
                        })
                        .response
                        .on_hover_text(
                            "The structure set or segmentation series the structures are \
                             taken from; the one drawn on the image is preselected.",
                        );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("To");
                    let current = match d.target {
                        PropTarget::Other => registered
                            .as_ref()
                            .map(|(fixed, moving, _, _)| {
                                if src_uid == moving.uid && src_slot == moving.slot {
                                    format!("{} (fixed image)", fixed.label)
                                } else {
                                    format!("{} (moving image)", moving.label)
                                }
                            })
                            .unwrap_or_else(|| "the registered image".into()),
                        PropTarget::Group { .. } => group_choices
                            .iter()
                            .find(|(t, _)| *t == d.target)
                            .map(|(_, l)| l.clone())
                            .unwrap_or_default(),
                    };
                    egui::ComboBox::from_id_salt("prop_target")
                        .selected_text(current)
                        .width(230.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut d.target,
                                PropTarget::Other,
                                "the other registered image",
                            );
                            for (target, label) in &group_choices {
                                ui.selectable_value(&mut d.target, *target, label);
                            }
                        });
                });
                ui.separator();
                match (to_group, &registered) {
                    (true, _) if reuse => {
                        ui.weak(format!(
                            "This group is already registered against this image, so the \
                             {n_phases} transforms are reused."
                        ));
                    }
                    (true, _) => {
                        ui.weak(format!(
                            "One registration per phase ({n_phases}), each with its own \
                             transform; method and parameters from the registration module."
                        ));
                    }
                    (false, None) => {
                        ui.colored_label(
                            alert_color(ui.visuals()),
                            "No active registration - run one in the registration module \
                             first, or send these to a 4D group instead.",
                        );
                    }
                    (false, Some((fixed, moving, method, region))) => {
                        ui.weak(format!(
                            "Using: {method}{}",
                            match region {
                                Some(r) => format!(" · restricted to {r}"),
                                None => String::new(),
                            }
                        ));
                        ui.weak(format!("Fixed {}, moving {}.", fixed.label, moving.label));
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
                    let n = d.ticked.iter().filter(|v| **v).count();
                    ui.weak(format!("{n} selected"));
                });

                egui::ScrollArea::vertical()
                    .max_height(260.0)
                    .show(ui, |ui| {
                        for (i, (name, color, cm3)) in entries.iter().enumerate() {
                            ui.horizontal(|ui| {
                                if let Some(on) = d.ticked.get_mut(i) {
                                    ui.checkbox(on, "");
                                }
                                ui.colored_label(
                                    Color32::from_rgb(color[0], color[1], color[2]),
                                    "◼",
                                );
                                ui.label(name);
                                if let Some(v) = cm3 {
                                    ui.weak(format!("{v:.1} cm³"));
                                }
                                if d.anchor == Some(i) {
                                    ui.weak("anchor");
                                }
                            });
                        }
                        if entries.is_empty() {
                            ui.weak("Nothing to propagate here.");
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
                             set (the one that references it; a new set when there is none).",
                        );
                });
                ui.horizontal(|ui| {
                    ui.label("Then");
                    let mut close = d.finish.close_mm > 0.0;
                    if ui
                        .checkbox(&mut close, "close gaps")
                        .on_hover_text(
                            "Morphological closing of each landed mask: pieces closer than \
                             twice the radius join into one surface. For a structure that \
                             arrives as a cloud of points.",
                        )
                        .changed()
                    {
                        d.finish.close_mm = if close { 2.0 } else { 0.0 };
                    }
                    if close {
                        ui.add(
                            egui::DragValue::new(&mut d.finish.close_mm)
                                .speed(0.5)
                                .range(0.5..=20.0)
                                .suffix(" mm"),
                        );
                    }
                    ui.checkbox(&mut d.finish.fill, "fill").on_hover_text(
                        "Fill the interior slice by slice: a surface becomes a solid.",
                    );
                });
                ui.separator();
                if to_group {
                    // Against a group the run may be anchored on a structure
                    // the source and every phase carry.
                    egui::CollapsingHeader::new("Anchor on a structure")
                        .id_salt("prop_anchor")
                        .default_open(d.anchor.is_some())
                        .show(ui, |ui| {
                            ui.label(
                                "Aligns on a structure that the source and every phase carry \
                                 (by name): centroids matched, a rigid fit on it, then a local \
                                 deformable refinement. It travels along as the check: its \
                                 overlap with each phase's own contour is reported.",
                            );
                            ui.horizontal(|ui| {
                                ui.label("Anchor");
                                let current = d
                                    .anchor
                                    .and_then(|i| entries.get(i))
                                    .map(|(n, _, _)| n.clone())
                                    .unwrap_or_else(|| "None (plain run)".into());
                                egui::ComboBox::from_id_salt("prop_anchor_roi")
                                    .selected_text(current)
                                    .width(200.0)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut d.anchor,
                                            None,
                                            "None (plain run)",
                                        );
                                        for (i, (name, _, _)) in entries.iter().enumerate() {
                                            ui.selectable_value(&mut d.anchor, Some(i), name);
                                        }
                                    });
                            });
                            if d.anchor.is_some() {
                                ui.horizontal(|ui| {
                                    ui.label("Margin");
                                    ui.add(
                                        egui::DragValue::new(&mut d.anchor_margin_mm)
                                            .speed(1.0)
                                            .range(0.0..=60.0)
                                            .suffix(" mm"),
                                    )
                                    .on_hover_text(
                                        "How far beyond the anchor the registration looks.",
                                    );
                                    ui.checkbox(&mut d.anchor_deformable, "Refine deformably")
                                        .on_hover_text(
                                            "A local B-spline on the anchor region after the \
                                             rigid fit. Off keeps the alignment rigid.",
                                        );
                                    ui.checkbox(&mut d.anchor_contours, "Match the contours")
                                        .on_hover_text(
                                            "Compare the anchor's surfaces (signed distance \
                                             maps) rather than the image intensities: works \
                                             across contrast, kernel and modality.",
                                        );
                                });
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
                                    "Through the transforms already made for this group",
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
                                    "One registration per phase, then the structures",
                                    n_phases >= 2,
                                )
                            } else {
                                (
                                    "▶ Propagate".to_string(),
                                    "Through the active registration onto its other image",
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
                        "Source, mapped and filed volume: the mapped one is what the \
                         deformation made of the source; closing and filling change the \
                         filed one.",
                    );
                }
            });
        ui.separator();

        if let Some(v) = set_all {
            d.ticked.iter_mut().for_each(|s| *s = v);
        }
        let _ = dst_slot;
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

/// One of the active registration's images, as the section reads it.
struct RegImageRef {
    slot: usize,
    uid: String,
    label: String,
}
