//! The *Registration* side-panel section: choosing a method, restricting it
//! to a structure, placing landmarks, starting the run, and everything the
//! result has to say afterwards.
//!
//! One section covers four algorithms because the choice between them is
//! the user's real question ("stochastic or dense? intensities or points?"),
//! and the rest of the conversation - direction, region, parameters,
//! analytics, fusion, vector field - is the same whichever they pick.

use anyhow::{anyhow, Result};

use super::*;
use crate::app::combine::ItemRef;
use crate::registration::{analysis, LandmarkKernel, RegParams, Warp};

/// What restricts the next registration: everything, or one structure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RegRoi {
    /// The whole fixed image (a global registration).
    Whole,
    /// One ROI of the fixed workspace's active structure set.
    Structure(usize),
    /// One painted / segmented mask of the fixed workspace.
    Segmentation(usize),
}

/// Where the next registration starts its search (see
/// [`crate::registration::Init`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RegInit {
    /// The identity when the images overlap, the centres of gravity when
    /// they do not.
    Auto,
    Identity,
    /// Match the centres of gravity.
    Gravity,
    /// Match the centroids of one structure of the fixed workspace with its
    /// namesake on the moving workspace.
    Structure(RegRoi),
}

/// What a registration run hands back: the result, the vector field sampled
/// from it, and the region it was restricted to (kept so the field can be
/// re-sampled later without rebuilding the mask).
pub(super) struct RegOutcome {
    pub result: RegistrationResult,
    pub field: VectorField,
    pub region: Option<Arc<RegionMask>>,
    /// The two images the run was on.
    pub fixed: RegImage,
    pub moving: RegImage,
}

/// One registered image: where it lives and its volume.
#[derive(Clone)]
pub(super) struct RegImage {
    pub slot: usize,
    pub uid: String,
    pub vol: Arc<Volume>,
}

/// An image the module can be pointed at: a series of a loaded workspace,
/// with its label for the pickers.
#[derive(Clone)]
pub(super) struct RegChoice {
    pub pick: RegPick,
    pub label: String,
    /// Whether it is the volume its slot displays right now.
    pub displayed: bool,
}

/// The field the views draw for `t` on `vol`: the whole displacement, or
/// with the rigid part left out ([`Transform3::warp_only`]) when that is
/// asked for and there is a deformation to show.
pub(super) fn sample_field(
    vol: &Volume,
    t: &Transform3,
    region: Option<&RegionMask>,
    step_mm: f64,
    warp_only: bool,
) -> VectorField {
    match warp_only.then(|| t.warp_only()).flatten() {
        Some(w) => VectorField::sample(vol, &w, region, step_mm),
        None => VectorField::sample(vol, t, region, step_mm),
    }
}

impl ViewerApp {
    // -- region ------------------------------------------------------------

    /// The regions a workspace offers, as `(choice, label)`.
    pub(super) fn region_choices_for(&self, slot: usize) -> Vec<(RegRoi, String)> {
        let mut out = vec![(RegRoi::Whole, "Whole image".to_string())];
        if let Some(ss) = self.slots[slot].active_structures() {
            for (i, roi) in ss.rois.iter().enumerate() {
                if roi.contours.iter().any(|c| c.points.len() >= 3) {
                    out.push((RegRoi::Structure(i), format!("▣ {}", roi.name)));
                }
            }
        }
        for (i, seg) in self.slots[slot].segs().iter().enumerate() {
            if seg.count > 0 {
                out.push((RegRoi::Segmentation(i), format!("✏ {}", seg.name)));
            }
        }
        out
    }

    /// The label of the current region choice, or `None` when it no longer
    /// exists (the structure set changed under it).
    fn region_label(&self, slot: usize) -> Option<String> {
        self.region_choices_for(slot)
            .into_iter()
            .find(|(c, _)| *c == self.reg_roi)
            .map(|(_, l)| l)
    }

    /// Turn the panel's current region choice into a mask.
    pub(super) fn build_region(&self, slot: usize) -> Result<Option<Arc<RegionMask>>> {
        self.region_for(slot, self.reg_roi, self.reg_margin_mm)
    }

    /// Turn a region choice on `slot` into a dilated voxel mask.
    pub(super) fn region_for(
        &self,
        slot: usize,
        choice: RegRoi,
        margin_mm: f64,
    ) -> Result<Option<Arc<RegionMask>>> {
        let RegRoi::Whole = choice else {
            let study = self.slots[slot]
                .study
                .as_ref()
                .ok_or_else(|| anyhow!("workspace {} is not loaded", SLOT_NAMES[slot]))?;
            let vol = &study.volume;
            let (mask, name) = match choice {
                RegRoi::Structure(i) => {
                    let roi = self.slots[slot]
                        .active_structures()
                        .and_then(|ss| ss.rois.get(i))
                        .ok_or_else(|| anyhow!("the selected structure is gone"))?;
                    let m = segmentation::rasterize_roi(&vol.grid(), roi).ok_or_else(|| {
                        anyhow!(
                            "'{}' has no planar contour inside the displayed volume",
                            roi.name
                        )
                    })?;
                    (m, roi.name.clone())
                }
                RegRoi::Segmentation(i) => {
                    let seg = self.slots[slot]
                        .segs()
                        .get(i)
                        .ok_or_else(|| anyhow!("the selected segmentation is gone"))?;
                    (seg.mask.clone(), seg.name.clone())
                }
                RegRoi::Whole => unreachable!(),
            };
            let region = RegionMask::from_mask(vol, &mask, name.clone(), margin_mm)
                .ok_or_else(|| anyhow!("'{name}' is empty on this volume"))?;
            return Ok(Some(Arc::new(region)));
        };
        Ok(None)
    }

    /// The initialisation choices the fixed workspace offers.
    fn init_choices_for(&self, slot: usize) -> Vec<(RegInit, String)> {
        let mut out = vec![
            (RegInit::Auto, "Automatic".to_string()),
            (RegInit::Identity, "Identity".to_string()),
            (RegInit::Gravity, "Centres of gravity".to_string()),
        ];
        for (choice, label) in self.region_choices_for(slot) {
            if choice != RegRoi::Whole {
                out.push((RegInit::Structure(choice), format!("Centroids of {label}")));
            }
        }
        out
    }

    /// Turn the panel's initialisation choice into what the engine takes:
    /// for a structure, its centroid on the fixed workspace and the centroid
    /// of the structure of the same name on the moving one.
    ///
    /// The moving side's structure is the one drawn on the moving *series*
    /// (a 4DCT carries one heart per phase), found through the series it
    /// references; the moving volume must be on display for its lattice.
    pub(super) fn build_init(&self, fixed: RegPick, moving: RegPick) -> Result<registration::Init> {
        use crate::registration::Init;
        let RegInit::Structure(choice) = self.reg_init else {
            return Ok(match self.reg_init {
                RegInit::Identity => Init::Identity,
                RegInit::Gravity => Init::CenterOfGravity,
                _ => Init::Auto,
            });
        };
        let region = self
            .region_for(fixed.slot, choice, 0.0)?
            .ok_or_else(|| anyhow!("pick a structure to start from"))?;
        let name = region.name.clone();
        let fstudy = self.slots[fixed.slot]
            .study
            .as_ref()
            .ok_or_else(|| anyhow!("workspace {} is not loaded", SLOT_NAMES[fixed.slot]))?;
        let mstudy = self.slots[moving.slot]
            .study
            .as_ref()
            .ok_or_else(|| anyhow!("workspace {} is not loaded", SLOT_NAMES[moving.slot]))?;
        let fgrid = fstudy.volume.grid();
        let fc = crate::motion::centroid_mm(region.mask(), &fgrid)
            .ok_or_else(|| anyhow!("'{name}' is empty on the fixed image"))?;
        let mseries = mstudy
            .series
            .get(moving.series)
            .ok_or_else(|| anyhow!("the moving series is gone"))?;
        let ms = crate::workflow::select::find_on_series(mstudy, &name, &mseries.uid, "")
            .ok_or_else(|| {
                anyhow!(
                    "workspace {} has no structure '{name}' to match the centroids with",
                    SLOT_NAMES[moving.slot]
                )
            })?;
        let mvol = self
            .pick_displayed_volume(moving)
            .ok_or_else(|| anyhow!("display the moving image to start from a structure"))?;
        let mgrid = mvol.grid();
        let mc = crate::motion::centroid_mm(&ms.mask_on(&mgrid)?, &mgrid)
            .ok_or_else(|| anyhow!("'{name}' is empty on the moving image"))?;
        Ok(Init::Points {
            fixed: fc,
            moving: mc,
        })
    }

    // -- the images ---------------------------------------------------------

    /// Every image series either workspace offers.
    pub(super) fn reg_choices(&self) -> Vec<RegChoice> {
        let mut out = Vec::new();
        for slot in self.open_slots() {
            let name = SLOT_NAMES[slot];
            let Some(st) = self.slots[slot].study.as_ref() else {
                continue;
            };
            for (i, se) in st.series.iter().enumerate() {
                let desc: String = se.description.chars().take(28).collect();
                out.push(RegChoice {
                    pick: RegPick { slot, series: i },
                    label: format!("{name} · {} {desc}", i + 1),
                    displayed: st.has_volume() && i == st.active_series,
                });
            }
        }
        out
    }

    /// Keep the two picks pointing at series that exist: a pick whose
    /// workspace went, or whose series is gone, falls back to the displayed
    /// series of its slot, and the moving image to the other workspace's.
    fn settle_reg_picks(&mut self) {
        let displayed = |slot: usize, slots: &[StudySlot; MAX_WORKSPACES]| -> Option<RegPick> {
            let st = slots[slot].study.as_ref()?;
            (!st.series.is_empty()).then(|| RegPick {
                slot,
                series: st.active_series.min(st.series.len() - 1),
            })
        };
        let valid = |p: RegPick, slots: &[StudySlot; MAX_WORKSPACES]| {
            slots[p.slot]
                .study
                .as_ref()
                .is_some_and(|st| p.series < st.series.len())
        };
        if !valid(self.reg_fixed, &self.slots) {
            if let Some(p) = self
                .open_slots()
                .into_iter()
                .find_map(|s| displayed(s, &self.slots))
            {
                self.reg_fixed = p;
            }
        }
        if !valid(self.reg_moving, &self.slots) || self.reg_moving == self.reg_fixed {
            let other = self
                .other_open(self.reg_fixed.slot)
                .unwrap_or(self.reg_fixed.slot);
            let fallback = displayed(other, &self.slots).or_else(|| {
                // One workspace: the next series of the same one, when it has
                // more than one - a cardiac CT beside its 4DCT.
                let st = self.slots[self.reg_fixed.slot].study.as_ref()?;
                (st.series.len() > 1).then(|| RegPick {
                    slot: self.reg_fixed.slot,
                    series: (self.reg_fixed.series + 1) % st.series.len(),
                })
            });
            if let Some(p) = fallback {
                self.reg_moving = p;
            }
        }
    }

    /// The series a pick names, and whether its slot displays it.
    fn pick_series(&self, p: RegPick) -> Option<(loader::SeriesInfo, bool)> {
        let st = self.slots[p.slot].study.as_ref()?;
        let se = st.series.get(p.series)?.clone();
        Some((se, st.has_volume() && st.active_series == p.series))
    }

    /// The volume of a pick when its slot displays it.
    fn pick_displayed_volume(&self, p: RegPick) -> Option<Arc<Volume>> {
        let (_, shown) = self.pick_series(p)?;
        shown.then(|| self.slots[p.slot].study.as_ref().unwrap().volume.clone())
    }

    /// Install a finished run as the active registration.
    pub(super) fn install_registration(&mut self, out: RegOutcome) {
        self.registration = Some(ActiveRegistration {
            result: out.result,
            fixed_slot: out.fixed.slot,
            fixed_uid: out.fixed.uid,
            fixed_vol: out.fixed.vol,
            moving_slot: out.moving.slot,
            moving_uid: out.moving.uid,
            moving_vol: out.moving.vol,
            field: Arc::new(out.field),
            region: out.region,
            group_phase: None,
            struct_dice: None,
        });
        self.reg_gen += 1;
    }

    /// Score every structure the two workspaces have in common.
    ///
    /// The overlap statistic in the analysis block is measured on a tissue
    /// threshold and says whether the *images* line up. This says whether the
    /// anatomy did, which is the question a plan is judged on - and it can
    /// only be asked where the same structure was drawn on both sides.
    ///
    /// Names are matched case-insensitively and nothing else is assumed: a
    /// structure of the fixed workspace is paired with the first structure of
    /// the moving workspace that shares its name, whether either of them is a
    /// contour or a segmentation.
    pub(super) fn score_registration_structures(&mut self) {
        let Some(reg) = &self.registration else {
            return;
        };
        let (fixed_slot, moving_slot) = (reg.fixed_slot, reg.moving_slot);
        let fixed_vol = reg.fixed_vol.clone();
        let moving_vol = reg.moving_vol.clone();
        let transform = reg.result.transform.clone();
        let fixed_grid = fixed_vol.grid();

        // Name -> the moving workspace's item of that name.
        let moving: Vec<(ItemRef, String)> = self.combine_candidates(moving_slot);
        let mut scores: Vec<StructDice> = Vec::new();
        for (item, _) in self.combine_candidates(fixed_slot) {
            let Some((fmask, fgrid, name, color)) = self.item_mask_grid(fixed_slot, item) else {
                continue;
            };
            let Some((mitem, _)) = moving.iter().find(|(mi, _)| {
                self.item_name(moving_slot, *mi)
                    .is_some_and(|n| n.eq_ignore_ascii_case(&name))
            }) else {
                continue;
            };
            let Some((mmask, mgrid, _, _)) = self.item_mask_grid(moving_slot, *mitem) else {
                continue;
            };
            // Both masks have to reach the fixed volume's lattice: the
            // structure's own may be a segmentation series on a different
            // one.
            let fmask = if fgrid.matches(&fixed_grid) {
                fmask
            } else {
                crate::dicomseg::resample_mask(&fmask, &fgrid, &fixed_grid)
            };
            // The moving mask on the moving volume's own lattice: what the
            // propagation starts from either way.
            let moving_on_its_own_grid = if mgrid.matches(&moving_vol.grid()) {
                mmask
            } else {
                crate::dicomseg::resample_mask(&mmask, &mgrid, &moving_vol.grid())
            };
            let subject = crate::propagate::Subject {
                name: name.clone(),
                color,
                mask: moving_on_its_own_grid,
                surface_cm3: None,
                keep_shape: false,
            };
            // Both numbers are measured the same way - the mask carried onto
            // the fixed lattice by the same code - so that the pair of them
            // says what the registration changed and nothing else. Scoring
            // "before" by a plain resample instead would charge the
            // registration for the interpolation the propagation costs, and
            // a perfect result would read below 1.
            let carried = |t: &Transform3| -> Option<Vec<u8>> {
                crate::propagate::propagate(
                    &moving_vol,
                    &fixed_vol,
                    t,
                    false,
                    std::slice::from_ref(&subject),
                    &progress::Quiet,
                )
                .ok()
                .and_then(|mut v| v.pop().map(|pr| pr.mask))
            };
            // The transform maps fixed patient coordinates to moving ones,
            // so arriving on the fixed lattice uses it as it is; the
            // identity is where the two workspaces started.
            let after_mask = carried(&transform);
            let before_mask = carried(&Transform3::rigid_only(
                crate::registration::RigidTransform::identity(crate::geometry::Vec3::ZERO),
            ));
            let (Some(before), Some(after)) = (
                before_mask
                    .as_ref()
                    .and_then(|m| crate::motion::dice(&fmask, m)),
                after_mask
                    .as_ref()
                    .and_then(|m| crate::motion::dice(&fmask, m)),
            ) else {
                continue;
            };
            let vox = fixed_vol.voxel_cm3();
            scores.push(StructDice {
                name,
                color,
                after,
                before,
                fixed_cm3: fmask.iter().filter(|v| **v != 0).count() as f64 * vox,
                moving_cm3: after_mask
                    .map(|m| m.iter().filter(|v| **v != 0).count() as f64 * vox)
                    .unwrap_or(0.0),
            });
        }
        if let Some(reg) = &mut self.registration {
            reg.struct_dice = Some(scores);
        }
    }

    // -- running -----------------------------------------------------------

    /// Everything a run needs, from the current panel state.
    pub(super) fn current_reg_params(
        &self,
        region: Option<Arc<RegionMask>>,
        refine: bool,
    ) -> RegParams {
        let start = if refine {
            self.registration
                .as_ref()
                .filter(|r| r.fixed_slot == self.reg_fixed.slot)
                .map(|r| r.result.transform.clone())
        } else {
            None
        };
        RegParams {
            method: self.reg_method,
            levels: self.reg_levels,
            iterations: self.reg_iterations,
            samples: self.reg_samples,
            grid_spacing_mm: self.reg_grid_mm,
            fixed_threshold: self.reg_threshold,
            regularization: self.reg_regularization,
            metric: self.reg_metric,
            stride: 1,
            landmark: self.reg_landmark,
            landmarks: self.reg_landmarks.clone(),
            region,
            init: registration::Init::Auto,
            start,
        }
    }

    /// Start a registration (or a refinement of the active one) between
    /// the two picked images. An image that is not on display is loaded on
    /// the worker thread first.
    pub(super) fn start_registration(&mut self, refine: bool) {
        if self.reg_job.is_some() {
            return;
        }
        self.settle_reg_picks();
        let (fpick, mpick) = (self.reg_fixed, self.reg_moving);
        if fpick == mpick {
            self.error = Some("The fixed and the moving image are the same series.".into());
            return;
        }
        let (Some((fseries, fshown)), Some((mseries, mshown))) =
            (self.pick_series(fpick), self.pick_series(mpick))
        else {
            self.error = Some("Pick a fixed and a moving image.".into());
            return;
        };
        let fixed_slot = fpick.slot;
        // A region and a structure start come from contours rasterized on
        // the fixed image, which the module reads from the displayed volume.
        if !fshown
            && (self.reg_roi != RegRoi::Whole || matches!(self.reg_init, RegInit::Structure(_)))
        {
            self.error = Some(
                "Display the fixed image (click its series in the data tree) to use a region \
                 or a structure as the start."
                    .into(),
            );
            return;
        }
        let region = match self.build_region(fixed_slot) {
            Ok(r) => r,
            Err(e) => {
                self.error = Some(format!("Local registration: {e:#}"));
                return;
            }
        };
        let mut params = self.current_reg_params(region.clone(), refine);
        match self.build_init(fpick, mpick) {
            Ok(init) => params.init = init,
            Err(e) => {
                self.error = Some(format!("Registration start: {e:#}"));
                return;
            }
        }
        let fixed_ready = fshown.then(|| self.pick_displayed_volume(fpick)).flatten();
        let moving_ready = mshown.then(|| self.pick_displayed_volume(mpick)).flatten();
        if params.method == RegMethod::PlastimatchLandmark && params.landmarks.is_empty() {
            self.error = Some(
                "The landmark warp needs paired points: put the crosshair on the same \
                 anatomy in both workspaces and press ➕ Add pair (turn off crosshair \
                 linking first, or both crosshairs move together)."
                    .into(),
            );
            return;
        }
        let step = self.field_step_mm;
        let warp_only = self.field_warp_only;
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.reg_job = Some(Job::spawn(progress, move |p| {
            let load = |ready: Option<Arc<Volume>>, se: &loader::SeriesInfo, what: &str| match ready
            {
                Some(v) => Ok(v),
                None => {
                    p.set(format!("Loading the {what} image"));
                    loader::load_series_volume(se, p).map(|(v, _, _)| Arc::new(v))
                }
            };
            let run = || -> anyhow::Result<RegOutcome> {
                let fixed = load(fixed_ready, &fseries, "fixed")?;
                let moving = load(moving_ready, &mseries, "moving")?;
                let result = registration::register(&fixed, &moving, &params, p)?;
                p.set("Sampling the vector field");
                let field = sample_field(
                    &fixed,
                    &result.transform,
                    region.as_deref(),
                    step,
                    warp_only,
                );
                Ok(RegOutcome {
                    result,
                    field,
                    region,
                    fixed: RegImage {
                        slot: fpick.slot,
                        uid: fseries.uid.clone(),
                        vol: fixed,
                    },
                    moving: RegImage {
                        slot: mpick.slot,
                        uid: mseries.uid.clone(),
                        vol: moving,
                    },
                })
            };
            (fixed_slot, run())
        }));
    }

    /// Install any transform read from a file (a DICOM REG object, a
    /// deformation grid) as the active registration, exactly as if it had
    /// been computed.
    ///
    /// A Deformable Spatial Registration's grid arrives here exactly as a
    /// REG matrix does, so everything downstream - fusion, the crosshair
    /// link, the analytics, the vector field, propagation - works on it
    /// without knowing where it came from.
    pub(super) fn apply_external_transform(
        &mut self,
        transform: Transform3,
        method: RegMethod,
        fixed_slot: usize,
    ) {
        let Some(moving_slot) = self.other_open(fixed_slot) else {
            self.error = Some("A transform pairs two workspaces; open another one first.".into());
            return;
        };
        self.apply_external_transform_between(transform, method, fixed_slot, moving_slot);
    }

    /// The same, with both sides named: a transform read from a file says
    /// which frames of reference it maps, and with more than two workspaces
    /// open only the caller knows which pair it meant.
    pub(super) fn apply_external_transform_between(
        &mut self,
        transform: Transform3,
        method: RegMethod,
        fixed_slot: usize,
        moving_slot: usize,
    ) {
        let transform = Arc::new(transform);
        let (Some(fstudy), Some(mstudy)) = (
            &self.slots[fixed_slot].study,
            &self.slots[moving_slot].study,
        ) else {
            self.error =
                Some("A transform from a file pairs two loaded workspaces; load them both.".into());
            return;
        };
        if !fstudy.has_volume() || !mstudy.has_volume() {
            return;
        }
        let uid_of = |st: &LoadedStudy| {
            st.series
                .get(st.active_series)
                .map(|se| se.uid.clone())
                .unwrap_or_default()
        };
        // A transform installed from elsewhere (a REG object in the tree, a
        // propagation) needs the section that shows and clears it.
        self.module_registration = true;
        let vol = fstudy.volume.clone();
        let mut analysis = analysis::analyse(&vol, &transform, None);
        analysis.overlap = analysis::overlap(&vol, &mstudy.volume, &transform, None);
        let field = sample_field(
            &vol,
            &transform,
            None,
            self.field_step_mm,
            self.field_warp_only,
        );
        self.registration = Some(ActiveRegistration {
            result: RegistrationResult {
                transform,
                method,
                metric: Metric::MeanSquares,
                initial_metric: 0.0,
                final_metric: 0.0,
                iterations_run: 0,
                elapsed_secs: 0.0,
                region: None,
                analysis,
            },
            fixed_slot,
            fixed_uid: uid_of(fstudy),
            fixed_vol: vol,
            moving_slot,
            moving_uid: uid_of(mstudy),
            moving_vol: mstudy.volume.clone(),
            field: Arc::new(field),
            region: None,
            group_phase: None,
            struct_dice: None,
        });
        self.fusion_on = true;
        self.reg_gen += 1;
        let cursor = self.slots[fixed_slot].cursor;
        self.set_cursor(fixed_slot, cursor, usize::MAX);
    }

    pub(super) fn clear_registration(&mut self) {
        if let Some(job) = &self.reg_job {
            job.progress.cancel();
        }
        if let Some(job) = &self.field_job {
            job.progress.cancel();
        }
        self.registration = None;
        // The per-phase transforms belong to a pair of images too: when the
        // workspace that made them goes, so do they.
        self.group_registration = None;
        self.reg_group = None;
        self.fusion_on = false;
        self.reg_gen += 1;
    }

    /// Drop the active registration and nothing else: what the Clear button
    /// does to a phase that was only put on display from the group
    /// registration, which stays for the next run and the next look.
    fn clear_shown_phase(&mut self) {
        if let Some(job) = &self.field_job {
            job.progress.cancel();
        }
        self.registration = None;
        self.fusion_on = false;
        self.reg_gen += 1;
    }

    /// Put phase `i` of the group registration on display as the active
    /// registration - fusion, crosshair link and vector field, exactly as
    /// after a registration run - without running anything. A phase that
    /// is not the displayed series of its workspace is read and shown
    /// first, and installed when it arrives (see `pending_phase_field`).
    pub(super) fn show_group_phase(&mut self, i: usize) {
        let Some(gr) = &self.group_registration else {
            return;
        };
        let Some(ph) = gr.phases.get(i) else {
            return;
        };
        let (slot, moving_slot) = (gr.slot, gr.moving_slot);
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let displayed = study
            .series
            .get(study.active_series)
            .is_some_and(|s| s.uid == ph.series_uid);
        if !displayed {
            match study.series.iter().position(|s| s.uid == ph.series_uid) {
                Some(idx) => {
                    self.pending_phase_field = Some((slot, ph.series_uid.clone(), i));
                    self.start_series_switch(slot, idx);
                }
                None => {
                    self.error = Some(format!(
                        "Phase {} is no longer in workspace {}.",
                        ph.label, SLOT_NAMES[slot]
                    ));
                }
            }
            return;
        }
        let transform = (*ph.transform).clone();
        let deformable = !transform.warp.is_none();
        let shown = format!("{} · {}", gr.group_name, ph.label);
        let metrics = ph.metrics;
        // A reused transform carries no measurement of its own, and is
        // filed like a matrix that was handed over: no row of zeros.
        let method = match (metrics.is_some(), deformable) {
            (false, _) => RegMethod::Given,
            (true, true) => RegMethod::ElastixBSpline,
            (true, false) => RegMethod::ElastixRigid,
        };
        // The rigid part of such a run is the jump between two frames of
        // reference - a cardiac CT onto a 4DCT phase, hundreds of mm - and
        // drawn whole the field would show that jump and nothing else.
        if deformable {
            self.field_warp_only = true;
        }
        self.apply_external_transform_between(transform, method, slot, moving_slot);
        if let Some(r) = &mut self.registration {
            r.group_phase = Some(shown);
            if let Some(m) = metrics {
                r.result.metric = if m.tag == Metric::MutualInformation.tag() {
                    Metric::MutualInformation
                } else {
                    Metric::MeanSquares
                };
                r.result.initial_metric = m.initial;
                r.result.final_metric = m.final_value;
                r.result.iterations_run = m.iterations;
                r.result.elapsed_secs = m.secs;
            }
        }
        self.field_on = true;
        if self.field_style == FieldStyle::None {
            self.field_style = FieldStyle::Arrows;
        }
    }

    /// After a group run: make the transform of the phase the group's
    /// workspace displays the active registration, quietly - the fusion and
    /// the field stay as they were switched - so the registration module has
    /// it ready. A registration of the user's own is left alone.
    pub(super) fn show_displayed_group_phase(&mut self) {
        if self
            .registration
            .as_ref()
            .is_some_and(|r| r.group_phase.is_none())
        {
            return;
        }
        let Some(gr) = &self.group_registration else {
            return;
        };
        let Some(uid) = self.slots[gr.slot]
            .study
            .as_ref()
            .and_then(|st| st.series.get(st.active_series))
            .map(|s| s.uid.clone())
        else {
            return;
        };
        let Some(i) = gr.phases.iter().position(|p| p.series_uid == uid) else {
            return;
        };
        let (fusion, field) = (self.fusion_on, self.field_on);
        self.show_group_phase(i);
        self.fusion_on = fusion;
        self.field_on = field;
    }

    /// A phase asked for by [`Self::show_group_phase`] whose series has now
    /// arrived on display: install it. Given up on when the read ended and
    /// something else is on display.
    pub(super) fn poll_pending_phase_field(&mut self) {
        let Some((slot, uid, i)) = self.pending_phase_field.clone() else {
            return;
        };
        if self.loading.is_some() || self.pending_switch.is_some() {
            return;
        }
        self.pending_phase_field = None;
        let displayed = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.series.get(st.active_series))
            .is_some_and(|s| s.uid == uid);
        if displayed {
            self.show_group_phase(i);
        }
    }

    /// Re-sample the vector field after the lattice step changed.
    pub(super) fn rebuild_field(&mut self) {
        if self.field_job.is_some() {
            return;
        }
        let Some(reg) = &self.registration else {
            return;
        };
        let vol = reg.fixed_vol.clone();
        let t = reg.result.transform.clone();
        let region = reg.region.clone();
        let step = self.field_step_mm;
        let warp_only = self.field_warp_only;
        let progress = Arc::new(Progress::default());
        self.field_job = Some(Job::spawn(progress, move |_| {
            sample_field(&vol, &t, region.as_deref(), step, warp_only)
        }));
    }

    /// Add a landmark pair from the two crosshairs.
    fn add_landmark_pair(&mut self) {
        let (fixed_slot, moving_slot) = (self.reg_fixed.slot, self.reg_moving.slot);
        let point = |s: &StudySlot| -> Option<Vec3> {
            let st = s.study.as_ref()?;
            let c = s.cursor;
            Some(st.volume.voxel_to_patient(c[0], c[1], c[2]))
        };
        if fixed_slot == moving_slot {
            self.error = Some(
                "Landmarks are picked with the two crosshairs, one per workspace: load the \
                 moving image in the other workspace for that."
                    .into(),
            );
            return;
        }
        let (Some(f), Some(m)) = (
            point(&self.slots[fixed_slot]),
            point(&self.slots[moving_slot]),
        ) else {
            self.error = Some("Load both workspaces before placing landmarks".into());
            return;
        };
        let n = self.reg_landmarks.len() + 1;
        self.reg_landmarks
            .push(LandmarkPair::new(format!("L{n}"), f, m));
    }

    // -- the panel section -------------------------------------------------

    /// Every transform this module has to show, and whether any is
    /// deformable.
    ///
    /// The active registration first - it is the one *Apply as the
    /// registration* acts on - then, after a group run, each phase's own
    /// transform. Ten grids one under the other would be a wall of numbers;
    /// one grid with a picker over it is the same information a phase at a
    /// time.
    pub(super) fn reg_matrix_choices(&self) -> (Vec<matrix_edit::MatrixChoice>, bool) {
        let mut out = Vec::new();
        let mut deformable = false;
        if let Some(r) = self.registration.as_ref() {
            deformable |= r.result.method.is_deformable();
            out.push(matrix_edit::MatrixChoice {
                label: r.result.method.label().to_string(),
                m: r.result.transform.as_matrix(),
            });
        }
        if let Some(gr) = self.group_registration.as_ref() {
            for ph in &gr.phases {
                deformable |= !ph.transform.warp.is_none();
                out.push(matrix_edit::MatrixChoice {
                    label: ph.label.clone(),
                    m: ph.transform.as_matrix(),
                });
            }
        }
        (out, deformable)
    }

    pub(super) fn registration_section(&mut self, ui: &mut egui::Ui) {
        let both = self.both_volumes();
        // The section is worth showing while two workspaces are loaded, while a
        // result is on display, while a run is in flight (that is where its
        // progress and its Cancel button live), and while one workspace holds a
        // 4D group, which can be registered against a volume of its own.
        let any_group = self.open_slots().into_iter().any(|slot| {
            self.slots[slot]
                .study
                .as_ref()
                .is_some_and(|st| !st.fourd_groups.is_empty())
        });
        if !both
            && !any_group
            && self.registration.is_none()
            && self.reg_job.is_none()
            && self.group_registration.is_none()
        {
            // The module is switched on, so the section says what it is
            // waiting for rather than leaving an empty panel.
            egui::CollapsingHeader::new(egui::RichText::new("Image registration").strong())
                .default_open(false)
                .show(ui, |ui| {
                    ui.weak(
                        "Load a second workspace (File > Add DICOM folder), or a workspace \
                         with more than one image series - registration aligns one image \
                         onto another",
                    );
                });
            ui.separator();
            return;
        }
        let mut run: Option<bool> = None;
        let mut cancel = false;
        let mut clear = false;
        let mut propagate_from: Option<usize> = None;
        let mut resample = false;
        let mut add_landmark = false;
        let mut drop_landmark: Option<usize> = None;
        let mut clear_landmarks = false;
        let mut save_field = false;
        let mut run_group: Option<(RegPick, usize, usize)> = None;
        let mut clear_group = false;
        let mut show_phase: Option<usize> = None;
        let mut hide_field = false;
        let mut score_structs = false;
        let mut apply_matrix = false;
        // 4D groups either workspace offers, keyed the way `reg_group` is.
        let group_choices: Vec<((usize, usize), String)> = self
            .propagate_group_choices()
            .into_iter()
            .filter_map(|(t, label)| match t {
                crate::app::propagate_win::PropTarget::Group { slot, group } => {
                    Some(((slot, group), label))
                }
                _ => None,
            })
            .collect();
        // A group removed while the module sat open leaves a stale choice.
        if self
            .reg_group
            .is_some_and(|k| !group_choices.iter().any(|(g, _)| *g == k))
        {
            self.reg_group = None;
        }

        egui::CollapsingHeader::new(egui::RichText::new("Image registration").strong())
            .default_open(false)
            .show(ui, |ui| {
                if let Some(job) = &self.reg_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                    ui.add(egui::ProgressBar::new(job.progress.frac()).show_percentage());
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                    return;
                }

                // ---- the two images ----
                // Any series of either workspace can be the fixed or the moving
                // image - the displayed ones, a phase of a 4DCT, a cardiac CT
                // against a phase of the same workspace. The fixed image may
                // also be every phase of a 4D group: one registration per
                // phase, since the phases differ by breathing, which is the
                // whole reason the acquisition exists.
                self.settle_reg_picks();
                let choices = self.reg_choices();
                let fixed_slot = self.reg_fixed.slot;
                let label_of = |p: RegPick| {
                    choices
                        .iter()
                        .find(|c| c.pick == p)
                        .map(|c| c.label.clone())
                        .unwrap_or_else(|| "(none)".into())
                };
                // Everything the run is made of, one form: the two
                // images, the method, the region and where the search
                // starts.
                form::form(ui, "reg_setup", |f| {
                    f.row_tip(
                        "Fixed image",
                        "The image the other is aligned onto; the transform maps its \
                         coordinates into the moving image's. A series that is not on \
                         display is loaded for the run.",
                        |ui| {
                            let current = match self.reg_group {
                                None => label_of(self.reg_fixed),
                                Some(k) => group_choices
                                    .iter()
                                    .find(|(g, _)| *g == k)
                                    .map(|(_, l)| l.clone())
                                    .unwrap_or_default(),
                            };
                            egui::ComboBox::from_id_salt("reg_fixed_image")
                                .selected_text(current)
                                .width(230.0)
                                .show_ui(ui, |ui| {
                                    for c in &choices {
                                        let mark = if c.displayed { " (displayed)" } else { "" };
                                        if ui
                                            .selectable_label(
                                                self.reg_group.is_none()
                                                    && self.reg_fixed == c.pick,
                                                format!("{}{mark}", c.label),
                                            )
                                            .clicked()
                                        {
                                            self.reg_fixed = c.pick;
                                            self.reg_group = None;
                                        }
                                    }
                                    if !group_choices.is_empty() {
                                        ui.separator();
                                        for (key, label) in &group_choices {
                                            ui.selectable_value(
                                                &mut self.reg_group,
                                                Some(*key),
                                                format!("every phase of {label}"),
                                            );
                                        }
                                    }
                                });
                        },
                    );
                    f.row_tip(
                        "Moving image",
                        "The image that is moved and deformed onto the fixed one",
                        |ui| {
                            egui::ComboBox::from_id_salt("reg_moving_image")
                                .selected_text(label_of(self.reg_moving))
                                .width(230.0)
                                .show_ui(ui, |ui| {
                                    for c in &choices {
                                        let mark = if c.displayed { " (displayed)" } else { "" };
                                        ui.selectable_value(
                                            &mut self.reg_moving,
                                            c.pick,
                                            format!("{}{mark}", c.label),
                                        );
                                    }
                                });
                        },
                    );
                    if self.reg_group.is_some() {
                        f.wide(|ui| {
                            ui.weak(
                                "Every phase is registered on its own against the moving \
                                 image. The transforms are kept, so sending structures to \
                                 the same group afterwards costs no registration at all.",
                            );
                        });
                    } else if self.reg_fixed.slot == self.reg_moving.slot {
                        f.wide(|ui| {
                            ui.weak(
                                "Two series of one workspace. The fusion overlay needs each \
                                 on display in its own workspace: load the same folder as \
                                 the other workspace to see it; propagation works either \
                                 way.",
                            );
                        });
                    }

                    // ---- method ----
                    f.row("Method", |ui| {
                        egui::ComboBox::from_id_salt("reg_method")
                            .selected_text(self.reg_method.short())
                            .width(190.0)
                            .show_ui(ui, |ui| {
                                for m in RegMethod::ALL {
                                    ui.selectable_value(&mut self.reg_method, m, m.short())
                                        .on_hover_text(m.hint());
                                }
                            });
                    });
                    f.wide(|ui| {
                        ui.weak(self.reg_method.hint());
                    });

                    // ---- region ----
                    let region_choices = self.region_choices_for(fixed_slot);
                    if self.region_label(fixed_slot).is_none() {
                        self.reg_roi = RegRoi::Whole;
                    }
                    f.row_tip(
                        "Region",
                        "Restrict the registration to one structure of the fixed \
                         workspace. Samples come from inside it only and the B-spline \
                         lattice covers it alone, so a small structure can be aligned at a \
                         fine grid - and, when it refines an existing result, the rest of \
                         the patient keeps that result untouched.",
                        |ui| {
                            let current = self
                                .region_label(fixed_slot)
                                .unwrap_or_else(|| "Whole image".into());
                            egui::ComboBox::from_id_salt("reg_roi")
                                .selected_text(current)
                                .width(190.0)
                                .show_ui(ui, |ui| {
                                    for (choice, label) in &region_choices {
                                        ui.selectable_value(&mut self.reg_roi, *choice, label);
                                    }
                                });
                        },
                    );
                    if self.reg_roi != RegRoi::Whole {
                        f.row_tip(
                            "Margin",
                            "The structure is grown by this much before sampling. Without a \
                             margin nothing outside the structure constrains its boundary, \
                             and the boundary is what you are aligning.",
                            |ui| {
                                ui.add(
                                    egui::DragValue::new(&mut self.reg_margin_mm)
                                        .speed(1.0)
                                        .range(0.0..=60.0)
                                        .suffix(" mm"),
                                );
                            },
                        );
                    }

                    // ---- initialisation ----
                    let init_choices = self.init_choices_for(fixed_slot);
                    if !init_choices.iter().any(|(c, _)| *c == self.reg_init) {
                        self.reg_init = RegInit::Auto;
                    }
                    f.row_tip(
                        "Start from",
                        "Where the search begins. The engines take steps of a few \
                         millimetres, so two images that do not overlap at the identity \
                         (different frames of reference: a cardiac CT and a 4DCT) never \
                         find each other. Automatic keeps the identity when they overlap \
                         and matches the centres of gravity when they do not; a structure \
                         contoured on both workspaces matches its centroids, which is the \
                         surest start for an organ.",
                        |ui| {
                            let current = init_choices
                                .iter()
                                .find(|(c, _)| *c == self.reg_init)
                                .map(|(_, l)| l.clone())
                                .unwrap_or_default();
                            egui::ComboBox::from_id_salt("reg_init")
                                .selected_text(current)
                                .width(190.0)
                                .show_ui(ui, |ui| {
                                    for (choice, label) in &init_choices {
                                        ui.selectable_value(&mut self.reg_init, *choice, label);
                                    }
                                });
                        },
                    );
                });

                // ---- parameters ----
                egui::CollapsingHeader::new("Parameters")
                    .id_salt("reg_params")
                    .default_open(false)
                    .show(ui, |ui| self.parameter_rows(ui));

                // ---- landmarks ----
                if self.reg_method == RegMethod::PlastimatchLandmark {
                    let residuals =
                        match self.registration.as_ref().map(|r| &r.result.transform.warp) {
                            Some(Warp::Rbf(w)) if w.centers.len() == self.reg_landmarks.len() => {
                                Some(w.residuals())
                            }
                            _ => None,
                        };
                    egui::CollapsingHeader::new(format!(
                        "Landmarks ({})",
                        self.reg_landmarks.len()
                    ))
                    .id_salt("reg_landmarks")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if tip_button(
                                ui,
                                "➕ Add pair",
                                "Take the crosshair of each workspace as one pair. Put \
                                 both crosshairs on the same anatomy first - and turn \
                                 off View ▶ Sync crosshairs, or they move together.",
                            ) {
                                add_landmark = true;
                            }
                            if ui
                                .add_enabled(
                                    !self.reg_landmarks.is_empty(),
                                    egui::Button::new("Clear all"),
                                )
                                .clicked()
                            {
                                clear_landmarks = true;
                            }
                        });
                        if self.reg_landmarks.is_empty() {
                            ui.weak("No pairs yet.");
                        }
                        egui::ScrollArea::vertical()
                            .max_height(160.0)
                            .show(ui, |ui| {
                                for (i, l) in self.reg_landmarks.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        if ui.small_button("🗑").clicked() {
                                            drop_landmark = Some(i);
                                        }
                                        let d = l.displacement();
                                        ui.monospace(format!("{}: {:.1} mm", l.name, d.length()))
                                            .on_hover_text(format!(
                                        "fixed ({:.1}, {:.1}, {:.1})\nmoving ({:.1}, {:.1}, {:.1})",
                                        l.fixed.x,
                                        l.fixed.y,
                                        l.fixed.z,
                                        l.moving.x,
                                        l.moving.y,
                                        l.moving.z
                                    ));
                                        if let Some(r) = residuals.as_ref().and_then(|r| r.get(i)) {
                                            ui.weak(format!("residual {r:.2} mm"));
                                        }
                                    });
                                }
                            });
                    });
                }

                // ---- run ----
                if let Some((gslot, group)) = self.reg_group {
                    let n = self.group_phase_count(gslot, group);
                    let moving_ok = self.pick_series(self.reg_moving).is_some();
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                n >= 2 && moving_ok && self.propagate_job.is_none(),
                                egui::Button::new(format!("▶ Register {n} phases")),
                            )
                            .on_hover_text("One registration per phase, against the moving image")
                            .clicked()
                        {
                            run_group = Some((self.reg_moving, gslot, group));
                        }
                    });
                    if self.propagate_job.is_some() {
                        ui.weak("A run against a group is already going.");
                    } else if !moving_ok {
                        ui.weak("Pick a moving image.");
                    }
                } else {
                    let can_refine = self
                        .registration
                        .as_ref()
                        .is_some_and(|r| r.fixed_slot == fixed_slot)
                        && self.reg_method.is_deformable();
                    ui.horizontal(|ui| {
                        if enabled_tip_button(
                            ui,
                            both,
                            "▶ Register",
                            "Recover the transform from scratch",
                        ) {
                            run = Some(false);
                        }
                        if enabled_tip_button(
                            ui,
                            both && can_refine,
                            "▶ Refine",
                            "Recover a correction on top of the active registration and \
                             add the two together - how a local registration is meant to \
                             be used after a global one",
                        ) {
                            run = Some(true);
                        }
                    });
                    if !both {
                        ui.weak("Load two workspaces (comparison mode) first");
                    }
                    // A transform that is already known does not have to be
                    // recovered: type it in and apply it, and everything
                    // downstream - fusion, the crosshair link, propagation,
                    // the vector field - reads it as it reads any other.
                    egui::CollapsingHeader::new("Transform matrix")
                        .default_open(false)
                        .show(ui, |ui| {
                            // Every transform this module has produced: the
                            // active registration, and - after a group run -
                            // each phase's own. One grid, a picker above it.
                            let (choices, deformable) = self.reg_matrix_choices();
                            if choices.is_empty() {
                                ui.weak(
                                    "Nothing has been registered yet, so there is no \
                                     transform to show. Type one in and tick *Use this \
                                     matrix* to hand it over.",
                                );
                            }
                            matrix_edit::matrix_editor_multi(
                                ui,
                                &mut self.reg_matrix,
                                &mut self.reg_matrix_pick,
                                &choices,
                            );
                            // A B-spline or a landmark warp is not a matrix;
                            // what the grid can show of it is the rigid part
                            // it starts from, and taking that over drops the
                            // deformation. Better said than discovered.
                            if deformable {
                                ui.weak(
                                    "the transform is deformable: this is its rigid part, \
                                     and using it drops the deformation",
                                );
                            }
                            if tip_widget(
                                ui,
                                both && self.reg_matrix.use_it,
                                egui::Button::new("▶ Apply as the registration"),
                                "Install these numbers as the active registration of the \
                                 two workspaces, without running anything. It replaces \
                                 whatever registration is active.",
                            ) {
                                apply_matrix = true;
                            }
                        });
                }

                // ---- what a group run left behind ----
                let shown_phase = self
                    .registration
                    .as_ref()
                    .and_then(|r| r.group_phase.clone());
                if let Some(gr) = &self.group_registration {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!("✔ {} phase by phase", gr.group_name)).strong(),
                    );
                    run_report::facts(
                        ui,
                        "reg_group_facts",
                        &[
                            ("Group", gr.group_name.clone()),
                            (
                                "Moving image",
                                format!("workspace {}", SLOT_NAMES[gr.moving_slot]),
                            ),
                            ("Phases", gr.phases.len().to_string()),
                        ],
                    );
                    ui.add_space(4.0);
                    // One row per phase, the numbers in columns: the phase
                    // whose metric barely moved is found by scanning down a
                    // column, not by reading ten sentences.
                    egui::ScrollArea::horizontal()
                        .id_salt("reg_group_rows")
                        .show(ui, |ui| {
                            egui::Grid::new("reg_group_grid")
                                .striped(true)
                                .spacing([10.0, 2.0])
                                .show(ui, |ui| {
                                    run_report::head(ui, "Phase");
                                    run_report::head(ui, "Metric ▶");
                                    run_report::head(ui, "Iters");
                                    run_report::head(ui, "t, s");
                                    run_report::head(ui, "Vector field");
                                    ui.end_row();
                                    for (pi, ph) in gr.phases.iter().enumerate() {
                                        ui.label(&ph.label);
                                        match &ph.metrics {
                                            Some(m) => {
                                                ui.monospace(format!(
                                                    "{} {:.0} ▶ {:.0}",
                                                    m.tag, m.initial, m.final_value
                                                ))
                                                .on_hover_text(&ph.metric_line);
                                                ui.monospace(format!("{}", m.iterations));
                                                ui.monospace(format!("{:.1}", m.secs));
                                            }
                                            None => {
                                                // A reused transform has no
                                                // measurement of its own.
                                                ui.weak(&ph.metric_line);
                                                ui.weak("-");
                                                ui.weak("-");
                                            }
                                        }
                                        // The phase's transform as the active
                                        // registration: its fusion and its
                                        // deformation field, drawn as after
                                        // any other registration.
                                        let on_show = shown_phase.as_deref()
                                            == Some(
                                                format!("{} · {}", gr.group_name, ph.label)
                                                    .as_str(),
                                            );
                                        let field_shown = on_show && self.field_on;
                                        if ui
                                            .selectable_label(
                                                field_shown,
                                                if field_shown {
                                                    "👁 on the views"
                                                } else {
                                                    "👁 show"
                                                },
                                            )
                                            .on_hover_text(
                                                "Draw this phase's deformation field on the views \
                                                 (and in 3D): its transform becomes the active \
                                                 registration, with the fusion overlay and the \
                                                 Vector field settings below, and the phase is put \
                                                 on display if it is not. Click again to hide the \
                                                 field. The group registration stays as it is.",
                                            )
                                            .clicked()
                                        {
                                            if field_shown {
                                                hide_field = true;
                                            } else {
                                                show_phase = Some(pi);
                                            }
                                        }
                                        ui.end_row();
                                    }
                                });
                        });
                    ui.add_space(4.0);
                    if tip_button(
                        ui,
                        "Clear group registration",
                        "The next run against this group registers again",
                    ) {
                        clear_group = true;
                    }
                }

                // ---- result ----
                if let Some(reg) = &self.registration {
                    ui.separator();
                    let res = &reg.result;
                    let (fixed, moving) = reg.describe(&self.slots);
                    ui.label(
                        egui::RichText::new(format!(
                            "✔ {}  ({moving} ▶ {fixed})",
                            res.method.label()
                        ))
                        .strong(),
                    );
                    if let Some(g) = &reg.group_phase {
                        ui.weak(format!(
                            "Phase {g} of the group registration above, on display. \
                             Clearing it here leaves the group as it is."
                        ));
                    }
                    // A matrix that was handed over was not optimized, so
                    // "MSD 0.0 ▶ 0.0 (0 iters)" would be a row of zeros
                    // pretending to be a measurement; those rows are left
                    // empty and `facts` drops them. The analysis below is
                    // computed from the transform itself and does mean
                    // something, so that stays.
                    let m = (res.method != RegMethod::Given).then(|| res.metrics());
                    run_report::facts(
                        ui,
                        "reg_result_facts",
                        &[
                            ("Method", res.method.label().to_string()),
                            ("Fixed", fixed.clone()),
                            ("Moving", moving.clone()),
                            ("Region", res.region.clone().unwrap_or_default()),
                            (
                                "Metric ▶",
                                m.map(|m| {
                                    format!("{} {:.0} ▶ {:.0}", m.tag, m.initial, m.final_value)
                                })
                                .unwrap_or_default(),
                            ),
                            (
                                "Iters",
                                m.map(|m| m.iterations.to_string()).unwrap_or_default(),
                            ),
                            (
                                "t, s",
                                m.map(|m| format!("{:.1}", m.secs)).unwrap_or_default(),
                            ),
                            ("Transform", res.transform.warp.describe()),
                        ],
                    );

                    egui::CollapsingHeader::new("Analysis")
                        .id_salt("reg_analysis")
                        .default_open(true)
                        .show(ui, |ui| {
                            score_structs |= analysis_rows(
                                ui,
                                res,
                                self.slots[reg.fixed_slot].active_structures(),
                                &res.transform,
                                reg.struct_dice.as_deref(),
                            );
                        });

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.fusion_on, "Fusion overlay on");
                        ui.selectable_value(
                            &mut self.fusion_side,
                            FusionSide::Fixed,
                            format!("fixed ({fixed})"),
                        )
                        .on_hover_text("The moving image warped onto the fixed one");
                        ui.selectable_value(
                            &mut self.fusion_side,
                            FusionSide::Moving,
                            format!("moving ({moving})"),
                        )
                        .on_hover_text("The fixed image carried back onto the moving one");
                    });
                    let shown = match self.fusion_side {
                        FusionSide::Fixed => reg.shows_fixed(reg.fixed_slot, &self.slots),
                        FusionSide::Moving => reg.shows_moving(reg.moving_slot, &self.slots),
                    };
                    if self.fusion_on && !shown {
                        ui.weak(
                            "That image is not on display: click its series in the data \
                             tree to see the overlay.",
                        );
                    }
                    ui.add(
                        egui::Slider::new(&mut self.fusion_weight, 0.0..=1.0).text("Fusion blend"),
                    );

                    egui::CollapsingHeader::new("Vector field")
                        .id_salt("reg_field")
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.checkbox(&mut self.field_on, "Show the deformation field")
                                .on_hover_text(
                                    "Draw the recovered displacement in every view - and \
                                     in the 3D window - instead of leaving it implicit in \
                                     the fusion colours",
                                );
                            if ui
                                .add_enabled(
                                    !res.transform.warp.is_none(),
                                    egui::Checkbox::new(
                                        &mut self.field_warp_only,
                                        "Deformation only (leave the rigid part out)",
                                    ),
                                )
                                .on_hover_text(
                                    "Draw what the B-spline adds on top of the rigid \
                                     alignment rather than the whole displacement. A \
                                     registration between two frames of reference - a \
                                     cardiac CT onto a 4DCT phase - moves every point by the \
                                     same hundreds of millimetres, and drawn whole the field \
                                     shows that jump and hides the deformation.",
                                )
                                .changed()
                            {
                                resample = true;
                            }
                            ui.horizontal(|ui| {
                                ui.label("Style");
                                for s in FieldStyle::ALL {
                                    ui.selectable_value(&mut self.field_style, s, s.label());
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Spacing");
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut self.field_step_mm)
                                            .speed(1.0)
                                            .range(2.0..=60.0)
                                            .suffix(" mm"),
                                    )
                                    .on_hover_text("Lattice the field is sampled on")
                                    .changed()
                                {
                                    resample = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Arrow scale");
                                ui.add(
                                    egui::Slider::new(&mut self.field_scale, 0.5..=20.0)
                                        .logarithmic(true),
                                )
                                .on_hover_text(
                                    "Arrows are drawn this many times their true length; \
                                     millimetre motion is invisible at 1×",
                                );
                            });
                            ui.checkbox(&mut self.field_color, "Colour by magnitude")
                                .on_hover_text(
                                    "Blue → red over the field's own range, so where the \
                                     deformation is largest reads at a glance",
                                );
                            if let Some(job) = &self.field_job {
                                ui.horizontal(|ui| {
                                    ui.spinner();
                                    ui.weak(job.progress.get());
                                });
                            } else {
                                ui.weak(reg.field.describe());
                            }
                            if tip_button(
                                ui,
                                "💾 Save as DICOM",
                                "Write the field as a Deformable Spatial Registration \
                                 object: the whole mapping in one grid, with identity \
                                 pre- and post-matrices, so another system has no \
                                 composition rule to get wrong",
                            ) {
                                save_field = true;
                            }
                        });

                    ui.horizontal(|ui| {
                        if tip_button(
                            ui,
                            "⇄ Propagate structures",
                            "Open the structures propagation module, aimed at this \
                             registration",
                        ) {
                            propagate_from = Some(reg.moving_slot);
                        }
                        if ui.button("Clear registration").clicked() {
                            clear = true;
                        }
                    });
                }
            });
        ui.separator();

        if clear_group {
            self.group_registration = None;
            self.reg_gen += 1;
        }
        if let Some(i) = show_phase {
            self.show_group_phase(i);
        }
        if hide_field {
            self.field_on = false;
        }
        if let Some((moving, gslot, group)) = run_group {
            self.start_group_run(
                moving,
                gslot,
                group,
                // The registration module registers the whole group; one
                // phase on its own is a propagation question, not this one.
                None,
                Vec::new(),
                crate::propagate::Finish::default(),
            );
        }
        if let Some(slot) = propagate_from {
            self.open_propagate_module(slot);
        }
        if let Some(refine) = run {
            self.start_registration(refine);
        }
        cancel_if(cancel, &self.reg_job);
        if apply_matrix {
            // The matrix is absolute patient millimetres, so it needs no
            // centre of its own; the fixed workspace is the one it maps from.
            let t = Transform3::from_matrix(self.reg_matrix.m, crate::geometry::Vec3::ZERO);
            let fixed = self.reg_fixed.slot;
            self.apply_external_transform(t, RegMethod::Given, fixed);
        }
        if clear {
            if self
                .registration
                .as_ref()
                .is_some_and(|r| r.group_phase.is_some())
            {
                self.clear_shown_phase();
            } else {
                self.clear_registration();
            }
        }
        if add_landmark {
            self.add_landmark_pair();
        }
        if let Some(i) = drop_landmark {
            self.reg_landmarks.remove(i);
        }
        if clear_landmarks {
            self.reg_landmarks.clear();
        }
        if resample {
            self.rebuild_field();
        }
        if save_field {
            self.save_vector_field();
        }
        if score_structs {
            self.score_registration_structures();
        }
    }

    /// Write the active registration's field as a DICOM Deformable Spatial
    /// Registration object.
    fn save_vector_field(&mut self) {
        let Some(reg) = &self.registration else {
            return;
        };
        let (fixed, moving) = (reg.fixed_slot, reg.moving_slot);
        if self.slots[fixed].study.is_none() || self.slots[moving].study.is_none() {
            return;
        }
        self.ask_save(
            "Save the deformation field as DICOM",
            "deformable_registration.dcm",
            None,
            None,
            |app, path| app.write_vector_field(&path),
        );
    }

    /// The writing half of [`Self::save_vector_field`], once there is a path.
    fn write_vector_field(&mut self, path: &std::path::Path) {
        let Some(reg) = &self.registration else {
            return;
        };
        let (fixed, moving) = (reg.fixed_slot, reg.moving_slot);
        let (Some(f), Some(m)) = (&self.slots[fixed].study, &self.slots[moving].study) else {
            return;
        };
        // The registration belongs in the fixed workspace's study when there
        // is one to belong to; a fresh UID is the honest fallback.
        let study_uid = f
            .series
            .first()
            .map(|s| s.study_uid.clone())
            .unwrap_or_default();
        let meta = dicom_export::DvfExport {
            source_for_uid: &f.volume.frame_of_reference_uid,
            target_for_uid: &m.volume.frame_of_reference_uid,
            study_uid: &study_uid,
            patient_name: &f.meta.patient_name,
            patient_id: &f.meta.patient_id,
            label: reg.result.method.family(),
            description: &format!(
                "{} - {}",
                reg.result.method.label(),
                reg.result.transform.warp.describe()
            ),
        };
        match dicom_export::write_deformable_registration(path, &reg.field, &meta) {
            Ok(()) => {
                self.error = Some(format!("✔ Deformation field written to {}", path.display()))
            }
            Err(e) => self.error = Some(format!("Writing the field failed: {e:#}")),
        }
    }

    /// The per-method parameter rows.
    fn parameter_rows(&mut self, ui: &mut egui::Ui) {
        let method = self.reg_method;
        // One form for the whole section, whichever method is picked: the
        // rows that apply change, the left edge does not.
        form::form(ui, "reg_params_form", |f| {
            if method.is_intensity_based() {
                f.row_tip(
                    "Resolutions",
                    "Pyramid levels, coarse to fine (elastix NumberOfResolutions)",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.reg_levels)
                                .speed(0.1)
                                .range(1..=5),
                        );
                    },
                );
                f.row_tip(
                    "Iterations/level",
                    "The stochastic engine wants hundreds of cheap iterations; the dense \
                     one converges in tens of expensive ones",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.reg_iterations)
                                .speed(10)
                                .range(10..=5000),
                        );
                    },
                );
                f.row_tip(
                    "Body threshold",
                    "Only fixed-image voxels above this drive the metric - a crude body \
                     mask that keeps air out of the cost",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.reg_threshold)
                                .speed(10.0)
                                .range(-2000.0..=2000.0)
                                .suffix(" HU"),
                        );
                    },
                );
            }
            match method {
                // Nothing to tune: a given matrix is not optimized.
                RegMethod::Given => {}
                RegMethod::ElastixRigid | RegMethod::ElastixBSpline => {
                    f.row_tip(
                        "Samples/iter",
                        "elastix NumberOfSpatialSamples, redrawn every iteration",
                        |ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.reg_samples)
                                    .speed(100)
                                    .range(500..=50000),
                            );
                        },
                    );
                }
                RegMethod::PlastimatchBSpline => {
                    f.row("Metric", |ui| {
                        for m in Metric::ALL {
                            ui.selectable_value(&mut self.reg_metric, m, m.label())
                                .on_hover_text(m.hint());
                        }
                    });
                    f.row_tip(
                        "Regularization",
                        "plastimatch young_modulus: the weight of the bending-energy \
                         penalty on the control lattice. Higher is smoother and less \
                         likely to fold; 0 turns it off.",
                        |ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.reg_regularization)
                                    .speed(0.005)
                                    .range(0.0..=2.0),
                            );
                        },
                    );
                }
                RegMethod::PlastimatchLandmark => {}
            }
            if matches!(
                method,
                RegMethod::ElastixBSpline | RegMethod::PlastimatchBSpline
            ) {
                f.row_tip(
                    "B-spline grid",
                    "Control-point spacing (elastix FinalGridSpacingInPhysicalUnits, \
                     plastimatch grid_spacing). Finer resolves more detail and costs more.",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.reg_grid_mm)
                                .speed(1.0)
                                .range(4.0..=128.0)
                                .suffix(" mm"),
                        );
                    },
                );
            }
            if method == RegMethod::PlastimatchLandmark {
                f.row("Kernel", |ui| {
                    egui::ComboBox::from_id_salt("reg_kernel")
                        .selected_text(self.reg_landmark.kernel.label())
                        .width(170.0)
                        .show_ui(ui, |ui| {
                            for k in LandmarkKernel::ALL {
                                ui.selectable_value(&mut self.reg_landmark.kernel, k, k.label())
                                    .on_hover_text(k.hint());
                            }
                        });
                });
                f.wide(|ui| {
                    ui.weak(self.reg_landmark.kernel.hint());
                });
                if self.reg_landmark.kernel.uses_radius() {
                    f.row("Reach", |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.reg_landmark.radius_mm)
                                .speed(1.0)
                                .range(2.0..=400.0)
                                .suffix(" mm"),
                        );
                    });
                }
                f.row_tip(
                    "Stiffness",
                    "0 passes exactly through every landmark. Larger values smooth the \
                     field instead - which is what inconsistent pairs need.",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut self.reg_landmark.stiffness)
                                .speed(0.01)
                                .range(0.0..=100.0),
                        );
                    },
                );
            }
        });
    }
}

/// The analysis block: the overlap that says whether the result is any
/// good, six degrees of freedom, displacements, Jacobian, and the
/// displacement (and, on request, the Dice) of each visible structure.
///
/// Returns whether the caller should score the structures, because the
/// panel holds `self` immutably while this runs.
#[must_use]
fn analysis_rows(
    ui: &mut egui::Ui,
    res: &RegistrationResult,
    structures: Option<&crate::rtstruct::StructureSet>,
    transform: &Transform3,
    struct_dice: Option<&[StructDice]>,
) -> bool {
    let a = &res.analysis;
    let mut score = false;

    // The headline: how much of the two images actually covers the same
    // anatomy now, against how much did before. Everything below explains
    // this number.
    if let Some(ov) = &a.overlap {
        ui.label("Image overlap:");
        ui.horizontal(|ui| {
            ui.monospace(
                egui::RichText::new(format!("Dice {:.3}", ov.after))
                    .color(theme::dice_color(ui.visuals(), ov.after))
                    .strong(),
            );
            let gain = ov.gain();
            let arrow = if gain >= 0.0 { "▲" } else { "▼" };
            ui.weak(format!("was {:.3}  {arrow} {:+.3}", ov.before, gain));
        })
        .response
        .on_hover_text(format!(
            "Dice of the tissue of the two images: every voxel above {:.0} HU \
             counts as tissue, and the score is the overlap of the fixed \
             image's tissue with the moving image's, before the transform and \
             after it. {} probes.\n\nIt is an image score, not an anatomical \
             one - it says the two workspaces now cover the same space. For \
             anatomy, score the structures below.",
            ov.threshold, ov.samples
        ));
        ui.add_space(2.0);
    }

    ui.label("Best-fitting rigid body:");
    ui.monospace(a.dof.line());
    if a.dof.residual_mm > 1e-6 {
        ui.weak(format!(
            "residual {:.2} mm - what the six numbers do not explain",
            a.dof.residual_mm
        ));
    } else {
        ui.weak("residual 0.00 mm - the result is a rigid body");
    }
    ui.add_space(2.0);
    ui.label("Displacement:");
    ui.monospace(a.displacement.line());
    ui.weak(format!(
        "mean vector ({:.2}, {:.2}, {:.2}) mm · RMS {:.2} mm",
        a.mean_vector.x, a.mean_vector.y, a.mean_vector.z, a.displacement.rms
    ));
    ui.add_space(2.0);
    ui.label("Jacobian:");
    ui.monospace(a.jacobian.line());
    ui.weak(format!(
        "{} probes on a {:.0} mm lattice",
        a.samples, a.step_mm
    ));

    // Per-structure displacement and overlap: the numbers a physicist asks
    // for next, because a good image score can still hide a mismatched organ.
    egui::CollapsingHeader::new("Per structure")
        .id_salt("reg_per_struct")
        .default_open(false)
        .show(ui, |ui| {
            if let Some(ss) = structures {
                let mut any = false;
                // A column each: the eye runs down "max" looking for the
                // structure the transform pulled hardest on, which a list of
                // sentences does not let it do.
                egui::Grid::new("reg_struct_disp")
                    .striped(true)
                    .num_columns(4)
                    .spacing([10.0, 2.0])
                    .show(ui, |ui| {
                        ui.label("");
                        ui.label(egui::RichText::new("Structure").strong().small());
                        ui.label(egui::RichText::new("Mean mm").strong().small());
                        ui.label(egui::RichText::new("Max mm").strong().small());
                        ui.end_row();
                        for roi in &ss.rois {
                            let pts: Vec<Vec3> = roi
                                .contours
                                .iter()
                                .flat_map(|c| c.points.iter().copied())
                                .step_by(7)
                                .collect();
                            if pts.len() < 4 {
                                continue;
                            }
                            any = true;
                            let (stats, mean) = analysis::stats_over_points(transform, &pts);
                            ui.colored_label(theme::rgb(roi.color), "◼");
                            ui.label(&roi.name);
                            ui.monospace(format!("{:.2}", stats.mean));
                            ui.monospace(format!("{:.2}", stats.max))
                                .on_hover_text(format!(
                                    "{}\nmean ({:.2}, {:.2}, {:.2}) mm over {} contour points",
                                    stats.line(),
                                    mean.x,
                                    mean.y,
                                    mean.z,
                                    pts.len()
                                ));
                            ui.end_row();
                        }
                    });
                if !any {
                    ui.weak("No contoured structure on this workspace.");
                }
            }

            // Structure Dice pairs each structure of the fixed workspace with
            // the one of the same name on the moving workspace, carries the
            // moving one through the transform, and scores the overlap. It
            // costs a rasterization per structure, so it is asked for.
            ui.separator();
            match struct_dice {
                None => {
                    if tip_button(
                        ui,
                        "Score structures (Dice)",
                        "Pair every structure of the fixed workspace with the one of \
                         the same name on the moving workspace, warp the moving one \
                         through this registration, and score the overlap",
                    ) {
                        score = true;
                    }
                }
                Some([]) => {
                    ui.weak(
                        "No structure of the fixed workspace shares its name with one \
                         on the moving workspace.",
                    );
                    if tip_button(ui, "Score again", "Rerun the pairing") {
                        score = true;
                    }
                }
                Some(rows) => {
                    egui::Grid::new("reg_struct_dice")
                        .striped(true)
                        .num_columns(5)
                        .spacing([10.0, 2.0])
                        .show(ui, |ui| {
                            ui.label("");
                            ui.label(egui::RichText::new("Structure").strong().small());
                            ui.label(egui::RichText::new("Dice").strong().small());
                            ui.label(egui::RichText::new("Was").strong().small());
                            ui.label(egui::RichText::new("Gain").strong().small());
                            ui.end_row();
                            for d in rows {
                                ui.colored_label(theme::rgb(d.color), "◼");
                                ui.label(&d.name);
                                ui.monospace(
                                    egui::RichText::new(format!("{:.3}", d.after))
                                        .color(theme::dice_color(ui.visuals(), d.after)),
                                );
                                ui.weak(format!("{:.3}", d.before));
                                ui.monospace(format!("{:+.3}", d.after - d.before))
                                    .on_hover_text(format!(
                                        "fixed {:.2} cm³, warped moving {:.2} cm³",
                                        d.fixed_cm3, d.moving_cm3
                                    ));
                                ui.end_row();
                            }
                        });
                    if tip_button(
                        ui,
                        "Score again",
                        "Recompute after editing a structure or refining the fit",
                    ) {
                        score = true;
                    }
                }
            }
        });
    score
}
