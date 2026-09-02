//! The *Registration* side-panel section: choosing a method, restricting it
//! to a structure, placing landmarks, starting the run, and everything the
//! result has to say afterwards.
//!
//! One section covers four algorithms because the choice between them is
//! the user's real question ("stochastic or dense? intensities or points?"),
//! and the rest of the conversation — direction, region, parameters,
//! analytics, fusion, vector field — is the same whichever they pick.

use anyhow::{anyhow, Result};

use super::*;
use crate::registration::{analysis, LandmarkKernel, RegParams, Warp};

/// What restricts the next registration: everything, or one structure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RegRoi {
    /// The whole fixed image (a global registration).
    Whole,
    /// One ROI of the fixed dataset's active structure set.
    Structure(usize),
    /// One painted / segmented mask of the fixed dataset.
    Segmentation(usize),
}

/// What a registration run hands back: the result, the vector field sampled
/// from it, and the region it was restricted to (kept so the field can be
/// re-sampled later without rebuilding the mask).
pub(super) struct RegOutcome {
    pub result: RegistrationResult,
    pub field: VectorField,
    pub region: Option<Arc<RegionMask>>,
}

impl ViewerApp {
    // -- region ------------------------------------------------------------

    /// The regions a dataset offers, as `(choice, label)`.
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
                .ok_or_else(|| anyhow!("dataset {} is not loaded", SLOT_NAMES[slot]))?;
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
                .filter(|r| r.fixed_slot == self.reg_fixed_slot.min(1))
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
            start,
        }
    }

    /// Start a registration (or a refinement of the active one).
    pub(super) fn start_registration(&mut self, refine: bool) {
        if self.reg_job.is_some() {
            return;
        }
        let fixed_slot = self.reg_fixed_slot.min(1);
        let moving_slot = 1 - fixed_slot;
        let (Some(f), Some(m)) = (
            &self.slots[fixed_slot].study,
            &self.slots[moving_slot].study,
        ) else {
            self.error = Some("Registration needs two loaded studies (comparison mode)".into());
            return;
        };
        let fixed = f.volume.clone();
        let moving = m.volume.clone();
        let region = match self.build_region(fixed_slot) {
            Ok(r) => r,
            Err(e) => {
                self.error = Some(format!("Local registration: {e:#}"));
                return;
            }
        };
        let params = self.current_reg_params(region.clone(), refine);
        if params.method == RegMethod::PlastimatchLandmark && params.landmarks.is_empty() {
            self.error = Some(
                "The landmark warp needs paired points: put the crosshair on the same \
                 anatomy in both datasets and press ➕ Add pair (turn off crosshair \
                 linking first, or both crosshairs move together)."
                    .into(),
            );
            return;
        }
        let step = self.field_step_mm;
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.reg_job = Some(Job::spawn(progress, move |p| {
            let out = registration::register(&fixed, &moving, &params, p).map(|result| {
                p.set("Sampling the vector field");
                let field = VectorField::sample(&fixed, &result.transform, region.as_deref(), step);
                RegOutcome {
                    result,
                    field,
                    region,
                }
            });
            (fixed_slot, out)
        }));
    }

    /// Install a rigid transform (e.g. from a DICOM REG object) as the
    /// active registration, exactly as if it had been computed.
    pub(super) fn apply_external_rigid(
        &mut self,
        rigid: registration::RigidTransform,
        fixed_slot: usize,
    ) {
        self.apply_external_transform(
            Transform3::rigid_only(rigid),
            RegMethod::ElastixRigid,
            fixed_slot,
        );
    }

    /// Install any transform read from a file as the active registration.
    ///
    /// A Deformable Spatial Registration's grid arrives here exactly as a
    /// REG matrix does, so everything downstream — fusion, the crosshair
    /// link, the analytics, the vector field, propagation — works on it
    /// without knowing where it came from.
    pub(super) fn apply_external_transform(
        &mut self,
        transform: Transform3,
        method: RegMethod,
        fixed_slot: usize,
    ) {
        let transform = Arc::new(transform);
        let Some(study) = &self.slots[fixed_slot].study else {
            return;
        };
        // A transform installed from elsewhere (a REG object in the tree, a
        // propagation) needs the section that shows and clears it.
        self.module_registration = true;
        let vol = study.volume.clone();
        let analysis = analysis::analyse(&vol, &transform, None);
        let field = VectorField::sample(&vol, &transform, None, self.field_step_mm);
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
            field: Arc::new(field),
            region: None,
        });
        self.fusion_on = self.slots[0].has_volume() && self.slots[1].has_volume();
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
        self.fusion_on = false;
        self.reg_gen += 1;
    }

    /// Re-sample the vector field after the lattice step changed.
    pub(super) fn rebuild_field(&mut self) {
        if self.field_job.is_some() {
            return;
        }
        let Some(reg) = &self.registration else {
            return;
        };
        let Some(study) = &self.slots[reg.fixed_slot].study else {
            return;
        };
        let vol = study.volume.clone();
        let t = reg.result.transform.clone();
        let region = reg.region.clone();
        let step = self.field_step_mm;
        let progress = Arc::new(Progress::default());
        self.field_job = Some(Job::spawn(progress, move |_| {
            VectorField::sample(&vol, &t, region.as_deref(), step)
        }));
    }

    /// Add a landmark pair from the two crosshairs.
    fn add_landmark_pair(&mut self) {
        let fixed_slot = self.reg_fixed_slot.min(1);
        let moving_slot = 1 - fixed_slot;
        let point = |s: &StudySlot| -> Option<Vec3> {
            let st = s.study.as_ref()?;
            let c = s.cursor;
            Some(st.volume.voxel_to_patient(c[0], c[1], c[2]))
        };
        let (Some(f), Some(m)) = (
            point(&self.slots[fixed_slot]),
            point(&self.slots[moving_slot]),
        ) else {
            self.error = Some("Load both datasets before placing landmarks".into());
            return;
        };
        let n = self.reg_landmarks.len() + 1;
        self.reg_landmarks
            .push(LandmarkPair::new(format!("L{n}"), f, m));
    }

    // -- the panel section -------------------------------------------------

    pub(super) fn registration_section(&mut self, ui: &mut egui::Ui) {
        let both = self.slots[0].has_volume() && self.slots[1].has_volume();
        // The section is worth showing while two datasets are loaded, while a
        // result is on display, and while a run is in flight — the last one
        // because that is where its progress and its Cancel button live.
        if !both && self.registration.is_none() && self.reg_job.is_none() {
            // The module is switched on, so the section says what it is
            // waiting for rather than leaving an empty panel.
            egui::CollapsingHeader::new(egui::RichText::new("Image registration").strong())
                .default_open(true)
                .show(ui, |ui| {
                    ui.weak(
                        "Load a second dataset (File > Add DICOM folder to B) - \
                         registration aligns one onto the other",
                    );
                });
            ui.separator();
            return;
        }
        let mut run: Option<bool> = None;
        let mut cancel = false;
        let mut clear = false;
        let mut resample = false;
        let mut add_landmark = false;
        let mut drop_landmark: Option<usize> = None;
        let mut clear_landmarks = false;
        let mut save_field = false;

        egui::CollapsingHeader::new(egui::RichText::new("Image registration").strong())
            .default_open(true)
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

                let fixed_slot = self.reg_fixed_slot.min(1);
                ui.horizontal(|ui| {
                    ui.label("Direction");
                    ui.selectable_value(&mut self.reg_fixed_slot, 0, "B ▶ A")
                        .on_hover_text("B is deformed/moved onto A; fusion shown on A");
                    ui.selectable_value(&mut self.reg_fixed_slot, 1, "A ▶ B")
                        .on_hover_text("A is deformed/moved onto B; fusion shown on B");
                });

                // ---- method ----
                ui.horizontal(|ui| {
                    ui.label("Method");
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
                ui.weak(self.reg_method.hint());

                // ---- region ----
                let choices = self.region_choices_for(fixed_slot);
                if self.region_label(fixed_slot).is_none() {
                    self.reg_roi = RegRoi::Whole;
                }
                ui.horizontal(|ui| {
                    ui.label("Region");
                    let current = self
                        .region_label(fixed_slot)
                        .unwrap_or_else(|| "Whole image".into());
                    egui::ComboBox::from_id_salt("reg_roi")
                        .selected_text(current)
                        .width(190.0)
                        .show_ui(ui, |ui| {
                            for (choice, label) in &choices {
                                ui.selectable_value(&mut self.reg_roi, *choice, label);
                            }
                        })
                        .response
                        .on_hover_text(
                            "Restrict the registration to one structure of the fixed \
                             dataset. Samples come from inside it only and the B-spline \
                             lattice covers it alone, so a small structure can be aligned \
                             at a fine grid - and, when it refines an existing result, \
                             the rest of the patient keeps that result untouched.",
                        );
                });
                if self.reg_roi != RegRoi::Whole {
                    ui.horizontal(|ui| {
                        ui.label("Margin");
                        ui.add(
                            egui::DragValue::new(&mut self.reg_margin_mm)
                                .speed(1.0)
                                .range(0.0..=60.0)
                                .suffix(" mm"),
                        )
                        .on_hover_text(
                            "The structure is grown by this much before sampling. Without \
                             a margin nothing outside the structure constrains its \
                             boundary, and the boundary is what you are aligning.",
                        );
                    });
                }

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
                            if ui
                                .button("➕ Add pair")
                                .on_hover_text(
                                    "Take the crosshair of each dataset as one pair. Put \
                                     both crosshairs on the same anatomy first - and turn \
                                     off View ▶ Sync crosshairs, or they move together.",
                                )
                                .clicked()
                            {
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
                let can_refine = self
                    .registration
                    .as_ref()
                    .is_some_and(|r| r.fixed_slot == fixed_slot)
                    && self.reg_method.is_deformable();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(both, egui::Button::new("▶ Register"))
                        .on_hover_text("Recover the transform from scratch")
                        .clicked()
                    {
                        run = Some(false);
                    }
                    if ui
                        .add_enabled(both && can_refine, egui::Button::new("▶ Refine"))
                        .on_hover_text(
                            "Recover a correction on top of the active registration and \
                             add the two together - how a local registration is meant to \
                             be used after a global one",
                        )
                        .clicked()
                    {
                        run = Some(true);
                    }
                });
                if !both {
                    ui.weak("Load two datasets (comparison mode) first");
                }

                // ---- result ----
                if let Some(reg) = &self.registration {
                    ui.separator();
                    let res = &reg.result;
                    let moving = SLOT_NAMES[1 - reg.fixed_slot];
                    let fixed = SLOT_NAMES[reg.fixed_slot];
                    ui.label(
                        egui::RichText::new(format!(
                            "✔ {}  ({moving} ▶ {fixed})",
                            res.method.label()
                        ))
                        .strong(),
                    );
                    if let Some(r) = &res.region {
                        ui.weak(format!("restricted to {r}"));
                    }
                    ui.weak(res.metric_line());
                    ui.weak(res.transform.warp.describe());

                    egui::CollapsingHeader::new("Analysis")
                        .id_salt("reg_analysis")
                        .default_open(true)
                        .show(ui, |ui| {
                            analysis_rows(
                                ui,
                                res,
                                self.slots[reg.fixed_slot].active_structures(),
                                &res.transform,
                            )
                        });

                    ui.checkbox(&mut self.fusion_on, format!("Fusion overlay on {fixed}"));
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
                            if ui
                                .button("💾 Save as DICOM")
                                .on_hover_text(
                                    "Write the field as a Deformable Spatial Registration \
                                     object: the whole mapping in one grid, with identity \
                                     pre- and post-matrices, so another system has no \
                                     composition rule to get wrong",
                                )
                                .clicked()
                            {
                                save_field = true;
                            }
                        });

                    if ui.button("Clear registration").clicked() {
                        clear = true;
                    }
                }
            });
        ui.separator();

        if let Some(refine) = run {
            self.start_registration(refine);
        }
        if cancel {
            if let Some(job) = &self.reg_job {
                job.progress.cancel();
            }
        }
        if clear {
            self.clear_registration();
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
    }

    /// Write the active registration's field as a DICOM Deformable Spatial
    /// Registration object.
    fn save_vector_field(&mut self) {
        let Some(reg) = &self.registration else {
            return;
        };
        let fixed = reg.fixed_slot;
        let (Some(f), Some(m)) = (&self.slots[fixed].study, &self.slots[1 - fixed].study) else {
            return;
        };
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save the deformation field as DICOM")
            .set_file_name("deformable_registration.dcm")
            .save_file()
        else {
            return;
        };
        // The registration belongs in the fixed dataset's study when there
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
        match dicom_export::write_deformable_registration(&path, &reg.field, &meta) {
            Ok(()) => {
                self.error = Some(format!("✔ Deformation field written to {}", path.display()))
            }
            Err(e) => self.error = Some(format!("Writing the field failed: {e:#}")),
        }
    }

    /// The per-method parameter rows.
    fn parameter_rows(&mut self, ui: &mut egui::Ui) {
        let method = self.reg_method;
        if method.is_intensity_based() {
            ui.horizontal(|ui| {
                ui.label("Resolutions");
                ui.add(
                    egui::DragValue::new(&mut self.reg_levels)
                        .speed(0.1)
                        .range(1..=5),
                )
                .on_hover_text("Pyramid levels, coarse to fine (elastix NumberOfResolutions)");
            });
            ui.horizontal(|ui| {
                ui.label("Iterations/level");
                ui.add(
                    egui::DragValue::new(&mut self.reg_iterations)
                        .speed(10)
                        .range(10..=5000),
                )
                .on_hover_text(
                    "The stochastic engine wants hundreds of cheap iterations; the dense \
                     one converges in tens of expensive ones",
                );
            });
            ui.horizontal(|ui| {
                ui.label("Body threshold");
                ui.add(
                    egui::DragValue::new(&mut self.reg_threshold)
                        .speed(10.0)
                        .range(-2000.0..=2000.0)
                        .suffix(" HU"),
                )
                .on_hover_text(
                    "Only fixed-image voxels above this drive the metric - a crude body \
                     mask that keeps air out of the cost",
                );
            });
        }
        match method {
            RegMethod::ElastixRigid | RegMethod::ElastixBSpline => {
                ui.horizontal(|ui| {
                    ui.label("Samples/iter");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_samples)
                            .speed(100)
                            .range(500..=50000),
                    )
                    .on_hover_text("elastix NumberOfSpatialSamples, redrawn every iteration");
                });
            }
            RegMethod::PlastimatchBSpline => {
                ui.horizontal(|ui| {
                    ui.label("Metric");
                    for m in Metric::ALL {
                        ui.selectable_value(&mut self.reg_metric, m, m.label())
                            .on_hover_text(m.hint());
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Regularization");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_regularization)
                            .speed(0.005)
                            .range(0.0..=2.0),
                    )
                    .on_hover_text(
                        "plastimatch young_modulus: the weight of the bending-energy \
                         penalty on the control lattice. Higher is smoother and less \
                         likely to fold; 0 turns it off.",
                    );
                });
            }
            RegMethod::PlastimatchLandmark => {}
        }
        if matches!(
            method,
            RegMethod::ElastixBSpline | RegMethod::PlastimatchBSpline
        ) {
            ui.horizontal(|ui| {
                ui.label("B-spline grid");
                ui.add(
                    egui::DragValue::new(&mut self.reg_grid_mm)
                        .speed(1.0)
                        .range(4.0..=128.0)
                        .suffix(" mm"),
                )
                .on_hover_text(
                    "Control-point spacing (elastix FinalGridSpacingInPhysicalUnits, \
                     plastimatch grid_spacing). Finer resolves more detail and costs more.",
                );
            });
        }
        if method == RegMethod::PlastimatchLandmark {
            ui.horizontal(|ui| {
                ui.label("Kernel");
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
            ui.weak(self.reg_landmark.kernel.hint());
            if self.reg_landmark.kernel.uses_radius() {
                ui.horizontal(|ui| {
                    ui.label("Reach");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_landmark.radius_mm)
                            .speed(1.0)
                            .range(2.0..=400.0)
                            .suffix(" mm"),
                    );
                });
            }
            ui.horizontal(|ui| {
                ui.label("Stiffness");
                ui.add(
                    egui::DragValue::new(&mut self.reg_landmark.stiffness)
                        .speed(0.01)
                        .range(0.0..=100.0),
                )
                .on_hover_text(
                    "0 passes exactly through every landmark. Larger values smooth the \
                     field instead - which is what inconsistent pairs need.",
                );
            });
        }
    }
}

/// The analysis block: six degrees of freedom, displacements, Jacobian, and
/// the displacement of each visible structure.
fn analysis_rows(
    ui: &mut egui::Ui,
    res: &RegistrationResult,
    structures: Option<&crate::rtstruct::StructureSet>,
    transform: &Transform3,
) {
    let a = &res.analysis;
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

    // Per-structure displacement: the number a physicist asks for next.
    if let Some(ss) = structures {
        egui::CollapsingHeader::new("Per structure")
            .id_salt("reg_per_struct")
            .default_open(false)
            .show(ui, |ui| {
                let mut any = false;
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
                    ui.horizontal(|ui| {
                        ui.colored_label(
                            Color32::from_rgb(roi.color[0], roi.color[1], roi.color[2]),
                            "◼",
                        );
                        ui.label(&roi.name);
                        ui.weak(format!("{:.2} mm", stats.mean));
                    })
                    .response
                    .on_hover_text(format!(
                        "{}\nmean ({:.2}, {:.2}, {:.2}) mm\nmax {:.2} mm over {} contour points",
                        stats.line(),
                        mean.x,
                        mean.y,
                        mean.z,
                        stats.max,
                        pts.len()
                    ));
                }
                if !any {
                    ui.weak("No contoured structure on this dataset.");
                }
            });
    }
}
