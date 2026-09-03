//! The body-contour tool window - the fourth of the segmentation tools, and
//! the only one that can answer without a network.
//!
//! It shares its bones with the other three ([`super::seg_engines`]): a
//! description, the tool's own inputs, an `Options` section, a licence line
//! and a button row that becomes a progress row while a run is in flight.
//! What it does differently is that its *method* choice changes what the
//! rest of the window means - the classical method has no device, no model
//! and nothing to download, so those rows appear only when they matter.
//!
//! The window re-seeds its thresholds from the displayed series' modality
//! whenever that changes, because −300 HU is meaningless on MR and a
//! fraction of the 99th percentile is meaningless on CT. A threshold the
//! user has edited by hand is left alone.

use crate::bodymask::{self, BodyModel, BodyParams, BodyResult, Foreground, Method};
use crate::models::Engine as ModelsEngine;

use super::*;

/// The tool window's state; it stays open across runs.
pub(super) struct BodyDialog {
    pub slot: usize,
    pub params: BodyParams,
    /// The modality the parameters were seeded from, so a series switch can
    /// re-seed them - and only then.
    pub seeded_for: String,
    /// One-line summary of the last finished run.
    pub status: Option<String>,
}

/// Everything a run needs, snapshotted from the window when it starts.
struct BodyRequest {
    params: BodyParams,
    models_dir: PathBuf,
}

impl ViewerApp {
    /// The modality of the series a slot is showing, upper-cased.
    fn slot_modality(&self, slot: usize) -> String {
        self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.series.get(st.active_series))
            .map(|s| s.modality.to_uppercase())
            .unwrap_or_default()
    }

    /// Tools ▶ body contour: open the tool window for `slot`.
    pub(super) fn open_body_dialog(&mut self, slot: usize) {
        if !self.slots[slot].has_volume() {
            return;
        }
        let modality = self.slot_modality(slot);
        match &mut self.body_dialog {
            // Re-target an open window unless it is busy with the other slot.
            Some(d) if self.body_job.is_none() => d.slot = slot,
            Some(_) => {}
            None => {
                self.body_dialog = Some(BodyDialog {
                    slot,
                    params: BodyParams::for_modality(&modality),
                    seeded_for: modality,
                    status: None,
                });
            }
        }
    }

    /// Snapshot the parameters and the volume and run on a worker thread.
    pub(super) fn start_body(&mut self) {
        if self.body_job.is_some() {
            return;
        }
        let Some(d) = &self.body_dialog else {
            return;
        };
        let Some(study) = self.slots[d.slot].study.as_ref() else {
            return;
        };
        let volume = study.volume.clone();
        let slot = d.slot;
        let mut params = d.params.clone();
        params.name = params.name.trim().to_string();
        if params.name.is_empty() {
            params.name = "BODY".into();
        }
        let req = BodyRequest {
            params,
            models_dir: self.engine_models_dir(ModelsEngine::TotalSegmentator),
        };
        self.persist_settings();
        let progress = Arc::new(Progress::default());
        progress.set("Preparing");
        self.body_slot = slot;
        self.body_job = Some(Job::spawn(progress, move |p| {
            (
                slot,
                bodymask::contour_body(&volume, &req.params, &req.models_dir, p),
            )
        }));
    }

    /// A run finished: verify the slot still shows the same volume, land the
    /// mask, and - when asked - file it as an RTSTRUCT `EXTERNAL` too.
    pub(super) fn on_body_done(&mut self, slot: usize, result: BodyResult) {
        if !self.slot_still_shows(slot, result.volume_dims, &result.frame_of_reference_uid) {
            self.error = Some(stale_result(&BODY_CONTOUR));
            return;
        }
        if result.voxels == 0 {
            self.error = Some(
                "The body contour came out empty - lower the threshold, or reduce the \
                 opening radius."
                    .into(),
            );
            return;
        }
        let idx = self.add_colored_segmentation(
            slot,
            result.name.clone(),
            // Bone-white: the outline is a reference, not one more coloured
            // structure competing with the anatomy inside it.
            [230, 230, 220],
            result.volume_dims,
            &result.mask,
        );
        if result.make_external {
            self.seg_to_rtstruct(slot, idx, "EXTERNAL");
        }
        // The cm³ conversions read `self`, so they happen before the
        // dialog is borrowed mutably.
        let removed_cm3 = self.voxels_to_cm3(slot, result.removed_voxels);
        let recovered_cm3 = self.voxels_to_cm3(slot, result.recovered_voxels);
        // `1250 + 980 cm³`: the size of each body when there is more than one.
        let pieces = match result.pieces.len() {
            0 | 1 => String::new(),
            n => format!(
                ", {n} separate bodies ({})",
                result
                    .pieces
                    .iter()
                    .map(|p| format!("{:.0}", p.cm3))
                    .collect::<Vec<_>>()
                    .join(" + ")
                    + " cm³"
            ),
        };
        let device = if result.device.is_empty() {
            String::new()
        } else {
            format!(" on {}", result.device)
        };
        if let Some(d) = &mut self.body_dialog {
            d.status = Some(format!(
                "✔ {}: {:.0} cm³{pieces} in {:.1} s{device} - {:.0} cm³ of couch, chair, \
                 immobilisation and stray objects left out{}",
                result.name,
                result.cm3,
                result.elapsed_secs,
                removed_cm3,
                if result.recovered_voxels > 0 {
                    format!(", {recovered_cm3:.0} cm³ of thin anatomy kept")
                } else {
                    String::new()
                }
            ));
        }
    }

    fn voxels_to_cm3(&self, slot: usize, voxels: u64) -> f64 {
        let sp = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.volume.spacing)
            .unwrap_or([1.0; 3]);
        voxels as f64 * sp[0] * sp[1] * sp[2] / 1000.0
    }

    /// The tool window; while a run is in flight its buttons become the
    /// progress row.
    pub(super) fn body_window(&mut self, ctx: &egui::Context) {
        let Some(slot) = self.body_dialog.as_ref().map(|d| d.slot) else {
            return;
        };
        if self.slots[slot].study.is_none() {
            self.body_dialog = None;
            return;
        }
        // Everything that reads the whole of `self` is settled before the
        // dialog is borrowed mutably for the frame.
        let has = [self.slots[0].has_volume(), self.slots[1].has_volume()];
        let mut switch: Option<usize> = None;
        let modality = self.slot_modality(slot);
        let idle = self.body_job.is_none();
        let models_dir = models::engine_dir(
            &models::root_from_setting(&self.models_dir),
            ModelsEngine::TotalSegmentator,
        );
        let Some(d) = &mut self.body_dialog else {
            return;
        };
        // Re-seed on a series switch: a CT threshold cannot be carried over
        // to an MR series, and the reverse is just as wrong.
        if modality != d.seeded_for && idle {
            let name = d.params.name.clone();
            d.params = BodyParams::for_modality(&modality);
            d.params.name = name;
            d.seeded_for = modality;
        }
        let running = self.body_job.as_ref().filter(|_| self.body_slot == slot);
        let mut open = true;
        let (mut run, mut close, mut browse, mut cancel) = (false, false, false, false);
        detach::tool_window(
            ctx,
            "body",
            BODY_CONTOUR.title(d.slot),
            &mut open,
            detach::WinOpts::width(430.0),
            |ui| {
                switch = dataset_row(ui, slot, has, idle);
                ui.label(
                    "Finds the patient's outer surface and leaves the couch, the chair and \
                     the immobilisation outside it - the EXTERNAL structure everything \
                     downstream starts from.",
                );
                ui.separator();
                ui.label("Method:");
                for m in Method::ALL {
                    let hint = match m {
                        Method::Classical => {
                            "Threshold and morphology only. Nothing to download, a few \
                             seconds, and the same answer every time. Equipment is found \
                             by being thin and repeating slice after slice."
                        }
                        Method::ModelAssisted => {
                            "A body-outline network decides what is patient, the threshold \
                             still places the skin. Slower and needs a one-off download, \
                             but it is the only thing that removes a mask or a couch \
                             touching the skin with no gap."
                        }
                    };
                    ui.radio_value(&mut d.params.method, m, m.label())
                        .on_hover_text(hint);
                }
                ui.separator();

                // ---- what counts as tissue ----------------------------
                foreground_row(ui, &mut d.params.foreground);

                if d.params.method == Method::ModelAssisted {
                    ui.horizontal(|ui| {
                        ui.label("Model:");
                        egui::ComboBox::from_id_salt("body_model")
                            .selected_text(d.params.model.label())
                            .show_ui(ui, |ui| {
                                for m in BodyModel::ALL {
                                    ui.selectable_value(&mut d.params.model, m, m.label());
                                }
                            });
                        let need = bodymask::download_needed(d.params.model, &models_dir);
                        ui.weak(if need == 0 {
                            "cached ✔".to_string()
                        } else {
                            format!("{} MB to download once", need / 1_000_000)
                        });
                    });
                    if matches!(d.params.model, BodyModel::Mr) != d.params.foreground.is_mr() {
                        ui.label(
                            egui::RichText::new(
                                "The chosen model was trained on the other modality.",
                            )
                            .small()
                            .color(warn_color(ui.visuals())),
                        );
                    }
                }

                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.add(egui::TextEdit::singleline(&mut d.params.name).desired_width(140.0));
                    ui.checkbox(&mut d.params.make_external, "as EXTERNAL structure")
                        .on_hover_text(
                            "Also file the result as an RTSTRUCT ROI of type EXTERNAL - \
                             what a planning system looks for to find the patient surface. \
                             It rides the DICOM export like any other contour.",
                        );
                });

                ui.separator();
                ui.collapsing("Options", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Smallest body detail:");
                        ui.add(
                            egui::Slider::new(&mut d.params.open_mm, 0.0..=20.0)
                                .suffix(" mm")
                                .fixed_decimals(1),
                        )
                        .on_hover_text(
                            "The opening that decides what is solid enough to be a body. \
                             Anything thicker than twice this keeps its exact surface; \
                             anything thinner is left to the thin-anatomy step below.",
                        );
                    });
                    ui.checkbox(
                        &mut d.params.remove_devices,
                        "Remove equipment that repeats slice after slice",
                    )
                    .on_hover_text(
                        "A couch, a backrest, a seat pan and an arm rest are surfaces swept \
                         along one axis, so their footprint repeats. An ear or a finger \
                         never does.",
                    );
                    ui.add_enabled_ui(d.params.remove_devices, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("    shells up to:");
                            ui.add(
                                egui::DragValue::new(&mut d.params.device_thin_mm)
                                    .range(0.5..=10.0)
                                    .speed(0.1)
                                    .suffix(" mm"),
                            )
                            .on_hover_text(
                                "Half the thickness a shell may have to count as equipment. \
                                 A couch skin is one or two millimetres of carbon, a mask \
                                 two or three; the thinnest tissue you would miss is a good \
                                 deal thicker. Raising this is how you start deleting \
                                 patients.",
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("    repeating over:");
                            ui.add(
                                egui::DragValue::new(&mut d.params.persist_window_mm)
                                    .range(20.0..=600.0)
                                    .suffix(" mm"),
                            );
                            ui.label("in at least");
                            ui.add(
                                egui::DragValue::new(&mut d.params.persist_frac)
                                    .range(0.5..=1.0)
                                    .speed(0.01)
                                    .fixed_decimals(2),
                            );
                            ui.label("of its slices");
                        });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Smallest body part:");
                        ui.add(
                            egui::DragValue::new(&mut d.params.min_volume_cm3)
                                .range(1.0..=5000.0)
                                .suffix(" cm³"),
                        )
                        .on_hover_text(
                            "Kept as a volume, not as 'the largest object', so a leg scan \
                             comes out as two bodies rather than one.",
                        );
                    });
                    ui.checkbox(
                        &mut d.params.recover_thin,
                        "Keep thin anatomy (skin, ears, fingers)",
                    )
                    .on_hover_text(
                        "What the opening shaved off the body's own surface is always \
                             given back. What stands clear of it - an ear, a nose, a \
                             fingertip - is given back if it is small enough; a pad, a \
                             blanket or a bolus is not.",
                    );
                    ui.add_enabled_ui(d.params.recover_thin, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("    standing clear, up to:");
                            ui.add(
                                egui::DragValue::new(&mut d.params.thin_max_extent_mm)
                                    .range(10.0..=400.0)
                                    .suffix(" mm"),
                            );
                            ui.label("across");
                        });
                    });
                    ui.checkbox(
                        &mut d.params.fill_interior,
                        "Solid body (fill lungs and gas)",
                    )
                    .on_hover_text(
                        "Lungs and bowel gas belong inside the body. Unticked gives \
                             the tissue shell the threshold sees instead, cavities open.",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Surface smoothing:");
                        ui.add(
                            egui::Slider::new(&mut d.params.close_mm, 0.0..=10.0)
                                .suffix(" mm")
                                .fixed_decimals(1),
                        )
                        .on_hover_text(
                            "A closing applied last, to take the staircase off the contour.",
                        );
                    });
                    if d.params.method == Method::ModelAssisted {
                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label("Network margin:");
                            ui.add(
                                egui::Slider::new(&mut d.params.guide_margin_mm, 0.0..=20.0)
                                    .suffix(" mm")
                                    .fixed_decimals(1),
                            )
                            .on_hover_text(
                                "How far the network's answer is grown before it is used as \
                                 a mask. It is planned at 6 mm or 1.5 mm, so it needs room \
                                 not to clip the skin the threshold found.",
                            );
                        });
                        device_row(ui, &mut d.params.device);
                        browse = models_dir_row(
                            ui,
                            &mut self.models_dir,
                            ModelsEngine::TotalSegmentator,
                        );
                    }
                });
                ui.separator();
                match d.params.method {
                    Method::Classical => licence_line(
                        ui,
                        "No weights and no network: thresholding and morphology, computed here.",
                        false,
                    ),
                    Method::ModelAssisted => {
                        let (note, warn) = weights_licence(ModelsEngine::TotalSegmentator);
                        licence_line(ui, note, warn)
                    }
                }
                ui.separator();
                match running {
                    Some(job) => cancel = progress_row(ui, &job.progress),
                    None => {
                        ui.horizontal(|ui| {
                            if ui
                                .button("▶ Contour")
                                .on_hover_text("Find the patient surface in the displayed series")
                                .clicked()
                            {
                                run = true;
                            }
                            if ui.button("Close").clicked() {
                                close = true;
                            }
                        });
                    }
                }
                if let Some(status) = &d.status {
                    ui.separator();
                    ui.weak(status);
                }
            },
        );
        if browse {
            if let Some(dir) = Self::pick_folder("Model folder") {
                self.models_dir = dir.display().to_string();
            }
        }
        if let Some(s) = switch {
            self.open_body_dialog(s);
            return;
        }
        if cancel {
            if let Some(job) = &self.body_job {
                job.progress.cancel();
            }
        }
        if run {
            self.start_body();
        }
        if !open || close {
            // The run, if any, carries on; the sidebar still shows it.
            self.body_dialog = None;
            self.persist_settings();
        }
    }
}

/// What an MR threshold starts at, and how far the bias estimate reaches.
const DEFAULT_MR_FRACTION: f32 = 0.12;
const DEFAULT_BIAS_SIGMA_MM: f64 = 40.0;

/// The threshold row - a different question on CT and on MR, so a different
/// row rather than one control that means two things.
fn foreground_row(ui: &mut egui::Ui, fg: &mut Foreground) {
    match fg {
        Foreground::Hu(t) => {
            ui.horizontal(|ui| {
                ui.label("Tissue above:");
                ui.add(egui::Slider::new(t, -900.0..=200.0).suffix(" HU"))
                    .on_hover_text(
                        "−300 HU sits in the gap between air and fat. The skin edge moves \
                         about half a millimetre per 100 HU through the partial-volume ramp, \
                         so this is the one number worth agreeing on with your planning \
                         system.",
                    );
            });
        }
        _ => {
            let mut otsu = matches!(fg, Foreground::MrOtsu { .. });
            // The fraction is remembered across a visit to Otsu and back;
            // losing a dialled-in threshold to a radio button is the kind of
            // small betrayal that stops people trying the other option.
            let id = ui.id().with("mr_fraction");
            let remembered: f32 = ui.data(|d| d.get_temp(id)).unwrap_or(DEFAULT_MR_FRACTION);
            let (mut fraction, mut sigma) = match *fg {
                Foreground::MrRelative { fraction, sigma_mm } => (fraction, sigma_mm),
                Foreground::MrOtsu { sigma_mm } => (remembered, sigma_mm),
                Foreground::Hu(_) => (remembered, DEFAULT_BIAS_SIGMA_MM),
            };
            ui.horizontal(|ui| {
                ui.label("Tissue above:");
                ui.radio_value(&mut otsu, false, "a fraction of the signal")
                    .on_hover_text(
                        "A low fraction of the 99th percentile. Robust, because the \
                         boundary being looked for is the largest step in the image.",
                    );
                ui.radio_value(&mut otsu, true, "Otsu").on_hover_text(
                    "No constant to pick, but Otsu splits bright from dark rather than \
                         tissue from air, so it runs high on fat-suppressed series.",
                );
            });
            ui.horizontal(|ui| {
                ui.add_enabled(
                    !otsu,
                    egui::Slider::new(&mut fraction, 0.02..=0.6)
                        .prefix("× p99  ")
                        .fixed_decimals(2),
                );
                ui.label("bias blur:");
                ui.add(
                    egui::DragValue::new(&mut sigma)
                        .range(0.0..=200.0)
                        .suffix(" mm"),
                )
                .on_hover_text(
                    "The receive coils shade the image, so one threshold cannot hold across \
                     it. Dividing by the image blurred far beyond any anatomy flattens the \
                     shading and leaves every edge intact.",
                );
            });
            ui.data_mut(|d| d.insert_temp(id, fraction));
            *fg = if otsu {
                Foreground::MrOtsu { sigma_mm: sigma }
            } else {
                Foreground::MrRelative {
                    fraction,
                    sigma_mm: sigma,
                }
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::device::DevicePref;

    #[test]
    fn the_tool_names_itself_like_the_others() {
        assert_eq!(BODY_CONTOUR.title(0), "👤 Body contour - dataset A");
        assert_eq!(BODY_CONTOUR.menu_entry(), "👤 Body contour");
        assert_eq!(BODY_CONTOUR.short_button(), "👤 Body");
    }

    #[test]
    fn a_device_preference_is_only_meaningful_with_a_network() {
        // The classical method never resolves a device; the default stays
        // Auto so that switching methods needs no second decision.
        let p = BodyParams::default();
        assert_eq!(p.method, Method::Classical);
        assert_eq!(p.device, DevicePref::Auto);
    }
}
