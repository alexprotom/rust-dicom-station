//! Modal and floating dialogs: auto-segmentation setup and results, the
//! synthetic study generator, the folder anonymizer and the DICOM exporter.

use super::*;

impl ViewerApp {
    /// Open the export dialog for `slot`, pre-filling the DICOM attributes
    /// from that study (an already-open dialog is re-targeted and refilled).
    pub(super) fn open_export_dialog(&mut self, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        self.export_params = Some(dicom_export::ExportParams::for_study(study));
        self.export_slot = slot;
        self.export_result = None;
        self.export_open = true;
    }

    // -- Auto-segmentation (TotalSegmentator, see the `autoseg` module) ----
    /// Open the tool window for auto-segmenting the given slot.
    pub(super) fn open_autoseg_dialog(&mut self, slot: usize) {
        if self.slots[slot].study.is_none() {
            return;
        }
        match &mut self.autoseg_dialog {
            // Re-target an open window unless it is busy with the other slot.
            Some(d) if self.autoseg_job.is_none() => d.slot = slot,
            Some(_) => {}
            None => {
                self.autoseg_dialog = Some(AutosegDialog {
                    slot,
                    variant: autoseg::Variant::Fast3mm,
                    device: autoseg::DevicePref::Auto,
                    parts: [true; 5],
                });
            }
        }
    }

    // -- Modals -----------------------------------------------------------
    pub(super) fn modals(&mut self, ctx: &egui::Context) {
        self.generator_window(ctx);
        self.anonymize_window(ctx);
        self.models_window(ctx);
        self.propagate_window(ctx);
        self.drr_window(ctx);
        self.export_window(ctx);
        self.rename_window(ctx);
        self.autoseg_run_window(ctx);
        self.segvol_window(ctx);
        self.medsam2_window(ctx);
        self.autoseg_result_window(ctx);
        if let Some(msg) = self.notice.clone() {
            egui::Window::new("Done")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(&msg);
                    ui.add_space(8.0);
                    if ui.button("OK").clicked() {
                        self.notice = None;
                    }
                });
        }
        if let Some(err) = self.error.clone() {
            egui::Window::new("Error")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    ui.label(&err);
                    ui.add_space(8.0);
                    if ui.button("OK").clicked() {
                        self.error = None;
                    }
                });
        }
    }

    /// The auto-segmentation tool window: model variant, compute device and
    /// model folder, then the run — whose progress replaces the buttons.
    pub(super) fn autoseg_run_window(&mut self, ctx: &egui::Context) {
        let Some(d) = &mut self.autoseg_dialog else {
            return;
        };
        if self.slots[d.slot].study.is_none() {
            self.autoseg_dialog = None;
            return;
        }
        let running = self
            .autoseg_job
            .as_ref()
            .filter(|_| self.autoseg_slot == d.slot);
        let mut open = true;
        let mut close = false;
        let mut run = false;
        let mut browse = false;
        let mut cancel = false;
        let models_dir = models::engine_dir(
            &models::root_from_setting(&self.models_dir),
            models::Engine::TotalSegmentator,
        );
        egui::Window::new(AUTOSEG.title(d.slot))
            .id(egui::Id::new("autoseg_window"))
            .collapsible(true)
            .resizable(false)
            .default_width(380.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(
                    "Segments the CT into up to 117 anatomical structures with \
                     TotalSegmentator's nnU-Net models, re-implemented natively in Rust.",
                );
                ui.separator();
                ui.label("Model:");
                for (variant, name, hint) in [
                    (
                        autoseg::Variant::Fast3mm,
                        "3 mm — fast",
                        "Single model, all 117 structures. Good quality, \
                         practical on any CPU.",
                    ),
                    (
                        autoseg::Variant::HighRes15mm,
                        "1.5 mm — high quality",
                        "Five sub-models at full resolution — the reference \
                         quality. Slow without a GPU.",
                    ),
                    (
                        autoseg::Variant::Preview6mm,
                        "6 mm — preview",
                        "Coarse but very fast — a quick look.",
                    ),
                ] {
                    let need = autoseg::download_needed(variant, d.parts, &models_dir);
                    let note = if need == 0 {
                        "weights cached ✓".to_string()
                    } else {
                        format!("downloads {} MB once", need / 1_000_000)
                    };
                    if ui
                        .add_enabled(
                            running.is_none(),
                            egui::RadioButton::new(
                                d.variant == variant,
                                format!("{name}  ({note})"),
                            ),
                        )
                        .on_hover_text(hint)
                        .clicked()
                    {
                        d.variant = variant;
                    }
                }
                if d.variant == autoseg::Variant::HighRes15mm {
                    ui.horizontal(|ui| {
                        ui.label("Sub-models:");
                        for (i, name) in autoseg::classes::PART_NAMES.iter().enumerate() {
                            ui.checkbox(&mut d.parts[i], *name);
                        }
                    });
                }
                ui.separator();
                ui.collapsing("Options", |ui| {
                    device_row(ui, &mut d.device);
                    browse =
                        models_dir_row(ui, &mut self.models_dir, models::Engine::TotalSegmentator);
                });
                ui.separator();
                licence_line(
                    ui,
                    "Weights: TotalSegmentator 'total' task (Apache-2.0), downloaded once \
                     from the official GitHub release.",
                    false,
                );
                ui.separator();
                match running {
                    Some(job) => cancel = progress_row(ui, &job.progress),
                    None => {
                        ui.horizontal(|ui| {
                            let can_run = d.variant != autoseg::Variant::HighRes15mm
                                || d.parts.iter().any(|p| *p);
                            if ui
                                .add_enabled(can_run, egui::Button::new("▶ Segment"))
                                .on_hover_text("Run the network on the whole volume")
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
            });
        if browse {
            if let Some(dir) = Self::pick_folder("Model folder") {
                self.models_dir = dir.display().to_string();
            }
        }
        if cancel {
            if let Some(job) = &self.autoseg_job {
                job.progress.cancel();
            }
        }
        if run {
            self.start_autoseg();
        }
        if !open || close {
            // The run, if any, carries on; the sidebar still shows it.
            self.autoseg_dialog = None;
            self.persist_settings();
        }
    }

    /// Organ-selection dialog shown when an auto-segmentation run finishes.
    pub(super) fn autoseg_result_window(&mut self, ctx: &egui::Context) {
        let Some(p) = &mut self.autoseg_pending else {
            return;
        };
        let mut open = true;
        let mut close_clicked = false;
        let mut apply_clicked = false;
        let vol_bytes = p.result.volume_dims[0] * p.result.volume_dims[1] * p.result.volume_dims[2];
        egui::Window::new(AUTOSEG.titled("results", p.slot))
            .collapsible(false)
            .resizable(true)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(format!(
                    "{} structures found on dataset {} — {} · {:.0} s",
                    p.result.organs.len(),
                    SLOT_NAMES[p.slot],
                    p.result.device,
                    p.result.elapsed_secs,
                ));
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.small_button("All").clicked() {
                        p.selected.iter_mut().for_each(|s| *s = true);
                    }
                    if ui.small_button("None").clicked() {
                        p.selected.iter_mut().for_each(|s| *s = false);
                    }
                    let n_sel = p.selected.iter().filter(|s| **s).count();
                    ui.weak(format!(
                        "{} selected · ≈ {} MB of masks",
                        n_sel,
                        n_sel * vol_bytes / 1_000_000
                    ));
                });
                egui::ScrollArea::vertical()
                    .max_height(320.0)
                    .show(ui, |ui| {
                        for (organ, sel) in p.result.organs.iter().zip(p.selected.iter_mut()) {
                            ui.horizontal(|ui| {
                                ui.checkbox(sel, "");
                                let (rect, _) =
                                    ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                                ui.painter().rect_filled(
                                    rect,
                                    2.0,
                                    Color32::from_rgb(
                                        organ.color[0],
                                        organ.color[1],
                                        organ.color[2],
                                    ),
                                );
                                ui.label(organ.name);
                                ui.weak(format!("{:.1} cm³", organ.cm3));
                            });
                        }
                    });
                ui.add_space(4.0);
                ui.checkbox(&mut p.also_rs, "Also convert to RTSTRUCT contours (→RS)")
                    .on_hover_text(
                        "Adds each selected structure as a ROI to the active structure \
                     set, so it renders like any ROI and rides the DICOM export",
                    );
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let n_sel = p.selected.iter().filter(|s| **s).count();
                    if ui
                        .add_enabled(
                            n_sel > 0,
                            egui::Button::new(format!("Add {n_sel} segmentation(s)")),
                        )
                        .clicked()
                    {
                        apply_clicked = true;
                    }
                    if ui.button("Discard").clicked() {
                        close_clicked = true;
                    }
                });
            });
        if apply_clicked && !close_clicked {
            self.apply_autoseg_selection();
        } else if !open || close_clicked {
            self.autoseg_pending = None;
        }
    }

    /// Built-in synthetic test-data generator: pick an output folder, tweak
    /// the phantom parameters and write a complete RT study.
    pub(super) fn generator_window(&mut self, ctx: &egui::Context) {
        if !self.gen_open {
            return;
        }
        let running = self.gen_job.is_some();
        let mut open = true;
        let mut do_generate = false;
        let mut browse = false;
        let mut reset_dir = false;
        let mut reset_params = false;

        egui::Window::new("🧪 Generate synthetic RT test study")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_max_width(560.0);
                ui.label(
                    "Writes a self-contained test study: 40-slice CT water phantom with a \
                     spherical target and a cord, matching RTSTRUCT contours, a Gaussian \
                     RTDOSE and a two-beam proton RTPLAN.",
                );
                ui.add_space(6.0);

                ui.label(egui::RichText::new("Output folder").strong());
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.gen_dir)
                            .desired_width(360.0)
                            .hint_text("folder to write the DICOM files into"),
                    );
                    if ui.button("📂 Browse…").clicked() {
                        browse = true;
                    }
                    if ui
                        .button("↺")
                        .on_hover_text("Reset to the application folder")
                        .clicked()
                    {
                        reset_dir = true;
                    }
                });
                ui.weak(format!(
                    "Files: {}",
                    gen_test_data::output_summary(&self.gen_params)
                ));

                ui.add_space(6.0);
                ui.label(egui::RichText::new("Phantom").strong());
                egui::Grid::new("gen_params_grid")
                    .num_columns(2)
                    .spacing([8.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("Dose peak (Gy)")
                            .on_hover_text("Also written as the plan's prescription dose");
                        ui.add(
                            egui::DragValue::new(&mut self.gen_params.peak)
                                .speed(0.5)
                                .range(0.1..=200.0),
                        );
                        ui.end_row();

                        ui.label("Target Y shift (mm)").on_hover_text(
                            "Moves the target sphere and the dose peak inside the phantom",
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.gen_params.target_shift_y)
                                .speed(0.5)
                                .range(-60.0..=60.0),
                        );
                        ui.end_row();

                        ui.label("Phantom shift X / Y (mm)")
                            .on_hover_text("Shifts the whole phantom — for registration tests");
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.gen_params.shift_x)
                                    .speed(0.5)
                                    .range(-60.0..=60.0),
                            );
                            ui.add(
                                egui::DragValue::new(&mut self.gen_params.shift_y)
                                    .speed(0.5)
                                    .range(-60.0..=60.0),
                            );
                        });
                        ui.end_row();

                        ui.label("Plan label");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.gen_params.plan_label)
                                .desired_width(160.0)
                                .char_limit(16),
                        );
                        ui.end_row();

                        ui.label("REG translation (mm)").on_hover_text(
                            "Translation written into the REG object's second matrix",
                        );
                        ui.horizontal(|ui| {
                            for v in &mut self.gen_params.reg_shift {
                                ui.add(egui::DragValue::new(v).speed(0.5).range(-100.0..=100.0));
                            }
                        });
                        ui.end_row();
                    });
                ui.checkbox(
                    &mut self.gen_params.extras,
                    "Also write DX, RTIMAGE, REG and RTRECORD objects",
                );
                ui.checkbox(
                    &mut self.gen_load_after,
                    "Load the study into slot A when done",
                );

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if running {
                        ui.spinner();
                        if let Some(job) = &self.gen_job {
                            ui.label(job.progress.get());
                        }
                    } else {
                        if ui
                            .add(egui::Button::new("⚙ Generate"))
                            .on_hover_text("Existing files with the same names are overwritten")
                            .clicked()
                        {
                            do_generate = true;
                        }
                        if ui.button("Defaults").clicked() {
                            reset_params = true;
                        }
                    }
                });
                if let Some(msg) = &self.gen_result {
                    ui.add_space(4.0);
                    ui.label(msg);
                }
            });

        self.gen_open = open;
        if browse {
            if let Some(dir) = Self::pick_folder("Select an output folder for the test data") {
                self.gen_dir = dir.display().to_string();
            }
        }
        if reset_dir {
            self.gen_dir = gen_test_data::default_output_dir().display().to_string();
        }
        if reset_params {
            self.gen_params = GenParams::default();
        }
        if do_generate {
            self.start_generate();
        }
    }

    /// The anonymizer tool window: pick a folder, scan it, review every
    /// identifying tag (current values, proposed replacement — editable),
    /// then rewrite the files.
    pub(super) fn anonymize_window(&mut self, ctx: &egui::Context) {
        if !self.anon_open {
            return;
        }
        // Poll background jobs.
        match poll_job(&mut self.anon_scan_job, ctx, "Scan", &mut self.error) {
            Some(Ok(mut scan)) => {
                if self.anon_out.trim().is_empty() {
                    self.anon_out = format!("{}_anon", scan.root.display());
                }
                scan.warnings.truncate(8);
                self.anon_scan = Some(scan);
            }
            Some(Err(e)) => self.error = Some(format!("{e:#}")),
            None => {}
        }
        match poll_job(&mut self.anon_apply_job, ctx, "Anonymize", &mut self.error) {
            Some(Ok(n)) => {
                let dest = if self.anon_in_place {
                    "in place".to_string()
                } else {
                    format!("into {}", self.anon_out.trim())
                };
                self.anon_result = Some(format!("✔ {n} file(s) anonymized {dest}"));
            }
            Some(Err(e)) => self.error = Some(format!("{e:#}")),
            None => {}
        }

        let busy = self.anon_scan_job.is_some() || self.anon_apply_job.is_some();
        let mut open = true;
        let mut browse_in = false;
        let mut browse_out = false;
        let mut do_scan = false;
        let mut do_apply = false;

        egui::Window::new("🔏 Anonymize DICOM folder")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([780.0, 560.0])
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(
                    "Scans a folder, shows every identifying tag with its current values \
                     and a proposed replacement (editable), then rewrites the files. \
                     UIDs are regenerated consistently across all files, so series, \
                     structure-set, plan and dose references stay linked.",
                );
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("DICOM folder").strong());
                    ui.add(
                        egui::TextEdit::singleline(&mut self.anon_dir)
                            .desired_width(420.0)
                            .hint_text("folder to scan (recursively)"),
                    );
                    if ui.button("📂 Browse…").clicked() {
                        browse_in = true;
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("🔍 Scan"))
                        .clicked()
                    {
                        do_scan = true;
                    }
                });

                if let Some(job) = &self.anon_scan_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                }

                if let Some(scan) = &mut self.anon_scan {
                    let mods = scan
                        .modalities
                        .iter()
                        .map(|(m, n)| format!("{m} {n}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    ui.weak(format!(
                        "{} DICOM file(s) ({mods}) — {} unique UID(s), {} private element(s)",
                        scan.files.len(),
                        scan.uid_count,
                        scan.private_count
                    ));
                    for w in &scan.warnings {
                        ui.colored_label(warn_color(ui.visuals()), format!("⚠ {w}"));
                    }
                    ui.add_space(4.0);

                    // ---- findings table ----
                    egui::ScrollArea::vertical()
                        .max_height(ui.available_height() - 130.0)
                        .show(ui, |ui| {
                            egui::Grid::new("anon_grid")
                                .num_columns(3)
                                .striped(true)
                                .spacing([10.0, 3.0])
                                .show(ui, |ui| {
                                    ui.label(egui::RichText::new("Tag").strong());
                                    ui.label(egui::RichText::new("Current value(s)").strong());
                                    ui.label(egui::RichText::new("New value").strong());
                                    ui.end_row();
                                    for f in &mut scan.findings {
                                        let label = format!(
                                            "({:04X},{:04X}) {}",
                                            f.tag.group(),
                                            f.tag.element(),
                                            f.name
                                        );
                                        ui.checkbox(&mut f.enabled, label).on_hover_text(format!(
                                            "present in {} file(s); unchecked rows are \
                                                 left unchanged",
                                            f.n_files
                                        ));
                                        let shown = f
                                            .values
                                            .iter()
                                            .map(|v| {
                                                if v.is_empty() {
                                                    "«empty»"
                                                } else {
                                                    v.as_str()
                                                }
                                            })
                                            .collect::<Vec<_>>()
                                            .join("  |  ");
                                        let extra = f.n_values.saturating_sub(f.values.len());
                                        let txt = if extra > 0 {
                                            format!("{shown}  (+{extra} more)")
                                        } else {
                                            shown
                                        };
                                        ui.add(
                                            egui::Label::new(egui::RichText::new(txt).weak())
                                                .truncate(),
                                        )
                                        .on_hover_text(f.values.join("\n"));
                                        ui.horizontal(|ui| {
                                            ui.add_enabled(
                                                f.enabled,
                                                egui::TextEdit::singleline(&mut f.replacement)
                                                    .desired_width(170.0)
                                                    .hint_text("(clear)"),
                                            );
                                            if f.replacement != f.suggested
                                                && ui
                                                    .small_button("↺")
                                                    .on_hover_text(format!(
                                                        "Back to the suggestion: “{}”",
                                                        if f.suggested.is_empty() {
                                                            "(clear)"
                                                        } else {
                                                            &f.suggested
                                                        }
                                                    ))
                                                    .clicked()
                                            {
                                                f.replacement = f.suggested.clone();
                                            }
                                        });
                                        ui.end_row();
                                    }
                                });
                        });

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.checkbox(
                            &mut self.anon_remap_uids,
                            format!("Regenerate {} UID(s)", scan.uid_count),
                        )
                        .on_hover_text(
                            "Every study / series / SOP instance / frame-of-reference UID is \
                             replaced by a fresh one — the same original always maps to the \
                             same new UID, so cross-references stay valid",
                        );
                        ui.checkbox(
                            &mut self.anon_remove_private,
                            format!("Remove {} private element(s)", scan.private_count),
                        )
                        .on_hover_text("Drops all odd-group (vendor private) elements");
                        ui.checkbox(&mut self.anon_mark, "Mark as de-identified")
                            .on_hover_text(
                                "Writes PatientIdentityRemoved=YES and DeidentificationMethod",
                            );
                    });

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Write to").strong());
                        ui.add_enabled(
                            !self.anon_in_place,
                            egui::TextEdit::singleline(&mut self.anon_out)
                                .desired_width(380.0)
                                .hint_text("output folder (files keep their relative paths)"),
                        );
                        if ui
                            .add_enabled(!self.anon_in_place, egui::Button::new("📂 Browse…"))
                            .clicked()
                        {
                            browse_out = true;
                        }
                        ui.checkbox(&mut self.anon_in_place, "overwrite in place")
                            .on_hover_text("Rewrites the original files — no copy is kept");
                    });

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if let Some(job) = &self.anon_apply_job {
                            ui.spinner();
                            ui.label(job.progress.get());
                        } else if ui
                            .add_enabled(!busy, egui::Button::new("🔏 Anonymize"))
                            .on_hover_text("Applies the checked replacements to every file")
                            .clicked()
                        {
                            do_apply = true;
                        }
                        if let Some(msg) = &self.anon_result {
                            ui.label(msg);
                        }
                    });
                }
            });

        if !open {
            // Closing the window forgets everything that was scanned — the
            // findings (they contain patient identity!), any running scan,
            // the result line and the derived output path. Only the folder
            // field is kept for convenience.
            self.anon_scan = None;
            self.anon_scan_job = None;
            self.anon_result = None;
            self.anon_out.clear();
            self.anon_in_place = false;
            // A running rewrite is not aborted — the background thread
            // finishes writing; only its completion message is dropped.
            self.anon_apply_job = None;
        }
        self.anon_open = open;
        if browse_in {
            if let Some(dir) = Self::pick_folder("Select a DICOM folder to anonymize") {
                self.anon_dir = dir.display().to_string();
            }
        }
        if browse_out {
            if let Some(dir) = Self::pick_folder("Select the output folder") {
                self.anon_out = dir.display().to_string();
            }
        }
        if do_scan {
            self.anon_start_scan();
        }
        if do_apply {
            self.anon_start_apply();
        }
    }

    /// The DICOM export dialog: choose the output folder, review and edit
    /// every patient / study / equipment attribute that will be written,
    /// then write the study out.
    pub(super) fn export_window(&mut self, ctx: &egui::Context) {
        if !self.export_open {
            return;
        }
        let slot = self.export_slot.min(1);
        if self.slots[slot].study.is_none() {
            self.export_open = false;
            return;
        }
        let busy = self.export_job.is_some();
        let mut open = true;
        let mut browse = false;
        let mut do_export = false;
        let mut reset_all = false;

        egui::Window::new(format!("💾 Export dataset {} as DICOM", SLOT_NAMES[slot]))
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size([720.0, 520.0])
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(
                    "Writes the displayed volume (one file per slice) plus every \
                     structure set, segmentation series (as DICOM SEG), dose grid \
                     and plan as DICOM objects. The \
                     attributes below are pre-filled from the loaded study and \
                     written into every exported file; SOP / series / study instance \
                     UIDs are always freshly generated, with the cross-references \
                     between the objects kept consistent.",
                );
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Write to").strong());
                    ui.add(
                        egui::TextEdit::singleline(&mut self.export_dir)
                            .desired_width(420.0)
                            .hint_text("output folder (created if missing)"),
                    );
                    if ui.button("📂 Browse…").clicked() {
                        browse = true;
                    }
                });
                ui.add_space(4.0);

                let Some(params) = &mut self.export_params else {
                    return;
                };
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("DICOM tags").strong());
                    ui.weak("unchecked rows are not written at all");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .small_button("↺ all")
                            .on_hover_text("Restore every value to the study's own")
                            .clicked()
                        {
                            reset_all = true;
                        }
                    });
                });

                egui::ScrollArea::vertical()
                    .max_height((ui.available_height() - 90.0).max(120.0))
                    .show(ui, |ui| {
                        egui::Grid::new("export_grid")
                            .num_columns(2)
                            .striped(true)
                            .spacing([10.0, 3.0])
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("Tag").strong());
                                ui.label(egui::RichText::new("Value").strong());
                                ui.end_row();
                                for f in &mut params.fields {
                                    let label = format!(
                                        "({:04X},{:04X}) {}",
                                        f.tag.group(),
                                        f.tag.element(),
                                        f.name
                                    );
                                    ui.checkbox(&mut f.enabled, label).on_hover_text(format!(
                                        "VR {} — unchecked: the tag is left out of the \
                                         exported files",
                                        f.vr
                                    ));
                                    ui.horizontal(|ui| {
                                        ui.add_enabled(
                                            f.enabled,
                                            egui::TextEdit::singleline(&mut f.value)
                                                .desired_width(300.0)
                                                .hint_text("(empty)"),
                                        );
                                        if f.value != f.suggested
                                            && ui
                                                .small_button("↺")
                                                .on_hover_text(format!(
                                                    "Back to the study's value: “{}”",
                                                    if f.suggested.is_empty() {
                                                        "(empty)"
                                                    } else {
                                                        &f.suggested
                                                    }
                                                ))
                                                .clicked()
                                        {
                                            f.value = f.suggested.clone();
                                        }
                                    });
                                    ui.end_row();
                                }
                            });
                    });

                ui.add_space(4.0);
                ui.checkbox(
                    &mut params.keep_frame_of_reference,
                    "Keep the source Frame of Reference UID",
                )
                .on_hover_text(
                    "On: the export stays spatially linked to its source study, so the \
                     two load as a comparable pair.\nOff: a fresh frame of reference \
                     is generated",
                );

                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if let Some(job) = &self.export_job {
                        ui.spinner();
                        ui.label(job.progress.get());
                    } else if ui
                        .add_enabled(!busy, egui::Button::new("💾 Export"))
                        .on_hover_text("Write the DICOM files into the output folder")
                        .clicked()
                    {
                        do_export = true;
                    }
                    if let Some(msg) = &self.export_result {
                        ui.label(msg);
                    }
                });
            });

        // A running export is not aborted when the window closes — the
        // background thread finishes writing; only its message is dropped.
        self.export_open = open;
        if !open {
            self.export_result = None;
        }
        if reset_all {
            if let Some(params) = &mut self.export_params {
                for f in &mut params.fields {
                    f.value = f.suggested.clone();
                    f.enabled = true;
                }
            }
        }
        if browse {
            if let Some(dir) = Self::pick_folder("Select the export output folder") {
                self.export_dir = dir.display().to_string();
            }
        }
        if do_export {
            self.start_export();
        }
    }
}
