//! The side panel and its per-dataset sections.
//!
//! Each section renders one kind of loaded object -- series, structures,
//! segmentations, dose, plan, planar images, registrations, records -- plus
//! the global registration and simulation controls.

use super::*;

impl ViewerApp {
    // -- Side panel -------------------------------------------------------
    pub(super) fn side_panel(&mut self, ui: &mut egui::Ui) {
        if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
            return;
        }
        egui::Panel::left(egui::Id::new("side"))
            .resizable(true)
            .default_size(280.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.registration_section(ui);
                    self.simulation_section(ui);
                    for slot in 0..2 {
                        if self.slots[slot].study.is_none() {
                            continue;
                        }
                        self.study_section(ui, slot);
                    }
                });
            });
    }

    pub(super) fn registration_section(&mut self, ui: &mut egui::Ui) {
        let both = self.slots[0].study.is_some() && self.slots[1].study.is_some();
        if !both && self.registration.is_none() && self.reg_job.is_none() {
            return;
        }
        let mut do_reg: Option<RegKind> = None;
        let mut cancel_reg = false;
        let mut clear_reg = false;
        egui::CollapsingHeader::new(egui::RichText::new("Registration").strong())
            .default_open(true)
            .show(ui, |ui| {
                if let Some(job) = &self.reg_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                    if ui.button("Cancel").clicked() {
                        cancel_reg = true;
                    }
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Direction");
                    ui.selectable_value(&mut self.reg_fixed_slot, 0, "B ▶ A")
                        .on_hover_text("B is deformed/moved onto A; fusion shown on A");
                    ui.selectable_value(&mut self.reg_fixed_slot, 1, "A ▶ B")
                        .on_hover_text("A is deformed/moved onto B; fusion shown on B");
                });

                ui.horizontal(|ui| {
                    ui.label("Iterations/level");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_iterations)
                            .speed(10)
                            .range(50..=5000),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Samples/iter");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_samples)
                            .speed(100)
                            .range(500..=50000),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("B-spline grid");
                    ui.add(
                        egui::DragValue::new(&mut self.reg_grid_mm)
                            .speed(1.0)
                            .range(8.0..=128.0)
                            .suffix(" mm"),
                    );
                });
                ui.horizontal(|ui| {
                    if ui.add_enabled(both, egui::Button::new("▶ Rigid")).clicked() {
                        do_reg = Some(RegKind::Rigid);
                    }
                    if ui
                        .add_enabled(both, egui::Button::new("▶ Deformable"))
                        .clicked()
                    {
                        do_reg = Some(RegKind::Deformable);
                    }
                });

                if let Some(reg) = &self.registration {
                    ui.separator();
                    let res = &reg.result;
                    let kind = match res.kind {
                        RegKind::Rigid => "Rigid (Euler 6-DOF)",
                        RegKind::Deformable => "Rigid + B-spline FFD",
                    };
                    let moving = SLOT_NAMES[1 - reg.fixed_slot];
                    let fixed = SLOT_NAMES[reg.fixed_slot];
                    ui.label(format!("✔ {kind}  ({moving} ▶ {fixed})"));
                    ui.weak(format!(
                        "MSD {:.1} ▶ {:.1}  ({} iters, {:.1} s)",
                        res.initial_metric, res.final_metric, res.iterations_run, res.elapsed_secs
                    ));
                    let t = res.transform.rigid.params();
                    ui.weak(format!(
                        "t = ({:.1}, {:.1}, {:.1}) mm  r = ({:.2}, {:.2}, {:.2})°",
                        t[3],
                        t[4],
                        t[5],
                        t[0].to_degrees(),
                        t[1].to_degrees(),
                        t[2].to_degrees()
                    ));
                    ui.checkbox(&mut self.fusion_on, format!("Fusion overlay on {fixed}"));
                    let resp = ui.add(
                        egui::Slider::new(&mut self.fusion_weight, 0.0..=1.0).text("Fusion blend"),
                    );
                    let _ = resp;
                    if ui.button("Clear registration").clicked() {
                        clear_reg = true;
                    }
                }
            });
        ui.separator();
        if let Some(kind) = do_reg {
            self.start_registration(kind);
        }
        if cancel_reg {
            if let Some(job) = &self.reg_job {
                job.progress.cancel();
            }
        }
        if clear_reg {
            self.clear_registration();
        }
    }

    /// Study transform simulator: apply a known rigid motion + optional
    /// Gaussian deformation to a study and generate the result into the
    /// other slot (the generated study is exportable via *File ▶ Export*).
    pub(super) fn simulation_section(&mut self, ui: &mut egui::Ui) {
        if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
            return;
        }
        let mut do_generate = false;
        egui::CollapsingHeader::new(egui::RichText::new("Simulation (registration QA)").strong())
            .default_open(false)
            .show(ui, |ui| {
                if let Some(job) = &self.sim_job {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(job.progress.get());
                    });
                    return;
                }

                ui.horizontal(|ui| {
                    ui.label("Source");
                    ui.selectable_value(&mut self.sim_source, 0, "A");
                    ui.selectable_value(&mut self.sim_source, 1, "B");
                    ui.weak(format!(
                        "▶ generates dataset {}",
                        SLOT_NAMES[1 - self.sim_source.min(1)]
                    ));
                });

                ui.label("Rigid motion:");
                ui.horizontal(|ui| {
                    ui.label("t (mm)");
                    for v in &mut self.sim_params.translation {
                        ui.add(egui::DragValue::new(v).speed(0.5).range(-200.0..=200.0));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("r (°)");
                    for v in &mut self.sim_params.rotation_deg {
                        ui.add(egui::DragValue::new(v).speed(0.2).range(-45.0..=45.0));
                    }
                });

                ui.label("Gaussian deformation (0 = off):");
                ui.horizontal(|ui| {
                    ui.label("amp (mm)");
                    for v in &mut self.sim_params.bump_amp {
                        ui.add(egui::DragValue::new(v).speed(0.5).range(-40.0..=40.0));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("σ (mm)");
                    ui.add(
                        egui::DragValue::new(&mut self.sim_params.bump_sigma)
                            .speed(1.0)
                            .range(5.0..=200.0),
                    );
                    ui.weak("centered at the crosshair");
                });

                let src_ok = self.slots[self.sim_source.min(1)].study.is_some();
                if ui
                    .add_enabled(
                        src_ok && self.loading.is_none(),
                        egui::Button::new(format!(
                            "⚙ Generate transformed dataset ▶ {}",
                            SLOT_NAMES[1 - self.sim_source.min(1)]
                        )),
                    )
                    .clicked()
                {
                    do_generate = true;
                }
                if let Some(s) = &self.last_sim {
                    ui.weak(format!("Ground truth {s}"));
                }
            });
        ui.separator();
        if do_generate {
            self.start_simulation();
        }
    }

    pub(super) fn study_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        // Plain header — the patient(s) always appear as tree nodes below.
        let header = format!("Dataset {}", SLOT_NAMES[slot]);
        let ch = egui::CollapsingHeader::new(egui::RichText::new(header).strong())
            .id_salt(("study_hdr", slot))
            .default_open(true)
            .show(ui, |ui| {
                self.series_selector(ui, slot);
                self.structures_section(ui, slot);
                self.segmentation_section(ui, slot);
                self.dose_section(ui, slot);
                self.plan_section(ui, slot);
                self.planar_section(ui, slot);
                self.reg_objects_section(ui, slot);
                self.records_section(ui, slot);
                self.warnings_section(ui, slot);
            });
        // Right-click on the dataset header: clear the slot.
        let mut clear = false;
        ch.header_response.context_menu(|ui| {
            if ui
                .button(format!("Clear dataset {}", SLOT_NAMES[slot]))
                .clicked()
            {
                clear = true;
                ui.close();
            }
        });
        if clear {
            if slot == 1 {
                self.close_comparison();
            } else {
                self.tree_clear_slot(slot);
            }
        }
        ui.separator();
    }

    /// DICOM data tree: patient ▶ study ▶ series, all visible at once. The
    /// active series (the displayed volume) is marked; clicking another
    /// series loads it. Right-click any level to copy / move it to the
    /// other dataset or remove it.
    pub(super) fn series_selector(&mut self, ui: &mut egui::Ui, slot: usize) {
        let mut switch_to = None;
        let mut act_series: Option<TreeAction> = None;
        let mut act_study: Option<TreeAction> = None;
        let mut act_patient: Option<TreeAction> = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            let active = study.active_series;
            let other = SLOT_NAMES[1 - slot];
            let label = |s: &loader::SeriesInfo| {
                format!(
                    "{} {} ({} sl.)",
                    s.modality,
                    if s.description.is_empty() {
                        "series"
                    } else {
                        &s.description
                    },
                    s.files.len()
                )
            };
            // Distinct patients, in first-seen order.
            let mut patients: Vec<&str> = Vec::new();
            for s in &study.series {
                let k = s.patient_key();
                if !patients.contains(&k) {
                    patients.push(k);
                }
            }
            for (pi, pkey) in patients.iter().enumerate() {
                let pinfo = study
                    .series
                    .iter()
                    .find(|s| s.patient_key() == *pkey)
                    .unwrap();
                let pname = pinfo.patient_name.replace('^', " ");
                let ptitle = if pname.is_empty() && pinfo.patient_id.is_empty() {
                    "Unknown patient".to_string()
                } else if pname.is_empty() {
                    format!("Patient {}", pinfo.patient_id)
                } else if pinfo.patient_id.is_empty() {
                    pname.clone()
                } else {
                    format!("{} ({})", pname, pinfo.patient_id)
                };
                let pch = egui::CollapsingHeader::new(ptitle)
                    .id_salt(("pat_hdr", slot, pi))
                    .default_open(true)
                    .show(ui, |ui| {
                        // Studies of this patient, in first-seen order.
                        let mut studies: Vec<&str> = Vec::new();
                        for s in &study.series {
                            if s.patient_key() == *pkey && !studies.contains(&s.study_uid.as_str())
                            {
                                studies.push(&s.study_uid);
                            }
                        }
                        for (si, study_uid) in studies.iter().enumerate() {
                            let info = study
                                .series
                                .iter()
                                .find(|s| s.study_uid == *study_uid && s.patient_key() == *pkey)
                                .unwrap();
                            let title = format!(
                                "Study {}{}",
                                if info.study_date.is_empty() {
                                    format!("{}", si + 1)
                                } else {
                                    info.study_date.clone()
                                },
                                if info.study_description.is_empty() {
                                    String::new()
                                } else {
                                    format!(" — {}", info.study_description)
                                }
                            );
                            let sch = egui::CollapsingHeader::new(title)
                                .id_salt(("study_tree", slot, pi, si))
                                .default_open(true)
                                .show(ui, |ui| {
                                    for (i, s) in study.series.iter().enumerate() {
                                        if s.study_uid != *study_uid || s.patient_key() != *pkey {
                                            continue;
                                        }
                                        let resp = ui.selectable_label(i == active, label(s));
                                        if resp.clicked() && i != active {
                                            switch_to = Some(i);
                                        }
                                        resp.context_menu(|ui| {
                                            if ui
                                                .button(format!("Copy series to dataset {other}"))
                                                .clicked()
                                            {
                                                act_series = Some(TreeAction {
                                                    from: slot,
                                                    sel: TreeSel::Series(i),
                                                    op: TreeOp::Copy,
                                                });
                                                ui.close();
                                            }
                                            if ui
                                                .button(format!("Move series to dataset {other}"))
                                                .clicked()
                                            {
                                                act_series = Some(TreeAction {
                                                    from: slot,
                                                    sel: TreeSel::Series(i),
                                                    op: TreeOp::Move,
                                                });
                                                ui.close();
                                            }
                                            ui.separator();
                                            if ui.button("Remove series").clicked() {
                                                act_series = Some(TreeAction {
                                                    from: slot,
                                                    sel: TreeSel::Series(i),
                                                    op: TreeOp::Remove,
                                                });
                                                ui.close();
                                            }
                                        });
                                        resp.on_hover_text(format!(
                                            "Series UID …{}\nright-click: copy / move to \
                                             dataset {other}, or remove",
                                            tail(&s.uid)
                                        ));
                                    }
                                });
                            sch.header_response.context_menu(|ui| {
                                if ui
                                    .button(format!("Copy study to dataset {other}"))
                                    .clicked()
                                {
                                    act_study = Some(TreeAction {
                                        from: slot,
                                        sel: TreeSel::Study(study_uid.to_string()),
                                        op: TreeOp::Copy,
                                    });
                                    ui.close();
                                }
                                if ui
                                    .button(format!("Move study to dataset {other}"))
                                    .clicked()
                                {
                                    act_study = Some(TreeAction {
                                        from: slot,
                                        sel: TreeSel::Study(study_uid.to_string()),
                                        op: TreeOp::Move,
                                    });
                                    ui.close();
                                }
                                ui.separator();
                                if ui.button("Remove study").clicked() {
                                    act_study = Some(TreeAction {
                                        from: slot,
                                        sel: TreeSel::Study(study_uid.to_string()),
                                        op: TreeOp::Remove,
                                    });
                                    ui.close();
                                }
                            });
                        }
                    });
                pch.header_response.context_menu(|ui| {
                    if ui
                        .button(format!("Copy patient to dataset {other}"))
                        .clicked()
                    {
                        act_patient = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Patient(pkey.to_string()),
                            op: TreeOp::Copy,
                        });
                        ui.close();
                    }
                    if ui
                        .button(format!("Move patient to dataset {other}"))
                        .clicked()
                    {
                        act_patient = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Patient(pkey.to_string()),
                            op: TreeOp::Move,
                        });
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Remove patient").clicked() {
                        act_patient = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Patient(pkey.to_string()),
                            op: TreeOp::Remove,
                        });
                        ui.close();
                    }
                });
            }
        }
        if let Some(a) = act_series.or(act_study).or(act_patient) {
            self.tree_action = Some(a);
        }
        if let Some(i) = switch_to {
            self.start_series_switch(slot, i);
        }
    }

    pub(super) fn structures_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n_sets = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.structure_sets.len())
            .unwrap_or(0);
        if n_sets == 0 {
            // No RTSTRUCT in this study — show nothing.
            return;
        }
        let mut changed = false;
        let mut new_active: Option<usize> = None;
        {
            let StudySlot {
                study,
                roi_visible,
                active_structs,
                ..
            } = &mut self.slots[slot];
            let study = study.as_ref().unwrap();
            let active_set = (*active_structs).min(n_sets - 1);
            let ss = &study.structure_sets[active_set];
            let n_vis = roi_visible.iter().filter(|v| **v).count();
            egui::CollapsingHeader::new(format!("Structures ({}/{})", n_vis, ss.rois.len()))
                .id_salt(("structs", slot))
                .default_open(true)
                .show(ui, |ui| {
                    // Structure-set selector with links to the referenced
                    // image series (one set per 4DCT phase, replans, …).
                    if n_sets > 1 {
                        for (i, set) in study.structure_sets.iter().enumerate() {
                            let series_ref = study
                                .series
                                .iter()
                                .find(|se| se.uid == set.referenced_series_uid)
                                .map(|se| {
                                    format!(
                                        " ▶ {} {}",
                                        se.modality,
                                        if se.description.is_empty() {
                                            "series"
                                        } else {
                                            &se.description
                                        }
                                    )
                                })
                                .unwrap_or_default();
                            let resp = ui.selectable_label(
                                i == active_set,
                                format!(
                                    "{} ({} ROIs){}",
                                    if set.label.is_empty() {
                                        &set.file_name
                                    } else {
                                        &set.label
                                    },
                                    set.rois.len(),
                                    series_ref
                                ),
                            );
                            if resp.clicked() && i != active_set {
                                new_active = Some(i);
                            }
                            resp.on_hover_text(format!(
                                "{}\nreferences series …{}",
                                set.file_name,
                                tail(&set.referenced_series_uid)
                            ));
                        }
                        ui.separator();
                    }
                    ui.horizontal(|ui| {
                        if ui.small_button("All").clicked() {
                            roi_visible.iter_mut().for_each(|v| *v = true);
                            changed = true;
                        }
                        if ui.small_button("None").clicked() {
                            roi_visible.iter_mut().for_each(|v| *v = false);
                            changed = true;
                        }
                        ui.weak(&ss.label);
                    });
                    for (i, roi) in ss.rois.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                            ui.painter().rect_filled(
                                rect,
                                2.0,
                                Color32::from_rgb(roi.color[0], roi.color[1], roi.color[2]),
                            );
                            let resp = ui.checkbox(
                                &mut roi_visible[i],
                                format!(
                                    "{}{}",
                                    roi.name,
                                    if roi.roi_type.is_empty() {
                                        String::new()
                                    } else {
                                        format!("  [{}]", roi.roi_type)
                                    }
                                ),
                            );
                            if resp.changed() {
                                changed = true;
                            }
                            resp.on_hover_text(format!(
                                "ROI {} · {} contour(s)",
                                roi.number,
                                roi.contours.len()
                            ));
                        });
                    }
                });
        }
        if let Some(i) = new_active {
            let s = &mut self.slots[slot];
            s.active_structs = i;
            let n = s
                .study
                .as_ref()
                .map(|st| st.structure_sets[i].rois.len())
                .unwrap_or(0);
            s.roi_visible = vec![true; n];
            changed = true;
        }
        if changed {
            self.settings_gen += 1;
        }
    }

    pub(super) fn segmentation_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let mut make_new = false;
        let mut open_auto = false;
        let mut cancel_auto = false;
        let mut delete: Option<usize> = None;
        let mut to_struct: Option<usize> = None;
        // Auto-segmentation status for this slot (read before the slot borrow).
        let auto_state = self
            .autoseg_job
            .as_ref()
            .filter(|_| self.autoseg_slot == slot)
            .map(|job| (job.progress.get(), job.progress.frac()));
        let auto_enabled = self.autoseg_job.is_none();
        {
            let StudySlot {
                study,
                segs,
                active_seg,
                ..
            } = &mut self.slots[slot];
            let Some(study) = study.as_ref() else { return };
            let spacing = study.volume.spacing;
            egui::CollapsingHeader::new(format!("Segmentations ({})", segs.len()))
                .id_salt(("segs", slot))
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.small_button("➕ New").clicked() {
                            make_new = true;
                        }
                        if ui
                            .add_enabled(auto_enabled, egui::Button::new("🤖 Auto…").small())
                            .on_hover_text(
                                "Automatic multi-organ segmentation \
                                 (TotalSegmentator, 117 structures, re-implemented \
                                 natively in Rust — runs locally on CPU or GPU)",
                            )
                            .clicked()
                        {
                            open_auto = true;
                        }
                        ui.weak("drawn with 🖌 / ✨ in the views");
                    });
                    if let Some((msg, frac)) = &auto_state {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.add(
                                egui::ProgressBar::new(*frac)
                                    .desired_width(120.0)
                                    .show_percentage(),
                            );
                            if ui.small_button("Cancel").clicked() {
                                cancel_auto = true;
                            }
                        });
                        ui.weak(msg);
                    }
                    for (i, seg) in segs.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.color_edit_button_srgb(&mut seg.color);
                            ui.checkbox(&mut seg.visible, "")
                                .on_hover_text("Show / hide this segmentation");
                            let resp = ui
                                .selectable_label(i == *active_seg, &seg.name)
                                .on_hover_text(
                                    "Click to make this the segmentation the tools edit",
                                );
                            if resp.clicked() {
                                *active_seg = i;
                            }
                            ui.weak(format!("{:.1} cm³", seg.volume_cm3(spacing)));
                            if ui
                                .add_enabled(seg.can_undo(), egui::Button::new("↶").small())
                                .on_hover_text("Undo the last stroke (Ctrl+Z)")
                                .clicked()
                            {
                                seg.undo_last();
                            }
                            if ui
                                .small_button("→RS")
                                .on_hover_text(
                                    "Convert to RTSTRUCT contours: adds a ROI to the \
                                     structure set, so it exports with \
                                     File ▶ 💾 Export",
                                )
                                .clicked()
                            {
                                to_struct = Some(i);
                            }
                            if ui
                                .small_button("🗑")
                                .on_hover_text("Delete this segmentation")
                                .clicked()
                            {
                                delete = Some(i);
                            }
                        });
                    }
                });
        }
        if make_new {
            self.create_seg(slot);
        }
        if open_auto {
            self.open_autoseg_dialog(slot);
        }
        if cancel_auto {
            if let Some(job) = &self.autoseg_job {
                job.progress.cancel();
            }
        }
        if let Some(i) = delete {
            let s = &mut self.slots[slot];
            if i < s.segs.len() {
                s.segs.remove(i);
                if s.active_seg >= s.segs.len() {
                    s.active_seg = s.segs.len().saturating_sub(1);
                }
            }
        }
        if let Some(i) = to_struct {
            self.seg_to_rtstruct(slot, i);
        }
    }

    pub(super) fn dose_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n_doses = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.doses.len())
            .unwrap_or(0);
        if n_doses == 0 {
            // No RTDOSE in this study — show nothing.
            return;
        }
        let mut mode = self.dose_mode;
        let mut opacity = self.dose_opacity;
        let mut threshold = self.dose_threshold_pct;
        {
            let StudySlot {
                study,
                active_dose,
                dose_reference,
                ..
            } = &mut self.slots[slot];
            let doses = &study.as_ref().unwrap().doses;
            let plans = &study.as_ref().unwrap().plans;
            egui::CollapsingHeader::new("Dose")
                .id_salt(("dose", slot))
                .default_open(true)
                .show(ui, |ui| {
                    if doses.len() > 1 {
                        let mut sel = (*active_dose).min(doses.len() - 1);
                        egui::ComboBox::from_id_salt(("dose_sel", slot))
                            .width(230.0)
                            .selected_text(&doses[sel].label)
                            .show_ui(ui, |ui| {
                                for (i, d) in doses.iter().enumerate() {
                                    ui.selectable_value(&mut sel, i, &d.label);
                                }
                            });
                        *active_dose = sel;
                    }
                    let d = &doses[(*active_dose).min(doses.len() - 1)];
                    ui.weak(format!(
                        "{}  max {:.2} {}",
                        d.summation_type,
                        d.max_dose,
                        d.units.to_lowercase()
                    ));
                    // DICOM cross-reference: which plan this dose belongs to.
                    if !d.referenced_plan_uid.is_empty() {
                        if let Some(p) = plans
                            .iter()
                            .find(|p| p.sop_instance_uid == d.referenced_plan_uid)
                        {
                            ui.weak(format!(
                                "▶ plan {}",
                                if p.label.is_empty() {
                                    "unnamed"
                                } else {
                                    &p.label
                                }
                            ));
                        }
                    }

                    egui::ComboBox::from_id_salt(("dose_mode", slot))
                        .selected_text(mode.label())
                        .show_ui(ui, |ui| {
                            for m in [
                                DoseMode::Off,
                                DoseMode::Colorwash,
                                DoseMode::Isodose,
                                DoseMode::Both,
                            ] {
                                ui.selectable_value(&mut mode, m, m.label());
                            }
                        });

                    ui.horizontal(|ui| {
                        ui.label("Reference");
                        ui.add(
                            egui::DragValue::new(dose_reference)
                                .speed(0.05)
                                .range(0.01..=1000.0)
                                .suffix(" Gy"),
                        );
                        if ui.small_button("max").clicked() {
                            *dose_reference = d.max_dose;
                        }
                    });
                    ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"));
                    ui.add(egui::Slider::new(&mut threshold, 0.0..=100.0).text("Threshold %"));
                });
        }
        self.dose_mode = mode;
        self.dose_opacity = opacity;
        self.dose_threshold_pct = threshold;

        // Isodose levels are shared; show them once (under the first slot
        // that has dose).
        let first_dose_slot = (0..2).find(|&s| {
            self.slots[s]
                .study
                .as_ref()
                .is_some_and(|st| !st.doses.is_empty())
        });
        if first_dose_slot == Some(slot) {
            egui::CollapsingHeader::new("Isodose levels (% of reference)")
                .id_salt("iso_levels")
                .default_open(true)
                .show(ui, |ui| {
                    for l in &mut self.iso_levels {
                        ui.horizontal(|ui| {
                            let (rect, _) =
                                ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                            ui.painter().rect_filled(rect, 2.0, l.color);
                            ui.checkbox(&mut l.on, format!("{:.0}%", l.pct));
                        });
                    }
                });
        }
    }

    pub(super) fn plan_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if study.plans.is_empty() {
            // No RTPLAN in this study — show nothing.
            return;
        }
        for (pi, plan) in study.plans.iter().enumerate() {
            egui::CollapsingHeader::new(format!(
                "Plan: {}",
                if plan.label.is_empty() {
                    "unnamed"
                } else {
                    &plan.label
                }
            ))
            .id_salt(("plan", slot, pi))
            .default_open(pi == 0)
            .show(ui, |ui| {
                if !plan.name.is_empty() && plan.name != plan.label {
                    ui.weak(format!("Name: {}", plan.name));
                }
                if !plan.plan_kind.is_empty() {
                    ui.weak(format!("Type: {}", plan.plan_kind));
                }
                if let Some(fx) = plan.n_fractions {
                    ui.weak(format!("Fractions: {fx}"));
                }
                if let Some(rx) = plan.target_prescription_dose {
                    ui.weak(format!("Prescription: {rx:.2} Gy"));
                }
                if !plan.date.is_empty() {
                    ui.weak(format!("Date: {}", plan.date));
                }
                // DICOM cross-reference: the structure set the plan was
                // created on.
                if !plan.referenced_structset_uid.is_empty() {
                    if let Some(ss) = study
                        .structure_sets
                        .iter()
                        .find(|s| s.sop_instance_uid == plan.referenced_structset_uid)
                    {
                        ui.weak(format!(
                            "▶ structures {}",
                            if ss.label.is_empty() {
                                &ss.file_name
                            } else {
                                &ss.label
                            }
                        ));
                    }
                }
                if !plan.beams.is_empty() {
                    egui::Grid::new(("beam_grid", slot, pi))
                        .striped(true)
                        .min_col_width(10.0)
                        .show(ui, |ui| {
                            ui.strong("Beam");
                            ui.strong("Type");
                            ui.strong("G°");
                            ui.strong("C°");
                            ui.strong("E (MeV)");
                            ui.strong("MU");
                            ui.strong("CPs");
                            ui.end_row();
                            for b in &plan.beams {
                                ui.label(&b.name).on_hover_text(format!(
                                    "Beam {} · {} · dose/fx {}",
                                    b.number,
                                    if b.delivery_type.is_empty() {
                                        "TREATMENT"
                                    } else {
                                        &b.delivery_type
                                    },
                                    b.beam_dose
                                        .map(|d| format!("{d:.2} Gy"))
                                        .unwrap_or_else(|| "n/a".into()),
                                ));
                                ui.label(format!(
                                    "{}{}",
                                    b.radiation_type,
                                    if b.scan_mode.is_empty() {
                                        String::new()
                                    } else {
                                        format!("/{}", b.scan_mode)
                                    }
                                ));
                                ui.label(
                                    b.gantry_angle
                                        .map(|g| format!("{g:.0}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(
                                    b.couch_angle
                                        .map(|c| format!("{c:.0}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(match (b.energy_min, b.energy_max) {
                                    (Some(a), Some(bb)) if (a - bb).abs() > 0.01 => {
                                        format!("{a:.0}–{bb:.0}")
                                    }
                                    (Some(a), _) => format!("{a:.0}"),
                                    _ => "–".into(),
                                });
                                ui.label(
                                    b.meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(format!("{}", b.n_control_points));
                                ui.end_row();
                            }
                        });
                }
            });
        }
    }

    /// DX / CR / RTIMAGE planar images: list with per-image viewer windows.
    pub(super) fn planar_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.planar_images.len())
            .unwrap_or(0);
        if n == 0 {
            return;
        }
        let mut open_idx = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            egui::CollapsingHeader::new(format!("Planar images ({n})"))
                .id_salt(("planar", slot))
                .default_open(false)
                .show(ui, |ui| {
                    for (i, img) in study.planar_images.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("[{}]", img.modality)).weak());
                            ui.label(&img.label);
                            if ui.small_button("View").clicked() {
                                open_idx = Some(i);
                            }
                        });
                    }
                });
        }
        if let Some(i) = open_idx {
            if let Some(w) = self
                .planar_windows
                .iter_mut()
                .find(|w| w.slot == slot && w.idx == i)
            {
                w.open = true;
            } else {
                let wl = self.slots[slot].study.as_ref().unwrap().planar_images[i].window;
                self.planar_windows.push(PlanarWindow {
                    slot,
                    idx: i,
                    open: true,
                    wl,
                    tex: None,
                    tex_wl: (f32::NAN, f32::NAN),
                });
            }
        }
    }

    /// REG spatial registration objects: matrices + apply as active
    /// registration.
    pub(super) fn reg_objects_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.registrations.len())
            .unwrap_or(0);
        if n == 0 {
            return;
        }
        let both = self.slots[0].study.is_some() && self.slots[1].study.is_some();
        let mut apply: Option<(registration::RigidTransform, usize)> = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            // Frame-of-reference UIDs of the loaded volumes for hints.
            let for_a = self.slots[0]
                .study
                .as_ref()
                .map(|s| s.volume.frame_of_reference_uid.clone())
                .unwrap_or_default();
            let for_b = self.slots[1]
                .study
                .as_ref()
                .map(|s| s.volume.frame_of_reference_uid.clone())
                .unwrap_or_default();
            let mut invert = self.reg_apply_invert;
            egui::CollapsingHeader::new(format!("Spatial registrations ({n})"))
                .id_salt(("regobj", slot))
                .default_open(false)
                .show(ui, |ui| {
                    for (ri, reg) in study.registrations.iter().enumerate() {
                        ui.label(
                            egui::RichText::new(format!(
                                "{}{}",
                                reg.label,
                                if reg.deformable { "  [deformable: matrices only]" } else { "" }
                            ))
                            .strong(),
                        );
                        for (ii, item) in reg.items.iter().enumerate() {
                            if item.is_identity {
                                ui.weak(format!("· item {}: identity ({})", ii + 1, item.matrix_type));
                                continue;
                            }
                            let m = &item.matrix;
                            ui.weak(format!("· item {}: {}", ii + 1, item.matrix_type));
                            for r in 0..3 {
                                ui.monospace(format!(
                                    "  [{:7.3} {:7.3} {:7.3} {:8.2}]",
                                    m[r * 4],
                                    m[r * 4 + 1],
                                    m[r * 4 + 2],
                                    m[r * 4 + 3]
                                ));
                            }
                            // FoR hints against loaded studies.
                            let src_hint = if !for_a.is_empty() && item.for_uid == for_a {
                                " (= A)"
                            } else if !for_b.is_empty() && item.for_uid == for_b {
                                " (= B)"
                            } else {
                                ""
                            };
                            let dst_hint = if !for_a.is_empty()
                                && reg.frame_of_reference_uid == for_a
                            {
                                " (= A)"
                            } else if !for_b.is_empty() && reg.frame_of_reference_uid == for_b {
                                " (= B)"
                            } else {
                                ""
                            };
                            ui.weak(format!(
                                "  maps FoR …{}{} ▶ …{}{}",
                                tail(&item.for_uid),
                                src_hint,
                                tail(&reg.frame_of_reference_uid),
                                dst_hint
                            ));
                            match extras::matrix_to_rigid(m, invert) {
                                Some(rigid) => {
                                    let rp = rigid.params();
                                    ui.weak(format!(
                                        "  t = ({:.1}, {:.1}, {:.1}) mm  r = ({:.2}, {:.2}, {:.2})°{}",
                                        rp[3],
                                        rp[4],
                                        rp[5],
                                        rp[0].to_degrees(),
                                        rp[1].to_degrees(),
                                        rp[2].to_degrees(),
                                        if invert { "  (inverted)" } else { "" }
                                    ));
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut invert, "Invert")
                                            .on_hover_text(
                                                "Invert the matrix before applying (flip the mapping direction)",
                                            );
                                        if ui
                                            .add_enabled(
                                                both,
                                                egui::Button::new("Apply as B ▶ A"),
                                            )
                                            .on_hover_text(
                                                "Use this matrix as the transform mapping A (fixed) coordinates into B (moving)",
                                            )
                                            .clicked()
                                        {
                                            if let Some(r2) =
                                                extras::matrix_to_rigid(m, invert)
                                            {
                                                apply = Some((r2, 0));
                                            }
                                        }
                                        if ui
                                            .add_enabled(
                                                both,
                                                egui::Button::new("Apply as A ▶ B"),
                                            )
                                            .on_hover_text(
                                                "Use this matrix as the transform mapping B (fixed) coordinates into A (moving)",
                                            )
                                            .clicked()
                                        {
                                            if let Some(r2) =
                                                extras::matrix_to_rigid(m, invert)
                                            {
                                                apply = Some((r2, 1));
                                            }
                                        }
                                    });
                                }
                                None => {
                                    ui.weak("  (matrix is not a pure rigid transform — cannot apply)");
                                }
                            }
                        }
                        if ri + 1 < study.registrations.len() {
                            ui.separator();
                        }
                    }
                });
            self.reg_apply_invert = invert;
        }
        if let Some((rigid, fixed_slot)) = apply {
            self.apply_external_rigid(rigid, fixed_slot);
        }
    }

    /// RT (Ion) Beams Treatment Records: per-beam delivered metersets.
    pub(super) fn records_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if study.treat_records.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(format!("Treatment records ({})", study.treat_records.len()))
            .id_salt(("records", slot))
            .default_open(false)
            .show(ui, |ui| {
                for (ri, rec) in study.treat_records.iter().enumerate() {
                    ui.label(
                        egui::RichText::new(format!(
                            "{}{}{}{}",
                            rec.label,
                            if rec.ion { "  [ion]" } else { "" },
                            rec.fraction
                                .map(|f| format!("  fx {f}"))
                                .unwrap_or_default(),
                            if rec.date.is_empty() {
                                String::new()
                            } else {
                                format!("  {}", rec.date)
                            }
                        ))
                        .strong(),
                    );
                    if !rec.machine.is_empty() {
                        ui.weak(format!("Machine: {}", rec.machine));
                    }
                    egui::Grid::new(("rec_grid", slot, ri))
                        .striped(true)
                        .min_col_width(10.0)
                        .show(ui, |ui| {
                            ui.strong("Beam");
                            ui.strong("MU spec");
                            ui.strong("MU del");
                            ui.strong("Δ%");
                            ui.strong("Status");
                            ui.end_row();
                            for b in &rec.beams {
                                ui.label(&b.name).on_hover_text(format!(
                                    "Beam {} · verification: {}",
                                    b.number,
                                    if b.verification_status.is_empty() {
                                        "n/a"
                                    } else {
                                        &b.verification_status
                                    }
                                ));
                                ui.label(
                                    b.specified_meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(
                                    b.delivered_meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "–".into()),
                                );
                                ui.label(match (b.specified_meterset, b.delivered_meterset) {
                                    (Some(s), Some(d)) if s > 1e-9 => {
                                        format!("{:+.1}", 100.0 * (d - s) / s)
                                    }
                                    _ => "–".into(),
                                });
                                let status = if b.termination_status.is_empty() {
                                    "–"
                                } else {
                                    &b.termination_status
                                };
                                if status == "NORMAL" || status == "–" {
                                    ui.label(status);
                                } else {
                                    let c = alert_color(ui.visuals());
                                    ui.label(egui::RichText::new(status).color(c));
                                }
                                ui.end_row();
                            }
                        });
                }
            });
    }

    pub(super) fn warnings_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if study.warnings.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(
            egui::RichText::new(format!("⚠ Warnings ({})", study.warnings.len()))
                .color(warn_color(ui.visuals())),
        )
        .id_salt(("warn", slot))
        .show(ui, |ui| {
            for w in &study.warnings {
                ui.label(egui::RichText::new(w).small());
            }
        });
    }
}
