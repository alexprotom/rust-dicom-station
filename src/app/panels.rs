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
        let mut rename: Option<RenameTarget> = None;
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
                                            if ui.button("✎ Rename series…").clicked() {
                                                rename =
                                                    Some(RenameTarget::Series { slot, idx: i });
                                                ui.close();
                                            }
                                            ui.separator();
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
                                            "Series UID …{}\nright-click: rename, copy / move \
                                             to dataset {other}, or remove",
                                            tail(&s.uid)
                                        ));
                                    }
                                });
                            sch.header_response.context_menu(|ui| {
                                if ui.button("✎ Rename study…").clicked() {
                                    rename = Some(RenameTarget::Study {
                                        slot,
                                        uid: study_uid.to_string(),
                                    });
                                    ui.close();
                                }
                                ui.separator();
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
                    if ui.button("✎ Rename patient…").clicked() {
                        rename = Some(RenameTarget::Patient {
                            slot,
                            key: pkey.to_string(),
                        });
                        ui.close();
                    }
                    ui.separator();
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
        if rename.is_some() {
            self.rename_request = rename;
        }
        if let Some(i) = switch_to {
            self.start_series_switch(slot, i);
        }
    }

    // -- Structure sets and segmentation series ----------------------------

    /// Right-click menu of a series node: what image series it is drawn on,
    /// where it goes, and whether it stays.
    fn set_context_menu(&self, ui: &mut egui::Ui, here: SetRef, out: &mut Option<SetAction>) {
        let other = SLOT_NAMES[1 - here.slot];
        if ui.button("✎ Rename series…").clicked() {
            *out = Some(SetAction::Rename(here));
            ui.close();
        }
        ui.separator();
        ui.menu_button("🔗 Connect to image series", |ui| {
            let Some(study) = self.slots[here.slot].study.as_ref() else {
                return;
            };
            let current = match here.kind {
                SetKind::Structures => study
                    .structure_sets
                    .get(here.idx)
                    .map(|s| s.referenced_series_uid.clone()),
                SetKind::Segmentations => study
                    .seg_series
                    .get(here.idx)
                    .map(|s| s.referenced_series_uid.clone()),
            }
            .unwrap_or_default();
            for se in &study.series {
                let label = format!(
                    "{} {} {} ({} sl.)",
                    if se.uid == current { "●" } else { "  " },
                    se.modality,
                    if se.description.is_empty() {
                        "series"
                    } else {
                        &se.description
                    },
                    se.files.len()
                );
                if ui.button(label).clicked() {
                    *out = Some(SetAction::Connect(here, se.uid.clone()));
                    ui.close();
                }
            }
        });
        ui.separator();
        if ui
            .button(format!("Copy series to dataset {other}"))
            .clicked()
        {
            *out = Some(SetAction::Transfer {
                from: here,
                copy: true,
            });
            ui.close();
        }
        if ui
            .button(format!("Move series to dataset {other}"))
            .clicked()
        {
            *out = Some(SetAction::Transfer {
                from: here,
                copy: false,
            });
            ui.close();
        }
        if here.kind == SetKind::Segmentations {
            ui.separator();
            if ui
                .button("💾 Export as DICOM SEG…")
                .on_hover_text("Write this series as one DICOM Segmentation file")
                .clicked()
            {
                *out = Some(SetAction::ExportSeg(here));
                ui.close();
            }
        }
        ui.separator();
        if ui
            .button(format!("🗑 Remove this {}", here.kind.series_name()))
            .clicked()
        {
            *out = Some(SetAction::Remove(here));
            ui.close();
        }
    }

    /// Every structure set and segmentation series of both datasets, as the
    /// destinations of a *Copy to ▶* / *Move to ▶* submenu — plus the two
    /// "make me a new one" entries, so a transfer never needs preparing.
    fn destination_menu(&self, ui: &mut egui::Ui, from: SetRef) -> Option<SetRef> {
        let mut picked = None;
        for (slot, slot_name) in SLOT_NAMES.iter().enumerate() {
            let Some(study) = self.slots[slot].study.as_ref() else {
                continue;
            };
            ui.label(egui::RichText::new(format!("Dataset {slot_name}")).strong());
            for (i, ss) in study.structure_sets.iter().enumerate() {
                let here = SetRef {
                    slot,
                    kind: SetKind::Structures,
                    idx: i,
                };
                if here == from {
                    continue;
                }
                let name = if ss.label.is_empty() {
                    &ss.file_name
                } else {
                    &ss.label
                };
                if ui
                    .button(format!("▣ {name} ({} ROIs)", ss.rois.len()))
                    .clicked()
                {
                    picked = Some(here);
                    ui.close();
                }
            }
            for (i, sr) in study.seg_series.iter().enumerate() {
                let here = SetRef {
                    slot,
                    kind: SetKind::Segmentations,
                    idx: i,
                };
                if here == from {
                    continue;
                }
                if ui
                    .button(format!("✎ {} ({} segments)", sr.label, sr.segs.len()))
                    .clicked()
                {
                    picked = Some(here);
                    ui.close();
                }
            }
            if ui.button("▣ ➕ a new RT structure set").clicked() {
                picked = Some(SetRef {
                    slot,
                    kind: SetKind::Structures,
                    idx: SetRef::NEW,
                });
                ui.close();
            }
            if ui.button("✎ ➕ a new segmentation series").clicked() {
                picked = Some(SetRef {
                    slot,
                    kind: SetKind::Segmentations,
                    idx: SetRef::NEW,
                });
                ui.close();
            }
            ui.separator();
        }
        picked
    }

    /// Right-click menu of one structure / segment. `selection` is what is
    /// ticked in the list: right-clicking a ticked row acts on all of them,
    /// right-clicking an unticked one acts on that row alone.
    fn item_context_menu(
        &self,
        ui: &mut egui::Ui,
        from: SetRef,
        clicked: usize,
        label: &str,
        selection: &[usize],
        out: &mut Option<ItemAction>,
    ) {
        let items: Vec<usize> = if selection.len() > 1 && selection.contains(&clicked) {
            selection.to_vec()
        } else {
            vec![clicked]
        };
        let what = if items.len() > 1 {
            format!(
                "the {} ticked {}",
                items.len(),
                from.kind.item_name(items.len())
            )
        } else {
            format!("'{label}'")
        };
        if ui.button(format!("✎ Rename '{label}'…")).clicked() {
            *out = Some(ItemAction::Rename { from, idx: clicked });
            ui.close();
        }
        ui.separator();
        ui.menu_button(format!("Copy {what} to"), |ui| {
            if let Some(to) = self.destination_menu(ui, from) {
                *out = Some(ItemAction::Transfer {
                    from,
                    items: items.clone(),
                    to,
                    copy: true,
                });
            }
        });
        ui.menu_button(format!("Move {what} to"), |ui| {
            if let Some(to) = self.destination_menu(ui, from) {
                *out = Some(ItemAction::Transfer {
                    from,
                    items: items.clone(),
                    to,
                    copy: false,
                });
            }
        });
        ui.separator();
        if ui.button(format!("🗑 Remove {what}")).clicked() {
            *out = Some(ItemAction::Remove {
                from,
                items: items.clone(),
            });
            ui.close();
        }
    }

    /// The image series a set is drawn on, as a tree suffix.
    fn series_suffix(study: &LoadedStudy, uid: &str) -> String {
        study
            .series
            .iter()
            .find(|se| se.uid == uid)
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
            .unwrap_or_else(|| " ▶ (unlinked)".to_string())
    }

    /// RT structure sets: one node per set, then the ROIs of the active one.
    ///
    /// A ROI's check box is both its visibility and its selection, so *All* /
    /// *None* tick everything or nothing and the right-click actions operate
    /// on whatever is ticked.
    pub(super) fn structures_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        if self.slots[slot].study.is_none() {
            return;
        }
        // Rendering runs behind a shared borrow of `self` (the context menus
        // need to list the other dataset's series), so the one piece of
        // mutable state in the list is edited on a copy and written back.
        let mut vis = std::mem::take(&mut self.slots[slot].roi_visible);
        let mut new_active: Option<usize> = None;
        let mut set_act: Option<SetAction> = None;
        let mut item_act: Option<ItemAction> = None;
        {
            let me = &*self;
            let study = me.slots[slot].study.as_ref().unwrap();
            let sets = &study.structure_sets;
            let active_set = me.slots[slot]
                .active_structs
                .min(sets.len().saturating_sub(1));
            let n_rois = sets.get(active_set).map(|ss| ss.rois.len()).unwrap_or(0);
            vis.resize(n_rois, true);
            let n_vis = vis.iter().filter(|v| **v).count();
            egui::CollapsingHeader::new(format!("RT structures ({n_vis}/{n_rois})"))
                .id_salt(("structs", slot))
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("➕ New series")
                            .on_hover_text(
                                "An empty RT structure set, drawn on the displayed image series",
                            )
                            .clicked()
                        {
                            set_act = Some(SetAction::New(SetRef {
                                slot,
                                kind: SetKind::Structures,
                                idx: SetRef::NEW,
                            }));
                        }
                        ui.weak(match sets.len() {
                            0 => "no structure sets".to_string(),
                            1 => "1 series".to_string(),
                            n => format!("{n} series"),
                        });
                    });
                    for (i, set) in sets.iter().enumerate() {
                        let here = SetRef {
                            slot,
                            kind: SetKind::Structures,
                            idx: i,
                        };
                        let name = if set.label.is_empty() {
                            &set.file_name
                        } else {
                            &set.label
                        };
                        let resp = ui.selectable_label(
                            i == active_set,
                            format!(
                                "▣ {name} ({} ROIs){}",
                                set.rois.len(),
                                Self::series_suffix(study, &set.referenced_series_uid)
                            ),
                        );
                        if resp.clicked() && i != active_set {
                            new_active = Some(i);
                        }
                        resp.context_menu(|ui| me.set_context_menu(ui, here, &mut set_act));
                        resp.on_hover_text(format!(
                            "{}\nreferences series …{}\nright-click: connect to another image \
                             series, copy / move to the other dataset, remove",
                            if set.file_name.is_empty() {
                                "created here"
                            } else {
                                &set.file_name
                            },
                            tail(&set.referenced_series_uid)
                        ));
                    }
                    let Some(ss) = sets.get(active_set) else {
                        return;
                    };
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("All")
                            .on_hover_text("Show and select every structure")
                            .clicked()
                        {
                            vis.iter_mut().for_each(|v| *v = true);
                        }
                        if ui.small_button("None").clicked() {
                            vis.iter_mut().for_each(|v| *v = false);
                        }
                        ui.weak(&ss.label);
                    });
                    let selection: Vec<usize> = vis
                        .iter()
                        .enumerate()
                        .filter(|(_, v)| **v)
                        .map(|(i, _)| i)
                        .collect();
                    let here = SetRef {
                        slot,
                        kind: SetKind::Structures,
                        idx: active_set,
                    };
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
                                &mut vis[i],
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
                            resp.context_menu(|ui| {
                                me.item_context_menu(
                                    ui,
                                    here,
                                    i,
                                    &roi.name,
                                    &selection,
                                    &mut item_act,
                                )
                            });
                            resp.on_hover_text(format!(
                                "ROI {} · {} contour(s)\nright-click: copy / move / remove — \
                                 every ticked structure at once",
                                roi.number,
                                roi.contours.len()
                            ));
                        });
                    }
                });
        }
        self.slots[slot].roi_visible = vis;
        if let Some(i) = new_active {
            let s = &mut self.slots[slot];
            s.active_structs = i;
            let n = s
                .study
                .as_ref()
                .map(|st| st.structure_sets[i].rois.len())
                .unwrap_or(0);
            s.roi_visible = vec![true; n];
        }
        if set_act.is_some() {
            self.set_action = set_act;
        }
        if item_act.is_some() {
            self.item_action = item_act;
        }
    }

    /// Segmentation series: one node per series, then the segments of the
    /// active one with the tools that edit them.
    pub(super) fn segmentation_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        if self.slots[slot].study.is_none() {
            return;
        }
        // Whichever engine is running on this slot: its glyph, message and
        // fraction, read before the section borrows anything.
        let running = self
            .running_tool(slot)
            .map(|(tool, p)| (tool.glyph, p.get(), p.frac()));
        // (name, colour, visible, cm³, can undo) of the active series'
        // segments — the editable columns live on this copy, see
        // `structures_section`.
        let mut rows: Vec<(String, [u8; 3], bool, f64, bool)> = {
            let s = &self.slots[slot];
            let spacing = s
                .study
                .as_ref()
                .map(|st| st.volume.spacing)
                .unwrap_or([1.0; 3]);
            s.segs()
                .iter()
                .map(|g| {
                    (
                        g.name.clone(),
                        g.color,
                        g.visible,
                        g.volume_cm3(spacing),
                        g.can_undo(),
                    )
                })
                .collect()
        };
        let before: Vec<([u8; 3], bool)> = rows.iter().map(|r| (r.1, r.2)).collect();
        let mut make_new = false;
        let mut new_series = false;
        let mut open_tool: Option<&ToolInfo> = None;
        let mut cancel_tool = false;
        let mut set_all: Option<bool> = None;
        let mut new_active_series: Option<usize> = None;
        let mut activate: Option<usize> = None;
        let mut undo: Option<usize> = None;
        let mut delete: Option<usize> = None;
        let mut to_struct: Option<usize> = None;
        let mut set_act: Option<SetAction> = None;
        let mut item_act: Option<ItemAction> = None;
        {
            let me = &*self;
            let study = me.slots[slot].study.as_ref().unwrap();
            let series = &study.seg_series;
            let active_series = me.slots[slot].seg_series_idx();
            let active_seg = me.slots[slot].active_seg;
            let n_vis = rows.iter().filter(|r| r.2).count();
            let n_segs = active_series.map(|i| series[i].segs.len()).unwrap_or(0);
            egui::CollapsingHeader::new(format!("Segmentations ({n_vis}/{n_segs})"))
                .id_salt(("segs", slot))
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("➕ New series")
                            .on_hover_text(
                                "An empty segmentation series, drawn on the displayed image \
                                 series — exports as one DICOM SEG file",
                            )
                            .clicked()
                        {
                            new_series = true;
                        }
                        for (tool, hint) in [
                            (
                                &AUTOSEG,
                                "Automatic multi-organ segmentation (TotalSegmentator, \
                                 117 structures)",
                            ),
                            (
                                &PROMPT_SEG,
                                "Segment whatever the crosshair points at — a box, a click \
                                 or a structure name (SegVol)",
                            ),
                            (
                                &SLICE_PROP,
                                "Box a structure on one slice and follow it through the \
                                 stack (MedSAM2)",
                            ),
                        ] {
                            if ui
                                .add(egui::Button::new(tool.short_button()).small())
                                .on_hover_text(hint)
                                .clicked()
                            {
                                open_tool = Some(tool);
                            }
                        }
                    });
                    for (i, sr) in series.iter().enumerate() {
                        let here = SetRef {
                            slot,
                            kind: SetKind::Segmentations,
                            idx: i,
                        };
                        let resp = ui.selectable_label(
                            Some(i) == active_series,
                            format!(
                                "✎ {} ({} segments){}",
                                sr.label,
                                sr.segs.len(),
                                Self::series_suffix(study, &sr.referenced_series_uid)
                            ),
                        );
                        if resp.clicked() && Some(i) != active_series {
                            new_active_series = Some(i);
                        }
                        resp.context_menu(|ui| me.set_context_menu(ui, here, &mut set_act));
                        resp.on_hover_text(format!(
                            "{}\nright-click: connect to another image series, copy / move \
                             to the other dataset, export as DICOM SEG, remove",
                            if sr.file_name.is_empty() {
                                "created here"
                            } else {
                                &sr.file_name
                            }
                        ));
                    }
                    let Some(active_series) = active_series else {
                        ui.weak("no segmentation series yet — ➕ New series, or just paint");
                        return;
                    };
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .small_button("➕ New")
                            .on_hover_text(
                                "An empty segmentation to paint with 🖌 / ✨ in the views",
                            )
                            .clicked()
                        {
                            make_new = true;
                        }
                        if ui
                            .small_button("All")
                            .on_hover_text("Show and select every segmentation")
                            .clicked()
                        {
                            set_all = Some(true);
                        }
                        if ui.small_button("None").clicked() {
                            set_all = Some(false);
                        }
                        ui.weak(&series[active_series].label);
                    });
                    // Masks of a series drawn on another image series are on
                    // that series' lattice — nothing here can index them.
                    if series[active_series].grid.dims != study.volume.dims {
                        ui.weak(
                            "drawn on another image series — display that series to see and \
                             edit these segments",
                        );
                        return;
                    }
                    if let Some((glyph, msg, frac)) = &running {
                        ui.horizontal(|ui| {
                            ui.label(*glyph);
                            ui.add(
                                egui::ProgressBar::new(*frac)
                                    .desired_width(120.0)
                                    .show_percentage(),
                            );
                            if ui.small_button("Cancel").clicked() {
                                cancel_tool = true;
                            }
                        });
                        ui.weak(msg);
                    }
                    let selection: Vec<usize> = rows
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.2)
                        .map(|(i, _)| i)
                        .collect();
                    let here = SetRef {
                        slot,
                        kind: SetKind::Segmentations,
                        idx: active_series,
                    };
                    for (i, row) in rows.iter_mut().enumerate() {
                        let name = row.0.clone();
                        ui.horizontal(|ui| {
                            ui.color_edit_button_srgb(&mut row.1);
                            ui.checkbox(&mut row.2, "")
                                .on_hover_text("Show / select this segmentation");
                            let resp = ui.selectable_label(i == active_seg, &name).on_hover_text(
                                "Click to make this the segmentation the tools edit",
                            );
                            if resp.clicked() {
                                activate = Some(i);
                            }
                            resp.context_menu(|ui| {
                                me.item_context_menu(ui, here, i, &name, &selection, &mut item_act)
                            });
                            ui.weak(format!("{:.1} cm³", row.3));
                            if ui
                                .add_enabled(row.4, egui::Button::new("↶").small())
                                .on_hover_text("Undo the last stroke (Ctrl+Z)")
                                .clicked()
                            {
                                undo = Some(i);
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
        if let Some(v) = set_all {
            rows.iter_mut().for_each(|r| r.2 = v);
        }
        let edited: Vec<(usize, [u8; 3], bool)> = rows
            .iter()
            .enumerate()
            .zip(&before)
            .filter(|((_, r), b)| (r.1, r.2) != **b)
            .map(|((i, r), _)| (i, r.1, r.2))
            .collect();
        if !edited.is_empty() {
            if let Some(segs) = self.slots[slot].segs_mut() {
                for (i, color, visible) in edited {
                    if let Some(seg) = segs.get_mut(i) {
                        seg.color = color;
                        seg.visible = visible;
                    }
                }
            }
        }
        if new_series {
            self.new_set(slot, SetKind::Segmentations);
        }
        if let Some(i) = new_active_series {
            let s = &mut self.slots[slot];
            s.active_seg_series = i;
            s.active_seg = 0;
            self.settings_gen += 1;
        }
        if let Some(i) = activate {
            self.slots[slot].active_seg = i;
        }
        if make_new {
            self.create_seg(slot);
        }
        match open_tool.map(|t| t.glyph) {
            Some(g) if g == AUTOSEG.glyph => self.open_autoseg_dialog(slot),
            Some(g) if g == PROMPT_SEG.glyph => self.open_segvol_dialog(slot),
            Some(_) => self.open_medsam2_panel(slot),
            None => {}
        }
        if cancel_tool {
            if let Some((_, p)) = self.running_tool(slot) {
                p.cancel();
            }
        }
        if let Some(i) = undo {
            let s = &mut self.slots[slot];
            if let Some(seg) = s.segs_mut().and_then(|g| g.get_mut(i)) {
                seg.undo_last();
            }
        }
        if let Some(i) = delete {
            let s = &mut self.slots[slot];
            let active = s.active_seg;
            if let Some(segs) = s.segs_mut() {
                if i < segs.len() {
                    segs.remove(i);
                    let n = segs.len();
                    if active >= n {
                        s.active_seg = n.saturating_sub(1);
                    }
                }
            }
        }
        if let Some(i) = to_struct {
            self.seg_to_rtstruct(slot, i);
        }
        if set_act.is_some() {
            self.set_action = set_act;
        }
        if item_act.is_some() {
            self.item_action = item_act;
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
        let mut rename: Option<RenameTarget> = None;
        {
            let StudySlot {
                study,
                active_dose,
                dose_reference,
                ..
            } = &mut self.slots[slot];
            let doses = &study.as_ref().unwrap().doses;
            let plans = &study.as_ref().unwrap().plans;
            let dose_hdr = egui::CollapsingHeader::new("Dose")
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
            let sel = (*active_dose).min(doses.len() - 1);
            dose_hdr.header_response.context_menu(|ui| {
                if ui
                    .button("✎ Rename this dose…")
                    .on_hover_text(format!("Renames '{}'", doses[sel].label))
                    .clicked()
                {
                    rename = Some(RenameTarget::Dose { slot, idx: sel });
                    ui.close();
                }
            });
        }
        if rename.is_some() {
            self.rename_request = rename;
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
        let mut rename: Option<RenameTarget> = None;
        {
            let Some(study) = &self.slots[slot].study else {
                return;
            };
            if study.plans.is_empty() {
                // No RTPLAN in this study — show nothing.
                return;
            }
            for (pi, plan) in study.plans.iter().enumerate() {
                let plan_hdr = egui::CollapsingHeader::new(format!(
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
                plan_hdr.header_response.context_menu(|ui| {
                    if ui.button("✎ Rename plan…").clicked() {
                        rename = Some(RenameTarget::Plan { slot, idx: pi });
                        ui.close();
                    }
                });
            }
        }
        if rename.is_some() {
            self.rename_request = rename;
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
        let mut rename: Option<RenameTarget> = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            egui::CollapsingHeader::new(format!("Planar images ({n})"))
                .id_salt(("planar", slot))
                .default_open(false)
                .show(ui, |ui| {
                    for (i, img) in study.planar_images.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("[{}]", img.modality)).weak());
                            let resp = ui
                                .label(&img.label)
                                .on_hover_text("right-click: rename this image");
                            resp.context_menu(|ui| {
                                if ui.button("✎ Rename image…").clicked() {
                                    rename = Some(RenameTarget::Planar { slot, idx: i });
                                    ui.close();
                                }
                            });
                            if ui.small_button("View").clicked() {
                                open_idx = Some(i);
                            }
                        });
                    }
                });
        }
        if rename.is_some() {
            self.rename_request = rename;
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
        let mut apply_grid: Option<(usize, usize)> = None;
        let mut rename: Option<RenameTarget> = None;
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
                        let resp = ui
                            .label(
                                egui::RichText::new(format!(
                                    "{}{}",
                                    reg.label,
                                    if reg.deformable {
                                        "  [deformable: matrices only]"
                                    } else {
                                        ""
                                    }
                                ))
                                .strong(),
                            )
                            .on_hover_text("right-click: rename this registration");
                        resp.context_menu(|ui| {
                            if ui.button("✎ Rename registration…").clicked() {
                                rename = Some(RenameTarget::Registration { slot, idx: ri });
                                ui.close();
                            }
                        });
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
                        // A Deformable Spatial Registration's displacement
                        // lattice applies exactly as a matrix does.
                        if let Some(grid) = &reg.grid {
                            ui.weak(format!("· deformation grid: {}", grid.describe()));
                            let src_hint = if !for_a.is_empty()
                                && reg.grid_source_for_uid == for_a
                            {
                                Some(0usize)
                            } else if !for_b.is_empty() && reg.grid_source_for_uid == for_b {
                                Some(1usize)
                            } else {
                                None
                            };
                            ui.horizontal(|ui| {
                                for fixed in 0..2 {
                                    let label = format!(
                                        "Apply grid as {} ▶ {}",
                                        SLOT_NAMES[1 - fixed],
                                        SLOT_NAMES[fixed]
                                    );
                                    let hint = match src_hint {
                                        Some(s) if s == fixed => {
                                            "The grid's own frame of reference matches this                                              dataset — this is the direction the file means"
                                        }
                                        Some(_) => {
                                            "The grid's frame of reference matches the *other*                                              dataset; applying it this way round inverts what                                              the file says"
                                        }
                                        None => {
                                            "Neither loaded dataset matches the grid's frame of                                              reference — check that this is the right pair"
                                        }
                                    };
                                    if ui
                                        .add_enabled(both, egui::Button::new(label))
                                        .on_hover_text(hint)
                                        .clicked()
                                    {
                                        apply_grid = Some((ri, fixed));
                                    }
                                }
                            });
                        }
                        if ri + 1 < study.registrations.len() {
                            ui.separator();
                        }
                    }
                });
            self.reg_apply_invert = invert;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if let Some((rigid, fixed_slot)) = apply {
            self.apply_external_rigid(rigid, fixed_slot);
        }
        if let Some((ri, fixed_slot)) = apply_grid {
            if let Some(field) = self.slots[slot]
                .study
                .as_ref()
                .and_then(|s| s.registrations.get(ri))
                .and_then(|r| r.grid.clone())
            {
                let center = field.origin;
                let transform = Transform3 {
                    rigid: registration::RigidTransform::identity(center),
                    warp: registration::Warp::Field(Arc::new(field)),
                };
                self.apply_external_transform(
                    transform,
                    registration::RegMethod::PlastimatchBSpline,
                    fixed_slot,
                );
            }
        }
    }

    /// RT (Ion) Beams Treatment Records: per-beam delivered metersets.
    pub(super) fn records_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let mut rename: Option<RenameTarget> = None;
        {
            let Some(study) = &self.slots[slot].study else {
                return;
            };
            if study.treat_records.is_empty() {
                return;
            }
            egui::CollapsingHeader::new(format!(
                "Treatment records ({})",
                study.treat_records.len()
            ))
            .id_salt(("records", slot))
            .default_open(false)
            .show(ui, |ui| {
                for (ri, rec) in study.treat_records.iter().enumerate() {
                    let resp = ui
                        .label(
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
                        )
                        .on_hover_text("right-click: rename this record");
                    resp.context_menu(|ui| {
                        if ui.button("✎ Rename record…").clicked() {
                            rename = Some(RenameTarget::Record { slot, idx: ri });
                            ui.close();
                        }
                    });
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
        if rename.is_some() {
            self.rename_request = rename;
        }
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
