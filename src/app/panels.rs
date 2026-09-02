//! The side panel and its per-dataset sections.
//!
//! Each section renders one kind of loaded object -- series, structures,
//! segmentations, dose, plan, planar images, registrations, records. The
//! optional registration and simulation sections live in `reg_panel.rs`.

use super::*;

impl ViewerApp {
    // -- Side panel -------------------------------------------------------
    pub(super) fn side_panel(&mut self, ui: &mut egui::Ui) {
        if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
            return;
        }
        // A thin strip along the window edge carries the show / hide arrow.
        // It stays there when the panel is gone — otherwise nothing on
        // screen would say how to get it back (View ▶ Left panel and F9 do
        // the same).
        let strip = egui::Frame::new()
            .fill(ui.visuals().panel_fill)
            .inner_margin(egui::Margin::symmetric(1, 4));
        egui::Panel::left(egui::Id::new("side_toggle"))
            .exact_size(22.0)
            .resizable(false)
            .frame(strip)
            .show(ui, |ui| {
                let (glyph, hint) = if self.side_open {
                    (
                        "◀",
                        "Hide the left panel - the views take the whole window (F9)",
                    )
                } else {
                    ("▶", "Show the left panel (F9)")
                };
                if ui
                    .add(
                        egui::Button::new(glyph)
                            .frame(false)
                            .min_size(egui::vec2(20.0, 22.0)),
                    )
                    .on_hover_text(hint)
                    .clicked()
                {
                    self.side_open = !self.side_open;
                }
            });
        // `show_collapsible` also lets the panel be dragged shut and pulled
        // back open by its edge; it wants the flag by reference, which the
        // body's `&mut self` cannot share, so it travels via a local.
        let mut open = self.side_open;
        egui::Panel::left(egui::Id::new("side"))
            .resizable(true)
            .default_size(280.0)
            .show_collapsible(ui, &mut open, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // The two optional modules (Modules menu) come first,
                    // then one data tree per loaded dataset.
                    if self.module_registration {
                        self.registration_section(ui);
                    }
                    if self.module_simulation {
                        self.simulation_section(ui);
                    }
                    for slot in 0..2 {
                        if self.slots[slot].study.is_none() {
                            continue;
                        }
                        self.study_section(ui, slot);
                    }
                });
            });
        self.side_open = open;
    }

    /// A tree node whose title wraps over as many lines as it needs.
    ///
    /// [`egui::CollapsingHeader`] lays its title out on a single line and
    /// lets it run past the panel edge, so a long patient name, a long
    /// study description or a long ID would pin the panel open at that
    /// width. Only the module headers keep that one-line behaviour.
    ///
    /// Returns the title's own response, so the caller can hang the node's
    /// context menu on it.
    fn wrapped_node<R>(
        ui: &mut egui::Ui,
        id_salt: impl std::hash::Hash + std::fmt::Debug,
        default_open: bool,
        title: impl Into<String>,
        body: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::Response {
        let id = ui.make_persistent_id(id_salt);
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            id,
            default_open,
        );
        let (_, header, _) = state
            .show_header(ui, |ui| {
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(title.into()).text_style(egui::TextStyle::Button),
                    )
                    .wrap()
                    .sense(egui::Sense::click()),
                )
            })
            .body(body);
        let resp = header.inner;
        // Clicking the title opens and closes the node, as it does on a
        // standard header; the arrow keeps working on its own.
        if resp.clicked() {
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                id,
                default_open,
            );
            state.toggle(ui);
            state.store(ui.ctx());
        }
        resp
    }

    /// A tree node whose header carries a tick box in front of the title:
    /// the object it stands for is drawn in the views, or it is not.
    ///
    /// Same wrapping behaviour as [`Self::wrapped_node`]; the tick box is
    /// part of the header row, so it reads as one line.
    fn checked_node<R>(
        ui: &mut egui::Ui,
        id_salt: impl std::hash::Hash + std::fmt::Debug,
        default_open: bool,
        shown: &mut bool,
        title: impl Into<String>,
        body: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::Response {
        let id = ui.make_persistent_id(id_salt);
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            id,
            default_open,
        );
        let title = title.into();
        let (_, header, _) = state
            .show_header(ui, |ui| {
                ui.checkbox(shown, "")
                    .on_hover_text("Show this in the views");
                ui.add(
                    egui::Label::new(
                        egui::RichText::new(title).text_style(egui::TextStyle::Button),
                    )
                    .wrap()
                    .sense(egui::Sense::click()),
                )
            })
            .body(body);
        let resp = header.inner;
        if resp.clicked() {
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                id,
                default_open,
            );
            state.toggle(ui);
            state.store(ui.ctx());
        }
        resp
    }

    /// Study transform simulator: apply a known rigid motion + optional
    /// Gaussian deformation to a study and generate the result into the
    /// other slot (the generated study is exportable via *File ▶ Export*).
    pub(super) fn simulation_section(&mut self, ui: &mut egui::Ui) {
        if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
            return;
        }
        let mut do_generate = false;
        egui::CollapsingHeader::new(egui::RichText::new("Image simulation").strong())
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

                let src_ok = self.slots[self.sim_source.min(1)].has_volume();
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
                // Patient ▶ study ▶ category ▶ series. Everything that
                // carries a StudyInstanceUID lives inside a study node; the
                // rest — planar images have no study link at all, REG objects
                // and records belong to a frame of reference rather than a
                // study, and the dose display settings are shared by both
                // datasets — stays at dataset level below it.
                self.data_tree(ui, slot);
                self.dose_display_section(ui, slot);
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

    /// The DICOM data tree of one dataset: patient ▶ study ▶ category ▶
    /// series, all visible at once.
    ///
    /// The nesting is rendered one level per method rather than as one deep
    /// stack of closures: each level hands the next a fresh `&mut self`
    /// reborrow that ends when the node's body returns, which is what lets a
    /// node three levels down still open a dialog or start a series switch.
    pub(super) fn data_tree(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let layout = tree_layout(study);
        for (pi, patient) in layout.iter().enumerate() {
            let me = &mut *self;
            let resp = Self::wrapped_node(
                ui,
                ("pat_hdr", slot, pi),
                true,
                patient.title.clone(),
                |ui| me.patient_body(ui, slot, pi, patient),
            );
            let other = SLOT_NAMES[1 - slot];
            let key = patient.key.clone();
            let mut act: Option<TreeAction> = None;
            let mut rename = None;
            resp.context_menu(|ui| {
                if ui.button("✏ Rename").clicked() {
                    rename = Some(RenameTarget::Patient {
                        slot,
                        key: key.clone(),
                    });
                    ui.close();
                }
                ui.separator();
                for (label, op) in [
                    (format!("Copy patient to dataset {other}"), TreeOp::Copy),
                    (format!("Move patient to dataset {other}"), TreeOp::Move),
                ] {
                    if ui.button(label).clicked() {
                        act = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Patient(key.clone()),
                            op,
                        });
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("🗑 Remove").clicked() {
                    act = Some(TreeAction {
                        from: slot,
                        sel: TreeSel::Patient(key.clone()),
                        op: TreeOp::Remove,
                    });
                    ui.close();
                }
            });
            if act.is_some() {
                self.tree_action = act;
            }
            if rename.is_some() {
                self.rename_request = rename;
            }
        }
    }

    /// The studies of one patient.
    fn patient_body(&mut self, ui: &mut egui::Ui, slot: usize, pi: usize, patient: &PatientNode) {
        for (si, node) in patient.studies.iter().enumerate() {
            let me = &mut *self;
            let resp = Self::wrapped_node(
                ui,
                ("study_tree", slot, pi, si),
                true,
                node.title.clone(),
                |ui| me.study_body(ui, slot, pi, si, node),
            );
            let other = SLOT_NAMES[1 - slot];
            let uid = node.uid.clone();
            let mut act: Option<TreeAction> = None;
            let mut rename = None;
            resp.context_menu(|ui| {
                if ui.button("✏ Rename").clicked() {
                    rename = Some(RenameTarget::Study {
                        slot,
                        uid: uid.clone(),
                    });
                    ui.close();
                }
                ui.separator();
                for (label, op) in [
                    (format!("Copy study to dataset {other}"), TreeOp::Copy),
                    (format!("Move study to dataset {other}"), TreeOp::Move),
                ] {
                    if ui.button(label).clicked() {
                        act = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Study(uid.clone()),
                            op,
                        });
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("🗑 Remove").clicked() {
                    act = Some(TreeAction {
                        from: slot,
                        sel: TreeSel::Study(uid.clone()),
                        op: TreeOp::Remove,
                    });
                    ui.close();
                }
            });
            if act.is_some() {
                self.tree_action = act;
            }
            if rename.is_some() {
                self.rename_request = rename;
            }
        }
    }

    /// The categories of one study: the image modalities it was acquired in,
    /// then the RT objects filed against it.
    fn study_body(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        pi: usize,
        si: usize,
        node: &StudyNode,
    ) {
        // A study opens showing the images being looked at and the
        // structures drawn on them. Everything else - dose, plans, planar
        // images, registrations, records - is a click away, so a tree of a
        // real study is readable the moment it appears.
        let active = self.slots[slot]
            .study
            .as_ref()
            .map(|st| st.active_series)
            .unwrap_or(0);
        for (mi, (modality, idxs)) in node.modalities.iter().enumerate() {
            let me = &mut *self;
            let title = format!("{modality} ({})", idxs.len());
            let showing = idxs.contains(&active);
            Self::wrapped_node(ui, ("mod", slot, pi, si, mi), showing, title, |ui| {
                me.series_rows(ui, slot, idxs)
            });
        }
        for &gi in &node.fourd {
            self.fourd_node(ui, slot, pi, si, gi);
        }
        self.structures_section(ui, slot, pi, si, &node.structs);
        self.segmentation_section(ui, slot, pi, si, &node.segs);
        self.dose_section(ui, slot, pi, si, &node.doses);
        self.plan_section(ui, slot, pi, si, &node.plans);
    }

    /// The image series of one modality node.
    fn series_rows(&mut self, ui: &mut egui::Ui, slot: usize, idxs: &[usize]) {
        let other = SLOT_NAMES[1 - slot];
        let mut switch_to = None;
        let mut act: Option<TreeAction> = None;
        let mut rename = None;
        let mut fourd: Option<FourDAction> = None;
        let group_names: Vec<(usize, String)> = self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                st.fourd_groups
                    .iter()
                    .enumerate()
                    .map(|(gi, g)| (gi, g.name.clone()))
                    .collect()
            })
            .unwrap_or_default();
        {
            let Some(study) = self.slots[slot].study.as_ref() else {
                return;
            };
            let active = study.active_series;
            for &i in idxs {
                let Some(s) = study.series.get(i) else {
                    continue;
                };
                let label = format!(
                    "{} ({} sl.)",
                    if s.description.is_empty() {
                        "series"
                    } else {
                        &s.description
                    },
                    s.files.len()
                );
                let resp = ui.add(egui::Button::selectable(i == active, label).wrap());
                if resp.clicked() && i != active {
                    switch_to = Some(i);
                }
                resp.context_menu(|ui| {
                    if ui.button("✏ Rename").clicked() {
                        rename = Some(RenameTarget::Series { slot, idx: i });
                        ui.close();
                    }
                    ui.separator();
                    for (label, op) in [
                        (format!("Copy series to dataset {other}"), TreeOp::Copy),
                        (format!("Move series to dataset {other}"), TreeOp::Move),
                    ] {
                        if ui.button(label).clicked() {
                            act = Some(TreeAction {
                                from: slot,
                                sel: TreeSel::Series(i),
                                op,
                            });
                            ui.close();
                        }
                    }
                    ui.separator();
                    ui.menu_button("4D group", |ui| {
                        for (gi, name) in &group_names {
                            if ui.button(format!("Add to {name}")).clicked() {
                                fourd = Some(FourDAction::Add {
                                    slot,
                                    group: *gi,
                                    series: i,
                                });
                                ui.close();
                            }
                        }
                        if ui.button("New 4D group from this series").clicked() {
                            fourd = Some(FourDAction::New { slot, series: i });
                            ui.close();
                        }
                        if ui.button("Re-detect 4D groups").clicked() {
                            fourd = Some(FourDAction::Redetect { slot });
                            ui.close();
                        }
                    });
                    ui.separator();
                    if ui.button("🗑 Remove").clicked() {
                        act = Some(TreeAction {
                            from: slot,
                            sel: TreeSel::Series(i),
                            op: TreeOp::Remove,
                        });
                        ui.close();
                    }
                });
                resp.on_hover_text(format!(
                    "{} · series UID …{}\nright-click: rename, copy / move to dataset \
                     {other}, or remove",
                    s.modality,
                    tail(&s.uid)
                ));
            }
        }
        if act.is_some() {
            self.tree_action = act;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if fourd.is_some() {
            self.fourd_action = fourd;
        }
        if let Some(i) = switch_to {
            self.start_series_switch(slot, i);
        }
    }

    /// One 4D group node: the ordered members, each row switching the
    /// displayed series like an ordinary series row.
    fn fourd_node(&mut self, ui: &mut egui::Ui, slot: usize, pi: usize, si: usize, gi: usize) {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(group) = study.fourd_groups.get(gi) else {
            return;
        };
        let title = format!("🎞 {}", group.name);
        let resolved = group.resolve(&study.series);
        // (member index, series index, row label) for every surviving member.
        let rows: Vec<(usize, usize, String)> = group
            .members
            .iter()
            .enumerate()
            .zip(&resolved)
            .filter_map(|((mi, m), r)| {
                r.map(|sidx| {
                    let se = &study.series[sidx];
                    let tag = m.role.tag();
                    let label = if tag.is_empty() {
                        format!("{} - {} ({} sl.)", m.label, se.description, se.files.len())
                    } else {
                        format!("{tag} - {} ({} sl.)", se.description, se.files.len())
                    };
                    (mi, sidx, label)
                })
            })
            .collect();
        let n_members = group.members.len();
        let active = study.active_series;

        let mut switch_to = None;
        let mut fourd: Option<FourDAction> = None;
        let mut rename = None;
        let showing = rows.iter().any(|(_, sidx, _)| *sidx == active);
        let resp = Self::wrapped_node(ui, ("fourd", slot, pi, si, gi), showing, title, |ui| {
            for (mi, sidx, label) in &rows {
                let resp = ui.add(egui::Button::selectable(*sidx == active, label).wrap());
                if resp.clicked() && *sidx != active {
                    switch_to = Some(*sidx);
                }
                resp.context_menu(|ui| {
                    if ui.button("⬆ Move up").clicked() {
                        fourd = Some(FourDAction::Shift {
                            slot,
                            group: gi,
                            member: *mi,
                            delta: -1,
                        });
                        ui.close();
                    }
                    if ui.button("⬇ Move down").clicked() {
                        fourd = Some(FourDAction::Shift {
                            slot,
                            group: gi,
                            member: *mi,
                            delta: 1,
                        });
                        ui.close();
                    }
                    ui.menu_button("Role", |ui| {
                        for (role, label) in [
                            (fourd::Role::Phase, "Phase"),
                            (fourd::Role::Average, "Average (AVG)"),
                            (fourd::Role::Mip, "MIP"),
                            (fourd::Role::MinIp, "MinIP"),
                        ] {
                            if ui.button(label).clicked() {
                                fourd = Some(FourDAction::SetRole {
                                    slot,
                                    group: gi,
                                    member: *mi,
                                    role,
                                });
                                ui.close();
                            }
                        }
                    });
                    ui.separator();
                    if ui.button("Remove from group").clicked() {
                        fourd = Some(FourDAction::RemoveMember {
                            slot,
                            group: gi,
                            member: *mi,
                        });
                        ui.close();
                    }
                });
            }
            if rows.len() < n_members {
                ui.weak(format!(
                    "{} member(s) whose series is gone",
                    n_members - rows.len()
                ));
            }
        });
        resp.context_menu(|ui| {
            if ui.button("✏ Rename").clicked() {
                rename = Some(RenameTarget::FourD { slot, idx: gi });
                ui.close();
            }
            if ui.button("📈 Motion / ITV analysis").clicked() {
                fourd = Some(FourDAction::Analyse { slot, group: gi });
                ui.close();
            }
            ui.separator();
            if ui.button("Re-detect 4D groups").clicked() {
                fourd = Some(FourDAction::Redetect { slot });
                ui.close();
            }
            if ui.button("Dissolve group").clicked() {
                fourd = Some(FourDAction::Dissolve { slot, group: gi });
                ui.close();
            }
        });
        resp.on_hover_text(
            "A 4D sub-study: the phases in temporal order, then the reconstructions.\n\
             Click a phase to display it; right-click for analysis and edits.",
        );
        if fourd.is_some() {
            self.fourd_action = fourd;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if let Some(i) = switch_to {
            self.start_series_switch(slot, i);
        }
    }

    /// Apply a deferred 4D-group edit from the tree's context menus.
    pub(super) fn apply_fourd_action(&mut self, act: FourDAction) {
        match act {
            FourDAction::Analyse { slot, group } => {
                self.open_motion_dialog(slot, Some(group));
                return;
            }
            FourDAction::Redetect { slot } => {
                if let Some(study) = self.slots[slot].study.as_mut() {
                    // An explicit re-detect is the one action that clears
                    // dissolved tombstones — the user asked for detection.
                    study.fourd_groups.retain(|g| !g.dissolved);
                    study.refresh_fourd();
                }
                return;
            }
            _ => {}
        }
        let slot = match act {
            FourDAction::Add { slot, .. }
            | FourDAction::New { slot, .. }
            | FourDAction::RemoveMember { slot, .. }
            | FourDAction::Shift { slot, .. }
            | FourDAction::SetRole { slot, .. }
            | FourDAction::Dissolve { slot, .. } => slot,
            _ => return,
        };
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        match act {
            FourDAction::Add { group, series, .. } => {
                let Some(se) = study.series.get(series) else {
                    return;
                };
                let member = fourd::member_for(se, {
                    study
                        .fourd_groups
                        .get(group)
                        .map(|g| g.phase_members().len() + 1)
                        .unwrap_or(1)
                });
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    if !g.members.iter().any(|m| m.series_uid == member.series_uid) {
                        g.members.push(member);
                        g.custom = true;
                    }
                }
            }
            FourDAction::New { series, .. } => {
                let Some(se) = study.series.get(series) else {
                    return;
                };
                let member = fourd::member_for(se, 1);
                let n = study.fourd_groups.len() + 1;
                study.fourd_groups.push(fourd::FourDGroup {
                    name: format!("4D group {n}"),
                    study_uid: se.study_uid.clone(),
                    members: vec![member],
                    custom: true,
                    dissolved: false,
                });
            }
            FourDAction::RemoveMember { group, member, .. } => {
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    if g.members.len() == 1 && member == 0 {
                        // Removing the last member dissolves the group; the
                        // member stays inside the tombstone so re-detection
                        // does not immediately rebuild what was taken apart.
                        g.dissolved = true;
                        g.custom = true;
                    } else if member < g.members.len() {
                        g.members.remove(member);
                        g.custom = true;
                    }
                }
            }
            FourDAction::Shift {
                group,
                member,
                delta,
                ..
            } => {
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    let to = member as isize + delta;
                    if to >= 0 && (to as usize) < g.members.len() {
                        g.members.swap(member, to as usize);
                        g.custom = true;
                    }
                }
            }
            FourDAction::SetRole {
                group,
                member,
                role,
                ..
            } => {
                if let Some(m) = study
                    .fourd_groups
                    .get_mut(group)
                    .and_then(|g| g.members.get_mut(member))
                {
                    m.role = role;
                    if role != fourd::Role::Phase {
                        m.label = role.tag().to_string();
                        m.percent = None;
                    } else if m.label.is_empty()
                        || m.label == "AVG"
                        || m.label == "MIP"
                        || m.label == "MinIP"
                    {
                        m.label = format!("t{}", member + 1);
                    }
                }
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    g.custom = true;
                }
            }
            FourDAction::Dissolve { group, .. } if group < study.fourd_groups.len() => {
                // A custom group leaves nothing behind; an auto-detected one
                // leaves a hidden tombstone so re-detection (on the next
                // series change) does not resurrect it. *Re-detect 4D
                // groups* clears tombstones explicitly.
                if study.fourd_groups[group].custom {
                    study.fourd_groups.remove(group);
                } else {
                    study.fourd_groups[group].dissolved = true;
                }
            }
            _ => {}
        }
    }

    // -- Structure sets and segmentation series ----------------------------

    /// Right-click menu of a series node: what image series it is drawn on,
    /// where it goes, and whether it stays.
    fn set_context_menu(&self, ui: &mut egui::Ui, here: SetRef, out: &mut Option<SetAction>) {
        let other = SLOT_NAMES[1 - here.slot];
        if ui.button("✏ Rename").clicked() {
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
                .button("💾 Export as DICOM SEG")
                .on_hover_text("Write this series as one DICOM Segmentation file")
                .clicked()
            {
                *out = Some(SetAction::ExportSeg {
                    set: here,
                    items: Vec::new(),
                });
                ui.close();
            }
        }
        ui.separator();
        if ui.button("🗑 Remove").clicked() {
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
                    .button(format!("✏ {} ({} segments)", sr.label, sr.segs.len()))
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
            if ui.button("✏ ➕ a new segmentation series").clicked() {
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
        if ui.button("✏ Rename").clicked() {
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
        if ui
            .button(format!("∪ Combine {what}"))
            .on_hover_text(
                "Open the structure-algebra window with these as its operands - union, \
                 intersection, subtraction, margins",
            )
            .clicked()
        {
            *out = Some(ItemAction::Combine {
                from,
                items: items.clone(),
            });
            ui.close();
        }
        if ui
            .button(format!("📊 Plot {what} on a DVH"))
            .on_hover_text(
                "Open the dose-volume histogram window with these structures against \
                 the loaded dose",
            )
            .clicked()
        {
            *out = Some(ItemAction::Dvh {
                from,
                items: items.clone(),
            });
            ui.close();
        }
        if from.kind == SetKind::Segmentations {
            ui.separator();
            if ui
                .button(format!("💾 Export {what} as DICOM SEG"))
                .on_hover_text("Writes just these segments, as a SEG series of their own")
                .clicked()
            {
                *out = Some(ItemAction::ExportSeg {
                    from,
                    items: items.clone(),
                });
                ui.close();
            }
        }
        ui.separator();
        if ui.button("🗑 Remove").clicked() {
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

    /// The *Copy to / Move to / Remove / Export* buttons that act on whatever
    /// is ticked, added inline so they share one row with *All* / *None* — a
    /// multi-item action should not have to be found by right-clicking
    /// exactly the right row.
    fn selection_buttons(
        &self,
        ui: &mut egui::Ui,
        here: SetRef,
        selection: &[usize],
        item_act: &mut Option<ItemAction>,
    ) {
        let n = selection.len();
        let what = format!("{n} ticked {}", here.kind.item_name(n));
        ui.add_enabled_ui(n > 0, |ui| {
            ui.menu_button("Copy to", |ui| {
                if let Some(to) = self.destination_menu(ui, here) {
                    *item_act = Some(ItemAction::Transfer {
                        from: here,
                        items: selection.to_vec(),
                        to,
                        copy: true,
                    });
                }
            })
            .response
            .on_hover_text(format!("Copy the {what} into another series"));
            ui.menu_button("Move to", |ui| {
                if let Some(to) = self.destination_menu(ui, here) {
                    *item_act = Some(ItemAction::Transfer {
                        from: here,
                        items: selection.to_vec(),
                        to,
                        copy: false,
                    });
                }
            })
            .response
            .on_hover_text(format!("Move the {what} into another series"));
            if ui
                .small_button("🗑")
                .on_hover_text(format!("Remove the {what}"))
                .clicked()
            {
                *item_act = Some(ItemAction::Remove {
                    from: here,
                    items: selection.to_vec(),
                });
            }
            if here.kind == SetKind::Segmentations
                && ui
                    .small_button("💾")
                    .on_hover_text(format!("Write the {what} as a DICOM SEG file of their own"))
                    .clicked()
            {
                *item_act = Some(ItemAction::ExportSeg {
                    from: here,
                    items: selection.to_vec(),
                });
            }
        });
        ui.weak(format!("{n} selected"));
    }

    /// The RT structure sets filed under one study, then the ROIs of the
    /// active one.
    ///
    /// A ROI's check box is both its visibility and its selection, so *All* /
    /// *None* tick everything or nothing, Shift-click extends a range, and
    /// the action row works on whatever is ticked.
    pub(super) fn structures_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        pat: usize,
        stu: usize,
        which: &[usize],
    ) {
        if self.slots[slot].study.is_none() {
            return;
        }
        // Sets can be listed, renamed, moved and deleted without an image
        // volume; only what would draw into one is held back.
        let has_volume = self.slots[slot].has_volume();
        // Rendering runs behind a shared borrow of `self` (the context menus
        // need to list the other dataset's series), so the one piece of
        // mutable state in the list is edited on a copy and written back.
        let mut vis = std::mem::take(&mut self.slots[slot].roi_visible);
        let mut new_active: Option<usize> = None;
        let mut set_act: Option<SetAction> = None;
        let mut item_act: Option<ItemAction> = None;
        let mut new_anchor: Option<(SetRef, usize)> = None;
        let shift = ui.input(|i| i.modifiers.shift);
        {
            let me = &*self;
            let study = me.slots[slot].study.as_ref().unwrap();
            let sets = &study.structure_sets;
            // The active set counts as belonging here only when this study
            // node actually holds it; otherwise this node shows its sets but
            // edits none of them.
            let active_set = which
                .contains(&me.slots[slot].active_structs)
                .then(|| me.slots[slot].active_structs);
            let n_rois = active_set
                .and_then(|i| sets.get(i))
                .map(|ss| ss.rois.len())
                .unwrap_or(0);
            if active_set.is_some() {
                vis.resize(n_rois, true);
            }
            let n_vis = vis.iter().filter(|v| **v).count();
            let title = match active_set {
                Some(_) => format!("RT structures ({n_vis}/{n_rois})"),
                None => format!("RT structures ({})", which.len()),
            };
            Self::wrapped_node(ui, ("structs", slot, pat, stu), true, title, |ui| {
                if ui
                    .add_enabled(has_volume, egui::Button::new("New series").small())
                    .on_hover_text(if has_volume {
                        "An empty RT structure set, drawn on the displayed image series"
                    } else {
                        "This dataset has no image volume to draw on"
                    })
                    .clicked()
                {
                    set_act = Some(SetAction::New(SetRef {
                        slot,
                        kind: SetKind::Structures,
                        idx: SetRef::NEW,
                    }));
                }
                for &i in which {
                    let Some(set) = sets.get(i) else { continue };
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
                    let resp = ui.add(
                        egui::Button::selectable(
                            active_set == Some(i),
                            format!(
                                "▣ {name} ({} ROIs){}",
                                set.rois.len(),
                                Self::series_suffix(study, &set.referenced_series_uid)
                            ),
                        )
                        .wrap(),
                    );
                    if resp.clicked() && active_set != Some(i) {
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
                let (Some(active_set), Some(ss)) =
                    (active_set, active_set.and_then(|i| sets.get(i)))
                else {
                    return;
                };
                let here = SetRef {
                    slot,
                    kind: SetKind::Structures,
                    idx: active_set,
                };
                let selection: Vec<usize> = vis
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v)
                    .map(|(i, _)| i)
                    .collect();
                ui.horizontal_wrapped(|ui| {
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
                    me.selection_buttons(ui, here, &selection, &mut item_act);
                });
                let anchor = me.tick_anchor.filter(|(r, _)| *r == here).map(|(_, i)| i);
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
                        if resp.clicked() {
                            new_anchor = Some((here, apply_tick(&mut vis, i, shift, anchor)));
                        }
                        resp.context_menu(|ui| {
                            me.item_context_menu(ui, here, i, &roi.name, &selection, &mut item_act)
                        });
                        resp.on_hover_text(format!(
                            "ROI {} · {} contour(s)\nShift-click: tick or untick the whole \
                             range from the last one\nright-click: copy / move / remove - \
                             every ticked structure at once",
                            roi.number,
                            roi.contours.len()
                        ));
                    });
                }
            });
        }
        self.slots[slot].roi_visible = vis;
        if new_anchor.is_some() {
            self.tick_anchor = new_anchor;
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
        }
        if set_act.is_some() {
            self.set_action = set_act;
        }
        if item_act.is_some() {
            self.item_action = item_act;
        }
    }

    /// The segmentation series filed under one study, then the segments of
    /// the active one with the tools that edit them.
    pub(super) fn segmentation_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        pat: usize,
        stu: usize,
        which: &[usize],
    ) {
        if self.slots[slot].study.is_none() {
            return;
        }
        // Sets can be listed, renamed, moved and deleted without an image
        // volume; only what would draw into one is held back.
        let has_volume = self.slots[slot].has_volume();
        // Whichever engine is running on this slot: its glyph, message and
        // fraction, read before the section borrows anything.
        let running = self
            .running_tool(slot)
            .map(|(tool, p)| (tool.glyph, p.get(), p.frac()));
        let active_here = self.slots[slot]
            .seg_series_idx()
            .filter(|i| which.contains(i));
        // (name, colour, visible, cm³) of the active series' segments — the
        // editable columns live on this copy, see `structures_section`.
        let mut rows: Vec<(String, [u8; 3], bool, f64)> = {
            let s = &self.slots[slot];
            let spacing = s
                .study
                .as_ref()
                .map(|st| st.volume.spacing)
                .unwrap_or([1.0; 3]);
            match active_here {
                Some(_) => s
                    .segs()
                    .iter()
                    .map(|g| (g.name.clone(), g.color, g.visible, g.volume_cm3(spacing)))
                    .collect(),
                None => Vec::new(),
            }
        };
        let before: Vec<([u8; 3], bool)> = rows.iter().map(|r| (r.1, r.2)).collect();
        let mut make_new = false;
        let mut new_series = false;
        let mut open_tool: Option<&ToolInfo> = None;
        let mut cancel_tool = false;
        let mut set_all: Option<bool> = None;
        let mut new_active_series: Option<usize> = None;
        let mut activate: Option<usize> = None;
        let mut set_act: Option<SetAction> = None;
        let mut item_act: Option<ItemAction> = None;
        let mut new_anchor: Option<(SetRef, usize)> = None;
        let shift = ui.input(|i| i.modifiers.shift);
        {
            let me = &*self;
            let study = me.slots[slot].study.as_ref().unwrap();
            let series = &study.seg_series;
            let active_seg = me.slots[slot].active_seg;
            let n_vis = rows.iter().filter(|r| r.2).count();
            let n_segs = active_here
                .and_then(|i| series.get(i))
                .map(|s| s.segs.len())
                .unwrap_or(0);
            let title = match active_here {
                Some(_) => format!("Segmentations ({n_vis}/{n_segs})"),
                None => format!("Segmentations ({})", which.len()),
            };
            Self::wrapped_node(
                ui,
                ("segs", slot, pat, stu),
                !which.is_empty(),
                title,
                |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .add_enabled(has_volume, egui::Button::new("New series").small())
                            .on_hover_text(if has_volume {
                                "An empty segmentation series, drawn on the displayed image \
                             series - exports as one DICOM SEG file"
                            } else {
                                "This dataset has no image volume to draw on"
                            })
                            .clicked()
                        {
                            new_series = true;
                        }
                        for (tool, hint) in [
                            (
                                &BODY_CONTOUR,
                                "Outline the patient without the couch, the chair or the \
                             immobilisation (EXTERNAL)",
                            ),
                            (
                                &COMBINE,
                                "Build one structure out of others: union, intersection, \
                             subtraction, margins",
                            ),
                            (
                                &AUTOSEG,
                                "Automatic multi-organ segmentation (TotalSegmentator, \
                             117 structures)",
                            ),
                            (
                                &PROMPT_SEG,
                                "Segment whatever the crosshair points at - a box, a click \
                             or a structure name (SegVol)",
                            ),
                            (
                                &SLICE_PROP,
                                "Box a structure on one slice and follow it through the \
                             stack (MedSAM2)",
                            ),
                        ] {
                            if ui
                                .add_enabled(
                                    has_volume,
                                    egui::Button::new(tool.short_button()).small(),
                                )
                                .on_hover_text(if has_volume {
                                    hint
                                } else {
                                    "This dataset has no image volume to segment"
                                })
                                .clicked()
                            {
                                open_tool = Some(tool);
                            }
                        }
                    });
                    for &i in which {
                        let Some(sr) = series.get(i) else { continue };
                        let here = SetRef {
                            slot,
                            kind: SetKind::Segmentations,
                            idx: i,
                        };
                        let resp = ui.add(
                            egui::Button::selectable(
                                active_here == Some(i),
                                format!(
                                    "✏ {} ({} segments){}",
                                    sr.label,
                                    sr.segs.len(),
                                    Self::series_suffix(study, &sr.referenced_series_uid)
                                ),
                            )
                            .wrap(),
                        );
                        if resp.clicked() && active_here != Some(i) {
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
                    let Some(active_series) = active_here else {
                        return;
                    };
                    // Masks of a series drawn on another image series are on that
                    // series' lattice, so nothing here can index them.
                    if !study.has_volume() {
                        ui.weak(
                            "this dataset has no image volume - add the image series these \
                         segments were drawn on to see and edit them",
                        );
                        return;
                    }
                    if series[active_series].grid.dims != study.volume.dims {
                        ui.weak(
                            "drawn on another image series - display that series to see and \
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
                    let here = SetRef {
                        slot,
                        kind: SetKind::Segmentations,
                        idx: active_series,
                    };
                    let selection: Vec<usize> = rows
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.2)
                        .map(|(i, _)| i)
                        .collect();
                    ui.horizontal_wrapped(|ui| {
                        if ui
                            .small_button("New")
                            .on_hover_text(
                                "An empty segmentation to paint with 🎨 / ✨ in the views",
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
                        me.selection_buttons(ui, here, &selection, &mut item_act);
                    });
                    let anchor = me.tick_anchor.filter(|(r, _)| *r == here).map(|(_, i)| i);
                    // The check boxes are edited on `ticks` so a Shift-range can
                    // reach rows the loop has already drawn.
                    let mut ticks: Vec<bool> = rows.iter().map(|r| r.2).collect();
                    for (i, row) in rows.iter_mut().enumerate() {
                        let name = row.0.clone();
                        ui.horizontal(|ui| {
                            ui.color_edit_button_srgb(&mut row.1);
                            let tick = ui.checkbox(&mut ticks[i], "").on_hover_text(
                                "Show / select this segmentation\nShift-click: tick or untick \
                             the whole range from the last one",
                            );
                            if tick.clicked() {
                                new_anchor = Some((here, apply_tick(&mut ticks, i, shift, anchor)));
                            }
                            let resp = ui
                                .add(egui::Button::selectable(i == active_seg, name.clone()).wrap())
                                .on_hover_text(
                                    "Click to make this the segmentation the tools edit",
                                );
                            if resp.clicked() {
                                activate = Some(i);
                            }
                            resp.context_menu(|ui| {
                                me.item_context_menu(ui, here, i, &name, &selection, &mut item_act)
                            });
                            ui.weak(format!("{:.1} cm³", row.3));
                        });
                    }
                    for (row, on) in rows.iter_mut().zip(ticks) {
                        row.2 = on;
                    }
                },
            );
        }
        if let Some(v) = set_all {
            rows.iter_mut().for_each(|r| r.2 = v);
        }
        if new_anchor.is_some() {
            self.tick_anchor = new_anchor;
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
            Some(g) if g == COMBINE.glyph => self.open_combine_dialog(slot, Vec::new()),
            Some(g) if g == BODY_CONTOUR.glyph => self.open_body_dialog(slot),
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
        if set_act.is_some() {
            self.set_action = set_act;
        }
        if item_act.is_some() {
            self.item_action = item_act;
        }
    }

    /// The RTDOSE grids filed under one study: which one is displayed, and
    /// what it is. How dose is *drawn* is a display setting shared by both
    /// datasets, so it lives in [`Self::dose_display_section`] instead.
    pub(super) fn dose_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        pat: usize,
        stu: usize,
        which: &[usize],
    ) {
        if which.is_empty() {
            return;
        }
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        // The views draw one dose grid at a time, so the tick boxes work as
        // a set of one: ticking a grid shows it and unticks the others,
        // unticking takes dose off the images altogether.
        let mut shown_dose: Option<Option<usize>> = None;
        let dose_on = self.dose_mode != DoseMode::Off;
        {
            let StudySlot {
                study,
                active_dose,
                dose_reference,
                ..
            } = &mut self.slots[slot];
            let Some(study) = study.as_ref() else { return };
            let doses = &study.doses;
            let plans = &study.plans;
            let mut picked = (*active_dose).min(doses.len().saturating_sub(1));
            let hdr = Self::wrapped_node(
                ui,
                ("dose", slot, pat, stu),
                false,
                format!("Dose ({})", which.len()),
                |ui| {
                    for &i in which {
                        let Some(d) = doses.get(i) else { continue };
                        let mut on = dose_on && i == picked;
                        let resp = ui.checkbox(&mut on, d.label.clone());
                        if resp.changed() {
                            shown_dose = Some(on.then_some(i));
                            if on {
                                picked = i;
                            }
                        } else if resp.clicked() {
                            picked = i;
                        }
                        resp.on_hover_text(format!(
                            "{}  max {:.2} {}\nright-click: rename or remove",
                            d.summation_type,
                            d.max_dose,
                            d.units.to_lowercase()
                        ))
                        .context_menu(|ui| {
                            if ui.button("✏ Rename").clicked() {
                                rename = Some(RenameTarget::Dose { slot, idx: i });
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("🗑 Remove").clicked() {
                                remove = Some(ObjRef {
                                    slot,
                                    kind: ObjKind::Dose,
                                    idx: i,
                                });
                                ui.close();
                            }
                        });
                    }
                    let Some(d) = doses.get(picked) else { return };
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
                },
            );
            *active_dose = picked;
            let _ = &hdr;
        }
        if let Some(pick) = shown_dose {
            match pick {
                // Ticking a grid turns dose display on if it was off; the
                // *Dose display* section decides how it is drawn.
                Some(_) if self.dose_mode == DoseMode::Off => {
                    self.dose_mode = DoseMode::Colorwash;
                }
                None => self.dose_mode = DoseMode::Off,
                _ => {}
            }
            self.settings_gen += 1;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if remove.is_some() {
            self.obj_remove = remove;
        }
    }

    /// How dose is drawn — colorwash, isodose lines, opacity, threshold and
    /// the isodose ladder. Shared by both datasets, so it is shown once, at
    /// dataset level, under the first dataset that actually has dose.
    pub(super) fn dose_display_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let first = (0..2).find(|&s| {
            self.slots[s]
                .study
                .as_ref()
                .is_some_and(|st| !st.doses.is_empty())
        });
        if first != Some(slot) {
            return;
        }
        let mut mode = self.dose_mode;
        let mut opacity = self.dose_opacity;
        let mut threshold = self.dose_threshold_pct;
        Self::wrapped_node(ui, ("dose_display", slot), false, "Dose display", |ui| {
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
            ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"));
            ui.add(egui::Slider::new(&mut threshold, 0.0..=100.0).text("Threshold %"));
            ui.weak("Isodose levels (% of reference)");
            for l in &mut self.iso_levels {
                ui.horizontal(|ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                    ui.painter().rect_filled(rect, 2.0, l.color);
                    ui.checkbox(&mut l.on, format!("{:.0}%", l.pct));
                });
            }
        });
        self.dose_mode = mode;
        self.dose_opacity = opacity;
        self.dose_threshold_pct = threshold;
    }

    pub(super) fn plan_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        pat: usize,
        stu: usize,
        which: &[usize],
    ) {
        if which.is_empty() {
            return;
        }
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        // One flag per plan, defaulting to shown: what is drawn from a plan
        // is its isocenters, and a plan the user has unticked keeps them out
        // of the views.
        let n_plans = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.plans.len())
            .unwrap_or(0);
        let mut visible = std::mem::take(&mut self.slots[slot].plan_visible);
        visible.resize(n_plans, true);
        {
            let Some(study) = &self.slots[slot].study else {
                self.slots[slot].plan_visible = visible;
                return;
            };
            for &pi in which {
                let Some(plan) = study.plans.get(pi) else {
                    continue;
                };
                let mut shown = visible.get(pi).copied().unwrap_or(true);
                let plan_hdr = Self::checked_node(
                    ui,
                    ("plan", slot, pat, stu, pi),
                    false,
                    &mut shown,
                    format!(
                        "Plan: {}",
                        if plan.label.is_empty() {
                            "unnamed"
                        } else {
                            &plan.label
                        }
                    ),
                    |ui| {
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
                            egui::Grid::new(("beam_grid", slot, pat, stu, pi))
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
                                                .unwrap_or_else(|| "-".into()),
                                        );
                                        ui.label(
                                            b.couch_angle
                                                .map(|c| format!("{c:.0}"))
                                                .unwrap_or_else(|| "-".into()),
                                        );
                                        ui.label(match (b.energy_min, b.energy_max) {
                                            (Some(a), Some(bb)) if (a - bb).abs() > 0.01 => {
                                                format!("{a:.0}-{bb:.0}")
                                            }
                                            (Some(a), _) => format!("{a:.0}"),
                                            _ => "-".into(),
                                        });
                                        ui.label(
                                            b.meterset
                                                .map(|m| format!("{m:.1}"))
                                                .unwrap_or_else(|| "-".into()),
                                        );
                                        ui.label(format!("{}", b.n_control_points));
                                        ui.end_row();
                                    }
                                });
                        }
                    },
                );
                if let Some(flag) = visible.get_mut(pi) {
                    *flag = shown;
                }
                plan_hdr.context_menu(|ui| {
                    if ui.button("✏ Rename").clicked() {
                        rename = Some(RenameTarget::Plan { slot, idx: pi });
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("🗑 Remove").clicked() {
                        remove = Some(ObjRef {
                            slot,
                            kind: ObjKind::Plan,
                            idx: pi,
                        });
                        ui.close();
                    }
                });
            }
        }
        self.slots[slot].plan_visible = visible;
        if rename.is_some() {
            self.rename_request = rename;
        }
        if remove.is_some() {
            self.obj_remove = remove;
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
        let mut close_idx = None;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let open_windows: Vec<usize> = self
            .planar_windows
            .iter()
            .filter(|w| w.slot == slot && w.open)
            .map(|w| w.idx)
            .collect();
        // Normally a side note beneath the tree; for a dataset with no image
        // volume these *are* the images, so the section opens itself.
        let sole_content = !self.slots[slot].has_volume();
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            egui::CollapsingHeader::new(format!("Planar images ({n})"))
                .id_salt(("planar", slot))
                .default_open(sole_content)
                .show(ui, |ui| {
                    for (i, img) in study.planar_images.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(format!("[{}]", img.modality)).weak());
                            // A planar image is shown in a window of its own,
                            // so the tick box is that window.
                            let mut shown = open_windows.contains(&i);
                            let resp = ui.checkbox(&mut shown, &img.label);
                            if resp.changed() {
                                if shown {
                                    open_idx = Some(i);
                                } else {
                                    close_idx = Some(i);
                                }
                            }
                            resp.on_hover_text(
                                "Show this image in its own window\nright-click: rename or remove",
                            )
                            .context_menu(|ui| {
                                if ui.button("✏ Rename").clicked() {
                                    rename = Some(RenameTarget::Planar { slot, idx: i });
                                    ui.close();
                                }
                                ui.separator();
                                if ui.button("🗑 Remove").clicked() {
                                    remove = Some(ObjRef {
                                        slot,
                                        kind: ObjKind::Planar,
                                        idx: i,
                                    });
                                    ui.close();
                                }
                            });
                        });
                    }
                });
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if remove.is_some() {
            self.obj_remove = remove;
        }
        if let Some(i) = close_idx {
            for w in self
                .planar_windows
                .iter_mut()
                .filter(|w| w.slot == slot && w.idx == i)
            {
                w.open = false;
            }
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
        let both = self.slots[0].has_volume() && self.slots[1].has_volume();
        let mut apply: Option<(registration::RigidTransform, usize)> = None;
        let mut apply_grid: Option<(usize, usize)> = None;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
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
                            if ui.button("✏ Rename").clicked() {
                                rename = Some(RenameTarget::Registration { slot, idx: ri });
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("🗑 Remove").clicked() {
                                remove = Some(ObjRef {
                                    slot,
                                    kind: ObjKind::Registration,
                                    idx: ri,
                                });
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
                                    ui.weak("  (matrix is not a pure rigid transform - cannot apply)");
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
                                            "The grid's own frame of reference matches this                                              dataset - this is the direction the file means"
                                        }
                                        Some(_) => {
                                            "The grid's frame of reference matches the *other*                                              dataset; applying it this way round inverts what                                              the file says"
                                        }
                                        None => {
                                            "Neither loaded dataset matches the grid's frame of                                              reference - check that this is the right pair"
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
        let mut remove: Option<ObjRef> = None;
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
                        if ui.button("✏ Rename").clicked() {
                            rename = Some(RenameTarget::Record { slot, idx: ri });
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("🗑 Remove").clicked() {
                            remove = Some(ObjRef {
                                slot,
                                kind: ObjKind::Record,
                                idx: ri,
                            });
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
                                        .unwrap_or_else(|| "-".into()),
                                );
                                ui.label(
                                    b.delivered_meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "-".into()),
                                );
                                ui.label(match (b.specified_meterset, b.delivered_meterset) {
                                    (Some(s), Some(d)) if s > 1e-9 => {
                                        format!("{:+.1}", 100.0 * (d - s) / s)
                                    }
                                    _ => "-".into(),
                                });
                                let status = if b.termination_status.is_empty() {
                                    "-"
                                } else {
                                    &b.termination_status
                                };
                                if status == "NORMAL" || status == "-" {
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
        if remove.is_some() {
            self.obj_remove = remove;
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

/// One study node of the data tree: which of the dataset's objects belong
/// under it, as indices into the parallel arrays of [`LoadedStudy`].
pub(super) struct StudyNode {
    uid: String,
    title: String,
    /// Image series of this study, grouped by modality in first-seen order —
    /// the CT / MR / US level DICOM implies but does not store as a node.
    modalities: Vec<(String, Vec<usize>)>,
    structs: Vec<usize>,
    segs: Vec<usize>,
    doses: Vec<usize>,
    plans: Vec<usize>,
    /// 4D groups of this study — indices into `LoadedStudy::fourd_groups`.
    fourd: Vec<usize>,
}

/// One patient node: the studies filed under them.
pub(super) struct PatientNode {
    key: String,
    title: String,
    studies: Vec<StudyNode>,
}

/// Sort a dataset into the patient ▶ study ▶ category ▶ series tree.
///
/// Series carry the patient and study they belong to, so those two levels
/// fall straight out of them. The RT objects carry a StudyInstanceUID as
/// well, but not always a usable one — a set built in the application from a
/// series that had none, or a file written by a tool that left it blank. An
/// object whose study is not in the tree is therefore filed under the study
/// of the image series it references, and failing that under the first study
/// there is: a structure set that cannot be reached is worse than one shown
/// a level away from where its header claims it lives.
pub(super) fn tree_layout(study: &LoadedStudy) -> Vec<PatientNode> {
    // Series filed under a 4D group render inside that node, not under
    // their modality — one series, one place in the tree.
    let mut grouped = vec![false; study.series.len()];
    for g in &study.fourd_groups {
        if g.dissolved {
            continue;
        }
        for r in g.resolve(&study.series).into_iter().flatten() {
            grouped[r] = true;
        }
    }
    // Patients and their studies, both in first-seen order.
    let mut patients: Vec<PatientNode> = Vec::new();
    for se in &study.series {
        let key = se.patient_key().to_string();
        if !patients.iter().any(|p| p.key == key) {
            let name = se.patient_name.replace('^', " ");
            let title = match (name.is_empty(), se.patient_id.is_empty()) {
                (true, true) => "Unknown patient".to_string(),
                (true, false) => format!("Patient {}", se.patient_id),
                (false, true) => name.clone(),
                (false, false) => format!("{} ({})", name, se.patient_id),
            };
            patients.push(PatientNode {
                key,
                title,
                studies: Vec::new(),
            });
        }
    }
    for (si, se) in study.series.iter().enumerate() {
        let p = patients
            .iter_mut()
            .find(|p| p.key == se.patient_key())
            .expect("every patient was collected above");
        let node = match p.studies.iter_mut().position(|s| s.uid == se.study_uid) {
            Some(i) => &mut p.studies[i],
            None => {
                let n = p.studies.len() + 1;
                let title = format!(
                    "Study {}{}",
                    if se.study_date.is_empty() {
                        n.to_string()
                    } else {
                        se.study_date.clone()
                    },
                    if se.study_description.is_empty() {
                        String::new()
                    } else {
                        format!(" - {}", se.study_description)
                    }
                );
                p.studies.push(StudyNode {
                    uid: se.study_uid.clone(),
                    title,
                    modalities: Vec::new(),
                    structs: Vec::new(),
                    segs: Vec::new(),
                    doses: Vec::new(),
                    plans: Vec::new(),
                    fourd: Vec::new(),
                });
                p.studies.last_mut().expect("just pushed")
            }
        };
        if grouped[si] {
            continue;
        }
        let modality = if se.modality.is_empty() {
            "Other".to_string()
        } else {
            se.modality.clone()
        };
        match node.modalities.iter_mut().find(|(m, _)| *m == modality) {
            Some((_, v)) => v.push(si),
            None => node.modalities.push((modality, vec![si])),
        }
    }

    // File each 4D group under its study node (falling back to the study of
    // its first surviving series, then to the first study — same rule as
    // the RT objects below).
    for (gi, g) in study.fourd_groups.iter().enumerate() {
        if g.dissolved {
            continue;
        }
        let resolved = g.resolve(&study.series);
        let Some(first) = resolved.iter().flatten().next() else {
            continue; // nothing left of this group
        };
        let fallback = &study.series[*first].study_uid;
        let find = |uid: &str| -> Option<(usize, usize)> {
            patients.iter().enumerate().find_map(|(pi, p)| {
                p.studies
                    .iter()
                    .position(|st| st.uid == uid)
                    .map(|si| (pi, si))
            })
        };
        if let Some((pi, si)) = find(&g.study_uid).or_else(|| find(fallback)) {
            patients[pi].studies[si].fourd.push(gi);
        }
    }

    // Studies that no image series announced.
    //
    // A dataset does not have to contain images at all — a folder or a file
    // selection can hold nothing but a structure set, a plan, a dose grid or
    // a handful of RT images. Those objects carry their own Study Instance
    // UID, so a study node is made from it and filed under the patient the
    // dataset's own metadata names. Without this the tree would be empty and
    // the objects invisible, which is the one outcome worse than no images.
    {
        let known: std::collections::HashSet<String> = patients
            .iter()
            .flat_map(|p| p.studies.iter().map(|s| s.uid.clone()))
            .collect();
        let mut missing: Vec<String> = Vec::new();
        let mut orphans = false;
        let mut want = |uid: &str| {
            // A blank UID names no study, so it gets no node of its own: the
            // fallback below files those objects under the first study there
            // is, which is the long-standing rule.
            if uid.is_empty() {
                orphans = true;
            } else if !known.contains(uid) && !missing.iter().any(|u| u == uid) {
                missing.push(uid.to_string());
            }
        };
        for ss in &study.structure_sets {
            want(&ss.study_uid);
        }
        for sr in &study.seg_series {
            want(&sr.study_uid);
        }
        for d in &study.doses {
            want(&d.study_uid);
        }
        for pl in &study.plans {
            want(&pl.study_uid);
        }
        // …unless there is no first study either. Then one study node has to
        // exist for those objects to be reachable at all.
        if missing.is_empty() && patients.is_empty() && orphans {
            missing.push(String::new());
        }
        if !missing.is_empty() {
            let m = &study.meta;
            let name = m.patient_name.replace('^', " ");
            let key = if !m.patient_id.is_empty() {
                m.patient_id.clone()
            } else if !name.is_empty() {
                name.clone()
            } else {
                "?".to_string()
            };
            let title = match (name.is_empty(), m.patient_id.is_empty()) {
                (true, true) => "Unknown patient".to_string(),
                (true, false) => format!("Patient {}", m.patient_id),
                (false, true) => name.clone(),
                (false, false) => format!("{name} ({})", m.patient_id),
            };
            let pi = match patients.iter().position(|p| p.key == key) {
                Some(i) => i,
                None => {
                    patients.push(PatientNode {
                        key,
                        title,
                        studies: Vec::new(),
                    });
                    patients.len() - 1
                }
            };
            for uid in missing {
                let n = patients[pi].studies.len() + 1;
                let title = format!(
                    "Study {}{}",
                    if m.study_date.is_empty() {
                        n.to_string()
                    } else {
                        m.study_date.clone()
                    },
                    if m.study_description.is_empty() {
                        String::new()
                    } else {
                        format!(" - {}", m.study_description)
                    }
                );
                patients[pi].studies.push(StudyNode {
                    uid,
                    title,
                    modalities: Vec::new(),
                    structs: Vec::new(),
                    segs: Vec::new(),
                    doses: Vec::new(),
                    plans: Vec::new(),
                    fourd: Vec::new(),
                });
            }
        }
    }

    // Where an RT object goes, by the rule in the doc comment above.
    let series_study = |uid: &str| -> Option<String> {
        study
            .series
            .iter()
            .find(|se| se.uid == uid)
            .map(|se| se.study_uid.clone())
    };
    // The (patient, study) address of every study node, so the placement
    // below can look one up without borrowing `patients` while it fills them.
    let index: Vec<(String, usize, usize)> = patients
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| {
            p.studies
                .iter()
                .enumerate()
                .map(move |(si, s)| (s.uid.clone(), pi, si))
        })
        .collect();
    let place = |own: &str, referenced: &str| -> Option<(usize, usize)> {
        let find = |uid: &str| {
            index
                .iter()
                .find(|(u, _, _)| u == uid)
                .map(|(_, pi, si)| (*pi, *si))
        };
        find(own)
            .or_else(|| series_study(referenced).and_then(|u| find(&u)))
            .or_else(|| index.first().map(|(_, pi, si)| (*pi, *si)))
    };
    for (i, ss) in study.structure_sets.iter().enumerate() {
        if let Some((pi, si)) = place(&ss.study_uid, &ss.referenced_series_uid) {
            patients[pi].studies[si].structs.push(i);
        }
    }
    for (i, sr) in study.seg_series.iter().enumerate() {
        if let Some((pi, si)) = place(&sr.study_uid, &sr.referenced_series_uid) {
            patients[pi].studies[si].segs.push(i);
        }
    }
    for (i, d) in study.doses.iter().enumerate() {
        if let Some((pi, si)) = place(&d.study_uid, "") {
            patients[pi].studies[si].doses.push(i);
        }
    }
    for (i, p) in study.plans.iter().enumerate() {
        if let Some((pi, si)) = place(&p.study_uid, "") {
            patients[pi].studies[si].plans.push(i);
        }
    }
    patients
}

/// Apply a check-box click to a visibility/selection list, extending from
/// `anchor` when Shift is held, and return the new anchor.
///
/// egui has already toggled `vis[i]` by the time this runs, so the clicked
/// row's *new* value is what the span is filled with: tick one row and
/// Shift-tick a later one and everything between turns on; untick and
/// Shift-untick and it all turns off. Rows outside the span are never
/// touched — the box is a visibility toggle as much as a selection, and
/// silently hiding structures the user did not point at would be worse than
/// any convenience gained.
fn apply_tick(vis: &mut [bool], i: usize, shift: bool, anchor: Option<usize>) -> usize {
    if let (true, Some(a)) = (shift, anchor) {
        if a < vis.len() && i < vis.len() {
            let v = vis[i];
            let (lo, hi) = if a <= i { (a, i) } else { (i, a) };
            vis[lo..=hi].iter_mut().for_each(|x| *x = v);
        }
    }
    i
}

#[cfg(test)]
mod tick_tests {
    use super::apply_tick;

    /// Without Shift a click is what egui already did to that one row.
    #[test]
    fn a_plain_click_touches_only_its_own_row() {
        let mut v = vec![false, true, false, false];
        assert_eq!(apply_tick(&mut v, 2, false, Some(0)), 2, "anchor moves");
        assert_eq!(v, vec![false, true, false, false]);
    }

    /// Shift fills the span with the clicked row's new value, in either
    /// direction, and leaves everything outside it alone.
    #[test]
    fn shift_fills_the_span_from_the_anchor() {
        let mut v = vec![false; 6];
        v[4] = true; // egui toggled the clicked row on
        assert_eq!(apply_tick(&mut v, 4, true, Some(1)), 4);
        assert_eq!(v, vec![false, true, true, true, true, false]);

        // Backwards, and unticking: the span follows the clicked row's value.
        let mut v = vec![true; 6];
        v[1] = false;
        apply_tick(&mut v, 1, true, Some(3));
        assert_eq!(v, vec![true, false, false, false, true, true]);
    }

    /// A stale anchor (the list shrank, or it belongs to nothing) must not
    /// panic or reach outside the list.
    #[test]
    fn a_stale_anchor_is_ignored() {
        let mut v = vec![false, true];
        assert_eq!(apply_tick(&mut v, 1, true, Some(9)), 1);
        assert_eq!(v, vec![false, true]);
        let mut v = vec![false, true];
        assert_eq!(apply_tick(&mut v, 1, true, None), 1);
        assert_eq!(v, vec![false, true]);
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::dicomseg::SegSeries;
    use crate::geometry::Vec3;
    use crate::rtstruct::StructureSet;

    fn series(uid: &str, modality: &str, patient: &str, study: &str) -> loader::SeriesInfo {
        loader::SeriesInfo {
            uid: uid.into(),
            modality: modality.into(),
            description: format!("{uid} desc"),
            patient_id: patient.into(),
            patient_name: format!("{patient}^Name"),
            study_uid: study.into(),
            study_date: "20260827".into(),
            study_description: String::new(),
            series_number: None,
            temporal_id: None,
            files: vec![std::path::PathBuf::from(format!("{uid}.dcm"))],
        }
    }

    fn structset(sop: &str, series_uid: &str, study: &str) -> StructureSet {
        StructureSet {
            label: sop.into(),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: sop.into(),
            study_uid: study.into(),
            referenced_series_uid: series_uid.into(),
            file_name: String::new(),
            rois: Vec::new(),
        }
    }

    fn study() -> LoadedStudy {
        let vol = Arc::new(Volume {
            data: vec![0],
            dims: [1, 1, 1],
            spacing: [1.0; 3],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        });
        LoadedStudy {
            meta: loader::PatientMeta::default(),
            series: vec![
                series("ct1", "CT", "P1", "st1"),
                series("ct2", "CT", "P1", "st1"),
                series("mr1", "MR", "P1", "st1"),
                series("ct3", "CT", "P1", "st2"),
                series("us1", "US", "P2", "st3"),
            ],
            active_series: 0,
            volume: vol.clone(),
            structure_sets: vec![
                structset("ss1", "ct1", "st1"),
                // No study of its own: it must follow the series it references.
                structset("ss2", "ct3", ""),
                // Neither: it must still be reachable, under the first study.
                structset("ss3", "", ""),
            ],
            seg_series: vec![SegSeries::new(
                "segs".into(),
                vol.grid(),
                "us1".into(),
                "st3".into(),
            )],
            doses: Vec::new(),
            plans: Vec::new(),
            planar_images: Vec::new(),
            registrations: Vec::new(),
            treat_records: Vec::new(),
            fourd_groups: Vec::new(),
            warnings: Vec::new(),
            default_window: (40.0, 400.0),
        }
    }

    /// Patients, their studies and the modality level DICOM implies but does
    /// not store, all in first-seen order.
    #[test]
    fn the_tree_nests_patient_study_modality_series() {
        let layout = tree_layout(&study());
        assert_eq!(layout.len(), 2, "two patients");
        assert_eq!(layout[0].key, "P1");
        assert_eq!(layout[0].studies.len(), 2);
        assert_eq!(layout[1].studies.len(), 1);

        let st1 = &layout[0].studies[0];
        let mods: Vec<&str> = st1.modalities.iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(mods, vec!["CT", "MR"], "one node per modality, first seen");
        assert_eq!(st1.modalities[0].1, vec![0, 1], "both CT series under CT");
        assert_eq!(st1.modalities[1].1, vec![2]);
        assert_eq!(layout[1].studies[0].modalities[0].0, "US");
    }

    /// An RT object with an incomplete StudyInstanceUID must still land
    /// somewhere reachable rather than disappearing from the tree.
    #[test]
    fn rt_objects_fall_back_to_their_series_then_to_the_first_study() {
        let layout = tree_layout(&study());
        assert_eq!(
            layout[0].studies[0].structs,
            vec![0, 2],
            "own study, then the orphan"
        );
        assert_eq!(
            layout[0].studies[1].structs,
            vec![1],
            "no study of its own - filed under the study of the series it references"
        );
        assert_eq!(
            layout[1].studies[0].segs,
            vec![0],
            "the segmentation series follows its own study"
        );
        let total: usize = layout
            .iter()
            .flat_map(|p| p.studies.iter())
            .map(|s| s.structs.len())
            .sum();
        assert_eq!(total, 3, "every structure set is reachable exactly once");
    }

    /// A dataset can hold no image series at all — a folder of RT images, a
    /// structure set opened on its own. Its objects must still be in the
    /// tree, under a patient and a study, or there is no way to reach them.
    #[test]
    fn a_dataset_without_image_series_still_has_a_patient_and_a_study() {
        let mut st = study();
        st.series.clear();
        st.volume = Arc::new(Volume::empty());
        st.meta = loader::PatientMeta {
            patient_name: "Doe^John".into(),
            patient_id: "P9".into(),
            study_date: "20260901".into(),
            study_description: "Portal images".into(),
        };
        let layout = tree_layout(&st);
        assert_eq!(layout.len(), 1, "one patient, from the dataset's own tags");
        assert_eq!(layout[0].title, "Doe John (P9)");
        // ss1 names st1, ss2 names none but references a series that is gone,
        // ss3 names none at all: two real studies plus the fallback.
        assert_eq!(
            layout[0].studies.len(),
            2,
            "one node per study the objects name: {:?}",
            layout[0].studies.iter().map(|s| &s.uid).collect::<Vec<_>>()
        );
        assert!(
            layout[0]
                .studies
                .iter()
                .any(|s| s.title.contains("20260901")),
            "the study is dated from the dataset's metadata"
        );
        let total: usize = layout[0].studies.iter().map(|s| s.structs.len()).sum();
        assert_eq!(total, 3, "every structure set is still reachable");
        assert_eq!(
            layout[0]
                .studies
                .iter()
                .map(|s| s.segs.len())
                .sum::<usize>(),
            1,
            "and so is the segmentation series"
        );
    }

    /// The degenerate case: objects that name no study whatsoever, in a
    /// dataset with no series to fall back to.
    #[test]
    fn objects_naming_no_study_at_all_still_get_one() {
        let mut st = study();
        st.series.clear();
        st.volume = Arc::new(Volume::empty());
        st.structure_sets = vec![structset("ss3", "", "")];
        st.seg_series.clear();
        let layout = tree_layout(&st);
        assert_eq!(layout.len(), 1);
        assert_eq!(layout[0].studies.len(), 1);
        assert_eq!(layout[0].studies[0].uid, "", "the catch-all study");
        assert_eq!(layout[0].studies[0].structs, vec![0]);
    }
}
