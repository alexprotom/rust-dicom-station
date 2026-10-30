//! The data tree of the left panel: the simulation and study sections at
//! its head, then one node per patient, study, modality and series, with
//! the transfer and modality menus of those rows. The sections below a
//! study (structures, segmentations, dose, ...) are in the `tree_*` files.

use super::tree_model::{PatientNode, StudyNode};
use super::*;

impl ViewerApp {
    pub(super) fn simulation_section(&mut self, ui: &mut egui::Ui) {
        if !self.slots.iter().any(|s| s.study.is_some()) {
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
                    for slot in self.open_slots() {
                        ui.selectable_value(&mut self.sim_source, slot, SLOT_NAMES[slot]);
                    }
                });
                // Where the copy lands: another open workspace, or the next
                // free letter. The default is whatever that rule offers
                // first, which with two open is the other one.
                let targets = self.copy_targets(self.sim_source);
                if !targets.contains(&self.sim_target) {
                    self.sim_target = targets.first().copied().unwrap_or(self.sim_source);
                }
                ui.horizontal(|ui| {
                    ui.label("▶ generates workspace");
                    for slot in &targets {
                        ui.selectable_value(&mut self.sim_target, *slot, SLOT_NAMES[*slot]);
                    }
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

                let src_ok = self.slots[self.sim_source.min(MAX_WORKSPACES - 1)].has_volume();
                if ui
                    .add_enabled(
                        src_ok && self.sim_target != self.sim_source && self.loading.is_none(),
                        egui::Button::new(format!(
                            "⚙ Generate transformed workspace ▶ {}",
                            SLOT_NAMES[self.sim_target]
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
        // A label, not a node. Which workspace a row belongs to is already
        // written on the panel by the order of the two blocks, so collapsing
        // one bought nothing and cost every row below it an indent; the
        // patients now sit at the same level as the name above them.
        let header = ui.add(
            egui::Label::new(
                egui::RichText::new(format!("Workspace {}", SLOT_NAMES[slot])).strong(),
            )
            .sense(egui::Sense::click()),
        );
        // Patient ▶ study ▶ category ▶ series. Every object read from a file
        // lives inside a study node - REG objects and treatment records too,
        // in their study's treatment sessions or its own lists; the setups
        // found here and the load's warnings stay at workspace level below
        // it. How dose is drawn is a module of the right panel, shared by
        // every workspace.
        self.data_tree(ui, slot);
        self.setups_section(ui, slot);
        self.warnings_section(ui, slot);
        // Right-click on the workspace header: clear the slot.
        let mut clear = false;
        header.context_menu(|ui| {
            if ui
                .button(format!("Clear workspace {}", SLOT_NAMES[slot]))
                .clicked()
            {
                clear = true;
                ui.close();
            }
        });
        if clear {
            self.clear_workspace(slot);
        }
        ui.separator();
    }

    /// The DICOM data tree of one workspace: patient ▶ study ▶ category ▶
    /// series, all visible at once.
    ///
    /// The nesting is rendered one level per method rather than as one deep
    /// stack of closures: each level hands the next a fresh `&mut self`
    /// reborrow that ends when the node's body returns, which is what lets a
    /// node three levels down still open a dialog or start a series switch.
    pub(super) fn data_tree(&mut self, ui: &mut egui::Ui, slot: usize) {
        if let Some((what, job)) = &self.phases_job {
            if what.slot == slot {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!("Copy to each phase: {}", job.progress.get()));
                    if ui.small_button("Cancel").clicked() {
                        job.progress.cancel();
                    }
                });
            }
        }
        let Some(layout) = self.tree_layout_cached(slot) else {
            return;
        };
        for (pi, patient) in layout.iter().enumerate() {
            let focused = self.tree_focus.as_ref()
                == Some(&RenameTarget::Patient {
                    slot,
                    key: patient.key.clone(),
                });
            let me = &mut *self;
            let resp = Self::wrapped_node_extra(
                ui,
                ("pat_hdr", slot, pi),
                true,
                patient.title.clone(),
                |ui| Self::keyboard_badge(ui, focused),
                |ui| me.patient_body(ui, slot, pi, patient),
            );
            let targets = self.copy_targets(slot);
            let open = self.open_slots();
            let key = patient.key.clone();
            let mut act: Option<TreeAction> = None;
            let mut rename = None;
            let mut focus = None;
            Self::focus_row(
                ui,
                &resp,
                RenameTarget::Patient {
                    slot,
                    key: key.clone(),
                },
                &self.tree_focus,
                &mut focus,
            );
            resp.context_menu(|ui| {
                if Self::rename_entry(ui) {
                    rename = Some(RenameTarget::Patient {
                        slot,
                        key: key.clone(),
                    });
                    ui.close();
                }
                Self::transfer_menu(ui, "patient", &targets, &open, |to, op| {
                    act = Some(TreeAction {
                        from: slot,
                        to,
                        sel: TreeSel::Patient(key.clone()),
                        op,
                    });
                });
                ui.separator();
                if Self::remove_entry(ui, "🗑 Remove") {
                    act = Some(TreeAction {
                        from: slot,
                        to: slot,
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
            if focus.is_some() {
                self.tree_focus = focus;
            }
        }
    }

    /// The copy / move pair of a data-tree context menu.
    ///
    /// The letters offered are the open workspaces bar this one, plus one
    /// new letter while there is room: with A alone that is "to B", with A
    /// and B it is B and C, and with four open there is nothing new to
    /// offer. One entry each while there is a single place to send it, a
    /// submenu of letters as soon as there is a choice.
    pub(super) fn transfer_menu(
        ui: &mut egui::Ui,
        what: &str,
        targets: &[usize],
        open: &[usize],
        mut act: impl FnMut(usize, TreeOp),
    ) {
        let name = |t: usize| {
            if open.contains(&t) {
                SLOT_NAMES[t].to_string()
            } else {
                format!("{} (new)", SLOT_NAMES[t])
            }
        };
        for (verb, op) in [("Copy", TreeOp::Copy), ("Move", TreeOp::Move)] {
            match targets {
                [] => {}
                [only] => {
                    if ui
                        .button(format!("{verb} {what} to workspace {}", name(*only)))
                        .clicked()
                    {
                        act(*only, op);
                        ui.close();
                    }
                }
                many => {
                    ui.menu_button(format!("{verb} {what} to workspace"), |ui| {
                        for t in many {
                            if ui.button(name(*t)).clicked() {
                                act(*t, op);
                                ui.close();
                            }
                        }
                    });
                }
            }
        }
    }

    /// The studies of one patient.
    fn patient_body(&mut self, ui: &mut egui::Ui, slot: usize, pi: usize, patient: &PatientNode) {
        for (si, node) in patient.studies.iter().enumerate() {
            let focused = self.tree_focus.as_ref()
                == Some(&RenameTarget::Study {
                    slot,
                    uid: node.uid.clone(),
                });
            let me = &mut *self;
            let resp = Self::wrapped_node_extra(
                ui,
                ("study_tree", slot, pi, si),
                true,
                node.title.clone(),
                |ui| Self::keyboard_badge(ui, focused),
                |ui| me.study_body(ui, slot, pi, si, node),
            );
            let resp = if node.uid.is_empty() {
                resp
            } else {
                resp.on_hover_text(format!("Study UID {}", node.uid))
            };
            let targets = self.copy_targets(slot);
            let open = self.open_slots();
            let uid = node.uid.clone();
            let mut act: Option<TreeAction> = None;
            let mut rename = None;
            let mut focus = None;
            let mut new_set: Option<SetKind> = None;
            // A new set is drawn on the series on screen, so it can only be
            // made under the study that series belongs to.
            let can_draw = self.slots[slot].has_volume() && self.study_displayed(slot, node);
            let no_draw = if !self.slots[slot].has_volume() {
                "This workspace has no image volume to draw on"
            } else {
                "Display one of this study's image series first - a new set is drawn on \
                 the series on screen"
            };
            Self::focus_row(
                ui,
                &resp,
                RenameTarget::Study {
                    slot,
                    uid: uid.clone(),
                },
                &self.tree_focus,
                &mut focus,
            );
            resp.context_menu(|ui| {
                for (kind, label, tip) in [
                    (
                        SetKind::Structures,
                        "➕ New RT structure set",
                        "An empty RT structure set, drawn on the displayed image series",
                    ),
                    (
                        SetKind::Segmentations,
                        "➕ New segmentation series",
                        "An empty segmentation series, drawn on the displayed image series - \
                         exports as one DICOM SEG file",
                    ),
                ] {
                    if enabled_tip_button(ui, can_draw, label, if can_draw { tip } else { no_draw })
                    {
                        new_set = Some(kind);
                        ui.close();
                    }
                }
                ui.separator();
                if Self::rename_entry(ui) {
                    rename = Some(RenameTarget::Study {
                        slot,
                        uid: uid.clone(),
                    });
                    ui.close();
                }
                Self::transfer_menu(ui, "study", &targets, &open, |to, op| {
                    act = Some(TreeAction {
                        from: slot,
                        to,
                        sel: TreeSel::Study(uid.clone()),
                        op,
                    });
                });
                ui.separator();
                if Self::remove_entry(ui, "🗑 Remove") {
                    act = Some(TreeAction {
                        from: slot,
                        to: slot,
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
            if focus.is_some() {
                self.tree_focus = focus;
            }
            if let Some(kind) = new_set {
                self.set_action = Some(SetAction::New(SetRef {
                    slot,
                    kind,
                    idx: SetRef::NEW,
                }));
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
        // images, reports - is a click away, so a tree of a
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
            let resp = Self::wrapped_node(ui, ("mod", slot, pi, si, mi), showing, title, |ui| {
                me.series_rows(ui, slot, idxs)
            });
            self.modality_menu(resp, slot, modality, idxs);
        }
        for &gi in &node.fourd {
            self.fourd_node(ui, slot, pi, si, gi);
        }
        // A study with no structure set or segmentation series has no such
        // category: a new one is made from the study's right-click menu.
        let displayed = self.study_displayed(slot, node);
        if !node.structs.is_empty() {
            self.structures_section(ui, slot, pi, si, &node.structs, displayed);
        }
        if !node.segs.is_empty() {
            self.segmentation_section(ui, slot, pi, si, &node.segs, displayed);
        }
        self.dose_section(ui, slot, pi, si, &node.doses);
        self.plan_section(ui, slot, pi, si, &node.plans);
        self.sessions_section(ui, slot, (pi, si), &node.sessions);
        self.planar_section(ui, slot, (pi, si), node);
        self.reg_objects_section(ui, slot, (pi, si), &node.regs);
        self.records_section(ui, slot, (pi, si), &node.records);
        self.reports_section(ui, slot, (pi, si), &node.reports);
    }

    /// Whether the images on screen are this study's. Everything that
    /// writes - the tools, "+ ROI", a new segmentation - works on the
    /// displayed series, so under another study of the same folder those
    /// buttons are held back instead of silently editing this one. A
    /// series filed in a 4D group has left its modality node, so the groups
    /// are asked as well: a 4DCT's structures were disabled whenever one of
    /// its phases was on screen, which is always. A daily CT has left its
    /// modality node for its treatment session.
    fn study_displayed(&self, slot: usize, node: &StudyNode) -> bool {
        let Some(st) = self.slots[slot].study.as_ref() else {
            return false;
        };
        let active = st.active_series;
        node.fourd.iter().any(|&gi| {
            st.fourd_groups
                .get(gi)
                .is_some_and(|g| g.resolve(&st.series).contains(&Some(active)))
        }) || node
            .modalities
            .iter()
            .any(|(_, idxs)| idxs.contains(&active))
            || node.sessions.iter().any(|s| s.series.contains(&active))
    }

    /// Right-click on a modality node: what can be done with every series
    /// under it at once. A 4DCT that arrived without phase markers in its
    /// descriptions is ten CT series in one node, and this is where they
    /// become one group in one click.
    fn modality_menu(&mut self, resp: egui::Response, slot: usize, modality: &str, idxs: &[usize]) {
        let n = idxs.len();
        let targets = self.copy_targets(slot);
        let open = self.open_slots();
        let group_names = self.group_names(slot);
        let what = format!("all {n} {modality} series");
        let mut act: Option<TreeAction> = None;
        let mut fourd: Option<FourDAction> = None;
        resp.context_menu(|ui| {
            if n >= 2 {
                if tip_button(
                    ui,
                    format!("🎞 New 4D group from {what}"),
                    "One custom 4D group of every series under this node, its phases in \
                     temporal order - by the percent in their descriptions, else by \
                     temporal position, else by series number",
                ) {
                    fourd = Some(FourDAction::NewFrom {
                        slot,
                        series: idxs.to_vec(),
                    });
                    ui.close();
                }
                for (gi, name) in &group_names {
                    if ui.button(format!("Add {what} to {name}")).clicked() {
                        fourd = Some(FourDAction::AddMany {
                            slot,
                            group: *gi,
                            series: idxs.to_vec(),
                        });
                        ui.close();
                    }
                }
                ui.separator();
            }
            Self::transfer_menu(ui, &what, &targets, &open, |to, op| {
                act = Some(TreeAction {
                    from: slot,
                    to,
                    sel: TreeSel::Many(idxs.to_vec()),
                    op,
                });
            });
            ui.separator();
            if ui.button(format!("🗑 Remove {what}")).clicked() {
                act = Some(TreeAction {
                    from: slot,
                    to: slot,
                    sel: TreeSel::Many(idxs.to_vec()),
                    op: TreeOp::Remove,
                });
                ui.close();
            }
        });
        if act.is_some() {
            self.tree_action = act;
        }
        if fourd.is_some() {
            self.fourd_action = fourd;
        }
    }

    /// The 4D groups of a workspace as (index, name), for the *Add to*
    /// entries of the series menus.
    fn group_names(&self, slot: usize) -> Vec<(usize, String)> {
        self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                st.fourd_groups
                    .iter()
                    .enumerate()
                    .filter(|(_, g)| !g.dissolved)
                    .map(|(gi, g)| (gi, g.name.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The image series of one modality node (or of a treatment session).
    pub(super) fn series_rows(&mut self, ui: &mut egui::Ui, slot: usize, idxs: &[usize]) {
        let targets = self.copy_targets(slot);
        let open = self.open_slots();
        let mut switch_to = None;
        let mut out = SeriesOut::default();
        let mut focus = None;
        let group_names = self.group_names(slot);
        let badges = self.series_badges(slot);
        // What Ctrl-click has ticked in this workspace: those rows are
        // marked, and a right-click on any of them acts on all of them.
        let picked = self.picked_series(slot);
        let (ctrl, shift) = ui.input(|i| (i.modifiers.command, i.modifiers.shift));
        {
            let Some(study) = self.slots[slot].study.as_ref() else {
                return;
            };
            let active = study.active_series;
            for &i in idxs {
                let Some(s) = study.series.get(i) else {
                    continue;
                };
                let is_picked = picked.contains(&i);
                let label = format!(
                    "{}{}{} ({} sl.)",
                    if is_picked { "✔ " } else { "" },
                    if s.description.is_empty() {
                        "series"
                    } else {
                        &s.description
                    },
                    Self::name_ordinal(study, i),
                    s.files.len()
                );
                // What the current selection has put on this series, right
                // where the series is named.
                let me = RenameTarget::Series { slot, idx: i };
                let resp = Self::badged_series_row(
                    ui,
                    i == active,
                    label,
                    &s.uid,
                    &badges,
                    self.tree_focus.as_ref() == Some(&me),
                );
                Self::focus_row(ui, &resp, me, &self.tree_focus, &mut focus);
                if resp.clicked() {
                    // Ctrl-click ticks the row for a multi-series action,
                    // Shift-click ticks a range, and both leave the display
                    // alone; a plain click shows it.
                    if shift {
                        out.range = Some(i);
                    } else if ctrl {
                        out.pick = Some(s.uid.clone());
                    } else if i != active {
                        switch_to = Some(i);
                    }
                }
                let menu = SeriesMenu::new(slot, i, &picked);
                let resp = resp.on_hover_text(Self::series_tip(s));
                resp.context_menu(|ui| {
                    if i != active {
                        if ui.button("👁 Display").clicked() {
                            switch_to = Some(i);
                            ui.close();
                        }
                        ui.separator();
                    }
                    ui.menu_button("🎞 4D group", |ui| {
                        menu.fourd_entries(ui, &group_names, &mut out);
                    });
                    ui.separator();
                    menu.common_entries(ui, &targets, &open, &mut out);
                });
            }
        }
        if let Some(i) = switch_to {
            self.start_series_switch(slot, i);
        }
        self.apply_series_out(slot, idxs, out, focus);
    }

    /// What a series row's click and menu asked for, written back once the
    /// rows are drawn.
    pub(super) fn apply_series_out(
        &mut self,
        slot: usize,
        level: &[usize],
        out: SeriesOut,
        focus: Option<RenameTarget>,
    ) {
        if out.act.is_some() {
            self.tree_action = out.act;
        }
        if out.rename.is_some() {
            self.rename_request = out.rename;
        }
        if focus.is_some() {
            self.tree_focus = focus;
        }
        if out.fourd.is_some() {
            self.fourd_action = out.fourd;
        }
        if let Some(uid) = out.pick {
            self.toggle_pick(slot, &uid);
        }
        if let Some(i) = out.range {
            self.pick_range(slot, level, i);
        }
    }
}

/// What the rows of image series (plain, a 4D group's phases, a session's
/// daily images) hand back: the tree's usual deferred actions.
#[derive(Default)]
pub(super) struct SeriesOut {
    pub(super) act: Option<TreeAction>,
    pub(super) rename: Option<RenameTarget>,
    pub(super) fourd: Option<FourDAction>,
    /// Ctrl-click: tick or untick this series.
    pub(super) pick: Option<String>,
    /// Shift-click: tick the range up to this series.
    pub(super) range: Option<usize>,
}

/// The menu of one image-series row: which series it acts on - the row's,
/// or every ticked one when the row is ticked with others.
pub(super) struct SeriesMenu {
    slot: usize,
    row: usize,
    many: Vec<usize>,
}

impl SeriesMenu {
    pub(super) fn new(slot: usize, row: usize, picked: &[usize]) -> Self {
        // Right-clicking a ticked row acts on every ticked row;
        // right-clicking an unticked one acts on that row alone.
        let many = if picked.contains(&row) && picked.len() > 1 {
            picked.to_vec()
        } else {
            vec![row]
        };
        SeriesMenu { slot, row, many }
    }

    fn what(&self) -> String {
        if self.many.len() > 1 {
            format!("the {} ticked series", self.many.len())
        } else {
            "series".to_string()
        }
    }

    fn sel(&self) -> TreeSel {
        if self.many.len() > 1 {
            TreeSel::Many(self.many.clone())
        } else {
            TreeSel::Series(self.row)
        }
    }

    /// The *4D group* submenu of a series that is in none: add it to one,
    /// make one of it, re-detect.
    pub(super) fn fourd_entries(
        &self,
        ui: &mut egui::Ui,
        group_names: &[(usize, String)],
        out: &mut SeriesOut,
    ) {
        let (slot, i, what) = (self.slot, self.row, self.what());
        for (gi, name) in group_names {
            if ui.button(format!("Add {what} to {name}")).clicked() {
                out.fourd = Some(if self.many.len() > 1 {
                    FourDAction::AddMany {
                        slot,
                        group: *gi,
                        series: self.many.clone(),
                    }
                } else {
                    FourDAction::Add {
                        slot,
                        group: *gi,
                        series: i,
                    }
                });
                ui.close();
            }
        }
        if self.many.len() > 1 {
            if tip_button(
                ui,
                format!("New 4D group from {what}"),
                "One custom group of the ticked series, its phases in temporal order - by \
                 the percent in their descriptions, else by temporal position, else by series \
                 number, else in the order they were ticked",
            ) {
                out.fourd = Some(FourDAction::NewFrom {
                    slot,
                    series: self.many.clone(),
                });
                ui.close();
            }
        } else if ui.button("New 4D group from this series").clicked() {
            out.fourd = Some(FourDAction::New { slot, series: i });
            ui.close();
        }
        if ui.button("Re-detect 4D groups").clicked() {
            out.fourd = Some(FourDAction::Redetect { slot });
            ui.close();
        }
    }

    /// The entries every series row ends with: rename, copy / move, remove.
    pub(super) fn common_entries(
        &self,
        ui: &mut egui::Ui,
        targets: &[usize],
        open: &[usize],
        out: &mut SeriesOut,
    ) {
        let (slot, what) = (self.slot, self.what());
        if ViewerApp::rename_entry(ui) {
            out.rename = Some(RenameTarget::Series {
                slot,
                idx: self.row,
            });
            ui.close();
        }
        ViewerApp::transfer_menu(ui, &what, targets, open, |to, op| {
            out.act = Some(TreeAction {
                from: slot,
                to,
                sel: self.sel(),
                op,
            });
        });
        ui.separator();
        if ViewerApp::remove_entry(ui, format!("🗑 Remove {what}")) {
            out.act = Some(TreeAction {
                from: slot,
                to: slot,
                sel: self.sel(),
                op: TreeOp::Remove,
            });
            ui.close();
        }
    }
}
