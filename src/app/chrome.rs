//! Window chrome: the menu bar, the toolbar and the status bar.

use super::*;

impl ViewerApp {
    // -- Menu bar ---------------------------------------------------------
    pub(super) fn menu_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut open_a = false;
        let mut open_b = false;
        let mut files_a = false;
        let mut files_b = false;
        let mut close_b = false;
        let mut reset_views = false;
        let mut open_gen = false;
        let mut open_models = false;
        let mut open_pacs = false;
        let mut open_propagate = false;
        let mut open_drr = false;
        let mut open_export: Option<usize> = None;
        let mut new_theme: Option<egui::ThemePreference> = None;
        // A module was switched on or off — remember it for the next run.
        let mut modules_changed = false;

        egui::Panel::top(egui::Id::new("menu_bar")).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui
                        .button("📂 Add DICOM folder to A…")
                        .on_hover_text(
                            "Scan a folder and add its patients / studies / series to \
                             dataset A (existing content stays loaded)",
                        )
                        .clicked()
                    {
                        open_a = true;
                        ui.close();
                    }
                    if ui
                        .button("📂 Add DICOM folder to B…")
                        .on_hover_text(
                            "Scan a folder and add its patients / studies / series to \
                             dataset B (existing content stays loaded)",
                        )
                        .clicked()
                    {
                        open_b = true;
                        ui.close();
                    }
                    ui.separator();
                    // Individual files, for the objects that do not come as a
                    // folder of slices: an RT image, a structure set, a plan,
                    // a single slice. They merge exactly as a folder does.
                    if ui
                        .button("📄 Add DICOM file(s) to A…")
                        .on_hover_text(
                            "Open one or more DICOM files directly — RT images, a \
                             structure set, a plan, single slices. They do not have to \
                             form an image volume",
                        )
                        .clicked()
                    {
                        files_a = true;
                        ui.close();
                    }
                    if ui
                        .button("📄 Add DICOM file(s) to B…")
                        .on_hover_text(
                            "Open one or more DICOM files directly — RT images, a \
                             structure set, a plan, single slices. They do not have to \
                             form an image volume",
                        )
                        .clicked()
                    {
                        files_b = true;
                        ui.close();
                    }
                    ui.separator();
                    let has_a = self.slots[0].study.is_some();
                    if ui
                        .add_enabled(has_a, egui::Button::new("Clear dataset A"))
                        .clicked()
                    {
                        self.tree_clear_slot(0);
                        ui.close();
                    }
                    let has_b = self.slots[1].study.is_some();
                    if ui
                        .add_enabled(has_b, egui::Button::new("Close dataset B"))
                        .clicked()
                    {
                        close_b = true;
                        ui.close();
                    }
                    ui.separator();
                    for (slot, slot_name) in SLOT_NAMES.iter().enumerate() {
                        if slot == 1 && !self.comparison && self.slots[1].study.is_none() {
                            continue;
                        }
                        if ui
                            .add_enabled(
                                self.slots[slot].study.is_some(),
                                egui::Button::new(format!(
                                    "💾 Export dataset {slot_name} as DICOM…"
                                )),
                            )
                            .on_hover_text(
                                "Write the displayed volume, structure sets, \
                                 segmentation series (DICOM SEG), dose grids and plans \
                                 as DICOM files — with the patient / study / equipment \
                                 tags reviewed and edited first",
                            )
                            .clicked()
                        {
                            open_export = Some(slot);
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui
                        .button("📐 Generate test data…")
                        .on_hover_text(
                            "Write a complete synthetic RT study (CT, RTSTRUCT, RTPLAN, \
                             RTDOSE, DX, RTIMAGE, REG, RTRECORD) into the application folder",
                        )
                        .clicked()
                    {
                        open_gen = true;
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui
                        .checkbox(&mut self.comparison, "Comparison mode (2 × 3 views)")
                        .clicked()
                    {
                        ui.close();
                    }
                    ui.separator();
                    ui.checkbox(&mut self.show_contours, "Contours");
                    ui.checkbox(&mut self.show_crosshair, "Crosshair");
                    // Syncing is a property of the crosshair, so it sits under
                    // it and goes away with it.
                    if self.show_crosshair {
                        let both = self.slots[0].has_volume() && self.slots[1].has_volume();
                        ui.add_enabled(
                            both,
                            egui::Checkbox::new(
                                &mut self.link_studies,
                                "Sync crosshairs between datasets",
                            ),
                        )
                        .on_hover_text(
                            "Move one crosshair and the other follows to the same patient \
                             point — through the active registration when there is one. \
                             Off, each dataset is navigated on its own.",
                        );
                    }
                    ui.checkbox(&mut self.show_labels, "Orientation labels");
                    ui.checkbox(&mut self.show_isocenters, "Isocenters");
                    ui.separator();
                    ui.checkbox(&mut self.side_open, "Left panel (F9)")
                        .on_hover_text(
                            "Hide the left panel and give the whole window to the views. \
                             The arrow on the window's left edge brings it back, as does \
                             F9.",
                        );
                    ui.separator();
                    ui.label("Appearance:");
                    let before = self.theme;
                    self.theme.radio_buttons(ui);
                    if self.theme != before {
                        new_theme = Some(self.theme);
                    }
                    if let Some(msg) = &self.settings_error {
                        ui.weak(msg);
                    }
                    ui.separator();
                    if ui.button("Reset all views").clicked() {
                        reset_views = true;
                        ui.close();
                    }
                });
                // This menu is a set of switches, not a list of actions:
                // it stays open until the pointer leaves it, so both
                // modules can be turned on in one visit.
                egui::containers::menu::MenuButton::new("Modules")
                    .config(
                        egui::containers::menu::MenuConfig::new()
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                    )
                    .ui(ui, |ui| {
                        // Two optional side-panel sections, each one line of the
                        // menu. Everything they do — direction, method, region,
                        // parameters, landmarks, analytics, fusion, the vector
                        // field, the simulated motion — lives in the section
                        // itself, so the menu only decides whether it is there.
                        ui.weak("Sections of the left panel:");
                        modules_changed |= ui
                            .checkbox(&mut self.module_registration, "Image registration")
                            .on_hover_text(
                                "Align two datasets: direction, method, region, parameters, \
                             landmarks, analysis, fusion and the deformation vector field. \
                             Needs two loaded datasets to run.",
                            )
                            .changed();
                        modules_changed |= ui
                            .checkbox(&mut self.module_simulation, "Image simulation")
                            .on_hover_text(
                                "Registration QA: apply a known rigid motion and Gaussian \
                             deformation to one dataset and generate the result into the \
                             other — the ground truth a registration can be measured against.",
                            )
                            .changed();
                    });
                ui.menu_button("Tools", |ui| {
                    // One block per dataset: the same six tools, in the same
                    // order, for A and B.
                    let tools: [(&ToolInfo, &str); 6] = [
                        (
                            &COMBINE,
                            "Build one structure out of others: union, intersection, \
                             subtraction or symmetric difference, with a margin on any of \
                             them. Contours and segmentations mix freely.",
                        ),
                        (
                            &BODY_CONTOUR,
                            "Outline the patient and leave the couch, the chair and the \
                             immobilisation outside — the EXTERNAL structure. Works on CT \
                             and MR, with or without a network.",
                        ),
                        (
                            &AUTOSEG,
                            "Automatic multi-organ segmentation of the displayed CT \
                             (TotalSegmentator's nnU-Net models, re-implemented natively \
                             in Rust; runs locally on CPU or GPU)",
                        ),
                        (
                            &PROMPT_SEG,
                            "Segment whatever you point at — a box, a click or a \
                             structure name (SegVol, re-implemented natively in Rust). \
                             Covers the lesions and targets a fixed-class model cannot.",
                        ),
                        (
                            &SLICE_PROP,
                            "Box a structure on one slice and follow it through the \
                             stack at full in-plane resolution (MedSAM2, re-implemented \
                             natively in Rust).",
                        ),
                        (
                            &MOTION,
                            "Register the reference phase of a 4D group to every other \
                             phase, carry the targets across, and measure their motion — \
                             trajectories, drift, correlations and the ITV.",
                        ),
                    ];
                    let mut open_tool: Option<(usize, &ToolInfo)> = None;
                    for slot in 0..SLOT_NAMES.len() {
                        if slot == 1 && !self.comparison {
                            continue;
                        }
                        if slot == 1 {
                            ui.separator();
                        }
                        // Every engine reads voxels: a dataset that holds
                        // only RT images or RT objects has nothing to give
                        // them.
                        let loaded = self.slots[slot].has_volume();
                        for (tool, hint) in tools {
                            if ui
                                .add_enabled(loaded, egui::Button::new(tool.menu_entry(slot)))
                                .on_hover_text(hint)
                                .clicked()
                            {
                                open_tool = Some((slot, tool));
                                ui.close();
                            }
                        }
                    }
                    match open_tool {
                        Some((slot, t)) if t.glyph == COMBINE.glyph => {
                            self.open_combine_dialog(slot, Vec::new())
                        }
                        Some((slot, t)) if t.glyph == MOTION.glyph => {
                            self.open_motion_dialog(slot, None)
                        }
                        Some((slot, t)) if t.glyph == BODY_CONTOUR.glyph => {
                            self.open_body_dialog(slot)
                        }
                        Some((slot, t)) if t.glyph == AUTOSEG.glyph => {
                            self.open_autoseg_dialog(slot)
                        }
                        Some((slot, t)) if t.glyph == PROMPT_SEG.glyph => {
                            self.open_segvol_dialog(slot)
                        }
                        Some((slot, _)) => self.open_medsam2_panel(slot),
                        None => {}
                    }
                    ui.separator();
                    let both = self.slots[0].has_volume() && self.slots[1].has_volume();
                    if ui
                        .add_enabled(both, egui::Button::new("⇄ Propagate structures…"))
                        .on_hover_text(
                            "Carry contours and segmentations from one dataset to the \
                             other through the active registration — globally, or refined \
                             on an enclosing structure first",
                        )
                        .clicked()
                    {
                        open_propagate = true;
                        ui.close();
                    }
                    let both = self.slots[0].has_volume() && self.slots[1].has_volume();
                    if ui
                        .add_enabled(both, egui::Button::new("◎ Transfer by relationship…"))
                        .on_hover_text(
                            "Place a structure into the other dataset at the same offset \
                             from a reference structure (e.g. the heart) — the \
                             target–reference relationship travels, not a registration",
                        )
                        .clicked()
                    {
                        self.open_transfer_dialog(0);
                        ui.close();
                    }
                    let any = self.slots[0].has_volume() || self.slots[1].has_volume();
                    if ui
                        .add_enabled(any, egui::Button::new("◑ Compare structures…"))
                        .on_hover_text(
                            "Volumes, centroid offset, Dice, HD95 and mean surface \
                             distance of any two structures — within a dataset or across \
                             the two",
                        )
                        .clicked()
                    {
                        self.open_compare_dialog(0);
                        ui.close();
                    }
                    let has_dose = self
                        .slots
                        .iter()
                        .any(|s| s.study.as_ref().is_some_and(|st| !st.doses.is_empty()));
                    if ui
                        .add_enabled(has_dose, egui::Button::new("📊 Dose–volume histograms…"))
                        .on_hover_text(
                            "Cumulative and differential DVHs of any structures against \
                             any loaded dose objects, with the metrics table, protocol \
                             constraint checking and CSV export — in a window that can \
                             go on its own monitor",
                        )
                        .clicked()
                    {
                        let slot = usize::from(
                            self.slots[0].study.is_none()
                                || self.slots[0]
                                    .study
                                    .as_ref()
                                    .is_some_and(|s| s.doses.is_empty()),
                        );
                        self.open_dvh_dialog(slot.min(1), Vec::new());
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            !self.motion_reports.is_empty(),
                            egui::Button::new("📈 Motion results…"),
                        )
                        .on_hover_text("The finished 4D motion runs of this session")
                        .clicked()
                    {
                        self.motion_results_open = true;
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.slots[0].has_volume() || self.slots[1].has_volume(),
                            egui::Button::new("☢ Digitally reconstructed radiograph…"),
                        )
                        .on_hover_text(
                            "Forward-project the CT onto a flat detector — two \
                             independent projectors, an exact ray tracer and an \
                             interpolating one, with the difference between them",
                        )
                        .clicked()
                    {
                        open_drr = true;
                        ui.close();
                    }
                    if ui
                        .button("🏥 PACS — patient archive…")
                        .on_hover_text(
                            "The local archive: every study filed here, ready to be taken \
                             into a dataset and given back the structures and \
                             segmentations drawn on it",
                        )
                        .clicked()
                    {
                        open_pacs = true;
                        ui.close();
                    }
                    if ui
                        .button("📦 Downloaded models…")
                        .on_hover_text(
                            "What every segmentation engine has downloaded, how much disk \
                             it costs, and the buttons to download, update or remove it — \
                             one model at a time or all of them",
                        )
                        .clicked()
                    {
                        open_models = true;
                        ui.close();
                    }
                    if ui
                        .button("🔏 Anonymize DICOM folder…")
                        .on_hover_text(
                            "Scan a folder, review every identifying tag with its current \
                             and proposed values, then rewrite the files (in place or into \
                             a new folder) with consistently regenerated UIDs",
                        )
                        .clicked()
                    {
                        self.anon_open = true;
                        ui.close();
                    }
                });
                ui.menu_button("Help", |ui| {
                    ui.label("MPR views — mouse:");
                    ui.weak("Left click / drag — move linked crosshair");
                    ui.weak("Mouse wheel — scroll slices");
                    ui.weak("Ctrl + wheel / pinch — zoom at cursor");
                    ui.weak("Middle drag — pan");
                    ui.weak("Right drag — window / level (x = width, y = center)");
                    ui.separator();
                    ui.label("Segmentation (🎨 ⊖ ✨ take over the left button):");
                    ui.weak("Left drag — paint / erase");
                    ui.weak("Left press + drag ↑↓ — grow / shrink the region (✨)");
                    ui.weak("Alt — erase while painting");
                    ui.weak("Shift + wheel, or [ ] — brush radius");
                    ui.weak("Ctrl + Z — undo the last stroke");
                    ui.weak("Esc — cancel the running region grow");
                    ui.separator();
                    ui.label(format!(
                        "{} (the box takes over the left button in its view):",
                        SLICE_PROP.name
                    ));
                    ui.weak(
                        "Left drag — draw the box; drag a corner to resize, the middle to move",
                    );
                    ui.weak("Left click — an include / exclude point, with ➕ / ➖ chosen");
                    ui.separator();
                    ui.label("Buttons:");
                    ui.weak("⟲ (view corner) — reset that view's zoom, pan and slice");
                    ui.weak("⛶ / ⊞ — maximize that view / restore the layout");
                    ui.weak("⟲ (toolbar) — reset every view of both datasets");
                    ui.weak(
                        "⌖ — show / hide the crosshair; hidden, left click no \
                         longer navigates",
                    );
                    ui.weak("🔗 Sync — sync the crosshairs of A and B (shown while ⌖ is on)");
                    ui.separator();
                    ui.weak(format!(
                        "rust-dicom-station {} — research / QA viewer, not a medical device",
                        env!("CARGO_PKG_VERSION")
                    ));
                });
            });
        });

        if open_a {
            if let Some(dir) = Self::pick_folder("Select DICOM folder to add to dataset A") {
                self.start_load(0, dir);
            }
        }
        if open_b {
            if let Some(dir) = Self::pick_folder("Select DICOM folder to add to dataset B") {
                self.comparison = true;
                self.start_load(1, dir);
            }
        }
        if files_a {
            if let Some(paths) = Self::pick_files("Select DICOM file(s) to add to dataset A") {
                self.start_load_files(0, paths);
            }
        }
        if files_b {
            if let Some(paths) = Self::pick_files("Select DICOM file(s) to add to dataset B") {
                self.comparison = true;
                self.start_load_files(1, paths);
            }
        }
        if close_b {
            self.close_comparison();
        }
        if reset_views {
            self.reset_all_views();
        }
        if open_gen {
            self.gen_open = true;
        }
        if open_pacs {
            self.open_pacs_window();
        }
        if open_models {
            self.open_models_window();
        }
        if open_drr {
            let slot = usize::from(!self.slots[0].has_volume());
            self.open_drr_window(slot);
        }
        if open_propagate {
            let src = self
                .registration
                .as_ref()
                .map(|r| r.fixed_slot)
                .unwrap_or(0);
            self.open_propagate_window(src);
        }
        if let Some(slot) = open_export {
            self.open_export_dialog(slot);
        }
        if let Some(theme) = new_theme {
            self.set_theme(ctx, theme);
        }
        if modules_changed {
            self.persist_settings();
        }
    }

    // -- Toolbar ----------------------------------------------------------
    pub(super) fn top_bar(&mut self, ui: &mut egui::Ui) {
        // Only the primary reading controls live here (window/level);
        // file actions, display toggles and appearance are in the menus.
        let any_study = self.slots[0].study.is_some() || self.slots[1].study.is_some();
        if !any_study {
            return;
        }
        egui::Panel::top(egui::Id::new("top_bar")).show(ui, |ui| {
            ui.horizontal(|ui| {
                {
                    ui.label("W/L:");
                    ui.add(
                        egui::DragValue::new(&mut self.window_center)
                            .speed(2.0)
                            .prefix("C "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.window_width)
                            .speed(4.0)
                            .range(1.0..=20000.0)
                            .prefix("W "),
                    );
                    let mut full_range = false;
                    // The closed combo carries the name of the preset in
                    // force. Its numbers are dropped there — the two drag
                    // values to the left already show them — but kept in the
                    // list, where they are what tells the presets apart.
                    // Any other window (a drag, a right-drag in a view, the
                    // full range) is nameless again.
                    self.wl_preset = self.wl_preset.filter(|i| {
                        WL_PRESETS.get(*i).is_some_and(|(_, c, w)| {
                            *c == self.window_center && *w == self.window_width
                        })
                    });
                    let selected = self
                        .wl_preset
                        .and_then(|i| WL_PRESETS.get(i))
                        .map_or("CT presets", |(name, ..)| *name);
                    let mut pick: Option<usize> = None;
                    let current = self.wl_preset;
                    egui::ComboBox::from_id_salt("wl_preset")
                        .selected_text(selected)
                        .width(150.0)
                        .show_ui(ui, |ui| {
                            for (i, (name, c, w)) in WL_PRESETS.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        current == Some(i),
                                        format!("{name}  (C {c:.0} / W {w:.0})"),
                                    )
                                    .clicked()
                                {
                                    pick = Some(i);
                                }
                            }
                            ui.separator();
                            if ui.button("Full range").clicked() {
                                full_range = true;
                            }
                        });
                    if let Some(i) = pick {
                        let (_, c, w) = WL_PRESETS[i];
                        self.window_center = c;
                        self.window_width = w;
                        self.wl_preset = Some(i);
                    }
                    if full_range {
                        // Read the range off a dataset that has one; an empty
                        // volume would otherwise set the shared window to the
                        // degenerate C 0 / W 1 and blank the other dataset.
                        let src = [self.hovered_slot.min(1), 1 - self.hovered_slot.min(1)]
                            .into_iter()
                            .find(|s| self.slots[*s].has_volume());
                        if let Some(study) = src.and_then(|s| self.slots[s].study.as_ref()) {
                            self.wl_preset = None;
                            let v = &study.volume;
                            self.window_center = (v.min_value as f32 + v.max_value as f32) * 0.5;
                            self.window_width = (v.max_value as f32 - v.min_value as f32).max(1.0);
                        }
                    }

                    ui.separator();
                    // 3D structure rendering windows.
                    for (slot, slot_name) in SLOT_NAMES.iter().enumerate() {
                        let has_3d = self.slots[slot]
                            .study
                            .as_ref()
                            .map(|s| !s.structure_sets.is_empty())
                            .unwrap_or(false)
                            || !self.slots[slot].segs().is_empty();
                        if slot == 1 && self.slots[1].study.is_none() {
                            continue;
                        }
                        if ui
                            .add_enabled(has_3d, egui::Button::new(format!("3D {slot_name}")))
                            .on_hover_text(format!(
                                "Open a 3D surface rendering of dataset {slot_name}'s structures \
                                 and segmentations"
                            ))
                            .clicked()
                        {
                            self.open_d3_window(slot);
                        }
                    }

                    // Slice-intersection (crosshair) toggle. With the
                    // crosshair hidden, left-click navigation is disabled.
                    if ui
                        .selectable_label(self.show_crosshair, "⌖")
                        .on_hover_text(
                            "Show / hide the slice intersection (crosshair).\n\
                             Hidden: left click does not navigate — slices change \
                             only by scrolling each view",
                        )
                        .clicked()
                    {
                        self.show_crosshair = !self.show_crosshair;
                    }

                    // Crosshair syncing: only meaningful while there is a
                    // crosshair, so it appears and disappears with it.
                    if self.show_crosshair {
                        let both = self.slots[0].has_volume() && self.slots[1].has_volume();
                        if ui
                            .add_enabled(
                                both,
                                egui::Button::selectable(self.link_studies, "🔗 Sync"),
                            )
                            .on_hover_text(
                                "Sync the crosshairs of datasets A and B: move one and the \
                                 other follows to the same patient point, through the active \
                                 registration when there is one.\n\
                                 Off: each dataset is navigated on its own",
                            )
                            .clicked()
                        {
                            self.link_studies = !self.link_studies;
                        }
                    }

                    // Reset every view of both datasets.
                    if ui
                        .button("⟲")
                        .on_hover_text(
                            "Reset every view of both datasets: fit zoom, clear pan \
                             and put the crosshairs back at the volume centers",
                        )
                        .clicked()
                    {
                        self.reset_all_views();
                    }

                    // Segmentation tools. Selecting a tool takes over the
                    // left mouse button in the MPR views.
                    ui.separator();
                    // Nothing to paint on in a dataset with no image volume.
                    let paintable = self.slots.iter().any(|s| s.has_volume());
                    if !paintable {
                        self.seg_tool = SegTool::None;
                    }
                    let mut pick = |ui: &mut egui::Ui, tool: SegTool, label: &str, tip: &str| {
                        if ui
                            .add_enabled(
                                paintable,
                                egui::Button::selectable(self.seg_tool == tool, label),
                            )
                            .on_hover_text(tip)
                            .clicked()
                        {
                            self.seg_tool = if self.seg_tool == tool {
                                SegTool::None
                            } else {
                                tool
                            };
                            if self.seg_tool != SegTool::Grow {
                                self.cancel_grow();
                            }
                        }
                    };
                    pick(
                        ui,
                        SegTool::Brush,
                        "🎨 Paint",
                        "Paint the active segmentation (LMB drag).\n\
                         Hold Alt to erase · Shift+wheel or [ ] resize the brush · Ctrl+Z undo",
                    );
                    pick(
                        ui,
                        SegTool::Erase,
                        "⊖ Erase",
                        "Erase from the active segmentation (LMB drag)",
                    );
                    pick(
                        ui,
                        SegTool::Grow,
                        "✨ Grow",
                        "Interactive organ segmentation (geodesic fast marching): press \
                         to place a seed, drag up/down to grow/shrink the region with a \
                         live preview. Intensity changes and edges act as barriers, so \
                         the organ under the seed is suggested before anything leaks. \
                         Release commits (enclosed holes are filled), Esc cancels",
                    );
                    if matches!(self.seg_tool, SegTool::Brush | SegTool::Erase) {
                        ui.add(
                            egui::DragValue::new(&mut self.brush_radius_mm)
                                .speed(0.5)
                                .range(0.5..=80.0)
                                .suffix(" mm"),
                        )
                        .on_hover_text("Brush radius");
                        if ui
                            .selectable_label(self.brush_3d, "3D")
                            .on_hover_text(
                                "Spherical 3D brush: paints through neighboring slices.\n\
                                 Off: flat 2D circle on the displayed slice only",
                            )
                            .clicked()
                        {
                            self.brush_3d = !self.brush_3d;
                        }
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut parts = Vec::new();
                    for (i, s) in self.slots.iter().enumerate() {
                        if let Some(study) = &s.study {
                            let m = &study.meta;
                            parts.push(format!(
                                "{}: {} {}",
                                SLOT_NAMES[i],
                                m.patient_name.replace('^', " "),
                                m.study_date
                            ));
                        }
                    }
                    ui.label(egui::RichText::new(parts.join("   ")).weak());
                });
            });
        });
    }

    // -- Status bar -------------------------------------------------------
    pub(super) fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom(egui::Id::new("status")).show(ui, |ui| {
            ui.horizontal(|ui| {
                let any = self.slots.iter().any(|s| s.study.is_some());
                if !any {
                    ui.weak("No data loaded");
                    return;
                }
                for (slot, slot_name) in SLOT_NAMES.iter().enumerate() {
                    if slot == 1 && !self.comparison {
                        // Study B is hidden while comparison mode is off.
                        continue;
                    }
                    let s = &self.slots[slot];
                    let Some(study) = &s.study else { continue };
                    if !study.has_volume() {
                        // No voxels, so no position and no value to report.
                        let prefix = if self.comparison && self.slots[1].study.is_some() {
                            format!("{slot_name}: ")
                        } else {
                            String::new()
                        };
                        ui.weak(format!("{prefix}no image volume"));
                        continue;
                    }
                    let v = &study.volume;
                    let c = s.cursor;
                    let p = v.voxel_to_patient(c[0], c[1], c[2]);
                    let both = self.comparison && self.slots[1].study.is_some();
                    let prefix = if both {
                        format!("{slot_name}: ")
                    } else {
                        String::new()
                    };
                    // Both datasets report in full: each one's own cursor is
                    // a real position in its own volume, whether it was
                    // clicked there or followed the other one.
                    ui.monospace(format!(
                        "{}({:6.1},{:6.1},{:6.1})mm ijk({:3},{:3},{:3})",
                        prefix,
                        p.x,
                        p.y,
                        p.z,
                        c[0].round() as i64,
                        c[1].round() as i64,
                        c[2].round() as i64
                    ));
                    if let Some(hu) =
                        v.get(c[0].round() as i64, c[1].round() as i64, c[2].round() as i64)
                    {
                        ui.monospace(format!("{hu:5} HU"));
                    }
                    if let Some(d) = study
                        .doses
                        .get(s.active_dose)
                        .and_then(|d| d.sample(p))
                    {
                        ui.monospace(format!(
                            "{:.2} Gy ({:.0}%)",
                            d,
                            100.0 * d / s.dose_reference.max(1e-6)
                        ));
                    }
                    if both && slot == 0 {
                        ui.separator();
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // The readouts are what the bar is for, so the mouse
                    // bindings fold into a single "?" that the pointer opens
                    // — always the bindings of the tool in force.
                    let hint = match self.seg_tool {
                        SegTool::None => {
                            "LMB crosshair · RMB W/L · MMB pan · wheel slice · Ctrl+wheel zoom"
                        }
                        SegTool::Brush => {
                            "LMB paint · Alt erase · Shift+wheel / [ ] brush size · Ctrl+Z undo · wheel slice"
                        }
                        SegTool::Erase => {
                            "LMB erase · Shift+wheel / [ ] brush size · Ctrl+Z undo · wheel slice"
                        }
                        SegTool::Grow => {
                            "LMB press seed · drag up/down = grow/shrink · release commit · Esc cancel · Ctrl+Z undo"
                        }
                    };
                    // `Sense::hover`: it looks like a button and answers the
                    // pointer, but there is nothing to click.
                    ui.add(egui::Button::new("?").small().sense(egui::Sense::hover()))
                        .on_hover_text(hint);
                });
            });
        });
    }
}
