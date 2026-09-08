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
        let mut open_gen = false;
        let mut open_models = false;
        let mut open_pacs = false;
        let mut open_drr = false;
        let mut open_export = false;
        let mut new_theme: Option<egui::ThemePreference> = None;
        let mut save_settings = false;
        // A module was switched on or off - remember it for the next run.
        let mut modules_changed = false;

        egui::Panel::top(egui::Id::new("menu_bar")).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if tip_button(
                        ui,
                        "📂 Add DICOM folder to A",
                        "Scan a folder and add its patients / studies / series to \
                         dataset A (existing content stays loaded)",
                    ) {
                        open_a = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "📂 Add DICOM folder to B",
                        "Scan a folder and add its patients / studies / series to \
                         dataset B (existing content stays loaded)",
                    ) {
                        open_b = true;
                        ui.close();
                    }
                    ui.separator();
                    // Individual files, for the objects that do not come as a
                    // folder of slices: an RT image, a structure set, a plan,
                    // a single slice. They merge exactly as a folder does.
                    if tip_button(
                        ui,
                        "📄 Add DICOM file(s) to A",
                        "Open one or more DICOM files directly - RT images, a \
                         structure set, a plan, single slices. They do not have to \
                         form an image volume",
                    ) {
                        files_a = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "📄 Add DICOM file(s) to B",
                        "Open one or more DICOM files directly - RT images, a \
                         structure set, a plan, single slices. They do not have to \
                         form an image volume",
                    ) {
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
                    // One export for everything that is loaded: which
                    // patients, studies and series go out is chosen in the
                    // window, not by which menu entry was clicked.
                    let anything = self.slots[0].study.is_some() || self.slots[1].study.is_some();
                    if enabled_tip_button(
                        ui,
                        anything,
                        "💾 Export DICOM",
                        "Write any patients, studies, series and RT objects of either \
                         dataset as DICOM - with every name and UID shown and editable, \
                         structures as RTSTRUCT or SEG, and the references between the \
                         objects kept intact",
                    ) {
                        open_export = true;
                        ui.close();
                    }
                    ui.separator();
                    if tip_button(
                        ui,
                        "📐 Generate test data",
                        "Write a complete synthetic RT study (CT, RTSTRUCT, RTPLAN, \
                         RTDOSE, DX, RTIMAGE, REG, RTRECORD) into the application folder",
                    ) {
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
                    // Syncing is a property of the crosshair and of having a
                    // second dataset, so it goes away with either.
                    let both = self.both_volumes();
                    if self.show_crosshair && both {
                        ui.checkbox(&mut self.link_studies, "Sync crosshairs between datasets")
                            .on_hover_text(
                                "Move one crosshair and the other follows to the same patient \
                             point - through the active registration when there is one. \
                             Off, each dataset is navigated on its own.",
                            );
                    }
                    ui.checkbox(&mut self.show_labels, "Orientation labels");
                    ui.checkbox(&mut self.show_isocenters, "Isocenters");
                    ui.separator();
                    ui.checkbox(&mut self.side_open, "Data tree (F9)")
                        .on_hover_text(
                            "The left panel. Hidden, its arrow stays on the window's left \
                             edge to bring it back, and so does F9.",
                        );
                    let any_module = self.any_module();
                    ui.add_enabled_ui(any_module, |ui| {
                        ui.checkbox(&mut self.right_open, "Modules (F10)")
                            .on_hover_text(if any_module {
                                "The right panel. Hidden, its arrow stays on the window's \
                                 right edge to bring it back, and so does F10."
                            } else {
                                "There is no modules panel until a module is turned on in \
                                 the Modules menu"
                            });
                    });
                    ui.separator();
                    ui.label("Appearance:");
                    let before = self.theme;
                    self.theme.radio_buttons(ui);
                    if self.theme != before {
                        new_theme = Some(self.theme);
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
                        // Every module is one line of this menu and one
                        // section of the right panel. Everything a module
                        // does - direction, method, region, parameters,
                        // landmarks, analytics, fusion, the vector field, the
                        // simulated motion, what travels and where - lives in
                        // its section, so the menu only decides whether it is
                        // there.
                        ui.weak("Sections of the right panel (F10):");
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
                             other - the ground truth a registration can be measured against.",
                            )
                            .changed();
                        modules_changed |= ui
                            .checkbox(&mut self.module_structures, "Structure editor")
                            .on_hover_text(
                                "Insert, edit and combine structures: empty structures and \
                                 points, the generators, everything that acts on the \
                                 selected structure (interpolation, tidying, moving, the \
                                 type, the derived recipe), and the structure algebra.",
                            )
                            .changed();
                        modules_changed |= ui
                            .checkbox(&mut self.module_auto, "Structure auto tools")
                            .on_hover_text(
                                "The tools that find a structure by themselves: the body \
                                 contour, automatic multi-organ segmentation \
                                 (TotalSegmentator), prompt segmentation (SegVol) and slice \
                                 propagation (MedSAM2). Every network runs locally, in Rust.",
                            )
                            .changed();
                        modules_changed |= ui
                            .checkbox(&mut self.module_propagation, "Structure propagation")
                            .on_hover_text(
                                "Carry contours and segmentations from one dataset to the \
                             other through the active registration - globally, or refined \
                             on an enclosing structure first. Sits next to the \
                             registration that drives it.",
                            )
                            .changed();
                        modules_changed |= ui
                            .checkbox(&mut self.module_dose, "Dose estimation")
                            .on_hover_text(
                                "Dmean, Dmin, Dmax, D_X% and V_X of every ticked structure \
                                 against one dose, in a table that follows every edit of \
                                 the structures.",
                            )
                            .changed();
                        if self.any_module() {
                            self.right_open = true;
                        }
                    });
                ui.menu_button("Tools", |ui| {
                    // The three structure tools that keep a window of their
                    // own, then the rest. The engines are sections of the
                    // Structure auto tools module.
                    let any = self.any_volume();
                    if enabled_tip_button(
                        ui,
                        any,
                        "◑ Structure comparison",
                        "Volumes, centroid offset, Dice, HD95, surface distances and \
                         the least-squares rigid offset of any two structures - \
                         within a dataset or across the two",
                    ) {
                        self.open_compare_dialog(0);
                        ui.close();
                    }
                    if enabled_tip_button(
                        ui,
                        any,
                        super::stats_win::DETAILS.menu_entry(),
                        "One row per structure: volume by planimetry and by voxel count, \
                         the Dice against a reference, the grey levels inside it, what \
                         the geometry costs in slices and points, and whether a derived \
                         structure still matches its recipe. With CSV export.",
                    ) {
                        let slot = self.first_volume_slot();
                        self.open_stats_dialog(slot);
                        ui.close();
                    }
                    if enabled_tip_button(
                        ui,
                        any,
                        MOTION.menu_entry(),
                        "Register the reference phase of a 4D group to every other phase, \
                         carry the targets across, and measure their motion - trajectories, \
                         drift, correlations and the ITV.",
                    ) {
                        let slot = self.first_volume_slot();
                        self.open_motion_dialog(slot, None);
                        ui.close();
                    }
                    ui.separator();
                    let both = self.both_volumes();
                    if enabled_tip_button(
                        ui,
                        both,
                        "◎ Transfer by relationship",
                        "Place a structure into the other dataset at the same offset \
                         from a reference structure (e.g. the heart) - the \
                         target-reference relationship travels, not a registration",
                    ) {
                        self.open_transfer_dialog(0);
                        ui.close();
                    }
                    let has_dose = self
                        .slots
                        .iter()
                        .any(|s| s.study.as_ref().is_some_and(|st| !st.doses.is_empty()));
                    if enabled_tip_button(
                        ui,
                        has_dose,
                        "📊 Dose-volume histograms",
                        "Cumulative and differential DVHs of any structures against \
                         any loaded dose objects, with the metrics table, protocol \
                         constraint checking and CSV export - in a window that can \
                         go on its own monitor",
                    ) {
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
                            egui::Button::new("📈 Motion results"),
                        )
                        .on_hover_text("The finished 4D motion runs of this session")
                        .clicked()
                    {
                        self.motion_results_open = true;
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            self.any_volume(),
                            egui::Button::new("☢ Digitally reconstructed radiograph"),
                        )
                        .on_hover_text(
                            "Forward-project the CT onto a flat detector - two \
                             independent projectors, an exact ray tracer and an \
                             interpolating one, with the difference between them",
                        )
                        .clicked()
                    {
                        open_drr = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "🏥 PACS - patient archive",
                        "The local archive: every study filed here, ready to be taken \
                         into a dataset and given back the structures and \
                         segmentations drawn on it",
                    ) {
                        open_pacs = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "📦 Downloaded models",
                        "What every segmentation engine has downloaded, how much disk \
                         it costs, and the buttons to download, update or remove it - \
                         one model at a time or all of them",
                    ) {
                        open_models = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "🔏 Anonymize DICOM folder",
                        "Scan a folder, review every identifying tag with its current \
                         and proposed values, then rewrite the files (in place or into \
                         a new folder) with consistently regenerated UIDs",
                    ) {
                        self.anon_open = true;
                        ui.close();
                    }
                });
                ui.menu_button("Settings", |ui| {
                    ui.menu_button("Graphics backend", |ui| {
                        ui.label("Which graphics API the program draws and computes with.");
                        ui.add_space(4.0);
                        let before = self.graphics_backend;
                        for b in crate::gfx::Backend::offered() {
                            ui.radio_value(&mut self.graphics_backend, b, b.label())
                                .on_hover_text(b.hint());
                        }
                        if self.graphics_backend != before {
                            save_settings = true;
                        }
                        ui.add_space(4.0);
                        // What is drawing right now, which is not always what
                        // was asked for: an unusable driver is fallen back
                        // from without telling anybody.
                        let running = self
                            .active_backend
                            .or_else(crate::gfx::from_env)
                            .unwrap_or(self.graphics_backend);
                        ui.weak(format!(
                            "{} is set now. A change takes effect the next time the \
                             program starts.",
                            running.label()
                        ));
                    });
                    ui.menu_button("MCP server", |ui| {
                        ui.label(
                            "rds-mcp lets an AI assistant drive the station's tools headlessly.",
                        );
                        ui.add_space(4.0);
                        let exe = crate::settings::mcp_exe_path();
                        if exe.is_file() {
                            ui.weak(format!("Installed: {}", exe.display()));
                        } else {
                            ui.weak(format!(
                                "Not installed: {} was not found. Build it with \
                                 cargo build --release --features mcp.",
                                exe.display()
                            ));
                        }
                        ui.weak(format!(
                            "Configuration (roots, output folder, PHI policy): {}",
                            crate::settings::mcp_config_path().display()
                        ));
                        ui.add_space(4.0);
                        if tip_button(
                            ui,
                            "Copy client configuration",
                            "Copies the JSON entry for claude_desktop_config.json (or an \
                             equivalent MCP client) to the clipboard.",
                        ) {
                            ui.ctx().copy_text(crate::settings::mcp_client_snippet());
                            ui.close();
                        }
                        ui.weak("See docs/mcp.md for the tools and the safety rules.");
                    });
                    if let Some(msg) = &self.settings_error {
                        ui.weak(msg);
                    }
                });
                ui.menu_button("Help", |ui| {
                    ui.label("MPR views - mouse:");
                    ui.weak("Left click / drag - move linked crosshair");
                    ui.weak("Mouse wheel - scroll slices");
                    ui.weak("Ctrl + wheel / pinch - zoom at cursor");
                    ui.weak("Middle drag - pan");
                    ui.weak("Right drag - window / level (x = width, y = center)");
                    ui.separator();
                    ui.label(
                        "Drawing tools (✏ Draw structure on the toolbar; they take over the \
                         left button):",
                    );
                    ui.weak("Left drag - paint / erase");
                    ui.weak("Left press + drag ↑↓ - grow / shrink the region (✨)");
                    ui.weak("Alt - erase while painting");
                    ui.weak("Shift + wheel, or [ ] - brush radius");
                    ui.weak("Ctrl + Z - undo the last stroke");
                    ui.weak("Esc - cancel the running region grow");
                    ui.separator();
                    ui.label(format!(
                        "{} (the box takes over the left button in its view):",
                        SLICE_PROP.name
                    ));
                    ui.weak(
                        "Left drag - draw the box; drag a corner to resize, the middle to move",
                    );
                    ui.weak("Left click - an include / exclude point, with ➕ / ➖ chosen");
                    ui.separator();
                    ui.label("Buttons:");
                    ui.weak("⟲ (view corner) - reset that view's zoom, pan and slice");
                    ui.weak("⛶ / ⊞ - maximize that view / restore the layout");
                    ui.weak("⟲ (toolbar) - reset every view of both datasets");
                    ui.weak("✏ Draw structure (toolbar) - unfold or fold the drawing tools");
                    ui.weak(
                        "⌖ - show / hide the crosshair; hidden, left click no \
                         longer navigates",
                    );
                    ui.weak("🔗 Sync - sync the crosshairs of A and B (shown while ⌖ is on)");
                    ui.separator();
                    ui.weak(format!(
                        "rust-dicom-station {} - research / QA viewer, not a medical device",
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
            let slot = self.first_volume_slot();
            self.open_drr_window(slot);
        }
        if open_export {
            self.open_export_dialog();
        }
        if let Some(theme) = new_theme {
            self.set_theme(ctx, theme);
        }
        if save_settings {
            self.persist_settings();
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
                    // force. Its numbers are dropped there - the two drag
                    // values to the left already show them - but kept in the
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
                        if enabled_tip_button(
                            ui,
                            has_3d,
                            format!("3D {slot_name}"),
                            format!(
                                "Open a 3D surface rendering of dataset {slot_name}'s structures \
                                 and segmentations"
                            ),
                        ) {
                            self.open_d3_window(slot);
                        }
                    }

                    // Slice-intersection (crosshair) toggle. With the
                    // crosshair hidden, left-click navigation is disabled.
                    if ui
                        .selectable_label(self.show_crosshair, "⌖")
                        .on_hover_text(
                            "Show / hide the slice intersection (crosshair).\n\
                             Hidden: left click does not navigate - slices change \
                             only by scrolling each view",
                        )
                        .clicked()
                    {
                        self.show_crosshair = !self.show_crosshair;
                    }

                    // Crosshair syncing: only meaningful while there is a
                    // crosshair to sync and a second dataset to sync it with,
                    // so it appears and disappears with them.
                    let both = self.both_volumes();
                    if self.show_crosshair
                        && both
                        && ui
                            .add(egui::Button::selectable(self.link_studies, "🔗 Sync"))
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

                    // Reset every view of both datasets.
                    if tip_button(
                        ui,
                        "⟲",
                        "Reset every view of both datasets: fit zoom, clear pan \
                         and put the crosshairs back at the volume centers",
                    ) {
                        self.reset_all_views();
                    }

                    // The drawing tools: one toggle that unfolds the row of
                    // tools and the options of the one in hand, right here,
                    // so a hand that is drawing never leaves the toolbar.
                    ui.separator();
                    let paintable = self.any_volume();
                    if !paintable {
                        self.seg_tool = SegTool::None;
                        self.draw_open = false;
                    }
                    if ui
                        .add_enabled(
                            paintable,
                            egui::Button::selectable(self.draw_open, "✏ Draw structure"),
                        )
                        .on_hover_text(
                            "Unfold the drawing tools: paint, erase and grow a segmentation; \
                             polygon, spline, freehand, live wire, brush and nudge for an RT \
                             structure. Fold them away to put the tool down",
                        )
                        .clicked()
                    {
                        self.draw_open = !self.draw_open;
                        if !self.draw_open {
                            self.set_seg_tool(SegTool::None);
                        }
                    }
                    if self.draw_open {
                        self.draw_strip(ui);
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
                    if let Some(hu) = v.get(
                        c[0].round() as i64,
                        c[1].round() as i64,
                        c[2].round() as i64,
                    ) {
                        ui.monospace(format!("{hu:5} HU"));
                    }
                    if let Some(d) = study.doses.get(s.active_dose).and_then(|d| d.sample(p)) {
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
                    // bindings fold into a single "?" that the pointer opens -
                    // always the bindings of the tool in force.
                    let hint = super::struct_tools::mouse_hint(self.seg_tool);
                    // `Sense::hover`: it looks like a button and answers the
                    // pointer, but there is nothing to click.
                    ui.add(egui::Button::new("?").small().sense(egui::Sense::hover()))
                        .on_hover_text(hint);
                });
            });
        });
    }
}
