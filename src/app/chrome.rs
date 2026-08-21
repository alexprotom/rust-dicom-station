//! Window chrome: the menu bar, the toolbar and the status bar.

use super::*;

impl ViewerApp {
    // -- Menu bar ---------------------------------------------------------
    pub(super) fn menu_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut open_a = false;
        let mut open_b = false;
        let mut close_b = false;
        let mut reset_views = false;
        let mut do_reg: Option<RegKind> = None;
        let mut open_gen = false;
        let mut open_export: Option<usize> = None;
        let mut new_theme: Option<egui::ThemePreference> = None;

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
                    #[allow(clippy::needless_range_loop)] // `slot` also indexes `self.slots`
                    for slot in 0..2 {
                        if slot == 1 && !self.comparison && self.slots[1].study.is_none() {
                            continue;
                        }
                        if ui
                            .add_enabled(
                                self.slots[slot].study.is_some(),
                                egui::Button::new(format!(
                                    "💾 Export dataset {} as DICOM…",
                                    SLOT_NAMES[slot]
                                )),
                            )
                            .on_hover_text(
                                "Write the displayed volume, structure sets, dose grids                                  and plans as DICOM files — with the patient / study /                                  equipment tags reviewed and edited first",
                            )
                            .clicked()
                        {
                            open_export = Some(slot);
                            ui.close();
                        }
                    }
                    ui.separator();
                    if ui
                        .button("🧪 Generate test data…")
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
                    ui.checkbox(&mut self.link_studies, "Link crosshairs between datasets");
                    ui.separator();
                    ui.checkbox(&mut self.show_contours, "Contours");
                    ui.checkbox(&mut self.show_crosshair, "Crosshair");
                    ui.checkbox(&mut self.show_labels, "Orientation labels");
                    ui.checkbox(&mut self.show_isocenters, "Isocenters");
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
                ui.menu_button("Registration", |ui| {
                    // Quick actions only — direction, parameters, fusion and
                    // cancel/clear live in the sidebar Registration section.
                    let both =
                        self.slots[0].study.is_some() && self.slots[1].study.is_some();
                    let running = self.reg_job.is_some();
                    let moving = SLOT_NAMES[1 - self.reg_fixed_slot.min(1)];
                    let fixed = SLOT_NAMES[self.reg_fixed_slot.min(1)];
                    if ui
                        .add_enabled(
                            both && !running,
                            egui::Button::new(format!("Rigid: register {moving} ▶ {fixed}")),
                        )
                        .on_hover_text(
                            "6-DOF Euler transform, ASGD optimizer (elastix-style), \
                             3 resolution levels",
                        )
                        .clicked()
                    {
                        do_reg = Some(RegKind::Rigid);
                        ui.close();
                    }
                    if ui
                        .add_enabled(
                            both && !running,
                            egui::Button::new(format!(
                                "Deformable: register {moving} ▶ {fixed} (B-spline)"
                            )),
                        )
                        .on_hover_text(
                            "Rigid pre-alignment + cubic B-spline free-form deformation, \
                             ASGD optimizer (elastix-style)",
                        )
                        .clicked()
                    {
                        do_reg = Some(RegKind::Deformable);
                        ui.close();
                    }
                    if !both {
                        ui.weak("Load two datasets (comparison mode) first");
                    }
                });
                ui.menu_button("Tools", |ui| {
                    let auto_free = self.autoseg_job.is_none();
                    for (slot, slot_name) in SLOT_NAMES.iter().enumerate() {
                        if slot == 1 && !self.comparison {
                            continue;
                        }
                        let loaded = self.slots[slot].study.is_some();
                        if ui
                            .add_enabled(
                                loaded && auto_free,
                                egui::Button::new(format!(
                                    "🤖 Auto-segment dataset {slot_name}…"
                                )),
                            )
                            .on_hover_text(
                                "Automatic multi-organ segmentation of the displayed CT \
                                 (TotalSegmentator's nnU-Net models, re-implemented \
                                 natively in Rust; runs locally on CPU or GPU)",
                            )
                            .clicked()
                        {
                            self.open_autoseg_dialog(slot);
                            ui.close();
                        }
                    }
                    ui.separator();
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
                    ui.label("Segmentation (🖌 ◻ ✨ take over the left button):");
                    ui.weak("Left drag — paint / erase");
                    ui.weak("Left press + drag ↑↓ — grow / shrink the region (✨)");
                    ui.weak("Alt — erase while painting");
                    ui.weak("Shift + wheel, or [ ] — brush radius");
                    ui.weak("Ctrl + Z — undo the last stroke");
                    ui.weak("Esc — cancel the running region grow");
                    ui.separator();
                    ui.label("Buttons:");
                    ui.weak("⟲ (view corner) — reset that view's zoom, pan and slice");
                    ui.weak("⛶ / ❐ — maximize that view / restore the layout");
                    ui.weak("⟲ (toolbar) — reset every view of both datasets");
                    ui.weak(
                        "⌖ — show / hide the crosshair; hidden, left click no \
                         longer navigates",
                    );
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
        if close_b {
            self.close_comparison();
        }
        if reset_views {
            self.reset_all_views();
        }
        if let Some(kind) = do_reg {
            self.start_registration(kind);
        }
        if open_gen {
            self.gen_open = true;
        }
        if let Some(slot) = open_export {
            self.open_export_dialog(slot);
        }
        if let Some(theme) = new_theme {
            self.set_theme(ctx, theme);
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
                    egui::ComboBox::from_id_salt("wl_preset")
                        .selected_text("CT presets")
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for (name, c, w) in WL_PRESETS {
                                if ui
                                    .button(format!("{name}  (C {c:.0} / W {w:.0})"))
                                    .clicked()
                                {
                                    self.window_center = *c;
                                    self.window_width = *w;
                                }
                            }
                            ui.separator();
                            if ui.button("Full range").clicked() {
                                full_range = true;
                            }
                        });
                    if full_range {
                        if let Some(study) = &self.slots[self.hovered_slot.min(1)].study {
                            let v = &study.volume;
                            self.window_center = (v.min_value as f32 + v.max_value as f32) * 0.5;
                            self.window_width = (v.max_value as f32 - v.min_value as f32).max(1.0);
                        }
                    }

                    ui.separator();
                    // 3D structure rendering windows.
                    #[allow(clippy::needless_range_loop)] // `slot` also indexes `self.slots`
                    for slot in 0..2 {
                        let has_3d = self.slots[slot]
                            .study
                            .as_ref()
                            .map(|s| !s.structure_sets.is_empty())
                            .unwrap_or(false)
                            || !self.slots[slot].segs.is_empty();
                        if slot == 1 && self.slots[1].study.is_none() {
                            continue;
                        }
                        if ui
                            .add_enabled(
                                has_3d,
                                egui::Button::new(format!("3D {}", SLOT_NAMES[slot])),
                            )
                            .on_hover_text(format!(
                                "Open a 3D surface rendering of dataset {}'s structures \
                                 and segmentations",
                                SLOT_NAMES[slot]
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
                    let mut pick = |ui: &mut egui::Ui, tool: SegTool, label: &str, tip: &str| {
                        if ui
                            .selectable_label(self.seg_tool == tool, label)
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
                        "🖌 Paint",
                        "Paint the active segmentation (LMB drag).\n\
                         Hold Alt to erase · Shift+wheel or [ ] resize the brush · Ctrl+Z undo",
                    );
                    pick(
                        ui,
                        SegTool::Erase,
                        "◻ Erase",
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
                #[allow(clippy::needless_range_loop)] // `slot` also indexes `self.slots`
                for slot in 0..2 {
                    if slot == 1 && !self.comparison {
                        // Study B is hidden while comparison mode is off.
                        continue;
                    }
                    let s = &self.slots[slot];
                    let Some(study) = &s.study else { continue };
                    let v = &study.volume;
                    let c = s.cursor;
                    let p = v.voxel_to_patient(c[0], c[1], c[2]);
                    let both = self.comparison && self.slots[1].study.is_some();
                    let prefix = if both {
                        format!("{}: ", SLOT_NAMES[slot])
                    } else {
                        String::new()
                    };
                    if slot == self.hovered_slot || !both {
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
                    } else {
                        ui.monospace(prefix.trim_end().to_string());
                    }
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
                    ui.weak(match self.seg_tool {
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
                    });
                });
            });
        });
    }
}
