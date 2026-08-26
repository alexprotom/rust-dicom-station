//! Window chrome: the menu bar, the toolbar and the status bar.

use super::*;

impl ViewerApp {
    // -- Menu bar ---------------------------------------------------------
    pub(super) fn menu_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut open_a = false;
        let mut open_b = false;
        let mut close_b = false;
        let mut reset_views = false;
        let mut do_reg: Option<(RegMethod, bool)> = None;
        let mut open_gen = false;
        let mut open_models = false;
        let mut open_propagate = false;
        let mut open_drr = false;
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
                    // Quick actions only — direction, region, parameters,
                    // landmarks, analytics, fusion and the vector field live
                    // in the sidebar Registration section.
                    let both = self.slots[0].study.is_some() && self.slots[1].study.is_some();
                    let running = self.reg_job.is_some();
                    let moving = SLOT_NAMES[1 - self.reg_fixed_slot.min(1)];
                    let fixed = SLOT_NAMES[self.reg_fixed_slot.min(1)];
                    ui.weak(format!("Register {moving} onto {fixed}:"));
                    for method in RegMethod::ALL {
                        if ui
                            .add_enabled(
                                both && !running,
                                egui::Button::new(format!(
                                    "{} — {}",
                                    method.family(),
                                    method.short()
                                )),
                            )
                            .on_hover_text(method.hint())
                            .clicked()
                        {
                            do_reg = Some((method, false));
                            ui.close();
                        }
                    }
                    ui.separator();
                    let can_refine = self
                        .registration
                        .as_ref()
                        .is_some_and(|r| r.fixed_slot == self.reg_fixed_slot.min(1));
                    if ui
                        .add_enabled(
                            both && !running && can_refine,
                            egui::Button::new("Refine the active registration"),
                        )
                        .on_hover_text(
                            "Recover a correction on top of the active result with the \
                             method and region chosen in the sidebar, and add the two \
                             together",
                        )
                        .clicked()
                    {
                        do_reg = Some((self.reg_method, true));
                        ui.close();
                    }
                    if !both {
                        ui.weak("Load two datasets (comparison mode) first");
                    }
                });
                ui.menu_button("Tools", |ui| {
                    // The three segmentation engines, one block per dataset:
                    // the same three entries, in the same order, for A and B.
                    let tools: [(&ToolInfo, &str); 3] = [
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
                    ];
                    let mut open_tool: Option<(usize, &ToolInfo)> = None;
                    for slot in 0..SLOT_NAMES.len() {
                        if slot == 1 && !self.comparison {
                            continue;
                        }
                        if slot == 1 {
                            ui.separator();
                        }
                        let loaded = self.slots[slot].study.is_some();
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
                    let both = self.slots[0].study.is_some() && self.slots[1].study.is_some();
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
                    if ui
                        .add_enabled(
                            self.slots[0].study.is_some() || self.slots[1].study.is_some(),
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
                    ui.label("Segmentation (🖌 ◻ ✨ take over the left button):");
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
        if let Some((method, refine)) = do_reg {
            self.reg_method = method;
            self.start_registration(refine);
        }
        if open_gen {
            self.gen_open = true;
        }
        if open_models {
            self.open_models_window();
        }
        if open_drr {
            let slot = usize::from(self.slots[0].study.is_none());
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
                for (slot, slot_name) in SLOT_NAMES.iter().enumerate() {
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
                        format!("{slot_name}: ")
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
