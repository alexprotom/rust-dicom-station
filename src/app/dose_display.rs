//! *Modules ▶ Dose display*: how the displayed dose grid is drawn over the
//! images - colorwash, isodose lines or both, the opacity, the threshold
//! under which nothing is drawn, and the isodose ladder.
//!
//! These are display settings of the viewer, not properties of a grid: one
//! set serves every workspace, which is why they left the data tree (where
//! they sat under whichever workspace happened to be the first with dose)
//! for a module of their own. *Which* grid is drawn, and its reference dose,
//! stay with the grid under *Dose* in the tree; a grid's context menu opens
//! this section.

use super::*;

impl ViewerApp {
    pub(super) fn dose_display_section(&mut self, ui: &mut egui::Ui) {
        self.module_header(
            ui,
            Module::DoseDisplay,
            |_, _| {},
            |app, ui| app.dose_display_body(ui),
        );
    }

    fn dose_display_body(&mut self, ui: &mut egui::Ui) {
        let any_dose = self.open_slots().into_iter().any(|s| {
            self.slots[s]
                .study
                .as_ref()
                .is_some_and(|st| !st.doses.is_empty())
        });
        if !any_dose {
            // The settings are the viewer's, so they can be set before any
            // dose is loaded; the note says why nothing changes yet.
            ui.weak("No workspace holds a dose grid; these apply once one is loaded");
        }
        let mut mode = self.dose_mode;
        let mut opacity = self.dose_opacity;
        let mut threshold = self.dose_threshold_pct;
        ui.horizontal(|ui| {
            ui.label("Show");
            egui::ComboBox::from_id_salt("dose_mode")
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
        self.dose_mode = mode;
        self.dose_opacity = opacity;
        self.dose_threshold_pct = threshold;
    }
}
