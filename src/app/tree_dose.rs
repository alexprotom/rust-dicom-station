//! The data tree's dose section: the dose grids of a study, and how the
//! active one is displayed.

use super::*;

impl ViewerApp {
    /// The RTDOSE grids filed under one study: which one is displayed, and
    /// what it is. How dose is *drawn* is a display setting shared by both
    /// workspaces, so it lives in [`Self::dose_display_section`] instead.
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
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        // The views draw one dose grid at a time, so the rows work the way
        // the structure sets' do: clicking another grid makes it the drawn
        // one, clicking the drawn grid takes dose off the images.
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
                        let selected = i == picked;
                        let resp = Self::series_row(
                            ui,
                            selected,
                            dose_on,
                            format!(
                                "{}{}",
                                d.label,
                                Self::plan_suffix(plans, &d.referenced_plan_uid)
                            ),
                            current.as_ref() == Some(&RenameTarget::Dose { slot, idx: i }),
                        );
                        Self::focus_row(
                            ui,
                            &resp,
                            RenameTarget::Dose { slot, idx: i },
                            &current,
                            &mut focus,
                        );
                        if resp.clicked() {
                            // Clicking the drawn grid takes dose off the
                            // images; clicking any other draws that one.
                            if selected {
                                shown_dose = Some((!dose_on).then_some(i));
                            } else {
                                picked = i;
                                shown_dose = Some(Some(i));
                            }
                        }
                        resp.on_hover_text(format!(
                            "{}  max {:.2} {}\nclick: draw this grid, or take dose off \
                             the images\nright-click: rename or remove\nF2: rename, \
                             Delete: remove",
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
        if focus.is_some() {
            self.tree_focus = focus;
        }
    }

    /// How dose is drawn - colorwash, isodose lines, opacity, threshold and
    /// the isodose ladder. Shared by every workspace, so it is shown once, at
    /// workspace level, under the first one that actually has dose.
    pub(super) fn dose_display_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let first = self.open_slots().into_iter().find(|&s| {
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
}
