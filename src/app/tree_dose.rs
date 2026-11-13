//! The data tree's dose section: the dose grids of a study, what radiation
//! each is of, and which one is displayed. How it is drawn, and the
//! reference dose it is drawn against, is the *Dose and plan display* module
//! (`dose_display.rs`).

use super::*;

impl ViewerApp {
    /// The RTDOSE grids filed under one study: which one is displayed, and
    /// what it is. How dose is *drawn* is a display setting shared by both
    /// workspaces, so it lives in [`Self::dose_display_section`] instead;
    /// *whether* it is drawn is this workspace's own (`StudySlot::dose_shown`),
    /// as its structure sets' and plans' are.
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
        let mut save: Option<crate::export::Pick> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        let mut display = false;
        let current = self.tree_focus.clone();
        // The views draw one dose grid at a time, so the rows work the way
        // the structure sets' do: clicking another grid makes it the drawn
        // one, clicking the drawn grid takes dose off the images.
        let mut shown_dose: Option<Option<usize>> = None;
        let dose_on = self.dose_mode != DoseMode::Off && self.slots[slot].dose_shown;
        {
            let StudySlot {
                study, active_dose, ..
            } = &mut self.slots[slot];
            let Some(study) = study.as_ref() else { return };
            let doses = &study.doses;
            let plans = &study.plans;
            let picked = (*active_dose).min(doses.len().saturating_sub(1));
            Self::wrapped_node(
                ui,
                ("dose", slot, pat, stu),
                false,
                format!("Dose ({})", which.len()),
                |ui| {
                    for &i in which {
                        let Some(d) = doses.get(i) else { continue };
                        let selected = i == picked;
                        let me = RenameTarget::Dose { slot, idx: i };
                        // The radiation, from the plan's beams: what a
                        // dose is of matters as much as its maximum.
                        let modality = super::dose_display::dose_modality(plans, d);
                        let resp = Self::series_row(
                            ui,
                            selected,
                            dose_on,
                            format!(
                                "{}{}{}",
                                d.label,
                                modality
                                    .as_ref()
                                    .map(|m| format!(" · {m}"))
                                    .unwrap_or_default(),
                                Self::plan_suffix(plans, &d.referenced_plan_uid)
                            ),
                            current.as_ref() == Some(&me),
                        );
                        Self::focus_row(ui, &resp, me.clone(), &current, &mut focus);
                        // Clicking the drawn grid takes dose off the images;
                        // clicking any other draws that one.
                        let toggle = || {
                            if selected {
                                Some((!dose_on).then_some(i))
                            } else {
                                Some(Some(i))
                            }
                        };
                        if resp.clicked() {
                            shown_dose = toggle();
                        }
                        resp.on_hover_text(Self::object_tip(
                            &d.sop_instance_uid,
                            format!(
                                "{} dose{} · max {:.2} {}",
                                d.summation_type,
                                modality.map(|m| format!(" · {m}")).unwrap_or_default(),
                                d.max_dose,
                                d.units.to_lowercase()
                            ),
                        ))
                        .context_menu(|ui| {
                            if Self::show_entry(ui, selected && dose_on) {
                                shown_dose = toggle();
                                ui.close();
                            }
                            if ui
                                .button("🎨 Display settings")
                                .on_hover_text("Open the Dose and plan display module")
                                .clicked()
                            {
                                display = true;
                                ui.close();
                            }
                            ui.separator();
                            if Self::save_dicom_entry(ui, "dose") {
                                save = Some(crate::export::Pick::Object {
                                    slot,
                                    kind: crate::export::ObjKind::Dose,
                                    index: i,
                                });
                                ui.close();
                            }
                            if Self::rename_entry(ui) {
                                rename = Some(me.clone());
                                ui.close();
                            }
                            ui.separator();
                            if Self::remove_entry(ui, "🗑 Remove") {
                                remove = Some(ObjRef {
                                    slot,
                                    kind: ObjKind::Dose,
                                    idx: i,
                                });
                                ui.close();
                            }
                        });
                    }
                },
            );
        }
        if let Some(pick) = shown_dose {
            self.show_dose(slot, pick);
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if let Some(p) = save {
            self.save_request = Some(super::save_menu::SaveRequest::Dicom(p));
        }
        if remove.is_some() {
            self.obj_remove = remove;
        }
        if focus.is_some() {
            self.tree_focus = focus;
        }
        if display {
            self.reveal_module(Module::DoseDisplay);
        }
    }

    /// Draw dose grid `pick` in this workspace, or (`None`) none at all. A
    /// grid to draw turns dose display on if it was off; the *Dose and plan display*
    /// section decides how it is drawn.
    pub(super) fn show_dose(&mut self, slot: usize, pick: Option<usize>) {
        match pick {
            Some(i) => {
                self.slots[slot].active_dose = i;
                self.slots[slot].dose_shown = true;
                if self.dose_mode == DoseMode::Off {
                    self.dose_mode = DoseMode::Colorwash;
                }
            }
            None => self.slots[slot].dose_shown = false,
        }
        self.settings_gen += 1;
    }
}
