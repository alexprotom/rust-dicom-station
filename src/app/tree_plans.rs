//! The data tree's plans section: the RT plans of a study and the suffixes
//! that tell two plans or two structure sets apart.

use super::*;

impl ViewerApp {
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
        let mut new_active: Option<usize> = None;
        let mut new_shown: Option<bool> = None;
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        {
            let Some(study) = &self.slots[slot].study else {
                return;
            };
            // Like the structure sets: the plan this study node holds counts
            // as the active one only when the node really holds it.
            let active_plan = which
                .contains(&self.slots[slot].active_plan)
                .then(|| self.slots[slot].active_plan);
            let shown = self.slots[slot].plans_shown;
            Self::wrapped_node(
                ui,
                ("plans", slot, pat, stu),
                false,
                format!("Plans ({})", which.len()),
                |ui| {
                    for &pi in which {
                        let Some(plan) = study.plans.get(pi) else {
                            continue;
                        };
                        let selected = active_plan == Some(pi);
                        let resp = Self::series_row(
                            ui,
                            selected,
                            shown,
                            format!(
                                "{}{}",
                                if plan.label.is_empty() {
                                    "unnamed"
                                } else {
                                    &plan.label
                                },
                                Self::structset_suffix(study, &plan.referenced_structset_uid)
                            ),
                            current.as_ref() == Some(&RenameTarget::Plan { slot, idx: pi }),
                        );
                        Self::focus_row(
                            ui,
                            &resp,
                            RenameTarget::Plan { slot, idx: pi },
                            &current,
                            &mut focus,
                        );
                        if resp.clicked() {
                            // Clicking the shown plan hides its isocenters;
                            // clicking any other makes it the shown one.
                            if selected {
                                new_shown = Some(!shown);
                            } else {
                                new_active = Some(pi);
                                new_shown = Some(true);
                            }
                        }
                        resp.on_hover_text(
                            "click: draw this plan's isocenters, or take them off the \
                             views\nright-click: rename or remove\nF2: rename, Delete: remove",
                        )
                        .context_menu(|ui| {
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
                        // What a plan is made of sits under *that* plan, not
                        // at the end of the list - the same place a set's
                        // structures are listed.
                        if !selected {
                            continue;
                        }
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
                    }
                },
            );
        }
        if let Some(pi) = new_active {
            self.slots[slot].active_plan = pi;
        }
        if let Some(on) = new_shown {
            self.slots[slot].plans_shown = on;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if remove.is_some() {
            self.obj_remove = remove;
        }
    }

    /// " ▶ plan NAME" for the plan a dose was computed for.
    pub(super) fn plan_suffix(plans: &[crate::rtplan::PlanInfo], sop: &str) -> String {
        if sop.is_empty() {
            return String::new();
        }
        plans
            .iter()
            .find(|p| p.sop_instance_uid == sop)
            .map(|p| {
                format!(
                    " ▶ plan {}",
                    if p.label.is_empty() {
                        "unnamed"
                    } else {
                        &p.label
                    }
                )
            })
            .unwrap_or_else(|| " ▶ (plan not loaded)".to_string())
    }

    /// " ▶ structures NAME" for the set a plan was made on, or a word about
    /// why there is none - the counterpart of [`Self::series_suffix`].
    fn structset_suffix(study: &LoadedStudy, sop: &str) -> String {
        if sop.is_empty() {
            return String::new();
        }
        study
            .structure_sets
            .iter()
            .find(|s| s.sop_instance_uid == sop)
            .map(|ss| {
                format!(
                    " ▶ {}",
                    if ss.label.is_empty() {
                        &ss.file_name
                    } else {
                        &ss.label
                    }
                )
            })
            .unwrap_or_else(|| " ▶ (structure set not loaded)".to_string())
    }
}
