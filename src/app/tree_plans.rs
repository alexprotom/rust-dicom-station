//! The data tree's plans section: the RT plans of a study, one row each,
//! and the suffixes that tell two plans or two structure sets apart. What a
//! plan holds (its facts and beams) is read in the additional pane, which a
//! click on its row opens (object_view.rs).

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
        let mut save: Option<crate::export::Pick> = None;
        let mut remove: Option<ObjRef> = None;
        let mut new_active: Option<usize> = None;
        let mut new_shown: Option<bool> = None;
        let mut focus: Option<RenameTarget> = None;
        let mut clicked_plan: Option<usize> = None;
        let current = self.tree_focus.clone();
        // Which plan the additional pane reads, if one: a click on the
        // shown plan that is not there yet opens it rather than hiding it.
        let in_pane: Vec<bool> = which
            .iter()
            .map(|&pi| self.object_shown(slot, super::object_view::PaneObject::Plan(pi)))
            .collect();
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
                    for (row, &pi) in which.iter().enumerate() {
                        let read_here = in_pane[row];
                        let Some(plan) = study.plans.get(pi) else {
                            continue;
                        };
                        let selected = active_plan == Some(pi);
                        let me = RenameTarget::Plan { slot, idx: pi };
                        let resp = Self::series_row(
                            ui,
                            selected,
                            shown,
                            format!(
                                "{}{}{}",
                                if plan.label.is_empty() {
                                    "unnamed"
                                } else {
                                    &plan.label
                                },
                                Self::structset_suffix(study, &plan.referenced_structset_uid),
                                Self::plan_ref_suffix(&study.plans, &plan.referenced_plan_uid)
                            ),
                            current.as_ref() == Some(&me),
                        );
                        Self::focus_row(ui, &resp, me.clone(), &current, &mut focus);
                        // Clicking the shown plan hides its isocenters and
                        // beams; clicking any other makes it the shown one.
                        let mut toggle = || {
                            clicked_plan = Some(pi);
                            if selected && shown && !read_here {
                                new_shown = Some(true);
                            } else if selected {
                                new_shown = Some(!shown);
                            } else {
                                new_active = Some(pi);
                                new_shown = Some(true);
                            }
                        };
                        if resp.clicked() {
                            toggle();
                        }
                        let mut menu_toggle = false;
                        resp.on_hover_text(Self::object_tip(
                            &plan.sop_instance_uid,
                            format!(
                                "{} plan{}",
                                if plan.plan_kind.is_empty() {
                                    "RT"
                                } else {
                                    &plan.plan_kind
                                },
                                if plan.name.is_empty() || plan.name == plan.label {
                                    String::new()
                                } else {
                                    format!(" · {}", plan.name)
                                }
                            ),
                        ))
                        .context_menu(|ui| {
                            if Self::show_entry(ui, selected && shown) {
                                menu_toggle = true;
                                ui.close();
                            }
                            ui.separator();
                            if Self::save_dicom_entry(ui, "plan") {
                                save = Some(crate::export::Pick::Object {
                                    slot,
                                    kind: crate::export::ObjKind::Plan,
                                    index: pi,
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
                                    kind: ObjKind::Plan,
                                    idx: pi,
                                });
                                ui.close();
                            }
                        });
                        if menu_toggle {
                            toggle();
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
        // The plan read in full in the additional pane while it is shown.
        if let (Some(pi), Some(on)) = (clicked_plan, new_shown) {
            let obj = super::object_view::PaneObject::Plan(pi);
            if on {
                self.show_object(slot, obj);
            } else if self.object_shown(slot, obj) {
                self.hide_additional(slot);
            }
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
            .unwrap_or_else(|| " ▶ ⚠ plan not loaded".to_string())
    }

    /// " ▶ verifies NAME" for the plan a verification plan names, or the badge
    /// of a reference that leads nowhere.
    fn plan_ref_suffix(plans: &[crate::rtplan::PlanInfo], sop: &str) -> String {
        if sop.is_empty() {
            return String::new();
        }
        match plans.iter().find(|p| p.sop_instance_uid == sop) {
            Some(p) => format!(
                " ▶ verifies {}",
                if p.label.is_empty() {
                    "unnamed"
                } else {
                    &p.label
                }
            ),
            None => " ▶ ⚠ its plan not loaded".to_string(),
        }
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
            .unwrap_or_else(|| " ▶ ⚠ structure set not loaded".to_string())
    }
}
