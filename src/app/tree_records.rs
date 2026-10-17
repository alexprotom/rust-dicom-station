//! The data tree's records and warnings sections: treatment records, and
//! what loading the study had to say.

use super::*;

impl ViewerApp {
    /// RT (Ion) Beams Treatment Records: per-beam delivered metersets.
    pub(super) fn records_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        {
            let Some(study) = &self.slots[slot].study else {
                return;
            };
            if study.treat_records.is_empty() {
                return;
            }
            egui::CollapsingHeader::new(format!(
                "Treatment records ({})",
                study.treat_records.len()
            ))
            .id_salt(("records", slot))
            .default_open(false)
            .show(ui, |ui| {
                for (ri, rec) in study.treat_records.iter().enumerate() {
                    let resp = ui
                        .label(
                            egui::RichText::new(format!(
                                "{}{}{}{}",
                                rec.label,
                                if rec.ion { "  [ion]" } else { "" },
                                rec.fraction
                                    .map(|f| format!("  fx {f}"))
                                    .unwrap_or_default(),
                                if rec.date.is_empty() {
                                    String::new()
                                } else {
                                    format!("  {}", rec.date)
                                }
                            ))
                            .strong(),
                        )
                        .on_hover_text(
                            "right-click: rename or remove this record\nF2: rename, Delete: \
                             remove",
                        );
                    Self::focus_row(
                        ui,
                        &resp,
                        RenameTarget::Record { slot, idx: ri },
                        &current,
                        &mut focus,
                    );
                    resp.context_menu(|ui| {
                        if ui.button("✏ Rename").clicked() {
                            rename = Some(RenameTarget::Record { slot, idx: ri });
                            ui.close();
                        }
                        ui.separator();
                        if ui.button("🗑 Remove").clicked() {
                            remove = Some(ObjRef {
                                slot,
                                kind: ObjKind::Record,
                                idx: ri,
                            });
                            ui.close();
                        }
                    });
                    if !rec.machine.is_empty() {
                        ui.weak(format!("Machine: {}", rec.machine));
                    }
                    egui::Grid::new(("rec_grid", slot, ri))
                        .striped(true)
                        .min_col_width(10.0)
                        .show(ui, |ui| {
                            ui.strong("Beam");
                            ui.strong("MU spec");
                            ui.strong("MU del");
                            ui.strong("Δ%");
                            ui.strong("Status");
                            ui.end_row();
                            for b in &rec.beams {
                                ui.label(&b.name).on_hover_text(format!(
                                    "Beam {} · verification: {}",
                                    b.number,
                                    if b.verification_status.is_empty() {
                                        "n/a"
                                    } else {
                                        &b.verification_status
                                    }
                                ));
                                ui.label(
                                    b.specified_meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "-".into()),
                                );
                                ui.label(
                                    b.delivered_meterset
                                        .map(|m| format!("{m:.1}"))
                                        .unwrap_or_else(|| "-".into()),
                                );
                                ui.label(match (b.specified_meterset, b.delivered_meterset) {
                                    (Some(s), Some(d)) if s > 1e-9 => {
                                        format!("{:+.1}", 100.0 * (d - s) / s)
                                    }
                                    _ => "-".into(),
                                });
                                let status = if b.termination_status.is_empty() {
                                    "-"
                                } else {
                                    &b.termination_status
                                };
                                if status == "NORMAL" || status == "-" {
                                    ui.label(status);
                                } else {
                                    let c = alert_color(ui.visuals());
                                    ui.label(egui::RichText::new(status).color(c));
                                }
                                ui.end_row();
                            }
                        });
                }
            });
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

    /// What the loader had to say about this workspace's files, and the
    /// button that says it has been read.
    ///
    /// A warning is about the load, not about the data: once it has been
    /// seen it is noise on every later glance at the tree. *Acknowledge*
    /// drops them, and the next load brings whatever that one has to say.
    pub(super) fn warnings_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if study.warnings.is_empty() {
            return;
        }
        let mut acknowledge = false;
        // The button belongs on the header's own row - beside the count,
        // not inside the list - so it is there whether the list is open or
        // shut. That needs the header drawn by hand.
        let id = ui.make_persistent_id(("warn", slot));
        egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false)
            .show_header(ui, |ui| {
                let study = self.slots[slot].study.as_ref().expect("checked above");
                ui.label(
                    egui::RichText::new(format!("⚠ Warnings ({})", study.warnings.len()))
                        .color(warn_color(ui.visuals())),
                );
                if ui
                    .add(egui::Button::new("Acknowledge").small())
                    .on_hover_text(
                        "Clear these warnings. They describe a load or a transfer that \
                         has already happened, so once they have been read they have \
                         nothing left to say; the next one reports its own.",
                    )
                    .clicked()
                {
                    acknowledge = true;
                }
            })
            .body(|ui| {
                let study = self.slots[slot].study.as_ref().expect("checked above");
                for w in &study.warnings {
                    ui.label(egui::RichText::new(w).small());
                }
            });
        if acknowledge {
            if let Some(study) = self.slots[slot].study.as_mut() {
                study.warnings.clear();
            }
        }
    }
}
