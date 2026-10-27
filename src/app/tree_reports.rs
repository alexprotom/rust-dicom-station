//! The data tree's reports section: the dose reports, protocol pages and
//! other structured reports of a workspace, shown as tables in the row's
//! additional pane.

use super::*;

impl ViewerApp {
    /// The reports, drawn like image series: a click shows one in the
    /// additional pane, an `A` badge says it is there.
    pub(super) fn reports_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n = self.report_count(slot);
        if n == 0 {
            return;
        }
        let mut show: Option<usize> = None;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        let shown: Vec<bool> = (0..n).map(|i| self.report_shown(slot, i)).collect();
        // For a workspace with no image volume and no radiograph these are
        // what there is to read, so the section opens itself.
        let sole_content = !self.slots[slot].has_volume()
            && !self.slots[slot]
                .study
                .as_ref()
                .is_some_and(|st| st.has_radiographs());
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            egui::CollapsingHeader::new(format!("Reports ({n})"))
                .id_salt(("reports", slot))
                .default_open(sole_content)
                .show(ui, |ui| {
                    for (i, r) in study.reports.iter().enumerate() {
                        let me = RenameTarget::Report { slot, idx: i };
                        let resp = ui
                            .scope(|ui| {
                                ui.spacing_mut().item_spacing.x = 3.0;
                                ui.horizontal(|ui| {
                                    let resp = ui.add(
                                        egui::Button::selectable(shown[i], r.label.as_str()).wrap(),
                                    );
                                    if shown[i] {
                                        ui.add(
                                            egui::Button::new(
                                                egui::RichText::new("A").small().strong(),
                                            )
                                            .small()
                                            .corner_radius(2.0)
                                            .sense(egui::Sense::hover()),
                                        )
                                        .on_hover_text("Shown in Additional");
                                    }
                                    Self::keyboard_badge(ui, current.as_ref() == Some(&me));
                                    resp
                                })
                                .inner
                            })
                            .inner;
                        Self::focus_row(ui, &resp, me.clone(), &current, &mut focus);
                        if resp.clicked() {
                            show = Some(i);
                        }
                        let mut tip = format!("{} · {}", r.kind.label(), r.modality);
                        if !r.date.is_empty() {
                            tip.push_str(&format!("\nDate: {}", r.date));
                        }
                        for (k, v) in r.info.iter().take(6) {
                            tip.push_str(&format!("\n{k}: {v}"));
                        }
                        let tables: usize = r.tables.len() + usize::from(!r.tree.is_empty());
                        tip.push_str(&format!(
                            "\n{} table{}{}",
                            tables,
                            if tables == 1 { "" } else { "s" },
                            if r.page.is_some() {
                                " · with the page"
                            } else {
                                ""
                            }
                        ));
                        resp.on_hover_text(tip).context_menu(|ui| {
                            if ui.button("👁 Show in Additional").clicked() {
                                show = Some(i);
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("✏ Rename").clicked() {
                                rename = Some(RenameTarget::Report { slot, idx: i });
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("🗑 Remove").clicked() {
                                remove = Some(ObjRef {
                                    slot,
                                    kind: ObjKind::Report,
                                    idx: i,
                                });
                                ui.close();
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
        if let Some(i) = show {
            self.show_report(slot, i);
            // Shown means on screen: a workspace put aside comes back, and a
            // maximized pane of another kind gives way to the row.
            self.shown[slot] = true;
            if self
                .maximized
                .is_some_and(|(s, k)| s != slot || k != PaneKind::Additional)
            {
                self.maximized = None;
            }
        }
    }
}
