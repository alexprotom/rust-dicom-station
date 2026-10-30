//! The data tree's reports section: the dose reports, protocol pages,
//! other structured reports and documents (encapsulated PDFs) of a study,
//! shown in the row's additional pane - as tables, a document as its
//! pages. The row drawn here is also the registrations' and the records'
//! ([`Self::additional_row`]): one line in the tree, the whole object in
//! the pane.

use super::*;

impl ViewerApp {
    /// The reports, drawn like image series: a click shows one in the
    /// additional pane, an `A` badge says it is there.
    pub(super) fn reports_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        (pi, si): (usize, usize),
        which: &[usize],
    ) {
        let n = which.len();
        if n == 0 {
            return;
        }
        // For a workspace with no image volume and no radiograph these are
        // what there is to read, so the section opens itself.
        let sole_content = !self.slots[slot].has_volume()
            && !self.slots[slot]
                .study
                .as_ref()
                .is_some_and(|st| st.has_radiographs());
        egui::CollapsingHeader::new(format!("Reports ({n})"))
            .id_salt(("reports", slot, pi, si))
            .default_open(sole_content)
            .show(ui, |ui| self.report_rows(ui, slot, which, None));
    }

    /// The rows of the reports `which`, named by their label - or, with
    /// `named` given, by what it makes of each (a session's dose report is
    /// named by its time).
    pub(super) fn report_rows(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        which: &[usize],
        named: Option<&dyn Fn(&crate::report::Report) -> String>,
    ) {
        let mut show: Option<usize> = None;
        let mut hide = false;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        let shown: Vec<bool> = which.iter().map(|&i| self.report_shown(slot, i)).collect();
        {
            let Some(study) = self.slots[slot].study.as_ref() else {
                return;
            };
            for (row, &i) in which.iter().enumerate() {
                let Some(r) = study.reports.get(i) else {
                    continue;
                };
                let me = RenameTarget::Report { slot, idx: i };
                let text = match named {
                    Some(f) => f(r),
                    None => r.label.clone(),
                };
                let resp =
                    Self::additional_row(ui, shown[row], text, current.as_ref() == Some(&me));
                Self::focus_row(ui, &resp, me.clone(), &current, &mut focus);
                // A click shows it; a click on it while it is shown takes
                // it out of the pane again.
                if resp.clicked() {
                    if shown[row] {
                        hide = true;
                    } else {
                        show = Some(i);
                    }
                }
                // What it is, then the file it came from: its facts are
                // what the pane shows.
                let mut tip = format!("{} · {}", r.kind.label(), r.modality);
                if !r.date.is_empty() {
                    tip.push_str(&format!(" · {}", r.date));
                }
                if !r.file_name.is_empty() {
                    tip.push_str(&format!("\n{}", r.file_name));
                }
                resp.on_hover_text(tip).context_menu(|ui| {
                    if Self::show_entry(ui, shown[row]) {
                        if shown[row] {
                            hide = true;
                        } else {
                            show = Some(i);
                        }
                        ui.close();
                    }
                    ui.separator();
                    if Self::rename_entry(ui) {
                        rename = Some(RenameTarget::Report { slot, idx: i });
                        ui.close();
                    }
                    ui.separator();
                    if Self::remove_entry(ui, "🗑 Remove") {
                        remove = Some(ObjRef {
                            slot,
                            kind: ObjKind::Report,
                            idx: i,
                        });
                        ui.close();
                    }
                });
            }
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
        if hide {
            self.hide_additional(slot);
        }
        if let Some(i) = show {
            self.show_report(slot, i);
            self.additional_on_screen(slot);
        }
    }

    /// One row of an object the additional pane shows (a report, a
    /// registration, a record): selected while it is there, with the `A`
    /// badge that says so, and the keyboard's badge when the focus is on
    /// it.
    pub(super) fn additional_row(
        ui: &mut egui::Ui,
        shown: bool,
        text: impl Into<egui::WidgetText>,
        focused: bool,
    ) -> egui::Response {
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.horizontal(|ui| {
                let resp = ui.add(egui::Button::selectable(shown, text).wrap());
                if shown {
                    ui.add(
                        egui::Button::new(egui::RichText::new("A").small().strong())
                            .small()
                            .corner_radius(2.0)
                            .sense(egui::Sense::hover()),
                    )
                    .on_hover_text("Shown in Additional");
                }
                Self::keyboard_badge(ui, focused);
                resp
            })
            .inner
        })
        .inner
    }
}
