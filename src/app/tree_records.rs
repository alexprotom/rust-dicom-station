//! The data tree's records and warnings sections: treatment records - in
//! the study's own list, or as the rows of its treatment sessions
//! ([`super::tree_sessions`]), each shown in full in the additional pane -
//! and what loading the study had to say.

use super::object_view::PaneObject;
use super::*;
use crate::extras::TreatRecord;

impl ViewerApp {
    /// The study's own RT (Ion) Beams Treatment Records, those no treatment
    /// session took: per-beam delivered metersets.
    pub(super) fn records_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        (pi, si): (usize, usize),
        records: &[usize],
    ) {
        if records.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(format!("Treatment records ({})", records.len()))
            .id_salt(("records", slot, pi, si))
            .default_open(false)
            .show(ui, |ui| self.record_rows(ui, slot, records, false));
    }

    /// The rows of the records `records`, by name, fraction and date - or
    /// `compact`, for a session: by the time, the beams and how they ended.
    /// A click shows the record's beams and metersets in the additional
    /// pane ([`super::object_view`]).
    pub(super) fn record_rows(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        records: &[usize],
        compact: bool,
    ) {
        let mut show: Option<usize> = None;
        let mut hide = false;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        let shown: Vec<bool> = records
            .iter()
            .map(|&ri| self.object_shown(slot, PaneObject::Record(ri)))
            .collect();
        {
            let Some(study) = &self.slots[slot].study else {
                return;
            };
            for (row, &ri) in records.iter().enumerate() {
                let Some(rec) = study.treat_records.get(ri) else {
                    continue;
                };
                let me = RenameTarget::Record { slot, idx: ri };
                let text = if compact {
                    compact_title(rec)
                } else {
                    format!(
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
                    )
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
                        show = Some(ri);
                    }
                }
                let tip = Self::object_tip(
                    &rec.sop_instance_uid,
                    if rec.machine.is_empty() {
                        rec.label.clone()
                    } else {
                        format!("{} · {}", rec.label, rec.machine)
                    },
                );
                resp.on_hover_text(tip).context_menu(|ui| {
                    if Self::show_entry(ui, shown[row]) {
                        if shown[row] {
                            hide = true;
                        } else {
                            show = Some(ri);
                        }
                        ui.close();
                    }
                    ui.separator();
                    if Self::rename_entry(ui) {
                        rename = Some(me.clone());
                        ui.close();
                    }
                    ui.separator();
                    if Self::remove_entry(ui, "🗑 Remove") {
                        remove = Some(ObjRef {
                            slot,
                            kind: ObjKind::Record,
                            idx: ri,
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
        if let Some(ri) = show {
            self.show_object(slot, PaneObject::Record(ri));
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

/// A record in one line, for a session: how its beams ended (`✔` for
/// `NORMAL`, `⚠` for anything else), its time, its beams and the status.
fn compact_title(rec: &TreatRecord) -> String {
    let status = crate::sessions::record_status(rec);
    let mark = match status {
        "NORMAL" => "✔ ",
        "" => "",
        _ => "⚠ ",
    };
    let mut t = format!("{mark}Record");
    let clock = crate::sessions::clock(&rec.datetime);
    if !clock.is_empty() {
        t.push_str(&format!(" · {clock}"));
    }
    let beams: Vec<&str> = rec.beams.iter().map(|b| b.name.as_str()).collect();
    if !beams.is_empty() {
        t.push_str(&format!(" · {}", beams.join(" + ")));
    }
    if !status.is_empty() {
        t.push_str(&format!(" · {status}"));
    }
    t
}
