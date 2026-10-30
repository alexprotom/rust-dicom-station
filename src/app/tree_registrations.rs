//! The data tree's registrations: the spatial registrations and
//! deformation fields of a study - in the study's own list, or as the rows
//! of its treatment sessions ([`super::tree_sessions`]). A row names one; a
//! click shows it in the row's additional pane, matrices and apply buttons
//! with it ([`super::object_view`]).

use super::object_view::PaneObject;
use super::*;
use crate::extras::SpatialReg;

impl ViewerApp {
    /// The study's own REG objects, those no treatment session took.
    pub(super) fn reg_objects_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        (pi, si): (usize, usize),
        regs: &[usize],
    ) {
        if regs.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(format!("Spatial registrations ({})", regs.len()))
            .id_salt(("regobj", slot, pi, si))
            .default_open(false)
            .show(ui, |ui| self.reg_rows(ui, slot, regs, false));
    }

    /// The rows of the registrations `regs`, by name - or `compact`, for a
    /// session: by what each is, with its time and, for a treatment
    /// registration, the setup correction it holds.
    pub(super) fn reg_rows(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        regs: &[usize],
        compact: bool,
    ) {
        let mut show: Option<usize> = None;
        let mut hide = false;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        let shown: Vec<bool> = regs
            .iter()
            .map(|&ri| self.object_shown(slot, PaneObject::Registration(ri)))
            .collect();
        {
            let Some(study) = self.slots[slot].study.as_ref() else {
                return;
            };
            for (row, &ri) in regs.iter().enumerate() {
                let Some(reg) = study.registrations.get(ri) else {
                    continue;
                };
                let me = RenameTarget::Registration { slot, idx: ri };
                let text = if compact {
                    compact_title(study, reg)
                } else {
                    format!(
                        "{}{}",
                        reg.label,
                        if reg.deformable { "  [deformable]" } else { "" }
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
                    &reg.sop_instance_uid,
                    if !reg.comment.is_empty() && reg.comment != reg.label {
                        format!("{} · {}", reg.label, reg.comment)
                    } else {
                        reg.label.clone()
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
                            kind: ObjKind::Registration,
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
            self.show_object(slot, PaneObject::Registration(ri));
        }
    }
}

/// A registration in one line, for a session: `⇄` and what it is - its own
/// name when it has one other than the vendor's comment (a name given
/// here), else the kind it is - with its time, and for a treatment
/// registration the correction.
fn compact_title(study: &LoadedStudy, reg: &SpatialReg) -> String {
    let kind = crate::sessions::reg_kind(reg, study);
    let named = !reg.label.is_empty() && reg.label != reg.comment;
    let what = match kind {
        _ if named => reg.label.clone(),
        crate::sessions::RegKind::Treatment => "Treatment position".to_string(),
        crate::sessions::RegKind::CtToCt => "Daily CT ▶ planning CT".to_string(),
        crate::sessions::RegKind::Imaging => {
            if reg.references.is_empty() {
                "Imaging position".to_string()
            } else if study.series.iter().any(|s| reg.references.contains(&s.uid)) {
                "Imaging position ▶ daily CT".to_string()
            } else {
                "Imaging position ▶ a series not loaded".to_string()
            }
        }
    };
    let mut t = format!("⇄ {what}");
    let clock = crate::sessions::clock(&reg.datetime);
    if !clock.is_empty() {
        t.push_str(&format!(" · {clock}"));
    }
    if kind == crate::sessions::RegKind::Treatment {
        if let Some(c) = crate::sessions::correction(reg) {
            t.push_str(&format!(" · {}", crate::sessions::correction_text(&c)));
        }
    }
    t
}
