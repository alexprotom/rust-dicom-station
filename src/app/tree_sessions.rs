//! The data tree's treatment sessions: a study's course day by day
//! ([`crate::sessions`] puts it together). A session holds the day's own
//! images (the daily CT), the registrations that place them, and a node per
//! beam - a *step*: its setup pairs, the treatment registration with the
//! correction it holds, the records of the delivery and the verification
//! pair. What a session holds is drawn here and nowhere else in the tree,
//! with the rows the study's own lists use (series, registrations,
//! records, planar images); a radiograph pair is a row of its own, which
//! shows its two images side by side in the row's two planar panes, with
//! the dose report written for the exposure under it. A row opens nothing
//! in the tree: registrations, records and reports are read in the
//! additional pane.

use super::*;

use crate::extras::PlanarKind;
use crate::sessions::{self, Pair, Session, Step};

impl ViewerApp {
    /// *Treatment sessions (n)* of one study.
    pub(super) fn sessions_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        (pi, si): (usize, usize),
        list: &[Session],
    ) {
        if list.is_empty() {
            return;
        }
        egui::CollapsingHeader::new(format!("Treatment sessions ({})", list.len()))
            .id_salt(("sessions", slot, pi, si))
            .default_open(false)
            .show(ui, |ui| {
                for s in list {
                    let title = self.session_title(slot, s);
                    let me = &mut *self;
                    Self::wrapped_node(
                        ui,
                        ("session", slot, pi, si, &s.date),
                        false,
                        title,
                        |ui| me.session_body(ui, slot, (pi, si), s),
                    );
                }
            });
    }

    /// `2024-09-17 · CT · 4 of 16 · LPO ✔ · PA ⚠✔`: the date, the modality
    /// of the images made that day (the daily CT), which session of the
    /// course it is, and each beam with a mark per record of it.
    fn session_title(&self, slot: usize, s: &Session) -> String {
        let mut t = sessions::date_label(&s.date);
        let Some(study) = self.slots[slot].study.as_ref() else {
            return format!("{t} · {} of {}", s.number, s.of);
        };
        let mut modalities: Vec<&str> = Vec::new();
        for m in s
            .series
            .iter()
            .filter_map(|&i| study.series.get(i))
            .map(|se| se.modality.as_str())
        {
            if !m.is_empty() && !modalities.contains(&m) {
                modalities.push(m);
            }
        }
        if !modalities.is_empty() {
            t.push_str(&format!(" · {}", modalities.join(" + ")));
        }
        t.push_str(&format!(" · {} of {}", s.number, s.of));
        // The steps, then the records no step took grouped by beam as they
        // follow each other.
        let mut beams: Vec<(String, String)> = s
            .steps
            .iter()
            .map(|st| (st.name.clone(), marks(study, &st.records)))
            .collect();
        for &r in &s.loose_records {
            let Some(rec) = study.treat_records.get(r) else {
                continue;
            };
            let name = rec
                .beams
                .iter()
                .map(|b| b.name.as_str())
                .collect::<Vec<_>>()
                .join(" + ");
            let mark = marks(study, &[r]);
            match beams.last_mut() {
                Some((n, m)) if *n == name && s.steps.is_empty() => m.push_str(&mark),
                _ => beams.push((name, mark)),
            }
        }
        for (name, mark) in beams {
            t.push_str(&format!(" · {name}"));
            if !mark.is_empty() {
                t.push_str(&format!(" {mark}"));
            }
        }
        t
    }

    /// One session: its images, the day's registrations, its steps, then
    /// what no step took.
    fn session_body(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        (pi, si): (usize, usize),
        s: &Session,
    ) {
        if !s.series.is_empty() {
            self.series_rows(ui, slot, &s.series);
        }
        let mut day_regs: Vec<usize> = s.image_regs.iter().chain(&s.ct_regs).copied().collect();
        if let Some(study) = self.slots[slot].study.as_ref() {
            day_regs.sort_by_key(|&r| {
                (
                    study
                        .registrations
                        .get(r)
                        .map(|x| x.datetime.clone())
                        .unwrap_or_default(),
                    r,
                )
            });
        }
        self.reg_rows(ui, slot, &day_regs, true);
        for (k, step) in s.steps.iter().enumerate() {
            let title = self.step_title(slot, step);
            let me = &mut *self;
            Self::wrapped_node(ui, ("step", slot, pi, si, &s.date, k), true, title, |ui| {
                me.step_body(ui, slot, step)
            });
        }
        if !s.loose_records.is_empty() {
            self.record_rows(ui, slot, &s.loose_records, true);
        }
        for p in &s.loose_pairs {
            self.pair_row(
                ui,
                slot,
                p,
                format!("Pair · {}", sessions::clock(&p.acquired)),
            );
        }
        if !s.loose_reports.is_empty() {
            self.report_rows(ui, slot, &s.loose_reports, Some(&report_title));
        }
    }

    /// `LPO ⚠✔`: the beam and a mark per record.
    fn step_title(&self, slot: usize, step: &Step) -> String {
        let mark = self.slots[slot]
            .study
            .as_ref()
            .map(|st| marks(st, &step.records))
            .unwrap_or_default();
        if mark.is_empty() {
            step.name.clone()
        } else {
            format!("{} {mark}", step.name)
        }
    }

    /// One beam, in the order it happened: the setup iterations, the
    /// correction, the delivery, the verification.
    fn step_body(&mut self, ui: &mut egui::Ui, slot: usize, step: &Step) {
        let n = step.setup.len();
        for (j, p) in step.setup.iter().enumerate() {
            let title = format!(
                "Setup pair {} · {}{}",
                j + 1,
                sessions::clock(&p.acquired),
                if j + 1 == n && n > 1 { " (final)" } else { "" }
            );
            self.pair_row(ui, slot, p, title);
        }
        if let Some(r) = step.treatment_reg {
            self.reg_rows(ui, slot, &[r], true);
        }
        self.record_rows(ui, slot, &step.records, true);
        for p in &step.verification {
            let title = format!("Verification pair · {}", sessions::clock(&p.acquired));
            self.pair_row(ui, slot, p, title);
        }
    }

    /// A radiograph pair: a click shows its first image in the first planar
    /// pane and its second in the second (Ctrl-click puts them on the
    /// panes' stacks to compare), and the row unfolds to the two images as
    /// ordinary planar rows.
    fn pair_row(&mut self, ui: &mut egui::Ui, slot: usize, pair: &Pair, title: String) {
        let Some(&first) = pair.images.first() else {
            return;
        };
        let shown = pair
            .images
            .iter()
            .enumerate()
            .all(|(k, &i)| self.planar_panes_showing(slot, i).contains(&k));
        let ctrl = ui.input(|i| i.modifiers.command);
        let me = RenameTarget::Pair {
            slot,
            images: pair.images.clone(),
        };
        let current = self.tree_focus.clone();
        let mut focus = None;
        let mut show: Option<bool> = None; // Some(stack)
        let mut hide = false;
        let mut remove = false;
        let tip = format!(
            "{}{}",
            if pair.images.len() == 2 {
                "Two images taken together"
            } else {
                "An image whose partner is missing"
            },
            if pair.reviewed.is_empty() {
                String::new()
            } else {
                format!(" · reviewed {}", sessions::clock(&pair.reviewed))
            },
        );
        let body = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            ui.make_persistent_id(("pair", slot, first)),
            false,
        )
        .show_header(ui, |ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            let resp = ui.add(egui::Button::selectable(shown, title).wrap());
            Self::keyboard_badge(ui, current.as_ref() == Some(&me));
            Self::focus_row(ui, &resp, me.clone(), &current, &mut focus);
            // A click shows the pair; a click on it while it is shown
            // takes it off the screen again.
            if resp.clicked() {
                if shown && !ctrl {
                    hide = true;
                } else {
                    show = Some(ctrl);
                }
            }
            resp.on_hover_text(tip).context_menu(|ui| {
                if Self::show_entry(ui, shown) {
                    if shown {
                        hide = true;
                    } else {
                        show = Some(false);
                    }
                    ui.close();
                }
                if ui
                    .button("➕ Compare in Planar 1 and 2")
                    .on_hover_text("Put them on top of what the panes show")
                    .clicked()
                {
                    show = Some(true);
                    ui.close();
                }
                ui.separator();
                if Self::remove_entry(
                    ui,
                    if pair.images.len() == 2 {
                        "🗑 Remove both images"
                    } else {
                        "🗑 Remove the image"
                    },
                ) {
                    remove = true;
                    ui.close();
                }
            });
        });
        body.body(|ui| self.planar_rows(ui, slot, PlanarKind::Projection, &pair.images));
        // The dose report written for the exposure, right under it.
        if !pair.reports.is_empty() {
            self.report_rows(ui, slot, &pair.reports, Some(&report_title));
        }
        if focus.is_some() {
            self.tree_focus = focus;
        }
        if remove {
            self.objs_remove = pair
                .images
                .iter()
                .map(|&idx| ObjRef {
                    slot,
                    kind: ObjKind::Planar,
                    idx,
                })
                .collect();
            return;
        }
        if hide {
            for (k, &i) in pair.images.iter().enumerate() {
                self.take_planar_image(slot, k, i);
            }
            return;
        }
        if let Some(stack) = show {
            self.show_pair(slot, &pair.images, stack);
        }
    }

    /// A pair's images side by side, the first in Planar 1 and the second in
    /// Planar 2 - or (`stack`) on top of what those panes show.
    pub(super) fn show_pair(&mut self, slot: usize, images: &[usize], stack: bool) {
        for (k, &i) in images.iter().enumerate() {
            if stack {
                self.stack_planar(slot, i, Some(k));
            } else {
                self.show_planar(slot, i, Some(k));
            }
        }
        self.planar_on_screen(slot);
    }
}

/// A report of a session in one line: `Dose report · 10:51:46`, a report
/// of another kind by its label.
fn report_title(r: &crate::report::Report) -> String {
    let what = match r.kind {
        crate::report::ReportKind::DoseReport => "Dose report",
        _ => r.label.as_str(),
    };
    let clock = sessions::clock(&r.datetime);
    if clock.is_empty() {
        what.to_string()
    } else {
        format!("{what} · {clock}")
    }
}

/// A mark per record: `✔` for one whose beams all ended `NORMAL`, `⚠` for
/// one that did not.
fn marks(study: &LoadedStudy, records: &[usize]) -> String {
    records
        .iter()
        .filter_map(|&r| study.treat_records.get(r))
        .map(|rec| match sessions::record_status(rec) {
            "NORMAL" => "✔",
            _ => "⚠",
        })
        .collect()
}
