//! The *Image information* module: what the displayed series actually is.
//!
//! Voxel spacing, slice thickness and the gap between slices, how many
//! slices there are, the frame of reference, the acquisition settings - the
//! things a physicist checks before registering two studies, contouring on
//! them or computing a DVH, and which otherwise take a DICOM tag browser to
//! find. The rows that deserve a second look carry the reason with them.

use super::*;
use crate::imginfo::{describe, ImageInfo, Row};

/// State of the module: which workspace it reads, and the last report of each.
#[derive(Default)]
pub(super) struct InfoState {
    pub(super) slot: usize,
    /// One cached report per workspace, with the series UID and the volume
    /// it describes, so switching between A and B does not read every header
    /// again - and a volume that replaced another under the same series is
    /// described afresh rather than reported with the last one's lattice.
    cache: [Option<(String, usize, ImageInfo)>; MAX_WORKSPACES],
    /// Show what this workspace and another disagree about.
    compare: bool,
    /// Which workspace to hold it against. None means "the next open one",
    /// which is the answer whenever there are only two.
    against: Option<usize>,
}

/// The rows worth putting side by side when two workspaces are compared: the
/// ones that decide whether they can be registered and measured together.
const COMPARED: [&str; 10] = [
    "Modality",
    "Patient position",
    "Dimensions",
    "Voxel spacing",
    "Field of view",
    "Slices",
    "Slice thickness",
    "Slice gap",
    "Row direction",
    "Frame of reference",
];

impl ViewerApp {
    pub(super) fn image_info_section(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("Image information");
        let state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        let header = state.show_header(ui, |ui| {
            ui.label(egui::RichText::new("Image information").strong());
        });
        header.body(|ui| self.image_info_body(ui));
        ui.separator();
    }

    /// The cached report of `slot`, read again when the workspace now shows a
    /// different series.
    fn info_report(&mut self, slot: usize) -> Option<&ImageInfo> {
        let uid = self.slots[slot].displayed_uid()?.to_string();
        let study = self.slots[slot].study.as_ref()?;
        let vol = Arc::as_ptr(&study.volume) as usize;
        let stale = !matches!(&self.info.cache[slot], Some((u, v, _)) if *u == uid && *v == vol);
        if stale {
            let series = study.series.get(study.active_series)?;
            let report = describe(series, &study.volume);
            self.info.cache[slot] = Some((uid, vol, report));
        }
        self.info.cache[slot].as_ref().map(|(_, _, r)| r)
    }

    fn image_info_body(&mut self, ui: &mut egui::Ui) {
        if !self.any_volume() {
            ui.weak("Load a workspace with an image volume");
            return;
        }
        if !self.slots[self.info.slot].has_volume() {
            self.info.slot = self.first_volume_slot();
        }
        if let Some(s) = seg_engines::workspace_row(ui, self.info.slot, self.volume_slots(), true) {
            self.info.slot = s;
        }
        let slot = self.info.slot;
        let others: Vec<usize> = self
            .volume_slot_list()
            .into_iter()
            .filter(|s| *s != slot)
            .collect();
        let both = !others.is_empty();

        let mut refresh = false;
        let mut copy = false;
        ui.horizontal(|ui| {
            if small_tip_button(
                ui,
                "Read again",
                "Read the series' headers again, after replacing the files on disk",
            ) {
                refresh = true;
            }
            if small_tip_button(ui, "Copy", "Put the whole report on the clipboard") {
                copy = true;
            }
            if both {
                ui.checkbox(&mut self.info.compare, "Compare")
                    .on_hover_text("Show what this workspace and another disagree about");
                // Which one to hold it against. With one other workspace
                // there is nothing to choose and the row stays quiet.
                if self.info.compare && others.len() > 1 {
                    let current = self
                        .info
                        .against
                        .filter(|s| others.contains(s))
                        .unwrap_or(others[0]);
                    ui.label("against");
                    for o in &others {
                        if ui
                            .add(egui::Button::selectable(current == *o, SLOT_NAMES[*o]).small())
                            .clicked()
                        {
                            self.info.against = Some(*o);
                        }
                    }
                }
            }
        });
        if refresh {
            self.info.cache[slot] = None;
        }

        let Some(report) = self.info_report(slot).cloned() else {
            ui.weak("This workspace shows no image series");
            return;
        };
        if copy {
            ui.ctx().copy_text(report.text());
        }
        ui.label(egui::RichText::new(&report.title).strong());
        if report.failed > 0 {
            ui.colored_label(
                theme::warn_color(ui.visuals()),
                format!(
                    "{} of {} files could not be read",
                    report.failed,
                    report.failed + report.read
                ),
            );
        }

        // Everything that wants attention, before the detail that explains
        // it: this is the reason to open the module at all.
        let warnings = report.warnings();
        if warnings.is_empty() {
            ui.colored_label(theme::good_color(ui.visuals()), "Nothing unusual");
        } else {
            for w in &warnings {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    ui.colored_label(theme::warn_color(ui.visuals()), "⚠");
                    ui.label(format!("{}: {}", w.label, w.value));
                })
                .response
                .on_hover_text(w.note.clone().unwrap_or_default());
                if let Some(n) = &w.note {
                    ui.horizontal_wrapped(|ui| {
                        ui.add_space(14.0);
                        ui.weak(n);
                    });
                }
            }
        }
        ui.add_space(4.0);

        for (name, rows) in &report.sections {
            egui::CollapsingHeader::new(name)
                .id_salt(("img_info", name))
                .default_open(name == "Sampling")
                .show(ui, |ui| info_table(ui, name, rows));
        }

        if both && self.info.compare {
            let other = self
                .info
                .against
                .filter(|s| others.contains(s))
                .unwrap_or(others[0]);
            let Some(theirs) = self.info_report(other).cloned() else {
                return;
            };
            egui::CollapsingHeader::new(format!(
                "{} against {}",
                SLOT_NAMES[slot], SLOT_NAMES[other]
            ))
            .id_salt("img_info_cmp")
            .default_open(true)
            .show(ui, |ui| {
                let mine = flatten(&report);
                let yours = flatten(&theirs);
                // Only what differs: the rows they agree on are the ones
                // nobody opened the comparison for.
                let differ: Vec<(&str, &str, &str)> = COMPARED
                    .iter()
                    .filter_map(|label| {
                        let (a, b) = (find(&mine, label)?, find(&yours, label)?);
                        (a != b).then_some((*label, a, b))
                    })
                    .collect();
                if differ.is_empty() {
                    ui.colored_label(
                        theme::good_color(ui.visuals()),
                        "The two workspaces agree on all of it",
                    );
                } else {
                    let room = value_room(
                        ui,
                        differ
                            .iter()
                            .map(|d| d.0)
                            .chain(std::iter::once("Property")),
                        2,
                    );
                    egui::ScrollArea::horizontal()
                        .id_salt("img_info_cmp_scroll")
                        .show(ui, |ui| {
                            egui::Grid::new("img_info_cmp_grid")
                                .num_columns(3)
                                .striped(true)
                                .spacing([10.0, 2.0])
                                .show(ui, |ui| {
                                    run_report::head(ui, "Property");
                                    run_report::head(ui, SLOT_NAMES[slot]);
                                    run_report::head(ui, SLOT_NAMES[other]);
                                    ui.end_row();
                                    for (label, a, b) in &differ {
                                        ui.label(*label);
                                        value_cell(ui, a, None, room).on_hover_text(*a);
                                        value_cell(ui, b, None, room).on_hover_text(*b);
                                        ui.end_row();
                                    }
                                });
                        });
                    if find(&mine, "Frame of reference") != find(&yours, "Frame of reference") {
                        ui.add_space(2.0);
                        ui.weak(
                            "Different frames of reference: nothing but a registration relates \
                             the two, and a structure set of one does not belong to the other.",
                        );
                    }
                }
            });
        }
    }
}

/// One section of the report as a two-column table of property and value,
/// the way every other module lays out a list of facts. A value to look
/// twice at is in the warning colour, with the reason on its tooltip.
fn info_table(ui: &mut egui::Ui, section: &str, rows: &[Row]) {
    let room = value_room(ui, rows.iter().map(|r| r.label.as_str()), 1);
    let warn = theme::warn_color(ui.visuals());
    egui::ScrollArea::horizontal()
        .id_salt(("img_info_scroll", section))
        .show(ui, |ui| {
            egui::Grid::new(("img_info_grid", section))
                .num_columns(2)
                .striped(true)
                .spacing([10.0, 2.0])
                .show(ui, |ui| {
                    for r in rows {
                        ui.label(&r.label);
                        let color = r.note.as_ref().map(|_| warn);
                        value_cell(ui, &r.value, color, room).on_hover_text(match &r.note {
                            Some(n) => format!("{}\n\n{}", r.value, n),
                            None => r.value.clone(),
                        });
                        ui.end_row();
                    }
                });
        });
}

/// How much width a value has, in points, to fit the panel beside its
/// property's name, with `columns` values to a row sharing what is left. A
/// table wider than the panel would hide its values' ends behind a scroll
/// bar.
fn value_room<'a>(ui: &egui::Ui, labels: impl Iterator<Item = &'a str>, columns: usize) -> f32 {
    let color = ui.visuals().text_color();
    let body = egui::TextStyle::Body.resolve(ui.style());
    let painter = ui.painter();
    let label_w = labels
        .map(|l| {
            painter
                .layout_no_wrap(l.to_string(), body.clone(), color)
                .size()
                .x
        })
        .fold(0.0f32, f32::max);
    // The grid's spacing between columns, and a little for the stripe's
    // own margin and the scroll area's.
    let gaps = 10.0 * columns as f32 + 24.0;
    ((ui.available_width() - label_w - gaps) / columns.max(1) as f32).max(40.0)
}

/// One value of a table, in `room` points: a value of several words (a
/// vector, a size with its unit) wraps onto a second line rather than lose
/// a component; a single long token (a UID) is cut from the front instead,
/// keeping the tail it is told apart by. The whole value belongs on the
/// caller's tooltip either way.
fn value_cell(
    ui: &mut egui::Ui,
    value: &str,
    color: Option<egui::Color32>,
    room: f32,
) -> egui::Response {
    let mono = egui::TextStyle::Monospace.resolve(ui.style());
    let char_w = ui
        .painter()
        .layout_no_wrap("0".into(), mono, ui.visuals().text_color())
        .size()
        .x
        .max(1.0);
    let text = if value.contains(char::is_whitespace) {
        value.to_string()
    } else {
        short_to(value, ((room / char_w).floor() as usize).max(6))
    };
    let mut text = egui::RichText::new(text).monospace();
    if let Some(c) = color {
        text = text.color(c);
    }
    // A vertical scope of bounded width is what lets a label wrap inside a
    // grid cell without spilling into the next row.
    ui.vertical(|ui| {
        ui.set_max_width(room);
        ui.add(egui::Label::new(text).wrap())
    })
    .inner
}

/// A long value (a UID, a path) shortened to `max` characters for the
/// panel, from the front; the hover text always carries the whole thing.
fn short_to(v: &str, max: usize) -> String {
    let n = v.chars().count();
    if n <= max {
        return v.to_string();
    }
    let tail: String = v.chars().skip(n - (max - 1)).collect();
    format!("\u{2026}{tail}")
}

fn flatten(r: &ImageInfo) -> Vec<(&str, &str)> {
    r.sections
        .iter()
        .flat_map(|(_, rows)| rows.iter())
        .map(|row| (row.label.as_str(), row.value.as_str()))
        .collect()
}

fn find<'a>(rows: &[(&'a str, &'a str)], label: &str) -> Option<&'a str> {
    rows.iter().find(|(l, _)| *l == label).map(|(_, v)| *v)
}

#[cfg(test)]
mod tests {
    use super::short_to;

    #[test]
    fn a_long_value_is_shortened_from_the_front() {
        // A UID is only ever recognised by its tail; the head is the same
        // organisation root on every one of them.
        let uid = "1.2.826.0.1.3680043.8.498.12345678901234567890123456789012";
        for max in [12, 22, 34] {
            let s = short_to(uid, max);
            assert_eq!(s.chars().count(), max, "{s}");
            assert!(s.starts_with('\u{2026}'));
            assert!(uid.ends_with(s.trim_start_matches('\u{2026}')));
        }
        // Anything that fits is left exactly as it is.
        assert_eq!(short_to("2 x 2 x 2 mm", 12), "2 x 2 x 2 mm");
        assert_eq!(short_to("", 12), "");
    }
}
