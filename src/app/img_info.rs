//! The *Image information* module: what the displayed series actually is.
//!
//! Voxel spacing, slice thickness and the gap between slices, how many
//! slices there are, the frame of reference, the acquisition settings - the
//! things a physicist checks before registering two studies, contouring on
//! them or computing a DVH, and which otherwise take a DICOM tag browser to
//! find. The rows that deserve a second look carry the reason with them.

use super::*;
use crate::imginfo::{describe, ImageInfo, Row};

/// State of the module: which dataset it reads, and the last report of each.
#[derive(Default)]
pub(super) struct InfoState {
    pub(super) slot: usize,
    /// One cached report per dataset, with the series UID it describes, so
    /// switching between A and B does not read every header again.
    cache: [Option<(String, ImageInfo)>; 2],
    /// Show what the two datasets disagree about.
    compare: bool,
}

/// The rows worth putting side by side when two datasets are compared: the
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

    /// The cached report of `slot`, read again when the dataset now shows a
    /// different series.
    fn info_report(&mut self, slot: usize) -> Option<&ImageInfo> {
        let uid = self.slots[slot].displayed_uid()?.to_string();
        let stale = !matches!(&self.info.cache[slot], Some((u, _)) if *u == uid);
        if stale {
            let study = self.slots[slot].study.as_ref()?;
            let series = study.series.get(study.active_series)?;
            let report = describe(series, &study.volume);
            self.info.cache[slot] = Some((uid, report));
        }
        self.info.cache[slot].as_ref().map(|(_, r)| r)
    }

    fn image_info_body(&mut self, ui: &mut egui::Ui) {
        if !self.any_volume() {
            ui.weak("Load a dataset with an image volume");
            return;
        }
        if !self.slots[self.info.slot].has_volume() {
            self.info.slot = self.first_volume_slot();
        }
        if let Some(s) = seg_engines::dataset_row(ui, self.info.slot, self.volume_slots(), true) {
            self.info.slot = s;
        }
        let slot = self.info.slot;
        let both = self.volume_slots() == [true, true];

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
                    .on_hover_text("Show what the two datasets disagree about");
            }
        });
        if refresh {
            self.info.cache[slot] = None;
        }

        let Some(report) = self.info_report(slot).cloned() else {
            ui.weak("This dataset shows no image series");
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
                .show(ui, |ui| info_rows(ui, rows));
        }

        if both && self.info.compare {
            let other = 1 - slot;
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
                let mut differ = 0;
                for label in COMPARED {
                    let (Some(a), Some(b)) = (find(&mine, label), find(&yours, label)) else {
                        continue;
                    };
                    if a == b {
                        continue;
                    }
                    differ += 1;
                    ui.label(label);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.add_space(10.0);
                        ui.weak(format!("{}:", SLOT_NAMES[slot]));
                        ui.monospace(short(a));
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        ui.add_space(10.0);
                        ui.weak(format!("{}:", SLOT_NAMES[other]));
                        ui.monospace(short(b));
                    });
                }
                if differ == 0 {
                    ui.colored_label(
                        theme::good_color(ui.visuals()),
                        "The two datasets agree on all of it",
                    );
                } else if find(&mine, "Frame of reference") != find(&yours, "Frame of reference") {
                    ui.add_space(2.0);
                    ui.weak(
                        "Different frames of reference: nothing but a registration relates \
                         the two, and a structure set of one does not belong to the other.",
                    );
                }
            });
        }
    }
}

fn info_rows(ui: &mut egui::Ui, rows: &[Row]) {
    for r in rows {
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            ui.weak(format!("{}:", r.label));
            let text = egui::RichText::new(short(&r.value)).monospace();
            match &r.note {
                Some(_) => ui.colored_label(theme::warn_color(ui.visuals()), text),
                None => ui.label(text),
            };
        })
        .response
        .on_hover_text(match &r.note {
            Some(n) => format!("{}\n\n{}", r.value, n),
            None => r.value.clone(),
        });
    }
}

/// A long value (a UID, a path) shortened for the 320 px panel; the hover
/// text always carries the whole thing.
fn short(v: &str) -> String {
    const MAX: usize = 34;
    if v.chars().count() <= MAX {
        return v.to_string();
    }
    let tail: String = v.chars().skip(v.chars().count() - (MAX - 1)).collect();
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
    use super::short;

    #[test]
    fn a_long_value_is_shortened_from_the_front() {
        // A UID is only ever recognised by its tail; the head is the same
        // organisation root on every one of them.
        let uid = "1.2.826.0.1.3680043.8.498.12345678901234567890123456789012";
        let s = short(uid);
        assert!(s.chars().count() <= 34, "{s}");
        assert!(s.starts_with('\u{2026}'));
        assert!(uid.ends_with(s.trim_start_matches('\u{2026}')));
        // Anything that fits is left exactly as it is.
        assert_eq!(short("2 x 2 x 2 mm"), "2 x 2 x 2 mm");
        assert_eq!(short(""), "");
    }
}
