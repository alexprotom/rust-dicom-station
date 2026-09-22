//! What a run leaves behind, as a table rather than a paragraph.
//!
//! A propagation onto ten phases reports four things per phase - what the
//! registration did to the metric, how well the anchor landed, what each
//! structure's volume became, and where it was filed - and written out as
//! sentences that is forty lines nobody reads to the end. The same numbers
//! in two grids are read by scanning a column: the phase whose Dice dropped,
//! the structure whose volume ran away.
//!
//! The report holds the numbers, not their formatting, so the same run can
//! be drawn on screen and copied to the clipboard as a table a spreadsheet
//! opens.

use super::*;
use crate::registration::RunMetrics;

/// One destination: a phase of a 4D group, or the single image a plain run
/// landed on.
#[derive(Clone, Default)]
pub(super) struct RunBlock {
    /// The phase's name, or empty when there is only one destination.
    pub(super) label: String,
    /// What the registration did, and what it was: `None` where the
    /// transform was reused or given rather than recovered.
    pub(super) metrics: Option<RunMetrics>,
    /// The line the run wrote about itself, kept for the row's tooltip -
    /// it names the stages, which the columns do not.
    pub(super) detail: String,
    /// The anchor's own overlap, when the run was anchored on a structure.
    pub(super) check: Option<RunCheck>,
    /// One per structure carried.
    pub(super) items: Vec<RunItem>,
    /// Where they were filed, when they went somewhere with a name.
    pub(super) landed: Option<String>,
}

/// How well the structure the run was anchored on landed on its own contour.
#[derive(Clone)]
pub(super) struct RunCheck {
    pub(super) name: String,
    /// `None` when the anchor did not land at all.
    pub(super) dice: Option<f64>,
    pub(super) hd95_mm: f64,
    pub(super) centroid_mm: f64,
    pub(super) verdict: &'static str,
}

/// One structure, as it arrived.
#[derive(Clone)]
pub(super) struct RunItem {
    pub(super) name: String,
    pub(super) source_cm3: f64,
    /// What the transform made of it, before it was filed on the
    /// destination lattice.
    pub(super) mapped_cm3: f64,
    pub(super) result_cm3: f64,
}

impl RunItem {
    fn change_pct(&self) -> f64 {
        if self.source_cm3 > 1e-9 {
            100.0 * (self.result_cm3 - self.source_cm3) / self.source_cm3
        } else {
            0.0
        }
    }
}

/// Everything one run has to say.
#[derive(Clone, Default)]
pub(super) struct RunReport {
    pub(super) blocks: Vec<RunBlock>,
    /// A line above the tables for what has no column of its own.
    pub(super) note: String,
    /// Did closing or filling run? When neither did, the filed volume is
    /// the deformed one to the last decimal and the column is a copy of its
    /// neighbour, so it is left out.
    pub(super) finished: bool,
}

impl RunReport {
    pub(super) fn is_empty(&self) -> bool {
        self.blocks.is_empty() && self.note.is_empty()
    }

    /// Does any block name a phase? A plain run does not, and then the
    /// phase column is a column of one repeated dash.
    fn phased(&self) -> bool {
        self.blocks.iter().any(|b| !b.label.is_empty())
    }

    /// Is the filed volume worth a column of its own? Only when something
    /// was done to the mask after it landed.
    fn filed_column(&self) -> bool {
        self.finished
    }

    fn any_metrics(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| b.metrics.is_some() || b.check.is_some() || b.landed.is_some())
    }

    /// The whole report as tab-separated tables - what the clipboard button
    /// puts down, and what a spreadsheet opens without being asked twice.
    pub(super) fn to_tsv(&self) -> String {
        let mut out = String::new();
        if !self.note.is_empty() {
            out.push_str(&self.note);
            out.push('\n');
        }
        if self.any_metrics() {
            out.push_str("phase\tmetric\tbefore\tafter\titerations\tseconds\tanchor\tdice\thd95_mm\tcentroid_mm\tlanded\n");
            for b in &self.blocks {
                let m = b.metrics.unwrap_or_default();
                let c = b.check.as_ref();
                out.push_str(&format!(
                    "{}\t{}\t{:.1}\t{:.1}\t{}\t{:.1}\t{}\t{}\t{}\t{}\t{}\n",
                    b.label,
                    if b.metrics.is_some() { m.tag } else { "" },
                    m.initial,
                    m.final_value,
                    m.iterations,
                    m.secs,
                    c.map(|c| c.name.as_str()).unwrap_or(""),
                    c.and_then(|c| c.dice)
                        .map(|d| format!("{d:.3}"))
                        .unwrap_or_default(),
                    c.map(|c| format!("{:.1}", c.hd95_mm)).unwrap_or_default(),
                    c.map(|c| format!("{:.1}", c.centroid_mm))
                        .unwrap_or_default(),
                    b.landed.clone().unwrap_or_default(),
                ));
            }
            out.push('\n');
        }
        let filed = self.filed_column();
        out.push_str("phase\tstructure\tsource_cm3\tdeformed_cm3");
        if filed {
            out.push_str("\tfiled_cm3");
        }
        out.push_str("\tchange_pct\n");
        for b in &self.blocks {
            for it in &b.items {
                out.push_str(&format!(
                    "{}\t{}\t{:.3}\t{:.3}",
                    b.label, it.name, it.source_cm3, it.mapped_cm3
                ));
                if filed {
                    out.push_str(&format!("\t{:.3}", it.result_cm3));
                }
                out.push_str(&format!("\t{:+.2}\n", it.change_pct()));
            }
        }
        out
    }

    /// Draw it: the runs above, the volumes below, both scrollable and both
    /// narrow enough for the modules panel.
    pub(super) fn ui(&self, ui: &mut egui::Ui, id: &str) {
        if !self.note.is_empty() {
            ui.label(&self.note);
        }
        let phased = self.phased();
        if self.any_metrics() {
            egui::ScrollArea::horizontal()
                .id_salt((id, "runs"))
                .show(ui, |ui| {
                    egui::Grid::new((id, "runs_grid"))
                        .striped(true)
                        .spacing([10.0, 2.0])
                        .show(ui, |ui| {
                            if phased {
                                head(ui, "Phase");
                            }
                            head(ui, "Metric ▶");
                            head(ui, "Iters");
                            head(ui, "t, s");
                            head(ui, "Dice");
                            head(ui, "Filed as");
                            ui.end_row();
                            for b in &self.blocks {
                                if phased {
                                    ui.label(&b.label);
                                }
                                match &b.metrics {
                                    Some(m) => {
                                        ui.monospace(format!(
                                            "{} {:.0} ▶ {:.0}",
                                            m.tag, m.initial, m.final_value
                                        ))
                                        .on_hover_text(&b.detail);
                                        ui.monospace(format!("{}", m.iterations));
                                        ui.monospace(format!("{:.1}", m.secs));
                                    }
                                    None => {
                                        ui.weak(if b.detail.is_empty() {
                                            "-".to_string()
                                        } else {
                                            b.detail.clone()
                                        });
                                        ui.weak("-");
                                        ui.weak("-");
                                    }
                                }
                                match &b.check {
                                    Some(c) => {
                                        let text = match c.dice {
                                            Some(d) => format!("{d:.2} {}", c.verdict),
                                            None => "did not land".to_string(),
                                        };
                                        ui.monospace(text).on_hover_text(format!(
                                            "{}: HD95 {:.1} mm, centroids {:.1} mm apart",
                                            c.name, c.hd95_mm, c.centroid_mm
                                        ));
                                    }
                                    None => {
                                        ui.weak("-");
                                    }
                                }
                                match &b.landed {
                                    Some(l) => {
                                        ui.label(ellipsis(l, 28)).on_hover_text(l);
                                    }
                                    None => {
                                        ui.weak("-");
                                    }
                                }
                                ui.end_row();
                            }
                        });
                });
            ui.add_space(4.0);
        }
        if self.blocks.iter().all(|b| b.items.is_empty()) {
            return;
        }
        let filed = self.filed_column();
        egui::ScrollArea::horizontal()
            .id_salt((id, "vols"))
            .show(ui, |ui| {
                egui::Grid::new((id, "vols_grid"))
                    .striped(true)
                    .spacing([10.0, 2.0])
                    .show(ui, |ui| {
                        if phased {
                            head(ui, "Phase");
                        }
                        head(ui, "Structure");
                        head(ui, "Source cm³");
                        head(ui, "Deformed cm³");
                        if filed {
                            head(ui, "Filed cm³");
                        }
                        head(ui, "Δ");
                        ui.end_row();
                        for b in &self.blocks {
                            for it in &b.items {
                                if phased {
                                    ui.label(&b.label);
                                }
                                ui.label(ellipsis(&it.name, 24)).on_hover_text(&it.name);
                                ui.monospace(format!("{:.3}", it.source_cm3));
                                ui.monospace(format!("{:.3}", it.mapped_cm3));
                                if filed {
                                    ui.monospace(format!("{:.3}", it.result_cm3));
                                }
                                let d = it.change_pct();
                                // A volume that moved by more than a tenth
                                // is the one to look at first, so it says so
                                // in colour rather than in a footnote.
                                let text = egui::RichText::new(format!("{d:+.2} %"));
                                ui.monospace(if d.abs() > 10.0 {
                                    text.color(theme::warn_color(ui.visuals()))
                                } else {
                                    text
                                });
                                ui.end_row();
                            }
                        }
                    });
            });
    }
}

/// A column heading: strong, at the table's own size - a heading set
/// smaller than its column reads as a footnote to it.
///
/// Public within the module tree so every table in the program - the run
/// report here, the registration's own blocks - heads its columns the same
/// way.
pub(super) fn head(ui: &mut egui::Ui, text: &str) {
    ui.label(egui::RichText::new(text).strong());
}

/// A two-column table of name and value, for the blocks that are a list of
/// facts rather than rows of one kind: what a registration was, what it
/// cost, what it left behind.
///
/// A row whose value is empty is left out, so a caller can hand over the
/// full list and let the ones that do not apply fall away.
pub(super) fn facts(ui: &mut egui::Ui, id: &str, rows: &[(&str, String)]) {
    egui::Grid::new((id, "facts"))
        .num_columns(2)
        .striped(true)
        .spacing([10.0, 2.0])
        .show(ui, |ui| {
            for (name, value) in rows.iter().filter(|(_, v)| !v.is_empty()) {
                head(ui, name);
                ui.label(value);
                ui.end_row();
            }
        });
}

/// Long names are cut rather than allowed to set a column's width; the whole
/// name is on the row's tooltip.
fn ellipsis(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let keep: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{keep}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(name: &str, from: f64, to: f64) -> RunItem {
        RunItem {
            name: name.into(),
            source_cm3: from,
            mapped_cm3: to,
            result_cm3: to,
        }
    }

    #[test]
    fn the_change_column_is_the_volume_the_transform_added_or_took() {
        let it = item("target", 0.618, 0.629);
        assert!(
            (it.change_pct() - 1.7799).abs() < 1e-3,
            "0.618 to 0.629 is +1.78 %, got {}",
            it.change_pct()
        );
        // A structure that was empty to begin with has no percentage to
        // report, and saying "+inf %" would be worse than saying nothing.
        assert_eq!(item("gone", 0.0, 1.0).change_pct(), 0.0);
    }

    #[test]
    fn the_clipboard_form_is_a_table_a_spreadsheet_opens() {
        let report = RunReport {
            blocks: vec![
                RunBlock {
                    label: "0%".into(),
                    metrics: Some(RunMetrics {
                        tag: "MSD",
                        initial: 356557.3,
                        final_value: 23902.1,
                        iterations: 1800,
                        secs: 0.8,
                    }),
                    detail: "rigid then B-spline".into(),
                    check: Some(RunCheck {
                        name: "heart_total".into(),
                        dice: Some(0.96),
                        hd95_mm: 4.2,
                        centroid_mm: 1.3,
                        verdict: "good",
                    }),
                    items: vec![item("target", 0.618, 0.629)],
                    landed: Some("4DCT 0% RTSTRUCT".into()),
                },
                RunBlock {
                    label: "10%".into(),
                    items: vec![item("target", 0.618, 0.604)],
                    ..RunBlock::default()
                },
            ],
            note: String::new(),
            finished: true,
        };
        let tsv = report.to_tsv();
        let lines: Vec<&str> = tsv.lines().collect();
        assert!(lines[0].starts_with("phase\tmetric\t"), "a header first");
        assert!(
            lines[1].contains("0%\tMSD\t356557.3\t23902.1\t1800"),
            "the numbers, not the sentence: {}",
            lines[1]
        );
        assert!(lines[1].contains("0.960"), "the anchor's Dice is a column");
        assert!(
            lines[2].contains("10%\t\t"),
            "a phase with no run of its own leaves those columns empty: {}",
            lines[2]
        );
        // The volumes are their own table, one row per structure per phase.
        let vols = lines
            .iter()
            .position(|l| l.starts_with("phase\tstructure"))
            .expect("a second table");
        assert_eq!(
            lines.len() - vols,
            3,
            "a header and two rows: {:?}",
            &lines[vols..]
        );
        assert!(lines[vols + 1].contains("target\t0.618\t0.629\t0.629\t+1.78"));
    }

    #[test]
    fn the_filed_column_is_left_out_when_nothing_was_done_to_the_mask() {
        let block = RunBlock {
            items: vec![item("target", 0.618, 0.629)],
            ..RunBlock::default()
        };
        // Neither closing nor filling: the filed volume is the deformed one,
        // and a column that repeats its neighbour is noise.
        let plain = RunReport {
            blocks: vec![block.clone()],
            note: String::new(),
            finished: false,
        };
        let tsv = plain.to_tsv();
        assert!(tsv.contains("deformed_cm3"), "the deformed volume stays");
        assert!(!tsv.contains("filed_cm3"), "the filed one goes: {tsv}");
        assert!(
            tsv.lines().last().unwrap().ends_with("+1.78"),
            "and the change is still the last column: {tsv}"
        );

        // With a finish, both columns are there and can differ.
        let finished = RunReport {
            blocks: vec![block],
            note: String::new(),
            finished: true,
        };
        assert!(finished.to_tsv().contains("filed_cm3"));
    }

    #[test]
    fn the_word_is_deformed_not_mapped() {
        let report = RunReport {
            blocks: vec![RunBlock {
                items: vec![item("target", 1.0, 1.0)],
                ..RunBlock::default()
            }],
            note: String::new(),
            finished: false,
        };
        let tsv = report.to_tsv();
        assert!(!tsv.contains("mapped"), "{tsv}");
        assert!(tsv.contains("deformed_cm3"), "{tsv}");
    }

    #[test]
    fn a_facts_table_drops_the_rows_that_do_not_apply() {
        // The filter is the contract: a caller hands over every row it could
        // show and the ones with nothing in them fall away, rather than each
        // caller writing the same `if !x.is_empty()`.
        let rows = [
            ("Method", "elastix rigid".to_string()),
            ("Region", String::new()),
            ("Fixed", "A · CT".to_string()),
        ];
        let shown: Vec<&str> = rows
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(n, _)| *n)
            .collect();
        assert_eq!(shown, vec!["Method", "Fixed"]);
    }

    #[test]
    fn a_long_structure_name_is_cut_rather_than_widening_the_table() {
        assert_eq!(ellipsis("heart", 24), "heart");
        let long = "a_very_long_structure_name_from_an_auto_segmentation";
        let cut = ellipsis(long, 24);
        assert_eq!(cut.chars().count(), 24, "cut to the width asked for");
        assert!(cut.ends_with('…'), "and says it was cut");
    }
}
