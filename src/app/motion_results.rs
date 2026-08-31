//! The 4D motion results window: tables, charts, CSV export, and the
//! side-by-side comparison of two runs (e.g. upright vs. supine, or
//! dataset A vs. B).
//!
//! The charts are drawn with the egui painter directly — a displacement-
//! vs-phase line chart and grouped bar charts are simple enough that a
//! plotting dependency would cost more than it gives.

use crate::motion::{self, MotionModel, MotionReport};

use super::*;

/// A color per track that stays stable across the charts and tables.
fn track_color(i: usize) -> Color32 {
    const C: [Color32; 8] = [
        Color32::from_rgb(0x4c, 0x8b, 0xf5), // blue
        Color32::from_rgb(0x38, 0xa1, 0x69), // green
        Color32::from_rgb(0xe2, 0x74, 0x3c), // orange
        Color32::from_rgb(0xb1, 0x5b, 0xd6), // purple
        Color32::from_rgb(0x2f, 0xa8, 0xa8), // teal
        Color32::from_rgb(0xd6, 0x5b, 0x7a), // rose
        Color32::from_rgb(0x8f, 0x9a, 0x2f), // olive
        Color32::from_rgb(0x80, 0x80, 0x80), // gray
    ];
    C[i % C.len()]
}

/// The reference structure's curve gets the manuscript's dashed red.
const REF_COLOR: Color32 = Color32::from_rgb(0xd6, 0x45, 0x45);

/// One line of a line chart: label, color, y per phase.
struct Series {
    label: String,
    color: Color32,
    values: Vec<f64>,
    dashed: bool,
}

/// Displacement magnitude (or drift) per phase, one polyline per track.
fn line_chart(ui: &mut egui::Ui, phases: &[String], series: &[Series], y_label: &str) {
    if series.is_empty() {
        return;
    }
    let h = 160.0f32;
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width().max(260.0), h),
        Sense::hover(),
    );
    let painter = ui.painter_at(rect);
    let axis_color = ui.visuals().weak_text_color();
    let font = FontId::proportional(10.0);

    let max_y = series
        .iter()
        .flat_map(|s| s.values.iter().copied())
        .fold(1.0f64, f64::max)
        .ceil();
    let left = rect.left() + 34.0;
    let bottom = rect.bottom() - 16.0;
    let top = rect.top() + 6.0;
    let right = rect.right() - 6.0;
    let x_of = |i: usize| {
        left + (right - left)
            * if phases.len() > 1 {
                i as f32 / (phases.len() - 1) as f32
            } else {
                0.5
            }
    };
    let y_of = |v: f64| bottom - (bottom - top) * (v / max_y) as f32;

    // Axes, y ticks and gridlines.
    painter.line_segment(
        [Pos2::new(left, top), Pos2::new(left, bottom)],
        Stroke::new(1.0, axis_color),
    );
    painter.line_segment(
        [Pos2::new(left, bottom), Pos2::new(right, bottom)],
        Stroke::new(1.0, axis_color),
    );
    let ticks = 4;
    for t in 0..=ticks {
        let v = max_y * t as f64 / ticks as f64;
        let y = y_of(v);
        if t > 0 {
            painter.line_segment(
                [Pos2::new(left, y), Pos2::new(right, y)],
                Stroke::new(0.5, axis_color.linear_multiply(0.3)),
            );
        }
        painter.text(
            Pos2::new(left - 4.0, y),
            Align2::RIGHT_CENTER,
            format!("{v:.0}"),
            font.clone(),
            axis_color,
        );
    }
    painter.text(
        Pos2::new(left, top - 2.0),
        Align2::LEFT_BOTTOM,
        y_label,
        font.clone(),
        axis_color,
    );
    // Phase labels, thinned when they would collide.
    let step = (phases.len() / 10).max(1);
    for (i, ph) in phases.iter().enumerate() {
        if i % step != 0 && i != phases.len() - 1 {
            continue;
        }
        painter.text(
            Pos2::new(x_of(i), bottom + 2.0),
            Align2::CENTER_TOP,
            ph,
            font.clone(),
            axis_color,
        );
    }
    // The polylines.
    for s in series {
        for w in s.values.windows(2).enumerate() {
            let (i, pair) = w;
            if s.dashed && i % 2 == 1 {
                continue;
            }
            painter.line_segment(
                [
                    Pos2::new(x_of(i), y_of(pair[0])),
                    Pos2::new(x_of(i + 1), y_of(pair[1])),
                ],
                Stroke::new(1.6, s.color),
            );
        }
        for (i, &v) in s.values.iter().enumerate() {
            painter.circle_filled(Pos2::new(x_of(i), y_of(v)), 2.2, s.color);
        }
    }
    // Legend.
    ui.horizontal_wrapped(|ui| {
        for s in series {
            ui.colored_label(s.color, format!("■ {}", s.label));
        }
    });
}

/// Grouped horizontal bars: one row per entry, value + label.
fn bar_rows(ui: &mut egui::Ui, entries: &[(String, f64, Color32)], unit: &str) {
    let max = entries.iter().map(|e| e.1).fold(1e-9f64, f64::max);
    for (label, v, color) in entries {
        ui.horizontal(|ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(120.0, 12.0), Sense::hover());
            let w = rect.width() * (*v / max) as f32;
            ui.painter_at(rect).rect_filled(
                Rect::from_min_size(rect.min, Vec2::new(w.max(1.0), rect.height())),
                2.0,
                *color,
            );
            ui.label(format!("{v:.2} {unit}  {label}"));
        });
    }
}

impl ViewerApp {
    pub(super) fn motion_results_window(&mut self, ctx: &egui::Context) {
        if !self.motion_results_open {
            return;
        }
        if self.motion_reports.is_empty() {
            self.motion_results_open = false;
            return;
        }
        let mut open = true;
        let mut export: Option<usize> = None;
        self.motion_sel = self.motion_sel.min(self.motion_reports.len() - 1);
        if let Some(c) = self.motion_cmp {
            if c >= self.motion_reports.len() || c == self.motion_sel {
                self.motion_cmp = None;
            }
        }
        let mut sel = self.motion_sel;
        let mut cmp = self.motion_cmp;
        {
            let reports = &self.motion_reports;
            detach::tool_window(
                ctx,
                "motion_results",
                MOTION.titled("results", self.motion_slot.min(1)),
                &mut open,
                detach::WinOpts::width(560.0),
                |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Run:");
                        egui::ComboBox::from_id_salt("motion_run")
                            .width(280.0)
                            .selected_text(reports[sel].run_name.clone())
                            .show_ui(ui, |ui| {
                                for (i, r) in reports.iter().enumerate() {
                                    ui.selectable_value(&mut sel, i, &r.run_name);
                                }
                            });
                        ui.label("Compare with:");
                        let cmp_text = cmp
                            .map(|i| reports[i].run_name.clone())
                            .unwrap_or_else(|| "(none)".into());
                        egui::ComboBox::from_id_salt("motion_cmp")
                            .width(220.0)
                            .selected_text(cmp_text)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut cmp, None, "(none)");
                                for (i, r) in reports.iter().enumerate() {
                                    if i != sel {
                                        ui.selectable_value(&mut cmp, Some(i), &r.run_name);
                                    }
                                }
                            });
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .max_height(480.0)
                        .show(ui, |ui| {
                            let r = &reports[sel];
                            Self::report_body(ui, r, sel);
                            if let Some(ci) = cmp {
                                ui.separator();
                                ui.strong(format!("Comparison — {}", reports[ci].run_name));
                                Self::report_body(ui, &reports[ci], ci);
                                ui.separator();
                                Self::comparison_body(ui, r, &reports[ci], (sel, ci));
                            }
                        });
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui
                            .button("💾 Export CSV")
                            .on_hover_text("The selected run as one long-format CSV file")
                            .clicked()
                        {
                            export = Some(sel);
                        }
                        if let Some(ci) = cmp {
                            if ui.button("💾 Export comparison CSV").clicked() {
                                export = Some(usize::MAX - ci);
                            }
                        }
                    });
                },
            );
        }
        self.motion_sel = sel;
        self.motion_cmp = cmp;
        if let Some(code) = export {
            let (i, also) = if code > usize::MAX / 2 {
                (self.motion_sel, Some(usize::MAX - code))
            } else {
                (code, None)
            };
            self.export_motion_csv(i, also);
        }
        if !open {
            self.motion_results_open = false;
        }
    }

    /// Tables and charts of one run. `idx` salts the widget ids, because two
    /// runs are on screen at once in the A-vs-B comparison.
    fn report_body(ui: &mut egui::Ui, r: &MotionReport, idx: usize) {
        ui.strong(&r.run_name);
        ui.weak(format!(
            "{} · reference phase {} · {} phase(s){}",
            r.patient,
            r.reference,
            r.phases.len(),
            r.reference_structure
                .as_deref()
                .map(|s| format!(" · reference structure: {s}"))
                .unwrap_or_default()
        ));

        // Displacement magnitude vs phase.
        let mut series: Vec<Series> = Vec::new();
        for (i, t) in r.tracks.iter().enumerate() {
            series.push(Series {
                label: format!("{} ({})", t.target, t.model.label()),
                color: track_color(i),
                values: t.magnitudes(),
                dashed: false,
            });
        }
        for t in &r.reference_tracks {
            if t.model == MotionModel::Deformable || r.reference_tracks.len() == 1 {
                series.push(Series {
                    label: format!("{} (reference)", t.target),
                    color: REF_COLOR,
                    values: t.magnitudes(),
                    dashed: true,
                });
                break;
            }
        }
        line_chart(ui, &r.phases, &series, "|d| mm");

        // Peak-to-peak amplitudes and drift.
        ui.add_space(6.0);
        ui.strong("Peak-to-peak amplitude");
        let mut bars: Vec<(String, f64, Color32)> = Vec::new();
        for (i, t) in r.tracks.iter().enumerate() {
            bars.push((
                format!("{} ({})", t.target, t.model.label()),
                t.peak_to_peak(),
                track_color(i),
            ));
        }
        for t in &r.reference_tracks {
            if t.model == MotionModel::Deformable || r.reference_tracks.len() == 1 {
                bars.push((
                    format!("{} (reference)", t.target),
                    t.peak_to_peak(),
                    REF_COLOR,
                ));
                break;
            }
        }
        bar_rows(ui, &bars, "mm");
        if !r.reference_tracks.is_empty() {
            ui.add_space(6.0);
            ui.strong("Peak-to-peak target–reference drift");
            let mut bars: Vec<(String, f64, Color32)> = Vec::new();
            for (i, t) in r.tracks.iter().enumerate() {
                if let Some(rt) = r.reference_track(t.model) {
                    if let Some(drift) = t.drift_against(rt) {
                        bars.push((
                            format!("{} ({})", t.target, t.model.label()),
                            motion::peak_to_peak(&drift),
                            track_color(i),
                        ));
                    }
                }
            }
            bar_rows(ui, &bars, "mm");
        }

        // The per-phase numbers.
        egui::CollapsingHeader::new("Per-phase table")
            .id_salt(("motion_table", idx))
            .show(ui, |ui| {
                egui::Grid::new(("motion_grid", idx))
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Phase");
                        for t in r.tracks.iter().chain(&r.reference_tracks) {
                            ui.strong(format!("{} ({})\n|d| mm · cm³", t.target, t.model.label()));
                        }
                        ui.end_row();
                        for (pi, ph) in r.phases.iter().enumerate() {
                            ui.label(ph);
                            for t in r.tracks.iter().chain(&r.reference_tracks) {
                                let d = t.magnitudes()[pi];
                                let v = t.samples[pi].volume_cm3;
                                ui.label(format!("{d:.2} · {v:.2}"));
                            }
                            ui.end_row();
                        }
                    });
            });

        // Correlations.
        if !r.correlations.is_empty() {
            egui::CollapsingHeader::new("Target–reference synchrony (Pearson)")
                .id_salt(("motion_corr", idx))
                .default_open(true)
                .show(ui, |ui| {
                    for (target, model, axes) in &r.correlations {
                        ui.label(format!("{target} ({}):", model.label()));
                        for c in axes {
                            ui.weak(format!("    {}", c.line()));
                        }
                    }
                });
        }

        // Registration quality.
        if !r.qa.is_empty() {
            egui::CollapsingHeader::new("Registration quality")
                .id_salt(("motion_qa", idx))
                .show(ui, |ui| {
                    for q in &r.qa {
                        ui.weak(format!(
                            "{} ({}): {} · p95 {:.1} mm · folding {:.2} %",
                            q.phase,
                            q.model.label(),
                            q.metric_line,
                            q.disp_p95_mm,
                            q.folding_pct
                        ));
                    }
                });
        }

        // ITVs.
        if !r.itvs.is_empty() {
            ui.add_space(6.0);
            ui.strong("ITV volumes");
            for itv in &r.itvs {
                ui.label(format!("    {} — {:.2} cm³", itv.seg_name, itv.volume_cm3));
            }
        }
    }

    /// The A-vs-B section: matched ITVs with the volume change, and matched
    /// peak-to-peak amplitudes.
    fn comparison_body(ui: &mut egui::Ui, a: &MotionReport, b: &MotionReport, idx: (usize, usize)) {
        ui.strong(format!("{}  vs  {}", a.run_name, b.run_name));
        let mut any = false;
        egui::Grid::new(("motion_cmp_grid", idx))
            .striped(true)
            .show(ui, |ui| {
                ui.strong("ITV");
                ui.strong(a.slot_label());
                ui.strong(b.slot_label());
                ui.strong("change");
                ui.end_row();
                for ia in &a.itvs {
                    let Some(ib) = b
                        .itvs
                        .iter()
                        .find(|x| x.target == ia.target && x.model == ia.model)
                    else {
                        continue;
                    };
                    any = true;
                    let change = if ib.volume_cm3 > 1e-9 {
                        100.0 * (ia.volume_cm3 - ib.volume_cm3) / ib.volume_cm3
                    } else {
                        0.0
                    };
                    ui.label(format!("{} ({})", ia.target, ia.model.label()));
                    ui.label(format!("{:.2} cm³", ia.volume_cm3));
                    ui.label(format!("{:.2} cm³", ib.volume_cm3));
                    ui.label(format!("{change:+.1} %"));
                    ui.end_row();
                }
            });
        if !any {
            ui.weak("No ITV appears in both runs under the same target name and model.");
        }
        // Peak-to-peak side by side.
        let matched: Vec<(String, f64, f64)> = a
            .tracks
            .iter()
            .filter_map(|ta| {
                b.tracks
                    .iter()
                    .find(|tb| tb.target == ta.target && tb.model == ta.model)
                    .map(|tb| {
                        (
                            format!("{} ({})", ta.target, ta.model.label()),
                            ta.peak_to_peak(),
                            tb.peak_to_peak(),
                        )
                    })
            })
            .collect();
        if !matched.is_empty() {
            ui.add_space(4.0);
            ui.strong("Peak-to-peak amplitude");
            egui::Grid::new(("motion_cmp_pp", idx))
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Track");
                    ui.strong(a.slot_label());
                    ui.strong(b.slot_label());
                    ui.end_row();
                    for (label, pa, pb) in matched {
                        ui.label(label);
                        ui.label(format!("{pa:.2} mm"));
                        ui.label(format!("{pb:.2} mm"));
                        ui.end_row();
                    }
                });
        }
    }

    /// Write one run (or a run plus its comparison) as CSV, via a save
    /// dialog.
    fn export_motion_csv(&mut self, sel: usize, also: Option<usize>) {
        let Some(r) = self.motion_reports.get(sel) else {
            return;
        };
        let mut csv = r.csv();
        if let Some(other) = also.and_then(|i| self.motion_reports.get(i)) {
            // The header line of the second report is dropped — one file,
            // one header.
            if let Some(pos) = other.csv().find('\n') {
                csv.push_str(&other.csv()[pos + 1..]);
            }
        }
        let name = format!(
            "motion_{}.csv",
            r.run_name
                .chars()
                .map(|c| if c.is_alphanumeric() { c } else { '_' })
                .collect::<String>()
        );
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name(&name)
            .add_filter("CSV", &["csv"])
            .save_file()
        {
            match std::fs::write(&path, csv) {
                Ok(()) => self.notice = Some(format!("✔ report written to {}", path.display())),
                Err(e) => self.error = Some(format!("CSV export: {e}")),
            }
        }
    }
}

impl MotionReport {
    /// `dataset A` — the comparison table's column header.
    fn slot_label(&self) -> String {
        format!("dataset {}", self.slot_name)
    }
}
