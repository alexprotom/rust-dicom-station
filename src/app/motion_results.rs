//! The 4D motion results window: tables, charts, CSV export, and the
//! side-by-side comparison of two runs (e.g. upright vs. supine, or
//! workspace A vs. B).
//!
//! The charts are drawn with the egui painter directly - a displacement-
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

/// Ticks of a value axis that has to cover `lo..=hi`: a step of 1, 2 or 5
/// times a power of ten giving about `target` intervals, the axis widened
/// out to whole steps, and the number of decimals that keeps every label
/// distinct (a 0.5 mm step is labelled `0.5`, `1.0`, `1.5`, never `0`, `1`,
/// `2`). An axis narrower than `min_span` is widened to it, so a trace that
/// hardly moves is not blown up to fill the chart.
///
/// Returns `(axis_lo, axis_hi, step, decimals)`.
fn nice_ticks(lo: f64, hi: f64, target: usize, min_span: f64) -> (f64, f64, f64, usize) {
    let (mut lo, mut hi) = if lo.is_finite() && hi.is_finite() && lo <= hi {
        (lo, hi)
    } else {
        (0.0, 0.0)
    };
    if hi - lo < min_span {
        // Grow upwards from a floor of zero, which is where a magnitude
        // starts; a trace below zero grows around its middle.
        if lo >= 0.0 {
            hi = lo + min_span;
        } else {
            let mid = 0.5 * (lo + hi);
            lo = mid - 0.5 * min_span;
            hi = mid + 0.5 * min_span;
        }
    }
    let target = target.max(1);
    let raw = (hi - lo) / target as f64;
    let mag = 10f64.powf(raw.log10().floor());
    // The axis a step makes: whole steps around the data. A small
    // tolerance, so a value sitting on a tick does not add a step.
    let axis = |step: f64| {
        let a = (lo / step + 1e-9).floor() * step;
        let b = ((hi / step - 1e-9).ceil() * step).max(a + step);
        (a, b)
    };
    // Of 1, 2, 5 and 10 times the magnitude, the step whose interval count
    // is nearest the target; on a tie, the one that wastes less room.
    let step = [1.0, 2.0, 5.0, 10.0]
        .map(|m| m * mag)
        .into_iter()
        .min_by(|&s, &t| {
            let cost = |s: f64| {
                let (a, b) = axis(s);
                (((b - a) / s).round() - target as f64).abs()
            };
            cost(s).total_cmp(&cost(t)).then_with(|| {
                let (a, b) = axis(s);
                let (c, d) = axis(t);
                (b - a).total_cmp(&(d - c))
            })
        })
        .unwrap_or(mag);
    let (axis_lo, axis_hi) = axis(step);
    let decimals = (-(step.log10() + 1e-9).floor()).max(0.0) as usize;
    (axis_lo, axis_hi, step, decimals)
}

/// Displacement magnitude (or drift) per phase, one polyline per track.
///
/// The chart fills the width it is given and scales its height with it
/// (within what the visible part of the window can show), and every margin
/// is measured from the labels that go in it, so the axis labels and the
/// last phase are never cut off. Hovering shows the values of the phase
/// under the pointer.
fn line_chart(ui: &mut egui::Ui, phases: &[String], series: &[Series], y_label: &str) {
    if series.is_empty() || phases.is_empty() {
        return;
    }
    let width = ui.available_width().max(240.0);
    // Tall enough to read, never taller than most of the visible area.
    let visible = ui.clip_rect().height().max(200.0);
    let h = (width * 0.42).min(visible * 0.6).clamp(150.0, 380.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(width, h), Sense::hover());
    let painter = ui.painter_at(rect);
    let axis_color = ui.visuals().weak_text_color();
    let text_color = ui.visuals().text_color();
    let font = FontId::proportional(11.0);
    let text_w = |t: &str| {
        painter
            .layout_no_wrap(t.to_string(), font.clone(), axis_color)
            .size()
            .x
    };
    let line_h = painter
        .layout_no_wrap("0%".into(), font.clone(), axis_color)
        .size()
        .y;

    // The value axis.
    let values = series.iter().flat_map(|s| s.values.iter().copied());
    let (lo, hi) = values
        .filter(|v| v.is_finite())
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
            (a.min(v), b.max(v))
        });
    let (lo, hi) = if lo.is_finite() {
        (lo.min(0.0), hi)
    } else {
        (0.0, 0.0)
    };
    let plot_h_guess = h - 2.0 * line_h - 12.0;
    let target_ticks = ((plot_h_guess / 34.0) as usize).clamp(2, 8);
    let (y_lo, y_hi, step, decimals) = nice_ticks(lo, hi, target_ticks, 1.0);
    let n_ticks = ((y_hi - y_lo) / step).round() as i64;
    let tick_value = |k: i64| {
        // From whole steps, so 0.1 + 0.2 never prints as 0.30000000000000004;
        // adding 0.0 turns a -0.0 into 0.0.
        ((y_lo / step).round() + k as f64) * step + 0.0
    };
    let tick_labels: Vec<(f64, String)> = (0..=n_ticks)
        .map(|k| {
            let v = tick_value(k);
            (v, format!("{v:.decimals$}"))
        })
        .collect();

    // Margins from the labels that live in them.
    let label_w = tick_labels
        .iter()
        .map(|(_, t)| text_w(t))
        .fold(0.0f32, f32::max);
    let left = rect.left() + label_w + 8.0;
    let top = rect.top() + line_h + 6.0;
    let bottom = rect.bottom() - line_h - 6.0;
    let last_w = phases.last().map(|p| text_w(p)).unwrap_or(0.0);
    let right = rect.right() - (0.5 * last_w + 4.0).max(8.0);
    let x_of = |i: usize| {
        left + (right - left)
            * if phases.len() > 1 {
                i as f32 / (phases.len() - 1) as f32
            } else {
                0.5
            }
    };
    let y_of = |v: f64| bottom - (bottom - top) * ((v - y_lo) / (y_hi - y_lo)) as f32;

    // Axes, y ticks and gridlines.
    painter.line_segment(
        [Pos2::new(left, top), Pos2::new(left, bottom)],
        Stroke::new(1.0, axis_color),
    );
    painter.line_segment(
        [
            Pos2::new(left, y_of(0.0f64.clamp(y_lo, y_hi))),
            Pos2::new(right, y_of(0.0f64.clamp(y_lo, y_hi))),
        ],
        Stroke::new(1.0, axis_color),
    );
    for (v, label) in &tick_labels {
        let y = y_of(*v);
        if v.abs() > 0.5 * step {
            painter.line_segment(
                [Pos2::new(left, y), Pos2::new(right, y)],
                Stroke::new(0.5, axis_color.linear_multiply(0.3)),
            );
        }
        painter.text(
            Pos2::new(left - 4.0, y),
            Align2::RIGHT_CENTER,
            label,
            font.clone(),
            axis_color,
        );
    }
    // The axis title above the plot, right of the value labels, so it never
    // meets the top one.
    painter.text(
        Pos2::new(left + 4.0, rect.top()),
        Align2::LEFT_TOP,
        y_label,
        font.clone(),
        axis_color,
    );
    // Phase labels, thinned to every n-th when they would collide.
    let widest = phases.iter().map(|p| text_w(p)).fold(0.0f32, f32::max);
    let gap = if phases.len() > 1 {
        (right - left) / (phases.len() - 1) as f32
    } else {
        f32::INFINITY
    };
    let every = ((widest + 6.0) / gap).ceil().max(1.0) as usize;
    for (i, ph) in phases.iter().enumerate() {
        let last = i == phases.len() - 1;
        // The last phase is always labelled; the one before it gives way
        // when the two would overlap.
        if !last && (i % every != 0 || (every > 1 && phases.len() - 1 - i < every)) {
            continue;
        }
        painter.text(
            Pos2::new(x_of(i), bottom + 3.0),
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
    // The phase under the pointer: a guide line and its values.
    if let Some(pos) = resp.hover_pos() {
        let i = if phases.len() > 1 {
            (((pos.x - left) / gap).round().max(0.0) as usize).min(phases.len() - 1)
        } else {
            0
        };
        painter.line_segment(
            [Pos2::new(x_of(i), top), Pos2::new(x_of(i), bottom)],
            Stroke::new(1.0, text_color.linear_multiply(0.4)),
        );
        let d = decimals.max(2);
        resp.on_hover_ui_at_pointer(|ui| {
            ui.strong(format!("Phase {} - {y_label}", phases[i]));
            for s in series {
                if let Some(v) = s.values.get(i) {
                    ui.colored_label(s.color, format!("{v:.d$}  {}", s.label));
                }
            }
        });
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
            // No outer scroll area: the window lays itself out to its own
            // size - the run pickers wrap, the export buttons stay at the
            // bottom, and only the report scrolls, vertically - so nothing
            // is ever wider than the window and the chart always fits it.
            detach::tool_window(
                ctx,
                "motion_results",
                MOTION.titled("results", self.motion_slot.min(MAX_WORKSPACES - 1)),
                &mut open,
                detach::WinOpts::size(760.0, 780.0).no_scroll(),
                |ui| {
                    ui.horizontal_wrapped(|ui| {
                        let combo_w = ((ui.available_width() - 190.0) / 2.0).clamp(140.0, 320.0);
                        ui.label("Run:");
                        egui::ComboBox::from_id_salt("motion_run")
                            .width(combo_w)
                            .truncate()
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
                            .width(combo_w)
                            .truncate()
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
                    egui::Panel::bottom(egui::Id::new("motion_results_export"))
                        .show_separator_line(true)
                        .show(ui, |ui| {
                            ui.horizontal_wrapped(|ui| {
                                if tip_button(
                                    ui,
                                    "💾 Export CSV",
                                    "The selected run as a CSV: phases as columns, one \
                                     row per method under every quantity",
                                ) {
                                    export = Some(sel);
                                }
                                if let Some(ci) = cmp {
                                    if tip_button(
                                        ui,
                                        "💾 Export comparison CSV",
                                        "Both runs, one after the other, and the matched \
                                         peak-to-peak amplitudes and ITVs side by side",
                                    ) {
                                        export = Some(usize::MAX - ci);
                                    }
                                }
                            });
                        });
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            let r = &reports[sel];
                            Self::report_body(ui, r, sel);
                            if let Some(ci) = cmp {
                                ui.separator();
                                ui.strong(format!("Comparison - {}", reports[ci].run_name));
                                Self::report_body(ui, &reports[ci], ci);
                                ui.separator();
                                Self::comparison_body(ui, r, &reports[ci], (sel, ci));
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
        if let Some(t) = preferred_reference(r) {
            series.push(Series {
                label: format!("{} (reference, {})", t.target, t.model.label()),
                color: REF_COLOR,
                values: t.magnitudes(),
                dashed: true,
            });
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
        if let Some(t) = preferred_reference(r) {
            bars.push((
                format!("{} (reference, {})", t.target, t.model.label()),
                t.peak_to_peak(),
                REF_COLOR,
            ));
        }
        bar_rows(ui, &bars, "mm");
        if !r.reference_tracks.is_empty() {
            ui.add_space(6.0);
            ui.strong("Peak-to-peak target-reference drift");
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
                // As wide as the tracks make it; it scrolls sideways on its
                // own rather than widening the window's contents.
                egui::ScrollArea::horizontal()
                    .id_salt(("motion_grid_scroll", idx))
                    .show(ui, |ui| {
                        egui::Grid::new(("motion_grid", idx))
                            .striped(true)
                            .show(ui, |ui| {
                                ui.strong("Phase");
                                for t in r.tracks.iter().chain(&r.reference_tracks) {
                                    ui.strong(format!(
                                        "{} ({})\n|d| mm · cm³",
                                        t.target,
                                        t.model.label()
                                    ));
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
            });

        // Correlations.
        if !r.correlations.is_empty() {
            egui::CollapsingHeader::new("Target-reference synchrony (Pearson)")
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
                    // Dice first on every row: it is the one number that
                    // says whether the phase landed on the reference.
                    for q in &r.qa {
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            ui.label(format!("{} ({}):", q.phase, q.model.label()));
                            match q.image_dice {
                                Some((after, before)) => {
                                    ui.monospace(
                                        egui::RichText::new(format!("Dice {after:.3}"))
                                            .color(theme::dice_color(ui.visuals(), after))
                                            .strong(),
                                    )
                                    .on_hover_text(
                                        "Overlap of the tissue of the two images after this \
                                         registration, against what it was before it",
                                    );
                                    ui.weak(format!("was {before:.3}"));
                                }
                                None => {
                                    ui.weak("Dice not measured");
                                }
                            }
                        });
                        // The engine's own line goes underneath: it is long,
                        // and the Dice is what the eye should land on first.
                        ui.horizontal_wrapped(|ui| {
                            ui.add_space(16.0);
                            ui.weak(format!(
                                "{} · p95 {:.1} mm · folding {:.2} %",
                                q.metric_line, q.disp_p95_mm, q.folding_pct
                            ));
                        });
                        // Where the phase carries its own contour, the
                        // propagation can be checked against it.
                        for (name, d) in &q.struct_dice {
                            ui.horizontal(|ui| {
                                ui.add_space(16.0);
                                ui.weak(format!("{name} vs contoured"));
                                ui.monospace(
                                    egui::RichText::new(format!("{d:.3}"))
                                        .color(theme::dice_color(ui.visuals(), *d)),
                                );
                            })
                            .response
                            .on_hover_text(
                                "Dice of the structure this model put on the phase \
                                 against the contour drawn on that phase",
                            );
                        }
                    }
                });
        }

        // ITVs.
        if !r.itvs.is_empty() {
            ui.add_space(6.0);
            ui.strong("ITV volumes");
            for itv in &r.itvs {
                ui.label(format!("    {} - {:.2} cm³", itv.seg_name, itv.volume_cm3));
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
    /// dialog. See [`MotionReport::csv`] for the layout.
    fn export_motion_csv(&mut self, sel: usize, also: Option<usize>) {
        let Some(r) = self.motion_reports.get(sel) else {
            return;
        };
        let mut csv = r.csv();
        if let Some(other) = also.and_then(|i| self.motion_reports.get(i)) {
            csv.push('\n');
            csv.push_str(&other.csv());
            csv.push('\n');
            csv.push_str(&motion::comparison_csv(r, other));
        }
        let bytes = motion::csv_file_bytes(&csv);
        let name = format!(
            "motion_{}.csv",
            r.run_name
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .collect::<String>()
        );
        self.ask_save(
            "",
            name,
            None,
            Some(CSV_FILES),
            move |app, path| match std::fs::write(&path, bytes) {
                Ok(()) => app.notice = Some(format!("✔ report written to {}", path.display())),
                Err(e) => app.error = Some(format!("CSV export: {e}")),
            },
        );
    }
}

impl MotionReport {
    /// `workspace A` - the comparison table's column header.
    fn slot_label(&self) -> String {
        format!("workspace {}", self.slot_name)
    }
}

/// The reference structure's track worth drawing beside the targets: as
/// contoured when every phase has it (measured, not modelled), else the
/// deformable one, else whatever there is.
fn preferred_reference(r: &crate::motion::MotionReport) -> Option<&crate::motion::Track> {
    [
        MotionModel::Contoured,
        MotionModel::Deformable,
        MotionModel::Rigid,
    ]
    .iter()
    .find_map(|m| r.reference_track(*m))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every piece of text the chart draws, with the area it is clipped
    /// to, from a window `width` points wide.
    fn chart_texts(width: f32) -> Vec<(String, Rect, Rect)> {
        let phases: Vec<String> = (0..9).map(|i| format!("{}%", i * 10)).collect();
        let series = vec![Series {
            label: "GTV (as contoured)".into(),
            color: Color32::RED,
            values: vec![0.0, 0.28, 0.64, 1.02, 1.23, 0.66, 0.76, 1.88, 1.45],
            dashed: false,
        }];
        let ctx = egui::Context::default();
        let input = || egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(width, 700.0))),
            ..Default::default()
        };
        // The first pass loads the fonts.
        ctx.run_ui(input(), |_| {}).textures_delta.clear();
        let mut out = ctx.run_ui(input(), |ui| {
            line_chart(ui, &phases, &series, "|d| mm");
        });
        out.textures_delta.clear();
        out.shapes
            .iter()
            .filter_map(|c| match &c.shape {
                egui::Shape::Text(t) => Some((
                    t.galley.text().to_string(),
                    t.visual_bounding_rect(),
                    c.clip_rect,
                )),
                _ => None,
            })
            .collect()
    }

    /// The labels on the left and the last phase on the right used to be
    /// cut off, and a 2 mm axis printed 0, 0, 1, 2, 2.
    #[test]
    fn the_whole_chart_fits_the_window_and_its_labels_are_distinct() {
        for width in [320.0f32, 560.0, 1000.0] {
            let texts = chart_texts(width);
            let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(width, 700.0));
            for (text, bounds, clip) in &texts {
                assert!(
                    clip.expand(0.5).contains_rect(*bounds),
                    "{width}: '{text}' at {bounds:?} is cut by {clip:?}"
                );
                assert!(
                    screen.expand(0.5).contains_rect(*bounds),
                    "{width}: '{text}' at {bounds:?} is outside the window"
                );
            }
            // No two labels on top of each other.
            for (i, (a, ra, _)) in texts.iter().enumerate() {
                for (b, rb, _) in &texts[i + 1..] {
                    assert!(
                        !ra.shrink(0.5).intersects(rb.shrink(0.5)),
                        "{width}: '{a}' {ra:?} overlaps '{b}' {rb:?}"
                    );
                }
            }
            let has = |s: &str| texts.iter().any(|(t, _, _)| t == s);
            assert!(
                has("0%") && has("80%"),
                "{width}: first and last phase labelled"
            );
            assert!(has("|d| mm"), "{width}: the axis says what it shows");
            let ticks: Vec<&str> = texts
                .iter()
                .map(|(t, _, _)| t.as_str())
                .filter(|t| t.parse::<f64>().is_ok())
                .collect();
            let mut distinct = ticks.clone();
            distinct.sort();
            distinct.dedup();
            assert_eq!(distinct.len(), ticks.len(), "{width}: {ticks:?}");
            assert!(ticks.len() >= 3, "{width}: {ticks:?}");
        }
    }

    /// The labels the chart would print for an axis over `lo..=hi`.
    fn labels(lo: f64, hi: f64, target: usize) -> Vec<String> {
        let (a, b, step, decimals) = nice_ticks(lo, hi, target, 1.0);
        let n = ((b - a) / step).round() as i64;
        (0..=n)
            .map(|k| {
                format!(
                    "{:.decimals$}",
                    ((a / step).round() + k as f64) * step + 0.0
                )
            })
            .collect()
    }

    #[test]
    fn axis_labels_are_distinct_and_cover_the_data() {
        // The case that printed 0, 0, 1, 2, 2: a 2 mm trace on four ticks.
        assert_eq!(labels(0.0, 1.98, 4), ["0.0", "0.5", "1.0", "1.5", "2.0"]);
        assert_eq!(labels(0.0, 1.126, 4), ["0.0", "0.5", "1.0", "1.5"]);
        assert_eq!(labels(0.0, 7.3, 5), ["0", "2", "4", "6", "8"]);
        assert_eq!(labels(0.0, 23.0, 4), ["0", "5", "10", "15", "20", "25"]);
        // A trace that hardly moves keeps a 1 mm axis instead of being
        // blown up into a big motion.
        assert_eq!(
            labels(0.0, 0.01, 4),
            ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
        );
        for (lo, hi, t) in [
            (0.0, 0.3, 4),
            (0.0, 1.0, 4),
            (0.0, 2.0, 4),
            (0.0, 3.3, 6),
            (-2.5, 4.0, 5),
            (0.0, 0.0, 4),
            (0.0, 137.0, 8),
        ] {
            let l = labels(lo, hi, t);
            let mut d = l.clone();
            d.dedup();
            assert_eq!(d, l, "duplicate labels for {lo}..{hi}");
            let (a, b, _, _) = nice_ticks(lo, hi, t, 1.0);
            assert!(a <= lo && b >= hi, "{a}..{b} does not cover {lo}..{hi}");
            assert!(l.len() >= 2 && l.len() <= 2 * t + 2, "{l:?}");
            assert!(!l
                .iter()
                .any(|s| s.starts_with("-0") && s.trim_start_matches(['-', '0', '.']).is_empty()));
        }
    }
}
