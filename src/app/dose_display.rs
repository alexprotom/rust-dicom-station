//! *Modules ▶ Dose and plan display*: how the displayed dose grid is drawn
//! over the images - colorwash, isodose lines or both, the opacity, the
//! threshold under which nothing is drawn, the isodose ladder - each
//! workspace's reference dose (the 100 % every level and the wash are
//! relative to), the dose legend on the views, and whether the shown plan's
//! beam fields are drawn ([`super::overlays`]).
//!
//! These are display settings of the viewer, not properties of a grid: one
//! set serves every workspace, which is why they left the data tree (where
//! they sat under whichever workspace happened to be the first with dose)
//! for a module of their own. *Which* grid is drawn stays with the grid
//! under *Dose* in the tree; a grid's context menu opens this section.
//!
//! The colorwash is a colour map laid over a range of dose
//! ([`crate::settings::DoseColors`]): continuous by default, or in bands of
//! one colour each (*Stepped bands*); the rainbow the viewer always drew,
//! MATLAB's jet, Google's turbo, matplotlib's viridis, or the isodose
//! levels' own colours (each level's colour at its percentage, interpolated
//! between levels, or banded at them). Its two ends are doses in % of each
//! workspace's reference or in Gy; under the threshold nothing is drawn.
//! The scale is a 1024-entry table ([`render::WashScale`]) worked out when a
//! texture is rebuilt, so a pixel is one lookup whatever the map. The
//! isodose lines stay at their levels, which are edited here: colour, on,
//! percentage, added and removed. Every choice of the scale and the ladder
//! is in `dose_settings_hash`, the key of the views' dose textures, and is
//! written to the settings once it settles.
//!
//! The legend is the one a planning system puts in the corner of its views:
//! the reference it is a percentage of, the wash as a colour bar with a tick
//! at every step of its scale (from the threshold, or the scale's lower end
//! if that is higher, up to its upper end), and beside it the isodose levels
//! drawn, each with its colour, highest first. It is drawn on the views
//! ticked here, below the pane's buttons on the right.

use super::*;
use crate::settings::{DoseColormap, DoseColors};

/// Where the dose legend is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DoseLegend {
    pub on: bool,
    /// Axial, sagittal, coronal, the 3D scene.
    pub views: [bool; 4],
}

impl Default for DoseLegend {
    fn default() -> Self {
        DoseLegend {
            on: true,
            views: [true; 4],
        }
    }
}

impl DoseLegend {
    /// Drawn on a view of `plane`, or (`None`) on the 3D scene.
    pub fn shows(&self, plane: Option<ViewPlane>) -> bool {
        self.on
            && self.views[match plane {
                Some(ViewPlane::Axial) => 0,
                Some(ViewPlane::Sagittal) => 1,
                Some(ViewPlane::Coronal) => 2,
                None => 3,
            }]
    }
}

/// What radiation a dose grid is of, from the beams of the plan it names:
/// `Photon`, `Proton`, `Electron`, ... (several joined by `/`), with `RBE`
/// for an effective dose. `None` when the plan is not loaded and the grid
/// says nothing either.
pub(super) fn dose_modality(
    plans: &[crate::rtplan::PlanInfo],
    d: &crate::rtdose::DoseGrid,
) -> Option<String> {
    let mut kinds: Vec<String> = Vec::new();
    if let Some(p) = plans
        .iter()
        .find(|p| !d.referenced_plan_uid.is_empty() && p.sop_instance_uid == d.referenced_plan_uid)
    {
        for b in &p.beams {
            let t = b.radiation_type.trim();
            if t.is_empty() {
                continue;
            }
            let mut w = t.to_lowercase();
            if let Some(c) = w.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            if !kinds.contains(&w) {
                kinds.push(w);
            }
        }
        if kinds.is_empty() && !p.plan_kind.is_empty() {
            kinds.push(p.plan_kind.clone());
        }
    }
    let rbe = d.dose_type.eq_ignore_ascii_case("EFFECTIVE");
    match (kinds.is_empty(), rbe) {
        (true, false) => None,
        (true, true) => Some("RBE".into()),
        (false, false) => Some(kinds.join("/")),
        (false, true) => Some(format!("{} (RBE)", kinds.join("/"))),
    }
}

/// The doses (absolute, the grid's units) the two ends of the wash stand
/// for in a workspace whose 100 % is `reference`.
fn wash_limits(dc: &DoseColors, reference: f32) -> (f32, f32) {
    if dc.in_gy {
        (dc.lo, dc.hi)
    } else {
        (dc.lo / 100.0 * reference, dc.hi / 100.0 * reference)
    }
}

/// The colorwash's colour scale in a workspace whose 100 % is `reference`:
/// what the views' dose textures, the 3D scene's dose-painted surfaces and
/// the legend's colour bar are all drawn with. The isodose map takes the
/// levels that are ticked.
pub(super) fn wash_scale(dc: &DoseColors, reference: f32, threshold_pct: f32) -> render::WashScale {
    let reference = reference.max(1e-6);
    let (lo, hi) = wash_limits(dc, reference);
    let levels: Vec<(f32, [u8; 3])> = dc
        .levels
        .iter()
        .filter(|l| l.on)
        .map(|l| (l.pct, l.color))
        .collect();
    render::WashScale::new(
        lo,
        hi,
        threshold_pct / 100.0 * reference,
        render::WashStyle {
            map: dc.map,
            steps: dc.steps,
            stepped: dc.stepped,
            reference,
            levels: &levels,
        },
    )
}

/// A number of the legend: as many decimals as the scale's ticks need.
fn tick_text(v: f32, decimals: usize) -> String {
    format!("{v:.decimals$}")
}

impl ViewerApp {
    pub(super) fn dose_display_section(&mut self, ui: &mut egui::Ui) {
        self.module_header(
            ui,
            Module::DoseDisplay,
            |_, _| {},
            |app, ui| app.dose_display_body(ui),
        );
    }

    fn dose_display_body(&mut self, ui: &mut egui::Ui) {
        let with_dose: Vec<usize> = self
            .open_slots()
            .into_iter()
            .filter(|s| {
                self.slots[*s]
                    .study
                    .as_ref()
                    .is_some_and(|st| !st.doses.is_empty())
            })
            .collect();
        if with_dose.is_empty() {
            // The settings are the viewer's, so they can be set before any
            // dose is loaded; the note says why nothing changes yet.
            ui.weak("No workspace holds a dose grid; these apply once one is loaded");
        }
        let mut mode = self.dose_mode;
        let mut opacity = self.dose_opacity;
        let mut threshold = self.dose_threshold_pct;
        ui.horizontal(|ui| {
            ui.label("Show");
            egui::ComboBox::from_id_salt("dose_mode")
                .selected_text(mode.label())
                .show_ui(ui, |ui| {
                    for m in [
                        DoseMode::Off,
                        DoseMode::Colorwash,
                        DoseMode::Isodose,
                        DoseMode::Both,
                    ] {
                        ui.selectable_value(&mut mode, m, m.label());
                    }
                });
        });

        // The 100 % of each workspace's drawn grid.
        let both = self.comparing();
        let mut first_reference = None;
        for &s in &with_dose {
            let st = &mut self.slots[s];
            let Some(study) = st.study.as_ref() else {
                continue;
            };
            let Some(d) = study
                .doses
                .get(st.active_dose.min(study.doses.len().saturating_sub(1)))
            else {
                continue;
            };
            let max = d.max_dose;
            let units = gy_units(&d.units);
            let reference = &mut st.dose_reference;
            ui.horizontal(|ui| {
                ui.label(if both {
                    format!("Reference {}", SLOT_NAMES[s])
                } else {
                    "Reference".to_string()
                });
                ui.add(
                    egui::DragValue::new(reference)
                        .speed(0.05)
                        .range(0.01..=1000.0)
                        .suffix(format!(" {units}")),
                )
                .tip("Dose at 100 %");
                if ui
                    .small_button("max")
                    .tip(format!("Use the maximum, {max:.2} {units}"))
                    .clicked()
                {
                    *reference = max;
                }
            });
            first_reference.get_or_insert(*reference);
        }

        ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"));
        ui.add(egui::Slider::new(&mut threshold, 0.0..=100.0).text("Threshold %"))
            .on_hover_text("No colour under this percentage of the reference");

        self.dose_scale_ui(ui, first_reference);
        self.dose_levels_ui(ui);

        // The legend over the views.
        ui.separator();
        let beams_before = self.overlays.beam_fields;
        ui.checkbox(&mut self.overlays.beam_fields, "Beam fields")
            .on_hover_text("The shown plan's beams on the views and in 3D");
        if self.overlays.beam_fields != beams_before {
            self.persist_settings();
        }
        let lg = &mut self.dose_legend;
        ui.checkbox(&mut lg.on, "Legend on the views")
            .on_hover_text("Dose scale in the corner of each view");
        ui.add_enabled_ui(lg.on, |ui| {
            ui.horizontal_wrapped(|ui| {
                for (on, name) in lg
                    .views
                    .iter_mut()
                    .zip(["Axial", "Sagittal", "Coronal", "3D"])
                {
                    ui.checkbox(on, name);
                }
            });
        });
        self.dose_mode = mode;
        self.dose_opacity = opacity;
        self.dose_threshold_pct = threshold;

        // The colour choices are written once they settle: not on every
        // frame of a drag, but as soon as no button is held.
        if self.dose_colors != self.dose_colors_saved && !ui.input(|i| i.pointer.any_down()) {
            self.dose_colors_saved = self.dose_colors.clone();
            self.persist_settings();
        }
    }

    /// The wash's colour map, its two ends, its steps. `reference` is the
    /// first workspace's 100 %, which a change of unit converts by.
    fn dose_scale_ui(&mut self, ui: &mut egui::Ui, reference: Option<f32>) {
        ui.separator();
        ui.weak("Colorwash scale");
        let dc = &mut self.dose_colors;
        ui.horizontal(|ui| {
            ui.label("Colour map");
            egui::ComboBox::from_id_salt("dose_colormap")
                .selected_text(dc.map.label())
                .show_ui(ui, |ui| {
                    for m in DoseColormap::ALL {
                        ui.selectable_value(&mut dc.map, m, m.label());
                    }
                });
        });
        ui.horizontal(|ui| {
            let unit = if dc.in_gy { " Gy" } else { " %" };
            let speed = if dc.in_gy { 0.05 } else { 0.5 };
            ui.label("From");
            ui.add(
                egui::DragValue::new(&mut dc.lo)
                    .speed(speed)
                    .range(0.0..=10000.0)
                    .suffix(unit),
            )
            .tip("The dose the map's first colour stands for");
            ui.label("to");
            ui.add(
                egui::DragValue::new(&mut dc.hi)
                    .speed(speed)
                    .range(0.0..=10000.0)
                    .suffix(unit),
            )
            .tip("The dose the map's last colour stands for; any higher keeps it");
            let mut in_gy = dc.in_gy;
            ui.selectable_value(&mut in_gy, false, "%")
                .tip("Limits in percent of each workspace's reference");
            ui.selectable_value(&mut in_gy, true, "Gy")
                .tip("Limits in Gy, the same for every workspace");
            if in_gy != dc.in_gy {
                // The same doses in the other unit, by the first
                // workspace's reference; with none loaded the numbers stay,
                // there being nothing to convert by.
                if let Some(r) = reference.filter(|r| *r > 0.0) {
                    let f = if in_gy { r / 100.0 } else { 100.0 / r };
                    dc.lo *= f;
                    dc.hi *= f;
                }
                dc.in_gy = in_gy;
            }
        });
        ui.horizontal(|ui| {
            ui.label("Steps");
            ui.add(egui::DragValue::new(&mut dc.steps).range(settings::DOSE_STEPS))
                .tip("Intervals of the legend's scale, and bands of a stepped wash");
            ui.checkbox(&mut dc.stepped, "Stepped bands").on_hover_text(
                "One colour per step instead of a continuous wash; with the isodose \
                 colours, one band per level",
            );
        });
        if ui
            .small_button("Default scale")
            .on_hover_text("Rainbow, continuous, from 0 to 100 % in 5 steps")
            .clicked()
        {
            let d = settings::DoseColors::default();
            dc.map = d.map;
            dc.in_gy = d.in_gy;
            dc.lo = d.lo;
            dc.hi = d.hi;
            dc.steps = d.steps;
            dc.stepped = d.stepped;
        }
    }

    /// The isodose ladder: each level's colour, whether it is drawn, its
    /// percentage; a level added or removed.
    fn dose_levels_ui(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.weak("Isodose levels (% of reference)");
        let levels = &mut self.dose_colors.levels;
        let mut remove = None;
        for (i, l) in levels.iter_mut().enumerate() {
            let pct = l.pct;
            ui.horizontal(|ui| {
                color_swatch(ui, &mut l.color, &format!("the {pct:.0} % level"));
                ui.checkbox(&mut l.on, "")
                    .tip("Draw this level")
                    .named_with(|| format!("Show the {pct:.0} % level"));
                ui.add(
                    egui::DragValue::new(&mut l.pct)
                        .speed(0.5)
                        .range(0.0..=1000.0)
                        .max_decimals(1)
                        .suffix(" %"),
                )
                .tip("The level's percentage of the reference");
                if ui
                    .small_button("✖")
                    .tip("Remove this level")
                    .named_with(|| format!("Remove the {pct:.0} % level"))
                    .clicked()
                {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            levels.remove(i);
        }
        ui.horizontal(|ui| {
            if ui
                .small_button("➕ Add level")
                .on_hover_text("A new level halfway under the lowest")
                .clicked()
            {
                // Halfway under the lowest, coloured as the rainbow would
                // colour it: the ladder stays sorted from the top.
                let pct = levels
                    .iter()
                    .map(|l| l.pct)
                    .reduce(f32::min)
                    .map_or(100.0, |p| (p / 2.0).round().max(1.0));
                levels.push(settings::IsoLevel {
                    pct,
                    color: render::dose_colormap(pct / 100.0),
                    on: true,
                });
            }
            if ui
                .small_button("Sort")
                .on_hover_text("Highest level first")
                .clicked()
            {
                levels.sort_by(|a, b| b.pct.total_cmp(&a.pct));
            }
            if ui
                .small_button("Default levels")
                .on_hover_text("107, 100, 95, 90, 80, 70, 50 and 30 %")
                .clicked()
            {
                *levels = settings::default_iso_levels();
            }
        });
    }

    /// The dose legend of `slot`, its top right corner at `top_right`:
    /// nothing while that workspace draws no dose. The wash is a colour bar
    /// with its scale's ticks; the isodose lines, beside it, a swatch and
    /// percentage each, highest first.
    pub(super) fn draw_dose_legend(&self, painter: &egui::Painter, slot: usize, top_right: Pos2) {
        if self.dose_mode == DoseMode::Off || !self.slots[slot].dose_shown {
            return;
        }
        let st = &self.slots[slot];
        let Some(study) = st.study.as_ref() else {
            return;
        };
        let Some(d) = study
            .doses
            .get(st.active_dose.min(study.doses.len().saturating_sub(1)))
        else {
            return;
        };
        let dc = &self.dose_colors;
        let reference = st.dose_reference.max(1e-6);
        let units = gy_units(&d.units);
        let mut title = if dc.in_gy && self.dose_mode.wash() {
            format!("{units}, 100 % = {reference:.2}")
        } else {
            format!("% of {reference:.2} {units}")
        };
        if d.dose_type.eq_ignore_ascii_case("EFFECTIVE") {
            title.push_str(" (RBE)");
        }
        let font = FontId::proportional(11.0);
        let text = Color32::from_gray(230);
        let line = 13.0;
        let text_w = |s: &str| {
            painter
                .layout_no_wrap(s.to_string(), font.clone(), text)
                .size()
                .x
        };

        // The isodose lines drawn.
        let mut levels: Vec<(f32, Color32)> = if self.dose_mode.iso() {
            dc.levels
                .iter()
                .filter(|l| l.on)
                .map(|l| (l.pct, Color32::from_rgb(l.color[0], l.color[1], l.color[2])))
                .collect()
        } else {
            Vec::new()
        };
        levels.sort_by(|a, b| b.0.total_cmp(&a.0));
        let iso_labels: Vec<String> = levels
            .iter()
            .map(|(p, _)| {
                if (p - p.round()).abs() < 0.05 {
                    format!("{p:.0}")
                } else {
                    format!("{p:.1}")
                }
            })
            .collect();
        let iso_w = iso_labels.iter().map(|s| text_w(s)).fold(0.0f32, f32::max) + 18.0;

        // The wash: its scale from what is drawn of it (the threshold, or
        // the scale's lower end if that is higher) to its upper end, with a
        // tick at every step.
        struct Bar {
            scale: render::WashScale,
            bottom: f32,
            ticks: Vec<(f32, String)>,
            label_w: f32,
        }
        let bar = self.dose_mode.wash().then(|| {
            let scale = wash_scale(dc, reference, self.dose_threshold_pct);
            let (lo, hi) = (scale.lo, scale.hi);
            let bottom = if scale.threshold < hi {
                lo.max(scale.threshold)
            } else {
                lo
            };
            // Ticks in the unit the limits are given in.
            let shown = |v: f32| if dc.in_gy { v } else { v / reference * 100.0 };
            let step = (hi - lo) / dc.steps.max(1) as f32;
            let mut values: Vec<f32> = (0..=dc.steps.max(1))
                .map(|k| lo + step * k as f32)
                .filter(|v| *v >= bottom - step * 1e-3)
                .collect();
            if values
                .first()
                .is_none_or(|v| (v - bottom).abs() > step * 0.25)
            {
                values.insert(0, bottom);
            }
            let whole = values
                .iter()
                .all(|v| (shown(*v) - shown(*v).round()).abs() < 0.05);
            let decimals = if whole { 0 } else { 1 };
            let ticks: Vec<(f32, String)> = values
                .into_iter()
                .map(|v| (v, tick_text(shown(v), decimals)))
                .collect();
            let label_w = ticks.iter().map(|(_, s)| text_w(s)).fold(0.0f32, f32::max);
            Bar {
                scale,
                bottom,
                ticks,
                label_w,
            }
        });

        let bar_h = 120.0;
        let bar_col_w = bar.as_ref().map_or(0.0, |b| b.label_w + 4.0 + 10.0);
        let iso_col_w = if levels.is_empty() { 0.0 } else { iso_w };
        let gap = if bar.is_some() && !levels.is_empty() {
            10.0
        } else {
            0.0
        };
        let body_h =
            (levels.len() as f32 * line).max(if bar.is_some() { bar_h + line } else { 0.0 });
        let w = text_w(&title).max(iso_col_w + gap + bar_col_w).max(46.0) + 10.0;
        let h = line + 4.0 + body_h + 4.0;
        let bg = Rect::from_min_size(top_right - Vec2::new(w, 0.0), Vec2::new(w, h));
        painter.rect_filled(bg, 3.0, Color32::from_black_alpha(150));
        painter.text(
            Pos2::new(bg.right() - 5.0, bg.top() + 3.0),
            Align2::RIGHT_TOP,
            title,
            font.clone(),
            text,
        );
        let top = bg.top() + 3.0 + line + 4.0;

        if let Some(b) = &bar {
            let r = Rect::from_min_size(
                Pos2::new(bg.right() - 16.0, top + line * 0.5),
                Vec2::new(10.0, bar_h),
            );
            let hi = b.scale.hi;
            let span = (hi - b.bottom).max(1e-6);
            let y_of = |v: f32| r.bottom() - (v - b.bottom) / span * r.height();
            // One strip a pixel, coloured as the wash colours that dose:
            // stepped bands show as bands, a continuous map as a gradient.
            let n = r.height().round().max(1.0) as usize;
            for i in 0..n {
                let v = b.bottom + (i as f32 + 0.5) / n as f32 * span;
                let [cr, cg, cb] = b.scale.rgb(v);
                let y1 = r.bottom() - i as f32 / n as f32 * r.height();
                let y0 = r.bottom() - (i + 1) as f32 / n as f32 * r.height();
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(r.left(), y0 - 0.5), Pos2::new(r.right(), y1)),
                    0.0,
                    Color32::from_rgb(cr, cg, cb),
                );
            }
            painter.rect_stroke(
                r,
                0.0,
                Stroke::new(1.0, Color32::from_gray(160)),
                egui::StrokeKind::Outside,
            );
            // Every tick marked, labels as many as fit without touching.
            let mut last_label_y = f32::INFINITY;
            for (v, label) in &b.ticks {
                let y = y_of(*v);
                painter.line_segment(
                    [Pos2::new(r.left() - 3.0, y), Pos2::new(r.left(), y)],
                    Stroke::new(1.0, text),
                );
                if last_label_y - y >= line - 2.0 {
                    painter.text(
                        Pos2::new(r.left() - 4.0, y),
                        Align2::RIGHT_CENTER,
                        label,
                        font.clone(),
                        text,
                    );
                    last_label_y = y;
                }
            }
        }

        let iso_right = bg.right() - 5.0 - bar_col_w - gap;
        for (i, ((_, color), label)) in levels.iter().zip(&iso_labels).enumerate() {
            let y = top + i as f32 * line + line * 0.5;
            let sw = Rect::from_center_size(Pos2::new(iso_right - 5.0, y), Vec2::new(10.0, 9.0));
            painter.rect_filled(sw, 1.0, *color);
            painter.text(
                Pos2::new(sw.left() - 4.0, y),
                Align2::RIGHT_CENTER,
                label,
                font.clone(),
                text,
            );
        }
    }
}

/// `Gy` for the `GY` a dose grid states, its own word otherwise.
fn gy_units(units: &str) -> String {
    if units.eq_ignore_ascii_case("GY") || units.is_empty() {
        "Gy".into()
    } else {
        units.to_string()
    }
}
