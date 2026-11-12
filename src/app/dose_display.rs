//! *Modules ▶ Dose display*: how the displayed dose grid is drawn over the
//! images - colorwash, isodose lines or both, the opacity, the threshold
//! under which nothing is drawn, the isodose ladder - each workspace's
//! reference dose (the 100 % every level and the wash are relative to), and
//! the dose legend on the views.
//!
//! These are display settings of the viewer, not properties of a grid: one
//! set serves every workspace, which is why they left the data tree (where
//! they sat under whichever workspace happened to be the first with dose)
//! for a module of their own. *Which* grid is drawn stays with the grid
//! under *Dose* in the tree; a grid's context menu opens this section.
//!
//! The legend is the one a planning system puts in the corner of its views:
//! the reference it is a percentage of, then the isodose levels drawn, each
//! with its colour, highest first - or, for a colorwash alone, the wash's
//! scale from the threshold up to 100 %. It is drawn on the views ticked
//! here, below the pane's buttons on the right.

use super::*;

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
        }

        ui.add(egui::Slider::new(&mut opacity, 0.0..=1.0).text("Opacity"));
        ui.add(egui::Slider::new(&mut threshold, 0.0..=100.0).text("Threshold %"));
        ui.weak("Isodose levels (% of reference)");
        for l in &mut self.iso_levels {
            ui.horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), Sense::hover());
                ui.painter().rect_filled(rect, 2.0, l.color);
                ui.checkbox(&mut l.on, format!("{:.0}%", l.pct));
            });
        }

        // The legend over the views.
        ui.separator();
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
    }

    /// The dose legend of `slot`, its top right corner at `top_right`:
    /// nothing while that workspace draws no dose.
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
        let units = gy_units(&d.units);
        let mut title = format!("% of {:.2} {units}", st.dose_reference);
        if d.dose_type.eq_ignore_ascii_case("EFFECTIVE") {
            title.push_str(" (RBE)");
        }
        let font = FontId::proportional(11.0);
        let text = Color32::from_gray(230);
        let line = 13.0;
        // Isodose lines are what a legend names; a wash alone gets its
        // scale instead.
        let mut levels: Vec<(f32, Color32)> = if self.dose_mode.iso() {
            self.iso_levels
                .iter()
                .filter(|l| l.on)
                .map(|l| (l.pct, l.color))
                .collect()
        } else {
            Vec::new()
        };
        levels.sort_by(|a, b| b.0.total_cmp(&a.0));
        let title_w = painter
            .layout_no_wrap(title.clone(), font.clone(), text)
            .size()
            .x;
        let wash_h = 96.0;
        let body_h = if levels.is_empty() {
            wash_h + 6.0
        } else {
            levels.len() as f32 * line
        };
        let w = title_w.max(56.0) + 10.0;
        let h = line + 4.0 + body_h + 6.0;
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
        if levels.is_empty() {
            // The wash: t = dose / reference, nothing under the threshold.
            let bar = Rect::from_min_size(
                Pos2::new(bg.right() - 16.0, top + 3.0),
                Vec2::new(10.0, wash_h),
            );
            let lo = (self.dose_threshold_pct / 100.0).clamp(0.0, 1.0);
            let steps = 48;
            for i in 0..steps {
                let a = i as f32 / steps as f32;
                let b = (i + 1) as f32 / steps as f32;
                let t = 1.0 - (a + b) * 0.5 * (1.0 - lo);
                let c = render::dose_colormap(t);
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(bar.left(), bar.top() + a * bar.height()),
                        Pos2::new(bar.right(), bar.top() + b * bar.height() + 0.5),
                    ),
                    0.0,
                    Color32::from_rgb(c[0], c[1], c[2]),
                );
            }
            for pct in [100.0f32, 75.0, 50.0, 25.0, self.dose_threshold_pct] {
                let t = pct / 100.0;
                if t + 1e-3 < lo || t > 1.0 {
                    continue;
                }
                let y = bar.top() + (1.0 - t) / (1.0 - lo).max(1e-3) * bar.height();
                painter.text(
                    Pos2::new(bar.left() - 4.0, y),
                    Align2::RIGHT_CENTER,
                    format!("{pct:.0}"),
                    font.clone(),
                    text,
                );
            }
        } else {
            for (i, (pct, color)) in levels.iter().enumerate() {
                let y = top + i as f32 * line + line * 0.5;
                let sw =
                    Rect::from_center_size(Pos2::new(bg.right() - 10.0, y), Vec2::new(10.0, 9.0));
                painter.rect_filled(sw, 1.0, *color);
                painter.text(
                    Pos2::new(sw.left() - 4.0, y),
                    Align2::RIGHT_CENTER,
                    format!("{pct:.0}"),
                    font.clone(),
                    text,
                );
            }
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
