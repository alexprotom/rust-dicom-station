//! The posture window's table and chart ([`crate::posture::measure`]).
//!
//! Four folds - States, Structures, Bones and joints, Model - each an
//! `egui::Grid`, every number in the anchor frame and in patient
//! directions. The time-dependent columns follow the slider: they read the
//! sampled times of [`crate::posture::measure::Tables`] interpolated at the
//! slider's time ([`crate::posture::measure::lerp_at`]), "from the first
//! state to here". A click on a structure's or body's name focuses it (its
//! contour drawn thicker in the views, the other surfaces of the 3D scene
//! dimmed); the 📈 at the end of a row puts its curves against time - or
//! against the posture parameter when every state has one - into the chart
//! pane, through [`super::chart_pane::ChartSource::Posture`]. With a beam
//! asked for, the Structures fold has the WEPL to each centroid along it
//! and its change, at the slider's time.

use super::motion_results::{line_chart, Series};
use super::posture_win::ChartRow;
use super::*;
use crate::posture::measure::{lerp_at, Tables};

/// A number, or a dash where there is none.
fn f(v: f64, d: usize) -> String {
    if v.is_finite() {
        format!("{v:.d$}")
    } else {
        "-".into()
    }
}

fn fo(v: Option<f64>, d: usize) -> String {
    v.map_or_else(|| "-".into(), |x| f(x, d))
}

impl ViewerApp {
    /// A sampled value at the slider's time.
    fn posture_at(tables: &Tables, t: f64, value: impl Fn(usize) -> f64) -> f64 {
        let v: Vec<f64> = (0..tables.times.len()).map(value).collect();
        lerp_at(&tables.times, &v, t)
    }

    /// The four folds.
    pub(super) fn posture_table(&mut self, ui: &mut egui::Ui) {
        let Some(b) = self.posture.built.as_ref() else {
            return;
        };
        let t = self.posture.t;
        let tb = &b.tables;
        let mut focus: Option<Option<String>> = None;
        let mut chart: Option<ChartRow> = None;
        let mut add_joint = false;
        let names = b.model.body_names();

        // ---- States ----
        egui::CollapsingHeader::new("States")
            .id_salt("posture_states")
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("posture_states_grid")
                    .striped(true)
                    .num_columns(7)
                    .show(ui, |ui| {
                        for h in ["State", "Series", "Param", "Grid", "Into anchor", "Body cm3", "Change"] {
                            ui.strong(h);
                        }
                        ui.end_row();
                        for s in &tb.states {
                            ui.label(&s.label);
                            ui.label(format!("{} {}", s.date, s.series))
                                .on_hover_text(format!("Patient position: {}", if s.position.is_empty() { "-" } else { &s.position }));
                            ui.label(fo(s.posture, 1));
                            ui.label(format!(
                                "{}x{}x{}, {:.1} mm",
                                s.dims[0],
                                s.dims[1],
                                s.dims[2],
                                s.spacing[2]
                            ))
                            .on_hover_text(format!("SI {:.0} to {:.0} mm in the anchor frame", s.si_range.0, s.si_range.1));
                            ui.label(s.to_anchor.line())
                                .on_hover_text("The rigid transform that brought this state into the anchor frame, about the anchor's centroid");
                            ui.label(fo(s.body_cm3, 0));
                            ui.label(fo(s.body_change_cm3, 1));
                            ui.end_row();
                        }
                    });
            });

        // ---- Structures ----
        let i_t = tb.nearest(t);
        let wepl = tb.model.beam.clone();
        egui::CollapsingHeader::new("Structures")
            .id_salt("posture_structures")
            .default_open(true)
            .show(ui, |ui| {
                ui.weak(format!(
                    "Model columns at t = {t:.2}; contoured columns first state to last. {} structure(s).",
                    tb.structures.len()
                ));
                if let Some(beam) = &wepl {
                    ui.weak(format!(
                        "WEPL along the beam at {beam}, fixed in the anchor frame, through the image at the \
                         slider's time (the scans at the states, the model's interpolation between them)."
                    ));
                }
                egui::Grid::new("posture_structures_grid")
                    .striped(true)
                    .num_columns(if wepl.is_some() { 15 } else { 13 })
                    .show(ui, |ui| {
                        for (h, tip) in [
                            ("Structure", "Click to focus it in the views and the 3D scene"),
                            ("Fit", "Used in the fit (a rigid body or the anchor)"),
                            ("dV cm3", "Volume change as contoured, first state to last"),
                            ("dV %", "The same in percent"),
                            ("dV FOV cm3", "Volume change inside the part every scan covers: the one to read for a structure a scan cuts (marked cut)"),
                            ("Contoured mm", "Centroid displacement RL / AP / SI as contoured, first state to last"),
                            ("Model mm", "Centroid displacement 3D as the model carries it, to the slider's time"),
                            ("Path mm", "Length of the centroid's path so far: longer than the displacement when it curves"),
                            ("Rot deg", "Rotation of the best rigid fit of the structure's own motion so far"),
                            ("det J", "Local volume change inside: mean, 5th and 95th percentile of the Jacobian"),
                            ("After rigid", "What is left after that rigid fit, against the last contour: Dice / HD95 mm"),
                            ("HU", "Mean HU on the first and last state: a change of content rather than of shape"),
                            ("Check", "The first contour carried by the model against the last state's own: Dice / HD95 mm"),
                        ] {
                            ui.strong(h).on_hover_text(tip);
                        }
                        if wepl.is_some() {
                            ui.strong("WEPL mm").on_hover_text(
                                "Water-equivalent path length from where the beam enters the image to the centroid, at the slider's time",
                            );
                            ui.strong("dWEPL mm")
                                .on_hover_text("Its change since the first state: how far the range must move to follow the centroid");
                        }
                        ui.label("");
                        ui.end_row();
                        for s in &tb.structures {
                            let on = self.posture.focus.as_deref() == Some(s.name.as_str());
                            let c = Color32::from_rgb(s.color[0], s.color[1], s.color[2]);
                            if ui
                                .selectable_label(on, egui::RichText::new(&s.name).color(c))
                                .on_hover_text("Focus it in the views and the 3D scene")
                                .clicked()
                            {
                                focus = Some((!on).then(|| s.name.clone()));
                            }
                            ui.label(if s.used { "yes" } else { "" });
                            let (dv, dp) = s.volume_change();
                            ui.label(f(dv, 1));
                            ui.label(f(dp, 0));
                            let common = f(s.common_change(), 1);
                            if s.is_cut() {
                                ui.label(format!("{common} cut")).on_hover_text(
                                    "A scan's field of view cuts it: its contoured volume change is mostly the cut",
                                );
                            } else {
                                ui.label(common);
                            }
                            ui.label(match s.contoured_shift() {
                                Some(v) => format!("{:.1} {:.1} {:.1}", v.x, v.y, v.z),
                                None => "-".into(),
                            });
                            let shift = Self::posture_at(tb, t, |i| s.model_shift(i).map_or(f64::NAN, |v| v.length()));
                            ui.label(f(shift, 1));
                            ui.label(f(Self::posture_at(tb, t, |i| s.at[i].path_mm), 1));
                            ui.label(f(Self::posture_at(tb, t, |i| s.at[i].rotation_deg), 1));
                            let j = s.at.get(i_t).map(|a| a.jacobian).unwrap_or([f64::NAN; 3]);
                            ui.label(format!("{} ({}-{})", f(j[0], 2), f(j[1], 2), f(j[2], 2)));
                            ui.label(match s.deformation {
                                Some(a) => format!("{:.2} / {:.1}", a.dice, a.hd95_mm),
                                None => "-".into(),
                            });
                            ui.label(format!(
                                "{} / {}",
                                fo(s.mean_hu.first().copied().flatten(), 0),
                                fo(s.mean_hu.last().copied().flatten(), 0)
                            ));
                            ui.label(match s.check {
                                Some(a) => format!("{:.2} / {:.1}", a.dice, a.hd95_mm),
                                None => "-".into(),
                            });
                            if wepl.is_some() {
                                let w = |i: usize| s.wepl_mm.get(i).copied().unwrap_or(f64::NAN);
                                ui.label(f(Self::posture_at(tb, t, w), 1));
                                ui.label(f(Self::posture_at(tb, t, |i| s.wepl_change(i).unwrap_or(f64::NAN)), 1));
                            }
                            if ui
                                .small_button("📈")
                                .named_with(|| format!("Chart {}", s.name))
                                .on_hover_text("Its displacement, rotation and volume change against time, in the chart pane")
                                .clicked()
                            {
                                chart = Some(ChartRow::Structure(s.name.clone()));
                            }
                            ui.end_row();
                        }
                    });
            });

        // ---- Bones and joints ----
        egui::CollapsingHeader::new("Bones and joints")
            .id_salt("posture_bodies")
            .default_open(true)
            .show(ui, |ui| {
                if tb.bodies.is_empty() {
                    ui.weak("No rigid bodies: the path is the residual alone.");
                }
                egui::Grid::new("posture_bodies_grid")
                    .striped(true)
                    .num_columns(8)
                    .show(ui, |ui| {
                        for (h, tip) in [
                            ("Body", "Click to focus it"),
                            ("Angle deg", "Turn so far on its screw (at the end: the whole screw)"),
                            ("Axis RL AP SI", "Direction of the screw's axis over the whole path"),
                            ("Travel mm", "Travel along the axis over the whole path"),
                            ("Shift mm", "How far its centroid moved so far"),
                            ("Fit mm", "Mean surface distance of its rigid fit"),
                            ("Stretch", "Largest stretch inside it along the path, and its 95th percentile: about zero for a bone that keeps its shape (the largest reads the blend near a touching joint)"),
                        ] {
                            ui.strong(h).on_hover_text(tip);
                        }
                        ui.label("");
                        ui.end_row();
                        for bd in &tb.bodies {
                            let on = self.posture.focus.as_deref() == Some(bd.name.as_str());
                            let c = Color32::from_rgb(bd.color[0], bd.color[1], bd.color[2]);
                            let mut text = egui::RichText::new(&bd.name).color(c);
                            if bd.anchor {
                                text = text.italics();
                            }
                            if ui
                                .selectable_label(on, text)
                                .on_hover_text(if bd.anchor {
                                    "The anchor: still by definition"
                                } else {
                                    "Focus it in the views and the 3D scene"
                                })
                                .clicked()
                            {
                                focus = Some((!on).then(|| bd.name.clone()));
                            }
                            ui.label(f(Self::posture_at(tb, t, |i| bd.at[i].angle_deg), 1));
                            ui.label(format!(
                                "{:.2} {:.2} {:.2}",
                                bd.screw.axis.x, bd.screw.axis.y, bd.screw.axis.z
                            ));
                            ui.label(f(bd.screw.travel_mm, 1));
                            ui.label(f(Self::posture_at(tb, t, |i| bd.at[i].shift.length()), 1));
                            ui.label(if bd.anchor { "anchor".to_string() } else { f(bd.fit_mean_mm, 2) });
                            ui.label(format!(
                                "{} / {}",
                                f(Self::posture_at(tb, t, |i| bd.at[i].stretch), 3),
                                f(bd.stretch_p95, 3)
                            ));
                            if ui
                                .small_button("📈")
                                .named_with(|| format!("Chart the turn of {}", bd.name))
                                .on_hover_text("Its angle against time, in the chart pane")
                                .clicked()
                            {
                                chart = Some(ChartRow::Body(bd.name.clone()));
                            }
                            ui.end_row();
                        }
                    });
                if !tb.joints.is_empty() {
                    ui.add_space(4.0);
                    egui::Grid::new("posture_joints_grid")
                        .striped(true)
                        .num_columns(5)
                        .show(ui, |ui| {
                            for (h, tip) in [
                                ("Joint", "The second body's rotation relative to the first"),
                                ("RL deg", "About right-left: flexion / extension"),
                                ("AP deg", "About anterior-posterior: abduction / adduction, side bending"),
                                ("SI deg", "About inferior-superior: axial rotation"),
                            ] {
                                ui.strong(h).on_hover_text(tip);
                            }
                            ui.label("");
                            ui.end_row();
                            for j in &tb.joints {
                                ui.label(&j.name)
                                    .on_hover_text(format!("{} relative to {}", j.second, j.first));
                                for a in 0..3 {
                                    ui.label(f(Self::posture_at(tb, t, |i| j.at[i][a]), 1));
                                }
                                if ui
                                    .small_button("📈")
                                    .named_with(|| format!("Chart the joint {}", j.name))
                                    .on_hover_text("Its three angles against time, in the chart pane")
                                    .clicked()
                                {
                                    chart = Some(ChartRow::Joint(j.name.clone()));
                                }
                                ui.end_row();
                            }
                        });
                }
                if names.len() >= 2 {
                    ui.horizontal(|ui| {
                        ui.label("Joint:");
                        for k in 0..2 {
                            egui::ComboBox::from_id_salt(("posture_joint", k))
                                .selected_text(if self.posture.joint_pick[k].is_empty() {
                                    "(body)".to_string()
                                } else {
                                    self.posture.joint_pick[k].clone()
                                })
                                .width(110.0)
                                .show_ui(ui, |ui| {
                                    for n in &names {
                                        ui.selectable_value(&mut self.posture.joint_pick[k], n.clone(), n);
                                    }
                                })
                                .response
                                .on_hover_text(if k == 0 { "The first body (the one the angle is measured from)" } else { "The second body" });
                        }
                        let ok = !self.posture.joint_pick[0].is_empty()
                            && !self.posture.joint_pick[1].is_empty()
                            && self.posture.joint_pick[0] != self.posture.joint_pick[1];
                        if enabled_tip_button(ui, ok, "➕ Add", "Report this pair as a joint (the next Build reads it)") {
                            add_joint = true;
                        }
                    });
                }
            });

        // ---- Model ----
        egui::CollapsingHeader::new("Model")
            .id_salt("posture_model")
            .default_open(false)
            .show(ui, |ui| {
                let m = &tb.model;
                egui::Grid::new("posture_model_grid")
                    .striped(true)
                    .num_columns(2)
                    .show(ui, |ui| {
                        let rows: Vec<(&str, String)> = vec![
                            ("Mode", m.mode.label().to_string()),
                            ("Anchor", m.anchor.clone()),
                            (
                                "Bodies",
                                if m.bodies.is_empty() {
                                    "-".into()
                                } else {
                                    m.bodies.join(", ")
                                },
                            ),
                            (
                                "Lattice",
                                format!(
                                    "{}x{}x{} at {:.1} mm, reach {:.0} mm",
                                    m.lattice_dims[0],
                                    m.lattice_dims[1],
                                    m.lattice_dims[2],
                                    m.lattice_mm,
                                    m.reach_mm
                                ),
                            ),
                            (
                                "Analysis grid",
                                format!(
                                    "{:.1} x {:.1} x {:.1} mm",
                                    m.analysis_mm[0], m.analysis_mm[1], m.analysis_mm[2]
                                ),
                            ),
                            (
                                "Inverse consistency",
                                format!(
                                    "half way {:.2} / {:.2} mm, end {:.2} / {:.2} mm (max / p95)",
                                    m.inverse_mm[0][0],
                                    m.inverse_mm[0][1],
                                    m.inverse_mm[1][0],
                                    m.inverse_mm[1][1]
                                ),
                            ),
                            (
                                "Jacobian",
                                format!(
                                    "min {:.3}, folded {}",
                                    m.jacobian_min,
                                    if m.folded_fraction > 0.0 {
                                        format!("{:.3} % of the lattice", 100.0 * m.folded_fraction)
                                    } else {
                                        "nowhere".into()
                                    }
                                ),
                            ),
                            (
                                "Check (Dice)",
                                format!(
                                    "used in the fit {}, not used {}",
                                    fo(m.check_used_dice, 2),
                                    fo(m.check_unused_dice, 2)
                                ),
                            ),
                            (
                                "Overlap of the scans",
                                format!(
                                    "{:.0} % of the first, SI {:.0} to {:.0} mm",
                                    100.0 * m.overlap_fraction,
                                    m.overlap_si.0,
                                    m.overlap_si.1
                                ),
                            ),
                        ];
                        for (k, v) in rows {
                            ui.strong(k);
                            ui.label(v);
                            ui.end_row();
                        }
                    });
                if m.mode == crate::posture::Mode::Linear && m.folded_fraction > 0.0 {
                    ui.colored_label(
                        ui.visuals().warn_fg_color,
                        "The linear path folds on the way: it is not invertible there.",
                    );
                }
                ui.add_space(4.0);
                ui.strong("Provenance");
                for line in &m.provenance {
                    ui.weak(line);
                }
            });

        if let Some(f) = focus {
            self.posture.focus = f;
            self.posture_dim_others();
        }
        if add_joint {
            let [a, b] = self.posture.joint_pick.clone();
            self.posture.joints.push((format!("{a} / {b}"), a, b));
            self.posture.status = Some("Joint added: Build again to measure it".into());
        }
        if let Some(row) = chart {
            self.posture.chart = Some(row);
            if let Some(slot) = self.posture.slot {
                self.chart[slot] = Some(chart_pane::ChartSource::Posture);
            }
        }
    }

    /// The focused structure stands out in the 3D scene: the others drawn
    /// faint.
    fn posture_dim_others(&mut self) {
        let Some(slot) = self.posture.slot else {
            return;
        };
        let focus = self.posture_focus(slot);
        let n = self.slots[slot]
            .active_structures()
            .map_or(0, |ss| ss.rois.len());
        if let Some(w) = self.d3_windows.iter_mut().find(|w| w.slot == slot) {
            w.roi_alpha.clear();
            if let Some(fi) = focus {
                for i in (0..n).filter(|&i| i != fi) {
                    w.roi_alpha.insert(i, 0.25);
                }
            }
            w.frame = Default::default();
        }
    }

    /// What the chart pane says above the posture chart.
    pub(super) fn posture_chart_subtitle(&self) -> String {
        match &self.posture.chart {
            Some(ChartRow::Structure(n)) | Some(ChartRow::Body(n)) | Some(ChartRow::Joint(n)) => {
                n.clone()
            }
            None => String::new(),
        }
    }

    /// The chart of the row picked in the table.
    pub(super) fn posture_chart(&mut self, ui: &mut egui::Ui, body: Rect) {
        let Some(b) = self.posture.built.as_ref() else {
            return;
        };
        let tb = &b.tables;
        let param = tb.params.iter().all(|p| p.is_some()) && !tb.params.is_empty();
        let xs: Vec<String> = tb
            .times
            .iter()
            .zip(&tb.params)
            .map(|(t, p)| match (param, p) {
                (true, Some(p)) => format!("{p:.0}"),
                _ => format!("{t:.1}"),
            })
            .collect();
        let palette = [
            Color32::from_rgb(0x4e, 0x79, 0xa7),
            Color32::from_rgb(0xf2, 0x8e, 0x2b),
            Color32::from_rgb(0x59, 0xa1, 0x4f),
        ];
        let line = |label: &str, i: usize, values: Vec<f64>| Series {
            label: label.to_string(),
            color: palette[i % palette.len()],
            values,
            dashed: false,
        };
        let mut clicked: Option<usize> = None;
        egui::ScrollArea::vertical()
            .id_salt("posture_chart")
            .max_height(body.height())
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.weak(if param {
                    "Against the posture parameter"
                } else {
                    "Against time along the path (one unit per pair of states)"
                });
                match &self.posture.chart {
                    Some(ChartRow::Structure(name)) => {
                        let Some(s) = tb.structures.iter().find(|s| &s.name == name) else {
                            return;
                        };
                        let shift = (0..s.at.len())
                            .map(|i| s.model_shift(i).map_or(f64::NAN, |v| v.length()))
                            .collect();
                        clicked = clicked.or(line_chart(
                            ui,
                            &xs,
                            &[
                                line("displacement", 0, shift),
                                line("path", 1, s.at.iter().map(|a| a.path_mm).collect()),
                            ],
                            "mm",
                        ));
                        clicked = clicked.or(line_chart(
                            ui,
                            &xs,
                            &[line(
                                "rotation",
                                2,
                                s.at.iter().map(|a| a.rotation_deg).collect(),
                            )],
                            "deg",
                        ));
                        if !s.wepl_mm.is_empty() {
                            let change = (0..s.wepl_mm.len())
                                .map(|i| s.wepl_change(i).unwrap_or(f64::NAN))
                                .collect();
                            clicked = clicked.or(line_chart(
                                ui,
                                &xs,
                                &[line("WEPL change", 3, change)],
                                "mm",
                            ));
                        }
                        clicked = clicked.or(line_chart(
                            ui,
                            &xs,
                            &[
                                line(
                                    "det J mean",
                                    0,
                                    s.at.iter().map(|a| a.jacobian[0]).collect(),
                                ),
                                line("p5", 1, s.at.iter().map(|a| a.jacobian[1]).collect()),
                                line("p95", 2, s.at.iter().map(|a| a.jacobian[2]).collect()),
                            ],
                            "det J",
                        ));
                    }
                    Some(ChartRow::Body(name)) => {
                        let Some(bd) = tb.bodies.iter().find(|x| &x.name == name) else {
                            return;
                        };
                        clicked = clicked.or(line_chart(
                            ui,
                            &xs,
                            &[line(
                                "angle",
                                0,
                                bd.at.iter().map(|a| a.angle_deg).collect(),
                            )],
                            "deg",
                        ));
                        clicked = clicked.or(line_chart(
                            ui,
                            &xs,
                            &[line(
                                "stretch",
                                1,
                                bd.at.iter().map(|a| a.stretch).collect(),
                            )],
                            "stretch",
                        ));
                    }
                    Some(ChartRow::Joint(name)) => {
                        let Some(j) = tb.joints.iter().find(|x| &x.name == name) else {
                            return;
                        };
                        let series: Vec<Series> = crate::motion::AXES
                            .iter()
                            .enumerate()
                            .map(|(a, ax)| line(ax, a, j.at.iter().map(|v| v[a]).collect()))
                            .collect();
                        clicked = clicked.or(line_chart(ui, &xs, &series, "deg"));
                    }
                    None => {}
                }
            });
        // A click on the chart moves the slider there.
        if let Some(i) = clicked {
            if let Some(t) = tb.times.get(i).copied() {
                let now = ui.input(|i| i.time);
                self.posture_set_t(t, now);
            }
        }
    }
}
