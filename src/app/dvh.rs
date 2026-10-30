//! *Modules ▶ Dose-volume histograms*: the pickers, the table and the
//! constraint check in a section of the right panel, the plot in the row's
//! chart pane (`chart_pane.rs`), which can be maximized to the whole window.
//!
//! The module is deliberately a *review* tool rather than a dialog. It stays
//! set up, recomputes when the picks change, and shows three things at once
//! because that is how a plan is actually read: the curves for shape, the
//! table for the numbers a report quotes, and - when a protocol is loaded -
//! the pass/fail column that says whether the plan is acceptable.
//!
//! It talks to the views: the curve under the pointer in the chart lights
//! its structure up in the views, and *Dose-volume histograms* on ticked
//! structures in the data tree adds them to the plot and unfolds the
//! section.
//!
//! Two design points worth stating, because both are easy to get wrong in a
//! way nobody notices:
//!
//! * **Every curve names its dose object.** Overlaying two plans is the
//!   reason to allow more than one, and a legend that says only "Cord" twice
//!   is worse than no legend. Structures keep their own colour and the dose
//!   object picks the line style, so the eye groups by structure and reads
//!   the comparison along each colour.
//!
//! * **A structure sticking out of the dose grid is called out.** Those
//!   voxels are counted at zero dose, which drags the curve down and is the
//!   honest reading - but silently, it looks like a cold structure rather
//!   than a truncated calculation, so the section says so in the table and
//!   in a warning line.
//!
//! The plot is drawn with the painter rather than a plotting crate: axes,
//! ticks, polylines and a hover readout are a hundred lines, and the
//! alternative is a dependency whose styling would have to be fought into
//! agreement with the rest of the interface anyway.

use crate::dvh::{self, Constraint, Dvh, DvhParams, Metric};
use crate::progress::ProgressSink;

use super::combine::ItemRef;
use super::*;

/// Which dose object, in which workspace.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct DoseRef {
    pub slot: usize,
    pub idx: usize,
}

/// One picked structure.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct StructRef {
    pub slot: usize,
    pub item: ItemRef,
}

/// The module's state; it stays set up across runs.
pub(super) struct DvhModule {
    pub structures: Vec<StructRef>,
    pub doses: Vec<DoseRef>,
    /// Cumulative, or the differential histogram.
    pub cumulative: bool,
    /// Dose axis as a percentage of [`Self::reference_dose`].
    pub dose_relative: bool,
    /// Volume axis as a percentage of each structure.
    pub volume_relative: bool,
    /// What 100 % means on the dose axis.
    pub reference_dose: f64,
    pub metrics: Vec<Metric>,
    /// The text of the "add a column" field.
    pub new_metric: String,
    pub constraints: Vec<Constraint>,
    pub protocol_name: String,
    pub show_constraints: bool,
    /// The last computed curves, in the order the structures were picked.
    pub curves: Vec<Dvh>,
    /// Which picked structure each curve is of, in the same order.
    pub curve_refs: Vec<StructRef>,
    /// The structures of the run in flight, in the order it was asked for.
    pub asked: Vec<StructRef>,
    /// The structure whose curve is under the pointer in the chart, for the
    /// views to draw thicker.
    pub hover: Option<StructRef>,
    /// The workspace whose row the chart pane goes into.
    pub row: usize,
    /// Put the chart pane into the row when the next run lands: set by a
    /// change the user made, so a pane they closed is not pushed back by a
    /// recompute they did not ask for.
    pub show_chart: bool,
    /// The `settings_gen` the curves were computed at. A structure edited
    /// afterwards makes every number in this module older than the geometry
    /// it claims to describe, which is worth saying out loud rather than
    /// recomputing behind the user's back.
    pub gen: u64,
    pub status: Option<String>,
    /// What was ticked in the data tree at the last pass the section was
    /// unfolded, and what that tick list belonged to: the pass number, the
    /// workspaces' studies and active sets, and the ticked items
    /// ([`ViewerApp::dvh_follow_ticks`]).
    ticks_seen: Option<(u64, TickOwners, Ticked)>,
}

/// Per open workspace: its slot, its study (by address), its active
/// structure set and its active segmentation series - what a tick list
/// belongs to.
type TickOwners = Vec<[usize; 4]>;
/// The ticked structures and segments, each with its workspace.
type Ticked = Vec<(usize, ItemRef)>;

impl DvhModule {
    fn new() -> DvhModule {
        DvhModule {
            structures: Vec::new(),
            doses: Vec::new(),
            cumulative: true,
            dose_relative: false,
            volume_relative: true,
            reference_dose: 0.0,
            metrics: dvh::default_metrics(),
            new_metric: String::new(),
            constraints: Vec::new(),
            protocol_name: String::new(),
            show_constraints: false,
            curves: Vec::new(),
            curve_refs: Vec::new(),
            asked: Vec::new(),
            hover: None,
            row: 0,
            show_chart: true,
            gen: 0,
            status: None,
            ticks_seen: None,
        }
    }
}

/// Everything a run needs, snapshotted when it starts.
struct DvhRequest {
    /// Mask, lattice, name and colour of each structure.
    items: Vec<(Vec<u8>, crate::volume::Grid, String, [u8; 3])>,
    doses: Vec<crate::rtdose::DoseGrid>,
}

/// What a finished run hands back.
pub struct DvhDone {
    pub curves: Vec<Dvh>,
    /// The index into the request's structures each curve is of.
    pub sources: Vec<usize>,
    pub elapsed_secs: f64,
}

impl ViewerApp {
    /// Every dose object of both workspaces, as (reference, label).
    pub(super) fn dvh_dose_candidates(&self) -> Vec<(DoseRef, String)> {
        let mut out = Vec::new();
        let comparing = self.comparing();
        for slot in self.open_slots() {
            let name = SLOT_NAMES[slot];
            let Some(study) = self.slots[slot].study.as_ref() else {
                continue;
            };
            for (i, d) in study.doses.iter().enumerate() {
                let label = if d.label.is_empty() {
                    format!("Dose {}", i + 1)
                } else {
                    d.label.clone()
                };
                out.push((
                    DoseRef { slot, idx: i },
                    if comparing {
                        format!("{name} · {label}")
                    } else {
                        label
                    },
                ));
            }
        }
        out
    }

    /// Every structure and segment of both workspaces, as (reference, label).
    pub(super) fn dvh_struct_candidates(&self) -> Vec<(StructRef, String)> {
        let mut out = Vec::new();
        let comparing = self.comparing();
        for slot in self.open_slots() {
            let name = SLOT_NAMES[slot];
            for (item, label) in self.combine_candidates(slot) {
                out.push((
                    StructRef { slot, item },
                    if comparing {
                        format!("{name} · {label}")
                    } else {
                        label
                    },
                ));
            }
        }
        out
    }

    /// The prescription of the first plan that declares one - what the
    /// percentage dose axis is measured against until the user says
    /// otherwise.
    fn prescription(&self) -> Option<(f64, String)> {
        for slot in &self.slots {
            let Some(study) = slot.study.as_ref() else {
                continue;
            };
            for p in &study.plans {
                if let Some(d) = p.target_prescription_dose.filter(|d| *d > 0.0) {
                    let name = if p.label.is_empty() {
                        p.name.clone()
                    } else {
                        p.label.clone()
                    };
                    return Some((d, name));
                }
            }
        }
        None
    }

    /// The module's state, made the first time it is asked for: the dose
    /// axis against the first prescription the plans declare, the chart in
    /// the row of the first workspace that holds a dose.
    fn ensure_dvh(&mut self) {
        if self.dvh.is_some() {
            return;
        }
        let mut d = DvhModule::new();
        if let Some((dose, _)) = self.prescription() {
            d.reference_dose = dose;
        }
        d.row = self
            .open_slots()
            .into_iter()
            .find(|s| {
                self.slots[*s]
                    .study
                    .as_ref()
                    .is_some_and(|st| !st.doses.is_empty())
            })
            .unwrap_or_else(|| self.first_volume_slot());
        self.dvh = Some(d);
    }

    /// The data tree's *Dose-volume histograms* on ticked structures: add
    /// them to the plot, with the dose on display when none is picked yet,
    /// and unfold the section.
    pub(super) fn seed_dvh(&mut self, slot: usize, seed: Vec<ItemRef>) {
        self.ensure_dvh();
        self.dvh_add(slot, seed);
        self.reveal_module(Module::Dvh);
    }

    /// What is ticked in the data tree: the shown structures of each open
    /// workspace's active set and the shown segments of its active
    /// segmentation series, with what that list belongs to (each
    /// workspace's study, its active set and series), so a switch of set or
    /// a new study is told from a tick.
    fn tree_ticks(&self) -> (TickOwners, Ticked) {
        let mut owners = Vec::new();
        let mut ticked = Vec::new();
        for slot in self.open_slots() {
            let s = &self.slots[slot];
            let Some(study) = s.study.as_ref() else {
                continue;
            };
            let series = s.seg_series_idx();
            owners.push([
                slot,
                study as *const LoadedStudy as usize,
                s.active_structs,
                series.unwrap_or(usize::MAX),
            ]);
            for (idx, _) in s.roi_visible.iter().enumerate().filter(|(_, v)| **v) {
                ticked.push((
                    slot,
                    ItemRef {
                        kind: SetKind::Structures,
                        set: s.active_structs,
                        idx,
                    },
                ));
            }
            if let Some(sr) = series.and_then(|i| study.seg_series.get(i).map(|sr| (i, sr))) {
                let (set, sr) = sr;
                for (idx, _) in sr.segs.iter().enumerate().filter(|(_, g)| g.visible) {
                    ticked.push((
                        slot,
                        ItemRef {
                            kind: SetKind::Segmentations,
                            set,
                            idx,
                        },
                    ));
                }
            }
        }
        (owners, ticked)
    }

    /// While the section is unfolded, a structure or segment ticked in the
    /// data tree joins the plot, as the tree's *DVH* action would add it.
    /// Unticking leaves it there (the plot's own list takes it off). Only a
    /// tick between two consecutive passes counts: on the pass the section
    /// unfolds, or after the active set or the study changed, the list is
    /// taken as it stands, so a set switched to or loaded is not poured in.
    fn dvh_follow_ticks(&mut self, pass: u64) {
        let (owners, now) = self.tree_ticks();
        let candidates = self.dvh_struct_candidates();
        let Some(d) = &mut self.dvh else {
            return;
        };
        let seen = d.ticks_seen.replace((pass, owners.clone(), now.clone()));
        let Some((last, was_owners, before)) = seen else {
            return;
        };
        if pass.saturating_sub(last) > 1 || was_owners != owners {
            return;
        }
        let mut by_slot: Vec<(usize, Vec<ItemRef>)> = Vec::new();
        for (slot, item) in now.into_iter().filter(|x| !before.contains(x)) {
            let r = StructRef { slot, item };
            if d.structures.contains(&r) || !candidates.iter().any(|(c, _)| *c == r) {
                continue;
            }
            match by_slot.iter_mut().find(|(s, _)| *s == slot) {
                Some((_, v)) => v.push(item),
                None => by_slot.push((slot, vec![item])),
            }
        }
        for (slot, items) in by_slot {
            self.dvh_add(slot, items);
        }
    }

    /// Add structures of `slot` to the plot and recompute.
    fn dvh_add(&mut self, slot: usize, seed: Vec<ItemRef>) {
        let doses = self.dvh_dose_candidates();
        let Some(d) = &mut self.dvh else {
            return;
        };
        for item in seed {
            let r = StructRef { slot, item };
            if !d.structures.contains(&r) {
                d.structures.push(r);
            }
        }
        if d.doses.is_empty() {
            // The dose the viewport is showing is the one meant, so start
            // there rather than with an empty plot.
            let active = DoseRef {
                slot,
                idx: self.slots[slot].active_dose,
            };
            if doses.iter().any(|(r, _)| *r == active) {
                d.doses.push(active);
            } else if let Some((r, _)) = doses.first() {
                d.doses.push(*r);
            }
        }
        d.row = slot;
        d.show_chart = true;
        self.start_dvh();
    }

    /// Snapshot the picks and compute on a worker thread.
    pub(super) fn start_dvh(&mut self) {
        if self.dvh_job.is_some() {
            return;
        }
        let Some(d) = &self.dvh else {
            return;
        };
        if d.structures.is_empty() || d.doses.is_empty() {
            if let Some(d) = &mut self.dvh {
                d.curves.clear();
                d.curve_refs.clear();
                d.hover = None;
            }
            return;
        }
        let mut items = Vec::with_capacity(d.structures.len());
        for s in &d.structures {
            match self.item_mask_grid(s.slot, s.item) {
                Some(v) => items.push(v),
                None => {
                    self.error = Some(
                        "One of the picked structures is empty on its image series, so \
                         its histogram would be meaningless. Remove it from the list."
                            .into(),
                    );
                    return;
                }
            }
        }
        let mut doses = Vec::with_capacity(d.doses.len());
        for r in &d.doses {
            let Some(g) = self.slots[r.slot]
                .study
                .as_ref()
                .and_then(|s| s.doses.get(r.idx))
            else {
                continue;
            };
            doses.push(g.clone());
        }
        if doses.is_empty() {
            return;
        }
        let req = DvhRequest { items, doses };
        // Which structure each curve will be of, by the index the worker
        // hands back.
        let asked = d.structures.clone();
        if let Some(d) = &mut self.dvh {
            d.asked = asked;
        }
        let progress = Arc::new(Progress::default());
        progress.set("Sampling dose");
        self.dvh_job = Some(Job::spawn(progress, move |p| {
            use rayon::prelude::*;
            use std::sync::atomic::{AtomicUsize, Ordering};
            let t0 = std::time::Instant::now();
            let total = (req.items.len() * req.doses.len()).max(1);
            // Every (dose, structure) curve is independent of the others, so
            // they are sampled in parallel; each one is still one sequential
            // walk, so a curve is the same number it always was, and the
            // list keeps the order the pairs were asked for in.
            type Item = (Vec<u8>, crate::volume::Grid, String, [u8; 3]);
            let pairs: Vec<(&crate::rtdose::DoseGrid, usize, &Item)> = req
                .doses
                .iter()
                .flat_map(|dose| {
                    req.items
                        .iter()
                        .enumerate()
                        .map(move |(i, item)| (dose, i, item))
                })
                .collect();
            let done = AtomicUsize::new(0);
            let curves: Vec<_> = pairs
                .par_iter()
                .filter_map(|(dose, si, (mask, grid, name, color))| {
                    if p.cancelled() {
                        return None;
                    }
                    let c = dvh::compute(name, *color, mask, grid, dose, DvhParams::default())
                        .ok()
                        .map(|c| (*si, c));
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    p.report(
                        n as f32 / total as f32,
                        &format!("{name} on {}", dose.label),
                    );
                    c
                })
                .collect();
            p.report(1.0, "Done");
            let (sources, curves) = curves.into_iter().unzip();
            Ok(DvhDone {
                curves,
                sources,
                elapsed_secs: t0.elapsed().as_secs_f64(),
            })
        }));
    }

    pub(super) fn on_dvh_done(&mut self, done: DvhDone) {
        let gen = self.all_structures_gen();
        let Some(d) = &mut self.dvh else {
            return;
        };
        let truncated = done
            .curves
            .iter()
            .filter(|c| c.outside_fraction() > 0.001)
            .count();
        d.status = Some(format!(
            "{} curve(s) in {:.2} s{}",
            done.curves.len(),
            done.elapsed_secs,
            match truncated {
                0 => String::new(),
                n => format!(" - {n} extend outside the dose grid"),
            }
        ));
        d.curve_refs = done
            .sources
            .iter()
            .filter_map(|i| d.asked.get(*i).copied())
            .collect();
        d.curves = done.curves;
        d.gen = gen;
        d.hover = None;
        let stale = d.structures != d.asked;
        if std::mem::take(&mut d.show_chart) && !d.curves.is_empty() {
            let row = d.row;
            self.show_chart(row, super::chart_pane::ChartSource::Dvh);
        }
        // Structures picked while this run was in flight (a tick in the
        // tree, the tree's DVH action) were not in it: run again for them.
        if stale {
            self.start_dvh();
        }
    }

    /// The chart pane's body when it shows the histograms: the plot, or the
    /// run's progress while one is in flight.
    pub(super) fn dvh_chart(&mut self, ui: &mut egui::Ui, body: Rect) {
        let progress = self.dvh_job.as_ref().map(|j| j.progress.clone());
        let Some(d) = &mut self.dvh else {
            return;
        };
        let mut cancel = false;
        match &progress {
            Some(p) => {
                cancel = progress_row(ui, p);
            }
            None => {
                let hover = plot(ui, d, body.height()).and_then(|i| d.curve_refs.get(i).copied());
                if hover != d.hover {
                    d.hover = hover;
                    // The section turns this into the structure the views
                    // draw thicker, on the pass this asks for.
                    ui.ctx().request_repaint();
                }
            }
        }
        cancel_if(cancel, &self.dvh_job);
    }

    /// The section.
    pub(super) fn dvh_section(&mut self, ui: &mut egui::Ui) {
        self.module_header(
            ui,
            Module::Dvh,
            |app, ui| {
                // The chart's own place is the row; the header brings it
                // back when it was closed.
                if app.dvh.as_ref().is_some_and(|d| !d.curves.is_empty())
                    && !app.chart_shown(super::chart_pane::ChartSource::Dvh)
                    && small_tip_button(ui, "📈", "Show the plot in the row's chart pane")
                {
                    let row = app.dvh.as_ref().map_or(0, |d| d.row);
                    app.show_chart(row, super::chart_pane::ChartSource::Dvh);
                }
            },
            |app, ui| app.dvh_body(ui),
        );
    }

    fn dvh_body(&mut self, ui: &mut egui::Ui) {
        if !self.slots.iter().any(|s| s.study.is_some()) {
            ui.weak("Load a workspace with a dose and structures");
            return;
        }
        self.ensure_dvh();
        self.dvh_follow_ticks(ui.ctx().cumulative_pass_nr());
        // Everything that reads the whole of `self` is settled first.
        let dose_list = self.dvh_dose_candidates();
        let struct_list = self.dvh_struct_candidates();
        let running = self.dvh_job.is_some();
        let progress = self.dvh_job.as_ref().map(|j| j.progress.clone());
        let prescription = self.prescription();
        // The curve under the pointer in the chart, drawn thicker in the
        // views while the section is unfolded.
        if let Some(h) = self.dvh.as_ref().and_then(|d| d.hover) {
            self.focus_structures([(h.slot, h.item)]);
        }

        let current_gen = self.all_structures_gen();
        let mut recompute = false;
        let mut cancel = false;
        let mut export: Option<bool> = None; // Some(true) = curves, false = metrics
        let mut load_protocol = false;
        let mut save_protocol = false;
        let d = self.dvh.as_mut().expect("made above");

        // ---- pickers ----------------------------------------------------
        ui.horizontal_wrapped(|ui| {
            ui.label("Dose:");
            for (r, label) in &dose_list {
                let on = d.doses.contains(r);
                if ui.selectable_label(on, label).clicked() {
                    if on {
                        d.doses.retain(|x| x != r);
                    } else {
                        d.doses.push(*r);
                        d.row = r.slot;
                    }
                    recompute = true;
                }
            }
            if dose_list.is_empty() {
                ui.label(egui::RichText::new("no RTDOSE loaded").color(warn_color(ui.visuals())));
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Structures:");
            egui::ComboBox::from_id_salt("dvh_add_struct")
                .selected_text("add")
                .width(170.0)
                .show_ui(ui, |ui| {
                    for (r, label) in &struct_list {
                        if d.structures.contains(r) {
                            continue;
                        }
                        if ui.selectable_label(false, label).clicked() {
                            d.structures.push(*r);
                            recompute = true;
                        }
                    }
                });
            let mut drop = None;
            for (i, s) in d.structures.iter().enumerate() {
                let label = struct_list
                    .iter()
                    .find(|(r, _)| r == s)
                    .map(|(_, l)| l.clone())
                    .unwrap_or_else(|| "(gone)".into());
                if ui
                    .selectable_label(true, format!("{label} ✖"))
                    .on_hover_text("Remove from the plot")
                    .clicked()
                {
                    drop = Some(i);
                }
            }
            if let Some(i) = drop {
                d.structures.remove(i);
                recompute = true;
            }
            if !d.structures.is_empty() && ui.small_button("Clear").clicked() {
                d.structures.clear();
                recompute = true;
            }
        });

        // ---- axes ---------------------------------------------------------
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut d.cumulative, true, "Cumulative")
                .on_hover_text("Volume receiving at least each dose - the usual view");
            ui.selectable_value(&mut d.cumulative, false, "Differential")
                .on_hover_text("Volume in each dose bin - where the cold spots are");
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Dose:");
            ui.selectable_value(&mut d.dose_relative, false, "Gy");
            let has_ref = d.reference_dose > 0.0;
            let rel = ui
                .add_enabled_ui(has_ref, |ui| ui.selectable_label(d.dose_relative, "% of"))
                .inner;
            rel.clone().on_hover_text(if has_ref {
                "Per cent of the reference dose"
            } else {
                "No plan in this study declares a prescription - type one"
            });
            if rel.clicked() {
                d.dose_relative = true;
            }
            ui.add(
                egui::DragValue::new(&mut d.reference_dose)
                    .range(0.0..=1000.0)
                    .speed(0.1)
                    .suffix(" Gy"),
            );
            if let Some((dose, from)) = &prescription {
                if (d.reference_dose - *dose).abs() > 1e-6
                    && ui
                        .small_button("↺")
                        .on_hover_text(format!("Back to the prescription of '{from}'"))
                        .clicked()
                {
                    d.reference_dose = *dose;
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Volume:");
            ui.selectable_value(&mut d.volume_relative, true, "%");
            ui.selectable_value(&mut d.volume_relative, false, "cm³");
        });
        if let Some(p) = &progress {
            cancel = progress_row(ui, p);
        }

        // ---- the table ----------------------------------------------------
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label("Columns:");
            let mut drop = None;
            for (i, m) in d.metrics.iter().enumerate() {
                if ui
                    .selectable_label(true, format!("{} ✖", m.label()))
                    .clicked()
                {
                    drop = Some(i);
                }
            }
            if let Some(i) = drop {
                d.metrics.remove(i);
            }
            let resp = ui.add(
                egui::TextEdit::singleline(&mut d.new_metric)
                    .hint_text("D98%, V20Gy, D2cc")
                    .desired_width(110.0),
            );
            // Nothing typed, nothing to add: the button stays dead rather
            // than answering a click with an error.
            let named = !d.new_metric.trim().is_empty();
            let add = ui
                .add_enabled(named, egui::Button::new("➕").small())
                .on_disabled_hover_text("Type a metric first, for example D95%")
                .clicked()
                || (named && resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
            if add {
                match Metric::parse(&d.new_metric) {
                    Some(m) => {
                        if !d.metrics.contains(&m) {
                            d.metrics.push(m);
                        }
                        d.new_metric.clear();
                    }
                    None if !d.new_metric.trim().is_empty() => {
                        d.status = Some(format!(
                            "'{}' is not a metric - try D95%, D2cc, V20Gy or Dmean.",
                            d.new_metric.trim()
                        ));
                    }
                    None => {}
                }
            }
        });
        metrics_table(ui, d);

        // ---- constraints --------------------------------------------------
        ui.separator();
        let header = if d.constraints.is_empty() {
            "Constraints".to_string()
        } else {
            let v = dvh::check(&d.constraints, &d.curves);
            let failed = v.iter().filter(|x| !x.pass).count();
            format!("Constraints - {} of {} met", v.len() - failed, v.len())
        };
        egui::CollapsingHeader::new(header)
            .id_salt("dvh_constraints_fold")
            .default_open(d.show_constraints)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Load protocol").clicked() {
                        load_protocol = true;
                    }
                    if ui
                        .add_enabled(
                            !d.constraints.is_empty(),
                            egui::Button::new("Save protocol"),
                        )
                        .clicked()
                    {
                        save_protocol = true;
                    }
                    if !d.protocol_name.is_empty() {
                        ui.weak(&d.protocol_name);
                    }
                });
                if d.constraints.is_empty() {
                    ui.weak(
                        "A protocol is a text file, one constraint per line:  Cord Dmax <= 45   \
                         ·   PTV* D95% >= 57   ·   \"Parotid L\" Dmean <= 26",
                    );
                } else {
                    egui::ScrollArea::horizontal()
                        .id_salt("dvh_constraints_scroll")
                        .show(ui, |ui| constraint_table(ui, d));
                }
            });

        // ---- footer -------------------------------------------------------
        ui.separator();
        ui.horizontal_wrapped(|ui| {
            if ui
                .add_enabled(!d.curves.is_empty(), egui::Button::new("Export curves"))
                .clicked()
            {
                export = Some(true);
            }
            if ui
                .add_enabled(!d.curves.is_empty(), egui::Button::new("Export table"))
                .clicked()
            {
                export = Some(false);
            }
            if ui.button("Recompute").clicked() {
                recompute = true;
            }
        });
        if let Some(s) = &d.status {
            ui.weak(s);
        }
        if !d.curves.is_empty() && d.gen != current_gen {
            ui.label(
                egui::RichText::new(
                    "⚠ A structure has changed since these curves were computed - the numbers \
                     describe the geometry as it was. Recompute.",
                )
                .color(theme::warn_color(ui.visuals())),
            );
        }
        let truncated: Vec<&Dvh> = d
            .curves
            .iter()
            .filter(|c| c.outside_fraction() > 0.001)
            .collect();
        if !truncated.is_empty() {
            let names: Vec<String> = truncated
                .iter()
                .map(|c| format!("{} ({:.0} %)", c.name, c.outside_fraction() * 100.0))
                .collect();
            ui.label(
                egui::RichText::new(format!(
                    "⚠ Outside the dose grid, counted at zero dose: {}",
                    names.join(", ")
                ))
                .small()
                .color(warn_color(ui.visuals())),
            );
        }
        if recompute {
            // A change the user made: the chart goes into the row when the
            // curves land.
            d.show_chart = true;
        }

        cancel_if(cancel, &self.dvh_job);
        if load_protocol {
            self.ask_file(
                "Open a constraint protocol",
                None,
                Some(PROTOCOL_FILES),
                |app, path| match std::fs::read_to_string(&path) {
                    Ok(text) => {
                        let cs = dvh::parse_protocol(&text);
                        if let Some(d) = &mut app.dvh {
                            if cs.is_empty() {
                                app.error = Some(
                                    "No constraints were recognised in that file. Each line \
                                     is STRUCTURE METRIC <= LIMIT, for example \
                                     'Cord Dmax <= 45'."
                                        .into(),
                                );
                            } else {
                                d.constraints = cs;
                                d.protocol_name = path
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_default();
                                d.show_constraints = true;
                            }
                        }
                    }
                    Err(e) => app.error = Some(format!("Could not read the protocol: {e}")),
                },
            );
        }
        if save_protocol {
            let text = self
                .dvh
                .as_ref()
                .map(|d| dvh::write_protocol(&d.constraints))
                .unwrap_or_default();
            self.ask_save(
                "Save the protocol",
                "protocol.txt",
                None,
                None,
                move |app, path| {
                    if let Err(e) = std::fs::write(&path, text) {
                        app.error = Some(format!("Could not write the protocol: {e}"));
                    }
                },
            );
        }
        if let Some(curves) = export {
            let text = self.dvh.as_ref().map(|d| {
                if curves {
                    dvh::curves_csv(&d.curves, d.volume_relative)
                } else {
                    dvh::metrics_csv(&d.curves, &d.metrics)
                }
            });
            if let Some(text) = text {
                self.ask_save(
                    if curves {
                        "Save the DVH curves"
                    } else {
                        "Save the metrics table"
                    },
                    if curves { "dvh.csv" } else { "dvh_metrics.csv" },
                    None,
                    None,
                    move |app, path| match std::fs::write(&path, text) {
                        Ok(()) => app.notice = Some(format!("Written to {}", path.display())),
                        Err(e) => app.error = Some(format!("Could not write the file: {e}")),
                    },
                );
            }
        }
        if recompute && !running {
            self.start_dvh();
        }
    }
}

/// The metrics table: one row per curve.
fn metrics_table(ui: &mut egui::Ui, d: &DvhModule) {
    if d.curves.is_empty() {
        ui.weak("Pick a dose object and one or more structures.");
        return;
    }
    let units = dvh::nice_units(&d.curves[0].units);
    let several_doses = d.doses.len() > 1;
    egui::ScrollArea::horizontal()
        .id_salt("dvh_table")
        .max_height(180.0)
        .show(ui, |ui| {
            egui::Grid::new("dvh_metrics")
                .striped(true)
                .num_columns(d.metrics.len() + 2)
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("Structure").strong());
                    if several_doses {
                        ui.label(egui::RichText::new("Dose").strong());
                    }
                    for m in &d.metrics {
                        ui.label(
                            egui::RichText::new(format!(
                                "{} [{}]",
                                m.label(),
                                display_unit(m, &units, d)
                            ))
                            .strong(),
                        );
                    }
                    ui.end_row();
                    for c in &d.curves {
                        let col = theme::rgb(c.color);
                        ui.horizontal(|ui| {
                            let (rect, _) = ui
                                .allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                            ui.painter().rect_filled(rect, 2.0, col);
                            ui.label(&c.name);
                        });
                        if several_doses {
                            ui.label(&c.dose_label);
                        }
                        for m in &d.metrics {
                            let v = m.evaluate(c);
                            let v = if m.is_dose() && d.dose_relative && d.reference_dose > 0.0 {
                                v / d.reference_dose * 100.0
                            } else {
                                v
                            };
                            ui.label(format!("{v:.2}"));
                        }
                        ui.end_row();
                    }
                });
        });
}

fn display_unit(m: &Metric, units: &str, d: &DvhModule) -> String {
    if m.is_dose() && d.dose_relative && d.reference_dose > 0.0 {
        "%".into()
    } else {
        m.unit(units)
    }
}

/// The constraint table, with the pass/fail column.
fn constraint_table(ui: &mut egui::Ui, d: &DvhModule) {
    let verdicts = dvh::check(&d.constraints, &d.curves);
    egui::Grid::new("dvh_constraints")
        .striped(true)
        .num_columns(5)
        .show(ui, |ui| {
            for h in ["", "Structure", "Metric", "Limit", "Value"] {
                ui.label(egui::RichText::new(h).strong());
            }
            ui.end_row();
            for v in &verdicts {
                let (mark, color) = match (v.value.is_some(), v.pass) {
                    (false, _) => ("-", warn_color(ui.visuals())),
                    (true, true) => ("✔", egui::Color32::from_rgb(60, 160, 80)),
                    (true, false) => ("✖", egui::Color32::from_rgb(200, 70, 70)),
                };
                ui.label(egui::RichText::new(mark).color(color).strong());
                ui.label(if v.structure.is_empty() {
                    format!("{} (not found)", v.constraint.structure)
                } else {
                    v.structure.clone()
                });
                ui.label(v.constraint.metric.label());
                ui.label(format!(
                    "{} {:.2}",
                    v.constraint.cmp.symbol(),
                    v.constraint.limit
                ));
                match v.value {
                    Some(x) => ui.label(egui::RichText::new(format!("{x:.2}")).color(color)),
                    None => ui.label("-"),
                };
                ui.end_row();
            }
        });
}

/// Axis limits and the mapping onto the panel.
struct Axes {
    x_max: f64,
    y_max: f64,
    rect: egui::Rect,
}

impl Axes {
    fn at(&self, x: f64, y: f64) -> egui::Pos2 {
        let fx = (x / self.x_max).clamp(0.0, 1.0) as f32;
        let fy = (y / self.y_max).clamp(0.0, 1.0) as f32;
        egui::pos2(
            self.rect.left() + fx * self.rect.width(),
            self.rect.bottom() - fy * self.rect.height(),
        )
    }
    fn dose_at(&self, px: f32) -> f64 {
        ((px - self.rect.left()) / self.rect.width().max(1.0)) as f64 * self.x_max
    }
}

/// "Nice" tick step: 1, 2 or 5 times a power of ten.
fn tick_step(span: f64, target: usize) -> f64 {
    if span <= 0.0 {
        return 1.0;
    }
    let raw = span / target.max(1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let n = raw / mag;
    mag * if n < 1.5 {
        1.0
    } else if n < 3.5 {
        2.0
    } else if n < 7.5 {
        5.0
    } else {
        10.0
    }
}

/// The plot itself: axes, gridlines, curves, legend and a hover readout.
/// Returns the curve under the pointer, by its index in `d.curves`.
fn plot(ui: &mut egui::Ui, d: &DvhModule, height: f32) -> Option<usize> {
    let width = ui.available_width();
    let (outer, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(outer);
    let vis = ui.visuals();
    painter.rect_filled(outer, 2.0, vis.extreme_bg_color);
    if d.curves.is_empty() {
        painter.text(
            outer.center(),
            egui::Align2::CENTER_CENTER,
            "Pick a dose object and one or more structures.",
            egui::FontId::proportional(13.0),
            vis.weak_text_color(),
        );
        return None;
    }

    let scale = if d.dose_relative && d.reference_dose > 0.0 {
        100.0 / d.reference_dose
    } else {
        1.0
    };
    let units = if d.dose_relative && d.reference_dose > 0.0 {
        "%".to_string()
    } else {
        dvh::nice_units(&d.curves[0].units)
    };
    // Room for the axis labels; the legend sits inside the panel.
    // Left margin holds the turned-on-its-side volume caption and the tick
    // numbers beside it, so it has to be wider than the numbers alone.
    let rect = egui::Rect::from_min_max(
        outer.min + egui::vec2(60.0, 10.0),
        outer.max - egui::vec2(12.0, 30.0),
    );
    let x_max = d
        .curves
        .iter()
        .map(|c| c.dose_extent() * scale)
        .fold(0.0, f64::max)
        .max(1e-6)
        * 1.02;
    let y_max = if d.cumulative {
        if d.volume_relative {
            100.0
        } else {
            d.curves.iter().map(|c| c.volume_cm3).fold(0.0, f64::max) * 1.05
        }
    } else {
        // Differential: each curve's tallest bin, normalised the same way
        // the curve itself will be drawn.
        d.curves
            .iter()
            .map(|c| {
                let peak = c.bins.iter().cloned().fold(0.0, f64::max);
                if d.volume_relative && c.volume_cm3 > 0.0 {
                    peak / c.volume_cm3 * 100.0
                } else {
                    peak
                }
            })
            .fold(0.0, f64::max)
            * 1.1
    }
    .max(1e-9);
    let ax = Axes { x_max, y_max, rect };

    // ---- grid and ticks ----
    let grid_stroke = egui::Stroke::new(1.0, vis.weak_text_color().gamma_multiply(0.25));
    let axis_stroke = egui::Stroke::new(1.0, vis.text_color().gamma_multiply(0.6));
    let font = egui::FontId::proportional(10.0);
    let xs = tick_step(x_max, 8);
    let mut t = 0.0;
    while t <= x_max + 1e-9 {
        let p = ax.at(t, 0.0);
        painter.line_segment(
            [egui::pos2(p.x, rect.top()), egui::pos2(p.x, rect.bottom())],
            grid_stroke,
        );
        painter.text(
            egui::pos2(p.x, rect.bottom() + 3.0),
            egui::Align2::CENTER_TOP,
            format!("{t:.0}"),
            font.clone(),
            vis.text_color(),
        );
        t += xs;
    }
    let ys = tick_step(y_max, 6);
    let mut t = 0.0;
    while t <= y_max + 1e-9 {
        let p = ax.at(0.0, t);
        painter.line_segment(
            [egui::pos2(rect.left(), p.y), egui::pos2(rect.right(), p.y)],
            grid_stroke,
        );
        painter.text(
            egui::pos2(rect.left() - 5.0, p.y),
            egui::Align2::RIGHT_CENTER,
            if ys < 1.0 {
                format!("{t:.2}")
            } else {
                format!("{t:.0}")
            },
            font.clone(),
            vis.text_color(),
        );
        t += ys;
    }
    painter.rect_stroke(rect, 0.0, axis_stroke, egui::StrokeKind::Inside);
    painter.text(
        egui::pos2(rect.center().x, outer.bottom() - 2.0),
        egui::Align2::CENTER_BOTTOM,
        format!("Dose [{units}]"),
        font.clone(),
        vis.text_color(),
    );
    // Turned on its side and centred on the axis. Drawn flat in the corner it
    // ran straight through the topmost tick number.
    let caption = if d.volume_relative {
        "Volume [%]".to_string()
    } else {
        "Volume [cm³]".to_string()
    };
    let galley = painter.layout_no_wrap(caption, font.clone(), vis.text_color());
    let anchor = egui::pos2(
        outer.left() + 2.0,
        (rect.center().y + galley.size().x / 2.0).min(rect.bottom()),
    );
    painter.add(
        egui::epaint::TextShape::new(anchor, galley, vis.text_color())
            .with_angle(-std::f32::consts::FRAC_PI_2),
    );

    // ---- the curves ----
    // Structures keep their colour; the dose object picks the line style, so
    // two plans over the same structures read as one colour in two dashes.
    let dose_order: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for c in &d.curves {
            if !v.contains(&c.dose_label) {
                v.push(c.dose_label.clone());
            }
        }
        v
    };
    for c in &d.curves {
        let col = theme::rgb(c.color);
        let stroke = egui::Stroke::new(1.6, col);
        let pts: Vec<egui::Pos2> = if d.cumulative {
            c.cumulative()
                .into_iter()
                .map(|(dose, vol)| {
                    let y = if d.volume_relative {
                        if c.volume_cm3 > 0.0 {
                            vol / c.volume_cm3 * 100.0
                        } else {
                            0.0
                        }
                    } else {
                        vol
                    };
                    ax.at(dose * scale, y)
                })
                .collect()
        } else {
            c.differential()
                .into_iter()
                .map(|(dose, vol)| {
                    let y = if d.volume_relative && c.volume_cm3 > 0.0 {
                        vol / c.volume_cm3 * 100.0
                    } else {
                        vol
                    };
                    ax.at(dose * scale, y)
                })
                .collect()
        };
        let style = dose_order
            .iter()
            .position(|l| *l == c.dose_label)
            .unwrap_or(0);
        match style {
            0 => painter.add(egui::Shape::line(pts, stroke)),
            1 => painter.add(egui::Shape::Vec(egui::Shape::dashed_line(
                &pts, stroke, 6.0, 4.0,
            ))),
            _ => painter.add(egui::Shape::Vec(egui::Shape::dashed_line(
                &pts, stroke, 2.0, 3.0,
            ))),
        };
    }

    // ---- legend ----
    let mut y = rect.top() + 4.0;
    for c in &d.curves {
        let col = theme::rgb(c.color);
        let sw = egui::Rect::from_min_size(
            egui::pos2(rect.right() - 150.0, y + 3.0),
            egui::vec2(14.0, 3.0),
        );
        painter.rect_filled(sw, 0.0, col);
        let label = if dose_order.len() > 1 {
            format!("{} · {}", c.name, c.dose_label)
        } else {
            c.name.clone()
        };
        painter.text(
            egui::pos2(rect.right() - 132.0, y),
            egui::Align2::LEFT_TOP,
            label,
            font.clone(),
            vis.text_color(),
        );
        y += 13.0;
        if y > rect.bottom() - 12.0 {
            break;
        }
    }

    // ---- hover readout ----
    let mut nearest: Option<(usize, f32)> = None;
    if let Some(pos) = response.hover_pos() {
        if rect.contains(pos) {
            let dose = ax.dose_at(pos.x);
            painter.line_segment(
                [
                    egui::pos2(pos.x, rect.top()),
                    egui::pos2(pos.x, rect.bottom()),
                ],
                egui::Stroke::new(1.0, vis.text_color().gamma_multiply(0.5)),
            );
            let real = dose / scale;
            let mut lines = vec![format!("{dose:.1} {units}")];
            // The curve under the pointer: the one nearest it at this dose,
            // within a few points. Cumulative curves only - a differential
            // one has no single height to be near.
            if d.cumulative {
                for (i, c) in d.curves.iter().enumerate() {
                    let y = if d.volume_relative {
                        c.volume_fraction_at_dose(real) * 100.0
                    } else {
                        c.volume_at_dose(real)
                    };
                    let dist = (ax.at(dose, y).y - pos.y).abs();
                    if dist < 8.0 && nearest.is_none_or(|(_, best)| dist < best) {
                        nearest = Some((i, dist));
                    }
                }
            }
            for c in &d.curves {
                let v = if d.volume_relative {
                    format!("{:.1} %", c.volume_fraction_at_dose(real) * 100.0)
                } else {
                    format!("{:.2} cm³", c.volume_at_dose(real))
                };
                lines.push(format!("{}: {v}", c.name));
            }
            response.clone().on_hover_text(lines.join("\n"));
        }
    }
    nearest.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_steps_are_readable_numbers() {
        assert_eq!(tick_step(70.0, 7), 10.0);
        assert_eq!(tick_step(100.0, 5), 20.0);
        assert_eq!(tick_step(1.0, 5), 0.2);
        assert_eq!(tick_step(0.0, 5), 1.0);
    }

    #[test]
    fn the_axes_map_the_corners_onto_the_panel() {
        let rect = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(100.0, 50.0));
        let ax = Axes {
            x_max: 10.0,
            y_max: 100.0,
            rect,
        };
        // Volume 100 % at dose 0 is the top left; dose 10 at 0 % the bottom right.
        assert_eq!(ax.at(0.0, 100.0), egui::pos2(0.0, 0.0));
        assert_eq!(ax.at(10.0, 0.0), egui::pos2(100.0, 50.0));
        // …and reading a position back gives the dose again.
        assert!((ax.dose_at(50.0) - 5.0).abs() < 1e-6);
    }
}
