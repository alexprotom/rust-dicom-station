//! *Modules ▶ Dose estimation*: one table of dose metrics for every ticked
//! structure of the active set against one dose, recomputed by itself
//! whenever the structures, the tick boxes, the dose or the columns change.
//!
//! The DVH window is for looking at curves and checking a protocol; this
//! section is what the Structure editor's move / rotate buttons are used
//! with - shift a chamber volume, read the new mean dose.

use std::hash::{Hash, Hasher};

use crate::dvh::{self, DvhParams, Metric};
use crate::rtdose::DoseGrid;
use crate::segmentation;

use super::widgets::small_tip_button;
use super::*;

/// The columns every table has, in this order.
const FIXED: [Metric; 4] = [Metric::Volume, Metric::Mean, Metric::Min, Metric::Max];

/// Physical or RBE-weighted dose, the Dose Type of an RTDOSE.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum DoseKind {
    Physical,
    Effective,
}

impl DoseKind {
    /// What an RTDOSE's Dose Type means here; an empty one is physical.
    pub(super) fn of(d: &DoseGrid) -> DoseKind {
        if d.dose_type.eq_ignore_ascii_case("EFFECTIVE") {
            DoseKind::Effective
        } else {
            DoseKind::Physical
        }
    }
}

pub(super) struct DoseEst {
    pub slot: usize,
    /// Which of the study's doses the table is against; `None` follows the
    /// dataset's active dose until a dose is picked here.
    pub dose: Option<usize>,
    pub kind: DoseKind,
    /// Columns after the fixed four: `D95%`, `V20`, `D2cc`, …
    pub extra: Vec<Metric>,
    pub new_metric: String,
    pub bad_metric: Option<String>,
    pub rows: Vec<DoseRow>,
    /// What `rows` were computed for (see [`ViewerApp::dose_est_key`]).
    pub key: Option<u64>,
    /// The key of the run in flight, so a change during it is noticed.
    pub running: Option<u64>,
    pub units: String,
    pub dose_label: String,
}

impl Default for DoseEst {
    fn default() -> Self {
        DoseEst {
            slot: 0,
            dose: None,
            kind: DoseKind::Physical,
            extra: vec![Metric::DoseAtPct(95.0), Metric::DoseAtPct(2.0)],
            new_metric: String::new(),
            bad_metric: None,
            rows: Vec::new(),
            key: None,
            running: None,
            units: "GY".into(),
            dose_label: String::new(),
        }
    }
}

impl DoseEst {
    pub(super) fn metrics(&self) -> Vec<Metric> {
        FIXED
            .iter()
            .copied()
            .chain(self.extra.iter().copied())
            .collect()
    }

    /// Add the column typed into the box. Anything that does not read as a
    /// metric, or that asks for a volume outside 0‥100 %, is refused with
    /// a note under the box; a column already there is not added twice.
    fn add_metric(&mut self) {
        let text = self.new_metric.trim().to_string();
        if text.is_empty() {
            return;
        }
        let m = match Metric::parse(&text) {
            Some(m) => m,
            None => {
                self.bad_metric = Some(format!(
                    "\"{text}\" is not a metric: try D95%, D2cc, V20 or V20cc"
                ));
                return;
            }
        };
        let ok = match m {
            Metric::DoseAtPct(p) => p.is_finite() && (0.0..=100.0).contains(&p),
            Metric::DoseAtCc(v) | Metric::VolumePctAtDose(v) | Metric::VolumeCcAtDose(v) => {
                v.is_finite() && v >= 0.0
            }
            _ => true,
        };
        if !ok {
            self.bad_metric = Some(format!(
                "\"{text}\": the number must be finite and not negative (a percentage at most 100)"
            ));
            return;
        }
        if !FIXED.contains(&m) && !self.extra.contains(&m) {
            self.extra.push(m);
        }
        self.new_metric.clear();
        self.bad_metric = None;
    }
}

/// One line of the table.
pub(super) struct DoseRow {
    pub name: String,
    pub color: [u8; 3],
    /// One value per metric, `None` where there is nothing to report.
    pub values: Vec<Option<f64>>,
    /// Fraction of the structure outside the dose grid, 0‥1.
    pub outside: f64,
}

/// What one run of the table is given: everything is owned, the worker
/// thread must not borrow the app.
struct Request {
    grid: crate::volume::Grid,
    rois: Vec<crate::rtstruct::Roi>,
    dose: DoseGrid,
    metrics: Vec<Metric>,
}

fn compute_rows(req: &Request) -> Vec<DoseRow> {
    req.rois
        .par_iter()
        .map(|roi| {
            let n = req.metrics.len();
            let mask = segmentation::rasterize_roi(&req.grid, roi);
            let dvh = mask.and_then(|m| {
                dvh::compute(
                    &roi.name,
                    roi.color,
                    &m,
                    &req.grid,
                    &req.dose,
                    DvhParams::default(),
                )
                .ok()
            });
            match dvh {
                Some(d) => DoseRow {
                    name: roi.name.clone(),
                    color: roi.color,
                    values: req
                        .metrics
                        .iter()
                        .map(|m| Some(m.evaluate(&d)).filter(|v| v.is_finite()))
                        .collect(),
                    outside: d.outside_fraction(),
                },
                None => DoseRow {
                    name: roi.name.clone(),
                    color: roi.color,
                    values: vec![None; n],
                    outside: 0.0,
                },
            }
        })
        .collect()
}

impl ViewerApp {
    /// The doses of the section's dataset that are of the chosen kind:
    /// index into `study.doses` and a label.
    fn dose_est_candidates(&self, slot: usize) -> Vec<(usize, String)> {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return Vec::new();
        };
        let both = self.dose_kinds(slot) == (true, true);
        study
            .doses
            .iter()
            .enumerate()
            .filter(|(_, d)| !both || DoseKind::of(d) == self.dose_est.kind)
            .map(|(i, d)| {
                let label = if d.label.is_empty() {
                    format!("Dose {}", i + 1)
                } else {
                    d.label.clone()
                };
                (i, label)
            })
            .collect()
    }

    /// Whether the dataset has physical and effective doses.
    fn dose_kinds(&self, slot: usize) -> (bool, bool) {
        let mut kinds = (false, false);
        if let Some(study) = self.slots[slot].study.as_ref() {
            for d in &study.doses {
                match DoseKind::of(d) {
                    DoseKind::Physical => kinds.0 = true,
                    DoseKind::Effective => kinds.1 = true,
                }
            }
        }
        kinds
    }

    /// The dose the table is against, after every fallback: the picked one
    /// if it is still there and of the chosen kind, else the active dose
    /// if it is, else the first of the kind.
    fn dose_est_dose(&self, slot: usize) -> Option<usize> {
        let candidates = self.dose_est_candidates(slot);
        let is = |i: usize| candidates.iter().any(|(c, _)| *c == i);
        if let Some(i) = self.dose_est.dose.filter(|i| is(*i)) {
            return Some(i);
        }
        let active = self.slots[slot].active_dose;
        if is(active) {
            return Some(active);
        }
        candidates.first().map(|(i, _)| *i)
    }

    /// Everything the table depends on, hashed: the dataset, the set, the
    /// tick boxes, the geometry of every ticked structure, the dose and
    /// the columns. A different key means the table is stale.
    fn dose_est_key(&self, slot: usize, dose: Option<usize>) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        slot.hash(&mut h);
        dose.hash(&mut h);
        self.settings_gen.hash(&mut h);
        for m in self.dose_est.metrics() {
            m.label().hash(&mut h);
        }
        let s = &self.slots[slot];
        s.active_structs.hash(&mut h);
        if let Some(study) = s.study.as_ref() {
            study.volume.dims.hash(&mut h);
            if let Some(ss) = study.structure_sets.get(s.active_structs) {
                for (i, roi) in ss.rois.iter().enumerate() {
                    if !s.roi_visible.get(i).copied().unwrap_or(true) {
                        continue;
                    }
                    roi.name.hash(&mut h);
                    roi.contours.len().hash(&mut h);
                    for c in &roi.contours {
                        for p in &c.points {
                            (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()).hash(&mut h);
                        }
                    }
                }
            }
        }
        h.finish()
    }

    fn start_dose_est(&mut self, slot: usize, dose_idx: usize, key: u64) {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(dose) = study.doses.get(dose_idx) else {
            return;
        };
        let s = &self.slots[slot];
        let rois: Vec<_> = study
            .structure_sets
            .get(s.active_structs)
            .map(|ss| {
                ss.rois
                    .iter()
                    .enumerate()
                    .filter(|(i, r)| s.roi_visible.get(*i).copied().unwrap_or(true) && !r.is_poi())
                    .map(|(_, r)| r.clone())
                    .collect()
            })
            .unwrap_or_default();
        let req = Request {
            grid: study.volume.grid(),
            rois,
            dose: dose.clone(),
            metrics: self.dose_est.metrics(),
        };
        self.dose_est.units = dose.units.clone();
        self.dose_est.dose_label = dose.label.clone();
        self.dose_est.running = Some(key);
        let progress = Arc::new(Progress::default());
        self.dose_est_job = Some(Job::spawn(progress, move |_| compute_rows(&req)));
    }

    pub(super) fn on_dose_est_done(&mut self, rows: Vec<DoseRow>) {
        self.dose_est.rows = rows;
        self.dose_est.key = self.dose_est.running.take();
    }

    pub(super) fn dose_est_section(&mut self, ui: &mut egui::Ui) {
        let title = egui::RichText::new("Dose estimation").strong();
        egui::CollapsingHeader::new(title)
            .default_open(true)
            .show(ui, |ui| self.dose_est_body(ui));
        ui.separator();
    }

    fn dose_est_body(&mut self, ui: &mut egui::Ui) {
        if !self.any_volume() {
            ui.weak("Load a dataset with an image volume and a dose");
            return;
        }
        if !self.slots[self.dose_est.slot].has_volume() {
            self.dose_est.slot = self.first_volume_slot();
        }
        if let Some(s) = seg_engines::dataset_row(ui, self.dose_est.slot, self.volume_slots(), true)
        {
            self.dose_est.slot = s;
            self.dose_est.dose = None;
        }
        let slot = self.dose_est.slot;
        let kinds = self.dose_kinds(slot);
        if kinds == (true, true) {
            ui.horizontal_wrapped(|ui| {
                ui.label("Dose:");
                for (k, name, tip) in [
                    (DoseKind::Physical, "Physical", "Dose Type PHYSICAL"),
                    (
                        DoseKind::Effective,
                        "Effective",
                        "Dose Type EFFECTIVE: RBE-weighted",
                    ),
                ] {
                    if ui
                        .selectable_label(self.dose_est.kind == k, name)
                        .on_hover_text(tip)
                        .clicked()
                    {
                        self.dose_est.kind = k;
                        self.dose_est.dose = None;
                    }
                }
            });
        }
        let candidates = self.dose_est_candidates(slot);
        if candidates.is_empty() {
            ui.weak("The dataset has no dose");
            self.dose_est.rows.clear();
            self.dose_est.key = None;
            return;
        }
        let dose = self.dose_est_dose(slot);
        ui.horizontal_wrapped(|ui| {
            let current = candidates
                .iter()
                .find(|(i, _)| Some(*i) == dose)
                .map(|(_, l)| l.as_str())
                .unwrap_or("(none)");
            egui::ComboBox::from_id_salt("dose_est_dose")
                .width(260.0)
                .selected_text(current)
                .show_ui(ui, |ui| {
                    for (i, label) in &candidates {
                        if ui.selectable_label(dose == Some(*i), label).clicked() {
                            self.dose_est.dose = Some(*i);
                        }
                    }
                });
        });
        // Columns: the fixed four, the extra ones each with a remover, and
        // the box that adds one.
        ui.horizontal_wrapped(|ui| {
            ui.label("Columns:");
            let mut drop = None;
            for (i, m) in self.dose_est.extra.iter().enumerate() {
                if ui
                    .small_button(format!("{} ✖", m.label()))
                    .on_hover_text("Remove this column")
                    .clicked()
                {
                    drop = Some(i);
                }
            }
            if let Some(i) = drop {
                self.dose_est.extra.remove(i);
            }
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.dose_est.new_metric)
                    .desired_width(70.0)
                    .hint_text("D95% / V20"),
            );
            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if small_tip_button(
                ui,
                "+",
                "Add a column: D<X>% (dose to X % of the volume), D<X>cc, V<X> (per cent \
                 of the volume at X dose units or more), V<X>cc",
            ) || enter
            {
                self.dose_est.add_metric();
            }
        });
        if let Some(bad) = &self.dose_est.bad_metric {
            ui.label(
                egui::RichText::new(bad)
                    .small()
                    .color(theme::rgb([230, 120, 90])),
            );
        }

        // Stale? Start a run - one at a time; a change during a run is
        // picked up when it lands, because the key differs again.
        let dose = self.dose_est_dose(slot);
        let key = self.dose_est_key(slot, dose);
        if self.dose_est.key != Some(key) && self.dose_est_job.is_none() {
            match dose {
                Some(d) => self.start_dose_est(slot, d, key),
                None => {
                    self.dose_est.rows.clear();
                    self.dose_est.key = Some(key);
                }
            }
        }

        let metrics = self.dose_est.metrics();
        let units = dvh::nice_units(&self.dose_est.units);
        let d = &self.dose_est;
        if d.rows.is_empty() {
            if self.dose_est_job.is_some() {
                ui.weak("Computing…");
            } else {
                ui.weak("Tick structures in the list on the left to see their dose");
            }
            return;
        }
        egui::ScrollArea::horizontal()
            .id_salt("dose_est_scroll")
            .show(ui, |ui| {
                egui::Grid::new("dose_est_grid")
                    .striped(true)
                    .min_col_width(44.0)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("Structure").strong());
                        for m in &metrics {
                            ui.label(egui::RichText::new(m.label()).strong())
                                .on_hover_text(format!("in {}", m.unit(&d.units)));
                        }
                        ui.end_row();
                        for row in &d.rows {
                            ui.horizontal(|ui| {
                                let (r, _) = ui.allocate_exact_size(
                                    egui::vec2(10.0, 10.0),
                                    egui::Sense::hover(),
                                );
                                ui.painter().rect_filled(r, 2.0, theme::rgb(row.color));
                                let name = ui.label(&row.name);
                                if row.outside > 0.001 {
                                    name.on_hover_text(format!(
                                        "{:.1} % of the structure lies outside the dose grid \
                                         and counts as zero dose",
                                        row.outside * 100.0
                                    ));
                                }
                            });
                            for (m, v) in metrics.iter().zip(&row.values) {
                                match v {
                                    Some(v) => {
                                        let text = match m {
                                            Metric::Volume | Metric::VolumeCcAtDose(_) => {
                                                format!("{v:.3}")
                                            }
                                            Metric::VolumePctAtDose(_) => format!("{v:.1}"),
                                            _ => format!("{v:.3}"),
                                        };
                                        ui.monospace(text);
                                    }
                                    None => {
                                        ui.weak("-");
                                    }
                                }
                            }
                            ui.end_row();
                        }
                    });
            });
        ui.label(
            egui::RichText::new(format!(
                "Doses in {units}, volumes in cm³; against {}. Follows every edit of the \
                 ticked structures.",
                if d.dose_label.is_empty() {
                    "the dose"
                } else {
                    &d.dose_label
                }
            ))
            .weak()
            .small(),
        );
        if self.dose_est_job.is_some() {
            ui.ctx().request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_column_box_refuses_nonsense_and_never_panics() {
        let mut d = DoseEst::default();
        let n = d.extra.len();
        for bad in [
            "", "   ", "D", "x", "Dnan%", "Dinf%", "D150%", "V-5", "D-1cc", "%",
        ] {
            d.new_metric = bad.into();
            d.add_metric();
            assert_eq!(d.extra.len(), n, "{bad:?} must not add a column");
        }
        assert!(d.bad_metric.is_some());
        for (good, m) in [
            ("d50", Metric::DoseAtPct(50.0)),
            ("D0.5cc", Metric::DoseAtCc(0.5)),
            ("V20", Metric::VolumePctAtDose(20.0)),
            ("v20cc", Metric::VolumeCcAtDose(20.0)),
        ] {
            d.new_metric = good.into();
            d.add_metric();
            assert_eq!(d.extra.last(), Some(&m), "{good}");
            assert!(d.new_metric.is_empty() && d.bad_metric.is_none());
        }
        // Not twice, and not one of the fixed columns.
        for again in ["D50%", "Dmean", "Volume"] {
            d.new_metric = again.into();
            d.add_metric();
        }
        assert_eq!(d.extra.len(), n + 4);
        assert_eq!(d.metrics().len(), FIXED.len() + n + 4);
    }
}
