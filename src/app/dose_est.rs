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

/// The columns a fresh table has, in this order; *Reset columns* brings
/// them back.
const DEFAULT_COLUMNS: [Metric; 6] = [
    Metric::Volume,
    Metric::Mean,
    Metric::Min,
    Metric::Max,
    Metric::DoseAtPct(95.0),
    Metric::DoseAtPct(2.0),
];

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
    /// The columns, every one of them removable: `Dmean`, `D95%`, `V20`, …
    pub columns: Vec<Metric>,
    pub new_metric: String,
    pub bad_metric: Option<String>,
    pub rows: Vec<DoseRow>,
    /// What `rows` were computed for (see [`ViewerApp::dose_est_key`]).
    pub key: Option<u64>,
    /// The key of the run in flight, so a change during it is noticed.
    pub running: Option<u64>,
    pub units: String,
    pub dose_label: String,
    /// *Dynamic*: every finished move of a structure adds the table as it
    /// then stands to `log`, with what was moved and by how much.
    pub dynamic: bool,
    pub log: Vec<LogEntry>,
    /// The move sequence the run in flight started at, and the last one
    /// logged - one entry per finished move, none for a drag in progress.
    pub run_seq: u64,
    pub logged_seq: Option<u64>,
    /// Structures left out of the table by name (the ✖ on a row), for the
    /// case where a ticked structure is wanted in the views but not here.
    pub excluded: Vec<String>,
}

/// One state of the table in the dynamic log.
pub(super) struct LogEntry {
    pub step: usize,
    /// The structure moved last (empty for the initial state).
    pub moved: String,
    pub summary: super::struct_tools::MoveSummary,
    pub rows: Vec<DoseRow>,
}

impl Default for DoseEst {
    fn default() -> Self {
        DoseEst {
            slot: 0,
            dose: None,
            kind: DoseKind::Physical,
            columns: DEFAULT_COLUMNS.to_vec(),
            new_metric: String::new(),
            bad_metric: None,
            rows: Vec::new(),
            key: None,
            running: None,
            units: "GY".into(),
            dose_label: String::new(),
            dynamic: false,
            log: Vec::new(),
            run_seq: 0,
            logged_seq: None,
            excluded: Vec::new(),
        }
    }
}

impl DoseEst {
    pub(super) fn metrics(&self) -> Vec<Metric> {
        self.columns.clone()
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
        if !self.columns.contains(&m) {
            self.columns.push(m);
        }
        self.new_metric.clear();
        self.bad_metric = None;
    }
}

/// The table as CSV: one header, one line per structure, the dose grid
/// coverage as a last column.
pub(super) fn rows_csv(metrics: &[Metric], units: &str, rows: &[DoseRow]) -> String {
    let mut out = String::from("structure");
    for m in metrics {
        out.push_str(&format!(",{}", column_title(m, units)));
    }
    out.push_str(",outside_dose_grid_pct\n");
    for r in rows {
        out.push_str(&row_csv(r));
        out.push('\n');
    }
    out
}

/// The dynamic log as CSV: the step and the move columns in front of the
/// same values.
pub(super) fn log_csv(metrics: &[Metric], units: &str, log: &[LogEntry]) -> String {
    let mut out =
        String::from("step,moved_structure,relative_to,shift_mm,rotation_deg,scale_pct,structure");
    for m in metrics {
        out.push_str(&format!(",{}", column_title(m, units)));
    }
    out.push_str(",outside_dose_grid_pct\n");
    for e in log {
        for r in &e.rows {
            out.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                e.step,
                csv_field(&e.moved),
                csv_field(&e.summary.relative),
                csv_field(&e.summary.shift),
                csv_field(&e.summary.rotation),
                csv_field(&e.summary.scale),
                row_csv(r)
            ));
        }
    }
    out
}

fn row_csv(r: &DoseRow) -> String {
    let mut out = csv_field(&r.name);
    for v in &r.values {
        match v {
            Some(v) => out.push_str(&format!(",{v:.4}")),
            None => out.push(','),
        }
    }
    out.push_str(&format!(",{:.2}", r.outside * 100.0));
    out
}

/// `Volume [cm³]`, `Dmean [Gy]`: the column heading.
fn column_title(m: &Metric, units: &str) -> String {
    format!("{} [{}]", m.label(), m.unit(units))
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
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
                    if !s.roi_visible.get(i).copied().unwrap_or(true)
                        || self.dose_est.excluded.contains(&roi.name)
                    {
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
                    .filter(|(i, r)| {
                        s.roi_visible.get(*i).copied().unwrap_or(true)
                            && !r.is_poi()
                            && !self.dose_est.excluded.contains(&r.name)
                    })
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
        self.dose_est.run_seq = self.tools.move_seq;
        let progress = Arc::new(Progress::default());
        self.dose_est_job = Some(Job::spawn(progress, move |_| compute_rows(&req)));
    }

    pub(super) fn on_dose_est_done(&mut self, rows: Vec<DoseRow>) {
        self.dose_est.rows = rows;
        self.dose_est.key = self.dose_est.running.take();
        // One log entry per finished move: the run has to have started
        // after the move, and no drag may be under way.
        let seq = self.tools.move_seq;
        if self.dose_est.dynamic
            && self.dose_est.run_seq == seq
            && self.dose_est.logged_seq != Some(seq)
        {
            self.log_dose_state();
        }
    }

    /// Append the table as it stands to the dynamic log, described by the
    /// last move.
    fn log_dose_state(&mut self) {
        let last = self.tools.moves.last().cloned();
        let (moved, summary) = match last {
            Some(e) => (e.name.clone(), self.move_summary(e.slot, e.set, e.roi)),
            None => (String::new(), Default::default()),
        };
        let d = &mut self.dose_est;
        let rows = d
            .rows
            .iter()
            .map(|r| DoseRow {
                name: r.name.clone(),
                color: r.color,
                values: r.values.clone(),
                outside: r.outside,
            })
            .collect();
        d.log.push(LogEntry {
            step: d.log.len(),
            moved,
            summary,
            rows,
        });
        d.logged_seq = Some(self.tools.move_seq);
    }

    pub(super) fn dose_est_section(&mut self, ui: &mut egui::Ui) {
        let id = ui.make_persistent_id("Dose estimation");
        let state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        let header = state.show_header(ui, |ui| {
            ui.label(egui::RichText::new("Dose estimation").strong());
            let was = self.dose_est.dynamic;
            ui.toggle_value(&mut self.dose_est.dynamic, "Dynamic")
                .on_hover_text(
                    "Log the table after every move of a structure: each finished Move / \
                     Rotate / Scale / hand drag adds the rows as they then stand, with \
                     what was moved and by how much since its origin",
                );
            if self.dose_est.dynamic && !was {
                // The state the moves start from is the log's first entry.
                self.dose_est.log.clear();
                if !self.dose_est.rows.is_empty() {
                    self.log_dose_state();
                }
            }
            if self.dose_est.dynamic
                && !self.dose_est.log.is_empty()
                && small_tip_button(
                    ui,
                    "Clear log",
                    "Start the log again from the table as it is",
                )
            {
                self.dose_est.log.clear();
                if !self.dose_est.rows.is_empty() {
                    self.log_dose_state();
                }
            }
        });
        header.body(|ui| self.dose_est_body(ui));
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
        // Columns: each with a remover, the box that adds one, the reset.
        ui.horizontal_wrapped(|ui| {
            ui.label("Columns:");
            let mut drop = None;
            for (i, m) in self.dose_est.columns.iter().enumerate() {
                if ui
                    .small_button(format!("{} ✖", m.label()))
                    .on_hover_text("Remove this column")
                    .clicked()
                {
                    drop = Some(i);
                }
            }
            if let Some(i) = drop {
                self.dose_est.columns.remove(i);
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
                "Add a column: Dmean, Dmin, Dmax, Volume, D<X>% (dose to X % of the volume), \
                 D<X>cc, V<X> (per cent of the volume at X dose units or more), V<X>cc",
            ) || enter
            {
                self.dose_est.add_metric();
            }
            if self.dose_est.columns != DEFAULT_COLUMNS
                && small_tip_button(ui, "Reset columns", "Volume, Dmean, Dmin, Dmax, D95%, D2%")
            {
                self.dose_est.columns = DEFAULT_COLUMNS.to_vec();
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
        let d = &self.dose_est;
        if d.rows.is_empty() {
            if self.dose_est_job.is_some() {
                ui.weak("Computing…");
            } else {
                ui.weak("Tick structures in the list on the left to see their dose");
            }
            return;
        }
        let dynamic = d.dynamic;
        let mut exclude: Option<String> = None;
        egui::ScrollArea::horizontal()
            .id_salt("dose_est_scroll")
            .show(ui, |ui| {
                egui::Grid::new("dose_est_grid")
                    .striped(true)
                    .min_col_width(44.0)
                    .show(ui, |ui| {
                        if dynamic {
                            ui.label(egui::RichText::new("Step").strong());
                        }
                        ui.label(egui::RichText::new("Structure").strong());
                        for m in &metrics {
                            ui.label(egui::RichText::new(column_title(m, &d.units)).strong());
                        }
                        if dynamic {
                            for h in [
                                "Moved",
                                "Relative to",
                                "Shift [mm]",
                                "Rotation [°]",
                                "Scale [%]",
                            ] {
                                ui.label(egui::RichText::new(h).strong());
                            }
                        }
                        ui.end_row();
                        let states: Vec<(Option<&LogEntry>, &[DoseRow])> = if dynamic {
                            d.log.iter().map(|e| (Some(e), e.rows.as_slice())).collect()
                        } else {
                            vec![(None, d.rows.as_slice())]
                        };
                        for (entry, rows) in states {
                            for row in rows {
                                if let Some(e) = entry {
                                    ui.monospace(e.step.to_string());
                                }
                                ui.horizontal(|ui| {
                                    let (r, _) = ui.allocate_exact_size(
                                        egui::vec2(10.0, 10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(r, 2.0, theme::rgb(row.color));
                                    let name = ui.label(&row.name);
                                    if row.outside > 0.001 {
                                        name.on_hover_text(format!(
                                            "{:.1} % of the structure lies outside the dose \
                                             grid and counts as zero dose",
                                            row.outside * 100.0
                                        ));
                                    }
                                    if entry.is_none()
                                        && ui
                                            .small_button("✖")
                                            .on_hover_text(
                                                "Leave this structure out of the table (it \
                                                 stays ticked in the views); the Excluded \
                                                 line below brings it back",
                                            )
                                            .clicked()
                                    {
                                        exclude = Some(row.name.clone());
                                    }
                                });
                                for (m, v) in metrics.iter().zip(&row.values) {
                                    match v {
                                        Some(v) => {
                                            let text = match m {
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
                                if let Some(e) = entry {
                                    for text in [
                                        &e.moved,
                                        &e.summary.relative,
                                        &e.summary.shift,
                                        &e.summary.rotation,
                                        &e.summary.scale,
                                    ] {
                                        if text.is_empty() {
                                            ui.weak("-");
                                        } else {
                                            ui.label(text);
                                        }
                                    }
                                }
                                ui.end_row();
                            }
                        }
                    });
            });
        if dynamic && d.log.is_empty() {
            ui.weak("Move a structure to add the first entry");
        }
        if let Some(name) = exclude {
            if !self.dose_est.excluded.contains(&name) {
                self.dose_est.excluded.push(name);
            }
        }
        if !self.dose_est.excluded.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.label("Excluded:");
                let mut back = None;
                for (i, name) in self.dose_est.excluded.iter().enumerate() {
                    if ui
                        .small_button(format!("{name} ✚"))
                        .on_hover_text("Back into the table")
                        .clicked()
                    {
                        back = Some(i);
                    }
                }
                if let Some(i) = back {
                    self.dose_est.excluded.remove(i);
                }
            });
        }
        if small_tip_button(
            ui,
            "💾 Export CSV",
            "Save the table as it stands, one line per structure",
        ) {
            let d = &self.dose_est;
            let text = if d.dynamic {
                log_csv(&metrics, &d.units, &d.log)
            } else {
                rows_csv(&metrics, &d.units, &d.rows)
            };
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Save the dose estimation table")
                .set_file_name("dose_estimation.csv")
                .save_file()
            {
                match std::fs::write(&path, text) {
                    Ok(()) => self.notice = Some(format!("Written to {}", path.display())),
                    Err(e) => self.error = Some(format!("Could not write the file: {e}")),
                }
            }
        }
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
        let n = d.columns.len();
        for bad in [
            "", "   ", "D", "x", "Dnan%", "Dinf%", "D150%", "V-5", "D-1cc", "%",
        ] {
            d.new_metric = bad.into();
            d.add_metric();
            assert_eq!(d.columns.len(), n, "{bad:?} must not add a column");
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
            assert_eq!(d.columns.last(), Some(&m), "{good}");
            assert!(d.new_metric.is_empty() && d.bad_metric.is_none());
        }
        // Not twice.
        for again in ["D50%", "Dmean", "Volume"] {
            d.new_metric = again.into();
            d.add_metric();
        }
        assert_eq!(d.columns.len(), n + 4);
        // Every column can go, and the CSV still has a header.
        d.columns.clear();
        assert_eq!(d.metrics().len(), 0);
        let csv = rows_csv(
            &[Metric::Mean, Metric::DoseAtPct(95.0)],
            "GY",
            &[DoseRow {
                name: "a,b".into(),
                color: [0; 3],
                values: vec![Some(1.5), None],
                outside: 0.25,
            }],
        );
        assert_eq!(
            csv,
            "structure,Dmean [Gy],D95% [Gy],outside_dose_grid_pct\n\"a,b\",1.5000,,25.00\n"
        );
    }
}
