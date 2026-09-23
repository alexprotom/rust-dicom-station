//! *Tools ▶ Structure comparison*: geometric comparison of any two
//! structures - volumes, centroids and their offset, Dice, HD95 and mean
//! surface distance.
//!
//! The two structures may live in either workspace and on different lattices;
//! the second is resampled onto the first's grid through patient
//! coordinates. Across two workspaces that is only meaningful when both are
//! in the same frame of reference (or have been registered and propagated
//! first) - the window says so instead of silently comparing apples to
//! oranges.

use crate::motion;
use crate::volume::Grid;

use super::combine::ItemRef;
use super::*;

/// The window's state.
pub(super) struct CompareDialog {
    pub slot_a: usize,
    pub item_a: Option<usize>,
    pub slot_b: usize,
    pub item_b: Option<usize>,
    /// The last computation, as printable lines.
    /// Notes: what has no number of its own - a warning, or why there is
    /// nothing to compare.
    pub result: Vec<String>,
    /// The measurements, as they go into the table and the CSV.
    pub rows: Vec<(String, String)>,
    /// The same numbers as a CSV table, ready to be written out.
    pub csv: String,
    /// Whether to run the closest-point rigid fit, which costs a second or
    /// two on a large structure and is the only part of the window that
    /// does.
    pub fit_rigid: bool,
}

impl ViewerApp {
    pub(super) fn open_compare_dialog(&mut self, slot: usize) {
        self.compare_dialog = Some(CompareDialog {
            slot_a: slot,
            item_a: None,
            slot_b: slot,
            item_b: None,
            result: Vec::new(),
            rows: Vec::new(),
            csv: String::new(),
            fit_rigid: true,
        });
    }

    /// One structure's mask on a definite grid, with its identity - the
    /// common currency of the compare and transfer tools. A contour is
    /// rasterized onto the displayed volume of its slot; a segment comes on
    /// its own series' lattice.
    pub(super) fn item_mask_grid(
        &self,
        slot: usize,
        item: ItemRef,
    ) -> Option<(Vec<u8>, Grid, String, [u8; 3])> {
        let study = self.slots[slot].study.as_ref()?;
        match item.kind {
            SetKind::Structures => {
                let roi = study.structure_sets.get(item.set)?.rois.get(item.idx)?;
                let grid = study.volume.grid();
                let mask = segmentation::rasterize_roi(&grid, roi)?;
                Some((mask, grid, roi.name.clone(), roi.color))
            }
            SetKind::Segmentations => {
                let ser = study.seg_series.get(item.set)?;
                let seg = ser.segs.get(item.idx)?;
                Some((
                    seg.mask.clone(),
                    ser.grid.clone(),
                    seg.name.clone(),
                    seg.color,
                ))
            }
        }
    }

    fn compare_now(&mut self) {
        let Some(d) = &self.compare_dialog else {
            return;
        };
        let (slot_a, slot_b) = (d.slot_a, d.slot_b);
        let want_fit = d.fit_rigid;
        let pick = |slot: usize, sel: Option<usize>| -> Option<(ItemRef, String)> {
            let cands = self.combine_candidates(slot);
            sel.and_then(|i| cands.get(i).cloned())
        };
        let (Some((ia, la)), Some((ib, lb))) = (pick(slot_a, d.item_a), pick(slot_b, d.item_b))
        else {
            if let Some(d) = &mut self.compare_dialog {
                d.result = vec!["Pick two structures first.".into()];
                d.rows.clear();
                d.csv.clear();
            }
            return;
        };
        // Two points of interest are compared as points: the distance
        // between them, which is the target registration error when the two
        // are the same anatomical landmark in two workspaces.
        if let (Some((na, pa)), Some((nb, pb))) =
            (self.poi_of_item(slot_a, ia), self.poi_of_item(slot_b, ib))
        {
            let d = pb - pa;
            let lines = vec!["Two points: the distance is the target registration error \
                 when they are meant to be the same landmark."
                .to_string()];
            let rows = vec![
                ("structure A".to_string(), na.clone()),
                ("structure B".to_string(), nb.clone()),
                (
                    "point A (mm)".to_string(),
                    format!("{:+.2} {:+.2} {:+.2}", pa.x, pa.y, pa.z),
                ),
                (
                    "point B (mm)".to_string(),
                    format!("{:+.2} {:+.2} {:+.2}", pb.x, pb.y, pb.z),
                ),
                (
                    "offset RL/AP/SI (mm)".to_string(),
                    format!("{:+.2} {:+.2} {:+.2}", d.x, d.y, d.z),
                ),
                ("distance (mm)".to_string(), format!("{:.2}", d.length())),
            ];
            let csv = format!(
                "quantity,value\nstructure A,\"{}\"\nstructure B,\"{}\"\n\
                 point A (mm),{:.3} {:.3} {:.3}\npoint B (mm),{:.3} {:.3} {:.3}\n\
                 offset (mm),{:.3} {:.3} {:.3}\ndistance (mm),{:.3}\n",
                na.replace('"', "'"),
                nb.replace('"', "'"),
                pa.x,
                pa.y,
                pa.z,
                pb.x,
                pb.y,
                pb.z,
                d.x,
                d.y,
                d.z,
                d.length()
            );
            if let Some(d) = &mut self.compare_dialog {
                d.result = lines;
                d.rows = rows;
                d.csv = csv;
            }
            return;
        }
        let (Some((ma, ga, _, _)), Some((mb, gb, _, _))) = (
            self.item_mask_grid(slot_a, ia),
            self.item_mask_grid(slot_b, ib),
        ) else {
            if let Some(d) = &mut self.compare_dialog {
                d.result = vec!["One of the structures is gone or empty.".into()];
                d.rows.clear();
                d.csv.clear();
            }
            return;
        };
        let mut lines = Vec::new();
        if ga.frame_of_reference_uid != gb.frame_of_reference_uid {
            lines.push(
                "⚠ Different frames of reference - the comparison assumes the patient \
                 coordinates already correspond (register + propagate first if they do not)."
                    .into(),
            );
        }
        let mb_on_a = if gb.matches(&ga) {
            mb
        } else {
            crate::dicomseg::resample_mask(&mb, &gb, &ga)
        };
        let mut rows: Vec<(String, String)> = Vec::new();
        match motion::overlap(&ma, &mb_on_a, &ga) {
            Some(o) => {
                rows.push(("structure A".into(), la.clone()));
                rows.push(("structure B".into(), lb.clone()));
                rows.push(("volume A (cm3)".into(), format!("{:.2}", o.vol_a_cm3)));
                rows.push(("volume B (cm3)".into(), format!("{:.2}", o.vol_b_cm3)));
                if let (Some(a), Some(b)) = (o.centroid_a, o.centroid_b) {
                    for (tag, c) in [("A", a), ("B", b)] {
                        rows.push((
                            format!("centroid {tag} (mm)"),
                            format!("{:.3} {:.3} {:.3}", c.x, c.y, c.z),
                        ));
                    }
                }
                if let Some(s) = o.centroid_shift() {
                    rows.push((
                        "centroid offset (mm)".into(),
                        format!("{:.3} {:.3} {:.3}", s.x, s.y, s.z),
                    ));
                    rows.push((
                        "centroid distance (mm)".into(),
                        format!("{:.3}", s.length()),
                    ));
                }
                rows.push(("dice".into(), format!("{:.4}", o.dice)));
                rows.push(("hd95 (mm)".into(), format!("{:.3}", o.hd95_mm)));
                rows.push(("surface mean (mm)".into(), format!("{:.3}", o.msd_mm)));
                rows.push(("surface sd (mm)".into(), format!("{:.3}", o.sd_mm)));
                rows.push(("surface max (mm)".into(), format!("{:.3}", o.max_mm)));
            }
            None => lines.push(
                "Nothing to compare - one of the masks is empty (a structure from the other \
                 workspace may lie outside this volume; resampling cannot invent it)."
                    .into(),
            ),
        }
        if want_fit {
            match motion::surface_fit(&ma, &mb_on_a, &ga) {
                Some(f) => {
                    let t = f.dof.translation;
                    let r = f.dof.rotation_deg;
                    rows.push((
                        "surface distance before the fit (mm)".into(),
                        format!("{:.3}", f.before_mm[0]),
                    ));
                    rows.push((
                        "rigid translation (mm)".into(),
                        format!("{:.3} {:.3} {:.3}", t.x, t.y, t.z),
                    ));
                    rows.push((
                        "rigid rotation (deg)".into(),
                        format!("{:.3} {:.3} {:.3}", r[0], r[1], r[2]),
                    ));
                    rows.push((
                        "fit residual mean/sd/max (mm)".into(),
                        format!(
                            "{:.3} {:.3} {:.3}",
                            f.residual_mm[0], f.residual_mm[1], f.residual_mm[2]
                        ),
                    ));
                    rows.push((
                        "surface points A/B".into(),
                        format!("{} {}", f.points[0], f.points[1]),
                    ));
                }
                None => lines.push("Rigid offset: one of the surfaces is empty.".into()),
            }
        }
        let mut csv = String::from("quantity,value\n");
        csv.push_str(&format!("structure A,\"{}\"\n", la.replace('"', "'")));
        csv.push_str(&format!("structure B,\"{}\"\n", lb.replace('"', "'")));
        for (k, v) in &rows {
            csv.push_str(&format!("{k},{v}\n"));
        }
        if let Some(d) = &mut self.compare_dialog {
            d.result = lines;
            d.csv = if rows.is_empty() { String::new() } else { csv };
            d.rows = rows;
        }
    }

    pub(super) fn compare_window(&mut self, ctx: &egui::Context) {
        let Some(d) = &self.compare_dialog else {
            return;
        };
        let both = [d.slot_a, d.slot_b];
        let cands: [Vec<String>; 2] = [
            self.combine_candidates(both[0])
                .into_iter()
                .map(|(_, l)| l)
                .collect(),
            self.combine_candidates(both[1])
                .into_iter()
                .map(|(_, l)| l)
                .collect(),
        ];
        let comparison = self.comparing();
        // The letters the two rows may be pointed at: the open workspaces,
        // so three or four of them are all offered and a closed one is not.
        let choices = self.open_slots();
        let mut compute = false;
        let mut save = false;
        let mut close = false;
        let mut open = true;
        let d = self.compare_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "compare",
            "◑ Structure comparison",
            &mut open,
            detach::WinOpts::default(),
            |ui| {
                ui.label(
                    "Volumes, centroid offset, Dice, HD95 and surface distances of any \
                     two structures, and the rigid body that best carries one onto the \
                     other.",
                );
                ui.add_space(4.0);
                let row = |ui: &mut egui::Ui,
                           what: &str,
                           slot: &mut usize,
                           item: &mut Option<usize>,
                           list: &[String],
                           salt: &str| {
                    ui.horizontal(|ui| {
                        ui.label(what);
                        if comparison {
                            for s in &choices {
                                if ui.selectable_label(*slot == *s, SLOT_NAMES[*s]).clicked() {
                                    *slot = *s;
                                    *item = None;
                                }
                            }
                        }
                        index_picker(ui, salt, item, list);
                    });
                };
                // The candidate lists were computed for the slots as they
                // were at the top of the frame; after a slot switch the next
                // frame refreshes them, so clear the pick to stay in bounds.
                row(
                    ui,
                    "Structure 1:",
                    &mut d.slot_a,
                    &mut d.item_a,
                    &cands[0],
                    "cmp_a",
                );
                row(
                    ui,
                    "Structure 2:",
                    &mut d.slot_b,
                    &mut d.item_b,
                    &cands[1],
                    "cmp_b",
                );
                ui.checkbox(
                    &mut d.fit_rigid,
                    "Least-squares rigid offset (closest-point fit over the surfaces)",
                )
                .on_hover_text(
                    "Translation and rotation that best carry structure 1 onto structure \
                     2, and what distance is left over afterwards. A second or two on a \
                     large structure.",
                );
                ui.add_space(4.0);
                for line in &d.result {
                    ui.label(line.clone());
                }
                if !d.rows.is_empty() {
                    egui::Grid::new("compare_rows")
                        .striped(true)
                        .num_columns(2)
                        .spacing([12.0, 2.0])
                        .show(ui, |ui| {
                            for (k, v) in &d.rows {
                                ui.label(k);
                                ui.monospace(v);
                                ui.end_row();
                            }
                        });
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Compare").clicked() {
                        compute = true;
                    }
                    ui.add_enabled_ui(!d.csv.is_empty(), |ui| {
                        if ui.button("Save CSV").clicked() {
                            save = true;
                        }
                    });
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            },
        );
        if compute {
            self.compare_now();
        }
        if save {
            let csv = self
                .compare_dialog
                .as_ref()
                .map(|d| d.csv.clone())
                .unwrap_or_default();
            self.ask_save(
                "Save the comparison",
                "structure_comparison.csv",
                None,
                Some(CSV_FILES),
                move |app, path| {
                    if let Err(e) = std::fs::write(&path, csv) {
                        app.error = Some(format!("Could not write {}: {e}", path.display()));
                    }
                },
            );
        }
        if close || !open {
            self.compare_dialog = None;
        }
    }
}
