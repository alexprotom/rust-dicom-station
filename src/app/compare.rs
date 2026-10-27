//! *Modules ▶ Structure comparison*: geometric comparison of any two
//! structures - volumes, centroids and their offset, Dice, HD95 and mean
//! surface distance.
//!
//! The two structures may live in either workspace and on different lattices;
//! the second is resampled onto the first's grid through patient
//! coordinates. Across two workspaces that is only meaningful when both are
//! in the same frame of reference (or have been registered and propagated
//! first) - the section says so instead of silently comparing apples to
//! oranges.
//!
//! The section works on the views while it is unfolded: the two structures
//! picked are drawn thicker there, and the centroid rows carry a button
//! that puts the crosshair on that centroid.

use crate::motion;
use crate::volume::Grid;

use super::combine::ItemRef;
use super::*;

/// The section's state.
pub(super) struct CompareState {
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
    /// two on a large structure and is the only part of the section that
    /// does.
    pub fit_rigid: bool,
    /// The two centroids of the last computation, in patient coordinates,
    /// with the workspace each belongs to: where the crosshair button goes.
    pub centroids: [Option<(usize, crate::geometry::Vec3)>; 2],
}

impl CompareState {
    fn new(slot: usize) -> Self {
        CompareState {
            slot_a: slot,
            item_a: None,
            slot_b: slot,
            item_b: None,
            result: Vec::new(),
            rows: Vec::new(),
            csv: String::new(),
            fit_rigid: true,
            centroids: [None, None],
        }
    }
}

impl ViewerApp {
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
        let Some(d) = &self.compare else {
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
            if let Some(d) = &mut self.compare {
                d.result = vec!["Pick two structures first.".into()];
                d.rows.clear();
                d.csv.clear();
                d.centroids = [None, None];
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
            if let Some(d) = &mut self.compare {
                d.result = lines;
                d.rows = rows;
                d.csv = csv;
                d.centroids = [Some((slot_a, pa)), Some((slot_b, pb))];
            }
            return;
        }
        let (Some((ma, ga, _, _)), Some((mb, gb, _, _))) = (
            self.item_mask_grid(slot_a, ia),
            self.item_mask_grid(slot_b, ib),
        ) else {
            if let Some(d) = &mut self.compare {
                d.result = vec!["One of the structures is gone or empty.".into()];
                d.rows.clear();
                d.csv.clear();
                d.centroids = [None, None];
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
        // Both centroids are in patient coordinates; each is gone to in its
        // own workspace.
        let mut centroids = [None, None];
        match motion::overlap(&ma, &mb_on_a, &ga) {
            Some(o) => {
                centroids = [
                    o.centroid_a.map(|c| (slot_a, c)),
                    o.centroid_b.map(|c| (slot_b, c)),
                ];
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
        if let Some(d) = &mut self.compare {
            d.result = lines;
            d.csv = if rows.is_empty() { String::new() } else { csv };
            d.rows = rows;
            d.centroids = centroids;
        }
    }

    /// The section: two pickers, the numbers, *Compare* and the CSV.
    pub(super) fn compare_section(&mut self, ui: &mut egui::Ui) {
        self.module_header(
            ui,
            Module::Comparison,
            |_, _| {},
            |app, ui| app.compare_body(ui),
        );
    }

    fn compare_body(&mut self, ui: &mut egui::Ui) {
        if !self.slots.iter().any(|s| s.study.is_some()) {
            ui.weak("Load a workspace with structures to compare");
            return;
        }
        let first = self.first_volume_slot();
        let d = self.compare.get_or_insert_with(|| CompareState::new(first));
        // A workspace closed under the section takes its pick with it.
        let open: [bool; MAX_WORKSPACES] = std::array::from_fn(|s| s == 0 || self.shown[s]);
        for (slot, item) in [
            (&mut d.slot_a, &mut d.item_a),
            (&mut d.slot_b, &mut d.item_b),
        ] {
            if !open[*slot] {
                *slot = first;
                *item = None;
            }
        }
        let both = [d.slot_a, d.slot_b];
        let cands: [Vec<(ItemRef, String)>; 2] = [
            self.combine_candidates(both[0]),
            self.combine_candidates(both[1]),
        ];
        // The two structures picked are what the section is about: the
        // views draw them thicker while it is unfolded.
        let picked: Vec<(usize, ItemRef)> = [
            (0, self.compare.as_ref().and_then(|d| d.item_a)),
            (1, self.compare.as_ref().and_then(|d| d.item_b)),
        ]
        .into_iter()
        .filter_map(|(k, i)| Some((both[k], cands[k].get(i?)?.0)))
        .collect();
        self.focus_structures(picked);
        let labels: [Vec<String>; 2] = [
            cands[0].iter().map(|(_, l)| l.clone()).collect(),
            cands[1].iter().map(|(_, l)| l.clone()).collect(),
        ];
        let comparison = self.comparing();
        // The letters the two rows may be pointed at: the open workspaces,
        // so three or four of them are all offered and a closed one is not.
        let choices = self.open_slots();
        let mut compute = false;
        let mut save = false;
        let mut jump: Option<(usize, crate::geometry::Vec3)> = None;
        let d = self.compare.as_mut().expect("made above");
        let row = |ui: &mut egui::Ui,
                   slot: &mut usize,
                   item: &mut Option<usize>,
                   list: &[String],
                   salt: &str| {
            if comparison {
                for s in &choices {
                    if ui.selectable_label(*slot == *s, SLOT_NAMES[*s]).clicked() && *slot != *s {
                        *slot = *s;
                        *item = None;
                    }
                }
            }
            index_picker(ui, salt, item, list, 170.0);
        };
        // The candidate lists were computed for the slots as they were at
        // the top of the frame; after a slot switch the next frame refreshes
        // them, so the pick is cleared to stay in bounds.
        form::form(ui, "compare_form", |f| {
            f.row("Structure 1", |ui| {
                row(ui, &mut d.slot_a, &mut d.item_a, &labels[0], "cmp_a")
            });
            f.row("Structure 2", |ui| {
                row(ui, &mut d.slot_b, &mut d.item_b, &labels[1], "cmp_b")
            });
            f.row_tip(
                "Rigid offset",
                "Translation and rotation that best carry structure 1 onto structure 2 \
                 (a closest-point fit over the surfaces), and what distance is left over \
                 afterwards. A second or two on a large structure.",
                |ui| ui.checkbox(&mut d.fit_rigid, "Least-squares fit"),
            );
        });
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
        });
        for line in &d.result {
            ui.label(line.clone());
        }
        if !d.rows.is_empty() {
            egui::ScrollArea::horizontal()
                .id_salt("compare_rows_scroll")
                .show(ui, |ui| {
                    egui::Grid::new("compare_rows")
                        .striped(true)
                        .num_columns(3)
                        .spacing([12.0, 2.0])
                        .show(ui, |ui| {
                            for (k, v) in &d.rows {
                                ui.label(k);
                                ui.monospace(v);
                                // The centroid and point rows go to that
                                // place in the views.
                                let which = if k.starts_with("centroid A")
                                    || k.starts_with("point A")
                                {
                                    Some(0)
                                } else if k.starts_with("centroid B") || k.starts_with("point B") {
                                    Some(1)
                                } else {
                                    None
                                };
                                match which.and_then(|w| d.centroids[w]) {
                                    Some(c)
                                        if small_tip_button(
                                            ui,
                                            "⌖",
                                            "Put the crosshair on this point",
                                        ) =>
                                    {
                                        jump = Some(c);
                                    }
                                    Some(_) => {}
                                    None => {
                                        ui.label("");
                                    }
                                }
                                ui.end_row();
                            }
                        });
                });
        }
        if compute {
            self.compare_now();
        }
        if let Some((slot, p)) = jump {
            self.set_cursor_patient(slot, p);
        }
        if save {
            let csv = self
                .compare
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
    }
}
