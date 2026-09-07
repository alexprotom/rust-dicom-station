//! *Tools ▶ Compare structures*: geometric comparison of any two
//! structures - volumes, centroids and their offset, Dice, HD95 and mean
//! surface distance.
//!
//! The two structures may live in either dataset and on different lattices;
//! the second is resampled onto the first's grid through patient
//! coordinates. Across two datasets that is only meaningful when both are
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
    pub result: Vec<String>,
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
                d.csv.clear();
            }
            return;
        };
        // Two points of interest are compared as points: the distance
        // between them, which is the target registration error when the two
        // are the same anatomical landmark in two datasets.
        if let (Some((na, pa)), Some((nb, pb))) =
            (self.poi_of_item(slot_a, ia), self.poi_of_item(slot_b, ib))
        {
            let d = pb - pa;
            let lines = vec![
                format!("A: {na} - {:.2}, {:.2}, {:.2} mm", pa.x, pa.y, pa.z),
                format!("B: {nb} - {:.2}, {:.2}, {:.2} mm", pb.x, pb.y, pb.z),
                format!(
                    "A → B: RL {:+.2} · AP {:+.2} · SI {:+.2} mm  (|d| = {:.2} mm)",
                    d.x,
                    d.y,
                    d.z,
                    d.length()
                ),
                "Two points: the distance is the target registration error when they \
                 are meant to be the same landmark."
                    .into(),
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
                lines.push(format!("A: {la} - {:.2} cm³", o.vol_a_cm3));
                lines.push(format!("B: {lb} - {:.2} cm³", o.vol_b_cm3));
                rows.push(("volume A (cm3)".into(), format!("{:.4}", o.vol_a_cm3)));
                rows.push(("volume B (cm3)".into(), format!("{:.4}", o.vol_b_cm3)));
                if let (Some(a), Some(b)) = (o.centroid_a, o.centroid_b) {
                    for (tag, c) in [("A", a), ("B", b)] {
                        rows.push((
                            format!("centroid {tag} (mm)"),
                            format!("{:.3} {:.3} {:.3}", c.x, c.y, c.z),
                        ));
                    }
                }
                if let Some(s) = o.centroid_shift() {
                    lines.push(format!(
                        "Centroid offset A → B: RL {:+.2} · AP {:+.2} · SI {:+.2} mm  (|d| = {:.2} mm)",
                        s.x,
                        s.y,
                        s.z,
                        s.length()
                    ));
                    rows.push((
                        "centroid offset (mm)".into(),
                        format!("{:.3} {:.3} {:.3}", s.x, s.y, s.z),
                    ));
                    rows.push((
                        "centroid distance (mm)".into(),
                        format!("{:.3}", s.length()),
                    ));
                }
                lines.push(format!("Dice: {:.3}", o.dice));
                lines.push(format!("HD95: {:.2} mm", o.hd95_mm));
                lines.push(format!(
                    "Surface distance: mean {:.2} · SD {:.2} · max {:.2} mm",
                    o.msd_mm, o.sd_mm, o.max_mm
                ));
                rows.push(("dice".into(), format!("{:.4}", o.dice)));
                rows.push(("hd95 (mm)".into(), format!("{:.3}", o.hd95_mm)));
                rows.push(("surface mean (mm)".into(), format!("{:.3}", o.msd_mm)));
                rows.push(("surface sd (mm)".into(), format!("{:.3}", o.sd_mm)));
                rows.push(("surface max (mm)".into(), format!("{:.3}", o.max_mm)));
            }
            None => lines.push(
                "Nothing to compare - one of the masks is empty (a structure from the other \
                 dataset may lie outside this volume; resampling cannot invent it)."
                    .into(),
            ),
        }
        if want_fit {
            match motion::surface_fit(&ma, &mb_on_a, &ga) {
                Some(f) => {
                    let t = f.dof.translation;
                    let r = f.dof.rotation_deg;
                    lines.push(format!(
                        "Rigid offset A → B: t = ({:+.2}, {:+.2}, {:+.2}) mm  \
                         r = ({:+.2}, {:+.2}, {:+.2})°",
                        t.x, t.y, t.z, r[0], r[1], r[2]
                    ));
                    lines.push(format!(
                        "Surface points after the fit: mean {:.2} · SD {:.2} · max {:.2} mm \
                         (before: mean {:.2})",
                        f.residual_mm[0], f.residual_mm[1], f.residual_mm[2], f.before_mm[0]
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
        let comparison = self.comparison;
        let mut compute = false;
        let mut save = false;
        let mut close = false;
        let mut open = true;
        let d = self.compare_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "compare",
            "◑ Compare structures",
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
                            for (s, name) in SLOT_NAMES.iter().enumerate() {
                                if ui.selectable_label(*slot == s, *name).clicked() {
                                    *slot = s;
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
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Save the comparison")
                .add_filter("CSV", &["csv"])
                .set_file_name("structure_comparison.csv")
                .save_file()
            {
                if let Err(e) = std::fs::write(&path, csv) {
                    self.error = Some(format!("Could not write {}: {e}", path.display()));
                }
            }
        }
        if close || !open {
            self.compare_dialog = None;
        }
    }
}
