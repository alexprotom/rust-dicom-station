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

use super::combine_win::ItemRef;
use super::*;

/// The window's state.
pub(super) struct CompareDialog {
    pub slot_a: usize,
    pub item_a: Option<usize>,
    pub slot_b: usize,
    pub item_b: Option<usize>,
    /// The last computation, as printable lines.
    pub result: Vec<String>,
}

impl ViewerApp {
    pub(super) fn open_compare_dialog(&mut self, slot: usize) {
        self.compare_dialog = Some(CompareDialog {
            slot_a: slot,
            item_a: None,
            slot_b: slot,
            item_b: None,
            result: Vec::new(),
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
        let pick = |slot: usize, sel: Option<usize>| -> Option<(ItemRef, String)> {
            let cands = self.combine_candidates(slot);
            sel.and_then(|i| cands.get(i).cloned())
        };
        let (Some((ia, la)), Some((ib, lb))) = (pick(slot_a, d.item_a), pick(slot_b, d.item_b))
        else {
            if let Some(d) = &mut self.compare_dialog {
                d.result = vec!["Pick two structures first.".into()];
            }
            return;
        };
        let (Some((ma, ga, _, _)), Some((mb, gb, _, _))) = (
            self.item_mask_grid(slot_a, ia),
            self.item_mask_grid(slot_b, ib),
        ) else {
            if let Some(d) = &mut self.compare_dialog {
                d.result = vec!["One of the structures is gone or empty.".into()];
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
        match motion::overlap(&ma, &mb_on_a, &ga) {
            Some(o) => {
                lines.push(format!("A: {la} - {:.2} cm³", o.vol_a_cm3));
                lines.push(format!("B: {lb} - {:.2} cm³", o.vol_b_cm3));
                if let Some(s) = o.centroid_shift() {
                    lines.push(format!(
                        "Centroid offset A → B: RL {:+.2} · AP {:+.2} · SI {:+.2} mm  (|d| = {:.2} mm)",
                        s.x,
                        s.y,
                        s.z,
                        s.length()
                    ));
                }
                lines.push(format!("Dice: {:.3}", o.dice));
                lines.push(format!("HD95: {:.2} mm", o.hd95_mm));
                lines.push(format!("Mean surface distance: {:.2} mm", o.msd_mm));
            }
            None => lines.push(
                "Nothing to compare - one of the masks is empty (a structure from the other \
                 dataset may lie outside this volume; resampling cannot invent it)."
                    .into(),
            ),
        }
        if let Some(d) = &mut self.compare_dialog {
            d.result = lines;
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
                    "Volumes, centroid offset, Dice, HD95 and mean surface distance of \
                     any two structures.",
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
                        let sel = item
                            .and_then(|i| list.get(i).cloned())
                            .unwrap_or_else(|| "(pick)".into());
                        egui::ComboBox::from_id_salt(salt.to_string())
                            .width(260.0)
                            .selected_text(sel)
                            .show_ui(ui, |ui| {
                                for (i, l) in list.iter().enumerate() {
                                    ui.selectable_value(item, Some(i), l);
                                }
                            });
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
                ui.add_space(4.0);
                for line in &d.result {
                    ui.label(line.clone());
                }
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Compare").clicked() {
                        compute = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            },
        );
        if compute {
            self.compare_now();
        }
        if close || !open {
            self.compare_dialog = None;
        }
    }
}
