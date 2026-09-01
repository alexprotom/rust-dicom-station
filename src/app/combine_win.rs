//! The structure-algebra window: combining contours and segmentations.
//!
//! Its one job that the core module ([`crate::structops`]) cannot do is
//! deciding *what the operands are*. Everything else — the four operations,
//! the margins, the tidying — is arithmetic; picking "the GTV from the second
//! structure set of dataset A" out of a data tree, rasterizing it onto the
//! displayed lattice, and putting the answer back as whichever kind the user
//! wants is the part that has to know about the application.
//!
//! The operand list is ordered and the order is shown, because three of the
//! four operations are not commutative in the way people expect: `A − B − C`
//! is not `B − A − C`, and a subtraction with its operands the wrong way
//! round is the most common mistake this tool can make. Hence the ↑ ↓ arrows
//! and the summary line above the buttons that spells the recipe out.

use crate::structops::{self, BoolOp, Cleanup, Combined, Margin, Operand, Recipe};
use crate::volume::Grid;

use super::*;

/// Where one operand comes from: a structure set or a segmentation series of
/// the slot, and an item within it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct ItemRef {
    pub kind: SetKind,
    /// Index of the set / series within the study.
    pub set: usize,
    /// Index of the structure / segment within it.
    pub idx: usize,
}

/// One row of the operand list.
pub(super) struct Row {
    pub item: ItemRef,
    pub margin: Margin,
    /// Shown while the row is edited; the margin fields are per-direction
    /// only when the user asks for them.
    pub per_direction: bool,
}

/// Where the answer goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Output {
    Segment,
    Structure,
}

impl Output {
    fn label(self) -> &'static str {
        match self {
            Output::Segment => "a segmentation",
            Output::Structure => "an RT structure",
        }
    }
}

/// The window's state; it stays open across runs.
pub(super) struct CombineDialog {
    pub slot: usize,
    pub op: BoolOp,
    pub rows: Vec<Row>,
    pub margin: Margin,
    pub margin_per_direction: bool,
    pub cleanup: Cleanup,
    pub name: String,
    pub output: Output,
    /// Interpreted type given to an RT structure result — PTV, ORGAN, …
    pub roi_type: String,
    pub status: Option<String>,
}

/// What a finished run hands back, with the identity of what it ran on.
pub struct CombineResult {
    pub combined: Combined,
    pub name: String,
    pub output: Output,
    pub roi_type: String,
    pub volume_dims: [usize; 3],
    pub frame_of_reference_uid: String,
    pub elapsed_secs: f64,
}

/// Everything a run needs, snapshotted when it starts.
struct CombineRequest {
    recipe: Recipe,
    grid: Grid,
    name: String,
    output: Output,
    roi_type: String,
}

/// The interpreted types offered for an RT structure result — the ones a
/// planning system actually branches on.
const ROI_TYPES: [&str; 7] = [
    "ORGAN",
    "PTV",
    "CTV",
    "GTV",
    "AVOIDANCE",
    "EXTERNAL",
    "CONTROL",
];

impl ViewerApp {
    /// Every structure and segment of `slot` that can be an operand, as
    /// (reference, label) — the pick list, and what the summary line names.
    pub(super) fn combine_candidates(&self, slot: usize) -> Vec<(ItemRef, String)> {
        let mut out = Vec::new();
        let Some(study) = self.slots[slot].study.as_ref() else {
            return out;
        };
        for (si, set) in study.structure_sets.iter().enumerate() {
            for (ii, roi) in set.rois.iter().enumerate() {
                out.push((
                    ItemRef {
                        kind: SetKind::Structures,
                        set: si,
                        idx: ii,
                    },
                    format!("{} / {}", set.label, roi.name),
                ));
            }
        }
        for (si, ser) in study.seg_series.iter().enumerate() {
            for (ii, seg) in ser.segs.iter().enumerate() {
                out.push((
                    ItemRef {
                        kind: SetKind::Segmentations,
                        set: si,
                        idx: ii,
                    },
                    format!("{} / {}", ser.label, seg.name),
                ));
            }
        }
        out
    }

    fn combine_label(&self, slot: usize, item: ItemRef) -> String {
        self.combine_candidates(slot)
            .into_iter()
            .find(|(r, _)| *r == item)
            .map(|(_, l)| l)
            .unwrap_or_else(|| "(gone)".to_string())
    }

    /// Rasterize one operand onto the displayed lattice.
    ///
    /// A contour is rasterized; a segment already on this lattice is taken as
    /// it is; a segment on another lattice is resampled onto this one. The
    /// third case is what makes it legal to combine a segmentation drawn on
    /// one image series with a structure drawn on another.
    fn operand_mask(&self, slot: usize, item: ItemRef, grid: &Grid) -> Option<Vec<u8>> {
        let study = self.slots[slot].study.as_ref()?;
        match item.kind {
            SetKind::Structures => {
                let roi = study.structure_sets.get(item.set)?.rois.get(item.idx)?;
                segmentation::rasterize_roi(grid, roi)
            }
            SetKind::Segmentations => {
                let ser = study.seg_series.get(item.set)?;
                let seg = ser.segs.get(item.idx)?;
                if ser.grid.dims == grid.dims {
                    Some(seg.mask.clone())
                } else {
                    Some(crate::dicomseg::resample_mask(&seg.mask, &ser.grid, grid))
                }
            }
        }
    }

    /// Tools ▶ combine structures: open the window for `slot`, optionally
    /// seeded with the items the tree had ticked.
    pub(super) fn open_combine_dialog(&mut self, slot: usize, seed: Vec<ItemRef>) {
        if self.slots[slot].study.is_none() {
            return;
        }
        let rows: Vec<Row> = seed
            .into_iter()
            .map(|item| Row {
                item,
                margin: Margin::NONE,
                per_direction: false,
            })
            .collect();
        match &mut self.combine_dialog {
            Some(d) if self.combine_job.is_none() => {
                d.slot = slot;
                if !rows.is_empty() {
                    d.rows = rows;
                }
            }
            Some(_) => {}
            None => {
                self.combine_dialog = Some(CombineDialog {
                    slot,
                    op: BoolOp::Union,
                    rows,
                    margin: Margin::NONE,
                    margin_per_direction: false,
                    cleanup: Cleanup::default(),
                    name: "Combined".to_string(),
                    output: Output::Segment,
                    roi_type: "ORGAN".to_string(),
                    status: None,
                });
            }
        }
    }

    /// Rasterize every operand, snapshot the recipe and run it on a worker.
    pub(super) fn start_combine(&mut self) {
        if self.combine_job.is_some() {
            return;
        }
        let Some(d) = &self.combine_dialog else {
            return;
        };
        let slot = d.slot;
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let grid = study.volume.grid();
        let mut operands = Vec::with_capacity(d.rows.len());
        for row in &d.rows {
            let name = self.combine_label(slot, row.item);
            match self.operand_mask(slot, row.item, &grid) {
                Some(mask) => operands.push(Operand {
                    name,
                    mask,
                    margin: row.margin,
                }),
                None => {
                    // An empty contour rasterizes to nothing; saying so beats
                    // silently dropping it out of the recipe.
                    self.error = Some(format!(
                        "'{name}' has nothing on this image series, so the result would \
                         not mean what it says. Remove it from the list or pick another."
                    ));
                    return;
                }
            }
        }
        let name = match d.name.trim() {
            "" => "Combined".to_string(),
            n => n.to_string(),
        };
        let req = CombineRequest {
            recipe: Recipe {
                op: d.op,
                operands,
                margin: d.margin,
                cleanup: d.cleanup,
            },
            grid,
            name,
            output: d.output,
            roi_type: d.roi_type.clone(),
        };
        let progress = Arc::new(Progress::default());
        progress.set("Preparing…");
        self.combine_slot = slot;
        self.combine_job = Some(Job::spawn(progress, move |p| {
            let t0 = std::time::Instant::now();
            let r = structops::combine(&req.recipe, &req.grid, p).map(|combined| CombineResult {
                combined,
                name: req.name.clone(),
                output: req.output,
                roi_type: req.roi_type.clone(),
                volume_dims: req.grid.dims,
                frame_of_reference_uid: req.grid.frame_of_reference_uid.clone(),
                elapsed_secs: t0.elapsed().as_secs_f64(),
            });
            (slot, r)
        }));
    }

    /// A run finished: land it as a segment or as an RT structure.
    pub(super) fn on_combine_done(&mut self, slot: usize, result: CombineResult) {
        if !self.slot_still_shows(slot, result.volume_dims, &result.frame_of_reference_uid) {
            self.error = Some(stale_result(&COMBINE));
            return;
        }
        if result.combined.voxels == 0 {
            self.error = Some(format!(
                "'{}' came out empty. Check the order of the list — a subtraction with \
                 its operands the wrong way round is the usual reason.",
                result.name
            ));
            return;
        }
        let idx = self.add_segmentation(
            slot,
            result.name.clone(),
            result.volume_dims,
            &result.combined.mask,
        );
        if result.output == Output::Structure {
            self.seg_to_rtstruct(slot, idx, &result.roi_type);
            // The mask was only the vehicle; the user asked for contours.
            if let Some(segs) = self.slots[slot].segs_mut() {
                if idx < segs.len() {
                    segs.remove(idx);
                }
            }
            self.slots[slot].active_seg = 0;
        }
        let pieces = match result.combined.pieces {
            0 | 1 => String::new(),
            n => format!(", in {n} separate pieces"),
        };
        if let Some(d) = &mut self.combine_dialog {
            d.status = Some(format!(
                "✔ {} → {}: {:.1} cm³{pieces} in {:.1} s",
                result.name,
                result.output.label(),
                result.combined.cm3,
                result.elapsed_secs
            ));
        }
        self.settings_gen += 1;
    }

    /// The tool window.
    pub(super) fn combine_window(&mut self, ctx: &egui::Context) {
        let Some(slot) = self.combine_dialog.as_ref().map(|d| d.slot) else {
            return;
        };
        if self.slots[slot].study.is_none() {
            self.combine_dialog = None;
            return;
        }
        // Settled before the dialog is borrowed mutably for the frame.
        let candidates = self.combine_candidates(slot);
        let labels: Vec<String> = self
            .combine_dialog
            .as_ref()
            .map(|d| {
                d.rows
                    .iter()
                    .map(|r| self.combine_label(slot, r.item))
                    .collect()
            })
            .unwrap_or_default();
        let Some(d) = &mut self.combine_dialog else {
            return;
        };
        let running = self
            .combine_job
            .as_ref()
            .filter(|_| self.combine_slot == slot);
        let mut open = true;
        let (mut run, mut close, mut cancel) = (false, false, false);
        let mut move_row: Option<(usize, isize)> = None;
        let mut drop_row: Option<usize> = None;
        detach::tool_window(
            ctx,
            "combine",
            COMBINE.title(slot),
            &mut open,
            detach::WinOpts::width(470.0),
            |ui| {
                ui.label(
                    "Builds one structure out of others: union, intersection, subtraction \
                     or symmetric difference, with a margin on any of them. Contours and \
                     segmentations mix freely — each is rasterized onto the displayed \
                     series first.",
                );
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Operation:");
                    egui::ComboBox::from_id_salt("combine_op")
                        .selected_text(d.op.label())
                        .show_ui(ui, |ui| {
                            for o in BoolOp::ALL {
                                ui.selectable_value(&mut d.op, o, o.label());
                            }
                        });
                });
                if d.op == BoolOp::Subtract {
                    ui.weak("The first row is what the rest are taken out of.");
                }

                ui.add_space(4.0);
                if candidates.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "This dataset has no structures or segments to combine yet.",
                        )
                        .color(warn_color(ui.visuals())),
                    );
                }
                // ---- the operand list --------------------------------
                let n_rows = d.rows.len();
                for (i, row) in d.rows.iter_mut().enumerate() {
                    ui.push_id(i, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(format!("{}.", i + 1));
                            let current = labels.get(i).cloned().unwrap_or_default();
                            egui::ComboBox::from_id_salt("pick")
                                .selected_text(shorten(&current))
                                .width(210.0)
                                .show_ui(ui, |ui| {
                                    for (r, label) in &candidates {
                                        ui.selectable_value(&mut row.item, *r, label);
                                    }
                                });
                            if !row.per_direction {
                                let mut mm = row.margin.right;
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut mm)
                                            .range(-200.0..=200.0)
                                            .speed(0.5)
                                            .prefix("margin ")
                                            .suffix(" mm"),
                                    )
                                    .on_hover_text(
                                        "Grow (+) or shrink (−) this operand before it is \
                                         combined. A crop is an intersection whose second \
                                         operand was shrunk.",
                                    )
                                    .changed()
                                {
                                    row.margin = Margin::uniform(mm);
                                }
                            } else {
                                ui.weak(row.margin.describe());
                            }
                            if ui
                                .selectable_label(row.per_direction, "R/L/A/P/S/I")
                                .on_hover_text("Give the margin a value per patient direction")
                                .clicked()
                            {
                                row.per_direction = !row.per_direction;
                            }
                            if ui.button("↑").clicked() && i > 0 {
                                move_row = Some((i, -1));
                            }
                            if ui.button("↓").clicked() && i + 1 < n_rows {
                                move_row = Some((i, 1));
                            }
                            if ui.button("✕").clicked() {
                                drop_row = Some(i);
                            }
                        });
                        if row.per_direction {
                            directional_margin(ui, &mut row.margin);
                        }
                    });
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(!candidates.is_empty(), egui::Button::new("➕ Add"))
                        .clicked()
                    {
                        d.rows.push(Row {
                            item: candidates[0].0,
                            margin: Margin::NONE,
                            per_direction: false,
                        });
                    }
                    if ui.button("Clear").clicked() {
                        d.rows.clear();
                    }
                });

                ui.separator();
                ui.collapsing("Result", |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Margin on the result:");
                        if !d.margin_per_direction {
                            let mut mm = d.margin.right;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut mm)
                                        .range(-200.0..=200.0)
                                        .speed(0.5)
                                        .suffix(" mm"),
                                )
                                .changed()
                            {
                                d.margin = Margin::uniform(mm);
                            }
                        } else {
                            ui.weak(d.margin.describe());
                        }
                        if ui
                            .selectable_label(d.margin_per_direction, "R/L/A/P/S/I")
                            .clicked()
                        {
                            d.margin_per_direction = !d.margin_per_direction;
                        }
                    });
                    if d.margin_per_direction {
                        directional_margin(ui, &mut d.margin);
                    }
                    ui.checkbox(&mut d.cleanup.fill_holes, "Fill interior cavities")
                        .on_hover_text(
                            "Slice by slice, so a lung that drains through the trachea \
                             still closes.",
                        );
                    ui.horizontal(|ui| {
                        ui.label("Smooth:");
                        ui.add(
                            egui::Slider::new(&mut d.cleanup.close_mm, 0.0..=10.0)
                                .suffix(" mm")
                                .fixed_decimals(1),
                        )
                        .on_hover_text("A closing, to take the staircase off the surface.");
                    });
                    ui.checkbox(&mut d.cleanup.keep_largest, "Keep only the largest piece")
                        .on_hover_text(
                            "Useful after a subtraction that leaves slivers; destructive \
                             on anything genuinely paired, like two lungs.",
                        );
                    ui.add_enabled_ui(!d.cleanup.keep_largest, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("…or drop pieces under:");
                            ui.add(
                                egui::DragValue::new(&mut d.cleanup.min_volume_cm3)
                                    .range(0.0..=1000.0)
                                    .speed(0.1)
                                    .suffix(" cm³"),
                            );
                        });
                    });
                });

                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.add(egui::TextEdit::singleline(&mut d.name).desired_width(140.0));
                    ui.label("as");
                    egui::ComboBox::from_id_salt("combine_out")
                        .selected_text(d.output.label())
                        .width(130.0)
                        .show_ui(ui, |ui| {
                            for o in [Output::Segment, Output::Structure] {
                                ui.selectable_value(&mut d.output, o, o.label());
                            }
                        });
                    if d.output == Output::Structure {
                        egui::ComboBox::from_id_salt("combine_roi_type")
                            .selected_text(&d.roi_type)
                            .width(110.0)
                            .show_ui(ui, |ui| {
                                for t in ROI_TYPES {
                                    ui.selectable_value(&mut d.roi_type, t.to_string(), t);
                                }
                            });
                    }
                });

                ui.separator();
                // The recipe, spelled out — the cheapest possible guard
                // against an operand list in the wrong order.
                ui.label(egui::RichText::new(recipe_line(d, &labels)).italics());
                ui.separator();
                match running {
                    Some(job) => cancel = progress_row(ui, &job.progress),
                    None => {
                        ui.horizontal(|ui| {
                            let ready = d.rows.len() > usize::from(d.op != BoolOp::Union);
                            if ui
                                .add_enabled(ready, egui::Button::new("▶ Combine"))
                                .on_hover_text("Evaluate the recipe on the displayed series")
                                .clicked()
                            {
                                run = true;
                            }
                            if ui.button("Close").clicked() {
                                close = true;
                            }
                        });
                    }
                }
                if let Some(status) = &d.status {
                    ui.separator();
                    ui.weak(status);
                }
            },
        );
        if let Some((i, delta)) = move_row {
            let j = (i as isize + delta) as usize;
            if let Some(d) = &mut self.combine_dialog {
                if j < d.rows.len() {
                    d.rows.swap(i, j);
                }
            }
        }
        if let Some(i) = drop_row {
            if let Some(d) = &mut self.combine_dialog {
                if i < d.rows.len() {
                    d.rows.remove(i);
                }
            }
        }
        if cancel {
            if let Some(job) = &self.combine_job {
                job.progress.cancel();
            }
        }
        if run {
            self.start_combine();
        }
        if !open || close {
            self.combine_dialog = None;
        }
    }
}

/// `PTV ∪ Nodes − (Cord + 5 mm)` — the recipe as one line of text.
fn recipe_line(d: &CombineDialog, labels: &[String]) -> String {
    if d.rows.is_empty() {
        return "Nothing selected yet.".to_string();
    }
    let mut parts: Vec<String> = Vec::with_capacity(d.rows.len());
    for (i, row) in d.rows.iter().enumerate() {
        let name = shorten(labels.get(i).map(String::as_str).unwrap_or("?"));
        parts.push(if row.margin.is_none() {
            name
        } else {
            format!("({name} {})", row.margin.describe())
        });
    }
    let mut line = parts.join(&format!(" {} ", d.op.joiner()));
    if !d.margin.is_none() {
        line = format!("({line}) {}", d.margin.describe());
    }
    format!(
        "{} = {line}",
        if d.name.trim().is_empty() {
            "result"
        } else {
            d.name.trim()
        }
    )
}

/// The last path component, so a long "Set / Structure" still fits a combo.
fn shorten(label: &str) -> String {
    label.rsplit(" / ").next().unwrap_or(label).to_string()
}

/// Six drag fields, laid out the way a planning system asks for them.
fn directional_margin(ui: &mut egui::Ui, m: &mut Margin) {
    ui.horizontal(|ui| {
        ui.add_space(18.0);
        for (label, value, hint) in [
            ("R", &mut m.right, "toward the patient's right"),
            ("L", &mut m.left, "toward the patient's left"),
            ("A", &mut m.anterior, "anterior"),
            ("P", &mut m.posterior, "posterior"),
            ("S", &mut m.superior, "superior"),
            ("I", &mut m.inferior, "inferior"),
        ] {
            ui.add(
                egui::DragValue::new(value)
                    .range(-200.0..=200.0)
                    .speed(0.5)
                    .prefix(format!("{label} "))
                    .suffix("mm"),
            )
            .on_hover_text(hint);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tool_names_itself_like_the_others() {
        assert_eq!(COMBINE.title(0), "◧ Combine structures — dataset A");
        assert_eq!(COMBINE.menu_entry(1), "◧ Combine structures in dataset B…");
        assert_eq!(COMBINE.short_button(), "◧ Combine");
    }

    #[test]
    fn the_recipe_line_spells_out_order_and_margins() {
        let d = CombineDialog {
            slot: 0,
            op: BoolOp::Subtract,
            rows: vec![
                Row {
                    item: ItemRef {
                        kind: SetKind::Structures,
                        set: 0,
                        idx: 0,
                    },
                    margin: Margin::NONE,
                    per_direction: false,
                },
                Row {
                    item: ItemRef {
                        kind: SetKind::Structures,
                        set: 0,
                        idx: 1,
                    },
                    margin: Margin::uniform(5.0),
                    per_direction: false,
                },
            ],
            margin: Margin::NONE,
            margin_per_direction: false,
            cleanup: Cleanup::default(),
            name: "PTV_eval".into(),
            output: Output::Segment,
            roi_type: "ORGAN".into(),
            status: None,
        };
        let labels = vec!["Set 1 / PTV".to_string(), "Set 1 / Cord".to_string()];
        assert_eq!(recipe_line(&d, &labels), "PTV_eval = PTV − (Cord +5.0 mm)");
    }

    #[test]
    fn a_long_set_name_is_shortened_to_the_structure() {
        assert_eq!(shorten("Structure Set 1 / Lung_L"), "Lung_L");
        assert_eq!(shorten("Lung_L"), "Lung_L");
    }
}
