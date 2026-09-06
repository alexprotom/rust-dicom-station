//! *Tools ▶ New structure*: the generators.
//!
//! A grey-level window, a shape, a dose level. Each is a couple of lines of
//! arithmetic in [`crate::generate`]; what this window adds is where the
//! result goes, which is the same place a drawn structure goes - an ordinary
//! RT structure, editable with the brush from the moment it appears, not a
//! read-only overlay.

use crate::contours::Stack;
use crate::generate::{self, Shape};
use crate::geometry::Vec3;

use super::combine_win::ItemRef;
use super::seg_engines::ToolInfo;
use super::*;

/// The eighth tool. A heavy cross, because everything here makes something
/// new; the glyph is one egui's bundled fonts carry (see the `glyphs` guard).
pub(super) const NEW_ROI: ToolInfo = ToolInfo {
    glyph: "✚",
    name: "New structure",
    verb: "Create a structure in",
};

/// Which generator the window is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Source {
    GreyLevel,
    Shape,
    Dose,
}

impl Source {
    const ALL: [Source; 3] = [Source::GreyLevel, Source::Shape, Source::Dose];
    fn label(self) -> &'static str {
        match self {
            Source::GreyLevel => "Grey level",
            Source::Shape => "Shape",
            Source::Dose => "Dose",
        }
    }
}

pub(super) struct NewRoiDialog {
    pub slot: usize,
    pub source: Source,
    pub name: String,
    pub roi_type: String,
    // Grey level.
    pub lo: f32,
    pub hi: f32,
    /// Restrict the threshold to a structure that is already drawn.
    pub limit: Option<ItemRef>,
    pub keep_largest: bool,
    pub fill_holes: bool,
    // Shape.
    pub shape: Shape,
    /// Half-sizes along the patient x, y, z, in millimetres.
    pub half: [f32; 3],
    /// Centre of the shape; seeded from the crosshair when the window opens.
    pub centre: [f32; 3],
    // Dose.
    pub dose_pct: f32,
    pub dose_absolute: bool,
    pub dose_gy: f32,
    pub status: Option<String>,
}

impl ViewerApp {
    pub(super) fn open_newroi_dialog(&mut self, slot: usize) {
        if !self.slots[slot].has_volume() {
            return;
        }
        let c = self.crosshair_patient(slot).unwrap_or(Vec3::ZERO);
        self.newroi_dialog = Some(NewRoiDialog {
            slot,
            source: Source::GreyLevel,
            name: "New ROI".to_string(),
            roi_type: "ORGAN".to_string(),
            lo: 200.0,
            hi: 4000.0,
            limit: None,
            keep_largest: false,
            fill_holes: false,
            shape: Shape::Sphere,
            half: [20.0, 20.0, 20.0],
            centre: [c.x as f32, c.y as f32, c.z as f32],
            dose_pct: 95.0,
            dose_absolute: false,
            dose_gy: 50.0,
            status: None,
        });
    }

    /// Patient coordinates of the slot's crosshair.
    fn crosshair_patient(&self, slot: usize) -> Option<Vec3> {
        let s = &self.slots[slot];
        let v = s.study.as_ref()?;
        let c = s.cursor;
        Some(v.volume.voxel_to_patient(c[0], c[1], c[2]))
    }

    /// Build the mask the current settings describe.
    fn newroi_mask(&self, d: &NewRoiDialog) -> Result<Vec<u8>, String> {
        let slot = d.slot;
        let study = self.slots[slot]
            .study
            .as_ref()
            .ok_or_else(|| "no dataset".to_string())?;
        let grid = study.volume.grid();
        match d.source {
            Source::GreyLevel => {
                let limit = match d.limit {
                    Some(item) => Some(self.operand_mask(slot, item, &grid).ok_or_else(|| {
                        "the limiting structure has nothing on this image series".to_string()
                    })?),
                    None => None,
                };
                Ok(generate::threshold_mask(
                    &study.volume,
                    d.lo,
                    d.hi,
                    limit.as_deref(),
                ))
            }
            Source::Shape => Ok(generate::shape_mask(
                d.shape,
                &grid,
                Vec3::new(d.centre[0] as f64, d.centre[1] as f64, d.centre[2] as f64),
                [d.half[0] as f64, d.half[1] as f64, d.half[2] as f64],
            )),
            Source::Dose => {
                let dose = study
                    .doses
                    .get(self.slots[slot].active_dose)
                    .ok_or_else(|| "this dataset has no dose".to_string())?;
                let level = if d.dose_absolute {
                    d.dose_gy
                } else {
                    self.slots[slot].dose_reference * d.dose_pct / 100.0
                };
                Ok(generate::dose_mask(&grid, dose, level))
            }
        }
    }

    /// Make the structure, and say what came out.
    fn create_newroi(&mut self) {
        let Some(d) = self.newroi_dialog.take() else {
            return;
        };
        let slot = d.slot;
        let mut mask = match self.newroi_mask(&d) {
            Ok(m) => m,
            Err(why) => {
                self.error = Some(format!("Nothing was created: {why}."));
                self.newroi_dialog = Some(d);
                return;
            }
        };
        let Some(grid) = self.slots[slot].study.as_ref().map(|s| s.volume.grid()) else {
            self.newroi_dialog = Some(d);
            return;
        };
        // The two tidying options every generator wants, and no more: a
        // threshold picks up specks, and a bone ROI is full of marrow.
        if d.source != Source::Shape && (d.keep_largest || d.fill_holes) {
            let cleanup = crate::structops::Cleanup {
                fill_holes: d.fill_holes,
                keep_largest: d.keep_largest,
                ..crate::structops::Cleanup::default()
            };
            cleanup.apply(&mut mask, &grid, &crate::progress::Quiet);
        }
        let set = mask.iter().filter(|&&v| v != 0).count();
        if set == 0 {
            self.error = Some(
                "Nothing was created: no voxel of the displayed series matches. Widen \
                 the window, or check the limiting structure."
                    .into(),
            );
            self.newroi_dialog = Some(d);
            return;
        }
        let stack = Stack::from_mask(&mask, grid.dims, 2);
        let cm3 = stack.volume_cm3(grid.spacing);
        let Some(roi) = self.new_roi(slot, Some(d.name.clone()), &d.roi_type) else {
            self.newroi_dialog = Some(d);
            return;
        };
        let set_idx = self.slots[slot].active_structs;
        if let Some(r) = self.slots[slot]
            .study
            .as_mut()
            .and_then(|st| st.structure_sets.get_mut(set_idx))
            .and_then(|ss| ss.rois.get_mut(roi))
        {
            stack.apply_to_roi(r, &grid);
        }
        self.settings_gen += 1;
        self.edit = None;
        let mut d = d;
        d.status = Some(format!(
            "✔ {} created: {:.1} cm³ on {} slice(s). It is now the structure the \
             contour tools edit.",
            d.name,
            cm3,
            stack.occupied()
        ));
        self.newroi_dialog = Some(d);
    }

    pub(super) fn newroi_window(&mut self, ctx: &egui::Context) {
        let Some(slot) = self.newroi_dialog.as_ref().map(|d| d.slot) else {
            return;
        };
        let candidates: Vec<(ItemRef, String)> = self.combine_candidates(slot);
        let has_dose = self.slots[slot]
            .study
            .as_ref()
            .is_some_and(|s| !s.doses.is_empty());
        let reference = self.slots[slot].dose_reference;
        let comparison = self.comparison;
        let mut open = true;
        let mut create = false;
        let mut close = false;
        let mut switch: Option<usize> = None;
        let d = self.newroi_dialog.as_mut().expect("checked above");
        detach::tool_window(
            ctx,
            "new_roi",
            format!("{} {}", NEW_ROI.glyph, NEW_ROI.name),
            &mut open,
            detach::WinOpts::width(420.0),
            |ui| {
                if comparison {
                    ui.horizontal(|ui| {
                        ui.label("Dataset:");
                        for (s, name) in SLOT_NAMES.iter().enumerate() {
                            if ui.selectable_label(d.slot == s, *name).clicked() {
                                switch = Some(s);
                            }
                        }
                    });
                }
                ui.label(
                    egui::RichText::new(
                        "A structure out of the image, a shape or the dose. It lands as \
                         an ordinary RT structure: correct it with the contour tools \
                         like any other.",
                    )
                    .weak(),
                );
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("From:");
                    for s in Source::ALL {
                        if ui
                            .add_enabled(
                                s != Source::Dose || has_dose,
                                egui::Button::selectable(d.source == s, s.label()),
                            )
                            .clicked()
                        {
                            d.source = s;
                        }
                    }
                });
                ui.add_space(4.0);

                match d.source {
                    Source::GreyLevel => {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Between");
                            ui.add(
                                egui::DragValue::new(&mut d.lo)
                                    .speed(5.0)
                                    .range(-2000.0..=6000.0)
                                    .suffix(" HU"),
                            );
                            ui.label("and");
                            ui.add(
                                egui::DragValue::new(&mut d.hi)
                                    .speed(5.0)
                                    .range(-2000.0..=6000.0)
                                    .suffix(" HU"),
                            );
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Bone:");
                            for (label, lo) in generate::BONE_PRESETS {
                                if ui
                                    .small_button(label)
                                    .on_hover_text(format!("Everything above {lo:.0} HU"))
                                    .clicked()
                                {
                                    d.lo = lo;
                                    d.hi = 6000.0;
                                }
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Only inside:");
                            let sel = d
                                .limit
                                .and_then(|i| {
                                    candidates
                                        .iter()
                                        .find(|(r, _)| *r == i)
                                        .map(|(_, l)| l.clone())
                                })
                                .unwrap_or_else(|| "(the whole image)".into());
                            egui::ComboBox::from_id_salt("newroi_limit")
                                .selected_text(sel)
                                .width(240.0)
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut d.limit, None, "(the whole image)");
                                    for (item, label) in &candidates {
                                        ui.selectable_value(&mut d.limit, Some(*item), label);
                                    }
                                });
                        });
                    }
                    Source::Shape => {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Shape:");
                            for s in Shape::ALL {
                                if ui
                                    .add(egui::Button::selectable(d.shape == s, s.label()))
                                    .clicked()
                                {
                                    d.shape = s;
                                    if s.is_uniform() {
                                        d.half = [d.half[0]; 3];
                                    }
                                }
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            if d.shape.is_uniform() {
                                ui.label("Radius:");
                                let mut r = d.half[0];
                                if ui
                                    .add(
                                        egui::DragValue::new(&mut r)
                                            .speed(0.5)
                                            .range(0.5..=400.0)
                                            .suffix(" mm"),
                                    )
                                    .changed()
                                {
                                    d.half = [r; 3];
                                }
                            } else {
                                ui.label("Half-sizes x/y/z:");
                                for v in &mut d.half {
                                    ui.add(
                                        egui::DragValue::new(v)
                                            .speed(0.5)
                                            .range(0.5..=400.0)
                                            .suffix(" mm"),
                                    );
                                }
                            }
                        });
                        ui.horizontal_wrapped(|ui| {
                            ui.label("Centre:");
                            for v in &mut d.centre {
                                ui.add(egui::DragValue::new(v).speed(1.0).suffix(" mm"));
                            }
                        })
                        .response
                        .on_hover_text("Patient coordinates; seeded from the crosshair");
                    }
                    Source::Dose => {
                        ui.horizontal_wrapped(|ui| {
                            ui.label("At least");
                            if d.dose_absolute {
                                ui.add(
                                    egui::DragValue::new(&mut d.dose_gy)
                                        .speed(0.5)
                                        .range(0.0..=1000.0)
                                        .suffix(" Gy"),
                                );
                            } else {
                                ui.add(
                                    egui::DragValue::new(&mut d.dose_pct)
                                        .speed(1.0)
                                        .range(0.0..=200.0)
                                        .suffix(" %"),
                                );
                                ui.label(
                                    egui::RichText::new(format!(
                                        "of {reference:.1} Gy = {:.1} Gy",
                                        reference * d.dose_pct / 100.0
                                    ))
                                    .weak(),
                                );
                            }
                            ui.checkbox(&mut d.dose_absolute, "absolute");
                        });
                        ui.label(
                            egui::RichText::new(
                                "The displayed dose, sampled the way the isodose lines \
                                 are, so the structure agrees with what is on screen.",
                            )
                            .weak(),
                        );
                    }
                }

                if d.source != Source::Shape {
                    ui.horizontal_wrapped(|ui| {
                        ui.checkbox(&mut d.keep_largest, "keep the largest piece");
                        ui.checkbox(&mut d.fill_holes, "fill cavities");
                    });
                }

                ui.separator();
                ui.horizontal_wrapped(|ui| {
                    ui.label("Name:");
                    ui.add(egui::TextEdit::singleline(&mut d.name).desired_width(150.0));
                    egui::ComboBox::from_id_salt("newroi_type")
                        .selected_text(&d.roi_type)
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for t in ["ORGAN", "PTV", "CTV", "GTV", "EXTERNAL", "AVOIDANCE"] {
                                ui.selectable_value(&mut d.roi_type, t.to_string(), t);
                            }
                        });
                });
                ui.horizontal(|ui| {
                    if ui
                        .button("▶ Create")
                        .on_hover_text("On the displayed image series")
                        .clicked()
                    {
                        create = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
                if let Some(status) = &d.status {
                    ui.separator();
                    ui.weak(status);
                }
            },
        );
        if let Some(s) = switch {
            self.open_newroi_dialog(s);
            return;
        }
        if create {
            self.create_newroi();
        }
        if !open || close {
            self.newroi_dialog = None;
        }
    }
}
