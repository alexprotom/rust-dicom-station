//! The *Structure editor* working on a segmentation rather than on contours.
//!
//! The editor's *Edit structure* section is built around an RT structure: a
//! stack of planar contours, which is what interpolation, point limits and
//! contour smoothing all mean something for. A segmentation is a mask, and
//! most of those operations have no meaning on one - but the operations that
//! *do* are the ones a painted segmentation needs most, and the program
//! already has every one of them for its own generators and auto tools.
//!
//! So this is the same section with the mask's own verbs in it: tidy it up
//! (largest piece, fill the holes), grow or shrink it by millimetres, empty
//! it, delete it, or turn it into an RT structure and go on editing it as
//! contours. Nothing here is a new algorithm; it is
//! [`crate::structops::Cleanup`], [`crate::morphology`] and
//! [`crate::segmentation::mask_to_roi`] given a place in the editor.

use super::*;

/// What the *Edit* section is editing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum EditSubject {
    /// A structure of the active RT structure set: contours.
    #[default]
    Structure,
    /// A segment of the active segmentation series: a mask.
    Segmentation,
}

/// What the segmentation editor asked for, applied once its borrows end.
#[derive(Clone, Copy)]
enum SegAct {
    KeepLargest,
    FillHoles,
    Grow,
    Shrink,
    Clear,
    Delete,
    ToStructure,
}

impl ViewerApp {
    /// Is there a segment to edit in this workspace?
    pub(super) fn has_editable_seg(&self, slot: usize) -> bool {
        let s = &self.slots[slot];
        s.segs().get(s.active_seg).is_some()
    }

    /// Which of the two the *Edit* section should act on.
    ///
    /// The switch is the user's when both are there, and follows whatever
    /// exists when only one is: an editor that insists on a structure in a
    /// workspace that holds nothing but segmentations is an editor that
    /// says "nothing is selected" and means "not that kind".
    pub(super) fn edit_subject(&self, slot: usize) -> EditSubject {
        let has_struct = self.edit_target(slot).is_some();
        let has_seg = self.has_editable_seg(slot);
        match (has_struct, has_seg) {
            (true, true) => self.tools.subject,
            (true, false) => EditSubject::Structure,
            (false, true) => EditSubject::Segmentation,
            (false, false) => self.tools.subject,
        }
    }

    /// The two-button switch, shown only where there is a choice to make.
    pub(super) fn edit_subject_row(&mut self, ui: &mut egui::Ui, slot: usize) {
        if !(self.edit_target(slot).is_some() && self.has_editable_seg(slot)) {
            return;
        }
        ui.horizontal(|ui| {
            ui.label("Edit:");
            for (s, label, hint) in [
                (
                    EditSubject::Structure,
                    "Structure",
                    "The selected structure of the RT structure set: contours, and \
                     everything that acts on contours",
                ),
                (
                    EditSubject::Segmentation,
                    "Segmentation",
                    "The selected segment of the segmentation series: a mask, and the \
                     operations a mask has",
                ),
            ] {
                if ui
                    .add(egui::Button::selectable(self.tools.subject == s, label).small())
                    .on_hover_text(hint)
                    .clicked()
                {
                    self.tools.subject = s;
                }
            }
        });
    }

    /// The *Edit* section for a segmentation.
    pub(super) fn seg_edit_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let Some((name, color, cm3, occupied, voxels)) = self.seg_summary(slot) else {
            ui.weak(
                "No segmentation is selected. Click one in the Segmentations list, or \
                 paint with a voxel tool and the first stroke creates one.",
            );
            return;
        };
        let mut act: Option<SegAct> = None;
        let mut radius = self.tools.seg_radius_mm;

        ui.horizontal_wrapped(|ui| {
            let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
            ui.painter().rect_filled(r, 2.0, theme::rgb(color));
            ui.label(egui::RichText::new(format!("✏ {name}")).strong());
        });
        ui.label(
            egui::RichText::new(format!(
                "{cm3:.1} cm³ · {occupied} slice(s) · {voxels} voxels"
            ))
            .weak(),
        );

        ui.label(egui::RichText::new("Tidy").strong())
            .on_hover_text(
                "The two things a painted or thresholded mask usually needs, and the same \
             pass the generators and the auto tools use",
            );
        ui.horizontal_wrapped(|ui| {
            if small_tip_button(
                ui,
                "Keep largest",
                "Throw away every connected piece but the biggest: the specks a \
                 threshold picks up, gone in one pass",
            ) {
                act = Some(SegAct::KeepLargest);
            }
            if small_tip_button(
                ui,
                "Remove holes",
                "Fill the enclosed holes slice by slice - the marrow inside a bone, the \
                 lumen inside a wall",
            ) {
                act = Some(SegAct::FillHoles);
            }
        });

        ui.label(egui::RichText::new("Grow and shrink").strong())
            .on_hover_text(
                "A true millimetre margin on the volume's own lattice, not a pixel \
                 count: anisotropic spacing is accounted for",
            );
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::DragValue::new(&mut radius)
                    .speed(0.5)
                    .range(0.1..=50.0)
                    .suffix(" mm"),
            );
            if small_tip_button(ui, "➕ Grow", "Dilate the mask by that margin") {
                act = Some(SegAct::Grow);
            }
            if small_tip_button(ui, "➖ Shrink", "Erode the mask by that margin") {
                act = Some(SegAct::Shrink);
            }
        });

        ui.label(egui::RichText::new("Whole segmentation").strong());
        ui.horizontal_wrapped(|ui| {
            if small_tip_button(
                ui,
                "▣ To RT structure",
                "Convert this mask to closed planar contours in the active structure \
                 set, and go on editing it there. The segmentation stays where it is",
            ) {
                act = Some(SegAct::ToStructure);
            }
            if small_tip_button(ui, "Clear", "Empty the mask, keeping the segmentation") {
                act = Some(SegAct::Clear);
            }
            if small_tip_button(ui, "🗑 Delete", "Remove this segmentation from the series") {
                act = Some(SegAct::Delete);
            }
        });

        self.tools.seg_radius_mm = radius;
        if let Some(a) = act {
            self.apply_seg_act(slot, a);
        }
    }

    /// Name, colour, volume, occupied slices and voxel count of the segment
    /// being edited.
    fn seg_summary(&self, slot: usize) -> Option<(String, [u8; 3], f64, usize, usize)> {
        let s = &self.slots[slot];
        let seg = s.segs().get(s.active_seg)?;
        let study = s.study.as_ref()?;
        let spacing = study.volume.spacing;
        let dims = seg.dims;
        let voxels = crate::morphology::count_set(&seg.mask);
        // How many slices of the stack the mask touches: the same count the
        // contour editor shows, so the two sections read alike.
        let per_slice = dims[0] * dims[1];
        let occupied = (0..dims[2])
            .filter(|k| {
                let from = k * per_slice;
                seg.mask[from..from + per_slice].iter().any(|v| *v != 0)
            })
            .count();
        Some((
            seg.name.clone(),
            seg.color,
            seg.volume_cm3(spacing),
            occupied,
            voxels,
        ))
    }

    fn apply_seg_act(&mut self, slot: usize, act: SegAct) {
        let Some(grid) = self.slots[slot].study.as_ref().map(|s| s.volume.grid()) else {
            return;
        };
        let radius = self.tools.seg_radius_mm as f64;
        let idx = self.slots[slot].active_seg;
        match act {
            SegAct::ToStructure => {
                self.seg_to_structure(slot);
                return;
            }
            SegAct::Delete => {
                if let Some(segs) = self.slots[slot].segs_mut() {
                    if idx < segs.len() {
                        segs.remove(idx);
                    }
                    let n = segs.len();
                    self.slots[slot].active_seg = idx.min(n.saturating_sub(1));
                }
                self.settings_gen += 1;
                return;
            }
            _ => {}
        }
        let Some(segs) = self.slots[slot].segs_mut() else {
            return;
        };
        let Some(seg) = segs.get_mut(idx) else { return };
        let dims = seg.dims;
        // Every one of these makes the next whole mask and hands it over:
        // `replace_mask` is what keeps the voxel count, the bounding box,
        // the redraw and the undo step honest, which a mask written into in
        // place does not.
        let next = match act {
            SegAct::KeepLargest | SegAct::FillHoles => {
                let cleanup = crate::structops::Cleanup {
                    keep_largest: matches!(act, SegAct::KeepLargest),
                    fill_holes: matches!(act, SegAct::FillHoles),
                    ..crate::structops::Cleanup::default()
                };
                let mut m = seg.mask.clone();
                cleanup.apply(&mut m, &grid, &crate::progress::Quiet);
                m
            }
            SegAct::Grow => crate::morphology::dilate_mm(&seg.mask, dims, grid.spacing, radius),
            SegAct::Shrink => crate::morphology::erode_mm(&seg.mask, dims, grid.spacing, radius),
            SegAct::Clear => vec![0; seg.mask.len()],
            SegAct::ToStructure | SegAct::Delete => unreachable!("handled above"),
        };
        seg.replace_mask(next);
        self.settings_gen += 1;
    }

    /// Copy the edited segmentation into the active structure set as closed
    /// planar contours, and make it the structure the editor works on.
    fn seg_to_structure(&mut self, slot: usize) {
        let Some(grid) = self.slots[slot].study.as_ref().map(|s| s.volume.grid()) else {
            return;
        };
        let idx = self.slots[slot].active_seg;
        let Some((name, color, roi)) = self.slots[slot].segs().get(idx).map(|seg| {
            (
                seg.name.clone(),
                seg.color,
                crate::segmentation::mask_to_roi(seg, &grid, 0),
            )
        }) else {
            return;
        };
        if roi.contours.is_empty() {
            self.error = Some(format!(
                "{name} has no voxels, so there is nothing to make contours from."
            ));
            return;
        }
        let Some(new) = self.new_roi(slot, Some(name.clone()), "ORGAN") else {
            return;
        };
        let set = self.slots[slot].active_structs;
        if let Some(r) = self.slots[slot].roi_mut(set, new) {
            r.contours = roi.contours;
            r.color = color;
        }
        // The editor follows what it just made, and to contours, because
        // that is what the new structure is.
        self.tools.subject = EditSubject::Structure;
        self.settings_gen += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_subject_follows_whatever_the_workspace_actually_has() {
        // Both present: the user's switch decides. Only one: that one, so
        // the editor never insists on a kind the workspace does not hold.
        for (has_struct, has_seg, switch, want) in [
            (
                true,
                true,
                EditSubject::Segmentation,
                EditSubject::Segmentation,
            ),
            (true, true, EditSubject::Structure, EditSubject::Structure),
            (
                true,
                false,
                EditSubject::Segmentation,
                EditSubject::Structure,
            ),
            (
                false,
                true,
                EditSubject::Structure,
                EditSubject::Segmentation,
            ),
        ] {
            let got = match (has_struct, has_seg) {
                (true, true) => switch,
                (true, false) => EditSubject::Structure,
                (false, true) => EditSubject::Segmentation,
                (false, false) => switch,
            };
            assert_eq!(
                got, want,
                "structure {has_struct}, segmentation {has_seg}, switch {switch:?}"
            );
        }
    }
}
