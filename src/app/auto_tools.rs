//! *Modules ▶ Structure auto tools*: the four tools that find a structure
//! by themselves, as one section of the modules panel.
//!
//! *Body contour* (threshold and morphology, or a network), *Auto-
//! segmentation* (TotalSegmentator), *Prompt segmentation* (SegVol) and
//! *Slice propagation* (MedSAM2) used to be four windows with the same
//! bones - a description, the tool's inputs, an *Options* fold with the
//! compute device and the model folder, the licence line, a run button that
//! turns into a progress row. Here they are four foldable sections under one
//! dataset row, drawn by the same code that drew the windows
//! (`body_win.rs`, `dialogs.rs`, `prompt_seg.rs`, `box_seg.rs`), so the
//! panel and the runs behave exactly as before; only the window is gone.
//!
//! The module works on one dataset; every section's state is re-targeted
//! when that changes, unless a run on the old dataset is still in flight.

use super::*;

/// The four sections of the module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum AutoSection {
    Body,
    Autoseg,
    PromptSeg,
    SliceProp,
}

/// The module's state: the dataset the sections act on.
#[derive(Default)]
pub(super) struct AutoTools {
    pub slot: usize,
}

impl ViewerApp {
    /// The whole module: the dataset row, then the four sections.
    pub(super) fn auto_tools_section(&mut self, ui: &mut egui::Ui) {
        let title = egui::RichText::new("Structure auto tools").strong();
        if !self.any_volume() {
            egui::CollapsingHeader::new(title)
                .default_open(true)
                .show(ui, |ui| {
                    ui.weak("Load a dataset with an image volume to segment");
                });
            ui.separator();
            return;
        }
        if !self.slots[self.auto.slot].has_volume() {
            self.auto.slot = self.first_volume_slot();
        }
        // A run pins the module to the dataset it started on: the sections
        // would otherwise be re-targeted under it.
        let busy = self.running_tool(self.auto.slot).is_some()
            || self.running_tool(1 - self.auto.slot).is_some();
        let mut new_slot = None;
        egui::CollapsingHeader::new(title)
            .default_open(true)
            .show(ui, |ui| {
                new_slot = seg_engines::dataset_row(ui, self.auto.slot, self.volume_slots(), !busy);
                for (section, info) in [
                    (AutoSection::Body, &BODY_CONTOUR),
                    (AutoSection::Autoseg, &AUTOSEG),
                    (AutoSection::PromptSeg, &PROMPT_SEG),
                    (AutoSection::SliceProp, &SLICE_PROP),
                ] {
                    egui::CollapsingHeader::new(info.menu_entry())
                        .default_open(false)
                        .show(ui, |ui| match section {
                            AutoSection::Body => self.body_section(ui),
                            AutoSection::Autoseg => self.autoseg_section(ui),
                            AutoSection::PromptSeg => self.segvol_section(ui),
                            AutoSection::SliceProp => self.medsam2_section(ui),
                        });
                }
            });
        ui.separator();
        if let Some(s) = new_slot {
            self.auto.slot = s;
        }
    }
}
