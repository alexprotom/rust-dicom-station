//! *Modules ▶ Image generation*: the tools that make one image out of
//! another, as one module of the right panel.
//!
//! *Image synthesis* (`synth_win.rs`) makes a synthetic CT from an MR or a
//! CBCT and files it as a CT series of the study; *DRR* (`drr_win.rs`)
//! forward-projects a CT onto a flat detector and files the radiograph under
//! the study's planar images, which the row's planar panes show. They are
//! foldable sections under one workspace row, the way *Structure auto tools*
//! holds the segmentation engines: the row says which workspace both work on,
//! and is locked while either has a run in flight, so a result always lands
//! on the workspace it was started from.

use super::*;

/// The DRR section's glyph and name.
pub(super) const DRR_TOOL: ToolInfo = ToolInfo {
    glyph: "☢",
    name: "Digitally reconstructed radiograph",
};

/// The module's state: the workspace its sections act on.
#[derive(Default)]
pub(super) struct ImageGen {
    pub slot: usize,
}

impl ViewerApp {
    /// Is a synthesis, an evaluation or a DRR running?
    fn generation_busy(&self) -> bool {
        self.synth_job.is_some() || self.synth_eval_job.is_some() || self.drr_job.is_some()
    }

    /// Point both sections at `slot`. The synthesis section's evaluation
    /// reference belonged to the other workspace; the DRR section follows by
    /// itself the next time it is drawn.
    fn generation_retarget(&mut self, slot: usize) {
        self.gen.slot = slot;
        if self.synth.slot != slot {
            self.synth.slot = slot;
            self.synth.eval_ref = None;
        }
    }

    /// The whole module: the workspace row, then the two sections.
    pub(super) fn image_generation_section(&mut self, ui: &mut egui::Ui) {
        let title = egui::RichText::new("Image generation").strong();
        if !self.any_volume() {
            egui::CollapsingHeader::new(title)
                .default_open(false)
                .show(ui, |ui| {
                    ui.weak(
                        "Load a workspace with an image volume: a CT to render a \
                         radiograph from, an MR or a CBCT to synthesize a CT from",
                    );
                });
            ui.separator();
            return;
        }
        let busy = self.generation_busy();
        if !busy {
            let slot = if self.slots[self.gen.slot].has_volume() {
                self.gen.slot
            } else {
                self.first_volume_slot()
            };
            if slot != self.gen.slot || slot != self.synth.slot {
                self.generation_retarget(slot);
            }
        }
        let mut new_slot = None;
        egui::CollapsingHeader::new(title)
            .default_open(false)
            .show(ui, |ui| {
                new_slot = workspace_row(ui, self.gen.slot, self.volume_slots(), !busy);
                egui::CollapsingHeader::new(synth_win::SYNTHESIS.menu_entry())
                    .default_open(false)
                    .show(ui, |ui| self.synthesis_tool(ui));
                egui::CollapsingHeader::new(DRR_TOOL.menu_entry())
                    .default_open(false)
                    .show(ui, |ui| self.drr_section(ui));
            });
        ui.separator();
        if let Some(s) = new_slot {
            self.generation_retarget(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sections_name_themselves_like_the_tools() {
        assert_eq!(
            DRR_TOOL.menu_entry(),
            "☢ Digitally reconstructed radiograph"
        );
        assert_eq!(synth_win::SYNTHESIS.menu_entry(), "🔮 Image synthesis");
    }
}
