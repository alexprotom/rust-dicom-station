//! *Modules ▶ Structure auto tools*: the tools that find a structure by
//! themselves or from a prompt, as one section of the modules panel.
//!
//! *Body contour* (threshold and morphology, or a network), *Auto-
//! segmentation* (the model registry: TotalSegmentator, MRSegmentator,
//! lungmask, MONAI, CT-FM, VISTA-3D), *Prompt segmentation* (SegVol) and
//! *Slice propagation* (MedSAM2) used to be four windows with the same
//! bones - a description, the tool's inputs, an *Options* fold with the
//! compute device and the model folder, the licence line, a run button that
//! turns into a progress row. Here they are foldable sections under one
//! workspace row, drawn by the same code that drew the windows
//! (`body_win.rs`, `dialogs.rs`, `prompt_seg.rs`, `box_seg.rs`), so the
//! panel and the runs behave exactly as before; only the window is gone.
//! *Interactive segmentation* (nnInteractive, `interactive_seg.rs`) joined
//! them as a section from the start. What an automatic run found waits in
//! the *Results* fold at the end of its section, which unfolds when the run
//! lands (it was a window of its own). While it waits, the classes ticked
//! in it are drawn over the views of the workspace the run was on
//! ([`ViewerApp::autoseg_preview`]), so what *Add* will keep is seen before
//! it is kept.
//!
//! The module works on one workspace; every section's state is re-targeted
//! when that changes, unless a run on the old workspace is still in flight.

use super::*;

/// What the views draw of an automatic run waiting in its *Results*
/// fold: which phase of the run lies on the displayed volume, a colour
/// per ticked label, and a key that changes with the ticks.
pub(super) struct AutosegPreview {
    pub phase: usize,
    pub colors: Box<[Option<Color32>; 256]>,
    pub key: u64,
}

impl ViewerApp {
    /// The preview of the waiting run on `slot`'s displayed volume: the
    /// phase computed on the series on screen (by its UID, and on the same
    /// lattice), with the ticked classes coloured. `None` when no run waits,
    /// the module is off, nothing is ticked, or the series on screen is none
    /// of the run's phases.
    pub(super) fn autoseg_preview(&self, slot: usize) -> Option<AutosegPreview> {
        let p = self.autoseg_pending.as_ref()?;
        if p.slot != slot || !self.module_on(Module::AutoTools) {
            return None;
        }
        let study = self.slots[slot].study.as_ref()?;
        let shown = study
            .series
            .get(study.active_series)
            .map(|se| se.uid.as_str());
        let phase = p.phases.iter().position(|(info, r)| {
            r.dims == study.volume.dims && (p.phases.len() == 1 || shown == Some(&info.series_uid))
        })?;
        let mut colors: Box<[Option<Color32>; 256]> = Box::new([None; 256]);
        let mut key = mix(0x0A07_05E6, phase as u64);
        for (organ, _) in p.organs.iter().zip(&p.selected).filter(|(_, s)| **s) {
            let [r, g, b] = organ.color;
            colors[organ.label as usize] = Some(Color32::from_rgba_unmultiplied(r, g, b, 110));
            key = mix(key, organ.label as u64);
        }
        if key == mix(0x0A07_05E6, phase as u64) {
            return None;
        }
        Some(AutosegPreview { phase, colors, key })
    }
}

/// The sections of the module.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum AutoSection {
    Body,
    Autoseg,
    PromptSeg,
    SliceProp,
    Interactive,
}

/// The module's state: the workspace the sections act on.
#[derive(Default)]
pub(super) struct AutoTools {
    pub slot: usize,
    /// An automatic segmentation has just landed: unfold the module, its
    /// section and the section's *Results* fold once.
    pub reveal_results: bool,
}

impl ViewerApp {
    /// The whole module: the workspace row, then the sections.
    pub(super) fn auto_tools_section(&mut self, ui: &mut egui::Ui) {
        let title = egui::RichText::new("Structure auto tools").strong();
        if !self.any_volume() {
            egui::CollapsingHeader::new(title)
                .default_open(false)
                .show(ui, |ui| {
                    ui.weak("Load a workspace with an image volume to segment");
                });
            ui.separator();
            return;
        }
        if !self.slots[self.auto.slot].has_volume() {
            self.auto.slot = self.first_volume_slot();
        }
        // A run pins the module to the workspace it started on: the sections
        // would otherwise be re-targeted under it.
        let busy = (0..MAX_WORKSPACES).any(|s| self.running_tool(s).is_some());
        let mut new_slot = None;
        let reveal = self.auto.reveal_results.then_some(true);
        egui::CollapsingHeader::new(title)
            .default_open(false)
            .open(reveal)
            .show(ui, |ui| {
                new_slot =
                    seg_engines::workspace_row(ui, self.auto.slot, self.volume_slots(), !busy);
                for (section, info) in [
                    (AutoSection::Body, &BODY_CONTOUR),
                    (AutoSection::Autoseg, &AUTOSEG),
                    (AutoSection::PromptSeg, &PROMPT_SEG),
                    (AutoSection::SliceProp, &SLICE_PROP),
                    (AutoSection::Interactive, &INTERACTIVE_SEG),
                ] {
                    egui::CollapsingHeader::new(info.menu_entry())
                        .default_open(false)
                        .open(reveal.filter(|_| section == AutoSection::Autoseg))
                        .show(ui, |ui| match section {
                            AutoSection::Body => self.body_section(ui),
                            AutoSection::Autoseg => self.autoseg_section(ui),
                            AutoSection::PromptSeg => self.segvol_section(ui),
                            AutoSection::SliceProp => self.medsam2_section(ui),
                            AutoSection::Interactive => self.nni_section(ui),
                        });
                }
            });
        ui.separator();
        if let Some(s) = new_slot {
            self.auto.slot = s;
        }
    }
}
