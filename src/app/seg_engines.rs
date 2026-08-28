//! What the three network-driven segmentation tools share.
//!
//! Auto-segmentation (TotalSegmentator), prompt segmentation (SegVol) and
//! slice propagation (MedSAM2) are different conversations with the user,
//! but they are the same *kind* of tool: a window per dataset with the same
//! bones — a one-line description, the tool's own inputs, an `Options`
//! section holding the compute device and the model folder, one line about
//! the weights' licence, and a button row that turns into a progress row
//! while the network runs. This module holds those bones, so the three
//! windows look and behave alike, and the plumbing every run needs: the
//! model folder, the check that the dataset is still the one the run
//! started on, and landing a mask as an editable [`Segmentation`].

use std::path::PathBuf;

use crate::models::{self, Engine};
use crate::nn::device::DevicePref;

use super::*;

/// A background run of one engine: the slot it works on, and its outcome.
pub(super) type SegJob<T> = Job<(usize, anyhow::Result<T>)>;

/// The glyph, name and menu wording of each tool, in one place.
pub(super) struct ToolInfo {
    pub glyph: &'static str,
    pub name: &'static str,
    /// What the menu and the sidebar say the tool does to a dataset.
    pub verb: &'static str,
}

pub(super) const AUTOSEG: ToolInfo = ToolInfo {
    glyph: "🤖",
    name: "Auto-segmentation",
    verb: "Auto-segment",
};
pub(super) const PROMPT_SEG: ToolInfo = ToolInfo {
    glyph: "🧠",
    name: "Prompt segmentation",
    verb: "Prompt-segment",
};
pub(super) const SLICE_PROP: ToolInfo = ToolInfo {
    glyph: "⏩",
    name: "Slice propagation",
    verb: "Propagate through",
};
/// The fourth tool. Its glyph is a person because that is what it outlines,
/// and because it is one of the few figures egui's bundled emoji font
/// actually carries.
pub(super) const BODY_CONTOUR: ToolInfo = ToolInfo {
    glyph: "👤",
    name: "Body contour",
    verb: "Body-contour",
};

impl ToolInfo {
    /// `🤖 Auto-segmentation — dataset A`, the window title.
    pub fn title(&self, slot: usize) -> String {
        format!(
            "{} {} — dataset {}",
            self.glyph, self.name, SLOT_NAMES[slot]
        )
    }
    /// `🤖 Auto-segmentation results — dataset A`, a companion window.
    pub fn titled(&self, what: &str, slot: usize) -> String {
        format!(
            "{} {} {what} — dataset {}",
            self.glyph, self.name, SLOT_NAMES[slot]
        )
    }
    /// `🤖 Auto-segment dataset A…`, the menu entry.
    pub fn menu_entry(&self, slot: usize) -> String {
        format!("{} {} dataset {}…", self.glyph, self.verb, SLOT_NAMES[slot])
    }
    /// `🤖 Auto…`, the small sidebar button.
    pub fn short_button(&self) -> String {
        let short = self.verb.split(['-', ' ']).next().unwrap_or(self.verb);
        format!("{} {short}", self.glyph)
    }
}

/// One line every tool window ends its options with.
pub(super) const RESEARCH_NOTE: &str = "Research / QA use — not a medical device.";

/// The message shown when a run finishes on a dataset that was replaced
/// meanwhile.
pub(super) fn stale_result(tool: &ToolInfo) -> String {
    format!(
        "{} finished, but the dataset changed while it was running — the result was discarded.",
        tool.name
    )
}

impl ViewerApp {
    /// The engine's folder under the model root the user chose.
    pub(super) fn engine_models_dir(&self, engine: Engine) -> PathBuf {
        models::engine_dir(&models::root_from_setting(&self.models_dir), engine)
    }

    /// Does `slot` still show the volume a run started on?
    pub(super) fn slot_still_shows(&self, slot: usize, dims: [usize; 3], uid: &str) -> bool {
        self.slots[slot]
            .study
            .as_ref()
            .is_some_and(|st| st.volume.dims == dims && st.volume.frame_of_reference_uid == uid)
    }

    /// The colour the next new segmentation gets, from the shared palette.
    pub(super) fn next_seg_color(&mut self) -> [u8; 3] {
        let color = segmentation::SEG_PALETTE[self.seg_counter % segmentation::SEG_PALETTE.len()];
        self.seg_counter += 1;
        color
    }

    /// Land a mask as a new, active, editable segmentation of `slot` and
    /// return its index.
    pub(super) fn add_segmentation(
        &mut self,
        slot: usize,
        name: String,
        dims: [usize; 3],
        mask: &[u8],
    ) -> usize {
        let color = self.next_seg_color();
        self.add_colored_segmentation(slot, name, color, dims, mask)
    }

    /// [`Self::add_segmentation`] keeping a colour the caller already has —
    /// a propagated structure should arrive in the colour it left in, not in
    /// the next one off the palette.
    pub(super) fn add_colored_segmentation(
        &mut self,
        slot: usize,
        name: String,
        color: [u8; 3],
        dims: [usize; 3],
        mask: &[u8],
    ) -> usize {
        if self.ensure_seg_series(slot).is_none() {
            return 0;
        }
        let s = &mut self.slots[slot];
        let Some(segs) = s.segs_mut() else { return 0 };
        segs.push(Segmentation::from_label_map(name, color, dims, mask, 1));
        let n = segs.len();
        s.active_seg = n - 1;
        s.active_seg
    }

    /// The tool that is running on `slot`, with its progress — for the
    /// sidebar, which shows one line whichever engine it is.
    pub(super) fn running_tool(&self, slot: usize) -> Option<(&ToolInfo, &Arc<Progress>)> {
        if let Some(job) = self
            .autoseg_job
            .as_ref()
            .filter(|_| self.autoseg_slot == slot)
        {
            return Some((&AUTOSEG, &job.progress));
        }
        if let Some(job) = self
            .segvol_job
            .as_ref()
            .filter(|_| self.segvol_slot == slot)
        {
            return Some((&PROMPT_SEG, &job.progress));
        }
        if let Some(job) = self
            .medsam2_job
            .as_ref()
            .filter(|_| self.medsam2.slot == slot)
        {
            return Some((&SLICE_PROP, &job.progress));
        }
        if let Some(job) = self.body_job.as_ref().filter(|_| self.body_slot == slot) {
            return Some((&BODY_CONTOUR, &job.progress));
        }
        None
    }
}

// ---- widgets shared by the tool windows ------------------------------------

/// `Compute:  Auto  GPU  CPU`
pub(super) fn device_row(ui: &mut egui::Ui, pref: &mut DevicePref) {
    ui.horizontal(|ui| {
        ui.label("Compute:");
        for p in DevicePref::ALL {
            let hint = match p {
                DevicePref::Auto => "Use the GPU when one is available, else the CPU",
                DevicePref::Gpu => "Any GPU via wgpu (Vulkan / DX12 / Metal) — no CUDA needed",
                DevicePref::Cpu => "Every core, no GPU",
            };
            ui.radio_value(pref, p, p.label()).on_hover_text(hint);
        }
    });
}

/// `Model folder: [ ... ] 📁` — the root every engine downloads into.
/// Returns true when the browse button was clicked.
pub(super) fn models_root_row(ui: &mut egui::Ui, models_dir: &mut String) -> bool {
    let mut browse = false;
    ui.horizontal(|ui| {
        ui.label("Model folder:");
        ui.add(egui::TextEdit::singleline(models_dir).desired_width(220.0))
            .on_hover_text(
                "Root folder of all downloaded weights; blank means `models/` next to the program",
            );
        if ui
            .button("📁")
            .on_hover_text("Choose the model folder")
            .clicked()
        {
            browse = true;
        }
    });
    browse
}

/// [`models_root_row`] with a hint naming the engine's sub-folder.
pub(super) fn models_dir_row(ui: &mut egui::Ui, models_dir: &mut String, engine: Engine) -> bool {
    let browse = models_root_row(ui, models_dir);
    ui.weak(format!(
        "This engine's files go to {}/{}/",
        models::DIR_NAME,
        engine.subdir()
    ));
    browse
}

/// The tool window each engine belongs to — the glyph and name the model
/// manager labels its rows with.
pub(super) fn tool_of(engine: Engine) -> &'static ToolInfo {
    match engine {
        Engine::TotalSegmentator => &AUTOSEG,
        Engine::SegVol => &PROMPT_SEG,
        Engine::MedSam2 => &SLICE_PROP,
    }
}

/// What an engine's published weights are licensed as, and whether that
/// needs highlighting. Kept here so the tool windows and the model manager
/// say the same thing.
pub(super) fn weights_licence(engine: Engine) -> (&'static str, bool) {
    match engine {
        Engine::TotalSegmentator => (
            "Weights: TotalSegmentator 'total' task (Apache-2.0), downloaded once from the \
             official GitHub release.",
            false,
        ),
        Engine::SegVol => (
            "Weights: no licence declaration in the model repository, training corpus \
             partly non-commercial — downloaded to this machine at your request only, \
             never redistributed.",
            true,
        ),
        Engine::MedSam2 => (
            "Weights: CC-BY-SA-4.0 with a 'research and education only' model card — \
             downloaded to this machine at your request only, never redistributed.",
            true,
        ),
    }
}

/// The licence line every tool ends with, before its buttons.
pub(super) fn licence_line(ui: &mut egui::Ui, weights: &str, warn: bool) {
    let text = format!("{weights} {RESEARCH_NOTE}");
    let mut rich = egui::RichText::new(text).small();
    if warn {
        rich = rich.color(warn_color(ui.visuals()));
    }
    ui.label(rich);
}

/// What a tool window shows in place of its buttons while a run is in
/// flight: the device, a bar, the message, and Cancel. Returns true when
/// Cancel was clicked.
pub(super) fn progress_row(ui: &mut egui::Ui, progress: &Progress) -> bool {
    let dev = progress.device();
    if !dev.is_empty() {
        ui.weak(format!("Running on: {dev}"));
    }
    ui.add(egui::ProgressBar::new(progress.frac()).show_percentage());
    let msg = progress.get();
    ui.label(if msg.is_empty() { "Working…" } else { &msg });
    ui.button("Cancel").clicked()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_menu_entries_and_buttons_follow_one_pattern() {
        assert_eq!(AUTOSEG.title(0), "🤖 Auto-segmentation — dataset A");
        assert_eq!(
            AUTOSEG.titled("results", 1),
            "🤖 Auto-segmentation results — dataset B"
        );
        assert_eq!(PROMPT_SEG.menu_entry(1), "🧠 Prompt-segment dataset B…");
        assert_eq!(SLICE_PROP.menu_entry(0), "⏩ Propagate through dataset A…");
        assert_eq!(AUTOSEG.short_button(), "🤖 Auto");
        assert_eq!(PROMPT_SEG.short_button(), "🧠 Prompt");
        assert_eq!(SLICE_PROP.short_button(), "⏩ Propagate");
        assert_eq!(BODY_CONTOUR.menu_entry(0), "👤 Body-contour dataset A…");
        let mut glyphs = vec![
            AUTOSEG.glyph,
            PROMPT_SEG.glyph,
            SLICE_PROP.glyph,
            BODY_CONTOUR.glyph,
        ];
        glyphs.sort();
        glyphs.dedup();
        assert_eq!(glyphs.len(), 4, "every tool has its own glyph");
    }

    #[test]
    fn every_engine_maps_to_a_tool_and_a_licence_line() {
        for engine in Engine::ALL {
            let tool = tool_of(engine);
            assert!(!tool.name.is_empty());
            let (note, _) = weights_licence(engine);
            assert!(note.starts_with("Weights:"), "{note}");
        }
        assert_eq!(tool_of(Engine::SegVol).glyph, PROMPT_SEG.glyph);
        assert!(!weights_licence(Engine::TotalSegmentator).1, "Apache-2.0");
        assert!(weights_licence(Engine::MedSam2).1, "research-only weights");
    }
}
