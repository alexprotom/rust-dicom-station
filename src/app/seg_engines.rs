//! What the three network-driven segmentation tools share.
//!
//! Auto-segmentation (TotalSegmentator), prompt segmentation (SegVol) and
//! slice propagation (MedSAM2) are different conversations with the user,
//! but they are the same *kind* of tool: a window per dataset with the same
//! bones - a one-line description, the tool's own inputs, an `Options`
//! section holding the compute device and the model folder, one line about
//! the weights' licence, and a button row that turns into a progress row
//! while the network runs. This module holds those bones, so the three
//! windows look and behave alike, and the plumbing every run needs: the
//! model folder, the check that the dataset is still the one the run
//! started on, and landing a mask as an editable [`Segmentation`].

use crate::models::Engine;
use crate::nn::device::DevicePref;

use super::*;

/// A background run of one engine: the slot it works on, and its outcome.
pub(super) type SegJob<T> = Job<(usize, anyhow::Result<T>)>;

/// Which tool a [`ToolInfo`] stands for - what a menu entry or a sidebar
/// button opens, matched exhaustively in [`ViewerApp::open_tool`] so a new
/// tool cannot fall through to the wrong window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ToolId {
    NewRoi,
    Contours,
    Details,
    Combine,
    Body,
    Autoseg,
    PromptSeg,
    SliceProp,
    Motion,
}

/// The glyph and name of each tool, in one place.
pub(super) struct ToolInfo {
    pub id: ToolId,
    pub glyph: &'static str,
    pub name: &'static str,
}

/// Every glyph in this file - and in the rest of the interface - has to be
/// one egui's bundled fonts actually carry, or it comes out as an empty box
/// on the user's screen. The microscope stands for the tool that examines
/// the whole scan by itself; a robot would have read better and does not
/// render (see the `glyphs` test module).
pub(super) const AUTOSEG: ToolInfo = ToolInfo {
    id: ToolId::Autoseg,
    glyph: "🔬",
    name: "Auto-segmentation",
};
/// The speech balloon is the prompt: this is the tool you tell what to find.
pub(super) const PROMPT_SEG: ToolInfo = ToolInfo {
    id: ToolId::PromptSeg,
    glyph: "💬",
    name: "Prompt segmentation",
};
pub(super) const SLICE_PROP: ToolInfo = ToolInfo {
    id: ToolId::SliceProp,
    glyph: "⏩",
    name: "Slice propagation",
};
/// The fourth tool. Its glyph is a person because that is what it outlines,
/// and because it is one of the few figures egui's bundled emoji font
/// actually carries.
pub(super) const BODY_CONTOUR: ToolInfo = ToolInfo {
    id: ToolId::Body,
    glyph: "👤",
    name: "Body contour",
};
/// The fifth tool, and the only one with no network behind it at all.
pub(super) const COMBINE: ToolInfo = ToolInfo {
    id: ToolId::Combine,
    glyph: "∪",
    name: "Combine structures",
};
/// The sixth tool: the 4D motion / ITV pipeline. A chart, because what it
/// produces is the motion curves and volumes (and the glyph is covered by
/// egui's bundled emoji fonts, which the quarter-clocks are not).
pub(super) const MOTION: ToolInfo = ToolInfo {
    id: ToolId::Motion,
    glyph: "📈",
    name: "4D motion / ITV",
};

/// The structure tools in the order the Tools menu and the segmentation
/// section's row list them, each with the one line its tooltip says. The 4D
/// motion tool is not among them: it works on a group, not on a structure.
pub(super) const TOOL_HINTS: &[(&ToolInfo, &str)] = &[
    (
        &super::struct_tools::NEW_ROI,
        "a structure from a grey-level window, a shape, the dose or the field of view",
    ),
    (
        &super::struct_tools::CONTOURS,
        "interpolation, tidying, moving - on the structure the contour tools edit",
    ),
    (
        &super::stats_win::DETAILS,
        "one row per structure: volume, grey levels, what the geometry costs, whether a \
         recipe still holds",
    ),
    (
        &BODY_CONTOUR,
        "outline the patient without the couch, the chair or the immobilisation (EXTERNAL)",
    ),
    (
        &COMBINE,
        "build one structure out of others: union, intersection, subtraction, margins",
    ),
    (
        &AUTOSEG,
        "automatic multi-organ segmentation (TotalSegmentator, 117 structures)",
    ),
    (
        &PROMPT_SEG,
        "segment whatever the crosshair points at - a box, a click or a structure name \
         (SegVol)",
    ),
    (
        &SLICE_PROP,
        "box a structure on one slice and follow it through the stack (MedSAM2)",
    ),
];

impl ToolInfo {
    /// `🔬 Auto-segmentation - dataset A`, the window title.
    pub fn title(&self, slot: usize) -> String {
        format!(
            "{} {} - dataset {}",
            self.glyph, self.name, SLOT_NAMES[slot]
        )
    }
    /// `🔬 Auto-segmentation results - dataset A`, a companion window.
    pub fn titled(&self, what: &str, slot: usize) -> String {
        format!(
            "{} {} {what} - dataset {}",
            self.glyph, self.name, SLOT_NAMES[slot]
        )
    }
    /// `🔬 Auto-segmentation`, the menu entry.
    ///
    /// The menu names the tool once. Which dataset it works on is a setting
    /// of the tool, chosen in its window by [`dataset_row`], because a menu
    /// that lists every tool twice is twice as long and no clearer.
    pub fn menu_entry(&self) -> String {
        format!("{} {}", self.glyph, self.name)
    }
}

impl ViewerApp {
    /// Open the tool's window - or, for the two that live in the Structure
    /// tools module, reveal its section - on `slot`.
    pub(super) fn open_tool(&mut self, id: ToolId, slot: usize) {
        match id {
            ToolId::NewRoi => self.open_newroi_dialog(slot),
            ToolId::Contours => self.open_contour_dialog(slot),
            ToolId::Details => self.open_stats_dialog(slot),
            ToolId::Combine => self.open_combine_dialog(slot, Vec::new()),
            ToolId::Body => self.open_body_dialog(slot),
            ToolId::Autoseg => self.open_autoseg_dialog(slot),
            ToolId::PromptSeg => self.open_segvol_dialog(slot),
            ToolId::SliceProp => self.open_medsam2_panel(slot),
            ToolId::Motion => self.open_motion_dialog(slot, None),
        }
    }
}

/// The dataset row every tool window starts with.
///
/// Returns the newly chosen slot, which the window answers by reopening
/// itself on that dataset: what a tool carries - the structures picked, the
/// 4D group, the box drawn on a slice - belongs to one dataset and cannot
/// follow it to the other. With one dataset loaded there is nothing to
/// choose and the row is not drawn.
pub(super) fn dataset_row(
    ui: &mut egui::Ui,
    slot: usize,
    has: [bool; 2],
    enabled: bool,
) -> Option<usize> {
    if has.iter().filter(|h| **h).count() < 2 {
        return None;
    }
    let mut picked = slot;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Dataset").strong());
        for (i, name) in SLOT_NAMES.iter().enumerate() {
            let r = ui.add_enabled(
                enabled && has[i],
                egui::RadioButton::new(picked == i, *name),
            );
            if r.clicked() {
                picked = i;
            }
        }
        if !enabled {
            ui.weak("(a run is in flight)");
        }
    });
    ui.separator();
    (picked != slot).then_some(picked)
}

/// One line every tool window ends its options with.
pub(super) const RESEARCH_NOTE: &str = "Research / QA use - not a medical device.";

/// The message shown when a run finishes on a dataset that was replaced
/// meanwhile.
pub(super) fn stale_result(tool: &ToolInfo) -> String {
    format!(
        "{} finished, but the dataset changed while it was running - the result was discarded.",
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
        // `has_volume` first: an empty volume has dims [0, 0, 0] and a
        // blank frame of reference, which would match a stale result's own
        // zeros and let it land on a dataset that shows nothing.
        self.slots[slot].has_volume()
            && self.slots[slot]
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

    /// [`Self::add_segmentation`] keeping a colour the caller already has -
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

    /// The tool that is running on `slot`, with its progress - for the
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
        if let Some(job) = self
            .combine_job
            .as_ref()
            .filter(|_| self.combine_slot == slot)
        {
            return Some((&COMBINE, &job.progress));
        }
        if let Some(job) = self
            .motion_job
            .as_ref()
            .filter(|_| self.motion_slot == slot)
        {
            return Some((&MOTION, &job.progress));
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
                DevicePref::Gpu => "Any GPU via wgpu (Vulkan / DX12 / Metal) - no CUDA needed",
                DevicePref::Cpu => "Every core, no GPU",
            };
            ui.radio_value(pref, p, p.label()).on_hover_text(hint);
        }
    });
}

/// `Model folder: [ ... ] 📁` - the root every engine downloads into.
/// Returns true when the browse button was clicked.
pub(super) fn models_root_row(ui: &mut egui::Ui, models_dir: &mut String) -> bool {
    let mut browse = false;
    ui.horizontal(|ui| {
        ui.label("Model folder:");
        ui.add(egui::TextEdit::singleline(models_dir).desired_width(220.0))
            .on_hover_text(format!(
                "Root folder of all downloaded weights; blank means the default, {}",
                models::default_root().display()
            ));
        if tip_button(ui, "📁", "Choose the model folder") {
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

/// The tool window each engine belongs to - the glyph and name the model
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
             partly non-commercial - downloaded to this machine at your request only, \
             never redistributed.",
            true,
        ),
        Engine::MedSam2 => (
            "Weights: CC-BY-SA-4.0 with a 'research and education only' model card - \
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
    ui.label(if msg.is_empty() { "Working" } else { &msg });
    ui.button("Cancel").clicked()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_menu_entries_and_buttons_follow_one_pattern() {
        assert_eq!(AUTOSEG.title(0), "🔬 Auto-segmentation - dataset A");
        assert_eq!(
            AUTOSEG.titled("results", 1),
            "🔬 Auto-segmentation results - dataset B"
        );
        assert_eq!(PROMPT_SEG.menu_entry(), "💬 Prompt segmentation");
        assert_eq!(SLICE_PROP.menu_entry(), "⏩ Slice propagation");
        assert_eq!(BODY_CONTOUR.menu_entry(), "👤 Body contour");
        let mut glyphs = vec![
            AUTOSEG.glyph,
            PROMPT_SEG.glyph,
            SLICE_PROP.glyph,
            BODY_CONTOUR.glyph,
            COMBINE.glyph,
            MOTION.glyph,
        ];
        glyphs.sort();
        glyphs.dedup();
        assert_eq!(glyphs.len(), 6, "every tool has its own glyph");
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
