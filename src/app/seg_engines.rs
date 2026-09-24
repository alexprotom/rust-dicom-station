//! What the three network-driven segmentation tools share.
//!
//! Auto-segmentation (TotalSegmentator), prompt segmentation (SegVol) and
//! slice propagation (MedSAM2) are different conversations with the user,
//! but they are the same *kind* of tool: a window per workspace with the same
//! bones - a one-line description, the tool's own inputs, an `Options`
//! section holding the compute device and the model folder, one line about
//! the weights' licence, and a button row that turns into a progress row
//! while the network runs. This module holds those bones, so the three
//! windows look and behave alike, and the plumbing every run needs: the
//! model folder, the check that the workspace is still the one the run
//! started on, and landing a mask as an editable [`Segmentation`].

use crate::loader::SeriesInfo;
use crate::models::Engine;
use crate::nn::device::DevicePref;
use crate::volume::Grid;
use crate::workflow::group::PhaseCopy;

use super::*;

/// A background run of one engine: the slot it works on, and its outcome.
pub(super) type SegJob<T> = Job<(usize, anyhow::Result<T>)>;

// ---- where a result lands, and on how many phases ---------------------------

/// What a tool's masks become once the run is over.
///
/// A segmentation series is the editable form - the brush works on it, the
/// 3D view meshes it live. An RT structure set is what a planning system
/// reads. Someone segmenting a 4DCT to plan on it wants the second and has
/// no use for the first; someone checking a network's answer wants the
/// first. So the tool asks, rather than always making the one and
/// sometimes also the other.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum OutputKind {
    /// Segments in a segmentation series bound to the image series.
    #[default]
    Segments,
    /// Contours in an RT structure set, and nothing else.
    Structures,
    /// Both: the segments, and contours made from them.
    Both,
}

impl OutputKind {
    pub const ALL: [OutputKind; 3] = [
        OutputKind::Segments,
        OutputKind::Structures,
        OutputKind::Both,
    ];
    pub fn label(self) -> &'static str {
        match self {
            OutputKind::Segments => "segments",
            OutputKind::Structures => "RT structures",
            OutputKind::Both => "both",
        }
    }
    pub fn segments(self) -> bool {
        matches!(self, OutputKind::Segments | OutputKind::Both)
    }
    pub fn structures(self) -> bool {
        matches!(self, OutputKind::Structures | OutputKind::Both)
    }
}

/// Where a tool's result goes, and how far the run reaches.
#[derive(Clone, Debug, Default)]
pub(super) struct ToolOutput {
    pub kind: OutputKind,
    /// The structure set contours are filed in: an index into the
    /// workspace's sets, or `None` for a new set called `new_label`. Only
    /// read when `kind` makes structures, and only for a run on the
    /// displayed series - a run over the phases files each phase's
    /// contours in that phase's own set.
    pub set: Option<usize>,
    /// The label of the set made when `set` is `None`; blank means the
    /// tool's own default.
    pub new_label: String,
    /// Run on every phase of the 4D group the displayed series belongs to,
    /// rather than on the displayed series alone.
    pub phases: bool,
}

/// `Run on:  (•) the displayed series  ( ) every phase of <group> (n)` -
/// drawn only when the displayed series is a phase of a 4D group, because
/// otherwise there is nothing to choose.
pub(super) fn scope_row(ui: &mut egui::Ui, phases: &mut bool, group: Option<&(String, usize)>) {
    let Some((name, n)) = group else {
        *phases = false;
        return;
    };
    // The group's name without the "(10 phases)" its detection adds - the
    // count in parentheses here is the number of phases the run visits.
    let name = super::propagate_win::strip_phase_count(name);
    ui.horizontal_wrapped(|ui| {
        ui.label("Run on:");
        ui.radio_value(phases, false, "the displayed series");
        ui.radio_value(phases, true, format!("every phase of {name} ({n})"))
            .on_hover_text(
                "Loads each phase in turn and runs the same tool on it. What it finds \
                 is filed on that phase - a segmentation series or a structure set of \
                 its own - under the same structure names on every phase, so the \
                 phases can be compared, played and propagated as one.",
            );
    });
}

/// The rows that say what the result becomes: the kind of output, and -
/// for contours on the displayed series - which structure set they go to,
/// an existing one or a new one with a name.
///
/// `sets` are the workspace's structure set labels; `phases` says the run
/// is over a 4D group, where each phase's own set is the destination and
/// there is no set to choose.
pub(super) fn output_rows(
    ui: &mut egui::Ui,
    out: &mut ToolOutput,
    sets: &[String],
    phases: bool,
    default_label: &str,
) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Output:");
        for k in OutputKind::ALL {
            let hint = match k {
                OutputKind::Segments => {
                    "Editable voxel masks in a segmentation series bound to the image \
                     series - what the brush and the segment tools work on. Exports as \
                     DICOM SEG."
                }
                OutputKind::Structures => {
                    "Contours in an RT structure set, and no segments. What a planning \
                     system reads; rides the RTSTRUCT export."
                }
                OutputKind::Both => "The segments, and contours made from them.",
            };
            ui.radio_value(&mut out.kind, k, k.label())
                .on_hover_text(hint);
        }
    });
    if !out.kind.structures() {
        return;
    }
    if phases {
        ui.weak(
            "Contours go to the structure set drawn on each phase, or to a new set \
             per phase when a phase has none.",
        );
        return;
    }
    if out.set.is_some_and(|i| i >= sets.len()) {
        out.set = None;
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Structure set:");
        let current = match out.set {
            Some(i) => sets[i].clone(),
            None => "➕ new structure set".to_string(),
        };
        egui::ComboBox::from_id_salt(ui.id().with("tool_out_set"))
            .selected_text(current)
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut out.set, None, "➕ new structure set");
                for (i, label) in sets.iter().enumerate() {
                    ui.selectable_value(&mut out.set, Some(i), format!("▣ {label}"));
                }
            });
        if out.set.is_none() {
            ui.label("named");
            ui.add(
                egui::TextEdit::singleline(&mut out.new_label)
                    .hint_text(default_label)
                    .desired_width(140.0),
            );
        }
    });
}

/// One phase an engine ran on: what to load, or the volume already in
/// memory when it is the displayed one.
pub(super) struct PhaseInput {
    pub label: String,
    pub series: SeriesInfo,
    pub preloaded: Option<Arc<Volume>>,
}

/// The identity of the volume one result was computed on: for a run over
/// the phases, which phase; for a run on the displayed series, that series
/// with `group` unset.
#[derive(Clone)]
pub(super) struct PhaseInfo {
    pub label: String,
    pub series_uid: String,
    pub study_uid: String,
    pub grid: Grid,
    /// The 4D group's name when this was one phase of a run over its
    /// phases.
    pub group: Option<String>,
}

impl PhaseInfo {
    /// The displayed volume of a workspace, for a run on it alone.
    pub fn displayed(study: &LoadedStudy) -> PhaseInfo {
        let se = study.series.get(study.active_series);
        PhaseInfo {
            label: String::new(),
            series_uid: se.map(|s| s.uid.clone()).unwrap_or_default(),
            study_uid: se.map(|s| s.study_uid.clone()).unwrap_or_default(),
            grid: study.volume.grid(),
            group: None,
        }
    }

    /// A result as one phase's copies, for [`ViewerApp::land_phases`].
    pub fn copy(&self, segs: Vec<Segmentation>, notes: Vec<String>) -> PhaseCopy {
        PhaseCopy {
            label: self.label.clone(),
            series_uid: self.series_uid.clone(),
            study_uid: self.study_uid.clone(),
            grid: self.grid.clone(),
            segs,
            notes,
        }
    }
}

/// Run one engine on every phase in turn, on the calling (worker) thread.
///
/// Each phase is loaded unless it is the displayed volume, which came along
/// in memory; the engine sees a progress handle that believes it owns the
/// whole bar, mapped onto that phase's slice of it and prefixed with the
/// phase's name, so its own messages read `Phase 30% (2/10): tile 4/12`.
/// The first error - a cancel included - ends the run; what earlier phases
/// made is not filed, because a 4D result with phases missing is not one
/// result.
pub(super) fn run_on_phases<R>(
    group: &str,
    phases: Vec<PhaseInput>,
    p: &Progress,
    run: impl Fn(&Volume, &Progress) -> anyhow::Result<R>,
) -> anyhow::Result<Vec<(PhaseInfo, R)>> {
    use anyhow::Context;
    let n = phases.len().max(1);
    let mut out = Vec::with_capacity(phases.len());
    for (i, ph) in phases.iter().enumerate() {
        p.set_outer(i as f32 / n as f32, 1.0 / n as f32);
        p.set_prefix(format!("Phase {} ({}/{n}): ", ph.label, i + 1));
        if p.cancelled() {
            anyhow::bail!(crate::progress::CANCELLED);
        }
        let vol = match &ph.preloaded {
            Some(v) => v.clone(),
            None => {
                p.set("loading");
                let (vol, _, _) = crate::loader::load_series_volume(&ph.series, p)
                    .with_context(|| format!("phase '{}' of {group}", ph.label))?;
                Arc::new(vol)
            }
        };
        let r = run(&vol, p).with_context(|| format!("phase '{}' of {group}", ph.label))?;
        out.push((
            PhaseInfo {
                label: ph.label.clone(),
                series_uid: ph.series.uid.clone(),
                study_uid: ph.series.study_uid.clone(),
                grid: vol.grid(),
                group: Some(group.to_string()),
            },
            r,
        ));
    }
    p.set_prefix("");
    p.set_outer(0.0, 1.0);
    Ok(out)
}

/// The glyph and name of each tool, in one place.
pub(super) struct ToolInfo {
    pub glyph: &'static str,
    pub name: &'static str,
}

/// Every glyph in this file - and in the rest of the interface - has to be
/// one egui's bundled fonts actually carry, or it comes out as an empty box
/// on the user's screen. The microscope stands for the tool that examines
/// the whole scan by itself; a robot would have read better and does not
/// render (see the `glyphs` test module).
pub(super) const AUTOSEG: ToolInfo = ToolInfo {
    glyph: "🔬",
    name: "Auto-segmentation",
};
/// The speech balloon is the prompt: this is the tool you tell what to find.
pub(super) const PROMPT_SEG: ToolInfo = ToolInfo {
    glyph: "💬",
    name: "Prompt segmentation",
};
pub(super) const SLICE_PROP: ToolInfo = ToolInfo {
    glyph: "⏩",
    name: "Slice propagation",
};
/// The fourth tool. Its glyph is a person because that is what it outlines,
/// and because it is one of the few figures egui's bundled emoji font
/// actually carries.
pub(super) const BODY_CONTOUR: ToolInfo = ToolInfo {
    glyph: "👤",
    name: "Body contour",
};
/// The fifth tool, and the only one with no network behind it at all.
pub(super) const COMBINE: ToolInfo = ToolInfo {
    glyph: "∪",
    name: "Combine structures",
};
/// The 4D motion / ITV pipeline. A chart, because what it
/// produces is the motion curves and volumes (and the glyph is covered by
/// egui's bundled emoji fonts, which the quarter-clocks are not).
pub(super) const MOTION: ToolInfo = ToolInfo {
    glyph: "📈",
    name: "Structure motion",
};

impl ToolInfo {
    /// `🔬 Auto-segmentation - workspace A`, the window title.
    pub fn title(&self, slot: usize) -> String {
        format!(
            "{} {} - workspace {}",
            self.glyph, self.name, SLOT_NAMES[slot]
        )
    }
    /// `🔬 Auto-segmentation results - workspace A`, a companion window.
    pub fn titled(&self, what: &str, slot: usize) -> String {
        format!(
            "{} {} {what} - workspace {}",
            self.glyph, self.name, SLOT_NAMES[slot]
        )
    }
    /// `🔬 Auto-segmentation`, the menu entry.
    ///
    /// The menu names the tool once. Which workspace it works on is a setting
    /// of the tool, chosen in its window by [`workspace_row`], because a menu
    /// that lists every tool twice is twice as long and no clearer.
    pub fn menu_entry(&self) -> String {
        format!("{} {}", self.glyph, self.name)
    }
}

/// The workspace row every tool window starts with.
///
/// Returns the newly chosen slot, which the window answers by reopening
/// itself on that workspace: what a tool carries - the structures picked, the
/// 4D group, the box drawn on a slice - belongs to one workspace and cannot
/// follow it to the other. With one workspace loaded there is nothing to
/// choose and the row is not drawn.
pub(super) fn workspace_row(
    ui: &mut egui::Ui,
    slot: usize,
    has: [bool; MAX_WORKSPACES],
    enabled: bool,
) -> Option<usize> {
    if has.iter().filter(|h| **h).count() < 2 {
        return None;
    }
    let mut picked = slot;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("Workspace").strong());
        // Only the workspaces there is something to run on, plus the one
        // the tool is on now: four letters with two of them dead would be a
        // row about what is missing rather than about what to pick.
        for (i, name) in SLOT_NAMES.iter().enumerate() {
            if !has[i] && i != slot {
                continue;
            }
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

/// The message shown when a run finishes on a workspace that was replaced
/// meanwhile.
pub(super) fn stale_result(tool: &ToolInfo) -> String {
    format!(
        "{} finished, but the workspace changed while it was running - the result was discarded.",
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
        // zeros and let it land on a workspace that shows nothing.
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

    /// The 4D group the displayed series of `slot` is a phase of, as
    /// (name, number of phases) - what the *Run on* row offers.
    pub(super) fn displayed_group(&self, slot: usize) -> Option<(String, usize)> {
        self.fourd_phases(slot)
            .map(|(name, idxs, _)| (name, idxs.len()))
    }

    /// The phases a run over `slot`'s group visits, in temporal order, with
    /// the displayed volume carried along in memory rather than re-read.
    pub(super) fn phase_inputs(&self, slot: usize) -> anyhow::Result<(String, Vec<PhaseInput>)> {
        let study = self.slots[slot]
            .study
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no study in workspace {}", SLOT_NAMES[slot]))?;
        let (name, _, _) = self
            .fourd_phases(slot)
            .ok_or_else(|| anyhow::anyhow!("the displayed series is not a phase of a 4D group"))?;
        let group = study
            .fourd_groups
            .iter()
            .find(|g| g.name == name && !g.dissolved)
            .ok_or_else(|| anyhow::anyhow!("the 4D group '{name}' is gone"))?;
        let displayed = study.series.get(study.active_series).map(|s| s.uid.clone());
        let phases = workflow::phases_of(group, &study.series)?
            .into_iter()
            .map(|(label, series)| {
                let preloaded = (displayed.as_deref() == Some(series.uid.as_str()))
                    .then(|| study.volume.clone());
                PhaseInput {
                    label,
                    series,
                    preloaded,
                }
            })
            .collect();
        Ok((name, phases))
    }

    /// The structure set labels of a workspace, for the *Structure set*
    /// combo of the output rows.
    pub(super) fn structure_set_labels(&self, slot: usize) -> Vec<String> {
        self.slots[slot]
            .study
            .as_ref()
            .map(|st| {
                st.structure_sets
                    .iter()
                    .map(|ss| {
                        if ss.label.is_empty() {
                            ss.file_name.clone()
                        } else {
                            ss.label.clone()
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The default target of a tool's contours: the active set when the
    /// workspace has one, else a new set.
    pub(super) fn default_set_target(&self, slot: usize) -> Option<usize> {
        let n = self.structure_set_labels(slot).len();
        (n > 0).then(|| self.slots[slot].active_structs.min(n - 1))
    }

    /// File masks made on the displayed volume of `slot` the way `out` says:
    /// as segments of the segmentation series bound to it, as contours of
    /// the chosen (or a new) structure set, or both. `roi_type` is the RT
    /// ROI Interpreted Type the contours get. Returns how many landed.
    pub(super) fn land_masks(
        &mut self,
        slot: usize,
        out: &ToolOutput,
        roi_type: &str,
        default_label: &str,
        masks: Vec<Segmentation>,
    ) -> usize {
        if masks.is_empty() {
            return 0;
        }
        let n = masks.len();
        let Some(grid) = self.slots[slot].study.as_ref().map(|st| st.volume.grid()) else {
            return 0;
        };
        // Contours first, from the masks as they are; the masks are moved
        // into the series afterwards, so nothing is cloned.
        if out.kind.structures() {
            let set = match out.set {
                Some(i) if i < self.structure_set_labels(slot).len() => Some(i),
                _ => {
                    let made = self.new_set(slot, SetKind::Structures);
                    if let Some(i) = made {
                        let label = out.new_label.trim();
                        let label = if label.is_empty() {
                            default_label
                        } else {
                            label
                        };
                        if let Some(ss) = self.slots[slot]
                            .study
                            .as_mut()
                            .and_then(|st| st.structure_sets.get_mut(i))
                        {
                            ss.label = label.to_string();
                        }
                    }
                    made
                }
            };
            let Some(set) = set else {
                self.error = Some("There is no image volume to draw the contours on.".into());
                return 0;
            };
            if self.set_locked(slot, set) {
                self.locked_notice(slot);
                return 0;
            }
            let rois: Vec<crate::rtstruct::Roi> = masks
                .iter()
                .map(|seg| {
                    let mut roi = segmentation::mask_to_roi(seg, &grid, 0);
                    roi.roi_type = roi_type.to_string();
                    roi
                })
                .collect();
            let StudySlot {
                study,
                active_structs,
                roi_visible,
                ..
            } = &mut self.slots[slot];
            if let Some(ss) = study.as_mut().and_then(|st| st.structure_sets.get_mut(set)) {
                for mut roi in rois {
                    roi.number = ss.rois.iter().map(|r| r.number).max().unwrap_or(0) + 1;
                    // A second run adds `liver (2)` rather than a second
                    // `liver`, as a propagation onto a phase does.
                    if ss.rois.iter().any(|r| r.name == roi.name) {
                        let base = roi.name.clone();
                        let mut k = 2;
                        while ss.rois.iter().any(|r| r.name == format!("{base} ({k})")) {
                            k += 1;
                        }
                        roi.name = format!("{base} ({k})");
                    }
                    ss.rois.push(roi);
                    if *active_structs == set {
                        roi_visible.push(true);
                    }
                }
                // The set that just gained structures is the one to look
                // at, with everything in it shown.
                if *active_structs != set {
                    *active_structs = set;
                    *roi_visible = vec![true; ss.rois.len()];
                }
            }
        }
        if out.kind.segments() {
            if self.ensure_seg_series(slot).is_none() {
                self.settings_gen += 1;
                return n;
            }
            let s = &mut self.slots[slot];
            let first = s.segs().len();
            if let Some(segs) = s.segs_mut() {
                segs.extend(masks);
            }
            s.active_seg = first;
        }
        self.settings_gen += 1;
        n
    }

    /// File what a run over a 4D group made on each phase: into the
    /// segmentation series bound to that phase (or a new one named after
    /// the group and the phase), as contours in the structure set drawn on
    /// that phase (or a new one), or both. Phases whose series are gone
    /// are skipped. Returns how many structures landed in all.
    pub(super) fn land_phases(
        &mut self,
        slot: usize,
        group_name: &str,
        kind: OutputKind,
        roi_type: &str,
        phases: Vec<PhaseCopy>,
    ) -> usize {
        let mut landed = 0usize;
        let mut notes: Vec<String> = Vec::new();
        for ph in &phases {
            notes.extend(ph.notes.iter().cloned());
            if ph.segs.is_empty() {
                continue;
            }
            let Some(study) = self.slots[slot].study.as_mut() else {
                return 0;
            };
            if !study.series.iter().any(|se| se.uid == ph.series_uid) {
                notes.push(format!(
                    "phase {}: its image series is no longer in the workspace",
                    ph.label
                ));
                continue;
            }
            if kind.structures() {
                let items: Vec<crate::propagate::Propagated> = ph
                    .segs
                    .iter()
                    .map(|seg| crate::propagate::Propagated {
                        name: seg.name.clone(),
                        color: seg.color,
                        mask: seg.mask.clone(),
                        voxels: seg.count,
                        source_cm3: 0.0,
                        result_cm3: 0.0,
                        mapped_cm3: 0.0,
                        source_surface_cm3: None,
                        rigid_residual_mm: None,
                    })
                    .collect();
                if let Some((_, names)) = workflow::group::land_in_structure_set_as(
                    study,
                    &ph.series_uid,
                    &ph.study_uid,
                    &ph.grid,
                    &items,
                    &format!("{group_name} {}", ph.label),
                    roi_type,
                ) {
                    landed += names.len();
                }
            }
            if kind.segments() {
                let existing = study
                    .seg_series
                    .iter()
                    .rposition(|sr| sr.referenced_series_uid == ph.series_uid);
                match existing {
                    Some(i) => {
                        let sr = &mut study.seg_series[i];
                        for seg in &ph.segs {
                            let mask = if sr.grid.matches(&ph.grid) {
                                seg.mask.clone()
                            } else {
                                crate::dicomseg::resample_mask(&seg.mask, &ph.grid, &sr.grid)
                            };
                            sr.segs.push(Segmentation::from_mask(
                                seg.name.clone(),
                                seg.color,
                                sr.grid.dims,
                                mask,
                            ));
                        }
                    }
                    None => study.seg_series.push(ph.seg_series(group_name)),
                }
                landed += ph.segs.len();
            }
        }
        if landed == 0 {
            self.error = Some(if notes.is_empty() {
                "nothing reached any phase".into()
            } else {
                notes.join("\n")
            });
            return 0;
        }
        if let Some(study) = self.slots[slot].study.as_mut() {
            study.warnings.extend(notes);
        }
        // The displayed phase's own series just gained segments (or was
        // just made): bring it onto the lattice the views index.
        self.rebind_seg_series(slot);
        if kind.segments() {
            let s = &mut self.slots[slot];
            if let Some(st) = s.study.as_ref() {
                let uid = st
                    .series
                    .get(st.active_series)
                    .map(|se| se.uid.clone())
                    .unwrap_or_default();
                if let Some(i) = st
                    .seg_series
                    .iter()
                    .rposition(|sr| sr.referenced_series_uid == uid)
                {
                    s.active_seg_series = i;
                    s.active_seg = st.seg_series[i].segs.len().saturating_sub(1);
                }
            }
        }
        self.settings_gen += 1;
        landed
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
    ui.horizontal_wrapped(|ui| {
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
    ui.horizontal_wrapped(|ui| {
        ui.label("Model folder:");
        ui.add(egui::TextEdit::singleline(models_dir).desired_width(160.0))
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
        assert_eq!(AUTOSEG.title(0), "🔬 Auto-segmentation - workspace A");
        assert_eq!(
            AUTOSEG.titled("results", 1),
            "🔬 Auto-segmentation results - workspace B"
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
