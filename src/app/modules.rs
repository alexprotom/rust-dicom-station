//! The modules of the right panel, as one list.
//!
//! A module is a section of the right panel that works on the viewer itself
//! (its views, its panes, its tree) and opens no window of its own. Which
//! modules there are is [`Module`] (in the settings, which persist the
//! `module_* = on | off` lines); everything the viewer draws for one lives
//! here: its title, the id salt of its header and the word the "unfolded
//! sections" list remembers it by, its group in the *Modules* menu, the
//! menu's tooltip, and [`ViewerApp::module_section`], the one dispatch from
//! a module to the function that draws it. Adding a module is one variant of
//! the enum, one arm of each table here and the function that draws it.
//!
//! The sections also act on the views while they are on screen, through one
//! small facility: [`Focus`]. A section that is unfolded names what it is
//! about - the two structures a comparison measures, the row of a table
//! under the pointer - and the views draw those contours thicker, the way
//! the editor has always drawn the structure it edits. The focus is
//! cleared at the top of every pass of the panel and set again by whatever
//! is unfolded, so nothing stays lit once its section is folded or the
//! panel is hidden.

pub(super) use crate::settings::Module;

use super::combine::ItemRef;
use super::*;

/// The headings the *Modules* menu groups its lines under, each a plural:
/// what the modules under it work on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Group {
    Images,
    Structures,
    Doses,
    Workflows,
}

impl Group {
    pub(super) const ALL: [Group; 4] = [
        Group::Images,
        Group::Structures,
        Group::Doses,
        Group::Workflows,
    ];

    pub(super) fn title(self) -> &'static str {
        match self {
            Group::Images => "Images",
            Group::Structures => "Structures/Segmentations",
            Group::Doses => "Doses",
            Group::Workflows => "Workflows",
        }
    }
}

/// What the viewer draws for a module. A trait rather than inherent
/// methods because the enum belongs to the domain crate.
pub(super) trait ModuleUi {
    /// The header's title and the menu line.
    fn title(self) -> &'static str;
    /// The id salt of its header: the title, unless the section's header
    /// is salted with something else.
    fn header_salt(self) -> &'static str;
    /// The word `modules_open` remembers an unfolded section by.
    fn open_word(self) -> &'static str;
    fn group(self) -> Group;
    fn menu_tip(self) -> &'static str;
}

impl ModuleUi for Module {
    fn title(self) -> &'static str {
        match self {
            Module::ImageInfo => "Image information",
            Module::Playback => "Image playback",
            Module::Registration => "Image registration",
            Module::Simulation => "Image simulation",
            Module::Generation => "Image generation",
            Module::StructureEditor => "Structure editor",
            Module::AutoTools => "Structure auto tools",
            Module::Propagation => "Structure propagation",
            Module::Comparison => "Structure comparison",
            Module::Details => "Structure details",
            Module::Motion => "Structure motion",
            Module::DoseDisplay => "Dose display",
            Module::DoseAnalysis => "Dose analysis",
            Module::Dvh => "Dose-volume histograms",
            Module::Workflows => "Workflows",
        }
    }

    fn header_salt(self) -> &'static str {
        match self {
            Module::Propagation => "module_propagate",
            m => m.title(),
        }
    }

    fn open_word(self) -> &'static str {
        match self {
            Module::ImageInfo => "information",
            Module::Playback => "playback",
            Module::Registration => "registration",
            Module::Simulation => "simulation",
            Module::Generation => "generation",
            Module::StructureEditor => "editor",
            Module::AutoTools => "auto",
            Module::Propagation => "propagation",
            Module::Comparison => "comparison",
            Module::Details => "details",
            Module::Motion => "motion",
            Module::DoseDisplay => "dose_display",
            // The word it had as Dose estimation.
            Module::DoseAnalysis => "dose",
            Module::Dvh => "dvh",
            Module::Workflows => "workflows",
        }
    }

    fn group(self) -> Group {
        match self {
            Module::ImageInfo
            | Module::Playback
            | Module::Registration
            | Module::Simulation
            | Module::Generation => Group::Images,
            Module::StructureEditor
            | Module::AutoTools
            | Module::Propagation
            | Module::Comparison
            | Module::Details
            | Module::Motion => Group::Structures,
            Module::DoseDisplay | Module::DoseAnalysis | Module::Dvh => Group::Doses,
            Module::Workflows => Group::Workflows,
        }
    }

    fn menu_tip(self) -> &'static str {
        match self {
            Module::ImageInfo => {
                "What the displayed series is: voxel spacing, slice thickness and the gap \
                 between slices, the number of slices, the frame of reference and the \
                 acquisition settings, with whatever wants a second look named."
            }
            Module::Playback => {
                "The ▶ buttons on the viewports: how fast they run, what they do at the end \
                 of the range, and what a phase change carries with it. Holds the transport \
                 controls and the phase scrubber too."
            }
            Module::Registration => {
                "Align two workspaces: direction, method, region, parameters, landmarks, \
                 analysis, fusion and the deformation vector field. Needs two loaded \
                 workspaces to run."
            }
            Module::Simulation => {
                "Registration QA: apply a known rigid motion and Gaussian deformation to one \
                 workspace and generate the result into the other - the ground truth a \
                 registration can be measured against."
            }
            Module::Generation => {
                "Images made from images. Image synthesis: a synthetic CT from an MR or a \
                 CBCT (TotalSynth), added to the study as a CT series, and its evaluation \
                 against a CT of the same patient. DRR: digitally reconstructed radiographs \
                 of a CT, with an exact and an interpolating projector, filed under Planar \
                 images and shown in the row's planar panes."
            }
            Module::StructureEditor => {
                "Insert, edit and combine structures: empty structures and points, the \
                 generators, everything that acts on the selected structure (interpolation, \
                 tidying, moving, the type, the derived recipe), and the structure algebra."
            }
            Module::AutoTools => {
                "The tools that find a structure by themselves: the body contour, automatic \
                 multi-organ segmentation (TotalSegmentator) with the fold that picks what a \
                 run keeps, prompt segmentation (SegVol) and slice propagation (MedSAM2). \
                 Every network runs locally, in Rust."
            }
            Module::Propagation => {
                "Carry contours and segmentations from one workspace to the other through \
                 the active registration - globally, or refined on an enclosing structure \
                 first - or by relationship: at the same offset from a reference structure. \
                 Sits next to the registration that drives it."
            }
            Module::Comparison => {
                "Volumes, centroid offset, Dice, HD95, surface distances and the \
                 least-squares rigid offset of any two structures - within a workspace or \
                 across two. The two are drawn thicker in the views while the section is \
                 unfolded."
            }
            Module::Details => {
                "One row per structure: volume surface-based and voxels-based, the Dice \
                 against a reference, the grey levels inside it, what the geometry costs in \
                 slices and points, and whether a derived structure still matches its \
                 recipe. A click on a row selects the structure and centres the views on \
                 it. With CSV export."
            }
            Module::Motion => {
                "Register the reference phase of a 4D group to every other phase, carry the \
                 targets across, and measure their motion - trajectories, drift, \
                 correlations and the ITV. The finished runs are the section's second half, \
                 their charts in the row's chart pane."
            }
            Module::DoseDisplay => {
                "How the displayed dose grid is drawn over the images: colorwash, isodose                  lines or both, the opacity, the threshold below which nothing is drawn,                  and the isodose levels. Shared by every workspace; which grid is drawn                  is picked under Dose in the data tree."
            }
            Module::DoseAnalysis => {
                "Dmean, Dmin, Dmax, D_X% and V_X of every ticked structure against one dose, \
                 in a table that follows every edit of the structures."
            }
            Module::Dvh => {
                "Cumulative and differential DVHs of any structures against any loaded dose \
                 objects, with the metrics table, protocol constraint checking and CSV \
                 export. The plot is drawn in the row's chart pane, which can be maximized."
            }
            Module::Workflows => {
                "Run a saved workflow from the panel: pick it, give each folder node its \
                 folders, run it in the background or step by step in the viewer, and put \
                 what it leaves into the workspaces. The editor stays a window."
            }
        }
    }
}

/// What the unfolded sections are about, for the views to show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Focus {
    /// Structures or segments, each of one workspace: drawn thicker.
    Structures(Vec<(usize, ItemRef)>),
}

impl ViewerApp {
    /// Is this module switched on?
    pub(super) fn module_on(&self, m: Module) -> bool {
        self.modules.contains(&m)
    }

    /// Switch a module on and unfold the panel - what a tool does when it
    /// sends the user to its section.
    pub(super) fn show_module(&mut self, m: Module) {
        if self.modules.insert(m) {
            self.persist_settings();
        }
        self.right_open = true;
    }

    /// Switch a module on and unfold its section, so it is on screen at the
    /// next pass of the panel.
    pub(super) fn reveal_module(&mut self, m: Module) {
        self.show_module(m);
        self.reveal_section = Some(m);
    }

    /// Is any section of the modules panel switched on?
    pub(super) fn any_module(&self) -> bool {
        !self.modules.is_empty()
    }

    /// Draw one module's section.
    pub(super) fn module_section(&mut self, m: Module, ui: &mut egui::Ui) {
        match m {
            Module::ImageInfo => self.image_info_section(ui),
            Module::Playback => self.playback_section(ui),
            Module::Registration => self.registration_section(ui),
            Module::Simulation => self.simulation_section(ui),
            Module::Generation => self.image_generation_section(ui),
            Module::StructureEditor => self.structures_editor_section(ui),
            Module::AutoTools => self.auto_tools_section(ui),
            Module::Propagation => self.propagate_section(ui),
            Module::Comparison => self.compare_section(ui),
            Module::Details => self.stats_section(ui),
            Module::Motion => self.motion_section(ui),
            Module::DoseDisplay => self.dose_display_section(ui),
            Module::DoseAnalysis => self.dose_est_section(ui),
            Module::Dvh => self.dvh_section(ui),
            Module::Workflows => self.workflows_section(ui),
        }
    }

    /// Name structures the views should draw thicker this pass. Several
    /// sections may be unfolded at once; their structures add up.
    pub(super) fn focus_structures(&mut self, items: impl IntoIterator<Item = (usize, ItemRef)>) {
        let mut add: Vec<(usize, ItemRef)> = items.into_iter().collect();
        if add.is_empty() {
            return;
        }
        match &mut self.section_focus {
            Some(Focus::Structures(v)) => {
                add.retain(|x| !v.contains(x));
                v.extend(add);
            }
            None => self.section_focus = Some(Focus::Structures(add)),
        }
    }

    /// The structures of the active set of `slot` the views draw thicker:
    /// the one the contour tools edit, and whatever the unfolded sections
    /// are about.
    pub(super) fn focused_rois(&self, slot: usize) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .seg_tool
            .draws_contours()
            .then(|| self.edit_target(slot).map(|(_, r)| r))
            .flatten()
            .into_iter()
            .collect();
        let set = self.slots[slot].active_structs;
        if let Some(Focus::Structures(items)) = &self.section_focus {
            out.extend(
                items
                    .iter()
                    .filter(|(s, i)| *s == slot && i.kind == SetKind::Structures && i.set == set)
                    .map(|(_, i)| i.idx),
            );
        }
        out
    }

    /// The header of a module's section: a [`CollapsingState`] under the
    /// module's salt, opened once when a tool asked for it
    /// ([`Self::reveal_module`]), with room after the title.
    ///
    /// [`CollapsingState`]: egui::collapsing_header::CollapsingState
    pub(super) fn module_header(
        &mut self,
        ui: &mut egui::Ui,
        m: Module,
        extra: impl FnOnce(&mut Self, &mut egui::Ui),
        body: impl FnOnce(&mut Self, &mut egui::Ui),
    ) {
        let id = ui.make_persistent_id(m.header_salt());
        let mut state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
        if self.reveal_section == Some(m) {
            self.reveal_section = None;
            state.set_open(true);
            state.store(ui.ctx());
            // The panel scrolls to the section that was asked for.
            ui.scroll_to_cursor(Some(egui::Align::TOP));
        }
        let header = state.show_header(ui, |ui| {
            ui.label(egui::RichText::new(m.title()).strong());
            extra(self, ui);
        });
        header.body(|ui| body(self, ui));
        ui.separator();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every module has a title, a salt and a word of its own: two that
    /// shared a salt would fold and unfold together, two that shared a
    /// word would be restored together.
    #[test]
    fn every_module_is_told_apart() {
        let n = Module::ALL.len();
        let titles: std::collections::BTreeSet<_> = Module::ALL.iter().map(|m| m.title()).collect();
        let salts: std::collections::BTreeSet<_> =
            Module::ALL.iter().map(|m| m.header_salt()).collect();
        let words: std::collections::BTreeSet<_> =
            Module::ALL.iter().map(|m| m.open_word()).collect();
        assert_eq!((titles.len(), salts.len(), words.len()), (n, n, n));
        // The nine words the settings files of older builds carry.
        for w in [
            "information",
            "playback",
            "registration",
            "simulation",
            "generation",
            "editor",
            "auto",
            "propagation",
            "dose",
        ] {
            assert!(words.contains(w), "{w}");
        }
        for g in Group::ALL {
            assert!(Module::ALL.iter().any(|m| m.group() == g), "{g:?} is empty");
        }
    }
}
