//! The small window that asks which workspace an action belongs to.
//!
//! *File ▸ Add DICOM folder*, *Add DICOM file(s)* and *Clear workspace* are
//! one entry each rather than one per workspace: the menu says what is being
//! done and this window says where. One entry that asks is shorter than a
//! menu that grows a line per workspace, and it is the same window whatever
//! the workspace count turns out to be.
//!
//! The file dialog comes after the choice, not before, so the destination is
//! settled while there is still something to cancel.

use super::*;

/// What the answer is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WsAsk {
    AddFolder,
    AddFiles,
    Clear,
}

impl WsAsk {
    fn title(self) -> &'static str {
        match self {
            WsAsk::AddFolder => "📂 Add DICOM folder",
            WsAsk::AddFiles => "📄 Add DICOM file(s)",
            WsAsk::Clear => "🗑 Clear workspace",
        }
    }

    fn blurb(self) -> &'static str {
        match self {
            WsAsk::AddFolder => {
                "Scan a folder and add its patients, studies and series to a workspace. \
                 What is already there stays loaded, and duplicates are skipped."
            }
            WsAsk::AddFiles => {
                "Open DICOM files directly - RT images, a structure set, a plan, single \
                 slices. They do not have to form an image volume, and they merge into \
                 the workspace exactly as a folder does."
            }
            WsAsk::Clear => {
                "Empty a workspace: its patients, studies, series and everything drawn \
                 on them. Nothing is written to disk and nothing else is touched."
            }
        }
    }

    /// Does this action need the workspace to hold something already?
    fn needs_content(self) -> bool {
        self == WsAsk::Clear
    }
}

/// The question, and the answer so far.
pub(super) struct WsPick {
    pub(super) ask: WsAsk,
    pub(super) slot: usize,
}

impl ViewerApp {
    /// Open the workspace question for one of the three actions.
    pub(super) fn ask_workspace(&mut self, ask: WsAsk) {
        // Start on a workspace the action can actually act on: the first
        // with something in it for *Clear*, the first empty one for a load,
        // and failing that the one on screen.
        let slot = match ask {
            WsAsk::Clear => (0..SLOT_NAMES.len())
                .find(|s| self.slots[*s].study.is_some())
                .unwrap_or(0),
            _ => (0..SLOT_NAMES.len())
                .find(|s| self.slots[*s].study.is_none())
                .unwrap_or(0),
        };
        self.ws_pick = Some(WsPick { ask, slot });
    }

    /// What a workspace holds, for the line under its button.
    fn workspace_summary(&self, slot: usize) -> String {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return "empty".to_string();
        };
        let patient = study
            .series
            .first()
            .map(|s| s.patient_name.clone())
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| "unnamed patient".to_string());
        format!("{patient} · {} series", study.series.len())
    }

    pub(super) fn workspace_pick_window(&mut self, ctx: &egui::Context) {
        let Some(mut pick) = self.ws_pick.take() else {
            return;
        };
        let ask = pick.ask;
        let holds: Vec<bool> = (0..SLOT_NAMES.len())
            .map(|s| self.slots[s].study.is_some())
            .collect();
        let summaries: Vec<String> = (0..SLOT_NAMES.len())
            .map(|s| self.workspace_summary(s))
            .collect();
        // Nothing to act on at all: say so rather than offering dead buttons.
        let any = !ask.needs_content() || holds.iter().any(|h| *h);
        if ask.needs_content() && !holds.get(pick.slot).copied().unwrap_or(false) {
            if let Some(s) = holds.iter().position(|h| *h) {
                pick.slot = s;
            }
        }
        let mut open = true;
        let mut close = false;
        let mut go = false;
        detach::tool_window(
            ctx,
            "workspace_pick",
            ask.title(),
            &mut open,
            detach::WinOpts::default(),
            |ui| {
                ui.set_max_width(440.0);
                ui.label(ask.blurb());
                ui.add_space(6.0);
                if !any {
                    ui.weak("No workspace holds anything yet.");
                    return;
                }
                ui.label("Workspace:");
                for (s, name) in SLOT_NAMES.iter().enumerate() {
                    let usable = !ask.needs_content() || holds[s];
                    ui.add_enabled_ui(usable, |ui| {
                        ui.horizontal(|ui| {
                            ui.radio_value(&mut pick.slot, s, *name);
                            ui.weak(&summaries[s]);
                        });
                    });
                }
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let label = match ask {
                        WsAsk::AddFolder => "📂 Choose folder",
                        WsAsk::AddFiles => "📄 Choose file(s)",
                        WsAsk::Clear => "🗑 Clear",
                    };
                    let usable = !ask.needs_content() || holds[pick.slot];
                    if ui.add_enabled(usable, egui::Button::new(label)).clicked() {
                        go = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            },
        );

        if go {
            let slot = pick.slot;
            // The question is answered, so it leaves the screen; the file
            // dialog is what happens next, and on Android that dialog is a
            // browser drawn in this same window, which needs the space.
            close = true;
            match ask {
                WsAsk::AddFolder => self.ask_folder(
                    &format!(
                        "Select DICOM folder to add to workspace {}",
                        SLOT_NAMES[slot]
                    ),
                    move |app, dir| {
                        app.open_workspace(slot);
                        app.start_load(slot, dir);
                    },
                ),
                WsAsk::AddFiles => self.ask_files(
                    &format!(
                        "Select DICOM file(s) to add to workspace {}",
                        SLOT_NAMES[slot]
                    ),
                    move |app, paths| {
                        app.open_workspace(slot);
                        app.start_load_files(slot, paths);
                    },
                ),
                WsAsk::Clear => self.clear_workspace(slot),
            }
        }
        if open && !close {
            self.ws_pick = Some(pick);
        }
    }

    /// Make sure a workspace past the first is on screen before anything is
    /// loaded into it.
    pub(super) fn open_workspace(&mut self, slot: usize) {
        if slot > 0 {
            self.comparison = true;
        }
    }

    /// Empty one workspace, whichever it is.
    ///
    /// The second one also leaves the screen, because a comparison with an
    /// empty half is a half-empty screen: emptying it is how it is closed.
    pub(super) fn clear_workspace(&mut self, slot: usize) {
        if slot == 0 {
            self.tree_clear_slot(0);
        } else {
            self.close_comparison();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_needs_something_to_clear_and_a_load_does_not() {
        assert!(WsAsk::Clear.needs_content());
        assert!(!WsAsk::AddFolder.needs_content());
        assert!(!WsAsk::AddFiles.needs_content());
    }

    #[test]
    fn every_action_names_itself_and_says_what_it_does() {
        for a in [WsAsk::AddFolder, WsAsk::AddFiles, WsAsk::Clear] {
            assert!(!a.title().is_empty(), "a window needs a title");
            assert!(
                a.blurb().len() > 40,
                "the window says what the action does, not just its name"
            );
        }
    }
}
