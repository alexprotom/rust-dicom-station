//! Which workspace data goes into, and the rules every such question
//! follows.
//!
//! *File ▸ Add DICOM folder*, *Add DICOM file(s)* and *Clear workspace* are
//! all submenus of the File menu rather than windows: each line names a
//! workspace and says what it holds, which is the thing worth knowing before
//! adding to one or emptying it, and the answer is one click rather than a
//! window to dismiss. The list is as long as the situation needs - the
//! workspaces on screen, plus one new letter while there is room
//! ([`transfer_targets`]).
//!
//! The file dialog comes after the choice, not before, so the destination is
//! settled while there is still something to cancel.

use super::*;

/// What the answer is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WsAsk {
    AddFolder,
    AddFiles,
}

impl WsAsk {
    /// The submenu's own line in the File menu.
    pub(super) fn menu_entry(self) -> &'static str {
        match self {
            WsAsk::AddFolder => "📂 Add DICOM folder",
            WsAsk::AddFiles => "📄 Add DICOM file(s)",
        }
    }

    /// What the submenu's line says on hover, before a workspace is picked.
    pub(super) fn blurb(self) -> &'static str {
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
        }
    }

    /// The file dialog's title, which names the destination.
    fn dialog_title(self, slot: usize) -> String {
        let ws = SLOT_NAMES[slot];
        match self {
            WsAsk::AddFolder => format!("Select DICOM folder to add to workspace {ws}"),
            WsAsk::AddFiles => format!("Select DICOM file(s) to add to workspace {ws}"),
        }
    }
}

/// Where a copy, a move or a load may go from `from`, given which
/// workspaces are on screen.
///
/// The rule the whole program follows: every other open workspace, and one
/// new letter while there is room. With A alone that is B and nothing else;
/// with A and B it is B and C; with all four open there is no new letter to
/// offer. `from` is excluded, and `usize::MAX` excludes nothing - which is
/// what "load this somewhere" asks for.
pub(super) fn transfer_targets(open: &[bool; MAX_WORKSPACES], from: usize) -> Vec<usize> {
    let mut out: Vec<usize> = (0..MAX_WORKSPACES)
        .filter(|s| *s != from && (*s == 0 || open[*s]))
        .collect();
    if let Some(new) = (1..MAX_WORKSPACES).find(|s| !open[*s]) {
        out.push(new);
    }
    out
}

impl ViewerApp {
    /// What a workspace holds, for the line beside its letter.
    pub(super) fn workspace_summary(&self, slot: usize) -> String {
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

    /// Answer a load question: put the workspace on screen and open the file
    /// dialog for it. The workspace appears first, so the row is there with
    /// its progress in it while the dialog is up.
    pub(super) fn load_into(&mut self, ask: WsAsk, slot: usize) {
        let title = ask.dialog_title(slot);
        match ask {
            WsAsk::AddFolder => self.ask_folder(&title, move |app, dir| {
                app.open_workspace(slot);
                app.start_load(slot, dir);
            }),
            WsAsk::AddFiles => self.ask_files(&title, move |app, paths| {
                app.open_workspace(slot);
                app.start_load_files(slot, paths);
            }),
        }
    }

    /// Put a workspace on screen before anything is loaded into it, so the
    /// row is there with its progress in it.
    pub(super) fn open_workspace(&mut self, slot: usize) {
        self.show_workspace(slot);
    }

    /// Empty one workspace, whichever it is.
    ///
    /// A workspace past the first also leaves the screen, because a row with
    /// an empty workspace in it is a row of nothing: emptying one is how it
    /// is closed. The letters of the others do not shift.
    pub(super) fn clear_workspace(&mut self, slot: usize) {
        if slot == 0 {
            self.tree_clear_slot(0);
        } else {
            self.close_workspace(slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[A, B, C, D]` as flags, for readable expectations below.
    fn open(flags: [bool; MAX_WORKSPACES]) -> [bool; MAX_WORKSPACES] {
        flags
    }

    #[test]
    fn a_transfer_offers_the_open_workspaces_and_one_new_letter() {
        // A alone: B, and nothing else - the menu does not list letters
        // nobody has asked for.
        assert_eq!(
            transfer_targets(&open([true, false, false, false]), 0),
            vec![1],
            "with A alone the only destination is a new B"
        );
        // A and B open: the other one, plus C.
        assert_eq!(
            transfer_targets(&open([true, true, false, false]), 0),
            vec![1, 2],
            "from A: B, and a new C"
        );
        assert_eq!(
            transfer_targets(&open([true, true, false, false]), 1),
            vec![0, 2],
            "from B: A, and a new C"
        );
        // Three open: the two others, plus D.
        assert_eq!(
            transfer_targets(&open([true, true, true, false]), 1),
            vec![0, 2, 3],
            "from B: A and C, and a new D"
        );
        // Four open: there is no fifth letter to offer.
        assert_eq!(
            transfer_targets(&open([true, true, true, true]), 2),
            vec![0, 1, 3],
            "with all four open, only the three that exist"
        );
        // Nothing to exclude: everything on screen, plus one.
        assert_eq!(
            transfer_targets(&open([true, true, false, false]), usize::MAX),
            vec![0, 1, 2],
            "a load may go anywhere that exists, plus one new letter"
        );
        // A gap left by a closed workspace is the letter offered next, so
        // closing B and copying again reuses B rather than opening C.
        assert_eq!(
            transfer_targets(&open([true, false, true, false]), 0),
            vec![2, 1],
            "the first free letter is the new one, gap or not"
        );
    }

    #[test]
    fn every_action_names_itself_and_says_what_it_does() {
        for a in [WsAsk::AddFolder, WsAsk::AddFiles] {
            assert!(!a.menu_entry().is_empty(), "a submenu needs a line");
            assert!(
                a.blurb().len() > 40,
                "the line says what the action does, not just its name"
            );
            // Every destination the menu can offer names itself in the file
            // dialog that follows, so a mis-click is visible before a folder
            // is chosen.
            for (slot, name) in SLOT_NAMES.iter().enumerate() {
                assert!(
                    a.dialog_title(slot).ends_with(name),
                    "the dialog title names the workspace"
                );
            }
        }
    }
}
