//! Where the downloaded network weights live.
//!
//! Every engine fetches its published checkpoint on first use and keeps it,
//! together with the converted `safetensors` cache, under one root:
//!
//! ```text
//! <folder of the executable>/models/
//!   totalsegmentator/   the nnU-Net models, one sub-folder per model
//!   segvol/             pytorch_model.bin, vocab.json, merges.txt, cache
//!   medsam2/            one .pt per fine-tune, with its cache beside it
//! ```
//!
//! The root can be moved (the interface has one "Model folder" field, kept
//! in the settings file); the engine sub-folders are fixed, so the installer
//! and the headless examples find the same files the viewer does. Older
//! installations kept three folders beside the executable
//! (`autoseg_models/`, `segvol_model/`, `medsam2_model/`);
//! [`migrate_legacy_layout`] moves them into place once.

use std::path::{Path, PathBuf};

use crate::settings::app_dir;

/// Name of the root folder.
pub const DIR_NAME: &str = "models";

/// The engines that download weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// Automatic multi-organ segmentation (nnU-Net models).
    TotalSegmentator,
    /// Prompt-driven segmentation.
    SegVol,
    /// Slice propagation.
    MedSam2,
}

impl Engine {
    pub const ALL: [Engine; 3] = [Engine::TotalSegmentator, Engine::SegVol, Engine::MedSam2];

    /// Sub-folder of the root.
    pub fn subdir(self) -> &'static str {
        match self {
            Engine::TotalSegmentator => "totalsegmentator",
            Engine::SegVol => "segvol",
            Engine::MedSam2 => "medsam2",
        }
    }

    /// The folder the engine used before the layout was unified.
    fn legacy_dir(self) -> &'static str {
        match self {
            Engine::TotalSegmentator => "autoseg_models",
            Engine::SegVol => "segvol_model",
            Engine::MedSam2 => "medsam2_model",
        }
    }
}

/// The default root: `models/` next to the executable.
pub fn default_root() -> PathBuf {
    app_dir().join(DIR_NAME)
}

/// An engine's folder under `root`.
pub fn engine_dir(root: &Path, engine: Engine) -> PathBuf {
    root.join(engine.subdir())
}

/// The root a settings field names, or the default when it is blank.
pub fn root_from_setting(text: &str) -> PathBuf {
    let t = text.trim();
    if t.is_empty() {
        default_root()
    } else {
        PathBuf::from(t)
    }
}

/// Move the pre-unification folders beside the executable into `root`.
///
/// Best-effort and idempotent: a legacy folder is renamed only when the
/// engine's new folder does not exist yet, a rename that fails (another
/// volume, permissions) is simply left alone, and nothing is ever deleted.
/// Returns the engines that were moved.
pub fn migrate_legacy_layout(root: &Path) -> Vec<Engine> {
    let app = app_dir();
    let mut moved = Vec::new();
    for engine in Engine::ALL {
        let old = app.join(engine.legacy_dir());
        let new = engine_dir(root, engine);
        if !old.is_dir() || new.exists() {
            continue;
        }
        if std::fs::create_dir_all(root).is_ok() && std::fs::rename(&old, &new).is_ok() {
            moved.push(engine);
        }
    }
    moved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_has_its_own_folder_under_the_root() {
        let root = PathBuf::from("/opt/rds/models");
        let dirs: Vec<PathBuf> = Engine::ALL.iter().map(|e| engine_dir(&root, *e)).collect();
        for d in &dirs {
            assert!(d.starts_with(&root));
        }
        let mut names: Vec<&str> = Engine::ALL.iter().map(|e| e.subdir()).collect();
        names.dedup();
        assert_eq!(names.len(), 3);
        assert_eq!(dirs[1], root.join("segvol"));
    }

    #[test]
    fn a_blank_setting_means_the_default_root() {
        assert_eq!(root_from_setting("   "), default_root());
        assert_eq!(root_from_setting(" D:/w "), PathBuf::from("D:/w"));
        assert!(default_root().ends_with(DIR_NAME));
    }
}
