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

use anyhow::{Context, Result};

use crate::progress::ProgressSink;
use crate::settings::{app_dir, default_models_dir};
use crate::{autoseg, medsam2, segvol};

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
    default_models_dir()
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

// ---------------------------------------------------------------------------
// The inventory: every model that can be downloaded, what it costs, and what
// it leaves on disk
// ---------------------------------------------------------------------------

/// Which download an inventory row stands for.
///
/// Each engine already describes its own published files —
/// [`autoseg::weights::ModelSpec`], [`segvol::weights::CHECKPOINT`],
/// [`medsam2::weights::Variant`]. This enum only records *which* of them a
/// row is, so [`ensure`] can hand the actual work straight back to the engine
/// that owns it rather than re-implementing three download paths here.
#[derive(Clone, Copy, Debug)]
pub enum AssetKind {
    /// One TotalSegmentator nnU-Net model, in its own sub-folder.
    Autoseg(autoseg::weights::ModelSpec),
    /// The SegVol network weights.
    SegVol,
    /// The CLIP byte-pair files SegVol's *text* prompts need.
    SegVolText,
    /// One MedSAM2 fine-tune.
    MedSam2(medsam2::weights::Variant),
}

/// One downloadable model as the model manager sees it.
#[derive(Clone)]
pub struct ModelAsset {
    pub kind: AssetKind,
    pub engine: Engine,
    /// Stable identity, unique across engines (`totalsegmentator/total_3mm`).
    pub key: String,
    /// Name shown in the interface.
    pub label: String,
    /// One line on what the model is for.
    pub detail: &'static str,
    /// Published bytes to fetch when nothing is on disk yet.
    pub download_bytes: u64,
    /// Folder holding the files, relative to the engine folder; empty means
    /// the engine folder itself.
    pub subdir: &'static str,
    /// With all of these present the model runs with no network access.
    pub ready: Vec<String>,
    /// Files the download leaves behind that nothing reads once the model is
    /// ready — the raw checkpoint, interrupted temporaries. Removing them
    /// frees disk without costing anything.
    pub spare: Vec<String>,
}

impl ModelAsset {
    /// The folder this model's files live in.
    pub fn dir(&self, root: &Path) -> PathBuf {
        let d = engine_dir(root, self.engine);
        if self.subdir.is_empty() {
            d
        } else {
            d.join(self.subdir)
        }
    }
}

/// What is on disk for one model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AssetStatus {
    /// Every file needed to run offline is present.
    pub ready: bool,
    /// Some — but not all — of them are.
    pub partial: bool,
    /// Bytes this model occupies, including the spare files.
    pub bytes: u64,
    /// Of those, bytes that could be freed without a re-download.
    pub spare_bytes: u64,
}

/// Every model of every engine, in the order the interface lists them.
pub fn inventory() -> Vec<ModelAsset> {
    let mut out = Vec::new();

    for spec in autoseg::weights::all_specs() {
        let detail = if spec.key == autoseg::weights::SPEC_3MM.key {
            "All 117 structures at 3 mm — the fast default."
        } else if spec.key == autoseg::weights::SPEC_6MM.key {
            "Coarse preview quality — the quickest look."
        } else {
            "Full-resolution sub-model; the five together are the reference quality."
        };
        out.push(ModelAsset {
            kind: AssetKind::Autoseg(spec),
            engine: Engine::TotalSegmentator,
            key: format!("{}/{}", Engine::TotalSegmentator.subdir(), spec.key),
            label: spec.label.to_string(),
            detail,
            download_bytes: spec.zip_bytes,
            subdir: spec.key,
            ready: vec![
                autoseg::weights::CACHE_NAME.to_string(),
                autoseg::weights::PLANS_NAME.to_string(),
            ],
            spare: vec![
                autoseg::weights::CHECKPOINT_TMP.to_string(),
                autoseg::weights::DOWNLOAD_TMP.to_string(),
            ],
        });
    }

    out.push(ModelAsset {
        kind: AssetKind::SegVol,
        engine: Engine::SegVol,
        key: format!("{}/weights", Engine::SegVol.subdir()),
        label: "SegVol — network weights".to_string(),
        detail: "3-D ViT image encoder, SAM-style prompt encoder and mask decoder \
                 (box and point prompts).",
        download_bytes: segvol::weights::CHECKPOINT.bytes,
        subdir: "",
        ready: vec![segvol::weights::CACHE_NAME.to_string()],
        spare: vec![segvol::weights::CHECKPOINT.name.to_string()],
    });
    out.push(ModelAsset {
        kind: AssetKind::SegVolText,
        engine: Engine::SegVol,
        key: format!("{}/tokenizer", Engine::SegVol.subdir()),
        label: "SegVol — CLIP tokenizer".to_string(),
        detail: "Byte-pair vocabulary and merge table; only *text* prompts need it.",
        download_bytes: segvol::weights::CLIP_FILES.iter().map(|f| f.bytes).sum(),
        subdir: "",
        ready: segvol::weights::CLIP_FILES
            .iter()
            .map(|f| f.name.to_string())
            .collect(),
        spare: Vec::new(),
    });

    for v in medsam2::weights::Variant::ALL {
        out.push(ModelAsset {
            kind: AssetKind::MedSam2(v),
            engine: Engine::MedSam2,
            key: format!("{}/{}", Engine::MedSam2.subdir(), v.key()),
            label: format!("MedSAM2 — {}", v.label()),
            detail: "SAM 2.1-T fine-tune with the memory bank; one architecture, \
                     one loader, a choice of training data.",
            download_bytes: v.file().bytes,
            subdir: "",
            ready: vec![v.cache_name()],
            spare: vec![v.file().name.to_string()],
        });
    }

    out
}

/// Bytes of the named files that exist in `dir`.
fn bytes_of(dir: &Path, names: &[String]) -> (u64, usize) {
    let mut bytes = 0;
    let mut present = 0;
    for n in names {
        if let Ok(m) = std::fs::metadata(dir.join(n)) {
            if m.is_file() {
                bytes += m.len();
                present += 1;
            }
        }
    }
    (bytes, present)
}

/// What is on disk for one model.
pub fn status(asset: &ModelAsset, root: &Path) -> AssetStatus {
    let dir = asset.dir(root);
    let (ready_bytes, ready_present) = bytes_of(&dir, &asset.ready);
    let (spare_bytes, spare_present) = bytes_of(&dir, &asset.spare);
    let ready = ready_present == asset.ready.len() && !asset.ready.is_empty();
    AssetStatus {
        ready,
        partial: !ready && (ready_present > 0 || spare_present > 0),
        bytes: ready_bytes + spare_bytes,
        spare_bytes: if ready { spare_bytes } else { 0 },
    }
}

/// Download and convert one model if it is not ready yet.
///
/// The work is the engine's own first-use path, so a model prepared here is
/// bit for bit the one a run would have prepared — there is no second
/// download route to keep in step.
pub fn ensure(asset: &ModelAsset, root: &Path, sink: &dyn ProgressSink) -> Result<()> {
    let dir = engine_dir(root, asset.engine);
    match asset.kind {
        AssetKind::Autoseg(spec) => {
            autoseg::weights::ensure_model(&spec, &dir, sink)?;
        }
        AssetKind::SegVol => {
            // The converted tensors are dropped again straight away: what is
            // wanted here is the cache the conversion writes.
            segvol::weights::load(&dir, sink)?;
        }
        AssetKind::SegVolText => {
            for f in segvol::weights::CLIP_FILES {
                f.ensure(&dir, sink)?;
            }
        }
        AssetKind::MedSam2(v) => {
            medsam2::weights::load(v, &dir, sink)?;
        }
    }
    Ok(())
}

/// Delete the named files that exist, and report the bytes freed.
fn delete(dir: &Path, names: &[String]) -> Result<u64> {
    let mut freed = 0;
    for n in names {
        let p = dir.join(n);
        let Ok(m) = std::fs::metadata(&p) else {
            continue;
        };
        if !m.is_file() {
            continue;
        }
        std::fs::remove_file(&p).with_context(|| format!("remove {}", p.display()))?;
        freed += m.len();
    }
    Ok(freed)
}

/// Remove everything one model owns; returns the bytes freed.
///
/// Only the file names the inventory lists are deleted — never a whole
/// folder — so a model folder the user also keeps something else in survives
/// intact. The model's own sub-folder is removed afterwards if it came out
/// empty.
pub fn remove(asset: &ModelAsset, root: &Path) -> Result<u64> {
    let dir = asset.dir(root);
    let mut freed = delete(&dir, &asset.ready)?;
    freed += delete(&dir, &asset.spare)?;
    if !asset.subdir.is_empty() {
        let _ = std::fs::remove_dir(&dir);
    }
    Ok(freed)
}

/// Remove the files nothing reads once the model is ready (the raw
/// checkpoint the conversion was made from); returns the bytes freed.
pub fn free_spare(asset: &ModelAsset, root: &Path) -> Result<u64> {
    if !status(asset, root).ready {
        return Ok(0);
    }
    delete(&asset.dir(root), &asset.spare)
}

/// A byte count as the interface shows it: three significant digits, decimal
/// units (what download sizes are quoted in).
pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "kB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1000.0 && u + 1 < UNITS.len() {
        v /= 1000.0;
        u += 1;
    }
    if u == 0 {
        format!("{n} B")
    } else if v < 10.0 {
        format!("{v:.2} {}", UNITS[u])
    } else if v < 100.0 {
        format!("{v:.1} {}", UNITS[u])
    } else {
        format!("{v:.0} {}", UNITS[u])
    }
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
    fn the_inventory_covers_every_engine_with_unique_keys() {
        let inv = inventory();
        assert!(inv.len() >= 11, "{} rows", inv.len());
        let mut keys: Vec<&str> = inv.iter().map(|a| a.key.as_str()).collect();
        keys.sort();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n, "inventory keys are unique");
        for engine in Engine::ALL {
            assert!(
                inv.iter().any(|a| a.engine == engine),
                "{} has no rows",
                engine.subdir()
            );
        }
        for a in &inv {
            assert!(!a.ready.is_empty(), "{} has no ready file", a.key);
            assert!(a.download_bytes > 0, "{} has no size", a.key);
            assert!(a.key.starts_with(a.engine.subdir()), "{}", a.key);
        }
    }

    #[test]
    fn an_empty_root_reports_every_model_as_missing() {
        let root = std::env::temp_dir().join("rds_models_inventory_empty");
        let _ = std::fs::remove_dir_all(&root);
        for a in inventory() {
            let s = status(&a, &root);
            assert!(!s.ready && !s.partial, "{}", a.key);
            assert_eq!(s.bytes, 0);
            assert_eq!(remove(&a, &root).unwrap(), 0);
            assert_eq!(free_spare(&a, &root).unwrap(), 0);
        }
    }

    #[test]
    fn a_model_becomes_ready_partial_and_gone_again() {
        let root = std::env::temp_dir().join("rds_models_inventory_cycle");
        let _ = std::fs::remove_dir_all(&root);
        let asset = inventory()
            .into_iter()
            .find(|a| matches!(a.kind, AssetKind::MedSam2(_)))
            .expect("a MedSAM2 row");
        let dir = asset.dir(&root);
        std::fs::create_dir_all(&dir).unwrap();
        // The spare source checkpoint alone is not a usable model.
        std::fs::write(dir.join(&asset.spare[0]), vec![0u8; 2048]).unwrap();
        let s = status(&asset, &root);
        assert!(!s.ready && s.partial && s.bytes == 2048 && s.spare_bytes == 0);
        // With the converted cache beside it, it is — and the source is spare.
        std::fs::write(dir.join(&asset.ready[0]), vec![0u8; 1024]).unwrap();
        let s = status(&asset, &root);
        assert!(s.ready && !s.partial);
        assert_eq!((s.bytes, s.spare_bytes), (3072, 2048));
        assert_eq!(free_spare(&asset, &root).unwrap(), 2048);
        assert!(status(&asset, &root).ready);
        assert_eq!(remove(&asset, &root).unwrap(), 1024);
        assert_eq!(status(&asset, &root), AssetStatus::default());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn byte_counts_read_the_way_downloads_are_quoted() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1_000), "1.00 kB");
        assert_eq!(human_bytes(155_851_016), "156 MB");
        assert_eq!(human_bytes(1_200_000_000), "1.20 GB");
    }

    #[test]
    fn a_blank_setting_means_the_default_root() {
        assert_eq!(root_from_setting("   "), default_root());
        assert_eq!(root_from_setting(" D:/w "), PathBuf::from("D:/w"));
        assert!(default_root().ends_with(DIR_NAME));
    }
}
