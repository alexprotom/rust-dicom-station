//! Optional download of model weights during installation.
//!
//! The viewer downloads and converts every model's published weights itself
//! on first use; doing it here just moves that wait into the installation,
//! on a machine that is already online. The list and the work are the
//! viewer's own: the model manager's inventory
//! (`rust_dicom_station::models::inventory`) says what exists, under which
//! licence and at what size, and `models::ensure` runs the engine's first-use
//! path, so a model fetched here is bit for bit the one a run would fetch.
//! The installer links the library without the GPU backend: it only writes
//! the model cache, it never runs inference.
//!
//! Nothing is fetched unless asked for. The window offers named sets (the
//! recommended TotalSegmentator model, every open-licence model, every
//! model) and a list to pick from, each row with its licence; the command
//! line takes the same sets or model keys. Non-commercial weights are
//! fetched only when chosen, never redistributed. The licensed
//! TotalSegmentator models need the user's licence number, read from the
//! viewer's own settings file of the user running the setup (which the
//! window writes it to when typed there); without one they are skipped.
//!
//! A model that fails to download does not fail the installation: the
//! program is installed, the failure is reported, and the viewer fetches
//! that model on first use.
//!
//! Building the installer with `--no-default-features` drops the dependency
//! (and the choice disappears from the UI).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use crate::plan::Models;

/// Whether this build can fetch weights at all.
pub const AVAILABLE: bool = cfg!(feature = "prefetch-models");

/// One downloadable model as the setup lists it.
#[derive(Clone, Debug)]
pub struct Row {
    /// The model manager's key (`totalsegmentator/total_3mm`).
    pub key: String,
    pub label: String,
    /// Which engine's list it belongs to.
    pub engine: &'static str,
    pub licence: &'static str,
    /// Apache-2.0 or MIT: any use, commercial work included.
    pub open: bool,
    /// Non-commercial (research) use only.
    pub research_only: bool,
    /// Downloads with the user's TotalSegmentator licence number.
    pub needs_licence: bool,
    /// Bytes still to fetch (0 when ready).
    pub bytes: u64,
    pub ready: bool,
}

#[cfg(feature = "prefetch-models")]
mod imp {
    use super::*;
    use rust_dicom_station::autoseg::weights;
    use rust_dicom_station::models::{self as m, ModelAsset};
    use rust_dicom_station::progress::ProgressSink;

    fn row(a: &ModelAsset, root: &Path) -> Row {
        let s = m::status(a, root);
        Row {
            key: a.key.clone(),
            label: a.label.clone(),
            engine: a.engine.label(),
            licence: a.licence.name(),
            open: a.licence.open(),
            research_only: a.licence.research_only(),
            needs_licence: a.licence.needs_licence_number(),
            bytes: if s.ready { 0 } else { a.download_bytes },
            ready: s.ready,
        }
    }

    /// Every model the setup can fetch, with what is already in `root`.
    pub fn rows(root: &Path) -> Vec<Row> {
        m::inventory().iter().map(|a| row(a, root)).collect()
    }

    /// The models a choice stands for.
    fn chosen(models: &Models) -> Vec<ModelAsset> {
        let inv = m::inventory();
        match models {
            Models::None => Vec::new(),
            Models::Recommended => inv
                .into_iter()
                .filter(|a| a.key == crate::plan::KEY_TOTAL_3MM)
                .collect(),
            Models::Open => inv.into_iter().filter(|a| a.licence.open()).collect(),
            Models::Every => inv,
            Models::Pick(keys) => inv
                .into_iter()
                .filter(|a| keys.iter().any(|k| k.eq_ignore_ascii_case(&a.key)))
                .collect(),
        }
    }

    /// Keys of a pick that name no model.
    pub fn unknown_keys(models: &Models) -> Vec<String> {
        let Models::Pick(keys) = models else {
            return Vec::new();
        };
        let inv = m::inventory();
        keys.iter()
            .filter(|k| !inv.iter().any(|a| a.key.eq_ignore_ascii_case(k)))
            .cloned()
            .collect()
    }

    /// The licence number in the viewer's settings of the user running the
    /// setup, if one is kept there.
    pub fn licence_number() -> Option<String> {
        let path = crate::plan::viewer_settings_path()?;
        let text = std::fs::read_to_string(path).ok()?;
        text.lines().find_map(|l| {
            let (k, v) = l.split_once('=')?;
            (k.trim().eq_ignore_ascii_case(crate::plan::SETTINGS_LICENCE_KEY)
                && !v.trim().is_empty())
            .then(|| v.trim().to_string())
        })
    }

    /// Total download in bytes, ignoring models already in `root` and,
    /// without a licence number, the licensed ones.
    pub fn download_size(models: &Models, root: &Path) -> u64 {
        let licensed = licence_number().is_some();
        chosen(models)
            .iter()
            .filter(|a| licensed || !a.licence.needs_licence_number())
            .filter(|a| !m::status(a, root).ready)
            .map(|a| a.download_bytes)
            .sum()
    }

    /// Adapts the viewer's progress trait onto the installer's callback,
    /// mapping each model onto its own slice of the progress bar.
    struct Slice<'a> {
        progress: &'a (dyn Fn(f32, &str) + Sync),
        cancel: &'a AtomicBool,
        base: f32,
        span: f32,
    }

    impl ProgressSink for Slice<'_> {
        fn report(&self, frac: f32, msg: &str) {
            (self.progress)(self.base + self.span * frac.clamp(0.0, 1.0), msg);
        }
        fn cancelled(&self) -> bool {
            self.cancel.load(Ordering::Relaxed)
        }
    }

    /// Fetch what `models` stands for into `root`. Returns one line per
    /// model that was skipped or failed; only cancellation is an error.
    pub fn prefetch(
        models: &Models,
        root: &Path,
        progress: &(dyn Fn(f32, &str) + Sync),
        cancel: &AtomicBool,
    ) -> Result<Vec<String>> {
        let want = chosen(models);
        let number = licence_number();
        weights::set_ts_licence(number.as_deref());
        std::fs::create_dir_all(root)?;
        let mut notes = Vec::new();
        let n = want.len().max(1);
        for (i, a) in want.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            if m::status(a, root).ready {
                continue;
            }
            if a.licence.needs_licence_number() && number.is_none() {
                notes.push(format!(
                    "{}: skipped, no TotalSegmentator licence number in the settings",
                    a.label
                ));
                continue;
            }
            let sink = Slice {
                progress,
                cancel,
                base: i as f32 / n as f32,
                span: 1.0 / n as f32,
            };
            if let Err(e) = m::ensure(a, root, &sink) {
                if cancel.load(Ordering::Relaxed) {
                    anyhow::bail!("cancelled");
                }
                notes.push(format!(
                    "{}: not downloaded ({e:#}); the viewer fetches it on first use",
                    a.label
                ));
            }
        }
        Ok(notes)
    }

    /// `--list-models`.
    pub fn print_list() {
        let root = crate::plan::Options::default().models_dir;
        println!(
            "{:46} {:28} {:>9}  label",
            "key", "licence", "download"
        );
        for r in rows(&root) {
            println!(
                "{:46} {:28} {:>9}  {}",
                r.key,
                r.licence,
                if r.ready {
                    "ready".to_string()
                } else {
                    crate::plan::human_size(r.bytes)
                },
                r.label
            );
        }
    }
}

#[cfg(not(feature = "prefetch-models"))]
mod imp {
    use super::*;

    pub fn rows(_root: &Path) -> Vec<Row> {
        Vec::new()
    }

    pub fn unknown_keys(_models: &Models) -> Vec<String> {
        Vec::new()
    }

    pub fn licence_number() -> Option<String> {
        None
    }

    pub fn download_size(_models: &Models, _dir: &Path) -> u64 {
        0
    }

    pub fn prefetch(
        _models: &Models,
        _dir: &Path,
        _progress: &(dyn Fn(f32, &str) + Sync),
        _cancel: &AtomicBool,
    ) -> Result<Vec<String>> {
        anyhow::bail!(
            "this installer was built with --no-default-features and cannot download \
             model weights; the viewer downloads them on first use"
        )
    }

    pub fn print_list() {
        println!("This setup was built without the model downloads.");
    }
}

pub use imp::{download_size, licence_number, prefetch, print_list, rows, unknown_keys};
