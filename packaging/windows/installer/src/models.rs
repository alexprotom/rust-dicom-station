//! Optional pre-fetch of the TotalSegmentator weights.
//!
//! The viewer downloads and converts the official nnU-Net checkpoints itself
//! on first use; doing it here just moves that wait into the installation, on
//! a machine that is already online. The work is done by the viewer's own
//! `autoseg::weights` code - the installer links the library without the GPU
//! backend, so this only writes the model cache, it never runs inference.
//! The files land where the viewer looks for them: the `totalsegmentator/`
//! sub-folder of the model root (`rust_dicom_station::models`). Only these
//! weights are ever pre-fetched - they are Apache-2.0; the SegVol and MedSAM2
//! weights must be downloaded by the user at their own request.
//!
//! Building the installer with `--no-default-features` drops the dependency
//! (and the option disappears from the UI).

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;

use crate::plan::Models;

/// Whether this build can pre-fetch weights at all.
pub const AVAILABLE: bool = cfg!(feature = "prefetch-models");

#[cfg(feature = "prefetch-models")]
mod imp {
    use super::*;
    use rust_dicom_station::autoseg::weights::{self, ModelSpec, SPECS_15MM, SPEC_3MM, SPEC_6MM};
    use rust_dicom_station::models::{engine_dir, Engine};
    use rust_dicom_station::progress::ProgressSink;

    pub fn specs(models: Models) -> Vec<ModelSpec> {
        match models {
            Models::None => vec![],
            Models::Preview6mm => vec![SPEC_6MM],
            Models::Fast3mm => vec![SPEC_3MM],
            Models::HighRes15mm => SPECS_15MM.to_vec(),
            Models::Everything => {
                let mut v = vec![SPEC_3MM];
                v.extend_from_slice(&SPECS_15MM);
                v
            }
        }
    }

    /// Total download in bytes, ignoring models already cached under `root`.
    pub fn download_size(models: Models, root: &Path) -> u64 {
        let dir = engine_dir(root, Engine::TotalSegmentator);
        specs(models)
            .iter()
            .filter(|s| !weights::is_cached(s, &dir))
            .map(|s| s.zip_bytes)
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

    pub fn prefetch(
        models: Models,
        root: &Path,
        progress: &(dyn Fn(f32, &str) + Sync),
        cancel: &AtomicBool,
    ) -> Result<()> {
        let specs = specs(models);
        let dir = engine_dir(root, Engine::TotalSegmentator);
        std::fs::create_dir_all(&dir)?;
        for (i, spec) in specs.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                anyhow::bail!("cancelled");
            }
            if weights::is_cached(spec, &dir) {
                continue;
            }
            let sink = Slice {
                progress,
                cancel,
                base: i as f32 / specs.len() as f32,
                span: 1.0 / specs.len() as f32,
            };
            // `ensure_model` downloads, converts and caches; the returned
            // tensors are dropped right away - we only wanted the cache.
            let _ = weights::ensure_model(spec, &dir, &sink)?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "prefetch-models"))]
mod imp {
    use super::*;

    pub fn download_size(_models: Models, _dir: &Path) -> u64 {
        0
    }

    pub fn prefetch(
        _models: Models,
        _dir: &Path,
        _progress: &(dyn Fn(f32, &str) + Sync),
        _cancel: &AtomicBool,
    ) -> Result<()> {
        anyhow::bail!(
            "this installer was built with --no-default-features and cannot pre-fetch \
             model weights; the viewer will download them on first use"
        )
    }
}

pub use imp::{download_size, prefetch};
