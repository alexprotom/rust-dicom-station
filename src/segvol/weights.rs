//! Checkpoint acquisition for SegVol.
//!
//! The published weights live in the Hugging Face repository
//! [`BAAI/SegVol`](https://huggingface.co/BAAI/SegVol) as a single
//! `pytorch_model.bin` — a plain `state_dict()` saved with `torch.save`, so
//! [`crate::nn::pickle`] reads it directly with an empty top-level key.
//!
//! ## Licensing
//!
//! The SegVol *code* is MIT (Copyright (c) 2023 BAAI-DCAI). The *weights*
//! carry **no license declaration at all** — the model repository has no
//! license tag and no LICENSE file — and the training corpus (M3D-Seg)
//! aggregates 25 datasets whose own terms differ, several of them
//! non-commercial. This is deliberately unlike the auto-segmentation module,
//! whose TotalSegmentator weights are Apache-2.0.
//!
//! Consequently this file only ever *downloads* to the user's own machine, at
//! the user's request. Nothing here is redistributed with the program, and
//! the weights must not be bundled into the installer the way the
//! TotalSegmentator ones may be.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use crate::nn::cache::ProgressSink;
use crate::nn::pickle::PthReader;

/// One file to fetch from the model repository.
#[derive(Clone, Copy, Debug)]
pub struct RemoteFile {
    /// File name, used both remotely and in the local cache.
    pub name: &'static str,
    pub url: &'static str,
    /// Published size in bytes; progress display only.
    pub bytes: u64,
}

/// The model weights: a bare `state_dict()`, fp32, ~724 MB.
pub const CHECKPOINT: RemoteFile = RemoteFile {
    name: "pytorch_model.bin",
    url: "https://huggingface.co/BAAI/SegVol/resolve/main/pytorch_model.bin",
    // Tensor bytes; the ZIP container adds a little. Only a fallback for the
    // progress bar when the server sends no Content-Length.
    bytes: super::layout::PAYLOAD_BYTES,
};

/// CLIP byte-pair vocabulary, needed to turn a text prompt into tokens.
///
/// Data, not a dependency: the tokenizer itself is implemented here, and
/// these two files are fetched alongside the weights exactly as the
/// auto-segmentation module fetches `plans.json` beside its checkpoint.
pub const CLIP_VOCAB: RemoteFile = RemoteFile {
    name: "vocab.json",
    url: "https://huggingface.co/BAAI/SegVol/resolve/main/vocab.json",
    bytes: 862_328,
};

/// CLIP byte-pair merge table.
pub const CLIP_MERGES: RemoteFile = RemoteFile {
    name: "merges.txt",
    url: "https://huggingface.co/BAAI/SegVol/resolve/main/merges.txt",
    bytes: 524_619,
};

/// Default cache directory: `segvol_model/` next to the executable.
pub fn default_models_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("segvol_model")
}

/// True when `f` is already in the cache directory.
pub fn is_cached(f: &RemoteFile, models_dir: &Path) -> bool {
    models_dir.join(f.name).is_file()
}

/// Bytes still to download for the given files.
pub fn download_needed(files: &[RemoteFile], models_dir: &Path) -> u64 {
    files
        .iter()
        .filter(|f| !is_cached(f, models_dir))
        .map(|f| f.bytes)
        .sum()
}

/// Fetch `f` into `models_dir` if it is not already there, and return its
/// path. Downloads land on a temporary name and are renamed only once
/// complete, so an interrupted download is never mistaken for a cached file.
pub fn ensure_file(f: &RemoteFile, models_dir: &Path, sink: &dyn ProgressSink) -> Result<PathBuf> {
    let dest = models_dir.join(f.name);
    if dest.is_file() {
        return Ok(dest);
    }
    std::fs::create_dir_all(models_dir)
        .with_context(|| format!("create {}", models_dir.display()))?;
    let tmp = dest.with_extension("part");
    crate::nn::cache::download_to_file(f.url, &tmp, f.bytes, f.name, sink)
        .with_context(|| format!("download {}", f.url))?;
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

/// Open the checkpoint's state dict. The archive root *is* the state dict,
/// so the top-level key is empty.
///
/// Only `data.pkl` is read here — the tensor metadata — so this is fast even
/// though the file is 724 MB; storage blobs are read on demand.
pub fn open_checkpoint(path: &Path) -> Result<PthReader> {
    PthReader::open(path, "").with_context(|| format!("read SegVol checkpoint {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_download_hint_matches_the_recorded_payload() {
        assert_eq!(CHECKPOINT.bytes, super::super::layout::PAYLOAD_BYTES);
    }

    #[test]
    fn every_remote_file_lives_in_the_same_repository() {
        for f in [CHECKPOINT, CLIP_VOCAB, CLIP_MERGES] {
            assert!(
                f.url
                    .starts_with("https://huggingface.co/BAAI/SegVol/resolve/main/"),
                "{}",
                f.url
            );
            assert!(f.url.ends_with(f.name), "{} vs {}", f.url, f.name);
        }
    }
}
