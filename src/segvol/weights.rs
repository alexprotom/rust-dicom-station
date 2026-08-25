//! Checkpoint acquisition for SegVol.
//!
//! The published weights live in the Hugging Face repository
//! [`BAAI/SegVol`](https://huggingface.co/BAAI/SegVol) as a single
//! `pytorch_model.bin` — a plain `state_dict()` saved with `torch.save`, so
//! [`crate::nn::pickle`] reads it directly with an empty top-level key. The
//! download, the conversion to the `safetensors` cache and the cache itself
//! are the shared machinery in [`crate::nn::cache`]; what stays here is what
//! is SegVol's alone — which files, which tensors, and under what names.
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
use std::path::Path;

use crate::nn::cache::{self, ConvertSpec, RemoteFile};
use crate::nn::params::Params;
use crate::nn::pickle::{Dtype, PthReader};
use crate::progress::ProgressSink;

use super::layout;

/// The model weights: a bare `state_dict()`, fp32, ~724 MB.
pub const CHECKPOINT: RemoteFile = RemoteFile {
    name: "pytorch_model.bin",
    url: "https://huggingface.co/BAAI/SegVol/resolve/main/pytorch_model.bin",
    // Tensor bytes; the ZIP container adds a little. Only a fallback for the
    // progress bar when the server sends no Content-Length.
    bytes: layout::PAYLOAD_BYTES,
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

/// The two tokenizer files, fetched together.
pub const CLIP_FILES: [RemoteFile; 2] = [CLIP_VOCAB, CLIP_MERGES];

/// Name of the converted-weight cache written beside the checkpoint.
pub const CACHE_NAME: &str = "segvol.safetensors";

/// True when the converted cache is present, so nothing has to be downloaded
/// or parsed to run.
pub fn is_ready(models_dir: &Path) -> bool {
    models_dir.join(CACHE_NAME).is_file()
}

/// Bytes still to download before the network (and, with `text`, the
/// tokenizer) can run.
pub fn download_needed(models_dir: &Path, text: bool) -> u64 {
    let weights = if is_ready(models_dir) {
        0
    } else {
        cache::download_needed([&CHECKPOINT], models_dir)
    };
    let tokenizer = if text {
        cache::download_needed(&CLIP_FILES, models_dir)
    } else {
        0
    };
    weights + tokenizer
}

/// Which tensors to convert: the live ones, under their normalized names.
/// CLIP's `position_ids` buffer is the checkpoint's only integer tensor and
/// nothing reads it.
fn convert_spec() -> ConvertSpec<'static> {
    ConvertSpec {
        top_key: "",
        keep: &|name, meta| {
            !layout::is_dead_weight(name)
                && matches!(meta.dtype, Dtype::F32 | Dtype::F16 | Dtype::F64)
        },
        rename: &|name| layout::normalize_key(name).to_string(),
        label: "SegVol",
    }
}

/// The whole first-use path: download if needed, convert once, and load from
/// the converted cache ever after.
pub fn load(models_dir: &Path, sink: &dyn ProgressSink) -> Result<Params> {
    let tensors = cache::ensure_converted(
        &models_dir.join(CACHE_NAME),
        &convert_spec(),
        || CHECKPOINT.ensure(models_dir, sink),
        sink,
    )?;
    Ok(Params::new(tensors))
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
        assert_eq!(CHECKPOINT.bytes, layout::PAYLOAD_BYTES);
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

    #[test]
    fn an_empty_folder_needs_the_whole_download() {
        let dir = std::env::temp_dir().join("rds_segvol_weights_empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(download_needed(&dir, false), CHECKPOINT.bytes);
        assert_eq!(
            download_needed(&dir, true),
            CHECKPOINT.bytes + CLIP_VOCAB.bytes + CLIP_MERGES.bytes
        );
    }
}
