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
    // The ZIP container adds a little to the payload below; this is only a
    // fallback for the progress bar when the server sends no Content-Length.
    bytes: FP32_PAYLOAD_BYTES,
};

/// Total size of the tensor data at fp32, `EXPECTED_PARAMS * 4`.
///
/// The published file is a ZIP wrapped around exactly this much data and is
/// listed at 724 MB, which is what corroborates the parameter arithmetic —
/// and with it the conclusion that the frozen CLIP text tower ships inside
/// the checkpoint rather than needing a separate download.
pub const FP32_PAYLOAD_BYTES: u64 = 723_560_256;

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

/// Total parameter count of the published checkpoint.
///
/// Derived from the module definitions: 87,388,416 image encoder (MONAI ViT,
/// 12 layers, dim 768, 2048 tokens) + 17,996 prompt encoder + 29,923,716 mask
/// decoder + 63,165,952 frozen CLIP text tower + 393,984 `dim_align`. See
/// [`FP32_PAYLOAD_BYTES`] for why the published file size corroborates this.
pub const EXPECTED_PARAMS: usize = 180_890_064;

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

/// Which part of the network a state-dict key belongs to. Used by the probe
/// and, later, to skip the parts inference never runs.
pub fn group_of(key: &str) -> &'static str {
    if key.starts_with("image_encoder.") {
        "image_encoder"
    } else if key.starts_with("prompt_encoder.") {
        "prompt_encoder"
    } else if key.starts_with("mask_decoder.") {
        "mask_decoder"
    } else if key.starts_with("text_encoder.clip_text_model.") {
        "clip_text_model"
    } else if key.starts_with("text_encoder.") {
        "text_encoder (dim_align)"
    } else {
        "other"
    }
}

/// True for tensors the inference path never touches.
///
/// `prompt_encoder.mask_downscaling` is SAM's 2-D mask-input branch. SegVol
/// never passes a mask prompt, so the branch is dead — but its parameters are
/// still in the checkpoint, and reproducing them would mean implementing
/// `Conv2d` for nothing.
pub fn is_dead_weight(key: &str) -> bool {
    key.starts_with("prompt_encoder.mask_downscaling.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouping_covers_the_published_key_prefixes() {
        assert_eq!(
            group_of("image_encoder.patch_embedding.position_embeddings"),
            "image_encoder"
        );
        assert_eq!(
            group_of("prompt_encoder.positional_encoding_gaussian_matrix"),
            "prompt_encoder"
        );
        assert_eq!(group_of("mask_decoder.iou_token.weight"), "mask_decoder");
        assert_eq!(
            group_of("text_encoder.clip_text_model.encoder.layers.0.mlp.fc1.weight"),
            "clip_text_model"
        );
        assert_eq!(
            group_of("text_encoder.dim_align.weight"),
            "text_encoder (dim_align)"
        );
        assert_eq!(group_of("something.else"), "other");
    }

    #[test]
    fn only_the_2d_mask_branch_is_dead() {
        assert!(is_dead_weight("prompt_encoder.mask_downscaling.0.weight"));
        assert!(!is_dead_weight("prompt_encoder.no_mask_embed.weight"));
        assert!(!is_dead_weight("prompt_encoder.point_embeddings.0.weight"));
    }

    #[test]
    fn the_payload_size_matches_the_parameter_count() {
        // Guards the two constants against drifting apart if either is edited.
        assert_eq!(EXPECTED_PARAMS as u64 * 4, FP32_PAYLOAD_BYTES);
        // 87.4 M encoder + 30 M decoder + 63 M CLIP + the small parts
        assert_eq!(
            EXPECTED_PARAMS,
            87_388_416 + 17_996 + 29_923_716 + 63_165_952 + 393_984
        );
    }
}
