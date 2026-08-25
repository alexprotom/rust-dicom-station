//! Checkpoint acquisition for MedSAM2.
//!
//! The published weights live in the Hugging Face repository
//! [`wanglab/MedSAM2`](https://huggingface.co/wanglab/MedSAM2) as several
//! `.pt` files that differ only in what they were fine-tuned on: one
//! architecture, one loader, a choice in the UI. Each is a
//! `torch.save({"model": state_dict, ...})`, so [`crate::nn::pickle`] reads it
//! with the top-level key `"model"`.
//!
//! ## Licensing
//!
//! The MedSAM2 *code* is Apache-2.0, as is Meta's SAM 2 underneath it. The
//! *weights* are tagged **CC-BY-SA-4.0** on Hugging Face, and the model card
//! additionally states that they "can only be used for research and education
//! purposes". Those two statements are in tension; the stricter one governs.
//!
//! Consequently this file only ever *downloads* to the user's own machine, at
//! the user's request — the same handling as SegVol's unlicensed weights, and
//! unlike the auto-segmentation module's Apache-2.0 TotalSegmentator ones.
//! Nothing here is redistributed with the program, the weights must not be
//! bundled into the installer, and the converted cache written beside them is
//! a derivative that must not be redistributed either.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::nn::cache::{self, ProgressSink, StoreDtype, WTensor};
use crate::nn::params::Params;
use crate::nn::pickle::PthReader;

use super::layout;

/// One file to fetch from the model repository.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    /// File name, used both remotely and in the local cache.
    pub name: &'static str,
    pub url: &'static str,
    /// Published size in bytes; progress display only.
    pub bytes: u64,
}

/// Which fine-tune to run. All of them are SAM 2.1-T at 512 with identical
/// tensor layouts, so the choice costs nothing but a download.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Variant {
    /// The authors' recommended general model.
    #[default]
    Latest,
    /// Fine-tuned for lesions on CT.
    CtLesion,
    /// Fine-tuned for liver lesions on MRI.
    MriLiverLesion,
    /// The November 2024 base model.
    Base2411,
}

impl Variant {
    pub const ALL: [Variant; 4] = [
        Variant::Latest,
        Variant::CtLesion,
        Variant::MriLiverLesion,
        Variant::Base2411,
    ];

    /// Name shown in the interface.
    pub fn label(self) -> &'static str {
        match self {
            Variant::Latest => "General (recommended)",
            Variant::CtLesion => "CT lesions",
            Variant::MriLiverLesion => "MRI liver lesions",
            Variant::Base2411 => "Base (2024-11)",
        }
    }

    pub fn file(self) -> RemoteFile {
        let name = match self {
            Variant::Latest => "MedSAM2_latest.pt",
            Variant::CtLesion => "MedSAM2_CTLesion.pt",
            Variant::MriLiverLesion => "MedSAM2_MRI_LiverLesion.pt",
            Variant::Base2411 => "MedSAM2_2411.pt",
        };
        let url = match self {
            Variant::Latest => {
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
            }
            Variant::CtLesion => {
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_CTLesion.pt"
            }
            Variant::MriLiverLesion => {
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_MRI_LiverLesion.pt"
            }
            Variant::Base2411 => {
                "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_2411.pt"
            }
        };
        RemoteFile {
            name,
            url,
            // Tensor bytes; the ZIP container adds a little. Only a fallback
            // for the progress bar when the server sends no Content-Length.
            bytes: layout::PAYLOAD_BYTES,
        }
    }

    /// Name of the converted-weight cache written beside the checkpoint.
    pub fn cache_name(self) -> String {
        format!("{}.safetensors", self.file().name.trim_end_matches(".pt"))
    }
}

/// Default cache directory: `medsam2_model/` next to the executable.
pub fn default_models_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("medsam2_model")
}

/// True when the converted cache is present, so nothing has to be downloaded
/// or parsed.
pub fn is_ready(v: Variant, models_dir: &Path) -> bool {
    models_dir.join(v.cache_name()).is_file()
}

/// True when `f` is already in the cache directory.
pub fn is_cached(f: &RemoteFile, models_dir: &Path) -> bool {
    models_dir.join(f.name).is_file()
}

/// Bytes still to download for this variant.
pub fn download_needed(v: Variant, models_dir: &Path) -> u64 {
    if is_ready(v, models_dir) || is_cached(&v.file(), models_dir) {
        0
    } else {
        v.file().bytes
    }
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
    cache::download_to_file(f.url, &tmp, f.bytes, f.name, sink)
        .with_context(|| format!("download {}", f.url))?;
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

/// Open a checkpoint's state dict. SAM 2 saves it under `"model"`.
pub fn open_checkpoint(path: &Path) -> Result<PthReader> {
    PthReader::open(path, "model")
        .with_context(|| format!("read MedSAM2 checkpoint {}", path.display()))
}

/// Read every live tensor out of an opened checkpoint.
pub fn read_all(reader: &mut PthReader, sink: &dyn ProgressSink) -> Result<Vec<(String, WTensor)>> {
    let metas: Vec<(String, crate::nn::pickle::TensorMeta)> = reader
        .tensors
        .iter()
        .filter(|(name, _)| !layout::is_dead_weight(name))
        .cloned()
        .collect();
    let n = metas.len().max(1);
    let mut out = Vec::with_capacity(metas.len());
    for (i, (name, meta)) in metas.iter().enumerate() {
        if sink.cancelled() {
            anyhow::bail!("cancelled");
        }
        sink.report(i as f32 / n as f32, "Converting weights");
        let data = reader
            .read_f32(meta)
            .with_context(|| format!("read tensor {name}"))?;
        let key = layout::normalize_key(name).to_string();
        out.push((
            key,
            WTensor {
                shape: meta.shape.clone(),
                data,
            },
        ));
    }
    Ok(out)
}

/// The whole first-use path: download if needed, convert once, and load from
/// the converted cache ever after.
///
/// The cache is written in `f32`. MedSAM2 is small enough (156 MB) that
/// halving it would save little, and keeping full precision leaves the port
/// comparable with a reference run tensor for tensor.
pub fn load(v: Variant, models_dir: &Path, sink: &dyn ProgressSink) -> Result<Params> {
    let cache_path = models_dir.join(v.cache_name());
    if cache_path.is_file() {
        sink.report(0.0, "Loading weights");
        let tensors = cache::load_safetensors(&cache_path)?;
        return Ok(Params::new(tensors));
    }
    let pt = ensure_file(&v.file(), models_dir, sink)?;
    let mut reader = open_checkpoint(&pt)?;
    let tensors = read_all(&mut reader, sink)?;
    let flat: Vec<(String, Vec<usize>, Vec<f32>)> = tensors
        .iter()
        .map(|(k, t)| (k.clone(), t.shape.clone(), t.data.clone()))
        .collect();
    sink.report(0.95, "Writing weight cache");
    cache::save_safetensors(&cache_path, &flat, StoreDtype::F32)
        .with_context(|| format!("write {}", cache_path.display()))?;
    let map: HashMap<String, WTensor> = tensors.into_iter().collect();
    Ok(Params::new(map))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_lives_in_the_same_repository() {
        for v in Variant::ALL {
            let f = v.file();
            assert!(
                f.url
                    .starts_with("https://huggingface.co/wanglab/MedSAM2/resolve/main/"),
                "{}",
                f.url
            );
            assert!(f.url.ends_with(f.name), "{} vs {}", f.url, f.name);
            assert_eq!(f.bytes, layout::PAYLOAD_BYTES);
        }
    }

    #[test]
    fn variants_have_distinct_files_and_cache_names() {
        let mut names: Vec<String> = Variant::ALL.iter().map(|v| v.cache_name()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), Variant::ALL.len());
        assert_eq!(Variant::default(), Variant::Latest);
        assert_eq!(Variant::Latest.cache_name(), "MedSAM2_latest.safetensors");
    }

    #[test]
    fn the_payload_estimate_matches_the_derived_inventory() {
        // 38,962,754 f32 elements is the 156 MB the model card advertises.
        assert_eq!(layout::PAYLOAD_BYTES, 155_851_016);
        // A guard on the derived layout, not on this expression: if the
        // inventory ever grows past the published file, something is wrong.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(layout::PAYLOAD_BYTES < 160_000_000);
        }
    }
}
