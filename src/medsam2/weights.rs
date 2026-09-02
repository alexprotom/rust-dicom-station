//! Checkpoint acquisition for MedSAM2.
//!
//! The published weights live in the Hugging Face repository
//! [`wanglab/MedSAM2`](https://huggingface.co/wanglab/MedSAM2) as several
//! `.pt` files that differ only in what they were fine-tuned on: one
//! architecture, one loader, a choice in the UI. Each is a
//! `torch.save({"model": state_dict, ...})`, so [`crate::nn::pickle`] reads it
//! with the top-level key `"model"`. The download, the conversion and the
//! `safetensors` cache are the shared machinery in [`crate::nn::cache`].
//!
//! ## Licensing
//!
//! The MedSAM2 *code* is Apache-2.0, as is Meta's SAM 2 underneath it. The
//! *weights* are tagged **CC-BY-SA-4.0** on Hugging Face, and the model card
//! additionally states that they "can only be used for research and education
//! purposes". Those two statements are in tension; the stricter one governs.
//!
//! Consequently this file only ever *downloads* to the user's own machine, at
//! the user's request - the same handling as SegVol's unlicensed weights, and
//! unlike the auto-segmentation module's Apache-2.0 TotalSegmentator ones.
//! Nothing here is redistributed with the program, the weights must not be
//! bundled into the installer, and the converted cache written beside them is
//! a derivative that must not be redistributed either.

use anyhow::{Context, Result};
use std::path::Path;

use crate::nn::cache::{self, ConvertSpec, RemoteFile};
use crate::nn::params::Params;
use crate::nn::pickle::PthReader;
use crate::progress::ProgressSink;

use super::layout;

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

    /// The short name the command-line tools accept.
    pub fn key(self) -> &'static str {
        match self {
            Variant::Latest => "latest",
            Variant::CtLesion => "ct-lesion",
            Variant::MriLiverLesion => "mri-liver",
            Variant::Base2411 => "2411",
        }
    }

    /// The variant a command-line name refers to, if any.
    pub fn from_key(key: &str) -> Option<Variant> {
        Variant::ALL.into_iter().find(|v| v.key() == key)
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

/// True when the converted cache is present, so nothing has to be downloaded
/// or parsed.
pub fn is_ready(v: Variant, models_dir: &Path) -> bool {
    models_dir.join(v.cache_name()).is_file()
}

/// Bytes still to download for this variant.
pub fn download_needed(v: Variant, models_dir: &Path) -> u64 {
    if is_ready(v, models_dir) {
        0
    } else {
        cache::download_needed([&v.file()], models_dir)
    }
}

/// Open a checkpoint's state dict. SAM 2 saves it under `"model"`.
pub fn open_checkpoint(path: &Path) -> Result<PthReader> {
    PthReader::open(path, "model")
        .with_context(|| format!("read MedSAM2 checkpoint {}", path.display()))
}

/// Which tensors to convert: the live ones, under their normalized names.
fn convert_spec() -> ConvertSpec<'static> {
    ConvertSpec {
        top_key: "model",
        keep: &|name, _| !layout::is_dead_weight(name),
        rename: &|name| layout::normalize_key(name).to_string(),
        label: "MedSAM2",
    }
}

/// The whole first-use path: download if needed, convert once, and load from
/// the converted cache ever after.
pub fn load(v: Variant, models_dir: &Path, sink: &dyn ProgressSink) -> Result<Params> {
    let tensors = cache::ensure_converted(
        &models_dir.join(v.cache_name()),
        &convert_spec(),
        || v.file().ensure(models_dir, sink),
        sink,
    )?;
    Ok(Params::new(tensors))
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
            assert_eq!(Variant::from_key(v.key()), Some(v));
        }
        assert_eq!(Variant::from_key("nope"), None);
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

    #[test]
    fn an_empty_folder_needs_the_whole_download() {
        let dir = std::env::temp_dir().join("rds_medsam2_weights_empty");
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(
            download_needed(Variant::Latest, &dir),
            layout::PAYLOAD_BYTES
        );
    }
}
