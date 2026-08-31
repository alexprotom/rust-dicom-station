//! Model-weight acquisition and caching for the auto-segmentation module.
//!
//! First use of a model downloads the official TotalSegmentator weight zip
//! from its GitHub release (the *openly licensed*, Apache-2.0 "total" task
//! weights), extracts the nnU-Net `plans.json` + `checkpoint_final.pth`,
//! parses the PyTorch checkpoint natively, and caches the result as
//! `model.safetensors` + `plans.json` in the model directory. Subsequent
//! runs load the cache directly — no network access.
//!
//! Downloading, checkpoint parsing and the cache format are generic and live
//! in [`crate::nn`]; what stays here is the part that is specific to
//! TotalSegmentator — which models exist, where they are published, and how
//! to get `plans.json` and `checkpoint_final.pth` out of the release zip.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use super::config::ModelConfig;
use crate::nn::cache::{self, ConvertSpec, WTensor};
use crate::progress::{ProgressSink, CANCELLED};

/// One downloadable nnU-Net model.
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    /// Cache sub-directory name, e.g. "total_3mm".
    pub key: &'static str,
    /// Human-readable name shown in progress messages.
    pub label: &'static str,
    pub url: &'static str,
    /// Approximate download size (progress display only).
    pub zip_bytes: u64,
    /// Added to the model's local foreground labels to obtain global
    /// TotalSegmentator class ids (0 for the single-model variants).
    pub global_offset: u8,
}

pub const SPEC_3MM: ModelSpec = ModelSpec {
    key: "total_3mm",
    label: "total 3 mm",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip",
    zip_bytes: 135_386_075,
    global_offset: 0,
};

pub const SPEC_6MM: ModelSpec = ModelSpec {
    key: "total_6mm",
    label: "total 6 mm",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset298_TotalSegmentator_total_6mm_1559subj.zip",
    zip_bytes: 134_827_240,
    global_offset: 0,
};

/// The body-outline task's models. Same nnU-Net architecture, same open
/// Apache-2.0 licence, a different question: two classes, trunk and
/// extremities, whose union is the patient. They are what the body-contour
/// tool's model-assisted method uses to tell patient from equipment — see
/// [`crate::bodymask`].
pub const SPEC_BODY_15MM: ModelSpec = ModelSpec {
    key: "body_1_5mm",
    label: "body 1.5 mm",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset299_body_1559subj.zip",
    zip_bytes: 233_211_222,
    global_offset: 0,
};

pub const SPEC_BODY_6MM: ModelSpec = ModelSpec {
    key: "body_6mm",
    label: "body 6 mm",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset300_body_6mm_1559subj.zip",
    zip_bytes: 124_286_256,
    global_offset: 0,
};

/// The MR counterpart (TotalSegmentator v2.5 weights). Plans a z-score
/// normalization and an anisotropic 3.0 × 1.19 × 0.99 mm grid, which is why
/// the engine carries both.
pub const SPEC_BODY_MR: ModelSpec = ModelSpec {
    key: "body_mr",
    label: "body MR",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset597_mri_body_139subj.zip",
    zip_bytes: 229_810_326,
    global_offset: 0,
};

pub const SPECS_15MM: [ModelSpec; 5] = [
    ModelSpec {
        key: "total_part1_organs",
        label: "1.5 mm organs (1/5)",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
        zip_bytes: 233_742_255,
        global_offset: 0,
    },
    ModelSpec {
        key: "total_part2_vertebrae",
        label: "1.5 mm vertebrae (2/5)",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
        zip_bytes: 234_050_721,
        global_offset: 24,
    },
    ModelSpec {
        key: "total_part3_cardiac",
        label: "1.5 mm cardiac (3/5)",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
        zip_bytes: 234_190_318,
        global_offset: 50,
    },
    ModelSpec {
        key: "total_part4_muscles",
        label: "1.5 mm muscles (4/5)",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
        zip_bytes: 233_625_081,
        global_offset: 68,
    },
    ModelSpec {
        key: "total_part5_ribs",
        label: "1.5 mm ribs (5/5)",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip",
        zip_bytes: 234_016_576,
        global_offset: 91,
    },
];

/// Every published model, in the order the interface lists them: the fast
/// single model, the five full-resolution sub-models, the preview model.
///
/// The inventory in [`crate::models`] walks this list; nothing else needs to
/// know that the 1.5 mm variant is five downloads rather than one.
pub fn all_specs() -> Vec<ModelSpec> {
    let mut v = vec![SPEC_3MM];
    v.extend(SPECS_15MM);
    v.push(SPEC_6MM);
    v.extend([SPEC_BODY_15MM, SPEC_BODY_6MM, SPEC_BODY_MR]);
    v
}

/// A ready-to-run model: architecture config + named weight tensors.
pub struct LoadedModel {
    pub spec: ModelSpec,
    pub config: ModelConfig,
    pub tensors: HashMap<String, WTensor>,
}

/// Load a model, downloading + converting on first use.
pub fn ensure_model(
    spec: &ModelSpec,
    models_dir: &Path,
    sink: &dyn ProgressSink,
) -> Result<LoadedModel> {
    let dir = models_dir.join(spec.key);
    let st_path = dir.join(CACHE_NAME);
    let plans_path = dir.join(PLANS_NAME);
    // `decoder.encoder.*` and `*.all_modules.*` are duplicate registrations
    // of the same storages; drop them.
    let convert = ConvertSpec {
        top_key: "network_weights",
        keep: &|name, _| !name.starts_with("decoder.encoder.") && !name.contains(".all_modules."),
        rename: &|name| name.to_string(),
        label: spec.label,
    };
    let tensors = cache::ensure_converted(
        &st_path,
        &convert,
        || {
            download_and_unpack(spec, &dir, sink)
                .with_context(|| format!("prepare model '{}'", spec.label))
        },
        sink,
    )?;
    let plans_text = std::fs::read_to_string(&plans_path)
        .with_context(|| format!("read {}", plans_path.display()))?;
    let config = ModelConfig::from_plans_json(&plans_text)?;
    let _ = std::fs::remove_file(dir.join(CHECKPOINT_TMP));
    Ok(LoadedModel {
        spec: *spec,
        config,
        tensors,
    })
}

/// Converted-weight cache and the nnU-Net plan, written per model.
pub const CACHE_NAME: &str = "model.safetensors";
pub const PLANS_NAME: &str = "plans.json";
/// The checkpoint extracted from the release zip; deleted after conversion.
pub const CHECKPOINT_TMP: &str = "checkpoint.tmp.pth";
/// The release zip while it is being fetched; deleted after unpacking.
pub const DOWNLOAD_TMP: &str = "download.zip.tmp";

/// True when the model's converted cache is already present.
pub fn is_cached(spec: &ModelSpec, models_dir: &Path) -> bool {
    let dir = models_dir.join(spec.key);
    dir.join(CACHE_NAME).is_file() && dir.join(PLANS_NAME).is_file()
}

/// Download the release zip and pull `plans.json` and the fold-0 checkpoint
/// out of it. Returns the checkpoint's path, ready for conversion.
fn download_and_unpack(
    spec: &ModelSpec,
    dir: &Path,
    sink: &dyn ProgressSink,
) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let zip_tmp = dir.join(DOWNLOAD_TMP);
    // ---- download --------------------------------------------------------
    cache::download_to_file(
        spec.url,
        &zip_tmp,
        spec.zip_bytes,
        &format!("weights ({})", spec.label),
        sink,
    )?;
    // ---- extract the two files we need ----------------------------------
    sink.report(0.0, &format!("Unpacking weights ({})…", spec.label));
    let ckpt_tmp = dir.join(CHECKPOINT_TMP);
    {
        let file = std::fs::File::open(&zip_tmp)?;
        let mut zip = zip::ZipArchive::new(file).context("weights zip")?;
        let mut plans_name = None;
        let mut ckpt_name = None;
        for i in 0..zip.len() {
            let name = zip.by_index_raw(i)?.name().to_owned();
            if name.contains("__MACOSX") {
                continue;
            }
            if name.ends_with("/plans.json") {
                plans_name = Some(name);
            } else if name.ends_with("fold_0/checkpoint_final.pth") {
                ckpt_name = Some(name);
            }
        }
        let plans_name = plans_name.context("weights zip: no plans.json")?;
        let ckpt_name = ckpt_name.context("weights zip: no fold_0/checkpoint_final.pth")?;
        let mut plans = String::new();
        zip.by_name(&plans_name)?
            .read_to_string(&mut plans)
            .context("read plans.json")?;
        // Validate before persisting anything.
        ModelConfig::from_plans_json(&plans)?;
        std::fs::write(dir.join(PLANS_NAME), &plans)?;
        let mut ckpt_entry = zip.by_name(&ckpt_name)?;
        let mut out = std::io::BufWriter::new(std::fs::File::create(&ckpt_tmp)?);
        let total = ckpt_entry.size();
        let mut buf = vec![0u8; 1024 * 1024];
        let mut done: u64 = 0;
        loop {
            if sink.cancelled() {
                bail!(CANCELLED);
            }
            let n = ckpt_entry.read(&mut buf).context("unpack checkpoint")?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            sink.report(
                done as f32 / total.max(1) as f32,
                &format!("Unpacking weights ({})…", spec.label),
            );
        }
        out.flush().ok();
    }
    let _ = std::fs::remove_file(&zip_tmp);
    Ok(ckpt_tmp)
}
