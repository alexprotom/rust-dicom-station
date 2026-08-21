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
use crate::nn::cache::{self, StoreDtype};
use crate::nn::pickle::PthReader;

// The download / cache plumbing lives in `crate::nn` now; re-exported so
// `autoseg::weights` stays the one import site for callers of this module.
pub use crate::nn::cache::{ProgressSink, WTensor};

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
    let st_path = dir.join("model.safetensors");
    let plans_path = dir.join("plans.json");
    if !(st_path.is_file() && plans_path.is_file()) {
        download_and_convert(spec, &dir, sink)
            .with_context(|| format!("prepare model '{}'", spec.label))?;
    }
    sink.report(0.0, &format!("Loading model ({})…", spec.label));
    let plans_text = std::fs::read_to_string(&plans_path)
        .with_context(|| format!("read {}", plans_path.display()))?;
    let config = ModelConfig::from_plans_json(&plans_text)?;
    let tensors = cache::load_safetensors(&st_path)?;
    Ok(LoadedModel {
        spec: *spec,
        config,
        tensors,
    })
}

/// True when the model's converted cache is already present.
pub fn is_cached(spec: &ModelSpec, models_dir: &Path) -> bool {
    let dir = models_dir.join(spec.key);
    dir.join("model.safetensors").is_file() && dir.join("plans.json").is_file()
}

fn download_and_convert(spec: &ModelSpec, dir: &Path, sink: &dyn ProgressSink) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let zip_tmp = dir.join("download.zip.tmp");
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
    let ckpt_tmp = dir.join("checkpoint.tmp.pth");
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
        std::fs::write(dir.join("plans.json"), &plans)?;
        let mut ckpt_entry = zip.by_name(&ckpt_name)?;
        let mut out = std::io::BufWriter::new(std::fs::File::create(&ckpt_tmp)?);
        let total = ckpt_entry.size();
        let mut buf = vec![0u8; 1024 * 1024];
        let mut done: u64 = 0;
        loop {
            if sink.cancelled() {
                bail!("cancelled");
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
    // ---- parse the torch checkpoint, keep only what inference needs ------
    sink.report(0.0, &format!("Converting weights ({})…", spec.label));
    {
        let mut reader = PthReader::open(&ckpt_tmp, "network_weights")?;
        let metas: Vec<(String, crate::nn::pickle::TensorMeta)> = reader
            .tensors
            .iter()
            .filter(|(name, _)| {
                // `decoder.encoder.*` and `*.all_modules.*` are duplicate
                // registrations of the same storages; drop them.
                !name.starts_with("decoder.encoder.") && !name.contains(".all_modules.")
            })
            .cloned()
            .collect();
        if metas.is_empty() {
            bail!("checkpoint: no usable tensors after filtering");
        }
        let mut named: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::with_capacity(metas.len());
        let n = metas.len();
        for (i, (name, meta)) in metas.iter().enumerate() {
            if sink.cancelled() {
                bail!("cancelled");
            }
            let data = reader
                .read_f32(meta)
                .with_context(|| format!("read tensor {name}"))?;
            named.push((name.clone(), meta.shape.clone(), data));
            sink.report(
                i as f32 / n as f32,
                &format!("Converting weights ({})…", spec.label),
            );
        }
        // F32: the auto-segmentation engine is validated against the
        // reference implementation, so the cache must reproduce the
        // checkpoint exactly.
        cache::save_safetensors(&dir.join("model.safetensors"), &named, StoreDtype::F32)?;
    }
    let _ = std::fs::remove_file(&ckpt_tmp);
    Ok(())
}
