//! The nnU-Net v1 pretrained models (Isensee et al., *Nature Methods*
//! 2021): the Medical Segmentation Decathlon's CT tasks and a few other
//! challenges, tumours among them, as nnU-Net v1 published them on Zenodo
//! (record 4003545, CC BY-NC 4.0).
//!
//! Each task is one archive of about 5 GB: every configuration, five folds
//! each, with their optimizer states. What runs here is one network of it,
//! `3d_fullres` fold 0, taken out by HTTP range requests when the server
//! allows them ([`crate::nn::remote_zip`]); otherwise the archive is
//! downloaded once and the same members are unpacked.
//!
//! nnU-Net v1's `Generic_UNet` is the network nnU-Net v2 calls
//! `PlainConvUNet`, under other names: per stage, conv-InstanceNorm-
//! LeakyReLU blocks, the first strided; transposed convolutions up; one
//! segmentation head per decoder stage. Three differences matter. The
//! transposed convolutions and the heads have no bias (a zero bias is
//! written in their place); decoder stage `u` convolves with the kernel of
//! encoder stage `n - 1 - u`, where v2 takes stage `n - 2 - u` (the two
//! differ when the first stage's kernel is anisotropic, and the rewritten
//! plans carry v1's choice); and the plans are a Python pickle with numpy
//! values (`plans.pkl`, read by [`crate::nn::pyobj`]) rather than JSON. The
//! plans are rewritten as an nnU-Net v2 `plans.json` with an explicit
//! architecture, so the rest of the engine runs these models as it runs
//! every other ([`super::task`]).
//!
//! What follows v1 and what does not. The network, its patch, spacing and
//! CT normalization (clip to the training set's 0.5 and 99.5 percentiles,
//! z-score with its mean and standard deviation) and the 0.5 window step are
//! v1's, and so is the post-processing the training chose
//! (`postprocessing.json`: keep the largest connected piece of the listed
//! classes, or drop only the pieces under a minimum volume; applied on the
//! scan's own grid, 6-connected, as v1 does). The rest is this engine's
//! pipeline: the scan is reoriented to [S, A, R] and resampled trilinearly
//! (v1 does not reorient, and resamples with third-order splines,
//! nearest-neighbour across thick slices), and the patches are not
//! mirrored (v1 averages eight mirrored predictions by default). One fold
//! of five runs where v1 ensembles all five.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::config::ModelConfig;
use super::task::{FoldUse, Licence, Modality, NnTask, Part, Post, IDENTITY_LUT};
use super::weights::{self, Home, ModelSpec, Source};
use crate::nn::cache::{self, StoreDtype, WTensor};
use crate::nn::pickle::PthReader;
use crate::nn::{pyobj, remote_zip};
use crate::progress::{ProgressSink, CANCELLED};
use crate::volume::AxisOrder;

/// The training every pretrained v1 model was published from.
pub const TRAINING: &str = "nnUNetTrainerV2__nnUNetPlansv2.1";
/// The post-processing the training chose, kept beside the plans.
pub const POST_NAME: &str = "postprocessing.json";
/// What one fold of a v1 `3d_fullres` checkpoint weighs, optimizer state
/// included: the download a range request makes (the archive index knows
/// the real figure, the table does not).
pub const FOLD_BYTES_ESTIMATE: u64 = 250_000_000;
/// `plans.pkl` while it waits to be rewritten as `plans.json`.
const PKL_TMP: &str = "plans.pkl.tmp";

/// The archive members one model needs.
#[derive(Debug, PartialEq)]
pub struct Entries {
    pub plans: String,
    pub post: Option<String>,
    /// `(fold, member)` of each fold's checkpoint.
    pub checkpoints: Vec<(u8, String)>,
}

/// The `3d_fullres` training of `task` in an archive's entry list.
pub fn pick(names: &[String], task: &str, folds: &[u8]) -> Result<Entries> {
    let folder = format!("3d_fullres/{task}/{TRAINING}/");
    let find = |file: &str| -> Option<String> {
        let want = format!("{folder}{file}");
        names
            .iter()
            .find(|n| n.ends_with(&want) && !n.contains("__MACOSX"))
            .cloned()
    };
    let plans = find("plans.pkl").with_context(|| format!("archive: no {folder}plans.pkl"))?;
    let mut checkpoints = Vec::new();
    for &f in folds {
        let c = find(&format!("fold_{f}/model_final_checkpoint.model"))
            .or_else(|| find(&format!("fold_{f}/model_best.model")))
            .with_context(|| format!("archive: no checkpoint of {folder}fold_{f}"))?;
        checkpoints.push((f, c));
    }
    Ok(Entries {
        plans,
        post: find(POST_NAME),
        checkpoints,
    })
}

/// Get `plans.json` (rewritten from `plans.pkl`), the post-processing and
/// the checkpoints of `folds` of a v1 model into `dir`. Returns `(fold,
/// checkpoint path)` pairs, ready for [`convert`].
pub fn fetch(
    spec: &ModelSpec,
    dir: &Path,
    folds: &[u8],
    sink: &dyn ProgressSink,
) -> Result<Vec<(u8, PathBuf)>> {
    let Source::NnUnetV1 { task } = spec.source else {
        bail!("{} is not an nnU-Net v1 model", spec.key);
    };
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let plans_path = dir.join(weights::PLANS_NAME);
    let want_plans = !plans_path.is_file();
    let found: std::cell::RefCell<Vec<(u8, PathBuf)>> = Default::default();
    let pick_members = |names: &[String]| -> Result<Vec<remote_zip::Member>> {
        let e = pick(names, task, folds)?;
        let mut m = Vec::new();
        if want_plans {
            m.push(remote_zip::Member {
                name: e.plans,
                dest: dir.join(PKL_TMP),
            });
            if let Some(p) = e.post {
                m.push(remote_zip::Member {
                    name: p,
                    dest: dir.join(POST_NAME),
                });
            }
        }
        for (f, name) in e.checkpoints {
            let dest = dir.join(weights::fold_checkpoint_tmp(f));
            found.borrow_mut().push((f, dest.clone()));
            m.push(remote_zip::Member { name, dest });
        }
        Ok(m)
    };
    remote_zip::fetch_members(
        spec.url,
        spec.zip_bytes,
        &dir.join(weights::DOWNLOAD_TMP),
        &format!("weights ({})", spec.label),
        &pick_members,
        sink,
    )?;
    if want_plans {
        let pkl = dir.join(PKL_TMP);
        let text = plans_json(&std::fs::read(&pkl).context("read plans.pkl")?)?;
        ModelConfig::from_plans_json_cfg(&text, spec.configuration)?;
        std::fs::write(&plans_path, text)?;
        let _ = std::fs::remove_file(pkl);
    }
    Ok(found.into_inner())
}

fn ints(v: &Value, what: &str) -> Result<Vec<i64>> {
    v.as_array()
        .with_context(|| format!("plans.pkl: {what} is not a list"))?
        .iter()
        .map(|x| x.as_i64().with_context(|| format!("plans.pkl: {what}")))
        .collect()
}

fn triples(v: &Value, what: &str) -> Result<Vec<Vec<i64>>> {
    v.as_array()
        .with_context(|| format!("plans.pkl: {what} is not a list"))?
        .iter()
        .map(|x| {
            let t = ints(x, what)?;
            if t.len() != 3 {
                bail!("plans.pkl: {what} entry is not of length 3");
            }
            Ok(t)
        })
        .collect()
}

/// nnU-Net v1's `plans.pkl` as an nnU-Net v2 `plans.json` with an explicit
/// `PlainConvUNet` architecture: the last stage of `plans_per_stage` (what
/// v1's `3d_fullres` trains on), base features doubling per stage up to
/// 320, `conv_per_stage` convolutions per stage on both sides.
pub fn plans_json(pkl: &[u8]) -> Result<String> {
    let v = pyobj::load(pkl).context("read plans.pkl")?;
    let tf = ints(&v["transpose_forward"], "transpose_forward")?;
    if tf != [0, 1, 2] {
        bail!("plans.pkl: transpose_forward {tf:?} is not supported");
    }
    if v["num_modalities"].as_i64().unwrap_or(1) != 1 {
        bail!("plans.pkl: models of more than one input channel are not supported");
    }
    let stages = v["plans_per_stage"]
        .as_object()
        .context("plans.pkl: plans_per_stage")?;
    let last = stages
        .keys()
        .filter_map(|k| k.parse::<i64>().ok())
        .max()
        .context("plans.pkl: no stage")?;
    let st = &stages[&last.to_string()];
    let patch = ints(&st["patch_size"], "patch_size")?;
    let spacing: Vec<f64> = st["current_spacing"]
        .as_array()
        .context("plans.pkl: current_spacing")?
        .iter()
        .map(|x| x.as_f64().context("plans.pkl: current_spacing"))
        .collect::<Result<_>>()?;
    let pool = triples(&st["pool_op_kernel_sizes"], "pool_op_kernel_sizes")?;
    let kernels = triples(&st["conv_kernel_sizes"], "conv_kernel_sizes")?;
    if patch.len() != 3 || spacing.len() != 3 || kernels.len() != pool.len() + 1 {
        bail!("plans.pkl: inconsistent stage {last}");
    }
    let n = kernels.len();
    let base = v["base_num_features"].as_i64().unwrap_or(32);
    let cps = v["conv_per_stage"].as_i64().unwrap_or(2);
    let features: Vec<i64> = (0..n).map(|i| (base << i.min(16)).min(320)).collect();
    let mut strides = vec![vec![1, 1, 1]];
    strides.extend(pool);
    let scheme = match v["normalization_schemes"]["0"].as_str() {
        Some("CT") => "CTNormalization",
        Some("nonCT") => "ZScoreNormalization",
        other => bail!("plans.pkl: normalization {other:?} is not supported"),
    };
    let ip = &v["dataset_properties"]["intensityproperties"]["0"];
    let num = |k: &str| -> Value { ip[k].as_f64().map(Value::from).unwrap_or(Value::Null) };
    let out = json!({
        "plans_name": "nnUNetPlansv2.1 (nnU-Net v1, rewritten)",
        "transpose_forward": [0, 1, 2],
        "transpose_backward": [0, 1, 2],
        "configurations": {
            "3d_fullres": {
                "patch_size": patch,
                "spacing": spacing,
                "normalization_schemes": [scheme],
                "architecture": {
                    "network_class_name":
                        "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                    "arch_kwargs": {
                        "n_stages": n,
                        "features_per_stage": features,
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "kernel_sizes": kernels,
                        "strides": strides,
                        "n_conv_per_stage": vec![cps; n],
                        "n_conv_per_stage_decoder": vec![cps; n - 1],
                        // v1's decoder stage u convolves with the kernel of
                        // encoder stage n - 1 - u (v2: n - 2 - u).
                        "decoder_kernel_sizes": (0..n - 1).map(|u| kernels[n - 1 - u].clone()).collect::<Vec<_>>(),
                        "conv_bias": true,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-5, "affine": true},
                        "dropout_op": null,
                        "nonlin": "torch.nn.LeakyReLU",
                        "nonlin_kwargs": {"negative_slope": 0.01, "inplace": true}
                    }
                }
            }
        },
        "foreground_intensity_properties_per_channel": {
            "0": {
                "mean": num("mean"),
                "std": num("sd"),
                "percentile_00_5": num("percentile_00_5"),
                "percentile_99_5": num("percentile_99_5")
            }
        },
        "nnunet_v1": {
            "num_classes": v["num_classes"],
            "stage": last,
            "conv_per_stage": cps
        }
    });
    Ok(serde_json::to_string_pretty(&out)?)
}

/// A v1 `Generic_UNet` state-dict name as nnU-Net v2's `PlainConvUNet`
/// calls the same tensor; `None` for anything else (InstanceNorm's running
/// statistics, which it does not track). `cps` is the plans'
/// `conv_per_stage`: the bottleneck and every decoder stage are a sequence
/// of `cps - 1` convolutions and then one more.
pub fn rename(key: &str, cps: usize) -> Option<String> {
    let p: Vec<&str> = key.split('.').collect();
    let leaf = |op: &str, w: &str| -> Option<String> {
        let op = match op {
            "conv" => "conv",
            "instnorm" => "norm",
            _ => return None,
        };
        matches!(w, "weight" | "bias").then(|| format!("{op}.{w}"))
    };
    let idx = |seq: &str, j: &str| -> Option<usize> {
        let j: usize = j.parse().ok()?;
        match seq {
            "0" => Some(j),
            "1" => Some(cps - 1 + j),
            _ => None,
        }
    };
    match p.as_slice() {
        ["conv_blocks_context", d, "blocks", j, op, w] => {
            Some(format!("encoder.stages.{d}.0.convs.{j}.{}", leaf(op, w)?))
        }
        ["conv_blocks_context", d, seq, "blocks", j, op, w] => Some(format!(
            "encoder.stages.{d}.0.convs.{}.{}",
            idx(seq, j)?,
            leaf(op, w)?
        )),
        ["conv_blocks_localization", u, seq, "blocks", j, op, w] => Some(format!(
            "decoder.stages.{u}.convs.{}.{}",
            idx(seq, j)?,
            leaf(op, w)?
        )),
        ["tu", u, "weight"] => Some(format!("decoder.transpconvs.{u}.weight")),
        ["seg_outputs", u, "weight"] => Some(format!("decoder.seg_layers.{u}.weight")),
        _ => None,
    }
}

/// Convert a v1 checkpoint (`state_dict` of a `.model` file) into the
/// cache under v2's names, with the zero biases v1's bias-free transposed
/// convolutions and heads stand for.
pub fn convert(
    ckpt: &Path,
    cache_path: &Path,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<HashMap<String, WTensor>> {
    let plans: Value = cache_path
        .parent()
        .map(|d| d.join(weights::PLANS_NAME))
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(Value::Null);
    let cps = plans["nnunet_v1"]["conv_per_stage"].as_u64().unwrap_or(2) as usize;
    let mut reader = PthReader::open(ckpt, "state_dict")
        .with_context(|| format!("read checkpoint {}", ckpt.display()))?;
    let metas = std::mem::take(&mut reader.tensors);
    let n = metas.len();
    let mut out: HashMap<String, WTensor> = HashMap::with_capacity(n + 16);
    for (i, (name, meta)) in metas.iter().enumerate() {
        if sink.cancelled() {
            bail!(CANCELLED);
        }
        let Some(to) = rename(name, cps) else {
            continue;
        };
        sink.report(
            i as f32 / n as f32,
            &format!("Converting weights ({label}): {}/{n}", i + 1),
        );
        let data = reader
            .read_f32(meta)
            .with_context(|| format!("read tensor {name}"))?;
        out.insert(
            to,
            WTensor {
                shape: meta.shape.clone(),
                data,
            },
        );
    }
    if !out.contains_key("encoder.stages.0.0.convs.0.conv.weight") {
        bail!(
            "{}: not an nnU-Net v1 Generic_UNet checkpoint",
            ckpt.display()
        );
    }
    // Zero biases: a transposed convolution's output channels are its
    // weight's second axis, a head's its first.
    let mut extra = Vec::new();
    for (k, t) in &out {
        let (bias, len) = if let Some(stem) = k.strip_suffix(".weight") {
            if stem.starts_with("decoder.transpconvs.") {
                (format!("{stem}.bias"), t.shape.get(1).copied())
            } else if stem.starts_with("decoder.seg_layers.") {
                (format!("{stem}.bias"), t.shape.first().copied())
            } else {
                continue;
            }
        } else {
            continue;
        };
        if let (Some(len), false) = (len, out.contains_key(&bias)) {
            extra.push((
                bias,
                WTensor {
                    shape: vec![len],
                    data: vec![0.0; len],
                },
            ));
        }
    }
    out.extend(extra);
    sink.report(1.0, &format!("Writing the weight cache ({label})"));
    cache::save_tensor_map(cache_path, &out, StoreDtype::F32)
        .with_context(|| format!("write {}", cache_path.display()))?;
    Ok(out)
}

// ---- post-processing ----------------------------------------------------------

/// One rule of `postprocessing.json`: of the voxels carrying one of
/// `classes`, keep the largest 6-connected piece, and drop the others (all
/// of them, or only those under `min_mm3` when the training set one).
#[derive(Clone, Debug, PartialEq)]
pub struct PostRule {
    pub classes: Vec<u8>,
    pub min_mm3: Option<f64>,
}

/// The rules of a model folder's `postprocessing.json`; empty when there is
/// none (v1 then post-processes nothing).
pub fn load_post(dir: &Path) -> Result<Vec<PostRule>> {
    let path = dir.join(POST_NAME);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(Vec::new());
    };
    parse_post(&text).with_context(|| format!("read {}", path.display()))
}

/// `postprocessing.json`: `for_which_classes` lists classes and groups of
/// classes; `min_valid_object_sizes`, when present, is a Python dict
/// literal keyed the same way (`"{(1, 2): 25089.0, 2: 8.0}"`).
pub fn parse_post(text: &str) -> Result<Vec<PostRule>> {
    let v: Value = serde_json::from_str(text)?;
    let mins: Vec<(Vec<u8>, f64)> = match v.get("min_valid_object_sizes") {
        Some(Value::String(s)) => parse_py_sizes(s)?,
        _ => Vec::new(),
    };
    let mut rules = Vec::new();
    for c in v["for_which_classes"].as_array().into_iter().flatten() {
        let classes: Vec<u8> = match c {
            Value::Array(a) => a
                .iter()
                .filter_map(|x| x.as_u64().map(|x| x as u8))
                .collect(),
            other => other.as_u64().map(|x| vec![x as u8]).unwrap_or_default(),
        };
        if classes.is_empty() || classes.contains(&0) {
            bail!("postprocessing.json: bad class entry {c}");
        }
        let min_mm3 = mins.iter().find(|(k, _)| *k == classes).map(|(_, m)| *m);
        rules.push(PostRule { classes, min_mm3 });
    }
    Ok(rules)
}

/// `{(1, 2): 25089.0, 2: 8.0}` → `[([1, 2], 25089.0), ([2], 8.0)]`.
fn parse_py_sizes(s: &str) -> Result<Vec<(Vec<u8>, f64)>> {
    let s = s.trim();
    if s == "None" {
        return Ok(Vec::new());
    }
    let body = s
        .strip_prefix('{')
        .and_then(|b| b.strip_suffix('}'))
        .context("min_valid_object_sizes is not a dict")?;
    // Split on the commas outside parentheses.
    let mut items = Vec::new();
    let (mut depth, mut start) = (0i32, 0usize);
    for (i, ch) in body.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                items.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    items.push(&body[start..]);
    let mut out = Vec::new();
    for it in items.into_iter().map(str::trim).filter(|i| !i.is_empty()) {
        let (k, val) = it
            .rsplit_once(':')
            .context("min_valid_object_sizes entry")?;
        let key: Vec<u8> = k
            .trim()
            .trim_start_matches('(')
            .trim_end_matches(')')
            .split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(|x| x.parse::<u8>().context("min_valid_object_sizes key"))
            .collect::<Result<_>>()?;
        out.push((
            key,
            val.trim()
                .parse::<f64>()
                .context("min_valid_object_sizes value")?,
        ));
    }
    Ok(out)
}

/// Apply the rules to labels on a grid of `dims` voxels of `spacing` mm
/// (x fastest), as v1's `remove_all_but_the_largest_connected_component`.
pub fn apply_post(labels: &mut [u8], dims: [usize; 3], spacing: [f64; 3], rules: &[PostRule]) {
    let [nx, ny, nz] = dims;
    let vox = spacing[0] * spacing[1] * spacing[2];
    for rule in rules {
        let mut want = [false; 256];
        for &c in &rule.classes {
            want[c as usize] = true;
        }
        // 6-connected pieces, breadth first.
        let mut comp = vec![u32::MAX; labels.len()];
        let mut sizes: Vec<usize> = Vec::new();
        let mut queue = Vec::new();
        for seed in 0..labels.len() {
            if !want[labels[seed] as usize] || comp[seed] != u32::MAX {
                continue;
            }
            let id = sizes.len() as u32;
            comp[seed] = id;
            queue.clear();
            queue.push(seed);
            let mut count = 0;
            while let Some(v) = queue.pop() {
                count += 1;
                let (x, y, z) = (v % nx, (v / nx) % ny, v / (nx * ny));
                let mut visit = |w: usize| {
                    if want[labels[w] as usize] && comp[w] == u32::MAX {
                        comp[w] = id;
                        queue.push(w);
                    }
                };
                if x > 0 {
                    visit(v - 1);
                }
                if x + 1 < nx {
                    visit(v + 1);
                }
                if y > 0 {
                    visit(v - nx);
                }
                if y + 1 < ny {
                    visit(v + nx);
                }
                if z > 0 {
                    visit(v - nx * ny);
                }
                if z + 1 < nz {
                    visit(v + nx * ny);
                }
            }
            sizes.push(count);
        }
        let Some(&largest) = sizes.iter().max() else {
            continue;
        };
        // Every piece of the largest size stays, as v1 compares sizes.
        let drop: Vec<bool> = sizes
            .iter()
            .map(|&s| {
                s != largest
                    && match rule.min_mm3 {
                        Some(m) => (s as f64 * vox) < m,
                        None => true,
                    }
            })
            .collect();
        for (l, &c) in labels.iter_mut().zip(&comp) {
            if c != u32::MAX && drop[c as usize] {
                *l = 0;
            }
        }
    }
}

// ---- the models ------------------------------------------------------------

const RECORD: &str = "https://zenodo.org/record/4003545/files/";

macro_rules! v1_spec {
    ($key:literal, $label:literal, $detail:literal, $task:literal, $bytes:expr) => {
        ModelSpec {
            key: $key,
            label: $label,
            detail: $detail,
            url: concat!(
                "https://zenodo.org/record/4003545/files/",
                $task,
                ".zip?download=1"
            ),
            zip_bytes: $bytes,
            folds: 1,
            axes: AxisOrder::Sar,
            home: Home::NnUnetV1,
            plans: "nnUNetPlansv2.1",
            configuration: "3d_fullres",
            source: Source::NnUnetV1 { task: $task },
        }
    };
}

pub const SPEC_LIVER: ModelSpec = v1_spec!(
    "msd_liver",
    "MSD liver and liver tumour",
    "Liver and liver tumours on contrast CT (MSD Task 3, LiTS).",
    "Task003_Liver",
    5_000_000_000
);
pub const SPEC_LUNG: ModelSpec = v1_spec!(
    "msd_lung",
    "MSD lung tumour",
    "Primary lung tumours, non-small-cell lung cancer (MSD Task 6).",
    "Task006_Lung",
    5_000_000_000
);
pub const SPEC_PANCREAS: ModelSpec = v1_spec!(
    "msd_pancreas",
    "MSD pancreas and pancreatic tumour",
    "Pancreas and pancreatic masses, portal-venous CT (MSD Task 7).",
    "Task007_Pancreas",
    5_000_000_000
);
pub const SPEC_HEPATIC_VESSEL: ModelSpec = v1_spec!(
    "msd_hepatic_vessel",
    "MSD hepatic vessels and liver tumour",
    "Hepatic vessels and the liver tumours next to them (MSD Task 8).",
    "Task008_HepaticVessel",
    5_000_000_000
);
pub const SPEC_SPLEEN: ModelSpec = v1_spec!(
    "msd_spleen",
    "MSD spleen",
    "The spleen, portal-venous CT (MSD Task 9).",
    "Task009_Spleen",
    5_000_000_000
);
pub const SPEC_COLON: ModelSpec = v1_spec!(
    "msd_colon",
    "MSD colon cancer",
    "Colon cancer primaries (MSD Task 10).",
    "Task010_Colon",
    5_000_000_000
);
pub const SPEC_BTCV: ModelSpec = v1_spec!(
    "btcv_abdomen",
    "BTCV abdominal organs",
    "Thirteen abdominal organs and vessels (Beyond the Cranial Vault, Task 17).",
    "Task017_AbdominalOrganSegmentation",
    5_000_000_000
);
pub const SPEC_KITS: ModelSpec = v1_spec!(
    "kits19",
    "KiTS19 kidney and kidney tumour",
    "Kidneys and renal tumours, arterial-phase CT (KiTS 2019, Task 48).",
    "Task048_KiTS_clean",
    5_000_000_000
);
pub const SPEC_SEGTHOR: ModelSpec = v1_spec!(
    "segthor",
    "SegTHOR thoracic organs at risk",
    "Oesophagus, heart, trachea and aorta on planning CT (SegTHOR, Task 55).",
    "Task055_SegTHOR",
    5_000_000_000
);

const C_LIVER: [&str; 2] = ["liver", "liver_tumor"];
const C_LUNG: [&str; 1] = ["lung_tumor"];
const C_PANCREAS: [&str; 2] = ["pancreas", "pancreatic_tumor"];
const C_HEPATIC: [&str; 2] = ["hepatic_vessel", "liver_tumor"];
const C_SPLEEN: [&str; 1] = ["spleen"];
const C_COLON: [&str; 1] = ["colon_cancer"];
const C_BTCV: [&str; 13] = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
];
const C_KITS: [&str; 2] = ["kidney", "kidney_tumor"];
const C_SEGTHOR: [&str; 4] = ["esophagus", "heart", "trachea", "aorta"];

const GROUP: &str = "Tumours and organs (nnU-Net v1)";

macro_rules! v1_task {
    ($spec:expr, $classes:expr) => {
        NnTask {
            key: $spec.key,
            label: $spec.label,
            group: GROUP,
            detail: $spec.detail,
            modality: Modality::Ct,
            licence: Licence::CcByNc4,
            classes: &$classes,
            parts: &[Part {
                spec: $spec,
                lut: &IDENTITY_LUT,
            }],
            folds: FoldUse::First,
            step: 0.5,
            crop: None,
            post: Post::NnUnetV1,
        }
    };
}

/// The nnU-Net v1 tasks, in the order the interface lists them.
pub static TASKS: [NnTask; 9] = [
    v1_task!(SPEC_LIVER, C_LIVER),
    v1_task!(SPEC_LUNG, C_LUNG),
    v1_task!(SPEC_PANCREAS, C_PANCREAS),
    v1_task!(SPEC_HEPATIC_VESSEL, C_HEPATIC),
    v1_task!(SPEC_SPLEEN, C_SPLEEN),
    v1_task!(SPEC_COLON, C_COLON),
    v1_task!(SPEC_BTCV, C_BTCV),
    v1_task!(SPEC_KITS, C_KITS),
    v1_task!(SPEC_SEGTHOR, C_SEGTHOR),
];

/// The URL prefix every v1 archive shares (a test keeps the rows on it).
pub fn record() -> &'static str {
    RECORD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_unet_names_map_onto_plain_conv_unet() {
        let r = |k| rename(k, 2);
        assert_eq!(
            r("conv_blocks_context.0.blocks.1.conv.weight").as_deref(),
            Some("encoder.stages.0.0.convs.1.conv.weight")
        );
        assert_eq!(
            r("conv_blocks_context.3.blocks.0.instnorm.bias").as_deref(),
            Some("encoder.stages.3.0.convs.0.norm.bias")
        );
        // The bottleneck: a sequence of (cps - 1) convs, then one.
        assert_eq!(
            r("conv_blocks_context.5.0.blocks.0.conv.bias").as_deref(),
            Some("encoder.stages.5.0.convs.0.conv.bias")
        );
        assert_eq!(
            r("conv_blocks_context.5.1.blocks.0.instnorm.weight").as_deref(),
            Some("encoder.stages.5.0.convs.1.norm.weight")
        );
        assert_eq!(
            r("conv_blocks_localization.2.1.blocks.0.conv.weight").as_deref(),
            Some("decoder.stages.2.convs.1.conv.weight")
        );
        assert_eq!(
            rename("conv_blocks_localization.0.1.blocks.0.conv.weight", 3).as_deref(),
            Some("decoder.stages.0.convs.2.conv.weight")
        );
        assert_eq!(
            r("tu.4.weight").as_deref(),
            Some("decoder.transpconvs.4.weight")
        );
        assert_eq!(
            r("seg_outputs.4.weight").as_deref(),
            Some("decoder.seg_layers.4.weight")
        );
        assert_eq!(
            r("conv_blocks_context.0.blocks.0.instnorm.running_mean"),
            None
        );
        assert_eq!(r("something.else"), None);
    }

    #[test]
    fn the_training_is_found_in_the_archive() {
        let names: Vec<String> = [
            "nnUNet/2d/Task006_Lung/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl",
            "nnUNet/3d_fullres/Task006_Lung/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl",
            "nnUNet/3d_fullres/Task006_Lung/nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json",
            "nnUNet/3d_fullres/Task006_Lung/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model",
            "nnUNet/3d_fullres/Task006_Lung/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model.pkl",
            "nnUNet/3d_fullres/Task006_Lung/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/model_final_checkpoint.model",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let e = pick(&names, "Task006_Lung", &[0]).unwrap();
        assert_eq!(e.plans, names[1]);
        assert_eq!(e.post.as_deref(), Some(names[2].as_str()));
        assert_eq!(e.checkpoints, vec![(0, names[3].clone())]);
        assert!(pick(&names, "Task006_Lung", &[2]).is_err());
        assert!(pick(&names, "Task003_Liver", &[0]).is_err());
    }

    #[test]
    fn v1_plans_become_a_v2_plain_unet() {
        let pkl = include_bytes!("../../../../tests/data/nnunet-v1-lung-plans.pkl");
        let text = plans_json(pkl).unwrap();
        let cfg = ModelConfig::from_plans_json_cfg(&text, "3d_fullres").unwrap();
        assert_eq!(cfg.patch_size, [80, 192, 160]);
        assert_eq!(cfg.spacing, [1.24, 0.79, 0.79]);
        assert_eq!(cfg.features, [32, 64, 128, 256, 320, 320]);
        assert_eq!(cfg.strides[0], [1, 1, 1]);
        assert_eq!(cfg.strides[1], [1, 2, 2]);
        assert_eq!(cfg.n_conv_per_stage, [2; 6]);
        assert_eq!(cfg.n_conv_per_stage_decoder, [2; 5]);
        assert_eq!((cfg.clip_lo, cfg.clip_hi), (-1024.0, 325.0));
        assert_eq!((cfg.mean, cfg.std), (-158.5, 324.5));
    }

    /// A small `Generic_UNet` built as nnUNetTrainerV2 builds it (nnunet
    /// 1.7.1: base 4 features, three stages, an anisotropic first stage),
    /// random weights, saved as a v1 `.model` with its `plans.pkl`; its
    /// output for one patch. The checkpoint goes through the conversion
    /// (renames, zero biases) and the plans through the rewrite, and the v2
    /// network the engine builds from them must give the same logits.
    #[test]
    fn a_v1_checkpoint_runs_as_the_v2_network() {
        let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data/");
        let dir = std::env::temp_dir().join("rds_v1_tiny");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pkl = std::fs::read(format!("{data}nnunet-v1-tiny-plans.pkl")).unwrap();
        let text = plans_json(&pkl).unwrap();
        std::fs::write(dir.join(weights::PLANS_NAME), &text).unwrap();
        struct Quiet;
        impl ProgressSink for Quiet {
            fn report(&self, _: f32, _: &str) {}
            fn cancelled(&self) -> bool {
                false
            }
        }
        let t = convert(
            Path::new(&format!("{data}nnunet-v1-tiny.model")),
            &dir.join(weights::CACHE_NAME),
            "tiny",
            &Quiet,
        )
        .unwrap();
        assert_eq!(t["decoder.transpconvs.0.bias"].data, vec![0.0; 8]);
        assert_eq!(t["decoder.seg_layers.1.bias"].data, vec![0.0; 3]);
        // The cache reads back as written.
        let back = cache::load_safetensors(&dir.join(weights::CACHE_NAME)).unwrap();
        assert_eq!(back.len(), t.len());
        let cfg = ModelConfig::from_plans_json_cfg(&text, "3d_fullres").unwrap();
        assert_eq!(cfg.features, [4, 8, 16]);
        assert_eq!(cfg.kernels[0], [1, 3, 3]);
        // The full-resolution decoder stage has stage 1's kernel in v1.
        assert_eq!(cfg.decoder_kernels, [[3, 3, 3], [3, 3, 3]]);
        let net = super::super::net::UNet::build(cfg, &back).unwrap();
        let io = cache::load_safetensors(&PathBuf::from(format!(
            "{data}nnunet-v1-tiny-io.safetensors"
        )))
        .unwrap();
        let x = crate::nn::tensor::Act {
            c: 1,
            d: 8,
            h: 16,
            w: 16,
            data: io["input"].data.clone(),
        };
        let y = net.forward_cpu(&x);
        let want = &io["output"].data;
        assert_eq!(y.data.len(), want.len());
        let range = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = y
            .data
            .iter()
            .zip(want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-4 * range.max(1.0), "worst {worst} of {range}");
    }

    #[test]
    fn post_processing_reads_v1s_file() {
        let rules = parse_post(
            r#"{"dc_per_class_raw": {}, "for_which_classes": [[1, 2], 2],
                "min_valid_object_sizes": "{(1, 2): 30.0, 2: 8.0}"}"#,
        )
        .unwrap();
        assert_eq!(
            rules,
            vec![
                PostRule {
                    classes: vec![1, 2],
                    min_mm3: Some(30.0)
                },
                PostRule {
                    classes: vec![2],
                    min_mm3: Some(8.0)
                },
            ]
        );
        let plain = parse_post(r#"{"for_which_classes": [1]}"#).unwrap();
        assert_eq!(
            plain,
            vec![PostRule {
                classes: vec![1],
                min_mm3: None
            }]
        );
        assert!(parse_post(r#"{"for_which_classes": []}"#)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn the_largest_piece_stays_and_small_ones_go() {
        // A 6 x 1 x 1 row: class 1 at 0..3 and 4, class 2 at 5.
        let mut l = vec![1, 1, 1, 0, 1, 2];
        apply_post(
            &mut l,
            [6, 1, 1],
            [1.0; 3],
            &[PostRule {
                classes: vec![1],
                min_mm3: None,
            }],
        );
        assert_eq!(l, [1, 1, 1, 0, 0, 2]);
        // Classes 1 and 2 together: [1 1 1 0 0 2] has pieces of 3 and 1.
        // With a 2 mm³ minimum the one-voxel piece goes, with 1 it stays.
        let mut a = l.clone();
        let r = |m| {
            vec![PostRule {
                classes: vec![1, 2],
                min_mm3: Some(m),
            }]
        };
        apply_post(&mut a, [6, 1, 1], [1.0; 3], &r(2.0));
        assert_eq!(a, [1, 1, 1, 0, 0, 0]);
        let mut b = l.clone();
        apply_post(&mut b, [6, 1, 1], [1.0; 3], &r(1.0));
        assert_eq!(b, l);
        // Diagonal neighbours are not connected (6-connectivity).
        let mut d = vec![1, 0, 0, 1];
        apply_post(
            &mut d,
            [2, 2, 1],
            [1.0; 3],
            &[PostRule {
                classes: vec![1],
                min_mm3: None,
            }],
        );
        // Two pieces of equal size: both are "the largest", both stay.
        assert_eq!(d, [1, 0, 0, 1]);
    }

    #[test]
    fn every_row_points_at_the_record() {
        for t in &TASKS {
            let s = t.parts[0].spec;
            assert!(s.url.starts_with(record()), "{}", s.key);
            assert!(matches!(s.source, Source::NnUnetV1 { task } if s.url.contains(task)));
            assert_eq!(s.home, Home::NnUnetV1);
        }
    }
}
