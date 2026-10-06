//! nnU-Net `plans.json` parsing → the static architecture + preprocessing
//! configuration of one TotalSegmentator model.
//!
//! Two generations of the file exist. nnU-Net 2.0 and 2.1 wrote the
//! architecture as flat keys of the configuration (`UNet_class_name`,
//! `UNet_base_num_features`, `pool_op_kernel_sizes`, ...; every
//! TotalSegmentator model released up to v2.5 is of this kind). nnU-Net 2.2
//! moved it into an `architecture` object with the class name and its
//! keyword arguments, which is what the `total_v3` weights and every model
//! trained since carry. Both parse into the same [`ModelConfig`].
//!
//! A configuration may inherit from another (`inherits_from`): the head and
//! neck models train `3d_fullres_high`, which is `3d_fullres` with another
//! spacing. The chain is resolved here, the child's keys over the parent's.
//!
//! Array-valued fields are stored in the nnU-Net array-axis order, i.e. the
//! order of the model tensor's spatial axes after the canonical
//! reorientation ([S, A, R] - see `preprocess`).

use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};

/// How the model wants its input scaled - nnU-Net's `normalization_schemes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// `CTNormalization`: clip to the training set's [p0.5, p99.5] window,
    /// then z-score with the dataset fingerprint's mean and standard
    /// deviation. Every constant comes from `plans.json`, so the same CT
    /// always normalizes the same way.
    Ct,
    /// `ZScoreNormalization`: subtract *this image's* mean and divide by
    /// *this image's* standard deviation. MR has no absolute scale, so the
    /// MR models use it - and it is why an MR run fills its normalization
    /// constants in only after resampling ([`ModelConfig::apply_image_norm`]).
    ZScore,
}

/// The network family a `plans.json` describes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arch {
    /// `PlainConvUNet`: every encoder stage is
    /// [`ModelConfig::n_conv_per_stage`] conv-norm-lrelu blocks, the first
    /// carrying the stage's stride.
    PlainConv,
    /// `ResidualEncoderUNet` (the `nnUNetResEncUNet*Plans`): a one-block
    /// stem at the first stage's width, then `blocks_per_stage[s]` residual
    /// blocks per stage (conv-norm-lrelu, conv-norm, plus a skip that is a
    /// 1x1x1 conv and norm wherever the stride or the width changes, lrelu
    /// after the sum). The decoder is the plain network's.
    ResidualEncoder { blocks_per_stage: Vec<usize> },
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// Which network the weights assemble into.
    pub arch: Arch,
    /// Which scheme [`Self::clip_lo`] … [`Self::std`] are to be read under.
    pub norm: Norm,
    /// Sliding-window patch size per spatial axis.
    pub patch_size: [usize; 3],
    /// Target voxel spacing (mm) per spatial axis (isotropic for these models).
    pub spacing: [f64; 3],
    /// Feature channels per encoder stage, e.g. [32, 64, 128, 256, 320].
    pub features: Vec<usize>,
    /// Conv kernel size per stage (all [3,3,3] for these models).
    pub kernels: Vec<[usize; 3]>,
    /// Downsampling stride entering each stage ([1,1,1] for stage 0).
    pub strides: Vec<[usize; 3]>,
    /// Convs per encoder stage (2 for the plain models). Unused by
    /// [`Arch::ResidualEncoder`], whose stages are counted in blocks.
    pub n_conv_per_stage: Vec<usize>,
    /// Convs per decoder stage.
    pub n_conv_per_stage_decoder: Vec<usize>,
    /// Kernel per decoder stage, deepest first; empty for nnU-Net v2's
    /// rule (decoder stage `t` takes encoder stage `n - 2 - t`'s kernel).
    /// nnU-Net v1 takes stage `n - 1 - t`'s, which differs where the first
    /// stage's kernel is anisotropic; its rewritten plans say so
    /// (`decoder_kernel_sizes`).
    pub decoder_kernels: Vec<[usize; 3]>,
    /// Whether the convolutions carry a bias (`conv_bias`; always true in
    /// the released models, and what the weight loader expects).
    pub conv_bias: bool,
    /// Clip bounds then z-score. For [`Norm::ZScore`] the bounds are
    /// infinite and the mean/std belong to the image, not the dataset.
    pub clip_lo: f32,
    pub clip_hi: f32,
    pub mean: f32,
    pub std: f32,
}

/// The configuration every TotalSegmentator model trains unless its task
/// says otherwise.
pub const DEFAULT_CONFIGURATION: &str = "3d_fullres";

impl ModelConfig {
    pub fn n_stages(&self) -> usize {
        self.features.len()
    }

    /// Fill the normalization constants from the resampled image itself, as
    /// `ZScoreNormalization` requires. A no-op for CT models, whose
    /// constants are fixed by the training set.
    pub fn apply_image_norm(&mut self, voxels: &[f32]) {
        if self.norm != Norm::ZScore || voxels.is_empty() {
            return;
        }
        let n = voxels.len() as f64;
        let mean = voxels.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = voxels
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        // nnU-Net: `image -= mean; image /= (std + 1e-8)`.
        self.clip_lo = f32::NEG_INFINITY;
        self.clip_hi = f32::INFINITY;
        self.mean = mean as f32;
        self.std = var.sqrt() as f32 + 1e-8;
    }

    /// The `3d_fullres` configuration of a `plans.json`.
    pub fn from_plans_json(text: &str) -> Result<ModelConfig> {
        Self::from_plans_json_cfg(text, DEFAULT_CONFIGURATION)
    }

    /// The named configuration of a `plans.json`, in either format.
    pub fn from_plans_json_cfg(text: &str, configuration: &str) -> Result<ModelConfig> {
        let root: Value = serde_json::from_str(text).context("parse plans.json")?;
        let tf = root
            .get("transpose_forward")
            .and_then(|v| v.as_array())
            .context("plans.json: transpose_forward missing")?;
        let ident: Vec<i64> = tf.iter().filter_map(|v| v.as_i64()).collect();
        if ident != [0, 1, 2] {
            bail!("plans.json: unsupported transpose_forward {:?}", ident);
        }
        let cfg = resolve_configuration(&root, configuration)?;
        let norm = cfg
            .get("normalization_schemes")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let norm = match norm {
            "CTNormalization" => Norm::Ct,
            "ZScoreNormalization" => Norm::ZScore,
            other => bail!("plans.json: unsupported normalization scheme {other:?}"),
        };
        let patch_size = usize3(cfg.get("patch_size").context("patch_size")?, "patch_size")?;
        let spacing: Vec<f64> = cfg
            .get("spacing")
            .and_then(|v| v.as_array())
            .context("spacing")?
            .iter()
            .filter_map(|x| x.as_f64())
            .collect();
        let spacing: [f64; 3] = spacing
            .try_into()
            .map_err(|_| anyhow::anyhow!("plans.json: spacing is not length 3"))?;

        let net = match cfg.get("architecture") {
            Some(arch) => parse_new_architecture(arch)?,
            None => parse_old_architecture(&cfg)?,
        };
        let n_stages = net.features.len();
        if net.strides.len() != n_stages || net.kernels.len() != n_stages || n_stages < 2 {
            bail!("plans.json: inconsistent stage counts");
        }
        if net.n_conv_per_stage_decoder.len() != n_stages - 1 {
            bail!("plans.json: conv-per-stage lengths do not match stage count");
        }
        match &net.arch {
            Arch::PlainConv if net.n_conv_per_stage.len() != n_stages => {
                bail!("plans.json: conv-per-stage lengths do not match stage count")
            }
            Arch::ResidualEncoder { blocks_per_stage } if blocks_per_stage.len() != n_stages => {
                bail!("plans.json: blocks-per-stage length does not match stage count")
            }
            _ => {}
        }

        // A z-score model never reads these: its constants come from the
        // image in `apply_image_norm`. The values left here are the
        // identity, so a model that somehow skips that step is merely
        // un-normalized rather than scaled by nonsense.
        let fg = root.pointer("/foreground_intensity_properties_per_channel/0");
        let f = |key: &str| -> Result<f32> {
            if norm == Norm::ZScore {
                return Ok(match key {
                    "percentile_00_5" => f32::NEG_INFINITY,
                    "percentile_99_5" => f32::INFINITY,
                    "std" => 1.0,
                    _ => 0.0,
                });
            }
            fg.context("plans.json: intensity properties missing")?
                .get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .with_context(|| format!("plans.json: intensity {key}"))
        };
        Ok(ModelConfig {
            arch: net.arch,
            norm,
            patch_size,
            spacing,
            features: net.features,
            kernels: net.kernels,
            strides: net.strides,
            n_conv_per_stage: net.n_conv_per_stage,
            n_conv_per_stage_decoder: net.n_conv_per_stage_decoder,
            decoder_kernels: net.decoder_kernels,
            conv_bias: net.conv_bias,
            clip_lo: f("percentile_00_5")?,
            clip_hi: f("percentile_99_5")?,
            mean: f("mean")?,
            std: f("std")?,
        })
    }
}

/// The architecture half of a configuration, before the normalization
/// constants join it.
struct NetSpec {
    arch: Arch,
    features: Vec<usize>,
    kernels: Vec<[usize; 3]>,
    strides: Vec<[usize; 3]>,
    n_conv_per_stage: Vec<usize>,
    n_conv_per_stage_decoder: Vec<usize>,
    decoder_kernels: Vec<[usize; 3]>,
    conv_bias: bool,
}

/// `configurations[name]` with its `inherits_from` chain folded in: the
/// child's keys over the parent's, so `3d_fullres_high` is `3d_fullres`
/// with its own spacing.
fn resolve_configuration(root: &Value, name: &str) -> Result<Map<String, Value>> {
    let configs = root
        .get("configurations")
        .and_then(|v| v.as_object())
        .context("plans.json: no configurations")?;
    let mut chain: Vec<&Map<String, Value>> = Vec::new();
    let mut cur = name.to_string();
    loop {
        let cfg = configs
            .get(&cur)
            .and_then(|v| v.as_object())
            .with_context(|| format!("plans.json: no {cur} configuration"))?;
        chain.push(cfg);
        match cfg.get("inherits_from").and_then(|v| v.as_str()) {
            Some(parent) => {
                // A plans file has two or three levels at most; a longer
                // chain is a cycle.
                if chain.len() > 8 || parent == cur {
                    bail!("plans.json: inherits_from chain of {name} does not end");
                }
                cur = parent.to_string();
            }
            None => break,
        }
    }
    // Base first, then every child over it.
    let mut merged = Map::new();
    for cfg in chain.iter().rev() {
        for (k, v) in cfg.iter() {
            if k != "inherits_from" {
                merged.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(merged)
}

fn usize3(v: &Value, what: &str) -> Result<[usize; 3]> {
    let a: Vec<usize> = v
        .as_array()
        .with_context(|| format!("plans.json: {what} not an array"))?
        .iter()
        .filter_map(|x| x.as_u64().map(|u| u as usize))
        .collect();
    a.try_into()
        .map_err(|_| anyhow::anyhow!("plans.json: {what} is not length 3"))
}

fn usize_list(obj: &Map<String, Value>, key: &str) -> Result<Vec<usize>> {
    Ok(obj
        .get(key)
        .and_then(|v| v.as_array())
        .with_context(|| format!("plans.json: {key}"))?
        .iter()
        .filter_map(|x| x.as_u64().map(|u| u as usize))
        .collect())
}

fn usize3_list(obj: &Map<String, Value>, key: &str) -> Result<Vec<[usize; 3]>> {
    obj.get(key)
        .and_then(|v| v.as_array())
        .with_context(|| format!("plans.json: {key}"))?
        .iter()
        .map(|s| usize3(s, &format!("{key}[i]")))
        .collect()
}

/// nnU-Net 2.0 / 2.1: the architecture as flat keys of the configuration.
fn parse_old_architecture(cfg: &Map<String, Value>) -> Result<NetSpec> {
    if cfg.get("UNet_class_name").and_then(|v| v.as_str()) != Some("PlainConvUNet") {
        bail!(
            "plans.json: unsupported architecture {:?} (only PlainConvUNet)",
            cfg.get("UNet_class_name")
        );
    }
    let base = cfg
        .get("UNet_base_num_features")
        .and_then(|v| v.as_u64())
        .context("UNet_base_num_features")? as usize;
    let max_features = cfg
        .get("unet_max_num_features")
        .and_then(|v| v.as_u64())
        .context("unet_max_num_features")? as usize;
    let strides = usize3_list(cfg, "pool_op_kernel_sizes")?;
    let kernels = usize3_list(cfg, "conv_kernel_sizes")?;
    if strides.len() != kernels.len() || strides.is_empty() {
        bail!("plans.json: inconsistent stage counts");
    }
    let n_stages = strides.len();
    let features: Vec<usize> = (0..n_stages)
        .map(|i| (base << i.min(31)).min(max_features))
        .collect();
    Ok(NetSpec {
        arch: Arch::PlainConv,
        features,
        kernels,
        strides,
        n_conv_per_stage: usize_list(cfg, "n_conv_per_stage_encoder")?,
        n_conv_per_stage_decoder: usize_list(cfg, "n_conv_per_stage_decoder")?,
        decoder_kernels: Vec::new(),
        conv_bias: true,
    })
}

/// The last component of a dotted Python class path.
fn class_tail(v: Option<&Value>) -> &str {
    v.and_then(|v| v.as_str())
        .map(|s| s.rsplit('.').next().unwrap_or(s))
        .unwrap_or("")
}

/// nnU-Net 2.2 and later: the `architecture` object with the class name
/// and its keyword arguments.
fn parse_new_architecture(arch: &Value) -> Result<NetSpec> {
    let class = class_tail(arch.get("network_class_name"));
    let kw = arch
        .get("arch_kwargs")
        .and_then(|v| v.as_object())
        .context("plans.json: architecture.arch_kwargs missing")?;
    // The ops every released model uses; anything else would need code
    // this engine does not have, so it is refused rather than mis-run.
    let conv_op = class_tail(kw.get("conv_op"));
    if conv_op != "Conv3d" {
        bail!("plans.json: unsupported conv_op {conv_op:?} (only Conv3d)");
    }
    let norm_op = class_tail(kw.get("norm_op"));
    if norm_op != "InstanceNorm3d" {
        bail!("plans.json: unsupported norm_op {norm_op:?} (only InstanceNorm3d)");
    }
    if let Some(nk) = kw.get("norm_op_kwargs").and_then(|v| v.as_object()) {
        if nk.get("affine").and_then(|v| v.as_bool()) == Some(false) {
            bail!("plans.json: InstanceNorm3d without affine weights is not supported");
        }
        if let Some(eps) = nk.get("eps").and_then(|v| v.as_f64()) {
            if (eps - 1e-5).abs() > 1e-9 {
                bail!("plans.json: unsupported InstanceNorm3d eps {eps}");
            }
        }
    }
    let nonlin = class_tail(kw.get("nonlin"));
    if nonlin != "LeakyReLU" {
        bail!("plans.json: unsupported nonlin {nonlin:?} (only LeakyReLU)");
    }
    if let Some(slope) = kw
        .get("nonlin_kwargs")
        .and_then(|v| v.get("negative_slope"))
        .and_then(|v| v.as_f64())
    {
        if (slope - 0.01).abs() > 1e-9 {
            bail!("plans.json: unsupported LeakyReLU slope {slope}");
        }
    }
    if kw.get("dropout_op").is_some_and(|v| !v.is_null()) {
        bail!("plans.json: dropout layers are not supported");
    }
    let conv_bias = kw
        .get("conv_bias")
        .and_then(|v| v.as_bool())
        .context("plans.json: conv_bias")?;
    if !conv_bias {
        bail!("plans.json: convolutions without bias are not supported");
    }
    let n_stages = kw
        .get("n_stages")
        .and_then(|v| v.as_u64())
        .context("plans.json: n_stages")? as usize;
    let features = usize_list(kw, "features_per_stage")?;
    if features.len() != n_stages {
        bail!("plans.json: features_per_stage does not match n_stages");
    }
    let kernels = usize3_list(kw, "kernel_sizes")?;
    let strides = usize3_list(kw, "strides")?;
    let n_conv_per_stage_decoder = usize_list(kw, "n_conv_per_stage_decoder")?;
    // Not an nnU-Net key: written by the nnU-Net v1 rewrite (`v1.rs`).
    let decoder_kernels = match kw.get("decoder_kernel_sizes") {
        Some(_) => {
            let k = usize3_list(kw, "decoder_kernel_sizes")?;
            if k.len() + 1 != n_stages {
                bail!("plans.json: decoder_kernel_sizes does not match n_stages");
            }
            k
        }
        None => Vec::new(),
    };
    match class {
        "PlainConvUNet" => Ok(NetSpec {
            arch: Arch::PlainConv,
            features,
            kernels,
            strides,
            n_conv_per_stage: usize_list(kw, "n_conv_per_stage")?,
            n_conv_per_stage_decoder,
            decoder_kernels,
            conv_bias,
        }),
        "ResidualEncoderUNet" => {
            let blocks_per_stage = usize_list(kw, "n_blocks_per_stage")?;
            if blocks_per_stage.contains(&0) {
                bail!("plans.json: a residual stage with no block");
            }
            Ok(NetSpec {
                arch: Arch::ResidualEncoder { blocks_per_stage },
                features,
                kernels,
                strides,
                n_conv_per_stage: Vec::new(),
                n_conv_per_stage_decoder,
                decoder_kernels,
                conv_bias,
            })
        }
        other => bail!(
            "plans.json: unsupported architecture {other:?} (PlainConvUNet or ResidualEncoderUNet)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTENSITY: &str = r#""foreground_intensity_properties_per_channel": {"0": {
        "max": 3000.0, "mean": 120.5, "median": 100.0, "min": -1000.0,
        "percentile_00_5": -958.0, "percentile_99_5": 1180.0, "std": 310.2}}"#;

    fn old_format() -> String {
        format!(
            r#"{{"transpose_forward": [0, 1, 2], {INTENSITY},
            "configurations": {{"3d_fullres": {{
                "patch_size": [112, 112, 128], "spacing": [3.0, 3.0, 3.0],
                "normalization_schemes": ["CTNormalization"],
                "UNet_class_name": "PlainConvUNet",
                "UNet_base_num_features": 32, "unet_max_num_features": 320,
                "n_conv_per_stage_encoder": [2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2, 2, 2],
                "pool_op_kernel_sizes": [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
                "conv_kernel_sizes": [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
            }}}}}}"#
        )
    }

    fn new_format(class: &str, per_stage: &str) -> String {
        format!(
            r#"{{"transpose_forward": [0, 1, 2], {INTENSITY},
            "configurations": {{"3d_fullres": {{
                "patch_size": [112, 112, 128], "spacing": [3.0, 3.0, 3.0],
                "normalization_schemes": ["CTNormalization"],
                "architecture": {{
                    "network_class_name": "dynamic_network_architectures.architectures.unet.{class}",
                    "arch_kwargs": {{
                        "n_stages": 5, "features_per_stage": [32, 64, 128, 256, 320],
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "kernel_sizes": [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
                        "strides": [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
                        {per_stage},
                        "n_conv_per_stage_decoder": [2, 2, 2, 2],
                        "conv_bias": true,
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {{"eps": 1e-05, "affine": true}},
                        "dropout_op": null, "dropout_op_kwargs": null,
                        "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {{"inplace": true}}
                    }},
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"]
                }}
            }},
            "3d_fullres_high": {{"inherits_from": "3d_fullres", "spacing": [1.0, 0.75, 0.75],
                                 "data_identifier": "nnUNetPlans_3d_fullres_high"}}
            }}}}"#
        )
    }

    #[test]
    fn both_formats_describe_the_same_plain_network() {
        let old = ModelConfig::from_plans_json(&old_format()).unwrap();
        let new = ModelConfig::from_plans_json(&new_format(
            "PlainConvUNet",
            r#""n_conv_per_stage": [2, 2, 2, 2, 2]"#,
        ))
        .unwrap();
        for cfg in [&old, &new] {
            assert_eq!(cfg.arch, Arch::PlainConv);
            assert_eq!(cfg.features, vec![32, 64, 128, 256, 320]);
            assert_eq!(cfg.strides.len(), 5);
            assert_eq!(cfg.strides[0], [1, 1, 1]);
            assert_eq!(cfg.n_conv_per_stage, vec![2; 5]);
            assert_eq!(cfg.n_conv_per_stage_decoder, vec![2; 4]);
            assert_eq!(cfg.patch_size, [112, 112, 128]);
            assert_eq!(cfg.norm, Norm::Ct);
            assert!(cfg.conv_bias);
            assert_eq!(cfg.clip_lo, -958.0);
            assert_eq!(cfg.std, 310.2);
        }
    }

    #[test]
    fn the_residual_encoder_is_read_from_the_new_format() {
        let cfg = ModelConfig::from_plans_json(&new_format(
            "ResidualEncoderUNet",
            r#""n_blocks_per_stage": [1, 3, 4, 6, 6]"#,
        ))
        .unwrap();
        assert_eq!(
            cfg.arch,
            Arch::ResidualEncoder {
                blocks_per_stage: vec![1, 3, 4, 6, 6]
            }
        );
        assert!(cfg.n_conv_per_stage.is_empty());
        assert_eq!(cfg.n_conv_per_stage_decoder, vec![2; 4]);
    }

    #[test]
    fn an_inheriting_configuration_takes_the_parent_and_its_own_spacing() {
        let text = new_format("PlainConvUNet", r#""n_conv_per_stage": [2, 2, 2, 2, 2]"#);
        let high = ModelConfig::from_plans_json_cfg(&text, "3d_fullres_high").unwrap();
        assert_eq!(high.spacing, [1.0, 0.75, 0.75]);
        assert_eq!(high.patch_size, [112, 112, 128]);
        assert_eq!(high.features.len(), 5);
        let e = ModelConfig::from_plans_json_cfg(&text, "2d").unwrap_err();
        assert!(format!("{e:#}").contains("no 2d configuration"), "{e:#}");
    }

    #[test]
    fn what_the_engine_cannot_run_is_refused_by_name() {
        let other = new_format("UNetXYZ", r#""n_conv_per_stage": [2, 2, 2, 2, 2]"#);
        let e = format!("{:#}", ModelConfig::from_plans_json(&other).unwrap_err());
        assert!(e.contains("unsupported architecture"), "{e}");
        let no_bias = new_format("PlainConvUNet", r#""n_conv_per_stage": [2, 2, 2, 2, 2]"#)
            .replace(r#""conv_bias": true"#, r#""conv_bias": false"#);
        let e = format!("{:#}", ModelConfig::from_plans_json(&no_bias).unwrap_err());
        assert!(e.contains("without bias"), "{e}");
        let old_resenc = old_format().replace("PlainConvUNet", "ResidualEncoderUNet");
        assert!(ModelConfig::from_plans_json(&old_resenc).is_err());
    }
}
