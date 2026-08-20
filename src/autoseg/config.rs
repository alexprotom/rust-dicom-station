//! nnU-Net `plans.json` parsing → the static architecture + preprocessing
//! configuration of one TotalSegmentator model.
//!
//! Only the `3d_fullres` configuration is used (that is what all
//! TotalSegmentator CT models train). Array-valued fields are stored in the
//! nnU-Net array-axis order, i.e. the order of the model tensor's spatial
//! axes after the canonical reorientation ([S, A, R] — see `preprocess`).

use anyhow::{bail, Context, Result};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct ModelConfig {
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
    /// Convs per encoder stage (2 for these models).
    pub n_conv_per_stage: Vec<usize>,
    /// Convs per decoder stage.
    pub n_conv_per_stage_decoder: Vec<usize>,
    /// CT normalization: clip bounds (HU) then z-score.
    pub clip_lo: f32,
    pub clip_hi: f32,
    pub mean: f32,
    pub std: f32,
}

impl ModelConfig {
    pub fn n_stages(&self) -> usize {
        self.features.len()
    }

    pub fn from_plans_json(text: &str) -> Result<ModelConfig> {
        let root: Value = serde_json::from_str(text).context("parse plans.json")?;
        let tf = root
            .get("transpose_forward")
            .and_then(|v| v.as_array())
            .context("plans.json: transpose_forward missing")?;
        let ident: Vec<i64> = tf.iter().filter_map(|v| v.as_i64()).collect();
        if ident != [0, 1, 2] {
            bail!("plans.json: unsupported transpose_forward {:?}", ident);
        }
        let cfg = root
            .pointer("/configurations/3d_fullres")
            .context("plans.json: no 3d_fullres configuration")?;
        if cfg.pointer("/UNet_class_name").and_then(|v| v.as_str()) != Some("PlainConvUNet") {
            bail!(
                "plans.json: unsupported architecture {:?} (only PlainConvUNet)",
                cfg.pointer("/UNet_class_name")
            );
        }
        let norm = cfg
            .pointer("/normalization_schemes/0")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if norm != "CTNormalization" {
            bail!("plans.json: unsupported normalization scheme {norm:?}");
        }
        let usize3 = |v: &Value, what: &str| -> Result<[usize; 3]> {
            let a: Vec<usize> = v
                .as_array()
                .with_context(|| format!("plans.json: {what} not an array"))?
                .iter()
                .filter_map(|x| x.as_u64().map(|u| u as usize))
                .collect();
            a.try_into()
                .map_err(|_| anyhow::anyhow!("plans.json: {what} is not length 3"))
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
        let base = cfg
            .get("UNet_base_num_features")
            .and_then(|v| v.as_u64())
            .context("UNet_base_num_features")? as usize;
        let max_features = cfg
            .get("unet_max_num_features")
            .and_then(|v| v.as_u64())
            .context("unet_max_num_features")? as usize;
        let strides_v = cfg
            .get("pool_op_kernel_sizes")
            .and_then(|v| v.as_array())
            .context("pool_op_kernel_sizes")?;
        let kernels_v = cfg
            .get("conv_kernel_sizes")
            .and_then(|v| v.as_array())
            .context("conv_kernel_sizes")?;
        let mut strides = Vec::new();
        for s in strides_v {
            strides.push(usize3(s, "pool_op_kernel_sizes[i]")?);
        }
        let mut kernels = Vec::new();
        for k in kernels_v {
            kernels.push(usize3(k, "conv_kernel_sizes[i]")?);
        }
        if strides.len() != kernels.len() || strides.is_empty() {
            bail!("plans.json: inconsistent stage counts");
        }
        let n_stages = strides.len();
        let features: Vec<usize> = (0..n_stages)
            .map(|i| (base << i.min(31)).min(max_features))
            .collect();
        let ints = |key: &str| -> Result<Vec<usize>> {
            Ok(cfg
                .get(key)
                .and_then(|v| v.as_array())
                .with_context(|| format!("plans.json: {key}"))?
                .iter()
                .filter_map(|x| x.as_u64().map(|u| u as usize))
                .collect())
        };
        let n_conv_per_stage = ints("n_conv_per_stage_encoder")?;
        let n_conv_per_stage_decoder = ints("n_conv_per_stage_decoder")?;
        if n_conv_per_stage.len() != n_stages || n_conv_per_stage_decoder.len() != n_stages - 1 {
            bail!("plans.json: conv-per-stage lengths do not match stage count");
        }
        let fg = root
            .pointer("/foreground_intensity_properties_per_channel/0")
            .context("plans.json: intensity properties missing")?;
        let f = |key: &str| -> Result<f32> {
            fg.get(key)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .with_context(|| format!("plans.json: intensity {key}"))
        };
        Ok(ModelConfig {
            patch_size,
            spacing,
            features,
            kernels,
            strides,
            n_conv_per_stage,
            n_conv_per_stage_decoder,
            clip_lo: f("percentile_00_5")?,
            clip_hi: f("percentile_99_5")?,
            mean: f("mean")?,
            std: f("std")?,
        })
    }
}
