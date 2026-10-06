//! Model-weight acquisition and caching for the auto-segmentation module.
//!
//! First use of a model downloads the official TotalSegmentator weight zip
//! from its GitHub release (the *openly licensed*, Apache-2.0 "total" task
//! weights), extracts the nnU-Net `plans.json` + `checkpoint_final.pth`,
//! parses the PyTorch checkpoint natively, and caches the result as
//! `model.safetensors` + `plans.json` in the model directory. Subsequent
//! runs load the cache directly - no network access.
//!
//! Downloading, checkpoint parsing and the cache format are generic and live
//! in [`crate::nn`]; what stays here is the part that is specific to
//! TotalSegmentator - which models exist, where they are published, and how
//! to get `plans.json` and `checkpoint_final.pth` out of the release zip.
//!
//! Two generations of the `total` task are offered. **v2** (release
//! `v2.0.0-weights`, Datasets 291-298, 1559 subjects) is what
//! TotalSegmentator still runs by default and what this engine has always
//! run. **v3** (release `v3.0.0-weights`, Datasets 831-837) retrained the
//! organs, cardiac and muscles parts on 1830 subjects and ships every zip
//! with two trainings: the plain network (`nnUNetPlans`, TotalSegmentator's
//! `big` model size) and a residual-encoder network
//! (`nnUNetResEncUNetLPlans_8`, its `small` size, 8 base features). The
//! class table is v2's with id 26 renamed from `vertebrae_S1` to
//! `vertebrae_L6`. One zip therefore holds two [`ModelSpec`]s, told apart by
//! [`ModelSpec::plans`], and each is unpacked into its own folder.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

use super::config::{ModelConfig, DEFAULT_CONFIGURATION};
use crate::nn::cache::{self, ConvertSpec, WTensor};
use crate::progress::{ProgressSink, CANCELLED};
use crate::volume::AxisOrder;

/// One downloadable nnU-Net model.
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    /// Cache sub-directory name, e.g. "total_3mm".
    pub key: &'static str,
    /// Human-readable name shown in progress messages.
    pub label: &'static str,
    /// One line on what the model is for, shown by the model manager.
    pub detail: &'static str,
    pub url: &'static str,
    /// Approximate download size (progress display only).
    pub zip_bytes: u64,
    /// How many folds the zip's training holds and the cache keeps
    /// (`fold_0` .. `fold_{n-1}`); a run may ensemble all of them or use
    /// the first. One for almost every model.
    pub folds: u8,
    /// The axis order the network was trained in.
    pub axes: AxisOrder,
    /// Which engine folder of the model root the model lives in.
    pub home: Home,
    /// The nnU-Net plans identifier of the training to take out of the zip
    /// (`nnUNetPlans`, or `nnUNetResEncUNetLPlans_8` for the v3 `small`
    /// models). The training folder inside the zip is named
    /// `<trainer>__<plans>__<configuration>`.
    pub plans: &'static str,
    /// The nnU-Net configuration of that training (`3d_fullres` for every
    /// model here; the head and neck tasks train `3d_fullres_high`).
    pub configuration: &'static str,
    /// How the weights are obtained.
    pub source: Source,
}

/// Where a model's weights come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// A public release zip at `url` holding nnU-Net v2 trainings.
    Release,
    /// TotalSegmentator's licence server: the same kind of zip, sent for
    /// the user's licence number ([`set_ts_licence`]); `task` is the name
    /// the server knows the model by. `url` is unused and `zip_bytes` an
    /// estimate (the server publishes no sizes).
    Licensed { task: &'static str },
    /// An nnU-Net v1 pretrained-model archive at `url` (Zenodo, about 5 GB
    /// each); `task` names its dataset folder (`Task006_Lung`). Only one
    /// fold of `3d_fullres` is taken out of it, by HTTP range requests
    /// when the server allows them ([`super::v1`]).
    NnUnetV1 { task: &'static str },
    /// A trained nnU-Net v2 model folder on this computer (a training
    /// folder with `plans.json`, `dataset.json` and `fold_<k>/`), added by
    /// the user; `url` is its path. Converted into the model folder, never
    /// written to.
    Local,
}

impl ModelSpec {
    /// Bytes the first use downloads: the release zip, or for an nnU-Net
    /// v1 archive the one network taken out of it (an estimate: the
    /// archive index says, the table cannot).
    pub fn download_bytes(&self) -> u64 {
        match self.source {
            Source::NnUnetV1 { .. } => super::v1::FOLD_BYTES_ESTIMATE * self.folds.max(1) as u64,
            Source::Local => 0,
            _ => self.zip_bytes,
        }
    }
}

/// The engine folders nnU-Net models live in, under the model root. The
/// viewer's `models::Engine` names the same folders; a unit test there
/// keeps the two in step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Home {
    /// `totalsegmentator/`: every TotalSegmentator task.
    TotalSegmentator,
    /// `mrsegmentator/`: MRSegmentator.
    MrSegmentator,
    /// `nnunet_v1/`: the nnU-Net v1 pretrained models.
    NnUnetV1,
    /// `custom/`: the converted weights of model folders the user added.
    Custom,
}

impl Home {
    pub fn subdir(self) -> &'static str {
        match self {
            Home::TotalSegmentator => "totalsegmentator",
            Home::MrSegmentator => "mrsegmentator",
            Home::NnUnetV1 => "nnunet_v1",
            Home::Custom => "custom",
        }
    }
}

// ---- the TotalSegmentator licence number ------------------------------------

static TS_LICENCE: std::sync::RwLock<Option<String>> = std::sync::RwLock::new(None);

/// The user's TotalSegmentator licence number, which the licensed tasks'
/// downloads send ([`Source::Licensed`]). The application sets it from the
/// user's own settings file; it is never written anywhere by this crate,
/// never logged and never part of an error message.
pub fn set_ts_licence(number: Option<&str>) {
    let v = number
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from);
    if let Ok(mut l) = TS_LICENCE.write() {
        *l = v;
    }
}

/// Whether a licence number is set.
pub fn has_ts_licence() -> bool {
    TS_LICENCE.read().map(|l| l.is_some()).unwrap_or(false)
}

fn ts_licence() -> Option<String> {
    TS_LICENCE.read().ok().and_then(|l| l.clone())
}

/// TotalSegmentator's licence server.
pub const TS_LICENCE_SERVER: &str = "https://backend.totalsegmentator.com:443/download_weights";
/// The TotalSegmentator release whose tables the licensed tasks follow,
/// sent with each request as upstream's own client sends its version.
pub const TS_VERSION: &str = "2.18.0";

/// Fetch a licensed model's zip to `dest`: a POST with the licence number,
/// as TotalSegmentator's `download_model_with_license_and_unpack` makes it.
fn download_licensed(
    task: &str,
    dest: &Path,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    download_licensed_from(TS_LICENCE_SERVER, task, dest, size_hint, label, sink)
}

/// [`download_licensed`] from another server (the tests' own).
fn download_licensed_from(
    server: &str,
    task: &str,
    dest: &Path,
    size_hint: u64,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let Some(number) = ts_licence() else {
        bail!(
            "'{label}' is a licensed TotalSegmentator model: enter your TotalSegmentator \
             licence number in the model manager first"
        );
    };
    let body = serde_json::json!({
        "license_number": number,
        "task": task,
        "version": TS_VERSION,
    })
    .to_string();
    sink.report(
        0.0,
        &format!("Asking the TotalSegmentator licence server for {label}"),
    );
    // Upstream waits up to five minutes for the server to answer.
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(300))
        .build();
    let resp = match agent
        .post(server)
        .set("Content-Type", "application/json")
        .send_string(&body)
    {
        Ok(r) => r,
        Err(ureq::Error::Status(code, r)) => {
            let text = r.into_string().unwrap_or_default();
            if text.contains("invalid_license") {
                bail!("the TotalSegmentator licence server refused the licence number");
            }
            bail!("the TotalSegmentator licence server answered {code} for {label}");
        }
        Err(e) => bail!("TotalSegmentator licence server: {e}"),
    };
    let is_json = resp
        .header("Content-Type")
        .is_some_and(|t| t.contains("json"));
    if is_json {
        let text = resp.into_string().unwrap_or_default();
        if text.contains("invalid_license") {
            bail!("the TotalSegmentator licence server refused the licence number");
        }
        bail!("the TotalSegmentator licence server sent no weights for {label}");
    }
    cache::save_response(resp, dest, size_hint, label, sink)?;
    let mut magic = [0u8; 4];
    std::fs::File::open(dest)
        .and_then(|mut f| f.read_exact(&mut magic))
        .context("read the download")?;
    if &magic != b"PK\x03\x04" {
        let _ = std::fs::remove_file(dest);
        bail!("the TotalSegmentator licence server sent something that is not a zip for {label}");
    }
    Ok(())
}

/// The folder a spec's files live in, given the model root.
pub fn spec_dir(root: &Path, spec: &ModelSpec) -> std::path::PathBuf {
    root.join(spec.home.subdir()).join(spec.key)
}

pub(crate) const PLAIN: &str = "nnUNetPlans";
pub(crate) const RESENC_L8: &str = "nnUNetResEncUNetLPlans_8";

// ---- the `total` task, v2 ---------------------------------------------------

pub const SPEC_3MM: ModelSpec = ModelSpec {
    key: "total_3mm",
    label: "total 3 mm",
    detail: "All 117 structures at 3 mm - the fast default.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset297_TotalSegmentator_total_3mm_1559subj.zip",
    zip_bytes: 135_386_075,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: PLAIN,
    configuration: DEFAULT_CONFIGURATION,
};

pub const SPEC_6MM: ModelSpec = ModelSpec {
    key: "total_6mm",
    label: "total 6 mm",
    detail: "Coarse preview quality - the quickest look.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset298_TotalSegmentator_total_6mm_1559subj.zip",
    zip_bytes: 134_827_240,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: PLAIN,
    configuration: DEFAULT_CONFIGURATION,
};

pub const SPECS_15MM: [ModelSpec; 5] = [
    ModelSpec {
        key: "total_part1_organs",
        label: "1.5 mm organs (1/5)",
        detail: "Full-resolution sub-model; the five together are the reference quality.",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset291_TotalSegmentator_part1_organs_1559subj.zip",
        zip_bytes: 233_742_255,
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_part2_vertebrae",
        label: "1.5 mm vertebrae (2/5)",
        detail: "Full-resolution sub-model; the five together are the reference quality.",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset292_TotalSegmentator_part2_vertebrae_1532subj.zip",
        zip_bytes: 234_050_721,
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_part3_cardiac",
        label: "1.5 mm cardiac (3/5)",
        detail: "Full-resolution sub-model; the five together are the reference quality.",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset293_TotalSegmentator_part3_cardiac_1559subj.zip",
        zip_bytes: 234_190_318,
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_part4_muscles",
        label: "1.5 mm muscles (4/5)",
        detail: "Full-resolution sub-model; the five together are the reference quality.",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset294_TotalSegmentator_part4_muscles_1559subj.zip",
        zip_bytes: 233_625_081,
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_part5_ribs",
        label: "1.5 mm ribs (5/5)",
        detail: "Full-resolution sub-model; the five together are the reference quality.",
        url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset295_TotalSegmentator_part5_ribs_1559subj.zip",
        zip_bytes: 234_016_576,
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
];

// ---- the `total_v3` task ------------------------------------------------------

/// v3, plain network (`big`), one model at 3 mm.
pub const SPEC_V3_3MM: ModelSpec = ModelSpec {
    key: "total_v3_3mm",
    label: "total v3 3 mm",
    detail: "v3 (1830 training subjects), all 117 structures at 3 mm, plain network.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset836_TotalSegmentator_total_3mm_1559subj.zip",
    zip_bytes: 242_248_414,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: PLAIN,
    configuration: DEFAULT_CONFIGURATION,
};

/// v3, plain network, one model at 6 mm (the 6 mm zip has no `small`).
pub const SPEC_V3_6MM: ModelSpec = ModelSpec {
    key: "total_v3_6mm",
    label: "total v3 6 mm",
    detail: "v3 coarse preview - the quickest look.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset837_TotalSegmentator_total_6mm_1559subj.zip",
    zip_bytes: 132_026_264,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: PLAIN,
    configuration: DEFAULT_CONFIGURATION,
};

/// v3, residual encoder (`small`), one model at 3 mm - the same zip as
/// [`SPEC_V3_3MM`], the other training in it.
pub const SPEC_V3_SMALL_3MM: ModelSpec = ModelSpec {
    key: "total_v3_small_3mm",
    label: "total v3 small 3 mm",
    detail: "v3 residual-encoder network (TotalSegmentator's 'small'), all 117 structures \
             at 3 mm; fewer weights, a 128-cubed patch.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset836_TotalSegmentator_total_3mm_1559subj.zip",
    zip_bytes: 242_248_414,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: RESENC_L8,
    configuration: DEFAULT_CONFIGURATION,
};

const V3_PART_URLS: [&str; 5] = [
    "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset831_TotalSegmentator_part1_organs_1830subj.zip",
    "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset832_TotalSegmentator_part2_vertebrae_1559subj.zip",
    "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset833_TotalSegmentator_part3_cardiac_1830subj.zip",
    "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset834_TotalSegmentator_part4_muscles_1830subj.zip",
    "https://github.com/wasserth/TotalSegmentator/releases/download/v3.0.0-weights/Dataset835_TotalSegmentator_part5_ribs_1559subj.zip",
];
const V3_PART_BYTES: [u64; 5] = [
    434_915_575,
    334_822_528,
    334_155_206,
    334_231_870,
    334_519_368,
];

/// v3, plain network, the five 1.5 mm sub-models.
pub const SPECS_V3_15MM: [ModelSpec; 5] = [
    ModelSpec {
        key: "total_v3_part1_organs",
        label: "v3 1.5 mm organs (1/5)",
        detail: "v3 full-resolution sub-model (1830 subjects); the five together are the \
                 reference quality.",
        url: V3_PART_URLS[0],
        zip_bytes: V3_PART_BYTES[0],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_part2_vertebrae",
        label: "v3 1.5 mm vertebrae (2/5)",
        detail: "v3 full-resolution sub-model; label 26 is vertebrae_L6 here.",
        url: V3_PART_URLS[1],
        zip_bytes: V3_PART_BYTES[1],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_part3_cardiac",
        label: "v3 1.5 mm cardiac (3/5)",
        detail: "v3 full-resolution sub-model (1830 subjects).",
        url: V3_PART_URLS[2],
        zip_bytes: V3_PART_BYTES[2],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_part4_muscles",
        label: "v3 1.5 mm muscles (4/5)",
        detail: "v3 full-resolution sub-model (1830 subjects).",
        url: V3_PART_URLS[3],
        zip_bytes: V3_PART_BYTES[3],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_part5_ribs",
        label: "v3 1.5 mm ribs (5/5)",
        detail: "v3 full-resolution sub-model.",
        url: V3_PART_URLS[4],
        zip_bytes: V3_PART_BYTES[4],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: PLAIN,
        configuration: DEFAULT_CONFIGURATION,
    },
];

/// v3, residual encoder (`small`), the five 1.5 mm sub-models - the same
/// zips as [`SPECS_V3_15MM`], the other training in each.
pub const SPECS_V3_SMALL_15MM: [ModelSpec; 5] = [
    ModelSpec {
        key: "total_v3_small_part1_organs",
        label: "v3 small 1.5 mm organs (1/5)",
        detail: "v3 residual-encoder sub-model; a 192-cubed patch, GPU recommended.",
        url: V3_PART_URLS[0],
        zip_bytes: V3_PART_BYTES[0],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: RESENC_L8,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_small_part2_vertebrae",
        label: "v3 small 1.5 mm vertebrae (2/5)",
        detail: "v3 residual-encoder sub-model; label 26 is vertebrae_L6 here.",
        url: V3_PART_URLS[1],
        zip_bytes: V3_PART_BYTES[1],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: RESENC_L8,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_small_part3_cardiac",
        label: "v3 small 1.5 mm cardiac (3/5)",
        detail: "v3 residual-encoder sub-model.",
        url: V3_PART_URLS[2],
        zip_bytes: V3_PART_BYTES[2],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: RESENC_L8,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_small_part4_muscles",
        label: "v3 small 1.5 mm muscles (4/5)",
        detail: "v3 residual-encoder sub-model.",
        url: V3_PART_URLS[3],
        zip_bytes: V3_PART_BYTES[3],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: RESENC_L8,
        configuration: DEFAULT_CONFIGURATION,
    },
    ModelSpec {
        key: "total_v3_small_part5_ribs",
        label: "v3 small 1.5 mm ribs (5/5)",
        detail: "v3 residual-encoder sub-model.",
        url: V3_PART_URLS[4],
        zip_bytes: V3_PART_BYTES[4],
        folds: 1,
        axes: AxisOrder::Sar,
        home: Home::TotalSegmentator,
        source: Source::Release,
        plans: RESENC_L8,
        configuration: DEFAULT_CONFIGURATION,
    },
];

// ---- the body-outline task ------------------------------------------------------

/// The body-outline task's models. Same nnU-Net architecture, same open
/// Apache-2.0 licence, a different question: two classes, trunk and
/// extremities, whose union is the patient. They are what the body-contour
/// tool's model-assisted method uses to tell patient from equipment - see
/// the viewer's `bodymask` module.
pub const SPEC_BODY_15MM: ModelSpec = ModelSpec {
    key: "body_1_5mm",
    label: "body 1.5 mm",
    detail: "Patient outline at full resolution; slower, for the same decision.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset299_body_1559subj.zip",
    zip_bytes: 233_211_222,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: PLAIN,
    configuration: DEFAULT_CONFIGURATION,
};

pub const SPEC_BODY_6MM: ModelSpec = ModelSpec {
    key: "body_6mm",
    label: "body 6 mm",
    detail: "Patient outline, 6 mm - what the body-contour tool's model-assisted method \
             uses. Plenty, because it only decides which side of the skin a voxel is on.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset300_body_6mm_1559subj.zip",
    zip_bytes: 124_286_256,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: PLAIN,
    configuration: DEFAULT_CONFIGURATION,
};

/// The MR counterpart (TotalSegmentator v2.5 weights). Plans a z-score
/// normalization and an anisotropic 3.0 × 1.19 × 0.99 mm grid, which is why
/// the engine carries both.
pub const SPEC_BODY_MR: ModelSpec = ModelSpec {
    key: "body_mr",
    label: "body MR",
    detail: "Patient outline on MR - the body-contour tool's model for MR series.",
    url: "https://github.com/wasserth/TotalSegmentator/releases/download/v2.5.0-weights/Dataset597_mri_body_139subj.zip",
    zip_bytes: 229_810_326,
    folds: 1,
    axes: AxisOrder::Sar,
    home: Home::TotalSegmentator,
    source: Source::Release,
    plans: PLAIN,
    configuration: DEFAULT_CONFIGURATION,
};

/// Every published model, in the order the interface lists them: the v2
/// `total` models (the fast single model, the five full-resolution
/// sub-models, the preview model), then the v3 ones (plain, then residual),
/// the body-outline models, then every other open TotalSegmentator task
/// ([`super::tasks`]) and MRSegmentator.
///
/// The inventory in the viewer's `models` module walks this list; nothing
/// else needs to know that the 1.5 mm variants are five downloads rather
/// than one, or that two v3 specs share a zip.
pub fn all_specs() -> Vec<ModelSpec> {
    let mut v = vec![SPEC_3MM];
    v.extend(SPECS_15MM);
    v.push(SPEC_6MM);
    v.push(SPEC_V3_3MM);
    v.extend(SPECS_V3_15MM);
    v.push(SPEC_V3_6MM);
    v.push(SPEC_V3_SMALL_3MM);
    v.extend(SPECS_V3_SMALL_15MM);
    v.extend([SPEC_BODY_15MM, SPEC_BODY_6MM, SPEC_BODY_MR]);
    v.push(super::tasks::SPEC_BODY_MR_6MM);
    v.extend([
        super::tasks::SPEC_TOTAL_MR_3MM,
        super::tasks::SPEC_TOTAL_MR_PART1_ORGANS,
        super::tasks::SPEC_TOTAL_MR_PART2_MUSCLES,
        super::tasks::SPEC_TOTAL_MR_6MM,
    ]);
    for t in super::tasks::TASKS.iter() {
        for p in t.parts {
            if !v.iter().any(|s| s.key == p.spec.key) {
                v.push(p.spec);
            }
        }
    }
    v.push(SPEC_MRSEGMENTATOR);
    for t in super::tasks::LICENSED_TASKS
        .iter()
        .chain(super::v1::TASKS.iter())
    {
        for p in t.parts {
            if !v.iter().any(|s| s.key == p.spec.key) {
                v.push(p.spec);
            }
        }
    }
    v
}

/// MRSegmentator 1.2 (Häntze et al., Radiology 2025): one nnU-Net for 40
/// structures on MR (T1, T2, Dixon) and CT, five cross-validation folds in
/// one zip with `plans.json` at its root. Read through SimpleITK after
/// `DICOMOrient("LPS")`, so its arrays are `[S, P, L]`.
pub const SPEC_MRSEGMENTATOR: ModelSpec = ModelSpec {
    key: "mrsegmentator",
    label: "MRSegmentator",
    detail: "40 structures on MR (T1, T2, Dixon) and CT; five folds in one download.",
    url: "https://github.com/hhaentze/MRSegmentator/releases/download/v1.2.0/weights.zip",
    zip_bytes: 1_150_758_476,
    plans: PLAIN,
    // What the checkpoints were trained as: `3d_fullres` with batch size 8.
    configuration: "3d_fullres_bs8",
    folds: 5,
    axes: AxisOrder::Spl,
    home: Home::MrSegmentator,
    source: Source::Release,
};

/// The spec with this `key`, if any.
pub fn spec_by_key(key: &str) -> Option<ModelSpec> {
    all_specs().into_iter().find(|s| s.key == key)
}

/// A ready-to-run model: architecture config + named weight tensors, one
/// map per fold.
pub struct LoadedModel {
    pub spec: ModelSpec,
    pub config: ModelConfig,
    /// The folds, `fold_0` first; never empty.
    pub folds: Vec<HashMap<String, WTensor>>,
}

impl LoadedModel {
    /// The first fold's tensors - all a single-fold model has.
    pub fn tensors(&self) -> &HashMap<String, WTensor> {
        &self.folds[0]
    }
}

/// Converted-weight cache and the nnU-Net plan, written per model.
pub const CACHE_NAME: &str = "model.safetensors";
pub const PLANS_NAME: &str = "plans.json";
/// The checkpoint extracted from the release zip; deleted after conversion.
pub const CHECKPOINT_TMP: &str = "checkpoint.tmp.pth";
/// The release zip while it is being fetched; deleted after unpacking.
pub const DOWNLOAD_TMP: &str = "download.zip.tmp";

/// The cache file of one fold: `model.safetensors` for fold 0 (the name a
/// single-fold model has always had), `model_fold<k>.safetensors` after it.
pub fn fold_cache_name(fold: u8) -> String {
    if fold == 0 {
        CACHE_NAME.to_string()
    } else {
        format!("model_fold{fold}.safetensors")
    }
}

/// The unpacked checkpoint of one fold while it waits for conversion.
pub fn fold_checkpoint_tmp(fold: u8) -> String {
    if fold == 0 {
        CHECKPOINT_TMP.to_string()
    } else {
        format!("checkpoint_fold{fold}.tmp.pth")
    }
}

/// Every file a ready model consists of: the plan and one cache per fold.
pub fn ready_files(spec: &ModelSpec) -> Vec<String> {
    let mut v = vec![PLANS_NAME.to_string()];
    v.extend((0..spec.folds.max(1)).map(fold_cache_name));
    v
}

/// Files a download leaves behind that nothing reads once the model is
/// ready.
pub fn spare_files(spec: &ModelSpec) -> Vec<String> {
    let mut v: Vec<String> = (0..spec.folds.max(1)).map(fold_checkpoint_tmp).collect();
    v.push(DOWNLOAD_TMP.to_string());
    v
}

/// Load a model with every fold it has, downloading + converting on first
/// use. `models_dir` is the spec's engine folder (`<root>/totalsegmentator`,
/// or `<root>/mrsegmentator` - see [`spec_dir`]).
pub fn ensure_model(
    spec: &ModelSpec,
    models_dir: &Path,
    sink: &dyn ProgressSink,
) -> Result<LoadedModel> {
    ensure_model_folds(spec, models_dir, u8::MAX, sink)
}

/// [`ensure_model`], loading at most `max_folds` folds (`fold_0` first).
/// Every fold is still converted on first use: they come in one download,
/// and a later run that wants them all should not fetch it again.
pub fn ensure_model_folds(
    spec: &ModelSpec,
    models_dir: &Path,
    max_folds: u8,
    sink: &dyn ProgressSink,
) -> Result<LoadedModel> {
    let dir = models_dir.join(spec.key);
    let plans_path = dir.join(PLANS_NAME);
    let n = spec.folds.max(1);
    // `decoder.encoder.*` and `*.all_modules.*` are duplicate registrations
    // of the same storages; drop them.
    let convert = ConvertSpec {
        top_key: "network_weights",
        keep: &|name, _| !name.starts_with("decoder.encoder.") && !name.contains(".all_modules."),
        rename: &|name| name.to_string(),
        label: spec.label,
    };
    let missing: Vec<u8> = (0..n)
        .filter(|f| !dir.join(fold_cache_name(*f)).is_file())
        .collect();
    let load = n.min(max_folds.max(1));
    let mut fresh: HashMap<u8, HashMap<String, WTensor>> = HashMap::new();
    if !missing.is_empty() || !plans_path.is_file() {
        let ckpts = match spec.source {
            Source::NnUnetV1 { .. } => super::v1::fetch(spec, &dir, &missing, sink),
            Source::Local => local_entries(spec, &dir, &missing),
            _ => download_and_unpack(spec, &dir, &missing, sink),
        }
        .with_context(|| format!("prepare model '{}'", spec.label))?;
        for (f, ckpt) in ckpts {
            let cache_path = dir.join(fold_cache_name(f));
            let t = match spec.source {
                Source::NnUnetV1 { .. } => {
                    super::v1::convert(&ckpt, &cache_path, spec.label, sink)?
                }
                _ => cache::convert_checkpoint(&ckpt, &cache_path, &convert, sink)?,
            };
            // A user's own checkpoint stays where it is.
            if spec.source != Source::Local {
                let _ = std::fs::remove_file(&ckpt);
            }
            if f < load {
                fresh.insert(f, t);
            }
        }
    }
    let mut folds = Vec::with_capacity(load as usize);
    for f in 0..load {
        match fresh.remove(&f) {
            Some(t) => folds.push(t),
            None => {
                sink.report(0.0, &format!("Loading weights ({})", spec.label));
                folds.push(cache::load_safetensors(&dir.join(fold_cache_name(f)))?);
            }
        }
    }
    let plans_text = std::fs::read_to_string(&plans_path)
        .with_context(|| format!("read {}", plans_path.display()))?;
    let config = ModelConfig::from_plans_json_cfg(&plans_text, spec.configuration)?;
    Ok(LoadedModel {
        spec: *spec,
        config,
        folds,
    })
}

/// True when the model's converted cache is already present.
pub fn is_cached(spec: &ModelSpec, models_dir: &Path) -> bool {
    let dir = models_dir.join(spec.key);
    ready_files(spec).iter().all(|f| dir.join(f).is_file())
}

/// The entries of a release zip a spec needs.
#[derive(Debug, PartialEq)]
pub struct ZipEntries {
    pub plans: String,
    /// `(fold, entry name)` of each fold's `checkpoint_final.pth`.
    pub checkpoints: Vec<(u8, String)>,
}

/// The `plans.json` and the fold checkpoints of the training `spec` names,
/// out of the zip's entry list.
///
/// A release zip holds one training folder per trainer and plans
/// (`<trainer>__<plans>__<configuration>/`); the v3 zips hold two. Entries
/// are taken from the folder whose name ends in the spec's plans and
/// configuration. A zip with a single training and no such folder (a
/// `plans.json` at its root, as MRSegmentator ships) is accepted as it is.
/// Every fold `0..spec.folds` must be there.
pub fn pick_entries(names: &[String], spec: &ModelSpec) -> Result<ZipEntries> {
    let marker = format!("__{}__{}/", spec.plans, spec.configuration);
    let live: Vec<&String> = names.iter().filter(|n| !n.contains("__MACOSX")).collect();
    let in_folder: Vec<&String> = live
        .iter()
        .copied()
        .filter(|n| n.contains(&marker))
        .collect();
    let pool: Vec<&String> = if in_folder.is_empty() {
        let plans: Vec<&&String> = live.iter().filter(|n| is_file(n, "plans.json")).collect();
        if plans.len() != 1 {
            bail!(
                "weights zip: no training folder '{}' (and {} plans.json files)",
                marker.trim_end_matches('/'),
                plans.len()
            );
        }
        live.clone()
    } else {
        in_folder
    };
    let plans = pool
        .iter()
        .find(|n| is_file(n, "plans.json"))
        .with_context(|| format!("weights zip: no plans.json for {}", spec.plans))?;
    let mut checkpoints = Vec::new();
    for f in 0..spec.folds.max(1) {
        let want = format!("fold_{f}/checkpoint_final.pth");
        let c = pool
            .iter()
            .find(|n| n.ends_with(&want))
            .with_context(|| format!("weights zip: no {want} for {}", spec.plans))?;
        checkpoints.push((f, (*c).clone()));
    }
    Ok(ZipEntries {
        plans: (*plans).clone(),
        checkpoints,
    })
}

/// Whether zip entry `name` is a file called `file` (not a macOS `._`
/// shadow of one).
fn is_file(name: &str, file: &str) -> bool {
    name == file || name.ends_with(&format!("/{file}"))
}

/// A model folder of the user's: copy its `plans.json` into the cache
/// folder and point the conversion at its checkpoints in place.
fn local_entries(
    spec: &ModelSpec,
    dir: &Path,
    folds: &[u8],
) -> Result<Vec<(u8, std::path::PathBuf)>> {
    let src = Path::new(spec.url);
    let plans = std::fs::read_to_string(src.join(PLANS_NAME))
        .with_context(|| format!("read {}", src.join(PLANS_NAME).display()))?;
    ModelConfig::from_plans_json_cfg(&plans, spec.configuration)?;
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    std::fs::write(dir.join(PLANS_NAME), &plans)?;
    let mut out = Vec::new();
    for &f in folds {
        let fold = src.join(format!("fold_{f}"));
        let ckpt = ["checkpoint_final.pth", "checkpoint_best.pth"]
            .iter()
            .map(|n| fold.join(n))
            .find(|p| p.is_file())
            .with_context(|| format!("no checkpoint_final.pth in {}", fold.display()))?;
        out.push((f, ckpt));
    }
    Ok(out)
}

/// Download the release zip and pull `plans.json` and the checkpoints of
/// the folds in `folds` out of it. Returns `(fold, checkpoint path)` pairs,
/// ready for conversion.
fn download_and_unpack(
    spec: &ModelSpec,
    dir: &Path,
    folds: &[u8],
    sink: &dyn ProgressSink,
) -> Result<Vec<(u8, std::path::PathBuf)>> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let zip_tmp = dir.join(DOWNLOAD_TMP);
    // ---- download --------------------------------------------------------
    let label = format!("weights ({})", spec.label);
    match spec.source {
        Source::Licensed { task } => {
            download_licensed(task, &zip_tmp, spec.zip_bytes, &label, sink)?
        }
        _ => cache::download_to_file(spec.url, &zip_tmp, spec.zip_bytes, &label, sink)?,
    }
    // ---- extract the files we need ---------------------------------------
    sink.report(0.0, &format!("Unpacking weights ({})", spec.label));
    let mut out = Vec::new();
    {
        let file = std::fs::File::open(&zip_tmp)?;
        let mut zip = zip::ZipArchive::new(file).context("weights zip")?;
        let mut names = Vec::with_capacity(zip.len());
        for i in 0..zip.len() {
            names.push(zip.by_index_raw(i)?.name().to_owned());
        }
        let entries = pick_entries(&names, spec)?;
        let mut plans = String::new();
        zip.by_name(&entries.plans)?
            .read_to_string(&mut plans)
            .context("read plans.json")?;
        // Validate before persisting anything.
        ModelConfig::from_plans_json_cfg(&plans, spec.configuration)?;
        std::fs::write(dir.join(PLANS_NAME), &plans)?;
        for (f, name) in entries.checkpoints {
            if !folds.contains(&f) {
                continue;
            }
            let ckpt_tmp = dir.join(fold_checkpoint_tmp(f));
            let mut ckpt_entry = zip.by_name(&name)?;
            let mut w = std::io::BufWriter::new(std::fs::File::create(&ckpt_tmp)?);
            let total = ckpt_entry.size();
            let mut buf = vec![0u8; 1024 * 1024];
            let mut done: u64 = 0;
            loop {
                if sink.cancelled() {
                    bail!(CANCELLED);
                }
                let k = ckpt_entry.read(&mut buf).context("unpack checkpoint")?;
                if k == 0 {
                    break;
                }
                w.write_all(&buf[..k])?;
                done += k as u64;
                sink.report(
                    done as f32 / total.max(1) as f32,
                    &format!("Unpacking weights ({}, fold {f})", spec.label),
                );
            }
            w.flush().ok();
            out.push((f, ckpt_tmp));
        }
    }
    let _ = std::fs::remove_file(&zip_tmp);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_and_every_spec_names_a_training() {
        let specs = all_specs();
        let mut keys: Vec<&str> = specs.iter().map(|s| s.key).collect();
        keys.sort();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n, "duplicate spec key");
        for s in &specs {
            match s.source {
                Source::Release => {
                    assert!(s.url.ends_with(".zip"), "{}", s.key);
                    assert!(s.plans == PLAIN || s.plans == RESENC_L8);
                }
                Source::Licensed { task } => assert!(!task.is_empty(), "{}", s.key),
                Source::NnUnetV1 { task } => assert!(s.url.contains(task), "{}", s.key),
                Source::Local => panic!("{}: a built-in spec cannot be local", s.key),
            }
            assert!(s.zip_bytes > 0 && !s.label.is_empty() && !s.detail.is_empty());
            assert!(s.configuration.starts_with("3d_"), "{}", s.key);
            assert!(s.folds >= 1);
        }
        assert_eq!(
            spec_by_key("total_v3_small_3mm").map(|s| s.plans),
            Some(RESENC_L8)
        );
        assert!(spec_by_key("nope").is_none());
        // The two v3 trainings of one zip share the download and nothing else.
        assert_eq!(SPEC_V3_3MM.url, SPEC_V3_SMALL_3MM.url);
        assert_ne!(SPEC_V3_3MM.key, SPEC_V3_SMALL_3MM.key);
    }

    /// One request to a loopback server that answers `status` and `body`;
    /// returns its URL and the request's body as the server received it.
    fn licence_server(
        status: &'static str,
        content_type: &'static str,
        body: Vec<u8>,
    ) -> (String, std::sync::mpsc::Receiver<String>) {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/download_weights", listener.local_addr().unwrap());
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                let h = h.trim_end().to_string();
                if h.is_empty() {
                    break;
                }
                if let Some((k, v)) = h.split_once(':') {
                    if k.eq_ignore_ascii_case("content-length") {
                        len = v.trim().parse().unwrap();
                    }
                }
            }
            let mut req = vec![0u8; len];
            r.read_exact(&mut req).unwrap();
            tx.send(String::from_utf8(req).unwrap()).unwrap();
            let head = format!(
                "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&body).unwrap();
        });
        (url, rx)
    }

    struct Quiet;
    impl ProgressSink for Quiet {
        fn report(&self, _: f32, _: &str) {}
        fn cancelled(&self) -> bool {
            false
        }
    }

    /// The request is upstream's (`license_number`, `task`, `version`), a
    /// zip answer is saved, a refusal says so without the number, and with
    /// no number nothing is sent at all. One test, since the number is
    /// process-wide.
    #[test]
    fn licensed_downloads_ask_as_upstream_does() {
        let dir = std::env::temp_dir().join("rds_licensed_download");
        std::fs::create_dir_all(&dir).unwrap();
        let dest = dir.join("dl.zip");

        set_ts_licence(None);
        assert!(!has_ts_licence());
        let err = download_licensed_from("http://127.0.0.1:9/x", "face", &dest, 0, "Face", &Quiet)
            .unwrap_err();
        assert!(format!("{err:#}").contains("licence number"), "{err:#}");

        set_ts_licence(Some("  aca_TEST0000 "));
        assert!(has_ts_licence());
        let mut zip_bytes = std::io::Cursor::new(Vec::new());
        {
            let mut z = zip::ZipWriter::new(&mut zip_bytes);
            let o: zip::write::FileOptions<()> = zip::write::FileOptions::default();
            z.start_file("Dataset303_face_1559subj/plans.json", o)
                .unwrap();
            z.write_all(b"{}").unwrap();
            z.finish().unwrap();
        }
        let (url, req) = licence_server("200 OK", "application/zip", zip_bytes.into_inner());
        download_licensed_from(&url, "face", &dest, 0, "Face", &Quiet).unwrap();
        let sent: serde_json::Value = serde_json::from_str(&req.recv().unwrap()).unwrap();
        assert_eq!(sent["license_number"], "aca_TEST0000");
        assert_eq!(sent["task"], "face");
        assert_eq!(sent["version"], TS_VERSION);
        assert!(std::fs::read(&dest).unwrap().starts_with(b"PK"));

        let (url, _rx) = licence_server(
            "401 Unauthorized",
            "application/json",
            br#"{"status": "invalid_license"}"#.to_vec(),
        );
        let err = download_licensed_from(&url, "face", &dest, 0, "Face", &Quiet).unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("refused"), "{text}");
        assert!(
            !text.contains("aca_TEST0000"),
            "the number never shows: {text}"
        );

        let (url, _rx) =
            licence_server("200 OK", "text/html", b"<html>maintenance</html>".to_vec());
        assert!(download_licensed_from(&url, "face", &dest, 0, "Face", &Quiet).is_err());
        assert!(!dest.exists(), "a download that is not a zip is not kept");
        set_ts_licence(None);
    }

    #[test]
    fn the_training_folder_is_chosen_by_plans_name() {
        let names: Vec<String> = [
            "D836/",
            "D836/.DS_Store",
            "D836/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/plans.json",
            "D836/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
            "D836/nnUNetTrainer_4000epochs_NoMirroring__nnUNetResEncUNetLPlans_8__3d_fullres/plans.json",
            "D836/nnUNetTrainer_4000epochs_NoMirroring__nnUNetResEncUNetLPlans_8__3d_fullres/fold_0/checkpoint_final.pth",
            "__MACOSX/D836/._plans.json",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let e = pick_entries(&names, &SPEC_V3_3MM).unwrap();
        assert!(
            e.plans.contains("__nnUNetPlans__3d_fullres/plans.json"),
            "{e:?}"
        );
        assert_eq!(e.checkpoints.len(), 1);
        assert!(
            e.checkpoints[0]
                .1
                .contains("__nnUNetPlans__3d_fullres/fold_0/"),
            "{e:?}"
        );
        let e = pick_entries(&names, &SPEC_V3_SMALL_3MM).unwrap();
        assert!(
            e.plans
                .contains("ResEncUNetLPlans_8__3d_fullres/plans.json"),
            "{e:?}"
        );
        assert!(
            e.checkpoints[0]
                .1
                .contains("ResEncUNetLPlans_8__3d_fullres/fold_0/"),
            "{e:?}"
        );
    }

    #[test]
    fn a_single_training_at_the_zip_root_is_accepted() {
        let names: Vec<String> = [
            "plans.json",
            "dataset.json",
            "fold_0/checkpoint_final.pth",
            "fold_1/checkpoint_final.pth",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let e = pick_entries(&names, &SPEC_3MM).unwrap();
        assert_eq!(
            e,
            ZipEntries {
                plans: "plans.json".into(),
                checkpoints: vec![(0, "fold_0/checkpoint_final.pth".into())],
            }
        );
        // MRSegmentator's zip: five folds at the root, all of them taken.
        let mut five: Vec<String> = vec!["plans.json".into(), "dataset.json".into()];
        five.extend((0..5).map(|f| format!("fold_{f}/checkpoint_final.pth")));
        let e = pick_entries(&five, &SPEC_MRSEGMENTATOR).unwrap();
        assert_eq!(e.checkpoints.len(), 5);
        assert_eq!(
            e.checkpoints[4],
            (4, "fold_4/checkpoint_final.pth".to_string())
        );
        // A fold short is refused.
        assert!(pick_entries(&five[..6], &SPEC_MRSEGMENTATOR).is_err());
        // A macOS shadow file is not a plans.json.
        let shadow: Vec<String> = ["._plans.json", "plans.json", "fold_0/checkpoint_final.pth"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            pick_entries(&shadow, &SPEC_3MM).unwrap().plans,
            "plans.json"
        );
        // Two trainings and neither is the one asked for: refused, not guessed.
        let two: Vec<String> = [
            "a/t__otherPlans__3d_fullres/plans.json",
            "a/t__otherPlans__3d_fullres/fold_0/checkpoint_final.pth",
            "a/t__thirdPlans__3d_fullres/plans.json",
            "a/t__thirdPlans__3d_fullres/fold_0/checkpoint_final.pth",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert!(pick_entries(&two, &SPEC_3MM).is_err());
    }
}
