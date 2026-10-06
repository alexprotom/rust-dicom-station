//! Whole-body CT segmentation with MONAI's SegResNet family - the MONAI
//! model zoo's `wholeBody_ct_segmentation` bundle (104 classes of
//! TotalSegmentator v1, at 1.5 mm and 3 mm; Apache-2.0) and CT-FM's
//! whole-body model (Pai et al., *Vision foundation models for computed
//! tomography*, 2025: a SegResNetDS fine-tuned on TotalSegmentator v2's 117
//! classes; Apache-2.0) - re-implemented natively ([`net`]).
//!
//! The two follow their own published pipelines to the transform
//! ([`pre`]):
//!
//! * MONAI bundle: RAS axes, resampled to 1.5 (3) mm, normalized over the
//!   non-zero voxels and scaled to [-1, 1]; 96-cubed windows, overlap
//!   0.25, Gaussian weights, edge-replicated padding; the argmax resampled
//!   back with nearest-neighbour.
//! * CT-FM: SPL axes at the scan's own spacing, HU -1024..2048 scaled to
//!   [0, 1], cropped to the foreground; 96 x 160 x 160 windows, overlap
//!   0.625; the argmax kept as each label's largest connected piece.
//!
//! The windows are [`crate::autoseg::infer`]'s; the network runs through
//! `burn`, on the GPU when there is one.

pub mod net;
pub mod pre;

use anyhow::{bail, Result};
use std::path::Path;

use crate::autoseg::classes::TOTAL_CLASS_NAMES;
use crate::autoseg::infer::{self, Fill, InferHooks};
use crate::autoseg::{organ_hits, AutosegResult};
use crate::medsam2::ops;
use crate::nn::cache::{self, ConvertSpec, RemoteFile};
use crate::nn::device::DevicePref;
use crate::nn::params::Params;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::volume::{AxisOrder, Volume};
use net::NormKind;

/// The engine folders under the model root.
pub const DIR_MONAI: &str = "monai_wholebody";
pub const DIR_CTFM: &str = "ctfm";

/// TotalSegmentator v1's 104 classes, the MONAI bundle's output 1..=104
/// (its `metadata.json`).
pub const CLASSES_TS_V1: [&str; 104] = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "vertebrae_L5",
    "vertebrae_L4",
    "vertebrae_L3",
    "vertebrae_L2",
    "vertebrae_L1",
    "vertebrae_T12",
    "vertebrae_T11",
    "vertebrae_T10",
    "vertebrae_T9",
    "vertebrae_T8",
    "vertebrae_T7",
    "vertebrae_T6",
    "vertebrae_T5",
    "vertebrae_T4",
    "vertebrae_T3",
    "vertebrae_T2",
    "vertebrae_T1",
    "vertebrae_C7",
    "vertebrae_C6",
    "vertebrae_C5",
    "vertebrae_C4",
    "vertebrae_C3",
    "vertebrae_C2",
    "vertebrae_C1",
    "esophagus",
    "trachea",
    "heart_myocardium",
    "heart_atrium_left",
    "heart_ventricle_left",
    "heart_atrium_right",
    "heart_ventricle_right",
    "pulmonary_artery",
    "brain",
    "iliac_artery_left",
    "iliac_artery_right",
    "iliac_vena_left",
    "iliac_vena_right",
    "small_bowel",
    "duodenum",
    "colon",
    "rib_left_1",
    "rib_left_2",
    "rib_left_3",
    "rib_left_4",
    "rib_left_5",
    "rib_left_6",
    "rib_left_7",
    "rib_left_8",
    "rib_left_9",
    "rib_left_10",
    "rib_left_11",
    "rib_left_12",
    "rib_right_1",
    "rib_right_2",
    "rib_right_3",
    "rib_right_4",
    "rib_right_5",
    "rib_right_6",
    "rib_right_7",
    "rib_right_8",
    "rib_right_9",
    "rib_right_10",
    "rib_right_11",
    "rib_right_12",
    "humerus_left",
    "humerus_right",
    "scapula_left",
    "scapula_right",
    "clavicula_left",
    "clavicula_right",
    "femur_left",
    "femur_right",
    "hip_left",
    "hip_right",
    "sacrum",
    "face",
    "gluteus_maximus_left",
    "gluteus_maximus_right",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "gluteus_minimus_left",
    "gluteus_minimus_right",
    "autochthon_left",
    "autochthon_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "urinary_bladder",
];

/// One SegResNet-family model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegResModel {
    /// MONAI `wholeBody_ct_segmentation`, the 1.5 mm model.
    MonaiWholeBody,
    /// The same bundle's 3 mm model.
    MonaiWholeBodyLowres,
    /// CT-FM whole-body segmentation (118 outputs: background + the 117
    /// TotalSegmentator v2 classes).
    CtFm,
}

impl SegResModel {
    pub const ALL: [SegResModel; 3] = [
        SegResModel::MonaiWholeBody,
        SegResModel::MonaiWholeBodyLowres,
        SegResModel::CtFm,
    ];

    pub fn key(self) -> &'static str {
        match self {
            SegResModel::MonaiWholeBody => "monai_wholebody",
            SegResModel::MonaiWholeBodyLowres => "monai_wholebody_lowres",
            SegResModel::CtFm => "ctfm_wholebody",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SegResModel::MonaiWholeBody => "MONAI whole body, 1.5 mm",
            SegResModel::MonaiWholeBodyLowres => "MONAI whole body, 3 mm",
            SegResModel::CtFm => "CT-FM whole body",
        }
    }

    pub fn detail(self) -> &'static str {
        match self {
            SegResModel::MonaiWholeBody => {
                "MONAI model zoo SegResNet on TotalSegmentator v1's 104 classes (heart \
                 chambers, myocardium, pulmonary artery, iliac vessels), 1.5 mm."
            }
            SegResModel::MonaiWholeBodyLowres => {
                "The MONAI whole-body SegResNet at 3 mm - quicker, coarser."
            }
            SegResModel::CtFm => {
                "CT-FM, a CT foundation model fine-tuned on TotalSegmentator v2's 117 \
                 classes; runs at the scan's own resolution (GPU recommended, memory-heavy)."
            }
        }
    }

    pub fn dir(self) -> &'static str {
        match self {
            SegResModel::CtFm => DIR_CTFM,
            _ => DIR_MONAI,
        }
    }

    /// The published weights.
    pub fn file(self) -> RemoteFile {
        match self {
            SegResModel::MonaiWholeBody => RemoteFile {
                name: "model.pt",
                url: "https://huggingface.co/MONAI/wholeBody_ct_segmentation/resolve/main/models/model.pt",
                bytes: 75_225_922,
            },
            SegResModel::MonaiWholeBodyLowres => RemoteFile {
                name: "model_lowres.pt",
                url: "https://huggingface.co/MONAI/wholeBody_ct_segmentation/resolve/main/models/model_lowres.pt",
                bytes: 75_225_922,
            },
            SegResModel::CtFm => RemoteFile {
                name: "model.safetensors",
                url: "https://huggingface.co/project-lighter/whole_body_segmentation/resolve/main/model.safetensors",
                bytes: 348_968_896,
            },
        }
    }

    /// The converted cache (CT-FM is published as safetensors and needs
    /// none: the download is read as it is).
    pub fn cache_name(self) -> &'static str {
        match self {
            SegResModel::MonaiWholeBody => "model.safetensors",
            SegResModel::MonaiWholeBodyLowres => "model_lowres.safetensors",
            SegResModel::CtFm => "model.safetensors",
        }
    }

    pub fn classes(self) -> &'static [&'static str] {
        match self {
            SegResModel::CtFm => &TOTAL_CLASS_NAMES,
            _ => &CLASSES_TS_V1,
        }
    }

    fn spacing(self) -> Option<f64> {
        match self {
            SegResModel::MonaiWholeBody => Some(1.5),
            SegResModel::MonaiWholeBodyLowres => Some(3.0),
            SegResModel::CtFm => None,
        }
    }
}

/// Bytes still to download before `model` runs offline.
pub fn download_needed(model: SegResModel, root: &Path) -> u64 {
    let dir = root.join(model.dir());
    if dir.join(model.cache_name()).is_file() || model.file().is_cached(&dir) {
        0
    } else {
        model.file().bytes
    }
}

/// The model's weights, downloading (and converting) on first use.
pub fn load(model: SegResModel, root: &Path, sink: &dyn ProgressSink) -> Result<Params> {
    let dir = root.join(model.dir());
    if model == SegResModel::CtFm {
        let path = model.file().ensure(&dir, sink)?;
        sink.report(0.0, &format!("Loading weights ({})", model.label()));
        return Ok(Params::new(cache::load_safetensors(&path)?));
    }
    let spec = ConvertSpec {
        // MONAI bundles save either the state dict or {"model": ...}.
        top_key: "?model",
        keep: &|name, _| !name.ends_with("num_batches_tracked"),
        rename: &|name| name.to_string(),
        label: model.label(),
    };
    let t = cache::ensure_converted(
        &dir.join(model.cache_name()),
        &spec,
        || model.file().ensure(&dir, sink),
        sink,
    )?;
    Ok(Params::new(t))
}

/// A loaded network, whichever of the two.
pub enum Net<B: burn::tensor::backend::Backend> {
    Plain(net::SegResNet<B>),
    Ds(net::SegResNetDs<B>),
}

impl<B: burn::tensor::backend::Backend> Net<B> {
    pub fn load(model: SegResModel, p: &Params, dev: &B::Device) -> Result<Self> {
        Ok(match model {
            SegResModel::CtFm => Net::Ds(net::SegResNetDs::load(
                p,
                "",
                32,
                &[1, 2, 2, 4, 4],
                118,
                NormKind::Batch,
                Some("up_layers"),
                None,
                true,
                dev,
            )?),
            _ => Net::Plain(net::SegResNet::load(
                p,
                32,
                &[1, 2, 2, 4],
                &[1, 1, 1],
                105,
                NormKind::Group(8),
                dev,
            )?),
        })
    }

    pub fn classes(&self) -> usize {
        match self {
            Net::Plain(n) => n.classes(),
            Net::Ds(n) => n.classes(),
        }
    }

    /// Logits for one window, flattened `[classes, p0, p1, p2]`.
    pub fn window(&self, patch: &[f32], p: [usize; 3], dev: &B::Device) -> Vec<f32> {
        let x = ops::from_slice::<B, 5>(patch, [1, 1, p[0], p[1], p[2]], dev);
        let y = match self {
            Net::Plain(n) => n.forward(x),
            Net::Ds(n) => n.forward(x),
        };
        ops::to_vec(y)
    }
}

struct Hooks<'a, B: burn::tensor::backend::Backend> {
    net: &'a Net<B>,
    dev: B::Device,
    patch: [usize; 3],
    progress: &'a Progress,
    label: &'static str,
}

impl<B: burn::tensor::backend::Backend> InferHooks for Hooks<'_, B>
where
    B::Device: Sync,
{
    fn forward(&self, patch: &[f32]) -> Result<Vec<f32>> {
        Ok(self.net.window(patch, self.patch, &self.dev))
    }
    fn tile_done(&self, done: usize, total: usize) -> bool {
        self.progress.report(
            done as f32 / total as f32,
            &format!("Segmenting ({}): window {done}/{total}", self.label),
        );
        !self.progress.cancelled()
    }
}

/// The windows over a prepared volume, on one backend.
fn windows<B: burn::tensor::backend::Backend>(
    model: SegResModel,
    params: &Params,
    dev: &B::Device,
    data: &[f32],
    dims: [usize; 3],
    progress: &Progress,
) -> Result<Vec<u8>>
where
    B::Device: Sync,
{
    let net = Net::<B>::load(model, params, dev)?;
    let (patch, overlap, fill) = match model {
        SegResModel::CtFm => ([96, 160, 160], 0.625, Fill::Zero),
        _ => ([96, 96, 96], 0.25, Fill::Replicate),
    };
    let plan = infer::monai_plan(dims, patch, overlap, fill);
    let hooks = Hooks {
        net: &net,
        dev: dev.clone(),
        patch,
        progress,
        label: model.label(),
    };
    infer::predict_plan(data, dims, net.classes(), &plan, infer::ACC_BUDGET, &hooks)
}

/// Run a SegResNet-family model on a CT volume. Blocking; observe and
/// cancel through `progress`. `root` is the model folder.
pub fn run(
    volume: &Volume,
    model: SegResModel,
    device: DevicePref,
    root: &Path,
    progress: &Progress,
) -> Result<AutosegResult> {
    let t0 = std::time::Instant::now();
    progress.set_phase(0.0, 0.1);
    let params = load(model, root, progress)?;
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- the published preprocessing ---------------------------------------
    progress.set_phase(0.1, 0.05);
    progress.report(0.0, "Preparing the volume");
    let order = match model {
        SegResModel::CtFm => AxisOrder::Spl,
        _ => AxisOrder::Ras,
    };
    let grid = pre::Oriented::new(volume, order);
    let raw = grid.read(volume);
    let (mut data, dims, crop_box) = match model.spacing() {
        Some(s) => {
            let (mut d, dims) = pre::resample_linear(&raw, grid.dims, grid.spacing, [s; 3]);
            pre::normalize_nonzero(&mut d);
            pre::scale_minmax(&mut d, -1.0, 1.0);
            (d, dims, None)
        }
        None => {
            let mut d = raw;
            pre::scale_range_clip(&mut d, -1024.0, 2048.0, 0.0, 1.0);
            let (lo, hi) = pre::foreground_box(&d, grid.dims);
            let dims = std::array::from_fn(|a| hi[a] - lo[a]);
            (pre::crop(&d, grid.dims, lo, hi), dims, Some((lo, hi)))
        }
    };
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- the network ----------------------------------------------------------
    progress.set_phase(0.15, 0.8);
    progress.set("Choosing the compute device");
    let gpu = device.resolve()?;
    let (mut labels, device_desc) = match gpu {
        #[cfg(feature = "gpu")]
        Some(ctx) => {
            let desc = ctx.describe();
            progress.set_device(&desc);
            let l = crate::nn::device::guarded(|| {
                windows::<crate::medsam2::engine::Gpu>(
                    model,
                    &params,
                    ctx.device(),
                    &data,
                    dims,
                    progress,
                )
            })?;
            (l, desc)
        }
        #[cfg(not(feature = "gpu"))]
        Some(ctx) => ctx.unreachable(),
        None => {
            let desc = crate::nn::device::describe_cpu();
            progress.set_device(&desc);
            let dev = Default::default();
            let l = windows::<crate::medsam2::engine::Cpu>(
                model, &params, &dev, &data, dims, progress,
            )?;
            (l, desc)
        }
    };
    data.clear();

    // ---- back onto the scan ----------------------------------------------------
    progress.set_phase(0.95, 0.05);
    progress.report(0.0, "Mapping labels back to the scan grid");
    let model_grid = match (model.spacing(), crop_box) {
        (Some(s), _) => pre::resample_nearest(&labels, dims, [s; 3], grid.dims, grid.spacing),
        (None, Some((lo, hi))) => {
            pre::keep_largest_per_label(&mut labels, dims);
            pre::uncrop(&labels, grid.dims, lo, hi)
        }
        (None, None) => labels,
    };
    let labels = grid.write_back(&model_grid, volume);
    let classes = model.classes();
    let organs = organ_hits(&labels, volume.spacing, classes);
    progress.report(1.0, "Segmentation finished");
    Ok(AutosegResult {
        labels,
        dims: volume.dims,
        organs,
        model: model.key().to_string(),
        model_label: model.label().to_string(),
        classes,
        device: device_desc,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
        volume_dims: volume.dims,
        notes: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_models_name_their_files_and_tables() {
        assert_eq!(CLASSES_TS_V1[0], "spleen");
        assert_eq!(CLASSES_TS_V1[103], "urinary_bladder");
        assert_eq!(SegResModel::CtFm.classes().len(), 117);
        for m in SegResModel::ALL {
            assert!(m.file().url.starts_with("https://huggingface.co/"));
            assert!(m.file().url.ends_with(m.file().name));
        }
        let root = std::env::temp_dir().join("rds_segresnet_none");
        assert_eq!(download_needed(SegResModel::CtFm, &root), 348_968_896);
    }
}
