//! Interactive 3-D segmentation - a pure-Rust port of
//! [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) (Isensee,
//! Rokuss, Krämer et al., *nnInteractive: Redefining 3D Promptable
//! Segmentation*, arXiv:2503.08373, 2025).
//!
//! One network, an nnU-Net `ResidualEncoderUNet` (the L preset, 32 to 320
//! features, 192-cubed patch) whose input is the image plus seven prompt
//! channels: the current segmentation, and a positive and a negative channel
//! for boxes and lassos, points, and scribbles. The network runs on the
//! nnU-Net engine of [`crate::autoseg`] (its residual encoder reads any
//! number of input channels); everything around it - prompt drawing, the
//! zoom-out loop, the refinement passes - is the [`session`].
//!
//! The weights (`MIC-DKFZ/nnInteractive`, folder `nnInteractive_v1.0`) are
//! licensed CC BY-NC-SA 4.0 - non-commercial use only; the code upstream is
//! Apache-2.0. They are downloaded on request and never redistributed.
//!
//! The volume is used in its own voxels, reordered to the `[S, A, R]` axes
//! nnU-Net's `NibabelIOWithReorient` gave the training data, and never
//! resampled.

pub mod ops;
#[cfg(test)]
mod parity;
pub mod session;

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub use session::{Kind, Outcome, Predictor, Session, Settings, N_PROMPT_CHANNELS};

use crate::autoseg::config::ModelConfig;
use crate::autoseg::cpu::Act;
use crate::autoseg::net::UNet;
use crate::nn::cache::{self, ConvertSpec, RemoteFile, WTensor};
use crate::nn::device::DevicePref;
use crate::progress::ProgressSink;
use crate::segresnet::pre::Oriented;
use crate::volume::{AxisOrder, Volume};

/// The engine folder under the model root.
pub const DIR: &str = "nninteractive";
/// The model's folder in the repository, and under [`DIR`].
pub const MODEL: &str = "nnInteractive_v1.0";
/// The configuration the official checkpoint was trained with.
const CONFIGURATION: &str = "3d_fullres_ps192_bs24";

macro_rules! hf {
    ($path:literal) => {
        concat!(
            "https://huggingface.co/MIC-DKFZ/nnInteractive/resolve/main/nnInteractive_v1.0/",
            $path
        )
    };
}

pub const PLANS: RemoteFile = RemoteFile {
    name: "plans.json",
    url: hf!("plans.json"),
    bytes: 6_287,
};
pub const SESSION_INFO: RemoteFile = RemoteFile {
    name: "inference_session_class.json",
    url: hf!("inference_session_class.json"),
    bytes: 121,
};
pub const CHECKPOINT: RemoteFile = RemoteFile {
    name: "checkpoint_final.pth",
    url: hf!("fold_0/checkpoint_final.pth"),
    bytes: 411_387_150,
};
/// The converted weights.
pub const CACHE: &str = "fold_0.safetensors";

pub fn model_dir(root: &Path) -> PathBuf {
    root.join(DIR).join(MODEL)
}

/// Bytes still to download before the model runs offline.
pub fn download_needed(root: &Path) -> u64 {
    let dir = model_dir(root);
    let mut need = cache::download_needed([&PLANS, &SESSION_INFO], &dir);
    if !dir.join(CACHE).is_file() && !CHECKPOINT.is_cached(&dir) {
        need += CHECKPOINT.bytes;
    }
    need
}

/// The files a ready model folder holds.
pub fn ready_files() -> [&'static str; 3] {
    [PLANS.name, SESSION_INFO.name, CACHE]
}

/// Settings from `inference_session_class.json` (the v1.0 checkpoint's
/// legacy metadata): point radius, decay, scribble thickness. A bare string
/// there means a 0.9 decay, as upstream reads it.
pub fn read_settings(text: &str, patch: [usize; 3]) -> Result<(Settings, usize)> {
    let v: serde_json::Value = serde_json::from_str(text).context("parse the session metadata")?;
    let mut s = Settings::v1(patch);
    let mut scribble = 2usize;
    match &v {
        serde_json::Value::String(_) => s.decay = 0.9,
        serde_json::Value::Object(o) => {
            if let Some(r) = o.get("point_radius").and_then(|x| x.as_f64()) {
                s.point_radius = r.round().max(1.0) as usize;
            }
            if let Some(d) = o.get("interaction_decay").and_then(|x| x.as_f64()) {
                s.decay = d;
            }
            match o.get("preferred_scribble_thickness") {
                Some(serde_json::Value::Array(a)) => {
                    if let Some(t) = a.first().and_then(|x| x.as_f64()) {
                        scribble = t.round().max(1.0) as usize;
                    }
                }
                Some(x) => {
                    if let Some(t) = x.as_f64() {
                        scribble = t.round().max(1.0) as usize;
                    }
                }
                None => {}
            }
            if let Some(m) = o.get("pad_mode_image").and_then(|x| x.as_str()) {
                if m != "constant" {
                    bail!("the checkpoint pads with {m:?}; only constant padding is implemented");
                }
            }
        }
        _ => bail!("unexpected session metadata"),
    }
    Ok((s, scribble))
}

enum Net {
    Cpu(Box<UNet>),
    #[cfg(feature = "gpu")]
    Gpu(Box<crate::autoseg::gpu::GpuNet>),
}

/// The loaded network.
pub struct Model {
    net: Net,
    pub settings: Settings,
    /// Stroke width of a scribble, in voxels.
    pub scribble_thickness: usize,
    /// Where the network runs, as the interface shows it.
    pub device: String,
}

/// Download and convert the model if that has not happened yet, and hand
/// back the converted tensors.
pub fn ensure(root: &Path, sink: &dyn ProgressSink) -> Result<HashMap<String, WTensor>> {
    let dir = model_dir(root);
    PLANS.ensure(&dir, sink)?;
    SESSION_INFO.ensure(&dir, sink)?;
    let spec = ConvertSpec {
        top_key: "network_weights",
        keep: &|name, _| !name.starts_with("decoder.encoder.") && !name.contains(".all_modules."),
        rename: &|name| name.to_string(),
        label: "nnInteractive",
    };
    let tensors = cache::ensure_converted(
        &dir.join(CACHE),
        &spec,
        || CHECKPOINT.ensure(&dir, sink),
        sink,
    )?;
    // The converted cache is all a later run reads.
    let _ = std::fs::remove_file(CHECKPOINT.path_in(&dir));
    Ok(tensors)
}

impl Model {
    /// Load the network, downloading and converting it on first use.
    pub fn load(root: &Path, device: DevicePref, sink: &dyn ProgressSink) -> Result<Model> {
        let dir = model_dir(root);
        let tensors = ensure(root, sink)?;
        let plans = PLANS.path_in(&dir);
        let info = SESSION_INFO.path_in(&dir);
        let text =
            std::fs::read_to_string(&plans).with_context(|| format!("read {}", plans.display()))?;
        let cfg = ModelConfig::from_plans_json_cfg(&text, CONFIGURATION)
            .context("nnInteractive plans.json")?;
        let (settings, scribble_thickness) =
            read_settings(&std::fs::read_to_string(&info)?, cfg.patch_size)?;
        let unet = UNet::build(cfg, &tensors).context("assemble the nnInteractive network")?;
        drop(tensors);
        if unet.in_channels != 1 + N_PROMPT_CHANNELS || unet.num_classes() != 2 {
            bail!(
                "unexpected nnInteractive network: {} input channels, {} outputs",
                unet.in_channels,
                unet.num_classes()
            );
        }
        Self::with_net(unet, settings, scribble_thickness, device)
    }

    /// A model around an assembled network (tests build small ones).
    pub fn with_net(
        unet: UNet,
        settings: Settings,
        scribble_thickness: usize,
        device: DevicePref,
    ) -> Result<Model> {
        let gpu = device.resolve()?;
        let (net, device) = match gpu {
            None => (Net::Cpu(Box::new(unet)), crate::nn::device::describe_cpu()),
            #[cfg(feature = "gpu")]
            Some(ctx) => {
                let g = crate::autoseg::gpu::GpuNet::new(&ctx, &unet)?;
                (Net::Gpu(Box::new(g)), ctx.describe())
            }
            #[cfg(not(feature = "gpu"))]
            Some(ctx) => ctx.unreachable(),
        };
        Ok(Model {
            net,
            settings,
            scribble_thickness,
            device,
        })
    }
}

impl Predictor for Model {
    fn predict(&self, input: &[f32], p: [usize; 3]) -> Result<Vec<u8>> {
        let n = p[0] * p[1] * p[2];
        let logits = match &self.net {
            Net::Cpu(unet) => {
                let x = Act {
                    c: input.len() / n,
                    d: p[0],
                    h: p[1],
                    w: p[2],
                    data: input.to_vec(),
                };
                unet.forward_cpu(&x).data
            }
            #[cfg(feature = "gpu")]
            Net::Gpu(g) => crate::nn::device::guarded(|| g.forward(input, p))?,
        };
        // argmax over the two classes; a tie is background, as torch's
        // argmax takes the first maximum
        Ok((0..n)
            .map(|v| u8::from(logits[n + v] > logits[v]))
            .collect())
    }
}

/// A session on a volume, in the network's axes.
pub struct VolumeSession {
    pub session: Session,
    pub axes: Oriented,
    pub vol_dims: [usize; 3],
}

impl VolumeSession {
    pub fn new(volume: &Volume, settings: Settings) -> Result<VolumeSession> {
        let axes = Oriented::new(volume, AxisOrder::Sar);
        let image = axes.read(volume);
        let session = Session::new(image, axes.dims, settings)?;
        Ok(VolumeSession {
            session,
            axes,
            vol_dims: volume.dims,
        })
    }

    /// A voxel position of the volume (`[x, y, z]`, fractional) in the
    /// session's axes.
    pub fn to_session(&self, v: [f64; 3]) -> [f64; 3] {
        self.axes.point_to_model(v)
    }

    /// The segmentation on the volume's own grid.
    pub fn mask_on_volume(&self) -> Vec<u8> {
        let [d0, d1, d2] = self.axes.dims;
        let mut out = vec![0u8; d0 * d1 * d2];
        for a in 0..d0 {
            for b in 0..d1 {
                for c in 0..d2 {
                    out[self.axes.volume_index(self.vol_dims, [a, b, c])] =
                        self.session.mask[(a * d1 + b) * d2 + c];
                }
            }
        }
        out
    }

    /// A label map on the volume's grid, in the session's axes.
    pub fn from_volume(&self, labels: &[u8]) -> Vec<u8> {
        self.axes.read_labels(labels, self.vol_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published network against PyTorch on one patch: set
    /// `RDS_NNINTERACTIVE_ROOT` to a model folder holding
    /// `nninteractive/nnInteractive_v1.0` (converted or with its
    /// checkpoint) and `RDS_NNINTERACTIVE_PROBE` to a safetensors file with
    /// `input` `[8, d, h, w]` and the network's `logits` `[2, d, h, w]` as
    /// PyTorch computed them. The whole 192-cubed session needs more memory
    /// than a CPU test should; one patch checks the weights and the network.
    #[test]
    #[ignore]
    fn published_weights_match_pytorch() {
        struct Quiet;
        impl ProgressSink for Quiet {
            fn report(&self, _: f32, _: &str) {}
            fn cancelled(&self) -> bool {
                false
            }
        }
        let root = PathBuf::from(std::env::var("RDS_NNINTERACTIVE_ROOT").expect("model root"));
        let probe = PathBuf::from(std::env::var("RDS_NNINTERACTIVE_PROBE").expect("probe"));
        let tensors = ensure(&root, &Quiet).unwrap();
        let text = std::fs::read_to_string(PLANS.path_in(&model_dir(&root))).unwrap();
        let cfg = ModelConfig::from_plans_json_cfg(&text, CONFIGURATION).unwrap();
        let unet = UNet::build(cfg, &tensors).unwrap();
        drop(tensors);
        let io = cache::load_safetensors(&probe).unwrap();
        let input = &io["input"];
        let [c, d, h, w]: [usize; 4] = input.shape.clone().try_into().unwrap();
        let y = unet.forward_cpu(&Act {
            c,
            d,
            h,
            w,
            data: input.data.clone(),
        });
        let want = &io["logits"].data;
        assert_eq!(y.data.len(), want.len());
        let range = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let worst = y
            .data
            .iter()
            .zip(want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let n = d * h * w;
        let fg = |l: &[f32], v: usize| l[n + v] > l[v];
        let differ = (0..n).filter(|&v| fg(&y.data, v) != fg(want, v)).count();
        eprintln!("worst {worst} of {range}; argmax differs at {differ} of {n}");
        assert!(worst <= 1e-3 * range, "worst {worst} of {range}");
        assert!(differ * 10_000 <= n, "{differ} voxels change side");
    }

    #[test]
    fn the_files_live_in_one_folder_of_the_repository() {
        for f in [PLANS, SESSION_INFO, CHECKPOINT] {
            assert!(f.url.contains("/nnInteractive_v1.0/"), "{}", f.url);
            assert!(f.url.ends_with(f.name), "{}", f.url);
        }
        let root = std::env::temp_dir().join("rds_nni_none");
        assert_eq!(download_needed(&root), 6_287 + 121 + 411_387_150);
    }

    #[test]
    fn the_v1_metadata_reads_as_upstream() {
        let (s, t) = read_settings(
            r#"{"inference_class": "nnInteractiveInferenceSession", "point_radius": 4, "preferred_scribble_thickness": 2}"#,
            [192, 192, 192],
        )
        .unwrap();
        assert_eq!(s.point_radius, 4);
        assert_eq!(s.decay, 0.98);
        assert_eq!(t, 2);
        let (s, _) = read_settings(r#""nnInteractiveInferenceSession""#, [192; 3]).unwrap();
        assert_eq!(s.decay, 0.9);
    }
}
