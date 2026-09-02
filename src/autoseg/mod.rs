//! Automatic CT multi-organ segmentation — a pure-Rust re-implementation of
//! [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) v2
//! inference (Wasserthal et al., Radiology AI 2023, doi 10.1148/ryai.230024).
//!
//! The nnU-Net v2 models of the openly licensed (Apache-2.0) "total" task
//! are downloaded from the official GitHub release on first use, converted
//! natively (no Python) and cached. Inference runs either on the CPU
//! (rayon + SIMD GEMM) or on any GPU through wgpu (Vulkan / DX12 / Metal —
//! no CUDA toolkit required) when the `gpu` feature is enabled.
//!
//! Pipeline (mirroring TotalSegmentator exactly): reorient to canonical
//! [S,A,R] axes → resample to the model's isotropic spacing (trilinear) →
//! clip to [p0.5, p99.5] + z-score (per-model constants from `plans.json`) →
//! sliding-window inference, Gaussian-weighted, tile step 0.8, no mirroring
//! TTA → argmax → merge sub-model labels (1.5 mm variant) → nearest-neighbor
//! map back to the original CT grid.

pub mod classes;
pub mod config;
pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod infer;
pub mod net;
pub mod preprocess;
pub mod weights;

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::volume::Volume;
use weights::ModelSpec;

/// Which TotalSegmentator model set to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Single 3 mm model, all 117 classes (`--fast`). ~135 MB download.
    Fast3mm,
    /// Five 1.5 mm sub-models (organs / vertebrae / cardiac / muscles /
    /// ribs), best quality. ~1.2 GB download.
    HighRes15mm,
    /// Single 6 mm model (`--fastest`) — quick preview quality.
    Preview6mm,
}

impl Variant {
    pub fn label(&self) -> &'static str {
        match self {
            Variant::Fast3mm => "3 mm (fast)",
            Variant::HighRes15mm => "1.5 mm (high quality)",
            Variant::Preview6mm => "6 mm (preview)",
        }
    }
    fn specs(&self, parts: [bool; 5]) -> Vec<ModelSpec> {
        match self {
            Variant::Fast3mm => vec![weights::SPEC_3MM],
            Variant::Preview6mm => vec![weights::SPEC_6MM],
            Variant::HighRes15mm => weights::SPECS_15MM
                .iter()
                .zip(parts.iter())
                .filter(|(_, on)| **on)
                .map(|(s, _)| *s)
                .collect(),
        }
    }
}

/// nnU-Net tile step fraction: 0.8 for the "total" task (TotalSegmentator
/// uses 0.5 only for other tasks).
const STEP_FRAC: f64 = 0.8;

pub use crate::nn::device::DevicePref;

/// One detected organ in the result.
#[derive(Clone, Debug)]
pub struct OrganHit {
    /// Global TotalSegmentator class id (1..=117).
    pub label: u8,
    pub name: &'static str,
    /// Voxel count on the original CT grid.
    pub voxels: u64,
    /// Volume in cm³ on the original CT grid.
    pub cm3: f64,
    pub color: [u8; 3],
}

/// Output of a segmentation run.
pub struct AutosegResult {
    /// Global class labels per voxel, `Volume::data` index order.
    pub labels: Vec<u8>,
    pub dims: [usize; 3],
    /// Classes present, sorted by voxel count descending.
    pub organs: Vec<OrganHit>,
    pub variant: Variant,
    /// Human-readable device description ("CPU (16 threads)", "GPU (wgpu)").
    pub device: String,
    pub elapsed_secs: f64,
    /// Identity of the volume this was computed on.
    pub frame_of_reference_uid: String,
    pub volume_dims: [usize; 3],
}

/// Total download size (bytes) still needed for the variant.
pub fn download_needed(variant: Variant, parts: [bool; 5], models_dir: &Path) -> u64 {
    variant
        .specs(parts)
        .iter()
        .filter(|s| !weights::is_cached(s, models_dir))
        .map(|s| s.zip_bytes)
        .sum()
}

// ---- engine dispatch ----------------------------------------------------

/// Boxed patch-forward closure: normalized patch → logits.
type ForwardFn<'a> = Box<dyn Fn(&[f32]) -> Result<Vec<f32>> + Sync + 'a>;

struct Hooks<'a> {
    forward: ForwardFn<'a>,
    progress: &'a Progress,
    /// (model index, model count) for progress text.
    model: (usize, usize),
    /// What the run is called in the progress line.
    label: &'static str,
}

impl infer::InferHooks for Hooks<'_> {
    fn forward(&self, patch: &[f32]) -> Result<Vec<f32>> {
        (self.forward)(patch)
    }
    fn tile_done(&self, done: usize, total: usize) -> bool {
        let (mi, mn) = self.model;
        self.progress.report(
            done as f32 / total as f32,
            &format!(
                "Segmenting ({}), model {}/{}: tile {}/{}",
                self.label,
                mi + 1,
                mn,
                done,
                total
            ),
        );
        !self.progress.cancelled()
    }
}

/// Run one or more nnU-Net models over a volume and return their merged
/// labels **on the volume's own grid**, plus the description of the device
/// the work ran on.
///
/// This is the whole engine minus the question being asked: the 117-class
/// "total" task ([`run`]) and the two-class body-outline task
/// ([`crate::bodymask`]) differ in which checkpoints they load and what
/// they do with the answer, not in how a checkpoint is fetched, converted,
/// resampled onto, tiled over or mapped back from.
///
/// `label` names the run in progress messages, and `window` is the slice of
/// the overall progress bar this run owns — `(0.0, 1.0)` for the whole of it.
/// [`Progress::set_phase`] is absolute rather than nested, so a caller that
/// has its own work to do afterwards has to say so here; otherwise the bar
/// reaches 100 % and then jumps backwards.
pub fn run_specs(
    volume: &Volume,
    specs: &[ModelSpec],
    label: &'static str,
    device: DevicePref,
    models_dir: &Path,
    window: (f32, f32),
    progress: &Progress,
) -> Result<(Vec<u8>, String)> {
    if specs.is_empty() {
        bail!("no sub-models selected");
    }
    let n_models = specs.len();
    let (base, span) = window;
    // Every phase below is expressed in this run's own 0..1 and mapped onto
    // the window the caller gave.
    let phase = |p: &Progress, at: f32, len: f32| p.set_phase(base + span * at, span * len);

    // Progress budget: 15% download/convert/load, 5% preprocess,
    // 75% inference, 5% postprocess.
    let dl_span = 0.15 / n_models as f32;

    // ---- load models (download + convert on first use) -------------------
    let mut models = Vec::with_capacity(n_models);
    for (i, spec) in specs.iter().enumerate() {
        phase(progress, i as f32 * dl_span, dl_span);
        let m = weights::ensure_model(spec, models_dir, progress)?;
        if progress.cancelled() {
            bail!(CANCELLED);
        }
        models.push(m);
    }
    // All sub-models of a variant must share the target spacing.
    let spacing = models[0].config.spacing;
    for m in &models {
        if m.config.spacing != spacing {
            bail!("sub-models disagree on spacing");
        }
    }

    // ---- engine ----------------------------------------------------------
    progress.set("Choosing the compute device");
    let gpu = device.resolve()?;
    let device_desc = gpu
        .as_ref()
        .map(|ctx| ctx.describe())
        .unwrap_or_else(crate::nn::device::describe_cpu);
    progress.set_device(&device_desc);

    // ---- preprocess ------------------------------------------------------
    phase(progress, 0.15, 0.05);
    progress.report(
        0.0,
        &format!(
            "Resampling volume to {} mm",
            if spacing[0] == spacing[1] && spacing[1] == spacing[2] {
                format!("{}", spacing[0])
            } else {
                format!("{} × {} × {}", spacing[0], spacing[1], spacing[2])
            }
        ),
    );
    let map = preprocess::SarMap::new(volume, spacing);
    let vol_model = preprocess::resample_to_model(volume, &map);
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- inference per model, merged into global labels ------------------
    let mut global = vec![0u8; vol_model.len()];
    let infer_span = 0.75 / n_models as f32;
    for (mi, model) in models.iter().enumerate() {
        phase(progress, 0.2 + mi as f32 * infer_span, infer_span);
        // A z-score model normalizes against this image, so its constants
        // are only knowable now, with the resampled volume in hand.
        let mut cfg = model.config.clone();
        cfg.apply_image_norm(&vol_model);
        let unet = net::UNet::build(cfg, &model.tensors)
            .with_context(|| format!("assemble network ({})", model.spec.label))?;
        let classes = unet.num_classes();
        let forward: ForwardFn = match &gpu {
            None => {
                let unet_ref = &unet;
                let p = unet.cfg.patch_size;
                Box::new(move |patch: &[f32]| {
                    let x = cpu::Act {
                        c: 1,
                        d: p[0],
                        h: p[1],
                        w: p[2],
                        data: patch.to_vec(),
                    };
                    Ok(unet_ref.forward_cpu(&x).data)
                })
            }
            #[cfg(feature = "gpu")]
            Some(ctx) => {
                let gnet = gpu::GpuNet::new(ctx, &unet)?;
                let p = unet.cfg.patch_size;
                Box::new(move |patch: &[f32]| gnet.forward(patch, p))
            }
            #[cfg(not(feature = "gpu"))]
            Some(ctx) => ctx.unreachable(),
        };
        let hooks = Hooks {
            forward,
            progress,
            model: (mi, n_models),
            label,
        };
        let local = infer::predict(
            &vol_model,
            map.model_dims,
            classes,
            &unet.cfg,
            STEP_FRAC,
            &hooks,
        )
        .with_context(|| format!("inference ({})", model.spec.label))?;
        // merge: local foreground labels → global ids; later models overwrite
        let off = model.spec.global_offset;
        for (g, l) in global.iter_mut().zip(local.iter()) {
            if *l != 0 {
                *g = off + *l;
            }
        }
    }

    // ---- back-map to the CT grid ----------------------------------------
    phase(progress, 0.95, 0.05);
    progress.report(0.0, "Mapping labels back to the CT grid");
    let labels = preprocess::labels_to_volume_grid(&global, &map, volume);
    Ok((labels, device_desc))
}

/// Run auto-segmentation on a CT volume. Blocking — call from a worker
/// thread; observe/cancel through `progress`.
///
/// `parts` selects sub-models for [`Variant::HighRes15mm`]
/// (organs, vertebrae, cardiac, muscles, ribs) and is ignored otherwise.
pub fn run(
    volume: &Volume,
    variant: Variant,
    device: DevicePref,
    parts: [bool; 5],
    models_dir: &Path,
    progress: &Progress,
) -> Result<AutosegResult> {
    let t_start = std::time::Instant::now();
    let specs = variant.specs(parts);
    let (labels, device_desc) = run_specs(
        volume,
        &specs,
        variant.label(),
        device,
        models_dir,
        (0.0, 1.0),
        progress,
    )?;

    // ---- statistics ------------------------------------------------------
    let mut counts = [0u64; 256];
    for l in &labels {
        counts[*l as usize] += 1;
    }
    let voxel_cm3 = volume.spacing[0] * volume.spacing[1] * volume.spacing[2] / 1000.0;
    let mut organs: Vec<OrganHit> = (1u16..=117)
        .filter(|l| counts[*l as usize] > 0)
        .map(|l| {
            let l = l as u8;
            OrganHit {
                label: l,
                name: classes::class_name(l),
                voxels: counts[l as usize],
                cm3: counts[l as usize] as f64 * voxel_cm3,
                color: classes::class_color(l),
            }
        })
        .collect();
    organs.sort_by_key(|o| std::cmp::Reverse(o.voxels));

    progress.report(1.0, "Auto-segmentation finished");
    Ok(AutosegResult {
        labels,
        dims: volume.dims,
        organs,
        variant,
        device: device_desc,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
        volume_dims: volume.dims,
    })
}
