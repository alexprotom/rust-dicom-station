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
pub mod pickle;
pub mod preprocess;
pub mod weights;

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

use crate::volume::Volume;
use weights::{ModelSpec, ProgressSink};

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
    /// nnU-Net tile step fraction: 0.8 for the "total" task (TotalSegmentator
    /// uses 0.5 only for other tasks).
    fn step_frac(&self) -> f64 {
        0.8
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

/// Device preference for inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DevicePref {
    /// GPU when available, CPU otherwise.
    Auto,
    Cpu,
    Gpu,
}

/// Progress handle shared with the UI thread: message + fraction + cancel.
#[derive(Default)]
pub struct AutosegProgress {
    msg: Mutex<String>,
    /// f32 bits of the overall progress fraction (0..=1).
    frac: AtomicU32,
    cancel: AtomicBool,
    /// Current phase window mapped onto the overall fraction.
    phase_base: AtomicU32,
    phase_span: AtomicU32,
}

impl AutosegProgress {
    pub fn set(&self, msg: impl Into<String>) {
        *self.msg.lock().unwrap_or_else(|e| e.into_inner()) = msg.into();
    }
    pub fn get(&self) -> String {
        self.msg.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub fn frac(&self) -> f32 {
        f32::from_bits(self.frac.load(Ordering::Relaxed))
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
    /// Map subsequent `report(frac in 0..1)` calls onto
    /// `[base, base+span]` of the overall progress bar.
    fn set_phase(&self, base: f32, span: f32) {
        self.phase_base.store(base.to_bits(), Ordering::Relaxed);
        self.phase_span.store(span.to_bits(), Ordering::Relaxed);
        self.frac.store(base.to_bits(), Ordering::Relaxed);
    }
}

impl ProgressSink for AutosegProgress {
    fn report(&self, frac: f32, msg: &str) {
        let base = f32::from_bits(self.phase_base.load(Ordering::Relaxed));
        let span = f32::from_bits(self.phase_span.load(Ordering::Relaxed));
        self.frac.store(
            (base + span * frac.clamp(0.0, 1.0)).to_bits(),
            Ordering::Relaxed,
        );
        self.set(msg);
    }
    fn cancelled(&self) -> bool {
        self.cancelled()
    }
}

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

/// Default model cache directory: `autoseg_models/` next to the executable
/// (falls back to the current directory).
pub fn default_models_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("autoseg_models")
}

/// True when every model of the variant is already downloaded + converted.
pub fn variant_cached(variant: Variant, parts: [bool; 5], models_dir: &Path) -> bool {
    variant
        .specs(parts)
        .iter()
        .all(|s| weights::is_cached(s, models_dir))
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

enum Engine {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(gpu::GpuContext),
}

impl Engine {
    fn describe(&self) -> String {
        match self {
            Engine::Cpu => format!("CPU ({} threads)", rayon::current_num_threads()),
            #[cfg(feature = "gpu")]
            Engine::Gpu(ctx) => format!("GPU ({})", ctx.adapter_name()),
        }
    }
}

struct Hooks<'a> {
    forward: ForwardFn<'a>,
    progress: &'a AutosegProgress,
    /// (model index, model count) for progress text.
    model: (usize, usize),
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
    progress: &AutosegProgress,
) -> Result<AutosegResult> {
    let t_start = std::time::Instant::now();
    let specs = variant.specs(parts);
    if specs.is_empty() {
        bail!("no sub-models selected");
    }
    let n_models = specs.len();

    // Progress budget: 15% download/convert/load, 5% preprocess,
    // 75% inference, 5% postprocess.
    let dl_span = 0.15 / n_models as f32;

    // ---- load models (download + convert on first use) -------------------
    let mut models = Vec::with_capacity(n_models);
    for (i, spec) in specs.iter().enumerate() {
        progress.set_phase(i as f32 * dl_span, dl_span);
        let m = weights::ensure_model(spec, models_dir, progress)?;
        if progress.cancelled() {
            bail!("cancelled");
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
    let target = spacing[0];

    // ---- engine ----------------------------------------------------------
    let engine = resolve_engine(device, progress)?;
    let device_desc = engine.describe();

    // ---- preprocess ------------------------------------------------------
    progress.set_phase(0.15, 0.05);
    progress.report(0.0, &format!("Resampling volume to {target} mm…"));
    let map = preprocess::SarMap::new(volume, target);
    let vol_model = preprocess::resample_to_model(volume, &map);
    if progress.cancelled() {
        bail!("cancelled");
    }

    // ---- inference per model, merged into global labels ------------------
    let mut global = vec![0u8; vol_model.len()];
    let infer_span = 0.75 / n_models as f32;
    for (mi, model) in models.iter().enumerate() {
        progress.set_phase(0.2 + mi as f32 * infer_span, infer_span);
        let unet = net::UNet::build(model.config.clone(), &model.tensors)
            .with_context(|| format!("assemble network ({})", model.spec.label))?;
        let classes = unet.num_classes();
        let forward: ForwardFn = match &engine {
            Engine::Cpu => {
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
                    Ok(unet_ref.forward_cpu(x).data)
                })
            }
            #[cfg(feature = "gpu")]
            Engine::Gpu(ctx) => {
                let gnet = gpu::GpuNet::new(ctx, &unet)?;
                let p = unet.cfg.patch_size;
                Box::new(move |patch: &[f32]| gnet.forward(patch, p))
            }
        };
        let hooks = Hooks {
            forward,
            progress,
            model: (mi, n_models),
            label: variant.label(),
        };
        let local = infer::predict(
            &vol_model,
            map.model_dims,
            classes,
            &unet.cfg,
            variant.step_frac(),
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

    // ---- back-map to the CT grid + statistics ----------------------------
    progress.set_phase(0.95, 0.05);
    progress.report(0.0, "Mapping labels back to the CT grid…");
    let labels = preprocess::labels_to_volume_grid(&global, &map, volume);
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

#[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
fn resolve_engine(device: DevicePref, progress: &AutosegProgress) -> Result<Engine> {
    match device {
        DevicePref::Cpu => Ok(Engine::Cpu),
        DevicePref::Gpu => {
            #[cfg(feature = "gpu")]
            {
                progress.set("Initializing GPU…");
                match gpu::GpuContext::try_new() {
                    Ok(ctx) => Ok(Engine::Gpu(ctx)),
                    Err(e) => bail!("GPU requested but not available: {e}"),
                }
            }
            #[cfg(not(feature = "gpu"))]
            bail!("this build has no GPU support (compiled without the 'gpu' feature)")
        }
        DevicePref::Auto => {
            #[cfg(feature = "gpu")]
            {
                progress.set("Looking for a GPU…");
                if let Ok(ctx) = gpu::GpuContext::try_new() {
                    return Ok(Engine::Gpu(ctx));
                }
            }
            Ok(Engine::Cpu)
        }
    }
}
