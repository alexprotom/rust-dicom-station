//! Synthetic CT from MR or CBCT: TotalSynth (Boussot et al., arXiv
//! 2609.13838, 2026), published as the ImpactSynth apps - a 2.5-D UNet++
//! with a ResNet-34 encoder that turns five neighbouring axial slices into
//! the CT of the middle one, trained on SynthRAD2023, SynthRAD2025 and four
//! prostate cohorts (brain, head and neck, thorax, abdomen, pelvis).
//! Weights CC BY-NC 4.0 (their training data's terms); code Apache-2.0.
//!
//! Re-implemented natively: the network ([`layout`], [`net`] on the GPU,
//! [`cpu`] on the CPU) and the prediction recipe of the apps'
//! `Prediction.yml` as KonfAI runs it, step for step and to the rounding
//! ([`pre`]):
//!
//! 1. the scan in its LPS axis order (KonfAI reads an image as stored; an
//!    axial DICOM series already is), cast through float16 and clipped to
//!    [-1024, 5000];
//! 2. resampled (linear) to 1 x 1 x 3 mm over the same field of view;
//! 3. clipped at its 99.5th percentile and mapped to [-1, 1];
//! 4. every slice with two neighbours on each side (reflected at the ends),
//!    in 512 x 512 patches padded with -1, through the network;
//! 5. the output in half precision, mapped back to [-1024, 3071] HU, int16,
//!    air outside the body (dilated by five voxels);
//! 6. the mean over the chosen checkpoints and the test-time mirrors (none,
//!    then flipped top-bottom, then left-right - KonfAI's prediction order),
//!    back onto the scan's grid.
//!
//! The body mask is the caller's (RDS's body contour); upstream runs its
//! own ImpactSeg body model at that step.

pub mod cpu;
pub mod custom;
pub mod layout;
pub mod net;
#[cfg(test)]
mod parity;
pub mod pre;

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use crate::nn::cache::{self, RemoteFile, StoreDtype, WTensor};
use crate::nn::device::DevicePref;
use crate::nn::params::Params;
use crate::nn::pickle::PthReader;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::segresnet::pre::Oriented;
use crate::volume::{AxisOrder, Volume};
use crate::zoo::{Licence, Modality};
use layout::Weights;

/// The engine's folder under the model root.
pub const DIR: &str = "synth";
/// Cross-validation checkpoints per app.
pub const CHECKPOINTS: usize = 5;
/// Test-time mirrors the apps allow (`app.json`'s `tta`).
pub const TTA_MAX: usize = 2;

/// One published TotalSynth app, or a folder of fine-tuned checkpoints.
#[derive(Clone, Copy, Debug)]
pub enum App {
    /// MR (any sequence) to CT.
    Mr,
    /// Linac CBCT to CT.
    Cbct,
    /// One model for both, within a few HU of the dedicated ones.
    MrCbct,
    /// The MR model fine-tuned on 45 whole-body Dixon MRs (BIC-MAC).
    Dixon,
    /// Checkpoints of the same network the user fine-tuned (KonfAI's
    /// `fine-tune` writes them), from a folder of theirs.
    Custom(&'static CustomApp),
}

impl PartialEq for App {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for App {}

/// A folder of fine-tuned checkpoints (`CV_0.pt`, ... or any `*.pt`).
#[derive(Debug)]
pub struct CustomApp {
    pub key: &'static str,
    pub label: &'static str,
    /// Its folder under [`DIR`]: `custom/<key>`.
    pub subdir: &'static str,
    pub folder: PathBuf,
    pub checkpoints: Vec<PathBuf>,
}

impl App {
    /// The published apps.
    pub const BUILTIN: [App; 4] = [App::Mr, App::Cbct, App::MrCbct, App::Dixon];

    pub fn key(self) -> &'static str {
        match self {
            App::Mr => "sct_mr",
            App::Cbct => "sct_cbct",
            App::MrCbct => "sct_mr_cbct",
            App::Dixon => "sct_mr_dixon",
            App::Custom(c) => c.key,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            App::Mr => "TotalSynth MR to CT",
            App::Cbct => "TotalSynth CBCT to CT",
            App::MrCbct => "TotalSynth MR or CBCT to CT",
            App::Dixon => "TotalSynth whole-body Dixon MR to CT",
            App::Custom(c) => c.label,
        }
    }

    pub fn detail(self) -> &'static str {
        match self {
            App::Mr => {
                "MR of any sequence, brain to pelvis; held-out MAE 67.5 HU (pelvis 48, thorax 59, \
                 abdomen 60, brain 80, head and neck 81)."
            }
            App::Cbct => "HU-calibrated linac CBCT, brain to pelvis; held-out MAE 53.6 HU.",
            App::MrCbct => "One model for MR and CBCT, within a few HU of the dedicated ones.",
            App::Dixon => {
                "The MR model fine-tuned on 45 whole-body Dixon MRs (BIC-MAC): 62 HU there, \
                 where the base model gives 101."
            }
            App::Custom(_) => "TotalSynth checkpoints fine-tuned on your own paired cases.",
        }
    }

    /// The published apps and the registered checkpoint folders
    /// ([`custom::register`]), in the order the interface lists them.
    pub fn all() -> Vec<App> {
        let mut v = App::BUILTIN.to_vec();
        v.extend(custom::apps());
        v
    }

    pub fn from_key(key: &str) -> Option<App> {
        App::all().into_iter().find(|a| a.key() == key)
    }

    pub fn is_custom(self) -> bool {
        matches!(self, App::Custom(_))
    }

    /// The weights' licence: the published apps inherit their training
    /// data's (SynthRAD and the prostate cohorts are CC BY-NC); a folder of
    /// the user's carries whatever its data allows, which nothing here
    /// can read.
    pub fn licence(self) -> Licence {
        match self {
            App::Custom(_) => Licence::Undeclared,
            _ => Licence::CcByNc4,
        }
    }

    /// What the app turns into a CT.
    pub fn modality(self) -> Modality {
        match self {
            App::Mr | App::Dixon => Modality::Mr,
            App::Cbct => Modality::Ct,
            App::MrCbct | App::Custom(_) => Modality::CtMr,
        }
    }

    /// The app's folder under [`DIR`]: `mr`, `cbct`, `mr_cbct`, `finetune`,
    /// `custom/<key>`.
    pub fn subdir(self) -> &'static str {
        match self {
            App::Mr => "mr",
            App::Cbct => "cbct",
            App::MrCbct => "mr_cbct",
            App::Dixon => "finetune",
            App::Custom(c) => c.subdir,
        }
    }

    /// The app's folder under the model root.
    pub fn dir(self, root: &Path) -> PathBuf {
        root.join(DIR).join(self.subdir())
    }

    pub fn checkpoint_count(self) -> usize {
        match self {
            App::Custom(c) => c.checkpoints.len(),
            _ => CHECKPOINTS,
        }
    }

    /// The published checkpoint `k` (none for a custom app).
    pub fn file(self, k: usize) -> Option<RemoteFile> {
        // Sizes as the repository lists them (2026-10-12).
        const SIZES: [[u64; 5]; 4] = [
            [104_510_103; 5],
            [104_510_295; 5],
            [
                104_510_295,
                104_510_103,
                104_510_103,
                104_510_295,
                104_510_295,
            ],
            [
                104_510_103,
                104_510_103,
                104_510_103,
                104_510_103,
                104_510_295,
            ],
        ];
        const NAMES: [&str; 5] = ["CV_0.pt", "CV_1.pt", "CV_2.pt", "CV_3.pt", "CV_4.pt"];
        const URLS: [[&str; 5]; 4] = [
            [
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR/CV_0.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR/CV_1.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR/CV_2.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR/CV_3.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR/CV_4.pt",
            ],
            [
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/CBCT/CV_0.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/CBCT/CV_1.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/CBCT/CV_2.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/CBCT/CV_3.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/CBCT/CV_4.pt",
            ],
            [
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR_CBCT/CV_0.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR_CBCT/CV_1.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR_CBCT/CV_2.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR_CBCT/CV_3.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/MR_CBCT/CV_4.pt",
            ],
            [
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/Finetune/CV_0.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/Finetune/CV_1.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/Finetune/CV_2.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/Finetune/CV_3.pt",
                "https://huggingface.co/VBoussot/ImpactSynth/resolve/main/Finetune/CV_4.pt",
            ],
        ];
        let i = match self {
            App::Mr => 0,
            App::Cbct => 1,
            App::MrCbct => 2,
            App::Dixon => 3,
            App::Custom(_) => return None,
        };
        (k < CHECKPOINTS).then(|| RemoteFile {
            name: NAMES[k],
            url: URLS[i][k],
            bytes: SIZES[i][k],
        })
    }

    /// The converted cache of checkpoint `k`.
    pub fn cache_name(k: usize) -> String {
        format!("CV_{k}.safetensors")
    }

    /// Bytes still to download for `checkpoints` (custom apps download
    /// nothing).
    pub fn download_needed(self, checkpoints: &[usize], root: &Path) -> u64 {
        let dir = self.dir(root);
        checkpoints
            .iter()
            .filter(|&&k| !dir.join(Self::cache_name(k)).is_file())
            .filter_map(|&k| self.file(k))
            .filter(|f| !f.is_cached(&dir))
            .map(|f| f.bytes)
            .sum()
    }
}

/// Download and convert every checkpoint of `app` that is not ready yet
/// (the model manager's and the installer's path).
pub fn ensure(app: App, root: &Path, sink: &dyn ProgressSink) -> Result<()> {
    for k in 0..app.checkpoint_count() {
        if !app.dir(root).join(App::cache_name(k)).is_file() {
            load(app, k, root, sink)?;
        }
    }
    Ok(())
}

/// Network passes one member makes over `volume`: one per slice of the
/// working grid and in-plane tile. A pass is about 148 GFLOP with the
/// published widths.
pub fn passes(volume: &Volume) -> usize {
    let o = Oriented::new(volume, AxisOrder::Spl);
    let w = pre::working_dims(o.dims, o.spacing);
    w[0] * pre::tiles(w[1], pre::PATCH).len() * pre::tiles(w[2], pre::PATCH).len()
}

/// Checkpoint `k` of `app` as a network, downloading and converting it on
/// first use. The cache holds the canonical names with the batch norms
/// folded in ([`layout::canonical`]); the published checkpoint stays
/// beside it until the model manager frees it.
pub fn load(app: App, k: usize, root: &Path, sink: &dyn ProgressSink) -> Result<Weights> {
    if k >= app.checkpoint_count() {
        bail!("{} has no checkpoint {k}", app.label());
    }
    let dir = app.dir(root);
    let cache_path = dir.join(App::cache_name(k));
    let params = if cache_path.is_file() {
        sink.report(0.0, &format!("Loading weights ({}, CV {k})", app.label()));
        cache::load_safetensors(&cache_path)?
    } else {
        let src = match app {
            App::Custom(c) => c.checkpoints[k].clone(),
            _ => app.file(k).expect("built-in").ensure(&dir, sink)?,
        };
        let t = convert(&src, &format!("{}, CV {k}", app.label()), sink)?;
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        cache::save_tensor_map(&cache_path, &t, StoreDtype::F32)
            .with_context(|| format!("write {}", cache_path.display()))?;
        t
    };
    Weights::load(&Params::new(params)).with_context(|| format!("{} checkpoint {k}", app.label()))
}

/// Read a KonfAI checkpoint (`{"Model": {"<network>": state dict}}`, or a
/// bare state dict) into the canonical, folded tensors.
pub fn convert(
    path: &Path,
    label: &str,
    sink: &dyn ProgressSink,
) -> Result<std::collections::HashMap<String, WTensor>> {
    let mut reader = PthReader::open(path, "?Model.*")
        .with_context(|| format!("read checkpoint {}", path.display()))?;
    let metas = std::mem::take(&mut reader.tensors);
    let n = metas.len();
    let mut raw = std::collections::HashMap::with_capacity(n);
    for (i, (name, meta)) in metas.into_iter().enumerate() {
        if sink.cancelled() {
            bail!(CANCELLED);
        }
        if name.ends_with("num_batches_tracked") {
            continue;
        }
        sink.report(
            i as f32 / n as f32,
            &format!("Converting weights ({label}): {}/{n}", i + 1),
        );
        let data = reader
            .read_f32(&meta)
            .with_context(|| format!("read tensor {name}"))?;
        raw.insert(
            name,
            WTensor {
                shape: meta.shape.clone(),
                data,
            },
        );
    }
    layout::canonical(raw).with_context(|| format!("{}", path.display()))
}

/// How to run an app.
#[derive(Clone, Debug)]
pub struct Options {
    /// Which checkpoints to average (indices into the app's).
    pub checkpoints: Vec<usize>,
    /// Test-time mirrors beside the plain pass, 0 to [`TTA_MAX`].
    pub tta: usize,
    /// Keep the members' spread as an uncertainty map.
    pub uncertainty: bool,
    pub device: DevicePref,
}

impl Default for Options {
    /// The quick look: one checkpoint, no mirrors.
    fn default() -> Self {
        Options {
            checkpoints: vec![0],
            tta: 0,
            uncertainty: false,
            device: DevicePref::Auto,
        }
    }
}

impl Options {
    /// The full recipe: every checkpoint, both mirrors.
    pub fn full(app: App) -> Options {
        Options {
            checkpoints: (0..app.checkpoint_count()).collect(),
            tta: TTA_MAX,
            ..Options::default()
        }
    }
}

/// A synthetic CT on the scan's own grid.
pub struct SctResult {
    /// HU, `Volume::data` order and dims.
    pub hu: Vec<i16>,
    /// The members' standard deviation (HU) per voxel, when asked for.
    pub sd: Option<Vec<f32>>,
    pub app: &'static str,
    pub checkpoints: Vec<usize>,
    pub tta: usize,
    /// Predictions averaged: checkpoints x (1 + mirrors).
    pub members: usize,
    /// The working grid, `[z, y, x]` of the LPS axis order.
    pub working_dims: [usize; 3],
    /// The clip the normalisation used: the volume's minimum and its 99.5th
    /// percentile, on the working grid (scan values after step 1).
    pub clip: (i16, f64),
    pub device: String,
    pub elapsed_secs: f64,
}

/// Stacks per network call.
const BATCH_CPU: usize = 2;
#[cfg(feature = "gpu")]
const BATCH_GPU: usize = 4;

/// One member's prediction over the working volume: every slice's stack in
/// every patch, the patches written in order (a later one overwrites the
/// overlap, as KonfAI assembles them without a blend).
fn predict(
    net: &dyn Fn(&[f32], usize) -> Vec<f32>,
    norm: &[f32],
    dims: [usize; 3],
    channels: usize,
    batch: usize,
    progress: &dyn Fn(usize),
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<f32>> {
    let [nz, ny, nx] = dims;
    let (ty, tx) = (pre::tiles(ny, pre::PATCH), pre::tiles(nx, pre::PATCH));
    let mut jobs = Vec::with_capacity(nz * ty.len() * tx.len());
    for z in 0..nz {
        for &y in &ty {
            for &x in &tx {
                jobs.push((z, y, x));
            }
        }
    }
    let plane = pre::PATCH * pre::PATCH;
    let mut out = vec![0f32; nz * ny * nx];
    let mut done = 0;
    for chunk in jobs.chunks(batch) {
        if cancelled() {
            bail!(CANCELLED);
        }
        let mut input = vec![0f32; chunk.len() * channels * plane];
        input
            .par_chunks_mut(channels * plane)
            .zip(chunk.par_iter())
            .for_each(|(dst, &(z, y, x))| pre::stack(norm, dims, z, channels, y, x, dst));
        let y = net(&input, chunk.len());
        for (s, &(z, yr, xr)) in chunk.iter().enumerate() {
            let tile = &y[s * plane..(s + 1) * plane];
            for (r, yy) in (yr.0..yr.1).enumerate() {
                let row = &tile[r * pre::PATCH..r * pre::PATCH + (xr.1 - xr.0)];
                out[(z * ny + yy) * nx + xr.0..(z * ny + yy) * nx + xr.1].copy_from_slice(row);
            }
        }
        done += chunk.len();
        progress(done);
    }
    Ok(out)
}

/// The mirrors KonfAI's prediction takes in turn for test-time copies of a
/// `Flip` over the in-plane axes: top-bottom, left-right, both.
const MIRRORS: [(bool, bool); 3] = [(true, false), (false, true), (true, true)];

/// Synthesize a CT from `volume` (MR or CBCT) with `app`. `body` is the
/// body mask on the scan's grid (`Volume::data` order); without one nothing
/// is set to air.
pub fn run(
    volume: &Volume,
    body: Option<&[u8]>,
    app: App,
    opts: &Options,
    root: &Path,
    progress: &Progress,
) -> Result<SctResult> {
    let start = std::time::Instant::now();
    if opts.checkpoints.is_empty() {
        bail!("no checkpoint chosen");
    }
    if let Some(&k) = opts
        .checkpoints
        .iter()
        .find(|&&k| k >= app.checkpoint_count())
    {
        bail!("{} has no checkpoint {k}", app.label());
    }
    if let Some(b) = body {
        if b.len() != volume.data.len() {
            bail!("the body mask is not on the scan's grid");
        }
    }
    let tta = opts.tta.min(TTA_MAX);

    // ---- the scan on the working grid -------------------------------------
    progress.set_phase(0.0, 0.08);
    progress.report(0.0, "Preparing the scan");
    let o = Oriented::new(volume, AxisOrder::Spl);
    let dims = o.dims;
    let n = dims.iter().product::<usize>();
    let read_i16 = |f: &(dyn Fn(usize) -> i16 + Sync)| -> Vec<i16> {
        let mut out = vec![0i16; n];
        out.par_chunks_mut(dims[1] * dims[2])
            .enumerate()
            .for_each(|(a, slab)| {
                for b in 0..dims[1] {
                    for c in 0..dims[2] {
                        slab[b * dims[2] + c] = f(o.volume_index(volume.dims, [a, b, c]));
                    }
                }
            });
        out
    };
    let scan = read_i16(&|i| pre::cast_input(volume.data[i]));
    let wdims = pre::working_dims(dims, o.spacing);
    let scan_f: Vec<f32> = scan.iter().map(|&v| v as f32).collect();
    let work: Vec<i16> = pre::resample_linear(&scan_f, dims, o.spacing, wdims)
        .into_iter()
        .map(pre::to_i16)
        .collect();
    drop(scan_f);
    let clip = (
        work.iter().copied().min().unwrap_or(0),
        pre::percentile(&work, pre::PERCENTILE),
    );
    let norm = pre::clip_normalize(&work);
    drop(work);
    let mask_w = match body {
        Some(b) => {
            let m = read_i16(&|i| b[i] as i16);
            let m: Vec<u8> = m.into_iter().map(|v| u8::from(v != 0)).collect();
            let m = pre::resample_nearest(&m, dims, o.spacing, wdims);
            pre::dilate(&m, wdims, pre::DILATE)
        }
        None => vec![1u8; wdims.iter().product()],
    };
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- the members ---------------------------------------------------------
    progress.set("Choosing the compute device");
    let gpu = opts.device.resolve()?;
    let copies = 1 + tta;
    let members = opts.checkpoints.len() * copies;
    let nw = wdims.iter().product::<usize>();
    let mut sum = vec![0f32; nw];
    let mut sumsq = opts.uncertainty.then(|| vec![0f64; nw]);
    let [_, ny, nx] = wdims;
    let passes = wdims[0] * pre::tiles(ny, pre::PATCH).len() * pre::tiles(nx, pre::PATCH).len();
    let mut device_desc = String::new();
    let span = 0.9 / opts.checkpoints.len() as f32;
    for (ci, &k) in opts.checkpoints.iter().enumerate() {
        progress.set_phase(0.08 + span * ci as f32, span * 0.1);
        let weights = load(app, k, root, progress)?;
        let channels = weights.in_ch;
        progress.set_phase(0.08 + span * ci as f32 + span * 0.1, span * 0.9);
        let mut run_copy =
            |copy: usize, net: &dyn Fn(&[f32], usize) -> Vec<f32>, batch: usize| -> Result<()> {
                let (fy, fx) = if copy == 0 {
                    (false, false)
                } else {
                    MIRRORS[copy - 1]
                };
                let input = pre::flip(&norm, wdims, fy, fx);
                let report = |done: usize| {
                    let all = passes * copies;
                    let at = copy * passes + done;
                    progress.report(
                        at as f32 / all as f32,
                        &format!(
                            "Synthesizing ({}, CV {k}{}): slice {}/{}",
                            app.label(),
                            if copy == 0 {
                                String::new()
                            } else {
                                format!(", mirror {copy}")
                            },
                            (done * wdims[0]).div_ceil(passes.max(1)),
                            wdims[0]
                        ),
                    );
                };
                let y = predict(net, &input, wdims, channels, batch, &report, &|| {
                    progress.cancelled()
                })?;
                let y = pre::flip(&y, wdims, fy, fx);
                sum.par_iter_mut()
                    .zip(y.par_iter())
                    .zip(mask_w.par_iter())
                    .for_each(|((s, &v), &m)| {
                        let hu = if m != 0 { pre::member_hu(v) } else { pre::AIR };
                        *s += hu as f32;
                    });
                if let Some(sq) = sumsq.as_mut() {
                    sq.par_iter_mut()
                        .zip(y.par_iter())
                        .zip(mask_w.par_iter())
                        .for_each(|((s, &v), &m)| {
                            let hu = if m != 0 { pre::member_hu(v) } else { pre::AIR } as f64;
                            *s += hu * hu;
                        });
                }
                Ok(())
            };
        match &gpu {
            #[cfg(feature = "gpu")]
            Some(ctx) => {
                device_desc = ctx.describe();
                progress.set_device(&device_desc);
                crate::nn::device::guarded(|| {
                    let net =
                        net::UnetPp::<crate::medsam2::engine::Gpu>::upload(&weights, ctx.device());
                    let f = |x: &[f32], n: usize| net.forward(x, n, pre::PATCH, pre::PATCH);
                    for copy in 0..copies {
                        run_copy(copy, &f, BATCH_GPU)?;
                    }
                    Ok(())
                })?;
            }
            #[cfg(not(feature = "gpu"))]
            Some(ctx) => ctx.unreachable(),
            None => {
                device_desc = crate::nn::device::describe_cpu();
                progress.set_device(&device_desc);
                let f = |x: &[f32], n: usize| cpu::forward(&weights, x, n, pre::PATCH, pre::PATCH);
                for copy in 0..copies {
                    run_copy(copy, &f, BATCH_CPU)?;
                }
            }
        }
    }

    // ---- back onto the scan ----------------------------------------------------
    progress.set_phase(0.98, 0.02);
    progress.report(0.0, "Writing the synthetic CT on the scan's grid");
    let mean: Vec<f32> = sum
        .par_iter()
        .map(|&s| pre::to_i16(s / members as f32) as f32)
        .collect();
    // The working grid's covering spacing (its extent is the scan's).
    let cover: [f64; 3] = std::array::from_fn(|a| o.spacing[a] * dims[a] as f64 / wdims[a] as f64);
    let back = pre::resample_linear(&mean, wdims, cover, dims);
    let sd_back = sumsq.map(|sq| {
        let sd: Vec<f32> = sq
            .par_iter()
            .zip(sum.par_iter())
            .map(|(&q, &s)| {
                if members < 2 {
                    return 0.0;
                }
                let n = members as f64;
                let var = (q - (s as f64) * (s as f64) / n) / (n - 1.0);
                var.max(0.0).sqrt() as f32
            })
            .collect();
        pre::resample_linear(&sd, wdims, cover, dims)
    });
    let mut hu = vec![0i16; volume.data.len()];
    let mut sd = sd_back.as_ref().map(|_| vec![0f32; volume.data.len()]);
    for a in 0..dims[0] {
        for b in 0..dims[1] {
            for c in 0..dims[2] {
                let m = (a * dims[1] + b) * dims[2] + c;
                let v = o.volume_index(volume.dims, [a, b, c]);
                hu[v] = pre::to_i16(back[m]);
                if let (Some(out), Some(src)) = (sd.as_mut(), sd_back.as_ref()) {
                    out[v] = src[m];
                }
            }
        }
    }
    Ok(SctResult {
        hu,
        sd,
        app: app.key(),
        checkpoints: opts.checkpoints.clone(),
        tta,
        members,
        working_dims: wdims,
        clip,
        device: device_desc,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
}
