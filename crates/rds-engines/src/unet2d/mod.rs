//! Lungs and lung lobes on CT - a pure-Rust re-implementation of
//! [lungmask](https://github.com/JoHof/lungmask) (Hofmanninger et al.,
//! *Automatic lung segmentation in routine imaging is primarily a data
//! diversity problem, not a methodology problem*, Eur Radiol Exp 4, 50,
//! 2020; Apache-2.0, weights published with the code).
//!
//! Three published models share one 2-D U-Net ([`net`]): **R231**, left
//! and right lung trained on a deliberately diverse set of routine scans
//! (it keeps tumours, effusions and dense pathology inside the lung);
//! **LTRCLobes**, the five lobes; **R231CovidWeb**, R231 with COVID-19
//! cases added. Every axial slice is cropped to the body, resized to
//! 256 x 256 and segmented on its own ([`pre`]); the stack of answers is
//! then cleaned in 3-D - stray pieces handed to their neighbours, each
//! label kept as its largest piece with its holes filled ([`post`]) - and
//! put back into each slice's box.
//!
//! A fourth model is upstream's `LTRCLobes_R231` fusion: LTRCLobes and
//! R231 both run, lung that R231 finds and LTRCLobes leaves unlabelled
//! becomes a *spare* label, whatever is outside R231's lungs is cleared,
//! and the spare voxels go to the lobe they border most (the same
//! post-processing, on the scan's own grid, with the spare label).
//!
//! The arrays are the ones lungmask sees: SimpleITK's after
//! `DICOMOrient("LPS")`, axes [S, P, L], no resampling. The network runs
//! through `burn` on the GPU when there is one, and on the CPU on the
//! U-Net GEMM kernels ([`cpu`]).

pub mod cpu;
pub mod net;
pub mod post;
pub mod pre;

use anyhow::{bail, Result};
use rayon::prelude::*;
use std::path::Path;

use crate::autoseg::{organ_hits, AutosegResult};
use crate::nn::cache::{self, ConvertSpec, RemoteFile};
use crate::nn::device::DevicePref;
use crate::nn::params::Params;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::volume::{AxisOrder, Volume};
use pre::{Img, SIZE};

/// The engine folder under the model root.
pub const DIR: &str = "lungmask";

/// R231 and R231CovidWeb: label 1 is the right lung, 2 the left.
pub const CLASSES_LUNGS: [&str; 2] = ["lung_right", "lung_left"];
/// LTRCLobes.
pub const CLASSES_LOBES: [&str; 5] = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
];

/// One of lungmask's published models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lungmask {
    R231,
    LtrcLobes,
    R231CovidWeb,
    /// `LTRCLobes_R231`: the lobes, with R231's lungs filling the gaps.
    LobesR231,
}

impl Lungmask {
    /// Every model, in the order the interface lists them.
    pub const ALL: [Lungmask; 4] = [
        Lungmask::R231,
        Lungmask::LtrcLobes,
        Lungmask::LobesR231,
        Lungmask::R231CovidWeb,
    ];
    /// The published networks, one checkpoint each.
    pub const NETWORKS: [Lungmask; 3] =
        [Lungmask::R231, Lungmask::LtrcLobes, Lungmask::R231CovidWeb];

    /// The networks the model runs, in order: itself, or LTRCLobes then
    /// R231 for the fusion.
    pub fn networks(self) -> &'static [Lungmask] {
        match self {
            Lungmask::R231 => &[Lungmask::R231],
            Lungmask::LtrcLobes => &[Lungmask::LtrcLobes],
            Lungmask::R231CovidWeb => &[Lungmask::R231CovidWeb],
            Lungmask::LobesR231 => &[Lungmask::LtrcLobes, Lungmask::R231],
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Lungmask::R231 => "lungmask_r231",
            Lungmask::LtrcLobes => "lungmask_lobes",
            Lungmask::R231CovidWeb => "lungmask_r231covid",
            Lungmask::LobesR231 => "lungmask_lobes_r231",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Lungmask::R231 => "lungmask R231",
            Lungmask::LtrcLobes => "lungmask LTRCLobes",
            Lungmask::R231CovidWeb => "lungmask R231CovidWeb",
            Lungmask::LobesR231 => "lungmask LTRCLobes + R231",
        }
    }

    pub fn detail(self) -> &'static str {
        match self {
            Lungmask::R231 => {
                "Left and right lung, slice by slice; trained on routine scans of every kind, \
                 so tumours, effusions and dense pathology stay inside the lung."
            }
            Lungmask::LtrcLobes => {
                "The five lung lobes (LTRC data); weaker where fissures are not visible or \
                 dense pathology is present."
            }
            Lungmask::R231CovidWeb => "R231 with COVID-19 cases added to its training data.",
            Lungmask::LobesR231 => {
                "The five lobes inside R231's lungs (upstream's LTRCLobes_R231): lung the lobe \
                 model leaves out - dense pathology, missing fissures - goes to the lobe it \
                 borders most. Runs both networks."
            }
        }
    }

    /// The published checkpoint (the fusion has none of its own:
    /// [`Lungmask::networks`]).
    pub fn file(self) -> RemoteFile {
        match self {
            Lungmask::R231 => RemoteFile {
                name: "unet_r231-d5d2fc3d.pth",
                url: "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231-d5d2fc3d.pth",
                bytes: 124_340_403,
            },
            Lungmask::LtrcLobes | Lungmask::LobesR231 => RemoteFile {
                name: "unet_ltrclobes-3a07043d.pth",
                url: "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_ltrclobes-3a07043d.pth",
                bytes: 124_338_462,
            },
            Lungmask::R231CovidWeb => RemoteFile {
                name: "unet_r231covid-0de78a7e.pth",
                url: "https://github.com/JoHof/lungmask/releases/download/v0.0/unet_r231covid-0de78a7e.pth",
                bytes: 124_340_403,
            },
        }
    }

    /// The converted cache beside it.
    pub fn cache_name(self) -> &'static str {
        match self {
            Lungmask::R231 => "unet_r231.safetensors",
            Lungmask::LtrcLobes | Lungmask::LobesR231 => "unet_ltrclobes.safetensors",
            Lungmask::R231CovidWeb => "unet_r231covid.safetensors",
        }
    }

    pub fn classes(self) -> &'static [&'static str] {
        match self {
            Lungmask::LtrcLobes | Lungmask::LobesR231 => &CLASSES_LOBES,
            _ => &CLASSES_LUNGS,
        }
    }
}

/// Bytes still to download before `model` runs offline.
pub fn download_needed(model: Lungmask, root: &Path) -> u64 {
    let dir = root.join(DIR);
    model
        .networks()
        .iter()
        .filter(|m| !(dir.join(m.cache_name()).is_file() || m.file().is_cached(&dir)))
        .map(|m| m.file().bytes)
        .sum()
}

/// A network's weights, downloading and converting on first use (for the
/// fusion, its lobe network's; [`Lungmask::networks`] lists both).
pub fn load(model: Lungmask, root: &Path, sink: &dyn ProgressSink) -> Result<Params> {
    let dir = root.join(DIR);
    let spec = ConvertSpec {
        top_key: "",
        // The residual branch is built but unused with `residual=False`.
        keep: &|name, _| !name.contains("residual") && !name.ends_with("num_batches_tracked"),
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

/// Slices per forward pass.
const BATCH_CPU: usize = 4;
#[cfg(feature = "gpu")]
const BATCH_GPU: usize = 8;

/// The network's labels for every prepared slice, `batch` slices at a time.
fn infer(
    net: &dyn Fn(&[f32], usize) -> Vec<u8>,
    inputs: &[Vec<f32>],
    batch: usize,
    progress: &Progress,
) -> Result<Vec<u8>> {
    let plane = SIZE * SIZE;
    let mut out = Vec::with_capacity(inputs.len() * plane);
    for (bi, chunk) in inputs.chunks(batch).enumerate() {
        if progress.cancelled() {
            bail!(CANCELLED);
        }
        let flat: Vec<f32> = chunk.iter().flat_map(|s| s.iter().copied()).collect();
        out.extend(net(&flat, chunk.len()));
        let done = (bi * batch + chunk.len()).min(inputs.len());
        progress.report(
            done as f32 / inputs.len() as f32,
            &format!("Segmenting (lungmask): slice {done}/{}", inputs.len()),
        );
    }
    Ok(out)
}

/// One network over the prepared slices: the cleaned label stack, put back
/// into each slice's box (`[S, P, L]`, `d`).
fn segment_stack(
    weights: &net::Weights,
    gpu: &Option<crate::nn::device::GpuContext>,
    prepared: &[(Vec<f32>, [usize; 4])],
    d: [usize; 3],
    progress: &Progress,
) -> Result<(Vec<u8>, String)> {
    let inputs: Vec<Vec<f32>> = prepared.iter().map(|(x, _)| x.clone()).collect();
    let (stack, device_desc) = match gpu {
        #[cfg(feature = "gpu")]
        Some(ctx) => {
            let desc = ctx.describe();
            progress.set_device(&desc);
            let labels = crate::nn::device::guarded(|| {
                let net = net::Unet2d::<crate::medsam2::engine::Gpu>::upload(weights, ctx.device());
                infer(
                    &|x: &[f32], n: usize| net.labels(x, n, SIZE, SIZE),
                    &inputs,
                    BATCH_GPU,
                    progress,
                )
            })?;
            (labels, desc)
        }
        #[cfg(not(feature = "gpu"))]
        Some(ctx) => ctx.unreachable(),
        None => {
            let desc = crate::nn::device::describe_cpu();
            progress.set_device(&desc);
            let labels = infer(
                &|x: &[f32], n: usize| cpu::labels(weights, x, n, SIZE, SIZE),
                &inputs,
                BATCH_CPU,
                progress,
            )?;
            (labels, desc)
        }
    };
    progress.report(1.0, "Cleaning up the lung labels");
    let stack = post::postprocess(&stack, [d[0], SIZE, SIZE]);
    if progress.cancelled() {
        bail!(CANCELLED);
    }
    let plane = SIZE * SIZE;
    let mut out = vec![0u8; d[0] * d[1] * d[2]];
    for (s, (_, b)) in prepared.iter().enumerate() {
        let mask = Img {
            h: SIZE,
            w: SIZE,
            data: stack[s * plane..(s + 1) * plane].to_vec(),
        };
        let back = pre::reshape_mask(&mask, *b, d[1], d[2]);
        out[s * d[1] * d[2]..(s + 1) * d[1] * d[2]].copy_from_slice(&back.data);
    }
    Ok((out, device_desc))
}

/// `LMInferer.apply` with a fill model: R231's lung that the lobes leave
/// unlabelled becomes a spare label, everything outside R231's lungs is
/// cleared, and the spare voxels go to their neighbouring lobes.
pub fn fuse(lobes: &[u8], lungs: &[u8], d: [usize; 3]) -> Vec<u8> {
    let spare = lobes.iter().copied().max().unwrap_or(0) + 1;
    let fused: Vec<u8> = lobes
        .iter()
        .zip(lungs)
        .map(|(&l, &r)| match (l, r) {
            (_, 0) => 0,
            (0, _) => spare,
            (l, _) => l,
        })
        .collect();
    post::postprocess_spare(&fused, d, &[spare])
}

/// Run one lungmask model on a CT volume. Blocking; observe and cancel
/// through `progress`. `root` is the model folder.
pub fn run(
    volume: &Volume,
    model: Lungmask,
    device: DevicePref,
    root: &Path,
    progress: &Progress,
) -> Result<AutosegResult> {
    let t0 = std::time::Instant::now();
    let nets = model.networks();
    progress.set_phase(0.0, 0.1);
    let mut weights = Vec::with_capacity(nets.len());
    for m in nets {
        let params = load(*m, root, progress)?;
        weights.push(net::Weights::load(&params)?);
        if progress.cancelled() {
            bail!(CANCELLED);
        }
    }

    // ---- the [S, P, L] stack lungmask reads -----------------------------
    progress.set_phase(0.1, 0.1);
    progress.report(0.0, "Finding the body on every slice");
    let (perm, flip) = volume.axes_toward(AxisOrder::Spl);
    let d: [usize; 3] = std::array::from_fn(|a| volume.dims[perm[a]]);
    let [nx, ny, _] = volume.dims;
    let vol_index = |m: [usize; 3]| -> usize {
        let mut c = [0usize; 3];
        for a in 0..3 {
            c[perm[a]] = if flip[a] { d[a] - 1 - m[a] } else { m[a] };
        }
        c[2] * nx * ny + c[1] * nx + c[0]
    };
    let prepared: Vec<(Vec<f32>, [usize; 4])> = (0..d[0])
        .into_par_iter()
        .map(|s| {
            let mut img = Img::new(d[1], d[2]);
            for r in 0..d[1] {
                for c in 0..d[2] {
                    img.data[r * d[2] + c] = volume.data[vol_index([s, r, c])].clamp(-1024, 600);
                }
            }
            pre::prepare_slice(&img)
        })
        .collect();
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- the networks -------------------------------------------------------
    progress.set("Choosing the compute device");
    let gpu = device.resolve()?;
    let fusing = nets.len() > 1;
    let span = if fusing { 0.6 } else { 0.8 } / nets.len() as f32;
    let mut stacks = Vec::with_capacity(nets.len());
    let mut device_desc = String::new();
    for (i, w) in weights.iter().enumerate() {
        progress.set_phase(0.2 + span * i as f32, span);
        let (stack, desc) = segment_stack(w, &gpu, &prepared, d, progress)?;
        stacks.push(stack);
        device_desc = desc;
    }
    drop(weights);
    let stack = if fusing {
        progress.set_phase(0.8, 0.2);
        progress.report(0.0, "Fusing the lobes with the lung mask");
        fuse(&stacks[0], &stacks[1], d)
    } else {
        stacks.pop().unwrap_or_default()
    };
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    // ---- back onto the scan's grid --------------------------------------------
    let mut labels = vec![0u8; volume.data.len()];
    for s in 0..d[0] {
        for r in 0..d[1] {
            for c in 0..d[2] {
                let l = stack[(s * d[1] + r) * d[2] + c];
                if l != 0 {
                    labels[vol_index([s, r, c])] = l;
                }
            }
        }
    }
    let classes = model.classes();
    let organs = organ_hits(&labels, volume.spacing, classes);
    progress.report(1.0, "Lung segmentation finished");
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
    fn the_three_networks_have_their_own_files_and_tables() {
        let mut names: Vec<&str> = Lungmask::NETWORKS.iter().map(|m| m.file().name).collect();
        names.dedup();
        assert_eq!(names.len(), 3);
        for m in Lungmask::NETWORKS {
            assert_eq!(m.networks(), [m]);
            assert!(m.file().url.ends_with(m.file().name));
            assert!(m.cache_name().ends_with(".safetensors"));
        }
        assert_eq!(Lungmask::LtrcLobes.classes().len(), 5);
        assert_eq!(Lungmask::R231.classes(), ["lung_right", "lung_left"]);
        let root = std::env::temp_dir().join("rds_lungmask_none");
        assert_eq!(download_needed(Lungmask::R231, &root), 124_340_403);
        // The fusion downloads both of its networks and lands the lobes.
        assert_eq!(
            Lungmask::LobesR231.networks(),
            [Lungmask::LtrcLobes, Lungmask::R231]
        );
        assert_eq!(
            download_needed(Lungmask::LobesR231, &root),
            124_338_462 + 124_340_403
        );
        assert_eq!(Lungmask::LobesR231.classes(), CLASSES_LOBES);
    }

    #[test]
    fn the_fusion_keeps_lobes_inside_the_lungs_and_fills_their_gaps() {
        // 1 x 3 x 6: lobes 1 | gap | 2 inside a lung that covers x 0-4; a
        // lobe voxel at x = 5 outside the lung.
        let d = [1, 3, 6];
        let mut lobes = vec![0u8; 18];
        let mut lungs = vec![0u8; 18];
        for y in 0..3 {
            lobes[y * 6] = 1;
            lobes[y * 6 + 1] = 1;
            lobes[y * 6 + 3] = 2;
            lobes[y * 6 + 4] = 2;
            lobes[y * 6 + 5] = 2;
            for x in 0..5 {
                lungs[y * 6 + x] = 1;
            }
        }
        let out = fuse(&lobes, &lungs, d);
        for y in 0..3 {
            assert_ne!(out[y * 6 + 2], 0, "the gap goes to a lobe");
            assert_eq!(out[y * 6 + 5], 0, "outside the lungs is cleared");
            assert_eq!(out[y * 6], 1);
            assert_eq!(out[y * 6 + 4], 2);
        }
    }
}
