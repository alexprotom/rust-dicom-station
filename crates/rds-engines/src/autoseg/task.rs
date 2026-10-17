//! Segmentation *tasks*: one question, asked of one or more nnU-Net
//! networks, with everything TotalSegmentator does around them.
//!
//! A task is data ([`NnTask`]): its class table, the networks that answer
//! it ([`Part`]: a [`ModelSpec`] and the look-up table that turns the
//! network's own labels into the task's), how many folds to ensemble, the
//! sliding-window step, the region to run on ([`Crop`]) and the
//! post-processing ([`Post`]). One runner ([`run_task_labels`]) serves them
//! all - the 117-class `total`, the MR models, MRSegmentator, the body
//! outline and every open task of TotalSegmentator's catalogue - so a new
//! nnU-Net model is a new row, not new code.
//!
//! The crop cascade is upstream's: a coarse model runs on the whole scan,
//! the bounding box of the classes the task names (widened by `addon_mm` on
//! every side, rounded down to voxels of the scan) is cut out of the scan,
//! the task's own networks run on that box only, and their labels are
//! pasted back into an empty volume. If the coarse model finds none of
//! those classes the task's answer is empty, as upstream's is.

use anyhow::{bail, Context, Result};
use std::borrow::Cow;
use std::path::Path;
use std::sync::Mutex;

use super::kept::{self, NetKey, Nets, PartNets};
use super::weights::{self, ModelSpec};
use super::{cpu, infer, net, post, preprocess};
use crate::nn::device::DevicePref;
use crate::progress::{Progress, ProgressSink, CANCELLED};
use crate::volume::{paste_labels, Volume};
pub use crate::zoo::{Licence, Modality};

/// One network of a task.
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub spec: ModelSpec,
    /// The network's label `l` is the task's label `lut[l]`; 0, or an index
    /// past the end, drops it (upstream's auxiliary classes).
    pub lut: &'static [u8],
}

/// The coarse model a crop is taken from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CropBy {
    /// `total` v2 at 6 mm (TotalSegmentator's default crop model).
    Total6mm,
    /// `total` v2 at 3 mm (its `robust_crop`).
    Total3mm,
    /// `total_mr` at 3 mm (every MR task).
    TotalMr3mm,
    /// The 6 mm body model (tasks cropped to the trunk).
    Body6mm,
    /// Another task of the catalogue, by key.
    Task(&'static str),
}

impl CropBy {
    /// The task that runs to find the crop.
    pub fn task(self) -> Option<&'static NnTask> {
        match self {
            CropBy::Total6mm => Some(&super::total::TOTAL_FASTEST),
            CropBy::Total3mm => Some(&super::total::TOTAL_FAST),
            CropBy::TotalMr3mm => Some(&super::total::TOTAL_MR_FAST),
            CropBy::Body6mm => Some(&super::total::BODY_FAST),
            CropBy::Task(key) => task_by_key(key),
        }
    }
}

/// Where a task runs: the bounding box of `classes` as `by` finds them,
/// widened by `addon_mm`.
#[derive(Clone, Copy, Debug)]
pub struct Crop {
    pub by: CropBy,
    pub classes: &'static [&'static str],
    pub addon_mm: f64,
}

/// Post-processing: on the model grid before the labels go back to the
/// scan, or on the scan's own grid after.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Post {
    None,
    /// [`post::vertebrae_pp`].
    VertebraePp,
    /// [`post::body`].
    Body,
    /// The nnU-Net v1 training's own `postprocessing.json`, on the scan
    /// grid ([`super::v1::apply_post`]).
    NnUnetV1,
    /// Upstream's `remove_outside`: labels outside the crop model's
    /// `classes`, dilated by `dilation_mm`, are cleared (on the scan grid,
    /// the dilation in whole voxels of its mean spacing, 6-connected, as
    /// TotalSegmentator's `remove_outside_of_mask`).
    RemoveOutside {
        classes: &'static [&'static str],
        dilation_mm: f64,
    },
}

/// Which of a model's cross-validation folds a task runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldUse {
    /// Every fold the model has, logits summed (nnU-Net's ensemble).
    All,
    /// Only `fold_0`.
    First,
}

/// One automatic segmentation task.
#[derive(Debug)]
pub struct NnTask {
    /// Stable identity (TotalSegmentator's task name where there is one).
    pub key: &'static str,
    pub label: &'static str,
    /// Heading the interface lists it under.
    pub group: &'static str,
    pub detail: &'static str,
    pub modality: Modality,
    pub licence: Licence,
    /// Task label `l` (1-based) is `classes[l - 1]`.
    pub classes: &'static [&'static str],
    /// The networks, merged in order (a later part's labels win).
    pub parts: &'static [Part],
    pub folds: FoldUse,
    /// Sliding-window step as a fraction of the patch (upstream: 0.8 for
    /// the `total` family, 0.5 otherwise).
    pub step: f64,
    pub crop: Option<Crop>,
    pub post: Post,
}

impl NnTask {
    /// Name of task label `l`; empty for 0 or out of range.
    pub fn class_name(&self, l: u8) -> &'static str {
        if l == 0 {
            ""
        } else {
            self.classes.get(l as usize - 1).copied().unwrap_or("")
        }
    }

    /// The task label of a class name.
    pub fn label_of(&self, name: &str) -> Option<u8> {
        self.classes
            .iter()
            .position(|c| c.eq_ignore_ascii_case(name.trim()))
            .map(|i| i as u8 + 1)
    }

    /// The specs a run downloads: its parts (the ones `parts` selects) and,
    /// recursively, its crop model's.
    pub fn specs(&self, parts: Option<&[bool]>) -> Vec<ModelSpec> {
        let mut out: Vec<ModelSpec> = self
            .parts
            .iter()
            .enumerate()
            .filter(|(i, _)| parts.is_none_or(|m| m.get(*i).copied().unwrap_or(true)))
            .map(|(_, p)| p.spec)
            .collect();
        if let Some(t) = self.crop.and_then(|c| c.by.task()) {
            for s in t.specs(None) {
                if !out.iter().any(|o| o.key == s.key) {
                    out.push(s);
                }
            }
        }
        out
    }
}

/// `[0, 1, 2, ..., 255]`: the look-up table of a network that predicts the
/// task's labels directly.
pub const IDENTITY_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = i as u8;
        i += 1;
    }
    t
};

/// `[0, off + 1, off + 2, ...]`: a part whose labels are a contiguous slice
/// of the task's table.
pub const fn shifted<const N: usize>(off: u8) -> [u8; N] {
    let mut t = [0u8; N];
    let mut i = 1;
    while i < N {
        t[i] = off + i as u8;
        i += 1;
    }
    t
}

/// Every task, in the order the interface lists them: the `total` family,
/// the MR models, the body outline, the catalogue, the licensed tasks, the
/// nnU-Net v1 models, then the model folders the user added.
pub fn all_tasks() -> Vec<&'static NnTask> {
    let mut v = builtin_tasks();
    v.extend(super::custom::tasks());
    v
}

/// [`all_tasks`] but the user's model folders.
pub fn builtin_tasks() -> Vec<&'static NnTask> {
    let mut v: Vec<&'static NnTask> = super::total::TASKS.to_vec();
    v.extend(super::tasks::TASKS.iter());
    v.extend(super::tasks::LICENSED_TASKS.iter());
    v.extend(super::v1::TASKS.iter());
    v
}

/// The task with this key.
pub fn task_by_key(key: &str) -> Option<&'static NnTask> {
    all_tasks().into_iter().find(|t| t.key == key)
}

/// How to run a task.
#[derive(Clone, Debug, Default)]
pub struct NnOptions {
    pub device: DevicePref,
    /// For a task of several parts, which ones to run (`None`: all).
    pub parts: Option<Vec<bool>>,
}

/// A task's answer on the scan's own grid.
pub struct TaskLabels {
    /// Task labels per voxel, `Volume::data` order.
    pub labels: Vec<u8>,
    /// The device the networks ran on.
    pub device: String,
    /// What the caller should be told (a crop that found nothing).
    pub notes: Vec<String>,
}

/// Bytes still to download before `task` runs offline.
pub fn download_needed(task: &NnTask, parts: Option<&[bool]>, root: &Path) -> u64 {
    let mut urls: Vec<&str> = Vec::new();
    let mut bytes = 0;
    for s in task.specs(parts) {
        let dir = root.join(s.home.subdir());
        // Two trainings of one release zip are one download.
        let shared = s.source == weights::Source::Release && urls.contains(&s.url);
        if weights::is_cached(&s, &dir) || shared {
            continue;
        }
        urls.push(s.url);
        bytes += s.download_bytes();
    }
    bytes
}

/// The voxel box `[lo, hi)` around every voxel whose label is in `ids`,
/// widened by `addon_mm` per side (rounded down to whole voxels of the
/// scan's own spacing, as upstream's `crop_to_mask` does), clamped to the
/// scan. `None` when no voxel carries one of the labels.
pub fn crop_box(
    labels: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    ids: &[u8],
    addon_mm: f64,
) -> Option<([usize; 3], [usize; 3])> {
    let mut want = [false; 256];
    for &i in ids {
        want[i as usize] = true;
    }
    let [nx, ny, _] = dims;
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for (v, &l) in labels.iter().enumerate() {
        if !want[l as usize] {
            continue;
        }
        let c = [v % nx, (v / nx) % ny, v / (nx * ny)];
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a] + 1);
        }
    }
    if lo[0] == usize::MAX {
        return None;
    }
    let add: [usize; 3] = std::array::from_fn(|a| (addon_mm / spacing[a]).max(0.0) as usize);
    Some((
        std::array::from_fn(|a| lo[a].saturating_sub(add[a])),
        std::array::from_fn(|a| (hi[a] + add[a]).min(dims[a])),
    ))
}

/// Run a task over a volume and return its labels **on the volume's own
/// grid**. Blocking; observe and cancel through `progress`.
///
/// `root` is the model folder (the engines' folders are below it), and
/// `window` the slice of the overall progress bar this run owns,
/// `(0.0, 1.0)` for all of it: [`Progress::set_phase`] is absolute, so a
/// caller with work of its own afterwards says so here.
pub fn run_task_labels(
    volume: &Volume,
    task: &NnTask,
    opts: &NnOptions,
    root: &Path,
    window: (f32, f32),
    progress: &Progress,
) -> Result<TaskLabels> {
    let parts: Vec<Part> = task
        .parts
        .iter()
        .enumerate()
        .filter(|(i, _)| {
            opts.parts
                .as_ref()
                .is_none_or(|m| m.get(*i).copied().unwrap_or(true))
        })
        .map(|(_, p)| *p)
        .collect();
    if parts.is_empty() {
        bail!("no sub-models selected");
    }
    let (base, span) = window;
    let Some(crop) = task.crop else {
        let (labels, device) =
            run_parts(volume, task, &parts, opts.device, root, window, progress)?;
        return Ok(TaskLabels {
            labels,
            device,
            notes: Vec::new(),
        });
    };
    let by = crop
        .by
        .task()
        .with_context(|| format!("task '{}': unknown crop model", task.key))?;
    let coarse_span = 0.3 * span;
    let coarse = run_task_labels(
        volume,
        by,
        &NnOptions {
            device: opts.device,
            parts: None,
        },
        root,
        (base, coarse_span),
        progress,
    )
    .with_context(|| format!("crop model for '{}'", task.label))?;
    let mut ids = Vec::with_capacity(crop.classes.len());
    for c in crop.classes {
        ids.push(
            by.label_of(c)
                .with_context(|| format!("crop class '{c}' is not one of '{}'", by.key))?,
        );
    }
    let Some((lo, hi)) = crop_box(
        &coarse.labels,
        volume.dims,
        volume.spacing,
        &ids,
        crop.addon_mm,
    ) else {
        return Ok(TaskLabels {
            labels: vec![0; volume.data.len()],
            device: coarse.device,
            notes: vec![format!(
                "{} found none of {}, so there was nothing to segment",
                by.label,
                crop.classes.join(", ")
            )],
        });
    };
    let (sub, lo, hi) = volume.crop(lo, hi);
    let (part_labels, device) = run_parts(
        &sub,
        task,
        &parts,
        opts.device,
        root,
        (base + coarse_span, span - coarse_span),
        progress,
    )?;
    let mut labels = paste_labels(volume.dims, lo, hi, &part_labels);
    if let Post::RemoveOutside {
        classes,
        dilation_mm,
    } = task.post
    {
        let keep: Vec<u8> = classes.iter().filter_map(|c| by.label_of(c)).collect();
        let mean = volume.spacing.iter().sum::<f64>() / 3.0;
        let iterations = (dilation_mm / mean) as usize;
        remove_outside(&mut labels, &coarse.labels, volume.dims, &keep, iterations);
    }
    Ok(TaskLabels {
        labels,
        device,
        notes: Vec::new(),
    })
}

/// Clear `labels` outside the voxels of `mask` carrying one of `ids`,
/// dilated `iterations` times with the 6-neighbourhood (scipy's
/// `binary_dilation` with its default structure).
///
/// `k` dilations with the 6-neighbourhood reach exactly the voxels within
/// `k` steps of a seed along the grid's axes (city-block distance), so the
/// region is found by one breadth-first walk out from every seed at once,
/// stopped at `k` steps, rather than by `k` passes over the whole grid.
pub fn remove_outside(
    labels: &mut [u8],
    mask: &[u8],
    dims: [usize; 3],
    ids: &[u8],
    iterations: usize,
) {
    let [nx, ny, nz] = dims;
    let mut want = [false; 256];
    for &i in ids {
        want[i as usize] = true;
    }
    // Steps from the nearest seed, `u16::MAX` for unreached.
    let far = u16::MAX;
    let limit = iterations.min(far as usize - 1) as u16;
    let mut dist: Vec<u16> = mask
        .iter()
        .map(|&l| if want[l as usize] { 0 } else { far })
        .collect();
    // The walk starts from the seeds' edge only: a seed whose six
    // neighbours are seeds too reaches nothing the others do not. Indices
    // as u32, the frontier being the one list as long as a surface.
    assert!(
        dist.len() <= u32::MAX as usize,
        "a grid of more than 2^32 voxels"
    );
    let plane = nx * ny;
    let is_seed = |u: usize| dist[u] == 0;
    let edge = |v: usize| {
        let (x, y, z) = (v % nx, (v / nx) % ny, v / plane);
        (x > 0 && !is_seed(v - 1))
            || (x + 1 < nx && !is_seed(v + 1))
            || (y > 0 && !is_seed(v - nx))
            || (y + 1 < ny && !is_seed(v + nx))
            || (z > 0 && !is_seed(v - plane))
            || (z + 1 < nz && !is_seed(v + plane))
    };
    let mut front: Vec<u32> = (0..dist.len())
        .filter(|&v| dist[v] == 0 && edge(v))
        .map(|v| v as u32)
        .collect();
    let mut next: Vec<u32> = Vec::new();
    for step in 1..=limit {
        if front.is_empty() {
            break;
        }
        for &v in &front {
            let v = v as usize;
            let (x, y, z) = (v % nx, (v / nx) % ny, v / plane);
            let mut visit = |u: usize| {
                if dist[u] == far {
                    dist[u] = step;
                    next.push(u as u32);
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
        std::mem::swap(&mut front, &mut next);
        next.clear();
    }
    for (l, &d) in labels.iter_mut().zip(&dist) {
        if d == far {
            *l = 0;
        }
    }
}

/// The sliding window's view of one part: its networks (the folds, summed)
/// and the progress row.
struct Hooks<'a> {
    nets: &'a PartNets,
    patch: [usize; 3],
    /// Activation buffers reused from patch to patch (CPU); the forward
    /// pass runs on one thread at a time, so the lock is never contended.
    arena: Mutex<cpu::Arena>,
    progress: &'a Progress,
    /// (model index, model count) for progress text.
    model: (usize, usize),
    label: &'a str,
}

impl Hooks<'_> {
    /// The folds' logits summed into `sum` (`[classes, p0, p1, p2]`), fold 0
    /// first and each later fold added in turn, as an ensemble of the CPU
    /// networks always summed them.
    fn cpu_sum(&self, nets: &[net::UNet], patch: &[f32], arena: &mut cpu::Arena) -> Vec<f32> {
        let [p0, p1, p2] = self.patch;
        let mut x = arena.act(1, p0, p1, p2);
        x.data.copy_from_slice(patch);
        let mut sum = nets[0].forward_cpu_in(&x, arena).data;
        for n in &nets[1..] {
            let more = n.forward_cpu_in(&x, arena);
            sum.iter_mut().zip(&more.data).for_each(|(a, b)| *a += b);
            arena.give(more.data);
        }
        arena.give(x.data);
        sum
    }
}

impl infer::InferHooks for Hooks<'_> {
    fn forward(&self, patch: &[f32]) -> Result<Vec<f32>> {
        let mut out = None;
        self.forward_classes(patch, &mut |_, piece| {
            out = Some(piece.into_owned());
            Ok(())
        })?;
        out.context("the network returned nothing")
    }

    fn forward_classes(&self, patch: &[f32], sink: &mut infer::ClassSink<'_>) -> Result<()> {
        match &self.nets.nets {
            Nets::Cpu(nets) => {
                let mut arena = self.arena.lock().unwrap_or_else(|p| p.into_inner());
                let sum = self.cpu_sum(nets, patch, &mut arena);
                let res = sink(0, Cow::Borrowed(&sum));
                // The logits go back to the arena for the next patch.
                arena.give(sum);
                res
            }
            #[cfg(feature = "gpu")]
            Nets::Gpu(gnets) => {
                let p = self.patch;
                if let [g] = gnets.as_slice() {
                    // One network: each piece of classes goes straight on.
                    return g
                        .forward_classes(patch, p, &mut |c0, piece| sink(c0, Cow::Owned(piece)));
                }
                // Folds: summed here, piece by piece, fold 0 first, as the
                // whole logits were.
                let n = p[0] * p[1] * p[2];
                let mut sum = vec![0f32; self.nets.classes * n];
                for (fi, g) in gnets.iter().enumerate() {
                    g.forward_classes(patch, p, &mut |c0, piece| {
                        let dst = &mut sum[c0 * n..c0 * n + piece.len()];
                        if fi == 0 {
                            dst.copy_from_slice(&piece);
                        } else {
                            dst.iter_mut().zip(&piece).for_each(|(a, b)| *a += b);
                        }
                        Ok(())
                    })?;
                }
                sink(0, Cow::Owned(sum))
            }
        }
    }

    /// One network on the GPU only: its logits go on in class pieces, so a
    /// tile computed ahead costs a piece of memory. Folds are summed into
    /// whole logits first, and a tile ahead would hold several of those.
    fn overlaps(&self) -> bool {
        #[cfg(feature = "gpu")]
        if matches!(&self.nets.nets, Nets::Gpu(g) if g.len() == 1) {
            return true;
        }
        false
    }

    fn tile_done(&self, done: usize, total: usize) -> bool {
        let (mi, mn) = self.model;
        let text = if mn > 1 {
            format!(
                "Segmenting ({}), model {}/{}: tile {done}/{total}",
                self.label,
                mi + 1,
                mn
            )
        } else {
            format!("Segmenting ({}): tile {done}/{total}", self.label)
        };
        self.progress.report(done as f32 / total as f32, &text);
        !self.progress.cancelled()
    }
}

/// The networks of one part on the chosen device, built from the folds in
/// the cache (or the ones kept from an earlier run, see [`kept`]).
fn part_nets(
    part: &Part,
    dir: &Path,
    max_folds: u8,
    config: &super::config::ModelConfig,
    gpu: Option<&crate::nn::device::GpuContext>,
    device_desc: &str,
    progress: &Progress,
) -> Result<std::sync::Arc<PartNets>> {
    let key = NetKey {
        spec: part.spec.key,
        dir: dir.to_path_buf(),
        folds: weights::folds_loaded(&part.spec, max_folds),
        device: device_desc.to_string(),
    };
    kept::get_or_build(&key, || {
        let folds = weights::load_folds(&part.spec, dir, max_folds, progress)?;
        if progress.cancelled() {
            bail!(CANCELLED);
        }
        let bytes = folds
            .iter()
            .flat_map(|f| f.values())
            .map(|t| t.data.len() * std::mem::size_of::<f32>())
            .sum();
        // Each fold's tensors move into its network: nothing is copied, and
        // the map is gone once the network stands.
        let nets: Vec<net::UNet> = folds
            .into_iter()
            .map(|t| net::UNet::build_owned(config.clone(), t))
            .collect::<Result<_>>()
            .with_context(|| format!("assemble network ({})", part.spec.label))?;
        let classes = nets[0].num_classes();
        let nets = match gpu {
            None => Nets::Cpu(nets),
            #[cfg(feature = "gpu")]
            Some(ctx) => {
                // Uploaded one at a time, each host copy dropped as soon as
                // its network is on the device.
                let mut g = Vec::with_capacity(nets.len());
                for n in nets {
                    g.push(super::gpu::GpuNet::new(ctx, &n)?);
                }
                Nets::Gpu(g)
            }
            #[cfg(not(feature = "gpu"))]
            Some(ctx) => ctx.unreachable(),
        };
        Ok(PartNets {
            classes,
            nets,
            bytes,
        })
    })
}

/// The networks of a task, merged, on the volume's own grid.
///
/// Every part is downloaded and converted first, so a missing file stops
/// the run before any window has run; then each part's weights are loaded
/// just before its windows and dropped after them (unless they are being
/// kept, see [`kept`]), so a model in parts holds one part at a time.
fn run_parts(
    volume: &Volume,
    task: &NnTask,
    parts: &[Part],
    device: DevicePref,
    root: &Path,
    window: (f32, f32),
    progress: &Progress,
) -> Result<(Vec<u8>, String)> {
    let n_models = parts.len();
    let (base, span) = window;
    let phase = |p: &Progress, at: f32, len: f32| p.set_phase(base + span * at, span * len);

    // Progress budget: 15% download/convert, 5% preprocess, 75% loading
    // and inference, 5% postprocess.
    let dl_span = 0.15 / n_models as f32;
    let max_folds = match task.folds {
        FoldUse::All => u8::MAX,
        FoldUse::First => 1,
    };
    let mut configs = Vec::with_capacity(n_models);
    for (i, part) in parts.iter().enumerate() {
        phase(progress, i as f32 * dl_span, dl_span);
        let dir = root.join(part.spec.home.subdir());
        let config = weights::prepare_model(&part.spec, &dir, progress)?;
        if progress.cancelled() {
            bail!(CANCELLED);
        }
        configs.push(config);
    }
    let spacing = configs[0].spacing;
    let axes = parts[0].spec.axes;
    for (c, part) in configs.iter().zip(parts) {
        if c.spacing != spacing || part.spec.axes != axes {
            bail!("the sub-models of '{}' disagree on their grid", task.label);
        }
    }

    progress.set("Choosing the compute device");
    let gpu = device.resolve()?;
    let device_desc = gpu
        .as_ref()
        .map(|ctx| ctx.describe())
        .unwrap_or_else(crate::nn::device::describe_cpu);
    progress.set_device(&device_desc);

    phase(progress, 0.15, 0.05);
    progress.report(
        0.0,
        &format!(
            "Resampling volume to {} mm",
            if spacing[0] == spacing[1] && spacing[1] == spacing[2] {
                format!("{}", spacing[0])
            } else {
                format!("{:.3} × {:.3} × {:.3}", spacing[0], spacing[1], spacing[2])
            }
        ),
    );
    let map = preprocess::SarMap::with_axes(volume, spacing, axes);
    let vol_model = preprocess::resample_to_model(volume, &map);
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    let mut global = vec![0u8; vol_model.len()];
    let infer_span = 0.75 / n_models as f32;
    for (mi, (config, part)) in configs.iter().zip(parts).enumerate() {
        phase(progress, 0.2 + mi as f32 * infer_span, infer_span);
        let dir = root.join(part.spec.home.subdir());
        let nets = part_nets(
            part,
            &dir,
            max_folds,
            config,
            gpu.as_ref(),
            &device_desc,
            progress,
        )?;
        // A z-score model normalizes against this image, so its constants
        // are only knowable now, with the resampled volume in hand; the
        // networks themselves do not depend on them.
        let mut cfg = config.clone();
        cfg.apply_image_norm(&vol_model);
        let hooks = Hooks {
            nets: &nets,
            patch: cfg.patch_size,
            arena: Mutex::new(cpu::Arena::default()),
            progress,
            model: (mi, n_models),
            label: task.label,
        };
        let local = infer::predict(
            &vol_model,
            map.model_dims,
            nets.classes,
            &cfg,
            task.step,
            &hooks,
        )
        .with_context(|| format!("inference ({})", part.spec.label))?;
        drop(hooks);
        drop(nets);
        // Merge: the network's labels through its table; later parts win.
        for (g, l) in global.iter_mut().zip(local.iter()) {
            if *l != 0 {
                if let Some(&t) = part.lut.get(*l as usize) {
                    if t != 0 {
                        *g = t;
                    }
                }
            }
        }
    }

    phase(progress, 0.95, 0.05);
    match task.post {
        Post::None | Post::NnUnetV1 | Post::RemoveOutside { .. } => {}
        Post::Body => {
            progress.report(0.0, "Cleaning up the outline");
            post::body(&mut global, map.model_dims, spacing, task.classes);
        }
        Post::VertebraePp => {
            progress.report(0.0, "Numbering the vertebrae");
            let s_axis = axes.targets().iter().position(|t| t.z > 0.5).unwrap_or(0);
            post::vertebrae_pp(&mut global, map.model_dims, spacing, s_axis, task.classes);
        }
    }
    progress.report(0.0, "Mapping labels back to the scan grid");
    let mut labels = preprocess::labels_to_volume_grid(&global, &map, volume);
    if task.post == Post::NnUnetV1 {
        let dir = weights::spec_dir(root, &parts[0].spec);
        let rules = super::v1::load_post(&dir)?;
        if !rules.is_empty() {
            progress.report(0.0, "Keeping the largest connected pieces");
            super::v1::apply_post(&mut labels, volume.dims, volume.spacing, &rules);
        }
    }
    Ok((labels, device_desc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_task_is_well_formed() {
        let tasks = all_tasks();
        let mut keys: Vec<&str> = tasks.iter().map(|t| t.key).collect();
        keys.sort();
        let n = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), n, "duplicate task key");
        for t in &tasks {
            assert!(!t.parts.is_empty(), "{}", t.key);
            assert!(!t.classes.is_empty() && t.classes.len() < 256, "{}", t.key);
            assert!(t.step > 0.0 && t.step <= 1.0, "{}", t.key);
            // Every label a part can produce is one of the task's classes.
            for p in t.parts {
                for &l in p.lut {
                    assert!(
                        (l as usize) <= t.classes.len() || p.lut.len() == 256,
                        "{}",
                        t.key
                    );
                }
            }
            // A crop names classes its crop model has.
            if let Some(c) = t.crop {
                let by =
                    c.by.task()
                        .unwrap_or_else(|| panic!("{}: crop model", t.key));
                assert!(
                    by.crop.is_none() || by.key == "craniofacial_structures",
                    "{}",
                    t.key
                );
                for name in c.classes {
                    assert!(
                        by.label_of(name).is_some(),
                        "{}: {name} not in {}",
                        t.key,
                        by.key
                    );
                }
            }
            // The parts of one task share a grid and an axis order.
            assert!(t.parts.iter().all(|p| p.spec.axes == t.parts[0].spec.axes));
        }
        assert!(task_by_key("lung_vessels").is_some());
        assert!(task_by_key("nope").is_none());
    }

    #[test]
    fn the_crop_box_is_widened_in_whole_voxels_and_clamped() {
        // 10 x 10 x 10 voxels of 2 x 2 x 5 mm; label 3 at (4..6, 5, 2).
        let dims = [10, 10, 10];
        let mut l = vec![0u8; 1000];
        for i in 4..6 {
            l[2 * 100 + 5 * 10 + i] = 3;
        }
        l[999] = 7;
        // 9 mm: 4 voxels in x and y, 1 in z.
        let (lo, hi) = crop_box(&l, dims, [2.0, 2.0, 5.0], &[3], 9.0).unwrap();
        assert_eq!((lo, hi), ([0, 1, 1], [10, 10, 4]));
        assert!(crop_box(&l, dims, [2.0, 2.0, 5.0], &[4], 9.0).is_none());
        let (lo, hi) = crop_box(&l, dims, [1.0; 3], &[3, 7], 0.0).unwrap();
        assert_eq!((lo, hi), ([4, 5, 2], [10, 10, 10]));
    }

    #[test]
    fn labels_outside_the_dilated_mask_are_cleared() {
        // 5 x 1 x 1: the mask has class 2 at x = 2; one dilation keeps 1..=3.
        let mut l = vec![7u8; 5];
        remove_outside(&mut l, &[0, 0, 2, 0, 9], [5, 1, 1], &[2], 1);
        assert_eq!(l, [0, 7, 7, 7, 0]);
        let mut l = vec![7u8; 5];
        remove_outside(&mut l, &[0, 0, 2, 0, 9], [5, 1, 1], &[2], 0);
        assert_eq!(l, [0, 0, 7, 0, 0]);
        // Diagonals are two steps away.
        let mut l = vec![1u8; 9];
        let mut m = vec![0u8; 9];
        m[4] = 1;
        remove_outside(&mut l, &m, [3, 3, 1], &[1], 1);
        assert_eq!(l, [0, 1, 0, 1, 1, 1, 0, 1, 0]);
    }

    /// The pass-by-pass dilation `remove_outside` used to run.
    fn remove_outside_by_passes(
        labels: &mut [u8],
        mask: &[u8],
        dims: [usize; 3],
        ids: &[u8],
        iterations: usize,
    ) {
        let [nx, ny, nz] = dims;
        let mut inside: Vec<bool> = mask.iter().map(|l| ids.contains(l)).collect();
        let mut next = inside.clone();
        for _ in 0..iterations {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let v = x + nx * (y + ny * z);
                        if inside[v] {
                            continue;
                        }
                        next[v] = (x > 0 && inside[v - 1])
                            || (x + 1 < nx && inside[v + 1])
                            || (y > 0 && inside[v - nx])
                            || (y + 1 < ny && inside[v + nx])
                            || (z > 0 && inside[v - nx * ny])
                            || (z + 1 < nz && inside[v + nx * ny]);
                    }
                }
            }
            std::mem::swap(&mut inside, &mut next);
            next.copy_from_slice(&inside);
        }
        for (l, &keep) in labels.iter_mut().zip(&inside) {
            if !keep {
                *l = 0;
            }
        }
    }

    #[test]
    fn the_breadth_first_walk_keeps_what_the_passes_kept() {
        let dims = [13, 9, 7];
        let n = dims[0] * dims[1] * dims[2];
        let mut seed = 0x9E37_79B9u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for trial in 0..6 {
            // Sparse seeds of two kinds and other labels around them.
            let mask: Vec<u8> = (0..n)
                .map(|_| match next() % 97 {
                    0 => 3,
                    1 => 5,
                    2..=5 => 7,
                    _ => 0,
                })
                .collect();
            let labels: Vec<u8> = (0..n).map(|_| (next() % 4) as u8 + 1).collect();
            for iterations in [0, 1, 2, 5, 40] {
                let mut a = labels.clone();
                let mut b = labels.clone();
                remove_outside(&mut a, &mask, dims, &[3, 5], iterations);
                remove_outside_by_passes(&mut b, &mask, dims, &[3, 5], iterations);
                assert_eq!(a, b, "trial {trial}, {iterations} iterations");
            }
        }
    }

    #[test]
    fn shifted_tables_place_a_part_in_the_task() {
        let t: [u8; 4] = shifted(24);
        assert_eq!(t, [0, 25, 26, 27]);
        assert_eq!(IDENTITY_LUT[117], 117);
    }
}
