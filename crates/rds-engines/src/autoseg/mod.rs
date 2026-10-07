//! Automatic segmentation with nnU-Net models - a pure-Rust re-implementation
//! of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
//! inference (Wasserthal et al., Radiology AI 2023, doi 10.1148/ryai.230024)
//! and of the nnU-Net v2 predictor it drives, which also runs
//! [MRSegmentator](https://github.com/hhaentze/MRSegmentator).
//!
//! What runs is a *task* ([`task::NnTask`]): TotalSegmentator's 117-class
//! `total` in both generations of its weights (v2 of 2023, its default, and
//! v3, with a plain and a residual-encoder network), its MR model, the body
//! outline, MRSegmentator, and every openly licensed task of upstream's
//! catalogue ([`tasks`]). Weights are downloaded from the official GitHub
//! releases on first use, converted natively (no Python) and cached.
//! Inference runs either on the CPU (rayon + SIMD GEMM) or on any GPU
//! through wgpu (Vulkan / DX12 / Metal - no CUDA toolkit required) when the
//! `gpu` feature is enabled.
//!
//! Pipeline (mirroring upstream): optionally run a coarse model and crop to
//! the classes the task needs → reorient to the model's axes ([S,A,R] for
//! TotalSegmentator, [S,P,L] for MRSegmentator) → resample to the model's
//! spacing (trilinear) → normalize (fixed CT window, or the image's own
//! z-score for MR) → sliding-window inference, Gaussian-weighted, no
//! mirroring TTA, folds summed → argmax → merge sub-model labels through
//! their tables → task post-processing → nearest-neighbour map back to the
//! scan grid.

pub mod classes;
pub mod config;
pub mod cpu;
pub mod custom;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod infer;
pub mod net;
pub mod post;
pub mod preprocess;
pub mod task;
pub mod tasks;
pub mod total;
pub mod v1;
pub mod weights;

use anyhow::Result;
use std::path::Path;

use crate::progress::{Progress, ProgressSink};
use crate::volume::Volume;
pub use task::{
    all_tasks, download_needed, run_task_labels, task_by_key, NnOptions, NnTask, TaskLabels,
};

/// Which TotalSegmentator model set to run.
///
/// The v2 variants are the weights TotalSegmentator itself still runs by
/// default (task `total`); the v3 variants are its `total_v3` task, whose
/// organs, cardiac and muscles parts were retrained on 1830 subjects and
/// which comes as a plain network (`big`) and a residual-encoder one
/// (`small`). The two generations share the class table but for label 26
/// ([`classes::TOTAL_V3_CLASS_NAMES`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// Single 3 mm model, all 117 classes (`--fast`). ~135 MB download.
    Fast3mm,
    /// Five 1.5 mm sub-models (organs / vertebrae / cardiac / muscles /
    /// ribs), best quality. ~1.2 GB download.
    HighRes15mm,
    /// Single 6 mm model (`--fastest`) - quick preview quality.
    Preview6mm,
    /// v3, plain network, one 3 mm model. ~242 MB download (the zip also
    /// holds the `small` training).
    V3Fast3mm,
    /// v3, plain network, the five 1.5 mm sub-models. ~1.8 GB download.
    V3HighRes15mm,
    /// v3, plain network, one 6 mm model. ~132 MB download.
    V3Preview6mm,
    /// v3, residual encoder (`small`), one 3 mm model; a 128-cubed patch.
    V3Small3mm,
    /// v3, residual encoder (`small`), the five 1.5 mm sub-models; a
    /// 192-cubed patch, GPU recommended.
    V3Small15mm,
}

/// The two generations of the `total` task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Generation {
    V2,
    V3,
}

impl Variant {
    pub const ALL: [Variant; 8] = [
        Variant::Fast3mm,
        Variant::HighRes15mm,
        Variant::Preview6mm,
        Variant::V3Fast3mm,
        Variant::V3HighRes15mm,
        Variant::V3Preview6mm,
        Variant::V3Small3mm,
        Variant::V3Small15mm,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Variant::Fast3mm => "3 mm (fast)",
            Variant::HighRes15mm => "1.5 mm (high quality)",
            Variant::Preview6mm => "6 mm (preview)",
            Variant::V3Fast3mm => "v3 3 mm (fast)",
            Variant::V3HighRes15mm => "v3 1.5 mm (high quality)",
            Variant::V3Preview6mm => "v3 6 mm (preview)",
            Variant::V3Small3mm => "v3 small 3 mm",
            Variant::V3Small15mm => "v3 small 1.5 mm",
        }
    }

    /// The short name the command-line tools accept.
    pub fn key(&self) -> &'static str {
        match self {
            Variant::Fast3mm => "fast3",
            Variant::HighRes15mm => "highres",
            Variant::Preview6mm => "preview6",
            Variant::V3Fast3mm => "fast3-v3",
            Variant::V3HighRes15mm => "highres-v3",
            Variant::V3Preview6mm => "preview6-v3",
            Variant::V3Small3mm => "small3-v3",
            Variant::V3Small15mm => "small15-v3",
        }
    }

    /// The variant a command-line name refers to, if any.
    pub fn from_key(key: &str) -> Option<Variant> {
        Variant::ALL.into_iter().find(|v| v.key() == key)
    }

    pub fn generation(&self) -> Generation {
        match self {
            Variant::Fast3mm | Variant::HighRes15mm | Variant::Preview6mm => Generation::V2,
            _ => Generation::V3,
        }
    }

    /// True for the 1.5 mm variants, whose five sub-models the `parts`
    /// mask of [`run`] selects.
    pub fn has_parts(&self) -> bool {
        matches!(
            self,
            Variant::HighRes15mm | Variant::V3HighRes15mm | Variant::V3Small15mm
        )
    }

    /// The class table this variant's labels index.
    pub fn class_names(&self) -> &'static [&'static str; 117] {
        match self.generation() {
            Generation::V2 => &classes::TOTAL_CLASS_NAMES,
            Generation::V3 => &classes::TOTAL_V3_CLASS_NAMES,
        }
    }

    /// Name of a global label (1-based) under this variant.
    pub fn class_name(&self, label: u8) -> &'static str {
        classes::class_name_in(self.class_names(), label)
    }

    /// The task this variant is.
    pub fn task(&self) -> &'static NnTask {
        match self {
            Variant::Fast3mm => &total::TOTAL_FAST,
            Variant::HighRes15mm => &total::TOTAL,
            Variant::Preview6mm => &total::TOTAL_FASTEST,
            Variant::V3Fast3mm => &total::TOTAL_V3_FAST,
            Variant::V3HighRes15mm => &total::TOTAL_V3,
            Variant::V3Preview6mm => &total::TOTAL_V3_FASTEST,
            Variant::V3Small3mm => &total::TOTAL_V3_SMALL_FAST,
            Variant::V3Small15mm => &total::TOTAL_V3_SMALL,
        }
    }

    /// The variant a task is, if it is one of the `total` CT tasks.
    pub fn of_task(task: &NnTask) -> Option<Variant> {
        Variant::ALL.into_iter().find(|v| v.task().key == task.key)
    }
}

pub use crate::nn::device::DevicePref;

/// One detected organ in the result.
#[derive(Clone, Debug)]
pub struct OrganHit {
    /// Global TotalSegmentator class id (1..=117).
    pub label: u8,
    /// The class name under the variant that ran ([`Variant::class_name`]).
    pub name: &'static str,
    /// Voxel count on the original CT grid.
    pub voxels: u64,
    /// Volume in cm³ on the original CT grid.
    pub cm3: f64,
    pub color: [u8; 3],
}

/// Output of a segmentation run, whichever family of model ran.
pub struct AutosegResult {
    /// Class labels per voxel (indices into [`AutosegResult::classes`],
    /// 1-based), `Volume::data` index order.
    pub labels: Vec<u8>,
    pub dims: [usize; 3],
    /// Classes present, sorted by voxel count descending.
    pub organs: Vec<OrganHit>,
    /// Registry key of the model that ran (`zoo::AutoModel::key`).
    pub model: String,
    /// The model's name as the interface shows it.
    pub model_label: String,
    /// The class table the labels index.
    pub classes: &'static [&'static str],
    /// Human-readable device description ("CPU (16 threads)", "GPU (wgpu)").
    pub device: String,
    pub elapsed_secs: f64,
    /// Identity of the volume this was computed on.
    pub frame_of_reference_uid: String,
    pub volume_dims: [usize; 3],
    /// What the run has to say beyond its labels (a crop that found
    /// nothing, a model used outside its modality).
    pub notes: Vec<String>,
}

impl AutosegResult {
    /// Name of label `l`; empty for 0 or out of range.
    pub fn class_name(&self, l: u8) -> &'static str {
        if l == 0 {
            ""
        } else {
            self.classes.get(l as usize - 1).copied().unwrap_or("")
        }
    }
}

/// The classes present in `labels`, largest first, with their volumes on a
/// grid of `spacing` and their display colours.
pub fn organ_hits(
    labels: &[u8],
    spacing: [f64; 3],
    classes: &'static [&'static str],
) -> Vec<OrganHit> {
    let mut counts = [0u64; 256];
    for l in labels {
        counts[*l as usize] += 1;
    }
    let voxel_cm3 = spacing[0] * spacing[1] * spacing[2] / 1000.0;
    let mut organs: Vec<OrganHit> = (1..=classes.len().min(255))
        .filter(|l| counts[*l] > 0)
        .map(|l| OrganHit {
            label: l as u8,
            name: classes[l - 1],
            voxels: counts[l],
            cm3: counts[l] as f64 * voxel_cm3,
            color: classes::color_of(classes[l - 1], l as u8),
        })
        .collect();
    organs.sort_by_key(|o| std::cmp::Reverse(o.voxels));
    organs
}

/// Run an nnU-Net task on a volume. Blocking - call from a worker thread;
/// observe/cancel through `progress`. `root` is the model folder (the
/// engine folders are below it).
pub fn run(
    volume: &Volume,
    task: &'static NnTask,
    opts: &NnOptions,
    root: &Path,
    progress: &Progress,
) -> Result<AutosegResult> {
    let t_start = std::time::Instant::now();
    let out = run_task_labels(volume, task, opts, root, (0.0, 1.0), progress)?;
    let organs = organ_hits(&out.labels, volume.spacing, task.classes);
    progress.report(1.0, "Auto-segmentation finished");
    Ok(AutosegResult {
        labels: out.labels,
        dims: volume.dims,
        organs,
        model: task.key.to_string(),
        model_label: task.label.to_string(),
        classes: task.classes,
        device: out.device,
        elapsed_secs: t_start.elapsed().as_secs_f64(),
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
        volume_dims: volume.dims,
        notes: out.notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_have_distinct_keys_tasks_and_the_right_tables() {
        let mut keys: Vec<&str> = Variant::ALL.iter().map(|v| v.key()).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), Variant::ALL.len());
        for v in Variant::ALL {
            assert_eq!(Variant::from_key(v.key()), Some(v));
            assert_eq!(Variant::of_task(v.task()), Some(v));
            let t = v.task();
            assert_eq!(t.parts.len(), if v.has_parts() { 5 } else { 1 }, "{v:?}");
            assert_eq!(t.classes.len(), 117);
            assert_eq!(t.classes, v.class_names().as_slice());
            // Every part of a variant comes from its own generation's release.
            for p in t.parts {
                let v3 = p.spec.url.contains("v3.0.0-weights");
                assert_eq!(
                    v3,
                    v.generation() == Generation::V3,
                    "{} under {v:?}",
                    p.spec.key
                );
                let resenc = p.spec.plans != "nnUNetPlans";
                assert_eq!(
                    resenc,
                    matches!(v, Variant::V3Small3mm | Variant::V3Small15mm),
                    "{}",
                    p.spec.key
                );
            }
        }
        assert_eq!(Variant::Fast3mm.class_name(26), "vertebrae_S1");
        assert_eq!(Variant::V3Small3mm.class_name(26), "vertebrae_L6");
        assert_eq!(Variant::V3Fast3mm.class_name(5), "liver");
        assert_eq!(Variant::from_key("v4"), None);
        assert_eq!(Variant::of_task(&total::TOTAL_MR), None);
    }

    #[test]
    fn hits_are_named_from_the_table_and_sorted_by_size() {
        static T: [&str; 3] = ["liver", "spleen", "unknown_thing"];
        let labels = [0u8, 1, 1, 1, 2, 3, 3, 9];
        let h = organ_hits(&labels, [10.0, 10.0, 10.0], &T);
        let names: Vec<&str> = h.iter().map(|o| o.name).collect();
        assert_eq!(names, ["liver", "unknown_thing", "spleen"]);
        assert_eq!(h[0].voxels, 3);
        assert!((h[0].cm3 - 3.0).abs() < 1e-9);
        assert_eq!(h[0].color, classes::color_of("liver", 1));
    }
}
