//! Wall-clock time per stage of an nnU-Net run, for measuring where a run
//! spends its time without a profiler: set `RDS_AUTOSEG_TIMES=1` and every
//! run of [`super::task::run_task_labels`] prints one line per stage to
//! stderr when it ends (`seg_cli` is the usual way in).
//!
//! The stages are those of the pipeline in [`super`]'s module docs: the
//! download and conversion check, the device, the resampling, the weights
//! (read from the safetensors cache, assembled, uploaded), the
//! normalization, and inside the sliding window the patch extraction, the
//! network (on the GPU this includes the readback of the logits, which is
//! where the device's work is waited for), the weighted accumulation, the
//! argmax, and, when the network runs on a thread of its own, the time the
//! accumulating thread waited for it; then the merge of the parts, the
//! post-processing and the map back to the scan. The network's time is
//! split further: on the CPU into the convolutions, the norms with their
//! activations, the upsampling with its concatenation, and the head; on the
//! GPU into the encoder and decoder (waited for by reading one value back,
//! a wait made only while timing, so a run without it pays nothing) and the
//! head with its readback.
//!
//! The counters are process-wide sums of nanoseconds (relaxed atomics, a
//! clock read per tile and per accumulator row: nothing that shows in the
//! timings themselves). Two runs at once share them, which only a
//! measurement would notice. A crop cascade's coarse run reports first, as
//! a run of its own. Nothing here touches what a run computes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// One stage of a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Prepare,
    Device,
    Resample,
    Weights,
    Normalize,
    Extract,
    Forward,
    GpuFeatures,
    GpuHeads,
    CpuConv,
    CpuNorm,
    CpuUp,
    CpuHead,
    Accumulate,
    Argmax,
    Wait,
    Merge,
    Post,
    BackMap,
}

impl Stage {
    const ALL: [Stage; 19] = [
        Stage::Prepare,
        Stage::Device,
        Stage::Resample,
        Stage::Weights,
        Stage::Normalize,
        Stage::Extract,
        Stage::Forward,
        Stage::GpuFeatures,
        Stage::GpuHeads,
        Stage::CpuConv,
        Stage::CpuNorm,
        Stage::CpuUp,
        Stage::CpuHead,
        Stage::Accumulate,
        Stage::Argmax,
        Stage::Wait,
        Stage::Merge,
        Stage::Post,
        Stage::BackMap,
    ];

    fn name(self) -> &'static str {
        match self {
            Stage::Prepare => "download / convert check",
            Stage::Device => "device",
            Stage::Resample => "resample to model grid",
            Stage::Weights => "weights (cache read, build, upload)",
            Stage::Normalize => "normalize",
            Stage::Extract => "window: extract patches",
            Stage::Forward => "window: network (+ GPU readback)",
            Stage::GpuFeatures => "  of which GPU: encoder + decoder",
            Stage::GpuHeads => "  of which GPU: head + readback",
            Stage::CpuConv => "  of which CPU: 3-D convolutions",
            Stage::CpuNorm => "  of which CPU: norms, activations",
            Stage::CpuUp => "  of which CPU: upsampling, concat",
            Stage::CpuHead => "  of which CPU: head",
            Stage::Accumulate => "window: weighted accumulation",
            Stage::Argmax => "window: argmax",
            Stage::Wait => "window: waiting for the network",
            Stage::Merge => "merge parts",
            Stage::Post => "post-processing",
            Stage::BackMap => "map back to the scan grid",
        }
    }
}

static NS: [AtomicU64; 19] = [const { AtomicU64::new(0) }; 19];
static TILES: AtomicU64 = AtomicU64::new(0);

/// Whether `RDS_AUTOSEG_TIMES` is set (read once).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("RDS_AUTOSEG_TIMES").is_some())
}

/// Add the time since `since` to `stage`.
pub fn add(stage: Stage, since: Instant) {
    add_ns(stage, since.elapsed().as_nanos() as u64);
}

/// Add `ns` nanoseconds to `stage`.
pub fn add_ns(stage: Stage, ns: u64) {
    if enabled() {
        NS[stage as usize].fetch_add(ns, Ordering::Relaxed);
    }
}

/// Count one window.
pub fn tile() {
    if enabled() {
        TILES.fetch_add(1, Ordering::Relaxed);
    }
}

/// Print the stages that took any time, with `what` and the whole run's
/// time, and start the counters again.
pub fn report(what: &str, total: Instant) {
    if !enabled() {
        return;
    }
    let tiles = TILES.swap(0, Ordering::Relaxed);
    eprintln!(
        "[times] {what}: {:.2} s in all, {tiles} windows",
        total.elapsed().as_secs_f64()
    );
    for s in Stage::ALL {
        let ns = NS[s as usize].swap(0, Ordering::Relaxed);
        if ns > 0 {
            eprintln!("[times]   {:38} {:9.3} s", s.name(), ns as f64 * 1e-9);
        }
    }
}
