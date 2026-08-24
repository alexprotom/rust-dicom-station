//! Prompt-driven segmentation: the SegVol engine's user interface.
//!
//! Where auto-segmentation runs to completion on its own, this one is a
//! conversation — the user points at something and the network answers, and
//! the answer lands as an ordinary [`Segmentation`], editable with the brush
//! and convertible to RTSTRUCT like any other.
//!
//! The prompt is anchored on the crosshair. A box is the crosshair plus an
//! extent in millimetres, a point is the crosshair itself, and text is a
//! structure name run through the trained template. That keeps the whole
//! interaction inside one dialog: the natural next step is dragging the box
//! directly in the viewports, which needs interaction state in
//! [`super::views`] rather than anything here.
//!
//! Coordinates cross one conversion before reaching the network. The
//! crosshair is fractional voxel indices in the slot's own volume;
//! [`preprocess::prepare`] reorients that volume to canonical `[S, A, R]` and
//! crops it, and the network sees indices in the prepared grid.
//! [`prompt_from_crosshair`] does that mapping and is the only place it
//! happens.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use crate::segvol::infer::{self, Config, Hooks};
use crate::segvol::params::Params;
use crate::segvol::preprocess::{self, Prepared};
use crate::segvol::prompt::{BBox, Point};
use crate::segvol::{bpe::Bpe, clip::TextEncoder, net::SegVolNet, weights};

use super::*;

/// Which kind of prompt the dialog is collecting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PromptKind {
    Box,
    Point,
    Text,
}

/// The dialog's state.
pub(super) struct SegVolDialog {
    pub slot: usize,
    pub kind: PromptKind,
    /// Half-extent of the box prompt, in millimetres.
    pub extent_mm: f32,
    /// Structure name for a text prompt.
    pub text: String,
    pub cfg: Config,
    pub name: String,
}

/// Progress handle shared with the worker: message, fraction, cancel flag,
/// and — once resolved — which device the image encoder runs on.
#[derive(Default)]
pub struct SegVolProgress {
    msg: Mutex<String>,
    device: Mutex<String>,
    frac: AtomicU32,
    cancel: AtomicBool,
}

impl SegVolProgress {
    pub fn set(&self, m: impl Into<String>) {
        *self.msg.lock().unwrap_or_else(|e| e.into_inner()) = m.into();
    }
    pub fn get(&self) -> String {
        self.msg.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
    pub fn set_device(&self, d: impl Into<String>) {
        *self.device.lock().unwrap_or_else(|e| e.into_inner()) = d.into();
    }
    pub fn device(&self) -> String {
        self.device
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
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
}

impl Hooks for SegVolProgress {
    fn report(&self, frac: f32, msg: &str) {
        self.frac
            .store(frac.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.set(msg);
    }
    fn cancelled(&self) -> bool {
        SegVolProgress::cancelled(self)
    }
}

impl crate::nn::cache::ProgressSink for SegVolProgress {
    fn report(&self, frac: f32, msg: &str) {
        self.frac
            .store(frac.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        self.set(msg);
    }
    fn cancelled(&self) -> bool {
        SegVolProgress::cancelled(self)
    }
}

/// What a finished run hands back.
pub struct SegVolResult {
    pub mask: Vec<u8>,
    pub name: String,
    pub voxels: u64,
    pub windows: usize,
    pub coarse: bool,
    pub elapsed_secs: f64,
    /// Which device the image encoder ran on, e.g. "GPU (wgpu)".
    pub device: String,
    pub volume_dims: [usize; 3],
    pub frame_of_reference_uid: String,
}

/// Convert the crosshair — fractional voxel indices in the slot's volume —
/// into prompt coordinates on the prepared grid.
///
/// Returns `None` when the crosshair falls outside the prepared volume's
/// foreground crop, which is the honest answer: there is nothing there to
/// prompt with.
pub(super) fn prompt_from_crosshair(prep: &Prepared, cursor: [f64; 3]) -> Option<[f32; 3]> {
    let vidx = cursor;
    let mut out = [0f32; 3];
    for (a, (pa, fa)) in prep.perm.iter().zip(prep.flip.iter()).enumerate() {
        let v = vidx[*pa];
        if v < 0.0 || v >= prep.oriented_dims[a] as f64 {
            return None;
        }
        let oriented = if *fa {
            prep.oriented_dims[a] as f64 - 1.0 - v
        } else {
            v
        };
        let c = oriented - prep.crop_lo[a] as f64;
        if c < 0.0 || c >= prep.dims[a] as f64 {
            return None;
        }
        out[a] = c as f32;
    }
    Some(out)
}

/// A box centred on `centre` with a half-extent of `mm` millimetres, in the
/// prepared grid's own indices.
pub(super) fn box_around(centre: [f32; 3], mm: f32, prep: &Prepared, vol: &Volume) -> BBox {
    // Spacing along each canonical axis, taken from the volume axis it maps to.
    let spacing = [
        vol.spacing[prep.perm[0]] as f32,
        vol.spacing[prep.perm[1]] as f32,
        vol.spacing[prep.perm[2]] as f32,
    ];
    let mut b = [0f32; 6];
    for a in 0..3 {
        let half = (mm / spacing[a].max(1e-3)).max(1.0);
        b[a] = (centre[a] - half).max(0.0);
        b[a + 3] = (centre[a] + half).min(prep.dims[a] as f32 - 1.0);
    }
    b
}

impl ViewerApp {
    /// Tools ▶ prompt segmentation.
    pub(super) fn open_segvol_dialog(&mut self, slot: usize) {
        if self.slots[slot].study.is_none() || self.segvol_job.is_some() {
            return;
        }
        self.segvol_dialog = Some(SegVolDialog {
            slot,
            kind: PromptKind::Box,
            extent_mm: 40.0,
            text: String::new(),
            cfg: Config::default(),
            name: "Prompted".to_string(),
        });
    }

    pub(super) fn start_segvol(&mut self, d: &SegVolDialog) {
        if self.segvol_job.is_some() {
            return;
        }
        let Some(study) = self.slots[d.slot].study.as_ref() else {
            return;
        };
        let volume = study.volume.clone();
        let cursor = self.slots[d.slot].cursor;
        let models_dir = if self.segvol_models_dir.trim().is_empty() {
            weights::default_models_dir()
        } else {
            PathBuf::from(self.segvol_models_dir.trim())
        };
        let progress = Arc::new(SegVolProgress::default());
        progress.set("Preparing the volume…");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        let (slot, kind, extent, text, cfg, name) = (
            d.slot,
            d.kind,
            d.extent_mm,
            d.text.trim().to_string(),
            d.cfg,
            d.name.trim().to_string(),
        );
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let res = run_segvol(
                &volume,
                cursor,
                kind,
                extent,
                &text,
                cfg,
                &models_dir,
                &p2,
                t0,
                name,
            );
            let _ = tx.send((slot, res));
        });
        self.segvol_slot = slot;
        self.segvol_job = Some(Job { progress, rx });
    }

    /// A run finished: verify the slot still shows the same volume, then add
    /// the mask as an editable segmentation.
    pub(super) fn on_segvol_done(&mut self, slot: usize, result: SegVolResult) {
        let valid = self.slots[slot].study.as_ref().is_some_and(|st| {
            st.volume.dims == result.volume_dims
                && st.volume.frame_of_reference_uid == result.frame_of_reference_uid
        });
        if !valid {
            self.error = Some(
                "Prompted segmentation finished, but the dataset changed while \
                 it was running — the result was discarded."
                    .into(),
            );
            return;
        }
        if result.voxels == 0 {
            self.error = Some(
                "The prompt produced an empty mask. Try a larger box or a different prompt.".into(),
            );
            return;
        }
        let color = segmentation::SEG_PALETTE
            [self.slots[slot].segs.len() % segmentation::SEG_PALETTE.len()];
        let seg = Segmentation::from_label_map(
            result.name.clone(),
            color,
            result.volume_dims,
            &result.mask,
            1,
        );
        self.slots[slot].segs.push(seg);
        self.slots[slot].active_seg = self.slots[slot].segs.len() - 1;
        self.segvol_status = Some(format!(
            "{}: {} voxels in {:.1}s on {} ({} refinement window(s), coarse pass {})",
            result.name,
            result.voxels,
            result.elapsed_secs,
            result.device,
            result.windows,
            if result.coarse { "ran" } else { "skipped" }
        ));
    }

    /// The prompt dialog, and the progress readout while a run is in flight.
    pub(super) fn segvol_window(&mut self, ctx: &egui::Context) {
        if let Some(job) = &self.segvol_job {
            let msg = job.progress.get();
            let frac = job.progress.frac();
            let mut cancel = false;
            egui::Window::new("🧠 Prompt segmentation")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("Dataset {}", SLOT_NAMES[self.segvol_slot]));
                    let dev = job.progress.device();
                    if !dev.is_empty() {
                        ui.weak(format!("Encoder: {dev}"));
                    }
                    ui.add(egui::ProgressBar::new(frac).show_percentage());
                    ui.label(if msg.is_empty() { "Working…" } else { &msg });
                    ui.separator();
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            if cancel {
                if let Some(job) = &self.segvol_job {
                    job.progress.cancel();
                }
            }
            return;
        }

        let Some(d) = &mut self.segvol_dialog else {
            return;
        };
        let mut open = true;
        let mut run = false;
        let mut close = false;
        egui::Window::new("🧠 Prompt segmentation")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(format!("Dataset {}", SLOT_NAMES[d.slot]));
                ui.separator();
                ui.label("Prompt");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut d.kind, PromptKind::Box, "Box")
                        .on_hover_text("A box centred on the crosshair — the most reliable prompt, and the only one that works well for lesions");
                    ui.radio_value(&mut d.kind, PromptKind::Point, "Point")
                        .on_hover_text("A single click at the crosshair");
                    ui.radio_value(&mut d.kind, PromptKind::Text, "Text")
                        .on_hover_text("A structure name, run through the model's trained prompt template");
                });
                match d.kind {
                    PromptKind::Box => {
                        ui.horizontal(|ui| {
                            ui.label("Half-extent:");
                            ui.add(
                                egui::DragValue::new(&mut d.extent_mm)
                                    .range(2.0..=300.0)
                                    .suffix(" mm"),
                            );
                        });
                        ui.weak("Centred on the crosshair. Move it first, then run.");
                    }
                    PromptKind::Point => {
                        ui.weak("Uses the crosshair position. Move it first, then run.");
                    }
                    PromptKind::Text => {
                        ui.horizontal(|ui| {
                            ui.label("Structure:");
                            ui.add(
                                egui::TextEdit::singleline(&mut d.text)
                                    .hint_text("liver, pancreas, aorta…")
                                    .desired_width(180.0),
                            );
                        });
                        egui::ComboBox::from_label("")
                            .selected_text("pick a known structure")
                            .show_ui(ui, |ui| {
                                for l in 1u8..=117 {
                                    let n = crate::autoseg::classes::class_name(l);
                                    if ui.selectable_label(false, n.replace('_', " ")).clicked() {
                                        d.text = n.replace('_', " ");
                                    }
                                }
                            });
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.add(egui::TextEdit::singleline(&mut d.name).desired_width(160.0));
                });
                ui.collapsing("Options", |ui| {
                    ui.checkbox(&mut d.cfg.use_zoom_in, "Refinement pass")
                        .on_hover_text(
                            "The second, sliding-window pass. Without it the result is a \
                             single coarse pass — much faster, much blockier.",
                        );
                    ui.add_enabled(
                        d.kind == PromptKind::Box,
                        egui::Checkbox::new(
                            &mut d.cfg.skip_coarse_with_box,
                            "Skip the search pass (box only)",
                        ),
                    )
                    .on_hover_text(
                        "The first pass only exists to locate the structure. With a box \
                         drawn by hand it is redundant: skipping it roughly halves the \
                         work and avoids losing small lesions to the downsample. This \
                         departs from the reference implementation.",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Threshold:");
                        ui.add(egui::Slider::new(&mut d.cfg.threshold, 0.05..=0.95));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Model folder:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.segvol_models_dir)
                                .desired_width(220.0),
                        );
                    });
                });
                ui.separator();
                ui.label(
                    egui::RichText::new(
                        "The SegVol weights (~724 MB) are downloaded from Hugging Face on \
                         first use. They carry no license declaration; they are fetched to \
                         this machine at your request and are not redistributed.",
                    )
                    .small()
                    .color(warn_color(ui.visuals())),
                );
                ui.label(
                    egui::RichText::new(
                        "Research and QA only — review every contour before use.",
                    )
                    .small(),
                );
                ui.separator();
                ui.horizontal(|ui| {
                    let ready = d.kind != PromptKind::Text || !d.text.trim().is_empty();
                    if ui
                        .add_enabled(ready, egui::Button::new("Run"))
                        .clicked()
                    {
                        run = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });
        if run {
            if let Some(d) = self.segvol_dialog.take() {
                self.start_segvol(&d);
            }
        } else if !open || close {
            self.segvol_dialog = None;
        }
    }
}

/// The worker body. Kept a free function so the closure captures no `&self`.
#[allow(clippy::too_many_arguments)]
fn run_segvol(
    volume: &Volume,
    cursor: [f64; 3],
    kind: PromptKind,
    extent_mm: f32,
    text: &str,
    cfg: Config,
    models_dir: &std::path::Path,
    progress: &SegVolProgress,
    t0: std::time::Instant,
    name: String,
) -> anyhow::Result<SegVolResult> {
    use anyhow::{bail, Context};

    let prep = preprocess::prepare(volume);
    if progress.cancelled() {
        bail!("cancelled");
    }

    // The converted-weight cache sits beside the checkpoint.
    let cache = models_dir.join("segvol.safetensors");
    if !cache.is_file() {
        progress.set("Fetching and converting the SegVol checkpoint…");
        convert_checkpoint(models_dir, &cache, progress)?;
    }
    if progress.cancelled() {
        bail!("cancelled");
    }
    progress.set("Loading the network…");
    let params = Params::new(crate::nn::cache::load_safetensors(&cache)?);
    #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
    let mut net = SegVolNet::build(&params).context("assemble the SegVol network")?;

    // Put the image encoder on the GPU when a usable adapter exists; fall
    // back to the CPU otherwise. Either way the device is reported, both in
    // the progress dialog and in the finished-run status line.
    #[cfg(feature = "gpu")]
    let device = {
        progress.set("Looking for a GPU…");
        match crate::segvol::gpu::GpuContext::try_new()
            .and_then(|ctx| crate::segvol::gpu::GpuVit::new(&ctx, &params).map(|v| (ctx, v)))
        {
            Ok((ctx, vit)) => {
                net.attach_gpu(vit);
                format!("GPU ({})", ctx.describe())
            }
            Err(e) => {
                eprintln!("segvol: no usable GPU, running on the CPU: {e:#}");
                format!("CPU ({} threads)", rayon::current_num_threads())
            }
        }
    };
    #[cfg(not(feature = "gpu"))]
    let device = format!("CPU ({} threads)", rayon::current_num_threads());
    progress.set_device(&device);

    let centre = prompt_from_crosshair(&prep, cursor);
    let mut points: Vec<Point> = Vec::new();
    let mut boxes: Vec<BBox> = Vec::new();
    let mut text_vec: Option<Vec<f32>> = None;
    match kind {
        PromptKind::Box => {
            let c = centre.context("the crosshair is outside the volume's foreground")?;
            boxes.push(box_around(c, extent_mm, &prep, volume));
        }
        PromptKind::Point => {
            let c = centre.context("the crosshair is outside the volume's foreground")?;
            points.push(Point::foreground(c));
        }
        PromptKind::Text => {
            if text.is_empty() {
                bail!("enter a structure name");
            }
            progress.set("Encoding the text prompt…");
            for f in [weights::CLIP_VOCAB, weights::CLIP_MERGES] {
                weights::ensure_file(&f, models_dir, progress)?;
            }
            let bpe = Bpe::from_dir(models_dir)?;
            let enc = TextEncoder::build(&params)?;
            text_vec = Some(enc.encode_structure(&bpe, text));
        }
    }

    let seg = infer::segment(
        &net,
        &prep,
        &points,
        &boxes,
        text_vec.as_deref(),
        cfg,
        progress,
    )?;
    let mask = prep.mask_to_volume_grid(&seg.mask, volume);
    Ok(SegVolResult {
        mask,
        name,
        voxels: seg.voxels,
        windows: seg.windows,
        coarse: seg.coarse,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        device,
        volume_dims: volume.dims,
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
    })
}

/// Download the checkpoint if needed and write the converted-weight cache.
fn convert_checkpoint(
    models_dir: &std::path::Path,
    cache: &std::path::Path,
    progress: &SegVolProgress,
) -> anyhow::Result<()> {
    use crate::nn::pickle::Dtype;
    use crate::segvol::layout;

    let path = weights::ensure_file(&weights::CHECKPOINT, models_dir, progress)?;
    let mut reader = weights::open_checkpoint(&path)?;
    let metas: Vec<_> = reader
        .tensors
        .iter()
        .filter(|(n, _)| !layout::is_dead_weight(n))
        .cloned()
        .collect();
    let mut named = Vec::with_capacity(metas.len());
    for (i, (name, meta)) in metas.iter().enumerate() {
        if progress.cancelled() {
            anyhow::bail!("cancelled");
        }
        // CLIP's position_ids buffer is the checkpoint's only integer tensor
        // and nothing reads it.
        if !matches!(meta.dtype, Dtype::F32 | Dtype::F16 | Dtype::F64) {
            continue;
        }
        named.push((
            layout::normalize_key(name).to_string(),
            meta.shape.clone(),
            reader.read_f32(meta)?,
        ));
        <SegVolProgress as crate::nn::cache::ProgressSink>::report(
            progress,
            i as f32 / metas.len() as f32,
            &format!("Converting weights: {}/{}", i + 1, metas.len()),
        );
    }
    std::fs::create_dir_all(models_dir)?;
    crate::nn::cache::save_safetensors(cache, &named, crate::nn::cache::StoreDtype::F32)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    fn axial(dims: [usize; 3]) -> Volume {
        Volume {
            data: vec![100i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [2.0, 2.0, 4.0],
            origin: Vec3::ZERO,
            row_dir: Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            col_dir: Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            normal: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 100,
        }
    }

    #[test]
    fn a_box_is_clamped_to_the_prepared_grid_and_sized_in_millimetres() {
        let vol = axial([20, 16, 8]);
        let prep = preprocess::prepare(&vol);
        // canonical [S, A, R] over a standard axial volume maps to [z, y, x],
        // so spacing along the canonical axes is [4, 2, 2] mm
        let centre = [1.0, 8.0, 10.0];
        let b = box_around(centre, 8.0, &prep, &vol);
        // 8 mm at 4 mm/voxel is 2 voxels along axis 0, clamped at zero
        assert_eq!(b[0], 0.0);
        assert!((b[3] - 3.0).abs() < 1e-5, "{b:?}");
        // 8 mm at 2 mm/voxel is 4 voxels along axes 1 and 2
        assert!(
            (b[1] - 4.0).abs() < 1e-5 && (b[4] - 12.0).abs() < 1e-5,
            "{b:?}"
        );
        // never outside the grid
        for a in 0..3 {
            assert!(
                b[a] >= 0.0 && b[a + 3] <= prep.dims[a] as f32 - 1.0,
                "{b:?}"
            );
        }
    }

    #[test]
    fn a_box_always_has_positive_extent_even_at_zero_millimetres() {
        let vol = axial([20, 16, 8]);
        let prep = preprocess::prepare(&vol);
        let b = box_around([5.0, 5.0, 2.0], 0.0, &prep, &vol);
        for a in 0..3 {
            assert!(b[a + 3] > b[a], "axis {a} collapsed: {b:?}");
        }
    }

    #[test]
    fn a_crosshair_outside_the_volume_yields_no_prompt() {
        let vol = axial([20, 16, 8]);
        let prep = preprocess::prepare(&vol);
        assert!(prompt_from_crosshair(&prep, [1e6, 1e6, 1e6]).is_none());
        // and one inside does yield a position within the prepared grid
        let inside = prompt_from_crosshair(&prep, [10.0, 8.0, 4.0]).unwrap();
        for (a, c) in inside.iter().enumerate() {
            assert!(*c >= 0.0 && *c < prep.dims[a] as f32, "{inside:?}");
        }
    }
}
