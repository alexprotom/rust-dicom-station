//! MedSAM2: the user interface of the slice-propagating engine.
//!
//! Where the SegVol dialog ([`super::prompt_seg`]) asks the network to find a
//! structure anywhere in a 32 x 256 x 256 view of the study, this one asks it
//! to follow something the user can already see: a box, a click, or an
//! existing contour on **one** slice, propagated up and down the stack at the
//! slice's own resolution.
//!
//! The prompt is anchored on the crosshair, like SegVol's. The natural next
//! step is dragging the box directly in the axial viewport, which needs
//! interaction state in [`super::views`] rather than anything here.
//!
//! Everything the network needs crosses one conversion: the crosshair is
//! fractional voxel indices in the slot's own volume, and
//! [`Prepared`](crate::medsam2::preprocess::Prepared) reorients that volume to
//! axial reading order. [`prompt_from_crosshair`] is the only place that
//! mapping happens.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use crate::medsam2::engine::{Engine, EnginePrompt, PixelPrompt};
use crate::medsam2::infer::{Config, Hooks};
use crate::medsam2::preprocess::{Prepared, Window};
use crate::medsam2::weights::{self, Variant};
use crate::volume::Volume;

use super::*;

/// Which kind of prompt the dialog is collecting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum PromptKind {
    /// A box on the crosshair's slice.
    Box,
    /// A single click.
    Point,
    /// The active segmentation's contour on that slice.
    Mask,
}

/// Where the intensity window comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum WindowSource {
    /// What the viewport is showing — what you see is what the model sees.
    Viewport,
    /// One of the windows the MedSAM2 paper trained with.
    Preset(usize),
}

/// Progress, cancellation and the device label, shared with the worker.
#[derive(Default)]
pub struct Medsam2Progress {
    message: Mutex<String>,
    device: Mutex<String>,
    frac: AtomicU32,
    cancelled: AtomicBool,
}

impl Medsam2Progress {
    pub fn set(&self, m: impl Into<String>) {
        *self.message.lock().unwrap() = m.into();
    }
    pub fn get(&self) -> String {
        self.message.lock().unwrap().clone()
    }
    pub fn set_device(&self, d: impl Into<String>) {
        *self.device.lock().unwrap() = d.into();
    }
    pub fn device(&self) -> String {
        self.device.lock().unwrap().clone()
    }
    pub fn frac(&self) -> f32 {
        f32::from_bits(self.frac.load(Ordering::Relaxed))
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
    fn advance(&self, frac: f32, msg: &str) {
        self.frac.store(frac.to_bits(), Ordering::Relaxed);
        if !msg.is_empty() {
            self.set(msg);
        }
    }
}

impl Hooks for Medsam2Progress {
    fn report(&self, frac: f32, msg: &str) {
        self.advance(frac, msg);
    }
    fn cancelled(&self) -> bool {
        Medsam2Progress::cancelled(self)
    }
}

impl crate::nn::cache::ProgressSink for Medsam2Progress {
    fn report(&self, frac: f32, msg: &str) {
        self.advance(frac, msg);
    }
    fn cancelled(&self) -> bool {
        Medsam2Progress::cancelled(self)
    }
}

/// A finished run, waiting to be landed on the slot it came from.
pub struct Medsam2Result {
    pub mask: Vec<u8>,
    pub name: String,
    pub voxels: u64,
    pub slices_visited: usize,
    pub extent: Option<(usize, usize)>,
    pub elapsed_secs: f64,
    pub device: String,
    pub volume_dims: [usize; 3],
    pub frame_of_reference_uid: String,
}

/// The dialog's state.
pub(super) struct Medsam2Dialog {
    pub slot: usize,
    pub kind: PromptKind,
    /// Half-extent of the box prompt, in millimetres.
    pub extent_mm: f32,
    pub window: WindowSource,
    pub variant: Variant,
    pub cfg: Config,
    /// Slices to track on each side; `bounded` turns it on.
    pub bounded: bool,
    pub reach: usize,
    pub name: String,
}

/// Turn the crosshair into a prompt on one prepared slice.
///
/// Returns the slice index and the prompt in that slice's pixels.
pub fn prompt_from_crosshair(
    prepared: &Prepared,
    volume: &Volume,
    cursor: [f64; 3],
    kind: PromptKind,
    extent_mm: f32,
    contour: Option<&[u8]>,
) -> Option<(usize, EnginePrompt)> {
    let voxel = [
        (cursor[0].round().max(0.0) as usize).min(volume.dims[0] - 1),
        (cursor[1].round().max(0.0) as usize).min(volume.dims[1] - 1),
        (cursor[2].round().max(0.0) as usize).min(volume.dims[2] - 1),
    ];

    let [slice, row, column] = prepared.from_volume_index(voxel);
    let (rows, cols) = (prepared.dims[1] as f32, prepared.dims[2] as f32);
    let half_row = (f64::from(extent_mm) / prepared.spacing[1]) as f32;
    let half_col = (f64::from(extent_mm) / prepared.spacing[2]) as f32;
    let (row, column) = (row as f32, column as f32);

    let prompt = match kind {
        PromptKind::Point => EnginePrompt::Points(vec![PixelPrompt::positive(row, column)]),
        PromptKind::Box => EnginePrompt::Points(PixelPrompt::box_corners(
            (row - half_row).max(0.0),
            (column - half_col).max(0.0),
            (row + half_row).min(rows - 1.0),
            (column + half_col).min(cols - 1.0),
        )),
        PromptKind::Mask => {
            let contour = contour?;
            let slice_mask = prepared.slice_from_volume_mask(contour, volume, slice);
            if slice_mask.iter().all(|v| *v == 0) {
                return None;
            }
            EnginePrompt::Mask(slice_mask)
        }
    };
    Some((slice, prompt))
}

/// The whole background run: prepare, load, propagate, map back.
#[allow(clippy::too_many_arguments)]
fn run_medsam2(
    volume: &Volume,
    cursor: [f64; 3],
    kind: PromptKind,
    extent_mm: f32,
    window: Window,
    variant: Variant,
    contour: Option<Vec<u8>>,
    cfg: Config,
    models_dir: &PathBuf,
    progress: &Medsam2Progress,
    started: std::time::Instant,
    name: String,
) -> anyhow::Result<Medsam2Result> {
    progress.set("Preparing the study…");
    let prepared = Prepared::prepare(volume, window);
    let Some((slice, prompt)) =
        prompt_from_crosshair(&prepared, volume, cursor, kind, extent_mm, contour.as_deref())
    else {
        anyhow::bail!("the crosshair slice has nothing to propagate");
    };

    progress.set("Loading the weights…");
    let params = weights::load(variant, models_dir, progress)?;
    let engine = Engine::load(&params, true)?;
    progress.set_device(engine.device().to_string());

    let (mask, seg) =
        engine.propagate_to_volume(&prepared, volume, slice, &prompt, &cfg, progress)?;
    Ok(Medsam2Result {
        mask,
        name,
        voxels: seg.voxels,
        slices_visited: seg.slices_visited,
        extent: seg.extent(),
        elapsed_secs: started.elapsed().as_secs_f64(),
        device: engine.device().to_string(),
        volume_dims: volume.dims,
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
    })
}

impl ViewerApp {
    /// Tools ▶ propagate from a slice.
    pub(super) fn open_medsam2_dialog(&mut self, slot: usize) {
        if self.slots[slot].study.is_none() || self.medsam2_job.is_some() {
            return;
        }
        self.medsam2_dialog = Some(Medsam2Dialog {
            slot,
            kind: PromptKind::Box,
            extent_mm: 25.0,
            window: WindowSource::Viewport,
            variant: Variant::default(),
            cfg: Config::default(),
            bounded: true,
            reach: 64,
            name: "Propagated".to_string(),
        });
    }

    pub(super) fn start_medsam2(&mut self, d: &Medsam2Dialog) {
        if self.medsam2_job.is_some() {
            return;
        }
        let Some(study) = self.slots[d.slot].study.as_ref() else {
            return;
        };
        let volume = study.volume.clone();
        let cursor = self.slots[d.slot].cursor;
        let window = match d.window {
            // The viewer's own window/level, so the model sees the image the
            // user is looking at.
            WindowSource::Viewport => {
                Window::from_width_level(self.window_width, self.window_center)
            }
            WindowSource::Preset(i) => {
                let (_, w, l) = Window::PRESETS[i];
                Window::from_width_level(w, l)
            }
        };
        // A mask prompt reads the active segmentation as it stands now.
        let contour = if d.kind == PromptKind::Mask {
            let slot = &self.slots[d.slot];
            slot.segs.get(slot.active_seg).map(|s| s.mask.clone())
        } else {
            None
        };
        let models_dir = if self.medsam2_models_dir.trim().is_empty() {
            weights::default_models_dir()
        } else {
            PathBuf::from(self.medsam2_models_dir.trim())
        };
        let mut cfg = d.cfg;
        cfg.max_slices = if d.bounded { Some(d.reach) } else { None };

        let progress = Arc::new(Medsam2Progress::default());
        progress.set("Preparing the study…");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        let (slot, kind, extent, variant, name) =
            (d.slot, d.kind, d.extent_mm, d.variant, d.name.trim().to_string());
        std::thread::spawn(move || {
            let t0 = std::time::Instant::now();
            let res = run_medsam2(
                &volume,
                cursor,
                kind,
                extent,
                window,
                variant,
                contour,
                cfg,
                &models_dir,
                &p2,
                t0,
                name,
            );
            let _ = tx.send((slot, res));
        });
        self.medsam2_slot = slot;
        self.medsam2_job = Some(Job { progress, rx });
    }

    /// A run finished: verify the slot still shows the same volume, then add
    /// the mask as an editable segmentation.
    pub(super) fn on_medsam2_done(&mut self, slot: usize, result: Medsam2Result) {
        let valid = self.slots[slot].study.as_ref().is_some_and(|st| {
            st.volume.dims == result.volume_dims
                && st.volume.frame_of_reference_uid == result.frame_of_reference_uid
        });
        if !valid {
            self.error = Some(
                "Propagation finished, but the dataset changed while it was \
                 running — the result was discarded."
                    .into(),
            );
            return;
        }
        if result.voxels == 0 {
            self.error = Some(
                "The prompt produced an empty mask. Try a larger box, or a slice \
                 where the structure is clearly visible."
                    .into(),
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
        let span = match result.extent {
            Some((a, b)) => format!("{} slice(s)", b - a + 1),
            None => "no slices".to_string(),
        };
        self.medsam2_status = Some(format!(
            "{}: {} voxels over {span} in {:.1}s on {} ({} slice(s) tracked)",
            result.name, result.voxels, result.elapsed_secs, result.device, result.slices_visited
        ));
    }

    /// The prompt dialog, and the progress readout while a run is in flight.
    pub(super) fn medsam2_window(&mut self, ctx: &egui::Context) {
        if let Some(job) = &self.medsam2_job {
            let msg = job.progress.get();
            let frac = job.progress.frac();
            let mut cancel = false;
            egui::Window::new("🧠 Propagate from a slice")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("Dataset {}", SLOT_NAMES[self.medsam2_slot]));
                    let dev = job.progress.device();
                    if !dev.is_empty() {
                        ui.weak(format!("Running on: {dev}"));
                    }
                    ui.add(egui::ProgressBar::new(frac).show_percentage());
                    ui.label(if msg.is_empty() { "Working…" } else { &msg });
                    ui.separator();
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            if cancel {
                if let Some(job) = &self.medsam2_job {
                    job.progress.cancel();
                }
            }
            return;
        }

        let Some(d) = &mut self.medsam2_dialog else {
            return;
        };
        let mut run = false;
        let mut close = false;
        egui::Window::new("🧠 Propagate from a slice")
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!("Dataset {}", SLOT_NAMES[d.slot]));
                ui.weak("Put the crosshair on the structure first: the prompt is anchored to it.");
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Prompt:");
                    ui.selectable_value(&mut d.kind, PromptKind::Box, "Box");
                    ui.selectable_value(&mut d.kind, PromptKind::Point, "Click");
                    ui.selectable_value(&mut d.kind, PromptKind::Mask, "Existing contour");
                });
                match d.kind {
                    PromptKind::Box => {
                        ui.add(
                            egui::Slider::new(&mut d.extent_mm, 5.0..=120.0)
                                .text("half-extent (mm)"),
                        );
                    }
                    PromptKind::Point => {
                        ui.weak("One positive click at the crosshair.");
                    }
                    PromptKind::Mask => {
                        ui.weak(
                            "The active segmentation's contour on the crosshair slice is \
                             propagated through the rest of the stack.",
                        );
                    }
                }

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Window:");
                    ui.selectable_value(&mut d.window, WindowSource::Viewport, "Viewport");
                    for (i, (name, _, _)) in Window::PRESETS.iter().enumerate() {
                        ui.selectable_value(&mut d.window, WindowSource::Preset(i), *name);
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Model:");
                    egui::ComboBox::from_id_salt("medsam2_variant")
                        .selected_text(d.variant.label())
                        .show_ui(ui, |ui| {
                            for v in Variant::ALL {
                                ui.selectable_value(&mut d.variant, v, v.label());
                            }
                        });
                });

                ui.separator();
                ui.checkbox(&mut d.bounded, "Limit how far it propagates");
                if d.bounded {
                    ui.add(egui::Slider::new(&mut d.reach, 4..=256).text("slices each way"));
                }
                ui.checkbox(&mut d.cfg.reverse_pass, "Propagate in both directions");
                ui.checkbox(
                    &mut d.cfg.largest_component,
                    "Keep only the largest connected component",
                );
                ui.add(egui::Slider::new(&mut d.cfg.threshold, -4.0..=4.0).text("threshold"));

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.add(egui::TextEdit::singleline(&mut d.name).desired_width(160.0));
                });
                ui.collapsing("Model cache", |ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.medsam2_models_dir)
                            .desired_width(320.0),
                    );
                    ui.weak(
                        "The weights (156 MB) are downloaded from Hugging Face on first use. \
                         They are licensed for research and education only.",
                    );
                });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Run").clicked() {
                        run = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
                if let Some(status) = &self.medsam2_status {
                    ui.separator();
                    ui.weak(status);
                }
            });

        if run {
            if let Some(d) = self.medsam2_dialog.take() {
                self.start_medsam2(&d);
            }
        } else if close {
            self.medsam2_dialog = None;
        }
    }
}
