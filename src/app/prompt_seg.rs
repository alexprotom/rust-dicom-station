//! Prompt-driven segmentation: the SegVol engine's user interface.
//!
//! Where auto-segmentation runs to completion on its own, this one is a
//! conversation - the user points at something and the network answers, and
//! the answer lands as an ordinary [`Segmentation`], editable with the brush
//! and convertible to RTSTRUCT like any other.
//!
//! The prompt is anchored on the crosshair. A box is the crosshair plus an
//! extent in millimetres, a point is the crosshair itself, and text is a
//! structure name run through the trained template. That keeps the whole
//! interaction inside one window, which stays open across runs and reports
//! each result on its last line. (Slice propagation, [`super::box_seg`], is
//! the tool whose box is drawn in the image instead.)
//!
//! Coordinates cross one conversion before reaching the network. The
//! crosshair is fractional voxel indices in the slot's own volume;
//! [`preprocess::prepare`] reorients that volume to canonical `[S, A, R]` and
//! crops it, and the network sees indices in the prepared grid.
//! [`prompt_from_crosshair`] does that mapping and is the only place it
//! happens.

use crate::models::{self, Engine as ModelsEngine};
use crate::nn::device::DevicePref;
use crate::progress::CANCELLED;
use crate::segvol::infer::{self, Config};
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

/// The tool window's state; it stays open across runs.
pub(super) struct SegVolDialog {
    pub slot: usize,
    pub kind: PromptKind,
    /// Half-extent of the box prompt, in millimetres.
    pub extent_mm: f32,
    /// Structure name for a text prompt.
    pub text: String,
    pub cfg: Config,
    pub device: DevicePref,
    pub name: String,
    /// One-line summary of the last finished run.
    pub status: Option<String>,
    /// Where the mask goes, and whether every phase of the displayed
    /// series' 4D group is prompted at the same spot.
    pub output: ToolOutput,
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

/// Everything a run needs, snapshotted from the window when it starts.
#[derive(Clone)]
struct SegVolRequest {
    /// The crosshair, as fractional voxel indices of the volume the run is
    /// on. For a run over the phases it is recomputed per phase from
    /// `patient`, since the phases need not share a lattice.
    cursor: [f64; 3],
    /// The crosshair in patient coordinates (mm).
    patient: Vec3,
    kind: PromptKind,
    extent_mm: f32,
    text: String,
    cfg: Config,
    device: DevicePref,
    models_dir: PathBuf,
    name: String,
}

/// Convert the crosshair - fractional voxel indices in the slot's volume -
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
    /// Tools ▶ prompt segmentation: open the tool window for `slot`.
    pub(super) fn open_segvol_dialog(&mut self, slot: usize) {
        if !self.slots[slot].has_volume() {
            return;
        }
        match &mut self.segvol_dialog {
            // Re-target an open window unless it is busy with the other slot.
            Some(d) if self.segvol_job.is_none() => d.slot = slot,
            Some(_) => {}
            None => {
                self.segvol_dialog = Some(SegVolDialog {
                    slot,
                    kind: PromptKind::Box,
                    extent_mm: 40.0,
                    text: String::new(),
                    cfg: Config::default(),
                    device: DevicePref::Auto,
                    name: "Prompted".to_string(),
                    status: None,
                    output: ToolOutput {
                        set: self.default_set_target(slot),
                        ..ToolOutput::default()
                    },
                });
            }
        }
    }

    /// Snapshot the prompt and the volume and run on a worker thread.
    pub(super) fn start_segvol(&mut self) {
        if self.segvol_job.is_some() {
            return;
        }
        let Some(d) = &self.segvol_dialog else {
            return;
        };
        let Some(study) = self.slots[d.slot].study.as_ref() else {
            return;
        };
        let volume = study.volume.clone();
        let displayed = PhaseInfo::displayed(study);
        let slot = d.slot;
        let c = self.slots[slot].cursor;
        let req = SegVolRequest {
            cursor: c,
            patient: volume.voxel_to_patient(c[0], c[1], c[2]),
            kind: d.kind,
            extent_mm: d.extent_mm,
            text: d.text.trim().to_string(),
            cfg: d.cfg,
            device: d.device,
            models_dir: self.engine_models_dir(ModelsEngine::SegVol),
            name: match d.kind {
                // A structure name is the natural name for what it finds.
                PromptKind::Text if d.name.trim() == "Prompted" => d.text.trim().to_string(),
                _ => d.name.trim().to_string(),
            },
        };
        let phases = if d.output.phases {
            match self.phase_inputs(slot) {
                Ok(p) => Some(p),
                Err(e) => {
                    self.error = Some(format!("{e:#}"));
                    return;
                }
            }
        } else {
            None
        };
        self.persist_settings();
        let progress = Arc::new(Progress::default());
        progress.set("Preparing the volume");
        self.segvol_slot = slot;
        self.segvol_job = Some(Job::spawn(progress, move |p| {
            let out = match phases {
                Some((group, inputs)) => run_on_phases(&group, inputs, p, |vol, p| {
                    // The same spot in the patient, on this phase's lattice.
                    let mut r = req.clone();
                    r.cursor = vol.patient_to_voxel(req.patient);
                    run_segvol(vol, &r, p)
                }),
                None => run_segvol(&volume, &req, p).map(|r| vec![(displayed, r)]),
            };
            (slot, out)
        }));
    }

    /// A run finished: verify the slot still shows the same volume, land
    /// the mask the way the output rows say - on the displayed series, or
    /// on every phase - and summarise it in the window.
    pub(super) fn on_segvol_done(&mut self, slot: usize, results: Vec<(PhaseInfo, SegVolResult)>) {
        let Some(output) = self.segvol_dialog.as_ref().map(|d| d.output.clone()) else {
            return;
        };
        let Some((first, result)) = results.first() else {
            return;
        };
        if let Some(group) = first.group.clone() {
            let n = results.len();
            let empty = results.iter().filter(|(_, r)| r.voxels == 0).count();
            let color = self.next_seg_color();
            let copies: Vec<_> = results
                .iter()
                .map(|(info, r)| {
                    let segs = if r.voxels == 0 {
                        Vec::new()
                    } else {
                        vec![Segmentation::from_label_map(
                            r.name.clone(),
                            color,
                            r.volume_dims,
                            &r.mask,
                            1,
                        )]
                    };
                    let notes = (r.voxels == 0)
                        .then(|| format!("phase {}: the prompt produced an empty mask", info.label))
                        .into_iter()
                        .collect();
                    info.copy(segs, notes)
                })
                .collect();
            let landed = self.land_phases(slot, &group, output.kind, "ORGAN", copies);
            if landed == 0 {
                return;
            }
            let name = result.name.clone();
            if let Some(d) = &mut self.segvol_dialog {
                d.status = Some(format!(
                    "✔ {name} on {landed} of {n} phases of {group}{}",
                    if empty > 0 {
                        format!(" - {empty} came out empty")
                    } else {
                        String::new()
                    }
                ));
            }
            return;
        }
        if !self.slot_still_shows(slot, result.volume_dims, &result.frame_of_reference_uid) {
            self.error = Some(stale_result(&PROMPT_SEG));
            return;
        }
        if result.voxels == 0 {
            self.error = Some(
                "The prompt produced an empty mask. Try a larger box or a different prompt.".into(),
            );
            return;
        }
        let color = self.next_seg_color();
        let seg = Segmentation::from_label_map(
            result.name.clone(),
            color,
            result.volume_dims,
            &result.mask,
            1,
        );
        if self.land_masks(slot, &output, "ORGAN", "Prompt segmentation", vec![seg]) == 0 {
            return;
        }
        let cm3 = self.slots[slot].voxels_cm3(result.voxels);
        if let Some(d) = &mut self.segvol_dialog {
            d.status = Some(format!(
                "✔ {}: {} voxels ({cm3:.1} cm³) in {:.1} s on {} - {} refinement window(s), \
                 coarse pass {}",
                result.name,
                result.voxels,
                result.elapsed_secs,
                result.device,
                result.windows,
                if result.coarse { "ran" } else { "skipped" }
            ));
        }
    }

    /// The tool window; while a run is in flight its buttons become the
    /// progress row.
    pub(super) fn segvol_section(&mut self, ui: &mut egui::Ui) {
        self.open_segvol_dialog(self.auto.slot);
        let sets = self.structure_set_labels(self.auto.slot);
        let group = self.displayed_group(self.auto.slot);
        let Some(d) = &mut self.segvol_dialog else {
            return;
        };
        let running = self
            .segvol_job
            .as_ref()
            .filter(|_| self.segvol_slot == d.slot);
        let models_dir = models::engine_dir(
            &models::root_from_setting(&self.models_dir),
            ModelsEngine::SegVol,
        );
        let mut run = false;
        let mut browse = false;
        let mut cancel = false;
        ui.label(
            "Segments whatever the prompt points at - a box, a click or a structure \
             name - with SegVol, re-implemented natively in Rust. For the lesions and \
             targets a fixed-class model cannot cover.",
        );
        ui.separator();
        ui.label("Prompt:");
        ui.horizontal_wrapped(|ui| {
            ui.radio_value(&mut d.kind, PromptKind::Box, "Box")
                .on_hover_text("A box centred on the crosshair - the most reliable prompt, and the only one that works well for lesions");
            ui.radio_value(&mut d.kind, PromptKind::Point, "Point")
                .on_hover_text("A single click at the crosshair");
            ui.radio_value(&mut d.kind, PromptKind::Text, "Text")
                .on_hover_text("A structure name, run through the model's trained prompt template");
        });
        match d.kind {
            PromptKind::Box => {
                ui.horizontal_wrapped(|ui| {
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
                ui.horizontal_wrapped(|ui| {
                    ui.label("Structure:");
                    ui.add(
                        egui::TextEdit::singleline(&mut d.text)
                            .hint_text("liver, pancreas, aorta")
                            .desired_width(180.0),
                    );
                });
                egui::ComboBox::from_id_salt("segvol_known_structure")
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
        ui.horizontal_wrapped(|ui| {
            ui.label("Name:");
            ui.add(egui::TextEdit::singleline(&mut d.name).desired_width(160.0));
        });
        scope_row(ui, &mut d.output.phases, group.as_ref());
        let phases = d.output.phases;
        output_rows(ui, &mut d.output, &sets, phases, "Prompt segmentation");
        ui.separator();
        ui.collapsing("Options", |ui| {
            ui.checkbox(&mut d.cfg.use_zoom_in, "Refinement pass")
                .on_hover_text(
                    "The second, sliding-window pass. Without it the result is a \
                     single coarse pass - much faster, much blockier.",
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
            ui.horizontal_wrapped(|ui| {
                ui.label("Threshold:");
                ui.add(egui::Slider::new(&mut d.cfg.threshold, 0.05..=0.95));
            });
            device_row(ui, &mut d.device);
            browse = models_dir_row(ui, &mut self.models_dir, ModelsEngine::SegVol);
        });
        ui.separator();
        let need = weights::download_needed(&models_dir, d.kind == PromptKind::Text);
        let weights_note = if need == 0 {
            "Weights: SegVol (BAAI, no licence declared) - cached ✔.".to_string()
        } else {
            format!(
                "Weights: SegVol (BAAI, no licence declared) - {} MB downloaded once \
                 from Hugging Face, at your request, never redistributed.",
                need / 1_000_000
            )
        };
        licence_line(ui, &weights_note, true);
        ui.separator();
        match running {
            Some(job) => cancel = progress_row(ui, &job.progress),
            None => {
                ui.horizontal_wrapped(|ui| {
                    let ready = d.kind != PromptKind::Text || !d.text.trim().is_empty();
                    if enabled_tip_button(
                        ui,
                        ready,
                        "▶ Segment",
                        if d.output.phases {
                            "Run the network on the prompt on every phase of the group"
                        } else {
                            "Run the network on the prompt"
                        },
                    ) {
                        run = true;
                    }
                });
            }
        }
        if let Some(status) = &d.status {
            ui.separator();
            ui.weak(status);
        }
        if browse {
            self.ask_folder("Model folder", |app, dir| {
                app.models_dir = dir.display().to_string();
            });
        }
        cancel_if(cancel, &self.segvol_job);
        if run {
            self.start_segvol();
        }
    }
}

/// The worker body. Kept a free function so the closure captures no `&self`.
fn run_segvol(
    volume: &Volume,
    req: &SegVolRequest,
    progress: &Progress,
) -> anyhow::Result<SegVolResult> {
    use anyhow::{bail, Context};

    let t0 = std::time::Instant::now();
    let prep = preprocess::prepare(volume);
    if progress.cancelled() {
        bail!(CANCELLED);
    }

    let params = weights::load(&req.models_dir, progress)?;
    if progress.cancelled() {
        bail!(CANCELLED);
    }
    progress.set("Assembling the network");
    #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
    let mut net = SegVolNet::build(&params).context("assemble the SegVol network")?;

    // Put the image encoder on the GPU when asked and a usable adapter
    // exists; fall back to the CPU otherwise. Either way the device is
    // reported, both in the progress row and in the finished-run status.
    progress.set("Choosing the compute device");
    let gpu = req.device.resolve()?;
    let device = match gpu {
        #[cfg(feature = "gpu")]
        Some(ctx) => {
            let vit = crate::segvol::gpu::GpuVit::new(&ctx, &params)
                .context("upload the image encoder")?;
            net.attach_gpu(vit);
            ctx.describe()
        }
        #[cfg(not(feature = "gpu"))]
        Some(ctx) => ctx.unreachable(),
        None => crate::nn::device::describe_cpu(),
    };
    progress.set_device(&device);

    let centre = prompt_from_crosshair(&prep, req.cursor);
    let mut points: Vec<Point> = Vec::new();
    let mut boxes: Vec<BBox> = Vec::new();
    let mut text_vec: Option<Vec<f32>> = None;
    match req.kind {
        PromptKind::Box => {
            let c = centre.context("the crosshair is outside the volume's foreground")?;
            boxes.push(box_around(c, req.extent_mm, &prep, volume));
        }
        PromptKind::Point => {
            let c = centre.context("the crosshair is outside the volume's foreground")?;
            points.push(Point::foreground(c));
        }
        PromptKind::Text => {
            if req.text.is_empty() {
                bail!("enter a structure name");
            }
            progress.set("Encoding the text prompt");
            for f in &weights::CLIP_FILES {
                f.ensure(&req.models_dir, progress)?;
            }
            let bpe = Bpe::from_dir(&req.models_dir)?;
            let enc = TextEncoder::build(&params)?;
            text_vec = Some(enc.encode_structure(&bpe, &req.text));
        }
    }

    let seg = infer::segment(
        &net,
        &prep,
        &points,
        &boxes,
        text_vec.as_deref(),
        req.cfg,
        progress,
    )?;
    let mask = prep.mask_to_volume_grid(&seg.mask, volume);
    Ok(SegVolResult {
        mask,
        name: req.name.clone(),
        voxels: seg.voxels,
        windows: seg.windows,
        coarse: seg.coarse,
        elapsed_secs: t0.elapsed().as_secs_f64(),
        device,
        volume_dims: volume.dims,
        frame_of_reference_uid: volume.frame_of_reference_uid.clone(),
    })
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
