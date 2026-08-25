//! *Tools ▶ Propagate structures*: carrying contours and segmentations
//! across the active registration.
//!
//! The window is deliberately thin — the hard part is the transform, and
//! that already exists. What it adds is the choice of *what* travels, the
//! direction, and one option that matters clinically: refining the
//! registration on an enclosing structure first, which is what makes a small
//! structure inside a larger one land where it belongs rather than where the
//! whole patient's average deformation puts it.

use super::*;
use crate::propagate::{self, Propagated, Subject};

/// The propagation window's state.
pub(super) struct PropagateDialog {
    /// Dataset the structures come from; they land on the other one.
    pub src_slot: usize,
    /// Selected ROIs of the source dataset's active structure set.
    pub structs: Vec<bool>,
    /// Selected segmentations of the source dataset.
    pub segs: Vec<bool>,
    /// Refine the registration on this region of the *fixed* dataset first.
    pub local: RegRoi,
    pub local_margin_mm: f64,
    /// What the last run produced.
    pub summary: Vec<String>,
}

impl Default for PropagateDialog {
    fn default() -> Self {
        PropagateDialog {
            src_slot: 0,
            structs: Vec::new(),
            segs: Vec::new(),
            local: RegRoi::Whole,
            local_margin_mm: 10.0,
            summary: Vec::new(),
        }
    }
}

/// What a propagation run hands back.
pub(super) struct PropOutcome {
    pub items: Vec<Propagated>,
    /// A local refinement run on the way, which becomes the active
    /// registration so the sidebar reports what was actually used.
    pub refined: Option<RegOutcome>,
}

impl ViewerApp {
    pub(super) fn open_propagate_window(&mut self, src_slot: usize) {
        let mut d = PropagateDialog {
            src_slot,
            ..Default::default()
        };
        self.sync_propagate_lists(&mut d);
        self.propagate_dialog = Some(d);
    }

    /// Keep the check-box lists the same length as what the dataset holds.
    fn sync_propagate_lists(&self, d: &mut PropagateDialog) {
        let n_struct = self.slots[d.src_slot]
            .active_structures()
            .map(|ss| ss.rois.len())
            .unwrap_or(0);
        d.structs.resize(n_struct, false);
        d.segs.resize(self.slots[d.src_slot].segs.len(), false);
    }

    /// Collect the masks of everything ticked.
    fn propagate_subjects(&self, d: &PropagateDialog) -> Result<Vec<Subject>, String> {
        let slot = d.src_slot;
        let Some(study) = &self.slots[slot].study else {
            return Err(format!("dataset {} is not loaded", SLOT_NAMES[slot]));
        };
        let vol = &study.volume;
        let mut out = Vec::new();
        if let Some(ss) = self.slots[slot].active_structures() {
            for (i, roi) in ss.rois.iter().enumerate() {
                if !d.structs.get(i).copied().unwrap_or(false) {
                    continue;
                }
                match segmentation::rasterize_roi(vol, roi) {
                    Some(mask) => out.push(Subject {
                        name: roi.name.clone(),
                        color: roi.color,
                        mask,
                    }),
                    None => {
                        return Err(format!(
                            "'{}' has no planar contour inside the displayed volume",
                            roi.name
                        ))
                    }
                }
            }
        }
        for (i, seg) in self.slots[slot].segs.iter().enumerate() {
            if d.segs.get(i).copied().unwrap_or(false) && seg.count > 0 {
                out.push(Subject {
                    name: seg.name.clone(),
                    color: seg.color,
                    mask: seg.mask.clone(),
                });
            }
        }
        if out.is_empty() {
            return Err("tick at least one structure or segmentation".into());
        }
        Ok(out)
    }

    /// Start the propagation (with its optional local refinement) on a
    /// worker thread.
    fn start_propagation(&mut self) {
        if self.propagate_job.is_some() {
            return;
        }
        let Some(d) = &self.propagate_dialog else {
            return;
        };
        let Some(reg) = &self.registration else {
            self.error = Some("Run a registration first — propagation needs one.".into());
            return;
        };
        let src_slot = d.src_slot;
        let dst_slot = 1 - src_slot;
        let subjects = match self.propagate_subjects(d) {
            Ok(s) => s,
            Err(e) => {
                self.error = Some(format!("Propagation: {e}"));
                return;
            }
        };
        let (Some(s), Some(t)) = (&self.slots[src_slot].study, &self.slots[dst_slot].study) else {
            self.error = Some("Propagation needs two loaded datasets".into());
            return;
        };
        let src_vol = s.volume.clone();
        let dst_vol = t.volume.clone();
        // The transform maps fixed → moving. Propagating *onto* the moving
        // dataset therefore runs through the inverse.
        let fixed_slot = reg.fixed_slot;
        let use_inverse = dst_slot != fixed_slot;
        let transform = reg.result.transform.clone();

        // The optional local refinement, run before anything is carried.
        let local = d.local;
        let margin = d.local_margin_mm;
        let region = if local == RegRoi::Whole {
            None
        } else {
            match self.region_for(fixed_slot, local, margin) {
                Ok(r) => r,
                Err(e) => {
                    self.error = Some(format!("Local refinement: {e:#}"));
                    return;
                }
            }
        };
        let (Some(fx), Some(mv)) = (
            &self.slots[fixed_slot].study,
            &self.slots[1 - fixed_slot].study,
        ) else {
            return;
        };
        let fixed_vol = fx.volume.clone();
        let moving_vol = mv.volume.clone();
        let mut params = self.current_reg_params(region.clone(), true);
        // A refinement is a deformation on top of what is there; a rigid
        // method would replace the alignment rather than refine it.
        if !params.method.is_deformable() {
            params.method = registration::RegMethod::PlastimatchBSpline;
        }
        let field_step = self.field_step_mm;

        let progress = Arc::new(Progress::default());
        progress.set("starting…");
        self.propagate_job = Some(Job::spawn(progress, move |p| {
            let mut refined = None;
            let transform = if region.is_some() {
                p.set_phase(0.0, 0.6);
                p.set("Refining the registration on the region…");
                match registration::register(&fixed_vol, &moving_vol, &params, p) {
                    Ok(result) => {
                        let field = VectorField::sample(
                            &fixed_vol,
                            &result.transform,
                            region.as_deref(),
                            field_step,
                        );
                        let t = result.transform.clone();
                        refined = Some(RegOutcome {
                            result,
                            field,
                            region,
                        });
                        t
                    }
                    Err(e) => return (dst_slot, Err(e)),
                }
            } else {
                transform
            };
            p.set_phase(
                if refined.is_some() { 0.6 } else { 0.0 },
                if refined.is_some() { 0.4 } else { 1.0 },
            );
            let items =
                propagate::propagate(&src_vol, &dst_vol, &transform, use_inverse, &subjects, p);
            (dst_slot, items.map(|items| PropOutcome { items, refined }))
        }));
    }

    /// A propagation run landed: install the masks (and the refinement).
    pub(super) fn on_propagation_done(&mut self, dst_slot: usize, out: PropOutcome) {
        if let Some(refined) = out.refined {
            let fixed_slot = self
                .registration
                .as_ref()
                .map(|r| r.fixed_slot)
                .unwrap_or(0);
            self.registration = Some(ActiveRegistration {
                result: refined.result,
                fixed_slot,
                field: Arc::new(refined.field),
                region: refined.region,
            });
            self.reg_gen += 1;
        }
        let Some(study) = &self.slots[dst_slot].study else {
            return;
        };
        let dims = study.volume.dims;
        let src = 1 - dst_slot;
        let mut lines = Vec::new();
        for item in out.items {
            lines.push(item.summary());
            if item.voxels == 0 {
                continue;
            }
            let name = format!("{} (from {})", item.name, SLOT_NAMES[src]);
            self.add_colored_segmentation(dst_slot, name, item.color, dims, &item.mask);
        }
        if let Some(d) = &mut self.propagate_dialog {
            d.summary = lines;
        }
        self.settings_gen += 1;
    }

    // -- the window --------------------------------------------------------

    pub(super) fn propagate_window(&mut self, ctx: &egui::Context) {
        if self.propagate_dialog.is_none() {
            return;
        }
        let mut open = true;
        let mut close = false;
        let mut run = false;
        let mut cancel = false;
        let mut set_all: Option<bool> = None;

        // Everything the closure needs to read while `self` is borrowed.
        let registered = self.registration.as_ref().map(|r| {
            (
                r.fixed_slot,
                r.result.method.label().to_string(),
                r.result.region.clone(),
            )
        });
        let running = self.propagate_job.is_some();
        let mut d = self.propagate_dialog.take().unwrap();
        self.sync_propagate_lists(&mut d);
        let src_slot = d.src_slot;
        let dst_slot = 1 - src_slot;
        let struct_rows: Vec<(String, [u8; 3])> = self.slots[src_slot]
            .active_structures()
            .map(|ss| ss.rois.iter().map(|r| (r.name.clone(), r.color)).collect())
            .unwrap_or_default();
        let seg_rows: Vec<(String, [u8; 3], f64)> = {
            let spacing = self.slots[src_slot]
                .study
                .as_ref()
                .map(|s| s.volume.spacing)
                .unwrap_or([1.0; 3]);
            self.slots[src_slot]
                .segs
                .iter()
                .map(|s| (s.name.clone(), s.color, s.volume_cm3(spacing)))
                .collect()
        };
        let local_choices = registered
            .as_ref()
            .map(|(fixed, _, _)| self.region_choices_for(*fixed))
            .unwrap_or_default();

        egui::Window::new(format!(
            "⇄ Propagate structures — {} ▶ {}",
            SLOT_NAMES[src_slot], SLOT_NAMES[dst_slot]
        ))
        .id(egui::Id::new("propagate_window"))
        .collapsible(true)
        .resizable(true)
        .default_width(420.0)
        .open(&mut open)
        .show(ctx, |ui| {
            ui.label(
                "Carries structures and segmentations from one dataset to the other \
                 through the active registration. Every destination voxel is asked \
                 where it comes from, so nothing is left with holes.",
            );
            ui.separator();
            match &registered {
                None => {
                    ui.colored_label(
                        alert_color(ui.visuals()),
                        "No active registration — run one in the sidebar first.",
                    );
                }
                Some((fixed, method, region)) => {
                    ui.weak(format!(
                        "Using: {method}{}",
                        match region {
                            Some(r) => format!(" · restricted to {r}"),
                            None => String::new(),
                        }
                    ));
                    ui.weak(format!(
                        "Fixed image: dataset {} — the transform is inverted \
                         automatically for the other direction.",
                        SLOT_NAMES[*fixed]
                    ));
                }
            }
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("From");
                ui.selectable_value(&mut d.src_slot, 0, "A ▶ B");
                ui.selectable_value(&mut d.src_slot, 1, "B ▶ A");
            });

            ui.horizontal(|ui| {
                if ui.small_button("All").clicked() {
                    set_all = Some(true);
                }
                if ui.small_button("None").clicked() {
                    set_all = Some(false);
                }
                let n = d.structs.iter().filter(|v| **v).count()
                    + d.segs.iter().filter(|v| **v).count();
                ui.weak(format!("{n} selected"));
            });

            egui::ScrollArea::vertical()
                .max_height(260.0)
                .show(ui, |ui| {
                    if !struct_rows.is_empty() {
                        ui.label(egui::RichText::new("Structures").strong());
                        for (i, (name, color)) in struct_rows.iter().enumerate() {
                            ui.horizontal(|ui| {
                                if let Some(on) = d.structs.get_mut(i) {
                                    ui.checkbox(on, "");
                                }
                                ui.colored_label(
                                    Color32::from_rgb(color[0], color[1], color[2]),
                                    "◼",
                                );
                                ui.label(name);
                            });
                        }
                    }
                    if !seg_rows.is_empty() {
                        ui.add_space(4.0);
                        ui.label(egui::RichText::new("Segmentations").strong());
                        for (i, (name, color, cm3)) in seg_rows.iter().enumerate() {
                            ui.horizontal(|ui| {
                                if let Some(on) = d.segs.get_mut(i) {
                                    ui.checkbox(on, "");
                                }
                                ui.colored_label(
                                    Color32::from_rgb(color[0], color[1], color[2]),
                                    "◼",
                                );
                                ui.label(name);
                                ui.weak(format!("{cm3:.1} cm³"));
                            });
                        }
                    }
                    if struct_rows.is_empty() && seg_rows.is_empty() {
                        ui.weak("This dataset has nothing to propagate.");
                    }
                });

            ui.separator();
            egui::CollapsingHeader::new("Refine locally first")
                .id_salt("prop_local")
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(
                        "A structure inside a larger one lands where the *larger* one's \
                         deformation puts it. Refining the registration on the enclosing \
                         structure first is what fixes that — and it only changes the \
                         transform inside that structure.",
                    );
                    ui.horizontal(|ui| {
                        ui.label("Region");
                        let current = local_choices
                            .iter()
                            .find(|(c, _)| *c == d.local)
                            .map(|(_, l)| l.clone())
                            .unwrap_or_else(|| "No refinement".into());
                        egui::ComboBox::from_id_salt("prop_region")
                            .selected_text(current)
                            .width(200.0)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut d.local, RegRoi::Whole, "No refinement");
                                for (choice, label) in &local_choices {
                                    if *choice == RegRoi::Whole {
                                        continue;
                                    }
                                    ui.selectable_value(&mut d.local, *choice, label);
                                }
                            });
                    });
                    if d.local != RegRoi::Whole {
                        ui.horizontal(|ui| {
                            ui.label("Margin");
                            ui.add(
                                egui::DragValue::new(&mut d.local_margin_mm)
                                    .speed(1.0)
                                    .range(0.0..=60.0)
                                    .suffix(" mm"),
                            );
                        });
                        ui.weak(
                            "The refinement replaces the active registration, so the \
                             sidebar reports exactly what the propagation used.",
                        );
                    }
                });

            ui.separator();
            match &self.propagate_job {
                Some(job) => cancel = progress_row(ui, &job.progress),
                None => {
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(registered.is_some(), egui::Button::new("▶ Propagate"))
                            .on_hover_text(
                                "Results land as editable segmentations on the other \
                                 dataset, convertible to RTSTRUCT like any other",
                            )
                            .clicked()
                        {
                            run = true;
                        }
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
                }
            }
            if !d.summary.is_empty() {
                ui.separator();
                ui.label(egui::RichText::new("Last run").strong());
                for line in &d.summary {
                    ui.monospace(line);
                }
                ui.weak(
                    "A volume change is the deformation's doing: it is exactly what the \
                     Jacobian in the registration panel reports.",
                );
            }
        });

        if let Some(v) = set_all {
            d.structs.iter_mut().for_each(|s| *s = v);
            d.segs.iter_mut().for_each(|s| *s = v);
        }
        // A run in flight keeps the window alive whatever the ✕ says: the
        // job's results have to land somewhere.
        if running || (!close && open) {
            self.propagate_dialog = Some(d);
        }
        if cancel {
            if let Some(job) = &self.propagate_job {
                job.progress.cancel();
            }
        }
        if run {
            self.start_propagation();
        }
    }
}
