//! *Tools ▶ DRR*: the digitally reconstructed radiograph window.
//!
//! Two forward projectors, one geometry, and the difference between them on
//! screen. The comparison is not decoration: an exact ray tracer and an
//! interpolating one are the two answers the field actually uses, and seeing
//! where they part company is how you find out whether a step size, a
//! detector resolution or a threshold is doing something you did not intend.

use crate::drr::{self, DrrComparison, DrrImage, DrrParams, Engine, Geometry, HuMode};

use super::*;

/// The DRR window's state.
pub(super) struct DrrDialog {
    /// Dataset the radiograph is computed from.
    pub slot: usize,
    pub params: DrrParams,
    /// Render both engines and compare them.
    pub both: bool,
    /// The last rendering(s), in engine order.
    pub images: Vec<DrrImage>,
    pub comparison: Option<DrrComparison>,
    /// Show the signed difference instead of the two images.
    pub show_diff: bool,
    /// Display window as fractions of the image range: (low, high).
    pub window: (f32, f32),
    /// Invert the greyscale, which is how a radiograph is usually read.
    pub invert: bool,
    textures: Vec<TextureHandle>,
    /// Identity of what the textures were built from.
    tex_key: u64,
}

impl DrrDialog {
    fn new(slot: usize, vol: &crate::volume::Volume) -> Self {
        DrrDialog {
            slot,
            params: DrrParams::for_volume(vol),
            both: true,
            images: Vec::new(),
            comparison: None,
            show_diff: false,
            window: (0.0, 1.0),
            invert: true,
            textures: Vec::new(),
            tex_key: 0,
        }
    }
}

impl ViewerApp {
    pub(super) fn open_drr_window(&mut self, slot: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        match &mut self.drr_dialog {
            Some(d) if self.drr_job.is_none() => {
                d.slot = slot;
                d.params.geometry = Geometry {
                    isocenter: d.params.geometry.isocenter,
                    ..Geometry::centered_on(&study.volume)
                };
            }
            Some(_) => {}
            None => self.drr_dialog = Some(DrrDialog::new(slot, &study.volume)),
        }
    }

    fn start_drr(&mut self) {
        if self.drr_job.is_some() {
            return;
        }
        let Some(d) = &self.drr_dialog else { return };
        let Some(study) = &self.slots[d.slot].study else {
            self.error = Some("Load a dataset first".into());
            return;
        };
        let vol = study.volume.clone();
        let params = d.params;
        let both = d.both;
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.drr_job = Some(Job::spawn(progress, move |p| {
            let engines: Vec<Engine> = if both {
                Engine::ALL.to_vec()
            } else {
                vec![params.engine]
            };
            let n = engines.len();
            let mut out = Vec::new();
            for (i, engine) in engines.into_iter().enumerate() {
                p.set_phase(i as f32 / n as f32, 1.0 / n as f32);
                p.set(format!("{} - {} of {n}", engine.label(), i + 1));
                out.push(drr::render(&vol, &DrrParams { engine, ..params }, p)?);
            }
            p.set_phase(0.0, 1.0);
            Ok(out)
        }));
    }

    /// A rendering landed.
    pub(super) fn on_drr_done(&mut self, images: Vec<DrrImage>) {
        let Some(d) = &mut self.drr_dialog else {
            return;
        };
        d.comparison = if images.len() == 2 {
            DrrComparison::of(&images[0], &images[1])
        } else {
            None
        };
        d.images = images;
        d.tex_key = 0;
        d.textures.clear();
    }

    pub(super) fn drr_window(&mut self, ctx: &egui::Context) {
        if self.drr_dialog.is_none() {
            return;
        }
        let mut open = true;
        let mut close = false;
        let mut run = false;
        let mut cancel = false;
        let mut set_iso = false;
        let mut add_to_tree = false;
        let mut beam_pick: Option<(usize, usize)> = None;

        // Read-only facts about the datasets, gathered before the closure.
        let loaded: [bool; 2] = [self.slots[0].has_volume(), self.slots[1].has_volume()];
        let mut d = self.drr_dialog.take().unwrap();
        let beams: Vec<(usize, usize, String)> = self.slots[d.slot]
            .study
            .as_ref()
            .map(|s| {
                s.plans
                    .iter()
                    .enumerate()
                    .flat_map(|(pi, plan)| {
                        plan.beams.iter().enumerate().map(move |(bi, b)| {
                            (
                                pi,
                                bi,
                                format!(
                                    "{} · {} (G {:.1}°)",
                                    plan.label,
                                    b.name,
                                    b.gantry_angle.unwrap_or(0.0)
                                ),
                            )
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let running = self.drr_job.is_some();
        self.refresh_drr_textures(ctx, &mut d);

        detach::tool_window(
            ctx,
            "drr",
            format!("☢ DRR - dataset {}", SLOT_NAMES[d.slot]),
            &mut open,
            detach::WinOpts::width(720.0).no_scroll(),
            |ui| {
                ui.label(
                    "A digitally reconstructed radiograph: the line integral of \
                     attenuation from a point source through the CT onto a flat \
                     detector. Two independent projectors are available, and running \
                     both shows exactly where they disagree.",
                );
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Dataset");
                    for slot in 0..2 {
                        ui.add_enabled_ui(loaded[slot] && !running, |ui| {
                            ui.selectable_value(&mut d.slot, slot, SLOT_NAMES[slot]);
                        });
                    }
                });

                egui::CollapsingHeader::new("Geometry")
                    .id_salt("drr_geom")
                    .default_open(true)
                    .show(ui, |ui| {
                        if !beams.is_empty() {
                            ui.horizontal(|ui| {
                                ui.label("From beam");
                                egui::ComboBox::from_id_salt("drr_beam")
                                    .selected_text("Choose a plan beam")
                                    .width(240.0)
                                    .show_ui(ui, |ui| {
                                        for (pi, bi, label) in &beams {
                                            if ui.selectable_label(false, label).clicked() {
                                                beam_pick = Some((*pi, *bi));
                                            }
                                        }
                                    })
                                    .response
                                    .on_hover_text(
                                        "Take the gantry angle, the couch angle and the \
                                         isocentre from a beam of the loaded plan - the \
                                         beam's-eye view it would actually deliver",
                                    );
                            });
                        }
                        let g = &mut d.params.geometry;
                        ui.horizontal(|ui| {
                            ui.label("Gantry");
                            ui.add(
                                egui::DragValue::new(&mut g.gantry_deg)
                                    .speed(1.0)
                                    .range(-360.0..=360.0)
                                    .suffix("°"),
                            )
                            .on_hover_text("IEC: 0° = source above the patient, 90° = left");
                            ui.label("Couch");
                            ui.add(
                                egui::DragValue::new(&mut g.couch_deg)
                                    .speed(1.0)
                                    .range(-180.0..=180.0)
                                    .suffix("°"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("SAD");
                            ui.add(
                                egui::DragValue::new(&mut g.sad)
                                    .speed(5.0)
                                    .range(100.0..=3000.0)
                                    .suffix(" mm"),
                            );
                            ui.label("SID");
                            ui.add(
                                egui::DragValue::new(&mut g.sid)
                                    .speed(5.0)
                                    .range(100.0..=4000.0)
                                    .suffix(" mm"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Isocentre");
                            for v in [&mut g.isocenter.x, &mut g.isocenter.y, &mut g.isocenter.z] {
                                ui.add(egui::DragValue::new(v).speed(1.0).suffix(" mm"));
                            }
                            if ui
                                .small_button("⌖")
                                .on_hover_text("Take the isocentre from this dataset's crosshair")
                                .clicked()
                            {
                                set_iso = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Panel");
                            ui.add(
                                egui::DragValue::new(&mut g.panel_mm[0])
                                    .speed(5.0)
                                    .range(10.0..=2000.0)
                                    .suffix(" mm"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut g.panel_mm[1])
                                    .speed(5.0)
                                    .range(10.0..=2000.0)
                                    .suffix(" mm"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Pixels");
                            for v in 0..2 {
                                let mut n = g.dims[v] as i64;
                                if ui
                                    .add(egui::DragValue::new(&mut n).speed(8.0).range(16..=2048))
                                    .changed()
                                {
                                    g.dims[v] = n as usize;
                                }
                            }
                            let iso = g.pixel_mm_at_isocenter();
                            ui.weak(format!(
                                "{:.2} × {:.2} mm/px at the isocentre",
                                iso[0], iso[1]
                            ));
                        });
                    });

                egui::CollapsingHeader::new("Projector")
                    .id_salt("drr_engine")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.checkbox(&mut d.both, "Run both and compare")
                            .on_hover_text(
                                "Render the same geometry with each projector and report \
                                 the difference - the honest way to know what either one \
                                 costs you",
                            );
                        if !d.both {
                            ui.horizontal(|ui| {
                                for e in Engine::ALL {
                                    ui.selectable_value(&mut d.params.engine, e, e.label())
                                        .on_hover_text(e.hint());
                                }
                            });
                        } else {
                            for e in Engine::ALL {
                                ui.weak(format!("· {}", e.label()));
                            }
                        }
                        ui.horizontal(|ui| {
                            ui.label("Values");
                            for m in HuMode::ALL {
                                ui.selectable_value(&mut d.params.hu, m, m.label());
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Threshold");
                            ui.add(
                                egui::DragValue::new(&mut d.params.threshold_hu)
                                    .speed(10.0)
                                    .range(-1024.0..=3000.0)
                                    .suffix(" HU"),
                            )
                            .on_hover_text("Voxels below this contribute nothing - air, couch");
                            ui.label("Ray step");
                            ui.add(
                                egui::DragValue::new(&mut d.params.step_mm)
                                    .speed(0.05)
                                    .range(0.05..=5.0)
                                    .suffix(" mm"),
                            )
                            .on_hover_text("Ray-cast sampling step; the exact tracer ignores it");
                        });
                    });

                ui.separator();
                match &self.drr_job {
                    Some(job) => cancel = progress_row(ui, &job.progress),
                    None => {
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(loaded[d.slot], egui::Button::new("▶ Render"))
                                .clicked()
                            {
                                run = true;
                            }
                            if ui
                                .add_enabled(
                                    !d.images.is_empty(),
                                    egui::Button::new(format!(
                                        "➕ Add to dataset {}",
                                        SLOT_NAMES[d.slot]
                                    )),
                                )
                                .on_hover_text(
                                    "File the rendering(s) under Planar images in the data \
                                     tree, with the geometry that produced them - from \
                                     there they open in their own viewer, rename, and \
                                     travel with the dataset",
                                )
                                .clicked()
                            {
                                add_to_tree = true;
                            }
                            if ui.button("Close").clicked() {
                                close = true;
                            }
                        });
                    }
                }

                if d.images.is_empty() {
                    return;
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut d.window.0, 0.0..=1.0).text("black"));
                    ui.add(egui::Slider::new(&mut d.window.1, 0.0..=1.0).text("white"));
                    ui.checkbox(&mut d.invert, "Invert");
                    if d.comparison.is_some() {
                        ui.checkbox(&mut d.show_diff, "Difference");
                    }
                });
                if let Some(c) = &d.comparison {
                    ui.monospace(c.line());
                    ui.weak(
                        "max / mean absolute difference and the Pearson correlation of \
                         the two projectors on this geometry.",
                    );
                }
                let avail = ui.available_width();
                let count = d.textures.len().max(1);
                let side = (avail / count as f32 - 8.0).clamp(80.0, 520.0);
                ui.horizontal(|ui| {
                    for (i, tex) in d.textures.iter().enumerate() {
                        ui.vertical(|ui| {
                            let label = if d.show_diff && d.comparison.is_some() && i == 0 {
                                "Difference".to_string()
                            } else {
                                d.images
                                    .get(i)
                                    .map(|im| im.engine.label().to_string())
                                    .unwrap_or_default()
                            };
                            ui.label(egui::RichText::new(label).strong());
                            ui.image((tex.id(), Vec2::splat(side)));
                            if let Some(im) = d.images.get(i) {
                                ui.weak(im.describe());
                            }
                        });
                    }
                });
            },
        );

        if set_iso {
            if let Some(study) = &self.slots[d.slot].study {
                let c = self.slots[d.slot].cursor;
                d.params.geometry.isocenter = study.volume.voxel_to_patient(c[0], c[1], c[2]);
            }
        }
        if let Some((pi, bi)) = beam_pick {
            if let Some(beam) = self.slots[d.slot]
                .study
                .as_ref()
                .and_then(|s| s.plans.get(pi))
                .and_then(|p| p.beams.get(bi))
            {
                d.params.geometry = d.params.geometry.from_beam(beam);
            }
        }
        if add_to_tree {
            self.add_drr_to_tree(&d);
        }
        if running || (!close && open) {
            self.drr_dialog = Some(d);
        }
        if cancel {
            if let Some(job) = &self.drr_job {
                job.progress.cancel();
            }
        }
        if run {
            self.start_drr();
        }
    }

    /// File the current rendering(s) under the source dataset's planar
    /// images. Labels are made unique on the way in, because rendering the
    /// same geometry twice is exactly what one does while tuning it.
    fn add_drr_to_tree(&mut self, d: &DrrDialog) {
        let made: Vec<crate::extras::PlanarImage> = d
            .images
            .iter()
            .map(|im| im.to_planar(&d.params, d.invert))
            .collect();
        let Some(study) = self.slots[d.slot].study.as_mut() else {
            self.error = Some(format!("dataset {} is not loaded", SLOT_NAMES[d.slot]));
            return;
        };
        let n = made.len();
        for mut img in made {
            let base = img.label.clone();
            let mut k = 2;
            while study.planar_images.iter().any(|e| e.label == img.label) {
                img.label = format!("{base} #{k}");
                k += 1;
            }
            study.planar_images.push(img);
        }
        self.settings_gen += 1;
        self.notice = Some(format!(
            "✔ {n} radiograph(s) added to dataset {} - see Planar images in the tree",
            SLOT_NAMES[d.slot]
        ));
    }

    /// Rebuild the display textures when the images or the window changed.
    fn refresh_drr_textures(&mut self, ctx: &egui::Context, d: &mut DrrDialog) {
        if d.images.is_empty() {
            return;
        }
        let mut key: u64 = 0x51ED_270B_u64;
        for v in [
            d.images.len() as u64,
            d.window.0.to_bits() as u64,
            d.window.1.to_bits() as u64,
            d.invert as u64,
            d.show_diff as u64,
            d.images[0].elapsed_secs.to_bits(),
        ] {
            key = (key ^ v).wrapping_mul(0x100000001b3);
        }
        if key == d.tex_key && d.textures.len() == expected_textures(d) {
            return;
        }
        d.tex_key = key;
        d.textures.clear();

        if d.show_diff && d.images.len() == 2 {
            let a = &d.images[0];
            let b = &d.images[1];
            let diff: Vec<f32> = a.pixels.iter().zip(&b.pixels).map(|(x, y)| x - y).collect();
            let m = diff
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.abs()))
                .max(1e-9);
            // Signed difference: blue below, red above, grey at zero.
            let pixels: Vec<Color32> = diff
                .iter()
                .map(|v| {
                    let t = (v / m).clamp(-1.0, 1.0);
                    let g = (110.0 + 60.0 * t.abs()) as u8;
                    if t >= 0.0 {
                        Color32::from_rgb(g, (110.0 * (1.0 - t)) as u8, (110.0 * (1.0 - t)) as u8)
                    } else {
                        Color32::from_rgb((110.0 * (1.0 + t)) as u8, (110.0 * (1.0 + t)) as u8, g)
                    }
                })
                .collect();
            let img = ColorImage::new([a.dims[0], a.dims[1]], pixels);
            d.textures
                .push(ctx.load_texture("drr_diff", img, TextureOptions::LINEAR));
            return;
        }

        for (i, im) in d.images.iter().enumerate() {
            let lo = im.min + (im.max - im.min) * d.window.0;
            let hi = im.min + (im.max - im.min) * d.window.1;
            let span = (hi - lo).abs().max(1e-9);
            let pixels: Vec<Color32> = im
                .pixels
                .iter()
                .map(|v| {
                    let t = ((v - lo) / span).clamp(0.0, 1.0);
                    let t = if d.invert { 1.0 - t } else { t };
                    Color32::from_gray((t * 255.0) as u8)
                })
                .collect();
            let img = ColorImage::new([im.dims[0], im.dims[1]], pixels);
            d.textures
                .push(ctx.load_texture(format!("drr_{i}"), img, TextureOptions::LINEAR));
        }
    }
}

/// How many textures the current display mode needs.
fn expected_textures(d: &DrrDialog) -> usize {
    if d.show_diff && d.images.len() == 2 {
        1
    } else {
        d.images.len()
    }
}
