//! *Modules ▶ Image generation ▶ DRR*: digitally reconstructed radiographs
//! of the module's workspace, filed straight into its data tree.
//!
//! The section sets the geometry (typed in, or taken from a plan beam) and
//! the projector, and **▶ Render** does the rest: every rendering becomes a
//! planar image of the workspace (*Planar images* in the tree, an RTIMAGE
//! with the geometry as its info rows, [`crate::drr::DrrImage::to_planar`])
//! and comes up in the row's planar panes. There is no window and no
//! preview to add from: what is rendered is in the study, to be looked at,
//! renamed, removed or compared there.
//!
//! *Run both and compare* renders the same geometry with each projector -
//! plastimatch's exact tracer and ITK's interpolating ray-cast - and puts
//! them side by side in *Planar 1* and *Planar 2*; the comparison line
//! (largest and mean absolute difference, Pearson r) stays in the section,
//! and *Also file the difference* adds the signed difference as a third
//! planar image. The comparison is not decoration: the two are the answers
//! the field actually uses, and where they part company shows what a step
//! size, a detector resolution or a threshold is doing.

use crate::drr::{self, DrrComparison, DrrImage, DrrParams, Engine, Geometry, HuMode};
use crate::volume::Volume;

use super::image_gen::DRR_TOOL;
use super::*;

/// The DRR section's state, made the first time the section is drawn.
pub(super) struct DrrState {
    /// Workspace the radiographs are computed from: the module's.
    pub slot: usize,
    pub params: DrrParams,
    /// Render both projectors and compare them.
    pub both: bool,
    /// Store the greyscale inverted - dark is high attenuation - which is
    /// how a radiograph is read.
    pub invert: bool,
    /// With both projectors, file their signed difference too.
    pub file_diff: bool,
    /// The comparison of the last run with both projectors.
    pub comparison: Option<DrrComparison>,
    /// What the last run did, for the line under the button.
    pub status: Option<String>,
    /// One line per rendering of the last run: projector, raster, spacing,
    /// value range and time.
    pub details: Vec<String>,
    /// The planar panes the last run showed its renderings in, so the next
    /// one replaces them rather than whatever else is on display - a
    /// radiograph put beside the DRR stays where it is while the DRR is
    /// tuned.
    pub last_panes: Vec<usize>,
    /// Where an exact camera in `params.camera` came from, and the geometry
    /// it was taken with: editing the geometry drops the camera.
    pub camera_from: Option<(String, Geometry)>,
    /// The station's calibrated rooms, read the first time they are asked
    /// for, and why they could not be read.
    pub rooms: Option<(crate::rooms::RoomsFile, Option<String>)>,
}

impl DrrState {
    fn new(slot: usize, vol: &Volume) -> Self {
        DrrState {
            slot,
            params: DrrParams::for_volume(vol),
            both: false,
            invert: true,
            file_diff: false,
            comparison: None,
            status: None,
            details: Vec::new(),
            last_panes: Vec::new(),
            camera_from: None,
            rooms: None,
        }
    }

    /// Point the section at another workspace's volume: the isocentre moves
    /// to that volume's centre (a different patient, or the same one in
    /// another frame), everything else stays as it was set.
    fn retarget(&mut self, slot: usize, vol: &Volume) {
        self.slot = slot;
        self.params.geometry.isocenter = Geometry::centered_on(vol).isocenter;
        self.params.camera = None;
        self.camera_from = None;
        self.comparison = None;
        self.status = None;
        self.details.clear();
        self.last_panes.clear();
    }
}

/// A finished render, with what it was asked for so it can land even when
/// the section has been changed meanwhile.
pub(super) struct DrrRun {
    slot: usize,
    params: DrrParams,
    invert: bool,
    file_diff: bool,
    /// The volume it was rendered from, for the stale-workspace check.
    dims: [usize; 3],
    for_uid: String,
    images: Vec<DrrImage>,
}

impl ViewerApp {
    /// Make the section's state, or point it at the module's workspace when
    /// that changed and no render is in flight.
    fn drr_follow_module(&mut self) {
        let slot = self.gen.slot;
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if !study.has_volume() {
            return;
        }
        // The patient lies as the CT says he did, until the section is told
        // otherwise.
        let position = study
            .series
            .get(study.active_series)
            .and_then(crate::loader::series_patient_position);
        let frame = |p: Option<crate::iec::PatientPosition>| {
            p.map(crate::iec::PatientFrame::Position)
                .unwrap_or_default()
        };
        match &mut self.drr {
            None => {
                let mut d = DrrState::new(slot, &study.volume);
                d.params.geometry.patient = frame(position);
                self.drr = Some(d);
            }
            Some(d) if d.slot != slot && self.drr_job.is_none() => {
                d.retarget(slot, &study.volume);
                d.params.geometry.patient = frame(position);
            }
            Some(_) => {}
        }
    }

    fn start_drr(&mut self) {
        if self.drr_job.is_some() {
            return;
        }
        let Some(d) = &self.drr else { return };
        let Some(study) = &self.slots[d.slot].study else {
            self.error = Some("Load a workspace first".into());
            return;
        };
        let vol = study.volume.clone();
        let (slot, params, both, invert) = (d.slot, d.params, d.both, d.invert);
        let file_diff = both && d.file_diff;
        let (dims, for_uid) = (vol.dims, vol.frame_of_reference_uid.clone());
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.drr_job = Some(Job::spawn(progress, move |p| {
            let engines: Vec<Engine> = if both {
                Engine::ALL.to_vec()
            } else {
                vec![params.engine]
            };
            let n = engines.len();
            let mut images = Vec::new();
            for (i, engine) in engines.into_iter().enumerate() {
                p.set_phase(i as f32 / n as f32, 1.0 / n as f32);
                p.set(format!("{} - {} of {n}", engine.label(), i + 1));
                images.push(drr::render(&vol, &DrrParams { engine, ..params }, p)?);
            }
            p.set_phase(0.0, 1.0);
            Ok(DrrRun {
                slot,
                params,
                invert,
                file_diff,
                dims,
                for_uid,
                images,
            })
        }));
    }

    /// A render landed: file every image under the workspace's planar
    /// images and show it in the row.
    pub(super) fn on_drr_done(&mut self, run: DrrRun) {
        if !self.slot_still_shows(run.slot, run.dims, &run.for_uid) {
            self.error = Some(stale_result(&DRR_TOOL));
            return;
        }
        let details: Vec<String> = run
            .images
            .iter()
            .map(|im| format!("{}: {}", im.engine.short(), im.describe()))
            .collect();
        let comparison = (run.images.len() == 2)
            .then(|| DrrComparison::of(&run.images[0], &run.images[1]))
            .flatten();
        let mut made: Vec<crate::extras::PlanarImage> = run
            .images
            .iter()
            .map(|im| {
                im.to_planar(
                    &DrrParams {
                        engine: im.engine,
                        ..run.params
                    },
                    run.invert,
                )
            })
            .collect();
        let shown = made.len();
        if run.file_diff && run.images.len() == 2 {
            made.extend(run.images[0].difference_to_planar(&run.images[1], &run.params));
        }
        let Some(study) = self.slots[run.slot].study.as_mut() else {
            return;
        };
        let mut filed = Vec::new();
        let study_uid = study.displayed_study_uid();
        for mut img in made {
            img.frame_of_reference_uid = run.for_uid.clone();
            // Under the study of the CT it was rendered from.
            img.study_uid = study_uid.clone();
            // Rendering the same geometry twice is exactly what one does
            // while tuning it, so the labels are made unique on the way in.
            let base = img.label.clone();
            let mut k = 2;
            while study.planar_images.iter().any(|e| e.label == img.label) {
                img.label = format!("{base} #{k}");
                k += 1;
            }
            study.planar_images.push(img);
            filed.push(study.planar_images.len() - 1);
        }
        self.settings_gen += 1;

        // On screen: both renderings side by side, or the one in the pane
        // the last render used.
        let panes: Vec<Option<usize>> = if shown == 2 {
            vec![Some(0), Some(1)]
        } else {
            let last = self
                .drr
                .as_ref()
                .and_then(|d| d.last_panes.first().copied())
                .filter(|k| {
                    self.row_layout(run.slot)
                        .contains(&PaneKind::Planar(*k as u8))
                });
            vec![last]
        };
        let mut used = Vec::new();
        for (idx, pane) in filed.iter().take(shown).zip(panes) {
            self.show_planar(run.slot, *idx, pane);
            used.extend(self.planar_panes_showing(run.slot, *idx).first().copied());
        }
        self.shown[run.slot] = true;
        if self.maximized.is_some_and(|(s, k)| {
            s != run.slot || !matches!(k, PaneKind::Planar(_) | PaneKind::Additional)
        }) {
            self.maximized = None;
        }
        if let Some(d) = &mut self.drr {
            d.comparison = comparison;
            d.details = details;
            d.last_panes = used;
            let n = filed.len();
            d.status = Some(format!(
                "✔ {n} {} filed under Planar images of workspace {}",
                if n == 1 { "image" } else { "images" },
                SLOT_NAMES[run.slot]
            ));
        }
    }

    /// The DRR section of the Image generation module.
    pub(super) fn drr_section(&mut self, ui: &mut egui::Ui) {
        self.drr_follow_module();
        let Some(mut d) = self.drr.take() else {
            ui.weak("Load a workspace with a CT to render a radiograph from");
            return;
        };
        let mut run = false;
        let mut cancel = false;
        let mut set_iso = false;
        let mut beam_pick: Option<(usize, usize)> = None;
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
        let ready = self.slots[d.slot].has_volume();
        // The workspace's RT images that state their geometry.
        let rt_images: Vec<(usize, String, bool)> = self.slots[d.slot]
            .study
            .as_ref()
            .map(|s| {
                s.radiographs()
                    .filter(|(_, p)| p.iec.is_some())
                    .map(|(i, p)| (i, p.label.clone(), p.camera.is_some()))
                    .collect()
            })
            .unwrap_or_default();
        let mut rt_pick: Option<usize> = None;
        let mut room_pick: Option<(usize, usize)> = None;
        if d.rooms.is_none() {
            d.rooms = Some(crate::rooms::load_default());
        }
        let room_views: Vec<(usize, usize, String)> = d
            .rooms
            .as_ref()
            .map(|(f, _)| {
                f.rooms
                    .iter()
                    .enumerate()
                    .flat_map(|(ri, r)| {
                        r.views
                            .iter()
                            .enumerate()
                            .map(move |(vi, v)| (ri, vi, format!("{} · {}", r.name, v.name)))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let rooms_error = d.rooms.as_ref().and_then(|(_, e)| e.clone());
        let mut drop_camera = false;

        egui::CollapsingHeader::new("Geometry")
            .id_salt("drr_geom")
            .default_open(true)
            .show(ui, |ui| {
                let g = &mut d.params.geometry;
                form::form(ui, "drr_geom_form", |f| {
                    if !beams.is_empty() {
                        f.row_tip(
                            "From beam",
                            "Take the gantry angle, the couch angle and the isocentre \
                             from a beam of the loaded plan - the beam's eye view it \
                             would actually deliver",
                            |ui| {
                                egui::ComboBox::from_id_salt("drr_beam")
                                    .selected_text("Choose a plan beam")
                                    .width(180.0)
                                    .show_ui(ui, |ui| {
                                        for (pi, bi, label) in &beams {
                                            if ui.selectable_label(false, label).clicked() {
                                                beam_pick = Some((*pi, *bi));
                                            }
                                        }
                                    });
                            },
                        );
                    }
                    if !rt_images.is_empty() {
                        f.row_tip(
                            "From RT image",
                            "Render exactly the view of an RT image of this workspace: its \
                             gantry, couch, distances, receptor and pixels",
                            |ui| {
                                egui::ComboBox::from_id_salt("drr_rtimage")
                                    .selected_text("Choose an RT image")
                                    .width(180.0)
                                    .show_ui(ui, |ui| {
                                        for (i, label, placed) in &rt_images {
                                            let text = if *placed {
                                                label.clone()
                                            } else {
                                                format!("{label} (no isocentre)")
                                            };
                                            if ui.selectable_label(false, text).clicked() {
                                                rt_pick = Some(*i);
                                            }
                                        }
                                    });
                            },
                        );
                    }
                    if !room_views.is_empty() {
                        f.row_tip(
                            "From room view",
                            "A calibrated view of imaging-rooms.json, about this isocentre",
                            |ui| {
                                egui::ComboBox::from_id_salt("drr_room")
                                    .selected_text("Choose a room view")
                                    .width(180.0)
                                    .show_ui(ui, |ui| {
                                        for (ri, vi, label) in &room_views {
                                            if ui.selectable_label(false, label).clicked() {
                                                room_pick = Some((*ri, *vi));
                                            }
                                        }
                                    });
                            },
                        );
                    }
                    if let Some(e) = &rooms_error {
                        f.wide(|ui| {
                            ui.colored_label(ui.visuals().warn_fg_color, "imaging-rooms.json")
                                .on_hover_text(e.as_str());
                        });
                    }
                    f.row_tip("Patient", "How the patient lies or sits", |ui| {
                        use crate::iec::{PatientFrame, PatientPosition};
                        let text = g.patient.describe();
                        egui::ComboBox::from_id_salt("drr_patient")
                            .selected_text(text)
                            .width(120.0)
                            .show_ui(ui, |ui| {
                                for p in PatientPosition::ALL {
                                    let on = g.patient == PatientFrame::Position(p);
                                    if ui.selectable_label(on, p.code()).clicked() {
                                        g.patient = PatientFrame::Position(p);
                                    }
                                }
                                let up = matches!(g.patient, PatientFrame::Upright { .. });
                                if ui.selectable_label(up, "Upright").clicked() && !up {
                                    g.patient = PatientFrame::Upright { facing_deg: 0.0 };
                                }
                            });
                        if let PatientFrame::Upright { facing_deg } = &mut g.patient {
                            ui.add(
                                egui::DragValue::new(facing_deg)
                                    .speed(1.0)
                                    .range(-360.0..=360.0)
                                    .suffix("°"),
                            )
                            .on_hover_text("The direction the patient faces, from the nozzle");
                        }
                    });
                    f.row_tip(
                        "Gantry · couch",
                        "IEC: 0° = source above the patient, 90° = left",
                        |ui| {
                            ui.add(
                                egui::DragValue::new(&mut g.gantry_deg)
                                    .speed(1.0)
                                    .range(-360.0..=360.0)
                                    .suffix("°"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut g.couch_deg)
                                    .speed(1.0)
                                    .range(-180.0..=180.0)
                                    .suffix("°"),
                            );
                        },
                    );
                    f.row_tip("Pitch · roll", "Table-top pitch and roll", |ui| {
                        for v in [&mut g.pitch_deg, &mut g.roll_deg] {
                            ui.add(
                                egui::DragValue::new(v)
                                    .speed(0.1)
                                    .range(-20.0..=20.0)
                                    .suffix("°"),
                            );
                        }
                    });
                    f.row_tip("SAD · SID", "Source-axis and source-image distance", |ui| {
                        ui.add(
                            egui::DragValue::new(&mut g.sad)
                                .speed(5.0)
                                .range(100.0..=3000.0)
                                .suffix(" mm"),
                        );
                        ui.add(
                            egui::DragValue::new(&mut g.sid)
                                .speed(5.0)
                                .range(100.0..=4000.0)
                                .suffix(" mm"),
                        );
                    });
                    // Millimetres in the label, not after every number: the
                    // three coordinates and the crosshair button have to fit
                    // the panel's width on one line.
                    f.row_tip(
                        "Isocentre, mm",
                        "x, y, z in patient coordinates (LPS)",
                        |ui| {
                            for v in [&mut g.isocenter.x, &mut g.isocenter.y, &mut g.isocenter.z] {
                                ui.add(egui::DragValue::new(v).speed(1.0).max_decimals(1));
                            }
                            if small_tip_button(
                                ui,
                                "⌖",
                                "Take the isocentre from this workspace's crosshair",
                            ) {
                                set_iso = true;
                            }
                        },
                    );
                    f.row("Panel", |ui| {
                        for v in 0..2 {
                            ui.add(
                                egui::DragValue::new(&mut g.panel_mm[v])
                                    .speed(5.0)
                                    .range(10.0..=2000.0)
                                    .suffix(" mm"),
                            );
                        }
                    });
                    f.row("Pixels", |ui| {
                        for v in 0..2 {
                            let mut n = g.dims[v] as i64;
                            if ui
                                .add(egui::DragValue::new(&mut n).speed(8.0).range(16..=2048))
                                .changed()
                            {
                                g.dims[v] = n as usize;
                            }
                        }
                    });
                    let iso = g.pixel_mm_at_isocenter();
                    f.wide(|ui| {
                        ui.weak(format!(
                            "{:.2} × {:.2} mm per pixel at the isocentre",
                            iso[0], iso[1]
                        ));
                    });
                    if let Some((name, _)) = &d.camera_from {
                        f.wide(|ui| {
                            ui.horizontal(|ui| {
                                ui.weak(format!("Camera of {name}"));
                                if small_tip_button(ui, "✖", "Back to the parameters above") {
                                    drop_camera = true;
                                }
                            });
                        });
                    }
                });
            });

        egui::CollapsingHeader::new("Projector")
            .id_salt("drr_engine")
            .default_open(true)
            .show(ui, |ui| {
                ui.checkbox(&mut d.both, "Run both and compare")
                    .on_hover_text(
                        "Render the same geometry with each projector, show the two side by \
                     side in the planar panes and report the difference - the honest way \
                     to know what either one costs you",
                    );
                if d.both {
                    ui.checkbox(&mut d.file_diff, "Also file the difference")
                        .on_hover_text(
                            "File the signed difference of the two (exact minus ray-cast) as a \
                         planar image as well: mid-grey where they agree",
                        );
                } else {
                    ui.horizontal_wrapped(|ui| {
                        for e in Engine::ALL {
                            ui.selectable_value(&mut d.params.engine, e, e.label())
                                .on_hover_text(e.hint());
                        }
                    });
                }
                form::form(ui, "drr_engine_form", |f| {
                    f.row("Values", |ui| {
                        egui::ComboBox::from_id_salt("drr_hu")
                            .selected_text(d.params.hu.label())
                            .width(170.0)
                            .show_ui(ui, |ui| {
                                for m in HuMode::ALL {
                                    ui.selectable_value(&mut d.params.hu, m, m.label());
                                }
                            });
                    });
                    f.row_tip(
                        "Threshold",
                        "Voxels below this contribute nothing - air, couch",
                        |ui| {
                            ui.add(
                                egui::DragValue::new(&mut d.params.threshold_hu)
                                    .speed(10.0)
                                    .range(-1024.0..=3000.0)
                                    .suffix(" HU"),
                            );
                        },
                    );
                    f.row_tip(
                        "Ray step",
                        "Ray-cast sampling step; the exact tracer ignores it",
                        |ui| {
                            ui.add(
                                egui::DragValue::new(&mut d.params.step_mm)
                                    .speed(0.05)
                                    .range(0.05..=5.0)
                                    .suffix(" mm"),
                            );
                        },
                    );
                });
                ui.checkbox(&mut d.invert, "Invert: dark is high attenuation")
                    .on_hover_text(
                        "Store the image the way a radiograph is read. Off, the values are \
                         the line integral itself, bright where attenuation is high.",
                    );
            });

        ui.separator();
        match &self.drr_job {
            Some(job) => cancel = progress_row(ui, &job.progress),
            None => {
                if ui
                    .add_enabled(ready, egui::Button::new("▶ Render"))
                    .on_hover_text(
                        "Render, file the result under Planar images of the workspace and \
                         show it in the row's planar panes",
                    )
                    .clicked()
                {
                    run = true;
                }
            }
        }
        if let Some(s) = &d.status {
            ui.weak(s);
        }
        for line in &d.details {
            ui.weak(line);
        }
        if let Some(c) = &d.comparison {
            ui.monospace(c.line()).on_hover_text(
                "Largest and mean absolute difference and the Pearson correlation of the \
                 two projectors on this geometry",
            );
        }

        if let Some(i) = rt_pick {
            if let Some(img) = self.slots[d.slot]
                .study
                .as_ref()
                .and_then(|s| s.planar_images.get(i))
            {
                if let Some(g) = img.iec {
                    // Without an isocentre of its own the image keeps the
                    // one set here.
                    let iso = d.params.geometry.isocenter;
                    d.params.geometry = g;
                    if img.camera.is_none() {
                        d.params.geometry.isocenter = iso;
                    }
                    d.params.camera = img.camera;
                    d.camera_from = img.camera.map(|_| (img.label.clone(), d.params.geometry));
                }
            }
        }
        if let Some((ri, vi)) = room_pick {
            let found = d.rooms.as_ref().and_then(|(f, _)| {
                let room = f.rooms.get(ri)?;
                Some((
                    room.patient_frame(),
                    room.views.get(vi)?.clone(),
                    room.name.clone(),
                ))
            });
            if let Some((frame, view, room)) = found {
                let g = &mut d.params.geometry;
                g.patient = frame;
                if let Some(cam) = view.camera_in_patient(
                    &frame,
                    g.isocenter,
                    g.couch_deg,
                    g.pitch_deg,
                    g.roll_deg,
                ) {
                    g.dims = cam.dims;
                    let sp = cam.spacing();
                    g.panel_mm = [sp[0] * cam.dims[0] as f64, sp[1] * cam.dims[1] as f64];
                    d.params.camera = Some(cam);
                    d.camera_from = Some((format!("{room} · {}", view.name), d.params.geometry));
                }
            }
        }
        // A camera taken from an image or a room holds while the parameters
        // it came with do; an edit is a request for the parameters instead.
        if drop_camera
            || d.camera_from
                .as_ref()
                .is_some_and(|(_, g)| *g != d.params.geometry)
        {
            d.params.camera = None;
            d.camera_from = None;
        }
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
        self.drr = Some(d);
        cancel_if(cancel, &self.drr_job);
        if run {
            self.start_drr();
        }
    }
}
