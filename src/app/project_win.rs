//! *⇩ Project structures*: the picker that projects ticked structures of
//! the CT onto ticked planar images as 2D contours
//! ([`crate::planar_structs::project_many`]).
//!
//! A small dialog over the window, not an OS window: the structures (and
//! *All*), the images that have a camera (or that a setup placed), the pose
//! (as rendered / at the plan, or a setup of the workspace), the set the
//! contours go into, and *⇩ Project*, which runs in the background and lands
//! the set. Reached from the DRR section, the planar image rows of the data
//! tree, the *Planar structures* section and the 2D/3D section.

use super::*;

use crate::planar_structs::{PoseChoice, ProjectRequest};

/// The dialog's state.
pub(super) struct ProjectPrompt {
    slot: usize,
    /// Name, colour, ticked.
    structures: Vec<(String, [u8; 3], bool)>,
    /// Planar image index and ticked.
    images: Vec<(usize, bool)>,
    pose: PoseChoice,
    /// An existing set of the workspace, or a new one.
    into: Option<usize>,
    label: String,
}

impl ViewerApp {
    /// Open the dialog on `slot`, with `images` ticked, at `pose`, into
    /// `into` (a new set when `None`).
    pub(super) fn open_project_prompt(
        &mut self,
        slot: usize,
        images: &[usize],
        pose: PoseChoice,
        into: Option<usize>,
    ) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if !study.has_volume() {
            self.error = Some("Projecting structures needs the CT of the workspace".into());
            return;
        }
        let mut structures: Vec<(String, [u8; 3], bool)> = Vec::new();
        for e in crate::workflow::select::list(study) {
            if !structures.iter().any(|(n, _, _)| *n == e.name) {
                structures.push((e.name, e.color, false));
            }
        }
        let list: Vec<(usize, bool)> = study
            .planar_images
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == crate::extras::PlanarKind::Projection)
            .map(|(i, _)| (i, images.contains(&i)))
            .collect();
        self.planar2d.last_lines.clear();
        self.planar2d.prompt = Some(ProjectPrompt {
            slot,
            structures,
            images: list,
            pose,
            into,
            label: String::new(),
        });
    }

    /// Can planar image `i` take a projection at `pose`?
    fn projectable(study: &LoadedStudy, i: usize, pose: &PoseChoice) -> bool {
        crate::planar_structs::camera_for(study, i, pose).is_ok()
    }

    pub(super) fn project_window(&mut self, ctx: &egui::Context) {
        let Some(mut d) = self.planar2d.prompt.take() else {
            return;
        };
        let Some(study) = &self.slots[d.slot].study else {
            return;
        };
        let mut open = true;
        let mut run = false;
        let mut close = ctx.input(|i| i.key_pressed(egui::Key::Escape));
        let busy = self.planar2d.project_job.is_some();
        egui::Window::new("⇩ Project structures")
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_width(420.0)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.columns(2, |cols| {
                    cols[0].strong("Structures");
                    cols[0].horizontal(|ui| {
                        if ui.small_button("All").clicked() {
                            d.structures.iter_mut().for_each(|s| s.2 = true);
                        }
                        if ui.small_button("None").clicked() {
                            d.structures.iter_mut().for_each(|s| s.2 = false);
                        }
                    });
                    egui::ScrollArea::vertical()
                        .id_salt("project_structs")
                        .max_height(220.0)
                        .show(&mut cols[0], |ui| {
                            for (name, color, on) in &mut d.structures {
                                ui.horizontal(|ui| {
                                    let (r, _) = ui.allocate_exact_size(
                                        Vec2::splat(10.0),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(
                                        r,
                                        2.0,
                                        Color32::from_rgb(color[0], color[1], color[2]),
                                    );
                                    ui.checkbox(on, name.as_str());
                                });
                            }
                        });
                    cols[1].strong("Images");
                    egui::ScrollArea::vertical()
                        .id_salt("project_images")
                        .max_height(240.0)
                        .show(&mut cols[1], |ui| {
                            for (i, on) in &mut d.images {
                                let Some(img) = study.planar_images.get(*i) else {
                                    continue;
                                };
                                let ok = Self::projectable(study, *i, &d.pose);
                                if !ok {
                                    *on = false;
                                }
                                ui.add_enabled(ok, egui::Checkbox::new(on, img.label.as_str()))
                                    .on_disabled_hover_text("no geometry");
                            }
                        });
                });
                ui.separator();
                form::form(ui, "project_form", |f| {
                    f.row("Pose", |ui| {
                        let name = |p: &PoseChoice| match p {
                            PoseChoice::AsIs => "As rendered / at the plan".to_string(),
                            PoseChoice::Setup(k) => study
                                .setups
                                .get(*k)
                                .map(|s| format!("At {}", s.label))
                                .unwrap_or_else(|| "?".into()),
                        };
                        egui::ComboBox::from_id_salt("project_pose")
                            .selected_text(name(&d.pose))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut d.pose,
                                    PoseChoice::AsIs,
                                    name(&PoseChoice::AsIs),
                                );
                                for k in 0..study.setups.len() {
                                    let p = PoseChoice::Setup(k);
                                    let n = name(&p);
                                    ui.selectable_value(&mut d.pose, p, n);
                                }
                            });
                    });
                    f.row("Into", |ui| {
                        let name = |k: Option<usize>| match k {
                            None => "a new set".to_string(),
                            Some(k) => study
                                .planar_structs
                                .get(k)
                                .map(|s| s.label.clone())
                                .unwrap_or_default(),
                        };
                        egui::ComboBox::from_id_salt("project_into")
                            .selected_text(name(d.into))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut d.into, None, name(None));
                                for (k, s) in study.planar_structs.iter().enumerate() {
                                    if !s.locked {
                                        ui.selectable_value(&mut d.into, Some(k), name(Some(k)));
                                    }
                                }
                            });
                    });
                    if d.into.is_none() {
                        f.row("Label", |ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut d.label)
                                    .desired_width(200.0)
                                    .hint_text("after the pose"),
                            );
                        });
                    }
                });
                let n_s = d.structures.iter().filter(|s| s.2).count();
                let n_i = d.images.iter().filter(|s| s.1).count();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !busy && n_s > 0 && n_i > 0,
                            egui::Button::new(format!("⇩ Project {n_s} onto {n_i}")),
                        )
                        .clicked()
                    {
                        run = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
                if let Some(job) = &self.planar2d.project_job {
                    if progress_row(ui, &job.progress) {
                        job.progress.cancel();
                    }
                }
                for l in &self.planar2d.last_lines {
                    ui.weak(l);
                }
            });
        if run {
            let req = ProjectRequest {
                structures: d
                    .structures
                    .iter()
                    .filter(|s| s.2)
                    .map(|s| s.0.clone())
                    .collect(),
                images: d.images.iter().filter(|s| s.1).map(|s| s.0).collect(),
                pose: d.pose.clone(),
                label: d.label.trim().to_string(),
            };
            self.start_projection(d.slot, req, d.into);
        }
        if open && !close {
            self.planar2d.prompt = Some(d);
        }
    }
}
