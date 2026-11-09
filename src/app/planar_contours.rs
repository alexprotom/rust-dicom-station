//! The 2D contours of the planar panes ([`crate::planar_structs`]): drawn
//! over every image of a pane's stack, projected from the CT's structures
//! in the background, and filed with the workspace.
//!
//! Each ring is in its image's pixels (centre of the first pixel at `(0,
//! 0)`) and is placed on screen through that image's own rectangle in the
//! pane, so a contour on a DRR rendered at a radiograph's geometry lies on
//! the radiograph it is stacked with. The image under the scrubber gets its
//! contours in full colour, the others of the stack fainter, the structure
//! being edited thicker (as `focused_rois` does in the MPR views).

use super::*;

use crate::planar_structs::{self, PlanarStructureSet, ProjectRequest, ProjectedLine};

/// The workspace state of the planar contours: the projection prompt and
/// its job, the contour tools on a planar target ([`super::planar_tools`]),
/// and the 2D/2D registration section ([`super::reg2d`]).
#[derive(Default)]
pub(super) struct Planar2d {
    pub(super) prompt: Option<super::project_win::ProjectPrompt>,
    pub(super) project_job: Option<Job<(usize, anyhow::Result<ProjectOut>)>>,
    /// The lines of the last projection, for the prompt's report.
    pub(super) last_lines: Vec<String>,
    /// The workspace whose planar pane took the last stroke of a contour
    /// tool: while set, the toolbar row, Ctrl+Z, Esc and Enter speak of the
    /// planar structure, not of the volume's (a stroke in an MPR view
    /// clears it).
    pub(super) target: Option<usize>,
    /// The contour under construction (polygon, spline, freehand, live
    /// wire).
    pub(super) draw: Option<super::planar_tools::PlanarDraw>,
    /// Undo snapshots of planar structures, newest last.
    pub(super) undo: Vec<super::planar_tools::PlanarSnapshot>,
    /// The contour brush's and the nudge's last sample, whether the brush
    /// stroke cuts (decided at its start) and whether the stroke has taken
    /// its undo snapshot.
    pub(super) last_sample: Option<[f64; 2]>,
    pub(super) brush_cut: bool,
    pub(super) stroke_saved: bool,
    /// The live wire on the image being drawn on.
    pub(super) wire: Option<super::planar_tools::PlanarWire>,
    /// The 2D/2D registration section and its run.
    pub(super) reg: Option<super::reg2d::Reg2dState>,
    pub(super) reg_job: Option<Job<(usize, anyhow::Result<super::reg2d::Reg2dOut>)>>,
}

/// A projection that finished.
pub(super) struct ProjectOut {
    pub(super) set: PlanarStructureSet,
    pub(super) lines: Vec<ProjectedLine>,
    /// File into this set of the workspace rather than a new one.
    pub(super) into: Option<usize>,
    /// What it was run on: the stale-workspace check.
    pub(super) dims: [usize; 3],
    pub(super) for_uid: String,
}

/// The key a planar set is remembered by in the workspace's visibility
/// state: its UID, else its label.
pub(super) fn set_key(set: &PlanarStructureSet) -> String {
    if set.sop_instance_uid.is_empty() {
        format!("label:{}", set.label)
    } else {
        set.sop_instance_uid.clone()
    }
}

/// A point of an image's pixels on screen, the image drawn in `on`.
pub(super) fn px_to_screen(p: [f64; 2], dims: [usize; 2], on: Rect) -> Pos2 {
    Pos2::new(
        on.left() + ((p[0] + 0.5) / dims[0].max(1) as f64) as f32 * on.width(),
        on.top() + ((p[1] + 0.5) / dims[1].max(1) as f64) as f32 * on.height(),
    )
}

/// The inverse of [`px_to_screen`].
pub(super) fn screen_to_px(s: Pos2, dims: [usize; 2], on: Rect) -> [f64; 2] {
    [
        ((s.x - on.left()) / on.width().max(1e-6)) as f64 * dims[0] as f64 - 0.5,
        ((s.y - on.top()) / on.height().max(1e-6)) as f64 * dims[1] as f64 - 0.5,
    ]
}

impl ViewerApp {
    /// Is structure `name` of planar set `set` drawn?
    pub(super) fn planar_visible(&self, slot: usize, set: &PlanarStructureSet, name: &str) -> bool {
        !self.slots[slot]
            .planar_hidden
            .contains(&(set_key(set), name.to_string()))
    }

    pub(super) fn set_planar_visible(&mut self, slot: usize, set: usize, name: &str, on: bool) {
        let Some(key) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_structs.get(set))
            .map(set_key)
        else {
            return;
        };
        let k = (key, name.to_string());
        if on {
            self.slots[slot].planar_hidden.remove(&k);
        } else {
            self.slots[slot].planar_hidden.insert(k);
        }
    }

    /// The contours of every planar set over the images of a pane's stack:
    /// `(image, its rectangle on screen, under the scrubber)`.
    pub(super) fn draw_planar_contours(
        &self,
        painter: &egui::Painter,
        slot: usize,
        layers: &[(usize, Rect, bool)],
    ) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if study.planar_structs.is_empty() {
            return;
        }
        let active = self.slots[slot].active_planar;
        for &(image, on, near) in layers {
            let Some(img) = study.planar_images.get(image) else {
                continue;
            };
            if img.sop_instance_uid.is_empty() {
                continue;
            }
            let dims = [img.cols, img.rows];
            for (si, set) in study.planar_structs.iter().enumerate() {
                for (ki, s) in set.structures.iter().enumerate() {
                    if !self.planar_visible(slot, set, &s.name) {
                        continue;
                    }
                    let edited = active == Some((si, ki));
                    let alpha = if near { 255 } else { 128 };
                    let color =
                        Color32::from_rgba_unmultiplied(s.color[0], s.color[1], s.color[2], alpha);
                    let width = if edited { 3.0 } else { 1.5 };
                    if let Some(c) = s.contour(&img.sop_instance_uid) {
                        for ring in &c.region.rings {
                            if ring.pts.len() < 2 {
                                continue;
                            }
                            let pts: Vec<Pos2> = ring
                                .pts
                                .iter()
                                .map(|p| px_to_screen(*p, dims, on))
                                .collect();
                            painter.add(egui::Shape::closed_line(
                                pts,
                                egui::Stroke::new(width, color),
                            ));
                        }
                    }
                    for (uid, p) in &s.points {
                        if *uid != img.sop_instance_uid {
                            continue;
                        }
                        let c = px_to_screen(*p, dims, on);
                        let r = 6.0;
                        let stroke = egui::Stroke::new(width, color);
                        painter.line_segment([c - Vec2::new(r, r), c + Vec2::new(r, r)], stroke);
                        painter.line_segment([c - Vec2::new(r, -r), c + Vec2::new(r, -r)], stroke);
                        if near {
                            painter.text(
                                c + Vec2::new(r + 2.0, -r),
                                Align2::LEFT_BOTTOM,
                                &s.name,
                                FontId::proportional(11.0),
                                color,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Project structures onto planar images in the background; the set
    /// lands through [`Self::on_projection_done`].
    pub(super) fn start_projection(
        &mut self,
        slot: usize,
        req: ProjectRequest,
        into: Option<usize>,
    ) {
        if self.planar2d.project_job.is_some() {
            self.error = Some("A projection is already running".into());
            return;
        }
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        if !study.has_volume() {
            self.error = Some("Projecting structures needs the CT of the workspace".into());
            return;
        }
        let view = planar_structs::projection_view(study);
        let (dims, for_uid) = (
            study.volume.dims,
            study.volume.frame_of_reference_uid.clone(),
        );
        let progress = Arc::new(Progress::default());
        progress.set("projecting");
        self.planar2d.project_job = Some(Job::spawn(progress, move |p| {
            let r = planar_structs::project_many(&view, &req, p).map(|(set, lines)| ProjectOut {
                set,
                lines,
                into,
                dims,
                for_uid,
            });
            (slot, r)
        }));
    }

    pub(super) fn poll_planar_jobs(&mut self, ctx: &egui::Context) {
        if let Some((slot, out)) = poll_tool_job(
            &mut self.planar2d.project_job,
            ctx,
            "Projection",
            &mut self.error,
        ) {
            self.on_projection_done(slot, out);
        }
        self.poll_reg2d_job(ctx);
    }

    /// A projection landed: the set filed (into the chosen one, or as a new
    /// set) and its first image put on screen when no pane shows it.
    fn on_projection_done(&mut self, slot: usize, out: ProjectOut) {
        if !self.slot_still_shows(slot, out.dims, &out.for_uid) {
            self.error = Some(
                "The projection finished, but the workspace changed while it was running - \
                 the result was discarded."
                    .into(),
            );
            return;
        }
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        let lines: Vec<String> = out
            .lines
            .iter()
            .map(|l| {
                let image = study
                    .planar_images
                    .get(l.image)
                    .map(|p| p.label.clone())
                    .unwrap_or_default();
                if l.rings == 0 {
                    format!("{} on {image}: nothing (outside the image)", l.structure)
                } else {
                    format!(
                        "{} on {image}: {} ring(s), {:.0} mm²",
                        l.structure, l.rings, l.area_mm2
                    )
                }
            })
            .collect();
        let first = out.set.images.first().and_then(|u| study.planar_by_uid(u));
        let set_idx = match out
            .into
            .filter(|&k| study.planar_structs.get(k).is_some_and(|s| !s.locked))
        {
            Some(k) => {
                let dest = &mut study.planar_structs[k];
                for s in out.set.structures {
                    let d = dest.structure_entry(&s.name, Some(s.color));
                    for c in s.contours {
                        match d.contour_mut(&c.image_uid) {
                            Some(old) => *old = c,
                            None => d.contours.push(c),
                        }
                    }
                }
                dest.refresh_images();
                k
            }
            None => {
                let mut set = out.set;
                // Labels are unique within the workspace.
                let base = set.label.clone();
                let mut n = 2;
                while study.planar_structs.iter().any(|s| s.label == set.label) {
                    set.label = format!("{base} #{n}");
                    n += 1;
                }
                study.planar_structs.push(set);
                study.planar_structs.len() - 1
            }
        };
        self.planar2d.last_lines = lines;
        if self.slots[slot].active_planar.is_none() {
            self.slots[slot].active_planar = Some((set_idx, 0));
        }
        self.settings_gen += 1;
        if let Some(i) = first {
            if self.planar_panes_showing(slot, i).is_empty() {
                self.show_planar(slot, i, None);
                self.planar_on_screen(slot);
            }
        }
    }
}

impl ViewerApp {
    /// *💾 Save as presentation state*: the set as a GSPS, with every image
    /// it references that has no file of its own as an RT Image beside it,
    /// each image with the window a pane shows it with.
    pub(super) fn save_planar_set(&mut self, slot: usize, set: usize) {
        if self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.planar_structs.get(set))
            .is_none()
        {
            return;
        }
        self.ask_folder(
            "Save the planar structures as a presentation state into",
            move |app, dir| {
                // The window each image is shown with in a pane, where one does.
                let mut shown: Vec<(usize, (f32, f32))> = Vec::new();
                for k in 0..super::planar::ADDITIONAL_PANE {
                    for li in 0..app.slots[slot].planar[k].layers.len() {
                        let image = app.slots[slot].planar[k].layers[li].image;
                        if let Some((_, wl)) = app.planar_wl((slot, k, li)) {
                            shown.push((image, wl));
                        }
                    }
                }
                let Some(study) = app.slots[slot].study.as_ref() else {
                    return;
                };
                let windows = |i: usize| shown.iter().find(|(j, _)| *j == i).map(|(_, w)| *w);
                match crate::gsps::save_set(study, set, &dir, &windows) {
                    Ok((files, uid)) => {
                        if let Some(s) = app.slots[slot]
                            .study
                            .as_mut()
                            .and_then(|st| st.planar_structs.get_mut(set))
                        {
                            s.sop_instance_uid = uid;
                        }
                        app.notice = Some(format!(
                            "{} file(s) written to {}",
                            files.len(),
                            dir.display()
                        ));
                    }
                    Err(e) => {
                        app.error = Some(format!("Saving the planar structures failed: {e:#}"))
                    }
                }
            },
        );
    }
}
