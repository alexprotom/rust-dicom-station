//! The central MPR viewports: layout, interaction, and the texture caches.
//!
//! Rendering is cache-driven -- each view keeps keyed textures for the
//! grayscale slice, dose colorwash, contour polylines, segmentation overlay
//! and fusion blend, rebuilt only when their inputs change. The `*_hash`
//! helpers are those cache keys.

use super::*;

impl ViewerApp {
    /// Combined hash of everything that affects dose overlays of a slot.
    pub(super) fn dose_settings_hash(&self, slot: usize) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100000001b3);
        };
        mix(slot as u64 + 1);
        mix(self.slots[slot].active_dose as u64);
        mix(self.dose_mode as u64);
        mix(self.dose_opacity.to_bits() as u64);
        mix(self.dose_threshold_pct.to_bits() as u64);
        mix(self.slots[slot].dose_reference.to_bits() as u64);
        for l in &self.iso_levels {
            mix(l.pct.to_bits() as u64 | ((l.on as u64) << 40));
        }
        mix(self.settings_gen);
        h
    }

    pub(super) fn contour_settings_hash(&self, slot: usize) -> u64 {
        let mut h: u64 = 0x9e3779b97f4a7c15 ^ (slot as u64).wrapping_mul(0xff51afd7ed558ccd);
        h = h.rotate_left(11) ^ (self.slots[slot].active_structs as u64 + 1);
        for (i, v) in self.slots[slot].roi_visible.iter().enumerate() {
            if *v {
                h = h.rotate_left(7) ^ (i as u64 + 1);
            }
        }
        h ^ self.settings_gen.wrapping_mul(0x2545F4914F6CDD1D)
    }

    // -- Central: one or two rows of three views --------------------------
    pub(super) fn central_views(&mut self, ui: &mut egui::Ui) {
        let backdrop = backdrop_color(ui.visuals());
        egui::CentralPanel::default_margins()
            .frame(egui::Frame::NONE.fill(backdrop))
            .show(ui, |ui| {
                if self.slots[0].study.is_none() && self.slots[1].study.is_none() {
                    self.empty_state(ui);
                    return;
                }
                // Maximized single-view layout: one view fills the window.
                if let Some((mslot, midx)) = self.maximized {
                    if self.slots[mslot.min(1)].study.is_some() && midx < 3 {
                        let full = ui.available_rect_before_wrap();
                        self.view_cell(ui, mslot.min(1), midx, full);
                        return;
                    }
                    self.maximized = None;
                }
                let two_rows = self.comparison;
                let full = ui.available_rect_before_wrap();
                let row_gap = 6.0;
                let n_rows = if two_rows { 2.0 } else { 1.0 };
                let row_h = (full.height() - (n_rows - 1.0) * row_gap) / n_rows;

                for row in 0..(n_rows as usize) {
                    let y0 = full.top() + row as f32 * (row_h + row_gap);
                    let row_rect = Rect::from_min_size(
                        Pos2::new(full.left(), y0),
                        Vec2::new(full.width(), row_h),
                    );
                    if self.slots[row].study.is_some() {
                        self.study_row(ui, row, row_rect);
                    } else {
                        self.empty_row(ui, row, row_rect);
                    }
                }
            });
    }

    pub(super) fn study_row(&mut self, ui: &mut egui::Ui, slot: usize, row_rect: Rect) {
        let gap = 4.0;
        let col_w = (row_rect.width() - 2.0 * gap) / 3.0;
        for idx in 0..3 {
            let x0 = row_rect.left() + idx as f32 * (col_w + gap);
            let col = Rect::from_min_size(
                Pos2::new(x0, row_rect.top()),
                Vec2::new(col_w, row_rect.height()),
            );
            self.view_cell(ui, slot, idx, col);
        }
    }

    /// One viewport plus its slice slider inside `cell` (used both by the
    /// three-in-a-row layout and by the maximized single-view layout).
    pub(super) fn view_cell(&mut self, ui: &mut egui::Ui, slot: usize, idx: usize, cell: Rect) {
        let slider_h = 26.0;
        let view_rect = Rect::from_min_max(cell.min, Pos2::new(cell.max.x, cell.max.y - slider_h));
        let slider_rect = Rect::from_min_max(
            Pos2::new(cell.min.x + 6.0, cell.max.y - slider_h + 2.0),
            Pos2::new(cell.max.x - 6.0, cell.max.y - 2.0),
        );
        self.one_view(ui, slot, idx, view_rect);
        let max_slice = self.slots[slot]
            .study
            .as_ref()
            .map(|s| {
                s.volume
                    .plane_slice_count(self.slots[slot].views[idx].plane)
                    .saturating_sub(1)
            })
            .unwrap_or(0);
        if max_slice > 0 {
            let mut slice = self.slots[slot].views[idx].slice.min(max_slice);
            let resp = ui.put(
                slider_rect,
                egui::Slider::new(&mut slice, 0..=max_slice).show_value(false),
            );
            if resp.changed() {
                self.slots[slot].views[idx].slice = slice;
            }
        }
    }

    pub(super) fn empty_row(&mut self, ui: &mut egui::Ui, slot: usize, rect: Rect) {
        // `text_color`, not `weak_text_color`: the dimmed variant drops below a
        // readable contrast on the dark row fill (see `theme_tests`).
        let (fill, hint, strong) = {
            let v = ui.visuals();
            (empty_row_color(v), v.text_color(), v.strong_text_color())
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, fill);
        painter.text(
            rect.center() - Vec2::new(0.0, 24.0),
            Align2::CENTER_CENTER,
            format!("No dataset {}", SLOT_NAMES[slot]),
            FontId::proportional(15.0),
            hint,
        );
        let btn_rect =
            Rect::from_center_size(rect.center() + Vec2::new(0.0, 10.0), Vec2::new(220.0, 28.0));
        if ui
            .put(btn_rect, egui::Button::new("📂 Add DICOM folder…"))
            .clicked()
        {
            if let Some(dir) = Self::pick_folder("Select DICOM folder to add to dataset B") {
                self.start_load(slot, dir);
            }
        }
        if self.loading.is_some() {
            if let Some(job) = &self.loading {
                painter.text(
                    rect.center() + Vec2::new(0.0, 44.0),
                    Align2::CENTER_CENTER,
                    format!("⏳ {}", job.progress.get()),
                    FontId::proportional(13.0),
                    strong,
                );
            }
        }
    }

    pub(super) fn empty_state(&mut self, ui: &mut egui::Ui) {
        ui.centered_and_justified(|ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.35);
                ui.heading("Rust DICOM / RT viewer");
                ui.add_space(8.0);
                if self.loading.is_some() {
                    ui.spinner();
                    if let Some(job) = &self.loading {
                        ui.label(job.progress.get());
                    }
                } else if let Some(job) = &self.gen_job {
                    ui.spinner();
                    ui.label(format!("Generating test data — {}", job.progress.get()));
                } else {
                    ui.label("Add a folder containing DICOM data");
                    ui.add_space(8.0);
                    if ui.button("📂 Add DICOM folder…").clicked() {
                        if let Some(dir) = Self::pick_folder("Select a DICOM folder") {
                            self.start_load(0, dir);
                        }
                    }
                    ui.add_space(12.0);
                    ui.weak("…or create a synthetic RT study to try the viewer on");
                    ui.add_space(4.0);
                    if ui
                        .button("🧪 Generate test data…")
                        .on_hover_text(
                            "Writes a synthetic CT + RTSTRUCT + RTPLAN + RTDOSE study \
                             into the application folder",
                        )
                        .clicked()
                    {
                        self.gen_open = true;
                    }
                }
            });
        });
    }

    // -- One viewport -----------------------------------------------------
    pub(super) fn one_view(&mut self, ui: &mut egui::Ui, slot: usize, idx: usize, rect: Rect) {
        let ctx = ui.ctx().clone();
        let plane = self.slots[slot].views[idx].plane;

        // ---- cache refresh (image, dose, contours) ----
        self.refresh_view_caches(&ctx, slot, idx);

        let slot_state = &self.slots[slot];
        let Some(study) = &slot_state.study else {
            return;
        };
        let vol = &study.volume;
        let view = &slot_state.views[idx];

        // The MedSAM2 box is drawn in whichever view the network's slices run
        // along, and while it is live the left button belongs to it.
        let medsam2_show = self.medsam2_showing_in(slot, plane);
        let medsam2_box = self.medsam2_drawing_in(slot, plane);
        let [w_px, h_px] = vol.plane_dims(plane);
        let [sx, sy] = vol.plane_spacing(plane);
        let w_mm = (w_px as f64 * sx) as f32;
        let h_mm = (h_px as f64 * sy) as f32;

        let fit_zoom = ((rect.width() / w_mm).min(rect.height() / h_mm) * 0.97).max(0.01);
        let zoom = if view.zoom > 0.0 { view.zoom } else { fit_zoom };
        let center = rect.center() + view.pan * zoom;
        let img_rect = Rect::from_center_size(center, Vec2::new(w_mm * zoom, h_mm * zoom));

        let px_to_screen = |p: [f32; 2]| -> Pos2 {
            Pos2::new(
                img_rect.left() + (p[0] + 0.5) * sx as f32 * zoom,
                img_rect.top() + (p[1] + 0.5) * sy as f32 * zoom,
            )
        };
        let screen_to_px = |s: Pos2| -> [f32; 2] {
            [
                (s.x - img_rect.left()) / (sx as f32 * zoom) - 0.5,
                (s.y - img_rect.top()) / (sy as f32 * zoom) - 0.5,
            ]
        };

        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);

        let fusion_active = self.fusion_on
            && self
                .registration
                .as_ref()
                .is_some_and(|r| r.fixed_slot == slot)
            && view.fusion_tex.is_some();
        if fusion_active {
            if let Some(tex) = &view.fusion_tex {
                painter.image(
                    tex.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        } else if let Some(tex) = &view.tex {
            painter.image(
                tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        if self.dose_mode.wash() {
            if let Some(tex) = &view.dose_tex {
                painter.image(
                    tex.id(),
                    img_rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        }

        // Painted segmentations (and the live region-growing preview).
        if let Some(tex) = &view.seg_tex {
            painter.image(
                tex.id(),
                img_rect,
                Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        // Isodose lines.
        if self.dose_mode.iso() {
            for (li, seg) in &view.iso_segs {
                if let Some(level) = self.iso_levels.get(*li) {
                    painter.line_segment(
                        [px_to_screen(seg.0), px_to_screen(seg.1)],
                        Stroke::new(1.6, level.color),
                    );
                }
            }
        }

        // Contours.
        if self.show_contours {
            if let Some(ss) = slot_state.active_structures() {
                for (ri, gfx) in &view.contours {
                    let Some(roi) = ss.rois.get(*ri) else {
                        continue;
                    };
                    let color = Color32::from_rgb(roi.color[0], roi.color[1], roi.color[2]);
                    let stroke = Stroke::new(1.8, color);
                    for pl in &gfx.polylines {
                        let pts: Vec<Pos2> = pl.iter().map(|p| px_to_screen(*p)).collect();
                        painter.add(egui::Shape::closed_line(pts, stroke));
                    }
                    for (a, b) in &gfx.segments {
                        painter.line_segment([px_to_screen(*a), px_to_screen(*b)], stroke);
                    }
                    for p in &gfx.points {
                        let c = px_to_screen(*p);
                        painter.line_segment(
                            [c + Vec2::new(-4.0, 0.0), c + Vec2::new(4.0, 0.0)],
                            stroke,
                        );
                        painter.line_segment(
                            [c + Vec2::new(0.0, -4.0), c + Vec2::new(0.0, 4.0)],
                            stroke,
                        );
                    }
                }
            }
        }

        // Isocenter markers.
        if self.show_isocenters {
            let mut seen: Vec<[i64; 3]> = Vec::new();
            for plan in &study.plans {
                for b in &plan.beams {
                    let Some(iso) = b.isocenter else { continue };
                    let key = [
                        (iso.x * 10.0) as i64,
                        (iso.y * 10.0) as i64,
                        (iso.z * 10.0) as i64,
                    ];
                    if seen.contains(&key) {
                        continue;
                    }
                    seen.push(key);
                    let (pp, dz) = render::patient_to_plane_pixel(vol, plane, view.slice, iso);
                    let on_plane = dz.abs() <= 1.0;
                    let alpha = if on_plane { 255 } else { 80 };
                    let col = Color32::from_rgba_unmultiplied(255, 230, 40, alpha);
                    let c = px_to_screen(pp);
                    if rect.expand(20.0).contains(c) {
                        let s = Stroke::new(1.5, col);
                        painter.circle_stroke(c, 6.0, s);
                        painter
                            .line_segment([c + Vec2::new(-9.0, 0.0), c + Vec2::new(9.0, 0.0)], s);
                        painter
                            .line_segment([c + Vec2::new(0.0, -9.0), c + Vec2::new(0.0, 9.0)], s);
                    }
                }
            }
        }

        // Crosshair.
        if self.show_crosshair {
        // ---- the MedSAM2 prompt ----
        if medsam2_show {
            if let Some(b) = &self.medsam2.prompt {
                if b.plane == plane {
                    let here = b.slice == view.slice;
                    let base = Color32::from_rgb(255, 205, 60);
                    let col = if here {
                        base
                    } else {
                        // On other slices the box is a reminder of where it
                        // is, not something to grab.
                        Color32::from_rgba_unmultiplied(255, 205, 60, 70)
                    };
                    let (lo, hi) = b.rect();
                    let r = Rect::from_two_pos(px_to_screen(lo), px_to_screen(hi));
                    painter.rect_stroke(
                        r,
                        0.0,
                        Stroke::new(if here { 2.0 } else { 1.0 }, col),
                        egui::StrokeKind::Middle,
                    );
                    if here {
                        for c in b.corners() {
                            painter.rect_filled(
                                Rect::from_center_size(px_to_screen(c), Vec2::splat(7.0)),
                                1.0,
                                col,
                            );
                        }
                        for (p, include) in &b.points {
                            let at = px_to_screen(*p);
                            let c = if *include {
                                Color32::from_rgb(90, 220, 130)
                            } else {
                                Color32::from_rgb(240, 95, 95)
                            };
                            painter.circle_filled(at, 4.5, c);
                            painter.circle_stroke(at, 4.5, Stroke::new(1.0, Color32::BLACK));
                        }
                    }
                }
            }
        }

            let cp = vol.voxel_to_plane_pixel(plane, slot_state.cursor);
            let c = px_to_screen([cp[0] as f32, cp[1] as f32]);
            let col = Color32::from_rgba_unmultiplied(120, 255, 120, 110);
            let s = Stroke::new(1.0, col);
            if rect.contains(Pos2::new(c.x, rect.center().y)) {
                painter.line_segment(
                    [Pos2::new(c.x, rect.top()), Pos2::new(c.x, rect.bottom())],
                    s,
                );
            }
            if rect.contains(Pos2::new(rect.center().x, c.y)) {
                painter.line_segment(
                    [Pos2::new(rect.left(), c.y), Pos2::new(rect.right(), c.y)],
                    s,
                );
            }
        }

        // Brush cursor and region-growing readout.
        if self.seg_tool != SegTool::None {
            let hover = ui
                .input(|i| i.pointer.hover_pos())
                .filter(|p| rect.contains(*p));
            match self.seg_tool {
                SegTool::Brush | SegTool::Erase => {
                    if let Some(mp) = hover {
                        let erase =
                            self.seg_tool == SegTool::Erase || ui.input(|i| i.modifiers.alt);
                        let col = if erase {
                            Color32::from_rgba_unmultiplied(255, 90, 90, 200)
                        } else {
                            let c = slot_state
                                .segs
                                .get(slot_state.active_seg)
                                .map(|s| s.color)
                                .unwrap_or([140, 255, 140]);
                            Color32::from_rgba_unmultiplied(c[0], c[1], c[2], 220)
                        };
                        painter.circle_stroke(
                            mp,
                            self.brush_radius_mm * zoom,
                            Stroke::new(1.5, col),
                        );
                    }
                }
                SegTool::Grow => {
                    if let Some(mp) = hover {
                        let s = Stroke::new(1.5, Color32::from_rgba_unmultiplied(255, 220, 0, 220));
                        painter
                            .line_segment([mp + Vec2::new(-6.0, 0.0), mp + Vec2::new(6.0, 0.0)], s);
                        painter
                            .line_segment([mp + Vec2::new(0.0, -6.0), mp + Vec2::new(0.0, 6.0)], s);
                    }
                    if let Some(g) = self.grow.as_ref().filter(|g| g.slot == slot) {
                        let vv = vol.spacing[0] * vol.spacing[1] * vol.spacing[2] / 1000.0;
                        painter.text(
                            rect.left_bottom() + Vec2::new(6.0, -22.0),
                            Align2::LEFT_BOTTOM,
                            format!(
                                "grow {:.1} cm³ · reach ×{:.2}{}",
                                self.grow_state.voxels.len() as f64 * vv,
                                g.level,
                                if g.capped { " · CAPPED" } else { "" }
                            ),
                            FontId::proportional(13.0),
                            Color32::from_rgb(255, 220, 0),
                        );
                    }
                }
                SegTool::None => {}
            }
        }

        // Annotations.
        if self.show_labels {
            let n_slices = vol.plane_slice_count(plane);
            let both = self.comparison;
            let title = if both {
                format!("{} · {}", plane.title(), SLOT_NAMES[slot])
            } else {
                plane.title().to_string()
            };
            painter.text(
                rect.left_top() + Vec2::new(6.0, 4.0),
                Align2::LEFT_TOP,
                title,
                FontId::proportional(14.0),
                if slot == 0 {
                    Color32::from_rgb(255, 170, 60)
                } else {
                    Color32::from_rgb(120, 200, 255)
                },
            );
            painter.text(
                rect.right_top() + Vec2::new(-6.0, 4.0),
                Align2::RIGHT_TOP,
                format!("{}/{}", view.slice + 1, n_slices),
                FontId::proportional(12.0),
                Color32::LIGHT_GRAY,
            );
            // Anatomical edge labels.
            let (dx, dy) = vol.plane_screen_dirs(plane);
            let lbl = |v| crate::geometry::direction_label(v);
            let f = FontId::proportional(12.0);
            let lc = Color32::from_rgb(120, 200, 255);
            painter.text(
                Pos2::new(rect.right() - 8.0, rect.center().y),
                Align2::RIGHT_CENTER,
                lbl(dx),
                f.clone(),
                lc,
            );
            painter.text(
                Pos2::new(rect.left() + 8.0, rect.center().y),
                Align2::LEFT_CENTER,
                lbl(dx * -1.0),
                f.clone(),
                lc,
            );
            painter.text(
                Pos2::new(rect.center().x, rect.bottom() - 6.0),
                Align2::CENTER_BOTTOM,
                lbl(dy),
                f.clone(),
                lc,
            );
            painter.text(
                Pos2::new(rect.center().x, rect.top() + 4.0),
                Align2::CENTER_TOP,
                lbl(dy * -1.0),
                f,
                lc,
            );
        }

        // Loading overlay.
        if self.loading.is_some() {
            painter.rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(0, 0, 0, 140));
            if idx == 1 && slot == 0 {
                if let Some(job) = &self.loading {
                    painter.text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        format!("⏳ {}", job.progress.get()),
                        FontId::proportional(15.0),
                        Color32::WHITE,
                    );
                }
            }
        }

        // ---- corner buttons: reset view & maximize / restore layout ----
        // Their rectangles are needed here (the viewport handlers below ignore
        // any pointer activity over them), but the buttons themselves are
        // registered *after* the viewport interaction: the last widget at a
        // position is the topmost one, so they get the hover — and show their
        // tooltips — instead of the full-viewport rectangle underneath.
        let is_max = self.maximized == Some((slot, idx));
        let bsize = egui::vec2(24.0, 20.0);
        let by = rect.top() + 22.0; // below the slice counter
        let max_rect = Rect::from_min_size(Pos2::new(rect.right() - bsize.x - 4.0, by), bsize);
        let fit_rect = Rect::from_min_size(Pos2::new(max_rect.left() - bsize.x - 4.0, by), bsize);
        let (pointer_pos, any_click) =
            ui.input(|i| (i.pointer.interact_pos(), i.pointer.any_click()));
        let over_buttons = pointer_pos
            .map(|p| max_rect.contains(p) || fit_rect.contains(p))
            .unwrap_or(false);

        // ---- interaction ----
        let resp = ui.interact(
            rect,
            egui::Id::new(("viewport", slot, idx)),
            Sense::click_and_drag(),
        );

        let max_resp = ui
            .put(
                max_rect,
                egui::Button::new(if is_max { "❐" } else { "⛶" }).small(),
            )
            .on_hover_text(if is_max {
                "Restore the multi-view layout"
            } else {
                "Maximize this view to the whole window"
            });
        let fit_resp = ui
            .put(fit_rect, egui::Button::new("⟲").small())
            .on_hover_text(
                "Reset this view: fit zoom, clear pan and put the crosshair back at \
             the volume center",
            );
        let clicked_max = max_resp.clicked()
            || (any_click && pointer_pos.map(|p| max_rect.contains(p)).unwrap_or(false));
        let clicked_fit = fit_resp.clicked()
            || (any_click && pointer_pos.map(|p| fit_rect.contains(p)).unwrap_or(false));
        // (applied below, in the mutable phase)
        let n_slices = vol.plane_slice_count(plane);
        let seg_active = self.seg_tool != SegTool::None;
        let cur_slice = view.slice;

        let mut new_slice = None;
        let mut new_zoom = None;
        let mut new_pan = None;
        let mut new_cursor = None;
        let mut wl_delta = None;
        let mut reset_view = false;
        let mut new_accum = None;
        let mut new_brush = None;

        if resp.hovered() {
            let (wheel_lines, brush_lines, zoom_delta, pointer) = ui.input(|i| {
                let mut lines = 0.0f32;
                let mut brush = 0.0f32;
                for e in &i.events {
                    if let egui::Event::MouseWheel {
                        unit,
                        delta,
                        modifiers,
                        ..
                    } = e
                    {
                        let scale = match unit {
                            egui::MouseWheelUnit::Line => 1.0,
                            egui::MouseWheelUnit::Point => 1.0 / 40.0,
                            egui::MouseWheelUnit::Page => 10.0,
                        };
                        if modifiers.shift && seg_active {
                            // Some platforms report shift+wheel horizontally.
                            brush += (delta.y + delta.x) * scale;
                        } else if !(modifiers.ctrl || modifiers.command) {
                            lines += delta.y * scale;
                        }
                    }
                }
                (lines, brush, i.zoom_delta(), i.pointer.hover_pos())
            });
            if brush_lines.abs() > 0.0 {
                new_brush =
                    Some((self.brush_radius_mm * (brush_lines * 0.12).exp()).clamp(0.5, 80.0));
            }
            if (zoom_delta - 1.0).abs() > 1e-4 {
                // Keep the point under the cursor fixed while zooming.
                let z0 = zoom;
                let z1 = (z0 * zoom_delta).clamp(fit_zoom * 0.2, fit_zoom * 40.0);
                if let Some(mp) = pointer {
                    let rel = (mp - rect.center()) / z0 - view.pan;
                    let pan1 = (mp - rect.center()) / z1 - rel;
                    new_pan = Some(pan1);
                }
                new_zoom = Some(z1);
            }
            if wheel_lines.abs() > 0.0 {
                let acc = view.scroll_accum + wheel_lines;
                let steps = acc.trunc() as i64;
                new_accum = Some(acc - steps as f32);
                if steps != 0 {
                    let s = (view.slice as i64 - steps).clamp(0, n_slices as i64 - 1) as usize;
                    new_slice = Some(s);
                }
            }
        }

        // Left-click crosshair navigation only while the crosshair is shown
        // and no segmentation tool holds the left button; with ⌖ off, slices
        // change only by scrolling the hovered view.
        if self.show_crosshair
            && !seg_active
            && !medsam2_box
            && (resp.dragged_by(egui::PointerButton::Primary) || resp.clicked())
            && !over_buttons
        {
            if let Some(mp) = resp.interact_pointer_pos() {
                let px = screen_to_px(mp);
                let vxl = vol.plane_pixel_to_voxel(plane, view.slice, px[0] as f64, px[1] as f64);
                new_cursor = Some(vxl);
            }
        }

        // Segmentation tools on the left button.
        let mut paint_to: Option<([f64; 3], bool)> = None;
        let mut paint_done = false;
        let mut grow_start: Option<([f64; 3], f32)> = None;
        let mut grow_move: Option<f32> = None;
        let mut grow_done = false;
        if seg_active && !over_buttons {
            let to_voxel = |mp: Pos2| {
                let px = screen_to_px(mp);
                vol.plane_pixel_to_voxel(plane, cur_slice, px[0] as f64, px[1] as f64)
            };
            match self.seg_tool {
                SegTool::Brush | SegTool::Erase => {
                    if resp.dragged_by(egui::PointerButton::Primary) || resp.clicked() {
                        if let Some(mp) = resp.interact_pointer_pos() {
                            let erase =
                                self.seg_tool == SegTool::Erase || ui.input(|i| i.modifiers.alt);
                            paint_to = Some((to_voxel(mp), erase));
                        }
                    }
                    if resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked() {
                        paint_done = true;
                    }
                }
                SegTool::Grow => {
                    if resp.drag_started_by(egui::PointerButton::Primary)
                        || (resp.clicked() && self.grow.is_none())
                    {
                        if let Some(mp) = resp.interact_pointer_pos() {
                            grow_start = Some((to_voxel(mp), mp.y));
                        }
                    } else if resp.dragged_by(egui::PointerButton::Primary) {
                        if let Some(mp) = resp.interact_pointer_pos() {
                            grow_move = Some(mp.y);
                        }
                    }
                    if resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked() {
                        grow_done = true;
                    }
                }
                SegTool::None => {}
            }
        }
        // The MedSAM2 box: press to start, grab a corner or move it; drag to
        // size it; release to commit. Include / exclude clicks go through the
        // same press.
        let mut box_press: Option<([f32; 2], f32)> = None;
        let mut box_drag: Option<[f32; 2]> = None;
        let mut box_release = false;
        if medsam2_box && !over_buttons {
            if resp.drag_started_by(egui::PointerButton::Primary) || resp.clicked() {
                if let Some(mp) = resp.interact_pointer_pos() {
                    // The grab tolerance is a screen distance, so it has to be
                    // converted into the pixel units the box is kept in.
                    let tol = medsam2_seg::HANDLE_GRAB / (sx as f32 * zoom).max(1e-3);
                    box_press = Some((screen_to_px(mp), tol));
                }
            } else if resp.dragged_by(egui::PointerButton::Primary) {
                if let Some(mp) = resp.interact_pointer_pos() {
                    box_drag = Some(screen_to_px(mp));
                }
            }
            if resp.drag_stopped_by(egui::PointerButton::Primary) || resp.clicked() {
                box_release = true;
            }
        }
        if resp.dragged_by(egui::PointerButton::Secondary) {
            let d = resp.drag_delta();
            wl_delta = Some((d.x, d.y));
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            new_pan = Some(view.pan + d / zoom);
        }
        let hovered = resp.hovered();

        // Apply interactions (mutable phase).
        if clicked_max {
            self.maximized = if is_max { None } else { Some((slot, idx)) };
        }
        if clicked_fit {
            reset_view = true;
        }
        if hovered {
            self.hovered_slot = slot;
        }
        if let Some(a) = new_accum {
            self.slots[slot].views[idx].scroll_accum = a;
        }
        if let Some(s) = new_slice {
            self.slots[slot].views[idx].slice = s;
        }
        if let Some(z) = new_zoom {
            self.slots[slot].views[idx].zoom = z;
        }
        if let Some(p) = new_pan {
            self.slots[slot].views[idx].pan = p;
        }
        if reset_view {
            self.slots[slot].views[idx].zoom = 0.0;
            self.slots[slot].views[idx].pan = Vec2::ZERO;
            // Also put the crosshair back at the volume center, which returns
            // this slot's three views to their default (central) slices.
            self.center_cursor(slot);
        }
        if let Some((dx, dy)) = wl_delta {
            self.window_width = (self.window_width * (1.0 + dx * 0.005)).clamp(1.0, 30000.0);
            self.window_center += dy * self.window_width * 0.002;
        }
        if let Some(c) = new_cursor {
            self.set_cursor(slot, c, idx);
        }
        if let Some(r) = new_brush {
            self.brush_radius_mm = r;
        }
        if let Some((p, tol)) = box_press {
            self.medsam2_press(plane, cur_slice, p, tol);
        }
        if let Some(p) = box_drag {
            self.medsam2_drag(p);
        }
        if box_release {
            self.medsam2_release([w_px, h_px]);
        }
        if let Some((vxl, erase)) = paint_to {
            self.apply_brush(slot, plane, cur_slice, vxl, erase);
        }
        if paint_done {
            self.end_paint_stroke(slot);
        }
        if let Some((seed, y)) = grow_start {
            self.begin_grow(slot, seed, y);
        }
        if let Some(y) = grow_move {
            self.update_grow(y);
        }
        if grow_done {
            self.commit_grow();
        }
    }

    /// Set the crosshair of `slot` (voxel coords), sync its other two views,
    /// and — when study linking is on — propagate the same patient-space
    /// point to the other study.
    pub(super) fn set_cursor(&mut self, slot: usize, c: [f64; 3], source_view: usize) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let dims = study.volume.dims;
        let clamped = [
            c[0].clamp(0.0, dims[0] as f64 - 1.0),
            c[1].clamp(0.0, dims[1] as f64 - 1.0),
            c[2].clamp(0.0, dims[2] as f64 - 1.0),
        ];
        let patient = study
            .volume
            .voxel_to_patient(clamped[0], clamped[1], clamped[2]);
        self.slots[slot].cursor = clamped;
        self.sync_views_to_cursor(slot, Some(source_view));

        if self.link_studies {
            let other = 1 - slot;
            let Some(ostudy) = &self.slots[other].study else {
                return;
            };
            // Map through the registration transform when one exists.
            // The transform maps fixed-slot patient coordinates into the
            // moving slot; clicks on the moving study use the inverse.
            let target = match &self.registration {
                Some(reg) if slot == reg.fixed_slot => reg.result.transform.map(patient),
                Some(reg) => reg.result.transform.unmap(patient),
                None => patient,
            };
            let odims = ostudy.volume.dims;
            let oc = ostudy.volume.patient_to_voxel(target);
            self.slots[other].cursor = [
                oc[0].clamp(0.0, odims[0] as f64 - 1.0),
                oc[1].clamp(0.0, odims[1] as f64 - 1.0),
                oc[2].clamp(0.0, odims[2] as f64 - 1.0),
            ];
            self.sync_views_to_cursor(other, None);
        }
    }

    /// Update slice indices of a slot's views to follow its cursor
    /// (skipping the view the user is interacting with, if any).
    pub(super) fn sync_views_to_cursor(&mut self, slot: usize, skip_view: Option<usize>) {
        let Some(study) = &self.slots[slot].study else {
            return;
        };
        let cursor = self.slots[slot].cursor;
        let mut new_slices = [None; 3];
        for (i, out) in new_slices.iter_mut().enumerate() {
            if skip_view == Some(i) {
                continue;
            }
            let pl = self.slots[slot].views[i].plane;
            let sc = match pl {
                ViewPlane::Axial => cursor[2],
                ViewPlane::Sagittal => cursor[0],
                ViewPlane::Coronal => cursor[1],
            };
            let max = study.volume.plane_slice_count(pl).saturating_sub(1);
            *out = Some((sc.round().max(0.0) as usize).min(max));
        }
        for (view, s) in self.slots[slot].views.iter_mut().zip(new_slices) {
            if let Some(s) = s {
                view.slice = s;
            }
        }
    }

    /// Rebuild per-view textures & cached geometry when their inputs changed.
    pub(super) fn refresh_view_caches(&mut self, ctx: &egui::Context, slot: usize, idx: usize) {
        if self.slots[slot].study.is_none() {
            return;
        }
        // Pre-compute hashes that need `&self` before borrowing mutably.
        let dose_hash = self.dose_settings_hash(slot);
        let contour_hash = self.contour_settings_hash(slot);
        let seg_hash = self.seg_overlay_hash(slot);
        let grow_here = self.grow.as_ref().is_some_and(|g| g.slot == slot);
        let wc = self.window_center;
        let ww = self.window_width;
        let dose_on = self.dose_mode != DoseMode::Off;
        let contours_on = self.show_contours;

        let StudySlot {
            study,
            views,
            roi_visible,
            active_structs,
            active_dose,
            dose_reference,
            segs,
            ..
        } = &mut self.slots[slot];
        let study = study.as_ref().unwrap();
        let vol = &study.volume;
        let plane = views[idx].plane;
        let n_slices = vol.plane_slice_count(plane);
        if views[idx].slice >= n_slices {
            views[idx].slice = n_slices.saturating_sub(1);
        }
        let slice = views[idx].slice;
        let [w, h] = vol.plane_dims(plane);

        // Grayscale image.
        let img_key = (slice, wc.to_bits(), ww.to_bits());
        if views[idx].img_key != Some(img_key) {
            let view = &mut views[idx];
            let mut slice_buf = std::mem::take(&mut view.slice_buf);
            let mut gray = Vec::new();
            vol.extract_slice(plane, slice, &mut slice_buf);
            render::slice_to_gray(&slice_buf, wc, ww, w, &mut gray);
            // Moved straight into the texture: keeping a second copy around
            // as a scratch buffer only bought an extra full-image memcpy.
            let img = ColorImage::new([w, h], gray);
            match &mut view.tex {
                Some(t) => t.set(img, TextureOptions::LINEAR),
                None => {
                    view.tex = Some(ctx.load_texture(
                        format!("img{slot}_{idx}"),
                        img,
                        TextureOptions::LINEAR,
                    ))
                }
            }
            view.slice_buf = slice_buf;
            view.img_key = Some(img_key);
        }

        // Dose overlay + isodose segments.
        if dose_on && !study.doses.is_empty() {
            let dose_key = dose_hash.wrapping_add((slice as u64).wrapping_mul(0x9E3779B97F4A7C15));
            if views[idx].dose_key != Some(dose_key) {
                let dose = &study.doses[(*active_dose).min(study.doses.len() - 1)];
                let reference = *dose_reference;
                let view = &mut views[idx];
                let mut dose_plane = std::mem::take(&mut view.dose_plane);
                let mut dose_rgba = Vec::new();
                render::sample_dose_plane(vol, dose, plane, slice, &mut dose_plane);
                render::dose_colorwash(
                    &dose_plane,
                    reference,
                    self.dose_threshold_pct / 100.0,
                    self.dose_opacity,
                    &mut dose_rgba,
                );
                let img = ColorImage::new([w, h], dose_rgba);
                match &mut view.dose_tex {
                    Some(t) => t.set(img, TextureOptions::LINEAR),
                    None => {
                        view.dose_tex = Some(ctx.load_texture(
                            format!("dose{slot}_{idx}"),
                            img,
                            TextureOptions::LINEAR,
                        ))
                    }
                }
                // Isodose segments — one marching-squares pass per enabled
                // level, and the levels are independent.
                let levels: Vec<(usize, f32)> = self
                    .iso_levels
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.on)
                    .map(|(li, l)| (li, l.pct / 100.0 * reference))
                    .filter(|&(_, abs)| abs > 0.0)
                    .collect();
                let per_level: Vec<Vec<(usize, render::Segment)>> = levels
                    .par_iter()
                    .map(|&(li, abs)| {
                        render::marching_squares(&dose_plane, w, h, abs)
                            .into_iter()
                            .map(|seg| (li, seg))
                            .collect()
                    })
                    .collect();
                view.iso_segs.clear();
                view.iso_segs.extend(per_level.into_iter().flatten());
                view.dose_plane = dose_plane;
                view.dose_key = Some(dose_key);
            }
        }

        // Contours.
        if contours_on {
            if let Some(ss) = study.structure_sets.get(*active_structs) {
                let ckey =
                    contour_hash.wrapping_add((slice as u64).wrapping_mul(0x517CC1B727220A95));
                if views[idx].contour_key != Some(ckey) {
                    let mut contours = Vec::new();
                    for (ri, roi) in ss.rois.iter().enumerate() {
                        if !roi_visible.get(ri).copied().unwrap_or(false) {
                            continue;
                        }
                        let gfx = render::roi_on_plane(vol, roi, plane, slice);
                        if !gfx.polylines.is_empty()
                            || !gfx.segments.is_empty()
                            || !gfx.points.is_empty()
                        {
                            contours.push((ri, gfx));
                        }
                    }
                    views[idx].contours = contours;
                    views[idx].contour_key = Some(ckey);
                }
            }
        }

        // Segmentation overlay: all visible masks plus the live
        // region-growing preview, blended into one RGBA texture. NEAREST
        // filtering keeps the voxel raster crisp while editing.
        if !segs.is_empty() || grow_here {
            let skey = seg_hash.wrapping_add((slice as u64).wrapping_mul(0x2545F4914F6CDD1D));
            if views[idx].seg_key != Some(skey) {
                let mut rgba = vec![Color32::TRANSPARENT; w * h];
                for seg in segs.iter().filter(|s| s.visible) {
                    segmentation::overlay_slice(
                        &seg.mask, vol.dims, plane, slice, seg.color, 90, &mut rgba,
                    );
                }
                let n = vol.dims[0] * vol.dims[1] * vol.dims[2];
                if grow_here && self.grow_preview.len() == n {
                    segmentation::overlay_slice(
                        &self.grow_preview,
                        vol.dims,
                        plane,
                        slice,
                        [255, 220, 0],
                        150,
                        &mut rgba,
                    );
                }
                let img = ColorImage::new([w, h], rgba);
                let view = &mut views[idx];
                match &mut view.seg_tex {
                    Some(t) => t.set(img, TextureOptions::NEAREST),
                    None => {
                        view.seg_tex = Some(ctx.load_texture(
                            format!("seg{slot}_{idx}"),
                            img,
                            TextureOptions::NEAREST,
                        ))
                    }
                }
                view.seg_key = Some(skey);
            }
        } else if views[idx].seg_tex.is_some() {
            views[idx].seg_tex = None;
            views[idx].seg_key = None;
        }

        self.refresh_fusion_cache(ctx, slot, idx);
    }

    /// Rebuild the magenta/green fusion texture of a fixed-study view: the
    /// fixed image stays gray in R/B, the transformed moving image is blended
    /// into the green channel. Aligned anatomy therefore reads gray,
    /// mismatches magenta/green. Drawn on whichever slot was the fixed image
    /// of the active registration.
    pub(super) fn refresh_fusion_cache(&mut self, ctx: &egui::Context, slot: usize, idx: usize) {
        if !self.fusion_on {
            return;
        }
        let Some(reg) = &self.registration else {
            return;
        };
        if reg.fixed_slot != slot {
            return;
        }
        if self.slots[0].study.is_none() || self.slots[1].study.is_none() {
            return;
        }
        let transform: Arc<Transform3> = reg.result.transform.clone();
        let fixed_slot = reg.fixed_slot;
        let wc = self.window_center;
        let ww = self.window_width.max(1.0);
        let weight = self.fusion_weight.clamp(0.0, 1.0);

        let (left, right) = self.slots.split_at_mut(1);
        let (a, bvol) = if fixed_slot == 0 {
            let bvol = &right[0].study.as_ref().unwrap().volume;
            (&mut left[0], bvol)
        } else {
            let bvol = &left[0].study.as_ref().unwrap().volume;
            (&mut right[0], bvol)
        };
        let avol = &a.study.as_ref().unwrap().volume;
        let view = &mut a.views[idx];
        let plane = view.plane;
        let slice = view.slice;
        let [w, h] = avol.plane_dims(plane);

        let mut key: u64 = 0x243F6A8885A308D3 ^ self.reg_gen.wrapping_mul(0x9E3779B97F4A7C15);
        for v in [
            slice as u64,
            wc.to_bits() as u64,
            ww.to_bits() as u64,
            weight.to_bits() as u64,
            self.settings_gen,
        ] {
            key ^= v;
            key = key.wrapping_mul(0x100000001b3);
        }
        if view.fusion_key == Some(key) {
            return;
        }

        // Ensure the A slice buffer matches the current slice.
        if view.slice_buf.len() != w * h {
            avol.extract_slice(plane, slice, &mut view.slice_buf);
        }
        let slice_buf = &view.slice_buf;

        let lo = wc - ww * 0.5;
        let scale = 255.0 / ww;
        let wl = |v: f32| -> f32 { ((v - lo) * scale).clamp(0.0, 255.0) };

        let mut pixels = vec![Color32::BLACK; w * h];
        pixels.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
            for (px, out) in row.iter_mut().enumerate() {
                let a_gray = wl(slice_buf[py * w + px] as f32);
                let vxl = avol.plane_pixel_to_voxel(plane, slice, px as f64, py as f64);
                let p_fixed = avol.voxel_to_patient(vxl[0], vxl[1], vxl[2]);
                let q = transform.map(p_fixed);
                let b_gray = bvol.sample_patient(q).map(&wl).unwrap_or(0.0);
                let g = a_gray + (b_gray - a_gray) * weight;
                *out = Color32::from_rgb(a_gray as u8, g as u8, a_gray as u8);
            }
        });

        let img = ColorImage::new([w, h], pixels);
        match &mut view.fusion_tex {
            Some(t) => t.set(img, TextureOptions::LINEAR),
            None => {
                view.fusion_tex = Some(ctx.load_texture(
                    format!("fusion{fixed_slot}_{idx}"),
                    img,
                    TextureOptions::LINEAR,
                ))
            }
        }
        view.fusion_key = Some(key);
    }
}
