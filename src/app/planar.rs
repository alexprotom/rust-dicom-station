//! Floating viewers for planar images (DX / CR / RTIMAGE).

use super::*;

impl ViewerApp {
    // -- Floating planar image viewers -------------------------------------
    pub(super) fn planar_windows_ui(&mut self, ctx: &egui::Context) {
        let mut windows = std::mem::take(&mut self.planar_windows);
        for w in &mut windows {
            let Some(study) = &self.slots[w.slot].study else {
                w.open = false;
                continue;
            };
            let Some(img) = study.planar_images.get(w.idx) else {
                w.open = false;
                continue;
            };
            if !w.open {
                continue;
            }

            // Rebuild the texture when W/L changed.
            if w.tex.is_none() || w.tex_wl != w.wl {
                let lo = w.wl.0 - w.wl.1.max(1.0) * 0.5;
                let scale = 255.0 / w.wl.1.max(1.0);
                let pixels: Vec<Color32> = img
                    .data
                    .iter()
                    .map(|&v| {
                        let g = ((v - lo) * scale).clamp(0.0, 255.0) as u8;
                        Color32::from_gray(g)
                    })
                    .collect();
                let ci = ColorImage::new([img.cols, img.rows], pixels);
                match &mut w.tex {
                    Some(t) => t.set(ci, TextureOptions::LINEAR),
                    None => {
                        w.tex = Some(ctx.load_texture(
                            format!("planar{}_{}", w.slot, w.idx),
                            ci,
                            TextureOptions::LINEAR,
                        ))
                    }
                }
                w.tex_wl = w.wl;
            }

            let title = format!("{}: {} [{}]", SLOT_NAMES[w.slot], img.label, img.modality);
            let mut open = w.open;
            egui::Window::new(title)
                .id(egui::Id::new(("planar_win", w.slot, w.idx)))
                .open(&mut open)
                .default_size([560.0, 640.0])
                .resizable(true)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("W/L:");
                        ui.add(egui::DragValue::new(&mut w.wl.0).speed(4.0).prefix("C "));
                        ui.add(
                            egui::DragValue::new(&mut w.wl.1)
                                .speed(8.0)
                                .range(1.0..=1.0e6)
                                .prefix("W "),
                        );
                        if ui.small_button("Auto").clicked() {
                            w.wl = (
                                (img.min_value + img.max_value) * 0.5,
                                (img.max_value - img.min_value).max(1.0),
                            );
                        }
                    });
                    // Physical aspect ratio, fitted to the available width.
                    let w_mm = (img.cols as f64 * img.spacing[0]) as f32;
                    let h_mm = (img.rows as f64 * img.spacing[1]) as f32;
                    let avail = ui.available_width().max(64.0);
                    let scale = (avail / w_mm).min(520.0 / h_mm.max(1.0));
                    if let Some(tex) = &w.tex {
                        // Same interactive window/level as the CT views:
                        // right-drag, x = width, y = center.
                        let resp = ui.add(
                            egui::Image::new(egui::load::SizedTexture::new(
                                tex.id(),
                                egui::vec2(w_mm * scale, h_mm * scale),
                            ))
                            .sense(Sense::click_and_drag()),
                        );
                        if resp.dragged_by(egui::PointerButton::Secondary) {
                            let d = resp.drag_delta();
                            w.wl.1 = (w.wl.1 * (1.0 + d.x * 0.005)).clamp(1.0, 1.0e6);
                            w.wl.0 += d.y * w.wl.1 * 0.002;
                        }
                        resp.on_hover_text("Right-drag: window/level (x = width, y = center)");
                    }
                    for (k, v) in &img.info {
                        ui.weak(format!("{k}: {v}"));
                    }
                });
            w.open = open;
        }
        windows.retain(|w| w.open);
        self.planar_windows = windows;
    }
}
