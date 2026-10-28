//! A report in the row's additional pane: its facts, its lines and its
//! tables, and for a scanner's page the page itself behind a toggle.
//!
//! A report is read, not windowed, so the pane draws it on the panel's own
//! background with the interface's text, scrolling both ways, and keeps
//! only the bar every pane has: maximize, fold, close. A page's rendering
//! (the overlay the scanner drew) is one button away, fitted to the pane.
//! A document (an encapsulated PDF) is its facts and then its pages, each
//! as wide as the pane, scrolled as one column; the pages are drawn off
//! the frame the first time the document is shown
//! ([`crate::report::document::Document::pages`]) and made textures once.

use super::*;

impl ViewerApp {
    /// The additional pane showing report `ri` of `slot`, filling `rect`.
    pub(super) fn report_pane(&mut self, ui: &mut egui::Ui, slot: usize, ri: usize, rect: Rect) {
        let kind = PaneKind::Additional;
        let k = super::planar::ADDITIONAL_PANE;
        let (fill, strong, weak) = {
            let v = ui.visuals();
            (v.panel_fill, v.strong_text_color(), v.weak_text_color())
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, fill);
        let Some(report) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.reports.get(ri))
        else {
            return;
        };
        let has_page = report.page.is_some();
        let show_page = has_page && self.slots[slot].planar[k].report_page;
        let title = if self.comparing() {
            format!("{} · {}", report.label, SLOT_NAMES[slot])
        } else {
            report.label.clone()
        };
        let subtitle = {
            let mut s = report.kind.label().to_string();
            if !report.date.is_empty() {
                s.push_str(" · ");
                s.push_str(&report.date);
            }
            s
        };

        // ---- the bar: as every pane has, plus the page / table switch ----
        let is_max = self.pane_is_max(ui, slot, kind);
        let bsize = egui::vec2(24.0, 20.0);
        let by = rect.top() + 4.0;
        let cell = |right: f32, w: f32| {
            Rect::from_min_size(Pos2::new(right - w, by), egui::vec2(w, bsize.y))
        };
        let max_rect = cell(rect.right() - 4.0, bsize.x);
        let close_rect = cell(max_rect.left() - 4.0, bsize.x);
        let detach_rect = cell(close_rect.left() - 4.0, bsize.x);
        let page_rect = cell(detach_rect.left() - 8.0, 70.0);
        let max_resp = ui
            .put(
                max_rect,
                egui::Button::new(if is_max { "⊞" } else { "⛶" }).small(),
            )
            .on_hover_text(if is_max {
                "Restore the multi-view layout"
            } else {
                "Maximize this pane to the whole window"
            });
        let close_resp = ui
            .put(close_rect, egui::Button::new("✖").small())
            .on_hover_text("Take this pane out of the row");
        let clicked_detach = self.detach_button(ui, detach_rect, slot, kind);
        let page_resp = has_page.then(|| {
            ui.put(
                page_rect,
                egui::Button::new(if show_page { "📋 Table" } else { "📄 Page" }).small(),
            )
            .on_hover_text(if show_page {
                "The content as a table"
            } else {
                "The page as the scanner drew it"
            })
        });

        // ---- the title ----
        let title_h = 40.0;
        if self.show_labels {
            let room = rect.width() - 140.0;
            let fitted = |text: String, size: f32, color: Color32| {
                let mut job = egui::text::LayoutJob::single_section(
                    text,
                    egui::TextFormat::simple(FontId::proportional(size), color),
                );
                job.wrap = egui::text::TextWrapping::truncate_at_width(room.max(40.0));
                painter.layout_job(job)
            };
            painter.galley(
                rect.left_top() + Vec2::new(6.0, 4.0),
                fitted(title, 14.0, strong),
                strong,
            );
            painter.galley(
                rect.left_top() + Vec2::new(6.0, 22.0),
                fitted(subtitle, 12.0, weak),
                weak,
            );
        }

        // ---- the content ----
        let body = Rect::from_min_max(
            Pos2::new(rect.left() + 6.0, rect.top() + title_h),
            Pos2::new(rect.right() - 6.0, rect.bottom() - 6.0),
        );
        if report.document.is_some() {
            self.report_document(ui, slot, ri, body);
        } else if show_page {
            self.report_page_image(ui, slot, ri, body);
        } else {
            let report = report.clone();
            let mut child = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(body)
                    .layout(egui::Layout::top_down(egui::Align::Min)),
            );
            child.set_clip_rect(body.intersect(ui.clip_rect()));
            egui::ScrollArea::both()
                .id_salt(("report_pane", slot))
                .auto_shrink([false, false])
                .show(&mut child, |ui| {
                    report_body(ui, &report, slot, ri);
                });
        }

        // ---- apply ----
        let resp = ui.interact(
            Rect::from_min_max(rect.min, Pos2::new(rect.right(), rect.top() + title_h)),
            egui::Id::new(("report_pane_title", slot)),
            Sense::hover(),
        );
        if resp.hovered() {
            self.hovered_slot = slot;
        }
        if max_resp.clicked() {
            self.toggle_pane_max(ui, slot, kind);
        }
        if clicked_detach {
            self.toggle_detached(slot, kind);
        }
        if page_resp.is_some_and(|r| r.clicked()) {
            let v = &mut self.slots[slot].planar[k];
            v.report_page = !v.report_page;
        }
        if close_resp.clicked() {
            self.remove_from_row(slot, kind);
        }
    }

    /// The page of report `ri` fitted into `body`, as the scanner drew it:
    /// white on black, no window to speak of.
    fn report_page_image(&mut self, ui: &mut egui::Ui, slot: usize, ri: usize, body: Rect) {
        let k = super::planar::ADDITIONAL_PANE;
        let ctx = ui.ctx().clone();
        let Some(img) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.reports.get(ri))
            .and_then(|r| r.page.as_ref())
        else {
            return;
        };
        let key: PlanarKey = (img.data.as_ptr() as usize, img.cols, img.rows);
        let (w, h) = (
            img.cols as f32 * img.spacing[0].max(1e-6) as f32,
            img.rows as f32 * img.spacing[1].max(1e-6) as f32,
        );
        let v = &self.slots[slot].planar[k];
        if v.page_key != Some(key) || v.page_tex.is_none() {
            // The page over its own values, as the loader set its window.
            let (c, wdt) = img.window;
            let lo = c - wdt * 0.5;
            let scale = 255.0 / wdt.max(1e-12);
            let ci = match &img.rgb {
                Some(rgb) if rgb.len() >= img.cols * img.rows * 3 => {
                    ColorImage::from_rgb([img.cols, img.rows], &rgb[..img.cols * img.rows * 3])
                }
                _ => ColorImage::new(
                    [img.cols, img.rows],
                    img.data
                        .iter()
                        .map(|&x| Color32::from_gray(((x - lo) * scale).clamp(0.0, 255.0) as u8))
                        .collect(),
                ),
            };
            let tex = ctx.load_texture(
                format!("report_page_{slot}_{ri}"),
                ci,
                TextureOptions::LINEAR,
            );
            let v = &mut self.slots[slot].planar[k];
            v.page_tex = Some(tex);
            v.page_key = Some(key);
        }
        let Some(tex) = &self.slots[slot].planar[k].page_tex else {
            return;
        };
        let painter = ui.painter_at(body);
        painter.rect_filled(body, 0.0, Color32::BLACK);
        let zoom = (body.width() / w).min(body.height() / h).max(1e-3) * 0.98;
        let shown = Rect::from_center_size(body.center(), Vec2::new(w * zoom, h * zoom));
        painter.image(
            tex.id(),
            shown,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    /// The document of report `ri` in `body`: its title and facts, then its
    /// pages as wide as the pane, top down; a spinner while they are drawn.
    fn report_document(&mut self, ui: &mut egui::Ui, slot: usize, ri: usize, body: Rect) {
        let k = super::planar::ADDITIONAL_PANE;
        let ctx = ui.ctx().clone();
        let Some(report) = self.slots[slot]
            .study
            .as_ref()
            .and_then(|st| st.reports.get(ri))
        else {
            return;
        };
        let Some(doc) = report.document.clone() else {
            return;
        };
        let mut info = report.info.clone();
        if !doc.title.is_empty() {
            info.insert(0, ("Title".to_string(), doc.title.clone()));
        }
        let key = std::sync::Arc::as_ptr(&doc.bytes) as usize;
        let state = doc.pages();
        if let Some(Ok(r)) = state {
            let v = &mut self.slots[slot].planar[k];
            if v.doc_key != Some(key) {
                v.doc_tex = r
                    .pages
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        ctx.load_texture(
                            format!("report_doc_{slot}_{ri}_{i}"),
                            ColorImage::from_rgba_premultiplied([p.width, p.height], &p.rgba),
                            TextureOptions::LINEAR,
                        )
                    })
                    .collect();
                v.doc_key = Some(key);
            }
        }
        let tex = self.slots[slot].planar[k].doc_tex.clone();
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(body)
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        child.set_clip_rect(body.intersect(ui.clip_rect()));
        egui::ScrollArea::vertical()
            .id_salt(("report_doc", slot))
            .auto_shrink([false, false])
            .show(&mut child, |ui| {
                ui.add_space(2.0);
                report_facts(ui, &info, slot, ri);
                match state {
                    None => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.weak("Drawing the pages…");
                        });
                        ctx.request_repaint_after(std::time::Duration::from_millis(100));
                    }
                    Some(Err(e)) => {
                        ui.weak(format!("The document cannot be shown: {e}"));
                    }
                    Some(Ok(r)) => {
                        let w = (ui.available_width() - 4.0).max(40.0);
                        for (p, t) in r.pages.iter().zip(&tex) {
                            let h = w * p.height as f32 / p.width.max(1) as f32;
                            ui.add(egui::Image::new((t.id(), Vec2::new(w, h))));
                            ui.add_space(6.0);
                        }
                        if r.total > r.pages.len() {
                            ui.weak(format!(
                                "{} more page(s) not drawn: the first {} are shown",
                                r.total - r.pages.len(),
                                r.pages.len()
                            ));
                        }
                    }
                }
            });
    }
}

/// The facts of a report, as a two-column grid.
fn report_facts(ui: &mut egui::Ui, info: &[(String, String)], slot: usize, ri: usize) {
    if info.is_empty() {
        return;
    }
    egui::Grid::new(("report_facts", slot, ri))
        .num_columns(2)
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            for (k, v) in info {
                ui.weak(k);
                ui.label(v);
                ui.end_row();
            }
        });
    ui.add_space(6.0);
}

/// The facts, the lines and the tables of a report, top down.
fn report_body(ui: &mut egui::Ui, report: &crate::report::Report, slot: usize, ri: usize) {
    ui.add_space(2.0);
    report_facts(ui, &report.info, slot, ri);
    for line in &report.notes {
        ui.label(line);
    }
    if !report.notes.is_empty() {
        ui.add_space(6.0);
    }
    for (ti, t) in report.all_tables(false).iter().enumerate() {
        if !t.title.is_empty() {
            ui.strong(&t.title);
        }
        egui::Grid::new(("report_table", slot, ri, ti))
            .striped(true)
            .min_col_width(10.0)
            .spacing([14.0, 3.0])
            .show(ui, |ui| {
                for c in &t.columns {
                    ui.strong(c);
                }
                ui.end_row();
                for row in &t.rows {
                    for cell in row {
                        ui.label(cell);
                    }
                    ui.end_row();
                }
            });
        ui.add_space(8.0);
    }
    if report.info.is_empty()
        && report.notes.is_empty()
        && report.tables.is_empty()
        && report.tree.is_empty()
    {
        ui.weak("Nothing readable in this report");
    }
}
