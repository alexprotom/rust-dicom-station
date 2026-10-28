//! The row's chart pane ([`PaneKind::Chart`]): the chart of whichever module
//! is showing one, where there is room to read it.
//!
//! Two modules draw charts - the dose-volume histograms and the curves of a
//! finished motion run - and a chart at the width of the side panel is not
//! one a plan is read from. So the module keeps its pickers and tables in
//! the panel and puts its chart here, in a pane of the row, which can be
//! maximized to the whole window like any other.
//!
//! The pane is the report pane (`report_view.rs`) with a chart in place of
//! the tables: the panel's own background, a title, a bar of its own
//! (*Save image* of this pane alone, maximize, close) and the content in a
//! child `Ui` over the body. What it shows is one [`ChartSource`] per
//! workspace, set by the module that computed something; with nothing to
//! show the pane is not drawn ([`ViewerApp::chart_source`]), exactly as the
//! additional pane with no report and no object.
//!
//! The chart talks back to the views the way the sections do, through the
//! module's own state: the curve under the pointer is kept as the module's
//! hover, which its section turns into the structure the views draw
//! thicker on the next pass.

use super::*;

/// What a workspace's chart pane shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum ChartSource {
    /// The plot of the *Dose-volume histograms* module.
    Dvh,
    /// The charts of one finished motion run, by its index in
    /// `motion_reports`.
    Motion(usize),
}

impl ChartSource {
    fn title(self) -> &'static str {
        match self {
            ChartSource::Dvh => "Dose-volume histograms",
            ChartSource::Motion(_) => "Structure motion",
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            ChartSource::Dvh => "dvh.png",
            ChartSource::Motion(_) => "motion.png",
        }
    }
}

impl ViewerApp {
    /// What `slot`'s chart pane has to show, if anything: the source it was
    /// given, while that still exists.
    pub(super) fn chart_source(&self, slot: usize) -> Option<ChartSource> {
        let src = (*self.chart.get(slot)?)?;
        let live = match src {
            ChartSource::Dvh => self.dvh.is_some(),
            ChartSource::Motion(i) => i < self.motion_reports.len(),
        };
        live.then_some(src)
    }

    /// Give `slot`'s chart pane something to show and put the pane into the
    /// row - what a module does once it has computed a chart.
    pub(super) fn show_chart(&mut self, slot: usize, src: ChartSource) {
        let slot = slot.min(MAX_WORKSPACES - 1);
        self.chart[slot] = Some(src);
        self.add_to_row(slot, PaneKind::Chart);
    }

    /// Does any row show this chart now?
    pub(super) fn chart_shown(&self, src: ChartSource) -> bool {
        self.open_slots().into_iter().any(|s| {
            self.chart_source(s) == Some(src) && self.row_panes(s).contains(&PaneKind::Chart)
        })
    }

    /// The chart pane of `slot`, filling `rect`.
    pub(super) fn chart_pane(&mut self, ui: &mut egui::Ui, slot: usize, rect: Rect) {
        let kind = PaneKind::Chart;
        let (fill, strong, weak) = {
            let v = ui.visuals();
            (v.panel_fill, v.strong_text_color(), v.weak_text_color())
        };
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, fill);
        let Some(src) = self.chart_source(slot) else {
            return;
        };
        let title = if self.comparing() {
            format!("{} · {}", src.title(), SLOT_NAMES[slot])
        } else {
            src.title().to_string()
        };
        let subtitle = match src {
            ChartSource::Dvh => self
                .dvh
                .as_ref()
                .and_then(|d| d.status.clone())
                .unwrap_or_default(),
            ChartSource::Motion(i) => self
                .motion_reports
                .get(i)
                .map(|r| r.run_name.clone())
                .unwrap_or_default(),
        };

        // ---- the bar: as every pane has, plus *Save image* ----
        let is_max = self.pane_is_max(ui, slot, kind);
        let out = self.is_detached(slot, kind);
        let bsize = egui::vec2(24.0, 20.0);
        let by = rect.top() + 4.0;
        let cell = |right: f32, w: f32| {
            Rect::from_min_size(Pos2::new(right - w, by), egui::vec2(w, bsize.y))
        };
        let max_rect = cell(rect.right() - 4.0, bsize.x);
        let close_rect = cell(max_rect.left() - 4.0, bsize.x);
        let detach_rect = cell(close_rect.left() - 4.0, bsize.x);
        let save_rect = cell(detach_rect.left() - 4.0, bsize.x);
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
        // The picture is cut from a screenshot of the main window, so a
        // chart in a window of its own is saved from its row.
        let save_resp = ui
            .put(save_rect, egui::Button::new("💾").small().sense(if out {
                Sense::hover()
            } else {
                Sense::click()
            }))
            .on_hover_text(if out {
                "Put the chart back into its row to save it as a picture: the picture is                  cut from the main window"
            } else {
                "Save this chart as a picture, as it is on screen"
            });

        // ---- the title ----
        let title_h = 40.0;
        let room = rect.width() - 100.0;
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

        // ---- the chart ----
        let body = Rect::from_min_max(
            Pos2::new(rect.left() + 6.0, rect.top() + title_h),
            Pos2::new(rect.right() - 6.0, rect.bottom() - 6.0),
        );
        let mut child = ui.new_child(
            egui::UiBuilder::new()
                .max_rect(body)
                .layout(egui::Layout::top_down(egui::Align::Min)),
        );
        child.set_clip_rect(body.intersect(ui.clip_rect()));
        match src {
            ChartSource::Dvh => self.dvh_chart(&mut child, body),
            ChartSource::Motion(i) => self.motion_chart(&mut child, slot, i, body),
        }

        // ---- apply ----
        let resp = ui.interact(
            Rect::from_min_max(rect.min, Pos2::new(rect.right(), rect.top() + title_h)),
            egui::Id::new(("chart_pane_title", slot)),
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
        if save_resp.clicked() {
            let ppp = ui.ctx().pixels_per_point();
            self.save_rect_image(rect, ppp, src.file_name());
        }
        if close_resp.clicked() {
            self.chart[slot] = None;
            self.remove_from_row(slot, kind);
        }
    }
}
