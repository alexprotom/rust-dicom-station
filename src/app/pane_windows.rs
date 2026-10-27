//! Panes of a row in windows of their own.
//!
//! Every pane's bar has a ⤴ button (on an image or a 3D pane, among the ones
//! the fold shows) that takes the pane out of its row and into a window of
//! the operating system - one more image on a
//! second monitor, a DVH beside the views rather than squeezed into a third
//! of a row. The pane is the same pane: the same function draws it, with
//! the same state (slice, zoom, window, the planar stack, the 3D camera),
//! only into the window's area instead of a column of the row. The row
//! closes up around the gap, and ⤵ in the window's bar, or closing the
//! window, puts the pane back where it was.
//!
//! In a window the bar's ⛶ maximizes the window itself, since there is no
//! row to fill. Which panes are out is not remembered between runs, like
//! the tool windows' *Keep on top*: a window left on a monitor that is not
//! there next time is worse than a row that comes back whole.

use super::detach::{tool_window, WinOpts};
use super::*;

impl ViewerApp {
    /// Is this pane drawn in a window of its own rather than in its row?
    pub(super) fn is_detached(&self, slot: usize, kind: PaneKind) -> bool {
        self.detached_panes.contains(&(slot, kind))
    }

    /// Take a pane out of its row into a window, or put it back.
    pub(super) fn toggle_detached(&mut self, slot: usize, kind: PaneKind) {
        if let Some(i) = self.detached_panes.iter().position(|p| *p == (slot, kind)) {
            self.detached_panes.remove(i);
        } else {
            self.detached_panes.push((slot, kind));
            // A pane cannot fill the main window and a window of its own.
            if self.maximized == Some((slot, kind)) {
                self.maximized = None;
            }
            // Nor be filmed where it no longer is: a recording crops the
            // main window to the rectangle the pane last had there.
            self.pane_rects.retain(|(s, k, _)| (*s, *k) != (slot, kind));
        }
    }

    /// The bar's ⤴ / ⤵ button at `rect`; whether it was clicked.
    pub(super) fn detach_button(
        &self,
        ui: &mut egui::Ui,
        rect: Rect,
        slot: usize,
        kind: PaneKind,
    ) -> bool {
        let out = self.is_detached(slot, kind);
        ui.put(rect, egui::Button::new(if out { "⤵" } else { "⤴" }).small())
            .on_hover_text(if out {
                "Put this view back into its row. Closing the window does the same."
            } else {
                "Take this view out of the row into a window of its own, to put on \
             another monitor or size as you like. The rest of the row closes up."
            })
            .clicked()
    }

    /// Is the pane maximized: in the row, over the whole central area; in a
    /// window of its own, the window over the screen.
    pub(super) fn pane_is_max(&self, ui: &egui::Ui, slot: usize, kind: PaneKind) -> bool {
        if self.is_detached(slot, kind) {
            ui.input(|i| i.viewport().maximized.unwrap_or(false))
        } else {
            self.maximized == Some((slot, kind))
        }
    }

    /// What the bar's ⛶ / ⊞ does for this pane.
    pub(super) fn toggle_pane_max(&mut self, ui: &egui::Ui, slot: usize, kind: PaneKind) {
        let on = self.pane_is_max(ui, slot, kind);
        if self.is_detached(slot, kind) {
            ui.ctx()
                .send_viewport_cmd(egui::ViewportCommand::Maximized(!on));
        } else {
            self.maximized = (!on).then_some((slot, kind));
        }
    }

    /// Draw the windows of the panes that are out of their rows, once a
    /// pass after the central area. A pane with nothing left to show - its
    /// workspace closed, its chart or image gone - goes back to its row,
    /// which leaves it out until it has something again.
    pub(super) fn pane_windows(&mut self, ctx: &egui::Context) {
        let panes = self.detached_panes.clone();
        for (slot, kind) in panes {
            if !self.is_open(slot)
                || self.slots[slot].study.is_none()
                || !self.row_layout(slot).contains(&kind)
                || !self.pane_available(slot, kind)
            {
                self.detached_panes.retain(|p| *p != (slot, kind));
                continue;
            }
            let title = if self.comparing() {
                format!("{} · {}", kind.label(), SLOT_NAMES[slot])
            } else {
                kind.label().to_string()
            };
            let mut open = true;
            tool_window(
                ctx,
                &format!("pane_{slot}_{kind:?}"),
                title,
                &mut open,
                WinOpts::size(640.0, 600.0).no_scroll(),
                |ui| {
                    let rect = ui.available_rect_before_wrap();
                    // The pane places its widgets by rectangle; this tells
                    // the window (an `egui::Window` where there are no
                    // native ones) that the area is in use.
                    ui.expand_to_include_rect(rect);
                    ui.painter()
                        .rect_filled(rect, 0.0, backdrop_color(ui.visuals()));
                    self.pane(ui, slot, kind, rect, true);
                },
            );
            if !open {
                self.detached_panes.retain(|p| *p != (slot, kind));
            }
        }
    }
}
