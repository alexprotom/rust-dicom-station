//! Keep the strips the system draws over free.
//!
//! The window covers the whole screen: on an iPad the status bar at the top
//! and the home indicator at the bottom included, on an iPhone in landscape
//! the Dynamic Island on one side, the rounded corners and the home
//! indicator. UIKit reports how
//! much of each edge is covered as the window's *safe-area insets*;
//! `egui-winit` reads them on iOS and hands them to egui, which is why
//! [`egui::Context::content_rect`] is smaller than
//! [`egui::Context::viewport_rect`] there ([`crate::fit`] rescales the
//! insets into egui's points first, so this holds at any zoom). The root
//! `Ui` an `eframe` app
//! draws into still spans the whole viewport, though, so the viewer's menu
//! bar would sit under the clock.
//!
//! [`reserve`] therefore puts an empty, frameless panel of exactly the
//! covered size on each edge before the viewer draws, the same way the
//! Android front end does with the insets it reads over JNI. Nothing here
//! calls UIKit: the numbers arrive through egui, which also makes the
//! arithmetic testable on any machine.

/// How much of each edge is covered, in points, rounded up to whole points
/// so that no half-covered row of pixels is left to the viewer.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Strips {
    pub top: f32,
    pub bottom: f32,
    pub left: f32,
    pub right: f32,
}

/// The strips between the full `viewport` and the safe `content` rectangle.
/// A content rectangle that reaches past the viewport (which UIKit never
/// reports, but a rotation in flight can briefly produce) counts as no strip.
pub fn strips(viewport: egui::Rect, content: egui::Rect) -> Strips {
    let edge = |covered: f32| {
        if covered.is_finite() && covered > 0.0 {
            covered.ceil()
        } else {
            0.0
        }
    };
    Strips {
        top: edge(content.top() - viewport.top()),
        bottom: edge(viewport.bottom() - content.bottom()),
        left: edge(content.left() - viewport.left()),
        right: edge(viewport.right() - content.right()),
    }
}

/// Draw the empty panels for this frame, before anything else is drawn.
pub fn reserve(ui: &mut egui::Ui) {
    let ctx = ui.ctx().clone();
    let s = strips(ctx.viewport_rect(), ctx.content_rect());
    let blank = egui::Frame::NONE;
    if s.top > 0.0 {
        egui::Panel::top(egui::Id::new("ios_safe_area_top"))
            .exact_size(s.top)
            .frame(blank)
            .show_separator_line(false)
            .resizable(false)
            .show(ui, |_| {});
    }
    if s.bottom > 0.0 {
        egui::Panel::bottom(egui::Id::new("ios_safe_area_bottom"))
            .exact_size(s.bottom)
            .frame(blank)
            .show_separator_line(false)
            .resizable(false)
            .show(ui, |_| {});
    }
    if s.left > 0.0 {
        egui::Panel::left(egui::Id::new("ios_safe_area_left"))
            .exact_size(s.left)
            .frame(blank)
            .show_separator_line(false)
            .resizable(false)
            .show(ui, |_| {});
    }
    if s.right > 0.0 {
        egui::Panel::right(egui::Id::new("ios_safe_area_right"))
            .exact_size(s.right)
            .frame(blank)
            .show_separator_line(false)
            .resizable(false)
            .show(ui, |_| {});
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, Rect};

    #[test]
    fn an_ipad_in_landscape_loses_the_status_bar_and_the_home_indicator() {
        // 1366 x 1024 points (a 12.9-inch iPad), 24 points of status bar,
        // 20 of home indicator.
        let viewport = Rect::from_min_max(pos2(0.0, 0.0), pos2(1366.0, 1024.0));
        let content = Rect::from_min_max(pos2(0.0, 24.0), pos2(1366.0, 1004.0));
        assert_eq!(
            strips(viewport, content),
            Strips {
                top: 24.0,
                bottom: 20.0,
                left: 0.0,
                right: 0.0
            }
        );
    }

    #[test]
    fn fractional_insets_round_up_and_nothing_negative_survives() {
        let viewport = Rect::from_min_max(pos2(0.0, 0.0), pos2(1000.0, 800.0));
        let content = Rect::from_min_max(pos2(47.2, 0.0), pos2(1000.5, 779.4));
        let s = strips(viewport, content);
        assert_eq!(s.left, 48.0);
        assert_eq!(s.bottom, 21.0);
        assert_eq!(s.top, 0.0);
        // The content reaching past the right edge is not a strip.
        assert_eq!(s.right, 0.0);
        assert_eq!(strips(viewport, viewport), Strips::default());
    }

    /// The panels take exactly the covered strips, so what is left for the
    /// viewer is the safe rectangle - checked on a headless egui context
    /// fed with the insets `egui-winit` would report.
    #[test]
    fn the_viewer_is_left_the_safe_rectangle() {
        let ctx = egui::Context::default();
        let mut input = egui::RawInput {
            screen_rect: Some(Rect::from_min_max(pos2(0.0, 0.0), pos2(1180.0, 820.0))),
            safe_area_insets: Some(egui::SafeAreaInsets(egui::epaint::MarginF32 {
                top: 24.0,
                bottom: 20.0,
                left: 0.0,
                right: 0.0,
            })),
            ..Default::default()
        };
        let mut left_over = Rect::NOTHING;
        for _ in 0..2 {
            let mut out = ctx.run_ui(input.clone(), |ui| {
                reserve(ui);
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        left_over = ui.max_rect();
                    });
            });
            // egui keeps the last insets it was given.
            input.safe_area_insets = None;
            // No renderer here to take the font texture; dropping it
            // unhandled is what epaint refuses.
            out.textures_delta.clear();
        }
        assert_eq!(left_over.top(), 24.0);
        assert_eq!(left_over.bottom(), 800.0);
        assert_eq!(left_over.left(), 0.0);
        assert_eq!(left_over.right(), 1180.0);
    }
}
