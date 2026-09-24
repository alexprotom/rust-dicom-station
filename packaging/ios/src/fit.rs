//! Fit the desktop layout into the screen it is given.
//!
//! The viewer is laid out for a window of at least 900 x 520 points - the
//! minimum size `src/main.rs` gives the desktop window. An iPad in landscape
//! has more than that; an iPhone in landscape does not (a 6.1-inch one
//! leaves about 734 x 372 points once the Dynamic Island and the home
//! indicator are kept free), and neither does an iPad mini in portrait or
//! an iPad app in Split View. Rather than a second layout, the whole UI is
//! drawn smaller there: egui's zoom factor is set so that the safe part of
//! the screen measures at least the desktop minimum in egui's points, the
//! same thing a user does on the desktop with Ctrl and minus. On a screen
//! that is large enough the zoom stays 1 and nothing changes.
//!
//! Everything happens in [`Fit::raw_input`], before egui sees a frame's
//! input, and is measured in UIKit's points from that input rather than
//! from egui's rectangles: on the frame a zoom changes, egui rescales the
//! previous frame's safe rectangle and subtracts the insets from it a second
//! time, and a fit that measured egui's rectangle would take that for a
//! smaller screen and shrink again, down to [`MIN_ZOOM`].
//!
//! One correction comes with it. `egui-winit` reports the safe-area insets
//! in UIKit's points, and egui subtracts them from the screen in its own
//! points, which are UIKit's divided by the zoom; at a zoom below 1 the
//! strips would come out too narrow and the menu bar would slide under the
//! Dynamic Island. The insets are therefore rescaled before egui sees them,
//! so that `content_rect` - and with it every window's constraint, the file
//! browser and [`crate::safe_area`] - is right at any zoom.
//!
//! No UIKit here: tested on the host with a headless egui context.

use egui::{epaint::MarginF32, SafeAreaInsets, Vec2};

/// The desktop window's minimum size (`with_min_inner_size` in
/// `src/main.rs`), which the safe part of the screen is fitted to.
pub const MIN_POINTS: Vec2 = Vec2::new(900.0, 520.0);

/// Below this, text is no longer readable on any iPhone; a screen smaller
/// still (an iPad app squeezed into Slide Over) gets this zoom and a
/// layout that is tighter than the desktop minimum.
pub const MIN_ZOOM: f32 = 0.6;

/// The zoom factor for a safe area of `native` UIKit points: 1 when it
/// holds the desktop minimum, less when it does not, never below
/// [`MIN_ZOOM`].
pub fn zoom_for(native: Vec2) -> f32 {
    if !(native.x > 0.0 && native.y > 0.0) {
        return 1.0;
    }
    (native.x / MIN_POINTS.x)
        .min(native.y / MIN_POINTS.y)
        .clamp(MIN_ZOOM, 1.0)
}

/// The zoom chosen so far and what UIKit last reported.
pub struct Fit {
    /// The zoom factor the coming pass is drawn at.
    zoom: f32,
    /// The screen in UIKit points.
    native_screen: Vec2,
    /// The safe-area insets in UIKit points, as `egui-winit` last sent them.
    native_insets: SafeAreaInsets,
}

impl Default for Fit {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            native_screen: Vec2::ZERO,
            native_insets: SafeAreaInsets::default(),
        }
    }
}

impl Fit {
    /// Before egui reads a frame's input (`eframe::App::raw_input_hook`):
    ///
    /// * take the screen size and the insets `egui-winit` reported, in
    ///   UIKit points (the screen arrives in egui points at the zoom it was
    ///   measured with, which is the context's current one; the insets
    ///   arrive only when they may have changed, `None` meaning "as
    ///   before");
    /// * choose the zoom for the safe area that leaves, and hand it to egui,
    ///   which applies it from this very pass;
    /// * give egui the insets in its own points at that zoom.
    pub fn raw_input(&mut self, ctx: &egui::Context, raw: &mut egui::RawInput) {
        if let Some(insets) = raw.safe_area_insets {
            self.native_insets = insets;
        }
        if let Some(screen) = raw.screen_rect {
            self.native_screen = screen.size() * ctx.zoom_factor();
        }
        let m = self.native_insets.0;
        let safe = self.native_screen - Vec2::new(m.left + m.right, m.top + m.bottom);
        let want = zoom_for(safe);
        if (want - self.zoom).abs() > 0.01 {
            log::info!(
                "safe area {:.0} x {:.0} points: zoom {:.2}",
                safe.x,
                safe.y,
                want
            );
            self.zoom = want;
            ctx.set_zoom_factor(want);
        }
        let k = 1.0 / self.zoom;
        raw.safe_area_insets = Some(SafeAreaInsets(MarginF32 {
            left: m.left * k,
            right: m.right * k,
            top: m.top * k,
            bottom: m.bottom * k,
        }));
    }

    /// The zoom the next pass is drawn at.
    #[cfg(test)]
    pub fn zoom(&self) -> f32 {
        self.zoom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2, Rect};

    #[test]
    fn large_screens_keep_zoom_one_and_small_ones_are_fitted() {
        // 13-inch iPad in landscape, status bar and home indicator off.
        assert_eq!(zoom_for(vec2(1366.0, 980.0)), 1.0);
        // 11-inch iPad in portrait: 834 wide.
        assert!((zoom_for(vec2(834.0, 1150.0)) - 834.0 / 900.0).abs() < 1e-6);
        // 6.9-inch iPhone in landscape: 956 x 440 less 62 + 62 and 21.
        let z = zoom_for(vec2(832.0, 419.0));
        assert!((z - 419.0 / 520.0).abs() < 1e-6, "{z}");
        // 6.1-inch iPhone in landscape: 852 x 393 less 59 + 59 and 21.
        let z = zoom_for(vec2(734.0, 372.0));
        assert!((z - 372.0 / 520.0).abs() < 1e-6, "{z}");
        // Slide Over: clamped.
        assert_eq!(zoom_for(vec2(320.0, 700.0)), MIN_ZOOM);
        // Nothing measured yet.
        assert_eq!(zoom_for(Vec2::ZERO), 1.0);
    }

    /// Runs `frames` passes of a screen of `native` UIKit points with
    /// `insets`, the way `egui-winit` drives egui: the screen in egui points
    /// is the UIKit size divided by the context's zoom, the insets are UIKit
    /// points and sent once.
    fn settle(ctx: &egui::Context, fit: &mut Fit, native: Vec2, insets: MarginF32) {
        for frame in 0..4 {
            let mut raw = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    pos2(0.0, 0.0),
                    native / ctx.zoom_factor(),
                )),
                safe_area_insets: (frame == 0).then_some(SafeAreaInsets(insets)),
                ..Default::default()
            };
            fit.raw_input(ctx, &mut raw);
            let mut out = ctx.run_ui(raw, |_| {});
            out.textures_delta.clear();
        }
    }

    #[test]
    fn an_ipad_in_landscape_is_left_alone() {
        let ctx = egui::Context::default();
        let mut fit = Fit::default();
        let insets = MarginF32 {
            left: 0.0,
            right: 0.0,
            top: 24.0,
            bottom: 20.0,
        };
        settle(&ctx, &mut fit, vec2(1194.0, 834.0), insets);
        assert_eq!(ctx.zoom_factor(), 1.0);
        assert_eq!(ctx.content_rect().top(), 24.0);
        assert_eq!(ctx.content_rect().bottom(), 814.0);
    }

    /// A 6.1-inch iPhone in landscape, driven the way `egui-winit` drives
    /// egui: the screen in egui points is the UIKit size divided by the
    /// zoom, the insets are UIKit points. After the fit has settled, the
    /// safe area measures the desktop minimum's height in egui points and
    /// sits exactly inside the insets on the device.
    #[test]
    fn an_iphone_settles_at_a_zoom_where_the_safe_area_holds_the_desktop_minimum() {
        let ctx = egui::Context::default();
        let native = vec2(852.0, 393.0);
        let insets = MarginF32 {
            left: 59.0,
            right: 59.0,
            top: 0.0,
            bottom: 21.0,
        };
        let mut fit = Fit::default();
        settle(&ctx, &mut fit, native, insets);
        let z = ctx.zoom_factor();
        assert!((z - 372.0 / 520.0).abs() < 1e-3, "zoom {z}");
        assert_eq!(fit.zoom(), z);
        let safe = ctx.content_rect();
        // In egui points the safe area is at least the desktop minimum...
        assert!(safe.height() >= MIN_POINTS.y - 1.0, "{safe:?}");
        assert!(safe.width() >= MIN_POINTS.x - 1.0, "{safe:?}");
        // ...and on the device it starts right after the Dynamic Island.
        assert!((safe.left() * z - 59.0).abs() < 1.0, "{safe:?}");
        assert!(((ctx.viewport_rect().bottom() - safe.bottom()) * z - 21.0).abs() < 1.0);
    }
}
