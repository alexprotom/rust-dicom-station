//! Theme-dependent colors.
//!
//! The image viewports stay black in both themes - that is the convention in
//! clinical viewers, keeps grayscale windowing and the dose colorwash reading
//! correctly, and lets the overlay annotations use one fixed palette. Only the
//! surrounding chrome and the few hand-painted accents follow the theme.

use super::*;

/// Fill of the area around and between the viewports.
pub(super) fn backdrop_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_gray(10)
    } else {
        Color32::from_gray(190)
    }
}

/// Fill of an empty study row (slightly lifted off the backdrop).
pub(super) fn empty_row_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_gray(14)
    } else {
        Color32::from_gray(205)
    }
}

/// Amber accent for warnings - darkened in light mode, where pale yellow on
/// white is unreadable.
pub(super) fn warn_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(240, 190, 60)
    } else {
        Color32::from_rgb(146, 98, 0)
    }
}

/// Red-orange accent for values needing attention (e.g. an abnormal
/// treatment termination status).
pub(super) fn alert_color(visuals: &egui::Visuals) -> Color32 {
    if visuals.dark_mode {
        Color32::from_rgb(240, 120, 60)
    } else {
        Color32::from_rgb(176, 56, 8)
    }
}

#[cfg(test)]
mod theme_tests {
    use super::*;

    /// Relative luminance per WCAG 2.1.
    fn luminance(c: Color32) -> f64 {
        let ch = |v: u8| {
            let s = v as f64 / 255.0;
            if s <= 0.03928 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * ch(c.r()) + 0.7152 * ch(c.g()) + 0.0722 * ch(c.b())
    }

    /// WCAG contrast ratio between two opaque colors (1.0 … 21.0).
    fn contrast(a: Color32, b: Color32) -> f64 {
        let (la, lb) = (luminance(a), luminance(b));
        (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
    }

    /// The hand-picked accents must stay legible on the panel background of
    /// *both* themes - a pale amber that works on near-black is unreadable on
    /// egui's light `panel_fill` (gray 248), which is exactly the regression
    /// this guards against. 4.5 is the WCAG AA threshold for body text.
    #[test]
    fn accent_colors_are_legible_in_both_themes() {
        for visuals in [egui::Visuals::dark(), egui::Visuals::light()] {
            let bg = visuals.panel_fill;
            let name = if visuals.dark_mode { "dark" } else { "light" };
            for (label, color) in [
                ("warn", warn_color(&visuals)),
                ("alert", alert_color(&visuals)),
            ] {
                let ratio = contrast(color, bg);
                assert!(
                    ratio >= 4.5,
                    "{name} theme: {label} accent {color:?} on {bg:?} has contrast {ratio:.2}"
                );
            }
        }
    }

    /// The viewport gutter and an empty study row must be distinguishable from
    /// each other and from the panels, in both themes.
    #[test]
    fn backdrops_stay_distinguishable() {
        for visuals in [egui::Visuals::dark(), egui::Visuals::light()] {
            let name = if visuals.dark_mode { "dark" } else { "light" };
            let (backdrop, row) = (backdrop_color(&visuals), empty_row_color(&visuals));
            assert_ne!(backdrop, row, "{name} theme: gutter equals empty row");
            // An empty row shows hint text; it needs real contrast against it.
            let ratio = contrast(visuals.text_color(), row);
            assert!(
                ratio >= 3.0,
                "{name} theme: hint text on an empty row has contrast {ratio:.2}"
            );
            // The gutter must read as a frame around the black viewports, not
            // blend into them.
            if !visuals.dark_mode {
                assert!(
                    contrast(backdrop, Color32::BLACK) >= 4.5,
                    "{name} theme: gutter is too dark to frame the viewports"
                );
            }
        }
    }
}
