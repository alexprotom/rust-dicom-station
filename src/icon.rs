//! The application's picture of itself.
//!
//! One file, `assets/rust-dicom-station.png`, is the icon everywhere: the
//! window icon of the viewer and of the installer (loaded here), the icon
//! resource compiled into both Windows executables (`build.rs`, from the
//! `.ico` beside it), and the desktop icon of the Linux AppImage. Explorer,
//! the task bar, Alt+Tab, the start-menu shortcut and *Add or remove
//! programs* therefore all show the same thing.
use egui::IconData;

/// The window icon, decoded from the PNG compiled into the binary.
///
/// A picture that fails to decode is not worth refusing to start over: the
/// window then opens with the platform's default icon, which is exactly what
/// happened before this existed.
pub fn window_icon() -> IconData {
    eframe::icon_data::from_png_bytes(include_bytes!("../assets/rust-dicom-station.png"))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_icon_decodes_to_a_square_image() {
        let icon = window_icon();
        assert_eq!(
            icon.width, icon.height,
            "the icon has to be square - window managers scale it to their own sizes"
        );
        assert!(
            icon.width >= 128,
            "a {}px icon is too small for a task bar on a high-DPI screen",
            icon.width
        );
        assert_eq!(
            icon.rgba.len(),
            (icon.width * icon.height * 4) as usize,
            "decoded to something that is not RGBA"
        );
        assert!(
            icon.rgba.as_chunks::<4>().0.iter().any(|px| px[3] > 0),
            "every pixel is transparent: the icon file is blank"
        );
    }
}
