//! rust-dicom-station — a fast, robust DICOM / RT DICOM viewer in pure Rust.
//!
//! Usage: `rust-dicom-station [DICOM_DIRECTORY] [COMPARISON_DIRECTORY]`
//!
//! With two directories, comparison mode starts automatically (study A on
//! top, study B below — six views total).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use rust_dicom_station::app;

use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let initial_path: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);
    let initial_path_b: Option<PathBuf> = std::env::args().nth(2).map(PathBuf::from);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust DICOM Station: Viewer")
            .with_inner_size([1680.0, 940.0])
            .with_min_inner_size([900.0, 520.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "rust-dicom-station",
        options,
        Box::new(move |cc| {
            Ok(Box::new(app::ViewerApp::new(
                cc,
                initial_path,
                initial_path_b,
            )))
        }),
    )
}
