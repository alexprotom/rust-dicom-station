//! rust-dicom-station - a fast, robust DICOM / RT DICOM viewer in pure Rust.
//!
//! Usage: `rust-dicom-station [DICOM_DIRECTORY] [COMPARISON_DIRECTORY]`
//!
//! With two directories, comparison mode starts automatically (study A on
//! top, study B below - six views total).
//!
//! ## Starting on a machine whose Vulkan driver does not work
//!
//! Some Windows machines advertise a Vulkan driver that cannot actually
//! create a device. `wgpu` prefers Vulkan, so the program would die before
//! drawing anything, on a machine where nothing else is wrong. So the window
//! is not opened once but *attempted*: the preferred backend first, then the
//! rest of [`gfx::candidates`], and a backend that fails - by error or by
//! panicking somewhere inside the driver - costs a line on standard error
//! rather than the program. See [`rust_dicom_station::gfx`].

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use rust_dicom_station::{app, gfx, icon, settings};

use std::path::PathBuf;

fn main() -> eframe::Result<()> {
    let initial_path: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);
    let initial_path_b: Option<PathBuf> = std::env::args().nth(2).map(PathBuf::from);

    // Before anything else, and in particular before any thread exists: the
    // environment is how the inference backend is told which graphics API to
    // use, and writing it is only sound while this process is alone.
    // `WGPU_BACKEND` set by the user wins over the settings file, because
    // someone who set it is working around something.
    let preferred = gfx::from_env().unwrap_or_else(|| settings::load().graphics_backend);
    preferred.export();

    let order = gfx::candidates(preferred);
    let mut last: Option<eframe::Error> = None;
    for (attempt, backend) in order.iter().copied().enumerate() {
        if attempt > 0 {
            eprintln!(
                "rust-dicom-station: {} did not work, trying {}",
                order[attempt - 1].label(),
                backend.label()
            );
        }
        match run(backend, initial_path.clone(), initial_path_b.clone()) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("rust-dicom-station: {} failed: {e}", backend.label());
                last = Some(e);
            }
        }
    }
    eprintln!(
        "rust-dicom-station: no graphics backend on this machine could open a window. \
         Set {}=dx12 (or vulkan, or opengl) to force one, or choose it under \
         View > Graphics backend after the program starts.",
        gfx::ENV_VAR
    );
    Err(last.expect("candidates() is never empty"))
}

/// One attempt at opening the window on a named backend.
///
/// A backend that cannot initialise tends to panic somewhere inside the
/// driver rather than return, so the attempt is caught: the point of trying
/// several is defeated if the first one aborts the process.
fn run(
    backend: gfx::Backend,
    initial_path: Option<PathBuf>,
    initial_path_b: Option<PathBuf>,
) -> eframe::Result<()> {
    let mut wgpu_options = eframe::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = backend.bits();
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust DICOM Station: Viewer")
            .with_icon(icon::window_icon())
            .with_inner_size([1680.0, 940.0])
            .with_min_inner_size([900.0, 520.0]),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };

    // `NativeOptions` holds boxed callbacks that are not `UnwindSafe`, which
    // is a fair warning in general and irrelevant here: nothing is read back
    // after a failed attempt - the next one builds its own options from
    // scratch.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
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
    }));
    match result {
        Ok(r) => r,
        Err(_) => Err(eframe::Error::AppCreation(
            format!("the {} driver crashed while starting", backend.label()).into(),
        )),
    }
}
