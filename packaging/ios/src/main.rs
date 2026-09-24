//! rust-dicom-station on iPadOS and iOS: the same viewer behind UIKit, on
//! an iPad or an iPhone.
//!
//! An iOS app does start at `main`, but it never returns from it: `winit`
//! hands the main thread to `UIApplicationMain`, and from then on UIKit
//! calls in. Everything drawn is the desktop viewer: the same
//! [`rust_dicom_station::app::ViewerApp`], the same modules, the same
//! windows - drawn inside the one window iOS gives an app, since egui embeds
//! the viewports on a platform without native ones.
//!
//! What this crate does that `src/main.rs` does not:
//!
//! * the settings live in the app's `Library/Application Support` and the
//!   data (models, archive, test data) in its `Documents`, the folder the
//!   Files app shows as *On My iPad* (or *On My iPhone*) *> Rust DICOM
//!   Station*
//!   (`settings::config_dir` / `settings::data_dir` have iOS arms);
//!   `Documents` is kept out of iCloud and computer backups;
//! * a folder anywhere else - iCloud Drive, a USB drive, a file server - is
//!   granted through the system's folder picker, registered with the
//!   viewer's file browser as [`places::Files`] and remembered across
//!   launches as a bookmark;
//! * the status bar and the home indicator are laid over the window, so the
//!   [`Shell`] keeps those strips free ([`safe_area`]) and the viewer's
//!   menu bar starts below the clock;
//! * on a screen smaller than the desktop window's minimum - any iPhone,
//!   an iPad mini in portrait, Split View - the whole UI is drawn at a
//!   smaller zoom so that the desktop layout still fits ([`fit`]);
//! * a panic is written to `last_panic.txt` in `Documents`, where it can be
//!   read in the Files app on an iPad that has never seen Xcode; log lines
//!   go to standard error, which Xcode's console, `xcrun devicectl ...
//!   --console` and the simulator show.
//!
//! The graphics device is asked for no more than an iOS GPU has ([`gpu`]):
//! egui-wgpu's default request is one inter-stage variable over it.
//!
//! There is no fallback loop over graphics backends as on the desktop:
//! `winit` allows one event loop per process on iOS, and that loop never
//! returns, so the choice is made once. Metal is the only backend Apple
//! offers, and `Auto` picks it.

#![cfg_attr(not(target_os = "ios"), allow(dead_code))]

mod fit;
mod gpu;
#[cfg(target_os = "ios")]
mod places;
mod safe_area;

/// Written into `Documents` when the program panics.
const PANIC_FILE: &str = "last_panic.txt";

#[cfg(target_os = "ios")]
fn main() {
    use rust_dicom_station::{gfx, settings};

    log::set_logger(&STDERR_LOGGER)
        .map(|()| log::set_max_level(log::LevelFilter::Info))
        .ok();

    let config = settings::config_dir();
    let data = settings::data_dir();
    for d in [&config, &data] {
        if let Err(e) = std::fs::create_dir_all(d) {
            log::warn!("could not create {}: {e}", d.display());
        }
    }
    install_panic_hook(data.join(PANIC_FILE));
    places::exclude_from_backup(&data);
    log::info!(
        "rust-dicom-station {} starting; settings in {}, data in {}",
        env!("RDS_VERSION"),
        config.display(),
        data.display()
    );

    places::restore();
    settings::ios::set_places(Box::new(places::Files::new()));

    // As on the desktop: the environment wins over the settings file, and
    // the inference backend reads the choice from the environment.
    let preferred = gfx::from_env().unwrap_or_else(|| settings::load().graphics_backend);
    preferred.export();
    let backend = if preferred.available_here() {
        preferred
    } else {
        gfx::Backend::Auto
    };
    log::info!("graphics backend: {}", backend.label());

    let mut wgpu_options = eframe::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = backend.bits();
        // egui-wgpu's own request, lowered to what an iOS GPU has (gpu.rs).
        setup.device_descriptor =
            std::sync::Arc::new(
                |adapter: &eframe::wgpu::Adapter| eframe::wgpu::DeviceDescriptor {
                    label: Some("egui wgpu device"),
                    required_limits: gpu::device_limits(&adapter.limits()),
                    ..Default::default()
                },
            );
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Rust DICOM Station: Viewer"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        ..Default::default()
    };
    // On success this does not return: UIKit ends the process.
    let result = eframe::run_native(
        "rust-dicom-station",
        options,
        Box::new(|cc| Ok(Box::new(Shell::new(cc)))),
    );
    match result {
        Ok(()) => log::info!("rust-dicom-station finished"),
        Err(e) => {
            log::error!("rust-dicom-station could not run: {e}");
            std::process::exit(1);
        }
    }
}

/// Any other target: this crate is only the iOS front end.
#[cfg(not(target_os = "ios"))]
fn main() {
    eprintln!(
        "rds-ios is the iOS / iPadOS front end of rust-dicom-station; build it for \
         aarch64-apple-ios with packaging/ios/build-ipa.sh (docs/ios.md)."
    );
    std::process::exit(2);
}

/// The viewer, fitted to the screen, with the system's strips kept free
/// around it.
#[cfg(target_os = "ios")]
struct Shell {
    viewer: rust_dicom_station::app::ViewerApp,
    fit: fit::Fit,
}

#[cfg(target_os = "ios")]
impl Shell {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // The zoom belongs to `fit` here; Cmd and minus on a hardware
        // keyboard would only be undone on the next frame.
        cc.egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);
        Self {
            viewer: rust_dicom_station::app::ViewerApp::new(cc, None, None),
            fit: fit::Fit::default(),
        }
    }
}

#[cfg(target_os = "ios")]
impl eframe::App for Shell {
    fn raw_input_hook(&mut self, ctx: &egui::Context, raw_input: &mut egui::RawInput) {
        self.fit.raw_input(ctx, raw_input);
    }

    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        safe_area::reserve(ui);
        self.viewer.ui(ui, frame);
    }
}

/// `log` to standard error, one line per record.
struct StderrLogger;

static STDERR_LOGGER: StderrLogger = StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        // The graphics stack is chatty at Info; its warnings are enough.
        metadata.level() <= log::Level::Warn || metadata.target().starts_with("rust_dicom_station")
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!(
                "[rds {}] {}: {}",
                record.level(),
                record.target(),
                record.args()
            );
        }
    }

    fn flush(&self) {}
}

/// Log a panic and keep it in a file, since an iPad or iPhone rarely has
/// Xcode attached. The file is in `Documents`, so the Files app shows it.
fn install_panic_hook(file: std::path::PathBuf) {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let text = format!(
            "rust-dicom-station {} panicked: {info}\n{}",
            env!("RDS_VERSION"),
            std::backtrace::Backtrace::force_capture()
        );
        log::error!("{text}");
        if let Err(e) = std::fs::write(&file, &text) {
            log::error!("could not write {}: {e}", file.display());
        }
        previous(info);
    }));
}
