//! rust-dicom-station on Android: the same viewer behind `android_main`.
//!
//! Android does not start a program at `main`. Its `NativeActivity` loads a
//! shared library from the package, and `android-activity` (the glue that
//! `winit` and `eframe` build on) calls [`android_main`] with a handle to
//! the activity. Everything from there on is the desktop viewer: the same
//! [`rust_dicom_station::app::ViewerApp`], the same modules, the same
//! windows - drawn inside the one window Android allows, since egui embeds
//! the viewports on a platform without native ones.
//!
//! What this crate does that `src/main.rs` does not:
//!
//! * the activity's two folders become the settings and data folders (see
//!   `settings::android`), since Android has no home directory;
//! * standard output and error are read back into logcat, where a crash
//!   can actually be seen; a panic is also written to a file in the app's
//!   folder, so a tablet without a debug cable still keeps the evidence;
//! * the one platform question is asked: *all files access*, without
//!   which the shared storage - the DICOM folders on the device or a USB
//!   stick - cannot be read. Android grants it on a system settings page;
//!   the [`Shell`] explains that and opens the page;
//! * the window is laid out edge to edge, under the status bar and the
//!   navigation bar, so the [`Shell`] keeps those strips free (see
//!   [`insets`]) and the viewer's menu bar starts below the clock.
//!
//! There is no fallback loop over graphics backends as on the desktop:
//! `winit` allows one event loop per process on Android, so the choice is
//! made once. `Auto` lets `wgpu` pick between Vulkan and OpenGL ES itself,
//! and the *Graphics backend* setting still decides the next start.

#![cfg(target_os = "android")]

use std::path::PathBuf;
use std::sync::Arc;

use android_activity::AndroidApp;
use rust_dicom_station::{app, gfx, settings};

mod insets;
mod permission;

/// Written into the app's private folder when the program panics.
const PANIC_FILE: &str = "last_panic.txt";

/// The tag every logcat line carries: `adb logcat -s rds`.
const LOG_TAG: &str = "rds";

/// Android's entry point (see the module documentation).
#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag(LOG_TAG),
    );
    redirect_stdio_to_logcat();

    let internal = app
        .internal_data_path()
        .unwrap_or_else(|| PathBuf::from("/data/local/tmp"));
    let external = app.external_data_path().unwrap_or_else(|| internal.clone());
    for d in [&internal, &external] {
        if let Err(e) = std::fs::create_dir_all(d) {
            log::warn!("could not create {}: {e}", d.display());
        }
    }
    install_panic_hook(internal.join(PANIC_FILE));
    settings::android::set_dirs(internal.clone(), external.clone());
    log::info!(
        "rust-dicom-station {} starting; settings in {}, data in {}",
        env!("CARGO_PKG_VERSION"),
        internal.display(),
        external.display()
    );

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
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("Rust DICOM Station: Viewer"),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        android_app: Some(app.clone()),
        ..Default::default()
    };
    let result = eframe::run_native(
        "rust-dicom-station",
        options,
        Box::new(move |cc| Ok(Box::new(Shell::new(cc, app)))),
    );
    match result {
        Ok(()) => log::info!("rust-dicom-station finished"),
        Err(e) => log::error!("rust-dicom-station could not run: {e}"),
    }
}

/// The viewer plus the one Android question drawn over it.
struct Shell {
    viewer: app::ViewerApp,
    app: AndroidApp,
    storage: StoragePrompt,
    /// What the system covers, in pixels, and when it was last read.
    insets: insets::Insets,
    insets_read: std::time::Instant,
    insets_for: egui::Rect,
}

impl Shell {
    fn new(cc: &eframe::CreationContext<'_>, app: AndroidApp) -> Self {
        let granted = permission::all_files_access(&app).unwrap_or_else(|e| {
            log::warn!("could not ask whether all files access is granted: {e}");
            true
        });
        Self {
            viewer: app::ViewerApp::new(cc, None, None),
            insets: insets::system_insets(&app).unwrap_or_else(|e| {
                log::warn!("could not read the window insets: {e}");
                insets::Insets::default()
            }),
            insets_read: std::time::Instant::now(),
            insets_for: egui::Rect::NOTHING,
            app,
            storage: StoragePrompt {
                granted,
                dismissed: false,
                last_check: std::time::Instant::now(),
                error: None,
            },
        }
    }

    /// Keep the strips the system draws over free, so the viewer's own
    /// bars start below the status bar and above the navigation bar.
    ///
    /// The insets are read again whenever the window changes size (a
    /// rotation moves the bars) and once a second otherwise, which is how a
    /// bar that appears or hides is noticed without a call per frame.
    fn reserve_system_insets(&mut self, ui: &mut egui::Ui) {
        let rect = ui.ctx().viewport_rect();
        if rect != self.insets_for || self.insets_read.elapsed() > std::time::Duration::from_secs(1)
        {
            self.insets_for = rect;
            self.insets_read = std::time::Instant::now();
            if let Ok(i) = insets::system_insets(&self.app) {
                self.insets = i;
            }
        }
        let ppp = ui.ctx().pixels_per_point().max(0.1);
        let strip = |px: i32| (px.max(0) as f32 / ppp).ceil();
        let blank = egui::Frame::NONE;
        let (top, bottom, left, right) = (
            strip(self.insets.top),
            strip(self.insets.bottom),
            strip(self.insets.left),
            strip(self.insets.right),
        );
        if top > 0.0 {
            egui::Panel::top(egui::Id::new("android_inset_top"))
                .exact_size(top)
                .frame(blank)
                .show_separator_line(false)
                .resizable(false)
                .show(ui, |_| {});
        }
        if bottom > 0.0 {
            egui::Panel::bottom(egui::Id::new("android_inset_bottom"))
                .exact_size(bottom)
                .frame(blank)
                .show_separator_line(false)
                .resizable(false)
                .show(ui, |_| {});
        }
        if left > 0.0 {
            egui::Panel::left(egui::Id::new("android_inset_left"))
                .exact_size(left)
                .frame(blank)
                .show_separator_line(false)
                .resizable(false)
                .show(ui, |_| {});
        }
        if right > 0.0 {
            egui::Panel::right(egui::Id::new("android_inset_right"))
                .exact_size(right)
                .frame(blank)
                .show_separator_line(false)
                .resizable(false)
                .show(ui, |_| {});
        }
    }
}

impl eframe::App for Shell {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        self.reserve_system_insets(ui);
        self.viewer.ui(ui, frame);
        self.storage_prompt(ui.ctx());
    }
}

/// Where the *all files access* question stands.
struct StoragePrompt {
    granted: bool,
    /// *Later* was pressed: not asked again this run. The file browser
    /// still says why a folder cannot be listed.
    dismissed: bool,
    last_check: std::time::Instant,
    error: Option<String>,
}

impl Shell {
    /// Ask for all files access until it is granted or waved away. The
    /// answer is re-read once a second while the window is open, which is
    /// how coming back from the settings page is noticed.
    fn storage_prompt(&mut self, ctx: &egui::Context) {
        if self.storage.granted || self.storage.dismissed {
            return;
        }
        if self.storage.last_check.elapsed() > std::time::Duration::from_secs(1) {
            self.storage.last_check = std::time::Instant::now();
            if let Ok(true) = permission::all_files_access(&self.app) {
                self.storage.granted = true;
                return;
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_secs(1));
        egui::Window::new("Storage access")
            .id(egui::Id::new("android_storage_access"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_max_width(460.0);
                ui.label(
                    "Rust DICOM Station reads DICOM folders from this device's storage and \
                     from USB sticks, and writes its exports there. Android allows that only \
                     to a program with \"All files access\", which is granted once, on a \
                     system settings page.",
                );
                ui.add_space(6.0);
                ui.label(
                    "Without it the program can still open its own folder and the archive \
                     it imports into, but not the rest of the storage.",
                );
                if let Some(e) = &self.storage.error {
                    ui.add_space(6.0);
                    ui.colored_label(ui.visuals().warn_fg_color, e);
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.button("Open the settings page").clicked() {
                        match permission::open_all_files_access_settings(&self.app) {
                            Ok(()) => self.storage.error = None,
                            Err(e) => {
                                self.storage.error = Some(format!(
                                    "The settings page could not be opened ({e}). Open Android \
                                     Settings > Apps > Rust DICOM Station > All files access."
                                ))
                            }
                        }
                    }
                    if ui.button("Later").clicked() {
                        self.storage.dismissed = true;
                    }
                });
            });
    }
}

/// Send everything written to standard output and error to logcat.
///
/// Android connects both to `/dev/null`. The viewer reports a failed
/// backend, a rejected settings file and a good deal else with `eprintln!`,
/// and a panic message goes to standard error before the hook runs - so a
/// pipe replaces the two descriptors and a thread copies it, line by line,
/// into the log.
fn redirect_stdio_to_logcat() {
    let mut fds = [0 as libc::c_int; 2];
    // SAFETY: `fds` is a valid two-element array for `pipe`; the descriptors
    // it returns are owned by this function until moved into the reader.
    let ok = unsafe {
        libc::pipe(fds.as_mut_ptr()) == 0
            && libc::dup2(fds[1], libc::STDOUT_FILENO) >= 0
            && libc::dup2(fds[1], libc::STDERR_FILENO) >= 0
    };
    if !ok {
        log::warn!("standard output could not be redirected to logcat");
        return;
    }
    let read_end = fds[0];
    std::thread::Builder::new()
        .name("stdio-to-logcat".into())
        .spawn(move || {
            use std::io::BufRead as _;
            use std::os::fd::FromRawFd as _;
            // SAFETY: `read_end` is the read side of the pipe created above
            // and is owned by nothing else.
            let file = unsafe { std::fs::File::from_raw_fd(read_end) };
            for line in std::io::BufReader::new(file).lines().map_while(Result::ok) {
                log::info!("{line}");
            }
        })
        .ok();
}

/// Log a panic and keep it in a file, since the log is gone with the
/// process and a tablet rarely has a debug cable attached.
fn install_panic_hook(file: PathBuf) {
    let file = Arc::new(file);
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let text = format!(
            "rust-dicom-station {} panicked: {info}\n{}",
            env!("CARGO_PKG_VERSION"),
            std::backtrace::Backtrace::force_capture()
        );
        log::error!("{text}");
        if let Err(e) = std::fs::write(file.as_path(), &text) {
            log::error!("could not write {}: {e}", file.display());
        }
        previous(info);
    }));
}
