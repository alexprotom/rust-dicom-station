//! The wizard. Same UI stack as the viewer itself (egui/eframe on wgpu), so
//! the installer looks like the program it installs and adds no new
//! technology to the project.
//!
//! Every screen is one function; the long-running work happens on a worker
//! thread that reports through a shared progress slot.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use egui::{Color32, RichText};

use crate::install::{self, Event};
use crate::models;
use crate::payload::Payload;
use crate::plan::*;
use crate::uninstall::{self, Target};

#[derive(PartialEq, Eq, Clone, Copy)]
enum Screen {
    Welcome,
    Options,
    Graphics,
    Working,
    Done,
    ConfirmUninstall,
}

enum Job {
    Install { payload: Arc<Payload> },
    Uninstall { target: Arc<Target> },
}

pub struct SetupApp {
    job: Job,
    screen: Screen,
    opts: Options,
    version: String,
    license: Option<String>,
    accepted: bool,
    dir_text: String,
    models_dir_text: String,
    remove_models: bool,
    payload_size: u64,
    progress: Arc<Mutex<(f32, String)>>,
    log: Arc<Mutex<Vec<String>>>,
    outcome: Arc<Mutex<Option<Result<(), String>>>>,
    cancel: Arc<AtomicBool>,
    error: Option<String>,
    /// Set when the wizard's own window had to fall back to another graphics
    /// backend: the same machine will fail the same way for the viewer, so
    /// the graphics page says so and preselects what actually worked.
    graphics_note: Option<String>,
    /// Begin immediately instead of showing the first page (used by the
    /// elevated re-launch, which inherits the options on the command line).
    autostart: bool,
    /// Set once the user asks to close after a finished run.
    quit: bool,
}

/// Show the install wizard. Returns `Err` when no window could be created
/// (head-less session, no usable GPU adapter) so the caller can fall back to
/// the text interface.
pub fn run_install(payload: Payload, opts: Options, autostart: bool) -> Result<()> {
    let license = payload.read_text("LICENSE.txt");
    let version = payload
        .read_text("payload-info.txt")
        .and_then(|t| {
            t.lines()
                .filter_map(|l| l.split_once('='))
                .find(|(k, _)| k.trim() == "version")
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
    let payload_size = payload.total_size().unwrap_or(0);
    let mut app = SetupApp::new(
        Job::Install {
            payload: Arc::new(payload),
        },
        Screen::Welcome,
        opts,
        version,
        license,
        payload_size,
    );
    // After an elevation re-launch the user has already made every choice in
    // the first window; go straight to work.
    app.autostart = autostart;
    launch(app, &format!("{APP_NAME}: Setup"))
}

/// Show the uninstall confirmation window.
pub fn run_uninstall(target: Target) -> Result<()> {
    let opts = Options {
        dir: target.manifest.install_dir.clone(),
        models_dir: target.manifest.models_dir.clone(),
        ..Default::default()
    };
    let version = target.manifest.version.clone();
    let app = SetupApp::new(
        Job::Uninstall {
            target: Arc::new(target),
        },
        Screen::ConfirmUninstall,
        opts,
        version,
        None,
        0,
    );
    launch(app, &format!("{APP_NAME}: Uninstall"))
}

/// Open the wizard window.
///
/// The installer draws with the same library as the viewer, on the same
/// machine - so the broken Vulkan driver that stops the viewer stops the
/// setup program too, and the setup program is where the page that fixes it
/// lives. The window is therefore *attempted* rather than opened: first
/// whatever `wgpu` would pick on its own (which already honours
/// `WGPU_BACKEND`), then Direct3D 12, then Vulkan. A backend that fails -
/// by error or by panicking inside the driver - costs a line on standard
/// error instead of the installation.
///
/// Falling back is also an answer: if this window only appeared on Direct3D
/// 12, the viewer will need Direct3D 12 as well, so the graphics page says
/// so and preselects it.
fn launch(app: SetupApp, title: &str) -> Result<()> {
    // The same picture as the viewer's window icon and as both executables'
    // Windows icon resource (see the two `build.rs`), so the setup program
    // is recognisably the same product before anything is installed.
    let icon = Arc::new(
        eframe::icon_data::from_png_bytes(include_bytes!("../../assets/rust-dicom-station.png"))
            .unwrap_or_default(),
    );
    // `run_native` consumes the app, so it waits here and the creation
    // closure takes it out. An attempt that dies before creating anything
    // never gets that far, and the next one finds it still in place.
    let slot = Arc::new(Mutex::new(Some(app)));
    let order = attempts();
    let mut last = "no graphics backend could open a window".to_string();
    for (i, attempt) in order.iter().enumerate() {
        if i > 0 {
            eprintln!(
                "{APP_NAME} setup: {} did not work, trying {}",
                order[i - 1].label,
                attempt.label
            );
            if let Some(app) = slot.lock().unwrap().as_mut() {
                // Not after an elevated re-launch: there the choice was
                // already made in the first window and passed on the command
                // line, and the page is skipped anyway.
                if app.is_install() && !app.autostart {
                    if let Some(choice) = attempt.choice {
                        app.opts.graphics = choice;
                        app.graphics_note = Some(format!(
                            "This window could not open with {}, so {} is preselected below.",
                            order[i - 1].label,
                            choice.label()
                        ));
                    }
                }
            }
        }
        match attempt.open(&slot, title, &icon) {
            Ok(()) => return Ok(()),
            Err(e) => {
                eprintln!("{APP_NAME} setup: {} failed: {e}", attempt.label);
                last = e;
            }
        }
    }
    Err(anyhow::anyhow!(
        "could not open the installer window: {last}"
    ))
}

/// One backend to try for the wizard's own window.
struct Attempt {
    label: &'static str,
    /// `None` leaves `wgpu`'s own default in place, which is what an
    /// unremarkable machine - and anyone who set `WGPU_BACKEND` - should get.
    bits: Option<eframe::wgpu::Backends>,
    /// What to preselect on the graphics page when this is the attempt that
    /// works. `None` for the first try, which says nothing about the machine.
    choice: Option<Graphics>,
}

fn attempts() -> Vec<Attempt> {
    use eframe::wgpu::Backends;
    vec![
        Attempt {
            label: "the default backend",
            bits: None,
            choice: None,
        },
        Attempt {
            label: "DirectX 12",
            bits: Some(Backends::DX12),
            choice: Some(Graphics::Dx12),
        },
        Attempt {
            label: "Vulkan",
            bits: Some(Backends::VULKAN),
            choice: Some(Graphics::Vulkan),
        },
    ]
}

impl Attempt {
    /// Try to run the wizard to completion on this backend. `Err` carries a
    /// message for the next attempt to report; the app is left in `slot`
    /// unless a window was actually created.
    fn open(
        &self,
        slot: &Arc<Mutex<Option<SetupApp>>>,
        title: &str,
        icon: &Arc<egui::IconData>,
    ) -> std::result::Result<(), String> {
        let mut wgpu_options = eframe::WgpuConfiguration::default();
        if let Some(bits) = self.bits {
            if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
                setup.instance_descriptor.backends = bits;
            }
        }
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title(title)
                .with_icon(Arc::clone(icon))
                .with_inner_size([760.0, 560.0])
                .with_min_inner_size([640.0, 460.0]),
            renderer: eframe::Renderer::Wgpu,
            wgpu_options,
            ..Default::default()
        };
        let slot = slot.clone();
        // `NativeOptions` holds boxed callbacks that are not `UnwindSafe`,
        // which is a fair warning in general and irrelevant here: nothing is
        // read back after a failed attempt - the next one builds its own.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            eframe::run_native(
                title,
                options,
                Box::new(move |cc| {
                    cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
                    let app = slot
                        .lock()
                        .unwrap()
                        .take()
                        .ok_or("the installer window was created twice")?;
                    Ok(Box::new(app) as Box<dyn eframe::App>)
                }),
            )
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(format!("{e}")),
            Err(_) => Err(format!("the {} driver crashed while starting", self.label)),
        }
    }
}

impl SetupApp {
    fn new(
        job: Job,
        screen: Screen,
        opts: Options,
        version: String,
        license: Option<String>,
        payload_size: u64,
    ) -> SetupApp {
        SetupApp {
            dir_text: opts.dir.to_string_lossy().to_string(),
            models_dir_text: opts.models_dir.to_string_lossy().to_string(),
            job,
            screen,
            opts,
            version,
            license,
            accepted: false,
            remove_models: false,
            payload_size,
            progress: Arc::new(Mutex::new((0.0, String::new()))),
            log: Arc::new(Mutex::new(Vec::new())),
            outcome: Arc::new(Mutex::new(None)),
            cancel: Arc::new(AtomicBool::new(false)),
            error: None,
            graphics_note: None,
            autostart: false,
            quit: false,
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        self.screen = Screen::Working;
        self.cancel.store(false, Ordering::Relaxed);
        self.log.lock().unwrap().clear();
        *self.outcome.lock().unwrap() = None;
        let progress = self.progress.clone();
        let log = self.log.clone();
        let outcome = self.outcome.clone();
        let cancel = self.cancel.clone();
        let ctx = ctx.clone();
        let opts = self.opts.clone();
        let remove_models = self.remove_models;
        let job = match &self.job {
            Job::Install { payload } => Job::Install {
                payload: payload.clone(),
            },
            Job::Uninstall { target } => Job::Uninstall {
                target: target.clone(),
            },
        };
        std::thread::spawn(move || {
            let sink = move |ev: Event| {
                match ev {
                    Event::Progress(f, msg) => *progress.lock().unwrap() = (f, msg),
                    Event::Log(line) => log.lock().unwrap().push(line),
                }
                ctx.request_repaint();
            };
            let res = match &job {
                Job::Install { payload } => install::run(&opts, payload, &sink, &cancel),
                Job::Uninstall { target } => uninstall::run(target, remove_models, &sink),
            };
            *outcome.lock().unwrap() = Some(res.map_err(|e| format!("{e:#}")));
        });
    }

    fn is_install(&self) -> bool {
        matches!(self.job, Job::Install { .. })
    }
}

impl eframe::App for SetupApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.autostart {
            self.autostart = false;
            self.start(&ctx);
        }
        if self.screen == Screen::Working {
            let done = self.outcome.lock().unwrap().take();
            if let Some(res) = done {
                self.error = res.err();
                self.screen = Screen::Done;
            }
            // The worker only repaints on new messages; keep the bar alive.
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        let title = if self.is_install() {
            format!("{APP_NAME} {}", self.version)
        } else {
            format!("Uninstall {APP_NAME}")
        };
        egui::Panel::top(egui::Id::new("head")).show(ui, |ui| {
            ui.add_space(8.0);
            ui.heading(title);
            ui.label(
                RichText::new(
                    "Open-source software designed for the medical imaging, radiotherapy \
                     research, analysis and QA",
                )
                .weak(),
            );
            ui.add_space(8.0);
        });
        let screen = self.screen;
        egui::CentralPanel::default_margins().show(ui, |ui| match screen {
            Screen::Welcome => self.welcome(ui),
            Screen::Options => self.options(ui),
            Screen::Graphics => self.graphics(ui, &ctx),
            Screen::Working => self.working(ui),
            Screen::Done => self.done(ui),
            Screen::ConfirmUninstall => self.confirm_uninstall(ui, &ctx),
        });
        if self.quit {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

impl SetupApp {
    fn welcome(&mut self, ui: &mut egui::Ui) {
        ui.label(
            "This will install the viewer, its Start-menu entry and, optionally, the \
                  auto-segmentation weights.",
        );
        ui.add_space(6.0);
        if let Some(text) = &self.license {
            ui.label(RichText::new("License").strong());
            egui::ScrollArea::vertical()
                .max_height(280.0)
                .show(ui, |ui| {
                    ui.monospace(text);
                });
            ui.add_space(6.0);
            ui.checkbox(&mut self.accepted, "I accept the license terms");
        } else {
            self.accepted = true;
            ui.label("MIT-licensed software. Not a medical device - for research and QA only.");
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.accepted, egui::Button::new("Next  ▶"))
                .clicked()
            {
                self.screen = Screen::Options;
            }
            if ui.button("Cancel").clicked() {
                self.quit = true;
            }
        });
    }

    fn options(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(RichText::new("Install for").strong());
            let mut scope = self.opts.scope;
            for choice in [Scope::CurrentUser, Scope::AllUsers] {
                ui.radio_value(&mut scope, choice, choice.label());
            }
            if scope != self.opts.scope {
                self.opts.set_scope(scope);
                self.dir_text = self.opts.dir.to_string_lossy().to_string();
                self.models_dir_text = self.opts.models_dir.to_string_lossy().to_string();
            }
            if self.opts.scope == Scope::AllUsers && !crate::win::is_elevated() {
                ui.colored_label(
                    Color32::from_rgb(230, 170, 60),
                    "Administrator rights will be requested when the installation starts.",
                );
            }

            ui.add_space(10.0);
            ui.label(RichText::new("Destination folder").strong());
            ui.horizontal(|ui| {
                let w = ui.available_width() - 90.0;
                if ui
                    .add_sized(
                        [w.max(200.0), 22.0],
                        egui::TextEdit::singleline(&mut self.dir_text),
                    )
                    .changed()
                {
                    self.opts.set_dir(PathBuf::from(self.dir_text.trim()));
                    self.models_dir_text = self.opts.models_dir.to_string_lossy().to_string();
                }
                if ui.button("Browse").clicked() {
                    if let Some(d) = rfd::FileDialog::new()
                        .set_title("Choose the installation folder")
                        .pick_folder()
                    {
                        self.opts.set_dir(d.join(APP_NAME));
                        self.dir_text = self.opts.dir.to_string_lossy().to_string();
                        self.models_dir_text = self.opts.models_dir.to_string_lossy().to_string();
                    }
                }
            });
            ui.label(
                RichText::new(format!(
                    "About {} of program files",
                    human_size(self.payload_size)
                ))
                .weak(),
            );

            ui.add_space(10.0);
            ui.label(RichText::new("Integration").strong());
            ui.checkbox(&mut self.opts.start_menu_shortcut, "Start menu shortcut");
            ui.checkbox(&mut self.opts.desktop_shortcut, "Desktop shortcut");
            ui.checkbox(
                &mut self.opts.file_association,
                "Add \"Open with Rust DICOM Station\" to folders and DICOM files",
            );
            ui.checkbox(
                &mut self.opts.add_to_path,
                "Add the program folder to PATH (command-line use)",
            );

            ui.add_space(10.0);
            ui.label(RichText::new("Dependencies").strong());
            match crate::deps::vcredist_state() {
                crate::deps::Dependency::Present => {
                    ui.label(RichText::new("Visual C++ runtime: present ✔").weak());
                }
                crate::deps::Dependency::Missing => {
                    ui.checkbox(
                        &mut self.opts.install_vcredist,
                        "Download and install the Microsoft Visual C++ runtime (required)",
                    );
                }
            }
            ui.label(
                RichText::new(
                    "Graphics: drawing uses Direct3D 12 or Vulkan through the display driver \
                     already on this machine - nothing to install. The next page chooses \
                     which one.",
                )
                .weak(),
            );

            ui.add_space(10.0);
            ui.label(RichText::new("Model folder and auto-segmentation weights").strong());
            if models::AVAILABLE {
                egui::ComboBox::from_id_salt("models")
                    .width(360.0)
                    .selected_text(self.opts.models.label())
                    .show_ui(ui, |ui| {
                        for m in Models::ALL {
                            ui.selectable_value(&mut self.opts.models, m, m.label());
                        }
                    });
                ui.horizontal(|ui| {
                    ui.label("Model folder:");
                    let w = ui.available_width() - 10.0;
                    if ui
                        .add_sized(
                            [w.max(200.0), 22.0],
                            egui::TextEdit::singleline(&mut self.models_dir_text),
                        )
                        .changed()
                    {
                        self.opts.models_dir = PathBuf::from(self.models_dir_text.trim());
                    }
                });
                if self.opts.models != Models::None {
                    let bytes = models::download_size(self.opts.models, &self.opts.models_dir);
                    ui.label(
                        RichText::new(format!(
                            "{} still to download from the TotalSegmentator release - this can \
                             take a while.",
                            human_size(bytes)
                        ))
                        .weak(),
                    );
                }
            } else {
                ui.label(
                    RichText::new(
                        "This build cannot pre-fetch weights; the viewer downloads them on \
                         first use.",
                    )
                    .weak(),
                );
            }

            ui.add_space(14.0);
            ui.horizontal(|ui| {
                if ui.button("◀  Back").clicked() {
                    self.screen = Screen::Welcome;
                }
                if ui.button("Next  ▶").clicked() {
                    self.opts.set_dir(PathBuf::from(self.dir_text.trim()));
                    self.opts.models_dir = PathBuf::from(self.models_dir_text.trim());
                    self.screen = Screen::Graphics;
                }
                if ui.button("Cancel").clicked() {
                    self.quit = true;
                }
            });
            if let Some(err) = &self.error {
                ui.add_space(6.0);
                ui.colored_label(Color32::from_rgb(230, 100, 100), err);
            }
        });
    }

    /// The page this whole change exists for: which graphics API the viewer
    /// should start on.
    ///
    /// Vulkan is preselected because it is right on the overwhelming majority
    /// of machines. The page exists for the rest - and when the installer's
    /// own window had to fall back to get here, the answer is already known
    /// and filled in.
    fn graphics(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(RichText::new("Graphics backend").strong());
            ui.label(
                "The viewer draws through your display driver. Almost every machine should \
                 leave this on Vulkan; a few Windows systems advertise a Vulkan driver that \
                 cannot actually start, and on those the viewer must use DirectX 12 instead.",
            );
            ui.add_space(4.0);
            ui.label(
                RichText::new(
                    "This is only the starting choice. It can be changed at any time under \
                     Settings > Graphics backend, and the viewer falls back on its own if \
                     the one chosen here does not work.",
                )
                .weak(),
            );
            if let Some(note) = &self.graphics_note {
                ui.add_space(8.0);
                ui.colored_label(Color32::from_rgb(230, 170, 60), note);
            }

            ui.add_space(10.0);
            for choice in Graphics::ALL {
                ui.radio_value(&mut self.opts.graphics, choice, choice.label());
                ui.indent(choice.key(), |ui| {
                    ui.label(RichText::new(choice.hint()).weak());
                });
                ui.add_space(4.0);
            }

            ui.add_space(14.0);
            ui.horizontal(|ui| {
                if ui.button("◀  Back").clicked() {
                    self.screen = Screen::Options;
                }
                if ui.button("Install").clicked() {
                    if self.opts.scope == Scope::AllUsers && !crate::win::is_elevated() {
                        match crate::win::relaunch_elevated(&crate::args_for_relaunch(&self.opts)) {
                            Ok(()) => self.quit = true,
                            Err(e) => self.error = Some(format!("{e:#}")),
                        }
                    } else {
                        self.start(ctx);
                    }
                }
                if ui.button("Cancel").clicked() {
                    self.quit = true;
                }
            });
            if let Some(err) = &self.error {
                ui.add_space(6.0);
                ui.colored_label(Color32::from_rgb(230, 100, 100), err);
            }
        });
    }

    fn working(&mut self, ui: &mut egui::Ui) {
        let (frac, msg) = self.progress.lock().unwrap().clone();
        ui.add(egui::ProgressBar::new(frac).show_percentage().animate(true));
        ui.add_space(4.0);
        ui.label(msg);
        ui.add_space(10.0);
        log_view(ui, &self.log);
        ui.add_space(8.0);
        if self.is_install() && ui.button("Cancel").clicked() {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }

    fn done(&mut self, ui: &mut egui::Ui) {
        match &self.error {
            None if self.is_install() => {
                ui.colored_label(
                    Color32::from_rgb(120, 200, 120),
                    RichText::new("Installation complete").heading(),
                );
                ui.label(format!("Installed into {}", self.opts.dir.display()));
            }
            None => {
                ui.colored_label(
                    Color32::from_rgb(120, 200, 120),
                    RichText::new("Uninstall complete").heading(),
                );
            }
            Some(err) => {
                ui.colored_label(
                    Color32::from_rgb(230, 100, 100),
                    RichText::new(if self.is_install() {
                        "Installation failed"
                    } else {
                        "Uninstall failed"
                    })
                    .heading(),
                );
                ui.add_space(4.0);
                ui.label(err.clone());
            }
        }
        ui.add_space(8.0);
        log_view(ui, &self.log);
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            if self.is_install() && self.error.is_none() {
                ui.checkbox(&mut self.opts.launch_after, format!("Start {APP_NAME} now"));
            }
            if ui.button("Close").clicked() {
                if self.is_install() && self.error.is_none() && self.opts.launch_after {
                    let _ = crate::win::shell_execute(&self.opts.exe_path(), "", false);
                }
                self.quit = true;
            }
        });
    }

    fn confirm_uninstall(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Job::Uninstall { target } = &self.job else {
            return;
        };
        ui.label(format!(
            "This removes {APP_NAME} from {}.",
            target.manifest.install_dir.display()
        ));
        ui.add_space(6.0);
        let models_dir = target.manifest.models_dir.clone();
        if models_dir.exists() {
            let size = dir_size(&models_dir);
            ui.checkbox(
                &mut self.remove_models,
                format!(
                    "Also delete every downloaded model in {} ({})",
                    models_dir.display(),
                    human_size(size)
                ),
            );
        }
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if ui.button("Uninstall").clicked() {
                self.start(ctx);
            }
            if ui.button("Cancel").clicked() {
                self.quit = true;
            }
        });
    }
}

fn log_view(ui: &mut egui::Ui, log: &Arc<Mutex<Vec<String>>>) {
    egui::ScrollArea::vertical()
        .stick_to_bottom(true)
        .max_height(260.0)
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for line in log.lock().unwrap().iter() {
                ui.label(RichText::new(line).monospace().weak());
            }
        });
}

fn dir_size(dir: &std::path::Path) -> u64 {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}
