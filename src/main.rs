//! rust-dicom-station - a fast, robust DICOM / RT DICOM viewer in pure Rust.
//!
//! Usage: `rust-dicom-station [DICOM_DIRECTORY] [COMPARISON_DIRECTORY]`
//!
//! With two directories, comparison mode starts automatically (study A on
//! top, study B below).
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
//!
//! ## One event loop per process
//!
//! `winit` allows one event loop per process, ever: it marks the loop as
//! created *before* it tries to make it, so an attempt that failed while the
//! loop was being made - a missing X11 library, no display - leaves every
//! later attempt in the same process with `RecreationAttempt`, and the
//! message that reached the user was that one instead of the real cause.
//! Two rules follow.
//!
//! * A failure of the **window system** itself (no event loop, or nothing
//!   gets as far as asking for a window) is not a graphics backend's
//!   problem, and no other backend can help: the program stops at once and
//!   says what failed and what to check.
//! * On **Linux** the other backends are tried in a fresh process each
//!   (this executable again, told which backend through
//!   [`ATTEMPT_ENV`]): an X11 event loop whose attempt panicked cannot be
//!   run a second time in the same process, and a new process starts clean.
//!   Windows and macOS keep trying in-process, as they always did - there
//!   `eframe` keeps the event loop it made and hands it to the next attempt.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use rust_dicom_station::{app, gfx, icon, settings};

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set by the window builder hook, which `eframe` calls from inside the
/// running event loop just before it creates the window: once it is set the
/// window system works, and whatever fails afterwards is the graphics
/// backend's.
static WINDOWING_UP: AtomicBool = AtomicBool::new(false);

/// Set when the application itself is created: an error after that is the
/// program's own, not a backend that would not start, and trying another
/// backend would only run it a second time.
static APP_STARTED: AtomicBool = AtomicBool::new(false);

/// The variable that makes this executable one attempt of its parent (Linux):
/// it holds the backend's key, and the process reports through its exit code.
#[cfg(target_os = "linux")]
const ATTEMPT_ENV: &str = "RDS_GFX_ATTEMPT";

/// Exit code of an attempt whose backend would not start (try the next one).
#[cfg(target_os = "linux")]
const EXIT_BACKEND_FAILED: i32 = 3;

/// Exit code of an attempt that found no working window system (stop).
#[cfg(target_os = "linux")]
const EXIT_NO_WINDOWING: i32 = 4;

/// How one attempt at opening the window went.
enum Attempt {
    /// The program ran; this is how it ended.
    Ran(eframe::Result<()>),
    /// The graphics backend would not start; another one may.
    BackendFailed(eframe::Error),
    /// The window system itself did not start; no backend will.
    NoWindowing(eframe::Error),
}

fn main() -> eframe::Result<()> {
    let initial_path: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);
    let initial_path_b: Option<PathBuf> = std::env::args().nth(2).map(PathBuf::from);

    // One attempt of a parent process: run the named backend and report.
    #[cfg(target_os = "linux")]
    if let Some(key) = std::env::var_os(ATTEMPT_ENV) {
        let backend = key
            .to_str()
            .and_then(gfx::Backend::from_key)
            .unwrap_or(gfx::Backend::Auto);
        let code = match run(backend, initial_path, initial_path_b) {
            Attempt::Ran(Ok(())) => 0,
            Attempt::Ran(Err(e)) => {
                eprintln!("rust-dicom-station: {e}");
                1
            }
            Attempt::BackendFailed(e) => {
                eprintln!("rust-dicom-station: {} failed: {e}", backend.label());
                EXIT_BACKEND_FAILED
            }
            Attempt::NoWindowing(e) => {
                eprintln!("rust-dicom-station: {e}");
                EXIT_NO_WINDOWING
            }
        };
        std::process::exit(code);
    }

    // Before anything else, and in particular before any thread exists: the
    // environment is how the inference backend is told which graphics API to
    // use, and writing it is only sound while this process is alone.
    // `WGPU_BACKEND` set by the user wins over the settings file, because
    // someone who set it is working around something.
    let preferred = gfx::from_env().unwrap_or_else(|| settings::load().graphics_backend);
    preferred.export();

    let order = gfx::candidates(preferred);
    let mut first: Option<eframe::Error> = None;
    for (attempt, backend) in order.iter().copied().enumerate() {
        if attempt > 0 {
            eprintln!(
                "rust-dicom-station: {} did not work, trying {}",
                order[attempt - 1].label(),
                backend.label()
            );
        }
        let outcome = if attempt == 0 {
            run(backend, initial_path.clone(), initial_path_b.clone())
        } else {
            retry(backend, initial_path.clone(), initial_path_b.clone())
        };
        match outcome {
            Attempt::Ran(r) => return r,
            Attempt::BackendFailed(e) => {
                eprintln!("rust-dicom-station: {} failed: {e}", backend.label());
                first.get_or_insert(e);
            }
            Attempt::NoWindowing(e) => {
                eprintln!(
                    "rust-dicom-station: the window system could not be started: {e}\n\
                     This is not a graphics driver problem, and no other graphics backend \
                     would get further. Check that a display is available (DISPLAY or \
                     WAYLAND_DISPLAY is set) and that the X11 / Wayland client libraries \
                     are installed (libxkbcommon-x11, libX11, libXcursor, libXi, \
                     libwayland-client)."
                );
                return Err(e);
            }
        }
    }
    eprintln!(
        "rust-dicom-station: no graphics backend on this machine could open a window. \
         Set {}=dx12 (or vulkan, or opengl) to force one, or choose it under \
         View > Graphics backend after the program starts.",
        gfx::ENV_VAR
    );
    Err(first.expect("candidates() is never empty"))
}

/// The next attempt after the first one failed.
///
/// Linux: in a fresh process (see the module notes), which tells how it went
/// through its exit code; its own messages go straight to standard error.
#[cfg(target_os = "linux")]
fn retry(
    backend: gfx::Backend,
    _initial_path: Option<PathBuf>,
    _initial_path_b: Option<PathBuf>,
) -> Attempt {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            return Attempt::BackendFailed(eframe::Error::AppCreation(
                format!("could not find this program to start it again: {e}").into(),
            ))
        }
    };
    let mut child = std::process::Command::new(exe);
    child
        .args(std::env::args_os().skip(1))
        .env(ATTEMPT_ENV, backend.key());
    // The backend reaches `wgpu` inside `burn` through the environment too,
    // so both halves of the program draw and compute on the same one. wgpu
    // has no spelling for "automatic": that is the variable being absent.
    if backend == gfx::Backend::Auto {
        child.env_remove(gfx::ENV_VAR);
    } else {
        child.env(gfx::ENV_VAR, backend.key());
    }
    match child.status() {
        Ok(status) => match status.code() {
            Some(0) => Attempt::Ran(Ok(())),
            Some(EXIT_BACKEND_FAILED) => Attempt::BackendFailed(eframe::Error::AppCreation(
                format!("the {} attempt did not open a window", backend.label()).into(),
            )),
            Some(EXIT_NO_WINDOWING) => Attempt::NoWindowing(eframe::Error::AppCreation(
                format!("the {} attempt found no window system", backend.label()).into(),
            )),
            // The program started and ended with its own error: pass the
            // code on rather than start it again on another backend.
            Some(code) => std::process::exit(code),
            None => std::process::exit(1),
        },
        Err(e) => Attempt::BackendFailed(eframe::Error::AppCreation(
            format!("could not start the {} attempt: {e}", backend.label()).into(),
        )),
    }
}

/// The next attempt after the first one failed: Windows and macOS keep the
/// event loop `eframe` made, so the next backend is tried in this process.
#[cfg(not(target_os = "linux"))]
fn retry(
    backend: gfx::Backend,
    initial_path: Option<PathBuf>,
    initial_path_b: Option<PathBuf>,
) -> Attempt {
    run(backend, initial_path, initial_path_b)
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
) -> Attempt {
    let mut wgpu_options = eframe::WgpuConfiguration::default();
    if let eframe::egui_wgpu::WgpuSetup::CreateNew(setup) = &mut wgpu_options.wgpu_setup {
        setup.instance_descriptor.backends = backend.bits();
    }

    WINDOWING_UP.store(false, Ordering::SeqCst);
    APP_STARTED.store(false, Ordering::SeqCst);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Rust DICOM Station: Viewer")
            .with_icon(icon::window_icon())
            .with_inner_size([1680.0, 940.0])
            .with_min_inner_size([900.0, 520.0]),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        window_builder: Some(Box::new(|builder| {
            WINDOWING_UP.store(true, Ordering::SeqCst);
            builder
        })),
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
                APP_STARTED.store(true, Ordering::SeqCst);
                Ok(Box::new(app::ViewerApp::new(
                    cc,
                    initial_path,
                    initial_path_b,
                )))
            }),
        )
    }));
    let result = match result {
        Ok(r) => r,
        Err(payload) => Err(eframe::Error::AppCreation(
            format!(
                "the {} attempt crashed while starting: {}",
                backend.label(),
                panic_text(payload.as_ref())
            )
            .into(),
        )),
    };
    match result {
        Ok(()) => Attempt::Ran(Ok(())),
        Err(e) if APP_STARTED.load(Ordering::SeqCst) => Attempt::Ran(Err(e)),
        Err(e @ eframe::Error::WinitEventLoop(_)) => Attempt::NoWindowing(e),
        // Nothing got as far as asking for a window. On Linux that is the
        // window system (a library winit loads at run time, the display);
        // elsewhere eframe keeps a loop it managed to make, so only its own
        // error above is taken to mean the loop cannot be had.
        Err(e) if cfg!(target_os = "linux") && !WINDOWING_UP.load(Ordering::SeqCst) => {
            Attempt::NoWindowing(e)
        }
        Err(e) => Attempt::BackendFailed(e),
    }
}

/// The message a panic carried, for the one line that says why an attempt
/// failed (the full report went to standard error already).
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "an unknown error".to_string()
    }
}
