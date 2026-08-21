//! The install itself: copy the payload, leave an uninstaller behind, create
//! shortcuts, write the registry entries, and pull in the two things the
//! viewer needs from outside the payload (the Visual C++ runtime and,
//! optionally, the TotalSegmentator weights).
//!
//! Every filesystem and registry change is recorded in `install-manifest.txt`
//! so the uninstaller can undo exactly what was done and nothing else.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{bail, Context, Result};

use crate::deps;
use crate::payload::Payload;
use crate::plan::*;
use crate::win::registry::{self, Key};
use crate::win::shortcut::{self, Shortcut};

/// Progress and log messages from the worker thread to whatever UI is driving.
pub enum Event {
    /// Overall progress in `0..=1` plus a one-line status.
    Progress(f32, String),
    /// A line for the detail log.
    Log(String),
}

pub type Sink<'a> = &'a (dyn Fn(Event) + Sync);

/// What the uninstaller needs to know, in the same order it should undo it.
#[derive(Default)]
pub struct Manifest {
    pub version: String,
    pub install_dir: PathBuf,
    pub models_dir: PathBuf,
    pub machine_wide: bool,
    pub path_added: bool,
    pub file_association: bool,
    /// Relative to the install directory, `/`-separated.
    pub files: Vec<String>,
    /// Absolute paths of `.lnk` files.
    pub shortcuts: Vec<PathBuf>,
}

impl Manifest {
    pub fn render(&self) -> String {
        let mut out = String::from(
            "# rust-dicom-station install manifest — used by uninstall.exe.\n\
             # Editing this file changes what the uninstaller removes.\n",
        );
        out.push_str(&format!("version = {}\n", self.version));
        out.push_str(&format!("install_dir = {}\n", self.install_dir.display()));
        out.push_str(&format!("models_dir = {}\n", self.models_dir.display()));
        out.push_str(&format!("machine_wide = {}\n", self.machine_wide));
        out.push_str(&format!("path_added = {}\n", self.path_added));
        out.push_str(&format!("file_association = {}\n", self.file_association));
        for s in &self.shortcuts {
            out.push_str(&format!("shortcut = {}\n", s.display()));
        }
        for f in &self.files {
            out.push_str(&format!("file = {f}\n"));
        }
        out
    }

    pub fn parse(text: &str) -> Manifest {
        let mut m = Manifest::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else { continue };
            let (key, value) = (key.trim(), value.trim());
            match key {
                "version" => m.version = value.to_string(),
                "install_dir" => m.install_dir = PathBuf::from(value),
                "models_dir" => m.models_dir = PathBuf::from(value),
                "machine_wide" => m.machine_wide = value == "true",
                "path_added" => m.path_added = value == "true",
                "file_association" => m.file_association = value == "true",
                "shortcut" => m.shortcuts.push(PathBuf::from(value)),
                "file" => m.files.push(value.to_string()),
                _ => {}
            }
        }
        m
    }
}

/// Run the whole installation. Long steps check `cancel` between chunks.
pub fn run(opts: &Options, payload: &Payload, sink: Sink, cancel: &AtomicBool) -> Result<()> {
    let log = |s: String| sink(Event::Log(s));
    let step = |f: f32, s: &str| sink(Event::Progress(f, s.to_string()));

    // ---- preflight -------------------------------------------------------
    step(0.0, "Preparing…");
    std::fs::create_dir_all(&opts.dir)
        .with_context(|| format!("create install directory {}", opts.dir.display()))?;
    check_writable(&opts.dir)?;
    if is_running(&opts.exe_path()) {
        bail!(
            "{} is currently running from {} — close it and start the installer again",
            APP_NAME,
            opts.dir.display()
        );
    }
    log(format!("Installing into {}", opts.dir.display()));

    let mut manifest = Manifest {
        version: payload_version(payload),
        install_dir: opts.dir.clone(),
        models_dir: opts.models_dir.clone(),
        machine_wide: opts.scope == Scope::AllUsers,
        ..Default::default()
    };

    // ---- files -----------------------------------------------------------
    let total_bytes = payload.total_size().unwrap_or(0);
    log(format!("Copying {} of program files", human_size(total_bytes)));
    let mut last = 0.0f32;
    manifest.files = payload.extract_to(&opts.dir, &mut |frac, name| {
        // The copy owns 0.05..0.55 of the bar.
        let f = 0.05 + 0.50 * frac;
        if f - last > 0.005 || frac >= 1.0 {
            last = f;
            sink(Event::Progress(f, format!("Copying {name}")));
        }
    })?;
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }

    // The manifest is saved after every step that creates something, so an
    // installation interrupted half-way can still be uninstalled cleanly.
    let save_manifest = |m: &Manifest| -> Result<()> {
        std::fs::write(opts.manifest_path(), m.render())
            .with_context(|| format!("write {}", opts.manifest_path().display()))
    };
    save_manifest(&manifest)?;

    // ---- uninstaller ------------------------------------------------------
    step(0.57, "Writing the uninstaller…");
    write_uninstaller(payload, &opts.uninstaller_path())?;
    manifest.files.push(UNINSTALLER_EXE.to_string());
    log(format!("Uninstaller: {}", opts.uninstaller_path().display()));

    // ---- settings seed ----------------------------------------------------
    // The viewer stores its settings next to the executable and, by default,
    // the model cache too. Under Program Files that folder is read-only for
    // normal users, so point the cache somewhere writable up front.
    if opts.models_dir != default_models_dir(Scope::CurrentUser, &opts.dir) {
        let settings = opts.dir.join(SETTINGS_FILE);
        if !settings.exists() {
            let text = format!(
                "# rust-dicom-station user settings\n\
                 # theme = dark | light | system\n\
                 theme = dark\n\
                 autoseg_models_dir = {}\n",
                opts.models_dir.display()
            );
            std::fs::write(&settings, text)
                .with_context(|| format!("write {}", settings.display()))?;
            manifest.files.push(SETTINGS_FILE.to_string());
            log(format!("Model cache set to {}", opts.models_dir.display()));
        }
    }
    std::fs::create_dir_all(&opts.models_dir).ok();

    save_manifest(&manifest)?;

    // ---- shortcuts --------------------------------------------------------
    step(0.60, "Creating shortcuts…");
    shortcut::init_com();
    if opts.start_menu_shortcut {
        let link = crate::win::start_menu_programs()?.join(format!("{APP_NAME}.lnk"));
        shortcut::create(&Shortcut {
            link: &link,
            target: &opts.exe_path(),
            args: "",
            working_dir: &opts.dir,
            description: "DICOM / RT DICOM viewer",
        })?;
        log(format!("Start menu: {}", link.display()));
        manifest.shortcuts.push(link);
    }
    if opts.desktop_shortcut {
        let link = crate::win::desktop_dir()?.join(format!("{APP_NAME}.lnk"));
        shortcut::create(&Shortcut {
            link: &link,
            target: &opts.exe_path(),
            args: "",
            working_dir: &opts.dir,
            description: "DICOM / RT DICOM viewer",
        })?;
        log(format!("Desktop: {}", link.display()));
        manifest.shortcuts.push(link);
    }

    save_manifest(&manifest)?;

    // ---- registry ---------------------------------------------------------
    step(0.65, "Registering the application…");
    let hive = opts.scope.hive();
    write_uninstall_entry(opts, hive, total_bytes, &manifest.version)?;
    log(format!("Listed in Apps & features ({})", hive.label()));
    if opts.file_association {
        write_file_association(opts, hive)?;
        manifest.file_association = true;
        log("Registered .dcm files and the folder context-menu entry".into());
    }
    if opts.add_to_path {
        let dir = opts.dir.to_string_lossy().to_string();
        if registry::path_add(hive, &dir)? {
            crate::win::broadcast_environment_change();
            manifest.path_added = true;
            log("Added the program folder to PATH".into());
        }
    }

    save_manifest(&manifest)?;

    // ---- dependencies -----------------------------------------------------
    step(0.70, "Checking dependencies…");
    match deps::vcredist_state() {
        deps::Dependency::Present => log("Visual C++ runtime: already installed".into()),
        deps::Dependency::Missing if opts.install_vcredist => {
            log("Visual C++ runtime: missing — downloading from Microsoft".into());
            deps::install_vcredist(&|f, msg| {
                sink(Event::Progress(0.70 + 0.10 * f, msg.to_string()))
            })?;
            log("Visual C++ runtime: installed".into());
        }
        deps::Dependency::Missing => {
            log("Visual C++ runtime: missing — skipped (the viewer may not start)".into())
        }
    }
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }

    // ---- optional model weights ------------------------------------------
    if opts.models != Models::None {
        step(0.80, "Downloading auto-segmentation models…");
        crate::models::prefetch(
            opts.models,
            &opts.models_dir,
            &|f, msg| sink(Event::Progress(0.80 + 0.19 * f, msg.to_string())),
            cancel,
        )?;
        log("Auto-segmentation models ready".into());
    }

    step(1.0, "Installation complete");
    Ok(())
}

/// The uninstaller is this very binary with the appended payload cut off, so
/// it stays a few megabytes instead of carrying a copy of the program.
fn write_uninstaller(payload: &Payload, dest: &Path) -> Result<()> {
    use std::io::{Read, Write};
    let exe = std::env::current_exe()?;
    let base_len = payload.base_exe_len()?;
    let mut src = std::fs::File::open(&exe)?;
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(dest).with_context(|| format!("write {}", dest.display()))?,
    );
    let mut left = base_len;
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let want = buf.len().min(left as usize);
        let n = src.read(&mut buf[..want])?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        left -= n as u64;
    }
    out.flush()?;
    Ok(())
}

fn write_uninstall_entry(
    opts: &Options,
    hive: registry::Hive,
    size_bytes: u64,
    version: &str,
) -> Result<()> {
    let key = Key::create(hive.hkey(), &uninstall_key_path())?;
    let uninst = opts.uninstaller_path();
    key.set_str("DisplayName", APP_NAME)?;
    key.set_str("DisplayVersion", version)?;
    key.set_str("Publisher", PUBLISHER)?;
    key.set_str("InstallLocation", &opts.dir.to_string_lossy())?;
    key.set_str("DisplayIcon", &opts.exe_path().to_string_lossy())?;
    key.set_str("UninstallString", &format!("\"{}\" --uninstall", uninst.display()))?;
    key.set_str(
        "QuietUninstallString",
        &format!("\"{}\" --uninstall --silent", uninst.display()),
    )?;
    if !HOMEPAGE.is_empty() {
        key.set_str("URLInfoAbout", HOMEPAGE)?;
    }
    key.set_u32("NoModify", 1)?;
    key.set_u32("NoRepair", 1)?;
    key.set_u32("EstimatedSize", (size_bytes / 1024) as u32)?;
    Ok(())
}

/// Register a ProgID for DICOM files and a folder context-menu verb.
///
/// The `.dcm` extension is only *added* to `OpenWithProgids`; whatever program
/// currently owns DICOM files keeps owning them. Opening a folder is the way
/// the viewer is normally used, so that verb is the more useful half.
fn write_file_association(opts: &Options, hive: registry::Hive) -> Result<()> {
    let root = hive.hkey();
    let exe = opts.exe_path();
    let exe_s = exe.to_string_lossy().to_string();
    let classes = r"Software\Classes";

    let progid = Key::create(root, &format!(r"{classes}\{PROGID}"))?;
    progid.set_str("", "DICOM image")?;
    Key::create(root, &format!(r"{classes}\{PROGID}\DefaultIcon"))?
        .set_str("", &format!("{exe_s},0"))?;
    Key::create(root, &format!(r"{classes}\{PROGID}\shell\open\command"))?
        .set_str("", &format!("\"{exe_s}\" \"%1\""))?;

    for ext in [".dcm", ".dicom"] {
        let k = Key::create(root, &format!(r"{classes}\{ext}\OpenWithProgids"))?;
        k.set_str(PROGID, "")?;
    }

    // "Open with Rust DICOM Station" on a folder, and on the empty space inside
    // one — the viewer takes a directory as its argument.
    for base in [
        format!(r"{classes}\Directory\shell\{PRODUCT_ID}"),
        format!(r"{classes}\Directory\Background\shell\{PRODUCT_ID}"),
    ] {
        let k = Key::create(root, &base)?;
        k.set_str("", &format!("Open with {APP_NAME}"))?;
        k.set_str("Icon", &format!("{exe_s},0"))?;
        Key::create(root, &format!(r"{base}\command"))?
            .set_str("", &format!("\"{exe_s}\" \"%V\""))?;
    }
    Ok(())
}

/// Product version: from the payload if `rds-pack` recorded one, else ours.
fn payload_version(payload: &Payload) -> String {
    payload
        .read_text("payload-info.txt")
        .and_then(|t| {
            t.lines()
                .filter_map(|l| l.split_once('='))
                .find(|(k, _)| k.trim() == "version")
                .map(|(_, v)| v.trim().to_string())
        })
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string())
}

/// Writability probe — cheaper and more honest than inspecting ACLs.
fn check_writable(dir: &Path) -> Result<()> {
    let probe = dir.join(".rds-write-test");
    match std::fs::write(&probe, b"x") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        Err(e) => bail!(
            "cannot write to {} ({e}).\nChoose another folder, or re-run the installer as \
             administrator for a machine-wide installation.",
            dir.display()
        ),
    }
}

/// A running executable cannot be overwritten on Windows; detect that early
/// by trying to open the file for writing.
fn is_running(exe: &Path) -> bool {
    if !exe.exists() {
        return false;
    }
    std::fs::OpenOptions::new().write(true).open(exe).is_err()
}
