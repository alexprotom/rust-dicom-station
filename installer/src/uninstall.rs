//! Removal of an installation, driven entirely by `install-manifest.txt`:
//! only files, shortcuts and registry keys that were actually created get
//! deleted, so an install into a shared folder cannot take the folder's other
//! contents with it.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::install::{Event, Manifest, Sink};
use crate::plan::*;
use crate::win::registry::{self, Hive, Key};

/// Everything the uninstaller needs to know, resolved from disk.
pub struct Target {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

/// Load the manifest for an installation. `dir` defaults to the folder the
/// running uninstaller sits in.
pub fn discover(dir: Option<PathBuf>) -> Result<Target> {
    let dir = match dir {
        Some(d) => d,
        None => std::env::current_exe()?
            .parent()
            .context("executable has no parent directory")?
            .to_path_buf(),
    };
    let manifest_path = dir.join(MANIFEST_FILE);
    let text = std::fs::read_to_string(&manifest_path).with_context(|| {
        format!(
            "{} not found - this uninstaller must run from the folder it was installed into",
            manifest_path.display()
        )
    })?;
    let mut manifest = Manifest::parse(&text);
    if manifest.install_dir.as_os_str().is_empty() {
        manifest.install_dir = dir.clone();
    }
    Ok(Target { dir, manifest })
}

/// True when the running uninstaller lives inside the folder it must delete.
pub fn running_from_target(target: &Target) -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
        .map(|p| paths_equal(&p, &target.dir))
        .unwrap_or(false)
}

/// Copy ourselves to `%TEMP%` and re-run from there, so the install folder can
/// be deleted completely. Returns the spawned child.
pub fn relaunch_from_temp(target: &Target, extra_args: &[&str]) -> Result<std::process::Child> {
    let exe = std::env::current_exe()?;
    let tmp = std::env::temp_dir().join(format!("{PRODUCT_ID}-uninstall.exe"));
    // A leftover from an earlier run may still be scheduled for deletion.
    let _ = std::fs::remove_file(&tmp);
    std::fs::copy(&exe, &tmp).with_context(|| format!("copy uninstaller to {}", tmp.display()))?;
    let mut cmd = std::process::Command::new(&tmp);
    cmd.arg("--uninstall")
        .arg("--from")
        .arg(&target.dir)
        .args(extra_args);
    cmd.spawn()
        .with_context(|| format!("start {}", tmp.display()))
}

/// Remove the installation. `remove_models` also deletes the model root —
/// every engine's downloads, not only what the installer fetched.
pub fn run(target: &Target, remove_models: bool, sink: Sink) -> Result<()> {
    let log = |s: String| sink(Event::Log(s));
    let step = |f: f32, s: &str| sink(Event::Progress(f, s.to_string()));
    let m = &target.manifest;
    let dir = &m.install_dir;

    let app_exe = dir.join(APP_EXE);
    if app_exe.exists()
        && std::fs::OpenOptions::new()
            .write(true)
            .open(&app_exe)
            .is_err()
    {
        bail!("{APP_NAME} is still running - close it and try again");
    }

    // ---- registry ---------------------------------------------------------
    step(0.05, "Removing registry entries");
    let hive = if m.machine_wide {
        Hive::LocalMachine
    } else {
        Hive::CurrentUser
    };
    registry::delete_tree(hive.hkey(), &uninstall_key_path())?;
    if m.file_association {
        let classes = r"Software\Classes";
        registry::delete_tree(hive.hkey(), &format!(r"{classes}\{PROGID}"))?;
        registry::delete_tree(
            hive.hkey(),
            &format!(r"{classes}\Directory\shell\{PRODUCT_ID}"),
        )?;
        registry::delete_tree(
            hive.hkey(),
            &format!(r"{classes}\Directory\Background\shell\{PRODUCT_ID}"),
        )?;
        for ext in [".dcm", ".dicom"] {
            if let Ok(k) = Key::open(
                hive.hkey(),
                &format!(r"{classes}\{ext}\OpenWithProgids"),
                true,
            ) {
                let _ = k.delete_value(PROGID);
            }
        }
        log("Removed the file association".into());
    }
    if m.path_added {
        let dir_s = dir.to_string_lossy().to_string();
        if registry::path_remove(hive, &dir_s)? {
            crate::win::broadcast_environment_change();
            log("Removed the program folder from PATH".into());
        }
    }

    // ---- shortcuts --------------------------------------------------------
    step(0.15, "Removing shortcuts");
    for link in &m.shortcuts {
        if link.exists() {
            match std::fs::remove_file(link) {
                Ok(()) => log(format!("Removed {}", link.display())),
                Err(e) => log(format!("Could not remove {}: {e}", link.display())),
            }
        }
    }

    // ---- files ------------------------------------------------------------
    step(0.25, "Removing program files");
    let total = m.files.len().max(1) as f32;
    let mut dirs: Vec<PathBuf> = Vec::new();
    for (i, rel) in m.files.iter().enumerate() {
        let path = dir.join(rel.replace('/', "\\"));
        if rel == UNINSTALLER_EXE {
            continue; // handled last, it may be the running image
        }
        if path.is_file() {
            if let Err(e) = remove_with_retry(&path) {
                log(format!("Could not remove {}: {e}", path.display()));
            }
        }
        let mut p = path.parent().map(Path::to_path_buf);
        while let Some(d) = p {
            if paths_equal(&d, dir) || !d.starts_with(dir) {
                break;
            }
            if !dirs.contains(&d) {
                dirs.push(d.clone());
            }
            p = d.parent().map(Path::to_path_buf);
        }
        step(0.25 + 0.45 * (i as f32 / total), "Removing program files");
    }
    // Deepest first, and only when empty.
    dirs.sort_by_key(|d| std::cmp::Reverse(d.components().count()));
    for d in dirs {
        let _ = std::fs::remove_dir(&d);
    }

    // ---- model folder -----------------------------------------------------
    if remove_models && !m.models_dir.as_os_str().is_empty() && m.models_dir.exists() {
        step(0.75, "Removing the model folder");
        match std::fs::remove_dir_all(&m.models_dir) {
            Ok(()) => log(format!("Removed {}", m.models_dir.display())),
            Err(e) => log(format!("Could not remove {}: {e}", m.models_dir.display())),
        }
        prune_empty_parents(&m.models_dir);
    } else if m.models_dir.exists() {
        // `remove_dir` only succeeds on an empty directory, so a cache that
        // was never filled disappears while a real one is kept.
        if std::fs::remove_dir(&m.models_dir).is_ok() {
            prune_empty_parents(&m.models_dir);
        } else {
            log(format!(
                "Kept the model folder in {}",
                m.models_dir.display()
            ));
        }
    }

    // ---- the last few files -----------------------------------------------
    step(0.9, "Cleaning up");
    let _ = remove_with_retry(&dir.join(MANIFEST_FILE));
    let _ = remove_with_retry(&dir.join(UNINSTALLER_EXE));
    // Only removes the folder when nothing unexpected is left in it.
    match std::fs::remove_dir(dir) {
        Ok(()) => log(format!("Removed {}", dir.display())),
        Err(_) => log(format!(
            "Left {} in place - it still contains files that were not installed by the setup",
            dir.display()
        )),
    }

    step(1.0, "Uninstall complete");
    Ok(())
}

/// Windows keeps files locked briefly after the owning process exits; the
/// uninstaller races its own parent, so retry for a few seconds.
fn remove_with_retry(path: &Path) -> std::io::Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut last = Ok(());
    for attempt in 0..25 {
        match std::fs::remove_file(path) {
            Ok(()) => return Ok(()),
            Err(e) => {
                last = Err(e);
                std::thread::sleep(std::time::Duration::from_millis(if attempt < 5 {
                    50
                } else {
                    200
                }));
            }
        }
        if !path.exists() {
            return Ok(());
        }
    }
    last
}

/// Delete the now-empty folders the installer itself created around a path —
/// never anything the user might have named.
fn prune_empty_parents(start: &Path) {
    let mut parent = start.parent();
    while let Some(d) = parent {
        let name = d.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name != PRODUCT_ID && name != APP_NAME {
            break;
        }
        if std::fs::remove_dir(d).is_err() {
            break;
        }
        parent = d.parent();
    }
}

fn paths_equal(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| {
        p.to_string_lossy()
            .trim_end_matches(['\\', '/'])
            .to_ascii_lowercase()
            .replace('/', "\\")
    };
    norm(a) == norm(b)
}
