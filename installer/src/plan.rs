//! Product constants, install options and the defaults the UI starts from.

use std::path::{Path, PathBuf};

use crate::win::registry::Hive;

pub const APP_NAME: &str = "Rust DICOM Viewer";
pub const APP_EXE: &str = "rust-dicom-viewer.exe";
pub const PUBLISHER: &str = "rust-dicom-viewer contributors";
/// Shown in Apps & features. Empty means "do not write the value".
pub const HOMEPAGE: &str = "";
/// Registry-safe product id, used for the Add/Remove Programs key.
pub const PRODUCT_ID: &str = "RustDicomViewer";
/// ProgID for the optional `.dcm` file association.
pub const PROGID: &str = "RustDicomViewer.DicomFile";
pub const UNINSTALLER_EXE: &str = "uninstall.exe";
pub const MANIFEST_FILE: &str = "install-manifest.txt";
/// Written next to the executable by the viewer itself; the installer only
/// pre-seeds it when the program folder is not user-writable.
pub const SETTINGS_FILE: &str = "viewer_settings.txt";
/// Official Microsoft download for the x64 Visual C++ 2015-2022 runtime.
pub const VCREDIST_URL: &str = "https://aka.ms/vs/17/release/vc_redist.x64.exe";

/// Who the installation is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    /// `%LOCALAPPDATA%\Programs\…` — no administrator rights needed.
    CurrentUser,
    /// `C:\Program Files\…` — needs elevation.
    AllUsers,
}

impl Scope {
    pub fn hive(self) -> Hive {
        match self {
            Scope::CurrentUser => Hive::CurrentUser,
            Scope::AllUsers => Hive::LocalMachine,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Scope::CurrentUser => "Just me (no administrator rights needed)",
            Scope::AllUsers => "All users (requires administrator rights)",
        }
    }
}

/// Which auto-segmentation weights to fetch during installation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Models {
    /// Leave the cache empty; the viewer downloads on first use.
    None,
    /// 6 mm preview model — ~135 MB.
    Preview6mm,
    /// 3 mm model, all 117 classes — ~135 MB.
    Fast3mm,
    /// The five 1.5 mm sub-models — ~1.2 GB.
    HighRes15mm,
    /// 3 mm plus the full 1.5 mm set — ~1.3 GB.
    Everything,
}

impl Models {
    pub fn label(self) -> &'static str {
        match self {
            Models::None => "Download later, on first use",
            Models::Preview6mm => "6 mm preview model (~135 MB)",
            Models::Fast3mm => "3 mm model, all 117 structures (~135 MB)",
            Models::HighRes15mm => "1.5 mm high-quality models (~1.2 GB)",
            Models::Everything => "3 mm + 1.5 mm — everything (~1.3 GB)",
        }
    }

    pub const ALL: [Models; 5] = [
        Models::None,
        Models::Fast3mm,
        Models::Preview6mm,
        Models::HighRes15mm,
        Models::Everything,
    ];
}

/// Everything the user can decide before the copy starts.
#[derive(Clone, Debug)]
pub struct Options {
    pub scope: Scope,
    pub dir: PathBuf,
    /// Model cache location; see [`default_models_dir`].
    pub models_dir: PathBuf,
    pub models: Models,
    pub start_menu_shortcut: bool,
    pub desktop_shortcut: bool,
    pub add_to_path: bool,
    /// Register `.dcm`/`.dicom` and an "Open with …" entry on folders.
    pub file_association: bool,
    /// Install the Microsoft Visual C++ runtime when it is missing.
    pub install_vcredist: bool,
    pub launch_after: bool,
}

impl Default for Options {
    fn default() -> Self {
        let scope = Scope::CurrentUser;
        let dir = default_install_dir(scope);
        Options {
            models_dir: default_models_dir(scope, &dir),
            scope,
            dir,
            models: Models::None,
            start_menu_shortcut: true,
            desktop_shortcut: true,
            add_to_path: false,
            file_association: true,
            install_vcredist: true,
            launch_after: true,
        }
    }
}

impl Options {
    pub fn exe_path(&self) -> PathBuf {
        self.dir.join(APP_EXE)
    }

    pub fn uninstaller_path(&self) -> PathBuf {
        self.dir.join(UNINSTALLER_EXE)
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.dir.join(MANIFEST_FILE)
    }

    /// Change the destination folder, keeping a still-default model cache
    /// pointed next to the new location.
    pub fn set_dir(&mut self, dir: PathBuf) {
        if self.models_dir == default_models_dir(self.scope, &self.dir) {
            self.models_dir = default_models_dir(self.scope, &dir);
        }
        self.dir = dir;
    }

    /// Switching scope moves the default directories along with it, unless the
    /// user has already typed a path of their own.
    pub fn set_scope(&mut self, scope: Scope) {
        let dir_was_default = self.dir == default_install_dir(self.scope);
        let models_were_default = self.models_dir == default_models_dir(self.scope, &self.dir);
        self.scope = scope;
        if dir_was_default {
            self.dir = default_install_dir(scope);
        }
        if models_were_default {
            self.models_dir = default_models_dir(scope, &self.dir);
        }
    }
}

/// `%LOCALAPPDATA%\Programs\Rust DICOM Viewer` or `%ProgramFiles%\Rust DICOM
/// Viewer`, with a plain-C: fallback should the shell folder lookup fail.
pub fn default_install_dir(scope: Scope) -> PathBuf {
    match scope {
        Scope::CurrentUser => crate::win::local_app_data()
            .unwrap_or_else(|_| PathBuf::from(r"C:\Users\Public"))
            .join("Programs")
            .join(APP_NAME),
        Scope::AllUsers => crate::win::program_files()
            .unwrap_or_else(|_| PathBuf::from(r"C:\Program Files"))
            .join(APP_NAME),
    }
}

/// Where the TotalSegmentator weight cache goes.
///
/// The viewer defaults to `autoseg_models/` next to its executable, which is
/// exactly right for a per-user install. A machine-wide install lands in
/// `Program Files`, which normal users cannot write to, so the cache moves to
/// `%LOCALAPPDATA%` and the installer records that in `viewer_settings.txt`.
pub fn default_models_dir(scope: Scope, install_dir: &Path) -> PathBuf {
    match scope {
        Scope::CurrentUser => install_dir.join("autoseg_models"),
        Scope::AllUsers => crate::win::local_app_data()
            .unwrap_or_else(|_| install_dir.to_path_buf())
            .join(PRODUCT_ID)
            .join("autoseg_models"),
    }
}

/// Add/Remove Programs key path for this product.
pub fn uninstall_key_path() -> String {
    format!(r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{PRODUCT_ID}")
}

/// Human-readable byte size, e.g. `1.3 GB`.
pub fn human_size(bytes: u64) -> String {
    const KB: f64 = 1_000.0;
    let b = bytes as f64;
    if b >= KB * KB * KB {
        format!("{:.1} GB", b / (KB * KB * KB))
    } else if b >= KB * KB {
        format!("{:.0} MB", b / (KB * KB))
    } else if b >= KB {
        format!("{:.0} kB", b / KB)
    } else {
        format!("{bytes} B")
    }
}
