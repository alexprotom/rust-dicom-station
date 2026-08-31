//! Product constants, install options and the defaults the UI starts from.

use std::path::{Path, PathBuf};

use crate::win::registry::Hive;

pub const APP_NAME: &str = "Rust DICOM Station";
pub const APP_EXE: &str = "rust-dicom-station.exe";
pub const PUBLISHER: &str = "Rust DICOM Station contributors";
/// Shown in Apps & features. Empty means "do not write the value".
pub const HOMEPAGE: &str = "";
/// Registry-safe product id, used for the Add/Remove Programs key.
pub const PRODUCT_ID: &str = "RustDicomStation";
/// ProgID for the optional `.dcm` file association.
pub const PROGID: &str = "RustDicomStation.DicomFile";
pub const UNINSTALLER_EXE: &str = "uninstall.exe";
pub const MANIFEST_FILE: &str = "install-manifest.txt";
/// The viewer's settings file, kept in its data folder; the installer only
/// pre-seeds it when the chosen model folder is not the viewer's default.
pub const SETTINGS_FILE: &str = "viewer_settings.txt";
/// The viewer's per-user folder under `%LOCALAPPDATA%`, where it keeps its
/// settings and, by default, the model folder. Must match
/// `rust_dicom_station::settings::APP_NAME`.
pub const VIEWER_DATA_DIR: &str = "RustDICOMStation";
/// The settings key naming the model root. Must match
/// `rust_dicom_station::settings::MODELS_DIR_KEY` (asserted by a test when
/// the viewer is linked in).
pub const SETTINGS_MODELS_KEY: &str = "models_dir";
/// The viewer's model root folder name; each engine keeps its own sub-folder
/// in it. Must match `rust_dicom_station::models::DIR_NAME`.
pub const MODELS_DIR_NAME: &str = "models";
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
    /// The model root — every engine's weights live in a sub-folder of it;
    /// see [`default_models_dir`].
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

    /// Change the destination folder, keeping a still-default model folder
    /// at the default for the new location.
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

/// `%LOCALAPPDATA%\Programs\Rust DICOM Station` or `%ProgramFiles%\Rust DICOM
/// Station`, with a plain-C: fallback should the shell folder lookup fail.
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

/// The viewer's own data folder, `%LOCALAPPDATA%\RustDICOMStation`.
pub fn viewer_data_dir() -> Option<PathBuf> {
    crate::win::local_app_data()
        .ok()
        .map(|d| d.join(VIEWER_DATA_DIR))
}

/// Where the viewer reads its settings from.
pub fn viewer_settings_path() -> Option<PathBuf> {
    viewer_data_dir().map(|d| d.join(SETTINGS_FILE))
}

/// Where the model root goes — the folder all three engines download into
/// (`models/totalsegmentator`, `models/segvol`, `models/medsam2`).
///
/// The viewer's default is `models/` in its per-user data folder, which is
/// writable whoever installed the program and wherever it went, so the same
/// default serves both scopes; only a folder chosen elsewhere has to be
/// recorded in `viewer_settings.txt`. The install folder is the fallback
/// when the shell cannot name `%LOCALAPPDATA%`.
pub fn default_models_dir(_scope: Scope, install_dir: &Path) -> PathBuf {
    viewer_data_dir()
        .unwrap_or_else(|| install_dir.to_path_buf())
        .join(MODELS_DIR_NAME)
}

#[cfg(all(test, feature = "prefetch-models"))]
mod tests {
    #[test]
    fn the_viewer_and_the_installer_agree_on_the_model_layout() {
        assert_eq!(
            super::SETTINGS_MODELS_KEY,
            rust_dicom_station::settings::MODELS_DIR_KEY
        );
        assert_eq!(super::MODELS_DIR_NAME, rust_dicom_station::models::DIR_NAME);
        assert_eq!(
            super::VIEWER_DATA_DIR,
            rust_dicom_station::settings::APP_NAME
        );
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
