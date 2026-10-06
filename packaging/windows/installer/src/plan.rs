//! Product constants, install options, exit codes and the defaults the UI
//! starts from.

use std::path::{Path, PathBuf};

use crate::win::registry::Hive;

pub use crate::product::*;

/// Shown in Apps & features. Empty means "do not write the value".
pub const HOMEPAGE: &str = "";
/// ProgID for the optional `.dcm` file association.
pub const PROGID: &str = "RustDicomStation.DicomFile";
/// The setup program itself, left in the program folder without its
/// payload: Apps & features runs it to uninstall (`--uninstall`) and the
/// *Update* shortcut runs it to fetch the newest release (`--update`).
pub const SETUP_EXE: &str = "rds-setup.exe";
/// What [`SETUP_EXE`] was called before the setup could update. An older
/// installation lists it in its manifest, so updating one removes it like
/// any other file the new version no longer ships.
pub const LEGACY_UNINSTALLER_EXE: &str = "uninstall.exe";
/// Where a [`SETUP_EXE`] that was still running when an update replaced it
/// is moved aside (Windows lets a running program be renamed, not
/// overwritten). Deleted by the next installation, update or uninstall.
pub const SETUP_EXE_OLD: &str = "rds-setup.exe.old";
pub const MANIFEST_FILE: &str = "install-manifest.txt";
/// The viewer's settings file, kept in the data folder of whoever runs the
/// installer. It wins over [`DEFAULTS_FILE`], so the installer updates it
/// too: re-running the installer to change an answer has to change it.
pub const SETTINGS_FILE: &str = "viewer_settings.txt";
/// Machine-wide defaults, written beside the installed executable and read
/// by every user of the machine before their own settings file. Must match
/// `rust_dicom_station::settings::DEFAULTS_FILE_NAME` (asserted by a test
/// when the viewer is linked in).
pub const DEFAULTS_FILE: &str = "viewer-defaults.txt";
/// The viewer's per-user folder under `%LOCALAPPDATA%`, where it keeps its
/// settings and, by default, the model folder. Must match
/// `rust_dicom_station::settings::APP_NAME`.
pub const VIEWER_DATA_DIR: &str = "RustDICOMStation";
/// The settings key naming the model root. Must match
/// `rust_dicom_station::settings::MODELS_DIR_KEY` (asserted by a test when
/// the viewer is linked in).
pub const SETTINGS_MODELS_KEY: &str = "models_dir";
/// The settings key naming the graphics backend. Must match
/// `rust_dicom_station::settings::GRAPHICS_BACKEND_KEY` (asserted by a test
/// when the viewer is linked in).
pub const SETTINGS_GRAPHICS_KEY: &str = "graphics_backend";

/// The viewer's settings key of the user's TotalSegmentator licence number.
/// Must match `rust_dicom_station::settings::TS_LICENCE_KEY`. It is only
/// ever written to the settings file of the user running the setup, never
/// to the machine-wide defaults.
pub const SETTINGS_LICENCE_KEY: &str = "totalsegmentator_licence";
/// The viewer's model root folder name; each engine keeps its own sub-folder
/// in it. Must match `rust_dicom_station::models::DIR_NAME`.
pub const MODELS_DIR_NAME: &str = "models";
/// Official Microsoft download for the x64 Visual C++ 2015-2022 runtime.
pub const VCREDIST_URL: &str = "https://aka.ms/vs/17/release/vc_redist.x64.exe";

/// Exit codes. Anything but 0 is a failure; the specific ones are listed in
/// the winget manifest (`ExpectedReturnCodes`), so winget can tell the user
/// what happened instead of printing a number.
pub const EXIT_OK: u8 = 0;
/// Any failure without a more specific code.
pub const EXIT_FAILED: u8 = 1;
/// The viewer is running from the folder being installed into or removed.
pub const EXIT_IN_USE: u8 = 2;
/// The user cancelled.
pub const EXIT_CANCELLED: u8 = 3;
/// `--silent` would replace a newer installed version (see
/// `--allow-downgrade`).
pub const EXIT_DOWNGRADE: u8 = 4;
/// The newest release could not be looked up or downloaded.
pub const EXIT_NO_NETWORK: u8 = 5;

/// An error that carries its own exit code.
#[derive(Debug)]
pub struct Failure {
    pub code: u8,
    pub message: String,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Failure {}

/// An `anyhow` error that `main` turns into exit code `code`.
pub fn failure(code: u8, message: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Failure {
        code,
        message: message.into(),
    })
}

/// The exit code an error stands for: the first [`Failure`] in its chain,
/// else [`EXIT_FAILED`].
pub fn exit_code_of(e: &anyhow::Error) -> u8 {
    e.chain()
        .find_map(|c| c.downcast_ref::<Failure>())
        .map(|f| f.code)
        .unwrap_or(EXIT_FAILED)
}

/// Who the installation is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    /// `%LOCALAPPDATA%\Programs\…` - no administrator rights needed.
    CurrentUser,
    /// `C:\Program Files\…` - needs elevation.
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

/// Which graphics API the viewer should use.
///
/// This is asked during installation because the failure it prevents happens
/// *before* the viewer can ask anything itself: a Windows machine that
/// advertises a Vulkan driver which cannot create a device gives a program
/// that dies on start, with no window to change a setting in. The viewer
/// falls back on its own nowadays, but starting on the right backend is
/// quicker and quieter than starting on the wrong one twice - and a site
/// that already knows its machines can set it once here.
///
/// The spellings must match `rust_dicom_station::gfx::Backend::key`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Graphics {
    /// Let the graphics library choose.
    Auto,
    /// The default, and the fastest where the driver is sound.
    Vulkan,
    /// Windows' own - the answer when Vulkan will not start.
    Dx12,
}

impl Graphics {
    pub const ALL: [Graphics; 3] = [Graphics::Vulkan, Graphics::Dx12, Graphics::Auto];

    /// What the viewer's settings file spells it.
    pub fn key(self) -> &'static str {
        match self {
            Graphics::Auto => "auto",
            Graphics::Vulkan => "vulkan",
            Graphics::Dx12 => "dx12",
        }
    }

    pub fn from_key(key: &str) -> Option<Graphics> {
        match key.trim().to_ascii_lowercase().as_str() {
            "auto" | "default" => Some(Graphics::Auto),
            "vulkan" | "vk" => Some(Graphics::Vulkan),
            "dx12" | "d3d12" | "directx" | "directx12" | "dx" => Some(Graphics::Dx12),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Graphics::Auto => "Automatic",
            Graphics::Vulkan => "Vulkan (recommended)",
            Graphics::Dx12 => "DirectX 12",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Graphics::Auto => {
                "Let the graphics library pick whichever it finds. Sensible, and the \
                 same thing older versions did."
            }
            Graphics::Vulkan => {
                "Usually the fastest, and right on almost every machine. A few Windows \
                 systems advertise a Vulkan driver that does not work; on those the \
                 viewer now falls back to DirectX 12 by itself, but choosing it here \
                 saves the first attempt."
            }
            Graphics::Dx12 => {
                "Windows' own graphics API. Present and dependable on every machine \
                 with Windows 10 or later. Choose this if the viewer has ever failed \
                 to start on this hardware."
            }
        }
    }
}

/// Which model weights to fetch during installation.
///
/// The viewer downloads every model on first use anyway; fetching here
/// moves the wait into the installation, on a machine that is online now.
/// A choice is either one of the named sets or a list of model keys as the
/// model manager names them (`totalsegmentator/total_3mm`,
/// `nnunet_v1/msd_lung`, ...; `rds-setup --list-models` prints them).
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub enum Models {
    /// Leave the model folder as it is.
    #[default]
    None,
    /// TotalSegmentator's `total` at 3 mm: what most runs use (~135 MB).
    Recommended,
    /// Every model whose weights may be used for anything, commercial work
    /// included (Apache-2.0, MIT).
    Open,
    /// Every model in the list, the non-commercial ones too (and the
    /// licensed TotalSegmentator models when a licence number is set).
    Every,
    /// These models, by key.
    Pick(Vec<String>),
}

/// The keys the TotalSegmentator sets of older installers stand for.
pub const KEY_TOTAL_3MM: &str = "totalsegmentator/total_3mm";
pub const KEY_TOTAL_6MM: &str = "totalsegmentator/total_6mm";
pub const KEYS_TOTAL_15MM: [&str; 5] = [
    "totalsegmentator/total_part1_organs",
    "totalsegmentator/total_part2_vertebrae",
    "totalsegmentator/total_part3_cardiac",
    "totalsegmentator/total_part4_muscles",
    "totalsegmentator/total_part5_ribs",
];

impl Models {
    /// The named sets, in the order the window offers them.
    pub fn presets() -> [Models; 4] {
        [Models::None, Models::Recommended, Models::Open, Models::Every]
    }

    pub fn label(&self) -> String {
        match self {
            Models::None => "None now - each model downloads on first use".to_string(),
            Models::Recommended => {
                "Recommended: TotalSegmentator total, 3 mm (~135 MB)".to_string()
            }
            Models::Open => "Every open-licence model (Apache-2.0, MIT)".to_string(),
            Models::Every => "Every model, the non-commercial ones too".to_string(),
            Models::Pick(k) => format!("{} chosen model(s)", k.len()),
        }
    }

    /// How the command line spells it.
    pub fn arg(&self) -> String {
        match self {
            Models::None => "none".to_string(),
            Models::Recommended => "recommended".to_string(),
            Models::Open => "open".to_string(),
            Models::Every => "every".to_string(),
            Models::Pick(k) if k.is_empty() => "none".to_string(),
            Models::Pick(k) => k.join(","),
        }
    }

    /// Parse `--models`: a set's name, one of the older TotalSegmentator
    /// sets (`3mm`, `6mm`, `1.5mm`, `all` = 3 mm and 1.5 mm), or model keys
    /// separated by commas.
    pub fn parse(v: &str) -> Option<Models> {
        let lower = v.trim().to_ascii_lowercase();
        Some(match lower.as_str() {
            "" | "none" => Models::None,
            "recommended" | "3mm" | "fast" => Models::Recommended,
            "open" => Models::Open,
            "every" => Models::Every,
            "6mm" => Models::Pick(vec![KEY_TOTAL_6MM.to_string()]),
            "1.5mm" | "15mm" | "highres" => {
                Models::Pick(KEYS_TOTAL_15MM.iter().map(|k| k.to_string()).collect())
            }
            "all" | "everything" => {
                let mut k = vec![KEY_TOTAL_3MM.to_string()];
                k.extend(KEYS_TOTAL_15MM.iter().map(|k| k.to_string()));
                Models::Pick(k)
            }
            _ => {
                let keys: Vec<String> = v
                    .split(',')
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty())
                    .collect();
                if keys.is_empty() || !keys.iter().all(|k| k.contains('/')) {
                    return None;
                }
                Models::Pick(keys)
            }
        })
    }
}

/// Everything the user can decide before the copy starts.
#[derive(Clone, Debug)]
pub struct Options {
    pub scope: Scope,
    pub dir: PathBuf,
    /// The model root - every engine's weights live in a sub-folder of it;
    /// see [`default_models_dir`].
    pub models_dir: PathBuf,
    pub models: Models,
    pub start_menu_shortcut: bool,
    pub desktop_shortcut: bool,
    pub add_to_path: bool,
    /// Register `.dcm`/`.dicom` and an "Open with " entry on folders.
    pub file_association: bool,
    /// Install the Microsoft Visual C++ runtime when it is missing.
    pub install_vcredist: bool,
    /// Install [`MCP_EXE`] alongside the viewer. Ignored when the installer
    /// carries no MCP server.
    pub install_mcp: bool,
    /// Install [`PACS_EXE`] alongside the viewer. Ignored when the installer
    /// carries no PACS server.
    pub install_pacs: bool,
    pub launch_after: bool,
    /// Which graphics API the viewer should start on.
    pub graphics: Graphics,
    /// Remove every other registered installation - another folder, or the
    /// other scope - so that one copy remains. The installation in [`dir`]
    /// itself is always updated in place.
    ///
    /// [`dir`]: Options::dir
    pub remove_others: bool,
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
            // On: an installer that carries the server is one somebody asked
            // to be built with it, and leaving a 20 MB executable out by
            // default only means answering the question twice.
            install_mcp: true,
            // Off: a default installation is a client. The server is for
            // the one machine whose archive others are to reach.
            install_pacs: false,
            launch_after: true,
            // Vulkan is the right default: it is the faster backend and it
            // works on the overwhelming majority of machines. The page
            // exists for the ones where it does not.
            graphics: Graphics::Vulkan,
            // On: a second copy is almost always a leftover, and one copy
            // per machine is what the Update shortcut and winget assume.
            remove_others: true,
        }
    }
}

impl Options {
    pub fn exe_path(&self) -> PathBuf {
        self.dir.join(APP_EXE)
    }

    /// The setup program kept in the program folder, see [`SETUP_EXE`].
    pub fn setup_path(&self) -> PathBuf {
        self.dir.join(SETUP_EXE)
    }

    pub fn mcp_path(&self) -> PathBuf {
        self.dir.join(MCP_EXE)
    }

    pub fn pacs_path(&self) -> PathBuf {
        self.dir.join(PACS_EXE)
    }

    /// The payload entries this installation leaves out.
    pub fn skipped_files(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if !self.install_mcp {
            out.push(MCP_EXE);
        }
        if !self.install_pacs {
            out.push(PACS_EXE);
        }
        out
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

/// Where the model root goes - the folder all three engines download into
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
        assert_eq!(
            super::DEFAULTS_FILE,
            rust_dicom_station::settings::DEFAULTS_FILE_NAME
        );
        assert_eq!(
            super::SETTINGS_LICENCE_KEY,
            rust_dicom_station::settings::TS_LICENCE_KEY
        );
    }

    /// The sets the setup offers name models the viewer has.
    #[test]
    fn the_named_models_exist() {
        let keys: Vec<String> = rust_dicom_station::models::inventory()
            .into_iter()
            .map(|a| a.key)
            .collect();
        for k in std::iter::once(super::KEY_TOTAL_3MM)
            .chain(std::iter::once(super::KEY_TOTAL_6MM))
            .chain(super::KEYS_TOTAL_15MM)
        {
            assert!(keys.iter().any(|x| x == k), "{k}");
        }
    }

    /// The installer writes a backend into the viewer's settings file; if the
    /// two ever disagreed about how to spell one, the viewer would silently
    /// ignore the choice the user made during installation.
    #[test]
    fn the_viewer_and_the_installer_agree_on_the_graphics_setting() {
        use rust_dicom_station::gfx::Backend;
        assert_eq!(
            super::SETTINGS_GRAPHICS_KEY,
            rust_dicom_station::settings::GRAPHICS_BACKEND_KEY
        );
        for g in super::Graphics::ALL {
            assert_eq!(
                Backend::from_key(g.key()).map(|b| b.key()),
                Some(g.key()),
                "the viewer reads back '{}' as itself",
                g.key()
            );
        }
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
