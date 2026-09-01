use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use egui::ThemePreference;

use crate::gfx::Backend;

const FILE_NAME: &str = "viewer_settings.txt";

/// Machine-wide defaults, written by the installer next to the executable.
pub const DEFAULTS_FILE_NAME: &str = "viewer-defaults.txt";

/// The folder name under the platform's config / data root
/// (`%LOCALAPPDATA%\RustDICOMStation`, `~/.config/RustDICOMStation`,
/// `~/.local/share/RustDICOMStation`); the installer must agree with it.
pub const APP_NAME: &str = "RustDICOMStation";

/// Settings key of the model root; the installer writes it too.
pub const MODELS_DIR_KEY: &str = "models_dir";

/// Settings key of the patient archive root.
pub const ARCHIVE_DIR_KEY: &str = "archive_dir";

/// Settings key of the graphics backend; the installer writes it from the
/// page it asks on, and the View menu changes it afterwards.
pub const GRAPHICS_BACKEND_KEY: &str = "graphics_backend";

/// Settings keys of the two optional side-panel modules.
const MODULE_REG_KEY: &str = "module_image_registration";
const MODULE_SIM_KEY: &str = "module_image_simulation";

/// User preferences that survive a restart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// Light / dark / follow-the-system appearance.
    pub theme: ThemePreference,

    /// Root of the downloaded network weights.
    ///
    /// `None` means use the platform-specific default returned by
    /// [`default_models_dir`].
    pub models_dir: Option<PathBuf>,

    /// Root of the local patient archive.
    ///
    /// `None` means the platform-specific default,
    /// [`crate::archive::default_root`].
    pub archive_dir: Option<PathBuf>,

    /// *Modules ▶ Image registration*: the registration section is shown in
    /// the side panel.
    pub module_registration: bool,

    /// *Modules ▶ Image simulation*: the simulation section is shown in the
    /// side panel.
    pub module_simulation: bool,

    /// Which graphics backend to draw and compute with. Read once at
    /// startup, before the window exists, so a change only takes effect on
    /// the next run — which the menu says.
    pub graphics_backend: Backend,
}

impl Default for Settings {
    fn default() -> Self {
        // The viewer has always started dark; keep that as the default rather
        // than following the system, which would surprise existing users.
        Settings {
            theme: ThemePreference::Dark,
            models_dir: None,
            archive_dir: None,
            // Both optional modules start hidden; the Modules menu turns them
            // on and the choice is remembered.
            module_registration: false,
            module_simulation: false,
            // Let wgpu choose. The installer writes an explicit value when
            // the person installing picks one.
            graphics_backend: Backend::Auto,
        }
    }
}

/// The folder the application runs from ("the main app folder"), falling back
/// to the current working directory when the executable path is unavailable.
pub fn app_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Return the platform-specific directory used for persistent application
/// configuration.
///
/// Linux:
///   $XDG_CONFIG_HOME/RustDICOMStation
///   or ~/.config/RustDICOMStation
///
/// Windows:
///   %LOCALAPPDATA%\RustDICOMStation
///
/// macOS:
///   ~/Library/Application Support/RustDICOMStation
pub fn config_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
            if !dir.is_empty() {
                return PathBuf::from(dir).join(APP_NAME);
            }
        }

        if let Some(home) = home_dir() {
            return home.join(".config").join(APP_NAME);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
            if !dir.is_empty() {
                return PathBuf::from(dir).join(APP_NAME);
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(home) = home_dir() {
            return home
                .join("Library")
                .join("Application Support")
                .join(APP_NAME);
        }
    }

    // Fallback for unsupported platforms or unusual environments.
    app_dir()
}

/// Return the platform-specific directory used for persistent application
/// data such as downloaded model weights.
///
/// Linux:
///   $XDG_DATA_HOME/RustDICOMStation
///   or ~/.local/share/RustDICOMStation
///
/// Windows:
///   %LOCALAPPDATA%\RustDICOMStation
///
/// macOS:
///   ~/Library/Application Support/RustDICOMStation
pub fn data_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
            if !dir.is_empty() {
                return PathBuf::from(dir).join(APP_NAME);
            }
        }

        if let Some(home) = home_dir() {
            return home.join(".local").join("share").join(APP_NAME);
        }
    }

    #[cfg(target_os = "windows")]
    {
        if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
            if !dir.is_empty() {
                return PathBuf::from(dir).join(APP_NAME);
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(home) = home_dir() {
            return home
                .join("Library")
                .join("Application Support")
                .join(APP_NAME);
        }
    }

    // Fallback for unsupported platforms or unusual environments.
    app_dir()
}

/// Default root directory for downloaded model weights.
pub fn default_models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Best-effort home directory lookup used only as a fallback for platforms
/// where the relevant standard environment variable is not available.
#[cfg(not(windows))]
fn home_dir() -> Option<PathBuf> {
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// Full path of the settings file.
pub fn settings_path() -> PathBuf {
    config_dir().join(FILE_NAME)
}

/// Full path of the machine-wide defaults file, beside the executable.
///
/// A machine-wide installation is performed by an administrator whose
/// `%LOCALAPPDATA%` is not the one the viewer will later run under, so the
/// installer's answers cannot be written into the settings file of everyone
/// who will use the program — those files do not exist yet. They go into a
/// small file next to the executable instead, in the same `key = value`
/// syntax, and every key in it is only a *default*: the user's own settings
/// file is read afterwards and wins, and so does anything they change from
/// the menus.
///
/// `None` when the executable's own path cannot be determined, which is not
/// a condition worth reporting — it just means there are no defaults.
pub fn defaults_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join(DEFAULTS_FILE_NAME))
}

fn theme_to_str(t: ThemePreference) -> &'static str {
    match t {
        ThemePreference::Dark => "dark",
        ThemePreference::Light => "light",
        ThemePreference::System => "system",
    }
}

fn bool_to_str(b: bool) -> &'static str {
    if b {
        "on"
    } else {
        "off"
    }
}

fn bool_from_str(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "1" => Some(true),
        "off" | "false" | "no" | "0" => Some(false),
        _ => None,
    }
}

fn theme_from_str(s: &str) -> Option<ThemePreference> {
    match s.trim().to_ascii_lowercase().as_str() {
        "dark" => Some(ThemePreference::Dark),
        "light" | "white" => Some(ThemePreference::Light),
        "system" | "auto" => Some(ThemePreference::System),
        _ => None,
    }
}

/// Read the settings.
///
/// Two files, in increasing order of authority: the machine-wide defaults
/// the installer left beside the executable (see [`defaults_path`]), then
/// the user's own file. Either may be missing, and a missing or unreadable
/// one simply contributes nothing.
pub fn load() -> Settings {
    let mut s = Settings::default();
    if let Some(text) = defaults_path().and_then(|p| std::fs::read_to_string(p).ok()) {
        s = parse_into(s, &text);
    }
    if let Ok(text) = std::fs::read_to_string(settings_path()) {
        s = parse_into(s, &text);
    }
    s
}

/// Write the settings file.
///
/// The configuration directory is created on demand because it normally does
/// not exist on a first run.
pub fn save(s: &Settings) -> Result<()> {
    let path = settings_path();

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }

    std::fs::write(&path, render(s)).with_context(|| format!("write {}", path.display()))
}

/// Parse a whole file on its own. Only the tests read a file in isolation;
/// [`load`] layers two of them with [`parse_into`].
#[cfg(test)]
fn parse(text: &str) -> Settings {
    parse_into(Settings::default(), text)
}

/// Apply every key the text sets on top of what is already known.
///
/// Keys the text does not mention are left alone, which is what makes the
/// two files layer: the user's file overrides only the settings it actually
/// contains.
fn parse_into(mut s: Settings, text: &str) -> Settings {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.eq_ignore_ascii_case("theme") {
            if let Some(t) = theme_from_str(value) {
                s.theme = t;
            }
        } else if key.eq_ignore_ascii_case(MODELS_DIR_KEY) {
            let v = value.trim();
            if !v.is_empty() {
                s.models_dir = Some(PathBuf::from(v));
            }
        } else if key.eq_ignore_ascii_case(ARCHIVE_DIR_KEY) {
            let v = value.trim();
            if !v.is_empty() {
                s.archive_dir = Some(PathBuf::from(v));
            }
        } else if key.eq_ignore_ascii_case(MODULE_REG_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_registration = b;
            }
        } else if key.eq_ignore_ascii_case(MODULE_SIM_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_simulation = b;
            }
        } else if key.eq_ignore_ascii_case(GRAPHICS_BACKEND_KEY) {
            // An unreadable value leaves the default rather than failing to
            // start: this file is edited by hand and by an installer, and a
            // typo in it must not cost someone their program.
            if let Some(b) = Backend::from_key(value) {
                s.graphics_backend = b;
            }
        }
    }
    s
}

fn render(s: &Settings) -> String {
    let mut out = format!(
        "# rust-dicom-station user settings\n\
         # theme = dark | light | system\n\
         theme = {}\n",
        theme_to_str(s.theme)
    );
    if let Some(dir) = &s.models_dir {
        out.push_str(&format!("{MODELS_DIR_KEY} = {}\n", dir.display()));
    }
    if let Some(dir) = &s.archive_dir {
        out.push_str(&format!("{ARCHIVE_DIR_KEY} = {}\n", dir.display()));
    }
    out.push_str(&format!(
        "# optional side-panel modules (Modules menu) = on | off\n\
         {MODULE_REG_KEY} = {}\n\
         {MODULE_SIM_KEY} = {}\n\
         # graphics backend = auto | vulkan | dx12 | metal | opengl\n\
         # (the WGPU_BACKEND environment variable overrides this)\n\
         {GRAPHICS_BACKEND_KEY} = {}\n",
        bool_to_str(s.module_registration),
        bool_to_str(s.module_simulation),
        s.graphics_backend.key()
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_every_theme() {
        for theme in [
            ThemePreference::Dark,
            ThemePreference::Light,
            ThemePreference::System,
        ] {
            let s = Settings {
                theme,
                ..Settings::default()
            };
            assert_eq!(parse(&render(&s)), s, "round trip of {theme:?}");
        }
    }

    #[test]
    fn tolerates_junk_and_falls_back_to_defaults() {
        assert_eq!(parse(""), Settings::default());
        assert_eq!(parse("# only a comment\n\n"), Settings::default());
        assert_eq!(parse("theme"), Settings::default(), "no separator");
        assert_eq!(parse("theme = mauve"), Settings::default(), "unknown value");
        assert_eq!(
            parse("unknown = 3\nTHEME =  Light \n"),
            Settings {
                theme: ThemePreference::Light,
                ..Settings::default()
            },
            "case-insensitive key and value, surrounding space ignored"
        );
        assert_eq!(
            parse("theme = white"),
            Settings {
                theme: ThemePreference::Light,
                ..Settings::default()
            },
            "\"white\" accepted as an alias for light"
        );
        let with_dir = Settings {
            theme: ThemePreference::Dark,
            models_dir: Some(PathBuf::from("D:/models")),
            ..Settings::default()
        };
        assert_eq!(parse(&render(&with_dir)), with_dir, "model dir round trip");
    }

    #[test]
    fn round_trips_the_graphics_backend() {
        for b in Backend::ALL {
            let s = Settings {
                graphics_backend: b,
                ..Settings::default()
            };
            assert_eq!(parse(&render(&s)), s, "round trip of {}", b.key());
        }
        // The installer and hand editing both produce spellings `render`
        // never emits; none of them may cost someone their program.
        assert_eq!(
            parse(&format!("{GRAPHICS_BACKEND_KEY} = DX12")).graphics_backend,
            Backend::Dx12
        );
        assert_eq!(
            parse(&format!("{GRAPHICS_BACKEND_KEY} = nonsense")).graphics_backend,
            Backend::default(),
            "an unreadable value leaves the default instead of failing"
        );
    }

    #[test]
    fn the_users_file_overrides_the_machine_wide_defaults_key_by_key() {
        // What `load` does, without touching the real filesystem: the
        // installer's file first, the user's on top.
        let machine = format!(
            "{GRAPHICS_BACKEND_KEY} = dx12
{MODELS_DIR_KEY} = C:/ProgramData/models
"
        );
        let user = "theme = light
";
        let merged = parse_into(parse_into(Settings::default(), &machine), user);
        assert_eq!(merged.theme, ThemePreference::Light, "the user's own key");
        assert_eq!(
            merged.graphics_backend,
            Backend::Dx12,
            "a key only the installer set survives"
        );
        assert_eq!(
            merged.models_dir,
            Some(PathBuf::from("C:/ProgramData/models"))
        );

        // …and when both files speak, the user wins.
        let user = format!(
            "{GRAPHICS_BACKEND_KEY} = vulkan
"
        );
        let merged = parse_into(parse_into(Settings::default(), &machine), &user);
        assert_eq!(merged.graphics_backend, Backend::Vulkan);
    }

    #[test]
    fn round_trips_the_module_flags() {
        for (reg, sim) in [(false, false), (true, false), (false, true), (true, true)] {
            let s = Settings {
                module_registration: reg,
                module_simulation: sim,
                ..Settings::default()
            };
            assert_eq!(parse(&render(&s)), s, "round trip of ({reg}, {sim})");
        }
        assert!(
            parse(&format!("{MODULE_REG_KEY} = TRUE")).module_registration,
            "case-insensitive alias"
        );
    }
}
