use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use egui::ThemePreference;

const FILE_NAME: &str = "viewer_settings.txt";
const APP_NAME: &str = "RustDICOMStation";

/// Settings key of the model root; the installer writes it too.
pub const MODELS_DIR_KEY: &str = "models_dir";

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
}

impl Default for Settings {
    fn default() -> Self {
        // The viewer has always started dark; keep that as the default rather
        // than following the system, which would surprise existing users.
        Settings {
            theme: ThemePreference::Dark,
            models_dir: None,
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

fn theme_to_str(t: ThemePreference) -> &'static str {
    match t {
        ThemePreference::Dark => "dark",
        ThemePreference::Light => "light",
        ThemePreference::System => "system",
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

/// Read the settings file. A missing or unreadable file yields the defaults.
pub fn load() -> Settings {
    match std::fs::read_to_string(settings_path()) {
        Ok(text) => parse(&text),
        Err(_) => Settings::default(),
    }
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

fn parse(text: &str) -> Settings {
    let mut s = Settings::default();
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
                models_dir: None,
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
                models_dir: None
            },
            "case-insensitive key and value, surrounding space ignored"
        );
        assert_eq!(
            parse("theme = white"),
            Settings {
                theme: ThemePreference::Light,
                models_dir: None
            },
            "\"white\" accepted as an alias for light"
        );
        let with_dir = Settings {
            theme: ThemePreference::Dark,
            models_dir: Some(PathBuf::from("D:/models")),
        };
        assert_eq!(parse(&render(&with_dir)), with_dir, "model dir round trip");
    }
}
