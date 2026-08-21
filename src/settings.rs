//! Persistent user preferences.
//!
//! Stored as a tiny `key = value` text file next to the executable
//! (`viewer_settings.txt`) — no extra dependencies, trivially inspectable and
//! safe to delete by hand. Unknown keys and malformed lines are ignored, so
//! the file format can grow without breaking older or newer builds.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use egui::ThemePreference;

const FILE_NAME: &str = "viewer_settings.txt";

/// User preferences that survive a restart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Settings {
    /// Light / dark / follow-the-system appearance.
    pub theme: ThemePreference,
    /// Auto-segmentation model cache directory (None = default:
    /// `autoseg_models/` next to the executable).
    pub autoseg_dir: Option<PathBuf>,
}

impl Default for Settings {
    fn default() -> Self {
        // The viewer has always started dark; keep that as the default rather
        // than following the system, which would surprise existing users.
        Settings {
            theme: ThemePreference::Dark,
            autoseg_dir: None,
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

/// Full path of the settings file.
pub fn settings_path() -> PathBuf {
    app_dir().join(FILE_NAME)
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

/// Write the settings file. Best-effort: the application folder can
/// legitimately be read-only (e.g. an install under `Program Files`), so
/// callers report failures non-fatally instead of blocking the UI.
pub fn save(s: &Settings) -> Result<()> {
    let path = settings_path();
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
        } else if key.eq_ignore_ascii_case("autoseg_models_dir") {
            let v = value.trim();
            if !v.is_empty() {
                s.autoseg_dir = Some(PathBuf::from(v));
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
    if let Some(dir) = &s.autoseg_dir {
        out.push_str(&format!("autoseg_models_dir = {}\n", dir.display()));
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
                autoseg_dir: None,
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
                autoseg_dir: None
            },
            "case-insensitive key and value, surrounding space ignored"
        );
        assert_eq!(
            parse("theme = white"),
            Settings {
                theme: ThemePreference::Light,
                autoseg_dir: None
            },
            "\"white\" accepted as an alias for light"
        );
        let with_dir = Settings {
            theme: ThemePreference::Dark,
            autoseg_dir: Some(PathBuf::from("D:/models")),
        };
        assert_eq!(parse(&render(&with_dir)), with_dir, "model dir round trip");
    }
}
