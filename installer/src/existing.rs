//! Installations that are already on the machine.
//!
//! Running a setup over an existing installation updates it in place rather
//! than adding a second copy: the one in the chosen folder keeps its folder
//! and the answers given when it was installed, and every other registered
//! installation (another folder, or the other scope) is removed. That is
//! also what winget relies on - `winget upgrade` simply runs the new setup.
//!
//! An installation is found through its Add/Remove Programs key, one per
//! hive, and described by the manifest in its folder.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use crate::install::Manifest;
use crate::plan::*;
use crate::win::registry::Key;

/// One registered installation.
#[derive(Clone, Debug)]
pub struct Installed {
    pub scope: Scope,
    pub dir: PathBuf,
    /// `DisplayVersion` of its Apps & features entry.
    pub version: String,
    /// Empty when the folder or its manifest has gone missing - the entry is
    /// then only a registry leftover, which removing it cleans up.
    pub manifest: Manifest,
}

impl Installed {
    pub fn machine_wide(&self) -> bool {
        self.scope == Scope::AllUsers
    }

    /// "0.8.7 for all users in C:\Program Files\Rust DICOM Station".
    pub fn describe(&self) -> String {
        let who = match self.scope {
            Scope::CurrentUser => "for this user",
            Scope::AllUsers => "for all users",
        };
        let version = if self.version.is_empty() {
            "An unknown version"
        } else {
            &self.version
        };
        format!("{version} {who} in {}", self.dir.display())
    }
}

/// Every registered installation, the per-user one first.
pub fn find_all() -> Vec<Installed> {
    let mut out = Vec::new();
    for scope in [Scope::CurrentUser, Scope::AllUsers] {
        let Ok(key) = Key::open(scope.hive().hkey(), &uninstall_key_path(), false) else {
            continue;
        };
        let Some(dir) = key
            .get_str("InstallLocation")
            .filter(|s| !s.trim().is_empty())
        else {
            continue;
        };
        let dir = PathBuf::from(dir.trim());
        let mut manifest = std::fs::read_to_string(dir.join(MANIFEST_FILE))
            .map(|t| Manifest::parse(&t))
            .unwrap_or_default();
        if manifest.install_dir.as_os_str().is_empty() {
            manifest.install_dir = dir.clone();
        }
        manifest.machine_wide = scope == Scope::AllUsers;
        out.push(Installed {
            scope,
            version: key.get_str("DisplayVersion").unwrap_or_default(),
            dir,
            manifest,
        });
    }
    out
}

/// The installation in `dir`, which an install there updates in place, and
/// the others, which it removes (unless told to keep them).
pub fn split(all: &[Installed], dir: &Path) -> (Option<Installed>, Vec<Installed>) {
    let mut same = None;
    let mut others = Vec::new();
    for i in all {
        if same.is_none() && paths_equal(&i.dir, dir) {
            same = Some(i.clone());
        } else {
            others.push(i.clone());
        }
    }
    (same, others)
}

/// Which installation a setup started without `--dir` should update: the one
/// in the scope asked for, else the newest (the per-user one on a tie, which
/// needs no administrator rights).
pub fn preferred<'a>(
    all: &'a [Installed],
    scope: Option<Scope>,
    dir: Option<&Path>,
) -> Option<&'a Installed> {
    if let Some(dir) = dir {
        return all.iter().find(|i| paths_equal(&i.dir, dir));
    }
    if let Some(scope) = scope {
        return all.iter().find(|i| i.scope == scope);
    }
    let mut best: Option<&Installed> = None;
    for i in all {
        best = match best {
            None => Some(i),
            Some(b) if compare_versions(&i.version, &b.version) == Ordering::Greater => Some(i),
            keep => keep,
        };
    }
    best
}

/// The options an installation was made with, as far as they can be read
/// back: its folder and scope, the model folder, which shortcuts and
/// components it has, the file association and `PATH` entry, and the
/// graphics backend in its `viewer-defaults.txt`. An update starts from
/// these, so running it changes nothing the user did not ask to change.
pub fn options_from(i: &Installed) -> Options {
    let mut o = Options {
        scope: i.scope,
        dir: i.dir.clone(),
        ..Options::default()
    };
    let m = &i.manifest;
    o.models_dir = if m.models_dir.as_os_str().is_empty() {
        default_models_dir(o.scope, &o.dir)
    } else {
        m.models_dir.clone()
    };
    // A manifest that could not be read says nothing about the rest; keep
    // the defaults rather than switch everything off.
    if m.files.is_empty() {
        return o;
    }
    let start = crate::win::start_menu_programs().ok();
    let desktop = crate::win::desktop_dir().ok();
    let kinds: Vec<Option<ShortcutKind>> = m
        .shortcuts
        .iter()
        .map(|l| classify_shortcut(l, start.as_deref(), desktop.as_deref()))
        .collect();
    o.start_menu_shortcut = kinds.contains(&Some(ShortcutKind::StartMenu));
    o.desktop_shortcut = kinds.contains(&Some(ShortcutKind::Desktop));
    o.file_association = m.file_association;
    o.add_to_path = m.path_added;
    o.install_mcp = m.files.iter().any(|f| f.eq_ignore_ascii_case(MCP_EXE));
    o.graphics = std::fs::read_to_string(i.dir.join(DEFAULTS_FILE))
        .ok()
        .and_then(|t| setting_value(&t, SETTINGS_GRAPHICS_KEY))
        .and_then(|v| Graphics::from_key(&v))
        .unwrap_or(o.graphics);
    // Weights are fetched at the first installation only; an update never
    // downloads them again.
    o.models = Models::None;
    o
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShortcutKind {
    StartMenu,
    Desktop,
}

/// Which of the two shortcuts a recorded `.lnk` path was. Compared with the
/// folders of the user running the setup first, then by the path itself,
/// so a manifest written for another account still reads sensibly.
pub fn classify_shortcut(
    link: &Path,
    start_menu: Option<&Path>,
    desktop: Option<&Path>,
) -> Option<ShortcutKind> {
    let parent = link.parent()?;
    if start_menu.is_some_and(|s| paths_equal(parent, s)) {
        return Some(ShortcutKind::StartMenu);
    }
    if desktop.is_some_and(|d| paths_equal(parent, d)) {
        return Some(ShortcutKind::Desktop);
    }
    let lower = link
        .to_string_lossy()
        .to_ascii_lowercase()
        .replace('/', "\\");
    if lower.contains("\\start menu\\") {
        Some(ShortcutKind::StartMenu)
    } else if lower.contains("\\desktop\\") {
        Some(ShortcutKind::Desktop)
    } else {
        None
    }
}

/// `value` of the first uncommented `key = value` line.
fn setting_value(text: &str, key: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case(key))
        .map(|(_, v)| v.trim().to_string())
}

/// Case- and separator-insensitive path equality, as Windows sees it.
pub fn paths_equal(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| {
        p.to_string_lossy()
            .trim()
            .trim_end_matches(['\\', '/'])
            .to_ascii_lowercase()
            .replace('/', "\\")
    };
    norm(a) == norm(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(scope: Scope, dir: &str, version: &str) -> Installed {
        Installed {
            scope,
            dir: PathBuf::from(dir),
            version: version.to_string(),
            manifest: Manifest::default(),
        }
    }

    #[test]
    fn the_folder_decides_what_is_updated_and_what_is_removed() {
        let all = vec![
            inst(
                Scope::CurrentUser,
                r"C:\Users\a\AppData\Local\Programs\RDS",
                "0.8.7",
            ),
            inst(Scope::AllUsers, r"C:\Program Files\RDS", "0.8.5"),
        ];
        let (same, others) = split(&all, Path::new(r"c:\program files\rds\"));
        assert_eq!(same.unwrap().scope, Scope::AllUsers);
        assert_eq!(others.len(), 1);
        assert_eq!(others[0].scope, Scope::CurrentUser);

        let (same, others) = split(&all, Path::new(r"D:\Apps\RDS"));
        assert!(same.is_none(), "a new folder updates nothing in place");
        assert_eq!(others.len(), 2, "and both old copies are in the way");
    }

    #[test]
    fn without_a_folder_the_newest_installation_is_the_one_to_update() {
        let all = vec![
            inst(Scope::CurrentUser, r"C:\u", "0.8.5"),
            inst(Scope::AllUsers, r"C:\m", "0.8.10"),
        ];
        assert_eq!(preferred(&all, None, None).unwrap().scope, Scope::AllUsers);
        assert_eq!(
            preferred(&all, Some(Scope::CurrentUser), None)
                .unwrap()
                .scope,
            Scope::CurrentUser,
            "unless a scope was asked for"
        );
        assert!(preferred(&all, None, Some(Path::new(r"D:\x"))).is_none());
        let tie = vec![
            inst(Scope::CurrentUser, r"C:\u", "0.8.7"),
            inst(Scope::AllUsers, r"C:\m", "0.8.7"),
        ];
        assert_eq!(
            preferred(&tie, None, None).unwrap().scope,
            Scope::CurrentUser,
            "a tie goes to the copy that needs no administrator rights"
        );
    }

    #[test]
    fn shortcuts_are_recognised_by_folder_then_by_path() {
        let start = Path::new(r"C:\Users\a\AppData\Roaming\Microsoft\Windows\Start Menu\Programs");
        let desk = Path::new(r"C:\Users\a\OneDrive\Desktop");
        let s = start.join("Rust DICOM Station.lnk");
        let d = desk.join("Rust DICOM Station.lnk");
        assert_eq!(
            classify_shortcut(&s, Some(start), Some(desk)),
            Some(ShortcutKind::StartMenu)
        );
        assert_eq!(
            classify_shortcut(&d, Some(start), Some(desk)),
            Some(ShortcutKind::Desktop)
        );
        // Another account's folders: the path alone still tells.
        assert_eq!(
            classify_shortcut(&s, None, None),
            Some(ShortcutKind::StartMenu)
        );
        assert_eq!(
            classify_shortcut(Path::new(r"C:\Users\b\Desktop\x.lnk"), None, None),
            Some(ShortcutKind::Desktop)
        );
        assert_eq!(classify_shortcut(Path::new(r"D:\x.lnk"), None, None), None);
    }

    #[test]
    fn the_graphics_answer_is_read_back_from_the_defaults_file() {
        let text = "# comment\n# graphics_backend = vulkan\nGraphics_Backend = dx12\n";
        assert_eq!(
            setting_value(text, SETTINGS_GRAPHICS_KEY).as_deref(),
            Some("dx12")
        );
        assert_eq!(setting_value("theme = dark\n", SETTINGS_GRAPHICS_KEY), None);
    }
}
