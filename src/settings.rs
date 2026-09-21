use std::ffi::OsString;
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
const MODULE_PROP_KEY: &str = "module_structures_propagation";
const MODULE_TOOLS_KEY: &str = "module_structure_editor";
const MODULE_AUTO_KEY: &str = "module_structure_auto_tools";
const MODULE_DOSE_KEY: &str = "module_dose_estimation";
const MODULE_INFO_KEY: &str = "module_image_information";
const MODULE_PLAY_KEY: &str = "module_playback";
/// Which module sections were unfolded, remembered for *Restore the last
/// session*.
const MODULES_OPEN_KEY: &str = "modules_open";

/// Settings keys of the last session's sources, one per dataset. The paths
/// are separated by `|`, which no path on any supported system contains.
const SESSION_KEYS: [&str; 2] = ["session_a", "session_b"];
const SESSION_SEP: char = '|';

/// Settings keys of what each row of the central area shows, one per row.
const VIEW_ROW_KEYS: [&str; 2] = ["view_row_a", "view_row_b"];

/// What one pane of a row shows.
///
/// A row is a list of these, left to right, and the layout gives each of
/// them an equal share of the row's width - so a row of two panes is two
/// larger images rather than two images and a gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaneKind {
    /// One plane of the dataset's volume: an ordinary MPR view.
    Plane(crate::volume::ViewPlane),
    /// The surface scene of that dataset's structures and segmentations -
    /// the same one the *3D* button opens in a window, drawn in the row
    /// instead.
    Scene3d,
}

/// How many panes one row may show: every kind there is, so a row can carry
/// the three planes and the surface scene at once. The layout would divide
/// by more perfectly happily; what stops it is the width each pane is left
/// with, and four across a wide screen is about where a CT stops being
/// readable.
pub const MAX_PANES: usize = PaneKind::ALL.len();

impl PaneKind {
    /// Every kind a row can be given, in the order the tick boxes list them.
    pub const ALL: [PaneKind; 4] = [
        PaneKind::Plane(crate::volume::ViewPlane::Axial),
        PaneKind::Plane(crate::volume::ViewPlane::Sagittal),
        PaneKind::Plane(crate::volume::ViewPlane::Coronal),
        PaneKind::Scene3d,
    ];

    /// What the tick box and the pane's own corner call it.
    pub fn label(self) -> &'static str {
        match self {
            PaneKind::Plane(p) => p.title(),
            PaneKind::Scene3d => "3D",
        }
    }

    fn key(self) -> &'static str {
        match self {
            PaneKind::Plane(crate::volume::ViewPlane::Axial) => "axial",
            PaneKind::Plane(crate::volume::ViewPlane::Sagittal) => "sagittal",
            PaneKind::Plane(crate::volume::ViewPlane::Coronal) => "coronal",
            PaneKind::Scene3d => "3d",
        }
    }

    fn from_key(s: &str) -> Option<PaneKind> {
        PaneKind::ALL.into_iter().find(|k| k.key() == s)
    }
}

/// What a row shows when nothing says otherwise: the three planes, which is
/// the layout the program has always had.
pub fn default_view_row() -> Vec<PaneKind> {
    vec![
        PaneKind::Plane(crate::volume::ViewPlane::Axial),
        PaneKind::Plane(crate::volume::ViewPlane::Sagittal),
        PaneKind::Plane(crate::volume::ViewPlane::Coronal),
    ]
}

/// Read one row back from the settings file.
///
/// A row that names nothing the program knows, or nothing at all, keeps the
/// default rather than leaving the user with an empty window; anything past
/// the third pane and any repeat is dropped.
fn parse_view_row(value: &str) -> Vec<PaneKind> {
    let mut out: Vec<PaneKind> = Vec::new();
    for word in value.split(',') {
        let Some(k) = PaneKind::from_key(word.trim().to_lowercase().as_str()) else {
            continue;
        };
        if !out.contains(&k) && out.len() < MAX_PANES {
            out.push(k);
        }
    }
    if out.is_empty() {
        default_view_row()
    } else {
        out
    }
}

fn render_view_row(row: &[PaneKind]) -> String {
    row.iter().map(|k| k.key()).collect::<Vec<_>>().join(",")
}

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
    /// the modules panel.
    pub module_registration: bool,

    /// *Modules ▶ Image simulation*: the simulation section is shown in the
    /// modules panel.
    pub module_simulation: bool,

    /// *Modules ▶ Structure propagation*: the propagation section is shown
    /// in the modules panel.
    pub module_propagation: bool,

    /// *Modules ▶ Structure editor*: inserting, editing and combining
    /// structures is shown in the modules panel. On by default.
    pub module_structures: bool,

    /// *Modules ▶ Structure auto tools*: body contour and the three
    /// segmentation engines are shown in the modules panel. On by default.
    pub module_auto: bool,

    /// *Modules ▶ Dose estimation*: the dose metrics table of the ticked
    /// structures is shown in the modules panel. On by default.
    pub module_dose: bool,

    /// *Modules ▶ Image information*: the geometry, sampling and
    /// acquisition of the displayed series. On by default - it reads
    /// nothing until its section is unfolded.
    pub module_info: bool,

    /// *Modules ▶ Playback*: the transport controls and the settings of
    /// the Play buttons on the viewports. On by default.
    pub module_play: bool,

    /// The module sections that were unfolded when the settings were last
    /// written (`registration`, `simulation`, `editor`, `auto`,
    /// `propagation`, `dose`). Every section starts folded; *Restore the
    /// last session* unfolds these again.
    pub modules_open: Vec<String>,

    /// Which graphics backend to draw and compute with. Read once at
    /// startup, before the window exists, so a change only takes effect on
    /// the next run - which the menu says.
    pub graphics_backend: Backend,

    /// What dataset A and dataset B were last loaded from: folders and
    /// files, in the order they were added, so *Restore the last session*
    /// can put the same data back.
    pub session: [Vec<PathBuf>; 2],

    /// What each row of the central area shows, left to right: up to three
    /// panes, chosen under *Settings ▸ View layout*.
    pub view_rows: [Vec<PaneKind>; 2],
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
            module_propagation: false,
            module_structures: true,
            module_auto: true,
            module_dose: true,
            module_info: true,
            module_play: true,
            modules_open: Vec::new(),
            session: [Vec::new(), Vec::new()],
            view_rows: [default_view_row(), default_view_row()],
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

/// Where the program runs from when it was installed as a snap.
///
/// snapd sets these variables for every command of a snap and nothing else
/// sets them, so their presence is how the program knows. Three things
/// change inside a snap:
///
/// * **The folders.** snapd points `$HOME` at `~/snap/<name>/<revision>` and
///   copies that folder on every refresh, keeping the last few revisions.
///   Multi-gigabyte model weights and the patient archive must not be copied
///   like that, so configuration and data both live in `$SNAP_USER_COMMON`
///   (`~/snap/<name>/common`), which every revision shares.
/// * **The MCP server's command.** An MCP client cannot run the binary
///   inside the snap; it runs the snap's command, `/snap/bin/<name>.rds-mcp`.
/// * **Starting the viewer from the MCP server** goes through the snap's
///   desktop launcher, which sets up the graphics environment the viewer
///   command gets from its own launcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapEnv {
    /// `$SNAP_INSTANCE_NAME`: what the commands are called under `/snap/bin`
    /// (the snap name, or `name_key` for a parallel installation).
    pub instance: String,
    /// `$SNAP`: the read-only mount of the running revision.
    pub root: PathBuf,
    /// `$SNAP_USER_COMMON`: per user, shared by every revision.
    pub user_common: PathBuf,
}

impl SnapEnv {
    /// Where the settings and `mcp.toml` live.
    pub fn config_dir(&self) -> PathBuf {
        self.user_common.join("config")
    }

    /// Where models, the archive, templates and the MCP audit log live.
    pub fn data_dir(&self) -> PathBuf {
        self.user_common.join("data")
    }

    /// The command snapd exposes for one of the snap's apps: `/snap/bin/<name>`
    /// for the app named like the snap, `/snap/bin/<name>.<app>` otherwise.
    pub fn command(&self, app: &str) -> PathBuf {
        let snap_name = self.instance.split('_').next().unwrap_or(&self.instance);
        if app == snap_name {
            PathBuf::from("/snap/bin").join(&self.instance)
        } else {
            PathBuf::from("/snap/bin").join(format!("{}.{app}", self.instance))
        }
    }

    /// The desktop launcher the snap's viewer command runs through (put
    /// there by the GNOME extension, see `packaging/linux/snap/snapcraft.yaml`), when it
    /// exists.
    pub fn desktop_launcher(&self) -> Option<PathBuf> {
        let p = self
            .root
            .join("snap")
            .join("command-chain")
            .join("desktop-launch");
        p.is_file().then_some(p)
    }

    /// Read the snap environment through `get`; `None` unless all three
    /// variables are set and not empty.
    fn from_vars(get: impl Fn(&str) -> Option<OsString>) -> Option<SnapEnv> {
        let var = |k: &str| get(k).filter(|v| !v.is_empty());
        let instance = var("SNAP_INSTANCE_NAME").or_else(|| var("SNAP_NAME"))?;
        Some(SnapEnv {
            instance: instance.to_string_lossy().into_owned(),
            root: PathBuf::from(var("SNAP")?),
            user_common: PathBuf::from(var("SNAP_USER_COMMON")?),
        })
    }
}

/// The snap this process runs from, if it runs from one (Linux only).
pub fn snap_env() -> Option<SnapEnv> {
    if cfg!(target_os = "linux") {
        SnapEnv::from_vars(|k| std::env::var_os(k))
    } else {
        None
    }
}

/// Return the platform-specific directory used for persistent application
/// configuration.
///
/// Linux:
///   $XDG_CONFIG_HOME/RustDICOMStation
///   or ~/.config/RustDICOMStation
///   or, installed as a snap, ~/snap/<name>/common/config (see [`SnapEnv`])
///
/// Windows:
///   %LOCALAPPDATA%\RustDICOMStation
///
/// macOS:
///   ~/Library/Application Support/RustDICOMStation
///
/// Android:
///   the app's private files folder, as handed over by the activity
///   (see [`android::set_dirs`])
pub fn config_dir() -> PathBuf {
    #[cfg(target_os = "android")]
    {
        if let Some(dir) = android::config_dir() {
            return dir;
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(snap) = snap_env() {
            return snap.config_dir();
        }
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
///   or, installed as a snap, ~/snap/<name>/common/data (see [`SnapEnv`])
///
/// Windows:
///   %LOCALAPPDATA%\RustDICOMStation
///
/// macOS:
///   ~/Library/Application Support/RustDICOMStation
///
/// Android:
///   the app's folder on the shared storage
///   (`Android/data/<package>/files`, see [`android::set_dirs`])
pub fn data_dir() -> PathBuf {
    #[cfg(target_os = "android")]
    {
        if let Some(dir) = android::data_dir() {
            return dir;
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Some(snap) = snap_env() {
            return snap.data_dir();
        }
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
/// Where the Structure editor keeps the axes a user saves:
/// `<data dir>/user_data/structure_editor/user_axes`, created on demand.
pub fn user_axes_dir() -> PathBuf {
    let dir = data_dir()
        .join("user_data")
        .join("structure_editor")
        .join("user_axes");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

pub fn default_models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Android has no home folder and no environment variable for the app's
/// storage: the activity hands the two folders over at start-up, and the
/// Android entry point (`packaging/android/src/lib.rs`) stores them here before
/// anything reads a setting. Both are private to the app: the first is
/// file-encrypted internal storage (settings), the second the app's folder
/// on the shared storage (models, archive), which needs no permission and
/// is removed with the app.
#[cfg(target_os = "android")]
pub mod android {
    use std::path::PathBuf;
    use std::sync::OnceLock;

    static DIRS: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();

    /// Record the folders. The first call wins; later ones are ignored.
    pub fn set_dirs(config: PathBuf, data: PathBuf) {
        let _ = DIRS.set((config, data));
    }

    pub(super) fn config_dir() -> Option<PathBuf> {
        DIRS.get().map(|(c, _)| c.clone())
    }

    pub(super) fn data_dir() -> Option<PathBuf> {
        DIRS.get().map(|(_, d)| d.clone())
    }
}

/// Best-effort home directory lookup used only as a fallback for platforms
/// where the relevant standard environment variable is not available.
#[cfg(not(any(windows, target_os = "android")))]
fn home_dir() -> Option<PathBuf> {
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }
}

/// Full path of the MCP server's configuration file (`docs/mcp.md`).
pub fn mcp_config_path() -> PathBuf {
    config_dir().join("mcp.toml")
}

/// Where the MCP server executable would be: beside this one.
pub fn mcp_exe_path() -> PathBuf {
    let name = if cfg!(windows) {
        "rds-mcp.exe"
    } else {
        "rds-mcp"
    };
    app_dir().join(name)
}

/// How an MCP client starts the server: a command and its arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpLaunch {
    pub command: PathBuf,
    pub args: Vec<String>,
}

impl McpLaunch {
    /// Where this copy of the program was installed from decides it:
    ///
    /// * a snap - the snap's command, `/snap/bin/<name>.rds-mcp`; the binary
    ///   inside the snap cannot be run from outside it;
    /// * a Flatpak (`$FLATPAK_ID`, set inside the sandbox) - `flatpak run
    ///   --command=rds-mcp <id>`, for the same reason;
    /// * an AppImage (`$APPIMAGE`, set by its runtime) - the AppImage file
    ///   with `mcp`, which its AppRun dispatches on; the executable beside
    ///   this one lives in a mount that is gone once the program exits;
    /// * anything else - `rds-mcp` beside this executable.
    fn resolve(
        snap: Option<&SnapEnv>,
        flatpak: Option<&str>,
        appimage: Option<PathBuf>,
        beside: PathBuf,
    ) -> McpLaunch {
        if let Some(snap) = snap {
            McpLaunch {
                command: snap.command("rds-mcp"),
                args: Vec::new(),
            }
        } else if let Some(id) = flatpak {
            McpLaunch {
                command: PathBuf::from("flatpak"),
                args: vec![
                    "run".to_string(),
                    "--command=rds-mcp".to_string(),
                    id.to_string(),
                ],
            }
        } else if let Some(file) = appimage {
            McpLaunch {
                command: file,
                args: vec!["mcp".to_string()],
            }
        } else {
            McpLaunch {
                command: beside,
                args: Vec::new(),
            }
        }
    }

    /// Shown to the user: the command and its arguments on one line.
    pub fn display(&self) -> String {
        let mut line = self.command.display().to_string();
        for a in &self.args {
            line.push(' ');
            line.push_str(a);
        }
        line
    }

    /// The entry for `claude_desktop_config.json` (or an equivalent MCP
    /// client configuration).
    pub fn client_snippet(&self) -> String {
        let command =
            serde_json::to_string(&self.command.to_string_lossy()).expect("a string serialises");
        let args = serde_json::to_string(&self.args).expect("strings serialise");
        format!(
            "{{\n  \"mcpServers\": {{\n    \"rust-dicom-station\": {{\n      \"command\": {command},\n      \"args\": {args}\n    }}\n  }}\n}}\n"
        )
    }
}

/// How an MCP client starts this installation's server (see
/// [`McpLaunch::resolve`]).
pub fn mcp_client_launch() -> McpLaunch {
    let linux_var = |name: &str| {
        if cfg!(target_os = "linux") {
            std::env::var_os(name).filter(|v| !v.is_empty())
        } else {
            None
        }
    };
    let flatpak = linux_var("FLATPAK_ID").map(|v| v.to_string_lossy().into_owned());
    let appimage = linux_var("APPIMAGE").map(PathBuf::from);
    McpLaunch::resolve(
        snap_env().as_ref(),
        flatpak.as_deref(),
        appimage,
        mcp_exe_path(),
    )
}

/// The entry an MCP client (Claude Desktop, Claude Code) needs in its
/// configuration to launch the server.
pub fn mcp_client_snippet() -> String {
    mcp_client_launch().client_snippet()
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
/// who will use the program - those files do not exist yet. They go into a
/// small file next to the executable instead, in the same `key = value`
/// syntax, and every key in it is only a *default*: the user's own settings
/// file is read afterwards and wins, and so does anything they change from
/// the menus.
///
/// `None` when the executable's own path cannot be determined, which is not
/// a condition worth reporting - it just means there are no defaults.
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
        } else if let Some(row) = VIEW_ROW_KEYS
            .iter()
            .position(|k| key.eq_ignore_ascii_case(k))
        {
            s.view_rows[row] = parse_view_row(value);
        } else if let Some(slot) = SESSION_KEYS
            .iter()
            .position(|k| key.eq_ignore_ascii_case(k))
        {
            s.session[slot] = value
                .split(SESSION_SEP)
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .collect();
        } else if key.eq_ignore_ascii_case(MODULE_SIM_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_simulation = b;
            }
        } else if key.eq_ignore_ascii_case(MODULE_PROP_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_propagation = b;
            }
        } else if key.eq_ignore_ascii_case(MODULE_TOOLS_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_structures = b;
            }
        } else if key.eq_ignore_ascii_case(MODULE_AUTO_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_auto = b;
            }
        } else if key.eq_ignore_ascii_case(MODULE_DOSE_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_dose = b;
            }
        } else if key.eq_ignore_ascii_case(MODULE_INFO_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_info = b;
            }
        } else if key.eq_ignore_ascii_case(MODULE_PLAY_KEY) {
            if let Some(b) = bool_from_str(value) {
                s.module_play = b;
            }
        } else if key.eq_ignore_ascii_case(MODULES_OPEN_KEY) {
            s.modules_open = value
                .split(',')
                .map(|v| v.trim().to_lowercase())
                .filter(|v| !v.is_empty())
                .collect();
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
        "# optional modules-panel sections (Modules menu) = on | off\n\
         {MODULE_REG_KEY} = {}\n\
         {MODULE_SIM_KEY} = {}\n\
         {MODULE_PROP_KEY} = {}\n\
         {MODULE_TOOLS_KEY} = {}\n\
         {MODULE_AUTO_KEY} = {}\n\
         {MODULE_DOSE_KEY} = {}\n\
         {MODULE_INFO_KEY} = {}\n\
         {MODULE_PLAY_KEY} = {}\n\
         # module sections unfolded at the last run, for Restore the last session\n\
         {MODULES_OPEN_KEY} = {}\n\
         # graphics backend = auto | vulkan | dx12 | metal | opengl\n\
         # (the WGPU_BACKEND environment variable overrides this)\n\
         {GRAPHICS_BACKEND_KEY} = {}\n",
        bool_to_str(s.module_registration),
        bool_to_str(s.module_simulation),
        bool_to_str(s.module_propagation),
        bool_to_str(s.module_structures),
        bool_to_str(s.module_auto),
        bool_to_str(s.module_dose),
        bool_to_str(s.module_info),
        bool_to_str(s.module_play),
        s.modules_open.join(","),
        s.graphics_backend.key()
    ));
    out.push_str("# what each row of the central area shows, left to right:\n");
    out.push_str("# up to three of axial, sagittal, coronal, 3d\n");
    for (key, row) in VIEW_ROW_KEYS.iter().zip(&s.view_rows) {
        out.push_str(&format!("{key} = {}\n", render_view_row(row)));
    }
    for (key, paths) in SESSION_KEYS.iter().zip(&s.session) {
        if paths.is_empty() {
            continue;
        }
        let joined: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        out.push_str(&format!(
            "# what this dataset was last loaded from\n{key} = {}\n",
            joined.join(&SESSION_SEP.to_string())
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap_vars(vars: &[(&str, &str)]) -> Option<SnapEnv> {
        let owned: Vec<(String, OsString)> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), OsString::from(v)))
            .collect();
        SnapEnv::from_vars(|k| {
            owned
                .iter()
                .find(|(name, _)| name == k)
                .map(|(_, v)| v.clone())
        })
    }

    const SNAP_VARS: [(&str, &str); 3] = [
        ("SNAP_INSTANCE_NAME", "rust-dicom-station"),
        ("SNAP", "/snap/rust-dicom-station/12"),
        ("SNAP_USER_COMMON", "/home/u/snap/rust-dicom-station/common"),
    ];

    #[test]
    fn a_snap_is_recognised_only_with_all_its_variables() {
        assert_eq!(snap_vars(&[]), None);
        for (missing, (name, _)) in SNAP_VARS.iter().enumerate() {
            let mut vars = SNAP_VARS.to_vec();
            vars.remove(missing);
            assert_eq!(snap_vars(&vars), None, "without {name}");
            // Set but empty counts as missing.
            vars.insert(missing, (name, ""));
            assert_eq!(snap_vars(&vars), None, "empty {name}");
        }
        // SNAP_NAME stands in for SNAP_INSTANCE_NAME (older snapd).
        let mut vars = SNAP_VARS.to_vec();
        vars[0].0 = "SNAP_NAME";
        assert_eq!(snap_vars(&vars).unwrap().instance, "rust-dicom-station");
    }

    #[test]
    fn inside_a_snap_config_and_data_are_shared_by_every_revision() {
        let snap = snap_vars(&SNAP_VARS).unwrap();
        let common = Path::new("/home/u/snap/rust-dicom-station/common");
        assert_eq!(snap.config_dir(), common.join("config"));
        assert_eq!(snap.data_dir(), common.join("data"));
        // Nothing lives under the revision's own folder or the read-only mount.
        for dir in [snap.config_dir(), snap.data_dir()] {
            assert!(!dir.starts_with(&snap.root), "{}", dir.display());
        }
    }

    #[test]
    fn snap_commands_are_named_the_way_snapd_names_them() {
        let snap = snap_vars(&SNAP_VARS).unwrap();
        assert_eq!(
            snap.command("rds-mcp"),
            Path::new("/snap/bin/rust-dicom-station.rds-mcp")
        );
        assert_eq!(
            snap.command("rust-dicom-station"),
            Path::new("/snap/bin/rust-dicom-station")
        );
        // A parallel installation carries its key in every command name.
        let mut vars = SNAP_VARS.to_vec();
        vars[0].1 = "rust-dicom-station_test";
        let snap = snap_vars(&vars).unwrap();
        assert_eq!(
            snap.command("rds-mcp"),
            Path::new("/snap/bin/rust-dicom-station_test.rds-mcp")
        );
        assert_eq!(
            snap.command("rust-dicom-station"),
            Path::new("/snap/bin/rust-dicom-station_test")
        );
    }

    #[test]
    fn an_mcp_client_runs_what_the_installation_offers() {
        let beside = PathBuf::from("/opt/rds/rds-mcp");
        let snap = snap_vars(&SNAP_VARS).unwrap();
        let appimage = PathBuf::from("/home/u/Apps/RDS.AppImage");

        let plain = McpLaunch::resolve(None, None, None, beside.clone());
        assert_eq!(plain.command, beside);
        assert!(plain.args.is_empty());

        let a = McpLaunch::resolve(None, None, Some(appimage.clone()), beside.clone());
        assert_eq!(a.command, appimage);
        assert_eq!(a.args, ["mcp"]);
        assert_eq!(a.display(), "/home/u/Apps/RDS.AppImage mcp");

        let f = McpLaunch::resolve(
            None,
            Some("io.github.alexprotom.rust-dicom-station"),
            None,
            beside.clone(),
        );
        assert_eq!(f.command, Path::new("flatpak"));
        assert_eq!(
            f.args,
            [
                "run",
                "--command=rds-mcp",
                "io.github.alexprotom.rust-dicom-station"
            ]
        );

        // A snap wins: another packaging's variable leaking into it would
        // be stale.
        let s = McpLaunch::resolve(
            Some(&snap),
            Some("io.github.alexprotom.rust-dicom-station"),
            Some(appimage),
            beside,
        );
        assert_eq!(s.command, Path::new("/snap/bin/rust-dicom-station.rds-mcp"));
        assert!(s.args.is_empty());

        for launch in [plain, a, f, s] {
            let v: serde_json::Value =
                serde_json::from_str(&launch.client_snippet()).expect("valid JSON");
            let entry = &v["mcpServers"]["rust-dicom-station"];
            assert_eq!(
                entry["command"].as_str(),
                Some(launch.command.to_string_lossy().as_ref())
            );
            let args: Vec<&str> = entry["args"]
                .as_array()
                .expect("an array")
                .iter()
                .map(|a| a.as_str().expect("a string"))
                .collect();
            assert_eq!(args, launch.args);
        }
    }

    #[test]
    fn the_client_snippet_escapes_a_windows_path() {
        let launch = McpLaunch {
            command: PathBuf::from(r"C:\Program Files\RDS\rds-mcp.exe"),
            args: Vec::new(),
        };
        let v: serde_json::Value =
            serde_json::from_str(&launch.client_snippet()).expect("valid JSON");
        assert_eq!(
            v["mcpServers"]["rust-dicom-station"]["command"],
            r"C:\Program Files\RDS\rds-mcp.exe"
        );
    }

    #[test]
    fn the_mcp_client_snippet_is_json_naming_the_executable() {
        let text = mcp_client_snippet();
        let v: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
        let cmd = v["mcpServers"]["rust-dicom-station"]["command"]
            .as_str()
            .expect("a command");
        assert!(cmd.contains("rds-mcp"), "{cmd}");
        assert!(v["mcpServers"]["rust-dicom-station"]["args"].is_array());
    }

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
    fn a_row_keeps_what_it_can_use_and_nothing_else() {
        use crate::volume::ViewPlane;
        let axial = PaneKind::Plane(ViewPlane::Axial);
        let cor = PaneKind::Plane(ViewPlane::Coronal);
        assert_eq!(parse_view_row("axial,coronal"), vec![axial, cor]);
        // Case and spacing are the user's business, not the parser's.
        assert_eq!(parse_view_row(" Axial , CORONAL "), vec![axial, cor]);
        // A repeat is not a pane, and a row takes no more than MAX_PANES.
        assert_eq!(
            parse_view_row("axial,axial,sagittal,coronal,3d"),
            vec![
                axial,
                PaneKind::Plane(ViewPlane::Sagittal),
                cor,
                PaneKind::Scene3d
            ]
        );
        assert_eq!(
            parse_view_row("axial,sagittal,coronal,3d").len(),
            MAX_PANES,
            "a row that names every kind keeps every kind"
        );
        // The order is the user's: this is left to right on screen.
        assert_eq!(parse_view_row("3d,axial"), vec![PaneKind::Scene3d, axial]);
        // Nothing usable leaves the default rather than an empty window.
        assert_eq!(parse_view_row(""), default_view_row());
        assert_eq!(parse_view_row("sideways, upside-down"), default_view_row());
    }

    #[test]
    fn round_trips_the_view_rows() {
        use crate::volume::ViewPlane;
        let s = Settings {
            view_rows: [
                vec![PaneKind::Plane(ViewPlane::Axial), PaneKind::Scene3d],
                vec![PaneKind::Plane(ViewPlane::Coronal)],
            ],
            ..Settings::default()
        };
        let back = parse(&render(&s));
        assert_eq!(back.view_rows, s.view_rows);
        // And the file says it in words a person can edit by hand.
        assert!(
            render(&s).contains("view_row_a = axial,3d"),
            "{}",
            render(&s)
        );
        assert!(render(&s).contains("view_row_b = coronal"));
        // A file that never mentions them gets the layout the program has
        // always had.
        assert_eq!(
            parse("").view_rows,
            [default_view_row(), default_view_row()]
        );
    }

    #[test]
    fn round_trips_the_last_session() {
        let s = Settings {
            session: [
                vec![
                    PathBuf::from("D:/studies/one"),
                    PathBuf::from("D:/studies/two"),
                ],
                vec![PathBuf::from("D:/studies/three")],
            ],
            ..Settings::default()
        };
        assert_eq!(parse(&render(&s)), s, "both datasets round trip");
        assert_eq!(
            parse("session_a = D:/one | D:/two |\n").session[0],
            vec![PathBuf::from("D:/one"), PathBuf::from("D:/two")],
            "spacing and a trailing separator are ignored"
        );
        assert!(
            parse("session_b =\n").session[1].is_empty(),
            "an empty list is no session, not one blank path"
        );
    }

    #[test]
    fn round_trips_the_module_flags() {
        for bits in 0..64u8 {
            let s = Settings {
                module_registration: bits & 1 != 0,
                module_simulation: bits & 2 != 0,
                module_propagation: bits & 4 != 0,
                module_structures: bits & 8 != 0,
                module_auto: bits & 16 != 0,
                module_dose: bits & 32 != 0,
                modules_open: if bits & 1 != 0 {
                    vec!["editor".into(), "dose".into()]
                } else {
                    Vec::new()
                },
                ..Settings::default()
            };
            assert_eq!(parse(&render(&s)), s, "round trip of {bits:06b}");
        }
        assert!(
            parse(&format!("{MODULE_REG_KEY} = TRUE")).module_registration,
            "case-insensitive alias"
        );
        assert!(
            parse(&format!("{MODULE_PROP_KEY} = on")).module_propagation,
            "the propagation module is remembered too"
        );
        assert!(
            parse("").module_structures
                && parse("").module_auto
                && !parse(&format!("{MODULE_TOOLS_KEY} = off")).module_structures
                && !parse(&format!("{MODULE_AUTO_KEY} = off")).module_auto,
            "the structures editor starts switched on and can be switched off"
        );
        assert!(
            parse("").module_info && !parse(&format!("{MODULE_INFO_KEY} = off")).module_info,
            "the image information module starts switched on and can be switched off"
        );
        assert!(
            parse("").module_play && !parse(&format!("{MODULE_PLAY_KEY} = off")).module_play,
            "the playback module starts switched on and can be switched off"
        );
        // Whatever was set survives a write and a read.
        let s = Settings {
            module_info: false,
            module_dose: false,
            module_play: false,
            ..Settings::default()
        };
        let back = parse(&render(&s));
        assert!(!back.module_info && !back.module_dose && !back.module_play);
    }
}
