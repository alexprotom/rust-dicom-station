//! The one door to a file dialog.
//!
//! On the desktop every "open a folder", "open a file" and "save as" goes
//! through the operating system's own dialog (`rfd`), which returns at once
//! with the answer. Android has no such dialog for a program without Java
//! code, so there the same request opens [`Browser`]: a folder browser drawn
//! in egui inside the main window, which takes as many frames as the user
//! needs. To let both work behind one call, a request carries *what to do
//! with the answer* as a closure: [`ViewerApp::ask`] runs it straight away
//! on the desktop and later on Android, and the calling code is the same.
//!
//! The browser model - roots, listing, sorting, selection - has no platform
//! code in it and is compiled and tested everywhere; only the choice of
//! roots (`/storage/...`) is Android's.

use std::path::{Path, PathBuf};

use egui::{Align2, Vec2};

use super::ViewerApp;

/// A file-type filter, as the desktop dialog shows it.
#[derive(Clone, Copy, Debug)]
pub(super) struct Filter {
    pub name: &'static str,
    pub exts: &'static [&'static str],
}

/// The comma-separated tables the tool windows write.
pub(super) const CSV_FILES: Filter = Filter {
    name: "CSV",
    exts: &["csv"],
};

/// A dose-constraint protocol (see `dvh::parse_protocol`).
pub(super) const PROTOCOL_FILES: Filter = Filter {
    name: "protocol",
    exts: &["txt", "csv", "protocol"],
};

/// What is being asked for.
#[derive(Clone, Debug)]
pub(super) enum Ask {
    /// One existing folder.
    Folder,
    /// One or more existing files (an empty choice counts as cancelled).
    Files,
    /// One existing file.
    File {
        dir: Option<PathBuf>,
        filter: Option<Filter>,
    },
    /// A file to write, with a proposed name.
    Save {
        name: String,
        dir: Option<PathBuf>,
        filter: Option<Filter>,
    },
}

impl Ask {
    fn wants_files(&self) -> bool {
        !matches!(self, Ask::Folder)
    }

    fn many(&self) -> bool {
        matches!(self, Ask::Files)
    }

    fn start_dir(&self) -> Option<&Path> {
        match self {
            Ask::File { dir, .. } | Ask::Save { dir, .. } => dir.as_deref(),
            _ => None,
        }
    }

    fn filter(&self) -> Option<Filter> {
        match self {
            Ask::File { filter, .. } | Ask::Save { filter, .. } => *filter,
            _ => None,
        }
    }
}

/// What happens with the answer: the paths chosen (one, or several for
/// [`Ask::Files`]). Never called when the dialog was cancelled.
pub(super) type Then = Box<dyn FnOnce(&mut ViewerApp, Vec<PathBuf>)>;

/// The dialog state the application owns: the browser while one is open,
/// and the folder the last one was left in, so the next opens there.
#[derive(Default)]
pub(super) struct Picker {
    open: Option<Browser>,
    last_dir: Option<PathBuf>,
}

impl ViewerApp {
    /// Ask for a folder or a file and run `then` with the answer.
    ///
    /// Desktop: the system dialog blocks, and `then` runs before this
    /// returns - exactly the `if let Some(path) = dialog()` it replaces.
    /// Android: the browser opens and `then` runs from a later frame, when
    /// the user has chosen; a second `ask` while one is open replaces it.
    /// An empty `title` leaves the dialog's default title alone.
    pub(super) fn ask(
        &mut self,
        title: &str,
        ask: Ask,
        then: impl FnOnce(&mut ViewerApp, Vec<PathBuf>) + 'static,
    ) {
        #[cfg(not(target_os = "android"))]
        {
            if let Some(paths) = desktop_dialog(title, &ask) {
                then(self, paths);
            }
        }
        #[cfg(target_os = "android")]
        {
            let start = ask
                .start_dir()
                .filter(|d| d.is_dir())
                .map(Path::to_path_buf)
                .or_else(|| self.picker.last_dir.clone());
            self.picker.open = Some(Browser::new(
                title,
                ask,
                Box::new(then),
                android_roots(),
                start,
            ));
        }
    }

    /// One existing folder.
    pub(super) fn ask_folder(
        &mut self,
        title: &str,
        then: impl FnOnce(&mut ViewerApp, PathBuf) + 'static,
    ) {
        self.ask(title, Ask::Folder, move |app, mut paths| {
            if let Some(p) = paths.pop() {
                then(app, p);
            }
        });
    }

    /// One or more existing files.
    pub(super) fn ask_files(
        &mut self,
        title: &str,
        then: impl FnOnce(&mut ViewerApp, Vec<PathBuf>) + 'static,
    ) {
        self.ask(title, Ask::Files, then);
    }

    /// One existing file, optionally starting in `dir`.
    pub(super) fn ask_file(
        &mut self,
        title: &str,
        dir: Option<PathBuf>,
        filter: Option<Filter>,
        then: impl FnOnce(&mut ViewerApp, PathBuf) + 'static,
    ) {
        self.ask(title, Ask::File { dir, filter }, move |app, mut paths| {
            if let Some(p) = paths.pop() {
                then(app, p);
            }
        });
    }

    /// A file to write, proposed as `name`, optionally starting in `dir`.
    pub(super) fn ask_save(
        &mut self,
        title: &str,
        name: impl Into<String>,
        dir: Option<PathBuf>,
        filter: Option<Filter>,
        then: impl FnOnce(&mut ViewerApp, PathBuf) + 'static,
    ) {
        let ask = Ask::Save {
            name: name.into(),
            dir,
            filter,
        };
        self.ask(title, ask, move |app, mut paths| {
            if let Some(p) = paths.pop() {
                then(app, p);
            }
        });
    }

    /// Draw the browser while one is open, and run its continuation when
    /// the user has answered. A no-op on the desktop, where nothing ever
    /// opens one.
    pub(super) fn picker_window(&mut self, ctx: &egui::Context) {
        let Some(browser) = self.picker.open.as_mut() else {
            return;
        };
        match browser.show(ctx) {
            Outcome::Pending => {}
            Outcome::Cancelled => {
                let b = self.picker.open.take().expect("checked above");
                self.picker.last_dir = Some(b.dir);
            }
            Outcome::Chosen(paths) => {
                let b = self.picker.open.take().expect("checked above");
                self.picker.last_dir = Some(b.dir);
                (b.then)(self, paths);
            }
        }
    }
}

/// The system dialog for one request.
#[cfg(not(target_os = "android"))]
fn desktop_dialog(title: &str, ask: &Ask) -> Option<Vec<PathBuf>> {
    let mut d = rfd::FileDialog::new();
    if !title.is_empty() {
        d = d.set_title(title);
    }
    if let Some(dir) = ask.start_dir() {
        d = d.set_directory(dir);
    }
    if let Some(f) = ask.filter() {
        d = d.add_filter(f.name, f.exts);
    }
    match ask {
        Ask::Folder => d.pick_folder().map(|p| vec![p]),
        Ask::Files => d.pick_files().filter(|v| !v.is_empty()),
        Ask::File { .. } => d.pick_file().map(|p| vec![p]),
        Ask::Save { name, .. } => d.set_file_name(name).save_file().map(|p| vec![p]),
    }
}

// ---------------------------------------------------------------------------
// The browser

/// A place the browser can start from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Root {
    pub label: String,
    pub path: PathBuf,
}

/// One line of a listing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub name: String,
    pub path: PathBuf,
    pub is_dir: bool,
    pub size: u64,
}

/// The result of one frame of the browser.
pub(super) enum Outcome {
    Pending,
    Cancelled,
    Chosen(Vec<PathBuf>),
}

/// The folder browser: one folder at a time, its entries, and the choice
/// being made in it.
pub(super) struct Browser {
    title: String,
    ask: Ask,
    then: Then,
    roots: Vec<Root>,
    /// The folder shown.
    dir: PathBuf,
    /// Its listing, or why there is none.
    entries: Result<Vec<Entry>, String>,
    /// Chosen files (paths), for the file requests.
    selected: Vec<PathBuf>,
    /// The name typed, for [`Ask::Save`].
    name: String,
    /// Set after the first *Save* over an existing file: the second replaces it.
    confirm_replace: bool,
}

impl Browser {
    /// Only the Android `ask` opens one; the desktop compiles it for the
    /// tests below.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub(super) fn new(
        title: &str,
        ask: Ask,
        then: Then,
        roots: Vec<Root>,
        start: Option<PathBuf>,
    ) -> Self {
        let name = match &ask {
            Ask::Save { name, .. } => name.clone(),
            _ => String::new(),
        };
        let dir = start
            .filter(|d| d.is_dir())
            .or_else(|| {
                ask.start_dir()
                    .filter(|d| d.is_dir())
                    .map(Path::to_path_buf)
            })
            .or_else(|| roots.iter().map(|r| r.path.clone()).find(|p| p.is_dir()))
            .unwrap_or_else(|| PathBuf::from("/"));
        let mut b = Self {
            title: if title.is_empty() {
                match &ask {
                    Ask::Folder => "Select a folder".into(),
                    Ask::Files | Ask::File { .. } => "Open".into(),
                    Ask::Save { .. } => "Save".into(),
                }
            } else {
                title.to_owned()
            },
            ask,
            then,
            roots,
            dir,
            entries: Ok(Vec::new()),
            selected: Vec::new(),
            name,
            confirm_replace: false,
        };
        b.refresh();
        b
    }

    /// Go to `dir` and list it.
    fn enter(&mut self, dir: PathBuf) {
        self.dir = dir;
        self.selected.clear();
        self.confirm_replace = false;
        self.refresh();
    }

    fn refresh(&mut self) {
        self.entries = list_dir(&self.dir, self.ask.filter());
    }

    /// The button that confirms, if the choice made so far is a valid one.
    fn ready(&self) -> bool {
        match &self.ask {
            Ask::Folder => self.entries.is_ok(),
            Ask::Files | Ask::File { .. } => !self.selected.is_empty(),
            Ask::Save { .. } => !self.name.trim().is_empty() && self.entries.is_ok(),
        }
    }

    fn confirm_label(&self) -> String {
        match &self.ask {
            Ask::Folder => "Select this folder".into(),
            Ask::Files => match self.selected.len() {
                0 | 1 => "Open".into(),
                n => format!("Open {n} files"),
            },
            Ask::File { .. } => "Open".into(),
            Ask::Save { .. } => "Save".into(),
        }
    }

    /// The paths the confirm button stands for.
    fn answer(&self) -> Vec<PathBuf> {
        match &self.ask {
            Ask::Folder => vec![self.dir.clone()],
            Ask::Files | Ask::File { .. } => self.selected.clone(),
            Ask::Save { .. } => vec![self.dir.join(save_name(&self.name, self.ask.filter()))],
        }
    }

    /// Draw one frame.
    pub(super) fn show(&mut self, ctx: &egui::Context) -> Outcome {
        let screen = ctx.content_rect();
        let size = Vec2::new(
            (screen.width() * 0.9).min(820.0),
            (screen.height() * 0.85).min(640.0),
        );
        let mut outcome = Outcome::Pending;
        let mut go: Option<PathBuf> = None;
        egui::Window::new(&self.title)
            .id(egui::Id::new("file_browser"))
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .fixed_size(size)
            .show(ctx, |ui| {
                // -- where: roots and the path ------------------------------
                ui.horizontal_wrapped(|ui| {
                    for r in &self.roots {
                        let here = self.dir.starts_with(&r.path);
                        if ui.selectable_label(here, &r.label).clicked() {
                            go = Some(r.path.clone());
                        }
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(self.dir.parent().is_some(), egui::Button::new("⬆ Up"))
                        .clicked()
                    {
                        if let Some(p) = self.dir.parent() {
                            go = Some(p.to_path_buf());
                        }
                    }
                    if ui
                        .button("⟳")
                        .on_hover_text("List the folder again")
                        .clicked()
                    {
                        self.refresh();
                    }
                    // Breadcrumbs: every ancestor is a button.
                    let mut acc = PathBuf::new();
                    for c in self.dir.components() {
                        acc.push(c.as_os_str());
                        let label = match c {
                            std::path::Component::RootDir => "/".to_owned(),
                            other => other.as_os_str().to_string_lossy().into_owned(),
                        };
                        if ui.small_button(label).clicked() {
                            go = Some(acc.clone());
                        }
                    }
                });
                ui.separator();

                // -- the listing ---------------------------------------------
                let list_h = size.y - 150.0;
                egui::ScrollArea::vertical()
                    .max_height(list_h)
                    .min_scrolled_height(list_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| match &self.entries {
                        Err(e) => {
                            ui.add_space(8.0);
                            ui.colored_label(ui.visuals().warn_fg_color, e);
                            if e.contains("denied") {
                                ui.label(
                                    "The program can only browse here once it has been allowed \
                                     to manage all files: Android Settings > Apps > Rust DICOM \
                                     Station > All files access.",
                                );
                            }
                        }
                        Ok(entries) if entries.is_empty() => {
                            ui.add_space(8.0);
                            ui.weak("Empty folder");
                        }
                        Ok(entries) => {
                            for e in entries {
                                if e.is_dir {
                                    let r = ui.add(
                                        egui::Button::new(format!("📁 {}", e.name))
                                            .frame(false)
                                            .wrap(),
                                    );
                                    if r.clicked() {
                                        go = Some(e.path.clone());
                                    }
                                } else if self.ask.wants_files() {
                                    let on = self.selected.iter().any(|p| p == &e.path);
                                    let r = ui.selectable_label(
                                        on,
                                        format!("📄 {}   {}", e.name, human_size(e.size)),
                                    );
                                    if r.clicked() {
                                        if self.ask.many() {
                                            if on {
                                                self.selected.retain(|p| p != &e.path);
                                            } else {
                                                self.selected.push(e.path.clone());
                                            }
                                        } else {
                                            self.selected = vec![e.path.clone()];
                                            if let Ask::Save { .. } = self.ask {
                                                self.name = e.name.clone();
                                                self.confirm_replace = false;
                                            }
                                        }
                                    }
                                    if r.double_clicked() && !self.ask.many() {
                                        if let Ask::File { .. } = self.ask {
                                            outcome = Outcome::Chosen(vec![e.path.clone()]);
                                        }
                                    }
                                } else {
                                    ui.add_enabled(
                                        false,
                                        egui::Label::new(format!(
                                            "📄 {}   {}",
                                            e.name,
                                            human_size(e.size)
                                        )),
                                    );
                                }
                            }
                        }
                    });
                ui.separator();

                // -- the choice ------------------------------------------------
                if let Some(f) = self.ask.filter() {
                    let exts: Vec<String> = f.exts.iter().map(|e| format!("*.{e}")).collect();
                    ui.weak(format!("{} files ({})", f.name, exts.join(", ")));
                }
                if let Ask::Save { .. } = self.ask {
                    ui.horizontal(|ui| {
                        ui.label("File name");
                        let r = ui.add(
                            egui::TextEdit::singleline(&mut self.name).desired_width(f32::INFINITY),
                        );
                        if r.changed() {
                            self.confirm_replace = false;
                        }
                    });
                    if self.confirm_replace {
                        ui.colored_label(
                            ui.visuals().warn_fg_color,
                            format!(
                                "{} exists here. Save again to replace it.",
                                save_name(&self.name, self.ask.filter())
                            ),
                        );
                    }
                } else if self.ask.many() && !self.selected.is_empty() {
                    ui.weak(format!("{} file(s) selected", self.selected.len()));
                } else if let Ask::Folder = self.ask {
                    ui.weak(self.dir.display().to_string());
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(self.ready(), egui::Button::new(self.confirm_label()))
                        .clicked()
                    {
                        let paths = self.answer();
                        let replaces = matches!(self.ask, Ask::Save { .. })
                            && paths.first().map(|p| p.exists()).unwrap_or(false);
                        if replaces && !self.confirm_replace {
                            self.confirm_replace = true;
                        } else {
                            outcome = Outcome::Chosen(paths);
                        }
                    }
                    if ui.button("Cancel").clicked() {
                        outcome = Outcome::Cancelled;
                    }
                });
            });
        if let Some(dir) = go {
            self.enter(dir);
        }
        outcome
    }
}

/// The name a save gets: what was typed, plus the filter's first extension
/// when the name has none and there is a filter to take it from.
fn save_name(typed: &str, filter: Option<Filter>) -> String {
    let name = typed.trim();
    match filter {
        Some(f) if !name.contains('.') => match f.exts.first() {
            Some(ext) => format!("{name}.{ext}"),
            None => name.to_owned(),
        },
        _ => name.to_owned(),
    }
}

/// List `dir`: folders first, then files, each sorted without regard to
/// case; hidden (dot) entries left out; files not matching `filter` left
/// out too.
pub(super) fn list_dir(dir: &Path, filter: Option<Filter>) -> Result<Vec<Entry>, String> {
    let rd =
        std::fs::read_dir(dir).map_err(|e| format!("Could not list {}: {e}", dir.display()))?;
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let is_dir = meta.is_dir();
        if !is_dir {
            if let Some(f) = filter {
                let ok = f.exts.iter().any(|ext| {
                    name.rsplit_once('.')
                        .map(|(_, e)| e.eq_ignore_ascii_case(ext))
                        .unwrap_or(false)
                });
                if !ok {
                    continue;
                }
            }
        }
        out.push(Entry {
            path: entry.path(),
            name,
            is_dir,
            size: if is_dir { 0 } else { meta.len() },
        });
    }
    out.sort_by(|a, b| {
        b.is_dir
            .cmp(&a.is_dir)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(out)
}

fn human_size(n: u64) -> String {
    const K: f64 = 1024.0;
    let f = n as f64;
    if f < K {
        format!("{n} B")
    } else if f < K * K {
        format!("{:.0} kB", f / K)
    } else if f < K * K * K {
        format!("{:.1} MB", f / (K * K))
    } else {
        format!("{:.2} GB", f / (K * K * K))
    }
}

/// The roots worth offering on Android: the shared storage, every other
/// mounted volume (SD card, USB stick), and the program's own data folder.
/// `storage` is `/storage`; `data` is [`crate::settings::data_dir`]. A root
/// that does not exist is not offered.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub(super) fn roots_from(storage: &Path, data: &Path) -> Vec<Root> {
    let mut roots = Vec::new();
    let shared = storage.join("emulated").join("0");
    if shared.is_dir() {
        roots.push(Root {
            label: "Internal storage".into(),
            path: shared.clone(),
        });
        let dl = shared.join("Download");
        if dl.is_dir() {
            roots.push(Root {
                label: "Downloads".into(),
                path: dl,
            });
        }
    }
    if let Ok(rd) = std::fs::read_dir(storage) {
        let mut others: Vec<PathBuf> = rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                let name = p
                    .file_name()
                    .map(|n| n.to_string_lossy())
                    .unwrap_or_default();
                p.is_dir() && name != "emulated" && name != "self" && !name.starts_with('.')
            })
            .collect();
        others.sort();
        for p in others {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            roots.push(Root {
                label: format!("Volume {name}"),
                path: p,
            });
        }
    }
    if data.is_dir() {
        roots.push(Root {
            label: "App data".into(),
            path: data.to_path_buf(),
        });
    }
    roots
}

#[cfg(target_os = "android")]
fn android_roots() -> Vec<Root> {
    roots_from(Path::new("/storage"), &crate::settings::data_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rds_pick_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn listing_puts_folders_first_hides_dot_entries_and_filters_files() {
        let d = tmp("list");
        std::fs::create_dir(d.join("zeta")).unwrap();
        std::fs::create_dir(d.join("Alpha")).unwrap();
        std::fs::create_dir(d.join(".hidden")).unwrap();
        std::fs::write(d.join("b.axis"), "1").unwrap();
        std::fs::write(d.join("A.AXIS"), "22").unwrap();
        std::fs::write(d.join("notes.txt"), "333").unwrap();
        let all = list_dir(&d, None).unwrap();
        let names: Vec<&str> = all.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["Alpha", "zeta", "A.AXIS", "b.axis", "notes.txt"]);
        assert!(all[0].is_dir && !all[2].is_dir);
        assert_eq!(all[4].size, 3);
        let axes = list_dir(
            &d,
            Some(Filter {
                name: "Axis",
                exts: &["axis"],
            }),
        )
        .unwrap();
        let names: Vec<&str> = axes.iter().map(|e| e.name.as_str()).collect();
        // Folders stay, whatever the filter; the match ignores case.
        assert_eq!(names, ["Alpha", "zeta", "A.AXIS", "b.axis"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_missing_folder_is_an_error_not_a_panic() {
        let e = list_dir(Path::new("/definitely/not/here"), None).unwrap_err();
        assert!(e.contains("Could not list"));
    }

    #[test]
    fn the_typed_name_takes_the_filter_extension_only_when_it_has_none() {
        let f = Some(Filter {
            name: "CSV",
            exts: &["csv"],
        });
        assert_eq!(save_name("table", f), "table.csv");
        assert_eq!(save_name(" table.txt ", f), "table.txt");
        assert_eq!(save_name("table", None), "table");
    }

    #[test]
    fn roots_offer_what_exists() {
        let d = tmp("roots");
        let storage = d.join("storage");
        std::fs::create_dir_all(storage.join("emulated").join("0").join("Download")).unwrap();
        std::fs::create_dir_all(storage.join("self")).unwrap();
        std::fs::create_dir_all(storage.join("1234-ABCD")).unwrap();
        let data = d.join("data");
        std::fs::create_dir_all(&data).unwrap();
        let roots = roots_from(&storage, &data);
        let labels: Vec<&str> = roots.iter().map(|r| r.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "Internal storage",
                "Downloads",
                "Volume 1234-ABCD",
                "App data"
            ]
        );
        assert_eq!(roots[2].path, storage.join("1234-ABCD"));
        // Nothing there at all: nothing offered, and no panic.
        assert!(roots_from(&d.join("nope"), &d.join("nope")).is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_browser_starts_in_the_first_existing_root_and_confirms_the_folder() {
        let d = tmp("browser");
        let roots = vec![
            Root {
                label: "gone".into(),
                path: d.join("gone"),
            },
            Root {
                label: "here".into(),
                path: d.clone(),
            },
        ];
        let b = Browser::new("t", Ask::Folder, Box::new(|_, _| {}), roots, None);
        assert_eq!(b.dir, d);
        assert!(b.ready());
        assert_eq!(b.answer(), vec![d.clone()]);
        assert_eq!(b.confirm_label(), "Select this folder");
        let s = Browser::new(
            "",
            Ask::Save {
                name: "x.csv".into(),
                dir: Some(d.clone()),
                filter: None,
            },
            Box::new(|_, _| {}),
            Vec::new(),
            None,
        );
        assert_eq!(s.title, "Save");
        assert_eq!(s.answer(), vec![d.join("x.csv")]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn sizes_read_like_a_file_manager() {
        assert_eq!(human_size(12), "12 B");
        assert_eq!(human_size(2048), "2 kB");
        assert_eq!(human_size(3 * 1024 * 1024 + 512 * 1024), "3.5 MB");
        assert_eq!(human_size(5 * 1024 * 1024 * 1024), "5.00 GB");
    }
}
