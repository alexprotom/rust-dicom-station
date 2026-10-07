//! *Tools ▶ Downloaded models*: one window over every model the engines
//! can fetch.
//!
//! The engines each download their own weights on first use, which is
//! convenient right up to the moment somebody asks what is actually on this
//! machine, how much disk it costs, or wants a checkpoint re-fetched after a
//! bad download. This window answers all three from the shared inventory in
//! [`crate::models`]: a row per model with its state and size, per-row
//! **Download / Update / Remove**, and the same three actions over
//! everything at once. Preparing a model runs the engine's *own* first-use
//! path, so a model fetched here is bit for bit the one a run would have
//! fetched - there is no second download route to keep in step.

use anyhow::Context;

use super::*;
use crate::models::{AssetStatus, ModelAsset};

/// A row's (or the header's) deferred button press.
///
/// Buttons are collected while the table is drawn and applied afterwards:
/// the table borrows the scan, the actions replace it.
enum ModelAction {
    /// Fetch what is missing of one model.
    Fetch(usize),
    /// Remove one model and fetch it again.
    Update(usize),
    Remove(usize),
    /// Drop one model's source checkpoint, keeping it runnable.
    FreeSpare(usize),
    /// Fetch every model that is not ready.
    FetchMissing,
    /// Fetch every openly licensed model that is not ready.
    FetchMissingOpen,
    /// Re-fetch every model that is ready.
    UpdateInstalled,
    /// Drop every source checkpoint.
    FreeAllSpare,
    /// Keep the licence number typed in the field.
    SaveLicence,
    /// Add an nnU-Net model folder (asks for it).
    AddFolder,
    /// Forget one of the added folders.
    RemoveFolder(usize),
}

impl ViewerApp {
    pub(super) fn open_models_window(&mut self) {
        self.models_open = true;
        self.models_result = None;
        self.models_scan.clear();
    }

    /// The inventory with each model's state, re-read at most twice a second
    /// (the window is a table of file sizes, not a file-system watcher).
    fn model_scan(&mut self, now: f64, force: bool) {
        if !force && !self.models_scan.is_empty() && now - self.models_scan_at < 0.5 {
            return;
        }
        let root = models::root_from_setting(&self.models_dir);
        self.models_scan = models::inventory_with_custom()
            .into_iter()
            .map(|a| {
                let s = models::status(&a, &root);
                (a, s)
            })
            .collect();
        self.models_scan_at = now;
    }

    /// Prepare every listed model on a worker thread, removing what is on
    /// disk first when `refresh` (an update). Each model gets its own slice
    /// of the progress bar.
    pub(super) fn start_model_fetch(&mut self, assets: Vec<ModelAsset>, refresh: bool) {
        if self.models_job.is_some() || assets.is_empty() {
            return;
        }
        let root = models::root_from_setting(&self.models_dir);
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        self.models_result = None;
        self.models_job = Some(Job::spawn(progress, move |p| {
            let n = assets.len();
            let mut done = 0usize;
            for (i, a) in assets.iter().enumerate() {
                if p.cancelled() {
                    break;
                }
                p.set_phase(i as f32 / n as f32, 1.0 / n as f32);
                p.set(format!("{} - {} of {n}", a.label, i + 1));
                if refresh {
                    models::remove(a, &root).with_context(|| format!("remove {}", a.label))?;
                }
                models::ensure(a, &root, p).with_context(|| format!("prepare {}", a.label))?;
                done += 1;
            }
            p.set_phase(0.0, 1.0);
            let verb = if refresh { "updated" } else { "ready" };
            Ok(if done == n {
                format!("✔ {n} model(s) {verb}")
            } else {
                format!("{done} of {n} model(s) {verb} - cancelled")
            })
        }));
    }

    /// Poll the fetch batch; called from the frame loop beside the others.
    pub(super) fn poll_models_job(&mut self, ctx: &egui::Context) {
        match poll_job(&mut self.models_job, ctx, "Model download", &mut self.error) {
            Some(Ok(msg)) => {
                self.models_result = Some(msg);
                self.models_scan.clear();
            }
            Some(Err(e)) => {
                if progress::is_cancellation(&e) {
                    self.models_result = Some("Cancelled.".to_string());
                } else {
                    self.error = Some(format!("Model download failed: {e:#}"));
                }
                self.models_scan.clear();
            }
            None => {}
        }
    }

    pub(super) fn models_window(&mut self, ctx: &egui::Context) {
        if !self.models_open {
            return;
        }
        let now = ctx.input(|i| i.time);
        self.model_scan(now, false);

        let running = self.models_job.is_some();
        let mut open = true;
        let mut close = false;
        let mut browse = false;
        let mut cancel = false;
        let mut action: Option<ModelAction> = None;

        let root = models::root_from_setting(&self.models_dir);
        let scan = std::mem::take(&mut self.models_scan);
        let total: u64 = scan.iter().map(|(_, s)| s.bytes).sum();
        let ready_n = scan.iter().filter(|(_, s)| s.ready).count();
        let fetchable = |a: &ModelAsset| {
            !a.licence.needs_licence_number() || !self.ts_licence.trim().is_empty()
        };
        let missing: u64 = scan
            .iter()
            .filter(|(a, s)| !s.ready && fetchable(a))
            .map(|(a, _)| a.download_bytes)
            .sum();
        let spare: u64 = scan.iter().map(|(_, s)| s.spare_bytes).sum();
        let missing_open: u64 = scan
            .iter()
            .filter(|(a, s)| !s.ready && a.licence.open())
            .map(|(a, _)| a.download_bytes)
            .sum();
        let has_licence = !self.ts_licence.trim().is_empty();

        detach::tool_window(
            ctx,
            "models",
            "📦 Downloaded models",
            &mut open,
            detach::WinOpts::width(680.0),
            |ui| {
                ui.label(
                    "Every model the segmentation tools can fetch. Weights are \
                     downloaded once, converted to a cache beside them, and never touched \
                     again - this window is where that inventory is managed.",
                );
                ui.separator();

                browse = models_root_row(ui, &mut self.models_dir);
                ui.weak(format!("Root: {}", root.display()));
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{ready_n} of {} model(s) ready · {} on disk",
                            scan.len(),
                            models::human_bytes(total)
                        ))
                        .strong(),
                    );
                });

                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(
                            !running && missing > 0,
                            egui::Button::new(format!(
                                "⬇ Download all missing ({})",
                                models::human_bytes(missing)
                            )),
                        )
                        .on_hover_text(
                            "Fetch and convert every model that is not ready yet, one after \
                             the other",
                        )
                        .clicked()
                    {
                        action = Some(ModelAction::FetchMissing);
                    }
                    if ui
                        .add_enabled(
                            !running && missing_open > 0 && missing_open < missing,
                            egui::Button::new(format!(
                                "⬇ Download open-licence only ({})",
                                models::human_bytes(missing_open)
                            )),
                        )
                        .on_hover_text(
                            "Fetch only the missing models whose weights are Apache-2.0 or \
                             MIT - the ones that may be used for anything, commercial work \
                             included",
                        )
                        .clicked()
                    {
                        action = Some(ModelAction::FetchMissingOpen);
                    }
                    if enabled_tip_button(
                        ui,
                        !running && ready_n > 0,
                        "⟳ Update all",
                        "Remove and re-fetch every model that is on disk - the published \
                         files carry no version, so an update is a fresh download",
                    ) {
                        action = Some(ModelAction::UpdateInstalled);
                    }
                    if ui
                        .add_enabled(
                            !running && spare > 0,
                            egui::Button::new(format!("♻ Free {}", models::human_bytes(spare))),
                        )
                        .on_hover_text(
                            "Delete the source checkpoints the converted caches were made \
                             from. The models keep running; only a future re-conversion \
                             would download them again",
                        )
                        .clicked()
                    {
                        action = Some(ModelAction::FreeAllSpare);
                    }
                });

                if let Some(job) = &self.models_job {
                    ui.separator();
                    cancel = progress_row(ui, &job.progress);
                }
                if let Some(msg) = &self.models_result {
                    ui.weak(msg);
                }

                ui.separator();
                licence_row(
                    ui,
                    &mut self.ts_licence,
                    &mut self.ts_licence_shown,
                    &mut action,
                );
                folders_rows(
                    ui,
                    &self.nnunet_folders,
                    &self.nnunet_folder_errors,
                    running,
                    &mut action,
                );

                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(420.0)
                    .show(ui, |ui| {
                        for engine in models::Engine::ALL {
                            if !scan.iter().any(|(a, _)| a.engine == engine) {
                                continue;
                            }
                            let tool = tool_of(engine);
                            ui.add_space(2.0);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} {} - {}/",
                                    tool.glyph,
                                    tool.name,
                                    engine.subdir()
                                ))
                                .strong(),
                            );
                            let (note, warn) = weights_licence(engine);
                            licence_line(ui, note, warn);
                            egui::Grid::new(("models_grid", engine.subdir()))
                                .num_columns(6)
                                .spacing([10.0, 4.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    let mut group = "";
                                    for (i, (a, s)) in scan.iter().enumerate() {
                                        if a.engine != engine {
                                            continue;
                                        }
                                        // A sub-heading where the rows of
                                        // one engine change group.
                                        if a.group != group && !a.group.is_empty() {
                                            group = a.group;
                                            ui.label("");
                                            ui.label(egui::RichText::new(group).italics().weak());
                                            ui.end_row();
                                        }
                                        model_row(ui, i, a, s, running, has_licence, &mut action);
                                        ui.end_row();
                                    }
                                });
                            ui.add_space(4.0);
                        }
                    });

                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                    ui.weak(RESEARCH_NOTE);
                });
            },
        );

        self.models_scan = scan;

        if browse {
            self.ask_folder("Model folder", |app, dir| {
                app.models_dir = dir.display().to_string();
                app.persist_settings();
                app.models_scan.clear();
            });
        }
        cancel_if(cancel, &self.models_job);
        if let Some(act) = action {
            self.apply_model_action(act, &root);
        }
        if !open || close {
            self.models_open = false;
            self.persist_settings();
        }
    }

    /// Register the model folders again, keep the list, and say what was
    /// refused.
    fn apply_model_folders(&mut self) {
        let regs = autoseg::custom::register(&self.nnunet_folders);
        self.nnunet_folder_errors = regs
            .into_iter()
            .filter_map(|(p, r)| r.err().map(|e| (p, format!("{e:#}"))))
            .collect();
        self.persist_settings();
        self.models_scan.clear();
    }

    fn apply_model_action(&mut self, act: ModelAction, root: &std::path::Path) {
        let assets: Vec<ModelAsset> = self.models_scan.iter().map(|(a, _)| a.clone()).collect();
        let states: Vec<AssetStatus> = self.models_scan.iter().map(|(_, s)| *s).collect();
        let mut freed = 0u64;
        let mut removed = 0usize;
        match act {
            ModelAction::Fetch(i) => {
                self.start_model_fetch(vec![assets[i].clone()], false);
            }
            ModelAction::Update(i) => {
                self.start_model_fetch(vec![assets[i].clone()], true);
            }
            ModelAction::Remove(i) => match models::remove(&assets[i], root) {
                Ok(n) => {
                    freed = n;
                    removed = 1;
                }
                Err(e) => self.error = Some(format!("Removing the model failed: {e:#}")),
            },
            ModelAction::FreeSpare(i) => match models::free_spare(&assets[i], root) {
                Ok(n) => freed = n,
                Err(e) => self.error = Some(format!("Freeing the checkpoint failed: {e:#}")),
            },
            ModelAction::FetchMissing => {
                let licensed = !self.ts_licence.trim().is_empty();
                let want: Vec<ModelAsset> = assets
                    .iter()
                    .zip(&states)
                    .filter(|(a, s)| !s.ready && (licensed || !a.licence.needs_licence_number()))
                    .map(|(a, _)| a.clone())
                    .collect();
                self.start_model_fetch(want, false);
            }
            ModelAction::SaveLicence => {
                self.persist_settings();
                autoseg::weights::set_ts_licence(Some(self.ts_licence.as_str()));
                self.models_result = Some(if self.ts_licence.trim().is_empty() {
                    "Licence number cleared.".to_string()
                } else {
                    "Licence number kept in your settings.".to_string()
                });
            }
            ModelAction::AddFolder => {
                self.ask_folder("nnU-Net model folder", |app, dir| {
                    if !app.nnunet_folders.contains(&dir) {
                        app.nnunet_folders.push(dir);
                    }
                    app.apply_model_folders();
                });
            }
            ModelAction::RemoveFolder(i) => {
                if i < self.nnunet_folders.len() {
                    self.nnunet_folders.remove(i);
                    self.apply_model_folders();
                }
            }
            ModelAction::FetchMissingOpen => {
                let want: Vec<ModelAsset> = assets
                    .iter()
                    .zip(&states)
                    .filter(|(a, s)| !s.ready && a.licence.open())
                    .map(|(a, _)| a.clone())
                    .collect();
                self.start_model_fetch(want, false);
            }
            ModelAction::UpdateInstalled => {
                let want: Vec<ModelAsset> = assets
                    .iter()
                    .zip(&states)
                    .filter(|(_, s)| s.ready)
                    .map(|(a, _)| a.clone())
                    .collect();
                self.start_model_fetch(want, true);
            }
            ModelAction::FreeAllSpare => {
                for a in &assets {
                    match models::free_spare(a, root) {
                        Ok(n) => freed += n,
                        Err(e) => self.error = Some(format!("Freeing failed: {e:#}")),
                    }
                }
            }
        }
        if freed > 0 || removed > 0 {
            self.models_result = Some(format!(
                "{} freed{}",
                models::human_bytes(freed),
                if removed > 0 { " (model removed)" } else { "" }
            ));
        }
        self.models_scan.clear();
    }
}

/// One inventory row: state, name, sizes, buttons.
fn model_row(
    ui: &mut egui::Ui,
    i: usize,
    a: &ModelAsset,
    s: &AssetStatus,
    running: bool,
    has_licence: bool,
    action: &mut Option<ModelAction>,
) {
    let local = a.engine == models::Engine::Custom;
    let blocked = a.licence.needs_licence_number() && !has_licence;
    let (glyph, tint) = if s.ready {
        ("✔", Color32::from_rgb(120, 200, 120))
    } else if s.partial {
        ("◐", Color32::from_rgb(230, 190, 90))
    } else {
        ("-", Color32::GRAY)
    };
    ui.label(egui::RichText::new(glyph).color(tint).monospace())
        .on_hover_text(if s.ready {
            "Ready - runs with no network access"
        } else if s.partial {
            "Partly downloaded - a run (or Download) finishes it"
        } else {
            "Not downloaded"
        });
    ui.label(&a.label).on_hover_text(a.detail);
    ui.label(
        egui::RichText::new(a.modality.map_or("any", |m| m.label()))
            .weak()
            .small(),
    )
    .on_hover_text("What the model segments");
    let lic = egui::RichText::new(a.licence.name()).small();
    ui.label(if a.licence.open() {
        lic.weak()
    } else {
        lic.color(warn_color(ui.visuals()))
    })
    .on_hover_text(if a.licence.needs_licence_number() {
        "Downloaded with your TotalSegmentator licence number (free for non-commercial \
         use; commercial use needs a paid licence)"
    } else if a.licence.research_only() {
        "Non-commercial licence: research use only"
    } else if a.licence.open() {
        "Open licence: any use, commercial work included"
    } else {
        "Read the licence before any use beyond research"
    });
    ui.label(
        egui::RichText::new(if s.ready {
            models::human_bytes(s.bytes)
        } else if local {
            "converted on first use".to_string()
        } else if s.partial {
            format!(
                "{} of ≈{}",
                models::human_bytes(s.bytes),
                models::human_bytes(a.download_bytes)
            )
        } else {
            format!("≈{} to fetch", models::human_bytes(a.download_bytes))
        })
        .weak()
        .monospace(),
    );
    ui.horizontal(|ui| {
        if !s.ready
            && ui
                .add_enabled(!running && !blocked, egui::Button::new("⬇").small())
                .on_hover_text(if local {
                    "Convert this model folder's weights"
                } else {
                    "Download and convert this model"
                })
                .on_disabled_hover_text("Enter your TotalSegmentator licence number above first")
                .clicked()
        {
            *action = Some(ModelAction::Fetch(i));
        }
        if s.ready
            && ui
                .add_enabled(!running && !blocked, egui::Button::new("⟳").small())
                .on_hover_text("Remove and fetch this model again")
                .clicked()
        {
            *action = Some(ModelAction::Update(i));
        }
        if s.spare_bytes > 0
            && ui
                .add_enabled(!running, egui::Button::new("♻").small())
                .on_hover_text(format!(
                    "Delete the {} source checkpoint; the model keeps running",
                    models::human_bytes(s.spare_bytes)
                ))
                .clicked()
        {
            *action = Some(ModelAction::FreeSpare(i));
        }
        if (s.ready || s.partial)
            && ui
                .add_enabled(!running, egui::Button::new("🗑").small())
                .on_hover_text(if local {
                    "Delete the converted copy; your model folder is not touched"
                } else {
                    "Delete every file of this model"
                })
                .clicked()
        {
            *action = Some(ModelAction::Remove(i));
        }
    });
}

/// The TotalSegmentator licence number: a masked field, a show toggle and a
/// button that keeps it in the user's settings file.
fn licence_row(
    ui: &mut egui::Ui,
    text: &mut String,
    shown: &mut bool,
    action: &mut Option<ModelAction>,
) {
    ui.horizontal(|ui| {
        ui.label("TotalSegmentator licence number:");
        ui.add(
            egui::TextEdit::singleline(text)
                .password(!*shown)
                .desired_width(180.0)
                .hint_text("for the licensed models"),
        );
        ui.checkbox(shown, "Show");
        if ui
            .button("Keep")
            .on_hover_text(
                "Store the number in your own settings file; the licensed models are \
                 downloaded with it",
            )
            .clicked()
        {
            *action = Some(ModelAction::SaveLicence);
        }
    });
    ui.weak(
        "The licensed TotalSegmentator models (heart chambers at 1.5 mm, coronary arteries, \
         tissue types, appendicular bones, brain structures and more) come from its licence \
         server; a licence is free for non-commercial use.",
    );
}

/// The nnU-Net model folders the user added, each with what became of it.
fn folders_rows(
    ui: &mut egui::Ui,
    folders: &[std::path::PathBuf],
    errors: &[(std::path::PathBuf, String)],
    running: bool,
    action: &mut Option<ModelAction>,
) {
    ui.horizontal(|ui| {
        ui.label("Your nnU-Net v2 models:");
        if ui
            .add_enabled(!running, egui::Button::new("➕ Add a model folder"))
            .on_hover_text(
                "A trained nnU-Net v2 model: a training folder with plans.json, \
                 dataset.json and fold_<k>/checkpoint_final.pth (or the dataset folder \
                 above it). It is then listed with the other automatic models.",
            )
            .clicked()
        {
            *action = Some(ModelAction::AddFolder);
        }
    });
    for (i, f) in folders.iter().enumerate() {
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!running, egui::Button::new("🗑").small())
                .on_hover_text("Forget this folder (its files are not touched)")
                .clicked()
            {
                *action = Some(ModelAction::RemoveFolder(i));
            }
            ui.label(
                egui::RichText::new(f.display().to_string())
                    .monospace()
                    .small(),
            );
            if let Some((_, why)) = errors.iter().find(|(p, _)| p == f) {
                ui.colored_label(warn_color(ui.visuals()), "not usable")
                    .on_hover_text(why.as_str());
            }
        });
    }
}
