//! *Tools ▶ Downloaded models*: one window over every model the three
//! engines can fetch.
//!
//! The engines each download their own weights on first use, which is
//! convenient right up to the moment somebody asks what is actually on this
//! machine, how much disk it costs, or wants a checkpoint re-fetched after a
//! bad download. This window answers all three from the shared inventory in
//! [`crate::models`]: a row per model with its state and size, per-row
//! **Download / Update / Remove**, and the same three actions over
//! everything at once. Preparing a model runs the engine's *own* first-use
//! path, so a model fetched here is bit for bit the one a run would have
//! fetched — there is no second download route to keep in step.

use anyhow::Context;

use super::*;
use crate::models::{self, AssetStatus, ModelAsset};

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
    /// Re-fetch every model that is ready.
    UpdateInstalled,
    /// Drop every source checkpoint.
    FreeAllSpare,
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
        self.models_scan = models::inventory()
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
        progress.set("starting…");
        self.models_result = None;
        self.models_job = Some(Job::spawn(progress, move |p| {
            let n = assets.len();
            let mut done = 0usize;
            for (i, a) in assets.iter().enumerate() {
                if p.cancelled() {
                    break;
                }
                p.set_phase(i as f32 / n as f32, 1.0 / n as f32);
                p.set(format!("{} — {} of {n}", a.label, i + 1));
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
                format!("{done} of {n} model(s) {verb} — cancelled")
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
        let missing: u64 = scan
            .iter()
            .filter(|(_, s)| !s.ready)
            .map(|(a, _)| a.download_bytes)
            .sum();
        let spare: u64 = scan.iter().map(|(_, s)| s.spare_bytes).sum();

        egui::Window::new("📦 Downloaded models")
            .id(egui::Id::new("models_window"))
            .collapsible(true)
            .resizable(true)
            .default_width(560.0)
            .open(&mut open)
            .show(ctx, |ui| {
                ui.label(
                    "Every model the three segmentation engines can fetch. Weights are \
                     downloaded once, converted to a cache beside them, and never touched \
                     again — this window is where that inventory is managed.",
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
                        .add_enabled(!running && ready_n > 0, egui::Button::new("⟳ Update all"))
                        .on_hover_text(
                            "Remove and re-fetch every model that is on disk — the published \
                             files carry no version, so an update is a fresh download",
                        )
                        .clicked()
                    {
                        action = Some(ModelAction::UpdateInstalled);
                    }
                    if ui
                        .add_enabled(
                            !running && spare > 0,
                            egui::Button::new(format!("🧹 Free {}", models::human_bytes(spare))),
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
                egui::ScrollArea::vertical()
                    .max_height(420.0)
                    .show(ui, |ui| {
                        for engine in models::Engine::ALL {
                            let tool = tool_of(engine);
                            ui.add_space(2.0);
                            ui.label(
                                egui::RichText::new(format!("{} {}", tool.glyph, tool.name))
                                    .strong(),
                            );
                            let (note, warn) = weights_licence(engine);
                            licence_line(ui, note, warn);
                            egui::Grid::new(("models_grid", engine.subdir()))
                                .num_columns(4)
                                .spacing([10.0, 4.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    for (i, (a, s)) in scan.iter().enumerate() {
                                        if a.engine != engine {
                                            continue;
                                        }
                                        model_row(ui, i, a, s, running, &mut action);
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
            });

        self.models_scan = scan;

        if browse {
            if let Some(dir) = Self::pick_folder("Model folder") {
                self.models_dir = dir.display().to_string();
                self.persist_settings();
                self.models_scan.clear();
            }
        }
        if cancel {
            if let Some(job) = &self.models_job {
                job.progress.cancel();
            }
        }
        if let Some(act) = action {
            self.apply_model_action(act, &root);
        }
        if !open || close {
            self.models_open = false;
            self.persist_settings();
        }
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
                let want: Vec<ModelAsset> = assets
                    .iter()
                    .zip(&states)
                    .filter(|(_, s)| !s.ready)
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
    action: &mut Option<ModelAction>,
) {
    let (glyph, tint) = if s.ready {
        ("✔", Color32::from_rgb(120, 200, 120))
    } else if s.partial {
        ("◐", Color32::from_rgb(230, 190, 90))
    } else {
        ("–", Color32::GRAY)
    };
    ui.label(egui::RichText::new(glyph).color(tint).monospace())
        .on_hover_text(if s.ready {
            "Ready — runs with no network access"
        } else if s.partial {
            "Partly downloaded — a run (or Download) finishes it"
        } else {
            "Not downloaded"
        });
    ui.label(&a.label).on_hover_text(a.detail);
    ui.label(
        egui::RichText::new(if s.ready {
            models::human_bytes(s.bytes)
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
                .add_enabled(!running, egui::Button::new("⬇").small())
                .on_hover_text("Download and convert this model")
                .clicked()
        {
            *action = Some(ModelAction::Fetch(i));
        }
        if s.ready
            && ui
                .add_enabled(!running, egui::Button::new("⟳").small())
                .on_hover_text("Remove and fetch this model again")
                .clicked()
        {
            *action = Some(ModelAction::Update(i));
        }
        if s.spare_bytes > 0
            && ui
                .add_enabled(!running, egui::Button::new("🧹").small())
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
                .on_hover_text("Delete every file of this model")
                .clicked()
        {
            *action = Some(ModelAction::Remove(i));
        }
    });
}
