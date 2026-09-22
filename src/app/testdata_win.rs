//! *Tools ▶ Download test data*: the window over `crate::testdata`, which
//! fetches the repository's `data-test/` folder (a whole real patient: a
//! ten-phase 4DFBCT with a structure set per phase and the matching
//! ten-phase 4DCBCT, `docs/example-data.md`) from GitHub.
//!
//! Same shape as the generator window beside it in the menu: a destination
//! folder with Browse and a reset to the application's data folder, one
//! button that starts a background job, the job's progress while it runs, a
//! line with the outcome, and the option to load the first study into
//! workspace A when everything is there. A run that finds every file already
//! present touches nothing and says so.

use super::*;

impl ViewerApp {
    /// Draw the window when it is open.
    pub(super) fn testdata_window(&mut self, ctx: &egui::Context) {
        if !self.testdata_open {
            return;
        }
        let running = self.testdata_job.is_some();
        let mut open = true;
        let mut start = false;
        let mut cancel = false;
        let mut browse = false;
        let mut reset_dir = false;

        detach::tool_window(
            ctx,
            "testdata",
            "📥 Download test data",
            &mut open,
            // Wide enough for the description and the folder row without a
            // scrollbar; the height is what the contents need.
            detach::WinOpts::size(600.0, 340.0),
            |ui| {
                ui.set_max_width(560.0);
                ui.label(
                    "Fetches the bundled patient data of the project from GitHub: TCIA \
                     4D-Lung P102 in full - a ten-phase 4DFBCT (133 slices a phase) \
                     with an RT Structure Set per phase, and the matching ten-phase \
                     4DCBCT (50 slices a phase). About 980 MB in 1840 files, so give \
                     it time; files already in the folder are kept, and an interrupted \
                     download continues where it stopped.",
                );
                ui.horizontal(|ui| {
                    ui.hyperlink_to("The folder on GitHub", testdata::browse_url());
                    ui.weak("(CC BY 3.0, see docs/example-data.md)");
                });
                ui.add_space(6.0);

                ui.label(egui::RichText::new("Destination folder").strong());
                ui.horizontal(|ui| {
                    ui.add_enabled(
                        !running,
                        egui::TextEdit::singleline(&mut self.testdata_dir)
                            .desired_width(360.0)
                            .hint_text("folder to write data-test into"),
                    );
                    if ui
                        .add_enabled(!running, egui::Button::new("📂 Browse"))
                        .clicked()
                    {
                        browse = true;
                    }
                    if ui
                        .add_enabled(!running, egui::Button::new("↺"))
                        .on_hover_text("Reset to the application's data folder")
                        .clicked()
                    {
                        reset_dir = true;
                    }
                });
                ui.weak(format!(
                    "Studies: {f}/{s}/4DCBCT, {f}/{s}/4DFBCT+RTS",
                    f = testdata::FOLDER,
                    s = testdata::PATIENT
                ));
                ui.checkbox(
                    &mut self.testdata_load_after,
                    "Load the first study into workspace A when done",
                );

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if running {
                        ui.spinner();
                        if let Some(job) = &self.testdata_job {
                            ui.add(
                                egui::ProgressBar::new(job.progress.frac())
                                    .desired_width(220.0)
                                    .show_percentage(),
                            );
                            ui.label(job.progress.get());
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    } else if ui
                        .add(egui::Button::new("📥 Download"))
                        .on_hover_text("Fetches only the files that are not there yet")
                        .clicked()
                    {
                        start = true;
                    }
                });
                if let Some(msg) = &self.testdata_result {
                    ui.add_space(4.0);
                    ui.label(msg);
                }
            },
        );

        self.testdata_open = open;
        if browse {
            self.ask_folder("Select a folder for the test data", |app, dir| {
                app.testdata_dir = dir.display().to_string();
            });
        }
        if reset_dir {
            self.testdata_dir = testdata::default_output_dir().display().to_string();
        }
        if cancel {
            if let Some(job) = &self.testdata_job {
                job.progress.cancel();
            }
        }
        if start {
            self.start_testdata_download();
        }
    }

    /// Fetch the folder on a worker thread; the window polls the job.
    pub(super) fn start_testdata_download(&mut self) {
        if self.testdata_job.is_some() {
            return;
        }
        let dir = PathBuf::from(self.testdata_dir.trim());
        if dir.as_os_str().is_empty() {
            self.error = Some("Choose a folder for the test data".into());
            return;
        }
        let progress = Arc::new(Progress::default());
        progress.set("starting");
        let p2 = progress.clone();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(testdata::download(&dir, &*p2));
        });
        self.testdata_result = None;
        self.testdata_job = Some(Job { progress, rx });
    }

    /// Land the finished download: the outcome line, and the first study
    /// into workspace A when asked for. A cancelled run is what the user
    /// asked for and raises no error dialog.
    pub(super) fn poll_testdata_job(&mut self, ctx: &egui::Context) {
        match poll_job(
            &mut self.testdata_job,
            ctx,
            "Test data download",
            &mut self.error,
        ) {
            Some(Ok(s)) => {
                self.testdata_result = Some(if s.downloaded == 0 {
                    format!(
                        "✔ All {} file(s) were already in {}",
                        s.kept,
                        s.dir.display()
                    )
                } else {
                    format!(
                        "✔ {} file(s), {} MB downloaded to {}{}",
                        s.downloaded,
                        s.bytes / 1_000_000,
                        s.dir.display(),
                        if s.kept > 0 {
                            format!(" ({} already there)", s.kept)
                        } else {
                            String::new()
                        }
                    )
                });
                if self.testdata_load_after {
                    if let Some(first) = s.datasets.first() {
                        self.testdata_open = false;
                        self.start_load(0, first.clone());
                    }
                }
            }
            Some(Err(e)) => {
                if progress::is_cancellation(&e) {
                    self.testdata_result =
                        Some("Cancelled - the files fetched so far are kept".into());
                } else {
                    self.error = Some(format!("Test data download failed: {e:#}"));
                }
            }
            None => {}
        }
    }
}
