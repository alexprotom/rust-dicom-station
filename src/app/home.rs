use super::*;

const BUTTON_WIDTH: f32 = 240.0;
const BUTTON_HEIGHT: f32 = 34.0;

fn home_section(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style())
        .inner_margin(egui::Margin::symmetric(18, 14))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(title)
                    .strong()
                    .text_style(egui::TextStyle::Heading),
            );

            ui.add_space(10.0);

            body(ui);
        });
}

fn centered_button_row(
    ui: &mut egui::Ui,
    total_width: f32,
    add_buttons: impl FnOnce(&mut egui::Ui),
) {
    ui.allocate_ui_with_layout(
        egui::vec2(total_width, BUTTON_HEIGHT),
        egui::Layout::left_to_right(egui::Align::Center),
        add_buttons,
    );
}

impl ViewerApp {
    pub(super) fn empty_state(&mut self, ui: &mut egui::Ui) {
        let now = ui.input(|i| i.time);
        let offer_pacs = self.archive_has_data(now);

        let mut open_folder = false;
        let mut open_pacs = false;
        let mut restore = false;
        let mut generate = false;
        let mut anonymize = false;

        let two_button_row_width = BUTTON_WIDTH * 2.0 + ui.spacing().item_spacing.x;

        let available_height = ui.available_height();
        let top_space = ((available_height - self.home_content_height) / 2.0).max(0.0);
        ui.add_space(top_space);

        let content_top = ui.cursor().top();

        ui.with_layout(
            egui::Layout::top_down(egui::Align::Center),
            |ui| {
                // ---------------------------------------------------------
                // Header
                // ---------------------------------------------------------
                ui.label(
                    egui::RichText::new("Rust DICOM Station")
                        .size(32.0)
                        .strong()
                        .color(egui::Color32::WHITE),
                );

                ui.add_space(6.0);

                ui.label(
                    egui::RichText::new(
                        "An open-source DICOM workstation for radiotherapy research, analysis, and QA, written entirely in Rust.",
                    )
                    .weak()
                    .size(14.0),
                );

                ui.add_space(8.0);

                ui.label(
                    egui::RichText::new(format!(
                        "Version {} - 2026",
                        env!("CARGO_PKG_VERSION")
                    ))
                    .weak()
                    .size(12.0),
                );

                ui.add_space(8.0);

                ui.hyperlink_to("GitHub","https://github.com/alexprotom/rust-dicom-station");

                ui.add_space(35.0);

                // ---------------------------------------------------------
                // Main content
                // ---------------------------------------------------------
                let content_width = ui.available_width().min(760.0);
                ui.allocate_ui_with_layout(
                    egui::vec2(content_width, 0.0),
                    egui::Layout::top_down(egui::Align::Center),
                    |ui| {
                        // =================================================
                        // INPUT
                        // =================================================
                        home_section(ui, "Input", |ui| {
                            centered_button_row(ui, two_button_row_width, |ui| {
                                if ui
                                    .add_sized(
                                        [BUTTON_WIDTH, BUTTON_HEIGHT],
                                        egui::Button::new("📂  Add DICOM folder"),
                                    )
                                    .on_hover_text(
                                        "Scan a folder of DICOM files into dataset A",
                                    )
                                    .clicked()
                                {
                                    open_folder = true;
                                }

                                if ui
                                    .add_sized(
                                        [BUTTON_WIDTH, BUTTON_HEIGHT],
                                        egui::Button::new(
                                            "⟳  Restore last session",
                                        ),
                                    )
                                    .on_hover_text(
                                        "Load again what was open when RDS was last closed",
                                    )
                                    .clicked()
                                {
                                    restore = true;
                                }
                            });

                            if offer_pacs {
                                ui.add_space(8.0);

                                centered_button_row(ui, two_button_row_width, |ui| {
                                    if ui
                                        .add_sized(
                                            [two_button_row_width, BUTTON_HEIGHT],
                                            egui::Button::new(
                                                "🏥  Patient archive",
                                            ),
                                        )
                                        .on_hover_text(
                                            "Open the local patient archive",
                                        )
                                        .clicked()
                                    {
                                        open_pacs = true;
                                    }
                                });
                            }
                        });

                        ui.add_space(14.0);

                        // =================================================
                        // TOOLS
                        // =================================================
                        home_section(ui, "Tools", |ui| {
                            centered_button_row(ui, two_button_row_width, |ui| {
                                if ui
                                    .add_sized(
                                        [BUTTON_WIDTH, BUTTON_HEIGHT],
                                        egui::Button::new(
                                            "📐  Generate test data",
                                        ),
                                    )
                                    .on_hover_text(
                                        "Generate a synthetic DICOM RT study",
                                    )
                                    .clicked()
                                {
                                    generate = true;
                                }

                                if ui
                                    .add_sized(
                                        [BUTTON_WIDTH, BUTTON_HEIGHT],
                                        egui::Button::new(
                                            "🔏   Anonymize DICOM folder",
                                        ),
                                    )
                                    .on_hover_text(
                                        "Remove or replace identifying DICOM information",
                                    )
                                    .clicked()
                                {
                                    anonymize = true;
                                }
                            });

                            ui.add_space(8.0);

                            centered_button_row(ui, two_button_row_width, |ui| {
                                if ui
                                    .add_sized(
                                        [two_button_row_width, BUTTON_HEIGHT],
                                        egui::Button::new(
                                            "📦  Downloaded models",
                                        ),
                                    )
                                    .on_hover_text(
                                        "Manage downloaded AI model weights",
                                    )
                                    .clicked()
                                {
                                    self.open_models_window();
                                }
                            });
                        });
                    },
                );
            },
        );

        self.home_content_height = ui.cursor().top() - content_top;

        // -------------------------------------------------------------
        // Actions are deliberately handled after the UI has been drawn.
        // This follows the existing empty_state pattern.
        // -------------------------------------------------------------

        if open_folder {
            if let Some(dir) = Self::pick_folder("Select a DICOM folder") {
                self.start_load(0, dir);
            }
        }

        if open_pacs {
            self.open_pacs_window();
        }

        if restore {
            self.restore_last_session();
        }

        if generate {
            self.gen_open = true;
        }

        if anonymize {
            self.anon_open = true;
        }
    }
}
