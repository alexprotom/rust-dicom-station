//! Window chrome: the menu bar, the toolbar and the status bar.

use super::*;

impl ViewerApp {
    // -- Menu bar ---------------------------------------------------------
    pub(super) fn menu_bar(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let mut load_into: Option<(workspace_pick::WsAsk, usize)> = None;
        let mut clear_ws: Option<usize> = None;
        let mut clear_all = false;
        let mut open_gen = false;
        let mut open_testdata = false;
        let mut open_save_img = false;
        let mut open_models = false;
        let mut open_pacs = false;
        let mut open_export = false;
        let mut wf_menu: Option<workflow_edit::WfPending> = None;
        let mut wf_show_editor = false;
        let mut wf_show_run = false;
        let mut new_theme: Option<egui::ThemePreference> = None;
        let mut save_settings = false;
        // A module was switched on or off - remember it for the next run.
        let mut modules_changed = false;

        egui::Panel::top(egui::Id::new("menu_bar")).show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    // A submenu each, one line per workspace the data may go
                    // into: those on screen plus one new letter while there is
                    // room. The line says what that workspace already holds,
                    // so adding to a full one is a deliberate act. The second
                    // entry is for the objects that do not come as a folder of
                    // slices - an RT image, a structure set, a plan, a single
                    // slice; they merge exactly as a folder does.
                    let targets = self.open_plus_new();
                    for ask in [
                        workspace_pick::WsAsk::AddFolder,
                        workspace_pick::WsAsk::AddFiles,
                    ] {
                        ui.menu_button(ask.menu_entry(), |ui| {
                            for slot in &targets {
                                let label = format!(
                                    "{} · {}",
                                    SLOT_NAMES[*slot],
                                    self.workspace_summary(*slot)
                                );
                                if ui.button(label).on_hover_text(ask.blurb()).clicked() {
                                    load_into = Some((ask, *slot));
                                    ui.close();
                                }
                            }
                        });
                    }
                    ui.separator();
                    // One line per workspace that holds something, and *All*
                    // under them: the question is short enough to answer in
                    // the menu, so there is no window for it.
                    let loaded: Vec<usize> = (0..MAX_WORKSPACES)
                        .filter(|s| self.slots[*s].study.is_some())
                        .collect();
                    ui.add_enabled_ui(!loaded.is_empty(), |ui| {
                        ui.menu_button("🗑 Clear workspace", |ui| {
                            for slot in &loaded {
                                let label = format!(
                                    "{} · {}",
                                    SLOT_NAMES[*slot],
                                    self.workspace_summary(*slot)
                                );
                                if ui
                                    .button(label)
                                    .on_hover_text(
                                        "Empty this workspace: its patients, studies, \
                                         series and everything drawn on them. Nothing is \
                                         written to disk",
                                    )
                                    .clicked()
                                {
                                    clear_ws = Some(*slot);
                                    ui.close();
                                }
                            }
                            if loaded.len() > 1 {
                                ui.separator();
                                if ui
                                    .button("All")
                                    .on_hover_text("Empty every workspace at once")
                                    .clicked()
                                {
                                    clear_all = true;
                                    ui.close();
                                }
                            }
                        });
                    });
                    ui.separator();
                    // One export for everything that is loaded: which
                    // patients, studies and series go out is chosen in the
                    // window, not by which menu entry was clicked.
                    let anything = self.slots.iter().any(|s| s.study.is_some());
                    if enabled_tip_button(
                        ui,
                        anything,
                        "💾 Export DICOM",
                        "Write any patients, studies, series and RT objects of either \
                         workspace as DICOM - with every name and UID shown and editable, \
                         structures as RTSTRUCT or SEG, and the references between the \
                         objects kept intact",
                    ) {
                        open_export = true;
                        ui.close();
                    }
                    ui.separator();
                    if tip_button(
                        ui,
                        "💾 Save image",
                        "A picture of the views as they are on screen - one workspace's row \
                         or both - as a PNG or a JPEG, at a chosen resolution",
                    ) {
                        open_save_img = true;
                        ui.close();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                // Switches, like the Modules menu: ticking one leaves the
                // menu open so several can be set in one visit. It closes on a
                // click outside it or on the "View" title again.
                egui::containers::menu::MenuButton::new("View")
                    .config(
                        egui::containers::menu::MenuConfig::new()
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                    )
                    .ui(ui, |ui| {
                        // One line per workspace: which of them are on
                        // screen. A workspace put aside keeps everything it
                        // holds - this is the screen, not the data.
                        ui.menu_button("Workspaces", |ui| {
                            for (slot, name) in SLOT_NAMES.iter().enumerate() {
                                let holds = self.slots[slot].study.is_some();
                                let mut on = self.is_open(slot);
                                let label = if holds {
                                    format!("{name} · loaded")
                                } else {
                                    format!("{name} · empty")
                                };
                                let resp = ui
                                    .add_enabled(slot > 0, egui::Checkbox::new(&mut on, label))
                                    .on_hover_text(if holds {
                                        "Put this workspace aside or bring it back. What it \
                                         holds stays loaded either way."
                                    } else {
                                        "Open an empty workspace to load into, or close it \
                                         again."
                                    });
                                if resp.changed() {
                                    self.shown[slot] = on;
                                }
                            }
                        });
                        ui.separator();
                        ui.checkbox(&mut self.show_contours, "Contours");
                        ui.checkbox(&mut self.show_crosshair, "Crosshair");
                        // Syncing carries the slice, the zoom, the pan and
                        // both players as well as the crosshair, so all it
                        // needs is a second workspace to carry them to.
                        if self.both_volumes() {
                            ui.checkbox(&mut self.link_studies, "Sync the workspaces")
                                .on_hover_text(
                                    "Move, scroll, zoom or play one workspace and every \
                             other open one follows: the crosshair to the same patient point \
                             (through the active registration when there is one), the slice \
                             with it, a scrolled or played slice the same way, the zoom and \
                             the pan of a view onto the other workspaces' views of the same \
                             plane, and a 4D run through the groups at once. Off, each \
                             workspace is navigated on its own.",
                                );
                        }
                        ui.checkbox(&mut self.show_labels, "Orientation labels");
                        ui.checkbox(&mut self.show_isocenters, "Isocenters");
                        ui.separator();
                        ui.checkbox(&mut self.side_open, "Data tree (F9)")
                            .on_hover_text(
                                "The left panel. Hidden, its arrow stays on the window's left \
                                 edge to bring it back, and so does F9.",
                            );
                        let any_module = self.any_module();
                        ui.add_enabled_ui(any_module, |ui| {
                            ui.checkbox(&mut self.right_open, "Modules (F10)")
                                .on_hover_text(if any_module {
                                    "The right panel. Hidden, its arrow stays on the window's \
                                     right edge to bring it back, and so does F10."
                                } else {
                                    "There is no modules panel until a module is turned on in \
                                     the Modules menu"
                                });
                        });
                        ui.separator();
                        ui.label("Appearance:");
                        let before = self.theme;
                        self.theme.radio_buttons(ui);
                        if self.theme != before {
                            new_theme = Some(self.theme);
                        }
                    });
                // This menu is a set of switches, not a list of actions:
                // it stays open until the pointer leaves it, so several
                // modules can be turned on in one visit.
                egui::containers::menu::MenuButton::new("Modules")
                    .config(
                        egui::containers::menu::MenuConfig::new()
                            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                    )
                    .ui(ui, |ui| {
                        // Every module is one line of this menu and one
                        // section of the right panel. Everything a module
                        // does - direction, method, region, parameters,
                        // landmarks, analytics, fusion, the vector field, the
                        // simulated motion, what travels and where - lives in
                        // its section, so the menu only decides whether it is
                        // there.
                        ui.weak("Sections of the right panel (F10):");
                        // Fifteen lines in one column is a wall; they are
                        // grouped by what they work on.
                        for group in modules::Group::ALL {
                            ui.separator();
                            ui.label(egui::RichText::new(group.title()).small().strong());
                            for m in Module::ALL.into_iter().filter(|m| m.group() == group) {
                                let mut on = self.module_on(m);
                                if ui
                                    .checkbox(&mut on, m.title())
                                    .on_hover_text(m.menu_tip())
                                    .changed()
                                {
                                    if on {
                                        self.modules.insert(m);
                                    } else {
                                        self.modules.remove(&m);
                                    }
                                    modules_changed = true;
                                }
                            }
                        }
                        if self.any_module() {
                            self.right_open = true;
                        }
                    });
                ui.menu_button("Tools", |ui| {
                    // The tools that are about files and folders rather than
                    // about the images on screen. Everything that reads the
                    // displayed study is a module of the right panel.
                    if tip_button(
                        ui,
                        "🏥 PACS",
                        "Picture Archiving and Communication System: this station's \
                         cohorts, patients, studies and series, and the PACS servers it is \
                         paired with; load into a workspace, organise, give back what was \
                         drawn",
                    ) {
                        open_pacs = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "📦 Downloaded models",
                        "What every segmentation engine has downloaded, how much disk \
                         it costs, and the buttons to download, update or remove it - \
                         one model at a time or all of them",
                    ) {
                        open_models = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "🔏 Anonymize DICOM folder",
                        "Scan a folder, review every identifying tag with its current \
                         and proposed values, then rewrite the files (in place or into \
                         a new folder) with consistently regenerated UIDs",
                    ) {
                        self.anon_open = true;
                        ui.close();
                    }
                    ui.separator();
                    if tip_button(
                        ui,
                        "📐 Generate test data",
                        "Write a complete synthetic RT study (CT, RTSTRUCT, RTPLAN, \
                         RTDOSE, DX, RTIMAGE, REG, RTRECORD) into the application folder",
                    ) {
                        open_gen = true;
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "📥 Download test data",
                        "Fetch the bundled patient data (a ten-phase 4DFBCT with \
                         structure sets and the matching 4DCBCT, 980 MB) from the \
                         project's GitHub repository into a folder of your choice",
                    ) {
                        open_testdata = true;
                        ui.close();
                    }
                });
                // Workflows: the program's steps wired into a graph, saved as
                // a file and run again on other data (see workflow_edit.rs).
                ui.menu_button("Workflows", |ui| {
                    if tip_button(
                        ui,
                        "🔀 New",
                        "Open the workflow editor on an empty canvas: add steps from the \
                         palette, wire what one makes into the next",
                    ) {
                        wf_menu = Some(workflow_edit::WfPending::New);
                        ui.close();
                    }
                    if tip_button(
                        ui,
                        "📂 Load",
                        "Open a workflow file (.rdsflow) in the editor, to run it again or \
                         change it",
                    ) {
                        wf_menu = Some(workflow_edit::WfPending::Open);
                        ui.close();
                    }
                    let recent = self.recent_workflows.clone();
                    ui.add_enabled_ui(!recent.is_empty(), |ui| {
                        ui.menu_button("🕐 Last saved", |ui| {
                            for p in &recent {
                                let name = p
                                    .file_stem()
                                    .map(|s| s.to_string_lossy().to_string())
                                    .unwrap_or_else(|| p.display().to_string());
                                if ui
                                    .button(name)
                                    .on_hover_text(p.display().to_string())
                                    .clicked()
                                {
                                    wf_menu = Some(workflow_edit::WfPending::File(p.clone()));
                                    ui.close();
                                }
                            }
                        });
                    });
                    ui.menu_button("Examples", |ui| {
                        for (i, e) in crate::workflow::graph::store::EXAMPLES.iter().enumerate() {
                            if ui.button(e.name).on_hover_text(e.blurb).clicked() {
                                wf_menu = Some(workflow_edit::WfPending::Example(i));
                                ui.close();
                            }
                        }
                    });
                    if self.wf_editor.is_some() || self.wf_run.is_some() {
                        ui.separator();
                    }
                    if let Some(ed) = &self.wf_editor {
                        if ui
                            .button(format!("Show the editor: {}", ed.name.trim()))
                            .clicked()
                        {
                            wf_show_editor = true;
                            ui.close();
                        }
                    }
                    if let Some(run) = &self.wf_run {
                        let what = if run.is_running() {
                            "Show the run in progress"
                        } else {
                            "Show the last run"
                        };
                        if ui.button(what).clicked() {
                            wf_show_run = true;
                            ui.close();
                        }
                    }
                    ui.separator();
                    ui.weak(format!(
                        "Your workflows: {}",
                        crate::workflow::graph::store::user_dir().display()
                    ));
                });
                ui.menu_button("Settings", |ui| {
                    // Tick boxes: the submenu stays open while rows are being
                    // put together, and goes away on a click outside it or on
                    // "Settings" again.
                    egui::containers::menu::SubMenuButton::new("View layout")
                        .config(
                            egui::containers::menu::MenuConfig::new()
                                .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside),
                        )
                        .ui(ui, |ui| {
                            if self.view_layout_menu(ui) {
                                save_settings = true;
                            }
                        });
                    ui.menu_button("Graphics backend", |ui| {
                        ui.label("Which graphics API the program draws and computes with.");
                        ui.add_space(4.0);
                        let before = self.graphics_backend;
                        for b in crate::gfx::Backend::offered() {
                            ui.radio_value(&mut self.graphics_backend, b, b.label())
                                .on_hover_text(b.hint());
                        }
                        if self.graphics_backend != before {
                            save_settings = true;
                        }
                        ui.add_space(4.0);
                        // What is drawing right now, which is not always what
                        // was asked for: an unusable driver is fallen back
                        // from without telling anybody.
                        let running = self
                            .active_backend
                            .or_else(crate::gfx::from_env)
                            .unwrap_or(self.graphics_backend);
                        ui.weak(format!(
                            "{} is set now. A change takes effect the next time the \
                             program starts.",
                            running.label()
                        ));
                    });
                    ui.menu_button("MCP server", |ui| {
                        let exe = crate::settings::mcp_exe_path();
                        if exe.is_file() {
                            // In a snap or an AppImage the file sits in a
                            // mount; what a client runs is shown instead.
                            ui.weak(format!(
                                "MCP installed: {}",
                                crate::settings::mcp_client_launch().display()
                            ));
                        } else {
                            ui.weak(format!(
                                "MCP not installed: {} was not found. Build it with \
                                 cargo build --release --features mcp.",
                                exe.display()
                            ));
                        }
                        ui.weak(format!(
                            "MCP configuration: {}",
                            crate::settings::mcp_config_path().display()
                        ));
                        ui.add_space(4.0);
                        if tip_button(
                            ui,
                            "Copy client configuration",
                            "Copies the JSON entry for claude_desktop_config.json (or an \
                             equivalent MCP client) to the clipboard.",
                        ) {
                            ui.ctx().copy_text(crate::settings::mcp_client_snippet());
                            ui.close();
                        }
                    });
                    if let Some(msg) = &self.settings_error {
                        ui.weak(msg);
                    }
                });
                ui.menu_button("Help", |ui| {
                    ui.label("MPR views - mouse:");
                    ui.weak("Left click / drag - move linked crosshair");
                    ui.weak("Mouse wheel - scroll slices");
                    ui.weak("Ctrl + wheel / pinch - zoom at cursor");
                    ui.weak("Middle drag - pan");
                    ui.weak("Right drag - window / level (x = width, y = center)");
                    ui.separator();
                    ui.label(
                        "Drawing tools (✏ Draw structure on the toolbar; they take over the \
                         left button):",
                    );
                    ui.weak("Left drag - paint / erase");
                    ui.weak("Left press + drag ↑↓ - grow / shrink the region (✨)");
                    ui.weak("Alt - erase while painting");
                    ui.weak("Shift + wheel, or [ ] - brush radius");
                    ui.weak("Ctrl + Z - undo the last stroke");
                    ui.weak("Esc - cancel the running region grow");
                    ui.separator();
                    ui.label(format!(
                        "{} (the box takes over the left button in its view):",
                        SLICE_PROP.name
                    ));
                    ui.weak(
                        "Left drag - draw the box; drag a corner to resize, the middle to move",
                    );
                    ui.weak("Left click - an include / exclude point, with ➕ / ➖ chosen");
                    ui.separator();
                    ui.label("Buttons:");
                    ui.weak("⟲ (view corner) - reset that view's zoom, pan and slice");
                    ui.weak("⛶ / ⊞ - maximize that view / restore the layout");
                    ui.weak("⟲ (toolbar) - reset every view of both workspaces");
                    ui.weak("✏ Draw structure (toolbar) - unfold or fold the drawing tools");
                    ui.weak(
                        "⌖ - show / hide the crosshair; hidden, left click no \
                         longer navigates",
                    );
                    ui.weak(
                        "Sync - keep workspaces A and B on the same point, the same scale \
                         and the same frame of a run (shown with two workspaces loaded)",
                    );
                    ui.separator();
                    ui.weak(format!(
                        "rust-dicom-station {} - research / QA viewer, not a medical device",
                        env!("CARGO_PKG_VERSION")
                    ));
                });
            });
        });

        if let Some((ask, slot)) = load_into {
            self.load_into(ask, slot);
        }
        if let Some(slot) = clear_ws {
            self.clear_workspace(slot);
        }
        if clear_all {
            // Backwards, so closing C and D does not move A and B about
            // under the loop.
            for slot in (0..MAX_WORKSPACES).rev() {
                self.clear_workspace(slot);
            }
        }
        if open_save_img && self.save_img.is_none() {
            self.save_img = Some(snapshot::SaveImgDialog::default());
        }
        if open_testdata {
            self.testdata_open = true;
        }
        if open_gen {
            self.gen_open = true;
        }
        if open_pacs {
            self.open_pacs_window();
        }
        if open_models {
            self.open_models_window();
        }
        if open_export {
            self.open_export_dialog();
        }
        if let Some(what) = wf_menu {
            self.wf_replace(what);
        }
        if wf_show_editor {
            if let Some(e) = &mut self.wf_editor {
                e.open = true;
            }
        }
        if wf_show_run {
            self.reveal_module(Module::Workflows);
        }
        if let Some(theme) = new_theme {
            self.set_theme(ctx, theme);
        }
        if save_settings {
            self.persist_settings();
        }
        if modules_changed {
            self.persist_settings();
        }
    }

    // -- Toolbar ----------------------------------------------------------
    /// The toolbar's window over a planar image: the C / W of the image at
    /// the pane's scrubber, and its presets - its own window, then windows
    /// read off its values, as an MR's are.
    fn planar_wl_controls(&mut self, ui: &mut egui::Ui, t: (usize, usize, usize)) {
        let (slot, k, _) = t;
        let Some((modality, (mut c, mut w))) = self
            .planar_wl(t)
            .map(|(img, wl)| (img.modality.clone(), wl))
        else {
            return;
        };
        let pane = super::planar::pane_kind_of(k).label();
        ui.label(if self.open_slots().len() > 1 {
            format!("W/L {pane} {}:", SLOT_NAMES[slot])
        } else {
            format!("W/L {pane}:")
        })
        .on_hover_text(
            "The window of the image at this pane's scrubber - the planar or additional pane \
             last pointed at. A preset picked here goes on every planar image of the \
             workspaces, and on each one shown later. Right-drag in the pane changes the \
             window too.",
        );
        let speed = (w * 0.005).max(1e-3) as f64;
        let mut changed = ui
            .add(egui::DragValue::new(&mut c).speed(speed).prefix("C "))
            .changed();
        changed |= ui
            .add(egui::DragValue::new(&mut w).speed(speed).prefix("W "))
            .changed();
        if changed {
            self.set_planar_wl(t, (c, w));
            self.wl_planar_preset = None;
        }
        let windows = self.planar_relative_windows(t);
        let presets: Vec<(String, f32, f32)> = RELATIVE_PRESETS
            .iter()
            .zip(windows)
            .map(|((n, ..), (c, w))| (n.to_string(), c, w))
            .collect();
        let current = self.wl_planar_preset.filter(|i| *i < presets.len());
        let mut pick: Option<usize> = None;
        let mut full_range = false;
        egui::ComboBox::from_id_salt("wl_preset_planar")
            .selected_text(match current {
                Some(i) => presets[i].0.clone(),
                None => format!("{modality} presets"),
            })
            .width(170.0)
            .show_ui(ui, |ui| {
                ui.weak(format!(
                    "{modality} values have no fixed scale: these windows are read off this \
                     image"
                ));
                for (i, (name, c, w)) in presets.iter().enumerate() {
                    if ui
                        .selectable_label(
                            current == Some(i),
                            format!("{name}  (C {c:.4} / W {w:.4})"),
                        )
                        .clicked()
                    {
                        pick = Some(i);
                    }
                }
                ui.separator();
                if ui.button("Full range").clicked() {
                    full_range = true;
                }
            });
        if let Some(i) = pick {
            self.pick_planar_preset(i);
        }
        if full_range {
            if let Some((img, _)) = self.planar_wl(t) {
                let (lo, hi) = (img.min_value, img.max_value);
                self.set_planar_wl(t, ((lo + hi) * 0.5, (hi - lo).max(1e-6)));
            }
            self.wl_planar_preset = None;
        }
    }

    pub(super) fn top_bar(&mut self, ui: &mut egui::Ui) {
        // Only the primary reading controls live here (window/level);
        // file actions, display toggles and appearance are in the menus.
        let any_study = self.slots.iter().any(|s| s.study.is_some());
        if !any_study {
            return;
        }
        egui::Panel::top(egui::Id::new("top_bar")).show(ui, |ui| {
            ui.horizontal(|ui| {
                // The volume's window, then - while a planar or additional
                // pane shows an image - that image's beside it.
                if self.any_volume() {
                    // The window of the workspace last pointed at; the
                    // others showing the same modality follow it.
                    let slot = self.wl_slot();
                    let modality = self.shown_modality(slot);
                    let is_ct = modality == "CT" || modality.is_empty();
                    let (mut c, mut w) = self.window_of(slot);
                    let several = self.open_slots().len() > 1;
                    ui.label(if several {
                        format!("W/L {}:", SLOT_NAMES[slot])
                    } else {
                        "W/L:".to_string()
                    })
                    .on_hover_text(
                        "The display window of the workspace last pointed at. Workspaces \
                         showing the same modality share it; a CT and an MR each keep \
                         their own. Right-drag in a view changes it too.",
                    );
                    let mut changed = ui
                        .add(egui::DragValue::new(&mut c).speed(2.0).prefix("C "))
                        .changed();
                    changed |= ui
                        .add(
                            egui::DragValue::new(&mut w)
                                .speed(4.0)
                                .range(1.0..=20000.0)
                                .prefix("W "),
                        )
                        .changed();
                    if changed {
                        self.set_window(slot, c, w);
                        self.forget_preset(slot);
                    }
                    // The presets of the modality shown: Hounsfield windows
                    // for a CT; for an MR or a PET, whose values have no
                    // fixed scale, windows read off the image's own values.
                    let presets: Vec<(String, f32, f32)> = if is_ct {
                        WL_PRESETS
                            .iter()
                            .map(|(n, c, w)| (n.to_string(), *c, *w))
                            .collect()
                    } else {
                        let windows = self.relative_windows(slot);
                        RELATIVE_PRESETS
                            .iter()
                            .zip(windows)
                            .map(|((n, ..), (c, w))| (n.to_string(), c, w))
                            .collect()
                    };
                    // The closed combo carries the name of the preset in
                    // force. Its numbers are dropped there - the two drag
                    // values to the left already show them - but kept in the
                    // list, where they are what tells the presets apart.
                    // Any other window (a drag, a right-drag in a view, the
                    // full range) is nameless again. The preset stays picked
                    // across series: the next CT comes up in it, the next MR
                    // in its own values' version of it.
                    let title = if is_ct {
                        "CT presets".to_string()
                    } else {
                        format!("{modality} presets")
                    };
                    let current = if is_ct {
                        self.wl_ct_preset
                    } else {
                        self.wl_rel_preset
                    };
                    let selected = current
                        .and_then(|i| presets.get(i))
                        .map_or(title.clone(), |(name, ..)| name.clone());
                    let mut pick: Option<usize> = None;
                    let mut full_range = false;
                    egui::ComboBox::from_id_salt("wl_preset")
                        .selected_text(selected)
                        .width(170.0)
                        .show_ui(ui, |ui| {
                            if !is_ct {
                                ui.weak(format!(
                                    "{modality} values have no fixed scale: these windows are \
                                     read off this image"
                                ));
                            }
                            for (i, (name, c, w)) in presets.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        current == Some(i),
                                        format!("{name}  (C {c:.0} / W {w:.0})"),
                                    )
                                    .clicked()
                                {
                                    pick = Some(i);
                                }
                            }
                            ui.separator();
                            if ui.button("Full range").clicked() {
                                full_range = true;
                            }
                        });
                    if let Some(i) = pick {
                        let (_, c, w) = presets[i];
                        self.set_window(slot, c, w);
                        if is_ct {
                            self.wl_ct_preset = Some(i);
                        } else {
                            self.wl_rel_preset = Some(i);
                        }
                    }
                    if full_range {
                        // Read the range off the workspace itself; an empty
                        // volume would otherwise set the window to the
                        // degenerate C 0 / W 1.
                        if let Some(study) = self.slots[slot]
                            .study
                            .as_ref()
                            .filter(|_| self.slots[slot].has_volume())
                        {
                            let v = &study.volume;
                            let c = (v.min_value as f32 + v.max_value as f32) * 0.5;
                            let w = (v.max_value as f32 - v.min_value as f32).max(1.0);
                            self.set_window(slot, c, w);
                            self.forget_preset(slot);
                        }
                    }
                }
                if let Some(t) = self.planar_wl_target() {
                    if self.any_volume() {
                        ui.separator();
                    }
                    self.planar_wl_controls(ui, t);
                }
                {
                    ui.separator();
                    // The surface scene of each workspace: a pane of its row,
                    // put in and taken out again by one button.
                    for slot in self.open_slots() {
                        let slot_name = SLOT_NAMES[slot];
                        if slot > 0 && self.slots[slot].study.is_none() {
                            continue;
                        }
                        let shown = self.row_shows_scene(slot);
                        let has_3d = self.slot_has_surfaces(slot);
                        if ui
                            .add_enabled(
                                has_3d || shown,
                                egui::Button::selectable(shown, format!("3D {slot_name}")),
                            )
                            .on_hover_text(if shown {
                                format!(
                                    "Take the 3D surface scene out of workspace {slot_name}'s row"
                                )
                            } else {
                                format!(
                                    "Put a 3D surface rendering of workspace {slot_name}'s \
                                     structures and segmentations into its row, beside the views"
                                )
                            })
                            .clicked()
                        {
                            if shown {
                                self.remove_from_row(slot, PaneKind::Scene3d);
                            } else {
                                self.add_to_row(slot, PaneKind::Scene3d);
                            }
                        }
                    }

                    // Slice-intersection (crosshair) toggle. With the
                    // crosshair hidden, left-click navigation is disabled.
                    if ui
                        .selectable_label(self.show_crosshair, "⌖")
                        .on_hover_text(
                            "Show / hide the slice intersection (crosshair).\n\
                             Hidden: left click does not navigate - slices change \
                             only by scrolling each view",
                        )
                        .clicked()
                    {
                        self.show_crosshair = !self.show_crosshair;
                    }

                    // Syncing needs a second workspace to sync with, and
                    // nothing else: it carries the slice, the zoom, the pan
                    // and both players as well as the crosshair, so it stays
                    // on the bar whether or not the crosshair is shown.
                    if self.both_volumes()
                        && ui
                            .add(egui::Button::selectable(self.link_studies, "Sync"))
                            .on_hover_text(
                                "Keep workspaces A and B together: the crosshair, the slice, \
                                 a scrolled or played slice, the zoom and the pan of a view \
                                 carry over to the other workspace's view of the same plane, \
                                 through the active registration when there is one, and a \
                                 4D run walks both groups at once, so the two rows show the \
                                 same thing at the same scale.\n\
                                 Off: each workspace is navigated on its own",
                            )
                            .clicked()
                    {
                        self.link_studies = !self.link_studies;
                    }

                    // Reset every view of both workspaces.
                    if tip_button(
                        ui,
                        "⟲",
                        "Reset every view of both workspaces: fit zoom, clear pan \
                         and put the crosshairs back at the volume centers",
                    ) {
                        self.reset_all_views();
                    }

                    // The drawing tools: one toggle that unfolds the row of
                    // tools and the options of the one in hand, right here,
                    // so a hand that is drawing never leaves the toolbar.
                    ui.separator();
                    let paintable = self.any_volume();
                    if !paintable {
                        self.seg_tool = SegTool::None;
                        self.draw_open = false;
                    }
                    if ui
                        .add_enabled(
                            paintable,
                            egui::Button::selectable(self.draw_open, "✏ Draw structure"),
                        )
                        .on_hover_text(
                            "Unfold the drawing tools: paint, erase and grow a segmentation; \
                             polygon, spline, freehand, live wire, brush and nudge for an RT \
                             structure. Fold them away to put the tool down",
                        )
                        .clicked()
                    {
                        self.draw_open = !self.draw_open;
                        if !self.draw_open {
                            self.set_seg_tool(SegTool::None);
                        }
                    }
                    if self.draw_open {
                        self.draw_strip(ui);
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // What is on screen, in row order. A workspace put
                    // aside is still loaded, but naming it here would be
                    // naming a row that is not there.
                    let mut parts = Vec::new();
                    for i in self.open_slots() {
                        if let Some(study) = &self.slots[i].study {
                            let m = &study.meta;
                            parts.push(format!(
                                "{}: {} {}",
                                SLOT_NAMES[i],
                                m.patient_name.replace('^', " "),
                                m.study_date
                            ));
                        }
                    }
                    ui.label(egui::RichText::new(parts.join("   ")).weak());
                });
            });
        });
    }

    // -- Status bar -------------------------------------------------------
    pub(super) fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom(egui::Id::new("status")).show(ui, |ui| {
            ui.horizontal(|ui| {
                let any = self.slots.iter().any(|s| s.study.is_some());
                if !any {
                    ui.weak("No data loaded");
                    return;
                }
                let comparing = self.comparing();
                for slot in self.open_slots() {
                    let slot_name = SLOT_NAMES[slot];
                    let s = &self.slots[slot];
                    let Some(study) = &s.study else { continue };
                    if !study.has_volume() {
                        // No voxels, so no position and no value to report.
                        let prefix = if comparing {
                            format!("{slot_name}: ")
                        } else {
                            String::new()
                        };
                        ui.weak(format!("{prefix}no image volume"));
                        continue;
                    }
                    let v = &study.volume;
                    let c = s.cursor;
                    let p = v.voxel_to_patient(c[0], c[1], c[2]);
                    let prefix = if comparing {
                        format!("{slot_name}: ")
                    } else {
                        String::new()
                    };
                    // Both workspaces report in full: each one's own cursor is
                    // a real position in its own volume, whether it was
                    // clicked there or followed the other one.
                    ui.monospace(format!(
                        "{}({:6.1},{:6.1},{:6.1})mm ijk({:3},{:3},{:3})",
                        prefix,
                        p.x,
                        p.y,
                        p.z,
                        c[0].round() as i64,
                        c[1].round() as i64,
                        c[2].round() as i64
                    ));
                    if let Some(hu) = v.get(
                        c[0].round() as i64,
                        c[1].round() as i64,
                        c[2].round() as i64,
                    ) {
                        // Hounsfield units are a CT's; an MR's or a PET's
                        // values are the scanner's, with no unit to name.
                        let m = study
                            .series
                            .get(study.active_series)
                            .map(|s| s.modality.trim().to_uppercase())
                            .unwrap_or_default();
                        ui.monospace(match m.as_str() {
                            "CT" | "" => format!("{hu:5} HU"),
                            other => format!("{other} value {hu}"),
                        });
                    }
                    if let Some(d) = study.doses.get(s.active_dose).and_then(|d| d.sample(p)) {
                        ui.monospace(format!(
                            "{:.2} Gy ({:.0}%)",
                            d,
                            100.0 * d / s.dose_reference.max(1e-6)
                        ));
                    }
                    if comparing {
                        ui.separator();
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // The readouts are what the bar is for, so the mouse
                    // bindings fold into a single "?" that the pointer opens -
                    // always the bindings of the tool in force.
                    let hint = super::struct_tools::mouse_hint(self.seg_tool);
                    // `Sense::hover`: it looks like a button and answers the
                    // pointer, but there is nothing to click.
                    ui.add(egui::Button::new("?").small().sense(egui::Sense::hover()))
                        .on_hover_text(hint);
                });
            });
        });
    }
}
