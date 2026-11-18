//! The interface itself, run without a window (`tests/common/headless.rs`):
//! the application is built the way `main.rs` builds it, stepped frame by
//! frame, and asked what it drew.
//!
//! These read the tree only - which widgets are there, with what text and
//! state - so no graphics adapter is opened and they run wherever the other
//! suites do. The one test that paints a frame is `#[ignore]`d like the
//! other GPU tests:
//!
//! ```text
//! cargo test --release --test ui
//! RDS_GPU_TESTS=1 cargo test --release --test ui -- --ignored gpu
//! ```
//!
//! **One application at a time.** Every test builds a whole `ViewerApp`,
//! with its job threads and its settings file, so they take turns behind
//! [`LOCK`]; and the settings they read and write are a scratch folder's
//! (`target/ui-test-state`), pointed at once, before the first of them
//! starts, and emptied at every turn: each test begins from a first start
//! of the program, whatever ran before it (the modules ticked, the last
//! session and the folded sections are all settings). Under nextest every
//! test is a process of its own and the lock orders nothing between them,
//! so `.config/nextest.toml` runs this suite one test at a time, as it does
//! the other suites that regenerate a fixture under `target/`.

mod common;
#[path = "common/headless.rs"]
mod headless;

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, Once, OnceLock};

use headless::{Opts, Target, Ui, SETTLE};

static LOCK: Mutex<()> = Mutex::new(());
static INIT: Once = Once::new();

/// Take the turn, and on the first one move the settings to the scratch
/// folder: every other test is waiting for the lock, so nothing reads the
/// environment while it is written.
fn turn() -> MutexGuard<'static, ()> {
    let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let state = common::target_dir("ui-test-state");
    INIT.call_once(|| headless::isolate_user_dirs(&state));
    guard
}

/// The three-phase phantom every suite uses, written once.
fn phantom() -> PathBuf {
    static FOLDER: OnceLock<PathBuf> = OnceLock::new();
    FOLDER
        .get_or_init(|| common::fourd_folder(&common::target_dir("ui-test-data"), [0.0, 5.0, 10.0]))
        .clone()
}

/// The application with nothing loaded.
fn start_screen() -> Ui {
    let mut ui = Ui::open(Opts::default());
    assert!(ui.settle(SETTLE), "the start screen did not settle");
    ui
}

/// The application with the phantom in workspace A, loaded.
fn with_phantom(gpu: bool) -> Ui {
    let mut ui = Ui::open(Opts {
        datasets: [Some(phantom()), None],
        gpu,
        ..Opts::default()
    });
    assert!(ui.settle(SETTLE), "the phantom did not finish loading");
    ui
}

#[test]
fn start_screen_offers_its_actions() {
    let _turn = turn();
    let ui = start_screen();
    assert!(ui.has_label("No data loaded"));
    for action in ["Add DICOM folder", "PACS", "Generate test data"] {
        assert!(
            ui.has(&Target::containing(action).role("button")),
            "no button for {action}:\n{}",
            ui.tree_text()
        );
    }
    // Nothing was ever open in settings that are new.
    let restore = Target::containing("Restore last session").role("button");
    assert!(ui.find(&restore).unwrap().disabled);
}

#[test]
fn modules_menu_opens_on_a_click() {
    let _turn = turn();
    let mut ui = start_screen();
    assert!(!ui.has_label("Dose-volume histograms"));
    ui.click(&Target::label("Modules").role("button")).unwrap();
    ui.settle(SETTLE);
    for module in [
        "Image registration",
        "Structure editor",
        "Dose-volume histograms",
        "Workflows",
    ] {
        assert!(ui.has_label(module), "{module} is not in the menu");
    }
}

#[test]
fn phantom_loads_with_its_structures() {
    let _turn = turn();
    let ui = with_phantom(false);
    assert!(!ui.has_label("No data loaded"));
    assert!(ui.has_label("Workspace A"));
    for roi in ["TARGET", "BODY", "CORD"] {
        assert!(
            ui.has(&Target::containing(roi)),
            "{roi} is not in the data tree:\n{}",
            ui.tree_text()
        );
    }
}

#[test]
fn f9_folds_the_data_tree_away_and_back() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    assert!(ui.has_label("Workspace A"));
    ui.key("F9").unwrap();
    ui.settle(SETTLE);
    assert!(!ui.has_label("Workspace A"));
    ui.key("F9").unwrap();
    ui.settle(SETTLE);
    assert!(ui.has_label("Workspace A"));
}

#[test]
fn module_section_appears_when_ticked_in_the_menu() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    // Steps as a script would give them, each waiting for the interface to
    // settle. The menu's own entry is gone once the menu is closed, so what
    // is left with that text is the section's header in the right panel.
    let out = common::target_dir("ui-test-out");
    for step in [
        r#"{"click": {"label": "Modules", "role": "button"}}"#,
        r#"{"click": "Structure details"}"#,
        r#"{"key": "Escape"}"#,
        r#"{"expect_none": "Structure editor"}"#,
        r#"{"expect": "Structure details"}"#,
    ] {
        ui.run_step(&serde_json::from_str(step).unwrap(), &out)
            .unwrap_or_else(|e| panic!("{step}: {e:#}\n{}", ui.tree_text()));
    }
}

#[test]
fn tool_window_opens_inside_the_main_one_and_closes() {
    let _turn = turn();
    let mut ui = start_screen();
    let window = Target::containing("Generate synthetic RT test study").role("window");
    assert!(!ui.has(&window));
    ui.click(&Target::containing("Generate test data").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    assert!(ui.has(&window), "no tool window:\n{}", ui.tree_text());
    // The form's defaults are readable as values.
    assert!(ui.has(&Target::label("SynthProton").role("textinput")));
    ui.click(&Target::label("Close window")).unwrap();
    ui.settle(SETTLE);
    assert!(!ui.has(&window));
}

#[test]
fn a_missing_widget_is_reported_with_what_is_there() {
    let _turn = turn();
    let mut ui = start_screen();
    let err = ui.click(&Target::label("Module")).unwrap_err().to_string();
    assert!(err.contains("button \"Modules\""), "{err}");
    let step = serde_json::json!({"press": "Modules"});
    let err = ui
        .run_step(&step, &std::env::temp_dir())
        .unwrap_err()
        .to_string();
    assert!(err.contains("not a step"), "{err}");
}

/// The fifteen modules, by the titles their sections carry.
const MODULES: [&str; 15] = [
    "Image information",
    "Image playback",
    "Image registration",
    "Image simulation",
    "Image generation",
    "Structure editor",
    "Structure auto tools",
    "Structure propagation",
    "Structure comparison",
    "Structure details",
    "Structure motion",
    "Dose and plan display",
    "Dose analysis",
    "Dose-volume histograms",
    "Workflows",
];

/// Fail with the list when a widget on screen says nothing in words.
fn assert_all_named(ui: &Ui, screen: &str) {
    let unnamed = ui.unnamed();
    assert!(
        unnamed.is_empty(),
        "{screen}: {} widget(s) without a name (see `Named` in src/app/widgets.rs):\n{}",
        unnamed.len(),
        unnamed.join("\n")
    );
}

#[test]
fn every_widget_on_the_main_screens_has_a_name() {
    let _turn = turn();
    assert_all_named(&start_screen(), "the start screen");
    let ui = with_phantom(false);
    assert_all_named(&ui, "the views and the data tree");
    // The ones several panes and rows share say which they are.
    for name in [
        "Axial slice",
        "Maximize Sagittal",
        "Maximize Coronal",
        "Show TARGET",
        "Colour of CORD",
    ] {
        assert!(ui.has_label(name), "no \"{name}\":\n{}", ui.tree_text());
    }
}

#[test]
fn every_widget_of_the_module_sections_has_a_name() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    ui.click(&Target::label("Modules").role("button")).unwrap();
    ui.settle(SETTLE);
    for module in MODULES {
        ui.click(&Target::label(module).role("checkbox")).unwrap();
        ui.settle(SETTLE);
    }
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);
    // From the last to the first: a section that unfolds moves only the
    // ones below it, which are done by then.
    for module in MODULES.iter().rev() {
        // Two kinds of header: the sections' own, whose arrow is called
        // *Unfold X* and then *Fold X*, and egui's, a button with the title.
        let arrow = Target::label(&format!("Unfold {module}"));
        let before = ui.tree_text().lines().count();
        if ui.has(&arrow) {
            ui.click(&arrow).unwrap();
            ui.settle(SETTLE);
            assert!(ui.has_label(&format!("Fold {module}")));
        } else {
            ui.click(&Target::label(module).role("button")).unwrap();
            ui.settle(SETTLE);
        }
        assert!(
            ui.tree_text().lines().count() > before,
            "{module} did not unfold:\n{}",
            ui.tree_text()
        );
    }
    assert_all_named(&ui, "the module sections, unfolded");
}

/// *Dose and plan display*: the colorwash's scale (map, range, steps,
/// bands) and the isodose ladder are there and named, a level can be
/// removed and added, the map changed, and what was chosen is written to
/// the settings.
#[test]
fn the_dose_colour_scale_and_levels_can_be_edited() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let out = common::target_dir("ui-test-out");
    for step in [
        r#"{"click": {"label": "Modules", "role": "button"}}"#,
        r#"{"click": {"label": "Dose and plan display", "role": "checkbox"}}"#,
        r#"{"key": "Escape"}"#,
    ] {
        ui.run_step(&serde_json::from_str(step).unwrap(), &out)
            .unwrap_or_else(|e| {
                panic!(
                    "{step}: {e:#}
{}",
                    ui.tree_text()
                )
            });
    }
    let arrow = Target::label("Unfold Dose and plan display");
    if ui.has(&arrow) {
        ui.invoke(&arrow).unwrap();
    } else {
        ui.invoke(&Target::label("Dose and plan display").role("button"))
            .unwrap();
    }
    ui.settle(SETTLE);
    for name in [
        "Colour map",
        "Rainbow",
        "From",
        "Steps",
        "Stepped bands",
        "Limits in percent of each workspace's reference",
        "Colour of the 107 % level",
        "Show the 107 % level",
        "Remove the 30 % level",
        "Default levels",
        "Levels from the map",
        "Map from the levels",
        "Colour scale",
        "Isodose levels",
    ] {
        assert!(
            ui.has(&Target::containing(name)),
            "no \"{name}\":
{}",
            ui.tree_text()
        );
    }
    assert!(ui.has(&Target::label("Gy").role("button")));
    // The tree has a *Remove the ticked structures* of its own.
    let removes = |ui: &headless::Ui| {
        ui.find_all(&Target::containing("Remove the ").role("button"))
            .iter()
            .filter(|h| h.text.ends_with("% level"))
            .count()
    };
    assert_eq!(removes(&ui), 8, "the eight default levels");
    assert_all_named(&ui, "the dose display section");

    // One level less, one more halfway under the new lowest (50 %).
    ui.invoke(&Target::label("Remove the 30 % level")).unwrap();
    ui.settle(SETTLE);
    assert!(!ui.has_label("Remove the 30 % level"));
    assert_eq!(removes(&ui), 7);
    ui.invoke(&Target::containing("Add level").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    assert!(ui.has_label("Remove the 25 % level"), "{}", ui.tree_text());

    // Another map, and banded.
    ui.invoke(&Target::label("Rainbow")).unwrap();
    ui.settle(SETTLE);
    ui.invoke(&Target::label("Viridis")).unwrap();
    ui.settle(SETTLE);
    ui.invoke(&Target::label("Stepped bands").role("checkbox"))
        .unwrap();
    ui.settle(SETTLE);
    assert!(ui.has_label("Viridis"), "{}", ui.tree_text());

    let saved = rust_dicom_station::settings::load().dose_colors;
    assert_eq!(
        saved.map,
        rust_dicom_station::settings::DoseColormap::Viridis
    );
    assert!(saved.stepped);
    let pcts: Vec<f32> = saved.levels.iter().map(|l| l.pct).collect();
    assert!(pcts.contains(&25.0) && !pcts.contains(&30.0), "{pcts:?}");

    // The levels coloured as the (stepped viridis) wash colours their
    // doses: 107 % is past the scale's top, so its last band; then the wash
    // in the levels' colours, after which there is nothing to sync from.
    ui.invoke(&Target::label("Levels from the map")).unwrap();
    ui.settle(SETTLE);
    let saved = rust_dicom_station::settings::load().dose_colors;
    assert_eq!(saved.levels[0].pct, 107.0);
    assert_eq!(saved.levels[0].color, [253, 231, 37]);
    ui.invoke(&Target::label("Map from the levels")).unwrap();
    ui.settle(SETTLE);
    assert_eq!(
        rust_dicom_station::settings::load().dose_colors.map,
        rust_dicom_station::settings::DoseColormap::Isodose
    );
    assert!(
        ui.find(&Target::label("Levels from the map"))
            .unwrap()
            .disabled
    );
}

/// The dose legend: on no view at first, put on one by the button on that
/// view's own bar (there on a folded bar too, the default) and taken off
/// again, each time remembered; its two parts are the section's.
#[test]
fn the_dose_legend_is_switched_on_the_view_and_remembered() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let views = || rust_dicom_station::settings::load().dose_legend.views;
    assert_eq!(views(), [false; 4], "off on every view at first");
    let coronal = Target::label("Dose legend on Coronal");
    ui.click(&coronal).unwrap_or_else(|e| {
        panic!(
            "{e:#}
{}",
            ui.tree_text()
        )
    });
    ui.settle(SETTLE);
    assert_eq!(views(), [false, false, true, false]);
    ui.click(&coronal).unwrap();
    ui.settle(SETTLE);
    assert_eq!(views(), [false; 4]);

    for step in [
        r#"{"click": {"label": "Modules", "role": "button"}}"#,
        r#"{"click": {"label": "Dose and plan display", "role": "checkbox"}}"#,
        r#"{"key": "Escape"}"#,
        r#"{"invoke": "Unfold Dose and plan display"}"#,
        r#"{"invoke": {"label": "Isodose levels", "role": "checkbox"}}"#,
    ] {
        ui.run_step(
            &serde_json::from_str(step).unwrap(),
            &common::target_dir("ui-test-out"),
        )
        .unwrap_or_else(|e| {
            panic!(
                "{step}: {e:#}
{}",
                ui.tree_text()
            )
        });
    }
    ui.settle(SETTLE);
    let legend = rust_dicom_station::settings::load().dose_legend;
    assert!(legend.scale && !legend.levels, "{legend:?}");
}

#[test]
fn every_widget_of_the_tool_windows_has_a_name() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    // Every window is looked at before anything fails, so one run lists
    // them all.
    let mut unnamed = Vec::new();
    for (menu, entry) in [
        ("Tools", "PACS"),
        ("Tools", "Downloaded models"),
        ("Tools", "Anonymize DICOM folder"),
        ("Tools", "Generate test data"),
        ("Tools", "Download test data"),
        ("Save", "Export DICOM"),
        ("Save", "Save image"),
    ] {
        ui.click(&Target::label(menu).role("button")).unwrap();
        ui.settle(SETTLE);
        ui.click(&Target::containing(entry).role("button")).unwrap();
        ui.settle(SETTLE);
        let close = Target::label("Close window");
        assert!(
            ui.has(&close),
            "{entry} opened no window:\n{}",
            ui.tree_text()
        );
        unnamed.extend(ui.unnamed().into_iter().map(|w| format!("{entry}: {w}")));
        ui.click(&close).unwrap();
        ui.settle(SETTLE);
        assert!(!ui.has(&close), "{entry} did not close");
    }
    assert!(
        unnamed.is_empty(),
        "{} widget(s) without a name (see `Named` in src/app/widgets.rs):\n{}",
        unnamed.len(),
        unnamed.join("\n")
    );
}

#[test]
fn the_save_menu_holds_every_way_out_and_one_pane_can_be_saved() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    // File opens and clears; what writes is under Save.
    ui.click(&Target::label("File").role("button")).unwrap();
    ui.settle(SETTLE);
    assert!(!ui.has(&Target::containing("Export DICOM").role("button")));
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("Save").role("button")).unwrap();
    ui.settle(SETTLE);
    for entry in [
        "Export DICOM",
        "Save to the PACS",
        "Save image",
        "Image of one pane",
    ] {
        assert!(
            ui.has(&Target::containing(entry).role("button")),
            "{entry} is not in the Save menu:\n{}",
            ui.tree_text()
        );
    }
    // The dialog lists the panes on screen by name, one choice each.
    let coronal = ui.find_all(&Target::label("Coronal")).len();
    ui.click(&Target::containing("Save image").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    assert!(ui.has_label("Pane"), "no pane choice:\n{}", ui.tree_text());
    assert_eq!(
        ui.find_all(&Target::label("Coronal")).len(),
        coronal + 1,
        "the coronal pane is not offered on its own:\n{}",
        ui.tree_text()
    );
}

#[test]
fn a_series_saved_from_its_menu_opens_the_export_with_it_alone() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let row = ui
        .find(&Target::containing(" sl.)").role("button"))
        .unwrap_or_else(|e| panic!("no series row: {e:#}\n{}", ui.tree_text()));
    ui.click_at(row.rect.center(), egui::PointerButton::Secondary, 1);
    ui.settle(SETTLE);
    ui.click(&Target::containing("Save as DICOM").role("button"))
        .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
    ui.settle(SETTLE);
    assert!(
        ui.has(&Target::containing(
            "1 image series, 0 RT object(s) selected"
        )),
        "not the series alone:\n{}",
        ui.tree_text()
    );
}

/// What was typed into an open Export window survives a tree's *Save as
/// DICOM*: only the ticks change.
#[test]
fn a_tree_save_keeps_what_was_typed_into_the_open_export_window() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let save_series = |ui: &mut Ui| {
        let row = ui
            .find(&Target::containing(" sl.)").role("button"))
            .unwrap_or_else(|e| panic!("no series row: {e:#}\n{}", ui.tree_text()));
        ui.click_at(row.rect.center(), egui::PointerButton::Secondary, 1);
        ui.settle(SETTLE);
        ui.click(&Target::containing("Save as DICOM").role("button"))
            .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
        ui.settle(SETTLE);
    };
    ui.click(&Target::label("Save").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Export DICOM").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    let field = ui
        .find(&Target::label("PHANTOM^RT").role("textinput"))
        .unwrap_or_else(|e| panic!("no patient name field: {e:#}\n{}", ui.tree_text()));
    // Past the end of the name, then a pseudonym's tail.
    ui.click_at(
        field.rect.right_center() - egui::vec2(4.0, 0.0),
        egui::PointerButton::Primary,
        1,
    );
    ui.settle(SETTLE);
    ui.key("End").unwrap();
    ui.type_text("-PSEUDO");
    ui.settle(SETTLE);
    assert!(
        ui.has(&Target::label("PHANTOM^RT-PSEUDO").role("textinput")),
        "the name was not typed:\n{}",
        ui.tree_text()
    );
    // The window covers the data tree: aside by its title bar, as a person
    // would, so the tree's rows can be reached.
    let win = ui
        .find(&Target::containing("Export DICOM").role("window"))
        .unwrap_or_else(|e| panic!("no export window: {e:#}\n{}", ui.tree_text()));
    let grip = egui::pos2(win.rect.min.x + 200.0, win.rect.min.y + 10.0);
    ui.drag(
        grip,
        grip + egui::vec2(700.0, 0.0),
        egui::PointerButton::Primary,
    );
    ui.settle(SETTLE);
    save_series(&mut ui);
    assert!(
        ui.has(&Target::label("PHANTOM^RT-PSEUDO").role("textinput")),
        "the typed name is gone:\n{}",
        ui.tree_text()
    );
    assert!(
        ui.has(&Target::containing(
            "1 image series, 0 RT object(s) selected"
        )),
        "not the series alone:\n{}",
        ui.tree_text()
    );
}

/// *Save to the PACS* opens at once and settles once the PACS was read:
/// here a PACS never scanned, which says so rather than scanning it.
#[test]
fn save_to_the_pacs_opens_at_once_and_settles_on_what_the_pacs_holds() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    ui.click(&Target::label("Save").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Save to the PACS").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("A · ").role("button"))
        .unwrap_or_else(|e| panic!("no workspace entry: {e:#}\n{}", ui.tree_text()));
    assert!(ui.settle(SETTLE), "the lookup did not land");
    assert!(
        ui.has(&Target::label("Close window")),
        "no window:\n{}",
        ui.tree_text()
    );
    assert!(
        !ui.has_label("reading the PACS"),
        "still reading:\n{}",
        ui.tree_text()
    );
    assert!(
        ui.has_label("PACS not read yet"),
        "the rows do not say the PACS was never scanned:\n{}",
        ui.tree_text()
    );
    let save = ui
        .find(&Target::containing("Save 0 objects").role("button"))
        .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
    assert!(save.disabled, "nothing can be saved");
}

/// Right-click a widget, the way a person opens its menu.
fn right_click(ui: &mut Ui, target: &Target) {
    let hit = ui
        .find(target)
        .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
    ui.click_at(hit.rect.center(), egui::PointerButton::Secondary, 1);
    ui.settle(SETTLE);
}

/// The title of the first node that folds, in tree order, that `pick`
/// takes, as its arrow names it.
fn fold_title(ui: &Ui, mut pick: impl FnMut(&str) -> bool) -> String {
    ui.find_all(&Target::containing("Fold ").role("button"))
        .into_iter()
        .filter_map(|h| h.text.strip_prefix("Fold ").map(str::to_string))
        .find(|t| pick(t))
        .unwrap_or_else(|| panic!("no such fold:\n{}", ui.tree_text()))
}

/// The status bar names the structures drawn at the crosshair, after its
/// value; a structure whose tick box is off is not drawn, and not named.
#[test]
fn the_readout_names_the_structures_at_the_crosshair() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    right_click(&mut ui, &Target::containing("TARGET  [").role("button"));
    ui.click(&Target::containing("Localize").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    let named = Target::containing("TARGET").role("label");
    let hit = ui
        .find(&named)
        .unwrap_or_else(|e| panic!("no name at the crosshair: {e:#}\n{}", ui.tree_text()));
    let hu = ui.find(&Target::containing(" HU").role("label")).unwrap();
    assert!(
        hit.rect.left() > hu.rect.left(),
        "the names come after the value"
    );
    ui.click(&Target::label("Show TARGET")).unwrap();
    ui.settle(SETTLE);
    assert!(
        !ui.has(&named),
        "a hidden structure is still named:\n{}",
        ui.tree_text()
    );
}

/// A structure set folds its structures away under its own arrow, as a
/// modality folds its series.
#[test]
fn a_structure_set_folds_its_structures() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let row = Target::containing("TARGET  [").role("button");
    assert!(ui.has(&row));
    let set = fold_title(&ui, |t| t.contains('▶'));
    ui.click(&Target::label(&format!("Fold {set}"))).unwrap();
    ui.settle(SETTLE);
    assert!(!ui.has(&row), "still listed:\n{}", ui.tree_text());
    assert!(ui.has(&Target::label(&set)), "the set's own row went too");
    ui.click(&Target::label(&format!("Unfold {set}"))).unwrap();
    ui.settle(SETTLE);
    assert!(ui.has(&row), "not listed again:\n{}", ui.tree_text());
}

/// A study this station's PACS does not hold (here, a PACS never
/// scanned) has no *Save changes to the PACS*, once the lookup has landed.
#[test]
fn no_save_changes_entry_on_a_study_the_pacs_does_not_hold() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let mut nodes = 0;
    let study = fold_title(&ui, |_| {
        nodes += 1;
        nodes == 2
    });
    right_click(&mut ui, &Target::label(&study));
    assert!(ui.has(&Target::containing("Save as DICOM")), "no menu");
    assert!(
        !ui.has(&Target::containing("Save changes to the PACS")),
        "{}",
        ui.tree_text()
    );
}

/// *Save changes to the PACS* on a study this station's PACS holds: greyed
/// while nothing changed; after a structure is removed, it opens *Save to
/// the PACS* on that study with the edited set ticked, and the save lands.
#[test]
fn save_changes_to_the_pacs_opens_on_the_study_with_what_changed() {
    let _turn = turn();
    // The scratch settings' own PACS, emptied however the test ends: the
    // other tests expect one never scanned.
    struct Clear(PathBuf);
    impl Drop for Clear {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let archive = rust_dicom_station::archive::Archive::station();
    let _clear = Clear(archive.root().to_path_buf());
    archive
        .import(
            &phantom(),
            &rust_dicom_station::progress::Progress::default(),
        )
        .expect("the phantom goes into the PACS");
    let mut ui = with_phantom(false);
    // The patient folds first, its study next.
    let mut nodes = 0;
    let study = fold_title(&ui, |_| {
        nodes += 1;
        nodes == 2
    });
    let entry = Target::containing("Save changes to the PACS").role("button");

    right_click(&mut ui, &Target::label(&study));
    let hit = ui.find(&entry).unwrap_or_else(|e| {
        panic!(
            "no entry on a study the PACS holds: {e:#}\n{}",
            ui.tree_text()
        )
    });
    assert!(hit.disabled, "nothing was changed: {}", hit.text);
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);

    // One structure out (the others unticked first, or the menu would act
    // on every ticked one).
    ui.click(&Target::label("None").role("button")).unwrap();
    ui.settle(SETTLE);
    right_click(&mut ui, &Target::containing("CORD  [").role("button"));
    ui.click(&Target::containing("🗑 Remove").role("button"))
        .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
    ui.settle(SETTLE);
    assert!(!ui.has(&Target::containing("CORD  [")));

    right_click(&mut ui, &Target::label(&study));
    let hit = ui.find(&entry).unwrap();
    assert!(!hit.disabled, "the edit is not seen");
    assert!(hit.text.ends_with("(1)"), "{}", hit.text);
    ui.click(&entry).unwrap();
    assert!(ui.settle(SETTLE), "the lookup did not land");
    assert!(
        ui.has(&Target::containing("What was made or changed in the study")),
        "not scoped:\n{}",
        ui.tree_text()
    );
    assert!(ui.has_label("changed"), "{}", ui.tree_text());
    let save = Target::label("📤 Save 1 object").role("button");
    assert!(!ui.find(&save).unwrap().disabled, "{}", ui.tree_text());
    ui.click(&save).unwrap();
    assert!(ui.settle(SETTLE), "the save did not land");
    assert!(
        ui.has(&Target::containing("1 object saved into the PACS")),
        "{}",
        ui.tree_text()
    );
}

/// Saving a set changed twice: the second save offers *Update* the series
/// the first one wrote (or the set came from), ticked by default beside
/// *Save as a new series*, and makes no further series in the PACS.
#[test]
fn a_set_saved_twice_updates_its_series_rather_than_adding_one() {
    let _turn = turn();
    struct Clear(PathBuf);
    impl Drop for Clear {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let archive = rust_dicom_station::archive::Archive::station();
    let _clear = Clear(archive.root().to_path_buf());
    archive
        .import(
            &phantom(),
            &rust_dicom_station::progress::Progress::default(),
        )
        .expect("the phantom goes into the PACS");
    let structure_series = || {
        archive
            .tree()
            .patients()
            .iter()
            .flat_map(|p| p.studies.iter())
            .flat_map(|s| s.series.iter())
            .filter(|s| s.modality == "RTSTRUCT")
            .count()
    };
    let mut ui = with_phantom(false);
    let mut nodes = 0;
    let study = fold_title(&ui, |_| {
        nodes += 1;
        nodes == 2
    });
    let entry = Target::containing("Save changes to the PACS").role("button");
    let save = Target::label("📤 Save 1 object").role("button");
    ui.click(&Target::label("None").role("button")).unwrap();
    ui.settle(SETTLE);

    let mut after_first = 0;
    for (round, roi) in ["CORD  [", "BODY  ["].into_iter().enumerate() {
        right_click(&mut ui, &Target::containing(roi).role("button"));
        ui.click(&Target::containing("🗑 Remove").role("button"))
            .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
        ui.settle(SETTLE);
        right_click(&mut ui, &Target::label(&study));
        ui.click(&entry)
            .unwrap_or_else(|e| panic!("round {round}: {e:#}\n{}", ui.tree_text()));
        assert!(ui.settle(SETTLE), "the lookup did not land");
        if round == 1 {
            let update = ui
                .find(&Target::containing("Update RTSTRUCT"))
                .unwrap_or_else(|e| panic!("no update offered: {e:#}\n{}", ui.tree_text()));
            assert_eq!(
                update.checked,
                Some(true),
                "not the default: {}",
                update.text
            );
            assert!(
                ui.has(&Target::containing("Save as a new series")),
                "{}",
                ui.tree_text()
            );
        }
        ui.click(&save)
            .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
        assert!(ui.settle(SETTLE), "the save did not land");
        assert!(
            ui.has(&Target::containing("1 object saved into the PACS")),
            "round {round}:\n{}",
            ui.tree_text()
        );
        if round == 0 {
            after_first = structure_series();
            // The window over the views goes, for the next edit.
            ui.click(&Target::label("Close window")).unwrap();
            ui.settle(SETTLE);
        }
    }
    assert!(
        ui.has(&Target::containing("into the series it came from")),
        "{}",
        ui.tree_text()
    );
    assert_eq!(structure_series(), after_first, "the update made no series");
}

/// *Tools ▸ Posture model* on the articulated phantom (two bones on a
/// hinge, `gen_test_data::articulated_pair`): the window opens, the anchor
/// is picked (it is Auto until then), the model builds, its verdict heads
/// the side, its tables name the bodies and the joint, the *Display* menu
/// holds the live slices and the wash, stepping to the end reads the
/// hinge's angle, playing moves the time, and nothing in the window is
/// unnamed.
#[test]
fn the_posture_window_builds_steps_and_plays_the_phantom() {
    let _turn = turn();
    static FOLDERS: OnceLock<[PathBuf; 2]> = OnceLock::new();
    let [a, b] = FOLDERS
        .get_or_init(|| {
            rust_dicom_station::gen_test_data::generate_articulated(
                &common::target_dir("ui-test-posture"),
                &rust_dicom_station::gen_test_data::ArticulatedParams::default(),
                &rust_dicom_station::progress::Progress::default(),
            )
            .expect("the phantom is written")
        })
        .clone();
    let mut ui = Ui::open(Opts {
        datasets: [Some(a), Some(b)],
        ..Opts::default()
    });
    assert!(ui.settle(SETTLE), "the two states did not load");
    fn fail(ui: &Ui, what: &str) -> ! {
        panic!(
            "{what}:
{}",
            ui.tree_text()
        )
    }
    ui.click(&Target::label("Tools").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Posture model").role("button"))
        .unwrap_or_else(|e| fail(&ui, &format!("{e:#}")));
    ui.settle(SETTLE);
    if !ui.has(&Target::containing("Posture model").role("window")) {
        fail(&ui, "no posture window");
    }
    // The anchor is Auto until one is picked: here the upper bone.
    ui.click(&Target::label("Auto")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("BONE_UPPER")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Build").role("button"))
        .unwrap();
    if !ui.settle(SETTLE) {
        fail(&ui, "the model did not build");
    }
    // The verdict heads the side; the rest of the options wait in menus.
    if !ui.has(&Target::containing(" model").role("label")) {
        fail(&ui, "no verdict");
    }
    for name in [
        "States",
        "Structures",
        "Bones and joints",
        "BONE_LOWER",
        "ORGAN",
    ] {
        if !ui.has(&Target::containing(name)) {
            fail(&ui, &format!("no {name}"));
        }
    }
    assert!(ui.has_label("t 0.00"));
    // The model's workspace is the window's, not a row of the main screen.
    assert!(!ui.has_label("Workspace C"), "{}", ui.tree_text());
    // The WEPL column along the default beam; under Display, live slices
    // by default and the Jacobian wash a tick away.
    for name in ["WEPL mm", "dWEPL mm"] {
        if !ui.has(&Target::containing(name)) {
            fail(&ui, &format!("no {name}"));
        }
    }
    ui.click(&Target::label("Display").role("button"))
        .unwrap_or_else(|e| fail(&ui, &format!("{e:#}")));
    ui.settle(SETTLE);
    if !ui.has(&Target::containing("Live slices")) {
        fail(&ui, "no Live slices");
    }
    ui.click(&Target::label("Jacobian wash"))
        .unwrap_or_else(|e| fail(&ui, &format!("{e:#}")));
    assert!(ui.settle(SETTLE), "the wash did not land");
    assert_all_named(&ui, "the posture window's Display menu");
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);

    // At the end, the lower bone has turned through the hinge's 40 degrees.
    ui.click(&Target::label("To the last state")).unwrap();
    assert!(ui.settle(SETTLE));
    assert!(ui.has_label("t 1.00"), "{}", ui.tree_text());
    let angles: Vec<f64> = ui
        .find_all(&Target::containing("."))
        .iter()
        .filter(|h| h.role == "label")
        .filter_map(|h| h.text.trim().parse::<f64>().ok())
        .collect();
    assert!(
        angles.iter().any(|v| (v.abs() - 40.0).abs() < 2.0),
        "no 40 degree reading in {angles:?}"
    );

    // Playing moves the time along the key frames.
    ui.click(&Target::label("To the first state")).unwrap();
    assert!(ui.settle(SETTLE));
    ui.click(&Target::label("Play the path")).unwrap();
    for _ in 0..90 {
        ui.step();
    }
    ui.click(&Target::label("Pause")).unwrap();
    ui.settle(SETTLE);
    assert!(
        !ui.has_label("t 0.00"),
        "the run did not move:
{}",
        ui.tree_text()
    );
    assert_all_named(&ui, "the posture window");
}

/// A state that lacks structures the other has: the window says which, and
/// offers the automatic model that makes them (not run here: it would
/// download its weights).
#[test]
fn the_posture_window_offers_a_model_for_missing_structures() {
    let _turn = turn();
    static FOLDERS: OnceLock<[PathBuf; 2]> = OnceLock::new();
    let [a, b] = FOLDERS
        .get_or_init(|| {
            use rust_dicom_station::gen_test_data::{articulated_pair, ArticulatedParams};
            let dir = common::target_dir("ui-test-posture-missing");
            let mut pair = articulated_pair(&ArticulatedParams::default());
            // The organ is "liver" on A, and B has none.
            for r in &mut pair[0].structure_sets[0].rois {
                if r.name == "ORGAN" {
                    r.name = "liver".into();
                }
            }
            pair[1].structure_sets[0].rois.retain(|r| r.name != "ORGAN");
            let out = [dir.join("state_a"), dir.join("state_b")];
            for (study, folder) in pair.iter().zip(&out) {
                if folder.exists() {
                    std::fs::remove_dir_all(folder).unwrap();
                }
                let ep = rust_dicom_station::dicom_export::ExportParams::for_study(study);
                rust_dicom_station::dicom_export::export_study(
                    study,
                    folder,
                    &ep,
                    &rust_dicom_station::progress::Progress::default(),
                )
                .expect("the state is written");
            }
            out
        })
        .clone();
    let mut ui = Ui::open(Opts {
        datasets: [Some(a), Some(b)],
        ..Opts::default()
    });
    assert!(ui.settle(SETTLE), "the two states did not load");
    ui.click(&Target::label("Tools").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Posture model").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    assert!(
        ui.has(&Target::containing("B lacks 1 structure")),
        "{}",
        ui.tree_text()
    );
    assert!(
        ui.has(&Target::containing("Contour B with").role("button")),
        "{}",
        ui.tree_text()
    );
    assert_all_named(&ui, "the posture window with an offer");
}

/// The eight tools of the *Draw structure* strip, by the names their
/// tooltips give them.
const DRAW_TOOLS: [&str; 8] = [
    "Paint",
    "Erase",
    "Grow",
    "Polygon",
    "Spline",
    "Free",
    "Live wire",
    "Nudge",
];

/// The strip's target line: the picker that names what the tools act on.
fn draw_target(ui: &Ui) -> String {
    ui.find(&Target::containing("▸").role("combobox"))
        .unwrap_or_else(|e| panic!("no target line: {e:#}\n{}", ui.tree_text()))
        .text
}

/// Every tool of the strip is there, under its own name.
fn assert_draw_tools(ui: &Ui, subject: &str) {
    for tool in DRAW_TOOLS {
        assert!(
            ui.has(&Target::label(tool).role("button")),
            "no {tool} with {subject} the subject:\n{}",
            ui.tree_text()
        );
    }
}

/// Make a new segment the strip's subject through its picker (invoked:
/// a popup entry is acted on, not clicked at a position).
fn pick_new_segment(ui: &mut Ui) {
    ui.click(&Target::containing("▸").role("combobox")).unwrap();
    ui.settle(SETTLE);
    ui.invoke(&Target::label("New segmentation"))
        .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
    ui.settle(SETTLE);
}

/// *Draw structure* is one row of tools whatever is selected: the same
/// eight buttons with a structure as the subject and with a segment, the
/// target line naming the subject, and the *Edit:* switch once both kinds
/// are there. Picking a structure in the data tree makes it the subject.
#[test]
fn the_draw_strip_is_one_row_for_structures_and_segments() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    ui.click(&Target::containing("Draw structure").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("Paint").role("button")).unwrap();
    ui.settle(SETTLE);

    ui.click(&Target::containing("TARGET  [").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    assert_draw_tools(&ui, "a structure");
    assert_eq!(draw_target(&ui), "▸ TARGET");
    assert!(!ui.has_label("Edit:"), "a switch with nothing to switch to");
    // The brush's 3D is there for a structure too, and off until asked
    // for: a structure is drawn slice by slice.
    let ball = ui.find(&Target::label("3D").role("button")).unwrap();
    assert_eq!(ball.checked, Some(false), "{}", ui.tree_text());
    assert_all_named(&ui, "the draw strip on a structure");

    pick_new_segment(&mut ui);
    assert_draw_tools(&ui, "a segment");
    assert_eq!(draw_target(&ui), "▸ Seg 1");
    assert!(ui.has_label("Edit:"), "{}", ui.tree_text());
    let seg = ui
        .find(&Target::label("Segmentation").role("button"))
        .unwrap();
    assert_eq!(seg.checked, Some(true));
    let ball = ui.find(&Target::label("3D").role("button")).unwrap();
    assert_eq!(ball.checked, Some(true), "{}", ui.tree_text());
    assert_all_named(&ui, "the draw strip on a segment");

    // The data tree's structure row takes the subject back.
    ui.click(&Target::containing("TARGET  [").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    assert_eq!(draw_target(&ui), "▸ TARGET");
    let structure = ui.find(&Target::label("Structure").role("button")).unwrap();
    assert_eq!(structure.checked, Some(true));
}

/// A Paint stroke in the axial view changes the structure that is the
/// subject - its volume in the editor's readout moves - and Ctrl+Z takes
/// the stroke back; the same stroke with a segment as the subject paints
/// the segment instead, and leaves the structure alone.
#[test]
fn a_paint_stroke_changes_the_subject_structure_or_segment() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let out = common::target_dir("ui-test-out");
    for step in [
        r#"{"click": {"label": "Modules", "role": "button"}}"#,
        r#"{"click": {"label": "Structure editor", "role": "checkbox"}}"#,
        r#"{"key": "Escape"}"#,
    ] {
        ui.run_step(&serde_json::from_str(step).unwrap(), &out)
            .unwrap_or_else(|e| panic!("{step}: {e:#}\n{}", ui.tree_text()));
    }
    let arrow = Target::label("Unfold Structure editor");
    if ui.has(&arrow) {
        ui.invoke(&arrow).unwrap();
    } else {
        ui.invoke(&Target::label("Structure editor").role("button"))
            .unwrap();
    }
    ui.settle(SETTLE);
    ui.click(&Target::containing("TARGET  [").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Draw structure").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("Paint").role("button")).unwrap();
    ui.settle(SETTLE);

    // The editor's one line about the structure: volume, slices, points.
    let readout = |ui: &Ui| -> String {
        ui.find(&Target::containing(" points · ").role("label"))
            .unwrap_or_else(|e| panic!("no readout: {e:#}\n{}", ui.tree_text()))
            .text
    };
    // The axial pane: between its maximize button and its slice slider.
    let stroke = |ui: &mut Ui| {
        let top = ui.find(&Target::label("Maximize Axial")).unwrap().rect;
        let bottom = ui.find(&Target::label("Axial slice")).unwrap().rect;
        let pane = egui::Rect::from_min_max(
            egui::pos2(bottom.left(), top.bottom()),
            egui::pos2(bottom.right(), bottom.top()),
        );
        let at = pane.center() + egui::vec2(pane.width() * 0.15, 0.0);
        ui.drag(
            at,
            at + egui::vec2(25.0, 15.0),
            egui::PointerButton::Primary,
        );
        ui.settle(SETTLE);
    };

    let before = readout(&ui);
    stroke(&mut ui);
    let painted = readout(&ui);
    assert_ne!(before, painted, "the stroke did not change TARGET");
    ui.key("Ctrl+Z").unwrap();
    ui.settle(SETTLE);
    assert_eq!(readout(&ui), before, "Ctrl+Z did not take the stroke back");

    // The same stroke on a segment: the segment's volume moves, the
    // structure's does not.
    pick_new_segment(&mut ui);
    let seg_volume = |ui: &Ui| -> String {
        ui.find(&Target::containing(" voxels").role("label"))
            .unwrap_or_else(|e| panic!("no segment readout: {e:#}\n{}", ui.tree_text()))
            .text
    };
    assert!(
        seg_volume(&ui).ends_with(" 0 voxels"),
        "{}",
        seg_volume(&ui)
    );
    stroke(&mut ui);
    assert!(
        !seg_volume(&ui).ends_with(" 0 voxels"),
        "the stroke did not paint the segment: {}",
        seg_volume(&ui)
    );
    ui.click(&Target::label("Structure").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    assert_eq!(readout(&ui), before, "the segment's stroke changed TARGET");
}

/// With *3D* on, a Paint stroke on a structure is a ball: one drag in the
/// axial view puts contours on the slices above and below the one drawn
/// on, and one Ctrl+Z takes the whole stroke back, every slice of it.
#[test]
fn a_3d_paint_stroke_reaches_the_neighbouring_slices_of_a_structure() {
    let _turn = turn();
    let mut ui = with_phantom(false);
    let out = common::target_dir("ui-test-out");
    for step in [
        r#"{"click": {"label": "Modules", "role": "button"}}"#,
        r#"{"click": {"label": "Structure editor", "role": "checkbox"}}"#,
        r#"{"key": "Escape"}"#,
    ] {
        ui.run_step(&serde_json::from_str(step).unwrap(), &out)
            .unwrap_or_else(|e| panic!("{step}: {e:#}\n{}", ui.tree_text()));
    }
    let arrow = Target::label("Unfold Structure editor");
    if ui.has(&arrow) {
        ui.invoke(&arrow).unwrap();
    } else {
        ui.invoke(&Target::label("Structure editor").role("button"))
            .unwrap();
    }
    ui.settle(SETTLE);
    ui.click(&Target::containing("TARGET  [").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Draw structure").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("Paint").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("3D").role("button")).unwrap();
    ui.settle(SETTLE);
    let ball = ui.find(&Target::label("3D").role("button")).unwrap();
    assert_eq!(ball.checked, Some(true), "{}", ui.tree_text());
    // A new, empty structure, so every slice with a contour is the stroke's.
    ui.click(&Target::containing("▸").role("combobox")).unwrap();
    ui.settle(SETTLE);
    ui.invoke(&Target::label("New RT structure"))
        .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
    ui.settle(SETTLE);
    assert_ne!(draw_target(&ui), "▸ TARGET");

    // How many slices carry a contour, from the editor's readout.
    let readout = |ui: &Ui| -> String {
        ui.find(&Target::containing(" points · ").role("label"))
            .unwrap_or_else(|e| panic!("no readout: {e:#}\n{}", ui.tree_text()))
            .text
    };
    let slices = |ui: &Ui| -> usize {
        let r = readout(ui);
        r.split(" · ")
            .find_map(|part| part.strip_suffix(" slice(s)"))
            .and_then(|n| n.trim().parse().ok())
            .unwrap_or_else(|| panic!("no slice count in {r:?}"))
    };
    let before = readout(&ui);
    assert_eq!(slices(&ui), 0, "{before}");

    let top = ui.find(&Target::label("Maximize Axial")).unwrap().rect;
    let bottom = ui.find(&Target::label("Axial slice")).unwrap().rect;
    let pane = egui::Rect::from_min_max(
        egui::pos2(bottom.left(), top.bottom()),
        egui::pos2(bottom.right(), bottom.top()),
    );
    let at = pane.center() + egui::vec2(pane.width() * 0.15, 0.0);
    ui.drag(
        at,
        at + egui::vec2(25.0, 15.0),
        egui::PointerButton::Primary,
    );
    ui.settle(SETTLE);
    let n = slices(&ui);
    assert!(n > 1, "a 3D stroke reached {n} slice(s): {}", readout(&ui));

    ui.key("Ctrl+Z").unwrap();
    ui.settle(SETTLE);
    assert_eq!(readout(&ui), before, "Ctrl+Z did not take the stroke back");

    // The same ball in the sagittal view, drawn one move a frame so that
    // it is written back while it goes (the axial slices it reached only):
    // it lands on axial slices, and one Ctrl+Z takes all of it back.
    let top = ui.find(&Target::label("Maximize Sagittal")).unwrap().rect;
    let bottom = ui.find(&Target::label("Sagittal slice")).unwrap().rect;
    let pane = egui::Rect::from_min_max(
        egui::pos2(bottom.left(), top.bottom()),
        egui::pos2(bottom.right(), bottom.top()),
    );
    let path: Vec<egui::Pos2> = (0..24)
        .map(|i| pane.center() + egui::vec2(i as f32 * 2.0, i as f32))
        .collect();
    ui.drag_timed(&path, egui::PointerButton::Primary);
    ui.settle(SETTLE);
    let n = slices(&ui);
    assert!(
        n > 1,
        "a sagittal 3D stroke reached {n} slice(s): {}",
        readout(&ui)
    );
    ui.key("Ctrl+Z").unwrap();
    ui.settle(SETTLE);
    assert_eq!(readout(&ui), before, "Ctrl+Z did not take the stroke back");

    // With 3D off the same stroke stays on its slice.
    ui.click(&Target::label("3D").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.drag(
        at,
        at + egui::vec2(25.0, 15.0),
        egui::PointerButton::Primary,
    );
    ui.settle(SETTLE);
    assert_eq!(slices(&ui), 1, "{}", readout(&ui));
}

/// The frame as a picture: the size asked for, and an image in the views
/// rather than an empty window. Needs a graphics adapter (a software one
/// will do).
#[test]
#[ignore]
fn gpu_snapshot_shows_the_views() {
    if std::env::var_os("RDS_GPU_TESTS").is_none() {
        eprintln!("skipped: set RDS_GPU_TESTS=1");
        return;
    }
    let _turn = turn();
    let mut ui = with_phantom(true);
    let path = common::target_dir("ui-test-snap").join("phantom.png");
    ui.snap(&path).unwrap();
    let image = image::open(&path).unwrap().to_luma8();
    assert_eq!(image.dimensions(), (1680, 940));
    // Right of the data tree and between the bars there is nothing but the
    // three views. Empty, they are black; with the phantom in them a good
    // part is water at the default window, a mid grey.
    let lit = image
        .enumerate_pixels()
        .filter(|(x, y, p)| *x > 320 && (60..900).contains(y) && p.0[0] >= 60)
        .count();
    assert!(
        lit > 20_000,
        "{lit} lit pixels in the views: the phantom was not drawn"
    );
}

/// The study of the restore tests (`common::study_with_radiographs`): the
/// synthetic RT study in `study/ct`, forty radiographs filed beside it in
/// `study`. Written once.
fn restore_fixture() -> (PathBuf, PathBuf, Vec<PathBuf>) {
    static FIXTURE: OnceLock<(PathBuf, PathBuf, Vec<PathBuf>)> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let (study, ct, kv) =
                common::study_with_radiographs(&common::target_dir("ui-test-restore"), 40);
            // Absolute, as the session keeps them.
            let abs = |p: &PathBuf| p.canonicalize().expect("written");
            (abs(&study), abs(&ct), kv.iter().map(abs).collect())
        })
        .clone()
}

/// Restore a session whose workspace A was loaded from `sources`: the start
/// screen's button, waited for. How long it took.
fn restore_session(sources: Vec<PathBuf>) -> (Ui, std::time::Duration) {
    let mut s = rust_dicom_station::settings::load();
    s.session[0] = sources;
    rust_dicom_station::settings::save(&s).expect("the settings are written");
    let mut ui = start_screen();
    let restore = Target::containing("Restore last session").role("button");
    assert!(
        !ui.find(&restore).unwrap().disabled,
        "no session to restore:\n{}",
        ui.tree_text()
    );
    let start = std::time::Instant::now();
    ui.click(&restore).unwrap();
    assert!(ui.settle(SETTLE), "the restore did not finish");
    (ui, start.elapsed())
}

/// The workspace's load warnings, unfolded (the arrow left of the header
/// is unnamed: egui's own), one line each.
fn load_warnings(ui: &mut Ui) -> Vec<String> {
    let Ok(header) = ui.find(&Target::containing("Warnings (").role("label")) else {
        return Vec::new();
    };
    let before = ui.tree_text();
    ui.click_at(
        header.rect.left_center() - egui::vec2(10.0, 0.0),
        egui::PointerButton::Primary,
        1,
    );
    ui.settle(SETTLE);
    assert_ne!(before, ui.tree_text(), "the warnings did not unfold");
    ui.find_all(&Target::containing("").role("label"))
        .into_iter()
        .map(|h| h.text)
        .collect()
}

/// Every radiograph and the CT listed once in the data tree: the planar
/// images' section counts the generator's two and the `kv` filed beside
/// them, and unfolded names each once.
fn assert_each_once(ui: &mut Ui, kv: usize) {
    let series = ui.find_all(&Target::containing(" sl.)").role("button"));
    assert_eq!(series.len(), 1, "the CT is listed {} times", series.len());
    let section = Target::label(&format!("Planar images ({})", kv + 2)).role("button");
    assert!(
        ui.has(&section),
        "not {} planar images:\n{}",
        kv + 2,
        ui.tree_text()
    );
    ui.click(&section).unwrap();
    ui.settle(SETTLE);
    for i in 1..=kv {
        let label = format!("KV {i:02}");
        let n = ui.find_all(&Target::containing(&label)).len();
        assert_eq!(n, 1, "{label} is listed {n} times:\n{}", ui.tree_text());
    }
}

/// *Restore last session* of a workspace loaded from a study folder of the
/// PACS and the folder of its series inside it (what a selection of the
/// whole study gives the session): one load, no "already present" warning,
/// the series and every radiograph once.
#[test]
fn restoring_a_study_folder_and_its_series_folder_reports_nothing_twice() {
    let _turn = turn();
    let (study, ct, kv) = restore_fixture();
    let (mut ui, took) = restore_session(vec![study, ct]);
    eprintln!("restore of [study, study/ct] took {took:.2?}");
    assert!(ui.has_label("Workspace A"), "{}", ui.tree_text());
    let warnings = load_warnings(&mut ui);
    assert!(
        !warnings.iter().any(|w| w.contains("already present")),
        "a plain restore says something was added twice: {warnings:?}"
    );
    assert_each_once(&mut ui, kv.len());
}

/// *Restore last session* of a workspace loaded from a series folder and
/// forty radiographs one file each (a selection of the PACS that took part
/// of a folder): one load, not forty-one, and each image once.
#[test]
fn restoring_radiographs_listed_one_by_one_is_one_load() {
    let _turn = turn();
    let (_, ct, kv) = restore_fixture();
    let mut sources = vec![ct];
    sources.extend(kv.iter().cloned());
    let (mut ui, took) = restore_session(sources);
    eprintln!(
        "restore of [study/ct + {} radiographs] took {took:.2?}",
        kv.len()
    );
    let warnings = load_warnings(&mut ui);
    assert!(
        !warnings.iter().any(|w| w.contains("already present")),
        "{warnings:?}"
    );
    assert_each_once(&mut ui, kv.len());
}

/// *Save GIF* in the posture window: once the model is built the button is
/// on the bottom bar, on the row of the time slider, where a recorder
/// button used to be (there is none now), and its menu lists the five
/// views - *All views* (the default), *Axial*, *Sagittal*, *Coronal*, *3D* -
/// any of them picked in the list; the *Vector field* option, the width and
/// the way back. What it saves is what its menu says: a click in a pane
/// (here the axial one) leaves *All views* as it was. The arrow spacing is
/// under the window's *Display* menu, and in the main viewer's *Settings ▸
/// Viewer*; nothing in either is unnamed. (The file dialog is the operating
/// system's and is not opened.)
#[test]
fn the_posture_window_saves_a_gif_of_all_views_or_the_one_picked() {
    let _turn = turn();
    let [a, b] = rust_dicom_station::gen_test_data::generate_articulated(
        &common::target_dir("ui-test-posture-gif"),
        &rust_dicom_station::gen_test_data::ArticulatedParams::default(),
        &rust_dicom_station::progress::Progress::default(),
    )
    .expect("the phantom is written");
    let mut ui = Ui::open(Opts {
        datasets: [Some(a), Some(b)],
        ..Opts::default()
    });
    assert!(ui.settle(SETTLE), "the two states did not load");

    // The main viewer's arrow spacing: Settings > Viewer.
    ui.click(&Target::label("Settings").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Viewer").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    let spacing = ui
        .find(&Target::label("Arrow spacing").role("slider"))
        .unwrap_or_else(|_| panic!("no arrow spacing in Settings > Viewer:\n{}", ui.tree_text()));
    assert!(spacing.text.contains("Arrow spacing"), "{}", spacing.text);
    assert_all_named(&ui, "Settings > Viewer");
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);

    ui.click(&Target::label("Tools").role("button")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Posture model").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    let gif = Target::label("Save GIF").role("button");
    assert!(!ui.has(&gif), "a GIF before there is a model");
    ui.click(&Target::label("Auto")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("BONE_UPPER")).unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::containing("Build").role("button"))
        .unwrap();
    assert!(ui.settle(SETTLE), "the model did not build");
    assert!(ui.has(&gif), "no Save GIF:\n{}", ui.tree_text());
    // On the bar, beside the time slider, and the only way to film the run.
    let bar = ui
        .find(&Target::label("Time along the path"))
        .unwrap_or_else(|_| panic!("no time slider:\n{}", ui.tree_text()));
    let button = ui.find(&gif).unwrap();
    assert!(
        (button.rect.center().y - bar.rect.center().y).abs() < bar.rect.height().max(12.0),
        "Save GIF at {:?} is not on the bar of the time slider at {:?}",
        button.rect,
        bar.rect
    );
    let win = ui
        .find(&Target::containing("Posture model").role("window"))
        .unwrap()
        .rect;
    assert!(
        !["Record", "recording", "⏺"].iter().any(|w| ui
            .find_all(&Target::containing(w).role("button"))
            .iter()
            .any(|h| win.contains(h.rect.center()))),
        "a recorder beside Save GIF:\n{}",
        ui.tree_text()
    );

    // The window's own arrow spacing, under its Display menu.
    let window = ui
        .find(&Target::containing("Posture model").role("window"))
        .unwrap();
    ui.click(&Target::label("Display").role("button")).unwrap();
    ui.settle(SETTLE);
    assert!(
        ui.find_all(&Target::label("Arrow spacing"))
            .iter()
            .any(|h| window.rect.contains(h.rect.center())),
        "no arrow spacing in the posture window's Display menu:\n{}",
        ui.tree_text()
    );
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);

    let views = ["All views", "Axial", "Sagittal", "Coronal", "3D"];
    let radio = |name: &str| Target::label(name).role("radiobutton");
    ui.click(&gif).unwrap();
    ui.settle(SETTLE);
    for choice in views {
        assert!(
            ui.has(&radio(choice)),
            "no \"{choice}\" view in the menu:\n{}",
            ui.tree_text()
        );
    }
    for choice in ["Vector field", "There and back", "Width", "Save..."] {
        assert!(
            ui.has(&Target::label(choice)),
            "no \"{choice}\" in the menu:\n{}",
            ui.tree_text()
        );
    }
    let field = ui
        .find(&Target::label("Vector field").role("checkbox"))
        .unwrap();
    assert_eq!(field.checked, Some(false), "the field is asked for");
    let all = ui.find(&radio("All views")).unwrap();
    assert_eq!(all.checked, Some(true), "all views is the default");
    // Any view from the list, the 3D scene too.
    ui.click(&radio("3D")).unwrap();
    ui.settle(SETTLE);
    assert_eq!(
        ui.find(&radio("3D")).unwrap().checked,
        Some(true),
        "{}",
        ui.tree_text()
    );
    ui.click(&Target::label("Vector field").role("checkbox"))
        .unwrap();
    ui.settle(SETTLE);
    assert_eq!(
        ui.find(&Target::label("Vector field").role("checkbox"))
            .unwrap()
            .checked,
        Some(true)
    );
    // And back to all of them: the radio lies over a pane, which takes no
    // part in the choice.
    ui.click(&radio("All views")).unwrap();
    ui.settle(SETTLE);
    assert_all_named(&ui, "the posture window's Save GIF menu");
    ui.key("Escape").unwrap();
    ui.settle(SETTLE);

    // The axial pane of the window is above its slice slider (the main
    // screen's, behind the window, start right of the data tree): a click
    // in it does not change what the GIF takes.
    let slider = ui
        .find_all(&Target::label("Axial slice"))
        .into_iter()
        .filter(|h| window.rect.contains(h.rect.center()))
        .min_by(|x, y| x.rect.left().total_cmp(&y.rect.left()))
        .unwrap_or_else(|| panic!("no axial pane in the window:\n{}", ui.tree_text()));
    ui.click_at(
        slider.rect.center_top() - egui::vec2(0.0, 120.0),
        egui::PointerButton::Primary,
        1,
    );
    ui.settle(SETTLE);
    ui.click(&gif).unwrap();
    ui.settle(SETTLE);
    for choice in views {
        let checked = ui.find(&radio(choice)).unwrap().checked;
        assert_eq!(
            checked,
            Some(choice == "All views"),
            "{choice} after a click in the axial pane:\n{}",
            ui.tree_text()
        );
    }
    assert_all_named(&ui, "the posture window's Save GIF menu");
}

/// What a brush stroke costs a frame on a real structure: a drag delivered
/// one pointer move a frame (`Ui::drag_timed`), with the 3-D contour brush,
/// the slice brush and the voxel brush on a segment, at three radii. Prints
/// the median and the worst frame of the moves and the release frame; it
/// asserts nothing about time, which depends on the machine. Needs a CT
/// with its RT structure set, one phase of the 4D-Lung data by default:
///
/// ```text
/// RDS_BRUSH_TIMING=1 cargo test --test ui -- --ignored --nocapture brush_frame_timing
/// RDS_BRUSH_CT=dir RDS_BRUSH_RS=file RDS_BRUSH_ROI=name ... (any other study)
/// ```
#[test]
#[ignore]
fn brush_frame_timing() {
    if std::env::var_os("RDS_BRUSH_TIMING").is_none() {
        eprintln!("RDS_BRUSH_TIMING not set: skipped");
        return;
    }
    let _turn = turn();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let ct = std::env::var_os("RDS_BRUSH_CT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            root.join("data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS/1_CT_4DFBCT__Gated__0.0_A")
        });
    let rs = std::env::var_os("RDS_BRUSH_RS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS/RS_RTS__0.0_A.dcm"));
    let roi = std::env::var("RDS_BRUSH_ROI").unwrap_or_else(|_| "RLung_c00".into());
    // One folder with the CT and its structure set, so it loads as one study.
    let data = common::target_dir("brush-timing-data");
    for e in std::fs::read_dir(&ct).expect("the CT folder") {
        let p = e.unwrap().path();
        if p.is_file() {
            std::fs::copy(&p, data.join(p.file_name().unwrap())).unwrap();
        }
    }
    std::fs::copy(&rs, data.join(rs.file_name().unwrap())).unwrap();
    let mut ui = Ui::open(Opts {
        datasets: [Some(data), None],
        ..Opts::default()
    });
    assert!(ui.settle(SETTLE), "the study did not load");
    let out = common::target_dir("ui-test-out");
    for step in [
        r#"{"click": {"label": "Modules", "role": "button"}}"#,
        r#"{"click": {"label": "Structure editor", "role": "checkbox"}}"#,
        r#"{"key": "Escape"}"#,
    ] {
        ui.run_step(&serde_json::from_str(step).unwrap(), &out)
            .unwrap_or_else(|e| panic!("{step}: {e:#}\n{}", ui.tree_text()));
    }
    let arrow = Target::label("Unfold Structure editor");
    if ui.has(&arrow) {
        ui.invoke(&arrow).unwrap();
    } else {
        ui.invoke(&Target::label("Structure editor").role("button"))
            .unwrap();
    }
    ui.settle(SETTLE);
    ui.click(&Target::containing(&format!("{roi}  [")).role("button"))
        .unwrap_or_else(|e| panic!("{e:#}\n{}", ui.tree_text()));
    ui.settle(SETTLE);
    ui.click(&Target::containing("Draw structure").role("button"))
        .unwrap();
    ui.settle(SETTLE);
    ui.click(&Target::label("Paint").role("button")).unwrap();
    ui.settle(SETTLE);

    let top = ui.find(&Target::label("Maximize Axial")).unwrap().rect;
    let bottom = ui.find(&Target::label("Axial slice")).unwrap().rect;
    let pane = egui::Rect::from_min_max(
        egui::pos2(bottom.left(), top.bottom()),
        egui::pos2(bottom.right(), bottom.top()),
    );
    // Back and forth across the right side of the body, a few points a
    // frame: the outline is crossed again and again, as a correction does.
    let path: Vec<egui::Pos2> = (0..60)
        .map(|i| {
            let t = i as f32 / 59.0;
            let x = 0.22 + 0.2 * (t * std::f32::consts::TAU * 1.5).sin().abs();
            pane.center() + egui::vec2(pane.width() * x, pane.height() * (t - 0.5) * 0.3)
        })
        .collect();
    let report = |what: &str, mut d: Vec<std::time::Duration>| {
        let release = d.pop().unwrap_or_default();
        let press = d.remove(0);
        let mut ms: Vec<f64> = d.iter().map(|d| d.as_secs_f64() * 1e3).collect();
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{what:<34} press {:7.1}  move median {:7.1}  worst {:7.1}  release {:7.1} ms",
            press.as_secs_f64() * 1e3,
            ms[ms.len() / 2],
            ms[ms.len() - 1],
            release.as_secs_f64() * 1e3,
        );
    };
    // An idle frame with the pointer moving over the view, for scale.
    let mut idle = Vec::new();
    for p in &path {
        ui.hover_at(*p);
        let t = std::time::Instant::now();
        ui.step();
        idle.push(t.elapsed().as_secs_f64() * 1e3);
    }
    idle.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "idle frame (pointer over the view)  median {:7.1} ms",
        idle[idle.len() / 2]
    );

    let readout = |ui: &Ui| -> String {
        ui.find(&Target::containing(" points · ").role("label"))
            .map(|h| h.text)
            .unwrap_or_default()
    };
    let mut steps_up = 0;
    for (radius, presses) in [(10, 4), (20, 8), (40, 12)] {
        while steps_up < presses {
            ui.key("]").unwrap();
            steps_up += 1;
        }
        ui.settle(SETTLE);
        let r = 5.0 * 1.2f64.powi(presses);
        for ball in [true, false] {
            let three_d = ui.find(&Target::label("3D").role("button")).unwrap();
            if three_d.checked != Some(ball) {
                ui.click(&Target::label("3D").role("button")).unwrap();
                ui.settle(SETTLE);
            }
            let before = readout(&ui);
            let d = ui.drag_timed(&path, egui::PointerButton::Primary);
            let t = std::time::Instant::now();
            ui.settle(SETTLE);
            let settle = t.elapsed();
            report(
                &format!(
                    "{roi} {} r {r:.1} mm (~{radius})",
                    if ball { "3D" } else { "2D" }
                ),
                d,
            );
            println!(
                "    settle after {:.1} ms; {before} -> {}",
                settle.as_secs_f64() * 1e3,
                readout(&ui)
            );
            ui.key("Ctrl+Z").unwrap();
            ui.settle(SETTLE);
        }
    }
    // The ball in the sagittal view: a stack re-cut onto the axial contours
    // the structure is stored as, every 150 ms and on release.
    while steps_up > 8 {
        ui.key("[").unwrap();
        steps_up -= 1;
    }
    ui.settle(SETTLE);
    let three_d = ui.find(&Target::label("3D").role("button")).unwrap();
    if three_d.checked != Some(true) {
        ui.click(&Target::label("3D").role("button")).unwrap();
        ui.settle(SETTLE);
    }
    let top = ui.find(&Target::label("Maximize Sagittal")).unwrap().rect;
    let bottom = ui.find(&Target::label("Sagittal slice")).unwrap().rect;
    let sag = egui::Rect::from_min_max(
        egui::pos2(bottom.left(), top.bottom()),
        egui::pos2(bottom.right(), bottom.top()),
    );
    let sag_path: Vec<egui::Pos2> = path
        .iter()
        .map(|p| sag.center() + (*p - pane.center()) * 0.6)
        .collect();
    let before = readout(&ui);
    let d = ui.drag_timed(&sag_path, egui::PointerButton::Primary);
    ui.settle(SETTLE);
    report(&format!("{roi} 3D sagittal r 21.5 mm"), d);
    println!("    {before} -> {}", readout(&ui));
    ui.key("Ctrl+Z").unwrap();
    ui.settle(SETTLE);

    // The voxel brush on a new segment, 3D (its default).
    pick_new_segment(&mut ui);
    ui.click(&Target::label("Paint").role("button")).unwrap();
    ui.settle(SETTLE);
    for presses in [12, 8, 4] {
        while steps_up > presses {
            ui.key("[").unwrap();
            steps_up -= 1;
        }
        while steps_up < presses {
            ui.key("]").unwrap();
            steps_up += 1;
        }
        ui.settle(SETTLE);
        let r = 5.0 * 1.2f64.powi(presses);
        let d = ui.drag_timed(&path, egui::PointerButton::Primary);
        ui.settle(SETTLE);
        report(&format!("segment 3D r {r:.1} mm"), d);
        ui.key("Ctrl+Z").unwrap();
        ui.settle(SETTLE);
    }
}
