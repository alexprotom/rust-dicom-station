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
    "Dose display",
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
        "Close Coronal",
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
