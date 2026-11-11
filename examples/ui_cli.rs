//! The viewer without a window, for whoever is changing its interface and
//! wants to see the result: the tree of what is on screen, a picture of it,
//! and scripted input (`tests/common/headless.rs` is how).
//!
//! ```text
//! cargo run --release --example ui_cli -- [DATASET [DATASET_B]] [options]
//! ```
//!
//! With no dataset the start screen is what comes up. The run opens the
//! datasets, waits for the loads, runs the script if there is one, then
//! writes the tree and the picture asked for.
//!
//! Options:
//!
//! * `--tree FILE`: what is on screen. A `.txt` is an outline, one widget a
//!   line (`button "Load" (12,40 64x20)`: role, text, x,y and size in
//!   points, then `checked` / `disabled` / `selected`); anything else is the
//!   same as JSON, every node included.
//! * `--snap FILE.png`: the frame as a picture. This is the one thing here
//!   that opens a graphics adapter (a software one where there is one).
//! * `--script FILE.json`: steps to run first, see below.
//! * `--out DIR`: where a script's own `snap` / `tree` steps write
//!   (default: the current folder).
//! * `--size WxH` (points, default 1680x940), `--scale S` (pixels per
//!   point, default 1), `--light` (the light theme).
//! * `--state DIR`: the folder that stands in for the user's configuration
//!   and data folders (settings, models, PACS). Default: `target/ui-state`,
//!   emptied at every start, so a run begins from a first start and never
//!   reads or writes the real ones. A folder given here is kept as it is,
//!   which is how to prepare a state once (downloaded models, an archive)
//!   and reuse it.
//! * `--timeout S`: how long the loads may take (default 120).
//!
//! A script is `{"steps": [...]}`; each step is an object with one verb:
//!
//! ```text
//! {"click": T}  {"right_click": T}  {"double_click": T}  {"hover": T}
//! {"invoke": T}                      the widget's action through AccessKit:
//!                                    it need not be scrolled into view
//! {"click_at": [x, y]}  {"right_click_at": [x, y]}
//! {"drag": {"from": [x, y], "to": [x, y], "button": "left"}}
//! {"scroll": {"at": [x, y], "delta": [0, -120]}}
//! {"key": "Ctrl+Z"}  {"type": "text"}  {"resize": [w, h]}
//! {"wait": "idle"}  {"wait": 1.5}
//! {"expect": T}  {"expect_none": T}
//! {"snap": "name.png"}  {"tree": "name.txt"}  {"unnamed": "name.txt"}
//! ```
//!
//! `unnamed` (and the option `--unnamed FILE`) lists the widgets on screen
//! that can be acted on and say nothing in words - a button that is one
//! glyph, a tick box with no label: what `Named` in `src/app/widgets.rs` is
//! for. A new one is worth fixing before it ships.
//!
//! `T` names a widget by its text: `"Load"`, or `{"label": "Load"}` /
//! `{"contains": "load"}` with optional `"role": "button"` and `"nth": 1`
//! (which of several, from 0). A step that acts waits for the interface to
//! settle afterwards - a click that starts a job returns when the job has
//! landed - for up to `"timeout"` seconds (default 120); `"settle": false`
//! skips the wait. A step that fails ends the run with a message that names
//! the step, and for a widget that is not there, the closest texts that
//! are.
//!
//! Buttons that open a file dialog of the operating system are not for
//! scripts: the dialog is real and the run would wait on it.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, bail, Context as _, Result};

#[path = "../tests/common/headless.rs"]
mod headless;

use headless::{isolate_user_dirs, Opts, Ui};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    const VALUED: [&str; 9] = [
        "--unnamed",
        "--tree",
        "--snap",
        "--script",
        "--out",
        "--size",
        "--scale",
        "--state",
        "--timeout",
    ];
    const FLAGS: [&str; 1] = ["--light"];
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let mut datasets = Vec::new();
    let mut k = 0;
    while k < args.len() {
        let a = args[k].as_str();
        if VALUED.contains(&a) {
            k += 2;
        } else if FLAGS.contains(&a) {
            k += 1;
        } else if a.starts_with("--") {
            bail!("unknown option {a} (the options are in the header of examples/ui_cli.rs)");
        } else {
            datasets.push(PathBuf::from(a));
            k += 1;
        }
    }
    if datasets.len() > 2 {
        bail!("at most two datasets (A and B)");
    }
    for d in &datasets {
        if !d.exists() {
            bail!("{} does not exist", d.display());
        }
    }

    // First, while this is the only thread: the settings, the models and
    // the PACS of this run are a scratch folder's, not the user's.
    match opt("--state") {
        Some(dir) => isolate_user_dirs(&PathBuf::from(dir)),
        None => {
            let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/ui-state");
            let _ = std::fs::remove_dir_all(&dir);
            isolate_user_dirs(&dir);
        }
    }

    let script: Option<serde_json::Value> = match opt("--script") {
        Some(path) => {
            let text = std::fs::read_to_string(&path).with_context(|| format!("reading {path}"))?;
            Some(serde_json::from_str(&text).with_context(|| format!("parsing {path}"))?)
        }
        None => None,
    };
    let snap = opt("--snap").map(PathBuf::from);
    let tree = opt("--tree").map(PathBuf::from);
    let out = opt("--out").map_or_else(|| PathBuf::from("."), PathBuf::from);
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let timeout = match opt("--timeout") {
        Some(s) => Duration::from_secs_f64(s.parse().context("--timeout takes seconds")?),
        None => headless::SETTLE,
    };

    let mut opts = Opts {
        dark: !args.iter().any(|a| a == "--light"),
        // A graphics device only when a picture is wanted, by the command
        // line or by a step of the script.
        gpu: snap.is_some()
            || script
                .as_ref()
                .is_some_and(|s| s.to_string().contains("\"snap\"")),
        ..Opts::default()
    };
    if let Some(size) = opt("--size") {
        let (w, h) = size
            .split_once('x')
            .ok_or_else(|| anyhow!("--size takes WxH, as in 1680x940"))?;
        opts.size = [w.parse()?, h.parse()?];
    }
    if let Some(scale) = opt("--scale") {
        opts.scale = scale.parse().context("--scale takes a number")?;
    }
    let mut it = datasets.into_iter();
    opts.datasets = [it.next(), it.next()];

    let mut ui = Ui::open(opts);
    if !ui.settle(timeout) {
        bail!(
            "the interface was still busy {:.0} s after opening (see --timeout)",
            timeout.as_secs_f64()
        );
    }
    if let Some(script) = &script {
        ui.run_script(script, &out, &mut |line| eprintln!("{line}"))?;
    }
    if let Some(path) = &tree {
        ui.write_tree(path)?;
        eprintln!("wrote {}", path.display());
    }
    if let Some(path) = opt("--unnamed") {
        let list = ui.unnamed();
        std::fs::write(&path, list.join("\n") + "\n").with_context(|| format!("writing {path}"))?;
        eprintln!("wrote {path}: {} widget(s) without a name", list.len());
    }
    if let Some(path) = &snap {
        ui.snap(path)?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}
