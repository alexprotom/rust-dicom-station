//! The viewer without a window: `ViewerApp` run frame by frame, read back as
//! a tree of what it drew, driven by scripted input, rendered to an image.
//!
//! This is what lets a test - or an agent working on the interface - check
//! its own work: open a dataset, wait for the load, ask "is there a row
//! called TARGET", click it, look at the picture. Nothing here opens a
//! window or needs a display.
//!
//! **How.** `egui_kittest` (egui's own test harness) builds the application
//! through the same `ViewerApp::new` the window does, with a creation
//! context that has no window behind it, and then calls its `ui` once per
//! step with the input it is handed. egui describes every widget it draws to
//! AccessKit - the role, the text, the rectangle, whether it is checked or
//! disabled - and that description is the tree: nothing in `src/app/` knows
//! it is being watched. With [`Opts::gpu`] the harness also owns a wgpu
//! device and paints a frame into a texture it reads back ([`Ui::snap`]);
//! without it no graphics adapter is touched at all, which is what the test
//! suite uses, so it runs on any machine.
//!
//! **Tool windows.** Outside `eframe` an egui context embeds its viewports,
//! so [`tool_window`](rust_dicom_station::app) takes the branch Android
//! takes: every tool is an `egui::Window` inside the main one, and is in the
//! tree and in the picture like everything else.
//!
//! **Waiting.** A background job is polled once a frame and asks for the
//! next frame 100 ms later (`app/jobs.rs`). That request is in the frame's
//! output, so "is anything still running" needs no access to the
//! application's fields: [`Ui::settle`] steps until several frames in a row
//! have asked for nothing sooner than [`BUSY`]. A playback run asks for
//! frames for as long as it plays and never settles; that is what the
//! timeout is for.
//!
//! **The user's own files.** `ViewerApp::new` reads the settings and the
//! interface writes them back, and the start screen, the PACS and the last
//! session all come from the user's folders. A run from here must neither
//! show them (a picture of this would carry patient names) nor change them,
//! so [`isolate_user_dirs`] points the folders `settings.rs` resolves at a
//! scratch folder first. Call it before anything else.
//!
//! **What does not work.** A button that asks the operating system for a
//! file (`app/pick.rs` on the desktop) opens a real dialog and blocks:
//! scripts open datasets through [`Opts::datasets`] and stay off those
//! buttons.
//!
//! Shared by `tests/ui.rs` and `examples/ui_cli.rs` (which pulls this file
//! in by path, as `gen_ops_fixtures` does with `ops_ref.rs`).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context as _, Result};
use egui::accesskit;
use egui_kittest::kittest::{AccessKitNode, NodeT};
use egui_kittest::Harness;
use rust_dicom_station::app::ViewerApp;
use serde_json::{json, Map, Value};

/// A frame that asks for the next one this soon is a frame of something
/// still running: the job polls ask for 100 ms, an animation for none.
pub const BUSY: Duration = Duration::from_millis(150);

/// This many quiet frames in a row is settled. More than one, because a
/// result landing on one frame is what starts the next job on the following
/// one (a second dataset queued behind the first, a pull followed by a
/// refresh).
const QUIET_FRAMES: usize = 4;

/// How long a step may take to settle unless it says otherwise.
pub const SETTLE: Duration = Duration::from_secs(120);

/// One simulated frame. Short enough that a press, a release and a second
/// press - one frame each - are still a double click to egui.
const STEP_DT: f32 = 1.0 / 60.0;

/// Point every folder `settings.rs` resolves (configuration, data, and with
/// them the models, the PACS and the last session) into `dir`.
///
/// The process environment is the only door: Windows reads `LOCALAPPDATA`,
/// Linux `XDG_CONFIG_HOME` / `XDG_DATA_HOME`, macOS `HOME`. Writing the
/// environment is sound only while no other thread reads it, so this is the
/// first thing a `main` does, and a test suite does it once behind a lock
/// before any of its tests builds the application.
pub fn isolate_user_dirs(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create the scratch folder for settings");
    if cfg!(target_os = "windows") {
        std::env::set_var("LOCALAPPDATA", dir);
    } else if cfg!(target_os = "macos") {
        std::env::set_var("HOME", dir);
    } else {
        std::env::set_var("XDG_CONFIG_HOME", dir.join("config"));
        std::env::set_var("XDG_DATA_HOME", dir.join("data"));
        // A snap's own folders win over XDG in `settings.rs`.
        for var in [
            "SNAP",
            "SNAP_NAME",
            "SNAP_INSTANCE_NAME",
            "SNAP_USER_COMMON",
        ] {
            std::env::remove_var(var);
        }
    }
}

/// How the application is opened.
#[derive(Clone, Debug)]
pub struct Opts {
    /// Dataset A and dataset B, as on the command line of the viewer.
    pub datasets: [Option<PathBuf>; 2],
    /// Window size in points (the viewer's own default).
    pub size: [f32; 2],
    /// Pixels per point.
    pub scale: f32,
    /// Dark theme (the default) or light.
    pub dark: bool,
    /// Own a wgpu device, so [`Ui::snap`] works. Off, no adapter is opened.
    pub gpu: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            datasets: [None, None],
            size: [1680.0, 940.0],
            scale: 1.0,
            dark: true,
            gpu: false,
        }
    }
}

/// A node of the tree as a script or a test names it.
#[derive(Clone, Debug)]
pub struct Target {
    /// The node's text: its label, or its value where it has no label.
    pub text: String,
    /// `text` may be any part of the node's text, in any case.
    pub contains: bool,
    /// The role, lower case as the tree prints it (`button`, `checkbox`).
    pub role: Option<String>,
    /// Which of several matches, in tree order.
    pub nth: usize,
}

impl Target {
    /// The node whose text is exactly `text`.
    pub fn label(text: &str) -> Target {
        Target {
            text: text.to_string(),
            contains: false,
            role: None,
            nth: 0,
        }
    }

    /// The node whose text contains `text`, case-insensitively.
    pub fn containing(text: &str) -> Target {
        Target {
            contains: true,
            ..Target::label(text)
        }
    }

    /// The same node, of this role only.
    pub fn role(mut self, role: &str) -> Target {
        self.role = Some(role.to_string());
        self
    }

    /// A script's spelling: `"Text"`, or `{"label": ..}` / `{"contains":
    /// ..}` with optional `"role"` and `"nth"`.
    pub fn from_json(v: &Value) -> Result<Target> {
        if let Some(s) = v.as_str() {
            return Ok(Target::label(s));
        }
        let obj = v
            .as_object()
            .ok_or_else(|| anyhow!("a target is a string or an object, got {v}"))?;
        let mut t = match (
            obj.get("label").and_then(Value::as_str),
            obj.get("contains").and_then(Value::as_str),
        ) {
            (Some(s), None) => Target::label(s),
            (None, Some(s)) => Target::containing(s),
            _ => bail!("a target needs \"label\" or \"contains\" (one of them), got {v}"),
        };
        t.role = obj.get("role").and_then(Value::as_str).map(str::to_string);
        t.nth = obj.get("nth").and_then(Value::as_u64).unwrap_or(0) as usize;
        Ok(t)
    }

    fn matches(&self, node: &AccessKitNode<'_>) -> bool {
        if let Some(role) = &self.role {
            if !role_name(node).eq_ignore_ascii_case(role) {
                return false;
            }
        }
        match text_of(node) {
            Some(text) if self.contains => text.to_lowercase().contains(&self.text.to_lowercase()),
            Some(text) => text == self.text,
            None => false,
        }
    }

    /// For a message: `button "Load"`, `widget containing "load"`.
    fn describe(&self) -> String {
        let how = if self.contains { "containing " } else { "" };
        let role = self.role.as_deref().unwrap_or("widget");
        format!("{role} {how}\"{}\"", self.text)
    }
}

/// A node that was found: where it is and how to address it.
#[derive(Clone, Debug)]
pub struct Hit {
    /// Its rectangle, in points.
    pub rect: egui::Rect,
    pub role: String,
    pub text: String,
    pub disabled: bool,
    /// Checked or not, for the widgets that are one or the other.
    pub checked: Option<bool>,
    /// The node in AccessKit's terms, for an action sent to it directly.
    node: accesskit::NodeId,
    tree: accesskit::TreeId,
}

/// The role as the tree prints it: AccessKit's name, lower case.
fn role_name(node: &AccessKitNode<'_>) -> String {
    format!("{:?}", node.role()).to_lowercase()
}

/// What a node says: its label, and where it has none, its value (a text
/// field's contents, a drag value's number).
fn text_of(node: &AccessKitNode<'_>) -> Option<String> {
    node.label()
        .filter(|s| !s.is_empty())
        .or_else(|| node.value().filter(|s| !s.is_empty()))
}

/// Whether a node is part of what the tree shows. egui gives every piece of
/// text a `TextRun` child that repeats it (for a screen reader's caret); here
/// it would only double every label.
fn shown(node: &AccessKitNode<'_>) -> bool {
    !node.is_hidden() && node.role() != accesskit::Role::TextRun
}

/// Containers egui makes for its own layout: with no text of their own they
/// say nothing, and the outline prints their children in their place.
fn is_plumbing(node: &AccessKitNode<'_>) -> bool {
    matches!(
        node.role(),
        accesskit::Role::GenericContainer | accesskit::Role::Unknown | accesskit::Role::Group
    ) && text_of(node).is_none()
}

/// The application, headless.
pub struct Ui {
    harness: Harness<'static, ViewerApp>,
    scale: f32,
    gpu: bool,
}

impl Ui {
    /// Build the application and draw its first frames. The datasets start
    /// loading; [`settle`](Self::settle) waits for them.
    pub fn open(opts: Opts) -> Ui {
        let [a, b] = opts.datasets;
        let mut builder = Harness::<ViewerApp>::builder()
            .with_size(egui::vec2(opts.size[0], opts.size[1]))
            .with_pixels_per_point(opts.scale)
            .with_step_dt(STEP_DT)
            .with_theme(if opts.dark {
                egui::Theme::Dark
            } else {
                egui::Theme::Light
            });
        if opts.gpu {
            builder = builder.wgpu();
        }
        let harness = builder.build_eframe(move |cc| ViewerApp::new(cc, a, b));
        Ui {
            harness,
            scale: opts.scale,
            gpu: opts.gpu,
        }
    }

    /// When the last frame asked for the next one.
    fn repaint_delay(&self) -> Duration {
        self.harness
            .output()
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay)
    }

    /// One frame per queued event, or one frame if there is none.
    pub fn step(&mut self) {
        self.harness.step();
    }

    /// Run frames until nothing asks for another one soon: the jobs have
    /// landed and what they changed has been drawn. `false` if `timeout`
    /// passed first.
    pub fn settle(&mut self, timeout: Duration) -> bool {
        let start = Instant::now();
        let mut quiet = 0;
        while quiet < QUIET_FRAMES {
            if start.elapsed() > timeout {
                return false;
            }
            self.harness.step();
            let delay = self.repaint_delay();
            if delay <= BUSY {
                quiet = 0;
                // A job being polled: give its thread the time, rather than
                // spin through frames that would all say the same.
                if !delay.is_zero() {
                    std::thread::sleep(Duration::from_millis(20));
                }
            } else {
                quiet += 1;
            }
        }
        true
    }

    /// [`settle`](Self::settle), as an error when it does not.
    fn settled(&mut self, timeout: Duration) -> Result<()> {
        if self.settle(timeout) {
            Ok(())
        } else {
            bail!(
                "the interface was still asking for frames after {:.0} s (a job that has not \
                 finished, or a playback run); give the step a longer \"timeout\", or \
                 \"settle\": false",
                timeout.as_secs_f64()
            )
        }
    }

    // ---- reading -------------------------------------------------------

    /// Every node that is shown, in tree order.
    fn nodes(&self) -> Vec<AccessKitNode<'_>> {
        fn walk<'t>(node: AccessKitNode<'t>, out: &mut Vec<AccessKitNode<'t>>) {
            for child in node.children() {
                if !shown(&child) {
                    continue;
                }
                out.push(child);
                walk(child, out);
            }
        }
        let root = self.harness.root().accesskit_node();
        let mut out = vec![root];
        walk(root, &mut out);
        out
    }

    /// A node's rectangle in points (AccessKit has it in pixels).
    fn rect_of(&self, node: &AccessKitNode<'_>) -> Option<egui::Rect> {
        let r = node.bounding_box()?;
        let s = self.scale;
        Some(egui::Rect::from_min_max(
            egui::pos2(r.x0 as f32 / s, r.y0 as f32 / s),
            egui::pos2(r.x1 as f32 / s, r.y1 as f32 / s),
        ))
    }

    /// Every node that matches, in tree order.
    pub fn find_all(&self, target: &Target) -> Vec<Hit> {
        self.nodes()
            .iter()
            .filter(|n| target.matches(n))
            .filter_map(|n| {
                let (node, tree) = n.locate();
                Some(Hit {
                    rect: self.rect_of(n)?,
                    role: role_name(n),
                    text: text_of(n).unwrap_or_default(),
                    disabled: n.is_disabled(),
                    checked: n.toggled().map(|t| t == accesskit::Toggled::True),
                    node,
                    tree,
                })
            })
            .collect()
    }

    /// The node a target names, or an error that says what is there
    /// instead: the texts on screen that come closest.
    pub fn find(&self, target: &Target) -> Result<Hit> {
        let mut hits = self.find_all(target);
        if target.nth < hits.len() {
            return Ok(hits.swap_remove(target.nth));
        }
        if !hits.is_empty() {
            bail!(
                "\"nth\" {} asks for more than the {} on screen that match {}",
                target.nth,
                hits.len(),
                target.describe()
            );
        }
        // The first word is usually enough to find what was meant.
        let needle = target
            .text
            .split_whitespace()
            .next()
            .unwrap_or(&target.text)
            .to_lowercase();
        let mut near: Vec<String> = self
            .nodes()
            .iter()
            .filter_map(|n| Some((role_name(n), text_of(n)?)))
            .filter(|(_, text)| text.to_lowercase().contains(&needle))
            .map(|(role, text)| format!("{role} \"{text}\""))
            .collect();
        near.dedup();
        near.truncate(12);
        if near.is_empty() {
            bail!("no {} on screen", target.describe());
        }
        bail!(
            "no {} on screen; the closest are: {}",
            target.describe(),
            near.join(", ")
        )
    }

    /// The widgets on screen that a person can act on and that say nothing
    /// in words: a button that is one glyph, a tick box or a slider with no
    /// label. Each as its outline line. A script can reach such a widget
    /// only by where it is, and a screen reader cannot say what it is; the
    /// application names them through `Named` (`src/app/widgets.rs`).
    pub fn unnamed(&self) -> Vec<String> {
        use accesskit::Role;
        self.nodes()
            .iter()
            .filter(|n| {
                matches!(
                    n.role(),
                    Role::Button
                        | Role::CheckBox
                        | Role::RadioButton
                        | Role::Slider
                        | Role::ColorWell
                        | Role::ComboBox
                        | Role::Link
                )
            })
            // The label alone: a slider's value is a number, not a name.
            // A combo box is the exception - what it shows is its choice.
            .filter(|n| {
                let worded =
                    |t: Option<String>| t.is_some_and(|t| t.chars().any(char::is_alphanumeric));
                !(worded(n.label()) || (n.role() == Role::ComboBox && worded(n.value())))
            })
            .map(|n| {
                let at = self.rect_of(n).map_or_else(String::new, |r| {
                    format!(
                        " ({:.0},{:.0} {:.0}x{:.0})",
                        r.min.x,
                        r.min.y,
                        r.width(),
                        r.height()
                    )
                });
                match n.label().filter(|l| !l.is_empty()) {
                    Some(label) => format!("{} {label:?}{at}", role_name(n)),
                    None => format!("{}{at}", role_name(n)),
                }
            })
            .collect()
    }

    /// Whether a target is on screen.
    pub fn has(&self, target: &Target) -> bool {
        !self.find_all(target).is_empty()
    }

    /// Whether a node with exactly this text is on screen.
    pub fn has_label(&self, text: &str) -> bool {
        self.has(&Target::label(text))
    }

    fn node_json(&self, node: &AccessKitNode<'_>) -> Value {
        let mut m = Map::new();
        m.insert("role".into(), json!(role_name(node)));
        if let Some(label) = node.label().filter(|s| !s.is_empty()) {
            m.insert("label".into(), json!(label));
        }
        if let Some(value) = node.value().filter(|s| !s.is_empty()) {
            if node.label().as_deref() != Some(value.as_str()) {
                m.insert("value".into(), json!(value));
            }
        }
        if let Some(r) = self.rect_of(node) {
            m.insert(
                "rect".into(),
                json!([
                    r.min.x.round(),
                    r.min.y.round(),
                    r.width().round(),
                    r.height().round()
                ]),
            );
        }
        if node.is_disabled() {
            m.insert("disabled".into(), json!(true));
        }
        if let Some(t) = node.toggled() {
            m.insert("checked".into(), json!(t == accesskit::Toggled::True));
        }
        if let Some(s) = node.is_selected() {
            m.insert("selected".into(), json!(s));
        }
        if node.is_focused() {
            m.insert("focused".into(), json!(true));
        }
        let children: Vec<Value> = node
            .children()
            .filter(shown)
            .map(|c| self.node_json(&c))
            .collect();
        if !children.is_empty() {
            m.insert("children".into(), Value::Array(children));
        }
        Value::Object(m)
    }

    /// The whole tree: every node with its role, text, rectangle
    /// (`[x, y, width, height]` in points) and state.
    pub fn tree_json(&self) -> Value {
        let size = self.harness.ctx.content_rect().size();
        json!({
            "size": [size.x, size.y],
            "scale": self.scale,
            "root": self.node_json(&self.harness.root().accesskit_node()),
        })
    }

    /// The tree as an outline, one node a line: the role, the text, the
    /// rectangle as `(x,y WxH)`, the state. egui's own layout containers
    /// are left out and their children moved up, so this is a few hundred
    /// lines where the JSON is thousands.
    pub fn tree_text(&self) -> String {
        fn line(ui: &Ui, node: &AccessKitNode<'_>, depth: usize, out: &mut String) {
            let mut next = depth;
            if !is_plumbing(node) {
                next += 1;
                out.push_str(&"  ".repeat(depth));
                out.push_str(&role_name(node));
                // The text, and after it the value where a node has both (a
                // slider's name and its number).
                let text = text_of(node);
                if let Some(text) = &text {
                    out.push_str(&format!(" {text:?}"));
                }
                if let Some(value) = node.value().filter(|s| !s.is_empty()) {
                    if text.as_deref() != Some(value.as_str()) {
                        out.push_str(&format!(" = {value:?}"));
                    }
                }
                if let Some(r) = ui.rect_of(node) {
                    out.push_str(&format!(
                        " ({:.0},{:.0} {:.0}x{:.0})",
                        r.min.x,
                        r.min.y,
                        r.width(),
                        r.height()
                    ));
                }
                if node.is_disabled() {
                    out.push_str(" disabled");
                }
                match node.toggled() {
                    Some(accesskit::Toggled::True) => out.push_str(" checked"),
                    Some(_) => out.push_str(" unchecked"),
                    None => {}
                }
                if node.is_selected() == Some(true) {
                    out.push_str(" selected");
                }
                if node.is_focused() {
                    out.push_str(" focused");
                }
                out.push('\n');
            }
            for child in node.children().filter(shown) {
                line(ui, &child, next, out);
            }
        }
        let mut out = String::new();
        line(self, &self.harness.root().accesskit_node(), 0, &mut out);
        out
    }

    /// Write the tree: an outline to a `.txt`, JSON to anything else.
    pub fn write_tree(&self, path: &Path) -> Result<()> {
        let text = if path.extension().is_some_and(|e| e == "txt") {
            self.tree_text()
        } else {
            serde_json::to_string_pretty(&self.tree_json())?
        };
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
    }

    /// Paint the current frame and write it as a PNG. Needs [`Opts::gpu`].
    pub fn snap(&mut self, path: &Path) -> Result<()> {
        if !self.gpu {
            bail!("a picture needs the harness opened with a graphics device (Opts::gpu)");
        }
        let image = self
            .harness
            .render()
            .map_err(|e| anyhow!("rendering the frame: {e}"))?;
        image
            .save(path)
            .with_context(|| format!("writing {}", path.display()))
    }

    // ---- input ---------------------------------------------------------

    /// Press and release a pointer button at a position, `count` times.
    pub fn click_at(&mut self, pos: egui::Pos2, button: egui::PointerButton, count: usize) {
        self.harness.event(egui::Event::PointerMoved(pos));
        for _ in 0..count {
            for pressed in [true, false] {
                self.harness.event(egui::Event::PointerButton {
                    pos,
                    button,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                });
            }
        }
        self.harness.step();
    }

    /// Click the middle of a node with the left button.
    pub fn click(&mut self, target: &Target) -> Result<()> {
        let hit = self.find(target)?;
        self.click_at(hit.rect.center(), egui::PointerButton::Primary, 1);
        Ok(())
    }

    /// Ask a node to act through AccessKit, the way a screen reader would:
    /// it need not be scrolled into view.
    pub fn invoke(&mut self, target: &Target) -> Result<()> {
        let hit = self.find(target)?;
        self.harness.event(egui::Event::AccessKitActionRequest(
            accesskit::ActionRequest {
                target_node: hit.node,
                target_tree: hit.tree,
                action: accesskit::Action::Click,
                data: None,
            },
        ));
        self.harness.step();
        Ok(())
    }

    /// Press `button` at `from`, move to `to` in a few steps, let go.
    pub fn drag(&mut self, from: egui::Pos2, to: egui::Pos2, button: egui::PointerButton) {
        const MOVES: usize = 8;
        let press = |pos, pressed| egui::Event::PointerButton {
            pos,
            button,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        self.harness.event(egui::Event::PointerMoved(from));
        self.harness.event(press(from, true));
        for k in 1..=MOVES {
            let t = k as f32 / MOVES as f32;
            self.harness
                .event(egui::Event::PointerMoved(from + (to - from) * t));
        }
        self.harness.event(press(to, false));
        self.harness.step();
    }

    /// Turn the wheel over a position: `delta` in points, positive `y`
    /// moving the content down (so the view scrolls up).
    pub fn scroll(&mut self, at: egui::Pos2, delta: egui::Vec2) {
        self.harness.event(egui::Event::PointerMoved(at));
        self.harness.event(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta,
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::NONE,
        });
        self.harness.step();
    }

    /// Press a key, written as the menus write it: `F9`, `Ctrl+Z`,
    /// `Shift+ArrowDown`.
    pub fn key(&mut self, combo: &str) -> Result<()> {
        let mut modifiers = egui::Modifiers::NONE;
        let mut key = None;
        for part in combo.split('+').map(str::trim) {
            match part.to_lowercase().as_str() {
                // What a desktop back end reports for the Ctrl key: `ctrl`
                // and, off the Mac, `command`, which is what the shortcuts
                // in `app/` read.
                "ctrl" | "control" => {
                    modifiers.ctrl = true;
                    modifiers.command = true;
                }
                "cmd" | "command" => {
                    modifiers.mac_cmd = true;
                    modifiers.command = true;
                }
                "shift" => modifiers.shift = true,
                "alt" => modifiers.alt = true,
                _ => {
                    key = Some(
                        egui::Key::from_name(part)
                            .ok_or_else(|| anyhow!("\"{part}\" is not a key egui knows"))?,
                    )
                }
            }
        }
        let key = key.ok_or_else(|| anyhow!("\"{combo}\" names no key"))?;
        self.harness.key_press_modifiers(modifiers, key);
        self.harness.step();
        Ok(())
    }

    /// Type text into whatever has the keyboard.
    pub fn type_text(&mut self, text: &str) {
        self.harness.event(egui::Event::Text(text.to_string()));
        self.harness.step();
    }

    /// Resize the window, in points.
    pub fn resize(&mut self, size: [f32; 2]) {
        self.harness.set_size(egui::vec2(size[0], size[1]));
        self.harness.step();
    }

    // ---- scripts -------------------------------------------------------

    /// Run a script: `{"steps": [...]}` or the array itself. Every step is
    /// an object with one verb; see `examples/ui_cli.rs` for the list.
    /// Files a step writes go into `out`. `log` gets one line per step.
    pub fn run_script(
        &mut self,
        script: &Value,
        out: &Path,
        log: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let steps = script
            .get("steps")
            .unwrap_or(script)
            .as_array()
            .ok_or_else(|| anyhow!("a script is {{\"steps\": [...]}} or an array of steps"))?;
        for (k, step) in steps.iter().enumerate() {
            log(&format!("step {}: {step}", k + 1));
            self.run_step(step, out)
                .with_context(|| format!("step {} ({step})", k + 1))?;
        }
        Ok(())
    }

    /// Run one step of a script.
    pub fn run_step(&mut self, step: &Value, out: &Path) -> Result<()> {
        const OPTIONS: [&str; 2] = ["timeout", "settle"];
        let obj = step
            .as_object()
            .ok_or_else(|| anyhow!("a step is an object"))?;
        let timeout = obj
            .get("timeout")
            .and_then(Value::as_f64)
            .map_or(SETTLE, Duration::from_secs_f64);
        let settle = obj.get("settle").and_then(Value::as_bool).unwrap_or(true);
        let mut verbs = obj.iter().filter(|(k, _)| !OPTIONS.contains(&k.as_str()));
        let (verb, arg) = verbs.next().ok_or_else(|| anyhow!("a step needs a verb"))?;
        if let Some((other, _)) = verbs.next() {
            bail!("a step has one verb, this one has \"{verb}\" and \"{other}\"");
        }
        let point = |v: &Value| -> Result<egui::Pos2> {
            match v.as_array().map(Vec::as_slice) {
                Some([x, y]) => match (x.as_f64(), y.as_f64()) {
                    (Some(x), Some(y)) => Ok(egui::pos2(x as f32, y as f32)),
                    _ => bail!("a position is [x, y] in points, got {v}"),
                },
                _ => bail!("a position is [x, y] in points, got {v}"),
            }
        };
        match verb.as_str() {
            "click" => self.click(&Target::from_json(arg)?)?,
            "right_click" => {
                let hit = self.find(&Target::from_json(arg)?)?;
                self.click_at(hit.rect.center(), egui::PointerButton::Secondary, 1);
            }
            "double_click" => {
                let hit = self.find(&Target::from_json(arg)?)?;
                self.click_at(hit.rect.center(), egui::PointerButton::Primary, 2);
            }
            "hover" => {
                let hit = self.find(&Target::from_json(arg)?)?;
                self.harness
                    .event(egui::Event::PointerMoved(hit.rect.center()));
                self.harness.step();
            }
            "invoke" => self.invoke(&Target::from_json(arg)?)?,
            "click_at" => self.click_at(point(arg)?, egui::PointerButton::Primary, 1),
            "right_click_at" => self.click_at(point(arg)?, egui::PointerButton::Secondary, 1),
            "drag" => {
                let from = point(arg.get("from").unwrap_or(&Value::Null))?;
                let to = point(arg.get("to").unwrap_or(&Value::Null))?;
                let button = match arg.get("button").and_then(Value::as_str) {
                    None | Some("left") => egui::PointerButton::Primary,
                    Some("right") => egui::PointerButton::Secondary,
                    Some("middle") => egui::PointerButton::Middle,
                    Some(other) => bail!("\"{other}\" is not left, right or middle"),
                };
                self.drag(from, to, button);
            }
            "scroll" => {
                let at = point(arg.get("at").unwrap_or(&Value::Null))?;
                let delta = point(arg.get("delta").unwrap_or(&Value::Null))?;
                self.scroll(at, delta.to_vec2());
            }
            "key" => self.key(
                arg.as_str()
                    .ok_or_else(|| anyhow!("\"key\" takes a string"))?,
            )?,
            "type" => self.type_text(
                arg.as_str()
                    .ok_or_else(|| anyhow!("\"type\" takes a string"))?,
            ),
            "resize" => {
                let p = point(arg)?;
                self.resize([p.x, p.y]);
            }
            "wait" => match (arg.as_str(), arg.as_f64()) {
                (Some("idle"), _) => return self.settled(timeout),
                (_, Some(seconds)) => {
                    // Frames keep coming while it waits, so the jobs are
                    // polled and what they report is drawn.
                    let until = Instant::now() + Duration::from_secs_f64(seconds);
                    while Instant::now() < until {
                        self.harness.step();
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    return Ok(());
                }
                _ => bail!("\"wait\" takes \"idle\" or a number of seconds"),
            },
            "unnamed" => {
                let name = arg
                    .as_str()
                    .ok_or_else(|| anyhow!("\"unnamed\" takes a file name"))?;
                let path = out.join(name);
                return std::fs::write(&path, self.unnamed().join("\n") + "\n")
                    .with_context(|| format!("writing {}", path.display()));
            }
            "expect" => {
                self.find(&Target::from_json(arg)?)?;
                return Ok(());
            }
            "expect_none" => {
                let target = Target::from_json(arg)?;
                if self.has(&target) {
                    bail!("{} is on screen and should not be", target.describe());
                }
                return Ok(());
            }
            "snap" => {
                let name = arg
                    .as_str()
                    .ok_or_else(|| anyhow!("\"snap\" takes a file name"))?;
                return self.snap(&out.join(name));
            }
            "tree" => {
                let name = arg
                    .as_str()
                    .ok_or_else(|| anyhow!("\"tree\" takes a file name"))?;
                return self.write_tree(&out.join(name));
            }
            other => bail!("\"{other}\" is not a step this runner knows"),
        }
        if settle {
            self.settled(timeout)
        } else {
            Ok(())
        }
    }
}
