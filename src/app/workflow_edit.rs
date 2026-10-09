//! The workflow editor: a canvas of nodes and wires, the palette they come
//! from, and the parameters of the one selected.
//!
//! The canvas is [`egui_snarl`]'s: nodes are dragged by their headers,
//! wires are drawn from an output pin to an input pin, the canvas pans with
//! the middle button (or a drag on empty space) and zooms with the wheel, a
//! right click on empty space offers every kind of node, and a wire dropped
//! on empty space offers the kinds that can take it. What the canvas holds
//! is an [`EdNode`] per node - the workflow's own id, title and parameters -
//! so turning the canvas back into a [`Workflow`] for saving or running is a
//! walk over its nodes and wires ([`WorkflowEditor::to_workflow`]).
//!
//! The rules a wire has to follow are the workflow's: the output's type
//! must be one the input accepts, an input that takes one wire loses the
//! old one to the new, and no wire may close a circle. A refused wire says
//! why in the editor's status line.
//!
//! Pins and wires are coloured by what they carry (the palette of
//! [`PortType::color`]), node headers by what the node is for. While a run is
//! going, each node's header carries its state - running, done, failed -
//! and the running one is outlined.
//!
//! **Editing.** Every change is a step of the history (*Undo*, *Redo*,
//! Ctrl+Z / Ctrl+Shift+Z or Ctrl+Y): the editor compares the workflow with
//! the last state it kept once the mouse is let go and no text field has
//! the keyboard, so a drag or a typed title is one step, not a hundred.
//! Ctrl+C / Ctrl+X copy or cut the selected steps (Shift-click or a
//! Shift-drag selects several; else the one in the inspector) with the
//! wires between them, as workflow JSON on the system clipboard, so they
//! paste into another workflow or another copy of the program; Ctrl+V puts
//! them under the pointer, Ctrl+D duplicates. The arrow keys move the
//! selection by 10 units (Shift: a grid square).
//!
//! **Frames** are titled boxes around groups of steps, drawn behind them;
//! the box follows its steps, and dragging its title moves them all. A
//! **map** in the corner shows the whole canvas and the part in view; a
//! click or a drag on it moves the view.
//!
//! Loading, saving, the recent list and the confirmation before unsaved
//! changes are dropped are here too; the run window is `workflow_run.rs`.

use std::collections::BTreeMap;
use std::path::Path;

use egui::emath::TSTransform;
use egui::{Color32, Pos2, Rect};
use egui_snarl::ui::{
    get_selected_nodes, AnyPins, BackgroundPattern, Grid, NodeLayout, PinInfo, PinPlacement,
    PinShape, SnarlStyle, SnarlViewer, SnarlWidget,
};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};

use super::*;
use crate::workflow::graph::catalog::{self as cat, Category, Kind, Op, PortType};
use crate::workflow::graph::{
    store, Findings, Frame as WfFrame, Node as WfNode, OutputSettings, Workflow,
};

/// The id the canvas widget keeps its state (pan, zoom, selection) under.
const CANVAS_ID: &str = "workflow_canvas";

/// The workflow file filter of the file dialogs.
pub(super) const WORKFLOW_FILES: pick::Filter = pick::Filter {
    name: "Workflow",
    exts: &[store::EXTENSION],
};

/// One node on the canvas.
#[derive(Clone, Debug)]
pub(super) struct EdNode {
    /// The workflow's own id, which links, runs and reports name it by.
    pub id: u32,
    pub title: String,
    pub op: Op,
}

impl EdNode {
    fn label(&self) -> String {
        if self.title.trim().is_empty() {
            self.op.kind().info().name.to_string()
        } else {
            self.title.trim().to_string()
        }
    }
}

/// How a node of a run is doing, as the canvas and the run window show it.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum Mark {
    Waiting,
    Running,
    Done(f64),
    /// Unchanged since the last run, whose result it took over.
    Reused,
    Failed(String),
    NotRun,
}

/// The one workflow being edited.
pub(super) struct WorkflowEditor {
    pub open: bool,
    pub snarl: Snarl<EdNode>,
    pub name: String,
    pub description: String,
    pub output: OutputSettings,
    /// The file it was read from or last saved to.
    pub path: Option<PathBuf>,
    /// The workflow as last saved or opened, to tell whether it changed.
    saved: String,
    /// The node whose parameters the right-hand panel shows.
    pub inspect: Option<u32>,
    pub findings: Findings,
    /// One line about the last thing that happened (a refused wire, a
    /// saved file).
    pub status: Option<String>,
    /// Where the next node from the palette goes, in canvas units.
    drop_at: Pos2,
    /// Which canvas this is: a workflow opened is a new canvas, with a view
    /// of its own (egui keeps the pan and zoom under the canvas's id).
    canvas: u64,
    /// Frames left in which the view is set to show every step.
    fit: u8,
    /// Boxes around groups of steps.
    pub frames: Vec<WfFrame>,
    /// Earlier states, oldest first, and the states undone (the next redo
    /// last), as workflow JSON.
    undo: Vec<String>,
    redo: Vec<String>,
    /// The state the history last kept.
    committed: String,
    /// Canvas to window as last drawn, and a view to set on the next frame
    /// (the minimap's, or the one kept across a rebuilt canvas).
    view: TSTransform,
    set_view: Option<TSTransform>,
    /// Every node's rectangle when last drawn, in canvas units, for the
    /// frames and the minimap.
    rects: Vec<(u32, Rect)>,
    /// Show the minimap.
    pub minimap: bool,
    /// What *Copy* took, for *Paste* when the system clipboard cannot be
    /// read.
    clip: Option<Workflow>,
    /// A frame being dragged.
    frame_drag: Option<FrameDrag>,
}

/// A frame's title being dragged: which frame, where the pointer was
/// pressed (window), and where its steps were then (canvas).
struct FrameDrag {
    frame: usize,
    from: Pos2,
    origins: Vec<(u32, Pos2)>,
}

/// States the history keeps.
const HISTORY: usize = 100;

/// Build the canvas of a workflow.
fn snarl_of(wf: &Workflow) -> Snarl<EdNode> {
    let mut snarl = Snarl::new();
    let mut ids: BTreeMap<u32, NodeId> = BTreeMap::new();
    for n in &wf.nodes {
        let id = snarl.insert_node(
            egui::pos2(n.pos[0], n.pos[1]),
            EdNode {
                id: n.id,
                title: n.title.clone(),
                op: n.op.clone(),
            },
        );
        ids.insert(n.id, id);
    }
    for l in &wf.links {
        if let (Some(&a), Some(&b)) = (ids.get(&l.from.0), ids.get(&l.to.0)) {
            snarl.connect(
                OutPinId {
                    node: a,
                    output: l.from.1,
                },
                InPinId {
                    node: b,
                    input: l.to.1,
                },
            );
        }
    }
    snarl
}

/// Canvases made so far, for [`WorkflowEditor::canvas`].
static CANVASES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a node takes on the canvas beyond its position, for fitting the
/// view: about the size of a node with three lines of summary.
const NODE_EXTENT: egui::Vec2 = egui::vec2(250.0, 190.0);

/// What the editor asks the application to do once it has been drawn:
/// anything that needs a file dialog or another window.
pub(super) enum EdAction {
    New,
    Open,
    Save,
    SaveAs,
    Run,
    BrowseFolder(u32),
    BrowseOutput,
}

/// A request that would drop unsaved changes, waiting for a yes.
pub(super) enum WfPending {
    New,
    Open,
    File(PathBuf),
    Example(usize),
}

impl WorkflowEditor {
    /// An editor on a workflow; `path` when it came from a file.
    pub(super) fn from_workflow(wf: Workflow, path: Option<PathBuf>) -> WorkflowEditor {
        let snarl = snarl_of(&wf);
        let max_x = wf.nodes.iter().map(|n| n.pos[0]).fold(0.0f32, f32::max);
        let findings = wf.check();
        let mut ed = WorkflowEditor {
            open: true,
            snarl,
            name: wf.name.clone(),
            description: wf.description.clone(),
            output: wf.output.clone(),
            path,
            saved: String::new(),
            inspect: None,
            findings,
            status: None,
            drop_at: egui::pos2(
                if wf.nodes.is_empty() {
                    40.0
                } else {
                    max_x + 300.0
                },
                40.0,
            ),
            canvas: CANVASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            // Two frames: the canvas's first frame may be discarded while it
            // places itself.
            fit: 2,
            frames: wf.frames.clone(),
            undo: Vec::new(),
            redo: Vec::new(),
            committed: String::new(),
            view: TSTransform::IDENTITY,
            set_view: None,
            rects: Vec::new(),
            minimap: true,
            clip: None,
            frame_drag: None,
        };
        ed.saved = ed.to_workflow().to_json();
        ed.committed = ed.snapshot();
        ed
    }

    /// The whole state the history keeps: the workflow, its name as typed.
    fn snapshot(&self) -> String {
        let mut wf = self.to_workflow();
        wf.name = self.name.clone();
        wf.to_json()
    }

    /// Put `wf` on the canvas in place of what is there, keeping the file,
    /// the view and the history. The canvas is a new one (the selection
    /// named nodes of the old), shown as the old was.
    fn load(&mut self, wf: Workflow) {
        self.snarl = snarl_of(&wf);
        self.name = wf.name;
        self.description = wf.description;
        self.output = wf.output;
        self.frames = wf.frames;
        self.canvas = CANVASES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.set_view = Some(self.view);
        if self
            .inspect
            .is_some_and(|i| !self.snarl.nodes().any(|n| n.id == i))
        {
            self.inspect = None;
        }
    }

    /// Keep the current state as a step of the history when it changed,
    /// unless the mouse is down or a text field has the keyboard (the
    /// change is still being made).
    fn track_history(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.pointer.any_down()) || ctx.egui_wants_keyboard_input() {
            return;
        }
        let now = self.snapshot();
        if now != self.committed {
            self.undo.push(std::mem::replace(&mut self.committed, now));
            if self.undo.len() > HISTORY {
                self.undo.remove(0);
            }
            self.redo.clear();
        }
    }

    /// Keep a change not yet in the history (a title still being typed)
    /// before moving through it, so it is undone rather than lost.
    fn commit_pending(&mut self) {
        let now = self.snapshot();
        if now != self.committed {
            self.undo.push(std::mem::replace(&mut self.committed, now));
            self.redo.clear();
        }
    }

    fn undo(&mut self) {
        self.commit_pending();
        let Some(prev) = self.undo.pop() else {
            return;
        };
        self.redo
            .push(std::mem::replace(&mut self.committed, prev.clone()));
        if let Ok(wf) = Workflow::from_json(&prev) {
            self.load(wf);
        }
        self.status = Some("Undone".into());
    }

    fn redo(&mut self) {
        let now = self.snapshot();
        if now != self.committed {
            // Changed since the undo: that is a new branch, and there is
            // nothing to redo on it.
            self.commit_pending();
            return;
        }
        let Some(next) = self.redo.pop() else {
            return;
        };
        self.undo
            .push(std::mem::replace(&mut self.committed, next.clone()));
        if let Ok(wf) = Workflow::from_json(&next) {
            self.load(wf);
        }
        self.status = Some("Redone".into());
    }

    /// The steps an edit applies to: the selected ones, else the one in
    /// the inspector.
    fn targets(&self, selected: &[NodeId]) -> Vec<NodeId> {
        if !selected.is_empty() {
            return selected.to_vec();
        }
        self.snarl
            .nodes_ids_data()
            .filter(|(_, n)| Some(n.value.id) == self.inspect)
            .map(|(nid, _)| nid)
            .collect()
    }

    /// The workflow ids of canvas nodes.
    fn ids_of(&self, nodes: &[NodeId]) -> std::collections::BTreeSet<u32> {
        nodes
            .iter()
            .filter_map(|n| self.snarl.get_node(*n).map(|x| x.id))
            .collect()
    }

    /// Copy the steps to the clipboard (and keep them for *Paste*).
    fn copy(&mut self, ctx: &egui::Context, nodes: &[NodeId]) -> bool {
        let ids = self.ids_of(nodes);
        if ids.is_empty() {
            return false;
        }
        let part = self.to_workflow().fragment(&ids);
        ctx.copy_text(part.to_json());
        self.status = Some(format!(
            "{} step{} copied",
            ids.len(),
            if ids.len() == 1 { "" } else { "s" }
        ));
        self.clip = Some(part);
        true
    }

    fn remove(&mut self, nodes: &[NodeId]) {
        for n in nodes {
            if self.snarl.get_node(*n).is_some() {
                self.snarl.remove_node(*n);
            }
        }
        self.inspect = None;
        self.prune_frames();
    }

    /// Put copied steps on the canvas: their top-left corner at `at`
    /// (canvas units), or a little down and right of where they were.
    fn paste(&mut self, part: &Workflow, at: Option<Pos2>) {
        if part.nodes.is_empty() {
            return;
        }
        let min = part.nodes.iter().fold([f32::MAX, f32::MAX], |m, n| {
            [m[0].min(n.pos[0]), m[1].min(n.pos[1])]
        });
        let offset = match at {
            Some(p) => [p.x - min[0], p.y - min[1]],
            None => [40.0, 40.0],
        };
        let mut wf = self.to_workflow();
        wf.name = self.name.clone();
        let ids = wf.paste(part, offset);
        self.load(wf);
        self.inspect = ids.values().next().copied();
        self.status = Some(format!(
            "{} step{} pasted",
            ids.len(),
            if ids.len() == 1 { "" } else { "s" }
        ));
    }

    /// Frames lose the steps that are gone, and go with the last of them.
    fn prune_frames(&mut self) {
        let have: std::collections::BTreeSet<u32> = self.snarl.nodes().map(|n| n.id).collect();
        for f in &mut self.frames {
            f.nodes.retain(|n| have.contains(n));
        }
        self.frames.retain(|f| !f.nodes.is_empty());
    }

    /// The canvas as a workflow: nodes in id order, wires as links.
    pub(super) fn to_workflow(&self) -> Workflow {
        let mut wf = Workflow::new(if self.name.trim().is_empty() {
            "Untitled workflow"
        } else {
            self.name.trim()
        });
        wf.description = self.description.clone();
        wf.output = self.output.clone();
        let mut by_node: BTreeMap<NodeId, u32> = BTreeMap::new();
        for (nid, pos, n) in self.snarl.nodes_pos_ids() {
            by_node.insert(nid, n.id);
            wf.nodes.push(WfNode {
                id: n.id,
                title: n.title.clone(),
                pos: [pos.x.round(), pos.y.round()],
                op: n.op.clone(),
            });
        }
        wf.nodes.sort_by_key(|n| n.id);
        wf.frames = self.frames.clone();
        wf.prune_frames();
        for (out, inp) in self.snarl.wires() {
            if let (Some(&a), Some(&b)) = (by_node.get(&out.node), by_node.get(&inp.node)) {
                wf.link(a, out.output, b, inp.input);
            }
        }
        wf.links.sort();
        wf
    }

    /// Changed since it was last saved or opened?
    pub(super) fn dirty(&self) -> bool {
        self.to_workflow().to_json() != self.saved
    }

    /// It was just saved as `path`.
    pub(super) fn mark_saved(&mut self, path: PathBuf) {
        self.path = Some(path);
        self.saved = self.to_workflow().to_json();
    }

    fn next_id(&self) -> u32 {
        self.snarl.nodes().map(|n| n.id).max().unwrap_or(0) + 1
    }

    /// Add a node of `kind` at `pos`; returns its canvas id.
    fn add(&mut self, kind: Kind, pos: Pos2) -> NodeId {
        let id = self.next_id();
        let nid = self.snarl.insert_node(
            pos,
            EdNode {
                id,
                title: String::new(),
                op: kind.default_op(),
            },
        );
        self.inspect = Some(id);
        nid
    }

    fn node_mut(&mut self, id: u32) -> Option<&mut EdNode> {
        self.snarl.nodes_mut().find(|n| n.id == id)
    }

    /// Point a folder node at `path`.
    pub(super) fn set_folder(&mut self, id: u32, path: &Path) {
        if let Some(n) = self.node_mut(id) {
            match &mut n.op {
                Op::LoadFolder(p) => p.path = path.display().to_string(),
                Op::LoadFolders(p) => p.path = path.display().to_string(),
                _ => {}
            }
        }
    }

    /// The window's title: the workflow's name, starred while unsaved.
    pub(super) fn title(&self) -> String {
        format!(
            "🔀 Workflow: {}{}",
            if self.name.trim().is_empty() {
                "untitled"
            } else {
                self.name.trim()
            },
            if self.dirty() { " *" } else { "" }
        )
    }

    /// Draw the whole editor into `ui` (the tool window's root).
    pub(super) fn ui(
        &mut self,
        ui: &mut egui::Ui,
        marks: &BTreeMap<u32, Mark>,
        running: bool,
        actions: &mut Vec<EdAction>,
    ) {
        self.findings = self.to_workflow().check();
        egui::Panel::top(egui::Id::new("wf_top")).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(240.0));
                ui.separator();
                if tip_button(ui, "➕ New", "Start an empty workflow") {
                    actions.push(EdAction::New);
                }
                if tip_button(ui, "📂 Load", "Open a workflow file") {
                    actions.push(EdAction::Open);
                }
                if tip_button(
                    ui,
                    "💾 Save",
                    "Save to the file it came from, or ask for one (your workflow folder)",
                ) {
                    actions.push(EdAction::Save);
                }
                if tip_button(ui, "💾 Save as", "Save under another name") {
                    actions.push(EdAction::SaveAs);
                }
                if tip_button(
                    ui,
                    "⛶ Fit",
                    "Show every step (double-click on empty canvas does the same)",
                ) {
                    self.fit = 2;
                }
                ui.separator();
                if ui
                    .add_enabled(!self.undo.is_empty(), egui::Button::new("⟲ Undo"))
                    .on_hover_text("Take back the last change (Ctrl+Z)")
                    .clicked()
                {
                    self.undo();
                }
                if ui
                    .add_enabled(!self.redo.is_empty(), egui::Button::new("⟳ Redo"))
                    .on_hover_text("Make the change taken back again (Ctrl+Shift+Z, Ctrl+Y)")
                    .clicked()
                {
                    self.redo();
                }
                let selected =
                    get_selected_nodes(egui::Id::new((CANVAS_ID, self.canvas)), ui.ctx());
                let targets = self.targets(&selected);
                if ui
                    .add_enabled(!targets.is_empty(), egui::Button::new("⬚ Frame"))
                    .on_hover_text(
                        "Draw a titled frame around the selected steps (Shift-click or \
                         Shift-drag to select several). Its title is set under Frames on the \
                         left; dragging the title moves every step inside",
                    )
                    .clicked()
                {
                    let nodes = self.ids_of(&targets).into_iter().collect();
                    self.frames.push(WfFrame {
                        title: format!("Frame {}", self.frames.len() + 1),
                        nodes,
                        ..WfFrame::default()
                    });
                }
                ui.checkbox(&mut self.minimap, "Map")
                    .on_hover_text("A map of the whole canvas in its corner");
                ui.separator();
                let ok = self.findings.ok();
                if ui
                    .add_enabled(!running, egui::Button::new("▶ Run"))
                    .on_hover_text(if running {
                        "A workflow is running; wait for it or cancel it in its window".to_string()
                    } else if ok {
                        "Choose the input folders and how to run, then run".to_string()
                    } else {
                        "Open the run dialog. Its folders may fill what is missing; anything \
                         else listed at the bottom has to be fixed first"
                            .to_string()
                    })
                    .clicked()
                {
                    actions.push(EdAction::Run);
                }
            });
            if let Some(p) = &self.path {
                ui.weak(p.display().to_string());
            }
        });
        egui::Panel::bottom(egui::Id::new("wf_bottom")).show(ui, |ui| {
            self.findings_ui(ui);
        });
        egui::Panel::left(egui::Id::new("wf_palette"))
            .resizable(true)
            .default_size(180.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.palette_ui(ui, actions);
                });
            });
        egui::Panel::right(egui::Id::new("wf_inspector"))
            .resizable(true)
            .default_size(310.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.inspector_ui(ui, actions);
                });
            });
        egui::CentralPanel::default().show(ui, |ui| {
            self.canvas_ui(ui, marks);
        });
        // Undo and redo wherever the pointer is in the editor, unless a
        // text field has the keyboard (it undoes its own typing). By
        // position: the canvas's layers lie over this one.
        let inside = ui
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| ui.max_rect().contains(p));
        if inside && !ui.ctx().egui_wants_keyboard_input() {
            use egui::{Key, KeyboardShortcut as Ks, Modifiers as M};
            // Redo first: Ctrl+Z also matches with Shift held.
            let (redo, undo) = ui.input_mut(|i| {
                let redo = i.consume_shortcut(&Ks::new(M::COMMAND | M::SHIFT, Key::Z))
                    || i.consume_shortcut(&Ks::new(M::COMMAND, Key::Y));
                (
                    redo,
                    !redo && i.consume_shortcut(&Ks::new(M::COMMAND, Key::Z)),
                )
            });
            if redo {
                self.redo();
            } else if undo {
                self.undo();
            }
        }
        self.prune_frames();
        self.track_history(ui.ctx());
    }

    fn findings_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if let Some(s) = &self.status {
                ui.label(s);
                ui.separator();
            }
            let f = &self.findings;
            if f.errors.is_empty() && f.warnings.is_empty() {
                ui.label(egui::RichText::new("✔ Ready to run").color(ok_color(ui)));
            } else {
                for (_, m) in f.errors.iter().take(3) {
                    ui.label(
                        egui::RichText::new(format!("✖ {m}")).color(ui.visuals().error_fg_color),
                    );
                }
                for (_, m) in f.warnings.iter().take(2) {
                    ui.label(
                        egui::RichText::new(format!("⚠ {m}")).color(ui.visuals().warn_fg_color),
                    );
                }
                let more = f.errors.len().saturating_sub(3) + f.warnings.len().saturating_sub(2);
                if more > 0 {
                    ui.weak(format!("and {more} more"));
                }
            }
        });
    }

    fn palette_ui(&mut self, ui: &mut egui::Ui, actions: &mut Vec<EdAction>) {
        ui.strong("Steps");
        ui.weak(
            "Click to add, or right-click the canvas. Wire an output to an input. \
             Delete removes the step shown on the right (or the ones Shift-clicked); \
             Ctrl+C, Ctrl+X, Ctrl+V and Ctrl+D copy, cut, paste and duplicate them, the \
             arrow keys move them; double-click empty canvas to see every step.",
        );
        for c in Category::ALL {
            egui::CollapsingHeader::new(c.label())
                .default_open(true)
                .show(ui, |ui| {
                    for k in Kind::of(c) {
                        let info = k.info();
                        if ui
                            .button(format!("{} {}", info.glyph, info.name))
                            .on_hover_text(info.blurb)
                            .clicked()
                        {
                            let at = self.drop_at;
                            self.add(k, at);
                            self.drop_at = at + egui::vec2(30.0, 30.0);
                        }
                    }
                });
        }
        if !self.frames.is_empty() {
            ui.separator();
            egui::CollapsingHeader::new("Frames")
                .default_open(true)
                .show(ui, |ui| {
                    let mut gone = None;
                    for (k, f) in self.frames.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.color_edit_button_srgb(&mut f.color);
                            ui.add(egui::TextEdit::singleline(&mut f.title).desired_width(90.0));
                            if ui
                                .small_button("🗑")
                                .on_hover_text("Remove the frame (the steps stay)")
                                .clicked()
                            {
                                gone = Some(k);
                            }
                        });
                    }
                    if let Some(k) = gone {
                        self.frames.remove(k);
                    }
                });
        }
        ui.separator();
        egui::CollapsingHeader::new("Workflow")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("Description");
                ui.add(
                    egui::TextEdit::multiline(&mut self.description)
                        .desired_rows(4)
                        .desired_width(f32::INFINITY),
                );
                ui.add_space(4.0);
                ui.label("Results go to")
                    .on_hover_text("The folder each run makes its own folder in. Empty is the program's workflow_runs folder.");
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.output.root)
                            .hint_text("the program's folder")
                            .desired_width(130.0),
                    );
                    if ui.small_button("📂").on_hover_text("Choose the folder").clicked() {
                        actions.push(EdAction::BrowseOutput);
                    }
                });
                ui.label("Run folder name").on_hover_text(
                    "{workflow}, {date} and {time} are filled in when a run starts",
                );
                ui.add(
                    egui::TextEdit::singleline(&mut self.output.folder)
                        .desired_width(f32::INFINITY),
                );
            });
    }

    fn inspector_ui(&mut self, ui: &mut egui::Ui, actions: &mut Vec<EdAction>) {
        let Some(id) = self.inspect else {
            ui.weak("Select a step to see and change what it does.");
            ui.add_space(8.0);
            ui.weak(
                "Wires carry what one step makes into the next: pins of one colour carry \
                 one kind of thing.",
            );
            for t in [
                PortType::Study,
                PortType::Image,
                PortType::Group,
                PortType::Structures,
                PortType::Registration,
                PortType::Report,
            ] {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("●").color(type_color(t)));
                    ui.label(t.label());
                });
            }
            return;
        };
        let problems: Vec<String> = self.findings.of(id).iter().map(|s| s.to_string()).collect();
        let Some(n) = self.node_mut(id) else {
            self.inspect = None;
            return;
        };
        let info = n.op.kind().info();
        ui.heading(format!("{} {}", info.glyph, info.name));
        ui.label(info.blurb);
        ui.add_space(4.0);
        form::form(ui, ("wf_title", id), |f| {
            f.row("Title", |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut n.title)
                        .hint_text(info.name)
                        .desired_width(180.0),
                );
            });
        });
        ui.separator();
        params_ui(ui, id, &mut n.op, actions);
        ui.separator();
        let ins = n.op.inputs();
        if !ins.is_empty() {
            ui.strong("Inputs");
            for s in ins {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("●").color(type_color(s.accepts[0])));
                    ui.label(format!(
                        "{}{}: {}",
                        s.name,
                        if s.optional { " (optional)" } else { "" },
                        s.hint
                    ));
                });
            }
        }
        let outs = n.op.outputs();
        if !outs.is_empty() {
            ui.strong("Outputs");
            for s in outs {
                ui.horizontal_wrapped(|ui| {
                    ui.label(egui::RichText::new("●").color(type_color(s.ty)));
                    ui.label(format!("{}: {}", s.name, s.hint));
                });
            }
        }
        if !problems.is_empty() {
            ui.separator();
            for p in problems {
                ui.label(egui::RichText::new(format!("⚠ {p}")).color(ui.visuals().warn_fg_color));
            }
        }
    }

    fn canvas_ui(&mut self, ui: &mut egui::Ui, marks: &BTreeMap<u32, Mark>) {
        let style = canvas_style();
        let fit = if self.fit > 0 {
            self.fit -= 1;
            let mut bb = Rect::NOTHING;
            for (_, pos, _) in self.snarl.nodes_pos_ids() {
                bb.extend_with(pos);
                bb.extend_with(pos + NODE_EXTENT);
            }
            bb.is_finite().then(|| (bb.expand(30.0), ui.max_rect()))
        } else {
            None
        };
        let id = egui::Id::new((CANVAS_ID, self.canvas));
        // The canvas's area, taken before the canvas: its response covers
        // an unbounded rectangle (the canvas is endless).
        let canvas = ui.max_rect();
        let mut viewer = Viewer {
            marks,
            findings: &self.findings,
            refused: None,
            clicked: None,
            fit,
            set_view: self.set_view.take(),
            to_global: TSTransform::IDENTITY,
            rects: Vec::new(),
            frames: &self.frames,
            last_rects: &self.rects,
        };
        let response =
            SnarlWidget::new()
                .id(id)
                .style(style)
                .show(&mut self.snarl, &mut viewer, ui);
        let refused = viewer.refused.take();
        let clicked = viewer.clicked;
        let to_global = viewer.to_global;
        let rects = std::mem::take(&mut viewer.rects);
        self.view = to_global;
        self.rects = rects;
        if let Some(why) = refused {
            self.status = Some(why);
        }
        if let Some(c) = clicked {
            self.inspect = Some(c);
        }
        // What the canvas holds is drawn over by a layer of its own: the
        // frames' handles and the map, which have to win the pointer over
        // the canvas (a sublayer added after the canvas's is above it).
        if canvas.is_finite() && canvas.is_positive() {
            let overlay = egui::LayerId::new(ui.layer_id().order, id.with("overlay"));
            ui.ctx().set_sublayer(ui.layer_id(), overlay);
            let mut over = ui.new_child(egui::UiBuilder::new().layer_id(overlay).max_rect(canvas));
            over.set_clip_rect(canvas);
            self.frame_handles(&mut over, id);
            if self.minimap && !self.rects.is_empty() {
                self.minimap_ui(&mut over, canvas, id);
            }
        }
        // A click on a node shows it in the inspector: the pointer taken
        // back into canvas units, the topmost node under it.
        let click = ui.input(|i| {
            if i.pointer.primary_clicked() {
                i.pointer.interact_pos()
            } else {
                None
            }
        });
        if let Some(at) = click.filter(|_| response.contains_pointer()) {
            let on_canvas = to_global.inverse() * at;
            if let Some((id, _)) = self.rects.iter().rev().find(|(_, r)| r.contains(on_canvas)) {
                self.inspect = Some(*id);
            }
        }
        // The selection drives the inspector: one node selected is the one
        // shown; clicking empty space keeps the last one.
        let selected = get_selected_nodes(id, ui.ctx());
        // The keys act while the pointer is over the canvas and no text
        // field has the keyboard.
        let over_canvas = ui
            .input(|i| i.pointer.hover_pos())
            .is_some_and(|p| canvas.contains(p));
        if over_canvas && !ui.ctx().egui_wants_keyboard_input() {
            self.canvas_keys(ui, &selected, to_global);
        }
        if selected.len() == 1 {
            if let Some(n) = self.snarl.get_node(selected[0]) {
                self.inspect = Some(n.id);
            }
        }
        if self
            .inspect
            .is_some_and(|i| !self.snarl.nodes().any(|n| n.id == i))
        {
            self.inspect = None;
        }
    }

    /// Delete, copy, cut, paste, duplicate and the arrow keys on the
    /// canvas. Delete and the arrows act on the selected steps (Shift-click
    /// or a Shift-drag selects several), or else the one in the inspector.
    fn canvas_keys(&mut self, ui: &egui::Ui, selected: &[NodeId], to_global: TSTransform) {
        use egui::{Key, KeyboardShortcut as Ks, Modifiers as M};
        let targets = self.targets(selected);
        let (copy, cut, paste, delete, dup, nudge, pointer) = ui.input_mut(|i| {
            let mut copy = false;
            let mut cut = false;
            let mut paste = None;
            for e in &i.events {
                match e {
                    egui::Event::Copy => copy = true,
                    egui::Event::Cut => cut = true,
                    egui::Event::Paste(t) => paste = Some(t.clone()),
                    _ => {}
                }
            }
            let dup = i.consume_shortcut(&Ks::new(M::COMMAND, Key::D));
            // Every arrow press of the frame, each with the Shift it was
            // pressed with (several can arrive between two frames).
            let mut nudge = egui::Vec2::ZERO;
            for e in &i.events {
                if let egui::Event::Key {
                    key,
                    pressed: true,
                    modifiers,
                    ..
                } = e
                {
                    if modifiers.command {
                        continue;
                    }
                    let step = if modifiers.shift { 40.0 } else { 10.0 };
                    nudge += match key {
                        Key::ArrowLeft => egui::vec2(-step, 0.0),
                        Key::ArrowRight => egui::vec2(step, 0.0),
                        Key::ArrowUp => egui::vec2(0.0, -step),
                        Key::ArrowDown => egui::vec2(0.0, step),
                        _ => egui::Vec2::ZERO,
                    };
                }
            }
            (
                copy,
                cut,
                paste,
                i.key_pressed(Key::Delete),
                dup,
                nudge,
                i.pointer.hover_pos(),
            )
        });
        if delete {
            self.remove(&targets);
            return;
        }
        if copy || cut {
            let done = self.copy(ui.ctx(), &targets);
            if cut && done {
                self.remove(&targets);
                self.status = Some("Cut; Ctrl+V puts the steps back".into());
            }
        }
        if dup && !targets.is_empty() {
            let part = self.to_workflow().fragment(&self.ids_of(&targets));
            self.paste(&part, None);
        }
        if let Some(text) = paste {
            match Workflow::from_json(&text) {
                Ok(part) if !part.nodes.is_empty() => {
                    let at = pointer.map(|p| to_global.inverse() * p);
                    self.paste(&part, at);
                }
                _ => match self.clip.clone() {
                    // Another program's text on the clipboard: what this
                    // editor copied last, if anything.
                    Some(part) => {
                        let at = pointer.map(|p| to_global.inverse() * p);
                        self.paste(&part, at);
                    }
                    None => {
                        self.status = Some("The clipboard holds no workflow steps".into());
                    }
                },
            }
        }
        if nudge != egui::Vec2::ZERO && !targets.is_empty() {
            for n in &targets {
                if let Some(info) = self.snarl.get_node_info_mut(*n) {
                    info.pos += nudge;
                }
            }
        }
    }

    /// The frames' title bars as handles: dragging one moves every step in
    /// its frame. The steps follow the pointer from where the drag began
    /// (not frame by frame), so a frame whose first delta also holds the
    /// move before the press, or a second pass of the same frame, does not
    /// throw them off.
    fn frame_handles(&mut self, over: &mut egui::Ui, id: egui::Id) {
        let mut started = None;
        let mut at = None;
        let mut stopped = false;
        for (k, f) in self.frames.iter().enumerate() {
            let Some(b) = frame_box(f, &self.rects) else {
                continue;
            };
            let bar = (self.view * frame_title(b)).intersect(over.clip_rect());
            if !bar.is_positive() {
                continue;
            }
            let r = over
                .interact(bar, id.with(("frame", k)), egui::Sense::drag())
                .on_hover_cursor(egui::CursorIcon::Grab)
                .on_hover_text("Drag to move the steps in this frame");
            if r.drag_started() {
                let from = over
                    .input(|i| i.pointer.press_origin())
                    .or(r.interact_pointer_pos());
                started = from.map(|p| (k, p));
            }
            if r.dragged() {
                at = r.interact_pointer_pos().map(|p| (k, p));
            }
            stopped |= r.drag_stopped();
        }
        if let Some((k, from)) = started {
            let ids = &self.frames[k].nodes;
            let origins = self
                .snarl
                .nodes_info()
                .filter(|n| ids.contains(&n.value.id))
                .map(|n| (n.value.id, n.pos))
                .collect();
            self.frame_drag = Some(FrameDrag {
                frame: k,
                from,
                origins,
            });
        }
        if let (Some((k, p)), Some(drag)) = (at, &self.frame_drag) {
            if k == drag.frame {
                let d = (p - drag.from) / self.view.scaling;
                for n in self.snarl.nodes_info_mut() {
                    if let Some((_, o)) = drag.origins.iter().find(|(id, _)| *id == n.value.id) {
                        n.pos = *o + d;
                    }
                }
            }
        }
        if stopped {
            self.frame_drag = None;
        }
    }

    /// The map: every step and frame, scaled into the canvas's corner, with
    /// the part in view outlined. A click or a drag on it centres the view
    /// there.
    fn minimap_ui(&mut self, over: &mut egui::Ui, canvas: Rect, id: egui::Id) {
        let size = egui::vec2(180.0, 120.0);
        let mini = Rect::from_min_size(canvas.right_bottom() - size - egui::vec2(10.0, 10.0), size);
        if !canvas.contains_rect(mini) || canvas.width() < 400.0 {
            return;
        }
        let view = self.view.inverse() * canvas;
        let mut world = view;
        for (_, r) in &self.rects {
            world = world.union(*r);
        }
        let world = world.expand(40.0);
        let inner = mini.shrink(6.0);
        let scale = (inner.width() / world.width()).min(inner.height() / world.height());
        let to_mini = |r: Rect| {
            Rect::from_min_max(
                inner.center() + (r.min - world.center()) * scale,
                inner.center() + (r.max - world.center()) * scale,
            )
        };
        let visuals = over.visuals().clone();
        let painter = over.painter();
        painter.rect(
            mini,
            6.0,
            visuals.window_fill.gamma_multiply(0.92),
            visuals.window_stroke,
            egui::StrokeKind::Inside,
        );
        for f in &self.frames {
            if let Some(b) = frame_box(f, &self.rects) {
                painter.rect_filled(to_mini(b), 2.0, rgb(f.color).gamma_multiply(0.35));
            }
        }
        let colors: BTreeMap<u32, Color32> = self
            .snarl
            .nodes()
            .map(|n| (n.id, rgb(n.op.kind().info().category.color())))
            .collect();
        for (nid, r) in &self.rects {
            let c = colors.get(nid).copied().unwrap_or(Color32::GRAY);
            painter.rect_filled(to_mini(*r), 1.0, c);
        }
        painter.rect_stroke(
            to_mini(view).intersect(inner.expand(2.0)),
            2.0,
            egui::Stroke::new(1.5, visuals.strong_text_color()),
            egui::StrokeKind::Middle,
        );
        let r = over
            .interact(mini, id.with("minimap"), egui::Sense::click_and_drag())
            .on_hover_text(
                "The whole canvas; the outline is the part in view. Click or drag to move the view",
            );
        if r.clicked() || r.dragged() {
            if let Some(p) = r.interact_pointer_pos() {
                let at = world.center() + (p - inner.center()) / scale;
                let scaling = self.view.scaling;
                self.set_view = Some(TSTransform {
                    scaling,
                    translation: canvas.center().to_vec2() - at.to_vec2() * scaling,
                });
                over.ctx().request_repaint();
            }
        }
    }
}

/// Room around a frame's steps, and its title bar's height, canvas units.
const FRAME_PAD: f32 = 18.0;
const FRAME_TITLE: f32 = 26.0;

/// A frame's box: around its steps as last drawn, with its title bar on
/// top. `None` while none of its steps has been drawn.
fn frame_box(f: &WfFrame, rects: &[(u32, Rect)]) -> Option<Rect> {
    let mut bb = Rect::NOTHING;
    for (id, r) in rects {
        if f.nodes.contains(id) {
            bb = bb.union(*r);
        }
    }
    bb.is_positive().then(|| {
        Rect::from_min_max(
            bb.min - egui::vec2(FRAME_PAD, FRAME_PAD + FRAME_TITLE),
            bb.max + egui::vec2(FRAME_PAD, FRAME_PAD),
        )
    })
}

/// The title bar of a frame's box.
fn frame_title(b: Rect) -> Rect {
    Rect::from_min_size(b.min, egui::vec2(b.width(), FRAME_TITLE))
}

/// The canvas's look: pins on the node edges, round, a little larger than
/// the default; a light grid behind; zoom from a fifth to one and a half.
fn canvas_style() -> SnarlStyle {
    let mut s = SnarlStyle::new();
    // Header, outputs, the summary, inputs: the node editor layout people
    // know from Blender, and narrower than inputs and outputs side by side.
    s.node_layout = Some(NodeLayout::flipped_sandwich());
    s.max_scale = Some(1.5);
    s.pin_placement = Some(PinPlacement::Edge);
    s.pin_size = Some(9.0);
    s.wire_width = Some(2.5);
    s.collapsible = Some(true);
    s.bg_pattern = Some(BackgroundPattern::Grid(Grid {
        spacing: egui::vec2(40.0, 40.0),
        angle: 0.0,
    }));
    s
}

fn rgb(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

pub(super) fn type_color(t: PortType) -> Color32 {
    rgb(t.color())
}

fn ok_color(ui: &egui::Ui) -> Color32 {
    if ui.visuals().dark_mode {
        Color32::from_rgb(110, 210, 120)
    } else {
        Color32::from_rgb(30, 130, 50)
    }
}

/// Would a wire from node `from` to node `to` close a circle on the canvas?
fn closes_circle(snarl: &Snarl<EdNode>, from: NodeId, to: NodeId) -> bool {
    if from == to {
        return true;
    }
    let mut stack = vec![to];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(n) = stack.pop() {
        if n == from {
            return true;
        }
        if !seen.insert(n) {
            continue;
        }
        stack.extend(
            snarl
                .wires()
                .filter(|(o, _)| o.node == n)
                .map(|(_, i)| i.node),
        );
    }
    false
}

/// The canvas's side of the conversation with `egui_snarl`.
struct Viewer<'a> {
    marks: &'a BTreeMap<u32, Mark>,
    findings: &'a Findings,
    /// Why the last wire was refused, for the status line.
    refused: Option<String>,
    /// A header was clicked: inspect that node.
    clicked: Option<u32>,
    /// Set the view to show this part of the canvas (canvas units) in this
    /// rectangle of the window.
    fit: Option<(Rect, Rect)>,
    /// Set the view to this (the map's, or the one kept across a rebuilt
    /// canvas).
    set_view: Option<TSTransform>,
    /// Canvas to window, as the canvas was drawn this frame.
    to_global: TSTransform,
    /// Every node's rectangle this frame, in canvas units, in drawing order.
    rects: Vec<(u32, Rect)>,
    /// The frames, drawn behind the nodes around where they were last
    /// drawn.
    frames: &'a [WfFrame],
    last_rects: &'a [(u32, Rect)],
}

impl SnarlViewer<EdNode> for Viewer<'_> {
    fn current_transform(&mut self, to_global: &mut TSTransform, _snarl: &mut Snarl<EdNode>) {
        if let Some(v) = self.set_view.take() {
            *to_global = v;
        }
        // Every step in view, no larger than life and not so small that the
        // summaries cannot be read; what does not fit is a pan away.
        if let Some((bb, rect)) = self.fit.take() {
            let scaling = (rect.width() / bb.width())
                .min(rect.height() / bb.height())
                .clamp(0.55, 1.0);
            *to_global = TSTransform {
                scaling,
                translation: rect.center().to_vec2() - bb.center().to_vec2() * scaling,
            };
        }
        self.to_global = *to_global;
    }

    fn draw_background(
        &mut self,
        background: Option<&BackgroundPattern>,
        viewport: &Rect,
        snarl_style: &SnarlStyle,
        style: &egui::Style,
        painter: &egui::Painter,
        _snarl: &Snarl<EdNode>,
    ) {
        if let Some(b) = background {
            b.draw(viewport, snarl_style, style, painter);
        }
        // The frames: behind every node, in canvas units (the painter is
        // the canvas's).
        for f in self.frames {
            let Some(b) = frame_box(f, self.last_rects) else {
                continue;
            };
            let c = rgb(f.color);
            painter.rect(
                b,
                8.0,
                c.gamma_multiply(0.16),
                egui::Stroke::new(1.5, c.gamma_multiply(0.8)),
                egui::StrokeKind::Inside,
            );
            let bar = frame_title(b);
            painter.rect_filled(
                bar,
                egui::CornerRadius {
                    nw: 8,
                    ne: 8,
                    sw: 0,
                    se: 0,
                },
                c.gamma_multiply(0.7),
            );
            painter.text(
                bar.left_center() + egui::vec2(10.0, 0.0),
                egui::Align2::LEFT_CENTER,
                &f.title,
                egui::FontId::proportional(15.0),
                Color32::WHITE,
            );
        }
    }

    fn final_node_rect(
        &mut self,
        node: NodeId,
        rect: Rect,
        _ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        self.rects.push((snarl[node].id, rect));
    }

    fn title(&mut self, node: &EdNode) -> String {
        format!("{} {}", node.op.kind().info().glyph, node.label())
    }

    fn show_header(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        let n = &snarl[node];
        let mark = match self.marks.get(&n.id) {
            Some(Mark::Running) => " ⏳",
            Some(Mark::Done(_)) => " ✔",
            Some(Mark::Reused) => " ♻",
            Some(Mark::Failed(_)) => " ✖",
            _ => "",
        };
        let text = egui::RichText::new(format!("{}{mark}", self.title(n)))
            .strong()
            .color(Color32::WHITE);
        // A plain label: sensing clicks here would take the drag that moves
        // the node away from the frame. Which node a click lands on is
        // worked out from the node rectangles after the canvas is drawn.
        let r = ui.label(text);
        if let Some(Mark::Failed(e)) = self.marks.get(&n.id) {
            r.on_hover_text(e.as_str());
        }
    }

    fn header_frame(
        &mut self,
        frame: egui::Frame,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        snarl: &Snarl<EdNode>,
    ) -> egui::Frame {
        frame.fill(rgb(snarl[node].op.kind().info().category.color()))
    }

    fn node_frame(
        &mut self,
        frame: egui::Frame,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        snarl: &Snarl<EdNode>,
    ) -> egui::Frame {
        let id = snarl[node].id;
        match self.marks.get(&id) {
            Some(Mark::Running) => {
                frame.stroke(egui::Stroke::new(3.0, Color32::from_rgb(240, 180, 40)))
            }
            Some(Mark::Failed(_)) => {
                frame.stroke(egui::Stroke::new(3.0, Color32::from_rgb(220, 60, 60)))
            }
            _ if self.findings.node_has_error(id) => {
                frame.stroke(egui::Stroke::new(1.5, Color32::from_rgb(200, 90, 60)))
            }
            _ => frame,
        }
    }

    fn inputs(&mut self, node: &EdNode) -> usize {
        node.op.inputs().len()
    }

    fn outputs(&mut self, node: &EdNode) -> usize {
        node.op.outputs().len()
    }

    #[allow(refining_impl_trait)]
    fn show_input(&mut self, pin: &InPin, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) -> PinInfo {
        let spec = snarl[pin.id.node].op.inputs()[pin.id.input];
        // The pin takes the colour of what arrives, or of the first thing
        // it accepts while nothing does.
        let arriving = pin.remotes.first().and_then(|r| {
            snarl
                .get_node(r.node)
                .and_then(|n| n.op.outputs().get(r.output).map(|o| o.ty))
        });
        let color = type_color(arriving.unwrap_or(spec.accepts[0]));
        let text = if spec.optional {
            egui::RichText::new(spec.name).italics()
        } else {
            egui::RichText::new(spec.name)
        };
        ui.label(text).on_hover_text(format!(
            "{} ({}{}): {}",
            spec.name,
            spec.accepts_text(),
            if spec.many { ", several wires" } else { "" },
            spec.hint
        ));
        let info = PinInfo::circle().with_fill(color);
        if spec.many {
            info.with_shape(PinShape::Square)
        } else {
            info
        }
    }

    #[allow(refining_impl_trait)]
    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) -> PinInfo {
        let spec = snarl[pin.id.node].op.outputs()[pin.id.output];
        ui.label(spec.name).on_hover_text(format!(
            "{} ({}): {}",
            spec.name,
            spec.ty.label(),
            spec.hint
        ));
        let c = type_color(spec.ty);
        PinInfo::circle().with_fill(c).with_wire_color(c)
    }

    fn has_body(&mut self, _node: &EdNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        let n = &snarl[node];
        ui.vertical(|ui| {
            ui.set_max_width(210.0);
            for line in n.op.summary().iter().take(3) {
                ui.add(egui::Label::new(egui::RichText::new(line).weak()).truncate());
            }
            match self.marks.get(&n.id) {
                Some(Mark::Done(secs)) => {
                    ui.label(egui::RichText::new(format!("✔ {secs:.1} s")).small());
                }
                Some(Mark::Reused) => {
                    ui.label(egui::RichText::new("♻ unchanged since the last run").small());
                }
                Some(Mark::Failed(e)) => {
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(format!("✖ {e}"))
                                .small()
                                .color(ui.visuals().error_fg_color),
                        )
                        .truncate(),
                    );
                }
                _ => {
                    if let Some(p) = self.findings.of(n.id).first() {
                        ui.add(
                            egui::Label::new(
                                egui::RichText::new(format!("⚠ {p}"))
                                    .small()
                                    .color(ui.visuals().warn_fg_color),
                            )
                            .truncate(),
                        );
                    }
                }
            }
        });
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<EdNode>) {
        let Some(out_ty) = snarl
            .get_node(from.id.node)
            .and_then(|n| n.op.outputs().get(from.id.output).map(|o| o.ty))
        else {
            return;
        };
        let Some(spec) = snarl
            .get_node(to.id.node)
            .and_then(|n| n.op.inputs().get(to.id.input).copied())
        else {
            return;
        };
        if !spec.accepts.contains(&out_ty) {
            self.refused = Some(format!(
                "'{}' takes {}, not {}",
                spec.name,
                spec.accepts_text(),
                out_ty.label()
            ));
            return;
        }
        if closes_circle(snarl, from.id.node, to.id.node) {
            self.refused = Some("that wire would close a circle; a workflow flows one way".into());
            return;
        }
        if !spec.many {
            for &remote in &to.remotes {
                snarl.disconnect(remote, to.id);
            }
        }
        snarl.connect(from.id, to.id);
        self.refused = None;
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<EdNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<EdNode>) {
        ui.label("Add a step");
        for c in Category::ALL {
            ui.menu_button(c.label(), |ui| {
                for k in Kind::of(c) {
                    let info = k.info();
                    if ui
                        .button(format!("{} {}", info.glyph, info.name))
                        .on_hover_text(info.blurb)
                        .clicked()
                    {
                        let id = snarl.nodes().map(|n| n.id).max().unwrap_or(0) + 1;
                        snarl.insert_node(
                            pos,
                            EdNode {
                                id,
                                title: String::new(),
                                op: k.default_op(),
                            },
                        );
                        self.clicked = Some(id);
                        ui.close();
                    }
                }
            });
        }
    }

    fn has_dropped_wire_menu(&mut self, _src: AnyPins, _snarl: &mut Snarl<EdNode>) -> bool {
        true
    }

    fn show_dropped_wire_menu(
        &mut self,
        pos: Pos2,
        ui: &mut egui::Ui,
        src: AnyPins,
        snarl: &mut Snarl<EdNode>,
    ) {
        ui.label("Add a step that takes it");
        match src {
            AnyPins::Out(pins) => {
                let Some(first) = pins.first() else {
                    return;
                };
                let Some(ty) = snarl
                    .get_node(first.node)
                    .and_then(|n| n.op.outputs().get(first.output).map(|o| o.ty))
                else {
                    return;
                };
                for k in Kind::ALL {
                    let Some(input) = k.inputs().iter().position(|s| s.accepts.contains(&ty))
                    else {
                        continue;
                    };
                    let info = k.info();
                    if ui.button(format!("{} {}", info.glyph, info.name)).clicked() {
                        let id = snarl.nodes().map(|n| n.id).max().unwrap_or(0) + 1;
                        let nid = snarl.insert_node(
                            pos,
                            EdNode {
                                id,
                                title: String::new(),
                                op: k.default_op(),
                            },
                        );
                        for p in pins {
                            snarl.connect(*p, InPinId { node: nid, input });
                        }
                        self.clicked = Some(id);
                        ui.close();
                    }
                }
            }
            AnyPins::In(pins) => {
                let Some(first) = pins.first() else {
                    return;
                };
                let Some(spec) = snarl
                    .get_node(first.node)
                    .and_then(|n| n.op.inputs().get(first.input).copied())
                else {
                    return;
                };
                for k in Kind::ALL {
                    let Some(output) = k
                        .outputs()
                        .iter()
                        .position(|o| spec.accepts.contains(&o.ty))
                    else {
                        continue;
                    };
                    let info = k.info();
                    if ui.button(format!("{} {}", info.glyph, info.name)).clicked() {
                        let id = snarl.nodes().map(|n| n.id).max().unwrap_or(0) + 1;
                        let nid = snarl.insert_node(
                            pos,
                            EdNode {
                                id,
                                title: String::new(),
                                op: k.default_op(),
                            },
                        );
                        for p in pins {
                            if !spec.many {
                                snarl.drop_inputs(*p);
                            }
                            snarl.connect(OutPinId { node: nid, output }, *p);
                        }
                        self.clicked = Some(id);
                        ui.close();
                    }
                }
            }
        }
    }

    fn has_node_menu(&mut self, _node: &EdNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        let id = snarl[node].id;
        if ui.button("Show its parameters").clicked() {
            self.clicked = Some(id);
            ui.close();
        }
        if ui.button("Duplicate").clicked() {
            let copy = snarl[node].clone();
            let pos = snarl.get_node_info(node).map(|i| i.pos).unwrap_or_default();
            let new_id = snarl.nodes().map(|n| n.id).max().unwrap_or(0) + 1;
            snarl.insert_node(pos + egui::vec2(40.0, 40.0), EdNode { id: new_id, ..copy });
            self.clicked = Some(new_id);
            ui.close();
        }
        if ui.button("🗑 Remove").clicked() {
            snarl.remove_node(node);
            ui.close();
        }
    }

    fn has_on_hover_popup(&mut self, _node: &EdNode) -> bool {
        true
    }

    fn show_on_hover_popup(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<EdNode>,
    ) {
        let n = &snarl[node];
        ui.set_max_width(320.0);
        ui.label(n.op.kind().info().blurb);
        for p in self.findings.of(n.id) {
            ui.label(egui::RichText::new(format!("⚠ {p}")).color(ui.visuals().warn_fg_color));
        }
    }
}

// ---- the parameter forms ---------------------------------------------------

/// A combo box over a fixed list of choices.
fn choice<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    value: &mut T,
    all: &[T],
    label: impl Fn(T) -> &'static str,
) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(label(*value))
        .show_ui(ui, |ui| {
            for &v in all {
                ui.selectable_value(value, v, label(v));
            }
        });
}

fn text(ui: &mut egui::Ui, s: &mut String, hint: &str, width: f32) {
    ui.add(
        egui::TextEdit::singleline(s)
            .hint_text(hint)
            .desired_width(width),
    );
}

fn effort_ui(f: &mut form::Form, id: u32, e: &mut cat::Effort) {
    f.row_tip("Levels", "Resolution levels of the image pyramid", |ui| {
        ui.add(egui::DragValue::new(&mut e.levels).range(1..=6));
    });
    f.row_tip("Iterations", "Optimiser iterations per level", |ui| {
        ui.add(egui::DragValue::new(&mut e.iterations).range(1..=5000));
    });
    f.row_tip("Samples", "Random samples per iteration (elastix)", |ui| {
        ui.add(egui::DragValue::new(&mut e.samples).range(100..=200_000));
    });
    f.row_tip("Grid", "B-spline control point spacing", |ui| {
        ui.add(
            egui::DragValue::new(&mut e.grid_spacing_mm)
                .range(4.0..=200.0)
                .suffix(" mm"),
        );
    });
    f.row_tip(
        "Threshold",
        "Sample only fixed-image voxels above this value (a crude body mask)",
        |ui| {
            ui.add(egui::DragValue::new(&mut e.fixed_threshold).suffix(" HU"));
        },
    );
    let _ = id;
}

fn finish_ui(f: &mut form::Form, fin: &mut cat::FinishParams) {
    f.row_tip(
        "Close gaps",
        "Morphological closing of each landed structure; 0 leaves it as it landed",
        |ui| {
            ui.add(
                egui::DragValue::new(&mut fin.close_mm)
                    .range(0.0..=50.0)
                    .suffix(" mm"),
            );
        },
    );
    f.row("", |ui| {
        ui.checkbox(&mut fin.fill, "Fill the interior");
    });
    f.row("", |ui| {
        ui.checkbox(&mut fin.keep_shape, "Keep each structure's shape")
            .on_hover_text("Carry each structure as a rigid body: shape and volume kept");
    });
}

/// The form of one node's parameters.
fn params_ui(ui: &mut egui::Ui, id: u32, op: &mut Op, actions: &mut Vec<EdAction>) {
    form::form(ui, ("wf_params", id), |f| match op {
        Op::LoadFolder(p) => {
            f.row("Folder", |ui| {
                text(
                    ui,
                    &mut p.path,
                    "choose, or set it in the run dialog",
                    170.0,
                );
                if ui
                    .small_button("📂")
                    .on_hover_text("Choose the folder")
                    .clicked()
                {
                    actions.push(EdAction::BrowseFolder(id));
                }
            });
            f.row_tip(
                "Workspace",
                "Where the study is shown when a run shows its steps in the viewer",
                |ui| {
                    choice(
                        ui,
                        ("ws", id),
                        &mut p.workspace,
                        &cat::Workspace::ALL,
                        |w| w.label(),
                    )
                },
            );
            f.wide(|ui| {
                ui.weak(
                    "A relative folder is looked for beside the workflow file, then in the \
                     current folder, then beside the program.",
                );
            });
        }
        Op::SelectImage(p) => {
            f.row_tip("Modality", "CT, MR, PT; empty takes any", |ui| {
                text(ui, &mut p.modality, "any", 60.0);
            });
            f.row_tip(
                "Description has",
                "Words the series description must contain; case does not matter",
                |ui| text(ui, &mut p.description, "anything", 150.0),
            );
            f.row("Take", |ui| {
                choice(ui, ("pick", id), &mut p.pick, &cat::ImagePick::ALL, |v| {
                    v.label()
                });
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.outside_4d, "Not a phase of a 4D group");
            });
        }
        Op::SelectGroup(p) => {
            f.row_tip("Name has", "Words the 4D group's name must contain", |ui| {
                text(ui, &mut p.name, "the first group", 150.0);
            });
            f.row("If there is none", |ui| {
                choice(
                    ui,
                    ("none", id),
                    &mut p.if_none,
                    &[cat::IfNoGroup::GroupAll, cat::IfNoGroup::Fail],
                    |v| match v {
                        cat::IfNoGroup::GroupAll => "group the series",
                        cat::IfNoGroup::Fail => "stop the run",
                    },
                );
            });
            if p.if_none == cat::IfNoGroup::GroupAll {
                f.row("Modality", |ui| text(ui, &mut p.modality, "CT", 60.0));
            }
        }
        Op::SelectStructures(p) => {
            f.row_tip(
                "Names",
                "Names or patterns separated by commas; * stands for anything: target*, GTV*",
                |ui| text(ui, &mut p.names, "target*, GTV*", 180.0),
            );
            f.row("", |ui| {
                ui.checkbox(&mut p.first_only, "Only the first match");
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.required, "Stop the run when none is found");
            });
        }
        Op::AutoSegment(p) => {
            f.row("Model", |ui| {
                // TotalSegmentator's CT variants first (an empty `model`),
                // then every other model of the registry by its key.
                let shown = match p.auto_model() {
                    Some(m) if m.total_variant().is_some() => {
                        format!("TotalSegmentator CT, {}", p.variant.label())
                    }
                    Some(m) => m.label().to_string(),
                    None => format!("unknown model '{}'", p.model.trim()),
                };
                egui::ComboBox::from_id_salt(("model", id))
                    .selected_text(shown)
                    .height(420.0)
                    .show_ui(ui, |ui| {
                        ui.label(egui::RichText::new("TotalSegmentator CT").weak().small());
                        for v in cat::AutosegVariant::ALL {
                            let on = p.model.trim().is_empty() && p.variant == v;
                            if ui.selectable_label(on, v.label()).clicked() {
                                p.model.clear();
                                p.variant = v;
                            }
                        }
                        let mut heading = "";
                        for m in crate::zoo::AutoModel::all() {
                            if m.total_variant().is_some() {
                                continue;
                            }
                            if m.group() != heading {
                                heading = m.group();
                                ui.label(egui::RichText::new(heading).weak().small());
                            }
                            if ui
                                .selectable_label(p.model == m.key(), m.label())
                                .on_hover_text(m.detail())
                                .clicked()
                            {
                                p.model = m.key().to_string();
                            }
                        }
                    });
            });
            // The chosen model's own class names (v3 calls label 26
            // vertebrae_L6); either spelling, and the TG-263 name, is
            // accepted when typed.
            let class_names: &[&str] = p.auto_model().map_or(&[], |m| m.classes());
            let checker = p.clone();
            let p_label_known = |name: &str| checker.label_of(name).is_some();
            f.wide(|ui| {
                ui.strong("Organs to keep, and their names");
                let mut remove = None;
                for (k, o) in p.organs.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        let known = p_label_known(&o.organ);
                        egui::ComboBox::from_id_salt(("organ", id, k))
                            .selected_text(if o.organ.trim().is_empty() {
                                "organ"
                            } else {
                                o.organ.trim()
                            })
                            .width(120.0)
                            .show_ui(ui, |ui| {
                                for name in class_names.iter() {
                                    ui.selectable_value(&mut o.organ, name.to_string(), *name);
                                }
                            });
                        if !known {
                            ui.label(egui::RichText::new("✖").color(ui.visuals().error_fg_color))
                                .on_hover_text("not a class of the chosen model");
                        }
                        ui.label("as");
                        text(ui, &mut o.name, &o.organ.clone(), 100.0);
                        if ui
                            .small_button("🗑")
                            .on_hover_text("Do not keep it")
                            .clicked()
                        {
                            remove = Some(k);
                        }
                    });
                }
                if let Some(k) = remove {
                    p.organs.remove(k);
                }
                if ui.small_button("➕ Organ").clicked() {
                    p.organs.push(cat::OrganRule {
                        organ: "heart".into(),
                        name: String::new(),
                    });
                }
                if p.organs.is_empty() {
                    ui.weak("None listed: every organ found is kept.");
                }
            });
            output_rows(f, id, &mut p.output, &mut p.set, &mut p.set_label);
            names_row(f, id, &mut p.names);
            f.row("", |ui| {
                ui.checkbox(&mut p.tg263, "Name structures by AAPM TG-263")
                    .on_hover_text(
                        "Structures without a name of their own land under their TG-263 name \
                         (Kidney_L, Lung_RUL) where the class has one",
                    );
            });
            f.row("Compute", |ui| {
                choice(ui, ("dev", id), &mut p.device, &cat::Device::ALL, |v| {
                    v.label()
                });
            });
        }
        Op::BodyContour(p) => {
            f.row("Name", |ui| text(ui, &mut p.name, "BODY", 120.0));
            f.row("Method", |ui| {
                choice(ui, ("m", id), &mut p.method, &cat::BodyMethod::ALL, |v| {
                    v.label()
                });
            });
            output_rows(f, id, &mut p.output, &mut p.set, &mut p.set_label);
            names_row(f, id, &mut p.names);
            if p.method == cat::BodyMethod::ModelAssisted {
                f.row("Compute", |ui| {
                    choice(ui, ("dev", id), &mut p.device, &cat::Device::ALL, |v| {
                        v.label()
                    });
                });
            }
        }
        Op::Register(p) => {
            f.row("Method", |ui| {
                choice(
                    ui,
                    ("m", id),
                    &mut p.method,
                    &cat::RegMethodChoice::ALL,
                    |v| v.label(),
                );
            });
            f.row("Start", |ui| {
                choice(ui, ("init", id), &mut p.init, &cat::RegInit::ALL, |v| {
                    v.label()
                });
            });
            effort_ui(f, id, &mut p.effort);
        }
        Op::RegisterByStructures(p) => {
            f.row_tip(
                "Pairs",
                "The structures arriving on Fixed and on Moving pair by name. Write the \
                 ones named differently here: fixed name = moving name, separated by commas \
                 (Heart = heart_total).",
                |ui| text(ui, &mut p.pairs, "Heart = heart_total", 160.0),
            );
            f.row("Transform", |ui| {
                choice(
                    ui,
                    ("dof", id),
                    &mut p.shape.dof,
                    &cat::ShapeDofChoice::ALL,
                    |v| v.label(),
                );
            });
            f.row("Start", |ui| {
                choice(ui, ("init", id), &mut p.init, &cat::RegInit::ALL, |v| {
                    v.label()
                });
            });
            f.row_tip(
                "Robust",
                "Huber width: distances beyond it count linearly, so a slice contoured \
                 differently does not steer the fit; 0 is least squares",
                |ui| {
                    ui.add(
                        egui::DragValue::new(&mut p.shape.robust_mm)
                            .range(0.0..=20.0)
                            .suffix(" mm"),
                    );
                },
            );
            f.row("", |ui| {
                ui.checkbox(&mut p.shape.symmetric, "Both ways")
                    .on_hover_text("Also lay the moving surfaces onto the fixed structures");
            });
            f.row("Refine", |ui| {
                choice(
                    ui,
                    ("refine", id),
                    &mut p.shape.refine,
                    &cat::ShapeRefine::ALL,
                    |v| v.label(),
                );
            });
            if p.shape.refine != cat::ShapeRefine::None {
                f.row_tip(
                    "Margin",
                    "Each structure grown by this much bounds its refinement",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut p.shape.margin_mm)
                                .range(0.0..=60.0)
                                .suffix(" mm"),
                        );
                    },
                );
                effort_ui(f, id, &mut p.shape.effort);
            }
        }
        Op::Propagate(p) => {
            f.row("File into", |ui| {
                choice(ui, ("land", id), &mut p.landing, &cat::Landing::ALL, |v| {
                    v.label()
                });
            });
            f.row_tip(
                "Name suffix",
                "Added to each landed structure's name",
                |ui| {
                    text(ui, &mut p.suffix, "none", 120.0);
                },
            );
            names_row(f, id, &mut p.names);
            finish_ui(f, &mut p.finish);
        }
        Op::PropagateToGroup(p) => {
            f.row_tip(
                "Deformable",
                "The deformable registration (the local refinement of an anchored run)",
                |ui| {
                    choice(ui, ("m", id), &mut p.method, &cat::DeformMethod::ALL, |v| {
                        v.label()
                    })
                },
            );
            f.wide(|ui| {
                ui.weak("With an anchor connected:");
            });
            f.row_tip(
                "Anchor by",
                "What the anchored registration compares",
                |ui| {
                    choice(ui, ("by", id), &mut p.anchor_by, &cat::AnchorBy::ALL, |v| {
                        v.label()
                    });
                },
            );
            f.row_tip(
                "Anchor margin",
                "How far around the anchor the registration looks",
                |ui| {
                    ui.add(
                        egui::DragValue::new(&mut p.anchor_margin_mm)
                            .range(0.0..=100.0)
                            .suffix(" mm"),
                    );
                },
            );
            f.row("", |ui| {
                ui.checkbox(
                    &mut p.rigid_only,
                    "Rigid only (no local deformable refinement)",
                );
            });
            f.row_tip(
                "Anchor lands as",
                "The carried anchor lands beside each phase's own contour under this name",
                |ui| text(ui, &mut p.anchor_landed_as, "<anchor>_prop", 130.0),
            );
            f.row("File into", |ui| {
                choice(ui, ("land", id), &mut p.landing, &cat::Landing::ALL, |v| {
                    v.label()
                });
            });
            names_row(f, id, &mut p.names);
            finish_ui(f, &mut p.finish);
            effort_ui(f, id, &mut p.effort);
        }
        Op::Motion(p) => {
            f.row_tip(
                "Reference phase",
                "Its label, 0% say; empty is the group's own reference",
                |ui| text(ui, &mut p.reference_phase, "the group's", 80.0),
            );
            f.row_tip(
                "Phases",
                "Labels separated by commas; empty takes every phase",
                |ui| text(ui, &mut p.phases, "all", 140.0),
            );
            f.row("Models", |ui| {
                ui.vertical(|ui| {
                    ui.checkbox(&mut p.contoured, "As contoured on every phase");
                    ui.checkbox(&mut p.rigid, "Rigid");
                    ui.checkbox(&mut p.deformable, "Deformable");
                });
            });
            if p.rigid {
                f.row_tip(
                    "Rigid around",
                    "The rigid fit is made on each structure dilated by this margin; 0 is one \
                     global rigid body",
                    |ui| {
                        ui.add(
                            egui::DragValue::new(&mut p.local_rigid_margin_mm)
                                .range(0.0..=100.0)
                                .suffix(" mm"),
                        );
                    },
                );
            }
            f.row("", |ui| {
                ui.checkbox(&mut p.build_itv, "Build the ITV");
            });
            if p.build_itv {
                f.row("ITV margin", |ui| {
                    ui.add(
                        egui::DragValue::new(&mut p.itv_margin_mm)
                            .range(0.0..=50.0)
                            .suffix(" mm"),
                    );
                });
                f.row("ITV into", |ui| {
                    choice(
                        ui,
                        ("itv", id),
                        &mut p.itv_landing,
                        &cat::Landing::ALL,
                        |v| v.label(),
                    );
                });
                names_row(f, id, &mut p.names);
            }
            f.row("", |ui| {
                ui.checkbox(&mut p.keep_phase_segs, "Keep every phase's propagated copy");
            });
            effort_ui(f, id, &mut p.effort);
        }
        Op::ExportDicom(p) => {
            f.row_tip(
                "Folder",
                "Inside the run folder, or an absolute path; {input} is the title of the folder \
                 node the study came from",
                |ui| text(ui, &mut p.folder, "{input}", 150.0),
            );
            f.row("Structures as", |ui| {
                choice(
                    ui,
                    ("fmt", id),
                    &mut p.format,
                    &cat::StructFormatChoice::ALL,
                    |v| v.label(),
                );
            });
            f.row("Write", |ui| {
                choice(ui, ("sets", id), &mut p.sets, &cat::WhichSets::ALL, |v| {
                    v.label()
                });
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.images, "The images too");
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.doses_and_plans, "Doses and plans too");
            });
            f.row("Identifiers", |ui| {
                choice(ui, ("uid", id), &mut p.uids, &cat::UidChoice::ALL, |v| {
                    v.label()
                });
            });
        }
        Op::SaveReport(p) => {
            f.row_tip(
                "Folder",
                "Inside the run folder, or an absolute path",
                |ui| {
                    text(ui, &mut p.folder, "reports", 150.0);
                },
            );
            f.row("", |ui| {
                ui.checkbox(&mut p.csv, "CSV tables");
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.text, "A readable summary (Markdown)");
            });
        }
        Op::LoadFolders(p) => {
            f.row("Folder", |ui| {
                text(ui, &mut p.path, "the folder of the cases", 170.0);
                if ui
                    .small_button("📂")
                    .on_hover_text("Choose the folder")
                    .clicked()
                {
                    actions.push(EdAction::BrowseFolder(id));
                }
            });
            f.row_tip(
                "Subfolders",
                "Which subfolders are cases: * is all, P* those starting with P",
                |ui| text(ui, &mut p.pattern, "*", 90.0),
            );
            f.wide(|ui| {
                ui.weak(
                    "The whole workflow runs once per subfolder, in the background, each \
                     case in a folder of its own; the run ends with one table across the \
                     cases for every table its reports make.",
                );
            });
        }
        Op::LoadFromArchive(p) => {
            f.row_tip(
                "Patient",
                "The patient's ID or name, or a pattern with *",
                |ui| text(ui, &mut p.patient, "ID or name", 150.0),
            );
            f.row_tip(
                "Study",
                "Words of its description, or its date (YYYYMMDD); empty takes the newest",
                |ui| text(ui, &mut p.study, "the newest", 150.0),
            );
            f.row_tip(
                "Archive",
                "The archive's folder; empty is the station's (Tools > PACS)",
                |ui| text(ui, &mut p.archive, "the station's", 150.0),
            );
            f.row_tip(
                "Workspace",
                "Where the study is shown when a run shows its steps in the viewer",
                |ui| {
                    choice(
                        ui,
                        ("ws", id),
                        &mut p.workspace,
                        &cat::Workspace::ALL,
                        |w| w.label(),
                    )
                },
            );
        }
        Op::Anonymize(p) => {
            f.row_tip(
                "Folder",
                "Inside the run folder, or an absolute path; {input} is the study's title",
                |ui| text(ui, &mut p.folder, "anonymized/{input}", 150.0),
            );
            f.row("", |ui| {
                ui.checkbox(&mut p.remap_uids, "New UIDs, consistently");
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.remove_private, "Remove private tags");
            });
            f.row("", |ui| {
                ui.checkbox(
                    &mut p.clear_descriptions,
                    "Clear study and series descriptions",
                );
            });
        }
        Op::SegVolText(p) => {
            f.wide(|ui| {
                ui.strong("Prompts, and the names they are filed under");
                let mut remove = None;
                for (k, r) in p.prompts.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        text(ui, &mut r.structure, "liver", 100.0);
                        ui.label("as");
                        let hint = r.structure.clone();
                        text(ui, &mut r.name, &hint, 100.0);
                        if ui
                            .small_button("🗑")
                            .on_hover_text("Remove this prompt")
                            .clicked()
                        {
                            remove = Some(k);
                        }
                    });
                }
                if let Some(k) = remove {
                    p.prompts.remove(k);
                }
                if ui.small_button("➕ Prompt").clicked() {
                    p.prompts.push(cat::PromptRule::default());
                }
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.refine, "Refinement pass (slower, sharper)");
            });
            f.row("Threshold", |ui| {
                ui.add(egui::Slider::new(&mut p.threshold, 0.05..=0.95));
            });
            output_rows(f, id, &mut p.output, &mut p.set, &mut p.set_label);
            names_row(f, id, &mut p.names);
            f.row("Compute", |ui| {
                choice(ui, ("dev", id), &mut p.device, &cat::Device::ALL, |v| {
                    v.label()
                });
            });
        }
        Op::Combine(p) => {
            f.row("Result", |ui| text(ui, &mut p.name, "PTV", 120.0));
            f.row("Operation", |ui| {
                choice(ui, ("op", id), &mut p.op, &cat::CombineOp::ALL, |v| {
                    v.label()
                });
            });
            margin_row(f, "A first", &mut p.margin_a);
            margin_row(f, "B first", &mut p.margin_b);
            margin_row(f, "Result", &mut p.margin);
            f.row("", |ui| {
                ui.checkbox(&mut p.fill_holes, "Fill holes");
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.keep_largest, "Keep the largest piece only");
            });
            f.row_tip("Close gaps", "Morphological closing of the result", |ui| {
                ui.add(
                    egui::DragValue::new(&mut p.close_mm)
                        .range(0.0..=50.0)
                        .suffix(" mm"),
                );
            });
            output_rows(f, id, &mut p.output, &mut p.set, &mut p.set_label);
            names_row(f, id, &mut p.names);
        }
        Op::Rename(p) => {
            f.row("Action", |ui| {
                choice(
                    ui,
                    ("act", id),
                    &mut p.action,
                    &cat::RenameAction::ALL,
                    |v| v.label(),
                );
            });
            let rename = p.action == cat::RenameAction::Rename;
            f.wide(|ui| {
                ui.weak(
                    "Names or patterns with * (GTV*). Only the structures that arrive are \
                     touched, when any do.",
                );
                let mut remove = None;
                for (k, r) in p.rules.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        text(ui, &mut r.from, "GTV*", 100.0);
                        if rename {
                            ui.label("to");
                            text(ui, &mut r.to, "GTV", 100.0);
                        }
                        if ui
                            .small_button("🗑")
                            .on_hover_text("Remove this rule")
                            .clicked()
                        {
                            remove = Some(k);
                        }
                    });
                }
                if let Some(k) = remove {
                    p.rules.remove(k);
                }
                if ui.small_button("➕ Rule").clicked() {
                    p.rules.push(cat::RenameRule::default());
                }
            });
        }
        Op::Transfer(p) => {
            f.row_tip(
                "Lands as",
                "The name on the destination; empty keeps the target's",
                |ui| text(ui, &mut p.name, "the target's", 120.0),
            );
            output_rows(f, id, &mut p.output, &mut p.set, &mut p.set_label);
            names_row(f, id, &mut p.names);
        }
        Op::CopyToPhases(p) => {
            f.row("File into", |ui| {
                choice(ui, ("land", id), &mut p.landing, &cat::Landing::ALL, |v| {
                    v.label()
                });
            });
            names_row(f, id, &mut p.names);
        }
        Op::Dvh(p) => {
            f.row_tip(
                "Dose",
                "Words of the dose's label; empty takes the study's first dose",
                |ui| text(ui, &mut p.dose, "the first", 120.0),
            );
            f.row_tip(
                "Metrics",
                "Separated by commas: D95%, D2cc, V20Gy, V20Gy[cc], Dmean, Dmax",
                |ui| text(ui, &mut p.metrics, "D95%, Dmean", 180.0),
            );
            f.wide(|ui| {
                ui.label("Protocol, one constraint per line (Heart Dmean < 5)");
                ui.add(
                    egui::TextEdit::multiline(&mut p.protocol)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY),
                );
            });
            f.row_tip(
                "Protocol file",
                "A protocol saved from the DVH window, read when the run starts",
                |ui| text(ui, &mut p.protocol_file, "none", 160.0),
            );
            f.row("", |ui| {
                ui.checkbox(&mut p.curves, "The cumulative curves too");
            });
        }
        Op::DoseMetrics(p) => {
            f.row("Dose", |ui| {
                choice(
                    ui,
                    ("kind", id),
                    &mut p.dose_kind,
                    &cat::DoseKindChoice::ALL,
                    |v| v.label(),
                );
            });
            f.row_tip(
                "Label has",
                "Words of the dose's label; empty takes the first",
                |ui| text(ui, &mut p.dose, "any", 120.0),
            );
            f.row_tip("Columns", "Metrics separated by commas", |ui| {
                text(ui, &mut p.metrics, "Dmean, D95%", 180.0)
            });
        }
        Op::ArchiveImport(p) => {
            f.row("File", |ui| {
                choice(
                    ui,
                    ("src", id),
                    &mut p.source,
                    &cat::ImportSource::ALL,
                    |v| v.label(),
                );
            });
            f.row_tip(
                "Archive",
                "The archive's folder; empty is the station's (Tools > PACS)",
                |ui| text(ui, &mut p.archive, "the station's", 150.0),
            );
        }
        Op::Drr(p) => {
            f.row("", |ui| {
                ui.checkbox(&mut p.plan_beams, "At the plan's beams");
            });
            if !p.plan_beams {
                f.row_tip("Gantry", "Angles in degrees, separated by commas", |ui| {
                    text(ui, &mut p.angles, "0, 90", 120.0)
                });
                f.row("Couch", |ui| {
                    ui.add(egui::DragValue::new(&mut p.couch_deg).suffix(" deg"));
                });
            }
            f.row("Size", |ui| {
                ui.add(
                    egui::DragValue::new(&mut p.size_px)
                        .range(64..=2048)
                        .suffix(" px"),
                );
            });
            f.row("", |ui| {
                ui.checkbox(&mut p.invert, "Dark bone, as a radiograph");
            });
            f.row_tip(
                "Folder",
                "Inside the run folder, or an absolute path; {input} is the study's title",
                |ui| text(ui, &mut p.folder, "drr/{input}", 150.0),
            );
            f.row("", |ui| {
                ui.checkbox(&mut p.file_into_study, "Also as planar images of the study");
            });
        }
    });
}

/// What happens when a name a step files under is taken.
fn names_row(f: &mut form::Form, id: u32, names: &mut cat::NameClash) {
    f.row_tip(
        "Name taken",
        "When the set it goes into already holds a structure of that name. The same \
         name lands on every phase of a 4D group, and later steps look it up by it.",
        |ui| {
            choice(ui, ("clash", id), names, &cat::NameClash::ALL, |v| {
                v.label()
            })
        },
    );
}

/// A margin: one number for every direction.
fn margin_row(f: &mut form::Form, label: &str, m: &mut cat::MarginMm) {
    f.row_tip(
        label,
        "Grow (positive) or shrink (negative); a per-direction margin is set in the file",
        |ui| {
            ui.add(
                egui::DragValue::new(&mut m.uniform_mm)
                    .range(-50.0..=50.0)
                    .speed(0.5)
                    .suffix(" mm"),
            );
            if [
                m.right_mm,
                m.left_mm,
                m.anterior_mm,
                m.posterior_mm,
                m.superior_mm,
                m.inferior_mm,
            ]
            .iter()
            .any(Option::is_some)
            {
                ui.weak("and per direction");
            }
        },
    );
}

fn output_rows(
    f: &mut form::Form,
    id: u32,
    output: &mut cat::OutputKind,
    set: &mut cat::SetChoice,
    label: &mut String,
) {
    f.row("Output", |ui| {
        choice(ui, ("out", id), output, &cat::OutputKind::ALL, |v| {
            v.label()
        });
    });
    f.row_tip(
        "Into",
        "The image's own structure set is the one that references it - each phase's own on \
         a 4D group",
        |ui| choice(ui, ("set", id), set, &cat::SetChoice::ALL, |v| v.label()),
    );
    f.row_tip(
        "New set label",
        "The label of a structure set or segmentation series the step has to make",
        |ui| text(ui, label, "", 150.0),
    );
}

/// The unsaved-changes question: `Some(true)` to drop them, `Some(false)`
/// to keep editing, `None` while unanswered.
fn confirm_window(ctx: &egui::Context, question: &str) -> Option<bool> {
    let mut answer = None;
    egui::Window::new("Unsaved workflow")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.label(question);
            ui.horizontal(|ui| {
                if ui.button("Drop the changes").clicked() {
                    answer = Some(true);
                }
                if ui.button("Keep editing").clicked() {
                    answer = Some(false);
                }
            });
        });
    answer
}

// ---- the application's side ---------------------------------------------

impl ViewerApp {
    /// Replace the editor's workflow, asking first when it has unsaved
    /// changes.
    pub(super) fn wf_replace(&mut self, what: WfPending) {
        if self
            .wf_editor
            .as_ref()
            .is_some_and(|e| e.dirty() && e.snarl.nodes().next().is_some())
        {
            self.wf_confirm = Some(what);
            return;
        }
        self.wf_do(what);
    }

    fn wf_do(&mut self, what: WfPending) {
        match what {
            WfPending::New => {
                self.wf_editor = Some(WorkflowEditor::from_workflow(
                    Workflow::new("Untitled workflow"),
                    None,
                ));
            }
            WfPending::Open => {
                let dir = Some(store::user_dir());
                self.ask_file("Load a workflow", dir, Some(WORKFLOW_FILES), |app, path| {
                    app.wf_open_file(&path);
                });
            }
            WfPending::File(path) => self.wf_open_file(&path),
            WfPending::Example(i) => {
                if let Some(e) = store::EXAMPLES.get(i) {
                    let mut ed = WorkflowEditor::from_workflow(e.workflow(), None);
                    // An example is a start, not a file of the user's: the
                    // first save asks where.
                    ed.saved = String::new();
                    ed.status = Some(format!("Example: {}", e.blurb));
                    self.wf_editor = Some(ed);
                }
            }
        }
    }

    /// Open a workflow file in the editor.
    pub(super) fn wf_open_file(&mut self, path: &Path) {
        match store::load(path) {
            Ok(wf) => {
                let mut ed = WorkflowEditor::from_workflow(wf, Some(path.to_path_buf()));
                ed.status = Some(format!("Opened {}", path.display()));
                self.wf_editor = Some(ed);
            }
            Err(e) => {
                // A recent file that is gone leaves the list.
                if !path.exists() {
                    self.recent_workflows.retain(|p| p != path);
                    self.persist_settings();
                }
                self.error = Some(format!("{e:#}"));
            }
        }
    }

    /// Save the editor's workflow to `path`, and remember it.
    fn wf_save_to(&mut self, path: PathBuf) {
        let Some(ed) = &mut self.wf_editor else {
            return;
        };
        match store::save(&ed.to_workflow(), &path) {
            Ok(written) => {
                ed.mark_saved(written.clone());
                ed.status = Some(format!("Saved {}", written.display()));
                store::remember(&mut self.recent_workflows, &written);
                self.persist_settings();
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn wf_save_as(&mut self) {
        let Some(ed) = &self.wf_editor else {
            return;
        };
        let name = store::file_name_for(&ed.name);
        let dir = ed
            .path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf))
            .or_else(|| Some(store::user_dir()));
        self.ask_save(
            "Save the workflow",
            name,
            dir,
            Some(WORKFLOW_FILES),
            |app, path| {
                app.wf_save_to(path);
            },
        );
    }

    /// Draw the editor window, and do what it asked for.
    pub(super) fn workflow_editor_window(&mut self, ctx: &egui::Context) {
        // The question about unsaved changes is asked where the editor is,
        // or in the main window while the editor is closed.
        let question = self.wf_confirm.as_ref().map(|_| {
            format!(
                "'{}' has changes that are not saved. Drop them?",
                self.wf_editor
                    .as_ref()
                    .map(|e| e.name.trim().to_string())
                    .unwrap_or_default()
            )
        });
        let editor_open = self.wf_editor.as_ref().is_some_and(|e| e.open);
        let mut answer = None;
        if let (Some(q), false) = (&question, editor_open) {
            answer = confirm_window(ctx, q);
        }
        let Some(mut ed) = self.wf_editor.take() else {
            return;
        };
        let marks = self
            .wf_run
            .as_ref()
            .map(|r| r.marks.clone())
            .unwrap_or_default();
        let running = self.wf_run.as_ref().is_some_and(|r| r.is_running());
        let mut actions = Vec::new();
        let title = ed.title();
        let mut open = ed.open;
        detach::tool_window(
            ctx,
            "workflow_editor",
            title,
            &mut open,
            detach::WinOpts::size(1600.0, 920.0).no_scroll(),
            |ui| {
                ed.ui(ui, &marks, running, &mut actions);
                if let Some(q) = &question {
                    answer = confirm_window(ui.ctx(), q);
                }
            },
        );
        ed.open = open;
        self.wf_editor = Some(ed);
        if let Some(yes) = answer {
            if let Some(what) = self.wf_confirm.take() {
                if yes {
                    self.wf_do(what);
                } else if let Some(e) = &mut self.wf_editor {
                    // Keep editing: the editor, back in view.
                    e.open = true;
                }
            }
        }
        for a in actions {
            match a {
                EdAction::New => self.wf_replace(WfPending::New),
                EdAction::Open => self.wf_replace(WfPending::Open),
                EdAction::Save => {
                    let path = self.wf_editor.as_ref().and_then(|e| e.path.clone());
                    match path {
                        Some(p) => self.wf_save_to(p),
                        None => self.wf_save_as(),
                    }
                }
                EdAction::SaveAs => self.wf_save_as(),
                EdAction::Run => self.open_workflow_run(),
                EdAction::BrowseFolder(id) => {
                    self.ask_folder("The folder this step reads", move |app, path| {
                        if let Some(e) = &mut app.wf_editor {
                            e.set_folder(id, &path);
                        }
                    });
                }
                EdAction::BrowseOutput => {
                    self.ask_folder("Where runs write their results", |app, path| {
                        if let Some(e) = &mut app.wf_editor {
                            e.output.root = path.display().to_string();
                        }
                    });
                }
            }
        }
    }
}
