//! Running a workflow from the viewer: the dialog that says on what and how,
//! the run itself on a worker thread, and what it leaves behind.
//!
//! **On what.** Every folder node of the workflow is a line of the dialog,
//! filled with the folder the node names and changeable here - which is how
//! a saved workflow is applied to the next patient without being edited.
//! The results go to a folder of their own under the root shown (the
//! workflow's, or `<data folder>/workflow_runs`).
//!
//! **How.** *In the background*: the run reads, computes and writes, and
//! the workspaces are left alone; when it is done its studies can be put in
//! the workspaces with one click. *Show every step in the viewer*: each
//! study the run reads is put in a workspace (the one its folder node asks
//! for, or the next free one), and after every step the run pauses while
//! the viewer shows what the step did - the image it worked on displayed,
//! the structures it filed in the tree and on the image, each phase of a 4D
//! group in turn as it is segmented, the motion results window when the
//! motion is measured. The pause is a few seconds, or until *Next step* when
//! the run is set to wait. While such a run is going the workspaces it shows
//! are its own: each step replaces the study with the run's copy.
//!
//! **What the workspaces held.** A run that shows its steps takes the empty
//! workspaces first. When it has to take one holding a study of the user's,
//! that study is parked, as it was (views, cursor, visible structures), and
//! the finished run offers to put it back. Parked studies outlive a *Run
//! again*: they are the user's, the run's studies only a view of files the
//! run folder holds anyway.
//!
//! **Again, faster.** The window keeps what every step of the last run
//! left ([`StepCache`]); the next run takes over every step whose settings,
//! inputs and files are unchanged, and the canvas marks them ♻. *Forget*
//! empties it. Independent rows at the start can run side by side, and the
//! memory kept for image volumes between steps is set here.
//!
//! **Batches.** A workflow with a *DICOM folders* step runs once per
//! subfolder, in the background (showing every step of fifty patients is
//! not what anyone wants); the finished window lists the cases and the
//! combined tables.
//!
//! The run is [`exec::run`] on a [`Job`]; its events arrive on a channel
//! polled each frame ([`ViewerApp::poll_workflow_run`]), and a step shown is
//! acknowledged on a second channel when its pause is over.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};

use super::workflow_edit::Mark;
use super::*;
use crate::workflow::graph::catalog::Op;
use crate::workflow::graph::exec::{
    self, Channel, Event, Outcome, RunOptions, Shown, ShownStudy, Status, StepCache,
};
use crate::workflow::graph::{store, Workflow};

/// How a run goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RunMode {
    Background,
    Viewer,
}

/// The run window: its settings, and the run once it started.
pub(super) struct WorkflowRun {
    pub open: bool,
    pub workflow: Workflow,
    /// The workflow file's folder, where relative input folders are found.
    pub base_dir: Option<PathBuf>,
    /// (node id, title, folder) of every folder node.
    pub inputs: Vec<(u32, String, String)>,
    /// Where the run folder is made.
    pub out_root: String,
    pub mode: RunMode,
    /// Seconds each shown step stays before the run goes on.
    pub pause_s: f32,
    /// Wait for *Next step* after each shown step.
    pub step: bool,
    pub allow_download: bool,
    /// Run the independent rows at the start side by side (background runs
    /// and batches only).
    pub parallel: bool,
    /// Take over unchanged steps from the last run.
    pub reuse: bool,
    /// Megabytes of image volumes kept between steps.
    pub budget_mb: usize,
    /// What the last run's steps left, for the next.
    cache: StepCache,
    /// The user's studies a run took the workspace of, by workspace.
    parked: BTreeMap<usize, Parked>,

    pub marks: BTreeMap<u32, Mark>,
    lines: BTreeMap<u32, Vec<String>>,
    log: Vec<String>,
    /// (node, index, of) of the step running.
    current: Option<(u32, usize, usize)>,
    job: Option<Job<Outcome>>,
    events: Option<Receiver<Event>>,
    ack: Option<Sender<()>>,
    /// A step is on screen, waiting to be acknowledged: since when (egui
    /// time).
    waiting: Option<f64>,
    next: bool,
    /// Where each of the run's studies is shown: study index to workspace.
    ws: BTreeMap<usize, usize>,
    /// Workspaces this run has put a study in.
    installed: BTreeSet<usize>,
    outcome: Option<Outcome>,
    /// The run's final studies were put in the workspaces.
    shown_after: bool,
    started_mode: RunMode,
}

/// A study of the user's a run took the workspace of, as it was.
struct Parked {
    slot: StudySlot,
    /// The folders the workspace was loaded from (the session's list).
    sources: Vec<PathBuf>,
    shown: bool,
}

impl WorkflowRun {
    /// The window for `wf`. What the previous window was set to, what its
    /// steps left and the studies it parked carry over.
    fn new(wf: Workflow, base_dir: Option<PathBuf>, prev: Option<&mut WorkflowRun>) -> WorkflowRun {
        let inputs = wf
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                Op::LoadFolder(p) => Some((n.id, n.label(), p.path.clone())),
                Op::LoadFolders(p) => Some((n.id, n.label(), p.path.clone())),
                _ => None,
            })
            .collect();
        let out_root = if wf.output.root.trim().is_empty() {
            store::default_runs_dir().display().to_string()
        } else {
            wf.output.root.clone()
        };
        let mut run = WorkflowRun {
            open: true,
            workflow: wf,
            base_dir,
            inputs,
            out_root,
            mode: RunMode::Viewer,
            pause_s: 2.0,
            step: false,
            allow_download: true,
            parallel: false,
            reuse: true,
            budget_mb: 4096,
            cache: StepCache::default(),
            parked: BTreeMap::new(),
            marks: BTreeMap::new(),
            lines: BTreeMap::new(),
            log: Vec::new(),
            current: None,
            job: None,
            events: None,
            ack: None,
            waiting: None,
            next: false,
            ws: BTreeMap::new(),
            installed: BTreeSet::new(),
            outcome: None,
            shown_after: false,
            started_mode: RunMode::Background,
        };
        if let Some(p) = prev {
            run.carry_settings(p);
            // The steps' states are keyed by what made them, so another
            // workflow's simply never match; the parked studies are the
            // user's whatever runs next.
            run.cache = p.cache.clone();
            run.parked = std::mem::take(&mut p.parked);
        }
        run
    }

    /// Take over how the previous window was set.
    fn carry_settings(&mut self, p: &WorkflowRun) {
        self.mode = p.mode;
        self.pause_s = p.pause_s;
        self.step = p.step;
        self.allow_download = p.allow_download;
        self.parallel = p.parallel;
        self.reuse = p.reuse;
        self.budget_mb = p.budget_mb;
    }

    pub(super) fn is_running(&self) -> bool {
        self.job.is_some()
    }

    /// The user loads into `slot`: that replaces its study, the parked one
    /// included.
    pub(super) fn forget_parked(&mut self, slot: usize) {
        self.parked.remove(&slot);
    }

    /// Does this workflow run once per subfolder?
    fn is_batch(&self) -> bool {
        self.workflow
            .nodes
            .iter()
            .any(|n| matches!(n.op, Op::LoadFolders(_)))
    }

    /// The workflow as it will run: the dialog's folders in its folder
    /// nodes, the dialog's root as its output root.
    fn effective(&self) -> Workflow {
        let mut wf = self.workflow.clone();
        for (id, _, path) in &self.inputs {
            if let Some(n) = wf.node_mut(*id) {
                match &mut n.op {
                    Op::LoadFolder(p) => p.path = path.trim().to_string(),
                    Op::LoadFolders(p) => p.path = path.trim().to_string(),
                    _ => {}
                }
            }
        }
        wf.output.root = self.out_root.trim().to_string();
        wf
    }

    /// The cases a batch would run, or why it cannot say.
    fn batch_preview(&self) -> Option<Result<usize, String>> {
        let n = self
            .workflow
            .nodes
            .iter()
            .find(|n| matches!(n.op, Op::LoadFolders(_)))?;
        let Op::LoadFolders(p) = &n.op else {
            return None;
        };
        let path = self
            .inputs
            .iter()
            .find(|(id, _, _)| *id == n.id)
            .map(|(_, _, p)| p.trim().to_string())
            .unwrap_or_default();
        if path.is_empty() {
            return Some(Err("no folder chosen".into()));
        }
        let mut folder = PathBuf::from(&path);
        if folder.is_relative() {
            if let Some(b) = &self.base_dir {
                folder = b.join(folder);
            }
        }
        Some(
            exec::batch_cases(&folder, &p.pattern)
                .map(|c| c.len())
                .map_err(|e| format!("{e:#}")),
        )
    }
}

/// What the run window asks the application to do after drawing.
enum RunAction {
    Start,
    Cancel,
    BrowseInput(usize),
    BrowseRoot,
    ShowResults,
    OpenMotion(usize),
    Again,
    /// Put the parked studies back in their workspaces.
    PutBack,
    /// Empty the kept step states.
    Forget,
}

impl ViewerApp {
    /// Open the run window on the editor's workflow (or bring back the one
    /// running).
    pub(super) fn open_workflow_run(&mut self) {
        if let Some(r) = &mut self.wf_run {
            if r.is_running() {
                r.open = true;
                return;
            }
        }
        let Some(ed) = &self.wf_editor else {
            return;
        };
        let wf = ed.to_workflow();
        let base = ed
            .path
            .as_ref()
            .and_then(|p| p.parent().map(Path::to_path_buf));
        let mut prev = self.wf_run.take();
        self.wf_run = Some(WorkflowRun::new(wf, base, prev.as_mut()));
    }

    fn start_workflow_run(&mut self) {
        let models_root = models::root_from_setting(&self.models_dir);
        let Some(run) = &mut self.wf_run else {
            return;
        };
        let wf = run.effective();
        let f = wf.check();
        if !f.ok() {
            self.error = Some(format!(
                "The workflow cannot run yet:\n{}",
                f.errors
                    .iter()
                    .map(|(_, m)| m.as_str())
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
            return;
        }
        let mut opts = RunOptions::new(&wf, &store::default_runs_dir(), models_root);
        opts.base_dir = run.base_dir.clone();
        opts.allow_download = run.allow_download;
        opts.show_steps = run.mode == RunMode::Viewer && !run.is_batch();
        opts.parallel = run.parallel;
        opts.volume_budget_mb = run.budget_mb.max(256);
        opts.reuse = run.reuse.then(|| run.cache.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        let (channel, ack) = if opts.show_steps {
            let (ack_tx, ack_rx) = std::sync::mpsc::channel();
            (Channel::new(tx, Some(ack_rx)), Some(ack_tx))
        } else {
            (Channel::new(tx, None), None)
        };
        run.marks = wf.nodes.iter().map(|n| (n.id, Mark::Waiting)).collect();
        run.lines.clear();
        run.log = vec![format!("Results: {}", opts.run_dir.display())];
        run.current = None;
        run.events = Some(rx);
        run.ack = ack;
        run.waiting = None;
        run.next = false;
        run.ws.clear();
        run.installed.clear();
        run.outcome = None;
        run.shown_after = false;
        run.started_mode = if opts.show_steps {
            RunMode::Viewer
        } else {
            RunMode::Background
        };
        let progress = Arc::new(Progress::default());
        run.job = Some(Job::spawn(progress, move |p| {
            exec::run(&wf, &opts, p, &channel)
        }));
    }

    /// Drain the run's events, land what it shows, pace it, and take its
    /// outcome when it ends. Called every frame.
    pub(super) fn poll_workflow_run(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        let Some(run) = &mut self.wf_run else {
            return;
        };
        if run.job.is_none() {
            return;
        }
        let events: Vec<Event> = run
            .events
            .as_ref()
            .map(|rx| rx.try_iter().collect())
            .unwrap_or_default();
        let mut shows = Vec::new();
        for e in events {
            match e {
                Event::Started { node, index, of } => {
                    run.marks.insert(node, Mark::Running);
                    run.current = Some((node, index, of));
                }
                Event::Finished {
                    node,
                    lines,
                    secs,
                    reused,
                } => {
                    let mark = if reused {
                        Mark::Reused
                    } else {
                        Mark::Done(secs)
                    };
                    run.marks.insert(node, mark);
                    run.lines.insert(node, lines);
                }
                Event::Failed { node, error } => {
                    run.log.push(format!("✖ {error}"));
                    run.marks.insert(node, Mark::Failed(error));
                }
                Event::Log(l) => run.log.push(l),
                Event::Show(s) => shows.push(*s),
            }
        }
        if !shows.is_empty() {
            for s in shows {
                self.wf_show(s);
            }
            if let Some(run) = &mut self.wf_run {
                run.waiting = Some(now);
            }
        }
        let Some(run) = &mut self.wf_run else {
            return;
        };
        if let Some(since) = run.waiting {
            let due = run.next || (!run.step && now - since >= run.pause_s as f64);
            if due {
                if let Some(ack) = &run.ack {
                    let _ = ack.send(());
                }
                run.waiting = None;
                run.next = false;
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
        let mut err = None;
        if let Some(out) = poll_job(&mut run.job, ctx, "Workflow", &mut err) {
            for r in &out.records {
                match &r.status {
                    Status::NotRun => {
                        run.marks.insert(r.node, Mark::NotRun);
                    }
                    Status::Failed(e) => {
                        run.marks.insert(r.node, Mark::Failed(e.clone()));
                    }
                    // The last step's own event can still be on its way
                    // when the outcome arrives; the record says the same.
                    Status::Done => {
                        let mark = if r.reused {
                            Mark::Reused
                        } else {
                            Mark::Done(r.secs)
                        };
                        run.marks.insert(r.node, mark);
                        run.lines.insert(r.node, r.lines.clone());
                    }
                }
            }
            run.current = None;
            run.events = None;
            run.ack = None;
            run.waiting = None;
            run.log.push(match &out.error {
                None => format!("✔ Finished in {:.0} s", out.secs),
                Some(e) => format!("✖ {e}"),
            });
            if !run.open {
                self.notice = Some(match &out.error {
                    None => format!(
                        "The workflow '{}' finished. Its results are in {}",
                        run.workflow.name,
                        out.run_dir.display()
                    ),
                    Some(e) => format!("The workflow '{}' stopped: {e}", run.workflow.name),
                });
            }
            run.outcome = Some(out);
        }
        if let Some(e) = err {
            self.error = Some(e);
        }
    }

    /// The workspace a study of the run is shown in: the one it has, the
    /// one its folder node asks for, else the first empty one, else the
    /// first the run is not using. A study of the user's in the workspace
    /// taken is parked first.
    fn wf_slot(&mut self, ds: usize, wanted: Option<usize>) -> Option<usize> {
        let run = self.wf_run.as_mut()?;
        if let Some(&s) = run.ws.get(&ds) {
            return Some(s);
        }
        let used: BTreeSet<usize> = run.ws.values().copied().collect();
        let free = |s: &usize| !used.contains(s);
        let slot = wanted
            .filter(|w| *w < MAX_WORKSPACES && free(w))
            .or_else(|| (0..MAX_WORKSPACES).find(|s| free(s) && self.slots[*s].study.is_none()))
            .or_else(|| (0..MAX_WORKSPACES).find(free))?;
        run.ws.insert(ds, slot);
        self.wf_park(slot);
        Some(slot)
    }

    /// Keep the study in `slot` aside before a run's study takes its
    /// place, unless it is a run's study itself (one parked already means
    /// what is there now came from a run).
    fn wf_park(&mut self, slot: usize) {
        let Some(run) = &self.wf_run else {
            return;
        };
        if self.slots[slot].study.is_none() || run.parked.contains_key(&slot) {
            return;
        }
        self.wf_detach(slot);
        let parked = Parked {
            slot: std::mem::replace(&mut self.slots[slot], StudySlot::empty()),
            sources: self.session[slot].clone(),
            shown: self.shown[slot],
        };
        if let Some(run) = &mut self.wf_run {
            run.parked.insert(slot, parked);
        }
    }

    /// Let go of everything drawn from `slot` before its study is swapped:
    /// the played phases, the extra windows on it, a registration that
    /// names it, the cached overlays.
    fn wf_detach(&mut self, slot: usize) {
        self.drop_phase_cache(slot);
        self.planar_windows.retain(|w| w.slot != slot);
        self.d3_windows.retain(|w| w.slot != slot);
        self.tree_picks.retain(|(s, _)| *s != slot);
        if self.tree_focus.as_ref().is_some_and(|t| t.slot() == slot) {
            self.tree_focus = None;
        }
        if self.maximized.map(|(s, _)| s == slot).unwrap_or(false) {
            self.maximized = None;
        }
        if self
            .registration
            .as_ref()
            .is_some_and(|r| r.fixed_slot == slot || r.moving_slot == slot)
        {
            self.clear_registration();
        }
        self.struct_gen[slot] += 1;
        self.settings_gen += 1;
    }

    /// Put every parked study back where it was, replacing what the run
    /// left in those workspaces.
    fn wf_put_back(&mut self) {
        let parked = match &mut self.wf_run {
            Some(r) => std::mem::take(&mut r.parked),
            None => return,
        };
        for (slot, p) in parked {
            self.wf_detach(slot);
            self.slots[slot] = p.slot;
            for v in &mut self.slots[slot].views {
                v.invalidate();
            }
            self.session[slot] = p.sources;
            self.shown[slot] = p.shown || slot == 0;
            if let Some(r) = &mut self.wf_run {
                r.installed.remove(&slot);
                r.ws.retain(|_, s| *s != slot);
            }
            self.rebind_seg_series(slot);
        }
        self.persist_settings();
    }

    /// Put what a step shows on screen.
    fn wf_show(&mut self, s: Shown) {
        for st in s.studies {
            if let Some(slot) = self.wf_slot(st.ds, st.workspace) {
                self.wf_install(slot, st);
            }
        }
        if let Some(r) = s.report {
            if let Some(m) = r.motion {
                self.motion_reports.push(m);
                self.motion_sel = self.motion_reports.len() - 1;
                self.motion_results_open = true;
            }
        }
    }

    /// Put one study of the run in a workspace: the first time as a study
    /// just loaded, after that as the same study a step later, with the
    /// view where it was.
    fn wf_install(&mut self, slot: usize, st: ShownStudy) {
        let fresh = self
            .wf_run
            .as_mut()
            .map(|r| r.installed.insert(slot))
            .unwrap_or(true);
        let focus = st.focus.clone();
        if fresh || self.slots[slot].study.is_none() {
            self.on_study_loaded(slot, st.study);
        } else {
            let s = &mut self.slots[slot];
            let dims_before = s.study.as_ref().map(|x| x.volume.dims);
            s.study = Some(st.study);
            let study = s.study.as_ref().expect("just set");
            let dims = study.volume.dims;
            if dims_before != Some(dims) {
                for (c, d) in s.cursor.iter_mut().zip(dims) {
                    *c = c.min(d.saturating_sub(1) as f64);
                }
            }
            for v in &mut s.views {
                let n = study.volume.plane_slice_count(v.plane);
                v.slice = v.slice.min(n.saturating_sub(1));
                v.invalidate();
            }
            self.settings_gen += 1;
        }
        // The structure set and the segmentation series to show are the
        // ones the run filed into last: the last that reference the image.
        let s = &mut self.slots[slot];
        if let Some(study) = &s.study {
            let uid = study
                .series
                .get(study.active_series)
                .map(|x| x.uid.clone())
                .unwrap_or_default();
            if let Some(i) = study
                .structure_sets
                .iter()
                .rposition(|x| x.referenced_series_uid == uid)
            {
                s.active_structs = i;
            }
            s.roi_visible = vec![
                true;
                study
                    .structure_sets
                    .get(s.active_structs)
                    .map(|x| x.rois.len())
                    .unwrap_or(0)
            ];
            if let Some(i) = study
                .seg_series
                .iter()
                .rposition(|x| x.referenced_series_uid == uid)
            {
                s.active_seg_series = i;
            }
        }
        self.rebind_seg_series(slot);
        self.show_workspace(slot);
        if let Some(uid) = focus {
            let idx = self.slots[slot]
                .study
                .as_ref()
                .and_then(|x| x.series.iter().position(|se| se.uid == uid));
            if let Some(idx) = idx {
                self.start_series_switch(slot, idx);
            }
        }
    }

    /// Put a finished background run's studies in the workspaces.
    fn wf_show_results(&mut self) {
        let Some(run) = &mut self.wf_run else {
            return;
        };
        let Some(out) = &run.outcome else {
            return;
        };
        let studies = out.studies.clone();
        let motion: Vec<_> = out
            .reports
            .iter()
            .filter_map(|r| r.motion.clone())
            .collect();
        run.ws.clear();
        run.installed.clear();
        run.shown_after = true;
        for st in studies {
            if let Some(slot) = self.wf_slot(st.ds, st.workspace) {
                self.wf_install(slot, st);
            }
        }
        if !motion.is_empty() {
            self.motion_reports.extend(motion);
            self.motion_sel = self.motion_reports.len() - 1;
            self.motion_results_open = true;
        }
    }

    /// Draw the run window and do what it asked for.
    pub(super) fn workflow_run_window(&mut self, ctx: &egui::Context) {
        let Some(mut run) = self.wf_run.take() else {
            return;
        };
        let now = ctx.input(|i| i.time);
        let progress = run.job.as_ref().map(|j| j.progress.clone());
        let mut actions = Vec::new();
        let mut open = run.open;
        let title = format!("▶ Run: {}", run.workflow.name);
        detach::tool_window(
            ctx,
            "workflow_run",
            title,
            &mut open,
            detach::WinOpts::size(600.0, 680.0),
            |ui| run_ui(ui, &mut run, progress.as_deref(), now, &mut actions),
        );
        run.open = open;
        self.wf_run = Some(run);
        for a in actions {
            match a {
                RunAction::Start => self.start_workflow_run(),
                RunAction::Cancel => {
                    if let Some(r) = &mut self.wf_run {
                        if let Some(j) = &r.job {
                            j.progress.cancel();
                        }
                        r.next = true;
                    }
                }
                RunAction::BrowseInput(k) => {
                    self.ask_folder("The folder this input reads", move |app, path| {
                        if let Some(r) = &mut app.wf_run {
                            if let Some(i) = r.inputs.get_mut(k) {
                                i.2 = path.display().to_string();
                            }
                        }
                    });
                }
                RunAction::BrowseRoot => {
                    self.ask_folder("Where the run writes its results", |app, path| {
                        if let Some(r) = &mut app.wf_run {
                            r.out_root = path.display().to_string();
                        }
                    });
                }
                RunAction::ShowResults => self.wf_show_results(),
                RunAction::OpenMotion(i) => {
                    let m = self
                        .wf_run
                        .as_ref()
                        .and_then(|r| r.outcome.as_ref())
                        .and_then(|o| o.reports.get(i))
                        .and_then(|r| r.motion.clone());
                    if let Some(m) = m {
                        self.motion_reports.push(m);
                        self.motion_sel = self.motion_reports.len() - 1;
                        self.motion_results_open = true;
                    }
                }
                RunAction::Again => {
                    if let Some(r) = &mut self.wf_run {
                        r.outcome = None;
                        r.marks.clear();
                    }
                }
                RunAction::PutBack => self.wf_put_back(),
                RunAction::Forget => {
                    if let Some(r) = &self.wf_run {
                        r.cache.clear();
                    }
                }
            }
        }
    }
}

/// The run window's contents.
fn run_ui(
    ui: &mut egui::Ui,
    run: &mut WorkflowRun,
    progress: Option<&Progress>,
    now: f64,
    actions: &mut Vec<RunAction>,
) {
    ui.heading(&run.workflow.name);
    if !run.workflow.description.trim().is_empty() {
        ui.label(run.workflow.description.trim());
    }
    ui.separator();
    if let Some(p) = progress {
        running_ui(ui, run, p, now, actions);
    } else if run.outcome.is_some() {
        finished_ui(ui, run, actions);
    } else {
        setup_ui(ui, run, actions);
    }
}

fn setup_ui(ui: &mut egui::Ui, run: &mut WorkflowRun, actions: &mut Vec<RunAction>) {
    ui.strong("Inputs");
    if run.inputs.is_empty() {
        ui.weak("This workflow reads no folder.");
    }
    form::form(ui, "wf_run_inputs", |f| {
        for (k, (_, label, path)) in run.inputs.iter_mut().enumerate() {
            f.row(label.as_str(), |ui| {
                ui.add(
                    egui::TextEdit::singleline(path)
                        .hint_text("a DICOM folder")
                        .desired_width(330.0),
                );
                if ui
                    .small_button("📂")
                    .on_hover_text("Choose the folder")
                    .clicked()
                {
                    actions.push(RunAction::BrowseInput(k));
                }
            });
        }
        f.row_tip(
            "Results go to",
            "The run makes a folder of its own here, named after the workflow and the time",
            |ui| {
                ui.add(egui::TextEdit::singleline(&mut run.out_root).desired_width(330.0));
                if ui
                    .small_button("📂")
                    .on_hover_text("Choose the folder")
                    .clicked()
                {
                    actions.push(RunAction::BrowseRoot);
                }
            },
        );
    });
    let batch = run.is_batch();
    if let Some(cases) = run.batch_preview() {
        match cases {
            Ok(n) => ui.weak(format!(
                "A batch: the workflow runs once for each of the {n} matching subfolders, \
                 in the background, and the tables of every case are put together."
            )),
            Err(e) => ui.label(
                egui::RichText::new(format!("⚠ The cases cannot be listed: {e}"))
                    .color(ui.visuals().warn_fg_color),
            ),
        };
    }
    ui.add_space(6.0);
    ui.strong("How");
    ui.add_enabled_ui(!batch, |ui| {
        ui.radio_value(
            &mut run.mode,
            RunMode::Viewer,
            "Show every step in the viewer",
        )
        .on_hover_text(
            "Each study is put in a workspace and every step's result is shown there before \
             the next step starts: the image it worked on, the structures it filed, each \
             phase in turn, the motion results.",
        )
        .on_disabled_hover_text("A batch runs in the background");
    });
    ui.radio_value(&mut run.mode, RunMode::Background, "In the background")
        .on_hover_text(
            "The workspaces are left alone. When the run is done its studies can be put in \
             the workspaces with one click.",
        );
    if run.mode == RunMode::Viewer && !batch {
        ui.indent("wf_viewer_opts", |ui| {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!run.step, |ui| {
                    ui.label("Show each step for");
                    ui.add(
                        egui::DragValue::new(&mut run.pause_s)
                            .range(0.0..=30.0)
                            .speed(0.1)
                            .suffix(" s"),
                    );
                });
            });
            ui.checkbox(&mut run.step, "Wait for me after each step (Next step)");
            let letters: Vec<&str> = {
                let mut wanted: Vec<usize> = Vec::new();
                for n in &run.workflow.nodes {
                    if let Op::LoadFolder(p) = &n.op {
                        wanted.push(p.workspace.slot().unwrap_or(usize::MAX));
                    }
                }
                let mut used = BTreeSet::new();
                let mut out = Vec::new();
                for w in wanted {
                    let s = if w < MAX_WORKSPACES && !used.contains(&w) {
                        w
                    } else {
                        (0..MAX_WORKSPACES)
                            .find(|s| !used.contains(s))
                            .unwrap_or(MAX_WORKSPACES - 1)
                    };
                    used.insert(s);
                    out.push(SLOT_NAMES[s]);
                }
                out
            };
            if !letters.is_empty() {
                ui.weak(format!(
                    "The run's studies go into workspace{} {}, the empty ones first; a \
                     study of yours in one of them is kept aside and can be put back \
                     when the run is done.",
                    if letters.len() == 1 { "" } else { "s" },
                    letters.join(" and "),
                ));
            }
        });
    }
    ui.checkbox(
        &mut run.allow_download,
        "Download model weights that are missing",
    );
    let shows = run.mode == RunMode::Viewer && !batch;
    ui.add_enabled_ui(!shows, |ui| {
        ui.checkbox(&mut run.parallel, "Run independent rows side by side")
            .on_hover_text(
                "Rows of the canvas that share nothing at the start (two studies loaded and \
                 segmented apart, say) run at the same time. Faster on a machine with cores \
                 and memory to spare; the results are the same. When the last run's steps \
                 can be taken over, they are, and the rows do not run at all; a run whose \
                 rows ran side by side leaves nothing to take over.",
            )
            .on_disabled_hover_text("A run that shows its steps takes them one at a time");
    });
    ui.horizontal(|ui| {
        ui.checkbox(
            &mut run.reuse,
            "Take over unchanged steps from the last run",
        )
        .on_hover_text(
            "A step whose settings, inputs and files are as they were in the last run of \
                 this window, after steps that are too, is not run again: its result is taken \
                 over (marked ♻). Change a late step and only the steps from there on run.",
        );
        let kept = run.cache.len();
        if kept > 0
            && ui
                .small_button("Forget")
                .on_hover_text(format!(
                    "Let go of what the last run's {kept} steps left, and the memory it holds"
                ))
                .clicked()
        {
            actions.push(RunAction::Forget);
        }
    });
    ui.horizontal(|ui| {
        ui.label("Keep up to");
        ui.add(
            egui::DragValue::new(&mut run.budget_mb)
                .range(256..=65536)
                .speed(64.0)
                .suffix(" MB"),
        );
        ui.label("of image volumes between steps");
    })
    .response
    .on_hover_text(
        "Steps that read the same image again take it from memory while it fits here; \
         past that the least used are let go and read again when needed.",
    );
    ui.add_space(6.0);
    let wf = run.effective();
    let f = wf.check();
    for (_, m) in &f.errors {
        ui.label(egui::RichText::new(format!("✖ {m}")).color(ui.visuals().error_fg_color));
    }
    for (_, m) in &f.warnings {
        ui.label(egui::RichText::new(format!("⚠ {m}")).color(ui.visuals().warn_fg_color));
    }
    let missing: Vec<String> = run
        .inputs
        .iter()
        .filter(|(_, _, p)| !p.trim().is_empty())
        .filter(|(_, _, p)| {
            let path = Path::new(p.trim());
            path.is_absolute() && !path.is_dir()
        })
        .map(|(_, l, _)| l.clone())
        .collect();
    if !missing.is_empty() {
        ui.label(
            egui::RichText::new(format!("⚠ Not a folder here: {}", missing.join(", ")))
                .color(ui.visuals().warn_fg_color),
        );
    }
    ui.add_space(4.0);
    if ui
        .add_enabled(f.ok(), egui::Button::new("▶ Run"))
        .on_hover_text("Start the run")
        .clicked()
    {
        actions.push(RunAction::Start);
    }
}

fn marks_ui(ui: &mut egui::Ui, run: &WorkflowRun) {
    let order = run.workflow.order().unwrap_or_default();
    egui::Grid::new("wf_run_steps")
        .num_columns(3)
        .striped(true)
        .spacing([8.0, 3.0])
        .show(ui, |ui| {
            for (i, id) in order.iter().enumerate() {
                let Some(n) = run.workflow.node(*id) else {
                    continue;
                };
                let mark = run.marks.get(id).cloned().unwrap_or(Mark::Waiting);
                let (glyph, color) = match &mark {
                    Mark::Waiting => ("·", ui.visuals().weak_text_color()),
                    Mark::Running => ("⏳", ui.visuals().warn_fg_color),
                    Mark::Done(_) => ("✔", ui.visuals().text_color()),
                    Mark::Reused => ("♻", ui.visuals().text_color()),
                    Mark::Failed(_) => ("✖", ui.visuals().error_fg_color),
                    Mark::NotRun => ("-", ui.visuals().weak_text_color()),
                };
                ui.label(egui::RichText::new(glyph).color(color));
                ui.label(format!(
                    "{}. {} {}",
                    i + 1,
                    n.op.kind().info().glyph,
                    n.label()
                ));
                ui.vertical(|ui| {
                    match &mark {
                        Mark::Done(s) => {
                            ui.weak(format!("{s:.1} s"));
                        }
                        Mark::Failed(e) => {
                            ui.label(egui::RichText::new(e).color(ui.visuals().error_fg_color));
                        }
                        _ => {}
                    }
                    if let Some(lines) = run.lines.get(id) {
                        for l in lines.iter().take(4) {
                            ui.weak(l);
                        }
                    }
                });
                ui.end_row();
            }
        });
}

fn running_ui(
    ui: &mut egui::Ui,
    run: &mut WorkflowRun,
    p: &Progress,
    now: f64,
    actions: &mut Vec<RunAction>,
) {
    if let Some((node, index, of)) = run.current {
        let label = run
            .workflow
            .node(node)
            .map(|n| n.label())
            .unwrap_or_default();
        ui.strong(format!("Step {} of {of}: {label}", index + 1));
    } else {
        ui.strong("Starting");
    }
    ui.add(egui::ProgressBar::new(p.frac()).show_percentage());
    let msg = p.get();
    if !msg.is_empty() {
        ui.weak(msg);
    }
    ui.horizontal(|ui| {
        if let Some(since) = run.waiting {
            if run.step {
                if ui
                    .button("⏭ Next step")
                    .on_hover_text(
                        "The viewer shows what this step did; go on when you have looked",
                    )
                    .clicked()
                {
                    run.next = true;
                }
            } else {
                let left = (run.pause_s as f64 - (now - since)).max(0.0);
                ui.weak(format!("Showing the step, {left:.0} s"));
                if ui.small_button("⏭ Go on").clicked() {
                    run.next = true;
                }
            }
        }
        if ui.button("⏹ Cancel").clicked() {
            actions.push(RunAction::Cancel);
        }
    });
    ui.separator();
    marks_ui(ui, run);
    egui::CollapsingHeader::new("Log")
        .default_open(false)
        .show(ui, |ui| {
            for l in &run.log {
                ui.weak(l);
            }
        });
}

fn finished_ui(ui: &mut egui::Ui, run: &mut WorkflowRun, actions: &mut Vec<RunAction>) {
    let Some(out) = &run.outcome else {
        return;
    };
    match &out.error {
        None => {
            ui.strong(format!("✔ Every step ran, in {:.0} s.", out.secs));
        }
        Some(e) if out.cancelled => {
            ui.strong(format!("The run was cancelled after {:.0} s.", out.secs));
            let _ = e;
        }
        Some(e) => {
            ui.label(egui::RichText::new(format!("✖ {e}")).color(ui.visuals().error_fg_color));
        }
    }
    ui.horizontal_wrapped(|ui| {
        ui.label("Results:");
        ui.monospace(out.run_dir.display().to_string());
        if ui
            .small_button("📋")
            .on_hover_text("Copy the folder's path")
            .clicked()
        {
            ui.ctx().copy_text(out.run_dir.display().to_string());
        }
    });
    ui.horizontal(|ui| {
        if !out.studies.is_empty()
            && (run.started_mode == RunMode::Background || run.shown_after)
            && ui
                .button("Show the studies in the viewer")
                .on_hover_text(
                    "Put every study the run read, with what it filed, in the workspaces",
                )
                .clicked()
        {
            actions.push(RunAction::ShowResults);
        }
        if ui
            .button("↺ Run again")
            .on_hover_text("Back to the run's settings")
            .clicked()
        {
            actions.push(RunAction::Again);
        }
        if !run.parked.is_empty() {
            let letters: Vec<&str> = run.parked.keys().map(|s| SLOT_NAMES[*s]).collect();
            if ui
                .button(format!("Put back {}", letters.join(" and ")))
                .on_hover_text(
                    "Put the studies the run took these workspaces from back where they \
                     were, replacing what the run left there",
                )
                .clicked()
            {
                actions.push(RunAction::PutBack);
            }
        }
    });
    if !out.cases.is_empty() {
        ui.separator();
        ui.strong(format!("Cases ({})", out.cases.len()));
        egui::Grid::new("wf_run_cases")
            .num_columns(3)
            .striped(true)
            .spacing([8.0, 2.0])
            .show(ui, |ui| {
                for c in &out.cases {
                    match &c.error {
                        None => ui.label("✔"),
                        Some(_) => {
                            ui.label(egui::RichText::new("✖").color(ui.visuals().error_fg_color))
                        }
                    };
                    ui.label(&c.name)
                        .on_hover_text(c.run_dir.display().to_string());
                    match &c.error {
                        None => ui.weak(format!("{:.0} s", c.secs)),
                        Some(e) => {
                            ui.label(egui::RichText::new(e).color(ui.visuals().error_fg_color))
                        }
                    };
                    ui.end_row();
                }
            });
    }
    ui.separator();
    marks_ui(ui, run);
    if !out.reports.is_empty() {
        ui.separator();
        ui.strong("Reports");
        for (i, r) in out.reports.iter().enumerate() {
            egui::CollapsingHeader::new(&r.title)
                .id_salt(("wf_report", i))
                .default_open(false)
                .show(ui, |ui| {
                    for n in &r.notes {
                        ui.label(n);
                    }
                    if r.motion.is_some()
                        && ui
                            .button("📈 Open in Motion results")
                            .on_hover_text("The charts, tables and comparison of this motion run")
                            .clicked()
                    {
                        actions.push(RunAction::OpenMotion(i));
                    }
                    for (k, t) in r.tables.iter().enumerate() {
                        if !t.title.is_empty() {
                            ui.label(egui::RichText::new(&t.title).strong());
                        }
                        egui::ScrollArea::horizontal()
                            .id_salt(("wf_table_scroll", i, k))
                            .show(ui, |ui| {
                                egui::Grid::new(("wf_table", i, k))
                                    .striped(true)
                                    .spacing([10.0, 2.0])
                                    .show(ui, |ui| {
                                        for h in &t.header {
                                            run_report::head(ui, h);
                                        }
                                        ui.end_row();
                                        for row in &t.rows {
                                            for c in row {
                                                ui.label(c);
                                            }
                                            ui.end_row();
                                        }
                                    });
                            });
                        if ui
                            .small_button("📋 Copy")
                            .on_hover_text("Copy the table (CSV)")
                            .clicked()
                        {
                            ui.ctx().copy_text(t.csv());
                        }
                    }
                });
        }
    }
    egui::CollapsingHeader::new("Log")
        .default_open(false)
        .show(ui, |ui| {
            for l in &run.log {
                ui.weak(l);
            }
        });
}
