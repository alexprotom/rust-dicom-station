//! Running a workflow: every node in dependency order, on the calling
//! thread, with what each one made handed along its wires.
//!
//! **What flows.** A wire carries a [`Value`], and a value is a *reference*
//! into the run's own state, never a copy of the data: `Study(0)` is the
//! first folder read, `Image { ds, uid }` one of its series,
//! `Structures { names, on }` names looked up again wherever they are
//! needed. Steps that add structures add them to the study they belong to,
//! so a later step that names `heart total` finds the one an earlier step
//! filed - on the image it was filed on, or on every phase.
//!
//! **Reporting.** The run speaks through a [`Channel`]: a step started,
//! finished with a few lines to show, or failed. When the caller asked to
//! see every step ([`RunOptions::show_steps`]), each step that changed a
//! study also hands over a copy of it ([`Shown`]), set to display the image
//! the step worked on, and the run waits until the caller has shown it (the
//! channel's acknowledgement) - that is what lets the viewer walk through a
//! workflow the way a person would, one step at a time, with the data on
//! screen.
//!
//! **Failing.** The first step that fails ends the run; nothing after it
//! runs, and the [`Outcome`] carries what the steps before it made, so the
//! studies can still be looked at. A cancel is a failure the user asked for.
//!
//! **Files.** Everything a run writes goes under one run folder, made when
//! the run starts: the exports, the reports, and two files of its own - the
//! workflow as it was run (`workflow.rdsflow`, input folders included) and
//! `run-summary.md`, what every step did and how long it took.
//!
//! **Volumes.** What the steps read is kept in one cache for the whole run,
//! under a memory budget ([`RunOptions::volume_budget_mb`],
//! [`crate::workflow::session::Volumes`]), and handed to the 4D pipelines
//! too: the phases a step walked are in memory for the next step that walks
//! them.
//!
//! **Reruns.** Given a [`StepCache`] ([`RunOptions::reuse`]), a run keeps
//! the state after every step, filed under a fingerprint of that step and of
//! every step before it - parameters, wires, the input folders' contents.
//! The next run takes the state of every step whose fingerprint it finds
//! instead of running it, so changing one step reruns that step and what
//! comes after it, and nothing before. A step that writes files always runs
//! again: its files belong in the new run's folder.
//!
//! **Branches.** With [`RunOptions::parallel`], the independent rows at the
//! start of a workflow - everything before the first step that joins two
//! inputs - run at the same time, each on a thread of its own; the rest runs
//! in order as always. Not while the run shows its steps, and not when the
//! step cache holds the first step: taking the steps over is quicker than
//! running them side by side. A run whose rows ran side by side keeps no
//! step states (they are kept in the one run order), so the cache stays as
//! the last plain run left it.
//!
//! **Batches.** A workflow that starts from *DICOM folders* (one study per
//! subfolder) runs once per subfolder, each case in a folder of its own
//! under the run folder, and ends with `batch-summary.md` and one CSV per
//! report table with a `Case` column ([`run_batch`]).

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};

use crate::loader::LoadedStudy;
use crate::motion::MotionReport;
use crate::progress::{self, Progress, ProgressSink};
use crate::registration::{RegistrationResult, Transform3};
use crate::volume::Volume;
use crate::workflow::session::Volumes;

use super::catalog::{Kind, Op};
use super::{fill_template, nodes, safe_name, Workflow};

/// How a run is set up.
#[derive(Clone, Debug)]
pub struct RunOptions {
    /// Where every file of this run goes. Made when the run starts.
    pub run_dir: PathBuf,
    /// Input folders to read instead of the ones the folder nodes name, by
    /// node id: the same workflow on other data.
    pub inputs: BTreeMap<u32, PathBuf>,
    /// Where a relative input folder is looked for first (the folder of the
    /// workflow file); then the current folder, then the program's.
    pub base_dir: Option<PathBuf>,
    /// The root of the engines' model folders.
    pub models_dir: PathBuf,
    /// May an engine download weights it does not have?
    pub allow_download: bool,
    /// Hand every step's data over for display, and wait for it to be shown.
    pub show_steps: bool,
    /// Megabytes of image volumes kept between steps.
    pub volume_budget_mb: usize,
    /// Reuse what an earlier run of the same steps left, and keep what this
    /// one does for the next (see the module notes). `None` runs every step.
    pub reuse: Option<StepCache>,
    /// Run the independent rows at the start at the same time (see the
    /// module notes). Ignored while the run shows its steps.
    pub parallel: bool,
}

impl RunOptions {
    /// Options for a run of `wf` with its run folder made from the
    /// workflow's output settings, under `default_root` when the workflow
    /// names no root. The folder is not created here.
    pub fn new(wf: &Workflow, default_root: &Path, models_dir: PathBuf) -> RunOptions {
        RunOptions {
            run_dir: run_folder(wf, default_root),
            inputs: BTreeMap::new(),
            base_dir: None,
            models_dir,
            allow_download: true,
            show_steps: false,
            volume_budget_mb: 4096,
            reuse: None,
            parallel: false,
        }
    }
}

/// The folder a run of `wf` would write into now: the workflow's root (or
/// `default_root`) and its folder template, with a counter added when that
/// folder exists already.
pub fn run_folder(wf: &Workflow, default_root: &Path) -> PathBuf {
    let root = if wf.output.root.trim().is_empty() {
        default_root.to_path_buf()
    } else {
        PathBuf::from(wf.output.root.trim())
    };
    let (date, time) = crate::dicom_export::today();
    let template = if wf.output.folder.trim().is_empty() {
        "{workflow} {date}-{time}"
    } else {
        wf.output.folder.trim()
    };
    let name = fill_template(
        template,
        &[
            ("workflow", wf.name.as_str()),
            ("date", date.as_str()),
            ("time", time.as_str()),
        ],
    );
    let mut dir = root.join(safe_name(&name));
    let mut n = 1;
    while dir.exists() {
        n += 1;
        dir = root.join(format!("{} ({n})", safe_name(&name)));
    }
    dir
}

// ---- what flows along a wire ---------------------------------------------

/// Where named structures are looked up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Anywhere in the study.
    Study,
    /// On one image series (its own structure sets and segmentations).
    Image(String),
    /// On every phase of a 4D group.
    Group(usize),
}

/// A value on a wire: a reference into the run's state.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Study(usize),
    Image {
        ds: usize,
        uid: String,
    },
    Group {
        ds: usize,
        group: usize,
    },
    Structures {
        ds: usize,
        names: Vec<String>,
        on: Scope,
    },
    Registration(usize),
    Report(usize),
}

impl Value {
    /// The same value in a run whose studies, registrations and reports
    /// were numbered from `ds`, `reg` and `rep` on: a branch's value, once
    /// its state is appended to the main run's.
    fn shifted(&self, ds: usize, reg: usize, rep: usize) -> Value {
        match self {
            Value::Study(d) => Value::Study(d + ds),
            Value::Image { ds: d, uid } => Value::Image {
                ds: d + ds,
                uid: uid.clone(),
            },
            Value::Group { ds: d, group } => Value::Group {
                ds: d + ds,
                group: *group,
            },
            Value::Structures { ds: d, names, on } => Value::Structures {
                ds: d + ds,
                names: names.clone(),
                on: on.clone(),
            },
            Value::Registration(r) => Value::Registration(r + reg),
            Value::Report(r) => Value::Report(r + rep),
        }
    }

    /// The study the value belongs to, when it belongs to one.
    pub fn dataset(&self) -> Option<usize> {
        match self {
            Value::Study(ds) => Some(*ds),
            Value::Image { ds, .. } | Value::Group { ds, .. } | Value::Structures { ds, .. } => {
                Some(*ds)
            }
            Value::Registration(_) | Value::Report(_) => None,
        }
    }
}

// ---- the run's state -------------------------------------------------------

/// One study the run read.
///
/// Cloned only when a kept state ([`StepCache`]) still shares it and a step
/// changes it: the run holds its studies behind `Arc`s.
#[derive(Clone)]
pub struct Dataset {
    /// The folder node's title: what reports and export folders call it.
    pub label: String,
    pub node: u32,
    pub origin: PathBuf,
    /// Where it asked to be shown (`None`: the next free workspace).
    pub workspace: Option<usize>,
    pub study: LoadedStudy,
    /// Structure sets (by index) this run made or added to.
    pub touched_sets: BTreeSet<usize>,
    /// Segmentation series (by index) this run made or added to.
    pub touched_segs: BTreeSet<usize>,
    /// Folders an Export DICOM step of this run wrote the study into.
    pub exported: Vec<PathBuf>,
}

/// A registration the run made.
#[derive(Clone)]
pub enum RegEntry {
    Pair {
        fixed: (usize, String),
        moving: (usize, String),
        result: Arc<RegistrationResult>,
    },
    Group {
        ds: usize,
        group: usize,
        moving: (usize, String),
        /// (phase label, phase series UID, phase → moving).
        phases: Vec<(String, String, Arc<Transform3>)>,
    },
}

impl RegEntry {
    /// The entry in a run whose studies were numbered from `ds` on.
    fn shifted(&self, ds: usize) -> RegEntry {
        match self {
            RegEntry::Pair {
                fixed,
                moving,
                result,
            } => RegEntry::Pair {
                fixed: (fixed.0 + ds, fixed.1.clone()),
                moving: (moving.0 + ds, moving.1.clone()),
                result: result.clone(),
            },
            RegEntry::Group {
                ds: d,
                group,
                moving,
                phases,
            } => RegEntry::Group {
                ds: d + ds,
                group: *group,
                moving: (moving.0 + ds, moving.1.clone()),
                phases: phases.clone(),
            },
        }
    }
}

/// A table of a report.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    pub title: String,
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(title: impl Into<String>, header: &[&str]) -> Table {
        Table {
            title: title.into(),
            header: header.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    /// The table as CSV: comma-separated, quoted where a cell needs it.
    pub fn csv(&self) -> String {
        let line = |cells: &[String]| {
            cells
                .iter()
                .map(|c| csv_cell(c))
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut out = line(&self.header);
        out.push('\n');
        for r in &self.rows {
            out.push_str(&line(r));
            out.push('\n');
        }
        out
    }

    /// The table in Markdown.
    pub fn markdown(&self) -> String {
        let esc = |s: &String| s.replace('|', "/");
        let mut out = format!(
            "| {} |\n|{}|\n",
            self.header.iter().map(esc).collect::<Vec<_>>().join(" | "),
            self.header
                .iter()
                .map(|_| "---")
                .collect::<Vec<_>>()
                .join("|")
        );
        for r in &self.rows {
            out.push_str(&format!(
                "| {} |\n",
                r.iter().map(esc).collect::<Vec<_>>().join(" | ")
            ));
        }
        out
    }
}

fn csv_cell(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// What a step measured.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub node: u32,
    pub title: String,
    pub notes: Vec<String>,
    pub tables: Vec<Table>,
    /// The whole motion report, for the results window and its own CSV.
    pub motion: Option<MotionReport>,
}

impl Report {
    /// The report in Markdown: title, notes, tables.
    pub fn markdown(&self) -> String {
        let mut out = format!("## {}\n\n", self.title);
        for n in &self.notes {
            out.push_str(&format!("{n}\n\n"));
        }
        for t in &self.tables {
            if !t.title.is_empty() {
                out.push_str(&format!("**{}**\n\n", t.title));
            }
            out.push_str(&t.markdown());
            out.push('\n');
        }
        out
    }
}

// ---- talking to the caller -------------------------------------------------

/// A study as the viewer should show it after a step.
#[derive(Clone)]
pub struct ShownStudy {
    /// Which of the run's studies this is (the order they were read in).
    pub ds: usize,
    pub label: String,
    /// The workspace the folder node asked for; `None` is the next free one.
    pub workspace: Option<usize>,
    /// A copy of the study, displaying the image the step worked on when
    /// its volume was at hand.
    pub study: LoadedStudy,
    /// The series the step worked on, when the copy could not be set to
    /// display it (its volume was not loaded): the viewer reads it.
    pub focus: Option<String>,
}

/// One step's data, for display.
pub struct Shown {
    pub node: u32,
    pub studies: Vec<ShownStudy>,
    /// The report the step made, if any.
    pub report: Option<Report>,
}

/// What the run says while it runs.
pub enum Event {
    Started {
        node: u32,
        /// 0-based position in the run, and how many steps there are.
        index: usize,
        of: usize,
    },
    Finished {
        node: u32,
        lines: Vec<String>,
        secs: f64,
        /// Taken from the last run rather than run (see [`StepCache`]).
        reused: bool,
    },
    Failed {
        node: u32,
        error: String,
    },
    /// A step's data to show; the run waits for the acknowledgement.
    Show(Box<Shown>),
    /// A line for the run's log.
    Log(String),
}

/// The run's side of the conversation with whoever started it.
pub struct Channel {
    tx: Sender<Event>,
    /// Where "shown, go on" arrives after a [`Event::Show`]. Without one the
    /// run does not wait. Behind a lock so the branches of a parallel run
    /// can share the channel (they never show).
    ack: Mutex<Option<Receiver<()>>>,
}

impl Channel {
    pub fn new(tx: Sender<Event>, ack: Option<Receiver<()>>) -> Channel {
        Channel {
            tx,
            ack: Mutex::new(ack),
        }
    }

    pub fn send(&self, e: Event) {
        // The other side may have gone (a closed window); the run goes on.
        let _ = self.tx.send(e);
    }

    pub fn log(&self, line: impl Into<String>) {
        self.send(Event::Log(line.into()));
    }

    /// Hand a step's data over and wait until it has been shown, or the run
    /// is cancelled, or nobody is listening any more.
    pub fn show(&self, s: Shown, p: &Progress) -> Result<()> {
        self.send(Event::Show(Box::new(s)));
        let ack = self.ack.lock().unwrap_or_else(|e| e.into_inner());
        let Some(ack) = ack.as_ref() else {
            return Ok(());
        };
        loop {
            match ack.recv_timeout(Duration::from_millis(100)) {
                Ok(()) => return Ok(()),
                Err(RecvTimeoutError::Timeout) => {
                    if p.cancelled() {
                        bail!(progress::CANCELLED);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => return Ok(()),
            }
        }
    }
}

// ---- the outcome ------------------------------------------------------------

/// How one step ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Done,
    Failed(String),
    /// Not reached: an earlier step failed or the run was cancelled.
    NotRun,
}

/// One step of a finished run.
#[derive(Clone, Debug)]
pub struct NodeRecord {
    pub node: u32,
    pub label: String,
    pub status: Status,
    pub lines: Vec<String>,
    pub secs: f64,
    /// Its state was taken from the last run instead of running it again.
    pub reused: bool,
}

/// One case of a batch: a subfolder and how its run went.
#[derive(Clone, Debug)]
pub struct CaseRecord {
    pub name: String,
    pub folder: PathBuf,
    pub run_dir: PathBuf,
    /// `None` when every step ran.
    pub error: Option<String>,
    pub secs: f64,
}

/// Everything a run leaves behind.
pub struct Outcome {
    pub run_dir: PathBuf,
    pub records: Vec<NodeRecord>,
    /// Every study the run read, as it ends. Empty for a batch, whose cases
    /// are not kept in memory.
    pub studies: Vec<ShownStudy>,
    pub reports: Vec<Report>,
    /// Why the run stopped early; `None` when every step ran.
    pub error: Option<String>,
    pub cancelled: bool,
    pub secs: f64,
    /// A batch's cases, in order; empty for a single run.
    pub cases: Vec<CaseRecord>,
}

impl Outcome {
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }
}

/// What a node hands back to the runner.
pub(super) struct Done {
    /// One per output port, in order.
    pub outputs: Vec<Value>,
    /// What the canvas and the summary say about it.
    pub lines: Vec<String>,
    /// Studies it changed (or read), with the series to display.
    pub touched: Vec<(usize, Option<String>)>,
    /// The report it made, by index.
    pub report: Option<usize>,
}

/// The run's state, handed to every node.
pub(super) struct Ctx<'a> {
    pub opts: &'a RunOptions,
    pub channel: &'a Channel,
    pub workflow_name: String,
    pub date: String,
    pub time: String,
    /// Shared with the [`StepCache`] states that still refer to them;
    /// [`Ctx::ds_mut`] copies one before changing it if so.
    pub datasets: Vec<Arc<Dataset>>,
    pub regs: Vec<RegEntry>,
    pub reports: Vec<Report>,
    /// The volumes read so far, shared with the pipelines a step calls.
    pub volumes: Volumes,
}

impl Ctx<'_> {
    pub fn ds(&self, i: usize) -> Result<&Dataset> {
        self.datasets
            .get(i)
            .map(|d| &**d)
            .ok_or_else(|| anyhow!("internal: no study {i}"))
    }

    pub fn ds_mut(&mut self, i: usize) -> Result<&mut Dataset> {
        self.datasets
            .get_mut(i)
            .map(Arc::make_mut)
            .ok_or_else(|| anyhow!("internal: no study {i}"))
    }

    pub fn add_dataset(
        &mut self,
        label: String,
        node: u32,
        origin: PathBuf,
        workspace: Option<usize>,
        study: LoadedStudy,
    ) -> usize {
        self.datasets.push(Arc::new(Dataset {
            label,
            node,
            origin,
            workspace,
            study,
            touched_sets: BTreeSet::new(),
            touched_segs: BTreeSet::new(),
            exported: Vec::new(),
        }));
        self.datasets.len() - 1
    }

    /// A registration, filed; returns its index.
    pub fn add_reg(&mut self, r: RegEntry) -> usize {
        self.regs.push(r);
        self.regs.len() - 1
    }

    /// The volume of one series: the displayed one, or read through the
    /// run's cache.
    pub fn volume(&mut self, ds: usize, uid: &str, p: &Progress) -> Result<Arc<Volume>> {
        let d = self.ds(ds)?;
        if d.study.has_volume()
            && d.study
                .series
                .get(d.study.active_series)
                .map(|s| s.uid.as_str())
                == Some(uid)
        {
            return Ok(d.study.volume.clone());
        }
        let info = d
            .study
            .series
            .iter()
            .find(|s| s.uid == uid)
            .cloned()
            .ok_or_else(|| anyhow!("the series is no longer in '{}'", d.label))?;
        self.volumes.load(&info, p)
    }

    /// The volume of a series only when it is already in memory.
    fn volume_at_hand(&self, ds: usize, uid: &str) -> Option<Arc<Volume>> {
        let d = self.datasets.get(ds)?;
        if d.study.has_volume()
            && d.study
                .series
                .get(d.study.active_series)
                .map(|s| s.uid.as_str())
                == Some(uid)
        {
            return Some(d.study.volume.clone());
        }
        let info = d.study.series.iter().find(|s| s.uid == uid)?;
        self.volumes.get(info)
    }

    /// A report, filed; returns its index.
    pub fn add_report(&mut self, r: Report) -> usize {
        self.reports.push(r);
        self.reports.len() - 1
    }

    /// Where a node's folder parameter points: absolute as given, else
    /// under the run folder, with the template filled.
    pub fn out_path(&self, template: &str, input: &str) -> PathBuf {
        let filled = fill_template(
            template.trim(),
            &[
                ("workflow", self.workflow_name.as_str()),
                ("date", self.date.as_str()),
                ("time", self.time.as_str()),
                ("input", input),
            ],
        );
        let p = PathBuf::from(&filled);
        if p.is_absolute() {
            p
        } else {
            self.opts.run_dir.join(filled)
        }
    }

    /// A copy of a study for display, set to show `focus` when its volume
    /// is at hand (or can be read, since the caller wants to look at it).
    pub fn snapshot(&mut self, ds: usize, focus: Option<&str>, p: &Progress) -> Option<ShownStudy> {
        let mut vol = focus.and_then(|u| self.volume_at_hand(ds, u));
        if vol.is_none() {
            if let Some(u) = focus {
                vol = self.volume(ds, u, p).ok();
            }
        }
        let d = self.datasets.get(ds)?;
        let mut study = d.study.clone();
        let mut pending = None;
        if let Some(uid) = focus {
            match (study.series.iter().position(|s| s.uid == uid), vol) {
                (Some(idx), Some(v)) => {
                    study.active_series = idx;
                    study.volume = v;
                }
                (Some(_), None) => pending = Some(uid.to_string()),
                _ => {}
            }
        }
        Some(ShownStudy {
            ds,
            label: d.label.clone(),
            workspace: d.workspace,
            study,
            focus: pending,
        })
    }

    /// Show what a step touched, when the caller asked to see every step.
    pub fn show(
        &mut self,
        node: u32,
        touched: &[(usize, Option<String>)],
        report: Option<usize>,
        p: &Progress,
    ) -> Result<()> {
        if !self.opts.show_steps {
            return Ok(());
        }
        let mut studies = Vec::new();
        for (ds, focus) in touched {
            if let Some(s) = self.snapshot(*ds, focus.as_deref(), p) {
                studies.push(s);
            }
        }
        let report = report.and_then(|i| self.reports.get(i).cloned());
        if studies.is_empty() && report.is_none() {
            return Ok(());
        }
        self.channel.show(
            Shown {
                node,
                studies,
                report,
            },
            p,
        )
    }
}

/// The workflow with the run's input folders put into its folder nodes -
/// what the run folder's copy records.
pub fn with_inputs(wf: &Workflow, inputs: &BTreeMap<u32, PathBuf>) -> Workflow {
    let mut wf = wf.clone();
    for n in &mut wf.nodes {
        if let (Op::LoadFolder(p), Some(path)) = (&mut n.op, inputs.get(&n.id)) {
            p.path = path.display().to_string();
        }
    }
    wf
}

// ---- reruns -----------------------------------------------------------------

/// The run's state after one step, as a later run takes it over.
struct Kept {
    datasets: Vec<Arc<Dataset>>,
    regs: Vec<RegEntry>,
    reports: Vec<Report>,
    values: BTreeMap<(u32, usize), Value>,
    lines: Vec<String>,
}

/// What each step of the last run left, by fingerprint (see the module
/// notes). Cloning shares it; the viewer keeps one per open run window.
#[derive(Clone, Default)]
pub struct StepCache(Arc<Mutex<KeptSteps>>);

/// Each step's state, under its fingerprint, in the order they ran.
type KeptSteps = Vec<(u64, Arc<Kept>)>;

impl std::fmt::Debug for StepCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StepCache({} steps)", self.len())
    }
}

impl StepCache {
    fn find(&self, fp: u64) -> Option<Arc<Kept>> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .find(|(f, _)| *f == fp)
            .map(|(_, k)| k.clone())
    }

    fn replace(&self, kept: KeptSteps) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = kept;
    }

    /// Forget everything kept.
    pub fn clear(&self) {
        self.replace(Vec::new());
    }

    /// How many steps' states are kept.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// What makes a step's result what it is, apart from the steps before it:
/// its kind and parameters, its title (the reports carry it), its wires, and
/// for a step that reads from disk what is on the disk.
fn step_fingerprint(wf: &Workflow, node: &super::Node, opts: &RunOptions) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    node.id.hash(&mut h);
    node.title.hash(&mut h);
    serde_json::to_string(&node.op)
        .unwrap_or_default()
        .hash(&mut h);
    for l in wf.links.iter().filter(|l| l.to.0 == node.id) {
        l.hash(&mut h);
    }
    opts.models_dir.hash(&mut h);
    nodes::source_signature(opts, node).hash(&mut h);
    h.finish()
}

fn chain(prev: u64, step: u64) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    prev.hash(&mut h);
    step.hash(&mut h);
    h.finish()
}

// ---- running ----------------------------------------------------------------

/// Run `wf` on the calling thread. See the module documentation.
pub fn run(wf: &Workflow, opts: &RunOptions, p: &Progress, channel: &Channel) -> Outcome {
    if wf.nodes.iter().any(|n| n.op.kind() == Kind::LoadFolders) {
        return run_batch(wf, opts, p, channel);
    }
    run_one(wf, opts, p, channel)
}

/// The outcome of a run that could not start.
fn refused(opts: &RunOptions, error: String) -> Outcome {
    Outcome {
        run_dir: opts.run_dir.clone(),
        records: Vec::new(),
        studies: Vec::new(),
        reports: Vec::new(),
        error: Some(error),
        cancelled: false,
        secs: 0.0,
        cases: Vec::new(),
    }
}

/// How one node's run went, before it is written into the outcome.
enum Step {
    Ran(Done, f64),
    Failed(String, bool, f64),
}

/// Run one node on the values that arrived, reporting as it goes.
fn run_step(
    ctx: &mut Ctx,
    wf: &Workflow,
    id: u32,
    values: &BTreeMap<(u32, usize), Value>,
    index: usize,
    of: usize,
    p: &Progress,
) -> Step {
    let node = wf.node(id).expect("ordered ids exist");
    ctx.channel.send(Event::Started {
        node: id,
        index,
        of,
    });
    p.set_prefix("");
    p.set_outer(0.0, 1.0);
    p.report(0.0, &node.label());
    let inputs: Vec<Vec<Value>> = (0..node.op.inputs().len())
        .map(|k| {
            wf.links_into(id, k)
                .iter()
                .filter_map(|l| values.get(&l.from).cloned())
                .collect()
        })
        .collect();
    let t0 = Instant::now();
    let result = nodes::run_node(ctx, node, &inputs, p);
    p.set_prefix("");
    p.set_outer(0.0, 1.0);
    let secs = t0.elapsed().as_secs_f64();
    match result {
        Ok(done) => {
            ctx.channel.send(Event::Finished {
                node: id,
                lines: done.lines.clone(),
                secs,
                reused: false,
            });
            Step::Ran(done, secs)
        }
        Err(e) => {
            let cancelled = progress::is_cancellation(&e) || p.cancelled();
            let msg = if cancelled {
                "cancelled".to_string()
            } else {
                format!("{e:#}")
            };
            ctx.channel.send(Event::Failed {
                node: id,
                error: msg.clone(),
            });
            Step::Failed(msg, cancelled, secs)
        }
    }
}

fn run_one(wf: &Workflow, opts: &RunOptions, p: &Progress, channel: &Channel) -> Outcome {
    let t_start = Instant::now();
    let wf = with_inputs(wf, &opts.inputs);
    let (date, time) = crate::dicom_export::today();
    let mut ctx = Ctx {
        opts,
        channel,
        workflow_name: wf.name.clone(),
        date,
        time,
        datasets: Vec::new(),
        regs: Vec::new(),
        reports: Vec::new(),
        volumes: Volumes::with_budget_mb(opts.volume_budget_mb),
    };
    let findings = wf.check();
    if !findings.ok() {
        return refused(
            opts,
            findings
                .errors
                .iter()
                .map(|(_, m)| m.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }
    let order = match wf.order() {
        Ok(o) => o,
        Err(e) => return refused(opts, format!("{e:#}")),
    };
    if let Err(e) = std::fs::create_dir_all(&opts.run_dir) {
        return refused(
            opts,
            format!(
                "the run folder {} cannot be made: {e}",
                opts.run_dir.display()
            ),
        );
    }
    channel.log(format!("Run folder: {}", opts.run_dir.display()));
    let mut outcome = refused(opts, String::new());
    outcome.error = None;

    let mut values: BTreeMap<(u32, usize), Value> = BTreeMap::new();
    let n = order.len();
    // The rows before the first join, all at once - unless the last run's
    // states can be taken over from the first step on: that is quicker
    // still, and the fingerprints chain in the one run order.
    let first_kept = opts.reuse.as_ref().is_some_and(|cache| {
        order
            .first()
            .and_then(|id| wf.node(*id))
            .is_some_and(|node| {
                !node.op.kind().info().writes
                    && cache
                        .find(chain(0, step_fingerprint(&wf, node, opts)))
                        .is_some()
            })
    });
    let mut start = 0;
    if opts.parallel && !opts.show_steps && !first_kept {
        match run_branches(&wf, &order, &mut ctx, &mut values, &mut outcome, p) {
            Ok(done) => start = done,
            Err(()) => start = n,
        }
    }
    // Reuse only in a plain run, from the start: the fingerprints chain in
    // the order the steps run.
    let reuse = opts.reuse.as_ref().filter(|_| start == 0 && outcome.ok());
    let mut kept: KeptSteps = Vec::new();
    let mut fp = 0u64;
    let mut matching = true;
    for (index, &id) in order.iter().enumerate().skip(start) {
        if !outcome.ok() {
            break;
        }
        let node = wf.node(id).expect("ordered ids exist");
        if p.cancelled() {
            outcome.cancelled = true;
            outcome.error = Some("the run was cancelled".into());
            break;
        }
        fp = chain(fp, step_fingerprint(&wf, node, opts));
        let writes = node.op.kind().info().writes;
        if let (Some(cache), true, false) = (reuse, matching, writes) {
            if let Some(k) = cache.find(fp) {
                ctx.datasets = k.datasets.clone();
                ctx.regs = k.regs.clone();
                ctx.reports = k.reports.clone();
                values = k.values.clone();
                let mut lines = vec!["unchanged: taken from the last run".to_string()];
                lines.extend(k.lines.iter().cloned());
                channel.send(Event::Finished {
                    node: id,
                    lines: lines.clone(),
                    secs: 0.0,
                    reused: true,
                });
                outcome.records.push(NodeRecord {
                    node: id,
                    label: node.label(),
                    status: Status::Done,
                    lines,
                    secs: 0.0,
                    reused: true,
                });
                kept.push((fp, k));
                continue;
            }
        }
        if !writes {
            matching = false;
        }
        match run_step(&mut ctx, &wf, id, &values, index, n, p) {
            Step::Ran(done, secs) => {
                for (k, v) in done.outputs.iter().enumerate() {
                    values.insert((id, k), v.clone());
                }
                outcome.records.push(NodeRecord {
                    node: id,
                    label: node.label(),
                    status: Status::Done,
                    lines: done.lines.clone(),
                    secs,
                    reused: false,
                });
                if reuse.is_some() {
                    kept.push((
                        fp,
                        Arc::new(Kept {
                            datasets: ctx.datasets.clone(),
                            regs: ctx.regs.clone(),
                            reports: ctx.reports.clone(),
                            values: values.clone(),
                            lines: done.lines.clone(),
                        }),
                    ));
                }
                // The step is done; now it is shown, and the run waits for
                // the viewer before the next one starts.
                if let Err(e) = ctx.show(id, &done.touched, done.report, p) {
                    outcome.cancelled = progress::is_cancellation(&e) || p.cancelled();
                    outcome.error = Some(if outcome.cancelled {
                        "the run was cancelled".into()
                    } else {
                        format!("showing '{}' failed: {e:#}", node.label())
                    });
                    break;
                }
            }
            Step::Failed(msg, cancelled, secs) => {
                outcome.records.push(NodeRecord {
                    node: id,
                    label: node.label(),
                    status: Status::Failed(msg.clone()),
                    lines: Vec::new(),
                    secs,
                    reused: false,
                });
                outcome.cancelled = cancelled;
                outcome.error = Some(if cancelled {
                    "the run was cancelled".into()
                } else {
                    format!("'{}' failed: {msg}", node.label())
                });
                break;
            }
        }
    }
    // What this run did is what the next run can reuse - the steps before a
    // failure included, so fixing the step that failed reruns only it.
    if let Some(cache) = reuse {
        cache.replace(kept);
    }
    finish(&wf, ctx, outcome, &order, t_start)
}

/// The records of the steps not reached, the run's own two files, and the
/// studies and reports as the run ends.
fn finish(
    wf: &Workflow,
    ctx: Ctx,
    mut outcome: Outcome,
    order: &[u32],
    t_start: Instant,
) -> Outcome {
    let opts = ctx.opts;
    let mut records = Vec::with_capacity(order.len());
    for &id in order {
        match outcome.records.iter().position(|r| r.node == id) {
            Some(i) => records.push(outcome.records[i].clone()),
            None => {
                let node = wf.node(id).expect("ordered ids exist");
                records.push(NodeRecord {
                    node: id,
                    label: node.label(),
                    status: Status::NotRun,
                    lines: Vec::new(),
                    secs: 0.0,
                    reused: false,
                });
            }
        }
    }
    outcome.records = records;
    outcome.secs = t_start.elapsed().as_secs_f64();

    // The run's own two files.
    let _ = std::fs::write(opts.run_dir.join("workflow.rdsflow"), wf.to_json());
    let _ = std::fs::write(
        opts.run_dir.join("run-summary.md"),
        summary_markdown(wf, &ctx, &outcome),
    );

    for (ds, d) in ctx.datasets.iter().enumerate() {
        outcome.studies.push(ShownStudy {
            ds,
            label: d.label.clone(),
            workspace: d.workspace,
            study: d.study.clone(),
            focus: None,
        });
    }
    outcome.reports = ctx.reports;
    outcome
}

// ---- branches -----------------------------------------------------------------

/// Run the independent rows at the start of the workflow side by side:
/// every step before the first one whose inputs reach back to more than one
/// source (a folder read), grouped by the source they reach back to. Their
/// states are appended to `ctx` in the order of the sources, their values
/// renumbered to match. Returns how many steps of `order` were run, or
/// `Err` when one failed (the outcome says which).
fn run_branches(
    wf: &Workflow,
    order: &[u32],
    ctx: &mut Ctx,
    values: &mut BTreeMap<(u32, usize), Value>,
    outcome: &mut Outcome,
    p: &Progress,
) -> std::result::Result<usize, ()> {
    // The sources each step reaches back to.
    let mut roots: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
    for &id in order {
        let node = wf.node(id).expect("ordered ids exist");
        let mut r = BTreeSet::new();
        if node.op.inputs().is_empty() {
            r.insert(id);
        }
        for l in wf.links.iter().filter(|l| l.to.0 == id) {
            if let Some(up) = roots.get(&l.from.0) {
                r.extend(up.iter().copied());
            }
        }
        roots.insert(id, r);
    }
    let prefix = order
        .iter()
        .position(|id| roots[id].len() != 1)
        .unwrap_or(order.len());
    let mut branches: BTreeMap<u32, Vec<(usize, u32)>> = BTreeMap::new();
    for (i, &id) in order[..prefix].iter().enumerate() {
        let root = *roots[&id].iter().next().expect("one root");
        branches.entry(root).or_default().push((i, id));
    }
    if branches.len() < 2 {
        return Ok(0);
    }
    ctx.channel.log(format!(
        "Running {} independent rows at the same time",
        branches.len()
    ));
    let n = order.len();
    let opts = ctx.opts;
    let channel = ctx.channel;
    type BranchResult = (
        Vec<Arc<Dataset>>,
        Vec<RegEntry>,
        Vec<Report>,
        BTreeMap<(u32, usize), Value>,
        Vec<NodeRecord>,
        Option<(String, bool)>,
    );
    let progresses: Vec<Arc<Progress>> = branches
        .keys()
        .map(|_| Arc::new(Progress::default()))
        .collect();
    let results: Vec<BranchResult> = std::thread::scope(|scope| {
        let handles: Vec<_> = branches
            .values()
            .zip(&progresses)
            .map(|(steps, bp)| {
                let bp = bp.clone();
                let (name, date, time, volumes) = (
                    ctx.workflow_name.clone(),
                    ctx.date.clone(),
                    ctx.time.clone(),
                    ctx.volumes.clone(),
                );
                scope.spawn(move || {
                    let mut bctx = Ctx {
                        opts,
                        channel,
                        workflow_name: name,
                        date,
                        time,
                        datasets: Vec::new(),
                        regs: Vec::new(),
                        reports: Vec::new(),
                        volumes,
                    };
                    let mut bvalues = BTreeMap::new();
                    let mut records = Vec::new();
                    let mut failed = None;
                    for &(index, id) in steps {
                        if bp.cancelled() {
                            failed = Some(("the run was cancelled".to_string(), true));
                            break;
                        }
                        let node = wf.node(id).expect("ordered ids exist");
                        match run_step(&mut bctx, wf, id, &bvalues, index, n, &bp) {
                            Step::Ran(done, secs) => {
                                for (k, v) in done.outputs.into_iter().enumerate() {
                                    bvalues.insert((id, k), v);
                                }
                                records.push(NodeRecord {
                                    node: id,
                                    label: node.label(),
                                    status: Status::Done,
                                    lines: done.lines,
                                    secs,
                                    reused: false,
                                });
                            }
                            Step::Failed(msg, cancelled, secs) => {
                                records.push(NodeRecord {
                                    node: id,
                                    label: node.label(),
                                    status: Status::Failed(msg.clone()),
                                    lines: Vec::new(),
                                    secs,
                                    reused: false,
                                });
                                failed = Some((
                                    if cancelled {
                                        "the run was cancelled".to_string()
                                    } else {
                                        format!("'{}' failed: {msg}", node.label())
                                    },
                                    cancelled,
                                ));
                                break;
                            }
                        }
                    }
                    (
                        bctx.datasets,
                        bctx.regs,
                        bctx.reports,
                        bvalues,
                        records,
                        failed,
                    )
                })
            })
            .collect();
        // One bar over the branches: the mean of theirs, their messages
        // side by side; a cancel reaches every branch, and so does the
        // failure of one.
        loop {
            let running = handles.iter().filter(|h| !h.is_finished()).count();
            let frac = progresses.iter().map(|b| b.frac()).sum::<f32>() / progresses.len() as f32;
            let msg = progresses
                .iter()
                .map(|b| b.get())
                .filter(|m| !m.is_empty())
                .collect::<Vec<_>>()
                .join(" | ");
            p.report(frac, &msg);
            if running == 0 {
                break;
            }
            if p.cancelled() {
                progresses.iter().for_each(|b| b.cancel());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("a branch does not panic"))
            .collect()
    });
    let mut failed = None;
    for (datasets, regs, reports, bvalues, records, fail) in results {
        let (ds0, reg0, rep0) = (ctx.datasets.len(), ctx.regs.len(), ctx.reports.len());
        ctx.datasets.extend(datasets);
        ctx.regs.extend(regs.iter().map(|r| r.shifted(ds0)));
        ctx.reports.extend(reports);
        for (k, v) in bvalues {
            values.insert(k, v.shifted(ds0, reg0, rep0));
        }
        outcome.records.extend(records);
        if failed.is_none() {
            failed = fail;
        }
    }
    match failed {
        None => Ok(prefix),
        Some((msg, cancelled)) => {
            outcome.cancelled = cancelled;
            outcome.error = Some(msg);
            Err(())
        }
    }
}

// ---- batches --------------------------------------------------------------------

/// The subfolders a *DICOM folders* node runs a batch over: every folder
/// directly under its root whose name matches its pattern, sorted by name.
pub fn batch_cases(folder: &Path, pattern: &str) -> Result<Vec<PathBuf>> {
    let rd = std::fs::read_dir(folder)
        .map_err(|e| anyhow!("the folder {} cannot be read: {e}", folder.display()))?;
    let pat = if pattern.trim().is_empty() {
        "*"
    } else {
        pattern.trim()
    };
    let mut out: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| super::catalog::name_matches(pat, &n.to_string_lossy()))
        })
        .collect();
    out.sort();
    Ok(out)
}

/// Run a workflow that starts from *DICOM folders* once per subfolder. See
/// the module notes.
pub fn run_batch(wf: &Workflow, opts: &RunOptions, p: &Progress, channel: &Channel) -> Outcome {
    let t_start = Instant::now();
    let Some(src) = wf
        .nodes
        .iter()
        .find(|n| n.op.kind() == Kind::LoadFolders)
        .cloned()
    else {
        return refused(opts, "no DICOM folders step".into());
    };
    if wf
        .nodes
        .iter()
        .filter(|n| n.op.kind() == Kind::LoadFolders)
        .count()
        > 1
    {
        return refused(
            opts,
            "a batch runs over one DICOM folders step; this workflow has several".into(),
        );
    }
    let Op::LoadFolders(prm) = &src.op else {
        unreachable!("found by kind")
    };
    let root = match nodes::input_folder(opts, &src, &prm.path) {
        Ok(r) => r,
        Err(e) => return refused(opts, format!("{e:#}")),
    };
    let cases = match batch_cases(&root, &prm.pattern) {
        Ok(c) if !c.is_empty() => c,
        Ok(_) => {
            return refused(
                opts,
                format!(
                    "{} has no subfolder matching '{}'",
                    root.display(),
                    prm.pattern
                ),
            )
        }
        Err(e) => return refused(opts, format!("{e:#}")),
    };
    if let Err(e) = std::fs::create_dir_all(&opts.run_dir) {
        return refused(
            opts,
            format!(
                "the run folder {} cannot be made: {e}",
                opts.run_dir.display()
            ),
        );
    }
    let mut outcome = refused(opts, String::new());
    outcome.error = None;
    // Every table of every case, by title, with the case in front.
    let mut combined: BTreeMap<String, Table> = BTreeMap::new();
    let n = cases.len();
    for (i, folder) in cases.iter().enumerate() {
        if p.cancelled() {
            outcome.cancelled = true;
            outcome.error = Some("the run was cancelled".into());
            break;
        }
        let name = folder
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("case {}", i + 1));
        channel.log(format!("Case {}/{n}: {name}", i + 1));
        // The case as a plain run: the folders step becomes a folder step
        // on this subfolder, titled after it so exports name the case.
        let mut case_wf = wf.clone();
        if let Some(node) = case_wf.node_mut(src.id) {
            node.op = Op::LoadFolder(super::catalog::LoadFolder {
                path: folder.display().to_string(),
                workspace: prm.workspace,
            });
            node.title = name.clone();
        }
        let mut case_opts = opts.clone();
        case_opts.run_dir = opts.run_dir.join(safe_name(&name));
        case_opts.inputs.remove(&src.id);
        case_opts.show_steps = false;
        case_opts.reuse = None;
        p.set_outer(i as f32 / n as f32, 1.0 / n as f32);
        let case_out = run_one(&case_wf, &case_opts, p, channel);
        p.set_outer(0.0, 1.0);
        for r in &case_out.reports {
            for t in &r.tables {
                let key = format!("{} - {}", r.title, t.title);
                let entry = combined.entry(key.clone()).or_insert_with(|| {
                    let mut header = vec!["Case".to_string()];
                    header.extend(t.header.iter().cloned());
                    Table {
                        title: key,
                        header,
                        rows: Vec::new(),
                    }
                });
                for row in &t.rows {
                    let mut cells = vec![name.clone()];
                    cells.extend(row.iter().cloned());
                    entry.rows.push(cells);
                }
            }
        }
        // The steps as the canvas shows them: the last case's marks, a
        // failure of any case kept.
        for r in case_out.records {
            match outcome.records.iter_mut().find(|x| x.node == r.node) {
                Some(x) if matches!(x.status, Status::Failed(_)) => {}
                Some(x) => *x = r,
                None => outcome.records.push(r),
            }
        }
        outcome.cases.push(CaseRecord {
            name,
            folder: folder.clone(),
            run_dir: case_out.run_dir,
            error: case_out.error,
            secs: case_out.secs,
        });
        if case_out.cancelled {
            outcome.cancelled = true;
            outcome.error = Some("the run was cancelled".into());
            break;
        }
    }
    let failed = outcome.cases.iter().filter(|c| c.error.is_some()).count();
    if outcome.error.is_none() && failed > 0 {
        outcome.error = Some(format!(
            "{failed} of {} cases stopped early",
            outcome.cases.len()
        ));
    }
    // The summary table, then the combined ones.
    let mut summary = Table::new("Cases", &["Case", "Result", "Time (s)", "Folder"]);
    for c in &outcome.cases {
        summary.row(vec![
            c.name.clone(),
            c.error.clone().unwrap_or_else(|| "every step ran".into()),
            format!("{:.0}", c.secs),
            c.run_dir.display().to_string(),
        ]);
    }
    let mut tables = vec![summary];
    tables.extend(combined.into_values());
    for t in &tables {
        let file = opts
            .run_dir
            .join(format!("batch - {}.csv", safe_name(&t.title)));
        let _ = std::fs::write(file, t.csv());
    }
    let report = Report {
        node: src.id,
        title: format!("Batch over {}", root.display()),
        notes: vec![format!(
            "{} cases, {} ran every step.",
            outcome.cases.len(),
            outcome.cases.len() - failed
        )],
        tables,
        motion: None,
    };
    let _ = std::fs::write(
        opts.run_dir.join("batch-summary.md"),
        format!("# {}\n\n{}", wf.name, report.markdown()),
    );
    let _ = std::fs::write(opts.run_dir.join("workflow.rdsflow"), wf.to_json());
    outcome.reports = vec![report];
    outcome.secs = t_start.elapsed().as_secs_f64();
    outcome
}

/// `run-summary.md`: what ran on what, what each step said, the reports.
fn summary_markdown(wf: &Workflow, ctx: &Ctx, outcome: &Outcome) -> String {
    let mut out = format!(
        "# {}\n\nRun {} {} by rust-dicom-station {}, {:.0} s.\n\n",
        wf.name,
        ctx.date,
        ctx.time,
        env!("CARGO_PKG_VERSION"),
        outcome.secs
    );
    if !wf.description.trim().is_empty() {
        out.push_str(&format!("{}\n\n", wf.description.trim()));
    }
    match &outcome.error {
        None => out.push_str("Every step ran.\n\n"),
        Some(e) => out.push_str(&format!("**The run stopped:** {e}\n\n")),
    }
    if !ctx.datasets.is_empty() {
        out.push_str("## Inputs\n\n");
        for d in &ctx.datasets {
            out.push_str(&format!("- {}: `{}`\n", d.label, d.origin.display()));
        }
        out.push('\n');
    }
    out.push_str("## Steps\n\n");
    for (i, r) in outcome.records.iter().enumerate() {
        let status = match &r.status {
            Status::Done if r.reused => "unchanged, taken from the last run".to_string(),
            Status::Done => format!("done in {:.1} s", r.secs),
            Status::Failed(e) => format!("failed: {e}"),
            Status::NotRun => "not run".into(),
        };
        out.push_str(&format!("{}. **{}** - {status}\n", i + 1, r.label));
        for l in &r.lines {
            out.push_str(&format!("   - {l}\n"));
        }
    }
    out.push('\n');
    for r in &ctx.reports {
        out.push_str(&r.markdown());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_write_as_csv_and_markdown() {
        let mut t = Table::new("Volumes", &["Structure", "cm3"]);
        t.row(vec!["heart, total".into(), "612.4".into()]);
        t.row(vec!["say \"hi\"".into(), "1".into()]);
        assert_eq!(
            t.csv(),
            "Structure,cm3\n\"heart, total\",612.4\n\"say \"\"hi\"\"\",1\n"
        );
        let md = t.markdown();
        assert!(md.starts_with("| Structure | cm3 |\n|---|---|\n"), "{md}");
    }

    #[test]
    fn run_folders_never_collide() {
        let base = std::env::temp_dir().join(format!("rds-wf-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let mut wf = Workflow::new("demo: one");
        wf.output.folder = "{workflow}".into();
        let a = run_folder(&wf, &base);
        assert_eq!(a, base.join("demo_ one"));
        std::fs::create_dir_all(&a).unwrap();
        let b = run_folder(&wf, &base);
        assert_eq!(b, base.join("demo_ one (2)"));
        let _ = std::fs::remove_dir_all(&base);
    }
}
