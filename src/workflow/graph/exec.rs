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

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};

use crate::loader::{self, LoadedStudy};
use crate::motion::MotionReport;
use crate::progress::{self, Progress, ProgressSink};
use crate::registration::{RegistrationResult, Transform3};
use crate::volume::Volume;

use super::catalog::Op;
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
pub struct Dataset {
    /// The folder node's title: what reports and export folders call it.
    pub label: String,
    pub node: u32,
    pub origin: PathBuf,
    /// Where it asked to be shown (`None`: the next free workspace).
    pub workspace: Option<usize>,
    pub study: LoadedStudy,
    /// Volumes of series other than the displayed one, newest last.
    volumes: Vec<(String, Arc<Volume>)>,
    /// Structure sets (by index) this run made or added to.
    pub touched_sets: BTreeSet<usize>,
    /// Segmentation series (by index) this run made or added to.
    pub touched_segs: BTreeSet<usize>,
}

/// Volumes kept per study beyond the displayed one. A 4D group is walked a
/// phase at a time; keeping them all would hold gigabytes.
const EXTRA_VOLUMES: usize = 2;

/// A registration the run made.
pub enum RegEntry {
    Pair {
        fixed: (usize, String),
        moving: (usize, String),
        result: Box<RegistrationResult>,
    },
    Group {
        ds: usize,
        group: usize,
        moving: (usize, String),
        /// (phase label, phase series UID, phase → moving).
        phases: Vec<(String, String, Arc<Transform3>)>,
    },
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
    /// run does not wait.
    ack: Option<Receiver<()>>,
}

impl Channel {
    pub fn new(tx: Sender<Event>, ack: Option<Receiver<()>>) -> Channel {
        Channel { tx, ack }
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
        let Some(ack) = &self.ack else {
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
}

/// Everything a run leaves behind.
pub struct Outcome {
    pub run_dir: PathBuf,
    pub records: Vec<NodeRecord>,
    /// Every study the run read, as it ends.
    pub studies: Vec<ShownStudy>,
    pub reports: Vec<Report>,
    /// Why the run stopped early; `None` when every step ran.
    pub error: Option<String>,
    pub cancelled: bool,
    pub secs: f64,
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
    pub datasets: Vec<Dataset>,
    pub regs: Vec<RegEntry>,
    pub reports: Vec<Report>,
}

impl Ctx<'_> {
    pub fn ds(&self, i: usize) -> Result<&Dataset> {
        self.datasets
            .get(i)
            .ok_or_else(|| anyhow!("internal: no study {i}"))
    }

    pub fn ds_mut(&mut self, i: usize) -> Result<&mut Dataset> {
        self.datasets
            .get_mut(i)
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
        self.datasets.push(Dataset {
            label,
            node,
            origin,
            workspace,
            study,
            volumes: Vec::new(),
            touched_sets: BTreeSet::new(),
            touched_segs: BTreeSet::new(),
        });
        self.datasets.len() - 1
    }

    /// The volume of one series, read when it is not the displayed one and
    /// not among the last few read.
    pub fn volume(&mut self, ds: usize, uid: &str, p: &Progress) -> Result<Arc<Volume>> {
        let d = self.ds_mut(ds)?;
        if d.study.has_volume()
            && d.study
                .series
                .get(d.study.active_series)
                .map(|s| s.uid.as_str())
                == Some(uid)
        {
            return Ok(d.study.volume.clone());
        }
        if let Some((_, v)) = d.volumes.iter().find(|(u, _)| u == uid) {
            return Ok(v.clone());
        }
        let info = d
            .study
            .series
            .iter()
            .find(|s| s.uid == uid)
            .cloned()
            .ok_or_else(|| anyhow!("the series is no longer in '{}'", d.label))?;
        let (vol, _, _) = loader::load_series_volume(&info, p)?;
        let vol = Arc::new(vol);
        let d = self.ds_mut(ds)?;
        while d.volumes.len() >= EXTRA_VOLUMES {
            d.volumes.remove(0);
        }
        d.volumes.push((uid.to_string(), vol.clone()));
        Ok(vol)
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
        d.volumes
            .iter()
            .find(|(u, _)| u == uid)
            .map(|(_, v)| v.clone())
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

/// Run `wf` on the calling thread. See the module documentation.
pub fn run(wf: &Workflow, opts: &RunOptions, p: &Progress, channel: &Channel) -> Outcome {
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
    };
    let mut outcome = Outcome {
        run_dir: opts.run_dir.clone(),
        records: Vec::new(),
        studies: Vec::new(),
        reports: Vec::new(),
        error: None,
        cancelled: false,
        secs: 0.0,
    };

    let findings = wf.check();
    if !findings.ok() {
        outcome.error = Some(
            findings
                .errors
                .iter()
                .map(|(_, m)| m.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        return outcome;
    }
    let order = match wf.order() {
        Ok(o) => o,
        Err(e) => {
            outcome.error = Some(format!("{e:#}"));
            return outcome;
        }
    };
    if let Err(e) = std::fs::create_dir_all(&opts.run_dir) {
        outcome.error = Some(format!(
            "the run folder {} cannot be made: {e}",
            opts.run_dir.display()
        ));
        return outcome;
    }
    channel.log(format!("Run folder: {}", opts.run_dir.display()));

    let mut values: BTreeMap<(u32, usize), Value> = BTreeMap::new();
    let n = order.len();
    for (index, &id) in order.iter().enumerate() {
        let node = wf.node(id).expect("ordered ids exist");
        if p.cancelled() {
            outcome.cancelled = true;
            outcome.error = Some("the run was cancelled".into());
            break;
        }
        channel.send(Event::Started {
            node: id,
            index,
            of: n,
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
        let result = nodes::run_node(&mut ctx, node, &inputs, p);
        p.set_prefix("");
        p.set_outer(0.0, 1.0);
        let secs = t0.elapsed().as_secs_f64();
        match result {
            Ok(done) => {
                for (k, v) in done.outputs.into_iter().enumerate() {
                    values.insert((id, k), v);
                }
                channel.send(Event::Finished {
                    node: id,
                    lines: done.lines.clone(),
                    secs,
                });
                outcome.records.push(NodeRecord {
                    node: id,
                    label: node.label(),
                    status: Status::Done,
                    lines: done.lines,
                    secs,
                });
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
            Err(e) => {
                let cancelled = progress::is_cancellation(&e) || p.cancelled();
                let msg = if cancelled {
                    "cancelled".to_string()
                } else {
                    format!("{e:#}")
                };
                channel.send(Event::Failed {
                    node: id,
                    error: msg.clone(),
                });
                outcome.records.push(NodeRecord {
                    node: id,
                    label: node.label(),
                    status: Status::Failed(msg.clone()),
                    lines: Vec::new(),
                    secs,
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
    for &id in &order {
        if !outcome.records.iter().any(|r| r.node == id) {
            let node = wf.node(id).expect("ordered ids exist");
            outcome.records.push(NodeRecord {
                node: id,
                label: node.label(),
                status: Status::NotRun,
                lines: Vec::new(),
                secs: 0.0,
            });
        }
    }
    outcome.secs = t_start.elapsed().as_secs_f64();

    // The run's own two files.
    let _ = std::fs::write(opts.run_dir.join("workflow.rdsflow"), wf.to_json());
    let _ = std::fs::write(
        opts.run_dir.join("run-summary.md"),
        summary_markdown(&wf, &ctx, &outcome),
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
