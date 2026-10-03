//! Tasks: workflows the clients hand in, run on the server's machine.
//!
//! The *task* mode of working: a client does not copy a study, it says
//! "run this workflow on that study of yours" and later pulls what the run
//! filed. A task is a workflow ([`crate::workflow::graph`]) plus
//! **bindings**: each input step of the workflow bound to a study (or a
//! patient) of the server's archive. The workflow is one of the server's
//! saved ones, one of the examples the program ships, one of the one-step
//! templates below, or a whole `.rdsflow` file sent inline.
//!
//! ## Running one
//!
//! One runner thread takes the queue in order, one task at a time (the
//! engines want the whole machine, and the rule is the one `rds-mcp` has).
//! For a task it:
//!
//! 1. rewrites the input steps to read the server's own data: a *DICOM
//!    folder* step reads the bound study's folder in the archive (a study
//!    folder *is* a DICOM folder), a *DICOM folders* batch the bound
//!    patient's folder (one case per study), a *From the archive* step the
//!    server's archive with the patient and the study's UID;
//! 2. refuses what would reach outside: a step writing to an absolute
//!    folder, an inline workflow reading a file of the server's, an input
//!    left unbound;
//! 3. runs it through [`exec::run`] into `<data>/pacs/tasks/<id>/run/`,
//!    turning the run's events into the task's log and its progress handle
//!    into the task's progress (and cancel, and the timeout);
//! 4. files every DICOM object the run wrote into the archive when the
//!    request asked for it (the default), so the client can pull the
//!    results - the archive files them by their own tags, into the study
//!    they were made on;
//! 5. keeps `TASK.json` beside the run, so a restarted server still tells
//!    what finished and what failed; a task that was queued or running when
//!    the server stopped is reported as interrupted.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

use super::config::Config;
use super::local::Paths;
use super::protocol::*;
use crate::archive::{Archive, PatientEntry};
use crate::progress::Progress;
use crate::workflow::graph::catalog::{self as cat, Op};
use crate::workflow::graph::exec::{self, Channel, Event, RunOptions};
use crate::workflow::graph::{store, Workflow};
use crate::workflow::params::SetChoice;

/// Kept in memory per task: at most this many log lines.
const MAX_LOG: usize = 5000;

/// One task as the queue holds it.
struct Entry {
    info: TaskInfo,
    progress: Arc<Progress>,
    /// What to run; `None` once it ran (or for a task read back from disk).
    plan: Option<Plan>,
}

struct Plan {
    wf: Workflow,
    file_results: bool,
    parallel: bool,
}

struct Inner {
    tasks: Vec<Entry>,
    queue: VecDeque<String>,
    stop: bool,
    counter: u64,
}

/// The queue and its runner.
pub struct Tasks {
    inner: Mutex<Inner>,
    wake: Condvar,
    config: Config,
    paths: Paths,
    archive_root: PathBuf,
    /// Taken for writing while the results are filed, as the server's
    /// upload and remove routes do.
    archive_lock: Arc<RwLock<()>>,
}

/// One workflow the server offers, with the graph itself.
pub struct Offered {
    pub info: WorkflowInfo,
    pub workflow: Workflow,
}

impl Tasks {
    /// Read the tasks of earlier runs and start the runner.
    pub fn start(
        config: Config,
        paths: Paths,
        archive_root: PathBuf,
        archive_lock: Arc<RwLock<()>>,
    ) -> Arc<Tasks> {
        let mut tasks = Vec::new();
        for (info, changed) in read_back(&paths.tasks()) {
            if changed {
                let _ = persist(&paths.tasks(), &info);
            }
            tasks.push(Entry {
                info,
                progress: Arc::new(Progress::default()),
                plan: None,
            });
        }
        let t = Arc::new(Tasks {
            inner: Mutex::new(Inner {
                tasks,
                queue: VecDeque::new(),
                stop: false,
                counter: 0,
            }),
            wake: Condvar::new(),
            config,
            paths,
            archive_root,
            archive_lock,
        });
        let runner = t.clone();
        std::thread::Builder::new()
            .name("rds-pacs tasks".into())
            .spawn(move || runner.run_queue())
            .expect("a thread for the task runner");
        t
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Stop taking tasks and cancel the one running.
    pub fn stop(&self) {
        let mut g = self.lock();
        g.stop = true;
        for e in &g.tasks {
            if e.info.state == TaskState::Running {
                e.progress.cancel();
            }
        }
        drop(g);
        self.wake.notify_all();
    }

    // ---- what can be run --------------------------------------------------

    /// Every workflow a client may name: the server's saved ones, the
    /// shipped examples, the templates.
    pub fn offered(&self) -> Vec<Offered> {
        let mut out: Vec<Offered> = store::list(&self.config.workflows_dir())
            .into_iter()
            .map(|(_, wf)| offer(wf, "file"))
            .collect();
        for e in store::EXAMPLES {
            out.push(offer(e.workflow(), "example"));
        }
        for wf in templates() {
            out.push(offer(wf, "template"));
        }
        out
    }

    // ---- the queue --------------------------------------------------------

    /// Check a request, plan it and put it at the end of the queue.
    pub fn submit(&self, req: TaskRequest, client: &str) -> Result<Submitted> {
        if !self.config.tasks {
            bail!("this server does not run tasks (tasks = false in its pacs.toml)");
        }
        let (wf, inline) = if !req.inline.trim().is_empty() {
            (
                Workflow::from_json(&req.inline).context("the workflow sent")?,
                true,
            )
        } else {
            let want = req.workflow.trim().to_lowercase();
            let o = self
                .offered()
                .into_iter()
                .find(|o| o.info.name.to_lowercase() == want)
                .ok_or_else(|| anyhow!("the server has no workflow '{}'", req.workflow))?;
            (o.workflow, false)
        };
        let archive = Archive::new(&self.archive_root);
        let patients = archive.scan()?;
        let wf = bind(&wf, &req.bindings, &patients, &self.archive_root, inline)?;
        let problems = wf.check();
        if !problems.ok() {
            bail!(
                "the workflow cannot run: {}",
                problems
                    .errors
                    .iter()
                    .map(|(_, m)| m.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            );
        }
        let mut g = self.lock();
        let waiting = g.tasks.iter().filter(|e| !e.info.state.finished()).count();
        if waiting >= self.config.max_queued_tasks {
            bail!("the queue is full ({waiting} task(s) waiting or running); try again later");
        }
        g.counter += 1;
        let id = format!("{}-{}", super::unix_now(), g.counter);
        let title = if req.title.trim().is_empty() {
            wf.name.clone()
        } else {
            req.title.trim().chars().take(120).collect()
        };
        let info = TaskInfo {
            id: id.clone(),
            title,
            workflow: wf.name.clone(),
            client: client.to_string(),
            state: TaskState::Queued,
            submitted: super::stamp(),
            ..Default::default()
        };
        let _ = persist(&self.paths.tasks(), &info);
        g.tasks.push(Entry {
            info,
            progress: Arc::new(Progress::default()),
            plan: Some(Plan {
                wf,
                file_results: req.file_results,
                parallel: req.parallel,
            }),
        });
        g.queue.push_back(id.clone());
        let position = g.queue.len() - 1
            + g.tasks
                .iter()
                .filter(|e| e.info.state == TaskState::Running)
                .count();
        drop(g);
        self.wake.notify_all();
        Ok(Submitted {
            task_id: id,
            position,
        })
    }

    /// Every task, newest first, without their logs.
    pub fn list(&self) -> Vec<TaskInfo> {
        let g = self.lock();
        let mut v: Vec<TaskInfo> = g
            .tasks
            .iter()
            .map(|e| {
                let mut i = live(e);
                i.log_len = i.log.len();
                i.log.clear();
                i
            })
            .collect();
        v.reverse();
        v
    }

    /// One task, with its log from line `log_from` on.
    pub fn get(&self, id: &str, log_from: usize) -> Option<TaskInfo> {
        let g = self.lock();
        let e = g.tasks.iter().find(|e| e.info.id == id)?;
        let mut i = live(e);
        i.log_len = i.log.len();
        i.log = i.log.split_off(log_from.min(i.log.len()));
        Some(i)
    }

    /// Cancel a task: a queued one at once, a running one at its next
    /// check. Only the client that handed it in, or an operator.
    pub fn cancel(&self, id: &str, caller: &str, admin: bool) -> Result<()> {
        let mut g = self.lock();
        let e = g
            .tasks
            .iter_mut()
            .find(|e| e.info.id == id)
            .ok_or_else(|| anyhow!("no task '{id}'"))?;
        if !admin && e.info.client != caller {
            bail!("only the client that handed the task in, or an operator, may cancel it");
        }
        match e.info.state {
            TaskState::Queued => {
                e.info.state = TaskState::Cancelled;
                e.info.finished = super::stamp();
                e.plan = None;
                let info = e.info.clone();
                g.queue.retain(|q| q != id);
                drop(g);
                let _ = persist(&self.paths.tasks(), &info);
            }
            TaskState::Running => e.progress.cancel(),
            _ => {}
        }
        Ok(())
    }

    /// A file the task wrote, by the name its info lists.
    pub fn file(&self, id: &str, name: &str) -> Option<PathBuf> {
        let g = self.lock();
        let e = g.tasks.iter().find(|e| e.info.id == id)?;
        e.info
            .files
            .iter()
            .any(|f| f == name)
            .then(|| run_dir(&self.paths, id).join(name))
    }

    // ---- the runner -------------------------------------------------------

    fn run_queue(self: Arc<Self>) {
        loop {
            let id = {
                let mut g = self.lock();
                loop {
                    if g.stop {
                        return;
                    }
                    if let Some(id) = g.queue.pop_front() {
                        break id;
                    }
                    g = self.wake.wait(g).unwrap_or_else(|e| e.into_inner());
                }
            };
            self.run_task(&id);
        }
    }

    fn log(&self, id: &str, line: String) {
        let mut g = self.lock();
        if let Some(e) = g.tasks.iter_mut().find(|e| e.info.id == id) {
            if e.info.log.len() < MAX_LOG {
                e.info.log.push(line);
            }
        }
    }

    fn run_task(self: &Arc<Self>, id: &str) {
        let (plan, progress) = {
            let mut g = self.lock();
            let Some(e) = g.tasks.iter_mut().find(|e| e.info.id == id) else {
                return;
            };
            let Some(plan) = e.plan.take() else {
                return;
            };
            e.info.state = TaskState::Running;
            e.info.started = super::stamp();
            (plan, e.progress.clone())
        };
        self.log(id, format!("started on {}", super::stamp()));
        let outcome = self.execute(id, &plan, &progress);
        let mut g = self.lock();
        let Some(e) = g.tasks.iter_mut().find(|e| e.info.id == id) else {
            return;
        };
        e.info.finished = super::stamp();
        e.info.progress = 1.0;
        match outcome {
            Ok(done) => {
                e.info.filed = done.filed;
                e.info.files = done.files;
                e.info.message = done.message;
                match done.error {
                    None => e.info.state = TaskState::Done,
                    Some(_) if done.cancelled => {
                        e.info.state = TaskState::Cancelled;
                        e.info.error = done.error;
                    }
                    Some(err) => {
                        e.info.state = TaskState::Failed;
                        e.info.error = Some(err);
                    }
                }
            }
            Err(err) => {
                e.info.state = if crate::progress::is_cancellation(&err) {
                    TaskState::Cancelled
                } else {
                    TaskState::Failed
                };
                e.info.error = Some(format!("{err:#}"));
            }
        }
        let line = format!(
            "{} on {}{}",
            e.info.state.label(),
            e.info.finished,
            e.info
                .error
                .as_deref()
                .map(|x| format!(": {x}"))
                .unwrap_or_default()
        );
        if e.info.log.len() < MAX_LOG {
            e.info.log.push(line);
        }
        let info = e.info.clone();
        drop(g);
        let _ = persist(&self.paths.tasks(), &info);
    }

    fn execute(self: &Arc<Self>, id: &str, plan: &Plan, progress: &Arc<Progress>) -> Result<Done> {
        let dir = run_dir(&self.paths, id);
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        let mut opts = RunOptions::new(&plan.wf, &dir, self.config.models_dir());
        opts.run_dir = dir.clone();
        opts.allow_download = self.config.allow_model_download;
        opts.volume_budget_mb = self.config.volume_cache_mb;
        opts.parallel = plan.parallel;

        // The run's events become the task's log, on a thread of their own
        // so the run never waits for the queue's lock.
        let (tx, rx) = mpsc::channel::<Event>();
        let labels: BTreeMap<u32, String> =
            plan.wf.nodes.iter().map(|n| (n.id, n.label())).collect();
        let logger = {
            let sink: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let s2 = sink.clone();
            let handle = std::thread::spawn(move || {
                for ev in rx {
                    let line = match ev {
                        Event::Started { node, index, of } => format!(
                            "step {}/{of}: {}",
                            index + 1,
                            labels.get(&node).cloned().unwrap_or_default()
                        ),
                        Event::Finished {
                            node,
                            lines,
                            secs,
                            reused,
                        } => {
                            let mut l = format!(
                                "done: {} ({secs:.1} s{})",
                                labels.get(&node).cloned().unwrap_or_default(),
                                if reused { ", reused" } else { "" }
                            );
                            for x in lines {
                                l.push_str("\n  ");
                                l.push_str(&x);
                            }
                            l
                        }
                        Event::Failed { node, error } => format!(
                            "failed: {}: {error}",
                            labels.get(&node).cloned().unwrap_or_default()
                        ),
                        Event::Log(l) => l,
                        Event::Show(_) => continue,
                    };
                    s2.lock().unwrap_or_else(|e| e.into_inner()).push(line);
                }
            });
            (sink, handle)
        };
        // Copy the collected lines and the progress into the task while it
        // runs, and enforce the timeout.
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watcher = {
            let finished = finished.clone();
            let progress = progress.clone();
            let sink = logger.0.clone();
            let timeout = Duration::from_secs(self.config.task_timeout_minutes.max(1) * 60);
            let started = Instant::now();
            let me = self.clone();
            let id = id.to_string();
            std::thread::spawn(move || {
                while !finished.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(250));
                    if started.elapsed() > timeout {
                        progress.cancel();
                    }
                    me.copy_live(&id, &progress, &sink);
                }
            })
        };

        let archive = Archive::new(&self.archive_root);
        let before = counts(&archive);
        let channel = Channel::new(tx, None);
        let out = exec::run(&plan.wf, &opts, progress, &channel);
        drop(channel);
        let _ = logger.1.join();
        finished.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = watcher.join();
        self.copy_live(id, progress, &logger.0);

        let mut message = String::new();
        if out.error.is_none() && plan.file_results {
            progress.set("Filing the results into the archive");
            let _w = self.archive_lock.write().unwrap_or_else(|e| e.into_inner());
            let sum = archive.import(&dir, &Progress::default())?;
            message = format!("results filed: {}", sum.describe());
            if sum.stored == 0 && sum.duplicates > 0 {
                message.push_str(
                    "; nothing new: what the run wrote carries UIDs the archive holds already \
                     (a changed structure set written under its own UIDs is not filed again; \
                     let the steps put their results into a new set)",
                );
            }
            self.log(id, message.clone());
        }
        let after = counts(&archive);
        let filed: Vec<String> = after
            .iter()
            .filter(|(uid, n)| before.get(*uid).is_none_or(|b| b < *n))
            .map(|(uid, _)| uid.clone())
            .collect();
        let files = written_files(&dir);
        Ok(Done {
            error: out.error.clone(),
            cancelled: out.cancelled,
            filed,
            files,
            message,
        })
    }

    /// Bring the live log and progress into the task's info.
    fn copy_live(&self, id: &str, progress: &Progress, sink: &Arc<Mutex<Vec<String>>>) {
        let lines: Vec<String> =
            std::mem::take(&mut *sink.lock().unwrap_or_else(|e| e.into_inner()));
        let mut g = self.lock();
        if let Some(e) = g.tasks.iter_mut().find(|e| e.info.id == id) {
            for l in lines {
                if e.info.log.len() < MAX_LOG {
                    e.info.log.push(l);
                }
            }
            e.info.progress = progress.frac();
            e.info.message = progress.get();
        }
    }
}

/// What one run left.
struct Done {
    error: Option<String>,
    cancelled: bool,
    filed: Vec<String>,
    files: Vec<String>,
    message: String,
}

/// The task's info with its live progress.
fn live(e: &Entry) -> TaskInfo {
    let mut i = e.info.clone();
    if i.state == TaskState::Running {
        i.progress = e.progress.frac();
        let m = e.progress.get();
        if !m.is_empty() {
            i.message = m;
        }
    }
    i
}

fn run_dir(paths: &Paths, id: &str) -> PathBuf {
    paths.tasks().join(id).join("run")
}

/// Files per study, for telling which studies a run filed into.
fn counts(a: &Archive) -> BTreeMap<String, usize> {
    a.scan()
        .unwrap_or_default()
        .into_iter()
        .flat_map(|p| p.studies.into_iter().map(|s| (s.study_uid, s.files)))
        .collect()
}

/// The files a run wrote other than DICOM: what a client may fetch.
fn written_files(dir: &Path) -> Vec<String> {
    const KEEP: &[&str] = &[
        "csv", "md", "txt", "json", "png", "gif", "jpg", "jpeg", "html",
    ];
    let mut out: Vec<String> = walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| KEEP.contains(&x.to_lowercase().as_str()))
        })
        .filter_map(|e| {
            e.path()
                .strip_prefix(dir)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
        })
        .collect();
    out.sort();
    out
}

fn persist(dir: &Path, info: &TaskInfo) -> Result<()> {
    let d = dir.join(&info.id);
    std::fs::create_dir_all(&d).with_context(|| format!("create {}", d.display()))?;
    let text = serde_json::to_string_pretty(info).expect("plain data serialises");
    let tmp = d.join("TASK.json.tmp");
    std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, d.join("TASK.json")).context("write TASK.json")
}

/// The tasks of earlier runs, oldest first. A task that was queued or
/// running when the server stopped comes back failed, with the reason; the
/// flag says its file needs writing again.
fn read_back(dir: &Path) -> Vec<(TaskInfo, bool)> {
    let mut v: Vec<(TaskInfo, bool)> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| std::fs::read_to_string(e.path().join("TASK.json")).ok())
                .filter_map(|t| serde_json::from_str::<TaskInfo>(&t).ok())
                .map(|mut i| {
                    let changed = !i.state.finished();
                    if changed {
                        i.state = TaskState::Failed;
                        i.error = Some("interrupted: the server stopped before it finished".into());
                        if i.finished.is_empty() {
                            i.finished = super::stamp();
                        }
                    }
                    (i, changed)
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by(|a, b| {
        (a.0.submitted.as_str(), task_number(&a.0.id))
            .cmp(&(b.0.submitted.as_str(), task_number(&b.0.id)))
    });
    v
}

fn task_number(id: &str) -> (u64, u64) {
    let mut it = id.split('-').map(|x| x.parse::<u64>().unwrap_or(0));
    (it.next().unwrap_or(0), it.next().unwrap_or(0))
}

// ---- binding --------------------------------------------------------------

/// What a workflow's inputs are.
pub fn inputs_of(wf: &Workflow) -> Vec<InputInfo> {
    wf.nodes
        .iter()
        .filter_map(|n| {
            let kind = match &n.op {
                Op::LoadFolder(_) => InputKind::Folder,
                Op::LoadFolders(_) => InputKind::Batch,
                Op::LoadFromArchive(_) => InputKind::Archive,
                _ => return None,
            };
            Some(InputInfo {
                node: n.id,
                title: n.label(),
                kind,
            })
        })
        .collect()
}

fn offer(wf: Workflow, source: &str) -> Offered {
    // Inputs are checked once they are bound; what the check says about
    // them now (no folder, no patient) is not a problem of the workflow.
    let mut probe = wf.clone();
    for n in &mut probe.nodes {
        match &mut n.op {
            Op::LoadFolder(p) => p.path = "bound".into(),
            Op::LoadFolders(p) => p.path = "bound".into(),
            Op::LoadFromArchive(p) => p.patient = "bound".into(),
            _ => {}
        }
    }
    let problems = probe
        .check()
        .errors
        .into_iter()
        .map(|(_, m)| m)
        .chain(
            crate::workflow::graph::absolute_outputs(&wf)
                .into_iter()
                .map(|l| format!("'{l}' writes outside its run folder")),
        )
        .collect();
    Offered {
        info: WorkflowInfo {
            name: wf.name.clone(),
            source: source.to_string(),
            description: wf.description.clone(),
            steps: wf.nodes.iter().map(|n| n.label()).collect(),
            inputs: inputs_of(&wf),
            problems,
        },
        workflow: wf,
    }
}

/// `wf` with its inputs pointed at the server's data, or the reason it
/// may not run here. `inline`: the workflow came with the request rather
/// than from the server, so it may read no file of the server's own.
pub fn bind(
    wf: &Workflow,
    bindings: &[Binding],
    patients: &[PatientEntry],
    archive_root: &Path,
    inline: bool,
) -> Result<Workflow> {
    let abs = crate::workflow::graph::absolute_outputs(wf);
    if !abs.is_empty() {
        bail!(
            "{} write to a folder of their own; a task writes only into its run folder",
            abs.join(", ")
        );
    }
    let inputs = inputs_of(wf);
    for b in bindings {
        if !inputs.iter().any(|i| i.node == b.node) {
            bail!("the workflow has no input step {}", b.node);
        }
    }
    let mut out = wf.clone();
    let root = archive_root.display().to_string();
    for n in &mut out.nodes {
        let label = n.label();
        let bound = bindings.iter().find(|b| b.node == n.id);
        let find = |b: &Binding| -> Result<(&PatientEntry, Option<PathBuf>)> {
            let p = patients
                .iter()
                .find(|p| p.key() == b.patient_key)
                .ok_or_else(|| {
                    anyhow!("'{label}': the archive has no patient '{}'", b.patient_key)
                })?;
            if b.study_uid.is_empty() {
                return Ok((p, None));
            }
            let s = p
                .studies
                .iter()
                .find(|s| s.study_uid == b.study_uid)
                .ok_or_else(|| anyhow!("'{label}': the patient has no such study"))?;
            Ok((p, Some(s.dir.clone())))
        };
        match &mut n.op {
            Op::LoadFolder(p) => {
                let b = bound.ok_or_else(|| {
                    anyhow!("'{label}' reads a folder: bind it to a study of the archive")
                })?;
                let (pt, study) = find(b)?;
                p.path = study
                    .unwrap_or_else(|| pt.dir.clone())
                    .display()
                    .to_string();
            }
            Op::LoadFolders(p) => {
                let b = bound.ok_or_else(|| {
                    anyhow!("'{label}' is a batch: bind it to a patient of the archive")
                })?;
                if !b.study_uid.is_empty() {
                    bail!("'{label}' is a batch over a patient's studies: bind it to the patient");
                }
                let (pt, _) = find(b)?;
                p.path = pt.dir.display().to_string();
                p.pattern = "*".into();
            }
            Op::LoadFromArchive(p) => {
                p.archive = root.clone();
                match bound {
                    Some(b) => {
                        let (pt, _) = find(b)?;
                        p.patient = if pt.id.is_empty() {
                            pt.name.clone()
                        } else {
                            pt.id.clone()
                        };
                        p.study = b.study_uid.clone();
                    }
                    None if p.patient.trim().is_empty() => {
                        bail!("'{label}' takes a study of the archive: bind it");
                    }
                    None => {}
                }
            }
            Op::ArchiveImport(p) => p.archive = root.clone(),
            Op::Dvh(p) if inline && !p.protocol_file.trim().is_empty() => {
                bail!("'{label}' reads a protocol file of the server's, which a workflow sent with the task may not");
            }
            _ => {}
        }
    }
    Ok(out)
}

// ---- templates --------------------------------------------------------------

/// The one-step workflows offered besides the saved ones and the examples:
/// a study of the archive in, one tool, the result exported (and filed
/// back by the task).
pub fn templates() -> Vec<Workflow> {
    let mut out = Vec::new();

    let mut wf = Workflow::new("Body contour");
    wf.description = "The patient's outline (BODY) on the largest CT of a study, \
                      filed back as an RT structure set."
        .into();
    let a = wf.add(Op::LoadFromArchive(Default::default()), "Study", [0.0, 0.0]);
    let s = wf.add(Op::SelectImage(Default::default()), "", [220.0, 0.0]);
    // Into a structure set of its own: a changed existing set written out
    // under its own UIDs is one the archive already holds, and would not
    // be filed again.
    let b = wf.add(
        Op::BodyContour(cat::BodyContour {
            set: SetChoice::New,
            set_label: "Body contour".into(),
            ..Default::default()
        }),
        "",
        [440.0, 0.0],
    );
    let e = wf.add(Op::ExportDicom(Default::default()), "", [660.0, 0.0]);
    wf.link(a, 0, s, 0);
    wf.link(s, 0, b, 0);
    wf.link(b, 0, e, 0);
    out.push(wf);

    let mut wf = Workflow::new("Organs (TotalSegmentator, fast)");
    wf.description = "Every organ TotalSegmentator finds on the largest CT of a study, \
                      filed back as an RT structure set. Needs the model on the server."
        .into();
    let a = wf.add(Op::LoadFromArchive(Default::default()), "Study", [0.0, 0.0]);
    let s = wf.add(Op::SelectImage(Default::default()), "", [220.0, 0.0]);
    let g = wf.add(
        Op::AutoSegment(cat::AutoSegment {
            organs: Vec::new(),
            set: SetChoice::New,
            ..Default::default()
        }),
        "",
        [440.0, 0.0],
    );
    let e = wf.add(Op::ExportDicom(Default::default()), "", [660.0, 0.0]);
    wf.link(a, 0, s, 0);
    wf.link(s, 0, g, 0);
    wf.link(g, 0, e, 0);
    out.push(wf);

    let mut wf = Workflow::new("Heart on every phase of a 4DCT");
    wf.description = "TotalSegmentator's heart on every phase of the study's 4D group, \
                      each phase's structures filed back on that phase. Needs the model \
                      on the server."
        .into();
    let a = wf.add(Op::LoadFromArchive(Default::default()), "Study", [0.0, 0.0]);
    let s = wf.add(Op::SelectGroup(Default::default()), "", [220.0, 0.0]);
    let g = wf.add(
        Op::AutoSegment(cat::AutoSegment {
            set: SetChoice::New,
            ..Default::default()
        }),
        "",
        [440.0, 0.0],
    );
    let e = wf.add(Op::ExportDicom(Default::default()), "", [660.0, 0.0]);
    wf.link(a, 0, s, 0);
    wf.link(s, 0, g, 0);
    wf.link(g, 0, e, 0);
    out.push(wf);

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_templates_are_sound_once_bound() {
        for wf in templates() {
            let o = offer(wf.clone(), "template");
            assert!(
                o.info.problems.is_empty(),
                "{}: {:?}",
                wf.name,
                o.info.problems
            );
            assert_eq!(o.info.inputs.len(), 1);
            assert_eq!(o.info.inputs[0].kind, InputKind::Archive);
        }
    }

    fn patients(root: &Path) -> Vec<PatientEntry> {
        vec![PatientEntry {
            name: "Doe^John".into(),
            id: "P1".into(),
            dir: root.join("P1"),
            studies: vec![crate::archive::StudyEntry {
                study_uid: "1.2.3".into(),
                dir: root.join("P1").join("1.2.3"),
                ..Default::default()
            }],
        }]
    }

    #[test]
    fn inputs_are_bound_to_the_server_archive_and_nothing_else() {
        let root = PathBuf::from("/srv/archive");
        let pts = patients(&root);
        let mut wf = Workflow::new("t");
        let f = wf.add(Op::LoadFolder(Default::default()), "CT", [0.0, 0.0]);
        let i = wf.add(Op::ArchiveImport(Default::default()), "", [200.0, 0.0]);
        wf.link(f, 0, i, 0);
        let b = |node, study: &str| Binding {
            node,
            patient_key: "P1".into(),
            study_uid: study.into(),
        };

        let bound = bind(&wf, &[b(f, "1.2.3")], &pts, &root, false).unwrap();
        match &bound.node(f).unwrap().op {
            Op::LoadFolder(p) => assert_eq!(PathBuf::from(&p.path), root.join("P1").join("1.2.3")),
            _ => unreachable!(),
        }
        match &bound.node(i).unwrap().op {
            Op::ArchiveImport(p) => assert_eq!(PathBuf::from(&p.archive), root),
            _ => unreachable!(),
        }
        // The whole patient.
        let whole = bind(&wf, &[b(f, "")], &pts, &root, false).unwrap();
        match &whole.node(f).unwrap().op {
            Op::LoadFolder(p) => assert_eq!(PathBuf::from(&p.path), root.join("P1")),
            _ => unreachable!(),
        }
        assert!(
            bind(&wf, &[], &pts, &root, false).is_err(),
            "unbound folder"
        );
        assert!(
            bind(&wf, &[b(f, "9.9")], &pts, &root, false).is_err(),
            "no such study"
        );
        assert!(
            bind(&wf, &[b(99, "1.2.3")], &pts, &root, false).is_err(),
            "no such input"
        );

        // Writing outside the run folder is refused.
        let mut out = wf.clone();
        let e = out.add(
            Op::ExportDicom(cat::ExportDicom {
                folder: if cfg!(windows) {
                    "C:\\elsewhere"
                } else {
                    "/elsewhere"
                }
                .into(),
                ..Default::default()
            }),
            "",
            [0.0, 200.0],
        );
        out.link(f, 0, e, 0);
        assert!(bind(&out, &[b(f, "1.2.3")], &pts, &root, false).is_err());

        // A batch takes a patient; the archive step the patient's id and the
        // study's UID.
        let mut wf2 = Workflow::new("t2");
        let l = wf2.add(Op::LoadFolders(Default::default()), "", [0.0, 0.0]);
        let r = wf2.add(Op::LoadFromArchive(Default::default()), "", [0.0, 100.0]);
        assert!(bind(&wf2, &[b(l, "1.2.3"), b(r, "1.2.3")], &pts, &root, false).is_err());
        let ok = bind(&wf2, &[b(l, ""), b(r, "1.2.3")], &pts, &root, false).unwrap();
        match &ok.node(r).unwrap().op {
            Op::LoadFromArchive(p) => {
                assert_eq!((p.patient.as_str(), p.study.as_str()), ("P1", "1.2.3"));
                assert_eq!(PathBuf::from(&p.archive), root);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn tasks_of_a_stopped_server_read_back_as_interrupted() {
        let dir = std::env::temp_dir().join("rds_pacs_tasks_back");
        let _ = std::fs::remove_dir_all(&dir);
        for (id, state) in [("100-1", TaskState::Done), ("100-2", TaskState::Running)] {
            persist(
                &dir,
                &TaskInfo {
                    id: id.into(),
                    state,
                    submitted: "2026-10-02 10:00:00 UTC".into(),
                    ..Default::default()
                },
            )
            .unwrap();
        }
        let back = read_back(&dir);
        assert_eq!(back.len(), 2);
        assert_eq!((back[0].0.state, back[0].1), (TaskState::Done, false));
        assert_eq!((back[1].0.state, back[1].1), (TaskState::Failed, true));
        assert!(back[1]
            .0
            .error
            .as_deref()
            .unwrap()
            .starts_with("interrupted"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
