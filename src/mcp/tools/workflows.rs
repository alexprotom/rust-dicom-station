//! Saved workflows, run by the server: what is there to run, and running
//! one on folders under the roots.
//!
//! A workflow ([`crate::workflow::graph`]) is the viewer's own steps wired
//! into a file. Running one here is running it headless, the way
//! `workflow_cli` does, with the server's rules around it:
//!
//! * every folder it reads is under a configured root (the ones given in
//!   the call, and any it names itself), and passes the PHI gate - a
//!   workflow reads its folders as they are, so under any policy but
//!   `allow` they must already be anonymized;
//! * everything it writes goes under the session's output folder: the run
//!   folder is made there, and a workflow whose steps name an absolute
//!   folder of their own is refused;
//! * what it answers - step lines, report tables, folder names - passes the
//!   redactor like every other answer ([`crate::mcp::Core::call_public`]).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Result};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::super::config::PhiPolicy;
use super::super::phi::{self, clean_text};
use super::super::Core;
use super::NoArgs;
use crate::loader;
use crate::progress::Progress;
use crate::workflow::graph::catalog::Op;
use crate::workflow::graph::exec::{self, Channel, RunOptions, Status};
use crate::workflow::graph::{store, Workflow};

/// A workflow the server can run: from the workflows folder, or an
/// example compiled in.
struct Offered {
    /// What the client names it by: the workflow's name.
    name: String,
    /// `file` or `example`.
    source: &'static str,
    file: Option<PathBuf>,
    workflow: Workflow,
}

fn workflows_dir(core: &Core) -> PathBuf {
    core.session
        .config
        .workflows_dir
        .clone()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(store::user_dir)
}

fn offered(core: &Core) -> Vec<Offered> {
    let mut out: Vec<Offered> = store::list(&workflows_dir(core))
        .into_iter()
        .map(|(path, wf)| Offered {
            name: wf.name.clone(),
            source: "file",
            file: Some(path),
            workflow: wf,
        })
        .collect();
    for e in store::EXAMPLES {
        out.push(Offered {
            name: e.name.to_string(),
            source: "example",
            file: None,
            workflow: e.workflow(),
        });
    }
    out
}

/// The inputs of a workflow, by title: its folder steps.
fn inputs_of(wf: &Workflow) -> Vec<(u32, String, &'static str)> {
    wf.nodes
        .iter()
        .filter_map(|n| match &n.op {
            Op::LoadFolder(_) => Some((n.id, n.label(), "folder")),
            Op::LoadFolders(_) => Some((n.id, n.label(), "folder of cases (a batch)")),
            _ => None,
        })
        .collect()
}

pub fn list_workflows(core: &mut Core, _a: NoArgs, _p: &Progress) -> Result<Value> {
    let list: Vec<Value> = offered(core)
        .iter()
        .map(|o| {
            let f = o.workflow.check();
            json!({
                "name": clean_text(&o.name),
                "source": o.source,
                "file": o.file.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()),
                "description": clean_text(&o.workflow.description),
                "steps": o.workflow.nodes.iter().map(|n| json!({
                    "step": clean_text(&n.label()),
                    "kind": n.op.kind().info().name,
                })).collect::<Vec<_>>(),
                // The folders the file names are not told: they are the
                // author's, and a folder name can carry a patient's.
                "inputs": inputs_of(&o.workflow).iter().map(|(_, title, what)| json!({
                    "input": clean_text(title),
                    "is": what,
                })).collect::<Vec<_>>(),
                "problems": f.errors.iter().map(|(_, m)| clean_text(m)).collect::<Vec<_>>(),
            })
        })
        .collect();
    Ok(json!({
        "workflows": list,
        "note": "run one with run_workflow, giving a folder under a root for each input",
    }))
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunArgs {
    /// The workflow, by the name list_workflows gives (or its file name).
    pub workflow: String,
    /// A folder under a root for each input, by the input's title:
    /// {"Cardiac CT": "root1/UPSTAR/CCT"}. An input left out reads the
    /// folder the workflow file names, which must be under a root too.
    #[serde(default)]
    pub inputs: BTreeMap<String, String>,
    /// Keep the studies the run read, with what it filed, open as datasets
    /// afterwards (as many as max_open_datasets allows).
    #[serde(default)]
    pub open_results: bool,
    /// Run the independent rows at the start of the workflow at the same
    /// time.
    #[serde(default)]
    pub parallel: bool,
}

/// Every folder a step names for its own output, which must stay inside
/// the run folder here.
fn absolute_outputs(wf: &Workflow) -> Vec<String> {
    let mut out = Vec::new();
    for n in &wf.nodes {
        let folder = match &n.op {
            Op::ExportDicom(p) => &p.folder,
            Op::SaveReport(p) => &p.folder,
            Op::Anonymize(p) => &p.folder,
            Op::Drr(p) => &p.folder,
            _ => continue,
        };
        if Path::new(folder.trim()).is_absolute() {
            out.push(n.label());
        }
    }
    out
}

/// Refuse what the server's rules do not allow a workflow to reach.
fn outside_reads(core: &Core, wf: &Workflow) -> Result<()> {
    for n in &wf.nodes {
        match &n.op {
            Op::LoadFromArchive(p) => {
                let root = if p.archive.trim().is_empty() {
                    crate::archive::default_root()
                } else {
                    PathBuf::from(p.archive.trim())
                };
                core.session.config.resolve_input(&root).map_err(|_| {
                    anyhow!(
                        "'{}' reads the archive, which is not under a configured root",
                        n.label()
                    )
                })?;
            }
            Op::Dvh(p) if !p.protocol_file.trim().is_empty() => {
                core.session
                    .resolve_input(Path::new(p.protocol_file.trim()))
                    .map_err(|_| {
                        anyhow!(
                            "'{}' reads a protocol file that is not under a configured root",
                            n.label()
                        )
                    })?;
            }
            Op::ArchiveImport(p) if !p.archive.trim().is_empty() => {
                bail!(
                    "'{}' files into an archive of its own; the server files only into the \
                     station's",
                    n.label()
                );
            }
            _ => {}
        }
    }
    Ok(())
}

/// The PHI gate for a folder a workflow reads: a header-only read of it,
/// the same sample and verdict `open_dataset` uses.
fn gate(core: &Core, dir: &Path, title: &str, p: &Progress) -> Result<()> {
    let study = loader::load_directory_to_merge(dir, p)?;
    let sample = phi::sample_files(&study, Some(dir));
    let verdict = phi::classify(&sample)?;
    core.session.add_values(verdict.values().to_vec());
    if !verdict.is_anonymized() && core.session.config.phi_policy != PhiPolicy::Allow {
        bail!(
            "the input '{title}' {}. A workflow reads its folders as they are, so under the \
             '{}' policy they must be anonymized first (the anonymize tool)",
            verdict.describe(),
            core.session.config.phi_policy.label()
        );
    }
    Ok(())
}

pub fn run_workflow(core: &mut Core, a: RunArgs, p: &Progress) -> Result<Value> {
    let want = a.workflow.trim().to_lowercase();
    let o = offered(core)
        .into_iter()
        .find(|o| {
            o.name.to_lowercase() == want
                || o.file.as_ref().is_some_and(|f| {
                    f.file_name()
                        .is_some_and(|n| n.to_string_lossy().to_lowercase() == want)
                        || f.file_stem()
                            .is_some_and(|n| n.to_string_lossy().to_lowercase() == want)
                })
        })
        .ok_or_else(|| anyhow!("no workflow '{}' (see list_workflows)", a.workflow))?;
    let wf = o.workflow;
    let abs = absolute_outputs(&wf);
    if !abs.is_empty() {
        bail!(
            "{} write to a folder of their own; here everything a run writes goes under the \
             output folder",
            abs.join(", ")
        );
    }
    outside_reads(core, &wf)?;

    // The inputs: each resolved under a root and through the gate.
    let mut inputs: BTreeMap<u32, PathBuf> = BTreeMap::new();
    let known = inputs_of(&wf);
    for given in a.inputs.keys() {
        if !known.iter().any(|(_, t, _)| t.eq_ignore_ascii_case(given)) {
            bail!(
                "the workflow has no input '{given}'; its inputs are {}",
                known
                    .iter()
                    .map(|(_, t, _)| format!("'{t}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }
    for (id, title, _) in &known {
        let raw = match a.inputs.iter().find(|(k, _)| k.eq_ignore_ascii_case(title)) {
            Some((_, v)) => PathBuf::from(v),
            None => {
                let node = wf.node(*id).expect("listed");
                PathBuf::from(match &node.op {
                    Op::LoadFolder(p) => p.path.trim().to_string(),
                    Op::LoadFolders(p) => p.path.trim().to_string(),
                    _ => String::new(),
                })
            }
        };
        if raw.as_os_str().is_empty() {
            bail!("give a folder for the input '{title}'");
        }
        let (real, _) = core
            .session
            .resolve_input(&raw)
            .map_err(|e| anyhow!("input '{title}': {e:#}"))?;
        if !real.is_dir() {
            bail!("input '{title}' is not a folder");
        }
        p.set(format!("Checking '{title}' for identifying data"));
        // A batch reads the subfolders its pattern matches, each a case:
        // every one of them passes the gate, the others are never read.
        match wf.node(*id).map(|n| &n.op) {
            Some(Op::LoadFolders(prm)) => {
                for case in exec::batch_cases(&real, &prm.pattern)? {
                    gate(core, &case, title, p)?;
                }
            }
            _ => gate(core, &real, title, p)?,
        }
        inputs.insert(*id, real);
    }

    let run_dir = core.session.fresh_out_subdir(&format!(
        "workflow-{}",
        crate::workflow::graph::safe_name(&wf.name)
    ))?;
    let cfg = &core.session.config;
    let mut opts = RunOptions::new(&wf, &run_dir, cfg.models_dir());
    opts.run_dir = run_dir.clone();
    opts.inputs = inputs;
    opts.allow_download = cfg.allow_model_download;
    opts.volume_budget_mb = cfg.volume_cache_mb;
    opts.parallel = a.parallel;
    // The run's messages are the progress the client already sees; the
    // events have nobody to go to.
    let (tx, _rx) = std::sync::mpsc::channel();
    let channel = Channel::new(tx, None);
    let out = exec::run(&wf, &opts, p, &channel);

    let steps: Vec<Value> = out
        .records
        .iter()
        .map(|r| {
            json!({
                "step": clean_text(&r.label),
                "status": match &r.status {
                    Status::Done => "done".to_string(),
                    Status::Failed(e) => format!("failed: {}", clean_text(e)),
                    Status::NotRun => "not run".to_string(),
                },
                "seconds": (r.secs * 10.0).round() / 10.0,
                "lines": r.lines.iter().map(|l| clean_text(l)).collect::<Vec<_>>(),
            })
        })
        .collect();
    let reports: Vec<Value> = out
        .reports
        .iter()
        .map(|r| {
            json!({
                "title": clean_text(&r.title),
                "notes": r.notes.iter().map(|n| clean_text(n)).collect::<Vec<_>>(),
                "tables": r.tables.iter().map(|t| json!({
                    "title": clean_text(&t.title),
                    "header": t.header,
                    "rows": t.rows.iter().map(|row| row.iter().map(|c| clean_text(c)).collect::<Vec<_>>()).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let cases: Vec<Value> = out
        .cases
        .iter()
        .map(|c| {
            json!({
                "case": clean_text(&c.name),
                "result": c.error.as_deref().map(clean_text).unwrap_or_else(|| "every step ran".into()),
            })
        })
        .collect();
    let mut opened = Vec::new();
    let mut not_opened = 0usize;
    if a.open_results {
        for st in out.studies {
            let origin = st
                .study
                .series
                .first()
                .and_then(|s| s.files.first())
                .and_then(|f| f.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| run_dir.clone());
            let label = core
                .session
                .resolve_input(&origin)
                .map(|(_, l)| l)
                .unwrap_or_else(|_| "output".into());
            match core
                .session
                .add_dataset(st.study, origin, label, Vec::new())
            {
                Ok(ds) => opened.push(json!({ "dataset": ds.id, "input": clean_text(&st.label) })),
                Err(_) => not_opened += 1,
            }
        }
    }
    Ok(json!({
        "workflow": clean_text(&wf.name),
        "folder": run_dir.to_string_lossy(),
        "ok": out.error.is_none(),
        "error": out.error.as_deref().map(clean_text),
        "seconds": out.secs.round(),
        "steps": steps,
        "reports": reports,
        "cases": cases,
        "datasets": opened,
        "not_opened": not_opened,
    }))
}
