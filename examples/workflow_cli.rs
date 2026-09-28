//! Run a saved workflow headless: the same run the viewer's *Workflows*
//! menu starts, with no window.
//!
//! ```text
//! cargo run --release --example workflow_cli -- <WORKFLOW.rdsflow> \
//!     [--input "<folder node title or id>=<DICOM_DIR>"]... \
//!     [--out <RESULTS_ROOT>] [--models <MODELS_DIR>] [--no-download] [--list]
//! ```
//!
//! `--input` points a folder node at another folder, which is what makes a
//! saved workflow a batch tool: the same file over one patient after
//! another. `--out` is the folder the run makes its own folder in (the
//! workflow's, else `<data folder>/workflow_runs`). `--list` names the
//! workflow's steps and folder nodes and exits.
//!
//! Every event of the run is printed to standard error; the exit code is 0
//! when every step ran.

use std::path::PathBuf;
use std::sync::mpsc;

use rust_dicom_station::models;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::workflow::graph::catalog::Op;
use rust_dicom_station::workflow::graph::exec::{self, Channel, Event, RunOptions};
use rust_dicom_station::workflow::graph::store;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut file: Option<PathBuf> = None;
    let mut inputs: Vec<(String, PathBuf)> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut models_dir: Option<PathBuf> = None;
    let mut download = true;
    let mut list = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--input" => {
                let v = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--input needs NAME=DIR"))?;
                let (k, p) = v
                    .split_once('=')
                    .ok_or_else(|| anyhow::anyhow!("--input needs NAME=DIR, got '{v}'"))?;
                inputs.push((k.trim().to_string(), PathBuf::from(p.trim())));
            }
            "--out" => out = args.next().map(PathBuf::from),
            "--models" => models_dir = args.next().map(PathBuf::from),
            "--no-download" => download = false,
            "--list" => list = true,
            other if file.is_none() && !other.starts_with("--") => {
                file = Some(PathBuf::from(other))
            }
            other => anyhow::bail!("unknown argument '{other}'"),
        }
    }
    let file = file.ok_or_else(|| anyhow::anyhow!("name a workflow file (.rdsflow)"))?;
    let mut wf = store::load(&file)?;
    if list {
        println!("{}", wf.name);
        for id in wf.order()? {
            let n = wf.node(id).expect("ordered ids exist");
            let path = match &n.op {
                Op::LoadFolder(p) => format!("  folder: {}", p.path),
                _ => String::new(),
            };
            println!("  {id}: {} ({}){path}", n.label(), n.op.kind().info().name);
        }
        return Ok(());
    }
    if let Some(o) = &out {
        wf.output.root = o.display().to_string();
    }
    let mut opts = RunOptions::new(
        &wf,
        &store::default_runs_dir(),
        models_dir.unwrap_or_else(models::default_root),
    );
    opts.base_dir = file.parent().map(|p| p.to_path_buf());
    opts.allow_download = download;
    for (key, path) in inputs {
        let node = wf
            .nodes
            .iter()
            .find(|n| {
                matches!(n.op, Op::LoadFolder(_))
                    && (n.id.to_string() == key || n.label().eq_ignore_ascii_case(&key))
            })
            .ok_or_else(|| anyhow::anyhow!("no folder node named '{key}' (see --list)"))?;
        opts.inputs.insert(node.id, path);
    }
    let (tx, rx) = mpsc::channel();
    let channel = Channel::new(tx, None);
    let labels: std::collections::BTreeMap<u32, String> =
        wf.nodes.iter().map(|n| (n.id, n.label())).collect();
    let printer = std::thread::spawn(move || {
        for e in rx {
            match e {
                Event::Started { node, index, of } => eprintln!(
                    "[{}/{of}] {}",
                    index + 1,
                    labels.get(&node).map(String::as_str).unwrap_or("?")
                ),
                Event::Finished { lines, secs, .. } => {
                    for l in lines {
                        eprintln!("      {l}");
                    }
                    eprintln!("      done in {secs:.1} s");
                }
                Event::Failed { error, .. } => eprintln!("      FAILED: {error}"),
                Event::Log(l) => eprintln!("{l}"),
                Event::Show(_) => {}
            }
        }
    });
    let outcome = exec::run(&wf, &opts, &Progress::default(), &channel);
    drop(channel);
    let _ = printer.join();
    eprintln!("results: {}", outcome.run_dir.display());
    match outcome.error {
        None => {
            eprintln!("every step ran, {:.0} s", outcome.secs);
            Ok(())
        }
        Some(e) => {
            eprintln!("the run stopped: {e}");
            std::process::exit(1);
        }
    }
}
