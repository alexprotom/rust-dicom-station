//! Development probe: fetch the SegVol checkpoint and check it against the
//! layout the port is written for.
//!
//! This was the first thing run against the real weights, and it is what
//! produced `tests/data/segvol-tensors.csv`. Re-run it whenever the upstream
//! repository changes: it parses only `data.pkl`, so it is instant once the
//! file is local, and it exits non-zero if anything the port depends on has
//! moved.
//!
//! ```text
//! cargo run --release --example segvol_probe -- [MODELS_DIR] [--keys] [--csv FILE]
//! ```
//!
//! `MODELS_DIR` defaults to `segvol/` in the viewer's model folder. `--keys`
//! lists every tensor; `--csv` rewrites the recorded inventory.

use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::models::{self, Engine};
use rust_dicom_station::progress::Stderr;
use rust_dicom_station::segvol::layout::{self, Inventory, TensorInfo};
use rust_dicom_station::segvol::weights::{self, CHECKPOINT};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut models_dir: Option<PathBuf> = None;
    let mut list_keys = false;
    let mut csv: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--keys" => list_keys = true,
            "--csv" => csv = Some(PathBuf::from(args.next().expect("--csv needs a path"))),
            "-h" | "--help" => {
                eprintln!("usage: segvol_probe [MODELS_DIR] [--keys] [--csv FILE]");
                return Ok(());
            }
            other => models_dir = Some(PathBuf::from(other)),
        }
    }
    let models_dir =
        models_dir.unwrap_or_else(|| models::engine_dir(&models::default_root(), Engine::SegVol));

    if !CHECKPOINT.is_cached(&models_dir) {
        eprintln!(
            "Fetching {} ({} MB) into {}",
            CHECKPOINT.name,
            CHECKPOINT.bytes / 1_000_000,
            models_dir.display()
        );
        eprintln!(
            "NOTE: the SegVol weights carry no license declaration. They are \
             downloaded to your machine at your request and are not \
             redistributed by this program."
        );
    }
    let path = CHECKPOINT.ensure(&models_dir, &Stderr)?;
    eprintln!("\rusing {}", path.display());

    let reader = weights::open_checkpoint(&path)?;
    let inv = Inventory::of(reader.tensors.iter().map(|(name, meta)| TensorInfo {
        name,
        dtype: meta.dtype,
        shape: &meta.shape,
        contiguous: Some(meta.is_contiguous()),
    }));

    println!("tensors        {:>13}", inv.tensors);
    println!("values         {:>13}", inv.params);
    println!("  learnable    {:>13}", layout::EXPECTED_LEARNABLE);
    println!("  live         {:>13}", inv.live_params());
    println!("  dead branch  {:>13}", inv.dead_params);
    println!();
    println!("{:<16} {:>7}  {:>13}", "group", "tensors", "values");
    for (g, (count, params)) in &inv.per_group {
        println!("{g:<16} {count:>7}  {params:>13}");
    }
    println!();

    let problems = inv.problems();
    if problems.is_empty() {
        println!(
            "OK: matches the recorded layout — {} tensors, {} values, \
             {} ViT blocks, {} decoder layers, {} CLIP layers.",
            layout::EXPECTED_TENSORS,
            layout::EXPECTED_PARAMS,
            layout::EXPECTED_VIT_BLOCKS,
            layout::EXPECTED_DECODER_LAYERS,
            layout::EXPECTED_CLIP_LAYERS
        );
    } else {
        println!("{} problem(s) against the recorded layout:", problems.len());
        for p in &problems {
            println!("  - {p}");
        }
    }

    if list_keys {
        println!();
        for (name, meta) in &reader.tensors {
            println!(
                "{:<74} {:?} {:?}",
                layout::normalize_key(name),
                meta.dtype,
                meta.shape
            );
        }
    }
    if let Some(p) = csv {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&p)?);
        writeln!(f, "name,dtype,shape,numel")?;
        for (name, meta) in &reader.tensors {
            let shape: Vec<String> = meta.shape.iter().map(|d| d.to_string()).collect();
            writeln!(
                f,
                "{name},{:?},{},{}",
                meta.dtype,
                shape.join(" "),
                meta.numel()
            )?;
        }
        eprintln!("wrote {}", p.display());
    }

    if !problems.is_empty() {
        std::process::exit(1);
    }
    Ok(())
}
