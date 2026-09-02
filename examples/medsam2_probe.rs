//! Development probe: fetch a MedSAM2 checkpoint and check it against the
//! layout the port is written for.
//!
//! Unlike its SegVol counterpart this one does not *discover* the layout -
//! [`layout::expected`] derives all 471 tensors from the architecture, and
//! the probe's job is to prove that a real file agrees, key for key and shape
//! for shape. Run it whenever the upstream repository changes: it parses only
//! `data.pkl`, so it is instant once the file is local, and it exits non-zero
//! if anything the port depends on has moved.
//!
//! ```text
//! cargo run --release --example medsam2_probe -- [MODELS_DIR] [--variant NAME] [--keys] [--csv FILE]
//! ```
//!
//! `MODELS_DIR` defaults to `medsam2/` in the viewer's model folder.
//! `--variant` is one of `latest` (the default), `ct-lesion`, `mri-liver` or
//! `2411`. `--keys` lists every tensor; `--csv` writes the inventory to a
//! file.

use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::medsam2::layout::{self, TensorInfo};
use rust_dicom_station::medsam2::weights::{self, Variant};
use rust_dicom_station::models::{self, Engine};
use rust_dicom_station::nn::pickle::Dtype;
use rust_dicom_station::progress::Stderr;

fn main() -> anyhow::Result<()> {
    let mut dir: Option<PathBuf> = None;
    let mut variant = Variant::Latest;
    let mut list_keys = false;
    let mut csv: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--keys" => list_keys = true,
            "--csv" => csv = args.next().map(PathBuf::from),
            "--variant" => {
                let name = args.next().unwrap_or_default();
                variant = Variant::from_key(&name)
                    .ok_or_else(|| anyhow::anyhow!("unknown variant {name}"))?;
            }
            other => dir = Some(PathBuf::from(other)),
        }
    }
    let dir = dir.unwrap_or_else(|| models::engine_dir(&models::default_root(), Engine::MedSam2));

    let file = variant.file();
    eprintln!("{} ({}) in {}", file.name, variant.label(), dir.display());
    let path = file.ensure(&dir, &Stderr)?;
    eprintln!();

    let reader = weights::open_checkpoint(&path)?;
    let actual: Vec<TensorInfo> = reader
        .tensors
        .iter()
        .map(|(name, meta)| TensorInfo {
            name: layout::normalize_key(name).to_string(),
            shape: meta.shape.clone(),
            dtype: match meta.dtype {
                Dtype::F32 => "f32",
                Dtype::F16 => "f16",
                Dtype::F64 => "f64",
                _ => "int",
            },
        })
        .collect();

    let elements: usize = actual
        .iter()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();
    println!("{} tensors, {elements} elements", actual.len());
    for (group, total) in layout::group_totals() {
        println!("  {group:18} {total:>12}");
    }

    if list_keys {
        for t in &actual {
            println!("{:60} {:?} {}", t.name, t.shape, t.dtype);
        }
    }
    if let Some(path) = csv {
        let mut out = std::fs::File::create(&path)?;
        writeln!(out, "name,group,dtype,shape")?;
        for t in &actual {
            let shape = t
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            writeln!(
                out,
                "{},{},{},{shape}",
                t.name,
                layout::group_of(&t.name),
                t.dtype
            )?;
        }
        eprintln!("wrote {}", path.display());
    }

    let problems = layout::problems(&actual);
    if problems.is_empty() {
        println!(
            "layout matches: {} tensors, {} elements",
            layout::TENSOR_COUNT,
            layout::STATE_ELEMENTS
        );
        Ok(())
    } else {
        for p in &problems {
            eprintln!("  {p}");
        }
        anyhow::bail!("{} problems", problems.len())
    }
}
