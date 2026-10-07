//! Headless automatic segmentation with any model of the registry
//! (development / validation tool).
//!
//! ```text
//! cargo run --release --example seg_cli -- --list [--models ROOT]
//! cargo run --release --example seg_cli -- <dicom_dir> <out_prefix>
//!     [--model KEY] [--models ROOT] [--device auto|gpu|cpu] [--parts a,b,...]
//!     [--nnunet-folder DIR]...
//! ```
//!
//! `--model` is a registry key (`--list` prints them: `total_fast`,
//! `total_v3`, `total_mr`, `mrsegmentator`, `lung_vessels`,
//! `lungmask_lobes`, ...); the older `total` variant names (`fast3`,
//! `highres-v3`, ...) are accepted too. `--models` is the model folder, the
//! viewer's by default (`~/.local/share/RustDICOMStation/models` on Linux,
//! `%LOCALAPPDATA%\RustDICOMStation\models` on Windows). The viewer's
//! settings apply as they do in the viewer: the TotalSegmentator licence
//! number the licensed models download with, and the nnU-Net model folders
//! added in its model manager; `--nnunet-folder` adds one more for this run.
//!
//! Writes `<out_prefix>.bin` (u8 labels, `Volume::data` order) and
//! `<out_prefix>.json` (dims, spacing, origin, the model, its class table
//! and the structures found) so external tools - a comparison against the
//! reference implementation, say - can read the result.

use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::loader;
use rust_dicom_station::models;
use rust_dicom_station::nn::device::DevicePref;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::zoo::{AutoModel, RunOptions};

#[allow(dead_code)]
fn main() {
    if let Err(e) = run(std::env::args().skip(1).collect()) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

/// The tool, on its arguments (without the program name).
pub fn run(argv: Vec<String>) -> anyhow::Result<()> {
    let mut args = argv.into_iter();
    let mut positional: Vec<String> = Vec::new();
    let mut model_key: Option<String> = None;
    let mut folders: Vec<PathBuf> = Vec::new();
    let mut root = models::default_root();
    let mut device = DevicePref::Auto;
    let mut parts: Option<Vec<String>> = None;
    let mut list = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--list" => list = true,
            "--model" | "--variant" => model_key = Some(args.next().ok_or_else(usage)?),
            "--nnunet-folder" => folders.push(PathBuf::from(args.next().ok_or_else(usage)?)),
            "--models" | "--models-dir" => root = PathBuf::from(args.next().ok_or_else(usage)?),
            "--device" => {
                let v = args.next().unwrap_or_default();
                device = DevicePref::from_key(&v)
                    .ok_or_else(|| anyhow::anyhow!("unknown device {v:?}"))?;
            }
            "--parts" => {
                parts = Some(
                    args.next()
                        .ok_or_else(usage)?
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .collect(),
                )
            }
            other if !other.starts_with("--") => positional.push(other.to_string()),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let mut prefs = rust_dicom_station::settings::load();
    prefs.nnunet_folders.extend(folders);
    for (dir, why) in models::apply_settings(&prefs) {
        eprintln!("nnU-Net folder {} not used: {why}", dir.display());
    }
    let model = match &model_key {
        Some(k) => AutoModel::from_key(k)
            .ok_or_else(|| anyhow::anyhow!("unknown model {k:?}; --list prints the models"))?,
        None => AutoModel::default_for("CT"),
    };
    // A folder that is one engine's folder (what the older tool took) means
    // the model folder above it.
    if root
        .file_name()
        .is_some_and(|n| models::Engine::ALL.iter().any(|e| n == e.subdir()))
    {
        if let Some(parent) = root.parent() {
            root = parent.to_path_buf();
        }
    }
    if list {
        println!(
            "{:28} {:8} {:9} {:26} {:>8}  label",
            "key", "modality", "classes", "licence", "download"
        );
        for m in AutoModel::all() {
            let need = m.download_needed(None, &root);
            println!(
                "{:28} {:8} {:9} {:26} {:>8}  {}",
                m.key(),
                m.modality().label(),
                m.classes().len(),
                m.licence().name(),
                if need == 0 {
                    "ready".to_string()
                } else {
                    models::human_bytes(need)
                },
                m.label()
            );
        }
        return Ok(());
    }
    if positional.len() != 2 {
        return Err(usage());
    }
    let dir = PathBuf::from(&positional[0]);
    let out_prefix = positional[1].clone();
    let names = model.part_names();
    let parts = match parts {
        None => None,
        Some(want) => {
            let mut on = vec![false; names.len()];
            for p in &want {
                let i = names
                    .iter()
                    .position(|n| n.eq_ignore_ascii_case(p))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown part {p}; {} has {}",
                            model.key(),
                            names.join(", ")
                        )
                    })?;
                on[i] = true;
            }
            Some(on)
        }
    };

    let progress = Progress::default();
    eprintln!("loading {} ...", dir.display());
    let study = loader::load_directory(&dir, &progress)?;
    let vol = &study.volume;
    eprintln!(
        "volume {}x{}x{} @ {:.3}/{:.3}/{:.3} mm; model {} ({})",
        vol.dims[0],
        vol.dims[1],
        vol.dims[2],
        vol.spacing[0],
        vol.spacing[1],
        vol.spacing[2],
        model.key(),
        model.label()
    );

    let ap = Progress::default();
    let t = std::time::Instant::now();
    let done = std::sync::atomic::AtomicBool::new(false);
    let opts = RunOptions { device, parts };
    let result = std::thread::scope(|s| {
        let (done_ref, ap_ref) = (&done, &ap);
        let printer = s.spawn(move || {
            let mut last = String::new();
            while !done_ref.load(std::sync::atomic::Ordering::Relaxed) {
                let msg = ap_ref.get();
                if msg != last {
                    eprintln!(
                        "[{:6.1}s] {:5.1}% {}",
                        t.elapsed().as_secs_f64(),
                        ap_ref.frac() * 100.0,
                        msg
                    );
                    last = msg;
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });
        let r = model.run(vol, &opts, &root, &ap);
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = printer.join();
        r
    })?;
    eprintln!(
        "finished in {:.1}s on {} ({} structures found)",
        t.elapsed().as_secs_f64(),
        result.device,
        result.organs.len()
    );
    for n in &result.notes {
        eprintln!("note: {n}");
    }

    std::fs::write(format!("{out_prefix}.bin"), &result.labels)?;
    let mut j = std::fs::File::create(format!("{out_prefix}.json"))?;
    writeln!(j, "{{")?;
    writeln!(j, "  \"model\": \"{}\",", result.model)?;
    writeln!(
        j,
        "  \"dims\": [{}, {}, {}],",
        vol.dims[0], vol.dims[1], vol.dims[2]
    )?;
    writeln!(
        j,
        "  \"spacing\": [{}, {}, {}],",
        vol.spacing[0], vol.spacing[1], vol.spacing[2]
    )?;
    writeln!(
        j,
        "  \"origin\": [{}, {}, {}],",
        vol.origin.x, vol.origin.y, vol.origin.z
    )?;
    writeln!(j, "  \"elapsed_secs\": {},", result.elapsed_secs)?;
    writeln!(j, "  \"device\": \"{}\",", result.device)?;
    let classes: Vec<String> = result.classes.iter().map(|c| format!("\"{c}\"")).collect();
    writeln!(j, "  \"classes\": [{}],", classes.join(", "))?;
    writeln!(j, "  \"organs\": [")?;
    for (i, o) in result.organs.iter().enumerate() {
        writeln!(
            j,
            "    {{\"label\": {}, \"name\": \"{}\", \"voxels\": {}, \"cm3\": {:.2}}}{}",
            o.label,
            o.name,
            o.voxels,
            o.cm3,
            if i + 1 < result.organs.len() { "," } else { "" }
        )?;
    }
    writeln!(j, "  ]")?;
    writeln!(j, "}}")?;
    for o in result.organs.iter().take(25) {
        eprintln!("  {:3}  {:32} {:9.1} cm3", o.label, o.name, o.cm3);
    }
    Ok(())
}

fn usage() -> anyhow::Error {
    anyhow::anyhow!(
        "usage: seg_cli --list [--models ROOT] | seg_cli <dicom_dir> <out_prefix> [--model KEY] \
         [--models ROOT] [--device auto|gpu|cpu] [--parts a,b,...] [--nnunet-folder DIR]..."
    )
}
