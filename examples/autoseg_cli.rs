//! Headless auto-segmentation runner (development / validation tool).
//!
//! ```text
//! cargo run --release --example autoseg_cli -- <dicom_dir> <out_prefix>
//!     [--variant fast3|highres|preview6] [--models DIR] [--device auto|gpu|cpu]
//!     [--parts organs,vertebrae,cardiac,muscles,ribs]
//! ```
//!
//! `--models` is the engine's folder, `models/totalsegmentator/` next to the
//! executable by default.
//!
//! Writes `<out_prefix>.bin` (u8 labels, `Volume::data` order) and
//! `<out_prefix>.json` (dims, spacing, origin, orientation, organ table) so
//! external tools (e.g. a Python comparison against the reference
//! TotalSegmentator) can consume the result.

use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::autoseg::{self, DevicePref, Variant};
use rust_dicom_station::loader;
use rust_dicom_station::models::{self, Engine};
use rust_dicom_station::progress::Progress;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or_else(usage)?);
    let out_prefix = args.next().ok_or_else(usage)?;
    let mut variant = Variant::Fast3mm;
    let mut models_dir = models::engine_dir(&models::default_root(), Engine::TotalSegmentator);
    let mut device = DevicePref::Auto;
    let mut parts = [true; 5];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--variant" => {
                variant = match args.next().as_deref() {
                    Some("fast3") => Variant::Fast3mm,
                    Some("highres") => Variant::HighRes15mm,
                    Some("preview6") => Variant::Preview6mm,
                    v => anyhow::bail!("unknown variant {v:?}"),
                }
            }
            "--models" | "--models-dir" => {
                models_dir = PathBuf::from(args.next().ok_or_else(usage)?)
            }
            "--device" => {
                let v = args.next().unwrap_or_default();
                device = DevicePref::from_key(&v)
                    .ok_or_else(|| anyhow::anyhow!("unknown device {v:?}"))?;
            }
            "--parts" => {
                parts = [false; 5];
                for p in args.next().ok_or_else(usage)?.split(',') {
                    let idx = autoseg::classes::PART_NAMES
                        .iter()
                        .position(|n| *n == p)
                        .ok_or_else(|| anyhow::anyhow!("unknown part {p}"))?;
                    parts[idx] = true;
                }
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
    }

    let progress = Progress::default();
    eprintln!("loading {} …", dir.display());
    let study = loader::load_directory(&dir, &progress)?;
    let vol = &study.volume;
    eprintln!(
        "volume {}x{}x{} @ {:.3}/{:.3}/{:.3} mm",
        vol.dims[0], vol.dims[1], vol.dims[2], vol.spacing[0], vol.spacing[1], vol.spacing[2]
    );

    let ap = Progress::default();
    let t = std::time::Instant::now();
    let done = std::sync::atomic::AtomicBool::new(false);
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
        let r = autoseg::run(vol, variant, device, parts, &models_dir, &ap);
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = printer.join();
        r
    })?;
    eprintln!(
        "finished in {:.1}s on {} ({} organs found)",
        t.elapsed().as_secs_f64(),
        result.device,
        result.organs.len()
    );

    std::fs::write(format!("{out_prefix}.bin"), &result.labels)?;
    let mut j = std::fs::File::create(format!("{out_prefix}.json"))?;
    writeln!(j, "{{")?;
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
        "usage: autoseg_cli <dicom_dir> <out_prefix> [--variant fast3|highres|preview6] [--models DIR] [--device auto|gpu|cpu] [--parts a,b,…]"
    )
}
