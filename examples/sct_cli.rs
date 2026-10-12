//! Headless synthetic CT (development / validation tool).
//!
//! ```text
//! cargo run --release --example sct_cli -- --list [--models ROOT]
//! cargo run --release --example sct_cli -- <dicom_dir> [--series N] [--app KEY]
//!     [--cv all|0,1,..] [--tta 0|1|2] [--mask network|classical|none|<structure>]
//!     [--uncertainty] [--device auto|gpu|cpu] [--models ROOT] [--out DIR]
//!     [--synth-folder DIR]... [--reference <ct_dir>] [--wet] [--structures]
//! ```
//!
//! `--app` is `sct_mr` (MR of any sequence), `sct_cbct`, `sct_mr_cbct`,
//! `sct_mr_dixon`, or the key of a checkpoint folder (the viewer's model
//! manager adds them; `--synth-folder` adds one for this run). `--cv all
//! --tta 2` is the published recipe; the default is the quick look (CV_0,
//! no mirrors). `--series` is the series' place in the folder (1-based, as
//! the data tree lists them); the largest MR or CT by default.
//!
//! The sCT is written as a CT series of the source's study under `--out`
//! (`<data folder>/derived` by default). With `--reference` (a folder whose
//! largest CT shares the source's frame of reference), it is evaluated and
//! the numbers printed as CSV.

use std::path::PathBuf;

use rust_dicom_station::loader;
use rust_dicom_station::models;
use rust_dicom_station::nn::device::DevicePref;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::registration::resample_through;
use rust_dicom_station::sct::{self, eval, BodySource, SctRequest};
use rust_dicom_station::synth::{self, App};

fn main() {
    if let Err(e) = run(std::env::args().skip(1).collect()) {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn usage() -> anyhow::Error {
    anyhow::anyhow!(
        "usage: sct_cli --list [--models ROOT] | sct_cli <dicom_dir> [--series N] [--app KEY] \
         [--cv all|0,1,..] [--tta 0|1|2] [--mask network|classical|none|<structure>] \
         [--uncertainty] [--device auto|gpu|cpu] [--models ROOT] [--out DIR] \
         [--synth-folder DIR]... [--reference <ct_dir>] [--wet] [--structures]"
    )
}

/// The largest series of a modality, by slice count.
fn largest(study: &loader::LoadedStudy, modality: &str) -> Option<usize> {
    study
        .series
        .iter()
        .enumerate()
        .filter(|(_, s)| s.modality.eq_ignore_ascii_case(modality))
        .max_by_key(|(_, s)| s.files.len())
        .map(|(i, _)| i)
}

fn run(argv: Vec<String>) -> anyhow::Result<()> {
    let mut args = argv.into_iter();
    let mut positional: Vec<String> = Vec::new();
    let mut root = models::default_root();
    let mut out_root = sct::default_out_root();
    let mut app_key: Option<String> = None;
    let mut series: Option<usize> = None;
    let mut cv = String::from("0");
    let mut tta = 0usize;
    let mut mask = String::from("network");
    let mut uncertainty = false;
    let mut device = DevicePref::Auto;
    let mut folders: Vec<PathBuf> = Vec::new();
    let mut reference: Option<PathBuf> = None;
    let mut wet = false;
    let mut structures = false;
    let mut list = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--list" => list = true,
            "--app" => app_key = Some(args.next().ok_or_else(usage)?),
            "--series" => series = Some(args.next().ok_or_else(usage)?.parse()?),
            "--cv" => cv = args.next().ok_or_else(usage)?,
            "--tta" => tta = args.next().ok_or_else(usage)?.parse()?,
            "--mask" => mask = args.next().ok_or_else(usage)?,
            "--uncertainty" => uncertainty = true,
            "--device" => {
                let v = args.next().unwrap_or_default();
                device = DevicePref::from_key(&v)
                    .ok_or_else(|| anyhow::anyhow!("unknown device {v:?}"))?;
            }
            "--models" | "--models-dir" => root = PathBuf::from(args.next().ok_or_else(usage)?),
            "--out" => out_root = PathBuf::from(args.next().ok_or_else(usage)?),
            "--synth-folder" => folders.push(PathBuf::from(args.next().ok_or_else(usage)?)),
            "--reference" => reference = Some(PathBuf::from(args.next().ok_or_else(usage)?)),
            "--wet" => wet = true,
            "--structures" => structures = true,
            other if !other.starts_with("--") => positional.push(other.to_string()),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let mut prefs = rust_dicom_station::settings::load();
    prefs.synth_folders.extend(folders);
    for (dir, why) in models::apply_settings(&prefs) {
        eprintln!("model folder {} not used: {why}", dir.display());
    }
    if list {
        println!(
            "{:28} {:8} {:16} {:>10}  label",
            "key", "from", "licence", "download"
        );
        for a in App::all() {
            let all: Vec<usize> = (0..a.checkpoint_count()).collect();
            let need = a.download_needed(&all, &root);
            println!(
                "{:28} {:8} {:16} {:>10}  {}",
                a.key(),
                a.modality().label(),
                a.licence().name(),
                if need == 0 {
                    "ready".to_string()
                } else {
                    models::human_bytes(need)
                },
                a.label()
            );
        }
        return Ok(());
    }
    if positional.len() != 1 {
        return Err(usage());
    }
    let dir = PathBuf::from(&positional[0]);
    let progress = Progress::default();
    eprintln!("loading {}", dir.display());
    let mut study = loader::load_directory(&dir, &progress)?;
    let idx = match series {
        Some(n) => n
            .checked_sub(1)
            .filter(|&i| i < study.series.len())
            .ok_or_else(|| anyhow::anyhow!("the folder has {} series", study.series.len()))?,
        None => largest(&study, "MR")
            .or_else(|| largest(&study, "CT"))
            .ok_or_else(|| anyhow::anyhow!("no MR or CT series in the folder"))?,
    };
    if idx != study.active_series {
        let (v, w, _) = loader::load_series_volume(&study.series[idx], &progress)?;
        study.volume = std::sync::Arc::new(v);
        study.default_window = w;
        study.active_series = idx;
    }
    let source = study.series[idx].clone();
    let app = match &app_key {
        Some(k) => App::from_key(k)
            .ok_or_else(|| anyhow::anyhow!("unknown app {k:?}; --list prints them"))?,
        None if source.modality.eq_ignore_ascii_case("MR") => App::Mr,
        None => App::Cbct,
    };
    let checkpoints: Vec<usize> = if cv.trim() == "all" {
        (0..app.checkpoint_count()).collect()
    } else {
        cv.split(',')
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<_, _>>()?
    };
    let body = match BodySource::from_key(&mask) {
        Some(b) => b,
        None => {
            let grid = study.volume.grid();
            let roi = study
                .structure_sets
                .iter()
                .flat_map(|s| s.rois.iter())
                .find(|r| r.name.eq_ignore_ascii_case(mask.trim()))
                .ok_or_else(|| anyhow::anyhow!("no structure {mask:?} in the folder"))?;
            BodySource::Structure {
                name: roi.name.clone(),
                mask: rust_dicom_station::segmentation::rasterize_roi(&grid, roi)
                    .ok_or_else(|| anyhow::anyhow!("structure {mask:?} is not on the series"))?,
            }
        }
    };
    let vol = study.volume.clone();
    eprintln!(
        "series {} ({} {}): {}x{}x{} @ {:.3}/{:.3}/{:.3} mm; {} ({}), checkpoints {:?}, \
         {} mirrors, {} passes a member",
        idx + 1,
        source.modality,
        source.description,
        vol.dims[0],
        vol.dims[1],
        vol.dims[2],
        vol.spacing[0],
        vol.spacing[1],
        vol.spacing[2],
        app.key(),
        app.label(),
        checkpoints,
        tta,
        synth::passes(&vol)
    );
    let req = SctRequest {
        app,
        opts: synth::Options {
            checkpoints,
            tta,
            uncertainty,
            device,
        },
        body,
        series_number: sct::next_series_number(&study),
    };
    let t = std::time::Instant::now();
    let done = std::sync::atomic::AtomicBool::new(false);
    let o = std::thread::scope(|s| {
        let (done_ref, p) = (&done, &progress);
        let printer = s.spawn(move || {
            let mut last = String::new();
            while !done_ref.load(std::sync::atomic::Ordering::Relaxed) {
                let msg = p.get();
                if msg != last {
                    eprintln!(
                        "[{:6.1}s] {:5.1}% {}",
                        t.elapsed().as_secs_f64(),
                        p.frac() * 100.0,
                        msg
                    );
                    last = msg;
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });
        let r = sct::synthesize(&vol, &source, &req, &root, &out_root, &progress);
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = printer.join();
        r
    })?;
    println!("{}", sct::summary(&o));
    println!(
        "clip [{}, {:.1}], working grid {:?}",
        o.clip.0, o.clip.1, o.working_dims
    );
    if let Some(f) = o.series.files.first().and_then(|f| f.parent()) {
        println!("written to {}", f.display());
    }
    if let Some(ref_dir) = reference {
        let r = loader::load_directory(&ref_dir, &progress)?;
        let ri =
            largest(&r, "CT").ok_or_else(|| anyhow::anyhow!("no CT in {}", ref_dir.display()))?;
        let rv = if ri == r.active_series {
            r.volume.clone()
        } else {
            std::sync::Arc::new(loader::load_series_volume(&r.series[ri], &progress)?.0)
        };
        if rv.frame_of_reference_uid != o.volume.frame_of_reference_uid {
            anyhow::bail!(
                "the reference is on another frame of reference; register the two in the viewer"
            );
        }
        let reference = resample_through(&rv, &o.volume.grid(), None);
        let ct: Vec<i16> = reference
            .iter()
            .map(|&v| if v.is_nan() { -1024 } else { v.round() as i16 })
            .collect();
        let b = rust_dicom_station::bodymask::contour_body(
            &sct::on_grid(&o.volume, ct),
            &rust_dicom_station::bodymask::BodyParams::default(),
            &root,
            &progress,
        )?;
        let e = eval::evaluate(
            &o.volume,
            &reference,
            Some(&b.mask),
            &eval::EvalOptions {
                wet: wet.then(eval::Hlut::default),
                structures,
                device,
            },
            &root,
            &progress,
        )?;
        print!("{}", eval::csv(&e));
    }
    Ok(())
}
