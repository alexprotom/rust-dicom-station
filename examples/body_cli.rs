//! Headless body / EXTERNAL contouring: point it at a DICOM folder.
//!
//! ```text
//! cargo run --release --example body_cli -- <DICOM_DIR> \
//!     [--method classical|model] [--model ct6|ct15|mr] \
//!     [--hu -300] [--mr-fraction 0.12] [--mr-otsu] [--bias-sigma 40] \
//!     [--open 8] [--no-devices] [--window 150] [--frac 0.8] \
//!     [--min-cm3 50] [--no-thin] [--thin-extent 100] [--margin 6] \
//!     [--thin-shell 3] [--no-fill] [--close 0] \
//!     [--models DIR] [--device auto|gpu|cpu] [--out FILE]
//! ```
//!
//! `--out` writes a raw `u8` mask on the original volume's grid, one byte
//! per voxel in `Volume::data` order — the same convention as the other
//! example tools, so the masks can be compared byte for byte.
//!
//! Batch use, which is what this is for: run it over a folder of upright
//! chair scans and check the reported equipment volume. A run that removes
//! nothing is a run whose threshold or opening radius is wrong.

use std::path::PathBuf;

use rust_dicom_station::bodymask::{self, BodyModel, BodyParams, Foreground, Method};
use rust_dicom_station::loader;
use rust_dicom_station::models::{self, Engine};
use rust_dicom_station::nn::device::DevicePref;
use rust_dicom_station::progress::Progress;

mod common;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut dicom: Option<PathBuf> = None;
    let mut models_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut p = BodyParams::default();
    // Set by --hu / --mr-*; otherwise the series' modality decides.
    let mut foreground: Option<Foreground> = None;
    let mut bias_sigma = 40.0f64;
    let mut mr_otsu = false;
    let mut mr_fraction: Option<f32> = None;
    let mut model_forced = false;
    while let Some(a) = args.next() {
        let mut next = || args.next().expect("missing value");
        match a.as_str() {
            "--method" => {
                p.method = match next().as_str() {
                    "classical" => Method::Classical,
                    "model" => Method::ModelAssisted,
                    other => panic!("--method classical|model, not {other:?}"),
                }
            }
            "--model" => {
                model_forced = true;
                p.model = match next().as_str() {
                    "ct6" => BodyModel::Ct6mm,
                    "ct15" => BodyModel::Ct15mm,
                    "mr" => BodyModel::Mr,
                    other => panic!("--model ct6|ct15|mr, not {other:?}"),
                }
            }
            "--hu" => foreground = Some(Foreground::Hu(next().parse().expect("number"))),
            "--mr-fraction" => mr_fraction = Some(next().parse().expect("number")),
            "--mr-otsu" => mr_otsu = true,
            "--bias-sigma" => bias_sigma = next().parse().expect("number"),
            "--open" => p.open_mm = next().parse().expect("number"),
            "--no-devices" => p.remove_devices = false,
            "--window" => p.persist_window_mm = next().parse().expect("number"),
            "--frac" => p.persist_frac = next().parse().expect("number"),
            "--min-cm3" => p.min_volume_cm3 = next().parse().expect("number"),
            "--no-thin" => p.recover_thin = false,
            "--thin-extent" => p.thin_max_extent_mm = next().parse().expect("number"),
            "--margin" => p.guide_margin_mm = next().parse().expect("number"),
            "--no-fill" => p.fill_interior = false,
            "--thin-shell" => p.device_thin_mm = next().parse().expect("number"),
            "--close" => p.close_mm = next().parse().expect("number"),
            "--models" => models_dir = Some(PathBuf::from(next())),
            "--device" => {
                p.device = DevicePref::from_key(&next()).expect("auto, gpu or cpu");
            }
            "--out" => out = Some(PathBuf::from(next())),
            other if dicom.is_none() => dicom = Some(PathBuf::from(other)),
            other => panic!("unexpected argument {other:?}"),
        }
    }
    let dicom = dicom.expect("usage: body_cli <DICOM_DIR> [options]");
    let models_dir = models_dir
        .unwrap_or_else(|| models::engine_dir(&models::default_root(), Engine::TotalSegmentator));

    let progress = Progress::default();
    let study = loader::load_directory(&dicom, &progress)?;
    let modality = study
        .series
        .get(study.active_series)
        .map(|s| s.modality.to_uppercase())
        .unwrap_or_default();
    eprintln!(
        "{} {:?} {} × {} × {} at {:.2} × {:.2} × {:.2} mm",
        modality,
        study.meta.patient_id,
        study.volume.dims[0],
        study.volume.dims[1],
        study.volume.dims[2],
        study.volume.spacing[0],
        study.volume.spacing[1],
        study.volume.spacing[2],
    );

    // Modality-appropriate defaults, overridden by whatever was asked for.
    // `--bias-sigma` on its own is a modifier, not a mode: it has to survive
    // this line rather than be overwritten by it.
    p.foreground = match Foreground::for_modality(&modality) {
        Foreground::MrRelative { fraction, .. } => Foreground::MrRelative {
            fraction,
            sigma_mm: bias_sigma,
        },
        other => other,
    };
    if !model_forced {
        p.model = BodyModel::for_modality(&modality);
    }
    if let Some(f) = foreground {
        p.foreground = f;
    } else if mr_otsu {
        p.foreground = Foreground::MrOtsu {
            sigma_mm: bias_sigma,
        };
    } else if let Some(fraction) = mr_fraction {
        p.foreground = Foreground::MrRelative {
            fraction,
            sigma_mm: bias_sigma,
        };
    }
    eprintln!("method {:?}, foreground {:?}", p.method, p.foreground);

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
        let r = bodymask::contour_body(&study.volume, &p, &models_dir, &ap);
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        let _ = printer.join();
        r
    });
    let result = result?;
    eprintln!(
        "{:.0} cm3 body in {:.1} s{}; {} piece(s); removed {} voxels of equipment, \
         recovered {} thin voxels",
        result.cm3,
        result.elapsed_secs,
        if result.device.is_empty() {
            String::new()
        } else {
            format!(" on {}", result.device)
        },
        result.pieces.len(),
        result.removed_voxels,
        result.recovered_voxels,
    );
    common::finish_mask(&result.mask, &study.volume, out.as_deref())
}
