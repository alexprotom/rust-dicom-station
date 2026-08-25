//! Headless MedSAM2 propagation: point it at a DICOM folder, a slice and a box.
//!
//! ```text
//! cargo run --release --example medsam2_cli -- <DICOM_DIR> \
//!     [--models DIR] [--variant latest|ct-lesion|mri-liver-lesion|base-2411] \
//!     [--slice N] [--box r0,c0,r1,c1] [--point r,c] \
//!     [--window LO,HI] [--preset NAME] [--range FIRST,LAST] [--max-slices N] \
//!     [--forward-only] \
//!     [--threshold F] [--no-cleanup] [--cpu] [--out FILE]
//! ```
//!
//! Slice, box and point coordinates are in the **prepared** stack — axial
//! slices in reading order, which for an ordinary head-first-supine CT is the
//! acquisition order. `--out` writes a raw `u8` mask on the original volume's
//! grid, one byte per voxel in `Volume::data` order.

use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::loader;
use rust_dicom_station::medsam2::engine::{Engine, EnginePrompt, PixelPrompt};
use rust_dicom_station::medsam2::infer::{Config, Hooks};
use rust_dicom_station::medsam2::preprocess::{Prepared, Window};
use rust_dicom_station::medsam2::weights::{self, Variant};
use rust_dicom_station::nn::cache::ProgressSink;

struct Stderr;

impl Stderr {
    fn line(msg: &str) {
        eprint!("\r\x1b[K{msg}");
        std::io::stderr().flush().ok();
    }
}

impl ProgressSink for Stderr {
    fn report(&self, _f: f32, m: &str) {
        Stderr::line(m);
    }
}

impl Hooks for Stderr {
    fn report(&self, _f: f32, m: &str) {
        Stderr::line(m);
    }
}

fn numbers(s: &str, n: usize, what: &str) -> Vec<f32> {
    let v: Vec<f32> = s
        .split(',')
        .map(|x| x.trim().parse().expect("number"))
        .collect();
    assert_eq!(v.len(), n, "{what} needs {n} comma-separated numbers");
    v
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut dicom: Option<PathBuf> = None;
    let mut models: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut variant = Variant::Latest;
    let mut slice: Option<usize> = None;
    let mut boxed: Option<Vec<f32>> = None;
    let mut point: Option<Vec<f32>> = None;
    let mut window: Option<Window> = None;
    let mut cfg = Config::default();
    let mut prefer_gpu = true;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--models" => models = Some(PathBuf::from(args.next().unwrap())),
            "--out" => out = Some(PathBuf::from(args.next().unwrap())),
            "--variant" => {
                variant = match args.next().unwrap_or_default().as_str() {
                    "latest" => Variant::Latest,
                    "ct-lesion" => Variant::CtLesion,
                    "mri-liver-lesion" => Variant::MriLiverLesion,
                    "base-2411" => Variant::Base2411,
                    other => anyhow::bail!("unknown variant {other}"),
                }
            }
            "--slice" => slice = Some(args.next().unwrap().parse()?),
            "--box" => boxed = Some(numbers(&args.next().unwrap(), 4, "--box")),
            "--point" => point = Some(numbers(&args.next().unwrap(), 2, "--point")),
            "--window" => {
                let v = numbers(&args.next().unwrap(), 2, "--window");
                window = Some(Window::new(v[0], v[1]));
            }
            "--preset" => {
                let name = args.next().unwrap();
                window = Some(
                    Window::preset(&name)
                        .ok_or_else(|| anyhow::anyhow!("unknown preset {name}"))?,
                );
            }
            "--max-slices" => cfg.max_slices = Some(args.next().unwrap().parse()?),
            "--range" => {
                let v = numbers(&args.next().unwrap(), 2, "--range");
                cfg.range = Some((v[0] as usize, v[1] as usize));
            }
            "--all-slices" => cfg.max_slices = None,
            "--forward-only" => cfg.reverse_pass = false,
            "--threshold" => cfg.threshold = args.next().unwrap().parse()?,
            "--no-cleanup" => cfg.largest_component = false,
            "--cpu" => prefer_gpu = false,
            "-h" | "--help" => {
                eprintln!(
                    "usage: medsam2_cli <DICOM_DIR> [--models DIR] [--variant NAME] \
                     [--slice N] [--box r0,c0,r1,c1] [--point r,c] [--window LO,HI] \
                     [--preset NAME] [--range FIRST,LAST] [--max-slices N] [--all-slices] \
                     [--forward-only] \
                     [--threshold F] [--no-cleanup] [--cpu] [--out FILE]"
                );
                return Ok(());
            }
            other => dicom = Some(PathBuf::from(other)),
        }
    }
    let dicom = dicom.expect("a DICOM directory is required");
    let models = models.unwrap_or_else(weights::default_models_dir);
    let window = window.unwrap_or_else(|| Window::preset("Abdomen").unwrap());

    eprintln!("loading {}", dicom.display());
    let study = loader::load_directory(&dicom, &loader::Progress::default())?;
    let vol = &study.volume;
    eprintln!("volume {:?} spacing {:?}", vol.dims, vol.spacing);

    let prepared = Prepared::prepare(vol, window);
    eprintln!(
        "prepared {:?} at {:?} mm, window [{}, {}]",
        prepared.dims, prepared.spacing, window.lower, window.upper
    );

    let slice = slice.unwrap_or(prepared.dims[0] / 2);
    anyhow::ensure!(
        slice < prepared.dims[0],
        "slice {slice} is outside a stack of {}",
        prepared.dims[0]
    );
    let (rows, cols) = (prepared.dims[1] as f32, prepared.dims[2] as f32);
    let prompt = match (&boxed, &point) {
        (Some(b), _) => EnginePrompt::Points(PixelPrompt::box_corners(b[0], b[1], b[2], b[3])),
        (None, Some(p)) => EnginePrompt::Points(vec![PixelPrompt::positive(p[0], p[1])]),
        (None, None) => {
            // A box over the middle half of the slice: a legitimate prompt,
            // and it makes the tool useful with no arguments.
            eprintln!("no prompt given; boxing the middle half of slice {slice}");
            EnginePrompt::Points(PixelPrompt::box_corners(
                rows * 0.25,
                cols * 0.25,
                rows * 0.75,
                cols * 0.75,
            ))
        }
    };

    eprintln!("loading {} …", variant.file().name);
    let params = weights::load(variant, &models, &Stderr)?;
    let engine = Engine::load(&params, prefer_gpu)?;
    eprintln!(
        "\rnetwork ready on {} ({} tensors)",
        engine.device(),
        params.len()
    );

    let t0 = std::time::Instant::now();
    let (mask, seg) = engine.propagate_to_volume(&prepared, vol, slice, &prompt, &cfg, &Stderr)?;
    let span = match seg.extent() {
        Some((a, b)) => format!("slices {a}..={b}"),
        None => "nothing".to_string(),
    };
    eprintln!(
        "\rpropagated in {:.1}s: {} voxels over {span}, {} slice(s) tracked",
        t0.elapsed().as_secs_f64(),
        seg.voxels,
        seg.slices_visited
    );
    let cm3 = seg.voxels as f64 * vol.spacing[0] * vol.spacing[1] * vol.spacing[2] / 1000.0;
    eprintln!("{cm3:.2} cm3");

    if let Some(p) = out {
        std::fs::write(&p, &mask)?;
        eprintln!("wrote {} ({} bytes)", p.display(), mask.len());
    }
    Ok(())
}
