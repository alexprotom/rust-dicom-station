//! Headless SegVol segmentation: point it at a DICOM folder and a prompt.
//!
//! ```text
//! cargo run --release --example segvol_cli -- <DICOM_DIR> \
//!     [--models DIR] [--box z0,y0,x0,z1,y1,x1] [--point z,y,x]... \
//!     [--no-zoom-in] [--fast-box] [--threshold F] [--out FILE]
//! ```
//!
//! Box and point coordinates are in the **prepared** grid — canonically
//! oriented `[S, A, R]` and cropped to the foreground — which is what the
//! network sees. `--out` writes a raw `u8` mask on the original volume's grid,
//! one byte per voxel in `Volume::data` order.

use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::loader;
use rust_dicom_station::nn::cache::{load_safetensors, ProgressSink};
use rust_dicom_station::segvol::infer::{self, Config, Hooks};
use rust_dicom_station::segvol::params::Params;
use rust_dicom_station::segvol::prompt::{BBox, Point};
use rust_dicom_station::segvol::{net::SegVolNet, preprocess, weights};

struct Stderr;
impl ProgressSink for Stderr {
    fn report(&self, _f: f32, m: &str) {
        eprint!("\r\x1b[K{m}");
        std::io::stderr().flush().ok();
    }
}
impl Hooks for Stderr {
    fn report(&self, _f: f32, m: &str) {
        eprint!("\r\x1b[K{m}");
        std::io::stderr().flush().ok();
    }
}

fn triple(s: &str) -> [f32; 3] {
    let v: Vec<f32> = s
        .split(',')
        .map(|x| x.trim().parse().expect("number"))
        .collect();
    assert_eq!(v.len(), 3, "expected three comma-separated numbers");
    [v[0], v[1], v[2]]
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut dicom: Option<PathBuf> = None;
    let mut models: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut boxes: Vec<BBox> = Vec::new();
    let mut points: Vec<Point> = Vec::new();
    let mut cfg = Config::default();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--models" => models = Some(PathBuf::from(args.next().unwrap())),
            "--out" => out = Some(PathBuf::from(args.next().unwrap())),
            "--box" => {
                let v: Vec<f32> = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|x| x.trim().parse().expect("number"))
                    .collect();
                assert_eq!(v.len(), 6, "--box needs six comma-separated numbers");
                boxes.push([v[0], v[1], v[2], v[3], v[4], v[5]]);
            }
            "--point" => points.push(Point::foreground(triple(&args.next().unwrap()))),
            "--negative-point" => points.push(Point::background(triple(&args.next().unwrap()))),
            "--no-zoom-in" => cfg.use_zoom_in = false,
            "--fast-box" => cfg.skip_coarse_with_box = true,
            "--threshold" => cfg.threshold = args.next().unwrap().parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "usage: segvol_cli <DICOM_DIR> [--models DIR] [--box z0,y0,x0,z1,y1,x1] \
                     [--point z,y,x] [--no-zoom-in] [--fast-box] [--threshold F] [--out FILE]"
                );
                return Ok(());
            }
            other => dicom = Some(PathBuf::from(other)),
        }
    }
    let dicom = dicom.expect("a DICOM directory is required");
    let models = models.unwrap_or_else(weights::default_models_dir);

    eprintln!("loading {}", dicom.display());
    let study = loader::load_directory(&dicom, &loader::Progress::default())?;
    let vol = &study.volume;
    eprintln!("volume {:?} spacing {:?}", vol.dims, vol.spacing);

    let prep = preprocess::prepare(vol);
    eprintln!(
        "prepared {:?} (oriented {:?}, crop at {:?})",
        prep.dims, prep.oriented_dims, prep.crop_lo
    );
    if boxes.is_empty() && points.is_empty() {
        // Default to the whole prepared extent, which is a legitimate prompt
        // and makes the tool useful with no arguments.
        let d = prep.dims;
        boxes.push([
            0.0,
            0.0,
            0.0,
            d[0] as f32 - 1.0,
            d[1] as f32 - 1.0,
            d[2] as f32 - 1.0,
        ]);
        eprintln!("no prompt given; using the whole prepared extent as a box");
    }

    // The converted-weight cache lives beside the checkpoint.
    let cache = models.join("segvol.safetensors");
    if !cache.is_file() {
        eprintln!("converting the checkpoint into {} …", cache.display());
        let path = weights::ensure_file(&weights::CHECKPOINT, &models, &Stderr)?;
        let mut reader = weights::open_checkpoint(&path)?;
        let metas: Vec<_> = reader
            .tensors
            .iter()
            .filter(|(n, _)| !rust_dicom_station::segvol::layout::is_dead_weight(n))
            .cloned()
            .collect();
        let mut named = Vec::with_capacity(metas.len());
        for (i, (name, meta)) in metas.iter().enumerate() {
            if !matches!(
                meta.dtype,
                rust_dicom_station::nn::pickle::Dtype::F32
                    | rust_dicom_station::nn::pickle::Dtype::F16
                    | rust_dicom_station::nn::pickle::Dtype::F64
            ) {
                continue; // CLIP's integer position_ids buffer
            }
            named.push((
                rust_dicom_station::segvol::layout::normalize_key(name).to_string(),
                meta.shape.clone(),
                reader.read_f32(meta)?,
            ));
            <Stderr as ProgressSink>::report(
                &Stderr,
                0.0,
                &format!("converting {}/{}", i + 1, metas.len()),
            );
        }
        std::fs::create_dir_all(&models)?;
        rust_dicom_station::nn::cache::save_safetensors(
            &cache,
            &named,
            rust_dicom_station::nn::cache::StoreDtype::F32,
        )?;
        eprintln!("\rwrote {}", cache.display());
    }

    eprintln!("loading weights …");
    let params = Params::new(load_safetensors(&cache)?);
    let net = SegVolNet::build(&params)?;
    eprintln!("network ready ({} tensors)", params.len());

    let t0 = std::time::Instant::now();
    let seg = infer::segment(&net, &prep, &points, &boxes, None, cfg, &Stderr)?;
    eprintln!(
        "\rsegmented in {:.1}s: {} voxels, {} refinement window(s), coarse pass {}",
        t0.elapsed().as_secs_f64(),
        seg.voxels,
        seg.windows,
        if seg.coarse { "ran" } else { "skipped" }
    );

    let full = prep.mask_to_volume_grid(&seg.mask, vol);
    let cm3 = full.iter().filter(|v| **v != 0).count() as f64
        * vol.spacing[0]
        * vol.spacing[1]
        * vol.spacing[2]
        / 1000.0;
    eprintln!("{cm3:.1} cm3 on the original grid");
    if let Some(p) = out {
        std::fs::write(&p, &full)?;
        eprintln!("wrote {} ({} bytes)", p.display(), full.len());
    }
    Ok(())
}
