//! Headless SegVol segmentation: point it at a DICOM folder and a prompt.
//!
//! ```text
//! cargo run --release --example segvol_cli -- <DICOM_DIR> \
//!     [--models DIR] [--device auto|gpu|cpu] [--box z0,y0,x0,z1,y1,x1] \
//!     [--point z,y,x]... [--negative-point z,y,x]... [--text STRUCTURE] \
//!     [--no-zoom-in] [--fast-box] [--threshold F] [--out FILE]
//! ```
//!
//! `--models` is the engine's folder, `segvol/` in the viewer's model folder
//! by default. Box and point coordinates are in the **prepared** grid -
//! canonically oriented `[S, A, R]` and cropped to the foreground - which is
//! what the network sees. `--out` writes a raw `u8` mask on the original
//! volume's grid, one byte per voxel in `Volume::data` order.

use std::path::PathBuf;

use rust_dicom_station::loader;
use rust_dicom_station::models::{self, Engine};
use rust_dicom_station::nn::device::DevicePref;
use rust_dicom_station::progress::{Progress, Stderr};
use rust_dicom_station::segvol::infer::{self, Config};
use rust_dicom_station::segvol::prompt::{BBox, Point};
use rust_dicom_station::segvol::{
    bpe::Bpe, clip::TextEncoder, net::SegVolNet, preprocess, weights,
};

mod common;

fn triple(s: &str) -> [f32; 3] {
    let v = common::numbers(s);
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
    let mut text_prompt: Option<String> = None;
    let mut device = DevicePref::Auto;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--models" => models = Some(PathBuf::from(args.next().unwrap())),
            "--device" => {
                device = DevicePref::from_key(&args.next().unwrap()).expect("auto, gpu or cpu")
            }
            "--out" => out = Some(PathBuf::from(args.next().unwrap())),
            "--box" => {
                let v = common::numbers(&args.next().unwrap());
                assert_eq!(v.len(), 6, "--box needs six comma-separated numbers");
                boxes.push([v[0], v[1], v[2], v[3], v[4], v[5]]);
            }
            "--point" => points.push(Point::foreground(triple(&args.next().unwrap()))),
            "--negative-point" => points.push(Point::background(triple(&args.next().unwrap()))),
            "--text" => text_prompt = Some(args.next().unwrap()),
            "--no-zoom-in" => cfg.use_zoom_in = false,
            "--fast-box" => cfg.skip_coarse_with_box = true,
            "--threshold" => cfg.threshold = args.next().unwrap().parse()?,
            "-h" | "--help" => {
                eprintln!(
                    "usage: segvol_cli <DICOM_DIR> [--models DIR] [--device auto|gpu|cpu] \
                     [--box z0,y0,x0,z1,y1,x1] [--point z,y,x] [--negative-point z,y,x] \
                     [--text STRUCTURE] [--no-zoom-in] [--fast-box] [--threshold F] \
                     [--out FILE]"
                );
                return Ok(());
            }
            other => dicom = Some(PathBuf::from(other)),
        }
    }
    let dicom = dicom.expect("a DICOM directory is required");
    let models =
        models.unwrap_or_else(|| models::engine_dir(&models::default_root(), Engine::SegVol));

    eprintln!("loading {}", dicom.display());
    let study = loader::load_directory(&dicom, &Progress::default())?;
    let vol = &study.volume;
    eprintln!("volume {:?} spacing {:?}", vol.dims, vol.spacing);

    let prep = preprocess::prepare(vol);
    eprintln!(
        "prepared {:?} (oriented {:?}, crop at {:?})",
        prep.dims, prep.oriented_dims, prep.crop_lo
    );
    if boxes.is_empty() && points.is_empty() && text_prompt.is_none() {
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

    // Download, convert and cache on first use; load the cache after that.
    let params = weights::load(&models, &Stderr)?;
    #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
    let mut net = SegVolNet::build(&params)?;
    let on = match device.resolve()? {
        #[cfg(feature = "gpu")]
        Some(ctx) => {
            net.attach_gpu(rust_dicom_station::segvol::gpu::GpuVit::new(&ctx, &params)?);
            ctx.describe()
        }
        #[cfg(not(feature = "gpu"))]
        Some(ctx) => ctx.unreachable(),
        None => rust_dicom_station::nn::device::describe_cpu(),
    };
    eprintln!(
        "\rnetwork ready ({} tensors), image encoder on {on}",
        params.len()
    );

    // A text prompt needs the tokenizer's two data files and the text tower.
    let text: Option<Vec<f32>> = match &text_prompt {
        None => None,
        Some(name) => {
            for f in &weights::CLIP_FILES {
                f.ensure(&models, &Stderr)?;
            }
            let bpe = Bpe::from_dir(&models)?;
            let enc = TextEncoder::build(&params)?;
            let v = enc.encode_structure(&bpe, name);
            eprintln!("text prompt {name:?} encoded");
            Some(v)
        }
    };

    let t0 = std::time::Instant::now();
    let seg = infer::segment(&net, &prep, &points, &boxes, text.as_deref(), cfg, &Stderr)?;
    eprintln!(
        "\rsegmented in {:.1}s: {} voxels, {} refinement window(s), coarse pass {}",
        t0.elapsed().as_secs_f64(),
        seg.voxels,
        seg.windows,
        if seg.coarse { "ran" } else { "skipped" }
    );

    let full = prep.mask_to_volume_grid(&seg.mask, vol);
    common::finish_mask(&full, vol, out.as_deref())
}
