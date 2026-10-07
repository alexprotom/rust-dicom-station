//! Headless interactive segmentation: prompts from the command line,
//! answered in order by nnInteractive or VISTA-3D's point mode
//! (development / validation tool).
//!
//! ```text
//! cargo run --release --example interactive_cli -- <dicom_dir> <out_prefix>
//!     [--engine nninteractive|vista3d] [--models ROOT] [--device auto|gpu|cpu]
//!     [--no-autozoom]
//!     --point X,Y,Z[,+|-] ... [--box X0,Y0,Z0,X1,Y1,Z1[,+|-]] ...
//! ```
//!
//! Positions are voxel indices of the loaded volume (`x` the column, `y`
//! the row, `z` the slice, as the viewer's crosshair shows them); `+`
//! (the default) marks the structure, `-` what must stay out. A box is
//! inclusive and has to be one voxel thick along one axis (nnInteractive
//! only). The prompts are given one at a time, each followed by a
//! prediction, as clicks in the viewer are.
//!
//! Writes `<out_prefix>.bin` (u8 mask, `Volume::data` order) and
//! `<out_prefix>.json`.

use std::io::Write;
use std::path::PathBuf;

use rust_dicom_station::loader;
use rust_dicom_station::models;
use rust_dicom_station::nn::device::DevicePref;
use rust_dicom_station::nninteractive::{Model, VolumeSession};
use rust_dicom_station::progress::Progress;
use rust_dicom_station::vista3d::session::{PointModel, PointSession};

enum Prompt {
    Point([f64; 3], bool),
    Box([f64; 3], [f64; 3], bool),
}

fn parse(v: &str, n: usize) -> anyhow::Result<(Vec<f64>, bool)> {
    let mut parts: Vec<&str> = v.split(',').map(str::trim).collect();
    let include = match parts.last() {
        Some(&"+") => {
            parts.pop();
            true
        }
        Some(&"-") => {
            parts.pop();
            false
        }
        _ => true,
    };
    let nums = parts
        .iter()
        .map(|p| p.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if nums.len() != n {
        anyhow::bail!("expected {n} numbers in {v:?}");
    }
    Ok((nums, include))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut positional = Vec::new();
    let mut vista = false;
    let mut root = models::default_root();
    let mut device = DevicePref::Auto;
    let mut autozoom = true;
    let mut prompts = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--engine" => match args.next().as_deref() {
                Some("nninteractive") => vista = false,
                Some("vista3d") | Some("vista3d_points") => vista = true,
                other => anyhow::bail!("unknown engine {other:?}"),
            },
            "--models" => root = PathBuf::from(args.next().ok_or_else(usage)?),
            "--device" => {
                let v = args.next().unwrap_or_default();
                device = DevicePref::from_key(&v)
                    .ok_or_else(|| anyhow::anyhow!("unknown device {v:?}"))?;
            }
            "--no-autozoom" => autozoom = false,
            "--point" => {
                let (n, inc) = parse(&args.next().ok_or_else(usage)?, 3)?;
                prompts.push(Prompt::Point([n[0], n[1], n[2]], inc));
            }
            "--box" => {
                let (n, inc) = parse(&args.next().ok_or_else(usage)?, 6)?;
                prompts.push(Prompt::Box([n[0], n[1], n[2]], [n[3], n[4], n[5]], inc));
            }
            other if !other.starts_with("--") => positional.push(other.to_string()),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    if positional.len() != 2 || prompts.is_empty() {
        return Err(usage());
    }
    let progress = Progress::default();
    eprintln!("loading {} ...", positional[0]);
    let study = loader::load_directory(std::path::Path::new(&positional[0]), &progress)?;
    let vol = &study.volume;
    let t = std::time::Instant::now();
    let (mask, device_desc) = if vista {
        let model = PointModel::load(&root, device, &progress)?;
        eprintln!("VISTA-3D points on {}", model.device);
        let mut s = PointSession::new(vol);
        let mut mask = Vec::new();
        for p in &prompts {
            let Prompt::Point(at, inc) = p else {
                anyhow::bail!("VISTA-3D takes points only");
            };
            s.add(*at, *inc)?;
            mask = s.segment(&model, vol, &progress)?;
            eprintln!(
                "[{:6.1}s] {} click(s): {} voxels",
                t.elapsed().as_secs_f64(),
                s.n_clicks(),
                mask.iter().filter(|&&v| v != 0).count()
            );
        }
        (mask, model.device.clone())
    } else {
        let model = Model::load(&root, device, &progress)?;
        eprintln!("nnInteractive on {}", model.device);
        let mut vs = VolumeSession::new(vol, model.settings.clone())?;
        vs.session.settings.autozoom = autozoom;
        for p in &prompts {
            match p {
                Prompt::Point(at, inc) => vs.session.add_point(vs.to_session(*at), *inc)?,
                Prompt::Box(a, b, inc) => {
                    // Inclusive voxel corners → a half-open box in the
                    // session's axes.
                    let sa = vs.to_session(*a);
                    let sb = vs.to_session(*b);
                    let lo: [f64; 3] = std::array::from_fn(|i| sa[i].min(sb[i]).round());
                    let hi: [f64; 3] = std::array::from_fn(|i| sa[i].max(sb[i]).round() + 1.0);
                    vs.session.add_box(lo, hi, *inc)?;
                }
            }
            let out = vs.session.predict(&model, false, &progress)?;
            eprintln!(
                "[{:6.1}s] prompt {}: {} network pass(es), zoom x{:.2}, {} voxels",
                t.elapsed().as_secs_f64(),
                vs.session.n_prompts(),
                out.passes,
                out.zoom,
                vs.session.mask.iter().filter(|&&v| v != 0).count()
            );
        }
        (vs.mask_on_volume(), model.device.clone())
    };
    let voxels = mask.iter().filter(|&&v| v != 0).count();
    let prefix = &positional[1];
    std::fs::write(format!("{prefix}.bin"), &mask)?;
    let mut j = std::fs::File::create(format!("{prefix}.json"))?;
    writeln!(j, "{{")?;
    writeln!(
        j,
        "  \"engine\": \"{}\",",
        if vista {
            "vista3d_points"
        } else {
            "nninteractive"
        }
    )?;
    writeln!(
        j,
        "  \"dims\": [{}, {}, {}],",
        vol.dims[0], vol.dims[1], vol.dims[2]
    )?;
    writeln!(j, "  \"device\": \"{device_desc}\",")?;
    writeln!(j, "  \"elapsed_secs\": {},", t.elapsed().as_secs_f64())?;
    writeln!(j, "  \"voxels\": {voxels}")?;
    writeln!(j, "}}")?;
    eprintln!(
        "{voxels} voxels ({:.2} cm3) in {:.1}s",
        voxels as f64 * vol.spacing.iter().product::<f64>() / 1000.0,
        t.elapsed().as_secs_f64()
    );
    Ok(())
}

fn usage() -> anyhow::Error {
    anyhow::anyhow!(
        "usage: interactive_cli <dicom_dir> <out_prefix> [--engine nninteractive|vista3d] \
         [--models ROOT] [--device auto|gpu|cpu] [--no-autozoom] --point X,Y,Z[,+|-] ... \
         [--box X0,Y0,Z0,X1,Y1,Z1[,+|-]] ..."
    )
}
