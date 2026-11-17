//! The posture model, headless ([`rust_dicom_station::workflow::posture`]):
//! two or more studies in, the tables and the key frames out.
//!
//! ```text
//! cargo run --release --example posture_cli -- <STATE_A> <STATE_B> [<STATE_C> ...] [options]
//! cargo run --release --example posture_cli -- --phantom [--angle DEG] [options]
//! cargo run --release --example posture_cli -- <CT_FOLDER> --simulate tx,ty,tz,rx,ry,rz [options]
//! ```
//!
//! Each state is a folder the viewer would open (its displayed series and
//! the structure set drawn on it), or `SERIES_DIR::STRUCTURE_SET_FILE` - one
//! series' folder and one structure set beside it, as an archive files
//! them, read where they are. `--phantom` builds the articulated
//! phantom of the tests (`gen_test_data::articulated_pair`: two bones on a
//! hinge turned by `--angle`, default 40) and prints its ground truth
//! beside the model's answer. `--simulate` is the self-test on any CT: the
//! second state is the first moved by the simulator's known rigid motion
//! (mm, degrees), so every body should read that motion back.
//!
//! Options:
//!
//! * `--anchor NAME|image|scanner` - what is held still (default: the
//!   phantom's `BONE_UPPER`, otherwise `image`);
//! * `--bodies bones|none|A,B,C` - the rigid bodies (default `bones`);
//! * `--mode diffeomorphic|linear`;
//! * `--lattice MM` (6), `--analysis MM` (2), `--reach MM` (40);
//! * `--residual` - add the B-spline residual layer;
//! * `--posture P1,P2,...` - a posture parameter per state;
//! * `--joint NAME:FIRST:SECOND` - a joint to report (repeatable);
//! * `--out DIR` - write `tables.csv`, `model.rdsposture`, the deformation
//!   as `posture_reg.dcm`, and with `--frames N` (default 0) N key frames
//!   as PNGs of the three planes through the centre (`frames/`);
//! * `--show forward|backward|crossfade|difference|jacobian` - what the key
//!   frames and the GIF show;
//! * `--gif FILE` - the path as an animated GIF (`posture::film`, on the
//!   CPU): `--gif-view all|axial|sagittal|coronal` (default `all`, the
//!   three planes through the centre side by side), `--gif-width PX`
//!   (800), `--gif-frames N` (21), `--fps F` (6), `--bounce` (there and
//!   back), `--wash` (the Jacobian wash); the structures of the first
//!   state outlined;
//! * `--export T` - the image and structures at time `T` as DICOM
//!   (`frame_T/`);
//! * `--wepl GANTRY[,COUCH]` - the WEPL to each structure's centroid along
//!   a beam at these IEC 61217 angles (degrees, in the first state's
//!   patient position): first, last and its change;
//! * `--no-smooth` - a chain without curves through its states (a constant
//!   turn per pair);
//! * `--bench-slices N` - N live steps of the three central planes, on the
//!   CPU and (with a GPU) on the GPU: the time per step and how far the two
//!   answers are apart.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use rust_dicom_station::gen_test_data::{self, ArticulatedParams};
use rust_dicom_station::loader::LoadedStudy;
use rust_dicom_station::posture::{apply, file, film, Anchor, Mode};
use rust_dicom_station::progress::Progress;
use rust_dicom_station::simulate::{generate_transformed_study, SimParams};
use rust_dicom_station::volume::{ViewPlane, Volume};
use rust_dicom_station::workflow::posture::{self as wp, Bodies, PostureRequest, StateInput};

/// Live steps of the three central planes, timed on the CPU and on the
/// GPU, and the largest difference between the two.
fn bench_slices(
    model: &rust_dicom_station::posture::PostureModel,
    studies: &[LoadedStudy],
    n: usize,
) -> Result<()> {
    use rust_dicom_station::posture::gpu::{render_planes_with, GpuSlicer};
    let images: Vec<Arc<Volume>> = studies.iter().map(|s| s.volume.clone()).collect();
    let g = &model.grid;
    let planes: Vec<apply::Plane> = (0..3)
        .map(|axis| apply::Plane {
            axis,
            index: g.dims[axis] / 2,
        })
        .collect();
    let slicer = match rust_dicom_station::nn::device::DevicePref::Auto.resolve()? {
        Some(ctx) => match GpuSlicer::new(&ctx, &images) {
            Ok(s) => {
                println!("slicer: {}", s.describe());
                Some(s)
            }
            Err(e) => {
                println!("no GPU slicer: {e:#}");
                None
            }
        },
        None => {
            println!("no GPU: the CPU only");
            None
        }
    };
    let mut worst: f32 = 0.0;
    let (mut t_frame, mut t_cpu, mut t_gpu) = (0.0, 0.0, 0.0);
    for k in 0..n {
        let t = model.span() * (k as f64 + 0.5) / n as f64;
        let a = std::time::Instant::now();
        let frame = model.frame(t);
        t_frame += a.elapsed().as_secs_f64();
        let a = std::time::Instant::now();
        let (cpu, _, _) = render_planes_with(
            None,
            model,
            &images,
            &frame,
            apply::Show::CrossFade,
            g,
            &planes,
        )?;
        t_cpu += a.elapsed().as_secs_f64();
        if let Some(s) = &slicer {
            let a = std::time::Instant::now();
            let gpu = s.render_planes(model, &frame, apply::Show::CrossFade, g, &planes)?;
            t_gpu += a.elapsed().as_secs_f64();
            for (x, y) in cpu.iter().zip(&gpu) {
                for (p, q) in x.iter().zip(y) {
                    worst = worst.max((p - q).abs());
                }
            }
        }
    }
    let ms = |s: f64| 1000.0 * s / n as f64;
    println!(
        "per step: the path {:.1} ms, three planes on the CPU {:.1} ms{}",
        ms(t_frame),
        ms(t_cpu),
        if slicer.is_some() {
            format!(
                ", on the GPU {:.1} ms; largest difference {worst:.3}",
                ms(t_gpu)
            )
        } else {
            String::new()
        }
    );
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    const VALUED: [&str; 21] = [
        "--wepl",
        "--gif",
        "--gif-view",
        "--gif-width",
        "--gif-frames",
        "--fps",
        "--bench-slices",
        "--anchor",
        "--bodies",
        "--mode",
        "--lattice",
        "--analysis",
        "--reach",
        "--posture",
        "--joint",
        "--out",
        "--frames",
        "--show",
        "--export",
        "--angle",
        "--simulate",
    ];
    let opt = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let opts_all = |name: &str| -> Vec<String> {
        args.iter()
            .enumerate()
            .filter(|(_, a)| *a == name)
            .filter_map(|(i, _)| args.get(i + 1).cloned())
            .collect()
    };
    let has = |name: &str| args.iter().any(|a| a == name);
    let num = |name: &str, d: f64| -> Result<f64> {
        opt(name)
            .map(|v| v.parse::<f64>().map_err(|e| anyhow!("{name} {v}: {e}")))
            .transpose()
            .map(|v| v.unwrap_or(d))
    };
    let mut folders = Vec::new();
    let mut k = 0;
    while k < args.len() {
        if VALUED.contains(&args[k].as_str()) {
            k += 2;
            continue;
        }
        if !args[k].starts_with("--") {
            folders.push(PathBuf::from(&args[k]));
        }
        k += 1;
    }

    // ---- the states ----
    let phantom = has("--phantom");
    let truth = ArticulatedParams {
        angle_deg: num("--angle", 40.0)?,
        ..ArticulatedParams::default()
    };
    let studies: Vec<LoadedStudy> = if phantom {
        gen_test_data::articulated_pair(&truth).to_vec()
    } else if let Some(sim) = opt("--simulate") {
        let v: Vec<f64> = sim
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<std::result::Result<_, _>>()
            .map_err(|e| anyhow!("--simulate {sim}: {e}"))?;
        if v.len() != 6 {
            bail!("--simulate takes tx,ty,tz,rx,ry,rz");
        }
        let first = folders.first().context("--simulate needs a CT folder")?;
        let a = load(first)?;
        let params = SimParams {
            translation: [v[0], v[1], v[2]],
            rotation_deg: [v[3], v[4], v[5]],
            ..SimParams::default()
        };
        let b = generate_transformed_study(&a, &params, &Progress::default());
        eprintln!("simulated: {}", params.describe());
        vec![a, b]
    } else {
        if folders.len() < 2 {
            bail!(
                "usage: posture_cli <STATE_A> <STATE_B> [...] | --phantom | <CT> --simulate tx,ty,tz,rx,ry,rz"
            );
        }
        folders.iter().map(|f| load(f)).collect::<Result<_>>()?
    };
    let postures: Vec<Option<f64>> = match opt("--posture") {
        Some(s) => s.split(',').map(|x| x.trim().parse::<f64>().ok()).collect(),
        None => Vec::new(),
    };
    let states: Vec<StateInput> = studies
        .iter()
        .enumerate()
        .map(|(i, s)| StateInput {
            label: ((b'A' + i as u8) as char).to_string(),
            study: s,
            structure_set: None,
            posture: postures.get(i).copied().flatten(),
        })
        .collect();

    // ---- the request ----
    let anchor = match opt("--anchor") {
        Some(a) => Anchor::parse(&a),
        None if phantom => Anchor::Structure("BONE_UPPER".into()),
        None if opt("--simulate").is_some() => Anchor::Scanner,
        None => Anchor::Image,
    };
    let bodies = match opt("--bodies").as_deref() {
        None | Some("bones") => Bodies::Bones,
        Some("none") => Bodies::None,
        Some(list) => Bodies::Named(list.split(',').map(|s| s.trim().to_string()).collect()),
    };
    let mode = match opt("--mode") {
        Some(m) => {
            Mode::from_key(&m).ok_or_else(|| anyhow!("--mode {m}: diffeomorphic or linear"))?
        }
        None => Mode::Diffeomorphic,
    };
    let joints = opts_all("--joint")
        .iter()
        .map(|j| {
            let p: Vec<&str> = j.split(':').collect();
            match p.as_slice() {
                [n, a, b] => Ok((n.to_string(), a.to_string(), b.to_string())),
                _ => Err(anyhow!("--joint {j}: NAME:FIRST:SECOND")),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let req = PostureRequest {
        anchor,
        bodies,
        mode,
        lattice_mm: num("--lattice", 6.0)?,
        analysis_mm: num("--analysis", if phantom { 3.0 } else { 2.0 })?,
        reach_mm: num("--reach", 40.0)?,
        residual: has("--residual").then(wp::default_residual),
        joints,
        wepl: match opt("--wepl") {
            Some(w) => {
                let v: Vec<f64> = w
                    .split(',')
                    .map(|x| x.trim().parse::<f64>())
                    .collect::<std::result::Result<_, _>>()
                    .map_err(|e| anyhow!("--wepl {w}: {e}"))?;
                Some(wp::WeplBeam {
                    gantry_deg: v.first().copied().unwrap_or(0.0),
                    couch_deg: v.get(1).copied().unwrap_or(0.0),
                    hlut: None,
                })
            }
            None => None,
        },
        smooth_chain: !has("--no-smooth"),
        ..PostureRequest::default()
    };

    // ---- the build, with its progress on standard error ----
    let p = Arc::new(Progress::default());
    let watch = {
        let p = p.clone();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d2 = done.clone();
        let h = std::thread::spawn(move || {
            let mut last = String::new();
            while !d2.load(std::sync::atomic::Ordering::Relaxed) {
                let m = p.get();
                if m != last {
                    eprintln!("[{:5.1}%] {m}", p.frac() * 100.0);
                    last = m;
                }
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
        });
        (done, h)
    };
    let t0 = std::time::Instant::now();
    let built = wp::build(&states, &req, &p);
    watch.0.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = watch.1.join();
    let o = built?;
    eprintln!("built in {:.1} s", t0.elapsed().as_secs_f64());
    for w in &o.warnings {
        eprintln!("warning: {w}");
    }

    // ---- what it found ----
    let t = &o.tables;
    println!("mode {}, anchor {}", t.model.mode.label(), t.model.anchor);
    println!(
        "{:<20} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}",
        "body", "angle", "axis RL", "axis AP", "axis SI", "fit mm", "stretch", "p95"
    );
    for b in &t.bodies {
        println!(
            "{:<20} {:>8.2}d {:>8.3} {:>8.3} {:>8.3} {:>8.2} {:>8.4} {:>7.4}{}",
            b.name,
            b.screw.angle_deg,
            b.screw.axis.x,
            b.screw.axis.y,
            b.screw.axis.z,
            b.fit_mean_mm,
            b.max_stretch,
            b.stretch_p95,
            if b.seeded {
                " (principal-axes start)"
            } else {
                ""
            }
        );
    }
    for j in &t.joints {
        let e = j.at.last().copied().unwrap_or_default();
        println!(
            "joint {}: RL {:.2}  AP {:.2}  SI {:.2} deg",
            j.name, e[0], e[1], e[2]
        );
    }
    println!(
        "{:<20} {:>10} {:>10} {:>9} {:>9}",
        "structure", "dV cm3", "shift mm", "Dice", "HD95 mm"
    );
    for s in &t.structures {
        let (dv, _) = s.volume_change();
        println!(
            "{:<20} {:>10.2} {:>10.2} {:>9.3} {:>9.2}",
            s.name,
            dv,
            s.contoured_shift().map_or(f64::NAN, |v| v.length()),
            s.check.map_or(f64::NAN, |c| c.dice),
            s.check.map_or(f64::NAN, |c| c.hd95_mm)
        );
    }
    if let Some(beam) = &t.model.beam {
        println!("WEPL along the beam at {beam}:");
        for s in &t.structures {
            println!(
                "  {:<20} {:>8.1} -> {:>8.1} mm ({:+.1})",
                s.name,
                s.wepl_mm.first().copied().unwrap_or(f64::NAN),
                s.wepl_mm.last().copied().unwrap_or(f64::NAN),
                s.wepl_change(s.wepl_mm.len().saturating_sub(1))
                    .unwrap_or(f64::NAN)
            );
        }
    }
    let m = &t.model;
    println!(
        "Jacobian min {:.3}, folded {:.4}; inverse consistency at the end: max {:.3} mm, p95 {:.3} mm",
        m.jacobian_min, m.folded_fraction, m.inverse_mm[1][0], m.inverse_mm[1][1]
    );
    if let Some(n) = opt("--bench-slices") {
        let n: usize = n.parse().map_err(|e| anyhow!("--bench-slices {n}: {e}"))?;
        bench_slices(&o.model, &studies, n.max(1))?;
    }
    if phantom {
        let w = truth.organ_cm3();
        println!(
            "truth: hinge {:.1} deg about RL; organ {:.2} -> {:.2} cm3 ({:+.2})",
            truth.angle_deg,
            w[0],
            w[1],
            w[1] - w[0]
        );
    }

    // ---- the film ----
    let show = match opt("--show") {
        Some(s) => apply::Show::from_key(&s).ok_or_else(|| anyhow!("--show {s}"))?,
        None => apply::Show::StartForward,
    };
    if let Some(gif) = opt("--gif") {
        let images: Vec<Arc<Volume>> = studies.iter().map(|s| s.volume.clone()).collect();
        let window = show.window().unwrap_or(studies[0].default_window);
        let mut fp = film::FilmParams::centred(&o.model.grid, show, window);
        fp.views = match opt("--gif-view").as_deref() {
            None | Some("all") => film::FilmViews::All,
            Some("axial") => film::FilmViews::One(ViewPlane::Axial),
            Some("sagittal") => film::FilmViews::One(ViewPlane::Sagittal),
            Some("coronal") => film::FilmViews::One(ViewPlane::Coronal),
            Some(v) => bail!("--gif-view {v}: all, axial, sagittal or coronal"),
        };
        fp.width = num("--gif-width", 800.0)? as usize;
        fp.frames = num("--gif-frames", 21.0)? as usize;
        fp.fps = num("--fps", 6.0)? as f32;
        fp.bounce = has("--bounce");
        fp.wash = has("--wash");
        let t0 = std::time::Instant::now();
        let s = film::write_gif(
            Path::new(&gif),
            &o.model,
            &images,
            None,
            studies[0].structure_sets.first(),
            &fp,
            &Progress::default(),
        )?;
        println!(
            "GIF: {} frames, {} x {} px, {} KB in {:.1} s: {gif}",
            s.frames,
            s.width,
            s.height,
            s.bytes.div_ceil(1024),
            t0.elapsed().as_secs_f64()
        );
    }

    // ---- what it writes ----
    if let Some(out) = opt("--out") {
        let out = PathBuf::from(out);
        std::fs::create_dir_all(&out).with_context(|| format!("create {}", out.display()))?;
        std::fs::write(
            out.join("tables.csv"),
            rust_dicom_station::motion::csv_file_bytes(&t.csv()),
        )?;
        file::write(
            &out.join(format!("model.{}", file::EXTENSION)),
            &o.model,
            &o.occupancy,
        )?;
        let last = studies.last().unwrap();
        wp::export_registration(
            &o.model,
            &studies[0],
            &last.volume.frame_of_reference_uid,
            &out.join("posture_reg.dcm"),
        )?;
        let images: Vec<Arc<Volume>> = studies.iter().map(|s| s.volume.clone()).collect();
        let frames = num("--frames", 0.0)? as usize;
        if frames > 0 {
            let dir = out.join("frames");
            std::fs::create_dir_all(&dir)?;
            let kf = apply::key_frames(
                &o.model,
                &images,
                frames,
                show,
                &o.model.grid,
                &apply::Extras::default(),
                &Progress::default(),
            )?;
            let window = show.window().unwrap_or(studies[0].default_window);
            for (i, f) in kf.iter().enumerate() {
                for (plane, tag) in [
                    (ViewPlane::Axial, "axial"),
                    (ViewPlane::Sagittal, "sagittal"),
                    (ViewPlane::Coronal, "coronal"),
                ] {
                    png(
                        &f.volume,
                        plane,
                        window,
                        &dir.join(format!("frame_{i:03}_{tag}.png")),
                    )?;
                }
            }
            println!("{} key frames in {}", kf.len(), dir.display());
        }
        if let Some(at) = opt("--export") {
            let at: f64 = at.parse().map_err(|e| anyhow!("--export {at}: {e}"))?;
            let dir = out.join(format!("frame_{at:.2}"));
            let e = wp::export_frame(
                &o.model,
                &images,
                &studies[0],
                None,
                at,
                show,
                &dir,
                &Progress::default(),
            )?;
            println!(
                "t = {at:.2}: {} images, {} structure set(s) in {}",
                e.image_files.len(),
                e.structure_files.len(),
                dir.display()
            );
        }
        println!("written to {}", out.display());
    }
    Ok(())
}

fn load(dir: &Path) -> Result<LoadedStudy> {
    wp::load_state(&dir.to_string_lossy(), &Progress::default())
}

/// The plane through the centre of `vol`, windowed, as an 8-bit PNG.
fn png(vol: &Volume, plane: ViewPlane, window: (f32, f32), path: &Path) -> Result<()> {
    let n = vol.plane_slice_count(plane);
    let [w, h] = vol.plane_dims(plane);
    let mut buf = Vec::new();
    vol.extract_slice(plane, n / 2, &mut buf);
    let (c, wd) = (window.0, window.1.max(1.0));
    let px: Vec<u8> = buf
        .iter()
        .map(|&v| (((v as f32 - (c - wd / 2.0)) / wd).clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    let img = image::GrayImage::from_raw(w as u32, h as u32, px).context("slice to image")?;
    img.save(path)
        .with_context(|| format!("write {}", path.display()))
}
