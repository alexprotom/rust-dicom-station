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
//! * `--anchor NAME|auto|image|scanner` - what is held still (default: the
//!   phantom's `BONE_UPPER`, otherwise `auto`: the sacrum when every scan
//!   sees all of it, else the middle vertebra every scan sees whole, else
//!   the largest such bone, else the whole image);
//! * `--whole-scan` - fit and draw the scans as they are; by default each
//!   state is its patient only, outside its EXTERNAL structure set to air
//!   (the couch, the chair), when it has one;
//! * `--bodies bones|none|A,B,C` - the rigid bodies (default `bones`);
//! * `--mode diffeomorphic|linear`;
//! * `--lattice MM` (6), `--analysis MM` (2), `--reach MM` (40);
//! * `--residual` - add the B-spline residual layer; `--residual-grid MM`
//!   (its control lattice), `--residual-reg X` (its bending penalty),
//!   `--residual-metric msd|mi` override the defaults
//!   (`workflow::posture::default_residual`);
//! * `--posture P1,P2,...` - a posture parameter per state;
//! * `--joint NAME:FIRST:SECOND` - a joint to report (repeatable);
//! * `--out DIR` - write `tables.csv`, `model.rdsposture`, the deformation
//!   as `posture_reg.dcm`, and with `--frames N` (default 0) N key frames
//!   as PNGs of the three planes through the centre (`frames/`);
//! * `--show forward|backward|crossfade|difference|jacobian` - what the key
//!   frames and the GIF show;
//! * `--gif FILE` - the path as an animated GIF (`posture::film`, on the
//!   CPU): `--gif-view all|axial|sagittal|coronal|3d` (default `all`: the
//!   three planes through the centre side by side and the 3D scene of the
//!   first state's structures under them, its camera as a fresh 3D pane
//!   has it), `--gif-width PX` (800), `--gif-frames N` (21), `--fps F`
//!   (6), `--bounce` (there and back), `--wash` (the Jacobian wash),
//!   `--gif-field arrows|grid` (the displacement over the planes, as the
//!   window's overlay) with `--arrow-spacing MM` (12) and `--arrow-scale X`
//!   (3); the structures of the first state outlined;
//! * `--export T` - the image and structures at time `T` as DICOM
//!   (`frame_T/`);
//! * `--carry SPEC` - another image of the first state's position (an MR:
//!   a folder or `DIR::FILE` like a state) carried along the path instead
//!   of the first state's own: the key frames and the GIF show it (with
//!   `--show forward`, the default), in its own window;
//!   `--carry-align rigid|scanner` (default `rigid`: mutual information)
//!   lays it onto the first state. Not written as DICOM here;
//! * `--wepl GANTRY[,COUCH]` - the WEPL to each structure's centroid along
//!   a beam at these IEC 61217 angles (degrees, in the first state's
//!   patient position): first, last and its change;
//! * `--no-smooth` - a chain without curves through its states (a constant
//!   turn per pair);
//! * `--folds` - where the path at its end folds or nearly folds (Jacobian
//!   under 0.1 on the lattice): each such node with the two bodies nearest
//!   it, grouped;
//! * `--bench-slices N` - N live steps of the three central planes, on the
//!   CPU and (with a GPU) on the GPU: the time per step and how far the two
//!   answers are apart.
//!
//! A cohort (`posture::template`):
//!
//! * with `--out` and two states, `sample.json` is written beside the
//!   model: what a posture template keeps of it, labelled `--label TEXT`
//!   (a pseudonymised name; nothing is read from the files);
//! * `--make-template OUT FILE...` - the samples (or templates) given as
//!   the positional arguments into one template, its summary and its
//!   leave-one-out printed; nothing is built;
//! * `--loo TEMPLATE` - the summary and the leave-one-out of a template:
//!   each patient's bones predicted from the others' joints against its own
//!   fitted motions, and its organs with the others' offsets;
//! * `--template FILE` with one state - the second state predicted from the
//!   template instead of fitted to a scan (`workflow::posture::predict`);
//!   `--no-organs` leaves the organs' offsets out. The tables, `--gif`,
//!   `--frames` and `--export 1` (the predicted image as DICOM) work as for
//!   a pair.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use rust_dicom_station::gen_test_data::{self, ArticulatedParams};
use rust_dicom_station::loader::LoadedStudy;
use rust_dicom_station::mesh3d;
use rust_dicom_station::posture::{apply, file, film, template, Anchor, Mode};
use rust_dicom_station::progress::Progress;
use rust_dicom_station::registration::dvf::{self, FieldStyle};
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
    const VALUED: [&str; 33] = [
        "--label",
        "--make-template",
        "--loo",
        "--template",
        "--residual-grid",
        "--residual-reg",
        "--residual-metric",
        "--carry",
        "--carry-align",
        "--wepl",
        "--gif",
        "--gif-view",
        "--gif-field",
        "--arrow-spacing",
        "--arrow-scale",
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

    // ---- a cohort: templates from samples, and their leave-one-out ----
    if let Some(out) = opt("--make-template") {
        let mut samples = Vec::new();
        for f in &folders {
            samples.extend(template::read_samples(f)?);
        }
        if samples.is_empty() {
            bail!("--make-template OUT SAMPLE...: no sample given");
        }
        let t = template::Template::new(samples);
        t.write(Path::new(&out))?;
        println!("template of {} patients written to {out}", t.samples.len());
        print_template(&t);
        return Ok(());
    }
    if let Some(f) = opt("--loo") {
        print_template(&template::Template::read(Path::new(&f))?);
        return Ok(());
    }
    let tmpl = opt("--template")
        .map(|f| template::Template::read(Path::new(&f)))
        .transpose()?;

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
        if folders.len() < 2 && !(tmpl.is_some() && folders.len() == 1) {
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
        None => Anchor::Auto,
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
        residual: if has("--residual") {
            let mut r = wp::default_residual();
            r.grid_spacing_mm = num("--residual-grid", r.grid_spacing_mm)?;
            r.regularization = num("--residual-reg", r.regularization)?;
            r.metric = match opt("--residual-metric").as_deref() {
                None => r.metric,
                Some("msd") => rust_dicom_station::registration::Metric::MeanSquares,
                Some("mi") => rust_dicom_station::registration::Metric::MutualInformation,
                Some(m) => bail!("--residual-metric {m}: msd or mi"),
            };
            Some(r)
        } else {
            None
        },
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
        patient_only: !has("--whole-scan"),
        sample: opt("--out").is_some() && studies.len() == 2,
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
    let built = match &tmpl {
        Some(t) => wp::predict(&states[0], t, &req, !has("--no-organs"), &p),
        None => wp::build(&states, &req, &p),
    };
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
    println!(
        "check Dice: bones {}, organs {}",
        m.check_used_dice.map_or("-".into(), |d| format!("{d:.3}")),
        m.check_unused_dice
            .map_or("-".into(), |d| format!("{d:.3}"))
    );
    let v = t.verdict();
    println!("verdict: {}", v.level.label());
    for (level, line) in &v.lines {
        println!("  [{}] {line}", level.label());
    }
    if has("--folds") {
        print_folds(&o);
    }
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

    // ---- the images the model draws: each state's patient, or the
    // carried image in place of the first ----
    let show = match opt("--show") {
        Some(s) => apply::Show::from_key(&s).ok_or_else(|| anyhow!("--show {s}"))?,
        None => apply::Show::StartForward,
    };
    let mut images: Vec<Arc<Volume>> = o.images.clone();
    let mut window = show.window().unwrap_or(studies[0].default_window);
    let carried = match opt("--carry") {
        Some(spec) => {
            let other = wp::load_state(&spec, &Progress::default())?;
            let align = match opt("--carry-align").as_deref() {
                None | Some("rigid") => wp::CarryAlign::Rigid,
                Some("scanner") => wp::CarryAlign::Scanner,
                Some(a) => bail!("--carry-align {a}: rigid or scanner"),
            };
            let (v, line) = wp::carry_image(
                &studies[0].volume,
                &other.volume,
                &align,
                req.analysis_mm,
                &Progress::default(),
            )?;
            println!("carried image: {line}");
            images[0] = Arc::new(v);
            if show.window().is_none() {
                window = other.default_window;
            }
            true
        }
        None => false,
    };
    if let Some(gif) = opt("--gif") {
        let mut fp = film::FilmParams::centred(&o.model.grid, show, window);
        fp.views = match opt("--gif-view") {
            None => film::FilmViews::All,
            Some(v) => film::FilmViews::from_key(&v)
                .ok_or_else(|| anyhow!("--gif-view {v}: all, axial, sagittal, coronal or 3d"))?,
        };
        fp.field = match opt("--gif-field").as_deref() {
            None => None,
            Some(s) => Some(film::FieldOverlay {
                style: match s {
                    "arrows" => FieldStyle::Arrows,
                    "grid" => FieldStyle::Grid,
                    _ => bail!("--gif-field {s}: arrows or grid"),
                },
                spacing_mm: num("--arrow-spacing", dvf::DEFAULT_ARROW_SPACING_MM)?,
                scale: num("--arrow-scale", 3.0)? as f32,
                colored: true,
            }),
        };
        // The 3D scene: the first state's structures, meshed.
        let scene = studies[0]
            .structure_sets
            .first()
            .filter(|_| fp.views.wants_scene())
            .map(|ss| {
                film::FilmScene::new(Arc::new(mesh3d::build_meshes(ss, &Progress::default())))
            });
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
            scene.as_ref(),
            &fp,
            &Progress::default(),
        )?;
        println!(
            "GIF: {} frames, {} x {} px, {} KB in {:.1} s{}: {gif}",
            s.frames,
            s.width,
            s.height,
            s.bytes.div_ceil(1024),
            t0.elapsed().as_secs_f64(),
            if s.scene { ", with the 3D scene" } else { "" }
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
        if let Some(s) = &o.sample {
            let mut s = s.clone();
            s.label = opt("--label").unwrap_or_default();
            s.write(&out.join(template::SAMPLE_FILE))?;
            println!(
                "sample: {} bones, {} joints, {} organs",
                s.bodies.len(),
                s.joints.len(),
                s.organs.len()
            );
        }
        let last = studies.last().unwrap();
        wp::export_registration(
            &o.model,
            &studies[0],
            &last.volume.frame_of_reference_uid,
            &out.join("posture_reg.dcm"),
        )?;
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
        if let Some(at) = opt("--export").filter(|_| {
            if carried {
                println!("--export: a carried image is not written as DICOM (frames and GIF only)");
            }
            !carried
        }) {
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

/// Where the path's end folds or nearly folds, by the two bodies nearest.
fn print_folds(o: &wp::PostureOutcome) {
    let m = &o.model;
    let lat = &m.lattice;
    let end = m.frame(m.span()).forward;
    let jac = end.jacobians();
    let names = m.body_names();
    let occ = o.occupancy.first().cloned().unwrap_or_default();
    let nodes: Vec<Vec<rust_dicom_station::geometry::Vec3>> = occ
        .iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .filter(|(_, &x)| x != 0)
                .map(|(n, _)| lat.node(n))
                .collect()
        })
        .collect();
    let mut groups: std::collections::BTreeMap<String, (usize, f64)> = Default::default();
    let mut total = 0;
    for (n, &j) in jac.iter().enumerate() {
        if j >= 0.1 {
            continue;
        }
        total += 1;
        let p = lat.node(n);
        let mut d: Vec<(f64, &str)> = nodes
            .iter()
            .zip(&names)
            .map(|(v, name)| {
                (
                    v.iter()
                        .map(|q| (*q - p).length())
                        .fold(f64::INFINITY, f64::min),
                    name.as_str(),
                )
            })
            .collect();
        d.sort_by(|a, b| a.0.total_cmp(&b.0));
        let key = match d.as_slice() {
            [a, b, ..] => format!("{} ({:.0} mm) / {} ({:.0} mm)", a.1, a.0, b.1, b.0),
            [a] => format!("{} ({:.0} mm)", a.1, a.0),
            [] => "no body".into(),
        };
        let e = groups.entry(key).or_insert((0, f64::INFINITY));
        e.0 += 1;
        e.1 = e.1.min(j);
    }
    println!("folds: {total} lattice node(s) with a Jacobian under 0.1 at the end");
    let mut v: Vec<_> = groups.into_iter().collect();
    v.sort_by_key(|a| std::cmp::Reverse(a.1 .0));
    for (k, (n, j)) in v.iter().take(15) {
        println!("  {n:>4} node(s), down to {j:.2}: nearest {k}");
    }
}

/// A template's summary and its leave-one-out, as tables.
fn print_template(t: &template::Template) {
    for l in t.summary() {
        println!("{l}");
    }
    let loo = template::leave_one_out(t);
    let groups = template::GROUPS;
    println!();
    println!(
        "leave one out: mean error at each bone's seven points, mm (no motion at all in brackets)"
    );
    print!("{:<12} {:<14}", "patient", "root");
    for g in groups {
        print!(" {g:>15}");
    }
    println!(" {:>9}", "unreached");
    for p in &loo {
        print!("{:<12} {:<14}", p.label, p.root);
        for g in groups {
            let v: Vec<&template::LooBody> = p
                .bodies
                .iter()
                .filter(|b| b.group == g && b.error_mm.is_some())
                .collect();
            if v.is_empty() {
                print!(" {:>15}", "-");
                continue;
            }
            let e = v.iter().filter_map(|b| b.error_mm).sum::<f64>() / v.len() as f64;
            let s = v.iter().map(|b| b.still_mm).sum::<f64>() / v.len() as f64;
            print!(" {:>15}", format!("{e:.1} ({s:.1})"));
        }
        println!(
            " {:>9}",
            p.bodies.iter().filter(|b| b.error_mm.is_none()).count()
        );
    }
    println!();
    println!("over every bone of every patient left out: mean / median / 90th percentile, mm");
    for g in template::loo_groups(&loo) {
        let (e, x) = (g.template, g.still);
        println!(
            "  {:<9} {:>4} bones  template {:5.1} / {:5.1} / {:5.1}    no motion {:5.1} / {:5.1} / {:5.1}",
            g.group, g.n, e.mean, e.median, e.p90, x.mean, x.median, x.p90
        );
    }
    let organs = template::loo_organs(&loo);
    if !organs.is_empty() {
        println!();
        println!(
            "organs, centroid from where it was contoured on the second state, mean mm: \
             still / own fitted bones / bones + template offset"
        );
        for o in organs {
            println!(
                "  {:<28} n {:>2}   {:6.1} {:6.1} {:6.1}",
                o.name, o.n, o.still_mm, o.bones_mm, o.template_mm
            );
        }
    }
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
