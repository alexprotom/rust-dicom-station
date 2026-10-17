//! Headless 2D/3D registration: a planning CT and radiographs in, the pose
//! out.
//!
//! ```text
//! cargo run --release --example reg2d3d_cli -- <CT_DIR> [IMAGE.dcm ...] \
//!     [--simulate tx,ty,tz,rx,ry,rz[,noise]] [--gantry 0,90] \
//!     [--iec sad,sid,gantry,couch]... [--room FILE --view NAME]... \
//!     [--iso x,y,z] [--patient HFS|HFP|FFS|FFP|HFDR|HFDL|FFDR|FFDL|upright:DEG] \
//!     [--metric ncc|gc|gd|pi|lncc|mi] [--optimizer powell|nelder-mead] \
//!     [--init plan|in-plane|grid] [--levels 3] [--dof 6|3] [--clip HALF_MM] \
//!     [--threshold -800] [--hu water|mv|raw] [--pixels 256] [--step 1.0] \
//!     [--range MM,DEG] [--refine] [--no-conditioning] [--transmission] \
//!     [--clip-structure NAME [--margin MM]] [--clip-rays] \
//!     [--drr-of CT_DIR [--compare-3d]] \
//!     [--device cpu|gpu|auto] [--out DIR] [--bench]
//! ```
//!
//! Each image's geometry comes from its own RT Image attributes; an image
//! without one takes the next `--iec` (SAD, SID, gantry and couch angles,
//! the panel the size of the image at its pixel spacing) or the next
//! `--view` of the `--room` file, in the order the images are given.
//!
//! `--simulate` needs no images: it renders radiographs of the CT itself
//! found at the pose given (translations in mm, rotations in degrees, the
//! patient's displacement as the result reports it), one per `--gantry`
//! angle (default 0 and 90), with Gaussian noise at `noise` of the range
//! (default 0.02), registers them from the plan and prints the error: the
//! one-command self-test on any CT.
//!
//! `--drr-of` renders the radiographs from another CT instead - a later
//! scan, a phase of a 4DCT, a cone-beam CT - placed as this one's (the same
//! frame of reference), at the `--simulate` pose if one is given; the
//! registration then finds this CT's pose under that anatomy, and
//! `--compare-3d` registers the two CTs in 3D as well (the rigid
//! elastix-style engine, restricted to the clip box) and prints how far the
//! two answers are apart at the box's corners (the 3D run over the voxels
//! above `--threshold`, what the DRRs see): the real-data check of
//! `docs/registration-2d3d.md`. `--clip-structure` puts the clip box about
//! a structure of the CT's folder, `--margin` (default 10 mm) larger. The
//! box says which pixels are compared (its shadow on each image); with
//! `--clip-rays` the rays also integrate inside it only.
//!
//! `--bench` times the projector on the CT at the three sizes of the plan
//! (512² at 1 mm, 256² at 2 mm, 128² at 4 mm) and exits. `--device` puts
//! the renderings of the registration (and of the bench) on the GPU; the
//! default is the CPU.
//!
//! `--out` writes, per view, the radiograph and the DRR at the answer as
//! PNG, and `report.json`; with `--simulate`, also the simulated
//! radiographs as RT Images placed as planned (`sim_G0.dcm`), to load with
//! the CT into the viewer.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use rust_dicom_station::dicom_export::{write_rt_image, ObjectMeta};
use rust_dicom_station::drr::{Geometry, HuMode};
use rust_dicom_station::extras::{self, PlanarImage};
use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::iec::{PatientFrame, PatientPosition};
use rust_dicom_station::loader;
use rust_dicom_station::nn::device::DevicePref;
use rust_dicom_station::progress::{Progress, Quiet, Stderr};
use rust_dicom_station::projection::gpu::GpuProjector;
use rust_dicom_station::projection::{render_posed, MuVolume, ProjectionGeometry};
use rust_dicom_station::registration::metric2d::{Image2d, Metric2d};
use rust_dicom_station::registration::projective::{
    register_2d3d, ClipBox, Dof2d3d, Init2d3d, Optimizer2d, Request2d3d, View2d,
};
use rust_dicom_station::registration::{self, RegMethod, RegParams, RegionMask, RigidTransform};
use rust_dicom_station::rooms;
use rust_dicom_station::setup;
use rust_dicom_station::volume::Volume;

mod common;

fn triple(s: &str) -> Result<Vec3> {
    let v = common::numbers(s);
    if v.len() != 3 {
        bail!("three numbers expected, not {s:?}");
    }
    Ok(Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
}

fn patient(s: &str) -> Result<PatientFrame> {
    let l = s.trim().to_ascii_lowercase();
    if let Some(deg) = l.strip_prefix("upright") {
        let facing_deg = deg.trim_start_matches(':').parse().unwrap_or(0.0);
        return Ok(PatientFrame::Upright { facing_deg });
    }
    PatientPosition::from_code(s)
        .map(PatientFrame::Position)
        .ok_or_else(|| anyhow!("--patient: a Patient Position code or upright:DEG, not {s:?}"))
}

/// Deterministic Gaussian noise.
struct Noise(u64);

impl Noise {
    fn gauss(&mut self) -> f64 {
        let mut u = || {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (a, b) = (u(), u());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}

fn png(path: &Path, img: &Image2d) -> Result<()> {
    let mut v: Vec<f32> = img.data.clone();
    v.sort_by(f32::total_cmp);
    let lo = v[v.len() / 200];
    let hi = v[v.len() - 1 - v.len() / 200].max(lo + 1e-6);
    let bytes: Vec<u8> = img
        .data
        .iter()
        .map(|&x| (((x - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    image::GrayImage::from_raw(img.dims[0] as u32, img.dims[1] as u32, bytes)
        .ok_or_else(|| anyhow!("image size"))?
        .save(path)
        .with_context(|| format!("write {}", path.display()))
}

fn bench(vol: &Volume, iso: Vec3, device: DevicePref) -> Result<()> {
    let mu = MuVolume::prepare(vol, HuMode::Water, -800.0, None);
    let gpu = match device.resolve()? {
        Some(ctx) => Some((ctx.describe(), GpuProjector::new(&ctx, &mu)?)),
        None => None,
    };
    let render = |cam: &ProjectionGeometry, step: f64| -> Result<Vec<f32>> {
        match &gpu {
            Some((_, g)) => g.render(cam, step),
            None => render_posed(&mu, cam, step, &Quiet),
        }
    };
    println!("| detector | step | rays | samples | time | per sample |");
    println!("|---|---|---|---|---|---|");
    for (px, step) in [(512usize, 1.0f64), (256, 2.0), (128, 4.0)] {
        let cam = Geometry {
            isocenter: iso,
            dims: [px, px],
            panel_mm: [400.0, 400.0],
            ..Geometry::default()
        }
        .camera();
        // One untimed run, then the best of three.
        let _ = render(&cam, step)?;
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            let _ = render(&cam, step)?;
            best = best.min(t.elapsed().as_secs_f64());
        }
        let (lo, hi) = mu.bounds();
        let depth = (hi - lo).length();
        let samples = (px * px) as f64 * depth / step;
        println!(
            "| {px} × {px} | {step:.0} mm | {} | {:.1} M | {:.1} ms | {:.2} ns |",
            px * px,
            samples / 1e6,
            best * 1e3,
            best * 1e9 / samples
        );
    }
    println!(
        "\n{}, CT {} × {} × {} at {:.2} × {:.2} × {:.2} mm",
        match &gpu {
            Some((name, _)) => name.clone(),
            None => format!("{} threads", rayon::current_num_threads()),
        },
        vol.dims[0],
        vol.dims[1],
        vol.dims[2],
        vol.spacing[0],
        vol.spacing[1],
        vol.spacing[2]
    );
    Ok(())
}

/// The 3D/3D rigid registration of `found` (fixed) to `planned` (moving)
/// inside `clip`, over the voxels above `threshold_hu` (what the DRRs
/// see), against the 2D/3D answer `t`: both map the room (the coordinates
/// `found` is in) onto the planning CT.
fn compare_in_3d(
    planned: &Volume,
    found: &Volume,
    clip: &ClipBox,
    threshold_hu: f32,
    t: &RigidTransform,
) -> Result<()> {
    let (lo, hi) = clip.corners();
    let [nx, ny, nz] = found.dims;
    let mask: Vec<u8> = (0..nx * ny * nz)
        .map(|o| {
            let p = found.voxel_to_patient(
                (o % nx) as f64,
                ((o / nx) % ny) as f64,
                (o / (nx * ny)) as f64,
            );
            (p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z)
                as u8
        })
        .collect();
    let region = RegionMask::from_mask(found, &mask, "clip box".into(), 0.0)
        .ok_or_else(|| anyhow!("the clip box holds no voxel of the --drr-of CT"))?;
    let params = RegParams {
        method: RegMethod::ElastixRigid,
        region: Some(std::sync::Arc::new(region)),
        fixed_threshold: threshold_hu,
        ..RegParams::default()
    };
    let started = Instant::now();
    let r3 = registration::register(found, planned, &params, &Progress::default())?;
    let r = r3.transform.rigid.recentered(clip.center).params();
    println!(
        "3D/3D rigid in the box: x {:+.2} · y {:+.2} · z {:+.2} mm · rx {:+.2} · ry {:+.2} · rz {:+.2}° ({:.1} s)",
        r[3],
        r[4],
        r[5],
        r[0].to_degrees(),
        r[1].to_degrees(),
        r[2].to_degrees(),
        started.elapsed().as_secs_f64()
    );
    let mut worst = 0.0f64;
    for c in 0..8 {
        let p = Vec3::new(
            if c & 1 == 0 { lo.x } else { hi.x },
            if c & 2 == 0 { lo.y } else { hi.y },
            if c & 4 == 0 { lo.z } else { hi.z },
        );
        worst = worst.max((t.map(p) - r3.transform.map(p)).length());
    }
    let centre = (t.map(clip.center) - r3.transform.map(clip.center)).length();
    println!(
        "2D/3D against 3D/3D: {centre:.2} mm at the box's centre, {worst:.2} mm at most at its corners"
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut ct_dir: Option<PathBuf> = None;
    let mut images: Vec<PathBuf> = Vec::new();
    let mut simulate: Option<Vec<f32>> = None;
    let mut gantries = vec![0.0f64, 90.0];
    let mut iecs: Vec<Vec<f32>> = Vec::new();
    let mut room: Option<PathBuf> = None;
    let mut views: Vec<String> = Vec::new();
    let mut iso: Option<Vec3> = None;
    let mut frame: Option<PatientFrame> = None;
    let mut out: Option<PathBuf> = None;
    let mut bench_only = false;
    let mut transmission = false;
    let mut clip_half: Option<f64> = None;
    let mut clip_structure: Option<String> = None;
    let mut margin = 10.0f64;
    let mut drr_of: Option<PathBuf> = None;
    let mut compare_3d = false;
    let mut req = Request2d3d::new(Vec::new(), Vec3::ZERO);
    while let Some(a) = args.next() {
        let mut next = || {
            args.next()
                .ok_or_else(|| anyhow!("{a}: a value is missing"))
        };
        match a.as_str() {
            "--simulate" => simulate = Some(common::numbers(&next()?)),
            "--gantry" => {
                gantries = common::numbers(&next()?)
                    .iter()
                    .map(|&g| g as f64)
                    .collect()
            }
            "--iec" => iecs.push(common::numbers(&next()?)),
            "--room" => room = Some(PathBuf::from(next()?)),
            "--view" => views.push(next()?),
            "--iso" => iso = Some(triple(&next()?)?),
            "--patient" => frame = Some(patient(&next()?)?),
            "--metric" => {
                let s = next()?;
                req.metric = Metric2d::from_key(&s).ok_or_else(|| anyhow!("--metric {s:?}"))?;
            }
            "--optimizer" => {
                let s = next()?;
                req.optimizer =
                    Optimizer2d::from_key(&s).ok_or_else(|| anyhow!("--optimizer {s:?}"))?;
            }
            "--init" => {
                let s = next()?;
                req.init = Init2d3d::from_key(&s).ok_or_else(|| anyhow!("--init {s:?}"))?;
            }
            "--levels" => req.levels = next()?.parse()?,
            "--dof" => {
                req.dof = match next()?.as_str() {
                    "6" => Dof2d3d::Six,
                    "3" => Dof2d3d::Translations,
                    other => bail!("--dof 6|3, not {other:?}"),
                }
            }
            "--clip" => clip_half = Some(next()?.parse()?),
            "--clip-structure" => clip_structure = Some(next()?),
            "--margin" => margin = next()?.parse()?,
            "--drr-of" => drr_of = Some(PathBuf::from(next()?)),
            "--compare-3d" => compare_3d = true,
            "--clip-rays" => req.clip_rays = true,
            "--threshold" => req.threshold_hu = next()?.parse()?,
            "--hu" => {
                let s = next()?;
                req.hu =
                    HuMode::from_key(&s).ok_or_else(|| anyhow!("--hu water|mv|raw, not {s:?}"))?;
            }
            "--pixels" => req.max_pixels = next()?.parse()?,
            "--step" => req.step_mm = next()?.parse()?,
            "--range" => {
                let r = common::numbers(&next()?);
                if r.len() != 2 {
                    bail!("--range MM,DEG");
                }
                req.range_mm = r[0] as f64;
                req.range_deg = r[1] as f64;
            }
            "--refine" => req.refine_gradient = true,
            "--no-conditioning" => req.conditioning = false,
            "--transmission" => transmission = true,
            "--out" => out = Some(PathBuf::from(next()?)),
            "--bench" => bench_only = true,
            "--device" => {
                let s = next()?;
                req.device = DevicePref::from_key(&s)
                    .ok_or_else(|| anyhow!("--device cpu|gpu|auto, not {s:?}"))?;
            }
            other if other.starts_with("--") => bail!("unknown option {other:?}"),
            other if ct_dir.is_none() => ct_dir = Some(PathBuf::from(other)),
            other => images.push(PathBuf::from(other)),
        }
    }
    let ct_dir =
        ct_dir.ok_or_else(|| anyhow!("usage: reg2d3d_cli <CT_DIR> [IMAGE ...] [options]"))?;
    let study = loader::load_directory(&ct_dir, &Progress::default())?;
    let vol = &study.volume;
    let centre = vol.voxel_to_patient(
        (vol.dims[0].max(1) - 1) as f64 * 0.5,
        (vol.dims[1].max(1) - 1) as f64 * 0.5,
        (vol.dims[2].max(1) - 1) as f64 * 0.5,
    );
    eprintln!(
        "CT {} × {} × {} at {:.2} × {:.2} × {:.2} mm",
        vol.dims[0], vol.dims[1], vol.dims[2], vol.spacing[0], vol.spacing[1], vol.spacing[2]
    );
    let frame = frame
        .or_else(|| {
            study
                .series
                .get(study.active_series)
                .and_then(loader::series_patient_position)
                .map(PatientFrame::Position)
        })
        .unwrap_or_default();
    if bench_only {
        return bench(vol, iso.unwrap_or(centre), req.device);
    }
    // A clip box about a structure is also where the isocentre goes when
    // none is given.
    let structure_box = match &clip_structure {
        Some(n) => {
            let b = setup::structure_box(&study, n, margin)?;
            eprintln!(
                "Clip box about {n}: centre ({:.1}, {:.1}, {:.1}), half ({:.1}, {:.1}, {:.1}) mm",
                b.center.x, b.center.y, b.center.z, b.half.x, b.half.y, b.half.z
            );
            Some(b)
        }
        None => None,
    };
    let iso = iso.or(structure_box.map(|b| b.center));
    let other = match &drr_of {
        Some(d) => Some(loader::load_directory(d, &Progress::default())?),
        None => None,
    };
    if compare_3d && other.is_none() {
        bail!("--compare-3d compares with the CT of --drr-of");
    }

    let mut truth: Option<RigidTransform> = None;
    if simulate.is_some() || other.is_some() {
        let s = simulate.clone().unwrap_or_else(|| vec![0.0; 6]);
        if s.len() < 6 {
            bail!("--simulate tx,ty,tz,rx,ry,rz[,noise]");
        }
        let iso = iso.unwrap_or(centre);
        let noise = s.get(6).copied().unwrap_or(0.02) as f64;
        // The patient displaced by D: the anatomy at room point x is the
        // CT's at D⁻¹(x), and the registration should answer T = D⁻¹.
        let d = RigidTransform::new(
            [
                (s[3] as f64).to_radians(),
                (s[4] as f64).to_radians(),
                (s[5] as f64).to_radians(),
                s[0] as f64,
                s[1] as f64,
                s[2] as f64,
            ],
            iso,
        );
        let inv = |p: Vec3| d.unmap(p);
        let source = other.as_ref().map(|o| &o.volume).unwrap_or(vol);
        let mu = MuVolume::prepare(source, req.hu, req.threshold_hu, None);
        let mut rng = Noise(0x2D3D);
        for &g in &gantries {
            let geometry = Geometry {
                gantry_deg: g,
                isocenter: iso,
                patient: frame,
                dims: [384, 384],
                panel_mm: [400.0, 400.0],
                ..Geometry::default()
            };
            let cam = geometry.camera();
            let mut px = render_posed(&mu, &cam.mapped(inv), 0.5, &Quiet)?;
            let (lo, hi) = px
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let sd = noise * (hi - lo) as f64;
            for v in &mut px {
                *v += (sd * rng.gauss()) as f32;
            }
            if let Some(dir) = &out {
                // The radiograph as an RT Image placed as planned: load it
                // with the CT to try the viewer's section on it.
                std::fs::create_dir_all(dir)?;
                let (lo, hi) = px
                    .iter()
                    .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                let img = PlanarImage {
                    label: format!("Simulated G{g:.0}"),
                    modality: "RTIMAGE".into(),
                    rows: cam.dims[1],
                    cols: cam.dims[0],
                    spacing: [400.0 / 384.0; 2],
                    data: px.clone(),
                    min_value: lo,
                    max_value: hi,
                    window: ((lo + hi) / 2.0, (hi - lo).max(1e-6)),
                    frame_of_reference_uid: vol.frame_of_reference_uid.clone(),
                    iec: Some(geometry),
                    camera: Some(cam),
                    frame_count: 1,
                    ..Default::default()
                };
                let meta = ObjectMeta::of_study(&study, "SIM", "simulated radiograph");
                write_rt_image(&dir.join(format!("sim_G{g:.0}.dcm")), &img, &meta)?;
            }
            req.views.push(View2d {
                label: format!("gantry {g:.0}"),
                image: Image2d::new(px, cam.dims),
                camera: cam,
                mask: None,
                weight: 1.0,
                transmission: false,
            });
        }
        req.isocenter = iso;
        // Another CT's anatomy has no known answer.
        truth = other.is_none().then_some(d);
    } else {
        if images.is_empty() {
            bail!("no radiographs: give image files, or --simulate");
        }
        let rooms = match &room {
            Some(p) => Some(rooms::load(p)?),
            None => None,
        };
        let mut iec_next = iecs.into_iter();
        let mut view_next = views.into_iter();
        let mut iso_found: Option<Vec3> = None;
        for path in &images {
            let img = extras::load_planar(path)?;
            let (camera, label) = if let Some(cam) = img.camera {
                if let Some(g) = &img.iec {
                    iso_found.get_or_insert(g.isocenter);
                }
                (cam, img.label.clone())
            } else if let Some(p) = iec_next.next() {
                if p.len() != 4 {
                    bail!("--iec sad,sid,gantry,couch");
                }
                let g = Geometry {
                    sad: p[0] as f64,
                    sid: p[1] as f64,
                    gantry_deg: p[2] as f64,
                    couch_deg: p[3] as f64,
                    isocenter: iso.unwrap_or(centre),
                    patient: frame,
                    dims: [img.cols, img.rows],
                    panel_mm: [
                        img.cols as f64 * img.spacing[0],
                        img.rows as f64 * img.spacing[1],
                    ],
                    ..Geometry::default()
                };
                (g.camera(), format!("{} (gantry {:.0})", img.label, p[2]))
            } else if let (Some(f), Some(name)) = (&rooms, view_next.next()) {
                let (r, v) = f
                    .rooms
                    .iter()
                    .find_map(|r| r.views.iter().find(|v| v.name == name).map(|v| (r, v)))
                    .ok_or_else(|| anyhow!("no view {name:?} in the rooms file"))?;
                let cam = v
                    .camera_in_patient(&r.patient_frame(), iso.unwrap_or(centre), 0.0, 0.0, 0.0)
                    .ok_or_else(|| anyhow!("view {name:?} has no usable calibration"))?;
                (cam, format!("{} ({name})", img.label))
            } else {
                bail!(
                    "{}: no geometry ({}); give --iec or --room/--view",
                    path.display(),
                    img.geometry_missing.as_deref().unwrap_or("none stated")
                );
            };
            if camera.dims != [img.cols, img.rows] {
                bail!("{}: the geometry is not the image's size", path.display());
            }
            req.views.push(View2d {
                label,
                image: Image2d::new(img.data.clone(), [img.cols, img.rows]),
                camera,
                mask: None,
                weight: 1.0,
                transmission,
            });
        }
        req.isocenter = iso.or(iso_found).unwrap_or(centre);
    }
    req.patient = frame;
    req.clip = Some(
        structure_box.unwrap_or_else(|| ClipBox::around(req.isocenter, clip_half.unwrap_or(100.0))),
    );

    let res = register_2d3d(vol, &req, &Stderr)?;
    print!("{}", res.report());
    if let (true, Some(o)) = (compare_3d, &other) {
        compare_in_3d(
            vol,
            &o.volume,
            req.clip.as_ref().unwrap(),
            req.threshold_hu,
            &res.transform,
        )?;
    }
    if let Some(d) = &truth {
        // The answer against D⁻¹, at the corners of the clip box.
        let (lo, hi) = req.clip.unwrap().corners();
        let mut worst = 0.0f64;
        for c in 0..8 {
            let p = Vec3::new(
                if c & 1 == 0 { lo.x } else { hi.x },
                if c & 2 == 0 { lo.y } else { hi.y },
                if c & 4 == 0 { lo.z } else { hi.z },
            );
            worst = worst.max((res.transform.map(p) - d.unmap(p)).length());
        }
        println!("Error at the clip box's corners: {worst:.2} mm at most");
    }
    if let Some(dir) = &out {
        std::fs::create_dir_all(dir)?;
        for (k, v) in res.views.iter().enumerate() {
            png(&dir.join(format!("view{}_image.png", k + 1)), &v.image)?;
            png(&dir.join(format!("view{}_drr.png", k + 1)), &v.drr)?;
        }
        let json = serde_json::json!({
            "correction_lps": res.correction(),
            "displacement_lps": res.displacement,
            "couch": {
                "lateral_mm": res.couch.lateral,
                "longitudinal_mm": res.couch.longitudinal,
                "vertical_mm": res.couch.vertical,
                "yaw_deg": res.couch.yaw,
                "pitch_deg": res.couch.pitch,
                "roll_deg": res.couch.roll,
            },
            "metric": res.metric.key(),
            "views": res.views.iter().map(|v| serde_json::json!({
                "label": v.label, "before": v.before, "after": v.after, "inverted": v.inverted,
            })).collect::<Vec<_>>(),
            "start": res.start,
            "evaluations": res.evaluations,
            "seconds": res.elapsed_secs,
            "warning": res.warning,
        });
        std::fs::write(
            dir.join("report.json"),
            serde_json::to_string_pretty(&json)?,
        )?;
        eprintln!("wrote {}", dir.display());
    }
    Ok(())
}
