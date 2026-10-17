//! 2D/3D registration against radiographs of a known pose.
//!
//! Run with `--release` (every test registers; the dev profile is several
//! times slower).
//!
//! The radiographs are not the CT rendered through a moved camera - that
//! would be the engine's own model handed back to it. They are renderings
//! of the phantom *function* sampled afresh in the room at the true pose
//! (another grid, another interpolation), at a finer ray step, with
//! Gaussian noise: what the engine has to explain is a different
//! discretisation of the same anatomy, as a real radiograph is.

use rust_dicom_station::drr::{Geometry, HuMode};
use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::iec::{PatientFrame, PatientPosition};
use rust_dicom_station::nn::device::DevicePref;
use rust_dicom_station::progress::Quiet;
use rust_dicom_station::projection::{render_posed, MuVolume, ProjectionGeometry};
use rust_dicom_station::registration::metric2d::{ncc, Image2d, Metric2d};
use rust_dicom_station::registration::projective::{
    register_2d3d, ClipBox, Dof2d3d, Init2d3d, Request2d3d, Result2d3d, View2d,
};
use rust_dicom_station::registration::RigidTransform;
use rust_dicom_station::volume::Volume;

/// A torso-like phantom in HU: a water body, two lungs, a spine and a few
/// bones placed so that no rotation or reflection maps it onto itself.
fn phantom(p: Vec3) -> f32 {
    fn inside(d: f64, edge: f64) -> f64 {
        // 1 well inside (d < 0), 0 well outside, smooth over `edge` mm.
        (0.5 - d / edge).clamp(0.0, 1.0)
    }
    let ell = |c: Vec3, r: Vec3| {
        let q = p - c;
        ((q.x / r.x).powi(2) + (q.y / r.y).powi(2) + (q.z / r.z).powi(2)).sqrt() - 1.0
    };
    let body = inside(ell(Vec3::ZERO, Vec3::new(85.0, 65.0, 95.0)) * 70.0, 4.0);
    if body <= 0.0 {
        return -1000.0;
    }
    let mut v = -1000.0 + 1000.0 * body;
    v -= 750.0
        * inside(
            ell(Vec3::new(-35.0, -5.0, 15.0), Vec3::new(28.0, 32.0, 50.0)) * 30.0,
            4.0,
        );
    v -= 750.0
        * inside(
            ell(Vec3::new(38.0, -2.0, 20.0), Vec3::new(25.0, 30.0, 45.0)) * 28.0,
            4.0,
        );
    // Spine: a cylinder along z, posterior.
    let spine = ((p.x).powi(2) + (p.y - 40.0).powi(2)).sqrt() - 11.0;
    v += 700.0 * inside(spine, 2.0) * inside(p.z.abs() - 85.0, 3.0);
    v += 600.0
        * inside(
            ell(Vec3::new(50.0, 15.0, -45.0), Vec3::new(9.0, 9.0, 14.0)) * 9.0,
            2.0,
        );
    v += 500.0
        * inside(
            ell(Vec3::new(-45.0, -30.0, -55.0), Vec3::new(14.0, 8.0, 8.0)) * 8.0,
            2.0,
        );
    v += 450.0
        * inside(
            ell(Vec3::new(10.0, -45.0, 50.0), Vec3::new(30.0, 5.0, 6.0)) * 5.0,
            2.0,
        );
    v as f32
}

/// An axis-aligned volume of `f` centred on the origin.
fn volume_of(n: usize, spacing: f64, f: impl Fn(Vec3) -> f32 + Sync) -> Volume {
    let half = (n as f64 - 1.0) * 0.5 * spacing;
    let origin = Vec3::new(-half, -half, -half);
    let mut data = vec![0i16; n * n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as f64, j as f64, k as f64) * spacing;
                data[k * n * n + j * n + i] = f(p).round().clamp(-32768.0, 32767.0) as i16;
            }
        }
    }
    Volume {
        data,
        dims: [n, n, n],
        spacing: [spacing; 3],
        origin,
        row_dir: Vec3::new(1.0, 0.0, 0.0),
        col_dir: Vec3::new(0.0, 1.0, 0.0),
        normal: Vec3::new(0.0, 0.0, 1.0),
        frame_of_reference_uid: String::new(),
        min_value: -1000,
        max_value: 1000,
    }
}

fn ct() -> Volume {
    volume_of(80, 3.0, phantom)
}

/// The camera of a gantry angle for a patient lying as `frame` says.
fn camera(gantry_deg: f64, frame: PatientFrame) -> ProjectionGeometry {
    Geometry {
        gantry_deg,
        patient: frame,
        panel_mm: [330.0, 330.0],
        dims: [96, 96],
        ..Geometry::default()
    }
    .camera()
}

/// Deterministic Gaussian noise (a 64-bit LCG through Box-Muller).
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Radiographs of the patient found at pose `truth` (planned frame → CT),
/// one per camera, with noise at `noise` of each image's range.
fn radiographs(
    truth: &RigidTransform,
    cams: &[(&str, ProjectionGeometry)],
    noise: f64,
) -> Vec<View2d> {
    // The anatomy at room point x is the CT's at truth(x), sampled on a
    // grid of its own.
    let room = volume_of(100, 2.5, |p| phantom(truth.map(p)));
    let mu = MuVolume::prepare(&room, HuMode::Water, -800.0, None);
    let mut rng = Noise(0x2D3D);
    cams.iter()
        .map(|(label, cam)| {
            let mut px = render_posed(&mu, cam, 0.5, &Quiet).unwrap();
            let (lo, hi) = px
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let sd = noise * (hi - lo) as f64;
            for v in &mut px {
                *v += (sd * rng.gauss()) as f32;
            }
            View2d {
                label: label.to_string(),
                image: Image2d::new(px, cam.dims),
                camera: *cam,
                mask: None,
                weight: 1.0,
                transmission: false,
            }
        })
        .collect()
}

fn request(views: Vec<View2d>) -> Request2d3d {
    let mut req = Request2d3d::new(views, Vec3::ZERO);
    req.clip = Some(ClipBox::around(Vec3::ZERO, 100.0));
    req.clip_rays = true;
    req.max_pixels = 96;
    req.levels = 3;
    req.step_mm = 1.5;
    req
}

/// Rotation error (degrees) and translation error (mm) of `got` against
/// `want`, both about the isocentre: the largest per-axis difference.
fn errors(got: &Result2d3d, want: &RigidTransform) -> (f64, f64) {
    let g = got.transform.params();
    let w = want.params();
    let rot = (0..3)
        .map(|k| (g[k] - w[k]).abs().to_degrees())
        .fold(0.0, f64::max);
    let tr = (3..6).map(|k| (g[k] - w[k]).abs()).fold(0.0, f64::max);
    (rot, tr)
}

fn ap_and_lateral(frame: PatientFrame) -> Vec<(&'static str, ProjectionGeometry)> {
    vec![("AP", camera(0.0, frame)), ("Lateral", camera(90.0, frame))]
}

#[test]
fn a_transformed_camera_sees_the_transformed_study() {
    let t = RigidTransform::new([0.05, -0.03, 0.04, 6.0, -4.0, 3.0], Vec3::ZERO);
    let cam = camera(30.0, PatientFrame::default());
    let ct = ct();
    let mu = MuVolume::prepare(&ct, HuMode::Water, -800.0, None);
    let posed = render_posed(&mu, &cam.transformed(&t), 1.0, &Quiet).unwrap();
    let moved = volume_of(80, 3.0, |p| phantom(t.map(p)));
    let mu_moved = MuVolume::prepare(&moved, HuMode::Water, -800.0, None);
    let direct = render_posed(&mu_moved, &cam, 1.0, &Quiet).unwrap();
    let r = ncc(&posed, &direct, None);
    assert!(r > 0.999, "NCC {r}");
}

#[test]
fn two_views_recover_a_known_pose() {
    let truth = RigidTransform::new(
        [
            3f64.to_radians(),
            -2f64.to_radians(),
            2.5f64.to_radians(),
            6.0,
            -4.0,
            5.0,
        ],
        Vec3::ZERO,
    );
    let views = radiographs(&truth, &ap_and_lateral(PatientFrame::default()), 0.02);
    let ct = ct();
    for metric in [Metric2d::Ncc, Metric2d::GradientCorrelation] {
        let mut req = request(views.clone());
        req.metric = metric;
        let res = register_2d3d(&ct, &req, &Quiet).unwrap();
        let (rot, tr) = errors(&res, &truth);
        eprintln!("{}", res.report());
        assert!(
            tr < 0.5 && rot < 0.3,
            "{}: {tr:.3} mm, {rot:.3}°",
            metric.tag()
        );
        assert_eq!(
            res.warning,
            None,
            "{}: curvature {:?}",
            metric.tag(),
            res.curvature
        );
        for v in &res.views {
            assert!(
                v.after > v.before,
                "{}: {} ▶ {}",
                v.label,
                v.before,
                v.after
            );
            assert!(!v.inverted);
        }
    }
}

/// The patient found 5 mm to his left: the displacement says +5 mm in x,
/// the correction −5 mm, and the couch moves by what the patient's
/// mounting on the table top makes of that.
#[test]
fn a_shift_to_the_left_reads_with_the_right_signs() {
    let truth = RigidTransform::new([0.0, 0.0, 0.0, -5.0, 0.0, 0.0], Vec3::ZERO);
    let ct = ct();
    let cases = [
        (PatientFrame::Position(PatientPosition::Hfs), -5.0),
        (PatientFrame::Position(PatientPosition::Ffp), -5.0),
        (PatientFrame::Position(PatientPosition::Ffs), 5.0),
        (PatientFrame::Upright { facing_deg: 0.0 }, 5.0),
    ];
    for (frame, lateral) in cases {
        let views = radiographs(&truth, &ap_and_lateral(frame), 0.01);
        let mut req = request(views);
        req.patient = frame;
        req.dof = Dof2d3d::Translations;
        req.conditioning = false;
        req.levels = 2;
        let res = register_2d3d(&ct, &req, &Quiet).unwrap();
        let d = res.displacement;
        assert!(
            (d[3] - 5.0).abs() < 0.4 && d[4].abs() < 0.4 && d[5].abs() < 0.4,
            "{}: {d:?}",
            frame.describe()
        );
        let c = res.correction();
        assert!((c[3] + 5.0).abs() < 0.4, "{}: {c:?}", frame.describe());
        assert!(
            (res.couch.lateral - lateral).abs() < 0.4,
            "{}: {}",
            frame.describe(),
            res.couch.line()
        );
        assert!(res.couch.longitudinal.abs() < 0.4 && res.couch.vertical.abs() < 0.4);
    }
}

#[test]
fn one_view_finds_the_plane_and_says_depth_is_weak() {
    let truth = RigidTransform::new([0.0, 0.0, 0.0, 4.0, 0.0, -3.0], Vec3::ZERO);
    let ap = vec![("AP", camera(0.0, PatientFrame::default()))];
    let views = radiographs(&truth, &ap, 0.01);
    let ct = ct();
    let req = request(views);
    let res = register_2d3d(&ct, &req, &Quiet).unwrap();
    let c = res.correction();
    eprintln!("{}", res.report());
    // Gantry 0, supine: the beam runs along A-P (y); the translations
    // across it and the rotation about it are what one view sees.
    assert!(
        (c[3] - 4.0).abs() < 0.5 && (c[5] + 3.0).abs() < 0.5,
        "{c:?}"
    );
    assert!(c[1].abs() < 0.3, "{c:?}");
    let w = res.warning.clone().unwrap_or_default();
    assert!(
        w.contains("translation A-P"),
        "{w:?}, curvature {:?}",
        res.curvature
    );
}

#[test]
fn the_in_plane_search_brings_a_far_start_within_reach() {
    let truth = RigidTransform::new(
        [
            2f64.to_radians(),
            -3f64.to_radians(),
            4f64.to_radians(),
            12.0,
            -14.0,
            10.0,
        ],
        Vec3::ZERO,
    );
    let views = radiographs(&truth, &ap_and_lateral(PatientFrame::default()), 0.02);
    let ct = ct();
    let mut req = request(views);
    req.conditioning = false;
    req.init = Init2d3d::InPlane;
    let res = register_2d3d(&ct, &req, &Quiet).unwrap();
    let (rot, tr) = errors(&res, &truth);
    assert!(
        tr < 0.5 && rot < 0.3,
        "{tr:.3} mm, {rot:.3}°, {}",
        res.report()
    );
    assert!(res.start.contains("in-plane"), "{}", res.start);
}

#[test]
fn the_grid_brings_a_farther_start_within_reach() {
    let truth = RigidTransform::new([0.0, 0.0, 0.0, 22.0, -24.0, 12.0], Vec3::ZERO);
    let views = radiographs(&truth, &ap_and_lateral(PatientFrame::default()), 0.02);
    let ct = ct();
    let mut req = request(views);
    req.conditioning = false;
    req.init = Init2d3d::Grid;
    req.range_mm = 40.0;
    req.dof = Dof2d3d::Translations;
    let res = register_2d3d(&ct, &req, &Quiet).unwrap();
    let (_, tr) = errors(&res, &truth);
    assert!(tr < 0.5, "{tr:.3} mm, {}", res.report());
}

#[test]
fn a_radiograph_stored_bone_dark_registers_as_well() {
    let truth = RigidTransform::new([0.0, 0.0, 0.0, 3.0, 2.0, -4.0], Vec3::ZERO);
    let mut views = radiographs(&truth, &ap_and_lateral(PatientFrame::default()), 0.01);
    for v in &mut views {
        for p in &mut v.image.data {
            *p = 1000.0 - *p;
        }
    }
    let ct = ct();
    let mut req = request(views);
    req.dof = Dof2d3d::Translations;
    req.conditioning = false;
    let res = register_2d3d(&ct, &req, &Quiet).unwrap();
    assert!(res.views.iter().all(|v| v.inverted));
    let (_, tr) = errors(&res, &truth);
    assert!(tr < 0.5, "{tr:.3} mm");
}

#[test]
fn the_answer_does_not_depend_on_the_thread_count() {
    let truth = RigidTransform::new([0.02, -0.01, 0.03, 3.0, -2.0, 4.0], Vec3::ZERO);
    let views = radiographs(&truth, &ap_and_lateral(PatientFrame::default()), 0.02);
    let ct = ct();
    let mut req = request(views);
    req.levels = 2;
    req.conditioning = false;
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| register_2d3d(&ct, &req, &Quiet).unwrap())
    };
    let one = run(1);
    for threads in [2, 8] {
        let other = run(threads);
        assert_eq!(
            one.transform.params(),
            other.transform.params(),
            "{threads} threads"
        );
        assert_eq!(one.evaluations, other.evaluations);
    }
}

// ---------------------------------------------------------------------------
// Markers
// ---------------------------------------------------------------------------

use rust_dicom_station::registration::markers::{
    detect_3d, register_hybrid, register_markers, Detect3d, Marker3d, MarkerRequest,
};

const SEEDS: [(f64, f64, f64); 3] = [(12.0, -8.0, 18.0), (-15.0, 6.0, -6.0), (4.0, 14.0, -20.0)];

/// The phantom with gold seeds: Gaussian blobs of 1 mm, dense enough to
/// stand out of the bone at any sampling.
fn seeded(p: Vec3, seeds: &[(f64, f64, f64)]) -> f32 {
    let mut v = phantom(p) as f64;
    for &(x, y, z) in seeds {
        let d2 = (p - Vec3::new(x, y, z)).length().powi(2);
        v += 15000.0 * (-d2 / 2.0).exp();
    }
    v.min(30000.0) as f32
}

fn seeded_ct() -> Volume {
    volume_of(100, 2.0, |p| seeded(p, &SEEDS))
}

fn seed_markers() -> Vec<Marker3d> {
    SEEDS
        .iter()
        .enumerate()
        .map(|(k, &(x, y, z))| Marker3d {
            name: format!("Seed {}", k + 1),
            position: Vec3::new(x, y, z),
        })
        .collect()
}

/// Radiographs of the seeded patient at pose `truth`, 256 pixels, the
/// seeds in the room at `seeds` (the CT's, unless one moved).
fn seeded_radiographs(truth: &RigidTransform, seeds: &[(f64, f64, f64)]) -> Vec<View2d> {
    let room = volume_of(110, 1.8, |p| seeded(truth.map(p), seeds));
    let mu = MuVolume::prepare(&room, HuMode::Water, -800.0, None);
    let mut rng = Noise(0x5EED);
    ["AP", "Lateral"]
        .iter()
        .zip([0.0, 90.0])
        .map(|(label, g)| {
            let cam = Geometry {
                gantry_deg: g,
                panel_mm: [330.0, 330.0],
                dims: [256, 256],
                ..Geometry::default()
            }
            .camera();
            let mut px = render_posed(&mu, &cam, 0.5, &Quiet).unwrap();
            let (lo, hi) = px
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            for v in &mut px {
                *v += (0.005 * (hi - lo) as f64 * rng.gauss()) as f32;
            }
            View2d {
                label: label.to_string(),
                image: Image2d::new(px, cam.dims),
                camera: cam,
                mask: None,
                weight: 1.0,
                transmission: false,
            }
        })
        .collect()
}

#[test]
fn seeds_are_found_in_the_ct() {
    let found = detect_3d(&seeded_ct(), &Detect3d::default());
    assert_eq!(found.len(), 3, "{found:?}");
    for &(x, y, z) in &SEEDS {
        let want = Vec3::new(x, y, z);
        let d = found
            .iter()
            .map(|m| (m.position - want).length())
            .fold(f64::MAX, f64::min);
        assert!(d < 0.6, "{want:?}: nearest {d:.2} mm");
    }
}

#[test]
fn seeds_in_two_views_give_the_pose() {
    let truth = RigidTransform::new(
        [
            2f64.to_radians(),
            -1.5f64.to_radians(),
            3f64.to_radians(),
            4.0,
            -3.0,
            5.0,
        ],
        Vec3::ZERO,
    );
    let views = seeded_radiographs(&truth, &SEEDS);
    let ct = seeded_ct();
    let req = MarkerRequest::new(seed_markers(), views, Vec3::ZERO);
    let res = register_markers(&ct, &req, &Quiet).unwrap();
    eprintln!("{}", res.report());
    // Three seeds within 25 mm of each other pin a rotation to a few
    // tenths of a degree; what matters is where they put the target. The
    // radiographs' seeds are sampled on a grid of their own, which moves
    // their centres by a tenth of a millimetre or two.
    let worst = SEEDS
        .iter()
        .map(|&(x, y, z)| {
            let p = truth.unmap(Vec3::new(x, y, z));
            (res.transform.map(p) - truth.map(p)).length()
        })
        .fold(0.0, f64::max);
    let (rot, _) = errors(&res, &truth);
    assert!(
        worst < 0.4 && rot < 1.0,
        "{worst:.3} mm at the seeds, {rot:.3}°"
    );
    let m = res.markers.as_ref().unwrap();
    assert_eq!(m.residuals.len(), 6);
    assert!(m.rms_mm < 0.3, "{}", m.rms_mm);
    assert_eq!(m.triangulated.len(), 3);
    assert!(m
        .triangulated
        .iter()
        .all(|t| !t.migrated && t.offset_mm < 0.5));
    assert_eq!(res.warning, None);
}

#[test]
fn one_seed_gives_the_translations() {
    let truth = RigidTransform::new([0.0, 0.0, 0.0, -6.0, 4.0, 3.0], Vec3::ZERO);
    let views = seeded_radiographs(&truth, &SEEDS);
    let ct = seeded_ct();
    let req = MarkerRequest::new(seed_markers()[..1].to_vec(), views, Vec3::ZERO);
    let res = register_markers(&ct, &req, &Quiet).unwrap();
    let m = res.markers.as_ref().unwrap();
    assert_eq!(m.dof, Dof2d3d::Translations);
    let (_, tr) = errors(&res, &truth);
    assert!(tr < 0.3, "{tr:.3} mm");
}

#[test]
fn a_seed_that_moved_is_reported_and_outvoted() {
    let truth = RigidTransform::new([0.0, 0.0, 0.0, 3.0, -2.0, 4.0], Vec3::ZERO);
    // Seed 2 sits 6 mm further right in the patient than the CT has it.
    let mut moved = SEEDS;
    moved[1].0 -= 6.0;
    // Four seeds, so three agree.
    let mut seeds = moved.to_vec();
    seeds.push((-6.0, -12.0, 25.0));
    let views = seeded_radiographs(&truth, &seeds);
    let mut ct_seeds = SEEDS.to_vec();
    ct_seeds.push((-6.0, -12.0, 25.0));
    let ct = volume_of(100, 2.0, |p| seeded(p, &ct_seeds));
    let markers: Vec<Marker3d> = ct_seeds
        .iter()
        .enumerate()
        .map(|(k, &(x, y, z))| Marker3d {
            name: format!("Seed {}", k + 1),
            position: Vec3::new(x, y, z),
        })
        .collect();
    let req = MarkerRequest::new(markers, views, Vec3::ZERO);
    let res = register_markers(&ct, &req, &Quiet).unwrap();
    eprintln!("{}", res.report());
    let m = res.markers.as_ref().unwrap();
    let flagged: Vec<&str> = m
        .triangulated
        .iter()
        .filter(|t| t.migrated)
        .map(|t| t.marker.as_str())
        .collect();
    assert_eq!(flagged, vec!["Seed 2"], "{}", m.report());
    assert!(res.warning.as_deref().unwrap_or("").contains("Seed 2"));
    let (_, tr) = errors(&res, &truth);
    assert!(tr < 1.0, "{tr:.3} mm");
}

#[test]
fn the_seeds_bring_the_intensity_run_in_from_far() {
    let truth = RigidTransform::new([0.0, 1f64.to_radians(), 0.0, 25.0, -12.0, 8.0], Vec3::ZERO);
    let views = seeded_radiographs(&truth, &SEEDS);
    let ct = seeded_ct();
    let markers = MarkerRequest::new(seed_markers(), views.clone(), Vec3::ZERO);
    let mut intensity = request(views);
    intensity.init = Init2d3d::Plan;
    intensity.conditioning = false;
    intensity.max_pixels = 128;
    let res = register_hybrid(&ct, &markers, &intensity, &Quiet).unwrap();
    eprintln!("{}", res.report());
    let (rot, tr) = errors(&res, &truth);
    assert!(tr < 0.5 && rot < 0.3, "{tr:.3} mm, {rot:.3}°");
    assert!(res.markers.is_some());
    assert_eq!(res.start, "the markers' pose");
}

// ---------------------------------------------------------------------------
// DICOM: the RT Image and the rigid registration written and read back
// ---------------------------------------------------------------------------

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rds_2d3d_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn an_rt_image_written_reads_back_with_its_camera() {
    use rust_dicom_station::dicom_export::{write_rt_image, ObjectMeta};
    use rust_dicom_station::drr::{self, DrrParams, Engine};
    let ct = ct();
    let mut params = DrrParams::for_volume(&ct);
    params.engine = Engine::RayCast;
    params.geometry = Geometry {
        gantry_deg: 35.0,
        couch_deg: 10.0,
        pitch_deg: 1.0,
        roll_deg: -2.0,
        isocenter: Vec3::new(5.0, -3.0, 8.0),
        receptor: Some(Vec3::new(4.0, -6.0, -480.0)),
        receptor_angle_deg: 3.0,
        image_center: [2.0, -5.0],
        panel_mm: [200.0, 160.0],
        dims: [50, 40],
        patient: PatientFrame::Position(PatientPosition::Ffs),
        ..Geometry::default()
    };
    let img = drr::render(&ct, &params, &Quiet)
        .unwrap()
        .to_planar(&params, false);
    let want = img.camera.expect("a DRR knows its camera");
    let dir = scratch("rtimage");
    let path = dir.join("drr.dcm");
    write_rt_image(
        &path,
        &img,
        &ObjectMeta {
            label: "DRR".into(),
            ..Default::default()
        },
    )
    .unwrap();
    let back = rust_dicom_station::extras::load_planar(&path).unwrap();
    let got = back.camera.expect("the camera reads back");
    for (a, b) in [
        (got.source, want.source),
        (got.origin, want.origin),
        (got.row_step, want.row_step),
        (got.col_step, want.col_step),
    ] {
        assert!((a - b).length() < 1e-6, "{a:?} vs {b:?}");
    }
    assert_eq!(got.dims, want.dims);
    let span = img.max_value - img.min_value;
    let worst = img
        .data
        .iter()
        .zip(&back.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= span / 65535.0 + 1e-6, "{worst} of {span}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_rigid_registration_written_reads_back_as_the_same_transform() {
    use rust_dicom_station::dicom_export::{write_rigid_registration, ObjectMeta};
    use rust_dicom_station::registration::Mat4;
    let t = RigidTransform::new(
        [0.03, -0.02, 0.05, 4.0, -2.0, 6.0],
        Vec3::new(10.0, 20.0, -5.0),
    );
    let m = Mat4::from_rigid(&t);
    let flat: [f64; 16] = std::array::from_fn(|k| m.0[k / 4][k % 4]);
    let dir = scratch("reg");
    let path = dir.join("reg.dcm");
    write_rigid_registration(&path, &flat, "1.2.3.4", "1.2.3.5", &ObjectMeta::default()).unwrap();
    let reg = rust_dicom_station::extras::load_reg(&path).unwrap();
    assert_eq!(reg.frame_of_reference_uid, "1.2.3.5");
    let item = reg
        .items
        .iter()
        .find(|i| i.for_uid == "1.2.3.4")
        .expect("the source frame's item");
    let back = rust_dicom_station::extras::matrix_to_rigid(&item.matrix, false).unwrap();
    for p in [Vec3::ZERO, Vec3::new(50.0, -40.0, 30.0)] {
        assert!((back.map(p) - t.map(p)).length() < 1e-6);
    }
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// The setup as the program files it
// ---------------------------------------------------------------------------

/// A study of the phantom CT with two radiographs that state their own
/// geometry, taken of the patient found at `truth`.
fn study_with_radiographs(truth: &RigidTransform) -> rust_dicom_station::loader::LoadedStudy {
    use rust_dicom_station::extras::PlanarImage;
    let ct = ct();
    let mut study = rust_dicom_station::loader::LoadedStudy {
        volume: std::sync::Arc::new(ct),
        ..Default::default()
    };
    for (label, g) in [("kV AP", 0.0), ("kV LAT", 90.0)] {
        let geometry = Geometry {
            gantry_deg: g,
            panel_mm: [330.0, 330.0],
            dims: [96, 96],
            ..Geometry::default()
        };
        let cam = geometry.camera();
        let view = radiographs(truth, &[(label, cam)], 0.01).remove(0);
        let (lo, hi) = view
            .image
            .data
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        study.planar_images.push(PlanarImage {
            label: label.into(),
            modality: "RTIMAGE".into(),
            rows: 96,
            cols: 96,
            spacing: [330.0 / 96.0; 2],
            data: view.image.data,
            min_value: lo,
            max_value: hi,
            window: ((lo + hi) / 2.0, hi - lo),
            iec: Some(geometry),
            camera: Some(cam),
            frame_count: 1,
            ..Default::default()
        });
    }
    study
}

#[test]
fn a_setup_run_from_the_study_files_what_it_found() {
    use rust_dicom_station::setup::{self, GeometrySource, SetupOptions, ViewChoice};
    let truth = RigidTransform::new([0.0, 0.0, 0.03, 5.0, -3.0, 2.0], Vec3::ZERO);
    let study = study_with_radiographs(&truth);
    let views = vec![
        ViewChoice::new(0, GeometrySource::Image),
        ViewChoice::new(1, GeometrySource::Image),
    ];
    let opts = SetupOptions {
        metric: Metric2d::Ncc,
        max_pixels: 96,
        step_mm: 1.5,
        conditioning: false,
        device: DevicePref::Cpu,
        ..SetupOptions::default()
    };
    let s = setup::run(&study, &views, &opts, None, &Quiet).unwrap();
    let (rot, tr) = errors(&s.result, &truth);
    assert!(tr < 0.5 && rot < 0.3, "{tr:.3} mm, {rot:.3}°");
    assert_eq!(
        s.image_labels,
        vec!["kV AP".to_string(), "kV LAT".to_string()]
    );
    assert_eq!(s.isocenter, Vec3::ZERO);
    let j = s.json();
    assert_eq!(j["views"].as_array().unwrap().len(), 2);
    // The DRR at the answer sits where the radiograph does.
    let drr = s.drr_planar(0, study.planar_images.first()).unwrap();
    assert_eq!((drr.cols, drr.rows), (96, 96));
    assert!((drr.spacing[0] - study.planar_images[0].spacing[0]).abs() < 1e-9);

    // The positioned copy is the CT where the patient was found: its value
    // at a room point is the CT's at T of it.
    let copy = setup::positioned_copy(
        &study,
        &s,
        &rust_dicom_station::progress::Progress::default(),
    );
    let mut worst = 0.0f32;
    for p in [
        Vec3::new(10.0, -20.0, 15.0),
        Vec3::new(-30.0, 5.0, -25.0),
        Vec3::new(0.0, 40.0, 0.0),
    ] {
        let a = copy.volume.sample_patient(p).unwrap();
        let b = study
            .volume
            .sample_patient(s.result.transform.map(p))
            .unwrap();
        worst = worst.max((a - b).abs());
    }
    assert!(worst < 25.0, "{worst} HU");
}

// ---------------------------------------------------------------------------
// MV portal images
// ---------------------------------------------------------------------------

/// A portal image of the phantom's patient found 2 mm to his left, through
/// a field delivered 3 mm further left than planned: the registration finds
/// the anatomy, the field edge the field, and the report the field 1 mm off
/// the anatomy.
#[test]
fn a_portal_image_gives_the_anatomy_and_the_field() {
    use rust_dicom_station::drr::HuMode;
    use rust_dicom_station::extras::PlanarImage;
    use rust_dicom_station::rtplan::{Aperture, BeamInfo, PlanInfo};
    use rust_dicom_station::setup::{self, GeometrySource, SetupOptions, ViewChoice};
    let planned = Aperture {
        jaw_x: Some([-50.0, 50.0]),
        jaw_y: Some([-45.0, 45.0]),
        mlc: None,
    };
    let geometry = Geometry {
        panel_mm: [330.0, 330.0],
        dims: [128, 128],
        ..Geometry::default()
    };
    let cam = geometry.camera();
    // The anatomy: the patient 2 mm to his left, so the room point x shows
    // the CT's x - 2.
    let truth = RigidTransform::new([0.0, 0.0, 0.0, -2.0, 0.0, 0.0], Vec3::ZERO);
    let room = volume_of(100, 2.5, |p| phantom(truth.map(p)));
    let mu = MuVolume::prepare(&room, HuMode::MegaVoltage, -800.0, None);
    let li = render_posed(&mu, &cam, 0.5, &Quiet).unwrap();
    // The field: delivered 3 mm towards +Xb (the patient's left at gantry
    // 0), with a penumbra; 3 % leakage outside.
    let axes = geometry.bld_axes();
    let ramp = |u: f64| 1.0 / (1.0 + (-u / 1.5).exp());
    let mut rng = Noise(0xE1D);
    let [w, h] = cam.dims;
    let signal: Vec<f32> = (0..w * h)
        .map(|k| {
            let d = cam.pixel((k % w) as f64, (k / w) as f64) - cam.source;
            let t = (Vec3::ZERO - cam.source).dot(axes[2]) / d.dot(axes[2]);
            let p = cam.source + d * t;
            let (x, y) = (p.dot(axes[0]), p.dot(axes[1]));
            let f = ramp(x - (-47.0)) * ramp(53.0 - x) * ramp(y + 45.0) * ramp(45.0 - y);
            let open = 0.03 + 0.97 * f;
            (open * (-(li[k] as f64)).exp() * (1.0 + 0.003 * rng.gauss())) as f32
        })
        .collect();
    let (lo, hi) = signal
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let mut study = rust_dicom_station::loader::LoadedStudy {
        volume: std::sync::Arc::new(ct()),
        ..Default::default()
    };
    study.plans.push(PlanInfo {
        label: "Plan".into(),
        beams: vec![BeamInfo {
            number: 1,
            name: "AP".into(),
            radiation_type: "PHOTON".into(),
            delivery_type: "TREATMENT".into(),
            scan_mode: String::new(),
            gantry_angle: Some(0.0),
            couch_angle: Some(0.0),
            isocenter: Some(Vec3::ZERO),
            energy_min: None,
            energy_max: None,
            n_control_points: 2,
            meterset: None,
            beam_dose: None,
            collimator_angle: Some(0.0),
            aperture: Some(planned),
        }],
        ..Default::default()
    });
    study.planar_images.push(PlanarImage {
        label: "EPID AP".into(),
        modality: "RTIMAGE".into(),
        rows: h,
        cols: w,
        spacing: [330.0 / 128.0; 2],
        data: signal,
        min_value: lo,
        max_value: hi,
        window: ((lo + hi) / 2.0, hi - lo),
        referenced_beam: Some(1),
        iec: Some(geometry),
        camera: Some(cam),
        megavoltage: true,
        frame_count: 1,
        ..Default::default()
    });
    let views = vec![ViewChoice {
        transmission: true,
        ..ViewChoice::new(0, GeometrySource::Image)
    }];
    let opts = SetupOptions {
        metric: Metric2d::LocalNcc,
        dof: Dof2d3d::Translations,
        max_pixels: 128,
        step_mm: 1.5,
        conditioning: false,
        device: DevicePref::Cpu,
        ..SetupOptions::default()
    };
    let s = setup::run(&study, &views, &opts, None, &Quiet).unwrap();
    eprintln!("{}", s.report());
    assert_eq!(s.hu, HuMode::MegaVoltage);
    let c = s.result.correction();
    assert!((c[3] + 2.0).abs() < 0.5 && c[5].abs() < 0.5, "{c:?}");
    let f = &s.fields[0];
    let edge = f.edge.expect("the field edge is found");
    assert!(
        (edge.shift_mm[0] - 3.0).abs() < 0.5 && edge.shift_mm[1].abs() < 0.5,
        "{edge:?}"
    );
    let rel = f.relative_mm().unwrap();
    assert!((rel[0] - 1.0).abs() < 0.7, "{}", f.line());
    assert!(!f.outline.is_empty());
}

// ---- the projector on the GPU ---------------------------------------------
//
// Ignored: CI has software adapters at best. Run with
// `RDS_GPU_TESTS=1 cargo test --release --test registration_2d3d -- --ignored gpu`.

/// The device, when `RDS_GPU_TESTS` asks for these tests.
#[cfg(feature = "gpu")]
fn gpu() -> Option<rust_dicom_station::nn::device::GpuContext> {
    std::env::var_os("RDS_GPU_TESTS")?;
    Some(
        rust_dicom_station::nn::device::GpuContext::try_new()
            .expect("RDS_GPU_TESTS is set but no GPU opens"),
    )
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a GPU and RDS_GPU_TESTS"]
fn gpu_projector_agrees_with_the_cpu() {
    use rust_dicom_station::projection::gpu::GpuProjector;
    let Some(ctx) = gpu() else { return };
    let t = RigidTransform::new([0.05, -0.03, 0.04, 6.0, -4.0, 3.0], Vec3::ZERO);
    let ct = ct();
    let levels = MuVolume::prepare(&ct, HuMode::Water, -800.0, None).levels(2);
    // A box smaller than the field, so rays miss it, graze it and cross it.
    let clipped = MuVolume::prepare(
        &ct,
        HuMode::Water,
        -800.0,
        Some(ClipBox::around(Vec3::new(10.0, 0.0, -5.0), 40.0).corners()),
    );
    for (name, mu) in [
        ("whole", &levels[0]),
        ("coarser", &levels[1]),
        ("clipped", &clipped),
    ] {
        let gpu = GpuProjector::new(&ctx, mu).unwrap();
        for gantry in [0.0, 30.0, 90.0] {
            for step in [1.0, 2.5] {
                let cam = camera(gantry, PatientFrame::default()).transformed(&t);
                let cpu = render_posed(mu, &cam, step, &Quiet).unwrap();
                let got = gpu.render(&cam, step).unwrap();
                assert_eq!(got.len(), cpu.len());
                let range = cpu.iter().fold(0f32, |a, &v| a.max(v)) as f64;
                let diff: Vec<f64> = cpu
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs() as f64 / range)
                    .collect();
                let worst = diff.iter().copied().fold(0.0, f64::max);
                let mean = diff.iter().sum::<f64>() / diff.len() as f64;
                let apart = diff.iter().filter(|&&d| d > 1e-4).count() as f64 / diff.len() as f64;
                eprintln!(
                    "{name}, gantry {gantry}, step {step}: mean {mean:.1e}, worst {worst:.1e} \
                     of the range; {:.3} % of the pixels beyond 1e-4",
                    apart * 100.0
                );
                assert!(range > 0.0);
                // A ray whose piece count the f32 set-up rounds the other way
                // moves its midpoints: that pixel differs by the midpoint
                // rule's own error, the others by rounding.
                assert!(
                    mean <= 1e-5 && apart <= 1e-3 && worst <= 5e-3,
                    "{name}, gantry {gantry}, step {step}: mean {mean}, worst {worst}, {apart}"
                );
            }
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a GPU and RDS_GPU_TESTS"]
fn gpu_registration_finds_the_pose() {
    if gpu().is_none() {
        return;
    }
    let truth = RigidTransform::new(
        [
            3f64.to_radians(),
            -2f64.to_radians(),
            2.5f64.to_radians(),
            6.0,
            -4.0,
            5.0,
        ],
        Vec3::ZERO,
    );
    let views = radiographs(&truth, &ap_and_lateral(PatientFrame::default()), 0.02);
    let ct = ct();
    let mut req = request(views);
    req.metric = Metric2d::Ncc;
    let cpu = register_2d3d(&ct, &req, &Quiet).unwrap();
    req.device = DevicePref::Gpu;
    let res = register_2d3d(&ct, &req, &Quiet).unwrap();
    eprintln!("{}", res.report());
    assert!(res.device.starts_with("GPU"), "{}", res.device);
    let (rot, tr) = errors(&res, &truth);
    assert!(tr < 0.5 && rot < 0.3, "{tr:.3} mm, {rot:.3}°");
    let (rot, tr) = errors(&res, &cpu.transform);
    eprintln!("from the CPU's answer: {tr:.4} mm, {rot:.4}°");
    assert!(tr < 0.1 && rot < 0.05, "{tr:.3} mm, {rot:.3}° from the CPU");
}
