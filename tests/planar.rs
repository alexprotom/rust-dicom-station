//! Planar images: 2D contours (projected, drawn, carried), the presentation
//! state they are saved as, and 2D/2D registration.
//!
//! Run with `--release`, as the 2D/3D suite: the projections and the
//! registrations are slow in the dev profile.

use rust_dicom_station::contours::{Poly, Region};
use rust_dicom_station::drr::Geometry;
use rust_dicom_station::extras::PlanarImage;
use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::loader::LoadedStudy;
use rust_dicom_station::planar_structs::{self, ContourSource, PlanarStructureSet};
use rust_dicom_station::projection::ProjectionGeometry;
use rust_dicom_station::registration::planar::{Rigid2d, Transform2d};
use rust_dicom_station::registration::RigidTransform;
use rust_dicom_station::volume::Volume;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A cube of `n` voxels of `spacing` mm centred on the origin, `f` at each
/// voxel centre.
fn volume_of(n: usize, spacing: f64, f: impl Fn(Vec3) -> i16) -> Volume {
    let half = (n as f64 - 1.0) * 0.5 * spacing;
    let origin = Vec3::new(-half, -half, -half);
    let mut data = vec![0i16; n * n * n];
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let p = origin + Vec3::new(i as f64, j as f64, k as f64) * spacing;
                data[k * n * n + j * n + i] = f(p);
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
        frame_of_reference_uid: "1.2.3.4.5".into(),
        min_value: -1000,
        max_value: 1000,
    }
}

/// The voxels of `vol` within `r` mm of `c`.
fn sphere_mask(vol: &Volume, c: Vec3, r: f64) -> Vec<u8> {
    let [nx, ny, nz] = vol.dims;
    let mut m = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = vol.voxel_to_patient(i as f64, j as f64, k as f64);
                m[k * nx * ny + j * nx + i] = ((p - c).length() <= r) as u8;
            }
        }
    }
    m
}

fn camera(gantry_deg: f64, dims: [usize; 2], panel: f64) -> ProjectionGeometry {
    Geometry {
        gantry_deg,
        panel_mm: [panel, panel],
        dims,
        ..Geometry::default()
    }
    .camera()
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rds_planar_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn square(c: [f64; 2], half: f64) -> Poly {
    Poly::new(vec![
        [c[0] - half, c[1] - half],
        [c[0] + half, c[1] - half],
        [c[0] + half, c[1] + half],
        [c[0] - half, c[1] + half],
    ])
}

// ---------------------------------------------------------------------------
// P0: projection, identity, the model on the study
// ---------------------------------------------------------------------------

#[test]
fn a_sphere_projects_to_one_ring_of_the_magnified_disc() {
    let vol = volume_of(90, 1.0, |_| 0);
    let r = 30.0;
    let mask = sphere_mask(&vol, Vec3::ZERO, r);
    // SAD 1000, SID 1500, 1.5 mm pixels at the detector: 1 mm at the
    // isocentre.
    let cam = camera(0.0, [200, 200], 300.0);
    let region = planar_structs::project_mask(&vol, &mask, &cam, None);
    assert_eq!(region.rings.len(), 1, "one ring");
    // The cone tangent to the sphere meets the detector in a circle of
    // radius SID · R / sqrt(SAD² − R²) mm.
    let radius_mm = 1500.0 * r / (1000.0f64 * 1000.0 - r * r).sqrt();
    let radius_px = radius_mm / 1.5;
    let want = std::f64::consts::PI * radius_px * radius_px;
    let got = region.area();
    assert!(
        (got - want).abs() / want < 0.01,
        "area {got:.1} px², the disc {want:.1}"
    );
    // On the central ray.
    let c = planar_structs::centroid_px(&region).unwrap();
    assert!(
        (c[0] - 99.5).abs() < 0.1 && (c[1] - 99.5).abs() < 0.1,
        "{c:?}"
    );
}

#[test]
fn a_projection_at_a_pose_is_the_moved_structure_projected() {
    let vol = volume_of(80, 1.5, |_| 0);
    // On a voxel centre, so a quarter turn about it maps the mask onto itself.
    let c = Vec3::new(9.75, -15.75, 6.75);
    let mask = sphere_mask(&vol, c, 25.0);
    // The pose turns a quarter about the sphere's centre (the camera turns
    // with it) and shifts by whole voxels:
    // the structure it puts where the patient was found (`T⁻¹`) is the
    // same sphere at T⁻¹(c), on a voxel centre again, which a mask of its own
    // can hold exactly.
    let t = [6.0, -4.5, 3.0];
    let pose = RigidTransform::new([0.0, 0.0, std::f64::consts::FRAC_PI_2, t[0], t[1], t[2]], c);
    let moved = sphere_mask(&vol, pose.inverse().map(c), 25.0);
    for gantry in [0.0, 90.0, 37.0] {
        let cam = camera(gantry, [160, 160], 320.0);
        let at_pose = planar_structs::project_mask(&vol, &mask, &cam, Some(&pose));
        let direct = planar_structs::project_mask(&vol, &moved, &cam, None);
        let (a, b) = (
            planar_structs::centroid_px(&at_pose).unwrap(),
            planar_structs::centroid_px(&direct).unwrap(),
        );
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        assert!(d < 0.05, "gantry {gantry}: centroids {d:.3} px apart");
        let (aa, ab) = (at_pose.area(), direct.area());
        assert!(
            (aa - ab).abs() / ab < 0.005,
            "gantry {gantry}: {aa} vs {ab}"
        );
    }
}

#[test]
fn an_rt_image_is_written_under_the_uid_it_already_had() {
    use rust_dicom_station::dicom_export::{write_rt_image, ObjectMeta};
    use rust_dicom_station::drr::{self, DrrParams, Engine};
    let vol = volume_of(40, 3.0, |p| if p.length() < 30.0 { 500 } else { -1000 });
    let mut params = DrrParams::for_volume(&vol);
    params.engine = Engine::RayCast;
    params.geometry = Geometry {
        dims: [48, 40],
        panel_mm: [200.0, 160.0],
        ..Geometry::default()
    };
    let img = drr::render(&vol, &params, &rust_dicom_station::progress::Quiet)
        .unwrap()
        .to_planar(&params, false);
    assert!(
        !img.sop_instance_uid.is_empty(),
        "a DRR has its UID when made"
    );
    assert!(!img.series_instance_uid.is_empty());
    let dir = scratch("uid");
    let path = dir.join("drr.dcm");
    let written = write_rt_image(&path, &img, &ObjectMeta::default()).unwrap();
    assert_eq!(written, img.sop_instance_uid);
    let back = rust_dicom_station::extras::load_planar(&path).unwrap();
    assert_eq!(back.sop_instance_uid, img.sop_instance_uid);
    assert_eq!(back.series_instance_uid, img.series_instance_uid);
    // An image with no UID of its own gets a fresh one.
    let bare = PlanarImage {
        sop_instance_uid: String::new(),
        series_instance_uid: String::new(),
        ..img.clone()
    };
    let other = write_rt_image(&dir.join("bare.dcm"), &bare, &ObjectMeta::default()).unwrap();
    assert!(!other.is_empty() && other != img.sop_instance_uid);
    let _ = std::fs::remove_dir_all(dir);
}

fn set_with(
    label: &str,
    uid: &str,
    structure: &str,
    image: &str,
    at: [f64; 2],
) -> PlanarStructureSet {
    let mut set = PlanarStructureSet::new(label, "1.2.3", "drawn");
    set.sop_instance_uid = uid.into();
    let s = set.structure_entry(structure, None);
    s.contour_entry(image).region = Region::from_ring(square(at, 5.0));
    set.refresh_images();
    set
}

#[test]
fn merging_studies_keeps_the_planar_structure_sets() {
    let mut a = LoadedStudy::default();
    a.planar_structs
        .push(set_with("Set A", "1.1", "PTV", "9.1", [20.0, 20.0]));
    let mut b = LoadedStudy::default();
    // The same state read again with another structure, and a set of its
    // own.
    b.planar_structs
        .push(set_with("Set A", "1.1", "Spine", "9.1", [40.0, 30.0]));
    b.planar_structs
        .push(set_with("Set B", "1.2", "PTV", "9.2", [10.0, 10.0]));
    rust_dicom_station::loader::merge_study(&mut a, b);
    assert_eq!(a.planar_structs.len(), 2);
    let names: Vec<&str> = a.planar_structs[0]
        .structures
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(names, ["PTV", "Spine"]);
    assert_eq!(a.planar_structs[1].images, ["9.2"]);
}

/// A planar image of `dims` and `spacing` with a UID.
fn blank(uid: &str, dims: [usize; 2], spacing: [f64; 2]) -> PlanarImage {
    PlanarImage {
        label: uid.into(),
        modality: "RTIMAGE".into(),
        cols: dims[0],
        rows: dims[1],
        spacing,
        data: vec![0.0; dims[0] * dims[1]],
        sop_instance_uid: uid.into(),
        frame_count: 1,
        ..Default::default()
    }
}

#[test]
fn a_ring_carried_rigidly_lands_where_the_transform_sends_it() {
    // Moving image 200 × 160 at 0.8 mm, fixed 256 × 256 at 0.6 mm.
    let moving = blank("m", [200, 160], [0.8, 0.8]);
    let fixed = blank("f", [256, 256], [0.6, 0.6]);
    let t = Transform2d::rigid(Rigid2d {
        tx: 7.0,
        ty: -4.0,
        theta: 3f64.to_radians(),
        scale: 1.0,
        center: [0.0, 0.0],
    });
    let ring = square([120.0, 70.0], 12.0);
    let region = Region::from_ring(ring.clone());
    let carried = planar_structs::carry(&region, &t, &moving, &fixed).unwrap();
    use rust_dicom_station::registration::planar::{mm_to_px, px_to_mm};
    for (p, q) in ring.pts.iter().zip(&carried.rings[0].pts) {
        // The carried vertex q maps through T onto the drawn vertex p.
        let back = t.map(px_to_mm([256, 256], [0.6, 0.6], *q));
        let p2 = mm_to_px([200, 160], [0.8, 0.8], back);
        assert!((p2[0] - p[0]).abs() < 1e-9 && (p2[1] - p[1]).abs() < 1e-9);
    }
    // The area scales with the pixel sizes.
    let want = region.area() * (0.8f64 / 0.6).powi(2);
    assert!((carried.area() - want).abs() / want < 1e-9);
    assert_eq!(carried.rings.len(), 1);
    let _ = ContourSource::Drawn;
}

// ---------------------------------------------------------------------------
// P2: the drawing rules on a pixel region
// ---------------------------------------------------------------------------

#[test]
fn the_auto_rule_adds_extends_and_cuts_as_on_a_slice() {
    use rust_dicom_station::contours::auto_cuts;
    let mut region = Region::from_ring(square([50.0, 50.0], 10.0));
    let base = region.area();
    // Crossing nothing: a new island.
    let far = square([100.0, 100.0], 5.0);
    assert!(!auto_cuts(&region, &far));
    planar_structs::apply_ring(&mut region, &far, false);
    assert_eq!(region.rings.len(), 2);
    assert!((region.area() - base - 100.0).abs() < 1e-6);
    // Starting inside and crossing the outline: it extends.
    let mut region = Region::from_ring(square([50.0, 50.0], 10.0));
    let out = Poly::new(vec![[50.0, 50.0], [70.0, 45.0], [70.0, 55.0]]);
    assert!(!auto_cuts(&region, &out));
    planar_structs::apply_ring(&mut region, &out, false);
    assert!(region.area() > base + 1.0);
    // Starting outside and crossing it: it cuts.
    let mut region = Region::from_ring(square([50.0, 50.0], 10.0));
    let bite = Poly::new(vec![[70.0, 50.0], [50.0, 45.0], [50.0, 55.0]]);
    assert!(auto_cuts(&region, &bite));
    planar_structs::apply_ring(&mut region, &bite, true);
    assert!(region.area() < base - 1.0);
    assert!(!region.contains([55.0, 50.0]));
}

#[test]
fn a_polygon_is_filed_on_its_image_by_uid() {
    let mut study = LoadedStudy::default();
    study
        .planar_images
        .push(blank("1.2.9.1", [64, 64], [1.0, 1.0]));
    study
        .planar_images
        .push(blank("1.2.9.2", [64, 64], [1.0, 1.0]));
    let mut set = PlanarStructureSet::new("Drawn", "", "drawn");
    let uid = study.planar_images[1].sop_instance_uid.clone();
    let c = set.structure_entry("Lesion", None).contour_entry(&uid);
    planar_structs::apply_ring(&mut c.region, &square([20.0, 30.0], 4.0), false);
    set.refresh_images();
    let k = planar_structs::file_set(&mut study, set);
    let s = &study.planar_structs[k];
    assert_eq!(s.images, std::slice::from_ref(&uid));
    // Removing the image before it does not move the contour onto another.
    study.planar_images.remove(0);
    assert_eq!(study.planar_by_uid(&uid), Some(0));
    assert!(s.structure("Lesion").unwrap().contour(&uid).is_some());
}

#[test]
fn the_brush_is_round_in_millimetres_on_unequal_pixels() {
    let mut region = Region::new();
    // 0.4 mm across, 0.8 mm down: a 4 mm brush is 10 px wide, 5 px tall.
    planar_structs::brush(
        &mut region,
        [50.0, 50.0],
        [50.0, 50.0],
        4.0,
        [0.4, 0.8],
        false,
    );
    let b = region.bbox();
    let (w_mm, h_mm) = ((b[2] - b[0]) * 0.4, (b[3] - b[1]) * 0.8);
    assert!(
        (w_mm - 8.0).abs() < 0.1 && (h_mm - 8.0).abs() < 0.1,
        "{w_mm} x {h_mm} mm"
    );
    let area_mm2 = planar_structs::area_mm2(&region, [0.4, 0.8]);
    let disc = std::f64::consts::PI * 16.0;
    assert!((area_mm2 - disc).abs() / disc < 0.02, "{area_mm2} mm²");
}

// ---------------------------------------------------------------------------
// P3: the presentation state
// ---------------------------------------------------------------------------

/// A planar set over two images: a ring on each, a second structure and a
/// point.
fn set_over(images: &[&PlanarImage]) -> PlanarStructureSet {
    let mut set = PlanarStructureSet::new("Setup check", "1.2.826.0.1", "drawn");
    let ptv = set.structure_entry("PTV left lung", Some([230, 60, 60]));
    ptv.contour_entry(&images[0].sop_instance_uid).region = Region::from_ring(Poly::new(vec![
        [10.25, 12.5],
        [40.75, 12.5],
        [30.125, 33.0],
    ]));
    ptv.contour_entry(&images[1].sop_instance_uid).region =
        Region::from_ring(square([20.0, 20.0], 6.5));
    let spine = set.structure_entry("Spine", Some([40, 210, 220]));
    spine.contour_entry(&images[1].sop_instance_uid).region =
        Region::from_ring(square([40.0, 30.0], 3.0));
    spine
        .points
        .push((images[0].sop_instance_uid.clone(), [7.5, 8.25]));
    set.refresh_images();
    set
}

fn drr_like(uid: &str, series: &str) -> PlanarImage {
    let mut img = blank(uid, [64, 48], [0.8, 0.8]);
    img.series_instance_uid = series.into();
    img.data = (0..64 * 48).map(|v| (v % 97) as f32).collect();
    img.min_value = 0.0;
    img.max_value = 96.0;
    img.window = (48.0, 96.0);
    img
}

#[test]
fn a_presentation_state_reads_back_as_the_set_it_was() {
    use rust_dicom_station::dicom_export::ObjectMeta;
    let a = drr_like("1.2.826.0.1.11", "1.2.826.0.1.10");
    let b = drr_like("1.2.826.0.1.12", "1.2.826.0.1.20");
    let set = set_over(&[&a, &b]);
    let dir = scratch("gsps");
    let path = dir.join("gsps.dcm");
    let uid = rust_dicom_station::gsps::write(
        &path,
        &set,
        &[(&a, (40.0, 80.0)), (&b, (50.0, 90.0))],
        &ObjectMeta::default(),
    )
    .unwrap();
    let (back, notes) = rust_dicom_station::gsps::load(&path).unwrap();
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(back.sop_instance_uid, uid);
    assert_eq!(back.label, "Setup check");
    assert_eq!(back.images, set.images);
    assert_eq!(back.structures.len(), 2);
    for (s, t) in set.structures.iter().zip(&back.structures) {
        assert_eq!(s.name, t.name);
        // Through CIELab and back: within a step of rounding.
        for k in 0..3 {
            assert!(
                (s.color[k] as i32 - t.color[k] as i32).abs() <= 1,
                "{:?} {:?}",
                s.color,
                t.color
            );
        }
        assert_eq!(s.contours.len(), t.contours.len());
        for c in &s.contours {
            let d = t
                .contour(&c.image_uid)
                .expect("the contour on the same image");
            assert_eq!(d.source, ContourSource::File);
            assert_eq!(c.region.rings.len(), d.region.rings.len());
            for (r, q) in c.region.rings.iter().zip(&d.region.rings) {
                assert_eq!(r.pts.len(), q.pts.len());
                for (p, x) in r.pts.iter().zip(&q.pts) {
                    assert!((p[0] - x[0]).abs() < 1e-3 && (p[1] - x[1]).abs() < 1e-3);
                }
            }
        }
        assert_eq!(s.points.len(), t.points.len());
        for (p, q) in s.points.iter().zip(&t.points) {
            assert_eq!(p.0, q.0);
            assert!((p.1[0] - q.1[0]).abs() < 1e-3 && (p.1[1] - q.1[1]).abs() < 1e-3);
        }
    }
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_state_whose_image_is_not_loaded_is_kept_and_attaches_later() {
    use rust_dicom_station::dicom_export::{write_rt_image, ObjectMeta};
    let a = drr_like("1.2.826.0.2.11", "1.2.826.0.2.10");
    let mut set = PlanarStructureSet::new("Later", "1.2.826.0.2", "drawn");
    set.structure_entry("Mark", None)
        .contour_entry(&a.sop_instance_uid)
        .region = Region::from_ring(square([30.0, 20.0], 5.0));
    set.refresh_images();
    // The state alone in one folder, its image in another.
    let (d1, d2) = (scratch("later_ps"), scratch("later_img"));
    rust_dicom_station::gsps::write(
        &d1.join("ps.dcm"),
        &set,
        &[(&a, a.window)],
        &ObjectMeta::default(),
    )
    .unwrap();
    write_rt_image(&d2.join("img.dcm"), &a, &ObjectMeta::default()).unwrap();
    let p = rust_dicom_station::progress::Progress::default();
    let mut study = rust_dicom_station::loader::load_directory(&d1, &p).unwrap();
    assert_eq!(study.planar_structs.len(), 1, "loaded, not left out");
    assert!(
        study
            .warnings
            .iter()
            .any(|w| w.contains("not in this load")),
        "{:?}",
        study.warnings
    );
    assert!(
        !study.warnings.iter().any(|w| w.contains("not displayed")),
        "a presentation state is no longer among the objects not displayed: {:?}",
        study.warnings
    );
    assert_eq!(study.planar_structs[0].missing_images(&study).len(), 1);
    let more = rust_dicom_station::loader::load_directory(&d2, &p).unwrap();
    rust_dicom_station::loader::merge_study(&mut study, more);
    let s = &study.planar_structs[0];
    assert!(s.missing_images(&study).is_empty(), "attached by UID");
    let _ = std::fs::remove_dir_all(d1);
    let _ = std::fs::remove_dir_all(d2);
}

#[test]
fn saving_a_set_writes_its_made_images_beside_it() {
    let mut study = LoadedStudy::default();
    let mut a = drr_like("1.2.826.0.3.11", "");
    a.series_instance_uid.clear();
    study.planar_images.push(a.clone());
    study.planar_structs.push({
        let mut set = PlanarStructureSet::new("Saved", "", "projected");
        set.structure_entry("PTV", None)
            .contour_entry(&a.sop_instance_uid)
            .region = Region::from_ring(square([30.0, 20.0], 5.0));
        set.refresh_images();
        set
    });
    let dir = scratch("save");
    let (files, _) = rust_dicom_station::gsps::save_set(&study, 0, &dir, &|_| None).unwrap();
    assert_eq!(files.len(), 2, "the image and the state");
    let p = rust_dicom_station::progress::Progress::default();
    let back = rust_dicom_station::loader::load_directory(&dir, &p).unwrap();
    assert_eq!(back.planar_images.len(), 1);
    assert_eq!(back.planar_structs.len(), 1);
    assert!(
        back.planar_structs[0].missing_images(&back).is_empty(),
        "self-contained"
    );
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy();
        assert!(!name.contains("Saved") && !name.contains("PTV"), "{name}");
    }
    let _ = std::fs::remove_dir_all(dir);
}

// ---------------------------------------------------------------------------
// P4: 2D/2D registration, rigid and by points
// ---------------------------------------------------------------------------

use rust_dicom_station::registration::metric2d::{Image2d, Metric2d};
use rust_dicom_station::registration::planar::{self as p2, Dof2d, Method2d, Planar2d, Request2d};

/// A torso-like phantom in HU with no symmetry a rotation or a reflection
/// could exploit (the 2D/3D suite's, smaller).
fn torso(p: Vec3) -> i16 {
    let ell = |c: Vec3, r: Vec3| {
        let q = p - c;
        ((q.x / r.x).powi(2) + (q.y / r.y).powi(2) + (q.z / r.z).powi(2)).sqrt()
    };
    if ell(Vec3::ZERO, Vec3::new(85.0, 65.0, 95.0)) > 1.0 {
        return -1000;
    }
    let mut v = 0i16;
    if ell(Vec3::new(-35.0, -5.0, 15.0), Vec3::new(28.0, 32.0, 50.0)) < 1.0 {
        v = -750;
    }
    if ell(Vec3::new(38.0, -2.0, 20.0), Vec3::new(25.0, 30.0, 45.0)) < 1.0 {
        v = -750;
    }
    if (p.x * p.x + (p.y - 40.0).powi(2)).sqrt() < 11.0 && p.z.abs() < 85.0 {
        v = 700;
    }
    if ell(Vec3::new(50.0, 15.0, -45.0), Vec3::new(9.0, 9.0, 14.0)) < 1.0 {
        v = 600;
    }
    if ell(Vec3::new(-20.0, 30.0, -60.0), Vec3::new(14.0, 6.0, 8.0)) < 1.0 {
        v = 500;
    }
    v
}

/// Deterministic noise.
struct Lcg(u64);
impl Lcg {
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

/// The AP DRR of the torso on a grid of `n` voxels of `spacing` mm, at 1.5
/// mm pixels.
fn radiograph(n: usize, spacing: f64, dims: [usize; 2]) -> Planar2d {
    use rust_dicom_station::drr::HuMode;
    use rust_dicom_station::projection::{render_posed, MuVolume};
    let vol = volume_of(n, spacing, torso);
    let mu = MuVolume::prepare(&vol, HuMode::Water, -800.0, None);
    let panel = dims[0] as f64 * 1.5;
    let cam = Geometry {
        panel_mm: [panel, dims[1] as f64 * 1.5],
        dims,
        ..Geometry::default()
    }
    .camera();
    let px = render_posed(&mu, &cam, 0.75, &rust_dicom_station::progress::Quiet).unwrap();
    Planar2d {
        image: Image2d::new(px, dims),
        spacing: [1.5, 1.5],
        camera: None,
        isocenter: None,
        transmission: false,
    }
}

/// `img` resampled through `s` (so registering it back onto `img` should
/// answer `s⁻¹`), with Gaussian noise at `noise` of its range.
fn moved(img: &Planar2d, s: &Transform2d, noise: f64, seed: u64) -> Planar2d {
    let mut r = p2::resample(img, s, img);
    let (lo, hi) = r
        .data
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    let sd = noise * (hi - lo) as f64;
    let mut rng = Lcg(seed);
    for v in &mut r.data {
        *v += (sd * rng.gauss()) as f32;
    }
    Planar2d {
        image: r,
        ..img.clone()
    }
}

fn truth(tx: f64, ty: f64, deg: f64, scale: f64) -> Rigid2d {
    Rigid2d {
        tx,
        ty,
        theta: deg.to_radians(),
        scale,
        center: [0.0, 0.0],
    }
}

/// The largest distance between where `got` and `want` put the corners
/// and the centre of a 200 mm field, and the rotation error in degrees.
fn error(got: &Rigid2d, want: &Rigid2d) -> (f64, f64) {
    let mut worst = 0.0f64;
    for p in [
        [0.0, 0.0],
        [-100.0, -100.0],
        [100.0, -100.0],
        [-100.0, 100.0],
        [100.0, 100.0],
    ] {
        let (a, b) = (got.map(p), want.map(p));
        worst = worst.max(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt());
    }
    let centre = {
        let (a, b) = (got.map([0.0, 0.0]), want.map([0.0, 0.0]));
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    };
    let _ = worst;
    (centre, (got.theta - want.theta).to_degrees().abs())
}

#[test]
fn a_shifted_and_turned_radiograph_is_found_again() {
    let fixed = radiograph(80, 3.0, [160, 160]);
    let want = truth(7.0, -4.0, 3.0, 1.0);
    let moving = moved(&fixed, &Transform2d::rigid(want.inverse()), 0.01, 7);
    for metric in [Metric2d::Ncc, Metric2d::GradientCorrelation] {
        let mut req = Request2d::new(fixed.clone(), moving.clone());
        req.metric = metric;
        req.max_pixels = 160;
        let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
        let (mm, deg) = error(&r.transform.rigid, &want);
        assert!(
            mm < 0.2 && deg < 0.1,
            "{metric:?}: {mm:.3} mm, {deg:.3} deg ({})",
            r.line()
        );
        assert!(r.after > r.before);
    }
}

#[test]
fn a_radiograph_against_a_drr_of_another_grid_is_found_again() {
    let fixed = radiograph(80, 3.0, [160, 160]);
    // The same anatomy sampled afresh on another lattice, then moved.
    let other = radiograph(100, 2.5, [160, 160]);
    let want = truth(-6.0, 5.0, -2.0, 1.0);
    let moving = moved(&other, &Transform2d::rigid(want.inverse()), 0.01, 11);
    let mut req = Request2d::new(fixed, moving);
    req.metric = Metric2d::GradientCorrelation;
    req.max_pixels = 160;
    let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    let (mm, deg) = error(&r.transform.rigid, &want);
    assert!(
        mm < 0.5 && deg < 0.2,
        "{mm:.3} mm, {deg:.3} deg ({})",
        r.line()
    );
}

#[test]
fn a_scaled_radiograph_is_found_with_its_scale() {
    let fixed = radiograph(80, 3.0, [160, 160]);
    let want = truth(3.0, 2.0, 1.0, 1.04);
    let moving = moved(&fixed, &Transform2d::rigid(want.inverse()), 0.005, 13);
    let mut req = Request2d::new(fixed, moving);
    req.dof = Dof2d::Similarity;
    req.max_pixels = 160;
    let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    let s = r.transform.rigid.scale;
    assert!((s - 1.04).abs() / 1.04 < 0.005, "scale {s} ({})", r.line());
}

#[test]
fn three_pairs_give_the_transform_and_a_pair_off_is_left_out() {
    let fixed = radiograph(40, 6.0, [64, 64]);
    let want = truth(5.0, -3.0, 4.0, 1.0);
    let moving = fixed.clone();
    let dims = fixed.dims();
    let to_moving = |p: [f64; 2]| -> [f64; 2] {
        p2::mm_to_px(
            dims,
            fixed.spacing,
            want.map(p2::px_to_mm(dims, fixed.spacing, p)),
        )
    };
    let pts = [
        [10.0, 12.0],
        [50.0, 20.0],
        [30.0, 55.0],
        [20.0, 40.0],
        [45.0, 45.0],
    ];
    let mut req = Request2d::new(fixed.clone(), moving.clone());
    req.method = Method2d::Points;
    req.pairs = pts[..3].iter().map(|&p| (p, to_moving(p))).collect();
    let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    let (mm, deg) = error(&r.transform.rigid, &want);
    assert!(mm < 1e-9 && deg < 1e-9, "exact: {mm} mm {deg} deg");
    assert!(r.pairs.iter().all(|p| p.residual_mm < 1e-9));
    // Five pairs, the last 6 mm off (4 pixels at 1.5 mm).
    req.pairs = pts.iter().map(|&p| (p, to_moving(p))).collect();
    req.pairs[4].1[0] += 6.0 / 1.5;
    let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    assert_eq!(r.left_out, [4], "{:?}", r.pairs);
    assert!(!r.pairs[4].used && r.pairs[4].residual_mm > 5.0);
    let (mm, _) = error(&r.transform.rigid, &want);
    assert!(mm < 1e-6, "the fit without it is exact: {mm}");
}

#[test]
fn markers_found_in_both_images_pair_and_fit() {
    // Five bright seeds on a noisy background, 1.2 mm pixels.
    let dims = [128usize, 128];
    let sp = [1.2, 1.2];
    let seeds_mm = [
        [-30.0, -25.0],
        [22.0, -18.0],
        [5.0, 30.0],
        [-15.0, 12.0],
        [35.0, 28.0],
    ];
    let draw = |t: &Rigid2d, seed: u64| -> Planar2d {
        let mut rng = Lcg(seed);
        let mut data = vec![0.0f32; dims[0] * dims[1]];
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let x = p2::px_to_mm(dims, sp, [i as f64, j as f64]);
                let mut v = 100.0 + 2.0 * rng.gauss();
                for s in &seeds_mm {
                    let c = t.map(*s);
                    let d2 = (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2);
                    v += 80.0 * (-d2 / (2.0 * 1.0f64.powi(2))).exp();
                }
                data[j * dims[0] + i] = v as f32;
            }
        }
        Planar2d {
            image: Image2d::new(data, dims),
            spacing: sp,
            camera: None,
            isocenter: None,
            transmission: false,
        }
    };
    let want = truth(6.0, -4.0, 5.0, 1.0);
    let fixed = draw(&Rigid2d::identity([0.0, 0.0]), 3);
    // The moving image's seeds are where T puts the fixed image's.
    let moving = draw(&want, 5);
    let mut req = Request2d::new(fixed, moving);
    req.method = Method2d::Points;
    req.detect_markers = Some(2.0);
    let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    assert!(r.pairs.len() >= 4, "{} pairs", r.pairs.len());
    let (mm, deg) = error(&r.transform.rigid, &want);
    assert!(mm < 0.3 && deg < 0.3, "{mm:.3} mm {deg:.3} deg");
}

#[test]
fn a_ring_carried_through_a_registration_lands_on_the_drawn_one() {
    let fixed = radiograph(80, 3.0, [160, 160]);
    let want = truth(7.0, -4.0, 3.0, 1.0);
    let moving = moved(&fixed, &Transform2d::rigid(want.inverse()), 0.01, 17);
    let mut req = Request2d::new(fixed.clone(), moving.clone());
    req.max_pixels = 160;
    let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    // The ring drawn on the fixed image, and the same anatomy's ring on the
    // moving image (where the true transform sends it).
    let drawn = Region::from_ring(Poly::circle([70.0, 60.0], 12.0, 48));
    let dims = fixed.dims();
    let on_moving = Region::from_ring(Poly::new(
        drawn.rings[0]
            .pts
            .iter()
            .map(|&p| {
                p2::mm_to_px(
                    dims,
                    fixed.spacing,
                    want.map(p2::px_to_mm(dims, fixed.spacing, p)),
                )
            })
            .collect(),
    ));
    let img = |p: &Planar2d, uid: &str| {
        let mut b = blank(uid, p.dims(), p.spacing);
        b.data = p.image.data.clone();
        b
    };
    let carried = planar_structs::carry(
        &on_moving,
        &r.transform,
        &img(&moving, "m"),
        &img(&fixed, "f"),
    )
    .unwrap();
    let worst = drawn.rings[0]
        .pts
        .iter()
        .zip(&carried.rings[0].pts)
        .map(|(a, b)| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
        .fold(0.0, f64::max);
    assert!(worst < 0.2, "{worst:.3} px");
}

#[test]
fn the_answer_does_not_depend_on_the_thread_count() {
    let fixed = radiograph(60, 4.0, [96, 96]);
    let want = truth(5.0, 3.0, -2.0, 1.0);
    let moving = moved(&fixed, &Transform2d::rigid(want.inverse()), 0.01, 19);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut req = Request2d::new(fixed.clone(), moving.clone());
            req.metric = Metric2d::GradientCorrelation;
            req.max_pixels = 96;
            p2::register(&req, &rust_dicom_station::progress::Quiet)
                .unwrap()
                .transform
                .rigid
        })
    };
    let one = run(1);
    for n in [2, 8] {
        let r = run(n);
        assert_eq!(
            [r.tx.to_bits(), r.ty.to_bits(), r.theta.to_bits()],
            [one.tx.to_bits(), one.ty.to_bits(), one.theta.to_bits()],
            "{n} threads"
        );
    }
}

// ---------------------------------------------------------------------------
// P5: the deformable method
// ---------------------------------------------------------------------------

/// A smooth warp of two sinusoids, about 6 mm at its largest, on a grid of
/// 20 mm over the image.
fn sinusoid_warp(img: &Planar2d) -> Transform2d {
    use rust_dicom_station::registration::planar::BSpline2d;
    let d = img.dims();
    let lo = p2::px_to_mm(d, img.spacing, [-0.5, -0.5]);
    let hi = p2::px_to_mm(d, img.spacing, [d[0] as f64 - 0.5, d[1] as f64 - 0.5]);
    let mut w = BSpline2d::zero(lo, hi, 20.0);
    let [nx, _] = w.grid_dims;
    for k in 0..w.coeffs.len() / 2 {
        let x = w.origin[0] + (k % nx) as f64 * w.spacing;
        let y = w.origin[1] + (k / nx) as f64 * w.spacing;
        let l = 2.0 * std::f64::consts::PI / 160.0;
        w.coeffs[2 * k] = 6.5 * (l * y).sin();
        w.coeffs[2 * k + 1] = 6.5 * (l * x).cos();
    }
    Transform2d {
        rigid: Rigid2d::identity([0.0, 0.0]),
        warp: Some(w),
    }
}

/// Mean and largest error of `got` against the inverse of `s`, over the
/// body (the pixels the radiograph has anatomy in), away from the edge.
fn warp_error(got: &Transform2d, s: &Transform2d, img: &Planar2d) -> (f64, f64, f64) {
    let d = img.dims();
    let hi = img.image.data.iter().copied().fold(f32::MIN, f32::max);
    let (mut sum, mut max, mut n, mut amp) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for j in (16..d[1] - 16).step_by(2) {
        for i in (16..d[0] - 16).step_by(2) {
            if img.image.data[j * d[0] + i] < 0.25 * hi {
                continue;
            }
            let x = p2::px_to_mm(d, img.spacing, [i as f64, j as f64]);
            let want = s.inverse_map(x, 1e-6).unwrap();
            let have = got.map(x);
            let e = ((want[0] - have[0]).powi(2) + (want[1] - have[1]).powi(2)).sqrt();
            amp = amp.max(((want[0] - x[0]).powi(2) + (want[1] - x[1]).powi(2)).sqrt());
            sum += e;
            max = max.max(e);
            n += 1.0;
        }
    }
    (sum / n, max, amp)
}

/// The radiograph with texture: small bright blobs scattered over the body,
/// as trabecular bone and vessels give a real one. A DRR of the smooth
/// phantom alone has too few edges to pin a warp down anywhere but at the
/// outline of each organ.
fn textured(img: &Planar2d) -> Planar2d {
    let d = img.dims();
    let hi = img.image.data.iter().copied().fold(f32::MIN, f32::max);
    let mut rng = Lcg(29);
    let blobs: Vec<([f64; 2], f64)> = (0..260)
        .map(|_| {
            (
                [rng.uniform() * d[0] as f64, rng.uniform() * d[1] as f64],
                1.0 + 1.5 * rng.uniform(),
            )
        })
        .collect();
    let mut out = img.clone();
    for j in 0..d[1] {
        for i in 0..d[0] {
            let o = j * d[0] + i;
            let base = img.image.data[o];
            if base < 0.15 * hi {
                continue;
            }
            let mut t = 0.0;
            for (c, r) in &blobs {
                let d2 = (i as f64 - c[0]).powi(2) + (j as f64 - c[1]).powi(2);
                if d2 < 16.0 * r * r {
                    t += (-d2 / (2.0 * r * r)).exp();
                }
            }
            out.image.data[o] = base + (0.12 * hi as f64 * t) as f32;
        }
    }
    out
}

fn deformable_run(metric: p2::DeformMetric, remap: impl Fn(f32) -> f32) -> (f64, f64, f64) {
    let fixed = textured(&radiograph(80, 3.0, [160, 160]));
    let s = sinusoid_warp(&fixed);
    let mut moving = moved(&fixed, &s, 0.0, 23);
    for v in &mut moving.image.data {
        *v = remap(*v);
    }
    let mut req = Request2d::new(fixed.clone(), moving);
    req.method = Method2d::Deformable;
    req.deformable_metric = metric;
    req.max_pixels = 160;
    req.grid_mm = 40.0;
    req.grid_min_mm = 10.0;
    req.dof = Dof2d::Shift;
    req.lambda = p2::DeformMetric::default_lambda(metric);
    let r = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    assert!(r.warp_stats.as_ref().is_some_and(|w| w.folds == 0));
    warp_error(&r.transform, &s, &fixed)
}

#[test]
fn a_sinusoidal_warp_is_recovered_with_msd() {
    let (mean, max, amp) = deformable_run(p2::DeformMetric::Msd, |v| v);
    assert!(amp > 4.0, "the warp is a real one: {amp:.2} mm");
    assert!(
        mean < 0.5 && max < 1.5,
        "mean {mean:.3} mm, max {max:.3} mm"
    );
}

#[test]
fn a_sinusoidal_warp_is_recovered_with_mi_across_a_change_of_contrast() {
    // The moving image's values remapped by a monotone curve: what MSD
    // cannot follow and mutual information does not see.
    let (mean, max, _) = deformable_run(p2::DeformMetric::Mi, |v| (v.max(0.0) + 1.0).sqrt() * 10.0);
    assert!(
        mean < 0.5 && max < 1.5,
        "mean {mean:.3} mm, max {max:.3} mm"
    );
}

#[test]
fn a_ring_carried_through_a_warp_and_back_returns() {
    let fixed = radiograph(40, 6.0, [96, 96]);
    let t = sinusoid_warp(&fixed);
    let img = |uid: &str| blank(uid, fixed.dims(), fixed.spacing);
    let (f, m) = (img("f"), img("m"));
    let ring = Region::from_ring(Poly::circle([48.0, 40.0], 15.0, 40));
    // Onto the fixed image (through the inverse), then forward again.
    let carried = planar_structs::carry(&ring, &t, &m, &f).unwrap();
    let d = fixed.dims();
    for (p, q) in ring.rings[0].pts.iter().zip(&carried.rings[0].pts) {
        let back = p2::mm_to_px(d, fixed.spacing, t.map(p2::px_to_mm(d, fixed.spacing, *q)));
        let e = ((back[0] - p[0]).powi(2) + (back[1] - p[1]).powi(2)).sqrt();
        assert!(e < 0.1, "{e:.4} px");
    }
}

// ---------------------------------------------------------------------------
// P6: two 2D matches as one 3D correction
// ---------------------------------------------------------------------------

/// A study of the torso CT with, per gantry angle, the DRR at the plan
/// (the reference) and the image of the patient found at `truth` (today's),
/// both placed by their geometry. Images 2k and 2k + 1.
fn setup_study(truth: &RigidTransform) -> LoadedStudy {
    use rust_dicom_station::drr::HuMode;
    use rust_dicom_station::projection::{render_posed, MuVolume};
    let ct = volume_of(80, 3.0, torso);
    let mu = MuVolume::prepare(&ct, HuMode::Water, -800.0, None);
    let mut study = LoadedStudy {
        volume: std::sync::Arc::new(ct),
        ..Default::default()
    };
    for (name, g) in [("AP", 0.0), ("LAT", 90.0)] {
        let geometry = Geometry {
            gantry_deg: g,
            panel_mm: [300.0, 300.0],
            dims: [128, 128],
            ..Geometry::default()
        };
        let cam = geometry.camera();
        for (what, c) in [("reference", cam), ("today", cam.transformed(truth))] {
            let px = render_posed(&mu, &c, 0.75, &rust_dicom_station::progress::Quiet).unwrap();
            let (lo, hi) = px
                .iter()
                .fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            study.planar_images.push(PlanarImage {
                label: format!("{name} {what}"),
                modality: "RTIMAGE".into(),
                rows: 128,
                cols: 128,
                spacing: [300.0 / 128.0; 2],
                data: px,
                min_value: lo,
                max_value: hi,
                window: ((lo + hi) / 2.0, hi - lo),
                iec: Some(geometry),
                camera: Some(cam),
                sop_instance_uid: rust_dicom_station::dicom_export::new_uid(),
                frame_count: 1,
                ..Default::default()
            });
        }
    }
    study
}

/// Each view's today image matched in 2D against its reference.
fn matches(study: &LoadedStudy) -> Vec<p2::PlanarReg> {
    (0..2)
        .map(|v| {
            let (refi, today) = (&study.planar_images[2 * v], &study.planar_images[2 * v + 1]);
            let mut req = Request2d::new(Planar2d::of(today), Planar2d::of(refi));
            req.max_pixels = 128;
            req.metric = Metric2d::GradientCorrelation;
            let result = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
            p2::PlanarReg {
                label: format!("match {v}"),
                fixed_uid: today.sop_instance_uid.clone(),
                moving_uid: refi.sop_instance_uid.clone(),
                result,
                moved_copy_uid: None,
                when: String::new(),
            }
        })
        .collect()
}

#[test]
fn two_matches_give_the_translation_and_name_the_weak_rotation() {
    use rust_dicom_station::iec::{PatientFrame, PatientPosition};
    use rust_dicom_station::setup::{self, SetupMethod, SetupOptions};
    let truth = RigidTransform::new([0.0, 0.0, 0.0, 4.0, -3.0, 5.0], Vec3::ZERO);
    let study = setup_study(&truth);
    let regs = matches(&study);
    let opts = SetupOptions {
        patient: Some(PatientFrame::Position(PatientPosition::Hfs)),
        max_pixels: 128,
        ..Default::default()
    };
    let refs: Vec<&p2::PlanarReg> = regs.iter().collect();
    let s = setup::from_planar(&study, &refs, &opts, &rust_dicom_station::progress::Quiet).unwrap();
    assert_eq!(s.method, SetupMethod::Planar);
    let got = s.result.transform.params();
    let want = truth.params();
    for k in 3..6 {
        assert!(
            (got[k] - want[k]).abs() < 0.5,
            "translation {got:?} vs {want:?}"
        );
    }
    let w = s.result.warning.clone().unwrap_or_default();
    assert!(
        w.contains("rotation about the axis the detectors share"),
        "{w}"
    );
    assert_eq!(s.result.views.len(), 2);
    // The DRRs at the pose replace the moved copies: there is a CT.
    assert_eq!(s.result.views[0].drr.dims, s.result.views[0].image.dims);
}

#[test]
fn a_shift_to_the_left_reads_with_the_2d3d_signs() {
    use rust_dicom_station::iec::{PatientFrame, PatientPosition};
    use rust_dicom_station::setup::{self, SetupOptions};
    let truth = RigidTransform::new([0.0, 0.0, 0.0, -5.0, 0.0, 0.0], Vec3::ZERO);
    let study = setup_study(&truth);
    let regs = matches(&study);
    let opts = SetupOptions {
        patient: Some(PatientFrame::Position(PatientPosition::Hfs)),
        max_pixels: 128,
        ..Default::default()
    };
    let refs: Vec<&p2::PlanarReg> = regs.iter().collect();
    let s = setup::from_planar(&study, &refs, &opts, &rust_dicom_station::progress::Quiet).unwrap();
    let d = s.result.displacement;
    assert!(
        (d[3] - 5.0).abs() < 0.5 && d[4].abs() < 0.5 && d[5].abs() < 0.5,
        "{d:?}"
    );
    let c = s.result.correction();
    assert!((c[3] + 5.0).abs() < 0.5, "{c:?}");
    assert!(
        (s.result.couch.lateral + 5.0).abs() < 0.5,
        "{}",
        s.result.couch.line()
    );
}

#[test]
fn a_roll_in_one_view_reads_as_the_rotation_about_its_ray() {
    use rust_dicom_station::iec::{PatientFrame, PatientPosition};
    use rust_dicom_station::setup::{self, SetupOptions};
    // A turn about the A-P axis: the AP view sees it as a roll, the
    // lateral one as nothing in its plane.
    let truth = RigidTransform::new([0.0, 0.03, 0.0, 0.0, 0.0, 0.0], Vec3::ZERO);
    let study = setup_study(&truth);
    let regs = matches(&study);
    let opts = SetupOptions {
        patient: Some(PatientFrame::Position(PatientPosition::Hfs)),
        max_pixels: 128,
        ..Default::default()
    };
    let refs: Vec<&p2::PlanarReg> = regs.iter().collect();
    let s = setup::from_planar(&study, &refs, &opts, &rust_dicom_station::progress::Quiet).unwrap();
    let got = s.result.transform.params();
    assert!((got[1] - 0.03).abs() < 0.005, "{got:?}");
}

// ---------------------------------------------------------------------------
// P7: the export plan carries planar images and planar structure sets
// ---------------------------------------------------------------------------

#[test]
fn a_study_export_carries_its_drr_and_the_contours_on_it() {
    use rust_dicom_station::dicom_export::ExportParams;
    use rust_dicom_station::export::{self, ExportPlan, ObjKind, UidMode};
    let mut study = LoadedStudy::default();
    let img = drr_like("1.2.826.0.7.11", "1.2.826.0.7.10");
    study.planar_images.push(img.clone());
    let mut set = PlanarStructureSet::new("Exported", "", "projected");
    set.structure_entry("PTV", Some([230, 60, 60]))
        .contour_entry(&img.sop_instance_uid)
        .region = Region::from_ring(square([30.0, 20.0], 6.0));
    set.refresh_images();
    study.planar_structs.push(set);

    for mode in [UidMode::Keep, UidMode::New] {
        let mut plan =
            ExportPlan::build(export::one_study(&study), ExportParams::for_study(&study));
        plan.set_uid_mode(mode);
        let kinds: Vec<ObjKind> = plan
            .studies()
            .flat_map(|s| s.objects.iter().map(|o| o.kind))
            .collect();
        assert!(kinds.contains(&ObjKind::Planar) && kinds.contains(&ObjKind::PlanarStructures));
        let dir = scratch(&format!("export_{mode:?}"));
        let sum = export::run(
            &plan,
            export::one_study(&study),
            &dir,
            &rust_dicom_station::progress::Progress::default(),
        )
        .unwrap();
        assert_eq!(sum.files, 2, "{:?}", sum.warnings);
        let back = rust_dicom_station::loader::load_directory(
            &dir,
            &rust_dicom_station::progress::Progress::default(),
        )
        .unwrap();
        assert_eq!(back.planar_images.len(), 1);
        assert_eq!(back.planar_structs.len(), 1);
        let s = &back.planar_structs[0];
        assert!(
            s.missing_images(&back).is_empty(),
            "{mode:?}: the state names the image as written"
        );
        let uid = &back.planar_images[0].sop_instance_uid;
        assert_eq!(
            uid == &img.sop_instance_uid,
            mode == UidMode::Keep,
            "{mode:?}: the image keeps its UID only when told to"
        );
        assert!(s.structure("PTV").unwrap().contour(uid).is_some());
        let _ = std::fs::remove_dir_all(dir);
    }
}

// ---------------------------------------------------------------------------
// Undo, removals, and a planar setup refined in 3D
// ---------------------------------------------------------------------------

#[test]
fn undo_restores_the_contours_and_goes_32_deep() {
    use rust_dicom_station::planar_structs::Undo;
    let mut study = LoadedStudy::default();
    study
        .planar_structs
        .push(set_with("S", "", "PTV", "9.1", [20.0, 20.0]));
    let before = study.planar_structs[0].structures[0].contours.clone();
    let mut undo = Undo::default();
    undo.push(0, &study, 0, 0);
    let c = study.planar_structs[0].structures[0]
        .contour_mut("9.1")
        .unwrap();
    planar_structs::apply_ring(&mut c.region, &square([40.0, 40.0], 5.0), false);
    assert_ne!(study.planar_structs[0].structures[0].contours, before);
    // Another workspace's step does not take this one back.
    assert!(!undo.undo(1, &mut study));
    assert!(undo.undo(0, &mut study));
    assert_eq!(study.planar_structs[0].structures[0].contours, before);
    assert!(!undo.undo(0, &mut study), "nothing left");
    for _ in 0..40 {
        undo.push(0, &study, 0, 0);
    }
    assert_eq!(undo.steps.len(), Undo::DEPTH);
}

#[test]
fn removing_a_moved_copy_drops_its_registration_and_nothing_else() {
    let fixed = radiograph(40, 6.0, [48, 48]);
    let mut req = Request2d::new(fixed.clone(), fixed.clone());
    req.max_pixels = 48;
    req.levels = 1;
    let result = p2::register(&req, &rust_dicom_station::progress::Quiet).unwrap();
    let mut study = LoadedStudy::default();
    for uid in ["f", "m", "copy"] {
        study.planar_images.push(blank(uid, [48, 48], [1.5, 1.5]));
    }
    study
        .planar_structs
        .push(set_with("On the copy", "", "PTV", "copy", [20.0, 20.0]));
    let reg = |copy: Option<&str>| p2::PlanarReg {
        label: "r".into(),
        fixed_uid: "f".into(),
        moving_uid: "m".into(),
        result: result.clone(),
        moved_copy_uid: copy.map(String::from),
        when: String::new(),
    };
    study.planar_regs.push(reg(Some("copy")));
    study.planar_regs.push(reg(None));
    assert!(planar_structs::orphaned_regs(&study).is_empty());
    study.planar_images.remove(2);
    assert_eq!(planar_structs::orphaned_regs(&study), [0]);
    // The set on the copy stays, its image now not loaded.
    assert_eq!(study.planar_structs[0].missing_images(&study), ["copy"]);
}

#[test]
fn a_setup_from_2d_matches_refines_in_3d() {
    use rust_dicom_station::iec::{PatientFrame, PatientPosition};
    use rust_dicom_station::nn::device::DevicePref;
    use rust_dicom_station::setup::{self, GeometrySource, SetupOptions, ViewChoice};
    let truth = RigidTransform::new([0.0, 0.0, 0.0, 4.0, -3.0, 5.0], Vec3::ZERO);
    let study = setup_study(&truth);
    let regs = matches(&study);
    let refs: Vec<&p2::PlanarReg> = regs.iter().collect();
    let frame = Some(PatientFrame::Position(PatientPosition::Hfs));
    let opts = SetupOptions {
        patient: frame,
        max_pixels: 128,
        ..Default::default()
    };
    let planar =
        setup::from_planar(&study, &refs, &opts, &rust_dicom_station::progress::Quiet).unwrap();
    // Today's images, each with its own geometry, from the planar pose.
    let views = vec![
        ViewChoice::new(1, GeometrySource::Image),
        ViewChoice::new(3, GeometrySource::Image),
    ];
    let refine = SetupOptions {
        patient: frame,
        metric: Metric2d::Ncc,
        max_pixels: 96,
        levels: 2,
        step_mm: 1.5,
        conditioning: false,
        device: DevicePref::Cpu,
        start: Some(planar.result.transform.clone()),
        ..Default::default()
    };
    let s = setup::run(
        &study,
        &views,
        &refine,
        None,
        &rust_dicom_station::progress::Quiet,
    )
    .unwrap();
    let (got, want) = (s.result.transform.params(), truth.params());
    for k in 3..6 {
        assert!((got[k] - want[k]).abs() < 0.5, "{got:?} vs {want:?}");
    }
    assert!(s.result.start.contains("start") || !s.result.start.is_empty());
}
