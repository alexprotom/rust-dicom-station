//! Integration test against the synthetic RT study in `test_data/`.
//!
//! Generate the data first with: `python3 tools/generate_test_data.py`
//! (the test is skipped when the directory is absent).

use rust_dicom_viewer::geometry::Vec3;
use rust_dicom_viewer::loader::{self, Progress};
use rust_dicom_viewer::render;
use rust_dicom_viewer::volume::ViewPlane;

fn test_data_dir() -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    if p.join("RS_synth.dcm").exists() {
        Some(p)
    } else {
        None
    }
}

#[test]
fn load_synthetic_study() {
    let Some(dir) = test_data_dir() else {
        eprintln!("test_data/ not found — run tools/generate_test_data.py first; skipping");
        return;
    };
    let progress = Progress::default();
    let t0 = std::time::Instant::now();
    let study = loader::load_directory(&dir, &progress).expect("study should load");
    eprintln!("loaded in {:?}; warnings: {:?}", t0.elapsed(), study.warnings);

    // ---- Volume ----
    let v = &study.volume;
    assert_eq!(v.dims, [96, 96, 40], "volume dims");
    assert!((v.spacing[0] - 2.0).abs() < 1e-6);
    assert!((v.spacing[1] - 2.0).abs() < 1e-6);
    assert!((v.spacing[2] - 2.0).abs() < 1e-3, "slice spacing {}", v.spacing[2]);

    // Patient (0,0,0) is the center of the target sphere (HU 100).
    let c = v.patient_to_voxel(Vec3::ZERO);
    assert!((c[0] - 47.5).abs() < 1e-6 && (c[1] - 47.5).abs() < 1e-6 && (c[2] - 19.5).abs() < 1e-3);
    let hu_center = v.index(48, 48, 20);
    assert_eq!(hu_center, 100, "target HU");
    let hu_air = v.index(2, 2, 20);
    assert_eq!(hu_air, -1000, "air HU");
    // Water shell inside body but outside target.
    let w = v.patient_to_voxel(Vec3::new(50.0, 0.0, 1.0));
    let hu_water = v.index(w[0].round() as usize, w[1].round() as usize, w[2].round() as usize);
    assert_eq!(hu_water, 0, "water HU");

    // Round-trip mapping.
    let p = v.voxel_to_patient(10.0, 20.0, 30.0);
    let back = v.patient_to_voxel(p);
    assert!((back[0] - 10.0).abs() < 1e-9 && (back[1] - 20.0).abs() < 1e-9 && (back[2] - 30.0).abs() < 1e-9);

    // ---- Structures ----
    let ss = study.structures.as_ref().expect("RTSTRUCT present");
    assert_eq!(ss.rois.len(), 3);
    let target = ss.rois.iter().find(|r| r.name == "TARGET").expect("TARGET roi");
    assert_eq!(target.roi_type, "PTV");
    assert!(target.contours.len() >= 20, "target contour count {}", target.contours.len());
    let body = ss.rois.iter().find(|r| r.name == "BODY").unwrap();
    assert_eq!(body.contours.len(), 40);

    // Axial contour of TARGET on the central slice must be a circle of r≈25 mm.
    let gfx = render::roi_on_plane(v, target, ViewPlane::Axial, 20);
    assert!(!gfx.polylines.is_empty(), "target polyline on central axial slice");
    let pl = &gfx.polylines[0];
    let r_mm: f32 = pl
        .iter()
        .map(|p| {
            let pat = v.voxel_to_patient(p[0] as f64, p[1] as f64, 20.0);
            ((pat.x * pat.x + pat.y * pat.y) as f32).sqrt()
        })
        .sum::<f32>()
        / pl.len() as f32;
    assert!((r_mm - 24.97).abs() < 1.0, "target radius on central slice: {r_mm}");

    // Sagittal silhouette segments should exist through the target center.
    let gfx_sag = render::roi_on_plane(v, target, ViewPlane::Sagittal, 48);
    assert!(!gfx_sag.segments.is_empty(), "sagittal silhouette");

    // ---- Dose ----
    assert_eq!(study.doses.len(), 1);
    let d = &study.doses[0];
    assert!((d.max_dose - 60.0).abs() < 0.5, "max dose {}", d.max_dose);
    let center = d.sample(Vec3::ZERO).expect("dose at isocenter");
    assert!((center - 60.0).abs() < 0.6, "isocenter dose {center}");
    // 1 sigma away along x: 60 * exp(-0.5) ≈ 36.39
    let one_sigma = d.sample(Vec3::new(20.0, 0.0, 0.0)).unwrap();
    assert!((one_sigma - 36.39).abs() < 1.0, "1-sigma dose {one_sigma}");
    assert!(d.sample(Vec3::new(500.0, 0.0, 0.0)).is_none(), "outside grid");

    // Dose plane resampling + isodose extraction on the central axial slice.
    let mut plane = Vec::new();
    render::sample_dose_plane(v, d, ViewPlane::Axial, 20, &mut plane);
    let max_in_plane = plane.iter().copied().fold(0.0f32, f32::max);
    assert!((max_in_plane - 60.0).abs() < 1.5, "max in plane {max_in_plane}");
    let segs = render::marching_squares(&plane, 96, 96, 30.0);
    assert!(segs.len() > 20, "50% isodose segments: {}", segs.len());
    // The 50% (30 Gy) isodose of a sigma-20 Gaussian lies at r = 20*sqrt(2 ln 2) ≈ 23.55 mm.
    let mean_r: f32 = segs
        .iter()
        .map(|(a, _)| {
            let pat = v.voxel_to_patient(a[0] as f64, a[1] as f64, 20.0);
            ((pat.x * pat.x + pat.y * pat.y) as f32).sqrt()
        })
        .sum::<f32>()
        / segs.len() as f32;
    assert!((mean_r - 23.55).abs() < 1.5, "50% isodose radius {mean_r}");

    // ---- Plan ----
    assert_eq!(study.plans.len(), 1);
    let plan = &study.plans[0];
    assert_eq!(plan.plan_kind, "Ion");
    assert_eq!(plan.n_fractions, Some(30));
    assert!((plan.target_prescription_dose.unwrap() - 60.0).abs() < 1e-6);
    assert_eq!(plan.beams.len(), 2);
    let b1 = &plan.beams[0];
    assert_eq!(b1.radiation_type, "PROTON");
    assert_eq!(b1.n_control_points, 4);
    assert!((b1.energy_min.unwrap() - 120.0).abs() < 1e-6);
    assert!((b1.energy_max.unwrap() - 180.0).abs() < 1e-6);
    assert_eq!(b1.isocenter.unwrap(), Vec3::ZERO);
    assert!((b1.meterset.unwrap() - 120.5).abs() < 1e-6);
    let b2 = &plan.beams[1];
    assert!((b2.gantry_angle.unwrap() - 90.0).abs() < 1e-6);

    // ---- Slice extraction sanity ----
    let mut buf = Vec::new();
    v.extract_slice(ViewPlane::Axial, 20, &mut buf);
    assert_eq!(buf.len(), 96 * 96);
    assert_eq!(buf[48 * 96 + 48], 100);
    v.extract_slice(ViewPlane::Sagittal, 48, &mut buf);
    assert_eq!(buf.len(), 96 * 40);
    // Sagittal row for k=20 is displayed at y = (nz-1) - 20 = 19; target center j=48.
    assert_eq!(buf[19 * 96 + 48], 100);
    v.extract_slice(ViewPlane::Coronal, 48, &mut buf);
    assert_eq!(buf.len(), 96 * 40);
    assert_eq!(buf[19 * 96 + 48], 100);
}

#[test]
fn perf_smoke() {
    let Some(dir) = test_data_dir() else {
        return;
    };
    let progress = Progress::default();
    let study = loader::load_directory(&dir, &progress).unwrap();
    let v = &study.volume;
    let d = &study.doses[0];

    let mut buf = Vec::new();
    let t = std::time::Instant::now();
    for _ in 0..200 {
        v.extract_slice(ViewPlane::Sagittal, 48, &mut buf);
    }
    eprintln!("sagittal slice extraction: {:?} / slice", t.elapsed() / 200);

    let mut plane = Vec::new();
    let t = std::time::Instant::now();
    for _ in 0..50 {
        render::sample_dose_plane(v, d, ViewPlane::Axial, 20, &mut plane);
    }
    eprintln!("dose plane resample: {:?} / slice", t.elapsed() / 50);
}
