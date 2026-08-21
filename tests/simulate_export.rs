//! Round-trip test for the study transform simulator and the DICOM export:
//! load synthetic study → apply known transform → verify in-memory result →
//! export as DICOM → reload with the normal loader → verify again.

use rust_dicom_station::dicom_export;
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::geometry::Vec3;
use rust_dicom_station::loader::{self, Progress};
use rust_dicom_station::simulate::{generate_transformed_study, SimParams, SimTransform};

/// Source study for this test, written by the built-in generator.
fn test_data_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_data_sim");
    let _ = std::fs::remove_dir_all(&dir);
    gen_test_data::generate(&dir, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    dir
}

#[test]
fn simulate_and_export_roundtrip() {
    let dir = test_data_dir();
    let progress = Progress::default();
    let src = loader::load_directory(&dir, &progress).expect("source study loads");

    let params = SimParams {
        translation: [10.0, -6.0, 4.0],
        rotation_deg: [0.0, 0.0, 3.0],
        bump_amp: [0.0, 5.0, 0.0],
        bump_center: [0.0, 0.0, 0.0],
        bump_sigma: 30.0,
    };
    let center = src.volume.voxel_to_patient(
        (src.volume.dims[0] as f64 - 1.0) * 0.5,
        (src.volume.dims[1] as f64 - 1.0) * 0.5,
        (src.volume.dims[2] as f64 - 1.0) * 0.5,
    );
    let t = SimTransform::new(&params, center);

    // Transform inverse must round-trip.
    for p in [
        Vec3::ZERO,
        Vec3::new(30.0, -20.0, 15.0),
        Vec3::new(-50.0, 40.0, -25.0),
    ] {
        let rt = (t.unmap(t.map(p)) - p).length();
        assert!(rt < 1e-3, "sim transform inverse round-trip {rt}");
    }

    let sim = generate_transformed_study(&src, &params, &progress);

    // ---- In-memory checks ------------------------------------------------
    // The target center (HU 100 at patient origin in the source) must now be
    // at T(0): sample the simulated volume there.
    let target_new = t.map(Vec3::ZERO);
    let hu = sim
        .volume
        .sample_patient(target_new)
        .expect("inside volume");
    assert!((hu - 100.0).abs() < 2.0, "HU at mapped target center: {hu}");

    // Contours were mapped exactly.
    let ss_src = src.structure_sets.first().unwrap();
    let ss_sim = sim.structure_sets.first().unwrap();
    assert_eq!(ss_src.rois.len(), ss_sim.rois.len());
    let p_src = ss_src.rois[0].contours[0].points[0];
    let p_sim = ss_sim.rois[0].contours[0].points[0];
    assert!(
        (p_sim - t.map(p_src)).length() < 1e-9,
        "contour point mapping"
    );

    // Dose peak moved with the anatomy: D_new(T(0)) ≈ D_old(0) = 60 Gy.
    // (±2 Gy: the peak lands between 4 mm dose-grid points, and the value
    // passes through two trilinear interpolations.)
    let d_new = sim.doses[0].sample(target_new).expect("dose sample");
    assert!((d_new - 60.0).abs() < 2.0, "dose at mapped peak: {d_new}");

    // Isocenter mapped.
    let iso_src = src.plans[0].beams[0].isocenter.unwrap();
    let iso_sim = sim.plans[0].beams[0].isocenter.unwrap();
    assert!(
        (iso_sim - t.map(iso_src)).length() < 1e-9,
        "isocenter mapping"
    );

    // ---- Export → reload ---------------------------------------------------
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/sim_export_test");
    let _ = std::fs::remove_dir_all(&out);
    // Export with edited DICOM attributes, as the export dialog would hand
    // them over: two overridden values and one row switched off.
    let mut params = dicom_export::ExportParams::for_study(&sim);
    fn field<'a>(
        params: &'a mut dicom_export::ExportParams,
        name: &str,
    ) -> &'a mut dicom_export::ExportField {
        params
            .fields
            .iter_mut()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("export field {name} missing"))
    }
    field(&mut params, "PatientID").value = "ANON_EXPORT_1".into();
    field(&mut params, "PatientName").value = "Anon^Export".into();
    field(&mut params, "StudyDescription").enabled = false;
    let n = dicom_export::export_study(&sim, &out, &params, &progress).expect("export succeeds");
    assert!(
        n >= sim.volume.dims[2] + 3,
        "expected CT slices + RS + RD + RP, got {n}"
    );

    let re = loader::load_directory(&out, &progress).expect("exported study reloads");
    assert_eq!(re.volume.dims, sim.volume.dims, "reloaded dims");
    assert!((re.volume.spacing[2] - sim.volume.spacing[2]).abs() < 1e-3);

    // Same HU at the mapped target center after the DICOM round trip.
    let hu_re = re
        .volume
        .sample_patient(target_new)
        .expect("inside reloaded volume");
    assert!(
        (hu_re - 100.0).abs() < 2.0,
        "reloaded HU at mapped target: {hu_re}"
    );

    // Structures round-trip: same count, contour points within DS precision.
    let ss_re = re.structure_sets.first().expect("RTSTRUCT reloads");
    assert_eq!(ss_re.rois.len(), ss_sim.rois.len());
    let target_roi_sim = ss_sim.rois.iter().find(|r| r.name == "TARGET").unwrap();
    let target_roi_re = ss_re.rois.iter().find(|r| r.name == "TARGET").unwrap();
    assert_eq!(target_roi_re.contours.len(), target_roi_sim.contours.len());
    let a = target_roi_sim.contours[0].points[0];
    let b = target_roi_re.contours[0].points[0];
    assert!(
        (a - b).length() < 1e-3,
        "contour DS round-trip error {}",
        (a - b).length()
    );
    assert_eq!(target_roi_re.roi_type, "PTV");

    // Dose round-trip (16-bit quantization ⇒ ~1e-3 relative tolerance).
    assert_eq!(re.doses.len(), 1);
    let d_re = re.doses[0]
        .sample(target_new)
        .expect("reloaded dose sample");
    assert!(
        (d_re - d_new).abs() < 0.05,
        "dose round-trip {d_re} vs {d_new}"
    );

    // Plan round-trip.
    assert_eq!(re.plans.len(), 1);
    let plan = &re.plans[0];
    assert_eq!(plan.plan_kind, "Ion");
    assert_eq!(plan.beams.len(), 2);
    assert!((plan.target_prescription_dose.unwrap() - 60.0).abs() < 1e-6);
    let iso_re = plan.beams[0].isocenter.unwrap();
    assert!((iso_re - iso_sim).length() < 1e-3, "isocenter round-trip");

    // The edited tags landed in the files, and the disabled row was skipped.
    assert_eq!(re.meta.patient_id, "ANON_EXPORT_1", "PatientID override");
    assert_eq!(re.meta.patient_name, "Anon^Export", "PatientName override");
    assert!(
        re.meta.study_description.is_empty(),
        "disabled StudyDescription must not be written, got {:?}",
        re.meta.study_description
    );

    // Frame of reference preserved so the pair stays comparable.
    assert_eq!(
        re.volume.frame_of_reference_uid,
        sim.volume.frame_of_reference_uid
    );

    eprintln!("round-trip OK: {n} files, HU {hu_re:.1}, dose {d_re:.2} Gy at mapped target");
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&dir);
}
