//! End-to-end test of the Tools ▶ Anonymize DICOM folder pipeline: generate
//! the synthetic RT study, anonymize it into a second folder, and verify
//! that identity is gone, UIDs changed consistently, the DICOM reference
//! chains still resolve, and the image content is untouched.

use rust_dicom_station::anonymize::{self, ApplyParams};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader::{self, Progress};

#[test]
fn anonymize_roundtrip() {
    let base = std::env::temp_dir().join(format!("anon_test_{}", std::process::id()));
    let src = base.join("src");
    let dst = base.join("dst");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&src).unwrap();

    let progress = Progress::default();
    let params = GenParams { extras: true, ..GenParams::default() };
    gen_test_data::generate(&src, &params, &progress).expect("generate test study");

    // --- scan ---
    let scan = anonymize::scan(&src, &progress).expect("scan");
    assert!(!scan.files.is_empty());
    assert!(scan.uid_count > 0, "expected private-root UIDs to remap");
    let name = scan
        .findings
        .iter()
        .find(|f| f.name == "PatientName")
        .expect("PatientName finding");
    assert!(
        name.values.iter().any(|v| v.contains("PHANTOM")),
        "current values should show the original name, got {:?}",
        name.values
    );
    assert!(name.suggested.starts_with("anon_"), "{}", name.suggested);

    // --- apply with the suggested replacements ---
    let apply = ApplyParams {
        replacements: scan
            .findings
            .iter()
            .filter(|f| f.enabled)
            .map(|f| (f.tag, f.vr, f.replacement.clone()))
            .collect(),
        remove_private: true,
        remap_uids: true,
        mark_deidentified: true,
        out_dir: Some(dst.clone()),
    };
    let n = anonymize::apply(&scan.files, &scan.root, &apply, &progress).expect("apply");
    assert_eq!(n, scan.files.len());

    // --- reload both and compare ---
    let orig = loader::load_directory(&src, &progress).expect("load original");
    let anon = loader::load_directory(&dst, &progress).expect("load anonymized");

    // Identity replaced.
    assert!(anon.meta.patient_name.starts_with("anon_"), "{}", anon.meta.patient_name);
    assert!(anon.meta.patient_id.starts_with("anon_"), "{}", anon.meta.patient_id);
    assert_eq!(anon.meta.study_date, "20000101");

    // UIDs changed but consistently: the structure set still references the
    // image series, and dose ▶ plan ▶ structure set still resolves.
    let se = &anon.series[anon.active_series];
    assert_ne!(se.uid, orig.series[orig.active_series].uid);
    let ss = anon
        .structure_sets
        .iter()
        .find(|s| s.referenced_series_uid == se.uid)
        .expect("RTSTRUCT ▶ series link survives anonymization");
    let plan = anon
        .plans
        .iter()
        .find(|p| p.referenced_structset_uid == ss.sop_instance_uid)
        .expect("RTPLAN ▶ RTSTRUCT link survives");
    assert!(
        anon.doses
            .iter()
            .any(|d| d.referenced_plan_uid == plan.sop_instance_uid),
        "RTDOSE ▶ RTPLAN link survives"
    );

    // Pixel data byte-identical, geometry preserved.
    assert_eq!(anon.volume.dims, orig.volume.dims);
    assert_eq!(anon.volume.data, orig.volume.data);
    // Frame of reference remapped away from the original.
    assert_ne!(
        anon.volume.frame_of_reference_uid,
        orig.volume.frame_of_reference_uid
    );

    // Contours still overlay the volume (same geometry, new UIDs).
    assert_eq!(ss.rois.len(), orig.structure_sets[0].rois.len());

    let _ = std::fs::remove_dir_all(&base);
}
