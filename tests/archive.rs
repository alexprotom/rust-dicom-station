//! The local archive, end to end: file a study into it, list it without
//! opening a DICOM file, load it back into a dataset the way the PACS window
//! does, draw something on it, and send the derived objects back so they land
//! in the same patient and study rather than beside them.
//!
//! This is the whole round trip the archive exists for, so it is tested as
//! one path rather than as four isolated calls.

use rust_dicom_station::archive::Archive;
use rust_dicom_station::dicom_export::{self, ExportParams};
use rust_dicom_station::dicomseg::SegSeries;
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::segmentation::Segmentation;

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A solid ellipsoid around the volume centre - something with a non-zero
/// voxel count, which is what makes the writer emit the object at all.
fn ball(dims: [usize; 3], radius: f64) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let c = [
        (nx as f64 - 1.0) * 0.5,
        (ny as f64 - 1.0) * 0.5,
        (nz as f64 - 1.0) * 0.5,
    ];
    let mut m = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d = ((i as f64 - c[0]).powi(2)
                    + (j as f64 - c[1]).powi(2)
                    + (k as f64 - c[2]).powi(2))
                .sqrt();
                if d <= radius {
                    m[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
    }
    m
}

#[test]
fn a_study_files_lists_loads_and_takes_back_what_was_drawn_on_it() {
    let src = scratch("test_archive_src");
    let written = gen_test_data::generate(&src, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");

    let root = scratch("test_archive_root");
    let archive = Archive::new(&root);

    // ---- file it --------------------------------------------------------
    let sum = archive
        .import(&src, &Progress::default())
        .expect("import succeeds");
    assert_eq!(sum.stored, written, "every generated file was filed");
    assert_eq!(sum.skipped, 0, "everything the generator wrote is DICOM");
    assert_eq!(sum.duplicates, 0);
    assert_eq!(
        (sum.patients, sum.studies),
        (1, 1),
        "one patient, one study"
    );

    // ---- list it --------------------------------------------------------
    let patients = archive.scan().expect("scan succeeds");
    assert_eq!(patients.len(), 1);
    let p = &patients[0];
    assert_eq!(p.title(), "PHANTOM RT (RTTEST001)", "the listed identity");
    assert_eq!(p.studies.len(), 1);
    let entry = p.studies[0].clone();
    assert_eq!(entry.files, written, "the sidecar counts what is there");
    assert!(
        entry.modalities.iter().any(|m| m == "CT")
            && entry.modalities.iter().any(|m| m == "RTSTRUCT"),
        "the sidecar lists the modalities it holds: {:?}",
        entry.modalities
    );
    let line = entry.describe();
    assert!(
        line.contains("CT") && line.ends_with(&format!("{written} files")),
        "the one-line description the window shows: {line}"
    );

    // Importing the same folder again must be a no-op, not a second copy.
    let again = archive
        .import(&src, &Progress::default())
        .expect("re-import succeeds");
    assert_eq!(
        (again.stored, again.duplicates),
        (0, written),
        "the same SOP Instance UIDs are recognised"
    );

    // ---- take it into a dataset -----------------------------------------
    // The PACS window loads an archived study folder with the ordinary
    // directory scanner, so that is exactly what is asserted here.
    let mut study =
        loader::load_directory(&entry.dir, &Progress::default()).expect("the archived study loads");
    assert!(!study.series.is_empty(), "the CT came back");
    assert_eq!(
        study.series[study.active_series].study_uid, entry.study_uid,
        "the loaded study is the one the archive listed"
    );
    let structs_before = study.structure_sets.len();

    // ---- draw on it -----------------------------------------------------
    let dims = study.volume.dims;
    let mut ser = SegSeries::new(
        "Archive QA".into(),
        study.volume.grid(),
        study.series[study.active_series].uid.clone(),
        study.series[study.active_series].study_uid.clone(),
    );
    ser.segs.push(Segmentation::from_mask(
        "Ball".into(),
        [220, 40, 40],
        dims,
        ball(dims, 8.0),
    ));
    study.seg_series.push(ser);

    // ---- send it back ---------------------------------------------------
    let derived = scratch("test_archive_derived");
    let params = ExportParams::for_study(&study);
    let n = dicom_export::export_derived(&study, &derived, &params, &Progress::default())
        .expect("the derived export succeeds");
    assert_eq!(
        n,
        structs_before + 1,
        "one file per structure set plus the new segmentation series, and nothing else"
    );
    for f in std::fs::read_dir(&derived).unwrap().filter_map(|e| e.ok()) {
        let name = f.file_name().to_string_lossy().to_string();
        assert!(
            name.starts_with("RS_") || name.starts_with("SEG_"),
            "no image data is re-sent, found {name}"
        );
    }

    let up = archive
        .import(&derived, &Progress::default())
        .expect("upload succeeds");
    assert_eq!(up.stored, n, "the derived objects are new instances");
    assert_eq!(
        (up.patients, up.studies),
        (1, 1),
        "they land in one patient and one study - the ones already there"
    );

    let patients = archive.scan().expect("rescan succeeds");
    assert_eq!(patients.len(), 1, "no second patient was invented");
    let p = &patients[0];
    assert_eq!(p.studies.len(), 1, "no second study was invented");
    assert_eq!(
        p.studies[0].study_uid, entry.study_uid,
        "the Study Instance UID is kept, which is what files them together"
    );
    assert_eq!(
        p.studies[0].files,
        written + n,
        "the archive grew by the derived objects"
    );
    assert!(
        p.studies[0].modalities.iter().any(|m| m == "SEG"),
        "the segmentation is now part of the study: {:?}",
        p.studies[0].modalities
    );

    // And the study reads back with what was drawn on it.
    let back = loader::load_directory(&p.studies[0].dir, &Progress::default())
        .expect("the enriched study loads");
    assert_eq!(back.seg_series.len(), 1, "the segmentation came back");
    assert_eq!(back.seg_series[0].segs.len(), 1);
    assert_eq!(back.seg_series[0].segs[0].name, "Ball");
    assert_eq!(
        back.structure_sets.len(),
        structs_before * 2,
        "the re-sent structure set is a new instance beside the original"
    );

    // ---- and it can be taken out again ----------------------------------
    archive.remove(&p.dir).expect("removing a patient succeeds");
    assert!(archive.scan().expect("scan succeeds").is_empty());
}
