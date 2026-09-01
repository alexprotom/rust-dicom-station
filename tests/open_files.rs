//! Opening DICOM that does not add up to a volume.
//!
//! Three things have to hold for "just open this file" to work: a selection
//! of individual files loads at all; a selection with no reconstructable
//! image series loads *without* an error, carrying whatever it does hold;
//! and adding an image series to such a dataset afterwards fills it in.

use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;

/// The synthetic RT study, generated once per test into its own folder.
fn phantom(tag: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    gen_test_data::generate(&dir, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    dir
}

/// Every generated file whose name starts with one of `prefixes`.
fn files(dir: &std::path::Path, prefixes: &[&str]) -> Vec<std::path::PathBuf> {
    let mut out: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .expect("the generated folder is readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            prefixes.iter().any(|pre| name.starts_with(pre))
        })
        .collect();
    out.sort();
    out
}

#[test]
fn rt_images_open_on_their_own_without_a_volume() {
    let dir = phantom("test_open_rtimage");
    let picked = files(&dir, &["RI_", "DX_"]);
    assert_eq!(
        picked.len(),
        2,
        "the generator wrote one RTIMAGE and one DX"
    );

    let study = loader::load_files(&picked, "two images", &Progress::default())
        .expect("RT images open even though they build no volume");

    assert!(!study.has_volume(), "there is nothing to reconstruct");
    assert!(study.series.is_empty(), "and therefore no image series");
    assert_eq!(study.planar_images.len(), 2, "both images are readable");
    assert!(
        study.planar_images.iter().any(|p| p.modality == "RTIMAGE"),
        "the RT image kept its modality: {:?}",
        study
            .planar_images
            .iter()
            .map(|p| &p.modality)
            .collect::<Vec<_>>()
    );
    assert!(
        study.planar_images.iter().all(|p| p.rows > 0 && p.cols > 0),
        "with pixels"
    );
    assert!(
        study
            .warnings
            .iter()
            .any(|w| w.contains("no image series") || w.to_lowercase().contains("no image series")),
        "and the dataset says why it has no images to scroll: {:?}",
        study.warnings
    );
    // The empty volume must be inert rather than degenerate: nothing may
    // divide by its spacing or index its data.
    assert_eq!(study.volume.dims, [0, 0, 0]);
    assert!(study.volume.spacing.iter().all(|s| *s > 0.0));
    assert!(study.volume.get(0, 0, 0).is_none());
}

#[test]
fn a_structure_set_opens_on_its_own() {
    let dir = phantom("test_open_rtstruct");
    let picked = files(&dir, &["RS_"]);
    assert_eq!(picked.len(), 1);

    let study = loader::load_files(&picked, "one structure set", &Progress::default())
        .expect("a structure set opens with no images behind it");
    assert!(!study.has_volume());
    assert_eq!(study.structure_sets.len(), 1);
    assert!(
        !study.structure_sets[0].rois.is_empty(),
        "with its contours"
    );
    // Frame-of-reference warnings compare against the volume; with no volume
    // there is nothing to disagree with, so none may be invented.
    assert!(
        !study
            .warnings
            .iter()
            .any(|w| w.contains("frame of reference")),
        "no mismatch is claimed against a volume that does not exist: {:?}",
        study.warnings
    );
}

#[test]
fn a_single_slice_opens_as_a_one_slice_volume() {
    let dir = phantom("test_open_one_slice");
    let picked = files(&dir, &["CT_000"]);
    assert_eq!(picked.len(), 1, "exactly one CT file");

    let study = loader::load_files(&picked, "one slice", &Progress::default())
        .expect("a single positioned slice is still a volume");
    assert!(study.has_volume(), "one slice is a volume of one slice");
    assert_eq!(study.volume.dims[2], 1);
    assert_eq!(study.series.len(), 1);
}

#[test]
fn adding_an_image_series_afterwards_completes_the_dataset() {
    let dir = phantom("test_open_then_ct");

    // Start from the RT objects alone.
    let mut study = loader::load_files(
        &files(&dir, &["RS_", "RI_"]),
        "objects",
        &Progress::default(),
    )
    .expect("the objects open");
    assert!(!study.has_volume());

    // Then the images they were drawn on, exactly as *Add DICOM folder*
    // would deliver them.
    let ct = loader::load_files(&files(&dir, &["CT_"]), "the CT", &Progress::default())
        .expect("the CT opens");
    assert!(ct.has_volume());
    let notes = loader::merge_study(&mut study, ct);
    assert!(notes.is_empty(), "nothing was skipped: {notes:?}");

    assert_eq!(study.series.len(), 1, "the CT series joined the dataset");
    assert!(
        !study.series[0].files.is_empty(),
        "with the files to load it from"
    );
    assert_eq!(study.structure_sets.len(), 1, "the objects are still there");
    assert_eq!(study.planar_images.len(), 1);
    // merge_study deliberately leaves the displayed volume alone; the
    // application is what switches to the new series (absorb_loaded_study),
    // and it can only do that because the series arrived with its files.
    let (vol, _, _) = loader::load_series_volume(&study.series[0], &Progress::default())
        .expect("the merged series loads");
    assert!(!vol.is_empty());
    assert_eq!(
        vol.frame_of_reference_uid, study.structure_sets[0].frame_of_reference_uid,
        "and the contours belong to it"
    );
}

#[test]
fn a_folder_of_nothing_but_rt_objects_loads_as_a_directory_too() {
    let src = phantom("test_open_objects_dir");
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_open_objects_only");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the folder");
    for f in files(&src, &["RS_", "RP_", "RI_"]) {
        std::fs::copy(&f, dir.join(f.file_name().expect("a file name"))).expect("copy");
    }

    let study = loader::load_directory(&dir, &Progress::default())
        .expect("a folder with no image series is not an error");
    assert!(!study.has_volume());
    assert_eq!(study.structure_sets.len(), 1);
    assert_eq!(study.plans.len(), 1);
    assert_eq!(study.planar_images.len(), 1);
    assert!(
        !study.meta.patient_id.is_empty(),
        "the patient is still identified: '{}' / '{}'",
        study.meta.patient_name,
        study.meta.patient_id
    );
}

#[test]
fn an_empty_selection_is_still_an_error() {
    let err = match loader::load_files(&[], "nothing", &Progress::default()) {
        Err(e) => e,
        Ok(_) => panic!("there is nothing to open"),
    };
    assert!(err.to_string().contains("No files"), "{err}");

    let dir = phantom("test_open_junk");
    let junk = dir.join("not-dicom.txt");
    std::fs::write(&junk, b"hello").expect("write");
    let err = match loader::load_files(&[junk], "junk", &Progress::default()) {
        Err(e) => e,
        Ok(_) => panic!("a file that is not DICOM opens nothing"),
    };
    assert!(err.to_string().contains("No DICOM files"), "{err}");
}
