//! Opening DICOM written without a file header.
//!
//! A scanner console or a research export that writes the naked data set to
//! disk produces files with no 128-byte preamble, no `DICM` and no File Meta
//! group: the first bytes are already the first element, and nothing in the
//! file says which transfer syntax it is in.
//! `example_data_star/raw/2025.07.28_STAR_Rambam_single_patient_2_CT_RT/`
//! is such an export, and every file in it used to be rejected during the
//! classification scan, so the folder failed to load with "No DICOM files
//! found".
//!
//! These tests rebuild the synthetic phantom study in that shape - once in
//! implicit VR little endian, which is what the real files use, and once in
//! explicit VR little endian - and require the whole study to come back.

use dicom_transfer_syntax_registry::{TransferSyntaxIndex, TransferSyntaxRegistry};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;

const IMPLICIT_VR_LE: &str = "1.2.840.10008.1.2";
const EXPLICIT_VR_LE: &str = "1.2.840.10008.1.2.1";

fn target(tag: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// The synthetic RT study, then a copy of it with every file rewritten as a
/// bare data set in `ts`. Returns the folder of bare files.
fn headerless_study(tag: &str, ts: &str) -> std::path::PathBuf {
    let src = target(&format!("{tag}_src"));
    gen_test_data::generate(&src, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");

    let dst = target(tag);
    std::fs::create_dir_all(&dst).expect("create the folder");
    let ts = TransferSyntaxRegistry
        .get(ts)
        .expect("a known transfer syntax");

    let mut n = 0usize;
    for entry in std::fs::read_dir(&src).expect("the generated folder is readable") {
        let path = entry.expect("a directory entry").path();
        if !path.is_file() {
            continue;
        }
        let obj = dicom_object::open_file(&path).expect("the generator writes valid DICOM");
        let out = std::fs::File::create(dst.join(path.file_name().expect("a file name")))
            .expect("create the bare file");
        obj.into_inner()
            .write_dataset_with_ts(std::io::BufWriter::new(out), ts)
            .expect("write the data set with no file header");
        n += 1;
    }
    assert!(n > 0, "the generator wrote something to strip");

    // What makes these files what they are: no preamble, no magic word.
    let sample = std::fs::read_dir(&dst)
        .expect("readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_file())
        .expect("at least one file");
    let head = std::fs::read(&sample).expect("read");
    assert_ne!(
        &head[128..132],
        b"DICM",
        "{} still has a header",
        sample.display()
    );

    dst
}

/// The whole study loads from files that never say what they are.
#[test]
fn a_folder_of_bare_implicit_vr_data_sets_loads() {
    let dir = headerless_study("test_headerless_implicit", IMPLICIT_VR_LE);

    let study = loader::load_directory(&dir, &Progress::default())
        .expect("a folder of header-less data sets is still a study");

    assert!(study.has_volume(), "the CT series was reconstructed");
    assert_eq!(study.volume.dims, [96, 96, 40], "in full");
    assert_eq!(study.series.len(), 1, "as one series");
    assert_eq!(study.structure_sets.len(), 1, "the structure set came too");
    assert!(
        study.structure_sets[0].rois.len() >= 3,
        "with its contours: {}",
        study.structure_sets[0].rois.len()
    );
    assert_eq!(study.doses.len(), 1, "and the dose");
    assert_eq!(study.plans.len(), 1, "and the plan");
    assert!(
        !study
            .warnings
            .iter()
            .any(|w| w.contains("not readable as DICOM")),
        "and nothing was skipped: {:?}",
        study.warnings
    );
    assert!(
        !study.meta.patient_id.is_empty(),
        "the patient is identified"
    );
}

/// The same, explicit VR - the other encoding a bare data set is found in.
#[test]
fn a_folder_of_bare_explicit_vr_data_sets_loads() {
    let dir = headerless_study("test_headerless_explicit", EXPLICIT_VR_LE);

    let study = loader::load_directory(&dir, &Progress::default())
        .expect("explicit VR without a header loads too");
    assert!(study.has_volume());
    assert_eq!(study.volume.dims, [96, 96, 40]);
    assert_eq!(study.structure_sets.len(), 1);
}

/// The pixels survive the round trip: a bare slice decodes to the same
/// Hounsfield units as the file it was stripped from.
#[test]
fn the_pixels_of_a_bare_data_set_are_unchanged() {
    let with_header = target("test_headerless_pixels_src");
    gen_test_data::generate(&with_header, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    let bare = headerless_study("test_headerless_pixels", IMPLICIT_VR_LE);

    let before = loader::load_directory(&with_header, &Progress::default()).expect("loads");
    let after = loader::load_directory(&bare, &Progress::default()).expect("loads");

    assert_eq!(before.volume.dims, after.volume.dims);
    assert_eq!(before.volume.spacing, after.volume.spacing);
    let mismatches = before
        .volume
        .data
        .iter()
        .zip(after.volume.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(mismatches, 0, "every voxel is the same");
}

/// A folder of files that are not DICOM is still not a study, and no
/// encoding is guessed into existence for them.
#[test]
fn junk_is_not_read_as_a_bare_data_set() {
    let dir = target("test_headerless_junk");
    std::fs::create_dir_all(&dir).expect("create the folder");
    std::fs::write(dir.join("notes.txt"), b"hello, this is not DICOM").expect("write");
    std::fs::write(dir.join("zeros.bin"), [0u8; 4096]).expect("write");
    // Starts with a plausible first tag and then nothing that follows on.
    let mut teaser = vec![0x08, 0x00, 0x05, 0x00];
    teaser.extend_from_slice(&[0xAB; 512]);
    std::fs::write(dir.join("teaser.bin"), teaser).expect("write");

    let err = match loader::load_directory(&dir, &Progress::default()) {
        Err(e) => e,
        Ok(_) => panic!("a folder of junk opens nothing"),
    };
    assert!(err.to_string().contains("No DICOM files"), "{err}");
}
