//! Shared fixtures: the synthetic phantom filed as a 4D acquisition on disk.
//!
//! `gen_test_data` writes one study; a 4D study is three of them with the
//! target moved between phases, rewritten so they share one study and one
//! frame of reference and carry their phase in the series description. The
//! result is a folder the loader recognises as a 4D group, which is what both
//! the workflow tests and the MCP tests need.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use dicom_core::{Tag, VR};
use dicom_dictionary_std::tags;
use rust_dicom_station::dicom_export::copy_patched;
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;

/// A fresh folder under `target/`.
pub fn target_dir(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the test folder");
    dir
}

/// Phase labels used by every fixture, in temporal order.
pub const PHASES: [&str; 3] = ["4DCT 0%", "4DCT 50%", "4DCT 90%"];

/// Write a three-phase 4D study into `dir/4dct`: the target shifted by
/// `shifts_y[i]` mm on phase `i`. Phase 0 keeps its RTSTRUCT (with TARGET,
/// BODY and CORD), dose and plan; the other phases are image series only.
/// Returns the folder.
pub fn fourd_folder(dir: &Path, shifts_y: [f64; 3]) -> PathBuf {
    let out = dir.join("4dct");
    std::fs::create_dir_all(&out).unwrap();
    let mut study_uid = String::new();
    let mut for_uid = String::new();
    for (i, shift) in shifts_y.iter().enumerate() {
        let src = dir.join(format!("phase{i}_src"));
        let params = GenParams {
            target_shift_y: *shift,
            extras: false,
            ..GenParams::default()
        };
        gen_test_data::generate(&src, &params, &Progress::default()).expect("phantom generated");
        let study = loader::load_directory(&src, &Progress::default()).expect("phantom loads");
        let series = &study.series[study.active_series];
        if i == 0 {
            study_uid = series.study_uid.clone();
            for_uid = study.volume.frame_of_reference_uid.clone();
        }
        let series_uid = if i == 0 {
            series.uid.clone()
        } else {
            format!("{}.{}", series.uid, i + 1)
        };
        let mut set: Vec<(Tag, VR, String)> = vec![
            (tags::STUDY_INSTANCE_UID, VR::UI, study_uid.clone()),
            (tags::FRAME_OF_REFERENCE_UID, VR::UI, for_uid.clone()),
            (tags::SERIES_DESCRIPTION, VR::LO, PHASES[i].to_string()),
            (tags::SERIES_NUMBER, VR::IS, (i + 1).to_string()),
        ];
        if i > 0 {
            set.push((tags::SERIES_INSTANCE_UID, VR::UI, series_uid.clone()));
        }
        for (k, f) in series.files.iter().enumerate() {
            let name = format!("phase{i}_{k:04}.dcm");
            // Distinct SOP instances per phase, or the loader would see one.
            let mut s = set.clone();
            if i > 0 {
                let sop = loader::str_of(
                    &rust_dicom_station::dicomfile::open_header(f).unwrap(),
                    tags::SOP_INSTANCE_UID,
                )
                .unwrap_or_default();
                s.push((tags::SOP_INSTANCE_UID, VR::UI, format!("{sop}.{}", i + 1)));
            }
            copy_patched(f, &out.join(name), &s).expect("slice copied");
        }
        if i == 0 {
            // The RT objects of the reference phase, study UID unified.
            for e in std::fs::read_dir(&src).unwrap().flatten() {
                let p = e.path();
                if series.files.contains(&p) || !p.is_file() {
                    continue;
                }
                let name = p.file_name().unwrap().to_string_lossy().to_string();
                let s = vec![(tags::STUDY_INSTANCE_UID, VR::UI, study_uid.clone())];
                copy_patched(&p, &out.join(name), &s).expect("RT object copied");
            }
        }
        let _ = std::fs::remove_dir_all(&src);
    }
    out
}

/// Give every file of a folder the identifiers of a (fictional) named
/// patient, so the PHI gate has something to find.
pub fn name_the_patient(folder: &Path, name: &str, id: &str, birth: &str) {
    for e in std::fs::read_dir(folder).unwrap().flatten() {
        let p = e.path();
        if !p.is_file() {
            continue;
        }
        let tmp = p.with_extension("tmp");
        copy_patched(
            &p,
            &tmp,
            &[
                (tags::PATIENT_NAME, VR::PN, name.to_string()),
                (tags::PATIENT_ID, VR::LO, id.to_string()),
                (tags::PATIENT_BIRTH_DATE, VR::DA, birth.to_string()),
                (
                    tags::REFERRING_PHYSICIAN_NAME,
                    VR::PN,
                    "HOUSE^GREGORY".to_string(),
                ),
                (
                    tags::INSTITUTION_NAME,
                    VR::LO,
                    "Princeton Plainsboro".to_string(),
                ),
            ],
        )
        .expect("patched");
        std::fs::rename(&tmp, &p).unwrap();
    }
}
