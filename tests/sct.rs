//! A synthetic CT end to end: the built-in phantom study, a folder of
//! TotalSynth checkpoints (the miniature network of the engine's parity
//! tests, registered as the user's own), the series written and read back
//! into the study, and the evaluation against the CT it came from.

use std::path::{Path, PathBuf};

use dicom_dictionary_std::tags;
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::registration::resample_through;
use rust_dicom_station::sct::{self, eval, BodySource, SctRequest};
use rust_dicom_station::synth::{self, App};

fn scratch(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join(format!("sct_{name}"));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

#[test]
fn a_synthetic_ct_joins_its_study_and_is_evaluated() {
    let base = scratch("study");
    let data = base.join("phantom");
    gen_test_data::generate(
        &data,
        &GenParams {
            extras: false,
            ..GenParams::default()
        },
        &Progress::default(),
    )
    .unwrap();
    let study = loader::load_directory(&data, &Progress::default()).unwrap();
    let source = study.series[study.active_series].clone();
    assert_eq!(source.modality, "CT");

    // The user's checkpoints: two members of the miniature network.
    let folder = base.join("My pelvis model");
    std::fs::create_dir_all(&folder).unwrap();
    std::fs::copy(fixture("synth-unetpp-cv0.pt"), folder.join("CV_0.pt")).unwrap();
    std::fs::copy(fixture("synth-unetpp-cv1.pt"), folder.join("CV_1.pt")).unwrap();
    let reg = synth::custom::register(std::slice::from_ref(&folder));
    let key = *reg[0].1.as_ref().unwrap();
    let app = App::from_key(key).unwrap();

    let models = base.join("models");
    let out = base.join("derived");
    let req = SctRequest {
        app,
        opts: synth::Options {
            checkpoints: vec![0, 1],
            tta: 1,
            uncertainty: true,
            device: rust_dicom_station::nn::device::DevicePref::Cpu,
        },
        body: BodySource::Classical,
        series_number: sct::next_series_number(&study),
    };
    let o = sct::synthesize(
        &study.volume,
        &source,
        &req,
        &models,
        &out,
        &Progress::default(),
    )
    .unwrap();
    assert_eq!(o.members, 4);
    assert_eq!(o.volume.dims, study.volume.dims);
    assert_eq!(o.series.files.len(), study.volume.dims[2]);
    assert!(o.series.files[0].starts_with(out.join(&source.study_uid)));
    // Air outside the body, something else inside it.
    assert_eq!(o.volume.index(1, 1, 20), -1024);
    assert!(o.volume.data.iter().any(|&v| v != -1024));
    let (sd_vol, sd_series) = o.uncertainty.as_ref().unwrap();
    assert_eq!(sd_series.files.len(), study.volume.dims[2]);
    assert!(sd_vol.data.iter().any(|&v| v > 0));

    // Read back: two new series of the same study, the sCT a CT.
    assert_eq!(o.study.series.len(), 2);
    let s = o
        .study
        .series
        .iter()
        .find(|s| s.uid == o.series.series_uid)
        .unwrap();
    assert_eq!(s.modality, "CT");
    assert_eq!(s.study_uid, source.study_uid);
    assert_eq!(s.patient_id, source.patient_id);
    assert_eq!(s.series_number, Some(req.series_number));
    assert!(s.description.starts_with("sCT ("), "{}", s.description);
    let (vol, _, _) = loader::load_series_volume(s, &Progress::default()).unwrap();
    assert_eq!(vol.data, o.volume.data, "the files hold the voxels");
    assert_eq!(
        vol.frame_of_reference_uid,
        study.volume.frame_of_reference_uid
    );

    // The files say what they are and where they came from.
    let f = rust_dicom_station::dicomfile::open_header(&o.series.files[10]).unwrap();
    let image_type = f
        .element(tags::IMAGE_TYPE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(image_type, "DERIVED\\SECONDARY");
    let derivation = f
        .element(tags::DERIVATION_DESCRIPTION)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(derivation.contains("CV_0, CV_1"), "{derivation}");
    let items = f
        .element(tags::SOURCE_IMAGE_SEQUENCE)
        .unwrap()
        .items()
        .unwrap();
    let referenced = loader::str_of(&items[0], tags::REFERENCED_SOP_INSTANCE_UID).unwrap();
    let source_uids: Vec<String> = source
        .files
        .iter()
        .map(|p| {
            let h = rust_dicom_station::dicomfile::open_header(p).unwrap();
            loader::str_of(&h, tags::SOP_INSTANCE_UID).unwrap()
        })
        .collect();
    assert!(source_uids.contains(&referenced));

    // Merged into the workspace, as the module does.
    let mut ws = study.clone();
    let before = ws.series.len();
    let notes = loader::merge_study(&mut ws, o.study.clone());
    assert!(notes.is_empty(), "{notes:?}");
    assert_eq!(ws.series.len(), before + 2);
    assert_eq!(sct::next_series_number(&ws), req.series_number + 2);

    // Against the CT it was made from, on the same grid.
    let reference = resample_through(&study.volume, &o.volume.grid(), None);
    let body = rust_dicom_station::bodymask::contour_body(
        &study.volume,
        &rust_dicom_station::bodymask::BodyParams::default(),
        &models,
        &Progress::default(),
    )
    .unwrap();
    let r = eval::evaluate(
        &o.volume,
        &reference,
        Some(&body.mask),
        &eval::EvalOptions {
            wet: Some(eval::Hlut::default()),
            ..Default::default()
        },
        &models,
        &Progress::default(),
    )
    .unwrap();
    assert_eq!(r.overall.voxels, body.voxels as usize);
    assert!(r.overall.mae.is_finite() && r.overall.mae > 0.0);
    assert!(r.ssim.is_finite() && r.ssim < 1.0);
    assert_eq!(r.wet.len(), 3);
    assert!(eval::csv(&r).contains("image,all,ssim,"));
    synth::custom::register(&[]);
    let _ = std::fs::remove_dir_all(&base);
}
