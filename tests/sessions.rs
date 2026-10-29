//! Treatment sessions from files, as a setup imaging system writes them.
//!
//! One treatment day is built from the synthetic RT study: a daily CT, an
//! imaging registration that names that CT by its *Series* Instance UID in
//! the slot of a SOP Instance UID, a treatment registration that names the
//! ion plan with the RT Plan class and has the plan's UID as its own frame
//! of reference, two RT Image pairs with nothing but their numbers and
//! times to tie them in, and two records of one interrupted delivery. The
//! loader reads it and [`sessions`] puts it together; nothing of a real
//! patient is in it, and every UID is made here.

use std::path::{Path, PathBuf};

use dicom_core::value::{PrimitiveValue, Value};
use dicom_core::{DataElement, Length, Tag, VR};
use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;
use rust_dicom_station::dicom_export::new_uid;
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader::{self, LoadedStudy};
use rust_dicom_station::progress::Progress;
use rust_dicom_station::sessions::{self, RegKind};

const DAY: &str = "20240917";
const SOP_RT_PLAN: &str = "1.2.840.10008.5.1.4.1.1.481.5";
const SOP_RT_ION_PLAN: &str = "1.2.840.10008.5.1.4.1.1.481.8";

fn target(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the test folder can be made");
    dir
}

fn put_str(o: &mut InMemDicomObject, tag: Tag, vr: VR, v: &str) {
    o.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
}

fn put_seq(o: &mut InMemDicomObject, tag: Tag, items: Vec<InMemDicomObject>) {
    o.put(DataElement::new(
        tag,
        VR::SQ,
        Value::new_sequence(items, Length::UNDEFINED),
    ));
}

fn reference(class: &str, uid: &str) -> InMemDicomObject {
    let mut it = InMemDicomObject::new_empty();
    put_str(&mut it, tags::REFERENCED_SOP_CLASS_UID, VR::UI, class);
    put_str(&mut it, tags::REFERENCED_SOP_INSTANCE_UID, VR::UI, uid);
    it
}

fn open(path: &Path) -> dicom_object::DefaultDicomObject {
    dicom_object::open_file(path).expect("a phantom file opens")
}

fn new_ids(o: &mut InMemDicomObject, series_number: i64) {
    put_str(o, tags::SOP_INSTANCE_UID, VR::UI, &new_uid());
    put_str(o, tags::SERIES_INSTANCE_UID, VR::UI, &new_uid());
    put_str(o, tags::SERIES_NUMBER, VR::IS, &series_number.to_string());
}

/// The phantom's CT again, as the day's CT: new series, new frame of
/// reference, made on [`DAY`]. Returns its Series Instance UID.
fn daily_ct(dir: &Path) -> String {
    let series_uid = new_uid();
    let for_uid = new_uid();
    let mut slices: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("CT_"))
        .collect();
    slices.sort();
    for f in slices {
        let mut o = open(&f);
        put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, &series_uid);
        put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, &new_uid());
        put_str(&mut o, tags::FRAME_OF_REFERENCE_UID, VR::UI, &for_uid);
        put_str(&mut o, tags::SERIES_NUMBER, VR::IS, "701");
        put_str(&mut o, tags::SERIES_DATE, VR::DA, DAY);
        put_str(&mut o, tags::SERIES_TIME, VR::TM, "103701.117");
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        o.write_to_file(dir.join(format!("daily_{name}"))).unwrap();
    }
    series_uid
}

/// One RT Image of a pair: no frame of reference, plan or beam, only its
/// numbers and its times.
fn rt_image(dir: &Path, name: &str, series: i64, instance: i64, at: &str, review: &str) {
    let mut o = open(&dir.join("RI_synth.dcm"));
    new_ids(&mut o, series);
    put_str(&mut o, tags::INSTANCE_NUMBER, VR::IS, &instance.to_string());
    o.remove_element(tags::FRAME_OF_REFERENCE_UID);
    put_str(&mut o, tags::ACQUISITION_DATE, VR::DA, DAY);
    put_str(&mut o, tags::ACQUISITION_TIME, VR::TM, at);
    put_str(&mut o, tags::REVIEW_DATE, VR::DA, DAY);
    put_str(&mut o, tags::REVIEW_TIME, VR::TM, review);
    o.write_to_file(dir.join(name)).unwrap();
}

/// A registration made at `time` referencing `refs` (class, UID) from its
/// Registration Sequence, with a transformation comment and its own frame
/// of reference `own_for`.
fn registration(
    dir: &Path,
    name: &str,
    series: i64,
    time: &str,
    own_for: &str,
    refs: &[(&str, &str)],
    comment: &str,
) {
    let mut o = open(&dir.join("REG_synth.dcm"));
    new_ids(&mut o, series);
    o.remove_element(tags::CONTENT_DESCRIPTION);
    put_str(&mut o, tags::CONTENT_DATE, VR::DA, DAY);
    put_str(&mut o, tags::CONTENT_TIME, VR::TM, time);
    put_str(&mut o, tags::FRAME_OF_REFERENCE_UID, VR::UI, own_for);
    let mut items: Vec<InMemDicomObject> = loader::items_of(&o, tags::REGISTRATION_SEQUENCE)
        .unwrap()
        .to_vec();
    for it in &mut items {
        put_seq(
            it,
            tags::REFERENCED_IMAGE_SEQUENCE,
            refs.iter().map(|(c, u)| reference(c, u)).collect(),
        );
        let mut mr: Vec<InMemDicomObject> =
            loader::items_of(it, tags::MATRIX_REGISTRATION_SEQUENCE)
                .unwrap()
                .to_vec();
        for m in &mut mr {
            put_str(
                m,
                tags::FRAME_OF_REFERENCE_TRANSFORMATION_COMMENT,
                VR::LO,
                comment,
            );
        }
        put_seq(it, tags::MATRIX_REGISTRATION_SEQUENCE, mr);
    }
    put_seq(&mut o, tags::REGISTRATION_SEQUENCE, items);
    o.write_to_file(dir.join(name)).unwrap();
}

/// A record of the plan delivered at `time`, its beams terminated `status`.
fn record(dir: &Path, name: &str, series: i64, time: &str, plan: &str, status: &str) {
    let mut o = open(&dir.join("RT_record_synth.dcm"));
    new_ids(&mut o, series);
    put_str(&mut o, tags::TREATMENT_DATE, VR::DA, DAY);
    put_str(&mut o, tags::TREATMENT_TIME, VR::TM, time);
    put_seq(
        &mut o,
        tags::REFERENCED_RT_PLAN_SEQUENCE,
        vec![reference(SOP_RT_ION_PLAN, plan)],
    );
    let mut beams: Vec<InMemDicomObject> =
        loader::items_of(&o, tags::TREATMENT_SESSION_ION_BEAM_SEQUENCE)
            .unwrap()
            .to_vec();
    for b in &mut beams {
        put_str(b, tags::TREATMENT_TERMINATION_STATUS, VR::CS, status);
    }
    put_seq(&mut o, tags::TREATMENT_SESSION_ION_BEAM_SEQUENCE, beams);
    o.write_to_file(dir.join(name)).unwrap();
}

/// The day, written into `target/sessions`; returns the folder and the
/// daily CT's series UID.
fn treatment_day() -> (PathBuf, String) {
    let dir = target("sessions");
    gen_test_data::generate(&dir, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    let plan = loader::str_of(&open(&dir.join("RP_synth.dcm")), tags::SOP_INSTANCE_UID).unwrap();
    let daily = daily_ct(&dir);

    registration(
        &dir,
        "REG_imaging.dcm",
        86,
        "104038",
        &daily,
        &[("1.2.840.10008.5.1.4.1.1.2", &daily)],
        "imaging position coordinate system",
    );
    rt_image(&dir, "RI_setup_a.dcm", 212, 3, "105145", "105402");
    rt_image(&dir, "RI_setup_b.dcm", 212, 3, "105146", "105402");
    registration(
        &dir,
        "REG_treatment.dcm",
        95,
        "105403",
        &plan,
        &[(SOP_RT_PLAN, &plan)],
        "treatment position coordinate system",
    );
    record(&dir, "RT_record_1.dcm", 96, "105418", &plan, "MACHINE");
    record(&dir, "RT_record_2.dcm", 97, "105530", &plan, "NORMAL");
    rt_image(&dir, "RI_verify_a.dcm", 213, 3, "105757", "105802");
    rt_image(&dir, "RI_verify_b.dcm", 213, 3, "105757", "105802");
    // The phantom's own objects were only the templates.
    for f in ["REG_synth.dcm", "RT_record_synth.dcm", "RI_synth.dcm"] {
        std::fs::remove_file(dir.join(f)).unwrap();
    }
    (dir, daily)
}

fn load(dir: &Path) -> LoadedStudy {
    loader::load_directory(dir, &Progress::default()).expect("the folder loads")
}

#[test]
fn a_treatment_day_on_disk_is_one_session_and_loses_nothing() {
    let (dir, daily) = treatment_day();
    let study = load(&dir);
    assert_eq!(study.registrations.len(), 2);
    assert_eq!(study.treat_records.len(), 2);

    let by_label = |label: &str| {
        study
            .registrations
            .iter()
            .position(|r| r.label == label)
            .unwrap_or_else(|| panic!("a registration labelled {label:?}"))
    };
    let imaging = by_label("imaging position coordinate system");
    let treatment = by_label("treatment position coordinate system");
    assert_eq!(
        sessions::reg_kind(&study.registrations[imaging], &study),
        RegKind::Imaging
    );
    assert_eq!(
        sessions::reg_kind(&study.registrations[treatment], &study),
        RegKind::Treatment,
        "the plan referenced with the RT Plan class is still the plan"
    );
    assert_eq!(study.registrations[imaging].references, vec![daily.clone()]);

    let all = sessions::sessions(&study);
    assert_eq!(all.len(), 1);
    let s = &all[0];
    assert_eq!((s.date.as_str(), s.number, s.of), (DAY, 1, 1));
    let daily_idx = study.series.iter().position(|se| se.uid == daily).unwrap();
    assert_eq!(s.series, vec![daily_idx]);
    assert_eq!(s.image_regs, vec![imaging]);
    assert!(s.ct_regs.is_empty());
    assert_eq!(s.steps.len(), 1);
    let step = &s.steps[0];
    assert_eq!(step.treatment_reg, Some(treatment));
    assert_eq!(step.setup.len(), 1);
    assert_eq!(step.setup[0].images.len(), 2);
    assert_eq!(
        step.records.len(),
        2,
        "both records of the interrupted beam"
    );
    assert!(!step.all_normal);
    assert_eq!(step.verification.len(), 1);
    assert_eq!(step.verification[0].images.len(), 2);
    let setup_numbers: Vec<Option<i64>> = step.setup[0]
        .images
        .iter()
        .map(|&i| study.planar_images[i].series_number)
        .collect();
    assert_eq!(setup_numbers, vec![Some(212), Some(212)]);
    assert!(step.name.contains("G000"), "{}", step.name);
    // The first matrix that is not the identity: the phantom's shift.
    let k = sessions::correction(&study.registrations[treatment]).unwrap();
    let [tx, ty, tz] = GenParams::default().reg_shift;
    assert_eq!([k[0], k[1], k[2]], [tx, ty, tz]);
    assert!(k[3..].iter().all(|a| a.abs() < 1e-9));

    // Nothing lost and nothing twice: what the session took and what it
    // left are every object of the folder, each once.
    let mut regs: Vec<usize> = s.regs().collect();
    let mut recs: Vec<usize> = s.records().collect();
    let mut imgs: Vec<usize> = s.images().collect();
    regs.sort();
    recs.sort();
    imgs.sort();
    assert_eq!(regs, vec![0, 1]);
    assert_eq!(recs, vec![0, 1]);
    assert_eq!(imgs.len(), 4);
    let left: Vec<usize> = (0..study.planar_images.len())
        .filter(|i| !imgs.contains(i))
        .collect();
    assert_eq!(left.len(), 1, "the phantom's DX radiograph stays outside");
    assert_eq!(study.planar_images[left[0]].modality, "DX");
}
