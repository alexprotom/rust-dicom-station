//! What a load says about what it found, and only what is true.
//!
//! The failures these tests exist for, each seen on a real archive: an
//! encapsulated PDF beside a CT was decoded as an image and reported as a
//! failed load; every structure set of a 4DCT (one frame of reference per
//! phase) was reported as misaligned because it was compared with the
//! volume on display instead of the series it was drawn on; and a planning
//! system's structure set without a Structure Set ROI Sequence, its frame
//! of reference in the wrong tag, opened with every ROI called "ROI n".
//!
//! Every case is built from the synthetic RT study: its files are copied
//! and re-identified, never anything of a real patient.

use std::path::{Path, PathBuf};

use dicom_core::value::{PrimitiveValue, Value};
use dicom_core::{DataElement, Length, Tag, VR};
use dicom_dictionary_std::tags;
use dicom_object::meta::FileMetaTableBuilder;
use dicom_object::InMemDicomObject;
use rust_dicom_station::dicom_export::{self, new_uid, ExportParams};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader::{self, LoadedStudy};
use rust_dicom_station::progress::Progress;

fn target(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the test folder can be made");
    dir
}

/// The synthetic RT study, generated into its own folder.
fn phantom(tag: &str) -> PathBuf {
    let dir = target(tag);
    gen_test_data::generate(&dir, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    dir
}

/// Every file of `dir` whose name starts with `prefix`, sorted.
fn files(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("the folder is readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
        })
        .collect();
    out.sort();
    out
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

fn first_item(o: &InMemDicomObject, tag: Tag) -> InMemDicomObject {
    loader::items_of(o, tag)
        .and_then(|items| items.first())
        .cloned()
        .expect("the sequence has an item")
}

fn str_of(o: &InMemDicomObject, tag: Tag) -> String {
    loader::str_of(o, tag).unwrap_or_default()
}

/// Copy the phantom's CT series into `dst` as a new series of its own:
/// new Series Instance UID, new SOP Instance UIDs, frame of reference
/// `for_uid`. Returns the series UID.
fn copy_ct(src: &Path, dst: &Path, prefix: &str, for_uid: &str, number: i64) -> String {
    let series_uid = new_uid();
    for f in files(src, "CT_") {
        let mut obj = dicom_object::open_file(&f).expect("a CT slice opens");
        put_str(&mut obj, tags::SERIES_INSTANCE_UID, VR::UI, &series_uid);
        put_str(&mut obj, tags::SOP_INSTANCE_UID, VR::UI, &new_uid());
        put_str(&mut obj, tags::FRAME_OF_REFERENCE_UID, VR::UI, for_uid);
        put_str(&mut obj, tags::SERIES_NUMBER, VR::IS, &number.to_string());
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        obj.write_to_file(dst.join(format!("{prefix}{name}")))
            .expect("the copy is written");
    }
    series_uid
}

/// The phantom's structure set, made to say it was drawn on `series_uid`
/// in `for_uid`, written to `dst/name`.
fn retarget_rtstruct(src: &Path, dst: &Path, name: &str, series_uid: &str, for_uid: &str) {
    let mut obj = dicom_object::open_file(src.join("RS_synth.dcm")).expect("the RTSTRUCT opens");
    put_str(&mut obj, tags::SOP_INSTANCE_UID, VR::UI, &new_uid());
    put_str(&mut obj, tags::SERIES_INSTANCE_UID, VR::UI, &new_uid());
    let mut rfr = first_item(&obj, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE);
    put_str(&mut rfr, tags::FRAME_OF_REFERENCE_UID, VR::UI, for_uid);
    let mut study = first_item(&rfr, tags::RT_REFERENCED_STUDY_SEQUENCE);
    let mut series = first_item(&study, tags::RT_REFERENCED_SERIES_SEQUENCE);
    put_str(&mut series, tags::SERIES_INSTANCE_UID, VR::UI, series_uid);
    put_seq(
        &mut study,
        tags::RT_REFERENCED_SERIES_SEQUENCE,
        vec![series],
    );
    put_seq(&mut rfr, tags::RT_REFERENCED_STUDY_SEQUENCE, vec![study]);
    put_seq(
        &mut obj,
        tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
        vec![rfr],
    );
    obj.write_to_file(dst.join(name))
        .expect("the set is written");
}

/// The phantom's dose grid in another frame of reference.
fn retarget_dose(src: &Path, dst: &Path, name: &str, for_uid: &str) {
    let f = files(src, "RD_").pop().expect("the phantom has a dose");
    let mut obj = dicom_object::open_file(&f).expect("the RTDOSE opens");
    put_str(&mut obj, tags::SOP_INSTANCE_UID, VR::UI, &new_uid());
    put_str(&mut obj, tags::FRAME_OF_REFERENCE_UID, VR::UI, for_uid);
    obj.write_to_file(dst.join(name))
        .expect("the dose is written");
}

/// The frame of reference of the phantom's CT.
fn phantom_for(dir: &Path) -> String {
    let f = files(dir, "CT_").remove(0);
    str_of(
        &dicom_object::open_file(&f).expect("a CT slice opens"),
        tags::FRAME_OF_REFERENCE_UID,
    )
}

/// The frame-of-reference warnings: every one ends in what may be wrong.
fn for_warnings(study: &LoadedStudy) -> Vec<&String> {
    study
        .warnings
        .iter()
        .filter(|w| w.contains("frame of reference") && w.contains("may be misaligned"))
        .collect()
}

fn load(dir: &Path) -> LoadedStudy {
    loader::load_directory(dir, &Progress::default()).expect("the folder loads")
}

/// An encapsulated document beside the study's images, alone in its series:
/// Modality OT, no position, no pixels - exactly what a hospital's
/// "Befund" series holds. Written to `dir/name`.
fn write_document(
    dir: &Path,
    name: &str,
    study_uid: &str,
    sop_class: &str,
    mime: &str,
    body: Vec<u8>,
) {
    let mut o = InMemDicomObject::new_empty();
    put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, sop_class);
    put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, &new_uid());
    put_str(&mut o, tags::STUDY_INSTANCE_UID, VR::UI, study_uid);
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, &new_uid());
    put_str(&mut o, tags::MODALITY, VR::CS, "OT");
    put_str(&mut o, tags::SERIES_DESCRIPTION, VR::LO, "Befund");
    put_str(&mut o, tags::DOCUMENT_TITLE, VR::ST, "Findings");
    put_str(
        &mut o,
        tags::MIME_TYPE_OF_ENCAPSULATED_DOCUMENT,
        VR::LO,
        mime,
    );
    let len = body.len();
    let mut padded = body;
    if padded.len() % 2 == 1 {
        padded.push(0);
    }
    o.put(DataElement::new(
        tags::ENCAPSULATED_DOCUMENT_LENGTH,
        VR::UL,
        PrimitiveValue::from(len as u32),
    ));
    o.put(DataElement::new(
        tags::ENCAPSULATED_DOCUMENT,
        VR::OB,
        PrimitiveValue::from(padded),
    ));
    o.with_meta(
        FileMetaTableBuilder::new()
            .transfer_syntax("1.2.840.10008.1.2.1")
            .media_storage_sop_class_uid(sop_class),
    )
    .expect("file meta")
    .write_to_file(dir.join(name))
    .expect("the document is written");
}

#[test]
fn an_encapsulated_pdf_is_a_document_and_other_documents_are_left_out() {
    let dir = phantom("test_warn_pdf");
    let before = load(&dir);
    let study_uid = before.series[0].study_uid.clone();
    write_document(
        &dir,
        "OT_report.dcm",
        &study_uid,
        rust_dicom_station::sop::SOP_ENCAPSULATED_PDF,
        "application/pdf",
        rust_dicom_station::report::document::sample_pdf(),
    );
    write_document(
        &dir,
        "OT_cda.dcm",
        &study_uid,
        "1.2.840.10008.5.1.4.1.1.104.2",
        "text/XML",
        b"<ClinicalDocument/>".to_vec(),
    );

    let study = load(&dir);
    assert!(
        !study.warnings.iter().any(|w| w.contains("load failed")),
        "no failed load: {:?}",
        study.warnings
    );
    assert_eq!(
        study.planar_images.len(),
        before.planar_images.len(),
        "the PDF is no planar image"
    );

    // The PDF is a report of the study, named after its series, whose
    // pages draw; its title is kept out of the facts.
    assert_eq!(study.reports.len(), before.reports.len() + 1);
    let r = study.reports.last().unwrap();
    assert_eq!(r.kind, rust_dicom_station::report::ReportKind::Document);
    assert_eq!(
        (r.label.as_str(), r.study_uid.as_str()),
        ("Befund", study_uid.as_str())
    );
    assert!(!r.info.iter().any(|(_, v)| v.contains("Findings")));
    let doc = r.document.as_ref().expect("the report keeps its document");
    assert_eq!(doc.title, "Findings");
    let drawn = doc.pages_blocking().as_ref().expect("the pages draw");
    assert_eq!((drawn.total, drawn.pages.len()), (1, 1));

    // The CDA is not shown, and says so in one line.
    let skipped: Vec<&String> = study
        .warnings
        .iter()
        .filter(|w| w.contains("not displayed"))
        .collect();
    assert_eq!(skipped.len(), 1, "one summary line: {:?}", study.warnings);
    assert!(skipped[0].contains("Encapsulated CDA"), "{}", skipped[0]);
}

#[test]
fn structure_sets_and_doses_are_held_to_their_own_series() {
    let src = phantom("test_warn_for_src");
    let dir = target("test_warn_for");
    let for1 = phantom_for(&src);
    for f in std::fs::read_dir(&src).unwrap().flatten() {
        std::fs::copy(f.path(), dir.join(f.file_name())).unwrap();
    }
    // A second CT of other frame of reference, with a set drawn on it and a
    // dose computed in it.
    let for2 = new_uid();
    let series2 = copy_ct(&src, &dir, "B_", &for2, 2);
    retarget_rtstruct(&src, &dir, "RS_on_B.dcm", &series2, &for2);
    retarget_dose(&src, &dir, "RD_on_B.dcm", &for2);

    let study = load(&dir);
    assert_eq!(study.series.len(), 2);
    assert_eq!(study.structure_sets.len(), 2);
    assert_eq!(study.doses.len(), 2);
    assert!(
        for_warnings(&study).is_empty(),
        "every object agrees with its own series: {:?}",
        study.warnings
    );

    // A set that says it was drawn on the second series but names the
    // first one's frame of reference contradicts itself: one warning,
    // naming it.
    retarget_rtstruct(&src, &dir, "RS_wrong.dcm", &series2, &for1);
    let study = load(&dir);
    let w = for_warnings(&study);
    assert_eq!(w.len(), 1, "exactly one: {:?}", study.warnings);
    assert!(w[0].contains("RS_wrong.dcm"), "{}", w[0]);
    assert!(!w[0].contains("RS_on_B.dcm"), "{}", w[0]);

    // Merged into a workspace, a set whose series is not in the load is
    // not reported: the workspace may hold it.
    let lone = target("test_warn_for_lone");
    retarget_rtstruct(&src, &lone, "RS_alone.dcm", &new_uid(), &new_uid());
    let merged = loader::load_directory_to_merge(&lone, &Progress::default()).unwrap();
    assert!(for_warnings(&merged).is_empty(), "{:?}", merged.warnings);
}

#[test]
fn a_4d_group_with_a_frame_of_reference_per_phase_raises_nothing() {
    let src = phantom("test_warn_4d_src");
    let dir = target("test_warn_4d");
    for phase in 0..10 {
        let for_uid = new_uid();
        let series = copy_ct(&src, &dir, &format!("P{phase}_"), &for_uid, phase + 1);
        retarget_rtstruct(&src, &dir, &format!("RS_P{phase}.dcm"), &series, &for_uid);
    }
    let study = load(&dir);
    assert_eq!(study.series.len(), 10);
    assert_eq!(study.structure_sets.len(), 10);
    assert!(for_warnings(&study).is_empty(), "{:?}", study.warnings);
    for ss in &study.structure_sets {
        let se = study
            .series
            .iter()
            .find(|se| se.uid == ss.referenced_series_uid)
            .expect("every set names a loaded phase");
        assert_eq!(se.frame_of_reference_uid, ss.frame_of_reference_uid);
    }
}

#[test]
fn a_set_without_its_roi_sequence_keeps_its_names_and_frame_of_reference() {
    let dir = phantom("test_warn_raystation");
    let for_uid = phantom_for(&dir);
    let original = load(&dir);
    let mut names: Vec<String> = original.structure_sets[0]
        .rois
        .iter()
        .map(|r| r.name.clone())
        .collect();
    names.sort();

    // What some planning system exports write: no Structure Set ROI
    // Sequence, the names only as observation labels, and the frame of
    // reference in Referenced SOP Instance UID instead of its own tag,
    // with the (non-standard) top-level copy right.
    let path = dir.join("RS_synth.dcm");
    let mut obj = dicom_object::open_file(&path).unwrap();
    let by_number: Vec<(String, String)> = loader::items_of(&obj, tags::STRUCTURE_SET_ROI_SEQUENCE)
        .unwrap()
        .iter()
        .map(|it| (str_of(it, tags::ROI_NUMBER), str_of(it, tags::ROI_NAME)))
        .collect();
    obj.remove_element(tags::STRUCTURE_SET_ROI_SEQUENCE);
    let observations: Vec<InMemDicomObject> =
        loader::items_of(&obj, tags::RTROI_OBSERVATIONS_SEQUENCE)
            .unwrap()
            .iter()
            .map(|it| {
                let mut it = it.clone();
                let number = str_of(&it, tags::REFERENCED_ROI_NUMBER);
                let name = &by_number.iter().find(|(n, _)| *n == number).unwrap().1;
                put_str(&mut it, Tag(0x3006, 0x0085), VR::SH, name);
                it
            })
            .collect();
    put_seq(&mut obj, tags::RTROI_OBSERVATIONS_SEQUENCE, observations);
    let mut rfr = first_item(&obj, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE);
    rfr.remove_element(tags::FRAME_OF_REFERENCE_UID);
    put_str(
        &mut rfr,
        tags::REFERENCED_SOP_INSTANCE_UID,
        VR::UI,
        &for_uid,
    );
    put_seq(
        &mut obj,
        tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
        vec![rfr],
    );
    put_str(&mut obj, tags::FRAME_OF_REFERENCE_UID, VR::UI, &for_uid);
    obj.write_to_file(&path).unwrap();

    let study = load(&dir);
    let ss = &study.structure_sets[0];
    let mut got: Vec<String> = ss.rois.iter().map(|r| r.name.clone()).collect();
    got.sort();
    assert_eq!(got, names, "the real names, not ROI n");
    assert_eq!(ss.frame_of_reference_uid, for_uid);
    let notes: Vec<&String> = study
        .warnings
        .iter()
        .filter(|w| w.starts_with("RTSTRUCT RS_synth.dcm"))
        .collect();
    assert_eq!(
        notes.len(),
        1,
        "one line for the file: {:?}",
        study.warnings
    );
    assert!(
        notes[0].contains("no Structure Set ROI Sequence"),
        "{}",
        notes[0]
    );
    assert!(for_warnings(&study).is_empty(), "{:?}", study.warnings);

    // Written back, the set is a conforming one with the same names in the
    // same frame of reference.
    let out = target("test_warn_raystation_out");
    let n = dicom_export::export_derived(
        &study,
        &out,
        &ExportParams::for_study(&study),
        &Progress::default(),
    )
    .expect("the set exports");
    assert!(n >= 1);
    let written = files(&out, "RS_").remove(0);
    let back = dicom_object::open_file(&written).unwrap();
    assert!(loader::items_of(&back, tags::STRUCTURE_SET_ROI_SEQUENCE).is_some());
    let again = loader::load_directory(&out, &Progress::default()).unwrap();
    let ss = &again.structure_sets[0];
    let mut back_names: Vec<String> = ss.rois.iter().map(|r| r.name.clone()).collect();
    back_names.sort();
    assert_eq!(back_names, names);
    assert_eq!(ss.frame_of_reference_uid, for_uid);
    assert!(
        !again.warnings.iter().any(|w| w.starts_with("RTSTRUCT")),
        "nothing left to patch: {:?}",
        again.warnings
    );
}
