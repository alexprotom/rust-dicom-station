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

/// Write a scanner's protocol page into the folder, as syngo writes one: a
/// Secondary Capture that calls itself `CT`, made from the first slice of
/// the folder's CT with its position taken away and its pixels blanked, in
/// a series of its own named `description`; the page's text as an overlay
/// plane (a few bits drawn) and, as text, in the private element *SIEMENS
/// MEDCOM OOG*. The loader must file it as a report with the page beside
/// it, never as a slice or a radiograph.
pub fn protocol_page_into(folder: &Path, description: &str) {
    use dicom_core::{DataElement, PrimitiveValue};
    use dicom_object::FileMetaTableBuilder;
    use rust_dicom_station::dicom_export::new_uid;
    let study = loader::load_directory(folder, &Progress::default()).expect("the CT loads");
    let slice = study.series[study.active_series].files[0].clone();
    let obj = dicom_object::open_file(&slice).expect("the slice opens");
    let mut o = obj.into_inner();
    let put = |o: &mut dicom_object::InMemDicomObject, tag: Tag, vr: VR, v: &str| {
        o.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
    };
    let sop_uid = new_uid();
    put(
        &mut o,
        tags::SOP_CLASS_UID,
        VR::UI,
        "1.2.840.10008.5.1.4.1.1.7",
    );
    put(&mut o, tags::SOP_INSTANCE_UID, VR::UI, &sop_uid);
    put(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, &new_uid());
    put(&mut o, tags::SERIES_NUMBER, VR::IS, "999");
    put(&mut o, tags::SERIES_DESCRIPTION, VR::LO, description);
    put(&mut o, tags::MODALITY, VR::CS, "CT");
    put(&mut o, tags::CONTENT_DATE, VR::DA, "20250307");
    o.put(DataElement::new(
        tags::IMAGE_TYPE,
        VR::CS,
        PrimitiveValue::Strs(
            ["DERIVED", "SECONDARY", "OTHER", "CT_SOM5 PROT"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        ),
    ));
    for tag in [
        tags::IMAGE_POSITION_PATIENT,
        tags::IMAGE_ORIENTATION_PATIENT,
        tags::FRAME_OF_REFERENCE_UID,
        tags::RESCALE_INTERCEPT,
        tags::RESCALE_SLOPE,
    ] {
        o.remove_element(tag);
    }
    // Blank pixels, a window that shows nothing of them, as the scanner
    // writes.
    let rows = o
        .element(tags::ROWS)
        .ok()
        .and_then(|e| e.to_int::<usize>().ok())
        .unwrap_or(64);
    let cols = o
        .element(tags::COLUMNS)
        .ok()
        .and_then(|e| e.to_int::<usize>().ok())
        .unwrap_or(64);
    o.put(DataElement::new(
        tags::PIXEL_DATA,
        VR::OW,
        PrimitiveValue::U16(vec![0u16; rows * cols].into()),
    ));
    o.put(DataElement::new(
        tags::BITS_ALLOCATED,
        VR::US,
        PrimitiveValue::from(16u16),
    ));
    o.put(DataElement::new(
        tags::BITS_STORED,
        VR::US,
        PrimitiveValue::from(12u16),
    ));
    o.put(DataElement::new(
        tags::HIGH_BIT,
        VR::US,
        PrimitiveValue::from(11u16),
    ));
    o.put(DataElement::new(
        tags::PIXEL_REPRESENTATION,
        VR::US,
        PrimitiveValue::from(0u16),
    ));
    put(&mut o, tags::WINDOW_CENTER, VR::DS, "200");
    put(&mut o, tags::WINDOW_WIDTH, VR::DS, "50");
    // The overlay: the first row's first eight pixels set.
    let mut bits = vec![0u8; rows * cols / 8 + 1];
    bits[0] = 0xFF;
    for (e, vr, v) in [
        (0x0010u16, VR::US, PrimitiveValue::from(rows as u16)),
        (0x0011, VR::US, PrimitiveValue::from(cols as u16)),
        (0x0040, VR::CS, PrimitiveValue::from("G")),
        (0x0050, VR::SS, PrimitiveValue::I16([1i16, 1].into())),
        (0x0100, VR::US, PrimitiveValue::from(1u16)),
        (0x0102, VR::US, PrimitiveValue::from(0u16)),
        (0x3000, VR::OW, PrimitiveValue::U8(bits.into())),
    ] {
        o.put(DataElement::new(Tag(0x6000, e), vr, v));
    }
    // The text, as the scanner's STEP file.
    put(&mut o, Tag(0x0029, 0x0011), VR::LO, "SIEMENS MEDCOM OOG");
    o.put(DataElement::new(
        Tag(0x0029, 0x1110),
        VR::OB,
        PrimitiveValue::U8(step_page().into_bytes().into()),
    ));
    o.with_meta(
        FileMetaTableBuilder::new()
            .transfer_syntax("1.2.840.10008.1.2.1")
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
            .media_storage_sop_instance_uid(sop_uid),
    )
    .expect("file meta")
    .write_to_file(folder.join("protocol_page.dcm"))
    .expect("page written");
}

/// A protocol page's STEP text: a date line, a two-line header, two rows.
pub fn step_page() -> String {
    fn text(id: u64, pos: u64, line: i64, s: &str) -> String {
        let codes: Vec<String> = s.bytes().map(|b| b.to_string()).collect();
        format!(
            "@{id} = CsaGraphicText(0, 0, 0, 3, 3.0, 0, #{pos}, 0, 1, 0, 0, #2252, $, $, $, $, 0, \
             0.0, 0.0, 0.0, 0, 0, 0, 0, 65535, 1.0, 1.0, 1.0, 0.0, 0, 0.299, 0.587, 0.114, 0, \
             65535, 0.0, 0.0, 0.0, 0.0, 0, 0.299, 0.587, 0.114, ({}, 0), 1, -1, {line}, 100, \
             -16, 0, 0, 0, 400, 0, 0, 0, 0, 3, 2, 1, 34, 77, 77, 0, 0, 1);",
            codes.join(", ")
        )
    }
    let mut s = String::from("STEP;\nHEADER;\nENDSEC;\nDATA;\n");
    let mut id = 10;
    let mut put = |x: f64, line: i64, t: &str| {
        s.push_str(&format!(
            "@{id} = CsaGraDoubleVec3DArray(0, ({x:.1}, 0.0, 0.0));\n"
        ));
        s.push_str(&text(id + 1, id, line, t));
        s.push('\n');
        id += 2;
    };
    put(0.0, 5, "  07-Mar-2025  08:31");
    put(0.0, 11, "  Gesamt mAs 2817      Gesamt DLP 511 mGy*cm");
    for (x, t) in [
        (167.0, "Scan"),
        (197.0, "kV"),
        (232.0, "mAs"),
        (240.0, "/"),
        (245.0, "ref."),
        (370.0, "CTDIvol"),
        (425.0, "DLP"),
    ] {
        put(x, 13, t);
    }
    for (x, t) in [(370.0, "mGy"), (425.0, "mGy*cm")] {
        put(x, 14, t);
    }
    put(0.0, 17, "  Topogram");
    for (x, t) in [
        (167.0, "1"),
        (197.0, "110"),
        (370.0, "0.06(L)"),
        (425.0, "2.89"),
    ] {
        put(x, 17, t);
    }
    put(0.0, 18, "  AbdRoutine");
    for (x, t) in [
        (167.0, "2"),
        (197.0, "130"),
        (232.0, "73"),
        (240.0, "/"),
        (270.0, "110"),
        (370.0, "8.18(L)"),
        (425.0, "389.33"),
    ] {
        put(x, 18, t);
    }
    s.push_str("ENDSEC;\nENDSTEP;\n");
    s
}

/// Write a CT dose report (an X-Ray Radiation Dose SR, TID 10011) into the
/// folder, for the study the folder holds: two acquisitions, the totals,
/// the scanner, and a physician's name as a `PNAME` item, which the MCP
/// server must never return.
pub fn dose_report_into(folder: &Path, physician: &str) {
    use dicom_core::value::Value;
    use dicom_core::{DataElement, Length, PrimitiveValue};
    use dicom_object::{FileMetaTableBuilder, InMemDicomObject};
    use rust_dicom_station::dicom_export::new_uid;
    const SR: &str = "1.2.840.10008.5.1.4.1.1.88.67";
    let study = loader::load_directory(folder, &Progress::default()).expect("the CT loads");
    let series = &study.series[study.active_series];
    fn put(o: &mut InMemDicomObject, tag: Tag, vr: VR, v: &str) {
        o.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
    }
    fn seq(o: &mut InMemDicomObject, tag: Tag, items: Vec<InMemDicomObject>) {
        o.put(DataElement::new(
            tag,
            VR::SQ,
            Value::new_sequence(items, Length::UNDEFINED),
        ));
    }
    fn code(value: &str, scheme: &str, meaning: &str) -> InMemDicomObject {
        let mut c = InMemDicomObject::new_empty();
        put(&mut c, tags::CODE_VALUE, VR::SH, value);
        put(&mut c, tags::CODING_SCHEME_DESIGNATOR, VR::SH, scheme);
        put(&mut c, tags::CODE_MEANING, VR::LO, meaning);
        c
    }
    fn item(rel: &str, vt: &str, concept: (&str, &str)) -> InMemDicomObject {
        let mut i = InMemDicomObject::new_empty();
        put(&mut i, tags::RELATIONSHIP_TYPE, VR::CS, rel);
        put(&mut i, tags::VALUE_TYPE, VR::CS, vt);
        seq(
            &mut i,
            tags::CONCEPT_NAME_CODE_SEQUENCE,
            vec![code(concept.0, "DCM", concept.1)],
        );
        i
    }
    fn num(concept: (&str, &str), v: &str, unit: &str) -> InMemDicomObject {
        let mut i = item("CONTAINS", "NUM", concept);
        let mut m = InMemDicomObject::new_empty();
        put(&mut m, tags::NUMERIC_VALUE, VR::DS, v);
        seq(
            &mut m,
            tags::MEASUREMENT_UNITS_CODE_SEQUENCE,
            vec![code(unit, "UCUM", unit)],
        );
        seq(&mut i, tags::MEASURED_VALUE_SEQUENCE, vec![m]);
        i
    }
    fn text(rel: &str, concept: (&str, &str), v: &str) -> InMemDicomObject {
        let mut i = item(rel, "TEXT", concept);
        put(&mut i, tags::TEXT_VALUE, VR::UT, v);
        i
    }
    fn coded(concept: (&str, &str), answer: (&str, &str, &str)) -> InMemDicomObject {
        let mut i = item("CONTAINS", "CODE", concept);
        seq(
            &mut i,
            tags::CONCEPT_CODE_SEQUENCE,
            vec![code(answer.0, answer.1, answer.2)],
        );
        i
    }
    fn container(concept: (&str, &str), children: Vec<InMemDicomObject>) -> InMemDicomObject {
        let mut i = item("CONTAINS", "CONTAINER", concept);
        put(&mut i, tags::CONTINUITY_OF_CONTENT, VR::CS, "SEPARATE");
        seq(&mut i, tags::CONTENT_SEQUENCE, children);
        i
    }
    let acquisition = |protocol: &str, kv: &str, ma: &str, ctdi: &str, dlp: &str| {
        container(
            ("113819", "CT Acquisition"),
            vec![
                text("CONTAINS", ("125203", "Acquisition Protocol"), protocol),
                coded(("123014", "Target Region"), ("T-D4000", "SRT", "Abdomen")),
                coded(
                    ("113820", "CT Acquisition Type"),
                    ("P5-08001", "SRT", "Spiral Acquisition"),
                ),
                container(
                    ("113822", "CT Acquisition Parameters"),
                    vec![
                        num(("113824", "Exposure Time"), "10.64", "s"),
                        num(("113825", "Scanning Length"), "425.5", "mm"),
                        num(("113827", "Nominal Total Collimation Width"), "19.2", "mm"),
                        num(("113828", "Pitch Factor"), "1.25", "{ratio}"),
                        container(
                            ("113831", "CT X-Ray Source Parameters"),
                            vec![
                                num(("113733", "KVP"), kv, "kV"),
                                num(("113833", "Maximum X-Ray Tube Current"), "181", "mA"),
                                num(("113734", "Mean X-Ray Tube Current"), ma, "mA"),
                                num(("113834", "Exposure Time per Rotation"), "0.6", "s"),
                            ],
                        ),
                    ],
                ),
                container(
                    ("113829", "CT Dose"),
                    vec![
                        num(("113830", "Mean CTDIvol"), ctdi, "mGy"),
                        coded(
                            ("113835", "CTDIw Phantom Type"),
                            ("113691", "DCM", "IEC Body Dosimetry Phantom"),
                        ),
                        num(("113838", "DLP"), dlp, "mGy.cm"),
                    ],
                ),
                text("CONTAINS", ("113842", "X-Ray Modulation Type"), "XYZ_EC"),
            ],
        )
    };
    let mut physician_item = item(
        "HAS OBS CONTEXT",
        "PNAME",
        ("121008", "Person Observer Name"),
    );
    put(&mut physician_item, tags::PERSON_NAME, VR::PN, physician);
    let mut o = InMemDicomObject::new_empty();
    let sop_uid = new_uid();
    put(&mut o, tags::SOP_CLASS_UID, VR::UI, SR);
    put(&mut o, tags::SOP_INSTANCE_UID, VR::UI, &sop_uid);
    put(&mut o, tags::STUDY_INSTANCE_UID, VR::UI, &series.study_uid);
    put(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, &new_uid());
    put(&mut o, tags::SERIES_NUMBER, VR::IS, "502");
    put(&mut o, tags::INSTANCE_NUMBER, VR::IS, "1");
    put(&mut o, tags::MODALITY, VR::CS, "SR");
    put(&mut o, tags::SERIES_DESCRIPTION, VR::LO, "Dose Report");
    put(&mut o, tags::CONTENT_DATE, VR::DA, "20250307");
    put(&mut o, tags::CONTENT_TIME, VR::TM, "084614");
    put(&mut o, tags::PATIENT_NAME, VR::PN, &series.patient_name);
    put(&mut o, tags::PATIENT_ID, VR::LO, &series.patient_id);
    put(&mut o, tags::VALUE_TYPE, VR::CS, "CONTAINER");
    put(&mut o, tags::CONTINUITY_OF_CONTENT, VR::CS, "SEPARATE");
    put(&mut o, tags::COMPLETION_FLAG, VR::CS, "COMPLETE");
    put(&mut o, tags::VERIFICATION_FLAG, VR::CS, "UNVERIFIED");
    seq(
        &mut o,
        tags::CONCEPT_NAME_CODE_SEQUENCE,
        vec![code("113701", "DCM", "X-Ray Radiation Dose Report")],
    );
    seq(
        &mut o,
        tags::CONTENT_SEQUENCE,
        vec![
            coded(
                ("121058", "Procedure reported"),
                ("P5-08000", "SRT", "Computed Tomography X-ray"),
            ),
            text(
                "HAS OBS CONTEXT",
                ("121014", "Device Observer Manufacturer"),
                "SIEMENS",
            ),
            text(
                "HAS OBS CONTEXT",
                ("121015", "Device Observer Model Name"),
                "SOMATOM Emotion 16",
            ),
            physician_item,
            container(
                ("113811", "CT Accumulated Dose Data"),
                vec![
                    num(
                        ("113812", "Total Number of Irradiation Events"),
                        "2",
                        "{events}",
                    ),
                    num(
                        ("113813", "CT Dose Length Product Total"),
                        "508.44",
                        "mGy.cm",
                    ),
                ],
            ),
            acquisition("Steinsuche", "110", "81", "2.8", "119.11"),
            acquisition("AbdRoutine", "130", "128", "8.18", "389.33"),
        ],
    );
    o.with_meta(
        FileMetaTableBuilder::new()
            .transfer_syntax("1.2.840.10008.1.2.1")
            .media_storage_sop_class_uid(SR)
            .media_storage_sop_instance_uid(sop_uid),
    )
    .expect("file meta")
    .write_to_file(folder.join("dose_report.dcm"))
    .expect("report written");
}

/// Write two RT Images of the folder's CT into the folder - a kV AP and a
/// kV lateral, 128 pixels, about the CT's centre - of the patient found
/// displaced by `shift` (mm, patient LPS), each with the RT Image geometry
/// that places it as planned and the label `{prefix} AP` / `{prefix} LAT`.
/// A 2D/3D registration of the two should answer a correction of
/// `-shift`.
pub fn radiographs_into(folder: &Path, shift: [f64; 3], prefix: &str) {
    use rust_dicom_station::dicom_export::{write_rt_image, ObjectMeta};
    use rust_dicom_station::drr::{self, DrrParams, Engine, Geometry};
    use rust_dicom_station::geometry::Vec3;
    let study = loader::load_directory(folder, &Progress::default()).expect("the CT loads");
    let vol = &study.volume;
    let iso = Geometry::centered_on(vol).isocenter;
    let d = Vec3::new(shift[0], shift[1], shift[2]);
    for (name, gantry) in [("AP", 0.0), ("LAT", 90.0)] {
        let geometry = Geometry {
            gantry_deg: gantry,
            isocenter: iso,
            dims: [128, 128],
            panel_mm: [300.0, 300.0],
            ..Geometry::default()
        };
        let mut params = DrrParams::for_volume(vol);
        params.geometry = geometry;
        params.engine = Engine::RayCast;
        params.step_mm = 1.0;
        // The anatomy at room point x is the CT's at x - shift.
        params.camera = Some(geometry.camera().mapped(|p| p - d));
        let img = drr::render(vol, &params, &Progress::default()).expect("rendered");
        let mut planar = img.to_planar(
            &DrrParams {
                camera: None,
                ..params
            },
            false,
        );
        planar.label = format!("{prefix} {name}");
        planar.frame_of_reference_uid = vol.frame_of_reference_uid.clone();
        let meta = ObjectMeta::of_study(&study, "KV", "kV image");
        write_rt_image(&folder.join(format!("rtimage_{name}.dcm")), &planar, &meta)
            .expect("RT image written");
    }
}
