//! A made-up 4DCT for the 4D tests (`fourd_scan`, `curation`): files
//! carrying only what the catalog and the 4D rules read, written with the
//! patient `P4D-01` (no real data).

#![allow(dead_code)]

/// One file of a made-up series: `(tag, VR, value)` beside what every file
/// carries.
pub fn file(
    path: &std::path::Path,
    class: &str,
    sop: &str,
    extra: &[(dicom_core::Tag, dicom_core::VR, String)],
) {
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_dictionary_std::tags;
    use dicom_object::{FileMetaTableBuilder, InMemDicomObject};
    let mut obj = InMemDicomObject::new_empty();
    for (tag, vr, v) in [
        (tags::SOP_CLASS_UID, VR::UI, class),
        (tags::SOP_INSTANCE_UID, VR::UI, sop),
        (tags::PATIENT_ID, VR::LO, "P4D-01"),
        (tags::PATIENT_NAME, VR::PN, "Phantom^Four"),
    ] {
        obj.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
    }
    for (tag, vr, v) in extra {
        let value = if *vr == VR::US {
            PrimitiveValue::from(v.parse::<u16>().unwrap())
        } else if *vr == VR::OW {
            // Pixels: as many zero samples as the value says.
            PrimitiveValue::U16(vec![0u16; v.parse::<usize>().unwrap()].into())
        } else {
            PrimitiveValue::from(v.as_str())
        };
        obj.put(DataElement::new(*tag, *vr, value));
    }
    let f = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .transfer_syntax("1.2.840.10008.1.2.1")
                .media_storage_sop_class_uid(class)
                .media_storage_sop_instance_uid(sop),
        )
        .expect("meta");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    f.write_to_file(path).expect("write");
}

/// A CT series of `slices` files: `(study, series, number, description,
/// acquisition date, time, image type, protocol)`.
#[allow(clippy::too_many_arguments)]
pub fn ct_series(
    root: &std::path::Path,
    study: &str,
    series: &str,
    number: i64,
    desc: &str,
    acq: (&str, &str),
    image_type: &str,
    protocol: &str,
    slices: usize,
) {
    use dicom_core::VR;
    use dicom_dictionary_std::tags;
    for k in 0..slices {
        let mut extra = vec![
            (tags::STUDY_INSTANCE_UID, VR::UI, study.to_string()),
            (tags::SERIES_INSTANCE_UID, VR::UI, series.to_string()),
            (tags::MODALITY, VR::CS, "CT".to_string()),
            (tags::SERIES_NUMBER, VR::IS, number.to_string()),
            (tags::IMAGE_TYPE, VR::CS, image_type.to_string()),
            (tags::ROWS, VR::US, "16".to_string()),
            (tags::COLUMNS, VR::US, "16".to_string()),
            (tags::SAMPLES_PER_PIXEL, VR::US, "1".to_string()),
            (
                tags::PHOTOMETRIC_INTERPRETATION,
                VR::CS,
                "MONOCHROME2".to_string(),
            ),
            (tags::BITS_ALLOCATED, VR::US, "16".to_string()),
            (tags::BITS_STORED, VR::US, "16".to_string()),
            (tags::HIGH_BIT, VR::US, "15".to_string()),
            (tags::PIXEL_REPRESENTATION, VR::US, "0".to_string()),
            (tags::PIXEL_SPACING, VR::DS, "1\\1".to_string()),
            (
                tags::IMAGE_ORIENTATION_PATIENT,
                VR::DS,
                "1\\0\\0\\0\\1\\0".to_string(),
            ),
            (tags::PIXEL_DATA, VR::OW, "256".to_string()),
            (
                tags::IMAGE_POSITION_PATIENT,
                VR::DS,
                format!("0\\0\\{}", k as f64 * 2.5),
            ),
            (
                tags::MANUFACTURER_MODEL_NAME,
                VR::LO,
                "Brilliance Big Bore".to_string(),
            ),
        ];
        if !desc.is_empty() {
            extra.push((tags::SERIES_DESCRIPTION, VR::LO, desc.to_string()));
        }
        if !acq.0.is_empty() {
            extra.push((tags::ACQUISITION_DATE, VR::DA, acq.0.to_string()));
            extra.push((tags::ACQUISITION_TIME, VR::TM, acq.1.to_string()));
        }
        if !protocol.is_empty() {
            extra.push((tags::PROTOCOL_NAME, VR::LO, protocol.to_string()));
        }
        file(
            &root.join(format!("Trial/P4D-01/{study}/{series}/{k}.dcm")),
            "1.2.840.10008.5.1.4.1.1.2",
            &format!("{series}.{k}"),
            &extra,
        );
    }
}

/// A structure set drawn on `on`.
pub fn structures(root: &std::path::Path, study: &str, series: &str, on: &str) {
    use dicom_core::{DataElement, VR};
    use dicom_dictionary_std::tags;
    use dicom_object::InMemDicomObject;
    let path = root.join(format!("Trial/P4D-01/{study}/{series}/rs.dcm"));
    let extra = vec![
        (tags::STUDY_INSTANCE_UID, VR::UI, study.to_string()),
        (tags::SERIES_INSTANCE_UID, VR::UI, series.to_string()),
        (tags::MODALITY, VR::CS, "RTSTRUCT".to_string()),
    ];
    file(
        &path,
        "1.2.840.10008.5.1.4.1.1.481.3",
        &format!("{series}.1"),
        &extra,
    );
    // The reference, nested as a structure set has it.
    let mut obj = dicom_object::open_file(&path).unwrap();
    let item = |tag, inner: InMemDicomObject| {
        InMemDicomObject::from_element_iter([DataElement::new(
            tag,
            VR::SQ,
            dicom_core::value::DataSetSequence::from(vec![inner]),
        )])
    };
    let se = InMemDicomObject::from_element_iter([DataElement::new(
        tags::SERIES_INSTANCE_UID,
        VR::UI,
        dicom_core::PrimitiveValue::from(on),
    )]);
    let st = item(tags::RT_REFERENCED_SERIES_SEQUENCE, se);
    let fr = item(tags::RT_REFERENCED_STUDY_SEQUENCE, st);
    let top = fr
        .element(tags::RT_REFERENCED_STUDY_SEQUENCE)
        .unwrap()
        .clone();
    let fr_item = InMemDicomObject::from_element_iter([top]);
    obj.put(DataElement::new(
        tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
        VR::SQ,
        dicom_core::value::DataSetSequence::from(vec![fr_item]),
    ));
    obj.write_to_file(&path).unwrap();
}

/// The Study Instance UIDs [`philips_4d`] writes: the 4DCT's, and the one
/// of a study holding only a 4D scout.
pub const STUDY_4D: &str = "1.2.826.0.1.4";
pub const STUDY_SCOUT: &str = "1.2.826.0.1.5";

/// Under `root/Trial/P4D-01/`: a Philips 4DCT exported without
/// descriptions (ten phases 201-210 of one acquisition time, four slices
/// each) beside three daily CTs numbered 201, its untagged reconstruction
/// (211), its respiratory trace and a structure set drawn on phase 205; and
/// a study holding only the scout of a 4DCT.
pub fn philips_4d(root: &std::path::Path) {
    use dicom_core::VR;
    use dicom_dictionary_std::tags;
    let st = "1.2.826.0.1.4";
    for i in 0..10 {
        ct_series(
            root,
            st,
            &format!("1.2.826.0.1.4.2{i:02}"),
            201 + i,
            "",
            ("20240610", "140847.12"),
            "ORIGINAL\\PRIMARY\\AXIAL",
            "",
            4,
        );
    }
    ct_series(
        root,
        st,
        "1.2.826.0.1.4.299",
        211,
        "Soft tissiu UnTagged",
        ("20240610", "140847.12"),
        "ORIGINAL\\PRIMARY\\AXIAL",
        "",
        4,
    );
    for d in 0..3 {
        ct_series(
            root,
            st,
            &format!("1.2.826.0.1.4.7{d}"),
            201,
            "",
            (&format!("2024061{}", d + 1), "090000"),
            "ORIGINAL\\PRIMARY\\AXIAL",
            "",
            3,
        );
    }
    structures(root, st, "1.2.826.0.1.4.88", "1.2.826.0.1.4.204");
    file(
        &root.join(format!("Trial/P4D-01/{st}/trace/1.dcm")),
        "1.2.840.10008.5.1.4.1.1.7",
        "1.2.826.0.1.4.99.1",
        &[
            (tags::STUDY_INSTANCE_UID, VR::UI, st.to_string()),
            (
                tags::SERIES_INSTANCE_UID,
                VR::UI,
                "1.2.826.0.1.4.99".to_string(),
            ),
            (tags::MODALITY, VR::CS, "OT".to_string()),
            (
                tags::SERIES_DESCRIPTION,
                VR::LO,
                "For Series: 201-210".to_string(),
            ),
        ],
    );
    let scout = "1.2.826.0.1.5";
    ct_series(
        root,
        scout,
        "1.2.826.0.1.5.1",
        100,
        "",
        ("20240903", "101000"),
        "ORIGINAL\\PRIMARY\\LOCALIZER",
        "4D LUNG /Onco Gating",
        2,
    );
}
