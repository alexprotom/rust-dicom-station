//! Reports read from files: a scanner's protocol page (a Secondary Capture
//! with blank pixels, its text in an overlay plane and in the private STEP
//! element) and a CT dose SR, written into the phantom's folder as a
//! scanner would and loaded as what they are: tables, with the page
//! rendered beside the protocol's.

mod common;

use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::report::ReportKind;

#[test]
fn a_protocol_page_and_a_dose_sr_load_as_reports_not_images() {
    let dir = common::target_dir("test_reports_load");
    let folder = common::fourd_folder(&dir, [0.0, 0.0, 0.0]);
    common::protocol_page_into(&folder, "Patientenprotokoll");
    common::dose_report_into(&folder, "HOUSE^GREGORY");
    let study = loader::load_directory(&folder, &Progress::default()).expect("loads");

    // Neither is a planar image, and nothing was dropped.
    assert!(
        study.planar_images.is_empty(),
        "a page is no image: {:?}",
        study
            .planar_images
            .iter()
            .map(|p| p.label.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(study.reports.len(), 2, "{:?}", study.warnings);
    assert!(
        !study
            .warnings
            .iter()
            .any(|w| w.contains("protocol_page") || w.contains("dose_report")),
        "{:?}",
        study.warnings
    );

    // The page: its table from the STEP text, the page itself with the
    // overlay drawn into blank pixels.
    let page = study
        .reports
        .iter()
        .find(|r| r.kind == ReportKind::ProtocolPage)
        .expect("the protocol page");
    assert_eq!(page.label, "Patientenprotokoll");
    assert_eq!(page.modality, "CT");
    assert_eq!(page.date, "2025-03-07");
    assert_eq!(page.tables.len(), 1, "{:?}", page.notes);
    let t = &page.tables[0];
    assert_eq!(
        t.columns,
        vec!["", "Scan", "kV", "mAs / ref.", "CTDIvol mGy", "DLP mGy*cm"]
    );
    assert_eq!(
        t.rows[1],
        vec!["AbdRoutine", "2", "130", "73 / 110", "8.18(L)", "389.33"]
    );
    assert!(page.notes.iter().any(|n| n.starts_with("Gesamt mAs")));
    let img = page.page.as_ref().expect("the page is kept");
    assert!(
        img.max_value > img.min_value,
        "the overlay lit the blank page"
    );
    assert_eq!(img.data[0], img.max_value, "the first pixel is overlay");
    assert_eq!(img.data[8], img.min_value, "the ninth is not");
    assert_eq!(
        img.window,
        (
            (img.min_value + img.max_value) * 0.5,
            img.max_value - img.min_value
        ),
        "windowed over its own values, not the header's 200 / 50"
    );

    // The dose SR: the acquisitions as rows, the totals as facts, the
    // physician in the tree and nowhere else.
    let dose = study
        .reports
        .iter()
        .find(|r| r.kind == ReportKind::DoseReport)
        .expect("the dose report");
    assert_eq!(dose.label, "Dose Report");
    assert_eq!(dose.modality, "SR");
    let t = &dose.tables[0];
    assert_eq!(t.rows.len(), 2);
    assert_eq!(t.rows[0][1], "Steinsuche");
    assert_eq!(t.rows[1][11], "8.18");
    assert_eq!(t.rows[1][12], "389.33");
    assert!(dose
        .info
        .iter()
        .any(|(k, v)| k == "Total DLP" && v == "508.44 mGy.cm"));
    assert!(dose.page.is_none());
    let tree = dose.tree_table(false).expect("the content tree");
    assert!(tree.rows.iter().any(|r| r[1] == "HOUSE GREGORY"));
    let hidden = dose.tree_table(true).unwrap();
    assert!(
        !hidden.rows.iter().any(|r| r[1].contains("GREGORY")),
        "names withheld on request"
    );
    for t in &dose.tables {
        for row in &t.rows {
            assert!(!row.iter().any(|c| c.contains("GREGORY")), "{row:?}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_secondary_capture_without_text_stays_an_additional_image() {
    use dicom_core::{DataElement, PrimitiveValue, Tag, VR};
    use dicom_dictionary_std::tags;
    let dir = common::target_dir("test_reports_plain_sc");
    let folder = common::fourd_folder(&dir, [0.0, 0.0, 0.0]);
    common::protocol_page_into(&folder, "Screenshot");
    // Take the text away again: a page with pixels (its overlay) and no
    // words is an image.
    let path = folder.join("protocol_page.dcm");
    let obj = dicom_object::open_file(&path).unwrap();
    let mut o = obj.into_inner();
    o.remove_element(Tag(0x0029, 0x1110));
    o.put(DataElement::new(
        tags::SOP_INSTANCE_UID,
        VR::UI,
        PrimitiveValue::from("1.2.3.4.5.6.7.8.9"),
    ));
    o.with_meta(
        dicom_object::FileMetaTableBuilder::new()
            .transfer_syntax("1.2.840.10008.1.2.1")
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
            .media_storage_sop_instance_uid("1.2.3.4.5.6.7.8.9"),
    )
    .unwrap()
    .write_to_file(&path)
    .unwrap();
    let study = loader::load_directory(&folder, &Progress::default()).expect("loads");
    assert!(study.reports.is_empty());
    assert_eq!(study.planar_images.len(), 1, "{:?}", study.warnings);
    let img = &study.planar_images[0];
    assert_eq!(img.kind, rust_dicom_station::extras::PlanarKind::Additional);
    assert_eq!(img.label, "Screenshot");
    assert!(img.max_value > img.min_value, "the overlay is drawn");
    let _ = std::fs::remove_dir_all(&dir);
}
