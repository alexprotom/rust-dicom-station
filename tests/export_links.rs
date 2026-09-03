//! What an export has to keep.
//!
//! The failure these tests exist for: a structure set came back from an
//! export with nothing but a Frame of Reference UID, so no planning system
//! would draw it on the CT it was made on. The reference chain is therefore
//! asserted at the DICOM level, on the written files, not on what the viewer
//! makes of them when it reads them back.

use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;
use rust_dicom_station::dicom_export::ExportParams;
use rust_dicom_station::dicomfile;
use rust_dicom_station::export::{self, ExportPlan, Layout, StructFormat, UidMode};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader::{self, LoadedStudy};
use rust_dicom_station::progress::Progress;

fn target(tag: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// The synthetic RT study, generated and loaded back.
fn phantom(tag: &str) -> (LoadedStudy, std::path::PathBuf) {
    let src = target(&format!("{tag}_src"));
    gen_test_data::generate(&src, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    let study = loader::load_directory(&src, &Progress::default()).expect("it loads");
    (study, src)
}

fn plan_for(study: &LoadedStudy) -> ExportPlan {
    ExportPlan::build([Some(study), None], ExportParams::for_study(study))
}

/// Every file under `dir` whose name starts with `prefix`.
fn files(dir: &std::path::Path, prefix: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with(prefix))
            {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

fn str_of(o: &InMemDicomObject, tag: dicom_core::Tag) -> String {
    loader::str_of(o, tag).unwrap_or_default()
}

/// `ReferencedFrameOfReferenceSequence ▶ RTReferencedStudy ▶ RTReferencedSeries`
/// of an RTSTRUCT object.
fn referenced_series(o: &InMemDicomObject) -> InMemDicomObject {
    let rfr = loader::items_of(o, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE)
        .expect("the structure set names a frame of reference");
    let rfr = rfr.first().expect("one frame of reference item");
    let study = loader::items_of(rfr, tags::RT_REFERENCED_STUDY_SEQUENCE)
        .expect("... and through it a study")
        .first()
        .expect("one study item")
        .clone();
    loader::items_of(&study, tags::RT_REFERENCED_SERIES_SEQUENCE)
        .expect("... and through that an image series")
        .first()
        .expect("one series item")
        .clone()
}

#[test]
fn the_structure_set_still_names_the_ct_it_was_drawn_on() {
    let (study, _) = phantom("test_exp_links");
    let out = target("test_exp_links_out");
    let plan = plan_for(&study);
    let sum = export::run(&plan, [Some(&study), None], &out, &Progress::default())
        .expect("the export runs");
    assert!(sum.files > study.volume.dims[2], "CT plus the RT objects");

    let ct: Vec<InMemDicomObject> = files(&out, "CT_")
        .iter()
        .map(|p| {
            dicomfile::open_full(p)
                .expect("a written slice reopens")
                .into_inner()
        })
        .collect();
    assert_eq!(ct.len(), study.volume.dims[2], "every slice was written");
    let ct_series: Vec<String> = ct
        .iter()
        .map(|o| str_of(o, tags::SERIES_INSTANCE_UID))
        .collect();
    let series_uid = ct_series[0].clone();
    assert!(
        ct_series.iter().all(|u| *u == series_uid),
        "one series for the whole stack"
    );
    let ct_sops: Vec<String> = ct
        .iter()
        .map(|o| str_of(o, tags::SOP_INSTANCE_UID))
        .collect();

    let rs_files = files(&out, "RS_");
    assert_eq!(rs_files.len(), 1, "one structure set");
    let rs = dicomfile::open_full(&rs_files[0])
        .expect("the structure set reopens")
        .into_inner();

    // The chain itself.
    let series_item = referenced_series(&rs);
    assert_eq!(
        str_of(&series_item, tags::SERIES_INSTANCE_UID),
        series_uid,
        "the structure set names the image series that was written"
    );
    let images = loader::items_of(&series_item, tags::CONTOUR_IMAGE_SEQUENCE)
        .expect("the series item lists its images");
    assert_eq!(images.len(), ct_sops.len(), "every slice is listed");
    for it in images {
        assert!(
            ct_sops.contains(&str_of(it, tags::REFERENCED_SOP_INSTANCE_UID)),
            "a listed image is one of the exported slices"
        );
    }

    // And every contour naming the slice it lies on.
    let rois = loader::items_of(&rs, tags::ROI_CONTOUR_SEQUENCE).expect("ROIs");
    let mut checked = 0usize;
    for roi in rois {
        for c in loader::items_of(roi, tags::CONTOUR_SEQUENCE).unwrap_or_default() {
            let refs = loader::items_of(c, tags::CONTOUR_IMAGE_SEQUENCE)
                .expect("a contour names the image it was drawn on");
            assert_eq!(refs.len(), 1);
            assert!(
                ct_sops.contains(&str_of(&refs[0], tags::REFERENCED_SOP_INSTANCE_UID)),
                "and that image is one of the exported slices"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 10,
        "a real number of contours was checked: {checked}"
    );
    assert!(
        sum.warnings.is_empty(),
        "nothing to report: {:?}",
        sum.warnings
    );
}

#[test]
fn keeping_the_uids_means_keeping_them() {
    let (study, _) = phantom("test_exp_keep");
    let out = target("test_exp_keep_out");
    let plan = plan_for(&study);
    assert_eq!(plan.uid_mode, UidMode::Keep, "the default is a true copy");
    export::run(&plan, [Some(&study), None], &out, &Progress::default()).expect("runs");

    let re = loader::load_directory(&out, &Progress::default()).expect("the export reloads");
    assert_eq!(
        re.series[re.active_series].uid, study.series[study.active_series].uid,
        "same Series Instance UID"
    );
    assert_eq!(
        re.series[re.active_series].study_uid, study.series[study.active_series].study_uid,
        "same Study Instance UID"
    );
    assert_eq!(
        re.volume.frame_of_reference_uid, study.volume.frame_of_reference_uid,
        "same Frame of Reference"
    );
    assert_eq!(
        re.structure_sets[0].sop_instance_uid, study.structure_sets[0].sop_instance_uid,
        "same structure set instance"
    );
    assert_eq!(
        re.structure_sets[0].referenced_series_uid, re.series[re.active_series].uid,
        "and it points at the CT"
    );
}

#[test]
fn new_uids_are_new_everywhere_and_still_agree_with_each_other() {
    let (study, _) = phantom("test_exp_new");
    let out = target("test_exp_new_out");
    let mut plan = plan_for(&study);
    plan.set_uid_mode(UidMode::New);
    export::run(&plan, [Some(&study), None], &out, &Progress::default()).expect("runs");

    let re = loader::load_directory(&out, &Progress::default()).expect("the export reloads");
    let a = &re.series[re.active_series];
    let b = &study.series[study.active_series];
    assert_ne!(a.uid, b.uid, "a new series");
    assert_ne!(a.study_uid, b.study_uid, "in a new study");
    assert_ne!(
        re.volume.frame_of_reference_uid, study.volume.frame_of_reference_uid,
        "with a new frame of reference"
    );
    // Internally consistent all the same.
    assert_eq!(
        re.structure_sets[0].referenced_series_uid, a.uid,
        "the structure set follows the new series"
    );
    assert_eq!(
        re.structure_sets[0].frame_of_reference_uid, re.volume.frame_of_reference_uid,
        "and the new frame of reference"
    );
    assert_eq!(
        re.plans[0].referenced_structset_uid, re.structure_sets[0].sop_instance_uid,
        "the plan follows the new structure set"
    );
    assert_eq!(
        re.doses[0].referenced_plan_uid, re.plans[0].sop_instance_uid,
        "and the dose follows the new plan"
    );
}

#[test]
fn a_structure_set_can_go_out_as_seg_and_come_back_as_masks() {
    let (study, _) = phantom("test_exp_seg");
    let out = target("test_exp_seg_out");
    let mut plan = plan_for(&study);
    plan.set_all_formats(StructFormat::Seg);
    let sum = export::run(&plan, [Some(&study), None], &out, &Progress::default()).expect("runs");
    assert!(files(&out, "RS_").is_empty(), "no RTSTRUCT was written");
    assert_eq!(files(&out, "SEG_").len(), 1, "one SEG instead");

    let re = loader::load_directory(&out, &Progress::default()).expect("reloads");
    assert!(re.structure_sets.is_empty());
    assert_eq!(re.seg_series.len(), 1);
    let n_rois = study.structure_sets[0].rois.len();
    assert_eq!(
        re.seg_series[0].segs.len(),
        n_rois,
        "every ROI became a segment ({:?})",
        sum.warnings
    );
    assert_eq!(
        re.seg_series[0].referenced_series_uid, re.series[re.active_series].uid,
        "and the SEG points at the CT"
    );
    for s in &re.seg_series[0].segs {
        assert!(s.count > 0, "segment “{}” is not empty", s.name);
    }
}

#[test]
fn a_segmentation_series_can_go_out_as_rtstruct() {
    let (mut study, _) = phantom("test_exp_rs");
    // Turn the structure set into a segmentation series, and export only that.
    let grid = study.volume.grid();
    let mut ser = rust_dicom_station::dicomseg::SegSeries::new(
        "From contours".into(),
        grid.clone(),
        study.series[study.active_series].uid.clone(),
        study.series[study.active_series].study_uid.clone(),
    );
    for roi in &study.structure_sets[0].rois {
        if let Some(mask) = rust_dicom_station::segmentation::rasterize_roi(&grid, roi) {
            ser.segs
                .push(rust_dicom_station::segmentation::Segmentation::from_mask(
                    roi.name.clone(),
                    roi.color,
                    grid.dims,
                    mask,
                ));
        }
    }
    let n_segs = ser.segs.len();
    assert!(n_segs >= 3);
    study.seg_series.push(ser);
    study.structure_sets.clear();

    let out = target("test_exp_rs_out");
    let mut plan = plan_for(&study);
    plan.set_all_formats(StructFormat::RtStruct);
    export::run(&plan, [Some(&study), None], &out, &Progress::default()).expect("runs");
    assert_eq!(files(&out, "RS_").len(), 1, "written as a structure set");
    assert!(files(&out, "SEG_").is_empty());

    let re = loader::load_directory(&out, &Progress::default()).expect("reloads");
    assert_eq!(re.structure_sets.len(), 1);
    assert_eq!(
        re.structure_sets[0].rois.len(),
        n_segs,
        "one ROI per segment"
    );
    assert!(
        re.structure_sets[0]
            .rois
            .iter()
            .all(|r| !r.contours.is_empty()),
        "with contours"
    );
    assert_eq!(
        re.structure_sets[0].referenced_series_uid, re.series[re.active_series].uid,
        "and still bound to the CT"
    );
}

/// A converted object is a different SOP class, so it must not be written
/// under the instance UID of the object it was made from, even in Keep mode.
#[test]
fn converting_the_format_mints_a_new_instance_uid() {
    let (study, _) = phantom("test_exp_conv");
    let mut plan = plan_for(&study);
    let native: String = plan
        .studies()
        .flat_map(|s| s.objects.iter())
        .find(|o| o.kind.is_structures())
        .map(|o| o.sop_uid.value.clone())
        .expect("a structure set");
    assert_eq!(
        native, study.structure_sets[0].sop_instance_uid,
        "kept as it is while the format is the native one"
    );

    plan.set_all_formats(StructFormat::Seg);
    let converted: String = plan
        .studies()
        .flat_map(|s| s.objects.iter())
        .find(|o| o.kind.is_structures())
        .map(|o| o.sop_uid.value.clone())
        .expect("still there");
    assert_ne!(converted, native, "a conversion is a new instance");

    plan.set_all_formats(StructFormat::RtStruct);
    let back: String = plan
        .studies()
        .flat_map(|s| s.objects.iter())
        .find(|o| o.kind.is_structures())
        .map(|o| o.sop_uid.value.clone())
        .expect("still there");
    assert_eq!(back, native, "and converting back restores the original");
}

/// The series an RT object came from is part of its identity.
#[test]
fn keeping_the_uids_keeps_the_series_of_the_rt_objects_too() {
    let (study, _) = phantom("test_exp_objser");
    let out = target("test_exp_objser_out");
    let want = study.structure_sets[0].series_instance_uid.clone();
    assert!(!want.is_empty(), "the loader read it");
    let plan = plan_for(&study);
    export::run(&plan, [Some(&study), None], &out, &Progress::default()).expect("runs");

    let rs = files(&out, "RS_");
    let o = dicomfile::open_full(&rs[0]).expect("reopens").into_inner();
    assert_eq!(
        str_of(&o, tags::SERIES_INSTANCE_UID),
        want,
        "the structure set went back into its own series"
    );
}

#[test]
fn structures_exported_without_their_images_say_so() {
    let (study, _) = phantom("test_exp_alone");
    let out = target("test_exp_alone_out");
    let mut plan = plan_for(&study);
    for st in plan.studies_mut() {
        for s in &mut st.series {
            s.selected = false;
        }
    }
    let sum = export::run(&plan, [Some(&study), None], &out, &Progress::default()).expect("runs");
    assert!(files(&out, "CT_").is_empty(), "no images were written");
    assert!(
        sum.warnings
            .iter()
            .any(|w| w.contains("without its image series")),
        "the run reports the missing link: {:?}",
        sum.warnings
    );
}

#[test]
fn the_folder_layout_separates_patients_studies_and_series() {
    let (study, _) = phantom("test_exp_tree");
    let out = target("test_exp_tree_out");
    let plan = plan_for(&study);
    assert_eq!(plan.layout, Layout::Tree);
    export::run(&plan, [Some(&study), None], &out, &Progress::default()).expect("runs");

    let ct = files(&out, "CT_");
    assert!(!ct.is_empty());
    let rel = ct[0].strip_prefix(&out).expect("under the output folder");
    assert_eq!(
        rel.components().count(),
        4,
        "patient / study / series / file, got {}",
        rel.display()
    );
    // And the whole thing still loads as one study.
    let re = loader::load_directory(&out, &Progress::default()).expect("reloads");
    assert_eq!(re.volume.dims, study.volume.dims);
    assert_eq!(re.structure_sets.len(), 1);
}
