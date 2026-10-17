//! The local archive, end to end: file a study into it, list it without
//! opening a DICOM file, load it back into a dataset the way the PACS window
//! does, draw something on it, and send the derived objects back so they land
//! in the same patient and study rather than beside them.
//!
//! This is the whole round trip the archive exists for, so it is tested as
//! one path rather than as four isolated calls.

use rust_dicom_station::archive::Archive;
use rust_dicom_station::dicom_export::{self, ExportParams};
use rust_dicom_station::dicomseg::SegSeries;
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::segmentation::Segmentation;

fn scratch(tag: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A solid ellipsoid around the volume centre - something with a non-zero
/// voxel count, which is what makes the writer emit the object at all.
fn ball(dims: [usize; 3], radius: f64) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let c = [
        (nx as f64 - 1.0) * 0.5,
        (ny as f64 - 1.0) * 0.5,
        (nz as f64 - 1.0) * 0.5,
    ];
    let mut m = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d = ((i as f64 - c[0]).powi(2)
                    + (j as f64 - c[1]).powi(2)
                    + (k as f64 - c[2]).powi(2))
                .sqrt();
                if d <= radius {
                    m[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
    }
    m
}

#[test]
fn a_study_files_lists_loads_and_takes_back_what_was_drawn_on_it() {
    let src = scratch("test_archive_src");
    let written = gen_test_data::generate(&src, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");

    let root = scratch("test_archive_root");
    let archive = Archive::new(&root);

    // ---- file it --------------------------------------------------------
    let sum = archive
        .import(&src, &Progress::default())
        .expect("import succeeds");
    assert_eq!(sum.stored, written, "every generated file was filed");
    assert_eq!(sum.skipped, 0, "everything the generator wrote is DICOM");
    assert_eq!(sum.duplicates, 0);
    assert_eq!(
        (sum.patients, sum.studies),
        (1, 1),
        "one patient, one study"
    );

    // ---- list it --------------------------------------------------------
    let patients = archive.scan().expect("scan succeeds");
    assert_eq!(patients.len(), 1);
    let p = &patients[0];
    assert_eq!(p.title(), "PHANTOM RT (RTTEST001)", "the listed identity");
    assert_eq!(p.studies.len(), 1);
    let entry = p.studies[0].clone();
    assert_eq!(entry.files, written, "the sidecar counts what is there");
    assert!(
        entry.modalities.iter().any(|m| m == "CT")
            && entry.modalities.iter().any(|m| m == "RTSTRUCT"),
        "the sidecar lists the modalities it holds: {:?}",
        entry.modalities
    );
    let line = entry.describe();
    assert!(
        line.contains("CT") && line.ends_with(&format!("{written} files")),
        "the one-line description the window shows: {line}"
    );

    // Importing the same folder again must be a no-op, not a second copy.
    let again = archive
        .import(&src, &Progress::default())
        .expect("re-import succeeds");
    assert_eq!(
        (again.stored, again.duplicates),
        (0, written),
        "the same SOP Instance UIDs are recognised"
    );

    // ---- take it into a dataset -----------------------------------------
    // The PACS window loads an archived study folder with the ordinary
    // directory scanner, so that is exactly what is asserted here.
    let mut study =
        loader::load_directory(&entry.dir, &Progress::default()).expect("the archived study loads");
    assert!(!study.series.is_empty(), "the CT came back");
    assert_eq!(
        study.series[study.active_series].study_uid, entry.study_uid,
        "the loaded study is the one the archive listed"
    );
    let structs_before = study.structure_sets.len();

    // ---- draw on it -----------------------------------------------------
    let dims = study.volume.dims;
    let mut ser = SegSeries::new(
        "Archive QA".into(),
        study.volume.grid(),
        study.series[study.active_series].uid.clone(),
        study.series[study.active_series].study_uid.clone(),
    );
    ser.segs.push(Segmentation::from_mask(
        "Ball".into(),
        [220, 40, 40],
        dims,
        ball(dims, 8.0),
    ));
    study.seg_series.push(ser);

    // ---- send it back ---------------------------------------------------
    let derived = scratch("test_archive_derived");
    let params = ExportParams::for_study(&study);
    let n = dicom_export::export_derived(&study, &derived, &params, &Progress::default())
        .expect("the derived export succeeds");
    assert_eq!(
        n,
        structs_before + 1,
        "one file per structure set plus the new segmentation series, and nothing else"
    );
    for f in std::fs::read_dir(&derived).unwrap().filter_map(|e| e.ok()) {
        let name = f.file_name().to_string_lossy().to_string();
        assert!(
            name.starts_with("RS_") || name.starts_with("SEG_"),
            "no image data is re-sent, found {name}"
        );
    }

    let up = archive
        .import(&derived, &Progress::default())
        .expect("upload succeeds");
    assert_eq!(up.stored, n, "the derived objects are new instances");
    assert_eq!(
        (up.patients, up.studies),
        (1, 1),
        "they land in one patient and one study - the ones already there"
    );

    let patients = archive.scan().expect("rescan succeeds");
    assert_eq!(patients.len(), 1, "no second patient was invented");
    let p = &patients[0];
    assert_eq!(p.studies.len(), 1, "no second study was invented");
    assert_eq!(
        p.studies[0].study_uid, entry.study_uid,
        "the Study Instance UID is kept, which is what files them together"
    );
    assert_eq!(
        p.studies[0].files,
        written + n,
        "the archive grew by the derived objects"
    );
    assert!(
        p.studies[0].modalities.iter().any(|m| m == "SEG"),
        "the segmentation is now part of the study: {:?}",
        p.studies[0].modalities
    );

    // And the study reads back with what was drawn on it.
    let back = loader::load_directory(&p.studies[0].dir, &Progress::default())
        .expect("the enriched study loads");
    assert_eq!(back.seg_series.len(), 1, "the segmentation came back");
    assert_eq!(back.seg_series[0].segs.len(), 1);
    assert_eq!(back.seg_series[0].segs[0].name, "Ball");
    assert_eq!(
        back.structure_sets.len(),
        structs_before * 2,
        "the re-sent structure set is a new instance beside the original"
    );

    // ---- and it can be taken out again ----------------------------------
    archive.remove(&p.dir).expect("removing a patient succeeds");
    assert!(archive.scan().expect("scan succeeds").is_empty());
}

/// A study that was not filed by the archive but copied into it by hand -
/// the folder called whatever the person called it, the files whatever the
/// exporting system called them, a folder per series below - is listed by
/// its real UID, found by that UID, served file by file, and takes new
/// objects into its own folder rather than into a second one.
#[test]
fn a_study_copied_in_by_hand_is_as_good_as_a_filed_one() {
    let src = scratch("test_archive_hand_src");
    let written = gen_test_data::generate(&src, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");

    // What a person does: a patient folder, a study folder named by date and
    // description, the files under two "series" folders with names of the
    // exporting system's choosing.
    let root = scratch("test_archive_hand_root");
    let sdir = root.join("Doe, John").join("2026.10.09 CT Abdomen");
    let series = [sdir.join("CT 1.0 Bf37"), sdir.join("other")];
    for d in &series {
        std::fs::create_dir_all(d).unwrap();
    }
    let mut n = 0usize;
    for f in std::fs::read_dir(&src).unwrap().filter_map(|e| e.ok()) {
        if !f.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let name = f.file_name().to_string_lossy().to_string();
        let into = if name.starts_with("CT_") {
            &series[0]
        } else {
            &series[1]
        };
        std::fs::copy(f.path(), into.join(format!("{:04}", n))).unwrap();
        n += 1;
    }
    assert_eq!(n, written);

    let archive = Archive::new(&root);
    let patients = archive.scan().expect("scan succeeds");
    assert_eq!(patients.len(), 1);
    let entry = patients[0].studies[0].clone();
    assert!(
        rust_dicom_station::archive::is_uid(&entry.study_uid),
        "the sidecar carries the UID from the headers: {:?}",
        entry.study_uid
    );
    assert_eq!(
        entry.files, written,
        "the files in the series folders are counted"
    );
    assert_eq!(entry.dir, sdir);
    assert!(
        entry.modalities.iter().any(|m| m == "CT"),
        "{:?}",
        entry.modalities
    );

    // Found by UID although the folder is not called that: what a PACS
    // server does for every request a station makes about the study.
    assert_eq!(archive.find_study(&entry.study_uid), Some(sdir.clone()));
    let inst = archive
        .study_instances(&entry.study_uid)
        .expect("instances");
    assert_eq!(
        inst.len(),
        written,
        "every file below the study folder, whatever its name"
    );
    assert!(inst
        .iter()
        .all(|i| rust_dicom_station::archive::is_uid(&i.sop_uid)));
    assert!(inst.iter().all(|i| i.path.starts_with(&sdir)));

    // New objects for the study go into its folder, and sending the
    // originals again files nothing.
    let mut study =
        loader::load_directory(&sdir, &Progress::default()).expect("the hand-copied study loads");
    let dims = study.volume.dims;
    let mut ser = SegSeries::new(
        "Hand QA".into(),
        study.volume.grid(),
        study.series[study.active_series].uid.clone(),
        study.series[study.active_series].study_uid.clone(),
    );
    ser.segs.push(Segmentation::from_mask(
        "Ball".into(),
        [220, 40, 40],
        dims,
        ball(dims, 8.0),
    ));
    study.seg_series.push(ser);
    let derived = scratch("test_archive_hand_derived");
    let params = ExportParams::for_study(&study);
    let made = dicom_export::export_derived(&study, &derived, &params, &Progress::default())
        .expect("the derived export succeeds");
    let up = archive
        .import(&derived, &Progress::default())
        .expect("upload");
    assert_eq!(up.stored, made);
    let again = archive
        .import(&src, &Progress::default())
        .expect("re-import");
    assert_eq!(
        (again.stored, again.duplicates),
        (0, written),
        "the originals are recognised by their SOP UIDs, not by file names"
    );
    let patients = archive.scan().expect("rescan");
    assert_eq!(patients.len(), 1, "no second patient folder was made");
    assert_eq!(
        patients[0].studies.len(),
        1,
        "no second study folder was made"
    );
    assert_eq!(patients[0].studies[0].files, written + made);
    assert!(patients[0].studies[0].modalities.iter().any(|m| m == "SEG"));
    let back = loader::load_directory(&sdir, &Progress::default()).expect("loads again");
    assert_eq!(
        back.seg_series.len(),
        1,
        "the segmentation sits beside the CT"
    );
}

/// A patient's CD: one folder, the files of two studies in it. The archive
/// lists two studies for the folder, serves each by its own files, and
/// removes one without touching the other.
#[test]
fn a_folder_holding_two_studies_is_listed_and_served_as_two() {
    let src_a = scratch("test_archive_two_src_a");
    let n_a = gen_test_data::generate(&src_a, &GenParams::default(), &Progress::default())
        .expect("generation A");
    let src_b = scratch("test_archive_two_src_b");
    let params = GenParams {
        plan_label: "SecondVisit".into(),
        ..GenParams::default()
    };
    let n_b = gen_test_data::generate(&src_b, &params, &Progress::default()).expect("generation B");

    let root = scratch("test_archive_two_root");
    let sdir = root.join("Doe, John").join("CD 2026");
    let (dir_a, dir_b) = (sdir.join("1"), sdir.join("2"));
    for (src, into) in [(&src_a, &dir_a), (&src_b, &dir_b)] {
        std::fs::create_dir_all(into).unwrap();
        for (i, f) in std::fs::read_dir(src)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .enumerate()
        {
            std::fs::copy(f.path(), into.join(format!("IM{i:05}"))).unwrap();
        }
    }

    let archive = Archive::new(&root);
    let patients = archive.scan().expect("scan");
    assert_eq!(patients.len(), 1);
    let studies = &patients[0].studies;
    assert_eq!(studies.len(), 2, "two studies, one folder: {studies:?}");
    assert!(
        studies.iter().all(|s| s.dir == dir_a || s.dir == dir_b),
        "each study's folder is the one holding its files"
    );
    assert_ne!(studies[0].study_uid, studies[1].study_uid);
    let total: usize = studies.iter().map(|s| s.files).sum();
    assert_eq!(
        total,
        n_a + n_b,
        "every file is counted once, under its study"
    );
    assert_eq!(patients[0].files(), n_a + n_b);

    // Each study is found and served by its own files only.
    for s in studies {
        assert_eq!(archive.find_study(&s.study_uid), Some(s.dir.clone()));
        let mine = archive
            .study_instances(&s.study_uid)
            .expect("instances of the study");
        assert_eq!(mine.len(), s.files);
        let files: Vec<_> = mine.into_iter().map(|i| i.path).collect();
        let loaded =
            loader::load_files(&files, "test", &Progress::default()).expect("the study loads");
        assert!(loaded.series.iter().all(|ser| ser.study_uid == s.study_uid));
    }
    // One catalog for the PACS, nothing written into the folders.
    assert!(root
        .join(rust_dicom_station::archive::CATALOG_FILE)
        .is_file());
    for d in [&sdir, &dir_a, &dir_b] {
        for name in ["STUDY.txt", "PATIENT.txt", "INDEX.txt"] {
            assert!(!d.join(name).exists(), "{name} in {}", d.display());
        }
    }

    // Removing one study leaves the other, its index and its listing whole.
    let gone = studies[1].study_uid.clone();
    let kept = studies[0].study_uid.clone();
    let kept_dir = studies[0].dir.clone();
    archive.remove_study(&gone).expect("remove one study");
    assert!(sdir.is_dir(), "the folder stays for the other study");
    assert_eq!(archive.find_study(&gone), None);
    assert_eq!(archive.find_study(&kept), Some(kept_dir));
    let patients = archive.scan().expect("rescan");
    assert_eq!(patients[0].studies.len(), 1);
    assert_eq!(patients[0].studies[0].study_uid, kept);
    assert_eq!(patients[0].studies[0].files, studies[0].files);
    assert_eq!(
        archive.study_instances(&kept).unwrap().len(),
        studies[0].files,
        "the kept study's files are all there"
    );
    archive.remove_study(&kept).expect("remove the last study");
    assert!(!sdir.exists(), "an emptied folder goes");
}

/// A DICOM file with nothing but what the catalog reads.
fn tiny(
    path: &std::path::Path,
    patient: (&str, &str),
    study: &str,
    series: &str,
    sop: &str,
    modality: &str,
) {
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_dictionary_std::tags;
    use dicom_object::{FileMetaTableBuilder, InMemDicomObject};
    let class = "1.2.840.10008.5.1.4.1.1.2";
    let mut obj = InMemDicomObject::new_empty();
    for (tag, vr, v) in [
        (tags::SOP_CLASS_UID, VR::UI, class),
        (tags::SOP_INSTANCE_UID, VR::UI, sop),
        (tags::STUDY_INSTANCE_UID, VR::UI, study),
        (tags::SERIES_INSTANCE_UID, VR::UI, series),
        (tags::PATIENT_ID, VR::LO, patient.0),
        (tags::PATIENT_NAME, VR::PN, patient.1),
        (tags::MODALITY, VR::CS, modality),
        (tags::STUDY_DATE, VR::DA, "20260102"),
    ] {
        obj.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
    }
    let file = obj
        .with_meta(
            FileMetaTableBuilder::new()
                .transfer_syntax("1.2.840.10008.1.2.1")
                .media_storage_sop_class_uid(class)
                .media_storage_sop_instance_uid(sop),
        )
        .expect("meta");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    file.write_to_file(path).expect("write");
}

/// The PACS as a person lays it out: cohort folders of patients, a patient
/// folder outside any, a further storage folder. Cohorts and names live in
/// the catalog (no file is touched for them, and a rescan keeps them);
/// moving files is its own act, within one storage folder.
#[test]
fn cohorts_names_and_moves_live_in_the_one_catalog() {
    use rust_dicom_station::archive::{Pick, UNSORTED};

    let root = scratch("test_archive_cohorts");
    let x24 = ("X24", "GC");
    tiny(
        &root.join("Trial/X24/CT/a"),
        x24,
        "1.1",
        "1.1.1",
        "9.1",
        "CT",
    );
    tiny(
        &root.join("Trial/X24/CT/b"),
        x24,
        "1.1",
        "1.1.1",
        "9.2",
        "CT",
    );
    tiny(
        &root.join("Trial/X24/RT/c"),
        x24,
        "1.1",
        "1.1.2",
        "9.3",
        "RTSTRUCT",
    );
    tiny(
        &root.join("Trial/X31/d"),
        ("X31", "HM"),
        "1.2",
        "1.2.1",
        "9.4",
        "MR",
    );
    tiny(
        &root.join("P79/2020/e"),
        ("P79", "ND"),
        "1.3",
        "1.3.1",
        "9.5",
        "CT",
    );
    tiny(
        &root.join("Personal/Visit/f"),
        ("RP-1", "DOE^JOHN"),
        "1.4",
        "1.4.1",
        "9.6",
        "CT",
    );
    // A sidecar an older version wrote goes; a person's own note stays.
    std::fs::write(root.join("Trial/X24/STUDY.txt"), "uid = 1.1\nfiles = 3\n").unwrap();
    std::fs::write(root.join("Trial/notes.txt"), "my notes\n").unwrap();

    let archive = Archive::new(&root);
    let tree = archive.scan_tree(&Progress::default()).expect("scan");
    let names: Vec<&str> = tree.cohorts.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["Personal", "Trial", "Unsorted"], "Unsorted last");
    let trial = tree
        .cohorts
        .iter()
        .find(|c| c.name == "Trial")
        .unwrap()
        .clone();
    assert_eq!(trial.patients.len(), 2);
    let gc = trial.patients.iter().find(|p| p.key == "X24").unwrap();
    assert_eq!(gc.studies.len(), 1);
    assert_eq!(gc.studies[0].series.len(), 2, "CT and RTSTRUCT");
    assert_eq!(gc.studies[0].files, 3);
    assert_eq!(gc.studies[0].date, "20260102");
    let unsorted = tree.cohorts.iter().find(|c| c.id == UNSORTED).unwrap();
    assert_eq!(
        unsorted.patients[0].key, "P79",
        "a patient's own folder is Unsorted"
    );
    assert!(!root.join("Trial/X24/STUDY.txt").exists());
    assert!(root.join("Trial/notes.txt").exists());

    // Names: the catalog's, not the files'.
    archive
        .rename(&Pick::Cohort(trial.id.clone()), "Compassionate")
        .unwrap();
    archive
        .rename(&Pick::Patient("X24".into()), "Case 24")
        .unwrap();
    archive
        .rename(&Pick::Series("1.1.2".into()), "Clinical contours")
        .unwrap();
    // Cohorts: a new one, X24 moved into it, X31 copied.
    let sel = archive.new_cohort("Selected").unwrap();
    archive
        .put_in_cohort(&["X24".into()], Some(&trial.id), &sel, false)
        .unwrap();
    archive
        .put_in_cohort(&["X31".into()], Some(&trial.id), &sel, true)
        .unwrap();
    // A rescan keeps all of it.
    let tree = archive.scan_tree(&Progress::default()).expect("rescan");
    let by = |name: &str| tree.cohorts.iter().find(|c| c.name == name).cloned();
    let comp = by("Compassionate").expect("renamed cohort");
    let selected = by("Selected").expect("new cohort");
    let keys = |c: &rust_dicom_station::archive::CohortEntry| {
        c.patients.iter().map(|p| p.key.clone()).collect::<Vec<_>>()
    };
    assert_eq!(keys(&comp), ["X31"], "X24 was moved out");
    assert_eq!(keys(&selected), ["X24", "X31"], "X31 copied, X24 moved");
    let case = &selected.patients[0];
    assert_eq!(case.title(), "Case 24");
    assert_eq!(case.dicom_title(), "GC (X24)");
    assert!(case.studies[0]
        .series
        .iter()
        .any(|s| s.title() == "Clinical contours"));
    assert!(
        root.join("Trial/X24/CT/a").is_file(),
        "no file moved for any of it"
    );

    // What a selection stands for.
    assert_eq!(archive.files_of(&[Pick::Series("1.1.2".into())]).len(), 1);
    assert_eq!(archive.files_of(&[Pick::Cohort(sel.clone())]).len(), 4);
    assert_eq!(
        archive
            .files_of(&[Pick::Study("1.1".into()), Pick::Series("1.1.1".into())])
            .len(),
        3,
        "a series inside a chosen study is not counted twice"
    );

    // Moving files: a study into a new folder, its series folders kept.
    let rel = archive.new_folder(0, "", "Moved").unwrap();
    let sum = archive
        .move_into(&[Pick::Study("1.1".into())], 0, &rel)
        .expect("move");
    assert_eq!(sum.moved, 3);
    assert!(root.join("Moved/CT/a").is_file() && root.join("Moved/RT/c").is_file());
    assert!(!root.join("Trial/X24").exists(), "the emptied folder went");
    assert_eq!(archive.find_study("1.1"), Some(root.join("Moved")));
    assert!(
        archive
            .move_into(&[Pick::Study("1.2".into())], 0, "Moved")
            .is_ok(),
        "a second move into the same folder"
    );
    assert!(
        archive
            .move_into(&[Pick::Study("1.2".into())], 0, "Trial")
            .is_ok(),
        "and back"
    );

    // A second storage folder is read where it is; the catalog stays in
    // the main one, and files do not move between the two.
    let more = scratch("test_archive_cohorts_more");
    tiny(
        &more.join("Q1/x"),
        ("Q1", "QQ"),
        "1.9",
        "1.9.1",
        "9.9",
        "CT",
    );
    let both = Archive::with_roots(vec![root.clone(), more.clone()]);
    let tree = both.scan_tree(&Progress::default()).expect("scan both");
    assert_eq!(tree.patients().len(), 5);
    assert!(!more
        .join(rust_dicom_station::archive::CATALOG_FILE)
        .exists());
    assert_eq!(both.storage_of(&[Pick::Study("1.9".into())]), Some(1));
    assert!(both
        .move_into(&[Pick::Study("1.9".into())], 0, "Moved")
        .is_err());
    // The one-folder view of the main folder still sees its own files.
    assert_eq!(Archive::new(&root).scan().unwrap().len(), 4);

    // Removing deletes the files and the folders they leave empty.
    let n = both.remove_picks(&[Pick::Patient("X31".into())]).unwrap();
    assert_eq!(n, 1);
    assert!(!root.join("Trial/X31").exists());
    assert!(
        root.join("Trial/notes.txt").exists(),
        "a person's file stays, and its folder"
    );
}

/// A scan that finds nothing new and nothing gone leaves the catalog file
/// alone: a PACS server listing the archive every few seconds walks the
/// folders and writes nothing. A file that does change is caught.
#[test]
fn a_rescan_that_finds_nothing_new_leaves_the_catalog_alone() {
    let src = scratch("test_archive_quiet_src");
    gen_test_data::generate(&src, &GenParams::default(), &Progress::default()).unwrap();
    let root = scratch("test_archive_quiet_root");
    let archive = Archive::new(&root);
    archive.import(&src, &Progress::default()).unwrap();
    archive.refresh(&Progress::default()).unwrap();
    let cat = archive.catalog_path();
    let stamp = |p: &std::path::Path| {
        let m = std::fs::metadata(p).unwrap();
        (m.modified().unwrap(), std::fs::read(p).unwrap())
    };
    let before = stamp(&cat);
    std::thread::sleep(std::time::Duration::from_millis(50));
    let listed = archive.scan_tree(&Progress::default()).unwrap();
    assert_eq!(listed.patients().len(), 1);
    assert_eq!(stamp(&cat), before, "nothing changed, nothing written");

    // A file that is not DICOM is remembered as such: the catalog changes
    // once, and the next scan is quiet again.
    std::fs::write(root.join("notes.txt"), "a person's notes").unwrap();
    archive.refresh(&Progress::default()).unwrap();
    let after_note = stamp(&cat);
    assert_ne!(after_note, before, "a new file is caught");
    std::thread::sleep(std::time::Duration::from_millis(50));
    archive.refresh(&Progress::default()).unwrap();
    assert_eq!(stamp(&cat), after_note);
}

/// An import of several folders writes the catalog once, at the end; a
/// folder the caller made for the purpose has its files moved, not copied;
/// a filed file the catalog never heard of is put back into it.
#[test]
fn several_folders_are_filed_under_one_catalog_update() {
    let src = scratch("test_archive_importer_src");
    let n = gen_test_data::generate(&src, &GenParams::default(), &Progress::default()).unwrap();
    // Two scratch folders, the generated files split between them.
    let a = scratch("test_archive_importer_a");
    let b = scratch("test_archive_importer_b");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&src)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    for (i, f) in files.iter().enumerate() {
        let into = if i % 2 == 0 { &a } else { &b };
        std::fs::copy(f, into.join(f.file_name().unwrap())).unwrap();
    }

    let root = scratch("test_archive_importer_root");
    let archive = Archive::new(&root);
    let mut im = archive.importer().taking_files();
    let (sa, ha) = im.file_folder(&a, &Progress::default()).unwrap();
    let (sb, hb) = im.file_folder(&b, &Progress::default()).unwrap();
    assert_eq!(sa.stored + sb.stored, n);
    assert_eq!(ha.len() + hb.len(), n, "every header that arrived");
    assert!(
        archive.tree().patients().is_empty(),
        "the catalog learns of them at the end"
    );
    assert_eq!(
        std::fs::read_dir(&a).unwrap().count() + std::fs::read_dir(&b).unwrap().count(),
        0,
        "the scratch folders' files were moved"
    );
    im.finish().unwrap();
    let pts = archive.tree().patients();
    assert_eq!(pts.len(), 1);
    assert_eq!(pts[0].studies[0].files, n);

    // A file filed by an import that broke off before the catalog was
    // written: the next import of it puts it into the catalog.
    let uid = pts[0].studies[0].study_uid.clone();
    let one = archive.study_instances(&uid).unwrap()[0].clone();
    rust_dicom_station::archive::catalog::update(&root, |c| {
        for s in c.series.values_mut() {
            s.files.retain(|f| f.4 != one.sop_uid);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(archive.study_instances(&uid).unwrap().len(), n - 1);
    let again = archive.import(&src, &Progress::default()).unwrap();
    assert_eq!((again.stored, again.duplicates), (0, n));
    assert_eq!(
        archive.study_instances(&uid).unwrap().len(),
        n,
        "the stray file is listed again"
    );
    assert!(
        src.join(files[0].file_name().unwrap()).exists(),
        "a plain import copies"
    );
}
