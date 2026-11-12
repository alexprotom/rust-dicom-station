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

/// A series carries the date and time it was made (the earliest of its
/// files), and a catalog written before the series had one gets it at the
/// next scan, with the names given in the window kept.
#[test]
fn a_series_knows_when_it_was_made_and_an_older_catalog_learns_it() {
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_dictionary_std::tags;
    let root = scratch("test_archive_series_time");
    for (k, time) in ["103712", "103701"].into_iter().enumerate() {
        let path = root.join(format!("P/CT/{k}.dcm"));
        tiny(
            &path,
            ("P9", "TEST^ONE"),
            "1.9",
            "1.9.1",
            &format!("1.9.1.{k}"),
            "CT",
        );
        let mut obj = dicom_object::open_file(&path).unwrap();
        obj.put(DataElement::new(
            tags::SERIES_DATE,
            VR::DA,
            PrimitiveValue::from("20240917"),
        ));
        obj.put(DataElement::new(
            tags::SERIES_TIME,
            VR::TM,
            PrimitiveValue::from(time),
        ));
        obj.write_to_file(&path).unwrap();
    }
    let archive = Archive::new(&root);
    let when = |a: &Archive| {
        a.scan_tree(&Progress::default()).unwrap().patients()[0].studies[0].series[0]
            .datetime
            .clone()
    };
    assert_eq!(when(&archive), "20240917103701", "the earliest file's");

    // The catalog as version 1 wrote it: no time, a name given by hand.
    let cat = archive.catalog_path();
    let text = std::fs::read_to_string(&cat).unwrap();
    assert!(text.contains("\"version\": 3"), "{text}");
    let old = text
        .replace("\"version\": 3", "\"version\": 1")
        .replace("\"datetime\":\"20240917103701\"", "\"label\":\"Daily\"")
        .replacen("\"label\":\"\",", "", 1);
    assert!(
        old.contains("\"label\":\"Daily\"") && !old.contains("103701"),
        "{old}"
    );
    std::fs::write(&cat, old).unwrap();
    let series =
        &archive.scan_tree(&Progress::default()).unwrap().patients()[0].studies[0].series[0];
    assert_eq!(series.datetime, "20240917103701", "read again once");
    assert_eq!(series.label, "Daily", "the person's name for it is kept");
    assert!(std::fs::read_to_string(&cat)
        .unwrap()
        .contains("\"version\": 3"));
}

/// A file rewritten in place (its study and patient changed, as a merge or
/// a pseudonymization does) moves in the catalog with its next scan: the
/// series goes to the study its files now name, the study to the patient,
/// the study that is left empty goes. A study's date is its earliest
/// file's, however late that file arrives.
#[test]
fn a_rescan_follows_files_rewritten_in_place_and_dates_a_study_by_its_earliest_file() {
    use dicom_dictionary_std::tags;
    use rust_dicom_station::curation::raw::{encode_text, rewrite_file};
    let root = scratch("test_archive_rekey");
    for k in 0..2 {
        tiny(
            &root.join(format!("P/CT/{k}.dcm")),
            ("P9", "TEST^ONE"),
            "1.9",
            "1.9.1",
            &format!("1.9.1.{k}"),
            "CT",
        );
    }
    let archive = Archive::new(&root);
    let pts = archive.scan_tree(&Progress::default()).unwrap().patients();
    assert_eq!((pts.len(), pts[0].key.as_str()), (1, "P9"));
    // The scan reads a file again when its size or time changed: a UID of
    // another length changes the size whatever the clock says.
    for k in 0..2 {
        let n = rewrite_file(&root.join(format!("P/CT/{k}.dcm")), &mut |at| {
            if !at.path.is_empty() {
                return None;
            }
            match at.tag {
                tags::STUDY_INSTANCE_UID => Some(encode_text(at.vr, "1.9.77")),
                tags::PATIENT_ID => Some(encode_text(at.vr, "P10")),
                _ => None,
            }
        })
        .unwrap();
        assert_eq!(n, 2);
    }
    // An earlier session of the same course study, with its description.
    tiny(
        &root.join("P/CT/early.dcm"),
        ("P10", "TEST^ONE"),
        "1.9.77",
        "1.9.2",
        "1.9.2.0",
        "CT",
    );
    {
        use dicom_core::{DataElement, PrimitiveValue, VR};
        let path = root.join("P/CT/early.dcm");
        let mut obj = dicom_object::open_file(&path).unwrap();
        obj.put(DataElement::new(
            tags::STUDY_DATE,
            VR::DA,
            PrimitiveValue::from("20251201"),
        ));
        obj.put(DataElement::new(
            tags::STUDY_DESCRIPTION,
            VR::LO,
            PrimitiveValue::from("Procedure #1"),
        ));
        obj.write_to_file(&path).unwrap();
    }
    let pts = archive.scan_tree(&Progress::default()).unwrap().patients();
    assert_eq!(pts.len(), 1, "the old patient is gone with its study");
    assert_eq!(pts[0].key, "P10");
    assert_eq!(pts[0].studies.len(), 1);
    let st = &pts[0].studies[0];
    assert_eq!(st.study_uid, "1.9.77");
    assert_eq!(st.series.len(), 2, "both series under the study they name");
    assert_eq!(st.date, "20251201", "the earliest session's date");
    assert_eq!(st.description, "Procedure #1");
}

/// A registration between frames of reference no loaded series has (a
/// setup system's room frames) loads all the same, and the load says so
/// once; a file with a private element whose length runs past the end of
/// the file loads past it.
#[test]
fn a_registration_to_unknown_frames_and_a_broken_private_element_still_load() {
    use dicom_dictionary_std::tags;
    use rust_dicom_station::curation::raw::{encode_text, rewrite_file, RawFile};
    let root = scratch("test_archive_unknown_frames");
    gen_test_data::generate(&root, &GenParams::default(), &Progress::default()).unwrap();
    let files: Vec<std::path::PathBuf> = rust_dicom_station::curation::files_under(&root);
    let reg = files
        .iter()
        .find(|f| {
            RawFile::read(f)
                .ok()
                .and_then(|r| r.text(tags::MODALITY))
                .as_deref()
                == Some("REG")
        })
        .expect("the generator writes a REG");
    rewrite_file(reg, &mut |at| {
        (at.tag == tags::FRAME_OF_REFERENCE_UID).then(|| encode_text(at.vr, "2.25.424242"))
    })
    .unwrap();
    // A CT slice whose last element is a private one claiming 5000 bytes.
    let ct = files
        .iter()
        .find(|f| f.file_name().unwrap().to_string_lossy().starts_with("CT_"))
        .unwrap();
    let mut bytes = std::fs::read(ct).unwrap();
    let r = RawFile::parse(bytes.clone()).unwrap();
    let explicit = r.elements[0].explicit;
    if explicit {
        bytes.extend_from_slice(&[0xF1, 0x01, 0x26, 0x10, b'U', b'N', 0, 0]);
    } else {
        bytes.extend_from_slice(&[0xF1, 0x01, 0x26, 0x10]);
    }
    bytes.extend_from_slice(&5000u32.to_le_bytes());
    bytes.extend_from_slice(&[1, 2, 3, 4]);
    std::fs::write(ct, &bytes).unwrap();
    let study = loader::load_directory(&root, &Progress::default()).unwrap();
    assert_eq!(study.registrations.len(), 1, "{:?}", study.warnings);
    assert!(
        study
            .warnings
            .iter()
            .any(|w| w.contains("frames of reference no loaded series has")),
        "{:?}",
        study.warnings
    );
    let ct_files: usize = study.series.iter().map(|s| s.files.len()).sum();
    assert_eq!(
        ct_files, 40,
        "the slice with the broken element is read: {:?}",
        study.warnings
    );
}

/// A staging folder's copies (the same UIDs under the source's own patient
/// codes) are never indexed; once copies filed under a series are gone,
/// the next scan reads the series again from its files; a study whose
/// files all lie in one patient's folder is that patient's.
#[test]
fn staging_copies_stay_out_and_a_study_follows_its_files_and_its_folder() {
    use rust_dicom_station::archive::catalog;
    let root = scratch("test_archive_staging");
    let p2 = ("P01-002", "P01-002");
    for k in 0..2 {
        tiny(
            &root.join(format!("Trial/P01-002/S1/CT/{k}.dcm")),
            p2,
            "7.1",
            "7.1.1",
            &format!("7.1.1.{k}"),
            "CT",
        );
        // The same files in an import still waiting to be sorted.
        tiny(
            &root.join(format!("to_be_sorted/Liver/{k}.dcm")),
            ("Liver", "Liver"),
            "7.1",
            "7.1.1",
            &format!("7.1.1.{k}"),
            "CT",
        );
    }
    tiny(
        &root.join("Trial/P01-003/S1/CT/0.dcm"),
        ("P01-003", "P01-003"),
        "7.2",
        "7.2.1",
        "7.2.1.0",
        "CT",
    );
    // A study sorted into P01-003's folder whose files still carry the
    // pseudonym of the export it came from.
    tiny(
        &root.join("Trial/P01-003/S2/CT/0.dcm"),
        ("Pancreas", "Pancreas"),
        "7.3",
        "7.3.1",
        "7.3.1.0",
        "CT",
    );
    let archive = Archive::new(&root);
    let keys = |a: &Archive| -> Vec<(String, usize)> {
        a.scan_tree(&Progress::default())
            .unwrap()
            .patients()
            .iter()
            .map(|p| (p.key.clone(), p.studies.len()))
            .collect()
    };
    assert_eq!(
        keys(&archive),
        [("P01-002".to_string(), 1), ("P01-003".to_string(), 2)],
        "no pseudonym patient, the sorted study under its folder's patient"
    );
    let c = catalog::load(&root);
    assert_eq!(c.series["7.1.1"].files.len(), 2, "the copies are not filed");

    // An import folder of another name is filed until the person says to
    // ignore it; a copy filed under a sorted series takes its study along.
    tiny(
        &root.join("incoming/a.dcm"),
        ("Liver", "Liver"),
        "7.1",
        "7.1.1",
        "7.1.1.9",
        "CT",
    );
    archive.refresh(&Progress::default()).unwrap();
    assert_eq!(catalog::load(&root).series["7.1.1"].files.len(), 3);
    catalog::update(&root, |c| {
        c.ignore.push("Incom*".into());
        Ok(())
    })
    .unwrap();
    assert_eq!(
        keys(&archive),
        [("P01-002".to_string(), 1), ("P01-003".to_string(), 2)],
        "the series is read again from the files left, its study P01-002's"
    );
    let c = catalog::load(&root);
    assert_eq!(c.series["7.1.1"].files.len(), 2);
    assert_eq!(c.studies["7.1"].patient, "P01-002");
    assert_eq!(c.ignore, ["Incom*"], "the person's patterns are kept");
    assert!(catalog::is_ignored_folder("Pryanick-PACS-duplicates", &[]));
    assert!(!catalog::is_ignored_folder("P01-002", &c.ignore));
}

/// One Patient ID in two cohorts, given there to two different people,
/// is found by the birth dates and sexes of their files.
#[test]
fn one_code_given_to_two_people_in_two_cohorts_is_found() {
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_dictionary_std::tags;
    use rust_dicom_station::archive::catalog;
    use rust_dicom_station::curation::catalog_fix::identity_clashes;
    let root = scratch("test_archive_identity");
    let put = |rel: &str, pid: &str, study: &str, birth: &str, sex: &str| {
        let path = root.join(rel);
        tiny(
            &path,
            (pid, pid),
            study,
            &format!("{study}.1"),
            &format!("{study}.1.1"),
            "CT",
        );
        let mut obj = dicom_object::open_file(&path).unwrap();
        obj.put(DataElement::new(
            tags::PATIENT_BIRTH_DATE,
            VR::DA,
            PrimitiveValue::from(birth),
        ));
        obj.put(DataElement::new(
            tags::PATIENT_SEX,
            VR::CS,
            PrimitiveValue::from(sex),
        ));
        obj.write_to_file(&path).unwrap();
    };
    put("Trial/P01-002/A/0.dcm", "P01-002", "8.1", "19560101", "F");
    put("Trial/P01-002/B/0.dcm", "P01-002", "8.2", "19560101", "F");
    put("Trial/P01-005/A/0.dcm", "P01-005", "8.3", "19700101", "M");
    put("Rambam/P01-002/A/0.dcm", "P01-002", "8.4", "19480101", "M");
    put("Rambam/P01-007/A/0.dcm", "P01-007", "8.5", "19500101", "M");
    // The same person in two cohorts is no clash.
    put("Rambam/P01-005/A/0.dcm", "P01-005", "8.6", "19700101", "M");
    Archive::new(&root).refresh(&Progress::default()).unwrap();
    let clashes = identity_clashes(&catalog::load(&root));
    assert_eq!(clashes.len(), 1, "{clashes:?}");
    assert_eq!(clashes[0].patient, "P01-002");
    let parts: Vec<(&str, &str, usize, usize)> = clashes[0]
        .parts
        .iter()
        .map(|p| (p.cohort.as_str(), p.folder.as_str(), p.studies, p.variants))
        .collect();
    assert_eq!(
        parts,
        [
            ("Rambam", "Rambam/P01-002", 1, 1),
            ("Trial", "Trial/P01-002", 2, 1)
        ]
    );
}

/// A segmentation drawn on a study kept in a cohort folder of a second
/// storage folder is saved into that folder, beside the study, not into
/// the main one; the catalog learns it, and what the PACS holds already
/// is told by its SOP Instance UID.
#[test]
fn a_new_segmentation_is_saved_beside_its_study_in_the_pacs_layout() {
    use rust_dicom_station::archive::save::save_derived;
    use rust_dicom_station::dicom_export::DerivedKind;

    let main = scratch("test_archive_save_main");
    let second = scratch("test_archive_save_second");
    let study_dir = second.join("Trial A").join("PHANTOM");
    gen_test_data::generate(&study_dir, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    std::fs::create_dir_all(&main).unwrap();
    let archive = Archive::with_roots(vec![main.clone(), second.clone()]);
    archive
        .refresh(&Progress::default())
        .expect("scan succeeds");

    let mut study =
        loader::load_directory(&study_dir, &Progress::default()).expect("the study loads");
    let study_uid = study.series[study.active_series].study_uid.clone();
    let home = archive
        .study_home(&study_uid, "")
        .expect("the PACS holds the study");
    assert!(
        home.starts_with(&study_dir),
        "the study's own folder: {home:?}"
    );
    assert!(archive.study_home("1.2.3.4.5.404", "").is_none());

    // What came from the PACS is known as such.
    let loaded: Vec<&str> = study
        .structure_sets
        .iter()
        .map(|s| s.sop_instance_uid.as_str())
        .collect();
    assert!(!loaded.is_empty());
    assert_eq!(
        archive.held_sops(loaded.iter().copied()).len(),
        loaded.len()
    );

    let dims = study.volume.dims;
    let mut ser = SegSeries::new(
        "Saved here".into(),
        study.volume.grid(),
        study.series[study.active_series].uid.clone(),
        study_uid.clone(),
    );
    ser.segs.push(Segmentation::from_mask(
        "Ball".into(),
        [40, 200, 40],
        dims,
        ball(dims, 6.0),
    ));
    study.seg_series.push(ser);
    let new_seg = study.seg_series.len() - 1;
    let before = archive.tree().studies();

    let params = ExportParams::for_study(&study);
    let saved = save_derived(&archive, &study, &params, &Progress::default(), &|k, i| {
        k == DerivedKind::Segmentation && i == new_seg
    })
    .expect("the save succeeds");
    assert_eq!(saved.summary.stored, 1, "the new series only");
    assert_eq!(saved.objects.len(), 1);
    let o = &saved.objects[0];
    assert_eq!((o.kind, o.index), (DerivedKind::Segmentation, new_seg));
    assert_eq!(o.path.parent(), Some(home.as_path()), "beside the study");
    assert!(o.path.is_file());
    // The main folder holds the catalog and nothing else.
    fn dicoms(d: &std::path::Path) -> usize {
        std::fs::read_dir(d)
            .map(|es| {
                es.flatten()
                    .map(|e| {
                        let p = e.path();
                        if p.is_dir() {
                            dicoms(&p)
                        } else {
                            usize::from(p.extension().is_some_and(|x| x == "dcm"))
                        }
                    })
                    .sum()
            })
            .unwrap_or(0)
    }
    assert_eq!(dicoms(&main), 0, "nothing filed into the main folder");

    // The catalog knows it: the same study, one modality more.
    let tree = archive.tree();
    assert_eq!(tree.studies(), before, "no new study");
    let st = tree
        .cohorts
        .iter()
        .flat_map(|c| c.patients.iter())
        .flat_map(|p| p.studies.iter())
        .find(|s| s.study_uid == study_uid)
        .expect("the study is listed");
    assert!(
        st.modalities.iter().any(|m| m == "SEG"),
        "{:?}",
        st.modalities
    );
    assert!(archive
        .held_sops([o.sop_instance_uid.as_str()])
        .contains(&o.sop_instance_uid));
}

/// The names of the cohorts that list patient `key`, sorted.
fn cohorts_of(archive: &Archive, key: &str) -> Vec<String> {
    let mut out: Vec<String> = archive
        .tree()
        .cohorts
        .iter()
        .filter(|c| c.patients.iter().any(|p| p.key == key))
        .map(|c| c.name.clone())
        .collect();
    out.sort();
    out
}

/// Where *Save to the PACS* puts a new object, and what its lookup says:
/// beside the image series the object was drawn on (in the patient's
/// folder when that series has a folder of its own), never loose in a
/// storage or a cohort folder, also when a study's series lie under two
/// cohort folders; and the one-pass lookup the window runs on a job says
/// what the two separate calls say.
#[test]
fn a_new_object_is_filed_beside_its_series_and_never_loose_in_a_cohort() {
    let root = scratch("test_archive_save_home");
    let p01 = ("P01", "AB");
    // One study, its CT in one cohort and its MR in another.
    tiny(
        &root.join("Trial/P01/CT/c1"),
        p01,
        "7.1",
        "7.1.1",
        "7.1.1.1",
        "CT",
    );
    tiny(
        &root.join("Trial/P01/CT/c2"),
        p01,
        "7.1",
        "7.1.1",
        "7.1.1.2",
        "CT",
    );
    tiny(
        &root.join("Imports/P01/MR/m1"),
        p01,
        "7.1",
        "7.1.2",
        "7.1.2.1",
        "MR",
    );
    // A study of one series.
    tiny(
        &root.join("Trial/P02/CT/c3"),
        ("P02", "CD"),
        "7.2",
        "7.2.1",
        "7.2.1.1",
        "CT",
    );
    let archive = Archive::new(&root);
    archive.refresh(&Progress::default()).expect("scan");
    let trial_p01 = root.join("Trial").join("P01");
    let trial_p02 = root.join("Trial").join("P02");

    // The rule.
    assert_eq!(archive.study_home("7.1", "7.1.1"), Some(trial_p01.clone()));
    assert_eq!(
        archive.study_home("7.1", ""),
        Some(trial_p01.clone()),
        "no series named: beside the largest, not the storage folder"
    );
    assert_eq!(archive.study_home("7.2", "7.2.1"), Some(trial_p02.clone()));
    assert_eq!(
        archive.study_home("7.2", ""),
        Some(trial_p02.clone()),
        "out of the series folder"
    );
    assert_eq!(archive.study_home("9.9", ""), None);

    // The lookup in one pass says what the two calls say.
    let keys: Vec<(String, String)> =
        [("7.1", "7.1.1"), ("7.1", ""), ("7.2", "7.2.1"), ("9.9", "")]
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
    let sops = vec!["7.1.1.1".to_string(), "8.8".to_string()];
    let t = archive.save_targets(&keys, &sops);
    assert!(t.catalog);
    for (study, series) in &keys {
        assert_eq!(
            t.homes[&(study.clone(), series.clone())],
            archive.study_home(study, series),
            "{study} {series}"
        );
    }
    assert_eq!(t.held, archive.held_sops(sops.iter().map(String::as_str)));
    assert_eq!(t.held.len(), 1);
    // Without a catalog nothing is read, and nothing is scanned.
    let unscanned = scratch("test_archive_save_unscanned");
    tiny(
        &unscanned.join("Trial/P09/x"),
        ("P09", "EF"),
        "7.9",
        "7.9.1",
        "7.9.1.1",
        "CT",
    );
    let none = Archive::new(&unscanned).save_targets(&keys, &sops);
    assert!(!none.catalog && none.homes.is_empty());
    assert!(
        !rust_dicom_station::archive::catalog::exists(&unscanned),
        "the lookup made no catalog"
    );

    let before = cohorts_of(&archive, "P01");
    // Two new objects, one per study, filed the way a save files them.
    let src = root.parent().unwrap().join("test_archive_save_home_src");
    let _ = std::fs::remove_dir_all(&src);
    let derived = |name: &str, patient: (&str, &str), study: &str, sop: &str, modality: &str| {
        let path = src.join(name);
        tiny(&path, patient, study, &format!("{sop}0"), sop, modality);
        path
    };
    use rust_dicom_station::archive::save::file_written;
    use rust_dicom_station::dicom_export::{DerivedFile, DerivedKind};
    let a = derived("a.dcm", p01, "7.1", "7.1.9.1", "RTSTRUCT");
    let b = derived("b.dcm", ("P02", "CD"), "7.2", "7.2.9.1", "SEG");
    let files = vec![
        (
            DerivedFile {
                kind: DerivedKind::StructureSet,
                index: 0,
                path: a.clone(),
            },
            "7.1.1".to_string(),
        ),
        (
            DerivedFile {
                kind: DerivedKind::Segmentation,
                index: 0,
                path: b.clone(),
            },
            "7.2.1".to_string(),
        ),
    ];
    // The second cannot be put in its place: a folder where its file is
    // written first.
    let block = trial_p02.join("7.2.9.1.dcm.tmp");
    std::fs::create_dir_all(&block).unwrap();
    let saved = file_written(&archive, &src, &files, &Progress::default())
        .expect("what was filed is no error");
    assert_eq!(saved.objects.len(), 1, "the first made it");
    assert!(saved.failed.is_some(), "the rest is said to have failed");
    let o = &saved.objects[0];
    assert_eq!(o.sop_instance_uid, "7.1.9.1");
    assert_eq!(o.path.parent(), Some(trial_p01.as_path()), "beside its CT");
    assert!(
        archive.held_sops(["7.1.9.1"]).contains("7.1.9.1"),
        "catalogued"
    );
    assert_eq!(
        cohorts_of(&archive, "P01"),
        before,
        "the cohorts as they were"
    );

    // The retry, with the first one's identity taken over: it is the PACS's
    // own now, so only the second is stored.
    std::fs::remove_dir_all(&block).unwrap();
    derived("a.dcm", p01, "7.1", "7.1.9.1", "RTSTRUCT");
    let again = file_written(&archive, &src, &files, &Progress::default()).expect("the retry");
    assert!(again.failed.is_none());
    assert_eq!(again.summary.stored, 1, "{:?}", again.summary);
    assert_eq!(again.summary.duplicates, 1);
    let b_saved = again
        .objects
        .iter()
        .find(|o| o.sop_instance_uid == "7.2.9.1")
        .expect("the second");
    assert_eq!(b_saved.path.parent(), Some(trial_p02.as_path()));
    assert_eq!(cohorts_of(&archive, "P01"), before);
    let _ = std::fs::remove_dir_all(&src);
}
