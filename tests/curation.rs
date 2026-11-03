//! The curation tools on an import, end to end through `Core`: a study
//! in the archive, and a copy of it as a re-anonymizing export would give
//! it - every UID new, the patient identified (a made-up name and ID, never
//! a real one), a stray screenshot beside it. Found as pixel twins and RT
//! copies, its structure set relinked, its references remapped, its
//! identifiers scanned and pseudonymized, sorted into the archive's names,
//! merged, the catalog labelled; every write planned, applied and undone
//! back to the byte.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use dicom_core::{Tag, VR};
use dicom_dictionary_std::tags;
use rust_dicom_station::curation::raw::{self, encode_text, RawFile};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::mcp::config::{PacsConfig, PhiPolicy};
use rust_dicom_station::mcp::{Config, Core};
use rust_dicom_station::progress::Progress;
use serde_json::{json, Value};

fn call(core: &mut Core, tool: &str, args: Value) -> Value {
    match core.call_public(tool, args, &Progress::default()) {
        Ok(v) => {
            // The made-up identity never comes back.
            let s = v.to_string();
            assert!(
                !s.contains("Doe") && !s.contains("987654321"),
                "{tool} returned an identifier: {s}"
            );
            v
        }
        Err(e) => panic!("{tool} failed: {}", e.as_str()),
    }
}

fn fail(core: &mut Core, tool: &str, args: Value) -> String {
    match core.call_public(tool, args, &Progress::default()) {
        Ok(v) => panic!("{tool} should have failed, got {v}"),
        Err(e) => e.into_string(),
    }
}

fn copy_tree(from: &Path, to: &Path) {
    for e in walkdir::WalkDir::new(from).into_iter().flatten() {
        let dst = to.join(e.path().strip_prefix(from).unwrap());
        if e.file_type().is_dir() {
            std::fs::create_dir_all(&dst).unwrap();
        } else {
            std::fs::copy(e.path(), &dst).unwrap();
        }
    }
}

fn snapshot(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut v: Vec<_> = walkdir::WalkDir::new(root)
        .into_iter()
        .flatten()
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            (
                e.path().strip_prefix(root).unwrap().to_path_buf(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    v.sort();
    v
}

fn files(root: &Path) -> Vec<PathBuf> {
    rust_dicom_station::curation::files_under(root)
}

fn modality(p: &Path) -> String {
    RawFile::read(p)
        .ok()
        .and_then(|f| f.text(tags::MODALITY))
        .unwrap_or_default()
}

/// Re-anonymize a folder the way an export does: every UID but the
/// classes and transfer syntaxes replaced by a new one (the same new one
/// wherever the old one was), and the patient given an identity.
fn reanonymize(dir: &Path) {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for f in files(dir) {
        let r = RawFile::read(&f).unwrap();
        for (_, e) in r.walk() {
            if e.vr == VR::UI && e.items.is_none() {
                let v = e.text(&r.bytes);
                if !v.starts_with("1.2.840.10008") && !v.is_empty() {
                    let n = map.len() + 1;
                    map.entry(v).or_insert_with(|| format!("2.25.4711{n:06}"));
                }
            }
        }
    }
    for f in files(dir) {
        raw::rewrite_file(&f, &mut |at| match at.tag {
            tags::PATIENT_NAME => Some(encode_text(VR::PN, "Doe^John")),
            tags::PATIENT_ID => Some(encode_text(VR::LO, "987654321")),
            tags::PATIENT_BIRTH_DATE => Some(encode_text(VR::DA, "19560317")),
            _ if at.vr == VR::UI => map.get(&at.text()).map(|n| encode_text(VR::UI, n)),
            _ => None,
        })
        .unwrap();
    }
}

/// Set one top-level text element of a file.
fn set(path: &Path, tag: Tag, vr: VR, value: &str) {
    raw::rewrite_file(path, &mut |at| {
        (at.tag == tag && at.path.is_empty()).then(|| encode_text(vr, value))
    })
    .unwrap();
}

#[test]
fn an_import_is_deduplicated_relinked_pseudonymized_and_sorted() {
    let dir = common::target_dir("test_curation_import");
    let gen = dir.join("gen");
    gen_test_data::generate(
        &gen,
        &GenParams {
            extras: false,
            ..GenParams::default()
        },
        &Progress::default(),
    )
    .unwrap();
    let (pacs, import) = (dir.join("pacs"), dir.join("import"));
    let (quarantine, records) = (dir.join("quarantine"), dir.join("records"));
    // The archive: Trial / P01-001 / study / series / files.
    copy_tree(&gen, &pacs.join("Trial/P01-001/study/all"));
    // The import: patient folder 01-001, the same study re-anonymized.
    let export = import.join("01-001/full/study");
    copy_tree(&gen, &export);
    reanonymize(&export);
    std::fs::write(
        import.join("01-001/full/plan screenshot.png"),
        b"\x89PNG not dicom",
    )
    .unwrap();
    std::fs::create_dir_all(&quarantine).unwrap();
    std::fs::create_dir_all(&records).unwrap();
    std::fs::write(
        records.join("pacs-rules.toml"),
        "version = 1\n[[patient_folders]]\npattern = '(\\d{2}-\\d{3})'\ncode = 'P$1'\n",
    )
    .unwrap();
    let out = dir.join("out");
    std::fs::create_dir_all(&out).unwrap();
    let mut core = Core::new(Config {
        output_dir: Some(out),
        archive: pacs.display().to_string(),
        phi_policy: PhiPolicy::Redact,
        device: "cpu".into(),
        audit_log: false,
        pacs: PacsConfig {
            roots: vec![pacs.clone(), import.clone()],
            quarantine: Some(quarantine.clone()),
            records: Some(records.clone()),
            patient_codes: vec![r"^P01-\d{3}$".into()],
        },
        ..Config::default()
    });
    let c = &mut core;
    let archived = files(&pacs);
    let ct_series = RawFile::read(archived.iter().find(|f| modality(f) == "CT").unwrap())
        .unwrap()
        .text(tags::SERIES_INSTANCE_UID)
        .unwrap();
    let archive_study = RawFile::read(&archived[0])
        .unwrap()
        .text(tags::STUDY_INSTANCE_UID)
        .unwrap();
    let rs = files(&export)
        .into_iter()
        .find(|f| modality(f) == "RTSTRUCT")
        .unwrap();
    let pristine = snapshot(&import);

    // The CT is a pixel twin of the archive's, within one patient folder.
    let d = call(
        c,
        "find_duplicates",
        json!({"source": import, "against": pacs}),
    );
    // The CT, and the dose grid (pixels at a position too).
    assert_eq!(d["pixel_twin"], json!(2), "{d}");
    assert_eq!(d["new"], json!(2), "the plan and the structure set: {d}");
    assert_eq!(d["splits"], json!([]), "{d}");
    assert!(d["uid_map_entries"].as_u64().unwrap() > 10, "{d}");
    // Under another patient folder the same pixels are no twins.
    let other = import.join("01-002");
    copy_tree(&import.join("01-001"), &other);
    let d2 = call(
        c,
        "find_duplicates",
        json!({"source": other, "against": pacs}),
    );
    assert_eq!(d2["pixel_twin"], json!(0), "{d2}");
    std::fs::remove_dir_all(&other).unwrap();

    // The RT objects, through the twins' map: structure set first, then
    // the plan on it, then the dose on the plan.
    let o = call(
        c,
        "dedupe_rt_objects",
        json!({"source": import, "against": pacs}),
    );
    assert_eq!(o["copies"], json!(3), "{o}");
    assert_eq!(o["matched_by"]["mapped"], json!(3), "{o}");
    let map = o["uid_map"].as_str().unwrap().to_string();

    // Relink the structure set onto the archive's CT, applied and undone.
    let rl = call(
        c,
        "relink_structure_set",
        json!({"rtstruct": rs, "target_series": ct_series, "target": pacs, "reason": "test"}),
    );
    assert!(rl["images_by_pixels"].as_u64().unwrap() > 10, "{rl}");
    let id = rl["manifest"].as_str().unwrap().to_string();
    let applied = call(c, "apply_manifest", json!({"manifest": id}));
    assert_eq!(applied["failed"], json!([]), "{applied}");
    let after = RawFile::read(&rs).unwrap();
    let archive_sops: std::collections::BTreeSet<String> = archived
        .iter()
        .filter_map(|f| RawFile::read(f).ok()?.text(tags::SOP_INSTANCE_UID))
        .collect();
    let refs = rust_dicom_station::curation::remap::references(&after);
    assert!(
        refs.iter()
            .filter(|(k, _)| *k == rust_dicom_station::curation::remap::UidKind::Object)
            .all(|(_, u)| archive_sops.contains(u) || *u == ct_series),
        "every image it names is the archive's"
    );
    assert_eq!(after.text(tags::STUDY_INSTANCE_UID).unwrap(), archive_study);
    call(c, "undo_journal", json!({"manifest": id}));
    assert_eq!(snapshot(&import), pristine, "the relink undone to the byte");

    // The same through remap_references with the saved map: nothing old left.
    let rm = call(
        c,
        "remap_references",
        json!({"objects": [export], "map": map, "reason": "test"}),
    );
    let id = rm["manifest"].as_str().unwrap().to_string();
    assert_eq!(rm["unresolved"], json!([]), "{rm}");
    call(c, "apply_manifest", json!({"manifest": id}));
    call(c, "undo_journal", json!({"manifest": id}));
    assert_eq!(snapshot(&import), pristine);

    // Identifiers: where they are, then a pseudonymization that the
    // leftover scan stops while the name is in a description.
    let ph = call(c, "phi_report", json!({"path": import}));
    let rep = &ph["patients"][0]["report"];
    assert_eq!(ph["patients"][0]["patient"], json!("P01-001"), "{ph}");
    assert!(rep["full_birth_dates"].as_u64().unwrap() > 0, "{ph}");
    let ct0 = files(&export)
        .into_iter()
        .find(|f| modality(f) == "CT")
        .unwrap();
    set(&ct0, tags::STUDY_DESCRIPTION, VR::LO, "for Doe");
    let refused = fail(
        c,
        "plan_pseudonymize",
        json!({"path": import, "reason": "test"}),
    );
    assert!(refused.contains("0008,1030"), "{refused}");
    assert!(!refused.contains("Doe"), "{refused}");
    set(&ct0, tags::STUDY_DESCRIPTION, VR::LO, "Thorax");
    let ps = call(
        c,
        "plan_pseudonymize",
        json!({"path": import, "reason": "test"}),
    );
    let id = ps["manifest"].as_str().unwrap().to_string();
    let applied = call(c, "apply_manifest", json!({"manifest": id}));
    assert_eq!(applied["failed"], json!([]), "{applied}");
    let f = RawFile::read(&ct0).unwrap();
    assert_eq!(f.text(tags::PATIENT_NAME).as_deref(), Some("P01-001"));
    assert_eq!(
        f.text(tags::PATIENT_BIRTH_DATE).as_deref(),
        Some("19560101")
    );
    let pseudonymized = snapshot(&import);

    // Sorted into the archive's names, the screenshot reported.
    let rg = call(
        c,
        "plan_reorganize",
        json!({"folder": import, "cohort": "Trial", "reason": "test"}),
    );
    let sum = &rg["summary"];
    assert_eq!(sum["attachment_count"], json!(1), "{rg}");
    assert_eq!(sum["identity_rewrites"], json!(0), "already the code: {rg}");
    let folder = sum["study_folders"][0]["folder"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        folder.starts_with("Trial/P01-001/") && folder.contains("_Supine-"),
        "{folder}"
    );
    let id = rg["manifest"].as_str().unwrap().to_string();
    let applied = call(c, "apply_reorganize", json!({"manifest": id}));
    assert_eq!(applied["failed"], json!([]), "{applied}");
    let sorted: Vec<PathBuf> = files(&pacs.join(&folder));
    assert_eq!(sorted.len(), pristine.len() - 1, "every DICOM file moved");
    for f in &sorted {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let series = f
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert!(name.ends_with(".dcm") && name.contains('.'), "{name}");
        assert!(
            series.starts_with("Series-") && series.contains("_SeriesUID-"),
            "{series}"
        );
        assert!(f.as_os_str().len() < 260);
    }
    call(c, "undo_journal", json!({"manifest": id}));
    assert_eq!(
        snapshot(&import),
        pseudonymized,
        "the sort undone to the byte"
    );

    // A merge: the import's study into the archive's.
    let import_study = RawFile::read(&ct0)
        .unwrap()
        .text(tags::STUDY_INSTANCE_UID)
        .unwrap();
    let mg = call(
        c,
        "merge_studies",
        json!({"paths": [import], "from": [import_study], "into": archive_study, "reason": "test"}),
    );
    assert_eq!(mg["operations"], json!(pristine.len() - 1), "{mg}");

    // The catalog: made by a listing, the files in the patient's folder
    // that are not DICOM attached to them, then labelled.
    let shots = pacs.join("Trial/P01-001/TPS-plan-screenshots");
    std::fs::create_dir_all(&shots).unwrap();
    std::fs::write(shots.join("plan.png"), b"\x89PNG not dicom").unwrap();
    std::fs::write(shots.join("kv.raw"), vec![0u8; 256 * 256 * 2]).unwrap();
    let s = call(c, "list_archive", json!({"summary": true}));
    assert_eq!(s["cohorts"][0]["attachments"], json!(2), "{s}");
    let fl = call(c, "fill_study_labels", json!({"cohort": "Trial"}));
    assert_eq!(fl["labelled"], json!(1), "{fl}");
    let cat = rust_dicom_station::archive::catalog::load(&pacs);
    let label = &cat.studies[&archive_study].label;
    assert!(label.ends_with("CT, RTDOSE, RTPLAN, RTSTRUCT"), "{label}");
    call(
        c,
        "set_label",
        json!({"study": archive_study, "text": "by hand"}),
    );
    let again = call(c, "fill_study_labels", json!({"cohort": "Trial"}));
    assert_eq!(again["labelled"], json!(0), "a typed label stays");
    let rl = call(c, "catalog_relink", json!({"cohort": "Trial"}));
    assert_eq!(rl["result"]["patient_folders"], json!(1), "{rl}");
    assert!(records.join("catalog-backups").read_dir().unwrap().count() >= 3);

    let _ = std::fs::remove_dir_all(&dir);
}

/// The 4D sets of the archive through the tools: listed with every study
/// and filtered on, a study's groups member by member, one group opened
/// alone under the id the catalog gives it, a group made by hand and one
/// dissolved - the made-up patient never named.
#[test]
fn the_archives_4d_sets_are_listed_opened_and_decided() {
    use common::fourd::{philips_4d, STUDY_4D, STUDY_SCOUT};
    let dir = common::target_dir("test_curation_fourd");
    let (pacs, records) = (dir.join("pacs"), dir.join("records"));
    philips_4d(&pacs);
    std::fs::create_dir_all(&records).unwrap();
    let mut core = Core::new(Config {
        output_dir: Some(dir.join("out")),
        archive: pacs.display().to_string(),
        phi_policy: PhiPolicy::Redact,
        device: "cpu".into(),
        audit_log: false,
        pacs: PacsConfig {
            roots: vec![pacs.clone()],
            records: Some(records.clone()),
            ..Default::default()
        },
        ..Config::default()
    });
    let c = &mut core;
    let named = |v: &Value| v.to_string().contains("Phantom") || v.to_string().contains("P4D-01");

    // One line per study with a 4DCT; none probable; the scout's study has none.
    let s = call(
        c,
        "list_archive",
        json!({"summary": true, "filter": {"has_4d": true}}),
    );
    assert_eq!(s["studies"], json!(1), "{s}");
    assert_eq!(s["listing"][0]["study"], json!(STUDY_4D));
    assert_eq!(s["listing"][0]["fourd_scans"], json!(1));
    let s = call(
        c,
        "list_archive",
        json!({"summary": true, "filter": {"has_4d": "probable"}}),
    );
    assert_eq!(s["studies"], json!(0), "{s}");
    let s = call(
        c,
        "list_archive",
        json!({"summary": true, "filter": {"has_4d": false}}),
    );
    assert_eq!(s["listing"][0]["study"], json!(STUDY_SCOUT), "{s}");
    let all = call(c, "list_archive", json!({}));
    assert!(!named(&all), "{all}");
    let studies = all["listing"][0]["studies"].as_array().unwrap();
    let st = studies
        .iter()
        .find(|s| s["study"] == json!(STUDY_4D))
        .unwrap();
    let group = st["fourd"][0]["group"].as_str().unwrap().to_string();
    assert_eq!(st["fourd"][0]["evidence"], json!("same-acquisition"));
    assert!(
        st["series"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["fourd"] == json!("~40%")),
        "{st}"
    );
    let scout = studies
        .iter()
        .find(|s| s["study"] == json!(STUDY_SCOUT))
        .unwrap();
    assert_eq!(scout["fourd_orphans"][0]["what"], json!("scout"), "{scout}");

    // Member by member, from the catalog alone.
    let g = call(c, "list_archive_4d_groups", json!({"study": STUDY_4D}));
    assert_eq!(
        g["groups"][0]["members"].as_array().unwrap().len(),
        12,
        "{g}"
    );
    assert_eq!(
        g["groups"][0]["structures"].as_array().unwrap().len(),
        1,
        "{g}"
    );

    // Opened alone: its phases, its reconstruction, the structure set on
    // one phase; the dataset lists the group under the same id.
    let o = call(
        c,
        "open_archive_study",
        json!({"study": STUDY_4D, "fourd_group": group}),
    );
    assert!(!named(&o), "{o}");
    let ds = o["dataset"].as_str().unwrap().to_string();
    let l = call(c, "list_4d_groups", json!({"dataset": ds}));
    assert_eq!(l["groups"][0]["group"], json!(group), "{l}");
    assert_eq!(
        l["groups"][0]["members"].as_array().unwrap().len(),
        11,
        "{l}"
    );
    let e = fail(
        c,
        "open_archive_study",
        json!({"study": STUDY_4D, "fourd_group": "nope"}),
    );
    assert!(e.contains("no 4D group"), "{e}");

    // Dissolved, it is listed no more; reset, it is back.
    call(c, "confirm_4d", json!({"group": group, "yes": false}));
    let g = call(c, "list_archive_4d_groups", json!({"study": STUDY_4D}));
    assert_eq!(g["groups"], json!([]), "{g}");
    call(
        c,
        "confirm_4d",
        json!({"group": group, "yes": true, "reset": true}),
    );
    // Made by hand of three phases.
    let m = call(
        c,
        "set_4d_group",
        json!({"study": STUDY_4D, "members": [
            "1.2.826.0.1.4.200",
            {"series": "1.2.826.0.1.4.201", "phase": "50%"},
            "1.2.826.0.1.4.202"
        ], "name": "Three phases"}),
    );
    assert_eq!(m["group"], json!(group), "{m}");
    assert_eq!(m["members"], json!(3), "{m}");
    let g = call(c, "list_archive_4d_groups", json!({"study": STUDY_4D}));
    assert_eq!(g["groups"][0]["custom"], json!(true), "{g}");
    assert_eq!(g["groups"][0]["members"][1]["phase"], json!("50%"), "{g}");
    assert!(records.join("catalog-backups").read_dir().unwrap().count() >= 3);
    let _ = std::fs::remove_dir_all(&dir);
}
