//! The 4D sets of a PACS found from headers alone (`fourd::scan`), and
//! kept in the catalog (`archive::catalog`): what the PACS window and
//! `list_archive` show of a 4DCT before it is loaded; and the person's
//! decisions about a PACS taken to another station (`archive::tags`).
//!
//! Everything runs on synthetic files. The golden test is `#[ignore]`d: it
//! replays the header dump of a real trial archive and compares with the
//! sets proven on it, so it needs that folder,
//! `RDS_4D_GOLDEN=<folder with hdr.json and groups2.json>`:
//!
//! ```text
//! RDS_4D_GOLDEN=D:/records/4DCT-20261102 cargo test --release --test fourd_scan -- --ignored
//! ```

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::fourd::{philips_4d, STUDY_4D, STUDY_SCOUT};

use rust_dicom_station::archive;
use rust_dicom_station::fourd::scan::{self, Evidence, Input, Kind, SeriesHead, Z};

/// One series of the header dump `hdr.py` wrote.
struct Dumped {
    uid: String,
    study: String,
    modality: String,
    number: Option<i64>,
    description: String,
    head: SeriesHead,
    z: Option<Z>,
}

fn text(v: &serde_json::Value, k: &str) -> String {
    match &v[k] {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(a) => a
            .iter()
            .map(|x| x.as_str().unwrap_or_default().to_string())
            .collect::<Vec<_>>()
            .join("\\"),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn read_dump(path: &std::path::Path) -> Vec<Dumped> {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("hdr.json")).expect("json");
    v.as_array()
        .expect("a list")
        .iter()
        .map(|x| {
            let mut private = BTreeMap::new();
            if let Some(m) = x["private"].as_object() {
                for (k, val) in m {
                    if scan::PRIVATE_TAGS
                        .iter()
                        .any(|(g, e)| scan::tag_key(*g, *e) == *k)
                    {
                        private.insert(k.clone(), val.as_str().unwrap_or_default().to_string());
                    }
                }
            }
            // z: [first, last, files, distinct], in millimetres to 0.01.
            let z = x["z"].as_array().map(|a| Z {
                min: (a[0].as_f64().unwrap() * 100.0).round() as i64,
                max: (a[1].as_f64().unwrap() * 100.0).round() as i64,
                n: a[2].as_u64().unwrap() as usize,
            });
            Dumped {
                uid: text(x, "series"),
                study: text(x, "cat_study"),
                modality: text(x, "modality"),
                number: text(x, "SeriesNumber").trim().parse().ok(),
                description: text(x, "SeriesDescription"),
                head: SeriesHead {
                    image_type: text(x, "ImageType"),
                    model: text(x, "ManufacturerModelName"),
                    protocol: text(x, "ProtocolName"),
                    comments: text(x, "ImageComments"),
                    derivation: text(x, "DerivationDescription"),
                    acq_date: text(x, "AcquisitionDate"),
                    acq_time: text(x, "AcquisitionTime"),
                    series_date: text(x, "SeriesDate"),
                    rows: text(x, "Rows").parse().unwrap_or(0),
                    nominal: text(x, "NominalPercentageOfRespiratoryPhase"),
                    temporal: text(x, "TemporalPositionIdentifier"),
                    private,
                    refs: Vec::new(),
                },
                z,
            }
        })
        .collect()
}

/// The 35 sets of the trial, as proven by hand: the same series, roles,
/// labels and evidence. The two the person confirmed are `geometry-only`
/// here (detection alone calls them probable).
#[test]
#[ignore]
fn the_trial_archive_gives_the_sets_proven_on_it() {
    let Ok(dir) = std::env::var("RDS_4D_GOLDEN") else {
        eprintln!("RDS_4D_GOLDEN not set");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let dump = read_dump(&dir.join("hdr.json"));
    let inputs: Vec<Input> = dump
        .iter()
        .map(|d| Input {
            uid: &d.uid,
            study: &d.study,
            modality: &d.modality,
            number: d.number,
            description: &d.description,
            head: Some(&d.head),
            temporal: None,
            z: d.z,
            image: matches!(d.modality.as_str(), "CT" | "MR"),
        })
        .collect();
    let t = std::time::Instant::now();
    let found = scan::detect(&inputs);
    let took = t.elapsed();

    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("groups2.json")).expect("groups2.json"),
    )
    .unwrap();
    // A set as (study, block, EFOV, members as (series, role, label)).
    type Key = (String, i64, bool, Vec<(String, String, String)>);
    let want: BTreeMap<Key, String> = golden
        .as_array()
        .unwrap()
        .iter()
        .map(|g| {
            let members = g["members"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| (text(m, "series"), text(m, "role"), text(m, "label")))
                .collect();
            let source = text(g, "source");
            (
                (
                    text(g, "study"),
                    g["block"].as_i64().unwrap(),
                    text(g, "variant") == "EFOV",
                    members,
                ),
                source.split(',').next().unwrap().to_string(),
            )
        })
        .collect();
    let got: BTreeMap<Key, String> = found
        .iter()
        .filter(|f| f.evidence != Evidence::Description)
        .map(|f| {
            let members = f
                .members
                .iter()
                .filter(|m| m.kind != Kind::Trace)
                .map(|m| {
                    let label = if m.inferred {
                        format!("{}*", m.label)
                    } else {
                        m.label.clone()
                    };
                    (m.series.clone(), m.kind.word().to_string(), label)
                })
                .collect();
            (
                (f.study.clone(), f.block, f.efov, members),
                f.evidence.word().to_string(),
            )
        })
        .collect();
    let missing: Vec<&Key> = want.keys().filter(|k| !got.contains_key(*k)).collect();
    let extra: Vec<&Key> = got.keys().filter(|k| !want.contains_key(*k)).collect();
    for k in &missing {
        eprintln!(
            "missing: {} b{} efov={} {:?}",
            k.0,
            k.1,
            k.2,
            k.3.iter().map(|m| (&m.1, &m.2)).collect::<Vec<_>>()
        );
    }
    for k in &extra {
        eprintln!(
            "extra: {} b{} efov={} {:?}",
            k.0,
            k.1,
            k.2,
            k.3.iter().map(|m| (&m.1, &m.2)).collect::<Vec<_>>()
        );
    }
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{} missing, {} extra",
        missing.len(),
        extra.len()
    );
    for (k, src) in &want {
        assert_eq!(&got[k], src, "evidence of {} b{}", k.0, k.1);
    }
    let mut evidence: BTreeMap<String, usize> = BTreeMap::new();
    for src in got.values() {
        *evidence.entry(src.clone()).or_default() += 1;
    }
    eprintln!("{evidence:?}");
    let studies: BTreeSet<&str> = found.iter().map(|f| f.study.as_str()).collect();
    let scans: BTreeSet<(&str, usize)> = found.iter().map(|f| (f.study.as_str(), f.scan)).collect();
    assert_eq!((got.len(), scans.len(), studies.len()), (35, 30, 25));
    assert!(took.as_secs_f64() < 1.0, "{took:?}");
    // Nothing found by the description rule that the header rules missed.
    let by_description = found
        .iter()
        .filter(|f| f.evidence == Evidence::Description)
        .count();
    assert_eq!(by_description, 0);

    // The labels the 4D script wrote on 2026-11-02 (its journal, the last
    // label of each), in a catalog made of the dump: every one agrees with
    // the sets found and is cleared, the two sets the person confirmed
    // become confirmed groups.
    let journal = std::fs::read_to_string(dir.join("4dct_tag_journal.jsonl")).expect("journal");
    let mut labels: BTreeMap<(String, String), String> = BTreeMap::new();
    for line in journal.lines().filter(|l| !l.trim().is_empty()) {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        labels.insert((text(&v, "kind"), text(&v, "uid")), text(&v, "new"));
    }
    let mut c = catalog_of(&dump);
    for ((kind, uid), label) in &labels {
        match kind.as_str() {
            "series" => c.series.get_mut(uid).expect("series").label = label.clone(),
            _ => c.studies.get_mut(uid).expect("study").label = label.clone(),
        }
    }
    archive::fourd::index(&mut c);
    let notes = archive::fourd::migrate_labels(&mut c);
    let cleared = |kind: &str| {
        notes
            .iter()
            .filter(|n| n.kind == kind && n.what.starts_with("cleared"))
            .count()
    };
    for n in notes.iter().filter(|n| !n.what.starts_with("cleared")) {
        eprintln!("{} {} {:?}: {}", n.kind, n.uid, n.old, n.what);
    }
    assert_eq!((cleared("series"), cleared("study")), (370, 25));
    assert_eq!(notes.len(), 395, "nothing kept");
    archive::fourd::index(&mut c);
    let confirmed = c
        .studies
        .values()
        .flat_map(|s| &s.fourd)
        .filter(|f| f.confirmed)
        .count();
    assert_eq!(confirmed, 35, "the person's two confirmations are kept");
}

/// A Philips 4DCT exported without descriptions beside the daily CTs of its
/// block, its untagged reconstruction, its trace and a structure set drawn
/// on one phase; a study holding only the scout of a 4DCT. The catalog finds
/// the set, keeps it through rescans (writing nothing when nothing
/// changed), takes a person's decisions and gives them back after an
/// upgrade from version 2.
#[test]
fn the_catalog_finds_keeps_and_takes_corrections_of_4d_sets() {
    use rust_dicom_station::archive::Archive;
    use rust_dicom_station::progress::Progress;

    let root = std::env::temp_dir().join("rds_fourd_catalog");
    let _ = std::fs::remove_dir_all(&root);
    philips_4d(&root);
    let st = STUDY_4D;
    let scout = STUDY_SCOUT;
    let ar = Archive::new(&root);
    let tree = ar.scan_tree(&Progress::default()).unwrap();
    let study = tree
        .patients()
        .into_iter()
        .flat_map(|p| p.studies)
        .find(|s| s.study_uid == st)
        .unwrap();
    assert_eq!(study.fourd.len(), 1, "{:#?}", study.fourd);
    let g = &study.fourd[0];
    assert_eq!(g.evidence, "same-acquisition");
    assert_eq!((g.phases, g.slices, g.scan), (10, 4, 1));
    assert_eq!(g.recons, vec!["untagged".to_string()]);
    assert_eq!(g.name, "4DCT 2024-06-10 (10 phases + 1)");
    let role = |uid: &str| {
        study
            .series
            .iter()
            .find(|s| s.series_uid == uid)
            .and_then(|s| s.fourd.clone())
    };
    let p = role("1.2.826.0.1.4.204").unwrap();
    assert_eq!(
        (p.phase.as_str(), p.inferred),
        ("40%", true),
        "ten phases: by their place"
    );
    assert_eq!(role("1.2.826.0.1.4.299").unwrap().role, "untagged");
    assert_eq!(role("1.2.826.0.1.4.99").unwrap().role, "trace");
    let rs = role("1.2.826.0.1.4.88").unwrap();
    assert_eq!((rs.role.as_str(), rs.phase.as_str()), ("structures", "40%"));
    assert!(role("1.2.826.0.1.4.70").is_none(), "a daily CT is no phase");
    let orphan = tree
        .patients()
        .into_iter()
        .flat_map(|p| p.studies)
        .find(|s| s.study_uid == scout)
        .unwrap();
    assert!(orphan.fourd.is_empty());
    assert_eq!(orphan.fourd_orphans[0].what, "scout");
    let listing = ar.fourd_groups(st);
    assert_eq!(listing[0].members.len(), 12);
    assert_eq!(listing[0].structures.len(), 1);
    assert_eq!(listing[0].members[0].phase, "0%");

    // Nothing new: nothing written.
    let cat = root.join("rds-pacs-catalog.json");
    let before = (
        std::fs::read(&cat).unwrap(),
        std::fs::metadata(&cat).unwrap().modified().unwrap(),
    );
    ar.refresh(&Progress::default()).unwrap();
    let after = (
        std::fs::read(&cat).unwrap(),
        std::fs::metadata(&cat).unwrap().modified().unwrap(),
    );
    assert!(before == after, "a scan with nothing new writes nothing");

    // Rejected, it stays away through a rescan; reset, it is back.
    let id = g.group.clone();
    ar.confirm_4d(&id, false).unwrap();
    ar.refresh(&Progress::default()).unwrap();
    assert!(ar.fourd_groups(st).is_empty());
    ar.reset_4d_group(&id).unwrap();
    assert_eq!(ar.fourd_groups(st).len(), 1);

    // Made by hand of three series: it replaces the set and survives a scan.
    let picked: Vec<(String, String, String)> = [
        "1.2.826.0.1.4.200",
        "1.2.826.0.1.4.201",
        "1.2.826.0.1.4.202",
    ]
    .iter()
    .map(|u| (u.to_string(), String::new(), String::new()))
    .collect();
    let mine = ar.set_4d_group(st, &picked, "Mine").unwrap();
    assert_eq!(mine, id, "it took the id of the set it replaced");
    ar.refresh(&Progress::default()).unwrap();
    let now = ar.fourd_groups(st);
    assert_eq!(now.len(), 1);
    assert_eq!(
        (
            now[0].set.name.as_str(),
            now[0].set.custom,
            now[0].members.len()
        ),
        ("Mine", true, 3)
    );
    ar.reset_4d_group(&id).unwrap();

    // A catalog of version 2 (no heads, no positions) with the labels the
    // script wrote: the CTs are read again, the sets found, the labels that
    // agree cleared.
    let mut c: serde_json::Value = serde_json::from_slice(&std::fs::read(&cat).unwrap()).unwrap();
    c["version"] = serde_json::json!(2);
    c["fourd_migrated"] = serde_json::json!(false);
    for (uid, s) in c["series"].as_object_mut().unwrap() {
        let s = s.as_object_mut().unwrap();
        s.remove("head");
        s.remove("head_file");
        s.remove("fourd");
        for f in s["files"].as_array_mut().unwrap() {
            f.as_array_mut().unwrap().truncate(5);
        }
        if uid == "1.2.826.0.1.4.203" {
            s.insert("label".into(), serde_json::json!("4DCT ~30%"));
        }
        if uid == "1.2.826.0.1.4.70" {
            s.insert("label".into(), serde_json::json!("4DCT t9"));
        }
    }
    c["studies"][st]["label"] = serde_json::json!("2024-06-10 \u{b7} CT \u{b7} 4DCT");
    c["studies"][st].as_object_mut().unwrap().remove("fourd");
    std::fs::write(&cat, serde_json::to_string(&c).unwrap()).unwrap();
    let tree = ar.scan_tree(&Progress::default()).unwrap();
    let study = tree
        .patients()
        .into_iter()
        .flat_map(|p| p.studies)
        .find(|s| s.study_uid == st)
        .unwrap();
    assert_eq!(study.fourd.len(), 1);
    assert_eq!(study.label, "2024-06-10 \u{b7} CT");
    let label = |uid: &str| {
        study
            .series
            .iter()
            .find(|s| s.series_uid == uid)
            .unwrap()
            .label
            .clone()
    };
    assert_eq!(label("1.2.826.0.1.4.203"), "", "agrees: cleared");
    assert_eq!(label("1.2.826.0.1.4.70"), "4DCT t9", "in no set: kept");
    let journal = std::fs::read_to_string(root.join(archive::fourd::LABEL_JOURNAL)).unwrap();
    assert_eq!(journal.lines().count(), 3);
    let _ = std::fs::remove_dir_all(&root);
}

/// A catalog holding the series of the dump, each with as many files as it
/// had, at its first and last slice position.
fn catalog_of(dump: &[Dumped]) -> archive::catalog::Catalog {
    use archive::catalog::{Catalog, FileRec, SeriesRec, StudyRec};
    let mut c = Catalog::default();
    for d in dump {
        c.studies
            .entry(d.study.clone())
            .or_insert_with(|| StudyRec {
                patient: "P".into(),
                ..Default::default()
            });
        let files = match d.z {
            Some(z) => (0..z.n)
                .map(|i| {
                    let at = if i + 1 == z.n { z.max } else { z.min };
                    FileRec(
                        0,
                        format!("{}/{i}", d.uid),
                        1,
                        0,
                        format!("{}.{i}", d.uid),
                        Some(at),
                    )
                })
                .collect(),
            None => vec![FileRec(
                0,
                format!("{}/0", d.uid),
                1,
                0,
                format!("{}.0", d.uid),
                None,
            )],
        };
        c.series.insert(
            d.uid.clone(),
            SeriesRec {
                study: d.study.clone(),
                modality: d.modality.clone(),
                number: d.number.map(|n| n.to_string()).unwrap_or_default(),
                description: d.description.clone(),
                files,
                head: Some(d.head.clone()),
                ..Default::default()
            },
        );
    }
    c
}

/// The viewer groups a study loaded from the PACS as the catalog does,
/// under the same id; what was decided in the catalog comes with the
/// study, and what the viewer decides goes back into it.
#[test]
fn the_viewer_and_the_catalog_are_one_source() {
    use rust_dicom_station::archive::Archive;
    use rust_dicom_station::progress::Progress;

    let root = std::env::temp_dir().join("rds_fourd_viewer");
    let _ = std::fs::remove_dir_all(&root);
    philips_4d(&root);
    let ar = Archive::new(&root);
    ar.refresh(&Progress::default()).unwrap();
    let id = ar.fourd_groups(STUDY_4D)[0].set.group.clone();
    let files: Vec<std::path::PathBuf> = ar
        .study_instances(STUDY_4D)
        .unwrap()
        .into_iter()
        .map(|i| i.path)
        .collect();
    let mut study =
        rust_dicom_station::loader::load_files(&files, "test", &Progress::default()).unwrap();
    assert_eq!(study.fourd_groups.len(), 1);
    assert_eq!(study.fourd_groups[0].id, id, "the same set, the same id");
    assert_eq!(study.fourd_groups[0].phase_members().len(), 10);
    assert_eq!(study.fourd_groups[0].members[4].label, "~40%");

    // Rejected in the catalog: the loaded study has it dissolved.
    ar.confirm_4d(&id, false).unwrap();
    archive::fourd::apply_to_study(&ar, &mut study);
    assert!(
        study.fourd_groups.iter().all(|g| g.dissolved),
        "{:?}",
        study.fourd_groups
    );
    ar.reset_4d_group(&id).unwrap();

    // Renamed by hand in the viewer: kept in the catalog, in the background.
    let mut groups = rust_dicom_station::fourd::detect(&study.series);
    groups[0].custom = true;
    groups[0].name = "Edited in the viewer".into();
    archive::fourd::keep_viewer_groups(&ar, STUDY_4D, groups, Vec::new());
    let t = std::time::Instant::now();
    loop {
        let g = ar.fourd_groups(STUDY_4D);
        if g.first()
            .is_some_and(|g| g.set.name == "Edited in the viewer")
        {
            assert!(g[0].set.custom);
            assert_eq!(g[0].set.group, id);
            break;
        }
        assert!(
            t.elapsed().as_secs() < 20,
            "the edit never reached the catalog: {g:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// Two stations holding the same studies, in folders of their own: the
/// names, the cohorts and the 4D decisions of one go to the other through
/// a tags file; what the other does not hold is counted, its own names it
/// keeps.
#[test]
fn the_tags_of_one_station_go_to_another() {
    use rust_dicom_station::archive::{Archive, Pick};
    use rust_dicom_station::progress::Progress;

    let base = std::env::temp_dir().join("rds_pacs_tags");
    let _ = std::fs::remove_dir_all(&base);
    let (one, two) = (base.join("one"), base.join("two"));
    philips_4d(&one);
    philips_4d(&two);
    // The second station lacks the scout's study.
    std::fs::remove_dir_all(two.join(format!("Trial/P4D-01/{STUDY_SCOUT}"))).unwrap();
    let (a, b) = (Archive::new(&one), Archive::new(&two));
    a.refresh(&Progress::default()).unwrap();
    b.refresh(&Progress::default()).unwrap();

    a.rename(&Pick::Study(STUDY_4D.into()), "Upright 4DCT")
        .unwrap();
    a.rename(&Pick::Study(STUDY_SCOUT.into()), "Scout only")
        .unwrap();
    a.rename(&Pick::Series("1.2.826.0.1.4.299".into()), "Ungated")
        .unwrap();
    let cohort = a.new_cohort("Motion study").unwrap();
    a.put_in_cohort(&["P4D-01".to_string()], None, &cohort, true)
        .unwrap();
    let group = a.fourd_groups(STUDY_4D)[0].set.group.clone();
    a.confirm_4d(&group, true).unwrap();
    b.rename(&Pick::Series("1.2.826.0.1.4.200".into()), "Mine")
        .unwrap();

    let file = base.join("tags.json");
    let out = a.export_tags(&file).unwrap();
    assert_eq!((out.studies, out.series, out.fourd), (2, 1, 1), "{out:?}");
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(
        !text.contains(&one.display().to_string()),
        "no path of the station travels"
    );

    let got = b.import_tags(&file).unwrap();
    assert_eq!((got.studies, got.series, got.fourd), (1, 1, 1), "{got:?}");
    assert_eq!(got.not_held, 1, "the scout's study is not held there");
    assert_eq!(got.cohorts_made, 1);
    let tree = b.tree();
    let study = tree
        .patients()
        .into_iter()
        .flat_map(|p| p.studies)
        .find(|s| s.study_uid == STUDY_4D)
        .unwrap();
    assert_eq!(study.label, "Upright 4DCT");
    let label = |uid: &str| {
        study
            .series
            .iter()
            .find(|s| s.series_uid == uid)
            .unwrap()
            .label
            .clone()
    };
    assert_eq!(label("1.2.826.0.1.4.299"), "Ungated");
    assert_eq!(
        label("1.2.826.0.1.4.200"),
        "Mine",
        "a name of its own stays"
    );
    let motion = tree
        .cohorts
        .iter()
        .find(|c| c.name == "Motion study")
        .unwrap();
    assert_eq!(motion.patients.len(), 1);
    assert!(
        tree.cohorts.iter().any(|c| c.name == "Trial"),
        "still in its folder's cohort"
    );
    assert!(
        b.fourd_groups(STUDY_4D)[0].set.custom,
        "the 4D decision came along"
    );
    // Twice is the same as once.
    b.import_tags(&file).unwrap();
    assert_eq!(b.tree().cohorts.len(), tree.cohorts.len());
    let _ = std::fs::remove_dir_all(&base);
}
