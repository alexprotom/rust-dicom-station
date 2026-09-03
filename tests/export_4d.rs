//! A 4D acquisition has to come out of an export as a 4D acquisition.
//!
//! The phases of a 4DCT are separate series that hold together only because
//! they share a study, carry their phase in the series description (or in
//! Temporal Position Identifier) and keep their own identity. An export that
//! merges them, renumbers them or takes one without the others quietly turns
//! a 4D study into a pile of scans.
//!
//! The study here is the synthetic phantom, filed three times over as the
//! phases of one acquisition - three because that is the smallest number
//! `fourd::detect` will call a 4D group.

use rust_dicom_station::dicom_export::ExportParams;
use rust_dicom_station::export::{self, ExportPlan};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader::{self, LoadedStudy};
use rust_dicom_station::progress::Progress;

const PHASES: [&str; 3] = ["4DCT 0%", "4DCT 50%", "4DCT 90%"];

fn target(tag: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// The phantom, filed as a three-phase 4D study.
fn fourd_study(tag: &str) -> LoadedStudy {
    let src = target(&format!("{tag}_src"));
    gen_test_data::generate(&src, &GenParams::default(), &Progress::default())
        .expect("test data generation succeeds");
    let mut study = loader::load_directory(&src, &Progress::default()).expect("it loads");
    let base = study.series[study.active_series].clone();
    study.series = PHASES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let mut s = base.clone();
            s.uid = format!("{}.{}", base.uid, i + 1);
            s.description = (*name).into();
            s.series_number = Some(i as i64 + 1);
            s
        })
        .collect();
    study.active_series = 0;
    study.refresh_fourd();
    assert_eq!(study.fourd_groups.len(), 1, "one 4D group was recognised");
    assert_eq!(study.fourd_groups[0].members.len(), 3, "of three phases");
    study
}

fn plan_for(study: &LoadedStudy) -> ExportPlan {
    let mut plan = ExportPlan::build([Some(study), None], ExportParams::for_study(study));
    // The three phases share their source files, so their slices have to be
    // written afresh rather than copied - otherwise all three would carry the
    // same SOP Instance UIDs.
    plan.rerender_images = true;
    plan
}

#[test]
fn the_tree_offers_a_4d_group_as_one_node() {
    let study = fourd_study("test_4d_tree");
    let plan = plan_for(&study);
    let st = plan.studies().next().expect("one study");
    assert_eq!(st.groups.len(), 1, "the group is a node of its own");
    assert_eq!(st.groups[0].members.len(), 3, "with every phase under it");
    assert_eq!(st.groups[0].phases, 3);
    assert!(
        st.series.iter().all(|s| s.fourd.is_some()),
        "and no phase is listed loose beside it"
    );
}

#[test]
fn every_phase_survives_the_round_trip_as_one_acquisition() {
    let study = fourd_study("test_4d_rt");
    let out = target("test_4d_rt_out");
    let plan = plan_for(&study);
    let sum = export::run(&plan, [Some(&study), None], &out, &Progress::default())
        .expect("the export runs");
    assert!(
        !sum.warnings.iter().any(|w| w.contains("4D group")),
        "a complete 4D export has nothing to report: {:?}",
        sum.warnings
    );

    let re = loader::load_directory(&out, &Progress::default()).expect("it reloads");
    assert_eq!(re.series.len(), 3, "still three series");
    let mut descs: Vec<String> = re.series.iter().map(|s| s.description.clone()).collect();
    descs.sort();
    assert_eq!(descs, PHASES.to_vec(), "each phase kept its name");
    assert!(
        re.series
            .windows(2)
            .all(|w| w[0].study_uid == w[1].study_uid),
        "in one study, which is what makes them one acquisition"
    );
    let uids: Vec<&str> = re.series.iter().map(|s| s.uid.as_str()).collect();
    let mut sorted = uids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), 3, "with three distinct Series Instance UIDs");
    assert_eq!(
        re.fourd_groups.len(),
        1,
        "and they are recognised as a 4D group again"
    );
    assert_eq!(re.fourd_groups[0].members.len(), 3);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn taking_only_part_of_a_4d_group_is_reported() {
    let study = fourd_study("test_4d_part");
    let mut plan = plan_for(&study);
    for st in plan.studies_mut() {
        st.series[0].selected = false;
    }
    let out = target("test_4d_part_out");
    let sum = export::run(&plan, [Some(&study), None], &out, &Progress::default())
        .expect("it still runs");
    assert!(
        sum.warnings.iter().any(|w| w.contains("4D group")),
        "half a 4D acquisition is worth saying out loud: {:?}",
        sum.warnings
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn each_phase_gets_its_own_folder() {
    let study = fourd_study("test_4d_dirs");
    let out = target("test_4d_dirs_out");
    export::run(
        &plan_for(&study),
        [Some(&study), None],
        &out,
        &Progress::default(),
    )
    .expect("runs");

    let mut with_files = Vec::new();
    let mut stack = vec![out.clone()];
    while let Some(d) = stack.pop() {
        let mut has_file = false;
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                has_file = true;
            }
        }
        if has_file && d != out {
            with_files.push(d);
        }
    }
    // Three series folders plus the study folder the RT objects sit in.
    assert_eq!(with_files.len(), 4, "one folder per phase: {with_files:?}");
    let _ = std::fs::remove_dir_all(&out);
}
