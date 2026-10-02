//! Workflows run end to end on the 4D phantom, headless.
//!
//! The example the program ships - a cardiac CT and a 4DCT, the heart
//! segmented on both, the target carried onto every phase anchored on the
//! heart, the motion measured and the ITV built, everything written out -
//! is rebuilt here with what CI can run: the body outline (classical, no
//! network) stands in for the heart, and the two inputs are two readings of
//! the phantom's three-phase 4D study, whose target sits at y = 0, 6 and
//! 3 mm on the phases. The run has to file the outline on every phase, land
//! the target on every phase where the phase has it, find that motion, file
//! the ITV on the reference phase and write the structure sets, the
//! reports, the summary and the workflow itself into the run folder.

mod common;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::mpsc;

use rust_dicom_station::motion::MotionModel;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::workflow::graph::catalog::*;
use rust_dicom_station::workflow::graph::exec::{self, Channel, Event, RunOptions, Status};
use rust_dicom_station::workflow::graph::Workflow;

const SHIFTS: [f64; 3] = [0.0, 6.0, 3.0];

/// Registration effort small enough for a test, as the workflow suite uses.
fn quick() -> Effort {
    Effort {
        levels: 2,
        iterations: 150,
        samples: 2000,
        grid_spacing_mm: 16.0,
        fixed_threshold: -500.0,
    }
}

/// The shipped example's shape, with the body outline for the heart.
fn anchored_workflow(folder: &Path) -> Workflow {
    let mut wf = Workflow::new("phantom anchored motion");
    let path = folder.display().to_string();
    let a = wf.add(
        Op::LoadFolder(LoadFolder {
            path: path.clone(),
            workspace: Workspace::A,
        }),
        "Source",
        [0.0, 0.0],
    );
    let img = wf.add(
        Op::SelectImage(SelectImage {
            description: "0%".into(),
            ..SelectImage::default()
        }),
        "",
        [250.0, 0.0],
    );
    let target = wf.add(
        Op::SelectStructures(SelectStructures {
            names: "target*".into(),
            first_only: true,
            required: true,
        }),
        "Find the target",
        [500.0, -100.0],
    );
    let body_src = wf.add(
        Op::BodyContour(BodyContour {
            name: "body total".into(),
            ..BodyContour::default()
        }),
        "Outline of the source",
        [500.0, 60.0],
    );
    let b = wf.add(
        Op::LoadFolder(LoadFolder {
            path,
            workspace: Workspace::B,
        }),
        "4D",
        [0.0, 400.0],
    );
    let group = wf.add(Op::SelectGroup(SelectGroup::default()), "", [250.0, 400.0]);
    let body_4d = wf.add(
        Op::BodyContour(BodyContour {
            name: "body total".into(),
            ..BodyContour::default()
        }),
        "Outline on every phase",
        [500.0, 400.0],
    );
    let prop = wf.add(
        Op::PropagateToGroup(PropagateToGroup {
            // The phantom's outline is the same on every phase - only the
            // target moves inside it - so matching the outline's contours
            // would find nothing to move; the images inside it do.
            anchor_by: AnchorBy::Intensity,
            effort: quick(),
            ..PropagateToGroup::default()
        }),
        "Target onto the phases",
        [800.0, 200.0],
    );
    let motion = wf.add(
        Op::Motion(Motion {
            // The phantom's body does not move; a global rigid body is what
            // the workflow suite checks too.
            local_rigid_margin_mm: 0.0,
            effort: quick(),
            ..Motion::default()
        }),
        "Motion",
        [1100.0, 300.0],
    );
    let export = wf.add(
        Op::ExportDicom(ExportDicom {
            folder: "{input}".into(),
            ..ExportDicom::default()
        }),
        "Structures",
        [1400.0, 400.0],
    );
    let report = wf.add(
        Op::SaveReport(SaveReport::default()),
        "Reports",
        [1400.0, 100.0],
    );
    wf.link(a, 0, img, 0);
    wf.link(img, 0, target, 0);
    wf.link(img, 0, body_src, 0);
    wf.link(b, 0, group, 0);
    wf.link(group, 0, body_4d, 0);
    wf.link(target, 0, prop, 0);
    wf.link(body_src, 0, prop, 1);
    wf.link(body_4d, 0, prop, 2);
    wf.link(prop, 0, motion, 0);
    wf.link(body_4d, 0, motion, 1);
    wf.link(motion, 0, export, 0);
    wf.link(prop, 0, export, 0);
    wf.link(prop, 2, report, 0);
    wf.link(motion, 1, report, 0);
    wf.link(body_4d, 1, report, 0);
    wf
}

fn options(dir: &Path, wf: &Workflow) -> RunOptions {
    let mut o = RunOptions::new(wf, &dir.join("runs"), dir.join("models"));
    o.allow_download = false;
    o
}

#[test]
fn the_anchored_example_runs_end_to_end_on_the_phantom() {
    let dir = common::target_dir("test_workflow_graph_anchored");
    let folder = common::fourd_folder(&dir, SHIFTS);
    let wf = anchored_workflow(&folder);
    let f = wf.check();
    assert!(f.ok(), "{:?}", f.errors);

    // The file round trip first: the run below runs what was read back.
    let wf = Workflow::from_json(&wf.to_json()).expect("reads back");

    let opts = options(&dir, &wf);
    let (tx, rx) = mpsc::channel();
    let channel = Channel::new(tx, None);
    let t0 = std::time::Instant::now();
    let out = exec::run(&wf, &opts, &Progress::default(), &channel);
    eprintln!("workflow run: {:.1} s", t0.elapsed().as_secs_f64());
    for r in &out.records {
        eprintln!("{:?} {} {:?} {:?}", r.status, r.label, r.lines, r.secs);
    }
    assert!(out.ok(), "the run stopped: {:?}", out.error);
    assert!(out.records.iter().all(|r| r.status == Status::Done));
    let events: Vec<Event> = rx.try_iter().collect();
    let started = events
        .iter()
        .filter(|e| matches!(e, Event::Started { .. }))
        .count();
    assert_eq!(started, wf.nodes.len(), "one Started per step");
    assert!(
        !events.iter().any(|e| matches!(e, Event::Show(_))),
        "a run in the background shows nothing"
    );

    // The 4D study: every phase carries the outline and the target in a
    // structure set of its own.
    let fourd = &out.studies[1].study;
    let g = &fourd.fourd_groups[0];
    let phases = rust_dicom_station::workflow::phases_of(g, &fourd.series).unwrap();
    for (label, se) in &phases {
        let names: Vec<&str> = fourd
            .structure_sets
            .iter()
            .filter(|s| s.referenced_series_uid == se.uid)
            .flat_map(|s| s.rois.iter().map(|r| r.name.as_str()))
            .collect();
        assert!(names.contains(&"body total"), "{label}: {names:?}");
        // Phase 0 was contoured with a TARGET of its own, so the carried
        // one lands beside it as TARGET (2) - and under that same name on
        // every phase, which is the name the motion step reads.
        assert!(names.contains(&"TARGET (2)"), "{label}: {names:?}");
        assert!(
            names.contains(&"body total_prop"),
            "{label}: the anchor's own landed copy is there, {names:?}"
        );
    }

    // The motion: as contoured, the landed targets follow the phantom's
    // 0 / 6 / 3 mm; peak-to-peak 6 mm.
    let motion = out
        .reports
        .iter()
        .find_map(|r| r.motion.as_ref())
        .expect("a motion report");
    let track = motion
        .tracks
        .iter()
        .find(|t| t.model == MotionModel::Contoured && t.target == "TARGET (2)")
        .expect("the carried target read as contoured, not phase 0's own");
    let p2p = track.peak_to_peak();
    eprintln!("as-contoured TARGET peak-to-peak {p2p:.2} mm");
    assert!(
        (p2p - 6.0).abs() < 2.0,
        "peak-to-peak {p2p:.2} mm, expected 6"
    );
    assert!(!motion.itvs.is_empty(), "an ITV was built");
    let ref_uid = &phases[0].1.uid;
    let itv_on_ref = fourd
        .structure_sets
        .iter()
        .filter(|s| &s.referenced_series_uid == ref_uid)
        .flat_map(|s| &s.rois)
        .any(|r| r.name.to_lowercase().contains("itv"));
    assert!(itv_on_ref, "the ITV is filed on the reference phase");

    // The files.
    let run = &out.run_dir;
    assert!(run.join("workflow.rdsflow").is_file());
    let summary = std::fs::read_to_string(run.join("run-summary.md")).unwrap();
    assert!(summary.contains("Every step ran."), "{summary}");
    let written: Vec<_> = walk(&run.join("4D"));
    assert!(
        written.len() >= phases.len(),
        "one structure set per phase at least: {written:?}"
    );
    let reports: Vec<_> = walk(&run.join("reports"));
    assert!(
        reports
            .iter()
            .any(|p| p.to_string_lossy().ends_with("full report.csv")),
        "{reports:?}"
    );
    assert!(reports
        .iter()
        .any(|p| p.extension().is_some_and(|e| e == "md")));
    // The source study was not changed beyond its own outline.
    let src = &out.studies[0].study;
    assert!(src
        .structure_sets
        .iter()
        .flat_map(|s| &s.rois)
        .any(|r| r.name == "body total"));
    let _ = std::fs::remove_dir_all(&dir);
}

fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out
}

/// Showing every step: each step's studies arrive, set to display what the
/// step worked on, and the run waits for each to be acknowledged.
#[test]
fn a_run_that_shows_its_steps_hands_over_each_one_and_waits() {
    let dir = common::target_dir("test_workflow_graph_show");
    let folder = common::fourd_folder(&dir, SHIFTS);
    let mut wf = Workflow::new("show");
    let load = wf.add(
        Op::LoadFolder(LoadFolder {
            path: String::new(),
            workspace: Workspace::Auto,
        }),
        "4D",
        [0.0, 0.0],
    );
    let group = wf.add(Op::SelectGroup(SelectGroup::default()), "", [250.0, 0.0]);
    let body = wf.add(Op::BodyContour(BodyContour::default()), "", [500.0, 0.0]);
    let export = wf.add(Op::ExportDicom(ExportDicom::default()), "", [750.0, 0.0]);
    wf.link(load, 0, group, 0);
    wf.link(group, 0, body, 0);
    wf.link(body, 0, export, 0);
    // No folder in the file: the run's input override supplies it.
    assert!(!wf.check().ok());
    let mut opts = options(&dir, &wf);
    opts.inputs = BTreeMap::from([(load, folder.clone())]);
    opts.show_steps = true;

    let (tx, rx) = mpsc::channel();
    let (ack_tx, ack_rx) = mpsc::channel();
    let channel = Channel::new(tx, Some(ack_rx));
    let worker = std::thread::spawn(move || exec::run(&wf, &opts, &Progress::default(), &channel));
    let mut shown = Vec::new();
    loop {
        match rx.recv() {
            Ok(Event::Show(s)) => {
                for st in &s.studies {
                    let active = &st.study.series[st.study.active_series].uid;
                    shown.push((s.node, active.clone(), st.focus.clone()));
                }
                ack_tx.send(()).unwrap();
            }
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let out = worker.join().unwrap();
    assert!(out.ok(), "{:?}", out.error);
    // Load, group, one per phase but the last, then the body step itself.
    let body_shows = shown.iter().filter(|(n, _, _)| *n == body).count();
    assert_eq!(body_shows, 3, "{shown:?}");
    // Each body show displays another phase, from memory.
    let phases: Vec<&String> = shown
        .iter()
        .filter(|(n, _, _)| *n == body)
        .map(|(_, uid, _)| uid)
        .collect();
    assert_eq!(
        phases
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3
    );
    assert!(shown.iter().all(|(_, _, pending)| pending.is_none()));
    // The workflow copy in the run folder records the folder it ran on.
    let copy = std::fs::read_to_string(out.run_dir.join("workflow.rdsflow")).unwrap();
    let copy = Workflow::from_json(&copy).unwrap();
    match &copy.node(load).unwrap().op {
        Op::LoadFolder(p) => assert_eq!(Path::new(&p.path), folder.as_path()),
        other => panic!("{other:?}"),
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_failing_step_stops_the_run_and_says_why() {
    let dir = common::target_dir("test_workflow_graph_fail");
    let mut wf = Workflow::new("fail");
    let load = wf.add(
        Op::LoadFolder(LoadFolder {
            path: dir.join("not-there").display().to_string(),
            workspace: Workspace::Auto,
        }),
        "Nowhere",
        [0.0, 0.0],
    );
    let img = wf.add(Op::SelectImage(SelectImage::default()), "", [250.0, 0.0]);
    let export = wf.add(Op::ExportDicom(ExportDicom::default()), "", [500.0, 0.0]);
    wf.link(load, 0, img, 0);
    wf.link(img, 0, export, 0);
    let (tx, _rx) = mpsc::channel();
    let out = exec::run(
        &wf,
        &options(&dir, &wf),
        &Progress::default(),
        &Channel::new(tx, None),
    );
    assert!(!out.ok());
    let e = out.error.unwrap();
    assert!(e.contains("Nowhere") && e.contains("does not exist"), "{e}");
    assert!(matches!(out.records[0].status, Status::Failed(_)));
    assert_eq!(out.records[1].status, Status::NotRun);
    assert_eq!(out.records[2].status, Status::NotRun);
    let summary = std::fs::read_to_string(out.run_dir.join("run-summary.md")).unwrap();
    assert!(summary.contains("The run stopped"), "{summary}");
    let _ = std::fs::remove_dir_all(&dir);
}
