//! The workflow steps beyond the example, and the ways a run can go: on
//! the synthetic RT phantom (CT, TARGET / BODY / CORD, a 60 Gy dose) and its
//! three-phase 4D form, headless, nothing downloaded.
//!
//! * Rename, Combine, DVH with a protocol and Dose estimation on one study,
//!   the structures filed where the next step finds them.
//! * Copy to each phase: the target of phase 0 on every phase, under one
//!   name.
//! * Reruns: a second run takes over every unchanged step, and a changed
//!   step reruns from itself on.
//! * Parallel rows give exactly what one row after the other gives.
//! * A batch runs once per matching subfolder and puts the tables together.
//! * Register by structures: two studies aligned on their targets alone.

mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::progress::Progress;
use rust_dicom_station::workflow::graph::catalog::*;
use rust_dicom_station::workflow::graph::exec::{
    self, Channel, Outcome, RunOptions, Status, StepCache,
};
use rust_dicom_station::workflow::graph::Workflow;

/// One phantom study in `dir/name`.
fn phantom(dir: &Path, name: &str, shift: f64) -> PathBuf {
    let out = dir.join(name);
    let p = GenParams {
        target_shift_y: shift,
        extras: false,
        ..GenParams::default()
    };
    gen_test_data::generate(&out, &p, &Progress::default()).expect("phantom generated");
    out
}

fn options(dir: &Path, wf: &Workflow) -> RunOptions {
    let mut o = RunOptions::new(wf, &dir.join("runs"), dir.join("models"));
    o.allow_download = false;
    o
}

fn run(wf: &Workflow, opts: &RunOptions) -> Outcome {
    let (tx, _rx) = mpsc::channel();
    let out = exec::run(wf, opts, &Progress::default(), &Channel::new(tx, None));
    for r in &out.records {
        eprintln!(
            "{:?} {} {:?} {:.2}s{}",
            r.status,
            r.label,
            r.lines,
            r.secs,
            if r.reused { " (reused)" } else { "" }
        );
    }
    out
}

fn select(names: &str) -> Op {
    Op::SelectStructures(SelectStructures {
        names: names.into(),
        first_only: true,
        required: true,
    })
}

fn load(path: &Path) -> Op {
    Op::LoadFolder(LoadFolder {
        path: path.display().to_string(),
        workspace: Workspace::Auto,
    })
}

/// Every structure name the study holds.
fn names_of(out: &Outcome, ds: usize) -> Vec<String> {
    out.studies[ds]
        .study
        .structure_sets
        .iter()
        .flat_map(|s| s.rois.iter().map(|r| r.name.clone()))
        .collect()
}

/// The single-study pipeline: TARGET renamed GTV, grown 5 mm into a PTV,
/// both measured against the dose with a protocol, the PTV's dose
/// estimated, the reports and the structures written.
fn edit_and_measure(folder: &Path) -> Workflow {
    let mut wf = Workflow::new("edit and measure");
    let l = wf.add(load(folder), "Plan CT", [0.0, 0.0]);
    let img = wf.add(Op::SelectImage(SelectImage::default()), "", [250.0, 0.0]);
    let t = wf.add(select("TARGET"), "", [500.0, 0.0]);
    let rn = wf.add(
        Op::Rename(Rename {
            action: RenameAction::Rename,
            rules: vec![RenameRule {
                from: "TARG*".into(),
                to: "GTV".into(),
            }],
        }),
        "",
        [750.0, 0.0],
    );
    let ptv = wf.add(Op::Combine(Combine::default()), "PTV", [1000.0, 0.0]);
    let dvh = wf.add(
        Op::Dvh(Dvh {
            metrics: "D95%, Dmean, Dmax".into(),
            protocol: "GTV Dmax > 30\nPTV Dmean < 1".into(),
            ..Dvh::default()
        }),
        "Plan check",
        [1250.0, 0.0],
    );
    let est = wf.add(Op::DoseMetrics(DoseMetrics::default()), "", [1250.0, 200.0]);
    let rep = wf.add(Op::SaveReport(SaveReport::default()), "", [1500.0, 0.0]);
    let exp = wf.add(Op::ExportDicom(ExportDicom::default()), "", [1500.0, 200.0]);
    wf.link(l, 0, img, 0);
    wf.link(img, 0, t, 0);
    wf.link(t, 0, rn, 0);
    wf.link(rn, 0, ptv, 0);
    wf.link(ptv, 0, dvh, 0);
    wf.link(rn, 0, dvh, 0);
    wf.link(ptv, 0, est, 0);
    wf.link(dvh, 0, rep, 0);
    wf.link(est, 0, rep, 0);
    wf.link(ptv, 1, rep, 0);
    wf.link(ptv, 0, exp, 0);
    wf
}

#[test]
fn rename_combine_and_the_dose_steps_run_on_the_phantom() {
    let dir = common::target_dir("test_workflow_steps_edit");
    let folder = phantom(&dir, "plan", 0.0);
    let wf = edit_and_measure(&folder);
    assert!(wf.check().ok(), "{:?}", wf.check().errors);
    let wf = Workflow::from_json(&wf.to_json()).expect("reads back");
    let out = run(&wf, &options(&dir, &wf));
    assert!(out.ok(), "{:?}", out.error);

    let names = names_of(&out, 0);
    assert!(names.contains(&"GTV".to_string()), "{names:?}");
    assert!(names.contains(&"PTV".to_string()), "{names:?}");
    assert!(!names.contains(&"TARGET".to_string()), "renamed: {names:?}");

    let table = |report: &str, table: &str| {
        out.reports
            .iter()
            .find(|r| r.title.starts_with(report))
            .and_then(|r| r.tables.iter().find(|t| t.title == table))
            .cloned()
            .unwrap_or_else(|| panic!("{report} / {table}"))
    };
    // The PTV is the GTV grown by 5 mm: larger, and the DVH says so too.
    let dvh = table("DVH", "DVH metrics");
    let volume = |name: &str| -> f64 {
        dvh.rows
            .iter()
            .find(|r| r[0] == name)
            .map(|r| r[1].parse().unwrap())
            .unwrap_or_else(|| panic!("{name} in {:?}", dvh.rows))
    };
    let (gtv, ptv) = (volume("GTV"), volume("PTV"));
    eprintln!("GTV {gtv} cm3, PTV {ptv} cm3");
    assert!(ptv > gtv * 1.3, "GTV {gtv}, PTV {ptv}");
    let combined = table("Combined", "Combined structure");
    let filed: f64 = combined.rows[0][2].parse().unwrap();
    assert!(
        (filed - ptv).abs() / ptv < 0.1,
        "the combined volume {filed} and the DVH's {ptv} agree"
    );
    // The protocol: the target sees the peak, the PTV's mean is not 1 Gy.
    let protocol = table("DVH", "Protocol");
    let verdict = |s: &str| {
        protocol
            .rows
            .iter()
            .find(|r| r[1] == s)
            .map(|r| r[3].clone())
            .unwrap_or_else(|| panic!("{s} in {:?}", protocol.rows))
    };
    assert_eq!(verdict("GTV"), "pass");
    assert_eq!(verdict("PTV"), "FAIL");
    let est = table("Dose estimation", "Dose metrics");
    assert_eq!(est.rows.len(), 1);
    assert_eq!(est.rows[0][0], "PTV");

    // The files: the reports and the new structure set.
    let run_dir = &out.run_dir;
    let files: Vec<PathBuf> = walk(run_dir);
    assert!(
        files
            .iter()
            .any(|p| p.to_string_lossy().ends_with("Protocol.csv")),
        "{files:?}"
    );
    assert!(files.iter().any(|p| p
        .file_name()
        .is_some_and(|n| n.to_string_lossy().starts_with("RS_"))));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn copy_to_phases_puts_one_structure_on_every_phase_under_one_name() {
    let dir = common::target_dir("test_workflow_steps_copy");
    let folder = common::fourd_folder(&dir, [0.0, 6.0, 3.0]);
    let mut wf = Workflow::new("copy");
    let l = wf.add(load(&folder), "4D", [0.0, 0.0]);
    let img = wf.add(
        Op::SelectImage(SelectImage {
            description: "0%".into(),
            ..SelectImage::default()
        }),
        "",
        [250.0, 0.0],
    );
    let cord = wf.add(select("CORD"), "", [500.0, 0.0]);
    let g = wf.add(Op::SelectGroup(SelectGroup::default()), "", [250.0, 200.0]);
    let copy = wf.add(
        Op::CopyToPhases(CopyToPhases::default()),
        "",
        [750.0, 100.0],
    );
    wf.link(l, 0, img, 0);
    wf.link(img, 0, cord, 0);
    wf.link(l, 0, g, 0);
    wf.link(cord, 0, copy, 0);
    wf.link(g, 0, copy, 1);
    assert!(wf.check().ok(), "{:?}", wf.check().errors);
    let out = run(&wf, &options(&dir, &wf));
    assert!(out.ok(), "{:?}", out.error);

    let study = &out.studies[0].study;
    let group = &study.fourd_groups[0];
    let phases = rust_dicom_station::workflow::phases_of(group, &study.series).unwrap();
    // Phase 0 holds a CORD already, so the copy is CORD (2) there - and
    // under that same name on the phases that had none.
    for (label, se) in &phases {
        let names: Vec<&str> = study
            .structure_sets
            .iter()
            .filter(|s| s.referenced_series_uid == se.uid)
            .flat_map(|s| s.rois.iter().map(|r| r.name.as_str()))
            .collect();
        assert!(names.contains(&"CORD (2)"), "{label}: {names:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn register_by_structures_aligns_two_studies_on_their_targets() {
    let dir = common::target_dir("test_workflow_steps_shape");
    // The repeat scan's target sits 6 mm away from the plan's; the rest of
    // the anatomy is where it was.
    let plan = phantom(&dir, "plan", 0.0);
    let repeat = phantom(&dir, "repeat", 6.0);
    let mut wf = Workflow::new("by structures");
    let a = wf.add(load(&plan), "Plan", [0.0, 0.0]);
    let b = wf.add(load(&repeat), "Repeat", [0.0, 300.0]);
    let ia = wf.add(Op::SelectImage(SelectImage::default()), "", [250.0, 0.0]);
    let ib = wf.add(Op::SelectImage(SelectImage::default()), "", [250.0, 300.0]);
    let ta = wf.add(select("TARGET"), "", [500.0, 0.0]);
    let tb = wf.add(select("TARGET"), "", [500.0, 300.0]);
    let reg = wf.add(
        Op::RegisterByStructures(RegisterByStructures {
            shape: ShapeRegParams {
                dof: ShapeDofChoice::Translation,
                ..ShapeRegParams::default()
            },
            ..RegisterByStructures::default()
        }),
        "",
        [750.0, 150.0],
    );
    let cord = wf.add(select("CORD"), "", [750.0, 0.0]);
    let prop = wf.add(Op::Propagate(Propagate::default()), "", [1000.0, 100.0]);
    wf.link(a, 0, ia, 0);
    wf.link(b, 0, ib, 0);
    wf.link(ia, 0, ta, 0);
    wf.link(ib, 0, tb, 0);
    wf.link(ta, 0, reg, 0);
    wf.link(tb, 0, reg, 1);
    wf.link(ia, 0, cord, 0);
    wf.link(reg, 0, prop, 0);
    wf.link(cord, 0, prop, 1);
    assert!(wf.check().ok(), "{:?}", wf.check().errors);
    let wf = Workflow::from_json(&wf.to_json()).expect("reads back");
    let out = run(&wf, &options(&dir, &wf));
    assert!(out.ok(), "{:?}", out.error);

    let rep = out
        .reports
        .iter()
        .find(|r| r.title.starts_with("Registration by structures"))
        .expect("a report");
    let quantity = |name: &str| -> String {
        rep.tables[0]
            .rows
            .iter()
            .find(|r| r[0].starts_with(name))
            .map(|r| r[1].clone())
            .unwrap_or_else(|| panic!("{name} in {:?}", rep.tables[0].rows))
    };
    let t: Vec<f64> = quantity("Translation")
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    let norm = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    eprintln!("by structures: translation {t:?}, {:?}", rep.notes);
    assert!((norm - 6.0).abs() < 1.0, "the target's 6 mm: {t:?}");
    for r in quantity("Rotation").split_whitespace() {
        assert!(r.parse::<f64>().unwrap().abs() < 0.01, "no rotation: {r}");
    }
    let row = &rep.tables[1].rows[0];
    assert_eq!(row[0], "TARGET");
    let after: f64 = row[2].parse().unwrap();
    assert!(after < 1.0, "the targets meet: {row:?}");
    // The registration goes on like any other: the cord crosses it.
    let cords = names_of(&out, 1)
        .into_iter()
        .filter(|n| n.starts_with("CORD"))
        .count();
    assert!(cords >= 2, "the repeat's own cord and the plan's");

    // A pair that cannot be read is a problem the editor shows.
    let bad = Op::RegisterByStructures(RegisterByStructures {
        pairs: "Heart heart_total".into(),
        ..RegisterByStructures::default()
    });
    assert_eq!(bad.param_problems().len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Steps that write files always run again; everything else is compared.
fn ran_again(out: &Outcome, wf: &Workflow) -> Vec<String> {
    out.records
        .iter()
        .filter(|r| !r.reused && wf.node(r.node).is_some_and(|n| !n.op.kind().info().writes))
        .map(|r| r.label.clone())
        .collect()
}

#[test]
fn a_rerun_takes_over_unchanged_steps_and_reruns_from_a_changed_one() {
    let dir = common::target_dir("test_workflow_steps_rerun");
    let folder = phantom(&dir, "plan", 0.0);
    let mut wf = edit_and_measure(&folder);
    let cache = StepCache::default();
    let mut opts = options(&dir, &wf);
    opts.reuse = Some(cache.clone());

    let first = run(&wf, &opts);
    assert!(first.ok(), "{:?}", first.error);
    assert!(first.records.iter().all(|r| !r.reused));
    assert!(!cache.is_empty());

    // Unchanged: every step is taken over, and the results are the same.
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let second = run(&wf, &opts);
    assert!(second.ok(), "{:?}", second.error);
    assert!(
        ran_again(&second, &wf).is_empty(),
        "{:?}",
        ran_again(&second, &wf)
    );
    assert_eq!(names_of(&first, 0), names_of(&second, 0));
    let tables = |o: &Outcome| -> Vec<_> {
        o.reports
            .iter()
            .map(|r| {
                (
                    r.title.clone(),
                    r.tables.iter().map(|t| t.rows.clone()).collect::<Vec<_>>(),
                )
            })
            .collect()
    };
    assert_eq!(tables(&first), tables(&second));
    assert!(walk(&second.run_dir)
        .iter()
        .any(|p| p.to_string_lossy().ends_with("Protocol.csv")));

    // The DVH's metrics change: it and the steps after it in the run
    // order run again (the state kept after a step is the run's whole
    // state, which holds the DVH's report from here on), the steps before
    // it do not.
    let dvh_id = wf
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::Dvh(_)))
        .map(|n| n.id)
        .unwrap();
    if let Some(Op::Dvh(p)) = wf.node_mut(dvh_id).map(|n| &mut n.op) {
        p.metrics = "D98%, Dmax".into();
    }
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let third = run(&wf, &opts);
    assert!(third.ok(), "{:?}", third.error);
    let order = wf.order().unwrap();
    let at = order.iter().position(|id| *id == dvh_id).unwrap();
    let expected: Vec<String> = order[at..]
        .iter()
        .filter_map(|id| wf.node(*id))
        .filter(|n| !n.op.kind().info().writes)
        .map(|n| n.label())
        .collect();
    assert_eq!(expected.first().map(String::as_str), Some("Plan check"));
    assert_eq!(ran_again(&third, &wf), expected);
    assert!(
        third.records.iter().filter(|r| r.reused).count() >= 5,
        "load, image, target, rename, PTV taken over"
    );
    assert!(third.reports.iter().any(|r| r
        .tables
        .iter()
        .any(|t| t.header.iter().any(|h| h.starts_with("D98%")))));

    // A changed input folder is a changed first step: everything runs.
    std::fs::write(folder.join("note.txt"), "changed").unwrap();
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let fourth = run(&wf, &opts);
    assert!(fourth.ok(), "{:?}", fourth.error);
    assert!(fourth.records.iter().all(|r| !r.reused));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rows_run_side_by_side_give_what_one_after_the_other_gives() {
    let dir = common::target_dir("test_workflow_steps_parallel");
    let a = phantom(&dir, "a", 0.0);
    let b = phantom(&dir, "b", 4.0);
    let mut wf = Workflow::new("two rows");
    let mut ends = Vec::new();
    for (k, (folder, structure)) in [(&a, "TARGET"), (&b, "CORD")].into_iter().enumerate() {
        let y = k as f32 * 300.0;
        let l = wf.add(load(folder), &format!("Row {k}"), [0.0, y]);
        let img = wf.add(Op::SelectImage(SelectImage::default()), "", [250.0, y]);
        let s = wf.add(select(structure), "", [500.0, y]);
        let c = wf.add(
            Op::Combine(Combine {
                name: format!("{structure} grown"),
                ..Combine::default()
            }),
            "",
            [750.0, y],
        );
        let body = wf.add(
            Op::BodyContour(BodyContour::default()),
            "",
            [500.0, y + 150.0],
        );
        let d = wf.add(Op::Dvh(Dvh::default()), "", [1000.0, y]);
        wf.link(l, 0, img, 0);
        wf.link(img, 0, s, 0);
        wf.link(s, 0, c, 0);
        wf.link(img, 0, body, 0);
        wf.link(c, 0, d, 0);
        wf.link(body, 0, d, 0);
        ends.push((c, d, body));
    }
    let rep = wf.add(Op::SaveReport(SaveReport::default()), "", [1250.0, 150.0]);
    for (c, d, body) in ends {
        wf.link(c, 1, rep, 0);
        wf.link(d, 0, rep, 0);
        wf.link(body, 1, rep, 0);
    }
    assert!(wf.check().ok(), "{:?}", wf.check().errors);

    let cache = StepCache::default();
    let mut opts = options(&dir, &wf);
    opts.reuse = Some(cache.clone());
    let serial = run(&wf, &opts);
    assert!(serial.ok(), "{:?}", serial.error);
    let mut opts = options(&dir, &wf);
    opts.parallel = true;
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let parallel = run(&wf, &opts);
    assert!(parallel.ok(), "{:?}", parallel.error);

    assert_eq!(serial.studies.len(), parallel.studies.len());
    for ds in 0..serial.studies.len() {
        assert_eq!(names_of(&serial, ds), names_of(&parallel, ds), "study {ds}");
    }
    let reports = |o: &Outcome| -> Vec<_> {
        o.reports
            .iter()
            .map(|r| {
                (
                    r.node,
                    r.title.clone(),
                    r.tables.iter().map(|t| t.rows.clone()).collect::<Vec<_>>(),
                )
            })
            .collect()
    };
    assert_eq!(reports(&serial), reports(&parallel));
    let order = |o: &Outcome| -> Vec<u32> { o.records.iter().map(|r| r.node).collect() };
    assert_eq!(
        order(&serial),
        order(&parallel),
        "the records keep the run order"
    );
    assert!(parallel.records.iter().all(|r| r.status == Status::Done));
    assert!(parallel.records.iter().all(|r| !r.reused));

    // Both asked for, with the serial run's states kept: they are taken
    // over, which is quicker than running the rows side by side.
    let mut opts = options(&dir, &wf);
    opts.parallel = true;
    opts.reuse = Some(cache);
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let again = run(&wf, &opts);
    assert!(again.ok(), "{:?}", again.error);
    assert!(
        ran_again(&again, &wf).is_empty(),
        "{:?}",
        ran_again(&again, &wf)
    );
    assert_eq!(reports(&serial), reports(&again));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_batch_runs_each_matching_subfolder_and_puts_the_tables_together() {
    let dir = common::target_dir("test_workflow_steps_batch");
    let root = dir.join("patients");
    phantom(&root, "case 1", 0.0);
    phantom(&root, "case 2", 5.0);
    // Not a case: the pattern leaves it out.
    std::fs::create_dir_all(root.join("notes")).unwrap();
    let mut wf = Workflow::new("batch");
    let l = wf.add(
        Op::LoadFolders(LoadFolders {
            path: root.display().to_string(),
            pattern: "case*".into(),
            workspace: Workspace::Auto,
        }),
        "Patients",
        [0.0, 0.0],
    );
    let img = wf.add(Op::SelectImage(SelectImage::default()), "", [250.0, 0.0]);
    let t = wf.add(select("TARGET"), "", [500.0, 0.0]);
    let d = wf.add(Op::DoseMetrics(DoseMetrics::default()), "", [750.0, 0.0]);
    let rep = wf.add(Op::SaveReport(SaveReport::default()), "", [1000.0, 0.0]);
    wf.link(l, 0, img, 0);
    wf.link(img, 0, t, 0);
    wf.link(t, 0, d, 0);
    wf.link(d, 0, rep, 0);
    assert!(wf.check().ok(), "{:?}", wf.check().errors);
    assert_eq!(
        exec::batch_cases(&root, "case*").unwrap().len(),
        2,
        "the cases the run will see"
    );

    let out = run(&wf, &options(&dir, &wf));
    assert!(out.ok(), "{:?}", out.error);
    assert_eq!(out.cases.len(), 2);
    assert!(out.cases.iter().all(|c| c.error.is_none()));
    assert_eq!(out.cases[0].name, "case 1");
    assert!(out.studies.is_empty(), "a batch keeps no study in memory");
    for c in &out.cases {
        assert!(c.run_dir.starts_with(&out.run_dir));
        assert!(
            c.run_dir.join("run-summary.md").is_file(),
            "{:?}",
            c.run_dir
        );
    }
    assert!(out.run_dir.join("batch-summary.md").is_file());
    let combined = out
        .reports
        .iter()
        .flat_map(|r| &r.tables)
        .find(|t| t.header.first().is_some_and(|h| h == "Case") && t.title.contains("Dose metrics"))
        .expect("the dose table of every case");
    let cases: Vec<&str> = combined.rows.iter().map(|r| r[0].as_str()).collect();
    assert_eq!(cases, vec!["case 1", "case 2"]);
    assert!(
        walk(&out.run_dir)
            .iter()
            .any(|p| p.to_string_lossy().contains("batch - ")
                && p.to_string_lossy().ends_with(".csv"))
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn walk(dir: &Path) -> Vec<PathBuf> {
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
