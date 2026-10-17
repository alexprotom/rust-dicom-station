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

    // The DVH's metrics change: it runs again. The dose estimation comes
    // after it in the run order but does not read from it, and the study
    // it reads the DVH left as it was: it is taken over.
    let dvh_id = wf
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::Dvh(_)))
        .map(|n| n.id)
        .unwrap();
    let est_id = wf
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::DoseMetrics(_)))
        .map(|n| n.id)
        .unwrap();
    let order = wf.order().unwrap();
    let pos = |id: u32| order.iter().position(|x| *x == id).unwrap();
    assert!(pos(est_id) > pos(dvh_id), "the case this checks");
    if let Some(Op::Dvh(p)) = wf.node_mut(dvh_id).map(|n| &mut n.op) {
        p.metrics = "D98%, Dmax".into();
    }
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let third = run(&wf, &opts);
    assert!(third.ok(), "{:?}", third.error);
    assert_eq!(ran_again(&third, &wf), vec!["Plan check".to_string()]);
    assert!(
        third.records.iter().filter(|r| r.reused).count() >= 6,
        "load, image, target, rename, PTV and the dose estimation taken over"
    );
    assert_eq!(
        tables(&second)
            .into_iter()
            .filter(|(t, _)| t.starts_with("Dose estimation"))
            .collect::<Vec<_>>(),
        tables(&third)
            .into_iter()
            .filter(|(t, _)| t.starts_with("Dose estimation"))
            .collect::<Vec<_>>(),
        "what was taken over is what the step made"
    );

    // Renaming a step that makes a report reruns it (the report carries
    // its title); renaming one that does not, does not.
    let img_id = wf
        .nodes
        .iter()
        .find(|n| matches!(n.op, Op::SelectImage(_)))
        .map(|n| n.id)
        .unwrap();
    wf.node_mut(img_id).unwrap().title = "The CT".into();
    wf.node_mut(dvh_id).unwrap().title = "Plan check, renamed".into();
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let renamed = run(&wf, &opts);
    assert!(renamed.ok(), "{:?}", renamed.error);
    assert_eq!(
        ran_again(&renamed, &wf),
        vec!["Plan check, renamed".to_string()]
    );
    assert!(renamed
        .reports
        .iter()
        .any(|r| r.title.ends_with("Plan check, renamed")));
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

/// Two rows that share nothing: a change in the first reruns the steps that
/// read from it (and the step that writes into the study the change
/// touched), and takes the whole second row over, although every step of
/// it comes later in the run order.
#[test]
fn a_change_in_one_row_leaves_the_other_row_taken_over() {
    let dir = common::target_dir("test_workflow_steps_rows_rerun");
    let mut wf = two_rows(&dir);
    let cache = StepCache::default();
    let mut opts = options(&dir, &wf);
    opts.reuse = Some(cache.clone());
    let first = run(&wf, &opts);
    assert!(first.ok(), "{:?}", first.error);

    // The first row's Combine grows by more.
    let order = wf.order().unwrap();
    let combines: Vec<u32> = order
        .iter()
        .copied()
        .filter(|id| matches!(wf.node(*id).unwrap().op, Op::Combine(_)))
        .collect();
    let (row0, row1) = (combines[0], combines[1]);
    let row_of = |wf: &Workflow, id: u32| -> usize {
        // The row is the y of the node: 0 and 150 for the first, 300 and
        // 450 for the second.
        (wf.node(id).unwrap().pos[1] >= 300.0) as usize
    };
    assert_eq!((row_of(&wf, row0), row_of(&wf, row1)), (0, 1));
    if let Some(Op::Combine(c)) = wf.node_mut(row0).map(|n| &mut n.op) {
        c.name = "TARGET grown more".into();
    }
    opts.run_dir = exec::run_folder(&wf, &dir.join("runs"));
    let second = run(&wf, &opts);
    assert!(second.ok(), "{:?}", second.error);
    let ran: Vec<u32> = second
        .records
        .iter()
        .filter(|r| !r.reused)
        .filter(|r| wf.node(r.node).is_some_and(|n| !n.op.kind().info().writes))
        .map(|r| r.node)
        .collect();
    assert!(ran.contains(&row0), "{ran:?}");
    assert!(
        ran.iter().all(|id| row_of(&wf, *id) == 0),
        "nothing of the second row ran again: {ran:?}"
    );
    let second_row_reused = second
        .records
        .iter()
        .filter(|r| row_of(&wf, r.node) == 1 && r.reused)
        .count();
    assert_eq!(second_row_reused, 6, "the whole second row");
    let row1_reports = |o: &Outcome| -> Vec<_> {
        o.reports
            .iter()
            .filter(|r| row_of(&wf, r.node) == 1)
            .map(|r| {
                (
                    r.title.clone(),
                    r.tables.iter().map(|t| t.rows.clone()).collect::<Vec<_>>(),
                )
            })
            .collect()
    };
    assert_eq!(row1_reports(&first), row1_reports(&second));
    assert!(names_of(&second, 0).contains(&"TARGET grown more".to_string()));
    assert!(!names_of(&second, 0).contains(&"TARGET grown".to_string()));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two parallel loads of one series read it from disk once.
#[test]
fn two_parallel_reads_of_one_series_read_it_once() {
    use rust_dicom_station::workflow::session::Volumes;
    let dir = common::target_dir("test_workflow_steps_volumes");
    let folder = phantom(&dir, "plan", 0.0);
    let study = rust_dicom_station::loader::load_directory(&folder, &Progress::default()).unwrap();
    let series = study.series[study.active_series].clone();
    let vols = Volumes::with_budget_mb(1024);
    let barrier = std::sync::Barrier::new(4);
    let got: Vec<_> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    vols.load(&series, &Progress::default()).unwrap()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(vols.reads(), 1, "read once");
    assert!(got.windows(2).all(|w| std::sync::Arc::ptr_eq(&w[0], &w[1])));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Two rows on two phantoms: image, a structure grown, the body, a DVH;
/// one report over both.
fn two_rows(dir: &Path) -> Workflow {
    let a = phantom(dir, "a", 0.0);
    let b = phantom(dir, "b", 4.0);
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
    wf
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

#[test]
fn a_synthetic_ct_step_makes_an_image_the_next_steps_take() {
    let dir = common::target_dir("test_workflow_steps_sct");
    let folder = phantom(&dir, "plan", 0.0);
    // The engine tests' miniature TotalSynth, as a checkpoint folder of the
    // user's (the phantom is a CT, which a folder's app accepts).
    let cps = dir.join("checkpoints");
    std::fs::create_dir_all(&cps).unwrap();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    std::fs::copy(data.join("synth-unetpp-cv0.pt"), cps.join("CV_0.pt")).unwrap();
    let reg = rust_dicom_station::synth::custom::register(std::slice::from_ref(&cps));
    let key = (*reg[0].1.as_ref().unwrap()).to_string();

    let mut wf = Workflow::new("sct");
    let a = wf.add(load(&folder), "Plan", [0.0, 0.0]);
    let s = wf.add(Op::SelectImage(SelectImage::default()), "", [200.0, 0.0]);
    let c = wf.add(
        Op::SynthesizeCt(SynthesizeCt {
            app: key,
            tta: 1,
            mask: "BODY".into(),
            ..SynthesizeCt::default()
        }),
        "",
        [400.0, 0.0],
    );
    let e = wf.add(
        Op::EvaluateSct(EvaluateSct {
            wet: true,
            difference: true,
            ..EvaluateSct::default()
        }),
        "",
        [600.0, 0.0],
    );
    let x = wf.add(Op::ExportDicom(ExportDicom::default()), "", [800.0, 0.0]);
    wf.link(a, 0, s, 0);
    wf.link(s, 0, c, 0);
    wf.link(c, 0, e, 0);
    wf.link(s, 0, e, 1);
    wf.link(c, 0, x, 0);
    assert!(wf.check().ok(), "{:?}", wf.check().errors);
    let wf = Workflow::from_json(&wf.to_json()).expect("reads back");
    let out = run(&wf, &options(&dir, &wf));
    assert!(out.ok(), "{:?}", out.error);

    // The study holds the sCT and the difference, both CT series.
    let study = &out.studies[0].study;
    let scts: Vec<_> = study
        .series
        .iter()
        .filter(|s| s.description.starts_with("sCT"))
        .collect();
    assert_eq!(scts.len(), 2, "the sCT and the difference");
    assert!(scts.iter().all(|s| s.modality == "CT"));
    let report = out
        .reports
        .iter()
        .find(|r| r.title.starts_with("Synthetic CT evaluation"))
        .expect("the evaluation's report");
    let hu = report.tables.iter().find(|t| t.title == "HU").unwrap();
    assert_eq!(hu.rows[0][1], "all");
    let mae: f64 = hu.rows[0][2].parse().unwrap();
    assert!(mae > 0.0);
    assert!(report
        .tables
        .iter()
        .any(|t| t.title == "Water-equivalent thickness"));
    // The export wrote the series the run made, images not asked for.
    let files = walk(&out.run_dir);
    let written_ct = files
        .iter()
        .filter(|p| {
            p.components().any(|c| c.as_os_str() == "Plan")
                && p.extension().is_some_and(|x| x == "dcm")
        })
        .count();
    assert!(written_ct >= 2 * 40, "{written_ct} files exported");
    rust_dicom_station::synth::custom::register(&[]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_setup_from_radiographs_moves_a_copy_of_the_study_where_the_patient_was() {
    let dir = common::target_dir("test_workflow_steps_2d3d");
    let plan = phantom(&dir, "plan", 0.0);
    common::radiographs_into(&plan, [4.0, 0.0, -3.0], "kV");
    let mut wf = Workflow::new("setup");
    let l = wf.add(load(&plan), "Plan", [0.0, 0.0]);
    let img = wf.add(Op::SelectImage(SelectImage::default()), "", [250.0, 0.0]);
    let reg = wf.add(
        Op::Register2d3d(Register2d3d {
            metric: "ncc".into(),
            translations_only: true,
            pixels: 96,
            device: Device::Cpu,
            ..Register2d3d::default()
        }),
        "",
        [500.0, 0.0],
    );
    let copy = wf.add(
        Op::PositionedCopy(PositionedCopy::default()),
        "",
        [750.0, 0.0],
    );
    let target = wf.add(select("TARGET"), "", [500.0, 200.0]);
    let prop = wf.add(Op::Propagate(Propagate::default()), "", [1000.0, 100.0]);
    wf.link(l, 0, img, 0);
    wf.link(img, 0, reg, 0);
    wf.link(img, 0, copy, 0);
    wf.link(reg, 0, copy, 1);
    wf.link(img, 0, target, 0);
    wf.link(copy, 1, prop, 0);
    wf.link(target, 0, prop, 1);
    assert!(wf.check().ok(), "{:?}", wf.check().errors);
    let wf = Workflow::from_json(&wf.to_json()).expect("reads back");
    let out = run(&wf, &options(&dir, &wf));
    assert!(out.ok(), "{:?}", out.error);

    let rep = out
        .reports
        .iter()
        .find(|r| r.title.starts_with("2D/3D registration"))
        .expect("a report");
    let row = rep.tables[0]
        .rows
        .iter()
        .find(|r| r[0].starts_with("Correction, translation"))
        .expect("the correction");
    let t: Vec<f64> = row[1]
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    for (g, want) in t.iter().zip([-4.0, 0.0, 3.0]) {
        assert!((g - want).abs() < 1.0, "{t:?}");
    }
    // Filed with the study, the DRRs beside the radiographs, the files
    // written, and the copy a study of its own with the target on it.
    let plan_study = &out.studies[0].study;
    assert_eq!(plan_study.setups.len(), 1);
    assert_eq!(plan_study.planar_images.len(), 4);
    let files = walk(&out.run_dir);
    assert!(
        files.iter().any(|f| f.ends_with("setup_reg.dcm")),
        "{files:?}"
    );
    assert_eq!(out.studies.len(), 2);
    let landed = names_of(&out, 1);
    assert!(landed.iter().any(|n| n.starts_with("TARGET")), "{landed:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
