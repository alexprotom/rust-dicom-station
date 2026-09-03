//! The workflow layer runs the 4D pipeline headless and gets the numbers the
//! viewer gets.
//!
//! This is the guard for the move of `run_motion` and `run_group` out of the
//! tool windows: the phantom's target sits at y = 0, 6 and 3 mm on the three
//! phases inside a body that does not move, so the deformable model has to
//! find those displacements and the rigid one has to find none. The ITV has
//! to be larger than the target and hold every phase's copy of it.

mod common;

use rust_dicom_station::loader;
use rust_dicom_station::motion::MotionModel;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::registration::{RegMethod, RegParams};
use rust_dicom_station::workflow::{self, group, motion, select};

const SHIFTS: [f64; 3] = [0.0, 6.0, 3.0];

#[test]
fn the_motion_pipeline_recovers_the_phantoms_target_motion() {
    let dir = common::target_dir("test_workflow_motion");
    let folder = common::fourd_folder(&dir, SHIFTS);
    let study = loader::load_directory(&folder, &Progress::default()).expect("the 4D folder loads");
    assert_eq!(study.fourd_groups.len(), 1, "one 4D group is recognised");
    let g = &study.fourd_groups[0];
    let phases = workflow::phases_of(g, &study.series).expect("three phases resolve");
    assert_eq!(phases.len(), 3);

    let target = select::find(&study, "TARGET", None).expect("TARGET exists");
    let body = select::find(&study, "body", None).expect("BODY is found case-insensitively");
    let req = motion::MotionRequest {
        run_name: "test".into(),
        slot_name: "A".into(),
        patient: "phantom".into(),
        group_name: g.name.clone(),
        study_uid: g.study_uid.clone(),
        phases,
        reference: 0,
        targets: vec![target],
        ref_struct: Some(body),
        models: vec![MotionModel::Rigid, MotionModel::Deformable],
        build_itv: true,
        itv_margin_mm: 0.0,
        keep_phase_segs: true,
        params: RegParams {
            method: RegMethod::ElastixRigid,
            levels: 2,
            iterations: 150,
            samples: 2000,
            grid_spacing_mm: 16.0,
            fixed_threshold: -500.0,
            ..RegParams::default()
        },
    };
    let t0 = std::time::Instant::now();
    let out = motion::run(req, &Progress::default()).expect("the pipeline runs");
    eprintln!("pipeline: {:.1} s", t0.elapsed().as_secs_f64());
    let r = &out.report;

    let deformable = r
        .tracks
        .iter()
        .find(|t| t.model == MotionModel::Deformable)
        .expect("a deformable track");
    let mags = deformable.magnitudes();
    eprintln!("deformable |d| per phase: {mags:?}");
    for (i, expect) in SHIFTS.iter().enumerate() {
        assert!(
            (mags[i] - expect).abs() < 1.5,
            "phase {i}: deformable displacement {:.2} mm, expected {expect} mm",
            mags[i]
        );
    }
    let rigid = r
        .tracks
        .iter()
        .find(|t| t.model == MotionModel::Rigid)
        .expect("a rigid track");
    let rigid_max = rigid.magnitudes().iter().cloned().fold(0.0, f64::max);
    eprintln!("rigid max |d|: {rigid_max:.2} mm");
    assert!(
        rigid_max < 1.5,
        "the body does not move, so the rigid model finds little: {rigid_max:.2} mm"
    );

    // The reference structure's track exists and the correlation ran.
    assert_eq!(r.reference_tracks.len(), 2);
    assert_eq!(r.qa.len(), 4, "two models on two non-reference phases");

    // The ITV: one per model, larger than the target and made of it.
    let itv = out.itv_series.expect("ITVs were built");
    assert_eq!(itv.segs.len(), 2);
    let target_cm3 = r.tracks[0].samples[0].volume_cm3;
    for i in &r.itvs {
        assert!(
            i.volume_cm3 >= target_cm3 * 0.98,
            "{} ({:.1} cm³) is not smaller than the target ({target_cm3:.1} cm³)",
            i.seg_name,
            i.volume_cm3
        );
    }
    let def_itv = r
        .itvs
        .iter()
        .find(|i| i.model == MotionModel::Deformable)
        .unwrap();
    assert!(
        def_itv.volume_cm3 > target_cm3 * 1.15,
        "a 6 mm excursion of a 25 mm sphere grows the ITV noticeably: {:.1} vs {target_cm3:.1} cm³",
        def_itv.volume_cm3
    );
    // Per-phase series were kept, one per non-reference phase.
    assert_eq!(out.phase_series.len(), 2);
    let ser = itv.into_seg_series(&out.study_uid);
    assert_eq!(ser.segs.len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn one_volume_onto_every_phase_reuses_cached_transforms() {
    let dir = common::target_dir("test_workflow_group");
    let folder = common::fourd_folder(&dir, SHIFTS);
    let study = loader::load_directory(&folder, &Progress::default()).expect("loads");
    let g = &study.fourd_groups[0];
    let phases = workflow::phases_of(g, &study.series).unwrap();
    let target = select::find(&study, "TARGET", None).unwrap();
    let subjects = || vec![target.subject_on(&study.volume.grid()).unwrap()];
    let params = RegParams {
        method: RegMethod::PlastimatchBSpline,
        levels: 2,
        iterations: 60,
        grid_spacing_mm: 16.0,
        ..RegParams::default()
    };
    let req = group::GroupRequest {
        src_vol: study.volume.clone(),
        subjects: subjects(),
        phases: phases.clone(),
        cached: vec![None; 3],
        params: params.clone(),
        group_name: g.name.clone(),
        group: 0,
        moving_slot: 0,
        moving_series_uid: study.series[0].uid.clone(),
    };
    let out = group::run(req, &Progress::default()).expect("the group run works");
    assert_eq!(out.phases.len(), 3);
    for ph in &out.phases {
        assert_eq!(ph.items.len(), 1);
        assert!(ph.items[0].voxels > 0, "the target lands on {}", ph.label);
        assert!(ph.seg_series(&g.name).is_some());
        assert_ne!(ph.metric_line, "transform reused");
    }
    // The same run with the transforms handed back skips every registration.
    let cached: Vec<_> = out
        .phases
        .iter()
        .map(|p| Some(p.transform.clone()))
        .collect();
    let req = group::GroupRequest {
        src_vol: study.volume.clone(),
        subjects: subjects(),
        phases,
        cached,
        params,
        group_name: g.name.clone(),
        group: 0,
        moving_slot: 0,
        moving_series_uid: study.series[0].uid.clone(),
    };
    let again = group::run(req, &Progress::default()).unwrap();
    assert!(again
        .phases
        .iter()
        .all(|p| p.metric_line == "transform reused"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn structures_are_found_by_name_and_set() {
    let dir = common::target_dir("test_workflow_select");
    let folder = common::fourd_folder(&dir, SHIFTS);
    let study = loader::load_directory(&folder, &Progress::default()).unwrap();
    let all = select::list(&study);
    let names: Vec<&str> = all.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, ["BODY", "TARGET", "CORD"]);
    assert!(select::find(&study, "cord", None).is_some());
    assert!(select::find(&study, "CORD", Some("no such set")).is_none());
    assert!(select::find(&study, "LIVER", None).is_none());
    let grid = study.volume.grid();
    let mask = select::find(&study, "TARGET", None)
        .unwrap()
        .mask_on(&grid)
        .unwrap();
    assert!(mask.contains(&1));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_structure_is_found_on_its_own_series_first() {
    let dir = common::target_dir("test_workflow_on_series");
    let folder = common::fourd_folder(&dir, SHIFTS);
    let study = loader::load_directory(&folder, &Progress::default()).unwrap();
    let phases = workflow::phases_of(&study.fourd_groups[0], &study.series).unwrap();
    // The structure set was drawn on phase 0 and says so; asked about phase
    // 0 it is found by that reference, asked about another phase the search
    // widens to the study rather than coming back empty.
    for (label, series) in &phases {
        let body = select::find_on_series(&study, "body", &series.uid, "");
        assert!(body.is_some(), "BODY resolves for phase {label}");
    }
    assert!(select::find_on_series(&study, "LIVER", &phases[0].1.uid, "").is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_volume_in_another_frame_is_anchored_on_a_structure_and_lands() {
    use rust_dicom_station::geometry::Vec3;
    use rust_dicom_station::motion;
    use rust_dicom_station::volume::Volume;
    use rust_dicom_station::workflow::anchored;
    use std::sync::Arc;

    let dir = common::target_dir("test_workflow_anchored");
    let folder = common::fourd_folder(&dir, SHIFTS);
    let study = loader::load_directory(&folder, &Progress::default()).unwrap();
    let g = &study.fourd_groups[0];
    let phases = workflow::phases_of(g, &study.series).unwrap();

    // The source: the phase-0 phantom filed in another frame of reference,
    // 600 mm away in patient coordinates - the way a cardiac CT and a 4DCT
    // of one patient sit. Its structures go with it, since masks live in
    // voxel space.
    let offset = Vec3::new(30.0, -250.0, 600.0);
    // Phase 0 explicitly: the displayed series of a 4D study is whichever
    // the loader put first, and the target's expected positions below are
    // relative to phase 0.
    let (phase0, _, _) = loader::load_series_volume(&phases[0].1, &Progress::default()).unwrap();
    let mut src: Volume = phase0;
    let grid0 = src.grid();
    src.origin = src.origin + offset;
    src.frame_of_reference_uid = "1.2.3.4.5.6.7.8.9".into();
    let src = Arc::new(src);
    let src_grid = src.grid();
    let on_source = |name: &str| {
        // Rasterized on phase 0's lattice, then filed on the shifted one:
        // the same voxels, so the structure moved with the image.
        let mask = select::find(&study, name, None)
            .unwrap()
            .mask_on(&grid0)
            .unwrap();
        select::Structure {
            name: name.to_string(),
            color: [200, 40, 40],
            source: select::Source::Mask {
                mask,
                grid: src_grid.clone(),
            },
        }
    };
    let src_anchor = on_source("BODY");
    let target = on_source("TARGET");
    let subjects = vec![target.subject_on(&src_grid).unwrap()];
    let anchored_phases: Vec<_> = phases
        .iter()
        .map(|(label, series)| anchored::AnchoredPhase {
            label: label.clone(),
            series: series.clone(),
            anchor: select::find_on_series(&study, "BODY", &series.uid, "").unwrap(),
        })
        .collect();
    let base = RegParams {
        levels: 2,
        iterations: 150,
        samples: 2000,
        grid_spacing_mm: 16.0,
        ..RegParams::default()
    };
    let req = anchored::AnchoredRequest {
        src_vol: src.clone(),
        src_anchor,
        subjects,
        phases: anchored_phases,
        margin_mm: 10.0,
        rigid: anchored::default_rigid(&base),
        deformable: Some(anchored::default_deformable(&base)),
        group_name: g.name.clone(),
        group: 0,
        moving_slot: 0,
        moving_series_uid: study.series[0].uid.clone(),
    };
    let t0 = std::time::Instant::now();
    let out = anchored::run(req, &Progress::default()).expect("the anchored run works");
    eprintln!("anchored run: {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!(out.group.phases.len(), 3);
    assert_eq!(out.qa.len(), 3);

    let mut centroids = Vec::new();
    for (ph, qa) in out.group.phases.iter().zip(&out.qa) {
        eprintln!("{}  |  {}", ph.metric_line, qa.line());
        assert!(
            (qa.initial_shift_mm - offset.length()).abs() < 5.0,
            "the initialisation closed the frame offset: {:.1} vs {:.1} mm",
            qa.initial_shift_mm,
            offset.length()
        );
        let o = qa.overlap.as_ref().expect("the anchor landed");
        assert!(
            o.dice > 0.9,
            "phase {}: the body lands on the body (Dice {:.3})",
            ph.label,
            o.dice
        );
        assert_eq!(qa.verdict(), "good");
        // The target and the anchor both travelled; the target is item 0.
        assert_eq!(ph.items.len(), 2);
        let t = &ph.items[0];
        assert_eq!(t.name, "TARGET");
        assert!(t.voxels > 0, "the target lands on {}", ph.label);
        assert!(
            (t.result_cm3 - t.source_cm3).abs() / t.source_cm3 < 0.3,
            "volume roughly preserved on {}: {:.1} -> {:.1}",
            ph.label,
            t.source_cm3,
            t.result_cm3
        );
        centroids.push(motion::centroid_mm(&t.mask, &ph.grid).unwrap());
        assert!(ph.seg_series(&g.name).is_some());
    }
    // The deformable refinement on the body region carries the target to
    // where each phase has it: y = 0, 6, 3 mm relative to phase 0.
    for (i, expect) in SHIFTS.iter().enumerate() {
        let dy = centroids[i].y - centroids[0].y;
        eprintln!("phase {i}: target dy = {dy:.2} mm (expected {expect})");
        assert!(
            (dy - expect).abs() < 2.0,
            "phase {i}: the target landed {dy:.2} mm away, expected {expect} mm"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
