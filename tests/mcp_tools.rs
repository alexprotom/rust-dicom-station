//! The tools do what they say, end to end on the phantom: the sequence of
//! the `heart_target_propagation` prompt, minus the network-bound organ
//! segmentation, run through `Core` exactly as the protocol layer runs it.
//!
//! The phantom's target moves 0 / 6 / 3 mm between the phases of the 4D
//! group, so `analyse_motion` has numbers to be right about; the rest of the
//! chain (register, propagate, propagate_to_group, compare, DVH, export,
//! re-open) is checked for consistency of handles and volumes.

mod common;

use std::path::Path;

use rust_dicom_station::mcp::config::PhiPolicy;
use rust_dicom_station::mcp::{tool_specs, Config, Core};
use rust_dicom_station::progress::Progress;
use serde_json::{json, Value};

fn call(core: &mut Core, tool: &str, args: Value) -> Value {
    match core.call_public(tool, args, &Progress::default()) {
        Ok(v) => v,
        Err(e) => panic!("{tool} failed: {}", e.as_str()),
    }
}

fn fail(core: &mut Core, tool: &str, args: Value) -> String {
    match core.call_public(tool, args, &Progress::default()) {
        Ok(v) => panic!("{tool} should have failed, got {v}"),
        Err(e) => e.into_string(),
    }
}

fn core_for(dir: &Path) -> Core {
    let out = dir.join("out");
    std::fs::create_dir_all(&out).unwrap();
    Core::new(Config {
        roots: vec![dir.to_path_buf()],
        output_dir: Some(out),
        // The phantom carries a name; redact it so the run can proceed.
        phi_policy: PhiPolicy::Redact,
        device: "cpu".into(),
        audit_log: false,
        ..Config::default()
    })
}

#[test]
fn every_tool_has_a_schema_and_a_description() {
    let specs = tool_specs();
    assert!(specs.len() >= 20);
    for s in &specs {
        assert!(s.schema.is_object(), "{}: schema is an object", s.name);
        assert!(s.description.len() > 40, "{}: describes itself", s.name);
        assert!(
            s.name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
            "{}: snake_case",
            s.name
        );
    }
    let names: Vec<&str> = specs.iter().map(|s| s.name).collect();
    for must in [
        "open_dataset",
        "segment_organs",
        "register",
        "propagate",
        "propagate_to_group",
        "analyse_motion",
        "compute_dvh",
        "export",
        "anonymize",
    ] {
        assert!(names.contains(&must), "{must} is a tool");
    }
}

#[test]
fn the_configuration_rules_are_enforced() {
    let dir = common::target_dir("test_mcp_tools_cfg");
    let folder = common::fourd_folder(&dir, [0.0, 6.0, 3.0]);
    // No roots at all.
    let mut none = Core::new(Config {
        device: "cpu".into(),
        audit_log: false,
        ..Config::default()
    });
    let e = fail(&mut none, "open_dataset", json!({"path": folder}));
    assert!(e.contains("no roots"), "{e}");
    // A root that does not contain the folder.
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir_all(&elsewhere).unwrap();
    let mut other = Core::new(Config {
        roots: vec![elsewhere],
        device: "cpu".into(),
        audit_log: false,
        ..Config::default()
    });
    let e = fail(&mut other, "open_dataset", json!({"path": folder}));
    assert!(e.contains("outside"), "{e}");
    // Too many datasets.
    let mut one = Core::new(Config {
        roots: vec![dir.clone()],
        max_open_datasets: 1,
        phi_policy: PhiPolicy::Redact,
        device: "cpu".into(),
        audit_log: false,
        ..Config::default()
    });
    call(&mut one, "open_dataset", json!({"path": folder}));
    let e = fail(&mut one, "open_dataset", json!({"path": folder}));
    assert!(e.contains("maximum"), "{e}");
    // No output folder: nothing can be written.
    let e = fail(&mut one, "export", json!({"dataset": "ds1"}));
    assert!(e.contains("output_dir"), "{e}");
    // Unknown arguments are refused rather than ignored.
    let e = fail(
        &mut one,
        "describe_dataset",
        json!({"dataset": "ds1", "verbose": true}),
    );
    assert!(e.contains("arguments"), "{e}");
    // Model downloads are off: segment_organs says so instead of fetching.
    let e = fail(
        &mut one,
        "segment_organs",
        json!({"dataset": "ds1", "variant": "fast"}),
    );
    assert!(
        e.contains("allow_model_download") || e.contains("weights"),
        "{e}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_heart_sequence_runs_on_the_phantom() {
    let dir = common::target_dir("test_mcp_tools_heart");
    let folder = common::fourd_folder(&dir, [0.0, 6.0, 3.0]);
    let mut core = core_for(&dir);
    let c = &mut core;

    // 1. Open. The same folder plays the planning CT (ds1) and the 4DCT
    // (ds2), the way a planning CT and its 4D acquisition arrive together.
    let d1 = call(c, "open_dataset", json!({"path": folder}));
    assert_eq!(d1["dataset"], "ds1");
    assert_eq!(d1["doses"].as_array().unwrap().len(), 1);
    assert_eq!(d1["plans"].as_array().unwrap().len(), 1);
    let groups = call(c, "list_4d_groups", json!({"dataset": "ds1"}));
    assert_eq!(groups["groups"][0]["members"].as_array().unwrap().len(), 3);
    let d2 = call(c, "open_dataset", json!({"path": folder}));
    assert_eq!(d2["dataset"], "ds2");

    // 2. The "heart" stands in for by the body outline here.
    let body = call(
        c,
        "segment_body",
        json!({"dataset": "ds1", "name": "HEART"}),
    );
    assert!(body["volume_cm3"].as_f64().unwrap() > 100.0);
    let ls = call(c, "list_structures", json!({"dataset": "ds1"}));
    assert!(ls["structures"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s["name"] == "HEART" && s["kind"] == "segment"));

    // 3. Rigid, then a local deformable refinement inside the region.
    let rigid = call(
        c,
        "register",
        json!({"fixed": {"dataset": "ds1"}, "moving": {"dataset": "ds2", "series": 2},
               "method": "elastix_rigid", "levels": 2, "iterations": 80, "samples": 1500}),
    );
    assert_eq!(rigid["reg"], "reg1");
    assert!(rigid["analysis"]["final_metric"].as_f64().is_some());
    let def = call(
        c,
        "register",
        json!({"fixed": {"dataset": "ds1"}, "moving": {"dataset": "ds2", "series": 2},
               "method": "elastix_bspline", "start": "reg1", "levels": 2, "iterations": 80,
               "samples": 1500, "grid_spacing_mm": 16.0,
               "region": {"structure": "TARGET", "margin_mm": 15.0}}),
    );
    assert_eq!(def["reg"], "reg2");
    assert_eq!(def["analysis"]["region"], "TARGET");
    // A refinement on another pair is refused.
    let e = fail(
        c,
        "register",
        json!({"fixed": {"dataset": "ds2"}, "moving": {"dataset": "ds1"},
               "method": "elastix_bspline", "start": "reg1"}),
    );
    assert!(e.contains("same pair"), "{e}");

    // 4. Propagate ds1's target onto the moving phase (target at y = 6 mm
    // there): it has to land about 6 mm away from where it started.
    let pr = call(
        c,
        "propagate",
        json!({"reg": "reg2", "structures": [{"structure": "TARGET"}], "to": "moving"}),
    );
    let item = &pr["structures"][0];
    assert!(item["voxels"].as_u64().unwrap() > 0, "{pr}");
    let src = item["source_cm3"].as_f64().unwrap();
    let dst = item["result_cm3"].as_f64().unwrap();
    assert!(
        (dst - src).abs() / src < 0.3,
        "volume roughly preserved: {src} -> {dst}"
    );
    let landed = item["landed_as"].as_str().unwrap().to_string();
    assert_eq!(landed, "TARGET (from ds1)");
    let cmp = call(
        c,
        "compare_structures",
        json!({"a": {"dataset": "ds2", "series": 2, "structure": landed},
               "b": {"dataset": "ds1", "structure": "TARGET"}}),
    );
    let shift = cmp["centroid_shift_norm_mm"].as_f64().unwrap();
    assert!(
        (shift - 6.0).abs() < 2.0,
        "the target moved about 6 mm: {cmp}"
    );
    assert_eq!(cmp["same_frame_of_reference"], true);

    // 5. The 4D pipeline on ds1's group, target and "heart".
    let run = call(
        c,
        "analyse_motion",
        json!({"dataset": "ds1", "group": "1", "targets": [{"structure": "TARGET", "set": "SynthStructs"}],
               "reference_structure": {"structure": "HEART"},
               "levels": 2, "iterations": 150, "samples": 2000, "grid_spacing_mm": 16.0}),
    );
    assert_eq!(run["run"], "run1");
    let tracks = run["report"]["tracks"].as_array().unwrap();
    let deformable = tracks
        .iter()
        .find(|t| t["model"] == "deformable")
        .expect("a deformable track");
    let p2p = deformable["peak_to_peak_mm"].as_f64().unwrap();
    assert!(
        (p2p - 6.0).abs() < 1.5,
        "peak-to-peak {p2p} mm, expected about 6"
    );
    assert!(!run["report"]["itvs"].as_array().unwrap().is_empty());
    assert_eq!(run["report"]["qa"].as_array().unwrap().len(), 4);
    let rep = call(c, "motion_report", json!({"run": "run1"}));
    assert!(rep["csv"].as_str().unwrap().contains("table"));
    assert!(
        !rep.to_string().contains("PHANTOM"),
        "the header carries the handle, not the name"
    );

    // One structure onto every phase, then again with cached transforms.
    let g1 = call(
        c,
        "propagate_to_group",
        json!({"dataset": "ds1", "group": "1", "structures": [{"structure": "HEART"}],
               "method": "plastimatch_bspline", "levels": 2, "iterations": 40, "grid_spacing_mm": 16.0}),
    );
    assert_eq!(g1["greg"], "greg1");
    assert_eq!(g1["transforms_reused"], 0);
    assert_eq!(g1["phases"].as_array().unwrap().len(), 3);
    let g2 = call(
        c,
        "propagate_to_group",
        json!({"dataset": "ds1", "group": "1", "structures": [{"structure": "CORD"}]}),
    );
    assert_eq!(g2["transforms_reused"], 3, "{g2}");
    assert_eq!(g2["greg"], "greg2");

    // 6. Dose.
    let itv_name = run["report"]["itvs"][0]["structure"]
        .as_str()
        .unwrap()
        .to_string();
    let dvh = call(
        c,
        "compute_dvh",
        json!({"dataset": "ds1", "structures": [{"structure": "TARGET", "set": "SynthStructs"}, {"structure": itv_name}],
               "metrics": ["D95%", "Dmean", "V20Gy"], "protocol": "TARGET D95% >= 50\nITV* Dmean >= 1\nCORD Dmax < 5",
               "include_curves": true}),
    );
    assert_eq!(dvh["structures"].as_array().unwrap().len(), 2);
    assert_eq!(dvh["protocol"]["constraints"], 3);
    assert!(dvh["curves_csv"].as_str().unwrap().lines().count() > 10);

    // 7. Export, re-open the export, and the session's bookkeeping.
    let ex = call(
        c,
        "export",
        json!({"dataset": "ds1", "format": "seg", "include_images": true, "folder": "planning"}),
    );
    assert!(ex["files"].as_u64().unwrap() > 120, "{ex}");
    let real_out = dir.join("out");
    let session_dir = std::fs::read_dir(&real_out)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let re = call(
        c,
        "open_dataset",
        json!({"path": session_dir.join("planning")}),
    );
    assert_eq!(re["dataset"], "ds3");
    assert!(
        re["segmentation_series"].as_array().unwrap().len() >= 3,
        "the SEG objects came back: {re}"
    );
    call(
        c,
        "export_registration",
        json!({"reg": "reg2", "step_mm": 8.0}),
    );
    let s = call(c, "describe_session", json!({}));
    assert_eq!(s["datasets"].as_array().unwrap().len(), 3);
    assert_eq!(s["registrations"].as_array().unwrap().len(), 2);
    // The second group run replaced the first's transforms (same group, same
    // moving series): one entry, the newest handle.
    assert_eq!(s["group_registrations"].as_array().unwrap().len(), 1);
    assert_eq!(s["group_registrations"][0]["greg"], "greg2");
    call(c, "close_dataset", json!({"dataset": "ds2"}));
    let s = call(c, "describe_session", json!({}));
    assert_eq!(
        s["registrations"].as_array().unwrap().len(),
        0,
        "regs involving ds2 went with it"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn paths_may_be_given_the_way_the_server_reports_them() {
    let dir = common::target_dir("test_mcp_tools_labels");
    let folder = common::fourd_folder(&dir, [0.0, 6.0, 3.0]);
    let mut core = core_for(&dir);
    let c = &mut core;
    // The root by its label, with either separator.
    let d = call(c, "open_dataset", json!({"path": "root1/4dct"}));
    assert_eq!(d["dataset"], "ds1");
    let d = call(c, "open_dataset", json!({"path": "root1\\4dct"}));
    assert_eq!(d["dataset"], "ds2");
    call(c, "close_dataset", json!({"dataset": "ds2"}));
    // What anonymize answers is what open_dataset takes next.
    let an = call(c, "anonymize", json!({"path": folder, "folder": "copy"}));
    let reported = an["folder"].as_str().unwrap().to_string();
    assert!(
        reported.starts_with("output/session"),
        "the folder is reported under its label: {reported}"
    );
    let d = call(c, "open_dataset", json!({"path": reported}));
    assert_eq!(d["dataset"], "ds3", "handles are never reused");
    assert_eq!(d["phi"]["status"], "anonymized");
    // The series descriptions survive the anonymizer *and* the redactor:
    // "4DCT" is a word, not a name.
    let desc: Vec<String> = d["series"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["description"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        desc.iter().any(|s| s.contains("4DCT")),
        "descriptions are not redacted: {desc:?}"
    );
    // A label that is not a folder is refused, as is a stranger's label.
    let e = fail(c, "open_dataset", json!({"path": "root2/4dct"}));
    assert!(e.contains("does not exist") || e.contains("outside"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_anchored_group_run_reports_the_check() {
    let dir = common::target_dir("test_mcp_tools_anchor");
    let folder = common::fourd_folder(&dir, [0.0, 6.0, 3.0]);
    let mut core = core_for(&dir);
    let c = &mut core;
    call(c, "open_dataset", json!({"path": folder}));

    // register accepts a structure as its start.
    let r = call(
        c,
        "register",
        json!({"fixed": {"dataset": "ds1", "series": 2}, "moving": {"dataset": "ds1", "series": 1},
               "method": "elastix_rigid", "levels": 2, "iterations": 40, "samples": 1500,
               "init": "BODY"}),
    );
    assert_eq!(r["reg"], "reg1");
    let e = fail(
        c,
        "register",
        json!({"fixed": {"dataset": "ds1", "series": 2}, "moving": {"dataset": "ds1", "series": 1},
               "init": "LIVER"}),
    );
    assert!(e.contains("LIVER"), "{e}");

    // The anchored run: phase-0's volume onto every phase, anchored on the
    // body, carrying the target.
    let g = call(
        c,
        "propagate_to_group",
        json!({"dataset": "ds1", "group": "1", "source_series": 1,
               "structures": [{"structure": "TARGET"}], "anchor": {"structure": "BODY"},
               "anchor_margin_mm": 10.0, "levels": 2, "iterations": 100, "samples": 2000,
               "grid_spacing_mm": 16.0}),
    );
    assert_eq!(g["greg"], "greg1");
    assert_eq!(g["anchor"], "BODY");
    assert_eq!(g["stages"], "centroids, rigid, deformable");
    let phases = g["phases"].as_array().unwrap();
    assert_eq!(phases.len(), 3);
    for ph in phases {
        let check = &ph["anchor_check"];
        assert_eq!(check["anchor"], "BODY");
        assert!(check["dice"].as_f64().unwrap() > 0.9, "{ph}");
        assert_eq!(check["verdict"], "good");
        assert!(check["rigid"].as_str().unwrap().contains("MSD"));
        assert!(check["deformable"].is_string());
        let items = ph["structures"].as_array().unwrap();
        assert_eq!(items.len(), 2, "the target and the anchor travelled");
        assert!(items.iter().all(|it| it["voxels"].as_u64().unwrap() > 0));
    }
    assert!(g["worst_anchor_dice"].as_f64().unwrap() > 0.9);
    // Rigid only stops after the rigid stage.
    let g = call(
        c,
        "propagate_to_group",
        json!({"dataset": "ds1", "group": "1", "source_series": 1,
               "anchor": {"structure": "BODY"}, "rigid_only": true,
               "levels": 2, "iterations": 40, "samples": 1500}),
    );
    assert_eq!(g["stages"], "centroids, rigid");
    assert!(g["phases"][0]["anchor_check"]["deformable"].is_null());
    // An anchor no phase carries is refused by name.
    let e = fail(
        c,
        "propagate_to_group",
        json!({"dataset": "ds1", "group": "1", "anchor": {"structure": "LIVER"}}),
    );
    assert!(e.contains("LIVER"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}
