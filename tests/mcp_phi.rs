//! Nothing that names the patient leaves the server.
//!
//! The phantom is given a patient name, an ID, a birth date, a physician and
//! an institution; then every tool is run against it (including the error
//! paths: a wrong handle, a missing folder, a cancelled run) under each of
//! the three policies, and every byte the server produced is searched for
//! those values. The in-process `Core::call_public` is what the protocol
//! layer calls, so the frames carry nothing else; the stdio test at the end
//! checks that statement on the real executable.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rust_dicom_station::mcp::config::PhiPolicy;
use rust_dicom_station::mcp::{Config, Core};
use rust_dicom_station::progress::Progress;
use serde_json::{json, Value};

const NAME: &str = "DOE^JANE";
const ID: &str = "PAT-778812";
const BIRTH: &str = "19581117";
/// Every string that must never appear, in any case.
const SECRETS: [&str; 9] = [
    "DOE^JANE",
    "DOE JANE",
    "JANE",
    "studies_Doe",
    "PAT-778812",
    "19581117",
    "GREGORY",
    "Plainsboro",
    "WILSON",
];

fn contains_secret(s: &str) -> Option<&'static str> {
    let lower = s.to_lowercase();
    SECRETS
        .iter()
        .find(|sec| lower.contains(&sec.to_lowercase()))
        .copied()
}

/// A named 4D phantom under a root, with an output folder beside it.
fn fixture(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let dir = common::target_dir(tag);
    // The root's own name carries the patient too: it must not come back.
    let root = dir.join(format!("studies_{}_{}", "Doe", ID));
    std::fs::create_dir_all(&root).unwrap();
    let folder = common::fourd_folder(&root, [0.0, 6.0, 3.0]);
    // Radiographs whose RT Image Label carries the patient's name, as a
    // console that labels images by patient does.
    common::radiographs_into(&folder, [2.0, 0.0, 0.0], "JANE");
    common::name_the_patient(&folder, NAME, ID, BIRTH);
    let out = dir.join("out");
    std::fs::create_dir_all(&out).unwrap();
    (dir, folder, out)
}

fn config(root: &Path, out: &Path, policy: PhiPolicy) -> Config {
    Config {
        roots: vec![root.to_path_buf()],
        output_dir: Some(out.to_path_buf()),
        phi_policy: policy,
        device: "cpu".into(),
        audit_log: false,
        ..Config::default()
    }
}

/// Run one tool, record every byte of the answer, and assert it is clean.
fn call(core: &mut Core, log: &mut Vec<String>, tool: &str, args: Value) -> Result<Value, String> {
    let r = core.call_public(tool, args, &Progress::default());
    let text = match &r {
        Ok(v) => v.to_string(),
        Err(e) => e.as_str().to_string(),
    };
    if let Some(sec) = contains_secret(&text) {
        panic!("{tool}: the answer carries '{sec}': {text}");
    }
    log.push(format!("{tool}: {text}"));
    r.map_err(|e| e.into_string())
}

#[test]
fn the_default_policy_refuses_a_named_dataset_without_saying_the_name() {
    let (dir, folder, out) = fixture("test_mcp_phi_refuse");
    let root = folder.parent().unwrap();
    let mut core = Core::new(config(root, &out, PhiPolicy::Refuse));
    let mut log = Vec::new();
    let err = call(&mut core, &mut log, "open_dataset", json!({"path": folder}))
        .expect_err("a named dataset is refused");
    assert!(
        err.contains("PatientName"),
        "the refusal names the tag: {err}"
    );
    assert!(err.contains("PatientBirthDate"), "{err}");
    assert!(err.contains("anonymize"), "and says what to do: {err}");
    // Nothing is open afterwards.
    let s = call(&mut core, &mut log, "describe_session", json!({})).unwrap();
    assert_eq!(s["datasets"].as_array().unwrap().len(), 0);

    // The anonymizer makes a copy that passes.
    let a = call(&mut core, &mut log, "anonymize", json!({"path": folder})).unwrap();
    assert!(a["files"].as_u64().unwrap() > 100);
    assert!(a["tags_replaced"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == "PatientName"));
    let copy = a["folder"].as_str().unwrap().to_string();
    assert!(
        copy.starts_with("output"),
        "paths are reported relative to a label: {copy}"
    );
    // The real folder, for the next call: under the output folder.
    let real_copy = std::fs::read_dir(out.read_dir().unwrap().next().unwrap().unwrap().path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let d = call(
        &mut core,
        &mut log,
        "open_dataset",
        json!({"path": real_copy}),
    )
    .unwrap();
    assert_eq!(d["phi"]["status"], "anonymized");
    assert_eq!(d["series"].as_array().unwrap().len(), 3);
    assert_eq!(d["fourd_groups"].as_array().unwrap().len(), 1, "{d}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The whole tool set, under one policy, on a named dataset.
fn exercise(policy: PhiPolicy, tag: &str) {
    let (dir, folder, out) = fixture(tag);
    let root = folder.parent().unwrap();
    let mut core = Core::new(config(root, &out, policy));
    let mut log = Vec::new();
    let c = &mut core;
    let l = &mut log;

    // Errors first: they quote paths and handles.
    let _ = call(c, l, "open_dataset", json!({"path": root.join("nowhere")}));
    let _ = call(c, l, "open_dataset", json!({"path": dir.join("out")}));
    let _ = call(c, l, "describe_dataset", json!({"dataset": "ds9"}));
    let _ = call(c, l, "no_such_tool", json!({}));
    let _ = call(
        c,
        l,
        "register",
        json!({"fixed": {"dataset": "x"}, "moving": {"dataset": "y"}}),
    );

    let d = call(c, l, "open_dataset", json!({"path": folder})).unwrap();
    assert_eq!(d["dataset"], "ds1");
    assert_eq!(d["phi"]["status"], "identifying (redacted in memory)");
    assert!(!d.to_string().contains("patient_name"));
    let origin = d["origin"].as_str().unwrap();
    assert!(
        origin.starts_with("root1"),
        "the origin is relative to the root label: {origin}"
    );

    call(c, l, "describe_dataset", json!({"dataset": "ds1"})).unwrap();
    let ls = call(c, l, "list_structures", json!({"dataset": "ds1"})).unwrap();
    assert_eq!(ls["structures"].as_array().unwrap().len(), 3);
    call(c, l, "list_4d_groups", json!({"dataset": "ds1"})).unwrap();
    call(c, l, "describe_session", json!({})).unwrap();

    // Fast tools on the phantom.
    call(
        c,
        l,
        "segment_body",
        json!({"dataset": "ds1", "name": "EXTERNAL"}),
    )
    .unwrap();
    call(
        c,
        l,
        "combine_structures",
        json!({"dataset": "ds1", "op": "subtract", "name": "BODY-TARGET",
               "operands": [{"structure": "BODY"}, {"structure": "TARGET", "margin": {"uniform_mm": 3.0}}]}),
    )
    .unwrap();
    let cmp = call(
        c,
        l,
        "compare_structures",
        json!({"a": {"dataset": "ds1", "structure": "BODY"}, "b": {"dataset": "ds1", "structure": "EXTERNAL"}}),
    )
    .unwrap();
    assert!(cmp["dice"].as_f64().unwrap() > 0.9, "{cmp}");
    let dvh = call(
        c,
        l,
        "compute_dvh",
        json!({"dataset": "ds1", "structures": [{"structure": "TARGET"}], "metrics": ["D95%", "Dmean"],
               "protocol": "TARGET D95% >= 1"}),
    )
    .unwrap();
    assert_eq!(dvh["protocol"]["passed"], 1, "{dvh}");

    // Registration between two phases of the one dataset, then propagation.
    let reg = call(
        c,
        l,
        "register",
        json!({"fixed": {"dataset": "ds1", "series": 2}, "moving": {"dataset": "ds1", "series": 1},
               "method": "elastix_rigid", "levels": 1, "iterations": 40, "samples": 800}),
    )
    .unwrap();
    assert_eq!(reg["reg"], "reg1");
    call(c, l, "describe_registration", json!({"reg": "reg1"})).unwrap();
    let pr = call(
        c,
        l,
        "propagate",
        json!({"reg": "reg1", "structures": [{"structure": "TARGET"}], "to": "fixed"}),
    )
    .unwrap();
    assert!(pr["structures"][0]["voxels"].as_u64().unwrap() > 0, "{pr}");
    call(
        c,
        l,
        "export_registration",
        json!({"reg": "reg1", "step_mm": 10.0}),
    )
    .unwrap();
    // By structures: an unknown one (the error lists what there is), then
    // the target and the body of phase 0 laid onto the 50% phase.
    let _ = call(
        c,
        l,
        "register_structures",
        json!({"fixed": {"dataset": "ds1"}, "moving": {"dataset": "ds1", "series": 2},
               "structures": [{"fixed": "NO_SUCH"}]}),
    );
    let sr = call(
        c,
        l,
        "register_structures",
        json!({"fixed": {"dataset": "ds1", "series": 1}, "moving": {"dataset": "ds1", "series": 2},
               "structures": [{"fixed": "TARGET"}, {"fixed": "BODY", "weight": 0.5}]}),
    )
    .unwrap();
    assert_eq!(sr["structures"].as_array().unwrap().len(), 2, "{sr}");

    // A setup from the two radiographs, named by the patient's name.
    let _ = call(
        c,
        l,
        "register_2d3d",
        json!({"dataset": "ds1", "images": [{"image": "JANE OBLIQUE"}]}),
    );
    let su = call(
        c,
        l,
        "register_2d3d",
        json!({"dataset": "ds1", "series": 1, "images": [{"image": "1"}, {"image": "2"}],
               "metric": "ncc", "pixels": 64, "levels": 2, "dof": "translations", "file_drrs": true}),
    )
    .unwrap();
    assert_eq!(su["setup"], "setup1");
    call(c, l, "describe_setup", json!({"setup": "setup1"})).unwrap();
    call(c, l, "export_setup", json!({"setup": "setup1"})).unwrap();
    call(c, l, "describe_dataset", json!({"dataset": "ds1"})).unwrap();

    // A synthetic CT of phase 0 (the miniature network as a checkpoint
    // folder of the user's), and its evaluation against the 50% phase.
    let key = sct_app();
    let _ = call(
        c,
        l,
        "synthesize_ct",
        json!({"dataset": "ds1", "app": "nope"}),
    );
    let sct = call(
        c,
        l,
        "synthesize_ct",
        json!({"dataset": "ds1", "series": 1, "app": key, "mask": "BODY"}),
    )
    .unwrap();
    let n = sct["series"].as_u64().unwrap();
    call(
        c,
        l,
        "evaluate_sct",
        json!({"dataset": "ds1", "sct": n, "reference": 2, "mask": "covered"}),
    )
    .unwrap();

    // Output: the export names folders after the patient under `allow`,
    // and the answer must still be clean.
    let ex = call(
        c,
        l,
        "export",
        json!({"dataset": "ds1", "format": "rtstruct", "include_images": false}),
    )
    .unwrap();
    assert!(ex["files"].as_u64().unwrap() >= 1, "{ex}");

    // A cancelled run: the error path of a long tool.
    let p = Progress::default();
    p.cancel();
    let r = c.call_public(
        "analyse_motion",
        json!({"dataset": "ds1", "group": "1", "targets": [{"structure": "TARGET"}], "iterations": 5}),
        &p,
    );
    let text = match &r {
        Ok(v) => v.to_string(),
        Err(e) => e.as_str().to_string(),
    };
    assert!(contains_secret(&text).is_none(), "{text}");

    call(c, l, "close_dataset", json!({"dataset": "ds1"})).unwrap();

    // The whole transcript, once more, as one string.
    let all = log.join("\n");
    assert!(contains_secret(&all).is_none());
    assert!(all.len() > 2000, "the tools did answer");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The engine tests' miniature TotalSynth as a checkpoint folder, made and
/// registered once for every test of this file (the registry is the
/// process's).
fn sct_app() -> String {
    static KEY: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    KEY.get_or_init(|| {
        let dir = common::target_dir("test_mcp_phi_checkpoints");
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        std::fs::copy(data.join("synth-unetpp-cv0.pt"), dir.join("CV_0.pt")).unwrap();
        let reg = rust_dicom_station::synth::custom::register(&[dir]);
        (*reg[0].1.as_ref().unwrap()).to_string()
    })
    .clone()
}

#[test]
fn no_tool_leaks_under_redact() {
    exercise(PhiPolicy::Redact, "test_mcp_phi_redact");
}

#[test]
fn no_tool_leaks_under_allow() {
    exercise(PhiPolicy::Allow, "test_mcp_phi_allow");
}

/// The archive tools: a named patient filed in a PACS is listed under an
/// alias, a study is described from its headers with its protocol page
/// told apart from its slices, and opening it goes through the gate.
#[test]
fn the_archive_tools_name_nobody() {
    use rust_dicom_station::archive::Archive;
    let (dir, folder, out) = fixture("test_mcp_phi_archive");
    // The page is a copy of a named slice, so it carries the names too; the
    // dose SR names the patient in its header and a physician in its
    // content. The PACS files the folder under the patient's name.
    common::protocol_page_into(&folder, "Patientenprotokoll");
    common::dose_report_into(&folder, "WILSON^JAMES");
    // A findings PDF whose title names the patient's ID and a physician.
    common::document_into(&folder, "Dr. WILSON");
    let pacs = dir.join("pacs");
    let ar = Archive::new(&pacs);
    ar.import(&folder, &Progress::default()).expect("filed");
    let root = folder.parent().unwrap();
    let mut log = Vec::new();

    let archive_config = |policy| Config {
        archive: pacs.display().to_string(),
        ..config(root, &out, policy)
    };
    let mut core = Core::new(archive_config(PhiPolicy::Refuse));
    let c = &mut core;
    let l = &mut log;

    let ls = call(c, l, "list_archive", json!({})).unwrap();
    assert_eq!(ls["patients"], 1, "{ls}");
    let patient = ls["listing"][0]["patient"].as_str().unwrap().to_string();
    assert!(patient.starts_with("anon_"), "an alias: {patient}");
    let studies = ls["listing"][0]["studies"].as_array().unwrap();
    assert_eq!(studies.len(), 1);
    let study = studies[0]["study"].as_str().unwrap().to_string();
    assert!(studies[0]["modalities"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m == "CT"));
    assert!(
        studies[0].get("date").is_none(),
        "no date, the year at most"
    );
    // The same alias when narrowed to the patient, and a miss is an error.
    let one = call(c, l, "list_archive", json!({"patient": patient})).unwrap();
    assert_eq!(one["patients"], 1);
    let _ = call(c, l, "list_archive", json!({"patient": "anon_000000"})).unwrap_err();
    let _ = call(c, l, "describe_archive_study", json!({"study": "../etc"})).unwrap_err();

    let d = call(c, l, "describe_archive_study", json!({"study": study})).unwrap();
    let series = d["series"].as_array().unwrap();
    let page = series
        .iter()
        .find(|s| s["description"] == "Patientenprotokoll")
        .expect("the protocol page is listed");
    assert_eq!(page["class"], "report", "{page}");
    assert_eq!(page["report"], "protocol page");
    assert_eq!(page["sop_class"], "Secondary Capture Image");
    assert_eq!(page["positioned"], false);
    assert_eq!(page["shown_in"], "the Additional pane, as a table");
    let sr = series
        .iter()
        .find(|s| s["description"] == "Dose Report")
        .expect("the dose SR is listed");
    assert_eq!(sr["class"], "report", "{sr}");
    assert_eq!(sr["sop_class"], "X-Ray Radiation Dose SR");
    let befund = series
        .iter()
        .find(|s| s["description"] == "Befund")
        .expect("the findings PDF is listed");
    assert_eq!(befund["class"], "report", "{befund}");
    assert_eq!(befund["report"], "document");
    assert_eq!(befund["sop_class"], "Encapsulated PDF");
    assert_eq!(befund["shown_in"], "the Additional pane, as its pages");
    assert!(
        series
            .iter()
            .filter(|s| s["class"] == "image volume")
            .count()
            >= 3,
        "{d}"
    );
    assert_eq!(
        series
            .iter()
            .filter(|s| s["class"] == "planar image")
            .count(),
        2,
        "{d}"
    );

    // Refused under the default policy, the tags named, never the values.
    let err = call(c, l, "open_archive_study", json!({"study": &study})).unwrap_err();
    assert!(err.contains("PatientName"), "{err}");

    // Under redact it opens: the radiographs as planar images, the page
    // and the dose SR as reports with their tables, the physician the SR
    // names nowhere.
    let mut core = Core::new(archive_config(PhiPolicy::Redact));
    let c = &mut core;
    let d = call(c, l, "open_archive_study", json!({"study": &study})).unwrap();
    assert_eq!(d["dataset"], "ds1");
    assert_eq!(d["root"], "archive");
    assert!(d["origin"].as_str().unwrap().starts_with("archive"), "{d}");
    let planar = d["planar_images"].as_array().unwrap();
    assert_eq!(planar.len(), 2, "{d}");
    assert!(planar.iter().all(|p| p["kind"] == "projection"));
    let reports = d["reports"].as_array().unwrap();
    assert_eq!(reports.len(), 3, "{d}");
    // The PDF as a document: its facts, never its title.
    let doc = reports
        .iter()
        .find(|r| r["kind"] == "document")
        .expect("the document");
    assert_eq!(doc["label"], "Befund");
    assert_eq!(doc["modality"], "OT");
    assert!(doc["facts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f[0] == "Type" && f[1] == "application/pdf"));
    let page = reports
        .iter()
        .find(|r| r["kind"] == "protocol_page")
        .expect("the page");
    assert_eq!(page["modality"], "CT");
    assert_eq!(page["page"], true);
    let t = &page["tables"][0];
    assert_eq!(t["columns"][4], "CTDIvol mGy");
    assert_eq!(t["rows"][1][0], "AbdRoutine");
    let dose = reports
        .iter()
        .find(|r| r["kind"] == "dose_report")
        .expect("the dose report");
    let t = &dose["tables"][0];
    assert_eq!(t["rows"][1][12], "389.33", "{t}");
    assert!(dose["facts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|f| f[0] == "Total DLP"));
    let tree = &dose["tables"][1];
    assert!(
        tree["rows"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r[1] == "(name)"),
        "the physician's item is there, the name is not: {tree}"
    );
    // The whole log, once more, for anything that names the patient.
    for line in l.iter() {
        assert!(contains_secret(line).is_none(), "{line}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The real executable over stdio: every frame it writes is searched.
#[test]
fn the_stdio_frames_are_clean_too() {
    let (dir, folder, out) = fixture("test_mcp_phi_stdio");
    let root = folder.parent().unwrap();
    let cfg = dir.join("mcp.toml");
    std::fs::write(
        &cfg,
        format!(
            "roots = [{:?}]\noutput_dir = {:?}\nphi_policy = \"redact\"\ndevice = \"cpu\"\naudit_log = false\n",
            root.to_string_lossy(),
            out.to_string_lossy()
        ),
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_rds-mcp"))
        .arg("--config")
        .arg(&cfg)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("rds-mcp starts");
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut reader = BufReader::new(stdout);
    let mut frames: Vec<String> = Vec::new();
    let send = |stdin: &mut std::process::ChildStdin, v: Value| {
        stdin.write_all(v.to_string().as_bytes()).unwrap();
        stdin.write_all(b"\n").unwrap();
        stdin.flush().unwrap();
    };
    let mut recv = |frames: &mut Vec<String>, id: i64| -> Value {
        loop {
            let mut line = String::new();
            let n = reader.read_line(&mut line).unwrap();
            assert!(n > 0, "the server closed the pipe");
            let v: Value = serde_json::from_str(line.trim()).expect("a JSON-RPC frame");
            frames.push(line.clone());
            if v["id"] == json!(id) {
                return v;
            }
        }
    };
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}}),
    );
    let init = recv(&mut frames, 1);
    assert_eq!(init["result"]["serverInfo"]["name"], "rds-mcp");
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    );
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    );
    let tools = recv(&mut frames, 2);
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"open_dataset"));
    assert!(names.contains(&"analyse_motion_async"));
    assert!(names.contains(&"list_jobs"));
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "open_dataset", "arguments": {"path": folder}, "_meta": {"progressToken": 7}}}),
    );
    let opened = recv(&mut frames, 3);
    assert_eq!(
        opened["result"]["structuredContent"]["dataset"], "ds1",
        "{opened}"
    );
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "describe_dataset", "arguments": {"dataset": "ds7"}}}),
    );
    let bad = recv(&mut frames, 4);
    assert_eq!(bad["result"]["isError"], true);
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 5, "method": "prompts/get",
        "params": {"name": "heart_target_propagation", "arguments": {"cct": folder.to_string_lossy()}}}),
    );
    let prompt = recv(&mut frames, 5);
    assert!(prompt["result"]["messages"][0]["content"]["text"]
        .as_str()
        .unwrap()
        .contains("root1"));
    send(
        &mut stdin,
        json!({"jsonrpc": "2.0", "id": 6, "method": "resources/read", "params": {"uri": "rds://docs/mcp"}}),
    );
    let res = recv(&mut frames, 6);
    assert!(res["result"]["contents"][0]["text"].as_str().is_some());
    drop(stdin);
    let status = child.wait().unwrap();
    assert!(status.success() || status.code().is_none(), "{status}");
    for f in &frames {
        if let Some(sec) = contains_secret(f) {
            panic!("a frame carries '{sec}': {f}");
        }
    }
    assert!(frames.len() >= 6);
    let _ = std::fs::remove_dir_all(&dir);
}
