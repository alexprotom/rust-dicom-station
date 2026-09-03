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
const SECRETS: [&str; 8] = [
    "DOE^JANE",
    "DOE JANE",
    "JANE",
    "studies_Doe",
    "PAT-778812",
    "19581117",
    "GREGORY",
    "Plainsboro",
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

#[test]
fn no_tool_leaks_under_redact() {
    exercise(PhiPolicy::Redact, "test_mcp_phi_redact");
}

#[test]
fn no_tool_leaks_under_allow() {
    exercise(PhiPolicy::Allow, "test_mcp_phi_allow");
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
