//! The PACS server, driven over HTTPS the way a remote station drives it.
//!
//! Each test starts a real `rds-pacs` server in-process (its own runtime,
//! TLS, a free loopback port, state and archive folders of its own under
//! `target/`) over the synthetic phantom study, and talks to it only through
//! [`Remote`], the client the viewer uses. Beyond the round trips, the suite
//! asserts what makes the link a *secure* one: nothing without a token, a
//! role is a ceiling, a pairing code works once and guessing is throttled,
//! a revoked token stops at once, a changed certificate is refused before
//! the token leaves the client, nothing is reachable through a URL that
//! names a path, and an oversized upload leaves the archive untouched.
//!
//! Needs the `pacs-server` feature:
//! `cargo test --features pacs-server --test pacs_server`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use rust_dicom_station::archive::{self, Archive};
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::pacs::client::{failure_of, Failure, Remote, Trust};
use rust_dicom_station::pacs::config::Config;
use rust_dicom_station::pacs::local::{self, Paths};
use rust_dicom_station::pacs::protocol::Role;
use rust_dicom_station::pacs::server;
use rust_dicom_station::progress::Progress;

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A server folder with the phantom filed into its archive. Returns the
/// folder, the generated study's folder and its file count.
fn served_phantom(tag: &str) -> (PathBuf, PathBuf, usize) {
    let root = scratch(tag);
    let src = root.join("phantom");
    let n = gen_test_data::generate(&src, &GenParams::default(), &Progress::default())
        .expect("the phantom is written");
    Archive::new(root.join("archive"))
        .import(&src, &Progress::default())
        .expect("the phantom is filed");
    (root, src, n)
}

fn config_for(root: &Path) -> Config {
    Config {
        name: "Test PACS".into(),
        bind: "127.0.0.1".into(),
        port: 0,
        archive_dir: root.join("archive").display().to_string(),
        tasks: false,
        ..Config::default()
    }
}

fn url(r: &server::Running) -> String {
    format!("https://127.0.0.1:{}", r.port())
}

/// The local operator, as the viewer on the server's machine is.
fn operator(root: &Path, r: &server::Running) -> Remote {
    let token = local::local_token(&Paths::at(root)).expect("the server wrote its token");
    Remote::new(&url(r), Trust::Pinned(r.fingerprint.clone()), Some(token)).unwrap()
}

/// Pair a station with `role` and return its client.
fn paired(op: &Remote, r: &server::Running, role: Role, name: &str) -> Remote {
    let code = op.new_pairing_code(role, 5).expect("a code");
    let anon = Remote::new(&url(r), Trust::Pinned(r.fingerprint.clone()), None).unwrap();
    // One attempt per second per address, and every test pairs from the
    // loopback.
    std::thread::sleep(Duration::from_millis(1100));
    let got = anon.pair(&code.code, name).expect("the code is accepted");
    assert_eq!(got.role, role);
    Remote::new(
        &url(r),
        Trust::Pinned(r.fingerprint.clone()),
        Some(got.token),
    )
    .unwrap()
}

fn failure(e: anyhow::Error) -> Failure {
    failure_of(&e)
        .cloned()
        .unwrap_or_else(|| panic!("not a protocol failure: {e:#}"))
}

#[test]
fn a_paired_station_lists_fetches_and_sends_no_more_than_its_role_allows() {
    let (root, src, n) = served_phantom("pacs_server_roles");
    let running = server::spawn(config_for(&root), Paths::at(&root.join("server"))).unwrap();
    let state = root.join("server");

    // ---- the first look: who it is, which certificate ------------------
    let (info, fp) = Remote::probe(&url(&running)).expect("the server answers");
    assert_eq!(
        fp, running.fingerprint,
        "the certificate the probe saw is the server's"
    );
    assert_eq!(info.fingerprint, running.fingerprint);
    assert_eq!(info.name, "Test PACS");
    assert!(!info.tasks);

    // ---- nothing without a token -----------------------------------------
    let anon = Remote::new(&url(&running), Trust::Pinned(fp.clone()), None).unwrap();
    assert!(matches!(
        failure(anon.patients().unwrap_err()),
        Failure::Unauthorized(_)
    ));

    let op = operator(&state, &running);
    let viewer = paired(&op, &running, Role::View, "laptop");

    // ---- list, manifest, one instance, a bundle ---------------------------
    let pts = viewer.patients().expect("the listing");
    assert_eq!(pts.len(), 1);
    assert_eq!(pts[0].title(), "PHANTOM RT (RTTEST001)");
    let study = &pts[0].studies[0];
    assert_eq!(study.files, n);
    let m = viewer.manifest(&study.study_uid).unwrap();
    assert_eq!(m.instances.len(), n);
    assert_eq!(m.patient_key, pts[0].key);

    let dir = Archive::new(root.join("archive"))
        .find_study(&study.study_uid)
        .unwrap();
    let first = &archive::instances(&dir).unwrap()[0];
    let got = root.join("one.dcm");
    viewer
        .instance(&study.study_uid, &first.sop_uid, &got)
        .unwrap();
    assert_eq!(
        std::fs::read(&got).unwrap(),
        std::fs::read(&first.path).unwrap(),
        "an instance arrives byte for byte"
    );
    let all: Vec<String> = m.instances.iter().map(|i| i.sop_uid.clone()).collect();
    let bundle_dir = root.join("bundle");
    assert_eq!(
        viewer.bundle(&study.study_uid, &all, &bundle_dir).unwrap(),
        n,
        "the phantom fits one bundle"
    );

    // ---- a role is a ceiling ---------------------------------------------
    let files: Vec<PathBuf> = archive::instances(&dir)
        .unwrap()
        .into_iter()
        .map(|i| i.path)
        .collect();
    let e = viewer
        .upload_files(&files, &root.join("up"), &Progress::default())
        .unwrap_err();
    assert!(
        matches!(failure(e), Failure::Forbidden(_)),
        "view cannot send"
    );
    assert!(matches!(
        failure(viewer.new_pairing_code(Role::Admin, 5).unwrap_err()),
        Failure::Forbidden(_)
    ));
    assert!(matches!(
        failure(viewer.workflows().unwrap_err()),
        Failure::Forbidden(_)
    ));

    let editor = paired(&op, &running, Role::Edit, "laptop");
    assert_eq!(
        editor.whoami().unwrap().name,
        "laptop (2)",
        "names are kept apart"
    );
    let again = editor
        .upload_files(&files, &root.join("up"), &Progress::default())
        .unwrap();
    assert_eq!(
        (again.stored, again.duplicates),
        (0, n),
        "the archive recognises what it has"
    );
    let junk = root.join("junk.txt");
    std::fs::write(&junk, "not DICOM").unwrap();
    let s = editor
        .upload_files(
            std::slice::from_ref(&junk),
            &root.join("up"),
            &Progress::default(),
        )
        .unwrap();
    assert_eq!((s.stored, s.skipped), (0, 1));
    assert!(matches!(
        failure(editor.delete_study(&study.study_uid).unwrap_err()),
        Failure::Forbidden(_)
    ));
    assert!(src.is_dir());

    // ---- nothing is reachable through a path ------------------------------
    assert_eq!(editor.raw_status("GET", "/studies/1..2"), 400);
    assert_eq!(editor.raw_status("GET", "/studies/..%2F..%2Fstate"), 400);
    assert_eq!(editor.raw_status("GET", "/studies/1.2.3"), 404);
    assert_eq!(editor.raw_status("GET", "/no/such/route"), 404);
    assert_eq!(
        editor.raw_status("GET", "/tasks"),
        403,
        "the task routes need the run role"
    );
    assert_eq!(op.raw_status("DELETE", "/patients/.."), 404);
    assert_eq!(op.raw_status("DELETE", "/patients/%2E%2E%2Fserver"), 404);
    assert_eq!(
        op.raw_status("GET", "/tasks/1-1/files/..%2F..%2Fstate%2Fclients.json"),
        404,
        "a server without tasks answers no task file"
    );
    assert!(root
        .join("server")
        .join("state")
        .join("clients.json")
        .is_file());

    // ---- a changed certificate is refused before the token is sent -------
    let wrong = Remote::new(
        &url(&running),
        Trust::Pinned("ab".repeat(32)),
        Some("would-be-token".into()),
    )
    .unwrap();
    match failure(wrong.patients().unwrap_err()) {
        Failure::CertificateChanged { seen } => assert_eq!(seen, running.fingerprint),
        other => panic!("expected a changed certificate, got {other:?}"),
    }

    // ---- guessing is throttled ---------------------------------------------
    std::thread::sleep(Duration::from_millis(1100));
    assert!(matches!(
        failure(anon.pair("AAAA-AAAA", "x").unwrap_err()),
        Failure::Unauthorized(_)
    ));
    match failure(anon.pair("AAAA-AAAA", "x").unwrap_err()) {
        Failure::Refused { status, .. } => assert_eq!(status, 429),
        other => panic!("expected 429, got {other:?}"),
    }

    // ---- the operator sees and revokes -------------------------------------
    let clients = op.clients().unwrap();
    assert_eq!(clients.len(), 2);
    assert!(clients
        .iter()
        .any(|c| c.name == "laptop" && c.role == Role::View));
    op.revoke("laptop").unwrap();
    assert!(matches!(
        failure(viewer.patients().unwrap_err()),
        Failure::Unauthorized(_)
    ));
    assert!(editor.patients().is_ok(), "only the revoked one is out");
    let log = op.audit(200).unwrap().join("\n");
    assert!(log.contains(" pair ") && log.contains(" revoke "), "{log}");
    assert!(
        !log.contains("PHANTOM"),
        "the audit log names stations and studies, never patients"
    );

    // ---- only the local operator stops it -----------------------------------
    assert!(matches!(
        failure(editor.shutdown().unwrap_err()),
        Failure::Forbidden(_)
    ));
    op.shutdown().unwrap();
    running.wait().unwrap();
    assert!(
        local::Running::read(&Paths::at(&state)).is_none(),
        "a stopped server leaves no running.json"
    );
}

#[test]
fn certificate_identity_and_tokens_survive_a_restart() {
    let (root, _, _) = served_phantom("pacs_server_restart");
    let state = root.join("server");
    let first = server::spawn(config_for(&root), Paths::at(&state)).unwrap();
    let op = operator(&state, &first);
    let station = paired(&op, &first, Role::View, "ipad");
    let (fp, id) = (first.fingerprint.clone(), first.server_id.clone());
    let port = first.port();
    first.shutdown().unwrap();
    assert!(
        station.patients().is_err(),
        "nothing answers while it is down"
    );

    let cfg = Config {
        port,
        ..config_for(&root)
    };
    let second = server::spawn(cfg, Paths::at(&state)).unwrap();
    assert_eq!(second.fingerprint, fp, "the same certificate");
    assert_eq!(second.server_id, id, "the same identity");
    assert_eq!(second.port(), port);
    assert_eq!(
        station.patients().unwrap().len(),
        1,
        "the station's token still works"
    );
    let run = local::Running::read(&Paths::at(&state)).unwrap();
    assert_eq!(run.port, port);
    assert_eq!(run.fingerprint, fp);
    second.shutdown().unwrap();
}

#[test]
fn an_oversized_upload_is_refused_and_leaves_the_archive_alone() {
    let (root, _, n) = served_phantom("pacs_server_too_large");
    let state = root.join("server");
    let cfg = Config {
        max_upload_mb: 1,
        ..config_for(&root)
    };
    let running = server::spawn(cfg, Paths::at(&state)).unwrap();
    let op = operator(&state, &running);
    let editor = paired(&op, &running, Role::Edit, "big");
    let big = root.join("big.bin");
    std::fs::write(&big, vec![7u8; 3 * 1024 * 1024]).unwrap();
    let zip = root.join("big.zip");
    rust_dicom_station::pacs::client::zip_files(std::slice::from_ref(&big), &zip).unwrap();
    match failure(editor.upload_zip(&zip).unwrap_err()) {
        Failure::Refused { status, .. } => assert_eq!(status, 413),
        // The server may close the connection before the client has
        // finished sending, which is a refusal all the same.
        Failure::Unreachable(_) => {}
        other => panic!("expected 413, got {other:?}"),
    }
    let pts = editor.patients().unwrap();
    assert_eq!(pts.len(), 1);
    assert_eq!(pts[0].files(), n, "nothing was filed");
    assert!(
        std::fs::read_dir(state.join("data").join("incoming"))
            .map(|d| d.count())
            .unwrap_or(0)
            == 0,
        "nothing is left half received"
    );
    running.shutdown().unwrap();
}
