//! The two ways of working with a remote archive, end to end over HTTPS.
//!
//! *Mirror*: `tests/archive.rs` over the wire. A station pulls a study of the
//! server into its mirror, loads it with the ordinary loader, draws on it,
//! sends the derived objects back, and both copies end up with them under
//! the same patient and study. While the server is down a send waits in
//! the outbox; the next sync delivers it, and fetches what others filed
//! meanwhile.
//!
//! *Tasks*: a station hands the server a workflow bound to a study of its
//! archive; the server runs it, files the result into that study, and the
//! station pulls it.
//!
//! Needs the `pacs-server` feature:
//! `cargo test --features pacs-server --test pacs_mirror`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rust_dicom_station::archive::Archive;
use rust_dicom_station::dicom_export::{self, ExportParams};
use rust_dicom_station::dicomseg::SegSeries;
use rust_dicom_station::gen_test_data::{self, GenParams};
use rust_dicom_station::loader;
use rust_dicom_station::pacs::client::{failure_of, Failure, Remote, Trust};
use rust_dicom_station::pacs::config::Config;
use rust_dicom_station::pacs::local::{self, Paths};
use rust_dicom_station::pacs::mirror::{self, Mirror, Sent};
use rust_dicom_station::pacs::protocol::{Binding, InputKind, Role, TaskRequest, TaskState};
use rust_dicom_station::pacs::server;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::segmentation::Segmentation;

fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn served_phantom(tag: &str) -> (PathBuf, usize) {
    let root = scratch(tag);
    let src = root.join("phantom");
    let n = gen_test_data::generate(&src, &GenParams::default(), &Progress::default()).unwrap();
    Archive::new(root.join("archive"))
        .import(&src, &Progress::default())
        .unwrap();
    (root, n)
}

fn config_for(root: &Path, port: u16, tasks: bool) -> Config {
    Config {
        bind: "127.0.0.1".into(),
        port,
        archive_dir: root.join("archive").display().to_string(),
        tasks,
        workflows_dir: root.join("workflows").display().to_string(),
        models_dir: root.join("models").display().to_string(),
        ..Config::default()
    }
}

fn station(root: &Path, running: &server::Running, role: Role) -> Remote {
    let url = format!("https://127.0.0.1:{}", running.port());
    let token = local::local_token(&Paths::at(&root.join("server"))).unwrap();
    let op = Remote::new(
        &url,
        Trust::Pinned(running.fingerprint.clone()),
        Some(token),
    )
    .unwrap();
    let code = op.new_pairing_code(role, 5).unwrap();
    let anon = Remote::new(&url, Trust::Pinned(running.fingerprint.clone()), None).unwrap();
    let got = anon.pair(&code.code, "station").unwrap();
    Remote::new(
        &url,
        Trust::Pinned(running.fingerprint.clone()),
        Some(got.token),
    )
    .unwrap()
}

/// A solid ball around the volume centre.
fn ball(dims: [usize; 3], radius: f64) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let c = [
        (nx as f64 - 1.0) * 0.5,
        (ny as f64 - 1.0) * 0.5,
        (nz as f64 - 1.0) * 0.5,
    ];
    let mut m = vec![0u8; nx * ny * nz];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let d = ((i as f64 - c[0]).powi(2)
                    + (j as f64 - c[1]).powi(2)
                    + (k as f64 - c[2]).powi(2))
                .sqrt();
                if d <= radius {
                    m[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
    }
    m
}

/// Load a mirrored study, draw a segmentation named `name` on it, and
/// write the derived objects into a fresh folder. Returns the folder and
/// how many files went into it.
fn draw_and_export(study_dir: &Path, name: &str, out: &Path) -> usize {
    let mut study = loader::load_directory(study_dir, &Progress::default()).unwrap();
    let dims = study.volume.dims;
    let s = &study.series[study.active_series];
    let mut ser = SegSeries::new(
        format!("{name} QA"),
        study.volume.grid(),
        s.uid.clone(),
        s.study_uid.clone(),
    );
    ser.segs.push(Segmentation::from_mask(
        name.into(),
        [40, 200, 40],
        dims,
        ball(dims, 6.0),
    ));
    study.seg_series.push(ser);
    // Only the new segmentation series: everything loaded is in the
    // archive already.
    study.structure_sets.clear();
    study
        .seg_series
        .retain(|s| s.segs.iter().any(|g| g.name == name));
    let params = ExportParams::for_study(&study);
    dicom_export::export_derived(&study, out, &params, &Progress::default()).unwrap()
}

fn files_of(archive: &Archive, study_uid: &str) -> usize {
    archive
        .scan()
        .unwrap()
        .iter()
        .flat_map(|p| p.studies.iter())
        .find(|s| s.study_uid == study_uid)
        .map(|s| s.files)
        .unwrap_or(0)
}

#[test]
fn a_mirrored_study_is_worked_on_offline_and_synced_back() {
    let (root, n) = served_phantom("pacs_mirror_roundtrip");
    let state = root.join("server");
    let running = server::spawn(config_for(&root, 0, false), Paths::at(&state)).unwrap();
    let port = running.port();
    let remote = station(&root, &running, Role::Edit);
    let m = Mirror::at(root.join("mirror"));
    let server_archive = Archive::new(root.join("archive"));

    // ---- pull ---------------------------------------------------------------
    let pts = remote.patients().unwrap();
    m.save_listing(&pts);
    assert_eq!(
        m.cached_listing(),
        pts,
        "the listing is kept for offline use"
    );
    let uid = pts[0].studies[0].study_uid.clone();
    let got = mirror::pull_patient(&remote, &m, &pts[0], &Progress::default()).unwrap();
    assert_eq!((got.fetched, got.present, got.studies), (n, 0, 1));
    let again = mirror::pull_study(&remote, &m, &uid, &Progress::default()).unwrap();
    assert_eq!(
        (again.fetched, again.present),
        (0, n),
        "a second pull fetches nothing"
    );
    let local_pts = m.archive().scan().unwrap();
    assert_eq!(local_pts.len(), 1);
    assert_eq!(
        local_pts[0].title(),
        pts[0].title(),
        "the same patient, filed alike"
    );
    assert_eq!(local_pts[0].studies[0].files, n);

    // ---- work on it and send it back ---------------------------------------
    let study_dir = m.archive().find_study(&uid).unwrap();
    let out = root.join("derived1");
    let k = draw_and_export(&study_dir, "Ball", &out);
    assert_eq!(k, 1, "one segmentation series");
    match mirror::send_folder(&remote, &m, &out, &Progress::default()).unwrap() {
        Sent::Delivered(s) => assert_eq!(s.stored, 1),
        other => panic!("expected delivery, got {other:?}"),
    }
    assert_eq!(
        files_of(&server_archive, &uid),
        n + 1,
        "filed into the same study"
    );
    assert_eq!(server_archive.scan().unwrap().len(), 1, "no second patient");
    let st = &server_archive.scan().unwrap()[0].studies[0];
    assert!(
        st.modalities.iter().any(|x| x == "SEG"),
        "{:?}",
        st.modalities
    );
    assert_eq!(files_of(&m.archive(), &uid), n + 1, "and into the mirror");

    // ---- the server goes away: the send waits ------------------------------
    running.shutdown().unwrap();
    let out2 = root.join("derived2");
    draw_and_export(&study_dir, "Offline", &out2);
    match mirror::send_folder(&remote, &m, &out2, &Progress::default()).unwrap() {
        Sent::Queued(k) => assert_eq!(k, 1),
        other => panic!("expected the outbox, got {other:?}"),
    }
    assert_eq!(m.pending_files(), 1);
    assert!(!out2.exists(), "the folder moved into the outbox");
    let e = mirror::sync(&remote, &m, &Progress::default()).unwrap_err();
    assert!(matches!(failure_of(&e), Some(Failure::Unreachable(_))));
    assert_eq!(m.pending_files(), 1, "a failed sync keeps the outbox");

    // ---- it comes back; someone else filed something meanwhile --------------
    let running = server::spawn(config_for(&root, port, false), Paths::at(&state)).unwrap();
    let other = root.join("derived3");
    draw_and_export(&study_dir, "Elsewhere", &other);
    server_archive.import(&other, &Progress::default()).unwrap();
    let sum = mirror::sync(&remote, &m, &Progress::default()).unwrap();
    assert_eq!(sum.outbox_sent, 1);
    assert_eq!(sum.pulled.fetched, 1, "what was filed meanwhile came down");
    assert!(sum.gone.is_empty());
    assert_eq!(m.pending_files(), 0);
    assert_eq!(files_of(&server_archive, &uid), n + 3);
    assert_eq!(files_of(&m.archive(), &uid), n + 3, "both copies agree");
    let quiet = mirror::sync(&remote, &m, &Progress::default()).unwrap();
    assert_eq!(
        (quiet.pulled.fetched, quiet.pushed.stored, quiet.outbox_sent),
        (0, 0, 0),
        "a sync of two equal copies moves nothing"
    );

    // ---- the server drops the study: it is reported, kept, not sent back -----
    let token = local::local_token(&Paths::at(&state)).unwrap();
    let op = Remote::new(
        &format!("https://127.0.0.1:{port}"),
        Trust::Pinned(running.fingerprint.clone()),
        Some(token),
    )
    .unwrap();
    op.delete_study(&uid).unwrap();
    let sum = mirror::sync(&remote, &m, &Progress::default()).unwrap();
    assert_eq!(sum.gone, vec![uid.clone()]);
    assert_eq!(files_of(&server_archive, &uid), 0, "not sent back");
    assert_eq!(files_of(&m.archive(), &uid), n + 3, "kept here");
    m.remove_study(&uid).unwrap();
    assert!(
        m.archive().scan().unwrap().is_empty(),
        "the patient goes with their last study"
    );
    running.shutdown().unwrap();
}

#[test]
fn a_task_runs_on_the_server_and_its_result_comes_back() {
    let (root, n) = served_phantom("pacs_mirror_task");
    let state = root.join("server");
    let running = server::spawn(config_for(&root, 0, true), Paths::at(&state)).unwrap();
    let remote = station(&root, &running, Role::Run);
    let pts = remote.patients().unwrap();
    let uid = pts[0].studies[0].study_uid.clone();

    let offered = remote.workflows().unwrap();
    let body = offered
        .iter()
        .find(|w| w.name == "Body contour")
        .expect("the template is offered");
    assert_eq!(body.source, "template");
    assert!(body.problems.is_empty(), "{:?}", body.problems);
    assert_eq!(body.inputs.len(), 1);
    assert_eq!(body.inputs[0].kind, InputKind::Archive);
    assert!(
        offered.iter().any(|w| w.source == "example"),
        "the shipped examples are offered too"
    );

    // An unbound input is refused before anything is queued.
    let e = remote
        .submit(&TaskRequest {
            workflow: "Body contour".into(),
            ..Default::default()
        })
        .unwrap_err();
    match failure_of(&e) {
        Some(Failure::Refused {
            status, message, ..
        }) => {
            assert_eq!(*status, 400);
            assert!(message.contains("bind"), "{message}");
        }
        other => panic!("expected a refusal, got {other:?}"),
    }

    let binding = Binding {
        node: body.inputs[0].node,
        patient_key: pts[0].key.clone(),
        study_uid: uid.clone(),
    };
    let sub = remote
        .submit(&TaskRequest {
            workflow: "Body contour".into(),
            bindings: vec![binding.clone()],
            title: "outline".into(),
            ..Default::default()
        })
        .unwrap();
    // A second one, cancelled while it waits or runs.
    let second = remote
        .submit(&TaskRequest {
            workflow: "Body contour".into(),
            bindings: vec![binding],
            ..Default::default()
        })
        .unwrap();
    remote.cancel_task(&second.task_id).unwrap();

    let started = Instant::now();
    let done = loop {
        let t = remote.task(&sub.task_id, 0).unwrap();
        if t.state.finished() {
            break t;
        }
        assert!(
            started.elapsed() < Duration::from_secs(600),
            "the task did not finish: {:?}",
            t
        );
        std::thread::sleep(Duration::from_millis(250));
    };
    assert_eq!(
        done.state,
        TaskState::Done,
        "{:?} {:?}",
        done.error,
        done.log
    );
    assert_eq!(done.title, "outline");
    assert_eq!(done.client, "station");
    assert!(
        done.filed.contains(&uid),
        "the result was filed into the study: {} {:?}",
        done.message,
        done.log
    );
    assert!(
        done.log.iter().any(|l| l.starts_with("done: Body contour")),
        "{:?}",
        done.log
    );
    let log_tail = remote.task(&sub.task_id, done.log_len - 1).unwrap();
    assert_eq!(log_tail.log.len(), 1, "the log is read from a cursor");

    let cancelled = loop {
        let t = remote.task(&second.task_id, 0).unwrap();
        if t.state.finished() {
            break t;
        }
        std::thread::sleep(Duration::from_millis(250));
    };
    assert_eq!(cancelled.state, TaskState::Cancelled);

    // Pull the result: the study has one object more, and it is the body.
    let m = Mirror::at(root.join("mirror"));
    let got = mirror::pull_study(&remote, &m, &uid, &Progress::default()).unwrap();
    assert_eq!(got.fetched, n + 1);
    let study =
        loader::load_directory(&m.archive().find_study(&uid).unwrap(), &Progress::default())
            .unwrap();
    assert!(
        study
            .structure_sets
            .iter()
            .flat_map(|ss| ss.rois.iter())
            .any(|r| r.name == "BODY"),
        "the outline came back as a structure"
    );

    // The task list, newest first, without logs.
    let list = remote.tasks().unwrap();
    assert_eq!(list[0].id, second.task_id);
    assert!(list.iter().all(|t| t.log.is_empty()));
    running.shutdown().unwrap();
}
