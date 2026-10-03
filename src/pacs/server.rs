//! The HTTPS door onto the archive: axum over tokio, TLS through rustls.
//!
//! Deliberately thin, like the MCP server's protocol layer. Every request
//! that touches data goes to [`Archive`] or to the task queue
//! ([`super::tasks`]) on a blocking thread; this file owns the socket, TLS,
//! the token check, bodies streamed to disk, and the audit line.
//!
//! ## Rules every route keeps
//!
//! * **Nothing is built from the URL.** A UID in a path is checked to be
//!   digits and dots ([`crate::archive::is_uid`]) and then *compared* with
//!   the archive's folder names ([`Archive::find_study`]); a patient key is
//!   compared the same way. No route joins request text onto a path.
//! * **Bodies go to disk, not memory.** An upload is streamed into
//!   `<data>/pacs/incoming/` and cut off past `max_upload_mb`; a zip is
//!   unpacked under numbered names, never its entries' own, and only up to
//!   four times that size.
//! * **The archive's writers take turns.** Uploads, removals and the
//!   filing of a task's results take one lock for writing; listings and
//!   downloads read alongside each other. (The viewer on the same machine
//!   writes without it; the archive's own rules - files named by UID,
//!   sidecars replaced whole - keep that safe.)
//! * **Every call is logged** to the audit log: who, what, which study,
//!   how it ended. Never a patient's name.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use axum::body::Body;
use axum::extract::{ConnectInfo, Path as UrlPath, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_server::tls_rustls::RustlsConfig;
use axum_server::Handle;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use super::auth::{Auth, Caller, PairError};
use super::config::Config;
use super::local::{Paths, Running as RunningFile};
use super::protocol::*;
use super::tasks::Tasks;
use crate::archive::{self, Archive};
use crate::audit::Audit;
use crate::progress::Progress;

/// How much one bundle carries at most (see [`super::client::BUNDLE_MB`]).
const BUNDLE_BYTES: u64 = super::client::BUNDLE_MB * 1024 * 1024;

/// Everything the routes share.
struct AppState {
    config: Config,
    paths: Paths,
    archive_root: PathBuf,
    info: ServerInfo,
    auth: Mutex<Auth>,
    audit: Audit,
    archive_lock: Arc<RwLock<()>>,
    tasks: Option<Arc<Tasks>>,
    handle: Handle<SocketAddr>,
    /// Uploads under way, for scratch folder names.
    uploads: std::sync::atomic::AtomicU64,
}

type S = Arc<AppState>;

/// The server's identity, made once: the id names the clients' mirror
/// folders, so it outlives renames and new certificates.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct Identity {
    server_id: String,
    created: String,
}

fn identity(paths: &Paths) -> Result<Identity> {
    if let Ok(text) = std::fs::read_to_string(paths.identity()) {
        if let Ok(id) = serde_json::from_str::<Identity>(&text) {
            if !id.server_id.is_empty() {
                return Ok(id);
            }
        }
    }
    let id = Identity {
        server_id: super::random_token(12)?,
        created: super::stamp(),
    };
    std::fs::create_dir_all(&paths.state)
        .with_context(|| format!("create {}", paths.state.display()))?;
    std::fs::write(
        paths.identity(),
        serde_json::to_string_pretty(&id).expect("plain data serialises"),
    )
    .with_context(|| format!("write {}", paths.identity().display()))?;
    Ok(id)
}

// ---- errors -------------------------------------------------------------

/// An error answer: status, message, code.
struct ApiErr(StatusCode, String, &'static str);

impl IntoResponse for ApiErr {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(ApiError {
                error: self.1,
                code: self.2.to_string(),
            }),
        )
            .into_response()
    }
}

impl From<anyhow::Error> for ApiErr {
    fn from(e: anyhow::Error) -> Self {
        ApiErr(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{e:#}"),
            "failed",
        )
    }
}

fn bad(m: impl Into<String>) -> ApiErr {
    ApiErr(StatusCode::BAD_REQUEST, m.into(), "bad_request")
}

fn not_found(m: impl Into<String>) -> ApiErr {
    ApiErr(StatusCode::NOT_FOUND, m.into(), "not_found")
}

type ApiResult<T> = std::result::Result<T, ApiErr>;

/// Run blocking work (the archive, the file system) off the async threads.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> ApiResult<T> + Send + 'static,
) -> ApiResult<T> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiErr(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), "failed"))?
}

// ---- who is calling -------------------------------------------------------

/// The address a request came from: the peer, or - behind a reverse proxy
/// on this machine - what the proxy says the client was.
fn client_ip(s: &AppState, peer: SocketAddr, headers: &HeaderMap) -> IpAddr {
    if s.config.behind_proxy && peer.ip().is_loopback() {
        if let Some(ip) = headers
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.split(',').next())
            .and_then(|v| v.trim().parse::<IpAddr>().ok())
        {
            return ip;
        }
    }
    peer.ip()
}

/// The caller, when their token checks out and their role allows `need`.
fn caller(s: &AppState, peer: SocketAddr, headers: &HeaderMap, need: Role) -> ApiResult<Caller> {
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .ok_or_else(|| {
            ApiErr(
                StatusCode::UNAUTHORIZED,
                "this server lets in paired stations only".into(),
                "unauthorized",
            )
        })?;
    let ip = client_ip(s, peer, headers).to_string();
    let who = s
        .auth
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .check(token, &ip)
        .ok_or_else(|| {
            s.audit.line("auth", 0, "refused", &ip);
            ApiErr(
                StatusCode::UNAUTHORIZED,
                "this station's token is not accepted (revoked, or the server's clients were \
                 reset); pair again"
                    .into(),
                "unauthorized",
            )
        })?;
    if !who.role.allows(need) {
        return Err(ApiErr(
            StatusCode::FORBIDDEN,
            format!(
                "'{}' is paired with the {} role; this needs {}",
                who.name,
                who.role.label(),
                need.label()
            ),
            "forbidden",
        ));
    }
    Ok(who)
}

fn check_uid(uid: &str) -> ApiResult<()> {
    if archive::is_uid(uid) {
        Ok(())
    } else {
        Err(bad("not a DICOM UID"))
    }
}

// ---- routes -------------------------------------------------------------

fn router(state: S) -> Router {
    Router::new()
        .route("/rds/v1/server", get(server_info))
        .route("/rds/v1/pair", post(pair))
        .route("/rds/v1/whoami", get(whoami))
        .route("/rds/v1/patients", get(patients))
        .route(
            "/rds/v1/patients/{key}",
            axum::routing::delete(delete_patient),
        )
        .route("/rds/v1/studies", post(upload))
        .route("/rds/v1/studies/{uid}", get(manifest).delete(delete_study))
        .route("/rds/v1/studies/{uid}/instances/{sop}", get(instance))
        .route("/rds/v1/studies/{uid}/bundle", post(bundle))
        .route("/rds/v1/workflows", get(workflows))
        .route("/rds/v1/tasks", get(list_tasks).post(submit_task))
        .route("/rds/v1/tasks/{id}", get(get_task))
        .route("/rds/v1/tasks/{id}/cancel", post(cancel_task))
        .route("/rds/v1/tasks/{id}/files/{*name}", get(task_file))
        .route("/rds/v1/admin/pairing-codes", post(new_code))
        .route("/rds/v1/admin/clients", get(clients))
        .route("/rds/v1/admin/clients/{name}/revoke", post(revoke))
        .route("/rds/v1/admin/audit", get(audit_tail))
        .route("/rds/v1/admin/shutdown", post(shutdown))
        .fallback(|| async { not_found("no such route on this PACS server") })
        .with_state(state)
}

async fn server_info(State(s): State<S>) -> Json<ServerInfo> {
    Json(s.info.clone())
}

async fn pair(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(req): Json<PairRequest>,
) -> ApiResult<Json<PairResponse>> {
    let ip = client_ip(&s, peer, &headers);
    let r = s
        .auth
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pair(&req.code, &req.client_name, ip);
    match r {
        Ok((token, role, name)) => {
            s.audit.line(
                "pair",
                0,
                "ok",
                &format!("{name} as {} from {ip}", role.label()),
            );
            Ok(Json(PairResponse {
                token,
                role,
                server_id: s.info.server_id.clone(),
                client_name: name,
            }))
        }
        Err(PairError::TooSoon) => Err(ApiErr(
            StatusCode::TOO_MANY_REQUESTS,
            "one pairing attempt per second; wait a moment".into(),
            "too_many",
        )),
        Err(PairError::BadCode) => {
            s.audit.line("pair", 0, "refused", &ip.to_string());
            Err(ApiErr(
                StatusCode::UNAUTHORIZED,
                "this pairing code is not valid (mistyped, used, or expired); ask the server's \
                 operator for a new one"
                    .into(),
                "unauthorized",
            ))
        }
    }
}

async fn whoami(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> ApiResult<Json<WhoAmI>> {
    let who = caller(&s, peer, &headers, Role::View)?;
    Ok(Json(WhoAmI {
        name: who.name,
        role: who.role,
    }))
}

async fn patients(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> ApiResult<Json<Listing>> {
    let who = caller(&s, peer, &headers, Role::View)?;
    let st = s.clone();
    let listing = blocking(move || {
        let _r = st.archive_lock.read().unwrap_or_else(|e| e.into_inner());
        let found = Archive::new(&st.archive_root).scan()?;
        Ok(Listing {
            patients: found
                .into_iter()
                .map(|p| RemotePatient {
                    key: p.key(),
                    name: p.name.clone(),
                    id: p.id.clone(),
                    studies: p
                        .studies
                        .iter()
                        .map(|st| RemoteStudy {
                            study_uid: st.study_uid.clone(),
                            date: st.date.clone(),
                            description: st.description.clone(),
                            modalities: st.modalities.clone(),
                            files: st.files,
                        })
                        .collect(),
                })
                .collect(),
        })
    })
    .await?;
    s.audit.line(
        "list",
        0,
        "ok",
        &format!("{} - {} patient(s)", who.name, listing.patients.len()),
    );
    Ok(Json(listing))
}

/// The study's folder, or 404.
fn study_dir(s: &AppState, uid: &str) -> ApiResult<PathBuf> {
    Archive::new(&s.archive_root)
        .find_study(uid)
        .ok_or_else(|| not_found("the archive has no such study"))
}

async fn manifest(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath(uid): UrlPath<String>,
) -> ApiResult<Json<Manifest>> {
    caller(&s, peer, &headers, Role::View)?;
    check_uid(&uid)?;
    let st = s.clone();
    let m = blocking(move || {
        let _r = st.archive_lock.read().unwrap_or_else(|e| e.into_inner());
        let dir = study_dir(&st, &uid)?;
        let key = dir
            .parent()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(Manifest {
            study_uid: uid,
            patient_key: key,
            instances: archive::instances(&dir)?
                .into_iter()
                .map(|i| InstanceInfo {
                    sop_uid: i.sop_uid,
                    bytes: i.bytes,
                })
                .collect(),
        })
    })
    .await?;
    Ok(Json(m))
}

async fn instance(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath((uid, sop)): UrlPath<(String, String)>,
) -> ApiResult<Response> {
    let who = caller(&s, peer, &headers, Role::View)?;
    check_uid(&uid)?;
    check_uid(&sop)?;
    let st = s.clone();
    let (bytes, uid2) = blocking(move || {
        let _r = st.archive_lock.read().unwrap_or_else(|e| e.into_inner());
        let dir = study_dir(&st, &uid)?;
        let inst = archive::instances(&dir)?
            .into_iter()
            .find(|i| i.sop_uid == sop)
            .ok_or_else(|| not_found("the study has no such instance"))?;
        let bytes = std::fs::read(&inst.path).map_err(|e| anyhow!("read an instance: {e}"))?;
        Ok((bytes, uid))
    })
    .await?;
    s.audit.line(
        "instance",
        0,
        "ok",
        &format!("{} study {uid2} {} bytes", who.name, bytes.len()),
    );
    Ok(([(header::CONTENT_TYPE, "application/dicom")], bytes).into_response())
}

async fn bundle(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath(uid): UrlPath<String>,
    Json(req): Json<BundleRequest>,
) -> ApiResult<Response> {
    let who = caller(&s, peer, &headers, Role::View)?;
    check_uid(&uid)?;
    if req.sops.iter().any(|s| !archive::is_uid(s)) {
        return Err(bad("an instance named is not a DICOM UID"));
    }
    let st = s.clone();
    let uid2 = uid.clone();
    let (zip, n) = blocking(move || {
        let _r = st.archive_lock.read().unwrap_or_else(|e| e.into_inner());
        let dir = study_dir(&st, &uid2)?;
        let wanted: std::collections::BTreeSet<&str> =
            req.sops.iter().map(String::as_str).collect();
        let mut picked = Vec::new();
        let mut size = 0u64;
        for i in archive::instances(&dir)? {
            if !wanted.contains(i.sop_uid.as_str()) {
                continue;
            }
            if !picked.is_empty() && size + i.bytes > BUNDLE_BYTES {
                break;
            }
            size += i.bytes;
            picked.push(i.path);
        }
        let n = picked.len();
        Ok((zip_in_memory(&picked)?, n))
    })
    .await?;
    s.audit.line(
        "bundle",
        0,
        "ok",
        &format!("{} study {uid} {n} file(s) {} bytes", who.name, zip.len()),
    );
    Ok(([(header::CONTENT_TYPE, "application/zip")], zip).into_response())
}

fn zip_in_memory(files: &[PathBuf]) -> Result<Vec<u8>> {
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        for (i, f) in files.iter().enumerate() {
            zip.start_file(format!("{i:06}.dcm"), opts)?;
            let mut src =
                std::fs::File::open(f).with_context(|| format!("read {}", f.display()))?;
            std::io::copy(&mut src, &mut zip)?;
        }
        zip.finish()?;
    }
    Ok(buf.into_inner())
}

async fn upload(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Body,
) -> ApiResult<Json<UploadSummary>> {
    let who = caller(&s, peer, &headers, Role::Edit)?;
    let started = Instant::now();
    let n = s.uploads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = s
        .paths
        .incoming()
        .join(format!("{}-{n}", super::unix_now()));
    let max = s.config.max_upload_mb * 1024 * 1024;
    let result = receive_and_file(&s, &dir, body, max).await;
    let _ = tokio::fs::remove_dir_all(&dir).await;
    let ms = started.elapsed().as_millis();
    match result {
        Ok(sum) => {
            s.audit.line(
                "upload",
                ms,
                "ok",
                &format!("{} - {}", who.name, sum.describe()),
            );
            Ok(Json(sum))
        }
        Err(e) => {
            s.audit
                .line("upload", ms, "error", &format!("{} - {}", who.name, e.1));
            Err(e)
        }
    }
}

/// Stream the body to disk (within `max` bytes), unpack it when it is a
/// zip, and file what it holds.
async fn receive_and_file(
    s: &S,
    dir: &std::path::Path,
    body: Body,
    max: u64,
) -> ApiResult<UploadSummary> {
    tokio::fs::create_dir_all(dir.join("files"))
        .await
        .map_err(|e| anyhow!("create the upload folder: {e}"))?;
    let raw = dir.join("upload.bin");
    let mut out = tokio::fs::File::create(&raw)
        .await
        .map_err(|e| anyhow!("write the upload: {e}"))?;
    let mut stream = body.into_data_stream();
    let mut size = 0u64;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| bad(format!("the upload broke off: {e}")))?;
        size += chunk.len() as u64;
        if size > max {
            return Err(ApiErr(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "the upload is larger than this server takes ({} MB); send it in parts",
                    max / (1024 * 1024)
                ),
                "too_large",
            ));
        }
        out.write_all(&chunk)
            .await
            .map_err(|e| anyhow!("write the upload: {e}"))?;
    }
    out.flush()
        .await
        .map_err(|e| anyhow!("write the upload: {e}"))?;
    drop(out);
    if size == 0 {
        return Err(bad("the upload is empty"));
    }
    let st = s.clone();
    let dir = dir.to_path_buf();
    blocking(move || {
        let files = dir.join("files");
        let mut magic = [0u8; 4];
        let is_zip = std::fs::File::open(&raw)
            .and_then(|mut f| std::io::Read::read_exact(&mut f, &mut magic))
            .is_ok()
            && magic == *b"PK\x03\x04";
        if is_zip {
            super::client::unzip_flat(&raw, &files, max.saturating_mul(4))
                .map_err(|e| bad(format!("{e:#}")))?;
        } else {
            std::fs::rename(&raw, files.join("upload.dcm"))
                .map_err(|e| anyhow!("keep the upload: {e}"))?;
        }
        let _w = st.archive_lock.write().unwrap_or_else(|e| e.into_inner());
        let sum = Archive::new(&st.archive_root).import(&files, &Progress::default())?;
        Ok(UploadSummary::from(sum))
    })
    .await
}

async fn delete_study(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath(uid): UrlPath<String>,
) -> ApiResult<StatusCode> {
    let who = caller(&s, peer, &headers, Role::Admin)?;
    check_uid(&uid)?;
    let st = s.clone();
    let u = uid.clone();
    blocking(move || {
        let _w = st.archive_lock.write().unwrap_or_else(|e| e.into_inner());
        let dir = study_dir(&st, &u)?;
        Archive::new(&st.archive_root).remove(&dir)?;
        Ok(())
    })
    .await?;
    s.audit.line(
        "remove_study",
        0,
        "ok",
        &format!("{} study {uid}", who.name),
    );
    Ok(StatusCode::NO_CONTENT)
}

async fn delete_patient(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath(key): UrlPath<String>,
) -> ApiResult<StatusCode> {
    let who = caller(&s, peer, &headers, Role::Admin)?;
    let st = s.clone();
    let k = key.clone();
    blocking(move || {
        let _w = st.archive_lock.write().unwrap_or_else(|e| e.into_inner());
        let a = Archive::new(&st.archive_root);
        let dir = a
            .find_patient(&k)
            .ok_or_else(|| not_found("the archive has no such patient"))?;
        a.remove(&dir)?;
        Ok(())
    })
    .await?;
    s.audit.line(
        "remove_patient",
        0,
        "ok",
        &format!("{} patient folder {key}", who.name),
    );
    Ok(StatusCode::NO_CONTENT)
}

// ---- tasks ----------------------------------------------------------------

fn tasks_of(s: &AppState) -> ApiResult<Arc<Tasks>> {
    s.tasks.clone().ok_or_else(|| {
        ApiErr(
            StatusCode::NOT_FOUND,
            "this server does not run tasks".into(),
            "not_found",
        )
    })
}

async fn workflows(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> ApiResult<Json<WorkflowList>> {
    caller(&s, peer, &headers, Role::Run)?;
    let tasks = tasks_of(&s)?;
    let list = blocking(move || {
        Ok(WorkflowList {
            workflows: tasks.offered().into_iter().map(|o| o.info).collect(),
        })
    })
    .await?;
    Ok(Json(list))
}

async fn submit_task(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(req): Json<TaskRequest>,
) -> ApiResult<Json<Submitted>> {
    let who = caller(&s, peer, &headers, Role::Run)?;
    let tasks = tasks_of(&s)?;
    let name = who.name.clone();
    let what = if req.workflow.is_empty() {
        "an inline workflow".to_string()
    } else {
        req.workflow.clone()
    };
    let r = blocking(move || tasks.submit(req, &name).map_err(|e| bad(format!("{e:#}")))).await;
    match &r {
        Ok(sub) => s.audit.line(
            "task",
            0,
            "queued",
            &format!("{} {} - {what}", who.name, sub.task_id),
        ),
        Err(e) => s
            .audit
            .line("task", 0, "refused", &format!("{} - {}", who.name, e.1)),
    }
    Ok(Json(r?))
}

async fn list_tasks(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> ApiResult<Json<TaskList>> {
    caller(&s, peer, &headers, Role::Run)?;
    let tasks = tasks_of(&s)?;
    Ok(Json(TaskList {
        tasks: tasks.list(),
    }))
}

async fn get_task(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<HashMap<String, String>>,
) -> ApiResult<Json<TaskInfo>> {
    caller(&s, peer, &headers, Role::Run)?;
    let tasks = tasks_of(&s)?;
    let from = q
        .get("log_from")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0usize);
    tasks
        .get(&id, from)
        .map(Json)
        .ok_or_else(|| not_found("no such task"))
}

async fn cancel_task(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath(id): UrlPath<String>,
) -> ApiResult<StatusCode> {
    let who = caller(&s, peer, &headers, Role::Run)?;
    let tasks = tasks_of(&s)?;
    tasks
        .cancel(&id, &who.name, who.role == Role::Admin)
        .map_err(|e| ApiErr(StatusCode::FORBIDDEN, format!("{e:#}"), "forbidden"))?;
    s.audit
        .line("cancel", 0, "ok", &format!("{} task {id}", who.name));
    Ok(StatusCode::NO_CONTENT)
}

async fn task_file(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath((id, name)): UrlPath<(String, String)>,
) -> ApiResult<Response> {
    caller(&s, peer, &headers, Role::Run)?;
    let tasks = tasks_of(&s)?;
    // Only a name the task's own list gives is answered.
    let path = tasks
        .file(&id, &name)
        .ok_or_else(|| not_found("the task wrote no such file"))?;
    let bytes = tokio::fs::read(&path)
        .await
        .map_err(|e| anyhow!("read the task's file: {e}"))?;
    let ty = match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "csv" => "text/csv; charset=utf-8",
        "md" | "txt" => "text/plain; charset=utf-8",
        "json" => "application/json",
        "png" => "image/png",
        "gif" => "image/gif",
        "jpg" | "jpeg" => "image/jpeg",
        "html" => "text/html; charset=utf-8",
        _ => "application/octet-stream",
    };
    Ok(([(header::CONTENT_TYPE, ty)], bytes).into_response())
}

// ---- the operator ----------------------------------------------------------

async fn new_code(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Json(req): Json<CodeRequest>,
) -> ApiResult<Json<PairingCode>> {
    let who = caller(&s, peer, &headers, Role::Admin)?;
    let minutes = if req.minutes == 0 {
        s.config.pairing_minutes
    } else {
        req.minutes
    };
    let code = s
        .auth
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .new_code(req.role, minutes)?;
    s.audit.line(
        "pairing_code",
        0,
        "ok",
        &format!("{} - {} for {minutes} min", who.name, req.role.label()),
    );
    Ok(Json(code))
}

async fn clients(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> ApiResult<Json<ClientList>> {
    caller(&s, peer, &headers, Role::Admin)?;
    let list = s.auth.lock().unwrap_or_else(|e| e.into_inner()).clients();
    Ok(Json(ClientList { clients: list }))
}

async fn revoke(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    UrlPath(name): UrlPath<String>,
) -> ApiResult<StatusCode> {
    let who = caller(&s, peer, &headers, Role::Admin)?;
    let gone = s
        .auth
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .revoke(&name)?;
    if !gone {
        return Err(not_found("no client of that name"));
    }
    s.audit
        .line("revoke", 0, "ok", &format!("{} revoked {name}", who.name));
    Ok(StatusCode::NO_CONTENT)
}

async fn audit_tail(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    Query(q): Query<HashMap<String, String>>,
) -> ApiResult<Json<AuditTail>> {
    caller(&s, peer, &headers, Role::Admin)?;
    let n = q
        .get("lines")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100usize)
        .min(2000);
    let st = s.clone();
    let lines = blocking(move || Ok(st.audit.tail(n))).await?;
    Ok(Json(AuditTail { lines }))
}

async fn shutdown(
    State(s): State<S>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> ApiResult<StatusCode> {
    let who = caller(&s, peer, &headers, Role::Admin)?;
    if !who.local {
        return Err(ApiErr(
            StatusCode::FORBIDDEN,
            "only the operator on the server's own machine can stop it".into(),
            "forbidden",
        ));
    }
    s.audit.line("shutdown", 0, "ok", &who.name);
    if let Some(t) = &s.tasks {
        t.stop();
    }
    s.handle.graceful_shutdown(Some(Duration::from_secs(5)));
    Ok(StatusCode::NO_CONTENT)
}

// ---- starting and stopping ------------------------------------------------

/// A server that listens.
pub struct Running {
    /// Where it listens (the real port when `port = 0` was asked for).
    pub addr: SocketAddr,
    pub fingerprint: String,
    pub server_id: String,
    pub name: String,
    pub archive: PathBuf,
    handle: Handle<SocketAddr>,
    runtime: tokio::runtime::Handle,
    thread: Option<std::thread::JoinHandle<Result<()>>>,
    paths: Paths,
}

impl Running {
    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    /// Stop on Ctrl+C (the terminal's `rds-pacs serve`).
    pub fn stop_on_ctrl_c(&self) {
        let handle = self.handle.clone();
        self.runtime.spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("rds-pacs: stopping");
                handle.graceful_shutdown(Some(Duration::from_secs(5)));
            }
        });
    }

    /// Stop and wait until it has.
    pub fn shutdown(mut self) -> Result<()> {
        self.handle.graceful_shutdown(Some(Duration::from_secs(2)));
        self.join()
    }

    /// Wait until it stops (by Ctrl+C or the operator's *Stop*).
    pub fn wait(mut self) -> Result<()> {
        self.join()
    }

    fn join(&mut self) -> Result<()> {
        let r = match self.thread.take() {
            Some(t) => t
                .join()
                .map_err(|_| anyhow!("the server's thread panicked"))?,
            None => Ok(()),
        };
        RunningFile::clear(&self.paths);
        r
    }
}

/// Start a server for `config`, with its state and work in `paths`, on a
/// runtime of its own (a background thread). Returns once it listens.
pub fn spawn(config: Config, paths: Paths) -> Result<Running> {
    std::fs::create_dir_all(&paths.state)
        .with_context(|| format!("create {}", paths.state.display()))?;
    std::fs::create_dir_all(&paths.data)
        .with_context(|| format!("create {}", paths.data.display()))?;
    // An upload that a stopped server left half received is of no use.
    let _ = std::fs::remove_dir_all(paths.incoming());
    let cert = super::tls::load_or_make(&paths, &config)?;
    let tls = super::tls::server_config(&cert)?;
    let id = identity(&paths)?;
    let auth = Auth::load(&paths)?;
    let archive_root = config.archive_root();
    std::fs::create_dir_all(&archive_root)
        .with_context(|| format!("create {}", archive_root.display()))?;
    let archive_lock = Arc::new(RwLock::new(()));
    let tasks = config.tasks.then(|| {
        Tasks::start(
            config.clone(),
            paths.clone(),
            archive_root.clone(),
            archive_lock.clone(),
        )
    });
    let audit = if paths == Paths::station() {
        Audit::new(config.audit_log, "pacs")
    } else {
        Audit::in_dir(config.audit_log.then(|| paths.data.clone()))
    };
    let name = config.server_name();
    let info = ServerInfo {
        name: name.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        api_version: API_VERSION,
        server_id: id.server_id.clone(),
        fingerprint: cert.fingerprint.clone(),
        tasks: config.tasks,
        max_upload_mb: config.max_upload_mb,
    };
    let ip: IpAddr = config
        .bind
        .trim()
        .parse()
        .with_context(|| format!("bind = '{}' is not an address", config.bind))?;
    let addr = SocketAddr::new(ip, config.port);
    let handle: Handle<SocketAddr> = Handle::new();
    let state = Arc::new(AppState {
        config: config.clone(),
        paths: paths.clone(),
        archive_root: archive_root.clone(),
        info,
        auth: Mutex::new(auth),
        audit,
        archive_lock,
        tasks: tasks.clone(),
        handle: handle.clone(),
        uploads: std::sync::atomic::AtomicU64::new(0),
    });

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("rds-pacs")
        .enable_all()
        .build()
        .context("start the server's runtime")?;
    let runtime = rt.handle().clone();
    let (tx, rx) = std::sync::mpsc::channel::<std::result::Result<SocketAddr, String>>();
    let h = handle.clone();
    let st = state.clone();
    let thread = std::thread::Builder::new()
        .name("rds-pacs server".into())
        .spawn(move || -> Result<()> {
            let st_run = st.clone();
            let result = rt.block_on(async move {
                let st = st_run;
                let h2 = h.clone();
                let tx2 = tx.clone();
                tokio::spawn(async move {
                    if let Some(a) = h2.listening().await {
                        let _ = tx2.send(Ok(a));
                    }
                });
                // `last_seen` reaches the file now and then, not per call.
                let st2 = st.clone();
                tokio::spawn(async move {
                    let mut tick = tokio::time::interval(Duration::from_secs(60));
                    loop {
                        tick.tick().await;
                        st2.auth.lock().unwrap_or_else(|e| e.into_inner()).flush();
                    }
                });
                let app = router(st.clone());
                let r = axum_server::bind_rustls(addr, RustlsConfig::from_config(tls))
                    .handle(h)
                    .serve(app.into_make_service_with_connect_info::<SocketAddr>())
                    .await;
                if let Err(e) = &r {
                    let _ = tx.send(Err(format!("listen on {addr}: {e}")));
                }
                r
            });
            st.auth.lock().unwrap_or_else(|e| e.into_inner()).flush();
            if let Some(t) = &st.tasks {
                t.stop();
            }
            rt.shutdown_timeout(Duration::from_secs(2));
            result.map_err(|e| anyhow!("the server stopped: {e}"))
        })
        .context("start the server's thread")?;

    let bound = match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(Ok(a)) => a,
        Ok(Err(e)) => {
            let _ = thread.join();
            return Err(anyhow!(e));
        }
        Err(_) => {
            handle.shutdown();
            let _ = thread.join();
            return Err(anyhow!("the server did not start listening on {addr}"));
        }
    };

    // What other machines can try: this machine's addresses and name when
    // it listens everywhere, the one address otherwise.
    let shown: Vec<String> = if ip.is_unspecified() {
        let mut v: Vec<String> = super::local::addresses()
            .into_iter()
            .map(|a| SocketAddr::new(a, bound.port()).to_string())
            .collect();
        let host = super::this_device_name();
        if host != "this computer" {
            v.push(format!("{host}:{}", bound.port()));
        }
        v
    } else {
        vec![bound.to_string()]
    };
    RunningFile {
        pid: std::process::id(),
        name: name.clone(),
        server_id: id.server_id.clone(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        bind: config.bind.clone(),
        port: bound.port(),
        fingerprint: cert.fingerprint.clone(),
        started: super::stamp(),
        archive: archive_root.display().to_string(),
        addresses: shown,
    }
    .write(&paths)?;
    state.audit.line(
        "start",
        0,
        "ok",
        &format!(
            "listening on {bound}, version {}",
            env!("CARGO_PKG_VERSION")
        ),
    );
    Ok(Running {
        addr: bound,
        fingerprint: cert.fingerprint,
        server_id: id.server_id,
        name,
        archive: archive_root,
        handle,
        runtime,
        thread: Some(thread),
        paths,
    })
}
