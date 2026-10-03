//! A PACS server as this station sees it: [`Remote`].
//!
//! Every call is synchronous and meant to run inside a background job
//! (`Job::spawn` in the viewer); none of them touches the UI. The HTTP
//! client is `ureq`, the same one the weight download uses, so every build
//! - Android and iOS included - can be a client.
//!
//! ## Trust
//!
//! [`Trust::Pinned`] is the normal case: the server's certificate is
//! self-signed and this station accepts exactly the certificate whose
//! SHA-256 it recorded when it paired, through a rustls verifier that
//! compares fingerprints and then checks the handshake signatures against
//! that certificate's key (so a server that merely *shows* the right
//! certificate without holding its key gets nowhere). A certificate that
//! changed fails the TLS handshake, before the token is ever sent, and is
//! reported as [`Failure::CertificateChanged`].
//!
//! [`Trust::System`] verifies through the system's roots (the bundled
//! Mozilla roots on Android and iOS), for an operator who has a real
//! certificate - a reverse proxy with Let's Encrypt, say.
//!
//! [`Trust::Capture`] accepts any certificate and records its fingerprint:
//! the first look at a server before pairing, when the person compares the
//! fingerprint shown with the one the server's window shows. It is only
//! ever used for `GET /server`, which needs no token.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use serde::de::DeserializeOwned;
use serde::Serialize;

use super::protocol::*;
use super::servers::ServerEntry;
use crate::progress::Progress;

/// How a client decides that it is talking to the right server.
#[derive(Clone, Debug)]
pub enum Trust {
    /// Exactly the certificate with this SHA-256 (64 hex digits).
    Pinned(String),
    /// Whatever the system's roots vouch for, under the name in the URL.
    System,
    /// Anything; the fingerprint seen is recorded. For the first look only.
    Capture,
}

/// Why a call failed, when the reason matters to what the window does next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// No answer: the server is not running, not reachable from here, or
    /// the network is down. Work for it waits in the outbox.
    Unreachable(String),
    /// The server presented a different certificate than the one pinned:
    /// regenerated on the server, or not the server at all.
    CertificateChanged { seen: String },
    /// The token is not (or no longer) accepted: revoked, or the server's
    /// client list was reset.
    Unauthorized(String),
    /// The token is good but its role does not allow this.
    Forbidden(String),
    /// Any other answer that was not a success.
    Refused {
        status: u16,
        code: String,
        message: String,
    },
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::Unreachable(why) => write!(f, "the server cannot be reached: {why}"),
            Failure::CertificateChanged { seen } => write!(
                f,
                "the server presented a different certificate than the one this station \
                 paired with (now {}); pair again once you have checked it in the server's \
                 window",
                super::show_fingerprint(seen)
            ),
            Failure::Unauthorized(m) => write!(f, "not let in: {m}"),
            Failure::Forbidden(m) => write!(f, "not allowed: {m}"),
            Failure::Refused {
                status, message, ..
            } => write!(f, "{message} (HTTP {status})"),
        }
    }
}

impl std::error::Error for Failure {}

/// The [`Failure`] behind an error, when there is one.
pub fn failure_of(e: &anyhow::Error) -> Option<&Failure> {
    e.chain().find_map(|c| c.downcast_ref::<Failure>())
}

/// Did the call fail only because the server could not be reached? Such
/// work is kept for later rather than given up.
pub fn is_offline(e: &anyhow::Error) -> bool {
    matches!(failure_of(e), Some(Failure::Unreachable(_)))
}

/// The verifier behind [`Trust::Pinned`] and [`Trust::Capture`].
#[derive(Debug)]
struct Pinned {
    /// `None`: capture mode.
    want: Option<String>,
    seen: Arc<Mutex<Option<String>>>,
    provider: Arc<CryptoProvider>,
}

impl ServerCertVerifier for Pinned {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        let fp = super::fingerprint(end_entity.as_ref());
        *self.seen.lock().unwrap_or_else(|e| e.into_inner()) = Some(fp.clone());
        match &self.want {
            None => Ok(ServerCertVerified::assertion()),
            Some(w) if *w == fp => Ok(ServerCertVerified::assertion()),
            Some(_) => Err(rustls::Error::General(
                "the certificate is not the one this station paired with".into(),
            )),
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// How much one bundle may carry: the server stops adding files past it,
/// the client asks again for the rest. Small enough to keep memory flat on
/// both sides, large enough that a 4DCT is a handful of requests.
pub const BUNDLE_MB: u64 = 64;

/// One server, ready to be called.
pub struct Remote {
    base: String,
    token: Option<String>,
    agent: ureq::Agent,
    /// The fingerprint the last handshake saw (pinned and capture modes).
    seen: Arc<Mutex<Option<String>>>,
    pinned: bool,
}

impl Remote {
    /// A client for the server at `base` (`https://host:port`).
    pub fn new(base: &str, trust: Trust, token: Option<String>) -> Result<Remote> {
        let base = base.trim().trim_end_matches('/').to_string();
        if !base.starts_with("https://") {
            bail!("a PACS server is reached over HTTPS: '{base}'");
        }
        let seen = Arc::new(Mutex::new(None));
        let mut builder = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(300))
            .timeout_write(Duration::from_secs(300))
            .max_idle_connections_per_host(4);
        let pinned = !matches!(trust, Trust::System);
        let want = match trust {
            Trust::Pinned(fp) => Some(
                super::normalize_fingerprint(&fp)
                    .ok_or_else(|| anyhow::anyhow!("the pinned fingerprint is malformed"))?,
            ),
            Trust::Capture => None,
            Trust::System => {
                // ureq's own configuration: the system's roots on the
                // desktop, the bundled roots on Android and iOS.
                return Ok(Remote {
                    base,
                    token,
                    agent: builder.build(),
                    seen,
                    pinned: false,
                });
            }
        };
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let config = rustls::ClientConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .context("TLS configuration")?
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(Pinned {
                want,
                seen: seen.clone(),
                provider,
            }))
            .with_no_client_auth();
        builder = builder.tls_config(Arc::new(config));
        Ok(Remote {
            base,
            token,
            agent: builder.build(),
            seen,
            pinned,
        })
    }

    /// A client for a paired server.
    pub fn for_server(s: &ServerEntry) -> Result<Remote> {
        Remote::new(&s.url, s.trust(), Some(s.token.clone()))
    }

    /// The first look at a server: who it is and which certificate it
    /// presents, before anything is trusted. Sends no token.
    pub fn probe(base: &str) -> Result<(ServerInfo, String)> {
        let r = Remote::new(base, Trust::Capture, None)?;
        let info: ServerInfo = r.get_json("/server")?;
        let fp = r
            .seen_fingerprint()
            .ok_or_else(|| anyhow::anyhow!("the server presented no certificate"))?;
        Ok((info, fp))
    }

    /// The fingerprint of the certificate the last connection saw.
    pub fn seen_fingerprint(&self) -> Option<String> {
        self.seen.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    fn url(&self, path: &str) -> String {
        format!("{}{PREFIX}{path}", self.base)
    }

    fn request(&self, method: &str, path: &str) -> ureq::Request {
        let r = self.agent.request(method, &self.url(path));
        match &self.token {
            Some(t) => r.set("Authorization", &format!("Bearer {t}")),
            None => r,
        }
    }

    /// Turn ureq's error into a [`Failure`] the window can act on.
    fn fail(&self, e: ureq::Error) -> anyhow::Error {
        match e {
            ureq::Error::Status(status, resp) => {
                let body = resp.into_string().unwrap_or_default();
                let api: ApiError = serde_json::from_str(&body).unwrap_or(ApiError {
                    error: body.chars().take(200).collect(),
                    code: String::new(),
                });
                let message = if api.error.is_empty() {
                    format!("HTTP {status}")
                } else {
                    api.error
                };
                anyhow::Error::new(match status {
                    401 => Failure::Unauthorized(message),
                    403 => Failure::Forbidden(message),
                    _ => Failure::Refused {
                        status,
                        code: api.code,
                        message,
                    },
                })
            }
            ureq::Error::Transport(t) => {
                let text = t.to_string();
                // A pinned connection that saw a certificate other than the
                // pinned one failed for that reason, whatever the transport
                // error says on top.
                if self.pinned && text.contains("not the one this station paired with") {
                    if let Some(seen) = self.seen_fingerprint() {
                        return anyhow::Error::new(Failure::CertificateChanged { seen });
                    }
                }
                anyhow::Error::new(Failure::Unreachable(text))
            }
        }
    }

    fn call(&self, req: ureq::Request) -> Result<ureq::Response> {
        req.call().map_err(|e| self.fail(e))
    }

    fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let resp = self.call(self.request("GET", path))?;
        serde_json::from_reader(resp.into_reader()).with_context(|| format!("read {path}"))
    }

    fn post_json<B: Serialize, T: DeserializeOwned>(&self, path: &str, body: &B) -> Result<T> {
        let text = serde_json::to_string(body).expect("plain data serialises");
        let resp = self
            .request("POST", path)
            .set("Content-Type", "application/json")
            .send_string(&text)
            .map_err(|e| self.fail(e))?;
        serde_json::from_reader(resp.into_reader()).with_context(|| format!("read {path}"))
    }

    fn post_empty(&self, path: &str) -> Result<()> {
        self.request("POST", path)
            .send_string("")
            .map_err(|e| self.fail(e))?;
        Ok(())
    }

    /// The HTTP status a request to `path` (under the API prefix, sent as
    /// it is, unchecked) is answered with; 0 when nothing answered. For the
    /// test suites, which ask the server things a well-behaved client never
    /// would.
    #[doc(hidden)]
    pub fn raw_status(&self, method: &str, path: &str) -> u16 {
        match self.request(method, path).call() {
            Ok(r) => r.status(),
            Err(ureq::Error::Status(code, _)) => code,
            Err(_) => 0,
        }
    }

    // ---- who ------------------------------------------------------------

    /// Who the server is; checks that its API is one this station speaks.
    pub fn info(&self) -> Result<ServerInfo> {
        let info: ServerInfo = self.get_json("/server")?;
        if info.api_version != API_VERSION {
            bail!(
                "the server speaks version {} of the PACS protocol, this station version {}; \
                 update the older of the two",
                info.api_version,
                API_VERSION
            );
        }
        Ok(info)
    }

    /// Trade a pairing code for a token.
    pub fn pair(&self, code: &str, client_name: &str) -> Result<PairResponse> {
        self.post_json(
            "/pair",
            &PairRequest {
                code: code.trim().to_uppercase(),
                client_name: client_name.trim().to_string(),
            },
        )
    }

    pub fn whoami(&self) -> Result<WhoAmI> {
        self.get_json("/whoami")
    }

    // ---- the archive ----------------------------------------------------

    pub fn patients(&self) -> Result<Vec<RemotePatient>> {
        Ok(self.get_json::<Listing>("/patients")?.patients)
    }

    pub fn manifest(&self, study_uid: &str) -> Result<Manifest> {
        check_uid(study_uid)?;
        self.get_json(&format!("/studies/{study_uid}"))
    }

    /// One instance into `dest`. Returns its size.
    pub fn instance(&self, study_uid: &str, sop_uid: &str, dest: &Path) -> Result<u64> {
        check_uid(study_uid)?;
        check_uid(sop_uid)?;
        let resp =
            self.call(self.request("GET", &format!("/studies/{study_uid}/instances/{sop_uid}")))?;
        save_body(resp, dest)
    }

    /// As many of `sops` as fit one bundle, unpacked into `dir` (one file
    /// each, named by a counter: the archive files them by their tags).
    /// Returns how many files arrived; the caller asks again for the rest.
    pub fn bundle(&self, study_uid: &str, sops: &[String], dir: &Path) -> Result<usize> {
        check_uid(study_uid)?;
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        let text = serde_json::to_string(&BundleRequest {
            sops: sops.to_vec(),
        })
        .expect("plain data serialises");
        let resp = self
            .request("POST", &format!("/studies/{study_uid}/bundle"))
            .set("Content-Type", "application/json")
            .send_string(&text)
            .map_err(|e| self.fail(e))?;
        let zip_path = dir.join(".bundle.zip");
        save_body(resp, &zip_path)?;
        let n = unzip_flat(&zip_path, dir, u64::MAX)?;
        let _ = std::fs::remove_file(&zip_path);
        Ok(n)
    }

    /// Send a zip of DICOM files (see [`zip_files`]) into the server's
    /// archive.
    pub fn upload_zip(&self, zip: &Path) -> Result<UploadSummary> {
        let len = std::fs::metadata(zip)
            .with_context(|| format!("read {}", zip.display()))?
            .len();
        let file = std::fs::File::open(zip).with_context(|| format!("read {}", zip.display()))?;
        let resp = self
            .request("POST", "/studies")
            .set("Content-Type", "application/zip")
            .set("Content-Length", &len.to_string())
            .send(file)
            .map_err(|e| self.fail(e))?;
        serde_json::from_reader(resp.into_reader()).context("read the upload's answer")
    }

    /// Send files into the server's archive, in bundles of at most
    /// [`BUNDLE_MB`] (or the server's own limit, when that is lower). What
    /// is not DICOM is counted as skipped by the server.
    ///
    /// The station's role is asked first. A server that refuses an upload
    /// answers before it has read the body and closes the connection, and
    /// a client still sending sees that as a broken connection, not as the
    /// refusal - which would put work into the outbox that the server will
    /// never take.
    pub fn upload_files(
        &self,
        files: &[PathBuf],
        scratch: &Path,
        p: &Progress,
    ) -> Result<UploadSummary> {
        let who = self.whoami()?;
        if !who.role.allows(Role::Edit) {
            return Err(anyhow::Error::new(Failure::Forbidden(format!(
                "'{}' is paired with the {} role, which cannot send to the server",
                who.name,
                who.role.label()
            ))));
        }
        let limit = match self.info()?.max_upload_mb {
            0 => BUNDLE_MB,
            m => m.min(BUNDLE_MB),
        };
        let mut total = UploadSummary::default();
        let batches = batches_of(files, limit * 1024 * 1024);
        std::fs::create_dir_all(scratch)
            .with_context(|| format!("create {}", scratch.display()))?;
        for (i, batch) in batches.iter().enumerate() {
            if p.cancelled() {
                bail!(crate::progress::CANCELLED);
            }
            p.set(format!(
                "Sending {} file(s), part {}/{}",
                batch.len(),
                i + 1,
                batches.len()
            ));
            let zip = scratch.join(format!(".upload-{i}.zip"));
            zip_files(batch, &zip)?;
            let r = self.upload_zip(&zip);
            let _ = std::fs::remove_file(&zip);
            total.add(&r?);
        }
        Ok(total)
    }

    pub fn delete_study(&self, study_uid: &str) -> Result<()> {
        check_uid(study_uid)?;
        self.request("DELETE", &format!("/studies/{study_uid}"))
            .call()
            .map_err(|e| self.fail(e))?;
        Ok(())
    }

    pub fn delete_patient(&self, key: &str) -> Result<()> {
        if key.is_empty() || key.contains(['/', '\\']) || key == "." || key == ".." {
            bail!("'{key}' is not a patient of the archive");
        }
        self.request("DELETE", &format!("/patients/{}", encode(key)))
            .call()
            .map_err(|e| self.fail(e))?;
        Ok(())
    }

    // ---- tasks ----------------------------------------------------------

    pub fn workflows(&self) -> Result<Vec<WorkflowInfo>> {
        Ok(self.get_json::<WorkflowList>("/workflows")?.workflows)
    }

    pub fn submit(&self, req: &TaskRequest) -> Result<Submitted> {
        self.post_json("/tasks", req)
    }

    pub fn tasks(&self) -> Result<Vec<TaskInfo>> {
        Ok(self.get_json::<TaskList>("/tasks")?.tasks)
    }

    /// One task, with its log lines from `log_from` on.
    pub fn task(&self, id: &str, log_from: usize) -> Result<TaskInfo> {
        check_id(id)?;
        self.get_json(&format!("/tasks/{id}?log_from={log_from}"))
    }

    pub fn cancel_task(&self, id: &str) -> Result<()> {
        check_id(id)?;
        self.post_empty(&format!("/tasks/{id}/cancel"))
    }

    /// A file a task wrote (a report, a table) into `dest`.
    pub fn task_file(&self, id: &str, name: &str, dest: &Path) -> Result<u64> {
        check_id(id)?;
        let resp =
            self.call(self.request("GET", &format!("/tasks/{id}/files/{}", encode_path(name))))?;
        save_body(resp, dest)
    }

    // ---- the operator ---------------------------------------------------

    pub fn new_pairing_code(&self, role: Role, minutes: u64) -> Result<PairingCode> {
        self.post_json("/admin/pairing-codes", &CodeRequest { role, minutes })
    }

    pub fn clients(&self) -> Result<Vec<ClientInfo>> {
        Ok(self.get_json::<ClientList>("/admin/clients")?.clients)
    }

    pub fn revoke(&self, name: &str) -> Result<()> {
        self.post_empty(&format!("/admin/clients/{}/revoke", encode(name)))
    }

    pub fn audit(&self, lines: usize) -> Result<Vec<String>> {
        Ok(self
            .get_json::<AuditTail>(&format!("/admin/audit?lines={lines}"))?
            .lines)
    }

    /// Ask the server to stop (the local operator only).
    pub fn shutdown(&self) -> Result<()> {
        self.post_empty("/admin/shutdown")
    }
}

fn check_uid(uid: &str) -> Result<()> {
    if !crate::archive::is_uid(uid) {
        bail!("'{uid}' is not a DICOM UID");
    }
    Ok(())
}

fn check_id(id: &str) -> Result<()> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        bail!("'{id}' is not a task");
    }
    Ok(())
}

/// Percent-encode one path segment.
pub fn encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Percent-encode a relative path, keeping its slashes.
fn encode_path(s: &str) -> String {
    s.split('/').map(encode).collect::<Vec<_>>().join("/")
}

/// Write a response body to `dest` through a `.part` file, so a cut-off
/// download never looks like a finished one.
fn save_body(resp: ureq::Response, dest: &Path) -> Result<u64> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let part = dest.with_extension("part");
    let mut out =
        std::fs::File::create(&part).with_context(|| format!("write {}", part.display()))?;
    let mut reader = resp.into_reader();
    let n = std::io::copy(&mut reader, &mut out).map_err(|e| {
        let _ = std::fs::remove_file(&part);
        anyhow::Error::new(Failure::Unreachable(format!("the download broke off: {e}")))
    })?;
    out.flush().ok();
    drop(out);
    std::fs::rename(&part, dest).with_context(|| format!("write {}", dest.display()))?;
    Ok(n)
}

/// Split `files` into groups of at most `max_bytes` each (a file larger
/// than that travels alone).
pub fn batches_of(files: &[PathBuf], max_bytes: u64) -> Vec<Vec<PathBuf>> {
    let mut out: Vec<Vec<PathBuf>> = Vec::new();
    let mut cur: Vec<PathBuf> = Vec::new();
    let mut size = 0u64;
    for f in files {
        let len = std::fs::metadata(f).map(|m| m.len()).unwrap_or(0);
        if !cur.is_empty() && size + len > max_bytes {
            out.push(std::mem::take(&mut cur));
            size = 0;
        }
        size += len;
        cur.push(f.clone());
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Pack files into a zip without compressing them (DICOM pixel data does
/// not compress usefully, and storing is fast). Entries are numbered: the
/// receiving archive files each by its own tags, not by its name.
pub fn zip_files(files: &[PathBuf], zip_path: &Path) -> Result<()> {
    let out =
        std::fs::File::create(zip_path).with_context(|| format!("write {}", zip_path.display()))?;
    let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(out));
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true);
    for (i, f) in files.iter().enumerate() {
        zip.start_file(format!("{i:06}.dcm"), opts)
            .context("write the bundle")?;
        let mut src = std::fs::File::open(f).with_context(|| format!("read {}", f.display()))?;
        std::io::copy(&mut src, &mut zip).with_context(|| format!("read {}", f.display()))?;
    }
    zip.finish().context("write the bundle")?;
    Ok(())
}

/// Unpack a zip into `dir`, one file per entry under a numbered name -
/// never under the entry's own name, which could point anywhere. Stops
/// with an error past `max_bytes` unpacked. Returns the number of files.
pub fn unzip_flat(zip_path: &Path, dir: &Path, max_bytes: u64) -> Result<usize> {
    let file =
        std::fs::File::open(zip_path).with_context(|| format!("read {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .context("the bundle is not a zip file")?;
    let mut total = 0u64;
    let mut n = 0usize;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).context("read the bundle")?;
        if entry.is_dir() {
            continue;
        }
        let dest = dir.join(format!("{:06}-{i:06}.dcm", std::process::id() % 1_000_000));
        let mut out =
            std::fs::File::create(&dest).with_context(|| format!("write {}", dest.display()))?;
        let mut limited = (&mut entry).take(max_bytes.saturating_sub(total).saturating_add(1));
        let copied = std::io::copy(&mut limited, &mut out)
            .with_context(|| format!("write {}", dest.display()))?;
        total += copied;
        if total > max_bytes {
            drop(out);
            let _ = std::fs::remove_file(&dest);
            bail!("the bundle unpacks to more than it may");
        }
        n += 1;
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn files_pack_into_bounded_bundles_and_unpack_under_safe_names() {
        let dir = std::env::temp_dir().join("rds_pacs_zip");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("in")).unwrap();
        let mut files = Vec::new();
        for i in 0..5 {
            let f = dir.join("in").join(format!("f{i}"));
            std::fs::write(&f, vec![i as u8; 100]).unwrap();
            files.push(f);
        }
        let groups = batches_of(&files, 250);
        assert_eq!(
            groups.iter().map(|g| g.len()).collect::<Vec<_>>(),
            vec![2, 2, 1]
        );
        let zip = dir.join("b.zip");
        zip_files(&files, &zip).unwrap();
        std::fs::create_dir_all(dir.join("out")).unwrap();
        assert_eq!(unzip_flat(&zip, &dir.join("out"), u64::MAX).unwrap(), 5);
        let mut sizes: Vec<u64> = std::fs::read_dir(dir.join("out"))
            .unwrap()
            .map(|e| e.unwrap().metadata().unwrap().len())
            .collect();
        sizes.sort();
        assert_eq!(sizes, vec![100; 5]);
        std::fs::create_dir_all(dir.join("small")).unwrap();
        assert!(
            unzip_flat(&zip, &dir.join("small"), 250).is_err(),
            "a bundle larger than allowed is refused"
        );

        // An entry named to climb out of the folder lands inside it.
        let evil = dir.join("evil.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&evil).unwrap());
            z.start_file(
                "../../escaped.dcm",
                zip::write::SimpleFileOptions::default(),
            )
            .unwrap();
            z.write_all(b"x").unwrap();
            z.finish().unwrap();
        }
        std::fs::create_dir_all(dir.join("safe")).unwrap();
        assert_eq!(unzip_flat(&evil, &dir.join("safe"), 1000).unwrap(), 1);
        assert!(!dir.join("escaped.dcm").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_segments_are_encoded() {
        assert_eq!(encode("Doe John"), "Doe%20John");
        assert_eq!(encode("a/b"), "a%2Fb");
        assert_eq!(encode_path("reports/a b.csv"), "reports/a%20b.csv");
    }

    #[test]
    fn only_https_and_well_formed_pins_are_accepted() {
        assert!(Remote::new("http://host:1", Trust::System, None).is_err());
        assert!(Remote::new("https://host:1", Trust::Pinned("12".into()), None).is_err());
        assert!(Remote::new("https://host:1", Trust::Pinned("ab".repeat(32)), None).is_ok());
        assert!(Remote::new("https://host:1", Trust::Capture, None).is_ok());
    }

    #[test]
    fn an_unreachable_server_is_offline_not_a_refusal() {
        // Port 1 on the loopback: nothing listens there.
        let r = Remote::new("https://127.0.0.1:1", Trust::Pinned("ab".repeat(32)), None).unwrap();
        let e = r.info().unwrap_err();
        assert!(is_offline(&e), "{e:#}");
    }
}
