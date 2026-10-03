//! The local copy of a server's studies: the *mirror* mode of working.
//!
//! A client keeps one folder per server it has paired with:
//!
//! ```text
//! <data folder>/pacs-mirror/<server id>/
//!     archive/        an archive in the layout of crate::archive
//!     outbox/<n>/     objects made here that the server has not received
//!     listing.json    the server's patient list as last seen
//! ```
//!
//! Because `archive/` is an ordinary archive, everything the station does
//! with its own archive works on it unchanged - [`Archive::scan`] lists it,
//! a study folder in it loads like any DICOM folder - and every tool of the
//! viewer works on a pulled study, connected or not.
//!
//! ## Sync is set differences
//!
//! Both archives are append-only, every file is named by its SOP Instance
//! UID, and every object the station derives (a structure set, a
//! segmentation) gets fresh UIDs. So the two copies of a study can only
//! differ by instances one of them lacks, never by two versions of one
//! instance: pulling is "what the server has and the mirror does not",
//! pushing the reverse, and there is nothing to merge and nothing to
//! overwrite, ever.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use super::client::{is_offline, Remote};
use super::protocol::{RemotePatient, UploadSummary};
use crate::archive::{self, Archive};
use crate::progress::Progress;
use crate::settings;

/// One server's copy on this device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mirror {
    root: PathBuf,
}

/// What a pull did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PullSummary {
    /// Instances fetched now.
    pub fetched: usize,
    /// Instances that were here already.
    pub present: usize,
    pub studies: usize,
}

impl PullSummary {
    pub fn add(&mut self, o: &PullSummary) {
        self.fetched += o.fetched;
        self.present += o.present;
        self.studies += o.studies;
    }

    pub fn describe(&self) -> String {
        format!(
            "{} file(s) fetched from {} study(ies){}",
            self.fetched,
            self.studies,
            if self.present > 0 {
                format!(", {} here already", self.present)
            } else {
                String::new()
            }
        )
    }
}

/// How sending something to the server ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Sent {
    /// The server has it.
    Delivered(UploadSummary),
    /// The server could not be reached; it waits in the outbox (this many
    /// files) and goes with the next sync.
    Queued(usize),
}

impl Sent {
    pub fn describe(&self) -> String {
        match self {
            Sent::Delivered(s) => format!("sent: {}", s.describe()),
            Sent::Queued(n) => format!(
                "the server could not be reached: {n} file(s) wait in the outbox and go with \
                 the next sync"
            ),
        }
    }
}

/// What a sync did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SyncSummary {
    pub pulled: PullSummary,
    /// Instances of mirrored studies the server lacked, sent now.
    pub pushed: UploadSummary,
    /// Outbox batches delivered.
    pub outbox_sent: usize,
    /// Mirrored studies the server no longer has (kept here).
    pub gone: Vec<String>,
}

impl SyncSummary {
    pub fn describe(&self) -> String {
        let mut parts = vec![self.pulled.describe()];
        if self.pushed.stored > 0 {
            parts.push(format!("{} file(s) sent", self.pushed.stored));
        }
        if self.outbox_sent > 0 {
            parts.push(format!("{} outbox batch(es) delivered", self.outbox_sent));
        }
        if !self.gone.is_empty() {
            parts.push(format!(
                "{} study(ies) kept here are no longer on the server",
                self.gone.len()
            ));
        }
        parts.join("; ")
    }
}

/// Where every mirror lives.
pub fn mirrors_root() -> PathBuf {
    settings::data_dir().join("pacs-mirror")
}

impl Mirror {
    /// The mirror of the server with this id, under the station's data
    /// folder.
    pub fn for_server(server_id: &str) -> Mirror {
        Mirror::at(mirrors_root().join(safe_id(server_id)))
    }

    /// A mirror rooted at `root` (the test suites').
    pub fn at(root: impl Into<PathBuf>) -> Mirror {
        Mirror { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn archive(&self) -> Archive {
        Archive::new(self.root.join("archive"))
    }

    pub fn outbox(&self) -> PathBuf {
        self.root.join("outbox")
    }

    fn scratch(&self, what: &str) -> PathBuf {
        self.root.join(format!(
            ".{what}-{}-{}",
            std::process::id(),
            super::unix_now()
        ))
    }

    /// The server's listing as last seen, for showing it while offline.
    pub fn cached_listing(&self) -> Vec<RemotePatient> {
        std::fs::read_to_string(self.root.join("listing.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }

    pub fn save_listing(&self, patients: &[RemotePatient]) {
        let _ = std::fs::create_dir_all(&self.root);
        if let Ok(text) = serde_json::to_string(patients) {
            let _ = std::fs::write(self.root.join("listing.json"), text);
        }
    }

    /// The SOP Instance UIDs of a study held here.
    pub fn local_sops(&self, study_uid: &str) -> BTreeSet<String> {
        self.archive()
            .find_study(study_uid)
            .and_then(|d| archive::instances(&d).ok())
            .map(|v| v.into_iter().map(|i| i.sop_uid).collect())
            .unwrap_or_default()
    }

    /// The studies held here, by UID.
    pub fn local_studies(&self) -> BTreeSet<String> {
        self.archive()
            .scan()
            .map(|ps| {
                ps.into_iter()
                    .flat_map(|p| p.studies.into_iter().map(|s| s.study_uid))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The batches waiting in the outbox, oldest first.
    pub fn pending(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(self.outbox())
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .map(|e| e.path())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    /// How many files wait in the outbox.
    pub fn pending_files(&self) -> usize {
        self.pending().iter().map(|d| files_under(d).len()).sum()
    }

    /// The size of everything held here, in bytes.
    pub fn size_bytes(&self) -> u64 {
        walkdir::WalkDir::new(&self.root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
            .sum()
    }

    /// Drop a study from this copy. The server is not touched.
    pub fn remove_study(&self, study_uid: &str) -> Result<()> {
        let a = self.archive();
        match a.find_study(study_uid) {
            Some(dir) => {
                a.remove(&dir)?;
                // A patient left without studies goes too.
                if let Some(pdir) = dir.parent() {
                    let empty = std::fs::read_dir(pdir)
                        .map(|rd| {
                            !rd.filter_map(|e| e.ok())
                                .any(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                        })
                        .unwrap_or(false);
                    if empty {
                        let _ = a.remove(pdir);
                    }
                }
                Ok(())
            }
            None => bail!("this study is not held here"),
        }
    }

    /// Put a folder of files into the outbox (moved when it can be, copied
    /// otherwise). Returns the number of files.
    fn queue(&self, folder: &Path) -> Result<usize> {
        let dest = self
            .outbox()
            .join(format!("{}-{}", super::unix_now(), std::process::id()));
        let mut dest = dest;
        let mut n = 1;
        while dest.exists() {
            dest = dest.with_extension(n.to_string());
            n += 1;
        }
        std::fs::create_dir_all(self.outbox())
            .with_context(|| format!("create {}", self.outbox().display()))?;
        if std::fs::rename(folder, &dest).is_err() {
            std::fs::create_dir_all(&dest)?;
            for (i, f) in files_under(folder).iter().enumerate() {
                std::fs::copy(f, dest.join(format!("{i:06}.dcm")))
                    .with_context(|| format!("copy {}", f.display()))?;
            }
        }
        Ok(files_under(&dest).len())
    }
}

/// A server id as a folder name.
fn safe_id(id: &str) -> String {
    let s: String = id
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
        .take(64)
        .collect();
    if s.is_empty() {
        "server".into()
    } else {
        s
    }
}

/// Every regular file under a folder.
pub fn files_under(dir: &Path) -> Vec<PathBuf> {
    walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect()
}

/// Bring one study's missing instances from the server into the mirror.
///
/// Bundle after bundle, until the mirror holds everything the server
/// listed. Each bundle is filed as it arrives, so a pull that breaks off
/// keeps what it got and the next one fetches only the rest.
pub fn pull_study(
    remote: &Remote,
    mirror: &Mirror,
    study_uid: &str,
    p: &Progress,
) -> Result<PullSummary> {
    let manifest = remote.manifest(study_uid)?;
    let have = mirror.local_sops(study_uid);
    let mut missing: Vec<String> = manifest
        .instances
        .iter()
        .map(|i| i.sop_uid.clone())
        .filter(|s| !have.contains(s))
        .collect();
    let total = missing.len();
    let mut sum = PullSummary {
        fetched: 0,
        present: manifest.instances.len() - total,
        studies: 1,
    };
    let scratch = mirror.scratch("pull");
    let result = (|| -> Result<()> {
        while !missing.is_empty() {
            if p.cancelled() {
                bail!(crate::progress::CANCELLED);
            }
            p.set(format!(
                "Fetching {}/{} file(s) of study …{}",
                sum.fetched,
                total,
                tail(study_uid)
            ));
            let n = remote.bundle(study_uid, &missing, &scratch)?;
            if n == 0 {
                bail!("the server sent nothing of what it listed");
            }
            mirror.archive().import(&scratch, &Progress::default())?;
            let _ = std::fs::remove_dir_all(&scratch);
            let now = mirror.local_sops(study_uid);
            let before = missing.len();
            missing.retain(|s| !now.contains(s));
            let got = before - missing.len();
            if got == 0 {
                bail!(
                    "the files the server sent do not file under this study; {} file(s) not \
                     pulled",
                    missing.len()
                );
            }
            sum.fetched += got;
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    result?;
    Ok(sum)
}

/// Pull every study of a patient.
pub fn pull_patient(
    remote: &Remote,
    mirror: &Mirror,
    patient: &RemotePatient,
    p: &Progress,
) -> Result<PullSummary> {
    let mut sum = PullSummary::default();
    for st in &patient.studies {
        sum.add(&pull_study(remote, mirror, &st.study_uid, p)?);
    }
    Ok(sum)
}

/// Send the files of `folder` (derived objects written by the export, or
/// a study to put on the server) and, once the server has them, file them
/// into the mirror too, so both copies agree. When the server cannot be
/// reached the folder goes into the outbox instead; any other refusal is
/// an error and the folder is left where it is.
pub fn send_folder(remote: &Remote, mirror: &Mirror, folder: &Path, p: &Progress) -> Result<Sent> {
    let files = files_under(folder);
    if files.is_empty() {
        bail!("nothing to send");
    }
    match remote.upload_files(&files, &mirror.scratch("send"), p) {
        Ok(sum) => {
            mirror.archive().import(folder, &Progress::default())?;
            Ok(Sent::Delivered(sum))
        }
        Err(e) if is_offline(&e) => Ok(Sent::Queued(mirror.queue(folder)?)),
        Err(e) => Err(e),
    }
}

/// Deliver what waits in the outbox, oldest first. Stops at the first batch
/// the server cannot be reached for (the rest stays). Returns the batches
/// delivered and what the server did with them.
pub fn flush_outbox(
    remote: &Remote,
    mirror: &Mirror,
    p: &Progress,
) -> Result<(usize, UploadSummary)> {
    let mut sent = 0;
    let mut total = UploadSummary::default();
    for batch in mirror.pending() {
        let files = files_under(&batch);
        if files.is_empty() {
            let _ = std::fs::remove_dir_all(&batch);
            continue;
        }
        let sum = remote.upload_files(&files, &mirror.scratch("send"), p)?;
        mirror.archive().import(&batch, &Progress::default())?;
        std::fs::remove_dir_all(&batch).with_context(|| format!("remove {}", batch.display()))?;
        total.add(&sum);
        sent += 1;
    }
    Ok((sent, total))
}

/// Both directions for everything held here: deliver the outbox, then for
/// every mirrored study fetch what the server has that is not here and
/// send what is here that the server lacks. Studies never pulled are not
/// fetched (that would copy the whole server), and a mirrored study the
/// server no longer has is reported, kept here and not sent back.
pub fn sync(remote: &Remote, mirror: &Mirror, p: &Progress) -> Result<SyncSummary> {
    let mut out = SyncSummary::default();
    p.set("Reading the server's archive");
    let listing = remote.patients()?;
    mirror.save_listing(&listing);
    let (n, _) = flush_outbox(remote, mirror, p)?;
    out.outbox_sent = n;
    let on_server: BTreeSet<String> = listing
        .iter()
        .flat_map(|pt| pt.studies.iter().map(|s| s.study_uid.clone()))
        .collect();
    for uid in mirror.local_studies() {
        if p.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        if !on_server.contains(&uid) {
            out.gone.push(uid);
            continue;
        }
        out.pulled.add(&pull_study(remote, mirror, &uid, p)?);
        // What is here and not there: objects whose earlier send broke off.
        let manifest = remote.manifest(&uid)?;
        let there: BTreeSet<String> = manifest.instances.into_iter().map(|i| i.sop_uid).collect();
        let extra: Vec<PathBuf> = mirror
            .archive()
            .find_study(&uid)
            .and_then(|d| archive::instances(&d).ok())
            .unwrap_or_default()
            .into_iter()
            .filter(|i| !there.contains(&i.sop_uid))
            .map(|i| i.path)
            .collect();
        if !extra.is_empty() {
            p.set(format!("Sending {} file(s) the server lacks", extra.len()));
            let s = remote.upload_files(&extra, &mirror.scratch("send"), p)?;
            out.pushed.add(&s);
        }
    }
    Ok(out)
}

/// The last eight characters of a UID, for a progress line.
fn tail(uid: &str) -> &str {
    let n = uid.len();
    &uid[n.saturating_sub(8)..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_ids_become_safe_folder_names() {
        assert_eq!(safe_id("aB3-_x"), "aB3-_x");
        assert_eq!(safe_id("../../etc"), "etc");
        assert_eq!(safe_id(""), "server");
        assert!(Mirror::for_server("../x").root().ends_with("x"));
    }

    #[test]
    fn the_outbox_keeps_what_could_not_be_sent() {
        let root = std::env::temp_dir().join("rds_pacs_outbox");
        let _ = std::fs::remove_dir_all(&root);
        let m = Mirror::at(&root);
        assert!(m.pending().is_empty());
        let src = root.join("export");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("a.dcm"), b"a").unwrap();
        std::fs::write(src.join("sub").join("b.dcm"), b"b").unwrap();
        assert_eq!(m.queue(&src).unwrap(), 2);
        assert_eq!(m.pending().len(), 1);
        assert_eq!(m.pending_files(), 2);
        assert!(m.size_bytes() >= 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn summaries_read_as_sentences() {
        let s = SyncSummary {
            pulled: PullSummary {
                fetched: 3,
                present: 2,
                studies: 1,
            },
            outbox_sent: 1,
            gone: vec!["1.2".into()],
            ..Default::default()
        };
        let text = s.describe();
        assert!(text.contains("3 file(s) fetched"));
        assert!(text.contains("1 outbox batch(es)"));
        assert!(text.contains("no longer on the server"));
        assert!(Sent::Queued(4).describe().contains("4 file(s) wait"));
    }
}
