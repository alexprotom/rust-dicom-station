//! *Tools ▶ Download test data*: fetch the repository's bundled patient data
//! (`data-test/`: TCIA 4D-Lung P102 in full - a ten-phase 4DFBCT with an RT
//! Structure Set per phase, and the matching ten-phase 4DCBCT; see
//! `docs/example-data.md`) from GitHub into a local folder, so an installed
//! copy of the viewer has real data to open without a clone of the source
//! tree. It is close to a gigabyte, which is what the resumability below is
//! for.
//!
//! Two requests are involved, neither needing a token:
//!
//! 1. the folder's contents are listed through the git *trees* API of the
//!    repository ([`tree_url`]): one call answers with every path in the
//!    branch and the size of each blob, and everything under [`FOLDER`] is
//!    kept ([`parse_tree`]);
//! 2. each file is then fetched from `raw.githubusercontent.com`
//!    ([`raw_url`]), which serves blobs without the API's rate limit, through
//!    the same downloader the model weights use (`nn::cache`).
//!
//! The download is resumable in the plain sense: a file already on disk with
//! the size the listing gives is not fetched again, so a run interrupted or
//! cancelled halfway continues where it stopped. Every file lands on a
//! temporary name and is renamed once complete, exactly like a weight file.
//!
//! The listing names the branch [`BRANCH`] rather than a release tag: the
//! data changes far more rarely than the code, and a folder that is on the
//! release branch is what every version of the viewer can open. When the
//! API refuses the listing for its rate limit, the download goes ahead with
//! the built-in list of the folder's files ([`builtin_listing`]) instead.

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

use crate::nn::cache::download_to_file;
use crate::progress::{ProgressSink, CANCELLED};
use crate::settings;

/// The repository the data is fetched from, `owner/name`.
pub const REPO: &str = "alexprotom/rust-dicom-station";
/// The branch that is listed: `main` carries every released state.
pub const BRANCH: &str = "main";
/// The folder inside the repository, and the name of the folder written
/// locally.
pub const FOLDER: &str = "data-test";
/// The collection and patient folders inside [`FOLDER`], the level the two
/// studies sit under (`data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS`). Shown in
/// the window; [`datasets_in`] finds the studies by walking, not by this.
pub const PATIENT: &str = "TCIA_4D-LUNG/P102";

/// One file of the folder, as the listing describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteEntry {
    /// Path relative to [`FOLDER`], with `/` separators
    /// (`TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm`).
    pub rel: String,
    /// Size in bytes, as the listing reports it.
    pub bytes: u64,
}

impl RemoteEntry {
    /// Where the file lands under the local folder.
    pub fn path_in(&self, dir: &Path) -> PathBuf {
        let mut p = dir.to_path_buf();
        for part in self.rel.split('/') {
            p.push(part);
        }
        p
    }

    /// True when the file is already there, with the size the listing gives.
    /// An entry of unknown size (the built-in listing) counts as present
    /// when the file exists and is not empty.
    pub fn is_present(&self, dir: &Path) -> bool {
        self.path_in(dir)
            .metadata()
            .map(|m| {
                m.is_file()
                    && if self.bytes == 0 {
                        m.len() > 0
                    } else {
                        m.len() == self.bytes
                    }
            })
            .unwrap_or(false)
    }
}

/// What a finished download reports.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Files fetched in this run.
    pub downloaded: usize,
    /// Files that were already there and were left alone.
    pub kept: usize,
    /// Bytes fetched in this run.
    pub bytes: u64,
    /// The local folder, `<dir>`, holding the studies.
    pub dir: PathBuf,
    /// The study folders under `dir` ([`datasets_in`]), sorted, so the
    /// caller can open the first one.
    pub datasets: Vec<PathBuf>,
}

/// The default destination: `<data folder>/data-test`, beside the models and
/// the generated test study.
pub fn default_output_dir() -> PathBuf {
    settings::data_dir().join(FOLDER)
}

/// The git trees API call that lists every path in the branch.
pub fn tree_url() -> String {
    format!("https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1")
}

/// Where one file of the folder is served from.
pub fn raw_url(rel: &str) -> String {
    format!("https://raw.githubusercontent.com/{REPO}/{BRANCH}/{FOLDER}/{rel}")
}

/// The web page of the folder, for the window's link.
pub fn browse_url() -> String {
    format!("https://github.com/{REPO}/tree/{BRANCH}/{FOLDER}")
}

/// Keep the blobs under [`FOLDER`] out of a trees API answer.
///
/// The answer is `{"sha": .., "tree": [{"path", "type", "size", ..}, ..],
/// "truncated": bool}`; `type` is `blob` for a file and `tree` for a folder,
/// and only blobs carry a size. Entries come back in path order, which is
/// kept, so the studies download one after the other, a phase at a time,
/// and an interrupted run leaves whole phases behind rather than a
/// scattering of slices.
pub fn parse_tree(json: &str) -> Result<Vec<RemoteEntry>> {
    let v: serde_json::Value =
        serde_json::from_str(json).context("GitHub answered with something other than JSON")?;
    if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
        // The API's own error shape: {"message": "API rate limit exceeded ..."}.
        bail!("GitHub: {msg}");
    }
    let tree = v
        .get("tree")
        .and_then(|t| t.as_array())
        .context("the listing has no \"tree\" array")?;
    let prefix = format!("{FOLDER}/");
    let mut out = Vec::new();
    for e in tree {
        if e.get("type").and_then(|t| t.as_str()) != Some("blob") {
            continue;
        }
        let Some(path) = e.get("path").and_then(|p| p.as_str()) else {
            continue;
        };
        let Some(rel) = path.strip_prefix(&prefix) else {
            continue;
        };
        if rel.is_empty()
            || rel
                .split('/')
                .any(|p| p.is_empty() || p == "." || p == "..")
        {
            continue;
        }
        let bytes = e.get("size").and_then(|s| s.as_u64()).unwrap_or(0);
        out.push(RemoteEntry {
            rel: rel.to_string(),
            bytes,
        });
    }
    if out.is_empty() {
        if v.get("truncated").and_then(|t| t.as_bool()) == Some(true) {
            bail!("the listing of {REPO}@{BRANCH} was truncated before reaching {FOLDER}/");
        }
        bail!("{REPO}@{BRANCH} has no files under {FOLDER}/");
    }
    Ok(out)
}

/// What the folder is known to hold, for when the listing cannot be
/// fetched: GitHub's unauthenticated API allows 60 calls an hour per
/// address, and a clinic's shared address may have spent them on something
/// else. TCIA 4D-Lung P102 (docs/example-data.md): ten 4DCBCT phases of 50
/// slices, then ten 4DFBCT phases of 133 slices, then one RTSTRUCT per
/// 4DFBCT phase - 1840 files, in the path order the API would give them.
/// The sizes are unknown here (`0`), so a file counts as present when it
/// exists and is not empty, and progress is counted in files rather than
/// bytes. A file this list names that the repository no longer has fails
/// the download with the server's answer, which is the right outcome: the
/// list is a stand-in for the listing, not a second source of truth.
pub fn builtin_listing() -> Vec<RemoteEntry> {
    // The ten gating phases, as they are written in every folder name.
    const PHASES: [&str; 10] = [
        "0.0", "10.0", "20.0", "30.0", "40.0", "50.0", "60.0", "70.0", "80.0", "90.0",
    ];
    let mut out = Vec::with_capacity(1840);
    let mut file = |rel: String| out.push(RemoteEntry { rel, bytes: 0 });
    // "4DCBCT" sorts before "4DFBCT+RTS", which is the order the API lists.
    for (i, p) in PHASES.iter().enumerate() {
        for slice in 0..50 {
            file(format!(
                "{PATIENT}/4DCBCT/{n}_CT_4DCBCT__Gated__{p}_A/CT_{slice:04}.dcm",
                n = 500 + i
            ));
        }
    }
    for p in PHASES {
        for slice in 0..133 {
            file(format!(
                "{PATIENT}/4DFBCT+RTS/1_CT_4DFBCT__Gated__{p}_A/CT_{slice:04}.dcm"
            ));
        }
    }
    for p in PHASES {
        file(format!("{PATIENT}/4DFBCT+RTS/RS_RTS__{p}_A.dcm"));
    }
    out
}

/// Ask GitHub for the folder's contents.
pub fn list(sink: &dyn ProgressSink) -> Result<Vec<RemoteEntry>> {
    sink.report(0.0, &format!("Listing {FOLDER}/ on GitHub"));
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(60))
        .build();
    let url = tree_url();
    let body = match agent
        .get(&url)
        .set("Accept", "application/vnd.github+json")
        .set(
            "User-Agent",
            concat!("rust-dicom-station/", env!("CARGO_PKG_VERSION")),
        )
        .call()
    {
        Ok(resp) => resp.into_string().context("read the listing")?,
        // A 403 with a JSON body is the rate limit (60 unauthenticated
        // calls an hour per address); its message is worth passing on.
        Err(ureq::Error::Status(code, resp)) => {
            let body = resp.into_string().unwrap_or_default();
            match parse_tree(&body) {
                Err(e) if body.contains("\"message\"") => {
                    return Err(e.context(format!("GitHub answered {code}")))
                }
                _ => bail!("GitHub answered {code} for {url}"),
            }
        }
        Err(e) => return Err(anyhow::Error::new(e).context(format!("list {url}"))),
    };
    parse_tree(&body)
}

/// Fetch every file of the folder that is not already in `dir`, reporting
/// progress as a fraction of the bytes to fetch. Cancellation through the
/// sink stops between files and inside a file alike (the downloader checks
/// it per chunk) and leaves no partial file behind.
pub fn download(dir: &Path, sink: &dyn ProgressSink) -> Result<Summary> {
    let entries = match list(sink) {
        Ok(entries) => entries,
        // The one failure with a way round it. Anything else (no network, a
        // proxy that refuses the host) would fail the file downloads just
        // the same and is reported as it is.
        Err(e) if is_rate_limit(&e) => {
            sink.report(
                0.0,
                "GitHub's listing is rate-limited - using the built-in file list",
            );
            builtin_listing()
        }
        Err(e) => return Err(e),
    };
    download_entries(&entries, dir, sink)
}

/// GitHub's "API rate limit exceeded" answer, at whatever depth of the
/// error chain it sits (`403` with the message, or `429`).
pub fn is_rate_limit(e: &anyhow::Error) -> bool {
    let text = format!("{e:#}").to_ascii_lowercase();
    text.contains("rate limit") || text.contains("answered 429")
}

/// [`download`] with a listing already in hand (what the tests use).
pub fn download_entries(
    entries: &[RemoteEntry],
    dir: &Path,
    sink: &dyn ProgressSink,
) -> Result<Summary> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let todo: Vec<&RemoteEntry> = entries.iter().filter(|e| !e.is_present(dir)).collect();
    // Progress in bytes when the listing gave sizes, in files when it did
    // not (the built-in list).
    let by_bytes = todo.iter().any(|e| e.bytes > 0);
    let total: u64 = if by_bytes {
        todo.iter().map(|e| e.bytes).sum::<u64>().max(1)
    } else {
        todo.len().max(1) as u64
    };
    let mut summary = Summary {
        kept: entries.len() - todo.len(),
        dir: dir.to_path_buf(),
        ..Summary::default()
    };
    let mut done: u64 = 0;
    for (i, e) in todo.iter().enumerate() {
        if sink.cancelled() {
            bail!(CANCELLED);
        }
        let dest = e.path_in(dir);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let window = Window {
            inner: sink,
            base: done as f32 / total as f32,
            span: if by_bytes { e.bytes } else { 1 } as f32 / total as f32,
            label: if by_bytes {
                format!(
                    "{FOLDER}: file {} of {}, {} / {} MB",
                    i + 1,
                    todo.len(),
                    done / 1_000_000,
                    total / 1_000_000
                )
            } else {
                format!("{FOLDER}: file {} of {}", i + 1, todo.len())
            },
        };
        let tmp = dest.with_extension("part");
        download_to_file(&raw_url(&e.rel), &tmp, e.bytes, &e.rel, &window)
            .with_context(|| format!("download {}", e.rel))?;
        std::fs::rename(&tmp, &dest).with_context(|| format!("rename to {}", dest.display()))?;
        let got = dest.metadata().map(|m| m.len()).unwrap_or(e.bytes);
        done += if by_bytes { e.bytes } else { 1 };
        summary.downloaded += 1;
        summary.bytes += got;
    }
    summary.datasets = datasets_in(dir);
    sink.report(1.0, &format!("{FOLDER}: {} file(s)", entries.len()));
    Ok(summary)
}

/// The study folders to offer under `dir`, sorted by name.
///
/// The data nests its two studies under a collection and a patient
/// (`data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS`), and what the window wants to
/// offer is the studies, not the one collection folder above them. So the
/// walk goes down while a level holds exactly one folder and that folder
/// has no files of its own, and stops at the first level that branches -
/// here `4DCBCT` and `4DFBCT+RTS`. A flat folder of datasets is unchanged
/// by the rule, since its first level already branches. The depth is capped
/// so that no symlink loop can turn this into a long walk.
pub fn datasets_in(dir: &Path) -> Vec<PathBuf> {
    let mut level = subfolders(dir);
    for _ in 0..8 {
        let [only] = &level[..] else { break };
        if holds_files(only) {
            break;
        }
        let next = subfolders(only);
        if next.is_empty() {
            break;
        }
        level = next;
    }
    level
}

/// The folders directly inside `dir`, sorted by name.
fn subfolders(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// True when `dir` holds at least one file of its own (so it is a folder of
/// data, not only a step on the way down to one).
fn holds_files(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok()).any(|e| e.path().is_file()))
        .unwrap_or(false)
}

/// Maps one file's `0..=1` onto its share of the whole, and replaces the
/// downloader's per-file message with one that counts files.
struct Window<'a> {
    inner: &'a dyn ProgressSink,
    base: f32,
    span: f32,
    label: String,
}

impl ProgressSink for Window<'_> {
    fn report(&self, frac: f32, _msg: &str) {
        self.inner
            .report(self.base + self.span * frac.clamp(0.0, 1.0), &self.label);
    }
    fn cancelled(&self) -> bool {
        self.inner.cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::Quiet;

    const SAMPLE: &str = r#"{
      "sha": "abc", "url": "u",
      "tree": [
        {"path": "Cargo.toml", "mode": "100644", "type": "blob", "size": 4729, "sha": "1"},
        {"path": "data-test", "mode": "040000", "type": "tree", "sha": "2"},
        {"path": "data-test/TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A", "mode": "040000", "type": "tree", "sha": "3"},
        {"path": "data-test/TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm", "mode": "100644", "type": "blob", "size": 528100, "sha": "4"},
        {"path": "data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS/1_CT_4DFBCT__Gated__0.0_A/CT_0000.dcm", "mode": "100644", "type": "blob", "size": 526336, "sha": "5"},
        {"path": "data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS/RS_RTS__0.0_A.dcm", "mode": "100644", "type": "blob", "size": 1118624, "sha": "6"},
        {"path": "data-test-other/x.dcm", "mode": "100644", "type": "blob", "size": 1, "sha": "7"},
        {"path": "docs/data-test/y.dcm", "mode": "100644", "type": "blob", "size": 1, "sha": "8"}
      ],
      "truncated": false
    }"#;

    #[test]
    fn the_listing_keeps_only_blobs_under_the_folder() {
        let e = parse_tree(SAMPLE).unwrap();
        assert_eq!(
            e,
            vec![
                RemoteEntry {
                    rel: "TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm".into(),
                    bytes: 528100
                },
                RemoteEntry {
                    rel: "TCIA_4D-LUNG/P102/4DFBCT+RTS/1_CT_4DFBCT__Gated__0.0_A/CT_0000.dcm"
                        .into(),
                    bytes: 526336
                },
                RemoteEntry {
                    rel: "TCIA_4D-LUNG/P102/4DFBCT+RTS/RS_RTS__0.0_A.dcm".into(),
                    bytes: 1118624
                },
            ]
        );
    }

    #[test]
    fn an_empty_or_foreign_listing_is_an_error_with_a_reason() {
        let e = parse_tree(r#"{"tree": [], "truncated": false}"#).unwrap_err();
        assert!(e.to_string().contains("no files under data-test/"), "{e}");
        let e = parse_tree(r#"{"tree": [], "truncated": true}"#).unwrap_err();
        assert!(e.to_string().contains("truncated"), "{e}");
        let e = parse_tree(r#"{"message": "API rate limit exceeded for 1.2.3.4."}"#).unwrap_err();
        assert!(e.to_string().contains("rate limit"), "{e}");
        assert!(parse_tree("<html>").is_err());
    }

    #[test]
    fn urls_name_the_repository_branch_and_folder() {
        assert_eq!(
            tree_url(),
            "https://api.github.com/repos/alexprotom/rust-dicom-station/git/trees/main?recursive=1"
        );
        assert_eq!(
            raw_url("TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm"),
            "https://raw.githubusercontent.com/alexprotom/rust-dicom-station/main/data-test/TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm"
        );
        assert_eq!(
            browse_url(),
            "https://github.com/alexprotom/rust-dicom-station/tree/main/data-test"
        );
    }

    #[test]
    fn a_file_of_the_listed_size_is_kept_and_not_fetched() {
        let dir = std::env::temp_dir().join(format!("rds-testdata-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let entry = RemoteEntry {
            rel: "TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm".into(),
            bytes: 3,
        };
        assert!(!entry.is_present(&dir));
        let p = entry.path_in(&dir);
        assert_eq!(
            p,
            dir.join("TCIA_4D-LUNG")
                .join("P102")
                .join("4DCBCT")
                .join("500_CT_4DCBCT__Gated__0.0_A")
                .join("CT_0000.dcm"),
            "the relative path is split on '/' into native components"
        );
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"abc").unwrap();
        assert!(entry.is_present(&dir));
        std::fs::write(&p, b"ab").unwrap();
        assert!(!entry.is_present(&dir), "a short file is fetched again");

        // Everything present: no network is touched, the summary counts
        // the kept files and walks down to the study folder.
        std::fs::write(&p, b"abc").unwrap();
        let s = download_entries(std::slice::from_ref(&entry), &dir, &Quiet).unwrap();
        assert_eq!(s.downloaded, 0);
        assert_eq!(s.kept, 1);
        // One file: the chain of single folders runs all the way down to
        // the series, and that is what is offered - correct for a tree with
        // nothing else in it.
        assert_eq!(
            s.datasets,
            vec![dir
                .join("TCIA_4D-LUNG")
                .join("P102")
                .join("4DCBCT")
                .join("500_CT_4DCBCT__Gated__0.0_A")]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_walk_down_stops_where_the_tree_branches() {
        let dir = std::env::temp_dir().join(format!("rds-testdata-walk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let patient = dir.join("TCIA_4D-LUNG").join("P102");
        for study in ["4DCBCT", "4DFBCT+RTS"] {
            let series = patient.join(study).join("s1");
            std::fs::create_dir_all(&series).unwrap();
            std::fs::write(series.join("CT_0000.dcm"), b"x").unwrap();
        }
        // Two collection levels are walked through; the two studies are what
        // is offered, not the one collection folder and not the series.
        assert_eq!(
            datasets_in(&dir),
            vec![patient.join("4DCBCT"), patient.join("4DFBCT+RTS")]
        );

        // A folder that holds files of its own is where the walk stops, even
        // when it has exactly one subfolder.
        std::fs::write(patient.join("README.txt"), b"x").unwrap();
        assert_eq!(
            datasets_in(&dir.join("TCIA_4D-LUNG")),
            vec![patient.clone()]
        );

        // A flat folder of datasets is untouched by the rule.
        let flat = dir.join("flat");
        for d in ["a", "b"] {
            std::fs::create_dir_all(flat.join(d)).unwrap();
            std::fs::write(flat.join(d).join("CT_0000.dcm"), b"x").unwrap();
        }
        assert_eq!(datasets_in(&flat), vec![flat.join("a"), flat.join("b")]);

        // Nothing there at all: nothing to offer, and no panic.
        assert!(datasets_in(&dir.join("nope")).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_built_in_list_is_the_folder_as_documented() {
        let b = builtin_listing();
        assert_eq!(b.len(), 10 * 50 + 10 * 133 + 10);
        assert_eq!(b.len(), 1840);
        assert_eq!(
            b[0].rel,
            "TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm"
        );
        assert_eq!(
            b[49].rel,
            "TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0049.dcm"
        );
        assert_eq!(
            b[50].rel,
            "TCIA_4D-LUNG/P102/4DCBCT/501_CT_4DCBCT__Gated__10.0_A/CT_0000.dcm"
        );
        assert_eq!(
            b[500].rel,
            "TCIA_4D-LUNG/P102/4DFBCT+RTS/1_CT_4DFBCT__Gated__0.0_A/CT_0000.dcm"
        );
        assert_eq!(
            b[632].rel,
            "TCIA_4D-LUNG/P102/4DFBCT+RTS/1_CT_4DFBCT__Gated__0.0_A/CT_0132.dcm"
        );
        assert_eq!(
            b[1830].rel,
            "TCIA_4D-LUNG/P102/4DFBCT+RTS/RS_RTS__0.0_A.dcm"
        );
        assert_eq!(
            b[1839].rel,
            "TCIA_4D-LUNG/P102/4DFBCT+RTS/RS_RTS__90.0_A.dcm"
        );
        let mut sorted: Vec<&str> = b.iter().map(|e| e.rel.as_str()).collect();
        let listed = sorted.clone();
        sorted.sort_unstable();
        assert_eq!(listed, sorted, "the list is in the API's path order");
        assert!(b.iter().all(|e| e.bytes == 0));

        // Unknown size: present means non-empty.
        let dir = std::env::temp_dir().join(format!("rds-testdata-builtin-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = b[0].path_in(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"").unwrap();
        assert!(!b[0].is_present(&dir));
        std::fs::write(&p, b"x").unwrap();
        assert!(b[0].is_present(&dir));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn only_the_rate_limit_answer_falls_back() {
        let e = anyhow::anyhow!("GitHub: API rate limit exceeded for 1.2.3.4.")
            .context("GitHub answered 403");
        assert!(is_rate_limit(&e));
        assert!(is_rate_limit(&anyhow::anyhow!("GitHub answered 429 for x")));
        assert!(!is_rate_limit(&anyhow::anyhow!(
            "GitHub answered 404 for x"
        )));
        assert!(!is_rate_limit(&anyhow::anyhow!(
            "list x: connection refused"
        )));
    }

    #[test]
    fn cancellation_is_honoured_before_the_first_request() {
        struct Cancelled;
        impl ProgressSink for Cancelled {
            fn cancelled(&self) -> bool {
                true
            }
        }
        let dir = std::env::temp_dir().join(format!("rds-testdata-cancel-{}", std::process::id()));
        let entry = RemoteEntry {
            rel: "TCIA_4D-LUNG/P102/4DCBCT/500_CT_4DCBCT__Gated__0.0_A/CT_0000.dcm".into(),
            bytes: 1,
        };
        let e = download_entries(&[entry], &dir, &Cancelled).unwrap_err();
        assert!(crate::progress::is_cancellation(&e));
        assert!(!dir.join("TCIA_4D-LUNG").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_window_maps_a_file_onto_its_share_of_the_whole() {
        use std::sync::atomic::{AtomicU32, Ordering};
        struct Last(AtomicU32);
        impl Last {
            fn get(&self) -> f32 {
                f32::from_bits(self.0.load(Ordering::Relaxed))
            }
        }
        impl ProgressSink for Last {
            fn report(&self, frac: f32, _msg: &str) {
                self.0.store(frac.to_bits(), Ordering::Relaxed);
            }
        }
        let last = Last(AtomicU32::new((-1f32).to_bits()));
        let w = Window {
            inner: &last,
            base: 0.25,
            span: 0.5,
            label: "x".into(),
        };
        w.report(0.0, "");
        assert_eq!(last.get(), 0.25);
        w.report(1.0, "");
        assert_eq!(last.get(), 0.75);
        w.report(2.0, "");
        assert_eq!(last.get(), 0.75, "clamped");
    }
}
