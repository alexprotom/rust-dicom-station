//! Manifests and journals: how every curation write is planned, applied and
//! taken back.
//!
//! A write tool first runs dry: it works out what it would do as a list of
//! [`Op`]s, saves that as a [`Manifest`] in the records folder and returns
//! its id. Applying takes the id back, re-reads the manifest from the
//! records folder (never from the client), and does the operations one by
//! one, appending a line to the manifest's journal after each - so a run
//! cut short is resumed by applying again, and the journal is the exact
//! record of what changed. [`undo`] walks a journal backwards and puts
//! everything back: a move is moved back, a rewritten file is restored from
//! the copy taken before it was rewritten, after checking that nobody has
//! changed it since (its SHA-256 is in the journal).
//!
//! Files: `records/manifests/<id>.json`, `records/journals/<id>.jsonl`,
//! `records/journals/<id>.backup/<n>` (the files as they were before a
//! rewrite), `records/journals/<id>.undo.jsonl`.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::raw::{self, encode_text};
use crate::progress::{Progress, CANCELLED};

/// Longest path a write may produce: Windows' classic limit, which a lot of
/// DICOM software still has.
pub const MAX_PATH: usize = 259;
/// A path this long is written, with a warning.
pub const WARN_PATH: usize = 250;

/// One planned change.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Move a file (the folders it needs are made).
    Move { from: PathBuf, to: PathBuf },
    /// Set top-level text elements of a file, verified ([`raw::rewrite_file`]):
    /// `(tag as "GGGGEEEE", new value)`.
    SetTags {
        path: PathBuf,
        tags: Vec<(String, String)>,
    },
}

impl Op {
    /// The paths the op writes.
    pub fn writes(&self) -> Vec<&Path> {
        match self {
            Op::Move { from, to } => vec![from, to],
            Op::SetTags { path, .. } => vec![path],
        }
    }
}

/// A dry run, saved.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub id: String,
    /// The tool that made it.
    pub tool: String,
    /// Why, as the client said.
    pub reason: String,
    pub created: String,
    pub ops: Vec<Op>,
    /// What the dry run noticed (long paths, files left alone).
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// One line of a journal.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    /// The op's index in the manifest.
    pub i: usize,
    pub op: Op,
    /// SHA-256 of the file before and after a rewrite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub after: Option<String>,
    /// Where the file was kept before a rewrite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backup: Option<PathBuf>,
    /// The op changed nothing (a file already as asked).
    #[serde(default)]
    pub unchanged: bool,
}

/// SHA-256 of a file, as hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(hex(&Sha256::digest(&bytes)))
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A new manifest id: when, and which tool.
pub fn new_id(tool: &str) -> String {
    let (d, t) = crate::dicom_export::today();
    let t: String = t.chars().take(6).collect();
    // Two dry runs in one second get different ids.
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_millis())
        .unwrap_or(0);
    format!("{d}-{t}-{n:03}-{tool}")
}

/// Whether `id` is one [`new_id`] could have made: no path in it.
fn plain_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn manifest_path(records: &Path, id: &str) -> PathBuf {
    records.join("manifests").join(format!("{id}.json"))
}

fn journal_path(records: &Path, id: &str) -> PathBuf {
    records.join("journals").join(format!("{id}.jsonl"))
}

/// Check what a manifest would write: every path under a folder the
/// caller allows, and short enough. Returns the warnings.
pub fn check_paths(ops: &[Op], allowed: &dyn Fn(&Path) -> bool) -> Result<Vec<String>> {
    let mut warnings = Vec::new();
    for op in ops {
        for p in op.writes() {
            if !allowed(p) {
                bail!("{} is outside the folders this tool may write", p.display());
            }
            let n = p.as_os_str().len();
            if n > MAX_PATH {
                bail!(
                    "{} would be {n} characters long, over {MAX_PATH}",
                    p.display()
                );
            }
            if n > WARN_PATH {
                warnings.push(format!("{} is {n} characters long", p.display()));
            }
        }
    }
    Ok(warnings)
}

/// Save a dry run in the records folder.
pub fn save(records: &Path, m: &Manifest) -> Result<PathBuf> {
    let path = manifest_path(records, &m.id);
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, serde_json::to_vec_pretty(m)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Read a saved dry run back by its id.
pub fn load(records: &Path, id: &str) -> Result<Manifest> {
    if !plain_id(id) {
        bail!("'{id}' is not a manifest id");
    }
    let path = manifest_path(records, id);
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("no manifest {id} in the records folder"))?;
    Ok(serde_json::from_str(&text)?)
}

/// The journal lines of manifest `id` so far.
pub fn entries(records: &Path, id: &str) -> Result<Vec<Entry>> {
    let path = journal_path(records, id);
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Ok(Vec::new());
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("a journal line"))
        .collect()
}

fn append(path: &Path, e: &Entry) -> Result<()> {
    std::fs::create_dir_all(path.parent().unwrap())?;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{}", serde_json::to_string(e)?)?;
    f.sync_data()?;
    Ok(())
}

/// Parse `GGGGEEEE` (or `GGGG,EEEE`).
pub fn parse_tag(s: &str) -> Result<dicom_core::Tag> {
    let h: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    if h.len() != 8 {
        bail!("'{s}' is not a tag");
    }
    let g = u16::from_str_radix(&h[..4], 16)?;
    let e = u16::from_str_radix(&h[4..], 16)?;
    Ok(dicom_core::Tag(g, e))
}

/// Move a file: a rename on the same volume, else a copy that is read back
/// and compared before the original goes.
fn move_file(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        bail!("{} exists already", to.display());
    }
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to).with_context(|| format!("copy to {}", to.display()))?;
    if std::fs::read(from)? != std::fs::read(to)? {
        let _ = std::fs::remove_file(to);
        bail!("the copy of {} does not read back the same", from.display());
    }
    std::fs::remove_file(from)?;
    Ok(())
}

/// What an apply did.
#[derive(Debug, Default, Serialize)]
pub struct Applied {
    pub done: usize,
    /// Already in the journal from an earlier, interrupted run.
    pub resumed: usize,
    pub unchanged: usize,
    pub failed: Vec<String>,
}

/// Apply manifest `id`, journaling each op; ops already journaled are
/// skipped. `allowed` is checked again for every path. Stops at the first
/// failure (the journal says how far it got).
pub fn apply(
    records: &Path,
    id: &str,
    allowed: &dyn Fn(&Path) -> bool,
    p: &Progress,
) -> Result<Applied> {
    let m = load(records, id)?;
    check_paths(&m.ops, allowed)?;
    let jpath = journal_path(records, id);
    let done: BTreeSet<usize> = entries(records, id)?.iter().map(|e| e.i).collect();
    let mut out = Applied {
        resumed: done.len(),
        ..Default::default()
    };
    let backups = records.join("journals").join(format!("{id}.backup"));
    for (i, op) in m.ops.iter().enumerate() {
        if done.contains(&i) {
            continue;
        }
        if p.cancelled() {
            bail!(CANCELLED);
        }
        if i.is_multiple_of(32) {
            p.set(format!("Applying {}/{}", i, m.ops.len()));
        }
        let mut e = Entry {
            i,
            op: op.clone(),
            before: None,
            after: None,
            backup: None,
            unchanged: false,
        };
        let r: Result<()> = (|| {
            match op {
                Op::Move { from, to } => move_file(from, to)?,
                Op::SetTags { path, tags } => {
                    let want: Vec<(dicom_core::Tag, String)> = tags
                        .iter()
                        .map(|(t, v)| Ok((parse_tag(t)?, v.clone())))
                        .collect::<Result<_>>()?;
                    let before = sha256_file(path)?;
                    std::fs::create_dir_all(&backups)?;
                    let backup = backups.join(format!("{i}"));
                    std::fs::copy(path, &backup)?;
                    let mut edit = |at: &raw::At| -> Option<Vec<u8>> {
                        if !at.path.is_empty() {
                            return None;
                        }
                        let (_, v) = want.iter().find(|(t, _)| *t == at.tag)?;
                        let new = encode_text(at.vr, v);
                        (new != at.value).then_some(new)
                    };
                    let changed = raw::rewrite_file(path, &mut edit)?;
                    if changed == 0 {
                        let _ = std::fs::remove_file(&backup);
                        e.unchanged = true;
                    } else {
                        e.before = Some(before);
                        e.after = Some(sha256_file(path)?);
                        e.backup = Some(backup);
                    }
                }
            }
            Ok(())
        })();
        match r {
            Ok(()) => {
                if e.unchanged {
                    out.unchanged += 1;
                } else {
                    out.done += 1;
                }
                append(&jpath, &e)?;
            }
            Err(err) => {
                out.failed.push(format!("op {i}: {err:#}"));
                break;
            }
        }
    }
    Ok(out)
}

/// What an undo did.
#[derive(Debug, Default, Serialize)]
pub struct Undone {
    pub undone: usize,
    /// Entries left alone, and why: a file changed since, a move whose
    /// source is taken again.
    pub skipped: Vec<String>,
}

/// Take back everything the journal of manifest `id` did, newest first.
/// Each entry is checked before it is undone; one that cannot be is skipped
/// and said, the rest go on. The undo has a journal of its own.
pub fn undo(
    records: &Path,
    id: &str,
    allowed: &dyn Fn(&Path) -> bool,
    p: &Progress,
) -> Result<Undone> {
    if !plain_id(id) {
        bail!("'{id}' is not a manifest id");
    }
    let undo_path = records.join("journals").join(format!("{id}.undo.jsonl"));
    let already: BTreeSet<usize> = std::fs::read_to_string(&undo_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Entry>(l).ok())
        .map(|e| e.i)
        .collect();
    let mut out = Undone::default();
    let list = entries(records, id)?;
    if list.is_empty() {
        bail!("manifest {id} has no journal: nothing was applied");
    }
    for e in list.iter().rev() {
        if p.cancelled() {
            bail!(CANCELLED);
        }
        if already.contains(&e.i) || e.unchanged {
            continue;
        }
        if e.op.writes().iter().any(|w| !allowed(w)) {
            out.skipped
                .push(format!("op {}: outside the writable folders", e.i));
            continue;
        }
        let r: Result<()> = match &e.op {
            Op::Move { from, to } => {
                if from.exists() {
                    Err(anyhow::anyhow!("{} is there again", from.display()))
                } else {
                    move_file(to, from)
                }
            }
            Op::SetTags { path, .. } => (|| {
                let (Some(after), Some(backup)) = (&e.after, &e.backup) else {
                    bail!("no copy was kept");
                };
                if &sha256_file(path)? != after {
                    bail!("{} has changed since", path.display());
                }
                let tmp = path.with_extension("undo.tmp");
                std::fs::copy(backup, &tmp)?;
                if Some(sha256_file(&tmp)?) != e.before {
                    let _ = std::fs::remove_file(&tmp);
                    bail!("the kept copy is not the file as it was");
                }
                std::fs::rename(&tmp, path)?;
                Ok(())
            })(),
        };
        match r {
            Ok(()) => {
                out.undone += 1;
                append(&undo_path, e)?;
            }
            Err(err) => out.skipped.push(format!("op {}: {err:#}", e.i)),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::raw::tests::{im, part10};
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rds_journal_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A tree's files and their bytes, for a byte-identical comparison.
    fn snapshot(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        let mut v: Vec<(PathBuf, Vec<u8>)> = walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| {
                (
                    e.path().strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        v.sort();
        v
    }

    #[test]
    fn plan_apply_undo_restores_the_byte_identical_tree() {
        let d = dir("roundtrip");
        let (tree, records) = (d.join("pacs"), d.join("records"));
        std::fs::create_dir_all(tree.join("P01-001/s1")).unwrap();
        let mut data = Vec::new();
        data.extend(im(dicom_core::Tag(0x0010, 0x0010), b"anon^x  "));
        data.extend(im(dicom_core::Tag(0x0010, 0x0020), b"1234"));
        let f1 = tree.join("P01-001/s1/a.dcm");
        let f2 = tree.join("P01-001/s1/b.dcm");
        std::fs::write(&f1, part10(raw::TS_IMPLICIT, &data)).unwrap();
        std::fs::write(&f2, part10(raw::TS_IMPLICIT, &data)).unwrap();
        let before = snapshot(&tree);
        let m = Manifest {
            id: new_id("test"),
            tool: "test".into(),
            reason: "round trip".into(),
            created: String::new(),
            ops: vec![
                Op::SetTags {
                    path: f1.clone(),
                    tags: vec![
                        ("00100010".into(), "P01-001".into()),
                        ("00100020".into(), "P01-001".into()),
                    ],
                },
                Op::Move {
                    from: f2.clone(),
                    to: tree.join("P01-001/q/b.dcm"),
                },
            ],
            warnings: Vec::new(),
        };
        let inside = |p: &Path| p.starts_with(&tree);
        save(&records, &m).unwrap();
        let a = apply(&records, &m.id, &inside, &Progress::default()).unwrap();
        assert_eq!((a.done, a.failed.len()), (2, 0));
        let f = raw::RawFile::read(&f1).unwrap();
        assert_eq!(
            f.text(dicom_core::Tag(0x0010, 0x0020)).as_deref(),
            Some("P01-001")
        );
        assert!(!f2.exists());
        // Applying again resumes: nothing is done twice.
        let again = apply(&records, &m.id, &inside, &Progress::default()).unwrap();
        assert_eq!((again.done, again.resumed), (0, 2));
        let u = undo(&records, &m.id, &inside, &Progress::default()).unwrap();
        assert_eq!(u.undone, 2, "{:?}", u.skipped);
        // The empty folder the move made is left; the files are as they were.
        assert_eq!(snapshot(&tree), before);
    }

    #[test]
    fn a_path_outside_or_too_long_is_refused_before_anything_moves() {
        let ok = |p: &Path| p.starts_with("/pacs");
        let out = [Op::Move {
            from: "/pacs/a".into(),
            to: "/elsewhere/a".into(),
        }];
        assert!(check_paths(&out, &ok).is_err());
        let long = format!("/pacs/{}", "x".repeat(300));
        let too_long = [Op::Move {
            from: "/pacs/a".into(),
            to: long.into(),
        }];
        assert!(check_paths(&too_long, &ok).is_err());
        assert!(load(Path::new("/r"), "../etc").is_err());
    }
}
