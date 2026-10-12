//! The PACS catalog: one file that says what every storage folder holds.
//!
//! Listing a PACS must stay quick however large it grows, and reading the
//! headers of ten thousand files is not quick. So every file is read once
//! and remembered here - its SOP Instance UID, and the patient, study and
//! series it belongs to - together with what the person decided about the
//! PACS: the names of the cohorts, which patient is in which, and the
//! names given to patients, studies and series. The next scan walks the
//! folders again (cheap) and reads only the files it has not seen, or that
//! changed (size or time).
//!
//! The file is `rds-pacs-catalog.json` in the main storage folder (the
//! first one). JSON because it is a format every tool reads and a person
//! can open: one line per patient, study and series, the files of a series
//! on its line. The DICOM files are the truth and the catalog a cache of
//! them, except for the person's decisions, which live only here: delete
//! the file and the next scan rebuilds the rest.
//!
//! Two programs write it (the viewer and a PACS server serving the same
//! folders), so a write takes a lock file beside it, reads the catalog
//! afresh, changes it and puts it back in one rename: the slow part of a
//! scan (reading headers) happens outside the lock, the merge inside.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use dicom_dictionary_std::tags;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::loader::str_of;
use crate::progress::{Progress, CANCELLED};

/// The catalog's file name, in the main storage folder.
pub const CATALOG_FILE: &str = "rds-pacs-catalog.json";
/// Held while the catalog is rewritten.
const LOCK_FILE: &str = ".rds-pacs-catalog.lock";
const FORMAT: &str = "rds-pacs-catalog";
const VERSION: u32 = 1;
/// The id of the cohort that holds the patients outside any cohort folder.
pub const UNSORTED: &str = "unsorted";

/// One DICOM file: `[storage folder, path below it, bytes, modified, SOP
/// Instance UID]`. The folder is an index into [`Catalog::storage`], the
/// path `/`-separated, the time in seconds since 1970.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileRec(pub u32, pub String, pub u64, pub i64, pub String);

/// A file that is not DICOM, remembered so that it is not read again:
/// `[storage folder, path, bytes, modified]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OtherRec(pub u32, pub String, pub u64, pub i64);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PatientRec {
    pub id: String,
    pub name: String,
    /// The name given in the PACS window; empty: the DICOM name.
    pub label: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StudyRec {
    /// The patient's key (see [`patient_key`]).
    pub patient: String,
    pub date: String,
    pub time: String,
    pub description: String,
    pub label: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SeriesRec {
    /// The Study Instance UID.
    pub study: String,
    pub modality: String,
    pub number: String,
    pub description: String,
    pub label: String,
    pub files: Vec<FileRec>,
}

/// Where a cohort's folder is: the storage folder, and the name of the
/// folder directly in it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CohortFolder {
    pub storage: String,
    pub name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CohortRec {
    pub id: String,
    /// What the window calls it; at first its folder's name.
    pub name: String,
    /// Its folder; none for a cohort made in the window (and for
    /// *Unsorted*).
    pub folder: Option<CohortFolder>,
}

/// The whole catalog. See the module doc.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Catalog {
    pub format: String,
    pub version: u32,
    pub updated: String,
    /// The storage folders as of the last scan; files name them by index.
    pub storage: Vec<String>,
    pub cohorts: Vec<CohortRec>,
    /// The cohorts a patient was put in by hand (moved or copied), by
    /// patient key. A patient not named here is in the cohorts its files
    /// lie in.
    pub membership: BTreeMap<String, Vec<String>>,
    pub patients: BTreeMap<String, PatientRec>,
    pub studies: BTreeMap<String, StudyRec>,
    pub series: BTreeMap<String, SeriesRec>,
    pub other: Vec<OtherRec>,
}

/// What a scan learns from one file's header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Head {
    pub sop: String,
    pub study: String,
    pub series: String,
    pub patient_id: String,
    pub patient_name: String,
    pub modality: String,
    pub study_date: String,
    pub study_time: String,
    pub study_description: String,
    pub series_number: String,
    pub series_description: String,
}

impl Head {
    pub fn patient_key(&self) -> String {
        patient_key(&self.patient_id, &self.patient_name)
    }
}

/// The key a patient is known by in the catalog and over the network:
/// the Patient ID, or the name when there is none.
pub fn patient_key(id: &str, name: &str) -> String {
    let id = id.trim();
    let name = name.trim();
    if !id.is_empty() {
        id.to_string()
    } else if !name.is_empty() {
        name.to_string()
    } else {
        "unknown".into()
    }
}

/// Read what the catalog needs from a file's header; `None` for a file
/// that is not DICOM or names no instance.
pub fn read_head(path: &Path) -> Option<Head> {
    let obj = crate::dicomfile::open_scan(path).ok()?;
    let s = |t| str_of(&obj, t).unwrap_or_default();
    let sop = s(tags::SOP_INSTANCE_UID);
    let study = s(tags::STUDY_INSTANCE_UID);
    if sop.is_empty() || study.is_empty() {
        return None;
    }
    let modality = s(tags::MODALITY);
    let mut series = s(tags::SERIES_INSTANCE_UID);
    if series.is_empty() {
        // Not a valid UID, but a key that keeps the study's objects of one
        // kind together; it never travels as a UID.
        series = format!("{study}~{modality}");
    }
    Some(Head {
        sop,
        study,
        series,
        patient_id: s(tags::PATIENT_ID),
        patient_name: s(tags::PATIENT_NAME),
        modality,
        study_date: s(tags::STUDY_DATE),
        study_time: s(tags::STUDY_TIME),
        study_description: s(tags::STUDY_DESCRIPTION),
        series_number: s(tags::SERIES_NUMBER),
        series_description: s(tags::SERIES_DESCRIPTION),
    })
}

/// A path below a storage folder, `/`-separated.
pub fn rel_of(root: &Path, path: &Path) -> Option<String> {
    let rel = path.strip_prefix(root).ok()?;
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// The absolute path of a catalog path.
pub fn abs_of(root: &Path, rel: &str) -> PathBuf {
    let mut p = root.to_path_buf();
    for part in rel.split('/') {
        p.push(part);
    }
    p
}

fn mtime_of(m: &std::fs::Metadata) -> i64 {
    m.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Is `a` the folder `b` names? Compared as text, separators and (on
/// Windows) case aside, since a folder list is typed and pasted by hand.
pub fn same_folder(a: &Path, b: &str) -> bool {
    let norm = |s: &str| {
        let t = s.trim().trim_end_matches(['/', '\\']).replace('\\', "/");
        if cfg!(windows) {
            t.to_lowercase()
        } else {
            t
        }
    };
    norm(&a.display().to_string()) == norm(b)
}

// ---- reading and writing ---------------------------------------------------

/// Catalogs read before, by file, with the time and size they had: a
/// server answering one request after another reads the file once.
type Cached = HashMap<PathBuf, (SystemTime, u64, Arc<Catalog>)>;
static CACHE: Mutex<Option<Cached>> = Mutex::new(None);

pub fn path_in(main: &Path) -> PathBuf {
    main.join(CATALOG_FILE)
}

/// The catalog of the PACS whose main folder is `main`; empty when there
/// is none yet (or it cannot be read: it is rebuilt by the next scan).
pub fn load(main: &Path) -> Arc<Catalog> {
    let path = path_in(main);
    let Ok(meta) = std::fs::metadata(&path) else {
        return Arc::new(Catalog::default());
    };
    let stamp = (meta.modified().unwrap_or(UNIX_EPOCH), meta.len());
    if let Ok(guard) = CACHE.lock() {
        if let Some((t, n, c)) = guard.as_ref().and_then(|m| m.get(&path)) {
            if (*t, *n) == stamp {
                return c.clone();
            }
        }
    }
    let cat = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<Catalog>(&t).ok())
        .filter(|c| c.format == FORMAT || c.format.is_empty())
        .unwrap_or_default();
    let cat = Arc::new(cat);
    if let Ok(mut guard) = CACHE.lock() {
        guard
            .get_or_insert_with(HashMap::new)
            .insert(path, (stamp.0, stamp.1, cat.clone()));
    }
    cat
}

/// Is there a catalog at all?
pub fn exists(main: &Path) -> bool {
    path_in(main).is_file()
}

/// The catalog as text: one line per patient, study and series, so that a
/// person can read it and a change shows as a few changed lines.
fn render(c: &Catalog) -> Result<String> {
    fn map<T: Serialize>(m: &BTreeMap<String, T>) -> Result<String> {
        if m.is_empty() {
            return Ok("{}".into());
        }
        let mut lines = Vec::with_capacity(m.len());
        for (k, v) in m {
            lines.push(format!(
                "  {}: {}",
                serde_json::to_string(k)?,
                serde_json::to_string(v)?
            ));
        }
        Ok(format!("{{\n{}\n}}", lines.join(",\n")))
    }
    fn list<T: Serialize>(v: &[T]) -> Result<String> {
        if v.is_empty() {
            return Ok("[]".into());
        }
        let mut lines = Vec::with_capacity(v.len());
        for x in v {
            lines.push(format!("  {}", serde_json::to_string(x)?));
        }
        Ok(format!("[\n{}\n]", lines.join(",\n")))
    }
    Ok(format!(
        "{{\n\"format\": {},\n\"version\": {},\n\"updated\": {},\n\"storage\": {},\n\
         \"cohorts\": {},\n\"membership\": {},\n\"patients\": {},\n\"studies\": {},\n\
         \"series\": {},\n\"other\": {}\n}}\n",
        serde_json::to_string(FORMAT)?,
        VERSION,
        serde_json::to_string(&c.updated)?,
        serde_json::to_string(&c.storage)?,
        list(&c.cohorts)?,
        map(&c.membership)?,
        map(&c.patients)?,
        map(&c.studies)?,
        map(&c.series)?,
        list(&c.other)?,
    ))
}

static WRITES: AtomicU64 = AtomicU64::new(0);

/// Write the catalog in one rename, so that a reader never sees half of it.
fn store(main: &Path, c: &mut Catalog) -> Result<()> {
    c.format = FORMAT.into();
    c.version = VERSION;
    let (date, time) = crate::dicom_export::today();
    c.updated = format!("{date}T{time}");
    std::fs::create_dir_all(main).with_context(|| format!("create {}", main.display()))?;
    let path = path_in(main);
    let tmp = main.join(format!(
        ".{CATALOG_FILE}.{}.{}.tmp",
        std::process::id(),
        WRITES.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&tmp, render(c)?).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("write {}", path.display()))?;
    if let Ok(mut guard) = CACHE.lock() {
        if let Some(m) = guard.as_mut() {
            m.remove(&path);
        }
    }
    Ok(())
}

/// The catalog's lock: a file made exclusively, removed when dropped. A
/// lock older than two minutes is a program that died holding it.
struct Lock(PathBuf);

impl Lock {
    fn take(main: &Path) -> Result<Lock> {
        std::fs::create_dir_all(main).with_context(|| format!("create {}", main.display()))?;
        let path = main.join(LOCK_FILE);
        let started = std::time::Instant::now();
        loop {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(Lock(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|d| d > Duration::from_secs(120));
                    if stale {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    if started.elapsed() > Duration::from_secs(150) {
                        bail!(
                            "the PACS catalog stays locked ({}); another program is writing it",
                            path.display()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(40));
                }
                Err(e) => {
                    return Err(e).with_context(|| format!("lock {}", path.display()));
                }
            }
        }
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Change the catalog: read afresh under the lock, `f` changes it, written
/// back. What `f` answers is handed on.
pub fn update<T>(main: &Path, f: impl FnOnce(&mut Catalog) -> Result<T>) -> Result<T> {
    let _lock = Lock::take(main)?;
    let mut c = (*load(main)).clone();
    let out = f(&mut c)?;
    store(main, &mut c)?;
    Ok(out)
}

// ---- the old sidecars --------------------------------------------------------

/// Is this one of the `STUDY.txt` / `PATIENT.txt` / `INDEX.txt` files
/// earlier versions wrote into every folder? Recognised by name *and*
/// content, so that a person's own file of the same name stays.
fn is_old_sidecar(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let keys: &[&str] = match name {
        "STUDY.txt" => &["uid", "date", "description", "modalities", "files", "also"],
        "PATIENT.txt" => &["name", "id"],
        "INDEX.txt" => &[],
        _ => return false,
    };
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if meta.len() > 64 * 1024 * 1024 {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        return false;
    };
    let mut any = false;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        any = true;
        let ok = if name == "INDEX.txt" {
            line.split('\t').count() == 5
        } else {
            line.split_once('=')
                .is_some_and(|(k, _)| keys.contains(&k.trim().to_lowercase().as_str()))
        };
        if !ok {
            return false;
        }
    }
    any || name == "INDEX.txt"
}

/// Remove an old sidecar from a folder, if that is what it is.
pub fn drop_old_sidecars(dir: &Path) {
    for name in ["STUDY.txt", "PATIENT.txt", "INDEX.txt"] {
        let p = dir.join(name);
        if is_old_sidecar(&p) {
            let _ = std::fs::remove_file(&p);
        }
    }
}

// ---- scanning ----------------------------------------------------------------

/// A file found by the walk.
struct Seen {
    root: u32,
    rel: String,
    bytes: u64,
    mtime: i64,
    path: PathBuf,
}

/// Is this a file of the catalog's own, or one a writer has not finished?
fn is_own_file(name: &str) -> bool {
    name == CATALOG_FILE || name == LOCK_FILE || name.ends_with(".tmp") || name.starts_with('.')
}

/// Every candidate file below a storage folder. Folders and files whose
/// names start with a dot are skipped (the catalog's own, a version
/// control folder); the old sidecars are removed on the way.
fn walk_root(root: &Path, r: u32, out: &mut Vec<Seen>) {
    let walker = walkdir::WalkDir::new(root)
        .min_depth(1)
        .into_iter()
        .filter_entry(|e| e.depth() == 0 || !e.file_name().to_string_lossy().starts_with('.'));
    for e in walker.filter_map(|e| e.ok()) {
        if !e.file_type().is_file() {
            continue;
        }
        let name = e.file_name().to_string_lossy();
        if is_own_file(&name) {
            continue;
        }
        if matches!(&*name, "STUDY.txt" | "PATIENT.txt" | "INDEX.txt") && is_old_sidecar(e.path()) {
            let _ = std::fs::remove_file(e.path());
            continue;
        }
        let Some(rel) = rel_of(root, e.path()) else {
            continue;
        };
        let meta = e.metadata().ok();
        out.push(Seen {
            root: r,
            rel,
            bytes: meta.as_ref().map(|m| m.len()).unwrap_or(0),
            mtime: meta.as_ref().map(mtime_of).unwrap_or(0),
            path: e.into_path(),
        });
    }
}

/// Put one header's records into `c` (creating the patient, study and
/// series where new) and the file under its series.
pub fn add_file(c: &mut Catalog, h: &Head, file: FileRec) {
    let key = h.patient_key();
    c.patients.entry(key.clone()).or_insert_with(|| PatientRec {
        id: h.patient_id.clone(),
        name: h.patient_name.clone(),
        label: String::new(),
    });
    c.studies
        .entry(h.study.clone())
        .or_insert_with(|| StudyRec {
            patient: key,
            date: h.study_date.clone(),
            time: h.study_time.clone(),
            description: h.study_description.clone(),
            label: String::new(),
        });
    let s = c
        .series
        .entry(h.series.clone())
        .or_insert_with(|| SeriesRec {
            study: h.study.clone(),
            modality: h.modality.clone(),
            number: h.series_number.clone(),
            description: h.series_description.clone(),
            label: String::new(),
            files: Vec::new(),
        });
    s.files.push(file);
}

/// Drop the studies and patients no series refers to any more.
pub fn prune(c: &mut Catalog) {
    c.series.retain(|_, s| !s.files.is_empty());
    let studies: HashSet<&str> = c.series.values().map(|s| s.study.as_str()).collect();
    c.studies.retain(|uid, _| studies.contains(uid.as_str()));
    let patients: HashSet<&str> = c.studies.values().map(|s| s.patient.as_str()).collect();
    c.patients.retain(|k, _| patients.contains(k.as_str()));
    for s in c.series.values_mut() {
        s.files.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    }
}

/// Bring the catalog up to date with the storage folders: walk them, read
/// the files not seen before, forget the ones that are gone. The person's
/// names and cohorts are kept.
pub fn refresh(roots: &[PathBuf], progress: &Progress) -> Result<Arc<Catalog>> {
    let Some(main) = roots.first() else {
        bail!("the PACS has no storage folder");
    };
    let old = load(main);
    // The old catalog's folder indices, in today's list.
    let remap: Vec<Option<u32>> = old
        .storage
        .iter()
        .map(|s| {
            roots
                .iter()
                .position(|r| same_folder(r, s))
                .map(|i| i as u32)
        })
        .collect();
    let mut known: HashMap<(u32, String), (u64, i64, String, String)> = HashMap::new();
    for (suid, s) in &old.series {
        for f in &s.files {
            if let Some(Some(r)) = remap.get(f.0 as usize) {
                known.insert((*r, f.1.clone()), (f.2, f.3, f.4.clone(), suid.clone()));
            }
        }
    }
    let mut known_other: HashSet<OtherRec> = HashSet::new();
    for o in &old.other {
        if let Some(Some(r)) = remap.get(o.0 as usize) {
            known_other.insert(OtherRec(*r, o.1.clone(), o.2, o.3));
        }
    }

    progress.set("Looking through the storage folders");
    let mut seen: Vec<Seen> = Vec::new();
    for (i, r) in roots.iter().enumerate() {
        walk_root(r, i as u32, &mut seen);
        if progress.cancelled() {
            bail!(CANCELLED);
        }
    }

    // What is known and unchanged keeps its records; the rest is read.
    let mut c = Catalog::default();
    let mut fresh: Vec<&Seen> = Vec::new();
    let mut seen_paths: HashSet<(u32, &str)> = HashSet::new();
    for s in &seen {
        seen_paths.insert((s.root, s.rel.as_str()));
        match known.get(&(s.root, s.rel.clone())) {
            Some((b, t, sop, suid)) if *b == s.bytes && *t == s.mtime => {
                let file = FileRec(s.root, s.rel.clone(), s.bytes, s.mtime, sop.clone());
                copy_records(&mut c, &old, suid);
                if let Some(sr) = c.series.get_mut(suid) {
                    sr.files.push(file);
                }
            }
            _ => {
                let o = OtherRec(s.root, s.rel.clone(), s.bytes, s.mtime);
                if known_other.contains(&o) {
                    c.other.push(o);
                } else {
                    fresh.push(s);
                }
            }
        }
    }
    let total = fresh.len();
    let done = AtomicUsize::new(0);
    let heads: Vec<Option<Head>> = fresh
        .par_iter()
        .map(|s| {
            if progress.cancelled() {
                return None;
            }
            let n = done.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(64) {
                progress.set(format!("Reading new files {n}/{total}"));
            }
            read_head(&s.path)
        })
        .collect();
    if progress.cancelled() {
        bail!(CANCELLED);
    }
    for (s, h) in fresh.iter().zip(heads) {
        match h {
            Some(h) => {
                copy_records(&mut c, &old, &h.series);
                add_file(
                    &mut c,
                    &h,
                    FileRec(s.root, s.rel.clone(), s.bytes, s.mtime, h.sop.clone()),
                );
            }
            None => c
                .other
                .push(OtherRec(s.root, s.rel.clone(), s.bytes, s.mtime)),
        }
    }

    progress.set("Updating the catalog");
    let mut storage: Vec<String> = roots.iter().map(|r| r.display().to_string()).collect();
    let roots = roots.to_vec();
    let out = update(main, move |cur| {
        // The catalog's folders in this scan's numbering: the scanned ones
        // first, then any other folder the catalog lists (another program
        // serving these folders with a list of its own), whose files are
        // carried over as they are.
        let cur_remap: Vec<u32> = cur
            .storage
            .iter()
            .map(|s| match roots.iter().position(|r| same_folder(r, s)) {
                Some(i) => i as u32,
                None => {
                    storage.push(s.clone());
                    (storage.len() - 1) as u32
                }
            })
            .collect();
        for (suid, s) in &cur.series {
            for f in &s.files {
                let Some(&r) = cur_remap.get(f.0 as usize) else {
                    continue;
                };
                if (r as usize) < roots.len() {
                    // What another program filed meanwhile (an import into
                    // the same folders), the walk having missed it: kept
                    // while the file is there.
                    if seen_paths.contains(&(r, f.1.as_str()))
                        || known.contains_key(&(r, f.1.clone()))
                        || !abs_of(&roots[r as usize], &f.1).is_file()
                    {
                        continue;
                    }
                }
                copy_records(&mut c, cur, suid);
                if let Some(sr) = c.series.get_mut(suid) {
                    sr.files
                        .push(FileRec(r, f.1.clone(), f.2, f.3, f.4.clone()));
                }
            }
        }
        for o in &cur.other {
            if let Some(&r) = cur_remap.get(o.0 as usize) {
                if r as usize >= roots.len() {
                    c.other.push(OtherRec(r, o.1.clone(), o.2, o.3));
                }
            }
        }
        // The person's decisions come from the catalog as it is now.
        for (k, p) in c.patients.iter_mut() {
            if let Some(cp) = cur.patients.get(k) {
                p.label = cp.label.clone();
            }
        }
        for (k, s) in c.studies.iter_mut() {
            if let Some(cs) = cur.studies.get(k) {
                s.label = cs.label.clone();
            }
        }
        for (k, s) in c.series.iter_mut() {
            if let Some(cs) = cur.series.get(k) {
                s.label = cs.label.clone();
            }
        }
        c.cohorts = cur.cohorts.clone();
        c.membership = cur.membership.clone();
        c.storage = storage;
        prune(&mut c);
        ensure_cohorts(&mut c);
        *cur = c;
        Ok(())
    });
    out?;
    Ok(load(main))
}

/// Copy a series' records (and its study's and patient's) from `from`
/// into `c`, without its files, unless `c` has them already.
fn copy_records(c: &mut Catalog, from: &Catalog, series_uid: &str) {
    if c.series.contains_key(series_uid) {
        return;
    }
    let Some(s) = from.series.get(series_uid) else {
        return;
    };
    c.series.insert(
        series_uid.to_string(),
        SeriesRec {
            files: Vec::new(),
            ..s.clone()
        },
    );
    if let Some(st) = from.studies.get(&s.study) {
        c.studies
            .entry(s.study.clone())
            .or_insert_with(|| st.clone());
        if let Some(p) = from.patients.get(&st.patient) {
            c.patients
                .entry(st.patient.clone())
                .or_insert_with(|| p.clone());
        }
    }
}

// ---- cohorts -------------------------------------------------------------------

/// What a folder directly in a storage folder is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopKind {
    /// A cohort: a group of patients.
    Cohort,
    /// One patient's folder, outside any cohort.
    Patient,
}

/// Letters and digits only, lower case: what two spellings of a name
/// share.
fn squash(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Does a folder's name name this patient (their ID, or their family
/// name)?
fn names_patient(folder: &str, p: &PatientRec, key: &str) -> bool {
    let f = squash(folder);
    if f.is_empty() {
        return false;
    }
    let id = squash(&p.id);
    if id.len() >= 2 && (f.contains(&id) || (f.len() >= 2 && id.contains(&f))) {
        return true;
    }
    let k = squash(key);
    if k.len() >= 2 && f == k {
        return true;
    }
    let family = squash(p.name.split('^').next().unwrap_or(""));
    family.len() >= 3 && f.contains(&family)
}

/// How each folder directly in a storage folder reads: `(storage index,
/// folder name) -> kind`.
///
/// The PACS is laid out `storage / cohort / patient / study / series`,
/// but a patient's folder may also sit directly in the storage folder. A
/// folder holding several patients is a cohort. A folder holding one
/// patient is that patient's when it holds files itself, or is named
/// after them (their ID or family name); otherwise it is a cohort that so
/// far holds one patient.
pub fn top_folders(c: &Catalog) -> BTreeMap<(u32, String), TopKind> {
    let mut who: BTreeMap<(u32, String), (BTreeSet<&str>, bool)> = BTreeMap::new();
    for s in c.series.values() {
        let Some(st) = c.studies.get(&s.study) else {
            continue;
        };
        for f in &s.files {
            let mut parts = f.1.splitn(3, '/');
            let (Some(top), Some(_)) = (parts.next(), parts.next()) else {
                continue;
            };
            let direct = parts.next().is_none();
            let e = who.entry((f.0, top.to_string())).or_default();
            e.0.insert(st.patient.as_str());
            e.1 |= direct;
        }
    }
    who.into_iter()
        .map(|(k, (patients, direct))| {
            let kind = if patients.len() > 1 {
                TopKind::Cohort
            } else if direct {
                TopKind::Patient
            } else {
                let key = patients.iter().next().copied().unwrap_or("");
                match c.patients.get(key) {
                    Some(p) if names_patient(&k.1, p, key) => TopKind::Patient,
                    _ => TopKind::Cohort,
                }
            };
            (k, kind)
        })
        .collect()
}

/// A new cohort id, unlike the ones in use.
pub fn new_cohort_id(c: &Catalog) -> String {
    let mut n = c.cohorts.len() + 1;
    loop {
        let id = format!("c{n}");
        if !c.cohorts.iter().any(|x| x.id == id) {
            return id;
        }
        n += 1;
    }
}

/// Give every cohort folder a cohort, and drop the cohorts whose folder
/// has gone (a cohort made in the window stays until it is deleted).
pub fn ensure_cohorts(c: &mut Catalog) {
    let tops = top_folders(c);
    let names = c.storage.clone();
    let storage = |r: u32| names.get(r as usize).cloned().unwrap_or_default();
    let mut wanted: Vec<CohortFolder> = Vec::new();
    for ((r, name), kind) in &tops {
        if *kind == TopKind::Cohort {
            wanted.push(CohortFolder {
                storage: storage(*r),
                name: name.clone(),
            });
        }
    }
    c.cohorts.retain(|k| match &k.folder {
        None => true,
        Some(f) => wanted
            .iter()
            .any(|w| w.name == f.name && same_folder(Path::new(&w.storage), &f.storage)),
    });
    for w in wanted {
        let have = c.cohorts.iter().any(|k| {
            k.folder
                .as_ref()
                .is_some_and(|f| f.name == w.name && same_folder(Path::new(&w.storage), &f.storage))
        });
        if !have {
            let id = new_cohort_id(c);
            c.cohorts.push(CohortRec {
                id,
                name: w.name.clone(),
                folder: Some(w),
            });
        }
    }
    if !c.cohorts.iter().any(|k| k.id == UNSORTED) {
        c.cohorts.push(CohortRec {
            id: UNSORTED.into(),
            name: "Unsorted".into(),
            folder: None,
        });
    }
    // A hand-made membership naming a cohort that is gone forgets it.
    let ids: HashSet<String> = c.cohorts.iter().map(|k| k.id.clone()).collect();
    for v in c.membership.values_mut() {
        v.retain(|id| ids.contains(id));
    }
    c.membership.retain(|_, v| !v.is_empty());
}

/// The cohorts each patient's files lie in, by patient key: what a patient
/// is in until the person moves them.
pub fn folder_membership(c: &Catalog) -> BTreeMap<String, BTreeSet<String>> {
    let tops = top_folders(c);
    let cohort_of = |r: u32, top: &str| -> Option<String> {
        if tops.get(&(r, top.to_string())) != Some(&TopKind::Cohort) {
            return None;
        }
        let storage = Path::new(c.storage.get(r as usize)?);
        c.cohorts
            .iter()
            .find(|k| {
                k.folder
                    .as_ref()
                    .is_some_and(|f| f.name == top && same_folder(storage, &f.storage))
            })
            .map(|k| k.id.clone())
    };
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for s in c.series.values() {
        let Some(st) = c.studies.get(&s.study) else {
            continue;
        };
        for f in &s.files {
            let top = f.1.split('/').next().unwrap_or("");
            let id = if f.1.contains('/') {
                cohort_of(f.0, top)
            } else {
                None
            };
            out.entry(st.patient.clone())
                .or_default()
                .insert(id.unwrap_or_else(|| UNSORTED.to_string()));
        }
    }
    out
}

/// The cohorts each patient is in now: as the person put them, or as
/// their files lie.
pub fn membership(c: &Catalog) -> BTreeMap<String, BTreeSet<String>> {
    let mut out = folder_membership(c);
    let ids: HashSet<&str> = c.cohorts.iter().map(|k| k.id.as_str()).collect();
    for (key, set) in &c.membership {
        if !c.patients.contains_key(key) {
            continue;
        }
        let chosen: BTreeSet<String> = set
            .iter()
            .filter(|id| ids.contains(id.as_str()))
            .cloned()
            .collect();
        if !chosen.is_empty() {
            out.insert(key.clone(), chosen);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(r: u32, rel: &str, sop: &str) -> FileRec {
        FileRec(r, rel.into(), 1, 0, sop.into())
    }

    fn head(pid: &str, name: &str, study: &str, series: &str, sop: &str) -> Head {
        Head {
            sop: sop.into(),
            study: study.into(),
            series: series.into(),
            patient_id: pid.into(),
            patient_name: name.into(),
            modality: "CT".into(),
            ..Default::default()
        }
    }

    #[test]
    fn folders_read_as_cohorts_or_patients_by_what_they_hold() {
        let mut c = Catalog::default();
        // A cohort of two patients.
        add_file(
            &mut c,
            &head("X24", "GC", "1.1", "1.1.1", "9.1"),
            file(0, "Trial/X24/CT/a", "9.1"),
        );
        add_file(
            &mut c,
            &head("X31", "HM", "1.2", "1.2.1", "9.2"),
            file(0, "Trial/X31/b", "9.2"),
        );
        // A cohort of one patient, not named after them.
        add_file(
            &mut c,
            &head("RP-157", "DOE^JOHN", "1.3", "1.3.1", "9.3"),
            file(0, "Personal/Study/c", "9.3"),
        );
        // A patient's folder in the storage folder: named by their ID.
        add_file(
            &mut c,
            &head("P79", "ND", "1.4", "1.4.1", "9.4"),
            file(0, "P79/2020/d", "9.4"),
        );
        // A patient's folder holding files itself.
        add_file(
            &mut c,
            &head("Q1", "", "1.5", "1.5.1", "9.5"),
            file(0, "Whatever/e", "9.5"),
        );
        // Named after the family name.
        add_file(
            &mut c,
            &head("777", "SMITH^ANN", "1.6", "1.6.1", "9.6"),
            file(0, "Smith_Ann/CT/f", "9.6"),
        );
        let tops = top_folders(&c);
        assert_eq!(tops[&(0, "Trial".into())], TopKind::Cohort);
        assert_eq!(tops[&(0, "Personal".into())], TopKind::Cohort);
        assert_eq!(tops[&(0, "P79".into())], TopKind::Patient);
        assert_eq!(tops[&(0, "Whatever".into())], TopKind::Patient);
        assert_eq!(tops[&(0, "Smith_Ann".into())], TopKind::Patient);

        c.storage = vec!["/pacs".into()];
        ensure_cohorts(&mut c);
        let m = membership(&c);
        let trial = c
            .cohorts
            .iter()
            .find(|k| k.name == "Trial")
            .unwrap()
            .id
            .clone();
        assert_eq!(m["X24"], BTreeSet::from([trial.clone()]));
        assert_eq!(m["X31"], BTreeSet::from([trial.clone()]));
        assert_eq!(m["P79"], BTreeSet::from([UNSORTED.to_string()]));

        // Moved by hand into another cohort: the folder no longer decides.
        c.membership.insert("P79".into(), vec![trial.clone()]);
        assert_eq!(membership(&c)["P79"], BTreeSet::from([trial]));
    }

    #[test]
    fn the_catalog_reads_back_what_was_written() {
        let main = std::env::temp_dir().join("rds_catalog_roundtrip");
        let _ = std::fs::remove_dir_all(&main);
        update(&main, |c| {
            add_file(
                c,
                &head("P1", "DOE^JOHN", "1.2", "1.2.3", "1.2.3.4"),
                file(0, "a/b.dcm", "1.2.3.4"),
            );
            c.storage = vec![main.display().to_string()];
            c.patients.get_mut("P1").unwrap().label = "John".into();
            Ok(())
        })
        .unwrap();
        let back = load(&main);
        assert_eq!(back.format, FORMAT);
        assert_eq!(back.series["1.2.3"].files[0].4, "1.2.3.4");
        assert_eq!(back.patients["P1"].label, "John");
        let text = std::fs::read_to_string(path_in(&main)).unwrap();
        assert!(text.lines().count() > 8, "one line per record:\n{text}");
        assert!(
            !main.join(LOCK_FILE).exists(),
            "the lock goes with the write"
        );
        let _ = std::fs::remove_dir_all(&main);
    }

    #[test]
    fn only_the_sidecars_this_program_wrote_are_removed() {
        let dir = std::env::temp_dir().join("rds_catalog_sidecars");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("STUDY.txt"), "uid = 1.2\ndate = \nfiles = 3\n").unwrap();
        std::fs::write(dir.join("PATIENT.txt"), "My notes about this patient\n").unwrap();
        std::fs::write(dir.join("INDEX.txt"), "1\t2\tCT\t5\ta/b\n").unwrap();
        drop_old_sidecars(&dir);
        assert!(!dir.join("STUDY.txt").exists());
        assert!(
            dir.join("PATIENT.txt").exists(),
            "a person's own file stays"
        );
        assert!(!dir.join("INDEX.txt").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
