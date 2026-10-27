//! The PACS of this station: its storage folders, the catalog that lists
//! them, and what can be done with what they hold.
//!
//! A small PACS in the sense that matters day to day: every study is
//! listed without opening a DICOM file, loaded into a viewer workspace on
//! demand, given back the contours and segmentations drawn on it, and kept
//! in order - cohorts of patients, names that mean something, folders
//! that say what they hold.
//!
//! ## Layout
//!
//! ```text
//! <storage folder>/                 one or more; the first is the main one
//!   rds-pacs-catalog.json           the catalog (main folder only)
//!   <cohort>/                       a group of patients (a trial, a study)
//!     <patient>/
//!       <study>/
//!         <series>/  or the files themselves
//!   <patient>/                      a patient outside any cohort: Unsorted
//! ```
//!
//! That is how a person lays out a PACS by hand, and none of it is
//! required: the catalog reads every file's header once (see
//! [`catalog`]), so the patient, study and series of a file come from the
//! file, never from where it lies. A folder holding the files of several
//! studies, or several patients, is listed as what it holds. Only the
//! cohorts come from the folders: a folder directly in a storage folder
//! that holds several patients is a cohort, one patient's folder there is
//! *Unsorted* (see [`catalog::top_folders`] for the rule). After that the
//! person decides: cohorts are renamed, made and removed, and patients
//! moved or copied between them in the catalog, without a DICOM file
//! moving. Moving files is a separate, explicit act
//! ([`Archive::move_into`]).
//!
//! What this station files itself (an import, the structures sent back
//! from a workspace) goes into the main storage folder as
//! `<patient>/<study uid>/<sop uid>.dcm`, beside the study when the main
//! folder holds it already.

pub mod catalog;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use crate::progress::Progress;
use crate::settings;
pub use catalog::{patient_key, CATALOG_FILE, UNSORTED};
use catalog::{Catalog, FileRec, Head};

/// One series of the PACS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeriesEntry {
    pub series_uid: String,
    pub modality: String,
    pub number: String,
    pub description: String,
    /// The name given in the PACS window; empty: none.
    pub label: String,
    pub files: usize,
    /// The folder that holds its files (the nearest one holding all).
    pub dir: PathBuf,
}

impl SeriesEntry {
    /// What the DICOM header calls it: `CT 2 - Thorax 3 mm`.
    pub fn dicom_title(&self) -> String {
        let mut t = if self.modality.is_empty() {
            "?".to_string()
        } else {
            self.modality.clone()
        };
        if !self.number.trim().is_empty() {
            t.push(' ');
            t.push_str(self.number.trim());
        }
        if !self.description.trim().is_empty() {
            t.push_str(" - ");
            t.push_str(self.description.trim());
        }
        t
    }

    /// What the window calls it: the name given, or the header's.
    pub fn title(&self) -> String {
        if self.label.is_empty() {
            self.dicom_title()
        } else {
            self.label.clone()
        }
    }
}

/// One study of the PACS.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StudyEntry {
    pub study_uid: String,
    pub date: String,
    pub description: String,
    /// Every modality present in the study, sorted.
    pub modalities: Vec<String>,
    pub files: usize,
    /// The folder that holds its files (the nearest one holding all).
    pub dir: PathBuf,
    /// The name given in the PACS window; empty: none.
    pub label: String,
    pub series: Vec<SeriesEntry>,
}

/// `20260827` as `2026-08-27`; anything else as it is.
pub fn show_date(d: &str) -> String {
    let d = d.trim();
    if d.len() == 8 && d.bytes().all(|b| b.is_ascii_digit()) {
        format!("{}-{}-{}", &d[..4], &d[4..6], &d[6..])
    } else {
        d.to_string()
    }
}

impl StudyEntry {
    /// `2026-08-27 - Planning · CT, RTSTRUCT`.
    pub fn dicom_title(&self) -> String {
        format!(
            "{}{} · {}",
            if self.date.is_empty() {
                "undated".into()
            } else {
                show_date(&self.date)
            },
            if self.description.is_empty() {
                String::new()
            } else {
                format!(" - {}", self.description)
            },
            if self.modalities.is_empty() {
                "?".into()
            } else {
                self.modalities.join(", ")
            },
        )
    }

    /// What the window calls it: the name given, or the header's.
    pub fn title(&self) -> String {
        if self.label.is_empty() {
            self.dicom_title()
        } else {
            self.label.clone()
        }
    }

    /// `2026-08-27 - Planning · CT, RTSTRUCT · 214 files`.
    pub fn describe(&self) -> String {
        format!(
            "{} · {} file{}",
            self.title(),
            self.files,
            if self.files == 1 { "" } else { "s" }
        )
    }
}

/// One patient of the PACS and their studies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PatientEntry {
    pub name: String,
    pub id: String,
    /// The folder that holds their files (the nearest one holding all).
    pub dir: PathBuf,
    pub studies: Vec<StudyEntry>,
    /// What the catalog and a remote station name them by
    /// ([`patient_key`]).
    pub key: String,
    /// The name given in the PACS window; empty: none.
    pub label: String,
}

impl PatientEntry {
    /// What the DICOM header calls them: `Doe John (P0001)`.
    pub fn dicom_title(&self) -> String {
        let name = self.name.replace('^', " ").trim().to_string();
        match (name.is_empty(), self.id.is_empty()) {
            (true, true) => "Unknown patient".into(),
            (true, false) => format!("Patient {}", self.id),
            (false, true) => name,
            (false, false) => format!("{name} ({})", self.id),
        }
    }

    /// What the window calls them: the name given, or the header's.
    pub fn title(&self) -> String {
        if self.label.is_empty() {
            self.dicom_title()
        } else {
            self.label.clone()
        }
    }

    pub fn files(&self) -> usize {
        self.studies.iter().map(|s| s.files).sum()
    }

    /// What routes and task bindings name the patient by.
    pub fn key(&self) -> String {
        if self.key.is_empty() {
            patient_key(&self.id, &self.name)
        } else {
            self.key.clone()
        }
    }
}

/// One cohort: a group of patients, from a folder or made in the window.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CohortEntry {
    pub id: String,
    pub name: String,
    /// Its folder; `None` for one made in the window and for *Unsorted*.
    pub folder: Option<PathBuf>,
    pub patients: Vec<PatientEntry>,
}

impl CohortEntry {
    pub fn files(&self) -> usize {
        self.patients.iter().map(|p| p.files()).sum()
    }

    pub fn is_unsorted(&self) -> bool {
        self.id == UNSORTED
    }
}

/// The PACS as the window shows it: cohorts, their patients, studies and
/// series.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tree {
    pub cohorts: Vec<CohortEntry>,
}

impl Tree {
    /// Every patient once, in name order (a patient in two cohorts is one
    /// patient).
    pub fn patients(&self) -> Vec<PatientEntry> {
        let mut seen = HashSet::new();
        let mut out: Vec<PatientEntry> = self
            .cohorts
            .iter()
            .flat_map(|c| c.patients.iter())
            .filter(|p| seen.insert(p.key.clone()))
            .cloned()
            .collect();
        out.sort_by_key(|p| p.title().to_lowercase());
        out
    }

    pub fn studies(&self) -> usize {
        self.patients().iter().map(|p| p.studies.len()).sum()
    }
}

/// One file of a study, as a remote copy of the PACS needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance {
    pub sop_uid: String,
    pub series_uid: String,
    pub path: PathBuf,
    pub bytes: u64,
}

/// A row of the PACS: what a selection, a removal or a name is about.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Pick {
    /// A cohort, by id: all its patients.
    Cohort(String),
    /// A patient, by key: all their studies.
    Patient(String),
    /// A study, by Study Instance UID.
    Study(String),
    /// A series, by Series Instance UID.
    Series(String),
}

/// Is `s` a DICOM UID: digits and single dots, 1 to 64 characters? What
/// comes over the network is checked with this before anything looks for
/// it.
pub fn is_uid(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes().all(|b| b.is_ascii_digit() || b == b'.')
        && !s.starts_with('.')
        && !s.ends_with('.')
        && !s.contains("..")
}

/// What an import did, for the line the window reports afterwards.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportSummary {
    pub stored: usize,
    /// Files the PACS holds already (the same SOP Instance UID).
    pub duplicates: usize,
    /// Files that were not readable as DICOM.
    pub skipped: usize,
    pub patients: usize,
    pub studies: usize,
}

impl ImportSummary {
    pub fn describe(&self) -> String {
        format!(
            "{} file(s) filed under {} patient(s) / {} study(ies){}{}",
            self.stored,
            self.patients,
            self.studies,
            if self.duplicates > 0 {
                format!(", {} already there", self.duplicates)
            } else {
                String::new()
            },
            if self.skipped > 0 {
                format!(", {} not DICOM", self.skipped)
            } else {
                String::new()
            }
        )
    }
}

/// What a move of files did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MoveSummary {
    pub moved: usize,
    /// Files already where they were to go.
    pub unchanged: usize,
    /// Folders left empty by the move, removed.
    pub folders_removed: usize,
}

/// Default main storage folder: `<platform data dir>/archive`.
pub fn default_root() -> PathBuf {
    settings::data_dir().join("archive")
}

/// The folder a settings value names, falling back to [`default_root`]
/// when it is blank.
pub fn root_from_setting(setting: &str) -> PathBuf {
    let t = setting.trim();
    if t.is_empty() {
        default_root()
    } else {
        PathBuf::from(t)
    }
}

/// Keep a free-text identifier usable as a folder name.
///
/// Patient identifiers arrive as whatever the acquiring system wrote -
/// slashes, colons, trailing spaces, non-ASCII. Anything outside a
/// conservative set becomes `_`.
pub fn sanitize(s: &str) -> String {
    let out: String = s
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let out = out.trim_matches('_').to_string();
    if out.is_empty() {
        "unknown".into()
    } else {
        out.chars().take(96).collect()
    }
}

/// The nearest folder holding every one of `paths` (files).
fn common_dir<'a>(mut paths: impl Iterator<Item = &'a Path>) -> PathBuf {
    let Some(first) = paths.next() else {
        return PathBuf::new();
    };
    let mut acc = first.parent().map(Path::to_path_buf).unwrap_or_default();
    for p in paths {
        while !p.starts_with(&acc) {
            match acc.parent() {
                Some(up) => acc = up.to_path_buf(),
                None => return PathBuf::new(),
            }
        }
    }
    acc
}

/// Remove `dir` and the folders above it while they are empty, stopping
/// at `stop` (a storage folder) and at any folder in `keep`. The old
/// sidecars do not count as content. Returns how many went.
fn prune_empty(dir: &Path, stop: &Path, keep: &[PathBuf]) -> usize {
    let mut n = 0;
    let mut d = dir.to_path_buf();
    while d.starts_with(stop) && d != stop && !keep.iter().any(|k| k.starts_with(&d)) {
        catalog::drop_old_sidecars(&d);
        let empty = std::fs::read_dir(&d)
            .map(|mut rd| rd.next().is_none())
            .unwrap_or(false);
        if !empty || std::fs::remove_dir(&d).is_err() {
            break;
        }
        n += 1;
        match d.parent() {
            Some(up) => d = up.to_path_buf(),
            None => break,
        }
    }
    n
}

/// Is `name` usable as one folder's name?
fn folder_name_ok(name: &str) -> bool {
    let t = name.trim();
    !t.is_empty()
        && t != "."
        && t != ".."
        && !t.starts_with('.')
        && !t.chars().any(|c| {
            matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()
        })
}

/// This station's PACS: the main storage folder and the others, from the
/// settings.
pub fn station_roots(main: &str, more: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = vec![root_from_setting(main)];
    for m in more {
        if !m.as_os_str().is_empty() && !roots.iter().any(|r| r == m) {
            roots.push(m.clone());
        }
    }
    roots
}

/// A station's DICOM store: one or more storage folders, the first the
/// main one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Archive {
    roots: Vec<PathBuf>,
}

impl Archive {
    /// A PACS with one storage folder.
    pub fn new(root: impl Into<PathBuf>) -> Archive {
        Archive {
            roots: vec![root.into()],
        }
    }

    /// A PACS with these storage folders, the first the main one.
    pub fn with_roots(roots: Vec<PathBuf>) -> Archive {
        let mut out: Vec<PathBuf> = Vec::new();
        for r in roots {
            if !r.as_os_str().is_empty() && !out.contains(&r) {
                out.push(r);
            }
        }
        if out.is_empty() {
            out.push(default_root());
        }
        Archive { roots: out }
    }

    /// This station's PACS, as its settings name it.
    pub fn station() -> Archive {
        let s = settings::load();
        let main = s
            .archive_dir
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        Archive::with_roots(station_roots(&main, &s.archive_more))
    }

    /// The PACS a workflow step or a tool names: blank is the station's,
    /// otherwise one folder or several joined by `|`.
    pub fn from_setting(setting: &str) -> Archive {
        let t = setting.trim();
        if t.is_empty() {
            return Archive::station();
        }
        Archive::with_roots(
            t.split('|')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from)
                .collect(),
        )
    }

    /// The main storage folder: where new data is written and the catalog
    /// lives.
    pub fn root(&self) -> &Path {
        &self.roots[0]
    }

    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// The storage folders joined by `|`, as [`Archive::from_setting`]
    /// reads them.
    pub fn setting(&self) -> String {
        self.roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join("|")
    }

    pub fn catalog_path(&self) -> PathBuf {
        catalog::path_in(self.root())
    }

    /// The catalog, scanned first if there is none yet.
    fn cat(&self) -> Arc<Catalog> {
        if !catalog::exists(self.root()) {
            if let Ok(c) = catalog::refresh(&self.roots, &Progress::default()) {
                return c;
            }
        }
        catalog::load(self.root())
    }

    /// The catalog's folder index of each storage folder of this PACS.
    fn storage_map(&self, c: &Catalog) -> HashMap<u32, usize> {
        let mut out = HashMap::new();
        for (i, s) in c.storage.iter().enumerate() {
            if let Some(r) = self.roots.iter().position(|r| catalog::same_folder(r, s)) {
                out.insert(i as u32, r);
            }
        }
        out
    }

    /// The absolute path of a file of the catalog, when it lies in one of
    /// this PACS's folders.
    fn path_of(&self, map: &HashMap<u32, usize>, f: &FileRec) -> Option<PathBuf> {
        let r = map.get(&f.0)?;
        Some(catalog::abs_of(&self.roots[*r], &f.1))
    }

    /// Walk the storage folders and bring the catalog up to date.
    pub fn refresh(&self, progress: &Progress) -> Result<()> {
        catalog::refresh(&self.roots, progress).map(|_| ())
    }

    /// Bring the catalog up to date and list the PACS.
    pub fn scan_tree(&self, progress: &Progress) -> Result<Tree> {
        let c = catalog::refresh(&self.roots, progress)?;
        Ok(self.tree_of(&c))
    }

    /// The PACS as the catalog last saw it, without walking the folders.
    pub fn tree(&self) -> Tree {
        self.tree_of(&self.cat())
    }

    /// Every patient of the PACS with their studies, in name order. Walks
    /// the folders first (reading only files not seen before).
    pub fn scan(&self) -> Result<Vec<PatientEntry>> {
        Ok(self.scan_tree(&Progress::default())?.patients())
    }

    fn tree_of(&self, c: &Catalog) -> Tree {
        let map = self.storage_map(c);
        // Series -> entry, with their files' paths.
        let mut by_study: BTreeMap<&str, Vec<(SeriesEntry, Vec<PathBuf>)>> = BTreeMap::new();
        for (uid, s) in &c.series {
            let mut paths = Vec::new();
            let mut sops = HashSet::new();
            for f in &s.files {
                if let Some(p) = self.path_of(&map, f) {
                    sops.insert(f.4.as_str());
                    paths.push(p);
                }
            }
            if paths.is_empty() {
                continue;
            }
            let entry = SeriesEntry {
                series_uid: uid.clone(),
                modality: s.modality.clone(),
                number: s.number.clone(),
                description: s.description.clone(),
                label: s.label.clone(),
                files: sops.len(),
                dir: common_dir(paths.iter().map(PathBuf::as_path)),
            };
            by_study
                .entry(s.study.as_str())
                .or_default()
                .push((entry, paths));
        }
        // Study -> entry, by patient.
        let mut by_patient: BTreeMap<&str, Vec<(StudyEntry, Vec<PathBuf>)>> = BTreeMap::new();
        for (uid, series) in by_study {
            let Some(st) = c.studies.get(uid) else {
                continue;
            };
            let mut series = series;
            series.sort_by(|a, b| {
                let n = |s: &SeriesEntry| s.number.trim().parse::<i64>().unwrap_or(i64::MAX);
                (n(&a.0), a.0.modality.as_str(), a.0.description.as_str()).cmp(&(
                    n(&b.0),
                    b.0.modality.as_str(),
                    b.0.description.as_str(),
                ))
            });
            let modalities: BTreeSet<String> = series
                .iter()
                .map(|(s, _)| s.modality.clone())
                .filter(|m| !m.is_empty())
                .collect();
            let files = c
                .series
                .values()
                .filter(|s| s.study == uid)
                .flat_map(|s| s.files.iter())
                .filter(|f| map.contains_key(&f.0))
                .map(|f| f.4.as_str())
                .collect::<HashSet<_>>()
                .len();
            let paths: Vec<PathBuf> = series.iter().flat_map(|(_, p)| p.iter().cloned()).collect();
            let entry = StudyEntry {
                study_uid: uid.to_string(),
                date: st.date.clone(),
                description: st.description.clone(),
                modalities: modalities.into_iter().collect(),
                files,
                dir: common_dir(paths.iter().map(PathBuf::as_path)),
                label: st.label.clone(),
                series: series.into_iter().map(|(s, _)| s).collect(),
            };
            by_patient
                .entry(st.patient.as_str())
                .or_default()
                .push((entry, paths));
        }
        let mut patients: BTreeMap<&str, PatientEntry> = BTreeMap::new();
        for (key, studies) in by_patient {
            let p = c.patients.get(key).cloned().unwrap_or_default();
            let dir = common_dir(
                studies
                    .iter()
                    .flat_map(|(_, p)| p.iter().map(PathBuf::as_path)),
            );
            let mut studies: Vec<StudyEntry> = studies.into_iter().map(|(s, _)| s).collect();
            // Newest study first - what one is normally after.
            studies.sort_by(|a, b| b.date.cmp(&a.date).then(a.study_uid.cmp(&b.study_uid)));
            patients.insert(
                key,
                PatientEntry {
                    name: p.name,
                    id: p.id,
                    dir,
                    studies,
                    key: key.to_string(),
                    label: p.label,
                },
            );
        }
        let members = catalog::membership(c);
        let mut cohorts: Vec<CohortEntry> = Vec::new();
        for k in &c.cohorts {
            let folder = k.folder.as_ref().and_then(|f| {
                let r = self
                    .roots
                    .iter()
                    .find(|r| catalog::same_folder(r, &f.storage))?;
                Some(r.join(&f.name))
            });
            let mut list: Vec<PatientEntry> = patients
                .iter()
                .filter(|(key, _)| members.get(**key).is_some_and(|m| m.contains(&k.id)))
                .map(|(_, p)| p.clone())
                .collect();
            if list.is_empty() && (k.folder.is_some() || k.id == UNSORTED) {
                // A folder cohort lives as long as its folder holds
                // someone; Unsorted shows only when someone is in it. A
                // cohort made in the window shows empty, to be filled.
                continue;
            }
            list.sort_by_key(|p| p.title().to_lowercase());
            cohorts.push(CohortEntry {
                id: k.id.clone(),
                name: k.name.clone(),
                folder,
                patients: list,
            });
        }
        // A patient the catalog has no cohort for yet (it was scanned by
        // an older catalog) is not lost: Unsorted takes them.
        let placed: HashSet<&str> = cohorts
            .iter()
            .flat_map(|c| c.patients.iter().map(|p| p.key.as_str()))
            .collect();
        let stray: Vec<PatientEntry> = patients
            .values()
            .filter(|p| !placed.contains(p.key.as_str()))
            .cloned()
            .collect();
        if !stray.is_empty() {
            match cohorts.iter_mut().find(|c| c.is_unsorted()) {
                Some(u) => u.patients.extend(stray),
                None => cohorts.push(CohortEntry {
                    id: UNSORTED.into(),
                    name: "Unsorted".into(),
                    folder: None,
                    patients: stray,
                }),
            }
        }
        cohorts.sort_by(|a, b| {
            (a.is_unsorted(), a.name.to_lowercase()).cmp(&(b.is_unsorted(), b.name.to_lowercase()))
        });
        Tree { cohorts }
    }

    /// The folder of the study with this Study Instance UID (the nearest
    /// one holding all its files). `None` when the PACS does not hold it.
    ///
    /// The UID is looked up in the catalog, never joined onto a path, so a
    /// UID that came over the network cannot name anything else.
    pub fn find_study(&self, study_uid: &str) -> Option<PathBuf> {
        if study_uid.is_empty() {
            return None;
        }
        let inst = self.study_instances(study_uid).ok()?;
        if inst.is_empty() {
            return None;
        }
        Some(common_dir(inst.iter().map(|i| i.path.as_path())))
    }

    /// The folder of the patient with this key (the nearest one holding
    /// all their files).
    pub fn find_patient(&self, key: &str) -> Option<PathBuf> {
        let files = self.files_of(&[Pick::Patient(key.to_string())]);
        if files.is_empty() {
            return None;
        }
        Some(common_dir(files.iter().map(|i| i.path.as_path())))
    }

    /// The key of the patient a study belongs to.
    pub fn patient_of_study(&self, study_uid: &str) -> Option<String> {
        self.cat().studies.get(study_uid).map(|s| s.patient.clone())
    }

    /// The files of one study, one per SOP Instance UID, in UID order.
    pub fn study_instances(&self, study_uid: &str) -> Result<Vec<Instance>> {
        Ok(self.files_of(&[Pick::Study(study_uid.to_string())]))
    }

    /// The series a pick stands for.
    fn series_of<'c>(&self, c: &'c Catalog, picks: &[Pick]) -> BTreeSet<&'c str> {
        let mut keys: HashSet<&str> = HashSet::new();
        let mut studies: HashSet<&str> = HashSet::new();
        let mut series: BTreeSet<&'c str> = BTreeSet::new();
        let members = if picks.iter().any(|p| matches!(p, Pick::Cohort(_))) {
            catalog::membership(c)
        } else {
            BTreeMap::new()
        };
        for p in picks {
            match p {
                Pick::Cohort(id) => {
                    for (k, m) in &members {
                        if m.contains(id) {
                            keys.insert(k.as_str());
                        }
                    }
                }
                Pick::Patient(k) => {
                    keys.insert(k.as_str());
                }
                Pick::Study(u) => {
                    studies.insert(u.as_str());
                }
                Pick::Series(u) => {
                    if let Some((k, _)) = c.series.get_key_value(u.as_str()) {
                        series.insert(k.as_str());
                    }
                }
            }
        }
        for (uid, s) in &c.series {
            let in_study = studies.contains(s.study.as_str());
            let of_patient = !keys.is_empty()
                && c.studies
                    .get(&s.study)
                    .is_some_and(|st| keys.contains(st.patient.as_str()));
            if in_study || of_patient {
                series.insert(uid.as_str());
            }
        }
        series
    }

    /// The files a selection stands for, one per SOP Instance UID, in UID
    /// order.
    pub fn files_of(&self, picks: &[Pick]) -> Vec<Instance> {
        let c = self.cat();
        let map = self.storage_map(&c);
        let mut out: BTreeMap<&str, Instance> = BTreeMap::new();
        for uid in self.series_of(&c, picks) {
            let Some(s) = c.series.get(uid) else {
                continue;
            };
            for f in &s.files {
                if out.contains_key(f.4.as_str()) {
                    continue;
                }
                if let Some(path) = self.path_of(&map, f) {
                    out.insert(
                        f.4.as_str(),
                        Instance {
                            sop_uid: f.4.clone(),
                            series_uid: uid.to_string(),
                            path,
                            bytes: f.2,
                        },
                    );
                }
            }
        }
        out.into_values().collect()
    }

    /// What a session remembers for a loaded selection: a folder where all
    /// of its DICOM files were loaded, the files themselves otherwise.
    pub fn sources_of(&self, files: &[PathBuf]) -> Vec<PathBuf> {
        let c = self.cat();
        let map = self.storage_map(&c);
        let mut in_dir: HashMap<PathBuf, usize> = HashMap::new();
        for s in c.series.values() {
            for f in &s.files {
                if let Some(p) = self.path_of(&map, f) {
                    if let Some(d) = p.parent() {
                        *in_dir.entry(d.to_path_buf()).or_default() += 1;
                    }
                }
            }
        }
        let mut chosen: BTreeMap<PathBuf, Vec<PathBuf>> = BTreeMap::new();
        for f in files {
            if let Some(d) = f.parent() {
                chosen.entry(d.to_path_buf()).or_default().push(f.clone());
            }
        }
        let mut out = Vec::new();
        for (d, fs) in chosen {
            if in_dir.get(&d).copied() == Some(fs.len()) {
                out.push(d);
            } else {
                out.extend(fs);
            }
        }
        out
    }

    /// Does the PACS hold anything at all? Cheap: the catalog when there is
    /// one, else one directory read per folder.
    pub fn has_patients(&self) -> bool {
        if catalog::exists(self.root()) {
            let c = catalog::load(self.root());
            if !c.series.is_empty() {
                return true;
            }
        }
        self.roots.iter().any(|r| {
            std::fs::read_dir(r)
                .map(|entries| {
                    entries.flatten().any(|e| {
                        let n = e.file_name().to_string_lossy().into_owned();
                        !n.starts_with('.') && n != CATALOG_FILE
                    })
                })
                .unwrap_or(false)
        })
    }

    /// Where the main folder files a study it does not hold yet.
    fn new_study_dir(&self, patient: &str, study_uid: &str) -> PathBuf {
        self.root()
            .join(sanitize(patient))
            .join(sanitize(study_uid))
    }

    /// File every DICOM file under `src` into the main storage folder.
    ///
    /// Files are copied, never moved: importing must not take the source
    /// folder apart. A file whose SOP Instance UID the PACS holds already
    /// is counted as a duplicate and left alone, so importing the same
    /// folder twice is a no-op rather than a second copy. A study the main
    /// folder holds already gets its new files beside the old ones.
    ///
    /// One folder, one catalog update; [`Archive::importer`] files several
    /// folders under one update.
    pub fn import(&self, src: &Path, progress: &Progress) -> Result<ImportSummary> {
        let mut im = self.importer();
        let filed = im.file_folder(src, progress);
        // What was filed before a failure is in the catalog all the same.
        progress.set("Updating the catalog");
        let finished = im.finish();
        let (sum, _) = filed?;
        finished?;
        progress.set("done");
        Ok(sum)
    }

    /// An import of several folders one after the other (the bundles of a
    /// study pulled from a server), with the catalog brought up to date
    /// once, by [`Importer::finish`].
    pub fn importer(&self) -> Importer<'_> {
        let c = self.cat();
        let map = self.storage_map(&c);
        let held: HashSet<String> = c
            .series
            .values()
            .flat_map(|s| s.files.iter())
            .filter(|f| map.contains_key(&f.0))
            .map(|f| f.4.clone())
            .collect();
        Importer {
            archive: self,
            cat: c,
            map,
            held,
            homes: HashMap::new(),
            filed: Vec::new(),
            take_files: false,
        }
    }

    /// Delete a folder of the PACS and forget what it held.
    ///
    /// Refuses anything that is not inside one of the storage folders (or
    /// is one), because the path comes from a listing that a stale rescan
    /// could have made wrong.
    pub fn remove(&self, dir: &Path) -> Result<()> {
        let target = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        let root = self.roots.iter().find(|r| {
            let r = r.canonicalize().unwrap_or_else(|_| r.to_path_buf());
            target.starts_with(&r) && target != r
        });
        let Some(root) = root else {
            bail!("{} is not inside the PACS", dir.display());
        };
        std::fs::remove_dir_all(&target).with_context(|| format!("remove {}", target.display()))?;
        let Some(rel) = catalog::rel_of(
            &root.canonicalize().unwrap_or_else(|_| root.clone()),
            &target,
        ) else {
            return Ok(());
        };
        if catalog::exists(self.root()) {
            let root = root.clone();
            catalog::update(self.root(), |c| {
                let r = storage_index(c, &root);
                let prefix = format!("{rel}/");
                for s in c.series.values_mut() {
                    s.files.retain(|f| !(f.0 == r && f.1.starts_with(&prefix)));
                }
                catalog::prune(c);
                catalog::ensure_cohorts(c);
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Delete the files of a selection (every copy of them in this PACS's
    /// folders), and the folders that leaves empty. Returns how many files
    /// went.
    pub fn remove_picks(&self, picks: &[Pick]) -> Result<usize> {
        let c = self.cat();
        let map = self.storage_map(&c);
        let uids: Vec<String> = self
            .series_of(&c, picks)
            .into_iter()
            .map(str::to_string)
            .collect();
        let mut gone: HashSet<(u32, String)> = HashSet::new();
        let mut dirs: BTreeSet<(usize, PathBuf)> = BTreeSet::new();
        for uid in &uids {
            let Some(s) = c.series.get(uid) else {
                continue;
            };
            for f in &s.files {
                let Some(&r) = map.get(&f.0) else {
                    continue;
                };
                let path = catalog::abs_of(&self.roots[r], &f.1);
                match std::fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(e).with_context(|| format!("remove {}", path.display()));
                    }
                }
                gone.insert((f.0, f.1.clone()));
                if let Some(d) = path.parent() {
                    dirs.insert((r, d.to_path_buf()));
                }
            }
        }
        for (r, d) in dirs.iter().rev() {
            prune_empty(d, &self.roots[*r], &[]);
        }
        let n = gone.len();
        catalog::update(self.root(), |c| {
            for s in c.series.values_mut() {
                s.files.retain(|f| !gone.contains(&(f.0, f.1.clone())));
            }
            catalog::prune(c);
            catalog::ensure_cohorts(c);
            Ok(())
        })?;
        Ok(n)
    }

    /// Remove one study (its files, every copy).
    pub fn remove_study(&self, study_uid: &str) -> Result<()> {
        if self.study_instances(study_uid)?.is_empty() {
            bail!("the PACS has no study {study_uid}");
        }
        self.remove_picks(&[Pick::Study(study_uid.to_string())])
            .map(|_| ())
    }

    /// Remove one patient (their files, every copy).
    pub fn remove_patient(&self, key: &str) -> Result<()> {
        if self.files_of(&[Pick::Patient(key.to_string())]).is_empty() {
            bail!("the PACS has no patient {key}");
        }
        self.remove_picks(&[Pick::Patient(key.to_string())])
            .map(|_| ())
    }

    // ---- names and cohorts -------------------------------------------------

    /// Give a row a name of the person's choosing; an empty name goes back
    /// to what the DICOM header (or the folder) says. Nothing is written
    /// into a DICOM file.
    pub fn rename(&self, target: &Pick, name: &str) -> Result<()> {
        let name = name.trim().to_string();
        catalog::update(self.root(), |c| {
            match target {
                Pick::Cohort(id) => {
                    let k = c
                        .cohorts
                        .iter_mut()
                        .find(|k| &k.id == id)
                        .context("the cohort is no longer there")?;
                    k.name = if !name.is_empty() {
                        name
                    } else if let Some(f) = &k.folder {
                        f.name.clone()
                    } else if k.id == UNSORTED {
                        "Unsorted".into()
                    } else {
                        "Cohort".into()
                    };
                }
                Pick::Patient(key) => {
                    c.patients
                        .get_mut(key)
                        .context("the patient is no longer there")?
                        .label = name;
                }
                Pick::Study(uid) => {
                    c.studies
                        .get_mut(uid)
                        .context("the study is no longer there")?
                        .label = name;
                }
                Pick::Series(uid) => {
                    c.series
                        .get_mut(uid)
                        .context("the series is no longer there")?
                        .label = name;
                }
            }
            Ok(())
        })
    }

    /// A new, empty cohort; returns its id.
    pub fn new_cohort(&self, name: &str) -> Result<String> {
        let name = name.trim().to_string();
        catalog::update(self.root(), |c| {
            let id = catalog::new_cohort_id(c);
            c.cohorts.push(catalog::CohortRec {
                id: id.clone(),
                name: if name.is_empty() {
                    "New cohort".into()
                } else {
                    name
                },
                folder: None,
            });
            Ok(id)
        })
    }

    /// Delete a cohort made in the window. Its patients go back to where
    /// their files put them (or stay in their other cohorts). A cohort
    /// with a folder goes when its folder does, and is refused here.
    pub fn delete_cohort(&self, id: &str) -> Result<()> {
        catalog::update(self.root(), |c| {
            let k = c
                .cohorts
                .iter()
                .find(|k| k.id == id)
                .context("the cohort is no longer there")?;
            if k.folder.is_some() || k.id == UNSORTED {
                bail!(
                    "'{}' is a folder's cohort: it goes when its folder does; rename it, or \
                     move its patients elsewhere",
                    k.name
                );
            }
            c.cohorts.retain(|k| k.id != id);
            for v in c.membership.values_mut() {
                v.retain(|x| x != id);
            }
            c.membership.retain(|_, v| !v.is_empty());
            Ok(())
        })
    }

    /// Put patients into cohort `to`: moved out of `from` (or out of every
    /// cohort, when `from` is `None`), or copied (kept where they are too).
    /// Only the catalog changes; the files stay where they are.
    pub fn put_in_cohort(
        &self,
        keys: &[String],
        from: Option<&str>,
        to: &str,
        copy: bool,
    ) -> Result<()> {
        catalog::update(self.root(), |c| {
            if !c.cohorts.iter().any(|k| k.id == to) {
                if to == UNSORTED {
                    c.cohorts.push(catalog::CohortRec {
                        id: UNSORTED.into(),
                        name: "Unsorted".into(),
                        folder: None,
                    });
                } else {
                    bail!("the cohort is no longer there");
                }
            }
            let now = catalog::membership(c);
            for key in keys {
                let mut set: BTreeSet<String> = now.get(key).cloned().unwrap_or_default();
                if !copy {
                    match from {
                        Some(f) => {
                            set.remove(f);
                        }
                        None => set.clear(),
                    }
                }
                set.insert(to.to_string());
                c.membership.insert(key.clone(), set.into_iter().collect());
            }
            Ok(())
        })
    }

    /// Take patients out of a cohort. A patient left in none goes to
    /// *Unsorted*.
    pub fn take_out_of_cohort(&self, keys: &[String], from: &str) -> Result<()> {
        catalog::update(self.root(), |c| {
            let now = catalog::membership(c);
            if !c.cohorts.iter().any(|k| k.id == UNSORTED) {
                c.cohorts.push(catalog::CohortRec {
                    id: UNSORTED.into(),
                    name: "Unsorted".into(),
                    folder: None,
                });
            }
            for key in keys {
                let mut set: BTreeSet<String> = now.get(key).cloned().unwrap_or_default();
                set.remove(from);
                if set.is_empty() {
                    set.insert(UNSORTED.into());
                }
                c.membership.insert(key.clone(), set.into_iter().collect());
            }
            Ok(())
        })
    }

    /// Hand every patient back to the cohorts their files lie in.
    pub fn reset_cohorts(&self, keys: &[String]) -> Result<()> {
        catalog::update(self.root(), |c| {
            for k in keys {
                c.membership.remove(k);
            }
            Ok(())
        })
    }

    // ---- moving files ------------------------------------------------------

    /// The one storage folder (by index) that holds every file of a
    /// selection, or `None` when they are spread over several.
    pub fn storage_of(&self, picks: &[Pick]) -> Option<usize> {
        let c = self.cat();
        let map = self.storage_map(&c);
        let mut found: Option<usize> = None;
        for uid in self.series_of(&c, picks) {
            for f in &c.series.get(uid)?.files {
                let r = *map.get(&f.0)?;
                if found.is_some_and(|x| x != r) {
                    return None;
                }
                found = Some(r);
            }
        }
        found
    }

    /// Every folder inside storage folder `root`, as paths below it, in
    /// order: what the window offers as places to move files into.
    pub fn folders(&self, root: usize) -> Vec<String> {
        let Some(base) = self.roots.get(root) else {
            return Vec::new();
        };
        let mut out: Vec<String> = walkdir::WalkDir::new(base)
            .min_depth(1)
            .max_depth(8)
            .into_iter()
            .filter_entry(|e| !e.file_name().to_string_lossy().starts_with('.'))
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_dir())
            .filter_map(|e| catalog::rel_of(base, e.path()))
            .collect();
        out.sort_by_key(|s| s.to_lowercase());
        out
    }

    /// Make a folder `name` in `parent` (a path below storage folder
    /// `root`, empty for the storage folder itself). Returns its path
    /// below the storage folder.
    pub fn new_folder(&self, root: usize, parent: &str, name: &str) -> Result<String> {
        let base = self.roots.get(root).context("no such storage folder")?;
        let name = name.trim();
        if !folder_name_ok(name) {
            bail!("'{name}' cannot be a folder's name");
        }
        if parent.split('/').any(|p| p == ".." || p == ".") {
            bail!("the folder is not inside the storage folder");
        }
        let rel = if parent.is_empty() {
            name.to_string()
        } else {
            format!("{parent}/{name}")
        };
        let dir = catalog::abs_of(base, &rel);
        if dir.exists() {
            bail!("{} exists already", dir.display());
        }
        std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
        Ok(rel)
    }

    /// Move the files of a selection into folder `dest` (a path below
    /// storage folder `root`), keeping the folders they had below the
    /// nearest folder holding them all: a study's series folders arrive as
    /// series folders. Only within one storage folder; refused, with
    /// nothing moved, when a file of that name is already there. Folders
    /// left empty are removed.
    pub fn move_into(&self, picks: &[Pick], root: usize, dest: &str) -> Result<MoveSummary> {
        let base = self
            .roots
            .get(root)
            .context("no such storage folder")?
            .clone();
        if dest.split('/').any(|p| p == ".." || p == ".") {
            bail!("the folder is not inside the storage folder");
        }
        let dest_dir = if dest.is_empty() {
            base.clone()
        } else {
            catalog::abs_of(&base, dest)
        };
        if self.storage_of(picks) != Some(root) {
            bail!(
                "the selection is not all in {}: files move only within one storage folder",
                base.display()
            );
        }
        let c = self.cat();
        let map = self.storage_map(&c);
        let mut files: Vec<(u32, String, PathBuf)> = Vec::new();
        for uid in self.series_of(&c, picks) {
            for f in &c.series[uid].files {
                if map.get(&f.0) == Some(&root) {
                    files.push((f.0, f.1.clone(), catalog::abs_of(&base, &f.1)));
                }
            }
        }
        if files.is_empty() {
            bail!("nothing to move");
        }
        let from = common_dir(files.iter().map(|(_, _, p)| p.as_path()));
        // Where each file goes, all checked before the first one moves.
        let mut plan: Vec<(u32, String, PathBuf, PathBuf)> = Vec::new();
        let mut sum = MoveSummary::default();
        let mut targets = HashSet::new();
        for (r, rel, path) in files {
            let below = path.strip_prefix(&from).unwrap_or(&path).to_path_buf();
            let target = dest_dir.join(below);
            if target == path {
                sum.unchanged += 1;
                continue;
            }
            if target.exists() || !targets.insert(target.clone()) {
                bail!("{} is there already; nothing was moved", target.display());
            }
            plan.push((r, rel, path, target));
        }
        let mut moved: Vec<(u32, String, String)> = Vec::new();
        let mut left: BTreeSet<PathBuf> = BTreeSet::new();
        let result = (|| -> Result<()> {
            for (r, rel, path, target) in &plan {
                if let Some(d) = target.parent() {
                    std::fs::create_dir_all(d)
                        .with_context(|| format!("create {}", d.display()))?;
                }
                if std::fs::rename(path, target).is_err() {
                    std::fs::copy(path, target)
                        .with_context(|| format!("copy {}", path.display()))?;
                    std::fs::remove_file(path)
                        .with_context(|| format!("remove {}", path.display()))?;
                }
                if let Some(new_rel) = catalog::rel_of(&base, target) {
                    moved.push((*r, rel.clone(), new_rel));
                }
                if let Some(d) = path.parent() {
                    left.insert(d.to_path_buf());
                }
            }
            Ok(())
        })();
        // Whatever moved is put right in the catalog, even after a failure
        // half way.
        sum.moved = moved.len();
        let renamed: HashMap<(u32, String), String> = moved
            .into_iter()
            .map(|(r, old, new)| ((r, old), new))
            .collect();
        catalog::update(self.root(), |c| {
            for s in c.series.values_mut() {
                for f in s.files.iter_mut() {
                    if let Some(new) = renamed.get(&(f.0, f.1.clone())) {
                        f.1 = new.clone();
                    }
                }
            }
            // A folder that now holds several patients is a cohort.
            catalog::ensure_cohorts(c);
            Ok(())
        })?;
        result?;
        for d in left.iter().rev() {
            sum.folders_removed += prune_empty(d, &base, std::slice::from_ref(&dest_dir));
        }
        Ok(sum)
    }

    /// Make `new_main` the main folder of this PACS: the catalog (with the
    /// cohorts and names given) is copied there unless one is there
    /// already. The settings are the caller's.
    pub fn carry_catalog_to(&self, new_main: &Path) -> Result<()> {
        let from = self.catalog_path();
        let to = catalog::path_in(new_main);
        if from.is_file() && !to.exists() && from != to {
            std::fs::create_dir_all(new_main)
                .with_context(|| format!("create {}", new_main.display()))?;
            std::fs::copy(&from, &to).with_context(|| format!("write {}", to.display()))?;
        }
        Ok(())
    }

    /// Forget a storage folder that left the list: its files stay on the
    /// disk, the catalog no longer lists them.
    pub fn forget_storage(&self, folder: &Path) -> Result<()> {
        if !catalog::exists(self.root()) {
            return Ok(());
        }
        catalog::update(self.root(), |c| {
            let Some(i) = c
                .storage
                .iter()
                .position(|s| catalog::same_folder(folder, s))
            else {
                return Ok(());
            };
            let i = i as u32;
            for s in c.series.values_mut() {
                s.files.retain(|f| f.0 != i);
            }
            c.other.retain(|o| o.0 != i);
            catalog::prune(c);
            catalog::ensure_cohorts(c);
            Ok(())
        })
    }
}

/// Files folders into the main storage folder of an [`Archive`] and brings
/// the catalog up to date once, at [`Importer::finish`].
///
/// The headers of a folder's files are read side by side (in file order,
/// so the filing is the same on any number of threads), then the files are
/// filed one by one. What the PACS holds already, by SOP Instance UID, is a
/// duplicate and left alone. A file that sits where the PACS files it (its
/// study's folder, its UID as its name) but is missing from the catalog (a
/// filing that broke off before the catalog was written) is counted as a
/// duplicate and put back into the catalog.
pub struct Importer<'a> {
    archive: &'a Archive,
    /// The catalog as the import started.
    cat: Arc<Catalog>,
    map: HashMap<u32, usize>,
    /// The SOP Instance UIDs held, those filed by this import included.
    held: HashSet<String>,
    /// Where each study's new files go, resolved once.
    homes: HashMap<String, PathBuf>,
    /// What the catalog has to learn at the end.
    filed: Vec<(Head, PathBuf, u64)>,
    take_files: bool,
}

impl Importer<'_> {
    /// Move the files into the PACS instead of copying them, where a rename
    /// can (the same file system): for a folder the caller made to be filed
    /// and throws away afterwards, never for a person's own files.
    pub fn taking_files(mut self) -> Self {
        self.take_files = true;
        self
    }

    /// File every DICOM file under `src`. Returns what happened and the
    /// header of every DICOM file found, filed or a duplicate, in file
    /// order: what arrived, by the files' own UIDs.
    pub fn file_folder(
        &mut self,
        src: &Path,
        progress: &Progress,
    ) -> Result<(ImportSummary, Vec<Head>)> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        progress.set("Scanning the folder");
        let files: Vec<PathBuf> = walkdir::WalkDir::new(src)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .collect();
        let total = files.len();
        let done = AtomicUsize::new(0);
        let heads: Vec<Option<Head>> = files
            .par_iter()
            .map(|path| {
                if progress.cancelled() {
                    return None;
                }
                let n = done.fetch_add(1, Ordering::Relaxed);
                if n.is_multiple_of(25) {
                    progress.set(format!("Reading {}/{total}", n + 1));
                }
                catalog::read_head(path)
            })
            .collect();
        if progress.cancelled() {
            bail!(crate::progress::CANCELLED);
        }
        let mut sum = ImportSummary::default();
        let mut arrived: Vec<Head> = Vec::new();
        let mut patients: BTreeSet<String> = BTreeSet::new();
        let mut studies: BTreeSet<String> = BTreeSet::new();
        for (n, (path, h)) in files.iter().zip(heads).enumerate() {
            if n % 25 == 0 {
                progress.set(format!("Filing {}/{total}", n + 1));
            }
            if progress.cancelled() {
                bail!(crate::progress::CANCELLED);
            }
            let Some(h) = h else {
                sum.skipped += 1;
                continue;
            };
            studies.insert(h.study.clone());
            patients.insert(h.patient_key());
            if self.held.contains(&h.sop) {
                sum.duplicates += 1;
                arrived.push(h);
                continue;
            }
            let home = self.home_of(&h);
            let dest = home.join(format!("{}.dcm", sanitize(&h.sop)));
            if dest.exists() {
                // Filed before, and the catalog never heard of it.
                sum.duplicates += 1;
                let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
                self.held.insert(h.sop.clone());
                self.filed.push((h.clone(), dest, bytes));
                arrived.push(h);
                continue;
            }
            std::fs::create_dir_all(&home).with_context(|| format!("create {}", home.display()))?;
            // Put beside its place and renamed into it, so that a reader (a
            // PACS server sending the study on) never takes half a file.
            let part = dest.with_extension("dcm.tmp");
            let moved = self.take_files && std::fs::rename(path, &part).is_ok();
            if !moved {
                std::fs::copy(path, &part)
                    .with_context(|| format!("copy {} into the PACS", path.display()))?;
            }
            if let Err(e) = std::fs::rename(&part, &dest) {
                // A file taken from its folder goes back there rather than
                // stay half filed where neither the folder nor the catalog
                // knows it; a copy is simply removed.
                if moved {
                    let _ = std::fs::rename(&part, path);
                } else {
                    let _ = std::fs::remove_file(&part);
                }
                return Err(e).with_context(|| format!("file {} into the PACS", path.display()));
            }
            let bytes = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            self.held.insert(h.sop.clone());
            self.filed.push((h.clone(), dest, bytes));
            arrived.push(h);
            sum.stored += 1;
        }
        sum.studies = studies.len();
        sum.patients = patients.len();
        Ok((sum, arrived))
    }

    /// The folder a study's new files go into: beside the study's files in
    /// the main folder, if it has any there.
    fn home_of(&mut self, h: &Head) -> PathBuf {
        if let Some(home) = self.homes.get(&h.study) {
            return home.clone();
        }
        let main_files: Vec<PathBuf> = self
            .cat
            .series
            .values()
            .filter(|s| s.study == h.study)
            .flat_map(|s| s.files.iter())
            .filter(|f| self.map.get(&f.0) == Some(&0))
            .filter_map(|f| self.archive.path_of(&self.map, f))
            .collect();
        let home = if main_files.is_empty() {
            self.archive.new_study_dir(&h.patient_key(), &h.study)
        } else {
            common_dir(main_files.iter().map(PathBuf::as_path))
        };
        self.homes.insert(h.study.clone(), home.clone());
        home
    }

    /// Bring the catalog up to date with everything filed: one update for
    /// the whole import.
    pub fn finish(self) -> Result<()> {
        if self.filed.is_empty() {
            return Ok(());
        }
        let main = self.archive.root().to_path_buf();
        let filed = self.filed;
        catalog::update(&main.clone(), move |c| {
            let r = storage_index(c, &main);
            for (h, dest, bytes) in &filed {
                let Some(rel) = catalog::rel_of(&main, dest) else {
                    continue;
                };
                // Put back into the catalog what it lists already (two
                // imports filing the same file) only once.
                if c.series
                    .get(&h.series)
                    .is_some_and(|s| s.files.iter().any(|f| f.0 == r && f.1 == rel))
                {
                    continue;
                }
                let mtime = std::fs::metadata(dest)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                catalog::add_file(c, h, FileRec(r, rel, *bytes, mtime, h.sop.clone()));
            }
            catalog::prune(c);
            catalog::ensure_cohorts(c);
            Ok(())
        })
    }
}

/// The catalog's index of a storage folder, added to its list when new.
fn storage_index(c: &mut Catalog, root: &Path) -> u32 {
    match c.storage.iter().position(|s| catalog::same_folder(root, s)) {
        Some(i) => i as u32,
        None => {
            c.storage.push(root.display().to_string());
            (c.storage.len() - 1) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folder_names_survive_whatever_an_acquiring_system_wrote() {
        assert_eq!(sanitize("P0001"), "P0001");
        assert_eq!(sanitize(" Doe^John / 3 "), "Doe_John___3");
        assert_eq!(sanitize("1.2.840.113619.2"), "1.2.840.113619.2");
        assert_eq!(sanitize(""), "unknown");
        assert_eq!(sanitize("___"), "unknown", "nothing usable is left");
        assert_eq!(sanitize(&"x".repeat(200)).len(), 96, "capped");
    }

    #[test]
    fn a_missing_root_is_an_empty_archive_rather_than_an_error() {
        let dir = std::env::temp_dir().join("rds_archive_missing_root");
        let _ = std::fs::remove_dir_all(&dir);
        let a = Archive::new(&dir);
        assert_eq!(a.scan().expect("scan succeeds").len(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uids_from_the_network_are_checked() {
        assert!(is_uid("1.2.840.10008.5.1.4.1.1.2"));
        assert!(!is_uid(""));
        assert!(!is_uid("1..2"));
        assert!(!is_uid(".1"));
        assert!(!is_uid("1.2/../3"));
        assert!(!is_uid(&"1".repeat(65)));
    }

    #[test]
    fn the_nearest_common_folder_is_found() {
        let p = |s: &str| PathBuf::from(s);
        let files = [p("/a/b/c/1.dcm"), p("/a/b/d/2.dcm"), p("/a/b/3.dcm")];
        assert_eq!(common_dir(files.iter().map(PathBuf::as_path)), p("/a/b"));
        let one = [p("/a/b/c/1.dcm")];
        assert_eq!(common_dir(one.iter().map(PathBuf::as_path)), p("/a/b/c"));
        assert_eq!(show_date("20260827"), "2026-08-27");
        assert_eq!(show_date("2026"), "2026");
    }

    #[test]
    fn folder_names_are_checked_before_one_is_made() {
        assert!(folder_name_ok("Trial 2026"));
        assert!(!folder_name_ok(""));
        assert!(!folder_name_ok(".."));
        assert!(!folder_name_ok("a/b"));
        assert!(!folder_name_ok("a\\b"));
        assert!(!folder_name_ok(".hidden"));
        assert!(!folder_name_ok("a:b"));
    }

    /// A path handed back by a stale listing must never delete anything
    /// outside the PACS.
    #[test]
    fn remove_refuses_to_step_outside_the_archive() {
        let root = std::env::temp_dir().join("rds_archive_guard");
        std::fs::create_dir_all(root.join("P1")).unwrap();
        let a = Archive::new(&root);
        assert!(a.remove(&root).is_err(), "the root itself is refused");
        assert!(
            a.remove(&std::env::temp_dir()).is_err(),
            "a folder outside is refused"
        );
        assert!(a.remove(&root.join("P1")).is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }
}
