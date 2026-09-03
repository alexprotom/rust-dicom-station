//! The local patient archive - the application's own store of DICOM studies.
//!
//! A small PACS in the sense that matters day to day: every study ever
//! imported is filed under its patient, listed without opening a single
//! DICOM file, loaded into a viewer dataset on demand, and given back the
//! contours and segmentations drawn on it.
//!
//! ## Layout
//!
//! ```text
//! <root>/
//!   <patient>/            PATIENT.txt   name, id
//!     <study uid>/        STUDY.txt     uid, date, description, modalities, files
//!       <sop uid>.dcm
//! ```
//!
//! Folder names are the DICOM UIDs, which are digits and dots and therefore
//! already safe; only the patient folder is derived from free text and needs
//! sanitizing.
//!
//! ## Why the sidecars
//!
//! Listing the archive must stay instant however large it grows, and reading
//! headers out of ten thousand files is not instant. Each study folder
//! therefore carries a `STUDY.txt` written when anything is filed into it,
//! in the same `key = value` shape as the settings file. A folder that
//! arrived without one - copied in by hand - gets it rebuilt from the
//! headers once, and is fast from then on.
//!
//! The sidecars are a cache, never the truth: the `.dcm` files are, and the
//! archive can always be rebuilt from them.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dicom_dictionary_std::tags;

use crate::loader::str_of;
use crate::progress::Progress;
use crate::settings;

const PATIENT_FILE: &str = "PATIENT.txt";
const STUDY_FILE: &str = "STUDY.txt";

/// One study of the archive, as its sidecar describes it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StudyEntry {
    pub study_uid: String,
    pub date: String,
    pub description: String,
    /// Every modality present in the study, sorted.
    pub modalities: Vec<String>,
    pub files: usize,
    pub dir: PathBuf,
}

impl StudyEntry {
    /// `20260827 - Planning · CT, RTSTRUCT · 214 files`.
    pub fn describe(&self) -> String {
        format!(
            "{}{} · {} · {} file{}",
            if self.date.is_empty() {
                "undated".into()
            } else {
                self.date.clone()
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
            self.files,
            if self.files == 1 { "" } else { "s" }
        )
    }
}

/// One patient of the archive and the studies filed under them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PatientEntry {
    pub name: String,
    pub id: String,
    pub dir: PathBuf,
    pub studies: Vec<StudyEntry>,
}

impl PatientEntry {
    /// What the list calls them: `Doe John (P0001)`.
    pub fn title(&self) -> String {
        let name = self.name.replace('^', " ");
        match (name.is_empty(), self.id.is_empty()) {
            (true, true) => "Unknown patient".into(),
            (true, false) => format!("Patient {}", self.id),
            (false, true) => name,
            (false, false) => format!("{name} ({})", self.id),
        }
    }

    pub fn files(&self) -> usize {
        self.studies.iter().map(|s| s.files).sum()
    }
}

/// What an import did, for the line the window reports afterwards.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImportSummary {
    pub stored: usize,
    /// Files already in the archive under the same SOP Instance UID.
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

/// Default archive root: `<platform data dir>/archive`.
pub fn default_root() -> PathBuf {
    settings::data_dir().join("archive")
}

/// The root a settings value names, falling back to [`default_root`] when it
/// is blank.
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
/// conservative set becomes `_`, which can map two identifiers onto one
/// folder; that merges two patients who already share an identifier, which
/// is the correct reading, and is the reason the folder name is never the
/// authority - `PATIENT.txt` is.
fn sanitize(s: &str) -> String {
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

/// Read a `key = value` sidecar.
fn read_sidecar(path: &Path) -> Vec<(String, String)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.trim().to_lowercase(), v.trim().to_string()))
        .collect()
}

fn field(pairs: &[(String, String)], key: &str) -> String {
    pairs
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.clone())
        .unwrap_or_default()
}

/// The application's store of DICOM studies, rooted at one folder.
pub struct Archive {
    root: PathBuf,
}

impl Archive {
    pub fn new(root: impl Into<PathBuf>) -> Archive {
        Archive { root: root.into() }
    }

    /// Does the archive hold anything at all?
    ///
    /// The start screen asks this to decide whether to offer *Load data from
    /// PACS*, so it must be cheap: one directory read that stops at the
    /// first patient folder, not the full [`Archive::scan`].
    pub fn has_patients(&self) -> bool {
        std::fs::read_dir(&self.root)
            .map(|entries| {
                entries
                    .flatten()
                    .any(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            })
            .unwrap_or(false)
    }

    /// Every patient in the archive, with their studies, in name order.
    ///
    /// Reads sidecars only; a study folder without one has it rebuilt from
    /// the headers first, so a folder dropped in by hand costs that once.
    pub fn scan(&self) -> Result<Vec<PatientEntry>> {
        let mut out = Vec::new();
        let Ok(dirs) = std::fs::read_dir(&self.root) else {
            // A root that does not exist yet is an empty archive, not a
            // failure - it is created on the first import.
            return Ok(out);
        };
        for pd in dirs.filter_map(|e| e.ok()) {
            if !pd.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let pdir = pd.path();
            let p = read_sidecar(&pdir.join(PATIENT_FILE));
            let mut patient = PatientEntry {
                name: field(&p, "name"),
                id: field(&p, "id"),
                dir: pdir.clone(),
                studies: Vec::new(),
            };
            let Ok(sdirs) = std::fs::read_dir(&pdir) else {
                continue;
            };
            for sd in sdirs.filter_map(|e| e.ok()) {
                if !sd.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let sdir = sd.path();
                let card = sdir.join(STUDY_FILE);
                if !card.exists() {
                    let _ = self.rebuild_sidecars(&sdir);
                }
                let s = read_sidecar(&card);
                patient.studies.push(StudyEntry {
                    study_uid: field(&s, "uid"),
                    date: field(&s, "date"),
                    description: field(&s, "description"),
                    modalities: field(&s, "modalities")
                        .split(',')
                        .map(|m| m.trim().to_string())
                        .filter(|m| !m.is_empty())
                        .collect(),
                    files: field(&s, "files").parse().unwrap_or(0),
                    dir: sdir,
                });
            }
            if patient.studies.is_empty() {
                continue;
            }
            // Newest study first - what one is normally after.
            patient.studies.sort_by(|a, b| b.date.cmp(&a.date));
            out.push(patient);
        }
        out.sort_by_key(|a| a.title().to_lowercase());
        Ok(out)
    }

    /// Rebuild a study folder's sidecar (and its patient's) from the headers
    /// of the files in it.
    fn rebuild_sidecars(&self, study_dir: &Path) -> Result<()> {
        let mut modalities: BTreeSet<String> = BTreeSet::new();
        let mut files = 0usize;
        let (mut uid, mut date, mut desc) = (String::new(), String::new(), String::new());
        let (mut pname, mut pid) = (String::new(), String::new());
        for f in std::fs::read_dir(study_dir)?.filter_map(|e| e.ok()) {
            if !f.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let Ok(obj) = crate::dicomfile::open_header(&f.path()) else {
                continue;
            };
            files += 1;
            if let Some(m) = str_of(&obj, tags::MODALITY) {
                modalities.insert(m);
            }
            if uid.is_empty() {
                uid = str_of(&obj, tags::STUDY_INSTANCE_UID).unwrap_or_default();
                date = str_of(&obj, tags::STUDY_DATE).unwrap_or_default();
                desc = str_of(&obj, tags::STUDY_DESCRIPTION).unwrap_or_default();
                pname = str_of(&obj, tags::PATIENT_NAME).unwrap_or_default();
                pid = str_of(&obj, tags::PATIENT_ID).unwrap_or_default();
            }
        }
        if files == 0 {
            return Ok(());
        }
        write_study_card(
            study_dir,
            &uid,
            &date,
            &desc,
            &modalities.iter().cloned().collect::<Vec<_>>(),
            files,
        )?;
        if let Some(pdir) = study_dir.parent() {
            let card = pdir.join(PATIENT_FILE);
            if !card.exists() {
                write_patient_card(pdir, &pname, &pid)?;
            }
        }
        Ok(())
    }

    /// Where a patient's study lives, creating neither.
    fn study_dir(&self, patient_key: &str, study_uid: &str) -> PathBuf {
        self.root
            .join(sanitize(patient_key))
            .join(sanitize(study_uid))
    }

    /// File every DICOM file under `src` into the archive.
    ///
    /// Files are copied, never moved: importing must not take the source
    /// folder apart. A file whose SOP Instance UID is already stored under
    /// the same study is counted as a duplicate and left alone, so importing
    /// the same folder twice is a no-op rather than a second copy.
    pub fn import(&self, src: &Path, progress: &Progress) -> Result<ImportSummary> {
        progress.set("Scanning the folder");
        let files: Vec<PathBuf> = walkdir::WalkDir::new(src)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .collect();
        let mut sum = ImportSummary::default();
        let mut touched: BTreeSet<PathBuf> = BTreeSet::new();
        let mut patients: BTreeSet<PathBuf> = BTreeSet::new();
        for (n, path) in files.iter().enumerate() {
            if n % 25 == 0 {
                progress.set(format!("Filing {}/{}", n + 1, files.len()));
            }
            let Ok(obj) = crate::dicomfile::open_header(path) else {
                sum.skipped += 1;
                continue;
            };
            let sop = str_of(&obj, tags::SOP_INSTANCE_UID).unwrap_or_default();
            let study_uid = str_of(&obj, tags::STUDY_INSTANCE_UID).unwrap_or_default();
            let pid = str_of(&obj, tags::PATIENT_ID).unwrap_or_default();
            let pname = str_of(&obj, tags::PATIENT_NAME).unwrap_or_default();
            if sop.is_empty() || study_uid.is_empty() {
                sum.skipped += 1;
                continue;
            }
            let key = if pid.is_empty() {
                pname.clone()
            } else {
                pid.clone()
            };
            let sdir = self.study_dir(&key, &study_uid);
            let dest = sdir.join(format!("{}.dcm", sanitize(&sop)));
            if dest.exists() {
                sum.duplicates += 1;
                touched.insert(sdir);
                continue;
            }
            std::fs::create_dir_all(&sdir).with_context(|| format!("create {}", sdir.display()))?;
            std::fs::copy(path, &dest)
                .with_context(|| format!("copy {} into the archive", path.display()))?;
            sum.stored += 1;
            if let Some(pdir) = sdir.parent() {
                if !pdir.join(PATIENT_FILE).exists() {
                    write_patient_card(pdir, &pname, &pid)?;
                }
                patients.insert(pdir.to_path_buf());
            }
            touched.insert(sdir);
        }
        // The sidecars are rebuilt once per touched study rather than per
        // file - the counts and modality list are only right at the end.
        progress.set("Updating the archive index");
        for sdir in &touched {
            let _ = self.rebuild_sidecars(sdir);
        }
        sum.studies = touched.len();
        sum.patients = patients.len();
        progress.set("done");
        Ok(sum)
    }

    /// Delete a study folder, or a whole patient.
    ///
    /// Refuses anything that is not inside the archive root, because the
    /// path comes from a listing that a stale rescan could have made wrong.
    pub fn remove(&self, dir: &Path) -> Result<()> {
        let root = self
            .root
            .canonicalize()
            .unwrap_or_else(|_| self.root.clone());
        let target = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        anyhow::ensure!(
            target.starts_with(&root) && target != root,
            "{} is not inside the archive",
            dir.display()
        );
        std::fs::remove_dir_all(&target).with_context(|| format!("remove {}", target.display()))
    }
}

fn write_patient_card(dir: &Path, name: &str, id: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(
        dir.join(PATIENT_FILE),
        format!("name = {name}\nid = {id}\n"),
    )
    .with_context(|| format!("write {}", dir.join(PATIENT_FILE).display()))
}

fn write_study_card(
    dir: &Path,
    uid: &str,
    date: &str,
    description: &str,
    modalities: &[String],
    files: usize,
) -> Result<()> {
    std::fs::write(
        dir.join(STUDY_FILE),
        format!(
            "uid = {uid}\ndate = {date}\ndescription = {description}\n\
             modalities = {}\nfiles = {files}\n",
            modalities.join(",")
        ),
    )
    .with_context(|| format!("write {}", dir.join(STUDY_FILE).display()))
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
    }

    #[test]
    fn a_study_reads_back_from_its_sidecar() {
        let root = std::env::temp_dir().join("rds_archive_sidecar");
        let _ = std::fs::remove_dir_all(&root);
        let sdir = root.join("P1").join("1.2.3");
        std::fs::create_dir_all(&sdir).unwrap();
        write_patient_card(sdir.parent().unwrap(), "Doe^John", "P1").unwrap();
        write_study_card(
            &sdir,
            "1.2.3",
            "20260827",
            "Planning",
            &["CT".into(), "RTSTRUCT".into()],
            214,
        )
        .unwrap();

        let found = Archive::new(&root).scan().unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].title(), "Doe John (P1)");
        assert_eq!(found[0].files(), 214);
        let st = &found[0].studies[0];
        assert_eq!(st.study_uid, "1.2.3");
        assert_eq!(st.modalities, vec!["CT", "RTSTRUCT"]);
        assert!(st
            .describe()
            .starts_with("20260827 - Planning · CT, RTSTRUCT · 214 files"));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A path handed back by a stale listing must never delete anything
    /// outside the archive.
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
