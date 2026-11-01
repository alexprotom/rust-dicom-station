//! Sorting files into the archive's layout and names, by a rules file.
//!
//! The layout is `cohort / patient / study / series / file`, every name
//! made from the files' own headers:
//!
//! * study: `<date>_<Upright|Supine>-<note>_StudyUID-<last 12 digits>`.
//!   The date is the earliest StudyDate of the study's files (a course study
//!   spans weeks, one StudyDate a session), else the earliest SeriesDate or
//!   AcquisitionDate, else `NoDate`. The note is the study's entry in
//!   `study_notes`, else the earliest session's StudyDescription, else the
//!   note of the folder it came from, else `Procedure#<n>` (Upright, `n`
//!   counting the patient's undescribed upright studies by date) or
//!   `CT-Simulation` (a Supine study with a CT) or `No-description`.
//!   Upright or Supine is decided by the scanner of its imaging files
//!   (`upright_scanners`, matched in Manufacturer, ManufacturerModelName and
//!   StationName); a cohort can leave the word out (`position_keyword`).
//! * series: `Series-<number>_<Modality>_<note>_SeriesUID-<last 12>`, the
//!   note being `series_notes` (by Series UID, or by the name of a folder
//!   the files lie in), else SeriesDescription, the RT label
//!   (StructureSetLabel, RTPlanLabel, RTImageLabel; `Anonymized` does not
//!   count), `Dose-<summation type>`, ProtocolName, `<SeriesDate>-<Time>`,
//!   `No-description`.
//! * file: `<CT|MR|RS|RP|RD|RI|RE|RR|...>.<SOP Instance UID>.dcm`.
//!
//! Names follow one set of character rules ([`clean`]): `_` between the
//! parts of a name, `-` within one; `->` and `, : ; _ / \ ^` and spaces
//! become `-`; `Procedure #7` becomes `Procedure#7`; a decimal comma `3,00`
//! becomes `3.00`; ä ö ü ß become ae oe ue ss; anything else outside
//! letters, digits and `. # + ( )` goes. Notes are cut to `max_note`
//! characters, and shorter still where the path would pass 250 characters.
//!
//! The patient folder is a code: the first folder below the import is
//! mapped through the cohort's `patient_code_map`, then the
//! `patient_folders` patterns (`^patient(\d{3})$` → `P01-$1`); a file whose
//! PatientID is another patient folder's code is reported (an export can
//! carry the wrong ID). The files' PatientName and PatientID become the code,
//! unless the cohort keeps them (`keep_patient_tags`).
//!
//! A study the archive holds already (by its UID: through the catalog, or
//! the `StudyUID-` suffix of a folder) is merged into its existing folder.
//! A file whose place is taken by a file of the same content goes to the
//! quarantine; one of other content is left where it is and reported.
//! Files that are not DICOM (plan screenshots, headerless detector frames)
//! are attachments: reported, or kept beside the series they came with, or
//! moved to the records or the quarantine folder.
//!
//! [`plan`] makes the operations; the MCP tools save them as a manifest
//! ([`super::journal`]) for an apply that can be undone.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use dicom_core::Tag;
use dicom_dictionary_std::tags;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::journal::{Op, WARN_PATH};
use super::raw::RawFile;
use super::remap::{simulate, Changes, UidMaps};
use crate::progress::{Progress, CANCELLED};

/// The rules file's name, in the records folder.
pub const RULES_FILE: &str = "pacs-rules.toml";

/// How a patient folder's name becomes a code.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FolderRule {
    /// A regular expression the whole folder name must match.
    pub pattern: String,
    /// The code, `$1` and so on standing for the groups.
    pub code: String,
}

/// What differs per cohort.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CohortRules {
    /// Folder name → patient code, before the patterns.
    pub patient_code_map: BTreeMap<String, String>,
    /// Leave PatientName / PatientID as the files have them.
    pub keep_patient_tags: bool,
    /// Upright / Supine in the study folder's name (default: the rules').
    pub position_keyword: Option<bool>,
}

/// `pacs-rules.toml`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Rules {
    pub version: u32,
    /// A study whose imaging files come from one of these (Manufacturer,
    /// ManufacturerModelName, StationName contain it, case aside) is
    /// Upright.
    pub upright_scanners: Vec<String>,
    /// The modalities that decide the position.
    pub imaging_modalities: Vec<String>,
    pub position_keyword: bool,
    /// Patient folder patterns, the first that matches wins.
    pub patient_folders: Vec<FolderRule>,
    /// Study Instance UID → note.
    pub study_notes: BTreeMap<String, String>,
    /// Series Instance UID, or the name of a folder its files are in → note.
    pub series_notes: BTreeMap<String, String>,
    /// The longest note.
    pub max_note: usize,
    pub cohorts: BTreeMap<String, CohortRules>,
}

impl Default for Rules {
    fn default() -> Self {
        Rules {
            version: 1,
            upright_scanners: vec![
                "Brilliance Big Bore".into(),
                "RayStation".into(),
                "P-Cure".into(),
            ],
            imaging_modalities: [
                "CT", "MR", "PT", "CR", "DX", "RTIMAGE", "XA", "RF", "NM", "US",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            position_keyword: true,
            patient_folders: Vec::new(),
            study_notes: BTreeMap::new(),
            series_notes: BTreeMap::new(),
            max_note: 48,
            cohorts: BTreeMap::new(),
        }
    }
}

impl Rules {
    pub fn parse(text: &str) -> Result<Rules> {
        let r: Rules = toml::from_str(text).context("pacs-rules.toml")?;
        if r.version != 1 {
            bail!(
                "pacs-rules.toml is version {}, this program reads 1",
                r.version
            );
        }
        for f in &r.patient_folders {
            regex::Regex::new(&format!("^(?:{})$", f.pattern))
                .with_context(|| format!("patient_folders pattern {}", f.pattern))?;
        }
        Ok(r)
    }

    /// The rules in `records/pacs-rules.toml`, or the defaults.
    pub fn load(records: &Path) -> Result<Rules> {
        match std::fs::read_to_string(records.join(RULES_FILE)) {
            Ok(t) => Rules::parse(&t),
            Err(_) => Ok(Rules::default()),
        }
    }

    fn cohort(&self, cohort: &str) -> CohortRules {
        self.cohorts.get(cohort).cloned().unwrap_or_default()
    }

    /// The patient code of a folder name in a cohort.
    pub fn patient_code(&self, cohort: &str, folder: &str) -> String {
        if let Some(c) = self.cohort(cohort).patient_code_map.get(folder) {
            return c.clone();
        }
        for f in &self.patient_folders {
            if let Ok(re) = regex::Regex::new(&format!("^(?:{})$", f.pattern)) {
                if re.is_match(folder) {
                    return clean(&re.replace(folder, f.code.as_str()));
                }
            }
        }
        clean(folder)
    }
}

/// Two spellings of one patient code: letters and digits, lower case,
/// without a leading `p`/`patient` (`P01-039`, `01-039`, `01039`).
pub fn code_key(s: &str) -> String {
    let t: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect();
    let t = t.strip_prefix("patient").unwrap_or(&t);
    t.strip_prefix('p').unwrap_or(t).to_string()
}

/// A text made into a name by the character rules (see the module doc).
pub fn clean(s: &str) -> String {
    let mut t = s.trim().to_string();
    // A decimal comma is a point; `->` an arrow, one separator.
    let comma = regex::Regex::new(r"(\d),(\d)").unwrap();
    while comma.is_match(&t) {
        t = comma.replace_all(&t, "$1.$2").into_owned();
    }
    t = t.replace("->", "-");
    let hash = regex::Regex::new(r"\s+#").unwrap();
    t = hash.replace_all(&t, "#").into_owned();
    let mut out = String::with_capacity(t.len());
    for c in t.chars() {
        match c {
            // ä ö ü Ä Ö Ü ß
            '\u{e4}' => out.push_str("ae"),
            '\u{f6}' => out.push_str("oe"),
            '\u{fc}' => out.push_str("ue"),
            '\u{c4}' => out.push_str("Ae"),
            '\u{d6}' => out.push_str("Oe"),
            '\u{dc}' => out.push_str("Ue"),
            '\u{df}' => out.push_str("ss"),
            c if c.is_ascii_alphanumeric() || ".#+()".contains(c) => out.push(c),
            _ => out.push('-'),
        }
    }
    let dashes = regex::Regex::new(r"-{2,}").unwrap();
    let out = dashes.replace_all(&out, "-");
    out.trim_matches(|c| c == '-' || c == '.').to_string()
}

/// A note, cleaned and cut to `max` characters.
fn note(s: &str, max: usize) -> String {
    let c = clean(s);
    let cut: String = c.chars().take(max.max(1)).collect();
    cut.trim_matches(|c| c == '-' || c == '.').to_string()
}

/// The last 12 digits of a UID (its dots left out).
pub fn last12(uid: &str) -> String {
    let d: Vec<char> = uid.chars().filter(|c| c.is_ascii_digit()).collect();
    d[d.len().saturating_sub(12)..].iter().collect()
}

pub fn study_folder(date: &str, position: Option<&str>, note: &str, uid: &str) -> String {
    let date = if date.is_empty() { "NoDate" } else { date };
    match position {
        Some(p) => format!("{date}_{p}-{note}_StudyUID-{}", last12(uid)),
        None => format!("{date}_{note}_StudyUID-{}", last12(uid)),
    }
}

pub fn series_folder(number: &str, modality: &str, note: &str, uid: &str) -> String {
    let n = clean(number);
    let n = if n.is_empty() { "NA".to_string() } else { n };
    let m = clean(modality);
    let m = if m.is_empty() { "OT".to_string() } else { m };
    format!("Series-{n}_{m}_{note}_SeriesUID-{}", last12(uid))
}

/// The file name's prefix by modality.
pub fn file_prefix(modality: &str) -> String {
    match modality {
        "RTSTRUCT" => "RS".into(),
        "RTPLAN" => "RP".into(),
        "RTDOSE" => "RD".into(),
        "RTIMAGE" => "RI".into(),
        "RTRECORD" => "RR".into(),
        "REG" => "RE".into(),
        "" => "OT".into(),
        m => clean(m),
    }
}

pub fn file_name(modality: &str, sop: &str) -> String {
    format!("{}.{}.dcm", file_prefix(modality), sop)
}

/// The note of a folder named by these rules (`..._Upright-<note>_StudyUID-...`),
/// or the whole name of one that is not.
fn note_of_folder(name: &str) -> String {
    let re =
        regex::Regex::new(r"^(?:\d{8}|NoDate)_(?:(?:Upright|Supine)-)?(.+?)_StudyUID-\d{1,12}$")
            .unwrap();
    match re.captures(name) {
        Some(c) => c[1].to_string(),
        None => name.to_string(),
    }
}

/// What a file's header says that names it.
#[derive(Clone, Debug, Default)]
pub struct Naming {
    pub path: PathBuf,
    pub sop: String,
    pub modality: String,
    pub study: String,
    pub series: String,
    pub series_number: String,
    pub study_date: String,
    pub study_time: String,
    pub series_date: String,
    pub series_time: String,
    pub acquisition_date: String,
    pub study_description: String,
    pub series_description: String,
    pub protocol: String,
    pub label: String,
    pub dose_summation: String,
    /// Manufacturer, model, station, joined.
    pub scanner: String,
    /// PatientName and PatientID: compared, never reported.
    pub patient_name: String,
    pub patient_id: String,
}

fn naming_of(path: &Path) -> Option<Naming> {
    let f = RawFile::read(path).ok()?;
    if f.elements.is_empty() {
        return None;
    }
    let t = |tag: Tag| f.text(tag).unwrap_or_default();
    let sop = t(tags::SOP_INSTANCE_UID);
    let study = t(tags::STUDY_INSTANCE_UID);
    if sop.is_empty() || study.is_empty() {
        return None;
    }
    let label = [
        tags::STRUCTURE_SET_LABEL,
        tags::RT_PLAN_LABEL,
        tags::RT_IMAGE_LABEL,
        tags::CONTENT_LABEL,
    ]
    .into_iter()
    .map(t)
    .find(|l| {
        let l = l.trim().to_lowercase();
        !l.is_empty() && l != "anonymized" && l != "anonymous"
    })
    .unwrap_or_default();
    let modality = t(tags::MODALITY);
    let mut series = t(tags::SERIES_INSTANCE_UID);
    if series.is_empty() {
        series = format!("{study}~{modality}");
    }
    Some(Naming {
        path: path.to_path_buf(),
        sop,
        modality,
        study,
        series,
        series_number: t(tags::SERIES_NUMBER),
        study_date: t(tags::STUDY_DATE),
        study_time: t(tags::STUDY_TIME),
        series_date: t(tags::SERIES_DATE),
        series_time: t(tags::SERIES_TIME),
        acquisition_date: t(tags::ACQUISITION_DATE),
        study_description: t(tags::STUDY_DESCRIPTION),
        series_description: t(tags::SERIES_DESCRIPTION),
        protocol: t(tags::PROTOCOL_NAME),
        label,
        dose_summation: t(tags::DOSE_SUMMATION_TYPE),
        scanner: [
            t(tags::MANUFACTURER),
            t(tags::MANUFACTURER_MODEL_NAME),
            t(tags::STATION_NAME),
        ]
        .join(" "),
        patient_name: t(tags::PATIENT_NAME),
        patient_id: t(tags::PATIENT_ID),
    })
}

/// Upright or Supine, by the scanner of the imaging files among `files`
/// (the first few of each modality are read), as [`plan`] decides it.
pub fn position_of(files: &[PathBuf], rules: &Rules) -> &'static str {
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut heads = Vec::new();
    for f in files {
        let Some(n) = naming_of(f) else { continue };
        let c = seen.entry(n.modality.clone()).or_default();
        *c += 1;
        if *c <= 3 {
            heads.push(n);
        }
    }
    let imaging: Vec<&Naming> = heads
        .iter()
        .filter(|n| rules.imaging_modalities.contains(&n.modality))
        .collect();
    let pick: Vec<&Naming> = if imaging.is_empty() {
        heads.iter().collect()
    } else {
        imaging
    };
    let up = pick.iter().any(|n| {
        let s = n.scanner.to_lowercase();
        rules
            .upright_scanners
            .iter()
            .any(|u| !u.is_empty() && s.contains(&u.to_lowercase()))
    });
    if up {
        "Upright"
    } else {
        "Supine"
    }
}

/// What to do with the files that are not DICOM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Attachments {
    /// List them, move nothing.
    Report,
    /// Beside the series of the DICOM files in their folder (a plan's
    /// screenshots beside the plan), else in the patient's `Attachments`.
    Keep,
    /// To `records/attachments/<cohort>/<patient>/`.
    Records,
    /// To the quarantine folder.
    Quarantine,
}

/// Where to sort to, and with what.
pub struct Input<'a> {
    /// The folder to sort: its first-level folders are patients (an
    /// import, or a cohort of the archive itself).
    pub source: &'a Path,
    /// The storage folder to sort into.
    pub root: &'a Path,
    pub cohort: &'a str,
    /// One patient code for the whole source (else from its folders).
    pub patient: Option<&'a str>,
    pub quarantine: Option<&'a Path>,
    pub records: &'a Path,
    pub attachments: Attachments,
    /// The archive's study folders, by Study Instance UID (from its
    /// catalog), relative to `root`.
    pub known_studies: &'a HashMap<String, PathBuf>,
}

/// What a plan found.
#[derive(Debug, Default, Serialize)]
pub struct Plan {
    #[serde(skip)]
    pub ops: Vec<Op>,
    pub files: usize,
    pub patients: usize,
    pub studies: usize,
    pub series: usize,
    pub moves: usize,
    pub in_place: usize,
    pub identity_rewrites: usize,
    pub to_quarantine: usize,
    pub merged_studies: usize,
    pub conflicts: usize,
    pub attachments: Vec<String>,
    pub warnings: Vec<String>,
    /// Study folders, for the person to check the names: (source study
    /// UID, folder below the root).
    pub study_folders: Vec<(String, String)>,
}

/// A headerless detector frame's size, if the file's length is one:
/// `w × w × 2` bytes.
fn raw_frame(bytes: u64) -> Option<u64> {
    let px = bytes / 2;
    let w = (px as f64).sqrt().round() as u64;
    (bytes.is_multiple_of(2) && w * w == px && w >= 64).then_some(w)
}

/// Plan sorting `input.source` by `rules`.
pub fn plan(input: &Input, rules: &Rules, p: &Progress) -> Result<Plan> {
    let cohort_rules = rules.cohort(input.cohort);
    let keyword = cohort_rules
        .position_keyword
        .unwrap_or(rules.position_keyword);
    let files = super::files_under(input.source);
    p.set(format!("Reading {} headers", files.len()));
    let read: Vec<(PathBuf, Option<Naming>)> = files
        .par_iter()
        .map(|f| (f.clone(), naming_of(f)))
        .collect();
    if p.cancelled() {
        bail!(CANCELLED);
    }
    let mut out = Plan {
        files: files.len(),
        ..Default::default()
    };
    // Patient folder of each file: the first folder below the source.
    let patient_of = |f: &Path| -> Option<(String, String)> {
        let rel = f.strip_prefix(input.source).ok()?;
        let folder = match input.patient {
            Some(code) => return Some((code.to_string(), code.to_string())),
            None if rel.components().count() >= 2 => rel
                .components()
                .next()?
                .as_os_str()
                .to_string_lossy()
                .into_owned(),
            None => return None,
        };
        Some((folder.clone(), rules.patient_code(input.cohort, &folder)))
    };
    let cohort_dir = input.root.join(clean(input.cohort));
    // The patient codes of the cohort, for the wrong-ID check.
    let mut codes: BTreeMap<String, String> = BTreeMap::new();
    if let Ok(rd) = std::fs::read_dir(&cohort_dir) {
        for e in rd.flatten().filter(|e| e.path().is_dir()) {
            let n = e.file_name().to_string_lossy().into_owned();
            codes.insert(code_key(&n), n);
        }
    }
    let mut dicom: Vec<(Naming, String, String)> = Vec::new();
    let mut other: Vec<(PathBuf, Option<(String, String)>)> = Vec::new();
    for (f, n) in read {
        let who = patient_of(&f);
        match (n, who) {
            (Some(n), Some((folder, code))) => {
                codes.insert(code_key(&code), code.clone());
                dicom.push((n, folder, code));
            }
            (Some(_), None) => out.warnings.push(format!(
                "{} lies directly in the source, not in a patient folder: left alone",
                f.display()
            )),
            (None, who) => other.push((f, who)),
        }
    }
    // Files whose PatientID is another patient folder's code.
    let mut wrong: BTreeMap<(String, String), usize> = BTreeMap::new();
    for (n, folder, code) in &dicom {
        let k = code_key(&n.patient_id);
        if !k.is_empty() && k != code_key(code) {
            if let Some(other_code) = codes.get(&k) {
                *wrong
                    .entry((folder.clone(), other_code.clone()))
                    .or_default() += 1;
            }
        }
    }
    for ((folder, other_code), n) in wrong {
        out.warnings.push(format!(
            "folder {folder}: {n} file(s) carry the PatientID of patient {other_code}; \
             filed under the folder's patient - check the pixels and dates"
        ));
    }

    // Studies and series.
    let mut by_study: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, (n, _, code)) in dicom.iter().enumerate() {
        by_study
            .entry((code.clone(), n.study.clone()))
            .or_default()
            .push(i);
    }
    out.patients = by_study
        .keys()
        .map(|(c, _)| c.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    out.studies = by_study.len();
    let imaging: BTreeSet<&str> = rules
        .imaging_modalities
        .iter()
        .map(|s| s.as_str())
        .collect();
    let upright = |list: &[&Naming]| -> bool {
        let pick: Vec<&&Naming> = {
            let im: Vec<&&Naming> = list
                .iter()
                .filter(|n| imaging.contains(n.modality.as_str()))
                .collect();
            if im.is_empty() {
                list.iter().collect()
            } else {
                im
            }
        };
        pick.iter().any(|n| {
            let s = n.scanner.to_lowercase();
            rules
                .upright_scanners
                .iter()
                .any(|u| !u.is_empty() && s.contains(&u.to_lowercase()))
        })
    };
    // Per study: date, position, note (Procedure#n numbered afterwards).
    struct StudyName {
        date: String,
        position: &'static str,
        note: Option<String>,
        has_ct: bool,
        folder: Option<PathBuf>,
    }
    let earliest = |a: &str, b: &str| a.is_empty() || (!b.is_empty() && b < a);
    let mut names: BTreeMap<(String, String), StudyName> = BTreeMap::new();
    for (key, idx) in &by_study {
        let list: Vec<&Naming> = idx.iter().map(|&i| &dicom[i].0).collect();
        let pick_date = |get: &dyn Fn(&Naming) -> &str| -> String {
            let mut d = String::new();
            for n in &list {
                let v = get(n).trim();
                if v.len() >= 8 && earliest(&d, v) {
                    d = v[..8].to_string();
                }
            }
            d
        };
        let mut date = pick_date(&|n| n.study_date.as_str());
        if date.is_empty() {
            date = pick_date(&|n| n.series_date.as_str());
        }
        if date.is_empty() {
            date = pick_date(&|n| n.acquisition_date.as_str());
        }
        // The earliest session's description.
        let mut first: Option<(&str, &str, &str)> = None;
        for n in &list {
            if n.study_description.trim().is_empty() {
                continue;
            }
            let k = (
                n.study_date.as_str(),
                n.study_time.as_str(),
                n.study_description.as_str(),
            );
            let better = match first {
                None => true,
                Some(f) => (k.0.is_empty(), k.0, k.1) < (f.0.is_empty(), f.0, f.1),
            };
            if better {
                first = Some(k);
            }
        }
        let old_folder = list.iter().find_map(|n| {
            let rel = n.path.strip_prefix(input.source).ok()?;
            let parts: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            // patient / study / series / file
            (parts.len() >= 4 && input.patient.is_none())
                .then(|| parts[1].clone())
                .or_else(|| (parts.len() >= 3 && input.patient.is_some()).then(|| parts[0].clone()))
        });
        let note = rules
            .study_notes
            .get(&key.1)
            .cloned()
            .or_else(|| first.map(|f| f.2.to_string()))
            .or_else(|| old_folder.map(|f| note_of_folder(&f)))
            .map(|s| note(&s, rules.max_note))
            .filter(|s| !s.is_empty());
        names.insert(
            key.clone(),
            StudyName {
                date,
                position: if upright(&list) { "Upright" } else { "Supine" },
                note,
                has_ct: list.iter().any(|n| n.modality == "CT"),
                folder: input.known_studies.get(&key.1).cloned(),
            },
        );
    }
    // Procedure#n: the patient's undescribed upright studies, by date.
    let mut counters: BTreeMap<&str, usize> = BTreeMap::new();
    let mut order: Vec<&(String, String)> = names.keys().collect();
    order.sort_by_key(|k| (&k.0, names[*k].date.is_empty(), &names[*k].date, &k.1));
    let mut fallback: BTreeMap<(String, String), String> = BTreeMap::new();
    for k in order {
        let s = &names[k];
        if s.note.is_some() {
            continue;
        }
        let f = if s.position == "Upright" {
            let c = counters.entry(k.0.as_str()).or_default();
            *c += 1;
            format!("Procedure#{c}")
        } else if s.has_ct {
            "CT-Simulation".to_string()
        } else {
            "No-description".to_string()
        };
        fallback.insert(k.clone(), f);
    }

    // The archive's folders by UID suffix, where the catalog said nothing.
    let mut suffixes: HashMap<(String, String), PathBuf> = HashMap::new();
    for code in by_study.keys().map(|k| &k.0).collect::<BTreeSet<_>>() {
        let pdir = cohort_dir.join(code);
        if let Ok(rd) = std::fs::read_dir(&pdir) {
            for e in rd.flatten().filter(|e| e.path().is_dir()) {
                let n = e.file_name().to_string_lossy().into_owned();
                if let Some((_, tail)) = n.rsplit_once("_StudyUID-") {
                    suffixes.insert(
                        (code.clone(), tail.to_string()),
                        e.path().strip_prefix(input.root).unwrap().to_path_buf(),
                    );
                }
            }
        }
    }

    // Each file's place.
    let mut taken: HashMap<PathBuf, PathBuf> = HashMap::new();
    let mut series_seen: BTreeSet<(String, String)> = BTreeSet::new();
    let quarantine_to = |f: &Path| -> Option<PathBuf> {
        let q = input.quarantine?;
        let rel = f.strip_prefix(input.source).ok()?;
        Some(q.join(clean(input.cohort)).join(rel))
    };
    let mut identity: Vec<(PathBuf, String)> = Vec::new();
    // Series folders by series, so attachments can follow them.
    let mut series_dir: HashMap<PathBuf, PathBuf> = HashMap::new();
    for (key, idx) in &by_study {
        let s = &names[key];
        let (code, study_uid) = key;
        let study_rel = match s
            .folder
            .clone()
            .or_else(|| suffixes.get(&(code.clone(), last12(study_uid))).cloned())
        {
            Some(existing) => {
                out.merged_studies += 1;
                existing
            }
            None => {
                let n = s.note.clone().unwrap_or_else(|| fallback[key].clone());
                PathBuf::from(clean(input.cohort))
                    .join(code)
                    .join(study_folder(
                        &s.date,
                        keyword.then_some(s.position),
                        &n,
                        study_uid,
                    ))
            }
        };
        out.study_folders.push((
            study_uid.clone(),
            study_rel.to_string_lossy().replace('\\', "/"),
        ));
        for &i in idx {
            let (n, _, code) = &dicom[i];
            let folder_note = n.path.strip_prefix(input.source).ok().and_then(|rel| {
                rel.parent()?.components().find_map(|c| {
                    rules
                        .series_notes
                        .get(c.as_os_str().to_string_lossy().as_ref())
                        .cloned()
                })
            });
            let raw_note = rules
                .series_notes
                .get(&n.series)
                .cloned()
                .or(folder_note)
                .or_else(|| Some(n.series_description.clone()).filter(|s| !s.trim().is_empty()))
                .or_else(|| Some(n.label.clone()).filter(|s| !s.trim().is_empty()))
                .or_else(|| {
                    (n.modality == "RTDOSE" && !n.dose_summation.trim().is_empty())
                        .then(|| format!("Dose-{}", n.dose_summation.trim()))
                })
                .or_else(|| Some(n.protocol.clone()).filter(|s| !s.trim().is_empty()))
                .or_else(|| {
                    (!n.series_date.trim().is_empty()).then(|| {
                        format!(
                            "{}-{}",
                            n.series_date.trim(),
                            n.series_time.trim().split('.').next().unwrap_or("")
                        )
                    })
                })
                .unwrap_or_else(|| "No-description".into());
            let mut snote = note(&raw_note, rules.max_note);
            if snote.is_empty() {
                snote = "No-description".into();
            }
            let fname = file_name(&n.modality, &n.sop);
            let mut sdir = study_rel.join(series_folder(
                &n.series_number,
                &n.modality,
                &snote,
                &n.series,
            ));
            // A path too long: the series note first gives way.
            let full = |d: &Path| crate::pathlen::len(&input.root.join(d).join(&fname));
            while full(&sdir) > WARN_PATH && snote.len() > 8 {
                snote = note(&snote[..snote.len() - 4], rules.max_note);
                sdir = study_rel.join(series_folder(
                    &n.series_number,
                    &n.modality,
                    &snote,
                    &n.series,
                ));
            }
            series_seen.insert((study_uid.clone(), n.series.clone()));
            if let Some(dir) = n.path.parent() {
                series_dir
                    .entry(dir.to_path_buf())
                    .or_insert_with(|| input.root.join(&sdir));
            }
            let to = input.root.join(&sdir).join(&fname);
            let keep_tags = cohort_rules.keep_patient_tags;
            if !keep_tags && (n.patient_name != *code || n.patient_id != *code) {
                identity.push((n.path.clone(), code.clone()));
            }
            if to == n.path {
                out.in_place += 1;
                continue;
            }
            // The place taken: by an archive file, or another source file.
            let occupant = if to.exists() {
                Some(to.clone())
            } else {
                taken.get(&to).cloned()
            };
            if let Some(other) = occupant {
                let ignore = super::dupes::default_ignored();
                if super::dupes::same_content(&n.path, &other, &ignore).unwrap_or(false) {
                    match quarantine_to(&n.path) {
                        Some(q) => {
                            out.to_quarantine += 1;
                            identity.retain(|(p, _)| *p != n.path);
                            out.ops.push(Op::Move {
                                from: n.path.clone(),
                                to: q,
                            });
                        }
                        None => out.warnings.push(format!(
                            "{} is a copy of {}; no quarantine folder to move it to",
                            n.path.display(),
                            other.display()
                        )),
                    }
                } else {
                    out.conflicts += 1;
                    identity.retain(|(p, _)| *p != n.path);
                    out.warnings.push(format!(
                        "{} would take the place of {}, which holds other content: left alone",
                        n.path.display(),
                        other.display()
                    ));
                }
                continue;
            }
            taken.insert(to.clone(), n.path.clone());
            out.moves += 1;
            out.ops.push(Op::Move {
                from: n.path.clone(),
                to,
            });
        }
    }
    out.series = series_seen.len();

    // The identity rewrites go first, in place, then the moves.
    if !identity.is_empty() {
        p.set(format!("Checking {} identity rewrites", identity.len()));
        let rewrites: Vec<Result<Option<Op>>> = identity
            .par_iter()
            .map(|(path, code)| {
                let changes = Changes {
                    set: vec![
                        ("00100010".into(), code.clone()),
                        ("00100020".into(), code.clone()),
                    ],
                    ..Default::default()
                };
                let s = simulate(path, &changes, &UidMaps::default())
                    .with_context(|| format!("{} cannot be rewritten", path.display()))?;
                Ok((s.changed > 0).then(|| Op::Rewrite {
                    path: path.clone(),
                    changes,
                    source: Some(s.source),
                    expect: Some(s.expect),
                }))
            })
            .collect();
        let mut first = Vec::new();
        for r in rewrites {
            match r {
                Ok(Some(op)) => first.push(op),
                Ok(None) => {}
                Err(e) => out
                    .warnings
                    .push(format!("{e:#}: its tags are left as they are")),
            }
        }
        out.identity_rewrites = first.len();
        first.append(&mut out.ops);
        out.ops = first;
    }

    // Attachments.
    for (f, who) in other {
        let bytes = std::fs::metadata(&f).map(|m| m.len()).unwrap_or(0);
        let what = match raw_frame(bytes) {
            Some(w) => format!("{} (headerless {w}x{w} 16-bit frame?)", f.display()),
            None => f.display().to_string(),
        };
        let code = who.map(|w| w.1);
        let dest = match input.attachments {
            Attachments::Report => None,
            Attachments::Keep => f
                .parent()
                .and_then(|d| series_dir.get(d).cloned())
                .or_else(|| {
                    code.as_ref()
                        .map(|c| cohort_dir.join(c).join("Attachments"))
                })
                .and_then(|d| Some(d.join(f.file_name()?))),
            Attachments::Records => {
                let rel = f.strip_prefix(input.source).ok().map(Path::to_path_buf);
                rel.map(|r| {
                    input
                        .records
                        .join("attachments")
                        .join(clean(input.cohort))
                        .join(r)
                })
            }
            Attachments::Quarantine => quarantine_to(&f),
        };
        match dest {
            Some(to) if to != f && !to.exists() => {
                out.attachments.push(format!("{what} -> {}", to.display()));
                out.ops.push(Op::Move { from: f, to });
            }
            _ => out.attachments.push(what),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The names of §3.5 of the curation plan, as golden strings.
    #[test]
    fn names_follow_the_character_rules() {
        assert_eq!(clean("Procedure #7"), "Procedure#7");
        assert_eq!(clean("Lung 3,00 mm"), "Lung-3.00-mm");
        assert_eq!(clean("CT->MR reg"), "CT-MR-reg");
        assert_eq!(clean("a,b:c;d_e/f\\g^h"), "a-b-c-d-e-f-g-h");
        assert_eq!(
            clean("Kopf Halsbestrahlung f\u{fc}r Gr\u{f6}\u{df}e"),
            "Kopf-Halsbestrahlung-fuer-Groesse"
        );
        assert_eq!(clean("  -x--y-  "), "x-y");
        assert_eq!(clean("Thorax <i/v> \"KM\""), "Thorax-i-v-KM");
        assert_eq!(
            last12("1.2.826.0.1.3680043.8.498.711174472480"),
            "711174472480"
        );
        assert_eq!(last12("1.2.3"), "123");
        assert_eq!(
            study_folder(
                "20240716",
                Some("Upright"),
                "Procedure#4",
                "1.2.3.4.5.6.7.8.9.10.11.12.13"
            ),
            "20240716_Upright-Procedure#4_StudyUID-678910111213"
        );
        assert_eq!(
            study_folder("", None, "CT-Simulation", "9.87654321098765"),
            "NoDate_CT-Simulation_StudyUID-654321098765"
        );
        assert_eq!(
            series_folder(
                "221",
                "CT",
                "4D-Phase-50",
                "1.2.840.113619.2.55.1234567890.123"
            ),
            "Series-221_CT_4D-Phase-50_SeriesUID-234567890123"
        );
        assert_eq!(
            series_folder("", "RTSTRUCT", "No-description", "1.1"),
            "Series-NA_RTSTRUCT_No-description_SeriesUID-11"
        );
        assert_eq!(file_name("RTSTRUCT", "1.2.3"), "RS.1.2.3.dcm");
        assert_eq!(file_name("CT", "1.2.3"), "CT.1.2.3.dcm");
        assert_eq!(file_name("REG", "1.2"), "RE.1.2.dcm");
        assert_eq!(
            note_of_folder("20240716_Upright-Procedure#4_StudyUID-711174472480"),
            "Procedure#4"
        );
        assert_eq!(note_of_folder("TPS plan 1"), "TPS plan 1");
    }

    #[test]
    fn patient_folders_map_to_codes() {
        let r = Rules::parse(
            r#"
version = 1
[[patient_folders]]
pattern = 'patient(\d{3})'
code = 'P01-$1'
[[patient_folders]]
pattern = '(\d{2}-\d{3})(_.*)?'
code = 'P$1'
[cohorts.Trial.patient_code_map]
"special" = "P01-099"
"#,
        )
        .unwrap();
        assert_eq!(r.patient_code("Trial", "patient031"), "P01-031");
        assert_eq!(r.patient_code("Trial", "01-031_case1_t"), "P01-031");
        assert_eq!(r.patient_code("Trial", "special"), "P01-099");
        assert_eq!(r.patient_code("Other", "special"), "special");
        assert_eq!(code_key("P01-039"), code_key("01-039"));
        assert_eq!(code_key("patient039"), "039");
        assert!(Rules::parse("version = 2").is_err());
        assert!(Rules::parse("unknown_key = 1").is_err());
    }
}
