//! What a person decided about a PACS, as a file another station can take
//! in: the names given to patients, studies and series, the cohorts and who
//! is in which, and the 4D groups made, confirmed or dissolved by hand.
//!
//! Everything else in the catalog comes from the DICOM files and is found
//! again by any station that holds them; these decisions live only in the
//! catalog ([`super::catalog`]), so two stations holding copies of the same
//! studies share them through this file. Objects are named by what travels
//! with the files: a study or a series by its UID, a patient by the key the
//! catalog files them under (the Patient ID), a cohort by its name - never
//! by a path, which is another station's own.
//!
//! An import merges: a name in the file is given to the object where the
//! station holds it and the file's name is not empty (a name of the
//! station's own that the file does not mention stays), a cohort is found
//! by name or made, its patients are put into it beside the cohorts they
//! are in already, a 4D decision is kept when the station holds one of its
//! series. What the station does not hold is counted, never an error.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::catalog::{self, Catalog};
use super::fourd::CustomGroup;
use super::Archive;

const FORMAT: &str = "rds-pacs-tags";
const VERSION: u32 = 1;

/// One cohort: its name, and the patients in it (by key).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CohortTags {
    pub name: String,
    pub patients: Vec<String>,
}

/// The file.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TagsFile {
    pub format: String,
    pub version: u32,
    /// When it was written, `YYYYMMDDTHHMMSS`.
    pub exported: String,
    /// The program that wrote it.
    pub program: String,
    pub cohorts: Vec<CohortTags>,
    /// Patient key -> name given.
    pub patients: BTreeMap<String, String>,
    /// Study Instance UID -> name given.
    pub studies: BTreeMap<String, String>,
    /// Series Instance UID -> name given.
    pub series: BTreeMap<String, String>,
    /// The 4D groups decided by hand, by group id.
    pub fourd: BTreeMap<String, CustomGroup>,
}

/// What an export wrote or an import did.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TagsSummary {
    pub cohorts: usize,
    pub patients: usize,
    pub studies: usize,
    pub series: usize,
    pub fourd: usize,
    /// Import: cohorts made because the station had none of that name.
    pub cohorts_made: usize,
    /// Import: names and members the station holds no object for.
    pub not_held: usize,
}

impl TagsSummary {
    /// `12 study names, 3 series names, 2 cohorts, 1 4D group`.
    pub fn describe(&self) -> String {
        let n = |k: usize, one: &str, many: &str| {
            (k > 0).then(|| format!("{k} {}", if k == 1 { one } else { many }))
        };
        let parts: Vec<String> = [
            n(self.cohorts, "cohort", "cohorts"),
            n(self.patients, "patient name", "patient names"),
            n(self.studies, "study name", "study names"),
            n(self.series, "series name", "series names"),
            n(self.fourd, "4D group", "4D groups"),
        ]
        .into_iter()
        .flatten()
        .collect();
        let mut s = if parts.is_empty() {
            "nothing".to_string()
        } else {
            parts.join(", ")
        };
        if self.cohorts_made > 0 {
            s.push_str(&format!(" ({} cohort(s) made)", self.cohorts_made));
        }
        if self.not_held > 0 {
            s.push_str(&format!("; {} not held here", self.not_held));
        }
        s
    }
}

/// The person's decisions of a catalog, as the file says them.
pub fn of_catalog(c: &Catalog) -> TagsFile {
    let members = catalog::membership(c);
    let mut cohorts: Vec<CohortTags> = c
        .cohorts
        .iter()
        .map(|k| CohortTags {
            name: k.name.clone(),
            patients: members
                .iter()
                .filter(|(_, ids)| ids.contains(&k.id))
                .map(|(key, _)| key.clone())
                .collect(),
        })
        .filter(|k| !k.patients.is_empty())
        .collect();
    cohorts.sort_by(|a, b| a.name.cmp(&b.name));
    let labelled = |m: &BTreeMap<String, String>| -> BTreeMap<String, String> {
        m.iter()
            .filter(|(_, l)| !l.trim().is_empty())
            .map(|(k, l)| (k.clone(), l.clone()))
            .collect()
    };
    let (date, time) = crate::dicom_export::today();
    TagsFile {
        format: FORMAT.into(),
        version: VERSION,
        exported: format!("{date}T{time}"),
        program: format!("Rust DICOM Station {}", env!("CARGO_PKG_VERSION")),
        cohorts,
        patients: labelled(
            &c.patients
                .iter()
                .map(|(k, p)| (k.clone(), p.label.clone()))
                .collect(),
        ),
        studies: labelled(
            &c.studies
                .iter()
                .map(|(k, s)| (k.clone(), s.label.clone()))
                .collect(),
        ),
        series: labelled(
            &c.series
                .iter()
                .map(|(k, s)| (k.clone(), s.label.clone()))
                .collect(),
        ),
        fourd: c.fourd_custom.clone(),
    }
}

fn counts(t: &TagsFile) -> TagsSummary {
    TagsSummary {
        cohorts: t.cohorts.len(),
        patients: t.patients.len(),
        studies: t.studies.len(),
        series: t.series.len(),
        fourd: t.fourd.len(),
        ..Default::default()
    }
}

/// Merge a file's decisions into a catalog (see the module doc).
pub fn merge_into(c: &mut Catalog, t: &TagsFile) -> TagsSummary {
    let mut sum = TagsSummary::default();
    for (key, label) in t.patients.iter().filter(|(_, l)| !l.trim().is_empty()) {
        match c.patients.get_mut(key) {
            Some(p) => {
                p.label = label.clone();
                sum.patients += 1;
            }
            None => sum.not_held += 1,
        }
    }
    for (uid, label) in t.studies.iter().filter(|(_, l)| !l.trim().is_empty()) {
        match c.studies.get_mut(uid) {
            Some(s) => {
                s.label = label.clone();
                sum.studies += 1;
            }
            None => sum.not_held += 1,
        }
    }
    for (uid, label) in t.series.iter().filter(|(_, l)| !l.trim().is_empty()) {
        match c.series.get_mut(uid) {
            Some(s) => {
                s.label = label.clone();
                sum.series += 1;
            }
            None => sum.not_held += 1,
        }
    }
    // Cohorts by name; the patients put in beside where they are already.
    let now = catalog::membership(c);
    for k in &t.cohorts {
        let name = k.name.trim();
        if name.is_empty() {
            continue;
        }
        let held: Vec<&String> = k
            .patients
            .iter()
            .filter(|p| c.patients.contains_key(*p))
            .collect();
        sum.not_held += k.patients.len() - held.len();
        if held.is_empty() {
            continue;
        }
        let id = match c
            .cohorts
            .iter()
            .find(|x| x.name.trim().eq_ignore_ascii_case(name))
        {
            Some(x) => x.id.clone(),
            None => {
                let id = catalog::new_cohort_id(c);
                c.cohorts.push(catalog::CohortRec {
                    id: id.clone(),
                    name: name.to_string(),
                    folder: None,
                });
                sum.cohorts_made += 1;
                id
            }
        };
        for key in held {
            let mut set: BTreeSet<String> = now.get(key).cloned().unwrap_or_default();
            if set.insert(id.clone()) {
                // Unsorted is where a patient in no cohort is; one put into
                // a cohort leaves it.
                set.remove(catalog::UNSORTED);
                c.membership.insert(key.clone(), set.into_iter().collect());
            }
        }
        sum.cohorts += 1;
    }
    for (id, g) in &t.fourd {
        if g.members.iter().any(|m| c.series.contains_key(&m.0)) {
            c.fourd_custom.insert(id.clone(), g.clone());
            sum.fourd += 1;
        } else {
            sum.not_held += 1;
        }
    }
    sum
}

impl Archive {
    /// Write the person's decisions about this PACS into `path` as JSON.
    pub fn export_tags(&self, path: &Path) -> Result<TagsSummary> {
        let t = of_catalog(&catalog::load(self.root()));
        crate::pathlen::check(path)?;
        let text = serde_json::to_string_pretty(&t)?;
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
        Ok(counts(&t))
    }

    /// Merge the decisions a file of [`Archive::export_tags`] holds into
    /// this PACS's catalog.
    pub fn import_tags(&self, path: &Path) -> Result<TagsSummary> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let t: TagsFile = serde_json::from_str(&text)
            .with_context(|| format!("{} is no file of PACS tags", path.display()))?;
        if t.format != FORMAT {
            bail!(
                "{} is no file of PACS tags (it says '{}')",
                path.display(),
                t.format
            );
        }
        if t.version > VERSION {
            bail!(
                "{} was written by a newer version of the program (tags version {})",
                path.display(),
                t.version
            );
        }
        catalog::update(self.root(), move |c| {
            let sum = merge_into(c, &t);
            catalog::ensure_cohorts(c);
            Ok(sum)
        })
    }
}
