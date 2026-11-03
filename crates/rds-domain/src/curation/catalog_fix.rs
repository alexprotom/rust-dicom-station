//! Keeping the station's catalog in step with a curated archive: one
//! patient entry per patient folder, and labels for the studies.
//!
//! A refresh files every study under the patient its files name
//! ([`crate::archive::catalog::add_file`]), so after files of one patient
//! folder were retagged to its code the old patient entries go once no
//! study uses them - and the labels given to them go with them. [`relink`]
//! makes the patient folder the unit: every study whose files lie in one
//! patient folder of a cohort is filed under one patient entry (the one
//! whose ID is the folder's code, else the one with most files), the other
//! entries' labels and cohort memberships carried over to it; entries no
//! study uses any more are dropped.
//!
//! [`study_label`] is `<YYYY-MM-DD|undated> (<Upright|Supine>: <note>) ·
//! <modalities>`: the study's date (or its folder's, when the files carry
//! none), the position and note of its folder's name (as
//! [`super::reorg`] names folders), else the position the caller works out
//! from a file and the study's description; the modalities sorted.
//! [`fill_labels`] gives it to every study of a cohort that has no label -
//! a label someone typed is never overwritten.
//!
//! [`identity_clashes`] finds two people under one code: the catalog keys a
//! patient by Patient ID alone, so a code given in two cohorts (each with
//! its own numbering) files both people's studies under one entry. A
//! Patient ID whose studies lie in several cohort folders is checked by
//! the birth date and sex of one file per study; cohorts that share no
//! (birth date, sex) are reported, and the person splits them by recoding
//! one cohort's folder (`set_patient_identity`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use dicom_dictionary_std::tags;
use rayon::prelude::*;
use serde::Serialize;

use super::reorg::code_key;
use crate::archive::catalog::{self, Catalog, PatientRec, TopKind};

/// The study folders of a cohort: study UID → (storage index, patient
/// folder, study folder), from where its files lie
/// (`cohort/patient/study/series/file`).
pub fn study_folders(c: &Catalog, cohort: &str) -> BTreeMap<String, (u32, String, String)> {
    let mut out = BTreeMap::new();
    for s in c.series.values() {
        for f in &s.files {
            let parts: Vec<&str> = f.1.split('/').collect();
            if parts.len() >= 5 && parts[0] == cohort {
                out.entry(s.study.clone()).or_insert((
                    f.0,
                    parts[1].to_string(),
                    parts[2].to_string(),
                ));
            }
        }
    }
    out
}

#[derive(Debug, Default, Serialize)]
pub struct Relinked {
    pub patient_folders: usize,
    pub studies_moved: usize,
    pub entries_dropped: usize,
    pub labels_carried: usize,
}

/// One patient entry per patient folder of `cohort` (see the module doc).
pub fn relink(c: &mut Catalog, cohort: &str) -> Relinked {
    let folders = study_folders(c, cohort);
    let mut by_folder: BTreeMap<(u32, String), Vec<String>> = BTreeMap::new();
    for (study, (r, patient, _)) in &folders {
        by_folder
            .entry((*r, patient.clone()))
            .or_default()
            .push(study.clone());
    }
    let files_of = |c: &Catalog, key: &str| -> usize {
        c.series
            .values()
            .filter(|s| c.studies.get(&s.study).is_some_and(|st| st.patient == key))
            .map(|s| s.files.len())
            .sum()
    };
    let mut out = Relinked {
        patient_folders: by_folder.len(),
        ..Default::default()
    };
    for ((_, folder), studies) in by_folder {
        let keys: BTreeSet<String> = studies
            .iter()
            .filter_map(|s| c.studies.get(s).map(|st| st.patient.clone()))
            .collect();
        // The entry whose ID is the folder's code, else the biggest.
        let target = keys
            .iter()
            .find(|k| {
                c.patients
                    .get(*k)
                    .is_some_and(|p| code_key(&p.id) == code_key(&folder))
            })
            .cloned()
            .or_else(|| keys.iter().max_by_key(|k| files_of(c, k)).cloned());
        let Some(target) = target else {
            continue;
        };
        for s in &studies {
            if let Some(st) = c.studies.get_mut(s) {
                if st.patient != target {
                    st.patient = target.clone();
                    out.studies_moved += 1;
                }
            }
        }
        // What the other entries carried: labels, memberships.
        for k in keys.iter().filter(|k| **k != target) {
            let old: Option<PatientRec> = c.patients.get(k).cloned();
            if let Some(old) = old {
                let t = c.patients.entry(target.clone()).or_default();
                if t.label.is_empty() && !old.label.is_empty() {
                    t.label = old.label;
                    out.labels_carried += 1;
                }
            }
            if let Some(m) = c.membership.get(k).cloned() {
                c.membership.entry(target.clone()).or_default().extend(m);
                c.membership.get_mut(&target).unwrap().sort();
                c.membership.get_mut(&target).unwrap().dedup();
            }
        }
    }
    // Entries no study uses any more.
    let used: BTreeSet<String> = c.studies.values().map(|s| s.patient.clone()).collect();
    let before = c.patients.len();
    c.patients.retain(|k, _| used.contains(k));
    c.membership.retain(|k, _| used.contains(k));
    out.entries_dropped = before - c.patients.len();
    out
}

/// `(position, note)` of a study folder named by [`super::reorg`].
fn folder_parts(name: &str) -> Option<(String, String, String)> {
    let re = regex::Regex::new(r"^(\d{8}|NoDate)_(?:(Upright|Supine)-)?(.+?)_StudyUID-\d{1,12}$")
        .ok()?;
    let c = re.captures(name)?;
    Some((
        c[1].to_string(),
        c.get(2).map(|m| m.as_str().to_string()).unwrap_or_default(),
        c[3].to_string(),
    ))
}

/// The label of a study (see the module doc). `position` is asked when
/// the folder's name does not say.
pub fn study_label(
    c: &Catalog,
    uid: &str,
    folder: Option<&str>,
    position: &dyn Fn() -> Option<&'static str>,
) -> Option<String> {
    let st = c.studies.get(uid)?;
    let parts = folder.and_then(folder_parts);
    let date = if st.date.len() >= 8 {
        st.date[..8].to_string()
    } else {
        parts
            .as_ref()
            .map(|p| p.0.clone())
            .filter(|d| d.len() == 8 && d.chars().all(|x| x.is_ascii_digit()))
            .unwrap_or_default()
    };
    let date = if date.len() == 8 {
        format!("{}-{}-{}", &date[..4], &date[4..6], &date[6..8])
    } else {
        "undated".to_string()
    };
    let (pos, note) = match &parts {
        Some((_, p, n)) if !p.is_empty() => (p.clone(), n.clone()),
        Some((_, _, n)) => (position().unwrap_or("Supine").to_string(), n.clone()),
        None => (
            position().unwrap_or("Supine").to_string(),
            if st.description.trim().is_empty() {
                "No-description".to_string()
            } else {
                st.description.trim().to_string()
            },
        ),
    };
    let modalities: BTreeSet<&str> = c
        .series
        .values()
        .filter(|s| s.study == uid && !s.modality.is_empty())
        .map(|s| s.modality.as_str())
        .collect();
    Some(format!(
        "{date} ({pos}: {note}) \u{b7} {}",
        modalities.into_iter().collect::<Vec<_>>().join(", ")
    ))
}

/// Label every unlabelled study of `cohort`; returns how many.
pub fn fill_labels(
    c: &mut Catalog,
    cohort: &str,
    position: &dyn Fn(&str) -> Option<&'static str>,
) -> usize {
    let folders = study_folders(c, cohort);
    let mut labels = Vec::new();
    for (uid, (_, _, folder)) in &folders {
        let Some(st) = c.studies.get(uid) else {
            continue;
        };
        if !st.label.trim().is_empty() {
            continue;
        }
        if let Some(l) = study_label(c, uid, Some(folder), &|| position(uid)) {
            labels.push((uid.clone(), l));
        }
    }
    let n = labels.len();
    for (uid, l) in labels {
        if let Some(st) = c.studies.get_mut(&uid) {
            st.label = l;
        }
    }
    n
}

/// One cohort folder of a patient in an [`IdentityClash`].
#[derive(Clone, Debug, Serialize)]
pub struct ClashPart {
    /// The cohort folder (empty: the patient's folder is directly in the
    /// storage folder).
    pub cohort: String,
    /// The storage folder, by its index in the catalog, and the patient's
    /// folder below it (`cohort/patient`, or `patient`).
    pub storage: u32,
    pub folder: String,
    pub studies: usize,
    /// How many distinct (birth date, sex) the files of those studies
    /// carry; the values themselves are never returned.
    pub variants: usize,
    #[serde(skip)]
    seen: BTreeSet<(String, String)>,
}

/// A Patient ID whose cohort folders hold files of different people.
#[derive(Clone, Debug, Serialize)]
pub struct IdentityClash {
    /// The patient key (the Patient ID).
    pub patient: String,
    pub parts: Vec<ClashPart>,
}

/// The patients filed in more than one cohort folder whose folders share no
/// (birth date, sex): see the module doc. One file of every study of such
/// a patient is read (its header only).
pub fn identity_clashes(c: &Catalog) -> Vec<IdentityClash> {
    let tops = catalog::top_folders(c);
    // Where a file lies: (storage, cohort folder, patient folder).
    let place = |r: u32, rel: &str| -> Option<(u32, String, String)> {
        let parts: Vec<&str> = rel.split('/').collect();
        if parts.len() < 2 {
            return None;
        }
        let cohort = tops.get(&(r, parts[0].to_string())) == Some(&TopKind::Cohort);
        if cohort && parts.len() >= 3 {
            Some((
                r,
                parts[0].to_string(),
                format!("{}/{}", parts[0], parts[1]),
            ))
        } else {
            Some((r, String::new(), parts[0].to_string()))
        }
    };
    // Patient -> (storage, cohort, folder) -> the studies there, each with
    // one file to read.
    type Places = BTreeMap<(u32, String, String), BTreeMap<String, String>>;
    let mut by_patient: BTreeMap<String, Places> = BTreeMap::new();
    for s in c.series.values() {
        let Some(st) = c.studies.get(&s.study) else {
            continue;
        };
        let Some(f) = s.files.first() else {
            continue;
        };
        let Some(at) = place(f.0, &f.1) else {
            continue;
        };
        let Some(root) = c.storage.get(f.0 as usize) else {
            continue;
        };
        let path = catalog::abs_of(Path::new(root), &f.1);
        by_patient
            .entry(st.patient.clone())
            .or_default()
            .entry(at)
            .or_default()
            .entry(s.study.clone())
            .or_insert_with(|| path.display().to_string());
    }
    // Only a patient in two cohort folders or more is read at all.
    by_patient.retain(|_, places| {
        places
            .keys()
            .map(|(r, cohort, _)| (*r, cohort.clone()))
            .collect::<BTreeSet<_>>()
            .len()
            > 1
    });
    let mut out: Vec<IdentityClash> = by_patient
        .into_par_iter()
        .filter_map(|(patient, places)| {
            let parts: Vec<ClashPart> = places
                .into_iter()
                .map(|((storage, cohort, folder), studies)| {
                    let seen: BTreeSet<(String, String)> = studies
                        .values()
                        .filter_map(|p| crate::dicomfile::open_scan(Path::new(p)).ok())
                        .map(|o| {
                            let t = |tag| {
                                crate::loader::str_of(&o, tag)
                                    .unwrap_or_default()
                                    .trim()
                                    .to_string()
                            };
                            (t(tags::PATIENT_BIRTH_DATE), t(tags::PATIENT_SEX))
                        })
                        .collect();
                    ClashPart {
                        cohort,
                        storage,
                        folder,
                        studies: studies.len(),
                        variants: seen.len(),
                        seen,
                    }
                })
                .collect();
            // Two cohorts with no (birth date, sex) in common are two people.
            let apart = parts.iter().enumerate().any(|(i, a)| {
                parts[i + 1..].iter().any(|b| {
                    a.cohort != b.cohort
                        && !a.seen.is_empty()
                        && !b.seen.is_empty()
                        && a.seen.is_disjoint(&b.seen)
                })
            });
            apart.then_some(IdentityClash { patient, parts })
        })
        .collect();
    out.sort_by(|a, b| a.patient.cmp(&b.patient));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::catalog::{add_file, FileRec, Head};

    fn head(pid: &str, study: &str, series: &str, modality: &str, date: &str) -> Head {
        Head {
            sop: format!("{series}.1"),
            study: study.into(),
            series: series.into(),
            patient_id: pid.into(),
            patient_name: pid.into(),
            modality: modality.into(),
            study_date: date.into(),
            ..Default::default()
        }
    }

    #[test]
    fn one_entry_per_patient_folder_with_the_labels_carried() {
        let mut c = Catalog::default();
        let f = |rel: &str| FileRec(0, rel.into(), 1, 0, rel.into(), None);
        // P01-031's folder: one study still under an old ID, labelled.
        add_file(&mut c, &head("P01-031", "1.1", "1.1.1", "CT", "20240716"),
            f("Trial/P01-031/20240716_Upright-Procedure#4_StudyUID-11/Series-1_CT_x_SeriesUID-111/a"));
        add_file(&mut c, &head("old-31", "1.2", "1.2.1", "REG", ""),
            f("Trial/P01-031/NoDate_Upright-Procedure#5_StudyUID-12/Series-1_REG_x_SeriesUID-121/b"));
        c.patients.get_mut("old-31").unwrap().label = "kept by hand".into();
        let r = relink(&mut c, "Trial");
        assert_eq!(
            (r.patient_folders, r.studies_moved, r.entries_dropped),
            (1, 1, 1)
        );
        assert_eq!(c.studies["1.2"].patient, "P01-031");
        assert_eq!(c.patients["P01-031"].label, "kept by hand");
        // Labels: date, folder position and note, modalities.
        c.studies.get_mut("1.1").unwrap().label = "typed".into();
        let n = fill_labels(&mut c, "Trial", &|_| Some("Supine"));
        assert_eq!(n, 1);
        assert_eq!(c.studies["1.1"].label, "typed", "never overwritten");
        assert_eq!(
            c.studies["1.2"].label,
            "undated (Upright: Procedure#5) \u{b7} REG"
        );
        assert_eq!(
            study_label(
                &c,
                "1.1",
                Some("20240716_Upright-Procedure#4_StudyUID-11"),
                &|| None
            )
            .unwrap(),
            "2024-07-16 (Upright: Procedure#4) \u{b7} CT"
        );
    }
}
