//! Saving what was made on a study of the PACS into the PACS itself: new
//! structure sets and segmentation series, filed beside the study they
//! belong to, in the PACS's own folder layout; and changed ones written
//! back into the series they came from.
//!
//! The objects are written as [`dicom_export::export_derived_as`] writes
//! them (a fresh SOP Instance UID each, the study's own Study Instance UID
//! and Frame of Reference UID) into a scratch folder, and then filed by an
//! [`Importer`] told to put a study's files beside the ones it holds
//! wherever they are ([`Importer::beside_the_study`]): beside the image
//! series each object was drawn on ([`Importer::referencing`]), in the
//! patient's folder of a cohort folder of a second storage folder as much
//! as in the main folder, never loose in a storage or a cohort folder. The
//! catalog learns them in the same step, so the PACS window lists them
//! without a rescan.
//!
//! **Into which series** ([`SaveAs`]). A new object, and one saved as a new
//! series, gets a Series Instance UID of its own, the study's next free
//! Series Number (one past the largest the catalog lists for the study, so
//! two saves never read `RTSTRUCT 2` twice) and a Series Description (the
//! one given, else the object's label). An object saved as an update of the
//! series the PACS holds it in keeps that series - its Series Instance UID,
//! number and description - and **replaces** the instance it was loaded as:
//!
//! - the new instance gets a new SOP Instance UID, as the standard asks of
//!   changed content, and as the PACS needs: its filing leaves alone a file
//!   whose SOP Instance UID it holds, and a server's mirror syncs by SOP
//!   Instance UID sets, so a changed file under its old UID would reach
//!   neither;
//! - it is filed in the folder of the file it replaces (an object's series
//!   stays together, wherever a planning system put it), under the path
//!   limit as any filed file;
//! - only once it is in its place is the old file removed, and the catalog
//!   then forgets the old file and learns the new one in one update: at no
//!   point is the series without a file, and a save that stops before
//!   filing leaves the old one where it was;
//! - an old structure set that an RT Plan of the study names (its
//!   Referenced Structure Set Sequence) is kept: the plan would name a set
//!   the PACS no longer holds. The new instance joins the series beside it,
//!   and [`Saved::notes`] says so.
//!
//! Only a study the PACS holds is saved this way; the caller checks
//! ([`Archive::save_targets`]) and says so otherwise. What the PACS holds
//! already is told by SOP Instance UID: an object loaded from it and left
//! as it was need not go again, and one saved here once is, after [`Saved`]
//! has been taken back into the workspace, an object of the PACS like any
//! other.
//!
//! **A save that stops half way** (a file that cannot be put in its place)
//! keeps what it filed: the catalog learns those files, [`Saved::objects`]
//! lists them and [`Saved::failed`] says what stopped the rest. The caller
//! takes over the identities of the objects saved, so a retry writes only
//! the others. An error is returned only when nothing was filed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use dicom_dictionary_std::tags;

use crate::dicom_export::{self, DerivedFile, DerivedKind, ExportParams, SeriesIdent};
use crate::loader::LoadedStudy;
use crate::progress::Progress;

use super::catalog::Catalog;
use super::{catalog, Archive, ImportSummary};

/// How one object goes into the PACS.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SaveAs {
    /// A series of its own: a new Series Instance UID, the study's next free
    /// Series Number, this Series Description (empty: the object's label).
    NewSeries { description: String },
    /// A new instance of the series the PACS holds the instance `replaces`
    /// (its SOP Instance UID) in, replacing it. When the PACS no longer
    /// holds it (or its files named no Series Instance UID), the object goes
    /// into a new series, and [`Saved::notes`] says so.
    Update { replaces: String },
}

/// One object saved: which it was, and what it is now in the PACS.
#[derive(Clone, Debug)]
pub struct SavedObject {
    pub kind: DerivedKind,
    pub index: usize,
    pub sop_instance_uid: String,
    pub series_instance_uid: String,
    /// Its file in the PACS.
    pub path: PathBuf,
    /// The instance it replaced in its series (removed from the PACS, or
    /// kept beside it for a plan that names it): its SOP Instance UID.
    pub replaced: Option<String>,
}

/// What a save did.
#[derive(Clone, Debug, Default)]
pub struct Saved {
    pub objects: Vec<SavedObject>,
    pub summary: ImportSummary,
    /// What stopped the save before every object was filed; the ones in
    /// [`Saved::objects`] are in the PACS all the same.
    pub failed: Option<String>,
    /// What went otherwise than asked without failing: an old instance kept
    /// for a plan that names it, or that could not be removed; an update
    /// whose series the PACS no longer holds, saved as a new series.
    pub notes: Vec<String>,
}

/// Write the derived objects of `study` that `keep` names into the PACS,
/// beside the study each belongs to, each as a new series.
pub fn save_derived(
    archive: &Archive,
    study: &LoadedStudy,
    params: &ExportParams,
    progress: &Progress,
    keep: &dyn Fn(DerivedKind, usize) -> bool,
) -> Result<Saved> {
    save_objects(archive, study, params, progress, &|k, i| {
        keep(k, i).then(|| SaveAs::NewSeries {
            description: String::new(),
        })
    })
}

/// The instance an update replaces, as the catalog lists it.
#[derive(Clone, Debug)]
struct Old {
    sop: String,
    /// Every copy of it in this PACS's folders.
    paths: Vec<PathBuf>,
    /// Where the new instance goes: the folder of its first copy.
    dir: PathBuf,
    /// A plan of the study that names it: it stays.
    named_by: Option<String>,
}

/// Write the derived objects of `study` that `how` names into the PACS:
/// each into a new series, or as a new instance of the series it came from
/// ([`SaveAs`]).
pub fn save_objects(
    archive: &Archive,
    study: &LoadedStudy,
    params: &ExportParams,
    progress: &Progress,
    how: &dyn Fn(DerivedKind, usize) -> Option<SaveAs>,
) -> Result<Saved> {
    let c = archive.cat();
    let map = archive.storage_map(&c);
    let mut numbers = Numbers::default();
    let mut notes = Vec::new();
    let mut idents: HashMap<(DerivedKind, usize), SeriesIdent> = HashMap::new();
    let mut replacing: HashMap<(DerivedKind, usize), Old> = HashMap::new();
    // Every object to save, with its study and label.
    let objects = study
        .structure_sets
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.rois.is_empty())
        .map(|(i, s)| {
            let label = if s.label.is_empty() {
                s.file_name.clone()
            } else {
                s.label.clone()
            };
            (
                DerivedKind::StructureSet,
                i,
                &s.study_uid,
                &s.referenced_series_uid,
                label,
            )
        })
        .chain(
            study
                .seg_series
                .iter()
                .enumerate()
                .filter(|(_, s)| s.segs.iter().any(|g| g.count > 0))
                .map(|(i, s)| {
                    (
                        DerivedKind::Segmentation,
                        i,
                        &s.study_uid,
                        &s.referenced_series_uid,
                        s.label.clone(),
                    )
                }),
        );
    for (kind, i, own, referenced, label) in objects {
        let Some(as_) = how(kind, i) else {
            continue;
        };
        let study_uid = dicom_export::derived_study_uid(study, own, referenced);
        let description = match as_ {
            // A series the catalog keys by something that is no UID (its
            // files named none) cannot be written into: a new one.
            SaveAs::Update { replaces } => match held_instance(archive, &c, &map, &replaces)
                .filter(|(series, ..)| super::is_uid(series))
            {
                Some((series_uid, number, description, paths)) => {
                    let named_by = (kind == DerivedKind::StructureSet)
                        .then(|| plan_naming(archive, &c, &map, &study_uid, &replaces))
                        .flatten();
                    let series_number = number
                        .trim()
                        .parse::<i64>()
                        .unwrap_or_else(|_| numbers.next(&c, &study_uid));
                    idents.insert(
                        (kind, i),
                        SeriesIdent {
                            series_uid,
                            series_number,
                            description,
                        },
                    );
                    let dir = paths[0].parent().map(Path::to_path_buf).unwrap_or_default();
                    replacing.insert(
                        (kind, i),
                        Old {
                            sop: replaces,
                            paths,
                            dir,
                            named_by,
                        },
                    );
                    continue;
                }
                None => {
                    notes.push(format!(
                        "{label}: its series could not be updated (no longer in the PACS, or \
                         its files name none); saved as a new series"
                    ));
                    String::new()
                }
            },
            SaveAs::NewSeries { description } => description,
        };
        let description = if description.trim().is_empty() {
            label
        } else {
            description.trim().to_string()
        };
        idents.insert(
            (kind, i),
            SeriesIdent {
                series_uid: dicom_export::new_uid(),
                series_number: numbers.next(&c, &study_uid),
                description,
            },
        );
    }

    let scratch = std::env::temp_dir().join(format!(
        "rds_pacs_save_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let out = (|| {
        let written =
            dicom_export::export_derived_as(study, &scratch, params, progress, &|k, i| {
                idents.get(&(k, i)).cloned()
            })?;
        // The image series each object was drawn on: where it is filed.
        let drawn_on = |f: &DerivedFile| -> String {
            match f.kind {
                DerivedKind::StructureSet => study
                    .structure_sets
                    .get(f.index)
                    .map(|s| s.referenced_series_uid.clone()),
                DerivedKind::Segmentation => study
                    .seg_series
                    .get(f.index)
                    .map(|s| s.referenced_series_uid.clone()),
            }
            .unwrap_or_default()
        };
        let files: Vec<(DerivedFile, String)> = written
            .into_iter()
            .map(|f| {
                let s = drawn_on(&f);
                (f, s)
            })
            .collect();
        file_replacing(archive, &scratch, &files, progress, &replacing)
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    out.map(|mut saved| {
        notes.append(&mut saved.notes);
        saved.notes = notes;
        saved
    })
}

/// The study's next free Series Numbers: one past the largest the catalog
/// lists for the study, then one more for each series made in this save.
#[derive(Default)]
struct Numbers(HashMap<String, i64>);

impl Numbers {
    fn next(&mut self, c: &Catalog, study: &str) -> i64 {
        let n = self.0.entry(study.to_string()).or_insert_with(|| {
            c.series
                .values()
                .filter(|s| s.study == study)
                .filter_map(|s| s.number.trim().parse::<i64>().ok())
                .max()
                .unwrap_or(0)
                .max(0)
                + 1
        });
        let out = *n;
        *n += 1;
        out
    }
}

/// The series an instance of this PACS is in (its Series Instance UID,
/// number and description) and every copy of its file, by its SOP Instance
/// UID; `None` when the PACS does not hold it.
fn held_instance(
    archive: &Archive,
    c: &Catalog,
    map: &HashMap<u32, usize>,
    sop: &str,
) -> Option<(String, String, String, Vec<PathBuf>)> {
    if sop.is_empty() {
        return None;
    }
    c.series.iter().find_map(|(uid, s)| {
        let paths: Vec<PathBuf> = s
            .files
            .iter()
            .filter(|f| f.4 == sop)
            .filter_map(|f| archive.path_of(map, f))
            .collect();
        (!paths.is_empty()).then(|| {
            (
                uid.clone(),
                s.number.clone(),
                s.description.trim().to_string(),
                paths,
            )
        })
    })
}

/// A plan of study `study` that names structure set `sop` (its Referenced
/// Structure Set Sequence), by its label: read from the plans' files, which
/// are few and small.
fn plan_naming(
    archive: &Archive,
    c: &Catalog,
    map: &HashMap<u32, usize>,
    study: &str,
    sop: &str,
) -> Option<String> {
    c.series
        .values()
        .filter(|s| s.study == study && s.modality == "RTPLAN")
        .flat_map(|s| s.files.iter())
        .filter_map(|f| archive.path_of(map, f))
        .find_map(|p| {
            let obj = crate::dicomfile::open_header(&p).ok()?;
            let names = crate::loader::items_of(&obj, tags::REFERENCED_STRUCTURE_SET_SEQUENCE)?
                .iter()
                .any(|it| {
                    crate::loader::str_of(it, tags::REFERENCED_SOP_INSTANCE_UID).as_deref()
                        == Some(sop)
                });
            names.then(|| {
                crate::loader::str_of(&obj, tags::RT_PLAN_LABEL)
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .unwrap_or_else(|| "a plan".into())
            })
        })
}

/// File `files` (written into `scratch`, each with the image series it was
/// drawn on) into the PACS beside their studies: the second half of
/// [`save_derived`]. Every file under `scratch` is filed; `files` says what
/// each one is.
pub fn file_written(
    archive: &Archive,
    scratch: &Path,
    files: &[(DerivedFile, String)],
    progress: &Progress,
) -> Result<Saved> {
    file_replacing(archive, scratch, files, progress, &HashMap::new())
}

/// [`file_written`], the objects of `replacing` filed beside the instance
/// each replaces, which goes once its replacement is in its place.
fn file_replacing(
    archive: &Archive,
    scratch: &Path,
    files: &[(DerivedFile, String)],
    progress: &Progress,
    replacing: &HashMap<(DerivedKind, usize), Old>,
) -> Result<Saved> {
    // What each file is, read back from the file itself.
    let heads: Vec<_> = files
        .iter()
        .filter_map(|(f, _)| catalog::read_head(&f.path).map(|h| (f, h)))
        .collect();
    let drawn_on: HashMap<String, String> = heads
        .iter()
        .filter_map(|(f, h)| {
            let series = &files.iter().find(|(g, _)| g.path == f.path)?.1;
            (!series.is_empty()).then(|| (h.sop.clone(), series.clone()))
        })
        .collect();
    let placing: HashMap<String, PathBuf> = heads
        .iter()
        .filter_map(|(f, h)| {
            let old = replacing.get(&(f.kind, f.index))?;
            (!old.dir.as_os_str().is_empty()).then(|| (h.sop.clone(), old.dir.clone()))
        })
        .collect();
    let mut im = archive
        .importer()
        .taking_files()
        .beside_the_study()
        .referencing(drawn_on)
        .placing(placing);
    let filed = im.file_folder(scratch, progress);
    let paths = im.filed_paths();
    // The instances replaced: off the disk once what replaces them is in
    // its place (never before, so the series never goes without a file),
    // and out of the catalog in the update that brings the new ones in.
    let mut notes = Vec::new();
    let mut gone: Vec<PathBuf> = Vec::new();
    let mut replaced: HashMap<(DerivedKind, usize), String> = HashMap::new();
    for (f, h) in &heads {
        let Some(old) = replacing.get(&(f.kind, f.index)) else {
            continue;
        };
        if !paths.get(&h.sop).is_some_and(|p| p.is_file()) {
            continue;
        }
        replaced.insert((f.kind, f.index), old.sop.clone());
        if let Some(plan) = &old.named_by {
            notes.push(format!(
                "the earlier version is kept beside it: plan {plan} refers to it"
            ));
            continue;
        }
        for p in &old.paths {
            match std::fs::remove_file(p) {
                Ok(()) => gone.push(p.clone()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => gone.push(p.clone()),
                Err(e) => notes.push(format!(
                    "the earlier version could not be removed ({e}); it stays beside the new one"
                )),
            }
        }
    }
    progress.set("Updating the catalog");
    let finished = im.finish_dropping(&gone);
    let objects: Vec<SavedObject> = heads
        .into_iter()
        .filter_map(|(f, h)| {
            Some(SavedObject {
                kind: f.kind,
                index: f.index,
                path: paths.get(&h.sop)?.clone(),
                replaced: replaced.get(&(f.kind, f.index)).cloned(),
                sop_instance_uid: h.sop,
                series_instance_uid: h.series,
            })
        })
        .collect();
    match filed {
        Ok((summary, _)) => {
            finished.context("update the PACS catalog")?;
            Ok(Saved {
                objects,
                summary,
                failed: None,
                notes,
            })
        }
        Err(e) => {
            if objects.is_empty() {
                return Err(e);
            }
            // Half way: what made it is in the PACS, and the catalog has to
            // say so, or a retry would write it a second time.
            if let Err(cat) = finished {
                bail!("{e:#}; the catalog could not be updated either: {cat:#}");
            }
            let summary = ImportSummary {
                stored: objects.len(),
                ..ImportSummary::default()
            };
            Ok(Saved {
                objects,
                summary,
                failed: Some(format!("{e:#}")),
                notes,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_series_take_the_numbers_past_the_largest_of_their_study() {
        let mut c = Catalog::default();
        for (uid, study, number) in [
            ("1", "A", "201"),
            ("2", "A", "2"),
            ("3", "A", ""),
            ("4", "B", "7"),
        ] {
            c.series.insert(
                uid.into(),
                catalog::SeriesRec {
                    study: study.into(),
                    number: number.into(),
                    ..Default::default()
                },
            );
        }
        let mut n = Numbers::default();
        assert_eq!(n.next(&c, "A"), 202);
        assert_eq!(n.next(&c, "A"), 203, "two in one save: two numbers");
        assert_eq!(n.next(&c, "B"), 8);
        assert_eq!(n.next(&c, "C"), 1, "a study with no series");
    }
}
