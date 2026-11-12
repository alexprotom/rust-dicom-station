//! Saving what was made on a study of the PACS into the PACS itself: new
//! structure sets and segmentation series, filed beside the study they
//! belong to, in the PACS's own folder layout.
//!
//! The objects are written as [`dicom_export::export_derived`] writes them
//! (fresh SOP Instance and Series Instance UIDs, the study's own Study
//! Instance UID and Frame of Reference UID) into a scratch folder, and
//! then filed by an [`Importer`] told to put a study's files beside the
//! ones it holds wherever they are ([`Importer::beside_the_study`]): beside
//! the image series each object was drawn on ([`Importer::referencing`]),
//! in the patient's folder of a cohort folder of a second storage folder as
//! much as in the main folder, never loose in a storage or a cohort folder.
//! The catalog learns them in the same step, so the PACS window lists them
//! without a rescan.
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

use crate::dicom_export::{self, DerivedFile, DerivedKind, ExportParams};
use crate::loader::LoadedStudy;
use crate::progress::Progress;

use super::{catalog, Archive, ImportSummary};

/// One object saved: which it was, and what it is now in the PACS.
#[derive(Clone, Debug)]
pub struct SavedObject {
    pub kind: DerivedKind,
    pub index: usize,
    pub sop_instance_uid: String,
    pub series_instance_uid: String,
    /// Its file in the PACS.
    pub path: PathBuf,
}

/// What a save did.
#[derive(Clone, Debug, Default)]
pub struct Saved {
    pub objects: Vec<SavedObject>,
    pub summary: ImportSummary,
    /// What stopped the save before every object was filed; the ones in
    /// [`Saved::objects`] are in the PACS all the same.
    pub failed: Option<String>,
}

/// Write the derived objects of `study` that `keep` names into the PACS,
/// beside the study each belongs to.
pub fn save_derived(
    archive: &Archive,
    study: &LoadedStudy,
    params: &ExportParams,
    progress: &Progress,
    keep: &dyn Fn(DerivedKind, usize) -> bool,
) -> Result<Saved> {
    let scratch = std::env::temp_dir().join(format!(
        "rds_pacs_save_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let out = (|| {
        let written = dicom_export::export_derived_some(study, &scratch, params, progress, keep)?;
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
        file_written(archive, &scratch, &files, progress)
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    out
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
    let mut im = archive
        .importer()
        .taking_files()
        .beside_the_study()
        .referencing(drawn_on);
    let filed = im.file_folder(scratch, progress);
    let paths = im.filed_paths();
    progress.set("Updating the catalog");
    let finished = im.finish();
    let objects: Vec<SavedObject> = heads
        .into_iter()
        .filter_map(|(f, h)| {
            Some(SavedObject {
                kind: f.kind,
                index: f.index,
                path: paths.get(&h.sop)?.clone(),
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
            })
        }
    }
}
