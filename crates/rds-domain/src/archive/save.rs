//! Saving what was made on a study of the PACS into the PACS itself: new
//! structure sets and segmentation series, filed beside the study they
//! belong to, in the PACS's own folder layout.
//!
//! The objects are written as [`dicom_export::export_derived`] writes them
//! (fresh SOP Instance and Series Instance UIDs, the study's own Study
//! Instance UID and Frame of Reference UID) into a scratch folder, and
//! then filed by an [`Importer`] told to put a study's files beside the
//! ones it holds wherever they are ([`Importer::beside_the_study`]): the
//! folder of a patient in a cohort folder of a second storage folder as
//! much as one in the main folder. The catalog learns them in the same
//! step, so the PACS window lists them without a rescan.
//!
//! Only a study the PACS holds is saved this way; the caller checks
//! ([`Archive::study_home`]) and says so otherwise. What the PACS holds
//! already is told by SOP Instance UID ([`Archive::held_sops`]): an object
//! loaded from it and left as it was need not go again, and one saved here
//! once is, after [`Saved`] has been taken back into the workspace, an
//! object of the PACS like any other.

use std::path::PathBuf;

use anyhow::Result;

use crate::dicom_export::{self, DerivedKind, ExportParams};
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
        // What each file is, read back from the file itself.
        let heads: Vec<_> = written
            .iter()
            .filter_map(|f| catalog::read_head(&f.path).map(|h| (f, h)))
            .collect();
        let mut im = archive.importer().taking_files().beside_the_study();
        let filed = im.file_folder(&scratch, progress);
        let paths = im.filed_paths();
        progress.set("Updating the catalog");
        let finished = im.finish();
        let (summary, _) = filed?;
        finished?;
        let objects = heads
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
        Ok(Saved { objects, summary })
    })();
    let _ = std::fs::remove_dir_all(&scratch);
    out
}
