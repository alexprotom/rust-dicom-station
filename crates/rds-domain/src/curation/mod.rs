//! Curating an archive: checking what its files hold, finding the copies,
//! and changing files in a way that can be checked and taken back.
//!
//! * [`raw`] - DICOM files as bytes: elements by position, hashes of the
//!   pixel data, rewrites that leave every untouched element byte for byte
//!   as it was, verified before they replace the file.
//! * [`journal`] - every write is a dry run first (a [`journal::Manifest`]
//!   of operations in the records folder), then an apply that journals each
//!   operation and can be resumed, then, if need be, an undo.
//! * [`audit`] - what is wrong with a folder: files that do not parse,
//!   pixel data shorter than its header says, gaps and doubles along a
//!   series, structure sets and registrations whose references lead
//!   nowhere.
//! * [`dupes`] - which files of one folder are already in another: the same
//!   SOP Instance UID with the same content, or the same pixel data under
//!   new UIDs (a re-anonymized copy), per series.
//!
//! The MCP server's curation tools (`mcp/tools/curation.rs`) are these
//! functions behind the configured `[pacs]` folders; nothing here knows
//! about the protocol, and nothing here returns a patient's name or ID -
//! the findings name files, UIDs and tags.

pub mod audit;
pub mod dupes;
pub mod facts;
pub mod journal;
pub mod raw;

/// The DICOM files under `dir`, sorted: every file whose header reads,
/// whatever its name.
pub fn files_under(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out: Vec<std::path::PathBuf> = walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| !is_own_file(p))
        .collect();
    out.sort();
    out
}

/// The catalog, its lock, and what a rewrite leaves for a moment: never a
/// file of the archive.
fn is_own_file(p: &std::path::Path) -> bool {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    name.starts_with("rds-pacs-") || name.starts_with(".rds-pacs-") || name.ends_with(".tmp")
}
