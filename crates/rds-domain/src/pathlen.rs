//! How long a path the station writes may be.
//!
//! Windows' classic limit is 260 characters (`MAX_PATH`, the terminating
//! NUL included), and a good deal of DICOM software - archives, planning
//! systems, the viewer of a CD - still has it: a file written deeper than
//! that may be written fine and then be unreadable on the next machine.
//! So every path the station writes (exports, the archive's filing and
//! moves, the curation tools' manifests) is held to 259 characters, and
//! one over 250 is written with a note.

use std::path::Path;

use anyhow::{bail, Result};

/// The longest path written.
pub const MAX_PATH: usize = 259;
/// A path this long is written, with a note.
pub const WARN_PATH: usize = 250;

/// The length that counts: characters, not bytes, and without the `\\?\`
/// prefix a canonical Windows path carries.
pub fn len(p: &Path) -> usize {
    let s = p.to_string_lossy();
    s.strip_prefix(r"\\?\").unwrap_or(&s).chars().count()
}

/// Refuse a path over [`MAX_PATH`].
pub fn check(p: &Path) -> Result<()> {
    let n = len(p);
    if n > MAX_PATH {
        bail!(
            "{} would be {n} characters long, over the {MAX_PATH} that DICOM software on \
             Windows reads; choose a shorter folder",
            p.display()
        );
    }
    Ok(())
}

/// A note for every path in `paths` over [`WARN_PATH`], at most `max`
/// (the rest counted).
pub fn notes<'a>(paths: impl IntoIterator<Item = &'a Path>, max: usize) -> Vec<String> {
    let long: Vec<(&Path, usize)> = paths
        .into_iter()
        .map(|p| (p, len(p)))
        .filter(|(_, n)| *n > WARN_PATH)
        .collect();
    let mut out: Vec<String> = long
        .iter()
        .take(max)
        .map(|(p, n)| {
            format!(
                "{} is {n} characters long (near the {MAX_PATH} limit)",
                p.display()
            )
        })
        .collect();
    if long.len() > max {
        out.push(format!(
            "... and {} more paths over {WARN_PATH} characters",
            long.len() - max
        ));
    }
    out
}

/// The notes for the files written under `root`.
pub fn notes_under(root: &Path) -> Vec<String> {
    let files: Vec<std::path::PathBuf> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    notes(files.iter().map(|p| p.as_path()), 3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_over_the_limit_is_refused_and_one_near_it_noted() {
        let ok = format!("C:/{}", "a".repeat(240));
        let near = format!("C:/{}", "a".repeat(252));
        let over = format!("C:/{}", "a".repeat(300));
        assert!(check(Path::new(&ok)).is_ok());
        assert!(check(Path::new(&near)).is_ok());
        assert!(check(Path::new(&over)).is_err());
        assert_eq!(notes([Path::new(&ok), Path::new(&near)], 3).len(), 1);
        assert_eq!(len(Path::new(r"\\?\C:\x")), 4);
        // Characters, not bytes.
        assert_eq!(len(Path::new("\u{e4}")), 1);
    }
}
