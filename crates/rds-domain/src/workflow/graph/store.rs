//! Where workflow files live, and the examples the program ships.
//!
//! A user's workflows are kept in `<data folder>/user_data/workflows` (the
//! same `user_data` tree the Structure editor keeps its axes in), one
//! `.rdsflow` file each; any other folder works as well, since the viewer
//! opens and saves through the ordinary file dialog, which merely starts
//! there. The five most recently saved or opened files are remembered in
//! the settings file (`recent_workflows`), which is what *Workflows ▸
//! Recent* lists.
//!
//! The examples are compiled in (`include_str!`), so they are there in
//! every installed copy; opening one gives an unsaved workflow to adapt.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use super::Workflow;

/// The file extension of a workflow.
pub const EXTENSION: &str = "rdsflow";

/// How many recent files are remembered.
pub const RECENT_MAX: usize = 5;

/// `<data folder>/user_data/workflows`, made on demand.
pub fn user_dir() -> PathBuf {
    let dir = crate::settings::data_dir()
        .join("user_data")
        .join("workflows");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Where runs write when a workflow names no output root:
/// `<data folder>/workflow_runs`.
pub fn default_runs_dir() -> PathBuf {
    crate::settings::data_dir().join("workflow_runs")
}

/// Read a workflow file.
pub fn load(path: &Path) -> Result<Workflow> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    Workflow::from_json(&text).with_context(|| format!("open {}", path.display()))
}

/// Every workflow file in `dir` (not its subfolders) that reads, sorted by
/// file name. Files that do not read as workflows are left out.
pub fn list(dir: &Path) -> Vec<(PathBuf, Workflow)> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == EXTENSION))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|p| load(&p).ok().map(|wf| (p, wf)))
        .collect()
}

/// Write a workflow file, adding the extension when the name has none.
/// Returns the path written.
pub fn save(wf: &Workflow, path: &Path) -> Result<PathBuf> {
    let path = if path.extension().is_none() {
        path.with_extension(EXTENSION)
    } else {
        path.to_path_buf()
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    std::fs::write(&path, wf.to_json()).with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Put `path` at the head of a recent-files list: once, newest first, at
/// most [`RECENT_MAX`] long.
pub fn remember(recent: &mut Vec<PathBuf>, path: &Path) {
    recent.retain(|p| p != path);
    recent.insert(0, path.to_path_buf());
    recent.truncate(RECENT_MAX);
}

/// An example that ships with the program.
pub struct Example {
    pub name: &'static str,
    pub blurb: &'static str,
    json: &'static str,
}

impl Example {
    pub fn workflow(&self) -> Workflow {
        Workflow::from_json(self.json).expect("the shipped examples are valid")
    }
}

/// The examples, for *Workflows ▸ Examples*.
pub const EXAMPLES: &[Example] = &[
    Example {
        name: "Cardiac CT and 4DCT: heart-anchored target motion",
        blurb: "Segment the heart on a cardiac CT and on every phase of a 4DCT, carry the \
                target from the CT onto the phases anchored on the heart, measure the motion \
                of the heart and the target, build the ITV, write everything out.",
        json: include_str!("examples/heart_anchored_target_motion.rdsflow"),
    },
    Example {
        name: "4DCT: target motion and ITV",
        blurb: "Find the target on the phases of a 4DCT, measure its motion and build the ITV.",
        json: include_str!("examples/fourd_target_itv.rdsflow"),
    },
];

/// A file name for a workflow name: what the save dialog proposes.
pub fn file_name_for(name: &str) -> String {
    let stem = super::safe_name(if name.trim().is_empty() {
        "workflow"
    } else {
        name.trim()
    });
    format!("{stem}.{EXTENSION}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_examples_open_and_pass_their_own_checks() {
        for e in EXAMPLES {
            let wf = e.workflow();
            let f = wf.check();
            // A shipped example may not say which folder to read (that is
            // the run dialog's question), but nothing else may be wrong.
            let other: Vec<_> = f
                .errors
                .iter()
                .filter(|(_, m)| !m.contains("no folder is given"))
                .collect();
            assert!(other.is_empty(), "{}: {other:?}", e.name);
            assert!(wf.order().is_ok());
            // Every example round-trips unchanged.
            let back = Workflow::from_json(&wf.to_json()).unwrap();
            let mut expect = wf.clone();
            expect.links.sort();
            assert_eq!(back, expect, "{}", e.name);
        }
    }

    #[test]
    fn the_recent_list_is_newest_first_without_repeats() {
        let mut r = Vec::new();
        for i in 0..7 {
            remember(&mut r, Path::new(&format!("/w/{i}.rdsflow")));
        }
        remember(&mut r, Path::new("/w/4.rdsflow"));
        assert_eq!(r.len(), RECENT_MAX);
        assert_eq!(r[0], PathBuf::from("/w/4.rdsflow"));
        assert_eq!(r[1], PathBuf::from("/w/6.rdsflow"));
        assert_eq!(r.iter().filter(|p| p.ends_with("4.rdsflow")).count(), 1);
    }

    #[test]
    fn saving_adds_the_extension_and_reads_back() {
        let dir = std::env::temp_dir().join(format!("rds-wf-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let wf = EXAMPLES[0].workflow();
        let path = save(&wf, &dir.join("mine")).unwrap();
        assert_eq!(path.extension().unwrap(), EXTENSION);
        assert_eq!(load(&path).unwrap().name, wf.name);
        assert_eq!(file_name_for("a/b"), "a_b.rdsflow");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
