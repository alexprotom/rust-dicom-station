//! The call logs of the station's two servers: one line per call, one file
//! per day, `data_dir()/<server>/audit-YYYY-MM-DD.log`.
//!
//! The MCP server (`rds-mcp`, folder `mcp`) writes a line after its
//! redactor, so the log itself can be shared; the PACS server (`rds-pacs`,
//! folder `pacs`) writes who did what to which study, never a patient's
//! name. Both use this one writer so the two logs have the same shape:
//!
//! ```text
//! 20261002T153012 open_dataset 412ms ok {...}
//! ```
//!
//! A log that cannot be written never stops the work it describes: every
//! failure here is swallowed.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::settings;

pub struct Audit {
    dir: Option<PathBuf>,
}

impl Audit {
    /// The log of the server whose folder under the data folder is `folder`
    /// (`mcp`, `pacs`). `enabled = false` gives a logger that writes nothing.
    pub fn new(enabled: bool, folder: &str) -> Audit {
        Audit {
            dir: enabled.then(|| settings::data_dir().join(folder)),
        }
    }

    /// A log written into `dir` itself; `None` writes nothing. For a server
    /// whose folders are not the station's (the test suites').
    pub fn in_dir(dir: Option<PathBuf>) -> Audit {
        Audit { dir }
    }

    /// Where the log goes, when it does.
    pub fn dir(&self) -> Option<&PathBuf> {
        self.dir.as_ref()
    }

    /// Append one line. Failures are swallowed: a log that cannot be written
    /// must not stop the work it describes.
    pub fn line(&self, tool: &str, elapsed_ms: u128, outcome: &str, detail: &str) {
        let Some(dir) = &self.dir else {
            return;
        };
        let (date, time) = crate::dicom_export::today();
        let _ = std::fs::create_dir_all(dir);
        let path = dir.join(format!("audit-{date}.log"));
        if let Ok(mut f) = OpenOptions::new().append(true).create(true).open(path) {
            let detail: String = detail.chars().take(400).collect();
            let _ = writeln!(
                f,
                "{date}T{time} {tool} {elapsed_ms}ms {outcome} {}",
                detail.replace(['\n', '\r'], " ")
            );
        }
    }

    /// The last `n` lines, oldest first, from the newest log files: what the
    /// operator's window shows. Empty when nothing was logged.
    pub fn tail(&self, n: usize) -> Vec<String> {
        let Some(dir) = &self.dir else {
            return Vec::new();
        };
        tail_of(dir, n)
    }
}

/// The last `n` lines of the `audit-*.log` files in `dir`, newest file
/// last, reading back as many days as it takes.
pub fn tail_of(dir: &Path, n: usize) -> Vec<String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|f| f.to_str())
                        .is_some_and(|f| f.starts_with("audit-") && f.ends_with(".log"))
                })
                .collect()
        })
        .unwrap_or_default();
    // The date is in the name, so name order is time order.
    files.sort();
    let mut out: Vec<String> = Vec::new();
    for f in files.iter().rev() {
        let Ok(text) = std::fs::read_to_string(f) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let take = (n - out.len()).min(lines.len());
        let mut chunk: Vec<String> = lines[lines.len() - take..]
            .iter()
            .map(|l| l.to_string())
            .collect();
        chunk.append(&mut out);
        out = chunk;
        if out.len() >= n {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tail_reads_back_across_days_oldest_first() {
        let dir = std::env::temp_dir().join("rds_audit_tail");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("audit-20261001.log"), "a\nb\nc\n").unwrap();
        std::fs::write(dir.join("audit-20261002.log"), "d\ne\n").unwrap();
        std::fs::write(dir.join("other.txt"), "x\n").unwrap();
        assert_eq!(tail_of(&dir, 3), vec!["c", "d", "e"]);
        assert_eq!(tail_of(&dir, 10), vec!["a", "b", "c", "d", "e"]);
        assert!(tail_of(&dir.join("missing"), 5).is_empty());
        let log = Audit::in_dir(Some(dir.clone()));
        log.line("pull", 12, "ok", "line\nbreak");
        assert!(log.tail(1)[0].ends_with("pull 12ms ok line break"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
