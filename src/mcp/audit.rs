//! The call log: `data_dir()/mcp/audit-YYYY-MM-DD.log`, one line per call,
//! written after the redactor so the log itself can be shared.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use crate::settings;

pub struct Audit {
    dir: Option<PathBuf>,
}

impl Audit {
    /// `enabled = false` gives a logger that writes nothing.
    pub fn new(enabled: bool) -> Audit {
        Audit {
            dir: enabled.then(|| settings::data_dir().join("mcp")),
        }
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
}
