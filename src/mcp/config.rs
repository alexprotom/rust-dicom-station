//! The operator's configuration: `config_dir()/mcp.toml`.
//!
//! Everything a tool call may not decide for itself lives here - which
//! folders may be read, where results go, what happens to a dataset that
//! still names its patient - and none of it can be changed through the
//! protocol. A missing file is a working configuration with no roots, which
//! means no dataset can be opened until someone writes one.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::nn::device::DevicePref;
use crate::settings;

/// What `open_dataset` does with a dataset that still carries identifying
/// tags. There is no `off`: even under `allow`, no tool returns a patient
/// name.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PhiPolicy {
    /// Close it again and say which tags (never their values) are the
    /// reason. The default.
    Refuse,
    /// Open it with the identifying values replaced in memory by the alias.
    Redact,
    /// As `redact` for everything that leaves the process, but exports may
    /// keep the original identifiers when asked.
    Allow,
}

impl PhiPolicy {
    pub fn label(self) -> &'static str {
        match self {
            PhiPolicy::Refuse => "refuse",
            PhiPolicy::Redact => "redact",
            PhiPolicy::Allow => "allow",
        }
    }
}

/// The file, as written by the operator.
#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Folders that may be opened. Nothing outside them is read.
    pub roots: Vec<PathBuf>,
    /// The one folder results are written into.
    pub output_dir: Option<PathBuf>,
    pub phi_policy: PhiPolicy,
    /// Empty: the viewer's model folder (`settings::default_models_dir`).
    pub models_dir: Option<PathBuf>,
    /// Model weights are the server's only use of the network; off by
    /// default so a missing model is an error, never a download.
    pub allow_model_download: bool,
    /// `auto`, `gpu` or `cpu`.
    pub device: String,
    /// Datasets that may be open at once.
    pub max_open_datasets: usize,
    /// A computing call that runs longer than this is cancelled.
    pub job_timeout_minutes: u64,
    /// Write `data_dir()/mcp/audit-YYYY-MM-DD.log`.
    pub audit_log: bool,
    /// The viewer executable `open_in_viewer` launches; empty finds it
    /// beside `rds-mcp`.
    pub viewer_exe: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            roots: Vec::new(),
            output_dir: None,
            phi_policy: PhiPolicy::Refuse,
            models_dir: None,
            allow_model_download: false,
            device: "auto".into(),
            max_open_datasets: 4,
            job_timeout_minutes: 60,
            audit_log: true,
            viewer_exe: None,
        }
    }
}

/// Where the file lives by default.
pub fn default_path() -> PathBuf {
    settings::config_dir().join("mcp.toml")
}

impl Config {
    /// Parse the file's text.
    pub fn parse(text: &str) -> Result<Config> {
        let c: Config = toml::from_str(text).context("mcp.toml")?;
        c.check()?;
        Ok(c)
    }

    /// Read `path`; a missing file is the default configuration.
    pub fn load(path: &Path) -> Result<Config> {
        match std::fs::read_to_string(path) {
            Ok(text) => Config::parse(&text).with_context(|| path.display().to_string()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
            Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
        }
    }

    fn check(&self) -> Result<()> {
        if self.max_open_datasets == 0 {
            bail!("max_open_datasets must be at least 1");
        }
        if !["auto", "gpu", "cpu"].contains(&self.device.as_str()) {
            bail!("device must be auto, gpu or cpu (got '{}')", self.device);
        }
        if let Some(out) = &self.output_dir {
            if out.as_os_str().is_empty() {
                bail!("output_dir must not be empty");
            }
        }
        Ok(())
    }

    pub fn device_pref(&self) -> DevicePref {
        match self.device.as_str() {
            "gpu" => DevicePref::Gpu,
            "cpu" => DevicePref::Cpu,
            _ => DevicePref::Auto,
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        self.models_dir
            .clone()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(settings::default_models_dir)
    }

    /// The output folder, or the reason there is none.
    pub fn output_dir(&self) -> Result<&Path> {
        self.output_dir
            .as_deref()
            .context("no output_dir is configured in mcp.toml; nothing can be written")
    }

    /// The folders a client may name: the roots, then the output folder
    /// (results are re-openable), each with its label.
    pub fn readable(&self) -> Vec<(&Path, String)> {
        let mut out: Vec<(&Path, String)> = self
            .roots
            .iter()
            .enumerate()
            .map(|(i, r)| (r.as_path(), format!("root{}", i + 1)))
            .collect();
        if let Some(o) = &self.output_dir {
            out.push((o.as_path(), "output".to_string()));
        }
        out
    }

    /// Resolve a path the client sent against the readable folders.
    ///
    /// The path is canonicalized first (so `..` and symlinks cannot step
    /// outside), then must start with one canonicalized folder. The returned
    /// pair is the resolved path and the label of the folder it is under.
    pub fn resolve_input(&self, given: &Path) -> Result<(PathBuf, String)> {
        let readable = self.readable();
        if readable.is_empty() {
            bail!("no roots are configured in mcp.toml; no dataset can be opened");
        }
        let real = given
            .canonicalize()
            .with_context(|| format!("'{}' does not exist", given.display()))?;
        for (root, label) in readable {
            let Ok(r) = root.canonicalize() else {
                continue;
            };
            if real.starts_with(&r) {
                return Ok((real, label));
            }
        }
        bail!(
            "'{}' is outside the configured roots ({} configured)",
            given.display(),
            self.roots.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_refuse_and_own_nothing() {
        let c = Config::parse("").unwrap();
        assert_eq!(c.phi_policy, PhiPolicy::Refuse);
        assert!(c.roots.is_empty());
        assert!(c.output_dir().is_err());
        assert!(!c.allow_model_download);
    }

    #[test]
    fn a_full_file_parses() {
        let c = Config::parse(
            r#"
roots = ["/data/a", "/data/b"]
output_dir = "/data/out"
phi_policy = "redact"
device = "cpu"
max_open_datasets = 2
allow_model_download = true
"#,
        )
        .unwrap();
        assert_eq!(c.roots.len(), 2);
        assert_eq!(c.phi_policy, PhiPolicy::Redact);
        assert_eq!(c.device_pref(), DevicePref::Cpu);
        assert_eq!(c.max_open_datasets, 2);
    }

    #[test]
    fn unknown_keys_and_bad_values_are_refused() {
        assert!(Config::parse("phi_policy = \"off\"").is_err());
        assert!(Config::parse("phi_polcy = \"refuse\"").is_err());
        assert!(Config::parse("device = \"cuda\"").is_err());
        assert!(Config::parse("max_open_datasets = 0").is_err());
    }

    #[test]
    fn paths_outside_the_roots_are_refused() {
        let dir = std::env::temp_dir().join("rds_mcp_cfg_roots");
        let _ = std::fs::create_dir_all(dir.join("in/sub"));
        let _ = std::fs::create_dir_all(dir.join("out"));
        let c = Config {
            roots: vec![dir.join("in")],
            ..Config::default()
        };
        assert_eq!(c.resolve_input(&dir.join("in/sub")).unwrap().1, "root1");
        assert!(c.resolve_input(&dir.join("out")).is_err());
        let with_out = Config {
            output_dir: Some(dir.join("out")),
            ..c.clone()
        };
        assert_eq!(
            with_out.resolve_input(&dir.join("out")).unwrap().1,
            "output"
        );
        assert!(c.resolve_input(&dir.join("in/sub/../../out")).is_err());
        assert!(Config::default().resolve_input(&dir.join("in")).is_err());
    }
}
