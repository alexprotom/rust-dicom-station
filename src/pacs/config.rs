//! The server's configuration: `config_dir()/pacs.toml`.
//!
//! Written by the operator, by hand or through the viewer's *Settings ▶ PACS
//! server* window, and read by `rds-pacs` when it starts. Nothing in it can
//! be changed by a client. A missing file is a working configuration: the
//! default port on every interface, the station's own archive, tasks on,
//! no model downloads.
//!
//! ```toml
//! name = ""                  # what clients see; empty: this machine's name
//! bind = "0.0.0.0"           # 127.0.0.1 behind a reverse proxy
//! port = 11443
//! archive_dir = ""           # empty: the viewer's archive (Tools > PACS)
//! advertise = []             # extra names the certificate carries
//! tls_cert = ""              # PEM files of a certificate of your own;
//! tls_key = ""               #   empty: a self-signed one, made once
//! behind_proxy = false       # take the client's address from X-Forwarded-For
//! tasks = true               # run the workflows clients hand in
//! max_upload_mb = 2048
//! max_queued_tasks = 16
//! task_timeout_minutes = 240
//! pairing_minutes = 10
//! models_dir = ""            # empty: the viewer's model folder
//! allow_model_download = false
//! volume_cache_mb = 4096
//! workflows_dir = ""         # empty: the viewer's workflow folder
//! audit_log = true           # data folder/pacs/audit-YYYY-MM-DD.log
//! ```
//!
//! Always compiled: the viewer reads and writes the file without the
//! server's own code.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::settings;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// What clients see the server as. Empty: this machine's name.
    pub name: String,
    /// The address to listen on: every interface, or the loopback only
    /// when a reverse proxy on the same machine takes the outside.
    pub bind: String,
    pub port: u16,
    /// The archive served. Empty: the viewer's (*Tools ▶ PACS*), so the
    /// server and the window on the same machine share one archive.
    pub archive_dir: String,
    /// Names and addresses put into the self-signed certificate besides
    /// this machine's (a dynamic DNS name, a VPN address). Only a client
    /// that verifies through the system's roots looks at them; a pinned
    /// client does not.
    pub advertise: Vec<String>,
    /// A certificate and key of the operator's own (PEM). Empty: the
    /// self-signed certificate made on first start.
    pub tls_cert: String,
    pub tls_key: String,
    /// The server sits behind a reverse proxy on this machine: a request
    /// from the loopback carrying `X-Forwarded-For` is logged and
    /// rate-limited under that address.
    pub behind_proxy: bool,
    /// Accept tasks: workflows the clients hand in, run on this machine.
    pub tasks: bool,
    pub max_upload_mb: u64,
    pub max_queued_tasks: usize,
    pub task_timeout_minutes: u64,
    /// How long a pairing code is good for unless the operator says
    /// otherwise.
    pub pairing_minutes: u64,
    /// Where the engines of a task find their weights. Empty: the viewer's.
    pub models_dir: String,
    /// May a task download weights it does not have? Off: a missing model
    /// is an error the task reports.
    pub allow_model_download: bool,
    /// Megabytes of image volumes a task keeps between its steps.
    pub volume_cache_mb: usize,
    /// Where the saved workflows the server offers are. Empty: the viewer's
    /// workflow folder.
    pub workflows_dir: String,
    /// Write the call log.
    pub audit_log: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            name: String::new(),
            bind: "0.0.0.0".into(),
            port: super::DEFAULT_PORT,
            archive_dir: String::new(),
            advertise: Vec::new(),
            tls_cert: String::new(),
            tls_key: String::new(),
            behind_proxy: false,
            tasks: true,
            max_upload_mb: 2048,
            max_queued_tasks: 16,
            task_timeout_minutes: 240,
            pairing_minutes: 10,
            models_dir: String::new(),
            allow_model_download: false,
            volume_cache_mb: 4096,
            workflows_dir: String::new(),
            audit_log: true,
        }
    }
}

/// Where the file lives.
pub fn default_path() -> PathBuf {
    settings::config_dir().join("pacs.toml")
}

const HEADER: &str = "\
# Rust DICOM Station - PACS server (rds-pacs).
# Read when the server starts; see docs/pacs-server.md for every key.
# Settings > PACS server in the viewer writes this file too.

";

impl Config {
    pub fn parse(text: &str) -> Result<Config> {
        let c: Config = toml::from_str(text).context("pacs.toml")?;
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

    /// Write the file (what the viewer's window does when its settings are
    /// saved). The comments of a hand-written file are not kept; the
    /// header says where the keys are explained.
    pub fn save(&self, path: &Path) -> Result<()> {
        self.check()?;
        let body = toml::to_string(self).context("write pacs.toml")?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        std::fs::write(path, format!("{HEADER}{body}"))
            .with_context(|| format!("write {}", path.display()))
    }

    fn check(&self) -> Result<()> {
        if self.port == 0 {
            bail!("port must not be 0");
        }
        if self.bind.trim().parse::<std::net::IpAddr>().is_err() {
            bail!(
                "bind must be an IP address (0.0.0.0, 127.0.0.1, ::), not '{}'",
                self.bind
            );
        }
        if self.tls_cert.trim().is_empty() != self.tls_key.trim().is_empty() {
            bail!("tls_cert and tls_key go together: give both or neither");
        }
        if self.max_upload_mb == 0 {
            bail!("max_upload_mb must be at least 1");
        }
        if self.max_queued_tasks == 0 {
            bail!("max_queued_tasks must be at least 1");
        }
        Ok(())
    }

    /// The name clients see.
    pub fn server_name(&self) -> String {
        if self.name.trim().is_empty() {
            super::this_device_name()
        } else {
            self.name.trim().to_string()
        }
    }

    /// The archive folder served.
    pub fn archive_root(&self) -> PathBuf {
        if self.archive_dir.trim().is_empty() {
            let prefs = settings::load();
            prefs
                .archive_dir
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(crate::archive::default_root)
        } else {
            PathBuf::from(self.archive_dir.trim())
        }
    }

    pub fn models_dir(&self) -> PathBuf {
        if self.models_dir.trim().is_empty() {
            settings::load()
                .models_dir
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(settings::default_models_dir)
        } else {
            PathBuf::from(self.models_dir.trim())
        }
    }

    pub fn workflows_dir(&self) -> PathBuf {
        if self.workflows_dir.trim().is_empty() {
            crate::workflow::graph::store::user_dir()
        } else {
            PathBuf::from(self.workflows_dir.trim())
        }
    }

    /// The operator's own certificate files, when there are any.
    pub fn own_certificate(&self) -> Option<(PathBuf, PathBuf)> {
        (!self.tls_cert.trim().is_empty()).then(|| {
            (
                PathBuf::from(self.tls_cert.trim()),
                PathBuf::from(self.tls_key.trim()),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_serve_the_station_archive_on_the_default_port() {
        let c = Config::parse("").unwrap();
        assert_eq!(c.port, super::super::DEFAULT_PORT);
        assert_eq!(c.bind, "0.0.0.0");
        assert!(c.tasks);
        assert!(!c.allow_model_download);
        assert!(c.own_certificate().is_none());
    }

    #[test]
    fn bad_values_and_unknown_keys_are_refused() {
        assert!(Config::parse("port = 0").is_err());
        assert!(Config::parse("bind = \"everywhere\"").is_err());
        assert!(Config::parse("tls_cert = \"a.pem\"").is_err());
        assert!(Config::parse("prot = 1").is_err());
        assert!(Config::parse("max_upload_mb = 0").is_err());
    }

    #[test]
    fn what_the_window_saves_reads_back() {
        let dir = std::env::temp_dir().join("rds_pacs_config_rt");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("pacs.toml");
        let c = Config {
            name: "Ward PACS".into(),
            port: 12000,
            advertise: vec!["pacs.example.org".into(), "100.64.0.7".into()],
            allow_model_download: true,
            ..Config::default()
        };
        c.save(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("# Rust DICOM Station"));
        assert_eq!(Config::load(&path).unwrap(), c);
        assert_eq!(
            Config::load(&dir.join("missing.toml")).unwrap(),
            Config::default()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
