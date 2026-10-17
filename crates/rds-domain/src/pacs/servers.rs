//! The servers this device has paired with: `config_dir()/pacs-servers.json`.
//!
//! One entry per server: where it is, the certificate fingerprint pinned
//! when it was paired (or none, for a server whose certificate the system's
//! roots vouch for), the token it issued and the role that came with it.
//! The token is a secret: the file is written readable by its owner only
//! where the platform has such a thing, and nothing shows the token.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::client::Trust;
use super::protocol::Role;
use crate::settings;

/// One paired server.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerEntry {
    /// The server's own identifier ([`super::protocol::ServerInfo::server_id`]),
    /// which names its mirror folder.
    pub server_id: String,
    /// What the window calls it; the server's name when it was paired,
    /// editable.
    pub name: String,
    /// `https://host:port`.
    pub url: String,
    /// The pinned certificate (64 hex digits); empty when the system's
    /// roots verify it instead (a server behind a reverse proxy with a
    /// public certificate).
    pub fingerprint: String,
    pub token: String,
    pub role: Role,
    /// The name the server knows this device by.
    pub client_name: String,
    /// When it was paired.
    pub paired: String,
}

impl ServerEntry {
    pub fn trust(&self) -> Trust {
        if self.fingerprint.trim().is_empty() {
            Trust::System
        } else {
            Trust::Pinned(self.fingerprint.clone())
        }
    }

    /// `Ward PACS (192.168.1.20:11443)`.
    pub fn label(&self) -> String {
        let at = self
            .url
            .trim_start_matches("https://")
            .trim_end_matches('/');
        if self.name.trim().is_empty() {
            at.to_string()
        } else {
            format!("{} ({at})", self.name.trim())
        }
    }
}

/// Every paired server.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Servers {
    pub servers: Vec<ServerEntry>,
}

/// Where the list is kept.
pub fn default_path() -> PathBuf {
    settings::config_dir().join("pacs-servers.json")
}

impl Servers {
    /// Read the list; a missing file is an empty list.
    pub fn load(path: &Path) -> Result<Servers> {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                serde_json::from_str(&text).with_context(|| format!("read {}", path.display()))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Servers::default()),
            Err(e) => Err(e).with_context(|| format!("read {}", path.display())),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self).expect("plain data serialises");
        write_private(path, &text)
    }

    pub fn get(&self, server_id: &str) -> Option<&ServerEntry> {
        self.servers.iter().find(|s| s.server_id == server_id)
    }

    /// Add a server, or replace the entry of the same server (paired
    /// again).
    pub fn upsert(&mut self, entry: ServerEntry) {
        match self
            .servers
            .iter_mut()
            .find(|s| s.server_id == entry.server_id)
        {
            Some(s) => *s = entry,
            None => self.servers.push(entry),
        }
    }

    /// Forget a server. Its mirror folder is left where it is: removing
    /// that is a separate decision.
    pub fn remove(&mut self, server_id: &str) -> bool {
        let before = self.servers.len();
        self.servers.retain(|s| s.server_id != server_id);
        self.servers.len() != before
    }
}

/// Write a file that holds a secret: through a temporary file renamed into
/// place, readable by its owner only on Unix. (On Windows the per-user
/// configuration folder is the owner's already.)
pub fn write_private(path: &Path, text: &str) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut f = opts
            .open(&tmp)
            .with_context(|| format!("write {}", tmp.display()))?;
        std::io::Write::write_all(&mut f, text.as_bytes())
            .with_context(|| format!("write {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_keeps_one_entry_per_server_and_reads_back() {
        let dir = std::env::temp_dir().join("rds_pacs_servers");
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("pacs-servers.json");
        assert!(Servers::load(&path).unwrap().servers.is_empty());
        let mut s = Servers::default();
        let a = ServerEntry {
            server_id: "s1".into(),
            name: "Ward".into(),
            url: "https://10.0.0.2:11443".into(),
            fingerprint: "ab".repeat(32),
            token: "secret".into(),
            role: Role::Edit,
            ..Default::default()
        };
        s.upsert(a.clone());
        s.upsert(ServerEntry {
            role: Role::Run,
            ..a.clone()
        });
        assert_eq!(s.servers.len(), 1, "paired again replaces");
        assert_eq!(s.servers[0].role, Role::Run);
        assert_eq!(s.servers[0].label(), "Ward (10.0.0.2:11443)");
        assert!(matches!(s.servers[0].trust(), Trust::Pinned(_)));
        s.save(&path).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "the token file is the owner's only");
        }
        assert_eq!(Servers::load(&path).unwrap(), s);
        assert!(s.remove("s1"));
        assert!(!s.remove("s1"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
