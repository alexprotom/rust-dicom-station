//! The PACS server on this machine, as the viewer sees it.
//!
//! The viewer never runs the server itself: it starts `rds-pacs` as a
//! process of its own (so the server outlives the window), learns that it
//! runs from the file the server writes when it starts listening
//! (`running.json`), and talks to it like any client - over the loopback,
//! with the fingerprint the server wrote down and the local operator's
//! token, which the server leaves in its state folder readable by its
//! owner only. That is why the operator's window needs no pairing.
//!
//! ## The folders
//!
//! ```text
//! <config folder>/pacs.toml           the operator's configuration
//! <config folder>/pacs/               the server's state
//!     server.crt, server.key          the self-signed certificate (PEM)
//!     identity.json                   the server's id
//!     clients.json                    paired clients, tokens as hashes
//!     local-admin.token               the local operator's token
//!     running.json                    written while the server listens
//! <data folder>/pacs/                 the server's work
//!     audit-YYYY-MM-DD.log            who did what
//!     server.log                      what the server printed, when the
//!                                     viewer started it
//!     tasks/<id>/                     one folder per task: TASK.json, run/
//!     incoming/                       uploads while they arrive
//! <data folder>/pacs-mirror/<id>/     a client's copy of a server
//! <config folder>/pacs-servers.json   a client's paired servers
//! ```

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::client::{Remote, Trust};
use super::protocol::{ClientInfo, Role};
use super::servers::ServerEntry;
use crate::settings;

/// The server's folders. [`Paths::station`] for the real ones; the test
/// suites give a server folders of its own with [`Paths::at`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paths {
    /// Certificate, identity, clients, tokens: `<config folder>/pacs`.
    pub state: PathBuf,
    /// Tasks, uploads, logs: `<data folder>/pacs`.
    pub data: PathBuf,
}

impl Paths {
    pub fn station() -> Paths {
        Paths {
            state: settings::config_dir().join("pacs"),
            data: settings::data_dir().join("pacs"),
        }
    }

    /// Everything under one folder: `<root>/state`, `<root>/data`.
    pub fn at(root: &Path) -> Paths {
        Paths {
            state: root.join("state"),
            data: root.join("data"),
        }
    }

    pub fn cert(&self) -> PathBuf {
        self.state.join("server.crt")
    }
    pub fn key(&self) -> PathBuf {
        self.state.join("server.key")
    }
    pub fn identity(&self) -> PathBuf {
        self.state.join("identity.json")
    }
    pub fn clients(&self) -> PathBuf {
        self.state.join("clients.json")
    }
    pub fn local_token(&self) -> PathBuf {
        self.state.join("local-admin.token")
    }
    pub fn running(&self) -> PathBuf {
        self.state.join("running.json")
    }
    pub fn tasks(&self) -> PathBuf {
        self.data.join("tasks")
    }
    pub fn incoming(&self) -> PathBuf {
        self.data.join("incoming")
    }
    pub fn log(&self) -> PathBuf {
        self.data.join("server.log")
    }
}

/// What a listening server writes into its state folder, and removes when
/// it stops. A file left behind by a server that died is told apart by
/// asking the port.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Running {
    pub pid: u32,
    pub name: String,
    pub server_id: String,
    pub version: String,
    pub bind: String,
    pub port: u16,
    pub fingerprint: String,
    pub started: String,
    pub archive: String,
    /// The addresses other machines can try, with the port.
    pub addresses: Vec<String>,
}

impl Running {
    pub fn read(paths: &Paths) -> Option<Running> {
        let text = std::fs::read_to_string(paths.running()).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn write(&self, paths: &Paths) -> Result<()> {
        std::fs::create_dir_all(&paths.state)
            .with_context(|| format!("create {}", paths.state.display()))?;
        let text = serde_json::to_string_pretty(self).expect("plain data serialises");
        std::fs::write(paths.running(), text)
            .with_context(|| format!("write {}", paths.running().display()))
    }

    pub fn clear(paths: &Paths) {
        let _ = std::fs::remove_file(paths.running());
    }

    /// Where this machine reaches the server: the loopback, unless it
    /// listens on one particular address.
    pub fn local_url(&self) -> String {
        let ip: IpAddr = self.bind.parse().unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST));
        let ip = match ip {
            IpAddr::V4(v) if v.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V6(v) if v.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
            other => other,
        };
        format!("https://{}", SocketAddr::new(ip, self.port))
    }
}

/// The local operator's token, when a server has run on this machine.
pub fn local_token(paths: &Paths) -> Option<String> {
    std::fs::read_to_string(paths.local_token())
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

/// The server on this machine as a paired server: the loopback, the
/// fingerprint it wrote down, the operator's token. `None` when it does
/// not run (or never ran).
pub fn local_entry(paths: &Paths) -> Option<ServerEntry> {
    let run = Running::read(paths)?;
    let token = local_token(paths)?;
    Some(ServerEntry {
        server_id: run.server_id.clone(),
        name: format!("{} (this computer)", run.name),
        url: run.local_url(),
        fingerprint: run.fingerprint.clone(),
        token,
        role: Role::Admin,
        client_name: "local operator".into(),
        paired: run.started.clone(),
    })
}

/// A client for the server on this machine, once it answers.
pub fn connect(paths: &Paths) -> Result<(Remote, Running)> {
    let entry = local_entry(paths).context("the PACS server is not running on this computer")?;
    let run = Running::read(paths).context("the PACS server is not running")?;
    let remote = Remote::new(
        &entry.url,
        Trust::Pinned(entry.fingerprint.clone()),
        Some(entry.token),
    )?;
    remote.info()?;
    Ok((remote, run))
}

/// Is `rds-pacs` part of this installation?
pub fn installed() -> bool {
    settings::pacs_exe_path().is_file()
}

/// Start `rds-pacs serve` as a process of its own, detached from this one
/// so it keeps running when the viewer closes. What it prints goes to
/// `<data folder>/pacs/server.log`.
pub fn start(paths: &Paths) -> Result<()> {
    let exe = settings::pacs_exe_path();
    if !exe.is_file() {
        bail!(
            "the PACS server is not installed ({} was not found); it is an optional component \
             of the installer, or `cargo build --release --features pacs-server`",
            exe.display()
        );
    }
    std::fs::create_dir_all(&paths.data)
        .with_context(|| format!("create {}", paths.data.display()))?;
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(paths.log())
        .with_context(|| format!("write {}", paths.log().display()))?;
    let log2 = log.try_clone().context("the server log")?;
    // An AppImage's own files vanish with its mount when the viewer
    // exits, so the server is started through the AppImage itself.
    let mut cmd = match std::env::var_os("APPIMAGE").filter(|v| !v.is_empty()) {
        Some(image) if cfg!(target_os = "linux") => {
            let mut c = std::process::Command::new(image);
            c.arg("pacs");
            c
        }
        _ => std::process::Command::new(&exe),
    };
    cmd.arg("serve")
        .stdin(std::process::Stdio::null())
        .stdout(log)
        .stderr(log2);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // A process group of its own: closing a terminal the viewer was
        // started from does not take the server with it.
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        cmd.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
    }
    cmd.spawn()
        .with_context(|| format!("start {}", exe.display()))?;
    Ok(())
}

/// The paired stations of a server that is not running, straight from
/// its `clients.json` (what `rds-pacs clients` and the operator's window
/// show while it is stopped).
pub fn clients_offline(paths: &Paths) -> Vec<ClientInfo> {
    let Ok(text) = std::fs::read_to_string(paths.clients()) else {
        return Vec::new();
    };
    let v: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
    v.get("clients")
        .and_then(|c| c.as_array())
        .map(|list| {
            list.iter()
                .filter_map(|c| serde_json::from_value::<ClientInfo>(c.clone()).ok())
                .collect()
        })
        .unwrap_or_default()
}

/// Forget a station of a server that is not running. (A running server
/// is asked instead: it holds the list in memory.)
pub fn revoke_offline(paths: &Paths, name: &str) -> Result<bool> {
    let Ok(text) = std::fs::read_to_string(paths.clients()) else {
        return Ok(false);
    };
    let mut v: serde_json::Value = serde_json::from_str(&text).context("read clients.json")?;
    let Some(list) = v.get_mut("clients").and_then(|c| c.as_array_mut()) else {
        return Ok(false);
    };
    let before = list.len();
    list.retain(|c| c.get("name").and_then(|n| n.as_str()) != Some(name));
    let gone = list.len() != before;
    if gone {
        super::servers::write_private(
            &paths.clients(),
            &serde_json::to_string_pretty(&v).expect("plain data serialises"),
        )?;
    }
    Ok(gone)
}

/// Run `rds-pacs` with `args` and wait for it: what the viewer does for
/// the things only the server's own code can do (a new certificate).
/// Returns what it printed on standard output.
pub fn run_cli(args: &[&str]) -> Result<String> {
    let exe = settings::pacs_exe_path();
    if !exe.is_file() {
        bail!(
            "the PACS server is not installed ({} was not found)",
            exe.display()
        );
    }
    let mut cmd = match std::env::var_os("APPIMAGE").filter(|v| !v.is_empty()) {
        Some(image) if cfg!(target_os = "linux") => {
            let mut c = std::process::Command::new(image);
            c.arg("pacs");
            c
        }
        _ => std::process::Command::new(&exe),
    };
    let out = cmd
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .with_context(|| format!("run {}", exe.display()))?;
    if !out.status.success() {
        bail!(
            "rds-pacs {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The addresses other machines are likely to reach this one at: the one
/// the system would send from (IPv4, then IPv6). Found without sending
/// anything - a UDP socket is "connected" to a public address and asked
/// which local address it got. A VPN's address is not among them unless
/// the VPN carries the default route; the VPN's own app shows it.
pub fn addresses() -> Vec<IpAddr> {
    let mut out = Vec::new();
    let probe = |bind: &str, to: &str| -> Option<IpAddr> {
        let s = UdpSocket::bind(bind).ok()?;
        s.connect(to).ok()?;
        let ip = s.local_addr().ok()?.ip();
        (!ip.is_loopback() && !ip.is_unspecified()).then_some(ip)
    };
    if let Some(ip) = probe("0.0.0.0:0", "192.0.2.1:9") {
        out.push(ip);
    }
    if let Some(ip) = probe("[::]:0", "[2001:db8::1]:9") {
        out.push(ip);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_local_url_is_the_loopback_unless_one_address_is_bound() {
        let mut r = Running {
            bind: "0.0.0.0".into(),
            port: 11443,
            ..Default::default()
        };
        assert_eq!(r.local_url(), "https://127.0.0.1:11443");
        r.bind = "::".into();
        assert_eq!(r.local_url(), "https://[::1]:11443");
        r.bind = "192.168.1.20".into();
        assert_eq!(r.local_url(), "https://192.168.1.20:11443");
    }

    #[test]
    fn a_stopped_server_lists_and_revokes_from_its_file() {
        let root = std::env::temp_dir().join("rds_pacs_local_offline");
        let _ = std::fs::remove_dir_all(&root);
        let paths = Paths::at(&root);
        assert!(clients_offline(&paths).is_empty());
        std::fs::create_dir_all(&paths.state).unwrap();
        std::fs::write(
            paths.clients(),
            r#"{"clients":[{"name":"ipad","role":"edit","token_hash":"x","paired":"p"},
                           {"name":"laptop","role":"view","token_hash":"y"}]}"#,
        )
        .unwrap();
        let list = clients_offline(&paths);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].role, Role::Edit);
        assert!(revoke_offline(&paths, "ipad").unwrap());
        assert!(!revoke_offline(&paths, "ipad").unwrap());
        assert_eq!(clients_offline(&paths).len(), 1);
        let text = std::fs::read_to_string(paths.clients()).unwrap();
        assert!(
            text.contains("token_hash"),
            "the other records are kept whole"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_server_that_never_ran_is_not_an_entry() {
        let root = std::env::temp_dir().join("rds_pacs_local_none");
        let _ = std::fs::remove_dir_all(&root);
        let paths = Paths::at(&root);
        assert!(local_entry(&paths).is_none());
        let run = Running {
            name: "Ward".into(),
            server_id: "abc".into(),
            bind: "0.0.0.0".into(),
            port: 12000,
            fingerprint: "ab".repeat(32),
            ..Default::default()
        };
        run.write(&paths).unwrap();
        assert!(local_entry(&paths).is_none(), "no token, no entry");
        std::fs::write(paths.local_token(), "tok\n").unwrap();
        let e = local_entry(&paths).unwrap();
        assert_eq!(e.role, Role::Admin);
        assert_eq!(e.token, "tok");
        assert_eq!(e.url, "https://127.0.0.1:12000");
        Running::clear(&paths);
        assert!(local_entry(&paths).is_none());
        let _ = std::fs::remove_dir_all(&root);
    }
}
