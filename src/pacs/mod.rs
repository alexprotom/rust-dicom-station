//! The PACS server and its client: the station's archive
//! ([`crate::archive`]) served to other stations over HTTPS.
//!
//! The archive is a folder that one station owns. This module puts a door
//! on it: `rds-pacs`, a headless executable built with the cargo feature
//! `pacs-server`, serves the folder to other stations, and every build of
//! the viewer - the Android and iOS ones included - is a client that can
//! pair with such a server, keep a copy of its studies, work on them, give
//! back what was drawn on them, and hand the server workflows to run.
//!
//! ## What is where
//!
//! Always compiled (the client side, and what the viewer needs to look
//! after a server on its own machine):
//!
//! * [`protocol`] - the JSON both sides exchange, as serde structs both
//!   sides share, so there is no second description of the protocol;
//! * [`client`] - [`client::Remote`], a paired server as the viewer sees it:
//!   HTTPS through `ureq`, with the certificate pinned;
//! * [`servers`] - the servers this device has paired with
//!   (`pacs-servers.json` in the configuration folder);
//! * [`mirror`] - the local copy of a server's studies: an archive in the
//!   same layout under `<data folder>/pacs-mirror/<server id>/`, an outbox
//!   for what could not be sent yet, and sync as set differences of SOP
//!   Instance UIDs;
//! * [`config`] - the server's `pacs.toml`, which the viewer's
//!   *Settings ▶ PACS server* window reads and writes;
//! * [`local`] - the server on this machine as the viewer sees it: its
//!   folders, whether it runs, how to start it.
//!
//! With the feature `pacs-server` (the `rds-pacs` executable and the test
//! suites):
//!
//! * `tls` - the self-signed certificate, made on first start, or the
//!   operator's own;
//! * `auth` - pairing codes, the clients' tokens (kept hashed), roles;
//! * `server` - the HTTPS routes (axum over tokio), a thin door: every
//!   request that touches data goes to [`crate::archive::Archive`] or to
//!   the task queue on a blocking thread;
//! * `tasks` - the queue of workflow runs the clients hand in, run one at
//!   a time with [`crate::workflow::graph::exec`].
//!
//! ## Security in one paragraph
//!
//! TLS always, also on the loopback. The server's certificate is pinned by
//! its SHA-256 fingerprint when a client pairs (the SSH model: no
//! certificate authority anywhere, and it works on a LAN, through a VPN and
//! through a forwarded port alike), or verified through the system's roots
//! for an operator who has a real certificate. A client is let in by a
//! pairing code the operator made (short-lived, one use, with a role) and
//! from then on by a bearer token the server keeps only as a hash. See
//! `docs/pacs-server.md`.

pub mod client;
pub mod config;
pub mod local;
pub mod mirror;
pub mod protocol;
pub mod servers;

#[cfg(feature = "pacs-server")]
pub mod auth;
#[cfg(feature = "pacs-server")]
pub mod server;
#[cfg(feature = "pacs-server")]
pub mod tasks;
#[cfg(feature = "pacs-server")]
pub mod tls;

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Result};
use sha2::{Digest, Sha256};

pub use protocol::{Role, API_VERSION};

/// The port a server listens on unless `pacs.toml` says otherwise. Chosen
/// only to clash with nothing common.
pub const DEFAULT_PORT: u16 = 11443;

/// The SHA-256 of a certificate (DER), as 64 lowercase hex digits.
pub fn fingerprint(der: &[u8]) -> String {
    hex(&Sha256::digest(der))
}

/// Lowercase hex of some bytes.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// A fingerprint as a person may have typed or pasted it - `sha256:` in
/// front, upper case, spaces or colons between the digits - in the one
/// form it is compared in. `None` when it is not 64 hex digits.
pub fn normalize_fingerprint(s: &str) -> Option<String> {
    let s = s.trim();
    let s = s
        .strip_prefix("sha256:")
        .or_else(|| s.strip_prefix("sha256="))
        .or_else(|| s.strip_prefix("SHA256:"))
        .unwrap_or(s);
    let digits: String = s
        .chars()
        .filter(|c| !matches!(c, ' ' | ':' | '-'))
        .collect::<String>()
        .to_lowercase();
    (digits.len() == 64 && digits.chars().all(|c| c.is_ascii_hexdigit())).then_some(digits)
}

/// A fingerprint as a person reads it: `AB12 CD34 ...`, sixteen groups.
pub fn show_fingerprint(fp: &str) -> String {
    let up = fp.to_uppercase();
    up.as_bytes()
        .chunks(4)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Seconds since 1970, UTC.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The time now as a person reads it: `2026-10-02 15:30:12 UTC`.
pub fn stamp() -> String {
    let (d, t) = crate::dicom_export::today();
    if d.len() == 8 && t.len() == 6 {
        format!(
            "{}-{}-{} {}:{}:{} UTC",
            &d[..4],
            &d[4..6],
            &d[6..],
            &t[..2],
            &t[2..4],
            &t[4..]
        )
    } else {
        format!("{d} {t}")
    }
}

/// `n` random bytes from the operating system, as unpadded base64url: a
/// token or an identifier nobody can guess.
pub fn random_token(n: usize) -> Result<String> {
    use base64::Engine as _;
    let mut buf = vec![0u8; n];
    if let Err(e) = getrandom::fill(&mut buf) {
        bail!("no random numbers from the operating system: {e}");
    }
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf))
}

/// What a token is kept as: its SHA-256 in hex. The server never stores a
/// token itself, so a copy of its files lets nobody in.
pub fn hash_token(token: &str) -> String {
    hex(&Sha256::digest(token.as_bytes()))
}

/// This machine's name, as a server calls itself and a client names the
/// device it pairs from.
pub fn this_device_name() -> String {
    let name = gethostname::gethostname()
        .to_string_lossy()
        .trim()
        .to_string();
    if name.is_empty() {
        "this computer".into()
    } else {
        name
    }
}

/// Where a server is and how to trust it, in one line a person can paste:
///
/// ```text
/// rds-pacs://192.168.1.20:11443/?code=K7QM-3TXA#sha256=ab12...
/// ```
///
/// The operator's window copies it; the client's *Add server* dialog reads
/// it. The code is optional (a pairing code travels with the line only
/// when the operator copied an invitation). Plain `host`, `host:port` and
/// `https://host:port` are read as well, without the fingerprint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectionLine {
    pub host: String,
    pub port: u16,
    pub fingerprint: Option<String>,
    pub code: Option<String>,
}

impl ConnectionLine {
    pub fn parse(text: &str) -> Result<ConnectionLine> {
        let t = text.trim();
        if t.is_empty() {
            bail!("no address");
        }
        let (rest, fragment) = match t.split_once('#') {
            Some((a, b)) => (a, Some(b)),
            None => (t, None),
        };
        let rest = rest
            .strip_prefix("rds-pacs://")
            .or_else(|| rest.strip_prefix("https://"))
            .unwrap_or(rest);
        if rest.starts_with("http://") {
            bail!("the PACS server speaks HTTPS only");
        }
        let (authority, query) = match rest.split_once('?') {
            Some((a, q)) => (a, Some(q)),
            None => (rest, None),
        };
        let authority = authority.trim_end_matches('/');
        let authority = authority.split('/').next().unwrap_or("");
        let (host, port) = split_host_port(authority)?;
        let fingerprint = match fragment {
            Some(f) if !f.trim().is_empty() => Some(
                normalize_fingerprint(f)
                    .ok_or_else(|| anyhow::anyhow!("the fingerprint is not 64 hex digits"))?,
            ),
            _ => None,
        };
        let code = query.and_then(|q| {
            q.split('&')
                .filter_map(|kv| kv.split_once('='))
                .find(|(k, _)| *k == "code")
                .map(|(_, v)| v.trim().to_string())
                .filter(|v| !v.is_empty())
        });
        Ok(ConnectionLine {
            host,
            port,
            fingerprint,
            code,
        })
    }

    /// The line, as the operator's window copies it.
    pub fn format(&self) -> String {
        let mut s = format!("rds-pacs://{}/", self.authority());
        if let Some(c) = &self.code {
            s.push_str(&format!("?code={c}"));
        }
        if let Some(fp) = &self.fingerprint {
            s.push_str(&format!("#sha256={fp}"));
        }
        s
    }

    /// `host:port`, with an IPv6 address in brackets.
    pub fn authority(&self) -> String {
        if self.host.contains(':') {
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }

    /// The address requests go to.
    pub fn base_url(&self) -> String {
        format!("https://{}", self.authority())
    }
}

fn split_host_port(a: &str) -> Result<(String, u16)> {
    if a.is_empty() {
        bail!("no address");
    }
    if let Some(rest) = a.strip_prefix('[') {
        let (host, after) = rest
            .split_once(']')
            .ok_or_else(|| anyhow::anyhow!("an IPv6 address needs its closing ]"))?;
        let port = match after.strip_prefix(':') {
            Some(p) => p
                .parse()
                .map_err(|_| anyhow::anyhow!("'{p}' is not a port"))?,
            None => DEFAULT_PORT,
        };
        return Ok((host.to_string(), port));
    }
    match a.rsplit_once(':') {
        // More than one colon and no brackets: a bare IPv6 address.
        Some((h, _)) if h.contains(':') => Ok((a.to_string(), DEFAULT_PORT)),
        Some((h, p)) => Ok((
            h.to_string(),
            p.parse()
                .map_err(|_| anyhow::anyhow!("'{p}' is not a port"))?,
        )),
        None => Ok((a.to_string(), DEFAULT_PORT)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprints_are_read_however_they_were_typed() {
        let fp = "ab".repeat(32);
        assert_eq!(normalize_fingerprint(&fp), Some(fp.clone()));
        assert_eq!(
            normalize_fingerprint(&format!("sha256:{}", fp.to_uppercase())),
            Some(fp.clone())
        );
        let shown = show_fingerprint(&fp);
        assert_eq!(shown.split(' ').count(), 16);
        assert_eq!(normalize_fingerprint(&shown), Some(fp.clone()));
        assert_eq!(normalize_fingerprint("abc"), None);
        assert_eq!(normalize_fingerprint(&"zz".repeat(32)), None);
        assert_eq!(fingerprint(b"x").len(), 64);
    }

    #[test]
    fn connection_lines_round_trip_and_plain_addresses_read() {
        let fp = "0f".repeat(32);
        let line = ConnectionLine {
            host: "192.168.1.20".into(),
            port: 11443,
            fingerprint: Some(fp.clone()),
            code: Some("K7QM-3TXA".into()),
        };
        let text = line.format();
        assert_eq!(
            text,
            format!("rds-pacs://192.168.1.20:11443/?code=K7QM-3TXA#sha256={fp}")
        );
        assert_eq!(ConnectionLine::parse(&text).unwrap(), line);

        let plain = ConnectionLine::parse("pacs.example.org").unwrap();
        assert_eq!(plain.port, DEFAULT_PORT);
        assert_eq!(plain.fingerprint, None);
        assert_eq!(
            ConnectionLine::parse("https://host:9000/").unwrap().port,
            9000
        );
        let v6 = ConnectionLine::parse("[fd7a::1]:12000").unwrap();
        assert_eq!((v6.host.as_str(), v6.port), ("fd7a::1", 12000));
        assert_eq!(v6.base_url(), "https://[fd7a::1]:12000");
        assert!(ConnectionLine::parse("http://host").is_err());
        assert!(ConnectionLine::parse("host:port").is_err());
        assert!(ConnectionLine::parse("host#sha256=12").is_err());
    }

    #[test]
    fn tokens_are_random_and_kept_only_as_hashes() {
        let a = random_token(32).unwrap();
        let b = random_token(32).unwrap();
        assert_ne!(a, b);
        assert_eq!(a.len(), 43, "32 bytes as unpadded base64");
        assert_eq!(hash_token(&a).len(), 64);
        assert_ne!(hash_token(&a), hash_token(&b));
        assert!(stamp().ends_with("UTC"));
    }
}
