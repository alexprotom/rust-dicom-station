//! Who may talk to the server: pairing codes, tokens, roles.
//!
//! A client gets in once, by a **pairing code** the operator made - eight
//! characters from an alphabet without the look-alikes `0/O/1/I`, good for
//! a few minutes and for one use, carrying the role the operator chose -
//! and from then on by the **token** the server issued for it: 32 random
//! bytes, sent as `Authorization: Bearer`. The server keeps only the
//! token's SHA-256 (in `clients.json`), compares hashes in constant time,
//! and forgets a client when the operator revokes it.
//!
//! Guessing is made slow and short-lived: one pairing attempt per second
//! per address, and a code that has seen five wrong attempts is void.
//! Codes live in memory only: they are made through the running server
//! (the operator's window, `rds-pacs pair`), so a restart voids them, which
//! costs nothing but a new code.
//!
//! The **local operator** is the viewer on the server's own machine: the
//! server keeps a token for it in its state folder (readable by its owner
//! only; made on the first start, kept across restarts so the viewer's
//! copy never goes stale), so *Settings ▶ PACS server* needs no pairing.

use std::collections::HashMap;
use std::net::IpAddr;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

use super::local::Paths;
use super::protocol::{ClientInfo, PairingCode, Role};
use super::servers::write_private;

/// The characters a pairing code is made of: no `0 O 1 I`, so a code read
/// aloud or off a screen is typed right.
const CODE_ALPHABET: &[u8] = b"ABCDEFGHJKLMNPQRSTUVWXYZ23456789";

/// Wrong attempts after which a code is void.
const CODE_MAX_FAILURES: u32 = 5;

/// How long one address waits between pairing attempts.
const ATTEMPT_GAP: Duration = Duration::from_secs(1);

/// One paired client, as `clients.json` keeps it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClientRecord {
    pub name: String,
    pub role: Role,
    /// SHA-256 of the token, hex. The token itself is never stored.
    pub token_hash: String,
    pub paired: String,
    pub last_seen: String,
    pub address: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
struct ClientsFile {
    clients: Vec<ClientRecord>,
}

/// A code waiting to be used.
struct Code {
    code: String,
    role: Role,
    expires: Instant,
    failures: u32,
}

/// Who is calling, once their token checked out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Caller {
    pub name: String,
    pub role: Role,
    /// The local operator (the viewer on this machine).
    pub local: bool,
}

/// Why a pairing attempt failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PairError {
    /// Too soon after the last attempt from the same address.
    TooSoon,
    /// No such code, or it expired, or it was used.
    BadCode,
}

/// The clients and the codes. Behind a mutex in the server.
pub struct Auth {
    path: std::path::PathBuf,
    clients: Vec<ClientRecord>,
    local_hash: String,
    codes: Vec<Code>,
    attempts: HashMap<IpAddr, Instant>,
    /// `last_seen` changed since the file was written.
    dirty: bool,
}

impl Auth {
    /// Read `clients.json` and the local operator token (made when there
    /// is none, or none that looks like one).
    pub fn load(paths: &Paths) -> Result<Auth> {
        let path = paths.clients();
        let clients = match std::fs::read_to_string(&path) {
            Ok(text) => {
                serde_json::from_str::<ClientsFile>(&text)
                    .with_context(|| format!("read {}", path.display()))?
                    .clients
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        let kept = std::fs::read_to_string(paths.local_token())
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| {
                t.len() >= 40
                    && t.chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            });
        let local = match kept {
            Some(t) => t,
            None => {
                let t = super::random_token(32)?;
                write_private(&paths.local_token(), &t)?;
                t
            }
        };
        Ok(Auth {
            path,
            clients,
            local_hash: super::hash_token(&local),
            codes: Vec::new(),
            attempts: HashMap::new(),
            dirty: false,
        })
    }

    fn save(&mut self) -> Result<()> {
        let text = serde_json::to_string_pretty(&ClientsFile {
            clients: self.clients.clone(),
        })
        .expect("plain data serialises");
        write_private(&self.path, &text)?;
        self.dirty = false;
        Ok(())
    }

    /// Write `last_seen` out if it changed (called now and then, and when
    /// the server stops).
    pub fn flush(&mut self) {
        if self.dirty {
            let _ = self.save();
        }
    }

    /// Who a bearer token belongs to. Every record is compared, in constant
    /// time, whatever matched first.
    pub fn check(&mut self, token: &str, address: &str) -> Option<Caller> {
        let h = super::hash_token(token);
        let mut found: Option<usize> = None;
        for (i, c) in self.clients.iter().enumerate() {
            if bool::from(c.token_hash.as_bytes().ct_eq(h.as_bytes())) {
                found = Some(i);
            }
        }
        let local = bool::from(self.local_hash.as_bytes().ct_eq(h.as_bytes()));
        if local {
            return Some(Caller {
                name: "local operator".into(),
                role: Role::Admin,
                local: true,
            });
        }
        let i = found?;
        let now = super::stamp();
        let c = &mut self.clients[i];
        // Minutes are enough for "last seen", and fewer writes.
        if c.last_seen.get(..16) != now.get(..16) || c.address != address {
            c.last_seen = now;
            c.address = address.to_string();
            self.dirty = true;
        }
        Some(Caller {
            name: c.name.clone(),
            role: c.role,
            local: false,
        })
    }

    /// Make a pairing code.
    pub fn new_code(&mut self, role: Role, minutes: u64) -> Result<PairingCode> {
        self.codes.retain(|c| c.expires > Instant::now());
        let mut bytes = [0u8; 8];
        getrandom::fill(&mut bytes)
            .map_err(|e| anyhow::anyhow!("no random numbers from the operating system: {e}"))?;
        let chars: String = bytes
            .iter()
            .map(|b| CODE_ALPHABET[*b as usize % CODE_ALPHABET.len()] as char)
            .collect();
        let code = format!("{}-{}", &chars[..4], &chars[4..]);
        let minutes = minutes.clamp(1, 24 * 60);
        self.codes.push(Code {
            code: code.clone(),
            role,
            expires: Instant::now() + Duration::from_secs(minutes * 60),
            failures: 0,
        });
        Ok(PairingCode {
            code,
            role,
            expires: expiry_stamp(minutes * 60),
            expires_in_s: minutes * 60,
        })
    }

    /// Trade a code for a token. Returns the token, the role and the name
    /// the client is filed under.
    pub fn pair(
        &mut self,
        code: &str,
        client_name: &str,
        from: IpAddr,
    ) -> std::result::Result<(String, Role, String), PairError> {
        let now = Instant::now();
        if let Some(last) = self.attempts.get(&from) {
            if now.duration_since(*last) < ATTEMPT_GAP {
                return Err(PairError::TooSoon);
            }
        }
        self.attempts.insert(from, now);
        // The table of attempts must not grow without end.
        if self.attempts.len() > 4096 {
            self.attempts
                .retain(|_, t| now.duration_since(*t) < Duration::from_secs(60));
        }
        self.codes
            .retain(|c| c.expires > now && c.failures < CODE_MAX_FAILURES);
        let typed = normalize_code(code);
        let hit = self
            .codes
            .iter()
            .position(|c| bool::from(normalize_code(&c.code).as_bytes().ct_eq(typed.as_bytes())));
        let Some(i) = hit else {
            for c in &mut self.codes {
                c.failures += 1;
            }
            return Err(PairError::BadCode);
        };
        let role = self.codes.remove(i).role;
        let token = super::random_token(32).map_err(|_| PairError::BadCode)?;
        let name = self.unique_name(client_name);
        self.clients.push(ClientRecord {
            name: name.clone(),
            role,
            token_hash: super::hash_token(&token),
            paired: super::stamp(),
            last_seen: super::stamp(),
            address: from.to_string(),
        });
        if self.save().is_err() {
            self.clients.pop();
            return Err(PairError::BadCode);
        }
        Ok((token, role, name))
    }

    fn unique_name(&self, wanted: &str) -> String {
        let base: String = wanted
            .trim()
            .chars()
            .filter(|c| !c.is_control())
            .take(64)
            .collect();
        let base = if base.is_empty() {
            "client".to_string()
        } else {
            base
        };
        let taken = |n: &str| self.clients.iter().any(|c| c.name.eq_ignore_ascii_case(n));
        if !taken(&base) {
            return base;
        }
        (2..)
            .map(|i| format!("{base} ({i})"))
            .find(|n| !taken(n))
            .expect("some number is free")
    }

    /// Forget a client: its token stops working at once.
    pub fn revoke(&mut self, name: &str) -> Result<bool> {
        let before = self.clients.len();
        self.clients.retain(|c| c.name != name);
        let gone = self.clients.len() != before;
        if gone {
            self.save()?;
        }
        Ok(gone)
    }

    pub fn clients(&self) -> Vec<ClientInfo> {
        self.clients
            .iter()
            .map(|c| ClientInfo {
                name: c.name.clone(),
                role: c.role,
                paired: c.paired.clone(),
                last_seen: c.last_seen.clone(),
                address: c.address.clone(),
            })
            .collect()
    }
}

/// Upper case, the dash and spaces dropped: `k7qm 3txa` is `K7QM-3TXA`.
fn normalize_code(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect()
}

/// When something `secs` from now happens: `16:40 UTC`.
fn expiry_stamp(secs: u64) -> String {
    let now = super::unix_now();
    let at = now + secs;
    let s = at % 86400;
    format!(
        "{:02}:{:02} UTC{}",
        s / 3600,
        (s % 3600) / 60,
        if at / 86400 > now / 86400 {
            " (tomorrow)"
        } else {
            ""
        }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn fresh(tag: &str) -> (Auth, Paths) {
        let root = std::env::temp_dir().join(format!("rds_pacs_auth_{tag}"));
        let _ = std::fs::remove_dir_all(&root);
        let paths = Paths::at(&root);
        (Auth::load(&paths).unwrap(), paths)
    }

    fn ip(n: u8) -> IpAddr {
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, n))
    }

    #[test]
    fn a_code_works_once_and_the_token_then_lets_in() {
        let (mut a, paths) = fresh("once");
        let code = a.new_code(Role::Edit, 10).unwrap();
        assert_eq!(code.code.len(), 9);
        assert!(!code.code.contains(['0', 'O', '1', 'I']));
        let typed = code.code.to_lowercase().replace('-', " ");
        let (token, role, name) = a.pair(&typed, "laptop", ip(1)).unwrap();
        assert_eq!((role, name.as_str()), (Role::Edit, "laptop"));
        std::thread::sleep(ATTEMPT_GAP);
        assert_eq!(a.pair(&code.code, "laptop", ip(1)), Err(PairError::BadCode));
        let who = a.check(&token, "10.0.0.1").unwrap();
        assert_eq!(
            (who.name.as_str(), who.role, who.local),
            ("laptop", Role::Edit, false)
        );
        assert!(a.check("not a token", "x").is_none());

        // The file keeps the hash, never the token; a reload still knows it.
        let text = std::fs::read_to_string(paths.clients()).unwrap();
        assert!(!text.contains(&token));
        let mut again = Auth::load(&paths).unwrap();
        assert!(again.check(&token, "x").is_some());

        // The local operator's token is kept across starts, so the
        // viewer's copy of it never goes stale.
        let local = std::fs::read_to_string(paths.local_token()).unwrap();
        assert!(again.check(&local, "127.0.0.1").unwrap().local);
        assert!(
            a.check(&local, "127.0.0.1").unwrap().local,
            "the earlier start knows the same token"
        );
        // A damaged file is replaced.
        std::fs::write(paths.local_token(), "short").unwrap();
        let mut third = Auth::load(&paths).unwrap();
        let renewed = std::fs::read_to_string(paths.local_token()).unwrap();
        assert_ne!(renewed.trim(), "short");
        assert!(third.check(renewed.trim(), "127.0.0.1").unwrap().local);
        assert!(third.check(&local, "127.0.0.1").is_none());

        assert!(again.revoke("laptop").unwrap());
        assert!(again.check(&token, "x").is_none());
        assert!(!again.revoke("laptop").unwrap());
    }

    #[test]
    fn guessing_is_slow_and_spoils_the_code() {
        let (mut a, _) = fresh("guess");
        let code = a.new_code(Role::View, 10).unwrap();
        assert_eq!(a.pair("AAAA-AAAA", "x", ip(2)), Err(PairError::BadCode));
        assert_eq!(
            a.pair(&code.code, "x", ip(2)),
            Err(PairError::TooSoon),
            "one attempt per second per address"
        );
        for n in 3..7 {
            assert_eq!(a.pair("AAAA-AAAA", "x", ip(n)), Err(PairError::BadCode));
        }
        assert_eq!(
            a.pair(&code.code, "x", ip(9)),
            Err(PairError::BadCode),
            "five wrong attempts void the code"
        );
    }

    #[test]
    fn names_are_kept_apart() {
        let (mut a, _) = fresh("names");
        for (n, want) in [(1, "ipad"), (2, "ipad (2)"), (3, "ipad (3)")] {
            let c = a.new_code(Role::View, 10).unwrap();
            let (_, _, name) = a.pair(&c.code, "ipad", ip(n)).unwrap();
            assert_eq!(name, want);
        }
        assert_eq!(a.clients().len(), 3);
        let c = a.new_code(Role::View, 10).unwrap();
        let (_, _, name) = a.pair(&c.code, "  ", ip(7)).unwrap();
        assert_eq!(name, "client");
    }
}
