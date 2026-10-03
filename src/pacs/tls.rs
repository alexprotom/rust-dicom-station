//! The server's certificate and its TLS configuration.
//!
//! On first start the server makes a self-signed certificate (`rcgen`, an
//! ECDSA P-256 key) and keeps it in its state folder as `server.crt` and
//! `server.key`, the key readable by its owner only. Its SHA-256 is the
//! fingerprint a client pins when it pairs, so the certificate is kept
//! for good: names in it do not matter to a pinned client, and a
//! certificate that changed would make every client pair again, which is
//! what `rds-pacs cert --regenerate` does on purpose (and says so).
//!
//! An operator who has a certificate of their own - a public name, a
//! reverse proxy, Let's Encrypt - gives its PEM files as `tls_cert` and
//! `tls_key` in `pacs.toml`; clients then pair with *trust the system's
//! certificates* instead of a fingerprint.
//!
//! TLS is rustls with the `ring` provider, set explicitly, the same one the
//! client side uses; ALPN offers HTTP/1.1 only.

use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};

use super::config::Config;
use super::local::Paths;
use super::servers::write_private;

/// The certificate a server presents.
pub struct ServerCert {
    pub chain: Vec<CertificateDer<'static>>,
    pub key: PrivateKeyDer<'static>,
    /// SHA-256 of the leaf certificate, hex.
    pub fingerprint: String,
    /// Made by the server itself (pinned by clients), not the operator's.
    pub self_signed: bool,
}

/// The names the self-signed certificate carries: the loopback, this
/// machine's name and addresses, and what `advertise` adds.
pub fn names(config: &Config) -> Vec<String> {
    let mut out: Vec<String> = vec!["localhost".into(), "127.0.0.1".into(), "::1".into()];
    let host = super::this_device_name();
    if host != "this computer" {
        out.push(host);
    }
    for ip in super::local::addresses() {
        out.push(ip.to_string());
    }
    for a in &config.advertise {
        let a = a.trim();
        if !a.is_empty() {
            out.push(a.to_string());
        }
    }
    let mut seen = std::collections::BTreeSet::new();
    out.retain(|n| seen.insert(n.to_lowercase()));
    // rcgen reads an address as an IP name; anything else must at least
    // look like a DNS name.
    out.retain(|n| {
        n.parse::<IpAddr>().is_ok()
            || n.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
    });
    out
}

/// The certificate to serve: the operator's own when `pacs.toml` names
/// one, else the self-signed one, made now when there is none yet.
pub fn load_or_make(paths: &Paths, config: &Config) -> Result<ServerCert> {
    if let Some((cert, key)) = config.own_certificate() {
        return read_pem(&cert, &key, false);
    }
    if !paths.cert().is_file() || !paths.key().is_file() {
        make(paths, config)?;
    }
    read_pem(&paths.cert(), &paths.key(), true)
}

/// Make a new self-signed certificate, replacing any old one. Every client
/// pinned to the old one has to pair again. Returns the new fingerprint.
pub fn make(paths: &Paths, config: &Config) -> Result<String> {
    let made = rcgen::generate_simple_self_signed(names(config))
        .context("make the server's certificate")?;
    std::fs::create_dir_all(&paths.state)
        .with_context(|| format!("create {}", paths.state.display()))?;
    write_private(&paths.key(), &made.signing_key.serialize_pem())?;
    std::fs::write(paths.cert(), made.cert.pem())
        .with_context(|| format!("write {}", paths.cert().display()))?;
    Ok(super::fingerprint(made.cert.der().as_ref()))
}

fn read_pem(cert: &Path, key: &Path, self_signed: bool) -> Result<ServerCert> {
    let chain: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert)
        .with_context(|| format!("read {}", cert.display()))?
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("read the certificates in {}", cert.display()))?;
    let Some(leaf) = chain.first() else {
        bail!("{} holds no certificate", cert.display());
    };
    let fingerprint = super::fingerprint(leaf.as_ref());
    let key = PrivateKeyDer::from_pem_file(key)
        .with_context(|| format!("read the private key in {}", key.display()))?;
    Ok(ServerCert {
        chain,
        key,
        fingerprint,
        self_signed,
    })
}

/// The rustls configuration the listener uses.
pub fn server_config(cert: &ServerCert) -> Result<Arc<rustls::ServerConfig>> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut cfg = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .context("TLS configuration")?
        .with_no_client_auth()
        .with_single_cert(cert.chain.clone(), cert.key.clone_key())
        .context("the server's certificate and key do not belong together")?;
    cfg.alpn_protocols = vec![b"http/1.1".to_vec()];
    Ok(Arc::new(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_certificate_is_made_once_and_kept() {
        let root = std::env::temp_dir().join("rds_pacs_tls");
        let _ = std::fs::remove_dir_all(&root);
        let paths = Paths::at(&root);
        let cfg = Config {
            advertise: vec!["pacs.example.org".into(), "bad name!".into()],
            ..Config::default()
        };
        let n = names(&cfg);
        assert!(n.contains(&"pacs.example.org".to_string()));
        assert!(!n.iter().any(|x| x.contains('!')));
        let a = load_or_make(&paths, &cfg).unwrap();
        assert!(a.self_signed);
        let b = load_or_make(&paths, &cfg).unwrap();
        assert_eq!(a.fingerprint, b.fingerprint, "stable across restarts");
        server_config(&b).unwrap();
        let fresh = make(&paths, &cfg).unwrap();
        assert_ne!(
            fresh, a.fingerprint,
            "a regenerated certificate is a new one"
        );
        assert_eq!(load_or_make(&paths, &cfg).unwrap().fingerprint, fresh);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(paths.key()).unwrap().permissions().mode();
            assert_eq!(mode & 0o077, 0, "the key is the owner's only");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
