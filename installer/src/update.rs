//! Updating to the newest release published on GitHub.
//!
//! The setup program kept in the program folder does this when started with
//! `--update` (the *Update Rust DICOM Station* shortcut), and the wizard
//! offers it when the setup being run is older than the newest release.
//!
//! Nothing new is trusted on the way: the latest release is found by
//! following GitHub's own `releases/latest` redirect, the installer is the
//! asset the release workflow attaches under a fixed name, and it is only
//! run after its SHA-256 matched the `SHA256SUMS` file of the same release.
//! The downloaded setup then does the actual work - updating an existing
//! installation in place, like any setup run over one.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};

use crate::plan::*;

/// A published release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Release {
    /// `0.8.9`
    pub version: String,
}

impl Release {
    pub fn asset_name(&self) -> String {
        release_asset_name(&self.version)
    }

    pub fn asset_url(&self) -> String {
        release_download_url(&self.version, &self.asset_name())
    }

    pub fn sums_url(&self) -> String {
        release_download_url(&self.version, "SHA256SUMS")
    }

    /// True when this release is newer than `version`.
    pub fn is_newer_than(&self, version: &str) -> bool {
        compare_versions(&self.version, version) == std::cmp::Ordering::Greater
    }
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(60))
        // HTTPS_PROXY / HTTP_PROXY, as in most clinical networks.
        .try_proxy_from_env(true)
        .user_agent(concat!("rds-setup/", env!("CARGO_PKG_VERSION")))
        .build()
}

/// The newest published release (drafts and pre-releases are not "latest"
/// on GitHub, so they are never offered).
pub fn latest() -> Result<Release> {
    let url = format!("https://github.com/{RELEASE_REPO}/releases/latest");
    // ureq's error names the URL already.
    let resp = agent().get(&url).call().map_err(|e| {
        failure(
            EXIT_NO_NETWORK,
            format!("could not look up the newest release: {e}"),
        )
    })?;
    let landed = resp.get_url().to_string();
    release_from_url(&landed).ok_or_else(|| {
        anyhow!("no release has been published yet ({landed} is not a release page)")
    })
}

/// `https://github.com/o/r/releases/tag/v0.8.9` names release 0.8.9.
pub fn release_from_url(url: &str) -> Option<Release> {
    let tag = url.split("/releases/tag/").nth(1)?;
    let tag = tag.split(['/', '?', '#']).next()?.trim();
    let version = tag.strip_prefix(['v', 'V']).unwrap_or(tag);
    let plausible = version.starts_with(|c: char| c.is_ascii_digit())
        && version
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+'));
    plausible.then(|| Release {
        version: version.to_string(),
    })
}

/// The hash `SHA256SUMS` lists for `name` (the `sha256sum` format: hash,
/// space, optional `*` for binary mode, file name).
pub fn sums_lookup(sums: &str, name: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let (hash, file) = line.trim().split_once(char::is_whitespace)?;
        let file = file.trim().trim_start_matches('*');
        let hex = hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit());
        (hex && file == name).then(|| hash.to_ascii_lowercase())
    })
}

/// Where downloads are kept: `%TEMP%\RustDicomStation-update`.
fn download_dir() -> PathBuf {
    std::env::temp_dir().join(format!("{PRODUCT_ID}-update"))
}

/// Download the release's Windows installer and verify it. Returns its path.
pub fn download(
    release: &Release,
    progress: &dyn Fn(f32, &str),
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    let dir = download_dir();
    // Earlier downloads: one may still be running (it is the setup that
    // started this one), so failures are fine.
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let _ = std::fs::remove_file(e.path());
        }
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;

    let name = release.asset_name();
    progress(0.0, "Reading the release checksums");
    let sums = agent()
        .get(&release.sums_url())
        .call()
        .map_err(|e| failure(EXIT_NO_NETWORK, format!("download failed: {e}")))?
        .into_string()
        .context("read SHA256SUMS")?;
    let expected = sums_lookup(&sums, &name)
        .ok_or_else(|| anyhow!("the release's SHA256SUMS has no entry for {name}"))?;

    let url = release.asset_url();
    let resp = agent()
        .get(&url)
        .call()
        .map_err(|e| failure(EXIT_NO_NETWORK, format!("download failed: {e}")))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(40_000_000)
        .max(1);
    let dest = dir.join(&name);
    let mut reader = resp.into_reader();
    let mut hasher = Sha256::new();
    {
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&dest).with_context(|| format!("write {}", dest.display()))?,
        );
        let mut buf = vec![0u8; 256 * 1024];
        let mut done: u64 = 0;
        let label = format!("Downloading {name}");
        loop {
            if cancel.load(Ordering::Relaxed) {
                drop(out);
                let _ = std::fs::remove_file(&dest);
                return Err(failure(EXIT_CANCELLED, "cancelled"));
            }
            let n = reader
                .read(&mut buf)
                .map_err(|e| failure(EXIT_NO_NETWORK, format!("download interrupted: {e}")))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n])?;
            done += n as u64;
            progress((done as f32 / total as f32).min(0.99), &label);
        }
        out.flush()?;
    }
    let actual = hex(&hasher.finalize());
    if actual != expected {
        let _ = std::fs::remove_file(&dest);
        return Err(anyhow!(
            "the downloaded {name} does not match the release's SHA256SUMS \
             (expected {expected}, got {actual}); nothing was run"
        ));
    }
    progress(1.0, "Download verified");
    Ok(dest)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// How to start the downloaded setup.
#[derive(Clone, Copy, Debug)]
pub enum HandOff {
    /// Its own window, going straight to work (`--autostart`).
    Window,
    /// The wizard from its first page - for a setup the user has not
    /// answered any questions for yet.
    Wizard,
    /// `--silent`, waited for; the exit code is passed on.
    Silent,
    /// The text interface in this console, waited for.
    Console,
}

/// Start the downloaded setup. `elevate` when the installation it will
/// update is machine-wide. Returns the exit code to finish with.
pub fn hand_off(setup: &Path, how: HandOff, elevate: bool) -> Result<u8> {
    let args: &[&str] = match how {
        HandOff::Window => &["--autostart"],
        HandOff::Wizard => &[],
        HandOff::Silent => &["--silent"],
        HandOff::Console => &["--console"],
    };
    let wait = matches!(how, HandOff::Silent | HandOff::Console);
    if elevate && !crate::win::is_elevated() {
        if wait {
            return Err(anyhow!(
                "the installation is for all users: run the update from an elevated prompt"
            ));
        }
        crate::win::run_elevated(setup, &args.join(" "))?;
        return Ok(EXIT_OK);
    }
    if wait {
        let status = std::process::Command::new(setup)
            .args(args)
            .status()
            .with_context(|| format!("run {}", setup.display()))?;
        return Ok(status
            .code()
            .and_then(|c| u8::try_from(c).ok())
            .unwrap_or(EXIT_FAILED));
    }
    crate::win::shell_execute(setup, &args.join(" "), false)?;
    Ok(EXIT_OK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_latest_redirect_names_the_release() {
        let r = release_from_url(
            "https://github.com/alexprotom/rust-dicom-station/releases/tag/v0.8.9",
        )
        .unwrap();
        assert_eq!(r.version, "0.8.9");
        assert_eq!(
            r.asset_url(),
            "https://github.com/alexprotom/rust-dicom-station/releases/download/v0.8.9/\
             rust-dicom-station-0.8.9-windows-x86_64.exe"
        );
        assert_eq!(
            release_from_url("https://github.com/o/r/releases/tag/1.2.0?x=1")
                .unwrap()
                .version,
            "1.2.0"
        );
    }

    #[test]
    fn a_repository_without_releases_is_not_mistaken_for_one() {
        // GitHub sends /releases/latest to the plain list when there is none.
        assert_eq!(release_from_url("https://github.com/o/r/releases"), None);
        assert_eq!(
            release_from_url("https://github.com/o/r/releases/tag/nightly"),
            None,
            "a tag that is not a version"
        );
    }

    #[test]
    fn the_checksum_is_found_by_exact_file_name() {
        let a = "a".repeat(64);
        let b = "B".repeat(64);
        let sums = format!(
            "{a}  rust-dicom-station-0.8.9-linux-x86_64.AppImage\n\
             {b} *rust-dicom-station-0.8.9-windows-x86_64.exe\n"
        );
        assert_eq!(
            sums_lookup(&sums, "rust-dicom-station-0.8.9-windows-x86_64.exe"),
            Some("b".repeat(64)),
            "binary-mode star accepted, hash compared in lower case"
        );
        assert_eq!(sums_lookup(&sums, "rust-dicom-station-0.8.9.exe"), None);
        assert_eq!(sums_lookup("nothex  x.exe\n", "x.exe"), None);
    }

    #[test]
    fn newer_means_newer_by_version_number() {
        let r = Release {
            version: "0.8.10".into(),
        };
        assert!(r.is_newer_than("0.8.9"));
        assert!(!r.is_newer_than("0.8.10"));
        assert!(!r.is_newer_than("0.9.0"));
    }

    #[test]
    fn the_hash_is_written_as_lower_case_hex() {
        let digest = Sha256::digest(b"abc");
        assert_eq!(
            hex(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
