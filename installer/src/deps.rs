//! External runtime dependencies.
//!
//! The viewer is a single static-ish Rust binary: it needs the Microsoft
//! Visual C++ runtime (`vcruntime140.dll`, which Rust's MSVC target links
//! against) and a Direct3D 12 or Vulkan capable driver, which Windows always
//! provides in some form — even the software WARP adapter works.
//!
//! Only the first of those is installable, and only when it is missing; on
//! Windows 10/11 it usually is not.

use std::io::{Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::plan::VCREDIST_URL;
use crate::win::registry::{Key, Hive};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dependency {
    Present,
    Missing,
}

/// Is the Visual C++ 2015-2022 x64 runtime available?
pub fn vcredist_state() -> Dependency {
    // The DLLs are what actually matters at load time.
    let sysdir = std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("System32");
    let dlls_present = sysdir.join("vcruntime140.dll").is_file()
        && sysdir.join("vcruntime140_1.dll").is_file();
    if dlls_present {
        return Dependency::Present;
    }
    // Fall back to the redistributable's own registry marker.
    let installed = Key::open(
        Hive::LocalMachine.hkey(),
        r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        false,
    )
    .ok()
    .and_then(|k| k.get_str("Version"))
    .is_some();
    if installed {
        Dependency::Present
    } else {
        Dependency::Missing
    }
}

/// Download the official redistributable from Microsoft and run it passively.
///
/// This is the one file the installer fetches from outside the payload; the
/// redistributable is Microsoft's own signed installer and it runs with the
/// privileges the installer already has (it will raise its own UAC prompt when
/// the installer is not elevated).
pub fn install_vcredist(progress: &dyn Fn(f32, &str)) -> Result<()> {
    let dest = std::env::temp_dir().join("vc_redist.x64.exe");
    progress(0.0, "Downloading the Visual C++ runtime…");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(60))
        .build();
    let resp = agent
        .get(VCREDIST_URL)
        .call()
        .with_context(|| format!("download {VCREDIST_URL}"))?;
    let total = resp
        .header("Content-Length")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(25_000_000);
    let mut reader = resp.into_reader();
    {
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&dest).with_context(|| format!("write {}", dest.display()))?,
        );
        let mut buf = vec![0u8; 256 * 1024];
        let mut done: u64 = 0;
        loop {
            let n = reader.read(&mut buf).context("download read")?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            done += n as u64;
            progress(
                (done as f32 / total.max(1) as f32).min(1.0),
                "Downloading the Visual C++ runtime…",
            );
        }
        out.flush()?;
    }
    progress(1.0, "Installing the Visual C++ runtime…");
    let status = std::process::Command::new(&dest)
        .args(["/install", "/passive", "/norestart"])
        .status()
        .with_context(|| format!("run {}", dest.display()))?;
    let _ = std::fs::remove_file(&dest);
    match status.code() {
        // 0 = installed, 1638 = a newer version is already there,
        // 3010 = success, reboot required.
        Some(0) | Some(1638) | Some(3010) | None => Ok(()),
        Some(code) => bail!("the Visual C++ runtime installer failed with exit code {code}"),
    }
}
