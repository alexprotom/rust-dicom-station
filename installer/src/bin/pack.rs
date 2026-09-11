//! `rds-pack` - turns the freshly built `rds-setup.exe` into a shippable
//! single-file installer by appending a zip of everything that should be
//! installed, plus the footer `payload.rs` looks for.
//!
//!     cargo build --release                 # the viewer, from the repo root
//!     cd installer
//!     cargo build --release                 # rds-setup.exe + rds-pack.exe
//!     cargo run --release --bin rds-pack    # dist/rust-dicom-station-setup.exe
//!
//! Nothing here runs during a normal `cargo build` of the viewer: this crate
//! is a separate workspace.
//!
//! With `--winget <DIR>` it also writes the three winget manifests for the
//! installer it just built (version, installer, default locale), hashed and
//! pointing at the GitHub release asset - what a first submission to
//! microsoft/winget-pkgs, or `winget install --manifest`, needs.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;

#[path = "../product.rs"]
#[allow(dead_code)]
mod product;
use product::*;

const USAGE: &str = "\
rds-pack - build the shippable installer

USAGE:
    rds-pack [OPTIONS]

OPTIONS:
    --repo <DIR>       repository root (default: the parent of this crate)
    --app <FILE>       viewer executable (default: <repo>/target/release/rust-dicom-station.exe)
    --mcp <FILE>       MCP server executable (default: <repo>/target/release/rds-mcp.exe,
                       shipped when it exists; build it with --features mcp)
    --no-mcp           leave the MCP server out even when it was built
    --setup <FILE>     setup binary to wrap (default: target/release/rds-setup.exe)
    --out <FILE>       installer to write (default: dist/rust-dicom-station-setup.exe)
    --example-data     also ship example_data/ (~137 MB before compression)
    --no-docs          leave the docs/ folder out
    --winget <DIR>     also write the winget manifests for this installer into DIR
    --url <URL>        download URL recorded in them (default: the installer
                       asset of this version's GitHub release)
    --release-date <YYYY-MM-DD>
                       release date recorded in them (default: none)
    -h, --help         this text
";

struct Opts {
    repo: PathBuf,
    app: Option<PathBuf>,
    mcp: Option<PathBuf>,
    no_mcp: bool,
    setup: Option<PathBuf>,
    out: Option<PathBuf>,
    example_data: bool,
    docs: bool,
    winget: Option<PathBuf>,
    url: Option<String>,
    release_date: Option<String>,
}

fn main() -> Result<()> {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut o = Opts {
        repo: crate_dir.parent().unwrap_or(&crate_dir).to_path_buf(),
        app: None,
        mcp: None,
        no_mcp: false,
        setup: None,
        out: None,
        example_data: false,
        docs: true,
        winget: None,
        url: None,
        release_date: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |flag: &str| -> Result<String> {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{flag} needs a value"))
        };
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--repo" => o.repo = PathBuf::from(val("--repo")?),
            "--app" => o.app = Some(PathBuf::from(val("--app")?)),
            "--mcp" => o.mcp = Some(PathBuf::from(val("--mcp")?)),
            "--no-mcp" => o.no_mcp = true,
            "--setup" => o.setup = Some(PathBuf::from(val("--setup")?)),
            "--out" => o.out = Some(PathBuf::from(val("--out")?)),
            "--example-data" => o.example_data = true,
            "--no-docs" => o.docs = false,
            "--winget" => o.winget = Some(PathBuf::from(val("--winget")?)),
            "--url" => o.url = Some(val("--url")?),
            "--release-date" => o.release_date = Some(val("--release-date")?),
            other => bail!("unknown option '{other}'\n\n{USAGE}"),
        }
    }

    let app = o
        .app
        .unwrap_or_else(|| o.repo.join("target/release/rust-dicom-station.exe"));
    let setup = o
        .setup
        .unwrap_or_else(|| crate_dir.join("target/release/rds-setup.exe"));
    let out = o
        .out
        .unwrap_or_else(|| crate_dir.join("dist/rust-dicom-station-setup.exe"));

    if !app.is_file() {
        bail!(
            "{} not found - build the viewer first:\n    cargo build --release   (in {})",
            app.display(),
            o.repo.display()
        );
    }
    if !setup.is_file() {
        bail!(
            "{} not found - build the installer first:\n    cargo build --release   (in {})",
            setup.display(),
            crate_dir.display()
        );
    }

    // ---- collect the payload ---------------------------------------------
    let cargo_toml = o.repo.join("Cargo.toml");
    let version = read_version(&cargo_toml).unwrap_or_else(|| "0.0.0".to_string());
    let mut files: Vec<(String, PathBuf)> = vec![(APP_EXE.into(), app.clone())];
    // The MCP server is a second executable of the same crate (feature
    // `mcp`); it rides along when it was built, and its absence is not an
    // error: the viewer works without it.
    let mcp = o
        .mcp
        .clone()
        .unwrap_or_else(|| o.repo.join("target/release/rds-mcp.exe"));
    if !o.no_mcp && mcp.is_file() {
        files.push((MCP_EXE.into(), mcp));
    } else if o.mcp.is_some() {
        bail!("{} not found", mcp.display());
    }
    for name in ["LICENSE.txt", "README.md"] {
        let p = o.repo.join(name);
        if p.is_file() {
            files.push((name.to_string(), p));
        }
    }
    if o.docs {
        collect_dir(&o.repo, "docs", &mut files)?;
    }
    if o.example_data {
        collect_dir(&o.repo, "example_data", &mut files)?;
    }

    let info = format!(
        "# generated by rds-pack\nversion = {version}\nexample_data = {}\n",
        o.example_data
    );

    // ---- write the zip ----------------------------------------------------
    println!("Packing {} files", files.len() + 1);
    let mut zip_bytes = Vec::new();
    {
        let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut zip_bytes));
        let opts = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .large_file(true);
        zip.start_file("payload-info.txt", opts)?;
        zip.write_all(info.as_bytes())?;
        for (name, path) in &files {
            zip.start_file(name.as_str(), opts)?;
            let mut f =
                std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
            std::io::copy(&mut f, &mut zip)?;
            println!("  {name}");
        }
        zip.finish()?;
    }

    // ---- base setup binary, minus any payload from an earlier run ---------
    let base_len = match read_existing_footer(&setup)? {
        Some(offset) => offset,
        None => setup.metadata()?.len(),
    };

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(&out).with_context(|| format!("write {}", out.display()))?,
    );
    let mut src = std::fs::File::open(&setup)?;
    let mut left = base_len;
    let mut buf = vec![0u8; 1 << 20];
    while left > 0 {
        let want = buf.len().min(left as usize);
        let n = src.read(&mut buf[..want])?;
        if n == 0 {
            break;
        }
        w.write_all(&buf[..n])?;
        left -= n as u64;
    }
    w.write_all(&zip_bytes)?;
    w.write_all(b"RDSPAY01")?;
    w.write_all(&base_len.to_le_bytes())?;
    w.write_all(&(zip_bytes.len() as u64).to_le_bytes())?;
    w.flush()?;
    drop(w);

    let total = out.metadata()?.len();
    println!(
        "\n{}\n  version {version}, {} of payload, {} in total",
        out.display(),
        human(zip_bytes.len() as u64),
        human(total)
    );

    if let Some(dir) = &o.winget {
        let sha = sha256_file(&out)?;
        let url = o
            .url
            .clone()
            .unwrap_or_else(|| release_download_url(&version, &release_asset_name(&version)));
        let description = read_key(&cargo_toml, "description").unwrap_or_default();
        let m = Winget {
            version: &version,
            url: &url,
            sha256: &sha,
            release_date: o.release_date.as_deref(),
            description: &description,
        };
        std::fs::create_dir_all(dir)?;
        for (name, text) in m.files() {
            let path = dir.join(&name);
            std::fs::write(&path, text).with_context(|| format!("write {}", path.display()))?;
            println!("  {}", path.display());
        }
        println!("  winget: {WINGET_ID} {version}, installer SHA-256 {sha}");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("read {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02X}")).collect())
}

/// The winget manifests (schema 1.12.0) for one installer.
///
/// Two installer entries share the one file: `user` scope passes
/// `--just-me`, `machine` scope passes `--all-users` and is run elevated by
/// winget. The Apps & features key the setup writes is the product code, so
/// winget recognises an installation made by the wizard too, and `winget
/// upgrade` runs the new setup over it - which updates it in place.
struct Winget<'a> {
    version: &'a str,
    url: &'a str,
    sha256: &'a str,
    release_date: Option<&'a str>,
    description: &'a str,
}

/// The package identifier in the winget community repository. The
/// `winget` job of the release workflow names it too.
const WINGET_ID: &str = "RDS.RustDICOMStation";
const WINGET_SCHEMA: &str = "1.12.0";

impl Winget<'_> {
    fn files(&self) -> Vec<(String, String)> {
        vec![
            (format!("{WINGET_ID}.yaml"), self.version_manifest()),
            (
                format!("{WINGET_ID}.installer.yaml"),
                self.installer_manifest(),
            ),
            (
                format!("{WINGET_ID}.locale.en-US.yaml"),
                self.locale_manifest(),
            ),
        ]
    }

    fn header(kind: &str) -> String {
        format!(
            "# Created by rds-pack\n\
             # yaml-language-server: $schema=https://aka.ms/winget-manifest.{kind}.{WINGET_SCHEMA}.schema.json\n\n"
        )
    }

    fn version_manifest(&self) -> String {
        let mut s = Self::header("version");
        s += &format!("PackageIdentifier: {WINGET_ID}\n");
        s += &format!("PackageVersion: {}\n", q(self.version));
        s += "DefaultLocale: en-US\n";
        s += "ManifestType: version\n";
        s += &format!("ManifestVersion: {WINGET_SCHEMA}\n");
        s
    }

    fn installer_manifest(&self) -> String {
        let mut s = Self::header("installer");
        s += &format!("PackageIdentifier: {WINGET_ID}\n");
        s += &format!("PackageVersion: {}\n", q(self.version));
        s += "InstallerType: exe\n";
        s += "InstallModes:\n- interactive\n- silent\n- silentWithProgress\n";
        s += "UpgradeBehavior: install\n";
        // The codes `rds-setup` exits with (plan.rs, EXIT_*).
        s += "ExpectedReturnCodes:\n";
        for (code, response) in [
            (2, "packageInUse"),
            (3, "cancelledByUser"),
            (4, "downgrade"),
            (5, "noNetwork"),
        ] {
            s += &format!("- InstallerReturnCode: {code}\n  ReturnResponse: {response}\n");
        }
        s += "FileExtensions:\n- dcm\n- dicom\n";
        s += &format!("ProductCode: {PRODUCT_ID}\n");
        if let Some(date) = self.release_date {
            s += &format!("ReleaseDate: {date}\n");
        }
        s += "AppsAndFeaturesEntries:\n";
        s += &format!("- DisplayName: {}\n", q(APP_NAME));
        s += &format!("  Publisher: {}\n", q(PUBLISHER));
        s += &format!("  ProductCode: {PRODUCT_ID}\n");
        s += "Dependencies:\n  PackageDependencies:\n  - PackageIdentifier: Microsoft.VCRedist.2015+.x64\n";
        s += "Installers:\n";
        for (scope, flag) in [("user", "--just-me"), ("machine", "--all-users")] {
            s += "- Architecture: x64\n";
            s += &format!("  Scope: {scope}\n");
            s += &format!("  InstallerUrl: {}\n", self.url);
            s += &format!("  InstallerSha256: {}\n", self.sha256);
            if scope == "machine" {
                s += "  ElevationRequirement: elevationRequired\n";
            }
            // Every switch in every entry: an entry's own InstallerSwitches
            // are not merged with ones written at the top level by every
            // tool that edits these files.
            s += "  InstallerSwitches:\n";
            s += "    Silent: --silent\n";
            s += "    SilentWithProgress: --passive\n";
            s += "    InstallLocation: --dir \"<INSTALLPATH>\"\n";
            s += &format!("    Custom: {flag} --no-launch\n");
        }
        s += "ManifestType: installer\n";
        s += &format!("ManifestVersion: {WINGET_SCHEMA}\n");
        s
    }

    fn locale_manifest(&self) -> String {
        let repo = format!("https://github.com/{RELEASE_REPO}");
        let owner = RELEASE_REPO.split('/').next().unwrap_or(RELEASE_REPO);
        let mut s = Self::header("defaultLocale");
        s += &format!("PackageIdentifier: {WINGET_ID}\n");
        s += &format!("PackageVersion: {}\n", q(self.version));
        s += "PackageLocale: en-US\n";
        s += &format!("Publisher: {}\n", q(PUBLISHER));
        s += &format!("PublisherUrl: https://github.com/{owner}\n");
        s += &format!("PublisherSupportUrl: {repo}/issues\n");
        s += &format!("PackageName: {}\n", q(APP_NAME));
        s += &format!("PackageUrl: {repo}\n");
        s += "License: MIT\n";
        s += &format!("LicenseUrl: {repo}/blob/main/LICENSE.txt\n");
        s += &format!(
            "ShortDescription: {}\n",
            q(if self.description.is_empty() {
                "DICOM / RT DICOM viewer in pure Rust"
            } else {
                self.description
            })
        );
        s += &format!(
            "Description: {}\n",
            q(
                "Open-source DICOM and RT DICOM station for medical imaging, radiotherapy \
               research, analysis and QA: CT/MR volumes with RTSTRUCT, RTDOSE and RTPLAN, \
               registration, contouring, DVH, 4D motion analysis and pure-Rust \
               auto-segmentation. Not a medical device."
            )
        );
        s += "Moniker: rust-dicom-station\n";
        s += "Tags:\n";
        for tag in [
            "dicom",
            "medical-imaging",
            "radiotherapy",
            "rtstruct",
            "dvh",
            "segmentation",
            "viewer",
        ] {
            s += &format!("- {tag}\n");
        }
        s += &format!("ReleaseNotesUrl: {repo}/releases/tag/v{}\n", self.version);
        s += "ManifestType: defaultLocale\n";
        s += &format!("ManifestVersion: {WINGET_SCHEMA}\n");
        s
    }
}

/// A YAML double-quoted scalar.
fn q(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Add every file under `<repo>/<rel>` to the payload list.
fn collect_dir(repo: &Path, rel: &str, files: &mut Vec<(String, PathBuf)>) -> Result<()> {
    let root = repo.join(rel);
    if !root.is_dir() {
        return Ok(());
    }
    for e in walkdir::WalkDir::new(&root).sort_by_file_name() {
        let e = e?;
        if !e.file_type().is_file() {
            continue;
        }
        let name = e
            .path()
            .strip_prefix(repo)?
            .to_string_lossy()
            .replace('\\', "/");
        files.push((name, e.path().to_path_buf()));
    }
    Ok(())
}

/// `version = "x.y.z"` from the viewer's `Cargo.toml`.
fn read_version(cargo_toml: &Path) -> Option<String> {
    read_key(cargo_toml, "version")
}

/// A `key = "value"` line of the `[package]` table of a `Cargo.toml`.
fn read_key(cargo_toml: &Path, key: &str) -> Option<String> {
    let text = std::fs::read_to_string(cargo_toml).ok()?;
    text.lines()
        .take_while(|l| !l.trim_start().starts_with("[dependencies"))
        .find_map(|l| {
            let (k, v) = l.split_once('=')?;
            (k.trim() == key).then(|| v.trim().trim_matches('"').to_string())
        })
}

/// If `exe` was already packed, return the length of its payload-free prefix.
fn read_existing_footer(exe: &Path) -> Result<Option<u64>> {
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::File::open(exe)?;
    let len = f.metadata()?.len();
    if len < 24 {
        return Ok(None);
    }
    let mut buf = [0u8; 24];
    f.seek(SeekFrom::End(-24))?;
    f.read_exact(&mut buf)?;
    if &buf[..8] != b"RDSPAY01" {
        return Ok(None);
    }
    Ok(Some(u64::from_le_bytes(buf[8..16].try_into().unwrap())))
}

fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.0} MB", b / 1e6)
    } else {
        format!("{:.0} kB", b / 1e3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The updater follows `RELEASE_REPO`; the viewer's `Cargo.toml` names
    /// the repository too. If they ever disagreed, the installed program
    /// would look for updates somewhere else than its source.
    #[test]
    fn the_release_repository_is_the_one_cargo_names() {
        let toml = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Cargo.toml");
        assert_eq!(
            read_key(&toml, "repository"),
            Some(format!("https://github.com/{RELEASE_REPO}"))
        );
    }

    fn manifests() -> Vec<(String, String)> {
        Winget {
            version: "0.8.9",
            url: "https://example.org/setup.exe",
            sha256: "AB",
            release_date: Some("2026-09-10"),
            description: "A \"quoted\" viewer",
        }
        .files()
    }

    #[test]
    fn the_three_manifests_agree_on_identity_and_schema() {
        let files = manifests();
        assert_eq!(files.len(), 3);
        for (name, text) in &files {
            assert!(name.starts_with(WINGET_ID), "{name}");
            assert!(
                text.contains(&format!("PackageIdentifier: {WINGET_ID}\n")),
                "{name}"
            );
            assert!(text.contains("PackageVersion: \"0.8.9\"\n"), "{name}");
            assert!(
                text.ends_with(&format!("ManifestVersion: {WINGET_SCHEMA}\n")),
                "{name}"
            );
            assert!(!text.contains('\t'), "YAML has no tabs: {name}");
        }
    }

    #[test]
    fn both_scopes_install_the_same_file_with_their_own_switch() {
        let installer = &manifests()[1].1;
        assert_eq!(
            installer
                .matches("InstallerUrl: https://example.org/setup.exe")
                .count(),
            2
        );
        assert_eq!(installer.matches("InstallerSha256: AB").count(), 2);
        assert!(installer.contains("  Scope: user\n"));
        assert!(installer.contains("  Scope: machine\n  InstallerUrl"));
        assert!(installer.contains("    Custom: --just-me --no-launch\n"));
        assert!(installer.contains("    Custom: --all-users --no-launch\n"));
        assert_eq!(
            installer
                .matches("ElevationRequirement: elevationRequired")
                .count(),
            1
        );
        assert!(installer.contains(&format!("ProductCode: {PRODUCT_ID}\n")));
        assert!(installer.contains("ReleaseDate: 2026-09-10\n"));
    }

    #[test]
    fn text_is_quoted_so_yaml_reads_it_back_verbatim() {
        let locale = &manifests()[2].1;
        assert!(locale.contains("ShortDescription: \"A \\\"quoted\\\" viewer\"\n"));
        assert_eq!(q(r"C:\x"), r#""C:\\x""#);
    }
}
