//! The viewer's version, for the log line and the panic file.
//!
//! This crate's own version is the front end's (0.1.0); what a user and a
//! bug report need is the viewer's, which `build-ipa.sh` also writes into
//! the bundle's `CFBundleShortVersionString`. It is read from the viewer's
//! `Cargo.toml` two folders up and handed to the compiler as `RDS_VERSION`.

fn main() {
    let manifest = std::path::Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../Cargo.toml");
    println!("cargo:rerun-if-changed={}", manifest.display());
    let text = std::fs::read_to_string(&manifest).unwrap_or_default();
    let version = text
        .lines()
        .find_map(|l| {
            l.strip_prefix("version = \"")
                .and_then(|rest| rest.strip_suffix('"'))
        })
        .unwrap_or(env!("CARGO_PKG_VERSION"));
    println!("cargo:rustc-env=RDS_VERSION={version}");
}
