//! The product's identity: names and version rules shared by the setup
//! program and by `rds-pack`, which writes the winget manifests.
//!
//! Plain data and pure functions only - `rds-pack` includes this file by
//! path, and it builds without the Win32 layer the setup program uses.

use std::cmp::Ordering;

pub const APP_NAME: &str = "Rust DICOM Station";
pub const APP_EXE: &str = "rust-dicom-station.exe";
/// The MCP server, shipped beside the viewer by `rds-pack` when it was built.
/// Optional at install time: it is only of use to somebody who drives the
/// station from an MCP client, and it is a second executable to trust and to
/// keep up to date for everybody else.
pub const MCP_EXE: &str = "rds-mcp.exe";
/// Shown in Apps & features, and matched by winget against the manifest's
/// `AppsAndFeaturesEntries`.
pub const PUBLISHER: &str = "Rust DICOM Station contributors";
/// Registry-safe product id: the name of the Add/Remove Programs key, which
/// winget reads as the product code.
pub const PRODUCT_ID: &str = "RustDicomStation";
/// `owner/name` of the GitHub repository whose releases the updater follows.
/// Must match `repository` in the viewer's `Cargo.toml` (asserted by a test
/// in `rds-pack`).
pub const RELEASE_REPO: &str = "alexprotom/rust-dicom-station";

/// File name of the Windows installer attached to a GitHub release. The
/// release workflow names it this way, the updater downloads it by this
/// name and looks it up in `SHA256SUMS` by it.
pub fn release_asset_name(version: &str) -> String {
    format!("rust-dicom-station-{version}-windows-x86_64.exe")
}

/// `https://github.com/<repo>/releases/download/v<version>/<file>`.
pub fn release_download_url(version: &str, file: &str) -> String {
    format!("https://github.com/{RELEASE_REPO}/releases/download/v{version}/{file}")
}

/// Compare two version strings the way the release workflow writes them:
/// dot-separated numbers, an optional leading `v`, and an optional
/// pre-release suffix after `-` that sorts before the release it precedes.
/// `0.8.10` is newer than `0.8.9`; `1.0` equals `1.0.0`.
pub fn compare_versions(a: &str, b: &str) -> Ordering {
    let (an, ap) = split_version(a);
    let (bn, bp) = split_version(b);
    let len = an.len().max(bn.len());
    for i in 0..len {
        let x = an.get(i).copied().unwrap_or(0);
        let y = bn.get(i).copied().unwrap_or(0);
        match x.cmp(&y) {
            Ordering::Equal => {}
            other => return other,
        }
    }
    match (ap, bp) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => x.cmp(y),
    }
}

fn split_version(v: &str) -> (Vec<u64>, Option<&str>) {
    let v = v.trim();
    let v = v.strip_prefix(['v', 'V']).unwrap_or(v);
    let v = v.split('+').next().unwrap_or(v);
    let (nums, pre) = match v.split_once('-') {
        Some((n, p)) => (n, Some(p)),
        None => (v, None),
    };
    let nums = nums
        .split('.')
        .map(|p| {
            // A part that is not a number ends the comparison at that point,
            // rather than making the whole version unreadable.
            p.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse::<u64>()
                .unwrap_or(0)
        })
        .collect();
    (nums, pre)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering::*;

    #[test]
    fn versions_compare_numerically_not_as_text() {
        assert_eq!(compare_versions("0.8.10", "0.8.9"), Greater);
        assert_eq!(compare_versions("0.9.0", "0.10.0"), Less);
        assert_eq!(compare_versions("1.0.0", "0.99.99"), Greater);
    }

    #[test]
    fn the_spellings_of_one_version_are_equal() {
        assert_eq!(compare_versions("v0.8.9", "0.8.9"), Equal);
        assert_eq!(compare_versions("1.0", "1.0.0"), Equal);
        assert_eq!(compare_versions(" 0.8.9 ", "0.8.9+build.7"), Equal);
    }

    #[test]
    fn a_pre_release_comes_before_its_release() {
        assert_eq!(compare_versions("0.9.0-rc1", "0.9.0"), Less);
        assert_eq!(compare_versions("0.9.0-rc2", "0.9.0-rc1"), Greater);
        assert_eq!(compare_versions("0.9.0-rc1", "0.8.9"), Greater);
    }

    #[test]
    fn the_asset_name_is_the_one_the_release_workflow_uploads() {
        assert_eq!(
            release_asset_name("0.8.9"),
            "rust-dicom-station-0.8.9-windows-x86_64.exe"
        );
        assert_eq!(
            release_download_url("0.8.9", "SHA256SUMS"),
            "https://github.com/alexprotom/rust-dicom-station/releases/download/v0.8.9/SHA256SUMS"
        );
    }
}
