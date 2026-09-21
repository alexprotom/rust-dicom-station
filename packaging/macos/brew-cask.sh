#!/usr/bin/env bash
# Write the Homebrew cask for a released version.
#
#   ./brew-cask.sh --version 0.9.5 \
#                  --arm-sha256 <sha> --intel-sha256 <sha> \
#                  [--out Casks/rust-dicom-station.rb]
#
# One cask covers both architectures: Homebrew fills #{arch} with arm64 or
# x86_64 and picks the matching checksum, so `brew install --cask
# rust-dicom-station` downloads the right disk image on either Mac.
#
# The release workflow runs this after the GitHub Release exists, with the
# two checksums out of SHA256SUMS, and commits the result to the tap named
# by the repository variable HOMEBREW_TAP (docs/macos.md). Nothing here
# needs Homebrew installed - it writes a Ruby file.
#
# Runs anywhere; it is the only script in this folder that is not macOS-only.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

version=""
arm_sha=""
intel_sha=""
outfile=""
repo=""
while [ $# -gt 0 ]; do
    case "$1" in
        --version) version="${2:?}"; shift 2 ;;
        --arm-sha256) arm_sha="${2:?}"; shift 2 ;;
        --intel-sha256) intel_sha="${2:?}"; shift 2 ;;
        --out) outfile="${2:?}"; shift 2 ;;
        --repo) repo="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

[ -n "$version" ] || version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -n1)"
[ -n "$repo" ] || repo="$(sed -n 's#^repository = "https://github.com/\([^"]*\)"#\1#p' "$root/Cargo.toml" | head -n1)"

[ -n "$version" ]   || { echo "no version: pass --version, or run inside the repository" >&2; exit 2; }
[ -n "$repo" ]      || { echo "no repository: pass --repo owner/name" >&2; exit 2; }
[ -n "$arm_sha" ]   || { echo "no checksum for Apple Silicon: pass --arm-sha256" >&2; exit 2; }
[ -n "$intel_sha" ] || { echo "no checksum for Intel: pass --intel-sha256" >&2; exit 2; }
# A checksum that is not 64 hex characters is a copied-wrong line, and the
# cask would fail for every user rather than in this job.
for sha in "$arm_sha" "$intel_sha"; do
    case "$sha" in
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
        *) echo "not a sha256: $sha" >&2; exit 2 ;;
    esac
    [ "${#sha}" = 64 ] || { echo "sha256 is ${#sha} characters, not 64: $sha" >&2; exit 2; }
done

# Not the Cargo description: Homebrew wants one short line that does not
# begin with an article or with the cask's own name, and the manifest's
# description is a sentence with parentheses in it.
desc="DICOM and RT DICOM viewer with a three-view MPR layout"

cask="$(cat <<RUBY
cask "rust-dicom-station" do
  arch arm: "arm64", intel: "x86_64"

  version "$version"
  sha256 arm:   "$arm_sha",
         intel: "$intel_sha"

  url "https://github.com/$repo/releases/download/v#{version}/rust-dicom-station-#{version}-macos-#{arch}.dmg"
  name "Rust DICOM Station"
  desc "$desc"
  homepage "https://github.com/$repo"

  livecheck do
    url :url
    strategy :github_latest
  end

  # The bundle is built for macOS 12 and newer (packaging/macos/build-app.sh checks
  # that the binaries agree with the plist).
  depends_on macos: ">= :monterey"

  app "Rust DICOM Station.app"
  # The MCP server ships inside the bundle; linking it into the prefix is
  # what lets an MCP client be pointed at plain \`rds-mcp\` (docs/mcp.md).
  binary "#{appdir}/Rust DICOM Station.app/Contents/MacOS/rds-mcp"

  zap trash: [
    "~/Library/Application Support/RustDICOMStation",
    "~/Library/Saved Application State/io.github.alexprotom.rust-dicom-station.savedState",
  ]
end
RUBY
)"

if [ -n "$outfile" ]; then
    mkdir -p "$(dirname "$outfile")"
    printf '%s\n' "$cask" > "$outfile"
    echo "wrote $outfile"
else
    printf '%s\n' "$cask"
fi
