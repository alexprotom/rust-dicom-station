#!/usr/bin/env bash
# Build the Linux AppImage: the viewer and the MCP server in one file.
#
#   ./build-appimage.sh                  # release build, then the image
#   ./build-appimage.sh --skip-build     # package what the last build left
#   ./build-appimage.sh --linuxdeploy /path/to/linuxdeploy-x86_64.AppImage
#
# Needs: a Rust toolchain, the GUI build dependencies of the viewer
# (libxkbcommon-dev libwayland-dev libx11-dev libxcursor-dev libxrandr-dev
# libxi-dev libgl1-mesa-dev), `file`, `wget` and libfuse2 (linuxdeploy is
# itself an AppImage). linuxdeploy is downloaded into out/ unless a copy is
# named with --linuxdeploy.
#
# Models are intentionally NOT included: the application downloads them at
# run time into ~/.local/share/RustDICOMStation/models.
#
# Result: out/rust-dicom-station-<version>-linux-x86_64.AppImage
#
# The release workflow runs this on every push to main (docs/release-versioning.md).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
out="$here/out"

build=1
linuxdeploy=""
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) build=0; shift ;;
        --linuxdeploy) linuxdeploy="${2:?--linuxdeploy needs a path}"; shift 2 ;;
        --linuxdeploy=*) linuxdeploy="${1#*=}"; shift ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

# ---- the version, from the viewer's Cargo.toml ----------------------------
version="$(sed -nE 's/^version = "([^"]+)"/\1/p' "$root/Cargo.toml" | head -n 1)"
[ -n "$version" ] || { echo "could not read the version from $root/Cargo.toml" >&2; exit 1; }
output="rust-dicom-station-$version-linux-x86_64.AppImage"
echo "rust-dicom-station $version, Linux x86_64 AppImage"

# ---- 1. the executables ----------------------------------------------------
# `--features mcp` adds the second executable, rds-mcp, which the AppImage
# carries beside the viewer.
if [ "$build" = 1 ]; then
    (cd "$root" && cargo build --release --features mcp)
fi
built="$root/target/release"
for f in rust-dicom-station rds-mcp; do
    [ -f "$built/$f" ] || { echo "$built/$f is missing - build it first" >&2; exit 1; }
done

# ---- 2. the AppDir ---------------------------------------------------------
appdir="$out/AppDir"
rm -rf "$appdir"
mkdir -p "$appdir/usr/bin" \
         "$appdir/usr/share/applications" \
         "$appdir/usr/share/icons/hicolor/256x256/apps"

cp "$built/rust-dicom-station" "$appdir/usr/bin/rust-dicom-station"
cp "$built/rds-mcp" "$appdir/usr/bin/rds-mcp"
cp "$root/assets/rust-dicom-station.png" \
   "$appdir/usr/share/icons/hicolor/256x256/apps/rust-dicom-station.png"
cp "$here/rust-dicom-station.desktop" "$appdir/usr/share/applications/"

# The AppRun is copied in and also named to linuxdeploy explicitly below,
# rather than left to its "keep whatever is in the AppDir" behaviour: what
# makes the MCP server reachable at all is the dispatcher in it, and that is
# not something to leave to a flag-free default in a tool pinned to
# `continuous`.
cp "$here/AppRun" "$appdir/AppRun"
chmod +x "$appdir/AppRun"

# ---- 3. linuxdeploy --------------------------------------------------------
if [ -z "$linuxdeploy" ]; then
    linuxdeploy="$out/linuxdeploy-x86_64.AppImage"
    if [ ! -x "$linuxdeploy" ]; then
        echo "downloading linuxdeploy"
        wget -q \
            https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage \
            -O "$linuxdeploy"
        chmod +x "$linuxdeploy"
    fi
fi

# ---- 4. the image ----------------------------------------------------------
# linuxdeploy writes the image into the current directory, named after the
# desktop entry; it is moved to its release name afterwards.
rm -f "$out"/*.AppImage.tmp
(
    cd "$out"
    rm -f "$output"
    NO_STRIP=1 "$linuxdeploy" \
        --appdir "$appdir" \
        --custom-apprun "$here/AppRun" \
        --output appimage
    made="$(find . -maxdepth 1 -type f -name '*.AppImage' \
        ! -name 'linuxdeploy-*.AppImage' ! -name "$output" -print -quit)"
    [ -n "$made" ] || { echo "linuxdeploy produced no AppImage" >&2; exit 1; }
    mv "$made" "$output"
    chmod +x "$output"
)

# ---- 5. checks -------------------------------------------------------------
file "$out/$output"
[ -x "$out/$output" ] || { echo "$output is not executable" >&2; exit 1; }

# The MCP server has to be reachable, which means the dispatcher in AppRun
# survived the packaging. `--check` reads the configuration and exits without
# speaking the protocol, so it is safe to run here.
if ! "$out/$output" mcp --check; then
    echo "the AppImage does not run the MCP server on 'mcp'" >&2
    exit 1
fi

echo "Created $out/$output"
