#!/usr/bin/env bash
# Build the macOS package: the viewer and the MCP server for one
# architecture, wrapped in a .app bundle and handed out as a .dmg. No Xcode
# project, no third-party packaging tool - the bundle is a folder with a
# plist in it, and hdiutil makes the disk image.
#
#   ./build-app.sh                      # this Mac's architecture, release
#   ./build-app.sh --arch x86_64        # Intel, cross-compiled from anywhere
#   ./build-app.sh --arch arm64         # Apple Silicon
#   ./build-app.sh --dev                # dev profile (quick to build, slow to run)
#   ./build-app.sh --skip-build         # package what the last build left
#   ./build-app.sh --no-dmg             # leave the .app, make no disk image
#
# Needs: macOS with the Xcode command line tools (codesign, hdiutil, sips,
# iconutil, otool, lipo - `xcode-select --install`), a Rust toolchain, and
# the target for the architecture asked for:
#
#     rustup target add x86_64-apple-darwin aarch64-apple-darwin
#
# An Apple Silicon Mac builds both; an Intel Mac can only build its own.
#
# macOS 12 Monterey is the floor. MACOSX_DEPLOYMENT_TARGET is exported
# before cargo, written into LSMinimumSystemVersion, and - because the two
# can drift apart without anything complaining until a user on an old
# machine is told the program is damaged - read back out of the linked
# binary afterwards and compared. A mismatch fails the build.
#
# Signing: with RDS_CODESIGN_IDENTITY set (a "Developer ID Application:
# ... (TEAMID)" identity in a keychain the session can reach) the bundle
# and the disk image are signed with it under the hardened runtime, which
# is what lets them be notarised. Without it everything is signed ad-hoc:
# it runs, but the first launch has to go through Finder's right-click ▸
# Open, because an ad-hoc signature carries no developer to trust.
#
# Notarisation: with RDS_NOTARY_APPLE_ID, RDS_NOTARY_PASSWORD (an
# app-specific password) and RDS_NOTARY_TEAM_ID set, the disk image is
# submitted to Apple and the ticket is stapled into it, so the program
# opens by double-clicking on a machine that has never seen it.
#
# Result: out/rust-dicom-station-<version>-macos-<arch>.dmg
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$here/out"

app_name="Rust DICOM Station"
exe_name="rust-dicom-station"
mcp_name="rds-mcp"
icon_stem="rust-dicom-station"

# The floor, in one place. Everything else reads it from here.
min_macos="${MACOSX_DEPLOYMENT_TARGET:-12.0}"

arch="$(uname -m)"
profile=release
build=1
make_dmg=1
while [ $# -gt 0 ]; do
    case "$1" in
        --arch) arch="${2:?--arch needs arm64 or x86_64}"; shift 2 ;;
        --arch=*) arch="${1#*=}"; shift ;;
        --dev) profile=dev; shift ;;
        --skip-build) build=0; shift ;;
        --no-dmg) make_dmg=0; shift ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$arch" in
    arm64|aarch64) arch=arm64; target=aarch64-apple-darwin ;;
    x86_64|x64|intel) arch=x86_64; target=x86_64-apple-darwin ;;
    *) echo "unknown architecture: $arch (arm64 or x86_64)" >&2; exit 2 ;;
esac

[ "$(uname -s)" = Darwin ] || {
    echo "this script builds a macOS bundle and needs macOS: codesign, hdiutil," >&2
    echo "sips and iconutil have no equivalent elsewhere." >&2
    exit 1
}

# ---- the version, from the viewer's Cargo.toml ----------------------------
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -n1)"
[ -n "$version" ] || { echo "could not read the version from $root/Cargo.toml" >&2; exit 1; }
echo "rust-dicom-station $version, macOS $arch ($target), $profile, min macOS $min_macos"

# ---- 1. the executables ----------------------------------------------------
# --features mcp adds the second binary, rds-mcp, which goes into the bundle
# beside the viewer; that is the whole of the macOS MCP story (docs/mcp.md).
mkdir -p "$out"
if [ "$build" = 1 ]; then
    # --profile rather than an optional --release: macOS still ships bash
    # 3.2, which cannot expand an empty array under `set -u`.
    (
        cd "$root"
        MACOSX_DEPLOYMENT_TARGET="$min_macos" \
            cargo build --features mcp --target "$target" --profile "$profile"
    )
fi

# `--profile dev` still writes into target/<triple>/debug.
built="$root/target/$target/release"
if [ "$profile" = dev ]; then built="$root/target/$target/debug"; fi
for f in "$exe_name" "$mcp_name"; do
    [ -f "$built/$f" ] || { echo "$built/$f is missing - build it first" >&2; exit 1; }
done

# ---- 2. the icon -----------------------------------------------------------
# One .icns made from the same PNG that is the window icon, the Windows
# resource and the Linux desktop icon (src/icon.rs). Only the sizes the
# source can actually fill are written: upscaling a 256 px picture to 1024
# would put a blurred representation in the bundle that macOS would prefer
# over the sharp one. Drop a larger assets/rust-dicom-station.png in and the
# bigger entries appear by themselves.
png="$root/assets/$icon_stem.png"
[ -f "$png" ] || { echo "$png is missing" >&2; exit 1; }
src_px="$(sips -g pixelWidth "$png" | awk '/pixelWidth:/ {print $2}')"
[ -n "$src_px" ] || { echo "could not read the size of $png" >&2; exit 1; }

iconset="$out/$icon_stem.iconset"
rm -rf "$iconset"
mkdir -p "$iconset"
icon_reps=0
for base in 16 32 128 256 512; do
    for scale in 1 2; do
        px=$((base * scale))
        [ "$px" -le "$src_px" ] || continue
        name="icon_${base}x${base}.png"
        [ "$scale" = 2 ] && name="icon_${base}x${base}@2x.png"
        sips -z "$px" "$px" "$png" --out "$iconset/$name" >/dev/null
        icon_reps=$((icon_reps + 1))
    done
done
[ "$icon_reps" -gt 0 ] || { echo "$png is ${src_px}px - too small for any icon size" >&2; exit 1; }
icns="$out/$icon_stem.icns"
iconutil --convert icns "$iconset" --output "$icns"
echo "icon: ${src_px}px source, $icon_reps representations"

# ---- 3. the bundle ---------------------------------------------------------
app="$out/$app_name.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"

cp "$built/$exe_name" "$app/Contents/MacOS/$exe_name"
cp "$built/$mcp_name" "$app/Contents/MacOS/$mcp_name"
chmod +x "$app/Contents/MacOS/$exe_name" "$app/Contents/MacOS/$mcp_name"
cp "$icns" "$app/Contents/Resources/$icon_stem.icns"

sed -e "s/@VERSION@/$version/g" -e "s/@MIN_MACOS@/$min_macos/g" \
    "$here/Info.plist.in" > "$app/Contents/Info.plist"
plutil -lint "$app/Contents/Info.plist" >/dev/null
printf 'APPL????' > "$app/Contents/PkgInfo"

# ---- 4. what was actually built --------------------------------------------
# Three things that are cheap to check here and expensive to discover on a
# user's machine: the wrong architecture, a deployment target that drifted
# away from LSMinimumSystemVersion, and a bundle that does not hold both
# executables.
for f in "$exe_name" "$mcp_name"; do
    bin="$app/Contents/MacOS/$f"
    have_arch="$(lipo -archs "$bin")"
    [ "$have_arch" = "$arch" ] || {
        echo "::error::$f is $have_arch, not $arch" >&2
        exit 1
    }
    # LC_BUILD_VERSION carries the deployment target of a modern binary;
    # LC_VERSION_MIN_MACOSX is the older spelling, still emitted for low
    # targets. Whichever is there has to say what the plist says.
    have_min="$(otool -l "$bin" \
        | awk '/LC_BUILD_VERSION|LC_VERSION_MIN_MACOSX/ {f=1} f && /minos|version/ && $1 ~ /^(minos|version)$/ {print $2; exit}')"
    [ -n "$have_min" ] || { echo "::error::$f carries no deployment target" >&2; exit 1; }
    # 12.0 and 12 are the same floor; compare as numbers.
    if [ "$(echo "$have_min" | awk -F. '{printf "%d%02d", $1, $2}')" \
         -ne "$(echo "$min_macos" | awk -F. '{printf "%d%02d", $1, $2}')" ]; then
        echo "::error::$f was built for macOS $have_min, the bundle claims $min_macos" >&2
        exit 1
    fi
    echo "  $f: $have_arch, macOS $have_min and newer"
done

# ---- 5. signing -------------------------------------------------------------
# Inside out: a bundle's signature covers everything under it, so the helper
# executable is signed first and the bundle last. --deep is not used - Apple
# deprecated it and it is the usual way a helper ends up signed with the
# wrong entitlements.
identity="${RDS_CODESIGN_IDENTITY:-}"
if [ -n "$identity" ]; then
    sign=(codesign --force --timestamp --options runtime
          --entitlements "$here/entitlements.plist" --sign "$identity")
    echo "signing with $identity (hardened runtime)"
else
    sign=(codesign --force --sign -)
    echo "no RDS_CODESIGN_IDENTITY - signing ad-hoc; the first launch needs Finder ▸ right-click ▸ Open"
fi
"${sign[@]}" "$app/Contents/MacOS/$mcp_name"
"${sign[@]}" "$app/Contents/MacOS/$exe_name"
"${sign[@]}" "$app"
codesign --verify --strict --verbose=2 "$app"

if [ "$make_dmg" = 0 ]; then
    echo "bundle: $app"
    exit 0
fi

# ---- 6. the disk image -------------------------------------------------------
# A staging folder with the bundle and a link to /Applications, which is what
# makes the window people expect: drag the icon onto the folder. ULFO is
# lzfse-compressed and readable from macOS 10.11, well below this floor.
stage="$out/dmg"
rm -rf "$stage"
mkdir -p "$stage"
cp -R "$app" "$stage/$app_name.app"
ln -s /Applications "$stage/Applications"

dmg="$out/rust-dicom-station-$version-macos-$arch.dmg"
rm -f "$dmg"
hdiutil create \
    -volname "$app_name $version" \
    -srcfolder "$stage" \
    -fs HFS+ \
    -format ULFO \
    -ov \
    "$dmg" >/dev/null
rm -rf "$stage"

# A signed disk image is what Safari checks before it is ever opened.
if [ -n "$identity" ]; then
    codesign --force --timestamp --sign "$identity" "$dmg"
fi

# ---- 7. notarisation ---------------------------------------------------------
# Apple staples the ticket into the image, so a machine that has never seen
# the program - and may be offline - still opens it by double-clicking.
if [ -n "${RDS_NOTARY_APPLE_ID:-}" ] && [ -n "${RDS_NOTARY_PASSWORD:-}" ] && [ -n "${RDS_NOTARY_TEAM_ID:-}" ]; then
    [ -n "$identity" ] || {
        echo "::error::notarisation needs a Developer ID signature; set RDS_CODESIGN_IDENTITY" >&2
        exit 1
    }
    echo "submitting $(basename "$dmg") to Apple..."
    xcrun notarytool submit "$dmg" \
        --apple-id "$RDS_NOTARY_APPLE_ID" \
        --password "$RDS_NOTARY_PASSWORD" \
        --team-id "$RDS_NOTARY_TEAM_ID" \
        --wait
    xcrun stapler staple "$dmg"
    xcrun stapler validate "$dmg"
    # Gatekeeper's own verdict, which is the thing a user will get.
    spctl --assess --type open --context context:primary-signature -vv "$dmg"
else
    echo "no RDS_NOTARY_* - the image is not notarised (docs/macos.md)"
fi

ls -lh "$dmg"
echo "$dmg"
