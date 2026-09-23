#!/usr/bin/env bash
# Build the iOS / iPadOS package: the viewer for arm64 iPads and iPhones
# (one binary for both), wrapped in an .app bundle and handed out as an .ipa.
# No Xcode project and no third-party packaging tool - the bundle is a folder
# with a plist in it, actool compiles the icon, codesign signs, and an .ipa
# is a zip of Payload/.
#
#   ./build-ipa.sh                 # device build, release, -> out/*.ipa
#   ./build-ipa.sh --simulator     # Apple Silicon simulator build, -> out/*-simulator.zip
#   ./build-ipa.sh --dev           # dev profile (quick to build, slow to run)
#   ./build-ipa.sh --skip-build    # package what the last build left
#   ./build-ipa.sh --no-package    # leave the .app, make no .ipa / .zip
#
# Needs: macOS with Xcode (xcrun, actool, codesign, PlistBuddy, plutil,
# otool - the command line tools alone have no iOS SDK), a Rust toolchain,
# and the target:
#
#     rustup target add aarch64-apple-ios aarch64-apple-ios-sim
#
# iOS / iPadOS 15 is the floor (IPHONEOS_DEPLOYMENT_TARGET). It is exported
# before cargo, written into MinimumOSVersion, and read back out of the
# linked binary afterwards; a mismatch fails the build.
#
# Versions: CFBundleShortVersionString is the viewer's version from
# Cargo.toml. CFBundleVersion (the build number the App Store and TestFlight
# want unique and rising) is `major*10000+minor*100+patch` followed by the
# minutes since 2024-01-01 UTC, e.g. 909.1426211; set RDS_IOS_BUILD to
# override it.
#
# Signing (device builds):
#
#   RDS_IOS_SIGN_IDENTITY   a signing identity in a keychain the session can
#                           reach ("Apple Distribution: ... (TEAMID)", its
#                           SHA-1, or "Apple Development: ...")
#   RDS_IOS_PROFILE         the matching .mobileprovision (development,
#                           ad hoc or App Store)
#   RDS_IOS_BUNDLE_ID       optional; otherwise taken from the profile when
#                           it names one, else io.github.alexprotom.rust-dicom-station
#
# With both set the app is signed for the devices (or the store) the profile
# covers, with the profile's entitlements, and installs like any other app.
# Without them it is signed ad hoc: the .ipa is complete and valid, and a
# sideloading tool (AltStore, Sideloadly, Apple Configurator with your own
# profile) re-signs it with an Apple ID when it installs it. iOS itself runs
# nothing that no Apple certificate has signed - see docs/ios.md.
#
# Simulator builds are always signed ad hoc, which is all the simulator asks.
#
# Result: out/rust-dicom-station-<version>-ios.ipa
#         out/rust-dicom-station-<version>-ios-simulator.zip   (--simulator)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$here/out"

exe_name="rust-dicom-station"
app_dir_name="RustDICOMStation.app"
default_bundle_id="io.github.alexprotom.rust-dicom-station"

# The floor, in one place. Everything else reads it from here.
min_ios="${IPHONEOS_DEPLOYMENT_TARGET:-15.0}"

plistbuddy="${PLISTBUDDY:-/usr/libexec/PlistBuddy}"

simulator=0
profile=release
build=1
package=1
while [ $# -gt 0 ]; do
    case "$1" in
        --simulator|--sim) simulator=1; shift ;;
        --dev) profile=dev; shift ;;
        --skip-build) build=0; shift ;;
        --no-package) package=0; shift ;;
        -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$simulator" = 1 ]; then
    target=aarch64-apple-ios-sim
    sdk=iphonesimulator
    platform=iPhoneSimulator
    kind=simulator
else
    target=aarch64-apple-ios
    sdk=iphoneos
    platform=iPhoneOS
    kind=device
fi

command -v xcrun >/dev/null 2>&1 || {
    echo "this script builds an iOS app and needs macOS with Xcode: the iOS SDK," >&2
    echo "actool and codesign have no equivalent elsewhere." >&2
    exit 1
}
xcrun --sdk "$sdk" --show-sdk-path >/dev/null 2>&1 || {
    echo "no $sdk SDK - install Xcode (not only the command line tools) and" >&2
    echo "select it with: sudo xcode-select -s /Applications/Xcode.app" >&2
    exit 1
}

# ---- the version, from the viewer's Cargo.toml ----------------------------
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -n1)"
[ -n "$version" ] || { echo "could not read the version from $root/Cargo.toml" >&2; exit 1; }
IFS=. read -r v_major v_minor v_patch <<EOF
$version
EOF
v_patch="${v_patch%%[!0-9]*}"
code=$((10#$v_major * 10000 + 10#$v_minor * 100 + 10#${v_patch:-0}))
build_number="${RDS_IOS_BUILD:-$code.$((($(date -u +%s) - 1704067200) / 60))}"
echo "rust-dicom-station $version ($build_number), iOS $kind ($target), $profile, min iOS $min_ios"

# ---- 1. the executable -----------------------------------------------------
mkdir -p "$out"
if [ "$build" = 1 ]; then
    # --profile rather than an optional --release: macOS still ships bash
    # 3.2, which cannot expand an empty array under `set -u`.
    (
        cd "$here"
        IPHONEOS_DEPLOYMENT_TARGET="$min_ios" \
            cargo build --target "$target" --profile "$profile"
    )
fi

# `--profile dev` still writes into target/<triple>/debug.
built="$here/target/$target/release/$exe_name"
if [ "$profile" = dev ]; then built="$here/target/$target/debug/$exe_name"; fi
[ -f "$built" ] || { echo "$built is missing - build it first" >&2; exit 1; }

# ---- 2. the signing inputs, which decide the bundle identifier -------------
identity="${RDS_IOS_SIGN_IDENTITY:-}"
provision="${RDS_IOS_PROFILE:-}"
if [ "$simulator" = 1 ]; then
    identity=""
    provision=""
fi
if [ -n "$identity" ] && [ -z "$provision" ]; then
    echo "::error::RDS_IOS_SIGN_IDENTITY is set but RDS_IOS_PROFILE is not; a device" >&2
    echo "::error::build needs the provisioning profile that goes with the identity." >&2
    exit 1
fi

work="$out/work-$kind"
rm -rf "$work"
mkdir -p "$work"

bundle_id="${RDS_IOS_BUNDLE_ID:-}"
team_id=""
if [ -n "$provision" ]; then
    [ -f "$provision" ] || { echo "$provision does not exist" >&2; exit 1; }
    security cms -D -i "$provision" > "$work/profile.plist"
    app_id="$("$plistbuddy" -c 'Print :Entitlements:application-identifier' "$work/profile.plist")"
    team_id="$("$plistbuddy" -c 'Print :TeamIdentifier:0' "$work/profile.plist")"
    profile_bundle="${app_id#"$team_id".}"
    if [ -z "$bundle_id" ] && [ "$profile_bundle" != "*" ]; then
        bundle_id="$profile_bundle"
    fi
    echo "profile: $("$plistbuddy" -c 'Print :Name' "$work/profile.plist") (team $team_id, app id $app_id)"
fi
bundle_id="${bundle_id:-$default_bundle_id}"
if [ -n "$provision" ]; then
    case "$profile_bundle" in
        "*") ;;
        *\*) [ "${bundle_id#"${profile_bundle%\*}"}" != "$bundle_id" ] || {
                 echo "::error::the profile covers $app_id, not $bundle_id" >&2; exit 1; } ;;
        *) [ "$profile_bundle" = "$bundle_id" ] || {
                 echo "::error::the profile covers $app_id, not $bundle_id" >&2; exit 1; } ;;
    esac
fi
echo "bundle identifier: $bundle_id"

# ---- 3. the bundle -----------------------------------------------------------
# An iOS bundle is flat: executable, Info.plist, icons and the compiled
# asset catalog all side by side.
app="$out/$kind/$app_dir_name"
rm -rf "$app"
mkdir -p "$app"
cp "$built" "$app/$exe_name"
chmod +x "$app/$exe_name"

sed -e "s/@VERSION@/$version/g" \
    -e "s/@BUILD@/$build_number/g" \
    -e "s/@MIN_IOS@/$min_ios/g" \
    -e "s/@BUNDLE_ID@/$bundle_id/g" \
    -e "s/@PLATFORM@/$platform/g" \
    "$here/Info.plist.in" > "$app/Info.plist"
printf 'APPL????' > "$app/PkgInfo"
# The privacy manifest: App Store Connect refuses an upload without it (the
# comment in the file says which APIs and why).
cp "$here/PrivacyInfo.xcprivacy" "$app/PrivacyInfo.xcprivacy"
plutil -lint "$app/PrivacyInfo.xcprivacy" >/dev/null

# The icon: actool turns the asset catalog into Assets.car plus the loose
# PNGs older systems read, and writes the Info.plist entries that name them
# into a partial plist, which is merged in.
xcrun actool "$here/Assets.xcassets" \
    --compile "$app" \
    --platform "$sdk" \
    --minimum-deployment-target "$min_ios" \
    --target-device iphone \
    --target-device ipad \
    --app-icon AppIcon \
    --output-partial-info-plist "$work/icon-info.plist" \
    --output-format human-readable-text \
    --notices --warnings --errors
"$plistbuddy" -c "Merge $work/icon-info.plist" "$app/Info.plist"

# What Xcode adds to every Info.plist it writes, and what App Store Connect
# checks for: the SDK, the platform and the Xcode the app was built with.
set_key() { # key type value
    "$plistbuddy" -c "Delete :$1" "$app/Info.plist" >/dev/null 2>&1 || true
    "$plistbuddy" -c "Add :$1 $2 $3" "$app/Info.plist"
}
sdk_version="$(xcrun --sdk "$sdk" --show-sdk-version)"
sdk_build="$(xcrun --sdk "$sdk" --show-sdk-build-version)"
set_key DTPlatformName string "$sdk"
set_key DTPlatformVersion string "$sdk_version"
set_key DTPlatformBuild string "$sdk_build"
set_key DTSDKName string "$sdk$sdk_version"
set_key DTSDKBuild string "$sdk_build"
set_key DTCompiler string com.apple.compilers.llvm.clang.1_0
if command -v xcodebuild >/dev/null 2>&1; then
    xcode_version="$(xcodebuild -version | awk '/^Xcode/ {print $2}')"
    xcode_build="$(xcodebuild -version | awk '/^Build version/ {print $3}')"
    # "16.4" -> 1640, "26.0.1" -> 2601: two digits of major, one each of
    # minor and patch, as Xcode writes it.
    IFS=. read -r x_major x_minor x_patch <<EOF
$xcode_version
EOF
    set_key DTXcode string "$(printf '%02d%d%d' "$x_major" "${x_minor:-0}" "${x_patch:-0}")"
    set_key DTXcodeBuild string "$xcode_build"
fi
if command -v sw_vers >/dev/null 2>&1; then
    set_key BuildMachineOSBuild string "$(sw_vers -buildVersion)"
fi
plutil -lint "$app/Info.plist" >/dev/null

# ---- 4. what was actually built --------------------------------------------
# Cheap to check here, expensive to discover on an iPad: an executable for
# the wrong platform (device vs simulator) or a deployment target that
# drifted away from MinimumOSVersion.
build_version="$(otool -l "$app/$exe_name" | awk '
    /cmd LC_BUILD_VERSION/ {f=1}
    f && $1 == "platform" {p=$2}
    f && $1 == "minos" {m=$2; print p, m; exit}')"
have_platform="${build_version%% *}"
have_min="${build_version##* }"
[ -n "$have_min" ] || { echo "::error::$exe_name carries no LC_BUILD_VERSION" >&2; exit 1; }
# otool prints the platform as a number (2 iOS, 7 iOS simulator) or a name.
case "$kind:$have_platform" in
    device:2|device:IOS|device:ios) ;;
    simulator:7|simulator:IOSSIMULATOR|simulator:iossimulator) ;;
    *) echo "::error::$exe_name was built for platform $have_platform, not the iOS $kind" >&2; exit 1 ;;
esac
if [ "$(echo "$have_min" | awk -F. '{printf "%d%02d", $1, $2}')" \
     -ne "$(echo "$min_ios" | awk -F. '{printf "%d%02d", $1, $2}')" ]; then
    echo "::error::$exe_name was built for iOS $have_min, the bundle claims $min_ios" >&2
    exit 1
fi
echo "  $exe_name: $(lipo -archs "$app/$exe_name"), platform $have_platform, iOS $have_min and newer"

# ---- 5. signing ---------------------------------------------------------------
if [ -n "$identity" ]; then
    cp "$provision" "$app/embedded.mobileprovision"
    # The entitlements are the profile's own, with a wildcard application
    # identifier narrowed to this bundle; keychain groups are not used.
    "$plistbuddy" -x -c 'Print :Entitlements' "$work/profile.plist" > "$work/entitlements.plist"
    "$plistbuddy" -c "Set :application-identifier $team_id.$bundle_id" "$work/entitlements.plist"
    "$plistbuddy" -c 'Delete :keychain-access-groups' "$work/entitlements.plist" >/dev/null 2>&1 || true
    echo "signing with $identity"
    codesign --force --timestamp=none --generate-entitlement-der \
        --entitlements "$work/entitlements.plist" --sign "$identity" "$app"
else
    codesign --force --sign - "$app"
    if [ "$kind" = device ]; then
        echo "no RDS_IOS_SIGN_IDENTITY / RDS_IOS_PROFILE - signed ad hoc; install it with a"
        echo "sideloading tool, which re-signs it (docs/ios.md#installing)"
    fi
fi
codesign --verify --strict --verbose=2 "$app"

if [ "$package" = 0 ]; then
    echo "bundle: $app"
    exit 0
fi

# ---- 6. the package -------------------------------------------------------------
if [ "$kind" = device ]; then
    # An .ipa is a zip with the bundle under Payload/.
    result="$out/rust-dicom-station-$version-ios.ipa"
    rm -rf "$work/Payload" "$result"
    mkdir -p "$work/Payload"
    cp -R "$app" "$work/Payload/"
    (cd "$work" && zip -qry "$result" Payload)
else
    # The simulator installs a bare .app (drag it onto a booted simulator,
    # or `xcrun simctl install booted RustDICOMStation.app`); zipped for the
    # download.
    result="$out/rust-dicom-station-$version-ios-simulator.zip"
    rm -f "$result"
    (cd "$out/$kind" && zip -qry "$result" "$app_dir_name")
fi
rm -rf "$work"

ls -lh "$result"
echo "$result"
