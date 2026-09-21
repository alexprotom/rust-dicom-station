#!/usr/bin/env bash
# Build the Android package: the shared library through cargo-ndk, then the
# APK by hand with the SDK's own tools - aapt2, zipalign, apksigner. No
# Gradle, no Java sources; the manifest and the icons are in this folder.
#
#   ./build-apk.sh                 # release build, arm64-v8a, debug-signed
#   ./build-apk.sh --dev           # dev profile (faster to build, slower to run)
#   ./build-apk.sh --skip-build    # package what the last build left in out/lib
#
# Needs: the Android SDK (ANDROID_HOME or ANDROID_SDK_ROOT) with build-tools
# and a platform, the NDK (ANDROID_NDK_HOME, or under $ANDROID_HOME/ndk),
# `cargo ndk` (cargo install cargo-ndk), the aarch64-linux-android target
# (rustup target add aarch64-linux-android) and a JDK for keytool/apksigner.
# The tools may also simply be on PATH, with ANDROID_JAR pointing at an
# android.jar; that is how the script is checked outside the SDK.
#
# Signing: with RDS_KEYSTORE, RDS_KEYSTORE_PASSWORD and RDS_KEY_ALIAS set
# (RDS_KEY_PASSWORD defaults to the keystore password) the APK is signed
# with that key, which is what a release wants: Android only updates an app
# in place when the new package is signed with the same key as the old one.
# Otherwise a debug key is generated once into out/debug.keystore.
#
# Result: out/rust-dicom-station-<version>-arm64-v8a.apk
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
out="$here/out"
abi=arm64-v8a
min_sdk=30
target_sdk=35
profile=release
build=1
for arg in "$@"; do
    case "$arg" in
        --dev) profile=dev ;;
        --skip-build) build=0 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# ---- versions, from the viewer's Cargo.toml -------------------------------
version="$(sed -n 's/^version = "\([^"]*\)"/\1/p' "$root/Cargo.toml" | head -n1)"
[ -n "$version" ] || { echo "could not read the version from $root/Cargo.toml" >&2; exit 1; }
IFS=. read -r vmaj vmin vpat <<<"${version%%-*}"
version_code=$((vmaj * 10000 + vmin * 100 + vpat))
echo "rust-dicom-station $version (versionCode $version_code), $abi, $profile"

# ---- the SDK tools ---------------------------------------------------------
sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
find_tool() {
    # $1 = tool name. Prefer the newest build-tools of the SDK, else PATH.
    if [ -n "$sdk" ] && [ -d "$sdk/build-tools" ]; then
        local newest
        newest="$(ls -d "$sdk"/build-tools/*/ 2>/dev/null | sort -V | tail -n1)"
        if [ -n "$newest" ] && [ -x "$newest/$1" ]; then
            echo "$newest/$1"
            return
        fi
    fi
    command -v "$1" || { echo "$1 not found: install the Android build-tools or put it on PATH" >&2; exit 1; }
}
aapt2="$(find_tool aapt2)"
zipalign="$(find_tool zipalign)"
apksigner="$(find_tool apksigner)"
android_jar="${ANDROID_JAR:-}"
if [ -z "$android_jar" ] && [ -n "$sdk" ] && [ -d "$sdk/platforms" ]; then
    android_jar="$(ls "$sdk"/platforms/android-*/android.jar 2>/dev/null | sort -V | tail -n1)"
fi
[ -f "$android_jar" ] || { echo "android.jar not found: install a platform into the SDK or set ANDROID_JAR" >&2; exit 1; }
echo "aapt2: $aapt2"
echo "android.jar: $android_jar"

# ---- 1. the shared library --------------------------------------------------
mkdir -p "$out"
if [ "$build" = 1 ]; then
    cargo_profile=()
    [ "$profile" = release ] && cargo_profile=(--release)
    (cd "$here" && cargo ndk -t "$abi" -P "$min_sdk" -o "$out/lib" build "${cargo_profile[@]}")
fi
so="$out/lib/$abi/librust_dicom_station.so"
[ -f "$so" ] || { echo "$so is missing" >&2; exit 1; }

# ---- 2. resources and manifest ---------------------------------------------
rm -f "$out/res.zip" "$out/unsigned.apk" "$out/aligned.apk"
"$aapt2" compile --dir "$here/res" -o "$out/res.zip"
"$aapt2" link -o "$out/unsigned.apk" \
    -I "$android_jar" \
    --manifest "$here/AndroidManifest.xml" \
    --version-code "$version_code" \
    --version-name "$version" \
    --min-sdk-version "$min_sdk" \
    --target-sdk-version "$target_sdk" \
    "$out/res.zip"

# ---- 3. the library goes in under lib/<abi>/ --------------------------------
# zip stores the path relative to the working directory, which is why it is
# run from out/: the entry must be exactly lib/arm64-v8a/librust_dicom_station.so.
(cd "$out" && zip -q -u unsigned.apk "lib/$abi/librust_dicom_station.so")

# ---- 4. align and sign -------------------------------------------------------
"$zipalign" -f 4 "$out/unsigned.apk" "$out/aligned.apk"
apk="$out/rust-dicom-station-$version-$abi.apk"
if [ -n "${RDS_KEYSTORE:-}" ]; then
    : "${RDS_KEYSTORE_PASSWORD:?RDS_KEYSTORE_PASSWORD is needed with RDS_KEYSTORE}"
    : "${RDS_KEY_ALIAS:?RDS_KEY_ALIAS is needed with RDS_KEYSTORE}"
    "$apksigner" sign --ks "$RDS_KEYSTORE" --ks-key-alias "$RDS_KEY_ALIAS" \
        --ks-pass "pass:$RDS_KEYSTORE_PASSWORD" \
        --key-pass "pass:${RDS_KEY_PASSWORD:-$RDS_KEYSTORE_PASSWORD}" \
        --out "$apk" "$out/aligned.apk"
    echo "signed with $RDS_KEYSTORE ($RDS_KEY_ALIAS)"
else
    ks="$out/debug.keystore"
    if [ ! -f "$ks" ]; then
        keytool -genkeypair -keystore "$ks" -storepass android -keypass android \
            -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 10000 \
            -dname "CN=Android Debug,O=Android,C=US" >/dev/null 2>&1
    fi
    "$apksigner" sign --ks "$ks" --ks-key-alias androiddebugkey \
        --ks-pass pass:android --key-pass pass:android \
        --out "$apk" "$out/aligned.apk"
    echo "signed with the debug key ($ks); set RDS_KEYSTORE for a release"
fi
"$apksigner" verify "$apk"
rm -f "$out/unsigned.apk" "$out/aligned.apk" "$out/res.zip" "$apk.idsig"
ls -l "$apk"
