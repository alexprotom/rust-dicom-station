#!/usr/bin/env bash
# The Android release key: made once, kept forever, used by every build.
#
# Android installs a new APK over an installed one only when both are signed
# with the same key; a package signed with another key is refused, and the
# only way past that is to uninstall, which deletes the app's settings and
# data. So there is one key, outside the repository, and both build-apk.sh
# on this machine and the Android workflow in CI sign with it.
#
#   ./release-key.sh new       # make the key and the settings file (once)
#   ./release-key.sh secrets   # put it into the repository's four secrets (gh)
#   ./release-key.sh show      # the key's certificate fingerprint
#
# `new` writes, under ~/.config/rust-dicom-station/ (RDS_KEY_DIR overrides):
#
#   android-release.keystore   the key (PKCS12, RSA 4096, alias `rds`)
#   android-signing.env        RDS_KEYSTORE / RDS_KEYSTORE_PASSWORD /
#                              RDS_KEY_ALIAS, which build-apk.sh reads
#
# and refuses to overwrite an existing keystore: a new key means one more
# uninstall on every device. Back both files up (a password manager holds
# them well); losing them costs the same.
#
# `secrets` sets ANDROID_KEYSTORE_BASE64, ANDROID_KEYSTORE_PASSWORD,
# ANDROID_KEY_ALIAS and ANDROID_KEY_PASSWORD on the repository the current
# folder belongs to, through the GitHub CLI (`gh auth login` first). Nothing
# is printed; the values go to gh on its standard input.
#
# Needs keytool (any JDK) or, without one, openssl (Git for Windows ships
# it; the PKCS12 file it writes is one apksigner and keytool read), and,
# for `secrets`, gh.
set -euo pipefail

dir="${RDS_KEY_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/rust-dicom-station}"
ks="$dir/android-release.keystore"
env_file="$dir/android-signing.env"
alias_name=rds

load() {
    [ -f "$env_file" ] || { echo "$env_file does not exist: run '$0 new' first" >&2; exit 1; }
    # shellcheck source=/dev/null
    . "$env_file"
    [ -f "$RDS_KEYSTORE" ] || { echo "the keystore $RDS_KEYSTORE named in $env_file does not exist" >&2; exit 1; }
}

fingerprint() {
    if command -v keytool >/dev/null; then
        keytool -list -v -keystore "$RDS_KEYSTORE" -storepass "$RDS_KEYSTORE_PASSWORD" \
            -alias "$RDS_KEY_ALIAS" | grep -E 'SHA256:' | head -n1 | sed 's/^[[:space:]]*//'
    else
        openssl pkcs12 -in "$RDS_KEYSTORE" -nokeys -clcerts -passin "pass:$RDS_KEYSTORE_PASSWORD" \
            | openssl x509 -noout -fingerprint -sha256 | sed 's/^[^=]*=/SHA256: /'
    fi
}

# The same key as keytool's, made by openssl: a self-signed RSA 4096
# certificate for 10000 days, put into a PKCS12 file whose friendly name is
# the alias. The private key exists unencrypted only in a folder of mode 700
# for the two commands, and is removed when they end.
openssl_keystore() {
    local tmp
    tmp="$(umask 077 && mktemp -d "$dir/.newkey.XXXXXX")"
    # shellcheck disable=SC2064
    trap "rm -rf '$tmp'" EXIT
    # MSYS2_ARG_CONV_EXCL: in Git Bash on Windows, an argument starting with
    # `/` would otherwise be rewritten into a Windows path before openssl
    # sees it; elsewhere the variable means nothing.
    MSYS2_ARG_CONV_EXCL='/CN=' openssl req -x509 -newkey rsa:4096 -sha256 -days 10000 -nodes \
        -subj "/CN=Rust DICOM Station" -keyout "$tmp/key.pem" -out "$tmp/cert.pem" 2>/dev/null
    openssl pkcs12 -export -inkey "$tmp/key.pem" -in "$tmp/cert.pem" -name "$1" \
        -out "$2" -passout "pass:$3"
    rm -rf "$tmp"
    trap - EXIT
}

case "${1:-}" in
    new)
        if [ -e "$ks" ]; then
            echo "$ks already exists; a release key is made once. '$0 show' prints it." >&2
            exit 1
        fi
        command -v keytool >/dev/null || command -v openssl >/dev/null \
            || { echo "neither keytool (a JDK) nor openssl found" >&2; exit 1; }
        mkdir -p "$dir"
        # 32 letters and digits: keytool takes them on the command line
        # without quoting trouble, on any shell. `|| true`: head closing the
        # pipe ends tr with SIGPIPE, which pipefail would report.
        password="$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 32 || true)"
        [ "${#password}" = 32 ] || { echo "could not read /dev/urandom" >&2; exit 1; }
        # PKCS12 has one password for the store and the key.
        if command -v keytool >/dev/null; then
            keytool -genkeypair -storetype PKCS12 -keystore "$ks" \
                -storepass "$password" -keypass "$password" \
                -alias "$alias_name" -keyalg RSA -keysize 4096 -validity 10000 \
                -dname "CN=Rust DICOM Station" >/dev/null
        else
            openssl_keystore "$alias_name" "$ks" "$password"
        fi
        (
            umask 077
            {
                echo "# The Android release key of rust-dicom-station (packaging/android/release-key.sh)."
                echo "RDS_KEYSTORE='$ks'"
                echo "RDS_KEYSTORE_PASSWORD='$password'"
                echo "RDS_KEY_ALIAS='$alias_name'"
            } >"$env_file"
        )
        chmod 600 "$ks" "$env_file" 2>/dev/null || true
        load
        echo "made $ks"
        echo "     $env_file"
        echo "certificate $(fingerprint)"
        echo
        echo "Back both files up. Next: '$0 secrets' to sign CI's builds with the same key."
        ;;
    secrets)
        load
        command -v gh >/dev/null || { echo "gh not found: https://cli.github.com" >&2; exit 1; }
        base64 <"$RDS_KEYSTORE" | tr -d '\n' | gh secret set ANDROID_KEYSTORE_BASE64
        printf '%s' "$RDS_KEYSTORE_PASSWORD" | gh secret set ANDROID_KEYSTORE_PASSWORD
        printf '%s' "$RDS_KEY_ALIAS" | gh secret set ANDROID_KEY_ALIAS
        printf '%s' "${RDS_KEY_PASSWORD:-$RDS_KEYSTORE_PASSWORD}" | gh secret set ANDROID_KEY_PASSWORD
        echo "set the four ANDROID_* secrets; certificate $(fingerprint)"
        ;;
    show)
        load
        echo "$RDS_KEYSTORE ($RDS_KEY_ALIAS)"
        echo "certificate $(fingerprint)"
        ;;
    -h|--help)
        sed -n '2,31p' "$0"
        ;;
    *)
        echo "usage: $0 new | secrets | show" >&2
        exit 2
        ;;
esac
