#!/bin/sh
# Regenerate flatpak/cargo-sources.json from Cargo.lock.
#
# Flathub builds with no network, so every crate the build needs is listed
# as a source with its checksum instead of being fetched by cargo. That
# list is derived from Cargo.lock, which is why the lock file is committed.
#
# Run this whenever Cargo.lock changes (a dependency added, `cargo update`),
# and commit the result together with the lock file. It needs python3 and
# downloads the generator from the Flatpak project; nothing else.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(dirname "$here")
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

generator=https://raw.githubusercontent.com/flatpak/flatpak-builder-tools/master/cargo/flatpak-cargo-generator.py

echo "Fetching the generator"
curl -sSfL "$generator" -o "$work/flatpak-cargo-generator.py"

echo "Preparing python"
python3 -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet aiohttp tomlkit

echo "Reading $root/Cargo.lock"
"$work/venv/bin/python" "$work/flatpak-cargo-generator.py" \
  "$root/Cargo.lock" -o "$here/cargo-sources.json"

crates=$(python3 -c "import json;print(sum(1 for e in json.load(open('$here/cargo-sources.json')) if e['type']=='archive'))")
echo "Wrote flatpak/cargo-sources.json: $crates crates"
