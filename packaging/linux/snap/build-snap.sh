#!/bin/sh
# Build the snap locally. snapcraft reads its recipe from snap/snapcraft.yaml
# in the project root and nowhere else, and the project root has to be the
# repository (the recipe's `source: .` is the viewer's sources), so this
# folder is copied to <repo>/snap first - /snap is git-ignored - and
# `snapcraft pack` runs from the repository root with whatever arguments are
# given here (`--destructive-mode`, `--verbose`, ...).
#
#   packaging/linux/snap/build-snap.sh
#   sudo snap install --dangerous ./rust-dicom-station_*.snap
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../../.." && pwd)

rm -rf "$root/snap"
mkdir -p "$root/snap"
cp "$here/snapcraft.yaml" "$root/snap/snapcraft.yaml"
cp -r "$here/local" "$root/snap/local"

cd "$root"
exec snapcraft pack "$@"
