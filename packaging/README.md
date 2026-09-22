# Packaging

Everything that turns the viewer into something a user installs, one
folder per platform. Nothing here is part of `cargo build` in the
repository root: the two crates are workspaces of their own, and the rest
is scripts and recipes that the workflows in `.github/workflows/` run.

```text
packaging/
  windows/
    installer/     the Windows setup program (rds-setup) and rds-pack, a crate of its own
    winget/        where rds-pack --winget writes the manifests; nothing committed
    windowsstore/  prepared for a Microsoft Store submission; nothing built yet
  linux/
    appimage/      AppRun, the desktop entry and build-appimage.sh
    flatpak/       the Flathub manifest, its vendored-crate list and metainfo
    snap/          snapcraft.yaml and the desktop entry, plus build-snap.sh
  macos/           build-app.sh (.app and .dmg for arm64 or x86_64), the plist, the cask script
  android/         the Android front end (a cdylib crate of its own) and build-apk.sh
```

| Platform | Built by | Result | Docs |
|---|---|---|---|
| Windows | `windows-viewer` + `windows-setup` + `windows` jobs of release.yml | `rust-dicom-station-X.Y.Z-windows-x86_64.exe`, winget manifests | [windows/installer/README.md](windows/installer/README.md), [windows/winget/README.md](windows/winget/README.md) |
| Linux AppImage | `linux` job of release.yml, `build-appimage.sh` | `rust-dicom-station-X.Y.Z-linux-x86_64.AppImage` | [linux/appimage/README.md](linux/appimage/README.md) |
| Linux snap | snap.yml (called by release.yml) | Snap Store `rust-dicom-station` | [docs/snap.md](../docs/snap.md) |
| Linux Flatpak | flatpak.yml (test builds); Flathub builds the release | Flathub `io.github.alexprotom.rust-dicom-station` | [docs/flatpak.md](../docs/flatpak.md) |
| macOS | macos.yml (called by release.yml), `build-app.sh` | `rust-dicom-station-X.Y.Z-macos-{arm64,x86_64}.dmg`, Homebrew cask | [macos/README.md](macos/README.md), [docs/macos.md](../docs/macos.md) |
| Android | android.yml (called by release.yml), `build-apk.sh` | `rust-dicom-station-X.Y.Z-android-arm64.apk` | [android/README.md](android/README.md), [docs/android.md](../docs/android.md) |

Every script finds the repository root from its own location, so all of
them run from any working directory. The two crates reach the viewer
through a path dependency in their `Cargo.toml` (`../..` from
`packaging/android`, `../../..` from `packaging/windows/installer`).

The macOS folder has no per-architecture subfolders on purpose: one script
builds either architecture (`--arch arm64` / `--arch x86_64`) from the same
plist and entitlements, and the release workflow runs it twice.

Two things stay outside this folder because their tools insist on it:
`.github/workflows/` (GitHub reads workflows from there and nowhere else)
and, at build time only, `snap/` in the repository root, which
`build-snap.sh` and snap.yml copy from `linux/snap/` because snapcraft
reads `snap/snapcraft.yaml` from the project root and nowhere else.

The release process as a whole - versions, tags, what each job produces,
the store submissions - is in [docs/release-versioning.md](../docs/release-versioning.md).
