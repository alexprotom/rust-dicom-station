# Release and Versioning

## Versioning

The application version is defined in the root `Cargo.toml`:

```toml
[package]
name = "rust-dicom-station"
version = "0.1.0"
```

GitHub Actions uses this value to create the GitHub Release tag:

```text
Cargo.toml version 0.1.0 -> Git tag v0.1.0
```

The version is **not incremented automatically**. It must be updated manually before merging a release into `main`.

Each version can only be released once. The [release workflow](../.github/workflows/release.yml) stops if the corresponding tag or GitHub Release already exists.

## Branch Workflow

The repository uses three main branches:

```text
develop -> release -> main
```

* `develop`: active development and feature integration.
* `release`: release candidate testing and preparation.
* `main`: production-ready code and release trigger.

## Creating a Release

### 1. Develop

Develop features on feature branches and merge them into `develop`.

### 2. Prepare the Release

When the code is ready:

1. Merge the release-ready changes into `release`.
2. Update the version in the root `Cargo.toml`:

```toml
version = "0.2.0"
```

3. Test the release candidate on the `release` branch.

### 3. Release

Once the release is approved, merge `release` into `main` and push the changes.

A push to `main` automatically triggers:

```text
.github/workflows/release.yml
```

The workflow:

1. Reads the version from `Cargo.toml`.
2. Checks that the version has not already been released.
3. Builds the Windows installer and the winget manifests for it.
4. Builds the Linux AppImage.
5. Builds the two macOS disk images, signed and notarised when the `MACOS_*` secrets are set (see [macOS](#macos)).
6. Builds the Android APK, signed with the release key when the `ANDROID_KEYSTORE_*` secrets are set (see [Android](#android)).
7. Builds the snap, tests it on the runner and releases it to the Snap Store, when the `SNAPCRAFT_STORE_CREDENTIALS` secret is set (see [Snap Store](#snap-store)).
8. Generates SHA256 checksums.
9. Creates the GitHub Release and uploads the binaries.
10. Submits the new version to winget, when the `WINGET_TOKEN` secret is set (see [winget](#winget)).
11. Writes the Homebrew cask, attaches it to the release and pushes it to the tap, when `HOMEBREW_TAP` and `HOMEBREW_TAP_TOKEN` are set (see [Homebrew](#homebrew)).

## Release Artifacts

Each successful release provides:

```text
rust-dicom-station-X.Y.Z-windows-x86_64.exe
rust-dicom-station-X.Y.Z-linux-x86_64.AppImage
rust-dicom-station-X.Y.Z-macos-arm64.dmg
rust-dicom-station-X.Y.Z-macos-x86_64.dmg
rust-dicom-station-X.Y.Z-arm64-v8a.apk
rust-dicom-station-X.Y.Z-winget-manifests.zip
rust-dicom-station.rb
SHA256SUMS
```

Models are not included in the release artifacts. They are downloaded by the application when required and stored in the configured models directory.

The installed program updates itself from these artifacts: *Start > Update Rust DICOM Station* (`rds-setup.exe --update`) follows GitHub's "latest release" link, downloads the Windows installer by the name above and runs it only if its hash matches the entry in `SHA256SUMS`. Two consequences for releases:

* keep the asset names and `SHA256SUMS` exactly as the workflow writes them - the updater looks them up by name;
* a release marked as a pre-release, or left as a draft, is never offered as an update.

A newer setup run over an older installation updates it in place; nothing has to be uninstalled first ([installer/README.md](../installer/README.md#updating)).

## winget

The package identifier is `RDS.RustDICOMStation`. Once it is in the [winget community repository](https://github.com/microsoft/winget-pkgs), users install and update with:

```text
winget install RDS.RustDICOMStation
winget upgrade RDS.RustDICOMStation
```

### First submission (once, by hand)

1. Let the release workflow publish a version, then download `rust-dicom-station-X.Y.Z-winget-manifests.zip` from the release and unpack it into a folder named `manifests`. It holds the three manifests (version, installer, default locale) with the installer's URL and SHA-256 already filled in.
2. Optionally test them on a Windows machine, from an elevated prompt:

   ```text
   winget settings --enable LocalManifestFiles
   winget validate --manifest .\manifests
   winget install --manifest .\manifests
   ```

3. Submit them, either with [wingetcreate](https://github.com/microsoft/winget-create):

   ```text
   wingetcreate submit --token <personal access token> .\manifests
   ```

   or by a pull request to `microsoft/winget-pkgs` that adds the files under `manifests/r/RDS/RustDICOMStation/X.Y.Z/`.
4. Wait for the pull request to pass validation and be merged. Reviewers may ask questions about the installer; the relevant facts are in [installer/README.md](../installer/README.md#winget).

### Every later release (automatic)

1. Fork `microsoft/winget-pkgs` to the GitHub account that owns the token.
2. Create a classic personal access token with the `public_repo` scope.
3. Save it as the repository secret `WINGET_TOKEN` (*Settings > Secrets and variables > Actions*).

From then on the `winget` job of the release workflow opens the pull request for each new version itself. Add the secret only after the first version has been merged: the job updates an existing package and fails on one that does not exist yet. Without the secret the job is skipped.

## Snap Store

The snap is `rust-dicom-station` ([docs/snap.md](snap.md)). The `snap` job of the release workflow runs [snap.yml](../.github/workflows/snap.yml), which builds it, installs it on the runner, starts both commands, and releases it to the `stable` channel, or to the channel named by the repository variable `SNAP_CHANNEL`. The GitHub Release does not wait for it, and the snap is not one of the release's files: users get it from the store.

```text
sudo snap install rust-dicom-station
```

Setting it up (register the name, the first upload, the `SNAPCRAFT_STORE_CREDENTIALS` secret) is described step by step in [docs/snap.md](snap.md#publishing). Until the secret exists the job builds and tests the snap and uploads nothing. *Actions > Snap > Run workflow* builds any branch the same way, for testing or for the `edge` / `beta` / `candidate` channels.

## macOS

Two disk images, one per architecture, both for **macOS 12 Monterey and
newer**: `rust-dicom-station-X.Y.Z-macos-arm64.dmg` and
`-macos-x86_64.dmg`. They are built by
[macos.yml](../.github/workflows/macos.yml), called by the `macos` job of
the release workflow; the GitHub Release waits for them like for the
Windows and Linux builds.

The images are built **twice**: once on the pull request that merges
`develop` into `main` - the release candidate - and once for real when that
merge lands. The first run publishes nothing; it exists because GitHub only
offers *Run workflow* for a file already on the default branch, so without
it a change to the packaging would first execute during the release itself,
and a mistake there would fail the release instead of a check. Watch the
release-candidate run before merging: what it builds is byte-for-byte what
the release will build.

Both are built on an **Apple Silicon** runner, the Intel half
cross-compiled. GitHub's `macos-15-intel` image is announced to be the last
x86_64 macOS runner and to go away in August 2027, and nothing in this
repository differs between the two architectures, so the Intel image is
inspected rather than run - architecture, deployment target, signature and
bundle contents, read back out of the finished `.dmg`. The floor is
enforced in three places at once (`MACOSX_DEPLOYMENT_TARGET`, the plist's
`LSMinimumSystemVersion`, and the load commands of the linked binary), and
a disagreement fails the build rather than a user's machine
([macos/README.md](../macos/README.md#macos-12-and-newer)).

With the secrets `MACOS_CERTIFICATE_BASE64`, `MACOS_CERTIFICATE_PASSWORD`
and `MACOS_SIGNING_IDENTITY` the bundle and the image are signed with a
Developer ID under the hardened runtime; adding `MACOS_NOTARY_APPLE_ID`,
`MACOS_NOTARY_PASSWORD` and `MACOS_NOTARY_TEAM_ID` sends the image to Apple
and staples the ticket into it, after which it opens by double-clicking.
Without them the images are signed ad-hoc: they run, but a user's first
launch has to be Finder's right-click ▸ *Open*
([docs/macos.md](macos.md#signing-and-notarisation)). *Actions ▸ macOS ▸ Run
workflow* builds both images of any branch the same way and attaches them to
the run.

Unlike Windows and Android, macOS has no in-place updater here: a newer
image is dragged over the old application. Settings, models and the archive
live outside the bundle and survive it.

## Homebrew

The cask is `rust-dicom-station`, one file covering both architectures. The
`homebrew` job writes it with
[macos/brew-cask.sh](../macos/brew-cask.sh) after the release exists, taking
the two checksums out of the release's own `SHA256SUMS` so the cask cannot
disagree with the files it points at, attaches it to the release as
`rust-dicom-station.rb`, and commits it to the tap named by the repository
variable `HOMEBREW_TAP` using the secret `HOMEBREW_TAP_TOKEN`.

```text
brew tap owner/tap
brew install --cask rust-dicom-station
brew upgrade --cask rust-dicom-station
```

Setting the tap up is three steps, once
([docs/macos.md](macos.md#homebrew)). Without the variable or the token the
job says so and succeeds - a release is never held up by it - and the cask
is still attached to the release.

## Android

The APK is built by [android.yml](../.github/workflows/android.yml), called by the `android` job of the release workflow; the GitHub Release waits for it like for the Windows and Linux builds. Its `versionCode` is derived from the version (`major * 10000 + minor * 100 + patch`, so 0.9.4 is 904), which is why versions must only ever go up: Android refuses to install a package whose code is lower than the installed one.

The APK is signed with the release key when the four secrets `ANDROID_KEYSTORE_BASE64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS` and `ANDROID_KEY_PASSWORD` are set, and with a throwaway debug key otherwise. Only a package signed with the same key as the installed one can update it in place, so the key is made once and kept ([docs/android.md](android.md#releasing)). *Actions > Android > Run workflow* builds any branch the same way and attaches the APK to the run.

## Important Rule

Always update `Cargo.toml` to a new version before merging a new release into `main`.

For example:

```text
Current release: 0.1.0
Next release:    0.2.0
```

Do not push to `main` again with an already released version.
