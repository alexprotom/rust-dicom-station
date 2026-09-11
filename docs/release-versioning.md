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
5. Generates SHA256 checksums.
6. Creates the GitHub Release and uploads the binaries.
7. Submits the new version to winget, when the `WINGET_TOKEN` secret is set (see [winget](#winget)).

## Release Artifacts

Each successful release provides:

```text
rust-dicom-station-X.Y.Z-windows-x86_64.exe
rust-dicom-station-X.Y.Z-linux-x86_64.AppImage
rust-dicom-station-X.Y.Z-winget-manifests.zip
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

## Important Rule

Always update `Cargo.toml` to a new version before merging a new release into `main`.

For example:

```text
Current release: 0.1.0
Next release:    0.2.0
```

Do not push to `main` again with an already released version.
