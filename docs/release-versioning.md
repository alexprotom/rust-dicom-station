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
3. Builds the Windows installer.
4. Builds the Linux AppImage.
5. Generates SHA256 checksums.
6. Creates the GitHub Release and uploads the binaries.

## Release Artifacts

Each successful release provides:

```text
rust-dicom-station-X.Y.Z-windows-x86_64.exe
rust-dicom-station-X.Y.Z-linux-x86_64.AppImage
SHA256SUMS
```

Models are not included in the release artifacts. They are downloaded by the application when required and stored in the configured models directory.

## Important Rule

Always update `Cargo.toml` to a new version before merging a new release into `main`.

For example:

```text
Current release: 0.1.0
Next release:    0.2.0
```

Do not push to `main` again with an already released version.
