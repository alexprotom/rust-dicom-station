# winget

The package identifier is `RDS.RustDICOMStation`; users install and update
with `winget install RDS.RustDICOMStation` and `winget upgrade
RDS.RustDICOMStation` once a version is in the
[winget community repository](https://github.com/microsoft/winget-pkgs).

Nothing in this folder is committed: the manifests are **generated**, per
version, from the finished installer - they carry its SHA-256 and the URL
it has on the GitHub release - by `rds-pack --winget <DIR>` in
[../installer](../installer/README.md#winget). The release workflow writes
them on every push to `main`, attaches them to the release as
`rust-dicom-station-X.Y.Z-winget-manifests.zip`, and, once the package
exists in winget-pkgs and the secret `WINGET_TOKEN` is set, opens the
pull request for the new version itself (the `winget` job of
[release.yml](../../../.github/workflows/release.yml)).

This folder is where a local run of `rds-pack --winget` and the
first-submission test belong:

```text
cd packaging\windows\installer
cargo run --release --bin rds-pack -- --winget ..\winget\manifests --release-date 2026-09-21

winget settings --enable LocalManifestFiles
winget validate --manifest ..\winget\manifests
winget install --manifest ..\winget\manifests
```

`manifests/` is git-ignored here. The whole procedure, including the one
submission that has to be made by hand, is in
[docs/release-versioning.md](../../../docs/release-versioning.md#winget).
