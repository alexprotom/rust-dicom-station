# Android front end

The same viewer as an Android package for tablets: this crate is
`android_main` over `rust_dicom_station::app::ViewerApp`, plus the
manifest, the launcher icons and the script that turns the shared library
into an APK. Everything about installing it, what differs on a tablet,
where its files are, building, signing and releasing is in
[docs/android.md](../docs/android.md).

**This crate is a separate workspace**, like `installer/`: `cargo build` in
the repository root never compiles it, and building it never touches the
viewer's `target/`.

```text
rustup target add aarch64-linux-android
cargo install cargo-ndk
export ANDROID_HOME=~/Android/Sdk
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/<version>
./build-apk.sh          # -> out/rust-dicom-station-<version>-arm64-v8a.apk
```

`cargo ndk -t arm64-v8a -P 30 check` in this folder is the
cross-compilation check that CI runs on pull requests.
