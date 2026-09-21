# The Android package

Rust DICOM Station runs on Android tablets as the same program it is on
Windows and Linux: the same viewer, the same modules, the same windows,
compiled for `arm64-v8a` and packaged as an APK. Nothing was taken out and
nothing was redrawn for touch; a tablet with a mouse, a keyboard or a stylus
gets the desktop program, and a finger gets it too, with the notes below.
The package is built by [packaging/android/build-apk.sh](../packaging/android/build-apk.sh)
from the crate in [packaging/android/](../packaging/android/), the way the Windows installer
is built from [packaging/windows/installer/](../packaging/windows/installer/README.md), and every release
attaches it as `rust-dicom-station-<version>-arm64-v8a.apk`.

Phones are not a target. The three linked panes, the data tree and the
module panel want the room of a tablet; the package installs on a phone but
is not laid out for one.

## Installing

Download `rust-dicom-station-<version>-arm64-v8a.apk` from the
[releases page](https://github.com/alexprotom/rust-dicom-station/releases)
on the tablet, open it, and allow the installation from this source when
Android asks. Android 11 or later on a 64-bit ARM device is required, which
is every tablet sold in the last few years. With a computer and `adb`:

```text
adb install rust-dicom-station-<version>-arm64-v8a.apk
```

A newer APK signed with the same key installs over the old one and keeps
its settings, archive and models. One signed with a different key (a
release over a test build, say) does not: uninstall the old one first.

### All files access

On the first start the program asks for *All files access*. Android grants
it on a system settings page, not in a dialog, and the button on the
program's own page opens that page directly: *Settings > Apps > Rust DICOM
Station > All files access*. With it the shared storage
(`/storage/emulated/0`, what a file manager calls *Internal storage*), the
`Download` folder, and any SD card or USB stick the tablet mounts under
`/storage` are ordinary folders to the program, exactly as on a desktop.

Without it the program still runs, and can open its own folder and the
archive it imports into, but a DICOM folder on the shared storage cannot
be listed; the file browser says so where it fails. *Later* dismisses the
question for the run.

The package is not in Google Play, which does not allow this permission to
a viewer. It is a deliberate trade: the alternative, Android's document
picker, returns handles rather than paths and would need a Java layer
between the picker and the loader, and this project is pure Rust.

## What is different on a tablet

| | Desktop | Android |
|---|---|---|
| File and folder dialogs | The operating system's | A folder browser drawn by the program (`src/app/pick.rs`): roots for the internal storage, *Downloads*, each mounted volume and the app's data folder; folders open on a tap, files are picked with a tap, *Save* takes the name typed |
| Tool windows | Each its own window of the operating system | Windows inside the one window Android allows (`app/detach.rs`), with egui's own title bar: they move, resize and close the same way. *Keep on top* is not offered, there being nothing to be on top of |
| Slice scrolling | Mouse wheel over a pane | The on-image slice scrubber, or a stylus or mouse wheel; two fingers pinch to zoom |
| Window / level, pan | Right drag, middle drag | Need a mouse or a stylus button; the presets and the crosshair work by touch |
| Context menus | Right click | Long press |
| Tooltips | Hover | Long press on the control |
| Text fields | Keyboard | The soft keyboard, which Android shows when a field is tapped |
| Graphics backend | Vulkan, DirectX 12, OpenGL, with fallback at start | Vulkan or OpenGL ES; `Auto` (the default) lets the graphics library pick. The setting applies at the next start; there is no second attempt in the same run, because Android allows one window loop per process |
| MCP server | `rds-mcp` beside the executable | Not applicable: a package has no standard input and output for a client to speak over. *Settings > MCP server* still shows the snippet, which has nothing to point at |
| Installer, updater | `rds-setup.exe` | Android's own installer; a newer APK from the releases page |

The inference engines are built exactly as on the desktop, with the `gpu`
feature on, and the same rules decide what runs: MedSAM2 fits a tablet
well, the 3 mm TotalSegmentator needs a device with 12 GB or more, the
1.5 mm models and SegVol do not fit ([auto-segmentation.md](auto-segmentation.md)).
A job of minutes is at the mercy of Android, which may stop a program in
the background; keep the viewer in front while one runs. Registration and
the 4D pipeline work; they heat the device.

## Where the files are

Android gives every app two folders, and the program uses both. The
activity hands them over at start (`settings::android::set_dirs`,
`packaging/android/src/lib.rs`), since Android has no home folder and no environment
variable for either.

| | Desktop (Linux) | Android |
|---|---|---|
| Settings | `~/.config/RustDICOMStation` | `/data/user/0/io.github.alexprotom.rustdicomstation/files` (file-encrypted, private, `last_panic.txt` goes here too) |
| Models, archive, templates | `~/.local/share/RustDICOMStation` | `/storage/emulated/0/Android/data/io.github.alexprotom.rustdicomstation/files` (the *App data* root of the file browser) |

Both are removed with the app. The second is on the shared storage, so
model weights already on a computer can be copied into its `models` folder
over USB instead of downloaded on the tablet; the model manager shows the
exact path. `allowBackup` is off in the manifest: no study reaches a cloud
backup.

## Logs

Standard output and error, which Android throws away, are copied into the
system log with the tag `rds`, and a panic is also written to
`last_panic.txt` in the settings folder above. With a debug cable:

```text
adb logcat -s rds
```

## The crate, piece by piece

[packaging/android/](../packaging/android/) is a separate workspace like `packaging/windows/installer/`:
`cargo build` in the repository root never compiles it, and the desktop
build graph does not change. It holds:

* **`Cargo.toml`.** A `cdylib` named `rust_dicom_station`, depending on the
  viewer by path with its default features. `android.app.lib_name` in the
  manifest is that name: Android's own `NativeActivity` loads
  `librust_dicom_station.so` and calls `android_main`. There is no Java or
  Kotlin (`hasCode="false"`).
* **`src/lib.rs`.** The entry point: logging to logcat, the two folders,
  the panic file, the graphics choice, and `eframe::run_native` with the
  same `ViewerApp` as `src/main.rs`, wrapped in a `Shell` that keeps the
  strips under the status bar and the navigation bar free (a program
  targeting Android 15 is laid out edge to edge, so the window library
  reports the whole screen) and draws the *All files access* question
  over the viewer.
* **`src/permission.rs`, `src/insets.rs`.** The calls into the Java side,
  over JNI: `Environment.isExternalStorageManager()`, the intent that
  opens the settings page, and the window's system-bar and cut-out insets
  (`getRootWindowInsets().getInsets(..)`, read again on a resize and once
  a second). The package name is a constant in `permission.rs` and must
  match the manifest.
* **`AndroidManifest.xml`.** `minSdkVersion` 30 (Android 11, the first
  with all files access), `targetSdkVersion` 35, the `INTERNET` and
  `MANAGE_EXTERNAL_STORAGE` permissions, OpenGL ES 3 required and Vulkan
  preferred, `allowBackup="false"`. Versions are filled in by the build
  script from the viewer's `Cargo.toml`: `versionName` is the crate
  version and `versionCode` is `major * 10000 + minor * 100 + patch`, so
  nothing in the folder is bumped on a release.
* **`res/mipmap-*/ic_launcher.png`.** The launcher icon, the project's
  `assets/rust-dicom-station.png` at the five densities Android reads.
* **`.cargo/config.toml`.** One linker flag: 16 KB page alignment of the
  library, which Android 15 devices with 16 KB memory pages require and
  which the NDK's linker only applies by itself from r28 on. The workflow
  checks the alignment of every build.
* **`build-apk.sh`.** `cargo ndk` for the library, then the SDK's own
  tools for the package: `aapt2` compiles the icons and links the
  manifest, the library is added under `lib/arm64-v8a/`, `zipalign` aligns
  the archive and `apksigner` signs it. No Gradle.

In the viewer's own crate the Android build differs in three places, all
keyed by target in `Cargo.toml` so that the desktop build is byte for byte
what it was:

* `eframe` takes the `android-native-activity` glue instead of `x11`,
  `wayland` and `accesskit` (the last is offered on Android only with the
  Java-backed GameActivity, which this package does not carry);
* `rfd`, the desktop file dialog, has no Android backend and is left out;
  every dialog goes through `app/pick.rs`, which runs the system dialog on
  the desktop and the egui browser on Android, and lets the calling code
  say what to do with the answer in one place for both;
* `ureq` uses the bundled Mozilla roots instead of the operating system's
  trust store, which Android does not expose to `rustls-native-certs`.

Plus one arm each in `settings::config_dir` and `settings::data_dir`.

## Building it

Needs the Android SDK with build-tools and a platform, the NDK, a JDK,
the Rust target and `cargo-ndk`:

```text
rustup target add aarch64-linux-android
cargo install cargo-ndk
export ANDROID_HOME=~/Android/Sdk            # or wherever Android Studio put it
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/<version>
packaging/android/build-apk.sh                          # -> packaging/android/out/rust-dicom-station-<version>-arm64-v8a.apk
```

`--dev` builds the dev profile (quicker to build, slower to run),
`--skip-build` repackages the library the last run left in `packaging/android/out/lib`.
The library alone, without a package, is `cargo ndk -t arm64-v8a -P 30
build --release` in `packaging/android/`; `cargo ndk -t arm64-v8a -P 30 check` there
is the cross-compilation check that CI runs on pull requests.

Without the SDK installed, the script also accepts `aapt2`, `zipalign` and
`apksigner` on `PATH` (Debian and Ubuntu package all three) and
`ANDROID_JAR` pointing at an `android.jar` of API 24 or later (the manifest
lists `density` among the configuration changes the activity handles
itself, which older platforms do not know).

The APK is signed with a key made on the spot unless `RDS_KEYSTORE`,
`RDS_KEYSTORE_PASSWORD` and `RDS_KEY_ALIAS` (and `RDS_KEY_PASSWORD`, when
it differs) name a keystore. A test build signed with the throwaway key
installs and runs like any other; it just does not update an installation
signed with the release key, and the release key does not update it.

## Releasing

[android.yml](../.github/workflows/android.yml) runs on pull requests
that touch `src/` or `packaging/android/`, with `cargo ndk check` of the Android
crate; *Actions > Android > Run workflow* builds the APK of any branch and
attaches it to the run (`android-apk`); and the release workflow calls it
with the version on every push to `main`, then attaches the APK to the
GitHub Release beside the Windows installer and the AppImage, listed in
`SHA256SUMS`.

The release key lives in four repository secrets (*Settings > Secrets and
variables > Actions*):

| Secret | Value |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | The keystore file, base64-encoded (`base64 -w0 release.keystore`) |
| `ANDROID_KEYSTORE_PASSWORD` | Its password |
| `ANDROID_KEY_ALIAS` | The key's alias in it |
| `ANDROID_KEY_PASSWORD` | The key's password (the same as the keystore's, usually) |

Make the keystore once and keep it: an APK can only ever be updated in
place by a package signed with the same key.

```text
keytool -genkeypair -keystore release.keystore -alias rds -keyalg RSA -keysize 4096 -validity 10000
```

Until the secrets exist the workflow signs with a debug key made on the
runner, which differs from run to run: each such release has to be
uninstalled before the next is installed. The workflow says which key it
used in the *Release keystore* step.
