# The Android package

Rust DICOM Station runs on Android tablets as the same program it is on
Windows and Linux: the same viewer, the same modules, the same windows,
compiled for `arm64-v8a` and packaged as an APK. Nothing was taken out and
nothing was redrawn for touch; a tablet with a mouse, a keyboard or a stylus
gets the desktop program, and a finger gets it too, with the notes below.
The package is built by [packaging/android/build-apk.sh](../packaging/android/build-apk.sh)
from the crate in [packaging/android/](../packaging/android/), the way the Windows installer
is built from [packaging/windows/installer/](../packaging/windows/installer/README.md), and every release
attaches it as `rust-dicom-station-<version>-android-arm64.apk`.

Phones are not a target. The three linked panes, the data tree and the
module panel want the room of a tablet; the package installs on a phone but
is not laid out for one.

## Installing

Download `rust-dicom-station-<version>-android-arm64.apk` from the
[releases page](https://github.com/alexprotom/rust-dicom-station/releases)
on the tablet, open it, and allow the installation from this source when
Android asks. Android 11 or later on a 64-bit ARM device is required, which
is every tablet sold in the last few years. With a computer and `adb`:

```text
adb install rust-dicom-station-<version>-android-arm64.apk
```

### Updating

A newer APK, a release or a [nightly](release-versioning.md#nightly-builds),
installs over the installed one: open it, and Android asks *Do you want to
update this app?* rather than to install it. The settings, the archive and
the downloaded models stay where they were. With `adb`, `adb install -r`.

That works because every APK is signed with the same release key and
carries a higher `versionCode` than every one built before it
([Signing](#signing)). An APK signed with another key is refused: Android
then does nothing at all when the file is opened, or says *App not
installed* (`adb` says `INSTALL_FAILED_UPDATE_INCOMPATIBLE`), and the only
way past it is to uninstall first, which deletes the app's settings and
data.

**Once, when moving to the release key.** Every APK built before the
release key existed - every release and nightly up to 0.11.8, and every
local build - was signed with a throwaway key, so the first release-signed
APK cannot update any of them. Uninstall the old app that one time,
install the new APK, and from then on each new one installs over the
last. To keep the models (gigabytes, otherwise downloaded again) and the
archive through that uninstall, copy the data folder off the tablet first
and back afterwards ([Where the files are](#where-the-files-are)):

```text
adb pull /storage/emulated/0/Android/data/io.github.alexprotom.rustdicomstation/files rds-data
adb uninstall io.github.alexprotom.rustdicomstation
adb install rust-dicom-station-<version>-android-arm64.apk
# start the app once, so that its folder exists again, then
adb push rds-data/. /storage/emulated/0/Android/data/io.github.alexprotom.rustdicomstation/files/
```

A file manager or a USB connection to a computer does the same without
`adb`. The settings (`viewer_settings.txt`, in the private folder) cannot be
copied this way; set them again once.

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
| PACS | the station's own PACS; `rds-pacs` serves it to others | A client: *Tools > PACS > Add server* pairs with a PACS server and pulls its studies into a mirror in the app's data folder, to work on them offline and sync back, or hands the server tasks ([pacs-server.md](pacs-server.md#phones-and-tablets)). The server itself does not run here, and *PACS > Settings > PACS server* is not offered |
| Installer, updater | `rds-setup.exe` | Android's own installer; a newer APK from the releases page |

The inference engines are built exactly as on the desktop, with the `gpu`
feature on. They compute through Vulkan (OpenGL ES only when it is the
*Graphics backend* chosen, and then it fails: the networks' reductions need
float atomics OpenGL does not have), and when the tablet's GPU cannot run
them, forcing *GPU* says what the adapter answered (see
[auto-segmentation.md](auto-segmentation.md#troubleshooting)). The same
rules decide what runs: MedSAM2 fits a tablet
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

Android ends a program that takes too much memory without a message and
without a crash report of its own, so the program keeps its own record.
Standard output and error, which Android throws away, are copied into the
system log with the tag `rds` and into `rds-log.txt` in the data folder
above: each line with the seconds since start, written as it comes, so
nothing already written is lost when the process is killed. Every job adds
a line for each new step of its progress (*Resampling*, *Segmenting
(total, 3 mm): tile 3/8*; one that only counts on is repeated every few
seconds), each with the memory beside it:

```text
   212.4 progress: Segmenting (total, 3 mm): tile 3/8 (memory: this program 2630 MB (peak 2840 MB), free 1210 of 7480 MB)
```

At start the previous run's log becomes `rds-log.previous.txt`, so after
the program has vanished, starting it again and reading that file (over
USB: *Android/data/io.github.alexprotom.rustdicomstation/files*, or in the
program's own file browser under *App data*) shows the step it was on and
how much memory it had. A panic is written to `last_panic.txt` in both
folders above; the data folder's copy is the one that can be read without
a cable. With a debug cable:

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
* **`src/permission.rs`, `src/insets.rs`, `src/clipboard.rs`.** The calls
  into the Java side, over JNI: `Environment.isExternalStorageManager()`,
  the intent that opens the settings page, the window's system-bar and
  cut-out insets (`getRootWindowInsets().getInsets(..)`, read again on a
  resize and once a second), and the system clipboard (`ClipboardManager`:
  the window library reaches none on Android, so the viewer's *📋 Paste*
  buttons read it through `settings::clipboard`, and `Shell::ui` hands what
  the viewer's copy buttons copied to it). The package name is a constant
  in `permission.rs` and must match the manifest.
* **`AndroidManifest.xml`.** `minSdkVersion` 30 (Android 11, the first
  with all files access), `targetSdkVersion` 35, the `INTERNET` and
  `MANAGE_EXTERNAL_STORAGE` permissions, OpenGL ES 3 required and Vulkan
  preferred, `allowBackup="false"`. Versions are filled in by the build
  script: `versionName` is the crate version from the viewer's
  `Cargo.toml` and `versionCode` the build time ([Signing](#signing)), so
  nothing in the folder is bumped on a release. The package name,
  `io.github.alexprotom.rustdicomstation`, never changes: to Android
  another name is another app, installed beside this one.
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
packaging/android/build-apk.sh                          # -> packaging/android/out/rust-dicom-station-<version>-android-arm64.apk
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

## Signing

Android installs an APK over the installed app only when two things hold:

* **the same signing key.** A package signed with another key is refused
  as an update, whatever its version. So there is exactly one key, the
  release key, and every APK that is meant to be installed is signed with
  it: the releases, the nightlies and the builds made on a computer.
* **a `versionCode` not lower than the installed one.** `build-apk.sh`
  sets it to the build time in minutes since 2024-01-01 UTC (about 1.5
  million in 2026; iOS's `CFBundleVersion` ends in the same number), so a
  later build always installs over an earlier one, whichever branch it
  came from: a nightly over the last release, the next release over the
  nightly, a local build over both. `RDS_VERSION_CODE` overrides it.
  The `versionName` Android shows is the crate version, a nightly's
  suffix included.

Make the key once, with
[packaging/android/release-key.sh](../packaging/android/release-key.sh). It
needs `keytool` from a JDK or, without one, `openssl`, which Git for Windows
ships in its Git Bash (macOS and Linux have it too), and `gh` for the
secrets; no Android SDK:

```text
bash packaging/android/release-key.sh new        # the key and its settings file
bash packaging/android/release-key.sh secrets    # the same key into CI's four secrets (gh)
bash packaging/android/release-key.sh show       # its certificate's SHA-256
```

`new` writes two files to `~/.config/rust-dicom-station/`, outside the
repository - nothing about the key is ever committed:
`android-release.keystore` (PKCS12, RSA 4096, alias `rds`, a random
password) and `android-signing.env`, the three lines `build-apk.sh` reads
when `RDS_KEYSTORE` is not set:

```text
RDS_KEYSTORE='/home/me/.config/rust-dicom-station/android-release.keystore'
RDS_KEYSTORE_PASSWORD='...'
RDS_KEY_ALIAS='rds'
```

It refuses to overwrite a keystore that exists. **Back both files up**; a
password manager holds them well. A lost key cannot be recovered, and
replacing it costs one more uninstall on every device. Another computer
that builds APKs gets a copy of the two files (or points `RDS_SIGNING_ENV`
at its copy of the second, or sets `RDS_KEYSTORE`, `RDS_KEYSTORE_PASSWORD`,
`RDS_KEY_ALIAS` and, when it differs, `RDS_KEY_PASSWORD` itself).

The key is made on a computer rather than by a workflow on purpose. A
secret can be written but never read back, so a key made on a runner and
stored only in the secrets would have no backup and could sign nothing
outside CI; getting it off the runner would mean an artifact, which anyone
can download from a public repository; and the workflow's own token may
not write secrets at all, so it would also need a personal access token
with that right.

Without a release key the script still builds, signed with a debug key it
makes once into `packaging/android/out/debug.keystore`, and says so in a
warning: such an APK installs on a tablet without the app, but neither
updates nor is updated by a release-signed one. `--release-key` makes that
an error instead; CI passes it for every release and nightly.

## Releasing

[android.yml](../.github/workflows/android.yml) runs on pull requests
that touch `src/` or `packaging/android/`, with `cargo ndk check` of the Android
crate; *Actions > Android > Run workflow* builds the APK of any branch and
attaches it to the run (`android-apk`); and the release workflow calls it
with the version on every push to `main`, then attaches the APK to the
GitHub Release beside the Windows installer and the AppImage, listed in
`SHA256SUMS`.

The nightly workflow calls it the same way on every push to `nightly`,
for the rolling pre-release.

CI signs with the [release key](#signing), from four repository secrets
(*Settings > Secrets and variables > Actions*), which
`release-key.sh secrets` sets:

| Secret | Value |
|---|---|
| `ANDROID_KEYSTORE_BASE64` | The keystore file, base64-encoded |
| `ANDROID_KEYSTORE_PASSWORD` | Its password |
| `ANDROID_KEY_ALIAS` | The key's alias in it (`rds`) |
| `ANDROID_KEY_PASSWORD` | The key's password (the keystore's, for PKCS12) |

A release or a nightly **fails** without them, in the *Release keystore*
step: an APK signed with a debug key made on the runner differs from run to
run and installs over nothing, which is why every earlier build needed an
uninstall. A manual *Run workflow* still builds without them, with a
warning. *Inspect the APK* prints the certificate the APK was signed with
(`apksigner verify --print-certs`); its SHA-256 is the one
`release-key.sh show` prints.
