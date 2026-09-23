# The iOS package (iPad and iPhone)

Rust DICOM Station runs on iPads and iPhones as the same program it is on
Windows, Linux, macOS and Android: the same viewer, the same modules, the
same windows, compiled for `aarch64-apple-ios` and packaged as one `.ipa`
that installs on both. Nothing was taken out and nothing was redrawn for
touch; an iPad with a Magic Keyboard or an Apple Pencil gets the desktop
program, a finger gets it too, and an iPhone gets it drawn smaller, with the
notes below. The package is built by
[packaging/ios/build-ipa.sh](../packaging/ios/build-ipa.sh) from the crate in
[packaging/ios/](../packaging/ios/), the way the Android APK is built from
[packaging/android/](../packaging/android/), and every release attaches it as
`rust-dicom-station-<version>-ios.ipa`.

The iPad is where the program is at home: the three linked panes, the data
tree and the module panel want the room of a tablet. On an iPhone the same
layout is fitted to the screen (see [On an iPhone](#on-an-iphone)); it is
good for looking at a study, checking contours and a dose, less so for an
hour of contouring.

## Installing

iOS runs nothing that no Apple certificate has signed, and what the release's
`.ipa` is signed with depends on what the repository has been given
([Releasing](#releasing)). There are three ways onto an iPad or iPhone:

| Route | Needs | Lasts |
|---|---|---|
| **TestFlight** | The release was uploaded to App Store Connect (the `APP_STORE_CONNECT_*` secrets); the tester is invited in App Store Connect and installs Apple's *TestFlight* app | 90 days per build; a new release is offered as an update |
| **Ad hoc** | The `.ipa` was signed with an ad hoc profile that lists the device's UDID | Until the profile expires (one year) |
| **Re-signed by the user** | The ad hoc-signed `.ipa` every release carries without secrets, and a sideloading tool: [AltStore](https://altstore.io), [Sideloadly](https://sideloadly.io), or Xcode with the user's own Apple ID | 7 days with a free Apple ID, one year with a paid developer account; the tool refreshes it |

An ad hoc build installs by dragging the `.ipa` onto the device in Finder
(macOS) or Apple Configurator, or with `xcrun devicectl device install app
--device <UDID> rust-dicom-station-<version>-ios.ipa`. iOS 16 and later
also ask for *Developer Mode* (Settings > Privacy & Security) before a
development- or ad hoc-signed app starts the first time.

A newer build with the same bundle identifier installs over the old one and
keeps its settings, studies, archive and models.

iOS / iPadOS 15 or later is required. Memory decides more than the chip: a
4DCT study and the viewer's caches want a device with 4 GB or more.

## Getting studies onto the device

The program's data folder is its `Documents` folder, which the Files app shows
as **On My iPad > Rust DICOM Station** (**On My iPhone** on a phone) and
Finder shows under the device's *Files* tab when it is connected to a Mac. A
DICOM folder copied there - from iCloud Drive, a USB stick, a file server, an
AirDrop - is an ordinary folder to the program, and it is the first place the
file browser offers (*On My iPad* / *On My iPhone*).

A folder can also stay where it is. **+ Folder from Files** in the file
browser opens the system's folder picker, which reaches iCloud Drive, other
apps' folders, USB drives and SMB servers; the folder picked is added to the
browser's places, opened, and kept across launches (a bookmark in the
settings folder). A long press on such a place offers to remove it again. A
folder in iCloud Drive whose files have not been downloaded to the device
yet reads as empty; *Download Now* in the Files app first.

## What is different on an iPad

Everything in this table holds for an iPhone too; what a phone adds is in
the next section.

| | Desktop | iPad |
|---|---|---|
| File and folder dialogs | The operating system's | The same folder browser as on Android (`src/app/pick.rs`): *On My iPad* (the app's `Documents`), then every folder granted with *+ Folder from Files*; folders open on a tap, files are picked with a tap, *Save* takes the name typed |
| Tool windows | Each its own window of the operating system | Windows inside the one window iPadOS gives an app (`app/detach.rs`), with egui's own title bar: they move, resize and close the same way. *Keep on top* is not offered, there being nothing to be on top of |
| Slice scrolling | Mouse wheel over a pane | The on-image slice scrubber; two fingers pinch to zoom |
| Window / level, pan, zoom | Right drag, middle drag, wheel | The W/L fields and the CT presets in the toolbar; the ✋ switch of a view, with which a one-finger drag moves the image; ➖ / ➕ for zoom. The crosshair, the drawing tools and the Pencil work by touch |
| Context menus | Right click | Long press |
| Tooltips | Hover | Long press on the control |
| Mouse, trackpad | Buttons, wheel, hover | A click is a tap and a drag is a finger drag. The window library (`winit` 0.30) reports only touches on iOS, so there is no right or middle button, no wheel and no hover |
| Text fields | Keyboard | The on-screen keyboard, which iPadOS shows when a field is tapped. It is laid over the window, not beside it: a field in the lower part of the screen can be covered while typing, so move its window up first |
| Graphics backend | Vulkan, DirectX 12, OpenGL, with fallback at start | Metal only; `Auto` (the default) picks it. There is no second attempt in the same run, because iOS allows one window loop per process |
| MCP server | `rds-mcp` beside the executable | Not applicable: an app has no standard input and output for a client to speak over. *Settings > MCP server* still shows the snippet, which has nothing to point at |
| Installer, updater | `rds-setup.exe` | TestFlight, or a newer `.ipa` |

The inference engines are built exactly as on the desktop, with the `gpu`
feature on (Metal), and the same rules decide what runs: MedSAM2 fits an
iPad well, the 3 mm TotalSegmentator needs an iPad with 12 GB or more (M-series
iPad Pro), the 1.5 mm models and SegVol do not fit
([auto-segmentation.md](auto-segmentation.md)). iPadOS stops a program in
the background after a short while; keep the viewer in front while a job of
minutes runs. Registration and the 4D pipeline work; they heat the device.

The PACS module works over the local network; the first connection makes
iOS ask whether the program may use it (*Local Network*, Settings >
Privacy & Security), and nothing gets through until that is allowed.

A screen smaller than the desktop window's minimum (900 x 520 points) gets
the whole UI at a smaller zoom, as on a phone (next section): an 11-inch
iPad in portrait is drawn at about 93 %, an iPad mini in portrait at about
83 %, an app in Split View as small as it has to be, down to 60 %. In
landscape every iPad has the room and nothing changes.

## On an iPhone

The same binary, the same layout, and three differences:

* **Landscape only.** In portrait a phone is under 450 points wide, half of
  what the layout needs, and no readable zoom closes that gap; the bundle
  therefore offers the iPhone landscape only (the iPad all four
  orientations).
* **Drawn smaller.** [packaging/ios/src/fit.rs](../packaging/ios/src/fit.rs)
  sets egui's zoom so that the safe part of the screen - between the
  Dynamic Island, the rounded corners and the home indicator - measures the
  desktop window's minimum of 900 x 520 points: about 72 % on a 6.1-inch
  iPhone, about 81 % on a 6.9-inch one, never below 60 %. It is the same
  thing Ctrl and minus do on the desktop, done once for the screen. Text is
  small but sharp; controls are small targets for a finger, so a stylus or
  a steady fingertip helps with the drawing tools. The zoom is re-chosen
  whenever the screen or the insets change.
* **Less memory.** An iPhone has 6 to 12 GB and iOS gives one app a part of
  it: viewing, contours, dose, DVH and registration of one study work;
  MedSAM2 fits a phone with 8 GB or more; the TotalSegmentator models and
  SegVol do not fit.

Everything else - where the files are, the folder picker, the logs, the
panic file - is as on the iPad.

## Where the files are

iOS has no folders outside an app's own container, and gives each app a
home folder of its own (`$HOME`), which `settings::config_dir` and
`settings::data_dir` build on.

| | Desktop (macOS) | iPad / iPhone |
|---|---|---|
| Settings | `~/Library/Application Support/RustDICOMStation` | `Library/Application Support/RustDICOMStation` in the app's container (private; the granted folders' bookmarks are in `places/` there) |
| Models, archive, templates, test data | `~/Library/Application Support/RustDICOMStation` | `Documents` in the app's container, i.e. *On My iPad* (*On My iPhone*) *> Rust DICOM Station* in the Files app. `last_panic.txt` goes here too |

Both are removed with the app. Because `Documents` is visible in the Files
app, model weights already on a computer can be copied into its `models`
folder (Finder, or the Files app from a USB stick) instead of downloaded on
the device; the model manager shows the exact path. `Documents` is marked
excluded from iCloud and computer backups at every start, as `allowBackup` is
off on Android: no study reaches a cloud backup.

## Logs

Log lines and everything the viewer writes to standard error go to the
program's standard error, which Xcode's device console, `xcrun devicectl
device process launch --console` and the simulator show. A panic is also
written to `last_panic.txt` in `Documents`, so a device that has never seen
a Mac still keeps the evidence: open it in the Files app.

## The crate, piece by piece

[packaging/ios/](../packaging/ios/) is a separate workspace like
`packaging/android/`: `cargo build` in the repository root never compiles
it, and the desktop build graph does not change. It holds:

* **`Cargo.toml`.** A binary named `rust-dicom-station` (the bundle's
  `CFBundleExecutable`), depending on the viewer by path with its default
  features, plus `objc2`, `objc2-foundation`, `objc2-ui-kit` and
  `objc2-uniform-type-identifiers` for the folder picker - the versions
  `egui-winit` already uses on iOS. All of it only for `target_os = "ios"`:
  on any other target the crate is a stub `main` and its unit tests, so
  `cargo test` there runs on a Linux or macOS host in seconds.
* **`src/main.rs`.** The entry point: the logger, the two folders, the panic
  file, the backup exclusion, the granted folders restored, the graphics
  choice, and `eframe::run_native` with the same `ViewerApp` as
  `src/main.rs`, wrapped in a `Shell` that fits the layout to the screen
  and keeps the status bar and the home indicator free. `run_native` never
  returns on iOS: `winit` hands the main thread to UIKit.
* **`src/fit.rs`.** The zoom that fits the desktop layout into a smaller
  screen ([On an iPhone](#on-an-iphone)). It works in `eframe`'s raw-input
  hook, on the screen size and insets in UIKit points, before egui sees a
  frame: measured on egui's own rectangles it would shrink itself to the
  minimum, because egui double-counts the insets on the one frame a zoom
  changes. It also rescales the insets into egui's points, which
  `egui-winit` leaves in UIKit's; without that the strips come out too
  narrow at any zoom below 1. No UIKit call; tested on the host with a
  simulated 6.1-inch iPhone and iPad.
* **`src/safe_area.rs`.** The strips iOS lays over the window. `egui-winit`
  reads the window's safe-area insets on iOS and egui's `content_rect` is
  the safe part, but an `eframe` app's root `Ui` spans the whole screen; so
  an empty panel of exactly the covered size goes on each edge before the
  viewer draws, as the Android front end does with the insets it reads over
  JNI. No UIKit call; tested on the host.
* **`src/places.rs`.** The system's folder picker
  (`UIDocumentPickerViewController` for `UTTypeFolder`, opened in place),
  its delegate, the security-scoped access, the bookmarks that bring a
  granted folder back on the next start, and the backup exclusion. It
  registers itself with the viewer as `settings::ios::Places`, which is
  what the file browser asks for places, for the picker, and for the name
  of the app's own folder (*On My iPad* or *On My iPhone*, from
  `UIDevice`'s idiom).
* **`Info.plist.in`.** iPad and iPhone (`UIDeviceFamily` 1 and 2), iOS 15,
  all four orientations on the iPad and landscape on the iPhone, the
  launch screen without a storyboard (`UILaunchScreen`), `Documents` shared
  with the Files app (`UIFileSharingEnabled`,
  `LSSupportsOpeningDocumentsInPlace`), the local-network question for the
  PACS module, `ITSAppUsesNonExemptEncryption` false. Versions and the bundle
  identifier are filled in by the build script, so nothing in the folder is
  bumped on a release. The comment at its top explains each entry.
* **`PrivacyInfo.xcprivacy`.** The privacy manifest App Store Connect
  requires. The executable imports two groups of Apple's "required reason"
  APIs - the file timestamp calls `std::fs` makes (`stat` and relatives)
  and `mach_absolute_time` behind `std::time::Instant` - and declares why;
  nothing is collected or tracked. After a dependency update, `nm -u` on the
  executable is how to see whether a new group appeared.
* **`Assets.xcassets/`.** The app icon, the project's
  `assets/rust-dicom-station.png` cropped to its square, set on black
  (the App Store refuses transparency in an icon) and scaled to the sizes an
  iPad and an iPhone read. The 1024 px App Store icon is scaled up from the 256 px
  source; a larger source would make it sharper.
* **`build-ipa.sh`.** `cargo build` for the device or the simulator, then the
  bundle: the executable and the plist, `actool` for the icon, the `DT*` keys
  Xcode would have written (App Store Connect checks them), a check that
  the executable's platform and deployment target match the plist, `codesign`
  with the profile's entitlements (or ad hoc), and the `.ipa`. No Xcode
  project.
* **`simulator-smoke.sh`.** Creates a fresh simulated iPad (`ipad`, the
  default) or iPhone (`iphone`) on the newest runtime, installs and starts
  the simulator build, and after 45 seconds saves a screenshot and fails if
  the program panicked or is no longer running.

In the viewer's own crate the iOS build differs in the places the Android
build does, all keyed by target so that the desktop and Android builds are
what they were:

* `Cargo.toml`: a table of its own for `target_os = "ios"`: `eframe`
  without `x11`, `wayland` and `accesskit` (which has no iOS adapter), no
  `rfd` (no iOS backend), and `ureq` with the bundled Mozilla roots. The
  desktop table's condition excludes iOS; `Cargo.lock` is unchanged;
* `app/pick.rs`: the Android browser is used on iOS too, with its own roots
  (`ios_roots_from`, tested on every platform), the *+ Folder from Files*
  button, the long-press menu of a granted place, and the iOS wording of
  the hint under a folder that cannot be listed;
* `settings.rs`: one arm each in `config_dir` and `data_dir`, and the
  `settings::ios::Places` registry the front end fills (with the label
  of the app's own folder);
* `gfx.rs`: Metal is the only backend offered on iOS, as on macOS.

## Building it

Needs a Mac with Xcode (the command line tools alone have no iOS SDK) and
the Rust targets:

```text
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
packaging/ios/build-ipa.sh                      # -> packaging/ios/out/rust-dicom-station-<version>-ios.ipa
packaging/ios/build-ipa.sh --simulator          # -> packaging/ios/out/simulator/RustDICOMStation.app (+ a .zip)
packaging/ios/simulator-smoke.sh                # start it on a simulated iPad
packaging/ios/simulator-smoke.sh iphone         # ... and on a simulated iPhone
```

`--dev` builds the dev profile (quicker to build, slower to run),
`--skip-build` repackages what the last build left, `--no-package` stops at
the `.app`. The executable alone is `cargo build --release --target
aarch64-apple-ios` in `packaging/ios/`; `cargo clippy --target
aarch64-apple-ios -- -D warnings` there is the cross-compilation check that
CI runs on pull requests. The root `cargo fmt --all` does not reach this
workspace: run `cargo fmt` in `packaging/ios/` after editing it.

To run a build on your own iPad or iPhone from a Mac, sign it with a
development profile; `xcrun devicectl` installs only signed apps:

```text
export RDS_IOS_SIGN_IDENTITY="Apple Development: Your Name (TEAMID)"
export RDS_IOS_PROFILE=~/Downloads/RDS_Development.mobileprovision
packaging/ios/build-ipa.sh
xcrun devicectl device install app --device <UDID> packaging/ios/out/rust-dicom-station-<version>-ios.ipa
```

## Releasing

[ios.yml](../.github/workflows/ios.yml) has four jobs:

* **check**, on every pull request that touches `src/` or `packaging/ios/`:
  `cargo fmt --check` and `cargo clippy --target aarch64-apple-ios -D
  warnings` of the iOS crate (which compiles the whole viewer for iOS), and
  the crate's host unit tests.
* **ipa** and **simulator**, on a pull request that changes `packaging/ios/`
  or the workflow itself (so the pull request that brings the port in builds
  the app), on *Actions > iOS > Run workflow*, and when the release workflow
  calls it with the version on every push to `main`. `ipa` builds, signs and
  inspects the device package and attaches it to the run as `ios-ipa`;
  `simulator` starts the simulator build on a simulated iPad and a
  simulated iPhone and attaches the screenshots and the logs as
  `ios-simulator`. For a release the simulator job
  cannot hold the release up; for a pull request it is a gate.

The release attaches the `.ipa` to the GitHub Release beside the other
platforms' packages, listed in `SHA256SUMS`.

Signing uses repository secrets (*Settings > Secrets and variables >
Actions*), all optional:

| Secret | Value |
|---|---|
| `IOS_CERTIFICATE_BASE64` | The signing certificate with its private key, exported from Keychain Access as a `.p12`, base64-encoded (`base64 -i cert.p12`). *Apple Distribution* for ad hoc and App Store profiles |
| `IOS_CERTIFICATE_PASSWORD` | The `.p12`'s password |
| `IOS_PROVISIONING_PROFILE_BASE64` | The provisioning profile for the app, base64-encoded: *Ad Hoc* to install on the iPads and iPhones listed in it, *App Store* for TestFlight |
| `APP_STORE_CONNECT_KEY_ID`, `APP_STORE_CONNECT_ISSUER_ID`, `APP_STORE_CONNECT_KEY_BASE64` | An App Store Connect API key (*Users and Access > Integrations*, role *App Manager*), the `.p8` base64-encoded. With an App Store profile, every release is uploaded to TestFlight; *Run workflow* uploads only when its *testflight* box is ticked |

and one optional repository *variable*, `IOS_BUNDLE_ID`, for a wildcard
profile; otherwise the bundle identifier is the one the profile names, or
`io.github.alexprotom.rust-dicom-station` (the macOS bundle's) without a
profile.

Setting it up once, with a paid Apple Developer Program membership:

1. *Certificates, Identifiers & Profiles > Identifiers*: register the App ID
   `io.github.alexprotom.rust-dicom-station` (no capabilities needed).
2. *Certificates*: create an *Apple Distribution* certificate, install it,
   export it with its key from Keychain Access as a `.p12`.
3. *Profiles*: an *Ad Hoc* profile for that App ID and the devices to
   install on (their UDIDs under *Devices*), or an *App Store* profile for
   TestFlight. Download it.
4. For TestFlight: create the app in App Store Connect with the same bundle
   identifier, and an API key.
5. Put the secrets in the repository. The workflow reports in its
   *Import the certificate and the profile* and *Inspect the .ipa* steps
   which identity and which profile it used.

Without the secrets the `.ipa` is signed ad hoc on the runner and the
workflow says so in a notice: a complete package for the re-signing route
above.

The build number (`CFBundleVersion`) is `major*10000+minor*100+patch`
followed by the minutes since 2024, so every upload of the same version has
a higher one, which App Store Connect requires. The version itself
(`CFBundleShortVersionString`) is the crate's.

## Not yet done

* Opening a study from another app (the Files app's *Share*, *Open in*) is
  not wired up: the program reads folders, so copy the study into the
  app's own folder or add its folder with *+ Folder from Files*.
* The on-screen keyboard is laid over the window; the viewer does not move a
  field that it covers out of the way (Android resizes the window instead).
  On an iPhone in landscape it covers half the screen.
* No phone layout: on an iPhone the desktop layout is drawn smaller, not
  rearranged. A layout of its own (one pane at a time, the tree and the
  module panel as drawers, larger targets) would mean phone-specific code in
  `src/app/`, which this port deliberately does not touch.
* `winit` 0.30 runs the app through the application delegate rather than
  UIKit scenes. That is supported by iOS 26, but Apple has announced that
  a later SDK will require scenes, at which point the window library has to
  move on first.
* No mouse buttons, wheel or hover (see the table above): right-drag
  window / level and middle-drag pan have touch equivalents in the toolbar
  and the ✋ switch, but a Magic Keyboard's trackpad adds nothing a finger
  cannot do.
