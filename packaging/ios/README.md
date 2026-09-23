# iOS / iPadOS front end

The same viewer as an iPad and iPhone app, one binary for both: this crate
is a `main` that hands `rust_dicom_station::app::ViewerApp` to UIKit, plus
the zoom that fits the desktop layout onto a phone, the system's folder
picker, the plist, the icon and the script that turns the executable into
an `.ipa`. Everything about installing it, getting studies onto the device,
what differs on a tablet and a phone, where its files are, building,
signing and releasing is in [docs/ios.md](../../docs/ios.md).

**This crate is a separate workspace**, like `packaging/android/` and
`packaging/windows/installer/`: `cargo build` in the repository root never
compiles it, and building it never touches the viewer's `target/`. The root
`cargo fmt --all` does not reach it either; run `cargo fmt` here.

```text
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
./build-ipa.sh                      # -> out/rust-dicom-station-<version>-ios.ipa
./build-ipa.sh --simulator          # -> out/simulator/RustDICOMStation.app
./simulator-smoke.sh                # start it on a simulated iPad, screenshot in out/smoke/ipad
./simulator-smoke.sh iphone         # ... on a simulated iPhone (landscape, fitted zoom)
```

Both scripts need a Mac with Xcode. `cargo clippy --target aarch64-apple-ios
-- -D warnings` in this folder is the cross-compilation check that CI runs
on pull requests; `cargo test` runs the crate's unit tests on any host.

| File | What it is |
|---|---|
| `src/main.rs` | The entry point and the `Shell` around the viewer |
| `src/fit.rs` | The zoom that fits the desktop layout into a smaller screen (every iPhone, an iPad in portrait or Split View), and the insets rescaled to it (no UIKit; tested on the host) |
| `src/gpu.rs` | The graphics device request, lowered to what an iOS GPU has (tested on the host) |
| `src/safe_area.rs` | Keeps the status bar and home indicator strips free (no UIKit; tested on the host) |
| `build.rs` | The viewer's version for the log and the panic file |
| `src/places.rs` | The folder picker, security-scoped access, bookmarks, backup exclusion |
| `Info.plist.in` | The bundle's plist; the comment at its top explains each entry |
| `PrivacyInfo.xcprivacy` | The privacy manifest App Store Connect requires: which "required reason" APIs the executable calls, and why |
| `Assets.xcassets/` | The app icon at the sizes an iPad reads |
| `build-ipa.sh` | cargo, bundle, actool, codesign, `.ipa` |
| `simulator-smoke.sh` | Start the simulator build on a fresh simulated iPad or iPhone and check it stays up |
