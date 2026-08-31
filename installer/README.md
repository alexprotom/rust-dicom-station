# Windows installer

`rust-dicom-station-setup.exe` — a single-file installer for the viewer,
written in Rust like everything else in this project. No WiX, no NSIS, no
Inno Setup: the wizard is egui/eframe (the viewer's own UI stack) and the
system integration is direct Win32 — shell links through `IShellLink`,
registry through `Reg*`, elevation through `ShellExecuteW`.

**This crate is a separate workspace.** `cargo build --release` in the
repository root builds only the viewer and never touches the installer;
building the installer never touches the viewer's `target/`.

## Building a release installer

```
cargo build --release                      # 1. the viewer, from the repo root
cd installer
cargo build --release                      # 2. rds-setup.exe + rds-pack.exe
cargo run --release --bin rds-pack         # 3. dist/rust-dicom-station-setup.exe
```

Step 3 appends the payload to the setup binary. Useful flags:

| flag | effect |
|---|---|
| `--example-data` | ship `example_data/` too (~137 MB before compression) |
| `--no-docs` | leave `docs/` out |
| `--app <FILE>` | use a different viewer executable |
| `--out <FILE>` | write the installer somewhere else |

Without `--example-data` the result is about 35 MB.

A `cargo build`-ed `rds-setup.exe` has no payload; it then looks for a
`payload/` directory next to itself, which is the convenient way to iterate
on the installer without re-packing.

The installer is not code-signed, so SmartScreen shows the usual "unknown
publisher" warning on first run.

## What the installer does

* **Copies the program** — `rust-dicom-station.exe`, `README.md`,
  `LICENSE.txt`, `docs/`, and `example_data/` when it was packed in — into
  `%LOCALAPPDATA%\Programs\Rust DICOM Station` (per user, the default) or
  `%ProgramFiles%\Rust DICOM Station` (all users, asks for elevation).
* **Dependencies** — checks for the Microsoft Visual C++ runtime that Rust's
  MSVC target links against and installs it from Microsoft when missing.
  Rendering needs Direct3D 12 or Vulkan, which the display driver already
  provides, so there is nothing to install for the GPU.
* **Model weights, optionally** — pre-downloads and converts the
  TotalSegmentator weights (6 mm, 3 mm, the 1.5 mm set, or everything) using
  the viewer's own downloader, so the first auto-segmentation run does not
  have to wait for a 135 MB … 1.3 GB download. Skipped by default. They go
  where the viewer keeps every engine's weights: the `totalsegmentator/`
  sub-folder of the model folder, by default
  `%LOCALAPPDATA%\RustDICOMStation\models` for either scope. A model folder
  chosen elsewhere is recorded as `models_dir` in the installing user's
  `%LOCALAPPDATA%\RustDICOMStation\viewer_settings.txt`. The SegVol and
  MedSAM2 weights are never pre-fetched — their licences allow only a
  download by the user, which the viewer does on first use.
* **Integration** — Start-menu and desktop shortcuts, an
  "Open with Rust DICOM Station" verb on folders (the viewer takes a
  directory), a `.dcm`/`.dicom` entry that is *added* to `OpenWithProgids`
  rather than hijacking whatever owns DICOM files today, and optionally the
  program folder on `PATH`.
* **Uninstall** — an entry in Apps & features plus `uninstall.exe` in the
  program folder.

Everything created is recorded in `install-manifest.txt`, and the uninstaller
removes exactly what is listed there — nothing else in the folder, and the
`PATH` entry only if the installer added it. The model folder is kept unless
you ask for it to go — and then it goes whole, every engine's downloads
included (an empty one is cleaned up either way).

## Command line

The same binary drives everything; `--silent` and `--console` skip the
wizard, which is what you want for deployment.

```
rds-setup --silent --dir "D:\Apps\RDS" --add-to-path --models 3mm
rds-setup --silent --all-users            # from an elevated prompt
uninstall.exe --uninstall --silent --remove-models
rds-setup --help
```

`--models` takes `none | 6mm | 3mm | 1.5mm | all`; the other flags are
`--just-me`, `--models-dir` (the model folder), `--no-start-menu`, `--no-desktop-shortcut`,
`--no-file-association`, `--no-vcredist`, `--no-launch`, and `--from` for the
uninstaller.

Building with `--no-default-features` drops the `prefetch-models` feature,
and with it the dependency on the viewer library; the option then disappears
from the wizard and the viewer downloads weights on first use as usual.

## Source map

| file | contents |
|---|---|
| `src/main.rs` | argument parsing, mode dispatch, elevation re-launch |
| `src/plan.rs` | product constants, install options, default paths |
| `src/payload.rs` | the appended-zip payload format and extraction |
| `src/install.rs` | the install steps and the manifest |
| `src/uninstall.rs` | manifest-driven removal, self-deleting uninstaller |
| `src/deps.rs` | Visual C++ runtime detection and installation |
| `src/models.rs` | optional TotalSegmentator weight pre-fetch |
| `src/ui.rs` | the egui wizard |
| `src/console.rs` | text-mode / silent front end |
| `src/win/` | shell links, registry, known folders, console attach |
| `src/bin/pack.rs` | `rds-pack`, builds the shippable installer |
