# Windows installer

`rust-dicom-station-setup.exe` - a single-file installer for the viewer,
written in Rust like everything else in this project. No WiX, no NSIS, no
Inno Setup: the wizard is egui/eframe (the viewer's own UI stack) and the
system integration is direct Win32 - shell links through `IShellLink`,
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
| `--mcp <FILE>` / `--no-mcp` | the MCP server `rds-mcp.exe` rides along when `target/release/rds-mcp.exe` exists (build it with `cargo build --release --features mcp`); these override that |
| `--out <FILE>` | write the installer somewhere else |
| `--winget <DIR>` | also write the winget manifests for this installer into `DIR` (see [winget](#winget)) |
| `--url <URL>` | the download URL recorded in them (default: this version's GitHub release asset) |
| `--release-date <YYYY-MM-DD>` | the release date recorded in them |

Without `--example-data` the result is about 35 MB.

A `cargo build`-ed `rds-setup.exe` has no payload; it then looks for a
`payload/` directory next to itself, which is the convenient way to iterate
on the installer without re-packing.

The installer is not code-signed, so SmartScreen shows the usual "unknown
publisher" warning on first run.

## What the installer does

* **Copies the program** - `rust-dicom-station.exe`, `README.md`,
  `LICENSE.txt`, `docs/`, `rds-mcp.exe` (the MCP server, see
  [docs/mcp.md](../docs/mcp.md)) when it was built, and `example_data/` when
  it was packed in - into
  `%LOCALAPPDATA%\Programs\Rust DICOM Station` (per user, the default) or
  `%ProgramFiles%\Rust DICOM Station` (all users, asks for elevation).
* **Dependencies** - checks for the Microsoft Visual C++ runtime that Rust's
  MSVC target links against and installs it from Microsoft when missing.
  Rendering needs Direct3D 12 or Vulkan, which the display driver already
  provides, so there is nothing to install for the GPU.
* **Graphics backend** - its own page, because a handful of Windows machines
  advertise a Vulkan driver that cannot start (see below).
* **Model weights, optionally** - pre-downloads and converts the
  TotalSegmentator weights (6 mm, 3 mm, the 1.5 mm set, or everything) using
  the viewer's own downloader, so the first auto-segmentation run does not
  have to wait for a 135 MB … 1.3 GB download. Skipped by default. They go
  where the viewer keeps every engine's weights: the `totalsegmentator/`
  sub-folder of the model folder, by default
  `%LOCALAPPDATA%\RustDICOMStation\models` for either scope. A model folder
  chosen elsewhere is recorded as `models_dir` in the installing user's
  `%LOCALAPPDATA%\RustDICOMStation\viewer_settings.txt`. The SegVol and
  MedSAM2 weights are never pre-fetched - their licences allow only a
  download by the user, which the viewer does on first use.
* **Integration** - Start-menu and desktop shortcuts, an
  "Open with Rust DICOM Station" verb on folders (the viewer takes a
  directory), a `.dcm`/`.dicom` entry that is *added* to `OpenWithProgids`
  rather than hijacking whatever owns DICOM files today, and optionally the
  program folder on `PATH`.
* **Uninstall and update** - an entry in Apps & features plus
  `rds-setup.exe` in the program folder: the setup program without its
  payload, which uninstalls (`--uninstall`) and fetches the newest release
  (`--update`, the *Update Rust DICOM Station* Start-menu entry). Earlier
  versions called it `uninstall.exe`; updating one of them removes that.

Everything created is recorded in `install-manifest.txt`, and the uninstaller
removes exactly what is listed there - nothing else in the folder, and the
`PATH` entry only if the installer added it. The model folder is kept unless
you ask for it to go - and then it goes whole, every engine's downloads
included (an empty one is cleaned up either way).

## Updating

There is one installation per machine, and a newer setup replaces it rather
than adding a second copy. Nothing has to be uninstalled first.

* **Run a newer setup over an installation** and the first page says which
  version is installed where, and offers **Update to X**. The update keeps
  the folder, the scope and the answers given at installation (shortcuts,
  file association, `PATH`, the MCP server, the graphics backend, the model
  folder), keeps your settings and the downloaded models, and removes the
  files the new version no longer ships. *Change options* goes through the
  usual pages first. The same version reinstalls (restores missing files);
  an older one warns before it replaces a newer installation.
* **Other installations are removed.** A copy in another folder, or in the
  other scope (one "just me" and one "all users"), is listed on the first
  page and removed - models kept - before the new one is written. Untick the
  box, or pass `--keep-others`, to leave it. Removing a copy that was
  installed for all users asks for administrator rights.
* **Update from the installed program.** *Start ▸ Update Rust DICOM Station*
  (or `rds-setup.exe --update` in the program folder) looks up the newest
  release on GitHub, downloads its installer, checks it against the
  release's `SHA256SUMS` and runs it; it updates in place as above. When the
  installation is already current it says so and does nothing.
* **An old setup knows it is old.** Started while a newer release exists,
  the wizard says so on its first page and offers to download and run that
  one instead.
* **winget**, once the package is published - see below.

How an installation is found: by its Apps & features key (`RustDicomStation`,
under `HKEY_CURRENT_USER` or `HKEY_LOCAL_MACHINE`) and the
`install-manifest.txt` in its folder. The manifest lists every file that was
written, which is what lets an update tell which old files are now obsolete.

Update from the command line, for scripts:

```
"%LOCALAPPDATA%\Programs\Rust DICOM Station\rds-setup.exe" --update --silent
rds-setup --silent                      # a downloaded newer setup, same effect
```

`--update --silent` exits with 0 when there was nothing to do. A silent run
refuses to replace a newer installed version with an older one (exit code 4)
unless `--allow-downgrade` is given.

## winget

The release workflow writes winget manifests for every installer it builds
(`rust-dicom-station-X.Y.Z-winget-manifests.zip` on the release) and, once
the package exists in the community repository, submits each new version to
[microsoft/winget-pkgs](https://github.com/microsoft/winget-pkgs) itself.
After that:

```
winget install RDS.RustDICOMStation                  # just me
winget install RDS.RustDICOMStation --scope machine  # all users
winget upgrade RDS.RustDICOMStation                  # or: winget upgrade --all
winget uninstall RDS.RustDICOMStation
```

`winget upgrade` runs the new setup over the existing installation, which
updates it in place as described above. An installation made with the wizard
is recognised too, because winget matches the Apps & features entry the setup
writes (product code `RustDicomStation`).

What the manifests declare:

| winget | setup |
|---|---|
| `Silent` | `--silent` |
| `SilentWithProgress` (winget's default) | `--passive`: a progress window, no questions, closes itself |
| `InstallLocation` | `--dir "<INSTALLPATH>"` (`winget install --location`) |
| scope `user` | `--just-me --no-launch` |
| scope `machine` | `--all-users --no-launch`, run elevated by winget |
| dependency | `Microsoft.VCRedist.2015+.x64` |
| exit codes | 2 `packageInUse`, 3 `cancelledByUser`, 4 `downgrade`, 5 `noNetwork` |

To try a manifest before it is published (from an elevated prompt, once):

```
winget settings --enable LocalManifestFiles
winget install --manifest .\winget-manifests
```

Publishing is described in
[docs/release-versioning.md](../docs/release-versioning.md#winget).

## Command line

The same binary drives everything; `--silent` and `--console` skip the
wizard, which is what you want for deployment.

```
rds-setup --silent --dir "D:\Apps\RDS" --add-to-path --models 3mm
rds-setup --silent --all-users            # from an elevated prompt
rds-setup --passive                       # progress window only
rds-setup.exe --update --silent           # in the program folder: newest release
rds-setup.exe --uninstall --silent --remove-models
rds-setup --help
```

Over an existing installation, the options on the command line override the
ones it was made with; everything not mentioned stays as it was.

`--models` takes `none | 6mm | 3mm | 1.5mm | all`, `--graphics` takes
`vulkan | dx12 | auto` (default `vulkan`); the other flags are `--just-me`,
`--models-dir` (the model folder), `--no-start-menu`, `--no-desktop-shortcut`,
`--no-file-association`, `--no-vcredist`, `--no-mcp`, `--no-launch`,
`--keep-others`, `--allow-downgrade`, and `--from` for the uninstaller.

| exit code | meaning |
|---|---|
| 0 | done, or `--update` found nothing newer |
| 1 | failed |
| 2 | the viewer is running from the folder being written or removed |
| 3 | cancelled |
| 4 | a newer version is installed (`--silent` / `--passive` without `--allow-downgrade`) |
| 5 | the newest release could not be looked up or downloaded |

Building with `--no-default-features` drops the `prefetch-models` feature,
and with it the dependency on the viewer library; the option then disappears
from the wizard and the viewer downloads weights on first use as usual.

## The graphics page

Some Windows machines advertise a Vulkan driver that cannot actually create a
device. `wgpu` prefers Vulkan, so the viewer would die before drawing
anything, on a machine where nothing else is wrong - and the only escape was
knowing to set `WGPU_BACKEND=dx12` before starting it. So the wizard asks,
between the options page and the installation itself:

* **Vulkan (recommended)** - preselected; right on the overwhelming majority
  of machines.
* **DirectX 12** - Windows' own, dependable on everything from Windows 10 on.
  The answer for a machine where the viewer will not start.
* **Automatic** - whatever the graphics library picks, which is what older
  versions did.

Three details make the page do its job rather than merely exist:

* **The installer draws with the same library, on the same machine**, so a
  broken Vulkan driver takes the setup program down too - and the setup
  program is where the page that fixes it lives. Its window is therefore
  attempted rather than opened: the default first, then Direct3D 12, then
  Vulkan, each attempt catching a panic from inside the driver. If all three
  fail the text interface takes over, as it already did for a head-less
  session.
* **Falling back is itself an answer.** If the wizard's window only appeared
  on Direct3D 12, the viewer will need Direct3D 12 too, so the page says so
  and preselects it.
* **The answer is written twice.** `viewer-defaults.txt` goes beside the
  installed executable and is read by every user of the machine before their
  own settings - the only thing that works for an all-users installation,
  where the administrator's `%LOCALAPPDATA%` is not the one the viewer will
  run under. The installing user's own `viewer_settings.txt` is updated as
  well, because it wins over the defaults and would otherwise keep an older
  answer: someone re-running the installer to change the backend must
  actually get the change.

None of this is load-bearing for a working machine, and none of it is the
last line of defence: the viewer falls back between backends on its own too.
See [docs/viewer.md](../docs/viewer.md#graphics-backend).

## Source map

| file | contents |
|---|---|
| `src/main.rs` | argument parsing, mode dispatch, elevation re-launch |
| `src/product.rs` | product identity and version comparison, shared with `rds-pack` |
| `src/plan.rs` | install options, default paths, exit codes |
| `src/existing.rs` | finding installed copies and reading back their options |
| `src/payload.rs` | the appended-zip payload format and extraction |
| `src/install.rs` | the install and update steps and the manifest |
| `src/update.rs` | the newest GitHub release: lookup, download, SHA-256 check, hand-over |
| `src/uninstall.rs` | manifest-driven removal, self-deleting uninstaller |
| `src/deps.rs` | Visual C++ runtime detection and installation |
| `src/models.rs` | optional TotalSegmentator weight pre-fetch |
| `src/ui.rs` | the egui wizard |
| `src/console.rs` | text-mode / silent front end |
| `src/win/` | shell links, registry, known folders, console attach |
| `src/bin/pack.rs` | `rds-pack`, builds the shippable installer and its winget manifests |
