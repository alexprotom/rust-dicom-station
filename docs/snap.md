# The snap package

Rust DICOM Station is published to the [Snap Store](https://snapcraft.io/rust-dicom-station)
as `rust-dicom-station`. The recipe is [snap/snapcraft.yaml](../snap/snapcraft.yaml);
this page explains what it does and why, where the program keeps its files
inside a snap, and how a version gets from `main` to the store.

## Installing

```text
sudo snap install rust-dicom-station
```

This installs two commands:

| Command | What it is |
|---|---|
| `rust-dicom-station` | The viewer, also in the desktop's application menu. Takes the same arguments as the plain executable: `rust-dicom-station [DICOM_FOLDER] [COMPARISON_FOLDER]` |
| `rust-dicom-station.rds-mcp` | The MCP server ([mcp.md](mcp.md)) |

snapd installs and connects two runtime snaps with it: `gnome-46-2404` (the
display libraries, fonts, cursor themes) and `mesa-2404` (the graphics
drivers). They are shared with every other snap that uses them.

Updates arrive by themselves; `sudo snap refresh rust-dicom-station` fetches
one at once, `sudo snap revert rust-dicom-station` goes back to the previous
revision.

## What the snap may read

A snap runs confined: it sees what its interfaces grant and nothing else.

| Interface | Connected | Grants |
|---|---|---|
| `home` | automatically | Your home folder, **except hidden files and folders** (names starting with a dot) at its top level |
| `removable-media` | **by hand, once** | `/media`, `/run/media` and `/mnt`: USB drives, mounted shares |
| `network` | automatically | Downloading model weights |
| display, GPU, desktop | automatically | The window, Vulkan / OpenGL, the file dialogs |

To open data outside your home folder:

```text
sudo snap connect rust-dicom-station:removable-media
```

A folder elsewhere (`/data`, `/srv`, a second disk mounted somewhere else)
is not reachable from a strictly confined snap at all; mount it under
`/mnt` or `/media`, or link it into your home folder with a bind mount (a
symbolic link does not help: the snap follows it to a place it may not
read). `snap connections rust-dicom-station` lists what is connected.

The file dialogs are the desktop's own (the XDG file chooser portal), so
they look like every other application's. A folder picked there may come
back as a path under `/run/user/<uid>/doc/`: the desktop's document portal
lends it to the snap that way, and the program reads it like any other
folder.

## Where the files are

snapd gives every revision its own home folder, `~/snap/rust-dicom-station/<revision>`,
and copies it on every refresh, keeping the last few. Model weights run to
gigabytes and the patient archive can be larger, so the program keeps
nothing there: inside a snap the configuration and data folders are in
`~/snap/rust-dicom-station/common`, which every revision shares.

| | Plain installation (Linux) | Snap |
|---|---|---|
| Settings, `mcp.toml` | `~/.config/RustDICOMStation` | `~/snap/rust-dicom-station/common/config` |
| Models, archive, templates, MCP audit log | `~/.local/share/RustDICOMStation` | `~/snap/rust-dicom-station/common/data` |

The switch is made in `settings::config_dir` / `settings::data_dir`, keyed on
the variables snapd sets (`SNAP_INSTANCE_NAME`, `SNAP`, `SNAP_USER_COMMON`;
see `settings::SnapEnv`). *Settings ▶ MCP server* and the model manager
show the exact paths.

Weights already downloaded by an AppImage or a source build can be moved
over instead of downloaded again. The snap cannot read `~/.local` itself
(it is hidden), so move them from a terminal:

```text
mkdir -p ~/snap/rust-dicom-station/common/data
mv ~/.local/share/RustDICOMStation/models ~/snap/rust-dicom-station/common/data/
```

A model folder chosen in the program must be one the snap can write: under
the home folder and not hidden, or on removable media.

## Graphics

The viewer and the inference engines draw and compute through Vulkan, which
`mesa-2404` provides for Intel, AMD and other Mesa-driven GPUs, with
lavapipe as a software fallback when there is no GPU at all. On a machine
with the proprietary NVIDIA driver, snapd makes the host's driver libraries
available to the snap. If Vulkan cannot create a device, the viewer falls back to OpenGL by
itself, as on any other installation ([viewer.md](viewer.md#graphics-backend));
`WGPU_BACKEND=gl rust-dicom-station` forces it from a terminal.

## The MCP server in a snap

An MCP client cannot run the executable inside the snap's read-only mount;
it runs the snap's command. *Settings ▶ MCP server ▶ Copy client
configuration* already writes it that way:

```json
{ "mcpServers": { "rust-dicom-station": {
    "command": "/snap/bin/rust-dicom-station.rds-mcp", "args": [] } } }
```

`mcp.toml` goes into `~/snap/rust-dicom-station/common/config`. Its `roots`
and `output_dir` must be folders the snap may use (see above): paths under
the home folder, or under `/media` / `/mnt` with `removable-media`
connected. Check it with:

```text
rust-dicom-station.rds-mcp --check
```

Two details of the recipe exist for the server:

* **It does not run through the GNOME launcher.** The launcher rebuilds
  icon and font caches after every refresh, and the tools it runs may write
  to standard output, which belongs to the protocol. The server gets only
  the GPU wrapper, so the inference engines still find Vulkan.
* **`open_in_viewer` starts the viewer through the launcher**
  (`$SNAP/snap/command-chain/desktop-launch`, `viewer_command` in
  `src/mcp/tools/output.rs`). The viewer runs under the server's
  confinement, which is why the server's app also plugs the display
  interfaces. A `viewer_exe` set in `mcp.toml` is started as it is.

## The recipe, piece by piece

* **`base: core24`, strict confinement.** Classic confinement is granted
  only to a few kinds of software (compilers, IDEs) after a manual review; a
  viewer does not qualify, and does not need it.
* **`rust-deps`, the toolchain.** The rust plugin would rather install the
  `rustup` snap into the build instance, but that install is skipped without
  an error when the instance cannot reach the snap store, and the build then
  stops with `'rustup' not found` before compiling anything (which is how
  the first build failed). This part installs the stable toolchain from
  rustup.rs instead, and `rust-channel: none` plus `after: [rust-deps]` on
  the part below tells the plugin not to look for the snap. `rust-deps` is
  the exact name the plugin's environment check accepts.
* **`rust-dicom-station`, `plugin: rust`.** It runs `cargo install --locked
  --path . --features mcp`, which builds both executables with the release
  profile of `Cargo.toml` (the default `gpu` feature stays on) into
  `$SNAP/bin`; the plugin puts `$HOME/.cargo/bin` on the build path itself.
  `Cargo.lock` is not in the repository, so dependencies resolve as they do
  for every other release build. `CARGO_PROFILE_RELEASE_STRIP=debuginfo` removes the standard
  library's debug information; symbol names stay, so a backtrace still
  names its frames. The version is read from `Cargo.toml`, with the same
  expression the release workflow uses.
* **The GNOME extension on the viewer.** It brings the display stack, the
  `gpu-2404` interface to `mesa-2404`, fonts, cursor themes, and the
  portals. The window library opens its libraries at run time (`dlopen`),
  so nothing in the build links against them.
* **`x11-keyboard`.** In an X11 session the window library needs
  `libxkbcommon-x11`, which neither runtime ships. It is the one library the
  snap carries itself; snapcraft puts `$SNAP/usr/lib/<arch>` into the snap's
  `LD_LIBRARY_PATH`, so staging it is all it takes.
* **The desktop entry** is `snap/local/rust-dicom-station.desktop`, installed
  as `usr/share/applications/rust-dicom-station.desktop`; snapcraft rewrites
  its `Exec` and points `Icon` at `assets/rust-dicom-station.png`, which is
  also the store icon. It declares `application/dicom`, so a file manager
  offers the viewer for `.dcm` files.
* **Only `amd64`**, like the AppImage. Nothing in the code is x86-specific;
  adding `arm64:` under `platforms` and an `ubuntu-24.04-arm` runner is all
  an ARM build would take.

## Building it

`snapcraft pack` builds in a fresh LXD container, so it needs a Linux machine
with snapd (Ubuntu, or Ubuntu under WSL 2 with systemd enabled):

```text
sudo snap install snapcraft --classic
sudo snap install lxd && sudo lxd init --auto
snapcraft pack
sudo snap install --dangerous ./rust-dicom-station_<version>_amd64.snap
```

Run it from a **fresh clone**: the part's source is the project folder as
it is, and snapcraft copies it whole into the container, `target/` and the
example data included. The first build takes 30 to 60 minutes; `snapcraft
clean` starts over.

Without a Linux machine, *Actions ▶ Snap ▶ Run workflow* builds any branch
on GitHub and attaches the snap to the run as an artifact (`linux-snap`).
The same workflow can put that build on the `edge`, `beta` or `candidate`
channel for testers:

```text
sudo snap install rust-dicom-station --edge
```

## Publishing

`.github/workflows/snap.yml` builds the snap, installs it on the runner, and
checks two things before anything is published: that
`rust-dicom-station.rds-mcp --check` runs inside the confinement, and that
the viewer opens a window under Xvfb and is still running 40 seconds later.
Then it uploads the snap as an artifact and, when the secret
`SNAPCRAFT_STORE_CREDENTIALS` is set, to the store.

The release workflow calls it on every push to `main`, after the version
check, and releases to `stable` (or to the channel in the repository
variable `SNAP_CHANNEL`, if one is set). The GitHub Release does not wait for
it. Without the secret the snap is built and tested but not uploaded.

### Once, by hand

1. Create an account at <https://snapcraft.io> (an Ubuntu One account).
2. On a machine with snapcraft: `snapcraft login`, then
   `snapcraft register rust-dicom-station`.
3. Upload the first build yourself, from a Linux machine or with the artifact
   of a *Snap* workflow run:

   ```text
   snapcraft upload --release=edge rust-dicom-station_<version>_amd64.snap
   ```

   A strictly confined snap with these interfaces normally passes the
   store's automatic review without a human.
4. Fill in the store listing (screenshots, category *Science* or
   *Health and fitness*) at <https://snapcraft.io/rust-dicom-station/listing>.
5. Export a login for the workflow, limited to this snap:

   ```text
   snapcraft export-login --snaps=rust-dicom-station \
     --acls package_access,package_push,package_update,package_release \
     snap-credentials.txt
   ```

   Save the file's content as the repository secret
   `SNAPCRAFT_STORE_CREDENTIALS` (*Settings > Secrets and variables >
   Actions*) and delete the file. The exported login expires (`--expires`
   sets the date); when uploads start failing with an authentication error,
   export a new one and replace the secret.
6. Optional: to have releases land on `candidate` first and promote them by
   hand (`snapcraft release rust-dicom-station <revision> stable`), set the
   repository variable `SNAP_CHANNEL` to `candidate`.

`removable-media` can be connected automatically for every user of the snap,
but only by a request to the store reviewers: a post in the
[store-requests](https://forum.snapcraft.io/c/store-requests/19) forum
category explaining that DICOM data lives on external drives and mounted
shares. Until then the install instructions carry the `snap connect` line.

## Troubleshooting

| Symptom | Cause, remedy |
|---|---|
| `ERROR: the gpu-2404 interface isn't connected` | `sudo snap install mesa-2404`, then `sudo snap connect rust-dicom-station:gpu-2404 mesa-2404` |
| A folder shows as empty or cannot be opened | It is hidden, or outside the home folder: see *What the snap may read* |
| The window does not open, nothing obvious on the terminal | `snap run --shell rust-dicom-station` opens a shell inside the confinement; `sudo snap install snappy-debug && sudo snappy-debug` shows what the confinement refuses while the program runs |
| Settings from an AppImage or a source build are missing | They are not shared: the snap has its own folders, see *Where the files are* |
| A build fails with `'rustup' not found and part ... does not depend on a part named 'rust-deps'` | The `rust-deps` part is missing, or the rust part lost `rust-channel: none` / `after: [rust-deps]`: see *The recipe, piece by piece* |
