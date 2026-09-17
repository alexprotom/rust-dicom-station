# The Flatpak

Rust DICOM Station is published to [Flathub](https://flathub.org/apps/io.github.alexprotom.rust-dicom-station)
as `io.github.alexprotom.rust-dicom-station`. The manifest is
[flatpak/io.github.alexprotom.rust-dicom-station.yml](../flatpak/io.github.alexprotom.rust-dicom-station.yml);
this page explains what it does and why, where the program keeps its files
inside the sandbox, and how a version gets from `main` to Flathub.

The snap ([snap.md](snap.md)) and the Flatpak are two packagings of the same
program with the same guarantees; which one a user wants depends on their
distribution. The AppImage stays as the no-installation option.

## Installing

```text
flatpak install flathub io.github.alexprotom.rust-dicom-station
flatpak run io.github.alexprotom.rust-dicom-station
```

Two commands come with it:

| Command | What it is |
|---|---|
| `flatpak run io.github.alexprotom.rust-dicom-station` | The viewer, also in the desktop's application menu. It takes the same arguments as the plain executable: a DICOM folder, and a second one for comparison mode |
| `flatpak run --command=rds-mcp io.github.alexprotom.rust-dicom-station` | The MCP server ([mcp.md](mcp.md)) |

Updates arrive with `flatpak update`, which desktops run by themselves.

## What the sandbox allows

| Permission | Why |
|---|---|
| Wayland or X11, IPC | The window |
| `--device=dri` | Vulkan for the views and for the inference engines |
| `--share=network` | Downloading model weights |

There is deliberately no filesystem permission. Flathub's linter rejects
blanket home access (`finish-args-home-filesystem-access`) and asks
applications to go through portals instead, which this one does: the file
dialogs are the desktop's own XDG file chooser portal, so **a folder opened
from the dialog is readable wherever it lives**, inside the home folder or
not. Settings, downloaded models, templates and the patient archive live in
the application's own folder and need no permission at all.

What the portal cannot cover is a path the program is handed rather than
asked for: a folder named on the command line, `roots` and `output_dir` in
`mcp.toml`, or a model folder moved to another disk. Grant those once:

```text
flatpak override --user --filesystem=~/DICOM io.github.alexprotom.rust-dicom-station
flatpak override --user --filesystem=/data io.github.alexprotom.rust-dicom-station
flatpak override --user --filesystem=/run/media/$USER io.github.alexprotom.rust-dicom-station
```

`flatpak info --show-permissions io.github.alexprotom.rust-dicom-station`
lists what is in force, and [Flatseal](https://flathub.org/apps/com.github.tchx84.Flatseal)
is the graphical way to the same settings.

## Where the files are

Flatpak points the XDG variables into the application's own folder, so the
program's ordinary Linux paths land there with no special case in the code:

| | Plain installation | Flatpak |
|---|---|---|
| Settings, `mcp.toml` | `~/.config/RustDICOMStation` | `~/.var/app/io.github.alexprotom.rust-dicom-station/config/RustDICOMStation` |
| Models, archive, templates, MCP audit log | `~/.local/share/RustDICOMStation` | `~/.var/app/io.github.alexprotom.rust-dicom-station/data/RustDICOMStation` |

Weights already downloaded by an AppImage or a source build can be moved
over instead of fetched again:

```text
mkdir -p ~/.var/app/io.github.alexprotom.rust-dicom-station/data/RustDICOMStation
mv ~/.local/share/RustDICOMStation/models \
   ~/.var/app/io.github.alexprotom.rust-dicom-station/data/RustDICOMStation/
```

## Graphics

The views and the inference engines draw and compute through Vulkan, which
the freedesktop runtime provides for Intel, AMD and other Mesa-driven GPUs,
with lavapipe as the software fallback. On a machine with the proprietary
NVIDIA driver, Flatpak installs the matching `org.freedesktop.Platform.GL.nvidia-*`
extension by itself. If Vulkan cannot create a device the viewer falls back
to OpenGL on its own ([viewer.md](viewer.md#graphics-backend)), and
`flatpak run --env=WGPU_BACKEND=gl io.github.alexprotom.rust-dicom-station`
forces it.

## The MCP server in a Flatpak

A client cannot run the executable inside the sandbox; it runs it through
`flatpak`. *Settings ▶ MCP server ▶ Copy client configuration* writes the
entry that way:

```json
{ "mcpServers": { "rust-dicom-station": { "command": "flatpak",
    "args": ["run", "--command=rds-mcp", "io.github.alexprotom.rust-dicom-station"] } } }
```

`mcp.toml` goes into
`~/.var/app/io.github.alexprotom.rust-dicom-station/config/RustDICOMStation`.
The server has no window, so nothing it reads can come from the file
dialog's portal: every folder in `roots` and `output_dir` has to be granted
with `flatpak override --filesystem=...` first (see above). Check the
result with:

```text
flatpak run --command=rds-mcp io.github.alexprotom.rust-dicom-station --check
```

`open_in_viewer` needs nothing special here: the viewer it starts is the
executable beside it, in the same sandbox.

## The manifest, piece by piece

* **`org.freedesktop.Platform` 26.08** rather than the GNOME runtime: the
  program draws with its own toolkit and needs the display, Mesa and the
  portals, none of GTK.
* **Vendored crates.** Flathub builds offline. `cargo-sources.json` lists
  every crate with its checksum, generated from `Cargo.lock` by
  [flatpak/update-cargo-sources.sh](../flatpak/update-cargo-sources.sh).
  This is why `Cargo.lock` is committed: the build that Flathub performs
  and the build you perform have to resolve to the same dependencies.
* **One module, `buildsystem: simple`.** The Rust SDK extension supplies
  the toolchain; `cargo --offline build --release --features mcp` produces
  both executables, and the install commands place them with the desktop
  entry, the metainfo, the icon and the licence under `/app`.
* **The app ID is the file name of everything**: desktop entry, metainfo
  and icon all carry `io.github.alexprotom.rust-dicom-station`, which
  Flathub requires and which is how the desktop matches the window to the
  application.
* **`x-checker-data`** lets Flathub's data checker notice a new `v*` tag
  and open the update pull request for the manifest.
* **The linter is part of the deal.** `flatpak-builder-lint` runs on the
  manifest and on the built repository, and Flathub will not take a
  submission that fails it. Two of its rules shaped this manifest: no
  blanket home access, and keep the runtime current. Genuine exceptions
  exist (a pull request to `flathub/flatpak-builder-lint` with the
  reasoning, which their policy says must be written by a person, not
  generated), but `--filesystem=home` for a viewer that has a working file
  dialog would be a hard sell.

## Building it yourself

Any Linux machine with Flatpak:

```text
flatpak install flathub org.flatpak.Builder
flatpak run org.flatpak.Builder --user --force-clean --install-deps-from=flathub \
  --install build flatpak/io.github.alexprotom.rust-dicom-station.yml
flatpak run io.github.alexprotom.rust-dicom-station
```

The build takes 30 to 60 minutes, most of it the release build of the
crate. Note that the manifest builds the **tagged release**, not your
working tree: change the `tag` and `commit` of the source to test something
else.

Before submitting anything, run the linter Flathub runs:

```text
flatpak run --command=flatpak-builder-lint org.flatpak.Builder manifest \
  flatpak/io.github.alexprotom.rust-dicom-station.yml
flatpak run --command=flatpak-builder-lint org.flatpak.Builder repo repo
```

`.github/workflows/flatpak.yml` does exactly this on demand and attaches
the resulting `.flatpak` bundle to the run, which is the easy way to test a
manifest change without a Linux machine at hand.

## Publishing

Flathub builds the application on its own infrastructure from a manifest in
its own repository, so releases are not pushed from here the way the snap
is. The flow is:

### Once, by hand

1. Fork [flathub/flathub](https://github.com/flathub/flathub) and create a
   branch named after the app ID.
2. Put the four files from `flatpak/` in the root of that branch: the
   manifest, the metainfo, the desktop entry and `cargo-sources.json`.
3. Open a pull request against the **`new-pr`** branch (not `master`),
   filling in the template. Reviewers look at the permissions, the metainfo
   and the licensing; the `--share=network` line is the one worth
   explaining in the description (model weights, downloaded on request).
4. Once merged, Flathub creates
   `github.com/flathub/io.github.alexprotom.rust-dicom-station` and gives
   you write access. Their buildbot builds every commit on `master` there
   and publishes it.

### Every release

1. `flatpak/update-cargo-sources.sh` after the version bump, so the vendored
   list matches the new `Cargo.lock`.
2. Update `tag` and `commit` in the manifest, and add a `<release>` entry to
   the metainfo.
3. Copy the four files into the Flathub repository and push (or let
   Flathub's data checker open that pull request from `x-checker-data` and
   just refresh `cargo-sources.json` in it).

Nothing in the GitHub release workflow has to change; the Flatpak is built
by Flathub, from the tag your release created.
