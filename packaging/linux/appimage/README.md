# AppImage

The Linux release for any distribution: one executable file holding the
viewer and the MCP server, built by the release workflow on every push to
`main` and attached to the GitHub release as
`rust-dicom-station-X.Y.Z-x86_64.AppImage` (no "linux" in the name: every
AppImage is for Linux, and the [AppImage catalog](https://appimage.github.io)
warns about names that say so).

```text
build-appimage.sh            cargo -> AppDir -> linuxdeploy -> the image, checked
AppRun                       the image's entry point: `mcp` starts the MCP server
rust-dicom-station.desktop   the desktop entry inside the image
out/                         everything this folder produces (git-ignored)
```

## Building

```bash
sudo apt-get install libxkbcommon-dev libxkbcommon-x11-0 libwayland-dev libx11-dev \
    libxcursor-dev libxrandr-dev libxi-dev libgl1-mesa-dev libfuse2 file wget

packaging/linux/appimage/build-appimage.sh
```

Result: `packaging/linux/appimage/out/rust-dicom-station-<version>-x86_64.AppImage`.
Without FUSE (a container) the script sets `APPIMAGE_EXTRACT_AND_RUN=1`, so
linuxdeploy and the final check unpack themselves instead of mounting.
`--skip-build` packages what the last `cargo build --release --features mcp`
left; `--linuxdeploy <file>` uses a linuxdeploy already on the machine
instead of downloading the `continuous` build into `out/`.

The script ends with two checks the workflow used to make: the image is
executable, and `<image> mcp --check` starts the MCP server from inside it,
which proves the dispatcher in `AppRun` survived the packaging.

## What is in the image

`usr/bin/rust-dicom-station`, `usr/bin/rds-mcp`, the desktop entry, the
256 px icon from `assets/`, the libraries linuxdeploy found the two
executables to need, and the window-system libraries the viewer loads while
it starts. Models are not included: the viewer downloads them on first use
into `~/.local/share/RustDICOMStation/models`.

The window-system libraries are the part linuxdeploy cannot find by itself:
`winit` does not link them, it opens them by name at run time
(`libxkbcommon-x11`, `libxkbcommon`, `libXcursor`, `libXi`, `libXrandr`,
`libwayland-cursor`), so a host without one of them - a minimal desktop, or
the AppImage catalog's test machine, which has no `libxkbcommon-x11` -
failed before any window existed. The script hands them to linuxdeploy
with `--library` and checks they arrived. What the AppImage excludelist
names (`libX11`, `libxcb`, `libwayland-client`, `libGL`, `libEGL`) still
comes from the host, as it must.

## The C library

The release builds the image inside an Ubuntu 20.04 container (glibc 2.31),
not on the runner's own newer Ubuntu: an executable asks at run time for
the C library version it was linked against, or newer, so the older the
build machine, the more distributions the image starts on. GitHub no longer
offers 20.04 runners, hence the container.

An MCP client is pointed at the image itself with `mcp` as the first
argument, because a binary inside an AppImage has no path of its own:

```json
{"command": "/path/to/rust-dicom-station-X.Y.Z-x86_64.AppImage", "args": ["mcp"]}
```

*Settings > MCP server > Copy client configuration* writes exactly that
when the viewer runs from an AppImage (`src/settings.rs`, `McpLaunch`). A
copy or symlink of the image named `rds-mcp` works as well, for clients
that will not pass an argument.
