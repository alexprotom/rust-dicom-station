# AppImage

The Linux release for any distribution: one executable file holding the
viewer and the MCP server, built by the release workflow on every push to
`main` and attached to the GitHub release as
`rust-dicom-station-X.Y.Z-linux-x86_64.AppImage`.

```text
build-appimage.sh            cargo -> AppDir -> linuxdeploy -> the image, checked
AppRun                       the image's entry point: `mcp` starts the MCP server
rust-dicom-station.desktop   the desktop entry inside the image
out/                         everything this folder produces (git-ignored)
```

## Building

```bash
sudo apt-get install libxkbcommon-dev libwayland-dev libx11-dev libxcursor-dev \
    libxrandr-dev libxi-dev libgl1-mesa-dev libfuse2 file wget

packaging/linux/appimage/build-appimage.sh
```

Result: `packaging/linux/appimage/out/rust-dicom-station-<version>-linux-x86_64.AppImage`.
`--skip-build` packages what the last `cargo build --release --features mcp`
left; `--linuxdeploy <file>` uses a linuxdeploy already on the machine
instead of downloading the `continuous` build into `out/`.

The script ends with two checks the workflow used to make: the image is
executable, and `<image> mcp --check` starts the MCP server from inside it,
which proves the dispatcher in `AppRun` survived the packaging.

## What is in the image

`usr/bin/rust-dicom-station`, `usr/bin/rds-mcp`, the desktop entry, the
256 px icon from `assets/`, and the libraries linuxdeploy found the two
executables to need. Models are not included: the viewer downloads them on
first use into `~/.local/share/RustDICOMStation/models`.

An MCP client is pointed at the image itself with `mcp` as the first
argument, because a binary inside an AppImage has no path of its own:

```json
{"command": "/path/to/rust-dicom-station-X.Y.Z-linux-x86_64.AppImage", "args": ["mcp"]}
```

*Settings > MCP server > Copy client configuration* writes exactly that
when the viewer runs from an AppImage (`src/settings.rs`, `McpLaunch`). A
copy or symlink of the image named `rds-mcp` works as well, for clients
that will not pass an argument.
