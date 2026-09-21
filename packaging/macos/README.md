# macOS packaging

The viewer's macOS front end is the viewer: nothing here is a second
program. `eframe` opens a Metal window on macOS exactly as it opens a Vulkan
one on Linux, `src/settings.rs` already puts configuration and model weights
under `~/Library/Application Support/RustDICOMStation`, and `src/gfx.rs`
already knows Metal is the only backend Apple has. What was missing was a
*package*: a `.app` bundle a user can drag into Applications, and a `.dmg`
to put it in.

This folder is that packaging, and nothing else. It holds no Rust code and
no second workspace - unlike `packaging/windows/installer/` (Windows) and `packaging/android/`, which
are crates of their own.

```text
build-app.sh        builds one architecture: cargo -> .app -> .dmg, signed
Info.plist.in       the bundle's description of itself; @VERSION@ substituted
entitlements.plist  hardened-runtime entitlements, used only when signing
brew-cask.sh        writes the Homebrew cask for a released version
out/                everything this folder produces (git-ignored)
```

## Building

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin

packaging/macos/build-app.sh                  # this Mac's architecture
packaging/macos/build-app.sh --arch x86_64    # Intel, from an Apple Silicon Mac
packaging/macos/build-app.sh --arch arm64
```

Result: `packaging/macos/out/rust-dicom-station-<version>-macos-<arch>.dmg`, holding
`Rust DICOM Station.app` and a link to `/Applications`.

An Apple Silicon Mac builds both architectures; an Intel Mac builds only its
own, because a Rust toolchain on Intel has no arm64 linker. This is why the
release workflow runs on an Apple Silicon runner and cross-compiles the
Intel half - and why it will keep working after August 2027, when GitHub
retires its last x86_64 macOS image.

`--dev` builds the dev profile, `--skip-build` packages what the last build
left, `--no-dmg` stops at the bundle. `--help` prints the whole of this.

## macOS 12 and newer

`MACOSX_DEPLOYMENT_TARGET` defaults to `12.0` and is used three times: it is
exported for `cargo build`, substituted into `LSMinimumSystemVersion`, and
then **read back out of the linked binaries** and compared against the
plist. The check is there because the two can drift apart silently: a
binary built for macOS 14 inside a bundle that claims 12 launches on
Monterey and dies in `dyld`, which Finder reports as "the application is
damaged", and nothing before the user's machine would have said otherwise.

To move the floor, set the variable - the plist and the check follow:

```bash
MACOSX_DEPLOYMENT_TARGET=13.0 packaging/macos/build-app.sh
```

## What is in the bundle

```text
Rust DICOM Station.app/Contents/
  Info.plist
  PkgInfo
  MacOS/rust-dicom-station      the viewer
  MacOS/rds-mcp                 the MCP server (--features mcp)
  Resources/rust-dicom-station.icns
```

`rds-mcp` sits beside the viewer because `settings::mcp_exe_path()` looks in
the folder of the running executable. That one fact is the whole of the
macOS MCP arrangement: no macOS-specific code, and an MCP client can be
pointed straight at

```text
/Applications/Rust DICOM Station.app/Contents/MacOS/rds-mcp
```

The icon is generated from `assets/rust-dicom-station.png`, the same picture
the window, the Windows resource and the Linux desktop entry use. Only the
sizes the source can fill are written into the `.icns`: the PNG is 256 px
today, so the bundle carries up to `icon_128x128@2x`. Replacing that asset
with a 1024 px version makes the larger representations appear on the next
build, with no change here.

No `CFBundleDocumentTypes`. It would put the program into *Open with* for
folders and DICOM files, but macOS delivers such an open as an Apple Event
and the viewer reads its paths from `argv` - the entry would promise
something that does nothing.

## Signing and notarisation

Both are optional and both are driven by environment variables, so a local
build needs no Apple account at all.

| Variable | Effect |
|---|---|
| `RDS_CODESIGN_IDENTITY` | Sign with this Developer ID under the hardened runtime. Unset: ad-hoc signature. |
| `RDS_NOTARY_APPLE_ID`, `RDS_NOTARY_PASSWORD`, `RDS_NOTARY_TEAM_ID` | Submit the disk image to Apple and staple the ticket into it. All three, or none. |

Ad-hoc is the default and it works - the program runs - but Gatekeeper has
no developer to trust, so the **first** launch has to be right-click ▸ *Open*
in Finder. A notarised image opens by double-clicking on a machine that has
never seen the program before, which is the only reason to bother with an
Apple Developer account here.

The release workflow sets these from repository secrets and skips both
steps when they are absent; [docs/macos.md](../../docs/macos.md#signing-and-notarisation)
lists the secrets and how to make the certificate.

## Homebrew

`brew-cask.sh` writes the cask for one released version, taking the two
checksums from the release's `SHA256SUMS`:

```bash
packaging/macos/brew-cask.sh --version 0.9.5 \
    --arm-sha256 <sha> --intel-sha256 <sha> \
    --out Casks/rust-dicom-station.rb
```

One cask covers both architectures. The release workflow runs this and
commits the result to the tap named by the repository variable
`HOMEBREW_TAP`, when there is one.
