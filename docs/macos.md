# The macOS package

Rust DICOM Station on macOS is the same program it is on Windows and Linux:
the same viewer, the same modules, the same detachable tool windows, drawing
through Metal instead of Vulkan or Direct3D. Nothing was taken out and
nothing was rewritten for the platform - `eframe` opens an AppKit window,
`src/settings.rs` already knew where *Application Support* is, and
`src/gfx.rs` already knew Metal is the only graphics backend Apple has.

Every release attaches two disk images, one per architecture:

```text
rust-dicom-station-<version>-macos-arm64.dmg     Apple Silicon (M1 and newer)
rust-dicom-station-<version>-macos-x86_64.dmg    Intel
```

Both are built for **macOS 12 Monterey and newer**. They are made by
[macos/build-app.sh](../macos/build-app.sh) from the packaging in
[macos/](../macos/README.md), the way the Windows installer is built from
[installer/](../installer/README.md) and the APK from
[android/](../android/), and the workflow that runs it is
[.github/workflows/macos.yml](../.github/workflows/macos.yml).

## Installing

Download the image for the Mac in question - *Apple menu ▸ About This Mac*
says which it is, "Apple M…" or "Intel" - open it, and drag **Rust DICOM
Station** onto the *Applications* folder in the same window.

With [Homebrew](#homebrew) and the tap installed, one line does it on either
architecture:

```text
brew install --cask rust-dicom-station
```

### The first launch

Until the release is notarised (see [below](#signing-and-notarisation)),
macOS will not open the program by double-clicking: Gatekeeper has no
developer to check it against. The first launch has to go through Finder:

1. open *Applications*,
2. **right-click** (or Control-click) *Rust DICOM Station*,
3. choose *Open*, then *Open* again in the dialog.

macOS remembers the decision. Every later launch, and every later version
installed over it, opens normally.

A Homebrew installation carries the same quarantine flag, and `brew` offers
its own way past it:

```text
brew install --cask --no-quarantine rust-dicom-station
```

"The application is damaged and can't be opened" rather than the usual
warning means something else: an incomplete download, or an image built for
a newer macOS than the machine runs. The published images require macOS 12,
and the build refuses to finish if the binaries and the bundle disagree
about that ([macos/README.md](../macos/README.md#macos-12-and-newer)), so an
incomplete download is the first thing to rule out - the release's
`SHA256SUMS` has the hash to compare against.

## Where the files are

| | Path |
|---|---|
| The program | `/Applications/Rust DICOM Station.app` |
| Settings | `~/Library/Application Support/RustDICOMStation/viewer_settings.txt` |
| Model weights | `~/Library/Application Support/RustDICOMStation/models` |
| Patient archive | `~/Library/Application Support/RustDICOMStation/archive`, unless another folder was chosen |
| MCP configuration | `~/Library/Application Support/RustDICOMStation/mcp.toml` |
| The MCP server | `/Applications/Rust DICOM Station.app/Contents/MacOS/rds-mcp` |

`~/Library` is hidden in Finder; *Go ▸ Go to Folder…* (⇧⌘G) opens it by
name, and holding ⌥ while the *Go* menu is open lists it.

Models are not in the disk image. They are downloaded on first use into the
folder above, or into whatever *Model manager* points at, exactly as on the
other platforms.

## Graphics

Metal, and only Metal. Apple deprecated OpenGL and never shipped Vulkan, and
`wgpu` reaches neither on macOS without a translation layer this program
does not link - so *View ▸ Graphics backend* offers *Automatic* and *Metal*
there and nothing else, and the start-up fallback that saves a Windows
machine with a broken Vulkan driver has nothing to fall back to. This is not
a limitation in practice: every Mac that runs macOS 12 has a working Metal
driver.

## The MCP server

`rds-mcp` ships inside the bundle, beside the viewer, because
`settings::mcp_exe_path()` looks for it in the folder of the running
executable. Point an MCP client at it by full path:

```json
{
  "command": "/Applications/Rust DICOM Station.app/Contents/MacOS/rds-mcp"
}
```

Installed through Homebrew, plain `rds-mcp` works as well: the cask links
that same file into the Homebrew prefix. *Settings ▸ MCP server* in the
viewer shows the snippet for the copy that is actually running. Everything
else about the server - the tools, the PHI gate, `mcp.toml` - is in
[docs/mcp.md](mcp.md).

## Building it

Needs the Xcode command line tools (`xcode-select --install`) and the
target for the architecture wanted:

```bash
rustup target add aarch64-apple-darwin x86_64-apple-darwin

macos/build-app.sh                  # this Mac
macos/build-app.sh --arch x86_64    # Intel, from an Apple Silicon Mac
```

The result is `macos/out/rust-dicom-station-<version>-macos-<arch>.dmg`.
[macos/README.md](../macos/README.md) covers the options, what goes into
the bundle and why.

The workflow builds the images on every pull request into `main` as well as
during the release, so the packaging is exercised by a check rather than
first run while a release is being published; a pull request into any other
branch only cross-compiles the Intel target, which is the cheap half.

An Apple Silicon Mac builds both architectures; an Intel Mac builds only its
own. This is why the release builds both on an Apple Silicon runner and
cross-compiles the Intel half, rather than using GitHub's `macos-15-intel`
image - which is announced to be the last x86_64 macOS runner and to go away
in August 2027. Nothing about this repository's code differs between the two
architectures, so the Intel image is checked rather than run: architecture,
deployment target, signature and bundle contents, out of the finished disk
image.

## Signing and notarisation

Both are optional. Without them the release still happens; users get the
[first-launch](#the-first-launch) detour.

| Secret | What it is |
|---|---|
| `MACOS_CERTIFICATE_BASE64` | A *Developer ID Application* certificate and its private key, exported from Keychain Access as a `.p12`, then `base64 -i cert.p12 \| pbcopy` |
| `MACOS_CERTIFICATE_PASSWORD` | The password given when exporting it |
| `MACOS_SIGNING_IDENTITY` | The identity's full name, e.g. `Developer ID Application: Alexander Pryanichnikov (TEAMID)` |
| `MACOS_NOTARY_APPLE_ID` | The Apple ID of the developer account |
| `MACOS_NOTARY_PASSWORD` | An [app-specific password](https://support.apple.com/en-us/102654) for that Apple ID - not the account password |
| `MACOS_NOTARY_TEAM_ID` | The ten-character team identifier |

The first three make the workflow sign the bundle and the disk image with
the Developer ID under the hardened runtime. All six make it submit the
image to Apple, wait for the ticket and staple it in, after which the
program opens by double-clicking on a Mac that has never seen it, even
offline.

They are set under *Settings ▸ Secrets and variables ▸ Actions*. All of it
needs a paid Apple Developer account; there is no free path to
notarisation. The same variables work locally, spelled
`RDS_CODESIGN_IDENTITY` and `RDS_NOTARY_*`
([macos/README.md](../macos/README.md#signing-and-notarisation)).

Signing is also what makes **updating** clean: two versions signed with the
same Developer ID replace each other without comment, while an ad-hoc
signature changes on every build and macOS treats the new copy as a program
it has never seen.

## Homebrew

The cask is one file covering both architectures: Homebrew fills in the
architecture and the matching checksum, so the same command is right on
either Mac.

```text
brew install --cask rust-dicom-station
brew upgrade --cask rust-dicom-station
```

The `homebrew` job of the release workflow writes it with
[macos/brew-cask.sh](../macos/brew-cask.sh) - taking the two checksums from
the release's own `SHA256SUMS`, so the cask can never disagree with the
files it points at - attaches it to the release as
`rust-dicom-station.rb`, and commits it to a tap.

Setting the tap up, once:

1. Create a public repository named `homebrew-<something>` under the same
   account; `homebrew-tap` is the usual name. An empty repository is
   enough - the job creates `Casks/` itself.
2. Add the repository variable `HOMEBREW_TAP` with `owner/homebrew-tap` in
   it (*Settings ▸ Secrets and variables ▸ Actions ▸ Variables*).
3. Add the secret `HOMEBREW_TAP_TOKEN`: a personal access token that may
   push to that repository - a classic token with the `public_repo` scope,
   or a fine-grained one scoped to it with *Contents: read and write*.

Users then add the tap once:

```text
brew tap owner/tap
brew install --cask rust-dicom-station
```

Without the variable or the token the job says so and succeeds; the cask is
still on the release, and `brew install --cask ./rust-dicom-station.rb`
installs from the downloaded file. The cask is not submitted to
`homebrew-cask` itself: that repository asks for a notarised, reasonably
well-known application, which is a conversation to have after the Developer
ID is in place.

## Troubleshooting

**"Rust DICOM Station" cannot be opened because the developer cannot be
verified.** The expected first launch on an unnotarised build; use
right-click ▸ *Open* as [above](#the-first-launch).

**"The application is damaged and can't be opened. You should move it to
the Trash."** Not about damage. Either the download is incomplete - check
it against `SHA256SUMS` - or the copy came from somewhere that stripped its
signature. Re-download from the releases page.

**The window opens black, or the program quits at start.** Metal failed,
which on a Mac that runs macOS 12 is unusual. Start it from Terminal to see
what it says:

```text
"/Applications/Rust DICOM Station.app/Contents/MacOS/rust-dicom-station"
```

**A folder does not appear in the file dialog.** macOS asks for permission
the first time a program reads *Desktop*, *Documents*, *Downloads*, an
external volume or an iCloud folder, and a denied prompt is remembered.
*System Settings ▸ Privacy & Security ▸ Files and Folders* lists what was
allowed; *Full Disk Access* in the same place covers everything at once.

**`rds-mcp` is not found by the client.** Use the full path inside the
bundle, quoted - it has spaces in it - or install through Homebrew, which
links it into the prefix.
