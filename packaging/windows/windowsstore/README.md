# Microsoft Store

Prepared, not built: nothing is submitted to the Store yet, and no file in
this folder is used by a workflow. It is the place for that work when it
starts, so that it lands beside the [installer](../installer/README.md) and
the [winget](../winget/README.md) manifests rather than in a folder of its
own at the top of the repository.

Two routes exist for a desktop program like this one, and the choice has
not been made:

1. **Win32 installer submission.** Partner Center accepts an `.exe`
   installer hosted at a public URL, with its silent switches and its
   uninstall registration, and lists it as an "app provided and updated by
   the publisher". Everything it asks for already exists: the release
   installer on GitHub (`rust-dicom-station-X.Y.Z-windows-x86_64.exe`), its
   silent mode (`--silent`, or `--passive` for a progress window), the
   per-user and machine-wide scopes (`--just-me` / `--all-users`), the
   Add/Remove Programs entry, and the exit codes the installer README
   lists. This route needs no packaging at all; it needs a Partner Center
   account and a submission per release, which could be scripted through
   the Store submission API later.

2. **MSIX package.** A `.msix` built from the two release executables
   (`rust-dicom-station.exe`, `rds-mcp.exe`), signed with the Store's
   certificate at ingestion, updated by the Store itself. This is the
   route that would put files here: an `AppxManifest.xml` (identity,
   publisher, the `.dcm` file association, the Start menu entry), the icon
   in the Store's tile sizes generated from `assets/rust-dicom-station.png`,
   and a `build-msix.ps1` that runs `makeappx pack` and `signtool` over a
   layout folder - plus a job in `release.yml` that attaches the package to
   the release. Two things need a decision first: whether the MSIX runs the
   viewer with full trust (it needs the models folder and any DICOM folder
   the user names, which an MSIX gets through `runFullTrust`), and how the
   MCP server is reached from a Store install (the package's install
   folder is read-only and its path carries the version, so
   *Settings > MCP server > Copy client configuration* would have to write
   the alias `rust-dicom-station.rds-mcp`, the way the snap does).

Either way the identity facts are in one place already:
`packaging/windows/installer/src/product.rs` (`APP_NAME`, `PUBLISHER`,
`PRODUCT_ID`, the release repository) - a Store manifest must use the same
names, so read them from there rather than typing them again.
