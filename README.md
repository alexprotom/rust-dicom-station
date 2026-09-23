# rust-dicom-station

[![rust-dicom-station](https://snapcraft.io/rust-dicom-station/badge.svg)](https://snapcraft.io/rust-dicom-station) [![CI](https://github.com/alexprotom/rust-dicom-station/actions/workflows/ci.yml/badge.svg)](https://github.com/alexprotom/rust-dicom-station/actions/workflows/ci.yml) 

RDS (Rust DICOM Station) is open-source software for medical imaging and radiotherapy research, analysis, and QA, **written entirely in Rust**. It loads complete radiotherapy studies (CT, MR and PET series, RTSTRUCT, RTDOSE, photon and ion RTPLAN, DICOM SEG, planar images, spatial and deformable registrations, and treatment records) into an integrated environment for visualization, comparison and quantitative analysis. Beyond the classic linked MPR layout and multi-workspace comparison (up to four studies side by side), RDS provides image registration, structure propagation, DRR generation, dose-volume histograms, 4D motion analysis, interactive and AI-assisted segmentation, 3D visualization, and DICOM editing and export. The entire processing stack is native Rust: functionality normally provided through C/C++ or Python frameworks, including elastix- and plastimatch-style registration, ITK-style ray casting, TotalSegmentator, SegVol, and MedSAM2, is re-implemented directly in Rust without bindings to those frameworks.

![overview](docs/screenshot_overview.png)

*Two breathing phases of the bundled 4D-Lung patient as two rows of linked
MPR views with their RTSTRUCT contours, and the 3D window showing the RTSTRUCT
surfaces together with organs auto-segmented by the built-in TotalSegmentator
engine.*

## What it does

* **Viewing** - parallel DICOM loading (compressed syntaxes included), true
  patient-space geometry, linked axial / sagittal / coronal views in rows you
  lay out yourself (up to three panes each, the 3D scene among them), W/L
  presets, dose colorwash and isodose lines, per-beam plan summaries, planar
  images (DX / CR / RTIMAGE), dark and light themes. Folders *or* individual
  files, and **the data does not have to be a volume**: a portal image, a
  structure set or a plan opens on its own, in the ordinary tree, with
  everything that does not need voxels still working.
* **Workspaces** - up to four (A - D), one row of panes each; a patient ▶
  study ▶ series tree per workspace; copy / move / remove / rename at every
  level with the reference chains kept intact; RT structure sets and
  segmentation series as tree nodes, contours and masks converting as they
  move between them; the crosshair, the slice, the zoom and the players
  synced across every open workspace.
* **Image information** - what the displayed series actually is, read back
  out of its own headers: voxel spacing, slice thickness and the gap or
  overlap between slices, uneven slice positions, matrix and field of view,
  gantry tilt, frame of reference, kV / mAs / CTDIvol / kernel, rescale and
  units - with whatever wants a second look named and explained, and a side
  by side of what two workspaces disagree about before you register them.
* **Playback** - ▶ on every viewport: through the slices of a view, and
  through the phases of a 4D group. Playing a group carries the structure
  set, the segmentation series and the dose of each phase with it and walks
  the selection down the group in the tree; the phases are read into memory
  once (under a budget you set) so it runs as a cine rather than a
  slideshow, and the 3D window breathes with it.
* **Patient archive** - a local PACS on plain folders and text sidecars:
  file a study, list patients without opening a DICOM file, load into either
  workspace, and send the structures and segmentations you drew back as derived
  objects under the original Study and Frame of Reference UIDs.
* **Registration** - rigid and B-spline after **elastix** (pyramids,
  stochastic sampling, ASGD), dense B-spline after **plastimatch** (analytic
  gradient, bending energy, L-BFGS, mean squares or Mattes mutual
  information) and plastimatch's **landmark warp**; any of them restricted to
  one structure or refined on top of a previous result. Every run reports the
  Dice of the two images before and after it, per-structure Dice on request,
  6 DOF, displacement statistics, Jacobian determinant and folding; the vector
  field draws in the views and in 3D; fusion overlay; DICOM REG and Deformable
  Spatial Registration read and written; a known-transform simulator for QA.
  The 4 × 4 transform can also be typed in by hand, Slicer-style, and used in
  place of a recovered one - in the registration, in propagation and in
  transfer by relationship.
* **Structure propagation** - contours and segmentations carried through a
  registration by per-voxel pull-back (no holes, any two grids), optionally
  refined on an enclosing structure first.
* **4D / motion** - phases recognised into 4D groups; the reference phase
  registered to every other, targets propagated and their centroids tracked;
  peak-to-peak, drift, correlation with a reference structure, ITV
  generation, a results window with run-vs-run comparison and CSV export;
  structure comparison (Dice, HD95, surface distance) and transfer by
  relationship.
* **MCP server** - `rds-mcp`, a second executable that lets an AI assistant
  drive the station's tools (load, segment, register, propagate, 4D motion,
  DVH, export) headlessly over the Model Context Protocol, with a
  ready-made prompt for heart target propagation; workspaces that still name
  their patient are refused by default and no tool ever returns identifiers.
* **DRR** - plastimatch's exact Siddon tracer and ITK's interpolating
  ray-cast on one IEC cone-beam geometry, beam's-eye view from an RTPLAN
  beam, side by side with their difference.
* **Dose-volume histograms** - cumulative and differential DVHs of any
  structures against any dose, sampled on the structure's own lattice;
  `D95%` / `D2cc` / `V20Gy` metrics, protocol constraint checking, CSV
  export; verified against an analytic phantom.
* **Segmentation** - spacing-aware 2D / 3D brush and eraser, geodesic region
  growing, undo, live 3D surfaces, mask ⇄ RTSTRUCT, DICOM SEG import and
  export (binary and fractional). The Structure editor edits a segment as a
  whole the way it edits a structure: keep the largest piece, fill the holes,
  grow or shrink by millimetres, or trace it into contours and carry on.
* **Structure algebra** - union / intersection / subtraction / symmetric
  difference with margins in patient directions (exact ellipsoids), crop,
  ring, cleanup.
* **Body contour** - the EXTERNAL structure without the couch, the chair or
  the mask, on CT and MR, classically or guided by TotalSegmentator's body
  network.
* **Auto-segmentation** - TotalSegmentator v2 rebuilt natively (117
  structures): official nnU-Net weights converted without Python, a SIMD CPU
  engine or a wgpu GPU path (no CUDA), mean Dice 0.9995 against the
  reference.
* **Prompt segmentation** - SegVol rebuilt natively: box, click or free-text
  prompts ("liver", "tumor") for the structures no fixed-class model covers.
* **Slice propagation** - MedSAM2 (SAM 2.1 with its memory bank) rebuilt
  natively: box a structure on one slice, refine with include / exclude
  clicks, follow it through the stack at native resolution.
* **Tools** - DICOM export with an editable tag table, a model manager for
  every downloadable weight, a folder anonymizer with consistent UID
  regeneration, a synthetic RT-study generator; every tool window can be
  moved to its own monitor, and the structure tools live in the modules
  panel.

## Architecture

One language, one binary. All image processing runs on the CPU with `rayon`
and caching; the GPU (`wgpu`: DX12 / Vulkan / Metal) blits the UI and,
optionally, runs the networks. Long operations run on worker threads with
progress and cancellation. The module map, threading model, geometry
conventions and test suites are in
[docs/architecture.md](docs/architecture.md).

## Quick start

Requires a Rust toolchain (<https://rustup.rs>).

```
cargo build --release
cargo run --release -- data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS
cargo run --release -- data-test/TCIA_4D-LUNG/P102/4DFBCT+RTS data-test/TCIA_4D-LUNG/P102/4DCBCT
cargo test --release
```

To try prompt segmentation on the bundled patient: put the crosshair on the
tumor, unfold *💬 Prompt segmentation* in the Structure auto tools module
(right panel), prompt **Box**, **▶ Segment**.
The engines fetch their weights on first use into one model folder
(`%LOCALAPPDATA%\RustDICOMStation\models` on Windows,
`~/.local/share/RustDICOMStation/models` on Linux,
`~/Library/Application Support/RustDICOMStation/models` on macOS,
`~/snap/rust-dicom-station/common/data/models` in the snap,
`~/.var/app/io.github.alexprotom.rust-dicom-station/data/RustDICOMStation/models`
in the Flatpak), movable from any tool window; each engine also has a headless CLI in [examples/](examples/).

If the program will not start at all, it is almost certainly one thing: a
Windows machine advertising a Vulkan driver that cannot create a device. It
now falls back to Direct3D 12 by itself, the installer asks which backend to
use, and *View ▸ Graphics backend* changes it afterwards - see
[docs/viewer.md](docs/viewer.md#graphics-backend).

Windows, Linux, macOS, Android tablets, iPads and iPhones are supported; `--no-default-features` builds a
CPU-only viewer without the GPU inference backend. Every push to `main`
publishes a release: a Windows installer
(`rust-dicom-station-<version>-windows-x86_64.exe` - shortcuts, "Open with"
on folders, the VC++ runtime check, optional weight prefetch, uninstaller),
a Linux AppImage, two macOS disk images
(`rust-dicom-station-<version>-macos-arm64.dmg` and `-macos-x86_64.dmg`, both
for macOS 12 Monterey and newer, [docs/macos.md](docs/macos.md)) and an
Android APK (`rust-dicom-station-<version>-android-arm64.apk`,
the same viewer on a tablet, [docs/android.md](docs/android.md)) and an
iOS package (`rust-dicom-station-<version>-ios.ipa`, iPad and iPhone, iOS 15 and newer,
also to TestFlight when configured, [docs/ios.md](docs/ios.md)), and puts the snap into the Snap Store (`sudo snap
install rust-dicom-station`, [docs/snap.md](docs/snap.md)); the same program
is on Flathub as `io.github.alexprotom.rust-dicom-station`
([docs/flatpak.md](docs/flatpak.md)) and, with a tap configured, in Homebrew
as the cask `rust-dicom-station`. A newer
installer updates an existing installation in place (no second copy,
nothing to uninstall first), *Start ▸ Update Rust
DICOM Station* fetches the newest release, and the package is published to
winget as `RDS.RustDICOMStation` (`winget install` / `winget
upgrade`). The installer is its own crate in
[packaging/windows/installer/](packaging/windows/installer/README.md). No data at hand? *Tools ▶ 📐 Generate test
data* writes a complete synthetic RT study, `data-test/` ships a real
patient - a ten-phase 4DFBCT with an RT Structure Set per phase and the
matching ten-phase 4DCBCT ([docs/example-data.md](docs/example-data.md)) -
and *Tools ▶ 📥 Download test data* fetches that folder from GitHub into an
installed copy.

## Documentation

https://alexprotom.github.io/rust-dicom-station/

| | |
|---|---|
| [docs/viewer.md](docs/viewer.md) | Loading folders and single files, workspaces with no volume, MPR views, workspace tree, the four workspaces and comparing them, interaction reference, the graphics backend |
| [docs/rt-objects.md](docs/rt-objects.md) | RTSTRUCT, RTDOSE, RTPLAN, REG, RTRECORD, reference chains |
| [docs/registration.md](docs/registration.md) | The four registration engines, local registration, analytics, vector fields, fusion, simulator, verification |
| [docs/propagation.md](docs/propagation.md) | Carrying contours and segmentations across a registration |
| [docs/motion-4d.md](docs/motion-4d.md) | 4D groups, the motion / ITV workflow, results, structure comparison and transfer |
| [docs/drr.md](docs/drr.md) | Digitally reconstructed radiographs: the two projectors and the geometry |
| [docs/dvh.md](docs/dvh.md) | Dose-volume histograms: curves, metrics, constraint checking, export |
| [docs/segmentation.md](docs/segmentation.md) | Brush / eraser / region growing, 3D view, mask → RTSTRUCT |
| [docs/contours.md](docs/contours.md) | Drawing and editing structures as contours: the draw row and the Structure editor, live wire, smart brush, interpolation, POIs, templates, locking |
| [docs/generators.md](docs/generators.md) | Structures without drawing: grey level (HU or SUV), shapes, isodose, field of view |
| [docs/structure-algebra.md](docs/structure-algebra.md) | Boolean operations, margins, cropping, cleanup |
| [docs/body-contour.md](docs/body-contour.md) | The body / EXTERNAL contour on CT and MR, verification |
| [docs/auto-segmentation.md](docs/auto-segmentation.md) | The pure-Rust TotalSegmentator: models, pipeline, engines, validation, classes, licensing |
| [docs/segvol.md](docs/segvol.md) | Prompt-driven segmentation: the SegVol re-implementation |
| [docs/medsam2.md](docs/medsam2.md) | Propagating a prompt through a stack: the MedSAM2 re-implementation |
| [docs/pacs.md](docs/pacs.md) | The local patient archive: window, on-disk layout, filing, loading, sending changes back |
| [docs/export-and-tools.md](docs/export-and-tools.md) | DICOM export, the model manager, anonymizer, test-data generator and download |
| [docs/mcp.md](docs/mcp.md) | The MCP server: tools, the heart workflow prompt, patient-identity safety, configuration |
| [docs/architecture.md](docs/architecture.md) | Design, functional overview, module map, threading, the model folder, conventions, testing |
| [docs/release-versioning.md](docs/release-versioning.md) | How versions and releases are produced |
| [docs/snap.md](docs/snap.md) | The Linux snap: confinement, where its files are, the MCP server in it, building and publishing |
| [docs/flatpak.md](docs/flatpak.md) | The Flatpak: the sandbox, where its files are, the MCP server in it, building and submitting to Flathub |
| [docs/macos.md](docs/macos.md) | The macOS package: the two disk images, the first launch, Metal, where its files are, building, signing, notarisation, Homebrew |
| [docs/android.md](docs/android.md) | The Android package: installing, all files access, what differs on a tablet, where its files are, building and signing |
| [docs/ios.md](docs/ios.md) | The iOS package for iPad and iPhone: installing (TestFlight, ad hoc, sideloading), getting studies onto the device, what differs on a tablet and a phone, where its files are, building, signing and releasing |
| [docs/example-data.md](docs/example-data.md) | Bundled patient data, source and citations |
| [packaging/README.md](packaging/README.md) | The packaging folder: one subfolder per platform, what each builds and where |
| [packaging/windows/installer/README.md](packaging/windows/installer/README.md) | The Windows installer: building it, what it installs, updating, winget, silent switches |

## License and citations

The code is MIT-licensed, so commercial use is permitted. The MIT License
covers this project's own code; the third-party Rust libraries RDS depends on
keep their own licences, reproduced in
[THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt). If you publish work
produced with RDS, a citation is appreciated: see
[CITATION.cff](CITATION.cff).

The bundled example data is TCIA **4D-Lung**
patient P102, redistributed under CC BY 3.0 (cite it as described in
[docs/example-data.md](docs/example-data.md)). Auto-segmentation uses
TotalSegmentator's Apache-2.0 "total"-task weights (cite Wasserthal et al.
(Radiology AI 2023) and nnU-Net (Isensee et al., Nature Methods 2021) as
described in [docs/auto-segmentation.md](docs/auto-segmentation.md)). Prompt
segmentation re-implements SegVol (Du et al., NeurIPS 2024) and slice
propagation MedSAM2 (Ma et al., 2025); their weights are only ever
downloaded from Hugging Face to your own machine at your request and are
never redistributed; see [docs/segvol.md](docs/segvol.md) and
[docs/medsam2.md](docs/medsam2.md).

This software is a viewer for research and QA convenience. **Not a medical
device, neither CE-marked nor FDA-cleared, and not for clinical
decision-making.** The ADDITIONAL NOTICE in [LICENSE.txt](LICENSE.txt) states
this in full. It is a statement of fact about the software, not a condition of
the MIT License, which permits commercial use.
