# rust-dicom-station

RDS (Rust DICOM Station) is a fast, robust DICOM / RT DICOM viewer written **entirely in Rust**. It
loads a full radiotherapy study: image series (CT/MRI/PET), RT Structure
Set, RT Dose, RT Plan (photon and ion/proton), planar images, spatial
registrations, treatment records; and displays it in the classic
three-view layout, with a second dataset row for comparison, built-in
elastix-style **image registration**, **interactive
segmentation**, a live **3D structure view**, and **automatic multi-organ
segmentation** (a pure-Rust re-implementation of TotalSegmentator, 117
structures, CPU or any GPU).

![overview](docs/screenshot_overview.png)

*One session, the bundled 4D-Lung patient: datasets A and B are two
breathing phases of a 4DCT shown as two rows of linked MPR views with
their phase-specific RTSTRUCT contours; the crosshair sits in the tumor.
The floating window is the live 3D view of dataset A — RTSTRUCT surfaces
(lungs, heart, tumor, cord) together with organs auto-segmented by the
built-in TotalSegmentator engine, which also fills the Segmentations list
in the sidebar (aorta, trachea, liver, stomach, spleen, kidneys — with
volumes, editable as masks, convertible to RTSTRUCT). The sidebar also
holds the registration controls and both dataset trees.*

## Highlights

* **Viewing** - parallel DICOM loading (incl. compressed syntaxes), true
  patient-space geometry, axial/sagittal/coronal with linked crosshairs,
  window/level with CT presets, dose colorwash + isodose lines, per-beam
  plan summaries, planar images (DX/CR/RTIMAGE), dark/light themes.
* **Datasets** - a patient ▶ study ▶ series tree per dataset, folder
  merging, copy/move/remove with correct reference-chain semantics,
  six-view comparison mode with patient-space crosshair linking.
* **Registration** - rigid (6-DOF) and deformable (cubic B-spline)
  registration re-implemented from elastix (multi-resolution pyramids,
  stochastic sampling, ASGD optimizer), magenta/green fusion overlay,
  DICOM REG support, a known-transform simulator for QA, sub-millimeter
  verified accuracy.
* **Segmentation** - spacing-aware 2D/3D brush and eraser, geodesic
  region growing with live preview, per-stroke undo, real-time 3D surface
  view, mask → RTSTRUCT conversion.
* **Auto-segmentation** - TotalSegmentator v2 inference rebuilt natively:
  official nnU-Net weights downloaded once and converted
  without Python, hand-written SIMD CPU engine and a wgpu GPU path
  (Vulkan/DX12/Metal, no CUDA), validated to mean Dice 0.9995 against the
  reference implementation.
* **Tools** - DICOM export with an editable patient/study tag table
  (CT + RTSTRUCT + RTDOSE + RTPLAN), an interactive folder anonymizer with
  consistent UID regeneration, and a synthetic RT-study generator; 40+ tests
  assert the whole stack against an analytically known phantom.

## Architecture

One language, one binary. Every algorithm - DICOM parsing, volume
reconstruction, rendering primitives, registration, meshing, neural-net
inference, DICOM writing - is implemented in Rust; where a feature
usually means binding a C/C++ library (ITK/elastix, ONNX Runtime, CUDA),
it is re-implemented natively instead. Image processing runs CPU-side
with `rayon` and aggressive caching; the GPU (via `wgpu`) only blits the
UI and, optionally, runs the segmentation network. Long operations run on
background threads with progress and cancellation. The full module map,
threading model, geometry conventions and performance numbers are in
[docs/architecture.md](docs/architecture.md).

## Quick start

Requires a Rust toolchain (<https://rustup.rs>).

```
cargo build --release

# open a study, or two studies straight into comparison mode:
cargo run --release -- example_data/lung_p1_4DCT_phase_000
cargo run --release -- example_data/lung_p1_4DCT_phase_000 example_data/lung_p1_4DCT_phase_050

cargo test --release
```

Windows, Linux and macOS are supported; rendering uses `wgpu`
(DX12/Vulkan/Metal). `--no-default-features` builds a CPU-only viewer
without the GPU inference backend.

On Windows there is also a proper installer — a single
`rust-dicom-station-setup.exe` with shortcuts, an "Open with" entry on
folders, the Visual C++ runtime check, an optional pre-download of the
auto-segmentation weights, and a clean uninstall. It is a separate Rust
program in [installer/](installer/README.md) and is *not* built by
`cargo build --release`; see its README for the three build steps. No data at hand? *File ▶ 🧪 Generate
test data…* creates a complete synthetic RT study, and `example_data/`
ships a real two-phase 4DCT (see
[docs/example-data.md](docs/example-data.md)).

## Documentation

| | |
|---|---|
| [docs/viewer.md](docs/viewer.md) | Loading, MPR views, dataset tree, comparison mode, interaction reference |
| [docs/rt-objects.md](docs/rt-objects.md) | RTSTRUCT, RTDOSE, RTPLAN, REG, RTRECORD, reference chains |
| [docs/registration.md](docs/registration.md) | Rigid + B-spline registration, fusion, simulator, verification |
| [docs/segmentation.md](docs/segmentation.md) | Brush / eraser / region growing, 3D view, mask → RTSTRUCT |
| [docs/segvol.md](docs/segvol.md) | Prompt-driven segmentation: box / point / text, the SegVol re-implementation |
| [docs/auto-segmentation.md](docs/auto-segmentation.md) | The pure-Rust TotalSegmentator: models, pipeline, engines, validation, classes, licensing |
| [docs/export-and-tools.md](docs/export-and-tools.md) | DICOM export, anonymizer, test-data generator |
| [docs/architecture.md](docs/architecture.md) | Design, module map, threading, conventions, testing |
| [docs/example-data.md](docs/example-data.md) | Bundled patient data, source and citations |
| [installer/README.md](installer/README.md) | The Windows installer: building it, what it installs, silent switches |

## License and citations

The code is MIT-licensed. The bundled example data is TCIA **4D-Lung**
patient P102, redistributed under CC BY 3.0 — cite it as described in
[docs/example-data.md](docs/example-data.md). The auto-segmentation uses
TotalSegmentator's openly licensed (Apache-2.0) "total"-task weights —
cite Wasserthal et al. (Radiology AI 2023) and nnU-Net (Isensee et al.,
Nature Methods 2021) as described in
[docs/auto-segmentation.md](docs/auto-segmentation.md).

This software is a viewer for research and QA convenience — **not a
medical device, and not for clinical decision-making.**
