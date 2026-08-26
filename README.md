# rust-dicom-station

[![CI](https://github.com/alexprotom/rust-dicom-station/actions/workflows/ci.yml/badge.svg)](https://github.com/alexprotom/rust-dicom-station/actions/workflows/ci.yml)

RDS (Rust DICOM Station) is a fast, robust DICOM / RT DICOM viewer written **entirely in Rust**. It
loads a full radiotherapy study: image series (CT/MRI/PET), RT Structure
Set, RT Dose, RT Plan (photon and ion/proton), planar images, spatial
registrations, treatment records; and displays it in the classic
three-view layout, with a second dataset row for comparison, built-in
**image registration** (elastix- and plastimatch-style, rigid, deformable
and landmark-based, with analytics and a deformation vector field),
**structure propagation**, **DRR generation**, **interactive
segmentation**, a live **3D structure view**, **automatic multi-organ
segmentation** (a pure-Rust re-implementation of TotalSegmentator, 117
structures, CPU or any GPU), **prompt-driven segmentation** (a
pure-Rust re-implementation of SegVol — point at anything with a box, a
click or a structure name, and get an editable mask back), and **slice
propagation** (a pure-Rust re-implementation of MedSAM2 — mark a structure
on one slice and follow it through the whole stack).

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
  RT structure sets and segmentation series as tree nodes (create one,
  connect it to another image series, move whole series or single
  structures / segments between any two of them - contours and masks
  converting as they cross), renaming at every level from patient down to a
  single segment, six-view comparison mode with patient-space crosshair
  linking.
* **Registration** - four engines, none of them a binding: rigid (6-DOF)
  and deformable (cubic B-spline) re-implemented from **elastix**
  (multi-resolution pyramids, stochastic sampling, ASGD); a dense
  **plastimatch** B-spline with the exact analytic gradient, a
  bending-energy regularizer, L-BFGS and a choice of mean squares or
  **Mattes mutual information** for CT-MR; and plastimatch's **landmark
  warp** (thin-plate spline, Gaussian, Wendland). Any of them can be
  restricted to a single structure, or refined on top of an existing
  result - a local deformation is provably zero outside its region.
  Every run reports its **six degrees of freedom**, displacement
  statistics, Jacobian determinant and folding, and per-structure
  displacement; the **deformation vector field** draws as arrows or a
  deformed grid in all views and as glyphs in 3D, where both datasets can
  stand in one scene with independent opacity. Magenta/green fusion
  overlay, DICOM REG *and* Deformable Spatial Registration read and
  written, a known-transform simulator for QA, sub-millimeter verified
  accuracy.
* **Structure propagation** - contours and segmentations carried between
  datasets through any registration, pulled back per destination voxel
  (no holes, sub-voxel boundaries, any two grids), with an optional local
  refinement on the enclosing structure first - which is what makes a
  small structure inside a larger one land where it belongs. Results
  arrive as ordinary editable segmentations, convertible to RTSTRUCT.
* **DRR generation** - two independent forward projectors on one IEC
  cone-beam geometry (beam's-eye view straight from an RTPLAN beam):
  plastimatch's **exact Siddon** voxel-intersection ray tracing, and
  ITK's **interpolating ray-cast**. Side by side with a signed difference
  image and its statistics - the honest measure of what either one costs
  you.
* **Segmentation** - spacing-aware 2D/3D brush and eraser, geodesic
  region growing with live preview, per-stroke undo, real-time 3D surface
  view, mask → RTSTRUCT conversion, and **DICOM SEG** import and export
  (binary and fractional multi-frame masks, read onto their own lattice
  and resampled onto whichever image series they belong to).
* **Auto-segmentation** - TotalSegmentator v2 inference rebuilt natively:
  official nnU-Net weights downloaded once and converted
  without Python, hand-written SIMD CPU engine and a wgpu GPU path
  (Vulkan/DX12/Metal, no CUDA), validated to mean Dice 0.9995 against the
  reference implementation.
* **Prompt segmentation** - SegVol (NeurIPS 2024) rebuilt natively: a
  181 M-parameter 3-D ViT with a SAM-style prompt encoder and mask
  decoder plus a CLIP text tower, prompted with a **box**, a **click**,
  or **free text** ("liver", "tumor"…) — for the structures no
  fixed-class model can cover: lesions, targets, post-surgical cavities.
  Two-pass zoom-out / zoom-in inference, the image encoder on the same
  no-CUDA wgpu GPU path, results landing as ordinary editable
  segmentations, convertible to RTSTRUCT.
* **Slice propagation** - MedSAM2 (2025) rebuilt natively: SAM 2.1 with its
  memory bank, so a structure boxed on **one** slice is followed through the
  rest of the stack at the slice's own resolution - no in-plane resampling at
  all on 512x512 CT. The box is **drawn in the image and stays there** with
  handles to resize and move it; the prompted slice previews on its own and
  takes include / exclude clicks until it is right, and a slice that drifts is
  corrected by boxing it again and re-running into the same segmentation.
  Validated against the reference implementation module by module and over a
  full propagation.
* **Tools** - DICOM export with an editable patient/study tag table
  (CT + RTSTRUCT + SEG + RTDOSE + RTPLAN), a **model manager** showing every
  downloadable network weight with its state and size and the buttons to
  download, update, remove or free one or all of them, an interactive
  folder anonymizer with consistent UID regeneration, and a synthetic
  RT-study generator; 280+ tests across nine integration suites assert the
  whole stack against an analytically known phantom, on Linux and Windows
  in CI.

## Architecture

One language, one binary. Every algorithm - DICOM parsing, volume
reconstruction, rendering primitives, registration, meshing, neural-net
inference, DICOM writing - is implemented in Rust; where a feature
usually means binding a C/C++ library (ITK/elastix, ONNX Runtime, CUDA),
it is re-implemented natively instead - elastix and plastimatch
registration, ITK forward projection, TotalSegmentator, SegVol and MedSAM2
inference all included. Image processing runs CPU-side
with `rayon` and aggressive caching; the GPU (via `wgpu`) only blits the
UI and, optionally, runs the segmentation networks. Long operations run on
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

To try prompt segmentation on the bundled patient: put the crosshair on
the tumor, then *Tools ▶ 🧠 Prompt-segment dataset A…*, prompt **Box**,
**▶ Segment**. All three segmentation engines fetch their weights on first
use into one folder, `models/` next to the executable (one sub-folder per
engine, movable from any of the tool windows); all three also have
headless CLIs in [examples/](examples/).

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
| [docs/registration.md](docs/registration.md) | The four registration engines, local registration, analytics, vector fields, fusion, simulator, verification |
| [docs/propagation.md](docs/propagation.md) | Carrying contours and segmentations across a registration |
| [docs/drr.md](docs/drr.md) | Digitally reconstructed radiographs: the two projectors and the geometry |
| [docs/segmentation.md](docs/segmentation.md) | Brush / eraser / region growing, 3D view, mask → RTSTRUCT |
| [docs/segvol.md](docs/segvol.md) | Prompt-driven segmentation: box / point / text, the SegVol re-implementation |
| [docs/medsam2.md](docs/medsam2.md) | Propagating a prompt through a stack: the MedSAM2 re-implementation |
| [docs/auto-segmentation.md](docs/auto-segmentation.md) | The pure-Rust TotalSegmentator: models, pipeline, engines, validation, classes, licensing |
| [docs/export-and-tools.md](docs/export-and-tools.md) | DICOM export, the model manager, anonymizer, test-data generator |
| [docs/architecture.md](docs/architecture.md) | Design, functional overview, module map, threading, the model folder, conventions, testing |
| [docs/example-data.md](docs/example-data.md) | Bundled patient data, source and citations |
| [installer/README.md](installer/README.md) | The Windows installer: building it, what it installs, silent switches |

## License and citations

The code is MIT-licensed. The bundled example data is TCIA **4D-Lung**
patient P102, redistributed under CC BY 3.0 — cite it as described in
[docs/example-data.md](docs/example-data.md). The auto-segmentation uses
TotalSegmentator's openly licensed (Apache-2.0) "total"-task weights —
cite Wasserthal et al. (Radiology AI 2023) and nnU-Net (Isensee et al.,
Nature Methods 2021) as described in
[docs/auto-segmentation.md](docs/auto-segmentation.md). Prompt
segmentation re-implements SegVol (Du et al., NeurIPS 2024); its weights
carry **no license declaration**, so they are only ever downloaded from
Hugging Face to your own machine at your request and are never
redistributed — see [docs/segvol.md](docs/segvol.md).

This software is a viewer for research and QA convenience — **not a
medical device, and not for clinical decision-making.**
