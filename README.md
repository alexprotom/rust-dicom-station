# rust-dicom-viewer

A fast, robust DICOM / RT DICOM viewer written entirely in Rust. It loads a
full radiotherapy study — image series (CT/MR/PT), RT Structure Set, RT Dose
and RT Plan (photon and ion/proton) — and displays it in the classic
three-view MPR layout: **axial, sagittal and coronal side by side** with
linked crosshairs.

![screenshot](docs/screenshot.png)

## Features

**Image volumes.** A directory scan classifies every DICOM file (parallel
header-only pass), groups image series, and reconstructs a 3D volume from the
largest series by default (switchable in the toolbar). Slices are decoded in
parallel with `rayon`, sorted along the true slice normal, checked for
uniform spacing and consistent dimensions, and rescaled to HU. Compressed
transfer syntaxes (JPEG lossless, RLE, …) are handled by `dicom-rs`'s pure
Rust decoders.

**RTSTRUCT.** ROI names, display colors, interpreted types (PTV, ORGAN,
EXTERNAL, …) and all planar contours. Axial views draw the native closed
contours; sagittal/coronal views show the reconstructed cross-section
silhouette. Per-ROI visibility toggles.

**RTDOSE.** 16/32-bit dose grids with `DoseGridScaling`,
`GridFrameOffsetVector` (uniform or not, ascending or descending), trilinear
patient-space sampling, translucent colorwash and marching-squares isodose
lines at configurable percentages of a reference dose (prescription dose is
picked up from the plan automatically). Multiple dose files (plan/beam doses)
are selectable.

**RTPLAN.** Photon (`BeamSequence`) and ion/proton (`IonBeamSequence`) plans:
prescription, fractionation, and a per-beam table with radiation type, scan
mode, gantry/couch angles, energy range, meterset and control-point count.
Beam isocenters are marked in all three views.

**Interaction** (shown in the status bar):

| Input | Action |
|---|---|
| Left click / drag | Move the linked crosshair (all views follow) |
| Mouse wheel | Scroll through slices |
| Ctrl + wheel / pinch | Zoom (anchored at the cursor) |
| Middle drag | Pan |
| Right drag | Window/level (x = width, y = center) |
| Double click | Reset zoom & pan |

Window/level presets (soft tissue, lung, bone, brain, full range), anatomical
edge labels (L/R/A/P/S/I) derived from the actual patient orientation, and a
status bar with patient coordinates, voxel indices, HU and dose (Gy and % of
reference) at the crosshair.

## Performance

Everything is CPU-side Rust + GPU texture blitting via `wgpu`; the hot paths
are cached and only recomputed when their inputs change (slice, W/L, dose
settings, ROI visibility). Measured on the bundled synthetic study: full
study load ≈ 40 ms, orthogonal slice extraction ≈ 6 µs, dose-plane resampling
≈ 0.3 ms. Loading runs on a background thread with progress display, and
per-slice decode is parallelized across cores. Volumes are stored as `i16`
(HU), dose grids as `f32`.

## Building

Requires a Rust toolchain (<https://rustup.rs>). Then:

```
cargo build --release
```

Run with an optional directory argument to load a study immediately:

```
cargo run --release -- "D:\path\to\dicom\study"
```

or start it empty and use *Open folder…*. Windows, Linux and macOS are
supported; rendering uses `wgpu` (DX12/Vulkan/Metal, with fallbacks).

## Tests & synthetic data

`tools/generate_test_data.py` (needs Python + `pydicom`, only for test-data
generation — the viewer itself is pure Rust) writes a synthetic study to
`test_data/`: a cylindrical water phantom with a spherical target, three
ROIs, a 3D Gaussian 60 Gy dose and a two-beam proton plan with analytically
known values. The integration tests verify geometry round-trips, HU values,
contour radii, trilinear dose values, isodose radii and plan fields against
the closed-form expectations:

```
python3 tools/generate_test_data.py
cargo test --release
```

## Structure

```
src/
  main.rs      entry point (eframe/wgpu window)
  app.rs       egui application: three-view layout, panels, interaction
  loader.rs    directory scan, classification, parallel volume loading
  volume.rs    3D volume, patient-space geometry, orthogonal slice extraction
  rtstruct.rs  RT Structure Set parsing
  rtdose.rs    RT Dose parsing + trilinear patient-space sampling
  rtplan.rs    RT Plan / RT Ion Plan parsing
  render.rs    window/level, dose colorwash, marching-squares isodose,
               contour/plane intersection
  geometry.rs  minimal 3D vector math
```

## Notes & limitations

The three views are extracted in the acquisition index space, which maps
directly onto axial/sagittal/coronal planes for standard axial acquisitions;
oblique acquisitions display consistently but the plane names are nominal
(edge labels always reflect the true patient directions). Enhanced
multi-frame image series are not yet supported (classic single-frame series
only). Non-uniform slice spacing is detected and reported as a warning, with
the median spacing used for display. This software is a viewer for research
and QA convenience — not a medical device, and not for clinical
decision-making.
