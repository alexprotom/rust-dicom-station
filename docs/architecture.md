# Architecture

## Design philosophy

**One language.** Everything is Rust — DICOM parsing, image
reconstruction, rendering primitives, registration, meshing, neural-net
inference, DICOM writing. Where a capability normally means binding a
C/C++ library (elastix, ITK, ONNX Runtime, CUDA), the algorithms are
re-implemented natively instead. The only system interface is the GPU,
reached twice through `wgpu` (Vulkan / DX12 / Metal): once by `eframe` to
blit the UI, once (optionally) by `burn` for auto-segmentation inference —
no vendor SDKs either way.

**CPU-side algorithms, GPU-side pixels.** All image processing runs on
the CPU with `rayon` data parallelism and aggressive caching; the GPU
receives finished textures. This keeps every algorithm debuggable,
deterministic and portable, and turns out to be fast enough: full study
load ≈ 40 ms, orthogonal slice extraction ≈ 6 µs, dose-plane resampling
≈ 0.3 ms (measured on the synthetic study).

**Long work never blocks the UI.** Anything that can take more than a
frame — loading, registration, meshing, simulation, export, anonymization,
auto-segmentation — runs on a worker thread and reports through a shared
progress handle (see below).

## Module map

```
src/
  main.rs          entry point (eframe/wgpu window)
  lib.rs           library root — every module is public, so the
                   integration tests drive the same code as the GUI
  app/             egui application, split by concern; every submodule is a
                   further `impl ViewerApp` block, so the struct and all its
                   state stay in one place while the behaviour is grouped:
    mod.rs           ViewerApp and every type it holds, construction,
                     the job plumbing, and the per-frame driver
    theme.rs         theme-dependent colors
    chrome.rs        menu bar, toolbar, status bar
    panels.rs        side panel and its per-dataset sections
    views.rs         central MPR viewports, interaction, texture caches
    d3.rs            live 3D structure window
    planar.rs        floating DX/CR/RTIMAGE viewers
    dialogs.rs       auto-segmentation, generator, anonymizer, export
    jobs.rs          the operations that spawn a background job
    tree.rs          dataset-tree copy / move / remove
    seg.rs           interactive segmentation state machine
  loader.rs        directory scan, classification, parallel volume
                   loading, dataset merging
  volume.rs        3D volume, patient-space geometry, slice extraction
  geometry.rs      minimal 3D vector math (Vec3, f64, patient mm)
  render.rs        window/level, dose colorwash, marching-squares
                   isodose, contour/plane intersection
  rtstruct.rs      RT Structure Set parsing
  rtdose.rs        RT Dose parsing + trilinear patient-space sampling
  rtplan.rs        RT Plan / RT Ion Plan parsing
  extras.rs        DX/CR/RTIMAGE planar images, REG, RTRECORD
  registration.rs  elastix-style rigid + B-spline registration
                   (ASGD, pyramids, stochastic sampling)
  segmentation.rs  voxel masks: brush, geodesic grow, undo, overlays,
                   mask ▶ RTSTRUCT contours
  mesh3d.rs        contour/mask ▶ surface meshes (scanline fill,
                   surface nets, Laplacian smoothing)
  simulate.rs      known-transform study generator (registration QA)
  dicom_export.rs  DICOM writer (CT series, RTSTRUCT, RTDOSE, RTPLAN)
  gen_test_data.rs synthetic RT phantom study generator
  anonymize.rs     interactive DICOM anonymizer engine
  settings.rs      persisted preferences (plain text file)
  nn/              shared neural-network infrastructure — nothing in here
                   knows about a particular architecture
    pickle.rs        native PyTorch checkpoint (.pth) reader
    cache.rs         download, safetensors weight cache, progress/cancel
    half.rs          binary16 <-> binary32 conversion
    tensor.rs        Mat [rows, cols] and Act [c,d,h,w]; transposed conv
    linalg.rs        gemm-backed linear/matmul, layer norm, softmax,
                     GELU / ReLU / QuickGELU
    attention.rs     multi-head attention, optionally causally masked
  segvol/          promptable segmentation (pure-Rust SegVol) — box, point
                   and text prompts, for the structures a fixed-class model
                   cannot cover
    weights.rs       checkpoint acquisition and licensing notes
    layout.rs        the published checkpoint's tensor layout and the
                     checks that verify a file really is that model
    config.rs        the network's fixed dimensions
    params.rs        shape-checked access to the checkpoint's tensors
    vit.rs           image encoder (MONAI 3-D ViT, 12 blocks, 2048 tokens)
    prompt.rs        prompt encoder: box / point / text -> sparse + dense
    decoder.rs       two-way transformer, upscaling, mask hypernetworks
    net.rs           assembly and the single-window forward pass
    preprocess.rs    foreground normalization, canonical orientation,
                     nearest-exact / trilinear resampling, mask back-mapping
    infer.rs         zoom-out / zoom-in orchestration, MONAI window layout
    bpe.rs           CLIP byte-pair tokenizer
    clip.rs          CLIP text tower + dim_align, with a prompt cache
    gpu.rs           image encoder on wgpu via burn (cargo feature `gpu`)
  autoseg/         automatic segmentation (pure-Rust TotalSegmentator)
    mod.rs           public API, engine selection, progress/cancel
    classes.rs       117-class table, sub-model maps, organ colors
    config.rs        nnU-Net plans.json parsing
    weights.rs       which models exist and where they are published
    cpu.rs           CPU tensor engine (im2col + SIMD GEMM conv3d)
    net.rs           PlainConvUNet assembly + CPU forward
    gpu.rs           wgpu forward via burn (cargo feature `gpu`)
    preprocess.rs    canonical reorientation, resampling, back-mapping
    infer.rs         Gaussian sliding window, streaming argmax
tests/             integration suites (see Testing below)
examples/          autoseg_cli, autoseg_probe (headless dev tools)
```

## UI architecture

`ViewerApp` is defined in `app/mod.rs` together with every type it holds;
the sibling modules only add `impl ViewerApp` blocks. Keeping the
definitions in the parent module is what lets each child reach the struct's
private fields without widening any visibility beyond `pub(super)`.

`ViewerApp` owns two `StudySlot`s (datasets A and B). Each slot holds the
loaded study (series list, volume, structure sets, doses, plans, planar
images, registrations, records), three `ViewState`s (per-plane slice,
zoom/pan, and all texture caches), the crosshair, per-ROI visibility, and
the segmentation masks. Global state covers window/level, dose display
settings, tool selection, the registration result, and the theme.

Rendering is cache-driven: each view keeps keyed textures for the
grayscale slice, dose colorwash, contour polylines, segmentation overlay
and fusion blend, rebuilt only when their inputs change (slice, W/L, dose
settings, ROI visibility, mask edits, registration). Invalidation uses
small generation counters bumped by the owning mutation sites. Repaints
are demand-driven; while background jobs run, the UI polls at 10 Hz.

## Background jobs

One pattern serves every long operation:

```rust
struct Job<T, P = Progress> { progress: Arc<P>, rx: mpsc::Receiver<T> }
```

The UI snapshots the inputs, spawns a `std::thread`, and polls the channel
each frame (`poll_job`): a received value lands the result, a disconnect
means the worker died and surfaces as an error. Progress handles carry a
message (and for cancellable jobs — registration, auto-segmentation — an
atomic cancel flag plus a fraction for progress bars). Workers use `rayon`
internally for data parallelism; the thread-per-job is only the container.

Results are validated on landing where the underlying data could have
changed meanwhile (e.g. auto-segmentation checks volume dimensions and
frame-of-reference UID before applying).

## Geometry conventions

* Patient space is DICOM **LPS**, `f64` millimeters (`Vec3`).
* Volume voxels are stored `data[k·nx·ny + j·nx + i]` with dims
  `[nx, ny, nz]` = [columns, rows, slices]; `origin` is the **center** of
  voxel (0,0,0); `row_dir`/`col_dir`/`normal` are unit vectors, so the
  code never assumes axis-aligned volumes.
* Segmentation masks use the identical index order, so mask ↔ volume
  operations are index-parallel.
* Display convention: sagittal/coronal view rows run superior → inferior
  (`y = (nz−1) − k`); every producer of view-space pixels honors the same
  flip (asserted by tests).
* Interpolation is trilinear unless stated; resampling loops use
  incremental affine stepping (one vector add per pixel) in hot paths.

## Error handling and style

`anyhow::Result` with `bail!`/`context` at operation boundaries; missing
or malformed *individual* DICOM attributes never error — safe extraction
helpers return `Option` and per-file failures inside a batch become
warnings shown in the UI. `rayon` idioms: `par_iter` over independent
files/ROIs, `par_chunks_mut` over image rows/slices, dense per-chunk
accumulators, out-parameter buffers to avoid re-allocation. Modules open
with a `//!` block explaining the algorithm and its conventions, usually
citing the reference implementation (elastix, MITK, 3D Slicer, nnU-Net).

## Dependencies

Runtime dependencies are all pure Rust: `dicom-rs` (DICOM), `egui`/
`eframe` (UI over wgpu), `rayon`, `rfd` (file dialogs), `walkdir`,
`anyhow`; for auto-segmentation additionally `gemm` (SIMD matrix kernels),
`serde_json`, `zip`, `ureq` (rustls + OS trust store), `safetensors`, and
optionally `burn` (wgpu compute backend, cargo feature `gpu`, default on).

## Testing

Six integration suites plus in-module unit tests run against the same
code paths the GUI uses, with no external data or tooling:

* **synthetic_study** — generate the analytic phantom, reload, verify
  geometry round-trips, HU values, contour radii, trilinear dose values,
  isodose radii and plan fields against closed-form expectations;
* **simulate_export** — simulate a known transform → export DICOM →
  reload → verify within format tolerances;
* **registration** — rigid and B-spline recovery of analytically known
  transforms (sub-voxel assertions);
* **segmentation** — brush/undo semantics, geodesic-grow no-leak,
  hole filling, mask → RTSTRUCT contours, meshing;
* **anonymize** — anonymize → reload: identity gone, references intact,
  pixels byte-identical;
* **autoseg** — miniature network assembly with exact checkpoint naming +
  forward pass; sliding-window steps and resampling conventions pinned to
  nnU-Net/scipy reference values; an `#[ignore]`d end-to-end test against
  the real 3 mm model;
* **segvol** — the published checkpoint's 475-tensor inventory, recorded in
  `tests/data/segvol-tensors.csv` and asserted module by module. The same
  fixture synthesizes a checkpoint with the real key names and shapes, so
  the network assembles and runs a genuine forward pass in CI without the
  724 MB download; `#[ignore]`d tests cover the real file and the full
  181 M-parameter image-encoder pass.

Beyond the automated tests, the auto-segmentation implementation was
validated against the reference implementation directly — exact
patch-level logit equivalence and mean Dice 0.9995 end-to-end (details in
[auto-segmentation.md](auto-segmentation.md#validation)).

```
cargo test --release
```
