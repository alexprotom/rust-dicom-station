# Architecture

## Design philosophy

**One language.** Everything is Rust — DICOM parsing, image
reconstruction, rendering primitives, registration, meshing, neural-net
inference, DICOM writing. Where a capability normally means binding a
C/C++ library (elastix, ITK, ONNX Runtime, CUDA), the algorithms are
re-implemented natively instead. The only system interface is the GPU,
reached twice through `wgpu` (Vulkan / DX12 / Metal): once by `eframe` to
blit the UI, once (optionally) by `burn` for neural-network inference —
auto-segmentation, SegVol's image encoder, and the whole MedSAM2 graph —
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
    prompt_seg.rs    prompt-driven segmentation: dialog, worker, result
    medsam2_seg.rs   slice propagation: the box drawn in the viewport, the
                     preview / refine / propagate loop, and the session that
                     keeps the network and the prepared stack alive
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
    params.rs        shape-checked view of a loaded state dict, shared by
                     every ported architecture
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
  medsam2/         slice propagation (pure-Rust MedSAM2 — SAM 2.1 fine-tuned
                   on medical images): prompt one slice, follow the structure
                   through the stack at the slice's own resolution. Every
                   module is generic over a `burn` backend, so one
                   implementation runs on the GPU and on the CPU
    weights.rs       the four published variants, download and conversion,
                     and the research-only licence that governs them
    layout.rs        the checkpoint's tensor layout, and the checks that
                     verify a file really is this model
    config.rs        the fixed dimensions: 512 input, 7 memories, 16 pointers
    ops.rs           the tensor helpers the port needs on top of burn
    layers.rs        conv, layer norm, MLP, the small shared pieces
    hiera.rs         Hiera-T image encoder: 4 stages, windowed attention
    neck.rs          FPN neck to 256 channels + the sine position encoding
    prompt.rs        SAM's prompt encoder: points, boxes, mask prompts
    decoder.rs       two-way transformer, hypernetwork mask filters, IoU and
                     object-presence heads
    sam.rs           the SAM head assembled: prompt -> masks for one slice
    memory.rs        memory encoder: mask downsampler + ConvNeXt fuser
    memattn.rs       memory attention: 4 layers, 2-D axial RoPE
    model.rs         the whole network, and the two ways a slice is
                     conditioned (a prompt, or the memory bank)
    track.rs         the memory bank and the slice-to-slice state machine:
                     temporal indices, object pointers, reverse tracking
    infer.rs         one-slice preview, the two propagation passes, the
                     slice range, thresholding, largest-component cleanup
    preprocess.rs    window, quantize to u8, orient; the prompt's and the
                     mask's way between the study grid and the network's
    resample.rs      PIL's resampling kernels, including the 8-bit
                     fixed-point arithmetic the reference depends on
    engine.rs        backend choice (wgpu, else CPU), the encoded-slice
                     cache, and the one call the user interface makes
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
examples/          autoseg_cli, autoseg_probe, segvol_cli, segvol_probe,
                   medsam2_cli, medsam2_probe (headless dev tools)
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

Eight integration suites plus in-module unit tests run against the same
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
* **segvol** — CPU/GPU agreement for the image encoder is `#[ignore]`d, not
  because it is unimportant but because `WgpuDevice::default()` returns a
  *software* adapter on CI runners; run it where the hardware is. The
  published checkpoint's 475-tensor inventory is recorded in
  `tests/data/segvol-tensors.csv` and asserted module by module. The same
  fixture synthesizes a checkpoint with the real key names and shapes, so
  the network assembles and runs a genuine forward pass in CI without the
  724 MB download; `#[ignore]`d tests cover the real file and the full
  181 M-parameter image-encoder pass.
* **medsam2** — the same synthesized-checkpoint trick assembles the real
  471-tensor network and runs genuine forward passes in CI: a slice through
  the engine with the documented shapes, a box prompt propagated through a
  small stack, an existing contour as the prompt, and the one-slice preview
  agreeing with the propagation's first step while proving the encoded
  slice is reused;
* **reference** — bit-level parity with the Python implementation. A
  randomly initialized SAM 2.1-T is built with `sam2` and PyTorch by
  `tools/gen_reference_activations.py`, which dumps every module's inputs
  and outputs *and* a ten-slice run of SAM 2's own video predictor; the
  suite reproduces all of it (worst 5.4e-6 relative). It skips when the
  dump is absent, so CI stays self-contained:
  `MEDSAM2_REF=/tmp/ref cargo test --release --test reference`.

Beyond the automated tests, the auto-segmentation implementation was
validated against the reference implementation directly — exact
patch-level logit equivalence and mean Dice 0.9995 end-to-end (details in
[auto-segmentation.md](auto-segmentation.md#validation)).

```
cargo test --release
```
