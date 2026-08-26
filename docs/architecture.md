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
the three segmentation engines — runs on a worker thread and reports
through one shared progress handle (see [Background jobs](#background-jobs)).

**Shared before specific.** What more than one feature needs lives one
level up: the progress handle, the model folder, the checkpoint
download / conversion / cache path, the device choice, the shape-checked
parameter view and the dense CPU kernels are written once in
`progress.rs`, `models.rs` and `nn/`; the three engines and the three tool
windows are built on top of them and hold only what is theirs.

## Functional overview

What the program does, by category. Every leaf exists in the code today; the
module map below says where.

```
rust-dicom-station
│
├── Application (GUI, egui over wgpu)
│   ├── Window chrome: menu bar, toolbar (W/L, presets, 3D, crosshair, reset), status bar
│   ├── Side panel: registration, simulation, per-dataset sections
│   │   (series tree, structures, segmentations, dose + isodose, plans, planar images,
│   │   spatial registrations, treatment records, warnings)
│   ├── Views: 1 × 3 or 2 × 3 (comparison) linked MPR viewports, crosshair,
│   │   zoom / pan / W-L interaction, maximize, per-view caches
│   ├── Floating windows: 3D structures (both datasets through the registration,
│   │   per-dataset opacity, vector-field glyphs), planar image viewers
│   ├── Data tree operations: rename every level (patient, study, series, sets,
│   │   structures, segments, dose, plan, planar, REG, records);
│   │   copy / move / remove patient · study · series across datasets;
│   │   create / connect / copy / move / remove RT structure sets and segmentation
│   │   series; copy / move / remove single or selected structures and segments
│   ├── Tool windows: auto-segmentation, prompt segmentation, slice propagation,
│   │   model manager, structure propagation, DRR, export, anonymizer,
│   │   test-data generator (one shared skeleton)
│   ├── Background jobs: one progress handle, one poll loop
│   ├── Settings: theme, model folder (viewer_settings.txt)
│   └── Theme: dark / light / system, accent colors
│
├── DICOM
│   ├── Import: directory scan, classification, patient ▶ study ▶ series tree, dataset merging
│   │   ├── Volumes: CT, MR, PT, NM, US, OT (parallel decode, compressed syntaxes, geometry)
│   │   └── Planar images: DX, CR, RTIMAGE, MG, XA, RF, PX
│   ├── RT objects
│   │   ├── RTSTRUCT (structure sets, contours, reference chain)
│   │   ├── SEG (DICOM Segmentation: binary / fractional multi-frame masks,
│   │   │   frame-position lattice, CIELab colors, read and written)
│   │   ├── RTDOSE (grids, trilinear patient-space sampling, plan reference)
│   │   ├── RTPLAN / RT Ion Plan (beams, control points, prescriptions)
│   │   ├── RTIMAGE (DRR / portal, as planar image)
│   │   ├── REG (spatial registration matrices and deformable grids, applied as
│   │   │   the active registration; a recovered field written back out)
│   │   └── RT (Ion) Beams Treatment Record (delivered metersets)
│   ├── Export: CT series + RTSTRUCT + SEG + RTDOSE + RTPLAN with an editable tag table
│   └── Anonymizer: scan, review every identifying tag, rewrite with consistent UID remap
│
├── Data simulation
│   ├── Synthetic RT phantom study (CT, RTSTRUCT, RTDOSE, RTPLAN, DX, RTIMAGE, REG, RTRECORD)
│   ├── Known-transform study generator (rigid + Gaussian deformation, registration QA)
│   └── Digitally reconstructed radiographs: exact Siddon ray tracing (plastimatch)
│       and interpolating ray-casting (ITK), IEC cone-beam geometry, beam's-eye view
│       from an RTPLAN beam, side-by-side difference
│
├── Image registration
│   ├── elastix-style rigid (6-DOF Euler, ASGD, pyramids, stochastic sampling)
│   ├── elastix-style deformable (rigid pre-alignment + cubic B-spline FFD)
│   ├── plastimatch-style deformable (align_center, dense analytic gradient,
│   │   bending-energy regularization, L-BFGS, mean squares or Mattes mutual information)
│   ├── plastimatch-style landmark warp (thin-plate spline, Gaussian, Wendland)
│   ├── Local registration: any method restricted to a structure with a margin;
│   │   refinement composed on top of an existing result
│   ├── Analytics: 6-DOF Procrustes fit, displacement statistics, Jacobian
│   │   determinant and folding, per-structure displacement
│   ├── Vector field: lattice sampling, arrows / deformed grid in the MPR views,
│   │   3-D glyphs, both datasets in one 3-D scene with per-dataset opacity
│   ├── Fusion overlay (magenta / green blend on the fixed dataset)
│   └── DICOM REG matrices and Deformable Spatial Registration grids applied as the
│       active registration; the recovered field written back out as one
│
├── Segmentation
│   ├── Voxel masks: brush / eraser (2D, 3D), geodesic region growing, undo,
│   │   slice overlays, hole filling, mask ▶ RTSTRUCT contours, RTSTRUCT ▶ mask,
│   │   grouped into segmentation series that live in the study and bind to an
│   │   image series (resampled onto its lattice when it is displayed)
│   ├── Propagation: structures and segmentations carried across a registration,
│   │   globally or refined on an enclosing structure first
│   ├── Surfaces: contour and mask ▶ meshes (scanline fill, surface nets, smoothing)
│   ├── Auto-segmentation — TotalSegmentator (nnU-Net), 117 classes,
│   │   3 mm / 1.5 mm × 5 / 6 mm models, CPU (im2col + SIMD GEMM) or GPU (burn/wgpu)
│   ├── Prompt segmentation — SegVol, box / point / text prompts,
│   │   3-D ViT + SAM-style decoder + CLIP text tower, zoom-out / zoom-in passes
│   └── Slice propagation — MedSAM2 (SAM 2.1 Hiera-T), box drawn in the view,
│       preview / include-exclude refinement, memory-bank propagation through the stack
│
├── Neural-network infrastructure (shared by every engine)
│   ├── Model folder: <exe>/models/{totalsegmentator, segvol, medsam2}, legacy migration
│   ├── Model manager: one inventory of every downloadable model, its state and size;
│   │   download / update / remove one or all, free redundant source checkpoints
│   ├── Weights: download (rustls), torch pickle reader, safetensors cache, conversion
│   ├── Device: Auto / GPU / CPU preference, one validated wgpu context, panic guard
│   ├── Parameters: shape-checked view of a state dict
│   └── CPU kernels: Mat / Act tensors, gemm linear, layer norm, activations, attention,
│       transposed conv, f16 ↔ f32
│
├── Core services
│   ├── Volume: patient-space geometry (LPS), slice extraction, sampling, canonical axes
│   ├── Geometry: Vec3 math, direction labels
│   ├── Render: window / level, dose colorwash, marching-squares isodose, contour ∩ plane
│   └── Progress: message, fraction, device, cancel, phase window
│
├── Tests: 9 integration suites + in-module unit tests, synthetic phantom, reference dumps
├── Examples: headless CLIs and probes for the three engines (shared examples/common)
├── Tools: Python scripts that produce the reference fixtures (never needed at runtime)
├── Installer: Windows setup (shortcuts, VC++ runtime, optional weight prefetch, uninstall)
└── CI: fmt, clippy -D warnings, tests on Linux + Windows, CPU-only build, installer build
```

### Sources of the algorithms

Nothing in the tree above is bound as a library; each of the heavy
algorithms is a native re-implementation of a published reference, and the
reference is what the tests compare against. Registration follows
[elastix](https://elastix.dev/) (rigid and B-spline, ASGD, pyramids) and
[plastimatch](https://plastimatch.org/) (dense B-spline with L-BFGS and a
bending-energy penalty, and the `landmark_warp` radial-basis kernels); the
mutual-information metric follows Mattes et al. (IEEE TMI 2003). The two DRR
projectors follow plastimatch's exact Siddon tracer and ITK's
`RayCastInterpolateImageFunction`. Auto-segmentation re-implements
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) on its
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) models; prompt segmentation
re-implements [SegVol](https://github.com/BAAI-DCAI/SegVol); slice
propagation re-implements [MedSAM2](https://github.com/bowang-lab/MedSAM2),
i.e. Meta's [SAM 2](https://github.com/facebookresearch/sam2) fine-tuned on
medical images. The papers to cite, the licences of the weights and the
numerical validation of each port are in the per-feature documents
([registration.md](registration.md), [auto-segmentation.md](auto-segmentation.md),
[segvol.md](segvol.md), [medsam2.md](medsam2.md)).

## Module map

Where each function above lives. The right-hand tag names its functional
category (**App**, **DICOM**, **Sim**, **Reg**, **Seg**, **NN**, **Core**).

```
src/
  main.rs           entry point (eframe/wgpu window)                              App
  lib.rs            library root — every module is public, so the integration
                    tests and the examples drive the same code as the GUI
  progress.rs       the one progress handle + ProgressSink, Quiet, Stderr         Core
  models.rs         the model folder: root, per-engine sub-folders, migration,
                    and the inventory of every downloadable model (state, size,
                    download / update / remove / free)                            NN
  settings.rs       persisted preferences (theme, model folder)                   App

  app/              egui application, split by concern; every submodule is a
                    further `impl ViewerApp` block, so the struct and all its
                    state stay in one place while the behaviour is grouped:     App
    mod.rs            ViewerApp and every type it holds, construction, the job
                      plumbing (Job::spawn, poll_job, poll_tool_job), per-frame driver
    theme.rs          theme-dependent colors
    chrome.rs         menu bar, toolbar, status bar, help
    panels.rs         side panel and its per-dataset sections
    views.rs          central MPR viewports, interaction, texture caches
    d3.rs             live 3D structure window
    planar.rs         floating DX / CR / RTIMAGE viewers
    tree.rs           dataset-tree copy / move / remove with reference chains
    rename.rs         renaming every level of the data tree: the targets, the
                      one-field dialog, and the study-only rename itself
    sets.rs           structure sets and segmentation series as tree nodes:
                      create, connect to an image series, copy / move / remove
                      whole series, and move single structures / segments
                      between any two of them (contour ⇄ mask conversion)
    jobs.rs           loading, simulation, export, generator, anonymizer and
                      auto-segmentation job starts
    dialogs.rs        auto-segmentation window + results, generator, anonymizer,
                      export, error dialog
    reg_panel.rs      the Registration section: method, region, parameters,
                      landmarks, the run, the analytics, the vector field
    models_win.rs     the model manager window
    propagate_win.rs  structure propagation window and worker
    drr_win.rs        the DRR window: geometry, projectors, comparison
    seg.rs            interactive segmentation state machine, mask ▶ RTSTRUCT,
                      landing an auto-segmentation result
    seg_engines.rs    what the three engine windows share: names and glyphs,
                      device / model-folder / licence / progress rows,
                      result landing, the "still the same dataset" check
    prompt_seg.rs     prompt segmentation window and worker (SegVol)
    box_seg.rs        slice propagation: the box drawn in the viewport, the
                      preview / refine / propagate loop, the resident session (MedSAM2)

  loader.rs         directory scan, classification, parallel volume loading,
                    dataset merging, safe DICOM element helpers                  DICOM
  volume.rs         3D volume, patient-space geometry, slice extraction,
                    trilinear sampling, canonical [S, A, R] axes                 Core
  geometry.rs       minimal 3D vector math (Vec3, f64, patient mm)               Core
  render.rs         window / level, dose colorwash, marching-squares isodose,
                    contour / plane intersection                                 Core
  rtstruct.rs       RT Structure Set parsing                                     DICOM
  dicomseg.rs       DICOM Segmentation: the segmentation-series model, SEG
                    reading (binary / fractional, frame-position lattice),
                    resampling between lattices, the SEG writer               DICOM
  rtdose.rs         RT Dose parsing + trilinear patient-space sampling           DICOM
  rtplan.rs         RT Plan / RT Ion Plan parsing                                DICOM
  extras.rs         DX / CR / RTIMAGE planar images, REG (matrices and
                    deformation grids), RTRECORD                                 DICOM
  dicom_export.rs   DICOM writer (CT series, RTSTRUCT, SEG, RTDOSE, RTPLAN, and
                    the Deformable Spatial Registration a recovered field
                    becomes)                                                     DICOM
  anonymize.rs      interactive DICOM anonymizer engine                          DICOM
  gen_test_data.rs  synthetic RT phantom study generator                         Sim
  simulate.rs       known-transform study generator (registration QA)           Sim
  registration.rs   parameters, transforms (rigid, B-spline, RBF, field,
                    composite), region masks, the image pyramid and the
                    samplers, and the engine dispatch                            Reg
    elastix.rs        stochastic sampling + ASGD, rigid and B-spline stages
    plastimatch.rs    align_center, dense analytic gradient, bending energy,
                      Mattes mutual information, L-BFGS
    landmark.rs       thin-plate / Gaussian / Wendland RBF warp, dense solve
    analysis.rs       6-DOF Procrustes fit, displacement and Jacobian statistics
    dvf.rs            vector-field sampling and its view-plane / 3-D glyphs
  propagate.rs      structures across a registration: pull-back with a cached
                    mapping lattice                                              Reg
  drr.rs            digitally reconstructed radiographs: IEC cone-beam geometry,
                    Siddon exact tracing and ITK-style interpolating ray-casting  Sim
  segmentation.rs   voxel masks: brush, geodesic grow, undo, overlays,
                    label map ▶ segmentations, mask ▶ RTSTRUCT contours and
                    RTSTRUCT contours ▶ mask                                     Seg
  mesh3d.rs         contour / mask ▶ surface meshes (scanline fill,
                    surface nets, Laplacian smoothing)                           Seg

  nn/               shared neural-network infrastructure — nothing in here
                    knows about a particular architecture                        NN
    cache.rs          RemoteFile download, torch checkpoint ▶ safetensors
                      conversion (ConvertSpec), the converted-weight cache
    pickle.rs         native PyTorch checkpoint (.pth / .pt / .bin) reader
    device.rs         DevicePref (Auto / GPU / CPU), the validated wgpu
                      context, the backend-panic guard
    params.rs         shape-checked view of a loaded state dict
    half.rs           binary16 ↔ binary32 conversion
    tensor.rs         Mat [rows, cols] and Act [c, d, h, w]; transposed conv
    linalg.rs         gemm-backed linear / matmul, layer norm, softmax,
                      GELU / ReLU / QuickGELU
    attention.rs      multi-head attention, optionally causally masked

  autoseg/          automatic segmentation (pure-Rust TotalSegmentator)         Seg
    mod.rs            public API: variants, run(), progress phases
    classes.rs        117-class table, sub-model maps, organ colors
    config.rs         nnU-Net plans.json parsing
    weights.rs        which models exist, where they are published, the
                      release-zip unpacking in front of the shared conversion
    cpu.rs            CPU conv engine (im2col + SIMD GEMM conv3d, norms)
    net.rs            PlainConvUNet assembly + CPU forward
    gpu.rs            wgpu forward via burn (cargo feature `gpu`)
    preprocess.rs     resampling to the model grid and back (scipy conventions)
    infer.rs          Gaussian sliding window, streaming argmax

  segvol/           prompt segmentation (pure-Rust SegVol) — box, point and
                    text prompts, for the structures a fixed-class model
                    cannot cover                                                 Seg
    weights.rs        the checkpoint and tokenizer files, load(), licensing notes
    layout.rs         the published checkpoint's tensor layout and its checks
    config.rs         the network's fixed dimensions
    vit.rs            image encoder (MONAI 3-D ViT, 12 blocks, 2048 tokens)
    prompt.rs         prompt encoder: box / point / text ▶ sparse + dense
    decoder.rs        two-way transformer, upscaling, mask hypernetworks
    net.rs            assembly and the single-window forward pass
    preprocess.rs     foreground normalization, canonical orientation,
                      nearest-exact / trilinear resampling, mask back-mapping
    infer.rs          zoom-out / zoom-in orchestration, MONAI window layout
    bpe.rs            CLIP byte-pair tokenizer
    clip.rs           CLIP text tower + dim_align, with a prompt cache
    gpu.rs            image encoder on wgpu via burn (cargo feature `gpu`)

  medsam2/          slice propagation (pure-Rust MedSAM2 — SAM 2.1 fine-tuned
                    on medical images); every module is generic over a `burn`
                    backend, so one implementation runs on GPU and CPU          Seg
    weights.rs        the four published variants, load(), the research-only licence
    layout.rs         the checkpoint's tensor layout and its checks
    config.rs         the fixed dimensions: 512 input, 7 memories, 16 pointers
    ops.rs            the tensor helpers the port needs on top of burn
    layers.rs         conv, layer norm, linear (kept transposed), MLP
    hiera.rs          Hiera-T image encoder: 4 stages, windowed attention
    neck.rs           FPN neck to 256 channels + the sine position encoding
    prompt.rs         SAM's prompt encoder: points, boxes, mask prompts
    decoder.rs        two-way transformer, hypernetwork mask filters, IoU and
                      object-presence heads
    sam.rs            the SAM head assembled: prompt ▶ masks for one slice
    memory.rs         memory encoder: mask downsampler + ConvNeXt fuser
    memattn.rs        memory attention: 4 layers, 2-D axial RoPE
    model.rs          the whole network, and the two ways a slice is conditioned
    track.rs          the memory bank and the slice-to-slice state machine
    infer.rs          one-slice preview, the two propagation passes, the slice
                      range, thresholding, largest-component cleanup
    preprocess.rs     window, quantize to u8, orient; the prompt's and the
                      mask's way between the study grid and the network's
    resample.rs       PIL's resampling kernels, incl. 8-bit fixed-point arithmetic
    engine.rs         backend choice, the encoded-slice cache, the one call
                      the user interface makes

tests/             nine integration suites (see Testing)
examples/          autoseg_cli, autoseg_probe, segvol_cli, segvol_probe,
                   medsam2_cli, medsam2_probe; common/ holds what they share
tools/             gen_reference_activations.py, gen_ops_fixtures.py — the
                   two PyTorch scripts that produce the fixtures and reference
                   dumps the MedSAM2 tests compare against (never run at
                   build time; needed only to regenerate them)
installer/         the Windows installer, its own workspace (see its README)
```

## UI architecture

`ViewerApp` is defined in `app/mod.rs` together with every type it holds;
the sibling modules only add `impl ViewerApp` blocks. Keeping the
definitions in the parent module is what lets each child reach the struct's
private fields without widening any visibility beyond `pub(super)`.

`ViewerApp` owns two `StudySlot`s (datasets A and B). Each slot holds the
loaded study (series list, the volume behind an `Arc`, structure sets,
doses, plans, planar images, registrations, records), three `ViewState`s
(per-plane slice, zoom/pan, and all texture caches), the crosshair, per-ROI
visibility, and the segmentation masks. Global state covers window/level,
dose display settings, tool selection, the registration result, the model
folder, and the theme.

Rendering is cache-driven: each view keeps keyed textures for the
grayscale slice, dose colorwash, contour polylines, segmentation overlay
and fusion blend, rebuilt only when their inputs change (slice, W/L, dose
settings, ROI visibility, mask edits, registration). Invalidation uses
small generation counters bumped by the owning mutation sites — and only by
those: a ROI visibility toggle, for instance, is part of the contour key
alone and leaves the dose and fusion textures untouched. Repaints are
demand-driven; while background jobs run, the UI polls at 10 Hz.

### The three engine windows

Auto-segmentation, prompt segmentation and slice propagation are different
conversations — a batch run with a result-selection dialog, a one-shot
prompt, an interactive box loop — but they are the same kind of tool, and
`app/seg_engines.rs` makes them look and behave alike:

* one `ToolInfo` per engine gives the glyph (🤖 🧠 ⏩), the window title
  (`🤖 Auto-segmentation — dataset A`, the same pattern as
  `3D structures — dataset A`), the menu entry (`🤖 Auto-segment dataset A…`)
  and the small sidebar button (`🤖 Auto…`);
* every window is floating, collapsible and closable, and stays open while
  its run is in flight — the button row becomes the progress row (device,
  bar, message, Cancel); closing it never stops the run, and the sidebar
  *Segmentations* section shows whichever engine is running on that
  dataset with the same Cancel;
* the sections come in the same order: a one-line description naming the
  engine, the tool's own inputs, `Name`, a collapsed **Options** header
  holding the engine's settings plus the shared `Compute: Auto / GPU / CPU`
  and `Model folder` rows, one small licence line ("… Research / QA use —
  not a medical device."), then `▶ Segment` / `▶ Propagate` and `Close`,
  and a status line summarising the last result;
* results land the same way — `add_segmentation` with the next palette
  colour — and a run that finishes after the dataset was replaced is
  discarded with the same message.

## Background jobs

One pattern serves every long operation:

```rust
struct Job<T, P = Progress> { progress: Arc<P>, rx: mpsc::Receiver<T> }
```

`Job::spawn` snapshots the inputs, starts a `std::thread` and hands the
worker the progress handle; the UI polls the channel each frame
(`poll_job`): a received value lands the result, a disconnect means the
worker died and surfaces as an error. The engines and the registration
answer with `(slot, Result)`, and `poll_tool_job` turns a failure into an
error dialog — except a cancellation, which is what the user asked for.

There is one `Progress` type (`progress.rs`): a message, a fraction for
progress bars, the device label once known, an atomic cancel flag, and a
phase window that maps a sub-step's own 0‥1 onto its slice of the overall
bar. Workers see it through the `ProgressSink` trait, which the headless
examples implement on standard error and the tests with `Quiet`. Workers
use `rayon` internally for data parallelism; the thread-per-job is only
the container.

Results are validated on landing where the underlying data could have
changed meanwhile (every engine checks volume dimensions and
frame-of-reference UID before applying).

## The model folder

Every engine downloads its published checkpoint on first use and keeps it,
with the converted `safetensors` cache beside it, under one root that the
user can move from any of the three tool windows and that is persisted as
`models_dir` in `viewer_settings.txt`:

```
<folder of the executable>/models/
  totalsegmentator/<model>/model.safetensors + plans.json
  segvol/pytorch_model.bin, vocab.json, merges.txt, segvol.safetensors
  medsam2/MedSAM2_<variant>.pt + .safetensors
```

`models.rs` owns the layout; `nn/cache.rs` owns the path from a URL to a
loaded tensor map (`RemoteFile::ensure` ▶ `convert_checkpoint` ▶
`load_safetensors`, wrapped as `ensure_converted`), and each engine's
`weights.rs` only says which files, which tensors and under what names.
Installations that predate the single root are migrated at startup: the
old `autoseg_models/`, `segvol_model/` and `medsam2_model/` folders beside
the executable are renamed into place, never re-downloaded. The Windows
installer writes the same key and pre-fetches only the Apache-2.0
TotalSegmentator weights, into `models/totalsegmentator/`.

## Geometry conventions

* Patient space is DICOM **LPS**, `f64` millimeters (`Vec3`).
* Volume voxels are stored `data[k·nx·ny + j·nx + i]` with dims
  `[nx, ny, nz]` = [columns, rows, slices]; `origin` is the **center** of
  voxel (0,0,0); `row_dir`/`col_dir`/`normal` are unit vectors, so the
  code never assumes axis-aligned volumes.
* `Volume::canonical_axes` finds the permutation and flips onto `[S, A, R]`
  by direction cosine; all three engines orient through it (MedSAM2 reads
  the in-plane axes the other way round, as SAM 2 does).
* Segmentation masks use the identical index order, so mask ↔ volume
  operations are index-parallel.
* Display convention: sagittal/coronal view rows run superior → inferior
  (`y = (nz−1) − k`); every producer of view-space pixels honors the same
  flip (asserted by tests).
* Interpolation is trilinear unless stated. The engines deliberately keep
  their reference implementations' resampling conventions — scipy `zoom`
  (nnU-Net), PyTorch `nearest-exact` / `align_corners=false` (SegVol), PIL
  antialiased bicubic in 8-bit fixed point (MedSAM2) — because each is
  validated numerically against that reference; they are not unified.

## Error handling and style

`anyhow::Result` with `bail!`/`context` at operation boundaries; missing
or malformed *individual* DICOM attributes never error — safe extraction
helpers return `Option` and per-file failures inside a batch become
warnings shown in the UI. Cancellation is an error whose message contains
`progress::CANCELLED`, recognised by the application. `rayon` idioms:
`par_iter` over independent files/ROIs, `par_chunks_mut` over image
rows/slices, dense per-chunk accumulators. Sums that decide a threshold or
a normalization stay sequential so a run reproduces itself. Modules open
with a `//!` block explaining the algorithm and its conventions, usually
citing the reference implementation (elastix, MITK, 3D Slicer, nnU-Net,
SAM 2).

Because `lib.rs` makes every module public, `cargo clippy -D warnings`
cannot see an unused `pub` item; the 2026-08 review found them with a
mechanical scan (every `pub` item referenced nowhere outside its own
tests) — worth repeating occasionally.

## Dependencies

Runtime dependencies are all pure Rust: `dicom-rs` (DICOM, with
`dicom-pixeldata` for decoding), `egui`/`eframe` (UI over wgpu), `rayon`,
`rfd` (file dialogs), `walkdir`, `anyhow`; for the engines additionally
`gemm` (SIMD matrix kernels), `serde_json` (plans.json, vocab.json), `zip`,
`ureq` (rustls + OS trust store), `safetensors`, and `burn` — always
compiled with its `ndarray` CPU backend (the MedSAM2 engine is written
against it), with the wgpu backend added by the cargo feature `gpu`
(default on).

## Testing

Nine integration suites plus in-module unit tests run against the same
code paths the GUI uses, with no external data or tooling:

* **synthetic_study** — generate the analytic phantom, reload, verify
  geometry round-trips, HU values, contour radii, trilinear dose values,
  isodose radii and plan fields against closed-form expectations;
* **simulate_export** — simulate a known transform → export DICOM →
  reload → verify within format tolerances;
* **registration** — rigid and B-spline recovery of analytically known
  transforms (sub-voxel assertions);
* **segmentation** — brush/undo semantics, geodesic-grow no-leak,
  hole filling, one-pass label-map splitting, mask → RTSTRUCT contours,
  meshing;
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
