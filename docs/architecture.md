# Architecture

An interactive map of the whole program — modules, the data path and the
background-job fan-out — is in
[architecture-diagram.html](architecture-diagram.html): open it in a browser,
click a box to focus it, or use the three guided views along the top. It is
rendered from [architecture-diagram.archify.json](architecture-diagram.archify.json)
and exports to PNG or SVG from the *Export* menu.

## Design philosophy

**One language.** Everything is Rust — DICOM parsing, image reconstruction,
rendering primitives, registration, meshing, neural-net inference, DICOM
writing. Where a capability normally means binding a C/C++ library (elastix,
ITK, ONNX Runtime, CUDA), the algorithm is re-implemented natively. The only
system interface is the GPU, reached through `wgpu` (Vulkan / DX12 / Metal)
by `eframe` to blit the UI and, optionally, by `burn` to run the networks.

**CPU-side algorithms, GPU-side pixels.** Image processing runs on the CPU
with `rayon` and aggressive caching; the GPU receives finished textures.
Every algorithm stays debuggable, deterministic and portable, and it is fast
enough: study load ≈ 40 ms, orthogonal slice ≈ 6 µs, dose-plane resampling
≈ 0.3 ms on the synthetic study.

**Long work never blocks the UI.** Anything longer than a frame runs on a
worker thread and reports through one progress handle
([Background jobs](#background-jobs)).

**Shared before specific.** What more than one feature needs lives one level
up: the progress handle, the model folder, the checkpoint download /
conversion / cache path, the device choice, the shape-checked parameter view
and the dense CPU kernels are written once (`progress.rs`, `models.rs`,
`nn/`); the engines and the tool windows hold only what is theirs.

## Functional overview

What the program does, by category; the [module map](#module-map) says
where each leaf lives.

```
rust-dicom-station
│
├── Application (GUI, egui over wgpu)
│   ├── Window chrome: menu bar, toolbar (W/L, presets, 3D, crosshair, reset), status bar
│   ├── Side panel: the optional registration and simulation sections, and per
│   │   dataset a DICOM tree — patient ▶ study ▶ modality ▶ series, with RT
│   │   structures, segmentations, 4D groups, dose and plans inside their study —
│   │   plus dose display, planar images, spatial registrations, records, warnings
│   ├── Views: 1 × 3 or 2 × 3 (comparison) linked MPR viewports, crosshair,
│   │   a dataset with no volume says so in place of the panes and holds back
│   │   the voxel tools;
│   │   zoom / pan / W-L interaction, maximize, per-view caches
│   ├── Tool windows (one shared skeleton; each can be docked over the views or
│   │   detached into its own window of the operating system):
│   │   3D structures, planar viewers, auto-segmentation, prompt segmentation,
│   │   slice propagation, body contour, structure algebra, structure propagation,
│   │   4D motion / ITV and its results, structure comparison, transfer by
│   │   relationship, DVH, DRR, PACS, model manager, export, anonymizer, generator
│   ├── Data tree operations: rename every level; Shift-click ranges; copy / move /
│   │   remove / export the ticked items; create / connect / copy / move / remove
│   │   structure sets and segmentation series; move single structures / segments
│   ├── Background jobs: one progress handle, one poll loop
│   ├── Settings: theme, model folder, archive folder, optional modules,
│   │   detached windows (viewer_settings.txt in the config folder)
│   └── Theme: dark / light / system, accent colors
│
├── DICOM
│   ├── Import: directory *or* file-list scan, classification, patient ▶ study ▶
│   │   series tree, merging; a selection that reconstructs no volume (RT images,
│   │   a structure set, a plan) loads as an ordinary dataset with an empty
│   │   volume, and unpositioned image series open as single images
│   │   ├── Volumes: CT, MR, PT, NM, US, OT (parallel decode, compressed syntaxes)
│   │   └── Planar images: DX, CR, RTIMAGE, MG, XA, RF, PX
│   ├── RT objects: RTSTRUCT, SEG (binary / fractional, read and written), RTDOSE,
│   │   RTPLAN / RT Ion Plan, RTIMAGE, REG (matrices and deformable grids, applied
│   │   as the active registration, written back out), RT (Ion) Treatment Record
│   ├── Export: CT + RTSTRUCT + SEG + RTDOSE + RTPLAN with an editable tag table
│   ├── Anonymizer: scan, review every identifying tag, rewrite with a UID remap
│   └── Patient archive: a local store filed patient ▶ study ▶ instance with text
│       sidecars; import with dedupe, listing without opening a file, loading into
│       a dataset, derived objects (RTSTRUCT, SEG) sent back under the original UIDs
│
├── Data simulation
│   ├── Synthetic RT phantom study (CT, RTSTRUCT, RTDOSE, RTPLAN, DX, RTIMAGE, REG, RTRECORD)
│   ├── Known-transform study generator (rigid + Gaussian deformation, registration QA)
│   └── DRR: exact Siddon tracing (plastimatch) and interpolating ray-casting (ITK),
│       IEC cone-beam geometry, beam's-eye view from an RTPLAN beam, difference image
│
├── Image registration
│   ├── elastix-style rigid (6-DOF Euler, ASGD, pyramids, stochastic sampling)
│   ├── elastix-style deformable (rigid pre-alignment + cubic B-spline FFD)
│   ├── plastimatch-style deformable (dense analytic gradient, bending energy,
│   │   L-BFGS, mean squares or Mattes mutual information)
│   ├── plastimatch-style landmark warp (thin-plate spline, Gaussian, Wendland)
│   ├── Local registration: any method restricted to a structure with a margin;
│   │   refinement composed on top of an existing result
│   ├── Analytics: 6-DOF Procrustes fit, displacement statistics, Jacobian
│   │   determinant and folding, per-structure displacement
│   ├── Vector field: arrows / deformed grid in the views, 3-D glyphs
│   ├── Fusion overlay (magenta / green)
│   └── Structure propagation across any registration, globally or refined on an
│       enclosing structure first
│
├── 4D and motion
│   ├── 4D groups: phases recognised from descriptions and temporal identifiers,
│   │   AVG / MIP filed with them, hand-built groups kept across re-detection
│   ├── Motion pipeline per phase: register (rigid / deformable) ▸ propagate the
│   │   targets ▸ centroid, volume, peak-to-peak, drift, correlation with a
│   │   reference structure (Pearson r, p), registration QA
│   ├── ITV: union over phases with a margin, landed as a segmentation
│   ├── Results window: charts, tables, CSV, run-vs-run (A/B) comparison
│   ├── Structure comparison: volumes, centroid offset, Dice, HD95, mean surface distance
│   └── Transfer by relationship: a structure placed in the other dataset at its
│       offset from a reference structure
│
├── Dose analysis
│   └── DVH: dose sampled over the structure's own lattice, cumulative and
│       differential curves, Dx% / Dxcc / Vx metrics, protocol constraints, CSV
│
├── Segmentation
│   ├── Voxel masks: brush / eraser (2D, 3D), geodesic region growing, undo,
│   │   overlays, hole filling, mask ⇄ RTSTRUCT, segmentation series bound to an
│   │   image series (resampled onto its lattice for display)
│   ├── Body / EXTERNAL contour: threshold by modality (HU, or bias-flattened MR),
│   │   spacing-aware opening, extruded-equipment removal along all three axes,
│   │   component selection, thin-anatomy recovery — classically, or guided by
│   │   TotalSegmentator's body network
│   ├── Structure algebra: union / intersection / subtraction / symmetric difference,
│   │   a margin per operand and on the result (six patient directions, exact
│   │   ellipsoids), crop, fill / smooth / prune
│   ├── Surfaces: contour and mask ▶ meshes (scanline fill, surface nets, smoothing)
│   ├── Auto-segmentation — TotalSegmentator (nnU-Net), 117 classes, 3 / 1.5 / 6 mm
│   │   models, CPU (im2col + SIMD GEMM) or GPU (burn / wgpu)
│   ├── Prompt segmentation — SegVol: box / point / text, 3-D ViT + SAM-style
│   │   decoder + CLIP text tower, zoom-out / zoom-in passes
│   └── Slice propagation — MedSAM2 (SAM 2.1 Hiera-T): box drawn in the view,
│       include / exclude refinement, memory-bank propagation through the stack
│
├── Neural-network infrastructure (shared by every engine)
│   ├── Model folder: <data folder>/models/{totalsegmentator, segvol, medsam2},
│   │   legacy migration; the model manager's inventory (state, size, download /
│   │   update / remove / free)
│   ├── Weights: download (rustls), torch pickle reader, safetensors cache, conversion
│   ├── Device: Auto / GPU / CPU, one validated wgpu context, panic guard
│   ├── Parameters: shape-checked view of a state dict
│   └── CPU kernels: Mat / Act tensors, gemm linear, layer norm, activations,
│       attention, transposed conv, f16 ↔ f32
│
├── Core services
│   ├── Volume: patient-space geometry (LPS), slice extraction, sampling, canonical axes
│   ├── Geometry: Vec3 math, direction labels
│   ├── Morphology: exact anisotropic distance transform, erode / dilate / open /
│   │   close, ellipsoidal margins, components, hole filling
│   ├── Render: window / level, dose colorwash, marching-squares isodose, contour ∩ plane
│   └── Progress: message, fraction, device, cancel, phase window
│
├── Tests: 15 integration suites + in-module unit tests, synthetic phantom, reference dumps
├── Examples: headless CLIs and probes for the engines (shared examples/common)
├── Tools: the two PyTorch scripts that produce the MedSAM2 reference fixtures
├── Installer: Windows setup (shortcuts, VC++ runtime, optional weight prefetch, uninstall)
└── CI: fmt, clippy -D warnings, tests on Linux + Windows, CPU-only build; every push
    to main builds the installer and a Linux AppImage into a GitHub release
```

### Sources of the algorithms

Nothing above is bound as a library; each heavy algorithm is a native
re-implementation of a published reference, and the reference is what the
tests compare against. Registration follows [elastix](https://elastix.dev/)
(rigid and B-spline, ASGD, pyramids) and
[plastimatch](https://plastimatch.org/) (dense B-spline with L-BFGS and a
bending-energy penalty, and the `landmark_warp` kernels); mutual information
follows Mattes et al. (IEEE TMI 2003). The DRR projectors follow
plastimatch's exact Siddon tracer and ITK's
`RayCastInterpolateImageFunction`. Auto-segmentation re-implements
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) on its
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) models; prompt segmentation
re-implements [SegVol](https://github.com/BAAI-DCAI/SegVol); slice
propagation re-implements [MedSAM2](https://github.com/bowang-lab/MedSAM2),
i.e. Meta's [SAM 2](https://github.com/facebookresearch/sam2) fine-tuned on
medical images. Papers, weight licences and the numerical validation of each
port are in the per-feature documents ([registration.md](registration.md),
[auto-segmentation.md](auto-segmentation.md), [segvol.md](segvol.md),
[medsam2.md](medsam2.md)).

## Module map

Where each function lives. The right-hand tag is the functional category
(**App**, **DICOM**, **Sim**, **Reg**, **4D**, **Dose**, **Seg**, **NN**,
**Core**).

```
src/
  main.rs           entry point: opens the eframe/wgpu window, retrying the
                    other graphics backends when one will not start              App
  lib.rs            library root — every module is public, so the integration
                    tests and the examples drive the same code as the GUI
  progress.rs       the one progress handle + ProgressSink, Quiet, Stderr         Core
  models.rs         the model folder: root, per-engine sub-folders, migration,
                    the inventory of every downloadable model                    NN
  settings.rs       persisted preferences and the config / data folders — the
                    machine-wide defaults the installer writes, then the user's
                    own file on top                                              App
  gfx.rs            which graphics backend to draw and compute with: the
                    settings key, the environment override, and the order to
                    fall back through when one will not start                    App
  archive.rs        the local patient archive: on-disk layout, sidecars,
                    scanning, importing, index rebuild, removal                   DICOM

  app/              egui application, split by concern; every submodule is a
                    further `impl ViewerApp` block, so the struct and its state
                    stay in one place while the behaviour is grouped:            App
    mod.rs            ViewerApp and every type it holds, construction, the job
                      plumbing (Job::spawn, poll_job, poll_tool_job), per-frame driver
    theme.rs          theme-dependent colors
    glyphs.rs         the font stack (Hack as the last proportional fallback)
                      and the test that fails on a glyph egui cannot draw
    chrome.rs         menu bar, toolbar, status bar, help
    detach.rs         every tool window as a window of the operating system
                      (immediate viewport), titled and placed alike
    panels.rs         left panel: show / hide, the per-dataset Data tree sections
    reg_panel.rs      the Image registration section: method, region, parameters,
                      landmarks, the run, the analytics, the vector field
    views.rs          central MPR viewports, interaction, texture caches
    d3.rs             live 3D structure window
    planar.rs         floating DX / CR / RTIMAGE viewers
    tree.rs           dataset-tree copy / move / remove with reference chains
    rename.rs         renaming every level of the data tree
    sets.rs           structure sets and segmentation series as tree nodes:
                      create, connect, copy / move / remove, move single
                      structures / segments (contour ⇄ mask conversion)
    jobs.rs           loading, simulation, export, generator, anonymizer and
                      auto-segmentation job starts
    dialogs.rs        auto-segmentation window + results, generator, anonymizer,
                      export, error dialog
    seg.rs            interactive segmentation state machine, mask ▶ RTSTRUCT,
                      landing an auto-segmentation result
    seg_engines.rs    what the tool windows share: names and glyphs, device /
                      model-folder / licence / progress rows, result landing,
                      the "still the same dataset" check
    body_win.rs       the body-contour window
    combine_win.rs    the structure-algebra window: operands, margins, the recipe
    prompt_seg.rs     prompt segmentation window and worker (SegVol)
    box_seg.rs        slice propagation: the box drawn in the viewport, the
                      preview / refine / propagate loop, the resident session (MedSAM2)
    propagate_win.rs  structure propagation window and worker
    motion_win.rs     the 4D motion / ITV window and its per-phase pipeline
                      worker (register ▸ propagate ▸ measure ▸ ITV)
    motion_results.rs the motion results window: charts, tables, correlations,
                      QA, CSV, run-vs-run comparison
    compare_win.rs    compare structures: volumes, centroid offset, Dice, HD95, MSD
    transfer_win.rs   transfer by relationship
    dvh_win.rs        the DVH window: pickers, the plot, the metrics table,
                      constraints, export
    drr_win.rs        the DRR window: geometry, projectors, comparison
    pacs_win.rs       the PACS window: archive root, patient / study list,
                      import, load, send back
    models_win.rs     the model manager window

  loader.rs         directory / file-list scan, classification, parallel volume
                    loading, dataset merging, safe DICOM element helpers         DICOM
  volume.rs         3D volume, patient-space geometry, slice extraction,
                    trilinear sampling, canonical [S, A, R] axes                 Core
  geometry.rs       minimal 3D vector math (Vec3, f64, patient mm)               Core
  render.rs         window / level, dose colorwash, marching-squares isodose,
                    contour / plane intersection                                 Core
  morphology.rs     binary-mask geometry in millimetres: exact anisotropic
                    distance transform, erode / dilate / open / close,
                    ellipsoidal margins, components, hole filling, the
                    extruded-equipment test, box-blur smoothing                  Core
  rtstruct.rs       RT Structure Set parsing                                     DICOM
  dicomseg.rs       DICOM Segmentation: the segmentation-series model, SEG
                    reading, resampling between lattices, the SEG writer         DICOM
  rtdose.rs         RT Dose parsing + trilinear patient-space sampling           DICOM
  rtplan.rs         RT Plan / RT Ion Plan parsing                                DICOM
  extras.rs         DX / CR / RTIMAGE planar images, REG, RTRECORD               DICOM
  dicom_export.rs   DICOM writer (CT, RTSTRUCT, SEG, RTDOSE, RTPLAN, Deformable
                    Spatial Registration)                                        DICOM
  anonymize.rs      interactive DICOM anonymizer engine                          DICOM
  gen_test_data.rs  synthetic RT phantom study generator                         Sim
  simulate.rs       known-transform study generator (registration QA)           Sim
  drr.rs            DRR: IEC cone-beam geometry, Siddon exact tracing and
                    ITK-style interpolating ray-casting                          Sim
  registration.rs   parameters, transforms (rigid, B-spline, RBF, field,
                    composite), region masks, pyramid, samplers, dispatch        Reg
    elastix.rs        stochastic sampling + ASGD, rigid and B-spline stages
    plastimatch.rs    align_center, dense analytic gradient, bending energy,
                      Mattes mutual information, L-BFGS
    landmark.rs       thin-plate / Gaussian / Wendland RBF warp, dense solve
    analysis.rs       6-DOF Procrustes fit, displacement and Jacobian statistics
    dvf.rs            vector-field sampling and its view-plane / 3-D glyphs
  propagate.rs      structures across a registration: pull-back with a cached
                    mapping lattice                                              Reg
  fourd.rs          4D sub-studies: phase recognition, ordered groups
                    (phases + AVG / MIP), custom-group rules                     4D
  motion.rs         motion arithmetic over phases: centroids, peak-to-peak,
                    drift, Pearson r with p-values, Dice / HD95 / MSD overlap,
                    ITV unions, the motion report + CSV                          4D
  dvh.rs            dose–volume histograms: sampling, curves, metrics,
                    protocol constraints, CSV                                    Dose
  segmentation.rs   voxel masks: brush, geodesic grow, undo, overlays,
                    label map ▶ segmentations, mask ⇄ RTSTRUCT contours          Seg
  structops.rs      structure algebra: the four boolean operations, margins,
                    crop, fill / smooth / prune, over masks on one lattice       Seg
  bodymask.rs       the body / EXTERNAL contour, classically or guided by the
                    body network                                                 Seg
  mesh3d.rs         contour / mask ▶ surface meshes (scanline fill, surface
                    nets, Laplacian smoothing)                                   Seg

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
    linalg.rs         gemm-backed linear / matmul, layer norm, softmax, activations
    attention.rs      multi-head attention, optionally causally masked

  autoseg/          automatic segmentation (pure-Rust TotalSegmentator)         Seg
    mod.rs            public API: variants, run(), run_specs() (shared with the
                      body contour), progress phases
    classes.rs        117-class table, sub-model maps, organ colors
    config.rs         nnU-Net plans.json parsing
    weights.rs        which models exist, where they are published, the
                      release-zip unpacking in front of the shared conversion
    cpu.rs            CPU conv engine (im2col + SIMD GEMM conv3d, norms)
    net.rs            PlainConvUNet assembly + CPU forward
    gpu.rs            wgpu forward via burn (cargo feature `gpu`)
    preprocess.rs     resampling to the model grid and back (scipy conventions)
    infer.rs          Gaussian sliding window, streaming argmax

  segvol/           prompt segmentation (pure-Rust SegVol)                       Seg
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

  medsam2/          slice propagation (pure-Rust MedSAM2); every module is
                    generic over a `burn` backend, so one implementation runs
                    on GPU and CPU                                               Seg
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

tests/             fifteen integration suites (see Testing)
examples/          autoseg_cli, autoseg_probe, body_cli, segvol_cli, segvol_probe,
                   medsam2_cli, medsam2_probe; common/ holds what they share
tools/             gen_reference_activations.py, gen_ops_fixtures.py — the two
                   PyTorch scripts that produce the fixtures and reference dumps
                   the MedSAM2 tests compare against (never run at build time)
installer/         the Windows installer, its own workspace (see its README);
                   built by the release workflow
```

## UI architecture

`ViewerApp` is defined in `app/mod.rs` together with every type it holds; the
sibling modules only add `impl ViewerApp` blocks, so each child reaches the
struct's private fields without widening any visibility beyond `pub(super)`.

`ViewerApp` owns two `StudySlot`s (datasets A and B). Each slot holds the
loaded study (series, the volume behind an `Arc`, structure sets, doses,
plans, planar images, registrations, records, 4D groups), three `ViewState`s
(per-plane slice, zoom / pan, texture caches), the crosshair, per-ROI
visibility and the segmentation masks. Global state covers window / level,
dose display, tool selection, the registration result, the model folder and
the theme.

Rendering is cache-driven: each view keeps keyed textures for the grayscale
slice, dose colorwash, contour polylines, segmentation overlay and fusion
blend, rebuilt only when their inputs change. Invalidation uses generation
counters bumped by the owning mutation sites — and only by those: a ROI
visibility toggle is part of the contour key alone and leaves the dose and
fusion textures untouched. Repaints are demand-driven; while background jobs
run, the UI polls at 10 Hz.

### Glyphs and the icon

Every non-ASCII character in the interface has to be one of the four fonts
egui bundles — Ubuntu-Light, Hack, Noto Emoji and a small icon font — or it
is drawn as an empty box that no compiler and no test would notice.
`app/glyphs.rs` closes that hole from both ends: `install` appends Hack to
the *proportional* family (arrows, ∩ ∪ ⊕ ⊖ and half a dozen others live only
there, which is why they rendered in the monospaced status bar and as boxes
in menus), and a unit test walks the sources and fails on any character
outside `ALLOWED`, the list verified against those fonts' `cmap` tables.

The application's picture of itself is one file, `assets/rust-dicom-station.png`
(with `.ico` beside it for Windows): `src/icon.rs` loads it as the window
icon of the viewer and the installer, the two `build.rs` compile the `.ico`
into both executables as a resource — which is what Explorer, the task bar,
the start-menu shortcut and *Add or remove programs* read — and the release
workflow copies the PNG into the AppImage as the Linux desktop icon.

### The tool windows

Every secondary window is drawn through `app/detach.rs::tool_window`, which
puts its contents in an *immediate viewport* — a real top-level window of the
operating system, on whichever monitor the user drags it to. Nothing floats
inside the main window, so the viewports always keep the whole of it. Two
rules live in that module: the window's position and size are applied **only
on the pass that creates it** (egui diffs the `ViewportBuilder` against the
one it stored and would otherwise command a dragged window back every frame,
which reads as shaking), and every title goes through `window_title` so the
whole program reads as `Rust DICOM Station: <what this window is>`. The
transient confirmations — *Error*, *Done*, *Rename* — stay inside the main
window, being answers to the last click rather than tools.

The segmentation-type tools — body contour, structure algebra,
auto-segmentation, prompt segmentation, slice propagation, 4D motion — are
different conversations but the same kind of tool, and `app/seg_engines.rs`
makes them alike: one `ToolInfo` per tool gives the glyph, the window title
(`🔬 Auto-segmentation — dataset A`), the menu entry and the small sidebar
button; every window stays open while its run is in flight, the button row
becoming the progress row (device, bar, message, Cancel); the sections come
in the same order (description, the tool's inputs, `Name`, a collapsed
**Options** with the shared `Compute` and `Model folder` rows, the licence
line, `▶ Segment` / `▶ Propagate` / `▶ Contour`, `Close`, status); rows a
tool has no use for are not shown; and results land the same way
(`add_segmentation`), a run that finishes after its dataset was replaced
being discarded with the same message.

## Background jobs

One pattern serves every long operation:

```rust
struct Job<T> { progress: Arc<Progress>, rx: mpsc::Receiver<T> }
```

`Job::spawn` snapshots the inputs, starts a `std::thread` and hands the
worker the progress handle; the UI polls the channel each frame
(`poll_job`): a value lands the result, a disconnect means the worker died
and surfaces as an error. The tools answer with `(slot, Result)`, and
`poll_tool_job` turns a failure into an error dialog — except a
cancellation, which is what the user asked for.

`Progress` (`progress.rs`) holds a message, a fraction, the device label,
an atomic cancel flag and a phase window that maps a sub-step's own 0‥1 onto
its slice of the overall bar. Workers see it through `ProgressSink`, which
the headless examples implement on standard error and the tests with
`Quiet`. Workers use `rayon` internally; the thread-per-job is only the
container. Results are validated on landing where the underlying data could
have changed meanwhile (volume dimensions, frame-of-reference UID).

## The model folder

Every engine downloads its published checkpoint on first use and keeps it,
with the converted `safetensors` cache beside it, under one root. The default
is `models/` in the application's data folder
(`%LOCALAPPDATA%\RustDICOMStation` on Windows,
`~/.local/share/RustDICOMStation` on Linux, `~/Library/Application
Support/RustDICOMStation` on macOS); it can be moved from any tool window
and is persisted as `models_dir` in `viewer_settings.txt`, which lives in
the config folder (the same folder on Windows, `~/.config/RustDICOMStation`
on Linux):

```
<data folder>/models/
  totalsegmentator/<model>/model.safetensors + plans.json
  segvol/pytorch_model.bin, vocab.json, merges.txt, segvol.safetensors
  medsam2/MedSAM2_<variant>.pt + .safetensors
```

`models.rs` owns the layout and the inventory behind the model manager;
`nn/cache.rs` owns the path from a URL to a loaded tensor map
(`RemoteFile::ensure` ▶ `convert_checkpoint` ▶ `load_safetensors`, wrapped as
`ensure_converted`); each engine's `weights.rs` only says which files, which
tensors and under what names. Installations that predate the single root are
migrated at startup: the old `autoseg_models/`, `segvol_model/` and
`medsam2_model/` folders beside the executable are renamed into place. The
Windows installer uses the same default, records `models_dir` only when a
different folder is chosen, and pre-fetches only the Apache-2.0
TotalSegmentator weights.

## Geometry conventions

* Patient space is DICOM **LPS**, `f64` millimetres (`Vec3`).
* Volume voxels are stored `data[k·nx·ny + j·nx + i]` with dims
  `[nx, ny, nz]` = [columns, rows, slices]; `origin` is the **center** of
  voxel (0,0,0); `row_dir` / `col_dir` / `normal` are unit vectors, so the
  code never assumes axis-aligned volumes.
* `Volume::canonical_axes` finds the permutation and flips onto `[S, A, R]`
  by direction cosine; all engines orient through it (MedSAM2 reads the
  in-plane axes the other way round, as SAM 2 does).
* Segmentation masks use the identical index order, so mask ↔ volume
  operations are index-parallel.
* Display: sagittal / coronal view rows run superior → inferior
  (`y = (nz−1) − k`); every producer of view-space pixels honours the same
  flip (asserted by tests).
* Interpolation is trilinear unless stated. The engines keep their reference
  implementations' resampling conventions — scipy `zoom` (nnU-Net), PyTorch
  `nearest-exact` / `align_corners=false` (SegVol), PIL antialiased bicubic
  in 8-bit fixed point (MedSAM2) — because each is validated numerically
  against that reference.

## Error handling and style

`anyhow::Result` with `bail!` / `context` at operation boundaries; missing or
malformed *individual* DICOM attributes never error — safe extraction
helpers return `Option`, and per-file failures inside a batch become
warnings in the UI. Cancellation is an error whose message contains
`progress::CANCELLED`. `rayon` idioms: `par_iter` over independent files /
ROIs, `par_chunks_mut` over rows / slices; sums that decide a threshold or a
normalization stay sequential so a run reproduces itself. Modules open with
a `//!` block explaining the algorithm and its conventions, usually citing
the reference implementation.

Because `lib.rs` makes every module public, `cargo clippy -D warnings`
cannot see an unused `pub` item; the periodic review finds them with a
mechanical scan (every `pub` item referenced nowhere outside its own
tests).

## Dependencies

All pure Rust: `dicom-rs` (DICOM, with `dicom-pixeldata` for decoding),
`egui` / `eframe` (UI over wgpu), `rayon`, `rfd` (file dialogs), `walkdir`,
`anyhow`; for the engines `gemm` (SIMD matrix kernels), `serde_json`, `zip`,
`ureq` (rustls + OS trust store), `safetensors`, and `burn` — always with
its `ndarray` CPU backend, with the wgpu backend added by the cargo feature
`gpu` (default on).

## Testing

Fifteen integration suites plus in-module unit tests run against the same
code paths the GUI uses, with no external data or tooling: the analytic
phantom round trip (**synthetic_study**), simulate → export → reload
(**simulate_export**), rigid and B-spline recovery of known transforms
(**registration**), masks, growing, contours and meshing (**segmentation**),
anonymize → reload (**anonymize**), SEG written and read back voxel for voxel
(**dicomseg**), the body contour on phantoms with couch, chair and mask
(**body**), the archive round trip (**archive**), opening what is not a volume
— RT images and a structure set on their own, a single slice, a folder of RT
objects, and an image series added afterwards (**open_files**) — the DVH
against an analytic
Gaussian phantom (**dvh**), structure algebra (**structops**), and the three
engines assembled and run without a download — a miniature nnU-Net with the
exact checkpoint naming (**autoseg**), and synthesized checkpoints with the
real key names and shapes for **segvol** and **medsam2**, so genuine forward
passes run in CI. **reference** asserts bit-level parity of the
MedSAM2 port with the Python implementation (worst 5.4e-6 relative) from a
dump made by `tools/gen_reference_activations.py`, and skips when the dump
is absent: `MEDSAM2_REF=/tmp/ref cargo test --release --test reference`.
End-to-end runs against the real weights are `#[ignore]`d.

```
cargo test --release
```
