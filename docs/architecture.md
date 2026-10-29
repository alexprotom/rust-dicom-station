# Architecture

An interactive map of the whole program - modules, the data path and the
background-job fan-out - is in
[architecture-diagram.html](architecture-diagram.html): open it in a browser,
click a box to focus it, or use the three guided views along the top. It is
rendered from [architecture-diagram.archify.json](architecture-diagram.archify.json)
and exports to PNG or SVG from the *Export* menu.

## Design philosophy

**One language.** Everything is Rust - DICOM parsing, image reconstruction,
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
`nn/`); the engines, the modules and the tool windows hold only what is theirs.

## Functional overview

What the program does, by category; the [module map](#module-map) says
where each leaf lives.

```
rust-dicom-station
│
├── Application (GUI, egui over wgpu)
│   ├── Window chrome: menu bar, toolbar (W/L, presets, 3D, crosshair, reset, the draw row), status bar
│   ├── Modules panel, fifteen sections from one list (app/modules.rs):
│   │   Image information, Image playback, registration, simulation,
│   │   Image generation (image synthesis: synthetic CT and its evaluation;
│   │   DRR, filed under the planar images), Structure editor
│   │   (insert, edit, combine), Structure auto tools (body contour, the
│   │   engines, the auto-segmentation results), propagation (with transfer
│   │   by relationship), Structure comparison, Structure details, Structure
│   │   motion (the run and its results), Dose display, Dose analysis,
│   │   Dose-volume histograms and Workflows (the run); the structures a section is
│   │   about drawn thicker in the views while it is unfolded
│   ├── Side panel: per workspace a DICOM tree - patient ▶ study ▶ modality ▶ series, with RT
│   │   structures, segmentations, 4D groups, dose, plans, planar images,
│   │   additional objects and reports inside their study - plus spatial
│   │   registrations, records, warnings with an Acknowledge button
│   ├── Views: one row per open workspace (up to four), each of up to eight
│   │   panes chosen under Settings ▸ View layout - the MPR planes, the 3D
│   │   surface scene (put in and taken out by the toolbar's 3D button), two
│   │   planar panes (radiographs, RT images, DRRs), the Additional pane (a
│   │   report as a table: a dose SR, a scanner's protocol page; a secondary
│   │   capture as an image) and the Chart pane (the DVH plot, a motion
│   │   run's charts); those four off in the standard row and not drawn
│   │   without something of their kind; any pane can be taken out of its
│   │   row into a window of its own (⤴ on its bar);
│   │   linked
│   │   viewports, crosshair,
│   │   a workspace with no volume says so in place of the panes and holds back
│   │   the voxel tools;
│   │   zoom / pan / W-L interaction, maximize, per-view caches
│   ├── Tool windows (one shared skeleton; each its own window of the
│   │   operating system), for what is about files and folders rather than
│   │   the images on screen: the workflow editor, PACS, model manager,
│   │   export, save image, anonymizer, generator, test-data download
│   ├── Data tree operations: rename every level; Shift-click ranges; copy / move /
│   │   remove / export the ticked items; create / connect / copy / move / remove
│   │   structure sets and segmentation series; move single structures / segments
│   ├── Workflows: a node editor (menu bar *Workflows*) whose steps are the
│   │   program's own tools; saved as .rdsflow files, run in the background
│   │   or step by step in the viewer, on other input folders, as a batch
│   │   over patient folders, with unchanged steps taken over from the
│   │   last run and independent rows side by side
│   ├── Background jobs: one progress handle, one poll loop
│   ├── Settings: theme, model folder, archive folder, optional modules,
│   │   detached windows (viewer_settings.txt in the config folder)
│   └── Theme: dark / light / system, accent colors
│
├── DICOM
│   ├── Import: directory *or* file-list scan, classification, patient ▶ study ▶
│   │   series tree, merging; a selection that reconstructs no volume (RT images,
│   │   a structure set, a plan) loads as an ordinary workspace with an empty
│   │   volume, and unpositioned image series open as single images
│   │   ├── Volumes: CT, MR, PT, NM, US, OT (parallel decode, compressed syntaxes)
│   │   ├── Planar images: DX, CR, RTIMAGE, MG, XA, RF, PX
│   │   ├── Additional objects: Secondary Capture (whatever Modality it
│   │   │   claims) and any unpositioned image of another modality; RGB
│   │   │   pages kept in colour, overlay planes drawn in
│   │   └── Reports: Structured Reports (the CT dose SR, TID 10011, as a
│   │       table of acquisitions; any other as its content tree) and a
│   │       Siemens page whose MEDCOM OOG text reads back as its table
│   ├── RT objects: RTSTRUCT, SEG (binary / fractional, read and written), RTDOSE,
│   │   RTPLAN / RT Ion Plan, RTIMAGE, REG (matrices and deformable grids, applied
│   │   as the active registration, written back out), RT (Ion) Treatment Record
│   ├── Export: CT + RTSTRUCT + SEG + RTDOSE + RTPLAN with an editable tag table
│   ├── Anonymizer: scan, review every identifying tag, rewrite with a UID remap
│   ├── Patient archive: a local store filed patient ▶ study ▶ instance with text
│   │   sidecars; import with dedupe, listing without opening a file, loading into
│   │   a workspace, derived objects (RTSTRUCT, SEG) sent back under the original UIDs
│   └── PACS server (optional, rds-pacs): the archive served over HTTPS with a
│       pinned self-signed certificate, pairing codes, roles and hashed tokens;
│       every build a client: a mirror per server with an outbox and two-way
│       sync, tasks (workflows run on the server against its studies)
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
│   ├── Results: tables and CSV in the module, charts in the chart pane,
│   │   run-vs-run (A/B) comparison
│   ├── Structure comparison: volumes, centroid offset, Dice, HD95, mean surface distance
│   └── Transfer by relationship: a structure placed in a chosen workspace at its
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
│   │   component selection, thin-anatomy recovery - classically, or guided by
│   │   TotalSegmentator's body network
│   ├── Structure algebra: union / intersection / subtraction / symmetric difference,
│   │   a margin per operand and on the result (six patient directions, exact
│   │   ellipsoids), crop, fill / smooth / prune
│   ├── Surfaces: contour and mask ▶ meshes (scanline fill, surface nets, smoothing)
│   ├── Auto-segmentation - one registry of 81 models: TotalSegmentator (nnU-Net)
│   │   CT v2 / v3, MR, body, 26 task models and 17 licensed ones (crop cascade,
│   │   fold ensembles, post-processing), the nnU-Net v1 tumour models (one fold
│   │   by HTTP range out of each archive), MRSegmentator, lungmask (2-D U-Net),
│   │   MONAI whole body and CT-FM (SegResNet), VISTA-3D and NV-Segment-CTMR
│   │   automatic, the user's own nnU-Net v2 folders; TG-263 naming; CPU (im2col
│   │   + SIMD GEMM) or GPU (burn / wgpu)
│   ├── Prompt segmentation - SegVol: box / point / text, 3-D ViT + SAM-style
│   │   decoder + CLIP text tower, zoom-out / zoom-in passes
│   ├── Slice propagation - MedSAM2 (SAM 2.1 Hiera-T): box drawn in the view,
│   │   include / exclude refinement, memory-bank propagation through the stack
│   └── Interactive segmentation - nnInteractive (points, boxes, scribbles,
│       lassos; AutoZoom and refinement) and VISTA-3D's point mode, prompts
│       drawn in every view, one object refined prompt by prompt
│
├── Neural-network infrastructure (shared by every engine)
│   ├── Model folder: <data folder>/models/<engine>/ (nine engine folders),
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
├── Tests: 27 integration suites + in-module unit tests, synthetic phantom, reference dumps
├── Examples: headless CLIs and probes for the engines (shared examples/common)
├── Tools: the two PyTorch scripts that produce the MedSAM2 reference fixtures
├── Installer: Windows setup (shortcuts, VC++ runtime, optional weight prefetch, uninstall,
│   in-place update of an existing installation, update to the newest release, winget)
└── CI: fmt, clippy -D warnings, tests on Linux + Windows + macOS, CPU-only build; every
    push to main builds the Windows installer and its winget manifests, the Linux
    AppImage and snap, the two macOS disk images, the Android APK and the iOS
    .ipa (packaging/) into a GitHub release, and submits the version to winget
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
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) and
[MRSegmentator](https://github.com/hhaentze/MRSegmentator) on their
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) models,
[lungmask](https://github.com/JoHof/lungmask), and MONAI's `SegResNet`
family as the MONAI whole-body bundle, [CT-FM](https://huggingface.co/project-lighter/whole_body_segmentation)
and [VISTA-3D](https://github.com/Project-MONAI/VISTA) use it; prompt
segmentation re-implements [SegVol](https://github.com/BAAI-DCAI/SegVol);
slice propagation re-implements [MedSAM2](https://github.com/bowang-lab/MedSAM2),
i.e. Meta's [SAM 2](https://github.com/facebookresearch/sam2) fine-tuned on
medical images; interactive segmentation re-implements
[nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) and VISTA-3D's
point head. Papers, weight licences and the numerical validation of each
port are in the per-feature documents ([registration.md](registration.md),
[auto-segmentation.md](auto-segmentation.md), [segvol.md](segvol.md),
[medsam2.md](medsam2.md), [interactive-segmentation.md](interactive-segmentation.md)).

## Module map

Where each function lives. The right-hand tag is the functional category
(**App**, **DICOM**, **Sim**, **Reg**, **4D**, **Dose**, **Seg**, **NN**,
**Core**, **MCP**).

The tree below is the one API the code sees, `crate::<module>` in the
viewer and `rust_dicom_station::<module>` outside it. On disk it is four
crates: the viewer itself (`src/`: `main.rs`, `bin/`, `lib.rs`, `app/`,
`icon.rs`, `gfx_wgpu.rs`), `crates/rds-domain/src/` with every other module
listed under `src/` here (so that an edit in `app/` recompiles the viewer
alone), and the two crates listed under `crates/` at the end. `lib.rs`
re-exports the three below it module by module, under these paths.

```
src/
  main.rs           entry point: opens the eframe/wgpu window, retrying the
                    other graphics backends when one will not start              App
  bin/rds-mcp.rs    the MCP server executable (cargo feature `mcp`): reads
                    mcp.toml, serves the tools over standard input and output    MCP
  lib.rs            library root - every module is public (the domain, core
                    and engines crates' re-exported by name), so the
                    integration tests and the examples drive the same code as
                    the GUI
  gfx_wgpu.rs       the backend choice of gfx.rs in the window library's
                    wgpu types (the domain crate does not link eframe)           App
  par.rs            parallel sums in a fixed order (ordered_fold): the same
                    result on every run and every thread count                   Core
  models.rs         the model folder: root, per-engine sub-folders, migration,
                    the inventory of every downloadable model                    NN
  settings.rs       persisted preferences and the config / data folders - the
                    machine-wide defaults the installer writes, then the user's
                    own file on top; the hooks the mobile front ends register
                    (settings::android dirs, settings::ios::Places, and
                    settings::clipboard: the system clipboard where the window
                    library has none, behind the viewer's Paste buttons)        App
  gfx.rs            which graphics backend to draw and compute with: the
                    settings key, the environment override, and the order to
                    fall back through when one will not start                    App
  archive.rs        the station's PACS: storage folders (the first the main
                    one), the tree of cohorts / patients / studies / series,
                    importing, selections (Pick) as files, names, cohorts,
                    moving files within a storage folder, removal; lookups
                    by UID through the catalog (what the PACS server
                    answers through)                                           DICOM
    catalog.rs      rds-pacs-catalog.json: one file per PACS, every file read
                    once (header -> patient / study / series), the cohort
                    rule for folders, the person's names and memberships;
                    lock file + rename for two writers; old sidecars removed;
                    a rescan that finds nothing new writes nothing; an
                    Importer files several folders under one update          DICOM
  audit.rs          the call log both servers write (data/<server>/audit-*.log)
                                                                                 App
  pacs/             the PACS server and its client: the archive served to
                    other stations over HTTPS (docs/pacs-server.md)              PACS
    protocol.rs       the JSON both sides share; roles; every struct
                      #[serde(default)] so versions read each other
    client.rs         Remote: ureq with the certificate pin (a rustls
                      verifier), listing, bundles, uploads, tasks, operator calls
    servers.rs        the paired servers (pacs-servers.json, owner-only)
    mirror.rs         a server's copy on this device: an archive under
                      pacs-mirror/<id>/, the outbox, pull / send / sync as set
                      differences of SOP Instance UIDs
    config.rs         pacs.toml (read and written by the viewer too)
    local.rs          the server on this machine: its folders, running.json,
                      the local operator's token, starting it detached
    tls.rs            (feature pacs-server) the self-signed certificate,
                      rustls with the ring provider
    auth.rs           (feature pacs-server) pairing codes, hashed tokens,
                      roles, rate limits
    server.rs         (feature pacs-server) the routes, axum over tokio
    tasks.rs          (feature pacs-server) the queue and runner over
                      workflow::graph::exec, bindings, the templates

  app/              egui application, split by concern; every submodule is a
                    further `impl ViewerApp` block, so the struct and its state
                    stay in one place while the behaviour is grouped:            App
    mod.rs            ViewerApp's struct, construction, the per-frame driver;
                      the types it holds live with the files that use them
    slot.rs           StudySlot (a workspace) and ViewState (a view), with the
                      prepared-structure caches (RoiPrep, RoiPreps)
    theme.rs          theme-dependent colors
    glyphs.rs         the font stack (Hack as the last proportional fallback)
                      and the test that fails on a glyph egui cannot draw
    chrome.rs         menu bar, toolbar, status bar, help
    widgets.rs        the small widgets every window reaches for: buttons
                      with a tooltip, the glyph button, the index and
                      structure pickers
    detach.rs         every tool window as a window of the operating system
                      (immediate viewport), titled and placed alike
    pane_windows.rs   a pane of a row in a window of its own: the bar's
                      ⤴ / ⤵ (detached_panes), ⛶ maximizing that window, the
                      windows drawn after the central area through
                      tool_window and the same pane function as the row
    pick.rs           the one door to a file dialog: the system dialog on the
                      desktop, an egui folder browser on Android and iOS, and
                      the answer handed to a continuation either way
    workspace_pick.rs which workspace an action is for, and the rule every
                      such question follows (the open ones plus one new
                      letter): File > Add DICOM folder / Add DICOM file(s) /
                      Clear workspace are submenus, and the file dialog comes
                      after the answer
    form.rs           the two-column parameter form every module and tool
                      window is built from, so labels and controls line up
                      within a section and between them
    panels.rs         both edge panels: the shared show / hide machinery, the
                      shared row widgets, and the right panel's list of
                      switched-on modules
    modules.rs        the modules as one list: what the viewer draws for each
                      settings::Module (ModuleUi: title, header salt, the
                      modules_open word, menu group and tooltip), the one
                      dispatch to its section (module_section), and Focus,
                      the structures the unfolded sections are about, drawn
                      thicker in the views
    data_tree.rs      the left panel's Data tree per workspace: patients,
                      studies, modality nodes, series rows, transfer menu
    tree_model.rs     the tree as data (PatientNode / StudyNode, the layout,
                      cached per study), the arrow-key walk
    fourd_tree.rs     the 4D groups in the tree and what their menus do
    tree_menus.rs     the set, destination and item context menus
    tree_*.rs         one file per tree section: structures, segmentations,
                      dose, plans, planar images and additional objects (one file,
                      a section per kind), reports, registrations, records
                      and warnings; planar images, additional objects and
                      reports, like dose and plans, are categories of a study
    dose_display.rs   the Dose display module: how the displayed dose is
                      drawn (colorwash / isodose, opacity, threshold, the
                      isodose ladder), one setting for every workspace
    reg_panel.rs      the Image registration module: method, region, parameters,
                      landmarks, the run (against another workspace or every
                      phase of a 4D group), the analytics, the vector field
    reg_shape.rs      its Align by structures sub-section: the structures
                      both images carry, paired and weighted, the run, the
                      per-structure report
    reg2d3d.rs        its 2D/3D registration sub-section: the radiographs and
                      their geometry, the isocentre, the region, the method,
                      the run, the result rows, the nudge, the positioned
                      copy and the files; the setups in the data tree; the
                      isocentre, markers and outlines drawn over the
                      radiographs in the planar panes
    matrix_edit.rs    the hand-typed 4 x 4 transform, as a planning system
                      shows one: the sixteen numbers, use
                      / identity / invert / from the result, clipboard in
                      and out. Shared by registration, propagation and
                      transfer by relationship
    views.rs          central MPR viewports: cache refresh (one_view), the
                      frame, the overlays and the pointer, each its own method
    d3.rs             the live 3D structure scene, the row's Scene3d pane (its
                      state still D3Window, made when the pane is first
                      drawn and dropped when it leaves the row)
    planar.rs         the planar panes: a workspace's DX / CR / RTIMAGE images
                      and DRRs in its row (Planar 1 and 2) and its additional
                      objects in the Additional pane (the third pane view, one
                      kind of image per pane); each holds a stack compared by
                      fade, checkerboard, difference, edges or flicker,
                      with a window per image, zoom and pan; showing an image
                      from the tree puts its pane into the row
    report_view.rs    a report in the Additional pane: its facts, lines and
                      tables, the page behind a toggle
    chart_pane.rs     the Chart pane, modelled on report_view.rs: the chart of
                      whichever module shows one (ChartSource: the DVH, a
                      motion run), kept per workspace, with its own bar
                      (save this pane, maximize, close)
    tree.rs           workspace-tree copy / move / remove with reference chains
                      (TreeSel, TreeAction and the object references)
    rename.rs         renaming every level of the data tree
    sets.rs           structure sets and segmentation series as tree nodes:
                      create, connect, copy / move / remove, move single
                      structures / segments (contour ⇄ mask conversion)
    jobs.rs           the job plumbing (Job::spawn, poll_job, poll_tool_job,
                      poll_all_jobs, called once a frame) and the loading,
                      simulation, export, generator, anonymizer and
                      auto-segmentation job starts
    dialogs.rs        the auto-segmentation section and its Results fold,
                      generator, anonymizer, error dialog
    export_win.rs     the export window: the selection tree over every workspace,
                      the name and UID editors, the RTSTRUCT / SEG radios
    seg.rs            interactive segmentation state machine, mask ▶ RTSTRUCT,
                      landing an auto-segmentation result
    seg_edit.rs       the Edit section when the subject is a segmentation:
                      the Structure / Segmentation switch, tidy, grow and
                      shrink in millimetres, clear, delete, and mask to RT
                      structure - structops, morphology and mask_to_roi
                      given a place in the editor
    contour_edit.rs   the contour tools' state machine: the ROI under the
                      tools, the working stack, drawing and nudging, contour
                      undo, the interpolation preview
    struct_tools.rs   the Structure editor module (the Insert structure
                      section - empty structure, point, the generators in HU
                      or SUV, shape, dose, field of view - and the Edit
                      structure section: interpolation, per-slice copy /
                      paste / delete / clear / thin, tidying, transforms, ROI
                      type, the derived recipe) and the toolbar's draw row
                      (the nine tools and the options of the one in hand)
    derived_app.rs    derived structures in the app: resolving operands by
                      name, the status cache, re-evaluation as a job, the
                      override rule
    livewire_app.rs   the live-wire tool: the slice's cost image and the
                      current anchor's tree, cached between frames
    poi.rs            points of interest: create at the crosshair or at a
                      structure's centre, localize, move, the localization
                      point
    stats.rs          the Structure details module: the table, its Dice
                      reference and its CSV; a row selects and centres on
                      its structure
    seg_engines.rs    what the engine sections share: names and glyphs, the
                      workspace A / B row, device / model-folder / licence /
                      progress rows, result landing, the "still the same
                      workspace" check
    auto_tools.rs     the Structure auto tools module: one workspace row and
                      the body contour, auto-segmentation, prompt
                      segmentation, slice propagation and interactive
                      segmentation sections
    body_win.rs       the body-contour section
    combine.rs        the structure algebra as the editor's Combine structures
                      section: operands, margins, the recipe
    prompt_seg.rs     the prompt segmentation section and worker (SegVol)
    box_seg.rs        slice propagation: the box drawn in the viewport, the
                      preview / refine / propagate loop, the resident session (MedSAM2)
    interactive_seg.rs interactive segmentation: prompts drawn in every view
                      (point, box, scribble, lasso), the prompt queue, the
                      resident session and network (nnInteractive, VISTA-3D
                      points), undo, new object, refine an existing mask
    run_report.rs     what a run leaves behind, as two tables rather than a
                      paragraph: per destination the metric, the cost and the
                      anchor's Dice, per structure the three volumes and the
                      change between them, plus the tab-separated form the
                      clipboard button writes
    propagate_win.rs  the Structure propagation module: onto another workspace,
                      or onto every phase of a 4D group through workflow::group
                      or workflow::anchored (transforms kept for the next run)
    transfer.rs       its By relationship fold: transfer by relationship
    motion.rs         the Structure motion (4D / ITV) module and its Run fold;
                      the pipeline itself is workflow::motion
    motion_results.rs its Results fold: the run picker, the targets, tables,
                      correlations, QA, CSV, run-vs-run comparison; the
                      charts it draws into the chart pane
    compare.rs        the Structure comparison module: volumes, centroid
                      offset, Dice, HD95, MSD, the least-squares fit
    dvh.rs            the Dose-volume histograms module (DvhModule): pickers,
                      the metrics table, constraints, export; the plot it
                      draws into the chart pane
    play.rs           playback: the ▶3D / ▶4D buttons on the viewports and on
                      the 3D pane, the Dynamic dose log's transport, the frame
                      clock, the 4D phase volume cache and its budget, the light
                      phase switch that keeps the view, and the Image playback
                      module that carries the settings
    snapshot.rs       File > Save image: one row of the central area or both,
                      cut out of egui's screenshot, resampled to the chosen
                      DPI and written as a PNG (with pHYs) or a JPEG (with
                      the JFIF density)
    record.rs         saving a run: armed first, it takes the next run that
                      starts and ends after one full cycle. The pane, frame
                      by frame, out of egui's own screenshot, written as an
                      animated GIF or a numbered PNG sequence. Pure Rust, no
                      other program; the run's clock waits for each picture,
                      so the file holds every frame the run played
    img_info.rs       the Image information module: the geometry, sampling and
                      acquisition of the displayed series (imginfo), what wants
                      a second look, and what two workspaces disagree about
    dose_est.rs       the Dose analysis module: the dose metrics table of
                      the ticked structures against one dose (physical /
                      effective), recomputed whenever they change
    image_gen.rs      the Image generation module: one workspace row over the
                      Image synthesis and DRR sections
    drr_win.rs        its DRR section: geometry, projectors, comparison; every
                      rendering filed under the planar images and shown in the
                      row's planar panes
    pacs_win.rs       the PACS window: menu bar (Settings: storage folders,
                      PACS server page), the row of sources, the local tree
                      and its context menu (rename, cohorts, move files,
                      remove), import, load, send back
    pacs_remote.rs    the PACS window on a paired server: its tree, pull
                      (patients, studies, single series), load, send, sync,
                      upload, the Tasks tab, Add server / pairing
    pacs_tree.rs      the tree both draw: rows drawn only where visible,
                      selection with Ctrl / Shift and the keyboard, filter
    pacs_server_win.rs  PACS ▶ Settings ▶ PACS server (a page of the PACS
                      window): start / stop rds-pacs, the
                      connection details, pairing codes, paired stations,
                      pacs.toml, the activity log
    models_win.rs     the model manager window
    testdata_win.rs   the Download test data window over testdata.rs
    workflow_edit.rs  the workflow editor: the egui-snarl canvas (typed pins,
                      wires, the add-a-step menus), the palette, the
                      parameter forms, load / save / recent, the Workflows
                      menu's actions; undo / redo (JSON snapshots), the
                      clipboard (workflow fragments), frames, the map,
                      keyboard nudging
    workflow_run.rs   the Workflows module: the workflow picked (load, last
                      saved, examples; New / Edit open the editor), inputs,
                      results folder, background or
                      step by step, reruns, parallel rows, memory, batches;
                      the run as a Job, its events polled each frame, each
                      shown step put in a workspace (the user's study there
                      parked, and put back on request), the results

  loader.rs         directory / file-list scan, classification, parallel volume
                    loading, workspace merging, safe DICOM element helpers         DICOM
  dicomfile.rs      the one way a file is opened: the standard reader, plus
                    the encoding sniffer for data sets written with no file
                    meta group (no preamble, no DICM)                            DICOM
  imginfo.rs        the geometry, sampling and acquisition of one series, read
                    back out of its slice headers: spacing, thickness, gaps,
                    uneven positions, tilt, frame of reference, kV / mAs / kernel,
                    each row carrying the reason it wants attention              DICOM
  render.rs         window / level, dose colorwash, marching-squares isodose,
                    contour / plane intersection                                 Core
  morphology.rs     binary-mask geometry in millimetres: exact anisotropic
                    distance transform, erode / dilate / open / close,
                    ellipsoidal margins, components, hole filling, the
                    extruded-equipment test, box-blur smoothing; the mask
                    helpers everything shares (count, extent, surface walk)     Core
  sop.rs            the SOP Class UIDs read, written or recognised, their
                    families (non-image objects, SR, SC) and listing names       DICOM
  rtstruct.rs       RT Structure Set parsing, with the fallbacks for sets that
                    lack the ROI sequence or misplace the frame of reference     DICOM
  rt_surface.rs     the surface-based volume of an RT structure: a closed
                    triangle surface from planar contours (keyholes, strips
                    between slices, smooth end caps, seams closed) and the
                    volume it encloses                                           DICOM
  dicomseg.rs       DICOM Segmentation: the segmentation-series model, SEG
                    reading, resampling between lattices, the SEG writer         DICOM
  rtdose.rs         RT Dose parsing + trilinear patient-space sampling           DICOM
  rtplan.rs         RT Plan / RT Ion Plan parsing                                DICOM
  extras.rs         DX / CR / RTIMAGE planar images (an RT Image's geometry
                    read into a camera), additional objects (PlanarKind:
                    secondary captures), overlay planes, REG, RTRECORD          DICOM
  report/           reports: the SR content tree (sr.rs), the CT dose SR as
                    a table of acquisitions (dose.rs, TID 10011 by concept
                    code), a Siemens page's MEDCOM OOG text laid out as its
                    table (siemens.rs), encapsulated PDFs
                    drawn page by page with hayro (document.rs), and
                    Report / Table                                              DICOM
  dicom_export.rs   DICOM writer: the object builders (CT slice, RTSTRUCT with
                    its image reference chain, SEG, RTDOSE, RTPLAN, Deformable
                    and rigid Spatial Registration, RT Image with its
                    geometry) and the patching file copier                     DICOM
  export.rs         what one export run is: the workspace / patient / study /
                    series selection tree, every editable name and UID, the
                    RTSTRUCT / SEG choice, and the runner that keeps the
                    references between the written objects resolvable          DICOM
  anonymize.rs      interactive DICOM anonymizer engine                          DICOM
  gen_test_data.rs  synthetic RT phantom study generator                         Sim
  testdata.rs       the bundled real patient (data-test/) fetched from GitHub:
                    git trees listing, raw downloads, resumable                  Sim
  simulate.rs       known-transform study generator (registration QA)           Sim
  drr.rs            DRR: IEC cone-beam geometry, Siddon exact tracing and
                    ITK-style interpolating ray-casting, the kV and MV
                    attenuation models                                           Sim
  iec.rs            the IEC 61217 chain (fixed, gantry, receptor, support,
                    table top), the eight patient positions and upright, the
                    couch readout of a pose                                      Sim
  projection.rs     the pinhole camera every geometry ends in (and a 3 x 4
                    projection matrix), the CT as attenuation with coarser
                    copies, the posed index-space ray-cast                      Sim
    gpu.rs          the same ray-cast as a WGSL compute shader on the
                    inference device (feature gpu)                              Sim
  rooms.rs          imaging-rooms.json: calibrated views of room-mounted kV
                    pairs                                                       Sim
  setup.rs          patient setup from radiographs: the request made from a
                    study (images, geometry sources, isocentre, markers), the
                    result as the study keeps it, the DRRs at a pose, the
                    outlines, the positioned copy, the files                   Reg
  registration.rs   parameters, transforms (rigid, B-spline, RBF, field,
                    composite), region masks, pyramid, samplers, the
                    initialisation (identity, centres of gravity, two points),
                    dispatch                                                    Reg
    elastix.rs        stochastic sampling + ASGD, rigid and B-spline stages
    plastimatch.rs    dense analytic gradient, bending energy,
                      Mattes mutual information, L-BFGS
    landmark.rs       thin-plate / Gaussian / Wendland RBF warp, dense solve
    analysis.rs       6-DOF Procrustes fit, displacement and Jacobian statistics
    dvf.rs            vector-field sampling and its view-plane / 3-D glyphs
    shape.rs          registration by structures: signed distance maps,
                      surface points, the symmetric Gauss-Newton rigid fit,
                      the per-structure B-spline refinement on the maps
    optim.rs          Powell with Brent line searches, Nelder-Mead, L-BFGS:
                      bounded, deterministic
    parzen.rs         the cubic B-spline Parzen window (shared by the 3D and
                      the 2D mutual information)
    metric2d.rs       2D images, pyramids and masks; NCC, gradient
                      correlation, gradient difference, pattern intensity,
                      local NCC, Mattes MI
    projective.rs     2D/3D registration by intensity: polarity, the in-plane
                      search and the grid, coarse to fine over posed
                      renderings, the curvature check, the readouts
    markers.rs        2D/3D registration by markers: detection in the CT and
                      on the images, pairing by layout, the LM pose,
                      triangulation and the moved-marker check, the hybrid
  propagate.rs      structures across a registration: pull-back with a cached
                    mapping lattice; each result kept as a BoxedMask (its
                    bounding box only), unpacked to a full mask on demand       Reg
  fourd.rs          4D sub-studies: phase recognition, ordered groups
                    (phases + AVG / MIP), custom-group rules                     4D
  motion.rs         motion arithmetic over phases: centroids, peak-to-peak,
                    drift, Pearson r with p-values, Dice / HD95 / MSD overlap,
                    the closest-point rigid offset between two surfaces,
                    ITV unions, the motion report + CSV                          4D
  dvh.rs            dose-volume histograms: sampling, curves, metrics,
                    protocol constraints, CSV                                    Dose
  derived.rs        the recipe a derived structure carries: the expression,
                    its storage in ROI Description, the three statuses and
                    the fingerprint they are computed from                    Seg
  generate.rs       structures out of a grey-level window, a shape, a dose
                    level or the reconstructed field of view                  Seg
  livewire.rs       the edge-following cost of one slice: the Mortensen-
                    Barrett terms, the shortest-path tree of an anchor, the
                    trained histogram, and the ring snapped onto a ridge      Seg
  templates.rs      structure templates: the names, types, colours and
                    recipes of a set, as JSON in the data folder              Seg
  contours.rs       planar contours as an editable representation: rings and
                    even-odd regions, the local supersampled boolean, the
                    slice stack, contour ⇄ mask ⇄ RTSTRUCT (one padded
                    field traces every mask), tidying, transforms, slice
                    interpolation                                               Seg
  segmentation.rs   voxel masks: brush, geodesic grow, undo, overlays,
                    label map ▶ segmentations, mask ⇄ RTSTRUCT contours          Seg
  structops.rs      structure algebra: the four boolean operations, margins,
                    crop, fill / smooth / prune, over masks on one lattice       Seg
  bodymask.rs       the body / EXTERNAL contour, classically or guided by the
                    body network                                                 Seg
  sct/              synthetic CT as a series of the study: the body mask, the
                    engine, the DICOM files (dicom_export::
                    write_derived_image_series) read back for merging; eval.rs
                    the comparison with a CT (MAE / ME / RMSE per HU class,
                    PSNR, SSIM, WET through an HU-to-RSP table, organ Dice)     Sim

  workflow/         the pipelines without a window around them, shared by the
                    viewer's module sections and the MCP server                  4D
    session.rs        the headless core under the MCP session and the
                      workflow runner: Volumes (a budgeted LRU cache of image
                      volumes, shared by every pipeline of a run or session),
                      UIDs, filing structures into a study under one
                      name-clash rule across phases (file_items)
    params.rs         the choices the dialogs, the MCP tools and the workflow
                      steps share (variant, device, method, landing, name
                      clash), with the names files and tools use
    select.rs         structures by name over RTSTRUCT ROIs and segmentation
                      series; contour or mask onto any lattice
    motion.rs         the per-phase 4D pipeline (register ▸ propagate ▸
                      measure ▸ ITV), moved out of the app's motion module
    group.rs          one volume onto every phase of a group, one registration
                      per phase, transforms reusable; moved out of propagate_win.rs
    anchored.rs       the same anchored on a structure contoured on both sides
                      (a cardiac CT onto a 4DCT by the heart): centroids, a rigid
                      fit on the structure, a local refinement, Dice as the check
    graph/            user workflows: the steps above (and the engines, loading,
                      export) as nodes of a graph the user draws and saves
                      (docs/workflows.md)
      mod.rs            the document: Workflow, Node, Link, Frame; check, run
                        order, folder templates, fragments for the clipboard
      catalog.rs        every kind of step: typed ports, parameter structs
                        (serde), what the canvas says about each
      exec.rs           the runner: wire values as references into the run's
                        copy-on-write state (Arc per dataset), events and the
                        show-and-wait handshake, step fingerprints and the
                        StepCache for reruns, parallel rows (scoped threads,
                        merged in run order), batches, the outcome, the run
                        folder's own files
      nodes.rs          what each step does and how it files its results
      nodes/more.rs     the archive, anonymize, prompt, combine, rename,
                        transfer, copy-to-phases, DVH, dose, archive-import
                        and DRR steps
      nodes/setup.rs    the 2D/3D registration and positioned copy steps
      store.rs          the user's workflow folder, load / save, the recent
                        list, the examples compiled in

  mcp/              the MCP server behind rds-mcp (cargo feature `mcp`)         MCP
    config.rs         the operator's mcp.toml: roots, output folder, PHI policy
    phi.rs            the gate (which workspaces still name their patient) and
                      the redactor every outgoing string passes
    session.rs        open workspaces, transforms, reports, and their handles
    tools/            the tools as plain functions with schema-deriving
                      argument structs: session, segment, register, setup
                      (2D/3D), fourd, analysis, output, archive (the
                      station's PACS listed under aliases, a study described
                      from its headers, opened through the gate)
    prompts.rs        the heart_target_propagation prompt, the doc resources
    audit             the call log (the shared crate::audit)
    server.rs         the rmcp glue: transport, progress, cancellation, _async jobs
  mesh3d.rs         contour / mask ▶ surface meshes (scanline fill, surface
                    nets, Laplacian smoothing)                                   Seg

crates/            the workspace's library crates; lib.rs re-exports their
                   modules under the paths they had in src/ (crate::volume,
                   rust_dicom_station::medsam2, ...)
  rds-domain/src/   every module listed under src/ above but the viewer's own
                    (app/, main.rs, bin/, icon.rs, gfx_wgpu.rs); the cargo
                    features gpu, mcp and pacs-server, which the root's
                    forward to
  rds-core/src/
    progress.rs     the one progress handle + ProgressSink, Quiet, Stderr         Core
    volume.rs       3D volume, patient-space geometry, slice extraction,
                    trilinear sampling, canonical [S, A, R] axes                 Core
    geometry.rs     minimal 3D vector math (Vec3, f64, patient mm)               Core
  rds-engines/src/  the inference engines and their shared plumbing, a crate of
                    their own so burn's wgpu backend (cargo feature `gpu`) is
                    compiled once rather than with every change to the viewer
    nn/               shared neural-network infrastructure - nothing in here
                      knows about a particular architecture                        NN
      cache.rs          RemoteFile download (length-checked, resumed from the
                        .part with Range + If-Range), torch checkpoint ▶
                        safetensors conversion (ConvertSpec), the
                        converted-weight cache read tensor by tensor
      pickle.rs         native PyTorch checkpoint (.pth / .pt / .bin) reader
      device.rs         DevicePref (Auto / GPU / CPU), the inference device
                        opened adapter by adapter and self-tested (its wgpu
                        device and queue kept for the viewer's own shaders),
                        the backend-panic guard
      params.rs         shape-checked view of a loaded state dict
      half.rs           binary16 ↔ binary32 conversion
      tensor.rs         Mat [rows, cols] and Act [c, d, h, w]; transposed conv
      linalg.rs         gemm-backed linear / matmul, layer norm, softmax, activations
      attention.rs      multi-head attention, optionally causally masked
      fastconv.rs       burn convolutions that hand the CPU backend's work to the
                        U-Net GEMM kernels (conv 2-D / 3-D, SegResNetDS's
                        stride-2 transposed conv as eight sub-pixel convs)

    zoo/              the registry of automatic models: AutoModel (every family
                      behind one key, label, group, modality, licence, class
                      table, download, run), Licence, Modality, Family, and the
                      TG-263 name table                                            Seg
    autoseg/          the nnU-Net family (pure-Rust TotalSegmentator,
                      MRSegmentator)                                               Seg
      mod.rs            public API: run(task), the result, organ hits
      task.rs           a model as data (parts with label tables, folds, step,
                        crop rule, post-processing); the crop cascade
      total.rs          the total / total_v3 / total_mr / body / MRSegmentator tasks
      tasks.rs          the 26 catalogue tasks, generated from upstream's tables
      post.rs           body and vertebrae_pp post-processing
      classes.rs        117-class tables (v2, v3), organ colours by name
      config.rs         nnU-Net plans.json parsing, both formats, inherits_from
      weights.rs        which models exist, where they are published, the
                        release-zip unpacking (folds, two trainings per zip)
      kept.rs           KeepNetworks: built networks kept across the phases
                        of one run while the guard lives, released after
      cpu.rs            CPU conv engine (im2col + SIMD GEMM conv3d with any
                        padding, transposed convs, norms)
      net.rs            PlainConvUNet / ResidualEncoderUNet assembly (any input
                        channel count) + CPU forward
      gpu.rs            wgpu forward via burn (cargo feature `gpu`)
      preprocess.rs     resampling to the model grid and back (scipy conventions)
      infer.rs          nnU-Net and MONAI sliding windows, streaming argmax in
                        strips under a memory budget

    unet2d/           lungmask: 2-D U-Net (burn, and a burn-free CPU path),
                      body crop and slice preparation, 3-D clean-up             Seg
    segresnet/        MONAI SegResNet and SegResNetDS (MONAI whole body, CT-FM),
                      MONAI's transforms (Spacing, CropForeground, ...)          Seg
    vista3d.rs        VISTA-3D automatic mode (class head over SegResNetDS2), the
                      bundle's preprocessing; vista3d/points.rs the point head
                      and the point-window pipeline, vista3d/session.rs the
                      interactive session                                         Seg
    nninteractive/    nnInteractive: model files and loading, the inference
                      session (prompt channels, AutoZoom, refinement), the
                      resampling / pooling / box-cover operations it is built
                      from                                                       Seg
    synth/            synthetic CT (pure-Rust TotalSynth / ImpactSynth): the
                      apps and their checkpoints (mod.rs), the checkpoint's
                      three namings and the batch-norm fold (layout.rs), the
                      UNet++ on the CPU (cpu.rs) and over burn for the GPU
                      (net.rs), KonfAI's prediction recipe to the rounding
                      (pre.rs), folders of the user's checkpoints (custom.rs) Sim

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

    medsam2/          slice propagation (pure-Rust MedSAM2 and Efficient MedSAM2); every module is
                      generic over a `burn` backend, so one implementation runs
                      on GPU and CPU                                               Seg
      weights.rs        the seven published variants, load(), the research-only licence
      layout.rs         the checkpoints' tensor layouts (both networks) and their checks
      config.rs         the fixed dimensions: 512 input, 7 memories, 16 pointers
      ops.rs            the tensor helpers the port needs on top of burn
      layers.rs         conv, layer norm, linear (kept transposed), MLP
      hiera.rs          Hiera-T image encoder: 4 stages, windowed attention
      neck.rs           FPN neck to 256 channels + the sine position encoding
      vitdet.rs         Efficient MedSAM2's encoder: EfficientTAM's plain ViT
                        (tiny / small) and its one-level neck
      prompt.rs         SAM's prompt encoder: points, boxes, mask prompts
      decoder.rs        two-way transformer, hypernetwork mask filters, IoU and
                        object-presence heads
      sam.rs            the SAM head assembled: prompt ▶ masks for one slice
      memory.rs         memory encoder: mask downsampler + ConvNeXt fuser
      memattn.rs        memory attention: 4 layers, 2-D axial RoPE
      model.rs          the whole network (which one the checkpoint says), and the
                        two ways a slice is conditioned
      track.rs          the memory bank and the slice-to-slice state machine
      infer.rs          one-slice preview, the two propagation passes, the slice
                        range, thresholding, largest-component cleanup
      preprocess.rs     window, quantize to u8, orient; the prompt's and the
                        mask's way between the study grid and the network's
      resample.rs       PIL's resampling kernels, incl. 8-bit fixed-point arithmetic
      engine.rs         backend choice, the encoded-slice cache, the one call
                        the user interface makes

tests/             the integration suites (see Testing); common/ holds the
                   4D phantom fixture the workflow and MCP suites share, and
                   ops_ref.rs, the naive reference kernels behind the MedSAM2
                   op fixture
examples/          seg_cli (autoseg_cli its older name), interactive_cli,
                   autoseg_probe, body_cli, segvol_cli, segvol_probe,
                   medsam2_cli, medsam2_probe, gen_ops_fixtures (writes the op
                   fixture), workflow_cli (runs a saved workflow headless);
                   common/ holds what the CLIs share
packaging/         everything that turns the viewer into an installable package,
                   one folder per platform (see packaging/README.md); nothing in
                   it is part of a root cargo build
  windows/
    installer/     the Windows installer, its own workspace (see its README);
                   built by the release workflow
    winget/        where rds-pack --winget writes the manifests (git-ignored)
    windowsstore/  prepared for a Microsoft Store submission; nothing built yet
  linux/
    appimage/      AppRun, the desktop entry and build-appimage.sh; run by
                   the release workflow
    flatpak/       the Flathub manifest, cargo-sources.json, desktop entry
                   and metainfo (docs/flatpak.md)
    snap/          snapcraft.yaml and the desktop entry (docs/snap.md);
                   copied to <repo>/snap at build time, where snapcraft
                   looks for it
  macos/           the macOS packaging and nothing else - no crate, no code:
                   the .app bundle's plist, the hardened-runtime
                   entitlements, the script that builds bundle and disk
                   image for one architecture, and the Homebrew cask
                   generator (docs/macos.md); run by the release workflow
                   once per architecture
  android/         the Android front end, its own workspace: android_main
                   over the same ViewerApp, the manifest, the icons and the
                   packaging script (docs/android.md); built by the release
                   workflow
  ios/             the iOS / iPadOS front end (iPad and iPhone), its own
                   workspace: a main that hands the same ViewerApp to
                   UIKit, the zoom that fits the desktop layout into a
                   small screen, the safe-area strips, the system's folder picker (security-scoped
                   bookmarks, registered as settings::ios::Places), the
                   plist, the icon, build-ipa.sh and the simulator smoke
                   test (docs/ios.md); built by the release workflow
```

## UI architecture

`ViewerApp` is defined in `app/mod.rs`; the types it holds sit with the files
that use them (`slot.rs`, `d3.rs`, `tree.rs`, `sets.rs`, `reg_panel.rs`,
`jobs.rs`) and are imported back into `mod.rs`, so `use super::*` reaches them
from every sibling. The sibling modules only add `impl ViewerApp` blocks, so
each child reaches the struct's private fields without widening any
visibility beyond `pub(super)`.

`ViewerApp` owns [`settings::MAX_WORKSPACES`] `StudySlot`s - four, workspaces
A to D - and a flag per slot saying which of them are on screen (A always is;
a workspace appears when something is loaded into it and goes when it is
emptied). The letters are fixed: closing B leaves C where it is, because
registrations, propagations and window titles all name workspaces by letter.
`open_slots`, `other_open` and `copy_targets` are the three questions the rest
of the program asks about them - which are on screen, which one to pair this
with, and where a copy may go (the open ones plus one new letter). Each slot
holds the
loaded study (series, the volume behind an `Arc`, structure sets, doses,
plans, planar images, registrations, records, 4D groups), three `ViewState`s
(per-plane slice, zoom / pan, texture caches), the crosshair, per-ROI
visibility and the segmentation masks. Global state covers window / level,
dose display, tool selection, the registration result, the model folder and
the theme.

Rendering is cache-driven: each view keeps keyed textures for the grayscale
slice, dose colorwash, contour polylines, segmentation overlay and fusion
blend, rebuilt only when their inputs change. Invalidation uses generation
counters bumped by the owning mutation sites - and only by those: a ROI
visibility toggle is part of the contour key alone and leaves the dose and
fusion textures untouched. Repaints are demand-driven; while background jobs
run, the UI polls at 10 Hz.

### Glyphs and the icon

Every non-ASCII character in the interface has to be one of the four fonts
egui bundles - Ubuntu-Light, Hack, Noto Emoji and a small icon font - or it
is drawn as an empty box that no compiler and no test would notice.
`app/glyphs.rs` closes that hole from both ends: `install` appends Hack to
the *proportional* family (arrows, ∩ ∪ ⊕ ⊖ and half a dozen others live only
there, which is why they rendered in the monospaced status bar and as boxes
in menus), and a unit test walks the sources and fails on any character
outside `ALLOWED`, the list verified against those fonts' `cmap` tables.

The application's picture of itself is one file, `assets/rust-dicom-station.png`
(with `.ico` beside it for Windows): `src/icon.rs` loads it as the window
icon of the viewer and the installer, the two `build.rs` compile the `.ico`
into both executables as a resource - which is what Explorer, the task bar,
the start-menu shortcut and *Add or remove programs* read - and the release
workflow copies the PNG into the AppImage as the Linux desktop icon.

### The tool windows and the modules

What works on the images on screen is a **module**, a section of the right
panel; what is about files and folders is a **tool window**. The windows
that remain are the workflow editor (its canvas needs the width), the PACS
(with its server page), the downloaded models, *Export DICOM*, *Save image*,
the anonymizer and the two test-data dialogs (generate, download); the
structure comparison, structure details, transfer by relationship,
auto-segmentation results, DVH, structure motion with its results, and the
workflow run were windows once and are sections now, and the 3D window is
gone, the scene being a pane of the row (`PaneKind::Scene3d`) that the
toolbar's *3D* button puts in and takes out.

**The modules are one list.** `settings::Module` (in the domain crate, with
`ALL` and `key()`, the `module_*` word of `viewer_settings.txt`; the nine
older keys unchanged, so an old file still reads; Dose analysis keeps the
key it had as Dose estimation) names the fifteen in
panel order; `app/modules.rs` holds what the viewer draws for each through
the `ModuleUi` trait (a trait because the enum is not the app's): the
title, the header's id salt, the word `modules_open` remembers an unfolded
section by, the group of the *Modules* menu (Images,
Structures/Segmentations, Doses, Workflows) and the menu's tooltip, plus `ViewerApp::module_section`, the
one dispatch from a module to the function that draws it.
`ViewerApp.modules` and `Settings.modules` are a `BTreeSet<Module>`. Adding
a module is one variant of the enum, one arm of each table in `modules.rs`
and its section function.

**Sections act on the views through one facility.** `section_focus`
(`modules::Focus::Structures(Vec<(slot, ItemRef)>)`) is cleared at the top
of every pass of the modules panel and set by whatever is unfolded: the two
structures a comparison measures, the row of the details table under the
pointer, the target and references of a transfer, the DVH curve under the
pointer in the chart, a motion target picked. `focused_rois` hands the
views the RT-structure contours to draw thicker (3 px), the same way, and
through the same call, as the structure the contour tools edit; segments
are not highlighted. Nothing stays lit once its section is folded or the
panel hidden.

**Charts go into a pane.** The two modules that draw charts keep their
pickers and tables in the panel and put the chart in the row's
`PaneKind::Chart` (`app/chart_pane.rs`, modelled on `report_view.rs`: the
panel's background, title and subtitle, a bar of its own - save this pane
as a PNG through the snapshot path, maximize, close - and the content in a
child `Ui`). What it shows is a `ChartSource { Dvh, Motion(run) }` per
workspace (`ViewerApp.chart`), set by `show_chart`, which also puts the
pane into the row with `add_to_row`; with nothing to show the pane is not
drawn, as the Additional pane with nothing in it. A module whose pane was
closed gets a 📈 on its section header to bring it back. With it the row
has eight pane kinds (`MAX_PANES` 8): the three planes, the 3D scene, two
planar panes, the Additional pane and the chart.

**Any pane can leave its row.** `ViewerApp.detached_panes` lists the
`(slot, PaneKind)`s taken out with the ⤴ on their bars (`app/pane_windows.rs`).
`study_row` leaves them out of the row, and `pane_windows`, called once a
frame after the central area, draws each in a `tool_window` through the same
`pane` function the row uses, so the pane keeps all its state. In such a
window ⛶ sends `ViewportCommand::Maximized` to that window instead of
setting `maximized`; the pane's rectangle is not noted for a recording, and
a chart there is not saved as a picture, both being cut from the main
window's screenshot. A pane that has nothing left to show, or left the
layout, goes back on its own; the list is not persisted.

Every secondary window is drawn through `app/detach.rs::tool_window`, which
puts its contents in an *immediate viewport* - a real top-level window of the
operating system, on whichever monitor the user drags it to. Nothing floats
inside the main window, so the viewports always keep the whole of it. Two
rules live in that module: the window's position and size are applied **only
on the pass that creates it** (egui diffs the `ViewportBuilder` against the
one it stored and would otherwise command a dragged window back every frame,
which reads as shaking), and every title goes through `window_title` so the
whole program reads as `Rust DICOM Station: <what this window is>`. The
transient confirmations - *Error*, *Done*, *Rename* - stay inside the main
window, being answers to the last click rather than tools. On Android and
iOS, which allow one window per process, egui draws every viewport as a
window inside the main one and nothing in `detach.rs` has to know.

File and folder dialogs go through `app/pick.rs`. A request names what is
wanted (a folder, files, a file, a file to save as) and carries the
continuation - what to do with the path - as a closure. On the desktop the
operating system's dialog (`rfd`) blocks and the closure runs before the
call returns, which is exactly the `if let Some(path) = dialog()` it
replaced; on Android and iOS, where `rfd` has no backend, the same call
opens a folder browser drawn in egui and the closure runs from the frame in
which the user answers. The browser's model (roots, listing, sorting) is
compiled and unit-tested on every platform. On iOS its roots are the app's
own `Documents` plus the folders the user granted in the system's folder
picker, which lives in the iOS front end and reaches the browser through
`settings::ios::Places`; the browser adds a *+ Folder from Files* button
that asks for one more.

The engine-type tools - body contour, auto-segmentation, prompt
segmentation, slice propagation - are different conversations but the
same kind of tool, and `app/seg_engines.rs` makes them alike: one
`ToolInfo` per tool gives the glyph and the name; every section stays as it
is while its run is in flight, the button row becoming the progress row
(device, bar, message, Cancel); the parts come in the same order
(description, the tool's inputs, `Name`, a collapsed **Options** with the
shared `Compute` and `Model folder` rows, the licence line, `▶ Segment` /
`▶ Propagate` / `▶ Contour`, status); rows a tool has no use for are not
shown; and results land the same way, a run that finishes after its
workspace was replaced being discarded with the same message. Since
2026-09-07 they are not windows but the four sections of the **Structure
auto tools** module (`app/auto_tools.rs`), under one workspace row; the
auto-segmentation *results* list, which appears once per run, is the
*Results* fold at the end of the auto-segmentation section (`dialogs.rs`):
a run that lands switches the module on, opens the panel and unfolds the
module, the section and the fold once, and the fold goes after *Add* or
*Discard*.

Where a result lands and how far a run reaches are the same two questions
for body contour, auto-segmentation and prompt segmentation, so they are
asked by the same two rows (`ToolOutput`, `scope_row`, `output_rows`):
*Output* - segments, RT structures (in a chosen or a new structure set),
or both - and, when the displayed series is a phase of a 4D group, *Run
on* - that series or every phase. A run on the displayed series lands
through `land_masks`; a run over the phases goes through `run_on_phases`
on the worker (each phase loaded in turn, the displayed one from memory,
the engine given a `Progress` whose `set_outer` window and message prefix
make it one slice of the bar) and lands through `land_phases`, the same
landing *Copy to each phase* uses. Both hand back `(PhaseInfo, R)` pairs,
one per volume the engine saw, so a tool's `on_*_done` tells the two
apart by whether the info names a group. Slice propagation keeps its
single-series loop.

Not everything is a window. Inserting, editing and combining structures
are sections of the **Structure editor** module (`app/struct_tools.rs`,
`app/combine.rs`) in the right panel, drawn below the simulation section:
what a planner keeps at hand while contouring should not need a window to
be found, and a folded section costs no screen. A context menu's *Edit in
the Structure editor* or *∪ Combine* and a derived structure's *Edit
recipe* switch the module on and unfold the section (`reveal_editor`). The
editor works on one workspace (the A / B row at its top). The drawing tools
themselves are the toolbar's draw row, unfolded by *✏ Draw structure*
(`draw_strip`), acting on whichever view the pointer is in.

## Background jobs

One pattern serves every long operation:

```rust
struct Job<T> { progress: Arc<Progress>, rx: mpsc::Receiver<T> }
```

`Job::spawn` snapshots the inputs, starts a `std::thread` and hands the
worker the progress handle; the UI polls the channel each frame
(`poll_job`): a value lands the result, a disconnect means the worker died
and surfaces as an error. The tools answer with `(slot, Result)`, and
`poll_tool_job` turns a failure into an error dialog - except a
cancellation, which is what the user asked for.

`Progress` (`progress.rs`) holds a message, a fraction, the device label,
an atomic cancel flag and a phase window that maps a sub-step's own 0‥1 onto
its slice of the overall bar. An *outer* window (`set_outer`) nests that:
an engine that sets its own phases over 0‥1 runs unchanged inside one
slice of a run over a 4D group, and a message prefix names the phase its
messages belong to. Workers see it through `ProgressSink`, which
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
  totalsegmentator/<model>/fold_k.safetensors + plans.json
  mrsegmentator/<model>/fold_0..4.safetensors + plans.json
  lungmask/unet_<model>.pth + .safetensors
  monai_wholebody/model[_lowres].pt + .safetensors
  ctfm/model.safetensors
  vista3d/model.safetensors
  segvol/pytorch_model.bin, vocab.json, merges.txt, segvol.safetensors
  medsam2/MedSAM2_<variant>.pt, eff_medsam2_<size>_FLARE25_RECIST_baseline.pt
          + .safetensors
  nninteractive/nnInteractive_v1.0/plans.json, inference_session_class.json,
                                   fold_0.safetensors
  synth/{mr,cbct,mr_cbct,finetune}/CV_0..4.pt + CV_0..4.safetensors
  synth/custom/<key>/CV_k.safetensors    (your checkpoint folders, converted)
```

`models.rs` owns the layout and the inventory behind the model manager;
`nn/cache.rs` owns the path from a URL to a loaded tensor map
(`RemoteFile::ensure` ▶ `convert_checkpoint` ▶ `load_safetensors`, wrapped as
`ensure_converted`); each engine's `weights.rs` only says which files, which
tensors and under what names. A download is checked against its
Content-Length and, when it breaks off, resumed from its `.part` file (with
the server's validator beside it) on the next try; the converted cache is
read tensor by tensor into the networks' own buffers, never through a copy
of the whole file. What a run builds from the weights lives as long as the
run: an nnU-Net network is kept across the phases of one run by a
`KeepNetworks` guard (up to 1 GB of weights, the most recently used first)
and dropped with it, SegVol keeps its last window
encodings and MedSAM2 its slice features while the model is resident, so a
second prompt pays the decoder only. Installations that predate the single root are
migrated at startup: the old `autoseg_models/`, `segvol_model/` and
`medsam2_model/` folders beside the executable are renamed into place. The
Windows installer uses the same default, records `models_dir` only when a
different folder is chosen, and downloads whatever the user picks from the
same inventory (nothing by default; named sets for the recommended model,
the open-licence ones and all of them), through the same `models::ensure`.
The model manager's *Download open-licence only* is the rule for an
unattended fetch: what the registry's `Licence` calls open (Apache-2.0,
MIT) is fetched, everything else waits for a user's request. The
TotalSegmentator licence number, the user's nnU-Net folders and folders of
TotalSynth checkpoints reach the engines through `models::apply_settings`,
which every program that runs models calls once with the loaded settings.

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
  implementations' resampling conventions - scipy `zoom` (nnU-Net), PyTorch
  `nearest-exact` / `align_corners=false` (SegVol), PIL antialiased bicubic
  in 8-bit fixed point (MedSAM2) - because each is validated numerically
  against that reference.

## Error handling and style

`anyhow::Result` with `bail!` / `context` at operation boundaries; missing or
malformed *individual* DICOM attributes never error - safe extraction
helpers return `Option`, and per-file failures inside a batch become
warnings in the UI. Cancellation is an error whose message contains
`progress::CANCELLED`. `rayon` idioms: `par_iter` over independent files /
ROIs, `par_chunks_mut` over rows / slices; sums that decide a threshold or a
normalization stay sequential so a run reproduces itself, or go through
`par::ordered_fold`, which folds fixed-size pieces in parallel and adds the
pieces in input order - never rayon's own `sum` / `reduce`, whose order
depends on how the work was stolen. Modules open with
a `//!` block explaining the algorithm and its conventions, usually citing
the reference implementation.

Because `lib.rs` makes every module public, `cargo clippy -D warnings`
cannot see an unused `pub` item; the periodic review finds them with a
mechanical scan (every `pub` item referenced nowhere outside its own
tests).

## Dependencies

All pure Rust, and the viewer crate names only what the window needs
(`eframe`, `rfd`, `egui-snarl`, the image encoders); the rest belongs to
`rds-domain` and the engines: `dicom-rs` (DICOM, with `dicom-pixeldata` for decoding),
`egui` / `eframe` (UI over wgpu), `egui-snarl` (the workflow editor's node
canvas), `serde` (the workflow files), `rayon`, `rfd` (file dialogs; desktop
only), `walkdir`, `anyhow`; for the engines `gemm` (SIMD matrix kernels),
`serde_json`, `zip`, `ureq` (rustls + OS trust store; the bundled Mozilla
roots on Android and iOS), `safetensors`, and `burn` - always with
its `ndarray` CPU backend (named directly as `burn-ndarray` as well, only to
switch on its thread pool and SIMD kernels, which burn's own `ndarray`
feature leaves off), with the wgpu backend added by the cargo feature
`gpu` (default on). The cargo feature `mcp` (off by default) adds `rmcp`
(the official MCP SDK), `tokio` and `schemars` for the `rds-mcp`
executable only; the viewer's build pulls none of them. The PACS client
(every build) uses `ureq` with `rustls` named directly for the certificate
pin, `sha2`, `base64`, `getrandom`, `zip`, `toml` and `gethostname`; the
cargo feature `pacs-server` (off by default) adds `axum`, `axum-server`
(TLS through rustls, the `ring` provider), `tokio`, `rcgen`, `subtle` and
`futures-util` for the `rds-pacs` executable only.

## Testing

Twenty-seven integration suites plus in-module unit tests run against the same
code paths the GUI uses, with no external data or tooling: the analytic
phantom round trip (**synthetic_study**), simulate → export → reload
(**simulate_export**), rigid and B-spline recovery of known transforms
(**registration**), masks, growing, contours and meshing (**segmentation**),
anonymize → reload (**anonymize**), SEG written and read back voxel for voxel
(**dicomseg**), the body contour on phantoms with couch, chair and mask
(**body**), the archive round trip (**archive**), opening what is not a volume -
RT images and a structure set on their own, a single slice, a folder of RT
objects, and an image series added afterwards (**open_files**) - the DVH
against an analytic
Gaussian phantom (**dvh**), structure algebra (**structops**), the shipped
workflow example rebuilt on the 4D phantom and run end to end, in the
background and step by step (**workflow_graph**), the other workflow steps,
reruns, parallel rows against the serial run, a batch and the synthetic-CT
steps (**workflow_steps**), a synthetic CT written into its study, read
back and evaluated (**sct**), and the three
engines assembled and run without a download - a miniature nnU-Net with the
exact checkpoint naming (**autoseg**), and synthesized checkpoints with the
real key names and shapes for **segvol** and **medsam2**, so genuine forward
passes run in CI. The newer engines are held to their references by files
the reference code wrote: `tests/data/zoo-nets.safetensors` (randomly
initialised lungmask, MONAI SegResNet, SegResNetDS and VISTA-3D networks,
their inputs and PyTorch's / MONAI's outputs), `tests/data/vista-points.safetensors`
(VISTA-3D's point head and the bundle's point pipeline) and
`tests/data/nninteractive-session.safetensors` (nine steps of the
nnInteractive inference session with a stand-in network) and
`tests/data/medsam2-vitdet.safetensors` (EfficientTAM's ViT and neck, built
small, on weights a generator in the test reproduces) and the
`tests/data/synth-*` files (a miniature TotalSynth saved by KonfAI, its
outputs, and `konfai PREDICTION` runs of the ImpactSynth recipe on a CT
block, matched to the voxel but for half-precision rounding steps); the
engine unit tests read them. The MedSAM2 kernels' arithmetic is held to PyTorch's by
`tests/data/medsam2-ops.safetensors`, a file PyTorch, PIL and SAM 2 wrote
themselves: 74 small tensors recording what each primitive returns on a
random input. **ops_fixtures** re-derives every output in that file from
its inputs with the naive reference kernels of `tests/common/ops_ref.rs` -
textbook loops in `f64`, sharing nothing with the engine - and matches them
to a few 1e-6 (the 8-bit PIL resize exactly), which is what lets
`cargo run --example gen_ops_fixtures -- <path>` regenerate the fixture
without Python while it still stands for the frameworks' semantics. The
committed file stays the one PyTorch wrote. The repository holds no Python
at all; the end-to-end activation dump of the 0.8 releases, which ran Meta's
own sam2 package, is gone with it, and the port's fidelity claim rests on
the per-operation fixtures, the synthesized-checkpoint forward passes and
the runs against the real weights, which are `#[ignore]`d.

Three suites need the `mcp` feature: **workflow** runs the 4D pipeline
headless on a three-phase phantom whose target moves 0 / 6 / 3 mm and checks
the recovered motion; **mcp_tools** runs the heart sequence through the
server's core (register, propagate, propagate onto a group, motion, DVH,
export, re-open) and saved workflows through `list_workflows` /
`run_workflow`, a batch included; **mcp_phi** gives the phantom a patient's name and asserts
that no tool, no error path and no protocol frame of the real executable ever
carries it.

Two suites need the `pacs-server` feature and start real servers on loopback
ports (TLS, a state folder and an archive of their own under `target/`):
**pacs_server** holds the protocol to its rules - nothing without a token,
a role is a ceiling, a pairing code works once and guessing is throttled,
revoking is immediate, a changed certificate is refused before the token
leaves, no path is reachable through a URL, an oversized upload leaves the
archive alone, certificate, identity and tokens survive a restart;
**pacs_mirror** is the archive round trip over the wire (pull, load, draw,
send back, the outbox while the server is down, sync both ways, a study the
server dropped) and a task run end to end.

```
cargo test --release
cargo test --features mcp --test workflow --test mcp_tools --test mcp_phi
cargo test --features pacs-server --test pacs_server --test pacs_mirror
```
