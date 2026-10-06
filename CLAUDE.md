# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rust-dicom-station` (RDS): a DICOM / RT DICOM viewer and analysis station written entirely in Rust (egui/eframe over wgpu). Everything that would normally be a C/C++ or Python binding (elastix, plastimatch, ITK ray-casting, TotalSegmentator, SegVol, MedSAM2) is re-implemented natively. There is no Python anywhere in the repository. Detailed design docs live in `docs/`; `docs/architecture.md` is the authoritative module map, threading model and conventions reference and should be read before any non-trivial change.

## Commands

The root crate is the viewer library + `rust-dicom-station` binary. Everything that packages it lives under `packaging/` (one folder per platform, see `packaging/README.md`); `packaging/windows/installer/`, `packaging/android/` and `packaging/ios/` are **separate Cargo workspaces** (empty `[workspace]` tables) with their own `target/`, reaching the viewer through a path dependency (`../../..` for the installer, `../..` for the other two); a root `cargo build` never touches them, and the root `cargo fmt --all` does not format them either.

```bash
cargo build --release
cargo run --release -- data-test/lung_p1_4DCT_phase_000                        # one dataset
cargo run --release -- data-test/lung_p1_4DCT_phase_000 data-test/lung_p1_4DCT_phase_050   # comparison mode
cargo test --release                                                           # default feature set (gpu)
cargo test --release --test registration                                       # one suite
cargo test --release --test dvh -- some_test_name                              # one test
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings                                      # CI fails on any warning
cargo check --no-default-features --all-targets                                # CPU-only build (no wgpu inference backend)
```

MCP server (`rds-mcp`) and its three suites are behind the `mcp` feature; the viewer build pulls none of that:

```bash
cargo clippy --features mcp --all-targets -- -D warnings
cargo test --features mcp --test workflow --test mcp_tools --test mcp_phi
cargo run --features mcp --bin rds-mcp -- --check
```

PACS server (`rds-pacs`) and its two suites are behind the `pacs-server` feature; the client side (`src/pacs/` minus `tls`, `auth`, `server`, `tasks`) is in every build, mobile included:

```bash
cargo clippy --features pacs-server --all-targets -- -D warnings
cargo test --features pacs-server --test pacs_server --test pacs_mirror
cargo run --features pacs-server --bin rds-pacs -- --check
```

Tests marked `#[ignore]` need real downloaded weights and are enabled per engine via env vars (`RDS_AUTOSEG_MODELS`, `RDS_SEGVOL_MODEL`, ...; see the `//!` header of each suite in `tests/`). Do not un-ignore them: CI runners only have software GPU adapters.

Headless engine CLIs live in `examples/` (`seg_cli` for every registry model, `autoseg_cli`, `segvol_cli`, `medsam2_cli`, `interactive_cli` for nnInteractive and VISTA-3D clicks, `body_cli`, `*_probe`, `gen_ops_fixtures`), run with `cargo run --release --example <name> -- <args>`; the argument syntax is in each file's `//!` header.

Windows installer: `cd packaging/windows/installer && cargo build --release` (Windows-only crate, `compile_error!` elsewhere). Android: see `packaging/android/Cargo.toml` header and `docs/android.md`. iOS / iPadOS: `packaging/ios/build-ipa.sh` (Mac with Xcode; `--simulator` for the simulator build, `simulator-smoke.sh` starts it), `cargo clippy --target aarch64-apple-ios -- -D warnings` in `packaging/ios/` as the check; see `docs/ios.md`. Linux AppImage: `packaging/linux/appimage/build-appimage.sh`; snap: `packaging/linux/snap/build-snap.sh` (snapcraft only reads `snap/snapcraft.yaml` in the repo root, so the recipe is copied there at build time); macOS: `packaging/macos/build-app.sh --arch arm64|x86_64`.

CI (`.github/workflows/ci.yml`) runs on pull requests: fmt + clippy on Linux, clippy on Windows and macOS (own jobs, with `--features pacs-server`), tests on Linux (default features) and on Windows and macOS (`--no-default-features`: no test drives the GPU, and the `gpu` feature is over half of what the crate compiles), CPU-only check, MCP job (suites in the CPU-only build), PACS job (likewise), installer job; a push to `main` runs the same jobs without the tests, only to save the cache every pull request restores (a PR's own cache is invisible to other PRs, so only `main` saves); `android.yml`, `macos.yml` and `ios.yml` add cross-compilation checks on PRs that touch `src/`, `crates/` or their packaging folder (`ios.yml` also builds the `.ipa` and runs it on a simulated iPad and iPhone when a PR changes `packaging/ios/`). Every workflow's paths point into `packaging/`; keep them in step when a folder moves. A push to `main` runs `release.yml`, which reads the version from `Cargo.toml` and fails if that tag already exists. **Bump the version in `Cargo.toml` before anything merges to `main`.** Branch flow is `develop -> nightly -> main`; feature branches merge into `develop`, `nightly` is fast-forwarded from `develop` (`git push origin develop:nightly`) and builds the rolling `nightly` pre-release (`nightly.yml`: Windows installer + Android APK, lighter checks), and the pull request `nightly -> main` is the release candidate that runs the full CI.

## Architecture in brief

**One library, several front ends.** `src/lib.rs` makes every module `pub`; the GUI (`main.rs`), the MCP server (`bin/rds-mcp.rs`), the PACS server (`bin/rds-pacs.rs`), the integration tests and the examples all drive the same code. Because everything is `pub`, clippy cannot flag unused public items; check for them by hand when removing features.

**Three crates, one API.** The root is a Cargo workspace: the viewer (`.`), `crates/rds-core` (`geometry`, `volume`, `progress`) and `crates/rds-engines` (`nn`, `zoo`, `autoseg`, `unet2d`, `segresnet`, `vista3d`, `segvol`, `medsam2`, `nninteractive`, and the `gpu` feature; the viewer's `gpu` forwards to it). `src/lib.rs` re-exports those modules, so `crate::volume::Volume` and `rust_dicom_station::medsam2` resolve as before; inside the engines crate, `crate::volume`/`crate::progress` resolve through re-exports too. The split exists for build time: burn's wgpu kernels (over half the machine code) compile once as a dependency instead of with every viewer change. `default-members` covers all three, so a bare `cargo build/test/clippy` in the root works on the whole; fixtures the engine unit tests read are addressed from `CARGO_MANIFEST_DIR/../../tests/data`.

**Layering** (dependencies flow downward):

- `app/`: the egui application. `ViewerApp` and all its state are defined once in `app/mod.rs`; every sibling file is just another `impl ViewerApp` block (visibility `pub(super)`), grouped by concern (views, panels, tree, each tool window / module). `ViewerApp` owns two `StudySlot`s (datasets A and B), each with three `ViewState`s.
- `workflow/`: the headless pipelines (4D motion, group propagation, anchored propagation, structure selection) shared by the tool windows and the MCP server. New multi-step logic that both UI and MCP need belongs here, not in `app/`. `workflow/graph/` is the user-facing *Workflows* feature: a JSON-saved graph of typed nodes (`catalog.rs`), run headless in dependency order by `exec.rs` (node bodies in `nodes.rs` and `nodes/more.rs`; reruns through `StepCache`, parallel rows, batches over `LoadFolders`); the editor (egui-snarl canvas) and the run window are `app/workflow_edit.rs` / `app/workflow_run.rs`, the headless runner `examples/workflow_cli.rs`, the MCP tools `mcp/tools/workflows.rs`. `workflow/session.rs` is the headless core under both the MCP session and the runner (`Volumes`, the budgeted volume cache; `file_items`, filing structures under one name-clash rule across phases) and `workflow/params.rs` the choices dialogs, MCP tools and steps share by name. Adding a node kind: see the checklist at the end of `docs/workflows.md`.
- `mcp/`: the MCP server (feature `mcp`): config (`mcp.toml`), the PHI gate/redactor, sessions with handles, tools as plain functions with schema-deriving arg structs, rmcp glue.
- `pacs/`: the archive served to other stations over HTTPS (`docs/pacs-server.md`). Always compiled: `protocol` (serde structs both sides share, `#[serde(default)]` everywhere), `client` (`Remote`: ureq + a rustls verifier that pins the server's certificate fingerprint), `servers` (`pacs-servers.json`), `mirror` (an `Archive` per server under `pacs-mirror/<id>/`, the outbox, sync as SOP-UID set differences), `config` (`pacs.toml`), `local` (the server on this machine: `running.json`, the local operator token, starting `rds-pacs` detached). Feature `pacs-server`: `tls` (rcgen self-signed cert), `auth` (pairing codes, hashed tokens, roles), `server` (axum over tokio; nothing is built from the URL, UIDs are compared with folder names), `tasks` (queue + runner over `workflow::graph::exec`, input bindings, templates). The viewer never hosts the server: `app/pacs_server_win.rs` starts it and talks to it over the loopback like any client; `app/pacs_remote.rs` is the client half of the PACS window.
- Domain modules at `src/` root: DICOM I/O (`loader`, `dicomfile`, `rtstruct`, `rtdose`, `rtplan`, `dicomseg`, `extras`, `dicom_export`, `export`, `anonymize`, `archive`), core geometry (`volume`, `geometry`, `render`, `morphology`), registration (`registration/` with `elastix`, `plastimatch`, `landmark`, `analysis`, `dvf`; `propagate`), 4D (`fourd`, `motion`), dose (`dvh`), structures (`contours`, `segmentation`, `structops`, `generate`, `derived`, `livewire`, `mesh3d`, `bodymask`, `templates`), simulators (`gen_test_data`, `simulate`, `drr`), the test-data download (`testdata`).
- `nn/` (in `crates/rds-engines/src/`, as are the engines): architecture-agnostic NN infrastructure: checkpoint download → torch pickle read (zip and legacy layouts) → safetensors conversion cache (`nn/cache.rs`), plain pickles with numpy values as JSON (`nn/pyobj.rs`, nnU-Net v1 plans), single members of a remote zip by HTTP range (`nn/remote_zip.rs`), device choice (`nn/device.rs`), tensors, gemm-backed linalg, attention, and `nn/fastconv.rs` (routes burn convolutions on the CPU backend to the gemm kernels). The engines sit on top and hold only what is theirs: `autoseg/` (the nnU-Net family: TotalSegmentator v2/v3 and its licensed tasks, MRSegmentator, the nnU-Net v1 models in `v1.rs`, the user's own nnU-Net v2 folders in `custom.rs`, nnInteractive's network), `unet2d/` (lungmask), `segresnet/` (MONAI whole body, CT-FM, VISTA-3D's encoder), `vista3d` (automatic and point modes; NV-Segment-CT and NV-Segment-CTMR), `segvol/`, `medsam2/`, `nninteractive/` (the promptable session). `zoo/` is the registry of automatic models (one row per model: key, classes, modality, licence, size, how to run it) that the viewer, `seg_cli`, workflows and MCP read; a new model is a new row. Per engine, `weights.rs` names files/tensors; `gpu.rs` is the burn/wgpu path behind feature `gpu`; `medsam2` and `segresnet` are generic over a burn backend so CPU and GPU share one implementation.
- Cross-cutting singletons: `progress.rs` (in `crates/rds-core/src/`; the one progress handle, `ProgressSink`, `Quiet`, cancel flag), `par.rs` (`ordered_fold`: parallel sums in a fixed order), `models.rs` (the model folder layout and inventory), `settings.rs` (persisted prefs, config/data folders), `gfx.rs` (graphics backend choice + fallback order; `WGPU_BACKEND` env overrides settings).

**Background jobs.** Anything longer than a frame runs via `Job::spawn` (a `std::thread` + `mpsc` + `Arc<Progress>`), polled each frame by `poll_job` / `poll_tool_job` in `app/mod.rs`. Workers parallelise internally with `rayon`. Cancellation is an `anyhow` error whose message contains `progress::CANCELLED`. Results are validated on landing against the current dataset (dims, frame-of-reference UID) because the dataset may have been replaced meanwhile.

**Rendering** is cache-driven: per-view keyed textures (grayscale, dose, contours, seg overlay, fusion) invalidated by generation counters bumped only at the owning mutation site. Repaints are demand-driven, 10 Hz while jobs run.

**Tool windows** are real OS windows (immediate viewports) drawn through `app/detach.rs::tool_window`; position/size are applied only on the creating pass, and every title goes through `window_title`. File dialogs go through `app/pick.rs` with a continuation closure (rfd on desktop, an egui browser on Android and iOS; on iOS its extra roots come from the system folder picker in `packaging/ios/src/places.rs` via `settings::ios::Places`). The engine sections (automatic models, SegVol, MedSAM2, body contour, interactive segmentation in `app/interactive_seg.rs`) share their layout via `app/seg_engines.rs` and live in the Structure auto tools module (`app/auto_tools.rs`).

**Platform keying.** Desktop vs Android vs iOS differences are expressed in `Cargo.toml` `[target.'cfg(...)']` tables (eframe glue, rfd, TLS roots), in `app/pick.rs`, and in the `config_dir` / `data_dir` arms of `settings.rs` (plus `gfx.rs`: Metal only on Apple); the rest of the code is one. Keep every iOS change behind `cfg(target_os = "ios")` so the other builds stay what they were.

## Conventions that matter

- **Geometry**: patient space is DICOM LPS in `f64` mm (`Vec3`). Volume voxels are `data[k·nx·ny + j·nx + i]`, dims `[nx, ny, nz]`; `origin` is the centre of voxel (0,0,0); orientation is by unit direction vectors, never assumed axis-aligned. `Volume::canonical_axes` maps onto `[S, A, R]` and every engine orients through it. Masks share the volume's index order. Sagittal/coronal view rows run superior → inferior (`y = (nz−1) − k`), asserted by tests.
- **Resampling fidelity**: each engine keeps its reference implementation's convention (scipy `zoom` for nnU-Net, scipy/skimage for lungmask, MONAI's `Spacingd`/`ScaleIntensityRanged` for SegResNet and VISTA-3D, PyTorch `nearest-exact`/`align_corners=false` for SegVol, PIL 8-bit fixed-point bicubic for MedSAM2, PyTorch float16 accumulation and fused multiply-adds for nnInteractive's prompt resampling). `tests/ops_fixtures.rs` holds the MedSAM2 kernels to `tests/data/medsam2-ops.safetensors`, a file PyTorch wrote; keep that committed file as is.
- **Errors**: `anyhow::Result` with `bail!`/`context` at operation boundaries. A missing or malformed individual DICOM attribute never errors: safe extraction helpers return `Option`, and per-file failures inside a batch become UI warnings.
- **Determinism**: `par_iter` over independent files/ROIs and `par_chunks_mut` over rows/slices are fine, but any sum that decides a threshold, a normalisation or an optimizer step stays sequential or goes through `par::ordered_fold` (fixed pieces, added in order), so a run reproduces itself on any thread count; never rayon's `sum`/`reduce` for such a float.
- **Glyphs**: every non-ASCII character in UI strings must be drawable by egui's bundled fonts. A unit test in `app/glyphs.rs` walks the sources and fails on anything outside its `ALLOWED` list; add to that list only after checking the font's cmap.
- **Module docs**: each module opens with a `//!` block explaining the algorithm, its conventions and the reference implementation it follows. Keep that when adding modules; the extensive inline comments are part of the codebase's style.
- **PHI**: no MCP tool, error path or protocol frame may ever carry patient identifiers (`tests/mcp_phi.rs` enforces this). Every outgoing MCP string passes the redactor in `mcp/phi.rs`.
- **Line endings**: `.gitattributes` forces LF everywhere except `.bat`/`.cmd`.

## Test fixtures

All suites run without external data or downloads. `tests/common/mod.rs` builds a synthetic three-phase 4D phantom on disk under `target/` from `gen_test_data` (used by `workflow`, `mcp_tools`, `mcp_phi`). `tests/common/ops_ref.rs` holds naive `f64` reference kernels for the MedSAM2 op fixture. The engine unit tests read reference outputs PyTorch/MONAI/nninteractive wrote: `tests/data/zoo-nets.safetensors` (lungmask's 2-D U-Net, SegResNet, VISTA-3D's encoder), `vista-points.safetensors` (VISTA-3D point head and pipeline), `nninteractive-session.safetensors` (nine steps of an nnInteractive session), `medsam2-vitdet.safetensors` (Efficient MedSAM2's encoder); keep them as committed. `data-test/` is a real two-phase 4DCT (TCIA 4D-Lung P102, CC BY 3.0) used by the `#[ignore]`d real-weight runs and for manual testing; `src/testdata.rs` (*Tools > Download test data*) fetches that folder from GitHub for installed copies.

## Where models live

Engines download weights on first use into `<data folder>/models/{totalsegmentator,mrsegmentator,lungmask,monai_wholebody,ctfm,vista3d,segvol,medsam2,nninteractive}/` (`%LOCALAPPDATA%\RustDICOMStation` on Windows, `~/.local/share/RustDICOMStation` on Linux, `~/Library/Application Support/RustDICOMStation` on macOS), configurable via `models_dir` in `viewer_settings.txt`. `models.rs` owns the layout; `nn/cache.rs` owns download → convert → load.
