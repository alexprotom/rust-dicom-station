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

Tests marked `#[ignore]` need real downloaded weights and are enabled per engine via env vars (`RDS_AUTOSEG_MODELS`, `RDS_SEGVOL_MODEL`, ...; see the `//!` header of each suite in `tests/`). Do not un-ignore them: CI runners only have software GPU adapters.

Headless engine CLIs live in `examples/` (`autoseg_cli`, `segvol_cli`, `medsam2_cli`, `body_cli`, `*_probe`, `gen_ops_fixtures`), run with `cargo run --release --example <name> -- <args>`; the argument syntax is in each file's `//!` header.

Windows installer: `cd packaging/windows/installer && cargo build --release` (Windows-only crate, `compile_error!` elsewhere). Android: see `packaging/android/Cargo.toml` header and `docs/android.md`. iOS / iPadOS: `packaging/ios/build-ipa.sh` (Mac with Xcode; `--simulator` for the simulator build, `simulator-smoke.sh` starts it), `cargo clippy --target aarch64-apple-ios -- -D warnings` in `packaging/ios/` as the check; see `docs/ios.md`. Linux AppImage: `packaging/linux/appimage/build-appimage.sh`; snap: `packaging/linux/snap/build-snap.sh` (snapcraft only reads `snap/snapcraft.yaml` in the repo root, so the recipe is copied there at build time); macOS: `packaging/macos/build-app.sh --arch arm64|x86_64`.

CI (`.github/workflows/ci.yml`) runs on pull requests only: fmt + clippy on Linux, tests on Linux, Windows and macOS, CPU-only check, MCP job, installer job; `android.yml`, `macos.yml` and `ios.yml` add cross-compilation checks on PRs that touch `src/` or their packaging folder (`ios.yml` also builds the `.ipa` and runs it on a simulated iPad and iPhone when a PR changes `packaging/ios/`). Every workflow's paths point into `packaging/`; keep them in step when a folder moves. A push to `main` runs `release.yml`, which reads the version from `Cargo.toml` and fails if that tag already exists. **Bump the version in `Cargo.toml` before anything merges to `main`.** Branch flow is `develop -> release -> main`; feature branches merge into `develop`.

## Architecture in brief

**One library, several front ends.** `src/lib.rs` makes every module `pub`; the GUI (`main.rs`), the MCP server (`bin/rds-mcp.rs`), the integration tests and the examples all drive the same code. Because everything is `pub`, clippy cannot flag unused public items; check for them by hand when removing features.

**Layering** (dependencies flow downward):

- `app/` – the egui application. `ViewerApp` and all its state are defined once in `app/mod.rs`; every sibling file is just another `impl ViewerApp` block (visibility `pub(super)`), grouped by concern (views, panels, tree, each tool window / module). `ViewerApp` owns two `StudySlot`s (datasets A and B), each with three `ViewState`s.
- `workflow/` – the headless pipelines (4D motion, group propagation, anchored propagation, structure selection) shared by the tool windows and the MCP server. New multi-step logic that both UI and MCP need belongs here, not in `app/`.
- `mcp/` – the MCP server (feature `mcp`): config (`mcp.toml`), the PHI gate/redactor, sessions with handles, tools as plain functions with schema-deriving arg structs, rmcp glue.
- Domain modules at `src/` root: DICOM I/O (`loader`, `dicomfile`, `rtstruct`, `rtdose`, `rtplan`, `dicomseg`, `extras`, `dicom_export`, `export`, `anonymize`, `archive`), core geometry (`volume`, `geometry`, `render`, `morphology`), registration (`registration/` with `elastix`, `plastimatch`, `landmark`, `analysis`, `dvf`; `propagate`), 4D (`fourd`, `motion`), dose (`dvh`), structures (`contours`, `segmentation`, `structops`, `generate`, `derived`, `livewire`, `mesh3d`, `bodymask`, `templates`), simulators (`gen_test_data`, `simulate`, `drr`), the test-data download (`testdata`).
- `nn/` – architecture-agnostic NN infrastructure: checkpoint download → torch pickle read → safetensors conversion cache (`nn/cache.rs`), device choice (`nn/device.rs`), tensors, gemm-backed linalg, attention. The three engines `autoseg/` (TotalSegmentator, nnU-Net), `segvol/` and `medsam2/` sit on top and hold only what is theirs (`weights.rs` per engine names files/tensors; `gpu.rs` is the burn/wgpu path behind feature `gpu`; `medsam2` is generic over a burn backend so CPU and GPU share one implementation).
- Cross-cutting singletons: `progress.rs` (the one progress handle, `ProgressSink`, `Quiet`, cancel flag), `models.rs` (the model folder layout and inventory), `settings.rs` (persisted prefs, config/data folders), `gfx.rs` (graphics backend choice + fallback order; `WGPU_BACKEND` env overrides settings).

**Background jobs.** Anything longer than a frame runs via `Job::spawn` (a `std::thread` + `mpsc` + `Arc<Progress>`), polled each frame by `poll_job` / `poll_tool_job` in `app/mod.rs`. Workers parallelise internally with `rayon`. Cancellation is an `anyhow` error whose message contains `progress::CANCELLED`. Results are validated on landing against the current dataset (dims, frame-of-reference UID) because the dataset may have been replaced meanwhile.

**Rendering** is cache-driven: per-view keyed textures (grayscale, dose, contours, seg overlay, fusion) invalidated by generation counters bumped only at the owning mutation site. Repaints are demand-driven, 10 Hz while jobs run.

**Tool windows** are real OS windows (immediate viewports) drawn through `app/detach.rs::tool_window`; position/size are applied only on the creating pass, and every title goes through `window_title`. File dialogs go through `app/pick.rs` with a continuation closure (rfd on desktop, an egui browser on Android and iOS; on iOS its extra roots come from the system folder picker in `packaging/ios/src/places.rs` via `settings::ios::Places`). The four engine tools share their section layout via `app/seg_engines.rs` and live in the Structure auto tools module (`app/auto_tools.rs`).

**Platform keying.** Desktop vs Android vs iOS differences are expressed in `Cargo.toml` `[target.'cfg(...)']` tables (eframe glue, rfd, TLS roots), in `app/pick.rs`, and in the `config_dir` / `data_dir` arms of `settings.rs` (plus `gfx.rs`: Metal only on Apple); the rest of the code is one. Keep every iOS change behind `cfg(target_os = "ios")` so the other builds stay what they were.

## Conventions that matter

- **Geometry**: patient space is DICOM LPS in `f64` mm (`Vec3`). Volume voxels are `data[k·nx·ny + j·nx + i]`, dims `[nx, ny, nz]`; `origin` is the centre of voxel (0,0,0); orientation is by unit direction vectors, never assumed axis-aligned. `Volume::canonical_axes` maps onto `[S, A, R]` and every engine orients through it. Masks share the volume's index order. Sagittal/coronal view rows run superior → inferior (`y = (nz−1) − k`), asserted by tests.
- **Resampling fidelity**: each engine keeps its reference implementation's convention (scipy `zoom` for nnU-Net, PyTorch `nearest-exact`/`align_corners=false` for SegVol, PIL 8-bit fixed-point bicubic for MedSAM2). `tests/ops_fixtures.rs` holds the MedSAM2 kernels to `tests/data/medsam2-ops.safetensors`, a file PyTorch wrote; keep that committed file as is.
- **Errors**: `anyhow::Result` with `bail!`/`context` at operation boundaries. A missing or malformed individual DICOM attribute never errors: safe extraction helpers return `Option`, and per-file failures inside a batch become UI warnings.
- **Determinism**: `par_iter` over independent files/ROIs and `par_chunks_mut` over rows/slices are fine, but any sum that decides a threshold or normalisation stays sequential so a run reproduces itself.
- **Glyphs**: every non-ASCII character in UI strings must be drawable by egui's bundled fonts. A unit test in `app/glyphs.rs` walks the sources and fails on anything outside its `ALLOWED` list; add to that list only after checking the font's cmap.
- **Module docs**: each module opens with a `//!` block explaining the algorithm, its conventions and the reference implementation it follows. Keep that when adding modules; the extensive inline comments are part of the codebase's style.
- **PHI**: no MCP tool, error path or protocol frame may ever carry patient identifiers (`tests/mcp_phi.rs` enforces this). Every outgoing MCP string passes the redactor in `mcp/phi.rs`.
- **Line endings**: `.gitattributes` forces LF everywhere except `.bat`/`.cmd`.

## Test fixtures

All suites run without external data or downloads. `tests/common/mod.rs` builds a synthetic three-phase 4D phantom on disk under `target/` from `gen_test_data` (used by `workflow`, `mcp_tools`, `mcp_phi`). `tests/common/ops_ref.rs` holds naive `f64` reference kernels for the MedSAM2 op fixture. `data-test/` is a real two-phase 4DCT (TCIA 4D-Lung P102, CC BY 3.0) used by the `#[ignore]`d real-weight runs and for manual testing; `src/testdata.rs` (*Tools > Download test data*) fetches that folder from GitHub for installed copies.

## Where models live

Engines download weights on first use into `<data folder>/models/{totalsegmentator,segvol,medsam2}/` (`%LOCALAPPDATA%\RustDICOMStation` on Windows, `~/.local/share/RustDICOMStation` on Linux, `~/Library/Application Support/RustDICOMStation` on macOS), configurable via `models_dir` in `viewer_settings.txt`. `models.rs` owns the layout; `nn/cache.rs` owns download → convert → load.
