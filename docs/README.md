# Documentation

Comprehensive documentation for **rust-dicom-station**. Start with the
[main README](../README.md) for the overview and quick start; each page here
covers one area in depth.

| Page | Contents |
|---|---|
| [viewer.md](viewer.md) | Loading DICOM data, volume reconstruction, the three-view MPR layout, window/level, the patient ▶ study ▶ series tree, comparison mode, planar images, interaction bindings, appearance |
| [rt-objects.md](rt-objects.md) | RT DICOM objects: RTSTRUCT, RTDOSE, RTPLAN, REG spatial registrations, RT treatment records, and how their reference chains are resolved |
| [registration.md](registration.md) | Rigid and deformable (B-spline) image registration: algorithms, parameters, the fusion overlay, the transform simulator for registration QA, accuracy verification |
| [segmentation.md](segmentation.md) | Interactive segmentation: 2D/3D brush, eraser, geodesic region growing, the live 3D structure view, mask → RTSTRUCT conversion |
| [auto-segmentation.md](auto-segmentation.md) | Automatic multi-organ segmentation — the pure-Rust TotalSegmentator re-implementation: models, usage, the inference pipeline, CPU/GPU engines, validation, the full 117-class table, licensing |
| [segvol.md](segvol.md) | Prompt-driven segmentation — the pure-Rust SegVol re-implementation: box / point / text prompts, the two-pass pipeline, weights and licensing, validation status |
| [export-and-tools.md](export-and-tools.md) | DICOM export, the interactive anonymizer, the synthetic test-data generator |
| [architecture.md](architecture.md) | Code architecture: design philosophy, the functional overview (what the program does, by category), the module map (where each function lives), the shared engine windows, threading model, the model folder, caching, geometry conventions, testing |
| [example-data.md](example-data.md) | The bundled example patient data: contents, source, citations, license |
