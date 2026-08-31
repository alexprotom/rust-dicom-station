# Documentation

Start with the [main README](../README.md) for the overview and quick start;
each page here covers one area.

| Page | Contents |
|---|---|
| [viewer.md](viewer.md) | Loading, volume reconstruction, the MPR layout, window/level, the patient ▶ study ▶ series tree, comparison mode, planar images, interaction bindings, appearance |
| [rt-objects.md](rt-objects.md) | RTSTRUCT, RTDOSE, RTPLAN, REG, RT treatment records, and how their reference chains are resolved |
| [registration.md](registration.md) | The four registration engines, local registration, analytics, vector fields, fusion, the transform simulator, verification |
| [propagation.md](propagation.md) | Carrying contours and segmentations across a registration |
| [motion-4d.md](motion-4d.md) | 4D groups, the per-phase register ▸ propagate ▸ measure pipeline, motion metrics, ITV generation, the results window, structure comparison and transfer |
| [drr.md](drr.md) | Digitally reconstructed radiographs: the two projectors, the geometry, the comparison |
| [dvh.md](dvh.md) | Dose–volume histograms: sampling, axes, metrics, protocol constraints, CSV export, the analytic phantom |
| [segmentation.md](segmentation.md) | Interactive segmentation: brush, eraser, geodesic region growing, the 3D view, mask → RTSTRUCT |
| [structure-algebra.md](structure-algebra.md) | Boolean operations, margins in patient directions, cropping, cleanup |
| [body-contour.md](body-contour.md) | The body / EXTERNAL contour: the classical and the model-assisted method, CT and MR, verification |
| [auto-segmentation.md](auto-segmentation.md) | The pure-Rust TotalSegmentator: models, pipeline, CPU/GPU engines, validation, the 117 classes, licensing |
| [segvol.md](segvol.md) | Prompt-driven segmentation: box / point / text, the SegVol re-implementation, weights and licensing |
| [medsam2.md](medsam2.md) | Slice propagation: the MedSAM2 re-implementation, validation, weights and licensing |
| [pacs.md](pacs.md) | The local patient archive: the window, the on-disk layout, filing, loading, sending changes back |
| [export-and-tools.md](export-and-tools.md) | DICOM export, the model manager, the anonymizer, the test-data generator |
| [architecture.md](architecture.md) | Design, functional overview, module map, the tool windows, background jobs, the model folder, conventions, testing |
| [release-versioning.md](release-versioning.md) | Versioning, the branch workflow, how CI produces a release |
| [example-data.md](example-data.md) | The bundled example patient: contents, source, citations, license |
| [early-detection-plan.md](early-detection-plan.md) | Plan: integrating early cancer detection methods (CT) |
