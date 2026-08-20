# Interactive segmentation and the 3D view

MITK-style manual and semi-automatic segmentation, implemented entirely in
Rust and CPU-side, plus a Slicer-style 3D surface view that follows every
edit in essentially real time. For the neural-network auto-segmentation
see [auto-segmentation.md](auto-segmentation.md) — its results land as the
same editable masks described here.

## Segmentation masks

A segmentation is a per-voxel label mask on the active volume's grid
(one byte per voxel, same index order as the volume), with a name, display
color, visibility flag, voxel count / volume readout and a per-stroke undo
journal. Any number of masks per dataset are managed in the sidebar
*Segmentations* section: visibility, color, active selection, volume in
cm³, per-stroke undo (**Ctrl+Z**), delete, and conversion to RTSTRUCT
(**→RS**, below).

## Tools

The toolbar tools take over the left mouse button in the MPR views:

* **🖌 Paint / ◻ Erase** — a spherical, spacing-aware **3D brush** (radius
  in mm, adjustable via the toolbar, `Shift+wheel` or `[` `]`) paints in
  any of the three views; a **3D** toggle switches to a flat 2D circle
  confined to the displayed slice. Strokes are swept as capsules between
  pointer samples, so fast drags stay gap-free. `Alt` temporarily erases
  while painting.
* **✨ Grow** — interactive **organ-wise** segmentation by geodesic fast
  marching (not a plain threshold): a Dijkstra front expands from the
  seed, and the cost of each step rises exponentially with the voxel's
  intensity deviation from robust seed statistics (median/MAD of the local
  neighborhood) **and** with the intensity jump of the crossing itself —
  organ boundaries, fat planes and edges act as barriers, so the organ
  under the cursor is suggested first instead of flooding all
  similar-intensity tissue. Press to seed, drag up/down to extend/shrink
  the geodesic reach with a live yellow preview; the front expands
  *incrementally* (drag up continues the same priority queue, drag down
  truncates the accepted prefix), so the preview never recomputes from
  scratch. Release commits — enclosed holes (vessels, calcifications) are
  filled slice-wise so the organ comes out solid — and `Esc` cancels.

| Input (tool active) | Action |
|---|---|
| Left drag | Paint / erase — or, with ✨, seed and drag ↑↓ to grow/shrink |
| Alt | Erase while painting |
| Shift + wheel, `[`, `]` | Brush radius |
| Ctrl + Z | Undo the last stroke |
| Esc | Cancel the running region grow |

Masks appear instantly in **all three MPR views** (crisp nearest-neighbor
colorwash over the grayscale).

## The 3D structure view

The **3D A / 3D B** toolbar buttons open a floating window with a 3D
surface rendering of the active structure set **and** all segmentation
masks. Surfaces are reconstructed on a background thread — scanline
rasterization of contours into a binary volume, a surface-nets mesher,
Laplacian smoothing, area-weighted vertex normals, `rayon`-parallel per
ROI — and drawn in the display colors with headlight shading;
EXTERNAL/body ROIs are rendered translucent so internal anatomy stays
visible.

Every mask edit re-meshes that segmentation in the background
(bounding-box-cropped surface nets with automatic striding for huge
masks), so the 3D surface follows the brush in essentially real time.
Meshes are cached per structure set, so reopening the window is instant.
Drag rotates, the wheel zooms, middle-drag pans, a slider controls global
opacity, and *⟲ Reset view* restores the default camera.

## Mask → RTSTRUCT (→RS)

The **→RS** button converts a mask to RTSTRUCT closed planar contours:
marching squares per axial slice, loops stitched and decimated, points
mapped to patient coordinates. The new ROI is appended to the active
structure set — or a new in-memory set ("Segmentations") is created when
the study has no RTSTRUCT — so it renders like any ROI, participates in
the 3D view, and rides the existing DICOM export
([export-and-tools.md](export-and-tools.md)).

## Verification

`tests/segmentation.rs` covers the invariants: the brush sphere respects
anisotropic spacing, 2D strokes stay slice-confined, undo restores voxel
counts exactly, the geodesic grow respects organ boundaries under a 20×
reach increase (no-leak test), hole filling, `mask → RTSTRUCT` contour
round-trips, and the mask/contour → mesh pipeline on an analytic sphere.
