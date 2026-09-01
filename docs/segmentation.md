# Interactive segmentation and the 3D view

MITK-style manual and semi-automatic segmentation, entirely in Rust and
CPU-side, plus a Slicer-style 3D surface view. Neural-network
auto-segmentation is [auto-segmentation.md](auto-segmentation.md), the patient
outline [body-contour.md](body-contour.md) — both land as the editable masks
described here — and combining structures is
[structure-algebra.md](structure-algebra.md).

## Segmentation masks

A segmentation is a per-voxel label mask (one byte per voxel, same index order
as the volume), with a name, display color, visibility flag, voxel count /
volume readout and a per-stroke undo journal.

Masks are grouped into **segmentation series** — what a DICOM SEG file is and
exports as. A series lives in the study, not the view state, and names the
image series it is drawn on, so painted work survives a series switch, travels
with tree copy/move, and can be re-pointed at a different image series later.
Its masks keep the lattice they were made on and are resampled onto the
displayed volume when their own image series is shown; a series belonging to
another image series stays intact and simply reports that it is not editable
here.

The sidebar *Segmentations* section shows the series as tree nodes and, below
the active one, its segments: visibility, color, active selection, volume in
cm³, per-stroke undo (**Ctrl+Z**), delete, and conversion to RTSTRUCT
(**→RS**, below). *All* / *None* tick every segment or none, Shift-click
extends a range, and the tick doubles as the selection for the actions beneath
— copy, move, remove, and *💾* to export just those segments as a DICOM SEG
file. See [viewer.md](viewer.md#structures-and-segmentations-in-the-tree).

## Tools

The toolbar tools take over the left mouse button in the MPR views:

* **🎨 Paint / ⊖ Erase** — a spherical, spacing-aware **3D brush** (radius in
  mm, set via the toolbar, `Shift+wheel` or `[` `]`) paints in any of the
  three views; a **3D** toggle switches to a flat 2D circle confined to the
  displayed slice. Strokes are swept as capsules between pointer samples, so
  fast drags stay gap-free; `Alt` temporarily erases while painting.
* **✨ Grow** — interactive **organ-wise** segmentation by geodesic fast
  marching, not a plain threshold: a Dijkstra front expands from the seed, the
  cost of each step rising exponentially with the voxel's intensity deviation
  from robust seed statistics (median/MAD of the local neighborhood) **and**
  with the intensity jump of the crossing itself — organ boundaries, fat
  planes and edges act as barriers, so the organ under the cursor is suggested
  first instead of all similar-intensity tissue. Press to seed, drag up/down
  to extend/shrink the geodesic reach with a live yellow preview; the front
  expands *incrementally* (drag up continues the same priority queue, drag
  down truncates the accepted prefix), never recomputing from scratch. Release
  commits — enclosed holes (vessels, calcifications) are filled slice-wise so
  the organ comes out solid — and `Esc` cancels.

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

The **3D A / 3D B** toolbar buttons open a floating window with a 3D surface
rendering of the active structure set **and** all segmentation masks. Surfaces
are built on a background thread — scanline rasterization of contours into a
binary volume, a surface-nets mesher, Laplacian smoothing, area-weighted
vertex normals, `rayon`-parallel per ROI — and drawn in the display colors
with headlight shading; EXTERNAL/body ROIs are translucent so internal anatomy
stays visible.

Every mask edit re-meshes that segmentation in the background
(bounding-box-cropped surface nets, automatic striding for huge masks), so the
3D surface follows the brush in essentially real time; meshes are cached per
structure set, so reopening the window is instant. Drag rotates, the wheel
zooms, middle-drag pans, a slider sets global opacity, and *⟲ Reset view*
restores the default camera.

## Mask → RTSTRUCT (→RS)

The **→RS** button converts a mask to RTSTRUCT closed planar contours:
marching squares per axial slice, loops stitched and decimated, points mapped
to patient coordinates. The new ROI joins the active structure set — or a new
in-memory set ("Segmentations") when the study has no RTSTRUCT — and renders
like any ROI, participates in the 3D view, and rides the existing DICOM export
([export-and-tools.md](export-and-tools.md)).

## Verification

`tests/segmentation.rs` covers the invariants: the brush sphere respects
anisotropic spacing, 2D strokes stay slice-confined, undo restores voxel
counts exactly, the geodesic grow respects organ boundaries under a 20× reach
increase (no-leak test), hole filling, `mask → RTSTRUCT` contour round-trips,
and the mask/contour → mesh pipeline on an analytic sphere.
