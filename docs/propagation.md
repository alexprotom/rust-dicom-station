# Contour and segmentation propagation

*Tools ▶ ⇄ Propagate structures* carries any RTSTRUCT ROI or painted
segmentation across the active registration and lands it on the other
dataset as an ordinary, editable segmentation — convertible back to
RTSTRUCT and exportable as DICOM.

## What it does

* **Pull, never push.** Every voxel of the *destination* is asked where it
  comes from; pushing a deformed mask forward would leave holes where the
  deformation expands and double-write where it compresses.
* **Sub-voxel boundaries.** The source mask is sampled trilinearly and
  thresholded at ½: the boundary lands where the contour really is, and
  structures cross grids of different spacing and orientation.
* **Either direction.** The transform maps fixed → moving, so propagating
  *onto* the moving dataset runs through its inverse; you only choose the
  source dataset.
* **A cached mapping.** A deformable inverse is a fixed-point iteration,
  twelve control-lattice evaluations per point — billions of operations
  over a 512³ study. So the mapping is evaluated on a 3 mm lattice across
  the destination bounding box and interpolated in between: exact for a
  rigid transform, far below the contour's accuracy for a deformable one.

Each propagated structure is reported as `name: 164.2 cm³ ▶ 170.1 cm³
(+3.6 %)` — the volume change the registration panel's Jacobian statistics
also describe; if the two disagree, look harder at the registration.

## Global and local

**Globally**, propagation uses whatever registration is active; one
restricted to a region gives structures inside it the local mapping and
everything else the global one.

**Locally** — when one structure sits inside another that actually
deformed — *Refine locally first* runs a local deformable refinement on
the enclosing structure before anything is carried; otherwise a small
structure lands where the *larger* one's average deformation puts it.

The refinement replaces the active registration, so the sidebar reports
exactly what the propagation used; method and parameters come from the
sidebar (forced to deformable), the margin from the propagation window.

## Using it

1. Register the two datasets (any method; see
   [registration.md](registration.md)).
2. *Tools ▶ ⇄ Propagate structures…*
3. Choose the source dataset, tick what to carry, optionally pick an
   enclosing region to refine on, and press **▶ Propagate**.

Results arrive named `<structure> (from A)`, in the source structure's
colour, as the destination's active segmentation — edit with the brush,
view in 3D, convert to RTSTRUCT, export as DICOM.

## Verification

`src/propagate.rs`'s unit tests assert that a translation carries a ball
by exactly that much (centroid within 0.5 mm, volume preserved to 6 %),
that the direction flag really reverses the mapping, that a structure
mapped outside the destination comes back *empty*, and that a structure
crosses between a 2 mm and a 3 mm grid with its volume intact to 10 %.
