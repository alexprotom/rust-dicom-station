# Contour and segmentation propagation

Once two datasets are aligned, the alignment is only half the answer: the
contours drawn on one of them have to arrive on the other. *Tools ▶ ⇄
Propagate structures* carries any RTSTRUCT ROI or painted segmentation
across the active registration and lands it as an ordinary, editable
segmentation on the other dataset — convertible back to RTSTRUCT like any
other, and exportable as DICOM.

## What it does

* **Pull, never push.** Every voxel of the *destination* is asked where it
  comes from, rather than every voxel of the source being asked where it
  goes. Pushing a deformed mask forward leaves holes wherever the
  deformation expands and double-writes wherever it compresses; pulling asks
  one question per destination voxel and answers it exactly.
* **Sub-voxel boundaries.** The source mask is sampled trilinearly and
  thresholded at ½, so the boundary lands where the contour really is rather
  than on the nearest voxel centre. Structures cross between grids of
  different spacing and orientation without any special case.
* **Either direction.** The transform maps fixed → moving, so propagating
  *onto* the moving dataset runs through its inverse. The window works this
  out from the registration's own direction; you only choose which dataset
  the structures come from.
* **A cached mapping.** The inverse of a deformable transform is a
  fixed-point iteration — twelve evaluations of a control lattice per point.
  Asked once per voxel of a 512³ study that is billions of operations for a
  mapping that is smooth to well under a millimetre over any few voxels. So
  the mapping is evaluated on a 3 mm lattice across the destination bounding
  box and interpolated in between: exact for a rigid transform (the map is
  affine, and so is the interpolation) and far below the contour's own
  accuracy for a deformable one.

Each propagated structure is reported as `name: 164.2 cm³ ▶ 170.1 cm³
(+3.6 %)`. That volume change is the deformation's doing, and it is exactly
what the Jacobian statistics in the registration panel describe — the two
numbers are the same fact seen from two directions, and disagreeing with
each other is a good reason to look harder at the registration.

## Global and local

**Globally**, propagation uses whatever registration is active. If that
registration was itself restricted to a region, the structures inside it get
the local mapping and everything else gets the global one — a refinement is
stored as the global warp *plus* a correction that is exactly zero outside
its lattice, so this needs no special handling.

**Locally** — when one structure sits inside another and the enclosing one
is what actually deformed — the window's *Refine locally first* section runs
a local deformable refinement on the enclosing structure before anything is
carried. A small structure otherwise lands where the *larger* structure's
average deformation puts it, which is the classic failure mode of
propagating a tumour through a whole-thorax registration.

The refinement replaces the active registration, so the sidebar reports
exactly what the propagation used rather than something that happened
invisibly. Its method and parameters are the ones chosen in the sidebar
(forced to a deformable method — a rigid one would replace the alignment
instead of refining it), and its margin is set in the propagation window.

## Using it

1. Register the two datasets (any method; see
   [registration.md](registration.md)).
2. *Tools ▶ ⇄ Propagate structures…*
3. Choose the source dataset, tick the structures and segmentations to
   carry, optionally pick an enclosing region to refine on, and press
   **▶ Propagate**.

Results arrive named `<structure> (from A)`, in the source structure's own
colour, as the active segmentation of the destination dataset. From there
they behave like anything else painted by hand: edit with the brush, view in
3D, convert to RTSTRUCT, export as DICOM.

## Verification

`src/propagate.rs`'s unit tests assert that a translation carries a ball by
exactly that much (centroid within 0.5 mm, volume preserved to 6 %), that
the direction flag really reverses the mapping, that a structure mapped
outside the destination comes back *empty* rather than wrong, and that a
structure crosses between a 2 mm and a 3 mm grid with its volume intact to
10 %.
