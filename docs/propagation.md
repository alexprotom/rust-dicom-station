# Contour and segmentation propagation

*Modules ▶ Structures propagation* carries any RTSTRUCT ROI or painted
segmentation across a registration and lands it as an ordinary, editable
segmentation, convertible back to RTSTRUCT and exportable as DICOM. It is a
section of the right panel, next to the image registration that drives it.

The destination is either **the other dataset**, through the registration
that is already active, or **every phase of a 4D group** of either dataset,
which the module registers as it goes.

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
  twelve control-lattice evaluations per point - billions of operations
  over a 512³ study. So the mapping is evaluated on a 3 mm lattice across
  the destination bounding box and interpolated in between: exact for a
  rigid transform, far below the contour's accuracy for a deformable one.

Each propagated structure is reported as `name: 164.2 cm³ ▶ 170.1 cm³
(+3.6 %)` - the volume change the registration panel's Jacobian statistics
also describe; if the two disagree, look harder at the registration.

## Global and local

**Globally**, propagation uses whatever registration is active; one
restricted to a region gives structures inside it the local mapping and
everything else the global one.

**Locally** - when one structure sits inside another that actually
deformed - *Refine locally first* runs a local deformable refinement on
the enclosing structure before anything is carried; otherwise a small
structure lands where the *larger* one's average deformation puts it.

The refinement replaces the active registration, so the registration module
reports exactly what the propagation used; method and parameters come from
that module (forced to deformable), the margin from the propagation section.

## Onto a 4D group

A planning CT with its structures on one side and a 4DCT on the other is the
case where a single transform is wrong: the phases differ by breathing, and
one transform would put every structure where the reference phase is. So
choosing a group as the destination runs **one registration per phase**: the
source volume is registered onto that phase, and the structures are pulled
through that phase's own transform.

The results arrive as one segmentation series per phase, each bound to that
phase's image series, so the tree files them under the right member and the
views show them when that phase is displayed. Every phase reports its own
metric line beside its structures' volume changes.

The transforms are kept. Registering a group in the registration module
(*Fixed image ▶ the group*) and then propagating onto it costs no
registration at all, and the button says **▶ Propagate to N phases** rather
than **▶ Register and propagate to N phases**. They are dropped when the
registration is cleared, or when the moving image changes.

A local refinement belongs to one pair of images, so it is not offered here:
there is one pair per phase.

## Using it

1. Register the two datasets (any method; see
   [registration.md](registration.md)). Skip this when the destination is a
   4D group: the module registers each phase itself.
2. *Modules ▶ Structures propagation*, or **⇄ Propagate structures** in the
   registration module once it has a result.
3. Choose the source dataset and the destination, tick what to carry,
   optionally pick an enclosing region to refine on, and press
   **▶ Propagate**.

Results arrive named `<structure> (from A)`, in the source structure's
colour, as the destination's active segmentation - edit with the brush,
view in 3D, convert to RTSTRUCT, export as DICOM.

## Verification

`src/propagate.rs`'s unit tests assert that a translation carries a ball
by exactly that much (centroid within 0.5 mm, volume preserved to 6 %),
that the direction flag really reverses the mapping, that a structure
mapped outside the destination comes back *empty*, and that a structure
crosses between a 2 mm and a 3 mm grid with its volume intact to 10 %.
