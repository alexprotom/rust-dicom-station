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

## The volume is kept

A destination voxel is not a point. Sampling the source mask only at the
voxel centre is exact when the destination lattice is at least as fine as
the structure, and silently wrong when it is not: a target exported as
1 mm cubes (an ablation map, voxel by voxel) carried onto 2 mm slices lost
four fifths of its volume that way, because most cubes contained no voxel
centre. So each destination voxel is sampled at several sub-points (per
axis as many as the spacing ratio asks for, up to four) and gets an
*occupancy*, the fraction of it that comes from inside the structure. The
sum of the occupancies is the volume of the structure as the transform maps
it, and the mask is filled with the most-occupied voxels until it holds
exactly that volume. For a structure larger than the voxels this is the
usual half threshold; for one smaller, every piece lands in the voxel that
holds most of it. The report lists three volumes: the source's, the mapped
one (what the deformation made of it) and the filed one (the mask).

## After landing: close, fill

**Then: close gaps / fill** works on each landed mask, for a structure that
arrives as a cloud (a map exported voxel by voxel, a thin contour on a
coarser lattice). *Close gaps* dilates every piece by the radius (a
Euclidean ball from the distance transform, so a millimetre is a millimetre
on an anisotropic lattice), which joins everything closer than twice the
radius, then erodes by half of it: one surface about a radius thicker than
the cloud was. It is deliberately not the textbook closing, which hands two
nearby points back as two points because the ball never fits between them.
*Fill* fills the interior slice by slice; with both on, the filling happens
between the dilation and an erosion by the full radius, so the solid's
surface comes back to where the cloud was. A solid structure needs neither,
and the report shows what they changed: the mapped volume stays what the
transform made of the source while the filed one grows. From the MCP server
these are `close_mm` and `fill` on `propagate` and `propagate_to_group`.

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

### Anchored on a structure

A cardiac CT onto a 4DCT is a different problem from a planning CT onto its
own 4DCT. The two are separate acquisitions in separate frames of reference,
so at the identity they do not overlap at all; the cardiac CT is a small,
sharp, contrast-enhanced volume at one cardiac phase, the 4DCT a wide,
coarse, unenhanced one at ten respiratory bins, so a registration of the
whole images would match the wrong things even once it had found the
patient. What is wanted is narrower: put the heart where the heart is on
every phase, and carry the target with it.

**Anchor on a structure** does that when the source and every phase carry a
structure of the same name (`heart_total` on the cardiac CT and in each
phase's own structure set). Per phase: the two centroids are matched, a
rigid registration sampling only the phase's structure plus the margin finds
the rotation and the residual shift, and (unless *Refine deformably* is off)
a local B-spline on the same region takes up what is not rigid. With
*Match the contours* (the default) the two stages compare the anchor's
surfaces, as signed distance maps of the two contours, rather than the
images: a contrast-enhanced cardiac CT and a plain 4DCT cannot be matched
by intensity - mean squares has every incentive to push the bright blood
pool out of correspondence - but their heart contours can, whatever the
contrast, kernel or cardiac phase. Turn it off for two images that are
alike. The ticked
structures travel through that transform, and the anchor travels with them
as the check: its Dice, HD95 and centroid distance against the phase's own
contour are reported per phase with a verdict (good from 0.85, check from
0.7, poor below). A heart that lands on the heart says the target landed too.
The anchor's own copy is filed under the name in *Lands as* (`<anchor>_prop`
by default; `anchor_landed_as` from the MCP server), so it never collides
with the contour the phase already has.

The anchored run always registers afresh; its transforms are kept like any
group registration's, so a later plain propagation onto the same group
reuses them. From the MCP server the same run is `propagate_to_group` with
`anchor`.

## Using it

1. Register the two images (any method; see
   [registration.md](registration.md)). Skip this when the destination is a
   4D group: the module registers each phase itself - or reuses the
   transforms when the group was registered from the registration module
   against the same moving image, on display or not.
2. *Modules ▶ Structures propagation*, or **⇄ Propagate structures** in the
   registration module once it has a result.
3. Choose the source image (any series of either dataset; through a
   registration, one of its two images), the structure set or segmentation
   series to take the structures from (the one drawn on that image is
   preselected), and the destination (through a registration, the other of
   its two images, named; otherwise a 4D group); tick what to carry, pick where they
   land and what is done to them afterwards, optionally an enclosing region
   to refine on, and press **▶ Propagate**.

**Land as** decides the form of the result. *Segmentation series*: editable
masks bound to the destination image (on the displayed volume they join the
active segmentation series; on any other image they become a new series
bound to it, filed under it in the tree). *Structure set*: contours appended
to the destination image's own RT structure set - the set that references
that series, or a new one bound to it when there is none - so on a 4DCT with
one set per phase the target goes next to that phase's heart, which is where
a planning system expects to find it. Results are named
`<structure> (from A)`; a name already in the set gets a counter.

## Verification

`src/propagate.rs`'s unit tests assert that a translation carries a ball
by exactly that much (centroid within 0.5 mm, volume preserved to 6 %),
that the direction flag really reverses the mapping, that a structure
mapped outside the destination comes back *empty*, and that a structure
crosses between a 2 mm and a 3 mm grid with its volume intact to 10 %.
