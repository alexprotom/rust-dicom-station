# Contour and segmentation propagation

*Modules ▶ Structure propagation* carries any RTSTRUCT ROI or painted
segmentation across a registration and lands it as an ordinary, editable
segmentation, convertible back to RTSTRUCT and exportable as DICOM. It is a
section of the right panel, next to the image registration that drives it.

The destination is either **the image at the far end of the last
registration** - whichever workspace it is in, which is what pairs the two -
or a **4D group** of any open workspace, which the module registers as it
goes: the whole group, or any **single phase** of it, each listed under its
group in the *To* list. One phase is the same run with everything else left
out - end-exhale alone is often all that is wanted, and it costs one
registration rather than ten.

The registered image is in that list by its own name - the series, and which
end of the registration it is - rather than as a standing phrase, and it is
in the list only while there is a registration. With none, the entry is not
offered at all and the destination falls to the first 4D group, because an
entry that cannot be chosen is worse than a shorter list.

**Without a registration, every loaded image series is a destination.** They
are listed under a rule of their own, below the groups and their phases, and
the run registers the source onto the one picked before carrying the
structures - exactly what the group path already does for each phase, done
once. So structures go from anything to anything: a planning CT onto a
diagnostic one, a cardiac CT onto a cone-beam scan, either direction. A
series that is already a phase of a 4D group is not listed twice; it is
reached as that group's phase. With a registration active the list is
unchanged, because then the two images it pairs are what "the other image"
means.

## What it does

* **Pull, never push.** Every voxel of the *destination* is asked where it
  comes from; pushing a deformed mask forward would leave holes where the
  deformation expands and double-write where it compresses.
* **Sub-voxel boundaries.** The source mask is sampled trilinearly and
  thresholded at ½: the boundary lands where the contour really is, and
  structures cross grids of different spacing and orientation.
* **Either direction.** The transform maps fixed → moving, so propagating
  *onto* the moving workspace runs through its inverse; you only choose the
  source workspace.
* **A cached mapping.** A deformable inverse is a fixed-point iteration,
  twelve control-lattice evaluations per point - billions of operations
  over a 512³ study. So the mapping is evaluated on a 3 mm lattice across
  the destination bounding box and interpolated in between: exact for a
  rigid transform, far below the contour's accuracy for a deformable one.

Each propagated structure is reported as `name: 164.213 cm³ ▶ 170.087 cm³
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

## Deformed, or keeping its shape

**Carry: deformed / keep shape** says how each structure goes through the
transform. *Deformed* (the default) carries it point by point, so its shape
and volume change the way the deformation changes where it lies. *Keep
shape* carries it as a rigid body instead: the transform's best rigid fit
over the structure's own voxels (orthogonal Procrustes). It lands where the
transform takes it and turns as the tissue around it turns, and keeps its
own shape and volume.

This matters most for a small target inside an organ whose outline was
what the registration matched, such as a cardiac target anchored on the
heart's contours. The transform knows where the heart's surface goes and
only interpolates inside it, so a target a few millimetres under that
surface is squeezed by whatever the interpolation does at that spot. In
one test pair (the cardiac CT of one patient onto the 4DCT of another, the
two hearts contoured at 947 and 794 cm³), the heart as a whole lost 16 %
but the deformation compressed the tissue at the target by a third (a mean
Jacobian of 0.65 over it): 2.90 cm³ became 1.89. Carried with its shape
kept, it lands at 2.89 cm³.

What the rigid carry leaves out is reported: the RMS distance between the
rigid body and the transform over the structure, marked *rigid* on the
structure's row, with the number on its tooltip (and `rigid_residual_mm`
from the MCP server). A few millimetres over a compact target means the
deformation there was mostly a squeeze; a large figure over a spread-out
one means the transform bends it, and a single rigid body is a
simplification you have chosen to make. An anchor always follows the
transform, whichever is chosen: it is the run's check on the transform.
From the MCP server this is `keep_shape` on `propagate` and
`propagate_to_group`.

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
views show them when that phase is displayed. Every phase gets its own row in
the run's report.

Picking one phase instead of the group narrows the same machinery to that
phase: one registration, one segmentation series, one row in the report -
including for an anchored run.

**Refine locally first** applies only to a run that goes through the active
registration: it is a second pass over that one pair of images. A
destination that makes its own transform - a 4D group, a lone series - has
nothing standing there to refine, and the section now says so with its
controls greyed rather than refusing to open, which is how it came to look
broken once a 4D group became the usual destination.

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
with the contour the phase already has. Beside the name is a colour swatch -
the same one the data tree gives a structure - and it is worth using: the
landed copy is drawn right on top of the phase's own contour of that organ,
and two curves of the same colour lying across each other is exactly the
picture the check was meant to make readable. It starts on the anchor's own
colour, and **↺** puts that back.

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
2. *Modules ▶ Structure propagation*, or **⇄ Propagate structures** in the
   registration module once it has a result.
3. Choose the source image (any series of any open workspace; through a
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

**Colour.** A copy lands in the colour of the structure it came from, which
is what a propagation means by default: the same anatomy, on another image.
Where a copy has to be told apart from an original at a glance - a
propagated target lying next to the one drawn on that phase, say - the
swatch in front of each structure in the list sets the colour that copy
lands in. It changes the copy only; the structure it was propagated from
keeps its own colour, the row says *recoloured*, and *Source colours* drops
every choice again. The colour rides along the whole way, so it applies to
every run the module starts: through a registration, onto a 4D group, and
anchored on a structure.

**Transform matrix** (foldable, under the run) is the same 4 × 4 the
registration module carries
([registration.md](registration.md#the-matrix-typed-in-by-hand)), and it
shows the transform the destination's own last run produced until it is
taken over, so what is on screen is what the next run will do. Against a 4D
group that is one transform per phase, and the picker above the grid says
which phase it is showing; against one phase it is that phase's alone. With **Use this matrix**
ticked the run carries the structures through those numbers instead of the
registration's transform: the pairing of images is still the
active registration's, the numbers are yours. That is how a known couch
shift or a transform from another program carries a set of structures, and
how a propagation is checked against a shift whose answer is known in
advance. To do it with no registration run at all, type the matrix into the
registration module and press *Apply as the registration* first - the
pairing is then the two images you chose there.

## What the run reports

*Last run* is two tables rather than a paragraph, because ten phases of four
facts each is forty sentences nobody reads to the end.

The first has a row per destination: the **phase** (on screen only when
there is more than one destination to tell apart), what the registration did
to the **metric** (its value before and after, with the stages on the row's
tooltip), the **iterations** and the time (**t, s**) it took, the anchor's
**Dice** where the run was anchored on a structure, and what the results were
**filed as**. The second has a row per structure per destination: its volume
at the **source**, as the transform **deformed** it, and - only when closing
or filling was asked for - as it was **filed** afterwards; without either of
those the filed volume is the deformed one to the last decimal, so the column
is left out rather than repeating its neighbour. The last column is the
**change** between the source and the result, in the warning colour past ten
per cent, which is where a propagated volume stops being the same organ.

When the structures land as a **structure set**, what arrives is contours,
and a contour has two volumes - the two *Structure details* shows. So the
second table measures both sides both ways, **Surface** first, then
**Voxels** (the contours rasterized on the image's lattice): under each, the
**Source**, the **Deformed** ROI that was filed, and the **Δ %** between
them, so either measure reads across on its own. The deformed figures are
read off the filed ROI with the very calls *Structure details* makes, so the
two windows agree to the last digit.

**Surface** is the volume 3D Slicer's *Segment Statistics* reports for a
structure imported from RTSTRUCT, computed by the same algorithm
(`src/rt_surface.rs`, a port of SlicerRT's planar-contour-to-closed-surface
conversion and of the VTK filters it uses). Slicer does not measure the
contours; it joins them into a closed surface and reports the volume inside:

- each contour is joined by a ribbon of triangles to every contour on the
  next slice whose bounding box overlaps it;
- a contour with nothing joined above or below gets a *smooth end cap*
  (SlicerRT's default): the contour rasterized, eroded until at most half of
  it is left, traced again and set half a slice away, then joined to the
  contour with another ribbon;
- before any of that, a path that comes back to one of its own points is cut
  into separate lines by SlicerRT's *keyhole* rule, which keeps a pair of
  voxels chained through a corner as two triangles of half a voxel each;
- the volume is `vtkMassProperties`' divergence sum over the triangles.

On an organ the surface and the voxels agree to within half a per cent: the
end caps are a sliver of the whole. On a target exported voxel by voxel as a
thin sheet they part by tens of per cent, because such a structure is mostly
open ends. The STAR target (1031 squares of 1 mm on 0.6 mm slices) is
0.619 cm³ of voxels and 0.534 cm³ of surface; the UPSTAR target, with 450
corner-chained pairs among its cells, is 2.90 cm³ of voxels and 1.95 of
surface. Neither is wrong; they measure different things. A filed ROI's
contours run along the edges of the voxels it was made from (see *Mask →
RTSTRUCT* in [segmentation.md](segmentation.md)), so its voxel volume is its
contour area times the slice spacing exactly, and its surface is smaller by
its end caps. So a small structure's two **Δ %** columns can disagree, and
it is the voxel one that says how much the transform itself changed.

Checked against Slicer 5.10 on four structures of two test patients, the
surface volume here and Slicer's agree to every digit Slicer shows: the STAR
target 0.533518 cm³ and heart 910.498, the UPSTAR target 1.95196 and heart
949.369. Slicer's *End capping* parameter set to 2 (straight caps: a copy of
the end contour itself half a slice away) gives 0.599 and 2.267 cm³ for the
two targets, not the voxel volume: SlicerRT copies a straight cap's closing
point as a point of its own, its ribbon matcher then takes the cap for an
open line, and at a top end the ribbon folds a sliver in. An earlier version
of this page predicted 0.619 and 2.45 there; those were wrong.

A structure that came from a segment has no contours on the source side;
its surface and the change by it are left as a dash rather than reported as
nought. Every volume in the program - these tables, the details, the tools'
own confirmations - is given to two decimals.

**📋 Copy** puts both tables on the clipboard tab separated, which a
spreadsheet opens as a table without being asked twice.

## Verification

`src/propagate.rs`'s unit tests assert that a translation carries a ball
by exactly that much (centroid within 0.5 mm, volume preserved to 6 %),
that a ball carried with its shape kept through a transform that halves its
volume keeps its volume to 3 % and lands within a voxel of where the
transform put it,
that the direction flag really reverses the mapping, that a structure
mapped outside the destination comes back *empty*, and that a structure
crosses between a 2 mm and a 3 mm grid with its volume intact to 10 %.
