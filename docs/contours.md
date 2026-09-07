# Drawing and editing structures

Contours drawn as contours: a polygon, a spline or a freehand stroke goes
into the RT structure as the curve it was drawn as, not as the outline of the
voxels it happened to cover. The brush and the geodesic grower still edit
voxel masks - [segmentation.md](segmentation.md) - and the two representations
meet where each is better, by the rule below. A structure that was generated
rather than drawn - a threshold, a shape, a dose level - is
[generators.md](generators.md), and it lands here, editable, like everything
else.

## Why a second representation

An RTSTRUCT stores closed planar polygons in patient coordinates, and that is
what the next system reads. Until now every editing tool in the viewer worked
on voxels, so a structure that arrived as contours had to be rasterized to be
touched and re-traced to be stored. On a 1 mm CT nobody notices. On a 3 or 5
mm one the round trip replaces a drawn curve with a staircase, every time, and
the minute somebody spent on the curve is gone.

So `src/contours.rs` carries the geometry as geometry:

* a **ring** (`Poly`) - one closed polygon, in the in-plane coordinates of a
  slice (fractional voxel indices, so nothing depends on the acquisition
  being axis-aligned);
* a **region** - the rings of one slice, filled by the **even-odd** rule,
  which is what RTSTRUCT means (a hole is a second polygon inside the first)
  and what the viewer's own scanline filler has always implemented;
* a **stack** - every slice of one structure, cut along one lattice axis.

## Which representation, when

The rule is not arbitrary: *the operation decides*, and the conversion is
automatic.

| Operation | On | Because |
|---|---|---|
| Drawing, nudging, per-slice editing, interpolation | contours | It is what the user sees and what the file stores |
| Boolean algebra between structures, margins, morphological cleanup, distance transforms | voxels | The exact anisotropic distance transform is a voxel algorithm ([structure-algebra.md](structure-algebra.md)) |
| Everything learned, region growing, thresholds | voxels | The engines produce label maps |
| The 3-D view | triangle mesh | Surface nets over the mask |

## The tools

The toolbar's **✏ Draw structure** button unfolds one row of glyphs on the
toolbar itself: the voxel tools first, then these, which take over the left
mouse button like the brush. After the row come only the options of the
tool in hand - the radius, the edge band, the draw mode, the structure or
segmentation being drawn into with a **+** to start another. Folding the
row puts the tool down.

* **📐 Polygon** - click by click. Right-click, double-click or `Enter`
  closes it; `Esc` throws it away.
* **✒ Spline** - the same clicks, joined by a closed centripetal
  Catmull-Rom spline. Four clicks are a smooth organ outline.
* **✏ Free** - press, drag round the structure, release. The stroke closes
  itself and is thinned on release, so a dense drag does not store two
  thousand points.
* **🖊 Brush** - a round brush *on the patient*: the swept stroke is added to
  the structure, or cut out of it with `Alt`. On a sagittal slice of a 1 x 3
  mm lattice the stamp is an ellipse on the lattice and a disc on the
  patient, which is the only way a brush radius in millimetres means
  anything. This is the tool a department uses most.
* **⌖ Nudge** - push the outline around. Every vertex within the tool radius
  follows the drag with a cosine falloff, so the curve deforms instead of
  developing a corner. The radius is the brush radius: `Shift`+wheel, `[`
  and `]`.

* **🔗 Live wire** - click once on the boundary, move along it, and the
  curve between the two follows the image edge instead of the straight line
  the hand would have drawn. Every click anchors what is on screen; right-
  click, double-click or `Enter` closes. See *Following the edge* below.

All six work in **any** of the three views. `Ctrl`-click picks the structure
under the pointer; `Ctrl+Z` undoes the last contour edit of the dataset under
the pointer.

### What a stroke does to what is there

| Mode | Effect |
|---|---|
| **Auto** | Crosses nothing → a new contour. Crosses something and starts *inside* it → extend. Crosses something and starts *outside* → cut. |
| **Extend** | Always add. |
| **Subtract** | Always cut. |

Auto is the rule planning systems have settled on, because a planner should
not have to learn a second set of habits to draw a contour here. The brush
follows it too: an Auto stroke that *starts* inside the structure adds to it
and one that starts outside cuts into it, decided once when the stroke
starts so the brush does not change its mind halfway through.

### Following the edge

Two of the tools read the picture rather than only the pointer, and both
work on the image *as it is windowed on screen*: change the window and the
edges they follow change with it, which is the honest behaviour, because
the edge a planner is aiming at is the one they can see.

**🔗 Live wire** ([`src/livewire.rs`](architecture.md#module-map)) is
Mortensen and Barrett's intelligent scissors. Every step from one pixel to a
neighbour costs a weighted sum of three terms: zero on a Laplacian zero
crossing and one elsewhere, the inverted gradient magnitude, and a direction
term that keeps the path running along an edge instead of hopping across it.
The cheapest path from the anchor to the pointer is then a shortest path,
and it is computed **once per click**, not once per frame: from a fixed
anchor the whole shortest-path tree is built, and following the pointer is a
walk up parent pointers.

Two deliberate departures from the paper: the cost of a step is multiplied
by its length, so a staircase of diagonals is not cheaper than the straight
line it approximates; and the search is bounded to a box around the anchor,
because a path that wanders across the whole slice is never what was wanted
and the bound keeps one click far inside a frame.

*learn* is the other half of the tool. The cost above knows nothing about
*which* edge is wanted - bone and air both have large gradients - so every
accepted segment feeds a histogram of the gradient magnitudes it ran along,
and afterwards an edge that looks like the ones already taken is cheaper
than an equally strong edge that does not. Half the cost stays static, so a
strong edge the tool has never seen is still cheaper than flat tissue.
*forget* is what to press when moving to a different organ.

**The smart brush** is the 🖊 brush with an *edge* setting. The stamp is cut
back to the pixels whose value falls in the band - *bone* and *air* are the
CT numbers themselves, *bright* and *dark* are relative to the display
window, so they mean something on MR and PET too - and then to the one piece
of that which the stroke is standing on, so a brush overlapping a second
organ across a gap does not fill it. *reach* is how far past the threshold
it may still paint, as a fraction of the window. With the setting on *any*
it is the plain brush, stamp and all.

### Which structure is being edited

One at a time: the one **selected** in the sidebar's *RT structures* list
(click its name, exactly as a segment is selected in the *Segmentations*
list; the tick box beside it is only whether it is shown) and named after
the draw row. Select it there, `Ctrl`-click a contour in a view, or simply
start drawing: with no structure chosen the first stroke **creates** one
(and a structure set to hold it, if the study has none). It creates rather
than adopting the list's first structure on purpose - a stroke must never
land in a structure nobody pointed at. *+ Empty structure* in the editor's
*Insert structure* section, and *+* after the selected structure's name on
the toolbar, make another.

## The Structure editor

*Modules ▶ Structure editor* (right panel, F10; on by default) is where a
whole structure is made or changed, in three foldable sections: **Insert
structure** (an empty structure, a point of interest, or a generated one -
[generators.md](generators.md)), **Edit structure** (below) and **Combine
structures** ([structure-algebra.md](structure-algebra.md)). Right-clicking
a structure in the list and choosing *📝 Edit in the Structure editor*
selects it and unfolds the section on it.

### Edit structure

Everything that acts on the selected structure as a whole rather than on
the stroke under the pointer. Every button is one undo step.

* **Interpolation** - contours for the slices between the drawn ones. The two
  neighbouring regions are turned into signed distance fields, blended, and
  the zero level re-traced, which handles what point matching gets wrong: a
  structure that splits in two, or one whose contours barely overlap. Shown
  **dashed** and stored only when accepted - one slice or all of them -
  which is also the reason *Thin* exists next to it: keep every n-th slice,
  correct two, interpolate again. *Snap to edges* is smart interpolation:
  every interpolated vertex is pulled along the curve's own normal onto the
  strongest edge within a few pixels, so the preview follows the boundary
  the image shows rather than the one the blend guessed. The preview is what
  gets stored, so what is dashed is what you accept.
* **This slice** - copy, paste (in the current draw mode), delete the one
  contour the crosshair is inside, or clear the lot. "This
  slice" is the one the view along the drawing plane is *showing*, which is
  also the slice a stroke would land on; the wheel moves it.
* **Tidy** - resolve contours of the structure that cross each other into one
  clean set of nested rings (the filled area is unchanged); remove holes;
  keep the largest contour per slice; drop contours under an area; cap the
  points per contour; smooth. Smoothing is area-preserving - a plain average
  shrinks a convex outline a little on every pass.
* **The piece under the crosshair** - keep that connected piece alone, or
  delete it and keep the rest. Connectivity is a three-dimensional question,
  so this one goes through the mask and re-traces the contours; it is the one
  operation here that does not leave untouched slices untouched, and it says
  so.
* **Move the whole structure** - translate in millimetres, scale about its
  own centroid, rotate, or put its centroid under the crosshair (*move to
  slice intersection*). In the plane it is drawn on.
* **Type** - the RT ROI Interpreted Type (`PTV`, `ORGAN`, `EXTERNAL`, …),
  which is what a planning system branches on. It is the first line of the
  section, under the structure's name and its volume.
* **Derived**, when the structure carries a recipe: the recipe as a line, its
  status, and Update / Edit recipe / Underive. See
  [structure-algebra.md](structure-algebra.md#derived-structures-the-recipe-stays);
  editing a derived structure by hand here marks it *overridden*, because the
  recipe no longer describes what is on the screen.

## The drawing plane

A structure is *stored* as axial contours, because that is what every reader
expects. Drawing in a sagittal or coronal view therefore cuts the geometry
along that axis first, and writing it back cuts it into axial contours again,
exactly as a planning system does when you change drawing plane; it is
announced when it happens. Between the two conversions the working stack
is cached, so a session of sagittal strokes costs one conversion in, not one
per stroke, and the shape does not erode with each.

## Points of interest

A marker, a reference point, the point the patient is lined up on: in an
RTSTRUCT these are not a separate object but an ROI whose geometry is one
`POINT` contour, which is why they load, draw and export through the same
code as everything else and only what a planner *does* with one is new.

*✱ Point of interest* in the editor's *Insert structure* section makes one
at the crosshair. In the list a
point carries **✱** and its coordinates in the tooltip, and its right-click
menu offers *Localize* (put the crosshair on it), *Move to the crosshair*
and *Make the localization point*. The localization point is **🎯**, one per
structure set, and setting it demotes any other - a patient is lined up on
one point. Any structure's menu offers *+ POI at its centre*, which is the
one point nobody wants to place by hand.

*Localize* works on ordinary structures too: it centres the three views on
the structure's centre of gravity.

Two points can be compared like anything else in *◑ Structure comparison*,
which then reports their separation instead of a Dice score. When the two
are meant to be the same landmark in two datasets, that separation is the
target registration error.

## Templates, and locking a set

The structure list's right-click menu on a **set** carries both.

**🏷 Template** saves the set's *list* - names, types, colours and derived
recipes, and no geometry at all - as a JSON file under the data folder, and
applies a saved one to any other set. Applying creates the structures that
are not there yet and leaves everything that is, so a template can be
applied twice, or on top of what an engine already produced, without
damage. The recipes travel; the geometry fingerprint of the patient they
came from does not, so a derived structure arrives asking to be updated,
which is the truth. A template deliberately runs nothing: the engines and
the generators are one click away and have their own windows.

**🔒 Lock** makes a set read-only - no drawing, no new structures, nothing
removed - and marks it in the list. It is Approval Status (300E,0002) by
another name: a set that arrives approved opens locked, and a set locked
here is written back as approved, so the next system sees the same thing. It
is a guard against a slip of the hand and not an electronic signature: there
is no user, no password and no audit trail behind it, and anyone here can
unlock it again.

## Accuracy

Two paths, and the fast one is exact:

* A stroke that crosses nothing and encloses no existing contour is applied
  by even-odd toggling alone: appended when its interior is empty (a new
  island, or a hole being filled), ignored when it is already filled, turned
  into a hole when subtracted from filled area. Nothing is resampled - every
  ring the stroke did not reach keeps its own vertices, bit for bit. There is
  a test that asserts precisely that.
* A stroke that *does* cross the outline goes through a boolean. Rather than
  a polygon clipper - whose failure mode on the degenerate input real
  contours are full of is a wrong answer, silently - the boolean rasterizes
  the operands locally and **supersampled** (8× per axis, over the bounding
  box of the operands only) and re-extracts the outline with marching
  squares. The boundary lands within half a fine cell of the true one, `1/16`
  of a voxel: on a 1 mm CT, 60 µm, two orders of magnitude below what a hand
  draws. It cannot fail, has no degenerate cases, and handles holes, multiple
  components and self-intersecting freehand strokes alike.

Volumes are reported by planimetry on the contours themselves - the area of
every slice times the slice thickness - which is what a planning system does
and is finer than counting voxels.

## What is deliberately not built

Four things a planning system has that this one does not, and why:

* **Model-based segmentation** - deformable shape models with hint contours.
  The value is the trained model library, not the algorithm; the three
  learned engines ([auto-segmentation.md](auto-segmentation.md),
  [segvol.md](segvol.md), [medsam2.md](medsam2.md)) cover the same need.
* **Atlas-based segmentation.** Every piece exists here - a local archive,
  rigid and deformable registration, propagation - and only the atlas
  library and the label-fusion vote are missing. It is a research project of
  its own rather than a gap in the contouring tools.
* **Editing the surface in 3-D**: pushing a mesh about, rather than the
  outline on a slice. ⌖ Nudge does it in the plane the structure is drawn
  on, which is where a contour is corrected in practice.
* **Beam-specific margin structures** from a loaded plan, and **lung-vessel
  segmentation**. Both are realistic here and neither is contouring.

Simulated organ motion is not missing but lives elsewhere: the
[transform simulator](registration.md) warps a study, its structures and its
dose through an exactly known deformation, which is the same operation from
the other end.

## Verification

`src/contours.rs`'s own tests cover the geometry: areas, orientation and
centroids; even-odd nesting; Douglas-Peucker keeping exactly the corners of a
walked square; the four boolean operations against the **analytic** area of
two overlapping discs; a subtraction that cuts a real hole; the exact fast
paths leaving untouched rings untouched (by `assert_eq!` on the ring itself);
tidying; a mask round-tripping through contours voxel for voxel; the trip
through patient coordinates; a sagittal stack converting to axial with its
volume; interpolation of a sphere; components; the point cap; the transforms.

`src/livewire.rs` has its own: a path between two points of a disc's rim
that follows the rim to within a pixel and a half instead of cutting across
it; a path asked for outside the search box that comes back empty rather
than wrong; training that makes the kind of edge already accepted cheaper
while leaving flat background where it was; and a ring snapped onto the edge
of a disc it was drawn 2.5 px inside of, landing four times closer, while a
ring on a flat picture does not move.

`tests/contours.rs` covers the seam with the rest of the viewer: the stack's
filler and `segmentation::rasterize_roi` agreeing voxel for voxel on a hollow
shell, on a feet-first lattice; a structure surviving the trip through patient
space; drawing a circle on every fifth slice and interpolating a cylinder to
within a few percent of the analytic volume; a subtract stroke taking a bite
of the size it looks like while the other slices keep their exact points;
the tidying operations; and a sagittal drawing landing as axial contours.
