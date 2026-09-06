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

The rule is the one RayStation states and it is not arbitrary: *the operation
decides*, and the conversion is automatic.

| Operation | On | Because |
|---|---|---|
| Drawing, nudging, per-slice editing, interpolation | contours | It is what the user sees and what the file stores |
| Boolean algebra between structures, margins, morphological cleanup, distance transforms | voxels | The exact anisotropic distance transform is a voxel algorithm ([structure-algebra.md](structure-algebra.md)) |
| Everything learned, region growing, thresholds | voxels | The engines produce label maps |
| The 3-D view | triangle mesh | Surface nets over the mask |

## The tools

The toolbar's second group takes over the left mouse button, like the brush:

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

All five work in **any** of the three views. `Ctrl`-click picks the structure
under the pointer (RayStation's pick tool); `Ctrl+Z` undoes the last contour
edit of the dataset under the pointer.

### What a stroke does to what is there

| Mode | Effect |
|---|---|
| **Auto** | Crosses nothing → a new contour. Crosses something and starts *inside* it → extend. Crosses something and starts *outside* → cut. |
| **Extend** | Always add. |
| **Subtract** | Always cut. |

Auto is RayStation's rule verbatim, because a planner should not have to
learn a second set of habits.

### Which structure is being edited

One at a time, marked **✏** in the sidebar's *RT structures* list and named
in the toolbar. Set it there, `Ctrl`-click a contour in a view, or simply
start drawing: with no structure chosen the first stroke **creates** one (and
a structure set to hold it, if the study has none). It creates rather than
adopting the list's first structure on purpose - a stroke must never land in
a structure nobody pointed at. *+ ROI* in the list, and *+* in the toolbar,
make another.

## The contour window

*Tools ▶ 📝 Contour tools* - everything that acts on the whole structure
rather than on the stroke under the pointer. Every button is one undo step.

* **Interpolation** - contours for the slices between the drawn ones. The two
  neighbouring regions are turned into signed distance fields, blended, and
  the zero level re-traced, which handles what point matching gets wrong: a
  structure that splits in two, or one whose contours barely overlap. Shown
  **dashed** and stored only when accepted - one slice or all of them - the
  rule RayStation is explicit about, and the reason *Thin* exists next to it:
  keep every n-th slice, correct two, interpolate again.
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
  which is what a planning system branches on.
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

## Verification

`src/contours.rs`'s own tests cover the geometry: areas, orientation and
centroids; even-odd nesting; Douglas-Peucker keeping exactly the corners of a
walked square; the four boolean operations against the **analytic** area of
two overlapping discs; a subtraction that cuts a real hole; the exact fast
paths leaving untouched rings untouched (by `assert_eq!` on the ring itself);
tidying; a mask round-tripping through contours voxel for voxel; the trip
through patient coordinates; a sagittal stack converting to axial with its
volume; interpolation of a sphere; components; the point cap; the transforms.

`tests/contours.rs` covers the seam with the rest of the viewer: the stack's
filler and `segmentation::rasterize_roi` agreeing voxel for voxel on a hollow
shell, on a feet-first lattice; a structure surviving the trip through patient
space; drawing a circle on every fifth slice and interpolating a cylinder to
within a few percent of the analytic volume; a subtract stroke taking a bite
of the size it looks like while the other slices keep their exact points;
the tidying operations; and a sagittal drawing landing as axial contours.
