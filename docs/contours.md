# Drawing and editing structures

Contours drawn as contours: a polygon, a spline or a freehand stroke goes
into the RT structure as the curve it was drawn as, not as the outline of the
voxels it happened to cover. A segmentation is a voxel mask -
[segmentation.md](segmentation.md) - and the drawing tools are one set for
both: each acts on whichever is selected, in that one's own representation,
and the two representations meet where each is better, by the rule below. A structure that was generated
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

So `crates/rds-domain/src/contours.rs` carries the geometry as geometry:

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
| The same tools on a segment | voxels, through the slice's outline where the tool is a curve | A segment is a mask; a curve fills the voxels whose centre it encloses |
| Boolean algebra between structures, margins, morphological cleanup, distance transforms | voxels | The exact anisotropic distance transform is a voxel algorithm ([structure-algebra.md](structure-algebra.md)) |
| Everything learned, region growing, thresholds | voxels | The engines produce label maps |
| The 3-D view | triangle mesh | Surface nets over the mask |

## The tools

The toolbar's **✏ Draw structure** button unfolds one row of eight glyphs
on the toolbar itself, which take over the left mouse button. After the row
come the options of the tool in hand - the radius, the edge band, the draw
mode - and the **target line**: what the tools act on. Folding the row puts
the tool down.

**One row for structures and segments.** The row is the same whatever is
selected: each tool acts on the *subject* the target line names, an RT
structure or a segment, and the program picks the representation. The
user picks a tool for what it does, never for what kind of thing it works
on; the tooltips and the status bar's *?* say what the tool in hand does to
the current subject.

| Tool | On an RT structure | On a segment |
|---|---|---|
| **🎨 Paint** | the contour brush: the swept stroke is added to the slice's contours, pushing the outline out; with *3D* a ball, cut into the contours of every slice it reaches | the voxel brush, a disc on the slice or a sphere through the slices (*3D*) |
| **⊖ Erase** | the same brush, cutting the outline back, on the slice or (*3D*) through the slices | the voxel brush, erasing |
| **✨ Grow** | the grown region's outline merged into the contours of every slice it reached | the grown voxels added to the mask |
| **📐 Polygon**, **✒ Spline**, **✏ Free**, **🔗 Live wire** | the closed curve added or cut by the *Auto / Extend / Subtract* rule | the voxels whose centre the curve encloses filled in or cut out on that slice, by the same rule |
| **⌖ Nudge** | the outline's vertices pushed | the slice's outline pushed, and filled back into the mask |

* **📐 Polygon** - click by click. Right-click, double-click or `Enter`
  closes it; `Esc` throws it away.
* **✒ Spline** - the same clicks, joined by a closed centripetal
  Catmull-Rom spline. Four clicks are a smooth organ outline.
* **✏ Free** - press, drag round the structure, release. The stroke closes
  itself and is thinned on release, so a dense drag does not store two
  thousand points.
* **🔗 Live wire** - click once on the boundary, move along it, and the
  curve between the two follows the image edge instead of the straight line
  the hand would have drawn. Every click anchors what is on screen; right-
  click, double-click or `Enter` closes. See *Following the edge* below.
* **🎨 Paint** and **⊖ Erase** - a round brush *on the patient*. On a
  structure the swept stroke is added to the slice's contours (Paint) or cut
  out of them (Erase); on a sagittal slice of a 1 x 3 mm lattice the stamp is
  an ellipse on the lattice and a disc on the patient, which is the only way
  a brush radius in millimetres means anything. On a segment it is the voxel
  brush of [segmentation.md](segmentation.md), with its *3D* sphere. `Alt`
  reverses either tool on both. A structure is painted on the slice in view
  unless *3D* is on (the toggle is kept per kind: on for a segment, off for
  a structure until asked for); then the brush is a ball, below under *The
  3-D contour brush*. This is the tool a department uses most.
* **✨ Grow** - press a seed, drag up and down to widen or narrow the
  region, release to keep it ([segmentation.md](segmentation.md)). On a
  segment the voxels are added; on a structure the region - holes inside a
  slice filled, the *inside:* limit honoured - is traced the way a generator
  traces a mask and united with the structure's contours on every slice it
  reached, as one undo step.
* **⌖ Nudge** - push the outline around. Every vertex within the tool radius
  follows the drag with a cosine falloff, so the curve deforms instead of
  developing a corner. The radius is the brush radius: `Shift`+wheel, `[`
  and `]`. On a segment the slice's outline is read once when the drag
  starts (the marching-squares trace a generator makes, resampled to half a
  voxel), pushed by every sample and filled back each time: kept for the
  whole drag, so a push of a fraction of a voxel is not rounded away and
  many of them add up as they do on a contour.

All eight work in **any** of the three views. `Ctrl`-click picks what is
under the pointer - a structure whose contour encloses it, or a segment that
holds the voxel, the subject's own kind looked at first - and makes it the
subject. `Ctrl+Z` takes back the last stroke of the subject, whichever tool
made it: one step per stroke, curve or drag, on either kind.

### The 3-D contour brush

With *3D* on and a structure the subject, Paint and Erase are a ball of
the brush radius in patient millimetres. On every slice of the drawing
plane the ball reaches, it adds to that slice's contours, or cuts out of
them, the disc where the sphere meets the slice - a capsule, swept along
the drag. How far it reaches is the radius over the spacing *along the
slice normal*, and every distance is measured in patient LPS through the
image's direction vectors, so on a tilted acquisition (slices stacked
along a direction that is not their normal) the circle on a neighbouring
slice is smaller by the true distance and shifted by the tilt, as the
sphere really crosses it. On a sagittal or coronal view the slices are
that plane's, and the result is re-cut onto the axial contours the
structure is stored as, as any off-axis stroke is (the axial slices it
reached; *The drawing plane* below).

The obvious implementation - a contour boolean on every reached slice on
every mouse sample - is what kept the brush on one slice: five to fifty
booleans per sample cannot follow a hand. So a stroke keeps a **raster of
what the ball has covered** (`contours::BallFootprint`): each sample only
marks the cells under the ball, finer than the voxels (a radius is at least
24 cells, a cell no finer than 0.2 mm), on every slice it reaches - a few
thousand distance tests, nothing else. The cells are merged into the
contours slice by slice: the slice being drawn on with every sample, so
the stroke under the pointer shows as it goes; each other slice every sixth
sample (a sixth of them with each sample, so no sample carries them all:
ten times a second at 60 Hz), and everything left on release. The
schedule is counted in samples, not in time, so the same stroke lands the
same however fast the frames came.

A merge covers only what was marked on that slice since its last merge
(`BallFootprint::merge`): a window around those cells, on the boolean's
own lattice of fine cells. Inside the window the answer is made from the
slice as the stroke found it and the stroke's cells (the outline of the
cells, marching squares between their centres, asked cell by cell), never
from the contours the last merge left, so however often a slice is merged
nothing compounds. Outside it the contours are what they were, vertex for
vertex, and the window's trace is spliced into them where they cross its
border (*Booleans within a window* below). Merging the whole slice from
where the stroke started, on every merge, is what this replaces: on a
body outline that was 50 ms a sample and 0.6 s every 100 ms for a 20 mm
ball (dev build); now it is 5 ms a sample, and the release 4 ms.

The other views, the 3D scene and *Save changes to the PACS* see the
contours as they are merged, through the same writes as every other
contour edit. The whole stroke is one undo step; `Alt`, `Shift`+wheel and
`[` `]` work as on the slice.

The smart brush's edge band applies voxel by voxel through the ball, as it
does through the segment brush's sphere: a cell is marked only if the voxel
it lies in is in the band. The slice brush's "keep the one piece the stroke
stands on" has no meaning in a sphere and is not applied. A planar pane is
one image, so it has no *3D*.

### What a stroke does to what is there

| Mode | Effect |
|---|---|
| **Auto** | Crosses nothing → a new contour. Crosses something and starts *inside* it → extend. Crosses something and starts *outside* → cut. |
| **Extend** | Always add. |
| **Subtract** | Always cut. |

Auto is the rule planning systems have settled on, because a planner should
not have to learn a second set of habits to draw a contour here. On a
segment the rule is asked of the slice's own outline (the trace of its
voxels), so a curve does the same thing to a mask as to a contour. The
brush does not follow the mode: Paint adds and Erase cuts, on a structure as
on a segment, and `Alt` reverses either - which tool is in hand says what a
stroke does, so it cannot depend on where the stroke happened to start.

### Following the edge

Two of the tools read the picture rather than only the pointer, and both
work on the image *as it is windowed on screen*: change the window and the
edges they follow change with it, which is the honest behaviour, because
the edge a planner is aiming at is the one they can see.

**🔗 Live wire** ([`crates/rds-domain/src/livewire.rs`](architecture.md#module-map)) is
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

**The smart brush** is Paint and Erase with an *edge* setting. The stamp is
cut back to the pixels whose value falls in the band - *bone* and *air* are
the CT numbers themselves, *bright* and *dark* are relative to the display
window, so they mean something on MR and PET too. On a structure it is then
cut to the one piece of that which the stroke is standing on, so a brush
overlapping a second organ across a gap does not fill it; on a segment the
band is applied voxel by voxel through the disc or the sphere, the
threshold-limited paintbrush of a mask editor
(`Segmentation::paint_capsule_within`). *reach* is how far past the
threshold it may still paint, as a fraction of the window. With the setting
on *any* it is the plain brush, stamp and all.

### What is being edited

One thing at a time, the **subject**: an RT structure or a segment. The rule
is the Structure editor's own (`ViewerApp::edit_subject`, below under
*Editing a segmentation*): a workspace that holds only one kind is edited as
that kind; one that holds both has the *Edit: Structure | Segmentation*
switch, on the target line as in the editor, and the two never disagree.

The target line after the row reads `▸ name` in the subject's colour. It is
a picker: the structures of the active set and the segments of the active
segmentation series are listed under their headings, and choosing one makes
it the subject, of whichever kind; *New RT structure* and *New
segmentation* at its foot start one of either. The **+** beside it starts a
new one of the subject's kind. Clicking a structure in the sidebar's *RT
structures* list makes it the subject, and clicking a segment in the
*Segmentations* list makes that one the subject (the tick box beside either
is only whether it is shown); so does `Ctrl`-clicking it in a view. Or
simply start drawing: with nothing chosen the first stroke **creates** one
of the subject's kind (and a structure set or a segmentation series to hold
it, if the study has none). It creates rather than adopting the list's first
structure on purpose - a stroke must never land in a structure nobody
pointed at. *+ Empty structure* in the editor's *Insert structure* section
makes another structure too.

### On a planar pane

The same tools draw on the planar panes, on the 2D contours of a planar
structure set: a stroke on a pane makes the pane the tools' target (the
target line reads `▸ 2D · set · structure`, and Ctrl+Z, Esc and Enter act
there), a stroke in an MPR view makes the volume the target again. A planar
image has no segments, so the subject there is always the planar structure:
Paint and Erase are its contour brush (only *bright* and *dark* of the edge
band mean anything on a radiograph), and Grow, which needs a volume to grow
through, does nothing on a pane. The rings are in the image's pixels and the
brush radius in millimetres of its spacing; the rule of *Auto / Extend /
Subtract* is the one above (`contours::auto_cuts`), the spline the same
curve (`contours::spline_points`).
See [planar-2d.md](planar-2d.md).

## The Structure editor

*Modules ▶ Structure editor* (right panel, F10; switch it on in the *Modules*
menu) is where a whole structure is made or changed, in three foldable
sections: **Insert structure** (an empty structure, a point of interest, or a
generated one -
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
* **Move the structure** (a foldable block) - translate by x / y / z
  millimetres along the image's own axes, rotate by three angles about its
  centroid, put its centroid under the crosshair (*move to slice
  intersection*), or scale about its centroid. A translation is exact: the
  contours keep their shape in the plane and move by whole slices along the
  stacking axis, as a planning system moves a structure. A rotation about
  the stacking axis and the scaling act on the contours as they are; a tilt
  goes through a voxel mask - rasterized on a lattice up to four times finer
  than the image in the plane and only as large as the structure, resampled
  at the moved position and traced again - so the structure lands on the
  slices it now crosses without visibly losing its shape. **Draw axis**
  switches the left button in the views of the editor's workspace to drawing
  an axis (a white line, one slice thick, shown in every view and in the 3D
  window); *Move* then shifts the structure by so many millimetres along it,
  from its first point towards its second, and *Rotate* turns the structure
  about it. The ✋ next to *Draw axis* moves the axis by hand instead: drag
  near an end to move that end, anywhere else to move the whole line.
  **Keep** finishes the axis: it stays where it is, the left button goes
  back to the crosshair so the image can be looked through without a drag
  redrawing the axis, and the line stays on screen and usable whether *Draw
  axis* is on or off (the ✋ still moves it, because that is asked for
  explicitly). Without *Keep*, the axis is dropped when the toggle is
  switched off. The line below keeps
  axes: **Save** writes the axis as two points in patient millimetres to
  `user_data/structure_editor/user_axes` in the program's data folder
  (`%LOCALAPPDATA%\RustDICOMStation` on Windows), a plain text `.axis` file
  that reads back on any workspace of the same frame; **Load** picks one from
  there; **Save last** / **Load last** keep one axis in memory for the
  running program only. An axis loaded from either arrives kept, since it
  is already the one that was wanted. The ✋ in the block's own
  title line drags the selected structure with the left button in any view,
  in the plane of that view, one undo step per drag. **Back** undoes the
  last edit of the workspace; **Reset** puts the structure back where it was
  before the first move. A structure moved past the first or last image
  slice is not cut: the slices outside the field of view are kept with
  their signed slice index, follow every further move, and come back into
  the image when it is moved back (only a tilt, which goes through the
  voxel mask, is limited to the image).
* **Type** - the RT ROI Interpreted Type (`PTV`, `ORGAN`, `EXTERNAL`, …),
  which is what a planning system branches on. It is the first line of the
  section, under the structure's name and its volume.
* **Derived**, when the structure carries a recipe: the recipe as a line, its
  status, and Update / Edit recipe / Underive. See
  [structure-algebra.md](structure-algebra.md#derived-structures-the-recipe-stays);
  editing a derived structure by hand here marks it *overridden*, because the
  recipe no longer describes what is on the screen.

### Editing a segmentation

The section above is built around contours, and most of it means nothing to a
mask: there is no interpolation between slices of something already filled
everywhere, no point limit, no outline to smooth. The operations a painted
mask *does* need are ones the program already has for its generators and its
auto tools, so when the editor is on a segmentation the section shows those
instead.

**Edit:** at the top of the section switches between the two. It appears only
when the workspace holds both an RT structure set and a segmentation series -
with one of them alone there is nothing to choose, and the editor follows
whatever is there rather than saying "nothing is selected" when it means "not
that kind". The subject is the selected segment of the segmentation series,
and it is the drawing tools' subject too: the toolbar's strip carries the
same switch on its target line, so what the editor's section shows and what
the tools draw on are always the same thing
([segmentation.md](segmentation.md)).

The header is the segment's colour, its name, its volume in cm³, how many
slices it touches and how many voxels it holds. Under it:

* **Tidy** - *Keep largest* throws away every connected piece but the
  biggest, which is what a threshold's specks need; *Remove holes* fills the
  enclosed holes slice by slice - the marrow inside a bone, the lumen inside
  a wall.
* **Grow and shrink** - a margin in millimetres on the volume's own lattice,
  so anisotropic spacing is accounted for rather than counted in pixels.
* **Whole segmentation** - *▣ To RT structure* traces the mask into closed
  planar contours in the active structure set and switches the editor onto
  it, so a mask can be finished with the contour tools; the segmentation
  itself stays where it is. *Clear* empties the mask and keeps the
  segmentation; *🗑 Delete* removes it from the series.

None of this is a new algorithm: it is the cleanup pass, the millimetre
morphology and the mask-to-contour tracer the rest of the program already
runs, given a place in the editor. Each of these operations is one undo step,
recorded as the voxels it changed, so it takes back exactly like a brush
stroke - and where a change is so wholesale that recording it would cost more
than the mask itself, the history is dropped rather than left describing a
mask that is no longer there.

## The drawing plane

A structure is *stored* as axial contours, because that is what every reader
expects. Drawing in a sagittal or coronal view therefore cuts the geometry
along that axis first, and writing it back cuts it into axial contours again,
exactly as a planning system does when you change drawing plane; it is
announced when it happens. Between the two conversions the working stack
is cached, so a session of sagittal strokes costs one conversion in, not one
per stroke, and the shape does not erode with each. While a brush, nudge
or drag stroke goes on in such a view, the axial contours are written back
at most a few times a second and once more when the button is let go, and
only the box the structure occupies is converted - and of it, only the
axial slices the edits since the last write reached: the vertices that
changed (with their neighbours along the ring) say which, since every edit
keeps the vertices it does not reach. The other axial slices keep the
contours they had. On a body outline a write of a 20 mm ball's stroke went
from 130 ms (all of it re-cut) to 21 ms (its 15 slices), dev build.

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
are meant to be the same landmark in two workspaces, that separation is the
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

**Booleans within a window.** A brush stroke changes a slice only where
the brush went, but a bounding box of both operands is the whole body
outline when the brush sits on its edge: a few million fine cells per mouse
sample, and the whole outline re-traced. So a brush stamp, a ball's merge
(and every `add_ring` / `subtract_ring`) rasterize only a **window**
around what changes - on the same lattice of fine cells, so the cells are
the ones the whole raster has there - and splice its trace into the outline
outside, which keeps its own vertices. The window's cells traced between
their centres give loops (wholly inside) and chains from border to border;
the outline as it was gives pieces that leave the window and come back.
The window's border cells are filled from the outline as it was, so every
piece meets a chain end of the same kind (leaving or entering) beside it,
in the same order round the border: they are matched in that order and
joined. Anything that does not match - an outline crossing the border
twice between two cell centres, a vertex grazing it - and any result whose
area is not the one its cells give is refused, and the window is tried
again with its border elsewhere, then the whole raster does it as before.
The slice brush on a body outline went from 50 ms to 2 ms a sample (20 mm,
dev build).

While you draw, the volume under the structure's name is planimetry on the
contours themselves - the area of every slice times the slice thickness -
which is cheap enough to recompute every frame and finer than counting
voxels. *Structure details* and the propagation report give instead a
**surface-based** volume: the volume inside a closed triangle surface
reconstructed from the contours - neighbouring slices joined by strips of
triangles, the ends closed by caps half a slice beyond the last contours
(the formulas of all three are in [volumes.md](volumes.md)). On an organ
the two agree to within a per cent; a small structure reads smaller as a
surface, by its end caps.

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

`crates/rds-domain/src/contours.rs`'s own tests cover the geometry: areas, orientation and
centroids; even-odd nesting; Douglas-Peucker keeping exactly the corners of a
walked square; the four boolean operations against the **analytic** area of
two overlapping discs; a subtraction that cuts a real hole; the exact fast
paths leaving untouched rings untouched (by `assert_eq!` on the ring itself);
tidying; a mask round-tripping through contours voxel for voxel; the trip
through patient coordinates; a sagittal stack converting to axial with its
volume; interpolation of a sphere; components; the point cap; the transforms;
the windowed boolean against the stamp's own area, voxel for voxel against
the whole raster around the stamp, and every far vertex kept (on a circle,
across a long edge of a square, into a hole); a ball merged sample by
sample against the same stroke merged at once; a slab of axial slices
re-cut from a sagittal stack against the whole re-cut, the other slices
of the ROI kept. `examples/brush_bench.rs`
times a stroke on a real structure both ways (`--body` makes a body
outline from the CT), and the `#[ignore]`d `brush_frame_timing` in
`tests/ui.rs` times the frames of a drag delivered one move a frame.

`crates/rds-domain/src/livewire.rs` has its own: a path between two points of a disc's rim
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
