# Combining structures and segmentations

Union, intersection, subtraction and symmetric difference over any mix of RT
structures and segmentations, with a margin on any operand and on the result,
and a little tidying at the end. The everyday arithmetic of a planning
department, done in the viewer instead of by hand.

## What it is for

`Lungs = Lung_L ∪ Lung_R`. `PTV = CTV + 5 mm`. `PTV_eval = PTV ∩ (BODY −
5 mm)`. `Ring = (PTV + 10 mm) − (PTV + 2 mm)`. `Parotid_spared = Parotid_L −
(PTV + 3 mm)`. None of these is difficult and all of them are tedious, easy
to get backwards, and impossible to check afterwards if you cannot see what
was combined with what.

So the tool does two things beyond the arithmetic. It keeps the operand list
**ordered and visible**, with ↑ ↓ arrows, because three of the four
operations are not commutative in the way people assume. And it prints the
recipe as one line above the buttons —

```
PTV_eval = PTV ∩ (BODY -5.0 mm)
```

— which is the cheapest possible guard against the mistake this tool exists
to make easy: a subtraction with its operands the wrong way round.

## Contours and masks are the same thing here

An RT structure stores contours in patient coordinates; a segmentation
stores voxels on a lattice. Which one you happen to have should not decide
what you can do with it, so every operand is rasterized onto the displayed
series' lattice on the way in — a contour through
`segmentation::rasterize_roi`, a segment on another lattice through
`dicomseg::resample_mask`, a segment already on this one not at all — and
the result goes back out as whichever kind you ask for. Mixing them is the
normal case, not a special one.

That also means the answer is a **voxel** answer, on the displayed series'
grid. Choosing *an RT structure* as the output converts it back with the
usual marching-squares walk, so the contour you get is the outline of the
voxels, not a polygon operation on the input polygons. On a 1 mm CT the
difference is invisible; on a 5 mm one it is a staircase, and the smoothing
option exists for that.

## Using it

*Tools ▶ ◧ Combine structures in dataset A…*, the **◧ Combine** button in
the sidebar, or — usually the quickest — tick the structures you want in the
data tree, right-click and choose **◧ Combine …**, which opens the window
with them already listed in the order they were ticked.

* **Operation** — union, intersection, subtraction or symmetric difference,
  folded left to right over the list. Three operands under subtraction mean
  `A − B − C`.
* **The operand list** — one row each: which structure, and the margin
  applied to it *before* it is combined. This is what makes the tool
  expressive rather than merely convenient: a crop is an intersection whose
  second operand was shrunk first, and a ring is a subtraction between two
  expansions of the same structure.
* **R/L/A/P/S/I** on any row opens six fields instead of one, for a margin
  that differs by direction.
* **Result** — a margin on the combined mask, then the tidying: fill
  interior cavities, smooth, and either keep only the largest piece or drop
  everything under a given volume.
* **Name … as** — a segmentation or an RT structure; for a structure, its
  interpreted type (`PTV`, `ORGAN`, `EXTERNAL`, …), which is what a planning
  system branches on.

The result lands like any other segmentation — editable with the brush,
visible in the 3-D view, exportable — or as a ROI in the active structure
set.

## Margins are in patient directions

"8 mm superiorly" has to mean the same thing on an axial CT, a coronal MR
and an obliquely acquired series. So a margin is given as six numbers in
**patient** directions — right, left, anterior, posterior, superior,
inferior — and the direction cosines decide which array axis each one is and
which way along it. A series stored feet-first grows toward the head just
the same; there is a test that says so.

Positive grows, negative shrinks, and the two may be mixed in one margin:
the expansion runs first, then the contraction.

### The shape of a margin

The structuring element is the ellipsoid whose semi-axis in each of the six
directions is the corresponding number — what a planning system means by
"5 mm laterally, 8 mm superiorly". Three cases, three costs:

| Margin | Structuring element | Cost |
|---|---|---|
| one number | a ball | one distance transform |
| three (symmetric per axis) | an ellipsoid | one distance transform |
| six (one-sided) | an ellipsoid per octant | eight |

The exact anisotropic Euclidean distance transform in
[`morphology.rs`](architecture.md#module-map) does the work, so a margin is
the same in millimetres along every axis whatever the slice thickness, and
its cost does not depend on how big it is. The asymmetric case is the union
of the eight octants of the shape — dilation distributes over a union of
structuring elements — and each octant is reached by three *one-sided*
passes of the same transform, restricted to sources on one side.

Erosion is the complement of dilating the complement, which inherits the
convention that voxels outside the volume are not background: a structure
truncated by the field of view is not eroded at the cut, because nothing is
inferred about what was never imaged.

## What it will not do

* **Cross datasets.** Operands come from the displayed dataset. Carrying a
  structure from the other one is what
  [propagation](propagation.md) is for, and doing it silently inside an
  algebra tool would make the result depend on registration quality without
  saying so.
* **Preserve contour geometry exactly.** As above: the result is the outline
  of a voxel answer.
* **Guess.** An empty operand list, a subtraction with one operand, or an
  operand that rasterizes to nothing on the displayed series is refused with
  a message rather than quietly dropped from the recipe — because a recipe
  missing one of its terms still produces a plausible-looking structure.

## Verification

`src/structops.rs`'s own tests cover the algebra: the four operations on
known bitmaps, left-to-right folding over three operands, a label-map
operand read as a mask rather than as numbers, margins in patient directions
on two lattices stored opposite ways up, mixed grow-and-shrink margins, the
cleanup steps, and the two recipes worth naming — a crop and a ring.

`tests/structops.rs` covers the seam with the rest of the application: a
contour rasterized from patient-space polygons intersected and unioned with
a painted mask, a result converted back to contours and rasterized again
(agreeing to better than half a percent of its volume, on a deliberately
non-convex L-shaped union), a superior margin on a feet-first lattice, a
subtraction the wrong way round coming out empty rather than wrong, and
`keep largest` rescuing a cut that left a sliver.

The margin machinery is checked in `src/morphology.rs` against a brute-force
dilation written straight from the definition, over isotropic, symmetric-
anisotropic, one-sided and axis-disabled margins, plus the identity that a
symmetric margin agrees with the plain ball.
