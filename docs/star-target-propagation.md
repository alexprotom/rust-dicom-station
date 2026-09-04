# A cardiac target from the CCT onto every 4DCT phase

The STAR workflow: the target is drawn on a contrast-enhanced, ECG-gated
cardiac CT; the plan is made on a 4DCT. The two are separate acquisitions in
separate frames of reference, hundreds of millimetres apart in patient
coordinates, one contrast-enhanced and one not, one at a single cardiac
phase and one at ten respiratory bins. What is wanted is narrow: put the
heart where the heart is on every phase, and carry the target with it. This
page is the step-by-step for it, in the viewer and from the MCP server.

## What you need

* The cardiac CT with its structure set: `heart_total` and the target.
* The 4DCT, with `heart_total` contoured on **every phase**, each in that
  phase's own structure set (the set that references the phase's series).
  The anchored run finds each phase's heart through that reference; a
  phase without one stops the run and names the phase.
* The anchor's name has to be the same on both sides (`heart_total` here);
  the match is case-insensitive.

The roles are fixed by the geometry: every 4DCT phase is a **fixed** image
(the destination the target lands on, the planning lattice that stays
untouched), the CCT is the **moving** image (the source of the structures).
That makes the phase → CCT transform exactly the direction propagation pulls
along, with no inversion. Ten phases, ten transforms, one CCT.

## In the viewer

1. **Load.** Dataset A: the folder holding the CCT (and its RTSTRUCT). If the
   4DCT arrived in the same folder it is in A too; otherwise load it as
   dataset B (*File ▶ Add DICOM folder to B*). Check in the data tree that
   the 4DCT shows as one 4D group with its ten phases and that every phase
   has its `RTSTRUCT` with `heart_total`.
2. **Open the module.** *Modules ▶ Structures propagation* (right panel, F10).
3. **From image** = the CCT series (any series of either dataset is listed;
   one that is not on display is loaded for the run). **Structures of** =
   `CCT RTSTRUCT`; the set drawn on the chosen image is preselected. **To**
   = the 4D group (the entry `… (10 phases)`).
4. **Tick the target** in the list. Do not tick `heart_total`: the anchored
   run carries it on its own, as the check.
5. **Land as** = *structure set* if the target should sit in each phase's
   own RTSTRUCT next to that phase's heart (what a planning system reads),
   or *segmentation series* for editable masks bound to each phase.
6. **Then**: for a target that is a cloud of points (an ablation map
   exported voxel by voxel) tick *close gaps* with 2 mm to join it into one
   surface, and *fill* to make that surface a solid. A solid target keeps
   its volume through the propagation and needs neither.
7. **Anchor on a structure**: Anchor = `heart_total`, Margin 10 mm, *Refine
   deformably* on, *Match the contours* on. The anchor's own copy lands
   next to the target under the name in **Lands as** (`heart_total_prop`
   by default, so it never collides with the phase's own `heart_total`);
   change it if the planning system expects something else. Leave the
   registration module's
   method and parameters at their defaults (3 resolutions, 300 iterations,
   3000 samples, 32 mm grid); the anchored run uses elastix rigid for the
   first stage and the module's deformable method for the refinement.
8. Press **▶ Anchor and propagate to 10 phases**. Progress shows the phase
   and stage; a run over ten phases at 2 mm takes a few minutes.
9. **Read the Last run block.** Per phase: the registration line
   (`contours rigid MSD … · B-spline … · heart_total Dice 0.93`), then for
   the target its three volumes (source, mapped, filed) and where it landed,
   then the anchor check: Dice, HD95 and centroid distance of the propagated
   heart against the phase's own contour, with a verdict. *good* is Dice
   0.85 or better; *check* (0.70 to 0.85) means look at that phase; *poor*
   or *failed* means the phase's heart contour or the CCT's is not what you
   think it is.
10. **Look.** Select a phase in the tree; its landed target draws in the
    views (as contours in the phase's set, or as the segmentation series
    bound to it). The registration module holds the per-phase transforms:
    switch *Fusion overlay on* to *moving* to see a phase warped back onto
    the CCT, which is the sharpest way to judge the heart alignment.
11. **Export.** *File ▶ Export DICOM* with the 4DCT dataset; structure sets
    as RTSTRUCT, identifiers kept, so the target arrives in each phase's set
    under that phase's series.

Carrying another structure onto the same phases afterwards costs no
registration: the transforms are kept, and the button reads
*▶ Propagate to 10 phases*.

### Then: the motion of the landed target

With the target in every phase's structure set, *Tools ▶ 4D motion / ITV*
on the 4DCT dataset lists it once, in the *On every phase* column of the
Targets list. Tick it there (one tick, not one per phase), keep the
reference structure at `heart_total`, and choose the models:

* **as contoured** reads the landed target from each phase's own set: the
  track is exactly what the anchored propagation put there, and the one to
  quote.
* **rigid** with a *local margin* of 15 mm fits a rigid body to the phase
  around the target and its surroundings. A margin of 0 would fit the
  whole image, which a breathing patient is not; that fit finds the spine
  and the couch and reports 0 mm of motion.
* **deformable** is the whole-image B-spline, for comparison.

*Build ITV* with *Phases: All* makes one ITV per model, the union of the
target over the phases. The results window then shows one line on the
chart and one peak-to-peak amplitude per target and model.

A structure that should sit at the same place on every phase (a fixed
margin, a couch structure, the ITV itself) is copied there with no
registration at all: right-click it in the tree, *Copy to ▶ each phase of
<group>*, as a segmentation series per phase or into each phase's own
structure set. A segmentation series can also be tied to no image series
(*Connect to image series ▶ no image series*): it then shows on every
image of its frame of reference, which is how an ITV drawn on the
reference phase is seen on all of them.

## From the MCP server

One call does steps 3 to 9:

```
propagate_to_group {
  dataset: "ds1",              // the dataset holding the 4D group
  group: "1",
  source_dataset: "ds1",       // where the CCT is (omit when the same)
  source_series: 1,            // the CCT series number from describe_dataset
  anchor: { structure: "heart_total", set: "CCT RTSTRUCT" },
  anchor_margin_mm: 10,
  anchor_by: "contours",
  structures: [ { structure: "target" } ],
  land: "structure_set",
  close_mm: 2, fill: true      // only for a target that is a cloud of points
}
```

The answer lists, per phase, `registration`, the `anchor_check` (Dice,
HD95, mean surface distance, centroid shift, displacement p95, folded
fraction, verdict), the target's `source_cm3` / `mapped_cm3` / `result_cm3`
and the set it landed in, plus `worst_anchor_dice` over the group. Then
`export` on the dataset with `format: "rtstruct"`. Use the `_async` twin and
`list_jobs` for a run this long.

## How the anchored run works

Notation: $F$ is a phase (the fixed image, lattice $\Omega_F$), $M$ the
cardiac CT (the moving image), $A_F$ and $A_M$ the anchor masks on each
(`heart_total`), $S_M$ the mask of a structure to carry (the target). A
transform $T$ always maps fixed patient coordinates to moving ones,
$T: \Omega_F \to \Omega_M$; that is the direction propagation pulls along
(below).

**Step 0, the region.** The phase's anchor mask dilated by the margin $m$:
$R = A_F \oplus B_m$, a separable box dilation of radius $m$ on each axis.
Every sample the registration draws comes from $R$, and the B-spline lattice
of step 3 covers the bounding box of $R$ plus one cell; outside $R$ the
deformation is zero by construction.

**Step 1, initialisation.** The two centroids
$c_F = \frac{1}{|A_F|}\sum_{x \in A_F} x$ and
$c_M = \frac{1}{|A_M|}\sum_{x \in A_M} x$ give the starting transform
$T_0(x) = x + (c_M - c_F)$. This is what makes two images in different
frames of reference registrable at all: the search steps of the optimiser are
a few millimetres, and the images sit hundreds of millimetres apart.

**What is compared.** With *Match the contours* (the default) neither
image's intensities enter. Each anchor mask becomes a signed distance map,

$$D(x) = \min_{y \notin A}\|x - y\| - \min_{y \in A}\|x - y\|,$$

negative inside, zero on the surface, positive outside, computed by the
exact Euclidean distance transform (Felzenszwalb and Huttenlocher, separable
lower envelope of parabolas) on each lattice with its own spacing, clamped
to $\pm 40$ mm so a sample far from the surface has no gradient to pull
on, and stored at 0.01 mm resolution. The registration then minimises

$$E(T) = \frac{1}{|P|}\sum_{x \in P}\bigl(D_M(T(x)) - D_F(x)\bigr)^2,
\qquad P \subset R,$$

the mean squared difference of the two distance maps over the samples $P$.
At the optimum the surface of $A_M$, pulled back through $T$, lies on the
surface of $A_F$, and since $D$ is linear near a surface with unit slope, $E$
is the mean squared surface-to-surface distance in mm². Contrast agent,
reconstruction kernel, cardiac phase and modality do not appear in $E$.
With *Match the contours* off, $D_F$ and $D_M$ are replaced by the images
themselves and $E$ is the ordinary mean squared HU difference (elastix's
`AdvancedMeanSquares`); that is the right metric only when the two images are
alike.

**Step 2, rigid.** $T_1(x) = \mathbf{R}(x - c) + c + t$, six parameters
(three Euler angles, three translations) about the region's centre $c$,
started at $T_0$. The engine is a native implementation of elastix's default:
`RandomCoordinate` sampling with new samples every iteration (3000 per
iteration, sub-voxel jittered inside $R$), the analytic gradient
$\partial E/\partial\theta = \frac{2}{|P|}\sum (D_M(T x) - D_F(x))\,
\nabla D_M(Tx)\cdot \partial T/\partial\theta$, and adaptive stochastic
gradient descent (Klein et al. 2009): step $\gamma_k = a/(t_k + A)$ with the
gain $a$ set so the first step moves a typical point by one voxel, the time
$t_k$ advanced by the sign of successive gradient inner products, and a
trust region of two voxels per step. Rotations are scaled by half the region
extent so one unit of every parameter moves a typical point by about one
millimetre. Three resolution levels of a Gaussian pyramid, 300 iterations
each. A step that maps fewer than a quarter of the samples inside the moving
image is undone; a start with fewer than a quarter is refused with a
message, which is what the identity start would trigger here.

**Step 3, deformable refinement.** Started from $T_1$, the correction is a
cubic B-spline free-form deformation on a lattice of spacing $h$ (32 mm by
default) aligned with the phase's axes over the bounding box of $R$:

$$T_2(x) = T_1(x) + \sum_{i}\sum_{j}\sum_{k} \beta^3\!\Bigl(\tfrac{x_1}{h}-i\Bigr)
\beta^3\!\Bigl(\tfrac{x_2}{h}-j\Bigr)\beta^3\!\Bigl(\tfrac{x_3}{h}-k\Bigr)\,
\phi_{ijk},$$

with $\beta^3$ the cubic B-spline basis and $\phi_{ijk}$ the control-point
displacements, the only unknowns. The same sampler, metric and optimiser as
in step 2 recover the $\phi_{ijk}$, with the gradient distributed to the 64
control points around each sample by the same basis weights; the rigid part
is frozen, and the result is the composition of the two. Because the lattice
is confined to $R$, the target inside the heart moves with the heart
surface and nothing outside the region is touched. With plastimatch chosen
as the deformable method in the registration module, the refinement uses
that engine instead: a dense analytic gradient, L-BFGS, and a bending-energy
penalty $\lambda \int \|\nabla^2 u\|^2$ on the lattice.

**Step 4, propagation.** For every phase voxel $v$ with centre $x_v$, a set
of sub-points $x_v + \delta_s$ (up to four per axis, from the spacing ratio)
is pulled back to $T_2(x_v + \delta_s)$ and the source mask is read there by
trilinear interpolation, giving the occupancy
$o_v = \frac{1}{n_s}\sum_s S_M(T_2(x_v + \delta_s)) \in [0, 1]$. The mapped
volume is $V = |v|\sum_v o_v$ (exactly the volume of $T_2^{-1}(S_M)$ up to
quadrature), and the landed mask is the $\lfloor V/|v| \rceil$ voxels of
highest occupancy. That is what keeps a target made of 1 mm cubes from
losing four fifths of itself on 2 mm slices. The anchor $A_M$ is pulled back
the same way and compared with $A_F$: Dice $2|X \cap A_F|/(|X| + |A_F|)$,
the 95th percentile of the pooled surface distances (HD95) and the mean
surface distance, both from exact distance transforms, and the centroid
offset.

**Step 5, finishing.** Optionally each landed mask $X$ is joined and
filled. The textbook closing $(X \oplus B_r) \ominus B_r$ does not join a
cloud: two points closer than $2r$ come back as two points, because the ball
never fits between them. So *close gaps* computes
$(X \oplus B_r) \ominus B_{r/2}$, one surface about $r$ thicker than the
cloud, with $B_r$ the Euclidean ball from the distance transform (a true
millimetre ball on an anisotropic lattice); and *fill* computes
$\mathrm{fill}(X \oplus B_r) \ominus B_r$, the interior filled slice by
slice between the two operations, so the solid's surface returns to where
the cloud was.

So yes: it is elastix-style registration, the same engine, sampler and
optimiser as *Modules ▶ Image registration* with the elastix methods, run
locally on the anchor region; what differs is what it compares (distance
maps of the contours instead of intensities), how it starts (matched
centroids), and that it does so ten times, once per phase.

## Why a 10 mm margin

The metric only sees samples inside $R = A_F \oplus B_m$. With $m = 0$ the
samples stop exactly at the phase's heart surface: the moving map is read at
$T(x)$ for $x$ inside the surface only, so a transform that shrinks the
moving heart *into* the fixed one is never penalised from outside, and the
rigid fit is free to drift along the surface. The margin puts a rim of
samples outside the surface, where $D_F > 0$, so the surface is constrained
from both sides. It also gives the B-spline lattice control points beyond
the surface, without which the deformation could not represent a motion of
the boundary itself.

Ten millimetres is roughly the typical residual after the centroid start:
the heart's own position differs between an ECG-gated breath-hold scan and a
respiratory bin by a few millimetres, the two contours (drawn on different
images by different tools) disagree by a few more, and the margin must
exceed the largest surface discrepancy the fit is expected to close, or
those samples never see the moving surface. Much wider, and the region takes
in lung, sternum and vertebra, whose distance-map values are not about the
heart at all and only add noise (and time) to $E$; on a short cardiac scan
the region would also run off the CCT field of view, where the samples are
lost. Ten millimetres is the width at which the region is all rim and no
bystanders. Use 15 when the contours are known to disagree more (a coarse
auto-contour on the 4DCT), 5 when both are tight.

## Why this rather than an image registration

A global registration of the whole CCT onto a phase answers a different
question, and answers it badly for this pair:

* **Nothing overlaps at the start.** The two frames of reference are
  hundreds of millimetres apart; every search-based engine needs to be put
  within a few voxels of the answer first, and a centre-of-gravity match of
  the whole images aligns the CCT's small field of view with the middle of
  the 4DCT torso, not the heart with the heart.
* **The intensities do not correspond.** The CCT is contrast-enhanced (blood
  pool 300 to 500 HU), the 4DCT is not (40 HU). A mean-squares metric is
  minimised by moving the bright chambers *away* from the 4DCT heart, which
  is exactly what the intensity-based rigid stage did on the 30 % phase
  before the contour mode existed: it pushed the region out of the CCT
  field of view. Mutual information copes with contrast differences but is
  weakest on a small, smooth region, which is what a heart is.
* **The anatomy does not correspond either.** One image is a single cardiac
  phase at breath-hold, the other a respiratory bin with the cardiac cycle
  averaged into it. Matching lung texture, ribs or diaphragm across that is
  matching motion that has nothing to do with where the heart is.
* **What the target needs is only the heart.** The target sits in the
  myocardium; its position relative to the heart surface is what has to be
  preserved. A transform fitted to the heart contours moves the target with
  the heart, with a smooth interpolation inside; a transform fitted to the
  whole thorax moves it with the average of everything.
* **The check is built in.** The anchor lands with the target and is
  compared with the contour that was drawn on the phase independently. A
  global registration reports a metric value; this reports a Dice per phase
  against a ground truth that exists anyway.

The price is that the result is only as good as the two heart contours. That
is the right dependence for a target defined relative to the heart, and it
is visible: a phase with a poor contour shows up as a poor Dice.

## What to expect of the numbers

* **Heart Dice** between 0.90 and 0.96 per phase with the contour match; the
  centroid distance under 2 mm. Lower on one phase alone points at that
  phase's contour.
* **Target volume.** The three volumes should agree to within a voxel of the
  phase lattice: the mapped volume is what the deformation made of the
  source, and the filed mask holds exactly that. A cardiac target exported
  as a cloud of 1 mm cubes (an ablation map, voxel by voxel) keeps its
  volume but becomes blockier on 1.2 × 1.2 × 2 mm voxels; if a solid region
  is wanted for planning, close it with the structure algebra (a margin and
  its negative) before propagating.
* **Folded fraction** of the refinement should be 0; a displacement p95 of a
  few millimetres inside the heart region is normal between a breath-hold
  cardiac CT and a respiratory bin.

## When to change the defaults

* *Match the contours* off (intensity matching) only for two images that
  are alike: same contrast, same kernel. Between a contrast CCT and a plain
  4DCT the mean-squares metric pushes the bright blood pool out of
  correspondence.
* *Refine deformably* off keeps the alignment rigid: use it when the heart
  contours on the 4DCT are coarse and you would rather not deform the target
  to follow them.
* A smaller B-spline grid (16 to 24 mm in the registration module's
  parameters) lets the refinement follow the ventricles more closely, at
  some cost in smoothness; 32 mm is the safe first pass.
* Margin: 10 mm bounds the registration to the heart and a rim around it.
  Widen it when the CCT field of view cuts the heart (the inferior wall on a
  short scan), so the fit still sees the surface on both sides.
