# Image registration

Intensity-based and landmark-based registration between the two loaded
datasets, with three independent engines, per-run analytics, a deformation
vector field you can see and export, and the option to restrict any of it to
a single structure. elastix and plastimatch are C++/ITK toolboxes; nothing of
either is linked here — the algorithms are **re-implemented natively in
Rust** to keep the application single-language.

![registration](screenshot_registration.png)

*Deformable registration of two breathing phases: the sidebar reports the
recovered transform and metric improvement; the fusion overlay shows aligned
anatomy in gray, residual respiratory mismatch as magenta/green fringes.*

## The three engines

| | elastix rigid | elastix B-spline | plastimatch B-spline | plastimatch landmarks |
|---|---|---|---|---|
| transform | Euler 6-DOF | rigid + cubic FFD | centre-of-gravity + cubic FFD | RBF warp |
| samples | ~3000 random, redrawn every iteration | same | every eligible voxel | none (geometric) |
| gradient | stochastic estimate | stochastic estimate | exact analytic | closed form |
| optimizer | ASGD | ASGD | L-BFGS + line search | direct solve |
| metric | mean squares | mean squares | mean squares **or** Mattes MI | landmark residual |
| regularizer | — | — | bending energy | stiffness |
| deterministic | seeded | seeded | yes | yes |
| multi-modal | no | no | yes (MI) | yes |

### elastix — stochastic sampling and ASGD

A native re-implementation of what an elastix parameter file with
`Optimizer AdaptiveStochasticGradientDescent`, `ImageSampler
RandomCoordinate`, `NewSamplesEveryIteration true` and `Metric
AdvancedMeanSquares` asks for — the toolbox's own defaults.

* **Multi-resolution Gaussian pyramids** (`NumberOfResolutions`, default 3):
  repeated [1 2 1]/4 smoothing + factor-2 decimation with correct
  voxel-centre origin bookkeeping.
* **Random coordinate sampling** with fresh samples every iteration
  (`NumberOfSpatialSamples`, default 3000), restricted to a body mask from a
  configurable HU threshold (default −500) and, for a local run, to the
  region. Sampling draws from a pre-built eligible-voxel list rather than
  rejecting random draws, so every draw is a hit.
* **Metric:** mean squared difference with analytic gradients.
* **Optimizer:** Adaptive Stochastic Gradient Descent (Klein et al., IJCV
  2009 — elastix's default) with automatic gain estimation, the sigmoid
  time-adaptation rule and a trust-region step cap.
* **Rigid:** 6-DOF Euler transform about the fixed-image centre, with
  automatic rotation/translation parameter scaling.
* **Deformable:** the recovered rigid transform composed with a cubic
  B-spline free-form deformation (`FinalGridSpacingInPhysicalUnits`, default
  32 mm), optimized coarse-to-fine across the pyramid.

The defining property is that an iteration costs almost nothing, so
thousands are affordable, and the noise in the estimate is what carries the
search past small local minima. It is the engine to reach for first.

### plastimatch — a dense exact gradient and L-BFGS

Following plastimatch's `bspline` (Shackleford et al., *High performance
deformable image registration algorithms for manycore processors*), and the
opposite trade: far more work per iteration, far fewer of them.

1. `xform=align_center` — a translation matching the centres of gravity of
   the two thresholded images, which removes the gross offset a deformable
   model should never have to represent. Skipped for a local run or a
   refinement, both of which already start from an alignment.
2. `xform=bspline` per resolution level, coarse to fine. The cost and its
   **exact analytic gradient** are evaluated over every eligible fixed
   voxel and each voxel's contribution is scattered onto the 64 control
   points that support it.

* **Metric.** `mse` is the mean squared difference, divided by the fixed
  image's variance so the cost is dimensionless. `mi` is **Mattes mutual
  information** over a 32 × 32 joint histogram with a zero-order Parzen
  window on the fixed image and a cubic B-spline window on the moving one
  (Mattes et al., IEEE TMI 2003) — the only metric here that survives two
  modalities, and therefore the CT–MR and CT–CBCT option. Both metrics share
  one gradient scatter loop: only the per-sample scalar differs.
* **Regularizer.** `young_modulus` weights the discrete bending energy of
  the control lattice (second differences, mixed terms counted twice), made
  dimensionless by the lattice spacing so the same weight means the same
  smoothing at any grid size. It is quadratic in the coefficients, so its
  gradient is exact — a unit test checks it against a central difference.
* **Optimizer.** L-BFGS (two-loop recursion, history 6) with an Armijo
  backtracking line search. plastimatch's default is L-BFGS-B; the bounded
  variant differs only in handling box constraints, and B-spline
  coefficients have none.

"Dense" is capped at 400 000 samples per level — on a 512³ study every
eligible voxel is tens of millions, and an exact gradient over all of them is
not what anybody wants to wait for. The thinning is deterministic, so the
sample set is the same on every iteration, which is what keeps the engine's
character.

### plastimatch — landmark warp

`landmark_warp`: a deformation that interpolates paired points, with no image
intensity read at any stage. That is what is wanted when the two images have
nothing an intensity metric can lock onto (CT against MR, a post-operative
cavity, anatomy that genuinely changed) or when the alignment must honour
specific anatomical points and nothing else.

| kernel | φ(r) | support | affine term |
|---|---|---|---|
| Thin-plate spline | `r` | global | yes |
| Gaussian | `exp(−r² / 2R²)` | global, decaying | no |
| Wendland ψ₃,₁ | `(1 − r/R)⁴ (4r/R + 1)` | compact, zero beyond `R` | no |

The thin-plate spline minimizes bending energy over the whole domain and
carries an affine term, so a global shift or rotation implied by the
landmarks is reproduced exactly — it needs at least four non-coplanar pairs.
The two radial kernels have no affine term, so the displacement decays back
to zero away from the landmarks; the compactly supported Wendland kernel is
the one that *provably* leaves distant anatomy untouched. `stiffness`
(plastimatch's regularization) is added to the diagonal of the interpolation
matrix: zero passes exactly through every landmark, larger values smooth the
field and tolerate inconsistent pairs.

Landmarks are placed from the interface: put the crosshair on the same
anatomy in both datasets and press **➕ Add pair** in the *Landmarks* section
(turn *View ▶ Sync crosshairs* off first, or both crosshairs move together).
Each pair shows its displacement, and after a run, its residual.

## Running a registration

*Modules ▶ Image registration* puts the section in the left panel. With
two datasets loaded it registers one onto the other — the direction is selectable (**B ▶ A** or
**A ▶ B**; the second-named dataset is the fixed image and receives the
fusion overlay). Everything lives in that one section: method, region,
parameters, landmarks, the result and the vector field. Runs happen on a
background thread with progress and a **Cancel** button.

The transform convention is **fixed → moving** patient coordinates, as in
elastix, ITK and plastimatch alike; the inverse (needed for the crosshair
link and for propagation) is exact for the rigid part and a fixed-point
iteration for the deformable one.

On the bundled data (512 × 512 × 133 CT, two breathing phases): elastix
rigid pre-alignment plus three B-spline resolution levels, 1800 iterations
total, ≈ 20 s on a desktop CPU, driving the mean-squared HU difference from
≈ 9700 to ≈ 1800.

## Local registration

Any method can be restricted to a **region** — an RTSTRUCT ROI or a painted
segmentation of the fixed dataset, dilated by a margin. That is what
"register this tumour, not the whole patient" means, and it changes three
things:

* samples come from inside the region only;
* the B-spline control lattice covers the region's bounding box rather than
  the volume, so a small structure can be aligned at a fine grid spacing
  that would be unaffordable globally;
* the centre of rotation and the parameter scaling are the region's, not the
  patient's — rotating a tumour about the patient's centre would put the
  whole recovered angle into the translation.

A **local deformable** run skips the rigid stage on purpose: a rigid body
fitted to one structure would be applied to the whole volume and move
anatomy nobody asked about. Confined to its lattice, the correction is
exactly zero outside the region — the integration test asserts this to
machine precision. A **local rigid** run is different by nature: it reports
how that structure moved *as a rigid body*, and the transform is global.

The margin matters. Without it nothing outside the structure constrains its
boundary, and the boundary is what you are aligning.

### Refining

**▶ Refine** recovers a correction *on top of* the active registration
instead of replacing it: the moving image is sampled through the existing
transform plus the new deformation, and the result is the two composed. The
intended workflow is a global registration first, then a local refinement on
the structure that matters — after which the rest of the patient still
carries the global result, unchanged.

## What the result says

The result block reports the method, the region if any, the metric before
and after, and the deformation model. The **Analysis** section is measured
on the transform itself, on a lattice over the fixed image (or the region),
so it means the same thing for every method:

* **Best-fitting rigid body** — the orthogonal Procrustes fit of the
  mapping: translation, three Euler angles in the same `Rz Ry Rx` convention
  as the rigid transform, and the RMS residual, which says how much of the
  transform those six numbers do *not* explain (zero for a rigid result, by
  construction). Usually the first number a physicist wants: how far did the
  patient move, and how far did they turn?
* **Displacements** — min / mean / p95 / max / RMS of `|T(p) − p|` in
  millimetres, plus the mean *vector*, which separates a systematic shift
  from scattered local motion.
* **Jacobian determinant** — `det(I + ∂d/∂x)` by central differences: above
  1 the tissue expanded, below 1 it compressed, and at or below zero the
  deformation folded onto itself, which is not anatomy but an artefact. The
  folded fraction is reported explicitly; a regularized B-spline should show
  none.
* **Per structure** — the mean and maximum displacement over each
  contoured structure's own points, which is what turns "the registration
  moved things by 4 mm on average" into "the tumour moved 9 mm and the cord
  0.4 mm".

## The fusion overlay and the vector field

**Fusion** blends the transformed moving image into the green channel of the
fixed image, so aligned anatomy reads gray and mismatch reads magenta/green,
with a blend slider. The cross-study crosshair link maps through the
recovered transform (inverse included), so clicking a point in either study
lands on the same anatomy in the other.

The **vector field** is the transform sampled onto a regular lattice — once,
rather than per pixel on every repaint, because a B-spline evaluation is 64
weighted lookups and a landmark warp is a sum over every landmark. It is
drawn in all three MPR views of the fixed dataset and, optionally, in the 3D
window:

* **Arrows** from where the anatomy is to where it goes, at an adjustable
  exaggeration (millimetre motion is invisible at 1×) and coloured by
  magnitude. Displacement that leaves the view plane cannot be drawn as an
  arrow, so it is drawn as a disc whose size is the out-of-plane component.
* **Deformed grid** — the sampling lattice pushed through the deformation,
  the classic warped graph paper: arrows show motion, a deformed grid shows
  compression and expansion.
* Lattice spacing, arrow scale and colouring are adjustable; changing the
  spacing re-samples on a worker thread.

In the **3D window**, *Dataset B through the registration* meshes the other
dataset's structures and maps every vertex through the recovered transform,
so both anatomies stand in one frame of reference with an independent
opacity each — the only way to see what a deformable registration actually
did to a surface. The field can be overlaid as 3-D arrows in the same scene.

## DICOM interchange

A rigid matrix from a DICOM **REG** object can be applied instead of running
the optimizer, and a **Deformable Spatial Registration** object's
displacement grid can be applied the same way — it becomes the active
registration and everything downstream (fusion, crosshair link, analytics,
propagation) works on it without knowing where it came from. See
[rt-objects.md](rt-objects.md).

**💾 Save as DICOM…** writes the active field back out as a Deformable
Spatial Registration. The IOD applies its grid after a pre-deformation
matrix and before a post-deformation one; both are written as the identity
and the grid carries the whole mapping, `T(p) − p`, so another system has no
composition rule to get wrong.

## Propagating structures

Once the datasets are aligned, the contours drawn on one of them can be
carried to the other — see [propagation.md](propagation.md).

## Transform simulator (registration QA)

The *Simulation* sidebar section applies an **exactly known** transform to a
loaded dataset — rigid motion (translation + Euler rotation about the volume
centre) plus an optional local Gaussian deformation (amplitude vector + σ,
centred at the crosshair) — and generates the result into the other dataset
slot: the CT is resampled through the inverse transform, and structure
contours, dose grids and plan isocentres are carried along. The applied
parameters stay displayed as ground truth, so you can immediately run any
engine and compare the recovered transform against it. Any dataset —
original or simulated — can then be exported as DICOM (see
[export-and-tools.md](export-and-tools.md)).

## Accuracy verification

`tests/registration.rs` registers analytically known transforms on a
synthetic phantom, with the same tolerances applied to every engine:

* **elastix rigid** recovers a known rotation + translation to ≈ 0.6 mm
  (asserted 1.5 mm), the inverse round-trips to 10⁻⁶ mm, and the six-DOF
  analysis reproduces the recovered transform to 10⁻³ degrees with zero
  residual and unit Jacobian.
* **elastix B-spline** and **plastimatch B-spline** each recover a 7 mm
  Gaussian-bump deformation to ≈ 0.3 mm (asserted 3 mm), with no folding.
* **plastimatch mutual information** recovers the same bump between images
  whose soft-tissue contrast has been *inverted* — a case where mean squares
  has no minimum at the truth at all.
* **landmark warp** lands on every landmark to 10⁻⁴ mm with all three
  kernels, the thin-plate spline reproduces a global shift everywhere
  including far outside the landmark hull, and the Wendland kernel leaves
  points beyond its radius at exactly zero.
* **local registration** recovers a displacement applied inside one blob and
  leaves every probe outside the region at exactly zero displacement; a
  refinement on top of a global result changes nothing outside its region.
* **the vector field** reproduces the transform it was sampled from to
  < 0.05 mm.

Unit tests cover the pieces the integration tests can only see through: the
Parzen window and its derivative against finite differences, the bending
energy's gradient against a central difference and its vanishing on an
affine field, the Procrustes fit against a reflection, region dilation by an
exact margin, and the local lattice's coverage of its region.

## Notes

Deformable results are intensity-driven: displacements inside large uniform
regions are interpolated from the control lattice rather than measured — the
Jacobian and the per-structure displacements are how you tell the difference.
Mean squares assumes comparable intensities (CT–CT); for CT–MR use the
plastimatch engine with mutual information, or place landmarks.
