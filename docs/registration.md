# Image registration

Intensity- and landmark-based registration between the two loaded datasets:
three independent engines, per-run analytics, a deformation vector field
you can see and export, and the option to restrict any of it to a single
structure. elastix and plastimatch are C++/ITK toolboxes; nothing of either
is linked - the algorithms are **re-implemented natively in Rust**.

![registration](screenshot_registration.png)

*Deformable registration of two breathing phases: the module reports the
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
| regularizer | - | - | bending energy | stiffness |
| deterministic | seeded | seeded | yes | yes |
| multi-modal | no | no | yes (MI) | yes |

### elastix - stochastic sampling and ASGD

A native re-implementation of elastix's own defaults - `Optimizer
AdaptiveStochasticGradientDescent`, `ImageSampler RandomCoordinate`,
`NewSamplesEveryIteration true` and `Metric AdvancedMeanSquares` - and the
engine to reach for first: iterations are cheap, so thousands are
affordable, and the estimate's noise carries the search past small local
minima.

* **Multi-resolution Gaussian pyramids** (`NumberOfResolutions`, default 3):
  [1 2 1]/4 smoothing + factor-2 decimation, voxel-centre origin bookkeeping.
* **Random coordinate sampling**, fresh every iteration
  (`NumberOfSpatialSamples`, default 3000), within a body mask from a
  configurable HU threshold (default −500) and, for a local run, the region;
  drawn from a pre-built eligible-voxel list, so every draw is a hit.
* **Metric:** mean squared difference with analytic gradients.
* **Optimizer:** Adaptive Stochastic Gradient Descent (Klein et al., IJCV
  2009 - elastix's default) with automatic gain estimation, the sigmoid
  time-adaptation rule and a trust-region step cap.
* **Rigid:** 6-DOF Euler transform about the fixed-image centre, with
  automatic rotation/translation parameter scaling.
* **Deformable:** the rigid result composed with a cubic B-spline free-form
  deformation (`FinalGridSpacingInPhysicalUnits`, default 32 mm), optimized
  coarse-to-fine across the pyramid.

### plastimatch - a dense exact gradient and L-BFGS

Following plastimatch's `bspline` (Shackleford et al., *High performance
deformable image registration algorithms for manycore processors*) - the
opposite trade: far more work per iteration, far fewer of them.

1. `xform=align_center` - a translation matching the centres of gravity of
   the two thresholded images. Skipped for a local run or a refinement,
   which already start aligned.
2. `xform=bspline` per resolution level, coarse to fine: the cost and its
   **exact analytic gradient** over every eligible fixed voxel, each voxel's
   contribution scattered onto the 64 control points that support it.

* **Metric.** `mse` is the mean squared difference divided by the fixed
  image's variance, so the cost is dimensionless. `mi` is **Mattes mutual
  information** over a 32 × 32 joint histogram, zero-order Parzen window on
  the fixed image and cubic B-spline window on the moving one (Mattes et
  al., IEEE TMI 2003) - the multi-modal (CT-MR, CT-CBCT) option.
* **Regularizer.** `young_modulus` weights the discrete bending energy of
  the control lattice (second differences, mixed terms counted twice), made
  dimensionless by the lattice spacing; its gradient is exact.
* **Optimizer.** L-BFGS (two-loop recursion, history 6) with an Armijo
  backtracking line search; plastimatch's default L-BFGS-B differs only in
  box constraints, which B-spline coefficients lack.

"Dense" is capped at 400 000 samples per level, thinned deterministically
so the set is the same on every iteration - on a 512³ study every eligible
voxel is tens of millions.

### plastimatch - landmark warp

`landmark_warp` interpolates paired points and reads no image intensity -
for CT against MR, a post-operative cavity, anatomy that genuinely changed,
or when specific anatomical points must be honoured and nothing else.

| kernel | φ(r) | support | affine term |
|---|---|---|---|
| Thin-plate spline | `r` | global | yes |
| Gaussian | `exp(−r² / 2R²)` | global, decaying | no |
| Wendland ψ₃,₁ | `(1 − r/R)⁴ (4r/R + 1)` | compact, zero beyond `R` | no |

The thin-plate spline minimizes bending energy over the whole domain; its
affine term reproduces exactly any global shift or rotation implied by the
landmarks, and it needs at least four non-coplanar pairs. The radial kernels
have no affine term, so displacement decays to zero away from the landmarks;
the compactly supported Wendland kernel *provably* leaves distant anatomy
untouched. `stiffness` (plastimatch's regularization) is added to the
diagonal of the interpolation matrix: zero passes exactly through every
landmark, larger values smooth the field and tolerate inconsistent pairs.

Put the crosshair on the same anatomy in both datasets and press
**➕ Add pair** in the *Landmarks* section (with *View ▶ Sync crosshairs*
off, or both crosshairs move together). Each pair shows its displacement
and, after a run, its residual.

## Running a registration

*Modules ▶ Image registration* puts the section - the two images, method,
region, parameters, landmarks, result and vector field - in the right panel.
**Fixed image** and **Moving image** each name one image series of either
dataset: the displayed ones, a phase of a 4DCT, or two series of one dataset
(a cardiac CT and a phase of the 4DCT it arrived with). A series that is not
on display is loaded for the run. The fixed image may also be *every phase
of a 4D group*, which is one registration per phase (below). The run goes
on a background thread with progress and a **Cancel** button.

The result names both images, and **Fusion overlay on** chooses which one
carries the overlay: on the fixed image the moving one is warped onto it, on
the moving image the fixed one comes back through the inverse. The overlay,
the vector field and the crosshair link appear in whichever dataset displays
that image, so two series of one dataset show the fusion once the same
folder is loaded as the other dataset too; propagation works either way.

The transform maps **fixed → moving** patient coordinates, as in elastix,
ITK and plastimatch; the inverse (for the crosshair link and propagation) is
exact for the rigid part and a fixed-point iteration for the deformable one.

**Start from** says where the search begins. The engines take steps of a
few millimetres, so two images that do not overlap at the identity never
find each other: a cardiac CT and a 4DCT of one patient are two
acquisitions in two frames of reference, hundreds of millimetres apart in
patient coordinates, and a run started from the identity has no gradient to
follow (it now says so rather than returning the identity as a result).
*Automatic* keeps the identity when the images overlap - every same-frame
pair, so nothing changes for those - and matches the centres of gravity when
they do not (elastix's `AutomaticTransformInitialization`, what the
plastimatch engine always did as `align_center`). *Centroids of a
structure* matches one structure contoured on both datasets, which is the
surest start for an organ: the heart on a cardiac CT and on a planning CT.
A local run always starts from the identity.

On the bundled data (512 × 512 × 133 CT, two breathing phases): elastix
rigid pre-alignment plus three B-spline resolution levels, 1800 iterations
total, ≈ 20 s on a desktop CPU, driving the mean-squared HU difference from
≈ 9700 to ≈ 1800.

## Against a 4D group

The **Fixed image** list ends with **every phase of a 4D group** of either
dataset. That runs one registration per phase against the moving image: the
phases of one acquisition differ by breathing, so a single transform for the
group would be answering a question nobody asked.

The moving image can be any series - of the group's own dataset (a planning
CT or a cardiac CT beside its 4DCT) or of the other one. Each phase
reports its own metric line, and the transforms are kept so that propagating
structures onto the same group afterwards costs no registration
([propagation.md](propagation.md)). **Clear group registration** drops them,
as does clearing the registration.

## Local registration

Any method can be restricted to a **region** - an RTSTRUCT ROI or a painted
segmentation of the fixed dataset, dilated by a margin. Three things change:

* samples come from inside the region only;
* the B-spline control lattice covers the region's bounding box, so a small
  structure can be aligned at a grid spacing unaffordable globally;
* the centre of rotation and parameter scaling are the region's, not the
  patient's.

A **local deformable** run skips the rigid stage on purpose: a rigid body
fitted to one structure would move the whole volume. Confined to its
lattice, the correction is exactly zero outside the region. A **local
rigid** run instead reports how that structure moved *as a rigid body*; the
transform is global. Without a margin nothing outside the structure
constrains the boundary being aligned.

### Refining

**▶ Refine** recovers a correction *on top of* the active registration: the
moving image is sampled through the existing transform plus the new
deformation, and the result is the two composed - typically a global
registration, then a local refinement on the structure that matters, leaving
the rest of the patient on the global result.

## What the result says

The result block reports method, region if any, metric before and after, and
deformation model. The **Analysis** section is measured on the transform
itself, on a lattice over the fixed image (or the region), so it means the
same for every method:

* **Best-fitting rigid body** - the orthogonal Procrustes fit: translation,
  three Euler angles in the same `Rz Ry Rx` convention as the rigid
  transform, and the RMS residual those six numbers do *not* explain.
* **Displacements** - min / mean / p95 / max / RMS of `|T(p) − p|` in
  millimetres, plus the mean *vector* (systematic shift vs. scattered local
  motion).
* **Jacobian determinant** - `det(I + ∂d/∂x)` by central differences: above
  1 the tissue expanded, below 1 it compressed, at or below zero it folded.
  The folded fraction is reported; a regularized B-spline should show none.
* **Per structure** - mean and maximum displacement over each contoured
  structure's own points: "the tumour moved 9 mm and the cord 0.4 mm"
  rather than "4 mm on average".

## The fusion overlay and the vector field

**Fusion** blends the transformed moving image into the fixed image's green
channel (aligned anatomy gray, mismatch magenta/green) with a blend slider;
the cross-study crosshair link maps through the recovered transform, inverse
included.

The **vector field** is the transform sampled onto a regular lattice - once,
not per pixel on every repaint: a B-spline evaluation is 64 weighted
lookups, a landmark warp a sum over every landmark. It is drawn in all three
MPR views of the fixed dataset and, optionally, in the 3D window:

* **Arrows** from where anatomy is to where it goes, exaggerated by an
  adjustable factor (millimetre motion is invisible at 1×) and coloured by
  magnitude; out-of-plane displacement becomes a disc sized by that component.
* **Deformed grid** - the sampling lattice pushed through the deformation:
  warped graph paper, showing compression and expansion.
* Lattice spacing, arrow scale and colouring are adjustable; changing the
  spacing re-samples on a worker thread.

In the **3D window**, *Dataset B through the registration* meshes the other
dataset's structures and maps every vertex through the recovered transform,
so both anatomies stand in one frame of reference with independent
opacities. The field can be overlaid as 3-D arrows in the same scene.

## DICOM interchange

A rigid matrix from a DICOM **REG** object or a **Deformable Spatial
Registration** object's displacement grid can be applied instead of running
the optimizer; it becomes the active registration and everything downstream
(fusion, crosshair link, analytics, propagation) works on it. See
[rt-objects.md](rt-objects.md).

**💾 Save as DICOM…** writes the active field out as a Deformable Spatial
Registration. The IOD applies its grid between a pre- and a post-deformation
matrix; both are written as the identity and the grid carries the whole
mapping, `T(p) − p`.

## Propagating structures

Once aligned, contours drawn on one dataset can be carried to the other -
see [propagation.md](propagation.md).

## Transform simulator (registration QA)

The *Simulation* module section applies an **exactly known** transform -
rigid motion (translation + Euler rotation about the volume centre) plus an
optional local Gaussian deformation (amplitude vector + σ, centred at the
crosshair) - to a loaded dataset and generates the result into the other
slot: the CT is resampled through the inverse transform; structure contours,
dose grids and plan isocentres are carried along. The applied parameters
stay displayed as ground truth. Any dataset, original or simulated, can then
be exported as DICOM (see [export-and-tools.md](export-and-tools.md)).

## Accuracy verification

`tests/registration.rs` registers analytically known transforms on a
synthetic phantom, with the same tolerances for every engine:

* **elastix rigid** recovers a known rotation + translation to ≈ 0.6 mm
  (asserted 1.5 mm); the inverse round-trips to 10⁻⁶ mm; the six-DOF
  analysis reproduces it to 10⁻³ degrees, zero residual, unit Jacobian.
* **elastix B-spline** and **plastimatch B-spline** each recover a 7 mm
  Gaussian-bump deformation to ≈ 0.3 mm (asserted 3 mm), with no folding.
* **plastimatch mutual information** recovers the same bump between images
  with *inverted* soft-tissue contrast, where mean squares has no minimum
  at the truth at all.
* **landmark warp** lands on every landmark to 10⁻⁴ mm with all three
  kernels; the thin-plate spline reproduces a global shift everywhere,
  including far outside the landmark hull, and the Wendland kernel leaves
  points beyond its radius at exactly zero.
* **local registration** recovers a displacement applied inside one blob and
  leaves every probe outside the region at exactly zero displacement; a
  refinement on top of a global result changes nothing outside its region.
* **the vector field** reproduces the transform it was sampled from to
  < 0.05 mm.

Unit tests check the Parzen window and its derivative against finite
differences, the bending energy's gradient against a central difference and
its vanishing on an affine field, the Procrustes fit against a reflection,
region dilation by an exact margin, and the local lattice's coverage of its
region.

## Notes

Deformable results are intensity-driven: displacements inside large uniform
regions are interpolated from the control lattice rather than measured - the
Jacobian and per-structure displacements tell the difference. Mean squares
assumes comparable intensities (CT-CT); for CT-MR use the plastimatch engine
with mutual information, or place landmarks.
