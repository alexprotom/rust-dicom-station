# Making a structure without drawing it

*Tools ▶ ✚ New structure* opens the *New structure* section of the Structure
tools module (right panel) - a grey-level window, a shape, a dose level or
the field of view.
These are the cheap generators every planning system has, and the arithmetic
behind them is a few lines each ([`src/generate.rs`](architecture.md#module-map)).
What matters is where the result lands: an ordinary RT structure, editable
with the contour tools from the moment it appears ([contours.md](contours.md)),
not a read-only overlay.

## Grey level

Everything between two Hounsfield values, optionally **only inside** a
structure that is already drawn. The limiting structure is what makes a
threshold usable: bone inside the body rather than bone and the couch rails,
contrast inside the liver rather than every vessel in the abdomen.

Three bone presets sit next to the fields: **high** for head and neck (above
250 HU), **medium** for thorax (200 HU), **low** for pelvis (150 HU). They
set the window; the rest of the dialog is unchanged, so a preset is a
starting point and not a different tool.

On a PET series that carries what an SUV takes - the pixels in Bq/mL, the
patient weight, the injected activity, the half-life and both times - a
**SUV (body weight)** tick reads the two bounds as standardized uptake
values instead of stored counts, and 2.5 is the usual starting point for a
lesion. The factor is this series' own: `weight / (A₀ · 2^(−Δt / T½))`,
where Δt is the delay from the injection to the acquisition. A series that
does not carry all of it is not offered the tick, because an SUV computed
from a guessed weight is worse than no SUV at all.

Two tidying options apply to the result: keep only the largest connected
piece, and fill interior cavities. A bone ROI wants neither by default (both
femurs are two pieces, and marrow is not bone), which is why they are ticks
rather than defaults.

## Shape

A box, cylinder, sphere or ellipsoid, aligned with the patient axes, centred
by default on the crosshair. The cylinder's axis is superior-inferior.

These exist for evaluation: a sphere of known volume to check a DVH against,
a box to crop with, a phantom target. The shape is rasterized onto the
displayed lattice and traced back to contours, so it is accurate to half a
voxel per face and behaves like any other structure afterwards.

## Field of view

A CT reconstructed on a circle smaller than the image matrix pads the
corners with one constant value, and every threshold, body contour and
registration then has to know where the data stop. This structure is that
circle: the padding value is read off the corners rather than assumed (-1000
and -2000 are both in use, and so is 0), the valid pixels of each slice are
kept, holes are filled, and only the piece the centre of the image sits in
survives - a couch rail sticking out of the reconstruction circle is not
part of the field of view.

It earns its keep as a *limiting* structure: a threshold or a body contour
restricted to it stops at the edge of the data instead of at the edge of the
matrix. Where the patient is cut off by it, intersect it with the EXTERNAL
in [the structure algebra](structure-algebra.md) and the cut surface is
exactly what comes out.

Images whose corners hold real data are not offered it: there is nothing to
outline, because the field of view is the image, and the window says so
rather than making an empty structure.

## Dose

Everything at or above a dose level, as a per cent of the reference dose or
in absolute Gy. The dose is sampled exactly the way the isodose lines on the
screen are sampled - trilinearly, in patient space, through the same affine
step - so the structure agrees with the line you can see rather than with a
second interpretation of the same grid.

The obvious use is an isodose structure to intersect with an organ, which is
[the structure algebra's](structure-algebra.md) job from there.

## What it does not do yet

* **Couch removal** as a separate tool: the body contour already leaves the
  couch outside.
* A **volume threshold** as its own generator. *Keep only the largest piece*
  and the algebra's *drop anything under n cm³* cover it between them.

## Verification

`src/generate.rs`'s own tests: a threshold takes exactly the voxels in its
window and no others, a limiting mask really limits it, and an upper bound
excludes what it should; the field of view of a padded reconstruction comes
out as the circle it is, to within five per cent of its analytic area, and
does not change when air is put inside it, while a full-field image is
refused; each of the four shapes comes out with the volume it
should (exactly for the box and the cylinder, where the sizes are chosen on
the half-voxel so that "the voxel centre is inside" is exact, and to under
three per cent for the round ones), and the same volume again after being
traced into contours; a shape placed off the lattice comes out empty rather
than panicking.
