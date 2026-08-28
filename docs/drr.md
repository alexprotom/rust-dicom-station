# Digitally reconstructed radiographs

A DRR is a line integral of attenuation from a point source through the CT
to a flat detector: the simulated radiograph a treatment beam would produce,
and the image every image-guidance workflow compares a portal or kV image
against. *Tools ▶ ☢ Digitally reconstructed radiograph…* renders one — with
**two independent forward projectors**, because there is more than one
honest way to compute it and the difference between them is worth seeing.

## The two projectors

### Siddon — plastimatch's exact ray tracer

`drr -i exact`, after Siddon (Med. Phys. 1985) with Jacobs' incremental
formulation. The ray is intersected with the three families of voxel planes
and each voxel contributes exactly the length of ray inside it: the
algorithm walks voxel to voxel, always crossing whichever plane comes next,
and never allocates or sorts. There is no interpolation and no sampling
step — for a piecewise-constant volume the result *is* the integral, which
makes it the reference the other projector is checked against. Edges come
out hard, because in the voxel model they are.

### Ray-cast — the ITK / elastix-stack interpolating projector

`itk::RayCastInterpolateImageFunction`, the projector behind ITK's 2-D/3-D
registration metrics. The ray is marched at a fixed step and trilinearly
interpolated values are accumulated with a midpoint rule. The volume is
treated as a smooth field rather than a set of boxes, so edges are softer
and the step size is a real accuracy/speed knob rather than a formality.

Running both on the same geometry and subtracting is the point: the
difference image and its statistics (max, mean absolute, RMS, relative, and
the Pearson correlation of the two images) are a direct measure of the
interpolation error you accept by choosing either. On a uniform phantom at
0.5 mm step the two agree to r > 0.999 and a few percent mean difference,
concentrated — as it must be — on the edges.

## Geometry

[`Geometry`] is a cone-beam geometry in IEC 61217 terms, because that is how
a linac states it and how an RTPLAN beam stores it:

* **SAD / SID** — source-to-axis and source-to-imager distances, mm.
* **Gantry angle** — 0° puts the source directly above the patient, 90° at
  the patient's left.
* **Couch angle** — patient-support rotation about the vertical axis.
* **Isocentre** — in patient coordinates; the ⌖ button takes it from the
  dataset's crosshair.
* **Panel size and pixel count** — the window reports the resulting
  resolution projected back to the isocentre plane, which is the number that
  matters when comparing against a real image.

The IEC fixed frame is mapped to the DICOM patient frame for a head-first
supine patient: `Xf` (patient left) = `+x`, `Yf` (the gantry rotation axis,
towards the head) = `+z`, `Zf` (vertical, up) = `−y`. Unit tests assert the
source position at 0° and 90°, and that the detector axes stay orthonormal
and perpendicular to the beam at every gantry/couch combination.

**From beam** takes the gantry angle, the couch angle and the isocentre from
a beam of the loaded plan — the beam's-eye view it would actually deliver.

## Values

* **Attenuation (μ from HU)** — `μ = μ_water · (1 + HU/1000)`, clamped at
  zero, with `μ_water = 0.0206 mm⁻¹` (≈ 60 keV, the effective energy
  plastimatch's DRR preprocessing assumes). The integral is then a real
  optical depth: 40 mm of water on the central axis integrates to
  `0.0206 × 40`, which is what the unit test checks, for both projectors.
* **Raw line integral** — integrate the values as they are (plastimatch
  `-h none`). No physics, but it is what you want when comparing against
  another tool's raw output.
* **Threshold** — voxels below it contribute nothing, which is the standard
  way to keep air and the couch out of a DRR.

## Display

The two renderings are shown side by side with a shared display window
(black/white points as fractions of the value range), an invert toggle —
radiographs are usually read dark-on-light — and a **Difference** view that
maps the signed difference blue↔red about a grey zero.

## Into the data tree

*➕ Add to dataset A/B* files the current rendering (or both, when the two
projectors were run together) under **Planar images** in that dataset's
tree. A DRR *is* an RT Image, so once it is one it inherits everything the
tree already does: its own viewer window with window/level and the correct
physical aspect ratio, renaming, and travelling with the dataset when it is
copied or moved.

The geometry that produced it rides along as the info rows the planar viewer
lists — engine, SAD/SID, gantry and couch angles, isocentre, panel size, HU
model, threshold, sampling step (ray-cast only) and render time — so a
radiograph that has been sitting in the tree for an hour can still say
exactly what it is. Labels are `DRR Siddon · G 90° C 0°` and are made unique
on the way in, because rendering the same geometry twice is what one does
while tuning it.

Whichever greyscale the window is showing is what gets stored: with
**Invert** on (the default) the values are mirrored about the middle of the
range so dark is high attenuation, as on a radiograph. The range itself is
unchanged either way, and the info rows say which convention was used.

Planar images are viewer-side objects: they are not written by
*File ▶ Export dataset*, which covers CT, RTSTRUCT, SEG, RTDOSE and RTPLAN.

## Where it fits

DRR generation is a *simulation* feature, not a registration one: it shares
no code with [registration.md](registration.md). It is, however, the natural
input to 2-D/3-D registration, which is why the interpolating projector is
the one ITK's 2-D/3-D metrics use — and why having the exact one beside it,
on the same geometry, is worth the second implementation.
