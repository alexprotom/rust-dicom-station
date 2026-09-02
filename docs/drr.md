# Digitally reconstructed radiographs

A DRR is a line integral of attenuation from a point source through the CT
to a flat detector - the simulated radiograph image guidance compares a
portal or kV image against. *Tools ▶ ☢ Digitally reconstructed
radiograph…* renders one with **two independent forward projectors**.

## The two projectors

### Siddon - plastimatch's exact ray tracer

`drr -i exact`, after Siddon (Med. Phys. 1985) with Jacobs' incremental
formulation: the ray walks voxel to voxel and each voxel contributes
exactly the length of ray inside it. For a piecewise-constant volume the
result *is* the integral, the reference for the other projector; edges
come out hard, because in the voxel model they are.

### Ray-cast - the ITK / elastix-stack interpolating projector

`itk::RayCastInterpolateImageFunction`, the projector behind ITK's 2-D/3-D
registration metrics: the ray is marched at a fixed step, trilinearly
interpolated values accumulated with a midpoint rule - the volume as a
smooth field, so edges are softer and step size a real accuracy/speed knob.

The difference image of the two on the same geometry and its statistics
(max, mean absolute, RMS, relative, Pearson correlation) measure the
interpolation error; on a uniform phantom at 0.5 mm step the two agree to
r > 0.999 and a few percent mean difference, concentrated on the edges.

## Geometry

[`Geometry`] is a cone-beam geometry in IEC 61217 terms - how a linac
states it and an RTPLAN beam stores it:

* **SAD / SID** - source-to-axis and source-to-imager distances, mm.
* **Gantry angle** - 0° source above the patient, 90° at the patient's left.
* **Couch angle** - patient-support rotation about the vertical axis.
* **Isocentre** - in patient coordinates; ⌖ takes the dataset's crosshair.
* **Panel size and pixel count** - the window reports the resolution
  projected back to the isocentre plane.

The IEC fixed frame maps to the DICOM patient frame for a head-first supine
patient: `Xf` (patient left) = `+x`, `Yf` (the gantry rotation axis,
towards the head) = `+z`, `Zf` (vertical, up) = `−y`. Unit tests assert the
source position at 0° and 90° and the detector axes orthonormal and
perpendicular to the beam.

**From beam** takes gantry angle, couch angle and isocentre from a beam of
the loaded plan.

## Values

* **Attenuation (μ from HU)** - `μ = μ_water · (1 + HU/1000)`, clamped at
  zero, with `μ_water = 0.0206 mm⁻¹` (≈ 60 keV, the effective energy
  plastimatch's DRR preprocessing assumes); the integral is a real optical
  depth - 40 mm of water on the central axis integrates to `0.0206 × 40`,
  unit-tested for both projectors.
* **Raw line integral** - the values as they are (plastimatch `-h none`);
  for comparing against another tool's raw output.
* **Threshold** - voxels below it contribute nothing, keeping air and the
  couch out.

## Display

The two renderings sit side by side with a shared display window
(black/white points as fractions of the value range), an invert toggle, and
a **Difference** view mapping signed difference blue↔red about a grey zero.

## Into the data tree

*➕ Add to dataset A/B* files the rendering (or both, when run together)
under **Planar images** in the dataset's tree as an RT Image, with its own
viewer (window/level, correct physical aspect ratio), renaming, and travel
with the dataset when copied or moved.

The producing geometry rides along as the planar viewer's info rows -
engine, SAD/SID, gantry and couch angles, isocentre, panel size, HU model,
threshold, sampling step (ray-cast only) and render time. Labels are
`DRR Siddon · G 90° C 0°`, made unique on the way in.

Whichever greyscale the window shows is what gets stored: with **Invert**
on (the default) values are mirrored about the middle of the range so dark
is high attenuation, as on a radiograph; the range itself is unchanged, and
the info rows say which convention was used.

Planar images are viewer-side objects: they are not written by
*File ▶ Export dataset*, which covers CT, RTSTRUCT, SEG, RTDOSE and RTPLAN.

## Where it fits

DRR generation is a *simulation* feature sharing no code with
[registration.md](registration.md); it is, however, the natural input to
2-D/3-D registration, whose ITK metrics use the interpolating projector.
