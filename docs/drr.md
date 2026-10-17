# Digitally reconstructed radiographs

A DRR is a line integral of attenuation from a point source through the CT
to a flat detector - the simulated radiograph image guidance compares a
portal or kV image against. *Modules ▶ Image generation ▶ ☢ Digitally
reconstructed radiograph* (right panel, F10) renders one with **two independent forward projectors** and
files it straight into the workspace's data tree, under *Planar images*,
where the row's planar panes show it.

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

*Run both and compare* renders the same geometry with each of them; the
difference of the two and its statistics (max and mean absolute
difference, relative, Pearson correlation) measure the interpolation
error. On a uniform phantom at 0.5 mm step the two agree to r > 0.999 and
a few percent mean difference, concentrated on the edges.

## Geometry

[`Geometry`] is a cone-beam geometry in IEC 61217 terms - how a linac
states it and an RTPLAN beam or an RT Image stores it:

* **SAD / SID** - source-to-axis and source-to-imager distances, mm.
* **Gantry angle** - 0° source above the patient, 90° at the patient's left
  (for a head-first supine patient).
* **Couch angle** - patient-support rotation about the vertical axis,
  counter-clockwise seen from above; **pitch / roll** of a 6-DOF table top.
* **Patient** - how the patient lies or sits: one of the eight DICOM
  Patient Positions (HFS, HFP, FFS, FFP and the four decubitus ones), or
  upright facing a direction (0° faces the gantry), the CT's own Patient
  Position unless changed.
* **Isocentre** - in patient coordinates; ⌖ takes the workspace's crosshair.
* **Panel size and pixel count** - the section reports the resolution
  projected back to the isocentre plane. An RT Image's receptor may sit
  off the beam axis (*X-Ray Image Receptor Translation*) and turned about
  it (*Angle*); its pixels need not be centred.

The chain from the room to the patient is `src/iec.rs`: the fixed frame
(`Zf` up, `Yf` along the gantry axis towards the gantry), the gantry, the
receptor, the patient support, the table top, and the patient's position
on it. For a head-first supine patient `Xf` (patient left) = `+x`, `Yf`
(towards the head) = `+z`, `Zf` (up) = `−y`. Unit tests check every link
on a hand-computed case, that each of the eight positions is a right-handed
frame putting the patient where its code says, and that the camera of a
geometry matches the beam frame the projectors used before.

Everything ends in one camera (`projection::ProjectionGeometry`): a source,
the centre of the first pixel and the steps along a row and down a column,
in patient millimetres. **From beam** takes gantry angle, couch angle and
isocentre from a beam of the loaded plan. **From RT image** renders exactly
the view of an RT image of the workspace - its gantry, couch, distances,
receptor, pixels and patient position (the section says *Camera of* and the
image's name; ✖ goes back to the parameters). **From room view** takes a calibrated view of a room, below.

### Calibrated rooms

A room-mounted kV pair has no gantry angle to describe it; it has a
calibration per view. `imaging-rooms.json` in the configuration folder
lists rooms, each with the way the patient lies in it and its views, each
view the camera outright in the room's fixed frame (millimetres from the
isocentre) or a 3 × 4 projection matrix with the detector distance to put
the panel at:

```json
{ "rooms": [ {
    "name": "Treatment room 2",
    "patient": "HFS",
    "views": [
      { "name": "kV A", "dims": [1024, 768],
        "matrix": [[...4], [...4], [...4]], "distance_mm": 1800 },
      { "name": "kV B", "dims": [1024, 768],
        "source": [x, y, z], "origin": [x, y, z],
        "row_step": [x, y, z], "col_step": [x, y, z] } ] } ] }
```

`"patient"` is a Patient Position code, or `"upright"` with
`"facing_deg"`. A file that cannot be read is named in the section, with
the reason on hover.

The section works on the workspace the module's *Workspace* row names (the
row is there once two workspaces hold a volume); switching it moves the
isocentre to the centre of that workspace's volume and keeps the rest of
the geometry. The row is locked while a render or a synthetic CT is
running, so a result lands on the workspace it was made from - and a
render that finishes after that workspace was replaced is discarded with a
message rather than filed under someone else's study.

## Values

* **Attenuation (μ from HU)** - `μ = μ_water · (1 + HU/1000)`, clamped at
  zero, with `μ_water = 0.0206 mm⁻¹` (≈ 60 keV, the effective energy
  plastimatch's DRR preprocessing assumes); the integral is a real optical
  depth - 40 mm of water on the central axis integrates to `0.0206 × 40`,
  unit-tested for both projectors.
* **Attenuation at MV (electron density)** - for a treatment beam, where
  Compton scattering dominates: `μ = 0.0049 mm⁻¹ · ρe(HU)`, the relative
  electron density by the bilinear calibration of Schneider, Pedroni and
  Lomax (`1 + HU/1000` up to 100 HU, `1.1 + (HU − 100)/1950` above). Bone
  stands out far less than at kV, as on a portal image.
* **Raw line integral** - the values as they are (plastimatch `-h none`);
  for comparing against another tool's raw output.
* **Threshold** - voxels below it contribute nothing, keeping air and the
  couch out.

## Into the data tree

**▶ Render** is the whole of it: there is no preview to add from. Every
rendering is filed under **Planar images** in the workspace's tree as an RT
Image and comes up in the row's **planar panes** (*Planar 1* and *Planar 2*,
see [viewer.md](viewer.md#planar-images-dx--cr--rtimage)), which are put into
the row when they are not there yet. One rendering goes to the pane the last
render used, so a radiograph shown beside it stays where it is while the
geometry is tuned; *Run both and compare* puts the exact tracer in *Planar 1*
and the ray-cast in *Planar 2*, side by side at the same scale, and writes
the comparison line in the section. **Also file the difference** adds the
signed difference (exact minus ray-cast) as a third planar image, windowed
about zero: mid-grey is agreement, bright is where the exact tracer
integrates more.

From there a DRR is a planar image like any other: its own window / level,
zoom and pan in its pane, the correct physical aspect ratio, renaming, and
travel with the workspace when it is copied or moved. The producing
geometry rides along as info rows - engine, SAD/SID, gantry and couch
angles, isocentre, panel size, HU model, threshold, sampling step (ray-cast
only) and render time - which the pane's image menu lists on hover. Labels
are `DRR Siddon · G 90° C 0°`, made unique on the way in (`#2`, `#3`), so
rendering the same geometry again files a new image rather than replacing
one.

**Invert** (on by default) stores the image the way a radiograph is read:
values mirrored about the middle of the range, so dark is high attenuation;
the range itself is unchanged, and the info rows say which convention was
used. Off, the values are the line integral itself.

Planar images are not written by *File ▶ Export DICOM*, which covers CT,
RTSTRUCT, SEG, RTDOSE and RTPLAN; a 2D/3D registration writes its DRRs as
RT Images with their geometry ([registration-2d3d.md](registration-2d3d.md)).

## Where it fits

The ray-cast is the projector of [2D/3D registration](registration-2d3d.md):
the same camera, the same attenuation, marched in index space through the
CT converted once per run (`projection::render_posed`). The section's
Siddon-against-ray-cast comparison is the check that the fast path still
integrates what the exact one does.
