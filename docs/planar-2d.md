# Planar images: 2D contours, presentation states, 2D/2D registration

The planar images of a study (radiographs, RT images, DRRs) carry 2D
contours of their own, can be registered one onto another, and can show
the CT's structures as they project. One object ties these together: the
**planar structure set**, to a planar image what an RT structure set is to
a CT. Drawing makes one, projection makes one, a 2D/2D registration carries
one from one image to another, and a DICOM Grayscale Softcopy Presentation
State is its file.

Code: `crates/rds-domain/src/planar_structs.rs` (the model, projection,
carrying), `gsps.rs` (the file), `registration/planar.rs` (2D/2D
registration), `setup.rs::from_planar` (a setup from two 2D matches); in
the viewer `app/planar_contours.rs` (display, projection job, saving),
`app/planar_tools.rs` (the contour tools on a pane), `app/project_win.rs`
(the projection dialog), `app/reg2d.rs` (the 2D/2D section),
`app/tree_planar_structs.rs` (the data tree); headless
`mcp/tools/planar.rs`, `workflow/graph/nodes/planar.rs`,
`examples/reg2d_cli.rs`; tests `tests/planar.rs` (run with `--release`).

## Conventions

- **Pixels** are `[column, row]`, fractional, with the centre of the first
  pixel at `(0, 0)` - the convention of the 2D/3D section's regions and of
  the MCP `ImageArg.region`. A presentation state puts the first pixel's
  top left corner at `0\0` instead (PS3.3 C.10.5.1.2), so its writer adds
  one half and its reader subtracts it.
- **A contour belongs to an image by SOP Instance UID**, never by its index
  among the planar images. Every planar image made here (a DRR, a DRR at a
  setup, a moved copy) is given its SOP Instance UID and a Series Instance
  UID the moment it is made, and `write_rt_image` writes those UIDs, so a
  contour, a presentation state and the RT Image it is on reference each
  other. Removing an image leaves the sets alone; a set whose images are
  not loaded says so (*n image(s) not loaded*) and attaches when they are.
- **A structure's rings on one image are filled even-odd**
  (`contours::Region`), exactly as the RT structures' contours on a slice.
- **An image's millimetre frame** is its pixels times its spacing about the
  image centre: `x = (i + 0.5 − cols / 2) · sx`. Every 2D transform is
  written in it, so a shift reads in millimetres at the detector, and at
  the isocentre when the image has a camera (divided by the magnification).
- **A 2D/2D transform maps the fixed image's frame onto the moving
  image's**, as every registration of the program does: the fixed image is
  the one the answer is shown on (today's radiograph), the moving one is
  resampled onto it (the reference image, the DRR at the plan).

## The planar structure set

A set has a label, its structures (name, colour, at most one contour per
image, points of interest), the UIDs of the images it references, whether
it is locked, and where it came from (`drawn`, `projected`, `carried`, or
the file it was read from). Each contour remembers its source - drawn,
projected from structure *S* (at setup *n*), carried by registration *R*,
read from a file - and the tree row's tooltip says it.

In the data tree, *Planar structures (n)* sits under *Planar images*: a
fold per set (its menu: rename, lock, *⇩ Project structures* into it, *💾
Save as presentation state*, remove), a row per structure (the tick that
draws it, the colour swatch, the name - a click makes it the structure the
contour tools edit and puts one of its images on screen - and its menu:
rename, *⇥ Carry to...*, remove).

The planar panes draw the contours of every image of their stack, each
ring through its own image's rectangle, so a contour on a DRR rendered at a
radiograph's geometry lies on the radiograph it is stacked with: the image
under the scrubber in full colour, the others at half alpha, the structure
being edited thicker. Points are crosses with their structure's name.

## Drawing on a pane

The toolbar's contour tools (polygon, spline, freehand, live wire, contour
brush, nudge) work on a planar pane as on an MPR view, with the same
gestures (click by click and a right click, Enter or a double click to
close; one drag for freehand; Alt erases with the brush; Ctrl-click picks
the structure under the pointer) and the same *Auto / Extend / Subtract*
rule (`contours::auto_cuts`: a ring that crosses a contour and starts
outside it cuts, anything else adds). A stroke on a pane makes the pane the
tools' target, a stroke in an MPR view makes the volume the target again;
the toolbar row then reads `▸ 2D · set · structure`, and Ctrl+Z, Esc and
Enter act on the planar structure (32 undo steps). The first stroke on an
image with no structure chosen makes one (`2D n`) in an unlocked set that
already has contours on the image, else in a new set labelled after it;
`+` makes a new structure. The brush radius is millimetres of the image's
own pixel spacing (an ellipse in pixels when the spacings differ). The
live wire runs on the image's pixels under the pane's window. The smart
brush's *Bright* and *Dark* bands cut the stamp back to what looks bright
or dark in the pane's window, as on a slice; *Bone* and *Air* are CT numbers
and leave the stamp as it is on a radiograph. The voxel tools (brush, erase,
grow) edit segmentations and do nothing on a pane. Undo is
`planar_structs::Undo` (32 steps, per workspace).

## Projection

`planar_structs::project_mask` casts a structure's voxel mask
(`MuVolume::from_mask`) through the image's camera, moved by a pose when
there is one. Each pixel keeps the *largest* trilinear sample along its ray
(`MuVolume::max_along`); the iso-line at one half of that field, by
marching squares with linear interpolation, stitched into closed rings by
the cell edges they cross, is the **silhouette** of the mask's half-level
surface - the shadow a beam's eye view shows. Only the pixels the mask's
box projects onto are cast, on a raster no finer than half a voxel at the
structure's depth. The 2D/3D section's live outlines are the same rings
(`setup::silhouette` is now a wrapper), drawn as segments.

Where the pose comes from: a DRR's own camera as rendered, a radiograph's
camera at the plan, or a setup of the study (the structure where the setup
found the patient: the camera that setup used for the image, moved by its
pose). The *⇩ Project structures* dialog chooses the structures, the images
(those with a camera, or placed by a setup), the pose and the set; it runs
in the background and files the set. It is reached from a planar image's
tree menu, a set's menu, the DRR section (a *Structures* tick list: every
render projects them at once) and the 2D/3D section's *Outlines* fold
(*⇩ As 2D contours*: the ticked outlines at the current setup's pose).

Accuracy (`tests/planar.rs`): a sphere of 30 mm projected at SAD 1000 / SID
1500 gives one ring whose area is the magnified disc's to 1 %; a projection
at a pose equals the moved structure projected, the centroids 0.05 px apart.

## The presentation state

`gsps::write` makes one Grayscale Softcopy Presentation State per set (SOP
class `1.2.840.10008.5.1.4.1.1.11.1`, modality `PR`, in the set's series):
Presentation State Identification (Content Label as a CS, Content
Description, creation date and time, Content Creator's Name `RDS`), the
Referenced Series Sequence of every referenced image, one Graphic Layer per
structure (a CS of up to 16 characters made unique - `PTV`, `LUNG_L`,
`STRUCT_7` - the full name in the Graphic Layer Description, the colour as
the Recommended Display CIELab Value and its luminance as the Grayscale
Value), one Graphic Annotation item per image and structure with a closed
`POLYLINE` per ring (the first point repeated last, `PIXEL` units, not
filled) and a `POINT` per point of interest, a Displayed Area Selection per
image (the whole image, `SCALE TO FIT`, the image's pixel spacing), the
window each image is shown with as its Softcopy VOI LUT, and an `IDENTITY`
presentation LUT. Many viewers draw every layer in one colour and show the
16-character layer name rather than the description.

A study export (the Export window, the MCP `export`, the workflow step
*Export DICOM*) carries the planar images (`ObjKind::Planar`: a file of its
own copied with only its identity patched, an image made here written as
an RT Image; their UIDs follow the export's UID mode as the series' do) and
the planar structure sets (`ObjKind::PlanarStructures`: a presentation
state per set, referencing the images under the UIDs they go out with).
The workflow step writes every planar structure set and the images made
here; the MCP tool takes planar sets with the structures and planar images
with the images. The Export window also writes, beside every planar image
with contours of a ticked set on it, a copy with the contours burned into
its pixels (a Secondary Capture, `Burned In Annotation YES`), for the
viewers that draw no presentation state
([export-and-tools.md](export-and-tools.md#2d-contours-burned-in)); the
MCP tool and the workflow step leave it off.

*💾 Save as presentation state* (and the MCP `save_planar_structures`)
writes the state into a folder together with every referenced image that
has no file of its own (a DRR, a moved copy) as an RT Image under the UIDs
the set already references, so the folder is self-contained
(`gsps::write_made_images`, then `gsps::write_state`). File names carry no
label: `gsps_1.dcm` (the set's number) and `rtimage_<16 hex digits>.dcm`, a
hash of the image's SOP Instance UID, so an image saved twice into one
folder - by two sets, or one set saved again - is one file, and two images
never share a name.

A planar image's own menu in the tree saves it alone: *💾 Save ▸ As DICOM*
(the Export window with it and the sets drawn on it ticked), *With contours
burned in (DICOM)* and *As PNG*, both drawn as the panes draw them (every
set's structures on that image less the hidden ones, through the pane's
window). The 2D/3D section's *💾 DICOM* writes, besides the registration,
the DRRs and the report, every planar structure set with contours on the
setup's radiographs as a presentation state that references those
radiographs and nothing else (its contours on other images are left out,
and so is a structure with none on them), the radiographs made here once
each however many sets are drawn on them, and each such radiograph with
the contours burned in (`setup_contours_1.dcm`, ...). A burned-in copy
carries the image's spacing as Imager Pixel Spacing, and Pixel Spacing only
at the isocentre plane when the geometry gives SID and SAD
([export-and-tools.md](export-and-tools.md#2d-contours-burned-in)).

`gsps::read` takes what RDS writes and what others write: `POLYLINE` and
`INTERPOLATED` (as a polyline through its points) become rings, `CIRCLE`
(centre and a point on it) and `ELLIPSE` (the ends of the major, then of
the minor axis) are resampled to 64 points, `POINT`s and text objects'
anchors become points of interest; `DISPLAY` units are mapped through the
image's Displayed Area Selection, and an object in them without one is
skipped with a warning; a graphic in a layer with no Graphic Layer item
gets a structure of its own; a layer's colour is its CIELab value, else
its grayscale value, else the palette. The loader reads grayscale and
colour softcopy presentation states (`FileClass::Presentation`) rather
than counting them among the objects not displayed, and warns once per
state that references images not in the load; the archive description
says a PR series is shown in *planar structures*.

The half-pixel offset of `PIXEL` units is the standard's (C.10.5.1.2:
`0.0\0.0` the top left corner of the top left pixel, `1.0\1.0` its bottom
right corner); a reader that put the first pixel's centre at `0\0` would
see every contour half a pixel up and left.

## 2D/2D registration

*Modules ▶ Image registration ▶ ⇄ 2D/2D registration* (and the MCP
`register_planar`, the workflow step *2D/2D registration*, `reg2d_cli`).

- **Intensity** (`register`): the moving image resampled onto the fixed
  raster through the current transform (bilinear; a pixel outside the
  moving image counts as masked out) and compared by one of the six 2D
  measures of the 2D/3D engine over the fixed mask (the region, less two
  pixels of border), coarse to fine (3 levels by default, the finest no
  larger than 256 pixels a side). Parameters `[tx, ty, θ·R, ln(s)·R]` with
  `R` half the fixed image's diagonal in mm, so Powell's steps mean the
  same in every direction; bounds `±range` mm, `±range` degrees, scale
  within `±10 %`. Freedom: *shift*, *shift and rotation* (the default),
  *similarity*. The coarsest level of a run with no start starts from the
  best whole-pixel shift in range (the 2D/3D engine's `best_shift`); a
  moving image that correlates negatively with the fixed one is negated
  first (a radiograph stored bone dark against a DRR stored bone bright).
  *Refine* adds L-BFGS with finite differences at the finest level. The
  rotation centre is where the isocentre projects when the fixed image has
  a camera and names an isocentre, else the image centre.
- **Points**: pairs placed with *📌 Place* (a click on the fixed image, then
  one on the moving image), markers found in both images (a difference of
  Gaussians at the marker's size, paired by the common shift that lays the
  most spots on spots, then by a fit and a tight re-pairing), or both. The
  closed-form weighted similarity of Umeyama (1991); one pair gives the
  shift alone. A pair beyond three times the median residual is reported
  and left out. *Points, then intensity* uses the fit as the start.
- **Deformable**: a cubic B-spline over the rigid start (Rueckert 1999), two
  coefficients per node, the grid from 32 mm halving to 8 mm, each grid
  fitted on a pyramid level by L-BFGS with the exact gradient of the mean
  squared difference of normalised intensities, or of Mattes mutual
  information (32 bins, the cubic B-spline Parzen window, for a radiograph
  against a DRR), plus the bending energy (second differences per node
  spacing, so dimensionless: λ 0.01 for MSD, 0.3 for MI, whose value moves
  less per millimetre). Coefficients pass from grid to grid by the dyadic
  B-spline refinement (the same displacement). The result keeps the rigid
  part and the warp, a dense field, and the warp's mean and largest
  displacement and Jacobian range; a fold is reported, and contours are
  not carried through it. Not offered as a default for two megavoltage
  images, which have not been tried.

The result is filed with the study (`study.planar_regs`, *2D registrations*
under the planar images: rename, remove), with the moving image resampled
onto the fixed one as a planar image of its own (`"<moving> onto
<fixed>"`, the fixed image's spacing and camera), stacked over the fixed
image in a pane where fade, checkerboard, difference and edges show the
agreement. Removing the moved copy removes the registration with it (the planar
structure sets stay: they name their images by UID). The
section's apply row: *⇥ Carry contours* (the moving image's structures onto
the fixed one, through the inverse - exact for the rigid part, by
fixed-point iteration for the warp - into *Carried · <registration>*),
*⌖ As setup*, *💾 Report* (JSON), *📋 Copy*.

The section's *Region* row draws the part of the fixed image the measure is
taken over: a rectangle (a drag corner to corner) or polygons (click the
corners, right click or double click to close), any number of them, shown
in green over the fixed image; *Place* and the region tools take the pane's
left button from the 2D/3D section's tools (one section at a time). For a
deformable result, *Field* draws the warp over the fixed image: an arrow
per node of a lattice about 24 screen pixels apart (where the warp sends
the point, the rigid part left out) and the lattice deformed. The method,
the measure, the freedom, the deformable measure, the grid and the
smoothness are remembered between runs (`reg2d` in the settings file).

### Two matches as one correction

`setup::from_planar` reads two (or more) 2D matches of two placed views as
one 3D correction, the reading of an orthogonal pair a therapist does by
hand (Murphy 1997): each match's shift at the isocentre's projection gives
`J t = −s` at the isocentre through the fixed image's camera, solved in the
weighted least-squares sense (`projective::translation_from_shifts`, the
2D/3D engine's in-plane start lifted out - the 2D/3D suite is unchanged);
each match's in-plane rotation is a roll about its detector's normal, and
the small-angle fit `ω · a = θ` gives the rotation
(`rotation_from_rolls`). Two orthogonal views see two rotation components;
the third, about the axis both detectors contain, is reported *weakly
constrained*, as the depth is for one view. The answer is a `SetupResult`
of method `planar` (*2D matches*): the couch readout, the tolerance
colouring, the report, the positioned copy, the REG and the RT Images work
as for a 2D/3D run; with a CT the views' DRRs are rendered at the pose.
Two fixed images taken more than two seconds apart are warned about.

**Refine in 3D.** The 2D/3D section's *Start* row takes a setup of the
study as the start pose (*from <setup>*); *⌖ As setup* sets it to the setup
it files, so the next 2D/3D run refines the two-step reading by intensity
(`SetupOptions::start`). The suite holds a planar setup refined in 3D to
0.5 mm.

## Headless

MCP (`mcp/tools/planar.rs`): `project_structures`, `list_planar_structures`,
`draw_planar_structure`, `register_planar`, `carry_planar_structures`,
`setup_from_planar`, `save_planar_structures`; `describe_dataset` lists the
planar structure sets and 2D registrations. Labels pass the redactor; file
names carry none.

Workflow steps: *Project structures* (structures and images by name or
number, at the plan or at the study's last setup) and *2D/2D registration*
(`fixed:moving` pairs, the parameters, optionally *read as a setup*), each
reporting a table; the results land on the study.

`reg2d_cli <fixed.dcm> <moving.dcm>`, or `<image.dcm> --simulate
tx,ty,deg[,scale]` (the moving image the fixed one resampled through the
inverse of that transform, with 1 % noise; the error printed), or
`--bench`. On a DRR of the bundled 4D-Lung case (192 × 192, 1.56 mm):

| Run | Error |
|---|---|
| `--simulate 7,-4,3`, NCC | 0.015 mm at the centre, 0.001° (0.02 s) |
| `--simulate 7,-4,3`, gradient correlation | 0.015 mm, 0.002° (0.06 s) |
| `--simulate 5,3,-2,1.04 --dof similarity` | 0.007 mm, scale −0.002 % |

A resample of 192 × 192 takes 0.4 ms, a measure 0.07 (NCC) to 1 ms
(pattern intensity).

## Validation

`tests/planar.rs`: the projection (area, pose); a written RT Image keeps
its UIDs; merging studies keeps the sets; carrying rigidly; the drawing
rules on a pixel region, filing by UID, the brush round in millimetres on
0.4 × 0.8 mm pixels; a presentation state read back (structures, colours
within a step, vertices to 1e-3 px, references), one whose image is not
loaded kept and attached later, a saved set self-contained; rigid 2D (7 mm
and 3° to 0.2 mm and 0.1° with NCC and gradient correlation; against the
anatomy rendered on another grid to 0.5 mm; a 4 % scale to 0.5 %); three
pairs exact, a pair 6 mm off left out; markers found and paired; a carried
ring on the drawn one to 0.2 px; the same bits on 1, 2 and 8 threads; a
6.5 mm sinusoidal warp recovered to a mean under 0.5 mm and a maximum under
1.5 mm with MSD and with MI across a change of contrast (on a radiograph
given bone-like texture - the smooth phantom's DRR alone leaves a warp
undetermined between the organs' outlines); a ring carried through a warp
and back to 0.1 px; two matches giving the translation to 0.5 mm with the
2D/3D suite's couch signs, the shared rotation axis named weak, a roll read
about its ray, and a planar setup refined in 3D to 0.5 mm; a study export
carrying a made-here DRR and its set (both UID modes) that loads back with
the set attached; undo 32 deep and per workspace; a removed moved copy
taking its registration and nothing else. Unit tests hold the cost gradients to finite differences,
the bending energy of an affine field at zero, the B-spline refinement and
the GSPS reader's substitutions. `tests/load_warnings.rs`,
`tests/mcp_tools.rs`, `tests/mcp_phi.rs` and `tests/workflow_steps.rs` have
a case each.

Before a correction of *As setup* is used in a room, the room's own images
decide, with the phantom procedure of `docs/registration-2d3d.md`.

## Limits

- The presentation state's conventions follow PS3.3 C.10.5 as checked on
  the standard's text (`PIXEL` units with `0\0` at the top left corner of
  the first pixel and `Columns\Rows` at the image's bottom right corner,
  column first; a closed polyline repeats its first point; `CIRCLE` the
  centre and a point on it; `ELLIPSE` the ends of the major then the minor
  axis; `DISPLAY` fractions of the displayed area). It has not been tried
  on a state written by another system, nor opened in another viewer.
- The deformable method has not been tried on two megavoltage images.
- The *2D/2D registration* workflow step puts out a report only; a setup it
  reads from its matches is filed on the study rather than handed on as a
  pose.
- Text objects are read as points of their layer; none is written.
