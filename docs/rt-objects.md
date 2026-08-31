# RT DICOM objects

Alongside the image series the viewer parses the RT objects below and
resolves the DICOM reference chains between them.

## RTSTRUCT — structure sets

Parsed per file: ROI names, display colors, interpreted types (PTV, CTV, GTV,
ORGAN, EXTERNAL, …) and all planar contours in patient coordinates. ROIs are
ordered EXTERNAL → PTV → CTV → GTV → alphabetical, and a fallback 12-color
palette fills in for structure sets without stored colors.

Axial views draw the **native closed contours**; sagittal and coronal views
show the **reconstructed cross-section silhouette** of each ROI (even–odd
crossing pairing of the contour stack). Per-ROI visibility toggles live in the
sidebar, with All/None shortcuts. **Every** structure set in the folder is
loaded (e.g. one per 4DCT phase) and selectable; the set referencing the
active image series (RTReferencedSeriesSequence) is chosen automatically and
follows series switches. Structure sets also feed the 3D surface view — see
[segmentation.md](segmentation.md).

## SEG — DICOM Segmentation objects

A Segmentation instance is a multi-frame image of binary masks, one frame per
(segment, slice) pair, placed in patient space by the per-frame functional
groups rather than a slice index. Reading one rebuilds a lattice from the
frame positions: frames are grouped into slice levels along the stack normal,
the slice spacing is the median level distance, and the in-plane geometry
comes from `PixelMeasuresSequence` / `PlaneOrientationSequence` (shared group
first, first per-frame group as a fallback).

Supported: `BINARY` (1 bit per pixel, packed across *all* frames as one
continuous stream) and `FRACTIONAL` (8 bit, thresholded at half
`MaximumFractionalValue`). Segment labels come from `SegmentSequence`, colors
from `RecommendedDisplayCIELabValue` via CIELab → XYZ (D65) → sRGB, with the
8-color segmentation palette as fallback. Compressed (encapsulated) Pixel Data
is reported as a load warning.

Each SEG file becomes one **segmentation series** in the data tree, linked to
the image series named in `ReferencedSeriesSequence`. The masks keep their own
lattice and are resampled onto the displayed volume only when their own image
series is shown, so a study can carry segmentations of several series at once.
Writing is the reverse: only the slices a segment occupies become frames, so a
ten-slice structure on a 200-slice CT costs ten frames. See
[export-and-tools.md](export-and-tools.md#dicom-export).

## RTDOSE — dose grids

16- and 32-bit dose grids with `DoseGridScaling` applied at load,
`GridFrameOffsetVector` handled in full generality (uniform or not, ascending
or descending — descending grids are re-ordered) and the frame offsets
re-based onto ImagePositionPatient. Multiple dose files (plan and/or per-beam)
are listed and selectable.

Sampling is trilinear in patient space (bilinear in-plane, linear across the
possibly non-uniform frame offsets), with an incremental affine fast path when
resampling a whole display plane. Display offers:

* a translucent **colorwash** with adjustable opacity and a lower threshold
  (in % of the reference dose);
* **isodose lines** at configurable percentages, extracted per level with
  marching squares (parallelized across levels).

The **reference dose** defaults to the plan's `TargetPrescriptionDose` and can
be overridden; the status bar shows Gy and % of reference at the crosshair for
both datasets.

## RTPLAN — photon and ion plans

Photon (`BeamSequence`) and ion/proton (`IonBeamSequence`) plans are
summarized: label, date, prescription and fractionation, and a per-beam table
with radiation type, delivery type, scan mode (for scanned ion beams),
gantry/couch angles, energy range, meterset and control-point count. Beam
isocenters are marked in all three views (toggleable).

## REG — spatial registration objects

Rigid Spatial Registration files are parsed into their 4×4 frame-of-reference
matrices, shown with the decomposed translation/rotation and
frame-of-reference hints (matched against the loaded studies' FoR UIDs). A
matrix can be **applied as the active registration** in either direction,
optionally inverted, so a TPS-exported registration drives the fusion overlay
and the cross-study crosshair link without running the optimizer; it is
validated (orthonormality, no reflection/scale) before being accepted.

**Deformable Spatial Registration** objects are read the same way, grid
included: the displacement lattice becomes a transform applicable in either
direction, and everything downstream — fusion, the crosshair link, the
analytics, the vector-field display, structure propagation — works on it
unchanged. The panel reports the lattice size, spacing and largest
displacement, and which loaded dataset the grid's frame of reference matches.

A registration recovered here can be written back out as a Deformable Spatial
Registration (*Image registration ▶ Vector field ▶ 💾 Save as DICOM…*); the
IOD's pre- and post-deformation matrices are written as the identity and the
grid carries the whole mapping. See [registration.md](registration.md).

## RTRECORD — treatment records

RT (Ion) Beams Treatment Records are summarized per session: fraction number,
date, machine, and a per-beam table of specified vs delivered meterset with
percentage difference and termination status (non-NORMAL highlighted).

## Reference chains

The viewer parses and preserves the standard chain

```
CT series ◀ RTSTRUCT ◀ RTPLAN ◀ RTDOSE
```

and uses it to select the structure set matching the displayed series, pair
doses with their plans (and hence the prescription dose), define tree
copy/move semantics (a series carries exactly its dependent RT objects) and
drive DICOM export (the chain is written back out). Frame-of-Reference UIDs
associate objects spatially; RT objects with a different FoR still load and
display, but patient-space overlays are only meaningful within one frame of
reference (or through an explicit registration).
