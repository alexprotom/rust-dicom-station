# RT DICOM objects

The viewer loads a complete radiotherapy study: alongside the image series
it parses RT Structure Sets, DICOM Segmentation objects, RT Dose, RT Plans
(photon and ion/proton), Spatial Registration objects and RT treatment
records, and resolves the DICOM reference chains between them.

## RTSTRUCT — structure sets

Parsed per file: ROI names, display colors, interpreted types (PTV, CTV,
GTV, ORGAN, EXTERNAL, …) and all planar contours in patient coordinates.
ROIs are ordered EXTERNAL → PTV → CTV → GTV → alphabetical, and a fallback
12-color palette fills in for structure sets without stored colors.

Display: axial views draw the **native closed contours**; sagittal and
coronal views show the **reconstructed cross-section silhouette** of each
ROI (even–odd crossing pairing of the contour stack). Per-ROI visibility
toggles live in the sidebar, with All/None shortcuts.

**Every** structure set found in the folder is loaded (e.g. one per 4DCT
phase) and selectable; the set that references the active image series
(RTReferencedSeriesSequence) is chosen automatically and follows series
switches. Structure sets also feed the 3D surface view — see
[segmentation.md](segmentation.md).

## SEG — DICOM Segmentation objects

A Segmentation instance is a multi-frame image whose frames are binary
masks, one per (segment, slice) pair, placed in patient space by the
per-frame functional groups rather than by a slice index. Reading one
therefore means rebuilding a lattice from the frame positions: the frames
are grouped into slice levels along the stack normal, the slice spacing is
the median level distance, and the in-plane geometry comes from
`PixelMeasuresSequence` / `PlaneOrientationSequence` (shared group first,
first per-frame group as a fallback).

Supported: `BINARY` (1 bit per pixel, packed across *all* frames as one
continuous stream) and `FRACTIONAL` (8 bit, thresholded at half
`MaximumFractionalValue`). Segment labels come from `SegmentSequence`, and
segment colors from `RecommendedDisplayCIELabValue`, converted through
CIELab → XYZ (D65) → sRGB; segments without a stored color fall back to the
8-color segmentation palette. Compressed (encapsulated) Pixel Data is
reported as a load warning rather than guessed at.

Each SEG file becomes one **segmentation series** in the data tree, linked
to the image series named in `ReferencedSeriesSequence`. The masks keep the
lattice they arrived on and are resampled onto the displayed volume only
when their own image series is the one being shown, so a study can carry
segmentations of several series at once without any of them being silently
reinterpreted on the wrong grid.

Writing is the same shape in reverse: only the slices a segment actually
occupies become frames, so a ten-slice structure on a 200-slice CT costs
ten frames. See [export-and-tools.md](export-and-tools.md#dicom-export).

## RTDOSE — dose grids

16- and 32-bit dose grids with `DoseGridScaling` applied at load,
`GridFrameOffsetVector` handled in full generality (uniform or not,
ascending or descending — descending grids are re-ordered), and the frame
offsets re-based onto ImagePositionPatient. Multiple dose files (plan
and/or per-beam doses) are listed and selectable.

Sampling is trilinear in patient space: bilinear in-plane plus linear
across the (possibly non-uniform) frame offsets, with an incremental
affine fast path used when resampling a whole display plane. Display
offers:

* a translucent **colorwash** with adjustable opacity and a lower
  threshold (in % of the reference dose);
* **isodose lines** at configurable percentages, extracted per level with
  marching squares (parallelized across levels).

The **reference dose** defaults to the prescription dose picked up from
the plan (`TargetPrescriptionDose`) and can be overridden. The status bar
shows Gy and % of reference at the crosshair for both datasets.

## RTPLAN — photon and ion plans

Photon (`BeamSequence`) and ion/proton (`IonBeamSequence`) plans are
summarized: label, date, prescription and fractionation, and a per-beam
table with radiation type, delivery type, scan mode (for scanned ion
beams), gantry/couch angles, energy range, meterset and control-point
count. Beam isocenters are marked in all three views (toggleable).

## REG — spatial registration objects

Rigid Spatial Registration files are parsed into their 4×4
frame-of-reference transformation matrices, shown with the decomposed
translation/rotation and frame-of-reference hints (matched against the
loaded studies' FoR UIDs). A matrix can be **applied as the active
registration** in either direction, with an optional inversion — a
TPS-exported registration then immediately drives the fusion overlay and
the cross-study crosshair link without running the optimizer. The matrix is
validated (orthonormality, no reflection/scale) before being accepted.

**Deformable Spatial Registration** objects are read the same way, grid
included: the displacement lattice becomes a transform that can be applied
in either direction, after which everything downstream — fusion, the
crosshair link, the analytics, the vector-field display, structure
propagation — works on it without knowing where it came from. The panel
reports the lattice size, its spacing and its largest displacement, and says
which loaded dataset the grid's own frame of reference matches, so applying
it the wrong way round is a deliberate act rather than an accident.

A registration recovered here can be written back out as a Deformable
Spatial Registration (*Image registration ▶ Vector field ▶ 💾 Save as DICOM…*).
The IOD applies its grid after a pre-deformation matrix and before a
post-deformation one; both are written as the identity and the grid carries
the whole mapping, so another system has no composition rule to get wrong.
See [registration.md](registration.md).

## RTRECORD — treatment records

RT (Ion) Beams Treatment Records are summarized per session: fraction
number, date, machine, and a per-beam table of specified vs delivered
meterset with the percentage difference and the termination status
(non-NORMAL terminations highlighted).

## Reference chains

The viewer parses and preserves the standard chain

```
CT series ◀ RTSTRUCT ◀ RTPLAN ◀ RTDOSE
```

and uses it for: automatic selection of the structure set matching the
displayed series, pairing doses with their plans (and hence the
prescription dose), tree copy/move semantics (a series carries exactly its
dependent RT objects), and DICOM export (the chain is written back out).
Frame-of-Reference UIDs associate objects spatially; RT objects with a
different FoR still load and display, but patient-space overlays are only
meaningful within one frame of reference (or through an explicit
registration).
