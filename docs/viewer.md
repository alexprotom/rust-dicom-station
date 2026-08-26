# Image viewing, datasets and interaction

This page covers everything about getting image data into the viewer and
navigating it: the loading pipeline, the three-view MPR layout, the dataset
tree, comparison mode, planar images, and the complete interaction reference.

![single dataset](screenshot.png)

*A lung 4DCT phase with its RT Structure Set. The crosshair sits in the
tumor; the axial view draws the native RTSTRUCT contours,
sagittal/coronal show reconstructed cross-sections of the same ROIs.*

## Loading and volume reconstruction

Opening a folder (*File ▶ Add DICOM folder…*, or directory arguments on the
command line) starts a background scan:

1. **Classification.** Every file in the directory tree is read header-only
   (up to but excluding pixel data) in parallel. Files are classified by SOP
   class / modality: image series (CT/MR/PT/…), RTSTRUCT, RTDOSE, RTPLAN,
   planar images (DX/CR/RTIMAGE), REG spatial registrations, RT treatment
   records. Unreadable or foreign files become warnings, never errors.
2. **Series grouping.** Image files are grouped by SeriesInstanceUID and
   presented in the dataset tree; the largest series is reconstructed first
   (switchable at any time by clicking another series).
3. **Volume reconstruction.** Slices of the chosen series are decoded in
   parallel (`rayon`), including compressed transfer syntaxes (JPEG
   lossless, RLE, …) via `dicom-rs`'s pure-Rust decoders. Slices are sorted
   by their projection onto the true slice normal (the cross product of the
   ImageOrientationPatient row/column vectors), checked for uniform spacing
   and consistent dimensions, and rescaled to HU with the per-file rescale
   slope/intercept. The result is a single `i16` volume with full
   patient-space geometry (origin at the center of voxel (0,0,0), unit
   direction vectors for the three axes, spacing in mm).

Non-uniform slice spacing is detected and reported as a warning (the median
spacing is used for display). Duplicate slice positions are collapsed.
Enhanced multi-frame image series are not yet supported (classic
single-frame series only).

RT objects found in the folder are parsed alongside and attached to the
study — see [rt-objects.md](rt-objects.md).

## The three-view MPR layout

The main area shows **axial, sagittal and coronal** planes side by side
with linked crosshairs: clicking a point in any view moves all three (and,
in comparison mode, the other dataset's views) to that patient-space
position. The three planes are extracted in acquisition index space, which
maps directly onto axial/sagittal/coronal for standard axial acquisitions;
oblique acquisitions display consistently but the plane names are nominal —
the anatomical edge labels (L/R/A/P/S/I) always reflect the true patient
directions derived from the direction cosines.

Each viewport carries two corner buttons (both name themselves on hover):
**⟲** resets that view's zoom and pan and puts the crosshair back at the
volume center — which returns that dataset's three views to their central
slices — and **⛶ / ❐** maximizes the view to fill the window and restores
the layout again. The toolbar holds a global **⟲** (the same reset for
every view of both datasets), the **⌖** crosshair toggle (while hidden,
left-click navigation is disabled entirely and slices change only by
scrolling), the **3D A / 3D B** buttons and the segmentation tools.

**Window/level.** Right-drag on any view adjusts interactively
(x = width, y = center); the toolbar offers the numeric fields and the
common CT presets: brain, subdural, stroke, head/neck soft tissue, temporal
bone, lungs, mediastinum, abdomen, liver, spine, bone, CT angio, full
range. Window/level is shared between datasets A and B so both CTs are
windowed identically.

**Status bar.** Patient coordinates, voxel indices, HU and dose (Gy and %
of the reference dose) at the crosshair; in comparison mode the readouts
for A and B are shown side by side.

## Interaction reference

| Input | Action |
|---|---|
| Left click / drag | Move the linked crosshair (all views follow) |
| Mouse wheel | Scroll through slices |
| Ctrl + wheel / pinch | Zoom (anchored at the cursor) |
| Middle drag | Pan |
| Right drag | Window/level (x = width, y = center) |

With a segmentation tool active the left button paints instead of
navigating — see [segmentation.md](segmentation.md) for those bindings.
The bindings of the active tool are always shown in the status bar and in
full under *Help*.

## Datasets and the patient ▶ study ▶ series tree

The two viewer slots are **dataset A** and **dataset B** — each is a
working set that can hold any number of patients, studies and series
accumulated from any number of folders. *File ▶ Add DICOM folder to A/B…*
merges a scanned folder into the slot without unloading what is already
there; duplicates (by UID) are skipped and reported.

The sidebar shows each dataset as a full DICOM hierarchy — patient
(PatientName/PatientID) ▶ study (StudyInstanceUID, with date and
description) ▶ image series — with the displayed series marked; clicking
another series loads it. The standard reference chain is parsed and shown
as links: each structure set displays the image series its contours were
drawn on (RTReferencedSeriesSequence), each dose the plan it was computed
for (ReferencedRTPlanSequence), and each plan the structure set it was
created on (ReferencedStructureSetSequence).

**Right-clicking** any level of the tree — patient, study or series —
opens a context menu to **rename**, **copy**, **move** or **remove** it. Copy/move
transfer the selection into the other dataset (A ▶ B or B ▶ A), merging it
with whatever is already loaded there and switching comparison mode on;
move and remove then delete the selection from its source. A single series
carries exactly its DICOM reference chain: the structure sets drawn on it,
the plans made on those structure sets, and the doses computed for those
plans — nothing else. Study and patient selections additionally take the
RT objects filed under the same studies. Right-clicking a dataset header
offers *Clear dataset*.

## Structures and segmentations in the tree

Below the image series, each dataset lists its **RT structures** and its
**Segmentations** as series nodes — one per RT structure set, one per
DICOM Segmentation series — each showing the image series it is drawn on
(`▶ CT chest`, or `▶ (unlinked)`). Clicking a node makes it the active
one; the items of the active node are listed underneath.

*➕ New series* creates an empty structure set / segmentation series bound
to the displayed image series. **Right-clicking a series node** offers:

* *🔗 Connect to image series ▶* — re-point the series at any image series
  of the dataset (● marks the current one). Contours are in patient
  coordinates and simply follow; a segmentation series is resampled onto
  the new series' lattice the next time that series is displayed.
* *Copy / Move series to dataset A/B*.
* *💾 Export as DICOM SEG…* (segmentation series only) — writes this one
  series as a single SEG file.
* *🗑 Remove this RT structure set / segmentation series*.
* *✎ Rename series…*.

Each item's **check box is both its visibility and its selection**, so
*All* / *None* tick everything or nothing and the right-click actions
operate on whatever is ticked. **Right-clicking a structure or segment**
offers:

* *Copy … to ▶* / *Move … to ▶* — a submenu of every structure set and
  segmentation series in **both** datasets, plus *➕ a new RT structure
  set* / *➕ a new segmentation series* as destinations. Right-clicking a
  ticked row acts on all ticked rows at once; right-clicking an unticked
  row acts on that row alone.
* *🗑 Remove …* — the same single-or-selected rule.
* *✎ Rename …* — always the row you clicked, never the whole selection.

Crossing between the two kinds is a conversion, done on transfer: a
structure moved into a segmentation series is rasterized onto that series'
lattice (even–odd fill, so a doughnut stays a doughnut), a segment moved
into a structure set becomes closed planar contours (marching squares),
and a segment moved between two segmentation series on different lattices
is resampled. Anything that cannot cross — a contour outside the
destination volume, a mask that does not overlap it — is reported in the
dataset's *Warnings* section instead of arriving empty.

## Renaming

Everything the tree names can be renamed from its own right-click menu:
patients, studies, image series, RT structure sets, segmentation series,
individual structures and segments, dose grids, plans, planar images,
spatial registrations and treatment records. The dialog is a single text
field — Enter applies, Esc cancels, an empty name is not accepted — and it
says which DICOM attribute the text lands in.

A patient and a study are *groupings* rather than objects, so renaming one
writes `PatientName` / `StudyDescription` into **every** series filed under
it; the tree would otherwise split into an old and a new node. Everything
else writes the one attribute it shows: `SeriesDescription`,
`StructureSetLabel`, `ROIName`, `SegmentLabel`, `RTPlanLabel`, and the
labels of the remaining objects.

Renames are in-memory. They change what the tree, the overlays and the 3D
view call things, and they are what a DICOM export writes out; the files a
study was loaded from are never modified.

## Comparison mode

![comparison mode](screenshot_comparison.png)

*Two opposite breathing phases of the same 4DCT as datasets A and B, each
with its phase-specific structure set; the linked crosshair pins all six
views to the same patient-space point inside the tumor.*

Load a second dataset (menu, tree copy/move, or two directories on the
command line) and the window splits into two rows of three views — dataset
A on top, dataset B below. Each dataset keeps its own structures, dose and
plan panels in the sidebar; window/level and dose display settings are
shared. The crosshair is linked between the datasets through **patient
coordinates** (toggleable via *View ▶ Link crosshairs between datasets*);
when a registration is active, the link maps through the recovered
transform instead — see [registration.md](registration.md).

A concrete example with the bundled data: load `example_data/` and both
4DCT phases appear as two series of one study. Right-click
*CT 4DCT_phase_050* ▶ *Copy series to dataset B* — the phase moves into
the lower row together with its own phase-specific RTSTRUCT (the reference
chain picks the correct one automatically), and comparison mode switches
on. Click the tumor in any view: all six panels jump to that point, and the
respiratory differences between the phases are read directly by comparing
the rows.

## Planar images (DX / CR / RTIMAGE)

Digital radiographs and RT images (DRRs, portal/setup images) found in the
study folder are listed in the sidebar and open in floating viewer windows
with their own window/level (opens at the DICOM default; auto, manual, or
interactive right-drag exactly like the CT views), correct physical aspect
ratio (imager / image-plane pixel spacing), MONOCHROME1 inversion, and the
relevant metadata — body part, view and kVp for DX; machine, gantry angle,
SAD and SID for RTIMAGE.

## Appearance

*View ▶ Appearance* switches between **🌙 Dark**, **☀ Light** and
**💻 System** (follows the OS setting and updates live). The choice is
remembered in `viewer_settings.txt` next to the executable — a tiny
`key = value` text file, safe to edit or delete. The image viewports stay
black in both themes, as in clinical viewers, so grayscale windowing, the
dose colorwash and the overlay annotations keep a single calibrated
appearance; unit tests assert the accent colors clear WCAG AA contrast
against both backgrounds.
