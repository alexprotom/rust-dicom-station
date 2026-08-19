# rust-dicom-viewer

A fast, robust DICOM / RT DICOM viewer written entirely in Rust. It loads a
full radiotherapy study — image series (CT/MR/PT), RT Structure Set, RT Dose
and RT Plan (photon and ion/proton) — and displays it in the classic
three-view MPR layout: **axial, sagittal and coronal side by side** with
linked crosshairs. A **comparison mode** stacks a second dataset below the
first for six views total, and built-in **rigid and deformable (B-spline)
image registration** — elastix-style algorithms implemented natively in
Rust — aligns the two datasets with a fusion overlay.

![screenshot](docs/screenshot.png)

*Single dataset: a lung 4DCT phase with its RT Structure Set (lungs, heart,
tumor, cord, markers). The crosshair sits in the tumor; the axial view draws
the native RTSTRUCT contours, sagittal/coronal show reconstructed
cross-sections of the same ROIs.*

![comparison mode](docs/screenshot_comparison.png)

*Comparison mode: two opposite breathing phases of the same 4DCT
(dataset A = phase 0 %, dataset B = phase 50 %), each with its own
phase-specific structure set. The linked crosshair pins all six views to the
same patient-space point inside the tumor, so the respiratory shift of the
diaphragm and tumor is read directly by comparing the rows.*

![registration](docs/screenshot_registration.png)

*Deformable registration B ▶ A on the two phases: the sidebar reports the
recovered transform and metric improvement, and the top row now shows the
magenta/green fusion — aligned anatomy reads gray, residual mismatch shows
as color fringes.*

## Features

**Image volumes.** A directory scan classifies every DICOM file (parallel
header-only pass), groups image series, and reconstructs a 3D volume from the
largest series by default (switchable in the sidebar). Slices are decoded in
parallel with `rayon`, sorted along the true slice normal, checked for
uniform spacing and consistent dimensions, and rescaled to HU. Compressed
transfer syntaxes (JPEG lossless, RLE, …) are handled by `dicom-rs`'s pure
Rust decoders.

**RTSTRUCT.** ROI names, display colors, interpreted types (PTV, ORGAN,
EXTERNAL, …) and all planar contours. Axial views draw the native closed
contours; sagittal/coronal views show the reconstructed cross-section
silhouette. Per-ROI visibility toggles. **Every** structure set in the folder
is loaded (e.g. one per 4DCT phase) and selectable in the sidebar; the set
that references the active image series is chosen automatically and follows
series switches.

**Datasets & the patient ▶ study ▶ series tree.** The two viewer slots are
called **dataset A** and **dataset B** — each is a working set that can hold
any number of patients, studies and series accumulated from any number of
folders (*File ▶ Add DICOM folder to A/B…* merges a scanned folder into the
slot without unloading what is already there; duplicates, by UID, are
skipped and reported). The sidebar shows each dataset as a full DICOM
hierarchy — patient (PatientName/PatientID) ▶ study (StudyInstanceUID, with
date and description) ▶ image series — all visible at once, with the
displayed series marked; clicking another series loads it. The standard
reference chain is parsed and shown as links: each structure set displays
the image series its contours were drawn on (RTReferencedSeriesSequence),
each dose the plan it was computed for (ReferencedRTPlanSequence), and each
plan the structure set it was created on (ReferencedStructureSetSequence).
Exported datasets keep this chain intact.

**Right-clicking** any level of the tree — patient, study or series — opens
a context menu to **copy**, **move** or **remove** it. Copy/move transfer
the selection into the other dataset (A ▶ B or B ▶ A), *merging* it with
whatever is already loaded there and switching comparison mode on; move and
remove then delete the selection from its source. A single series carries
exactly its DICOM reference chain: the structure sets drawn on it, the
plans made on those structure sets, and the doses computed for those plans
— nothing else. Study and patient selections additionally take the RT
objects filed under the same studies. Right-clicking a dataset header
offers *Clear dataset*.

**RTDOSE.** 16/32-bit dose grids with `DoseGridScaling`,
`GridFrameOffsetVector` (uniform or not, ascending or descending), trilinear
patient-space sampling, translucent colorwash and marching-squares isodose
lines at configurable percentages of a reference dose (prescription dose is
picked up from the plan automatically). Multiple dose files (plan/beam doses)
are selectable.

**Comparison mode.** Load a second dataset (menu *File ▶ Add DICOM folder
to B…* or *View ▶ Comparison mode*, or pass two directories on the command
line) and the window splits into two rows of three views — dataset A on
top, dataset B below, six panels total. Each dataset keeps its own
structures, dose and plan panels in the sidebar; window/level and dose
display settings are shared so both CTs are windowed identically. The
crosshair is linked between the datasets through **patient coordinates**
(toggleable via *View ▶ Link crosshairs between datasets*): clicking a
point in one dataset moves the other dataset's crosshair and slices to the
same anatomical position — the status bar then shows HU and dose readouts
for A and B side by side. Dataset B can be closed again from the File menu,
and comparison mode can be switched on/off at any time without unloading
anything.

A concrete example, shown in the screenshot above with the bundled patient
data: load `example_data/` and both 4DCT phases appear as two series of one
study. Right-click *CT 4DCT_phase_050* ▶ *Copy series to dataset B* — the
phase moves into the lower row together with its own phase-specific
RTSTRUCT (the reference chain picks the correct one automatically), and
comparison mode switches on. Click the tumor in any view: because the
crosshairs are linked through patient coordinates, all six panels jump to
that point, and the differences between the phases — diaphragm position,
tumor displacement, chest-wall shape — are read directly by comparing the
two rows slice by slice. The two phases share one frame of reference, so
before any registration this is a pure anatomical comparison of the same
coordinates at two moments of the breathing cycle.

**Image registration (rigid & non-rigid).** With two datasets loaded, the
*Registration* menu (or the sidebar section) registers one dataset onto the
other — the direction is selectable (**B ▶ A** or **A ▶ B**; the second
dataset named is the fixed image, and the fusion overlay is drawn on its
views). The engine follows the [elastix](https://elastix.dev) framework —
[SuperElastix/elastix](https://github.com/SuperElastix/elastix) is a C++/ITK
toolbox, so its core algorithms are **re-implemented natively in Rust** to
keep the application single-language:

* multi-resolution Gaussian pyramids (`NumberOfResolutions`, default 3);
* random-coordinate sampling with fresh samples every iteration
  (`NumberOfSpatialSamples`, default 3000), restricted to a body threshold
  mask;
* mean-squared-difference metric with analytic gradients;
* **Adaptive Stochastic Gradient Descent** (Klein et al., IJCV 2009 —
  elastix's default optimizer) with automatic gain estimation, the
  sigmoid time-adaptation rule and a trust-region step cap;
* **rigid**: 6-DOF Euler transform about the fixed-image center with
  automatic rotation/translation parameter scaling;
* **deformable**: rigid pre-alignment composed with a cubic B-spline
  free-form deformation on a regular grid
  (`FinalGridSpacingInPhysicalUnits`, default 32 mm).

Registration runs on a background thread (progress + cancel in the sidebar)
and typically takes seconds thanks to stochastic sampling. The result panel
reports the direction, the metric before/after, the recovered
translation/rotation, and enables a **magenta/green fusion overlay** on the
fixed study (fixed image in magenta, the transformed moving image in green —
aligned anatomy reads gray) with a blend slider. The cross-study crosshair
link maps through the recovered transform (inverse included), so clicking a
point in either study lands on the same anatomy in the other. Iterations,
samples and grid spacing are adjustable in the sidebar.

On the bundled patient data (registration screenshot above) the workflow
is: two breathing phases loaded as datasets A and B, direction left at
**B ▶ A**, one click on *▶ Deformable*. On this 512 × 512 × 133 CT the
whole run — rigid pre-alignment plus three B-spline resolution levels,
1800 iterations total — takes about 20 s on a desktop CPU and drives the
mean-squared HU difference from ≈ 9700 down to ≈ 1800. The recovered rigid
part is sub-millimeter (the phases share one frame of reference, so almost
all real motion is breathing deformation, which the B-spline absorbs), and
the fusion overlay turns mostly gray: the remaining magenta/green fringes
at the diaphragm and chest wall mark exactly the residual respiratory
motion the 32 mm grid cannot fully model — tightening the grid spacing
refines it further at the cost of runtime. Accuracy is verified in
`tests/registration.rs` against analytically known transforms:
sub-millimeter recovery for both a rigid rotation + translation and a 7 mm
Gaussian-bump deformation.

**Planar images (DX / CR / RTIMAGE).** Digital radiographs and RT images
(DRRs, portal / setup images) found in the study folder are listed in the
sidebar and open in floating viewer windows with their own window/level
(opens at the DICOM default; auto, manual, or interactive right-drag on the
image — x = width, y = center, exactly like the CT views), correct physical aspect ratio
(imager / image-plane pixel spacing), MONOCHROME1 inversion, and the
relevant metadata — body part, view and kVp for DX; machine, gantry angle,
SAD and SID for RTIMAGE.

**REG — Spatial Registration objects.** Rigid Spatial Registration files are
parsed into their 4×4 frame-of-reference transformation matrices, shown with
the decomposed translation/rotation and frame-of-reference hints (matched
against the loaded studies' FoR UIDs). A matrix can be **applied as the
active registration** in either direction (with an optional inversion), so a
TPS-exported registration immediately drives the fusion overlay and the
cross-study crosshair link without running the optimizer. Deformable REG
objects are recognized and their rigid matrices read; deformation grids are
not applied.

**RTRECORD — treatment records.** RT (Ion) Beams Treatment Records are
summarized per session: fraction number, date, machine, and a per-beam table
of specified vs delivered meterset with the percentage difference and the
termination status (non-NORMAL terminations highlighted).

**Transform simulator & DICOM export (registration QA).** The *Simulation*
sidebar section applies an exactly-known transform to a loaded dataset —
rigid motion (translation + Euler rotation about the volume center) plus an
optional local Gaussian deformation (amplitude vector + σ, centered at the
crosshair) — and generates the result into the other dataset slot: the CT is
resampled through the inverse transform, and structure contours, dose grids
and plan isocenters are carried along. The applied parameters stay displayed
as the ground truth, so you can immediately run the built-in registration and
compare the recovered transform against it (on the synthetic phantom the
rigid recovery matches to sub-millimeter/sub-degree). Any loaded dataset —
original or simulated — can be exported as a set of DICOM files (*Export
A/B…*): one CT Image Storage file per slice plus RTSTRUCT, RTDOSE (16-bit
with `DoseGridScaling`) and an RTPLAN skeleton (photon or ion), written with
`dicom-rs` in Explicit VR Little Endian, sharing the source frame of
reference. The exports round-trip through this viewer and pydicom; they are
QA/research objects, not guaranteed-complete clinical IODs.

**DICOM anonymizer (Tools menu).** *Tools ▶ 🔏 Anonymize DICOM folder…* is
an interactive, generalized Rust port of the one-off
`tools/anonymize_dicom.py` that was used to prepare `example_data/`. Pick a
folder and **Scan** it (recursive, background thread): the dialog then lists
every identifying tag actually present — patient identity, birth date/sex,
dates and times, accession number, physicians, institution, station,
device — with its **current value(s)** across the files and a **proposed
replacement**: a deterministic `anon_xxxxxx` patient alias derived from the
original PatientID, the fixed anonymization date `20000101` / time
`000000`, or a cleared value. Every proposal is editable, each row can be
unchecked, and Study/Series descriptions are offered opt-in (unchecked by
default). Three switches control the rest: **regenerate UIDs** replaces
every non-standard UID (study, series, SOP instances, frame of reference,
and every reference to them inside sequences) with fresh `2.25.` UIDs,
consistently across all files, so RTSTRUCT ▶ series, RTDOSE ▶ RTPLAN ▶
RTSTRUCT and per-slice image references stay intact; **remove private
elements** drops all odd-group vendor tags (also inside sequences); and
**mark as de-identified** writes `PatientIdentityRemoved=YES` +
`DeidentificationMethod`. Output goes to a separate folder (files keep
their relative paths; the default is `<input>_anon`) or in place; files are
written via a temp file so an interrupted run never corrupts an original,
and pixel data is copied through byte-identical. The whole pipeline is
covered by `tests/anonymize.rs`: generate the synthetic study, anonymize
it, reload it, and assert identity is gone, the reference chains still
resolve, and the volume is unchanged.

**Built-in test-data generator.** *File ▶ 🧪 Generate test data…* writes a
complete synthetic RT study — CT water phantom with a spherical target and a
cord, matching RTSTRUCT contours, a Gaussian RTDOSE, a two-beam proton
RTPLAN, and optionally DX / RTIMAGE / REG / RTRECORD objects — into
`test_data/` next to the executable, then loads it. Dose peak, target shift,
whole-phantom shift, plan label and REG translation are adjustable in the
dialog, so a deliberately misaligned second study for registration testing is
one more click. Generation runs on a background thread with progress, and the
whole phantom is analytically known, which is what the integration tests
assert against (see [Synthetic test data](#synthetic-test-data)).

**RTPLAN.** Photon (`BeamSequence`) and ion/proton (`IonBeamSequence`) plans:
prescription, fractionation, and a per-beam table with radiation type, scan
mode, gantry/couch angles, energy range, meterset and control-point count.
Beam isocenters are marked in all three views.

**3D structure view.** The **3D A** / **3D B** toolbar buttons open a
floating window with a Slicer-style 3D surface rendering of the active
structure set. Surfaces are reconstructed from the RTSTRUCT contours on a
background thread (scanline rasterization into a binary volume, a
surface-nets mesher, Laplacian smoothing, area-weighted vertex normals —
`rayon`-parallel per ROI) and drawn in the ROI display colors with headlight
shading; EXTERNAL/body ROIs are rendered translucent so the internal anatomy
stays visible. Drag rotates, the wheel zooms, middle-drag pans, and a slider
controls global opacity. The meshes are cached per structure set, so
reopening the window is instant.

**Interactive segmentation (🖌 Paint · ◻ Erase · ✨ Grow).** MITK-style
manual and semi-automatic segmentation, implemented entirely in Rust and
CPU-side. The toolbar tools take over the left mouse button in the MPR
views:

* **🖌 Paint / ◻ Erase** — a spherical, spacing-aware **3D brush** (radius
  in mm, adjustable via the toolbar, `Shift+wheel` or `[` `]`) paints a
  voxel label mask in any of the three views; a **3D** toggle switches to a
  flat 2D circle confined to the displayed slice. Strokes are swept as
  capsules between pointer samples, so fast drags stay gap-free. `Alt`
  temporarily erases while painting, `Ctrl+Z` undoes stroke by stroke.
* **✨ Grow** — blazing-fast interactive **organ-wise** segmentation by
  geodesic fast marching (not a plain threshold): a Dijkstra front expands
  from the seed, and the cost of each step rises exponentially with the
  voxel's intensity deviation from robust seed statistics (median/MAD of
  the local neighborhood) **and** with the intensity jump of the crossing
  itself — organ boundaries, fat planes and edges act as barriers, so the
  organ under the cursor is suggested first instead of flooding all
  similar-intensity tissue. Press to seed, drag up/down to extend/shrink
  the geodesic reach with a **live yellow preview**; the front expands
  *incrementally* (drag up continues the same priority queue, drag down
  truncates the accepted prefix), so the preview never recomputes from
  scratch. Release commits — enclosed holes (vessels, calcifications) are
  filled slice-wise so the organ comes out solid — and `Esc` cancels.

Segmentations appear instantly in **all three MPR views** (crisp
nearest-neighbor colorwash) **and live in the 3D window**: every mask edit
re-meshes the segmentation on a background thread (bounding-box-cropped
surface nets with automatic striding for huge masks, same smoothing/normals
pipeline as the RTSTRUCT surfaces), so the 3D surface follows the brush in
essentially real time. The sidebar *Segmentations* section manages any
number of masks per dataset — visibility, display color, active selection,
volume in cm³, per-stroke undo, delete — and **→RS** converts a mask to
RTSTRUCT closed planar contours (marching squares per slice, stitched and
decimated), appending it to the active structure set so it renders like any
ROI and rides the existing DICOM export.

**Interaction** (shown in the status bar):

| Input | Action |
|---|---|
| Left click / drag | Move the linked crosshair (all views follow) |
| Mouse wheel | Scroll through slices |
| Ctrl + wheel / pinch | Zoom (anchored at the cursor) |
| Middle drag | Pan |
| Right drag | Window/level (x = width, y = center) |
| Double click | Reset zoom & pan |

Each viewport also carries two corner buttons: **⟲** resets that view's zoom,
pan **and slice** (back to the volume's central slice), and **⛶ / ❐**
maximizes the view to fill the whole window and restores the multi-view
layout again. The toolbar holds the **3D A / 3D B** buttons, the **⌖**
toggle and a global **⟲** that resets every view of both datasets at once.
**⌖** shows/hides the slice intersection lines (crosshair) — and while the
crosshair is hidden, left-click navigation is disabled entirely: clicking a
view no longer moves the other views' slices, and slices change only by
scrolling each view individually.

Common CT window presets (brain, subdural, stroke, head/neck soft tissue,
temporal bone, lungs, mediastinum, abdomen, liver, spine, bone, CT angio,
full range), anatomical
edge labels (L/R/A/P/S/I) derived from the actual patient orientation, and a
status bar with patient coordinates, voxel indices, HU and dose (Gy and % of
reference) at the crosshair.

**Dark / light appearance.** *View ▶ Appearance* switches between **🌙 Dark**,
**☀ Light** and **💻 System** (follows the OS setting and updates live when it
changes). The choice is remembered in `viewer_settings.txt` next
to the executable — a two-line text file, safe to edit or delete. The image
viewports themselves stay black in both themes, as in clinical viewers, so
grayscale windowing, the dose colorwash and the overlay annotations keep a
single calibrated appearance; the surrounding chrome and the hand-painted
accents follow the theme (unit tests assert the accent colors clear WCAG AA
contrast against both backgrounds).

## Performance

Everything is CPU-side Rust + GPU texture blitting via `wgpu`; the hot paths
are cached and only recomputed when their inputs change (slice, W/L, dose
settings, ROI visibility). Measured on the bundled synthetic study: full
study load ≈ 40 ms, orthogonal slice extraction ≈ 6 µs, dose-plane resampling
≈ 0.3 ms. Loading runs on a background thread with progress display, and
per-slice decode is parallelized across cores. Volumes are stored as `i16`
(HU), dose grids as `f32`.

## Building

Requires a Rust toolchain (<https://rustup.rs>). Then:

```
cargo build --release
```

Run with optional directory arguments — one study, or two to start straight
in comparison mode:

```
cargo run --release -- "D:\path\to\dataset_A"
cargo run --release -- "D:\path\to\dataset_A" "D:\path\to\dataset_B"
```

or start it empty and use *Add DICOM folder…* / the *File* menu. Windows, Linux
and macOS are supported; rendering uses `wgpu` (DX12/Vulkan/Metal, with
fallbacks).

## Example data

`example_data/` holds a small real patient study (137 MB) so the viewer can be
exercised on clinical data and not only on the synthetic phantom — two
breathing phases of a 4DCT, each with its own RT Structure Set:

```
example_data/
  lung_p1_4DCT_phase_000/   133 CT slices + 1-1.dcm (RTSTRUCT, 13 ROIs)
  lung_p1_4DCT_phase_050/   133 CT slices + 1-1.dcm (RTSTRUCT, 12 ROIs)
```

512 × 512, 0.977 mm in-plane, 3 mm slices, 396 mm of coverage; ROIs are cord,
both lungs, heart, esophagus, carina, lymph node, tumor and four implanted
gold fiducial markers (`_c00` / `_c50` suffix = breathing phase). Both series
share one Study Instance UID and one Frame of Reference, so they load as
inhale/exhale of the same study:

```
cargo run --release -- example_data/lung_p1_4DCT_phase_000 example_data/lung_p1_4DCT_phase_050
```

That is a ready-made comparison-mode and registration test case with real
respiratory motion: the tumor and the markers move visibly between the two
phases, and *Registration ▶ Deformable* has something anatomically real to
recover — unlike the synthetic phantom below, where the ground truth is known
but the deformation is analytic. All three screenshots at the top of this
README are made from this dataset. Equivalently, load the whole
`example_data/` folder as dataset A (both phases appear as two series of one
study) and right-click one phase ▶ *Copy series to dataset B* — same result
without a second command-line argument.

### Source and citation

The data is patient **P102** from the public **4D-Lung** collection on The
Cancer Imaging Archive (TCIA), a longitudinal 4D fan-beam CT / 4D cone-beam CT
dataset of 20 locally advanced NSCLC patients treated with chemoradiotherapy:

<https://www.cancerimagingarchive.net/collection/4d-lung/>

It is redistributed here under **CC BY 3.0**, the license of the original
collection. If you use it, cite the data and the associated publications:

> **Data.** Hugo, G. D., Weiss, E., Sleeman, W. C., Balik, S., Keall, P. J.,
> Lu, J., & Williamson, J. F. (2016). *Data from 4D Lung Imaging of NSCLC
> Patients* (Version 2) [Data set]. The Cancer Imaging Archive.
> <https://doi.org/10.7937/K9/TCIA.2016.ELN8YGLE>
>
> **Publication.** Hugo, G. D., Weiss, E., Sleeman, W. C., Balik, S., Keall,
> P. J., Lu, J., & Williamson, J. F. (2017). A longitudinal four-dimensional
> computed tomography and cone beam computed tomography dataset for
> image-guided radiation therapy research in lung cancer. *Medical Physics*,
> 44(2), 762–771. <https://doi.org/10.1002/mp.12059>
>
> **TCIA.** Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel,
> P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., & Prior,
> F. (2013). The Cancer Imaging Archive (TCIA): Maintaining and Operating a
> Public Information Repository. *Journal of Digital Imaging*, 26(6),
> 1045–1057. <https://doi.org/10.1007/s10278-013-9622-6>

### Anonymization

The TCIA data is already de-identified; the copy here was additionally
rewritten by `tools/anonymize_dicom.py` to minimal, readable identifiers —
patient `lung_p1`, and a UID tree that is easy to read in a debugger:

| | phase_000 | phase_050 |
|---|---|---|
| CT series | `1.2.3.4.5.10` | `1.2.3.4.5.20` |
| CT slices | `1.2.3.4.5.10.<InstanceNumber>` | `1.2.3.4.5.20.<InstanceNumber>` |
| RTSTRUCT series / instance | `1.2.3.4.5.11` / `.11.1` | `1.2.3.4.5.21` / `.21.1` |

with `1.2.3.4.5.1` as the shared Study Instance UID and `1.2.3.4.5.2` as the
shared Frame of Reference UID. Everything not needed to render the images and
contours — accession number, device manufacturer and model, software versions,
acquisition dates and private tags — was dropped; pixel data, geometry, ROI
names, colors, types and contour points are untouched, and every RTSTRUCT
image reference still resolves to a slice of its own series. The script is
pure standard library (no pydicom) and re-runnable:

```
python3 tools/anonymize_dicom.py example_data
```

## Synthetic test data

The viewer generates its own test study — no Python, no external tooling.
*File ▶ 🧪 Generate test data…* (also offered on the empty start screen)
opens a dialog that writes a complete, analytically known RT study into
`test_data/` **next to the executable** and loads it straight away:

* CT — 40 slices, 96 × 96, 2 mm isotropic; water cylinder (r = 70 mm),
  spherical target (r = 25 mm, HU 100) at the origin, cord (r = 8 mm, HU 40);
* RTSTRUCT — BODY (EXTERNAL), TARGET (PTV), CORD (ORGAN);
* RTDOSE — 3D Gaussian, 60 Gy at the isocenter, σ = 20 mm, 32-bit, 4 mm grid;
* RTPLAN — ion (proton) plan, 2 beams, 60 Gy / 30 fx;
* DX radiograph, RTIMAGE (DRR), REG spatial registration and an RT Ion Beams
  Treatment Record (optional).

The dialog exposes the dose peak, the target Y shift, a whole-phantom X/Y
shift, the plan label and the REG translation, so a second, deliberately
misaligned study for comparison mode and registration is a matter of
generating once more into another folder:

```
# rigid scenario: whole phantom translated (12, -9) mm
#   → generate into test_data_shifted with shift X = 12, shift Y = -9
# deformable scenario: same body, target displaced 15 mm
#   → generate into test_data_target15 with target Y shift = 15, peak = 66
cargo run --release -- test_data test_data_shifted
```

Then *Registration ▶ Rigid* should recover the (12, −9, 0) mm shift to within
a fraction of a millimeter, and *Registration ▶ Deformable* on the
target-displaced study warps it back onto dataset A.

## Tests

The integration tests generate the study through the same code path and
verify geometry round-trips, HU values, contour radii, trilinear dose values,
isodose radii and plan fields against the closed-form expectations, plus a
simulate → export → reload round trip. Nothing external is needed:

```
cargo test --release
```

## Structure

```
src/
  main.rs      entry point (eframe/wgpu window)
  app.rs       egui application: three-view layout, panels, interaction
  loader.rs    directory scan, classification, parallel volume loading,
               dataset merging (Add folder / tree copy-move)
  extras.rs    DX/CR/RTIMAGE planar images, REG spatial registrations,
               RTRECORD treatment records
  registration.rs  elastix-style rigid + B-spline registration (ASGD,
               multi-resolution, random sampling) in pure Rust
  simulate.rs  known-transform study generator for registration QA
  dicom_export.rs  DICOM writer (CT series, RTSTRUCT, RTDOSE, RTPLAN)
  gen_test_data.rs synthetic RT phantom study generator (CT/RTSTRUCT/RTPLAN/
               RTDOSE + DX/RTIMAGE/REG/RTRECORD), driven from the GUI
  anonymize.rs interactive DICOM anonymizer (Tools menu): tag scan with
               editable replacements, consistent UID regeneration,
               private-tag removal
  settings.rs  persisted preferences (theme) in a plain text file next to
               the executable
  volume.rs    3D volume, patient-space geometry, orthogonal slice extraction
  mesh3d.rs    contour ▶ surface reconstruction (scanline fill, surface nets,
               Laplacian smoothing) for the 3D structure windows
  rtstruct.rs  RT Structure Set parsing
  rtdose.rs    RT Dose parsing + trilinear patient-space sampling
  rtplan.rs    RT Plan / RT Ion Plan parsing
  render.rs    window/level, dose colorwash, marching-squares isodose,
               contour/plane intersection
  geometry.rs  minimal 3D vector math
tools/
  anonymize_dicom.py  stdlib-only DICOM anonymizer used on example_data
               (the interactive Tools ▶ Anonymize is its generalized
               Rust successor)
```

## Notes & limitations

The three views are extracted in the acquisition index space, which maps
directly onto axial/sagittal/coronal planes for standard axial acquisitions;
oblique acquisitions display consistently but the plane names are nominal
(edge labels always reflect the true patient directions). Enhanced
multi-frame image series are not yet supported (classic single-frame series
only). Non-uniform slice spacing is detected and reported as a warning, with
the median spacing used for display. The registration metric is
mean-squared-difference, appropriate for mono-modal (CT–CT) alignment;
mutual information for CT–MR is a natural extension. Deformable results are
intensity-driven: displacements inside large uniform regions are
interpolated from the B-spline grid rather than measured. This software is a
viewer for research and QA convenience — not a medical device, and not for
clinical decision-making.
