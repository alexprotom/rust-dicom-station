# Image viewing, datasets and interaction

![single dataset](screenshot.png)

*A lung 4DCT phase with its RT Structure Set. The crosshair sits in the tumor;
the axial view draws the native RTSTRUCT contours, sagittal/coronal show
reconstructed cross-sections of the same ROIs.*

## The start screen

With nothing loaded the window is two panels, **Input** and **Tools**, each a
wide button over a pair of narrow ones. The wide button is the one that
starts the work; the two below it are the shortcuts.

**Input**

* **Add DICOM folder** scans a folder into dataset A.
* **Restore last session** loads again what was open when the program was last
  closed. The sources of both datasets are remembered in `session_a` /
  `session_b` in the settings file as they are loaded, so an unclean exit
  loses nothing. If the folders it names have since moved or been deleted, it
  says so and forgets the session.
* **Load data from PACS** opens the local patient archive.

A button that cannot do its job is greyed rather than hidden - there is no
last session, or the archive is empty - so the screen keeps the same shape
and the answer to "where has it gone" is on the button itself.

**Tools**

* **Generate test data** writes a synthetic RT study to try the program on.
* **Anonymize DICOM folder** opens the anonymizer.
* **Downloaded models** opens the model manager.

## Loading and volume reconstruction

Data comes in either way round: a whole folder (*File > Add DICOM folder*, or
directory arguments on the command line) or an explicit handful of files
(*File > Add DICOM file(s)*, multi-select). Both start the same background
scan and both merge into the dataset the same way - a file selection is not a
separate mode with its own rules, it is a study that happens to be small.

The scan:

1. **Classification.** Every file in the tree is read header-only (no pixel
   data) in parallel and classified by SOP class / modality: image series
   (CT/MR/PT/…), RTSTRUCT, RTDOSE, RTPLAN, planar images (DX/CR/RTIMAGE), REG
   spatial registrations, RT treatment records. Unreadable or foreign files
   become warnings, never errors.
2. **Series grouping.** Image files are grouped by SeriesInstanceUID into the
   dataset tree; the largest series is reconstructed first (click another to
   switch).
3. **Volume reconstruction.** Slices are decoded in parallel (`rayon`) -
   compressed transfer syntaxes (JPEG lossless, RLE, …) via `dicom-rs`'s
   pure-Rust decoders - sorted by projection onto the true slice normal (cross
   product of the ImageOrientationPatient row/column vectors), checked for
   uniform spacing and consistent dimensions, and rescaled to HU with the
   per-file rescale slope/intercept. When that rescale lands on integers
   (slope 1, an integer intercept - every CT) the pixels are converted
   straight into the volume's `i16`; only a fractional rescale (PET counts,
   say) goes through a floating-point pass and a rounding. The result is one
   `i16` volume with full patient-space geometry (origin at the center of
   voxel (0,0,0), unit direction vectors for the three axes, spacing in mm).

Non-uniform slice spacing is reported as a warning (the median spacing is used
for display) and duplicate slice positions are collapsed. Enhanced multi-frame
image series are not yet supported (classic single-frame only). RT objects
found in the folder are parsed alongside and attached to the study - see
[rt-objects.md](rt-objects.md).

### Datasets with no volume

Not everything worth opening reconstructs into slices, and a viewer that
insists otherwise is a viewer you cannot use to look at a portal image. So
**a dataset without an image volume is a normal dataset here**, not a failed
load. Three cases arrive at it:

* the selection holds only RT images, DX/CR radiographs or other projection
  images - nothing with slice positions to stack;
* the selection holds only RT objects - a structure set, a plan, a dose grid,
  a registration, a treatment record;
* an image file carries no `ImagePositionPatient` at all. That is judged per
  *series*, not per file: a series where nothing is positioned cannot be
  reconstructed and its files are opened as single images, while a series
  where one slice happens to lack the tag is still a series. Before, such
  files were dropped silently.

Such a dataset appears in the tree under its patient and study exactly like
any other, and everything it holds is usable: planar images open in their
viewers (the *Planar images* section opens itself, since for these datasets it
is the content rather than a footnote), structure sets render in the 3D
window, plans and dose objects show their tables, and any of it can be
renamed, copied to the other dataset or exported. What is held back is only
what needs voxels: the MPR views say so in place of three black panes, and the
segmentation tools, the four engines, registration, propagation, combination,
comparison and the DRR are disabled until there is something to run them on.

Adding an image series afterwards completes the dataset. *File ▶ Add DICOM
folder…* into the same slot merges the images in and the views switch to
them - which is the ordinary way to open a structure set first and its CT
second, and have the contours land on the right images.

## The rows and their panes

The main area is **one row per dataset** - a second appears in *Comparison
mode* (View menu) - and each row shows up to **four panes**, chosen under
*Settings ▸ View layout*: the **axial**, **sagittal** and **coronal** planes,
and the **3D** surface scene. The two rows are chosen separately, so the
planning CT can sit above one plane of the repeat scan, or a single large
axial above the same plane of the other dataset.

The panes of a row split its width evenly, so a row of two is two large
images rather than two and a gap, and a row of one is one image across the
window. A row always shows something: the last tick of a row cannot be
taken off, and a tick greys out once the row holds as many panes as it
carries - every kind there is, so a row can hold the three planes and the
surface scene at once. The **◀ ▶** buttons beside each tick move that pane
one place left or right in the row, and the list is drawn in the row's own
order, so it reads left to right the way the row does; a newly ticked pane
joins at the right. **⟲ Reset** beside a row's heading puts that row back to
the standard three - axial, sagittal, coronal, in that order - in one click,
and greys out when the row is already that. The choice is remembered between
runs (`view_row_a` /
`view_row_b` in the settings file, which can also be edited by hand). Both
that submenu and the **View** menu are sets of switches rather than lists of
actions, so ticking one leaves the menu open and a whole layout can be put
together in one visit; they close on a click outside them or on their own
title again.

The MPR panes carry **linked crosshairs**: clicking a point in any of them
moves the others to that patient-space position. Planes are extracted in
acquisition index space; oblique acquisitions display consistently but their
plane names are nominal, and the anatomical edge labels (L/R/A/P/S/I) always
reflect the true patient directions from the direction cosines.

The panes tile the central area edge to edge, each with its own **slice
scrubber** drawn over its bottom edge; the plane and dataset name in the
top-left corner is white in every pane, the edge labels keep their colour.
The corner buttons (named on hover), right to left: **⛶ / ⊞** maximizes the
pane and restores the layout, **⟲** resets the pane's zoom and pan and
re-centers the crosshair in the volume, **✋** hands the left button the
image - drag it and the image moves instead of the crosshair, as a middle
drag always does - and **➕ / ➖** zoom a step about the middle of the pane.
The hand is one switch for every pane of both datasets. The toolbar holds a
global **⟲** (the same reset for every pane of both datasets), the **⌖**
crosshair toggle (while hidden,
left-click navigation is off and slices change only by scrolling), the
**Sync** toggle beside it (shown whenever two datasets are loaded - see
below), the **3D A / 3D B** buttons and the segmentation tools.

**The 3D scene** can live in a row or in a window. Ticking **3D** for a row
draws it there, and the pane is furnished like any other: **3D** (with the
dataset's letter in comparison mode) in the top-left corner, and the same
corner buttons in the same places - **⛶ / ⊞**, **⟲** (the camera back to
its default angle, fit zoom and no offset), **✋** (a left drag moves the
scene instead of turning it), **➕ / ➖**, and the **▶4D** transport with
**Prepare** beside it where the dataset has a 4D group. What only a scene
has sits under them, wrapping onto another line where a pane is narrow:
**Opacity** with its percentage, **Structures**, and **Dose** / **Isodose**
where the dataset carries a dose. **Structures** lays the per-structure
opacity panel down the pane's right-hand edge, the same panel and the same
sliders the window shows. While a row carries the scene, that dataset's
**3D** button leaves the toolbar - there is no window to open - and it
comes back when the tick does.

**The 3D window** (3D A / 3D B) meshes the active structure set and keeps
up with it: a structure moved or redrawn in the Structure editor is
re-meshed in the background while the scene stays on screen, and the
editor's drawn axis is shown in it. Its camera buttons - **➕ / ➖**, **✋**
and **⟲** - sit in the window's top-right corner, where a pane keeps them.
The *Opacity* slider is the whole scene's; the **Structures** toggle opens
a panel with one slider per structure on top of that, so the target can
fade while a chamber volume stays solid (*All 100 %* clears them). With a
dose in the dataset, **Dose on the surface** colours every surface by the
dose that lands on it, on the
isodose scale, and **Isodose surfaces** adds the active dose as translucent
shells at the isodose lines switched on in the Dose display, in their
colours and relative to the same reference dose, with their own opacity
slider; they follow a change of dose, reference or lines by themselves.

**Window/level.** Right-drag on any view adjusts interactively
(x = width, y = center); the toolbar offers numeric fields and the common CT
presets: brain, subdural, stroke, head/neck soft tissue, temporal bone, lungs,
mediastinum, abdomen, liver, spine, bone, CT angio, full range. The list
shows each preset's center and width; the closed list carries only the chosen
name, and any other window - a drag or the full range - leaves it nameless.
Window/level is shared between datasets A and B.

**Every tool has its own window.** The archive, the model manager, the DRR,
the 3D scenes, the segmentation, motion and DVH tools, the export and
anonymizer dialogs - none of them float inside the main window. Each opens as
a window of the operating system in its own right, with its own title bar and
task-bar entry, to be dragged onto a second or third monitor, resized or
maximized there, and left open beside the images while the main window keeps
all six viewports. Any number can be open at once, on any mix of screens, and
each one reopens at the size and place it was last left - on the monitor it
was left on. Closing a window closes that tool alone. Each has a **📌 Keep
on top** switch in its top right corner: on, the window stays above the
main window (and everything else) while it is open. It is off whenever a
window opens and is not remembered between runs.

Every window of the program is titled the same way: **Rust DICOM Station:**
followed by what the window is - *Viewer* for the main one, then *PACS -
patient archive*, *Downloaded models*, *DRR - dataset A*, and so on.

**Status bar.** Patient coordinates, voxel indices, HU and dose (Gy and % of
the reference dose) at the crosshair; in comparison mode both datasets report
the full set side by side, each at its own crosshair. Hover the **?** at the
right end to read the active tool's mouse bindings.

**The two panels.** The left one is the data tree, the right one the modules.
Each hides and shows from the *View* menu (*Data tree*, *Modules*), from a
shortcut (**F9**, **F10**) and from the arrow on its edge of the window;
dragging a panel's inner edge past the minimum does the same, and the arrow
brings it back.

The *Modules* menu chooses the right panel's sections, in the order they
appear: **Image information** (what the displayed series is - see below),
**Playback** (the ▶ buttons on the viewports - see below), **Image
registration**, **Image simulation**, the **Structure editor** (insert, edit
and combine structures - [contours.md](contours.md)), the **Structure auto
tools** (the body contour and the three segmentation engines -
[segmentation.md](segmentation.md)), **Structure propagation** and **Dose
estimation** (the dose metrics table - [dvh.md](dvh.md)). The image
information, the playback, the editor, the auto tools and the dose
estimation start switched on, the other three off; every choice is
remembered between runs, and with all eight off there is no right panel at
all. Every section starts folded; which ones were
unfolded is remembered too, and *Restore the last session* unfolds them
again. The drawing tools are not in the panel: the toolbar's **✏
Draw structure** button unfolds them on the toolbar. The *Tools* menu keeps
the windows: **◑ Structure comparison**, **📋 Structure details** (one row
per structure, with a Dice column against a reference of your choice) and
**📈 Structure motion**, then transfer, DVH, DRR, the archive, the models
and the anonymizer.

## Image information

The first section of the right panel says what the displayed series actually
is, read back out of its own DICOM headers rather than out of the
reconstruction:

* **Series** - patient, study, modality, series number, body part, protocol,
  patient position and when it was acquired.
* **Sampling** - the matrix, the voxel spacing, the field of view, how many
  slices there are (and how many files they came from), the slice thickness,
  the gap or overlap between slices, whether the slice positions are evenly
  spaced, and whether the header's own *SpacingBetweenSlices* agrees with
  them.
* **Geometry** - origin, orientation, gantry tilt, frame of reference, series
  and study UID.
* **Acquisition** - scanner, station, software, kV, mA, mAs, CTDIvol,
  reconstruction kernel and scan options.
* **Pixels** - value range, rescale slope and intercept, units, photometric
  interpretation and bits stored.

Anything worth a second look is repeated at the top of the section in amber
with the reason: slices that do not touch (everything between them is
interpolated, and a structure volume or a DVH is only as good as that
interpolation), slices that are unevenly spaced, a tilted gantry, an
anisotropic in-plane spacing, oblique axes, a missing frame of reference,
files that are in the series but not in the volume. A regular study says
*Nothing unusual*.

*Compare* puts the two datasets side by side and lists only what they
disagree about - the check to make before registering them, contouring
across them or carrying a dose from one to the other. *Copy* puts the whole
report on the clipboard; *Read again* re-reads the headers after the files
on disk have changed.

## Playing through slices and phases

Every viewport carries two play buttons in its top-right corner, beside the
reset and maximize ones, and each appears only where there is something to
play:

* **▶3D** runs through the **slices** of that view, the way one scrolls a
  stack by hand but without the hand. It appears on any view with more than
  one slice, and it moves that view only, exactly as the wheel and the
  scrubber under the pane do.
* **▶4D** runs the whole dataset through the **phases** of its 4D group. It
  appears only when the displayed series belongs to a group that still has
  at least two phases. All three views change together because the image
  itself changes, the selection walks down the group in the data tree, and
  the structure set, the segmentation series and the dose of each phase come
  with it.

Either button turns into **⏸** while it runs, and one run is in flight at a
time. The 3D scene has a **▶4D** of its own, in its window and on its pane's
bar, that starts the same run, so the surfaces and the isodose shells
breathe with the views, and the Dose estimation module's *Dynamic* log has
a third ([dvh.md](dvh.md#the-dose-estimation-module)) that walks the
structures back through the moves they were given.

**With Sync on, both datasets play.** **▶3D** takes the paired pane through
its own stack by the same rule the wheel follows - the slice position in
patient coordinates, through the registration where there is one - so the
two rows stay on the same anatomy rather than on the same slice number.
**▶4D** walks the other dataset's group beside this one's, which is why the
first press reads *both* groups into memory and the budget in the Playback
module has to cover them. Two groups of the same length step phase for
phase; groups of different lengths are walked proportionally, so a tenth of
one breathing cycle meets a tenth of the other. A dataset that is not
showing a 4D group is simply left where it is.

### Playback

The module is where the settings live.

* **Slices** - which view the module's own transport runs (axial, sagittal
  or coronal), ⏮ ▶ ⏭ and a scrubber with the slice number.
* **Phases** - the group being played, the same transport, and a scrubber
  showing the phase label and its place in the group.
* **Speed** - separately for slices, phases and the steps of the Dose
  estimation module's *Dynamic* log: a stack of 200 slices wants to move
  faster than a ten-phase breathing cycle, and a log step slower still,
  because every step brings its own dose numbers to read.
* **At the end** - *Loop* starts again from the beginning, *Bounce* turns
  around and runs back, *Once* stops. For a breathing cycle *Bounce* is the
  honest one: the jump from the last phase to the first is a jump the
  patient never made.
* **Slice step** - show every n-th slice, so a long stack can be watched in
  one pass. Phases always step one at a time.
* **A phase change carries with it** - the structure set drawn on the new
  phase, the segmentation series that belongs to it, and the dose named
  after it. The first two are matched by the image series the object
  references. A dose carries no such reference, so the only thing that can
  tie one to a phase is what it is called: a dose whose label contains the
  phase's own token (`50%`, `T3`) as a whole word follows the phase, and a
  study whose doses are not named after its phases simply keeps the dose it
  has.
* **Phases in memory** - a phase switch means a different image series, and
  reading one off the disk takes long enough that playing straight from
  disk would be a slideshow. So the first press of ▶4D reads every phase of
  the group into memory with a progress bar and then plays from there;
  *Read phases* does the same without playing, and *Free* gives the memory
  back. **Budget** refuses a group whose phases would need more than that,
  with the estimate shown before anything is read: ten phases of a large CT
  run to about a gigabyte.
* **Save a run** - record a run as a picture sequence. **⏺ Record** arms the
  recorder: it asks where the file goes and then waits. The next run started
  with ▶ is the one taken, and the pane is whichever that ▶ belongs to - ▶3D
  on the sagittal pane of dataset B records that pane, ▶4D on a 3D pane
  records the surfaces. One picture per played frame, of the pane's own
  pixels, so contours, dose wash, annotations and 3D surfaces are all in it;
  the run's clock waits for each picture, so nothing the run played is
  missing from the file. It ends by itself when the run comes back to the
  frame it started on - one full cycle, a loop through the range or a bounce
  out and back - and otherwise when the run stops, when the frame limit is
  reached, or when **⏹ Stop and save** is pressed. Written as an animated
  **GIF** at the run's own rate (pure Rust, no other program involved, and
  therefore 256 colours a frame), or as numbered **PNG frames** in a folder,
  which is full colour and what a video tool wants as input. **At most n
  frames** is the backstop, because every frame is held in memory until the
  recording finishes.

Stepping between the phases of the group on display - with ⏮ ⏭, with the
scrubber, or by clicking a phase in the data tree - keeps the view where it
is: the crosshair, the zoom, the pan, the slice of every view and the active
registration all stay, because two phases are the same patient a moment
apart and a view that jumps back to the middle slice hides the motion one is
looking for. Picking any other series is an ordinary switch and still starts
the dataset afresh.

In the **3D structures** window, each phase brings its own structure set and
so its own surfaces. **Prepare phases** meshes them all up front, which is
what makes the cine smooth; without it the first pass through the group
waits for the mesher at every phase, and the second pass is smooth because
the meshes are then cached. The isodose shells cache the same way as they
are seen.

## Interaction reference

| Input | Action |
|---|---|
| Left click / drag | Move the linked crosshair (all views follow) |
| Mouse wheel | Scroll through slices |
| Ctrl + wheel / pinch | Zoom (anchored at the cursor) |
| Middle drag | Pan |
| Right drag | Window/level (x = width, y = center) |

With a segmentation tool active the left button paints instead of navigating -
see [segmentation.md](segmentation.md) - and with a contour tool it draws into
the RT structure being edited ([contours.md](contours.md)): click by click for
the polygon and spline tools (right-click, double-click or Enter closes,
Esc cancels), one drag for freehand and nudge, Ctrl-click to pick the
structure under the pointer. The full bindings of whichever tool is in hand
are under the status bar's *?*.

## Datasets and the patient ▶ study ▶ series tree

The two viewer slots, **dataset A** and **dataset B**, each hold any number of
patients, studies and series from any number of folders. *File ▶ Add DICOM
folder to A/B…* merges a scanned folder into the slot without unloading what
is there; duplicates (by UID) are skipped and reported. *Tools ▶ 🏥 PACS -
patient archive…* fills a slot the same way from the application's own store
of studies ([pacs.md](pacs.md)) - an archived study folder is ordinary DICOM.

The left panel shows each dataset as a full DICOM hierarchy:

```
Dataset A
 └ Doe John (P1)                     patient - PatientName / PatientID
    └ Study 20260827 - Planning      study - StudyInstanceUID, date, description
       ├ CT (2)                      modality
       │   ├ chest (120 sl.) [RTS][RTD][RTP]   image series, and what is on it
       │   └ abdomen (90 sl.)
       ├ MR (1)
       ├ RT structures (1)           how many sets
       │   └ Approved (9/12) ▶ CT chest    ticked of all, in the open set
       ├ Segmentations (1)
       │   └ TotalSeg (8/8) ▶ CT chest
       ├ Dose (1)
       │   └ Plan dose ▶ plan IMRT
       └ Plans (1)
           └ IMRT ▶ Approved
 Dose display · Planar images · Spatial registrations · Records · Warnings
```

The modality level (CT / MR / US / PT …) is one DICOM implies but does not
store as a node; it is grouped from the series' Modality in first-seen order.
Everything with a StudyInstanceUID - image series, RT structure sets,
segmentation series, dose grids and plans - sits inside its study; an RT
object whose StudyInstanceUID is blank or names an unloaded study goes under
the study of the image series it references, failing that under the first
study. Planar images, spatial registrations and treatment records have no
study and sit below the tree, as does **Dose display** - colorwash, isodose
ladder, opacity, threshold - one setting shared by both datasets, shown once.

Structure sets, segmentation series, **dose grids and plans** all use the
same row - a name, and nothing in front of it. The views draw one of each
kind at a time, so selection and visibility are one thing: clicking a row
makes it the one on display, clicking the row that is already displayed hides
that kind from the views (the row stays selected, drawn weak, and the list,
the drawing tools and any 3D scene go on working on it). For a dose that
means the colorwash and the isodose lines; for a plan, its isocenters.

**Badges on the image series.** A series row carries a small `RTS`, `SEG`,
`RTD` or `RTP` frame for each of the four selections that lands on it: the
structure set now active was drawn on this series, the segmentation series
belongs to it, the active dose was computed on it (through its plan and that
plan's structure set), the active plan was made on it. Selecting another set,
series, dose or plan moves the badge to whichever image series that one
points at, so the tree says at a glance what the current selection is
attached to. Nothing is claimed where the files do not say: a dose whose plan
was not loaded carries no badge.

**What the counts mean.** *RT structures (n)* and *Segmentations (n)* count
the **sets and series** the study node holds, not their contents. The
contents are counted on the row of the set or series they belong to: the one
whose list is open reads `(ticked/all)` - 9 of 12 structures ticked, 8 of 8
segments - and the rest simply say how many they hold, because tick boxes
belong to the open one and there is nothing to count on the others.

**Series that share a name** get `#0`, `#1`, ... after the description, in
tree order. Two acquisitions both called *CT 4DCT* are two different series,
and the tree is where one is picked.

The displayed series is marked; clicking another loads it. Long names,
descriptions and IDs wrap, so the panel can be dragged narrow. The reference
chain is shown as links: each structure set and segmentation series shows the
image series it is drawn on, each dose the plan it was computed for
(ReferencedRTPlanSequence), each plan the structure set it was created on
(ReferencedStructureSetSequence).

The dataset's own name is a heading rather than a node: the patients sit at
the same level as it, since a tree that can only hold two datasets does not
need a level for choosing between them.

**Right-clicking** a patient, study or series opens a context menu to
**rename**, **copy**, **move** or **remove** it. Copy/move transfer the
selection into the other dataset (A ▶ B or B ▶ A), merging with what is there
and switching comparison mode on; move and remove then delete it from its
source. A series carries exactly its DICOM reference chain - the structure
sets drawn on it, the plans made on those, the doses computed for those
plans - and study and patient selections also take the RT objects of their
studies. Right-clicking a dataset header offers *Clear dataset*.

## Structures and segmentations in the tree

Below the image series, each dataset lists its **RT structures** and
**Segmentations** as series nodes - one per RT structure set or DICOM
Segmentation series - each showing the image series it is drawn on
(`▶ CT chest`; `▶ (any image of this frame)` for a segmentation series tied
to no image series; `▶ (image series not loaded)` when the referenced series
is not in the dataset). Clicking a node makes it active and lists
its items **under that row**, not at the end of the list - with ten phases of
a 4D group in the node, the buttons that act on a set belong beside the set
they act on. The **+** on the *RT structures* / *Segmentations* heading
creates an empty structure set or segmentation series bound to the displayed
image series. **Right-clicking a series node** offers:

* *🔗 Connect to image series ▶* - re-point the series at any image series of
  the dataset (● marks the current one); contours are in patient coordinates
  and simply follow, a segmentation series is resampled onto the new lattice
  when next displayed. A segmentation series can also be tied to *no image
  series*: it then shows on every image of its frame of reference, every
  phase of a 4D study say, resampled onto whichever is displayed.
* *Copy / Move series to dataset A/B*.
* *💾 Export as DICOM SEG…* (segmentation series only) - writes this one series
  as a single SEG file.
* *🗑 Remove this RT structure set / segmentation series*.
* *✏ Rename series…*.

The small colour square in front of every structure and segment opens a
colour picker; the colour of an RT structure is kept when the set is
exported. Each item's **check box is both its visibility and its selection**, so *All*
/ *None* tick everything or nothing and the actions act on whatever is
ticked. **Shift-click** a check box to tick - or untick - the range from the
last one you clicked: the span takes the clicked row's new value.

One row carries the lot, the same for both kinds: **All · None · Copy to ·
Move to · 🗑 · *n* selected**, plus **💾** for segmentations, which exports
the ticked segments as their own SEG file. *Copy to* and *Move to* open the
destination submenu described below. The buttons grey out when nothing is
ticked. Ctrl+Z undoes the last stroke, *Copy to ▶ an RT structure set* is
what →RS did, and **🗑** deletes the ticked rows. Clicking an item's **name**
selects it - the segmentation the voxel tools paint, the structure the
contour tools and the Structure editor work on - independently of whether
it is shown; new items come from the **+** after the selected name on the
toolbar's draw row and, for structures, from the editor's *Insert structure*
section.

**Right-clicking a structure or segment** offers the same set for one row or
the ticked group:

* *Copy … to ▶* / *Move … to ▶* - a submenu of every structure set and
  segmentation series in **both** datasets, plus *➕ a new RT structure set* /
  *➕ a new segmentation series*, and, per 4D group, *⏱ each phase of <group>*
  as a segmentation series per phase or into each phase's own RT structure
  set. That copies the structure as it is, in patient coordinates, onto
  every phase (the phases are loaded for their lattices in the background);
  a structure that should follow the motion goes through the propagation
  module instead. A ticked row acts on all ticked rows at once; an unticked
  row acts alone.
* *🗑 Remove …* - the same single-or-selected rule.
* *💾 Export … as DICOM SEG…* (segments only) - writes the chosen segments as a
  SEG series of their own: same lattice, same referenced image series, a fresh
  SOP Instance UID, only those segments; the file reloads as an ordinary
  segmentation series.
* *✏ Rename …* - always the row you clicked, never the whole selection.

Crossing between the two kinds converts on transfer: a structure moved into a
segmentation series is rasterized onto its lattice (even-odd fill), a segment
moved into a structure set becomes closed planar contours (marching squares),
and a segment moved between different lattices is resampled. Anything that
cannot cross - a contour outside the destination volume, a mask that does not
overlap it - lands in the dataset's *Warnings* section.

## Renaming

Everything the tree names - patients, studies, image series, RT structure
sets, segmentation series, structures and segments, dose grids, plans, planar
images, spatial registrations and treatment records - can be renamed from its
right-click menu. The dialog is a single text field - Enter applies, Esc
cancels, empty names are rejected - and names the DICOM attribute it writes.

A patient and a study are *groupings* rather than objects, so renaming one
writes `PatientName` / `StudyDescription` into **every** series filed under
it; everything else writes the one attribute it shows: `SeriesDescription`,
`StructureSetLabel`, `ROIName`, `SegmentLabel`, `RTPlanLabel`, and the labels
of the remaining objects. Renames are in-memory: they change what the tree,
the overlays and the 3D view call things and what a DICOM export writes; the
files a study was loaded from are never modified.

## Comparison mode

![comparison mode](screenshot_comparison.png)

*Two opposite breathing phases of the same 4DCT as datasets A and B, each with
its phase-specific structure set; the synced crosshair pins every pane to
the same patient-space point inside the tumor.*

Load a second dataset (menu, tree copy/move, or two directories on the command
line) and the window splits into two rows - dataset A on top, dataset B below,
each showing whatever *Settings ▸ View layout* gives it. Each dataset keeps
its own structures, dose and plan panels in the sidebar; window/level and dose
display are shared.

**Sync** (the toolbar button, or *View ▸ Sync the two datasets*; both appear
whenever two datasets are loaded - it carries far more than the crosshair,
so it no longer goes away with it) keeps the two rows showing the same
thing. It carries six things across: the crosshair, through **patient
coordinates** - with a registration active, through the recovered transform
instead, see [registration.md](registration.md) - the slice that follows it,
a slice **scrolled or scrubbed** in a pane (through the same patient
coordinates, so the paired pane lands on whatever slice of the other volume
lies there, and neither crosshair moves), the **zoom** and **pan** of a
pane onto the other dataset's pane of the same plane, and both **players**
(see below). Zoom is screen pixels
per millimetre and pan is millimetres off the image centre, so copying them
puts the two rows at the same scale and the same offset whatever the two
matrices are; what cannot be carried across two
unrelated images is not pretended. Window/level is shared by both datasets
either way. Off, each dataset is navigated on its own.

With the bundled data: load `example_data/`, and both 4DCT phases appear as
two series of one study. Right-click *CT 4DCT_phase_050* ▶ *Copy series to
dataset B* - the phase moves into the lower row with its own phase-specific
RTSTRUCT and comparison mode switches on. Click the tumor in any pane: every
pane jumps to that point, and the rows show the respiratory differences.

## Planar images (DX / CR / RTIMAGE)

Digital radiographs and RT images (portal/setup images) in the study folder -
plus any DRR added from the DRR window with *➕ Add to dataset A/B* (see
[drr.md](drr.md)) - are listed in the sidebar and open in floating viewer
windows with their own window/level (DICOM default at open; auto, manual, or
right-drag like the CT views), correct physical aspect ratio (imager /
image-plane pixel spacing), MONOCHROME1 inversion, and metadata - body part,
view and kVp for DX; machine, gantry angle, SAD and SID for RTIMAGE.

Any image that carries no slice position lands here, whatever its modality -
that is what makes *File > Add DICOM file(s)* on a single RT image, an
unpositioned secondary capture or a stray slice give you something to look at.
The section is closed by default when there is a volume beside it and open
when there is not. Multi-frame images are the one exception: they are reported
as a warning rather than loaded.

## Appearance

*View > Appearance* switches between **🌙 Dark**, **☀ Light** and **💻 System**
(follows the OS setting and updates live). The choice is remembered in
`viewer_settings.txt` (`%LOCALAPPDATA%\RustDICOMStation` on Windows,
`~/.config/RustDICOMStation` on Linux), a tiny `key = value` text file, safe
to edit or delete. The image viewports stay black in both themes so
windowing, the dose colorwash and the overlays keep one calibrated
appearance; unit tests assert the accent colors clear WCAG AA contrast on
both backgrounds.

## Graphics backend

The viewer draws - and, with the GPU feature, runs the segmentation networks -
through `wgpu`, which speaks Vulkan, Direct3D 12, Metal or OpenGL depending on
the machine. Normally there is nothing to think about. The exception is real
and was the reason this section exists: **some Windows machines advertise a
Vulkan driver that cannot actually create a device.** `wgpu` prefers Vulkan,
finds the broken one, and the program dies before drawing anything - on a
machine where nothing else is wrong. The only escape used to be knowing to
type

```powershell
$env:WGPU_BACKEND = "dx12"
```

before starting it, which is not a thing to ask of a physicist in a clinic.

Three things now decide which backend is used, in this order of authority:

1. **`WGPU_BACKEND`**, if set. It stays the escape hatch and it still wins -
   someone who set it is debugging something.
2. **`graphics_backend`** in the settings, which the installer writes from the
   page it asks on and *Settings > Graphics backend* changes afterwards. Accepted
   values: `auto`, `vulkan`, `dx12`, `metal`, `opengl`.
3. Failing both, whatever `wgpu` picks on its own.

And whichever is chosen, **the program falls back by itself when it does not
work.** The window is not opened once but attempted: the preferred backend
first, then Direct3D 12, Vulkan and OpenGL, ending at whatever `wgpu` would
have chosen. A backend that fails - by returning an error, or by panicking
somewhere inside the driver, which is the usual shape of this failure - costs
one line on standard error instead of the program:

```
rust-dicom-station: Vulkan failed: …
rust-dicom-station: Vulkan did not work, trying DirectX 12…
```

So on a machine with a broken Vulkan driver the viewer now starts unaided. The
setting only saves it the first failed attempt - worth having, because the
attempt costs a second or two and prints a line that looks alarming.

*Settings > Graphics backend* lists the backends this platform could have (no
Direct3D outside Windows, no Metal outside macOS, and on macOS nothing but
Metal - Apple deprecated OpenGL and never shipped Vulkan, and `wgpu` reaches
neither there without a translation layer this program does not link), each
with a one-line hint,
and remembers the choice. Under the list it names the backend the program is
actually drawing with at that moment, which after a fallback is not always the
one that was asked for, and says that a change takes effect at the next start:
the backend is read once, before the window exists.

### Where the setting is read from

Two files, in increasing order of authority:

* `viewer-defaults.txt` **beside the executable**, written by the installer.
  A machine-wide installation is made by an administrator whose
  `%LOCALAPPDATA%` is not the one the viewer will run under, so this is the
  only place an installer-time answer can reach every user of the machine.
  Every key in it is a default.
* `viewer_settings.txt` in the per-user config folder
  (`%LOCALAPPDATA%\RustDICOMStation`, `~/.config/RustDICOMStation`), which is
  read afterwards and wins - key by key, so a setting the user has never
  touched keeps the machine-wide default.

Both are plain `key = value` text, safe to edit or delete. An unreadable value
leaves the default rather than failing to start: these files are edited by
hand and by an installer, and a typo in one must not cost someone their
program.

### Note on the inference backend

The program creates two independent `wgpu` instances: `eframe` draws the
interface with one, and `burn` runs the networks on another. The first takes
its backends as a typed argument; the second is several layers down inside
`cubecl` and takes them only from the environment. So the chosen backend is
also exported as `WGPU_BACKEND` for this process - once, at the very top of
`main` before any thread exists, which is both the documented contract for
writing the environment and exactly the workaround that was already known to
work. A value the user set themselves is never overwritten.
