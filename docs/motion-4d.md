# 4D motion analysis and ITV generation

The 4D workflow reproduces, inside the viewer, the pipeline of ITV-based
motion studies (e.g. upright-vs-supine STAR target evaluation): per-phase
registration, target propagation, centroid motion metrics, target-reference
drift and correlation, and ITV volumes.

## 4D groups (`src/fourd.rs`)

A 4DCT arrives as one image series per respiratory phase, usually with an
average and sometimes MIP/MinIP reconstructions beside them; DICOM stores
no node for the acquisition they form, so the viewer reconstructs one:

- Series are bucketed by (study, modality). A series whose description
  carries a number directly before a `%` (e.g. `Thorax 4D 30%`, `CT 0 %
  Ex`) or `phase` + number (`4DCT_phase_000`, `Phase 3`) is a **phase**;
  the description minus the number (the *template*) tells two 4D sets in
  one study apart - thin- and thick-slice reconstructions become two
  groups - and is the group's name stem (`4DCT - Thorax 4D (10 phases)`).
- Series with a `TemporalPositionIdentifier` but no percent group by
  identical description and order by that identifier (`t1`, `t2`, …).
- `AVG`/`average`/`mean`, `MIP` and `MinIP` in the description mark the
  reconstructions; they attach to the bucket's first group.
- A group needs **at least three phases** - two series with "50%" in the
  name are more likely a coincidence than an acquisition.

Detection is a heuristic, so everything can be corrected from the data
tree: right-click a series ▸ *4D group* to add it to a group or start a
new one; a group member to reorder it, change its role or remove it; the
group node to rename it, dissolve it or re-run detection. Hand-edited
groups are marked *custom* and survive re-detection (`fourd::refresh`, run
whenever the series list changes); a dissolved auto-detected group leaves
a hidden tombstone that only the explicit *Re-detect 4D groups* clears.
Members reference series by **SeriesInstanceUID**, so renames, removals
and copies never corrupt a group.

A group can also be **built from a selection** when detection has nothing
to go on - ten series called *CT* with no percent, no `phase` and no
temporal index. Ctrl-click ticks series in the tree; right-click a ticked
row ▸ *4D group* ▸ *New 4D group from the N ticked series*, or right-click
the modality node for every series under it (`fourd::group_from`). The
phases are ordered by percent when every phase has one, else by
TemporalPositionIdentifier, else by series number, else in tick order, and
a phase with no label of its own is called `t<n>` by position;
reconstructions the descriptions name (AVG, MIP, MinIP) go last. The series
taken leave any group they were in, and a group emptied that way is
dropped. *Add the N ticked series to <group>* appends them as phases.

Every engine of the *Structure auto tools* module can **run on every phase
of a group** in one go ([auto-segmentation.md](auto-segmentation.md#running-on-every-phase-of-a-4d-group)).

In the tree a group renders as a `🎞` node inside its study - phases in
temporal order, then the reconstructions; grouped series leave their
modality node so each series has one place.

### Watching a group move

Any viewport showing a series of a group carries a **▶4D** button, and the
*Playback* module carries the same transport with a phase scrubber; both are
described in [viewer.md](viewer.md#playing-through-slices-and-phases). The
short version: the first press reads every phase into memory (a group whose
phases would exceed the module's budget is refused rather than read), and
then the workspace runs through the phases with the structure set,
segmentation series and dose of each one, the tree selection walking down
the group as it goes. Stepping between the phases of the group on display
keeps the crosshair, the zoom, the pan and the registration where they are,
which is what makes two phases comparable by eye. The 3D window plays the
same run and can mesh every phase up front so the surfaces keep up.

Playing is for looking; the pipeline below is for measuring.

## The pipeline (`src/app/motion_win.rs`)

*Tools ▸ 📈 Structure motion* (the workspace is chosen on the window's
**Workspace A / B** row), or right-click a 4D group ▸
*Motion / ITV analysis…*. One run:

1. **Reference phase** - chosen in the dialog (default: the 0 % phase).
   Targets are defined on it: contours rasterized onto its lattice,
   segmentations from another lattice resampled onto it.

   The **Targets** list groups structures by name across the phases. The
   first column, *On every phase*, holds names that exist on each phase
   of the group (a target propagated onto every phase's structure set,
   say); one tick there is one target, not ten. The second column, *On
   some phases*, lists the rest with the series they belong to; such a
   target is read from the reference phase alone.
2. **Per-phase registration** - the reference volume is registered to
   every other phase with the elastix engine. Settings (levels,
   iterations, samples, grid spacing, sampling threshold) come from the
   Registration panel, adjustable in the dialog. The models:
   - **rigid** is, by default, a *local* rigid fit per structure: the
     region registered is the structure dilated by the *local margin* (15
     mm by default), so the fit follows that structure and not the whole
     image. Set the margin to 0 for one global rigid fit of the whole
     volume. A global fit of a breathing thorax is dominated by the
     spine, ribs and couch, which do not move, and reports a motion of
     zero for everything; that is what the local fit is for.
   - **deformable** is a B-spline refinement *started from* the global
     rigid result.
   - **as contoured** uses no registration at all: the structure is read
     from each phase's own contours, so the track is exactly what was
     drawn (or propagated) on each phase. It is offered when a ticked
     target exists on every phase and is then the reference the others
     are judged against.
3. **Propagation** - every target (and the reference structure, when
   chosen) is carried through each transform onto each phase's lattice,
   once per model; the transform maps reference → phase, so landing on
   the phase samples through the inverse, as in the propagation tool.
   *As contoured* rasterizes the phase's own contour instead.
4. **Measurement** (`src/motion.rs`) - per phase and model: centroid (mm,
   patient LPS), volume; from those: displacement from the reference
   phase, 3D magnitude, peak-to-peak amplitude (largest pairwise centroid
   distance), target-reference drift `|TV − ref|` and its peak-to-peak,
   and Pearson correlation of target vs. reference displacement along RL /
   AP / SI with two-tailed p-values (t-test, n−2 dof).
5. **ITV** - one per target and model, the union of the propagated masks
   over the selected phases, resampled onto the reference lattice, plus
   an optional uniform margin. The *Phases* row next to *Build ITV* (All /
   None, and the *Select phases* list under it) chooses which phases take
   part in the run; the reference phase always does. ITVs land as a segmentation series `4D ITV - <group>`
   referencing the reference phase series (display that phase to see and
   edit them; they export like any segmentation - SEG or RTSTRUCT).
6. **Registration QA** - per phase and model: the Dice of the two images'
   tissue after the registration against what it was before it, the
   engine's metric line, the 95th-percentile displacement, and the folding
   rate (fraction of sampled points with a non-positive Jacobian). The Dice
   leads every row and is coloured by the usual bands (0.80 and above
   green, 0.60 to 0.80 amber, below red), so a phase that did not land on
   the reference is visible without reading the metric line. Where a phase
   carries its own contour of a target, the propagated structure is also
   scored against that contour and the score is listed under the phase
   (`GTV vs contoured`): the honest measure of whether the model followed
   the anatomy, not just the image. Both go into the CSV's *Registration
   quality* section (`Image Dice after` / `before`, `Dice <structure> vs
   contoured`) and into the MCP report as `image_dice` and
   `structure_dice`.

*Keep per-phase segmentations* additionally stores every propagated mask
as a segmentation series on its phase (`4D <phase> - <group>`).

Cancel stops the run at the next phase boundary; a finished run is never
applied to a workspace that was replaced while it ran.

### Recipes - several studies, one workflow

Starting a run remembers the dialog as a *recipe*: target and
reference-structure names, models, ITV options and registration settings.
*Apply last recipe* re-ticks the same structures **by name** in whatever
workspace the dialog is open on - load the next patient (or the paired
upright/supine study into workspace B), open the tool, apply, run. Recipes
are name-based on purpose: indices and UIDs do not travel between patients.

## Results (`src/app/motion_results.rs`)

The results window opens when a run finishes (later: *Tools ▸ 📈 Motion
results*). Per run: the displacement-magnitude-vs-phase chart (one line
per target and model, reference structure dashed red), peak-to-peak
amplitude and drift bars (one per target and model), the per-phase table
(|d| and volume per track), correlation lines (r, p, significance stars,
synchrony wording), registration-QA lines and ITV volumes. A target that
was ticked once is one line and one amplitude; ticking the same name on
every phase separately would have made ten targets of it, which is what
the grouped *Targets* list prevents.

The window lays itself out to its own size: the run pickers wrap onto a
second line when the window is narrow, the export buttons stay at the
bottom, and only the report scrolls, vertically. The chart takes the full
width and scales its height with it (never more than about 60 % of the
visible height), and its margins are measured from the labels that go in
them - the widest value label on the left, half the last phase label on
the right - so the axis labels and the last phase are always on screen. A
per-phase table wider than the window scrolls sideways on its own.

The value axis uses a step of 1, 2 or 5 × 10ⁿ - the one whose interval
count is nearest to what the chart's height has room for (about one tick
per 34 points) - rounded out to whole steps, with as many decimals as the
step needs (a 0.5 mm step reads `0.0 0.5 1.0 1.5 2.0`), so no two ticks
carry the same label. The axis spans at least 1 mm, so a track that hardly
moves stays flat instead of being blown up to fill the chart. Hovering the
chart marks the phase under the pointer and lists every line's value there.

**Compare with** puts a second run beside the first - workspace A vs. B,
upright vs. supine - matching ITVs and tracks *by target name and model*:
ITV volumes with percentage change, peak-to-peak amplitudes, side by side.

**Export CSV** writes the selected run as a table a spreadsheet opens as
it is: **the phases are the columns and every method has its own row**,
the methods of one quantity on consecutive rows, so rigid, deformable and
as-contoured read against each other phase by phase. The file has a short
header (run, workspace, reference phase, reference structure, the axis
convention), then three sections separated by a blank line, each with its
own header row:

| Section | Label columns | Rows | Values |
|---|---|---|---|
| *Per-phase values* | Structure, Quantity, Unit, Method | per structure (targets, then the reference structure as `<name> (reference)`) and quantity, one per method | one column per phase |
| *Summary* | Structure, Method | one per structure and method | peak-to-peak, largest \|d\|, target-reference drift peak-to-peak, correlation r and p per axis, ITV volume, margin and name |
| *Registration quality* | Quantity, Unit, Method, Fit | per quantity, one per method and fit (`whole image`, or `<structure> neighbourhood` for a local rigid fit) | one column per phase; the reference phase is not registered and stays empty |

The per-phase quantities are the centroid position (`Centroid RL (x)`,
`AP (y)`, `SI (z)`, mm, patient LPS), the displacement from the reference
phase per axis and as `Displacement |d|`, the volume (cm3), the grey-level
minimum, mean and maximum inside the structure (where the phase images
were at hand), and for a target, where the same method also carried the
reference structure, `Offset to <reference> RL / AP / SI` (target centroid
minus reference-structure centroid). The registration-quality quantities
are the image Dice after and before, the metric at start and at end (its
unit column names the metric, `MSD` or `MI`), iterations, time, the
95th-percentile displacement, the folding rate and, where a phase carries
its own contour, `Dice <structure> vs contoured`.

Everything in the file is readable in any program: the typographic
symbols the window uses are written in plain ASCII (`·` as `-`, `▶` as
`->`, `cm³` as `cm3`), and a file that still holds a letter outside
ASCII (a structure named in another alphabet, say) starts with a UTF-8
byte-order mark, which is what makes Excel read it as UTF-8 rather than
in the system's code page. Names with a comma are quoted. **Export comparison
CSV** writes both runs one after the other and then a *Comparison*
section: one row per structure and method, with the two peak-to-peak
amplitudes and their difference, and the two ITV volumes and their change
in percent. The MCP `motion_report` tool returns the same CSV text.

## Transfer by relationship (`src/app/transfer_win.rs`)

*Tools ▸ ◎ Transfer by relationship…* places a structure of one workspace
into the other at the same **offset from a reference structure's
centroid** - the STAR workflow's target-heart relationship: a target is
projected into a workspace registration cannot reach (another patient,
another posture) via anatomy both workspaces can segment. The target keeps
its shape; the tool reports the offset (RL / AP / SI) it applied.
Reference structures whose name contains "heart" are pre-picked.

The window carries the same **Transform matrix** as the registration module
([registration.md](registration.md#the-matrix-typed-in-by-hand)), and it
shows the relationship's own answer: pick the two reference structures and
the grid fills with the shift between their centroids, which is exactly what
*▶ Transfer* would apply. Ticking **Use this matrix** takes that over - nudge
a number, or put a rotation into it - and places the structure through the
numbers instead of the centroid-to-centroid offset, so a rotation can be put into a placement
the relationship alone only shifts, and a transform worked out elsewhere can
be applied to anatomy the two workspaces do not share. The two reference
structures go quiet while it is on, because nothing then asks them anything:
only the target is needed, and the result is named `<target> (matrix)`
rather than after a reference it was not placed against.

## Structure comparison (`src/app/compare_win.rs`)

*Tools ▸ ◑ Structure comparison* computes, for any two structures (either
workspace, contours or segmentations, different lattices): volumes, centroid
offset (vector and magnitude), Dice, 95th-percentile symmetric Hausdorff
distance and the surface distance as mean, SD and maximum. The second mask
is resampled onto the first's lattice through patient coordinates; across
two frames of reference the window notes the comparison assumes
corresponding coordinates. *Save CSV* writes the lot as a two-column table.

**The rigid offset** is the one genuinely new computation and has its own
tick, because it costs a second on a large structure. The two surfaces carry
no point correspondence, so one is invented and then improved: every point
of A is paired with the nearest point of B, the rigid body that best
explains those pairs is fitted (orthogonal Procrustes, the same
`registration::analysis::fit_rigid` the registration analytics use), A is
moved, and the pairing is done again. The fit is always taken from A's
*original* points, so the iteration refines one global transform instead of
accumulating a chain of small ones. What comes out is a translation, three
Euler angles, and the surface distance left over afterwards next to what it
was before - which is how one sees whether a rigid body explains the
difference at all.

**Two points of interest** are compared as points: the window reports their
separation instead of a Dice score, which is the target registration error
when the two are meant to be the same landmark.

Everything the window measures is shown as a two-column table - quantity and
value, the same rows *Save CSV* writes - with only the warnings and the "why
there is nothing to compare" messages left as sentences.

## Structure details (`src/app/stats_win.rs`)

*Tools ▸ 📋 Structure details* is the other half: one row per structure of
one workspace rather than two structures against each other. The columns
are the structure's **Format** (contours, voxels or a point) and its volume
twice over - **Surface-based**, inside a closed triangle surface
reconstructed from the contours, and **Voxels-based**, the voxels the
contours fill - which disagree by a few per cent on a coarse series, more on
a small structure, and neither of which is wrong; then the grey levels
inside the structure (which is how a mis-drawn organ gives itself away),
what the geometry costs in slices and points, and whether a derived
structure still matches its recipe. Points of interest show their
coordinates instead of a volume. How each column is calculated, formula by
formula, is in [volumes.md](volumes.md). The CSV carries them as
`surface_based_cm3` and `voxels_based_cm3`.

The **Dice** column measures every row against a reference chosen above the
table: the structure or segment of *the same name in another workspace* -
one entry per open workspace, so with three of them the table can be measured
against whichever is the comparison; it is the default as soon as a second
one is loaded, and it is what a propagation, a phase or a second observer is
checked with - or one structure or segment of any workspace for
all rows (an auto-segmentation against the manual one). Contours are
rasterized onto the row's own lattice, a reference on another lattice is
resampled onto it, and a row with no counterpart shows `-`; the tooltip
names what it was measured against, and the CSV carries both. The table is
computed on demand rather than every frame, and says so when the geometry
has changed under it. *Save CSV* writes it out.

## Numerics worth knowing

- Centroids are exact under the affine lattice→patient map (mean index,
  then map); peak-to-peak is the largest pairwise distance, independent of
  the reference-phase choice.
- The p-values come from the regularized incomplete beta function
  (continued-fraction evaluation) - the exact t-distribution tail, not a
  normal approximation; with 10 phases n is small enough for that to matter.
- HD95/MSD use the exact anisotropic Euclidean distance transform
  (`morphology::dist2_to_foreground`) evaluated on surface voxels of each
  mask against the other, both directions pooled for the percentile.
- ITV volumes inherit every caveat of nearest-neighbour resampling between
  phase lattices; centroid metrics are the primary motion descriptors, as
  in the underlying study design.
