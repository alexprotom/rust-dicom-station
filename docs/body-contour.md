# Automatic body / EXTERNAL contouring

The outer patient surface, with couch, chair and immobilisation left outside
it. Two methods in one tool: a deterministic geometric one that needs nothing,
and a model-assisted one that borrows TotalSegmentator's openly licensed body
network for what geometry cannot decide. Both work on CT and MR, supine or
upright.

## Why this contour and not another

Every downstream calculation starts here. A dose engine's range budget starts
at the entrance surface - for protons a 3 mm error there is a 3 mm error in
every distal edge behind it; a DRR that does not know what is *not* patient
projects the couch through the anatomy; a registration that samples outside
the body spends its iterations aligning the table. And nobody wants to draw
it, so it has to be right on the first scan of the day, untouched.

## What makes it hard

Not the skin - the largest step in the image, which any threshold finds - but
everything else in the field of view:

* a **couch top** is two carbon skins, as dense as bone, round a foam core;
* a **thermoplastic mask**, a **headrest shell**, a **vacuum-bag** fabric and
  a **chair backrest** are thin, dense and *touching the patient*, so a
  threshold joins them to it;
* **blankets, cables, positioning pads** drift in and out of the field;
* **the reconstruction circle** leaves a bright rim on some scanners;
* and the patient is not always one object - a leg scan is two, an arm cut off
  by the field of view is another.

## Using it

*Tools ▶ 👤 Body contour* (the dataset is chosen on the window's **Dataset
A / B** row), or the **👤 Body…** button in the
sidebar *Segmentations* section. The window shares the segmentation tools'
layout (see [architecture.md](architecture.md#the-segmentation-tool-windows)).

* **Method** - *Classical* or *Model-assisted*; everything below adapts.
* **Tissue above** - on CT a Hounsfield threshold (default −300 HU); on MR a
  fraction of the bias-corrected 99th percentile, or Otsu. Re-seeded whenever
  the displayed series changes modality.
* **Name** and **as EXTERNAL structure** - the mask lands as an ordinary
  editable segmentation, optionally also as an RTSTRUCT ROI of interpreted
  type `EXTERNAL`, the tag a planning system looks for; it then renders like
  any ROI and rides the DICOM export.
* **Options** - the smallest body detail, the equipment test with its shell
  thickness and repeat window, the smallest body part, thin-anatomy recovery,
  whether the body is reported solid, surface smoothing, and (model-assisted
  only) the network margin, compute device and model folder.

**▶ Contour** runs on a background thread with the usual progress row and
Cancel. The status line reports body volume, bodies kept, equipment removed
and thin anatomy given back - the number to watch: a run that removes nothing
has the wrong threshold or opening radius.

## Method A - classical

Deterministic, nothing to download, and every step explainable to a physicist
doing QA.

### 1. Foreground

CT thresholds directly: −300 HU sits in the gap between air and fat. (The skin
edge moves about half a millimetre per 100 HU through the partial-volume ramp -
the one number worth agreeing with your planning system.)

MR has no absolute scale, and the receive coils shade each image. So the MR
path divides the image by a heavy blur of itself (σ = 40 mm by default) - a
poor man's N4 - which flattens the shading and keeps every edge, then
thresholds at a low fraction of the 99th percentile. Not Otsu by default: Otsu
splits *bright from dark*, not tissue from air, so it bites into subcutaneous
fat on fat-suppressed series.

### 2. Equipment, by two geometric facts

Equipment is separated from anatomy without knowing what either looks like, by
two properties no patient has together - only both, never one alone, mark a
voxel as equipment:

**It is thin.** An opening removes every shell whose largest inscribed ball is
smaller than its radius and leaves everything thicker with its surface
*exactly* intact. Its distance transform is the exact anisotropic Euclidean
one, so the radius means the same along every axis whatever the slice
thickness, at a cost independent of it.

The radius is **2 mm**, not the 8 mm used later to decide what is big enough
to be a body. A couch skin is one or two millimetres of carbon, a
thermoplastic mask two or three, and the thinnest tissue anyone would miss -
the chest wall over a lung - five or six. 2 mm separates them cleanly; at 3 mm
a six-millimetre chest wall is a candidate too, repeats slice after slice like
a couch skin, and the ribcage goes with the table - as the first version of
this code proved on the bundled 4D-Lung study; hence the hollow cylinder in
the test suite.

**It is extruded.** A couch top, backrest, seat pan or arm rest is a surface
swept along one axis; its footprint in the orthogonal plane repeats slice
after slice, while a pinna is 25 mm long, a nose 30 mm, a fingertip 15 mm.
Requiring a footprint to repeat over 150 mm in 80 % of that window's slices
separates them with room to spare - along **all three** axes, since a supine
couch is extruded along z and an upright chair's seat pan and arm rests along
x.

### 3. A body is a solid object

A threshold sees not a body but a shell of tissue round lungs, stomach and
bowel - and left that way, the chest wall over a lung is a thin repeating
sheet: a couch skin by the rules above. So the interior is closed *before* any
size reasoning - fill what the slice border cannot reach and the wall is solid
again.

Slice by slice, not in three dimensions: the lungs drain to the outside air
through the trachea, so on any scan including the neck a 3-D fill leaves them
open while slice by slice they close. (An open mouth stays a cavity, the usual
convention, as does a lung on the two or three slices where the airway is
actually open.)

Order matters - **equipment first, then fill**: fill first and a couch top
with a closed profile becomes a solid slab before anything can recognise it.

### 4. Which components are a patient

The opened mask is split into 6-connected components and every component of at
least 50 cm³ is kept - *not* merely the largest, so a leg scan comes out as
two bodies and a truncated arm as a third. (A shared corner is not contact:
6-connectivity stops a couch rail grazing the skin diagonally from merging
with it.)

### 5. Giving the thin anatomy back

The 8 mm opening shaved a rim off the body and took the ears with it. Two
questions put it back.

What it removed from the body's **own surface** - a skin rim, the edge of a
shoulder, the sharp flank of a cross-section - lies by construction within one
opening radius of what is left, can run the whole length of the scan and still
be nothing but patient, and is given back without asking its size.

What stands **clear** of that is a separate object that happens to touch: an
ear, a nose or a fingertip, which are small, or a pad, a blanket or a bolus,
which are not - so there size is the question, and anything more than 100 mm
across stays out. Two rounds, because a fingertip hangs off a finger.

### 6. Surface

An optional closing at the end takes the staircase off the contour.

### Where it fails, stated plainly

Where a shell touches the skin with **no air gap at all** - the mask on the
forehead and chin, a bare couch skin under the back, a bolus - its thickness
over the contact patch, 2-5 mm, stays inside the body: locally it *is* a
slightly thicker patient, and no geometry can tell. Cushions make bare couch
contact rare, bolus in the external is the convention anyway, and the
model-assisted method answers the rest.

## Method B - model-assisted

TotalSegmentator publishes a **body-outline nnU-Net** under the same
Apache-2.0 licence as its "total" task, in three flavours:

| Model | Dataset | Grid | Download |
|---|---|---|---|
| CT 6 mm | 300 | 6 mm isotropic | 124 MB |
| CT 1.5 mm | 299 | 1.5 mm isotropic | 233 MB |
| MR | 597 | 3.0 × 1.19 × 0.99 mm | 230 MB |

They run through the *same* engine as the 117-class auto-segmentation -
`autoseg::run_specs`, the same `PlainConvUNet` rebuilt from `plans.json`, the
same sliding window and CPU/GPU choice - with weights beside it in
`models/totalsegmentator/`, and the model manager lists them like any other.

Planned at 6 mm or 1.5 mm, the network is far too coarse for a skin surface,
so it is **not** the answer but a *classifier*:

```
body  =  threshold(image)  ∧  dilate(network_body, 6 mm)
```

The network decides **what** is patient - removing a mask contact patch or a
couch sliver semantically, which no geometry can - and the threshold decides
**where** the skin is, at full image resolution. The result then goes through
the same components / thin-recovery / fill / closing steps as the classical
method.

Its two classes (`body_trunc`, `body_extremities`) are first cleaned as the
reference implementation does - the trunk keeps only its largest blob, the
extremities are filtered at 50 000 mm³, the same constant used upstream - and
the body is their union.

The equipment test still runs: little is left for it once the network has
answered, but the guide is used *dilated* - 6 mm by default - and that margin
can pull a touching rail back in.

### What is new in the engine for this

The MR body model needed three additions, all additive - existing models take
exactly the code path they took before, so their numerics are untouched:

* `ZScoreNormalization` alongside `CTNormalization`: CT normalizes against
  dataset constants from `plans.json`, MR against *this image*, so its
  constants are only knowable after resampling.
* **Anisotropic target spacing.** The MR model plans 3.0 × 1.19 × 0.99 mm;
  `SarMap` now takes a spacing per axis rather than one number.
* **A general transposed convolution.** The MR decoder upsamples `[1, 2, 2]`
  at two of its five stages. Isotropic models keep the hand-tuned 2× routine;
  a general `kernel = stride` version handles the rest, on CPU and through
  burn on the GPU.

## Cost

The classical method is a handful of distance transforms and flood fills: on
the bundled 4D-Lung study - 512 × 512 × 133 at 0.98 × 0.98 × 3 mm, two
throttled cores - **13.8 s** end to end, reporting a 23.2 L body with 85 cm³
of couch left out. It allocates about one byte per voxel per intermediate
mask, plus four bytes per *set* voxel for the component lists.

The model-assisted method adds one nnU-Net inference: 34 s for the 6 mm model
on the same cores (50 s in total), minutes with the 1.5 mm or MR model. The
network only says which side of the skin a voxel is on, so **6 mm is the
sensible default**.

## Verification

On real data first: the bundled 4D-Lung study carries a real couch rail at the
bottom of the field; the classical method removes it on every slice and
follows the skin to the voxel, including the three separate pieces - two arms
and the neck - in the most superior slices.

The two methods are also each other's check: run separately on that study, the
classical geometry and the 6 mm network agree on **8 098 425 of 8 098 443
voxels** - eighteen apart, Dice 0.999999 - and neither was tuned against the
other.

Then `tests/body.rs`, whose phantom deliberately contains every failure mode -
an elliptical body, a couch skin and a rail under the back, a *moulded* mask
shell 2 mm clear of the skin over most of its span and pressed against it over
a patch, ears thin enough for the opening to shave off, lungs draining through
an airway, a cable - on anisotropic 2 × 2 × 5 mm voxels, where a
voxel-counting implementation goes wrong and a millimetre-aware one does not.
It asserts:

* Dice > 0.99 against the body it was built from;
* no couch skin, rail, cable or free-standing mask shell anywhere;
* the shell **is** kept where it presses against the skin, with the total
  error beyond the patient under 10 cm³ - the documented limitation, pinned;
* both ears kept, and a non-zero recovered-anatomy count;
* both lungs inside the body on every slice past the airway;
* a hollow cylinder - a 6 mm wall round a cavity, beside a 2 mm couch skin -
  keeps its wall and loses its couch;
* two separated legs come out as two bodies, not as the larger one;
* an MR version with an exponential receive gradient still comes out whole at
  both ends of the field (Dice > 0.97);
* a model folder that cannot exist is an error, not a panic.

`the_model_assisted_method_runs_the_published_network` runs the real
Dataset300 weights end to end through the hybrid; `#[ignore]`d because it
downloads 124 MB.

`src/morphology.rs`'s own tests check the pieces underneath: the distance
transform against brute force on an anisotropic grid, the opening against a
sheet and a block, 6-connectivity against a shared corner, slice-wise versus
3-D filling, the persistence test against an extruded rail and a bump, and the
blur against a constant and a step.

## Command-line tool

```
cargo run --release --example body_cli -- <dicom_dir> \
    [--method classical|model] [--model ct6|ct15|mr] \
    [--hu -300] [--mr-fraction 0.12] [--mr-otsu] [--bias-sigma 40] \
    [--open 8] [--thin-shell 2] [--no-devices] [--window 150] [--frac 0.8] \
    [--min-cm3 50] [--no-thin] [--thin-extent 100] [--margin 6] \
    [--thin-shell 2] [--no-fill] [--close 0] \
    [--models DIR] [--device auto|gpu|cpu] [--out mask.bin]
```

For batch checks over a folder of scans, `--out` writes a raw `u8` mask on the
original grid, one byte per voxel in `Volume::data` order - the convention of
the other example tools, so masks compare byte for byte between methods.

## Licensing and citation

The classical method has no weights and no third-party code. The
model-assisted method uses TotalSegmentator's `body` and `body_mr` tasks,
which the authors publish under **Apache-2.0** for any usage, commercial
included. In academic work, cite TotalSegmentator and nnU-Net as in
[auto-segmentation.md](auto-segmentation.md#licensing-and-citation).

As with everything in this viewer: research and QA use - not a medical device,
not for clinical decision-making.
