# Automatic body / EXTERNAL contouring

The outer patient surface, with the couch, the chair and the immobilisation
left outside it. Two methods in one tool — a deterministic geometric one
that needs nothing, and a model-assisted one that borrows
TotalSegmentator's openly licensed body network for the part geometry
cannot decide. Both work on CT and on MR, supine or upright.

## Why this contour and not another

Every downstream calculation starts here. A dose engine needs to know where
the patient begins, because that is where the range budget starts — for
protons a 3 mm error in the entrance surface is a 3 mm error in every
distal edge behind it. A DRR needs to know what is *not* patient, or it
projects the couch straight through the anatomy. A registration wants to
sample inside the body and nowhere else, or it spends its iterations
aligning the table.

And it is the contour nobody wants to draw. So it has to be right on the
first scan of the day, with no parameters touched.

## What makes it hard

Not the skin — that is the largest step in the image and any threshold
finds it. The difficulty is everything else in the field of view:

* a **couch top** is two carbon skins around a foam core, and the skins are
  as dense as bone;
* a **thermoplastic mask**, a **headrest shell**, a **vacuum-bag** fabric
  and a **chair backrest** are all thin, dense and *touching the patient*,
  so in a threshold mask they are one connected object with them;
* **blankets, cables, positioning pads** drift in and out of the field;
* **the reconstruction circle** leaves a bright rim on some scanners;
* and the patient is not always one object — a leg scan is two, an arm cut
  off by the field of view is another.

Meanwhile the parts of the patient that look most like equipment — an ear,
a nose, a fingertip — are exactly the thin structures that any method
aggressive enough to remove a mask will also remove.

## Using it

*Tools ▶ 👤 Body-contour dataset A/B…*, or the **👤 Body…** button in the
sidebar *Segmentations* section. The window follows the shared layout of
the segmentation tools (see
[architecture.md](architecture.md#the-segmentation-tool-windows)).

* **Method** — *Classical* or *Model-assisted*; everything below adapts.
* **Tissue above** — on CT a Hounsfield threshold (default −300 HU); on MR
  a fraction of the bias-corrected 99th percentile, or Otsu. The window
  re-seeds this whenever the displayed series changes modality, because a
  CT threshold is meaningless on MR and the reverse is just as wrong.
* **Name** and **as EXTERNAL structure** — the mask lands as an ordinary
  editable segmentation, and optionally also as an RTSTRUCT ROI of
  interpreted type `EXTERNAL`, which is the tag a planning system looks for.
  It then renders like any ROI and rides the DICOM export.
* **Options** — the smallest body detail, the equipment test with its shell
  thickness and repeat window, the smallest body part, thin-anatomy
  recovery, whether the body is reported solid, surface smoothing, and
  (model-assisted only) the network margin, compute device and model
  folder.

**▶ Contour** runs on a background thread with the usual progress row and
Cancel. The status line reports the body volume, how many separate bodies
were kept, how much equipment was removed and how much thin anatomy was
given back — which is the number to watch: a run that removes nothing is a
run whose threshold or opening radius is wrong.

## Method A — classical

Deterministic, nothing to download, a few seconds on a whole-body CT, and
every step explainable to a physicist doing QA.

### 1. Foreground

CT thresholds directly: −300 HU sits in the gap between air and fat. (The
skin edge moves about half a millimetre per 100 HU through the
partial-volume ramp — this is the one number worth agreeing on with your
planning system.)

MR has no absolute scale: the same tissue is a different number on the next
sequence, and a different number again on the other side of the same slice,
because the receive coils shade the image. So the MR path divides the image
by a heavy blur of itself (σ = 40 mm by default) — a poor man's N4, no
iteration and no histogram model — which flattens the shading while leaving
every edge intact, then thresholds at a low fraction of the 99th percentile.
A low fraction rather than Otsu by default: Otsu splits *bright from dark*,
not tissue from air, so it runs high on fat-suppressed series and bites into
subcutaneous fat.

### 2. Equipment, by two geometric facts

Equipment is separated from anatomy without knowing what either looks like,
using two properties no patient has together:

**It is thin.** An opening removes every shell whose largest inscribed ball
is smaller than its radius, while leaving everything thicker with its
surface *exactly* intact — an opening is the union of every ball that fits
inside the mask, so the ball rolls along the inside of the skin and touches
every point of it. The distance transform behind it is the exact
anisotropic Euclidean one, so the radius means the same along every axis
whatever the slice thickness, and the cost does not depend on it.

The radius here is **2 mm**, not the 8 mm used later to decide what is big
enough to be a body, and the difference matters more than anything else on
this page. A couch skin is one or two millimetres of carbon and a
thermoplastic mask two or three; the thinnest tissue anyone would miss —
the chest wall over a lung — is five or six. At 2 mm the two are cleanly
separated. At 3 mm a six-millimetre chest wall becomes a candidate too, and
since it repeats slice after slice it is then indistinguishable from a
couch skin: the ribcage goes with the table. (This is not hypothetical. It
is what the first version of this code did to the bundled 4D-Lung study,
and it is why the test suite contains a hollow cylinder.)

**It is extruded.** A couch top, a backrest, a seat pan, an arm rest: each
is a surface swept along one axis, so its footprint in the orthogonal plane
repeats slice after slice after slice. A pinna is 25 mm long, a nose 30 mm,
a fingertip 15 mm. Requiring a footprint to repeat over 150 mm in 80 % of
that window's slices separates them with room to spare — and the test runs
along **all three** axes, because a supine couch is extruded along z while
an upright chair's seat pan and arm rests are extruded along x.

Being thin alone is not enough to be discarded, and neither is repeating;
only both together mark a voxel as equipment.

### 3. A body is a solid object

A threshold does not see a body. It sees a shell of tissue wrapped round two
lungs, a stomach and a bowel — and left that way, the chest wall over a lung
is a thin sheet that repeats slice after slice, which is to say
indistinguishable from a couch skin by the rules just described. So the
interior is closed *before* any of the size reasoning below: fill what the
slice border cannot reach, and the wall becomes part of a solid object
again.

Slice by slice, not in three dimensions — the lungs drain to the outside air
through the trachea, so on any scan that includes the neck a 3-D fill leaves
both lungs open, while slice by slice they close. (An open mouth stays a
cavity, which is the usual convention, and so does a lung on the two or
three slices where the airway is actually open to the air.)

The order is the point: **equipment first, then fill**. Fill first and a
couch top with a closed profile becomes a solid slab before anything has a
chance to recognise it.

### 4. Which components are a patient

The opened mask is split into 6-connected components, and every component
of at least 50 cm³ is kept — *not* merely the largest, so a leg scan comes
out as two bodies and a truncated arm as a third. (A shared corner is not
contact: 6-connectivity is what stops a couch rail grazing the skin
diagonally from merging with it.)

### 5. Giving the thin anatomy back

The 8 mm opening shaved a rim off the body, and took the ears with it. Two
questions put it back, because there are two different things in there.

What the opening removed from the body's **own surface** lies, by
construction, within one opening radius of what is left of it — a skin rim,
the edge of a shoulder, the sharp flank of a cross-section. It can run the
whole length of the scan and still be nothing but patient, so its size is
not asked about; it is simply given back.

What stands **clear** of that is a separate object that happens to touch:
an ear, a nose or a fingertip, which are small, or a pad, a blanket or a
bolus, which are not. There, size is exactly the right question, and
anything more than 100 mm across stays out. Two rounds, because a fingertip
hangs off a finger.

### 6. Surface

An optional closing at the end takes the staircase off the contour.

### Where it fails, stated plainly

Where a shell touches the skin with **no air gap at all** — the mask on the
forehead and chin, a bare couch skin under the back, a bolus — the shell's
thickness over the contact patch stays inside the body. It is 2–5 mm, over
the contact patches only, and it is not detectable by geometry, because
locally it *is* a slightly thicker patient. In practice cushions mean bare
couch contact is rare, and bolus in the external is the convention anyway.
The model-assisted method is the answer to the rest.

## Method B — model-assisted

TotalSegmentator publishes a **body-outline nnU-Net** under the same
Apache-2.0 licence as its "total" task, in three flavours:

| Model | Dataset | Grid | Download |
|---|---|---|---|
| CT 6 mm | 300 | 6 mm isotropic | 124 MB |
| CT 1.5 mm | 299 | 1.5 mm isotropic | 233 MB |
| MR | 597 | 3.0 × 1.19 × 0.99 mm | 230 MB |

They run through the *same* engine as the 117-class auto-segmentation —
`autoseg::run_specs`, the same `PlainConvUNet` rebuilt from `plans.json`,
the same sliding window, the same CPU/GPU choice — and their weights live
beside it in `models/totalsegmentator/`. The model manager lists them like
any other.

The network is **not** used as the answer. It is planned at 6 mm or 1.5 mm;
its boundary is far too coarse to be a skin surface. It is used as a
*classifier*:

```
body  =  threshold(image)  ∧  dilate(network_body, 6 mm)
```

The network decides **what** is patient — it removes a mask contact patch or
a couch sliver semantically, which no geometry can — and the threshold still
decides **where** the skin is, at full image resolution. The result then goes
through exactly the same components / thin-recovery / fill / closing steps as
the classical method.

The network's own two classes (`body_trunc`, `body_extremities`) are cleaned
first the way the reference implementation cleans them: the trunk is one
object, so only its largest blob survives; extremities are several, so they
are filtered at 50 000 mm³ — the same constant used upstream. The body is
the union.

The equipment test still runs. It has little left to do once the network has
answered, but the guide is used *dilated* — 6 mm by default — and a margin
that generous can pull a touching rail back in. Two cheap passes over the
volume are a fair price for not having to think about that.

### What is new in the engine for this

Supporting the MR body model meant three additions, all of them additive —
the existing models take exactly the code path they took before, so their
numerics are untouched:

* `ZScoreNormalization` alongside `CTNormalization`. Where CT normalizes
  against dataset constants from `plans.json`, MR normalizes against *this
  image*, so its constants are only knowable after resampling.
* **Anisotropic target spacing.** The MR model plans 3.0 × 1.19 × 0.99 mm;
  `SarMap` now takes a spacing per axis rather than one number.
* **A general transposed convolution.** The MR decoder upsamples
  `[1, 2, 2]` at two of its five stages. The hand-tuned 2× routine is still
  what every isotropic model uses; a general `kernel = stride` version
  handles the rest, on CPU and through burn on the GPU.

## Cost

The classical method is a handful of distance transforms and flood fills.
Measured on the bundled 4D-Lung study — 512 × 512 × 133 at 0.98 × 0.98 ×
3 mm, on two throttled cores — it takes **13.8 s** end to end and reports a
23.2 L body with 85 cm³ of couch left out. It allocates about one byte per
voxel per intermediate mask, plus four bytes per *set* voxel for the
component lists.

The model-assisted method adds one nnU-Net inference: 34 s for the 6 mm
model on the same two cores (50 s in total), minutes with the 1.5 mm or MR
model. Since the network only has to say which side of the skin a voxel is
on, **6 mm is the sensible default** — the resolution comes from the
threshold, not from it.

## Verification

First, on real data. The bundled 4D-Lung study carries a real couch rail at
the bottom of the field; the classical method removes it on every slice and
follows the skin to the voxel, including the three separate pieces — two
arms and the neck — that the most superior slices contain.

The two methods are also each other's check, and on that study they pass
it: run separately, the classical geometry and the 6 mm network agree on
**8 098 425 of 8 098 443 voxels** — eighteen voxels apart, Dice 0.999999.
Neither was tuned against the other; they simply have to be looking at the
same surface.

Then `tests/body.rs`, which builds a phantom containing every failure mode
deliberately: an elliptical body, a couch skin and a rail under the back, a
*moulded* mask shell that stands 2 mm clear of the skin over most of its
span and presses against it over a patch, ears thin enough for the opening
to shave off, lungs draining through an airway, a cable — with anisotropic
2 × 2 × 5 mm voxels, since that is where a voxel-counting implementation
goes wrong and a millimetre-aware one does not. It asserts:

* Dice > 0.99 against the body it was built from;
* no couch skin, rail, cable or free-standing mask shell anywhere;
* the shell **is** kept where it presses against the skin, and the total
  error beyond the patient stays under 10 cm³ — the documented limitation,
  pinned so that a change which quietly makes it worse is caught;
* both ears kept, and a non-zero recovered-anatomy count;
* both lungs inside the body on every slice past the airway;
* a hollow cylinder — a 6 mm wall around a cavity, beside a 2 mm couch skin
  — comes out with its wall intact and its couch gone, which is the
  regression test for the failure real data taught;
* two separated legs come out as two bodies, not as the larger one;
* an MR version with an exponential receive gradient still comes out whole
  at both ends of the field (Dice > 0.97);
* a model folder that cannot exist is an error, not a panic.

`the_model_assisted_method_runs_the_published_network` is `#[ignore]`d
because it downloads 124 MB; it runs the real Dataset300 weights end to end
through the hybrid.

`src/morphology.rs`'s own tests check the pieces underneath: the distance
transform against brute force on an anisotropic grid, the opening against a
sheet and a block, 6-connectivity against a shared corner, slice-wise versus
3-D filling, the persistence test against an extruded rail and a bump, and
the blur against a constant and a step.

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

For batch checks over a folder of scans: `--out` writes a raw `u8` mask on
the original grid, one byte per voxel in `Volume::data` order — the same
convention as the other example tools, so masks can be compared byte for
byte between methods.

## Licensing and citation

The classical method has no weights and no third-party code. The
model-assisted method uses TotalSegmentator's `body` and `body_mr` tasks,
which the authors publish under **Apache-2.0** as openly available for any
usage, commercial included. If you use it in academic work, cite
TotalSegmentator and nnU-Net as in
[auto-segmentation.md](auto-segmentation.md#licensing-and-citation).

As with everything in this viewer: research and QA use — not a medical
device, not for clinical decision-making.
