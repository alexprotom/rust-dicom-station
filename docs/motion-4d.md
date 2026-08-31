# 4D motion analysis and ITV generation

This document covers the 4D workflow: how image series are grouped into 4D
sub-studies, what the **📈 4D motion / ITV** tool computes and how, where the
results land, and the two companion tools — **◎ Transfer by relationship**
and **◑ Compare structures**. The workflow reproduces, inside the viewer,
the pipeline of ITV-based motion studies (e.g. upright-vs-supine STAR
target evaluation): per-phase registration, target propagation, centroid
motion metrics, target–reference drift and correlation, and ITV volumes,
with the same run repeatable on another dataset for a posture or cohort
comparison.

## 4D groups (`src/fourd.rs`)

A 4DCT arrives as one image series per respiratory phase, usually with an
average and sometimes MIP/MinIP reconstructions beside them. DICOM implies
the acquisition they form but stores no node for it, so the viewer
reconstructs one:

- Series are bucketed by (study, modality). A series whose description
  carries a number directly before a `%` (e.g. `Thorax 4D 30%`, `CT 0 %
  Ex`) or the keyword form `phase` + number (`4DCT_phase_000`, `Phase 3`)
  is a **phase**; the description with the number removed (the
  *template*) tells two 4D sets in one study apart, so a thin- and a
  thick-slice reconstruction of the same phases become two groups. The
  template is also the group's name stem (`4D CT — Thorax 4D (10 phases)`).
- Series with a `TemporalPositionIdentifier` but no percent group by
  identical description and order by that identifier (`t1`, `t2`, …).
- `AVG`/`average`/`mean`, `MIP` and `MinIP` in the description mark the
  reconstructions; they attach to the bucket's first group.
- A group needs **at least three phases** — two series with "50%" in the
  name are more likely a coincidence than an acquisition.

Detection is a heuristic, so everything can be corrected by hand from the
data tree: right-click a series ▸ *4D group* to add it to a group or start
a new one; right-click a group member to reorder it, change its role or
remove it; right-click the group node to rename it, dissolve it or re-run
detection. Hand-edited groups are marked *custom* and survive re-detection
(`fourd::refresh`, which runs whenever the series list changes). A
dissolved auto-detected group leaves a hidden tombstone so the next refresh
does not rebuild it; only the explicit *Re-detect 4D groups* clears
tombstones. Members reference series by **SeriesInstanceUID**, so renames,
removals and copies never corrupt a group — a member whose series is gone
simply drops out of view.

In the tree a group renders as a `🎞` node inside its study, phases in
temporal order, then the reconstructions; grouped series leave their
modality node so each series has one place in the tree. Clicking a member
displays that series, exactly like an ordinary series row.

## The pipeline (`src/app/motion_win.rs`)

*Tools ▸ 📈 Motion-analyse dataset A/B…*, or right-click a 4D group ▸
*Motion / ITV analysis…*. One run:

1. **Reference phase** — chosen in the dialog (default: the 0 % phase).
   The targets are defined on it: contours are rasterized onto its
   lattice, segmentations drawn on another lattice are resampled onto it.
2. **Per-phase registration** — the reference volume is registered to
   every other phase with the elastix engine: a rigid stage, and (for the
   deformable model) a B-spline refinement *started from* the rigid result,
   so the deformable transform is rigid + correction. The settings
   (levels, iterations, samples, grid spacing, sampling threshold) start
   from the Registration panel's current values and can be adjusted in the
   dialog.
3. **Propagation** — every target (and the reference structure, when one
   is chosen) is carried through each transform onto each phase's lattice,
   once per model. The transform maps reference → phase, so landing on the
   phase samples through the inverse — the same convention as the
   propagation tool.
4. **Measurement** (`src/motion.rs`) — per phase and model: centroid (mm,
   patient LPS), volume; from those: displacement from the reference
   phase, 3D magnitude, peak-to-peak amplitude (largest pairwise centroid
   distance), target–reference drift `|TV − ref|` and its peak-to-peak,
   and Pearson correlation of target vs. reference displacement along RL /
   AP / SI with two-tailed p-values (t-test, n−2 dof).
5. **ITV** — per target and model, the union of the propagated masks over
   all phases, resampled onto the reference lattice, plus an optional
   uniform margin. ITVs land as a segmentation series `4D ITV — <group>`
   referencing the reference phase series (display that phase to see and
   edit them; from there they export like any segmentation — SEG or
   RTSTRUCT).
6. **Registration QA** — per phase and model: the engine's metric line,
   the 95th-percentile displacement, and the folding rate (fraction of
   sampled points with a non-positive Jacobian). Dice / HD95 between any
   two structures is a click away in ◑ Compare structures.

*Keep per-phase segmentations* additionally stores every propagated mask
as a segmentation series on its phase (`4D <phase> — <group>`).

Cancel stops the run at the next phase boundary; a finished run is never
applied to a dataset that was replaced while it ran.

### Recipes — several studies, one workflow

Starting a run remembers the dialog as a *recipe*: the target and
reference-structure names, models, ITV options and registration settings.
*Apply last recipe* re-ticks the same structures **by name** in whatever
dataset the dialog is open on — load the next patient (or the paired
upright/supine study into dataset B), open the tool, apply, run. Recipes
are name-based on purpose: cohort workflows name their structures
consistently, indices and UIDs do not travel between patients.

## Results (`src/app/motion_results.rs`)

The results window opens when a run finishes (*Tools ▸ 📈 Motion results…*
later). Per run: the displacement-magnitude-vs-phase chart (targets ×
models, the reference structure dashed red), peak-to-peak amplitude and
drift bars, the per-phase table (|d| and volume per track), the
correlation lines (r, p, significance stars, synchrony wording), the
registration-QA lines and the ITV volumes.

**Compare with** puts a second run beside the first — dataset A vs. B,
upright vs. supine — and matches ITVs and tracks *by target name and
model*: ITV volumes side by side with the percentage change, peak-to-peak
amplitudes side by side.

**Export CSV** writes one long-format CSV (a `table` column separates the
sections: per-phase centroids and displacements, peak-to-peak rows,
correlations, QA, ITVs); the comparison export appends the second run's
rows to the same file.

## Transfer by relationship (`src/app/transfer_win.rs`)

*Tools ▸ ◎ Transfer by relationship…* places a structure of one dataset
into the other at the same **offset from a reference structure's
centroid** — the target–heart relationship of the STAR workflow: a target
defined on one patient's imaging is projected into a dataset registration
cannot reach (another patient, another posture) via anatomy both datasets
can segment. The target keeps its shape; the tool reports the offset (RL /
AP / SI) it applied. Deformable adaptation afterwards, when wanted, is the
propagation tool's job. Reference structures whose name contains "heart"
are pre-picked.

## Compare structures (`src/app/compare_win.rs`)

*Tools ▸ ◑ Compare structures…* computes, for any two structures (either
dataset, contours or segmentations, different lattices): volumes, centroid
offset (vector and magnitude), Dice, 95th-percentile symmetric Hausdorff
distance and mean symmetric surface distance. The second mask is resampled
onto the first's lattice through patient coordinates; across two frames of
reference the window says the comparison assumes the coordinates already
correspond.

## Numerics worth knowing

- Centroids are exact under the affine lattice→patient map (mean index,
  then map). Peak-to-peak is the largest pairwise distance, independent of
  the reference-phase choice.
- The p-values come from the regularized incomplete beta function
  (continued-fraction evaluation), i.e. the exact t-distribution tail, not
  a normal approximation — with 10 phases n is small enough for that to
  matter.
- HD95/MSD use the exact anisotropic Euclidean distance transform
  (`morphology::dist2_to_foreground`) evaluated on surface voxels of each
  mask against the other, both directions pooled for the percentile.
- ITV volumes inherit every caveat of nearest-neighbour resampling between
  phase lattices; centroid metrics are the primary motion descriptors, as
  in the underlying study design.
