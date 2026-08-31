# 4D motion analysis and ITV generation

The 4D workflow reproduces, inside the viewer, the pipeline of ITV-based
motion studies (e.g. upright-vs-supine STAR target evaluation): per-phase
registration, target propagation, centroid motion metrics, target–reference
drift and correlation, and ITV volumes.

## 4D groups (`src/fourd.rs`)

A 4DCT arrives as one image series per respiratory phase, usually with an
average and sometimes MIP/MinIP reconstructions beside them; DICOM stores
no node for the acquisition they form, so the viewer reconstructs one:

- Series are bucketed by (study, modality). A series whose description
  carries a number directly before a `%` (e.g. `Thorax 4D 30%`, `CT 0 %
  Ex`) or `phase` + number (`4DCT_phase_000`, `Phase 3`) is a **phase**;
  the description minus the number (the *template*) tells two 4D sets in
  one study apart — thin- and thick-slice reconstructions become two
  groups — and is the group's name stem (`4D CT — Thorax 4D (10 phases)`).
- Series with a `TemporalPositionIdentifier` but no percent group by
  identical description and order by that identifier (`t1`, `t2`, …).
- `AVG`/`average`/`mean`, `MIP` and `MinIP` in the description mark the
  reconstructions; they attach to the bucket's first group.
- A group needs **at least three phases** — two series with "50%" in the
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

In the tree a group renders as a `🎞` node inside its study — phases in
temporal order, then the reconstructions; grouped series leave their
modality node so each series has one place.

## The pipeline (`src/app/motion_win.rs`)

*Tools ▸ 📈 Motion-analyse dataset A/B…*, or right-click a 4D group ▸
*Motion / ITV analysis…*. One run:

1. **Reference phase** — chosen in the dialog (default: the 0 % phase).
   Targets are defined on it: contours rasterized onto its lattice,
   segmentations from another lattice resampled onto it.
2. **Per-phase registration** — the reference volume is registered to
   every other phase with the elastix engine: a rigid stage and, for the
   deformable model, a B-spline refinement *started from* the rigid
   result. Settings (levels, iterations, samples, grid spacing, sampling
   threshold) come from the Registration panel, adjustable in the dialog.
3. **Propagation** — every target (and the reference structure, when
   chosen) is carried through each transform onto each phase's lattice,
   once per model; the transform maps reference → phase, so landing on
   the phase samples through the inverse, as in the propagation tool.
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
   edit them; they export like any segmentation — SEG or RTSTRUCT).
6. **Registration QA** — per phase and model: the engine's metric line,
   the 95th-percentile displacement, and the folding rate (fraction of
   sampled points with a non-positive Jacobian).

*Keep per-phase segmentations* additionally stores every propagated mask
as a segmentation series on its phase (`4D <phase> — <group>`).

Cancel stops the run at the next phase boundary; a finished run is never
applied to a dataset that was replaced while it ran.

### Recipes — several studies, one workflow

Starting a run remembers the dialog as a *recipe*: target and
reference-structure names, models, ITV options and registration settings.
*Apply last recipe* re-ticks the same structures **by name** in whatever
dataset the dialog is open on — load the next patient (or the paired
upright/supine study into dataset B), open the tool, apply, run. Recipes
are name-based on purpose: indices and UIDs do not travel between patients.

## Results (`src/app/motion_results.rs`)

The results window opens when a run finishes (later: *Tools ▸ 📈 Motion
results…*). Per run: the displacement-magnitude-vs-phase chart (targets ×
models, reference structure dashed red), peak-to-peak amplitude and drift
bars, the per-phase table (|d| and volume per track), correlation lines
(r, p, significance stars, synchrony wording), registration-QA lines and
ITV volumes.

**Compare with** puts a second run beside the first — dataset A vs. B,
upright vs. supine — matching ITVs and tracks *by target name and model*:
ITV volumes with percentage change, peak-to-peak amplitudes, side by side.

**Export CSV** writes one long-format CSV (a `table` column separates the
sections: per-phase centroids and displacements, peak-to-peak rows,
correlations, QA, ITVs); a comparison appends the second run's rows.

## Transfer by relationship (`src/app/transfer_win.rs`)

*Tools ▸ ◎ Transfer by relationship…* places a structure of one dataset
into the other at the same **offset from a reference structure's
centroid** — the STAR workflow's target–heart relationship: a target is
projected into a dataset registration cannot reach (another patient,
another posture) via anatomy both datasets can segment. The target keeps
its shape; the tool reports the offset (RL / AP / SI) it applied.
Reference structures whose name contains "heart" are pre-picked.

## Compare structures (`src/app/compare_win.rs`)

*Tools ▸ ◑ Compare structures…* computes, for any two structures (either
dataset, contours or segmentations, different lattices): volumes, centroid
offset (vector and magnitude), Dice, 95th-percentile symmetric Hausdorff
distance and mean symmetric surface distance. The second mask is resampled
onto the first's lattice through patient coordinates; across two frames of
reference the window notes the comparison assumes corresponding coordinates.

## Numerics worth knowing

- Centroids are exact under the affine lattice→patient map (mean index,
  then map); peak-to-peak is the largest pairwise distance, independent of
  the reference-phase choice.
- The p-values come from the regularized incomplete beta function
  (continued-fraction evaluation) — the exact t-distribution tail, not a
  normal approximation; with 10 phases n is small enough for that to matter.
- HD95/MSD use the exact anisotropic Euclidean distance transform
  (`morphology::dist2_to_foreground`) evaluated on surface voxels of each
  mask against the other, both directions pooled for the percentile.
- ITV volumes inherit every caveat of nearest-neighbour resampling between
  phase lattices; centroid metrics are the primary motion descriptors, as
  in the underlying study design.
