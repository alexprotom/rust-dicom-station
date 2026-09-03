# Plan: RayStation-class contouring in RDS

*A working plan, not user documentation — the same role `medsam2-plan.md` and
`early-detection-plan.md` play for their features. Written 2026-09-01 against
`0e6f8de`, from* RSL-D-RS-2024B-USM-EN-1.0-2024-06-25, RayStation 2024B User
Manual *(chapter 3.4 — the ROI/POI lists and dialogs; chapter 5.3 — the
Structure Definition module; 5.4.10/5.4.13/5.4.17 — geometry statistics,
structure mapping and simulated organ motion).*

The question this plan answers: **which of RayStation's structure tools can RDS
have, and in what order.** Constraints unchanged — pure Rust, CPU-first with
the GPU as an option, every numeric claim covered by a test, research / QA use,
explicitly not a medical device.

Two decisions were taken before writing, and everything below follows from
them:

1. **Scope: only what is realistic in RDS.** Anything that presupposes a clinic
   database, a user-account system, a licensed model library or a dose engine
   is listed once with its reason (§3) and then dropped.
2. **Representation: RDS grows a real contour model.** Today every editing tool
   in RDS works on voxels and RTSTRUCT ROIs are read-only geometry that gets
   rasterized in and marching-squared out. That is the single biggest
   behavioural gap against RayStation, and closing it is Phase 0–2 of this
   plan rather than a footnote.

---

## 1. What RayStation has

Verdicts: **have** — already in RDS; **plan** — in this document with a phase
number; **out** — §3, with the reason.

### 1.1 Structure administration and metadata (USM 3.4.1, 3.4.2, 3.4.5, 5.3.2)

| RayStation function | Verdict |
|---|---|
| ROI list with per-ROI visibility, three-state group visibility | **have** — sidebar *RT structures*, All/None, per-item ticks |
| ROI geometry status per image set (defined / not defined / derived / MBS / DLS badge) | **plan** (P5 for the derived badge; the half-filled colour square already exists for segmentations) |
| ROI properties: name, type, colour | **have** — rename at every tree level (`app/rename.rs`), colours from the file or the palette |
| ROI interpreted type (GTV/CTV/PTV/Organ/External/Bolus/Support/Fixation/Avoidance) | **have** on write (`seg_to_rtstruct` takes the type; the Combine window offers it), **plan** as an editable property of an existing ROI (P2) |
| ROI/POI details table: representation, absolute volume, min / max / mean intensity per geometry | **plan** (P7) |
| Copy ROI (new or existing ROI, chosen image sets) | **have** in part — tree copy/move of whole sets and items (`app/sets.rs`); **plan** for per-ROI copy inside one dataset (P2) |
| Delete ROI vs. delete geometry | **have** (item remove) |
| Convert to contours (explicit representation change) | **plan** (P0 — the representation switch is the point of Phase 0) |
| Localize ROI (centre all views on it), Pick tool (click an ROI to make it current) | **plan** (P1, both trivial once ROIs are editable objects) |
| ROI statistics over a 4DCT group (volume, voxels, mean/min/max grey level per phase) | **plan** (P7 — `fourd.rs` already groups the phases) |
| *Exclude from export* flag on an ROI or POI (USM 3.4.2) | **plan** (P2 — one bool honoured by `dicom_export.rs`, which already writes the whole chain) |
| Material override, patient-specific / common materials, RBE cell type, beam-set inclusion | **out** |

### 1.2 Geometry representations (USM 5.3.2)

RayStation carries five: *contours*, *voxels*, *triangle mesh*, *MBS mesh*,
*open triangle mesh*, and converts between them automatically depending on the
operation ("for contouring the type has to be Contours, for Create Expanded ROI
the type has to be Voxels").

RDS has voxels (`Segmentation`), contours (read-only `rtstruct::Roi`) and
triangle meshes (surface nets, for the 3-D view only). **Plan (P0):** promote
contours to a first-class *editable* representation and adopt RayStation's
rule — the operation decides the representation, and the conversion is
automatic but visible. MBS meshes and open meshes are **out**.

### 1.3 Manual contouring — the 3-D tools (USM 5.3.3)

| RayStation tool | Verdict |
|---|---|
| Deform (3-D, falloff with distance from the interaction point) | **plan** (P8, on the mesh; a 2-D nudge lands earlier in P2) |
| Translate / Rotate / Scale a whole ROI in 3-D | **plan** (P2 as affine transforms of the contour stack — no mesh needed) |
| Region growing, 3-D, with threshold slider, limiting box and limiting ROI | **have**, and better — the ✨ Grow tool is a geodesic front with robust seed statistics, not a plain threshold; **plan** the *limiting ROI / limiting box* option (P6) |
| Delete component (3-D) | **plan** (P2) |

### 1.4 Manual contouring — the drawing tools (USM 5.3.3)

| RayStation tool | Verdict |
|---|---|
| Brush, with Auto / Draw / Erase modes and automatic hole removal | **have** as a voxel brush (3-D spherical, capsule-swept, spacing-aware); **plan** (P1) the contour-native behaviour: a stroke becomes a polygon boolean against the slice |
| Smart brush — edge-aware, five edge settings (None/Bone/Dark/Air/Bright), sensitivity, level/window dependent | **plan** (P4) |
| Smart interpolation (interpolate hand-drawn contours to a 3-D ROI, snap the surface to image edges, live update) | **plan** (P4) |
| Smart contour (live-wire that snaps to the grey-level gradient; training mode, lasso mode) | **plan** (P4) |
| Spline, Polygon, Freehand, each in Auto / Extend / Subtract mode | **plan** (P1) |
| Slice-aligned contouring on oblique image sets | **plan** (P1 — the contour plane comes from the volume's own direction cosines) |

### 1.5 Interpolation (USM 5.3.3)

Show interpolation (dashed, on slices with no drawn contour, in all three
views), Accept current, Accept all, recomputed on every edit, never saved
unless accepted. **Plan (P3)** — in full, including the "not saved unless
accepted" rule.

### 1.6 2-D editing and the extra tools (USM 5.3.3)

| RayStation tool | Verdict |
|---|---|
| 2-D region growing | **have** (the Grow tool is 3-D; a slice-confined mode is P6) |
| 2-D Deform / Translate / Rotate / Scale of a single contour | **plan** (P2) |
| Interpolate (fill missing contours linearly between the extreme drawn slices) | **plan** (P3) |
| Copy contour / Paste contour | **plan** (P2) |
| Delete contour; delete multiple contours keeping every 2nd/3rd/5th/n-th, over a slice range | **plan** (P2) |
| Remove holes (current slice / all slices) | **plan** (P2; the voxel equivalent is `morphology::fill_holes_2d/3d`) |
| Simplify contours — resolve overlap conflicts, remove holes, remove contours under an area, reduce point count | **plan** (P2) |
| Delete component / Keep component (3-D connected pieces, picked in a view) | **plan** (P2; `morphology::components` for masks, slice-overlap graph for contours) |
| Move to slice intersection | **plan** (P2) |
| Couch removal (cut the External below a line picked in the coronal view) | **plan** (P6; RDS's 👤 body tool already excludes the couch, so this is the manual fallback) |

### 1.7 Automatic and semi-automatic generation (USM 5.3.4, 5.3.6–5.3.8, 5.3.10–5.3.14)

| RayStation function | Verdict |
|---|---|
| Generate External ROI (threshold → largest object → hole removal) | **have**, and beyond it — `bodymask.rs` has a geometric and a model-assisted method, couch/mask/chair removal, CT and MR, supine and upright ([body-contour.md](body-contour.md)) |
| Deep Learning Segmentation (model library, per-model ROI name/colour aliases) | **have**, three engines — TotalSegmentator (117 classes), SegVol (box/point/text), MedSAM2 ([auto-segmentation.md](auto-segmentation.md), [segvol.md](segvol.md), [medsam2.md](medsam2.md)); **plan** the name/colour alias table (P9) |
| Create Bone ROI (three HU presets) | **plan** (P6 — a preset of the grey-level threshold tool) |
| Gray Level Threshold ROI (with SUV units for PET) | **plan** (P6) |
| Volume Threshold ROI | **plan** (P6) |
| Basic shapes: box, cylinder, sphere, ellipsoid | **plan** (P6) |
| Create ROI from dose (threshold on a computed dose) | **plan** (P6 — RDS loads RTDOSE and samples it trilinearly already) |
| Field-of-View ROI (brute-force centre detection / minimum covering circle) | **plan** (P6) |
| External ROI on limited-FOV data | **plan** (P6, without the material override half) |
| Lung vessel segmentation, five detail levels | **plan** (P10, as a Frangi/Sato vesselness inside a lung ROI — the tool is realistic, the parity with RayStation's is not) |
| Model-Based Segmentation: shape models, hint contours, model flexibility, MBS Model Manager | **out** |
| Structure templates (create, apply, import/export, DLS bindings, derived expressions) | **plan** (P9, as a local template store) |
| Atlas-based segmentation (multi-atlas, rigid + deformable, label fusion) | **plan** (P10 — every piece exists: local PACS, rigid + deformable registration, propagation; what is missing is the atlas library and the fusion vote) |
| Import brachy applicator model (XML) | **out** |

### 1.8 Derived ROIs, margins and algebra (USM 5.3.5)

| RayStation function | Verdict |
|---|---|
| Expand / contract, uniform or per-direction, with a limiting ROI | **have** except the limiting ROI (`structops::Margin`, six patient directions, exact anisotropic EDT); **plan** the limiting ROI (P5) |
| Create wall (outward + inward distance) | **have** as a recipe (ring = two expansions subtracted); **plan** as a named one-click tool (P5) |
| ROI algebra: `Result = Margin(Op(Margin(Op(sources)), Margin(Op(sources))))` | **have**, in a more general form — an ordered operand list with a margin per operand, folded left to right ([structure-algebra.md](structure-algebra.md)) |
| Algebra computed slice-wise on **contours**, margins on **voxels** | **plan** (P5) — RDS does everything on voxels today; the manual's split is the right one and Phase 0 makes it possible |
| Derived ROI: saved expression, auto-update, status *up to date / needs update / overridden*, Edit / Update / Underive / Override | **plan** (P5) — the largest single feature in this plan after the contour model |
| Create ITV (union over a 4D group, with a margin) | **have** — the 📈 4D pipeline builds ITVs per target and model ([motion-4d.md](motion-4d.md)) |
| Create beam-specific margin ROI (distal / proximal / radial to a beam) | **plan** (P10 — RDS parses beam geometry from RTPLAN, so it is possible read-only) |
| Create 1-view margin ROI (USM Appendix K.5): expand a target by a motion vector taken from the centre-of-mass shift across a 4DCT group, then add imager-view margins | **plan** (P10 — the 4D centroids are already computed by `motion.rs`; the imager frame is the only new geometry) |

### 1.9 Across image sets (USM 3.4.1, 5.3.6, 5.4.13, 5.4.17)

| RayStation function | Verdict |
|---|---|
| Copy ROI geometries to other image sets through a rigid registration | **have** in substance (⇄ Propagate handles rigid and deformable), **plan** the missing halves: land the result as an *ROI* rather than a segmentation, and offer it from the ROI context menu (P7) |
| Copy/Map ROI(s) and reversed, from the ROI list, using the selected registration | **plan** (P7) |
| Map structures deformably (direction, new ROI vs. new geometry, grid auto-expansion) | **have** ([propagation.md](propagation.md)); **plan** the naming convention and the "new geometry for an existing ROI" output (P7) |
| Simulated organ motion (a motion ROI drives a deformation, fixed ROIs pin it, everything else is mapped) | **plan** (P10 — `simulate.rs` and the landmark warp are the raw material) |

### 1.10 Evaluating geometry (USM 5.4.10)

Volume in both sets; number of mesh points; centroid; least-squares
translational and rotational offset; mesh-point distance (mean/SD/max) after
rigid registration; mesh-point difference after the deformable one; **Dice
similarity** in both directions; POI distance, displacement and target
registration error. **Plan (P7)** — plus mean surface distance and HD95, which
RayStation does not report and which every contouring study asks for.

Two smaller things live here too. **Show dose on ROI surface** (USM 5.3.1, the
3-D view context menu) is cheap in RDS — the surface-net mesh exists, RTDOSE is
already sampled trilinearly in patient space, so it is one colour-map lookup per
vertex; **plan (P7)**. And **dose statistics invalidation** (USM 8.1.18, *Update
ROI voxel volumes*): RayStation blanks a structure's dose fields when its
geometry changes and offers an update button. RDS's DVH window recomputes
silently and has no staleness marker at all — the same rule, stated the same
way, belongs in it; **plan (P7)**.

### 1.11 POIs and workflow (USM 5.3.15–5.3.17)

| RayStation function | Verdict |
|---|---|
| POIs: create (manual, or at an ROI centre), type, colour, diameter, Localize, Move, Move to slice intersection, per-image-set geometries | **plan** (P8) — RTSTRUCT `POINT` contours are already parsed and drawn as markers, so this is mostly UI, editing and export |
| Localization point for patient setup (one per image set, golden badge) | **plan** (P8) |
| Approval of the structure set (authenticated, read-only afterwards, unapprove) | **plan** as *lock*, **out** as *authentication* — §3 |
| Image fusion display modes (overlay, checkers, blinds, spy glass, difference) | **have** (fusion overlay and comparison modes; checkers/spy-glass are cosmetic gaps) |

---

## 2. What RDS has today

| Capability | Where |
|---|---|
| Voxel masks with per-stroke undo, series-scoped, resampled onto the displayed lattice | `segmentation.rs`, `dicomseg.rs`, `app/seg.rs` |
| 3-D spherical brush, capsule-swept; geodesic region grow with live preview | `app/mod.rs` (`SegTool`), `segmentation.rs` |
| RTSTRUCT read: names, colours, interpreted types, closed planar contours; native axial contours plus reconstructed sagittal/coronal silhouettes | `rtstruct.rs`, `render::roi_on_plane` |
| Mask → RTSTRUCT (marching squares, stitched, decimated) and RTSTRUCT → mask (`rasterize_roi`) | `segmentation.rs` |
| Boolean algebra + patient-direction margins + cleanup over any mix of ROIs and masks | `structops.rs`, `morphology.rs`, `app/combine_win.rs` |
| Exact anisotropic EDT, one-sided sweeps, components, 2-D/3-D hole filling, open/close | `morphology.rs` |
| Body / EXTERNAL contouring, two methods | `bodymask.rs`, `app/body_win.rs` |
| Three learned engines with a shared tool-window skeleton, model manager, GPU/CPU choice | `autoseg/`, `segvol/`, `medsam2/`, `app/seg_engines.rs` |
| Rigid + deformable registration, REG/DSR import and export, structure propagation | `registration/`, `propagate.rs` |
| 4D grouping, motion metrics, ITV generation, A/B comparison | `fourd.rs`, `motion.rs`, `app/motion_win.rs` |
| DVH and dose statistics against a structure's own lattice | `dvh.rs`, `app/dvh_win.rs` |
| DICOM export of the whole chain, SEG write, local PACS | `dicom_export.rs`, `archive.rs` |

**The gap in one sentence:** RDS can *compute* structures better than it can
*draw* them — everything a planner does with a mouse on a contour (draw it,
push it, interpolate it, tidy it, keep it up to date when its parent changes)
is missing, and that is what §4 onwards builds.

---

## 3. Out of scope, and why

| RayStation function | Why not |
|---|---|
| Model-Based Segmentation (shape models, hint contours, flexibility slider, Model Manager) | The value is the trained shape-model library — ~30–100 expert contour sets per organ, not published. Reproducing the algorithm without the models buys nothing; RDS's three learned engines cover the same clinical need. |
| Material override, patient-specific / common materials, RBE cell types, beam-set inclusion of support/fixation | These exist to feed a dose engine. RDS reads RTDOSE, it does not compute dose; the properties would be inert metadata. |
| Approval with authentication (user name / password, *Plan approval* user group) | Needs a user database and an audit trail. A local **lock** flag (P9) gives the read-only protection without pretending to be an electronic signature. |
| Brachy applicator model import | A vendor XML specification (RSL-D-RS-2024B-BAMDS) for applicator libraries RDS has no use for. |
| Structure template import/export in `.rsbak` | Proprietary backup container. RDS's templates are JSON in its own data folder. |
| Snapshots / report modules | A reporting product, unrelated to contouring. |
| Open triangle mesh ROIs (STL via scripting) | Only creatable by scripting, cannot be converted or exported, no volume. A curiosity. |

---

## 4. The representation change (the heart of the plan)

Today: `rtstruct::Roi { contours: Vec<Contour> }`, points in patient
coordinates, no slice structure, no orientation convention, no nesting, no
editing. Every operation that touches an ROI rasterizes it (`rasterize_roi`)
and, if something must come back, walks marching squares over voxels. On a
1 mm CT nobody notices; on 3–5 mm slices the round trip is a staircase, and
the drawn polygon a user spent a minute on is gone.

### 4.1 The model

New module `src/contours.rs` (UI-free, like `structops.rs`, so the tests drive
it directly):

```rust
pub struct Poly { pub pts: Vec<[f64; 2]> }      // plane mm, closed implicitly
pub struct SlicePolys { pub level: f64, pub polys: Vec<Poly> }
pub struct Stack {                               // one ROI geometry
    pub frame: PlaneFrame,                       // origin + u,v,n from the volume's cosines
    pub slices: Vec<SlicePolys>,                 // sorted by level
}
```

* **Orientation is meaning.** Counter-clockwise in the (u,v) frame is solid,
  clockwise is a hole; ingest normalizes by signed area and even–odd nesting
  depth. RayStation handles three levels of nesting and ignores deeper ones
  (USM 5.3.5, *Algorithms used for ROI algebra*); RDS keeps arbitrary depth in
  the model and matches the three-level guarantee only where the algorithm
  requires it.
* **Slice binding** is by level with a tolerance of half the slice spacing, so
  a contour imported at 2.5001 mm lands on the 2.5 mm slice instead of
  creating a phantom level.
* **The plane comes from the volume**, not from the axial assumption — the
  `Slice aligned` behaviour RayStation requires for oblique sets (USM 5.3.3)
  falls out.

### 4.2 Which representation is authoritative

Follow the manual (USM 5.3.2) rather than inventing a rule: **the operation
chooses**, the conversion is automatic, and the current representation is
visible in the ROI properties.

| Operation | Representation | Why |
|---|---|---|
| Drawing: polygon, spline, freehand, brush, live-wire, nudge, per-slice edits | contours | It is what the user sees and what RTSTRUCT stores |
| Interpolation between slices | contours in, contours out (via a 2-D signed-distance field) | §5, Phase 3 |
| Boolean algebra | contours, slice-wise | The manual's own choice; keeps polygon fidelity |
| Margins, walls, morphological cleanup, distance transforms | voxels | Exact anisotropic EDT is a voxel algorithm; the manual does the same |
| Anything learned (the three engines), region growing, thresholds | voxels | The engines produce label maps |
| 3-D display, mesh statistics | triangle mesh | Already the case |

`Roi` gains `repr: Repr` and a `Stack` beside the raw `contours`;
`ensure_contours()` / `ensure_voxels(grid)` do the conversion once and cache
it. A voxel round trip marks the geometry *rasterized* so the UI can say so —
the honest version of RayStation's silent conversion.

### 4.3 Polygon booleans

Needed for the drawing modes (Auto/Extend/Subtract are boolean ops between the
stroke and the slice), for Simplify, and for contour-domain algebra. Options,
in the project's dependency-averse spirit:

1. **In-tree Greiner–Hormann with degeneracy handling** (perturbation-free
   variant), ~400 lines plus tests. Matches how `morphology.rs`,
   `render::marching_squares` and the surface-net mesher were done, and keeps
   the crate list short.
2. `i_overlay` or `geo`'s `BooleanOps` — pure Rust, MIT, well tested, and one
   more dependency each.

**Recommendation: (1)**, with the existing voxel algebra as the oracle in the
tests (agreement to a fraction of a percent of area on a fine lattice). If the
degeneracy work bogs down, (2) is a two-hour swap behind the same interface.

### 4.4 Undo

`Segmentation` journals voxels per stroke. Contours get the cheaper analogue:
before an edit, push the touched `SlicePolys` (a few hundred points) onto a
bounded journal. Same `Ctrl+Z`, same window, one shared trait so the sidebar
does not care which kind of thing is active.

---

## 5. The phases

Each phase is independently shippable, ends green, and adds a documented tool.
Sizes: **S** ≈ a day, **M** ≈ 2–4 days, **L** ≈ a week or more of focused work.

### Phase 0 — the contour engine (M)

*New:* `src/contours.rs` (model, ingest/normalize, orientation, nesting,
slice binding, boolean ops, area/centroid, Douglas–Peucker, affine transforms),
`tests/contours.rs`.
*Touched:* `rtstruct.rs` (`Roi::stack()`, `Repr`), `segmentation.rs`
(`rasterize_roi` and `mask_to_roi` go through `Stack`), `dicom_export.rs`
(write from the stack, orientation preserved).

*Done when:* a real RTSTRUCT loads → `Stack` → RTSTRUCT byte-identical in
geometry (point order may rotate, area and nesting may not); booleans agree
with the voxel algebra to < 0.5 % of area on an L-shaped test case with a hole
and an island; a feet-first and an oblique series both round-trip.

### Phase 1 — drawing tools (L)

*New:* `src/app/draw.rs` (tool state machine), `SegTool::{Polygon, Spline,
Freehand, Nudge}` + `DrawMode::{Auto, Extend, Subtract}`.
*Touched:* `app/mod.rs` (tool enum, keyboard), `app/views.rs` (rubber-band
preview, right-click to close, `X` to switch live-wire → polygon later),
`app/panels.rs` (a *Contours* toolbar group, current-ROI selector, Pick,
Localize).

Details that matter: Catmull-Rom for the spline (through its control points,
so the drawn points are on the curve); Auto mode's rule is RayStation's —
*new contour if disjoint; cut if the first point is outside; extend if the
first point is inside; keep the longest part* (USM 5.3.3); the brush becomes a
boolean of the swept capsule's outline with the slice when the target is an
ROI, and stays the voxel brush for segmentations.

*Done when:* one can draw an organ on 30 slices with polygon + brush, in any
of the three planes, on an oblique series, and export it as RTSTRUCT that
another viewer reads back identically.

### Phase 2 — per-contour editing and tidying (M)

Nudge (push the boundary with a falloff ring, the 2-D *Deform*), translate /
rotate / scale of a contour and of a whole ROI, copy / paste contour, delete
contour, delete every n-th over a range, remove holes (slice / all), simplify
(the four options — resolve overlaps by self-union, drop holes, drop contours
under an area, reduce points by binary search on the Douglas–Peucker
tolerance), delete / keep component (slice-overlap graph across the stack),
move to slice intersection, edit interpreted type, colour and the
*exclude from export* flag of an existing ROI.

*New:* `src/contourops.rs` + `tests/contourops.rs`, `app/contour_tools.rs`
(the window, per [detachable windows](architecture.md)).

### Phase 3 — interpolation (M)

Linear interpolation between drawn slices, shown dashed on every empty slice
in all three views, recomputed on every edit, **not saved unless accepted**;
Accept current / Accept all.

Implementation: per slice pair, rasterize both into a 2-D signed distance
field on the display lattice, interpolate linearly, extract the zero level with
the existing `render::marching_squares`. This handles the branching and the
partial-overlap cases the naive point-matching interpolation gets wrong, and
reuses code that is already tested. The manual's caveat ("assumes lateral
overlap") stays true and is stated in the UI.

*Done when:* five contours on every fifth slice interpolate to a smooth
structure whose volume is within a percent of the hand-drawn one, and
`Esc`/edit invalidates the preview without touching stored geometry.

### Phase 4 — the smart tools (L)

* **Live-wire (Smart contour):** Dijkstra over the slice's pixel graph with the
  Mortensen–Barrett cost (gradient magnitude, gradient direction, Laplacian
  zero crossing), level/window dependent as the manual specifies; training
  mode accumulates a gradient histogram from the last accepted segment; lasso
  mode closes the path live. The priority-queue machinery from the geodesic
  grow in `segmentation.rs` is the starting point.
* **Smart brush:** the stroke's disc, then an edge-constrained refinement in
  the same geodesic metric, with the five presets mapped to intensity bands
  (Bone / Dark / Air / Bright / None) and a sensitivity slider.
* **Smart interpolation:** Phase 3's interpolation followed by an edge-snapping
  pass on the interpolated surface, cancellable, progress-reported through
  `progress.rs`, restarting when new contours arrive.

### Phase 5 — derived ROIs (L)

The dependency machinery: `src/derived.rs` with an `Expr` tree (algebra,
expand/contract with an optional limiting ROI, wall, ITV), a content hash per
dependency, and the three statuses — *up to date*, *needs update*,
*overridden* — with Edit / Update / Underive / Override, exactly as USM 5.3.5
defines them. `structops::Recipe` becomes the serialized form of an `Expr`, so
the ◧ Combine window gains a **Derived** tick and nothing else changes.

Persistence: the expression is written into RTSTRUCT **ROI Description
(3006,0028)** with an `RDS:` prefix (ST, 1024 chars — enough for any realistic
recipe) so it survives export and re-import, with a sidecar JSON in the data
folder for anything that overflows. Other systems ignore the field; the
geometry is always present regardless.

Algebra moves to the contour domain here (slice-wise polygon booleans),
margins stay on voxels — §4.2.

### Phase 6 — generators (M)

Grey-level threshold (with SUV for PET), volume threshold, bone presets
(High/Medium/Low, the manual's HU pairs), basic shapes (box / cylinder /
sphere / ellipsoid, axis-aligned), ROI from dose (threshold on the loaded
RTDOSE, absolute or % of reference), FOV ROI (brute-force centre detection and
minimum covering circle, both), External on limited FOV, couch removal, and
the limiting-ROI / limiting-box option for the Grow tool.

Each is small; they share one dialog pattern and one entry in the *New ROI
geometry* menu that this phase also introduces.

### Phase 7 — cross-dataset and geometry statistics (M)

* Copy/Map ROI(s) and the reversed direction from the ROI context menu, using
  the active registration; output as ROI or segmentation, into a new ROI or as
  a new geometry for an existing one.
* A **Geometry statistics** window: volume in both sets, centroid, least-squares
  translational and rotational offset (Procrustes on the surface-net vertices),
  surface-point distance mean/SD/max, Dice in both directions, plus mean
  surface distance and HD95. CSV export, per the DVH window's habit.
* ROI/POI details: representation, volume, min/max/mean intensity per geometry;
  the 4D variant of the same table over a phase group.
* Dose on the ROI surface in the 3-D window, and a staleness marker on the DVH
  and metrics table when a structure is edited after the curves were computed
  (USM 8.1.18's rule, without pretending the dose itself is invalidated —
  RDS does not compute dose).

### Phase 8 — POIs (M)

`Poi { name, type, colour, diameter, per-series position }` in `rtstruct.rs`
(`POINT` contours already parse, and `render.rs` already draws them as markers), a POI list beside the ROI list, create /
localize / move / move-to-slice-intersection, POI at an ROI centre, the
localization-point rule (one per image set, marked), export as `POINT`
contours, target registration error in the Phase 7 statistics.
Then the 3-D interaction tools — deform / translate / rotate / scale on the
mesh — for ROIs whose editing is easier in 3-D.

### Phase 9 — templates and locking (M)

A local structure-template store (JSON under `%LOCALAPPDATA%\RustDICOMStation`,
per `settings::data_dir()`): ROI names, types, colours, derived expressions,
and *engine bindings* — a TotalSegmentator class list, a SegVol text prompt, or
"draw by hand". *Create structures from template* creates the ROIs, runs the
bound engines, then updates the derived ones — the practical replacement for
RayStation's DLS-in-template and MBS-in-template flows. Plus per-model ROI
name/colour aliases (USM 5.3.8) and a structure-set **lock** that makes
geometry read-only and shows in the tree.

### Phase 10 — the long tail (L, optional)

Atlas-based segmentation (an atlas library over the local PACS, rigid
pre-selection, deformable mapping of the best *k*, majority-vote fusion —
every component already exists), simulated organ motion, beam-specific and
1-view margin
ROIs from a loaded RTPLAN, lung-vessel segmentation by vesselness filtering.

---

## 6. Order, dependencies and effort

| Phase | Depends on | Size | Why here |
|---|---|---|---|
| 0 contour engine | — | M | Everything below needs it |
| 1 drawing tools | 0 | L | The visible gap; makes RDS usable for hand contouring |
| 2 contour editing | 0, 1 | M | Turns drawing into correcting, which is the real workload |
| 3 interpolation | 0, 2 | M | The single biggest time-saver per organ |
| 4 smart tools | 1, 3 | L | Quality-of-life on top of a working pipeline |
| 5 derived ROIs | 0, 2 | L | Changes how the whole structure set behaves; wants a settled contour model first |
| 6 generators | 0, 5 | M | Cheap individually, better once derived exists |
| 7 cross-dataset + statistics | 0 | M | Independent of 1–6; could be pulled forward if evaluation is the priority |
| 8 POIs + 3-D tools | 0, 7 | M | POIs are needed by the statistics panel's TRE |
| 9 templates + locking | 5, 6 | M | Templates only pay off once derived ROIs and generators exist |
| 10 long tail | 5, 7 | L | Research-grade extras |

If the goal is *"contour a patient in RDS end to end"*, stop after Phase 3.
If it is *"reproduce a planning department's structure workflow"*, Phase 5 is
the one that cannot be skipped.

---

## 7. Risks and things to decide early

* **Degenerate polygons.** Self-intersections and coincident edges are where
  every clipper dies, and clinical RTSTRUCTs are full of both. Budget the test
  corpus first: the example data, plus deliberately pathological cases.
* **Contour ↔ voxel churn.** With both representations live, a sloppy call
  order can rasterize and re-contour repeatedly and quietly destroy detail.
  Mitigation: `Repr` is explicit, conversions are logged in debug builds, and
  a test asserts that a draw → margin → draw sequence rasterizes exactly once.
* **The 5 mm staircase is not a bug.** Any voxel-domain result on a coarse
  series looks worse than a drawn contour. Say so in the UI where it applies
  (the Combine window already does).
* **`app/panels.rs` tool dispatch.** The known issue from the 2026-08-30 review
  — the section receives `pat`/`stu` but dispatches on the *displayed* slot —
  becomes worse once there are ten more tools. Fix before Phase 1.
* **Test time.** `tests/` is already 14 suites; the contour work adds two or
  three. Keep them lattice-small and never assert on wall-clock time.

## 8. Verification

Per the project's habit — the algorithm's own unit tests, an integration suite
for the seam with the application, and an independent oracle wherever one
exists:

* **Phase 0:** RTSTRUCT round trips (area, nesting, orientation); polygon
  booleans against the existing voxel algebra; oblique and feet-first lattices.
* **Phase 1–2:** each drawing mode's rule as a table test on synthetic
  polygons; simplify's four options with known inputs; component keep/delete on
  a two-blob stack.
* **Phase 3:** volume of an interpolated sphere vs. the analytic one;
  interpolation never mutates stored geometry until accepted.
* **Phase 4:** live-wire path on a synthetic edge with noise; the smart brush
  must not cross a boundary the geodesic grow already respects (reuse the
  no-leak test).
* **Phase 5:** status transitions under a dependency edit; an expression that
  survives export → import; agreement between a derived ROI and the same
  recipe computed by hand.
* **Phase 6:** each generator against a phantom with a known answer (a sphere
  of known HU, a dose grid with an analytic profile — `tests/dvh.rs` already
  builds one).
* **Phase 7:** Dice and surface distances against analytically offset spheres;
  a rigid offset recovered by the Procrustes fit.

`cargo fmt --all` and `clippy --all-targets -D warnings` clean before every
hand-off, and the full suite green — including the medsam2 timing test, which
is slow but not flaky on a real machine.

---

## 9. Sources

RSL-D-RS-2024B-USM-EN-1.0-2024-06-25, *RayStation 2024B User Manual*: 3.4.1
(ROI list), 3.4.2 (ROI properties), 3.4.5 (ROI/POI details), 5.3.1–5.3.17 (the
Structure Definition module in full), 5.4.10 (geometry statistics and Dice),
5.4.13 (Map structures), 5.4.17 (simulated organ motion), 8.1.13 (Create ROI
from dose), 8.1.18 (Update ROI voxel volumes), Appendix K.5 (Create a 1-view
margin ROI). RDS side:
[segmentation.md](segmentation.md), [structure-algebra.md](structure-algebra.md),
[body-contour.md](body-contour.md), [propagation.md](propagation.md),
[rt-objects.md](rt-objects.md), [motion-4d.md](motion-4d.md),
[architecture.md](architecture.md).
