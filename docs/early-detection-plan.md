# Plan: early cancer detection on CT in RDS

*A working plan, not user documentation — the same role `medsam2-plan.md`
played for the third engine. Written 2026-08-28 against the refactored tree
(shared `nn/`, `progress.rs`, `models.rs`, `app/seg_engines.rs`).*

Everything here stays inside the project's constraints: pure Rust, weights
downloaded on request into `models/`, every port validated numerically
against its reference, research / QA use — explicitly not a medical device
and not a clinical screening product. "Early detection" in this plan means:
**find** small lesions a reader can miss, **quantify** them reproducibly,
and **track** them across timepoints — the three things that decide whether
a screening-round CT gets a timely follow-up.

## 1. What RDS already has

A CT early-detection workflow has four stages — *find, delineate,
quantify, follow* — and the codebase already contains the second and most
of the raw material for the fourth:

| Stage | Status | Where |
|---|---|---|
| Find (CADe: propose suspicious locations) | **missing** | — |
| Delineate | done | SegVol (box/point/text), MedSAM2 (box + propagation), brush/grow correction |
| Quantify (volume, diameter, density, growth) | partial | `Segmentation::count`/`volume_cm3`, spacing-aware masks; no per-lesion analysis |
| Follow (register timepoints, match lesions, growth rate) | raw material done | elastix-style rigid + B-spline registration, comparison mode, REG import |
| Risk (per-scan future-cancer probability) | missing | — |

And the infrastructure a new engine needs is now generic: torch-checkpoint
reading and safetensors caching (`nn/cache.rs`), one model folder with
per-engine sub-folders (`models.rs`), device choice with a validated wgpu
context (`nn/device.rs`), shape-checked parameters (`nn/params.rs`), a
`burn` backend-generic engine pattern (the whole of `medsam2/`), the
background-job and progress plumbing, and the shared tool-window skeleton
(`app/seg_engines.rs`) that makes a fourth tool cheap to add and
automatically consistent.

The strategic point: **RDS does not need to become a CAD product to be
useful for early detection — it needs a *finder* and a *follow-up ledger*
around the delineation tools it already has.** A detector that proposes a
box is exactly a MedSAM2/SegVol prompt; a registration that already
achieves sub-voxel recovery is exactly a lesion matcher.

## 2. Methods reviewed, and what is actually portable

Focus: CT. Criteria: published reference implementation, weights openly
downloadable (the SegVol/MedSAM2 handling — fetch on request — is fine),
architecture portable to the existing `nn/` + `burn` stack.

### Tier 1 — lung nodule segmentation via the stack we already run

TotalSegmentator ships subtasks beyond `total`, and its **`lung_nodules`**
task (lung + lung_nodules classes, trained on 1353 subjects partly from
LIDC-IDRI, contributed by BLUEMIND AI) is listed in the **openly available,
Apache-2.0** group — same licence as the `total` weights RDS already
fetches. `kidney_cysts` (left/right) is in the same open group. These are
ordinary nnU-Net models: the existing `autoseg` engine (plans.json parsing,
CPU im2col+GEMM, burn/wgpu path, sliding window) should run them with a new
`ModelSpec` and class table — no new network code.

*To verify in P0*: that the `lung_nodules` weights zip is on the public
release channel the viewer downloads from (the open tasks are; the
licensed tasks come from the TotalSegmentator backend with a key), and its
plans/spacing. One probe run of the Python original for reference outputs.

### Tier 2 — dedicated lung-nodule detector (boxes + scores)

Segmentation-as-detection (Tier 1) misses what a purpose-built CADe is
optimized for: sensitivity at low false-positive rates on *small* nodules.
The **MONAI `lung_nodule_ct_detection` bundle** is the portable reference:
RetinaNet-style 3-D detector (ResNet backbone + FPN + classification/box
heads + anchors), input resampled to 0.703125 × 0.703125 × 1.25 mm,
192 × 192 × 80 patches, trained on LUNA16 (LIDC-IDRI), **Apache-2.0**,
weights published (`models/model.pt` on the MONAI model zoo / NGC / HF
mirror). Port effort is comparable to a slice of the MedSAM2 work: ResNet
conv blocks and FPN are routine on `burn` (conv3d is already used by the
autoseg GPU path); the genuinely new pieces are small and testable — anchor
generation, box coding, and 3-D NMS.

nnDetection would be the stronger framework, but it publishes **no trained
LUNA16 weights** (you must train the 10 folds yourself), so it is not
portable under this project's rules. The DSB-2017 winners are old 2-D/2.5-D
pipelines; skip.

### Tier 3 — per-scan future-risk (the actual "early" in early detection)

**Sybil** (MIT Jameel Clinic / MGH, *J Clin Oncol* 2023): predicts 1–6-year
lung-cancer risk from a **single LDCT**, no nodule annotation required at
inference. Architecture per the paper and its audit literature: a 3-D
ResNet-18-style encoder over the whole volume with max-pooling and a
guided-attention branch, a cumulative-hazard head emitting the six year
risks; distributed as an ensemble of five checkpoints. Code is **MIT**;
checkpoints are on the repo's GitHub releases. This is squarely portable
with the `medsam2/` pattern (backend-generic modules, layout inventory,
op-parity fixtures). It is also the most caveat-laden piece: validated on
NLST-like screening LDCT; outputs must be shown as research numbers with
the cohort caveat attached, never as a recommendation.

### Rule-based layer — no ML, highest confidence per line of code

The screening trials' own decision logic is arithmetic on masks, and RDS
can compute it exactly and reproducibly:

* **Volumetry** per nodule (connected component): volume in mm³, longest
  axial diameter and mean diameter, mean/min/max HU, location by lobe
  (from the `total` lung lobes already segmented).
* **NELSON-style volume management**: < 100 mm³ negative, 100–300 mm³
  indeterminate → short-interval follow-up, > 300 mm³ positive; growth
  assessed by **volume-doubling time**, VDT < 400 days suspicious
  (thresholds as published by the NELSON protocol / NEJM 2020 / Radiology
  2024 review; cite, and keep them as data, not hard-code).
* **VDT** from two timepoints: `VDT = Δt · ln2 / ln(V2/V1)` — the follow-up
  ledger's core number.
* Optional later: a small **radiomics** module (first-order, shape, GLCM)
  on any segmentation, exported as CSV — cheap in pure Rust, deterministic,
  and useful for research even without a classifier on top.

(Lung-RADS is ACR-copyrighted material; present RDS's output as
measurements plus the NELSON volume categories with citations, and let the
user do the Lung-RADS assignment — or add it later as a clearly referenced
optional table.)

### Watch list — not portable today

* **PANDA** (pancreatic cancer on non-contrast CT, Alibaba DAMO, *Nature
  Medicine* 2023): the flagship result for a silent-killer cancer, a
  three-stage nnU-Net cascade — but trained weights are not openly
  downloadable. Re-evaluate if they publish; the Tier-1 machinery would
  carry stage 1–2 naturally.
* **CT foundation models** (e.g. CT-FM, open 3-D SSL model trained on
  ~150k CTs): not a detector by itself; interesting later as a backbone for
  fine-tunes, if a concrete downstream checkpoint appears.
* Abdominal opportunistic finds (liver/renal lesions): no open weights of
  PANDA's calibre yet; `kidney_cysts` from Tier 1 is the near-term token.

## 3. How it fits together in the app

One workflow, four existing pieces and two new ones:

```
                        ┌───────────────────────────── new ─┐
CT loaded ─▶ 🔍 Detect (Tier 1/2) ─▶ Findings list (per-nodule:
                                     volume, diameter, HU, lobe, score)
                                           │ click = jump crosshair
                                           │ "⏩/🧠 Segment" = box prompt
                                           ▼
             MedSAM2 / SegVol refine ─▶ editable Segmentation  (existing)
                                           │
             follow-up study in slot B ────┤ registration (existing)
                                           ▼
             📈 Follow-up: matched findings A↔B, ΔV, VDT,     (new)
                NELSON category, report export (CSV/Markdown,
                masks ▶ RTSTRUCT as today)
```

The detector's box becomes a MedSAM2 prompt *mechanically* — `box_seg.rs`
already turns a rectangle on a slice into `EnginePrompt::Points`, and the
finding carries exactly that rectangle. Matching across timepoints is the
existing transform applied to finding centroids (nearest neighbour with a
distance gate), so the follow-up table needs no new registration code.

## 4. New modules (module-map deltas)

```
src/
  findings.rs       Finding {centroid, bbox, score, volume, diameter, HU,
                    lobe, matched_to}: connected-component extraction from
                    label masks, measurement, NELSON categories, VDT,
                    A↔B matching through a Transform3              Seg/Core
  detect/           RetinaNet-3D lung-nodule detector (MONAI bundle port,
                    backend-generic like medsam2/): layout, config,
                    weights, backbone, fpn, heads, anchors, nms, infer,
                    engine                                             Seg
  risk/             Sybil (5-model ensemble, 3-D encoder + attention,
                    cumulative-hazard head): same file pattern          Seg
  app/
    findings_panel  sidebar section + view markers + jump/segment/dismiss
    screening.rs    🔍 and 📈 tool windows on the seg_engines skeleton
```

`models/` grows `detect/` and `sybil/` sub-folders (one `models::Engine`
variant each); the TS `lung_nodules` weights live where the other TS models
do, `models/totalsegmentator/lung_nodules/`. Licence lines in the tool
windows follow the existing pattern (Apache-2.0 for Tier 1/2 —
prefetchable by the installer like `total`; Sybil MIT but fetched on
request like the others).

## 5. Phases

* **P0 — TS `lung_nodules` as a task of the existing engine.** Verify the
  open-release URL and plans; add the `ModelSpec`, class entries and a
  "Lung nodules" model choice in the 🤖 window; masks land as
  segmentations. Smallest possible CADe. *(days)*
* **P1 — findings.** `findings.rs` + sidebar findings list + view markers:
  split the nodule mask into components, measure (volume, diameters, HU,
  lobe via `total`), sort by size, one-click box-prompt hand-off to
  MedSAM2/SegVol, NELSON volume category per finding. Closed-form tests on
  synthetic spheres from `gen_test_data` (known volume/diameter to
  sub-voxel tolerance). *(≈1 week)*
* **P2 — follow-up.** Match findings A↔B through the active registration,
  ΔV and VDT, growth table in a 📈 window, CSV/Markdown report export.
  Test on `simulate.rs` studies with a known transform and a grown bump
  (the generator already supports a Gaussian bump — give it a size
  parameter and the VDT is analytic). *(≈1 week)*
* **P3 — RetinaNet detector port** (MONAI bundle): probe script dumps
  reference activations (same method as `tools/gen_reference_activations.py`),
  layout inventory, ops (anchors, box coding, NMS) with parity fixtures,
  backbone+FPN+heads on `burn`, sliding-window over the lung mask only
  (the `total` lungs gate the search volume — cheaper and fewer FPs),
  score-thresholded findings into P1's list. Offline FROC sanity check on
  a LUNA16 subset, not in CI. *(2–4 weeks, the big one)*
* **P4 — Sybil port**: layout (5 checkpoints), encoder + attention +
  hazard head, ensemble average; preprocessing pinned against the
  reference implementation exactly as MedSAM2's was; a ⚠ risk read-out
  with the cohort caveat in the window and in the docs. *(2–3 weeks)*
* **P5 — optional**: `kidney_cysts` task (config only), radiomics CSV
  export, PANDA/foundation-model watch.

P0→P2 deliver a working find-quantify-follow loop with almost no new
network code; P3/P4 upgrade the *find* and add *risk* and are independent
of each other.

## 6. Validation and honesty

Same discipline as the three engines: a Python reference script per port in
`tools/` producing fixtures/activation dumps; parity asserted to ~1e-5;
synthetic-phantom closed-form tests for everything rule-based; the real
datasets (LUNA16, screening cases) used offline, never fetched by tests.
Every window carries the existing licence + "Research / QA use — not a
medical device" line; the risk window additionally names the training
cohort. Nothing here claims a clinical detection performance — the docs
report the upstream models' published numbers and our parity with them,
which is the same honest posture the auto-segmentation docs already take.

## 7. Sources

TS subtasks incl. open `lung_nodules`: github.com/wasserth/TotalSegmentator
(README, Subtasks). MONAI detection bundle: MONAI model zoo
`lung_nodule_ct_detection` (Apache-2.0, LUNA16, RetinaNet 3-D). Sybil:
github.com/reginabarzilaygroup/Sybil (MIT; JCO 2023
doi:10.1200/JCO.22.01345). NELSON management: NEJM 2020
doi:10.1056/NEJMoa0906085 protocol lineage; Radiology 2024
doi:10.1148/radiol.240535. PANDA: Nature Medicine 2023
doi:10.1038/s41591-023-02640-w (weights not open). nnDetection:
github.com/MIC-DKFZ/nnDetection (no released LUNA16 weights). CT-FM:
arXiv:2501.09001, github.com/project-lighter/CT-FM.
