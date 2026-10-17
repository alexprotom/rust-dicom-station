# Synthetic CT from MR and CBCT

A CT synthesized from an MR or a cone-beam CT of the same patient, written as
a new CT series of the study on the same frame of reference, and a tool that
measures how far it is from a real CT. The network is TotalSynth (Boussot et
al., 2026), published as the ImpactSynth apps; it is re-implemented here in
Rust, recipe included, and checked against KonfAI, the framework the apps run
in.

What it is for: research and QA - MR-only and adaptive-CBCT workflows, dose
recalculation studies, the evaluation of synthesis models on a department's
own cases. It is not a medical device, and an sCT is not a planning CT until
the department's own acceptance test says so.

## The models

| App | From | What it was trained on | Held-out MAE (model card) |
|---|---|---|---|
| TotalSynth MR to CT (`sct_mr`) | MR of any sequence | SynthRAD 2023 and 2025, four prostate cohorts; brain, head and neck, thorax, abdomen, pelvis | 67.5 HU (pelvis 48, thorax 59, abdomen 60, brain 80, head and neck 81) |
| TotalSynth CBCT to CT (`sct_cbct`) | HU-calibrated linac CBCT | SynthRAD 2023 and 2025 | 53.6 HU |
| TotalSynth MR or CBCT to CT (`sct_mr_cbct`) | either | both | within a few HU of the dedicated ones |
| TotalSynth whole-body Dixon MR to CT (`sct_mr_dixon`) | Dixon MR | the MR model fine-tuned on 45 whole-body Dixon MRs (BIC-MAC) | 62 HU there, where the base model gives 101 |
| Your checkpoints (`sct_custom_<folder>`) | either | what you fine-tuned them on (KonfAI's `fine-tune`) | yours to measure |

Each app is five cross-validation checkpoints (`CV_0.pt` to `CV_4.pt`,
105 MB each) of one network: a 2.5-D UNet++ with a ResNet-34 encoder (26.1 M
parameters) that reads five neighbouring axial slices and writes the CT of
the middle one. The published recipe averages all five, each with two
test-time mirrors.

**Licence.** The weights are CC BY-NC 4.0 (their training data's terms):
non-commercial use only. They are downloaded to your machine from Hugging
Face (`VBoussot/ImpactSynth`) when you ask for them, the converted cache
beside them is a derivative under the same terms, and nothing is
redistributed. The model manager, the installer and the MCP server treat
them as the other non-commercial models ([auto-segmentation.md](auto-segmentation.md)).
A folder of your own checkpoints carries whatever licence its data allows.

## Using it

*Modules ▶ Image generation ▶ Image synthesis* (right panel, F10). The
section works on the module's workspace (the **Workspace** row above it,
shared with the DRR section) and its displayed series.

* **Model** - the app; the default follows the series' modality. A model
  offered for another modality than the series' says so in the warning
  colour.
* **Checkpoints** - *CV_0 (quick look)*, *all 5* (the published recipe), or
  the ones ticked.
* **Mirrors** - test-time augmentation: 0, 1 (also the slice flipped top to
  bottom) or 2 (then left to right as well: the published recipe). Members
  averaged = checkpoints x (1 + mirrors).
* **Spread of the members as a second series** - the voxel-wise standard
  deviation (HU) of the members, as its own series to fuse as an overlay.
* **Air outside** - what masks the air: the *body network* (the body
  contour guided by TotalSegmentator's 6 mm body network: `body` on CT and
  CBCT, `body_mr` on MR), the *body contour* by threshold and morphology
  (nothing to download), a structure of the workspace's active set (BODY,
  External), or nothing.
* **Show it in** - another workspace (the default: beside its source, on
  the same frame of reference, so the fusion needs no registration) or this
  one.
* The cost line says how many network passes a member makes and what is
  left to download. **Options** holds the compute device and the model
  folder.

**▶ Synthesize** runs in the background with the progress row and Cancel.
The result is a series of the source's study:

* CT Image Storage files under `<data folder>/derived/<study UID>/<series
  UID>/` (never beside the source, which may be read-only), read back
  through the loader, so the series can be shown again after another series
  was displayed, exported, archived, served by the PACS, and taken by every
  tool that takes a CT (segmentation, DVH, DRR, registration);
* the source's patient, study and Frame of Reference, a new Series Instance
  UID and the next Series Number, Image Type `DERIVED\SECONDARY`, a
  DerivationDescription naming the app, the checkpoints, the mirrors, the
  mask and the program's version, and a Source Image Sequence naming the
  source slice each image lies on;
* SeriesDescription `sCT (MR, 5 CV, TTA 2) from <source description>`.

### Evaluate against a CT

The *Evaluate against a CT* fold compares the displayed series (the sCT)
with a reference CT of the same patient:

* **Reference CT** - the series another workspace shows, or a CT series of
  the same study. It is resampled onto the sCT's grid (trilinear) through
  the active registration when one pairs the two images (either way round),
  or directly when they share a frame of reference. Across MR and CT, the
  registration to use is the one by structures
  ([registration.md](registration.md#registration-by-structures)):
  TotalSegmentator on both sides, then the images aligned on the organ
  surfaces, which no intensity metric does well across modalities.
* **Inside** - the reference's body contour (the default), a structure, or
  everywhere the reference has a value.
* **Water-equivalent thickness** - for proton work: ray by ray through the
  mask along the three axes of the grid, through an HU to relative stopping
  power table (`HU:RSP` pairs). The default table is a generic curve of the
  usual shape, not any scanner's calibration: the WET *difference* between
  two images of one patient depends little on it, the WET itself does, so
  paste your department's.
* **Dice of TotalSegmentator's structures** - TotalSegmentator (`total`,
  3 mm) on both images and the Dice of every structure it finds on the
  reference (1 cm³ or more), the model card's third metric (0.698 on its
  held-out set). Minutes on a CPU.
* **Add the difference as a series** - sCT minus CT, HU, on the sCT's grid,
  filed in the study like the sCT, for the fusion view.

The fold reports, inside the mask: MAE, mean error (sCT minus CT) and RMSE
overall and per class of the reference (air below -200 HU, soft tissue -200
to 200, bone above 200); PSNR and SSIM over the range the apps write,
[-1024, 3071] HU (4095 HU: both images clipped to it, outside the mask set
to -1024; SSIM as scikit-image computes it, a 7 x 7 x 7 uniform window with
sample covariance, K1 0.01, K2 0.03, reflected at the edges, averaged inside
the mask); the WET rows (mean, mean and 95th percentile of the absolute
difference, the relative difference over rays with 10 mm or more of water);
the Dice table. **💾 Export CSV** saves them (`section,item,metric,value`).

These say how two images differ. Whether a dose computed on the sCT is good
enough is a different question, with gamma rates and the department's
acceptance criteria; nothing here answers it.

## The recipe

The apps' `Prediction.yml`, step for step, as KonfAI (October 2026) runs it:

1. The scan in its LPS axis order with slices along S (any orientation is
   reoriented first; an axial series already is), cast through float16 and
   clipped to [-1024, 5000], int16.
2. Resampled (linear, ITK's rule) to 1 x 1 x 3 mm over the same extent:
   `n' = floor(n s / s')` voxels of the spacing that covers the extent,
   truncated to int16.
3. Clipped to [minimum, 99.5th percentile] (numpy's linear percentile) and
   mapped to [-1, 1].
4. Every slice with two neighbours on each side (reflected at the ends of
   the volume), in 512 x 512 patches padded with -1; a field of view wider
   than 512 voxels is tiled, the last tile aligned to the far edge and
   written over the overlap (KonfAI assembles patches without a blend).
5. The network's output kept in half precision, mapped back to HU as
   `((y + 1) / 2) x 4095 - 1024` in half precision step by step, int16;
   voxels outside the body mask (resampled nearest, dilated by five voxels
   of the working grid) set to -1024.
6. The mean over the members (float, then int16), resampled back onto the
   scan's grid (linear) and reoriented: the sCT has the source's dimensions
   and orientation.

The mirrors are KonfAI's at prediction time: the first copy flipped top to
bottom, the second left to right, each prediction flipped back before the
average. Upstream's body mask comes from its own ImpactSeg body network; RDS
substitutes the body contour or a structure (see *Air outside*). The spread
series is the standard deviation (n - 1) in HU; upstream writes the
variance as a percentage of a reference value instead.

The network works at 3 mm between slices; the sCT is written on the source's
grid but carries 3 mm information along S. Through-plane detail finer than
that is not there to be had.

A CBCT whose values are not HU still runs (the percentile normalisation
does not care), but its intensity distribution is not what the CBCT model
saw; the Evaluate fold is the judge.

## How long it takes

One network pass (one 512 x 512 stack) is 148 GFLOP. Measured on the
two-core validation machine: 2.6 s a pass on the CPU (PyTorch on the same
machine: 1.4 s). A pelvis at 3 mm is about 70 slices, a thorax about 120,
so a member is three to five minutes there, faster on a desktop with more
cores, and the full recipe (fifteen members) is a GPU job (wgpu: Vulkan,
DX12, Metal; upstream reports 24 s for one checkpoint and 82 s for the
five on a 24 GB GPU; the wgpu path has not been timed here). The quick look
(one member) is what a CPU-only machine runs; the section says the pass
count before the run.

## Headless

* MCP ([mcp.md](mcp.md)): `synthesize_ct` (`app`, `checkpoints` or
  `all_checkpoints`, `tta`, `uncertainty`, `mask`) adds the series to the
  dataset and writes it under the output folder; `evaluate_sct` (`sct`,
  `reference`, `reg`, `mask`, `wet`, `hlut`, `structures`) returns the
  numbers and the CSV. Both have `_async` twins. The download gate
  (`allow_model_download`) applies as for every model.
* Workflows ([workflows.md](workflows.md)): the **🔮 Synthetic CT** step (an
  image in, the sCT out as an image the next steps take) and the
  **⚖ Evaluate synthetic CT** step (sCT, reference, an optional registration
  and mask; a report, and the difference as a series). *Export DICOM* writes
  the series a run made with the structures it made.
* The PACS server's task templates include *Synthetic CT from MR, then
  organs* ([pacs-server.md](pacs-server.md)).
* `cargo run --release --example sct_cli -- <dicom_dir> --app sct_mr --cv
  all --tta 2 [--reference <ct_dir>]` writes the sCT and, with a reference
  on the same frame of reference, prints the evaluation; `sct_probe
  <CV_k.pt> [--io out.safetensors]` prints what a checkpoint holds and the
  network's output on a fixed stack, for checking against PyTorch.

## Your own checkpoints

KonfAI's `konfai-apps fine-tune` makes checkpoints of the same network from a
department's own registered MR-CT or CBCT-CT pairs (45 cases were enough for
the Dixon app). *Tools ▸ Models ▸ Your TotalSynth checkpoints ▸ ➕ Add a
checkpoint folder* takes a folder of them (`CV_0.pt`, `CV_1.pt`, ... or any
`.pt`): it is then offered by *Image synthesis*, the MCP tools and the
workflow step beside the published apps, its checkpoints averaged like
theirs, the converted caches under `models/synth/custom/`. The folder is
only read. The recipe is the published apps'; checkpoints trained with
another one will not say so. The registration by structures is how the
pairs are aligned inside RDS before they go to KonfAI.

## Validation

Everything below runs in CI from files KonfAI and PyTorch wrote outside the
repository (`tests/data/synth-*`); the repository has no Python.

* **The network.** A miniature of the shipped `UNetPlusPlus.yml` (the same
  graph, narrow stages, batch norms calibrated on CT stacks, two seeds), saved
  as KonfAI saves a checkpoint, against its PyTorch output: logits within
  1e-4 of their range on the CPU path and through burn (the GPU's code on
  the CPU backend). At full size (26.1 M parameters, random weights, the
  catalog graph unchanged) on one real 512 x 512 stack: 9e-7 of the range
  (an `#[ignore]`d test, `RDS_SYNTH_FULL`). The three namings a checkpoint
  can carry (segmentation_models_pytorch's, KonfAI's declarative and
  parametric) are read alike, the batch norms folded into the convolutions
  at conversion.
* **The recipe.** A small CT block (12 x 50 x 56 voxels of 2.0 x 1.1 x
  1.3 mm, so every resampling step is exercised) through `konfai PREDICTION`
  with the apps' `Prediction.yml`: one checkpoint without mirrors, and two
  checkpoints with both mirrors. The working-grid mask matches to the voxel;
  every member matches but where the network's float rounding, which
  differs from PyTorch's in the last bits, moves a half-precision step of
  the output (1 HU, 2 HU above 1024): 0.05 to 0.3 % of the voxels; the sCT
  on the scan's grid then differs at 0.14 % (one checkpoint) and 0.18 %
  (six members) of the voxels, by 1 or 2 HU.
* **The landing.** The phantom study through the whole module path: the
  series written, read back, merged, its files' identity, derivation and
  source references; the evaluation's metrics against the formulas (an
  exact offset, SSIM against a brute-force reference).
* **Not yet.** The published checkpoints themselves have not been run here:
  Hugging Face refused their download from the development machines. The
  first run on a public SynthRAD case against the model card's numbers is
  the remaining check; `sct_probe` prints what a downloaded checkpoint holds.

## Sources

* TotalSynth: Boussot, Hémon, Barateau, Lafond, Nunes, Dillenseger,
  arXiv:2609.13838 (2026).
* ImpactSynth models and recipe: <https://huggingface.co/VBoussot/ImpactSynth>;
  3D Slicer extension: <https://github.com/vboussot/SlicerImpactSynth>;
  KonfAI: <https://github.com/vboussot/KonfAI>.
* SynthRAD2025 challenge report: arXiv:2605.13555.
