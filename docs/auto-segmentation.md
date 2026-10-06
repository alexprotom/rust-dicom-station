# Automatic segmentation

One section of the viewer, one command-line tool and one MCP tool run
**81 automatic segmentation models** from eight published sources, plus any
nnU-Net v2 model of your own, all re-implemented in Rust: no Python, ONNX
Runtime or vendor toolkit at build or run time, the same results on the CPU
or on any GPU through wgpu.

| Source | Models | Network | Licence of the weights |
|---|---|---|---|
| [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) (Wasserthal et al., *Radiology: AI* 2023; *MRI*, *Radiology* 2025) | `total` v2 and v3 (CT, 117 structures), `total_mr` (MR, 50), `body` / `body_mr`, and 26 task models (lung vessels and nodules, liver segments and vessels, head and neck, vertebrae, teeth...) | nnU-Net v2 (`PlainConvUNet`, `ResidualEncoderUNet`) | Apache-2.0 (`brain_aneurysm`: CC BY-NC 4.0) |
| TotalSegmentator's licensed models | 17 tasks: heart chambers at 1.5 mm, coronary arteries, aortic sinuses, annulus and dissection, pulmonary artery landmarks, renal arteries, tissue types, appendicular bones, thigh and shoulder muscles, face, brain structures (CT and MR) | nnU-Net v2 | served by its licence server for your licence number (free for non-commercial use) |
| [nnU-Net v1 pretrained models](https://zenodo.org/records/4003545) (Isensee et al., *Nature Methods* 2021) | the Medical Segmentation Decathlon's CT tasks (liver and liver tumour, lung tumour, pancreas and tumour, hepatic vessels and tumour, spleen, colon cancer), BTCV's 13 abdominal organs, KiTS19 kidney and tumour, SegTHOR's thoracic organs at risk | nnU-Net v1 `Generic_UNet` | CC BY-NC 4.0 |
| [MRSegmentator](https://github.com/hhaentze/MRSegmentator) (Häntze et al., *Radiology* 2025) | 40 structures on MR and CT, five folds | nnU-Net v2 | Apache-2.0 repository, weights in its GitHub release |
| [lungmask](https://github.com/JoHof/lungmask) (Hofmanninger et al., *Eur Radiol Exp* 2020) | R231 (left / right lung), LTRCLobes (five lobes), their fusion `LTRCLobes_R231`, R231CovidWeb | 2-D U-Net, slice by slice | Apache-2.0 |
| [MONAI model zoo](https://github.com/Project-MONAI/model-zoo/tree/dev/models/wholeBody_ct_segmentation) `wholeBody_ct_segmentation` | 104 structures (the TotalSegmentator v1 table), 1.5 and 3 mm | `SegResNet` | Apache-2.0 |
| [CT-FM](https://huggingface.co/project-lighter/whole_body_segmentation) (Pai et al., 2025) | 117 structures (the TotalSegmentator v2 table), a foundation model fine-tuned | `SegResNetDS` | Apache-2.0 |
| [VISTA-3D](https://huggingface.co/nvidia/NV-Segment-CT) (He et al., CVPR 2025), automatic mode | 117 classes (`everything`), or its seven lesion classes | `SegResNetDS2` + class head | NVIDIA Open Model License |
| [NV-Segment-CTMR](https://huggingface.co/nvidia/NV-Segment-CTMR) | VISTA-3D trained on CT and MR: 117 CT classes, 50 MR body classes, a 132-structure brain parcellation (skull-stripped T1) | the same network | NVIDIA non-commercial licence |
| Your own | any nnU-Net v2 model folder added in the model manager | nnU-Net v2 | whatever its authors set |

The same registry also feeds the body contour tool ([body-contour.md](body-contour.md));
VISTA-3D's *point* mode and nnInteractive are the interactive tools of
[interactive-segmentation.md](interactive-segmentation.md).

## The models

`seg_cli --list` prints the table below, with what is still to download
for the model folder given.

| Key | Modality | Classes | Download | What |
|---|---|---|---|---|
| `total_fast` | CT | 117 | 135 MB | TotalSegmentator v2, 3 mm - the default |
| `total` | CT | 117 | 1.17 GB | v2, 1.5 mm: five sub-models (organs, vertebrae, cardiac, muscles, ribs) |
| `total_fastest` | CT | 117 | 135 MB | v2, 6 mm preview |
| `total_v3_fast`, `total_v3`, `total_v3_fastest` | CT | 117 | 242 MB, 1.77 GB, 132 MB | v3 (organs, cardiac and muscles retrained on 1830 subjects) |
| `total_v3_small_fast`, `total_v3_small` | CT | 117 | 242 MB, 1.77 GB | v3's residual-encoder `small` networks |
| `total_mr_fast`, `total_mr`, `total_mr_fastest` | MR | 50 | 126 MB, 464 MB, 45 MB | TotalSegmentator MRI, any sequence |
| `mrsegmentator`, `mrsegmentator_fold0` | CT, MR | 40 | 1.15 GB | five folds averaged, or the first alone |
| `body_fast`, `body`, `body_mr_fast`, `body_mr` | CT / MR | 2 | 43-233 MB | body and trunk; the body contour tool's guides |
| `lung_vessels` | CT | 4 | 365 MB | airways, lung arteries and veins (cropped to the lungs) |
| `lung_nodules` | CT | 2 | 900 MB | lung nodules (residual encoder, cropped to the lungs) |
| `pleural_pericard_effusion` | CT | 3 | 369 MB | pleural and pericardial effusion, five folds |
| `trunk_cavities` | CT | 4 | 233 MB | thoracic, abdominal, pelvic cavities, mediastinum |
| `breasts` | CT | 1 | 234 MB | |
| `liver_vessels`, `liver_segments`, `liver_lesions` | CT | 2, 8, 1 | 365 MB each | cropped to the liver; segments in Couinaud numbering |
| `kidney_cysts` | CT | 2 | 365 MB | |
| `abdominal_muscles` | CT | 22 | 249 MB | |
| `head_glands_cavities`, `headneck_bones_vessels`, `head_muscles`, `headneck_muscles`, `oculomotor_muscles`, `craniofacial_structures` | CT | 7-23 | 365-595 MB | the head and neck set; `headneck_muscles` is two sub-models |
| `teeth` | CT | 77 | 597 MB | every tooth (FDI), jaws, canals, pulp (ToothFairy3, CBCT-trained); cropped to the teeth `craniofacial_structures` finds |
| `cerebral_bleed`, `ventricle_parts` | CT | 1, 12 | 460, 368 MB | |
| `hip_implant` | CT | 1 | 368 MB | |
| `vertebrae_body`, `vertebrae_pp` | CT | 2, 24 | 234 MB | vertebral bodies and discs; `vertebrae_pp` numbers them with the `total` vertebrae |
| `vertebrae_mr`, `liver_segments_mr`, `liver_lesions_mr` | MR | 25, 8, 1 | 231-358 MB | |
| `brain_aneurysm` | MR (TOF) | 1 | 1.17 GB | five folds; **CC BY-NC 4.0, non-commercial** |
| `lungmask_r231`, `lungmask_lobes`, `lungmask_r231covid` | CT | 2, 5, 2 | 124 MB each | robust to dense pathology and low dose |
| `lungmask_lobes_r231` | CT | 5 | 249 MB | upstream's `LTRCLobes_R231`: the lobes inside R231's lungs, lung the lobe model leaves out handed to the lobe it borders most; runs both networks |
| `monai_wholebody`, `monai_wholebody_lowres` | CT | 104 | 75 MB each | 1.5 mm and 3 mm; includes heart chambers, myocardium, pulmonary artery |
| `ctfm_wholebody` | CT | 117 | 349 MB | |
| `vista3d`, `vista3d_lesions` | CT | 132 | 872 MB | VISTA-3D asked for 117 classes, or for its lesion classes; one download for both |
| `nv_ctmr_ct`, `nv_ctmr_mr`, `nv_ctmr_brain` | CT, MR, MR (T1, skull-stripped) | 117, 50, 132 | 872 MB | NV-Segment-CTMR's three label sets; one download for all three; **non-commercial** |
| `msd_liver`, `msd_lung`, `msd_pancreas`, `msd_hepatic_vessel`, `msd_spleen`, `msd_colon` | CT | 1-2 | ≈250 MB each | nnU-Net v1, Medical Segmentation Decathlon: liver + tumour, lung tumour, pancreas + tumour, hepatic vessels + tumour, spleen, colon cancer; **CC BY-NC 4.0** |
| `btcv_abdomen`, `kits19`, `segthor` | CT | 13, 2, 4 | ≈250 MB each | nnU-Net v1: BTCV abdominal organs, KiTS19 kidney + tumour, SegTHOR oesophagus, heart, trachea, aorta; **CC BY-NC 4.0** |
| `heartchambers_highres` | CT | 7 | ≈235 MB | licensed: myocardium, atria, ventricles, aorta, pulmonary artery at 1.5 mm, cropped to the heart and cleared outside heart, aorta and IVC (dilated 10 mm), as upstream |
| `coronary_arteries`, `aortic_sinuses` | CT | 1, 4 | ≈235 MB each | licensed, 0.7 mm, cropped to the heart |
| `aorta_annulus`, `aortic_dissection`, `pulmonary_artery_landmarks` | CT | 2, 2, 7 | ≈1.2 GB each | licensed, five folds |
| `renal_arteries`, `tissue_types`, `tissue_4_types`, `appendicular_bones`, `thigh_shoulder_muscles`, `face` | CT | 1-18 | ≈235 MB each | licensed |
| `brain_structures` | CT | 16 | ≈235 MB | licensed, cropped to the brain |
| `tissue_types_mr`, `appendicular_bones_mr`, `thigh_shoulder_muscles_mr`, `face_mr` | MR | 1-18 | ≈235 MB each | licensed |
| `custom_<name>` | CT or MR | from `dataset.json` | converted, not downloaded | an nnU-Net v2 folder you added |

The **licensed TotalSegmentator models** download from TotalSegmentator's
licence server with your licence number (a licence is free for
non-commercial use, see the TotalSegmentator page). Enter it once in the
model manager (or in the Windows setup); it is kept in your own
`viewer_settings.txt` as `totalsegmentator_licence` and nowhere else, never
logged, never part of an error message. Their sizes are estimates: the
server publishes none. Without a number they are listed but cannot be
downloaded.

The **nnU-Net v1 models** are published one archive per task, about 5 GB
each (every configuration, five folds, optimizer states). The engine takes
one network out of it, `3d_fullres` fold 0 with its plans and
post-processing, by HTTP range requests; a server that does not answer them
gets the whole archive downloaded once and unpacked the same way.

## Using it in the viewer

The **🔬 Auto-segmentation** section of the *Structure auto tools* module
(*Modules ▶ Structure auto tools*, right panel, F10; the workspace is the
module's **Workspace A / B** row; the sections share one layout, see
[architecture.md](architecture.md#the-tool-windows-and-the-modules)):

* **Model** - one list. The first entry, *TotalSegmentator CT, 117
  structures*, opens its own three choices below it: **Weights** v2 or v3,
  the resolution (3 mm, 1.5 mm, 6 mm, each with what it still has to
  download) and, for v3, the **small network**. Every other model is an
  entry of its own under its group (*TotalSegmentator MR*, *Thorax*,
  *Abdomen*, *Head and neck*, *Bones*, *Brain*, *Licensed
  (TotalSegmentator)*, *Tumours and organs (nnU-Net v1)*, *Lungs
  (lungmask)*, *Whole body (SegResNet)*, *VISTA-3D*, *NV-Segment-CTMR
  (VISTA-3D, CT and MR)*, *Your nnU-Net models*), with its modality and
  class count;
  the line under the list says what it does, its family, its licence and
  its download.
* **Sub-models** - for a model made of several networks (the 1.5 mm
  `total` models, `total_mr`, `headneck_muscles`): each can be left out;
  *organs* + *cardiac* alone takes a fifth of the full set's time.
* A **⚠** line when the model was trained on another modality than the
  displayed series (an MR model on a CT, say: it runs, the answer is
  rarely useful), and when the weights are for non-commercial use.
* **Run on** - shown only when the displayed series is a phase of a 4D
  group: *the displayed series*, or *every phase of <group> (n)*; see
  below.
* **Options ▸ Compute** - *Auto* (GPU when available, else CPU), *GPU*, or
  *CPU*. **Options ▸ Model folder** - the root every engine downloads into
  (`%LOCALAPPDATA%\RustDICOMStation\models` on Windows,
  `~/.local/share/RustDICOMStation/models` on Linux, by default; persisted
  as `models_dir` in `viewer_settings.txt`).

The section opens on the model for the displayed series' modality:
TotalSegmentator CT 3 mm for a CT, `total_mr` 3 mm for an MR.

**▶ Segment** runs in the background; the buttons become a progress row
(device, bar, message, **Cancel** - effective during download, conversion
and between inference tiles), mirrored in the sidebar. A **results dialog**
then lists every structure found with its volume, any notes from the run
(a crop model that found nothing, a modality mismatch), and asks what the
checked ones become:

* **Output ▸ segments** - ordinary editable segmentations in the
  segmentation series bound to the displayed image series: brush/erase/grow
  correction, live 3D view, per-structure colours from a curated
  anatomical palette; exports as DICOM SEG.
* **Output ▸ RT structures** - contours in an RT structure set, and **no
  segments**: what a planning system reads. A second row picks the set -
  any set of the workspace, or **➕ new structure set** with a name of its
  own (blank means *Auto-segmentation*). The structures get the RT ROI
  Interpreted Type `ORGAN`; a name already in the set gets a `(2)` suffix.
* **Output ▸ both** - the segments, and contours made from them.
* **Name structures by AAPM TG-263** - each structure that has a TG-263
  name ([below](#tg-263-names)) lands under it (`Lung_L`, `Kidney_R`,
  `VB_T07`); the list shows both names. A structure set built from two
  models then reads as one.

Materialize only what you need: every mask is a full-volume voxel map
(≈ 35 MB at 512 × 512 × 133). If the workspace is switched or modified
during a run, the result is discarded with a message rather than applied
to the wrong volume.

### Running on every phase of a 4D group

With *Run on ▸ every phase of <group>*, the run visits the phases in
temporal order - the displayed one from memory, the others loaded from
their files - and runs the same model on each; the progress row reads
`Phase 30% (4/10): ...` and the bar covers the whole group. One results
dialog then lists **every class any phase found**, with its mean volume
over the phases that have it, and the checked classes are filed **on each
phase under the same names**:

* as segments, in the segmentation series already bound to that phase, or
  in a new one called `<group> <phase>`;
* as contours, in the structure set drawn on that phase, or in a new set
  called `<group> <phase>` when the phase has none - which is the layout a
  planning system expects of a 4DCT, one set per phase.

A class a phase did not find lands nowhere on that phase. The phases stay
linked to their series by UID, so the tree files each result under its
phase, the 4D player steps through them, and the motion pipeline
([motion-4d.md](motion-4d.md)) can take the per-phase structures from
there. Cancelling stops at the current phase and files nothing: a 4D
result with phases missing is not one result.

The same *Run on* row is on **Body contour** and **Prompt segmentation**;
**Slice propagation** and **Interactive segmentation** stay single-series
tools, their prompts being drawn on one series.

### The model manager

*Tools ▶ 📦 Downloaded models* lists every download of every engine under
its engine's folder, grouped as the section groups them, with its modality,
its licence (non-open licences in the warning colour), its size and its
state: *ready*, *partial*, or not there. Rows can be downloaded, removed,
or stripped of the source checkpoint once the converted cache exists.
**⬇ Download open-licence only** fetches the Apache-2.0 and MIT models and
skips the rest; **⬇ Download all missing** fetches everything (the licensed
TotalSegmentator models only once a licence number is set; see
[export-and-tools.md](export-and-tools.md#model-manager)).

Two rows above the list:

* **TotalSegmentator licence number** - a masked field (*Show* reveals it)
  and **Keep**, which stores it in your settings file; the licensed models'
  ⬇ buttons are greyed out until there is one.
* **Your nnU-Net v2 models** - **➕ Add a model folder** takes a trained
  nnU-Net v2 model: a training folder as nnU-Net writes it
  (`<trainer>__<plans>__<configuration>/` with `plans.json`,
  `dataset.json` and `fold_<k>/checkpoint_final.pth`), or the dataset
  folder above it when it holds one training. It joins the model list
  under *Your nnU-Net models* as `custom_<dataset name>`, with the classes
  of its `dataset.json`, every fold it has ensembled, CT or MR by its
  channel name. The network has to be one this engine builds (a 3-D plain
  or residual-encoder U-Net, one input channel, labels rather than
  regions); a folder that is not is listed with *not usable* and the
  reason. The folder is only read: its weights are converted into
  `custom/<key>/` of the model folder on first use. The image is fed in
  the `[S, A, R]` orientation TotalSegmentator's models are trained in.

## Weights and where they come from

Every model is downloaded on first use (or from the model manager) from the
place its authors publish it, over TLS with the **operating-system
certificate store** (a clinical TLS-inspection proxy's CA installed
system-wide is honoured), and converted once:

| Engine folder | Source | What is kept |
|---|---|---|
| `totalsegmentator/<model>/` | TotalSegmentator's GitHub releases (`v2.0.0-weights` ... `v3.0.0-weights`), one zip per nnU-Net dataset | `plans.json`, `fold_k.safetensors` per fold |
| `mrsegmentator/<model>/` | MRSegmentator's GitHub release `v1.2.0` (`weights.zip`, five folds) | the same |
| `lungmask/` | lungmask's GitHub release `v0.0` (`.pth` in PyTorch's pre-1.6 format) | `unet_*.safetensors` |
| `monai_wholebody/` | `MONAI/wholeBody_ct_segmentation` on Hugging Face (`model.pt`, `model_lowres.pt`) | `*.safetensors` |
| `ctfm/` | `project-lighter/whole_body_segmentation` on Hugging Face | its `model.safetensors`, read as it is |
| `vista3d/` | `nvidia/NV-Segment-CT` on Hugging Face | its `model.safetensors`, read as it is |
| `vista3d/ctmr/` | `nvidia/NV-Segment-CTMR` on Hugging Face | the same |
| `totalsegmentator/<model>/` (licensed) | TotalSegmentator's licence server, a zip per model for your licence number | `plans.json`, `fold_k.safetensors` |
| `nnunet_v1/<model>/` | Zenodo record 4003545, one network taken out of each task's archive | `plans.json` (rewritten from `plans.pkl`), `postprocessing.json`, `model.safetensors` |
| `custom/<key>/` | a model folder of yours | `plans.json`, `fold_k.safetensors` |

The conversion is native: `nn/pickle.rs` reads PyTorch's zip checkpoints and
its older pickle-stream format (a minimal pickle machine: persistent storage
ids, `_rebuild_tensor_v2`, `_rebuild_from_type_v2`, `_rebuild_parameter`,
shapes, strides, offsets), and the state dict is written as `safetensors`.
`nn/pyobj.rs` reads plain pickles with numpy values (nnU-Net v1's
`plans.pkl`) as JSON, and `nn/remote_zip.rs` reads single members of a
remote zip through range requests. A TotalSegmentator v3 zip holds two trainings side
by side; the folder the model names is taken. Later runs read only the
cache. For **air-gapped machines**, run (or download in the model manager)
on a connected machine and copy the engine folder.

## How each family runs

### nnU-Net (TotalSegmentator, MRSegmentator)

A model is data: a task (`autoseg/total.rs`, and `autoseg/tasks.rs`
generated from TotalSegmentator's `map_tasks_config.py` and
`map_to_binary.py` at the upstream commit named in its header) lists its
sub-models with the label table each maps into, its folds, its tile step,
its crop rule and its post-processing. The network itself is rebuilt from
each model's `plans.json` - the nnU-Net 2.0/2.1 format and the
`architecture` format of 2.2 and later, any configuration, `inherits_from`
resolved - as a `PlainConvUNet` or a `ResidualEncoderUNet`.

1. **Crop cascade.** A task with crop classes first runs a coarse model on
   the whole scan - the 6 mm `total` model, the 3 mm one where upstream asks
   for a robust crop, the 3 mm `total_mr` model for MR tasks, the 6 mm body
   model for `body_trunc`, `craniofacial_structures` for `teeth` - and
   keeps the bounding box of the named classes, widened by 20 mm (in whole
   voxels, clamped to the scan). The task then runs on that box only and
   its labels are pasted back. A crop model that finds none of its classes
   ends the run with an empty result and a note, as upstream does.
2. **Canonical orientation.** Axes permuted and flipped to the order the
   model was trained in: `[S, A, R]` for TotalSegmentator, `[S, P, L]` for
   MRSegmentator.
3. **Resampling** to the model's spacing, trilinear, with
   `scipy.ndimage.zoom`'s endpoint-aligned coordinates (TotalSegmentator's
   resampler), including its int32 truncation. Where upstream resamples
   twice (a task with its own `resample` value, `lung_nodules`), this
   engine goes to the plans' spacing in one step.
4. **Normalization** per model: CT models clip to the training set's
   [0.5, 99.5] HU percentiles and z-score with its mean and standard
   deviation; MR models z-score against the image itself.
5. **Sliding windows** with nnU-Net's tiling - step 0.8 of the patch for
   `total` and `total_mr`, 0.5 for every other task, as upstream - and
   Gaussian importance weighting (σ = patch/8). Several folds are an
   ensemble: their logits are summed in the accumulator. The accumulator is
   a ring buffer along the leading axis, and for models with many classes
   at fine resolution also split into strips across the second axis, so its
   memory stays under 3 GB whatever the scan.
6. **Label merging** of sub-models into the task's table, later sub-models
   winning at overlaps (TotalSegmentator's order).
7. **Post-processing** where the task has it: `body` keeps the largest trunk
   piece and drops small extremity islands; `vertebrae_pp` renumbers the
   vertebral bodies from the `total` vertebrae they touch.
8. **Back-mapping** to the scan's grid by nearest neighbour.
9. **On the scan's grid**, for the tasks that have it: upstream's
   `remove_outside` (`heartchambers_highres`: labels outside heart, aorta
   and inferior vena cava of the crop model, dilated by 10 mm in whole
   voxels of the mean spacing with the 6-neighbourhood, are cleared), and
   an nnU-Net v1 model's own `postprocessing.json` (below).

**nnU-Net v1 models** (`autoseg/v1.rs`) run through the same steps. v1's
`Generic_UNet` is v2's `PlainConvUNet` under other names, with one
difference: each decoder stage convolves with the kernel of the encoder
stage one level deeper than v2 takes (they differ when the first stage's
kernel is anisotropic, `1 × 3 × 3`), which the rewritten plans carry. The
checkpoint's tensors are renamed, and the bias-free transposed
convolutions and heads get zero biases. Its `plans.pkl` (a Python pickle with numpy values) is
rewritten as a v2 `plans.json` with an explicit architecture: the last
stage of `plans_per_stage` (what v1's `3d_fullres` trains on), features
doubling from `base_num_features` up to 320, `conv_per_stage` convolutions
per stage, the CT normalization constants. The 0.5 window step and the
training's `postprocessing.json` are v1's (the largest 6-connected piece of
the listed classes kept, or only pieces under the minimum volume dropped,
on the scan's grid as v1 does it). Three things are this engine's rather
than v1's, and documented as such: the scan is reoriented to `[S, A, R]`
(v1 does not reorient; its mirroring augmentation makes the networks
indifferent to flips), resampling is trilinear (v1: third-order in-plane,
nearest across thick slices), and the patches are not mirrored (v1
averages eight mirrored predictions by default). One fold runs where v1
ensembles five.

### lungmask

Every axial slice (the scan in `[S, P, L]` axes, as SimpleITK hands it to
lungmask after `DICOMOrient("LPS")`) is clipped to [-1024, 600] HU, cropped
to the body (a threshold, binary morphology and the largest components, all
with scipy's and skimage's conventions), resized to 256 × 256 and
segmented. The stack is then cleaned in 3-D - stray pieces handed to the
neighbour they touch most, each label kept as its largest 26-connected
piece with its voids filled - and every slice's answer is resized back into
its body box.

`lungmask_lobes_r231` is upstream's `LTRCLobes_R231`: LTRCLobes and R231
both run; lung R231 finds and the lobe model leaves unlabelled becomes a
spare label, everything outside R231's lungs is cleared, and the same
clean-up runs once more on the scan's own grid with every spare piece
handed to the lobe it borders most. Upstream skips a neighbour whose
*piece number* (not its label) equals the spare label; the port keeps that
quirk, since the point is the reference's answer.

### MONAI whole body and CT-FM

`segresnet/`: the bundle's or the model card's pipeline. MONAI whole body:
RAS orientation, 1.5 or 3 mm spacing (MONAI's `Spacing`, first-voxel
aligned), z-score over nonzero voxels then [-1, 1], 96-cubed windows with
overlap 0.25 and edge-replicated padding, Gaussian weights centred as MONAI
centres them, nearest back. CT-FM: `[S, P, L]` axes, HU -1024..2048 to
[0, 1], cropped to the foreground, 96 × 160 × 160 windows with overlap
0.625, the largest component of each label kept.

### VISTA-3D automatic

`vista3d.rs`: resampled to 1.5 mm in the scan's own axis order, cropped to
the voxels above 0 HU with a 10-voxel margin, HU -963.8..1053.7 to [0, 1],
reoriented to RAS; 128-cubed windows with uniform weights, overlap 0.25,
edge-replicated padding. For every class asked for, its learned embedding
(through the class head's MLP) is dotted with the features of every voxel;
a voxel takes the class with the largest positive logit.

NV-Segment-CTMR runs the same network with its own weights and the three
differences of its pipeline (`vista3d_pipeline.py` of the repository):
intensities scaled between the cropped image's 1st and 99th percentiles
(numpy's linear interpolation) instead of the HU window, 192 × 192 × 128
windows, and the label set by modality - `CT_BODY` (the 117 classes),
`MRI_BODY` (50) or `MRI_BRAIN` (classes 214-345, for a skull-stripped T1).
Its class ids reach 345, so its output labels are numbered in the order the
classes are asked for, and the class table says which is which.

## Compute engines

**CPU.** The nnU-Net family runs on hand-written kernels
(`autoseg/cpu.rs`): convolution as per-output-slice im2col + pure-Rust SIMD
GEMM (the `gemm` crate), parallel over slices with rayon, tiled so the
im2col block stays in cache; transposed convolutions, instance norm and
LeakyReLU fused or hand-rolled. The other families are written against
`burn` and run on its pure-Rust `ndarray` backend, but their convolutions
go through the same GEMM kernels (`nn/fastconv.rs`: any kernel, stride and
padding, and SegResNetDS's 3 × 3 × 3 stride-2 transposed convolution as
eight sub-pixel convolutions), and lungmask's whole network has a CPU path
of its own (`unet2d/cpu.rs`) - burn's backend spent seconds per batch in
its bilinear upsampling alone. lungmask segments the bundled 133-slice
CT in 7.4 minutes on two throttled cores (LTRCLobes; PyTorch: 4.2 minutes on
one core with oneDNN).

**GPU.** The same networks through
[burn](https://github.com/tracel-ai/burn)'s **wgpu** backend: Vulkan / DX12 /
Metal, so NVIDIA, AMD, Intel and Apple GPUs, with **no CUDA toolkit or
vendor SDK**. *Auto* probes for a usable adapter with a self-test and falls
back to the CPU. The GPU path is optional at build time (`gpu` cargo
feature, on by default).

## Validation

Every port is checked against its reference implementation:

* **TotalSegmentator `total`, v2** - on the bundled phase-0 CT the full
  pipeline agrees with the official Python TotalSegmentator at **mean Dice
  0.9995 across 90 structures** (worst 0.992); a single preprocessed patch
  reproduces the 3 mm checkpoint's logits to 1 × 10⁻⁴ with 100 % argmax
  agreement; CPU and GPU engines gave bit-identical labels over a full run.
* **v3** - the plain and the residual-encoder 3 mm checkpoints reproduce
  PyTorch's logits to 1.4 × 10⁻⁴ and 1.3 × 10⁻³ (5 × 10⁻⁶ of the logit
  range), argmax identical. v2 and v3 agree at mean Dice 0.95 over the 90
  structures both find, v3 big and small at 0.92: two trainings, not a port.
* **Task models and the crop cascade** - against TotalSegmentator 2.18 on
  the same CT (Python run with the same weights): see the table below.
* **lungmask** - R231, LTRCLobes and their `LTRCLobes_R231` fusion on the
  same CT, against lungmask 0.2.21 in Python: **every voxel identical**
  (34.9 million, all three).
* **MONAI whole body, CT-FM, VISTA-3D** - with their published weights,
  against the same pipelines written with MONAI 1.6's own transforms and
  sliding-window inferer and PyTorch, on a 288 × 288 × 70 block of the
  bundled CT (200 × 200 × 70 where the Python run of the whole block did
  not fit 8 GB): see the table. Networks of the exact architectures with
  random weights reproduce MONAI to 1 × 10⁻⁴ as well
  (`tests/data/zoo-nets.safetensors`), and MONAI's `Spacing`,
  `ScaleIntensityRange`, `CropForeground`, `KeepLargestConnectedComponent`
  and sliding-window placement are unit tested against values MONAI
  produced.
* **MONAI whole body, the bundle's order.** The bundle inverts the softmax
  probabilities to the scan's grid (bilinear) and takes the argmax there;
  the port takes the argmax on the 1.5 mm grid and maps labels back by
  nearest neighbour, because 105 channels of probabilities on a scan's grid
  would need more than 10 GB. Against the bundle's own order the 46
  structures over 1000 voxels agree at mean Dice 0.971 (worst 0.942, a rib):
  the difference is at boundaries, between 3 mm slices.

| Model | Classes found | Mean Dice | Worst | Notes |
|---|---|---|---|---|
| `liver_segments` | 8 | 0.9999 | 0.9998 | cropped to the liver the 6 mm `total` finds, the same crop as upstream's (264 × 233 × 48) |
| `vertebrae_pp` | 18 | 1.0000 | 0.9997 | with the upstream relabelling of touching vertebrae; on a 256 × 256 in-plane crop of the CT around the spine (the Python run of the whole image did not fit the 8 GB machine) |
| `trunk_cavities` | 4 | 0.9975 | 0.9943 | new-format plans; every differing voxel lies within two voxels of a cavity's border, where the network is least certain |
| `total_mr_fast` | 32 | 0.9995 | 0.9986 | the MR model run on the CT, to compare the ports, not the anatomy |
| `monai_wholebody` | 52 | 1.0000 | 0.9993 | published `model.pt`; MONAI's transforms and inferer, argmax then nearest back as the port does; 99.997 % of the labelled voxels identical |
| `vista3d` | 63 | 1.0000 | 1.0000 | published NV-Segment-CT weights; the bundle's transforms and `SlidingWindowInfererAdapt` settings; every voxel identical |
| `ctfm_wholebody` | 53 | 0.9969 | 0.9091 | published weights; the 34 structures over 1000 voxels at mean 0.9994, worst 0.9963 (the worst row is a rib of 6 voxels); the Python run kept its logits in float16 to fit 8 GB |
| VISTA-3D point mode | 1 | 1.0000 | 1.0000 | one click in the pulmonary artery, `point_based_window_inferer` and `keep_components_with_positive_points`: the same 122 063 voxels |

**nnInteractive** with its published weights: the network on a 96-cubed
patch of the same CT with a point prompt, against PyTorch through
nninteractive 2.6.0's own loader, agrees to 0.011 on logits of range 640
(1.7 × 10⁻⁵), the argmax differing at one voxel of 884 736 (the `#[ignore]`d
test `nninteractive::tests::published_weights_match_pytorch`, with
`RDS_NNINTERACTIVE_ROOT` and `RDS_NNINTERACTIVE_PROBE`). The session around
it is the one checked voxel for voxel against nninteractive's own; a full
192-cubed session on the CPU needs more memory than the 8 GB validation
machine, in Python as in Rust.

**nnU-Net v1**: a `Generic_UNet` built by nnunet 1.7.1 the way its
`nnUNetTrainerV2` builds one (anisotropic first stage), saved as a `.model`
with a protocol-4 `plans.pkl`, runs through the conversion and the plans
rewrite to the same logits (`autoseg::v1::tests`). That test found the one
architectural difference between v1 and v2 (the decoder's kernels, above).
No published v1 archive was reachable from the validation machine.

`lung_nodules` (the residual-encoder L network at a 192-cubed patch) has no
row: on the same CT the Rust run takes upstream's lung crop and the same 36
windows and stays within 2.5 GB, but PyTorch's CPU run needed more memory
than the 8 GB validation machine had, so there is no reference to compare
with. Its network is the one validated on v3's residual-encoder
checkpoints. The remaining differences in the table are voxels on
structure boundaries, where last-bit differences in the arithmetic decide
between two nearly equal logits.

Unit tests pin the window positions (nnU-Net's and MONAI's), the
resampling conventions, the crop box and the post-processing to reference
values; `tests/autoseg.rs` assembles miniature plain and residual-encoder
networks from synthetic tensors with the checkpoint key naming; an
`#[ignore]`d end-to-end test runs the real 3 mm model against the bundled
example data:

```
RDS_AUTOSEG_MODELS=path/to/models/totalsegmentator \
  cargo test --release --test autoseg -- --ignored
```

## Command-line tools

```
# list the models (and what each still has to download)
cargo run --release --example seg_cli -- --list [--models ROOT]

# segment: labels .bin + class table and organ list .json
cargo run --release --example seg_cli -- <dicom_dir> <out_prefix> \
    [--model KEY] [--models ROOT] [--device auto|gpu|cpu] [--parts organs,cardiac,...]

# dump one preprocessed patch + its logits (for numerical comparison)
cargo run --release --example autoseg_probe -- <dicom_dir> <models_dir> \
    total_3mm <out_prefix>
```

`autoseg_cli` is the older name of `seg_cli` and takes the same arguments
(and the older `--variant fast3|highres|...` names). The MCP server's
`segment_organs` takes `model` (any key above, default `total_fast`) and
`list_models` returns the table ([mcp.md](mcp.md)); the workflow node *Auto
segmentation* has a `model` parameter ([workflows.md](workflows.md)).

## TG-263 names

`zoo/tg263.rs` maps every model's class names to the AAPM TG-263 standard
nomenclature where one exists: organs (`Liver`, `Kidney_L`, `Lung_LUL`,
`Bowel_Small`), vessels (`A_Aorta`, `V_Venacava_I`), bones and muscles,
vertebrae (`VB_C1` ... `VB_L5`, `VB_S1`) and ribs (`Rib01_L` ... `Rib12_R`). The
different spellings of the models meet in one name: TotalSegmentator's
`lung_upper_lobe_left`, MRSegmentator's `left_lung` (a whole lung) and
VISTA-3D's `left lung upper lobe` land as `Lung_LUL`, `Lung_L` and
`Lung_LUL`. Classes with no TG-263 equivalent (`costal_cartilages`,
`autochthon_left`) keep their model name.

## The 117 classes of `total`

Global label ids follow TotalSegmentator v2's `class_map["total"]`:

| Ids | Group (1.5 mm sub-model) | Structures |
|---|---|---|
| 1-24 | organs | spleen, kidney R/L, gallbladder, liver, stomach, pancreas, adrenal gland R/L, lung upper/lower lobe L, lung upper/middle/lower lobe R, esophagus, trachea, thyroid, small bowel, duodenum, colon, urinary bladder, prostate, kidney cyst L/R |
| 25-50 | vertebrae | sacrum, S1 (v3: L6), L5-L1, T12-T1, C7-C1 |
| 51-68 | cardiac | heart, aorta, pulmonary vein, brachiocephalic trunk, subclavian artery R/L, common carotid artery R/L, brachiocephalic vein L/R, left atrial appendage, superior/inferior vena cava, portal + splenic vein, iliac artery L/R, iliac vena L/R |
| 69-91 | muscles | humerus L/R, scapula L/R, clavicula L/R, femur L/R, hip L/R, spinal cord, gluteus maximus/medius/minimus L+R, autochthon L/R, iliopsoas L/R, brain, skull |
| 92-117 | ribs | ribs left 1-12, ribs right 1-12, sternum, costal cartilages |

The other models' tables are in the engine sources (`autoseg/tasks.rs`,
`autoseg/total.rs`, `unet2d/mod.rs`, `segresnet/mod.rs`, `vista3d.rs`), each
with its upstream reference.

## Licensing and citation

The weights keep their publishers' licences; the viewer downloads them at
the user's request and never redistributes them. TotalSegmentator's open
tasks, MRSegmentator, lungmask, MONAI and CT-FM are Apache-2.0
(`brain_aneurysm` is CC BY-NC 4.0, non-commercial); VISTA-3D is under the
NVIDIA Open Model License (commercial use allowed, with attribution and its
conditions). For non-commercial use only: the nnU-Net v1 models (CC BY-NC
4.0, and the licences of the challenge data they were trained on),
NV-Segment-CTMR (NVIDIA's non-commercial licence), and TotalSegmentator's
licensed models without a commercial licence. The section and the model
manager say which applies, in the warning colour when it is not an open
licence.

Cite the model you use: Wasserthal et al., *TotalSegmentator: Robust
Segmentation of 104 Anatomic Structures in CT Images*, Radiology: AI 2023
(<https://doi.org/10.1148/ryai.230024>); D'Antonoli et al.,
*TotalSegmentator MRI*, Radiology 2025; Häntze et al., *MRSegmentator*,
Radiology 2025; Hofmanninger et al., Eur Radiol Exp 4, 50 (2020); Pai et al.,
*CT-FM*, 2025; He et al., *VISTA3D*, CVPR 2025; and Isensee et al., *nnU-Net*,
Nature Methods 2021 (<https://doi.org/10.1038/s41592-020-01323-z>).

## Troubleshooting

* **"no usable wgpu adapter found"** - no Vulkan/DX12/Metal device
  (headless machine, missing driver). *Auto* silently uses the CPU;
  forcing *GPU* reports the error. If the whole program will not start,
  that is the related and more common failure - a Windows machine
  advertising a Vulkan driver that cannot create a device; see
  [viewer.md](viewer.md#graphics-backend). The backend chosen there governs
  inference too.
* **Download fails behind a proxy** - the downloader uses the OS trust
  store, so an inspection proxy's CA installed system-wide is honoured;
  Hugging Face also rate-limits anonymous downloads now and then: retry, or
  fill the engine folder by hand.
* **A task found nothing** - its crop model did not find the organ it
  crops to (no liver in a head CT); the results dialog says so.
* **Memory** - the 3 mm `total` model peaks around 2-3 GB for a thorax CT;
  the 1.5 mm models, VISTA-3D and the 1.5 mm MONAI model need several GB
  more, and each materialized mask ≈ volume-size bytes.
* As with everything in this viewer: research and QA use - not a medical
  device, not for clinical decision-making.
