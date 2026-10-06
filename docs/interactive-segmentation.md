# Interactive segmentation

Click on a structure, look at the 3-D answer, click again where it is
wrong. Two engines answer such prompts, both re-implemented in Rust (no
Python, no ONNX Runtime, no CUDA toolkit):

* **[nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)** (Isensee,
  Rokuss, Krämer et al., *nnInteractive: Redefining 3D Promptable
  Segmentation*, arXiv:2503.08373, 2025): points, boxes, scribbles and
  lassos, positive and negative, on CT, MR or PET; it zooms out by itself
  when the structure is larger than its window.
* **VISTA-3D's point mode** (He et al., *VISTA3D*, CVPR 2025; NVIDIA
  NV-Segment-CT): clicks only, on CT; the same weights as its automatic
  mode ([auto-segmentation.md](auto-segmentation.md)).

They complement the other prompt tools: [segvol.md](segvol.md) answers one
prompt (a box, a point, a structure name) in one pass, and
[medsam2.md](medsam2.md) follows a structure boxed on one slice through the
stack. Here every prompt refines the same object, with all the earlier
prompts and the current answer taken into account.

## Using it

The **🎯 Interactive segmentation** section of the *Structure auto tools*
module (*Modules ▶ Structure auto tools*, right panel, F10; the workspace is
the module's **Workspace A / B** row). While the section is unfolded, the
left button in the workspace's views belongs to it - in all three views.

1. **Pick the engine** (*nnInteractive* or *VISTA-3D points*) and the
   **prompt**: *● Point*, *⬚ Box* (drag a rectangle on one slice),
   *✏ Scribble* (draw a stroke) or *◌ Lasso* (draw a closed outline);
   VISTA-3D takes points only. *⌖ Navigate* gives the left button back to
   the crosshair without folding the section.
2. **➕ Include** or **➖ Exclude**: whether the prompt marks the structure
   or what must stay out. Green prompts include, red ones exclude; the ones
   on the current slice are drawn bright, those on other slices of the same
   view faintly.
3. **Use the prompt in a view.** Each prompt goes to the network as soon as
   it is released. The answer lands in one segmentation of the workspace
   (named by *Name*), replaced after every prompt - an ordinary
   segmentation, editable with the brush, shown in 3D, exported as DICOM
   SEG or turned into contours. Prompts given while the network is busy
   wait their turn.
4. **Correct it.** Click where it is wrong: an include prompt where it is
   missing, an exclude prompt where it overreaches. **↺ Undo prompt**
   takes the last prompt and its answer back. **➕ New object** keeps the
   segmentation as it is and starts the next one.

**Refine "<name>"** (nnInteractive) makes the workspace's active
segmentation the starting point - a TotalSegmentator organ, a contour drawn
by hand and converted - so the prompts correct it instead of starting from
nothing. This is nnInteractive's *initial segmentation*.

*Options*: *Zoom out when the answer reaches the border* (nnInteractive's
AutoZoom, on by default), the compute device, the model folder. Both
networks are large: on a GPU a prompt takes seconds; on the CPU, a minute
or more (nnInteractive's 192-cubed residual U-Net runs once per prompt, more
often when it zooms out; VISTA-3D runs a 128-cubed window per click), which
the section says.

## Headless

```
cargo run --release --example interactive_cli -- <dicom_dir> <out_prefix> \
    [--engine nninteractive|vista3d] [--models ROOT] [--device auto|gpu|cpu] \
    [--no-autozoom] --point X,Y,Z[,+|-] ... [--box X0,Y0,Z0,X1,Y1,Z1[,+|-]] ...
```

Positions are voxel indices of the loaded volume (`x` the column, `y` the
row, `z` the slice); the prompts are given in order, each followed by a
prediction, as clicks in the viewer are. Writes the mask (`.bin`,
`Volume::data` order) and a `.json` summary.

## How nnInteractive works

The network is an nnU-Net `ResidualEncoderUNet` (the L preset: six stages,
32 to 320 features, 192-cubed patch) whose input is the image and seven
prompt channels: the current segmentation, and a positive and a negative
channel each for boxes and lassos, points, and scribbles. It runs on the
nnU-Net engine of [auto-segmentation.md](auto-segmentation.md), whose
residual encoder reads any number of input channels. The rest is the
inference session (`nninteractive/session.rs`), a port of
`nnInteractiveInferenceSession` from nninteractive 2.6.0:

* **The image** is used in its own voxels - no resampling - reordered to
  the `[S, A, R]` axes nnU-Net's `NibabelIOWithReorient` gave the training
  data, and z-scored with the mean and (unbiased) standard deviation of its
  nonzero bounding box.
* **Prompts** are drawn into their channel: a point as a radius-4 ball
  whose values fall off with the distance transform, a box or a lasso as
  its filled area on one slice, a scribble as its stroke. Each prompt is
  stored at an intensity that grows by 1/0.98 per prompt; before the network
  sees them, the channels are divided by the current intensity, so the
  newest prompt counts most. The values are the float16 values upstream
  stores.
* **AutoZoom.** The first pass is the 192-cubed window around the prompt
  (around a box, one large enough to show it with a third of a patch of
  context). If the answer changes along the window's border, the window
  grows 1.5 times - up to 4 times the patch - and is resampled to the patch
  (trilinear for the image, area averages for the prompts, the point and
  scribble channels dilated first so they survive the averaging) and run
  again.
* **Refinement.** After a zoomed pass, the region where the coarse answer
  differs from the current segmentation (opened, so specks do not each
  earn a pass) is covered with patch-sized boxes, greedily, and each box
  is run again at full resolution with the coarse answer as its previous
  segmentation. Only those boxes are written back.

The weights (`MIC-DKFZ/nnInteractive`, folder `nnInteractive_v1.0`: its
`plans.json`, the session metadata and the 411 MB fold-0 checkpoint) are
downloaded on first use into `models/nninteractive/` and converted once.

## How VISTA-3D's point mode works

The volume is prepared exactly as for VISTA-3D's automatic mode (1.5 mm,
cropped to the voxels above 0 HU, scaled, RAS), and each click is mapped
into that grid. The pipeline is the bundle's with `use_point_window`
(`monai.apps.vista3d.inferer.point_based_window_inferer`):

* every click opens a 128-cubed window centred on it (shifted inside the
  volume); every click inside a window prompts it;
* the network is the encoder's *point* decoder and MONAI's
  `PointMappingSAM` head - a SAM-style mask decoder in 3-D: the features at
  half resolution, the clicks as tokens (a random-Fourier position code
  plus a learned embedding for positive or negative), a two-way transformer,
  the image side brought back to full resolution and dotted with the mask
  token through a small MLP;
* overlapping windows' logits are summed (the inferer marks a voxel as
  covered, it does not count how often); voxels no window reached are
  undecided and come out empty;
* `VistaPostTransformd` keeps the 26-connected pieces of the positive
  region that hold a positive click.

A window's answer depends only on the image under it and the clicks inside
it, so the session keeps answered windows and a new click re-runs only the
windows it falls into; the result is the one a run over all the clicks at
once gives.

## Validation

Neither set of weights could be downloaded on the development machines
(Hugging Face is not reachable from them), so both ports were validated
against the reference code itself:

* **nnInteractive session** - the reference session (nninteractive 2.6.0)
  and this one were run side by side on a synthetic 30 × 40 × 26 image with
  a stand-in network (one fixed 3 × 3 × 3 convolution, the same in both)
  and a small patch (12 × 16 × 10), through nine steps: positive and
  negative clicks, a 2-D box, a scribble, a lasso, an initial segmentation
  kept as it is and one refined whole. AutoZoom, refinement with the greedy
  cover and its random fallback (made deterministic in both) are all
  exercised. Every step's segmentation is **identical voxel for voxel**,
  the number of network passes is the same, and every pass's input agrees
  per channel to 1 × 10⁻⁴ (`tests/data/nninteractive-session.safetensors`).
  Matching it took PyTorch's exact arithmetic: float16 accumulation in the
  area pooling, fused multiply-adds in the trilinear resampling.
* **nnInteractive network** - the residual encoder U-Net is the one
  validated on TotalSegmentator v3's real `small` checkpoints; the first
  convolution takes the eight channels.
* **VISTA-3D points** - a network of VISTA-3D's architecture (eight
  features, three encoder levels), initialised at random with weights
  rounded to float16, run through MONAI 1.6: the point decoder's features
  and the point head's logits agree to 1 × 10⁻³ relative, and the bundle's
  whole point pipeline (three windows on a 40 × 28 × 44 image, padding,
  stitching, the positive-component filter) gives the same mask
  (`tests/data/vista-points.safetensors`).

A run with the published weights on a real case is the remaining check.

## Licences

nnInteractive's code is Apache-2.0; its **weights are CC BY-NC-SA 4.0 -
non-commercial use only**. VISTA-3D's weights are under the **NVIDIA Open
Model License** (commercial use allowed, with attribution and its
conditions). Both are downloaded at the user's request and never
redistributed; the section's licence line says which applies. Research and
QA use - not a medical device.
