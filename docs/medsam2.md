# Propagating a prompt through a stack

The auto-segmentation engine ([auto-segmentation.md](auto-segmentation.md))
gives 117 fixed anatomical classes with no interaction. The prompt engine
([segvol.md](segvol.md)) segments whatever you point at, but through a
fixed **32 x 256 x 256** window — a very coarse view of a 300-slice CT,
with masks at a quarter of the in-plane resolution.

This engine is the third answer: mark a structure on **one** slice — a box,
a click, or a contour you already drew — and it follows that structure
through the rest of the stack at the slice's own resolution. It is a
pure-Rust re-implementation of
[MedSAM2](https://github.com/bowang-lab/MedSAM2) (Ma et al., 2025), SAM 2.1
fine-tuned on medical images — no Python, no ONNX Runtime, no CUDA.

For CT, whose slices are natively 512 x 512, the network's input resolution
is the slice's own: nothing is resampled in-plane and the mask is as sharp
as the image.

## Using it

**Tools ▶ ⏩ Propagate through dataset A…**, or the **⏩ Propagate…** button
in the sidebar *Segmentations* section, opens the tool window (**⏩ Slice
propagation — dataset A**; the three segmentation engines share one window
layout, see [architecture.md](architecture.md#the-three-engine-windows)).
The workflow is the one the [MedSAM2 extension for 3D
Slicer](https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2)
established — box the structure on one slice, check it, propagate — minus
the round trips: no server to configure, and the network stays loaded
between steps.

1. **Draw the box.** Scroll to a slice where the structure is clear and
   drag a rectangle around it in the image. Drag a **corner** to resize,
   the **middle** to move, anywhere else to start a new one. The box
   belongs to the slice it was drawn on and is shown faintly on the others
   while you scroll. It is drawn in the view whose slices the network
   propagates through — the axial one for an ordinary CT — and the panel
   names it.
2. **Look at that one slice.** *Preview this slice* segments the prompted
   slice alone; with **automatically** ticked (the default) that happens
   whenever the box changes, as soon as you release the mouse. The result
   is an ordinary segmentation, shaded in all three views and in 3D.
3. **Correct it with clicks.** Switch the tool to **➕ Include** or **➖
   Exclude** and click: green points say *this is the structure*, red ones
   *this is not*. Both go to the network with the box, which is how SAM was
   trained to be corrected. The slice is encoded only once, so each click
   costs the prompt path alone — milliseconds on a GPU.
4. **Set the range and propagate.** The range starts as ±32 slices around
   the box and follows it until you set it yourself; *from* / *to* take the
   current slice, and *Whole study* is one click. **▶ Propagate** follows
   the structure through that range.

The crosshair is not involved: while the panel is open the left button in
the drawing view belongs to the box; the other two views navigate as usual.

### Correcting a slice that drifted

Propagation is a chain, and a long one eventually loses its grip — a thin
neck between two lobes, a slice where the structure nearly disappears.
Scroll to where it went wrong, draw a fresh box, and propagate again with
**Add to what is already there** ticked (the default once there is a
result): the new run is unioned into the segmentation instead of replacing
it, fixing the tail without discarding the part that was right.

This is *not* re-conditioning a single pass on two prompted slices, which
the architecture would also allow — it is two independent propagations,
OR-ed: the honest, predictable version, and what the reference pipeline
does too (it never uses more than one conditioning slice per run).

### What the panel holds

| Control | What it does |
|---|---|
| **▭ Box / ➕ Include / ➖ Exclude** | what a left-drag or click in the drawing view does |
| **Preview this slice**, *automatically* | segment the prompted slice, on demand or after every change |
| **from / to**, *this slice*, *Whole study* | the slice range to propagate through |
| **Add to what is already there** | union this run with the current result instead of replacing it |
| **Name** | what the segmentation is called |
| **Options ▸ Window** | the intensity window the model sees — the viewport's own by default, so what you see is what it segments, plus the paper's presets |
| **Options ▸ Model** | which fine-tune to run: general (default), CT lesions, MRI liver lesions, or the 2024-11 base |
| **Options ▸ Both directions** | off tracks only towards higher slice numbers |
| **Options ▸ Largest connected component** | drop everything but the biggest 26-connected blob — usually right for a single lesion |
| **Options ▸ Threshold** | the logit cut, 0 by default (probability 0.5) |
| **Options ▸ Compute** | *Auto* (GPU when available, else CPU), *GPU*, or *CPU* |
| **Options ▸ Model folder** | the root every engine downloads into; this engine's files go to its `medsam2/` sub-folder |

The result is an ordinary segmentation: editable with brush and eraser,
visible in 3D, convertible to RTSTRUCT. The usual loop is *box, preview,
correct, propagate, fix by hand, export*.

The window matters more than it looks — it **is** the model's contrast, and
changing it rebuilds the prepared stack. Everything else (weights, encoded
prompted slice) survives between runs, so only the first run of a session
pays for loading.

## Headless

```
cargo run --release --example medsam2_cli -- <DICOM_DIR> \
    [--models DIR] [--variant latest|ct-lesion|mri-liver|2411] \
    [--device auto|gpu|cpu] [--slice N] [--box r0,c0,r1,c1] [--point r,c] \
    [--window LO,HI] [--preset Abdomen] [--range FIRST,LAST] \
    [--max-slices N] [--all-slices] [--forward-only] [--threshold F] \
    [--no-cleanup] [--out mask.raw]
```

`--models` is the engine's folder, `medsam2/` in the viewer's model folder
by default.

Coordinates are in the *prepared* stack — axial slices in reading order,
the acquisition order for an ordinary head-first-supine CT. `--out` writes
one byte per voxel on the original volume's grid.

`examples/medsam2_probe` fetches a checkpoint, checks it against the layout
the port expects and prints the tensor inventory.

## How it works

MedSAM2 is SAM 2.1 Hiera-Tiny with the input halved to 512; the
architecture is Meta's, unmodified, and the medical part is in the weights.
A volume is handed to it as SAM 2 is handed a video — **slices are frames**
— so the port needs SAM 2's memory bank as well as its image encoder.

| | |
|---|---|
| Image encoder | Hiera-T, four stages (128² → 16²), windowed attention, 27.2 M params |
| Neck | FPN to 256 channels; the 32² map is the image embedding, the 128² and 64² maps feed the decoder's upscaling |
| Prompt encoder | SAM's: random-Fourier point positions, learned box-corner and click embeddings, a convolutional stack for mask prompts |
| Mask decoder | Two-way transformer, depth 2, hypernetwork mask filters, IoU and object-presence heads |
| Memory attention | 4 layers, single-headed at width 256, 2-D axial RoPE |
| Memory encoder | Mask downsampler + two ConvNeXt blocks, 64 channels out |
| Parameters | 38,962,498 across 471 tensors |

Segmenting any slice but the prompted one conditions its image features on
a **memory bank**: every prompted slice, the six nearest slices already
tracked, and up to sixteen *object pointers* — 256-dimensional summaries of
what was segmented on each decided slice. The prompted slice skips all that
and gets one learned "no memory" vector instead.

A study is segmented in two independent passes, exactly as the reference
does: prompt, track to the end, discard the memory, prompt again, track to
the beginning, OR the two results.

Everything runs through `burn`: the whole graph on the GPU with the `gpu`
feature (wgpu — Vulkan, DX12 or Metal, no CUDA toolkit), on a pure-Rust CPU
backend otherwise; the panel reports which one it got. Expect roughly 48 G
multiply-accumulates per slice, about half in the strictly sequential
memory path — which is why the propagation range is bounded by default.

The interactive loop works because **a prompt is cheap and a slice is
not**: encoding a slice is the expensive half, the prompt encoder and mask
decoder that turn a box into a mask a small fraction of it. The engine
keeps the prompted slice's encoder output, so previewing after moving the
box or adding a click re-runs only that fraction — roughly half the cost of
a cold preview on the CPU backend, proportionally far less where the
encoder is fast. Nothing else is cached: propagation is a fresh walk each
time, because its memory bank depends on the prompt.

## Preprocessing, and why it is not the other engines'

```
clip to the window  ->  min-max the clipped volume to [0, 255] and quantize to u8
                    ->  resize each slice to 512 x 512 (PIL's bicubic kernel)
                    ->  /255, then the ImageNet mean and standard deviation
```

There is **no resample to a target spacing and no foreground crop** — the
auto-segmentation engine's nnU-Net-style pipeline and SegVol's
statistics-based one would both quietly change the distribution the weights
were fitted to. The `u8` quantization is no formality either: the network
never saw anything finer.

The resize is PIL's, not PyTorch's — a bicubic kernel with `a = -0.5`, a
support that widens when shrinking, and 8-bit fixed-point arithmetic. It is
reproduced bit for bit, and on 512 x 512 CT it does not run at all.

Slices are taken along the patient's superior axis and oriented as a
radiologist reads them: rows anterior to posterior, columns right to left.

## Divergences from the reference

Three, all deliberate, all visible in the panel:

1. **The propagation range is bounded.** The reference always runs to both
   ends of the volume; a lesion spanning twenty slices does not need three
   hundred sequential steps, and the far end has drifted anyway. The range
   starts at ±32 slices around the box.
2. **The largest-component cleanup is per segmentation.** The reference
   accumulates every lesion of a study into one array and keeps the largest
   connected component of the *union*, silently deleting all but one
   lesion.
3. **The window comes from the viewport** rather than from a per-lesion CSV.

Not a divergence: MedSAM2 enables `fill_hole_area = 8`, but that hole
filling is a CUDA extension which falls back to a no-op on the CPU — the
reference itself only fills holes when it happens to run on a GPU. This
port never does.

## Weights, and their licence

The checkpoint (156 MB) is downloaded from
[huggingface.co/wanglab/MedSAM2](https://huggingface.co/wanglab/MedSAM2) on
first use into `models/medsam2/` under the model folder and converted once
into a `safetensors` cache beside it. The tool window says whether the
chosen variant is cached or how much a run will download.

**The MedSAM2 code is Apache-2.0, but the weights are tagged CC-BY-SA-4.0 and
the model card adds that they "can only be used for research and education
purposes."** Those two statements are in tension; the stricter one governs.

Consequently the weights are only ever fetched to your own machine, at your
request. Nothing is redistributed with this program, and — unlike the
auto-segmentation models — they are **not** offered in the installer's
optional pre-download. The converted cache is a derivative of them and must
not be redistributed either.

Cite Ma et al., *MedSAM2: Segment Anything in 3D Medical Images and Videos*
(arXiv 2504.03600), and SAM 2 (Ravi et al., 2024).

## Accuracy, and what that means here

The paper reports median Dice of 86.7 % on CT lesions (n = 409), 88.8 % on
CT organs, 88.4 % on MRI lesions and 87.2 % on PET lesions, and an 86–87 %
reduction in annotation time in its user study.

This port has been checked against the reference implementation module by
module and end to end — trunk, neck, prompt encoder, mask decoder, memory
pair and a full ten-slice propagation all agree to within about 5e-6
relative, f32 accumulation noise. That is *fidelity to MedSAM2*, not
MedSAM2 being right on your data; the authors' own limitations are worth
knowing.

* Box prompts do not suit thin, branching structures — vessels, airways.
* Nothing models 3-D continuity explicitly; a strongly curved or elongated
  structure can drift.
* The memory bank is eight slices deep and does not adapt to slice
  thickness, so thick slices and abrupt changes between them are where it
  loses track.
* The far end of a long propagation is the least trustworthy part of the
  result — what the range limit is for.

This software is a viewer for research and QA convenience — **not a medical
device, and not for clinical decision-making.**
