# Prompt-driven segmentation

The auto-segmentation engine ([auto-segmentation.md](auto-segmentation.md))
gives 117 anatomical classes with no interaction. Every one of them is normal
anatomy, which for a radiotherapy station means organs at risk and nothing
else — no GTV, no nodal disease, no post-surgical cavity, no recurrence.
Those are exactly the structures that take longest to draw and that no
fixed-class model will ever cover, because they are patient-specific by
definition.

This engine is the complement. You point at something and it segments it:
a box, a click, or a structure name in plain text. It is a pure-Rust
re-implementation of [SegVol](https://github.com/BAAI-DCAI/SegVol)
(Du et al., NeurIPS 2024) — no Python, no ONNX Runtime, no CUDA.

## Using it

**Tools ▶ 🧠 Prompt-segment dataset A…**

Move the crosshair onto the structure first; the prompt is anchored to it.

| Prompt | What it does | Best for |
|---|---|---|
| **Box** | A box centred on the crosshair, with a half-extent in millimetres | Lesions, targets — the most reliable prompt |
| **Point** | A single foreground click at the crosshair | Compact, well-separated structures |
| **Text** | A structure name through the model's trained template | Anatomy the 117-class model does not cover |

The result arrives as an ordinary segmentation: editable with the brush and
eraser, visible in the 3D window, and convertible to RTSTRUCT — so the usual
loop is *prompt, fix by hand, export*.

### Options

* **Refinement pass** — the second, sliding-window pass. Without it you get a
  single coarse pass: much faster, much blockier.
* **Skip the search pass (box only)** — the first pass exists only to *locate*
  the structure. With a box drawn by hand it is redundant, so skipping it
  roughly halves the work and avoids losing small lesions to the downsample.
  **This departs from the reference implementation**, which always runs both;
  it is off by default.
* **Threshold** — probability cut applied to the network's output, 0.5 by
  default.

## Headless

```
cargo run --release --example segvol_cli -- <DICOM_DIR> \
    [--box z0,y0,x0,z1,y1,x1] [--point z,y,x] [--text liver] \
    [--no-zoom-in] [--fast-box] [--threshold F] [--out mask.raw]
```

Coordinates are in the *prepared* grid — canonically oriented `[S, A, R]` and
cropped to the foreground — which is what the network sees. `--out` writes one
byte per voxel on the original volume's grid.

`examples/segvol_probe` fetches the checkpoint and checks it against the
layout the port expects, printing the tensor inventory.

## How it works

The network only ever accepts a **32 × 256 × 256** volume. That is not a
configuration choice: the image encoder's position embedding is a learned
2048-token parameter with no interpolation logic, and the mask decoder
contains a `LayerNorm` whose shape is the literal `(192, 16, 32, 32)`
activation. So a study is segmented by running the same fixed graph twice —
once over the whole volume squashed into that shape to find the structure,
then again as a sliding window over a crop around what the first pass found.

Preprocessing is unlike the nnU-Net engine's: **no HU window and no resample
to a target spacing**. Intensities are normalized from the volume's own
statistics — threshold at the mean, take the 0.05/99.95 percentiles and the
mean and standard deviation of the voxels above it, clip and z-score — then
min-max to [0,1] and crop the resulting zero rim.

| | |
|---|---|
| Image encoder | MONAI 3-D ViT, 12 blocks, width 768, 2048 tokens, global attention |
| Prompt encoder | SAM's, 3-D: random-Fourier positions, learned point/box embeddings |
| Mask decoder | Two-way transformer, depth 2, hypernetwork mask filters |
| Text tower | CLIP ViT-B/32, frozen, shipped inside the checkpoint |
| Parameters | 180,891,293 across 475 tensors |

Roughly 6 s per window on a desktop CPU; the image encoder is ~97% of that
and runs on any GPU `wgpu` can drive when the `gpu` feature is on (default).

## Weights, and their licence

The checkpoint (~724 MB) is downloaded from
[huggingface.co/BAAI/SegVol](https://huggingface.co/BAAI/SegVol) on first use,
along with the CLIP tokenizer's two small data files.

**The SegVol code is MIT, but the weights carry no licence declaration at
all** — no licence tag, no LICENSE file — and the training corpus aggregates
25 datasets whose terms differ, several of them non-commercial. This is
unlike TotalSegmentator's Apache-2.0 weights.

Consequently the weights are only ever fetched to your own machine, at your
request. Nothing is redistributed with this program, and unlike the
auto-segmentation models they are **not** offered in the installer's optional
pre-download.

## Accuracy, and what that means here

The paper reports ≈0.86 mean Dice on AMOS22 organs and ≈0.70 on lesions.
0.70 is assistance, not delineation. Treat every prompted mask as a starting
point to be corrected, not a contour — and note that this whole program is a
viewer for research and QA convenience, **not a medical device**.

## Validation status

The port is verified structurally rather than numerically: the published
checkpoint's 475-tensor inventory is recorded in
`tests/data/segvol-tensors.csv` and asserted module by module, the network
assembles and runs against those exact key names and shapes in CI, and every
kernel is checked against hand-computed values. What has **not** been done is
a layer-by-layer numerical comparison against the reference implementation,
which is what the auto-segmentation engine's mean Dice 0.9995 rests on. Until
that is run, treat the output as untested against the original.
