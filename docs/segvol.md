# Prompt-driven segmentation

The auto-segmentation engine ([auto-segmentation.md](auto-segmentation.md))
gives 117 anatomical classes with no interaction — all normal anatomy,
which for a radiotherapy station means organs at risk and nothing else: no
GTV, no nodal disease, no post-surgical cavity, no recurrence. Those take
longest to draw, and no fixed-class model will ever cover them, because
they are patient-specific by definition.

This engine is the complement: point at something — a box, a click, or a
structure name in plain text — and it segments it. It is a pure-Rust
re-implementation of [SegVol](https://github.com/BAAI-DCAI/SegVol)
(Du et al., NeurIPS 2024) — no Python, no ONNX Runtime, no CUDA.

## Using it

**Tools ▶ 🧠 Prompt-segment dataset A…**, or the **🧠 Prompt…** button in
the sidebar *Segmentations* section, opens the tool window (**🧠 Prompt
segmentation — dataset A**; the three engines share one window layout, see
[architecture.md](architecture.md#the-three-engine-windows)). It stays open
across runs and reports each result on its last line.

Move the crosshair onto the structure first; the prompt is anchored to it.

| Prompt | What it does | Best for |
|---|---|---|
| **Box** | A box centred on the crosshair, with a half-extent in millimetres | Lesions, targets — the most reliable prompt |
| **Point** | A single foreground click at the crosshair | Compact, well-separated structures |
| **Text** | A structure name through the model's trained template | Anatomy the 117-class model does not cover |

The result is an ordinary segmentation — editable with brush and eraser,
visible in 3D, convertible to RTSTRUCT — so the usual loop is *prompt, fix
by hand, export*.

### Options

* **Refinement pass** — the second, sliding-window pass; without it a
  single coarse pass: much faster, much blockier.
* **Skip the search pass (box only)** — the first pass only *locates* the
  structure and is redundant with a hand-drawn box; skipping it roughly
  halves the work and avoids losing small lesions to the downsample. **This
  departs from the reference implementation**, which always runs both; off
  by default.
* **Threshold** — probability cut on the network's output, 0.5 by default.
* **Compute** — *Auto* (GPU for the image encoder when available, else
  CPU), *GPU*, or *CPU*.
* **Model folder** — the root every engine downloads into (default: see
  [auto-segmentation.md](auto-segmentation.md#using-it-in-the-viewer));
  this engine uses its `segvol/` sub-folder.

A box drawn in the image, with include / exclude clicks and a live preview,
is what [slice propagation](medsam2.md) offers; this window keeps the
crosshair-anchored prompt because SegVol's box is three-dimensional.

## Headless

```
cargo run --release --example segvol_cli -- <DICOM_DIR> \
    [--models DIR] [--device auto|gpu|cpu] \
    [--box z0,y0,x0,z1,y1,x1] [--point z,y,x] [--negative-point z,y,x] \
    [--text liver] [--no-zoom-in] [--fast-box] [--threshold F] [--out mask.raw]
```

`--models` is the engine's folder, `segvol/` in the viewer's model folder
by default.

Coordinates are in the *prepared* grid — canonically oriented `[S, A, R]`
and cropped to the foreground, what the network sees. `--out` writes one
byte per voxel on the original volume's grid.

`examples/segvol_probe` fetches the checkpoint, checks it against the
layout the port expects and prints the tensor inventory.

## How it works

The network only accepts a **32 × 256 × 256** volume — not a configuration
choice: the image encoder's position embedding is a learned 2048-token
parameter with no interpolation logic, and the mask decoder contains a
`LayerNorm` whose shape is the literal `(192, 16, 32, 32)` activation. So
the same fixed graph runs twice — over the whole volume squashed into that
shape to find the structure, then as a sliding window over a crop around
what the first pass found.

Preprocessing is unlike the nnU-Net engine's: **no HU window and no
resample to a target spacing**. Intensities are normalized from the
volume's own statistics — threshold at the mean, take the 0.05/99.95
percentiles and the mean and standard deviation of the voxels above it,
clip and z-score — then min-max to [0,1] and crop the resulting zero rim.

| | |
|---|---|
| Image encoder | MONAI 3-D ViT, 12 blocks, width 768, 2048 tokens, global attention |
| Prompt encoder | SAM's, 3-D: random-Fourier positions, learned point/box embeddings |
| Mask decoder | Two-way transformer, depth 2, hypernetwork mask filters |
| Text tower | CLIP ViT-B/32, frozen, shipped inside the checkpoint |
| Parameters | 180,891,293 across 475 tensors |

Roughly 6 s per window on a desktop CPU; the image encoder is ~97% of that
and runs on any GPU `wgpu` can drive when the `gpu` feature is on (default).

### Why the text tower is native

The smaller build — a table of precomputed embeddings over a curated
structure vocabulary — was rejected for two reasons: generating it means
running PyTorch offline, so the shipped artifact would be downstream of a
Python step this project cannot reproduce, and it removes free-text
prompts, SegVol's headline capability. The tower's weights are inside the
checkpoint being downloaded anyway, so the native tower costs no extra
bytes over the wire beyond the tokenizer's two small data files; encoded
prompts are cached by string, so repeated prompts skip the tower entirely,
which recovers the table's practical benefit.

## Weights, and their licence

The checkpoint (~724 MB) is downloaded from
[huggingface.co/BAAI/SegVol](https://huggingface.co/BAAI/SegVol) on first
use, with the CLIP tokenizer's two small data files, into `models/segvol/`
under the model folder, and converted once into a `safetensors` cache
beside it. The tool window says whether the weights are cached or how much
a run will download.

**The SegVol code is MIT, but the weights carry no licence declaration at
all** — no licence tag, no LICENSE file — and the training corpus aggregates
25 datasets whose terms differ, several of them non-commercial. This is
unlike TotalSegmentator's Apache-2.0 weights.

Consequently the weights are only ever fetched to your own machine, at your
request. Nothing is redistributed with this program, and unlike the
auto-segmentation models they are **not** offered in the installer's optional
pre-download.

## Accuracy, and what that means here

The paper reports ≈0.86 mean Dice on AMOS22 organs and ≈0.70 on lesions —
assistance, not delineation. Treat every prompted mask as a starting point
to be corrected, not a contour — and this whole program is a viewer for
research and QA convenience, **not a medical device**.

## Validation status

The port is verified structurally, not numerically: the published
checkpoint's 475-tensor inventory is recorded in
`tests/data/segvol-tensors.csv` and asserted module by module, the network
assembles and runs against those exact key names and shapes in CI, and
every kernel is checked against hand-computed values. Not yet done: a
layer-by-layer numerical comparison against the reference implementation
— what the auto-segmentation engine's mean Dice 0.9995 rests on. Until
then, treat the output as untested against the original.

That validation should use HF `model_segvol_single.py` as the normative
reference — self-contained, what the published demo runs; the GitHub and
Hugging Face pipelines differ in places (reorientation, padding). Dump its
activations at four cut points — after the patch embed, after ViT block 6,
after the final encoder norm, and at the decoder's low-resolution logits —
and compare each to 1e-4 absolute; then a full two-pass, box-prompted run
on the bundled example study, compared by Dice against the reference:
**> 0.99 is the pass mark; below that is a bug, not noise.** CPU and GPU
masks on identical input should agree bit-for-bit: the final step is a
threshold, not an argmax over near-equal logits. Running the Python
reference at development time does not violate the one-language rule,
which is about what ships in the binary — a comparison harness is a test
fixture.
