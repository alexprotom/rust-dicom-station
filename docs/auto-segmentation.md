# Automatic multi-organ segmentation

A **pure-Rust re-implementation of
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) v2 CT
inference** (Wasserthal et al., *Radiology: Artificial Intelligence* 2023):
no Python, ONNX runtime or vendor toolkits, at build or run time. It
segments a CT into up to **117 anatomical structures** with the official
nnU-Net models of TotalSegmentator's openly licensed "total" task, locally
on the CPU or on any GPU via wgpu.

## Models

| Variant | nnU-Net workspace(s) | Classes | Download | Practical use |
|---|---|---|---|---|
| **3 mm (fast)** | 297 | all 117 in one model | ≈ 135 MB | good quality, practical on any CPU |
| **1.5 mm (high quality)** | 291-295 (organs / vertebrae / cardiac / muscles / ribs) | 117 across five sub-models, individually selectable | ≈ 1.2 GB | reference quality; GPU recommended |
| **6 mm (preview)** | 298 | all 117 in one model | ≈ 135 MB | coarse but very fast |

All variants are nnU-Net v2 `PlainConvUNet` 3D networks (5 stages at
3/6 mm, 6 at 1.5 mm): Conv3d → InstanceNorm → LeakyReLU blocks,
strided-conv downsampling, a transposed-conv decoder with skip connections,
and a 1×1×1 segmentation head - rebuilt at load time from each model's
`plans.json`, not hard-coded.

## Using it in the viewer

The **🔬 Auto-segmentation** section of the *Structure auto tools* module
(*Modules ▶ Structure auto tools*, right panel, F10; the workspace is the
module's **Workspace A / B** row; the four sections share one layout, see
[architecture.md](architecture.md#the-tool-windows-and-the-modules)):

* **Model** - one of the three variants; the dialog shows whether weights
  are cached or how much will be downloaded once. For 1.5 mm the five
  sub-models can be toggled individually - *organs* + *cardiac* alone takes
  a fifth of the full set's time.
* **Options ▸ Compute** - *Auto* (GPU when available, else CPU), *GPU*, or
  *CPU*.
* **Options ▸ Model folder** - the root every engine downloads into
  (`%LOCALAPPDATA%\RustDICOMStation\models` on Windows,
  `~/.local/share/RustDICOMStation/models` on Linux, by default); this
  engine uses its `totalsegmentator/` sub-folder. Persisted as `models_dir`
  in `viewer_settings.txt`.

* **Run on** - shown only when the displayed series is a phase of a 4D
  group: *the displayed series*, or *every phase of <group> (n)*; see
  below.

**▶ Segment** runs in the background; the buttons become a progress row
(device, bar, message, **Cancel** - effective during download, conversion
and between inference tiles), mirrored in the sidebar. A **results dialog**
then lists every detected structure with its volume, and asks what the
checked ones become:

* **Output ▸ segments** - ordinary editable segmentations in the
  segmentation series bound to the displayed image series:
  brush/erase/grow correction, live 3D view, per-structure colours from a
  curated anatomical palette; exports as DICOM SEG.
* **Output ▸ RT structures** - contours in an RT structure set, and **no
  segments**: what a planning system reads. A second row picks the set -
  any set of the workspace, or **➕ new structure set** with a name of its
  own (blank means *Auto-segmentation*) - so a run can start a fresh set
  from this window rather than land in whatever set happened to be active.
  The structures get the RT ROI Interpreted Type `ORGAN`; a name already in
  the set gets a `(2)` suffix rather than a twin.
* **Output ▸ both** - the segments, and contours made from them.

Materialize only what you need: every mask is a full-volume voxel map
(≈ 35 MB at 512 × 512 × 133).

If the workspace is switched or modified during a run, the result is
discarded with a message rather than applied to the wrong volume.

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

The same *Run on* row is on **Body contour** and **Prompt segmentation**
(the prompt is placed at the same point in patient coordinates on every
phase, so the crosshair need only be set once); **Slice propagation**
stays a single-series tool, its box being drawn on one slice of one
series.

## Weight acquisition and caching

On first use of a variant the viewer downloads the official weight zip(s)
from the TotalSegmentator GitHub release (TLS via rustls with the
**operating-system certificate store**) and converts them natively:

1. `plans.json` is extracted and validated;
2. `fold_0/checkpoint_final.pth` - a PyTorch zip/pickle checkpoint - is
   parsed by the built-in **torch-pickle reader** (`autoseg/pickle.rs`, a
   minimal pickle virtual machine covering `torch.save`'s persistent
   storage IDs, `_rebuild_tensor_v2`, shapes, strides and offsets);
3. the `network_weights` state dict is re-saved as `model.safetensors` in
   the model folder; duplicate tensor aliases (`*.all_modules.*`,
   `decoder.encoder.*`) are dropped.

Later runs load the cache with no network access. For **air-gapped
machines**, run any variant once on a connected machine and copy
`models/totalsegmentator/<variant>/` (`model.safetensors` + `plans.json`).
Installations from before the single model folder are migrated at startup:
an old `autoseg_models/` beside the executable is renamed into place,
nothing is downloaded twice.

## The inference pipeline

The pipeline mirrors TotalSegmentator exactly:

1. **Canonical orientation.** Axes (from the DICOM direction cosines) are
   permuted/flipped to the closest canonical [S, A, R] frame, the order
   nnU-Net models were trained in.
2. **Resampling** to the model's isotropic spacing (1.5 / 3 / 6 mm),
   trilinear, with `scipy.ndimage.zoom`'s endpoint-aligned coordinate
   convention (TotalSegmentator's resampler), including its int32
   truncation.
3. **Normalization** per model: clip to the training-set foreground's
   [0.5, 99.5] HU percentiles, then z-score with the workspace-fingerprint
   mean/std - all constants from `plans.json`.
4. **Sliding-window inference** with nnU-Net's exact tiling (step 0.8 ×
   patch for the "total" task), Gaussian importance weighting
   (σ = patch/8), zero-padded borders, no mirroring test-time augmentation
   (the "total" models are trained without it). The logit accumulator is a
   **ring buffer along the leading patient axis**: rows are finalized
   (argmax → label) once no future tile can touch them, so peak memory
   stays ≈ classes × patch-depth × slice-area floats regardless of scan
   length.
5. **Label merging** (1.5 mm variant): each sub-model's local labels map
   onto the global 117-class ids; later sub-models overwrite earlier ones
   at overlaps, in TotalSegmentator's order.
6. **Back-mapping** to the original CT grid by nearest neighbor (order-0).

## Compute engines

**CPU** - a hand-written engine (`autoseg/cpu.rs`): 3D convolution as
per-output-slice im2col + pure-Rust SIMD GEMM (the `gemm` crate),
parallelized over slices with rayon; the transposed conv
(kernel = stride = 2) is a GEMM plus disjoint scatter; instance norm and
LeakyReLU are fused. That is 15-50× faster than a direct convolution loop
(25-100 GFLOP/s measured on modest hardware): a thorax CT with the 3 mm
model takes well under a minute on a desktop CPU, ≈ 3.5 min even on a
throttled 2-core VM.

**GPU** - the same network through
[burn](https://github.com/tracel-ai/burn)'s **wgpu** backend
(`autoseg/gpu.rs`): Vulkan / DX12 / Metal, i.e. NVIDIA, AMD, Intel and
Apple GPUs, with **no CUDA toolkit or vendor SDK**; kernels are generated
and autotuned at runtime. Weights upload once per model; patches stream
through. *Auto* probes for a usable adapter with a self-test and falls back
to the CPU. The GPU path is optional at build time (`gpu` cargo feature, on
by default; `--no-default-features` builds a CPU-only viewer without the
burn dependency tree).

## Validation

Verified against the reference at three levels:

* **Network equivalence** - on an identical preprocessed patch, the Rust
  forward pass reproduces the actual 3 mm checkpoint's PyTorch/nnU-Net
  logits to ≈ 1 × 10⁻⁴ absolute (float accumulation-order noise on logits
  spanning ±86) with **100 % argmax agreement**.
* **End-to-end** - on the bundled example study, the full pipeline agrees
  with the official Python TotalSegmentator at **mean Dice 0.9995 across
  90 detected structures** (worst 0.992, spleen 1.0000); residual
  differences are single-voxel boundary tie-breaks.
* **CPU vs GPU** - the wgpu engine produced **bit-identical labels** to
  the CPU engine over a full run (34.9 M voxels, zero differences, on a
  software Vulkan implementation).

Unit tests pin the sliding-window step positions and resampling
conventions to nnU-Net/scipy reference values; `tests/autoseg.rs`
assembles a miniature network from synthetic tensors with the exact
checkpoint key naming and verifies the forward pass; an `#[ignore]`d
end-to-end test runs the real 3 mm model against the bundled example data:

```
RDS_AUTOSEG_MODELS=path/to/models/totalsegmentator \
  cargo test --release --test autoseg -- --ignored
```

## Command-line tools

Two developer examples ship with the source:

```
# headless segmentation → labels .bin + organ-table .json
cargo run --release --example autoseg_cli -- <dicom_dir> <out_prefix> \
    [--variant fast3|highres|preview6] [--device auto|gpu|cpu] \
    [--models DIR] [--parts organs,cardiac,...]

# dump one preprocessed patch + its logits (for numerical comparison)
cargo run --release --example autoseg_probe -- <dicom_dir> <models_dir> \
    total_3mm <out_prefix>
```

## The 117 classes

Global label ids follow TotalSegmentator v2's `class_map["total"]`:

| Ids | Group (1.5 mm sub-model) | Structures |
|---|---|---|
| 1-24 | organs | spleen, kidney R/L, gallbladder, liver, stomach, pancreas, adrenal gland R/L, lung upper/lower lobe L, lung upper/middle/lower lobe R, esophagus, trachea, thyroid, small bowel, duodenum, colon, urinary bladder, prostate, kidney cyst L/R |
| 25-50 | vertebrae | sacrum, S1, L5-L1, T12-T1, C7-C1 |
| 51-68 | cardiac | heart, aorta, pulmonary vein, brachiocephalic trunk, subclavian artery R/L, common carotid artery R/L, brachiocephalic vein L/R, left atrial appendage, superior/inferior vena cava, portal + splenic vein, iliac artery L/R, iliac vena L/R |
| 69-91 | muscles | humerus L/R, scapula L/R, clavicula L/R, femur L/R, hip L/R, spinal cord, gluteus maximus/medius/minimus L+R, autochthon L/R, iliopsoas L/R, brain, skull |
| 92-117 | ribs | ribs left 1-12, ribs right 1-12, sternum, costal cartilages |

The exact per-id table lives in `src/autoseg/classes.rs` and is verified
against upstream `map_to_binary.py` by unit test.

## Licensing and citation

The TotalSegmentator **code and the weights of the "total" task are
Apache-2.0** - the authors publish these models as "openly available for
any usage", commercial use included. (Other TotalSegmentator sub-tasks
have restricted licenses; this viewer only uses the open "total" task.)

If you use the auto-segmentation in academic work, cite:

> Wasserthal, J. et al. *TotalSegmentator: Robust Segmentation of 104
> Anatomic Structures in CT Images.* Radiology: Artificial Intelligence
> 2023. <https://doi.org/10.1148/ryai.230024>
>
> Isensee, F. et al. *nnU-Net: a self-configuring method for deep
> learning-based biomedical image segmentation.* Nature Methods 2021.
> <https://doi.org/10.1038/s41592-020-01323-z>

## Troubleshooting

* **"no usable wgpu adapter found"** - no Vulkan/DX12/Metal device
  (headless machine, missing driver). *Auto* silently uses the CPU;
  forcing *GPU* reports the error. If the whole program will not start,
  that is the related and more common failure - a Windows machine
  advertising a Vulkan driver that cannot create a device; see
  [viewer.md](viewer.md#graphics-backend). The backend chosen there governs
  inference too, so a machine forced onto Direct3D 12 runs the networks on
  Direct3D 12.
* **Download fails behind a proxy** - the downloader uses the OS trust
  store, so a corporate/clinical inspection proxy's CA installed
  system-wide is honored; fully offline machines, see the air-gapped note
  above.
* **Memory** - the 3 mm model peaks around 2-3 GB for a thorax CT
  (streaming accumulator + activations); the 1.5 mm variant needs several
  GB more, and each materialized mask ≈ volume-size bytes.
* As with everything in this viewer: research and QA use - not a medical
  device, not for clinical decision-making.
