# Automatic multi-organ segmentation

A complete, **pure-Rust re-implementation of
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) v2 CT
inference** (Wasserthal et al., *Radiology: Artificial Intelligence* 2023).
No Python, no ONNX runtime, no vendor toolkits — at build time or at run
time. The viewer segments a CT into up to **117 anatomical structures**
using the official nnU-Net models of TotalSegmentator's openly licensed
"total" task, running locally on the CPU or on any GPU via wgpu.

## Models

| Variant | nnU-Net dataset(s) | Classes | Download | Practical use |
|---|---|---|---|---|
| **3 mm (fast)** | 297 | all 117 in one model | ≈ 135 MB | good quality, practical on any CPU |
| **1.5 mm (high quality)** | 291–295 (organs / vertebrae / cardiac / muscles / ribs) | 117 across five sub-models, individually selectable | ≈ 1.2 GB | reference quality; GPU recommended |
| **6 mm (preview)** | 298 | all 117 in one model | ≈ 135 MB | coarse but very fast |

All variants are nnU-Net v2 `PlainConvUNet` 3D networks (5 stages at
3/6 mm, 6 at 1.5 mm): blocks of Conv3d → InstanceNorm → LeakyReLU with
strided-conv downsampling, a transposed-conv decoder with skip
connections, and a 1×1×1 segmentation head. The architecture is rebuilt at
load time from each model's `plans.json`, so the code carries no
hard-coded network definition.

## Using it in the viewer

*Tools ▶ 🤖 Auto-segment dataset A/B…* or the **🤖 Auto…** button in the
sidebar *Segmentations* section opens the run dialog:

* **Model** — one of the three variants (the dialog shows whether weights
  are already cached or how much will be downloaded once). For 1.5 mm, the
  five sub-models can be toggled individually — running only *organs* +
  *cardiac*, for instance, takes a fifth of the time of the full set.
* **Compute** — *Auto* (GPU when available, else CPU), *GPU*, or *CPU*.
* **Model folder** — where weights are cached; defaults to
  `autoseg_models/` next to the executable and is persisted in
  `viewer_settings.txt`.

The run executes on a background thread with a progress bar and **Cancel**
(effective during download, conversion and between inference tiles). When
it finishes, a **results dialog** lists every detected structure with its
volume; the checked ones are materialized as ordinary editable
segmentations — brush/erase/grow correction, live 3D view, per-structure
colors from a curated anatomical palette — and can optionally be converted
to **RTSTRUCT contours** in the same step, after which they render like any
ROI and ride the DICOM export. Materializing only what you need matters on
large studies: every mask is a full-volume voxel map (≈ 35 MB at
512 × 512 × 133).

If the dataset is switched or modified while a run is in flight, the
result is discarded with a message rather than applied to the wrong
volume.

## Weight acquisition and caching

On first use of a variant the viewer downloads the official weight zip(s)
from the TotalSegmentator GitHub release (TLS via rustls using the
**operating-system certificate store**, so corporate/clinical inspection
proxies with custom CAs work), then converts them natively:

1. `plans.json` is extracted and validated;
2. `fold_0/checkpoint_final.pth` — a PyTorch zip/pickle checkpoint — is
   parsed by the built-in **torch-pickle reader** (`autoseg/pickle.rs`, a
   minimal pickle virtual machine that understands `torch.save`'s
   serialization: persistent storage IDs, `_rebuild_tensor_v2`, shapes,
   strides and offsets);
3. the `network_weights` state dict is re-saved as
   `model.safetensors` in the model folder. Duplicate tensor aliases in
   the checkpoint (`*.all_modules.*`, `decoder.encoder.*`) are dropped.

Subsequent runs load the cache directly with no network access. For
**air-gapped machines**, run any variant once on a connected machine and
copy the model folder (`autoseg_models/<variant>/` containing
`model.safetensors` + `plans.json`); the viewer never needs the network
again.

## The inference pipeline

The pipeline mirrors TotalSegmentator exactly:

1. **Canonical orientation.** The volume's axes (from the DICOM direction
   cosines) are permuted/flipped to the closest canonical [S, A, R] frame
   — the axis order nnU-Net models were trained in.
2. **Resampling** to the model's isotropic spacing (1.5 / 3 / 6 mm),
   trilinear, using the endpoint-aligned coordinate convention of
   `scipy.ndimage.zoom` (the resampler TotalSegmentator uses), including
   its int32 truncation.
3. **Normalization** per model: clip to the [0.5, 99.5] HU percentiles of
   the training-set foreground, then z-score with the dataset-fingerprint
   mean/std — all constants read from `plans.json`.
4. **Sliding-window inference** with nnU-Net's exact tiling (step 0.8 ×
   patch for the "total" task), Gaussian importance weighting
   (σ = patch/8), zero-padded borders, and no mirroring test-time
   augmentation (the "total" models are trained without it). The logit
   accumulator is a **ring buffer along the leading patient axis**: rows
   are finalized (argmax → label) as soon as no future tile can touch
   them, so peak memory stays ≈ classes × patch-depth × slice-area floats
   regardless of scan length — long whole-body scans stay bounded.
5. **Label merging** (1.5 mm variant): each sub-model's local labels map
   onto the global 117-class ids; later sub-models overwrite earlier ones
   at overlaps, in TotalSegmentator's order.
6. **Back-mapping** to the original CT grid by nearest neighbor (order-0),
   exactly as the reference implementation resamples its label map back.

## Compute engines

**CPU** — a hand-written inference engine (`autoseg/cpu.rs`): 3D
convolution as per-output-slice im2col + pure-Rust SIMD GEMM (the `gemm`
crate), parallelized over slices with rayon; the transposed conv
(kernel = stride = 2) is a GEMM plus disjoint scatter; instance norm and
LeakyReLU are fused. This is 15–50× faster than a direct convolution loop
— measured 25–100 GFLOP/s on modest hardware — which is what makes CPU
inference practical: a thorax CT with the 3 mm model takes well under a
minute on a desktop CPU (≈ 3.5 min even on a throttled 2-core VM).

**GPU** — the same network runs through
[burn](https://github.com/tracel-ai/burn)'s **wgpu** backend
(`autoseg/gpu.rs`): Vulkan / DX12 / Metal, i.e. NVIDIA, AMD, Intel and
Apple GPUs, with **no CUDA toolkit or vendor SDK** — kernels are generated
and autotuned at runtime. Weights are uploaded once per model; patches
stream through. *Auto* device selection probes for a usable adapter with a
self-test and falls back to the CPU. The GPU path is optional at build
time (`gpu` cargo feature, on by default; `--no-default-features` builds a
CPU-only viewer without the burn dependency tree).

## Validation

The implementation is verified against the reference at three levels:

* **Network equivalence** — on an identical preprocessed input patch, the
  Rust forward pass reproduces PyTorch/nnU-Net logits of the actual 3 mm
  checkpoint to ≈ 1 × 10⁻⁴ absolute (float accumulation-order noise on
  logits spanning ±86) with **100 % argmax agreement**.
* **End-to-end** — on the bundled example study, the full pipeline agrees
  with the official Python TotalSegmentator at **mean Dice 0.9995 across
  90 detected structures** (worst structure 0.992, spleen 1.0000);
  residual differences are single-voxel boundary tie-breaks.
* **CPU vs GPU** — the wgpu engine produced **bit-identical labels** to
  the CPU engine over a full run (34.9 M voxels, zero differences, tested
  on a software Vulkan implementation).

Unit tests pin the sliding-window step positions and resampling
conventions to nnU-Net/scipy reference values; `tests/autoseg.rs`
assembles a miniature network from synthetic tensors with the exact
checkpoint key naming and verifies the forward pass, and an `#[ignore]`d
end-to-end test runs the real 3 mm model against the bundled example data:

```
RDV_AUTOSEG_MODELS=path/to/autoseg_models \
  cargo test --release --test autoseg -- --ignored
```

## Command-line tools

Two developer examples ship with the source:

```
# headless segmentation → labels .bin + organ-table .json
cargo run --release --example autoseg_cli -- <dicom_dir> <out_prefix> \
    [--variant fast3|highres|preview6] [--device auto|cpu|gpu] \
    [--models-dir DIR] [--parts organs,cardiac,...]

# dump one preprocessed patch + its logits (for numerical comparison)
cargo run --release --example autoseg_probe -- <dicom_dir> <models_dir> \
    total_3mm <out_prefix>
```

## The 117 classes

Global label ids follow TotalSegmentator v2's `class_map["total"]`:

| Ids | Group (1.5 mm sub-model) | Structures |
|---|---|---|
| 1–24 | organs | spleen, kidney R/L, gallbladder, liver, stomach, pancreas, adrenal gland R/L, lung upper/lower lobe L, lung upper/middle/lower lobe R, esophagus, trachea, thyroid, small bowel, duodenum, colon, urinary bladder, prostate, kidney cyst L/R |
| 25–50 | vertebrae | sacrum, S1, L5–L1, T12–T1, C7–C1 |
| 51–68 | cardiac | heart, aorta, pulmonary vein, brachiocephalic trunk, subclavian artery R/L, common carotid artery R/L, brachiocephalic vein L/R, left atrial appendage, superior/inferior vena cava, portal + splenic vein, iliac artery L/R, iliac vena L/R |
| 69–91 | muscles | humerus L/R, scapula L/R, clavicula L/R, femur L/R, hip L/R, spinal cord, gluteus maximus/medius/minimus L+R, autochthon L/R, iliopsoas L/R, brain, skull |
| 92–117 | ribs | ribs left 1–12, ribs right 1–12, sternum, costal cartilages |

The exact per-id table lives in `src/autoseg/classes.rs` and is verified
against upstream `map_to_binary.py` by unit test.

## Licensing and citation

The TotalSegmentator **code and the weights of the "total" task are
Apache-2.0** — the authors publish these models as "openly available for
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

* **"no usable wgpu adapter found"** — no Vulkan/DX12/Metal device is
  available (headless machine, missing driver). *Auto* silently uses the
  CPU; forcing *GPU* reports the error.
* **Download fails behind a proxy** — the downloader uses the OS trust
  store, so an inspection proxy's CA installed system-wide is honored. For
  fully offline machines, copy a converted model folder from another
  machine (see above).
* **Memory** — the 3 mm model peaks around 2–3 GB for a thorax CT
  (streaming accumulator + activations); the 1.5 mm variant needs several
  GB more. Materialized masks add ≈ volume-size bytes each.
* As with everything in this viewer: research and QA use — not a medical
  device, not for clinical decision-making.
