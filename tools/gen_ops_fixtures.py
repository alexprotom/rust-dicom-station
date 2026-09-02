#!/usr/bin/env python3
"""Generate the op-parity fixtures for the MedSAM2 port.

Run once, offline, with PyTorch installed; the output is a small safetensors
file committed to `tests/data/`. Nothing in the shipped program depends on
PyTorch - this only records what PyTorch's semantics *are*, so the Rust
engine's kernels can be asserted against them.

    python3 tools/gen_ops_fixtures.py tests/data/medsam2-ops.safetensors
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(20260825)
out = {}


def put(name, **tensors):
    for k, v in tensors.items():
        out[f"{name}.{k}"] = v.detach().clone().contiguous().float()


def rnd(*shape):
    return torch.randn(*shape, dtype=torch.float32)


# --- convolutions ---------------------------------------------------------
# patch embed: Conv2d(3, 8, k7, s4, p3)
x, w, b = rnd(1, 3, 32, 32), rnd(8, 3, 7, 7), rnd(8)
put("conv_k7s4p3", x=x, w=w, b=b, y=F.conv2d(x, w, b, stride=4, padding=3))

# mask downsampler: Conv2d(4, 6, k3, s2, p1)
x, w, b = rnd(1, 4, 16, 16), rnd(6, 4, 3, 3), rnd(6)
put("conv_k3s2p1", x=x, w=w, b=b, y=F.conv2d(x, w, b, stride=2, padding=1))

# neck lateral / projections: Conv2d(6, 5, k1)
x, w, b = rnd(1, 6, 7, 7), rnd(5, 6, 1, 1), rnd(5)
put("conv_k1", x=x, w=w, b=b, y=F.conv2d(x, w, b))

# CXBlock depthwise: Conv2d(6, 6, k7, p3, groups=6)
x, w, b = rnd(1, 6, 12, 12), rnd(6, 1, 7, 7), rnd(6)
put("conv_dw", x=x, w=w, b=b, y=F.conv2d(x, w, b, padding=3, groups=6))

# decoder upscaling: ConvTranspose2d(6, 4, k2, s2)
x, w, b = rnd(1, 6, 8, 8), rnd(6, 4, 2, 2), rnd(4)
put("convt_k2s2", x=x, w=w, b=b, y=F.conv_transpose2d(x, w, b, stride=2))

# --- pooling --------------------------------------------------------------
# Hiera q-pool, on an odd grid so ceil_mode=False actually drops a row
x = rnd(1, 3, 9, 9)
put("maxpool2x2", x=x, y=F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=False))

# --- interpolation --------------------------------------------------------
x = rnd(1, 2, 8, 8)
put(
    "interp_bilinear",
    x=x,
    y=F.interpolate(x, size=(20, 20), mode="bilinear", align_corners=False),
)
x = rnd(1, 2, 5, 5)
put("interp_nearest", x=x, y=F.interpolate(x, scale_factor=2.0, mode="nearest"))
# the trunk's background position embedding, at reduced width
x = rnd(1, 3, 7, 7)
put(
    "interp_bicubic",
    x=x,
    y=F.interpolate(x, size=(32, 32), mode="bicubic", align_corners=False),
)

# --- pointwise / normalization -------------------------------------------
x = rnd(64) * 3.0
put("gelu", x=x, y=F.gelu(x))  # exact, erf-based
put("relu", x=x, y=F.relu(x))
put("sigmoid", x=x, y=torch.sigmoid(x))

x, w, b = rnd(5, 12), rnd(12), rnd(12)
put("layernorm_last", x=x, w=w, b=b, y=F.layer_norm(x, (12,), w, b, eps=1e-6))

# LayerNorm2d: mean/var over the channel axis only, per spatial location
x, w, b = rnd(1, 6, 4, 5), rnd(6), rnd(6)
u = x.mean(1, keepdim=True)
s = (x - u).pow(2).mean(1, keepdim=True)
y = (x - u) / torch.sqrt(s + 1e-6)
y = w[:, None, None] * y + b[:, None, None]
put("layernorm2d", x=x, w=w, b=b, y=y)

x = rnd(4, 9)
put("softmax", x=x, y=F.softmax(x, dim=-1))

a, bm = rnd(6, 5), rnd(5, 7)
put("matmul", a=a, b=bm, y=a @ bm)

# --- attention primitives -------------------------------------------------
# scaled dot-product attention, 2 heads of width 4, rectangular (q shorter
# than k, as Hiera's q-pooled blocks are)
q, k, v = rnd(1, 2, 6, 4), rnd(1, 2, 10, 4), rnd(1, 2, 10, 4)
put(
    "sdpa",
    q=q,
    k=k,
    v=v,
    y=F.scaled_dot_product_attention(q, k, v),
)


# --- resampling: PIL's own kernel, and PyTorch's antialiased one ----------
# MedSAM2 preprocesses with `PIL.Image.resize`, whose default is a bicubic
# kernel with a = -0.5 and a support that widens when shrinking; the mask
# prompt path uses `F.interpolate(..., antialias=True)`.
from PIL import Image  # noqa: E402

x = rnd(7, 5)
arr = x.numpy().astype("float32")
up = Image.fromarray(arr, mode="F").resize((13, 16))  # (width, height)
put("pil_up", x=x, y=torch.from_numpy(np.array(up)))

x = rnd(32, 32)
arr = x.numpy().astype("float32")
down = Image.fromarray(arr, mode="F").resize((12, 12))
put("pil_down", x=x, y=torch.from_numpy(np.array(down)))

x = rnd(1, 1, 16, 16)
put(
    "torch_bilinear_aa",
    x=x,
    y=F.interpolate(x, size=(4, 4), mode="bilinear", align_corners=False, antialias=True),
)


# --- the preprocessing pipeline, on one windowed slice --------------------
# `resize_grayscale_to_rgb_and_resize` then /255 and the ImageNet statistics,
# at a size that fits in a fixture rather than at 512.
slice_u8 = (torch.rand(40, 36, generator=torch.Generator().manual_seed(7)) * 255).to(
    torch.uint8
)
_target = 64
_img = Image.fromarray(slice_u8.numpy()).convert("RGB").resize((_target, _target))
_arr = np.array(_img).transpose(2, 0, 1).astype("float64") / 255.0
_arr -= np.array([0.485, 0.456, 0.406])[:, None, None]
_arr /= np.array([0.229, 0.224, 0.225])[:, None, None]
put(
    "preprocess",
    u8=slice_u8.to(torch.float32),
    pil_u8=torch.from_numpy(np.array(_img)[:, :, 0].astype("float32")),
    y=torch.from_numpy(_arr[None]).float(),
)

path = sys.argv[1] if len(sys.argv) > 1 else "medsam2-ops.safetensors"
save_file(out, path)
print(f"wrote {path}: {len(out)} tensors")

# --- SAM 2's own positional encodings -------------------------------------
# Taken from the reference implementation rather than reimplemented here, so
# the Rust port is asserted against the real thing and not against a second
# transcription of it.
from sam2.modeling.position_encoding import (  # noqa: E402
    PositionEmbeddingSine,
    PositionEmbeddingRandom,
    compute_axial_cis,
    apply_rotary_enc,
)

sine = PositionEmbeddingSine(num_pos_feats=8, temperature=10000, normalize=True)
put("pe_sine", y=sine(torch.zeros(1, 1, 3, 4)))

rnd_pe = PositionEmbeddingRandom(num_pos_feats=4)
put(
    "pe_random",
    gaussian=rnd_pe.positional_encoding_gaussian_matrix,
    dense=rnd_pe(size=(3, 4)),
    coords=torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),
    y=rnd_pe.forward_with_coords(
        torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]), (64, 64)
    ),
)

# 2-D axial RoPE: head dim 16 over a 3 x 4 grid (t_x = i % end_x)
cis = compute_axial_cis(dim=16, end_x=4, end_y=3, theta=10000.0)
q = rnd(1, 2, 12, 16)
k = rnd(1, 2, 12, 16)
qr, kr = apply_rotary_enc(q, k, freqs_cis=cis, repeat_freqs_k=False)
put("rope", freqs_real=torch.view_as_real(cis)[..., 0], freqs_imag=torch.view_as_real(cis)[..., 1], q=q, k=k, q_out=qr, k_out=kr)

# the same, with the keys three times as long: `repeat_freqs_k` tiles the
# rotations across memory frames
k_long = rnd(1, 2, 36, 16)
qr2, kr2 = apply_rotary_enc(q, k_long, freqs_cis=cis, repeat_freqs_k=True)
put("rope_repeat", k=k_long, q_out=qr2, k_out=kr2)

save_file(out, path)
print(f"rewrote {path}: {len(out)} tensors")
