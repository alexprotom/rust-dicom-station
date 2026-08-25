//! The 2-D primitives SAM 2 needs, expressed once against `burn`.
//!
//! Everything in the port that is not a matrix multiply goes through this
//! module, for two reasons. The first is that these are the operations whose
//! *semantics* differ between frameworks — padding conventions, the
//! half-pixel rule in interpolation, whether a `LayerNorm` reduces over the
//! channel axis or the last one — and a single definition is a single place
//! to be right. The second is that `tests/data/medsam2-ops.safetensors`
//! records PyTorch's answer for each of them on a small input, so the
//! assertions at the bottom of this file are the port's contract with the
//! reference implementation.
//!
//! Two conventions carried over from PyTorch, because the weights come from
//! PyTorch: a `Linear` weight is stored `[out, in]` and applied as
//! `x @ wT + b`, and a `Conv2d` weight is `[out, in / groups, kh, kw]` while a
//! `ConvTranspose2d` weight is `[in, out, kh, kw]`.

use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::module::{conv2d as burn_conv2d, conv_transpose2d, interpolate, max_pool2d};
use burn::tensor::ops::{ConvOptions, ConvTransposeOptions, InterpolateMode, InterpolateOptions};
use burn::tensor::{Tensor, TensorData};

/// PyTorch's `nn.LayerNorm` default.
pub const EPS: f64 = 1e-5;
/// The eps SAM 2 uses in the trunk, in `LayerNorm2d` and in the CXBlocks.
pub const EPS_6: f64 = 1e-6;

/// Build a tensor from a row-major slice.
pub fn from_slice<B: Backend, const D: usize>(
    data: &[f32],
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    debug_assert_eq!(
        data.len(),
        shape.iter().product::<usize>(),
        "{:?} does not hold {} values",
        shape,
        data.len()
    );
    Tensor::from_data(TensorData::new(data.to_vec(), shape), device)
}

/// Read a tensor back as a row-major `Vec`.
pub fn to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor is not f32")
}

/// `y = x @ wT + b`, with `w` stored `[out, in]` as PyTorch does.
pub fn linear<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    w: &Tensor<B, 2>,
    b: Option<&Tensor<B, 1>>,
) -> Tensor<B, D> {
    let wt: Tensor<B, D> = w.clone().transpose().unsqueeze::<D>();
    let y = x.matmul(wt);
    match b {
        Some(b) => y + b.clone().unsqueeze::<D>(),
        None => y,
    }
}

/// `Conv2d` with symmetric padding and optional grouping.
pub fn conv2d<B: Backend>(
    x: Tensor<B, 4>,
    w: &Tensor<B, 4>,
    b: Option<&Tensor<B, 1>>,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Tensor<B, 4> {
    burn_conv2d(
        x,
        w.clone(),
        b.cloned(),
        ConvOptions::new([stride, stride], [padding, padding], [1, 1], groups),
    )
}

/// `ConvTranspose2d` with kernel = stride = 2 — the decoder's upscaling.
pub fn conv_transpose2d_2x<B: Backend>(
    x: Tensor<B, 4>,
    w: &Tensor<B, 4>,
    b: Option<&Tensor<B, 1>>,
) -> Tensor<B, 4> {
    conv_transpose2d(
        x,
        w.clone(),
        b.cloned(),
        ConvTransposeOptions::new([2, 2], [0, 0], [0, 0], [1, 1], 1),
    )
}

/// 2 x 2 max pooling, `ceil_mode = false` — Hiera's query and residual pooling.
pub fn max_pool2x2<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    max_pool2d(x, [2, 2], [2, 2], [0, 0], [1, 1], false)
}

/// Nearest-neighbour upsampling by 2 — the FPN's top-down path.
pub fn upsample_nearest_2x<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let [_, _, h, w] = x.dims();
    interpolate(
        x,
        [h * 2, w * 2],
        InterpolateOptions {
            mode: InterpolateMode::Nearest,
            align_corners: false,
        },
    )
}

/// Bilinear resize with the half-pixel rule (`align_corners = false`).
pub fn resize_bilinear<B: Backend>(x: Tensor<B, 4>, size: [usize; 2]) -> Tensor<B, 4> {
    interpolate(
        x,
        size,
        InterpolateOptions {
            mode: InterpolateMode::Bilinear,
            align_corners: false,
        },
    )
}

/// Bicubic resize with the half-pixel rule — used exactly once, to stretch
/// the trunk's 7 x 7 background position embedding over the token grid.
pub fn resize_bicubic<B: Backend>(x: Tensor<B, 4>, size: [usize; 2]) -> Tensor<B, 4> {
    interpolate(
        x,
        size,
        InterpolateOptions {
            mode: InterpolateMode::Bicubic,
            align_corners: false,
        },
    )
}

/// `nn.LayerNorm` over the last axis, with PyTorch's biased variance.
pub fn layer_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    w: &Tensor<B, 1>,
    b: &Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, D> {
    let dim = D - 1;
    let mean = x.clone().mean_dim(dim);
    let centered = x - mean;
    let var = centered.clone().powi_scalar(2).mean_dim(dim);
    let normed = centered / var.add_scalar(eps).sqrt();
    normed * w.clone().unsqueeze::<D>() + b.clone().unsqueeze::<D>()
}

/// SAM 2's `LayerNorm2d`: statistics over the **channel** axis of an
/// `[N, C, H, W]` tensor, per spatial location, with a per-channel affine.
/// This is not `nn.LayerNorm` on a permuted view — mixing the two up is
/// silent and wrong.
pub fn layer_norm_2d<B: Backend>(
    x: Tensor<B, 4>,
    w: &Tensor<B, 1>,
    b: &Tensor<B, 1>,
    eps: f64,
) -> Tensor<B, 4> {
    let mean = x.clone().mean_dim(1);
    let centered = x - mean;
    let var = centered.clone().powi_scalar(2).mean_dim(1);
    let normed = centered / var.add_scalar(eps).sqrt();
    let c = normed.dims()[1];
    let w = w.clone().reshape([1, c, 1, 1]);
    let b = b.clone().reshape([1, c, 1, 1]);
    normed * w + b
}

/// Scaled dot-product attention over `[batch, heads, tokens, head_dim]`.
///
/// `q` may be shorter than `k` — Hiera's query-pooled blocks produce exactly
/// that, and so does the memory attention.
pub fn sdpa<B: Backend>(q: Tensor<B, 4>, k: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 4> {
    let dh = q.dims()[3];
    let scale = 1.0 / (dh as f64).sqrt();
    let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
    activation::softmax(scores, 3).matmul(v)
}

/// Zero-pad the two middle axes of a `[B, H, W, C]` tensor.
fn pad_hw<B: Backend>(x: Tensor<B, 4>, pad_h: usize, pad_w: usize) -> Tensor<B, 4> {
    let device = x.device();
    let [b, h, w, c] = x.dims();
    let x = if pad_h > 0 {
        Tensor::cat(vec![x, Tensor::zeros([b, pad_h, w, c], &device)], 1)
    } else {
        x
    };
    if pad_w > 0 {
        let [b, h2, _, c] = x.dims();
        debug_assert_eq!(h2, h + pad_h);
        Tensor::cat(vec![x, Tensor::zeros([b, h2, pad_w, c], &device)], 2)
    } else {
        x
    }
}

/// Cut `[B, H, W, C]` into `window x window` tiles, zero-padding first.
///
/// Returns the tiles as `[B * tiles, window, window, C]` and the padded grid,
/// which the caller needs to put them back.
pub fn window_partition<B: Backend>(x: Tensor<B, 4>, window: usize) -> (Tensor<B, 4>, [usize; 2]) {
    let [b, h, w, c] = x.dims();
    let pad_h = (window - h % window) % window;
    let pad_w = (window - w % window) % window;
    let x = pad_hw(x, pad_h, pad_w);
    let (hp, wp) = (h + pad_h, w + pad_w);
    let x = x
        .reshape([b, hp / window, window, wp / window, window, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b * (hp / window) * (wp / window), window, window, c]);
    (x, [hp, wp])
}

/// The inverse, cropping back to `hw` afterwards.
pub fn window_unpartition<B: Backend>(
    x: Tensor<B, 4>,
    window: usize,
    pad_hw: [usize; 2],
    hw: [usize; 2],
) -> Tensor<B, 4> {
    let [hp, wp] = pad_hw;
    let [h, w] = hw;
    let c = x.dims()[3];
    let tiles = (hp / window) * (wp / window);
    let b = x.dims()[0] / tiles;
    let x = x
        .reshape([b, hp / window, wp / window, window, window, c])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b, hp, wp, c]);
    if hp > h || wp > w {
        x.slice([0..b, 0..h, 0..w, 0..c])
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::{load_safetensors, WTensor};
    use std::collections::HashMap;
    use std::path::Path;

    type B = burn::backend::NdArray;

    fn fixtures() -> HashMap<String, WTensor> {
        load_safetensors(Path::new("tests/data/medsam2-ops.safetensors"))
            .expect("op fixtures; regenerate with tools/gen_ops_fixtures.py")
    }

    fn dev() -> burn::tensor::Device<B> {
        Default::default()
    }

    fn get<const D: usize>(f: &HashMap<String, WTensor>, key: &str) -> Tensor<B, D> {
        let t = f.get(key).unwrap_or_else(|| panic!("fixture {key}"));
        let mut shape = [1usize; D];
        assert_eq!(t.shape.len(), D, "{key} has rank {}", t.shape.len());
        shape.copy_from_slice(&t.shape);
        from_slice(&t.data, shape, &dev())
    }

    /// Assert two tensors agree to within what f32 accumulation allows.
    fn same<const D: usize>(got: Tensor<B, D>, want: Tensor<B, D>, what: &str) {
        assert_eq!(got.dims(), want.dims(), "{what}: shape");
        let g = to_vec(got);
        let w = to_vec(want);
        let mut worst = 0.0f32;
        for (a, b) in g.iter().zip(w.iter()) {
            let d = (a - b).abs() / (1.0 + b.abs());
            worst = worst.max(d);
        }
        // f32 reductions in a different order: the 147-term dot product of
        // the 7x7 patch embedding is the widest one here and lands around
        // 4e-6.
        assert!(worst < 1e-5, "{what}: relative error {worst:e}");
    }

    #[test]
    fn convolutions_match_pytorch() {
        let f = fixtures();
        for (name, stride, padding, groups) in [
            ("conv_k7s4p3", 4, 3, 1),
            ("conv_k3s2p1", 2, 1, 1),
            ("conv_k1", 1, 0, 1),
            ("conv_dw", 1, 3, 6),
        ] {
            let y = conv2d(
                get::<4>(&f, &format!("{name}.x")),
                &get::<4>(&f, &format!("{name}.w")),
                Some(&get::<1>(&f, &format!("{name}.b"))),
                stride,
                padding,
                groups,
            );
            same(y, get::<4>(&f, &format!("{name}.y")), name);
        }
    }

    #[test]
    fn the_transposed_convolution_matches_pytorch() {
        let f = fixtures();
        let y = conv_transpose2d_2x(
            get::<4>(&f, "convt_k2s2.x"),
            &get::<4>(&f, "convt_k2s2.w"),
            Some(&get::<1>(&f, "convt_k2s2.b")),
        );
        same(y, get::<4>(&f, "convt_k2s2.y"), "convt_k2s2");
    }

    #[test]
    fn max_pooling_drops_the_odd_row_like_ceil_mode_false() {
        let f = fixtures();
        let y = max_pool2x2(get::<4>(&f, "maxpool2x2.x"));
        assert_eq!(y.dims(), [1, 3, 4, 4]);
        same(y, get::<4>(&f, "maxpool2x2.y"), "maxpool2x2");
    }

    #[test]
    fn interpolation_follows_the_half_pixel_rule() {
        let f = fixtures();
        same(
            resize_bilinear(get::<4>(&f, "interp_bilinear.x"), [20, 20]),
            get::<4>(&f, "interp_bilinear.y"),
            "bilinear",
        );
        same(
            upsample_nearest_2x(get::<4>(&f, "interp_nearest.x")),
            get::<4>(&f, "interp_nearest.y"),
            "nearest",
        );
        same(
            resize_bicubic(get::<4>(&f, "interp_bicubic.x"), [32, 32]),
            get::<4>(&f, "interp_bicubic.y"),
            "bicubic",
        );
    }

    #[test]
    fn the_activations_are_the_exact_ones() {
        let f = fixtures();
        let x = get::<1>(&f, "gelu.x");
        same(activation::gelu(x.clone()), get::<1>(&f, "gelu.y"), "gelu");
        same(activation::relu(x.clone()), get::<1>(&f, "relu.y"), "relu");
        same(activation::sigmoid(x), get::<1>(&f, "sigmoid.y"), "sigmoid");
    }

    #[test]
    fn both_layer_norms_match_pytorch() {
        let f = fixtures();
        same(
            layer_norm(
                get::<2>(&f, "layernorm_last.x"),
                &get::<1>(&f, "layernorm_last.w"),
                &get::<1>(&f, "layernorm_last.b"),
                EPS_6,
            ),
            get::<2>(&f, "layernorm_last.y"),
            "layer_norm",
        );
        same(
            layer_norm_2d(
                get::<4>(&f, "layernorm2d.x"),
                &get::<1>(&f, "layernorm2d.w"),
                &get::<1>(&f, "layernorm2d.b"),
                EPS_6,
            ),
            get::<4>(&f, "layernorm2d.y"),
            "layer_norm_2d",
        );
    }

    #[test]
    fn softmax_matmul_and_attention_match_pytorch() {
        let f = fixtures();
        same(
            activation::softmax(get::<2>(&f, "softmax.x"), 1),
            get::<2>(&f, "softmax.y"),
            "softmax",
        );
        same(
            get::<2>(&f, "matmul.a").matmul(get::<2>(&f, "matmul.b")),
            get::<2>(&f, "matmul.y"),
            "matmul",
        );
        same(
            sdpa(
                get::<4>(&f, "sdpa.q"),
                get::<4>(&f, "sdpa.k"),
                get::<4>(&f, "sdpa.v"),
            ),
            get::<4>(&f, "sdpa.y"),
            "sdpa",
        );
    }

    #[test]
    fn linear_applies_the_pytorch_weight_layout() {
        let d = dev();
        // y = x @ wT + b, with w [out, in]
        let x: Tensor<B, 2> = from_slice(&[1.0, 2.0, 3.0, 4.0], [2, 2], &d);
        let w: Tensor<B, 2> = from_slice(&[1.0, 0.0, 0.0, 2.0, 1.0, 1.0], [3, 2], &d);
        let b: Tensor<B, 1> = from_slice(&[0.5, -0.5, 0.0], [3], &d);
        let y = to_vec(linear(x, &w, Some(&b)));
        assert_eq!(y, vec![1.5, 3.5, 3.0, 3.5, 7.5, 7.0]);
    }

    #[test]
    fn windows_survive_a_partition_round_trip_with_padding() {
        let d = dev();
        // 3 x 5 grid, window 2 -> padded to 4 x 6, six windows, cropped back.
        let n = 3 * 5 * 2;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x: Tensor<B, 4> = from_slice(&data, [1, 3, 5, 2], &d);
        let (win, pad) = window_partition(x.clone(), 2);
        assert_eq!(pad, [4, 6]);
        assert_eq!(win.dims(), [6, 2, 2, 2]);
        let back = window_unpartition(win, 2, pad, [3, 5]);
        assert_eq!(back.dims(), [1, 3, 5, 2]);
        assert_eq!(to_vec(back), data);
    }

    #[test]
    fn a_grid_that_divides_evenly_is_not_padded() {
        let d = dev();
        let data: Vec<f32> = (0..(4 * 4)).map(|i| i as f32).collect();
        let x: Tensor<B, 4> = from_slice(&data, [1, 4, 4, 1], &d);
        let (win, pad) = window_partition(x.clone(), 2);
        assert_eq!(pad, [4, 4]);
        // window 0 is the top-left 2 x 2 block
        assert_eq!(
            to_vec(win.clone().slice([0..1, 0..2, 0..2, 0..1])),
            vec![0.0, 1.0, 4.0, 5.0]
        );
        assert_eq!(to_vec(window_unpartition(win, 2, pad, [4, 4])), data);
    }
}
