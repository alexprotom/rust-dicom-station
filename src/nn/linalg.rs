//! The dense kernels a transformer needs: matrix multiply, layer
//! normalization, softmax, and the three activation functions the SegVol
//! network uses.
//!
//! Matrix multiplication goes through the `gemm` crate's SIMD kernels — the
//! same ones the auto-segmentation convolutions use. Everything else is
//! memory-bound and hand-rolled with `rayon`.
//!
//! Conventions follow PyTorch exactly, because the weights come from PyTorch:
//! a `Linear` weight is stored `[out, in]` and applied as `x @ wᵀ + b`;
//! `LayerNorm` divides by the **biased** variance and defaults to `eps` 1e-5.

use rayon::prelude::*;

use super::tensor::Mat;

/// PyTorch's `nn.LayerNorm` default.
pub const LAYER_NORM_EPS: f32 = 1e-5;

fn parallelism() -> gemm::Parallelism {
    match rayon::current_num_threads() {
        0 | 1 => gemm::Parallelism::None,
        n => gemm::Parallelism::Rayon(n),
    }
}

/// `y = x @ wᵀ + b`, with `w` stored `[out, in]` row-major as PyTorch does.
///
/// Passing the weight transposed costs nothing: `gemm` takes independent row
/// and column strides, so the transpose is expressed in the strides rather
/// than by materializing `wᵀ`.
pub fn linear(x: &Mat, w: &[f32], out: usize, bias: Option<&[f32]>) -> Mat {
    let (n, k) = (x.rows, x.cols);
    assert_eq!(w.len(), out * k, "weight is not [{out}, {k}]");
    if let Some(b) = bias {
        assert_eq!(b.len(), out, "bias is not [{out}]");
    }
    let mut y = Mat::zeros(n, out);
    if n > 0 && out > 0 && k > 0 {
        unsafe {
            gemm::gemm(
                n,
                out,
                k,
                y.data.as_mut_ptr(),
                1,
                out as isize,
                false,
                x.data.as_ptr(),
                1,
                k as isize,
                // wᵀ is [k, out]: element (i, o) lives at w[o * k + i], so the
                // step between columns is k and between rows is 1.
                w.as_ptr(),
                k as isize,
                1,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                parallelism(),
            );
        }
    }
    if let Some(b) = bias {
        y.data.par_chunks_mut(out).for_each(|row| {
            for (v, bv) in row.iter_mut().zip(b.iter()) {
                *v += bv;
            }
        });
    }
    y
}

/// Plain `a @ b`, both row-major. Used where the right operand is an
/// activation rather than a stored `[out, in]` weight — the mask decoder
/// multiplies its hypernetwork outputs into the upscaled feature volume.
pub fn matmul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(
        a.cols, b.rows,
        "cannot multiply {}x{} by {}x{}",
        a.rows, a.cols, b.rows, b.cols
    );
    let (m, k, n) = (a.rows, a.cols, b.cols);
    let mut out = Mat::zeros(m, n);
    if m > 0 && n > 0 && k > 0 {
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                out.data.as_mut_ptr(),
                1,
                n as isize,
                false,
                a.data.as_ptr(),
                1,
                k as isize,
                b.data.as_ptr(),
                1,
                n as isize,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                parallelism(),
            );
        }
    }
    out
}

/// LayerNorm over the trailing `group` elements of each row of `data`.
///
/// One function covers both shapes in the network: `group = dim` normalizes
/// each token independently (the usual case), and `group = C*D*H*W` with a
/// single group normalizes a whole volume jointly — which is what the mask
/// decoder's `output_upscaling.1` does, with an affine pair as large as the
/// activation itself.
pub fn layer_norm(data: &mut [f32], group: usize, weight: &[f32], bias: &[f32], eps: f32) {
    assert!(group > 0 && data.len().is_multiple_of(group));
    assert_eq!(weight.len(), group);
    assert_eq!(bias.len(), group);
    let inv_n = 1.0 / group as f64;
    data.par_chunks_mut(group).for_each(|row| {
        let mut sum = 0f64;
        let mut sq = 0f64;
        for v in row.iter() {
            let v = *v as f64;
            sum += v;
            sq += v * v;
        }
        let mean = sum * inv_n;
        // PyTorch normalizes by the biased variance (divide by N, not N-1).
        let var = (sq * inv_n - mean * mean).max(0.0);
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        for ((v, w), b) in row.iter_mut().zip(weight.iter()).zip(bias.iter()) {
            *v = (((*v as f64 - mean) * inv_std) as f32) * w + b;
        }
    });
}

/// In-place row-wise softmax over `cols`-wide rows, max-subtracted.
pub fn softmax_rows(data: &mut [f32], cols: usize) {
    assert!(cols > 0 && data.len().is_multiple_of(cols));
    data.par_chunks_mut(cols).for_each(|row| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            // an entirely masked row: leave it as a uniform distribution
            let u = 1.0 / cols as f32;
            row.fill(u);
            return;
        }
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    });
}

/// The error function, Abramowitz & Stegun 7.1.26 evaluated in `f64`
/// (|error| < 1.5e-7 — well inside `f32` precision).
fn erf(x: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A: [f64; 5] = [
        0.254_829_592,
        -0.284_496_736,
        1.421_413_741,
        -1.453_152_027,
        1.061_405_429,
    ];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Exact (erf-based) GELU — PyTorch's `nn.GELU()` default, used by the image
/// encoder's MLP blocks and the decoder's upscaling.
pub fn gelu(data: &mut [f32]) {
    data.par_iter_mut().for_each(|v| {
        let x = *v as f64;
        *v = (0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))) as f32;
    });
}

/// ReLU — the two-way transformer's MLP, the hypernetwork MLPs and the IoU
/// head all use this, *not* GELU. Mixing the two up is silent and costly.
pub fn relu(data: &mut [f32]) {
    data.par_iter_mut().for_each(|v| {
        if *v < 0.0 {
            *v = 0.0;
        }
    });
}

/// QuickGELU, `x * sigmoid(1.702 x)` — CLIP's activation, and only CLIP's.
pub fn quick_gelu(data: &mut [f32]) {
    data.par_iter_mut().for_each(|v| {
        let x = *v;
        *v = x / (1.0 + (-1.702 * x).exp());
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "index {i}: {x} vs {y}");
        }
    }

    #[test]
    fn linear_matches_hand_computation() {
        // x is 2x3, w is [out=2, in=3]; y = x @ wᵀ + b
        let x = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let w = [1.0, 0.0, -1.0, 2.0, 2.0, 2.0];
        let b = [0.5, -1.0];
        let y = linear(&x, &w, 2, Some(&b));
        assert_eq!((y.rows, y.cols), (2, 2));
        // row 0: [1*1 + 2*0 + 3*-1, 1*2+2*2+3*2] = [-2, 12]  (+ bias)
        // row 1: [4 + 0 - 6, 8+10+12] = [-2, 30]              (+ bias)
        approx(&y.data, &[-1.5, 11.0, -1.5, 29.0], 1e-6);
        // and without a bias
        let y = linear(&x, &w, 2, None);
        approx(&y.data, &[-2.0, 12.0, -2.0, 30.0], 1e-6);
    }

    #[test]
    fn linear_agrees_with_a_naive_loop_on_larger_shapes() {
        let mut s = 12345u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        };
        let (n, k, out) = (37, 53, 29);
        let x = Mat::from_vec(n, k, (0..n * k).map(|_| rnd()).collect());
        let w: Vec<f32> = (0..out * k).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..out).map(|_| rnd()).collect();
        let y = linear(&x, &w, out, Some(&b));
        for r in 0..n {
            for o in 0..out {
                let mut acc = b[o];
                for i in 0..k {
                    acc += x.data[r * k + i] * w[o * k + i];
                }
                assert!((y.data[r * out + o] - acc).abs() < 1e-4, "at {r},{o}");
            }
        }
    }

    #[test]
    fn matmul_matches_hand_computation() {
        let a = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Mat::from_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let c = matmul(&a, &b);
        assert_eq!((c.rows, c.cols), (2, 2));
        // [1+3, 2+3] and [4+6, 5+6]
        approx(&c.data, &[4.0, 5.0, 10.0, 11.0], 1e-6);
    }

    #[test]
    fn layer_norm_standardizes_each_group() {
        let mut d = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let w = vec![1.0f32; 4];
        let b = vec![0.0f32; 4];
        layer_norm(&mut d, 4, &w, &b, 0.0);
        for row in d.chunks(4) {
            let mean: f32 = row.iter().sum::<f32>() / 4.0;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5, "mean {mean}");
            assert!((var - 1.0).abs() < 1e-4, "var {var}");
        }
        // both rows normalize to the same shape: the scale is divided out
        approx(&d[0..4], &d[4..8], 1e-5);
    }

    #[test]
    fn layer_norm_applies_the_affine_and_uses_biased_variance() {
        // x = [1,2,3]: mean 2, biased var 2/3, std 0.8164966
        let mut d = vec![1.0f32, 2.0, 3.0];
        layer_norm(&mut d, 3, &[2.0, 2.0, 2.0], &[1.0, 1.0, 1.0], 0.0);
        let s = (2.0f32 / 3.0).sqrt();
        approx(&d, &[1.0 - 2.0 / s, 1.0, 1.0 + 2.0 / s], 1e-5);
    }

    #[test]
    fn layer_norm_over_a_whole_volume_is_one_group() {
        // the decoder's (C,D,H,W) norm: one group spanning everything
        let n = 2 * 2 * 2 * 2;
        let mut d: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let w = vec![1.0f32; n];
        let b = vec![0.0f32; n];
        layer_norm(&mut d, n, &w, &b, 0.0);
        let mean: f32 = d.iter().sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-5);
        // a per-channel norm would have produced a different result entirely
        assert!(d[0] < -1.5 && d[n - 1] > 1.5);
    }

    #[test]
    fn softmax_rows_are_distributions() {
        let mut d = vec![1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0];
        softmax_rows(&mut d, 3);
        for row in d.chunks(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        // exp(1),exp(2),exp(3) normalized
        approx(&d[0..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94], 1e-6);
        // a uniform row is uniform
        approx(&d[3..6], &[1.0 / 3.0; 3], 1e-6);
    }

    #[test]
    fn softmax_is_stable_for_large_and_masked_rows() {
        let mut d = vec![1000.0f32, 1000.0, 1001.0];
        softmax_rows(&mut d, 3);
        assert!(d.iter().all(|v| v.is_finite()));
        assert!((d.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // an entirely -inf row (fully masked) must not produce NaNs
        let mut d = vec![f32::NEG_INFINITY; 4];
        softmax_rows(&mut d, 4);
        assert!(d.iter().all(|v| (*v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn activations_match_reference_values() {
        // GELU values from PyTorch's exact erf implementation
        let mut d = vec![-3.0f32, -1.0, 0.0, 0.5, 1.0, 3.0];
        gelu(&mut d);
        approx(
            &d,
            &[
                -0.004_049_69,
                -0.158_655_26,
                0.0,
                0.345_731_23,
                0.841_344_7,
                2.995_950_2,
            ],
            1e-5,
        );
        let mut d = vec![-2.0f32, 0.0, 3.0];
        relu(&mut d);
        approx(&d, &[0.0, 0.0, 3.0], 0.0);
        // QuickGELU: x * sigmoid(1.702 x)
        let mut d = vec![-1.0f32, 0.0, 1.0];
        quick_gelu(&mut d);
        let s = |x: f32| x / (1.0 + (-1.702 * x).exp());
        approx(&d, &[s(-1.0), 0.0, s(1.0)], 1e-6);
        // The three are genuinely distinct, which is what makes using the
        // wrong one a real bug. ReLU is trivially far from GELU at negative
        // x; QuickGELU is a deliberate approximation of GELU, so it only
        // separates by ~0.02 and only away from the origin — hence a scan
        // rather than a single probe.
        let mut a = vec![-1.0f32];
        let mut b = vec![-1.0f32];
        gelu(&mut a);
        relu(&mut b);
        assert!((a[0] - b[0]).abs() > 0.1, "relu vs gelu at -1");
        let xs: Vec<f32> = (-40..=40).map(|i| i as f32 * 0.1).collect();
        let (mut g, mut q) = (xs.clone(), xs.clone());
        gelu(&mut g);
        quick_gelu(&mut q);
        let worst = g
            .iter()
            .zip(q.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            (0.01..0.05).contains(&worst),
            "gelu vs quick_gelu peak {worst}"
        );
    }

    #[test]
    fn erf_is_accurate() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.5204998778),
            (1.0, 0.8427007929),
            (2.0, 0.9953222650),
            (-1.5, -0.9661051465),
            (3.0, 0.9999779095),
        ] {
            assert!((erf(x) - want).abs() < 2e-7, "erf({x}) = {}", erf(x));
        }
    }
}
