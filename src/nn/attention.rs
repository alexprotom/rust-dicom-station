//! Multi-head attention.
//!
//! One function serves every attention in the network — the image encoder's
//! 2048-token self-attention, the mask decoder's self- and cross-attentions
//! at a reduced internal width, and CLIP's causally masked text attention.
//! They differ only in what is projected before the call and whether a mask
//! is applied.
//!
//! Heads are processed one at a time rather than all at once. The image
//! encoder's score matrix is 2048x2048 per head, 16 MB in `f32`; materializing
//! all twelve would cost 200 MB for no gain, whereas one head at a time keeps
//! the working set small and still hands each matrix multiply to `gemm` with
//! full parallelism.

use super::linalg::softmax_rows;
use super::tensor::Mat;

/// Which positions a query may attend to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    /// Every query sees every key.
    None,
    /// Query `i` sees keys `j <= i` — CLIP's text encoder.
    Causal,
}

/// Scaled dot-product attention over pre-projected `q`, `k`, `v`.
///
/// `q` is `[n_q, internal]`, `k` and `v` are `[n_kv, internal]`, and the
/// result is `[n_q, internal]`. `internal` must divide evenly into `heads`;
/// the scale is `1/sqrt(internal / heads)`, taken from the head width rather
/// than the embedding width — which is why the decoder's downsampled
/// attentions scale by `1/sqrt(48)` and not `1/sqrt(96)`.
pub fn attention(q: &Mat, k: &Mat, v: &Mat, heads: usize, mask: Mask) -> Mat {
    assert_eq!(q.cols, k.cols, "q and k must share the internal width");
    assert_eq!(k.cols, v.cols, "k and v must share the internal width");
    assert_eq!(
        k.rows, v.rows,
        "k and v must have the same number of tokens"
    );
    let internal = q.cols;
    assert!(
        internal.is_multiple_of(heads),
        "{heads} heads do not divide an internal width of {internal}"
    );
    let hd = internal / heads;
    let (n_q, n_kv) = (q.rows, k.rows);
    let scale = 1.0 / (hd as f32).sqrt();

    let mut out = Mat::zeros(n_q, internal);
    // Scratch buffers, reused across heads.
    let mut qh = vec![0f32; n_q * hd];
    let mut kh = vec![0f32; n_kv * hd];
    let mut vh = vec![0f32; n_kv * hd];
    let mut scores = vec![0f32; n_q * n_kv];
    let mut ctx = vec![0f32; n_q * hd];

    for h in 0..heads {
        let off = h * hd;
        gather_head(&q.data, &mut qh, n_q, internal, off, hd);
        gather_head(&k.data, &mut kh, n_kv, internal, off, hd);
        gather_head(&v.data, &mut vh, n_kv, internal, off, hd);

        // scores = qh @ khᵀ  -> [n_q, n_kv]
        matmul((&qh, n_q, hd), (&kh, n_kv, hd), true, &mut scores);
        for s in scores.iter_mut() {
            *s *= scale;
        }
        if mask == Mask::Causal {
            for (i, row) in scores.chunks_mut(n_kv).enumerate() {
                for (j, s) in row.iter_mut().enumerate() {
                    if j > i {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
        }
        softmax_rows(&mut scores, n_kv);

        // ctx = scores @ vh -> [n_q, hd]
        matmul((&scores, n_q, n_kv), (&vh, n_kv, hd), false, &mut ctx);
        for t in 0..n_q {
            out.data[t * internal + off..t * internal + off + hd]
                .copy_from_slice(&ctx[t * hd..(t + 1) * hd]);
        }
    }
    out
}

/// Copy one head's columns out of a `[n, stride]` matrix into a contiguous
/// `[n, hd]` buffer.
fn gather_head(src: &[f32], dst: &mut [f32], n: usize, stride: usize, off: usize, hd: usize) {
    for t in 0..n {
        dst[t * hd..(t + 1) * hd].copy_from_slice(&src[t * stride + off..t * stride + off + hd]);
    }
}

/// `dst = a @ b` (or `a @ bᵀ` when `transpose_b`), all row-major.
fn matmul(
    a: (&[f32], usize, usize),
    b: (&[f32], usize, usize),
    transpose_b: bool,
    dst: &mut [f32],
) {
    let (a, a_rows, a_cols) = a;
    let (b, b_rows, b_cols) = b;
    let (n, k) = if transpose_b {
        assert_eq!(a_cols, b_cols);
        (b_rows, a_cols)
    } else {
        assert_eq!(a_cols, b_rows);
        (b_cols, a_cols)
    };
    let m = a_rows;
    debug_assert_eq!(dst.len(), m * n);
    // b is [b_rows, b_cols] row-major. As the right operand we need [k, n]:
    // without a transpose that is b itself (cs = 1, rs = b_cols); with one,
    // element (i, j) is b[j * b_cols + i], so the strides swap.
    let (rhs_cs, rhs_rs) = if transpose_b {
        (b_cols as isize, 1isize)
    } else {
        (1isize, b_cols as isize)
    };
    let par = match rayon::current_num_threads() {
        0 | 1 => gemm::Parallelism::None,
        t => gemm::Parallelism::Rayon(t),
    };
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            1,
            a_cols as isize,
            b.as_ptr(),
            rhs_cs,
            rhs_rs,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            par,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    }

    fn rand_mat(rows: usize, cols: usize, seed: &mut u64) -> Mat {
        Mat::from_vec(rows, cols, (0..rows * cols).map(|_| rnd(seed)).collect())
    }

    /// Straightforward definition, one head, for cross-checking.
    fn naive_single_head(q: &Mat, k: &Mat, v: &Mat, mask: Mask) -> Mat {
        let d = q.cols;
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = Mat::zeros(q.rows, d);
        for i in 0..q.rows {
            let mut s: Vec<f32> = (0..k.rows)
                .map(|j| {
                    if mask == Mask::Causal && j > i {
                        f32::NEG_INFINITY
                    } else {
                        (0..d)
                            .map(|c| q.data[i * d + c] * k.data[j * d + c])
                            .sum::<f32>()
                            * scale
                    }
                })
                .collect();
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut tot = 0.0;
            for x in s.iter_mut() {
                *x = (*x - m).exp();
                tot += *x;
            }
            for x in s.iter_mut() {
                *x /= tot;
            }
            for c in 0..d {
                out.data[i * d + c] = (0..k.rows).map(|j| s[j] * v.data[j * d + c]).sum();
            }
        }
        out
    }

    #[test]
    fn single_head_matches_the_definition() {
        let mut s = 7u64;
        let q = rand_mat(5, 8, &mut s);
        let k = rand_mat(6, 8, &mut s);
        let v = rand_mat(6, 8, &mut s);
        let got = attention(&q, &k, &v, 1, Mask::None);
        let want = naive_single_head(&q, &k, &v, Mask::None);
        for (a, b) in got.data.iter().zip(want.data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn multi_head_splits_columns_and_concatenates_back() {
        // Two heads over an 8-wide internal dimension must equal two
        // independent 4-wide single-head attentions on the column halves.
        let mut s = 11u64;
        let (nq, nkv, heads, hd) = (4, 7, 2, 4);
        let q = rand_mat(nq, heads * hd, &mut s);
        let k = rand_mat(nkv, heads * hd, &mut s);
        let v = rand_mat(nkv, heads * hd, &mut s);
        let got = attention(&q, &k, &v, heads, Mask::None);
        for h in 0..heads {
            let slice = |m: &Mat| {
                Mat::from_vec(
                    m.rows,
                    hd,
                    (0..m.rows)
                        .flat_map(|r| {
                            m.data[r * heads * hd + h * hd..r * heads * hd + (h + 1) * hd].to_vec()
                        })
                        .collect(),
                )
            };
            let want = naive_single_head(&slice(&q), &slice(&k), &slice(&v), Mask::None);
            for r in 0..nq {
                for c in 0..hd {
                    let g = got.data[r * heads * hd + h * hd + c];
                    let w = want.data[r * hd + c];
                    assert!((g - w).abs() < 1e-5, "head {h} at {r},{c}: {g} vs {w}");
                }
            }
        }
    }

    #[test]
    fn causal_mask_hides_the_future() {
        let mut s = 3u64;
        let x = rand_mat(6, 8, &mut s);
        let got = attention(&x, &x, &x, 2, Mask::Causal);
        let want = {
            // per head, the naive causal version
            let hd = 4;
            let mut out = Mat::zeros(6, 8);
            for h in 0..2 {
                let slice = Mat::from_vec(
                    6,
                    hd,
                    (0..6)
                        .flat_map(|r| x.data[r * 8 + h * hd..r * 8 + (h + 1) * hd].to_vec())
                        .collect(),
                );
                let o = naive_single_head(&slice, &slice, &slice, Mask::Causal);
                for r in 0..6 {
                    out.data[r * 8 + h * hd..r * 8 + (h + 1) * hd]
                        .copy_from_slice(&o.data[r * hd..(r + 1) * hd]);
                }
            }
            out
        };
        for (a, b) in got.data.iter().zip(want.data.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // token 0 can only see itself, so its output is exactly v[0]
        for c in 0..8 {
            assert!((got.data[c] - x.data[c]).abs() < 1e-5);
        }
    }

    #[test]
    fn attending_to_one_key_returns_that_value() {
        // With a single key/value pair the softmax is degenerate and the
        // output must be the value itself, whatever the query is.
        let mut s = 99u64;
        let q = rand_mat(3, 4, &mut s);
        let k = rand_mat(1, 4, &mut s);
        let v = rand_mat(1, 4, &mut s);
        let got = attention(&q, &k, &v, 1, Mask::None);
        for r in 0..3 {
            for c in 0..4 {
                assert!((got.data[r * 4 + c] - v.data[c]).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "do not divide")]
    fn head_count_must_divide_the_internal_width() {
        let m = Mat::zeros(2, 9);
        attention(&m, &m, &m, 2, Mask::None);
    }
}
