//! Separable image resampling with a normalized filter kernel.
//!
//! `burn`'s `interpolate` covers everything the *network* does, and matches
//! PyTorch's plain bilinear, nearest and bicubic to f32 noise. Two places in
//! the MedSAM2 pipeline need something else:
//!
//! * **preprocessing**, where the reference resizes each slice to 512 x 512
//!   with `PIL.Image.resize`, whose default is a bicubic kernel with
//!   `a = -0.5` — not PyTorch's `a = -0.75` — and which scales the filter
//!   support when downscaling, so a shrink is area-averaged rather than
//!   point-sampled;
//! * **mask prompts**, where `_use_mask_as_output` downsamples 512 -> 128 with
//!   `antialias=True`, which is the same construction with a triangular
//!   kernel.
//!
//! Both are the classic PIL resampling loop: for each output pixel take the
//! kernel centred on `(i + 0.5) * scale`, widened by `scale` when shrinking,
//! truncated at the image edge and **renormalized**. (That renormalization is
//! what distinguishes it from PyTorch's non-antialiased path, which clamps to
//! the edge pixel instead.) Both axes are done separately, rows first.
//!
//! This runs on the host on one slice at a time, which is where the data
//! already is: a mask prompt comes from the user's own segmentation, and a
//! slice is windowed and quantized on the CPU before it is ever uploaded.

/// Which kernel to resample with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Filter {
    /// Support 1: `F.interpolate(mode="bilinear")`'s kernel.
    Triangle,
    /// Support 2, with the given `a`. PIL uses -0.5, PyTorch -0.75.
    Cubic(f64),
}

impl Filter {
    /// PIL's `Image.resize` default, and so MedSAM2's preprocessing.
    pub const PIL_BICUBIC: Filter = Filter::Cubic(-0.5);
    /// PyTorch's `mode="bicubic"`.
    pub const TORCH_BICUBIC: Filter = Filter::Cubic(-0.75);

    fn support(self) -> f64 {
        match self {
            Filter::Triangle => 1.0,
            Filter::Cubic(_) => 2.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Filter::Triangle => {
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            }
            Filter::Cubic(a) => {
                if x < 1.0 {
                    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
        }
    }
}

/// The weights for one axis: for each output index, where its window starts
/// and the normalized kernel over it.
struct Weights {
    starts: Vec<usize>,
    /// `taps` values per output index, row-major.
    values: Vec<f64>,
    taps: usize,
}

fn weights(src: usize, dst: usize, filter: Filter, antialias: bool) -> Weights {
    let scale = src as f64 / dst as f64;
    let filter_scale = if antialias && scale > 1.0 { scale } else { 1.0 };
    let support = filter.support() * filter_scale;
    // The reference implementation sizes every window the same and pads with
    // zeros, which keeps the inner loop branch-free.
    let taps = ((support * 2.0).ceil() as usize + 1).max(2);
    let mut starts = Vec::with_capacity(dst);
    let mut values = vec![0.0; dst * taps];
    for i in 0..dst {
        let center = (i as f64 + 0.5) * scale;
        let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
        let hi = ((center + support + 0.5).floor() as usize).min(src).max(lo + 1);
        let mut sum = 0.0;
        for k in lo..hi.min(lo + taps) {
            let w = filter.eval((k as f64 + 0.5 - center) / filter_scale);
            values[i * taps + (k - lo)] = w;
            sum += w;
        }
        if sum != 0.0 {
            for k in 0..taps {
                values[i * taps + k] /= sum;
            }
        }
        starts.push(lo);
    }
    Weights {
        starts,
        values,
        taps,
    }
}

/// Resample a single-channel row-major image.
///
/// `antialias` widens the kernel when shrinking; it has no effect when
/// enlarging, where the kernel is always the plain one.
pub fn resize(
    src: &[f32],
    src_hw: [usize; 2],
    dst_hw: [usize; 2],
    filter: Filter,
    antialias: bool,
) -> Vec<f32> {
    let [sh, sw] = src_hw;
    let [dh, dw] = dst_hw;
    assert_eq!(src.len(), sh * sw, "source is not {sh}x{sw}");
    if [sh, sw] == [dh, dw] {
        return src.to_vec();
    }

    // horizontal pass: sh x sw -> sh x dw
    let wx = weights(sw, dw, filter, antialias);
    let mut mid = vec![0f32; sh * dw];
    for y in 0..sh {
        for x in 0..dw {
            let start = wx.starts[x];
            let mut acc = 0.0;
            for k in 0..wx.taps {
                let sx = start + k;
                if sx >= sw {
                    break;
                }
                acc += f64::from(src[y * sw + sx]) * wx.values[x * wx.taps + k];
            }
            mid[y * dw + x] = acc as f32;
        }
    }

    // vertical pass: sh x dw -> dh x dw
    let wy = weights(sh, dh, filter, antialias);
    let mut out = vec![0f32; dh * dw];
    for y in 0..dh {
        let start = wy.starts[y];
        for x in 0..dw {
            let mut acc = 0.0;
            for k in 0..wy.taps {
                let sy = start + k;
                if sy >= sh {
                    break;
                }
                acc += f64::from(mid[sy * dw + x]) * wy.values[y * wy.taps + k];
            }
            out[y * dw + x] = acc as f32;
        }
    }
    out
}

/// PIL's fixed-point precision for 8-bit images.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Resample an 8-bit image the way PIL does, bit for bit.
///
/// The float path above is the algorithm; this is the arithmetic. PIL
/// quantizes the kernel to 22-bit fixed point, accumulates in integers and
/// clips to a byte **after each pass**, so a float implementation drifts from
/// it by a few least-significant bits — visible enough to matter when the
/// result is the network's input.
pub fn resize_u8(
    src: &[u8],
    src_hw: [usize; 2],
    dst_hw: [usize; 2],
    filter: Filter,
    antialias: bool,
) -> Vec<u8> {
    let [sh, sw] = src_hw;
    let [dh, dw] = dst_hw;
    assert_eq!(src.len(), sh * sw, "source is not {sh}x{sw}");
    if [sh, sw] == [dh, dw] {
        return src.to_vec();
    }
    let quantize = |w: &Weights| -> Vec<i64> {
        w.values
            .iter()
            .map(|v| {
                let scaled = v * f64::from(1u32 << PRECISION_BITS);
                if *v < 0.0 {
                    (scaled - 0.5) as i64
                } else {
                    (scaled + 0.5) as i64
                }
            })
            .collect()
    };
    let round = 1i64 << (PRECISION_BITS - 1);
    let clip8 = |acc: i64| -> u8 { (acc >> PRECISION_BITS).clamp(0, 255) as u8 };

    let wx = weights(sw, dw, filter, antialias);
    let kx = quantize(&wx);
    let mut mid = vec![0u8; sh * dw];
    for y in 0..sh {
        for x in 0..dw {
            let start = wx.starts[x];
            let mut acc = round;
            for k in 0..wx.taps {
                let sx = start + k;
                if sx >= sw {
                    break;
                }
                acc += i64::from(src[y * sw + sx]) * kx[x * wx.taps + k];
            }
            mid[y * dw + x] = clip8(acc);
        }
    }

    let wy = weights(sh, dh, filter, antialias);
    let ky = quantize(&wy);
    let mut out = vec![0u8; dh * dw];
    for y in 0..dh {
        let start = wy.starts[y];
        for x in 0..dw {
            let mut acc = round;
            for k in 0..wy.taps {
                let sy = start + k;
                if sy >= sh {
                    break;
                }
                acc += i64::from(mid[sy * dw + x]) * ky[y * wy.taps + k];
            }
            out[y * dw + x] = clip8(acc);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::cache::{load_safetensors, WTensor};
    use std::collections::HashMap;
    use std::path::Path;

    fn fixtures() -> HashMap<String, WTensor> {
        load_safetensors(Path::new("tests/data/medsam2-ops.safetensors")).unwrap()
    }

    fn check(f: &HashMap<String, WTensor>, name: &str, filter: Filter, antialias: bool) {
        let x = f.get(&format!("{name}.x")).expect("input");
        let y = f.get(&format!("{name}.y")).expect("output");
        let (sh, sw) = (x.shape[x.shape.len() - 2], x.shape[x.shape.len() - 1]);
        let (dh, dw) = (y.shape[y.shape.len() - 2], y.shape[y.shape.len() - 1]);
        let got = resize(&x.data, [sh, sw], [dh, dw], filter, antialias);
        assert_eq!(got.len(), y.data.len(), "{name}: size");
        let worst = got
            .iter()
            .zip(y.data.iter())
            .map(|(a, b)| (a - b).abs() / (1.0 + b.abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "{name}: relative error {worst:e}");
    }

    #[test]
    fn enlarging_matches_pil_bicubic() {
        check(&fixtures(), "pil_up", Filter::PIL_BICUBIC, true);
    }

    #[test]
    fn shrinking_matches_pil_bicubic_which_always_antialiases() {
        check(&fixtures(), "pil_down", Filter::PIL_BICUBIC, true);
    }

    #[test]
    fn shrinking_matches_pytorchs_antialiased_bilinear() {
        check(&fixtures(), "torch_bilinear_aa", Filter::Triangle, true);
    }

    #[test]
    fn the_eight_bit_path_matches_pil_exactly() {
        let f = fixtures();
        let src = f.get("preprocess.u8").expect("fixture");
        let want = f.get("preprocess.pil_u8").expect("fixture");
        let (sh, sw) = (src.shape[0], src.shape[1]);
        let (dh, dw) = (want.shape[0], want.shape[1]);
        let bytes: Vec<u8> = src.data.iter().map(|v| *v as u8).collect();
        let got = resize_u8(&bytes, [sh, sw], [dh, dw], Filter::PIL_BICUBIC, true);
        let wanted: Vec<u8> = want.data.iter().map(|v| *v as u8).collect();
        assert_eq!(got, wanted, "fixed-point resampling must be exact");
    }

    #[test]
    fn a_resize_to_the_same_size_is_the_identity() {
        let x: Vec<f32> = (0..12).map(|i| i as f32).collect();
        assert_eq!(resize(&x, [3, 4], [3, 4], Filter::PIL_BICUBIC, true), x);
    }

    #[test]
    fn the_kernels_are_the_published_ones() {
        // both are 1 at the centre and 0 at their support
        for f in [Filter::Triangle, Filter::PIL_BICUBIC, Filter::TORCH_BICUBIC] {
            assert!((f.eval(0.0) - 1.0).abs() < 1e-12, "{f:?}");
            assert_eq!(f.eval(f.support()), 0.0, "{f:?}");
            assert_eq!(f.eval(f.support() + 0.5), 0.0, "{f:?}");
        }
        // and the cubics differ from each other where it matters
        assert!((Filter::PIL_BICUBIC.eval(1.5) - Filter::TORCH_BICUBIC.eval(1.5)).abs() > 1e-3);
    }
}
