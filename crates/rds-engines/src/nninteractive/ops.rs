//! The array operations the nnInteractive session is built from, each one
//! matching the PyTorch, scipy or skimage call it stands for in
//! `nnInteractive.inference.inference_session` (nninteractive 2.6.0):
//! float16 storage, the point's structuring element, trilinear, area and
//! nearest resampling, separable max / min pooling with replicated borders,
//! and the greedy box cover that plans the refinement passes.
//!
//! Volumes are C-order `[d0][d1][d2]` (the last axis fastest), as the
//! session's arrays are.

use crate::nn::half::{f16_to_f32, f32_to_f16};

/// Store a value in a float16 tensor (round to nearest, ties to even).
#[inline]
pub fn h(v: f32) -> f32 {
    f16_to_f32(f32_to_f16(v))
}

/// The largest finite float16.
pub const F16_MAX: f64 = 65504.0;

/// Python's `round` on a float: half to even.
#[inline]
pub fn round_py(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `nnInteractive.utils.rounding.round_to_nearest_odd`.
pub fn round_to_nearest_odd(x: f64) -> usize {
    debug_assert!(x > 0.0);
    let cl = x.ceil() as i64;
    let fl = x.floor() as i64;
    let v = if cl % 2 == 1 {
        cl
    } else if fl % 2 == 1 {
        fl
    } else {
        round_py(x) + 1
    };
    v.max(1) as usize
}

/// A half-open box `[lo, hi)` per axis, in voxels; may reach outside the
/// image.
pub type BBox = [[i64; 2]; 3];

/// A half-open box as its `lo` and `hi` corners.
type Span = ([usize; 3], [usize; 3]);

pub fn bbox_size(b: &BBox) -> [usize; 3] {
    std::array::from_fn(|a| (b[a][1] - b[a][0]).max(0) as usize)
}

/// The part of `b` inside `[0, dims)`, or `None` when nothing is.
pub fn clip(b: &BBox, dims: [usize; 3]) -> Option<BBox> {
    let c: BBox = std::array::from_fn(|a| [b[a][0].max(0), b[a][1].min(dims[a] as i64)]);
    c.iter().all(|r| r[1] > r[0]).then_some(c)
}

/// `b` in the coordinates of a frame whose origin is `frame`'s lower corner.
pub fn to_local(b: &BBox, frame: &BBox) -> BBox {
    std::array::from_fn(|a| [b[a][0] - frame[a][0], b[a][1] - frame[a][0]])
}

pub fn union(boxes: &[BBox]) -> Option<BBox> {
    let first = boxes.first()?;
    Some(std::array::from_fn(|a| {
        [
            boxes.iter().map(|b| b[a][0]).min().unwrap_or(first[a][0]),
            boxes.iter().map(|b| b[a][1]).max().unwrap_or(first[a][1]),
        ]
    }))
}

#[inline]
pub fn idx(dims: [usize; 3], i: usize, j: usize, k: usize) -> usize {
    (i * dims[1] + j) * dims[2] + k
}

/// Copy `src` (dims `sd`) into `dst` (dims `dd`) at `bbox` (in `dst`
/// coordinates, size = `sd`), writing only the part inside `dst` -
/// `nnInteractive.utils.crop.paste_tensor`.
pub fn paste<T: Copy>(dst: &mut [T], dd: [usize; 3], src: &[T], sd: [usize; 3], bbox: &BBox) {
    let mut t = [[0usize; 2]; 3];
    let mut s0 = [0usize; 3];
    for a in 0..3 {
        let lo = bbox[a][0].max(0);
        let hi = bbox[a][1].min(dd[a] as i64);
        if lo >= hi {
            return;
        }
        t[a] = [lo as usize, hi as usize];
        s0[a] = (lo - bbox[a][0]) as usize;
    }
    for i in t[0][0]..t[0][1] {
        for j in t[1][0]..t[1][1] {
            let si = s0[0] + i - t[0][0];
            let sj = s0[1] + j - t[1][0];
            let sk = s0[2];
            let n = t[2][1] - t[2][0];
            let d = idx(dd, i, j, t[2][0]);
            let s = idx(sd, si, sj, sk);
            dst[d..d + n].copy_from_slice(&src[s..s + n]);
        }
    }
}

/// The `[lo, hi)` box of the nonzero voxels, or `None` for an empty mask.
pub fn nonzero_bbox<T: Copy + PartialEq + Default>(data: &[T], dims: [usize; 3]) -> Option<BBox> {
    let zero = T::default();
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            let row = &data[idx(dims, i, j, 0)..idx(dims, i, j, 0) + dims[2]];
            let (Some(a), Some(b)) = (
                row.iter().position(|v| *v != zero),
                row.iter().rposition(|v| *v != zero),
            ) else {
                continue;
            };
            lo[0] = lo[0].min(i);
            hi[0] = hi[0].max(i + 1);
            lo[1] = lo[1].min(j);
            hi[1] = hi[1].max(j + 1);
            lo[2] = lo[2].min(a);
            hi[2] = hi[2].max(b + 1);
        }
    }
    (lo[0] != usize::MAX).then(|| std::array::from_fn(|a| [lo[a] as i64, hi[a] as i64]))
}

// ------------------------------------------------------------------ point --

/// The structuring element a point is drawn with:
/// `nnInteractive.interaction.point.build_point(radii, True, False)` -
/// skimage's `ball(r)`, its Euclidean distance transform, divided by its
/// maximum. Returns the edge length and the values, C-order.
pub fn point_strel(radius: usize) -> (usize, Vec<f64>) {
    let n = 2 * radius + 1;
    let r = radius as i64;
    let mut ball = vec![false; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let (x, y, z) = (i as i64 - r, j as i64 - r, k as i64 - r);
                ball[(i * n + j) * n + k] = x * x + y * y + z * z <= r * r;
            }
        }
    }
    // scipy's distance_transform_edt: the distance to the nearest zero
    // voxel inside the array (the array edge is not background).
    let zeros: Vec<(i64, i64, i64)> = (0..n * n * n)
        .filter(|&v| !ball[v])
        .map(|v| ((v / (n * n)) as i64, ((v / n) % n) as i64, (v % n) as i64))
        .collect();
    let mut d = vec![0f64; n * n * n];
    for v in 0..n * n * n {
        if !ball[v] {
            continue;
        }
        let (i, j, k) = ((v / (n * n)) as i64, ((v / n) % n) as i64, (v % n) as i64);
        let best = zeros
            .iter()
            .map(|&(a, b, c)| (a - i) * (a - i) + (b - j) * (b - j) + (c - k) * (c - k))
            .min()
            .unwrap_or(0);
        d[v] = (best as f64).sqrt();
    }
    let max = d.iter().cloned().fold(0.0, f64::max);
    if max > 0.0 {
        d.iter_mut().for_each(|v| *v /= max);
    }
    (n, d)
}

// ------------------------------------------------------------- resampling --

/// One axis of PyTorch's linear interpolation with `align_corners=False`
/// (`compute_indices_weights_linear`): for every output index, the two
/// source indices and their weights.
///
/// PyTorch's x86 build compiles these kernels with fused multiply-adds
/// (`scale * (i + 0.5) - 0.5`, and `a * wa + b * wb` as `fma(a, wa, b * wb)`
/// below); `mul_add` reproduces its results bit for bit.
fn linear_axis(input: usize, output: usize) -> Vec<(usize, usize, f32, f32)> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|o| {
            if input == output {
                return (o, o, 1.0, 0.0);
            }
            let src = scale.mul_add(o as f32 + 0.5, -0.5).max(0.0);
            let i0 = (src.floor() as usize).min(input - 1);
            let l1 = (src - i0 as f32).clamp(0.0, 1.0);
            let i1 = if i0 < input - 1 { i0 + 1 } else { i0 };
            (i0, i1, 1.0 - l1, l1)
        })
        .collect()
}

/// `F.interpolate(x, size, mode="trilinear")` of a one-channel volume read
/// through `at` (dims `input`), evaluated on the output box `region`
/// (`[lo, hi)` per axis of the output grid `output`). The weights are
/// applied the way PyTorch's CPU kernel nests them, outermost axis last.
pub fn trilinear_region(
    at: &(dyn Fn(usize, usize, usize) -> f32 + Sync),
    input: [usize; 3],
    output: [usize; 3],
    region: [[usize; 2]; 3],
) -> Vec<f32> {
    use rayon::prelude::*;
    let ax: [Vec<(usize, usize, f32, f32)>; 3] =
        std::array::from_fn(|a| linear_axis(input[a], output[a]));
    let rd = [
        region[0][1] - region[0][0],
        region[1][1] - region[1][0],
        region[2][1] - region[2][0],
    ];
    let mut out = vec![0f32; rd[0] * rd[1] * rd[2]];
    out.par_chunks_mut(rd[1] * rd[2])
        .enumerate()
        .for_each(|(oi, plane)| {
            let (i0, i1, t0, t1) = ax[0][region[0][0] + oi];
            for oj in 0..rd[1] {
                let (j0, j1, h0, h1) = ax[1][region[1][0] + oj];
                for ok in 0..rd[2] {
                    let (k0, k1, w0, w1) = ax[2][region[2][0] + ok];
                    let lin = |a: f32, wa: f32, b: f32, wb: f32| a.mul_add(wa, b * wb);
                    let inner = |i: usize, j: usize| lin(at(i, j, k0), w0, at(i, j, k1), w1);
                    let mid = |i: usize| lin(inner(i, j0), h0, inner(i, j1), h1);
                    plane[oj * rd[2] + ok] = lin(mid(i0), t0, mid(i1), t1);
                }
            }
        });
    out
}

/// `F.interpolate(x, size, mode="area")` - `adaptive_avg_pool3d` - of a
/// one-channel float16 volume read through `at`, as PyTorch's CPU kernel
/// computes it for float16: the window summed in float16 (every addition
/// rounded), then divided by its extent along each axis in turn, each
/// quotient rounded again.
pub fn area_f16(
    at: &(dyn Fn(usize, usize, usize) -> f32 + Sync),
    input: [usize; 3],
    output: [usize; 3],
) -> Vec<f32> {
    use rayon::prelude::*;
    let win = |a: usize, o: usize| -> (usize, usize) {
        let (i, n) = (input[a], output[a]);
        let start = (o * i) / n;
        let end = ((o + 1) * i).div_ceil(n);
        (start, end)
    };
    let mut out = vec![0f32; output[0] * output[1] * output[2]];
    out.par_chunks_mut(output[1] * output[2])
        .enumerate()
        .for_each(|(od, plane)| {
            let (d0, d1) = win(0, od);
            for oh in 0..output[1] {
                let (h0, h1) = win(1, oh);
                for ow in 0..output[2] {
                    let (w0, w1) = win(2, ow);
                    let mut sum = 0f32;
                    for d in d0..d1 {
                        for hh in h0..h1 {
                            for w in w0..w1 {
                                sum = h(sum + at(d, hh, w));
                            }
                        }
                    }
                    let q = h(sum / (d1 - d0) as f32);
                    let q = h(q / (h1 - h0) as f32);
                    plane[oh * output[2] + ow] = h(q / (w1 - w0) as f32);
                }
            }
        });
    out
}

/// The source index of `F.interpolate(x, size, mode="nearest")` along one
/// axis (`nearest_idx`).
pub fn nearest_index(o: usize, input: usize, output: usize) -> usize {
    if output == input {
        o
    } else if output == 2 * input {
        o >> 1
    } else {
        let scale = input as f32 / output as f32;
        ((o as f32 * scale).floor() as usize).min(input - 1)
    }
}

// ---------------------------------------------------------------- pooling --

/// Max (or min) over a `k`-cubed window, stride 1, same size, the volume
/// extended by replicating its border: `iterative_3x3_same_padding_pool3d`.
/// With replicated borders that is the extreme over the window clipped to
/// the volume, computed separably.
pub fn pool_extreme(data: &mut [f32], dims: [usize; 3], k: usize, min: bool) {
    if k <= 1 {
        return;
    }
    let r = (k - 1) / 2;
    let pick = |a: f32, b: f32| if min { a.min(b) } else { a.max(b) };
    let strides = [dims[1] * dims[2], dims[2], 1];
    let mut line = Vec::new();
    for axis in 0..3 {
        let n = dims[axis];
        let (o1, o2) = match axis {
            0 => (1, 2),
            1 => (0, 2),
            _ => (0, 1),
        };
        for a in 0..dims[o1] {
            for b in 0..dims[o2] {
                let base = a * strides[o1] + b * strides[o2];
                line.clear();
                line.extend((0..n).map(|t| data[base + t * strides[axis]]));
                for t in 0..n {
                    let lo = t.saturating_sub(r);
                    let hi = (t + r).min(n - 1);
                    let mut v = line[lo];
                    for &x in &line[lo + 1..=hi] {
                        v = pick(v, x);
                    }
                    data[base + t * strides[axis]] = v;
                }
            }
        }
    }
}

// -------------------------------------------------------- the box cover --

/// The deterministic stand-in for `np.random.choice` in the cover's random
/// fallback: a 64-bit LCG (the reference fixture patches the same one into
/// the Python session).
#[derive(Clone, Debug)]
pub struct Lcg(pub u64);

impl Default for Lcg {
    fn default() -> Lcg {
        Lcg(0x853C_49E6_748F_EA9B)
    }
}

impl Lcg {
    pub fn choice(&mut self, n: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

/// Inclusive-exclusive prefix sums of a mask, for box sums in O(1).
struct Prefix {
    dims: [usize; 3],
    s: Vec<u32>,
}

impl Prefix {
    fn new(mask: &[u8], dims: [usize; 3]) -> Prefix {
        let p = [dims[0] + 1, dims[1] + 1, dims[2] + 1];
        let mut s = vec![0u32; p[0] * p[1] * p[2]];
        let at = |i: usize, j: usize, k: usize| (i * p[1] + j) * p[2] + k;
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                let mut row = 0u32;
                for k in 0..dims[2] {
                    row += u32::from(mask[idx(dims, i, j, k)] != 0);
                    s[at(i + 1, j + 1, k + 1)] =
                        row + s[at(i, j + 1, k + 1)] + s[at(i + 1, j, k + 1)] - s[at(i, j, k + 1)];
                }
            }
        }
        Prefix { dims: p, s }
    }

    fn sum(&self, lo: [usize; 3], hi: [usize; 3]) -> u64 {
        if (0..3).any(|a| hi[a] <= lo[a]) {
            return 0;
        }
        let p = self.dims;
        let at = |i: usize, j: usize, k: usize| self.s[(i * p[1] + j) * p[2] + k] as i64;
        let v = at(hi[0], hi[1], hi[2])
            - at(lo[0], hi[1], hi[2])
            - at(hi[0], lo[1], hi[2])
            - at(hi[0], hi[1], lo[2])
            + at(lo[0], lo[1], hi[2])
            + at(lo[0], hi[1], lo[2])
            + at(hi[0], lo[1], lo[2])
            - at(lo[0], lo[1], lo[2]);
        v as u64
    }
}

fn zero_box(mask: &mut [u8], dims: [usize; 3], lo: [usize; 3], hi: [usize; 3]) {
    for i in lo[0]..hi[0] {
        for j in lo[1]..hi[1] {
            let a = idx(dims, i, j, lo[2]);
            mask[a..a + hi[2].saturating_sub(lo[2])].fill(0);
        }
    }
}

/// The part of a box centred on `c` that a pick marks as covered: the box
/// shrunk by the margin and clipped to the mask.
fn covered_bounds(
    c: [usize; 3],
    half: [usize; 3],
    end: [usize; 3],
    margin: [usize; 3],
    dims: [usize; 3],
) -> ([usize; 3], [usize; 3]) {
    let lo =
        std::array::from_fn(|a| (c[a] as i64 - half[a] as i64 + margin[a] as i64).max(0) as usize);
    let hi = std::array::from_fn(|a| {
        (c[a] as i64 + end[a] as i64 - margin[a] as i64)
            .min(dims[a] as i64)
            .max(0) as usize
    });
    (lo, hi)
}

fn box_at(c: [usize; 3], half: [usize; 3], end: [usize; 3]) -> BBox {
    std::array::from_fn(|a| [c[a] as i64 - half[a] as i64, c[a] as i64 + end[a] as i64])
}

/// How the candidate centres are spaced.
#[derive(Clone, Copy, Debug)]
pub enum Stride {
    /// A quarter of the object's extent per axis.
    Auto,
    Fixed([usize; 3]),
}

/// `nnInteractive.utils.bboxes.generate_bounding_boxes`: patch-sized boxes
/// that together cover the nonzero voxels of `mask`, chosen greedily from
/// candidate centres on a grid over the object, each pick marking its box
/// shrunk by `margin` as covered; whatever the grid cannot reach is covered
/// by [`random_fallback`]. `mask` is consumed (the fallback clears it).
#[allow(clippy::too_many_arguments)]
pub fn generate_bounding_boxes(
    mask: &mut [u8],
    dims: [usize; 3],
    bbox_size: [usize; 3],
    stride: Stride,
    margin: [usize; 3],
    max_depth: usize,
    depth: usize,
    rng: &mut Lcg,
) -> Vec<BBox> {
    let Some(nz) = nonzero_bbox(mask, dims) else {
        return Vec::new();
    };
    if depth > max_depth {
        return random_fallback(mask, dims, bbox_size, margin, 25, rng);
    }
    let half: [usize; 3] = std::array::from_fn(|a| bbox_size[a] / 2);
    let end: [usize; 3] = std::array::from_fn(|a| bbox_size[a] - half[a]);
    let min_c: [usize; 3] = std::array::from_fn(|a| nz[a][0] as usize);
    let max_c: [usize; 3] = std::array::from_fn(|a| nz[a][1] as usize - 1);
    let stride: [usize; 3] = match stride {
        Stride::Auto => {
            std::array::from_fn(|a| round_py((max_c[a] - min_c[a]) as f64 / 4.0).max(1) as usize)
        }
        Stride::Fixed(s) => s,
    };
    let mut centres: Vec<[usize; 3]> = Vec::new();
    for i in (min_c[0]..=max_c[0].min(dims[0] - 1)).step_by(stride[0]) {
        for j in (min_c[1]..=max_c[1].min(dims[1] - 1)).step_by(stride[1]) {
            for k in (min_c[2]..=max_c[2].min(dims[2] - 1)).step_by(stride[2]) {
                if mask[idx(dims, i, j, k)] != 0 {
                    centres.push([i, j, k]);
                }
            }
        }
    }
    if centres.is_empty() {
        let finer = std::array::from_fn(|a| (stride[a] / 2).max(1));
        return generate_bounding_boxes(
            mask,
            dims,
            bbox_size,
            Stride::Fixed(finer),
            margin,
            max_depth,
            depth + 1,
            rng,
        );
    }
    let mut uncovered = mask.to_vec();
    let mut boxes = Vec::new();
    while !centres.is_empty() && uncovered.iter().any(|&v| v != 0) {
        let prefix = Prefix::new(&uncovered, dims);
        let mut best: Option<(usize, Span)> = None;
        let mut best_n = 0u64;
        for (ci, &c) in centres.iter().enumerate() {
            let (lo, hi) = covered_bounds(c, half, end, margin, dims);
            let n = prefix.sum(lo, hi);
            if n > best_n {
                best_n = n;
                best = Some((ci, (lo, hi)));
            }
        }
        let Some((ci, (lo, hi))) = best else {
            break;
        };
        boxes.push(box_at(centres[ci], half, end));
        zero_box(&mut uncovered, dims, lo, hi);
        centres.retain(|c| uncovered[idx(dims, c[0], c[1], c[2])] != 0);
    }
    if uncovered.iter().any(|&v| v != 0) {
        boxes.extend(random_fallback(
            &mut uncovered,
            dims,
            bbox_size,
            margin,
            10,
            rng,
        ));
    }
    boxes
}

/// `random_sampling_fallback`: until `mask` is empty, draw `n_samples`
/// nonzero voxels as candidate centres and keep the one whose box covers
/// most. Clears `mask`.
pub fn random_fallback(
    mask: &mut [u8],
    dims: [usize; 3],
    bbox_size: [usize; 3],
    margin: [usize; 3],
    n_samples: usize,
    rng: &mut Lcg,
) -> Vec<BBox> {
    let half: [usize; 3] = std::array::from_fn(|a| bbox_size[a] / 2);
    let end: [usize; 3] = std::array::from_fn(|a| bbox_size[a] - half[a]);
    let mut boxes = Vec::new();
    loop {
        let indices: Vec<usize> = (0..mask.len()).filter(|&v| mask[v] != 0).collect();
        if indices.is_empty() {
            break;
        }
        let prefix = Prefix::new(mask, dims);
        let mut best: Option<([usize; 3], Span)> = None;
        let mut best_n = 0u64;
        for _ in 0..n_samples {
            let v = indices[rng.choice(indices.len())];
            let c = [
                v / (dims[1] * dims[2]),
                (v / dims[2]) % dims[1],
                v % dims[2],
            ];
            let (lo, hi) = covered_bounds(c, half, end, margin, dims);
            let n = prefix.sum(lo, hi);
            if n > best_n {
                best_n = n;
                best = Some((c, (lo, hi)));
            }
        }
        // A margin of half the patch or more covers nothing; upstream fails
        // here, and so does nothing else: stop rather than loop.
        let Some((c, (lo, hi))) = best else {
            break;
        };
        boxes.push(box_at(c, half, end));
        zero_box(mask, dims, lo, hi);
    }
    boxes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_point_is_a_normalized_distance_ball() {
        let (n, s) = point_strel(4);
        assert_eq!(n, 9);
        let c = s[(4 * 9 + 4) * 9 + 4];
        assert_eq!(c, 1.0, "the centre is the maximum");
        // a corner of the cube is outside the ball
        assert_eq!(s[0], 0.0);
        // a face centre is inside, one voxel from the background
        let face = s[(4 * 9 + 4) * 9];
        assert!(face > 0.0 && face < 0.5, "{face}");
        assert!(s.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn odd_rounding_matches_upstream() {
        assert_eq!(round_to_nearest_odd(1.0), 1);
        assert_eq!(round_to_nearest_odd(2.0), 3);
        assert_eq!(round_to_nearest_odd(2.5), 3);
        assert_eq!(round_to_nearest_odd(3.75), 3);
        assert_eq!(round_to_nearest_odd(4.0), 5);
        assert_eq!(round_to_nearest_odd(7.0), 7);
    }

    #[test]
    fn linear_resampling_is_exact_at_identity_and_averages_when_halving() {
        let data: Vec<f32> = (0..4 * 4 * 4).map(|v| v as f32).collect();
        let at = |i: usize, j: usize, k: usize| data[idx([4, 4, 4], i, j, k)];
        let same = trilinear_region(&at, [4, 4, 4], [4, 4, 4], [[0, 4], [0, 4], [0, 4]]);
        assert_eq!(same, data);
        let half = trilinear_region(&at, [4, 4, 4], [2, 2, 2], [[0, 2], [0, 2], [0, 2]]);
        // the first output voxel averages the first 2-cube
        let want = (0..2)
            .flat_map(|i| (0..2).flat_map(move |j| (0..2).map(move |k| at(i, j, k))))
            .sum::<f32>()
            / 8.0;
        assert!((half[0] - want).abs() < 1e-5, "{} vs {want}", half[0]);
    }

    #[test]
    fn area_windows_follow_adaptive_pooling() {
        // 5 -> 3 along one axis: windows [0,2), [1,4), [3,5)
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let at = |_: usize, _: usize, k: usize| data[k];
        let out = area_f16(&at, [1, 1, 5], [1, 1, 3]);
        assert_eq!(out, vec![1.5, 3.0, 4.5]);
        // the sum is kept in float16: 2048 + 1 is still 2048 there
        let big = [2048.0f32, 1.0];
        let at = |_: usize, _: usize, k: usize| big[k];
        assert_eq!(area_f16(&at, [1, 1, 2], [1, 1, 1]), vec![1024.0]);
    }

    #[test]
    fn nearest_follows_pytorch() {
        let v: Vec<usize> = (0..4).map(|o| nearest_index(o, 6, 4)).collect();
        assert_eq!(v, [0, 1, 3, 4]);
        let v: Vec<usize> = (0..6).map(|o| nearest_index(o, 3, 6)).collect();
        assert_eq!(v, [0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn pooling_clips_its_window_at_the_border() {
        let mut d = vec![0f32; 7];
        d[0] = 1.0;
        pool_extreme(&mut d, [1, 1, 7], 5, false);
        assert_eq!(d, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let mut e = vec![1f32; 7];
        e[3] = 0.0;
        pool_extreme(&mut e, [1, 1, 7], 3, true);
        assert_eq!(e, [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn the_cover_reaches_every_voxel() {
        let dims = [20, 20, 20];
        let mut mask = vec![0u8; 8000];
        for i in 3..17 {
            for j in 5..9 {
                for k in 2..18 {
                    mask[idx(dims, i, j, k)] = 1;
                }
            }
        }
        let want = mask.clone();
        let mut rng = Lcg::default();
        let boxes = generate_bounding_boxes(
            &mut mask,
            dims,
            [8, 8, 8],
            Stride::Auto,
            [1, 1, 1],
            3,
            0,
            &mut rng,
        );
        assert!(!boxes.is_empty());
        for (v, &m) in want.iter().enumerate() {
            if m == 0 {
                continue;
            }
            let c = [v / 400, (v / 20) % 20, v % 20];
            assert!(
                boxes
                    .iter()
                    .any(|b| (0..3).all(|a| (c[a] as i64) >= b[a][0] && (c[a] as i64) < b[a][1])),
                "voxel {c:?} not covered"
            );
        }
        for b in &boxes {
            assert_eq!(bbox_size(b), [8, 8, 8]);
        }
    }

    #[test]
    fn pasting_clips_to_the_target() {
        let mut dst = vec![0u8; 27];
        let src = vec![1u8; 8];
        paste(
            &mut dst,
            [3, 3, 3],
            &src,
            [2, 2, 2],
            &[[-1, 1], [2, 4], [0, 2]],
        );
        let set: Vec<usize> = (0..27).filter(|&v| dst[v] == 1).collect();
        // only i = 0 and j = 2 are inside; k = 0, 1
        assert_eq!(set, vec![idx([3, 3, 3], 0, 2, 0), idx([3, 3, 3], 0, 2, 1)]);
    }
}
