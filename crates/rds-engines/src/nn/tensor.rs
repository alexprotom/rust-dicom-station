//! Dense f32 tensors in the two shapes the inference engines actually use.
//!
//! A transformer is almost entirely `[tokens, dim]` matrices, and a
//! convolutional decoder is almost entirely `[channels, d, h, w]` volumes, so
//! there are two concrete types rather than one n-dimensional one. Both are
//! row-major and own their data; every kernel takes and returns them by
//! value or reference, and no operation is lazy.

use rayon::prelude::*;

/// A row-major `rows` x `cols` matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Mat {
        assert_eq!(
            data.len(),
            rows * cols,
            "{rows}x{cols} needs {} values",
            rows * cols
        );
        Mat { rows, cols, data }
    }

    /// A matrix whose single row is `v`.
    pub fn row_vec(v: &[f32]) -> Mat {
        Mat {
            rows: 1,
            cols: v.len(),
            data: v.to_vec(),
        }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let c = self.cols;
        &mut self.data[r * c..(r + 1) * c]
    }

    /// Stack `other` underneath: `[self; other]`. Columns must agree.
    pub fn vcat(mut self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.cols);
        self.data.extend_from_slice(&other.data);
        self.rows += other.rows;
        self
    }

    /// Elementwise `self += other`, broadcasting a single row over all rows.
    pub fn add_assign(&mut self, other: &Mat) {
        assert_eq!(self.cols, other.cols);
        if other.rows == self.rows {
            for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
                *a += b;
            }
        } else if other.rows == 1 {
            let cols = self.cols;
            self.data.par_chunks_mut(cols).for_each(|r| {
                for (a, b) in r.iter_mut().zip(other.data.iter()) {
                    *a += b;
                }
            });
        } else {
            panic!("cannot add {} rows to {} rows", other.rows, self.rows);
        }
    }

    /// Rows `[from, to)` as a new matrix.
    pub fn rows_slice(&self, from: usize, to: usize) -> Mat {
        Mat {
            rows: to - from,
            cols: self.cols,
            data: self.data[from * self.cols..to * self.cols].to_vec(),
        }
    }
}

/// A dense f32 volume `[c][d][h][w]`, C-contiguous.
#[derive(Clone, Debug)]
pub struct Act {
    pub c: usize,
    pub d: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Act {
    pub fn zeros(c: usize, d: usize, h: usize, w: usize) -> Act {
        Act {
            c,
            d,
            h,
            w,
            data: vec![0.0; c * d * h * w],
        }
    }

    #[inline]
    pub fn spatial(&self) -> usize {
        self.d * self.h * self.w
    }

    /// View the volume as a `[c, d*h*w]` matrix - channels are rows; the
    /// layouts are identical, so nothing is copied.
    pub fn into_mat(self) -> Mat {
        Mat {
            rows: self.c,
            cols: self.spatial(),
            data: self.data,
        }
    }

    /// Interpret a `[tokens, channels]` matrix as a volume, transposing so
    /// channels become the outer axis. `tokens` must equal `d*h*w`.
    pub fn from_tokens(m: &Mat, d: usize, h: usize, w: usize) -> Act {
        assert_eq!(m.rows, d * h * w);
        let c = m.cols;
        let mut out = Act::zeros(c, d, h, w);
        let sp = d * h * w;
        out.data
            .par_chunks_mut(sp)
            .enumerate()
            .for_each(|(ch, dst)| {
                for (t, v) in dst.iter_mut().enumerate() {
                    *v = m.data[t * c + ch];
                }
            });
        out
    }
}

/// Wrapper making a raw pointer Send/Sync for disjoint-slice parallel writes.
pub(crate) struct SendPtr(pub(crate) *mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// Method (not field) access, so closures capture the whole wrapper.
    pub(crate) fn get(&self) -> *mut f32 {
        self.0
    }
}

/// Spare activation buffers, handed from one layer, and one patch, to the
/// next instead of being freed and allocated again.
///
/// At full resolution a layer's output is hundreds of megabytes. Buffers
/// that large come straight from the operating system and go back to it
/// when freed, so every layer of every patch would fault its output in page
/// by page again; through an arena the pages stay mapped for the whole run.
/// A buffer taken from here keeps what its last user left in it: it is only
/// ever handed to an operation that writes every element of its output.
#[derive(Default)]
pub struct Arena {
    spare: Vec<Vec<f32>>,
}

impl Arena {
    /// A buffer of `len` floats with stale contents: the smallest spare that
    /// already holds that many, else the largest one grown, else a new one.
    pub fn take(&mut self, len: usize) -> Vec<f32> {
        let fits = (0..self.spare.len())
            .filter(|&i| self.spare[i].len() >= len)
            .min_by_key(|&i| self.spare[i].len());
        let pick = fits.or_else(|| (0..self.spare.len()).max_by_key(|&i| self.spare[i].capacity()));
        match pick {
            Some(i) => {
                let mut v = self.spare.swap_remove(i);
                if v.len() >= len {
                    v.truncate(len);
                } else {
                    v.resize(len, 0.0);
                }
                v
            }
            None => vec![0.0; len],
        }
    }

    /// [`take`](Self::take) shaped as an activation volume.
    pub fn act(&mut self, c: usize, d: usize, h: usize, w: usize) -> Act {
        Act {
            c,
            d,
            h,
            w,
            data: self.take(c * d * h * w),
        }
    }

    /// Hand a buffer back for the next [`take`](Self::take).
    pub fn give(&mut self, v: Vec<f32>) {
        if v.capacity() > 0 {
            self.spare.push(v);
        }
    }
}

/// Transposed 3-D convolution with `kernel = stride`, for any stride.
///
/// Since kernel equals stride the output tiles are disjoint - no
/// overlap-add, just a GEMM that expands every input voxel into its
/// `s0·s1·s2` outputs and a scatter. The 2× case (every nnU-Net decoder,
/// SegVol's) and the anisotropic ones (the MR body model halves two axes at
/// a time and leaves the third alone) are the one routine,
/// [`conv_transpose3d_stride_into_slice`].
pub fn conv_transpose3d_stride(
    x: &Act,
    weight: &[f32],
    bias: &[f32],
    cout: usize,
    stride: [usize; 3],
) -> Act {
    let out = Act::zeros(cout, x.d * stride[0], x.h * stride[1], x.w * stride[2]);
    conv_transpose3d_stride_into(x, weight, bias, stride, out)
}

/// [`conv_transpose3d_stride`] into `out`, already shaped
/// `[cout, d*s0, h*s1, w*s2]`; every element is written, so its contents
/// may be stale (an [`Arena`] buffer).
pub fn conv_transpose3d_stride_into(
    x: &Act,
    weight: &[f32],
    bias: &[f32],
    stride: [usize; 3],
    mut out: Act,
) -> Act {
    debug_assert_eq!(
        (out.d, out.h, out.w),
        (x.d * stride[0], x.h * stride[1], x.w * stride[2])
    );
    let cout = out.c;
    conv_transpose3d_stride_into_slice(x, weight, bias, stride, cout, &mut out.data);
    out
}

thread_local! {
    /// Per-thread GEMM output of one input slice (`[cout * taps, h * w]`),
    /// reused from slice to slice and layer to layer rather than allocated
    /// - and page-faulted in - for every slice.
    static UP_TMP: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// [`conv_transpose3d_stride`] into `out`, the `cout * d*s0 * h*s1 * w*s2`
/// values of the output in `[cout, d*s0, h*s1, w*s2]` order: the front of
/// a larger buffer, say, whose other channels a concatenation fills, so the
/// upsampled channels are never copied. Every element is written.
///
/// One input slice `z` writes the output slices `z*s0 .. z*s0+s0`,
/// disjoint across the parallel loop. Its GEMM, `[cout*taps, cin] x
/// [cin, h*w]`, reads the slice straight out of the input (row stride: one
/// channel, `d*h*w`) rather than from a gathered copy; the GEMM packs its
/// operands itself, so the products and their sums are the ones the copy
/// gave, to the bit.
pub fn conv_transpose3d_stride_into_slice(
    x: &Act,
    weight: &[f32],
    bias: &[f32],
    stride: [usize; 3],
    cout: usize,
    out: &mut [f32],
) {
    let [s0, s1, s2] = stride;
    let (cin, d, h, w) = (x.c, x.d, x.h, x.w);
    let ks = s0 * s1 * s2;
    debug_assert_eq!(weight.len(), cin * cout * ks);
    let (od, oh, ow) = (d * s0, h * s1, w * s2);
    assert_eq!(out.len(), cout * od * oh * ow, "output size");
    // Repack the weight as [cout*ks, cin] for a row-major GEMM.
    let mut wt = vec![0f32; cout * ks * cin];
    for ci in 0..cin {
        for co in 0..cout {
            for t in 0..ks {
                wt[(co * ks + t) * cin + ci] = weight[(ci * cout + co) * ks + t];
            }
        }
    }
    let hw = h * w;
    let ohw = oh * ow;
    let out_ptr = SendPtr(out.as_mut_ptr());
    let od_stride = od * ohw; // per-channel stride in the output
    (0..d).into_par_iter().for_each(|z| {
        UP_TMP.with_borrow_mut(|tmp| {
            // Every element is written by the GEMM (beta 1, alpha 0).
            tmp.resize(cout * ks * hw, 0.0);
            unsafe {
                gemm::gemm(
                    cout * ks,
                    hw,
                    cin,
                    tmp.as_mut_ptr(),
                    1,
                    hw as isize,
                    false,
                    wt.as_ptr(),
                    1,
                    cin as isize,
                    x.data.as_ptr().add(z * hw),
                    1,
                    (d * hw) as isize,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            // scatter: tmp[(co*ks + t) * hw + y*w + x]
            //   -> out[co][z*s0+dz][y*s1+dy][x*s2+dx], plus the bias
            for co in 0..cout {
                let bv = bias[co];
                for dz in 0..s0 {
                    let obase = co * od_stride + (z * s0 + dz) * ohw;
                    for dy in 0..s1 {
                        for dx in 0..s2 {
                            let t = (dz * s1 + dy) * s2 + dx;
                            let src = &tmp[(co * ks + t) * hw..(co * ks + t + 1) * hw];
                            for y in 0..h {
                                let orow = obase + (y * s1 + dy) * ow + dx;
                                // SAFETY: input slice z owns output slices
                                // z*s0..z*s0+s0 of every channel.
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        out_ptr.get().add(orow),
                                        (w - 1) * s2 + 1,
                                    )
                                };
                                let srow = &src[y * w..(y + 1) * w];
                                for (xi, sv) in srow.iter().enumerate() {
                                    dst[xi * s2] = sv + bv;
                                }
                            }
                        }
                    }
                }
            }
        });
    });
}

/// Transposed 3D convolution with kernel = stride = 2: every input voxel
/// projects to a disjoint 2x2x2 output block. `weight`: `[cin, cout, 2, 2, 2]` -
/// PyTorch's `ConvTranspose3d` layout.
pub fn conv_transpose3d_2x(x: &Act, weight: &[f32], bias: &[f32], cout: usize) -> Act {
    conv_transpose3d_stride(x, weight, bias, cout, [2, 2, 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transposed convolution as it was before it read its input in
    /// place: the input slice gathered into a copy, the GEMM's output in a
    /// fresh buffer. The routine that replaced it must give the same bits.
    fn conv_transpose_gathered(
        x: &Act,
        weight: &[f32],
        bias: &[f32],
        cout: usize,
        stride: [usize; 3],
    ) -> Vec<f32> {
        let [s0, s1, s2] = stride;
        let (cin, d, h, w) = (x.c, x.d, x.h, x.w);
        let ks = s0 * s1 * s2;
        let (od, oh, ow) = (d * s0, h * s1, w * s2);
        let mut wt = vec![0f32; cout * ks * cin];
        for ci in 0..cin {
            for co in 0..cout {
                for t in 0..ks {
                    wt[(co * ks + t) * cin + ci] = weight[(ci * cout + co) * ks + t];
                }
            }
        }
        let hw = h * w;
        let ohw = oh * ow;
        let mut out = vec![f32::NAN; cout * od * ohw];
        for z in 0..d {
            let mut xin = vec![0f32; cin * hw];
            for c in 0..cin {
                xin[c * hw..(c + 1) * hw]
                    .copy_from_slice(&x.data[c * d * hw + z * hw..c * d * hw + (z + 1) * hw]);
            }
            let mut tmp = vec![0f32; cout * ks * hw];
            unsafe {
                gemm::gemm(
                    cout * ks,
                    hw,
                    cin,
                    tmp.as_mut_ptr(),
                    1,
                    hw as isize,
                    false,
                    wt.as_ptr(),
                    1,
                    cin as isize,
                    xin.as_ptr(),
                    1,
                    hw as isize,
                    0.0f32,
                    1.0f32,
                    false,
                    false,
                    false,
                    gemm::Parallelism::None,
                );
            }
            for co in 0..cout {
                for dz in 0..s0 {
                    for dy in 0..s1 {
                        for dx in 0..s2 {
                            let t = (dz * s1 + dy) * s2 + dx;
                            for y in 0..h {
                                for xi in 0..w {
                                    let o = co * od * ohw
                                        + (z * s0 + dz) * ohw
                                        + (y * s1 + dy) * ow
                                        + xi * s2
                                        + dx;
                                    out[o] = tmp[(co * ks + t) * hw + y * w + xi] + bias[co];
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn transposed_conv_in_place_is_bit_identical_to_the_gathered_one() {
        let mut seed = 0x5eed_u64;
        for (cin, cout, dims, stride) in [
            (64usize, 32usize, [7usize, 9usize, 11usize], [2usize, 2, 2]),
            (320, 256, [3, 4, 5], [2, 2, 2]),
            (16, 8, [5, 6, 3], [1, 2, 2]),
            (3, 5, [2, 3, 4], [2, 1, 3]),
        ] {
            let mut x = Act::zeros(cin, dims[0], dims[1], dims[2]);
            for v in &mut x.data {
                *v = rngf(&mut seed);
            }
            let ks = stride.iter().product::<usize>();
            let weight: Vec<f32> = (0..cin * cout * ks).map(|_| rngf(&mut seed)).collect();
            let bias: Vec<f32> = (0..cout).map(|_| rngf(&mut seed)).collect();
            let want = conv_transpose_gathered(&x, &weight, &bias, cout, stride);
            // Stale contents in the output must not matter.
            let mut out = Act::zeros(
                cout,
                dims[0] * stride[0],
                dims[1] * stride[1],
                dims[2] * stride[2],
            );
            out.data.fill(f32::NAN);
            let got = conv_transpose3d_stride_into(&x, &weight, &bias, stride, out);
            assert_eq!(got.data.len(), want.len());
            for (i, (a, b)) in got.data.iter().zip(&want).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{cin}->{cout} {stride:?}: element {i}"
                );
            }
        }
    }

    fn rngf(seed: &mut u64) -> f32 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    }

    #[test]
    fn transpose_conv_matches_the_definition() {
        let mut s = 5u64;
        let mut x = Act::zeros(3, 3, 4, 2);
        for v in &mut x.data {
            *v = rngf(&mut s);
        }
        let w: Vec<f32> = (0..3 * 2 * 8).map(|_| rngf(&mut s)).collect();
        let b: Vec<f32> = (0..2).map(|_| rngf(&mut s)).collect();
        let y = conv_transpose3d_2x(&x, &w, &b, 2);
        assert_eq!((y.c, y.d, y.h, y.w), (2, 6, 8, 4));
        // y[co, 2z+dz, 2y+dy, 2x+dx] = b + sum_ci x[ci,z,y,x] * w[ci,co,dz,dy,dx]
        for co in 0..2 {
            for z in 0..3 {
                for yy in 0..4 {
                    for xx in 0..2 {
                        for dz in 0..2 {
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let mut acc = b[co];
                                    for ci in 0..3 {
                                        let xv = x.data[((ci * 3 + z) * 4 + yy) * 2 + xx];
                                        let wv = w[((ci * 2 + co) * 8) + dz * 4 + dy * 2 + dx];
                                        acc += xv * wv;
                                    }
                                    let got = y.data[((co * 6 + 2 * z + dz) * 8 + 2 * yy + dy) * 4
                                        + 2 * xx
                                        + dx];
                                    assert!((got - acc).abs() < 1e-4);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The general `kernel = stride` form against its own definition, at the
    /// anisotropic strides the MR body model's decoder actually plans. The
    /// 2× fast path is covered above; without this the general path - the
    /// only code the MR model runs through - has no test at all.
    #[test]
    fn transpose_conv_matches_the_definition_at_any_stride() {
        for stride in [[1, 2, 2], [2, 2, 1], [1, 1, 2], [3, 2, 1], [1, 1, 1]] {
            let [s0, s1, s2] = stride;
            let ks = s0 * s1 * s2;
            let mut seed = 11u64;
            let (cin, cout, d, h, w) = (3usize, 2usize, 3usize, 4usize, 2usize);
            let mut x = Act::zeros(cin, d, h, w);
            for v in &mut x.data {
                *v = rngf(&mut seed);
            }
            let wt: Vec<f32> = (0..cin * cout * ks).map(|_| rngf(&mut seed)).collect();
            let b: Vec<f32> = (0..cout).map(|_| rngf(&mut seed)).collect();
            let y = conv_transpose3d_stride(&x, &wt, &b, cout, stride);
            assert_eq!(
                (y.c, y.d, y.h, y.w),
                (cout, d * s0, h * s1, w * s2),
                "shape at {stride:?}"
            );
            for co in 0..cout {
                for z in 0..d {
                    for yy in 0..h {
                        for xx in 0..w {
                            for dz in 0..s0 {
                                for dy in 0..s1 {
                                    for dx in 0..s2 {
                                        let mut acc = b[co];
                                        let t = (dz * s1 + dy) * s2 + dx;
                                        for ci in 0..cin {
                                            let xv = x.data[((ci * d + z) * h + yy) * w + xx];
                                            let wv = wt[(ci * cout + co) * ks + t];
                                            acc += xv * wv;
                                        }
                                        let got = y.data[((co * d * s0 + z * s0 + dz) * h * s1
                                            + yy * s1
                                            + dy)
                                            * w
                                            * s2
                                            + xx * s2
                                            + dx];
                                        assert!(
                                            (got - acc).abs() < 1e-4,
                                            "stride {stride:?} at {co},{z},{yy},{xx}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tokens_to_volume_transposes() {
        // 2 tokens x 3 channels -> [3,1,1,2]
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = Act::from_tokens(&m, 1, 1, 2);
        assert_eq!((a.c, a.d, a.h, a.w), (3, 1, 1, 2));
        assert_eq!(a.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // and back through into_mat: channels are rows
        assert_eq!(a.into_mat().data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn broadcast_add_and_concat() {
        let mut m = Mat::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        m.add_assign(&Mat::row_vec(&[10.0, 20.0]));
        assert_eq!(m.data, vec![11.0, 22.0, 13.0, 24.0]);
        let joined = m.clone().vcat(&Mat::row_vec(&[0.0, 0.0]));
        assert_eq!(joined.rows, 3);
        assert_eq!(joined.rows_slice(0, 2), m);
    }
}
