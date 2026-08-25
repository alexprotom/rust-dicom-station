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

    /// View the volume as a `[c, d*h*w]` matrix — channels are rows.
    pub fn to_mat(&self) -> Mat {
        Mat {
            rows: self.c,
            cols: self.spatial(),
            data: self.data.clone(),
        }
    }

    /// The same view, taking the storage along: the layouts are identical,
    /// so nothing is copied.
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

/// Transposed 3D convolution with kernel = stride = 2: every input voxel
/// projects to a disjoint 2x2x2 output block. `weight`: `[cin, cout, 2, 2, 2]`
/// — PyTorch's `ConvTranspose3d` layout.
pub fn conv_transpose3d_2x(x: &Act, weight: &[f32], bias: &[f32], cout: usize) -> Act {
    let (cin, d, h, w) = (x.c, x.d, x.h, x.w);
    debug_assert_eq!(weight.len(), cin * cout * 8);
    let (od, oh, ow) = (d * 2, h * 2, w * 2);
    // Repack weight as [cout*8, cin] for a row-major GEMM.
    let mut wt = vec![0f32; cout * 8 * cin];
    for ci in 0..cin {
        for co in 0..cout {
            for t in 0..8 {
                wt[(co * 8 + t) * cin + ci] = weight[(ci * cout + co) * 8 + t];
            }
        }
    }
    let hw = h * w;
    let mut out = Act::zeros(cout, od, oh, ow);
    let ohw = oh * ow;
    let out_ptr = SendPtr(out.data.as_mut_ptr());
    let od_stride = od * ohw; // per-channel stride in the output
                              // one input slice z -> output slices 2z, 2z+1 (disjoint across the loop)
    (0..d).into_par_iter().for_each(|z| {
        // gather input slice as [cin, hw]
        let mut xin = vec![0f32; cin * hw];
        for c in 0..cin {
            let src = &x.data[c * d * hw + z * hw..c * d * hw + (z + 1) * hw];
            xin[c * hw..(c + 1) * hw].copy_from_slice(src);
        }
        // GEMM: [cout*8, cin] x [cin, hw] -> [cout*8, hw]
        let mut tmp = vec![0f32; cout * 8 * hw];
        unsafe {
            gemm::gemm(
                cout * 8,
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
        // scatter: tmp[(co*8 + (dz*4+dy*2+dx)) * hw + y*w + x]
        //   -> out[co][2z+dz][2y+dy][2x+dx]
        for co in 0..cout {
            let bv = bias[co];
            for dz in 0..2usize {
                let obase = co * od_stride + (2 * z + dz) * ohw;
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        let t = dz * 4 + dy * 2 + dx;
                        let src = &tmp[(co * 8 + t) * hw..(co * 8 + t + 1) * hw];
                        for y in 0..h {
                            let orow = obase + (2 * y + dy) * ow + dx;
                            let dst = unsafe {
                                std::slice::from_raw_parts_mut(out_ptr.get().add(orow), 2 * w - 1)
                            };
                            let srow = &src[y * w..(y + 1) * w];
                            for (xi, sv) in srow.iter().enumerate() {
                                dst[2 * xi] = sv + bv;
                            }
                        }
                    }
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn tokens_to_volume_transposes() {
        // 2 tokens x 3 channels -> [3,1,1,2]
        let m = Mat::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let a = Act::from_tokens(&m, 1, 1, 2);
        assert_eq!((a.c, a.d, a.h, a.w), (3, 1, 1, 2));
        assert_eq!(a.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // and back through to_mat: channels are rows
        assert_eq!(a.to_mat().data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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
