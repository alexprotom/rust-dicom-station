//! lungmask's per-slice preparation (`lungmask/utils.py`): clip to
//! [-1024, 600] HU, find the body with a coarse threshold-and-morphology
//! mask, crop the slice to the body's bounding box, resize the crop to
//! 256 x 256, and later map the network's 256 x 256 answer back into the
//! box.
//!
//! Every step is scipy's or scikit-image's, reproduced to the voxel:
//! `ndimage.zoom` (endpoint-aligned coordinates, nearest for order 0 as
//! `floor(c + 0.5)`, linear for order 1, integer results rounded half away
//! from zero, output size `round(n * factor)` with Python's rounding), the
//! binary morphology with its default cross structure and a zero border,
//! `binary_fill_holes`, and connected components numbered in raster order.

/// Network input size.
pub const SIZE: usize = 256;
/// The coarse grid the body mask is found on.
const COARSE: usize = 128;

/// A 2-D image, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Img<T> {
    pub h: usize,
    pub w: usize,
    pub data: Vec<T>,
}

impl<T: Copy + Default> Img<T> {
    pub fn new(h: usize, w: usize) -> Img<T> {
        Img {
            h,
            w,
            data: vec![T::default(); h * w],
        }
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> T {
        self.data[r * self.w + c]
    }
}

/// `scipy.ndimage.zoom`'s output length for one axis: `round(n * f)`, with
/// Python's round-half-to-even.
fn zoom_len(n: usize, factor: f64) -> usize {
    (n as f64 * factor).round_ties_even() as usize
}

/// The input coordinate step of a zoom from `n_in` to `n_out` samples
/// (`grid_mode=False`: first and last sample centres coincide).
fn zoom_step(n_in: usize, n_out: usize) -> f64 {
    if n_out > 1 {
        (n_in as f64 - 1.0) / (n_out as f64 - 1.0)
    } else {
        1.0
    }
}

/// `ndimage.zoom(img, factors, order=0)`.
pub fn zoom_nearest<T: Copy + Default>(img: &Img<T>, out_h: usize, out_w: usize) -> Img<T> {
    let (sy, sx) = (zoom_step(img.h, out_h), zoom_step(img.w, out_w));
    let mut out = Img::new(out_h, out_w);
    for r in 0..out_h {
        let rr = ((r as f64 * sy + 0.5).floor() as usize).min(img.h - 1);
        for c in 0..out_w {
            let cc = ((c as f64 * sx + 0.5).floor() as usize).min(img.w - 1);
            out.data[r * out_w + c] = img.at(rr, cc);
        }
    }
    out
}

/// `ndimage.zoom(img, factors, order=1)` on an integer image, the result
/// rounded half away from zero back to an integer as scipy does for an
/// integer output.
pub fn zoom_linear_i16(img: &Img<i16>, out_h: usize, out_w: usize) -> Img<i16> {
    let (sy, sx) = (zoom_step(img.h, out_h), zoom_step(img.w, out_w));
    let tap = |n: usize, c: f64| -> (usize, usize, f64) {
        let i0 = c.floor() as usize;
        let t = c - i0 as f64;
        // At the last sample t is 0 and the second tap (outside, the
        // constant 0) carries no weight.
        (i0.min(n - 1), (i0 + 1).min(n - 1), t)
    };
    let mut out = Img::new(out_h, out_w);
    for r in 0..out_h {
        let (r0, r1, ty) = tap(img.h, r as f64 * sy);
        for c in 0..out_w {
            let (c0, c1, tx) = tap(img.w, c as f64 * sx);
            let v00 = img.at(r0, c0) as f64;
            let v01 = img.at(r0, c1) as f64;
            let v10 = img.at(r1, c0) as f64;
            let v11 = img.at(r1, c1) as f64;
            // scipy applies the axes as a tensor product of 1-D weights.
            let v = (1.0 - ty) * ((1.0 - tx) * v00 + tx * v01) + ty * ((1.0 - tx) * v10 + tx * v11);
            out.data[r * out_w + c] = v.round() as i16;
        }
    }
    out
}

const CROSS: [(isize, isize); 5] = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)];

fn neighbours(
    img: &Img<bool>,
    r: usize,
    c: usize,
    offs: &[(isize, isize)],
    border: bool,
    any: bool,
) -> bool {
    // `any`: dilation (one neighbour set suffices); otherwise erosion (all).
    for &(dr, dc) in offs {
        let (y, x) = (r as isize + dr, c as isize + dc);
        let v = if y < 0 || x < 0 || y >= img.h as isize || x >= img.w as isize {
            border
        } else {
            img.at(y as usize, x as usize)
        };
        if any && v {
            return true;
        }
        if !any && !v {
            return false;
        }
    }
    !any
}

/// `ndimage.binary_dilation` with the cross, `iterations` times.
pub fn dilate(img: &Img<bool>, iterations: usize) -> Img<bool> {
    let mut cur = img.clone();
    for _ in 0..iterations {
        let mut next = Img::new(cur.h, cur.w);
        for r in 0..cur.h {
            for c in 0..cur.w {
                next.data[r * cur.w + c] = neighbours(&cur, r, c, &CROSS, false, true);
            }
        }
        cur = next;
    }
    cur
}

/// `ndimage.binary_erosion` with the cross and a zero border, `iterations`
/// times.
pub fn erode(img: &Img<bool>, iterations: usize) -> Img<bool> {
    let mut cur = img.clone();
    for _ in 0..iterations {
        let mut next = Img::new(cur.h, cur.w);
        for r in 0..cur.h {
            for c in 0..cur.w {
                next.data[r * cur.w + c] = neighbours(&cur, r, c, &CROSS, false, false);
            }
        }
        cur = next;
    }
    cur
}

/// `ndimage.binary_fill_holes(img, structure=np.ones((3, 3)))`: the
/// background that cannot reach the outside through 8-connected background
/// becomes foreground.
pub fn fill_holes_8(img: &Img<bool>) -> Img<bool> {
    let (h, w) = (img.h, img.w);
    let mut outside = vec![false; h * w];
    let mut stack = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if (r == 0 || c == 0 || r + 1 == h || c + 1 == w) && !img.at(r, c) {
                outside[r * w + c] = true;
                stack.push((r, c));
            }
        }
    }
    while let Some((r, c)) = stack.pop() {
        for dr in -1isize..=1 {
            for dc in -1isize..=1 {
                let (y, x) = (r as isize + dr, c as isize + dc);
                if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
                    continue;
                }
                let v = y as usize * w + x as usize;
                if !img.data[v] && !outside[v] {
                    outside[v] = true;
                    stack.push((y as usize, x as usize));
                }
            }
        }
    }
    Img {
        h,
        w,
        data: outside.iter().map(|o| !o).collect(),
    }
}

/// Connected components of a binary image, 4- or 8-connected, numbered
/// from 1 in raster order of their first pixel (scikit-image's `label`).
/// Returns the ids and the pixel count of each (index 0 unused).
pub fn label2d(img: &Img<bool>, eight: bool) -> (Vec<u32>, Vec<usize>) {
    let (h, w) = (img.h, img.w);
    let mut id = vec![0u32; h * w];
    let mut sizes = vec![0usize];
    let mut stack = Vec::new();
    for start in 0..h * w {
        if !img.data[start] || id[start] != 0 {
            continue;
        }
        let k = sizes.len() as u32;
        sizes.push(0);
        id[start] = k;
        stack.push(start);
        while let Some(v) = stack.pop() {
            sizes[k as usize] += 1;
            let (r, c) = ((v / w) as isize, (v % w) as isize);
            for dr in -1isize..=1 {
                for dc in -1isize..=1 {
                    if (dr == 0 && dc == 0) || (!eight && dr != 0 && dc != 0) {
                        continue;
                    }
                    let (y, x) = (r + dr, c + dc);
                    if y < 0 || x < 0 || y >= h as isize || x >= w as isize {
                        continue;
                    }
                    let u = y as usize * w + x as usize;
                    if img.data[u] && id[u] == 0 {
                        id[u] = k;
                        stack.push(u);
                    }
                }
            }
        }
    }
    (id, sizes)
}

/// `utils.simple_bodymask`: the body on one clipped slice, at the slice's
/// own size.
pub fn simple_bodymask(img: &Img<i16>) -> Img<bool> {
    let small = zoom_nearest(
        img,
        zoom_len(img.h, COARSE as f64 / img.h as f64),
        zoom_len(img.w, COARSE as f64 / img.w as f64),
    );
    let mut m = Img {
        h: small.h,
        w: small.w,
        data: small.data.iter().map(|&v| v > -500).collect(),
    };
    m = erode(&dilate(&m, 1), 1); // binary_closing
    m = fill_holes_8(&m);
    m = erode(&m, 2);
    let (id, sizes) = label2d(&m, false);
    if sizes.len() > 1 {
        let mut best = 1;
        for k in 2..sizes.len() {
            if sizes[k] > sizes[best] {
                best = k;
            }
        }
        m.data = id.iter().map(|&k| k as usize == best).collect();
        m = dilate(&m, 2);
    }
    let (sh, sw) = (m.h, m.w);
    zoom_nearest(
        &m,
        zoom_len(sh, img.h as f64 / sh as f64),
        zoom_len(sw, img.w as f64 / sw as f64),
    )
}

/// The crop box `[r0, c0, r1, c1)` of one slice: the bounding box of the
/// first (raster order) 8-connected piece of its body mask, or the whole
/// slice when there is none.
pub fn body_box(img: &Img<i16>) -> [usize; 4] {
    let m = simple_bodymask(img);
    let (id, sizes) = label2d(&m, true);
    if sizes.len() <= 1 {
        return [0, 0, img.h, img.w];
    }
    let (mut r0, mut c0, mut r1, mut c1) = (usize::MAX, usize::MAX, 0, 0);
    for (v, &k) in id.iter().enumerate() {
        if k == 1 {
            let (r, c) = (v / m.w, v % m.w);
            r0 = r0.min(r);
            c0 = c0.min(c);
            r1 = r1.max(r + 1);
            c1 = c1.max(c + 1);
        }
    }
    [r0, c0, r1, c1]
}

/// `utils.crop_and_resize` + the normalization of `LMInferer._inference`:
/// one slice (already clipped to [-1024, 600]) → the network's 256 x 256
/// input in [0, 1], and the box it came from.
pub fn prepare_slice(img: &Img<i16>) -> (Vec<f32>, [usize; 4]) {
    let b = body_box(img);
    let (h, w) = (b[2] - b[0], b[3] - b[1]);
    let mut crop = Img::new(h, w);
    for r in 0..h {
        for c in 0..w {
            crop.data[r * w + c] = img.at(b[0] + r, b[1] + c);
        }
    }
    let resized = zoom_linear_i16(
        &crop,
        zoom_len(h, SIZE as f64 / h as f64),
        zoom_len(w, SIZE as f64 / w as f64),
    );
    let x = resized
        .data
        .iter()
        .map(|&v| ((v.min(600) as f64 + 1024.0) / 1624.0) as f32)
        .collect();
    (x, b)
}

/// `utils.reshape_mask`: the network's 256 x 256 labels of one slice back
/// into its box on an `h x w` slice.
pub fn reshape_mask(mask: &Img<u8>, b: [usize; 4], h: usize, w: usize) -> Img<u8> {
    let (bh, bw) = (b[2] - b[0], b[3] - b[1]);
    let z = zoom_nearest(
        mask,
        zoom_len(mask.h, bh as f64 / mask.h as f64),
        zoom_len(mask.w, bw as f64 / mask.w as f64),
    );
    let mut out = Img::new(h, w);
    for r in 0..z.h.min(bh) {
        for c in 0..z.w.min(bw) {
            out.data[(b[0] + r) * w + b[1] + c] = z.at(r, c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_matches_scipy_on_small_cases() {
        // scipy.ndimage.zoom(np.array([[0, 10], [20, 31]], np.int16), 2, order=1)
        let a = Img {
            h: 2,
            w: 2,
            data: vec![0i16, 10, 20, 31],
        };
        let z = zoom_linear_i16(&a, 4, 4);
        assert_eq!(
            z.data,
            [0, 3, 7, 10, 7, 10, 14, 17, 13, 17, 20, 24, 20, 24, 27, 31]
        );
        // ... and negative halves round away from zero: zoom 1.5 of
        // [[-1, -2], [-3, -4]].
        let n = Img {
            h: 2,
            w: 2,
            data: vec![-1i16, -2, -3, -4],
        };
        assert_eq!(
            zoom_linear_i16(&n, 3, 3).data,
            [-1, -2, -2, -2, -3, -3, -3, -4, -4]
        );
        // Order 0 picks floor(c + 0.5).
        let m = Img {
            h: 1,
            w: 4,
            data: vec![1u8, 2, 3, 4],
        };
        assert_eq!(zoom_nearest(&m, 1, 7).data, [1, 2, 2, 3, 3, 4, 4]);
        assert_eq!(zoom_len(512, 128.0 / 512.0), 128);
    }

    #[test]
    fn holes_fill_only_where_the_background_is_shut_in() {
        // A ring with a hole, and a notch open to the outside diagonally.
        let rows = ["00000", "01110", "01010", "01110", "00000"];
        let img = Img {
            h: 5,
            w: 5,
            data: rows
                .iter()
                .flat_map(|r| r.bytes().map(|b| b == b'1'))
                .collect(),
        };
        let f = fill_holes_8(&img);
        assert!(f.at(2, 2));
        assert!(!f.at(0, 0));
        // Erosion with a zero border eats the edge.
        let full = Img {
            h: 3,
            w: 3,
            data: vec![true; 9],
        };
        assert_eq!(erode(&full, 1).data.iter().filter(|v| **v).count(), 1);
    }

    #[test]
    fn components_number_in_raster_order() {
        let rows = ["1001", "0001", "1100"];
        let img = Img {
            h: 3,
            w: 4,
            data: rows
                .iter()
                .flat_map(|r| r.bytes().map(|b| b == b'1'))
                .collect(),
        };
        let (id, sizes) = label2d(&img, false);
        assert_eq!(sizes, vec![0, 1, 2, 2]);
        assert_eq!((id[0], id[3], id[8]), (1, 2, 3));
        // Eight-connected, the lone pixel at (0,0) still stands alone.
        let (_, s8) = label2d(&img, true);
        assert_eq!(s8.len(), 4);
    }
}
