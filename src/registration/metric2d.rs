//! How alike a radiograph and a rendering are: the similarity measures of
//! the 2D/3D registration, and the 2D images, pyramids and masks they are
//! computed on.
//!
//! Every measure here is a *similarity* - larger is better - over the pixels
//! a mask lets through, and every one is computed in a fixed order on one
//! thread, so the same pair of images gives the same number on any machine.
//! The registration minimises its negative.
//!
//! * [`Metric2d::Ncc`]: normalised cross-correlation, in `[-1, 1]`;
//!   invariant to a gain and an offset of either image.
//! * [`Metric2d::GradientCorrelation`]: the mean of the NCCs of the
//!   horizontal and of the vertical Sobel gradients (Penney 1998). Sees
//!   edges only, so scatter, the heel effect and soft-tissue changes do not
//!   pull it.
//! * [`Metric2d::GradientDifference`]: `Σ A/(A + (∇I − s∇D)²)` over both
//!   gradient directions, `A` the variance of the radiograph's gradient and
//!   `s` the least-squares scale (Penney 1998), divided by the pixel count:
//!   in `(0, 1]`.
//! * [`Metric2d::PatternIntensity`]: on the difference image of the two
//!   scaled to 0..255, `Σ σ²/(σ² + (d(p) − d(q))²)` over neighbours `q`
//!   within 3 pixels, `σ = 10` (Weese 1997), divided by the pair count: in
//!   `(0, 1]`.
//! * [`Metric2d::LocalNcc`]: NCC in windows of `2r + 1` pixels around every
//!   pixel, averaged (Khamene 2006); for portal images whose contrast varies
//!   across the field. Window sums come from integral images.
//! * [`Metric2d::MutualInformation`]: Mattes mutual information, 32 × 32 bins,
//!   a cubic B-spline Parzen window on the rendering's axis, in nats.

use super::parzen::beta3;

/// A 2D image: row-major `[rows][columns]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Image2d {
    pub data: Vec<f32>,
    /// `[columns, rows]`.
    pub dims: [usize; 2],
}

impl Image2d {
    pub fn new(data: Vec<f32>, dims: [usize; 2]) -> Self {
        assert_eq!(
            data.len(),
            dims[0] * dims[1],
            "pixels and dimensions disagree"
        );
        Image2d { data, dims }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Half the size in each direction (rounded up): the `[1 3 3 1] / 8`
    /// binomial, centred between each pair of pixels, so coarse pixel `I`
    /// sits where fine pixels `2I` and `2I + 1` meet - where
    /// [`crate::projection::ProjectionGeometry::coarser`] puts it.
    pub fn halved(&self) -> Image2d {
        let (data, dims) = halve(&self.data, self.dims);
        Image2d { data, dims }
    }

    /// This image and `n − 1` halvings, finest first.
    pub fn pyramid(self, n: usize) -> Vec<Image2d> {
        let mut out = vec![self];
        while out.len() < n.max(1) {
            let next = out.last().unwrap().halved();
            out.push(next);
        }
        out
    }

    /// A transmission image (the detector's dose, bright where little
    /// attenuates) as line integrals: `−ln(I / I₀)`, `I₀` the brightest
    /// pixel, so it measures what a DRR measures.
    pub fn transmission_to_line_integral(&mut self) {
        let i0 = self.data.iter().copied().fold(0.0f32, f32::max).max(1e-6);
        for v in &mut self.data {
            *v = -((*v).max(i0 * 1e-6) / i0).ln();
        }
    }

    /// The same image with the sign of every pixel turned.
    pub fn negate(&mut self) {
        for v in &mut self.data {
            *v = -*v;
        }
    }
}

fn halve(data: &[f32], dims: [usize; 2]) -> (Vec<f32>, [usize; 2]) {
    let [w, h] = dims;
    let (cw, ch) = (w.div_ceil(2).max(1), h.div_ceil(2).max(1));
    let k = [1.0f32, 3.0, 3.0, 1.0];
    let at = |n: usize, i: isize| i.clamp(0, n as isize - 1) as usize;
    // Along the rows, then down the columns.
    let mut tmp = vec![0.0f32; cw * h];
    for j in 0..h {
        for ci in 0..cw {
            let mut s = 0.0;
            for (d, kd) in k.iter().enumerate() {
                s += kd * data[j * w + at(w, 2 * ci as isize - 1 + d as isize)];
            }
            tmp[j * cw + ci] = s / 8.0;
        }
    }
    let mut out = vec![0.0f32; cw * ch];
    for cj in 0..ch {
        for ci in 0..cw {
            let mut s = 0.0;
            for (d, kd) in k.iter().enumerate() {
                s += kd * tmp[at(h, 2 * cj as isize - 1 + d as isize) * cw + ci];
            }
            out[cj * cw + ci] = s / 8.0;
        }
    }
    (out, [cw, ch])
}

/// A mask of the same pixels: nonzero where a pixel counts.
pub type Mask2d = Vec<u8>;

/// Every pixel but a border `px` wide.
pub fn border_mask(dims: [usize; 2], px: usize) -> Mask2d {
    let [w, h] = dims;
    let mut m = vec![0u8; w * h];
    if w <= 2 * px || h <= 2 * px {
        m.iter_mut().for_each(|v| *v = 1);
        return m;
    }
    for j in px..h - px {
        for i in px..w - px {
            m[j * w + i] = 1;
        }
    }
    m
}

/// A mask at the next coarser level: a coarse pixel counts when most of
/// what it averages does.
pub fn halve_mask(mask: &[u8], dims: [usize; 2]) -> (Mask2d, [usize; 2]) {
    let f: Vec<f32> = mask
        .iter()
        .map(|&m| if m != 0 { 1.0 } else { 0.0 })
        .collect();
    let (h, d) = halve(&f, dims);
    (h.iter().map(|&v| (v >= 0.5) as u8).collect(), d)
}

/// The similarity measures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Metric2d {
    #[default]
    Ncc,
    GradientCorrelation,
    GradientDifference,
    PatternIntensity,
    LocalNcc,
    MutualInformation,
}

impl Metric2d {
    pub const ALL: [Metric2d; 6] = [
        Metric2d::Ncc,
        Metric2d::GradientCorrelation,
        Metric2d::GradientDifference,
        Metric2d::PatternIntensity,
        Metric2d::LocalNcc,
        Metric2d::MutualInformation,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Metric2d::Ncc => "Normalised cross-correlation",
            Metric2d::GradientCorrelation => "Gradient correlation",
            Metric2d::GradientDifference => "Gradient difference",
            Metric2d::PatternIntensity => "Pattern intensity",
            Metric2d::LocalNcc => "Local cross-correlation",
            Metric2d::MutualInformation => "Mutual information",
        }
    }

    /// The short name of a readout (`GC 0.41 ▶ 0.87`).
    pub fn tag(self) -> &'static str {
        match self {
            Metric2d::Ncc => "NCC",
            Metric2d::GradientCorrelation => "GC",
            Metric2d::GradientDifference => "GD",
            Metric2d::PatternIntensity => "PI",
            Metric2d::LocalNcc => "LNCC",
            Metric2d::MutualInformation => "MI",
        }
    }

    /// The name on a command line, in a workflow file and in an MCP call.
    pub fn key(self) -> &'static str {
        match self {
            Metric2d::Ncc => "ncc",
            Metric2d::GradientCorrelation => "gc",
            Metric2d::GradientDifference => "gd",
            Metric2d::PatternIntensity => "pi",
            Metric2d::LocalNcc => "lncc",
            Metric2d::MutualInformation => "mi",
        }
    }

    pub fn from_key(s: &str) -> Option<Metric2d> {
        let s = s.trim().to_ascii_lowercase();
        Metric2d::ALL
            .into_iter()
            .find(|m| m.key() == s || m.tag().eq_ignore_ascii_case(&s))
    }

    /// The similarity of `image` and `drr` (same dimensions) over the
    /// pixels `mask` lets through, larger is better. `radius` is the window
    /// half-width of [`Metric2d::LocalNcc`].
    pub fn similarity(
        self,
        image: &[f32],
        drr: &[f32],
        dims: [usize; 2],
        mask: Option<&[u8]>,
        radius: usize,
    ) -> f64 {
        debug_assert_eq!(image.len(), dims[0] * dims[1]);
        debug_assert_eq!(drr.len(), image.len());
        match self {
            Metric2d::Ncc => ncc(image, drr, mask),
            Metric2d::GradientCorrelation => gradient_correlation(image, drr, dims, mask),
            Metric2d::GradientDifference => gradient_difference(image, drr, dims, mask),
            Metric2d::PatternIntensity => pattern_intensity(image, drr, dims, mask),
            Metric2d::LocalNcc => local_ncc(image, drr, dims, mask, radius.max(1)),
            Metric2d::MutualInformation => mutual_information(image, drr, mask),
        }
    }
}

#[inline]
fn counts(mask: Option<&[u8]>, i: usize) -> bool {
    mask.is_none_or(|m| m[i] != 0)
}

/// Normalised cross-correlation of two images over a mask; 0 when either
/// is flat there.
pub fn ncc(a: &[f32], b: &[f32], mask: Option<&[u8]>) -> f64 {
    let (mut n, mut sa, mut sb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        if counts(mask, i) {
            n += 1.0;
            sa += a[i] as f64;
            sb += b[i] as f64;
        }
    }
    if n < 2.0 {
        return 0.0;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut cab, mut caa, mut cbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        if counts(mask, i) {
            let (da, db) = (a[i] as f64 - ma, b[i] as f64 - mb);
            cab += da * db;
            caa += da * da;
            cbb += db * db;
        }
    }
    let den = (caa * cbb).sqrt();
    if den <= 1e-300 || caa <= 1e-12 * n || cbb <= 1e-12 * n {
        0.0
    } else {
        cab / den
    }
}

/// Horizontal and vertical Sobel gradients, and the mask of the pixels they
/// are defined at (interior pixels whose 3 × 3 neighbourhood the mask lets
/// through entirely).
fn sobel(a: &[f32], dims: [usize; 2]) -> (Vec<f32>, Vec<f32>) {
    let [w, h] = dims;
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    if w < 3 || h < 3 {
        return (gx, gy);
    }
    for j in 1..h - 1 {
        for i in 1..w - 1 {
            let p = |di: isize, dj: isize| {
                a[(j as isize + dj) as usize * w + (i as isize + di) as usize]
            };
            gx[j * w + i] =
                (p(1, -1) + 2.0 * p(1, 0) + p(1, 1)) - (p(-1, -1) + 2.0 * p(-1, 0) + p(-1, 1));
            gy[j * w + i] =
                (p(-1, 1) + 2.0 * p(0, 1) + p(1, 1)) - (p(-1, -1) + 2.0 * p(0, -1) + p(1, -1));
        }
    }
    (gx, gy)
}

fn gradient_mask(dims: [usize; 2], mask: Option<&[u8]>) -> Mask2d {
    let [w, h] = dims;
    let mut m = vec![0u8; w * h];
    if w < 3 || h < 3 {
        return m;
    }
    for j in 1..h - 1 {
        for i in 1..w - 1 {
            let ok = match mask {
                None => true,
                Some(mk) => {
                    (0..3).all(|dj| (0..3).all(|di| mk[(j + dj - 1) * w + i + di - 1] != 0))
                }
            };
            m[j * w + i] = ok as u8;
        }
    }
    m
}

fn gradient_correlation(a: &[f32], b: &[f32], dims: [usize; 2], mask: Option<&[u8]>) -> f64 {
    let gm = gradient_mask(dims, mask);
    let (ax, ay) = sobel(a, dims);
    let (bx, by) = sobel(b, dims);
    0.5 * (ncc(&ax, &bx, Some(&gm)) + ncc(&ay, &by, Some(&gm)))
}

fn gradient_difference(a: &[f32], b: &[f32], dims: [usize; 2], mask: Option<&[u8]>) -> f64 {
    let gm = gradient_mask(dims, mask);
    let (ax, ay) = sobel(a, dims);
    let (bx, by) = sobel(b, dims);
    let (mut n, mut sab, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sx, mut sxx, mut sy, mut syy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..gm.len() {
        if gm[i] != 0 {
            let (x, y, u, v) = (ax[i] as f64, ay[i] as f64, bx[i] as f64, by[i] as f64);
            n += 1.0;
            sab += x * u + y * v;
            sbb += u * u + v * v;
            sx += x;
            sxx += x * x;
            sy += y;
            syy += y * y;
        }
    }
    if n < 2.0 {
        return 0.0;
    }
    let s = if sbb > 1e-300 { sab / sbb } else { 0.0 };
    let var_x = (sxx / n - (sx / n).powi(2)).max(1e-12);
    let var_y = (syy / n - (sy / n).powi(2)).max(1e-12);
    let mut acc = 0.0f64;
    for i in 0..gm.len() {
        if gm[i] != 0 {
            let dx = ax[i] as f64 - s * bx[i] as f64;
            let dy = ay[i] as f64 - s * by[i] as f64;
            acc += var_x / (var_x + dx * dx) + var_y / (var_y + dy * dy);
        }
    }
    acc / (2.0 * n)
}

/// `a` scaled to 0..255 over the mask.
fn scaled_255(a: &[f32], mask: Option<&[u8]>) -> Vec<f64> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (i, &v) in a.iter().enumerate() {
        if counts(mask, i) {
            lo = lo.min(v as f64);
            hi = hi.max(v as f64);
        }
    }
    let span = if hi > lo { hi - lo } else { 1.0 };
    a.iter().map(|&v| (v as f64 - lo) / span * 255.0).collect()
}

fn pattern_intensity(a: &[f32], b: &[f32], dims: [usize; 2], mask: Option<&[u8]>) -> f64 {
    const R: isize = 3;
    const SIGMA2: f64 = 100.0;
    let [w, h] = dims;
    let a = scaled_255(a, mask);
    let b = scaled_255(b, mask);
    // The least-squares scale of the rendering, so the difference image is
    // flat when the two agree up to a gain.
    let (mut n, mut sa, mut sb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        if counts(mask, i) {
            n += 1.0;
            sa += a[i];
            sb += b[i];
        }
    }
    if n < 2.0 {
        return 0.0;
    }
    let (ma, mb) = (sa / n, sb / n);
    let (mut cab, mut cbb) = (0.0f64, 0.0f64);
    for i in 0..a.len() {
        if counts(mask, i) {
            cab += (a[i] - ma) * (b[i] - mb);
            cbb += (b[i] - mb).powi(2);
        }
    }
    let s = if cbb > 1e-300 { cab / cbb } else { 0.0 };
    let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - s * y).collect();
    let offsets: Vec<(isize, isize)> = (-R..=R)
        .flat_map(|dj| (-R..=R).map(move |di| (di, dj)))
        .filter(|&(di, dj)| (di, dj) != (0, 0) && di * di + dj * dj <= R * R)
        .collect();
    let (mut acc, mut pairs) = (0.0f64, 0.0f64);
    for j in 0..h as isize {
        for i in 0..w as isize {
            let p = j as usize * w + i as usize;
            if !counts(mask, p) {
                continue;
            }
            for &(di, dj) in &offsets {
                let (x, y) = (i + di, j + dj);
                if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
                    continue;
                }
                let q = y as usize * w + x as usize;
                if !counts(mask, q) {
                    continue;
                }
                let e = d[p] - d[q];
                acc += SIGMA2 / (SIGMA2 + e * e);
                pairs += 1.0;
            }
        }
    }
    if pairs > 0.0 {
        acc / pairs
    } else {
        0.0
    }
}

/// Summed-area table of `f(i)` over the mask, `(w + 1) × (h + 1)`.
fn integral(dims: [usize; 2], f: impl Fn(usize) -> f64) -> Vec<f64> {
    let [w, h] = dims;
    let mut s = vec![0.0f64; (w + 1) * (h + 1)];
    for j in 0..h {
        let mut row = 0.0;
        for i in 0..w {
            row += f(j * w + i);
            s[(j + 1) * (w + 1) + i + 1] = s[j * (w + 1) + i + 1] + row;
        }
    }
    s
}

fn local_ncc(a: &[f32], b: &[f32], dims: [usize; 2], mask: Option<&[u8]>, r: usize) -> f64 {
    let [w, h] = dims;
    let m = |i: usize| counts(mask, i) as u8 as f64;
    let sn = integral(dims, m);
    let sa = integral(dims, |i| m(i) * a[i] as f64);
    let sb = integral(dims, |i| m(i) * b[i] as f64);
    let saa = integral(dims, |i| m(i) * (a[i] as f64).powi(2));
    let sbb = integral(dims, |i| m(i) * (b[i] as f64).powi(2));
    let sab = integral(dims, |i| m(i) * a[i] as f64 * b[i] as f64);
    let window = ((2 * r + 1) * (2 * r + 1)) as f64;
    let box_sum = |s: &[f64], x0: usize, y0: usize, x1: usize, y1: usize| {
        s[y1 * (w + 1) + x1] - s[y0 * (w + 1) + x1] - s[y1 * (w + 1) + x0] + s[y0 * (w + 1) + x0]
    };
    let (mut acc, mut count) = (0.0f64, 0.0f64);
    for j in 0..h {
        for i in 0..w {
            if !counts(mask, j * w + i) {
                continue;
            }
            let (x0, y0) = (i.saturating_sub(r), j.saturating_sub(r));
            let (x1, y1) = ((i + r + 1).min(w), (j + r + 1).min(h));
            let n = box_sum(&sn, x0, y0, x1, y1);
            if n < 0.5 * window {
                continue;
            }
            let ma = box_sum(&sa, x0, y0, x1, y1) / n;
            let mb = box_sum(&sb, x0, y0, x1, y1) / n;
            let va = box_sum(&saa, x0, y0, x1, y1) / n - ma * ma;
            let vb = box_sum(&sbb, x0, y0, x1, y1) / n - mb * mb;
            let cab = box_sum(&sab, x0, y0, x1, y1) / n - ma * mb;
            // A flat window says nothing either way; it is left out rather
            // than counted as no correlation.
            let floor = 1e-9 * (ma * ma + mb * mb + 1e-12);
            if va <= floor || vb <= floor {
                continue;
            }
            acc += (cab / (va * vb).sqrt()).clamp(-1.0, 1.0);
            count += 1.0;
        }
    }
    if count > 0.0 {
        acc / count
    } else {
        0.0
    }
}

fn mutual_information(a: &[f32], b: &[f32], mask: Option<&[u8]>) -> f64 {
    const BINS: usize = 32;
    let range = |x: &[f32]| {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for (i, &v) in x.iter().enumerate() {
            if counts(mask, i) {
                lo = lo.min(v as f64);
                hi = hi.max(v as f64);
            }
        }
        (lo, if hi > lo { hi - lo } else { 1.0 })
    };
    let (alo, aspan) = range(a);
    let (blo, bspan) = range(b);
    if !alo.is_finite() || !blo.is_finite() {
        return 0.0;
    }
    // The rendering's bins leave two at each end for the window's support.
    let a_step = aspan / BINS as f64;
    let b_step = bspan / (BINS - 4) as f64;
    let mut joint = vec![0.0f64; BINS * BINS];
    let mut n = 0.0f64;
    for i in 0..a.len() {
        if !counts(mask, i) {
            continue;
        }
        let fa = (((a[i] as f64 - alo) / a_step) as usize).min(BINS - 1);
        let u = (b[i] as f64 - blo) / b_step + 2.0;
        let b0 = u.floor() as isize - 1;
        for d in 0..4 {
            let bin = b0 + d;
            if bin < 0 || bin >= BINS as isize {
                continue;
            }
            joint[fa * BINS + bin as usize] += beta3(u - bin as f64);
        }
        n += 1.0;
    }
    if n < 2.0 {
        return 0.0;
    }
    let total: f64 = joint.iter().sum();
    let mut pa = [0.0f64; BINS];
    let mut pb = [0.0f64; BINS];
    for fa in 0..BINS {
        for fb in 0..BINS {
            let p = joint[fa * BINS + fb] / total;
            pa[fa] += p;
            pb[fb] += p;
        }
    }
    let mut mi = 0.0f64;
    for fa in 0..BINS {
        for fb in 0..BINS {
            let p = joint[fa * BINS + fb] / total;
            if p > 1e-300 && pa[fa] > 0.0 && pb[fb] > 0.0 {
                mi += p * (p / (pa[fa] * pb[fb])).ln();
            }
        }
    }
    mi
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth, asymmetric test pattern: a ramp, a bright disc and a dark
    /// bar.
    fn pattern(dims: [usize; 2], shift: f64) -> Vec<f32> {
        let [w, h] = dims;
        let mut v = vec![0.0f32; w * h];
        for j in 0..h {
            for i in 0..w {
                let x = i as f64 - shift;
                let y = j as f64;
                let disc = ((x - 20.0).powi(2) + (y - 18.0).powi(2)).sqrt();
                let mut p = 0.02 * x + 0.01 * y;
                p += 2.0 / (1.0 + ((disc - 8.0) / 1.5).exp());
                p -= 1.0 / (1.0 + ((x - 40.0).abs() - 3.0).exp())
                    * (y > 10.0 && y < 40.0) as u8 as f64;
                v[j * w + i] = p as f32;
            }
        }
        v
    }

    #[test]
    fn every_measure_peaks_at_the_identical_image() {
        let dims = [56, 48];
        let a = pattern(dims, 0.0);
        let mask = border_mask(dims, 2);
        for m in Metric2d::ALL {
            let same = m.similarity(&a, &a, dims, Some(&mask), 4);
            for shift in [1.5, 4.0] {
                let b = pattern(dims, shift);
                let off = m.similarity(&a, &b, dims, Some(&mask), 4);
                assert!(
                    same > off,
                    "{}: {same} at the match, {off} {shift} px off",
                    m.tag()
                );
            }
        }
    }

    #[test]
    fn correlations_ignore_gain_and_offset() {
        let dims = [40, 40];
        let a = pattern(dims, 0.0);
        let b: Vec<f32> = a.iter().map(|v| 3.0 * v - 7.0).collect();
        for m in [
            Metric2d::Ncc,
            Metric2d::GradientCorrelation,
            Metric2d::LocalNcc,
        ] {
            let s = m.similarity(&a, &b, dims, None, 3);
            assert!((s - 1.0).abs() < 1e-6, "{}: {s}", m.tag());
        }
        let neg: Vec<f32> = a.iter().map(|v| -v).collect();
        assert!((ncc(&a, &neg, None) + 1.0).abs() < 1e-9);
        // Gradient difference and pattern intensity fit the gain.
        for m in [Metric2d::GradientDifference, Metric2d::PatternIntensity] {
            let s = m.similarity(&a, &b, dims, None, 3);
            assert!((s - 1.0).abs() < 1e-6, "{}: {s}", m.tag());
        }
    }

    #[test]
    fn mutual_information_is_the_entropy_of_a_copy_and_zero_for_noise() {
        let dims = [64, 64];
        let a = pattern(dims, 0.0);
        let self_mi = Metric2d::MutualInformation.similarity(&a, &a, dims, None, 1);
        assert!(self_mi > 1.0, "{self_mi}");
        // A deterministic hash as unrelated noise.
        let noise: Vec<f32> = (0..a.len())
            .map(|i| ((i as u64).wrapping_mul(2654435761) % 1000) as f32)
            .collect();
        let none = Metric2d::MutualInformation.similarity(&a, &noise, dims, None, 1);
        assert!(none < 0.25 * self_mi, "{none} vs {self_mi}");
    }

    #[test]
    fn a_mask_keeps_out_what_it_covers() {
        let dims = [40, 40];
        let a = pattern(dims, 0.0);
        let mut b = a.clone();
        // A bright object in one image only, inside the masked-out corner.
        for j in 0..10 {
            for i in 0..10 {
                b[j * 40 + i] += 50.0;
            }
        }
        let mut mask = vec![1u8; 1600];
        for j in 0..12 {
            for i in 0..12 {
                mask[j * 40 + i] = 0;
            }
        }
        assert!(ncc(&a, &b, None) < 0.9);
        assert!((ncc(&a, &b, Some(&mask)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn halving_matches_the_coarser_camera() {
        // A linear ramp in i halves to the ramp sampled where fine pixels
        // 2I and 2I + 1 meet (away from the clamped edges).
        let dims = [16, 8];
        let a: Vec<f32> = (0..128).map(|p| (p % 16) as f32).collect();
        let h = Image2d::new(a, dims).halved();
        assert_eq!(h.dims, [8, 4]);
        for ci in 1..7 {
            assert!((h.data[ci] - (2.0 * ci as f32 + 0.5)).abs() < 1e-6);
        }
        let (m, d) = halve_mask(&border_mask([16, 16], 4), [16, 16]);
        assert_eq!(d, [8, 8]);
        assert_eq!(m[0], 0);
        assert_eq!(m[4 * 8 + 4], 1);
        let p = Image2d::new(vec![1.0; 100], [10, 10]).pyramid(3);
        assert_eq!(
            p.iter().map(|i| i.dims).collect::<Vec<_>>(),
            vec![[10, 10], [5, 5], [3, 3]]
        );
    }

    #[test]
    fn transmission_becomes_line_integrals() {
        let mut t = Image2d::new(vec![1.0, (-1.0f32).exp(), (-2.0f32).exp(), 1.0], [2, 2]);
        t.transmission_to_line_integral();
        for (v, want) in t.data.iter().zip([0.0, 1.0, 2.0, 0.0]) {
            assert!((v - want).abs() < 1e-5);
        }
    }

    #[test]
    fn keys_read_back() {
        for m in Metric2d::ALL {
            assert_eq!(Metric2d::from_key(m.key()), Some(m));
            assert_eq!(Metric2d::from_key(m.tag()), Some(m));
        }
        assert_eq!(Metric2d::from_key("ssd"), None);
    }
}
