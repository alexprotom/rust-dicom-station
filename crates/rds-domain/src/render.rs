//! CPU rendering primitives: window/level grayscale conversion, dose plane
//! resampling + colorwash, marching-squares isodose lines and contour/plane
//! intersection. All functions operate in display-pixel space of a view plane
//! (see [`Volume::extract_slice`] for orientation conventions).

use egui::Color32;
use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::rtdose::DoseGrid;
use crate::rtstruct::Roi;
use crate::volume::{ViewPlane, Volume};

// ---------------------------------------------------------------------------
// Grayscale window/level
// ---------------------------------------------------------------------------

/// Convert an i16 slice buffer to grayscale RGBA pixels with window/level.
/// Parallelized over rows of `row_len` pixels.
pub fn slice_to_gray(
    slice: &[i16],
    center: f32,
    width: f32,
    row_len: usize,
    out: &mut Vec<Color32>,
) {
    let w = width.max(1.0);
    let lo = center - w * 0.5;
    let scale = 255.0 / w;
    out.clear();
    out.resize(slice.len(), Color32::BLACK);
    let chunk = row_len.max(1);
    out.par_chunks_mut(chunk)
        .zip(slice.par_chunks(chunk))
        .for_each(|(dst, src)| {
            for (o, &v) in dst.iter_mut().zip(src) {
                let g = ((v as f32 - lo) * scale).clamp(0.0, 255.0) as u8;
                *o = Color32::from_gray(g);
            }
        });
}

// ---------------------------------------------------------------------------
// Dose resampling and colorwash
// ---------------------------------------------------------------------------

/// Sample a dose grid on the display-pixel lattice of a view plane.
/// Out-of-grid pixels are 0. Parallelized over rows.
pub fn sample_dose_plane(
    vol: &Volume,
    dose: &DoseGrid,
    plane: ViewPlane,
    slice: usize,
    out: &mut Vec<f32>,
) {
    let [w, h] = vol.plane_dims(plane);
    out.clear();
    out.resize(w * h, 0.0);
    // The display-pixel lattice maps affinely into patient space, so the
    // per-pixel coordinate is one vector addition rather than a full
    // index->patient transform.
    let at = |px: f64, py: f64| {
        let v = vol.plane_pixel_to_voxel(plane, slice, px, py);
        vol.voxel_to_patient(v[0], v[1], v[2])
    };
    let g00 = dose.grid_coords(at(0.0, 0.0));
    let gx = sub3(dose.grid_coords(at(1.0, 0.0)), g00);
    let gy = sub3(dose.grid_coords(at(0.0, 1.0)), g00);
    out.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        let f = py as f64;
        let mut g = [g00[0] + gy[0] * f, g00[1] + gy[1] * f, g00[2] + gy[2] * f];
        for o in row.iter_mut() {
            *o = dose.sample_uvw(g).unwrap_or(0.0);
            g = [g[0] + gx[0], g[1] + gx[1], g[2] + gx[2]];
        }
    });
}

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Jet-like colormap (blue → cyan → green → yellow → red), t in [0, 1]: the
/// dose wash's [`DoseColormap::Rainbow`], and what the deformation arrows
/// are coloured with.
#[inline]
pub fn dose_colormap(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.25 {
        (0.0, t / 0.25, 1.0)
    } else if t < 0.5 {
        (0.0, 1.0, 1.0 - (t - 0.25) / 0.25)
    } else if t < 0.75 {
        ((t - 0.5) / 0.25, 1.0, 0.0)
    } else {
        (1.0, 1.0 - (t - 0.75) / 0.25, 0.0)
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// The colour maps a dose wash can be drawn with.
///
/// `Rainbow` is the viewer's own (blue → cyan → green → yellow → red, full
/// saturation at both ends); `Jet` is MATLAB's, which starts and ends darker
/// (dark blue, dark red); `Turbo` is Google's improved rainbow (Mikhailov's
/// polynomial approximation); `Viridis` is matplotlib's perceptually
/// uniform map (linear between nine published samples). `Isodose` takes
/// the colours of the isodose levels themselves, each at its percentage -
/// the planning-system look where the wash and the lines agree; the caller
/// hands the levels over ([`WashScale::new`], [`isodose_rgb`]), since they
/// are not the map's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DoseColormap {
    #[default]
    Rainbow,
    Jet,
    Turbo,
    Viridis,
    Isodose,
}

impl DoseColormap {
    pub const ALL: [DoseColormap; 5] = [
        DoseColormap::Rainbow,
        DoseColormap::Jet,
        DoseColormap::Turbo,
        DoseColormap::Viridis,
        DoseColormap::Isodose,
    ];

    /// The word the settings file writes.
    pub fn key(self) -> &'static str {
        match self {
            DoseColormap::Rainbow => "rainbow",
            DoseColormap::Jet => "jet",
            DoseColormap::Turbo => "turbo",
            DoseColormap::Viridis => "viridis",
            DoseColormap::Isodose => "isodose",
        }
    }

    pub fn from_key(key: &str) -> Option<DoseColormap> {
        let key = key.trim();
        DoseColormap::ALL
            .into_iter()
            .find(|m| m.key().eq_ignore_ascii_case(key))
    }

    pub fn label(self) -> &'static str {
        match self {
            DoseColormap::Rainbow => "Rainbow",
            DoseColormap::Jet => "Jet",
            DoseColormap::Turbo => "Turbo",
            DoseColormap::Viridis => "Viridis",
            DoseColormap::Isodose => "Isodose level colours",
        }
    }

    /// The colour at `t` in [0, 1] (clamped). `Isodose` has no colours of
    /// its own and answers with the rainbow's.
    pub fn rgb(self, t: f32) -> [u8; 3] {
        let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
        let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        match self {
            DoseColormap::Rainbow | DoseColormap::Isodose => dose_colormap(t),
            DoseColormap::Jet => {
                let f = |c: f32| to8(1.5 - (4.0 * t - c).abs());
                [f(3.0), f(2.0), f(1.0)]
            }
            DoseColormap::Turbo => {
                // The published coefficients, worked in f64 as given.
                let t = t as f64;
                let poly = |c: [f64; 6]| {
                    let v = c[0] + t * (c[1] + t * (c[2] + t * (c[3] + t * (c[4] + t * c[5]))));
                    to8(v as f32)
                };
                [
                    poly([
                        0.135_721_38,
                        4.615_392_60,
                        -42.660_322_58,
                        132.131_082_34,
                        -152.942_393_96,
                        59.286_379_43,
                    ]),
                    poly([
                        0.091_402_61,
                        2.194_188_39,
                        4.842_966_58,
                        -14.185_033_33,
                        4.277_298_57,
                        2.829_566_04,
                    ]),
                    poly([
                        0.106_673_30,
                        12.641_946_08,
                        -60.582_048_36,
                        110.362_767_71,
                        -89.903_109_12,
                        27.348_249_73,
                    ]),
                ]
            }
            DoseColormap::Viridis => {
                const V: [[f32; 3]; 9] = [
                    [68.0, 1.0, 84.0],
                    [71.0, 44.0, 122.0],
                    [59.0, 82.0, 139.0],
                    [44.0, 114.0, 142.0],
                    [33.0, 145.0, 140.0],
                    [40.0, 174.0, 128.0],
                    [94.0, 201.0, 98.0],
                    [173.0, 220.0, 48.0],
                    [253.0, 231.0, 37.0],
                ];
                let x = t * 8.0;
                let i = (x.floor() as usize).min(7);
                let f = x - i as f32;
                let mix = |c: usize| (V[i][c] + (V[i + 1][c] - V[i][c]) * f).round() as u8;
                [mix(0), mix(1), mix(2)]
            }
        }
    }
}

/// The colour of a dose at `pct` % of the reference under the
/// [`DoseColormap::Isodose`] map: `levels` are (percentage, colour), in any
/// order. Stepped, a dose takes the colour of the highest level it reaches
/// (below the lowest, the lowest's colour); continuous, the colours are
/// interpolated between neighbouring levels and held beyond the ends.
pub fn isodose_rgb(levels: &[(f32, [u8; 3])], pct: f32, stepped: bool) -> [u8; 3] {
    let mut sorted: Vec<(f32, [u8; 3])> = levels.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    isodose_rgb_sorted(&sorted, pct, stepped)
}

fn isodose_rgb_sorted(sorted: &[(f32, [u8; 3])], pct: f32, stepped: bool) -> [u8; 3] {
    let (Some(first), Some(last)) = (sorted.first(), sorted.last()) else {
        return dose_colormap(pct / 100.0);
    };
    if pct <= first.0 {
        return first.1;
    }
    if pct >= last.0 {
        return last.1;
    }
    let i = sorted.partition_point(|l| l.0 <= pct);
    let (a, b) = (sorted[i - 1], sorted[i]);
    if stepped {
        return a.1;
    }
    let f = ((pct - a.0) / (b.0 - a.0).max(1e-6)).clamp(0.0, 1.0);
    let m = |c: usize| (a.1[c] as f32 + (b.1[c] as f32 - a.1[c] as f32) * f).round() as u8;
    [m(0), m(1), m(2)]
}

/// Entries of a [`WashScale`]'s table: fine enough that a stepped band's
/// edge sits within a thousandth of the range of where it should.
pub const WASH_LUT_LEN: usize = 1024;

/// The colour scale of one dose wash, in absolute dose: `lo` takes the
/// first entry of `lut`, `hi` the last (doses beyond either end keep the
/// end's colour), and nothing under `threshold` is drawn. The table is
/// worked out once per change of the settings ([`WashScale::new`]), so a
/// pixel costs one division and one lookup whatever the map.
#[derive(Clone, Debug, PartialEq)]
pub struct WashScale {
    pub lut: Vec<[u8; 3]>,
    pub lo: f32,
    pub hi: f32,
    pub threshold: f32,
}

/// What a [`WashScale`] is made of, besides the doses it spans.
#[derive(Clone, Copy, Debug)]
pub struct WashStyle<'a> {
    pub map: DoseColormap,
    /// Bands of the scale when `stepped` (at least two).
    pub steps: u32,
    pub stepped: bool,
    /// The 100 % the isodose levels are of.
    pub reference: f32,
    /// The isodose levels (percentage, colour), for [`DoseColormap::Isodose`].
    pub levels: &'a [(f32, [u8; 3])],
}

impl WashScale {
    /// The scale from `lo` to `hi` (absolute dose; swapped if given the
    /// wrong way round): `style.steps` bands when stepped (each band one
    /// colour, the first the map's start, the last its end), the map
    /// continuous otherwise. [`DoseColormap::Isodose`] bands at the levels
    /// themselves rather than at `steps`.
    pub fn new(lo: f32, hi: f32, threshold: f32, style: WashStyle<'_>) -> WashScale {
        let (lo, hi) = if hi < lo { (hi, lo) } else { (lo, hi) };
        let hi = if hi - lo < 1e-6 { lo + 1e-6 } else { hi };
        let n = style.steps.max(2) as f32;
        let reference = style.reference.max(1e-6);
        let mut sorted: Vec<(f32, [u8; 3])> = style.levels.to_vec();
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let lut = (0..WASH_LUT_LEN)
            .map(|i| {
                let t = i as f32 / (WASH_LUT_LEN - 1) as f32;
                match style.map {
                    DoseColormap::Isodose => {
                        let pct = (lo + t * (hi - lo)) / reference * 100.0;
                        isodose_rgb_sorted(&sorted, pct, style.stepped)
                    }
                    map if style.stepped => {
                        let band = (t * n).floor().min(n - 1.0);
                        map.rgb(band / (n - 1.0))
                    }
                    map => map.rgb(t),
                }
            })
            .collect();
        WashScale {
            lut,
            lo,
            hi,
            threshold,
        }
    }

    /// The colour of `dose` on this scale, the threshold aside.
    #[inline]
    pub fn rgb(&self, dose: f32) -> [u8; 3] {
        let t = ((dose - self.lo) / (self.hi - self.lo)).clamp(0.0, 1.0);
        let t = if t.is_nan() { 0.0 } else { t };
        self.lut[(t * (self.lut.len() - 1) as f32).round() as usize]
    }

    /// A hash of everything the scale paints with, for the keys of caches
    /// built from it.
    pub fn key(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(0x100_0000_01b3);
        };
        mix(self.lo.to_bits() as u64);
        mix(self.hi.to_bits() as u64);
        mix(self.threshold.to_bits() as u64);
        for c in &self.lut {
            mix(((c[0] as u64) << 16) | ((c[1] as u64) << 8) | c[2] as u64);
        }
        h
    }
}

/// Build a translucent colorwash RGBA image from a resampled dose plane on
/// `scale`; pixels under its threshold, and those without dose, are fully
/// transparent.
pub fn dose_colorwash(dose_plane: &[f32], scale: &WashScale, opacity: f32, out: &mut Vec<Color32>) {
    let a_max = (opacity.clamp(0.0, 1.0) * 255.0) as u8;
    out.clear();
    out.resize(dose_plane.len(), Color32::TRANSPARENT);
    out.par_iter_mut()
        .zip(dose_plane.par_iter())
        .for_each(|(o, &d)| {
            *o = if d < scale.threshold || d <= 0.0 {
                Color32::TRANSPARENT
            } else {
                let [r, g, b] = scale.rgb(d);
                Color32::from_rgba_unmultiplied(r, g, b, a_max)
            };
        });
}

// ---------------------------------------------------------------------------
// Isodose lines (marching squares)
// ---------------------------------------------------------------------------

pub type Segment = ([f32; 2], [f32; 2]);

/// Extract iso-contour line segments of `field` (w×h, pixel-center lattice)
/// at `level` using marching squares with linear interpolation.
pub fn marching_squares(field: &[f32], w: usize, h: usize, level: f32) -> Vec<Segment> {
    let mut segs = Vec::new();
    if w < 2 || h < 2 {
        return segs;
    }
    let at = |x: usize, y: usize| field[y * w + x];
    #[inline]
    fn lerp_t(a: f32, b: f32, level: f32) -> f32 {
        let d = b - a;
        if d.abs() < 1e-12 {
            0.5
        } else {
            ((level - a) / d).clamp(0.0, 1.0)
        }
    }
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let v00 = at(x, y);
            let v10 = at(x + 1, y);
            let v01 = at(x, y + 1);
            let v11 = at(x + 1, y + 1);
            let mut case = 0u8;
            if v00 >= level {
                case |= 1;
            }
            if v10 >= level {
                case |= 2;
            }
            if v11 >= level {
                case |= 4;
            }
            if v01 >= level {
                case |= 8;
            }
            if case == 0 || case == 15 {
                continue;
            }
            let xf = x as f32;
            let yf = y as f32;
            // Edge crossing points.
            let top = [xf + lerp_t(v00, v10, level), yf];
            let bottom = [xf + lerp_t(v01, v11, level), yf + 1.0];
            let left = [xf, yf + lerp_t(v00, v01, level)];
            let right = [xf + 1.0, yf + lerp_t(v10, v11, level)];
            match case {
                1 | 14 => segs.push((left, top)),
                2 | 13 => segs.push((top, right)),
                4 | 11 => segs.push((right, bottom)),
                8 | 7 => segs.push((bottom, left)),
                3 | 12 => segs.push((left, right)),
                6 | 9 => segs.push((top, bottom)),
                5 | 10 => {
                    // Ambiguous saddle: disambiguate with center average.
                    let center = 0.25 * (v00 + v10 + v01 + v11);
                    let flip = (center >= level) == (case == 5);
                    if flip {
                        segs.push((left, top));
                        segs.push((right, bottom));
                    } else {
                        segs.push((top, right));
                        segs.push((bottom, left));
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    segs
}

// ---------------------------------------------------------------------------
// RTSTRUCT contour → view plane intersection
// ---------------------------------------------------------------------------

/// Contour geometry of one ROI on one displayed plane, in display-pixel coords.
#[derive(Default)]
pub struct RoiPlaneGraphics {
    /// Closed polylines (in-plane contours, axial view).
    pub polylines: Vec<Vec<[f32; 2]>>,
    /// Short boundary segments (cross-plane silhouette, sagittal/coronal).
    pub segments: Vec<Segment>,
    /// Point markers (POINT contours, e.g. reference points).
    pub points: Vec<[f32; 2]>,
}

/// Compute the drawable geometry of a ROI on the given plane/slice.
///
/// Axial: contours whose plane lies within half a slice of `slice` are drawn
/// as closed polylines. Sagittal/coronal: each axial contour is intersected
/// with the viewing plane; interior runs are found by even-odd pairing of the
/// crossings and drawn as short vertical boundary ticks (plus horizontal caps
/// where the structure begins/ends along the stack).
///
/// For many slices of one ROI, [`PreparedRoi`] does the mapping into voxel
/// space once; this is `PreparedRoi::new(..).on_plane(..)`.
pub fn roi_on_plane(vol: &Volume, roi: &Roi, plane: ViewPlane, slice: usize) -> RoiPlaneGraphics {
    PreparedRoi::new(vol, roi).on_plane(vol, plane, slice)
}

/// A ROI mapped into a volume's voxel space once, so that drawing it on a
/// slice costs the contours that touch that slice rather than every point of
/// the structure. Scrolling, dragging the crosshair or playing a 4D cine
/// used to map a whole structure set - hundreds of thousands of points -
/// per view per step.
///
/// [`PreparedRoi::on_plane`] gives exactly what [`roi_on_plane`] gave: the
/// same expressions, in the same order; it only skips the contours that
/// provably draw nothing on the requested slice.
pub struct PreparedRoi {
    contours: Vec<PreparedContour>,
    /// Slice indices (rounded k of each contour's first point) - the end caps.
    k_occupied: Vec<i64>,
}

struct PreparedContour {
    /// The points in voxel coordinates.
    vox: Vec<[f64; 3]>,
    /// A POINT (or a one-vertex contour): drawn as a marker.
    point: bool,
    /// Mean k of the points: the slice a planar contour lies on.
    mean_k: f64,
    /// Extent of the points along i and j, for the sagittal / coronal cut.
    lo: [f64; 2],
    hi: [f64; 2],
}

impl PreparedRoi {
    pub fn new(vol: &Volume, roi: &Roi) -> Self {
        let mut k_occupied: Vec<i64> = Vec::new();
        let mut contours = Vec::with_capacity(roi.contours.len());
        for c in &roi.contours {
            if let Some(p0) = c.points.first() {
                k_occupied.push(vol.patient_to_voxel(*p0)[2].round() as i64);
            }
            if c.points.is_empty() {
                continue;
            }
            let vox: Vec<[f64; 3]> = c.points.iter().map(|&p| vol.patient_to_voxel(p)).collect();
            let mean_k = vox.iter().map(|v| v[2]).sum::<f64>() / vox.len() as f64;
            let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
            for v in &vox {
                for a in 0..2 {
                    lo[a] = lo[a].min(v[a]);
                    hi[a] = hi[a].max(v[a]);
                }
            }
            contours.push(PreparedContour {
                point: c.geometric_type == "POINT" || vox.len() == 1,
                vox,
                mean_k,
                lo,
                hi,
            });
        }
        k_occupied.sort_unstable();
        k_occupied.dedup();
        PreparedRoi {
            contours,
            k_occupied,
        }
    }

    /// Memory the prepared points hold, for a cache to count.
    pub fn bytes(&self) -> usize {
        self.contours
            .iter()
            .map(|c| {
                c.vox.len() * std::mem::size_of::<[f64; 3]>()
                    + std::mem::size_of::<PreparedContour>()
            })
            .sum::<usize>()
            + self.k_occupied.len() * std::mem::size_of::<i64>()
    }

    /// Whether the point `v` (voxel coordinates of the volume the ROI was
    /// prepared on) lies inside the ROI as the views draw it: inside the
    /// contours of the slice nearest to it - those the axial view draws on
    /// slice `round(k)` - by the even-odd rule over all of them together, so
    /// that a hole drawn as a second contour inside the first is a hole, as
    /// [`crate::segmentation::rasterize_roi`] fills it. In plane the point
    /// keeps its fraction: what is asked is where the crosshair is, not the
    /// centre of its voxel. A POINT encloses nothing; an open contour is
    /// taken closed, as it is drawn.
    ///
    /// The mapping into voxels went through the volume's direction vectors
    /// ([`Volume::patient_to_voxel`]), so an oblique or tilted volume needs
    /// nothing else here. Costs the contours of one slice.
    pub fn contains(&self, v: [f64; 3]) -> bool {
        let k = v[2].round();
        let mut inside = false;
        for c in &self.contours {
            if c.point || c.vox.len() < 3 || (c.mean_k - k).abs() > 0.5 {
                continue;
            }
            // A ray to +i from a point outside the contour's box crosses it
            // an even number of times: the parity would not move.
            if v[0] < c.lo[0] || v[0] > c.hi[0] || v[1] < c.lo[1] || v[1] > c.hi[1] {
                continue;
            }
            let n = c.vox.len();
            let mut b = c.vox[n - 1];
            for &a in &c.vox {
                // Half-open in j, so a vertex on the ray counts once.
                if (a[1] > v[1]) != (b[1] > v[1]) {
                    let x = a[0] + (v[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
                    if v[0] < x {
                        inside = !inside;
                    }
                }
                b = a;
            }
        }
        inside
    }

    /// The ROI on one slice of one plane (see [`roi_on_plane`]).
    pub fn on_plane(&self, vol: &Volume, plane: ViewPlane, slice: usize) -> RoiPlaneGraphics {
        let mut g = RoiPlaneGraphics::default();
        let nz = vol.dims[2] as f64;
        // One buffer for the crossings of every contour on this slice.
        let mut crossings: Vec<f64> = Vec::new();
        for c in &self.contours {
            let vox = &c.vox;
            if c.point {
                let v = vox[0];
                let pp = vol.voxel_to_plane_pixel(plane, v);
                let on_plane = (pp[2] - slice as f64).abs() < 0.75;
                if on_plane {
                    g.points.push([pp[0] as f32, pp[1] as f32]);
                }
                continue;
            }

            match plane {
                ViewPlane::Axial => {
                    if (c.mean_k - slice as f64).abs() <= 0.5 {
                        g.polylines
                            .push(vox.iter().map(|v| [v[0] as f32, v[1] as f32]).collect());
                    }
                }
                ViewPlane::Sagittal | ViewPlane::Coronal => {
                    // Contours are planar at ~constant k.
                    let kc = c.mean_k;
                    let y = ((nz - 1.0) - kc) as f32; // display row
                    let plane_coord = slice as f64;
                    // Axis crossed by the cutting plane and in-plane axis.
                    let (cut_axis, run_axis) = match plane {
                        ViewPlane::Sagittal => (0usize, 1usize), // cut i, runs along j
                        _ => (1usize, 0usize),                   // cut j, runs along i
                    };
                    // An edge crosses the plane only between a point at or
                    // before it and one beyond it; a contour entirely on one
                    // side has no crossing, and would be skipped below.
                    if !(c.lo[cut_axis] <= plane_coord && c.hi[cut_axis] > plane_coord) {
                        continue;
                    }
                    crossings.clear();
                    let n = vox.len();
                    for e in 0..n {
                        let a = vox[e];
                        let b = vox[(e + 1) % n];
                        let da = a[cut_axis] - plane_coord;
                        let db = b[cut_axis] - plane_coord;
                        if (da <= 0.0 && db > 0.0) || (db <= 0.0 && da > 0.0) {
                            let t = da / (da - db);
                            crossings.push(a[run_axis] + t * (b[run_axis] - a[run_axis]));
                        }
                    }
                    if crossings.len() < 2 {
                        continue;
                    }
                    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let k_int = kc.round() as i64;
                    let cap_above = self.k_occupied.binary_search(&(k_int + 1)).is_err();
                    let cap_below = self.k_occupied.binary_search(&(k_int - 1)).is_err();
                    for pair in crossings.as_chunks::<2>().0 {
                        let (xa, xb) = (pair[0] as f32, pair[1] as f32);
                        // Vertical boundary ticks spanning one slice row.
                        g.segments.push(([xa, y - 0.5], [xa, y + 0.5]));
                        g.segments.push(([xb, y - 0.5], [xb, y + 0.5]));
                        // Horizontal caps where the structure starts/ends in k.
                        if cap_above {
                            g.segments.push(([xa, y - 0.5], [xb, y - 0.5]));
                        }
                        if cap_below {
                            g.segments.push(([xa, y + 0.5], [xb, y + 0.5]));
                        }
                    }
                }
            }
        }
        g
    }
}

/// Map a patient-space point onto display-pixel coordinates of a plane,
/// returning (x, y, distance-from-plane-in-slices).
pub fn patient_to_plane_pixel(
    vol: &Volume,
    plane: ViewPlane,
    slice: usize,
    p: Vec3,
) -> ([f32; 2], f64) {
    let v = vol.patient_to_voxel(p);
    let pp = vol.voxel_to_plane_pixel(plane, v);
    ([pp[0] as f32, pp[1] as f32], pp[2] - slice as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtstruct::Contour;

    /// `roi_on_plane` as it was before [`PreparedRoi`]: every point mapped on
    /// every call, every contour intersected.
    fn roi_on_plane_direct(
        vol: &Volume,
        roi: &Roi,
        plane: ViewPlane,
        slice: usize,
    ) -> RoiPlaneGraphics {
        let mut g = RoiPlaneGraphics::default();
        let nz = vol.dims[2] as f64;
        let mut k_occupied: Vec<i64> = Vec::new();
        if plane != ViewPlane::Axial {
            for c in &roi.contours {
                if let Some(p0) = c.points.first() {
                    let v = vol.patient_to_voxel(*p0);
                    k_occupied.push(v[2].round() as i64);
                }
            }
            k_occupied.sort_unstable();
            k_occupied.dedup();
        }
        for c in &roi.contours {
            if c.points.is_empty() {
                continue;
            }
            let vox: Vec<[f64; 3]> = c.points.iter().map(|&p| vol.patient_to_voxel(p)).collect();
            if c.geometric_type == "POINT" || vox.len() == 1 {
                let v = vox[0];
                let pp = vol.voxel_to_plane_pixel(plane, v);
                if (pp[2] - slice as f64).abs() < 0.75 {
                    g.points.push([pp[0] as f32, pp[1] as f32]);
                }
                continue;
            }
            match plane {
                ViewPlane::Axial => {
                    let mean_k = vox.iter().map(|v| v[2]).sum::<f64>() / vox.len() as f64;
                    if (mean_k - slice as f64).abs() <= 0.5 {
                        g.polylines
                            .push(vox.iter().map(|v| [v[0] as f32, v[1] as f32]).collect());
                    }
                }
                ViewPlane::Sagittal | ViewPlane::Coronal => {
                    let kc = vox.iter().map(|v| v[2]).sum::<f64>() / vox.len() as f64;
                    let y = ((nz - 1.0) - kc) as f32;
                    let plane_coord = slice as f64;
                    let (cut_axis, run_axis) = match plane {
                        ViewPlane::Sagittal => (0usize, 1usize),
                        _ => (1usize, 0usize),
                    };
                    let mut crossings: Vec<f64> = Vec::new();
                    let n = vox.len();
                    for e in 0..n {
                        let a = vox[e];
                        let b = vox[(e + 1) % n];
                        let da = a[cut_axis] - plane_coord;
                        let db = b[cut_axis] - plane_coord;
                        if (da <= 0.0 && db > 0.0) || (db <= 0.0 && da > 0.0) {
                            let t = da / (da - db);
                            crossings.push(a[run_axis] + t * (b[run_axis] - a[run_axis]));
                        }
                    }
                    if crossings.len() < 2 {
                        continue;
                    }
                    crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let k_int = kc.round() as i64;
                    let cap_above = !k_occupied.contains(&(k_int + 1));
                    let cap_below = !k_occupied.contains(&(k_int - 1));
                    for pair in crossings.as_chunks::<2>().0 {
                        let (xa, xb) = (pair[0] as f32, pair[1] as f32);
                        g.segments.push(([xa, y - 0.5], [xa, y + 0.5]));
                        g.segments.push(([xb, y - 0.5], [xb, y + 0.5]));
                        if cap_above {
                            g.segments.push(([xa, y - 0.5], [xb, y - 0.5]));
                        }
                        if cap_below {
                            g.segments.push(([xa, y + 0.5], [xb, y + 0.5]));
                        }
                    }
                }
            }
        }
        g
    }

    /// The crosshair's structures: inside a ring and outside its hole, on
    /// the slice nearest the point only, on a volume tilted out of the
    /// patient axes (the contours are given in LPS millimetres, as a file
    /// gives them).
    #[test]
    fn a_point_is_inside_what_the_slice_nearest_it_draws() {
        let dims = [30, 30, 10];
        // Rows and columns turned 30 degrees about an axis that is not the
        // patient's, the slices stacked along their cross product.
        let (s, c) = (30f64.to_radians().sin(), 30f64.to_radians().cos());
        let row = Vec3::new(c, s * 0.8, s * 0.6).normalized();
        let col = Vec3::new(-s, c * 0.8, c * 0.6).normalized();
        let normal = row.cross(col).normalized();
        let vol = Volume {
            data: vec![0; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [1.2, 0.8, 3.0],
            origin: Vec3::new(-7.0, 21.0, -40.0),
            row_dir: row,
            col_dir: col,
            normal,
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        };
        // A square ring on slice 4 (outer 5..20, hole 10..15), drawn in
        // patient space through the volume's own geometry.
        let square = |lo: f64, hi: f64, k: f64| Contour {
            points: [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]
                .iter()
                .map(|[i, j]| vol.voxel_to_patient(*i, *j, k))
                .collect(),
            geometric_type: "CLOSED_PLANAR".into(),
        };
        let roi = Roi {
            number: 1,
            name: "ring".into(),
            color: [1, 2, 3],
            roi_type: "ORGAN".into(),
            description: String::new(),
            contours: vec![
                square(5.0, 20.0, 4.0),
                square(10.0, 15.0, 4.0),
                Contour {
                    points: vec![vol.voxel_to_patient(7.0, 7.0, 6.0)],
                    geometric_type: "POINT".into(),
                },
            ],
        };
        let p = PreparedRoi::new(&vol, &roi);
        let at = |i: f64, j: f64, k: f64| {
            // Through patient space and back, as the crosshair goes.
            let q = vol.voxel_to_patient(i, j, k);
            p.contains(vol.patient_to_voxel(q))
        };
        assert!(at(7.0, 7.0, 4.0), "in the ring");
        assert!(at(7.0, 7.0, 4.4), "nearest slice is 4");
        assert!(
            !at(7.0, 7.0, 4.6),
            "nearest slice is 5: nothing drawn there"
        );
        assert!(!at(12.5, 12.5, 4.0), "in the hole");
        assert!(at(19.9, 12.5, 4.0), "a fraction inside the edge");
        assert!(!at(20.1, 12.5, 4.0), "a fraction outside it");
        assert!(!at(2.0, 12.5, 4.0), "left of the ring");
        assert!(!at(7.0, 7.0, 6.0), "a point of interest encloses nothing");
    }

    #[test]
    fn prepared_rois_draw_exactly_what_the_direct_mapping_drew() {
        let dims = [40, 36, 18];
        let vol = Volume {
            data: vec![0; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [1.1, 0.9, 2.5],
            origin: Vec3::new(-20.0, 13.0, 40.0),
            row_dir: Vec3::new(1.0, 0.02, 0.0).normalized(),
            col_dir: Vec3::new(-0.02, 1.0, 0.0).normalized(),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        };
        // Two lobes of different sizes on several slices (one contour with
        // a gap in k), plus a point of interest.
        let mut contours = Vec::new();
        for k in [3usize, 4, 5, 7, 8, 9, 10, 14] {
            for (ci, cj, r, n) in [(12.0, 15.0, 6.3, 23usize), (27.0, 20.0, 4.1, 17)] {
                let points = (0..n)
                    .map(|t| {
                        let a = t as f64 / n as f64 * std::f64::consts::TAU;
                        let v = [ci + r * a.cos(), cj + 0.8 * r * a.sin(), k as f64];
                        vol.voxel_to_patient(v[0], v[1], v[2])
                    })
                    .collect();
                contours.push(Contour {
                    points,
                    geometric_type: "CLOSED_PLANAR".into(),
                });
            }
        }
        contours.push(Contour {
            points: vec![vol.voxel_to_patient(20.0, 18.0, 6.0)],
            geometric_type: "POINT".into(),
        });
        let roi = Roi {
            number: 1,
            name: "lobes".into(),
            color: [1, 2, 3],
            roi_type: "ORGAN".into(),
            description: String::new(),
            contours,
        };
        let prepared = PreparedRoi::new(&vol, &roi);
        for (plane, n) in [
            (ViewPlane::Axial, dims[2]),
            (ViewPlane::Sagittal, dims[0]),
            (ViewPlane::Coronal, dims[1]),
        ] {
            for slice in 0..n {
                let a = prepared.on_plane(&vol, plane, slice);
                let b = roi_on_plane_direct(&vol, &roi, plane, slice);
                assert_eq!(a.polylines, b.polylines, "{plane:?} {slice}");
                assert_eq!(a.segments, b.segments, "{plane:?} {slice}");
                assert_eq!(a.points, b.points, "{plane:?} {slice}");
            }
        }
    }

    fn style(map: DoseColormap, steps: u32, stepped: bool) -> WashStyle<'static> {
        WashStyle {
            map,
            steps,
            stepped,
            reference: 50.0,
            levels: &[(100.0, [255, 0, 0]), (50.0, [0, 0, 255])],
        }
    }

    /// The maps start and end where their references do.
    #[test]
    fn colour_maps_end_where_they_should() {
        assert_eq!(DoseColormap::Rainbow.rgb(0.0), [0, 0, 255]);
        assert_eq!(DoseColormap::Rainbow.rgb(1.0), [255, 0, 0]);
        assert_eq!(DoseColormap::Jet.rgb(0.0), [0, 0, 128]);
        assert_eq!(DoseColormap::Jet.rgb(1.0), [128, 0, 0]);
        assert_eq!(DoseColormap::Viridis.rgb(0.0), [68, 1, 84]);
        assert_eq!(DoseColormap::Viridis.rgb(1.0), [253, 231, 37]);
        let t0 = DoseColormap::Turbo.rgb(0.1);
        let t1 = DoseColormap::Turbo.rgb(1.0);
        assert!(t0[2] > t0[0] && t1[0] > t1[2], "{t0:?} {t1:?}");
        for m in DoseColormap::ALL {
            assert_eq!(DoseColormap::from_key(m.key()), Some(m));
        }
    }

    /// Continuous, two doses close together get different colours; stepped,
    /// the same one inside a band, and the bands span the whole map.
    #[test]
    fn a_wash_scale_is_continuous_or_banded() {
        let c = WashScale::new(0.0, 50.0, 0.0, style(DoseColormap::Rainbow, 5, false));
        assert_ne!(c.rgb(20.0), c.rgb(21.0));
        assert_eq!(c.rgb(-5.0), [0, 0, 255]);
        assert_eq!(c.rgb(80.0), [255, 0, 0]);
        let s = WashScale::new(0.0, 50.0, 0.0, style(DoseColormap::Rainbow, 5, true));
        assert_eq!(s.rgb(20.5), s.rgb(29.5));
        assert_ne!(s.rgb(19.5), s.rgb(20.5));
        assert_eq!(s.rgb(0.0), [0, 0, 255]);
        assert_eq!(s.rgb(49.9), [255, 0, 0]);
        assert_ne!(c.key(), s.key());
    }

    /// The isodose map takes the levels' colours at their percentages of
    /// the reference.
    #[test]
    fn the_isodose_map_follows_the_levels() {
        let levels = [(100.0, [255, 0, 0]), (50.0, [0, 0, 255])];
        assert_eq!(isodose_rgb(&levels, 75.0, false), [128, 0, 128]);
        assert_eq!(isodose_rgb(&levels, 75.0, true), [0, 0, 255]);
        assert_eq!(isodose_rgb(&levels, 120.0, true), [255, 0, 0]);
        let s = WashScale::new(0.0, 60.0, 0.0, style(DoseColormap::Isodose, 5, true));
        // 50 Gy is the 100 % level here.
        assert_eq!(s.rgb(49.0), [0, 0, 255]);
        assert_eq!(s.rgb(51.0), [255, 0, 0]);
    }
}
