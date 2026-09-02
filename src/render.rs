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

/// Jet-like colormap (blue → cyan → green → yellow → red), t in [0, 1].
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

/// Build a translucent colorwash RGBA image from a resampled dose plane.
/// `reference` maps to colormap t = 1; pixels below `threshold_frac * reference`
/// are fully transparent.
pub fn dose_colorwash(
    dose_plane: &[f32],
    reference: f32,
    threshold_frac: f32,
    opacity: f32,
    out: &mut Vec<Color32>,
) {
    let reference = reference.max(1e-6);
    let thr = threshold_frac * reference;
    let a_max = (opacity.clamp(0.0, 1.0) * 255.0) as u8;
    out.clear();
    out.resize(dose_plane.len(), Color32::TRANSPARENT);
    out.par_iter_mut()
        .zip(dose_plane.par_iter())
        .for_each(|(o, &d)| {
            *o = if d < thr || d <= 0.0 {
                Color32::TRANSPARENT
            } else {
                let [r, g, b] = dose_colormap(d / reference);
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
pub fn roi_on_plane(vol: &Volume, roi: &Roi, plane: ViewPlane, slice: usize) -> RoiPlaneGraphics {
    let mut g = RoiPlaneGraphics::default();
    let nz = vol.dims[2] as f64;

    // Slice indices (rounded k) occupied by this ROI - used for end caps.
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
        // Map to voxel space once.
        let vox: Vec<[f64; 3]> = c.points.iter().map(|&p| vol.patient_to_voxel(p)).collect();

        if c.geometric_type == "POINT" || vox.len() == 1 {
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
                let mean_k = vox.iter().map(|v| v[2]).sum::<f64>() / vox.len() as f64;
                if (mean_k - slice as f64).abs() <= 0.5 {
                    g.polylines
                        .push(vox.iter().map(|v| [v[0] as f32, v[1] as f32]).collect());
                }
            }
            ViewPlane::Sagittal | ViewPlane::Coronal => {
                // Contours are planar at ~constant k.
                let kc = vox.iter().map(|v| v[2]).sum::<f64>() / vox.len() as f64;
                let y = ((nz - 1.0) - kc) as f32; // display row
                let plane_coord = slice as f64;
                // Axis crossed by the cutting plane and in-plane axis.
                let (cut_axis, run_axis) = match plane {
                    ViewPlane::Sagittal => (0usize, 1usize), // cut i, runs along j
                    _ => (1usize, 0usize),                   // cut j, runs along i
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
