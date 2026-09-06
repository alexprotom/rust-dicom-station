//! Structures made out of something other than a hand: a grey-level window,
//! a shape, a dose level.
//!
//! These are the cheap generators every planning system has, and their value
//! is not the arithmetic but where the result lands. Each one produces a
//! voxel mask on the displayed lattice, which
//! [`crate::contours::Stack::from_mask`] turns into an ordinary editable
//! structure: a bone ROI comes out as contours a planner can correct with
//! the brush, not as a read-only overlay.

use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::rtdose::DoseGrid;
use crate::volume::{Grid, Volume};

/// The grey-level presets for a bone structure, in Hounsfield units: the
/// three windows a planning system offers by region. The upper bound is
/// open - bone has no ceiling, and a prosthesis is bone as far as a
/// threshold is concerned.
pub const BONE_PRESETS: [(&str, f32); 3] = [
    ("High (head and neck)", 250.0),
    ("Medium (thorax)", 200.0),
    ("Low (pelvis)", 150.0),
];

/// Voxels whose value lies in `[lo, hi]`, optionally only inside `limit`.
///
/// The limiting mask is the *limiting structure*: a threshold is a blunt
/// instrument, and restricting it to a structure that is already drawn is
/// what makes it usable (bone inside the body, contrast inside the liver).
pub fn threshold_mask(vol: &Volume, lo: f32, hi: f32, limit: Option<&[u8]>) -> Vec<u8> {
    let n = vol.dims[0] * vol.dims[1] * vol.dims[2];
    let mut out = vec![0u8; n];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        if let Some(m) = limit {
            if m.get(i).copied().unwrap_or(0) == 0 {
                return;
            }
        }
        let v = vol.data[i] as f32;
        if v >= lo && v <= hi {
            *o = 1;
        }
    });
    out
}

/// The basic shapes, aligned with the patient axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shape {
    Box,
    Cylinder,
    Sphere,
    Ellipsoid,
}

impl Shape {
    pub const ALL: [Shape; 4] = [Shape::Box, Shape::Cylinder, Shape::Sphere, Shape::Ellipsoid];

    pub fn label(self) -> &'static str {
        match self {
            Shape::Box => "Box",
            Shape::Cylinder => "Cylinder",
            Shape::Sphere => "Sphere",
            Shape::Ellipsoid => "Ellipsoid",
        }
    }

    /// Whether the three sizes are independent, or one radius does.
    pub fn is_uniform(self) -> bool {
        self == Shape::Sphere
    }
}

/// A shape centred on `centre` (patient coordinates, mm) with half-sizes
/// `half` along the patient x, y and z axes, rasterized onto `grid`.
///
/// The cylinder's axis is the patient z (superior-inferior), which is what
/// "aligned with the coordinate axes" means for the one shape where it is
/// not obvious.
pub fn shape_mask(shape: Shape, grid: &Grid, centre: Vec3, half: [f64; 3]) -> Vec<u8> {
    let [nx, ny, nz] = grid.dims;
    let mut out = vec![0u8; nx * ny * nz];
    let h = [half[0].max(1e-6), half[1].max(1e-6), half[2].max(1e-6)];
    // Only the voxels the shape can reach: its patient bounding box mapped
    // back onto the lattice, with a voxel of slack for the rounding.
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    for cx in [-1.0f64, 1.0] {
        for cy in [-1.0f64, 1.0] {
            for cz in [-1.0f64, 1.0] {
                let p = centre + Vec3::new(cx * h[0], cy * h[1], cz * h[2]);
                let v = grid.patient_to_voxel(p);
                for a in 0..3 {
                    let l = (v[a].floor() - 1.0).max(0.0) as usize;
                    let u = (v[a].ceil() + 1.0).max(0.0) as usize;
                    lo[a] = lo[a].min(l);
                    hi[a] = hi[a].max(u.min(grid.dims[a].saturating_sub(1)));
                }
            }
        }
    }
    if (0..3).any(|a| lo[a] > hi[a]) {
        return out;
    }
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let p = grid.voxel_to_patient(i as f64, j as f64, k as f64) - centre;
                let (dx, dy, dz) = (p.x / h[0], p.y / h[1], p.z / h[2]);
                let inside = match shape {
                    Shape::Box => dx.abs() <= 1.0 && dy.abs() <= 1.0 && dz.abs() <= 1.0,
                    Shape::Cylinder => dx * dx + dy * dy <= 1.0 && dz.abs() <= 1.0,
                    Shape::Sphere | Shape::Ellipsoid => dx * dx + dy * dy + dz * dz <= 1.0,
                };
                if inside {
                    out[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
    }
    out
}

/// The reconstructed field of view: the part of each slice that carries
/// data at all.
///
/// A CT reconstructed on a circle smaller than the image matrix pads the
/// corners with one constant value, and every threshold, body contour and
/// registration then has to know where the data stop. The padding value is
/// read off the corners rather than assumed (-1000 and -2000 are both in
/// use, and so is 0), the valid pixels of each slice are kept, holes are
/// filled and only the piece the centre of the image sits in survives.
///
/// `None` when the corners disagree, which is what a full-field image
/// looks like: there is no field-of-view structure to make, because the
/// field of view is the image.
pub fn fov_mask(vol: &Volume) -> Option<Vec<u8>> {
    let [nx, ny, nz] = vol.dims;
    if nx < 4 || ny < 4 || nz == 0 {
        return None;
    }
    let mid = nz / 2;
    let corners = [
        vol.index(0, 0, mid),
        vol.index(nx - 1, 0, mid),
        vol.index(0, ny - 1, mid),
        vol.index(nx - 1, ny - 1, mid),
    ];
    // Three of four corners agreeing is a padded reconstruction; anything
    // else is an image whose corners hold real tissue.
    let pad = corners
        .iter()
        .find(|&&c| corners.iter().filter(|&&o| o == c).count() >= 3)
        .copied()?;
    if vol.index(nx / 2, ny / 2, mid) == pad {
        // The middle is padding too: this is not a field of view, it is an
        // empty image.
        return None;
    }
    let mut mask = vec![0u8; nx * ny * nz];
    mask.par_chunks_mut(nx * ny)
        .enumerate()
        .for_each(|(k, sl)| {
            for j in 0..ny {
                for i in 0..nx {
                    if vol.index(i, j, k) != pad {
                        sl[j * nx + i] = 1;
                    }
                }
            }
        });
    crate::morphology::fill_holes_2d(&mut mask, vol.dims, 2);
    // Keep the piece the image centre is in: a couch rail sticking out of
    // the reconstruction circle is not part of the field of view.
    let comps = crate::morphology::components(&mask, vol.dims);
    let centre = mid * nx * ny + (ny / 2) * nx + nx / 2;
    let keep = comps
        .iter()
        .find(|c| c.voxels.contains(&(centre as u32)))
        .or_else(|| comps.iter().max_by_key(|c| c.len()))?;
    let mut out = vec![0u8; nx * ny * nz];
    for &v in &keep.voxels {
        out[v as usize] = 1;
    }
    Some(out)
}

/// Voxels of `grid` where the dose is at least `level` (in the dose object's
/// own units), sampled trilinearly - the same sampling the isodose lines are
/// drawn from, so the structure agrees with what is on the screen.
pub fn dose_mask(grid: &Grid, dose: &DoseGrid, level: f32) -> Vec<u8> {
    let [nx, ny, nz] = grid.dims;
    let mut out = vec![0u8; nx * ny * nz];
    out.par_chunks_mut(nx * ny)
        .enumerate()
        .for_each(|(k, slab)| {
            for j in 0..ny {
                // The map from lattice to dose grid is affine, so a row costs
                // one projection and then an addition per voxel.
                let p0 = grid.voxel_to_patient(0.0, j as f64, k as f64);
                let step = grid.voxel_to_patient(1.0, j as f64, k as f64) - p0;
                let c0 = dose.grid_coords(p0);
                let cs = [
                    step.dot(dose.row_dir) / dose.spacing[0],
                    step.dot(dose.col_dir) / dose.spacing[1],
                    step.dot(dose.normal),
                ];
                for i in 0..nx {
                    let c = [
                        c0[0] + cs[0] * i as f64,
                        c0[1] + cs[1] * i as f64,
                        c0[2] + cs[2] * i as f64,
                    ];
                    if dose.sample_uvw(c).is_some_and(|d| d >= level) {
                        slab[j * nx + i] = 1;
                    }
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contours::Stack;

    fn grid(dims: [usize; 3], spacing: [f64; 3]) -> Grid {
        Grid {
            dims,
            spacing,
            origin: Vec3::new(-50.0, -50.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: "1.2.3".into(),
        }
    }

    fn volume(g: &Grid, f: impl Fn(usize, usize, usize) -> i16) -> Volume {
        let [nx, ny, nz] = g.dims;
        let mut data = vec![0i16; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    data[k * nx * ny + j * nx + i] = f(i, j, k);
                }
            }
        }
        Volume {
            data,
            dims: g.dims,
            spacing: g.spacing,
            origin: g.origin,
            row_dir: g.row_dir,
            col_dir: g.col_dir,
            normal: g.normal,
            frame_of_reference_uid: g.frame_of_reference_uid.clone(),
            min_value: -1000,
            max_value: 1000,
        }
    }

    #[test]
    fn a_threshold_takes_what_it_says_and_the_limit_holds_it() {
        let g = grid([20, 20, 4], [1.0, 1.0, 2.0]);
        // A block of "bone" in the middle, everything else soft tissue.
        let v = volume(&g, |i, j, _| {
            if (5..15).contains(&i) && (5..15).contains(&j) {
                400
            } else {
                30
            }
        });
        let m = threshold_mask(&v, 200.0, f32::MAX, None);
        assert_eq!(m.iter().filter(|&&x| x != 0).count(), 10 * 10 * 4);

        // The same threshold, restricted to the left half.
        let [nx, ny, nz] = g.dims;
        let mut limit = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..10 {
                    limit[k * nx * ny + j * nx + i] = 1;
                }
            }
        }
        let m = threshold_mask(&v, 200.0, f32::MAX, Some(&limit));
        assert_eq!(m.iter().filter(|&&x| x != 0).count(), 5 * 10 * 4);
        // And a window with an upper bound excludes the bone entirely.
        let m = threshold_mask(&v, -100.0, 100.0, None);
        assert_eq!(
            m.iter().filter(|&&x| x != 0).count(),
            20 * 20 * 4 - 10 * 10 * 4
        );
    }

    #[test]
    fn the_shapes_have_the_volumes_they_should() {
        let g = grid([100, 100, 60], [1.0, 1.0, 1.0]);
        let c = Vec3::new(0.0, 0.0, 30.0);
        let r = 20.0;

        let cases: [(Shape, [f64; 3], f64); 4] = [
            // Half-sizes on the half-voxel, so that "the voxel centre is
            // inside" counts exactly the box: 21 x 25 x 29 of them.
            (Shape::Box, [10.5, 12.5, 14.5], 8.0 * 10.5 * 12.5 * 14.5),
            // Likewise in z: a half-height on the half-voxel is 21 slices.
            (
                Shape::Cylinder,
                [r, r, 10.5],
                std::f64::consts::PI * r * r * 21.0,
            ),
            (
                Shape::Sphere,
                [r, r, r],
                4.0 / 3.0 * std::f64::consts::PI * r * r * r,
            ),
            (
                Shape::Ellipsoid,
                [20.0, 10.0, 15.0],
                4.0 / 3.0 * std::f64::consts::PI * 20.0 * 10.0 * 15.0,
            ),
        ];
        for (shape, half, want_mm3) in cases {
            let m = shape_mask(shape, &g, c, half);
            let got = m.iter().filter(|&&x| x != 0).count() as f64;
            // A rasterized shape is accurate to half a voxel per face; on
            // these sizes that is well under a per cent for the round ones
            // and exact for the box.
            assert!(
                (got - want_mm3).abs() < 0.03 * want_mm3,
                "{}: {got} mm³ vs {want_mm3}",
                shape.label()
            );
            // And it comes out as contours, which is the point of it.
            let st = Stack::from_mask(&m, g.dims, 2);
            let v = st.volume_cm3(g.spacing);
            assert!(
                (v * 1000.0 - want_mm3).abs() < 0.05 * want_mm3,
                "{}: {v} cm³",
                shape.label()
            );
        }
    }

    #[test]
    fn the_field_of_view_is_the_circle_the_data_are_on() {
        let g = grid([64, 64, 6], [1.0, 1.0, 2.0]);
        let r = 26.0;
        // A padded reconstruction: -2000 outside the circle, tissue inside.
        let v = volume(&g, |i, j, _| {
            let d = ((i as f64 - 31.5).powi(2) + (j as f64 - 31.5).powi(2)).sqrt();
            if d <= r {
                40
            } else {
                -2000
            }
        });
        let m = fov_mask(&v).expect("a padded reconstruction");
        let got = m.iter().filter(|&&x| x != 0).count() as f64;
        let want = std::f64::consts::PI * r * r * 6.0;
        assert!((got - want).abs() < 0.05 * want, "{got} voxels vs {want}");
        // Air inside the circle belongs to the field of view all the same:
        // the mask is where the data are, not where the tissue is.
        let v2 = volume(&g, |i, j, _| {
            let d = ((i as f64 - 31.5).powi(2) + (j as f64 - 31.5).powi(2)).sqrt();
            if d > r {
                -2000
            } else if d > 10.0 {
                40
            } else {
                -1000
            }
        });
        let m2 = fov_mask(&v2).expect("a padded reconstruction");
        assert_eq!(
            m.iter().filter(|&&x| x != 0).count(),
            m2.iter().filter(|&&x| x != 0).count()
        );

        // A full-field image has nothing to outline.
        let full = volume(&g, |_, _, _| 40);
        assert!(fov_mask(&full).is_none());
    }

    #[test]
    fn a_shape_off_the_lattice_is_empty_not_a_panic() {
        let g = grid([20, 20, 4], [1.0, 1.0, 1.0]);
        let m = shape_mask(Shape::Sphere, &g, Vec3::new(500.0, 500.0, 500.0), [5.0; 3]);
        assert!(m.iter().all(|&x| x == 0));
    }
}
