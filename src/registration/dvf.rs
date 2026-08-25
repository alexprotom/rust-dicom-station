//! The deformation vector field: sampling a recovered transform onto a
//! regular lattice, and turning that lattice into things a view can draw.
//!
//! A transform is an equation; a vector field is a picture of it. Sampling
//! it once into a lattice (rather than evaluating the transform per pixel on
//! every repaint) is what makes the display affordable — a B-spline
//! evaluation is 64 weighted lookups and a landmark warp is a sum over every
//! landmark, neither of which belongs in a paint loop.
//!
//! Everything here is geometry: patient-space displacements and their
//! projection into a view plane. Nothing paints; the colours and the
//! arrowheads belong to the application.

use crate::render;
use crate::volume::ViewPlane;

use super::*;

/// How a vector field is drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum FieldStyle {
    #[default]
    /// Arrow per lattice node, from where the anatomy is to where it goes.
    Arrows,
    /// The lattice itself, pushed through the deformation — the classic
    /// "warped graph paper" that shows compression and expansion at a
    /// glance, where arrows only show motion.
    Grid,
    /// No glyphs, only the magnitude colour wash.
    None,
}

impl FieldStyle {
    pub const ALL: [FieldStyle; 3] = [FieldStyle::Arrows, FieldStyle::Grid, FieldStyle::None];

    pub fn label(self) -> &'static str {
        match self {
            FieldStyle::Arrows => "Arrows",
            FieldStyle::Grid => "Deformed grid",
            FieldStyle::None => "Colour only",
        }
    }
}

/// A displacement field sampled on a regular lattice over the fixed image.
#[derive(Clone)]
pub struct VectorField {
    pub dims: [usize; 3],
    /// Lattice spacing along each axis, mm.
    pub spacing: [f64; 3],
    pub origin: Vec3,
    pub axes: [Vec3; 3],
    /// Displacement at each node, patient mm, `[i + nx*(j + ny*k)]`.
    pub data: Vec<Vec3>,
    /// Largest magnitude present — the natural scale for colours and arrows.
    pub max_mag: f64,
    /// Name of the region the field was restricted to, if any.
    pub region: Option<String>,
}

impl VectorField {
    /// Sample `t` over the fixed volume, or over a region's bounding box,
    /// on a lattice of about `step_mm`.
    pub fn sample(
        vol: &Volume,
        t: &Transform3,
        region: Option<&RegionMask>,
        step_mm: f64,
    ) -> VectorField {
        let (lo, hi) = match region {
            Some(r) => r.bbox(),
            None => (
                [0, 0, 0],
                [vol.dims[0] - 1, vol.dims[1] - 1, vol.dims[2] - 1],
            ),
        };
        // Node steps in voxels, at least one voxel, at most the extent.
        let mut steps = [1usize; 3];
        let mut dims = [1usize; 3];
        for a in 0..3 {
            let span = hi[a] - lo[a] + 1;
            steps[a] = ((step_mm / vol.spacing[a]).round() as usize).clamp(1, span.max(1));
            dims[a] = span.div_ceil(steps[a]).max(1);
        }
        let axes = [vol.row_dir, vol.col_dir, vol.normal];
        let origin = vol.voxel_to_patient(lo[0] as f64, lo[1] as f64, lo[2] as f64);
        let spacing = [
            steps[0] as f64 * vol.spacing[0],
            steps[1] as f64 * vol.spacing[1],
            steps[2] as f64 * vol.spacing[2],
        ];
        let [nx, ny, nz] = dims;
        let data: Vec<Vec3> = (0..nz)
            .into_par_iter()
            .flat_map(|k| {
                let mut plane = Vec::with_capacity(nx * ny);
                for j in 0..ny {
                    for i in 0..nx {
                        let p = vol.voxel_to_patient(
                            (lo[0] + i * steps[0]) as f64,
                            (lo[1] + j * steps[1]) as f64,
                            (lo[2] + k * steps[2]) as f64,
                        );
                        let inside = region.map(|r| r.contains(p)).unwrap_or(true);
                        plane.push(if inside {
                            t.displacement(p)
                        } else {
                            Vec3::ZERO
                        });
                    }
                }
                plane
            })
            .collect();
        let max_mag = data.par_iter().map(|v| v.length()).reduce(|| 0.0, f64::max);
        VectorField {
            dims,
            spacing,
            origin,
            axes,
            data,
            max_mag,
            region: region.map(|r| r.name.clone()),
        }
    }

    /// Nodes in the lattice.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Patient position of a lattice node.
    pub fn node_position(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.origin
            + self.axes[0] * (i as f64 * self.spacing[0])
            + self.axes[1] * (j as f64 * self.spacing[1])
            + self.axes[2] * (k as f64 * self.spacing[2])
    }

    /// Displacement at a lattice node.
    pub fn at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        self.data[i + self.dims[0] * (j + self.dims[1] * k)]
    }

    /// Trilinear displacement at a patient point; zero outside the lattice.
    pub fn sample_patient(&self, p: Vec3) -> Vec3 {
        let d = p - self.origin;
        let mut u = [0.0f64; 3];
        for (a, slot) in u.iter_mut().enumerate() {
            *slot = d.dot(self.axes[a]) / self.spacing[a];
            if *slot < 0.0 || *slot > (self.dims[a] - 1) as f64 {
                return Vec3::ZERO;
            }
        }
        let i0 = u[0].floor() as usize;
        let j0 = u[1].floor() as usize;
        let k0 = u[2].floor() as usize;
        let i1 = (i0 + 1).min(self.dims[0] - 1);
        let j1 = (j0 + 1).min(self.dims[1] - 1);
        let k1 = (k0 + 1).min(self.dims[2] - 1);
        let (fu, fv, fw) = (u[0] - i0 as f64, u[1] - j0 as f64, u[2] - k0 as f64);
        let lerp = |a: Vec3, b: Vec3, t: f64| a * (1.0 - t) + b * t;
        let c00 = lerp(self.at(i0, j0, k0), self.at(i1, j0, k0), fu);
        let c10 = lerp(self.at(i0, j1, k0), self.at(i1, j1, k0), fu);
        let c01 = lerp(self.at(i0, j0, k1), self.at(i1, j0, k1), fu);
        let c11 = lerp(self.at(i0, j1, k1), self.at(i1, j1, k1), fu);
        lerp(lerp(c00, c10, fv), lerp(c01, c11, fv), fw)
    }

    /// Magnitude statistics of the whole field.
    pub fn stats(&self) -> VectorStats {
        VectorStats::of(&self.data)
    }

    /// `47 × 47 × 26 nodes at 10 mm · max 12.4 mm`.
    pub fn describe(&self) -> String {
        format!(
            "{} × {} × {} nodes at {:.0} mm · max {:.1} mm{}",
            self.dims[0],
            self.dims[1],
            self.dims[2],
            self.spacing.iter().sum::<f64>() / 3.0,
            self.max_mag,
            match &self.region {
                Some(r) => format!(" · inside {r}"),
                None => String::new(),
            }
        )
    }
}

/// One arrow in a view: where it starts and ends in display-pixel space,
/// how long the displacement really is, and how much of it points out of
/// the plane (which a 2-D arrow cannot show and a colour must).
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub from: [f32; 2],
    pub to: [f32; 2],
    /// Full 3-D magnitude, mm.
    pub magnitude: f32,
    /// Component along the view normal, mm — signed.
    pub out_of_plane: f32,
}

/// Arrows for one slice of one view plane, at about `step_mm` apart.
///
/// The lattice is re-walked in display-pixel space rather than reused from
/// the field, so the arrow density is the same in every view whatever the
/// slice thickness — three views of one field should look like three views
/// of one field.
pub fn glyphs_on_plane(
    field: &VectorField,
    vol: &Volume,
    plane: ViewPlane,
    slice: usize,
    step_mm: f64,
) -> Vec<Glyph> {
    let [w, h] = vol.plane_dims(plane);
    let [sx, sy] = vol.plane_spacing(plane);
    let step_x = ((step_mm / sx).round() as usize).max(1);
    let step_y = ((step_mm / sy).round() as usize).max(1);
    let normal = plane_normal(vol, plane);
    let mut out = Vec::new();
    let mut py = step_y / 2;
    while py < h {
        let mut px = step_x / 2;
        while px < w {
            let v = vol.plane_pixel_to_voxel(plane, slice, px as f64, py as f64);
            let p = vol.voxel_to_patient(v[0], v[1], v[2]);
            let d = field.sample_patient(p);
            if d.length() > 1e-6 {
                let (a, _) = render::patient_to_plane_pixel(vol, plane, slice, p);
                let (b, _) = render::patient_to_plane_pixel(vol, plane, slice, p + d);
                out.push(Glyph {
                    from: a,
                    to: b,
                    magnitude: d.length() as f32,
                    out_of_plane: d.dot(normal) as f32,
                });
            }
            px += step_x;
        }
        py += step_y;
    }
    out
}

/// The deformed lattice of one slice, as polylines in display-pixel space:
/// every row and every column of the sampling lattice, pushed through the
/// deformation.
pub fn deformed_grid_on_plane(
    field: &VectorField,
    vol: &Volume,
    plane: ViewPlane,
    slice: usize,
    step_mm: f64,
) -> Vec<Vec<[f32; 2]>> {
    let [w, h] = vol.plane_dims(plane);
    let [sx, sy] = vol.plane_spacing(plane);
    let step_x = ((step_mm / sx).round() as usize).max(1);
    let step_y = ((step_mm / sy).round() as usize).max(1);
    // Sub-sample each line finely enough that a curved cell reads as curved.
    let fine_x = (step_x / 4).max(1);
    let fine_y = (step_y / 4).max(1);
    let warped = |px: usize, py: usize| -> [f32; 2] {
        let v = vol.plane_pixel_to_voxel(plane, slice, px as f64, py as f64);
        let p = vol.voxel_to_patient(v[0], v[1], v[2]);
        let d = field.sample_patient(p);
        render::patient_to_plane_pixel(vol, plane, slice, p + d).0
    };
    let mut lines = Vec::new();
    let mut py = 0;
    while py < h {
        let mut line = Vec::new();
        let mut px = 0;
        while px < w {
            line.push(warped(px, py));
            px += fine_x;
        }
        lines.push(line);
        py += step_y;
    }
    let mut px = 0;
    while px < w {
        let mut line = Vec::new();
        let mut py = 0;
        while py < h {
            line.push(warped(px, py));
            py += fine_y;
        }
        lines.push(line);
        px += step_x;
    }
    lines
}

/// Arrows for a 3-D scene: node position, displaced position and magnitude,
/// thinned to at most `max_count` of them.
pub fn glyphs_3d(field: &VectorField, max_count: usize) -> Vec<(Vec3, Vec3, f64)> {
    let total = field.len();
    if total == 0 {
        return Vec::new();
    }
    let stride = total.div_ceil(max_count.max(1)).max(1);
    let [nx, ny, _] = field.dims;
    (0..total)
        .step_by(stride)
        .filter_map(|o| {
            let k = o / (nx * ny);
            let rem = o - k * nx * ny;
            let d = field.data[o];
            let m = d.length();
            if m < 1e-6 {
                return None;
            }
            let p = field.node_position(rem % nx, rem / nx, k);
            Some((p, p + d, m))
        })
        .collect()
}

/// Patient-space normal of a view plane (the direction a 2-D arrow cannot
/// show).
fn plane_normal(vol: &Volume, plane: ViewPlane) -> Vec3 {
    match plane {
        ViewPlane::Axial => vol.normal,
        ViewPlane::Sagittal => vol.row_dir,
        ViewPlane::Coronal => vol.col_dir,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol(dims: [usize; 3]) -> Volume {
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [2.0, 2.0, 2.0],
            origin: Vec3::ZERO,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 1,
        }
    }

    #[test]
    fn a_translation_samples_to_a_uniform_field() {
        let v = vol([40, 40, 20]);
        let shift = Vec3::new(3.0, -4.0, 0.0);
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, shift.x, shift.y, shift.z],
            Vec3::ZERO,
        ));
        let f = VectorField::sample(&v, &t, None, 10.0);
        assert!(!f.is_empty());
        assert_eq!(f.spacing, [10.0, 10.0, 10.0]);
        assert!((f.max_mag - 5.0).abs() < 1e-9);
        for d in &f.data {
            assert!((*d - shift).length() < 1e-9);
        }
        // Interpolation inside the lattice reproduces it exactly.
        let p = f.node_position(1, 1, 1) + Vec3::new(3.0, 1.0, 2.0);
        assert!((f.sample_patient(p) - shift).length() < 1e-9);
        // …and outside it is zero, so nothing is drawn where nothing is known.
        assert_eq!(f.sample_patient(Vec3::new(-500.0, 0.0, 0.0)), Vec3::ZERO);
        assert!(f.describe().contains("nodes at 10 mm"));
        assert!((f.stats().mean - 5.0).abs() < 1e-9);
    }

    #[test]
    fn glyphs_land_in_the_plane_and_know_what_leaves_it() {
        let v = vol([40, 40, 20]);
        // 6 mm straight through the axial plane, nothing in it.
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
            Vec3::ZERO,
        ));
        let f = VectorField::sample(&v, &t, None, 10.0);
        let g = glyphs_on_plane(&f, &v, ViewPlane::Axial, 10, 20.0);
        assert!(!g.is_empty());
        for a in &g {
            assert!((a.magnitude - 6.0).abs() < 1e-4);
            assert!((a.out_of_plane - 6.0).abs() < 1e-4);
            // In-plane the arrow has nowhere to go.
            assert!((a.from[0] - a.to[0]).abs() < 1e-3);
            assert!((a.from[1] - a.to[1]).abs() < 1e-3);
        }
        // In a sagittal view the same motion is in-plane.
        let g = glyphs_on_plane(&f, &v, ViewPlane::Sagittal, 20, 20.0);
        assert!(g.iter().all(|a| a.out_of_plane.abs() < 1e-4));
        assert!(g.iter().any(|a| (a.from[1] - a.to[1]).abs() > 1.0));
    }

    #[test]
    fn the_deformed_grid_and_the_3d_glyphs_are_bounded_in_size() {
        let v = vol([40, 40, 20]);
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            Vec3::ZERO,
        ));
        let f = VectorField::sample(&v, &t, None, 8.0);
        let lines = deformed_grid_on_plane(&f, &v, ViewPlane::Coronal, 10, 20.0);
        assert!(!lines.is_empty());
        assert!(lines.iter().all(|l| l.len() > 1));
        let g = glyphs_3d(&f, 100);
        assert!(!g.is_empty() && g.len() <= 100, "{}", g.len());
        for (a, b, m) in g {
            assert!(((b - a).length() - m).abs() < 1e-9);
        }
        assert!(glyphs_3d(&VectorField::sample(&v, &t, None, 1e9), 0).len() <= 1);
    }
}
