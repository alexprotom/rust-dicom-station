//! 3D image volume (CT/MR/PT) reconstructed from a DICOM series, with full
//! patient-coordinate geometry and fast orthogonal slice extraction.

use rayon::prelude::*;

use crate::geometry::Vec3;

/// Viewing orientation of one of the three MPR panes.
///
/// Extraction is done in *index space* of the acquired volume, which for the
/// overwhelmingly common axial acquisitions maps directly onto anatomical
/// axial / sagittal / coronal planes. Edge labels are always derived from the
/// true patient-space direction vectors, so oblique or non-HFS data remains
/// annotated correctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewPlane {
    /// (i, j) plane at fixed k — native acquisition plane.
    Axial,
    /// (j, k) plane at fixed i.
    Sagittal,
    /// (i, k) plane at fixed j.
    Coronal,
}

impl ViewPlane {
    pub fn title(self) -> &'static str {
        match self {
            ViewPlane::Axial => "Axial",
            ViewPlane::Sagittal => "Sagittal",
            ViewPlane::Coronal => "Coronal",
        }
    }
}

/// A scalar image volume in HU (or raw modality units), i16 storage.
#[derive(Clone)]
pub struct Volume {
    /// Voxel data, index order: `data[k * nx * ny + j * nx + i]`
    /// (i = column, j = row, k = slice).
    pub data: Vec<i16>,
    /// Dimensions `[nx, ny, nz]` = [columns, rows, slices].
    pub dims: [usize; 3],
    /// Voxel spacing `[sx, sy, sz]` in mm along i / j / k.
    pub spacing: [f64; 3],
    /// Patient coordinates of the center of voxel (0, 0, 0).
    pub origin: Vec3,
    /// Direction of increasing column index i (unit vector, patient coords).
    pub row_dir: Vec3,
    /// Direction of increasing row index j (unit vector, patient coords).
    pub col_dir: Vec3,
    /// Direction of increasing slice index k (unit vector, patient coords).
    pub normal: Vec3,
    /// Frame of Reference UID (used to associate RT objects).
    pub frame_of_reference_uid: String,
    /// Value range present in the data (for auto window suggestions).
    pub min_value: i16,
    pub max_value: i16,
}

impl Volume {
    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> i16 {
        self.data[k * self.dims[0] * self.dims[1] + j * self.dims[0] + i]
    }

    /// Voxel value with bounds check; returns None outside the volume.
    #[inline]
    pub fn get(&self, i: i64, j: i64, k: i64) -> Option<i16> {
        if i < 0
            || j < 0
            || k < 0
            || i >= self.dims[0] as i64
            || j >= self.dims[1] as i64
            || k >= self.dims[2] as i64
        {
            None
        } else {
            Some(self.index(i as usize, j as usize, k as usize))
        }
    }

    /// Map fractional voxel indices to patient coordinates (mm).
    #[inline]
    pub fn voxel_to_patient(&self, i: f64, j: f64, k: f64) -> Vec3 {
        self.origin
            + self.row_dir * (i * self.spacing[0])
            + self.col_dir * (j * self.spacing[1])
            + self.normal * (k * self.spacing[2])
    }

    /// Map patient coordinates (mm) to fractional voxel indices.
    #[inline]
    pub fn patient_to_voxel(&self, p: Vec3) -> [f64; 3] {
        let d = p - self.origin;
        [
            d.dot(self.row_dir) / self.spacing[0],
            d.dot(self.col_dir) / self.spacing[1],
            d.dot(self.normal) / self.spacing[2],
        ]
    }

    /// Number of slices along the scroll axis of a view plane.
    pub fn plane_slice_count(&self, plane: ViewPlane) -> usize {
        match plane {
            ViewPlane::Axial => self.dims[2],
            ViewPlane::Sagittal => self.dims[0],
            ViewPlane::Coronal => self.dims[1],
        }
    }

    /// In-plane pixel dimensions `[width, height]` of an extracted slice.
    pub fn plane_dims(&self, plane: ViewPlane) -> [usize; 2] {
        match plane {
            ViewPlane::Axial => [self.dims[0], self.dims[1]],
            ViewPlane::Sagittal => [self.dims[1], self.dims[2]],
            ViewPlane::Coronal => [self.dims[0], self.dims[2]],
        }
    }

    /// In-plane physical pixel spacing `[mm_per_px_x, mm_per_px_y]`.
    pub fn plane_spacing(&self, plane: ViewPlane) -> [f64; 2] {
        match plane {
            ViewPlane::Axial => [self.spacing[0], self.spacing[1]],
            ViewPlane::Sagittal => [self.spacing[1], self.spacing[2]],
            ViewPlane::Coronal => [self.spacing[0], self.spacing[2]],
        }
    }

    /// Extract an orthogonal slice as a contiguous i16 buffer (row-major,
    /// top-left origin as displayed).
    ///
    /// Display conventions:
    /// * Axial: stored orientation (row 0 top) — radiological convention for
    ///   HFS axial data.
    /// * Sagittal: horizontal = j, vertical = k *flipped* (last slice at top,
    ///   i.e. superior up for ascending axial stacks).
    /// * Coronal: horizontal = i, vertical = k *flipped*.
    pub fn extract_slice(&self, plane: ViewPlane, slice: usize, out: &mut Vec<i16>) {
        let [nx, ny, nz] = self.dims;
        let (w, h) = {
            let d = self.plane_dims(plane);
            (d[0], d[1])
        };
        if plane == ViewPlane::Axial {
            // Native plane: one contiguous copy.
            let k = slice.min(nz.saturating_sub(1));
            let base = k * nx * ny;
            out.clear();
            out.reserve(w * h);
            out.extend_from_slice(&self.data[base..base + nx * ny]);
            return;
        }
        // Reformatted planes read across the slice stride, so every element
        // is its own cache line. Rows are independent — run them in parallel.
        out.clear();
        out.resize(w * h, 0);
        match plane {
            ViewPlane::Sagittal => {
                let i = slice.min(nx.saturating_sub(1));
                // rows: k from top (nz-1) down to 0; cols: j 0..ny
                out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
                    let base = (nz - 1 - r) * nx * ny + i;
                    for (j, o) in row.iter_mut().enumerate() {
                        *o = self.data[base + j * nx];
                    }
                });
            }
            ViewPlane::Coronal => {
                let j = slice.min(ny.saturating_sub(1));
                out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
                    let base = (nz - 1 - r) * nx * ny + j * nx;
                    row.copy_from_slice(&self.data[base..base + nx]);
                });
            }
            ViewPlane::Axial => unreachable!("handled above"),
        }
    }

    /// Convert an in-plane display pixel position (fractional) plus the view's
    /// current slice index into fractional volume indices (i, j, k).
    pub fn plane_pixel_to_voxel(&self, plane: ViewPlane, slice: usize, px: f64, py: f64) -> [f64; 3] {
        let nz = self.dims[2] as f64;
        match plane {
            ViewPlane::Axial => [px, py, slice as f64],
            ViewPlane::Sagittal => [slice as f64, px, (nz - 1.0) - py],
            ViewPlane::Coronal => [px, slice as f64, (nz - 1.0) - py],
        }
    }

    /// Inverse of [`plane_pixel_to_voxel`]: volume indices → (in-plane x,
    /// in-plane y, slice-axis index) for the given plane.
    pub fn voxel_to_plane_pixel(&self, plane: ViewPlane, v: [f64; 3]) -> [f64; 3] {
        let nz = self.dims[2] as f64;
        match plane {
            ViewPlane::Axial => [v[0], v[1], v[2]],
            ViewPlane::Sagittal => [v[1], (nz - 1.0) - v[2], v[0]],
            ViewPlane::Coronal => [v[0], (nz - 1.0) - v[2], v[1]],
        }
    }

    /// Trilinear interpolation of the volume at a patient-space point.
    /// Returns `None` outside the volume.
    pub fn sample_patient(&self, p: Vec3) -> Option<f32> {
        let [u, v, w] = self.patient_to_voxel(p);
        let [nx, ny, nz] = self.dims;
        if u < 0.0 || v < 0.0 || w < 0.0 {
            return None;
        }
        let i0 = u.floor() as usize;
        let j0 = v.floor() as usize;
        let k0 = w.floor() as usize;
        if i0 + 1 >= nx || j0 + 1 >= ny || k0 + 1 >= nz {
            return None;
        }
        let fu = (u - i0 as f64) as f32;
        let fv = (v - j0 as f64) as f32;
        let fw = (w - k0 as f64) as f32;
        let at = |i: usize, j: usize, k: usize| self.index(i, j, k) as f32;
        let c00 = at(i0, j0, k0) + (at(i0 + 1, j0, k0) - at(i0, j0, k0)) * fu;
        let c10 = at(i0, j0 + 1, k0) + (at(i0 + 1, j0 + 1, k0) - at(i0, j0 + 1, k0)) * fu;
        let c01 = at(i0, j0, k0 + 1) + (at(i0 + 1, j0, k0 + 1) - at(i0, j0, k0 + 1)) * fu;
        let c11 =
            at(i0, j0 + 1, k0 + 1) + (at(i0 + 1, j0 + 1, k0 + 1) - at(i0, j0 + 1, k0 + 1)) * fu;
        let c0 = c00 + (c10 - c00) * fv;
        let c1 = c01 + (c11 - c01) * fv;
        Some(c0 + (c1 - c0) * fw)
    }

    /// Patient-space direction vectors of the displayed +x and +y screen axes
    /// for a view plane (used for L/R/A/P/S/I edge labels).
    pub fn plane_screen_dirs(&self, plane: ViewPlane) -> (Vec3, Vec3) {
        match plane {
            ViewPlane::Axial => (self.row_dir, self.col_dir),
            ViewPlane::Sagittal => (self.col_dir, self.normal * -1.0),
            ViewPlane::Coronal => (self.row_dir, self.normal * -1.0),
        }
    }
}
