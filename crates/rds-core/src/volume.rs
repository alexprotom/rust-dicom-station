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
    /// (i, j) plane at fixed k - native acquisition plane.
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

/// The voxel lattice of a 3-D object in patient space, without its data.
///
/// A [`Volume`] carries one; so does every DICOM Segmentation object, whose
/// frames may sit on a grid of their own. Keeping the geometry separable is
/// what lets a segmentation be resampled from the grid it arrived on onto
/// the grid of whatever image series it is shown against
/// (`crate::dicomseg::SegSeries::rebind`).
#[derive(Clone, PartialEq)]
pub struct Grid {
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
    /// Frame of Reference UID the coordinates above are expressed in.
    pub frame_of_reference_uid: String,
}

/// The order and direction of a network's three spatial array axes, by
/// the anatomical direction each one increases toward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AxisOrder {
    /// Superior, anterior, right: nibabel's `as_closest_canonical` (RAS)
    /// seen through nnU-Net's axis transpose - TotalSegmentator, SegVol.
    #[default]
    Sar,
    /// Superior, posterior, left: SimpleITK's array of an image oriented
    /// `DICOMOrient("LPS")` - MRSegmentator, lungmask, CT-FM
    /// (`Orientation("SPL")`).
    Spl,
    /// Right, anterior, superior: MONAI's `Orientation("RAS")` with no
    /// transpose - the MONAI bundles, VISTA-3D.
    Ras,
}

impl AxisOrder {
    /// The three target directions in LPS patient space.
    pub fn targets(self) -> [Vec3; 3] {
        let s = Vec3::new(0.0, 0.0, 1.0);
        let a = Vec3::new(0.0, -1.0, 0.0);
        let r = Vec3::new(-1.0, 0.0, 0.0);
        match self {
            AxisOrder::Sar => [s, a, r],
            AxisOrder::Spl => [s, a * -1.0, r * -1.0],
            AxisOrder::Ras => [r, a, s],
        }
    }
}

/// Volume of one voxel of `spacing` (mm), in cm³.
#[inline]
pub fn voxel_cm3(spacing: [f64; 3]) -> f64 {
    spacing[0] * spacing[1] * spacing[2] / 1000.0
}

impl Grid {
    /// Find the permutation and flips that carry a lattice's own axes onto the
    /// canonical `[S, A, R]` order - superior, anterior and right, each
    /// increasing with the array index.
    ///
    /// Every inference engine wants this: it is what `nibabel`'s
    /// `as_closest_canonical` followed by nnU-Net's axis convention produces, and
    /// it is also what SegVol's `Orientationd(axcodes="RAS")` plus its
    /// `DimTranspose` (which swaps the first and last spatial axes) produces.
    /// Volumes are not assumed to be axis-aligned; the best match is chosen by
    /// direction cosine.
    pub fn canonical_axes(&self) -> ([usize; 3], [bool; 3]) {
        self.axes_toward(AxisOrder::Sar.targets())
    }

    /// [`Grid::canonical_axes`] for any target order: the permutation and
    /// flips that carry the lattice's own axes onto `targets`, three
    /// directions in LPS patient space, each the way one array axis must
    /// increase. Networks disagree on the order they were trained in -
    /// nnU-Net through nibabel sees `[S, A, R]`, nnU-Net through SimpleITK
    /// after `DICOMOrient("LPS")` sees `[S, P, L]`, MONAI's
    /// `Orientation("RAS")` sees `[R, A, S]` - and a 3-D convolution is not
    /// symmetric under a change of axes, so each is fed its own.
    pub fn axes_toward(&self, targets: [Vec3; 3]) -> ([usize; 3], [bool; 3]) {
        // LPS direction vectors of the three volume axes.
        let dirs: [Vec3; 3] = [self.row_dir, self.col_dir, self.normal];
        let mut perm = [0usize; 3];
        let mut flip = [false; 3];
        let mut used = [false; 3];
        for a in 0..3 {
            let mut best = 0usize;
            let mut best_dot = f64::NEG_INFINITY;
            for v in 0..3 {
                if used[v] {
                    continue;
                }
                let dot = dirs[v].dot(targets[a]);
                if dot.abs() > best_dot {
                    best_dot = dot.abs();
                    best = v;
                }
            }
            used[best] = true;
            perm[a] = best;
            flip[a] = dirs[best].dot(targets[a]) < 0.0;
        }
        (perm, flip)
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

    /// Volume of one voxel, cm³ - what every count of voxels is multiplied
    /// by to become a volume.
    #[inline]
    pub fn voxel_cm3(&self) -> f64 {
        voxel_cm3(self.spacing)
    }

    /// Same lattice to within a fraction of a voxel - the test that decides
    /// whether a mask can be reused as it is instead of being resampled.
    pub fn matches(&self, other: &Grid) -> bool {
        if self.dims != other.dims {
            return false;
        }
        let tol = self.spacing.iter().fold(f64::MAX, |a, b| a.min(*b)) * 0.01;
        (0..3).all(|a| (self.spacing[a] - other.spacing[a]).abs() < tol)
            && (self.origin - other.origin).length() < tol
            && (self.row_dir - other.row_dir).length() < 1e-4
            && (self.col_dir - other.col_dir).length() < 1e-4
            && (self.normal - other.normal).length() < 1e-4
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
    /// A volume with no voxels, for a workspace that carries no image series.
    ///
    /// A workspace does not have to contain a reconstructable volume - a folder
    /// or a handful of files can hold nothing but RT images, a structure set
    /// or a plan, and those are legitimate things to open. Rather than making
    /// the viewer's `LoadedStudy::volume` optional and forcing a hundred
    /// call sites to unwrap it, such a study carries this: dimensions of
    /// zero, so every voxel loop is empty and every lookup misses.
    ///
    /// The geometry is deliberately the identity rather than zeros. Spacing
    /// of zero would divide in [`Volume::patient_to_voxel`] and make
    /// [`Grid::matches`] false even against itself, and zero direction
    /// vectors would label every view edge the same way. Nothing should read
    /// this geometry - [`Volume::is_empty`] says not to - but if something
    /// does, it gets an answer that is merely useless rather than poisonous.
    pub fn empty() -> Volume {
        Volume {
            data: Vec::new(),
            dims: [0, 0, 0],
            spacing: [1.0, 1.0, 1.0],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        }
    }

    /// No voxels - nothing to display, sample, register or segment.
    ///
    /// Every feature that needs image data asks this first; see
    /// [`Volume::empty`].
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims.contains(&0)
    }

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

    /// Volume of one voxel, cm³.
    #[inline]
    pub fn voxel_cm3(&self) -> f64 {
        voxel_cm3(self.spacing)
    }

    /// This volume's lattice, for objects that must be resampled onto it.
    pub fn grid(&self) -> Grid {
        Grid {
            dims: self.dims,
            spacing: self.spacing,
            origin: self.origin,
            row_dir: self.row_dir,
            col_dir: self.col_dir,
            normal: self.normal,
            frame_of_reference_uid: self.frame_of_reference_uid.clone(),
        }
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
    /// * Axial: stored orientation (row 0 top) - radiological convention for
    ///   HFS axial data.
    /// * Sagittal: horizontal = j, vertical = k *flipped* (last slice at top,
    ///   i.e. superior up for ascending axial stacks).
    /// * Coronal: horizontal = i, vertical = k *flipped*.
    pub fn extract_slice(&self, plane: ViewPlane, slice: usize, out: &mut Vec<i16>) {
        let [nx, ny, nz] = self.dims;
        if self.is_empty() {
            // Nothing to reformat, and `par_chunks_mut(0)` would panic.
            out.clear();
            return;
        }
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
        // is its own cache line. Rows are independent - run them in parallel.
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
    pub fn plane_pixel_to_voxel(
        &self,
        plane: ViewPlane,
        slice: usize,
        px: f64,
        py: f64,
    ) -> [f64; 3] {
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

    /// Find the permutation and flips that carry a volume's own axes onto the
    /// canonical `[S, A, R]` order - see [`Grid::canonical_axes`].
    pub fn canonical_axes(&self) -> ([usize; 3], [bool; 3]) {
        self.grid().canonical_axes()
    }

    /// [`Volume::canonical_axes`] for any [`AxisOrder`].
    pub fn axes_toward(&self, order: AxisOrder) -> ([usize; 3], [bool; 3]) {
        self.grid().axes_toward(order.targets())
    }

    /// The voxels `lo..hi` (per axis, `hi` exclusive, clamped to the
    /// volume) as a volume of their own, at the same spacing and
    /// orientation, its origin moved onto the first voxel kept - the crop a
    /// model run on part of a scan sees. The bounds come back with it, as
    /// [`paste_labels`] needs them.
    pub fn crop(&self, lo: [usize; 3], hi: [usize; 3]) -> (Volume, [usize; 3], [usize; 3]) {
        let hi: [usize; 3] = std::array::from_fn(|a| hi[a].min(self.dims[a]));
        let lo: [usize; 3] = std::array::from_fn(|a| lo[a].min(hi[a]));
        let dims: [usize; 3] = std::array::from_fn(|a| hi[a] - lo[a]);
        let [nx, ny, _] = self.dims;
        let mut data = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
        for k in lo[2]..hi[2] {
            for j in lo[1]..hi[1] {
                let row = k * nx * ny + j * nx;
                data.extend_from_slice(&self.data[row + lo[0]..row + hi[0]]);
            }
        }
        let (min_value, max_value) = data
            .iter()
            .fold((i16::MAX, i16::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        let v = Volume {
            data,
            dims,
            spacing: self.spacing,
            origin: self.voxel_to_patient(lo[0] as f64, lo[1] as f64, lo[2] as f64),
            row_dir: self.row_dir,
            col_dir: self.col_dir,
            normal: self.normal,
            frame_of_reference_uid: self.frame_of_reference_uid.clone(),
            min_value: if dims.contains(&0) { 0 } else { min_value },
            max_value: if dims.contains(&0) { 0 } else { max_value },
        };
        (v, lo, hi)
    }
}

/// Write labels computed on a [`Volume::crop`] of a volume with `dims` back
/// into a zero label volume of the full size (`Volume::data` order).
pub fn paste_labels(dims: [usize; 3], lo: [usize; 3], hi: [usize; 3], part: &[u8]) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let mut out = vec![0u8; nx * ny * nz];
    let w = hi[0] - lo[0];
    let h = hi[1] - lo[1];
    for (kk, k) in (lo[2]..hi[2]).enumerate() {
        for (jj, j) in (lo[1]..hi[1]).enumerate() {
            let src = (kk * h + jj) * w;
            let dst = k * nx * ny + j * nx + lo[0];
            out[dst..dst + w].copy_from_slice(&part[src..src + w]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axial(nx: usize, ny: usize, nz: usize) -> Volume {
        let mut v = Volume::empty();
        v.dims = [nx, ny, nz];
        v.spacing = [0.5, 0.75, 2.0];
        v.origin = Vec3::new(-10.0, -20.0, 30.0);
        v.data = (0..nx * ny * nz).map(|i| i as i16).collect();
        v
    }

    #[test]
    fn the_three_axis_orders_of_a_standard_axial_scan() {
        // Axial DICOM in LPS: i toward the patient's left, j posterior, k
        // superior.
        let v = axial(4, 3, 2);
        assert_eq!(v.canonical_axes(), ([2, 1, 0], [false, true, true]));
        assert_eq!(v.axes_toward(AxisOrder::Sar), v.canonical_axes());
        assert_eq!(v.axes_toward(AxisOrder::Spl), ([2, 1, 0], [false; 3]));
        assert_eq!(
            v.axes_toward(AxisOrder::Ras),
            ([0, 1, 2], [true, true, false])
        );
    }

    #[test]
    fn a_crop_keeps_its_voxels_where_they_were_in_patient_space() {
        let v = axial(5, 4, 3);
        let (c, lo, hi) = v.crop([1, 2, 1], [9, 4, 3]);
        assert_eq!((lo, hi, c.dims), ([1, 2, 1], [5, 4, 3], [4, 2, 2]));
        for (k, j, i) in [(0, 0, 0), (1, 1, 3), (0, 1, 2)] {
            assert_eq!(c.index(i, j, k), v.index(i + 1, j + 2, k + 1));
            let p = c.voxel_to_patient(i as f64, j as f64, k as f64);
            let q = v.voxel_to_patient((i + 1) as f64, (j + 2) as f64, (k + 1) as f64);
            assert!((p - q).length() < 1e-9);
        }
        assert_eq!(
            (c.min_value, c.max_value),
            (v.index(1, 2, 1), v.index(4, 3, 2))
        );
        // Labels computed on the crop land back on the voxels they came from.
        let labels: Vec<u8> = (0..c.data.len()).map(|n| n as u8 + 1).collect();
        let full = paste_labels(v.dims, lo, hi, &labels);
        assert_eq!(full.len(), v.data.len());
        assert_eq!(full.iter().filter(|l| **l != 0).count(), labels.len());
        // (i, j, k) = (1, 2, 1) and (4, 3, 2) in a 5 x 4 x 3 volume.
        assert_eq!(full[31], 1);
        assert_eq!(full[59], labels.len() as u8);
        // An empty crop is a volume with no voxels, not a panic.
        let (e, _, _) = v.crop([3, 0, 0], [3, 4, 3]);
        assert!(e.is_empty());
    }
}
