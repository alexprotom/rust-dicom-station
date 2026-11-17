//! Interactive segmentation: voxel label masks painted with a 2D/3D brush,
//! seeded region growing ("smart" segmentation, as in MITK's interactive
//! tools), per-stroke undo, and conversion back to RTSTRUCT contours so a
//! drawn segmentation can ride the existing DICOM export pipeline.
//!
//! Everything is pure Rust and CPU-side: the mask is one byte per voxel in
//! the exact index order of [`Volume::data`], so slice overlays reuse the
//! same display conventions as the volume itself.

use std::collections::BinaryHeap;

use egui::Color32;
use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::rtstruct::{Contour, Roi};
use crate::volume::{Grid, ViewPlane, Volume};

/// Colors handed out to newly created segmentations.
pub const SEG_PALETTE: &[[u8; 3]] = &[
    [255, 89, 94],
    [138, 201, 38],
    [25, 130, 196],
    [255, 202, 58],
    [106, 76, 147],
    [255, 121, 198],
    [82, 255, 213],
    [251, 133, 0],
];

/// Cap on how many strokes can be undone (bounds undo memory).
const UNDO_DEPTH: usize = 64;

/// Where [`Segmentation::gen`] comes from: one counter for every
/// segmentation in the process, so a generation names one state of one mask
/// (and of its copies, until either is edited). The 3D window keeps a mesh
/// per segment on the strength of that and re-meshes only the segment that
/// was painted; with a counter per segmentation, two copies edited once each
/// would have looked alike.
static GENERATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_gen() -> u64 {
    GENERATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A meshing snapshot of a mask: `(padded bool grid, grid dims, bbox lo,
/// stride)` - see [`Segmentation::mesh_grid`].
pub type MeshGrid = (Vec<bool>, [usize; 3], [usize; 3], usize);

/// A binary label mask over a study's volume.
#[derive(Clone)]
pub struct Segmentation {
    pub name: String,
    pub color: [u8; 3],
    pub visible: bool,
    /// One byte per voxel (0 or 1), index order `k * nx * ny + j * nx + i` -
    /// identical to [`Volume::data`].
    pub mask: Vec<u8>,
    pub dims: [usize; 3],
    /// Number of set voxels (kept incrementally).
    pub count: usize,
    /// A new value on every edit, so the 2D overlays and the 3D meshes
    /// rebuild. Unique in the process (see `GENERATIONS`): two
    /// segmentations with the same `gen` hold the same mask.
    pub gen: u64,
    /// Extent of all voxels ever set (inclusive). May overestimate after
    /// erasing - harmless, it only bounds later scans.
    pub bbox: Option<([usize; 3], [usize; 3])>,
    /// Undo stack: per stroke, the changed voxels with their prior values.
    undo: Vec<Vec<(u32, u8)>>,
    /// Changes of the stroke currently in progress.
    pending: Vec<(u32, u8)>,
}

impl Segmentation {
    pub fn new(name: String, color: [u8; 3], dims: [usize; 3]) -> Self {
        Segmentation {
            name,
            color,
            visible: true,
            mask: vec![0; dims[0] * dims[1] * dims[2]],
            dims,
            count: 0,
            gen: next_gen(),
            bbox: None,
            undo: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// Adopt a ready-made mask - a DICOM SEG frame stack, a resampling onto
    /// another lattice, a propagated structure. The count and bounding box
    /// are recomputed in one pass; no undo history is carried, because the
    /// voxels it would refer to were never edited here.
    pub fn from_mask(name: String, color: [u8; 3], dims: [usize; 3], mask: Vec<u8>) -> Self {
        let mut seg = Segmentation::new(name, color, dims);
        let (bbox, count) = crate::morphology::mask_extent(&mask, dims);
        seg.mask = mask;
        seg.mask.resize(dims[0] * dims[1] * dims[2], 0);
        seg.count = count;
        seg.bbox = bbox;
        seg
    }

    /// Builder form of [`Segmentation::visible`], for the rebind path.
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Segmented volume in cm³ for the given voxel spacing (mm).
    pub fn volume_cm3(&self, spacing: [f64; 3]) -> f64 {
        self.count as f64 * crate::volume::voxel_cm3(spacing)
    }

    /// Build a segmentation from one class of a multi-label voxel map
    /// (`labels` in `Volume::data` index order, mask = `labels == label`).
    /// Count and bounding box are filled in the same pass; there is no undo
    /// history.
    pub fn from_label_map(
        name: String,
        color: [u8; 3],
        dims: [usize; 3],
        labels: &[u8],
        label: u8,
    ) -> Self {
        Self::from_label_map_many(dims, labels, &[(label, name, color)])
            .pop()
            .expect("one class requested")
    }

    /// Build one segmentation per requested class of a multi-label voxel
    /// map in a single pass over it - what an auto-segmentation run with a
    /// hundred structures needs, rather than a hundred passes.
    pub fn from_label_map_many(
        dims: [usize; 3],
        labels: &[u8],
        classes: &[(u8, String, [u8; 3])],
    ) -> Vec<Self> {
        debug_assert_eq!(labels.len(), dims[0] * dims[1] * dims[2]);
        let [nx, ny, _] = dims;
        // Which requested segmentation a label maps to, if any.
        let mut slot = [usize::MAX; 256];
        for (s, (label, _, _)) in classes.iter().enumerate() {
            slot[*label as usize] = s;
        }
        let mut segs: Vec<Segmentation> = classes
            .iter()
            .map(|(_, name, color)| Segmentation::new(name.clone(), *color, dims))
            .collect();
        // Per class: the extent seen so far, tracked per row so the inner
        // loop touches nothing but the mask.
        let mut boxes: Vec<Option<([usize; 3], [usize; 3])>> = vec![None; segs.len()];
        for (row, chunk) in labels.chunks(nx).enumerate() {
            let (j, k) = (row % ny, row / ny);
            for (i, l) in chunk.iter().enumerate() {
                let s = slot[*l as usize];
                if s == usize::MAX {
                    continue;
                }
                segs[s].mask[row * nx + i] = 1;
                segs[s].count += 1;
                boxes[s] = Some(match boxes[s] {
                    None => ([i, j, k], [i, j, k]),
                    Some((lo, hi)) => (
                        [lo[0].min(i), lo[1].min(j), lo[2].min(k)],
                        [hi[0].max(i), hi[1].max(j), hi[2].max(k)],
                    ),
                });
            }
        }
        for (seg, bbox) in segs.iter_mut().zip(boxes) {
            seg.bbox = bbox;
        }
        segs
    }

    #[inline]
    fn touch(&mut self, i: usize, j: usize, k: usize) {
        match &mut self.bbox {
            Some((lo, hi)) => {
                lo[0] = lo[0].min(i);
                lo[1] = lo[1].min(j);
                lo[2] = lo[2].min(k);
                hi[0] = hi[0].max(i);
                hi[1] = hi[1].max(j);
                hi[2] = hi[2].max(k);
            }
            None => self.bbox = Some(([i, j, k], [i, j, k])),
        }
    }

    /// Paint (or erase) a capsule swept from `a` to `b` (fractional voxel
    /// coords) with the given radius in mm - sweeping between the previous
    /// and current pointer sample keeps fast strokes gap-free. With
    /// `plane2d` set, painting is confined to that single displayed slice
    /// (2D circle); otherwise the brush is a 3D sphere.
    pub fn paint_capsule(
        &mut self,
        vol: &Volume,
        a: [f64; 3],
        b: [f64; 3],
        radius_mm: f64,
        erase: bool,
        plane2d: Option<(ViewPlane, usize)>,
    ) {
        self.paint_capsule_within(vol, a, b, radius_mm, erase, plane2d, None);
    }

    /// [`Self::paint_capsule`] with an edge band: with `values` set, only
    /// the voxels of the capsule whose image value lies in `lo..=hi` are
    /// painted (or erased) - the smart brush of the contour tools, as the
    /// threshold-limited paintbrush of a mask editor has it: voxel by voxel,
    /// through the slice or the sphere alike. (On a structure the stamp is
    /// also cut back to the piece the pointer stands on; a sphere has no
    /// such piece to speak of, and a voxel brush has never needed one.)
    #[allow(clippy::too_many_arguments)]
    pub fn paint_capsule_within(
        &mut self,
        vol: &Volume,
        a: [f64; 3],
        b: [f64; 3],
        radius_mm: f64,
        erase: bool,
        plane2d: Option<(ViewPlane, usize)>,
        values: Option<(f32, f32)>,
    ) {
        let [nx, ny, nz] = self.dims;
        let sp = vol.spacing;
        let r = radius_mm.max(0.01);

        // Voxel-space search box around the capsule.
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for ax in 0..3 {
            let rv = r / sp[ax];
            let l = (a[ax].min(b[ax]) - rv).floor();
            let h = (a[ax].max(b[ax]) + rv).ceil();
            let max = [nx, ny, nz][ax] as f64 - 1.0;
            if h < 0.0 || l > max {
                return;
            }
            lo[ax] = l.max(0.0) as usize;
            hi[ax] = h.min(max) as usize;
        }
        if let Some((plane, slice)) = plane2d {
            let ax = match plane {
                ViewPlane::Axial => 2,
                ViewPlane::Sagittal => 0,
                ViewPlane::Coronal => 1,
            };
            let s = slice.min([nx, ny, nz][ax] - 1);
            lo[ax] = s;
            hi[ax] = s;
        }

        // Distance to the segment a→b, measured in mm.
        let ab = [
            (b[0] - a[0]) * sp[0],
            (b[1] - a[1]) * sp[1],
            (b[2] - a[2]) * sp[2],
        ];
        let ab2 = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
        let want: u8 = if erase { 0 } else { 1 };
        let r2 = r * r;
        let mut any = false;
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                let row = k * nx * ny + j * nx;
                for i in lo[0]..=hi[0] {
                    let d = [
                        (i as f64 - a[0]) * sp[0],
                        (j as f64 - a[1]) * sp[1],
                        (k as f64 - a[2]) * sp[2],
                    ];
                    let t = if ab2 > 1e-12 {
                        ((d[0] * ab[0] + d[1] * ab[1] + d[2] * ab[2]) / ab2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let e = [d[0] - ab[0] * t, d[1] - ab[1] * t, d[2] - ab[2] * t];
                    if e[0] * e[0] + e[1] * e[1] + e[2] * e[2] > r2 {
                        continue;
                    }
                    let idx = row + i;
                    if let Some((lo, hi)) = values {
                        let g = vol.data.get(idx).map_or(f32::NAN, |&g| g as f32);
                        // NaN (no image under the mask) fails both tests.
                        if !(g >= lo && g <= hi) {
                            continue;
                        }
                    }
                    let cur = self.mask[idx];
                    if cur != want {
                        self.pending.push((idx as u32, cur));
                        self.mask[idx] = want;
                        self.count = if want == 1 {
                            self.count + 1
                        } else {
                            self.count - 1
                        };
                        self.touch(i, j, k);
                        any = true;
                    }
                }
            }
        }
        if any {
            self.gen = next_gen();
        }
    }

    /// Add a set of voxels (linear indices), e.g. a committed region-grow.
    pub fn add_voxels(&mut self, voxels: &[u32]) {
        let [nx, ny, _] = self.dims;
        let sl = nx * ny;
        let mut any = false;
        for &v in voxels {
            let idx = v as usize;
            if self.mask[idx] == 0 {
                self.pending.push((v, 0));
                self.mask[idx] = 1;
                self.count += 1;
                let k = idx / sl;
                let r = idx % sl;
                self.touch(r % nx, r / nx, k);
                any = true;
            }
        }
        if any {
            self.gen = next_gen();
        }
    }

    /// Replace the whole mask: a tidy pass, a margin in millimetres, an
    /// emptying - anything that acts on the segmentation rather than on the
    /// voxels under a brush.
    ///
    /// The count and the bounding box are recomputed in one pass and the
    /// overlays and the meshes are told to rebuild, which painting does for
    /// itself voxel by voxel and a wholesale replacement cannot. The change
    /// is one undo step like any stroke, recorded as the voxels that
    /// differ, unless there are so many of them that the record would cost
    /// more than the mask it describes; then the history is dropped instead
    /// of grown, because no undo is better than an undo that restores a mask
    /// nobody drew.
    pub fn replace_mask(&mut self, mut mask: Vec<u8>) {
        let n = self.dims[0] * self.dims[1] * self.dims[2];
        mask.resize(n, 0);
        self.end_stroke();
        // Five bytes an entry against one byte a voxel: past a fifth of the
        // volume the record is the more expensive of the two.
        let budget = (n / 5).max(4096);
        let mut changes: Vec<(u32, u8)> = Vec::new();
        let mut too_many = false;
        for (i, (old, new)) in self.mask.iter().zip(mask.iter()).enumerate() {
            if old != new {
                if changes.len() >= budget {
                    too_many = true;
                    break;
                }
                changes.push((i as u32, *old));
            }
        }
        if !too_many && changes.is_empty() {
            return; // nothing differs, so nothing happened
        }
        self.mask = mask;
        let (bbox, count) = crate::morphology::mask_extent(&self.mask, self.dims);
        self.count = count;
        self.bbox = bbox;
        self.gen = next_gen();
        if too_many {
            self.undo.clear();
        } else {
            self.undo.push(changes);
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
    }

    /// Close the stroke in progress, making it one undo step.
    pub fn end_stroke(&mut self) {
        if !self.pending.is_empty() {
            self.undo.push(std::mem::take(&mut self.pending));
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() || !self.pending.is_empty()
    }

    /// Undo the last stroke (finishing an in-progress one first).
    pub fn undo_last(&mut self) -> bool {
        self.end_stroke();
        let Some(changes) = self.undo.pop() else {
            return false;
        };
        let plane = self.dims[0] * self.dims[1];
        for &(idx, old) in changes.iter().rev() {
            let cur = self.mask[idx as usize];
            if cur != old {
                self.count = if old != 0 {
                    self.count + 1
                } else {
                    self.count - 1
                };
            }
            self.mask[idx as usize] = old;
            // A voxel set back may lie outside the box a shrinking edit
            // left: the box only ever has to cover every set voxel.
            if old != 0 && plane > 0 {
                let idx = idx as usize;
                let (k, r) = (idx / plane, idx % plane);
                self.touch(r % self.dims[0], r / self.dims[0], k);
            }
        }
        self.gen = next_gen();
        true
    }

    /// The linear index of cell `(u, v)` of slice `level` along lattice
    /// axis `axis`, `(u, v)` running along [`crate::contours::plane_axes`].
    fn slice_index(&self, axis: usize, level: usize, u: usize, v: usize) -> usize {
        let [ua, va] = crate::contours::plane_axes(axis);
        let mut idx = [0usize; 3];
        idx[ua] = u;
        idx[va] = v;
        idx[axis] = level;
        idx[2] * self.dims[0] * self.dims[1] + idx[1] * self.dims[0] + idx[0]
    }

    /// The width and height of a slice along `axis`: the lattice sizes of
    /// its two in-plane axes, in [`crate::contours::plane_axes`] order.
    pub fn slice_dims(&self, axis: usize) -> [usize; 2] {
        let [ua, va] = crate::contours::plane_axes(axis);
        [self.dims[ua], self.dims[va]]
    }

    /// One slice of the mask, row-major in the slice's own `(u, v)` - the
    /// raster the contour tools draw on (a drawn ring's coordinates are
    /// fractional `(u, v)`). Empty when `level` is off the lattice.
    pub fn slice(&self, axis: usize, level: usize) -> Vec<u8> {
        let axis = axis.min(2);
        if level >= self.dims[axis] {
            return Vec::new();
        }
        let [w, h] = self.slice_dims(axis);
        let mut out = vec![0u8; w * h];
        for v in 0..h {
            for u in 0..w {
                out[v * w + u] = self.mask[self.slice_index(axis, level, u, v)];
            }
        }
        out
    }

    /// Write one slice back (the raster [`Self::slice`] reads), as part of
    /// the stroke in progress: every voxel that changes is recorded for the
    /// undo of the stroke, as a brush sample's are, and the count, the box
    /// and the generation follow. Returns whether anything changed; a
    /// slice of the wrong size, or off the lattice, changes nothing.
    pub fn write_slice(&mut self, axis: usize, level: usize, slice: &[u8]) -> bool {
        let axis = axis.min(2);
        let [w, h] = self.slice_dims(axis);
        if level >= self.dims[axis] || slice.len() != w * h {
            return false;
        }
        let [nx, ny, _] = self.dims;
        let mut any = false;
        for v in 0..h {
            for u in 0..w {
                let want = u8::from(slice[v * w + u] != 0);
                let idx = self.slice_index(axis, level, u, v);
                let cur = self.mask[idx];
                if cur == want {
                    continue;
                }
                self.pending.push((idx as u32, cur));
                self.mask[idx] = want;
                if want == 1 {
                    self.count += 1;
                    let (k, r) = (idx / (nx * ny), idx % (nx * ny));
                    self.touch(r % nx, r / nx, k);
                } else {
                    self.count -= 1;
                }
                any = true;
            }
        }
        if any {
            self.gen = next_gen();
        }
        any
    }

    /// Snapshot the mask's bounding box (plus a one-cell empty margin so the
    /// surface closes) as a bool grid ready for surface-nets meshing.
    /// Very large masks are max-pooled with an integer stride to keep the
    /// grid - and thus the rebuild latency - bounded for interactive use.
    /// Returns `(grid, grid_dims, bbox_lo, stride)`.
    pub fn mesh_grid(&self) -> Option<MeshGrid> {
        let (lo, hi) = self.bbox?;
        if self.count == 0 {
            return None;
        }
        let [nx, ny, _] = self.dims;
        let size = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
        const MAX_CELLS: usize = 6_000_000;
        let mut stride = 1usize;
        let gdim = |s: usize, st: usize| s.div_ceil(st) + 2;
        while gdim(size[0], stride) * gdim(size[1], stride) * gdim(size[2], stride) > MAX_CELLS {
            stride += 1;
        }
        let g = [
            gdim(size[0], stride),
            gdim(size[1], stride),
            gdim(size[2], stride),
        ];
        let mut grid = vec![false; g[0] * g[1] * g[2]];
        let mask = &self.mask;
        // Each grid layer max-pools its own voxel slab - layers are independent.
        grid.par_chunks_mut(g[0] * g[1])
            .enumerate()
            .for_each(|(gk, layer)| {
                if gk == 0 || gk == g[2] - 1 {
                    return; // padding layers stay empty
                }
                let k0 = lo[2] + (gk - 1) * stride;
                let k1 = (k0 + stride).min(hi[2] + 1);
                for gj in 1..g[1] - 1 {
                    let j0 = lo[1] + (gj - 1) * stride;
                    let j1 = (j0 + stride).min(hi[1] + 1);
                    for gi in 1..g[0] - 1 {
                        let i0 = lo[0] + (gi - 1) * stride;
                        let i1 = (i0 + stride).min(hi[0] + 1);
                        'blk: for k in k0..k1 {
                            for j in j0..j1 {
                                let base = k * nx * ny + j * nx;
                                for i in i0..i1 {
                                    if mask[base + i] != 0 {
                                        layer[gj * g[0] + gi] = true;
                                        break 'blk;
                                    }
                                }
                            }
                        }
                    }
                }
            });
        Some((grid, g, lo, stride))
    }
}

/// Blend a mask into an RGBA slice image, using the same display conventions
/// as [`Volume::extract_slice`] (sagittal/coronal rows are k-flipped).
/// `out` must already hold `w * h` pixels for the plane.
pub fn overlay_slice(
    mask: &[u8],
    dims: [usize; 3],
    plane: ViewPlane,
    slice: usize,
    color: [u8; 3],
    alpha: u8,
    out: &mut [Color32],
) {
    overlay_slice_within(mask, dims, None, plane, slice, color, alpha, out);
}

/// [`overlay_slice`] for a mask whose extent is known ([`Segmentation::bbox`],
/// which may overestimate but never underestimates): a slice the extent does
/// not reach is skipped outright and a slice it does is scanned only inside
/// it. With a hundred organs from an auto-segmentation that is the difference
/// between reading the whole plane a hundred times and reading a few small
/// boxes. `None` scans the whole lattice.
#[allow(clippy::too_many_arguments)]
pub fn overlay_slice_within(
    mask: &[u8],
    dims: [usize; 3],
    bbox: Option<([usize; 3], [usize; 3])>,
    plane: ViewPlane,
    slice: usize,
    color: [u8; 3],
    alpha: u8,
    out: &mut [Color32],
) {
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 {
        return;
    }
    let col = Color32::from_rgba_unmultiplied(color[0], color[1], color[2], alpha);
    // Unknown extent: the whole lattice.
    let (lo, hi) = bbox.unwrap_or(([0; 3], [nx - 1, ny - 1, nz - 1]));
    let hi = [hi[0].min(nx - 1), hi[1].min(ny - 1), hi[2].min(nz - 1)];
    if lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2] {
        return;
    }
    match plane {
        ViewPlane::Axial => {
            let k = slice.min(nz - 1);
            if k < lo[2] || k > hi[2] {
                return;
            }
            let base = k * nx * ny;
            for j in lo[1]..=hi[1] {
                let row = base + j * nx;
                let src = &mask[row + lo[0]..=row + hi[0]];
                let dst = &mut out[j * nx + lo[0]..=j * nx + hi[0]];
                for (o, &m) in dst.iter_mut().zip(src) {
                    if m != 0 {
                        *o = col;
                    }
                }
            }
        }
        ViewPlane::Sagittal => {
            let i = slice.min(nx - 1);
            if i < lo[0] || i > hi[0] {
                return;
            }
            for k in lo[2]..=hi[2] {
                let r = nz - 1 - k;
                let base = k * nx * ny + i;
                let row = &mut out[r * ny..(r + 1) * ny];
                for j in lo[1]..=hi[1] {
                    if mask[base + j * nx] != 0 {
                        row[j] = col;
                    }
                }
            }
        }
        ViewPlane::Coronal => {
            let j = slice.min(ny - 1);
            if j < lo[1] || j > hi[1] {
                return;
            }
            for k in lo[2]..=hi[2] {
                let r = nz - 1 - k;
                let base = k * nx * ny + j * nx;
                let src = &mask[base + lo[0]..=base + hi[0]];
                let dst = &mut out[r * nx + lo[0]..=r * nx + hi[0]];
                for (o, &m) in dst.iter_mut().zip(src) {
                    if m != 0 {
                        *o = col;
                    }
                }
            }
        }
    }
}

/// [`overlay_slice`] for a whole label map at once: every voxel whose label
/// has a colour in `colors` (indexed by label, `None` for a label not drawn)
/// is painted in it. One pass over the slice whatever the number of classes,
/// which is what previewing the hundred classes of an automatic run as they
/// are ticked needs; the plane and row conventions are [`overlay_slice`]'s.
pub fn overlay_labels_slice(
    labels: &[u8],
    dims: [usize; 3],
    plane: ViewPlane,
    slice: usize,
    colors: &[Option<Color32>; 256],
    out: &mut [Color32],
) {
    let [nx, ny, nz] = dims;
    if nx == 0 || ny == 0 || nz == 0 || labels.len() != nx * ny * nz {
        return;
    }
    let paint = |o: &mut Color32, l: u8| {
        if let Some(c) = colors[l as usize] {
            *o = c;
        }
    };
    match plane {
        ViewPlane::Axial => {
            let k = slice.min(nz - 1);
            let src = &labels[k * nx * ny..(k + 1) * nx * ny];
            for (o, &l) in out.iter_mut().zip(src) {
                paint(o, l);
            }
        }
        ViewPlane::Sagittal => {
            let i = slice.min(nx - 1);
            for k in 0..nz {
                let r = nz - 1 - k;
                for j in 0..ny {
                    paint(&mut out[r * ny + j], labels[k * nx * ny + j * nx + i]);
                }
            }
        }
        ViewPlane::Coronal => {
            let j = slice.min(ny - 1);
            for k in 0..nz {
                let r = nz - 1 - k;
                let src = &labels[k * nx * ny + j * nx..k * nx * ny + (j + 1) * nx];
                for (o, &l) in out[r * nx..(r + 1) * nx].iter_mut().zip(src) {
                    paint(o, l);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Seeded organ growing (interactive "smart" segmentation)
// ---------------------------------------------------------------------------
//
// Geodesic fast marching rather than a plain intensity-threshold flood fill:
// a Dijkstra front expands from the seed, and the cost of traversing a voxel
// rises exponentially both with its intensity deviation from the seed
// statistics and with the local gradient magnitude. Organ boundaries (edges,
// fat planes, tissue transitions) therefore act as barriers - the organ
// under the seed fills long before the front leaks into neighboring tissue,
// which a pure threshold grow cannot distinguish. The user's drag selects an
// arrival-time (geodesic reach) threshold; inside homogeneous tissue the
// cost is ≈1 per mm, so the reach reads roughly as a radius in mm.
//
// The front is expanded *incrementally*: dragging up pops more of the same
// priority queue, dragging down truncates the already-accepted prefix
// (Dijkstra accepts in nondecreasing time), so the live preview never
// recomputes from scratch.

/// Hard cap on the accepted region (bounds memory and drag latency).
const GROW_MAX_VOXELS: usize = 8_000_000;

/// Half-extent of the working box around the seed, mm.
const GROW_BOX_MM: f64 = 130.0;

/// Geodesic reach (≈ mm in homogeneous tissue) at drag level 1.0.
pub const GROW_BASE_REACH: f32 = 15.0;

/// Incremental geodesic region growing from a seed voxel.
#[derive(Default)]
pub struct GrowState {
    /// Working-box origin (volume voxel coords) and dimensions.
    lo: [usize; 3],
    bdims: [usize; 3],
    /// Best known arrival time per box voxel (INFINITY = untouched).
    times: Vec<f32>,
    /// Tentative front: (arrival-time bits, box index), min-first.
    heap: BinaryHeap<std::cmp::Reverse<(u32, u32)>>,
    /// Accepted voxels in nondecreasing-time order: (time, volume index).
    accepted: Vec<(f32, u32)>,
    /// Seed statistics: mean and 1 / (2.5 σ) of the local neighborhood.
    mu: f32,
    inv_sigma: f32,
    /// Voxels the front may enter at all: the limiting structure, as a
    /// volume-sized mask. `None` means the whole image.
    ///
    /// A limit is not a cheaper barrier than the geodesic cost, it is a
    /// different kind of statement: *never here, whatever the picture
    /// says*. Bone inside the body, contrast inside the liver, a target
    /// inside a box drawn for the purpose.
    limit: Option<std::sync::Arc<Vec<u8>>>,
    /// Current selection: the prefix of `accepted` below the drag threshold.
    pub voxels: Vec<u32>,
    /// The voxel cap was hit - the region would grow further.
    pub capped: bool,
}

impl GrowState {
    /// Confine every following grow to `limit` (a volume-sized mask), or
    /// to nothing when it is `None`. Set before [`GrowState::seed`].
    pub fn set_limit(&mut self, limit: Option<std::sync::Arc<Vec<u8>>>) {
        self.limit = limit;
    }

    /// Whether a voxel index is inside the limit (always true without one).
    #[inline]
    fn allowed(&self, idx: usize) -> bool {
        match &self.limit {
            Some(m) => m.get(idx).copied().unwrap_or(0) != 0,
            None => true,
        }
    }

    /// Start a grow at `seed`: estimate local statistics, reset the front
    /// and expand to the base reach ([`GROW_BASE_REACH`], drag level 1).
    pub fn seed(&mut self, vol: &Volume, seed: [usize; 3]) {
        let [nx, ny, nz] = vol.dims;
        self.lo = [0; 3];
        self.bdims = [0; 3];
        self.times.clear();
        self.heap.clear();
        self.accepted.clear();
        self.voxels.clear();
        self.capped = false;
        if seed[0] >= nx || seed[1] >= ny || seed[2] >= nz {
            return;
        }

        // Robust seed statistics from a 5×5×3 neighborhood: median and MAD.
        let mut samples: Vec<f32> = Vec::with_capacity(75);
        for dk in -1i64..=1 {
            for dj in -2i64..=2 {
                for di in -2i64..=2 {
                    if let Some(v) = vol.get(
                        seed[0] as i64 + di,
                        seed[1] as i64 + dj,
                        seed[2] as i64 + dk,
                    ) {
                        samples.push(v as f32);
                    }
                }
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mu = samples[samples.len() / 2];
        let mut devs: Vec<f32> = samples.iter().map(|v| (v - mu).abs()).collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sigma = (1.4826 * devs[devs.len() / 2]).clamp(12.0, 150.0);
        self.mu = mu;
        self.inv_sigma = 1.0 / (2.5 * sigma);

        // Working box around the seed.
        for (ax, &s) in seed.iter().enumerate() {
            let r = (GROW_BOX_MM / vol.spacing[ax]).ceil() as usize;
            let n = vol.dims[ax];
            self.lo[ax] = s.saturating_sub(r);
            self.bdims[ax] = ((s + r + 1).min(n)) - self.lo[ax];
        }
        let bn = self.bdims[0] * self.bdims[1] * self.bdims[2];
        self.times.resize(bn, f32::INFINITY);

        let b0 = self.box_index(seed);
        self.times[b0] = 0.0;
        self.heap
            .push(std::cmp::Reverse((0f32.to_bits(), b0 as u32)));
        self.expand_until(vol, GROW_BASE_REACH);
        self.select(GROW_BASE_REACH);
    }

    /// Set the drag level (multiplier on the base reach) and update the
    /// selection, expanding the front further if needed.
    pub fn set_level(&mut self, vol: &Volume, level: f32) {
        if self.times.is_empty() {
            return;
        }
        let t = GROW_BASE_REACH * level.max(1e-3);
        self.expand_until(vol, t);
        self.select(t);
    }

    /// Drop the large working buffers (called on commit / cancel).
    pub fn release(&mut self) {
        self.times = Vec::new();
        self.heap = BinaryHeap::new();
        self.accepted = Vec::new();
        self.voxels.clear();
        self.capped = false;
    }

    #[inline]
    fn box_index(&self, v: [usize; 3]) -> usize {
        (v[2] - self.lo[2]) * self.bdims[0] * self.bdims[1]
            + (v[1] - self.lo[1]) * self.bdims[0]
            + (v[0] - self.lo[0])
    }

    /// Pop the front until its earliest arrival time exceeds `t`.
    ///
    /// Step cost from voxel a to neighbor b (per mm): 1 inside tissue that
    /// matches the seed statistics, times an exponential penalty on b's
    /// deviation from the seed mean, plus an exponential penalty on the
    /// intensity jump of the a→b crossing itself. The jump term puts the
    /// barrier exactly on the organ boundary, so the organ's own outer
    /// voxel shell is still cheap to enter from the inside.
    fn expand_until(&mut self, vol: &Volume, t: f32) {
        let [nx, ny, _] = vol.dims;
        let sl = nx * ny;
        while let Some(&std::cmp::Reverse((tb, b))) = self.heap.peek() {
            let time = f32::from_bits(tb);
            if time > t {
                break;
            }
            if self.accepted.len() >= GROW_MAX_VOXELS {
                self.capped = true;
                break;
            }
            self.heap.pop();
            let b = b as usize;
            if time > self.times[b] {
                continue; // stale entry
            }
            let bi = b % self.bdims[0];
            let bj = (b / self.bdims[0]) % self.bdims[1];
            let bk = b / (self.bdims[0] * self.bdims[1]);
            let v = [self.lo[0] + bi, self.lo[1] + bj, self.lo[2] + bk];
            self.accepted
                .push((time, (v[2] * sl + v[1] * nx + v[0]) as u32));
            let va = vol.index(v[0], v[1], v[2]) as f32;

            for (ax, dir) in [(0, -1i64), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
                let mut nb = [v[0] as i64, v[1] as i64, v[2] as i64];
                nb[ax] += dir;
                if nb[ax] < self.lo[ax] as i64 || nb[ax] >= (self.lo[ax] + self.bdims[ax]) as i64 {
                    continue;
                }
                let nv = [nb[0] as usize, nb[1] as usize, nb[2] as usize];
                if !self.allowed(nv[2] * sl + nv[1] * nx + nv[0]) {
                    continue;
                }
                let nbx = self.box_index(nv);
                if self.times[nbx] <= time {
                    continue; // already settled cheaper
                }
                let s = vol.spacing[ax] as f32;
                let vb = vol.index(nv[0], nv[1], nv[2]) as f32;
                let dev = (vb - self.mu) * self.inv_sigma;
                let intensity = (dev * dev).min(9.0).exp();
                let edge = (((vb - va).abs() / s) / 60.0).min(9.0).exp();
                let nt = time + s * (intensity + edge - 1.0);
                if nt < self.times[nbx] {
                    self.times[nbx] = nt;
                    self.heap
                        .push(std::cmp::Reverse((nt.to_bits(), nbx as u32)));
                }
            }
        }
    }

    /// Select the accepted prefix with arrival time ≤ `t` into `voxels`.
    fn select(&mut self, t: f32) {
        let n = self.accepted.partition_point(|&(time, _)| time <= t);
        self.voxels.clear();
        self.voxels
            .extend(self.accepted[..n].iter().map(|&(_, v)| v));
    }
}

/// Fill background holes of a voxel selection that are fully enclosed
/// within an axial slice (vessels, calcifications, …), so a committed
/// grow yields a solid organ. Extends `voxels` with the filled cells.
pub fn fill_holes_slicewise(voxels: &mut Vec<u32>, dims: [usize; 3]) {
    if voxels.is_empty() {
        return;
    }
    let [nx, ny, _] = dims;
    let sl = nx * ny;
    let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
    for &v in voxels.iter() {
        let idx = v as usize;
        let c = [idx % nx, (idx % sl) / nx, idx / sl];
        for ax in 0..3 {
            lo[ax] = lo[ax].min(c[ax]);
            hi[ax] = hi[ax].max(c[ax]);
        }
    }
    // One-cell padding guarantees the outer background is connected.
    let w = hi[0] - lo[0] + 3;
    let h = hi[1] - lo[1] + 3;
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); hi[2] - lo[2] + 1];
    for &v in voxels.iter() {
        buckets[v as usize / sl - lo[2]].push(v);
    }
    let mut occ = vec![false; w * h];
    let mut reach = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    for (dk, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        occ.fill(false);
        reach.fill(false);
        for &v in bucket {
            let idx = v as usize;
            occ[(idx % sl / nx - lo[1] + 1) * w + (idx % nx - lo[0] + 1)] = true;
        }
        stack.push(0);
        reach[0] = true;
        while let Some(p) = stack.pop() {
            let (x, y) = (p % w, p / w);
            for (qx, qy) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if qx < w && qy < h {
                    let q = qy * w + qx;
                    if !reach[q] && !occ[q] {
                        reach[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        let k = lo[2] + dk;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let p = y * w + x;
                if !occ[p] && !reach[p] {
                    voxels.push((k * sl + (lo[1] + y - 1) * nx + (lo[0] + x - 1)) as u32);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The contour tools on a mask slice
// ---------------------------------------------------------------------------
//
// The drawing strip's tools act on whatever is selected: an RT structure's
// contours or a segment's mask. A closed curve (polygon, spline, freehand,
// live wire) and a nudge are contour operations, so on a mask they go
// through the slice's outline - the marching-squares trace a generator
// makes ([`crate::contours::region_of_mask`]) - and land back as the voxels
// whose centre the curve encloses ([`crate::contours::Region::rasterize`]).
// Everything here works on one slice raster, `w` x `h`, in the slice's own
// `(u, v)` ([`Segmentation::slice`]).

/// The *auto* rule of the drawing tools on a mask slice: a ring that
/// crosses the slice's outline and starts outside it cuts, anything else
/// adds - [`crate::contours::auto_cuts`] asked of the outline the slice has.
pub fn slice_auto_cuts(slice: &[u8], w: usize, h: usize, ring: &crate::contours::Poly) -> bool {
    let region = crate::contours::region_of_mask(slice, w, h);
    crate::contours::auto_cuts(&region, ring)
}

/// Fill a closed ring into a mask slice, or cut it out (`cut`): the cells
/// whose centre the ring encloses are set, or cleared. Returns whether any
/// cell changed.
pub fn ring_into_slice(
    slice: &mut [u8],
    w: usize,
    h: usize,
    ring: &crate::contours::Poly,
    cut: bool,
) -> bool {
    if ring.is_empty() || slice.len() != w * h {
        return false;
    }
    let inside = crate::contours::Region::from_ring(ring.clone()).rasterize(w, h);
    let want = u8::from(!cut);
    let mut any = false;
    for (m, &r) in slice.iter_mut().zip(&inside) {
        if r != 0 && *m != want {
            *m = want;
            any = true;
        }
    }
    any
}

/// A nudge stroke on one slice of a mask: the slice's outline, read once
/// when the stroke starts, pushed by every sample of the drag and filled
/// back into the slice each time.
///
/// The outline is kept for the whole stroke, not read again per sample,
/// because a sample moves the pointer by a fraction of a voxel: read back
/// from the voxels each time, every such push would be rounded away and the
/// drag would move nothing. Kept, the pushes add up the way they do on a
/// structure's contour, and the voxels follow. The outline is resampled to
/// half a voxel, since the trace keeps only the corners of a straight run
/// and the push moves vertices.
pub struct OutlineNudge {
    region: crate::contours::Region,
    w: usize,
    h: usize,
}

impl OutlineNudge {
    /// The stroke's outline, of the slice as it is when it starts; `None`
    /// when the slice is empty (there is nothing to push).
    pub fn new(slice: &[u8], w: usize, h: usize) -> Option<OutlineNudge> {
        if slice.len() != w * h {
            return None;
        }
        let mut region = crate::contours::region_of_mask(slice, w, h);
        if region.is_empty() {
            return None;
        }
        for r in &mut region.rings {
            *r = r.resample(0.5);
        }
        Some(OutlineNudge { region, w, h })
    }

    /// One sample of the drag, `from` to `to` in the slice's `(u, v)`: every
    /// outline vertex within `radius_mm` moves with it by the cosine falloff
    /// of [`crate::contours::Region::nudge`] (`mm` the millimetres per unit
    /// of `u` and `v`). The slice the outline now fills, when anything moved.
    pub fn push(
        &mut self,
        from: crate::contours::Pt,
        to: crate::contours::Pt,
        radius_mm: f64,
        mm: crate::contours::Pt,
    ) -> Option<Vec<u8>> {
        if !self.region.nudge(from, to, radius_mm, mm) {
            return None;
        }
        Some(self.region.rasterize(self.w, self.h))
    }
}

// ---------------------------------------------------------------------------
// RTSTRUCT contours → mask
// ---------------------------------------------------------------------------

/// Rasterize an RTSTRUCT ROI onto a volume's own voxel grid.
///
/// The inverse of [`mask_to_roi`], and the bridge every feature that works
/// on *voxels* needs when the user points at a *contour*: restricting a
/// registration to one structure, propagating a structure through a
/// deformation, measuring how far a structure moved.
///
/// Contours are grouped by the slice they fall on and each slice is filled
/// once with the even-odd rule over **all** of that slice's contours
/// together - which is what makes a doughnut a doughnut rather than a disc,
/// since RTSTRUCT expresses a hole as a second contour inside the first.
/// Returns `None` when the ROI has no planar contour inside the volume.
///
/// A contour that encloses no area is dropped before any of that: a
/// zero-width sliver, a loop that crosses itself. These are not points -
/// their outlines still sweep across voxel centres, so an even-odd fill
/// would paint voxels out of geometry that describes nothing. The contour
/// machinery drops them under the same rule
/// ([`crate::contours::Region::normalize`]), and a mask has to agree with
/// the geometry it was made from.
///
/// A path that passes through one of its own vertices again is first cut
/// into the loops it is made of ([`crate::contours::simple_loops`]), and
/// the rule applies to each. Some exporters chain the voxels of a target
/// that touch at a corner into one path through that corner; a pair one
/// above the other comes out with the second square drawn the opposite way
/// round, and the pair's shoelace area as a whole is zero. The structure
/// that first brought zero-area contours here had 450 of them, and they
/// were taken for debris and dropped - when they were 900 of its 2899
/// voxels. Cut at the shared corner they are two squares and count. A
/// sliver that runs out and back along itself still comes apart into
/// nothing.
/// Below this, in voxel units, a contour encloses nothing and is debris.
/// The same threshold [`crate::contours`] drops its rings at, so the mask
/// and the contours agree on what is there.
const MIN_CONTOUR_AREA: f64 = 1e-6;

/// The area a closed contour encloses, in voxel units, by the shoelace
/// formula on its (i, j) coordinates. Unsigned: which way round it was
/// drawn is not the question here.
fn enclosed_area(idx: &[[f64; 3]]) -> f64 {
    let n = idx.len();
    let mut s = 0.0;
    for a in 0..n {
        let p = idx[a];
        let q = idx[(a + 1) % n];
        s += p[0] * q[1] - q[0] * p[1];
    }
    (s * 0.5).abs()
}

pub fn rasterize_roi(grid: &Grid, roi: &Roi) -> Option<Vec<u8>> {
    let [nx, ny, nz] = grid.dims;
    // Slice index → the contours drawn on it, in voxel (i, j) coordinates.
    let mut by_slice: Vec<Vec<Vec<[f64; 2]>>> = vec![Vec::new(); nz];
    for c in &roi.contours {
        if c.geometric_type == "POINT" || c.points.len() < 3 {
            continue;
        }
        let idx: Vec<[f64; 3]> = c.points.iter().map(|p| grid.patient_to_voxel(*p)).collect();
        // A planar contour lies in one slice; take the nearest one.
        let mean_k = idx.iter().map(|v| v[2]).sum::<f64>() / idx.len() as f64;
        let k = mean_k.round();
        if k < 0.0 || k >= nz as f64 {
            continue;
        }
        // One path may be several loops through a shared vertex; each is
        // judged, and filled, on its own.
        for ring in crate::contours::simple_loops(&idx) {
            if enclosed_area(&ring) <= MIN_CONTOUR_AREA {
                continue;
            }
            by_slice[k as usize].push(ring.iter().map(|v| [v[0], v[1]]).collect());
        }
    }
    if by_slice.iter().all(|s| s.is_empty()) {
        return None;
    }

    if nx * ny == 0 {
        return None;
    }
    // Every slice is filled on its own, so the slices go in parallel; the
    // result does not depend on the order they finish in.
    let mut mask = vec![0u8; nx * ny * nz];
    let any = mask
        .par_chunks_mut(nx * ny)
        .zip(by_slice.par_iter())
        .map(|(plane, polys)| fill_slice(plane, polys, nx, ny))
        .reduce(|| false, |a, b| a || b);
    any.then_some(mask)
}

/// Scan-convert one slice's contours (voxel (i, j) coordinates) into its
/// plane of the mask. Whether any voxel was set.
fn fill_slice(plane: &mut [u8], polys: &[Vec<[f64; 2]>], nx: usize, ny: usize) -> bool {
    if polys.is_empty() {
        return false;
    }
    let mut any = false;
    let mut crossings: Vec<f64> = Vec::new();
    // Rows the contours of this slice can possibly touch.
    let (mut j0, mut j1) = (f64::MAX, f64::MIN);
    for poly in polys {
        for p in poly {
            j0 = j0.min(p[1]);
            j1 = j1.max(p[1]);
        }
    }
    let j_start = j0.floor().max(0.0) as usize;
    let j_end = (j1.ceil() as i64).clamp(0, ny as i64 - 1) as usize;
    for j in j_start..=j_end {
        // Scanline through the *centre* of the row, so a voxel counts as
        // inside when its centre is - the rule `mask_to_roi` reverses,
        // and the one every planning system uses.
        let y = j as f64;
        crossings.clear();
        for poly in polys {
            let n = poly.len();
            for a in 0..n {
                let p = poly[a];
                let q = poly[(a + 1) % n];
                // Half-open edge test: a vertex exactly on the scanline
                // is counted once, never twice, so the parity holds.
                if (p[1] <= y) != (q[1] <= y) {
                    let t = (y - p[1]) / (q[1] - p[1]);
                    crossings.push(p[0] + t * (q[0] - p[0]));
                }
            }
        }
        if crossings.len() < 2 {
            continue;
        }
        crossings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for span in crossings.as_chunks::<2>().0 {
            let from = span[0].ceil().max(0.0) as i64;
            let to = (span[1].floor() as i64).min(nx as i64 - 1);
            if from > to {
                continue;
            }
            let row = j * nx;
            for i in from..=to {
                plane[row + i as usize] = 1;
                any = true;
            }
        }
    }
    any
}

// ---------------------------------------------------------------------------
// Mask → RTSTRUCT contours
// ---------------------------------------------------------------------------

/// Convert a segmentation mask into an RTSTRUCT ROI: on every axial slice
/// the outline of the voxels themselves, along their edges (padded so
/// border voxels close), collinear points merged, mapped to patient
/// coordinates.
///
/// The outline encloses exactly the voxels, so the ROI's planimetric volume
/// is the mask's voxel volume and [`rasterize_roi`] gives the mask back. A
/// marching-squares outline would look smoother and cut every corner, which
/// a large organ does not notice and a small or scattered structure does: a
/// target carried onto a 1.2 mm lattice as a cloud of single voxels lost a
/// tenth of its planimetric volume that way.
pub fn mask_to_roi(seg: &Segmentation, grid: &Grid, number: i32) -> Roi {
    let [nx, ny, _] = seg.dims;
    let mut contours = Vec::new();
    if let Some((lo, hi)) = seg.bbox {
        let mut field = crate::contours::PaddedField::new(nx, ny);
        for k in lo[2]..=hi[2] {
            field.clear();
            let base = k * nx * ny;
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    if seg.mask[base + j * nx + i] != 0 {
                        field.set(i, j);
                    }
                }
            }
            for pts in field.trace_voxels() {
                let points: Vec<Vec3> = pts
                    .iter()
                    .map(|p| grid.voxel_to_patient(p[0] as f64, p[1] as f64, k as f64))
                    .collect();
                contours.push(Contour {
                    points,
                    geometric_type: "CLOSED_PLANAR".into(),
                });
            }
        }
    }
    Roi {
        number,
        name: seg.name.clone(),
        color: seg.color,
        roi_type: "ORGAN".into(),
        description: String::new(),
        contours,
    }
}

#[cfg(test)]
mod raster_tests {
    use super::*;
    use crate::geometry::Vec3;

    fn grid() -> Grid {
        Grid {
            dims: [32, 32, 3],
            spacing: [1.0, 1.0, 1.0],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        }
    }

    fn roi(contours: Vec<Contour>) -> Roi {
        Roi {
            number: 1,
            name: "probe".into(),
            color: [255, 0, 0],
            roi_type: "GTV".into(),
            description: String::new(),
            contours,
        }
    }

    fn closed(points: Vec<Vec3>) -> Contour {
        Contour {
            points,
            geometric_type: "CLOSED_PLANAR".into(),
        }
    }

    fn filled(roi: &Roi) -> usize {
        rasterize_roi(&grid(), roi)
            .map(|m| m.iter().filter(|v| **v != 0).count())
            .unwrap_or(0)
    }

    /// Two voxels written as one path through the corner they share - the
    /// first square, then the second the other way round - are two voxels,
    /// not a zero-area contour to drop.
    #[test]
    fn a_path_of_two_squares_through_a_shared_corner_paints_both() {
        let p = |x: f64, y: f64| Vec3::new(x, y, 1.0);
        let figure_eight = closed(vec![
            p(4.5, 5.5),
            p(4.5, 4.5),
            p(5.5, 4.5),
            p(5.5, 5.5),
            p(4.5, 5.5),
            p(4.5, 6.5),
            p(5.5, 6.5),
            p(5.5, 5.5),
            p(4.5, 5.5),
        ]);
        let r = roi(vec![figure_eight]);
        assert_eq!(filled(&r), 2);
        let plan = crate::contours::Stack::from_roi(&r, &grid()).volume_cm3(grid().spacing);
        assert!((plan - 0.002).abs() < 1e-12, "{plan}");
    }

    /// A mask made contours has the planimetric volume of its voxels and
    /// rasterizes back to the same voxels - here a scattered cloud of single
    /// voxels and small clumps, which is where cutting corners cost a tenth
    /// of a propagated target's volume.
    #[test]
    fn a_mask_made_contours_keeps_its_voxels_exactly() {
        let g = grid();
        let [nx, ny, nz] = g.dims;
        let mut mask = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 4..28 {
                for i in 4..28 {
                    if (i * 7 + j * 13 + k * 5) % 5 == 0 || (i / 4 + j / 3) % 4 == 0 {
                        mask[k * nx * ny + j * nx + i] = 1;
                    }
                }
            }
        }
        let n = mask.iter().filter(|v| **v != 0).count();
        let seg = Segmentation::from_label_map("cloud".into(), [1, 2, 3], g.dims, &mask, 1);
        let roi = mask_to_roi(&seg, &g, 1);
        let back = rasterize_roi(&g, &roi).expect("it rasterizes");
        assert_eq!(back, mask, "the same voxels come back");
        let plan = crate::contours::Stack::from_roi(&roi, &g).volume_cm3(g.spacing);
        let vox = n as f64 * g.spacing.iter().product::<f64>() / 1000.0;
        assert!(
            (plan - vox).abs() < 1e-9,
            "planimetry {plan} against voxels {vox}"
        );
    }

    /// A contour that encloses nothing paints nothing - even though its
    /// outline crosses voxel centres, which an even-odd fill would count.
    #[test]
    fn a_contour_enclosing_no_area_paints_no_voxels() {
        let z = 1.0;
        let square = roi(vec![closed(vec![
            Vec3::new(10.0, 10.0, z),
            Vec3::new(14.0, 10.0, z),
            Vec3::new(14.0, 14.0, z),
            Vec3::new(10.0, 14.0, z),
        ])]);
        let n = filled(&square);
        assert!(n > 0, "a real square fills voxels");

        // A zero-width sliver: out and back along the same line.
        let sliver = roi(vec![closed(vec![
            Vec3::new(4.0, 20.0, z),
            Vec3::new(9.0, 20.0, z),
            Vec3::new(4.0, 20.0, z),
        ])]);
        assert_eq!(filled(&sliver), 0, "a sliver encloses nothing");

        // A figure of eight: the two lobes cancel to nil signed area, and
        // the outline still sweeps several voxels.
        let eight = roi(vec![closed(vec![
            Vec3::new(20.0, 4.0, z),
            Vec3::new(24.0, 4.0, z),
            Vec3::new(20.0, 8.0, z),
            Vec3::new(24.0, 8.0, z),
        ])]);
        assert_eq!(filled(&eight), 0, "a self-crossing loop encloses nothing");

        // And the debris does not inflate a sound structure it travels with.
        let together = roi(vec![
            square.contours[0].clone(),
            sliver.contours[0].clone(),
            eight.contours[0].clone(),
        ]);
        assert_eq!(
            filled(&together),
            n,
            "the sound contour measures the same with debris beside it"
        );
    }
}

#[cfg(test)]
mod overlay_tests {
    use super::*;

    /// A sparse mask with a known extent, filled pseudo-randomly inside it.
    fn blob(dims: [usize; 3], lo: [usize; 3], hi: [usize; 3]) -> Vec<u8> {
        let [nx, ny, nz] = dims;
        let mut m = vec![0u8; nx * ny * nz];
        let mut s = 0x9E37_79B9_u32;
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    m[k * nx * ny + j * nx + i] = u8::from(s.is_multiple_of(3));
                }
            }
        }
        m
    }

    #[test]
    fn the_extent_only_saves_work_and_never_changes_a_pixel() {
        let dims = [23, 17, 11];
        let (lo, hi) = ([4, 3, 2], [15, 12, 8]);
        let mask = blob(dims, lo, hi);
        let (exact, _) = crate::morphology::mask_extent(&mask, dims);
        // The stored extent may be larger than the voxels (after erasing).
        let loose = Some(([1, 0, 1], [20, 16, 10]));
        for (plane, n, w, h) in [
            (ViewPlane::Axial, dims[2], dims[0], dims[1]),
            (ViewPlane::Sagittal, dims[0], dims[1], dims[2]),
            (ViewPlane::Coronal, dims[1], dims[0], dims[2]),
        ] {
            for slice in 0..n {
                let mut full = vec![Color32::TRANSPARENT; w * h];
                overlay_slice(&mask, dims, plane, slice, [1, 2, 3], 90, &mut full);
                for bbox in [exact, loose] {
                    let mut boxed = vec![Color32::TRANSPARENT; w * h];
                    overlay_slice_within(
                        &mask,
                        dims,
                        bbox,
                        plane,
                        slice,
                        [1, 2, 3],
                        90,
                        &mut boxed,
                    );
                    assert_eq!(full, boxed, "{plane:?} slice {slice} with {bbox:?}");
                }
            }
        }
    }

    /// A label map previewed in one pass paints what the classes' own
    /// masks would, one after the other, on every plane; an untinted class
    /// paints nothing.
    #[test]
    fn a_label_map_overlays_as_its_masks_would() {
        let dims = [13, 9, 7];
        let a = blob(dims, [1, 1, 1], [8, 6, 5]);
        let b = blob(dims, [5, 2, 0], [12, 8, 6]);
        // Label 2 wins where both are set, label 3 is never drawn.
        let labels: Vec<u8> = a
            .iter()
            .zip(&b)
            .enumerate()
            .map(|(n, (&x, &y))| {
                if y != 0 {
                    2
                } else if x != 0 {
                    1
                } else {
                    3 * (n % 2) as u8
                }
            })
            .collect();
        let mask = |l: u8| -> Vec<u8> { labels.iter().map(|&v| u8::from(v == l)).collect() };
        let c1 = Color32::from_rgba_unmultiplied(9, 8, 7, 90);
        let c2 = Color32::from_rgba_unmultiplied(1, 2, 3, 90);
        let mut colors = [None; 256];
        colors[1] = Some(c1);
        colors[2] = Some(c2);
        for (plane, n, w, h) in [
            (ViewPlane::Axial, dims[2], dims[0], dims[1]),
            (ViewPlane::Sagittal, dims[0], dims[1], dims[2]),
            (ViewPlane::Coronal, dims[1], dims[0], dims[2]),
        ] {
            for slice in 0..n {
                let mut want = vec![Color32::TRANSPARENT; w * h];
                overlay_slice(&mask(1), dims, plane, slice, [9, 8, 7], 90, &mut want);
                overlay_slice(&mask(2), dims, plane, slice, [1, 2, 3], 90, &mut want);
                let mut got = vec![Color32::TRANSPARENT; w * h];
                overlay_labels_slice(&labels, dims, plane, slice, &colors, &mut got);
                assert_eq!(want, got, "{plane:?} slice {slice}");
            }
        }
    }

    /// Undoing an edit that shrank a segment grows its box back over the
    /// voxels it restores: everything that reads only the box (the overlay,
    /// meshing, a segment taken into a workflow) sees the whole segment.
    #[test]
    fn undo_grows_the_box_back() {
        let dims = [16, 12, 10];
        let whole = blob(dims, [2, 3, 1], [13, 9, 8]);
        let mut seg = Segmentation::from_label_map("s".into(), [1, 2, 3], dims, &whole, 1);
        let before = (seg.bbox, seg.count);
        let mut small = vec![0u8; whole.len()];
        let at = 5 * 16 * 12 + 6 * 16 + 7;
        small[at] = 1;
        seg.replace_mask(small);
        assert_eq!(seg.bbox, Some(([7, 6, 5], [7, 6, 5])));
        assert!(seg.undo_last());
        assert_eq!(seg.mask, whole);
        assert_eq!((seg.bbox, seg.count), before);
    }
}

#[cfg(test)]
mod slice_tool_tests {
    use super::*;
    use crate::contours::Poly;
    use crate::geometry::Vec3;

    fn square(x: f64, y: f64, s: f64) -> Poly {
        Poly::new(vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]])
    }

    fn set_count(m: &[u8]) -> usize {
        m.iter().filter(|&&v| v != 0).count()
    }

    /// A slice read and written along each axis is the same voxels, and a
    /// write is one undo step like a brush stroke, with the count and the
    /// box kept.
    #[test]
    fn a_slice_reads_and_writes_back_on_every_axis_with_undo() {
        let dims = [7, 5, 4];
        let mut seg = Segmentation::new("s".into(), [1, 2, 3], dims);
        for (axis, &n) in dims.iter().enumerate() {
            let [w, h] = seg.slice_dims(axis);
            let level = 2.min(n - 1);
            let mut s = seg.slice(axis, level);
            assert_eq!(s.len(), w * h);
            s[w + 1] = 1;
            s[(h - 1) * w + w - 1] = 1;
            let before = seg.count;
            assert!(seg.write_slice(axis, level, &s));
            assert_eq!(seg.count, before + 2);
            assert_eq!(seg.slice(axis, level), s);
            assert!(
                !seg.write_slice(axis, level, &s),
                "the same slice changes nothing"
            );
            seg.end_stroke();
        }
        assert_eq!(seg.count, crate::morphology::count_set(&seg.mask));
        // The written cell of the axial slice is where the axial index says.
        assert_eq!(seg.mask[2 * 7 * 5 + 7 + 1], 1);
        assert!(seg.bbox.is_some());
        for _ in 0..3 {
            assert!(seg.undo_last());
        }
        assert_eq!(seg.count, 0);
        assert!(seg.mask.iter().all(|&v| v == 0));
        // A slice of the wrong size or off the lattice is refused.
        assert!(!seg.write_slice(2, 0, &[1, 1]));
        assert!(!seg.write_slice(2, 9, &[1; 35]));
        assert!(seg.slice(2, 9).is_empty());
    }

    /// A polygon fills the voxels whose centre it encloses and a cut takes
    /// them out again; the auto rule cuts only a ring that crosses the
    /// outline from outside.
    #[test]
    fn a_drawn_ring_lands_in_a_slice_by_the_drawing_rule() {
        let (w, h) = (16usize, 12usize);
        let mut s = vec![0u8; w * h];
        let ring = square(1.5, 1.5, 5.0); // centres 2..=6: 25 cells
        assert!(!slice_auto_cuts(&s, w, h, &ring), "nothing to cross: add");
        assert!(ring_into_slice(&mut s, w, h, &ring, false));
        assert_eq!(set_count(&s), 25);
        assert!(
            !ring_into_slice(&mut s, w, h, &ring, false),
            "already there"
        );

        // Starting outside and crossing in: a cut.
        let bite = Poly::new(vec![[4.5, -1.0], [9.0, -1.0], [9.0, 9.0], [4.5, 9.0]]);
        assert!(slice_auto_cuts(&s, w, h, &bite));
        // Starting inside and crossing out: an extension.
        let grow = Poly::new(vec![[4.0, 4.0], [10.0, 3.0], [10.0, 5.0]]);
        assert!(!slice_auto_cuts(&s, w, h, &grow));
        assert!(ring_into_slice(&mut s, w, h, &bite, true));
        // Columns 5 and 6 of the square are gone.
        assert_eq!(set_count(&s), 15);
        assert_eq!(s[3 * w + 6], 0);
        assert_eq!(s[3 * w + 4], 1);
    }

    /// The edge band paints only the voxels whose value is in it.
    #[test]
    fn the_edge_band_limits_the_voxel_brush_to_its_values() {
        let dims = [12, 12, 3];
        // Left half bone, right half soft tissue.
        let data: Vec<i16> = (0..dims[0] * dims[1] * dims[2])
            .map(|n| if n % dims[0] < 6 { 700 } else { 40 })
            .collect();
        let vol = Volume {
            data,
            dims,
            spacing: [1.0, 1.0, 1.0],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 1000,
        };
        let c = [5.5, 5.5, 1.0];
        let mut all = Segmentation::new("a".into(), [0; 3], dims);
        all.paint_capsule(&vol, c, c, 3.0, false, Some((ViewPlane::Axial, 1)));
        let mut bone = Segmentation::new("b".into(), [0; 3], dims);
        bone.paint_capsule_within(
            &vol,
            c,
            c,
            3.0,
            false,
            Some((ViewPlane::Axial, 1)),
            Some((200.0, f32::MAX)),
        );
        assert!(bone.count > 0 && bone.count < all.count);
        for (i, &m) in bone.mask.iter().enumerate() {
            if m != 0 {
                assert!(i % dims[0] < 6, "voxel {i} is not bone");
                assert_eq!(all.mask[i], 1, "and it is inside the brush");
            }
        }
    }

    /// A nudge pushes the edge of a mask the way it pushes a contour, and
    /// many small pushes add up: the outline is kept across the stroke, so
    /// a sub-voxel step is not rounded away.
    #[test]
    fn a_nudge_pushes_the_mask_edge_in_small_steps() {
        let (w, h) = (30usize, 20usize);
        let mut s = vec![0u8; w * h];
        ring_into_slice(&mut s, w, h, &square(4.5, 4.5, 10.0), false); // 5..=14
        let n0 = set_count(&s);
        let mut nudge = OutlineNudge::new(&s, w, h).expect("an outline");
        // Twenty pushes of a fifth of a voxel at the right edge (x = 14.5).
        let mut now = s.clone();
        for k in 0..20 {
            let from = [14.5 + 0.2 * k as f64, 9.5];
            let to = [from[0] + 0.2, 9.5];
            if let Some(next) = nudge.push(from, to, 4.0, [1.0, 1.0]) {
                now = next;
            }
        }
        assert!(set_count(&now) > n0 + 6, "{} against {n0}", set_count(&now));
        assert_eq!(now[9 * w + 17], 1, "the edge moved right");
        // Far from the pointer nothing changed.
        for j in 0..h {
            for i in 0..12 {
                assert_eq!(now[j * w + i], s[j * w + i], "cell ({i}, {j})");
            }
        }
        assert!(OutlineNudge::new(&vec![0u8; w * h], w, h).is_none());
    }
}
