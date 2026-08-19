//! Interactive segmentation: voxel label masks painted with a 2D/3D brush,
//! seeded region growing ("smart" segmentation, as in MITK's interactive
//! tools), per-stroke undo, and conversion back to RTSTRUCT contours so a
//! drawn segmentation can ride the existing DICOM export pipeline.
//!
//! Everything is pure Rust and CPU-side: the mask is one byte per voxel in
//! the exact index order of [`Volume::data`], so slice overlays reuse the
//! same display conventions as the volume itself.

use std::collections::VecDeque;

use egui::Color32;
use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::render;
use crate::rtstruct::{Contour, Roi};
use crate::volume::{ViewPlane, Volume};

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

/// A meshing snapshot of a mask: `(padded bool grid, grid dims, bbox lo,
/// stride)` — see [`Segmentation::mesh_grid`].
pub type MeshGrid = (Vec<bool>, [usize; 3], [usize; 3], usize);

/// A binary label mask over a study's volume.
pub struct Segmentation {
    pub name: String,
    pub color: [u8; 3],
    pub visible: bool,
    /// One byte per voxel (0 or 1), index order `k * nx * ny + j * nx + i`
    /// — identical to [`Volume::data`].
    pub mask: Vec<u8>,
    pub dims: [usize; 3],
    /// Number of set voxels (kept incrementally).
    pub count: usize,
    /// Bumped on every edit → 2D overlays and 3D meshes rebuild.
    pub gen: u64,
    /// Extent of all voxels ever set (inclusive). May overestimate after
    /// erasing — harmless, it only bounds later scans.
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
            gen: 0,
            bbox: None,
            undo: Vec::new(),
            pending: Vec::new(),
        }
    }

    /// Segmented volume in cm³ for the given voxel spacing (mm).
    pub fn volume_cm3(&self, spacing: [f64; 3]) -> f64 {
        self.count as f64 * spacing[0] * spacing[1] * spacing[2] / 1000.0
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
    /// coords) with the given radius in mm — sweeping between the previous
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
                    let cur = self.mask[idx];
                    if cur != want {
                        self.pending.push((idx as u32, cur));
                        self.mask[idx] = want;
                        self.count = if want == 1 { self.count + 1 } else { self.count - 1 };
                        self.touch(i, j, k);
                        any = true;
                    }
                }
            }
        }
        if any {
            self.gen += 1;
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
            self.gen += 1;
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
        let Some(changes) = self.undo.pop() else { return false };
        for &(idx, old) in changes.iter().rev() {
            let cur = self.mask[idx as usize];
            if cur != old {
                self.count = if old != 0 { self.count + 1 } else { self.count - 1 };
            }
            self.mask[idx as usize] = old;
        }
        self.gen += 1;
        true
    }

    /// Snapshot the mask's bounding box (plus a one-cell empty margin so the
    /// surface closes) as a bool grid ready for surface-nets meshing.
    /// Very large masks are max-pooled with an integer stride to keep the
    /// grid — and thus the rebuild latency — bounded for interactive use.
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
        // Each grid layer max-pools its own voxel slab — layers are independent.
        grid.par_chunks_mut(g[0] * g[1]).enumerate().for_each(|(gk, layer)| {
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
    let [nx, ny, nz] = dims;
    let col = Color32::from_rgba_unmultiplied(color[0], color[1], color[2], alpha);
    match plane {
        ViewPlane::Axial => {
            let k = slice.min(nz.saturating_sub(1));
            let base = k * nx * ny;
            for (o, &m) in out.iter_mut().zip(&mask[base..base + nx * ny]) {
                if m != 0 {
                    *o = col;
                }
            }
        }
        ViewPlane::Sagittal => {
            let i = slice.min(nx.saturating_sub(1));
            for (r, row) in out.chunks_mut(ny).enumerate() {
                let base = (nz - 1 - r) * nx * ny + i;
                for (j, o) in row.iter_mut().enumerate() {
                    if mask[base + j * nx] != 0 {
                        *o = col;
                    }
                }
            }
        }
        ViewPlane::Coronal => {
            let j = slice.min(ny.saturating_sub(1));
            for (r, row) in out.chunks_mut(nx).enumerate() {
                let base = (nz - 1 - r) * nx * ny + j * nx;
                for (i, o) in row.iter_mut().enumerate() {
                    if mask[base + i] != 0 {
                        *o = col;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Seeded region growing (interactive "smart" segmentation)
// ---------------------------------------------------------------------------

/// Reusable buffers for seeded region growing. The BFS is recomputed live
/// while the user drags the tolerance, so allocations are kept across runs.
#[derive(Default)]
pub struct GrowState {
    visited: Vec<u64>,
    queue: VecDeque<u32>,
    /// Linear voxel indices of the grown region (result of the last run).
    pub voxels: Vec<u32>,
    /// The voxel cap was hit — the region would grow further.
    pub capped: bool,
}

impl GrowState {
    /// 6-connected flood fill from `seed` over voxels whose value lies
    /// within ±`tol` of the seed value, capped at `max_voxels`.
    pub fn run(&mut self, vol: &Volume, seed: [usize; 3], tol: f32, max_voxels: usize) {
        let [nx, ny, nz] = vol.dims;
        let n = nx * ny * nz;
        self.visited.clear();
        self.visited.resize(n.div_ceil(64), 0);
        self.queue.clear();
        self.voxels.clear();
        self.capped = false;
        if seed[0] >= nx || seed[1] >= ny || seed[2] >= nz {
            return;
        }
        let sv = vol.index(seed[0], seed[1], seed[2]) as f32;
        let (lo, hi) = (sv - tol, sv + tol);
        let data = &vol.data;
        let sl = nx * ny;
        let start = seed[2] * sl + seed[1] * nx + seed[0];
        self.visited[start / 64] |= 1 << (start % 64);
        self.queue.push_back(start as u32);
        while let Some(cur) = self.queue.pop_front() {
            self.voxels.push(cur);
            if self.voxels.len() >= max_voxels {
                self.capped = true;
                break;
            }
            let c = cur as usize;
            let k = c / sl;
            let r = c % sl;
            let j = r / nx;
            let i = r % nx;
            let mut try_nb = |idx: usize| {
                let w = idx / 64;
                let b = 1u64 << (idx % 64);
                if self.visited[w] & b == 0 {
                    self.visited[w] |= b;
                    let v = data[idx] as f32;
                    if v >= lo && v <= hi {
                        self.queue.push_back(idx as u32);
                    }
                }
            };
            if i > 0 {
                try_nb(c - 1);
            }
            if i + 1 < nx {
                try_nb(c + 1);
            }
            if j > 0 {
                try_nb(c - nx);
            }
            if j + 1 < ny {
                try_nb(c + nx);
            }
            if k > 0 {
                try_nb(c - sl);
            }
            if k + 1 < nz {
                try_nb(c + sl);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mask → RTSTRUCT contours
// ---------------------------------------------------------------------------

/// Convert a segmentation mask into an RTSTRUCT ROI: marching squares on
/// every axial slice (padded so border voxels close), stitched into closed
/// loops, collinear points merged, mapped to patient coordinates.
pub fn mask_to_roi(seg: &Segmentation, vol: &Volume, number: i32) -> Roi {
    let [nx, ny, _] = seg.dims;
    let mut contours = Vec::new();
    if let Some((lo, hi)) = seg.bbox {
        let (pw, ph) = (nx + 2, ny + 2);
        let mut field = vec![0.0f32; pw * ph];
        for k in lo[2]..=hi[2] {
            field.fill(0.0);
            let base = k * nx * ny;
            let mut any = false;
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    if seg.mask[base + j * nx + i] != 0 {
                        field[(j + 1) * pw + i + 1] = 1.0;
                        any = true;
                    }
                }
            }
            if !any {
                continue;
            }
            for pts in stitch_loops(&render::marching_squares(&field, pw, ph, 0.5)) {
                let pts = drop_collinear(pts);
                if pts.len() < 3 {
                    continue;
                }
                let points: Vec<Vec3> = pts
                    .iter()
                    .map(|p| {
                        vol.voxel_to_patient(p[0] as f64 - 1.0, p[1] as f64 - 1.0, k as f64)
                    })
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
        contours,
    }
}

/// Endpoint key for loop stitching. On a binary field every marching-squares
/// endpoint lies exactly on a half-integer, so doubling is lossless.
#[inline]
fn ep_key(p: [f32; 2]) -> (i64, i64) {
    (
        (p[0] * 2.0).round() as i64,
        (p[1] * 2.0).round() as i64,
    )
}

/// Chain unordered marching-squares segments into closed loops.
fn stitch_loops(segs: &[render::Segment]) -> Vec<Vec<[f32; 2]>> {
    use std::collections::HashMap;
    let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (si, s) in segs.iter().enumerate() {
        adj.entry(ep_key(s.0)).or_default().push(si);
        adj.entry(ep_key(s.1)).or_default().push(si);
    }
    let mut used = vec![false; segs.len()];
    let mut out = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let start_key = ep_key(segs[start].0);
        let mut pts = vec![segs[start].0, segs[start].1];
        let mut cur = ep_key(segs[start].1);
        let mut closed = cur == start_key;
        while !closed {
            let next = adj
                .get(&cur)
                .and_then(|c| c.iter().copied().find(|&si| !used[si]));
            let Some(nxt) = next else { break };
            used[nxt] = true;
            let s = &segs[nxt];
            let np = if ep_key(s.0) == cur { s.1 } else { s.0 };
            cur = ep_key(np);
            if cur == start_key {
                closed = true;
            } else {
                pts.push(np);
            }
        }
        if closed && pts.len() >= 3 {
            out.push(pts);
        }
    }
    out
}

/// Remove points that lie on the straight line between their neighbors —
/// marching squares on a binary mask produces long collinear runs.
fn drop_collinear(pts: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    let n = pts.len();
    if n < 4 {
        return pts;
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = pts[(i + n - 1) % n];
        let c = pts[i];
        let q = pts[(i + 1) % n];
        let cross = (c[0] - p[0]) * (q[1] - c[1]) - (c[1] - p[1]) * (q[0] - c[0]);
        if cross.abs() > 1e-4 {
            out.push(c);
        }
    }
    if out.len() >= 3 {
        out
    } else {
        pts
    }
}
