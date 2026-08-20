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

    /// Build a segmentation from one class of a multi-label voxel map
    /// (`labels` in `Volume::data` index order, mask = `labels == label`).
    /// Used by the auto-segmentation pipeline; count and bounding box are
    /// filled in the same pass, there is no undo history.
    pub fn from_label_map(
        name: String,
        color: [u8; 3],
        dims: [usize; 3],
        labels: &[u8],
        label: u8,
    ) -> Self {
        let mut seg = Segmentation::new(name, color, dims);
        debug_assert_eq!(labels.len(), seg.mask.len());
        let [nx, ny, _] = dims;
        let mut count = 0usize;
        for (idx, (m, l)) in seg.mask.iter_mut().zip(labels.iter()).enumerate() {
            if *l == label {
                *m = 1;
                count += 1;
                let k = idx / (nx * ny);
                let r = idx % (nx * ny);
                seg.bbox = match seg.bbox {
                    None => Some(([r % nx, r / nx, k], [r % nx, r / nx, k])),
                    Some((mut lo, mut hi)) => {
                        let (i, j) = (r % nx, r / nx);
                        lo[0] = lo[0].min(i);
                        lo[1] = lo[1].min(j);
                        lo[2] = lo[2].min(k);
                        hi[0] = hi[0].max(i);
                        hi[1] = hi[1].max(j);
                        hi[2] = hi[2].max(k);
                        Some((lo, hi))
                    }
                };
            }
        }
        seg.count = count;
        seg
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
// Seeded organ growing (interactive "smart" segmentation)
// ---------------------------------------------------------------------------
//
// Geodesic fast marching rather than a plain intensity-threshold flood fill:
// a Dijkstra front expands from the seed, and the cost of traversing a voxel
// rises exponentially both with its intensity deviation from the seed
// statistics and with the local gradient magnitude. Organ boundaries (edges,
// fat planes, tissue transitions) therefore act as barriers — the organ
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
    /// Current selection: the prefix of `accepted` below the drag threshold.
    pub voxels: Vec<u32>,
    /// The voxel cap was hit — the region would grow further.
    pub capped: bool,
}

impl GrowState {
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
                    if let Some(v) =
                        vol.get(seed[0] as i64 + di, seed[1] as i64 + dj, seed[2] as i64 + dk)
                    {
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
        self.heap.push(std::cmp::Reverse((0f32.to_bits(), b0 as u32)));
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
            self.accepted.push((time, (v[2] * sl + v[1] * nx + v[0]) as u32));
            let va = vol.index(v[0], v[1], v[2]) as f32;

            for (ax, dir) in [(0, -1i64), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)] {
                let mut nb = [v[0] as i64, v[1] as i64, v[2] as i64];
                nb[ax] += dir;
                if nb[ax] < self.lo[ax] as i64
                    || nb[ax] >= (self.lo[ax] + self.bdims[ax]) as i64
                {
                    continue;
                }
                let nv = [nb[0] as usize, nb[1] as usize, nb[2] as usize];
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
                    self.heap.push(std::cmp::Reverse((nt.to_bits(), nbx as u32)));
                }
            }
        }
    }

    /// Select the accepted prefix with arrival time ≤ `t` into `voxels`.
    fn select(&mut self, t: f32) {
        let n = self.accepted.partition_point(|&(time, _)| time <= t);
        self.voxels.clear();
        self.voxels.extend(self.accepted[..n].iter().map(|&(_, v)| v));
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
