//! 3D surface meshing of RT structure sets and painted segmentation masks
//! for the 3D view windows.
//!
//! Pipeline per ROI (pure Rust, background thread):
//! 1. rasterize the planar contours into a patient-axis-aligned binary mask
//!    (even–odd scanline fill, contours extruded to the nearest mask layers);
//! 2. extract the boundary surface with the **surface nets** algorithm
//!    (one vertex per mixed cell, quads across sign-changing grid edges);
//! 3. Laplacian smoothing (3 iterations) to remove voxel staircase artifacts;
//! 4. area-weighted per-vertex normals for Gouraud shading.
//!
//! Painted segmentations skip step 1 — their voxel mask feeds the same
//! surface-nets pipeline directly via [`mesh_from_mask`].
//!
//! The result is comparable to the segmentation surfaces shown by tools like
//! 3D Slicer (which use flying edges + windowed-sinc smoothing).

use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::loader::Progress;
use crate::rtstruct::{Roi, StructureSet};
use crate::volume::Volume;

/// Raw surface geometry: (vertices, unit normals, triangle indices).
pub type SurfaceMesh = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[u32; 3]>);

pub struct RoiMesh {
    /// Index of the ROI within the structure set (drives visibility).
    pub roi_index: usize,
    pub name: String,
    pub color: [u8; 3],
    /// EXTERNAL / body contour — drawn translucent so interior structures
    /// stay visible (as in 3D Slicer).
    pub external: bool,
    /// Vertex positions in patient coordinates (mm).
    pub verts: Vec<[f32; 3]>,
    /// Unit vertex normals.
    pub normals: Vec<[f32; 3]>,
    /// Triangle indices.
    pub tris: Vec<[u32; 3]>,
}

/// Build surface meshes for every ROI of a structure set (parallel).
pub fn build_meshes(ss: &StructureSet, progress: &Progress) -> Vec<RoiMesh> {
    let n = ss.rois.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let mut meshes: Vec<RoiMesh> = ss
        .rois
        .par_iter()
        .enumerate()
        .filter_map(|(i, roi)| {
            let m = build_roi_mesh(i, roi);
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            progress.set(format!("Meshing structures… {d}/{n}"));
            m
        })
        .collect();
    // Large, opaque structures first is a nicer default draw order.
    meshes.sort_by_key(|m| std::cmp::Reverse(m.tris.len()));
    progress.set("done");
    meshes
}

fn build_roi_mesh(roi_index: usize, roi: &Roi) -> Option<RoiMesh> {
    // Collect usable planar contours.
    let contours: Vec<&crate::rtstruct::Contour> = roi
        .contours
        .iter()
        .filter(|c| c.geometric_type != "POINT" && c.points.len() >= 3)
        .collect();
    if contours.is_empty() {
        return None;
    }

    // Patient-space bounding box.
    let (mut min, mut max) = ([f64::MAX; 3], [f64::MIN; 3]);
    for c in &contours {
        for p in &c.points {
            for (a, v) in [(0, p.x), (1, p.y), (2, p.z)] {
                min[a] = min[a].min(v);
                max[a] = max[a].max(v);
            }
        }
    }
    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    let max_extent = extent.iter().cloned().fold(0.0f64, f64::max).max(1.0);
    // Grid resolution: fine enough for small organs, capped for large ones.
    let h = (max_extent / 110.0).clamp(0.8, 4.0);
    let margin = 2.0 * h;
    let origin = [min[0] - margin, min[1] - margin, min[2] - margin];
    let dims = [
        ((extent[0] + 2.0 * margin) / h).ceil() as usize + 2,
        ((extent[1] + 2.0 * margin) / h).ceil() as usize + 2,
        ((extent[2] + 2.0 * margin) / h).ceil() as usize + 2,
    ];
    let (gx, gy, gz) = (dims[0], dims[1], dims[2]);
    if gx < 3 || gy < 3 || gz < 3 {
        return None;
    }

    // ---- 1. binary mask -----------------------------------------------
    // Group contours by their plane z.
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for (ci, c) in contours.iter().enumerate() {
        let z = c.points.iter().map(|p| p.z).sum::<f64>() / c.points.len() as f64;
        match groups.iter_mut().find(|(gzv, _)| (*gzv - z).abs() < 0.1) {
            Some((_, v)) => v.push(ci),
            None => groups.push((z, vec![ci])),
        }
    }
    groups.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Half-slab thickness: half the typical contour spacing (≥ half a cell).
    let dz_typ = if groups.len() > 1 {
        let mut ds: Vec<f64> = groups.windows(2).map(|w| w[1].0 - w[0].0).collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ds[ds.len() / 2]
    } else {
        2.0 * h
    };
    let half_slab = (dz_typ * 0.5).max(h * 0.5) + 1e-6;

    // Each grid layer writes only its own plane of the mask, so the layers
    // rasterize independently.
    let mut inside = vec![false; gx * gy * gz];
    inside
        .par_chunks_mut(gx * gy)
        .enumerate()
        .for_each(|(k, plane)| {
            let z = origin[2] + k as f64 * h;
            // Nearest contour plane.
            let Some((gzv, idxs)) = groups.iter().min_by(|a, b| {
                (a.0 - z)
                    .abs()
                    .partial_cmp(&(b.0 - z).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) else {
                return;
            };
            if (gzv - z).abs() > half_slab {
                return;
            }
            // Even–odd scanline fill across all contours of this plane. The
            // crossing buffer is reused down the layer rather than reallocated
            // for every scanline.
            let mut xs: Vec<f64> = Vec::new();
            for j in 0..gy {
                let y = origin[1] + j as f64 * h;
                xs.clear();
                for &ci in idxs {
                    let pts = &contours[ci].points;
                    let n = pts.len();
                    for e in 0..n {
                        let (p, q) = (pts[e], pts[(e + 1) % n]);
                        if (p.y - y) * (q.y - y) <= 0.0 && p.y != q.y {
                            xs.push(p.x + (y - p.y) / (q.y - p.y) * (q.x - p.x));
                        }
                    }
                }
                if xs.len() < 2 {
                    continue;
                }
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let row = j * gx;
                for pair in xs.as_chunks::<2>().0 {
                    let i0 = ((pair[0] - origin[0]) / h).ceil().max(0.0) as usize;
                    let i1_f = (pair[1] - origin[0]) / h;
                    if i1_f < 0.0 {
                        continue;
                    }
                    let i1 = (i1_f.floor() as usize).min(gx - 1);
                    for i in i0..=i1 {
                        plane[row + i] = true;
                    }
                }
            }
        });

    // ---- 2-4. surface nets + smoothing + normals ------------------------
    let map = |x: f64, y: f64, z: f64| -> [f32; 3] {
        [
            (origin[0] + x * h) as f32,
            (origin[1] + y * h) as f32,
            (origin[2] + z * h) as f32,
        ]
    };
    let (verts, normals, tris) = net_surface(&inside, dims, &map)?;

    let lname = roi.name.to_lowercase();
    Some(RoiMesh {
        roi_index,
        name: roi.name.clone(),
        color: roi.color,
        external: roi.roi_type == "EXTERNAL"
            || lname.contains("body")
            || lname.contains("external")
            || lname.contains("outer"),
        verts,
        normals,
        tris,
    })
}

// ---------------------------------------------------------------------------
// Segmentation masks → surface meshes
// ---------------------------------------------------------------------------

/// Geometry of a voxel grid in patient space (a [`Volume`] without its data),
/// small enough to hand to a background meshing thread.
#[derive(Clone, Copy)]
pub struct GridGeom {
    pub origin: Vec3,
    pub row_dir: Vec3,
    pub col_dir: Vec3,
    pub normal: Vec3,
    pub spacing: [f64; 3],
}

impl GridGeom {
    pub fn of(vol: &Volume) -> Self {
        GridGeom {
            origin: vol.origin,
            row_dir: vol.row_dir,
            col_dir: vol.col_dir,
            normal: vol.normal,
            spacing: vol.spacing,
        }
    }
}

/// Surface-mesh a segmentation mask snapshot produced by
/// `Segmentation::mesh_grid`: a padded bool grid covering the mask's
/// bounding box at voxel indices `lo + (g - 1) * stride`.
pub fn mesh_from_mask(
    grid: &[bool],
    gdims: [usize; 3],
    lo: [usize; 3],
    stride: usize,
    geom: &GridGeom,
) -> Option<SurfaceMesh> {
    let map = |x: f64, y: f64, z: f64| -> [f32; 3] {
        let vi = lo[0] as f64 + (x - 1.0) * stride as f64;
        let vj = lo[1] as f64 + (y - 1.0) * stride as f64;
        let vk = lo[2] as f64 + (z - 1.0) * stride as f64;
        let p = geom.origin
            + geom.row_dir * (vi * geom.spacing[0])
            + geom.col_dir * (vj * geom.spacing[1])
            + geom.normal * (vk * geom.spacing[2]);
        [p.x as f32, p.y as f32, p.z as f32]
    };
    net_surface(grid, gdims, &map)
}

// ---------------------------------------------------------------------------
// Surface nets over a binary grid
// ---------------------------------------------------------------------------

/// Surface-nets extraction, Laplacian smoothing and vertex normals over a
/// binary grid. `map` converts fractional grid coordinates to patient mm
/// (applied before smoothing, so smoothing acts on the embedded surface).
fn net_surface(
    inside: &[bool],
    gdims: [usize; 3],
    map: &dyn Fn(f64, f64, f64) -> [f32; 3],
) -> Option<SurfaceMesh> {
    let [gx, gy, gz] = gdims;
    if gx < 3 || gy < 3 || gz < 3 {
        return None;
    }
    let (cx, cy, cz) = (gx - 1, gy - 1, gz - 1);
    let cell_at = |i: usize, j: usize, k: usize| k * cx * cy + j * cx + i;
    let corner = |i: usize, j: usize, k: usize| inside[k * gx * gy + j * gx + i];

    let mut cell_vert = vec![-1i64; cx * cy * cz];
    let mut verts: Vec<[f32; 3]> = Vec::new();
    // Cell-local corner offsets and the 12 cell edges (corner index pairs).
    const CORNERS: [[usize; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];
    const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];
    for k in 0..cz {
        for j in 0..cy {
            for i in 0..cx {
                let mut vals = [false; 8];
                for (ci, o) in CORNERS.iter().enumerate() {
                    vals[ci] = corner(i + o[0], j + o[1], k + o[2]);
                }
                if vals.iter().all(|&v| v) || vals.iter().all(|&v| !v) {
                    continue;
                }
                // Vertex = mean of crossing-edge midpoints.
                let mut acc = [0.0f64; 3];
                let mut cnt = 0.0;
                for e in &EDGES {
                    if vals[e[0]] != vals[e[1]] {
                        let a = CORNERS[e[0]];
                        let b = CORNERS[e[1]];
                        for ax in 0..3 {
                            acc[ax] += (a[ax] as f64 + b[ax] as f64) * 0.5;
                        }
                        cnt += 1.0;
                    }
                }
                let p = map(
                    i as f64 + acc[0] / cnt,
                    j as f64 + acc[1] / cnt,
                    k as f64 + acc[2] / cnt,
                );
                cell_vert[cell_at(i, j, k)] = verts.len() as i64;
                verts.push(p);
            }
        }
    }
    if verts.is_empty() {
        return None;
    }

    // Quads across sign-changing grid edges (two-sided shading downstream,
    // so winding does not need to be globally consistent).
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut quad = |c: [i64; 4]| {
        if c.iter().all(|&v| v >= 0) {
            let v = [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32];
            tris.push([v[0], v[1], v[2]]);
            tris.push([v[0], v[2], v[3]]);
        }
    };
    for k in 1..gz - 1 {
        for j in 1..gy - 1 {
            for i in 1..gx - 1 {
                // x-edge
                if i + 1 < gx && corner(i, j, k) != corner(i + 1, j, k) {
                    quad([
                        cell_vert[cell_at(i, j - 1, k - 1)],
                        cell_vert[cell_at(i, j, k - 1)],
                        cell_vert[cell_at(i, j, k)],
                        cell_vert[cell_at(i, j - 1, k)],
                    ]);
                }
                // y-edge
                if j + 1 < gy && corner(i, j, k) != corner(i, j + 1, k) {
                    quad([
                        cell_vert[cell_at(i - 1, j, k - 1)],
                        cell_vert[cell_at(i, j, k - 1)],
                        cell_vert[cell_at(i, j, k)],
                        cell_vert[cell_at(i - 1, j, k)],
                    ]);
                }
                // z-edge
                if k + 1 < gz && corner(i, j, k) != corner(i, j, k + 1) {
                    quad([
                        cell_vert[cell_at(i - 1, j - 1, k)],
                        cell_vert[cell_at(i, j - 1, k)],
                        cell_vert[cell_at(i, j, k)],
                        cell_vert[cell_at(i - 1, j, k)],
                    ]);
                }
            }
        }
    }
    if tris.is_empty() {
        return None;
    }

    // ---- Laplacian smoothing --------------------------------------------
    // Vertex adjacency in compressed-row form. Storing it as a `Vec` per
    // vertex deduplicated by linear scan meant one heap allocation per vertex
    // — on a body contour that is well over a hundred thousand allocations.
    let nv = verts.len();
    let mut start = vec![0u32; nv + 1];
    for t in &tris {
        for &v in t {
            start[v as usize + 1] += 2;
        }
    }
    for i in 0..nv {
        start[i + 1] += start[i];
    }
    let mut adj = vec![0u32; start[nv] as usize];
    let mut fill = start.clone();
    for t in &tris {
        for [a, b] in [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] {
            adj[fill[a as usize] as usize] = b;
            fill[a as usize] += 1;
            adj[fill[b as usize] as usize] = a;
            fill[b as usize] += 1;
        }
    }
    // Each edge is visited once per incident triangle, so collapse repeats.
    let mut nbrs = vec![0u32; nv];
    for v in 0..nv {
        let (a, b) = (start[v] as usize, start[v + 1] as usize);
        let seg = &mut adj[a..b];
        seg.sort_unstable();
        let mut n = 0;
        for i in 0..seg.len() {
            if i == 0 || seg[i] != seg[i - 1] {
                seg[n] = seg[i];
                n += 1;
            }
        }
        nbrs[v] = n as u32;
    }

    let mut prev = vec![[0.0f32; 3]; nv];
    for _ in 0..3 {
        prev.copy_from_slice(&verts);
        verts.par_iter_mut().enumerate().for_each(|(vi, v)| {
            let n = nbrs[vi] as usize;
            if n == 0 {
                return;
            }
            let nb = &adj[start[vi] as usize..start[vi] as usize + n];
            let mut m = [0.0f32; 3];
            for &o in nb {
                for ax in 0..3 {
                    m[ax] += prev[o as usize][ax];
                }
            }
            for ax in 0..3 {
                let mean = m[ax] / n as f32;
                v[ax] = prev[vi][ax] + 0.5 * (mean - prev[vi][ax]);
            }
        });
    }

    // ---- per-vertex normals ----------------------------------------------
    let mut normals = vec![[0.0f32; 3]; verts.len()];
    for t in &tris {
        let a = verts[t[0] as usize];
        let b = verts[t[1] as usize];
        let c = verts[t[2] as usize];
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            u[1] * w[2] - u[2] * w[1],
            u[2] * w[0] - u[0] * w[2],
            u[0] * w[1] - u[1] * w[0],
        ];
        for &vi in t {
            for ax in 0..3 {
                normals[vi as usize][ax] += n[ax];
            }
        }
    }
    for n in &mut normals {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-12);
        for c in n.iter_mut() {
            *c /= l;
        }
    }

    Some((verts, normals, tris))
}
