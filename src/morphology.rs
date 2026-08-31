//! Binary-mask morphology on voxel grids — the geometry every mask-shaped
//! feature needs and nobody should write twice.
//!
//! Masks are `[u8]` in [`crate::volume::Volume`] index order
//! (`k * nx * ny + j * nx + i`), non-zero meaning *set*. Every operation that
//! has a physical size takes it in **millimetres** and reads the voxel
//! spacing, so a 5 mm opening is 5 mm along each axis whatever the slice
//! thickness — which is the whole point on clinical CT, where in-plane and
//! through-plane spacing routinely differ by a factor of five.
//!
//! The distance transform is the exact anisotropic Euclidean one
//! (Felzenszwalb & Huttenlocher's lower envelope of parabolas, *Theory of
//! Computing* 2012): three separable O(n) passes, no approximation by
//! chamfer weights, no dependence of the cost on the radius. Erosion and
//! dilation are then thresholds on it, so an opening costs four passes
//! whether the radius is 1 mm or 50 mm.
//!
//! Outside the volume counts as **set** for the distance-to-background
//! transform. That convention is what keeps anatomy truncated by the scan
//! FoV — the arms at the edge of the field, the body at the first and last
//! slice — from being eroded away at the cut: nothing is inferred about
//! what was never imaged.

use rayon::prelude::*;

/// Squared distance (mm²) from every voxel to the nearest **unset** voxel;
/// zero on unset voxels. Voxels outside the volume count as set.
pub fn dist2_to_background(mask: &[u8], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    debug_assert_eq!(mask.len(), n);
    // Seed: 0 where unset, +inf where set.
    let mut f: Vec<f32> = mask
        .par_iter()
        .map(|&v| if v != 0 { f32::INFINITY } else { 0.0 })
        .collect();
    edt_in_place(&mut f, dims, spacing);
    f
}

/// Squared distance (mm²) from every voxel to the nearest **set** voxel;
/// zero on set voxels.
pub fn dist2_to_foreground(mask: &[u8], dims: [usize; 3], spacing: [f64; 3]) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    debug_assert_eq!(mask.len(), n);
    let mut f: Vec<f32> = mask
        .par_iter()
        .map(|&v| if v == 0 { f32::INFINITY } else { 0.0 })
        .collect();
    edt_in_place(&mut f, dims, spacing);
    f
}

/// Three separable passes of the 1-D squared-distance transform.
fn edt_in_place(f: &mut [f32], dims: [usize; 3], spacing: [f64; 3]) {
    for (axis, step) in spacing.iter().enumerate() {
        pass_along(f, dims, axis, *step as f32, Sweep::Both);
    }
}

/// Which sources a pass is allowed to reach back to.
///
/// `Both` is the ordinary distance transform. The one-sided forms are what
/// make an *asymmetric* margin possible: restricted to sources at a lower
/// index, a pass measures only how far the mask has grown in the `+axis`
/// direction, so three such passes give the distance within one octant and
/// eight octants tile the whole neighbourhood — each with its own radius.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sweep {
    Both,
    /// Sources at a lower index only (growth toward `+axis`).
    Forward,
    /// Sources at a higher index only (growth toward `−axis`).
    Backward,
}

/// Length, stride and the start index of every line running along `axis`.
fn lines_along(dims: [usize; 3], axis: usize) -> (usize, usize, Vec<usize>) {
    let [nx, ny, nz] = dims;
    match axis {
        0 => (nx, 1, (0..ny * nz).map(|l| l * nx).collect()),
        1 => (
            ny,
            nx,
            (0..nx * nz)
                .map(|l| (l / nx) * nx * ny + (l % nx))
                .collect(),
        ),
        _ => (nz, nx * ny, (0..nx * ny).collect()),
    }
}

/// Per-thread scratch for the 1-D transform, so that a pass over a whole CT
/// allocates a handful of buffers rather than one per line.
struct Envelope {
    d: Vec<f32>,
    v: Vec<usize>,
    z: Vec<f32>,
    out: Vec<f32>,
}

impl Envelope {
    fn new(n: usize) -> Envelope {
        Envelope {
            d: vec![0.0; n],
            v: vec![0; n],
            z: vec![0.0; n + 1],
            out: vec![0.0; n],
        }
    }
}

/// A pointer to the buffer, handed to each rayon task. Lines along one axis
/// are disjoint sets of indices, so writing them in parallel is sound; the
/// alternative — collecting every line and writing back afterwards — costs a
/// second copy of the volume and one allocation per line, which on a routine
/// CT is 300 MB and 150 000 allocations per pass.
#[derive(Clone, Copy)]
struct LinePtr(*mut f32);
unsafe impl Send for LinePtr {}
unsafe impl Sync for LinePtr {}
impl LinePtr {
    /// Reached through a method, not the field: closure capture in Rust 2021
    /// is field-precise, and capturing the bare `*mut f32` would sidestep the
    /// `Sync` promise made just above.
    fn get(self) -> *mut f32 {
        self.0
    }
}

/// The 1-D lower envelope of parabolas along one axis, in place.
///
/// Felzenszwalb & Huttenlocher's algorithm: every position contributes a
/// parabola `f(q) + h²(p − q)²`, and their lower envelope is built in one
/// left-to-right sweep and read off in another. `Sweep::Forward` reads each
/// value off the moment its own parabola has been inserted, which is exactly
/// the minimum over sources at or below that index.
fn pass_along(f: &mut [f32], dims: [usize; 3], axis: usize, step: f32, sweep: Sweep) {
    let (n, stride, starts) = lines_along(dims, axis);
    if n == 0 || !step.is_finite() || step <= 0.0 {
        return;
    }
    let sq = step * step;
    let ptr = LinePtr(f.as_mut_ptr());
    starts.par_iter().for_each_init(
        || Envelope::new(n),
        |e, &base| {
            // Gather the line, reversed for a backward sweep so that the
            // same forward machinery serves both.
            for q in 0..n {
                let src = if sweep == Sweep::Backward {
                    n - 1 - q
                } else {
                    q
                };
                e.d[q] = unsafe { *ptr.get().add(base + src * stride) };
            }
            let mut k = 0usize;
            e.v[0] = 0;
            e.z[0] = f32::NEG_INFINITY;
            e.z[1] = f32::INFINITY;
            if sweep != Sweep::Both {
                e.out[0] = e.d[0];
            }
            for q in 1..n {
                if e.d[q].is_finite() {
                    loop {
                        let p = e.v[k];
                        // Where the parabolas rooted at p and q cross.
                        let s = if e.d[p].is_infinite() {
                            f32::NEG_INFINITY
                        } else {
                            (e.d[q] + sq * (q * q) as f32 - e.d[p] - sq * (p * p) as f32)
                                / (2.0 * sq * (q as f32 - p as f32))
                        };
                        if s <= e.z[k] && k > 0 {
                            k -= 1;
                        } else {
                            k += 1;
                            e.v[k] = q;
                            e.z[k] = s;
                            e.z[k + 1] = f32::INFINITY;
                            break;
                        }
                    }
                }
                if sweep != Sweep::Both {
                    // Only parabolas up to q are in the envelope, and q sits
                    // in its last piece — so this is the one-sided minimum.
                    let p = e.v[k];
                    e.out[q] = if e.d[p].is_infinite() {
                        f32::INFINITY
                    } else {
                        let dq = (q as f32 - p as f32) * step;
                        e.d[p] + dq * dq
                    };
                }
            }
            if sweep == Sweep::Both {
                // Walk the finished envelope left to right.
                let mut k = 0usize;
                for (q, slot) in e.out.iter_mut().enumerate() {
                    while e.z[k + 1] < q as f32 {
                        k += 1;
                    }
                    let p = e.v[k];
                    *slot = if e.d[p].is_infinite() {
                        f32::INFINITY
                    } else {
                        let dq = (q as f32 - p as f32) * step;
                        e.d[p] + dq * dq
                    };
                }
            }
            for q in 0..n {
                let dst = if sweep == Sweep::Backward {
                    n - 1 - q
                } else {
                    q
                };
                unsafe { *ptr.get().add(base + dst * stride) = e.out[q] };
            }
        },
    );
}

/// Erosion by a ball of `radius_mm`: the voxels further than the radius from
/// any background voxel. Voxels outside the volume are not background, so a
/// mask that runs into the volume boundary is not eroded there.
pub fn erode_mm(mask: &[u8], dims: [usize; 3], spacing: [f64; 3], radius_mm: f64) -> Vec<u8> {
    if radius_mm <= 0.0 {
        return mask.to_vec();
    }
    let r2 = (radius_mm * radius_mm) as f32;
    let d = dist2_to_background(mask, dims, spacing);
    d.par_iter().map(|&v| u8::from(v > r2)).collect()
}

/// Dilation by a ball of `radius_mm`.
pub fn dilate_mm(mask: &[u8], dims: [usize; 3], spacing: [f64; 3], radius_mm: f64) -> Vec<u8> {
    if radius_mm <= 0.0 {
        return mask.to_vec();
    }
    let r2 = (radius_mm * radius_mm) as f32;
    let d = dist2_to_foreground(mask, dims, spacing);
    d.par_iter().map(|&v| u8::from(v <= r2)).collect()
}

/// A margin in millimetres per array axis and per direction:
/// `radii[axis] = [toward decreasing index, toward increasing index]`.
///
/// The structuring element is the ellipsoid whose semi-axis in each of the
/// six directions is the corresponding entry — the shape a planning system
/// means by "5 mm laterally, 8 mm superiorly". Symmetric cases are detected
/// and take a single distance transform; only a genuinely one-sided margin
/// pays for the eight-octant form.
pub type Radii = [[f64; 2]; 3];

/// True when the margin is the same in both directions along every axis, so
/// the structuring element is centrally symmetric.
fn symmetric(radii: &Radii) -> bool {
    radii.iter().all(|r| (r[0] - r[1]).abs() < 1e-9)
}

/// Largest radius anywhere in the margin.
fn widest(radii: &Radii) -> f64 {
    radii.iter().flatten().cloned().fold(0.0, f64::max)
}

/// Scale factor that turns a physical spacing into "fractions of the radius",
/// so that thresholding the transform at 1 tests the ellipsoid equation
/// `Σ (dₐ/rₐ)² ≤ 1`. A radius of zero forbids any offset along that axis,
/// which a very large scale expresses without a special case.
fn unit_step(spacing: f64, radius: f64) -> f32 {
    if radius <= 0.0 {
        (spacing * 1e6) as f32
    } else {
        (spacing / radius) as f32
    }
}

/// Dilation by the ellipsoid of [`Radii`].
///
/// Dilation distributes over a union of structuring elements, and an
/// asymmetric ellipsoid is the union of its eight octants — each of which is
/// an octant of an *ordinary* ellipsoid, and so is reached by three one-sided
/// passes. Eight of those is the price of a one-sided margin; a symmetric one
/// costs a single transform.
pub fn dilate_radii(mask: &[u8], dims: [usize; 3], spacing: [f64; 3], radii: &Radii) -> Vec<u8> {
    let n = dims[0] * dims[1] * dims[2];
    debug_assert_eq!(mask.len(), n);
    if widest(radii) <= 0.0 {
        return mask.to_vec();
    }
    let seed = || -> Vec<f32> {
        mask.par_iter()
            .map(|&v| if v == 0 { f32::INFINITY } else { 0.0 })
            .collect()
    };
    if symmetric(radii) {
        let mut f = seed();
        for (axis, sp) in spacing.iter().enumerate() {
            pass_along(
                &mut f,
                dims,
                axis,
                unit_step(*sp, radii[axis][1]),
                Sweep::Both,
            );
        }
        return f.par_iter().map(|&v| u8::from(v <= 1.0)).collect();
    }
    let mut out = vec![0u8; n];
    for octant in 0..8u8 {
        let mut f = seed();
        for (axis, sp) in spacing.iter().enumerate() {
            // Bit set = this octant grows toward +axis, so its sources lie
            // at lower indices and the pass sweeps forward.
            let positive = octant >> axis & 1 == 1;
            let r = radii[axis][usize::from(positive)];
            let sweep = if positive {
                Sweep::Forward
            } else {
                Sweep::Backward
            };
            pass_along(&mut f, dims, axis, unit_step(*sp, r), sweep);
        }
        out.par_iter_mut()
            .zip(f.par_iter())
            .for_each(|(o, &v)| *o |= u8::from(v <= 1.0));
    }
    out
}

/// Erosion by the ellipsoid of [`Radii`] — everything that survives having
/// the shape swept round the inside of the mask.
///
/// Computed as the complement of dilating the complement, which is the
/// definition; it also inherits the convention that voxels outside the volume
/// are not background, so anatomy truncated by the field of view is not
/// eroded at the cut.
pub fn erode_radii(mask: &[u8], dims: [usize; 3], spacing: [f64; 3], radii: &Radii) -> Vec<u8> {
    if widest(radii) <= 0.0 {
        return mask.to_vec();
    }
    let inverted: Vec<u8> = mask.par_iter().map(|&v| u8::from(v == 0)).collect();
    // Reflected: eroding the anterior surface by 5 mm is dilating the
    // background toward the posterior.
    let mirrored: Radii = std::array::from_fn(|a| [radii[a][1], radii[a][0]]);
    let grown = dilate_radii(&inverted, dims, spacing, &mirrored);
    grown.par_iter().map(|&v| u8::from(v == 0)).collect()
}

/// Opening — erosion then dilation. Equivalently: the union of every ball of
/// `radius_mm` that fits entirely inside the mask, so everything thinner
/// than twice the radius disappears and everything thicker keeps its exact
/// surface.
pub fn open_mm(mask: &[u8], dims: [usize; 3], spacing: [f64; 3], radius_mm: f64) -> Vec<u8> {
    if radius_mm <= 0.0 {
        return mask.to_vec();
    }
    let eroded = erode_mm(mask, dims, spacing, radius_mm);
    dilate_mm(&eroded, dims, spacing, radius_mm)
}

/// Closing — dilation then erosion. Bridges gaps narrower than twice the
/// radius; used to take the staircase off a contour.
pub fn close_mm(mask: &[u8], dims: [usize; 3], spacing: [f64; 3], radius_mm: f64) -> Vec<u8> {
    if radius_mm <= 0.0 {
        return mask.to_vec();
    }
    let dilated = dilate_mm(mask, dims, spacing, radius_mm);
    erode_mm(&dilated, dims, spacing, radius_mm)
}

// ---------------------------------------------------------------------------
// Connected components
// ---------------------------------------------------------------------------

/// One 6-connected component: the voxels it owns and its bounding box.
///
/// The voxel list rather than a per-voxel label volume is deliberate: a
/// label volume costs four bytes for every voxel of the *image* (313 MB on a
/// 512 × 512 × 300 CT), the lists together cost four bytes per *set* voxel,
/// which for a body mask is an order of magnitude less.
#[derive(Clone, Debug)]
pub struct Component {
    pub voxels: Vec<u32>,
    /// Inclusive voxel bounding box.
    pub lo: [usize; 3],
    pub hi: [usize; 3],
}

impl Component {
    pub fn len(&self) -> usize {
        self.voxels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.voxels.is_empty()
    }
    /// Volume in cm³ for the given voxel spacing.
    pub fn cm3(&self, spacing: [f64; 3]) -> f64 {
        self.voxels.len() as f64 * spacing[0] * spacing[1] * spacing[2] / 1000.0
    }
    /// Longest side of the bounding box, in millimetres. Zero for an empty
    /// component, whose bounding box is the uninitialised one.
    pub fn extent_mm(&self, spacing: [f64; 3]) -> f64 {
        if self.voxels.is_empty() {
            return 0.0;
        }
        (0..3)
            .map(|a| (self.hi[a] - self.lo[a] + 1) as f64 * spacing[a])
            .fold(0.0, f64::max)
    }
}

/// Every 6-connected component of the mask, largest first.
pub fn components(mask: &[u8], dims: [usize; 3]) -> Vec<Component> {
    let [nx, ny, nz] = dims;
    let sl = nx * ny;
    let n = sl * nz;
    debug_assert_eq!(mask.len(), n);
    let mut seen = vec![false; n];
    let mut out: Vec<Component> = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    for start in 0..n {
        if mask[start] == 0 || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start as u32);
        let mut voxels: Vec<u32> = Vec::new();
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        while let Some(p) = stack.pop() {
            let idx = p as usize;
            let (i, j, k) = (idx % nx, (idx % sl) / nx, idx / sl);
            for (a, c) in [i, j, k].into_iter().enumerate() {
                lo[a] = lo[a].min(c);
                hi[a] = hi[a].max(c);
            }
            voxels.push(p);
            // 6-neighbourhood. Diagonal contact is not contact: two organs
            // that share only a corner are two components, and so are a
            // couch rail and the skin it grazes.
            if i > 0 {
                push(&mut stack, &mut seen, mask, idx - 1);
            }
            if i + 1 < nx {
                push(&mut stack, &mut seen, mask, idx + 1);
            }
            if j > 0 {
                push(&mut stack, &mut seen, mask, idx - nx);
            }
            if j + 1 < ny {
                push(&mut stack, &mut seen, mask, idx + nx);
            }
            if k > 0 {
                push(&mut stack, &mut seen, mask, idx - sl);
            }
            if k + 1 < nz {
                push(&mut stack, &mut seen, mask, idx + sl);
            }
        }
        out.push(Component { voxels, lo, hi });
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.voxels.len()));
    out
}

#[inline]
fn push(stack: &mut Vec<u32>, seen: &mut [bool], mask: &[u8], idx: usize) {
    if mask[idx] != 0 && !seen[idx] {
        seen[idx] = true;
        stack.push(idx as u32);
    }
}

/// True when any voxel of `comp` has a 6-neighbour set in `other`.
pub fn touches(comp: &Component, other: &[u8], dims: [usize; 3]) -> bool {
    let [nx, ny, nz] = dims;
    let sl = nx * ny;
    comp.voxels.par_iter().any(|&p| {
        let idx = p as usize;
        let (i, j, k) = (idx % nx, (idx % sl) / nx, idx / sl);
        (i > 0 && other[idx - 1] != 0)
            || (i + 1 < nx && other[idx + 1] != 0)
            || (j > 0 && other[idx - nx] != 0)
            || (j + 1 < ny && other[idx + nx] != 0)
            || (k > 0 && other[idx - sl] != 0)
            || (k + 1 < nz && other[idx + sl] != 0)
    })
}

// ---------------------------------------------------------------------------
// Hole filling
// ---------------------------------------------------------------------------

/// Fill every background region of a slice that the slice border cannot
/// reach, for each slice perpendicular to `axis`.
///
/// Two-dimensional on purpose. A lung is connected to the outside air
/// through the trachea, so a three-dimensional fill leaves both lungs open
/// on any scan that includes the neck; slice by slice they close.
pub fn fill_holes_2d(mask: &mut [u8], dims: [usize; 3], axis: usize) {
    let [nx, ny, nz] = dims;
    let sl = nx * ny;
    // In-plane extents and the two in-plane strides, per slicing axis.
    let (n_slices, slice_stride, (w, h), (su, sv)) = match axis {
        0 => (nx, 1usize, (ny, nz), (nx, sl)),
        1 => (ny, nx, (nx, nz), (1usize, sl)),
        _ => (nz, sl, (nx, ny), (1usize, nx)),
    };
    if n_slices == 0 || w == 0 || h == 0 {
        return;
    }
    let filled: Vec<(usize, Vec<u32>)> = (0..n_slices)
        .into_par_iter()
        .map(|s| {
            let base = s * slice_stride;
            let mut reach = vec![false; w * h];
            let mut stack: Vec<usize> = Vec::new();
            let at = |u: usize, v: usize| base + u * su + v * sv;
            // Seed from the whole slice border.
            let seed = |u: usize, v: usize, reach: &mut Vec<bool>, stack: &mut Vec<usize>| {
                let p = v * w + u;
                if !reach[p] && mask[at(u, v)] == 0 {
                    reach[p] = true;
                    stack.push(p);
                }
            };
            for u in 0..w {
                seed(u, 0, &mut reach, &mut stack);
                seed(u, h - 1, &mut reach, &mut stack);
            }
            for v in 0..h {
                seed(0, v, &mut reach, &mut stack);
                seed(w - 1, v, &mut reach, &mut stack);
            }
            while let Some(p) = stack.pop() {
                let (u, v) = (p % w, p / w);
                let visit = |qu: usize, qv: usize, reach: &mut Vec<bool>, st: &mut Vec<usize>| {
                    let q = qv * w + qu;
                    if !reach[q] && mask[at(qu, qv)] == 0 {
                        reach[q] = true;
                        st.push(q);
                    }
                };
                if u > 0 {
                    visit(u - 1, v, &mut reach, &mut stack);
                }
                if u + 1 < w {
                    visit(u + 1, v, &mut reach, &mut stack);
                }
                if v > 0 {
                    visit(u, v - 1, &mut reach, &mut stack);
                }
                if v + 1 < h {
                    visit(u, v + 1, &mut reach, &mut stack);
                }
            }
            let mut add: Vec<u32> = Vec::new();
            for v in 0..h {
                for u in 0..w {
                    let idx = at(u, v);
                    if mask[idx] == 0 && !reach[v * w + u] {
                        add.push(idx as u32);
                    }
                }
            }
            (s, add)
        })
        .collect();
    for (_, add) in filled {
        for idx in add {
            mask[idx as usize] = 1;
        }
    }
}

/// Fill every background region the volume border cannot reach.
pub fn fill_holes_3d(mask: &mut [u8], dims: [usize; 3]) {
    let [nx, ny, nz] = dims;
    let sl = nx * ny;
    let n = sl * nz;
    let mut reach = vec![false; n];
    let mut stack: Vec<u32> = Vec::new();
    let seed = |idx: usize, reach: &mut Vec<bool>, stack: &mut Vec<u32>| {
        if mask[idx] == 0 && !reach[idx] {
            reach[idx] = true;
            stack.push(idx as u32);
        }
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                if i == 0 || j == 0 || k == 0 || i + 1 == nx || j + 1 == ny || k + 1 == nz {
                    seed(k * sl + j * nx + i, &mut reach, &mut stack);
                }
            }
        }
    }
    while let Some(p) = stack.pop() {
        let idx = p as usize;
        let (i, j, k) = (idx % nx, (idx % sl) / nx, idx / sl);
        let visit = |q: usize, reach: &mut Vec<bool>, st: &mut Vec<u32>| {
            if mask[q] == 0 && !reach[q] {
                reach[q] = true;
                st.push(q as u32);
            }
        };
        if i > 0 {
            visit(idx - 1, &mut reach, &mut stack);
        }
        if i + 1 < nx {
            visit(idx + 1, &mut reach, &mut stack);
        }
        if j > 0 {
            visit(idx - nx, &mut reach, &mut stack);
        }
        if j + 1 < ny {
            visit(idx + nx, &mut reach, &mut stack);
        }
        if k > 0 {
            visit(idx - sl, &mut reach, &mut stack);
        }
        if k + 1 < nz {
            visit(idx + sl, &mut reach, &mut stack);
        }
    }
    mask.par_iter_mut()
        .zip(reach.par_iter())
        .for_each(|(m, r)| {
            if *m == 0 && !*r {
                *m = 1;
            }
        });
}

// ---------------------------------------------------------------------------
// Axis persistence
// ---------------------------------------------------------------------------

/// Mark the voxels whose *column* along `axis` is occupied in at least
/// `frac` of the slices of a `window_mm` neighbourhood.
///
/// This is the shape prior that separates equipment from anatomy without
/// knowing what either looks like. A couch top, a chair backrest, a seat
/// pan, an arm rest, a bright reconstruction-circle rim: each is a surface
/// swept along one axis, so its footprint in the orthogonal plane repeats
/// slice after slice after slice. A pinna, a nose, a finger — the thin
/// anatomy a plain opening also removes — never repeats over anything like
/// the same distance, so a window of 150 mm at 80 % separates them cleanly.
///
/// The scan has to contain slices where the columns in question are free of
/// the patient. A couch strip directly beneath the body over the *entire*
/// scan length is not distinguishable this way, and is left to the
/// model-assisted method.
pub fn axis_persistence(
    mask: &[u8],
    dims: [usize; 3],
    spacing: [f64; 3],
    axis: usize,
    window_mm: f64,
    frac: f64,
) -> Vec<u8> {
    let [nx, ny, nz] = dims;
    let sl = nx * ny;
    let n = sl * nz;
    debug_assert_eq!(mask.len(), n);
    // Slice count and stride along `axis`, and the in-plane geometry.
    let (n_slices, slice_stride, w, h, su, sv) = match axis {
        0 => (nx, 1usize, ny, nz, nx, sl),
        1 => (ny, nx, nx, nz, 1usize, sl),
        _ => (nz, sl, nx, ny, 1usize, nx),
    };
    let plane = w * h;
    let mut out = vec![0u8; n];
    if n_slices == 0 || plane == 0 {
        return out;
    }
    let win = ((window_mm / spacing[axis]).round() as usize).clamp(1, n_slices);
    let need = ((win as f64) * frac).ceil().max(1.0) as u32;
    let at = |s: usize, p: usize| s * slice_stride + (p % w) * su + (p / w) * sv;

    // The window [lo, hi) advances monotonically with `s`, so each slice is
    // added once and removed once: the whole scan costs two passes.
    let mut count = vec![0u32; plane];
    let (mut cur_lo, mut cur_hi) = (0usize, 0usize);
    for s in 0..n_slices {
        let lo = s.saturating_sub(win / 2).min(n_slices - win);
        let hi = lo + win;
        while cur_hi < hi {
            for (p, c) in count.iter_mut().enumerate() {
                *c += u32::from(mask[at(cur_hi, p)] != 0);
            }
            cur_hi += 1;
        }
        while cur_lo < lo {
            for (p, c) in count.iter_mut().enumerate() {
                *c -= u32::from(mask[at(cur_lo, p)] != 0);
            }
            cur_lo += 1;
        }
        for (p, &c) in count.iter().enumerate() {
            if c >= need {
                let idx = at(s, p);
                if mask[idx] != 0 {
                    out[idx] = 1;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Smoothing
// ---------------------------------------------------------------------------

/// Three successive box blurs of `sigma_mm` — a close approximation of a
/// Gaussian (the central limit theorem does the work) at O(voxels) per axis
/// whatever the width, which is what makes a 40 mm blur of a whole MR study
/// affordable. Used to estimate the receive-coil bias field.
pub fn blur_mm(src: &[f32], dims: [usize; 3], spacing: [f64; 3], sigma_mm: f64) -> Vec<f32> {
    let mut buf = src.to_vec();
    if sigma_mm <= 0.0 {
        return buf;
    }
    for (axis, step) in spacing.iter().enumerate() {
        if !step.is_finite() || *step <= 0.0 {
            continue;
        }
        // A box of width w has variance (w² − 1)/12; three of them give 3×.
        let var = sigma_mm * sigma_mm / (step * step) / 3.0;
        // Capped at the extent of the axis: a window wider than the data
        // averages the same clamped values over and over, and an unclamped
        // width from a degenerate spacing (a series that declares a slice
        // thickness of zero) would run for the age of the universe.
        let n = dims[axis];
        let w = ((12.0 * var + 1.0).sqrt().round() as usize).clamp(1, 2 * n.max(1) + 1) | 1;
        if w <= 1 {
            continue;
        }
        for _ in 0..3 {
            box_pass(&mut buf, dims, axis, w);
        }
    }
    buf
}

/// One box blur of odd width `w` along `axis`, edges clamped.
fn box_pass(buf: &mut [f32], dims: [usize; 3], axis: usize, w: usize) {
    let (n, stride, starts) = lines_along(dims, axis);
    if n == 0 || w <= 1 {
        return;
    }
    let r = (w / 2) as isize;
    let last = n as isize - 1;
    let out: Vec<(usize, Vec<f32>)> = starts
        .par_iter()
        .map(|&base| {
            let line: Vec<f32> = (0..n).map(|q| buf[base + q * stride]).collect();
            let cl = |t: isize| line[t.clamp(0, last) as usize];
            let mut acc: f32 = (-r..=r).map(cl).sum();
            let mut res = vec![0f32; n];
            for (q, slot) in res.iter_mut().enumerate() {
                *slot = acc / w as f32;
                acc += cl(q as isize + r + 1) - cl(q as isize - r);
            }
            (base, res)
        })
        .collect();
    for (base, line) in out {
        for (q, v) in line.into_iter().enumerate() {
            buf[base + q * stride] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force squared distance to the nearest background voxel, with
    /// the same "outside is not background" convention.
    fn brute(mask: &[u8], dims: [usize; 3], sp: [f64; 3]) -> Vec<f32> {
        let [nx, ny, nz] = dims;
        let mut out = vec![f32::INFINITY; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = k * nx * ny + j * nx + i;
                    if mask[idx] == 0 {
                        out[idx] = 0.0;
                        continue;
                    }
                    let mut best = f32::INFINITY;
                    for kk in 0..nz {
                        for jj in 0..ny {
                            for ii in 0..nx {
                                if mask[kk * nx * ny + jj * nx + ii] != 0 {
                                    continue;
                                }
                                let d = (((i as f64 - ii as f64) * sp[0]).powi(2)
                                    + ((j as f64 - jj as f64) * sp[1]).powi(2)
                                    + ((k as f64 - kk as f64) * sp[2]).powi(2))
                                    as f32;
                                best = best.min(d);
                            }
                        }
                    }
                    out[idx] = best;
                }
            }
        }
        out
    }

    #[test]
    fn the_distance_transform_is_exact_and_anisotropic() {
        let dims = [9, 7, 5];
        let sp = [0.8, 1.3, 3.0];
        // A deterministic pseudo-random mask.
        let mut state = 12345u32;
        let mask: Vec<u8> = (0..dims[0] * dims[1] * dims[2])
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                u8::from(!(state >> 28).is_multiple_of(4))
            })
            .collect();
        let got = dist2_to_background(&mask, dims, sp);
        let want = brute(&mask, dims, sp);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() < 1e-3 || (g.is_infinite() && w.is_infinite()),
                "{g} vs {w}"
            );
        }
    }

    /// Brute-force dilation by the asymmetric ellipsoid, straight from the
    /// definition: p is in iff some set voxel sits within the shape.
    fn brute_dilate(mask: &[u8], dims: [usize; 3], sp: [f64; 3], r: &Radii) -> Vec<u8> {
        let [nx, ny, nz] = dims;
        let mut out = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = [i, j, k];
                    'search: for kk in 0..nz {
                        for jj in 0..ny {
                            for ii in 0..nx {
                                if mask[kk * nx * ny + jj * nx + ii] == 0 {
                                    continue;
                                }
                                let q = [ii, jj, kk];
                                let mut sum = 0.0f64;
                                let mut ok = true;
                                for a in 0..3 {
                                    let d = (p[a] as f64 - q[a] as f64) * sp[a];
                                    // d > 0 means p lies at a higher index
                                    // than the source: growth toward +axis.
                                    let rad = if d >= 0.0 { r[a][1] } else { r[a][0] };
                                    if d == 0.0 {
                                        continue;
                                    }
                                    if rad <= 0.0 {
                                        ok = false;
                                        break;
                                    }
                                    sum += (d / rad).powi(2);
                                }
                                if ok && sum <= 1.0 + 1e-9 {
                                    out[k * nx * ny + j * nx + i] = 1;
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn a_directional_margin_matches_the_shape_it_claims() {
        let dims = [11, 9, 7];
        let sp = [1.0, 1.5, 2.5];
        let mut mask = vec![0u8; dims[0] * dims[1] * dims[2]];
        let at = |i: usize, j: usize, k: usize| k * dims[0] * dims[1] + j * dims[0] + i;
        mask[at(5, 4, 3)] = 1;
        mask[at(2, 1, 1)] = 1;
        for radii in [
            [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]], // isotropic
            [[4.0, 4.0], [2.0, 2.0], [6.0, 6.0]], // symmetric, anisotropic
            [[0.0, 5.0], [3.0, 1.0], [2.0, 8.0]], // one-sided
            [[5.0, 0.0], [0.0, 0.0], [0.0, 4.0]], // axes switched off
        ] {
            let got = dilate_radii(&mask, dims, sp, &radii);
            let want = brute_dilate(&mask, dims, sp, &radii);
            assert_eq!(got, want, "dilation by {radii:?}");
        }
    }

    #[test]
    fn a_symmetric_margin_agrees_with_the_plain_ball() {
        let dims = [15, 13, 5];
        let sp = [1.0, 1.0, 3.0];
        let mut mask = vec![0u8; dims[0] * dims[1] * dims[2]];
        for k in 1..4 {
            for j in 4..9 {
                for i in 5..10 {
                    mask[k * dims[0] * dims[1] + j * dims[0] + i] = 1;
                }
            }
        }
        for r in [1.0, 2.5, 4.0] {
            assert_eq!(
                dilate_radii(&mask, dims, sp, &[[r; 2]; 3]),
                dilate_mm(&mask, dims, sp, r),
                "dilate by {r}"
            );
            assert_eq!(
                erode_radii(&mask, dims, sp, &[[r; 2]; 3]),
                erode_mm(&mask, dims, sp, r),
                "erode by {r}"
            );
        }
    }

    #[test]
    fn eroding_one_side_moves_only_that_surface() {
        // A slab, shrunk 4 mm from the low-index side of axis 0 alone.
        let dims = [20, 5, 3];
        let sp = [2.0, 2.0, 2.0];
        let at = |i: usize, j: usize, k: usize| k * 100 + j * 20 + i;
        let mut mask = vec![0u8; 300];
        for k in 0..3 {
            for j in 0..5 {
                for i in 4..16 {
                    mask[at(i, j, k)] = 1;
                }
            }
        }
        let out = erode_radii(&mask, dims, sp, &[[4.0, 0.0], [0.0; 2], [0.0; 2]]);
        assert_eq!(out[at(5, 2, 1)], 0, "the low side moved in");
        assert_eq!(out[at(6, 2, 1)], 1, "…by 4 mm and no further");
        assert_eq!(out[at(15, 2, 1)], 1, "the high side did not move");
    }

    #[test]
    fn a_blur_wider_than_the_volume_still_terminates() {
        // A slice thickness of a micron is what a series that declares zero
        // gets clamped to; the box width must not follow it to infinity.
        let dims = [8, 4, 2];
        let src = vec![3.0f32; dims[0] * dims[1] * dims[2]];
        let out = blur_mm(&src, dims, [1.0, 1.0, 1e-6], 40.0);
        assert!(out.iter().all(|v| (v - 3.0).abs() < 1e-3));
    }

    #[test]
    fn a_fully_set_volume_has_no_background_to_measure() {
        let dims = [4, 4, 4];
        let mask = vec![1u8; 64];
        // Outside is not background, so nothing is within any finite
        // distance — the convention that stops truncated anatomy eroding.
        assert!(dist2_to_background(&mask, dims, [1.0; 3])
            .iter()
            .all(|d| d.is_infinite()));
        assert_eq!(erode_mm(&mask, dims, [1.0; 3], 100.0), vec![1u8; 64]);
    }

    #[test]
    fn opening_removes_sheets_thinner_than_the_ball_and_keeps_the_rest() {
        // A 40 × 40 × 40 mm block and, 10 mm away, a 2 mm sheet.
        let dims = [40, 40, 20];
        let sp = [1.0, 1.0, 2.0];
        let mut mask = vec![0u8; dims[0] * dims[1] * dims[2]];
        let at = |i: usize, j: usize, k: usize| k * dims[0] * dims[1] + j * dims[0] + i;
        for k in 2..18 {
            for j in 4..36 {
                for i in 4..36 {
                    mask[at(i, j, k)] = 1;
                }
            }
        }
        // A one-voxel (2 mm) sheet along the whole i/k extent.
        for k in 0..dims[2] {
            for i in 0..dims[0] {
                mask[at(i, 1, k)] = 1;
            }
        }
        let opened = open_mm(&mask, dims, sp, 5.0);
        for k in 0..dims[2] {
            for i in 0..dims[0] {
                assert_eq!(opened[at(i, 1, k)], 0, "the sheet survived at i={i} k={k}");
            }
        }
        // The block's interior is untouched, and so is its surface.
        assert_eq!(opened[at(20, 20, 10)], 1);
        assert_eq!(opened[at(20, 4, 10)], 1, "the block's own face is kept");
    }

    #[test]
    fn components_are_six_connected_and_sorted_by_size() {
        let dims = [6, 6, 1];
        let mut mask = vec![0u8; 36];
        // Two 2×2 squares touching only at a corner.
        for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
            mask[j * 6 + i] = 1;
        }
        for (i, j) in [(2, 2), (3, 2), (2, 3), (3, 3), (4, 3)] {
            mask[j * 6 + i] = 1;
        }
        let c = components(&mask, dims);
        assert_eq!(c.len(), 2, "a shared corner is not contact");
        assert_eq!(c[0].len(), 5, "largest first");
        assert_eq!(c[1].len(), 4);
        assert_eq!(c[0].lo, [2, 2, 0]);
        assert_eq!(c[0].hi, [4, 3, 0]);
    }

    #[test]
    fn slicewise_filling_closes_a_lung_that_a_three_d_fill_leaves_open() {
        // A block with a cavity that connects to the outside on one slice
        // only — a lung and its trachea.
        let dims = [12, 12, 6];
        let at = |i: usize, j: usize, k: usize| k * 144 + j * 12 + i;
        let mut mask = vec![0u8; 12 * 12 * 6];
        for k in 0..6 {
            for j in 1..11 {
                for i in 1..11 {
                    mask[at(i, j, k)] = 1;
                }
            }
        }
        // Cavity in the middle of every slice…
        for k in 0..6 {
            for j in 4..8 {
                for i in 4..8 {
                    mask[at(i, j, k)] = 0;
                }
            }
        }
        // …with a channel to the outside on slice 0 only.
        for j in 0..4 {
            mask[at(5, j, 0)] = 0;
        }
        let mut three_d = mask.clone();
        fill_holes_3d(&mut three_d, dims);
        assert_eq!(
            three_d[at(6, 6, 3)],
            0,
            "the whole cavity drains through the one channel"
        );
        let mut two_d = mask.clone();
        fill_holes_2d(&mut two_d, dims, 2);
        for k in 1..6 {
            assert_eq!(two_d[at(6, 6, k)], 1, "slice {k} closed");
        }
        assert_eq!(
            two_d[at(6, 6, 0)],
            0,
            "the slice with the channel stays open"
        );
    }

    #[test]
    fn persistence_marks_an_extruded_rail_and_spares_a_bump() {
        let dims = [20, 20, 40];
        let sp = [1.0, 1.0, 2.0]; // 80 mm along k
        let at = |i: usize, j: usize, k: usize| k * 400 + j * 20 + i;
        let mut mask = vec![0u8; 20 * 20 * 40];
        // A rail running the whole length in k.
        for k in 0..40 {
            for i in 5..15 {
                mask[at(i, 2, k)] = 1;
            }
        }
        // A bump 8 mm long (4 slices).
        for k in 18..22 {
            for i in 8..12 {
                mask[at(i, 15, k)] = 1;
            }
        }
        let p = axis_persistence(&mask, dims, sp, 2, 60.0, 0.8);
        assert_eq!(p[at(9, 2, 20)], 1, "the rail is persistent");
        assert_eq!(p[at(9, 15, 20)], 0, "the bump is not");
        // Nothing outside the mask is ever marked.
        assert!(p.iter().zip(mask.iter()).all(|(a, m)| *a == 0 || *m != 0));
    }

    #[test]
    fn blurring_preserves_a_constant_and_spreads_a_step() {
        let dims = [32, 4, 4];
        let n = dims[0] * dims[1] * dims[2];
        let flat = vec![7.0f32; n];
        let out = blur_mm(&flat, dims, [1.0; 3], 4.0);
        for v in &out {
            assert!((v - 7.0).abs() < 1e-3, "{v}");
        }
        let step: Vec<f32> = (0..n)
            .map(|idx| if idx % dims[0] < 16 { 0.0 } else { 100.0 })
            .collect();
        let sm = blur_mm(&step, dims, [1.0; 3], 4.0);
        let mid = sm[15];
        assert!((0.0..100.0).contains(&mid), "the step is smoothed: {mid}");
        assert!(
            sm[0] < 5.0 && sm[dims[0] - 1] > 95.0,
            "the plateaus survive"
        );
    }
}
