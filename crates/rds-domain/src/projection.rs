//! A projection image as a pinhole camera, and the fast forward projector
//! the 2D/3D registration renders through.
//!
//! [`ProjectionGeometry`] is the one description every source of geometry
//! ends in - the IEC 61217 parameters of a linac or an upright room
//! ([`crate::drr::Geometry::camera`]), the attributes of an RT Image, a
//! calibrated 3 × 4 projection matrix of a room-mounted kV pair - and the one
//! the projector reads: a source point, the centre of the first pixel, and
//! the steps from a pixel to the next one along a row and down a column, all
//! in patient millimetres (DICOM LPS). The ray of a pixel is the line from the
//! source through its centre; where along that line the detector sits is
//! irrelevant to a line integral, which is why a projection matrix (which
//! knows no detector distance) defines the same camera once a plane is
//! chosen.
//!
//! A rigid motion of the patient moves no voxel here: a rigid transform maps
//! lines to lines, so the DRR of a CT moved by `T` is the DRR of the CT as it
//! is, through the camera moved by `T` ([`ProjectionGeometry::transformed`]).
//! A registration therefore never resamples the volume, and the cost of a
//! rendering does not depend on the pose.
//!
//! [`MuVolume`] is the volume prepared for that: attenuation per millimetre
//! (`f32`, threshold applied) cropped to a box around the anatomy that
//! matters, with coarser copies for coarse-to-fine work. [`render_posed`]
//! marches every ray in index space - the entry point and the step converted
//! once per ray, one trilinear read per sample, no patient-to-voxel
//! conversion inside the loop - with the midpoint rule of ITK's
//! `RayCastInterpolateImageFunction`, the projector ITK's 2D/3D metrics use.
//! Blocks of 8 × 8 × 8 voxels in which every sample is exactly zero (air
//! below the threshold, the outside of a structure's shadow) are crossed
//! without sampling: the position still advances by the same additions, so
//! every later sample and every sum keeps its bits.
//! It integrates the interpolated attenuation, where the DRR section's
//! ray-cast integrates the attenuation of the interpolated HU; the two differ
//! only where a voxel straddles the threshold. [`gpu::GpuProjector`] is the
//! same ray-cast as a compute shader.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::drr::HuMode;
use crate::geometry::Vec3;
use crate::iec::{self, M3};
use crate::progress::{ProgressSink, CANCELLED};
use crate::volume::Volume;

pub mod gpu;

/// A pinhole camera in patient coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionGeometry {
    /// The focal spot.
    pub source: Vec3,
    /// Centre of pixel (0, 0).
    pub origin: Vec3,
    /// From pixel `(i, j)` to `(i + 1, j)`: along a row, mm.
    pub row_step: Vec3,
    /// From pixel `(i, j)` to `(i, j + 1)`: down a column, mm.
    pub col_step: Vec3,
    /// `[columns, rows]`.
    pub dims: [usize; 2],
}

fn invert(m: &M3) -> Option<M3> {
    let a = m;
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-18 {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv,
        ],
    ])
}

fn columns(a: Vec3, b: Vec3, c: Vec3) -> M3 {
    [[a.x, b.x, c.x], [a.y, b.y, c.y], [a.z, b.z, c.z]]
}

impl ProjectionGeometry {
    /// Patient coordinates of the centre of pixel `(i, j)` (fractions
    /// allowed).
    #[inline]
    pub fn pixel(&self, i: f64, j: f64) -> Vec3 {
        self.origin + self.row_step * i + self.col_step * j
    }

    /// Pixel spacing on the detector, `[along a row, down a column]`, mm.
    pub fn spacing(&self) -> [f64; 2] {
        [self.row_step.length(), self.col_step.length()]
    }

    /// The centre of the detector.
    pub fn center(&self) -> Vec3 {
        self.pixel(
            (self.dims[0] as f64 - 1.0) * 0.5,
            (self.dims[1] as f64 - 1.0) * 0.5,
        )
    }

    /// The matrix taking a direction from the source to `(u', v', w')` with
    /// pixel `(u'/w', v'/w')`: the inverse of `[row_step, col_step, origin −
    /// source]`.
    fn inverse_basis(&self) -> Option<M3> {
        invert(&columns(
            self.row_step,
            self.col_step,
            self.origin - self.source,
        ))
    }

    /// Continuous pixel coordinates `[i, j]` of the projection of `p`, or
    /// `None` when `p` is not in front of the source.
    pub fn project(&self, p: Vec3) -> Option<[f64; 2]> {
        let inv = self.inverse_basis()?;
        let q = iec::mul_vec(&inv, p - self.source);
        (q.z > 1e-12).then(|| [q.x / q.z, q.y / q.z])
    }

    /// The projection and its derivative with respect to the point: the two
    /// rows `∂i/∂p`, `∂j/∂p`.
    pub fn project_with_jacobian(&self, p: Vec3) -> Option<([f64; 2], [Vec3; 2])> {
        let inv = self.inverse_basis()?;
        let q = iec::mul_vec(&inv, p - self.source);
        if q.z <= 1e-12 {
            return None;
        }
        let row = |r: usize| Vec3::new(inv[r][0], inv[r][1], inv[r][2]);
        let (r0, r1, r2) = (row(0), row(1), row(2));
        let w2 = q.z * q.z;
        let di = r0 * (1.0 / q.z) - r2 * (q.x / w2);
        let dj = r1 * (1.0 / q.z) - r2 * (q.y / w2);
        Some(([q.x / q.z, q.y / q.z], [di, dj]))
    }

    /// The same camera seen through a point mapping: the source, the first
    /// pixel and its two neighbours moved by `f`. For a rigid `f` this is
    /// exact, and it is how a pose is applied: the DRR of a CT moved by `T`
    /// is the DRR of the CT through this camera moved by `T`.
    pub fn mapped(&self, f: impl Fn(Vec3) -> Vec3) -> ProjectionGeometry {
        let o = f(self.origin);
        ProjectionGeometry {
            source: f(self.source),
            origin: o,
            row_step: f(self.origin + self.row_step) - o,
            col_step: f(self.origin + self.col_step) - o,
            dims: self.dims,
        }
    }

    /// [`Self::mapped`] through a rigid transform's `map`.
    pub fn transformed(&self, t: &crate::registration::RigidTransform) -> ProjectionGeometry {
        self.mapped(|p| t.map(p))
    }

    /// The 3 × 4 projection matrix `P` with `(u, v, w) = P · (x, y, z, 1)`
    /// and pixel `(u / w, v / w)`.
    pub fn to_matrix(&self) -> Option<[[f64; 4]; 3]> {
        let inv = self.inverse_basis()?;
        let t = iec::mul_vec(&inv, self.source * -1.0);
        Some([
            [inv[0][0], inv[0][1], inv[0][2], t.x],
            [inv[1][0], inv[1][1], inv[1][2], t.y],
            [inv[2][0], inv[2][1], inv[2][2], t.z],
        ])
    }

    /// The camera of a projection matrix, with its detector put in the plane
    /// `distance` mm from the source, perpendicular to the principal axis
    /// (the matrix's third row): points of that plane map affinely onto the
    /// pixels, so the pixel centres and steps there are exact.
    pub fn from_matrix(
        p: &[[f64; 4]; 3],
        dims: [usize; 2],
        distance: f64,
    ) -> Option<ProjectionGeometry> {
        let m: M3 = [
            [p[0][0], p[0][1], p[0][2]],
            [p[1][0], p[1][1], p[1][2]],
            [p[2][0], p[2][1], p[2][2]],
        ];
        let minv = invert(&m)?;
        let p4 = Vec3::new(p[0][3], p[1][3], p[2][3]);
        let source = iec::mul_vec(&minv, p4) * -1.0;
        let n = Vec3::new(m[2][0], m[2][1], m[2][2]);
        let nn = n.length();
        if nn < 1e-18 {
            return None;
        }
        let n = n * (1.0 / nn);
        let hit = |u: f64, v: f64| -> Option<Vec3> {
            let d = iec::mul_vec(&minv, Vec3::new(u, v, 1.0));
            let s = n.dot(d);
            (s.abs() > 1e-12).then(|| source + d * (distance / s))
        };
        let o = hit(0.0, 0.0)?;
        Some(ProjectionGeometry {
            source,
            origin: o,
            row_step: hit(1.0, 0.0)? - o,
            col_step: hit(0.0, 1.0)? - o,
            dims,
        })
    }

    /// The geometry of a rendering `factor` times coarser: every pixel
    /// `factor × factor` of these, the same field of view.
    pub fn coarser(&self, factor: usize) -> ProjectionGeometry {
        let f = factor.max(1) as f64;
        let dims = [
            self.dims[0].div_ceil(factor.max(1)),
            self.dims[1].div_ceil(factor.max(1)),
        ];
        // Pixel (0, 0) of the coarse grid covers the first f × f fine ones:
        // its centre is half a coarse pixel minus half a fine one in.
        let shift = (f - 1.0) * 0.5;
        ProjectionGeometry {
            source: self.source,
            origin: self.pixel(shift, shift),
            row_step: self.row_step * f,
            col_step: self.col_step * f,
            dims,
        }
    }
}

/// A volume prepared for projection: attenuation per millimetre, cropped
/// to a box, `data[k·nx·ny + j·nx + i]` as every volume.
///
/// `data` is read-only once the volume is made: the occupancy the ray
/// march skips empty space by is computed from it then.
#[derive(Clone, Debug)]
pub struct MuVolume {
    pub data: Vec<f32>,
    pub dims: [usize; 3],
    /// Patient coordinates of voxel (0, 0, 0)'s centre.
    pub origin: Vec3,
    /// The steps from a voxel to the next along i, j and k, mm.
    pub axes: [Vec3; 3],
    /// Patient offset from `origin` to continuous index.
    inv: M3,
    /// Which blocks of the lattice a sample can read anything but zero in.
    occ: Occupancy,
}

/// log2 of the side of an occupancy block, voxels.
const BLOCK_SHIFT: u32 = 3;
/// The side of an occupancy block: 8 voxels.
const BLOCK: usize = 1 << BLOCK_SHIFT;

/// Empty space, block by block: a block of 8 × 8 × 8 voxels is *empty* when
/// every voxel a trilinear read with its lower corner inside the block
/// touches (the block itself and the layer beyond its three far faces) is
/// zero. Every sample whose lower corner lies in an empty block is then
/// exactly `+0.0`, and adding it changes no sum, so a march that leaves those
/// samples out gives the same bits as one that takes them all.
#[derive(Clone, Debug, Default)]
struct Occupancy {
    /// Blocks along each axis.
    dims: [usize; 3],
    /// `true` where the block holds something, `[bk][bj][bi]`.
    any: Vec<bool>,
    /// What one `f32` addition of the march can be off by, at most, inside
    /// the lattice: half an ulp of the largest block face, and again as
    /// much for good measure.
    err: f64,
}

/// A ray's step as [`Occupancy::run_at`] reads it, worked out once per ray.
struct Stride {
    d: [f32; 3],
    /// `1 / (|d| + err)` per axis; zero where the ray does not move.
    per: [f64; 3],
    err: f64,
}

impl Occupancy {
    fn of(data: &[f32], dims: [usize; 3]) -> Occupancy {
        let [nx, ny, nz] = dims;
        if nx * ny * nz == 0 || data.len() != nx * ny * nz {
            return Occupancy::default();
        }
        let b = [nx.div_ceil(BLOCK), ny.div_ceil(BLOCK), nz.div_ceil(BLOCK)];
        let mut any = vec![false; b[0] * b[1] * b[2]];
        any.par_chunks_mut(b[0] * b[1])
            .enumerate()
            .for_each(|(bk, plane)| {
                let (k0, k1) = (bk * BLOCK, (bk * BLOCK + BLOCK + 1).min(nz));
                for bj in 0..b[1] {
                    let (j0, j1) = (bj * BLOCK, (bj * BLOCK + BLOCK + 1).min(ny));
                    for bi in 0..b[0] {
                        let (i0, i1) = (bi * BLOCK, (bi * BLOCK + BLOCK + 1).min(nx));
                        // `!= 0.0` holds for a NaN too: such a block is read.
                        plane[bj * b[0] + bi] = (k0..k1).any(|k| {
                            (j0..j1).any(|j| {
                                let row = (k * ny + j) * nx;
                                data[row + i0..row + i1].iter().any(|&v| v != 0.0)
                            })
                        });
                    }
                }
            });
        let far = (BLOCK * b[0].max(b[1]).max(b[2]) + 1) as f64;
        Occupancy {
            dims: b,
            any,
            err: far * f64::from(f32::EPSILON),
        }
    }

    fn stride(&self, d: [f32; 3]) -> Stride {
        let per = std::array::from_fn(|a| {
            let da = d[a] as f64;
            if da == 0.0 {
                0.0
            } else {
                1.0 / (da.abs() + self.err)
            }
        });
        Stride {
            d,
            per,
            err: self.err,
        }
    }

    /// From the march's position `p` (continuous index, `f32` as the march
    /// keeps it) and its step: whether the samples from here on are surely
    /// zero, and for how many steps, this one included, that answer holds
    /// (at least one).
    ///
    /// The count is the number of steps the sample's lower corner surely
    /// stays in this block. The march adds the step to `p` in `f32`, each
    /// addition off by at most [`Occupancy::err`], so every step is charged
    /// that much on top of its length (and the quotient a millionth of a
    /// step less, for its own rounding); an addition never moves a
    /// coordinate against the sign of its step, so the near face needs no
    /// margin.
    #[inline]
    fn run_at(&self, p: [f32; 3], s: &Stride) -> (bool, usize) {
        // A negative (or NaN) coordinate samples as zero.
        if !(p[0] >= 0.0 && p[1] >= 0.0 && p[2] >= 0.0) {
            return (true, 1);
        }
        let b: [usize; 3] = std::array::from_fn(|a| (p[a] as usize) >> BLOCK_SHIFT);
        if b[0] >= self.dims[0] || b[1] >= self.dims[1] || b[2] >= self.dims[2] {
            return (false, 1);
        }
        let full = self.any[(b[2] * self.dims[1] + b[1]) * self.dims[0] + b[0]];
        let mut steps = f64::INFINITY;
        for a in 0..3 {
            if s.per[a] == 0.0 {
                continue;
            }
            let lo = (b[a] << BLOCK_SHIFT) as f64;
            let x = p[a] as f64;
            let room = if s.d[a] > 0.0 {
                lo + BLOCK as f64 - x
            } else {
                x - lo
            };
            steps = steps.min(((room - s.err) * s.per[a] - 1e-6).floor());
        }
        let extra = if steps.is_finite() {
            steps.max(0.0) as usize
        } else {
            usize::MAX - 1
        };
        (!full, extra + 1)
    }
}

impl MuVolume {
    fn with_geometry(data: Vec<f32>, dims: [usize; 3], origin: Vec3, axes: [Vec3; 3]) -> Self {
        let inv = invert(&columns(axes[0], axes[1], axes[2])).unwrap_or(iec::IDENTITY);
        let occ = Occupancy::of(&data, dims);
        MuVolume {
            data,
            dims,
            origin,
            axes,
            inv,
            occ,
        }
    }

    /// The fraction of the lattice in blocks that hold anything: what the
    /// march cannot skip.
    pub fn occupied_fraction(&self) -> f64 {
        if self.occ.any.is_empty() {
            return 0.0;
        }
        self.occ.any.iter().filter(|&&a| a).count() as f64 / self.occ.any.len() as f64
    }

    /// Attenuation of `vol` as the DRR's HU model has it, below-threshold
    /// voxels zero, cropped to the patient-space box `clip` (min, max corners)
    /// with a voxel of margin - or the whole volume.
    pub fn prepare(vol: &Volume, hu: HuMode, threshold: f32, clip: Option<(Vec3, Vec3)>) -> Self {
        let [nx, ny, nz] = vol.dims;
        let (lo, hi) = match clip {
            None => ([0usize; 3], [nx, ny, nz]),
            Some((a, b)) => {
                let mut lo = [f64::INFINITY; 3];
                let mut hi = [f64::NEG_INFINITY; 3];
                for c in 0..8 {
                    let p = Vec3::new(
                        if c & 1 == 0 { a.x } else { b.x },
                        if c & 2 == 0 { a.y } else { b.y },
                        if c & 4 == 0 { a.z } else { b.z },
                    );
                    let v = vol.patient_to_voxel(p);
                    for ax in 0..3 {
                        lo[ax] = lo[ax].min(v[ax]);
                        hi[ax] = hi[ax].max(v[ax]);
                    }
                }
                let mut l = [0usize; 3];
                let mut h = [0usize; 3];
                for ax in 0..3 {
                    let n = vol.dims[ax] as f64;
                    l[ax] = (lo[ax].floor() - 1.0).clamp(0.0, n) as usize;
                    h[ax] = (hi[ax].ceil() + 2.0).clamp(0.0, n) as usize;
                    if h[ax] <= l[ax] {
                        h[ax] = (l[ax] + 1).min(vol.dims[ax]);
                        l[ax] = h[ax].saturating_sub(1);
                    }
                }
                (l, h)
            }
        };
        let dims = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let mut data = vec![0.0f32; dims[0] * dims[1] * dims[2]];
        data.par_chunks_mut((dims[0] * dims[1]).max(1))
            .enumerate()
            .for_each(|(k, plane)| {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        let v = vol.index(lo[0] + i, lo[1] + j, lo[2] + k) as f32;
                        plane[j * dims[0] + i] = mu_of(v, hu, threshold);
                    }
                }
            });
        let axes = [
            vol.row_dir * vol.spacing[0],
            vol.col_dir * vol.spacing[1],
            vol.normal * vol.spacing[2],
        ];
        let origin = vol.voxel_to_patient(lo[0] as f64, lo[1] as f64, lo[2] as f64);
        MuVolume::with_geometry(data, dims, origin, axes)
    }

    /// A structure as a volume of ones on the CT's lattice, cropped to the
    /// box around it with a voxel of margin: projected, it is the
    /// structure's shadow (its depth along each ray). `None` when the mask
    /// is empty or not of `vol`'s size.
    pub fn from_mask(vol: &Volume, mask: &[u8]) -> Option<MuVolume> {
        let [nx, ny, nz] = vol.dims;
        if mask.len() != nx * ny * nz {
            return None;
        }
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        for (o, &m) in mask.iter().enumerate() {
            if m != 0 {
                let c = [o % nx, (o / nx) % ny, o / (nx * ny)];
                for a in 0..3 {
                    lo[a] = lo[a].min(c[a]);
                    hi[a] = hi[a].max(c[a]);
                }
            }
        }
        if lo[0] == usize::MAX {
            return None;
        }
        let lo: [usize; 3] = std::array::from_fn(|a| lo[a].saturating_sub(1));
        let hi: [usize; 3] = std::array::from_fn(|a| (hi[a] + 2).min(vol.dims[a]));
        let dims = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        let mut data = vec![0.0f32; dims[0] * dims[1] * dims[2]];
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let o = (lo[2] + k) * nx * ny + (lo[1] + j) * nx + lo[0] + i;
                    data[(k * dims[1] + j) * dims[0] + i] = (mask[o] != 0) as u8 as f32;
                }
            }
        }
        let axes = [
            vol.row_dir * vol.spacing[0],
            vol.col_dir * vol.spacing[1],
            vol.normal * vol.spacing[2],
        ];
        let origin = vol.voxel_to_patient(lo[0] as f64, lo[1] as f64, lo[2] as f64);
        Some(MuVolume::with_geometry(data, dims, origin, axes))
    }

    /// The next coarser copy: every axis finer than one and a half times
    /// the finest spacing is halved, the others kept, so a CT of thin
    /// pixels and thick slices becomes isotropic before it becomes coarse.
    /// Each voxel is the mean of the block it stands for (fewer at an odd
    /// edge), centred on it.
    pub fn decimated(&self) -> MuVolume {
        let sp: Vec<f64> = self.axes.iter().map(|a| a.length()).collect();
        let finest = sp.iter().copied().fold(f64::INFINITY, f64::min);
        let f: [usize; 3] = std::array::from_fn(|a| {
            if sp[a] < 1.5 * finest && self.dims[a] > 1 {
                2
            } else {
                1
            }
        });
        self.decimated_by(f)
    }

    /// A copy `f[a]` times as coarse along axis `a` (1 or 2).
    fn decimated_by(&self, f: [usize; 3]) -> MuVolume {
        let [nx, ny, nz] = self.dims;
        let d = [nx.div_ceil(f[0]), ny.div_ceil(f[1]), nz.div_ceil(f[2])];
        let mut data = vec![0.0f32; d[0] * d[1] * d[2]];
        data.par_chunks_mut((d[0] * d[1]).max(1))
            .enumerate()
            .for_each(|(k, plane)| {
                for j in 0..d[1] {
                    for i in 0..d[0] {
                        let (mut s, mut c) = (0.0f32, 0u32);
                        for dk in 0..f[2] {
                            for dj in 0..f[1] {
                                for di in 0..f[0] {
                                    let (x, y, z) = (f[0] * i + di, f[1] * j + dj, f[2] * k + dk);
                                    if x < nx && y < ny && z < nz {
                                        s += self.data[z * nx * ny + y * nx + x];
                                        c += 1;
                                    }
                                }
                            }
                        }
                        plane[j * d[0] + i] = s / c.max(1) as f32;
                    }
                }
            });
        let shift = |a: usize| self.axes[a] * ((f[a] as f64 - 1.0) * 0.5);
        MuVolume::with_geometry(
            data,
            d,
            self.origin + shift(0) + shift(1) + shift(2),
            [
                self.axes[0] * f[0] as f64,
                self.axes[1] * f[1] as f64,
                self.axes[2] * f[2] as f64,
            ],
        )
    }

    /// The finest voxel spacing, mm.
    pub fn finest_spacing(&self) -> f64 {
        self.axes
            .iter()
            .map(|a| a.length())
            .fold(f64::INFINITY, f64::min)
    }

    /// This volume and `n − 1` coarser copies, finest first.
    pub fn levels(self, n: usize) -> Vec<MuVolume> {
        let mut out = vec![self];
        while out.len() < n.max(1) {
            let next = out.last().unwrap().decimated();
            out.push(next);
        }
        out
    }

    /// Continuous index coordinates of a patient point.
    #[inline]
    pub fn to_index(&self, p: Vec3) -> Vec3 {
        iec::mul_vec(&self.inv, p - self.origin)
    }

    /// The patient-space corners of the volume's box, min and max.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for c in 0..8 {
            let f = |bit: usize, n: usize| {
                if c & bit == 0 {
                    0.0
                } else {
                    (n.max(1) - 1) as f64
                }
            };
            let p = self.origin
                + self.axes[0] * f(1, self.dims[0])
                + self.axes[1] * f(2, self.dims[1])
                + self.axes[2] * f(4, self.dims[2]);
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        (lo, hi)
    }

    /// Trilinear attenuation at continuous index `(u, v, w)`, zero outside.
    #[inline]
    fn sample(&self, u: f32, v: f32, w: f32) -> f32 {
        let [nx, ny, nz] = self.dims;
        // Written so a NaN fails too.
        if !(u >= 0.0 && v >= 0.0 && w >= 0.0) {
            return 0.0;
        }
        let (fx, fy, fz) = (u.floor(), v.floor(), w.floor());
        let (i0, j0, k0) = (fx as usize, fy as usize, fz as usize);
        let (tx, ty, tz) = (u - fx, v - fy, w - fz);
        let plane = nx * ny;
        if i0 + 1 < nx && j0 + 1 < ny && k0 + 1 < nz {
            // Inside: the eight corners from one slice of the data.
            let base = k0 * plane + j0 * nx + i0;
            let c = &self.data[base..base + plane + nx + 2];
            let (r1, p1) = (nx, plane);
            let c00 = c[0] + (c[1] - c[0]) * tx;
            let c10 = c[r1] + (c[r1 + 1] - c[r1]) * tx;
            let c01 = c[p1] + (c[p1 + 1] - c[p1]) * tx;
            let c11 = c[p1 + r1] + (c[p1 + r1 + 1] - c[p1 + r1]) * tx;
            let c0 = c00 + (c10 - c00) * ty;
            let c1 = c01 + (c11 - c01) * ty;
            return c0 + (c1 - c0) * tz;
        }
        if i0 >= nx || j0 >= ny || k0 >= nz {
            return 0.0;
        }
        let i1 = (i0 + 1).min(nx - 1);
        let j1 = (j0 + 1).min(ny - 1);
        let k1 = (k0 + 1).min(nz - 1);
        let at = |i: usize, j: usize, k: usize| self.data[k * plane + j * nx + i];
        let c00 = at(i0, j0, k0) + (at(i1, j0, k0) - at(i0, j0, k0)) * tx;
        let c10 = at(i0, j1, k0) + (at(i1, j1, k0) - at(i0, j1, k0)) * tx;
        let c01 = at(i0, j0, k1) + (at(i1, j0, k1) - at(i0, j0, k1)) * tx;
        let c11 = at(i0, j1, k1) + (at(i1, j1, k1) - at(i0, j1, k1)) * tx;
        let c0 = c00 + (c10 - c00) * ty;
        let c1 = c01 + (c11 - c01) * ty;
        c0 + (c1 - c0) * tz
    }

    /// The line integral from `a` to `b` (patient coordinates) at a step of
    /// about `step_mm`, midpoint rule.
    pub fn integrate(&self, a: Vec3, b: Vec3, step_mm: f64) -> f32 {
        let ia = self.to_index(a);
        let ib = self.to_index(b);
        let d = ib - ia;
        let (av, dv) = ([ia.x, ia.y, ia.z], [d.x, d.y, d.z]);
        let mut t0 = 0.0f64;
        let mut t1 = 1.0f64;
        for ax in 0..3 {
            let hi = (self.dims[ax].max(1) - 1) as f64;
            if dv[ax].abs() < 1e-12 {
                if av[ax] < 0.0 || av[ax] > hi {
                    return 0.0;
                }
                continue;
            }
            let mut ta = (0.0 - av[ax]) / dv[ax];
            let mut tb = (hi - av[ax]) / dv[ax];
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 >= t1 {
                return 0.0;
            }
        }
        let total_mm = (b - a).length();
        let span_mm = (t1 - t0) * total_mm;
        if span_mm <= 0.0 {
            return 0.0;
        }
        let n = ((span_mm / step_mm.max(1e-3)).ceil() as usize).max(1);
        let h = span_mm / n as f64;
        let dt = (t1 - t0) / n as f64;
        let start = t0 + 0.5 * dt;
        let (mut u, mut v, mut w) = (
            (av[0] + dv[0] * start) as f32,
            (av[1] + dv[1] * start) as f32,
            (av[2] + dv[2] * start) as f32,
        );
        let (du, dvv, dw) = (
            (dv[0] * dt) as f32,
            (dv[1] * dt) as f32,
            (dv[2] * dt) as f32,
        );
        let mut acc = 0.0f32;
        // A step of more than three voxels meets each block once or twice:
        // finding the runs would cost what skipping them saves.
        let coarse = du.abs().max(dvv.abs()).max(dw.abs()) > 3.0;
        if self.occ.any.is_empty() || coarse {
            for _ in 0..n {
                acc += self.sample(u, v, w);
                u += du;
                v += dvv;
                w += dw;
            }
            return acc * h as f32;
        }
        // Run by run: through empty blocks only the position moves (by the
        // same additions, so every later sample is where it was), through
        // the others every step is sampled.
        let stride = self.occ.stride([du, dvv, dw]);
        let mut left = n;
        while left > 0 {
            let (zero, run) = self.occ.run_at([u, v, w], &stride);
            let run = run.min(left);
            left -= run;
            if zero {
                for _ in 0..run {
                    u += du;
                    v += dvv;
                    w += dw;
                }
            } else {
                for _ in 0..run {
                    acc += self.sample(u, v, w);
                    u += du;
                    v += dvv;
                    w += dw;
                }
            }
        }
        acc * h as f32
    }

    /// The same volume marched without skipping empty space: every step
    /// sampled. The same renderings to the bit, slower; what the skipping
    /// is measured (`reg2d3d_cli --bench`) and proved against.
    pub fn every_step(&self) -> MuVolume {
        let mut plain = self.clone();
        plain.occ = Occupancy::default();
        plain
    }

    /// [`MuVolume::integrate`] of [`MuVolume::every_step`].
    #[cfg(test)]
    fn integrate_every_step(&self, a: Vec3, b: Vec3, step_mm: f64) -> f32 {
        self.every_step().integrate(a, b, step_mm)
    }
}

/// Voxel value to attenuation per millimetre, as the DRR has it.
#[inline]
pub fn mu_of(v: f32, mode: HuMode, threshold: f32) -> f32 {
    if v < threshold {
        return 0.0;
    }
    match mode {
        HuMode::Water => (crate::drr::MU_WATER as f32 * (1.0 + v / 1000.0)).max(0.0),
        HuMode::Raw => v,
        HuMode::MegaVoltage => crate::drr::mu_megavoltage(v),
    }
}

/// Render `mu` through `cam`: one line integral per pixel, row-major
/// `[rows][columns]`. Rows run in parallel; every pixel is computed alone,
/// so the image is the same on any number of threads.
pub fn render_posed(
    mu: &MuVolume,
    cam: &ProjectionGeometry,
    step_mm: f64,
    sink: &dyn ProgressSink,
) -> Result<Vec<f32>> {
    render_posed_in(mu, cam, step_mm, None, sink)
}

/// [`render_posed`] for the pixels of `mask` only (`cam`'s raster), the
/// others zero: what a measure that reads only those needs.
pub fn render_posed_in(
    mu: &MuVolume,
    cam: &ProjectionGeometry,
    step_mm: f64,
    mask: Option<&[u8]>,
    sink: &dyn ProgressSink,
) -> Result<Vec<f32>> {
    let [w, h] = cam.dims;
    if w == 0 || h == 0 {
        bail!("the detector has no pixels");
    }
    let mask = mask.filter(|m| m.len() == w * h);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(j, row)| {
        if sink.cancelled() {
            return;
        }
        for (i, px) in row.iter_mut().enumerate() {
            if mask.is_some_and(|m| m[j * w + i] == 0) {
                continue;
            }
            let target = cam.pixel(i as f64, j as f64);
            *px = mu.integrate(cam.source, target, step_mm);
        }
    });
    if sink.cancelled() {
        bail!(CANCELLED);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drr::{DrrParams, Engine, Geometry};
    use crate::progress::Quiet;

    fn phantom(n: usize, spacing: f64) -> Volume {
        // An off-centre block of bone in water in air: asymmetric, so a
        // wrong axis or sign shows.
        let mut data = vec![-1000i16; n * n * n];
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let inside = |a: usize, lo: usize, hi: usize| a >= lo && a < hi;
                    let mut v = -1000i16;
                    if inside(i, n / 5, 4 * n / 5)
                        && inside(j, n / 5, 4 * n / 5)
                        && inside(k, n / 6, 5 * n / 6)
                    {
                        v = 0;
                    }
                    if inside(i, n / 4, n / 2)
                        && inside(j, n / 3, n / 2)
                        && inside(k, n / 3, 2 * n / 3)
                    {
                        v = 1000;
                    }
                    data[k * n * n + j * n + i] = v;
                }
            }
        }
        let half = (n as f64 - 1.0) * 0.5 * spacing;
        Volume {
            data,
            dims: [n, n, n],
            spacing: [spacing; 3],
            origin: Vec3::new(-half, -half, -half),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: -1000,
            max_value: 1000,
        }
    }

    fn correlation(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().map(|v| *v as f64).sum::<f64>() / n;
        let mb = b.iter().map(|v| *v as f64).sum::<f64>() / n;
        let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            let (dx, dy) = (*x as f64 - ma, *y as f64 - mb);
            c += dx * dy;
            va += dx * dx;
            vb += dy * dy;
        }
        c / (va * vb).sqrt()
    }

    #[test]
    fn the_camera_matches_the_beam_frame() {
        let g = Geometry {
            gantry_deg: 37.0,
            couch_deg: 15.0,
            dims: [64, 48],
            ..Geometry::default()
        };
        let cam = g.camera();
        let f = g.frame();
        assert!((cam.source - f.source).length() < 1e-9);
        assert!((cam.center() - f.panel_center).length() < 1e-9);
        let px = g.pixel_mm();
        assert!((cam.row_step - f.panel_x * px[0]).length() < 1e-9);
        assert!(
            (cam.col_step + f.panel_y * px[1]).length() < 1e-9,
            "rows run down"
        );
        // Every pixel projects back onto itself.
        for (i, j) in [(0.0, 0.0), (63.0, 47.0), (10.5, 3.25)] {
            let p = cam.source + (cam.pixel(i, j) - cam.source) * 0.6;
            let q = cam.project(p).unwrap();
            assert!((q[0] - i).abs() < 1e-9 && (q[1] - j).abs() < 1e-9);
        }
        // Behind the source projects nowhere.
        assert!(cam.project(cam.source * 2.0 - cam.center()).is_none());
    }

    #[test]
    fn a_projection_matrix_round_trips() {
        let cam = Geometry {
            gantry_deg: 120.0,
            dims: [40, 30],
            ..Geometry::default()
        }
        .camera();
        let p = cam.to_matrix().unwrap();
        let back = ProjectionGeometry::from_matrix(&p, cam.dims, 1500.0).unwrap();
        assert!((back.source - cam.source).length() < 1e-6);
        assert!((back.origin - cam.origin).length() < 1e-6);
        assert!((back.row_step - cam.row_step).length() < 1e-9);
        assert!((back.col_step - cam.col_step).length() < 1e-9);
        // At another distance the rays are the same, only the plane moves.
        let near = ProjectionGeometry::from_matrix(&p, cam.dims, 1000.0).unwrap();
        let x = Vec3::new(12.0, -7.0, 30.0);
        let (a, b) = (cam.project(x).unwrap(), near.project(x).unwrap());
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    }

    #[test]
    fn the_jacobian_of_the_projection_is_its_derivative() {
        let cam = Geometry {
            gantry_deg: 70.0,
            couch_deg: -20.0,
            dims: [64, 64],
            ..Geometry::default()
        }
        .camera();
        let p = Vec3::new(14.0, -22.0, 9.0);
        let (q, jac) = cam.project_with_jacobian(p).unwrap();
        let h = 1e-4;
        for (ax, e) in [
            Vec3::new(h, 0.0, 0.0),
            Vec3::new(0.0, h, 0.0),
            Vec3::new(0.0, 0.0, h),
        ]
        .into_iter()
        .enumerate()
        {
            let a = cam.project(p + e).unwrap();
            let b = cam.project(p - e).unwrap();
            let fd = [(a[0] - b[0]) / (2.0 * h), (a[1] - b[1]) / (2.0 * h)];
            let an = [
                [jac[0].x, jac[0].y, jac[0].z][ax],
                [jac[1].x, jac[1].y, jac[1].z][ax],
            ];
            assert!((fd[0] - an[0]).abs() < 1e-6 && (fd[1] - an[1]).abs() < 1e-6);
        }
        assert!((q[0] - cam.project(p).unwrap()[0]).abs() < 1e-12);
    }

    /// The fast projector integrates what the ray-cast does: same geometry,
    /// the same picture to the interpolation of the threshold.
    #[test]
    fn the_posed_projector_agrees_with_the_ray_cast() {
        let vol = phantom(40, 2.0);
        let params = DrrParams {
            geometry: Geometry {
                gantry_deg: 30.0,
                dims: [48, 48],
                panel_mm: [240.0, 240.0],
                ..Geometry::centered_on(&vol)
            },
            engine: Engine::RayCast,
            step_mm: 0.5,
            ..DrrParams::for_volume(&vol)
        };
        let reference = crate::drr::render(&vol, &params, &Quiet).unwrap();
        let mu = MuVolume::prepare(&vol, params.hu, params.threshold_hu, None);
        let fast = render_posed(&mu, &params.geometry.camera(), 0.5, &Quiet).unwrap();
        let r = correlation(&reference.pixels, &fast);
        assert!(r > 0.999, "r = {r}");
        let mean = |v: &[f32]| v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64;
        let rel = (mean(&fast) - mean(&reference.pixels)).abs() / mean(&reference.pixels);
        assert!(rel < 0.02, "mean differs by {rel}");
    }

    /// A ball of water with a bone in it, in air, on an oblique lattice:
    /// most of the blocks are empty, and the rays cross them at every angle.
    fn ball(n: usize, spacing: f64) -> Volume {
        let mut data = vec![-1000i16; n * n * n];
        let c = [n as f64 * 0.47, n as f64 * 0.41, n as f64 * 0.55];
        let r = n as f64 * 0.3;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d2 = |cc: [f64; 3]| {
                        (i as f64 - cc[0]).powi(2)
                            + (j as f64 - cc[1]).powi(2)
                            + (k as f64 - cc[2]).powi(2)
                    };
                    let o = (k * n + j) * n + i;
                    if d2(c) < r * r {
                        data[o] = 20;
                    }
                    if d2([c[0] + 4.0, c[1] - 3.0, c[2]]) < (r * 0.3).powi(2) {
                        data[o] = 900;
                    }
                }
            }
        }
        let a = 0.3f64;
        let half = (n as f64 - 1.0) * 0.5 * spacing;
        Volume {
            data,
            dims: [n, n, n],
            spacing: [spacing, spacing, spacing * 1.7],
            origin: Vec3::new(-half, -half, -half),
            row_dir: Vec3::new(a.cos(), a.sin(), 0.0),
            col_dir: Vec3::new(-a.sin(), a.cos(), 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: -1000,
            max_value: 900,
        }
    }

    /// Leaving out the samples of empty blocks gives every ray the same bits
    /// as sampling every step: on the CT, on its coarser copies and on a
    /// structure's shadow, at three steps and from five directions.
    #[test]
    fn skipping_empty_space_changes_no_bit() {
        let vol = ball(70, 1.6);
        let mu = MuVolume::prepare(&vol, HuMode::Water, -800.0, None);
        assert!(
            mu.occupied_fraction() < 0.8,
            "the test needs empty blocks: {}",
            mu.occupied_fraction()
        );
        let mask: Vec<u8> = vol.data.iter().map(|&v| (v > 500) as u8).collect();
        let mut volumes = mu.clone().levels(3);
        volumes.push(MuVolume::from_mask(&vol, &mask).unwrap());
        let mut checked = 0usize;
        let mut nonzero = 0usize;
        for (gantry, couch) in [
            (0.0, 0.0),
            (37.0, 15.0),
            (90.0, 0.0),
            (211.0, -20.0),
            (300.0, 5.0),
        ] {
            let cam = Geometry {
                gantry_deg: gantry,
                couch_deg: couch,
                dims: [33, 29],
                panel_mm: [260.0, 240.0],
                ..Geometry::centered_on(&vol)
            }
            .camera();
            for m in &volumes {
                for step in [0.37, 1.0, 2.2] {
                    for j in 0..cam.dims[1] {
                        for i in 0..cam.dims[0] {
                            let target = cam.pixel(i as f64, j as f64);
                            let fast = m.integrate(cam.source, target, step);
                            let slow = m.integrate_every_step(cam.source, target, step);
                            assert_eq!(
                                fast.to_bits(),
                                slow.to_bits(),
                                "pixel ({i}, {j}), gantry {gantry}, step {step}: {fast} vs {slow}"
                            );
                            checked += 1;
                            nonzero += (slow != 0.0) as usize;
                        }
                    }
                }
            }
        }
        assert!(
            nonzero > checked / 10,
            "{nonzero} of {checked} rays hit anything"
        );
    }

    /// The DRR of a moved CT is the DRR of the CT through the moved camera.
    #[test]
    fn a_pose_moves_the_camera_not_the_volume() {
        use crate::registration::RigidTransform;
        let vol = phantom(36, 2.0);
        let cam = Geometry {
            gantry_deg: 90.0,
            dims: [40, 40],
            panel_mm: [220.0, 220.0],
            ..Geometry::centered_on(&vol)
        }
        .camera();
        let mu = MuVolume::prepare(&vol, HuMode::Water, -800.0, None);
        // Shift the CT 6 mm along x by hand: the moved volume's voxel (i,j,k)
        // is the original's (i - 3, j, k).
        let mut moved = vol.clone();
        moved.origin = moved.origin + Vec3::new(6.0, 0.0, 0.0);
        let mu_moved = MuVolume::prepare(&moved, HuMode::Water, -800.0, None);
        let direct = render_posed(&mu_moved, &cam, 1.0, &Quiet).unwrap();
        // The same through the camera: T maps room to CT, so the room point
        // p sees the moved CT's p, which is the original's p - 6x.
        let t = RigidTransform::new([0.0, 0.0, 0.0, -6.0, 0.0, 0.0], Vec3::ZERO);
        let posed = render_posed(&mu, &cam.transformed(&t), 1.0, &Quiet).unwrap();
        let worst = direct
            .iter()
            .zip(&posed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "worst difference {worst}");
    }

    #[test]
    fn the_levels_halve_and_keep_their_place() {
        let vol = phantom(20, 2.0);
        let mu = MuVolume::prepare(&vol, HuMode::Water, -800.0, None);
        let total = mu.data.iter().map(|v| *v as f64).sum::<f64>();
        let levels = mu.levels(3);
        assert_eq!(levels[1].dims, [10, 10, 10]);
        assert_eq!(levels[2].dims, [5, 5, 5]);
        // The mass (attenuation × volume) is kept by a box average.
        let coarse = levels[1].data.iter().map(|v| *v as f64).sum::<f64>() * 8.0;
        assert!((coarse - total).abs() / total < 1e-6);
        // The boxes cover the same space.
        let (a, b) = levels[0].bounds();
        let (c, d) = levels[1].bounds();
        assert!((a - c).length() < 2.0 && (b - d).length() < 2.0);
        // A clip box crops to it, with a voxel of margin.
        let clipped = MuVolume::prepare(
            &vol,
            HuMode::Water,
            -800.0,
            Some((Vec3::new(-5.0, -5.0, -5.0), Vec3::new(5.0, 5.0, 5.0))),
        );
        assert!(
            clipped.dims.iter().all(|n| *n >= 5 && *n <= 9),
            "{:?}",
            clipped.dims
        );
        // Thin pixels and thick slices: the pixels coarsen first.
        let mut thick = phantom(20, 1.0);
        thick.spacing[2] = 3.0;
        let levels = MuVolume::prepare(&thick, HuMode::Water, -800.0, None).levels(4);
        let dims: Vec<[usize; 3]> = levels.iter().map(|l| l.dims).collect();
        assert_eq!(
            dims,
            vec![[20, 20, 20], [10, 10, 20], [5, 5, 20], [3, 3, 10]]
        );
        assert!((levels[2].finest_spacing() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn a_coarser_detector_covers_the_same_field() {
        let cam = Geometry {
            dims: [64, 48],
            ..Geometry::default()
        }
        .camera();
        let c = cam.coarser(4);
        assert_eq!(c.dims, [16, 12]);
        assert!((c.center() - cam.center()).length() < 1e-9);
    }
}
