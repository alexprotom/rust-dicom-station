//! The path applied: points, meshes, structure sets, masks and images at
//! any time, and the key frames a playback runs from.
//!
//! **Pushing and pulling.** Geometry is *pushed*: a contour point or a mesh
//! vertex of the first state goes through [`super::Frame::forward`] to where
//! it is at `t`, topology untouched (a mesh's normals are recomputed from
//! its moved triangles, area-weighted, as [`crate::mesh3d`] makes them).
//! Images are *pulled*: the value at a voxel `y` of the output is read from
//! a state's image at the point that is at `y` at time `t`
//! (`to_prev(y)`, `to_next(y)`), through the rigid transform that takes the
//! anchor frame into that state's own coordinates, trilinearly, the
//! outermost voxel centres included; outside a scan the value is the
//! image's lowest. A mask is pulled the same way and kept where it reads
//! one, in the box its pushed corners reach.
//!
//! **What is shown** ([`Show`]): the start state moved forward, the end
//! state moved backward, their cross-fade with weights `1 - s` and `s` (the
//! animation that lands exactly on each scan at its end), their difference,
//! or the Jacobian determinant of the path from the first state, times
//! 1000, where the tissue at `y` came from.
//!
//! **Key frames** are [`render`]ed at evenly spaced times on a working grid
//! chosen so they fit the playback budget (the first state's grid coarsened
//! until they do, [`working_grid`]); the exact frame at the slider's time
//! is rendered on the full grid when the slider stops.
//!
//! **Three slices** ([`render_planes`]): what is on screen while the slider
//! moves is three planes of the full grid, not a volume. They are the same
//! pixels [`render`] gives those planes - one per-pixel function
//! ([`Sampler`]) serves both, and the points are computed the same way, so
//! the two agree to the bit - and [`super::gpu`] computes them on the
//! inference GPU with the same arithmetic in `f32`.
//!
//! **Dose** ([`carry_dose`]): the first state's dose grid pulled along the
//! path, `D_t(y) = D(to_first(y))` - the dose *drawn* where the tissue it
//! was planned on has moved, for looking at. It is not a dose calculation:
//! nothing is recomputed for the new geometry (the beam would cross other
//! tissue), and no dose number is derived from it.
//!
//! **The Jacobian wash** ([`jacobian_now`], [`wash_plane`]): the local
//! volume change since the first state, read where the tissue now is, as a
//! colour over the image - blue compressed, red expanded, magenta folded,
//! nothing where the volume is kept, nor over air when the image is a scan.

use std::sync::Arc;

use anyhow::{bail, Result};
use rayon::prelude::*;

use super::svf::Field;
use super::{coarsen, Frame, PostureModel};
use crate::geometry::Vec3;
use crate::mesh3d::RoiMesh;
use crate::progress::{self, Progress, ProgressSink};
use crate::propagate::BoxedMask;
use crate::rtdose::DoseGrid;
use crate::rtstruct::{Contour, Roi, StructureSet};
use crate::volume::{Grid, ViewPlane, Volume};

/// Trilinear value of `vol` at `p`, the outermost voxel centres included;
/// `None` outside.
pub fn sample(vol: &Volume, p: Vec3) -> Option<f32> {
    let c = vol.patient_to_voxel(p);
    let mut i0 = [0usize; 3];
    let mut f = [0f32; 3];
    for a in 0..3 {
        let n = vol.dims[a];
        let u = c[a];
        if n == 0 || u < -1e-4 || u > (n - 1) as f64 + 1e-4 {
            return None;
        }
        let u = u.clamp(0.0, (n - 1) as f64);
        let lo = (u.floor() as usize).min(n.saturating_sub(2));
        i0[a] = lo;
        f[a] = if n == 1 { 0.0 } else { (u - lo as f64) as f32 };
    }
    let at = |di: usize, dj: usize, dk: usize| -> f32 {
        let i = (i0[0] + di).min(vol.dims[0] - 1);
        let j = (i0[1] + dj).min(vol.dims[1] - 1);
        let k = (i0[2] + dk).min(vol.dims[2] - 1);
        vol.index(i, j, k) as f32
    };
    let x00 = at(0, 0, 0) + (at(1, 0, 0) - at(0, 0, 0)) * f[0];
    let x10 = at(0, 1, 0) + (at(1, 1, 0) - at(0, 1, 0)) * f[0];
    let x01 = at(0, 0, 1) + (at(1, 0, 1) - at(0, 0, 1)) * f[0];
    let x11 = at(0, 1, 1) + (at(1, 1, 1) - at(0, 1, 1)) * f[0];
    let y0 = x00 + (x10 - x00) * f[1];
    let y1 = x01 + (x11 - x01) * f[1];
    Some(y0 + (y1 - y0) * f[2])
}

/// The centre of voxel `(i, j, k)` of `onto`, computed one way everywhere
/// ([`fill`] and [`render_planes`]), so a plane and the volume it is a plane
/// of read the same points to the bit.
#[inline]
fn voxel_point(onto: &Grid, i: usize, j: usize, k: usize) -> Vec3 {
    onto.voxel_to_patient(0.0, 0.0, k as f64)
        + onto.col_dir * (onto.spacing[1] * j as f64)
        + onto.row_dir * (onto.spacing[0] * i as f64)
}

/// Values on `onto`, one per voxel, from `value(p)` at each voxel centre.
fn fill(onto: &Grid, value: impl Fn(Vec3) -> f32 + Sync) -> Vec<f32> {
    let [nx, ny, nz] = onto.dims;
    let mut out = vec![0f32; nx * ny * nz];
    out.par_chunks_mut((nx * ny).max(1))
        .enumerate()
        .for_each(|(k, plane)| {
            for (j, row) in plane.chunks_exact_mut(nx.max(1)).enumerate() {
                for (i, o) in row.iter_mut().enumerate() {
                    *o = value(voxel_point(onto, i, j, k));
                }
            }
        });
    out
}

/// Rounded into a volume on `onto`, with its value range.
fn to_volume(onto: &Grid, values: Vec<f32>) -> Volume {
    let data: Vec<i16> = values
        .par_iter()
        .map(|v| v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16)
        .collect();
    let (lo, hi) = data
        .iter()
        .fold((i16::MAX, i16::MIN), |(a, b), &v| (a.min(v), b.max(v)));
    Volume {
        data,
        dims: onto.dims,
        spacing: onto.spacing,
        origin: onto.origin,
        row_dir: onto.row_dir,
        col_dir: onto.col_dir,
        normal: onto.normal,
        frame_of_reference_uid: onto.frame_of_reference_uid.clone(),
        min_value: if lo > hi { 0 } else { lo },
        max_value: if lo > hi { 0 } else { hi },
    }
}

/// `src` pulled onto `onto`: the value at `y` is `src` at `pull(y)`.
pub fn resample(src: &Volume, onto: &Grid, pull: impl Fn(Vec3) -> Vec3 + Sync) -> Volume {
    let fill_value = src.min_value as f32;
    to_volume(
        onto,
        fill(onto, |p| sample(src, pull(p)).unwrap_or(fill_value)),
    )
}

/// A structure set with every contour point pushed through `f`.
pub fn carry_structure_set(ss: &StructureSet, f: &Field, label: String) -> StructureSet {
    StructureSet {
        label,
        frame_of_reference_uid: ss.frame_of_reference_uid.clone(),
        sop_instance_uid: ss.sop_instance_uid.clone(),
        series_instance_uid: ss.series_instance_uid.clone(),
        study_uid: ss.study_uid.clone(),
        referenced_series_uid: ss.referenced_series_uid.clone(),
        file_name: ss.file_name.clone(),
        locked: true,
        rois: ss
            .rois
            .par_iter()
            .map(|roi| Roi {
                number: roi.number,
                name: roi.name.clone(),
                color: roi.color,
                roi_type: roi.roi_type.clone(),
                description: roi.description.clone(),
                contours: roi
                    .contours
                    .iter()
                    .map(|c| Contour {
                        points: c.points.iter().map(|&p| f.map(p)).collect(),
                        geometric_type: c.geometric_type.clone(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// A mesh with every vertex pushed through `f`; the normals recomputed.
pub fn carry_mesh(m: &RoiMesh, f: &Field) -> RoiMesh {
    let verts: Vec<[f32; 3]> = m
        .verts
        .par_iter()
        .map(|v| {
            let q = f.map(Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64));
            [q.x as f32, q.y as f32, q.z as f32]
        })
        .collect();
    RoiMesh {
        normals: normals(&verts, &m.tris),
        verts,
        ..m.clone()
    }
}

/// Every mesh of a set through `f`.
pub fn carry_meshes(ms: &[RoiMesh], f: &Field) -> Vec<RoiMesh> {
    ms.iter().map(|m| carry_mesh(m, f)).collect()
}

/// Area-weighted unit vertex normals (the cross product of two edges is
/// twice the area along the normal, so the sum weights by area).
fn normals(verts: &[[f32; 3]], tris: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0f32; 3]; verts.len()];
    for t in tris {
        let [a, b, c] = [t[0] as usize, t[1] as usize, t[2] as usize];
        if a >= verts.len() || b >= verts.len() || c >= verts.len() {
            continue;
        }
        let (pa, pb, pc) = (verts[a], verts[b], verts[c]);
        let u = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let v = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for i in [a, b, c] {
            for k in 0..3 {
                acc[i][k] += n[k];
            }
        }
    }
    acc.into_iter()
        .map(|n| {
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if l > 0.0 {
                [n[0] / l, n[1] / l, n[2] / l]
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

/// A mask on `grid` moved by the path: set where `backward` takes a voxel
/// into the mask, in the box `forward` takes the mask's box to.
pub fn carry_mask(mask: &BoxedMask, grid: &Grid, forward: &Field, backward: &Field) -> BoxedMask {
    carry_mask_by(mask, grid, |p| forward.map(p), |p| backward.map(p))
}

/// [`carry_mask`] through any pair of maps (a rigid motion, say).
pub fn carry_mask_by(
    mask: &BoxedMask,
    grid: &Grid,
    forward: impl Fn(Vec3) -> Vec3,
    backward: impl Fn(Vec3) -> Vec3 + Sync,
) -> BoxedMask {
    let dims = grid.dims;
    let Some((lo, hi)) = mask.bbox() else {
        return BoxedMask::empty(dims);
    };
    let (mut dlo, mut dhi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for &ci in &[lo[0], hi[0]] {
        for &cj in &[lo[1], hi[1]] {
            for &ck in &[lo[2], hi[2]] {
                let p = forward(grid.voxel_to_patient(ci as f64, cj as f64, ck as f64));
                let v = grid.patient_to_voxel(p);
                for a in 0..3 {
                    dlo[a] = dlo[a].min(v[a]);
                    dhi[a] = dhi[a].max(v[a]);
                }
            }
        }
    }
    // A deformation bulges between the corners: pad the box generously.
    let mut blo = [0usize; 3];
    let mut bhi = [0usize; 3];
    for a in 0..3 {
        let pad = 0.25 * (dhi[a] - dlo[a]) + 3.0;
        blo[a] = (dlo[a] - pad).floor().clamp(0.0, (dims[a] - 1) as f64) as usize;
        bhi[a] = (dhi[a] + pad).ceil().clamp(0.0, (dims[a] - 1) as f64) as usize;
    }
    let mut cells: Vec<u32> = Vec::new();
    let planes: Vec<Vec<u32>> = (blo[2]..=bhi[2])
        .into_par_iter()
        .map(|k| {
            let mut out = Vec::new();
            for j in blo[1]..=bhi[1] {
                for i in blo[0]..=bhi[0] {
                    let src = backward(grid.voxel_to_patient(i as f64, j as f64, k as f64));
                    let v = grid.patient_to_voxel(src);
                    let (a, b, c) = (v[0].round(), v[1].round(), v[2].round());
                    if a < 0.0 || b < 0.0 || c < 0.0 {
                        continue;
                    }
                    let (a, b, c) = (a as usize, b as usize, c as usize);
                    if a < dims[0] && b < dims[1] && c < dims[2] && mask.get(a, b, c) {
                        out.push((i + dims[0] * (j + dims[1] * k)) as u32);
                    }
                }
            }
            out
        })
        .collect();
    for p in planes {
        cells.extend(p);
    }
    BoxedMask::from_cells(&cells, dims)
}

/// What the views draw at a time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Show {
    /// The segment's start state, moved forward.
    #[default]
    StartForward,
    /// The segment's end state, moved backward.
    EndBackward,
    /// The two, cross-faded: `(1 - s) start + s end`.
    CrossFade,
    /// Start moved forward minus end moved backward.
    Difference,
    /// Jacobian determinant of the path from the first state, x 1000.
    Jacobian,
}

impl Show {
    pub const ALL: [Show; 5] = [
        Show::StartForward,
        Show::EndBackward,
        Show::CrossFade,
        Show::Difference,
        Show::Jacobian,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Show::StartForward => "A moving forward",
            Show::EndBackward => "B moving back",
            Show::CrossFade => "Cross-fade",
            Show::Difference => "Difference",
            Show::Jacobian => "Jacobian",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Show::StartForward => "forward",
            Show::EndBackward => "backward",
            Show::CrossFade => "crossfade",
            Show::Difference => "difference",
            Show::Jacobian => "jacobian",
        }
    }

    pub fn from_key(s: &str) -> Option<Show> {
        Show::ALL
            .into_iter()
            .find(|x| x.key().eq_ignore_ascii_case(s.trim()))
    }

    pub fn hint(self) -> &'static str {
        match self {
            Show::StartForward => "The first scan of the segment, carried along the path",
            Show::EndBackward => "The second scan of the segment, carried back along the path",
            Show::CrossFade => {
                "Both carried to the slider's time and mixed by how far along it is: exactly \
                 each scan at its end"
            }
            Show::Difference => {
                "The two carried scans subtracted: what the model does not explain shows \
                 bright or dark"
            }
            Show::Jacobian => {
                "Local volume change since the first state, times 1000: 1000 is no change, \
                 below compressed, above expanded, at or below zero folded"
            }
        }
    }

    /// The window the views open this kind of image in, when it is not a
    /// scan's own.
    pub fn window(self) -> Option<(f32, f32)> {
        match self {
            Show::Difference => Some((0.0, 600.0)),
            Show::Jacobian => Some((1000.0, 1000.0)),
            _ => None,
        }
    }
}

/// Trilinear value of a per-node scalar of `lattice` at `p`, clamped to
/// the outermost nodes.
pub fn node_scalar(lat: &super::svf::Lattice, values: &[f64], p: Vec3) -> f64 {
    let u = lat.coords(p);
    let mut b = [0usize; 3];
    let mut w = [0f64; 3];
    for a in 0..3 {
        let n = lat.dims[a];
        if n < 2 {
            continue;
        }
        let x = u[a].clamp(0.0, (n - 1) as f64);
        b[a] = (x.floor() as usize).min(n - 2);
        w[a] = x - b[a] as f64;
    }
    let mut acc = 0.0;
    for dk in 0..2 {
        for dj in 0..2 {
            for di in 0..2 {
                let (i, j, k) = (
                    (b[0] + di).min(lat.dims[0] - 1),
                    (b[1] + dj).min(lat.dims[1] - 1),
                    (b[2] + dk).min(lat.dims[2] - 1),
                );
                let wt = (if di == 1 { w[0] } else { 1.0 - w[0] })
                    * (if dj == 1 { w[1] } else { 1.0 - w[1] })
                    * (if dk == 1 { w[2] } else { 1.0 - w[2] });
                acc += wt * values[lat.index(i, j, k)];
            }
        }
    }
    acc
}

/// What one pixel of a frame is: the per-pixel function of [`render`] and
/// [`render_planes`], and what the GPU's shader transcribes.
pub struct Sampler<'a> {
    a_img: &'a Volume,
    b_img: &'a Volume,
    ta: super::se3::Rigid,
    tb: super::se3::Rigid,
    frame: &'a Frame,
    show: Show,
    low: f32,
    s: f32,
    /// `det` of the forward map at every node, for [`Show::Jacobian`].
    jac: Option<Vec<f64>>,
}

impl<'a> Sampler<'a> {
    /// The sampler of `frame` over the states' own images.
    pub fn new(
        model: &PostureModel,
        states: &'a [Arc<Volume>],
        frame: &'a Frame,
        show: Show,
    ) -> Result<Sampler<'a>> {
        let i = frame.segment;
        if states.len() < i + 2 {
            bail!(
                "the model has {} states, {} images were given",
                model.states.len(),
                states.len()
            );
        }
        let (a_img, b_img) = (&*states[i], &*states[i + 1]);
        Ok(Sampler {
            a_img,
            b_img,
            ta: model.to_state[i],
            tb: model.to_state[i + 1],
            frame,
            show,
            low: a_img.min_value.min(b_img.min_value) as f32,
            s: frame.s as f32,
            jac: (show == Show::Jacobian).then(|| frame.forward.jacobians()),
        })
    }

    /// The value at the patient point `y` of the frame's time.
    #[inline]
    pub fn value(&self, y: Vec3) -> f32 {
        let start = || sample(self.a_img, self.ta.map(self.frame.to_prev.map(y)));
        let end = || sample(self.b_img, self.tb.map(self.frame.to_next.map(y)));
        let (low, s) = (self.low, self.s);
        match self.show {
            Show::StartForward => start().unwrap_or(low),
            Show::EndBackward => end().unwrap_or(low),
            Show::CrossFade => match (start(), end()) {
                (Some(a), Some(b)) => (1.0 - s) * a + s * b,
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => low,
            },
            Show::Difference => match (start(), end()) {
                (Some(a), Some(b)) => a - b,
                _ => 0.0,
            },
            Show::Jacobian => {
                let x = self.frame.to_first.map(y);
                match self.jac.as_deref() {
                    Some(jac) => (1000.0 * node_scalar(&self.frame.forward.lattice, jac, x)) as f32,
                    None => 1000.0,
                }
            }
        }
    }
}

/// The image of `frame` on `onto`. `states` are the states' own images, in
/// their own patient coordinates, in the model's order.
pub fn render(
    model: &PostureModel,
    states: &[Arc<Volume>],
    frame: &Frame,
    show: Show,
    onto: &Grid,
) -> Result<Volume> {
    let sampler = Sampler::new(model, states, frame, show)?;
    let values = fill(onto, |y| sampler.value(y));
    let mut vol = to_volume(onto, values);
    vol.frame_of_reference_uid = model.grid.frame_of_reference_uid.clone();
    Ok(vol)
}

/// One plane of a grid: the voxels whose index along `axis` (0: `i`, the
/// sagittal planes; 1: `j`, coronal; 2: `k`, axial) is `index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plane {
    pub axis: usize,
    pub index: usize,
}

impl Plane {
    /// The plane a view of `plane` at `slice` shows.
    pub fn of_view(plane: ViewPlane, slice: usize) -> Plane {
        let axis = match plane {
            ViewPlane::Sagittal => 0,
            ViewPlane::Coronal => 1,
            ViewPlane::Axial => 2,
        };
        Plane { axis, index: slice }
    }

    /// The two axes that run along it, lower first.
    pub fn along(&self) -> [usize; 2] {
        match self.axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        }
    }

    /// Its pixels along its two axes.
    pub fn size(&self, grid: &Grid) -> [usize; 2] {
        let [a, b] = self.along();
        [grid.dims[a], grid.dims[b]]
    }

    /// The voxel at pixel `(u, v)`.
    #[inline]
    pub fn voxel(&self, u: usize, v: usize) -> [usize; 3] {
        match self.axis {
            0 => [self.index, u, v],
            1 => [u, self.index, v],
            _ => [u, v, self.index],
        }
    }

    /// Is it a plane of `grid`?
    pub fn fits(&self, grid: &Grid) -> bool {
        self.axis < 3 && self.index < grid.dims[self.axis]
    }
}

/// The values of `frame` on some planes of `onto`, each `u`-fastest along
/// [`Plane::along`]: exactly what [`render`] gives those voxels, before the
/// rounding to whole numbers.
pub fn render_planes(
    model: &PostureModel,
    states: &[Arc<Volume>],
    frame: &Frame,
    show: Show,
    onto: &Grid,
    planes: &[Plane],
) -> Result<Vec<Vec<f32>>> {
    let sampler = Sampler::new(model, states, frame, show)?;
    planes
        .iter()
        .map(|pl| {
            if !pl.fits(onto) {
                bail!("plane {pl:?} is not on the grid");
            }
            let [nu, nv] = pl.size(onto);
            let mut out = vec![0f32; nu * nv];
            out.par_chunks_mut(nu.max(1))
                .enumerate()
                .for_each(|(v, row)| {
                    for (u, o) in row.iter_mut().enumerate() {
                        let [i, j, k] = pl.voxel(u, v);
                        *o = sampler.value(voxel_point(onto, i, j, k));
                    }
                });
            Ok(out)
        })
        .collect()
}

/// Write rendered planes into a volume on their grid, rounded as
/// [`render`] rounds; the rest of the volume is left as it was.
pub fn write_planes(vol: &mut Volume, planes: &[Plane], values: &[Vec<f32>]) {
    let g = vol.grid();
    let [nx, ny, _] = vol.dims;
    for (pl, vals) in planes.iter().zip(values) {
        if !pl.fits(&g) {
            continue;
        }
        let [nu, nv] = pl.size(&g);
        if vals.len() != nu * nv {
            continue;
        }
        for v in 0..nv {
            for u in 0..nu {
                let [i, j, k] = pl.voxel(u, v);
                let x = vals[v * nu + u];
                vol.data[k * nx * ny + j * nx + i] =
                    x.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            }
        }
    }
}

/// The first state's dose drawn along the path: each dose voxel `y` of the
/// first state's dose grid takes the dose at `to_first(y)`. For display
/// only (see the module doc); the colour scale keeps the plan's maximum.
pub fn carry_dose(dose: &DoseGrid, to_first: &Field) -> DoseGrid {
    let [nx, ny, nf] = dose.dims;
    let mut data = vec![0f32; nx * ny * nf];
    let du = dose.row_dir * dose.spacing[0];
    let dv = dose.col_dir * dose.spacing[1];
    data.par_chunks_mut((nx * ny).max(1))
        .enumerate()
        .for_each(|(f, plane)| {
            let off = dose.offsets.get(f).copied().unwrap_or(0.0);
            let p0 = dose.origin + dose.normal * off;
            for (j, row) in plane.chunks_exact_mut(nx.max(1)).enumerate() {
                for (i, o) in row.iter_mut().enumerate() {
                    let y = p0 + dv * j as f64 + du * i as f64;
                    *o = dose.sample(to_first.map(y)).unwrap_or(0.0);
                }
            }
        });
    DoseGrid {
        data,
        label: format!(
            "{} - carried along the posture path, display only",
            dose.label
        ),
        ..dose.clone()
    }
}

/// The Jacobian determinant of the path from the first state, at every
/// node of the frame's lattice, read where the tissue now is: the node `y`
/// gets `det D forward` at `to_first(y)`.
pub fn jacobian_now(frame: &Frame) -> Vec<f32> {
    let jac = frame.forward.jacobians();
    let lat = &frame.forward.lattice;
    (0..lat.len())
        .into_par_iter()
        .map(|n| node_scalar(lat, &jac, frame.to_first.map(lat.node(n))) as f32)
        .collect()
}

/// The wash's colour for a Jacobian determinant `j`: unmultiplied RGBA,
/// clear within about one per cent of no change, blue where the tissue is
/// compressed and red where it is expanded (fully coloured at half or
/// twice the volume), magenta where it is folded.
pub fn wash_color(j: f32) -> [u8; 4] {
    if !j.is_finite() {
        return [0; 4];
    }
    if j <= 0.0 {
        return [230, 0, 230, 210];
    }
    let l = j.ln();
    let a = (l.abs() / std::f32::consts::LN_2).clamp(0.0, 1.0);
    if a < 0.015 {
        return [0; 4];
    }
    let alpha = (40.0 + 150.0 * a).round() as u8;
    if l < 0.0 {
        [40, 120, 255, alpha]
    } else {
        [255, 70, 30, alpha]
    }
}

/// Below this the wash is not drawn over a scan: air has no volume to
/// keep, and the blend between moving bones squeezes and stretches it
/// freely.
pub const WASH_TISSUE_HU: i16 = -500;

/// The wash of one view's slice of `grid`, in the views' pixel order (rows
/// top down, sagittal and coronal rows superior first), RGBA per pixel;
/// `values` per node of `lattice` ([`jacobian_now`]). With `tissue`, the
/// image on screen (on `grid`), only where it is above
/// [`WASH_TISSUE_HU`].
pub fn wash_plane(
    lattice: &super::svf::Lattice,
    values: &[f32],
    grid: &Grid,
    plane: ViewPlane,
    slice: usize,
    tissue: Option<&Volume>,
) -> Vec<[u8; 4]> {
    let [nx, ny, nz] = grid.dims;
    let (w, h) = match plane {
        ViewPlane::Axial => (nx, ny),
        ViewPlane::Sagittal => (ny, nz),
        ViewPlane::Coronal => (nx, nz),
    };
    if values.len() != lattice.len() || w == 0 || h == 0 {
        return vec![[0; 4]; w * h];
    }
    let vals: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    let mut out = vec![[0u8; 4]; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        for (c, o) in row.iter_mut().enumerate() {
            let (i, j, k) = match plane {
                ViewPlane::Axial => (c, r, slice.min(nz.saturating_sub(1))),
                ViewPlane::Sagittal => (slice.min(nx.saturating_sub(1)), c, nz - 1 - r),
                ViewPlane::Coronal => (c, slice.min(ny.saturating_sub(1)), nz - 1 - r),
            };
            if tissue.is_some_and(|v| v.dims == grid.dims && v.index(i, j, k) < WASH_TISSUE_HU) {
                continue;
            }
            let p = grid.voxel_to_patient(i as f64, j as f64, k as f64);
            *o = wash_color(node_scalar(lattice, &vals, p) as f32);
        }
    });
    out
}

/// What a frame carries besides its image, when asked for.
#[derive(Clone, Default)]
pub struct Extras {
    /// The first state's dose, to be drawn along the path ([`carry_dose`]).
    pub dose: Option<Arc<DoseGrid>>,
    /// The Jacobian for the wash ([`jacobian_now`]).
    pub jacobian: bool,
}

/// A frame's extras, made.
#[derive(Clone, Default)]
pub struct FrameExtras {
    pub dose: Option<Arc<DoseGrid>>,
    pub jacobian: Option<Arc<Vec<f32>>>,
}

impl Extras {
    /// The extras of `frame`.
    pub fn of(&self, frame: &Frame) -> FrameExtras {
        FrameExtras {
            dose: self
                .dose
                .as_ref()
                .map(|d| Arc::new(carry_dose(d, &frame.to_first))),
            jacobian: self.jacobian.then(|| Arc::new(jacobian_now(frame))),
        }
    }
}

/// A rendered moment of the playback.
#[derive(Clone)]
pub struct KeyFrame {
    pub t: f64,
    pub volume: Arc<Volume>,
    /// The forward field at `t`, for the structures and the meshes.
    pub forward: Arc<Field>,
    pub extras: FrameExtras,
}

/// The first state's grid coarsened until `frames` images of it fit in
/// `budget_bytes`.
pub fn working_grid(grid: &Grid, frames: usize, budget_bytes: usize) -> Grid {
    let bytes = |g: &Grid| g.dims.iter().product::<usize>() * 2 * frames.max(1);
    let mut g = grid.clone();
    let mut mm = grid.spacing.iter().cloned().fold(0.0, f64::max);
    for _ in 0..40 {
        if bytes(&g) <= budget_bytes.max(1) {
            break;
        }
        mm *= 1.25;
        g = coarsen(grid, mm);
    }
    g
}

/// `count` key frames evenly over the whole path, on `onto`.
#[allow(clippy::too_many_arguments)]
pub fn key_frames(
    model: &PostureModel,
    states: &[Arc<Volume>],
    count: usize,
    show: Show,
    onto: &Grid,
    extras: &Extras,
    p: &Progress,
) -> Result<Vec<KeyFrame>> {
    let n = count.max(2);
    let span = model.span();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        if p.cancelled() {
            bail!("key frames {}", progress::CANCELLED);
        }
        let t = span * k as f64 / (n - 1) as f64;
        p.report(k as f32 / n as f32, &format!("Key frame {}/{n}", k + 1));
        let frame = model.frame(t);
        let volume = render(model, states, &frame, show, onto)?;
        let extras = extras.of(&frame);
        out.push(KeyFrame {
            t,
            volume: Arc::new(volume),
            forward: Arc::new(frame.forward),
            extras,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posture::svf::Lattice;

    #[test]
    fn a_shifted_mesh_keeps_its_normals_and_a_mask_moves_with_the_field() {
        let lat = Lattice {
            dims: [12, 12, 12],
            spacing: [4.0; 3],
            origin: Vec3::new(-24.0, -24.0, -24.0),
            axes: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        };
        let shift = Vec3::new(4.0, 0.0, 0.0);
        let f = Field::from_map(&lat, |p| p + shift);
        let b = Field::from_map(&lat, |p| p - shift);
        let m = RoiMesh {
            roi_index: 0,
            color: [1, 2, 3],
            external: false,
            verts: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            tris: vec![[0, 1, 2]],
        };
        let moved = carry_mesh(&m, &f);
        assert!((moved.verts[1][0] - 5.0).abs() < 1e-5);
        assert!(moved.normals.iter().all(|n| (n[2] - 1.0).abs() < 1e-6));
        let grid = Grid {
            dims: [20, 20, 20],
            spacing: [2.0; 3],
            origin: Vec3::new(-19.0, -19.0, -19.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        };
        let mut mask = vec![0u8; 8000];
        mask[10 * 400 + 10 * 20 + 10] = 1;
        let boxed = BoxedMask::pack(&mask, grid.dims);
        let carried = carry_mask(&boxed, &grid, &f, &b);
        assert_eq!(carried.count(), 1);
        assert!(carried.get(12, 10, 10), "two voxels of 2 mm along x");
    }

    fn lattice() -> Lattice {
        Lattice {
            dims: [12, 12, 12],
            spacing: [4.0; 3],
            origin: Vec3::new(-24.0, -24.0, -24.0),
            axes: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        }
    }

    #[test]
    fn a_dose_is_drawn_where_its_tissue_went() {
        let lat = lattice();
        // Tissue moved 4 mm along +x: a point now at y was at y - 4.
        let to_first = Field::from_map(&lat, |p| p - Vec3::new(4.0, 0.0, 0.0));
        let (nx, ny, nf) = (10, 10, 10);
        let mut data = vec![0f32; nx * ny * nf];
        // A hot voxel at (3, 5, 5).
        data[5 * nx * ny + 5 * nx + 3] = 60.0;
        let dose = DoseGrid {
            data,
            dims: [nx, ny, nf],
            spacing: [2.0, 2.0],
            origin: Vec3::new(-10.0, -10.0, -10.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            offsets: (0..nf).map(|f| 2.0 * f as f64).collect(),
            units: "GY".into(),
            summation_type: "PLAN".into(),
            dose_type: "PHYSICAL".into(),
            max_dose: 60.0,
            frame_of_reference_uid: String::new(),
            sop_instance_uid: "1.2.3".into(),
            series_instance_uid: String::new(),
            study_uid: String::new(),
            referenced_plan_uid: String::new(),
            label: "Plan".into(),
        };
        let moved = carry_dose(&dose, &to_first);
        // The hot voxel is two dose voxels (4 mm) further along x; the scale
        // is the plan's, and the label says what it is.
        assert_eq!(moved.data[5 * nx * ny + 5 * nx + 5], 60.0);
        assert_eq!(moved.data[5 * nx * ny + 5 * nx + 3], 0.0);
        assert_eq!(moved.max_dose, 60.0);
        assert!(moved.label.contains("display only"), "{}", moved.label);
    }

    #[test]
    fn the_wash_reads_a_uniform_growth_where_the_tissue_is() {
        let lat = lattice();
        // Everything grown by 10 % along each axis about the origin.
        let s = 1.1;
        let forward = Field::from_map(&lat, |p| p * s);
        let to_first = Field::from_map(&lat, |p| p * (1.0 / s));
        let frame = Frame {
            t: 1.0,
            segment: 0,
            s: 1.0,
            forward,
            to_prev: to_first.clone(),
            to_next: Field::zero(&lat),
            to_first,
        };
        let j = jacobian_now(&frame);
        let inner = lat.index(6, 6, 6);
        assert!((j[inner] as f64 - s * s * s).abs() < 1e-3, "{}", j[inner]);
        // Expanded: red; kept: clear; folded: magenta; squeezed: blue.
        assert_eq!(wash_color(j[inner])[0], 255);
        assert_eq!(wash_color(1.0)[3], 0);
        assert_eq!(wash_color(-0.2), [230, 0, 230, 210]);
        assert_eq!(wash_color(0.7)[2], 255);
    }
}
