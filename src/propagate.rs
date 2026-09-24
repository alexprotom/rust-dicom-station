//! Carrying structures across a registration.
//!
//! Once two workspaces are aligned, the alignment is only half the answer: the
//! contours drawn on one of them have to arrive on the other. That is what
//! this module does - an RTSTRUCT ROI or a painted segmentation of one
//! workspace becomes an editable voxel mask on the other, mapped through
//! whatever [`crate::registration`] recovered, rigid or deformable, global
//! or local.
//!
//! ## Pull, never push
//!
//! Every voxel of the *destination* is asked where it comes from, rather
//! than every voxel of the source being asked where it goes. Pushing a
//! deformed mask forward leaves holes wherever the deformation expands and
//! double-writes wherever it compresses; pulling asks one question per
//! destination voxel and answers it exactly.
//!
//! ## Occupancy, and the volume kept
//!
//! A destination voxel is not a point. Asking only at its centre works when
//! the destination lattice is at least as fine as the structure - and fails
//! silently when it is not: a target made of 1 mm cubes (an ablation map
//! exported voxel by voxel) landing on 2 mm slices loses four fifths of its
//! volume, because most cubes contain no voxel centre. So each destination
//! voxel is sampled at several sub-points (the number per axis follows the
//! spacing ratio, up to four) and gets an *occupancy*, the fraction of it
//! that comes from inside the structure. The sum of the occupancies is the
//! volume of the structure as the deformation maps it, and that is the
//! volume the result keeps: the mask is filled with the most-occupied voxels
//! until it holds exactly that volume. For a structure larger than the
//! voxels this is the ½ threshold as before; for one smaller, every piece
//! lands in the voxel that holds most of it instead of vanishing.
//!
//! ## Keeping a structure's shape
//!
//! A deformable transform is only as good inside an organ as what it was
//! fitted to there. Matched on an outline (the heart's contours, say), it
//! knows where the surface goes and interpolates everything inside, and a
//! small target a few millimetres under that surface is squeezed or
//! stretched by whatever the interpolation does at that spot - a third of
//! its volume, when two differently shaped hearts are matched. With
//! [`Subject::keep_shape`] the structure is carried instead by the rigid
//! body that best fits the transform over the structure itself: it goes
//! where the transform takes it and turns as the tissue around it turns,
//! and keeps its shape and volume. How much deformation that leaves out is
//! reported, as the RMS distance between the fit and the transform over the
//! structure.
//!
//! ## The mapping cache
//!
//! The inverse of a deformable transform is a fixed-point iteration -
//! twelve evaluations of a B-spline lattice per point. Asked once per voxel
//! of a 512³ study that is billions of operations for a mapping that is
//! smooth to well under a millimetre over any few voxels. So the mapping is
//! evaluated on a lattice of a few millimetres across the destination
//! bounding box and interpolated in between: exact for a rigid transform
//! (the map is affine, and so is the interpolation) and far below the
//! contour's own accuracy for a deformable one.

use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::progress::{ProgressSink, CANCELLED};
use crate::registration::Transform3;
use crate::volume::{Grid, Volume};

use anyhow::{bail, Result};

/// One structure to carry across: its identity and its mask on the source
/// volume's grid.
pub struct Subject {
    pub name: String,
    pub color: [u8; 3],
    /// One byte per source voxel, 1 inside.
    pub mask: Vec<u8>,
    /// The structure's surface-based volume, cm³ - the volume enclosed by
    /// the closed surface reconstructed from its contours (see
    /// [`crate::rt_surface`]) - when it was drawn as contours; `None` for a
    /// segment, which has no contours to build a surface from. Carried
    /// through untouched so the report can set it beside the surface volume
    /// of what lands.
    pub surface_cm3: Option<f64>,
    /// Carry it as a rigid body - the transform's best rigid fit over the
    /// structure's own voxels - rather than through the transform itself:
    /// shape and volume kept, place and orientation followed. See the
    /// module notes.
    pub keep_shape: bool,
}

/// What arrived on the other side.
pub struct Propagated {
    pub name: String,
    pub color: [u8; 3],
    /// One byte per *destination* voxel.
    pub mask: Vec<u8>,
    pub voxels: usize,
    /// Volume of the source structure, cm³.
    pub source_cm3: f64,
    /// Volume of the propagated structure, cm³ - the two differ by exactly
    /// what the deformation compressed or expanded, which is the first thing
    /// worth checking about a propagated contour.
    pub result_cm3: f64,
    /// Volume of the structure as the transform maps it, cm³, before it was
    /// filed on the destination lattice: the sum of the occupancies. The
    /// mask holds this to within one voxel.
    pub mapped_cm3: f64,
    /// The source structure's surface-based volume, cm³, when it was
    /// contours (see [`Subject::surface_cm3`]).
    pub source_surface_cm3: Option<f64>,
    /// When it was carried rigidly ([`Subject::keep_shape`]): the RMS
    /// distance between that rigid body and the transform over the
    /// structure, mm - the deformation that was left out.
    pub rigid_residual_mm: Option<f64>,
}

impl Propagated {
    /// `liver: 1642.318 cm³ ▶ 1701.052 cm³ (+3.576 %)`.
    pub fn summary(&self) -> String {
        let change = if self.source_cm3 > 1e-9 {
            100.0 * (self.result_cm3 - self.source_cm3) / self.source_cm3
        } else {
            0.0
        };
        let rigid = match self.rigid_residual_mm {
            Some(r) => format!(", carried rigidly, {r:.1} mm of deformation left out"),
            None => String::new(),
        };
        format!(
            "{}: {:.2} cm³ ▶ {:.2} cm³ ({:+.2} %){rigid}",
            self.name, self.source_cm3, self.result_cm3, change
        )
    }
}

/// The destination → source mapping, sampled on a lattice in destination
/// voxel coordinates and interpolated in between.
struct MapCache {
    /// First destination voxel the lattice covers.
    lo: [usize; 3],
    /// Node spacing in destination voxels.
    step: [usize; 3],
    dims: [usize; 3],
    /// Source patient position of each node.
    nodes: Vec<Vec3>,
}

impl MapCache {
    /// Build the cache over the destination voxel box `lo..=hi`.
    fn build(
        dst: &Volume,
        lo: [usize; 3],
        hi: [usize; 3],
        map: &(dyn Fn(Vec3) -> Vec3 + Sync),
        node_mm: f64,
    ) -> MapCache {
        let mut step = [1usize; 3];
        let mut dims = [2usize; 3];
        for a in 0..3 {
            step[a] = ((node_mm / dst.spacing[a]).round() as usize).max(1);
            // +2: one node past the far edge, so every voxel is bracketed.
            dims[a] = (hi[a] - lo[a]) / step[a] + 2;
        }
        let [nx, ny, nz] = dims;
        let nodes: Vec<Vec3> = (0..nz)
            .into_par_iter()
            .flat_map(|k| {
                let mut plane = Vec::with_capacity(nx * ny);
                for j in 0..ny {
                    for i in 0..nx {
                        let p = dst.voxel_to_patient(
                            (lo[0] + i * step[0]) as f64,
                            (lo[1] + j * step[1]) as f64,
                            (lo[2] + k * step[2]) as f64,
                        );
                        plane.push(map(p));
                    }
                }
                plane
            })
            .collect();
        MapCache {
            lo,
            step,
            dims,
            nodes,
        }
    }

    /// Source patient position of a destination point, in (fractional)
    /// destination voxel indices.
    #[inline]
    fn at(&self, i: f64, j: f64, k: f64) -> Vec3 {
        let u = [
            ((i - self.lo[0] as f64) / self.step[0] as f64).max(0.0),
            ((j - self.lo[1] as f64) / self.step[1] as f64).max(0.0),
            ((k - self.lo[2] as f64) / self.step[2] as f64).max(0.0),
        ];
        let i0 = (u[0] as usize).min(self.dims[0] - 2);
        let j0 = (u[1] as usize).min(self.dims[1] - 2);
        let k0 = (u[2] as usize).min(self.dims[2] - 2);
        let (fu, fv, fw) = (u[0] - i0 as f64, u[1] - j0 as f64, u[2] - k0 as f64);
        let idx =
            |i: usize, j: usize, k: usize| self.nodes[i + self.dims[0] * (j + self.dims[1] * k)];
        let lerp = |a: Vec3, b: Vec3, t: f64| a * (1.0 - t) + b * t;
        let c00 = lerp(idx(i0, j0, k0), idx(i0 + 1, j0, k0), fu);
        let c10 = lerp(idx(i0, j0 + 1, k0), idx(i0 + 1, j0 + 1, k0), fu);
        let c01 = lerp(idx(i0, j0, k0 + 1), idx(i0 + 1, j0, k0 + 1), fu);
        let c11 = lerp(idx(i0, j0 + 1, k0 + 1), idx(i0 + 1, j0 + 1, k0 + 1), fu);
        lerp(lerp(c00, c10, fv), lerp(c01, c11, fv), fw)
    }
}

/// Trilinear sample of a binary mask at fractional source voxel indices.
#[inline]
fn sample_mask(mask: &[u8], dims: [usize; 3], v: [f64; 3]) -> f32 {
    let [nx, ny, nz] = dims;
    if v[0] < 0.0 || v[1] < 0.0 || v[2] < 0.0 {
        return 0.0;
    }
    let (i0, j0, k0) = (v[0] as usize, v[1] as usize, v[2] as usize);
    if i0 + 1 >= nx || j0 + 1 >= ny || k0 + 1 >= nz {
        return 0.0;
    }
    let (fu, fv, fw) = (
        (v[0] - i0 as f64) as f32,
        (v[1] - j0 as f64) as f32,
        (v[2] - k0 as f64) as f32,
    );
    let at = |i: usize, j: usize, k: usize| mask[k * nx * ny + j * nx + i] as f32;
    let c00 = at(i0, j0, k0) + (at(i0 + 1, j0, k0) - at(i0, j0, k0)) * fu;
    let c10 = at(i0, j0 + 1, k0) + (at(i0 + 1, j0 + 1, k0) - at(i0, j0 + 1, k0)) * fu;
    let c01 = at(i0, j0, k0 + 1) + (at(i0 + 1, j0, k0 + 1) - at(i0, j0, k0 + 1)) * fu;
    let c11 = at(i0, j0 + 1, k0 + 1) + (at(i0 + 1, j0 + 1, k0 + 1) - at(i0, j0 + 1, k0 + 1)) * fu;
    let c0 = c00 + (c10 - c00) * fv;
    let c1 = c01 + (c11 - c01) * fv;
    c0 + (c1 - c0) * fw
}

/// A point mapping, destination → source or back.
type MapFn<'a> = Box<dyn Fn(Vec3) -> Vec3 + Sync + 'a>;

/// Points of a structure the rigid fit is made on, at most.
const RIGID_FIT_POINTS: usize = 4000;

/// The rigid body that best explains `to_dst` over the structure `mask` on
/// `src` (orthogonal Procrustes on its voxel centres), as a destination →
/// source mapping, with the RMS distance between the two over the
/// structure, mm. `None` for an empty mask.
fn rigid_fit_over(
    src: &Volume,
    mask: &[u8],
    to_dst: &(dyn Fn(Vec3) -> Vec3 + Sync),
) -> Option<(crate::registration::RigidTransform, f64)> {
    let n = crate::morphology::count_set(mask);
    if n == 0 {
        return None;
    }
    let stride = n.div_ceil(RIGID_FIT_POINTS);
    let [nx, ny, _] = src.dims;
    let at: Vec<Vec3> = mask
        .iter()
        .enumerate()
        .filter(|(_, &m)| m != 0)
        .step_by(stride)
        .map(|(i, _)| {
            src.voxel_to_patient(
                (i % nx) as f64,
                ((i / nx) % ny) as f64,
                (i / (nx * ny)) as f64,
            )
        })
        .collect();
    let lands: Vec<Vec3> = at.par_iter().map(|&q| to_dst(q)).collect();
    // Fitted destination → source: from where each point lands, back to
    // where it is.
    let fit = crate::registration::analysis::fit_rigid(&lands, &at);
    let centre = lands.iter().fold(Vec3::ZERO, |a, b| a + *b) * (1.0 / lands.len() as f64);
    let r = fit.rotation_deg.map(f64::to_radians);
    let t = fit.translation;
    Some((
        crate::registration::RigidTransform::new([r[0], r[1], r[2], t.x, t.y, t.z], centre),
        fit.residual_mm,
    ))
}

/// Carry `subjects` from `src` onto `dst` through `t`.
///
/// `use_inverse` says which way the transform runs relative to the two
/// volumes: the transform always maps *fixed* patient coordinates to
/// *moving* ones, so propagating onto the moving workspace needs its inverse
/// and propagating onto the fixed one does not.
pub fn propagate(
    src: &Volume,
    dst: &Volume,
    t: &Transform3,
    use_inverse: bool,
    subjects: &[Subject],
    sink: &dyn ProgressSink,
) -> Result<Vec<Propagated>> {
    if subjects.is_empty() {
        bail!("nothing selected to propagate");
    }
    let src_vox_cm3 = src.voxel_cm3();
    let dst_vox_cm3 = dst.voxel_cm3();
    // Destination → source, and its opposite (used only to find the box).
    let to_src = move |p: Vec3| if use_inverse { t.unmap(p) } else { t.map(p) };
    let to_dst = move |p: Vec3| if use_inverse { t.map(p) } else { t.unmap(p) };

    let n = subjects.len();
    let mut out = Vec::with_capacity(n);
    for (si, s) in subjects.iter().enumerate() {
        if sink.cancelled() {
            bail!(CANCELLED);
        }
        sink.report(
            si as f32 / n as f32,
            &format!("Propagating {} ({}/{n})", s.name, si + 1),
        );
        if s.mask.len() != src.dims[0] * src.dims[1] * src.dims[2] {
            continue;
        }
        let Some((slo, shi)) = crate::morphology::mask_bbox(&s.mask, src.dims) else {
            continue;
        };
        let source_voxels = crate::morphology::count_set(&s.mask);
        // The mapping this structure is pulled along: the transform, or
        // its rigid fit over the structure.
        let fitted = if s.keep_shape {
            rigid_fit_over(src, &s.mask, &to_dst)
        } else {
            None
        };
        let (to_src, to_dst): (MapFn, MapFn) = match &fitted {
            Some((r, _)) => (Box::new(|p| r.map(p)), Box::new(|p| r.unmap(p))),
            None => (Box::new(to_src), Box::new(to_dst)),
        };
        let rigid_residual_mm = fitted.as_ref().map(|(_, res)| *res);

        // Where does this structure land in the destination? Map the eight
        // corners of its box across, then keep a generous margin: for a
        // deformable transform the image of a box is not a box, and a
        // clipped structure is a silent error nobody would notice.
        // The margin covers what a deformable map does inside the box that
        // its corners do not show: a quarter of the box plus 20 mm. It is
        // deliberately not the distance the corners travelled - between two
        // frames of reference that is the whole patient, and the box would
        // become the volume.
        let (mut dlo, mut dhi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for ci in 0..8 {
            let c = src.voxel_to_patient(
                if ci & 1 == 0 { slo[0] } else { shi[0] } as f64,
                if ci & 2 == 0 { slo[1] } else { shi[1] } as f64,
                if ci & 4 == 0 { slo[2] } else { shi[2] } as f64,
            );
            let v = dst.patient_to_voxel(to_dst(c));
            for a in 0..3 {
                dlo[a] = dlo[a].min(v[a]);
                dhi[a] = dhi[a].max(v[a]);
            }
        }
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for a in 0..3 {
            let margin = 20.0 / dst.spacing[a] + 0.25 * (dhi[a] - dlo[a]) + 4.0;
            lo[a] = (dlo[a] - margin).max(0.0) as usize;
            hi[a] = ((dhi[a] + margin) as i64).clamp(0, dst.dims[a] as i64 - 1) as usize;
            if lo[a] > hi[a] {
                lo[a] = hi[a];
            }
        }
        if hi.iter().zip(&lo).any(|(h, l)| h <= l) {
            // The structure maps entirely outside the destination volume.
            out.push(Propagated {
                name: s.name.clone(),
                color: s.color,
                mask: vec![0u8; dst.dims[0] * dst.dims[1] * dst.dims[2]],
                voxels: 0,
                source_cm3: source_voxels as f64 * src_vox_cm3,
                result_cm3: 0.0,
                mapped_cm3: 0.0,
                source_surface_cm3: s.surface_cm3,
                rigid_residual_mm,
            });
            continue;
        }

        let cache = MapCache::build(dst, lo, hi, &*to_src, 3.0);
        let [dnx, dny, dnz] = dst.dims;
        let src_dims = src.dims;
        let src_mask = &s.mask;
        // Sub-points per axis: enough that a source voxel is not skipped
        // over, capped so a 512³ box stays affordable.
        let sub = [0, 1, 2]
            .map(|a| ((dst.spacing[a] / src.spacing[a]).round() as usize).clamp(1, MAX_SUBSAMPLES));
        let offsets = subpoint_offsets(sub);
        let weight = 1.0 / offsets.len() as f32;
        // Occupancy of every destination voxel in the box, as (voxel, o).
        let rows: Vec<Vec<(usize, f32)>> = (lo[2]..=hi[2])
            .into_par_iter()
            .map(|k| {
                let mut plane = Vec::new();
                for j in lo[1]..=hi[1] {
                    for i in lo[0]..=hi[0] {
                        let mut o = 0.0f32;
                        for d in &offsets {
                            let p = cache.at(i as f64 + d[0], j as f64 + d[1], k as f64 + d[2]);
                            let v = src.patient_to_voxel(p);
                            o += sample_mask(src_mask, src_dims, v);
                        }
                        let o = o * weight;
                        if o > 0.0 {
                            plane.push((k * dnx * dny + j * dnx + i, o));
                        }
                    }
                }
                plane
            })
            .collect();
        let mut cells: Vec<(usize, f32)> = rows.into_iter().flatten().collect();
        let mapped_voxels: f64 = cells.iter().map(|(_, o)| *o as f64).sum();
        // Fill with the most-occupied voxels until the mapped volume is held.
        let keep = mapped_voxels.round() as usize;
        cells.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut mask = vec![0u8; dnx * dny * dnz];
        let mut voxels = 0usize;
        for (idx, _) in cells.into_iter().take(keep) {
            mask[idx] = 1;
            voxels += 1;
        }
        out.push(Propagated {
            name: s.name.clone(),
            color: s.color,
            mask,
            voxels,
            source_cm3: source_voxels as f64 * src_vox_cm3,
            result_cm3: voxels as f64 * dst_vox_cm3,
            mapped_cm3: mapped_voxels * dst_vox_cm3,
            source_surface_cm3: s.surface_cm3,
            rigid_residual_mm,
        });
    }
    sink.report(1.0, "done");
    Ok(out)
}

/// What is done to a propagated mask after it lands: a structure that
/// arrives as a cloud of pieces (an ablation map exported voxel by voxel, a
/// contour carried onto a coarser lattice) can be closed into one surface
/// and, if wanted, filled into one solid.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Finish {
    /// Morphological closing radius, mm: gaps narrower than twice this are
    /// bridged. 0 leaves the mask as it landed.
    pub close_mm: f64,
    /// Fill the interior afterwards (slice by slice, so a shell becomes a
    /// solid and a solid stays one).
    pub fill: bool,
    /// Carry the structures rigidly ([`Subject::keep_shape`]) - not
    /// something done after landing, but chosen beside it, so it travels
    /// with the rest of how a run lands its structures. The workflows set
    /// it on their subjects; an anchor that is the run's own check follows
    /// the transform regardless.
    pub keep_shape: bool,
}

impl Finish {
    /// Is nothing done to the masks once they have landed? (Keeping the
    /// shape is how they land, not something done afterwards.)
    pub fn is_none(&self) -> bool {
        self.close_mm <= 0.0 && !self.fill
    }

    /// `closed 2 mm, filled`, `shape kept`.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.keep_shape {
            parts.push("shape kept".to_string());
        }
        if self.close_mm > 0.0 {
            parts.push(format!("closed {:.0} mm", self.close_mm));
        }
        if self.fill {
            parts.push("filled".into());
        }
        if parts.is_empty() {
            "as landed".into()
        } else {
            parts.join(", ")
        }
    }

    /// Mark `subjects` to be carried as this asks.
    pub fn carry(&self, subjects: &mut [Subject]) {
        for s in subjects {
            s.keep_shape |= self.keep_shape;
        }
    }

    /// Apply to a landed structure on `grid`, updating its voxel count and
    /// filed volume; the mapped volume stays what the transform made of it.
    ///
    /// Closing a *cloud* is not the textbook closing: dilation then erosion
    /// by the same ball gives two nearby points back as two points, since
    /// the ball never fits between them. So the pieces are dilated by the
    /// radius (which joins everything closer than twice it) and eroded by
    /// half of it, leaving one surface about a radius thicker than the
    /// cloud was. With `fill` the interior is filled between the two, and
    /// the erosion is the full radius: the solid's surface comes back to
    /// where the cloud was.
    pub fn apply(&self, item: &mut Propagated, grid: &Grid, sink: &dyn ProgressSink) {
        use crate::morphology as morph;
        if self.is_none() || item.voxels == 0 {
            return;
        }
        let (dims, spacing) = (grid.dims, grid.spacing);
        let axis = grid.canonical_axes().0[0];
        if self.close_mm > 0.0 {
            sink.report(0.0, "Closing gaps");
            let r = self.close_mm;
            let mut m = morph::dilate_mm(&item.mask, dims, spacing, r);
            if self.fill {
                sink.report(0.4, "Filling");
                morph::fill_holes_2d(&mut m, dims, axis);
            }
            let back = if self.fill { r } else { 0.5 * r };
            let eroded = morph::erode_mm(&m, dims, spacing, back);
            // The erosion must not lose what landed: the cloud itself stays.
            let original = std::mem::take(&mut item.mask);
            item.mask = eroded
                .iter()
                .zip(&original)
                .map(|(&e, &o)| u8::from(e != 0 || o != 0))
                .collect();
        } else if self.fill {
            sink.report(0.0, "Filling");
            morph::fill_holes_2d(&mut item.mask, dims, axis);
        }
        item.voxels = crate::morphology::count_set(&item.mask);
        item.result_cm3 = item.voxels as f64 * grid.voxel_cm3();
    }

    /// Apply to every landed structure.
    pub fn apply_all(&self, items: &mut [Propagated], grid: &Grid, sink: &dyn ProgressSink) {
        for it in items {
            self.apply(it, grid, sink);
        }
    }
}

/// Sub-points per axis a destination voxel is sampled at, at most.
const MAX_SUBSAMPLES: usize = 4;

/// The offsets (in destination voxel units, about the centre) of a voxel's
/// sub-points: `sub[a]` evenly spaced along axis `a`, centred, so one point
/// per axis is the voxel centre and the pattern never touches the faces.
fn subpoint_offsets(sub: [usize; 3]) -> Vec<[f64; 3]> {
    let axis =
        |n: usize| -> Vec<f64> { (0..n).map(|i| (i as f64 + 0.5) / n as f64 - 0.5).collect() };
    let (xs, ys, zs) = (axis(sub[0]), axis(sub[1]), axis(sub[2]));
    let mut out = Vec::with_capacity(sub[0] * sub[1] * sub[2]);
    for &z in &zs {
        for &y in &ys {
            for &x in &xs {
                out.push([x, y, z]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::Quiet;
    use crate::registration::RigidTransform;

    fn vol(dims: [usize; 3], spacing: f64, origin: Vec3) -> Volume {
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [spacing; 3],
            origin,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 1,
        }
    }

    /// A solid ball of radius `r` mm about `c`, on `v`'s grid.
    fn ball(v: &Volume, c: Vec3, r: f64) -> Vec<u8> {
        let [nx, ny, nz] = v.dims;
        let mut m = vec![0u8; nx * ny * nz];
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = v.voxel_to_patient(i as f64, j as f64, k as f64);
                    if (p - c).length() <= r {
                        m[k * nx * ny + j * nx + i] = 1;
                    }
                }
            }
        }
        m
    }

    /// A structure carried with its shape kept goes where the transform
    /// takes it and keeps its volume, where the transform itself would
    /// have shrunk it to half; the report says how much deformation was
    /// left out.
    #[test]
    fn a_structure_carried_with_its_shape_kept_keeps_its_volume() {
        use crate::registration::Mat4;
        let src = vol([48, 48, 48], 1.0, Vec3::ZERO);
        let dst = vol([48, 48, 48], 1.0, Vec3::ZERO);
        let c = Vec3::new(20.0, 22.0, 24.0);
        let mask = ball(&src, c, 8.0);
        // Destination -> source: a scale by 1.25 about (24, 24, 24) and a
        // shift, so the ball lands 0.8 times the size (half the volume).
        let (o, k) = (24.0, 1.25);
        let m = Mat4([
            [k, 0.0, 0.0, o * (1.0 - k) + 1.0],
            [0.0, k, 0.0, o * (1.0 - k) - 2.0],
            [0.0, 0.0, k, o * (1.0 - k)],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let t = Transform3::from_matrix(m, Vec3::ZERO);
        let run = |keep_shape: bool| {
            propagate(
                &src,
                &dst,
                &t,
                false,
                &[Subject {
                    name: "ball".into(),
                    color: [255, 0, 0],
                    mask: mask.clone(),
                    surface_cm3: None,
                    keep_shape,
                }],
                &Quiet,
            )
            .unwrap()
            .pop()
            .unwrap()
        };
        let grid = dst.grid();
        let centroid = |m: &[u8]| crate::motion::centroid_mm(m, &grid).unwrap();
        let deformed = run(false);
        let kept = run(true);
        let ratio = |it: &Propagated| it.result_cm3 / it.source_cm3;
        assert!(
            (ratio(&deformed) - 0.512).abs() < 0.05,
            "{}",
            ratio(&deformed)
        );
        assert!((ratio(&kept) - 1.0).abs() < 0.03, "{}", ratio(&kept));
        assert!(deformed.rigid_residual_mm.is_none());
        // The rigid body leaves out the scale: points 4 mm from the centre
        // on average, a fifth of that each.
        let res = kept.rigid_residual_mm.expect("reported");
        assert!(res > 0.5 && res < 1.5, "{res}");
        // Where the transform took it, within a voxel.
        let d = centroid(&kept.mask) - centroid(&deformed.mask);
        assert!(d.length() < 1.0, "{d:?}");
        assert!(kept.summary().contains("carried rigidly"));
    }

    #[test]
    fn a_translation_carries_a_structure_by_exactly_that_much() {
        let dims = [40, 40, 40];
        let src = vol(dims, 2.0, Vec3::new(-40.0, -40.0, -40.0));
        let dst = vol(dims, 2.0, Vec3::new(-40.0, -40.0, -40.0));
        let centre = Vec3::new(0.0, 0.0, 0.0);
        let mask = ball(&src, centre, 12.0);
        let shift = Vec3::new(6.0, -4.0, 2.0);
        // fixed → moving is a shift; src is the fixed workspace, dst the moving
        // one, so the propagation runs through the inverse.
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, shift.x, shift.y, shift.z],
            Vec3::ZERO,
        ));
        let out = propagate(
            &src,
            &dst,
            &t,
            true,
            &[Subject {
                name: "ball".into(),
                color: [255, 0, 0],
                mask,
                surface_cm3: None,
                keep_shape: false,
            }],
            &Quiet,
        )
        .unwrap();
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert!(p.voxels > 0);
        // A rigid transform preserves volume, so the two must agree closely.
        assert!(
            (p.result_cm3 - p.source_cm3).abs() < 0.06 * p.source_cm3,
            "{}",
            p.summary()
        );
        // The propagated ball must be centred on the shifted centre.
        let [nx, ny, _] = dst.dims;
        let mut sum = Vec3::ZERO;
        let mut n = 0.0;
        for (o, &v) in p.mask.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let k = o / (nx * ny);
            let rem = o - k * nx * ny;
            sum = sum + dst.voxel_to_patient((rem % nx) as f64, (rem / nx) as f64, k as f64);
            n += 1.0;
        }
        let got = sum * (1.0 / n);
        assert!(
            (got - (centre + shift)).length() < 0.5,
            "centre {got:?} vs {:?}",
            centre + shift
        );
    }

    #[test]
    fn a_cloud_of_millimetre_cubes_keeps_its_volume_on_a_coarse_lattice() {
        // The STAR case: an ablation map exported as 1 mm cubes on a 0.5 mm
        // cardiac CT, carried onto a 4DCT at 1.2 x 1.2 x 2 mm. Centre
        // sampling lost four fifths of it; the occupancy fill must not.
        let src = Volume {
            spacing: [0.5; 3],
            ..vol([80, 80, 80], 0.5, Vec3::new(-20.0, -20.0, -20.0))
        };
        let mut dst = vol([40, 40, 24], 1.2, Vec3::new(-24.0, -24.0, -24.0));
        dst.spacing = [1.2, 1.2, 2.0];
        let [nx, ny, nz] = src.dims;
        let mut mask = vec![0u8; nx * ny * nz];
        // A pseudo-random cloud of 1 mm cubes (2 x 2 x 2 source voxels) at
        // integer-millimetre positions, none touching.
        let mut seed = 12345u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut cubes = 0;
        while cubes < 150 {
            let cx = 2 * (4 + (next() % 32) as usize);
            let cy = 2 * (4 + (next() % 32) as usize);
            let cz = 2 * (4 + (next() % 32) as usize);
            let free = (0..3).all(|dz| {
                (0..3).all(|dy| {
                    (0..3).all(|dx| mask[(cz + dz) * nx * ny + (cy + dy) * nx + cx + dx] == 0)
                })
            });
            if !free {
                continue;
            }
            for dz in 0..2 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        mask[(cz + dz) * nx * ny + (cy + dy) * nx + cx + dx] = 1;
                    }
                }
            }
            cubes += 1;
        }
        let t = Transform3::rigid_only(RigidTransform::identity(Vec3::ZERO));
        let out = propagate(
            &src,
            &dst,
            &t,
            false,
            &[Subject {
                name: "cloud".into(),
                color: [255, 0, 0],
                mask,
                surface_cm3: None,
                keep_shape: false,
            }],
            &Quiet,
        )
        .unwrap();
        let p = &out[0];
        eprintln!("{}  (mapped {:.3} cm³)", p.summary(), p.mapped_cm3);
        assert!(
            (p.mapped_cm3 - p.source_cm3).abs() < 0.1 * p.source_cm3,
            "the occupancies sum to the source volume: {} vs {}",
            p.mapped_cm3,
            p.source_cm3
        );
        assert!(
            (p.result_cm3 - p.source_cm3).abs() < 0.15 * p.source_cm3,
            "the filed mask keeps the volume: {}",
            p.summary()
        );
    }

    #[test]
    fn a_landed_cloud_can_be_closed_into_one_surface_and_filled() {
        // A spherical shell sampled as separate dots on a coarse lattice:
        // closing joins the dots into one surface, filling makes a solid.
        let dst = vol([40, 40, 40], 1.5, Vec3::new(-30.0, -30.0, -30.0));
        let [nx, ny, nz] = dst.dims;
        let mut mask = vec![0u8; nx * ny * nz];
        let mut dots = 0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let p = dst.voxel_to_patient(i as f64, j as f64, k as f64);
                    let r = p.length();
                    // The shell, every other voxel of it: a cloud.
                    if (r - 15.0).abs() < 0.8 && (i + j + k) % 2 == 0 {
                        mask[k * nx * ny + j * nx + i] = 1;
                        dots += 1;
                    }
                }
            }
        }
        let grid = dst.grid();
        let mut item = Propagated {
            name: "shell".into(),
            color: [255, 0, 0],
            mask,
            voxels: dots,
            source_cm3: 0.0,
            result_cm3: 0.0,
            mapped_cm3: 0.0,
            source_surface_cm3: None,
            rigid_residual_mm: None,
        };
        let pieces = crate::morphology::components(&item.mask, dst.dims).len();
        assert!(pieces > 50, "a cloud to begin with: {pieces} pieces");
        // Dots 2.1 mm apart on 1.5 mm voxels: a 3 mm radius joins them.
        let mut surface = Propagated {
            name: item.name.clone(),
            color: item.color,
            mask: item.mask.clone(),
            voxels: item.voxels,
            source_cm3: 0.0,
            result_cm3: 0.0,
            mapped_cm3: 0.0,
            source_surface_cm3: None,
            rigid_residual_mm: None,
        };
        Finish {
            close_mm: 3.0,
            fill: false,
            ..Finish::default()
        }
        .apply(&mut surface, &grid, &Quiet);
        let closed = crate::morphology::components(&surface.mask, dst.dims).len();
        assert_eq!(closed, 1, "closing joins the dots into one surface");
        assert!(
            surface.voxels > dots && surface.voxels < 6 * dots,
            "a surface a few millimetres thick, not a ball: {} from {dots} dots",
            surface.voxels
        );
        // Closed and filled in one go: a solid whose surface is where the
        // cloud was, so its volume is the ball's.
        Finish {
            close_mm: 3.0,
            fill: true,
            ..Finish::default()
        }
        .apply(&mut item, &grid, &Quiet);
        let ball = 4.0 / 3.0 * std::f64::consts::PI * 15.0f64.powi(3) / 1000.0;
        assert_eq!(crate::morphology::components(&item.mask, dst.dims).len(), 1);
        assert!(
            (item.result_cm3 - ball).abs() < 0.2 * ball,
            "filled to a solid ball: {:.1} cm³ vs {ball:.1}",
            item.result_cm3
        );
    }

    #[test]
    fn the_direction_flag_decides_which_way_the_structure_travels() {
        let dims = [32, 32, 32];
        let src = vol(dims, 2.0, Vec3::new(-32.0, -32.0, -32.0));
        let dst = vol(dims, 2.0, Vec3::new(-32.0, -32.0, -32.0));
        let mask = ball(&src, Vec3::ZERO, 10.0);
        let shift = Vec3::new(8.0, 0.0, 0.0);
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, shift.x, 0.0, 0.0],
            Vec3::ZERO,
        ));
        let subject = || {
            vec![Subject {
                name: "ball".into(),
                color: [0, 255, 0],
                mask: mask.clone(),
                surface_cm3: None,
                keep_shape: false,
            }]
        };
        let centroid = |m: &[u8]| {
            let [nx, ny, _] = dst.dims;
            let mut sum = Vec3::ZERO;
            let mut n = 0.0f64;
            for (o, &v) in m.iter().enumerate() {
                if v == 0 {
                    continue;
                }
                let k = o / (nx * ny);
                let rem = o - k * nx * ny;
                sum = sum + dst.voxel_to_patient((rem % nx) as f64, (rem / nx) as f64, k as f64);
                n += 1.0;
            }
            sum * (1.0 / n.max(1.0))
        };
        let fwd = propagate(&src, &dst, &t, true, &subject(), &Quiet).unwrap();
        let bwd = propagate(&src, &dst, &t, false, &subject(), &Quiet).unwrap();
        assert!((centroid(&fwd[0].mask).x - 8.0).abs() < 0.5);
        assert!((centroid(&bwd[0].mask).x + 8.0).abs() < 0.5);
    }

    #[test]
    fn a_structure_that_leaves_the_volume_comes_back_empty_not_wrong() {
        let dims = [24, 24, 24];
        let src = vol(dims, 2.0, Vec3::new(-24.0, -24.0, -24.0));
        let dst = vol(dims, 2.0, Vec3::new(-24.0, -24.0, -24.0));
        let mask = ball(&src, Vec3::ZERO, 8.0);
        // Half a metre away: nothing of it can land inside.
        let t = Transform3::rigid_only(RigidTransform::new(
            [0.0, 0.0, 0.0, 500.0, 0.0, 0.0],
            Vec3::ZERO,
        ));
        let out = propagate(
            &src,
            &dst,
            &t,
            true,
            &[Subject {
                name: "gone".into(),
                color: [1, 2, 3],
                mask,
                surface_cm3: None,
                keep_shape: false,
            }],
            &Quiet,
        )
        .unwrap();
        assert_eq!(out[0].voxels, 0);
        assert_eq!(out[0].result_cm3, 0.0);
        assert!(out[0].source_cm3 > 0.0);
        assert!(out[0].summary().contains("gone"));
        assert!(propagate(&src, &dst, &t, true, &[], &Quiet).is_err());
    }

    #[test]
    fn structures_cross_between_grids_of_different_spacing() {
        // The destination has its own geometry - a propagated contour has to
        // land on *its* voxels, not on a copy of the source lattice.
        let src = vol([40, 40, 40], 2.0, Vec3::new(-40.0, -40.0, -40.0));
        let dst = vol([27, 27, 27], 3.0, Vec3::new(-39.0, -39.0, -39.0));
        let mask = ball(&src, Vec3::new(4.0, -2.0, 6.0), 14.0);
        let t = Transform3::rigid_only(RigidTransform::identity(Vec3::ZERO));
        let out = propagate(
            &src,
            &dst,
            &t,
            true,
            &[Subject {
                name: "ball".into(),
                color: [9, 9, 9],
                mask,
                surface_cm3: None,
                keep_shape: false,
            }],
            &Quiet,
        )
        .unwrap();
        let p = &out[0];
        assert_eq!(p.mask.len(), 27 * 27 * 27);
        assert!(p.voxels > 0);
        assert!(
            (p.result_cm3 - p.source_cm3).abs() < 0.1 * p.source_cm3,
            "{}",
            p.summary()
        );
    }
}
