//! Carrying structures across a registration.
//!
//! Once two datasets are aligned, the alignment is only half the answer: the
//! contours drawn on one of them have to arrive on the other. That is what
//! this module does - an RTSTRUCT ROI or a painted segmentation of one
//! dataset becomes an editable voxel mask on the other, mapped through
//! whatever [`crate::registration`] recovered, rigid or deformable, global
//! or local.
//!
//! ## Pull, never push
//!
//! Every voxel of the *destination* is asked where it comes from, rather
//! than every voxel of the source being asked where it goes. Pushing a
//! deformed mask forward leaves holes wherever the deformation expands and
//! double-writes wherever it compresses; pulling asks one question per
//! destination voxel and answers it exactly. The source mask is sampled
//! trilinearly and thresholded at ½, so the boundary lands where the
//! contour really is rather than on the nearest voxel centre.
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
use crate::volume::Volume;

use anyhow::{bail, Result};

/// One structure to carry across: its identity and its mask on the source
/// volume's grid.
pub struct Subject {
    pub name: String,
    pub color: [u8; 3],
    /// One byte per source voxel, 1 inside.
    pub mask: Vec<u8>,
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
}

impl Propagated {
    /// `liver: 1642 cm³ ▶ 1701 cm³ (+3.6 %)`.
    pub fn summary(&self) -> String {
        let change = if self.source_cm3 > 1e-9 {
            100.0 * (self.result_cm3 - self.source_cm3) / self.source_cm3
        } else {
            0.0
        };
        format!(
            "{}: {:.1} cm³ ▶ {:.1} cm³ ({:+.1} %)",
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

    /// Source patient position of a destination voxel.
    #[inline]
    fn at(&self, i: usize, j: usize, k: usize) -> Vec3 {
        let u = [
            (i - self.lo[0]) as f64 / self.step[0] as f64,
            (j - self.lo[1]) as f64 / self.step[1] as f64,
            (k - self.lo[2]) as f64 / self.step[2] as f64,
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

/// Inclusive voxel bounding box of a mask.
fn mask_bbox(mask: &[u8], dims: [usize; 3]) -> Option<([usize; 3], [usize; 3])> {
    let [nx, ny, _] = dims;
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    let mut any = false;
    for (o, &v) in mask.iter().enumerate() {
        if v == 0 {
            continue;
        }
        any = true;
        let k = o / (nx * ny);
        let rem = o - k * nx * ny;
        for (a, c) in [rem % nx, rem / nx, k].into_iter().enumerate() {
            lo[a] = lo[a].min(c);
            hi[a] = hi[a].max(c);
        }
    }
    any.then_some((lo, hi))
}

/// Carry `subjects` from `src` onto `dst` through `t`.
///
/// `use_inverse` says which way the transform runs relative to the two
/// volumes: the transform always maps *fixed* patient coordinates to
/// *moving* ones, so propagating onto the moving dataset needs its inverse
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
    let src_vox_cm3 = src.spacing[0] * src.spacing[1] * src.spacing[2] / 1000.0;
    let dst_vox_cm3 = dst.spacing[0] * dst.spacing[1] * dst.spacing[2] / 1000.0;
    // Destination → source, and its opposite (used only to find the box).
    let to_src = |p: Vec3| if use_inverse { t.unmap(p) } else { t.map(p) };
    let to_dst = |p: Vec3| if use_inverse { t.map(p) } else { t.unmap(p) };

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
        let Some((slo, shi)) = mask_bbox(&s.mask, src.dims) else {
            continue;
        };
        let source_voxels = s.mask.iter().filter(|v| **v != 0).count();

        // Where does this structure land in the destination? Map the eight
        // corners of its box across, then keep a generous margin: for a
        // deformable transform the image of a box is not a box, and a
        // clipped structure is a silent error nobody would notice.
        let (mut dlo, mut dhi) = ([f64::MAX; 3], [f64::MIN; 3]);
        let mut max_shift = 0.0f64;
        for ci in 0..8 {
            let c = src.voxel_to_patient(
                if ci & 1 == 0 { slo[0] } else { shi[0] } as f64,
                if ci & 2 == 0 { slo[1] } else { shi[1] } as f64,
                if ci & 4 == 0 { slo[2] } else { shi[2] } as f64,
            );
            let q = to_dst(c);
            max_shift = max_shift.max((q - c).length());
            let v = dst.patient_to_voxel(q);
            for a in 0..3 {
                dlo[a] = dlo[a].min(v[a]);
                dhi[a] = dhi[a].max(v[a]);
            }
        }
        let mut lo = [0usize; 3];
        let mut hi = [0usize; 3];
        for a in 0..3 {
            let margin = 20.0 / dst.spacing[a] + max_shift / dst.spacing[a] + 4.0;
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
            });
            continue;
        }

        let cache = MapCache::build(dst, lo, hi, &to_src, 3.0);
        let [dnx, dny, dnz] = dst.dims;
        let mut mask = vec![0u8; dnx * dny * dnz];
        let src_dims = src.dims;
        let src_mask = &s.mask;
        let rows: Vec<(usize, Vec<u8>)> = (lo[2]..=hi[2])
            .into_par_iter()
            .map(|k| {
                let mut plane = vec![0u8; dnx * dny];
                for j in lo[1]..=hi[1] {
                    for i in lo[0]..=hi[0] {
                        let p = cache.at(i, j, k);
                        let v = src.patient_to_voxel(p);
                        if sample_mask(src_mask, src_dims, v) >= 0.5 {
                            plane[j * dnx + i] = 1;
                        }
                    }
                }
                (k, plane)
            })
            .collect();
        let mut voxels = 0usize;
        for (k, plane) in rows {
            let base = k * dnx * dny;
            for (o, v) in plane.into_iter().enumerate() {
                if v != 0 {
                    mask[base + o] = 1;
                    voxels += 1;
                }
            }
        }
        out.push(Propagated {
            name: s.name.clone(),
            color: s.color,
            mask,
            voxels,
            source_cm3: source_voxels as f64 * src_vox_cm3,
            result_cm3: voxels as f64 * dst_vox_cm3,
        });
    }
    sink.report(1.0, "done");
    Ok(out)
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

    #[test]
    fn a_translation_carries_a_structure_by_exactly_that_much() {
        let dims = [40, 40, 40];
        let src = vol(dims, 2.0, Vec3::new(-40.0, -40.0, -40.0));
        let dst = vol(dims, 2.0, Vec3::new(-40.0, -40.0, -40.0));
        let centre = Vec3::new(0.0, 0.0, 0.0);
        let mask = ball(&src, centre, 12.0);
        let shift = Vec3::new(6.0, -4.0, 2.0);
        // fixed → moving is a shift; src is the fixed dataset, dst the moving
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
