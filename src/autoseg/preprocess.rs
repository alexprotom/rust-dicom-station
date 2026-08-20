//! Volume → model-space preprocessing and label back-mapping.
//!
//! TotalSegmentator's pipeline reorients the image to the closest canonical
//! RAS orientation (nibabel `as_closest_canonical`), resamples it to the
//! model's isotropic spacing (spline order 1 = trilinear, int32 dtype), and
//! feeds nnU-Net arrays whose spatial axes are, in order, [S, A, R]
//! (superior, anterior, right — each increasing with the array index).
//! This module reproduces that from the DICOM volume directly: it finds the
//! permutation + flips of the volume's own axes that best align with
//! [S, A, R] (LPS patient space: S = +z, A = −y, R = −x), then resamples
//! along the volume's (possibly slightly oblique) axes onto the model grid
//! using the endpoint-aligned coordinate convention of `scipy.ndimage.zoom`
//! (first and last voxel centers coincide) — the resampler TotalSegmentator
//! uses.
//!
//! The inverse mapping (`labels_to_volume_grid`) assigns every voxel of the
//! original volume the nearest model-grid label (order-0, the same way
//! TotalSegmentator resamples its label map back).

use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::volume::Volume;

/// Mapping between the volume's index space and the model's [S,A,R] grid.
#[derive(Clone, Debug)]
pub struct SarMap {
    /// SAR axis → volume axis (0 = i/x, 1 = j/y, 2 = k/z).
    pub perm: [usize; 3],
    /// SAR axis runs opposite to the volume axis.
    pub flip: [bool; 3],
    /// Volume dims along the mapped axes ([S,A,R] order).
    pub orig_dims: [usize; 3],
    /// Volume spacing along the mapped axes (mm).
    pub orig_spacing: [f64; 3],
    /// Model grid dims ([S,A,R] order).
    pub model_dims: [usize; 3],
    /// Isotropic model spacing (mm).
    pub target: f64,
}

/// numpy-compatible round-half-to-even.
fn np_round(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 {
        // exactly .5 → nearest even
        let f = v.floor();
        if (f as i64) % 2 == 0 {
            f
        } else {
            f + 1.0
        }
    } else {
        r
    }
}

impl SarMap {
    pub fn new(vol: &Volume, target_spacing: f64) -> SarMap {
        // LPS direction vectors of the three volume axes.
        let dirs: [Vec3; 3] = [vol.row_dir, vol.col_dir, vol.normal];
        // Canonical targets in LPS: S = +z, A = −y, R = −x.
        let targets: [Vec3; 3] = [
            Vec3 { x: 0.0, y: 0.0, z: 1.0 },
            Vec3 { x: 0.0, y: -1.0, z: 0.0 },
            Vec3 { x: -1.0, y: 0.0, z: 0.0 },
        ];
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
        let dims = [vol.dims[0], vol.dims[1], vol.dims[2]];
        let spac = [vol.spacing[0], vol.spacing[1], vol.spacing[2]];
        let orig_dims = [dims[perm[0]], dims[perm[1]], dims[perm[2]]];
        let orig_spacing = [spac[perm[0]], spac[perm[1]], spac[perm[2]]];
        let model_dims = [
            (np_round(orig_dims[0] as f64 * orig_spacing[0] / target_spacing) as usize).max(1),
            (np_round(orig_dims[1] as f64 * orig_spacing[1] / target_spacing) as usize).max(1),
            (np_round(orig_dims[2] as f64 * orig_spacing[2] / target_spacing) as usize).max(1),
        ];
        SarMap {
            perm,
            flip,
            orig_dims,
            orig_spacing,
            model_dims,
            target: target_spacing,
        }
    }
}

/// Resample the volume onto the model grid (trilinear, edge-clamped,
/// truncated toward zero like TotalSegmentator's int32 conversion).
/// Output layout: `[s][a][r]` (C-contiguous, dims = `map.model_dims`).
pub fn resample_to_model(vol: &Volume, map: &SarMap) -> Vec<f32> {
    let [d0, d1, d2] = map.model_dims;
    let nx = vol.dims[0];
    let ny = vol.dims[1];
    let nz = vol.dims[2];
    let data = &vol.data;
    // model index → continuous voxel coordinate along the mapped axis.
    // TotalSegmentator resamples with scipy.ndimage.zoom, whose (default)
    // coordinate convention is endpoint-aligned:
    //   in = out * (n_in - 1) / (n_out - 1)
    // (first and last voxel centers coincide). Flips commute with this
    // mapping, so applying them on the input side is exact.
    let scale = std::array::from_fn::<f64, 3, _>(|a| {
        if map.model_dims[a] > 1 {
            (map.orig_dims[a] - 1) as f64 / (map.model_dims[a] - 1) as f64
        } else {
            0.0
        }
    });
    let coord = |a: usize, m: usize| -> f64 {
        let c = m as f64 * scale[a];
        if map.flip[a] {
            (map.orig_dims[a] - 1) as f64 - c
        } else {
            c
        }
    };
    let mut out = vec![0f32; d0 * d1 * d2];
    out.par_chunks_mut(d1 * d2).enumerate().for_each(|(m0, slab)| {
        let c_a0 = coord(0, m0);
        let mut coords = [0f64; 3]; // per volume axis (x, y, z)
        coords[map.perm[0]] = c_a0;
        for m1 in 0..d1 {
            coords[map.perm[1]] = coord(1, m1);
            for m2 in 0..d2 {
                coords[map.perm[2]] = coord(2, m2);
                // trilinear with edge clamp
                let cx = coords[0].clamp(0.0, (nx - 1) as f64);
                let cy = coords[1].clamp(0.0, (ny - 1) as f64);
                let cz = coords[2].clamp(0.0, (nz - 1) as f64);
                let (x0, y0, z0) = (cx.floor() as usize, cy.floor() as usize, cz.floor() as usize);
                let (x1, y1, z1) = (
                    (x0 + 1).min(nx - 1),
                    (y0 + 1).min(ny - 1),
                    (z0 + 1).min(nz - 1),
                );
                let (fx, fy, fz) = (
                    (cx - x0 as f64) as f32,
                    (cy - y0 as f64) as f32,
                    (cz - z0 as f64) as f32,
                );
                let at = |x: usize, y: usize, z: usize| -> f32 {
                    data[z * nx * ny + y * nx + x] as f32
                };
                let c00 = at(x0, y0, z0) * (1.0 - fx) + at(x1, y0, z0) * fx;
                let c10 = at(x0, y1, z0) * (1.0 - fx) + at(x1, y1, z0) * fx;
                let c01 = at(x0, y0, z1) * (1.0 - fx) + at(x1, y0, z1) * fx;
                let c11 = at(x0, y1, z1) * (1.0 - fx) + at(x1, y1, z1) * fx;
                let c0 = c00 * (1.0 - fy) + c10 * fy;
                let c1 = c01 * (1.0 - fy) + c11 * fy;
                let v = c0 * (1.0 - fz) + c1 * fz;
                // int32 conversion (truncation toward zero), as in
                // TotalSegmentator's change_spacing(dtype=np.int32)
                slab[m1 * d2 + m2] = v.trunc();
            }
        }
    });
    out
}

/// Map model-grid labels back onto the original volume grid (nearest
/// neighbor). Output is in `Volume::data` index order (`k*nx*ny + j*nx + i`).
pub fn labels_to_volume_grid(labels_model: &[u8], map: &SarMap, vol: &Volume) -> Vec<u8> {
    let [_d0, d1, d2] = map.model_dims;
    let nx = vol.dims[0];
    let ny = vol.dims[1];
    let nz = vol.dims[2];
    // Endpoint-aligned inverse mapping (scipy.ndimage.zoom convention, the
    // way TotalSegmentator resamples its label map back, order 0).
    let inv_scale = std::array::from_fn::<f64, 3, _>(|a| {
        if map.orig_dims[a] > 1 {
            (map.model_dims[a] - 1) as f64 / (map.orig_dims[a] - 1) as f64
        } else {
            0.0
        }
    });
    // For each volume axis v: which SAR axis does it feed?
    let mut sar_of_axis = [0usize; 3];
    for a in 0..3 {
        sar_of_axis[map.perm[a]] = a;
    }
    let model_idx = |a: usize, orig_idx: usize| -> usize {
        let f = if map.flip[a] {
            (map.orig_dims[a] - 1 - orig_idx) as f64
        } else {
            orig_idx as f64
        };
        // order-0 resampling: round half up (scipy nearest)
        let m = (f * inv_scale[a] + 0.5).floor() as isize;
        m.clamp(0, (map.model_dims[a] - 1) as isize) as usize
    };
    let mut out = vec![0u8; nx * ny * nz];
    out.par_chunks_mut(nx * ny).enumerate().for_each(|(k, slab)| {
        let a_of_z = sar_of_axis[2];
        let m_z = model_idx(a_of_z, k);
        for j in 0..ny {
            let a_of_y = sar_of_axis[1];
            let m_y = model_idx(a_of_y, j);
            for (i, dst) in slab[j * nx..(j + 1) * nx].iter_mut().enumerate() {
                let a_of_x = sar_of_axis[0];
                let m_x = model_idx(a_of_x, i);
                let mut m = [0usize; 3];
                m[a_of_z] = m_z;
                m[a_of_y] = m_y;
                m[a_of_x] = m_x;
                *dst = labels_model[(m[0] * d1 + m[1]) * d2 + m[2]];
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    fn axial_volume(dims: [usize; 3], spacing: [f64; 3]) -> Volume {
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing,
            origin: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
            row_dir: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
            col_dir: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
            normal: Vec3 { x: 0.0, y: 0.0, z: 1.0 },
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        }
    }

    #[test]
    fn standard_axial_mapping() {
        // Standard axial LPS volume: i→+x(L), j→+y(P), k→+z(S).
        let vol = axial_volume([512, 512, 133], [0.9766, 0.9766, 3.0]);
        let map = SarMap::new(&vol, 3.0);
        // S axis = volume z (no flip), A axis = volume y (flip: +y is P),
        // R axis = volume x (flip: +x is L).
        assert_eq!(map.perm, [2, 1, 0]);
        assert_eq!(map.flip, [false, true, true]);
        assert_eq!(map.orig_dims, [133, 512, 512]);
        assert_eq!(map.model_dims, [133, 167, 167]);
    }

    #[test]
    fn resample_round_trip_labels() {
        // A small volume with a bright block; check the block's label round-trips.
        let mut vol = axial_volume([20, 20, 10], [2.0, 2.0, 2.0]);
        for k in 3..7 {
            for j in 4..12 {
                for i in 6..14 {
                    vol.data[k * 400 + j * 20 + i] = 100;
                }
            }
        }
        let map = SarMap::new(&vol, 2.0); // same spacing → pure reorientation
        assert_eq!(map.model_dims, [10, 20, 20]);
        let res = resample_to_model(&vol, &map);
        // voxel count preserved under pure flips/permutation
        let bright = res.iter().filter(|v| **v > 50.0).count();
        assert_eq!(bright, 4 * 8 * 8);
        // label the bright voxels in model space, map back, compare exactly
        let labels_model: Vec<u8> = res.iter().map(|v| (*v > 50.0) as u8).collect();
        let back = labels_to_volume_grid(&labels_model, &map, &vol);
        for (idx, v) in vol.data.iter().enumerate() {
            assert_eq!(back[idx], (*v == 100) as u8, "voxel {idx}");
        }
    }

    #[test]
    fn np_round_half_even() {
        assert_eq!(np_round(2.5), 2.0);
        assert_eq!(np_round(3.5), 4.0);
        assert_eq!(np_round(2.4), 2.0);
        assert_eq!(np_round(166.67), 167.0);
    }
}
