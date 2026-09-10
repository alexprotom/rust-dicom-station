//! RT Dose (RTDOSE) parsing and trilinear dose sampling in patient space.
//!
//! Pixel data is decoded manually (16- or 32-bit, signed or unsigned,
//! little-endian native encoding - which covers effectively all RTDOSE files
//! in the wild) and converted to physical dose via DoseGridScaling.

use std::path::Path;

use anyhow::{bail, Context, Result};
use dicom_core::value::PrimitiveValue;
use dicom_core::DicomValue;
use dicom_dictionary_std::tags;

use crate::geometry::Vec3;
use crate::loader::{f64_of, f64s_of, i32_of, str_of};

#[derive(Clone)]
pub struct DoseGrid {
    /// Dose in `units`, frame-major: `data[f * nx * ny + j * nx + i]`.
    pub data: Vec<f32>,
    /// `[nx, ny, n_frames]`
    pub dims: [usize; 3],
    /// In-plane spacing `[sx, sy]` (mm).
    pub spacing: [f64; 2],
    /// Patient position of the center of pixel (0,0) of the first frame.
    pub origin: Vec3,
    pub row_dir: Vec3,
    pub col_dir: Vec3,
    pub normal: Vec3,
    /// Frame offsets along `normal` (mm), ascending, first element 0.
    pub offsets: Vec<f64>,
    pub units: String,
    pub summation_type: String,
    /// Dose Type (3004,0004): `PHYSICAL`, `EFFECTIVE` (RBE-weighted) or
    /// `ERROR`; empty when the file did not say.
    pub dose_type: String,
    pub max_dose: f32,
    pub frame_of_reference_uid: String,
    /// SOP Instance UID of this dose object.
    pub sop_instance_uid: String,
    /// Series Instance UID of this object's own series. Kept so that an
    /// export asked to preserve the identifiers really preserves them: an
    /// archive files an instance under its series, and the same instance
    /// arriving under a different one is a different object to it.
    pub series_instance_uid: String,
    /// Study this dose belongs to.
    pub study_uid: String,
    /// SOP Instance UID of the RTPLAN this dose was computed for.
    pub referenced_plan_uid: String,
    pub label: String,
}

impl DoseGrid {
    /// The lattice of this grid for meshing: its frame spacing is the mean
    /// frame step (1 mm for a single frame).
    pub fn mesh_geom(&self) -> crate::mesh3d::GridGeom {
        let n = self.offsets.len();
        let dz = if n >= 2 {
            (self.offsets[n - 1] - self.offsets[0]) / (n - 1) as f64
        } else {
            1.0
        };
        crate::mesh3d::GridGeom {
            origin: self.origin,
            row_dir: self.row_dir,
            col_dir: self.col_dir,
            normal: self.normal,
            spacing: [self.spacing[0], self.spacing[1], dz.abs().max(1e-6)],
        }
    }

    /// The voxels at or above `threshold`, snapshotted the way a
    /// segmentation is for surface meshing (see
    /// [`crate::segmentation::Segmentation::mesh_grid`]): a bool grid over
    /// their bounding box with one empty cell of margin, max-pooled by an
    /// integer stride when the box is very large. `None` when nothing
    /// reaches the threshold.
    pub fn iso_mesh_grid(&self, threshold: f32) -> Option<crate::segmentation::MeshGrid> {
        let [nx, ny, nz] = self.dims;
        let mut lo = [usize::MAX; 3];
        let mut hi = [0usize; 3];
        let mut any = false;
        for k in 0..nz {
            for j in 0..ny {
                let row = &self.data[k * nx * ny + j * nx..k * nx * ny + j * nx + nx];
                for (i, &d) in row.iter().enumerate() {
                    if d >= threshold {
                        lo = [lo[0].min(i), lo[1].min(j), lo[2].min(k)];
                        hi = [hi[0].max(i), hi[1].max(j), hi[2].max(k)];
                        any = true;
                    }
                }
            }
        }
        if !any {
            return None;
        }
        let size = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
        const MAX_CELLS: usize = 6_000_000;
        let gdim = |s: usize, st: usize| s.div_ceil(st) + 2;
        let mut stride = 1usize;
        while gdim(size[0], stride) * gdim(size[1], stride) * gdim(size[2], stride) > MAX_CELLS {
            stride += 1;
        }
        let g = [
            gdim(size[0], stride),
            gdim(size[1], stride),
            gdim(size[2], stride),
        ];
        let mut grid = vec![false; g[0] * g[1] * g[2]];
        for gk in 1..g[2] - 1 {
            let k0 = lo[2] + (gk - 1) * stride;
            let k1 = (k0 + stride).min(hi[2] + 1);
            for gj in 1..g[1] - 1 {
                let j0 = lo[1] + (gj - 1) * stride;
                let j1 = (j0 + stride).min(hi[1] + 1);
                for gi in 1..g[0] - 1 {
                    let i0 = lo[0] + (gi - 1) * stride;
                    let i1 = (i0 + stride).min(hi[0] + 1);
                    let hit = (k0..k1).any(|k| {
                        (j0..j1).any(|j| {
                            let base = k * nx * ny + j * nx;
                            self.data[base + i0..base + i1]
                                .iter()
                                .any(|&d| d >= threshold)
                        })
                    });
                    if hit {
                        grid[gk * g[0] * g[1] + gj * g[0] + gi] = true;
                    }
                }
            }
        }
        Some((grid, g, lo, stride))
    }

    /// Trilinear dose sample at a patient-space point. `None` outside grid.
    pub fn sample(&self, p: Vec3) -> Option<f32> {
        self.sample_uvw(self.grid_coords(p))
    }

    /// Grid coordinates of a patient point: fractional column and row indices
    /// plus the distance along the grid normal in mm. Affine in `p`, so a
    /// caller walking a regular lattice can step these incrementally instead
    /// of re-projecting every point (see [`crate::render::sample_dose_plane`]).
    #[inline]
    pub fn grid_coords(&self, p: Vec3) -> [f64; 3] {
        let d = p - self.origin;
        [
            d.dot(self.row_dir) / self.spacing[0],
            d.dot(self.col_dir) / self.spacing[1],
            d.dot(self.normal),
        ]
    }

    /// Trilinear dose sample at grid coordinates from [`DoseGrid::grid_coords`].
    pub fn sample_uvw(&self, [u, v, w]: [f64; 3]) -> Option<f32> {
        let [nx, ny, nf] = self.dims;

        if u < 0.0 || v < 0.0 || u > (nx - 1) as f64 || v > (ny - 1) as f64 {
            return None;
        }
        if w < self.offsets[0] || w > *self.offsets.last().unwrap() {
            return None;
        }

        // Find bracketing frames (offsets ascending, possibly non-uniform).
        // Frame offsets are uniform in practically every RTDOSE, so guess the
        // index directly and only fall back to a search if the guess misses;
        // this runs once per displayed pixel.
        let last = *self.offsets.last().unwrap();
        let hi = 'find: {
            if nf > 1 {
                let step = (last - self.offsets[0]) / (nf - 1) as f64;
                if step > 1e-9 {
                    let g =
                        (((w - self.offsets[0]) / step).ceil() as i64).clamp(0, nf as i64) as usize;
                    let below = g == 0 || self.offsets[g - 1] <= w;
                    let above = g == nf || self.offsets[g] >= w;
                    if below && above {
                        break 'find g;
                    }
                }
            }
            match self
                .offsets
                .binary_search_by(|o| o.partial_cmp(&w).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i,
            }
        };
        let (f0, f1, tz) = if hi == 0 {
            (0, 0, 0.0)
        } else if hi >= nf {
            (nf - 1, nf - 1, 0.0)
        } else {
            let o0 = self.offsets[hi - 1];
            let o1 = self.offsets[hi];
            let t = if (o1 - o0).abs() > 1e-9 {
                (w - o0) / (o1 - o0)
            } else {
                0.0
            };
            (hi - 1, hi, t)
        };

        let i0 = u.floor() as usize;
        let j0 = v.floor() as usize;
        let i1 = (i0 + 1).min(nx - 1);
        let j1 = (j0 + 1).min(ny - 1);
        let tx = (u - i0 as f64) as f32;
        let ty = (v - j0 as f64) as f32;

        let bilerp = |f: usize| -> f32 {
            let base = f * nx * ny;
            let d00 = self.data[base + j0 * nx + i0];
            let d10 = self.data[base + j0 * nx + i1];
            let d01 = self.data[base + j1 * nx + i0];
            let d11 = self.data[base + j1 * nx + i1];
            let a = d00 + (d10 - d00) * tx;
            let b = d01 + (d11 - d01) * tx;
            a + (b - a) * ty
        };

        let dz0 = bilerp(f0);
        let dz1 = bilerp(f1);
        Some(dz0 + (dz1 - dz0) * tz as f32)
    }
}

pub fn load(path: &Path) -> Result<DoseGrid> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open RTDOSE {}", path.display()))?;

    let rows = i32_of(&obj, tags::ROWS).context("RTDOSE missing Rows")? as usize;
    let cols = i32_of(&obj, tags::COLUMNS).context("RTDOSE missing Columns")? as usize;
    let n_frames = i32_of(&obj, tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1) as usize;

    let ipp = f64s_of(&obj, tags::IMAGE_POSITION_PATIENT)
        .filter(|v| v.len() >= 3)
        .context("RTDOSE missing ImagePositionPatient")?;
    let iop = f64s_of(&obj, tags::IMAGE_ORIENTATION_PATIENT)
        .filter(|v| v.len() >= 6)
        .unwrap_or_else(|| vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let ps = f64s_of(&obj, tags::PIXEL_SPACING)
        .filter(|v| v.len() >= 2)
        .unwrap_or_else(|| vec![1.0, 1.0]);

    let row_dir = Vec3::from_slice(&iop[0..3]).normalized();
    let col_dir = Vec3::from_slice(&iop[3..6]).normalized();
    let normal = row_dir.cross(col_dir).normalized();

    let mut offsets = f64s_of(&obj, tags::GRID_FRAME_OFFSET_VECTOR)
        .unwrap_or_else(|| (0..n_frames).map(|i| i as f64).collect());
    if offsets.len() != n_frames {
        bail!(
            "GridFrameOffsetVector length {} != NumberOfFrames {}",
            offsets.len(),
            n_frames
        );
    }

    let scaling = f64_of(&obj, tags::DOSE_GRID_SCALING).unwrap_or(1.0);
    let units = str_of(&obj, tags::DOSE_UNITS).unwrap_or_else(|| "GY".into());
    let summation_type = str_of(&obj, tags::DOSE_SUMMATION_TYPE).unwrap_or_default();
    let dose_type = str_of(&obj, tags::DOSE_TYPE)
        .unwrap_or_default()
        .trim()
        .to_uppercase();

    let bits = i32_of(&obj, tags::BITS_ALLOCATED).unwrap_or(32);
    let signed = i32_of(&obj, tags::PIXEL_REPRESENTATION).unwrap_or(0) == 1;

    // ---- Manual pixel decode -------------------------------------------
    let elem = obj
        .element(tags::PIXEL_DATA)
        .context("RTDOSE has no PixelData")?;

    let n = rows * cols * n_frames;
    let raw: Vec<f64> = match elem.value() {
        DicomValue::Primitive(PrimitiveValue::U16(words)) => match bits {
            16 => {
                if signed {
                    words.iter().map(|&w| w as i16 as f64).collect()
                } else {
                    words.iter().map(|&w| w as f64).collect()
                }
            }
            32 => {
                if words.len() < n * 2 {
                    bail!(
                        "RTDOSE pixel words ({}) < expected ({})",
                        words.len(),
                        n * 2
                    );
                }
                words
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| {
                        let v = (c[0] as u32) | ((c[1] as u32) << 16);
                        if signed {
                            v as i32 as f64
                        } else {
                            v as f64
                        }
                    })
                    .collect()
            }
            other => bail!("Unsupported RTDOSE BitsAllocated {other} (OW)"),
        },
        DicomValue::Primitive(PrimitiveValue::U8(bytes)) => match bits {
            16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&c| {
                    let v = u16::from_le_bytes(c);
                    if signed {
                        v as i16 as f64
                    } else {
                        v as f64
                    }
                })
                .collect(),
            32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&c| {
                    let v = u32::from_le_bytes(c);
                    if signed {
                        v as i32 as f64
                    } else {
                        v as f64
                    }
                })
                .collect(),
            other => bail!("Unsupported RTDOSE BitsAllocated {other} (OB)"),
        },
        _ => bail!("RTDOSE PixelData is not in native (uncompressed) form"),
    };

    if raw.len() < n {
        bail!("RTDOSE pixel count {} < expected {}", raw.len(), n);
    }

    let mut data: Vec<f32> = raw[..n].iter().map(|&v| (v * scaling) as f32).collect();

    // Normalize frame order to ascending offsets.
    let ascending = offsets.windows(2).all(|w| w[1] >= w[0]);
    if !ascending {
        let descending = offsets.windows(2).all(|w| w[1] <= w[0]);
        if descending {
            offsets.reverse();
            let frame = rows * cols;
            let mut flipped = Vec::with_capacity(data.len());
            for f in (0..n_frames).rev() {
                flipped.extend_from_slice(&data[f * frame..(f + 1) * frame]);
            }
            data = flipped;
        } else {
            bail!("GridFrameOffsetVector is not monotonic");
        }
    }
    // Re-base offsets so the first frame sits at the IPP position.
    let base = offsets[0];
    let origin = Vec3::from_slice(&ipp) + normal * base;
    for o in offsets.iter_mut() {
        *o -= base;
    }

    let max_dose = data.iter().copied().fold(0.0_f32, f32::max);

    let label = format!(
        "{} [{}] max {:.2} {}",
        path.file_stem().unwrap_or_default().to_string_lossy(),
        if summation_type.is_empty() {
            "DOSE"
        } else {
            &summation_type
        },
        max_dose,
        units.to_lowercase()
    );

    Ok(DoseGrid {
        data,
        dims: [cols, rows, n_frames],
        spacing: [ps[1], ps[0]],
        origin,
        row_dir,
        col_dir,
        normal,
        offsets,
        units,
        summation_type,
        dose_type,
        max_dose,
        frame_of_reference_uid: str_of(&obj, tags::FRAME_OF_REFERENCE_UID).unwrap_or_default(),
        sop_instance_uid: str_of(&obj, tags::SOP_INSTANCE_UID).unwrap_or_default(),
        series_instance_uid: str_of(&obj, tags::SERIES_INSTANCE_UID).unwrap_or_default(),
        study_uid: str_of(&obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        referenced_plan_uid: crate::loader::items_of(&obj, tags::REFERENCED_RT_PLAN_SEQUENCE)
            .and_then(|items| items.first())
            .and_then(|it| str_of(it, tags::REFERENCED_SOP_INSTANCE_UID))
            .unwrap_or_default(),
        label,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_isodose_grid_boxes_the_voxels_above_the_threshold() {
        let (nx, ny, nz) = (12usize, 10usize, 8usize);
        let mut data = vec![0f32; nx * ny * nz];
        for k in 2..5 {
            for j in 3..7 {
                for i in 4..9 {
                    data[k * nx * ny + j * nx + i] = 10.0;
                }
            }
        }
        let dose = DoseGrid {
            data,
            dims: [nx, ny, nz],
            spacing: [2.0, 2.0],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            offsets: (0..nz).map(|f| 3.0 * f as f64).collect(),
            units: "GY".into(),
            summation_type: "PLAN".into(),
            dose_type: String::new(),
            max_dose: 10.0,
            frame_of_reference_uid: String::new(),
            sop_instance_uid: String::new(),
            series_instance_uid: String::new(),
            study_uid: String::new(),
            referenced_plan_uid: String::new(),
            label: String::new(),
        };
        assert_eq!(dose.mesh_geom().spacing, [2.0, 2.0, 3.0]);
        let (grid, g, lo, stride) = dose.iso_mesh_grid(5.0).unwrap();
        assert_eq!((lo, stride), ([4, 3, 2], 1));
        assert_eq!(g, [7, 6, 5], "the box plus one empty cell each side");
        assert_eq!(grid.iter().filter(|&&v| v).count(), 5 * 4 * 3);
        assert!(!grid[0] && !grid[grid.len() - 1], "the margin stays empty");
        assert!(dose.iso_mesh_grid(11.0).is_none());
        assert!(crate::mesh3d::mesh_from_mask(&grid, g, lo, stride, &dose.mesh_geom()).is_some());
    }
}
