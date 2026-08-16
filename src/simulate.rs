//! Synthetic study transformation for registration QA.
//!
//! Applies a user-defined, exactly-known transform (rigid motion about the
//! volume center plus an optional local Gaussian deformation "bump") to a
//! loaded study and produces a new in-memory study: the CT volume is
//! resampled through the inverse transform, structure contours, dose grids
//! and plan isocenters are carried along. The applied parameters are the
//! ground truth against which the built-in registration can be tested.

use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::loader::{LoadedStudy, Progress, SeriesInfo};
use crate::registration::RigidTransform;
use crate::rtdose::DoseGrid;
use crate::rtstruct::{Contour, Roi, StructureSet};
use crate::volume::Volume;

/// User-specified simulation parameters (all in patient coordinates).
#[derive(Clone, Copy)]
pub struct SimParams {
    /// Translation in mm.
    pub translation: [f64; 3],
    /// Euler rotation about the volume center, degrees (Rz·Ry·Rx).
    pub rotation_deg: [f64; 3],
    /// Peak displacement vector of the Gaussian bump, mm (0 ⇒ no bump).
    pub bump_amp: [f64; 3],
    /// Bump center in patient coordinates (typically the crosshair).
    pub bump_center: [f64; 3],
    /// Bump standard deviation, mm.
    pub bump_sigma: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        SimParams {
            translation: [10.0, -8.0, 5.0],
            rotation_deg: [0.0, 0.0, 2.0],
            bump_amp: [0.0, 0.0, 0.0],
            bump_center: [0.0, 0.0, 0.0],
            bump_sigma: 25.0,
        }
    }
}

impl SimParams {
    /// ASCII-only description (this string is also written into exported
    /// DICOM StudyDescription, so it must stay default-repertoire safe).
    pub fn describe(&self) -> String {
        let mut s = format!(
            "t=({:.1},{:.1},{:.1})mm r=({:.1},{:.1},{:.1})deg",
            self.translation[0],
            self.translation[1],
            self.translation[2],
            self.rotation_deg[0],
            self.rotation_deg[1],
            self.rotation_deg[2]
        );
        if self.has_bump() {
            s += &format!(
                " bump=({:.1},{:.1},{:.1})mm sigma={:.0} at ({:.0},{:.0},{:.0})",
                self.bump_amp[0],
                self.bump_amp[1],
                self.bump_amp[2],
                self.bump_sigma,
                self.bump_center[0],
                self.bump_center[1],
                self.bump_center[2]
            );
        }
        s
    }

    pub fn has_bump(&self) -> bool {
        self.bump_amp.iter().any(|a| a.abs() > 1e-9)
            && self.bump_sigma > 1e-6
    }
}

/// The exact forward transform: original point → transformed ("moved
/// patient") point. `T(p) = R(p − c) + c + t + A·exp(−|p − p₀|²/2σ²)`.
pub struct SimTransform {
    rigid: RigidTransform,
    amp: Vec3,
    center: Vec3,
    sigma2: f64,
    has_bump: bool,
}

impl SimTransform {
    pub fn new(params: &SimParams, volume_center: Vec3) -> Self {
        SimTransform {
            rigid: RigidTransform {
                params: [
                    params.rotation_deg[0].to_radians(),
                    params.rotation_deg[1].to_radians(),
                    params.rotation_deg[2].to_radians(),
                    params.translation[0],
                    params.translation[1],
                    params.translation[2],
                ],
                center: volume_center,
            },
            amp: Vec3::from_slice(&params.bump_amp),
            center: Vec3::from_slice(&params.bump_center),
            sigma2: params.bump_sigma * params.bump_sigma,
            has_bump: params.has_bump(),
        }
    }

    #[inline]
    fn bump(&self, p: Vec3) -> Vec3 {
        if !self.has_bump {
            return Vec3::ZERO;
        }
        let d = p - self.center;
        self.amp * (-d.dot(d) / (2.0 * self.sigma2)).exp()
    }

    #[inline]
    pub fn map(&self, p: Vec3) -> Vec3 {
        self.rigid.map(p) + self.bump(p)
    }

    /// Inverse via fixed-point iteration on the bump term (exact for pure
    /// rigid). Converges for |∇bump| < 1, i.e. amplitude ≲ 1.5 σ.
    pub fn unmap(&self, q: Vec3) -> Vec3 {
        if !self.has_bump {
            return self.rigid.unmap(q);
        }
        let mut x = self.rigid.unmap(q);
        for _ in 0..15 {
            let x_new = self.rigid.unmap(q - self.bump(x));
            if (x_new - x).length() < 1e-4 {
                return x_new;
            }
            x = x_new;
        }
        x
    }
}

/// Generate the transformed study. Everything is carried along:
/// volume (resampled), structures (points mapped), dose grids (resampled on
/// their own geometry) and plan isocenters (mapped).
pub fn generate_transformed_study(
    src: &LoadedStudy,
    params: &SimParams,
    progress: &Progress,
) -> LoadedStudy {
    let vol = &src.volume;
    let center = vol.voxel_to_patient(
        (vol.dims[0] as f64 - 1.0) * 0.5,
        (vol.dims[1] as f64 - 1.0) * 0.5,
        (vol.dims[2] as f64 - 1.0) * 0.5,
    );
    let t = SimTransform::new(params, center);

    // ---- Volume: V_new(x) = V_old(T⁻¹(x)) ------------------------------
    progress.set("Resampling volume…");
    let [nx, ny, nz] = vol.dims;
    let fill = vol.min_value as f32;
    let mut data = vec![0i16; nx * ny * nz];
    data.par_chunks_mut(nx * ny).enumerate().for_each(|(k, plane)| {
        for j in 0..ny {
            for i in 0..nx {
                let x = vol.voxel_to_patient(i as f64, j as f64, k as f64);
                let v = vol
                    .sample_patient(t.unmap(x))
                    .unwrap_or(fill)
                    .round()
                    .clamp(i16::MIN as f32, i16::MAX as f32);
                plane[j * nx + i] = v as i16;
            }
        }
    });
    let mut min_v = i16::MAX;
    let mut max_v = i16::MIN;
    for &v in &data {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    let volume = Volume {
        data,
        dims: vol.dims,
        spacing: vol.spacing,
        origin: vol.origin,
        row_dir: vol.row_dir,
        col_dir: vol.col_dir,
        normal: vol.normal,
        frame_of_reference_uid: vol.frame_of_reference_uid.clone(),
        min_value: min_v,
        max_value: max_v,
    };

    // ---- Structures: contour points mapped forward ---------------------
    progress.set("Transforming structures…");
    let structures = src.structures.as_ref().map(|ss| StructureSet {
        label: format!("{} (sim)", ss.label),
        frame_of_reference_uid: ss.frame_of_reference_uid.clone(),
        rois: ss
            .rois
            .iter()
            .map(|roi| Roi {
                number: roi.number,
                name: roi.name.clone(),
                color: roi.color,
                roi_type: roi.roi_type.clone(),
                contours: roi
                    .contours
                    .iter()
                    .map(|c| Contour {
                        points: c.points.iter().map(|&p| t.map(p)).collect(),
                        geometric_type: c.geometric_type.clone(),
                    })
                    .collect(),
            })
            .collect(),
    });

    // ---- Dose grids: resampled on their own geometry -------------------
    let mut doses = Vec::with_capacity(src.doses.len());
    for (di, d) in src.doses.iter().enumerate() {
        progress.set(format!("Resampling dose {}/{}…", di + 1, src.doses.len()));
        let [dnx, dny, dnf] = d.dims;
        let mut ddata = vec![0.0f32; dnx * dny * dnf];
        ddata
            .par_chunks_mut(dnx * dny)
            .enumerate()
            .for_each(|(f, plane)| {
                let base = d.origin + d.normal * d.offsets[f];
                for j in 0..dny {
                    for i in 0..dnx {
                        let x = base
                            + d.row_dir * (i as f64 * d.spacing[0])
                            + d.col_dir * (j as f64 * d.spacing[1]);
                        plane[j * dnx + i] = d.sample(t.unmap(x)).unwrap_or(0.0);
                    }
                }
            });
        let max_dose = ddata.iter().copied().fold(0.0f32, f32::max);
        doses.push(DoseGrid {
            data: ddata,
            dims: d.dims,
            spacing: d.spacing,
            origin: d.origin,
            row_dir: d.row_dir,
            col_dir: d.col_dir,
            normal: d.normal,
            offsets: d.offsets.clone(),
            units: d.units.clone(),
            summation_type: d.summation_type.clone(),
            max_dose,
            frame_of_reference_uid: d.frame_of_reference_uid.clone(),
            label: format!("{} (sim)", d.label),
        });
    }

    // ---- Plans: isocenters mapped ---------------------------------------
    let plans = src
        .plans
        .iter()
        .map(|p| {
            let mut p = p.clone();
            p.label = format!("{} (sim)", p.label);
            for b in &mut p.beams {
                if let Some(iso) = b.isocenter {
                    b.isocenter = Some(t.map(iso));
                }
            }
            p
        })
        .collect();

    let mut meta = src.meta.clone();
    meta.study_description = format!(
        "{} [simulated: {}]",
        if meta.study_description.is_empty() { "Study" } else { &meta.study_description },
        params.describe()
    );

    progress.set("done");
    LoadedStudy {
        meta,
        series: vec![SeriesInfo {
            uid: format!("sim.{}", src.series.first().map(|s| s.uid.as_str()).unwrap_or("0")),
            modality: src
                .series
                .first()
                .map(|s| s.modality.clone())
                .unwrap_or_else(|| "CT".into()),
            description: format!("Simulated [{}]", params.describe()),
            files: Vec::new(),
        }],
        active_series: 0,
        volume,
        structures,
        doses,
        plans,
        // Planar images / registrations / records are carried over unchanged
        // (2D projections and metadata objects are not resampled).
        planar_images: src.planar_images.clone(),
        registrations: src.registrations.clone(),
        treat_records: src.treat_records.clone(),
        warnings: vec![format!("Simulated dataset — ground truth: {}", params.describe())],
        default_window: src.default_window,
    }
}
