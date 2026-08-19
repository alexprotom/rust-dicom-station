//! Additional DICOM object types beyond volumes and the core RT set:
//!
//! * planar projection images — DX / CR digital radiographs and RTIMAGE
//!   (DRRs, portal / setup images), shown in floating viewer windows;
//! * REG — Spatial Registration objects (rigid 4×4 frame-of-reference
//!   transformation matrices; deformable REG objects are recognized and
//!   their matrices read, but deformation grids are not applied);
//! * RTRECORD — RT (Ion) Beams Treatment Records with per-beam specified vs
//!   delivered metersets and termination status.

use std::path::Path;

use anyhow::{bail, Context, Result};
use dicom_dictionary_std::tags;
use dicom_pixeldata::{ConvertOptions, ModalityLutOption, PixelDecoder};

use crate::geometry::Vec3;
use crate::loader::{f64_of, f64s_of, i32_of, items_of, str_of};
use crate::registration::RigidTransform;

// ---------------------------------------------------------------------------
// Planar images (DX / CR / RTIMAGE)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct PlanarImage {
    pub label: String,
    pub modality: String,
    pub rows: usize,
    pub cols: usize,
    /// Pixel spacing [x, y] in mm (imager / image plane), 1.0 if unknown.
    pub spacing: [f64; 2],
    /// Decoded values after Modality LUT (MONOCHROME1 already inverted).
    pub data: Vec<f32>,
    pub min_value: f32,
    pub max_value: f32,
    /// Default window (center, width).
    pub window: (f32, f32),
    /// Free-form metadata (name, value) for display.
    pub info: Vec<(String, String)>,
}

pub fn load_planar(path: &Path) -> Result<PlanarImage> {
    let obj = dicom_object::open_file(path)
        .with_context(|| format!("open planar image {}", path.display()))?;
    let modality = str_of(&obj, tags::MODALITY).unwrap_or_else(|| "DX".into());

    let decoded = obj
        .decode_pixel_data()
        .with_context(|| format!("decode pixel data of {}", path.display()))?;
    let rows = decoded.rows() as usize;
    let cols = decoded.columns() as usize;
    if decoded.number_of_frames() > 1 {
        bail!("multi-frame planar image not supported: {}", path.display());
    }
    let opts = ConvertOptions::new().with_modality_lut(ModalityLutOption::Default);
    let mut data: Vec<f32> = decoded
        .to_vec_with_options(&opts)
        .with_context(|| format!("convert pixels of {}", path.display()))?;
    if data.len() < rows * cols {
        bail!("pixel buffer smaller than Rows×Columns in {}", path.display());
    }
    data.truncate(rows * cols);

    let mono1 = str_of(&obj, tags::PHOTOMETRIC_INTERPRETATION)
        .map(|p| p == "MONOCHROME1")
        .unwrap_or(false);
    let (mut min_v, mut max_v) = (f32::MAX, f32::MIN);
    for &v in &data {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if mono1 {
        // Invert so that higher = brighter, like MONOCHROME2.
        for v in &mut data {
            *v = max_v + min_v - *v;
        }
    }

    let spacing = f64s_of(&obj, tags::IMAGE_PLANE_PIXEL_SPACING)
        .or_else(|| f64s_of(&obj, tags::IMAGER_PIXEL_SPACING))
        .or_else(|| f64s_of(&obj, tags::PIXEL_SPACING))
        .filter(|v| v.len() >= 2)
        .map(|v| [v[1], v[0]])
        .unwrap_or([1.0, 1.0]);

    let window = match (
        f64s_of(&obj, tags::WINDOW_CENTER).and_then(|v| v.first().copied()),
        f64s_of(&obj, tags::WINDOW_WIDTH).and_then(|v| v.first().copied()),
    ) {
        (Some(c), Some(w)) if w > 0.0 && !mono1 => (c as f32, w as f32),
        _ => (
            (min_v + max_v) * 0.5,
            (max_v - min_v).max(1.0),
        ),
    };

    let label = str_of(&obj, tags::RT_IMAGE_LABEL)
        .or_else(|| str_of(&obj, tags::SERIES_DESCRIPTION))
        .or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| modality.clone());

    // Display metadata.
    let mut info: Vec<(String, String)> = Vec::new();
    let mut add = |k: &str, v: Option<String>| {
        if let Some(v) = v {
            if !v.is_empty() {
                info.push((k.to_string(), v));
            }
        }
    };
    add("Date", str_of(&obj, tags::CONTENT_DATE).or_else(|| str_of(&obj, tags::STUDY_DATE)));
    if modality == "RTIMAGE" {
        add("Machine", str_of(&obj, tags::RADIATION_MACHINE_NAME));
        add(
            "Gantry",
            f64_of(&obj, tags::GANTRY_ANGLE).map(|g| format!("{g:.1}°")),
        );
        add(
            "SAD",
            f64_of(&obj, tags::RADIATION_MACHINE_SAD).map(|v| format!("{v:.0} mm")),
        );
        add(
            "SID",
            f64_of(&obj, tags::RT_IMAGE_SID).map(|v| format!("{v:.0} mm")),
        );
        add("Description", str_of(&obj, tags::RT_IMAGE_DESCRIPTION));
    } else {
        add("Body part", str_of(&obj, tags::BODY_PART_EXAMINED));
        add("View", str_of(&obj, tags::VIEW_POSITION));
        add("KVP", str_of(&obj, tags::KVP));
    }
    info.push((
        "Size".into(),
        format!("{cols}×{rows} px · {:.2}/{:.2} mm", spacing[0], spacing[1]),
    ));

    Ok(PlanarImage {
        label,
        modality,
        rows,
        cols,
        spacing,
        data,
        min_value: min_v,
        max_value: max_v,
        window,
        info,
    })
}

// ---------------------------------------------------------------------------
// REG — Spatial Registration
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RegMatrixItem {
    /// Frame of Reference the matrix applies to (the "source" frame).
    pub for_uid: String,
    /// Row-major 4×4 frame-of-reference transformation matrix.
    pub matrix: [f64; 16],
    /// RIGID / RIGID_SCALE / AFFINE.
    pub matrix_type: String,
    pub is_identity: bool,
}

#[derive(Clone)]
pub struct SpatialReg {
    pub label: String,
    /// True for Deformable Spatial Registration Storage (grids not applied).
    pub deformable: bool,
    /// Frame of Reference of the registration instance itself (the frame
    /// the matrices transform *into*).
    pub frame_of_reference_uid: String,
    pub items: Vec<RegMatrixItem>,
}

const SOP_SPATIAL_REG: &str = "1.2.840.10008.5.1.4.1.1.66.1";
const SOP_DEFORMABLE_REG: &str = "1.2.840.10008.5.1.4.1.1.66.3";

pub fn is_reg_sop(sop: &str) -> bool {
    sop == SOP_SPATIAL_REG || sop == SOP_DEFORMABLE_REG
}

pub fn load_reg(path: &Path) -> Result<SpatialReg> {
    let obj =
        dicom_object::open_file(path).with_context(|| format!("open REG {}", path.display()))?;
    let sop = str_of(&obj, tags::SOP_CLASS_UID).unwrap_or_default();
    let deformable = sop == SOP_DEFORMABLE_REG;
    let label = str_of(&obj, tags::CONTENT_DESCRIPTION)
        .or_else(|| str_of(&obj, tags::SERIES_DESCRIPTION))
        .or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "Registration".into());
    let frame_of_reference_uid = str_of(&obj, tags::FRAME_OF_REFERENCE_UID).unwrap_or_default();

    let mut items = Vec::new();
    if let Some(regs) = items_of(&obj, tags::REGISTRATION_SEQUENCE) {
        for r in regs {
            let for_uid = str_of(r, tags::FRAME_OF_REFERENCE_UID).unwrap_or_default();
            let Some(mreg) = items_of(r, tags::MATRIX_REGISTRATION_SEQUENCE) else { continue };
            for mr in mreg {
                let Some(ms) = items_of(mr, tags::MATRIX_SEQUENCE) else { continue };
                for m in ms {
                    let Some(vals) =
                        f64s_of(m, tags::FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX)
                            .filter(|v| v.len() >= 16)
                    else {
                        continue;
                    };
                    let mut matrix = [0.0; 16];
                    matrix.copy_from_slice(&vals[..16]);
                    let matrix_type =
                        str_of(m, tags::FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX_TYPE)
                            .unwrap_or_else(|| "RIGID".into());
                    let is_identity = matrix
                        .iter()
                        .enumerate()
                        .all(|(i, &v)| (v - if i % 5 == 0 { 1.0 } else { 0.0 }).abs() < 1e-6);
                    items.push(RegMatrixItem { for_uid: for_uid.clone(), matrix, matrix_type, is_identity });
                }
            }
        }
    }
    if items.is_empty() {
        bail!("REG object contains no transformation matrices");
    }
    Ok(SpatialReg { label, deformable, frame_of_reference_uid, items })
}

/// Convert a rigid 4×4 DICOM frame-of-reference matrix into our Euler-
/// parameterized rigid transform (center at the origin). Returns `None` when
/// the upper-left 3×3 block is not (close to) a pure rotation.
pub fn matrix_to_rigid(m: &[f64; 16], invert: bool) -> Option<RigidTransform> {
    // Row-major: p' = M·p (homogeneous).
    let mut r = [
        [m[0], m[1], m[2]],
        [m[4], m[5], m[6]],
        [m[8], m[9], m[10]],
    ];
    let mut t = Vec3::new(m[3], m[7], m[11]);

    // Orthonormality check (allow small numeric noise).
    for row in &r {
        let n = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
        if (n - 1.0).abs() > 1e-3 {
            return None;
        }
    }

    if invert {
        // Inverse of rigid: R' = Rᵀ, t' = −Rᵀ t.
        let rt = [
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ];
        let tt = Vec3::new(
            -(rt[0][0] * t.x + rt[0][1] * t.y + rt[0][2] * t.z),
            -(rt[1][0] * t.x + rt[1][1] * t.y + rt[1][2] * t.z),
            -(rt[2][0] * t.x + rt[2][1] * t.y + rt[2][2] * t.z),
        );
        r = rt;
        t = tt;
    }

    // Euler extraction for R = Rz(c)·Ry(b)·Rx(a).
    let b = (-r[2][0]).asin();
    let (a, c) = if b.cos().abs() > 1e-6 {
        (r[2][1].atan2(r[2][2]), r[1][0].atan2(r[0][0]))
    } else {
        (r[0][1].atan2(r[1][1]), 0.0)
    };

    // Verify the decomposition reproduces the matrix (guards against
    // reflections / scaling).
    let rec = RigidTransform::new([a, b, c, t.x, t.y, t.z], Vec3::ZERO);
    for (probe, orig) in [
        Vec3::new(100.0, 0.0, 0.0),
        Vec3::new(0.0, 100.0, 0.0),
        Vec3::new(0.0, 0.0, 100.0),
    ]
    .iter()
    .map(|&p| {
        let mp = Vec3::new(
            r[0][0] * p.x + r[0][1] * p.y + r[0][2] * p.z + t.x,
            r[1][0] * p.x + r[1][1] * p.y + r[1][2] * p.z + t.y,
            r[2][0] * p.x + r[2][1] * p.y + r[2][2] * p.z + t.z,
        );
        (rec.map(p), mp)
    }) {
        if (probe - orig).length() > 1e-3 {
            return None;
        }
    }
    Some(rec)
}

// ---------------------------------------------------------------------------
// RTRECORD — RT (Ion) Beams Treatment Record
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct RecordBeam {
    pub number: i32,
    pub name: String,
    pub specified_meterset: Option<f64>,
    pub delivered_meterset: Option<f64>,
    pub termination_status: String,
    pub verification_status: String,
}

#[derive(Clone)]
pub struct TreatRecord {
    pub label: String,
    pub date: String,
    pub machine: String,
    pub fraction: Option<i32>,
    pub ion: bool,
    pub beams: Vec<RecordBeam>,
}

const SOP_RT_BEAMS_RECORD: &str = "1.2.840.10008.5.1.4.1.1.481.4";
const SOP_RT_ION_BEAMS_RECORD: &str = "1.2.840.10008.5.1.4.1.1.481.9";

pub fn is_record_sop(sop: &str) -> bool {
    sop == SOP_RT_BEAMS_RECORD || sop == SOP_RT_ION_BEAMS_RECORD
}

pub fn load_record(path: &Path) -> Result<TreatRecord> {
    let obj = dicom_object::open_file(path)
        .with_context(|| format!("open RTRECORD {}", path.display()))?;
    let sop = str_of(&obj, tags::SOP_CLASS_UID).unwrap_or_default();

    let (beam_items, ion) =
        if let Some(items) = items_of(&obj, tags::TREATMENT_SESSION_ION_BEAM_SEQUENCE) {
            (Some(items), true)
        } else if let Some(items) = items_of(&obj, tags::TREATMENT_SESSION_BEAM_SEQUENCE) {
            (Some(items), false)
        } else {
            (None, sop == SOP_RT_ION_BEAMS_RECORD)
        };

    let mut beams = Vec::new();
    let mut fraction = None;
    let mut date = str_of(&obj, tags::TREATMENT_DATE).unwrap_or_default();

    if let Some(items) = beam_items {
        for b in items {
            if fraction.is_none() {
                fraction = i32_of(b, tags::CURRENT_FRACTION_NUMBER);
            }
            if date.is_empty() {
                date = str_of(b, tags::TREATMENT_DATE).unwrap_or_default();
            }
            let number = i32_of(b, tags::REFERENCED_BEAM_NUMBER).unwrap_or(-1);
            beams.push(RecordBeam {
                number,
                name: str_of(b, tags::BEAM_NAME).unwrap_or_else(|| format!("Beam {number}")),
                specified_meterset: f64_of(b, tags::SPECIFIED_PRIMARY_METERSET),
                delivered_meterset: f64_of(b, tags::DELIVERED_PRIMARY_METERSET),
                termination_status: str_of(b, tags::TREATMENT_TERMINATION_STATUS)
                    .unwrap_or_default(),
                verification_status: str_of(b, tags::TREATMENT_VERIFICATION_STATUS)
                    .unwrap_or_default(),
            });
        }
    }

    if beams.is_empty() {
        bail!("RTRECORD contains no treatment session beams");
    }

    Ok(TreatRecord {
        label: path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "Record".into()),
        date,
        machine: str_of(&obj, tags::TREATMENT_MACHINE_NAME).unwrap_or_default(),
        fraction,
        ion,
        beams,
    })
}
