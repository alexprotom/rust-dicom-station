//! RT Structure Set (RTSTRUCT) parsing: ROI names, colors and contours.

use std::path::Path;

use anyhow::{Context, Result};
use dicom_dictionary_std::tags;

use crate::geometry::Vec3;
use crate::loader::{f64s_of, i32_of, items_of, str_of};

#[derive(Clone)]
pub struct Contour {
    /// Points in patient coordinates (mm). Closed planar contours are not
    /// explicitly closed (last point != first); rendering closes them.
    pub points: Vec<Vec3>,
    /// CLOSED_PLANAR, OPEN_PLANAR, POINT, …
    pub geometric_type: String,
}

#[derive(Clone)]
pub struct Roi {
    pub number: i32,
    pub name: String,
    pub color: [u8; 3],
    /// RT ROI Interpreted Type: PTV, CTV, GTV, ORGAN, EXTERNAL, AVOIDANCE, …
    pub roi_type: String,
    pub contours: Vec<Contour>,
}

#[derive(Clone)]
pub struct StructureSet {
    pub label: String,
    pub frame_of_reference_uid: String,
    pub rois: Vec<Roi>,
}

/// Fallback palette for ROIs without a display color.
const PALETTE: &[[u8; 3]] = &[
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [170, 110, 40],
];

pub fn load(path: &Path) -> Result<StructureSet> {
    let obj = dicom_object::open_file(path)
        .with_context(|| format!("open RTSTRUCT {}", path.display()))?;

    let label = str_of(&obj, tags::STRUCTURE_SET_LABEL)
        .or_else(|| str_of(&obj, tags::STRUCTURE_SET_NAME))
        .unwrap_or_else(|| "Structure Set".into());

    let frame_of_reference_uid = items_of(&obj, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE)
        .and_then(|items| items.first())
        .and_then(|it| str_of(it, tags::FRAME_OF_REFERENCE_UID))
        .unwrap_or_default();

    // ROI number -> (name)
    let mut names: Vec<(i32, String)> = Vec::new();
    if let Some(items) = items_of(&obj, tags::STRUCTURE_SET_ROI_SEQUENCE) {
        for it in items {
            let number = i32_of(it, tags::ROI_NUMBER).unwrap_or(-1);
            let name = str_of(it, tags::ROI_NAME).unwrap_or_else(|| format!("ROI {number}"));
            names.push((number, name));
        }
    }

    // ROI number -> interpreted type
    let mut types: Vec<(i32, String)> = Vec::new();
    if let Some(items) = items_of(&obj, tags::RTROI_OBSERVATIONS_SEQUENCE) {
        for it in items {
            let number = i32_of(it, tags::REFERENCED_ROI_NUMBER).unwrap_or(-1);
            let t = str_of(it, tags::RTROI_INTERPRETED_TYPE).unwrap_or_default();
            types.push((number, t));
        }
    }

    let mut rois = Vec::new();
    if let Some(items) = items_of(&obj, tags::ROI_CONTOUR_SEQUENCE) {
        for (idx, it) in items.iter().enumerate() {
            let number = i32_of(it, tags::REFERENCED_ROI_NUMBER).unwrap_or(-1);
            let name = names
                .iter()
                .find(|(n, _)| *n == number)
                .map(|(_, s)| s.clone())
                .unwrap_or_else(|| format!("ROI {number}"));
            let roi_type = types
                .iter()
                .find(|(n, _)| *n == number)
                .map(|(_, s)| s.clone())
                .unwrap_or_default();
            let color = f64s_of(it, tags::ROI_DISPLAY_COLOR)
                .filter(|v| v.len() >= 3)
                .map(|v| {
                    [
                        v[0].clamp(0.0, 255.0) as u8,
                        v[1].clamp(0.0, 255.0) as u8,
                        v[2].clamp(0.0, 255.0) as u8,
                    ]
                })
                .unwrap_or(PALETTE[idx % PALETTE.len()]);

            let mut contours = Vec::new();
            if let Some(citems) = items_of(it, tags::CONTOUR_SEQUENCE) {
                for c in citems {
                    let geometric_type =
                        str_of(c, tags::CONTOUR_GEOMETRIC_TYPE).unwrap_or_else(|| "CLOSED_PLANAR".into());
                    let Some(data) = f64s_of(c, tags::CONTOUR_DATA) else { continue };
                    if data.len() < 3 {
                        continue;
                    }
                    let points: Vec<Vec3> = data
                        .chunks_exact(3)
                        .map(Vec3::from_slice)
                        .collect();
                    contours.push(Contour { points, geometric_type });
                }
            }

            rois.push(Roi { number, name, color, roi_type, contours });
        }
    }

    // Stable, user-friendly ordering: external/body first, then targets, then rest alphabetically.
    rois.sort_by(|a, b| {
        fn rank(r: &Roi) -> u8 {
            match r.roi_type.as_str() {
                "EXTERNAL" => 0,
                "PTV" => 1,
                "CTV" => 2,
                "GTV" => 3,
                _ => 4,
            }
        }
        rank(a).cmp(&rank(b)).then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    Ok(StructureSet { label, frame_of_reference_uid, rois })
}
