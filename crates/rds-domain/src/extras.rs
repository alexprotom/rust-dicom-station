//! Additional DICOM object types beyond volumes and the core RT set:
//!
//! * planar projection images - DX / CR digital radiographs and RTIMAGE
//!   (DRRs, portal / setup images), shown in the rows' planar panes; an RT
//!   Image's geometry (gantry, couch, SAD, SID, receptor, isocentre, patient
//!   position) is read into the IEC parameters and the exact camera of its
//!   pixels ([`rt_image_geometry`]), which is what the 2D/3D registration
//!   projects the planning CT through;
//! * additional objects - secondary captures and other single images that
//!   are not images of the patient: a scanner's patient protocol or dose
//!   report page, a screenshot, a scanned document. The same
//!   [`PlanarImage`] with [`PlanarKind::Additional`], shown in the row's
//!   *Additional* pane and nowhere a radiograph is expected;
//! * REG - Spatial Registration objects (rigid 4×4 frame-of-reference
//!   transformation matrices, and Deformable Spatial Registration objects
//!   with their displacement grids, both of which can be applied as the
//!   active registration);
//! * RTRECORD - RT (Ion) Beams Treatment Records with per-beam specified vs
//!   delivered metersets and termination status.

use std::path::Path;

use anyhow::{bail, Context, Result};
use dicom_dictionary_std::tags;
use dicom_pixeldata::{ConvertOptions, ModalityLutOption, PixelDecoder};

use crate::geometry::Vec3;
use crate::loader::{f64_of, f64s_of, i32_of, items_of, str_of};
use crate::registration::{RigidTransform, VectorField};
use crate::sop::{SOP_DEFORMABLE_REG, SOP_RT_ION_BEAMS_RECORD};

// ---------------------------------------------------------------------------
// Planar images (DX / CR / RTIMAGE) and additional objects
// ---------------------------------------------------------------------------

/// What a single image is: a projection of the patient (a radiograph, a
/// portal image, a DRR), or something else that happens to be an image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum PlanarKind {
    /// A radiograph or an RT image: a camera looked through the patient,
    /// and the image can be compared with a DRR or registered to the CT.
    #[default]
    Projection,
    /// A secondary capture or another image with no projection behind it:
    /// a patient protocol or dose report page a scanner writes, a
    /// screenshot, a scanned document.
    Additional,
}

impl PlanarKind {
    pub fn label(self) -> &'static str {
        match self {
            PlanarKind::Projection => "planar image",
            PlanarKind::Additional => "additional object",
        }
    }
}

/// Modalities whose images are projections (no volume reconstruction).
pub const PROJECTION_MODALITIES: &[&str] = &["DX", "CR", "RTIMAGE", "MG", "XA", "RF", "PX"];
pub use crate::sop::{SOP_DX_PRESENTATION, SOP_DX_PROCESSING, SOP_RTIMAGE, SOP_SECONDARY_CAPTURE};

/// Is this a (multi-frame) Secondary Capture SOP class? An image made of
/// something other than the patient (a report page, a screen).
pub fn is_secondary_capture_sop(sop_class: &str) -> bool {
    crate::sop::is_secondary_capture(sop_class)
}

/// Does this SOP class carry no image at all ([`crate::sop::is_non_image`]):
/// a PDF, a presentation state, a key object selection, raw data, a
/// waveform. The loader leaves such files out rather than decode them.
pub fn is_non_image_sop(sop_class: &str) -> bool {
    crate::sop::is_non_image(sop_class)
}

/// Is this file, by its header, a projection of the patient?
pub fn is_projection(modality: &str, sop_class: &str) -> bool {
    !is_secondary_capture_sop(sop_class)
        && (PROJECTION_MODALITIES.contains(&modality)
            || sop_class == SOP_RTIMAGE
            || sop_class == SOP_DX_PRESENTATION
            || sop_class == SOP_DX_PROCESSING)
}

/// What a single image is, by its header. Everything that is not a
/// projection is an additional object: a secondary capture whatever its
/// modality says (a scanner's protocol page says `CT`), and any other
/// image that is read on its own.
pub fn planar_kind(modality: &str, sop_class: &str) -> PlanarKind {
    if is_projection(modality, sop_class) {
        PlanarKind::Projection
    } else {
        PlanarKind::Additional
    }
}

#[derive(Clone, Default)]
pub struct PlanarImage {
    pub label: String,
    pub modality: String,
    /// A projection of the patient, or an additional object.
    pub kind: PlanarKind,
    pub rows: usize,
    pub cols: usize,
    /// Pixel spacing [x, y] in mm (imager / image plane), 1.0 if unknown.
    pub spacing: [f64; 2],
    /// Decoded values after Modality LUT (MONOCHROME1 already inverted); of
    /// a colour image, the luminance.
    pub data: Vec<f32>,
    /// The colour pixels of an RGB image (a report page, a screenshot), 3
    /// bytes per pixel, drawn as they are; `None` for a grey image.
    pub rgb: Option<Vec<u8>>,
    pub min_value: f32,
    pub max_value: f32,
    /// Default window (center, width).
    pub window: (f32, f32),
    /// Free-form metadata (name, value) for display.
    pub info: Vec<(String, String)>,
    /// SOP Instance UID (empty for an image made here and not yet written).
    pub sop_instance_uid: String,
    /// Frame of Reference the image's geometry is in.
    pub frame_of_reference_uid: String,
    /// Study Instance UID: the study the data tree files it under. Empty
    /// for an image made here (a DRR), which goes under the study of the
    /// series it was made from.
    pub study_uid: String,
    /// Referenced Beam Number of an RT Image.
    pub referenced_beam: Option<i32>,
    /// The IEC parameters the image states - an RT Image's gantry, couch,
    /// SAD, SID, receptor, patient position and isocentre, or the geometry a
    /// DRR was rendered with. `None` when it states too few to place it.
    pub iec: Option<crate::drr::Geometry>,
    /// The exact camera of the image's pixels in patient coordinates. `None`
    /// without `iec`, or without an isocentre to close the chain.
    pub camera: Option<crate::projection::ProjectionGeometry>,
    /// What a usable geometry still lacks, for the UI (`"isocentre"`, `"SID"`).
    pub geometry_missing: Option<String>,
    /// A megavoltage image: exposed by the treatment beam (Primary Dosimeter
    /// Unit MU), so low bone contrast and an aperture in the field.
    pub megavoltage: bool,
    /// Frames in the file: more than one for a cine, of which `data` is the
    /// first ([`load_frame`] reads the others).
    pub frame_count: usize,
    /// The file the image was read from.
    pub source_path: Option<std::path::PathBuf>,
}

/// Frame `frame` of a multi-frame planar image, with the same Modality LUT
/// and MONOCHROME1 handling as [`load_planar`].
pub fn load_frame(path: &Path, frame: usize) -> Result<Vec<f32>> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open planar image {}", path.display()))?;
    let decoded = obj
        .decode_pixel_data()
        .with_context(|| format!("decode pixel data of {}", path.display()))?;
    let n = (decoded.rows() as usize) * (decoded.columns() as usize);
    if frame >= decoded.number_of_frames().max(1) as usize {
        bail!("{} has no frame {}", path.display(), frame + 1);
    }
    let opts = ConvertOptions::new().with_modality_lut(ModalityLutOption::Default);
    let mut data: Vec<f32> = decoded
        .to_vec_frame_with_options(frame as u32, &opts)
        .with_context(|| format!("convert frame {} of {}", frame + 1, path.display()))?;
    if decoded.samples_per_pixel() == 3 && data.len() >= n * 3 {
        data = data[..n * 3]
            .as_chunks::<3>()
            .0
            .iter()
            .map(|px| 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
            .collect();
    }
    data.truncate(n);
    let mono1 = str_of(&obj, tags::PHOTOMETRIC_INTERPRETATION).is_some_and(|p| p == "MONOCHROME1");
    if mono1 {
        let (lo, hi) = data
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        for v in &mut data {
            *v = lo + hi - *v;
        }
    }
    Ok(data)
}

/// The geometry an RT Image states, or what it is missing.
///
/// The chain is PS3.3 C.8.8.2: *Radiation Machine SAD* and *RT Image SID*
/// place source and image plane; *Gantry Angle*, *Patient Support Angle*,
/// *Table Top Pitch / Roll* and *Beam Limiting Device Angle* (inside the
/// first Exposure Sequence item when present there, which supersedes the
/// top level, C.8.8.2.4) turn it; *X-Ray Image Receptor Translation* (the
/// receptor origin's absolute position in gantry coordinates) and *Angle*
/// place the receptor; *RT Image Position* is the centre of the first pixel
/// in `(Xr, Yr)`, and *RT Image Orientation* the row and column directions
/// (default rows along `+Xr`, columns along `−Yr`, C.8.8.2.7); *Isocenter
/// Position* and *Patient Position* bring it into patient coordinates.
pub fn rt_image_geometry(
    obj: &dicom_object::InMemDicomObject,
    cols: usize,
    rows: usize,
    spacing: [f64; 2],
) -> (
    Option<crate::drr::Geometry>,
    Option<crate::projection::ProjectionGeometry>,
    Option<String>,
) {
    // An attribute from the first exposure when it is there, else the top.
    let exposure = items_of(obj, tags::EXPOSURE_SEQUENCE).and_then(|s| s.first());
    let angle = |tag| {
        exposure
            .and_then(|e| f64_of(e, tag))
            .or_else(|| f64_of(obj, tag))
    };
    let sad = f64_of(obj, tags::RADIATION_MACHINE_SAD).filter(|v| *v > 0.0);
    let sid = f64_of(obj, tags::RT_IMAGE_SID).filter(|v| *v > 0.0);
    let Some(sad) = sad else {
        return (None, None, Some("SAD".into()));
    };
    let Some(sid) = sid else {
        return (None, None, Some("SID".into()));
    };
    let vec3 = |tag| {
        f64s_of(obj, tag)
            .filter(|v| v.len() >= 3)
            .map(|v| Vec3::new(v[0], v[1], v[2]))
    };
    let isocenter = vec3(tags::ISOCENTER_POSITION);
    let patient = str_of(obj, tags::PATIENT_POSITION)
        .and_then(|s| crate::iec::PatientPosition::from_code(&s))
        .map(crate::iec::PatientFrame::Position)
        .unwrap_or_default();
    let first = f64s_of(obj, tags::RT_IMAGE_POSITION)
        .filter(|v| v.len() >= 2)
        .map(|v| [v[0], v[1]])
        .unwrap_or([
            -(cols as f64 - 1.0) * 0.5 * spacing[0],
            (rows as f64 - 1.0) * 0.5 * spacing[1],
        ]);
    let (row, col) = f64s_of(obj, tags::RT_IMAGE_ORIENTATION)
        .filter(|v| v.len() >= 6)
        .map(|v| {
            (
                Vec3::new(v[0], v[1], v[2]).normalized(),
                Vec3::new(v[3], v[4], v[5]).normalized(),
            )
        })
        .unwrap_or((Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0)));
    let center = |half: [f64; 2]| {
        let c = Vec3::new(first[0], first[1], 0.0) + row * half[0] + col * half[1];
        [c.x, c.y]
    };
    let geometry = crate::drr::Geometry {
        sad,
        sid,
        gantry_deg: angle(tags::GANTRY_ANGLE).unwrap_or(0.0),
        couch_deg: angle(tags::PATIENT_SUPPORT_ANGLE).unwrap_or(0.0),
        pitch_deg: angle(tags::TABLE_TOP_PITCH_ANGLE).unwrap_or(0.0),
        roll_deg: angle(tags::TABLE_TOP_ROLL_ANGLE).unwrap_or(0.0),
        collimator_deg: angle(tags::BEAM_LIMITING_DEVICE_ANGLE).unwrap_or(0.0),
        isocenter: isocenter.unwrap_or(Vec3::ZERO),
        patient,
        receptor: vec3(tags::X_RAY_IMAGE_RECEPTOR_TRANSLATION),
        receptor_angle_deg: f64_of(obj, tags::X_RAY_IMAGE_RECEPTOR_ANGLE).unwrap_or(0.0),
        image_center: center([
            (cols as f64 - 1.0) * 0.5 * spacing[0],
            (rows as f64 - 1.0) * 0.5 * spacing[1],
        ]),
        panel_mm: [cols as f64 * spacing[0], rows as f64 * spacing[1]],
        dims: [cols, rows],
    };
    match isocenter {
        Some(_) => {
            let cam = geometry.camera_of_pixels(first, row, col, spacing, [cols, rows]);
            (Some(geometry), Some(cam), None)
        }
        None => (Some(geometry), None, Some("isocentre".into())),
    }
}

pub fn load_planar(path: &Path) -> Result<PlanarImage> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open planar image {}", path.display()))?;
    planar_of(&obj, path)
}

/// The overlay planes of an image (groups 6000 to 601E, one bit per
/// pixel) drawn into its values: a set bit becomes the brightest value the
/// image has, or one step above a blank image's. A scanner's report page
/// is nothing but its overlay (blank pixels, the text as bits), and a
/// radiograph's overlay is the graphics burnt onto it at the console.
/// Returns how many planes were drawn.
pub fn draw_overlays(
    obj: &dicom_object::InMemDicomObject,
    rows: usize,
    cols: usize,
    data: &mut [f32],
    rgb: &mut Option<Vec<u8>>,
) -> usize {
    let (mut min_v, mut max_v) = (f32::MAX, f32::MIN);
    for &v in data.iter() {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if data.is_empty() {
        return 0;
    }
    let on = max_v.max(min_v + 1.0);
    let mut drawn = 0;
    for group in (0x6000u16..=0x601E).step_by(2) {
        let t = |e: u16| dicom_core::Tag(group, e);
        let Some(bytes) = obj.element(t(0x3000)).ok().and_then(|e| e.to_bytes().ok()) else {
            continue;
        };
        let o_rows = i32_of(obj, t(0x0010)).unwrap_or(rows as i32).max(0) as usize;
        let o_cols = i32_of(obj, t(0x0011)).unwrap_or(cols as i32).max(0) as usize;
        if i32_of(obj, t(0x0100)).unwrap_or(1) != 1 || o_rows == 0 || o_cols == 0 {
            continue;
        }
        // The origin names the image pixel of the overlay's first bit,
        // one-based, as [row, column].
        let origin = obj
            .element(t(0x0050))
            .ok()
            .and_then(|e| e.to_multi_int::<i32>().ok())
            .filter(|v| v.len() >= 2)
            .map(|v| [v[0], v[1]])
            .unwrap_or([1, 1]);
        if bytes.len() * 8 < o_rows * o_cols {
            continue;
        }
        for r in 0..o_rows {
            let y = r as i64 + origin[0] as i64 - 1;
            if y < 0 || y >= rows as i64 {
                continue;
            }
            for c in 0..o_cols {
                let x = c as i64 + origin[1] as i64 - 1;
                if x < 0 || x >= cols as i64 {
                    continue;
                }
                let bit = r * o_cols + c;
                if bytes[bit / 8] >> (bit % 8) & 1 == 0 {
                    continue;
                }
                let i = y as usize * cols + x as usize;
                data[i] = on;
                if let Some(rgb) = rgb {
                    if let Some(px) = rgb.get_mut(i * 3..i * 3 + 3) {
                        px.fill(255);
                    }
                }
            }
        }
        drawn += 1;
    }
    drawn
}

/// The planar image of an opened file; [`load_planar`] opens it.
pub fn planar_of(obj: &dicom_object::DefaultDicomObject, path: &Path) -> Result<PlanarImage> {
    let sop_class = str_of(obj, tags::SOP_CLASS_UID).unwrap_or_default();
    let modality = str_of(obj, tags::MODALITY).unwrap_or_else(|| "DX".into());
    let kind = planar_kind(&modality, &sop_class);

    let decoded = obj
        .decode_pixel_data()
        .with_context(|| format!("decode pixel data of {}", path.display()))?;
    let rows = decoded.rows() as usize;
    let cols = decoded.columns() as usize;
    // A cine (XA / RF) is one view over time: its first frame is the image,
    // and the rest are read when something asks for them (`load_frame`).
    let frame_count = decoded.number_of_frames().max(1) as usize;
    let samples = decoded.samples_per_pixel().max(1) as usize;
    let opts = ConvertOptions::new().with_modality_lut(ModalityLutOption::Default);
    let mut data: Vec<f32> = decoded
        .to_vec_frame_with_options(0, &opts)
        .with_context(|| format!("convert pixels of {}", path.display()))?;
    if data.len() < rows * cols * samples {
        bail!(
            "pixel buffer smaller than Rows×Columns in {}",
            path.display()
        );
    }
    // A colour image (a report page, a screenshot: three samples per pixel)
    // keeps its colours for the pane and is windowed by its luminance.
    let mut rgb = (samples == 3).then(|| {
        data[..rows * cols * 3]
            .iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect::<Vec<u8>>()
    });
    if samples == 3 {
        data = data[..rows * cols * 3]
            .as_chunks::<3>()
            .0
            .iter()
            .map(|px| 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2])
            .collect();
    }
    data.truncate(rows * cols);

    let mono1 = str_of(obj, tags::PHOTOMETRIC_INTERPRETATION)
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
    // The overlay planes, drawn in white on top: a report page is nothing
    // else, and a radiograph's are the console's graphics.
    let overlays = draw_overlays(obj, rows, cols, &mut data, &mut rgb);
    if overlays > 0 {
        max_v = f32::MIN;
        min_v = f32::MAX;
        for &v in &data {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }

    let spacing = f64s_of(obj, tags::IMAGE_PLANE_PIXEL_SPACING)
        .or_else(|| f64s_of(obj, tags::IMAGER_PIXEL_SPACING))
        .or_else(|| f64s_of(obj, tags::PIXEL_SPACING))
        .filter(|v| v.len() >= 2)
        .map(|v| [v[1], v[0]])
        .unwrap_or([1.0, 1.0]);

    // The window the file carries, when it is a window onto these values;
    // else the values' own range. A page (a protocol, a dose report) often
    // carries a window that shows nothing of it, or one meant for the CT it
    // came with, so an additional object always comes up over its range.
    let range = ((min_v + max_v) * 0.5, (max_v - min_v).max(1.0));
    let window = match (
        f64s_of(obj, tags::WINDOW_CENTER).and_then(|v| v.first().copied()),
        f64s_of(obj, tags::WINDOW_WIDTH).and_then(|v| v.first().copied()),
    ) {
        (Some(c), Some(w))
            if w > 0.0
                && !mono1
                && kind == PlanarKind::Projection
                && (c - w * 0.5) < max_v as f64
                && (c + w * 0.5) > min_v as f64 =>
        {
            (c as f32, w as f32)
        }
        _ => range,
    };

    let label = str_of(obj, tags::RT_IMAGE_LABEL)
        .or_else(|| str_of(obj, tags::SERIES_DESCRIPTION))
        .or_else(|| path.file_stem().map(|s| s.to_string_lossy().into_owned()))
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
    add(
        "Date",
        str_of(obj, tags::CONTENT_DATE).or_else(|| str_of(obj, tags::STUDY_DATE)),
    );
    if modality == "RTIMAGE" {
        add("Machine", str_of(obj, tags::RADIATION_MACHINE_NAME));
        add(
            "Gantry",
            f64_of(obj, tags::GANTRY_ANGLE).map(|g| format!("{g:.1}°")),
        );
        add(
            "SAD",
            f64_of(obj, tags::RADIATION_MACHINE_SAD).map(|v| format!("{v:.0} mm")),
        );
        add(
            "SID",
            f64_of(obj, tags::RT_IMAGE_SID).map(|v| format!("{v:.0} mm")),
        );
        add("Description", str_of(obj, tags::RT_IMAGE_DESCRIPTION));
    } else if kind == PlanarKind::Additional {
        add(
            "Kind",
            Some(if is_secondary_capture_sop(&sop_class) {
                "secondary capture".to_string()
            } else {
                format!("{modality} image without a position")
            }),
        );
        add(
            "Image type",
            str_of(obj, tags::IMAGE_TYPE).map(|t| t.replace('\\', " / ")),
        );
        // What the pixels hold, so a page that comes up blank says why.
        add(
            "Values",
            Some(if rgb.is_some() {
                format!("colour, luminance {min_v:.0} to {max_v:.0}")
            } else {
                format!("{min_v:.0} to {max_v:.0}")
            }),
        );
    } else {
        add("Body part", str_of(obj, tags::BODY_PART_EXAMINED));
        add("View", str_of(obj, tags::VIEW_POSITION));
        add("KVP", str_of(obj, tags::KVP));
    }
    info.push((
        "Size".into(),
        format!("{cols}×{rows} px · {:.2}/{:.2} mm", spacing[0], spacing[1]),
    ));
    if frame_count > 1 {
        info.push(("Frames".into(), format!("{frame_count} (the first shown)")));
    }

    let (iec, camera, geometry_missing) = if modality == "RTIMAGE" {
        rt_image_geometry(obj, cols, rows, spacing)
    } else {
        (None, None, None)
    };
    let megavoltage = str_of(obj, tags::PRIMARY_DOSIMETER_UNIT).is_some_and(|u| u == "MU");
    Ok(PlanarImage {
        label,
        modality,
        kind,
        rows,
        cols,
        spacing,
        data,
        rgb,
        min_value: min_v,
        max_value: max_v,
        window,
        info,
        sop_instance_uid: str_of(obj, tags::SOP_INSTANCE_UID).unwrap_or_default(),
        frame_of_reference_uid: str_of(obj, tags::FRAME_OF_REFERENCE_UID).unwrap_or_default(),
        study_uid: str_of(obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        referenced_beam: i32_of(obj, tags::REFERENCED_BEAM_NUMBER),
        iec,
        camera,
        geometry_missing,
        megavoltage,
        frame_count,
        source_path: Some(path.to_path_buf()),
    })
}

// ---------------------------------------------------------------------------
// REG - Spatial Registration
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
    /// True for Deformable Spatial Registration Storage.
    pub deformable: bool,
    /// Frame of Reference of the registration instance itself (the frame
    /// the matrices transform *into*).
    pub frame_of_reference_uid: String,
    pub items: Vec<RegMatrixItem>,
    /// The displacement lattice of a Deformable Spatial Registration, when
    /// the object carries one - it can be applied as the active
    /// registration exactly like a matrix.
    pub grid: Option<VectorField>,
    /// Frame of Reference the grid's own lattice lives in.
    pub grid_source_for_uid: String,
}

pub fn is_reg_sop(sop: &str) -> bool {
    crate::sop::is_registration(sop)
}

pub fn load_reg(path: &Path) -> Result<SpatialReg> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open REG {}", path.display()))?;
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
            let Some(mreg) = items_of(r, tags::MATRIX_REGISTRATION_SEQUENCE) else {
                continue;
            };
            for mr in mreg {
                let Some(ms) = items_of(mr, tags::MATRIX_SEQUENCE) else {
                    continue;
                };
                for m in ms {
                    let Some(vals) = f64s_of(m, tags::FRAME_OF_REFERENCE_TRANSFORMATION_MATRIX)
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
                    items.push(RegMatrixItem {
                        for_uid: for_uid.clone(),
                        matrix,
                        matrix_type,
                        is_identity,
                    });
                }
            }
        }
    }
    let (grid, grid_source_for_uid) = if deformable {
        read_deformation_grid(&obj)
    } else {
        (None, String::new())
    };
    if items.is_empty() && grid.is_none() {
        bail!("REG object contains no transformation matrices and no deformation grid");
    }
    Ok(SpatialReg {
        label,
        deformable,
        frame_of_reference_uid,
        items,
        grid,
        grid_source_for_uid,
    })
}

// The Deformable Spatial Registration IOD's own tags, by number so the
// reader does not depend on which release of the data dictionary is linked.
const TAG_DEFORMABLE_REGISTRATION_SEQ: dicom_core::Tag = dicom_core::Tag(0x0064, 0x0002);
const TAG_SOURCE_FRAME_OF_REFERENCE_UID: dicom_core::Tag = dicom_core::Tag(0x0064, 0x0003);
const TAG_DEFORMABLE_REGISTRATION_GRID_SEQ: dicom_core::Tag = dicom_core::Tag(0x0064, 0x0005);
const TAG_GRID_DIMENSIONS: dicom_core::Tag = dicom_core::Tag(0x0064, 0x0007);
const TAG_GRID_RESOLUTION: dicom_core::Tag = dicom_core::Tag(0x0064, 0x0008);
const TAG_VECTOR_GRID_DATA: dicom_core::Tag = dicom_core::Tag(0x0064, 0x0009);

/// Read the displacement lattice of a Deformable Spatial Registration.
///
/// Everything is optional and everything is checked: a file whose grid does
/// not describe itself consistently is simply reported as having no grid,
/// which leaves its matrices usable rather than failing the whole load.
fn read_deformation_grid(obj: &dicom_object::DefaultDicomObject) -> (Option<VectorField>, String) {
    let Some(regs) = items_of(obj, TAG_DEFORMABLE_REGISTRATION_SEQ) else {
        return (None, String::new());
    };
    for r in regs {
        let source = str_of(r, TAG_SOURCE_FRAME_OF_REFERENCE_UID).unwrap_or_default();
        let Some(grids) = items_of(r, TAG_DEFORMABLE_REGISTRATION_GRID_SEQ) else {
            continue;
        };
        for g in grids {
            let dims: Vec<usize> = match f64s_of(g, TAG_GRID_DIMENSIONS) {
                Some(v) if v.len() >= 3 => v[..3].iter().map(|x| *x as usize).collect(),
                _ => continue,
            };
            let res: Vec<f64> = match f64s_of(g, TAG_GRID_RESOLUTION) {
                Some(v) if v.len() >= 3 => v[..3].to_vec(),
                _ => continue,
            };
            let Some(pos) = f64s_of(g, tags::IMAGE_POSITION_PATIENT).filter(|v| v.len() >= 3)
            else {
                continue;
            };
            let Some(orient) = f64s_of(g, tags::IMAGE_ORIENTATION_PATIENT).filter(|v| v.len() >= 6)
            else {
                continue;
            };
            let Some(raw) = g
                .element(TAG_VECTOR_GRID_DATA)
                .ok()
                .and_then(|e| e.value().to_multi_float32().ok())
            else {
                continue;
            };
            let n = dims[0] * dims[1] * dims[2];
            if n == 0 || raw.len() < 3 * n {
                continue;
            }
            let row = Vec3::from_slice(&orient[0..3]).normalized();
            let col = Vec3::from_slice(&orient[3..6]).normalized();
            let data: Vec<Vec3> = raw[..3 * n]
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64))
                .collect();
            let max_mag = data.iter().map(|v| v.length()).fold(0.0f64, f64::max);
            return (
                Some(VectorField {
                    dims: [dims[0], dims[1], dims[2]],
                    spacing: [res[0], res[1], res[2]],
                    origin: Vec3::from_slice(&pos[0..3]),
                    axes: [row, col, row.cross(col).normalized()],
                    data,
                    max_mag,
                    region: None,
                }),
                source,
            );
        }
    }
    (None, String::new())
}

/// Convert a rigid 4×4 DICOM frame-of-reference matrix into our Euler-
/// parameterized rigid transform (center at the origin). Returns `None` when
/// the upper-left 3×3 block is not (close to) a pure rotation.
pub fn matrix_to_rigid(m: &[f64; 16], invert: bool) -> Option<RigidTransform> {
    // Row-major: p' = M·p (homogeneous).
    let mut r = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
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
// RTRECORD - RT (Ion) Beams Treatment Record
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

pub fn is_record_sop(sop: &str) -> bool {
    crate::sop::is_beams_record(sop)
}

pub fn load_record(path: &Path) -> Result<TreatRecord> {
    let obj = crate::dicomfile::open_full(path)
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

#[cfg(test)]
mod rt_image_tests {
    use super::*;
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_object::InMemDicomObject;

    fn ds(obj: &mut InMemDicomObject, tag: dicom_core::Tag, vals: &[f64]) {
        crate::dicom_export::put_ds(obj, tag, vals);
    }

    fn rt_image(extra: impl FnOnce(&mut InMemDicomObject)) -> InMemDicomObject {
        let mut o = InMemDicomObject::new_empty();
        ds(&mut o, tags::RADIATION_MACHINE_SAD, &[1000.0]);
        ds(&mut o, tags::RT_IMAGE_SID, &[1500.0]);
        ds(&mut o, tags::GANTRY_ANGLE, &[90.0]);
        ds(&mut o, tags::PATIENT_SUPPORT_ANGLE, &[0.0]);
        ds(&mut o, tags::ISOCENTER_POSITION, &[1.0, 2.0, 3.0]);
        o.put(DataElement::new(
            tags::PATIENT_POSITION,
            VR::CS,
            PrimitiveValue::from("HFS"),
        ));
        extra(&mut o);
        o
    }

    /// A centred RT image with nothing but the IEC parameters is the camera
    /// the DRR geometry of those parameters makes.
    #[test]
    fn a_centred_rt_image_is_the_drr_camera() {
        let (g, cam, missing) = rt_image_geometry(&rt_image(|_| {}), 8, 6, [0.5, 0.5]);
        assert!(missing.is_none(), "{missing:?}");
        let g = g.unwrap();
        let cam = cam.unwrap();
        let want = crate::drr::Geometry {
            gantry_deg: 90.0,
            isocenter: Vec3::new(1.0, 2.0, 3.0),
            panel_mm: [4.0, 3.0],
            dims: [8, 6],
            ..Default::default()
        };
        assert_eq!(g.panel_mm, want.panel_mm);
        assert!(g.image_center[0].abs() < 1e-12 && g.image_center[1].abs() < 1e-12);
        let w = want.camera();
        assert!((cam.source - w.source).length() < 1e-9);
        assert!((cam.origin - w.origin).length() < 1e-9);
        assert!((cam.row_step - w.row_step).length() < 1e-9);
        assert!((cam.col_step - w.col_step).length() < 1e-9);
        assert!((g.camera().origin - w.origin).length() < 1e-9);
    }

    /// RT Image Position moves the pixels in the receptor plane, and the
    /// receptor translation is an absolute position of the receptor origin.
    #[test]
    fn the_position_and_the_receptor_translation_move_the_pixels() {
        let o = rt_image(|o| {
            ds(o, tags::RT_IMAGE_POSITION, &[-10.0, 20.0]);
            ds(
                o,
                tags::X_RAY_IMAGE_RECEPTOR_TRANSLATION,
                &[5.0, 0.0, -500.0],
            );
        });
        let (g, cam, _) = rt_image_geometry(&o, 8, 6, [0.5, 0.5]);
        let (g, cam) = (g.unwrap(), cam.unwrap());
        // Gantry 90 for HFS: the gantry's Zg is the patient's +x (left), its
        // Xg the patient's anterior... the receptor origin sits 500 mm past
        // the isocentre on the far side (towards -x), shifted 5 mm along Xg.
        let f = g.frame();
        let first = cam.pixel(0.0, 0.0);
        let rec = Vec3::new(1.0, 2.0, 3.0) + f.axis * 500.0 + f.panel_x * 5.0;
        let want = rec + f.panel_x * -10.0 + f.panel_y * 20.0;
        assert!((first - want).length() < 1e-9, "{first:?} vs {want:?}");
        // The IEC parameters carry the same picture for the default
        // orientation: the image centre in the receptor plane.
        assert!((g.camera().origin - cam.origin).length() < 1e-9);
        // An RT image with no isocentre has parameters, but no camera.
        let mut o = rt_image(|_| {});
        o.remove_element(tags::ISOCENTER_POSITION);
        let (g, cam, missing) = rt_image_geometry(&o, 8, 6, [0.5, 0.5]);
        assert!(g.is_some() && cam.is_none());
        assert_eq!(missing.as_deref(), Some("isocentre"));
        let mut o = rt_image(|_| {});
        o.remove_element(tags::RT_IMAGE_SID);
        assert_eq!(
            rt_image_geometry(&o, 8, 6, [0.5, 0.5]).2.as_deref(),
            Some("SID")
        );
    }
}
