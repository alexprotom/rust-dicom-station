//! DICOM Segmentation (SEG) objects: the segmentation-series data model,
//! reading Segmentation Storage instances, and writing painted segmentations
//! back out as binary SEG.
//!
//! A SEG instance is a multi-frame image whose frames are binary masks, one
//! per (segment, slice) pair, positioned in patient space by the per-frame
//! functional groups rather than by a slice index. Reading therefore means
//! reconstructing a lattice out of the frame positions, and writing means
//! emitting one frame per slice a segment actually occupies - a segment that
//! covers ten slices of a 200-slice CT costs ten frames, not two hundred.
//!
//! The masks keep the lattice they arrived on ([`SegSeries::grid`]) instead
//! of being forced onto the displayed volume at load time: a study can hold
//! segmentations of several image series at once, and each is resampled onto
//! the display grid only when its own series is the one being shown
//! ([`SegSeries::rebind`]).

use std::path::Path;

use anyhow::{bail, Context, Result};
use dicom_core::value::{PrimitiveValue, C};
use dicom_core::{DataElement, VR};
use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;
use rayon::prelude::*;

use crate::dicom_export::{
    new_uid, put_ds, put_is, put_seq, put_str, put_strs, put_us, write_object, ExportParams,
};
use crate::geometry::Vec3;
use crate::loader::{f64s_of, i32_of, items_of, str_of};
use crate::segmentation::{Segmentation, SEG_PALETTE};
use crate::volume::{Grid, Volume};

/// Segmentation Storage.
pub const SOP_SEG: &str = "1.2.840.10008.5.1.4.1.1.66.4";
/// CT Image Storage - what the referenced instances of an export are.
const SOP_CT: &str = "1.2.840.10008.5.1.4.1.1.2";

// ---------------------------------------------------------------------------
// The segmentation series
// ---------------------------------------------------------------------------

/// A DICOM Segmentation series: binary segments sharing one Series Instance
/// UID, one voxel lattice and one referenced image series.
///
/// This is the home of *every* segmentation in the application, painted or
/// imported: the interactive tools edit the segments of the active series,
/// the DICOM export writes one SEG file per series, and the data tree shows
/// each series as a node that can be re-pointed at another image series.
#[derive(Clone)]
pub struct SegSeries {
    /// Series description / content label shown in the tree.
    pub label: String,
    pub sop_instance_uid: String,
    /// Study this series belongs to.
    pub study_uid: String,
    /// Image series the segments belong to (ReferencedSeriesSequence).
    pub referenced_series_uid: String,
    /// Source file name; empty for series created in the application.
    pub file_name: String,
    /// The lattice `segs` currently live on.
    pub grid: Grid,
    pub segs: Vec<Segmentation>,
}

impl SegSeries {
    /// An empty series on `grid`, drawn on the image series `series_uid`.
    pub fn new(
        label: String,
        grid: Grid,
        referenced_series_uid: String,
        study_uid: String,
    ) -> Self {
        SegSeries {
            label,
            sop_instance_uid: new_uid(),
            study_uid,
            referenced_series_uid,
            file_name: String::new(),
            grid,
            segs: Vec::new(),
        }
    }

    /// Resample every segment onto `vol`'s lattice, so the overlays, the
    /// brush and the meshes can all index them with the displayed volume's
    /// dimensions. A no-op when the series is already on that lattice.
    ///
    /// Undo history does not survive a rebind - the voxels it refers to no
    /// longer exist.
    pub fn rebind(&mut self, vol: &Volume) -> bool {
        let to = vol.grid();
        if self.grid.matches(&to) {
            return false;
        }
        let from = self.grid.clone();
        for seg in &mut self.segs {
            let mask = resample_mask(&seg.mask, &from, &to);
            *seg = Segmentation::from_mask(seg.name.clone(), seg.color, to.dims, mask)
                .with_visible(seg.visible);
        }
        self.grid = to;
        true
    }
}

/// Nearest-neighbour resample of a binary mask from one lattice onto another.
///
/// Destination driven - every voxel of `to` asks `from` what it holds - so a
/// segmentation coarser than the image it is shown on never breaks up into
/// stripes, which is exactly the common case (a 3 mm SEG on a 1 mm CT).
///
/// The index map between two lattices is affine, so the inner loop is three
/// additions per axis rather than a matrix product per voxel.
pub fn resample_mask(mask: &[u8], from: &Grid, to: &Grid) -> Vec<u8> {
    let [nx, ny, nz] = to.dims;
    let [sx, sy, sz] = from.dims;
    let mut out = vec![0u8; nx * ny * nz];
    if mask.len() < sx * sy * sz || sx == 0 || sy == 0 || sz == 0 {
        return out;
    }
    // Columns of the destination-index → source-index affine map.
    let axis = |d: Vec3| {
        [
            d.dot(from.row_dir) / from.spacing[0],
            d.dot(from.col_dir) / from.spacing[1],
            d.dot(from.normal) / from.spacing[2],
        ]
    };
    let ci = axis(to.row_dir * to.spacing[0]);
    let cj = axis(to.col_dir * to.spacing[1]);
    let ck = axis(to.normal * to.spacing[2]);
    let c0 = axis(to.origin - from.origin);

    out.par_chunks_mut(nx * ny)
        .enumerate()
        .for_each(|(k, plane)| {
            let base = [
                c0[0] + ck[0] * k as f64,
                c0[1] + ck[1] * k as f64,
                c0[2] + ck[2] * k as f64,
            ];
            for j in 0..ny {
                let row = [
                    base[0] + cj[0] * j as f64,
                    base[1] + cj[1] * j as f64,
                    base[2] + cj[2] * j as f64,
                ];
                for i in 0..nx {
                    let u = (row[0] + ci[0] * i as f64).round();
                    let v = (row[1] + ci[1] * i as f64).round();
                    let w = (row[2] + ci[2] * i as f64).round();
                    if u < 0.0 || v < 0.0 || w < 0.0 {
                        continue;
                    }
                    let (u, v, w) = (u as usize, v as usize, w as usize);
                    if u >= sx || v >= sy || w >= sz {
                        continue;
                    }
                    plane[j * nx + i] = mask[w * sx * sy + v * sx + u];
                }
            }
        });
    out
}

// ---------------------------------------------------------------------------
// Colors: DICOM CIELab ⇄ sRGB
// ---------------------------------------------------------------------------

/// Decode a Recommended Display CIELab Value (three 16-bit unsigned values,
/// L\* over 0-100 and a\*/b\* over −128-127) into sRGB.
pub fn cielab_to_rgb(v: [u16; 3]) -> [u8; 3] {
    let l = v[0] as f64 / 65535.0 * 100.0;
    let a = v[1] as f64 / 65535.0 * 255.0 - 128.0;
    let b = v[2] as f64 / 65535.0 * 255.0 - 128.0;
    // CIELab → XYZ (D65 white point).
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let g = |t: f64| {
        if t > 6.0 / 29.0 {
            t * t * t
        } else {
            3.0 * (6.0f64 / 29.0).powi(2) * (t - 4.0 / 29.0)
        }
    };
    let (x, y, z) = (0.95047 * g(fx), g(fy), 1.08883 * g(fz));
    // XYZ → linear sRGB → gamma.
    let lin = [
        3.2406 * x - 1.5372 * y - 0.4986 * z,
        -0.9689 * x + 1.8758 * y + 0.0415 * z,
        0.0557 * x - 0.2040 * y + 1.0570 * z,
    ];
    let mut out = [0u8; 3];
    for (o, c) in out.iter_mut().zip(lin) {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        *o = (s * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Encode sRGB as a Recommended Display CIELab Value.
pub fn rgb_to_cielab(rgb: [u8; 3]) -> [u16; 3] {
    let lin: Vec<f64> = rgb
        .iter()
        .map(|&c| {
            let c = c as f64 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let (r, g, b) = (lin[0], lin[1], lin[2]);
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f64| {
        if t > (6.0f64 / 29.0).powi(3) {
            t.cbrt()
        } else {
            t / (3.0 * (6.0f64 / 29.0).powi(2)) + 4.0 / 29.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let bb = 200.0 * (fy - fz);
    let q = |v: f64, lo: f64, hi: f64| {
        (((v - lo) / (hi - lo)) * 65535.0)
            .round()
            .clamp(0.0, 65535.0) as u16
    };
    [q(l, 0.0, 100.0), q(a, -128.0, 127.0), q(bb, -128.0, 127.0)]
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

fn u16s_of(obj: &InMemDicomObject, tag: dicom_core::Tag) -> Option<Vec<u16>> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_multi_int::<u16>().ok())
}

/// One frame's identity: which segment it belongs to and where it sits.
struct FrameRef {
    segment: i32,
    pos: Vec3,
    /// Distance along the stack normal, for grouping frames into slices.
    proj: f64,
}

/// Read one Segmentation Storage instance.
///
/// The masks come back on the lattice the frames describe, which is *not*
/// necessarily that of any loaded volume - [`SegSeries::rebind`] does that
/// step when the series is displayed.
pub fn load(path: &Path) -> Result<SegSeries> {
    let obj =
        dicom_object::open_file(path).with_context(|| format!("open SEG {}", path.display()))?;

    let rows = i32_of(&obj, tags::ROWS).unwrap_or(0).max(0) as usize;
    let cols = i32_of(&obj, tags::COLUMNS).unwrap_or(0).max(0) as usize;
    let n_frames = i32_of(&obj, tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1) as usize;
    if rows == 0 || cols == 0 {
        bail!("SEG {} has no Rows/Columns", path.display());
    }
    let bits = i32_of(&obj, tags::BITS_ALLOCATED).unwrap_or(1).max(1) as usize;
    let seg_type = str_of(&obj, tags::SEGMENTATION_TYPE).unwrap_or_default();
    let fractional = seg_type == "FRACTIONAL";
    let threshold = if fractional {
        (i32_of(&obj, tags::MAXIMUM_FRACTIONAL_VALUE)
            .unwrap_or(255)
            .max(1) as u32)
            .div_ceil(2)
    } else {
        1
    };

    let pixels = obj
        .element(tags::PIXEL_DATA)
        .ok()
        .and_then(|e| e.to_bytes().ok())
        .map(|b| b.into_owned())
        .with_context(|| {
            format!(
                "SEG {} has no readable native Pixel Data (compressed segmentations \
                 are not supported)",
                path.display()
            )
        })?;

    // ---- segments -------------------------------------------------------
    struct SegmentInfo {
        number: i32,
        label: String,
        color: [u8; 3],
    }
    let mut segments: Vec<SegmentInfo> = Vec::new();
    if let Some(items) = items_of(&obj, tags::SEGMENT_SEQUENCE) {
        for (idx, it) in items.iter().enumerate() {
            let number = i32_of(it, tags::SEGMENT_NUMBER).unwrap_or(idx as i32 + 1);
            let label = str_of(it, tags::SEGMENT_LABEL)
                .or_else(|| str_of(it, tags::SEGMENT_DESCRIPTION))
                .unwrap_or_else(|| format!("Segment {number}"));
            let color = u16s_of(it, tags::RECOMMENDED_DISPLAY_CIE_LAB_VALUE)
                .filter(|v| v.len() >= 3)
                .map(|v| cielab_to_rgb([v[0], v[1], v[2]]))
                .unwrap_or(SEG_PALETTE[idx % SEG_PALETTE.len()]);
            segments.push(SegmentInfo {
                number,
                label,
                color,
            });
        }
    }
    if segments.is_empty() {
        bail!("SEG {} has an empty Segment Sequence", path.display());
    }

    // ---- geometry, shared and per frame ---------------------------------
    let shared = items_of(&obj, tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE)
        .and_then(|i| i.first())
        .cloned();
    let group = |g: Option<&InMemDicomObject>, seq, tag| {
        g.and_then(|g| items_of(g, seq))
            .and_then(|i| i.first())
            .and_then(|i| f64s_of(i, tag))
    };
    let sh = shared.as_ref();

    let per_frame = items_of(&obj, tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE)
        .map(|i| i.to_vec())
        .unwrap_or_default();

    let orient = group(
        sh,
        tags::PLANE_ORIENTATION_SEQUENCE,
        tags::IMAGE_ORIENTATION_PATIENT,
    )
    .or_else(|| {
        per_frame.iter().find_map(|f| {
            group(
                Some(f),
                tags::PLANE_ORIENTATION_SEQUENCE,
                tags::IMAGE_ORIENTATION_PATIENT,
            )
        })
    })
    .filter(|v| v.len() >= 6)
    .unwrap_or_else(|| vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let row_dir = Vec3::from_slice(&orient[0..3]).normalized();
    let col_dir = Vec3::from_slice(&orient[3..6]).normalized();
    let normal = row_dir.cross(col_dir).normalized();

    let measures = |tag| {
        group(sh, tags::PIXEL_MEASURES_SEQUENCE, tag).or_else(|| {
            per_frame
                .iter()
                .find_map(|f| group(Some(f), tags::PIXEL_MEASURES_SEQUENCE, tag))
        })
    };
    let ps = measures(tags::PIXEL_SPACING)
        .filter(|v| v.len() >= 2)
        .unwrap_or_else(|| vec![1.0, 1.0]);
    let declared_thickness = measures(tags::SPACING_BETWEEN_SLICES)
        .or_else(|| measures(tags::SLICE_THICKNESS))
        .and_then(|v| v.first().copied())
        .filter(|v| *v > 1e-6);

    let mut frames: Vec<FrameRef> = Vec::with_capacity(n_frames);
    for (fi, f) in per_frame.iter().enumerate() {
        let segment = items_of(f, tags::SEGMENT_IDENTIFICATION_SEQUENCE)
            .and_then(|i| i.first())
            .and_then(|i| i32_of(i, tags::REFERENCED_SEGMENT_NUMBER))
            .unwrap_or_else(|| segments[fi.min(segments.len() - 1)].number);
        let pos = group(
            Some(f),
            tags::PLANE_POSITION_SEQUENCE,
            tags::IMAGE_POSITION_PATIENT,
        )
        .filter(|v| v.len() >= 3)
        .map(|v| Vec3::from_slice(&v))
        .unwrap_or_else(|| normal * (fi as f64 * declared_thickness.unwrap_or(1.0)));
        frames.push(FrameRef {
            segment,
            proj: pos.dot(normal),
            pos,
        });
    }
    if frames.is_empty() {
        bail!(
            "SEG {} has no Per-frame Functional Groups - its frames cannot be placed",
            path.display()
        );
    }

    // ---- slice lattice from the distinct frame positions ------------------
    let mut projs: Vec<f64> = frames.iter().map(|f| f.proj).collect();
    projs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Frames of different segments on one slice repeat its position exactly,
    // so the smallest non-zero gap *is* the slice spacing; deriving the
    // grouping tolerance from it beats trusting SliceThickness, which some
    // writers set to the segmented thickness rather than the frame pitch.
    let min_gap = projs
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 1e-3)
        .fold(f64::MAX, f64::min);
    let tol = if min_gap.is_finite() {
        min_gap * 0.4
    } else {
        declared_thickness.unwrap_or(1.0) * 0.25
    };
    let mut levels: Vec<f64> = Vec::new();
    for p in projs {
        if levels.last().map(|l| (p - l).abs() > tol).unwrap_or(true) {
            levels.push(p);
        }
    }
    let nz = levels.len();
    let slice_spacing = if nz > 1 {
        let mut d: Vec<f64> = levels.windows(2).map(|w| w[1] - w[0]).collect();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        d[d.len() / 2].max(1e-6)
    } else {
        declared_thickness.unwrap_or(1.0)
    };
    // The origin is the in-plane position of the frame lowest along the
    // normal - every frame of one lattice shares it up to the slice offset.
    let origin = frames
        .iter()
        .min_by(|a, b| {
            a.proj
                .partial_cmp(&b.proj)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|f| f.pos)
        .unwrap_or(Vec3::new(0.0, 0.0, 0.0));

    let grid = Grid {
        dims: [cols, rows, nz],
        // PixelSpacing is [between rows, between columns] = [along j, along i].
        spacing: [ps[1], ps[0], slice_spacing],
        origin,
        row_dir,
        col_dir,
        normal,
        frame_of_reference_uid: str_of(&obj, tags::FRAME_OF_REFERENCE_UID).unwrap_or_default(),
    };

    // ---- unpack the frames into one mask per segment ----------------------
    let plane = rows * cols;
    let bit_of = |frame: usize, idx: usize| -> bool {
        if bits == 1 {
            let bit = frame * plane + idx;
            pixels
                .get(bit >> 3)
                .map(|b| (b >> (bit & 7)) & 1 == 1)
                .unwrap_or(false)
        } else {
            let step = bits / 8;
            pixels
                .get((frame * plane + idx) * step)
                .map(|b| *b as u32 >= threshold)
                .unwrap_or(false)
        }
    };

    let mut segs = Vec::with_capacity(segments.len());
    for info in &segments {
        let mut mask = vec![0u8; cols * rows * nz];
        for (fi, f) in frames.iter().enumerate() {
            if f.segment != info.number || fi >= n_frames {
                continue;
            }
            let k = levels
                .iter()
                .position(|l| (f.proj - l).abs() <= tol)
                .unwrap_or(0);
            let base = k * plane;
            for idx in 0..plane {
                if bit_of(fi, idx) {
                    mask[base + idx] = 1;
                }
            }
        }
        segs.push(Segmentation::from_mask(
            info.label.clone(),
            info.color,
            grid.dims,
            mask,
        ));
    }

    let referenced_series_uid = items_of(&obj, tags::REFERENCED_SERIES_SEQUENCE)
        .and_then(|i| i.first())
        .and_then(|i| str_of(i, tags::SERIES_INSTANCE_UID))
        .unwrap_or_default();

    Ok(SegSeries {
        label: str_of(&obj, tags::SERIES_DESCRIPTION)
            .or_else(|| str_of(&obj, tags::CONTENT_LABEL))
            .or_else(|| str_of(&obj, tags::CONTENT_DESCRIPTION))
            .unwrap_or_else(|| "Segmentation".into()),
        sop_instance_uid: str_of(&obj, tags::SOP_INSTANCE_UID).unwrap_or_default(),
        study_uid: str_of(&obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        referenced_series_uid,
        file_name: path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        grid,
        segs,
    })
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Everything a SEG file needs from the export around it.
pub struct SegWriteCtx<'a> {
    pub study_uid: &'a str,
    pub for_uid: &'a str,
    pub date: &'a str,
    pub time: &'a str,
    pub series_number: i64,
    /// The image series the segmentation is filed against, and the SOP
    /// Instance UIDs of its slices (empty when they are not known).
    pub image_series_uid: &'a str,
    pub image_sop_uids: &'a [String],
    /// Patient / study / equipment attributes, as reviewed in the dialog.
    pub params: &'a ExportParams,
}

fn put_uss(o: &mut InMemDicomObject, tag: dicom_core::Tag, vals: &[u16]) {
    o.put(DataElement::new(
        tag,
        VR::US,
        PrimitiveValue::U16(C::from_vec(vals.to_vec())),
    ));
}

fn code_item(value: &str, scheme: &str, meaning: &str) -> InMemDicomObject {
    let mut o = InMemDicomObject::new_empty();
    put_str(&mut o, tags::CODE_VALUE, VR::SH, value);
    put_str(&mut o, tags::CODING_SCHEME_DESIGNATOR, VR::SH, scheme);
    put_str(&mut o, tags::CODE_MEANING, VR::LO, meaning);
    o
}

/// Build the Segmentation Storage object for one series.
///
/// Only the slices a segment actually occupies become frames, so the file
/// size follows the segmented anatomy rather than the image stack.
pub fn build(ser: &SegSeries, ctx: &SegWriteCtx) -> InMemDicomObject {
    let [nx, ny, nz] = ser.grid.dims;
    let plane = nx * ny;

    // ---- which (segment, slice) pairs carry anything ---------------------
    let mut frames: Vec<(usize, usize)> = Vec::new();
    for (si, seg) in ser.segs.iter().enumerate() {
        let (k0, k1) = match seg.bbox {
            Some((lo, hi)) => (lo[2], hi[2].min(nz.saturating_sub(1))),
            None => continue,
        };
        for k in k0..=k1 {
            let base = k * plane;
            if seg.mask[base..(base + plane).min(seg.mask.len())]
                .iter()
                .any(|v| *v != 0)
            {
                frames.push((si, k));
            }
        }
    }

    let mut o = InMemDicomObject::new_empty();
    put_str(&mut o, tags::SPECIFIC_CHARACTER_SET, VR::CS, "ISO_IR 100");
    ctx.params.write_common(&mut o);
    put_str(&mut o, tags::STUDY_INSTANCE_UID, VR::UI, ctx.study_uid);
    put_str(&mut o, tags::MODALITY, VR::CS, "SEG");
    put_str(&mut o, tags::SOP_CLASS_UID, VR::UI, SOP_SEG);
    put_str(&mut o, tags::SOP_INSTANCE_UID, VR::UI, new_uid());
    put_str(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, new_uid());
    put_is(&mut o, tags::SERIES_NUMBER, ctx.series_number);
    put_is(&mut o, tags::INSTANCE_NUMBER, 1);
    put_str(&mut o, tags::SERIES_DESCRIPTION, VR::LO, ser.label.clone());
    put_str(&mut o, tags::CONTENT_DATE, VR::DA, ctx.date);
    put_str(&mut o, tags::CONTENT_TIME, VR::TM, ctx.time);
    put_strs(
        &mut o,
        tags::IMAGE_TYPE,
        VR::CS,
        &["DERIVED".to_string(), "PRIMARY".to_string()],
    );
    put_str(&mut o, tags::FRAME_OF_REFERENCE_UID, VR::UI, ctx.for_uid);
    put_str(&mut o, tags::POSITION_REFERENCE_INDICATOR, VR::LO, "");
    put_str(&mut o, tags::CONTENT_LABEL, VR::CS, "SEGMENTATION");
    put_str(&mut o, tags::CONTENT_DESCRIPTION, VR::LO, ser.label.clone());
    put_str(&mut o, tags::CONTENT_CREATOR_NAME, VR::PN, "");
    put_str(&mut o, tags::LOSSY_IMAGE_COMPRESSION, VR::CS, "00");

    // ---- image pixel module ---------------------------------------------
    put_us(&mut o, tags::SAMPLES_PER_PIXEL, 1);
    put_str(
        &mut o,
        tags::PHOTOMETRIC_INTERPRETATION,
        VR::CS,
        "MONOCHROME2",
    );
    put_us(&mut o, tags::ROWS, ny as u16);
    put_us(&mut o, tags::COLUMNS, nx as u16);
    put_us(&mut o, tags::BITS_ALLOCATED, 1);
    put_us(&mut o, tags::BITS_STORED, 1);
    put_us(&mut o, tags::HIGH_BIT, 0);
    put_us(&mut o, tags::PIXEL_REPRESENTATION, 0);
    put_is(&mut o, tags::NUMBER_OF_FRAMES, frames.len() as i64);
    put_str(&mut o, tags::SEGMENTATION_TYPE, VR::CS, "BINARY");

    // ---- referenced image series ----------------------------------------
    if !ctx.image_series_uid.is_empty() {
        let mut rs = InMemDicomObject::new_empty();
        put_str(
            &mut rs,
            tags::SERIES_INSTANCE_UID,
            VR::UI,
            ctx.image_series_uid,
        );
        let refs: Vec<InMemDicomObject> = ctx
            .image_sop_uids
            .iter()
            .map(|uid| {
                let mut r = InMemDicomObject::new_empty();
                put_str(&mut r, tags::REFERENCED_SOP_CLASS_UID, VR::UI, SOP_CT);
                put_str(
                    &mut r,
                    tags::REFERENCED_SOP_INSTANCE_UID,
                    VR::UI,
                    uid.clone(),
                );
                r
            })
            .collect();
        if !refs.is_empty() {
            put_seq(&mut rs, tags::REFERENCED_INSTANCE_SEQUENCE, refs);
        }
        put_seq(&mut o, tags::REFERENCED_SERIES_SEQUENCE, vec![rs]);
    }

    // ---- dimension organization (segment, then slice) --------------------
    let dim_uid = new_uid();
    let mut dorg = InMemDicomObject::new_empty();
    put_str(
        &mut dorg,
        tags::DIMENSION_ORGANIZATION_UID,
        VR::UI,
        dim_uid.clone(),
    );
    put_seq(&mut o, tags::DIMENSION_ORGANIZATION_SEQUENCE, vec![dorg]);
    let dim_index = |pointer: dicom_core::Tag, group: dicom_core::Tag, label: &str| {
        let mut d = InMemDicomObject::new_empty();
        put_str(
            &mut d,
            tags::DIMENSION_ORGANIZATION_UID,
            VR::UI,
            dim_uid.clone(),
        );
        d.put(DataElement::new(
            tags::DIMENSION_INDEX_POINTER,
            VR::AT,
            PrimitiveValue::from(pointer),
        ));
        d.put(DataElement::new(
            tags::FUNCTIONAL_GROUP_POINTER,
            VR::AT,
            PrimitiveValue::from(group),
        ));
        put_str(&mut d, tags::DIMENSION_DESCRIPTION_LABEL, VR::LO, label);
        d
    };
    put_seq(
        &mut o,
        tags::DIMENSION_INDEX_SEQUENCE,
        vec![
            dim_index(
                tags::REFERENCED_SEGMENT_NUMBER,
                tags::SEGMENT_IDENTIFICATION_SEQUENCE,
                "Segment Number",
            ),
            dim_index(
                tags::IMAGE_POSITION_PATIENT,
                tags::PLANE_POSITION_SEQUENCE,
                "Image Position Patient",
            ),
        ],
    );

    // ---- segments --------------------------------------------------------
    let seg_items: Vec<InMemDicomObject> = ser
        .segs
        .iter()
        .enumerate()
        .map(|(si, seg)| {
            let mut s = InMemDicomObject::new_empty();
            put_us(&mut s, tags::SEGMENT_NUMBER, si as u16 + 1);
            put_str(&mut s, tags::SEGMENT_LABEL, VR::LO, seg.name.clone());
            put_str(
                &mut s,
                tags::SEGMENT_ALGORITHM_TYPE,
                VR::CS,
                "SEMIAUTOMATIC",
            );
            put_str(
                &mut s,
                tags::SEGMENT_ALGORITHM_NAME,
                VR::LO,
                "rust-dicom-station",
            );
            put_uss(
                &mut s,
                tags::RECOMMENDED_DISPLAY_CIE_LAB_VALUE,
                &rgb_to_cielab(seg.color),
            );
            put_seq(
                &mut s,
                tags::SEGMENTED_PROPERTY_CATEGORY_CODE_SEQUENCE,
                vec![code_item("123037004", "SCT", "Anatomical Structure")],
            );
            put_seq(
                &mut s,
                tags::SEGMENTED_PROPERTY_TYPE_CODE_SEQUENCE,
                vec![code_item("85756007", "SCT", "Tissue")],
            );
            s
        })
        .collect();
    put_seq(&mut o, tags::SEGMENT_SEQUENCE, seg_items);

    // ---- functional groups -----------------------------------------------
    let mut measures = InMemDicomObject::new_empty();
    put_ds(
        &mut measures,
        tags::PIXEL_SPACING,
        &[ser.grid.spacing[1], ser.grid.spacing[0]],
    );
    put_ds(&mut measures, tags::SLICE_THICKNESS, &[ser.grid.spacing[2]]);
    put_ds(
        &mut measures,
        tags::SPACING_BETWEEN_SLICES,
        &[ser.grid.spacing[2]],
    );
    let mut orient = InMemDicomObject::new_empty();
    put_ds(
        &mut orient,
        tags::IMAGE_ORIENTATION_PATIENT,
        &[
            ser.grid.row_dir.x,
            ser.grid.row_dir.y,
            ser.grid.row_dir.z,
            ser.grid.col_dir.x,
            ser.grid.col_dir.y,
            ser.grid.col_dir.z,
        ],
    );
    let mut shared = InMemDicomObject::new_empty();
    put_seq(&mut shared, tags::PIXEL_MEASURES_SEQUENCE, vec![measures]);
    put_seq(&mut shared, tags::PLANE_ORIENTATION_SEQUENCE, vec![orient]);
    put_seq(
        &mut o,
        tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE,
        vec![shared],
    );

    let per_frame: Vec<InMemDicomObject> = frames
        .iter()
        .map(|&(si, k)| {
            let mut f = InMemDicomObject::new_empty();
            let mut content = InMemDicomObject::new_empty();
            content.put(DataElement::new(
                tags::DIMENSION_INDEX_VALUES,
                VR::UL,
                PrimitiveValue::U32(C::from_vec(vec![si as u32 + 1, k as u32 + 1])),
            ));
            put_seq(&mut f, tags::FRAME_CONTENT_SEQUENCE, vec![content]);

            let p = ser.grid.voxel_to_patient(0.0, 0.0, k as f64);
            let mut pos = InMemDicomObject::new_empty();
            put_ds(&mut pos, tags::IMAGE_POSITION_PATIENT, &[p.x, p.y, p.z]);
            put_seq(&mut f, tags::PLANE_POSITION_SEQUENCE, vec![pos]);

            let mut id = InMemDicomObject::new_empty();
            put_us(&mut id, tags::REFERENCED_SEGMENT_NUMBER, si as u16 + 1);
            put_seq(&mut f, tags::SEGMENT_IDENTIFICATION_SEQUENCE, vec![id]);
            f
        })
        .collect();
    put_seq(
        &mut o,
        tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE,
        per_frame,
    );

    // ---- pixel data: one continuous bit stream over all frames ------------
    let total_bits = frames.len() * plane;
    let mut bytes = vec![0u8; total_bits.div_ceil(8)];
    for (fi, &(si, k)) in frames.iter().enumerate() {
        let src = &ser.segs[si].mask;
        let base = k * plane;
        let bit0 = fi * plane;
        for idx in 0..plane {
            if src.get(base + idx).copied().unwrap_or(0) != 0 {
                let bit = bit0 + idx;
                bytes[bit >> 3] |= 1 << (bit & 7);
            }
        }
    }
    // Pixel Data must be an even number of bytes.
    if bytes.len() % 2 == 1 {
        bytes.push(0);
    }
    o.put(DataElement::new(
        tags::PIXEL_DATA,
        VR::OB,
        PrimitiveValue::U8(C::from_vec(bytes)),
    ));
    o
}

/// Write one segmentation series as a DICOM SEG file.
pub fn write(ser: &SegSeries, ctx: &SegWriteCtx, path: &Path) -> Result<()> {
    write_object(build(ser, ctx), SOP_SEG, path)
}
