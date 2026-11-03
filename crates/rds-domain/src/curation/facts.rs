//! What the curation tools read off one file: its identity (SOP, series,
//! study, frame of reference), its image geometry, what its pixel data
//! should hold and does hold, a hash of those pixels, and the references
//! it makes - read from the raw elements ([`super::raw`]), with nothing
//! decoded but the few values asked for. The patient's name and ID are not
//! among them.

use std::path::{Path, PathBuf};

use dicom_core::Tag;
use dicom_dictionary_std::tags;
use sha2::{Digest, Sha256};

use super::journal::hex;
use super::raw::{RawFile, PIXEL_DATA, TS_IMPLICIT};

/// What a file is and refers to.
#[derive(Clone, Debug, Default)]
pub struct Facts {
    pub path: PathBuf,
    pub bytes: u64,
    /// The file did not read as DICOM at all, and why.
    pub unreadable: Option<String>,
    /// Parse problems of a file that did read ([`RawFile::problems`]).
    pub problems: Vec<String>,
    pub transfer_syntax: String,
    pub sop_class: String,
    pub sop: String,
    pub modality: String,
    pub study: String,
    pub series: String,
    pub series_number: String,
    pub frame: String,
    pub instance: Option<i64>,
    /// ImagePositionPatient, ImageOrientationPatient.
    pub position: Option<[f64; 3]>,
    pub orientation: Option<[f64; 6]>,
    /// PixelSpacing (row, column).
    pub spacing: Option<[f64; 2]>,
    pub rows: u32,
    pub cols: u32,
    pub frames: u32,
    pub samples: u32,
    pub bits: u32,
    /// The pixel data's length as written (`None`: encapsulated or none).
    pub pixel_declared: Option<u64>,
    /// What rows × columns × frames × samples × bits says it should be.
    pub pixel_expected: Option<u64>,
    /// The pixel data element runs past the end of the file.
    pub pixel_truncated: bool,
    /// SHA-256 of the pixel bytes (every fragment, for encapsulated data).
    pub pixel_hash: Option<String>,
    /// SHA-256 of the values the pixels stand for (stored value × slope +
    /// intercept, rounded: Hounsfield units for a CT) with the matrix size,
    /// for a native single-frame grayscale image: the same for a copy that
    /// re-encoded its pixels (another rescale, another pixel
    /// representation) as long as the values are the same.
    pub value_hash: Option<String>,
    /// Every Referenced SOP Instance UID, Referenced Series (Series
    /// Instance UID inside a reference sequence), and frame of reference
    /// named anywhere below the top level.
    pub ref_sops: Vec<String>,
    pub ref_series: Vec<String>,
    pub ref_frames: Vec<String>,
    /// An RTSTRUCT's counts: StructureSetROISequence items,
    /// ROIContourSequence items.
    pub rois: Option<(usize, usize)>,
    /// A structure set whose frame of reference sits in (0008,1155) of its
    /// ReferencedFrameOfReferenceSequence (RaySearch's unapproved sets).
    pub frame_misplaced: bool,
    /// The z of every contour point's first plane (RTSTRUCT), rounded to
    /// 0.01 mm.
    pub contour_z: Vec<f64>,
}

fn nums(s: &str) -> Vec<f64> {
    s.split('\\')
        .filter_map(|x| x.trim().parse().ok())
        .collect()
}

fn u16_value(f: &RawFile, tag: Tag) -> u32 {
    f.get(tag)
        .map(|e| e.value(&f.bytes))
        .filter(|v| v.len() >= 2)
        .map(|v| u16::from_le_bytes([v[0], v[1]]) as u32)
        .unwrap_or(0)
}

/// Read a file's facts. A file that is not DICOM has only `unreadable`.
/// `hash` computes the pixel hash (it reads every pixel byte, which the
/// other facts do not need).
pub fn read(path: &Path, hash: bool) -> Facts {
    let mut out = Facts {
        path: path.to_path_buf(),
        bytes: std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
        ..Default::default()
    };
    let f = match RawFile::read(path) {
        Ok(f) => f,
        Err(e) => {
            out.unreadable = Some(format!("{e:#}"));
            return out;
        }
    };
    if f.elements.is_empty() {
        let why = f
            .problems
            .first()
            .cloned()
            .unwrap_or_else(|| "no data set".into());
        // Without the header, nothing that parses: not DICOM at all.
        out.unreadable = Some(if f.part10 {
            why
        } else {
            format!("not DICOM: {why}")
        });
        return out;
    }
    let t = |tag| f.text(tag).unwrap_or_default();
    // Without the `DICM` header, bytes that happen to parse are DICOM only
    // when they name an instance (a screenshot's do not).
    if !f.is_dicom() {
        out.unreadable = Some("not DICOM: no file header and no SOP Instance UID".into());
        return out;
    }
    out.problems = f.problems.clone();
    out.transfer_syntax = f.transfer_syntax.clone();
    out.sop_class = t(tags::SOP_CLASS_UID);
    out.sop = t(tags::SOP_INSTANCE_UID);
    out.modality = t(tags::MODALITY);
    out.study = t(tags::STUDY_INSTANCE_UID);
    out.series = t(tags::SERIES_INSTANCE_UID);
    out.series_number = t(tags::SERIES_NUMBER);
    out.frame = t(tags::FRAME_OF_REFERENCE_UID);
    out.instance = t(tags::INSTANCE_NUMBER).trim().parse().ok();
    let ipp = nums(&t(tags::IMAGE_POSITION_PATIENT));
    if ipp.len() == 3 {
        out.position = Some([ipp[0], ipp[1], ipp[2]]);
    }
    let ps = nums(&t(tags::PIXEL_SPACING));
    if ps.len() == 2 {
        out.spacing = Some([ps[0], ps[1]]);
    }
    let iop = nums(&t(tags::IMAGE_ORIENTATION_PATIENT));
    if iop.len() == 6 {
        out.orientation = Some([iop[0], iop[1], iop[2], iop[3], iop[4], iop[5]]);
    }
    out.rows = u16_value(&f, tags::ROWS);
    out.cols = u16_value(&f, tags::COLUMNS);
    out.samples = u16_value(&f, tags::SAMPLES_PER_PIXEL).max(1);
    out.bits = u16_value(&f, tags::BITS_ALLOCATED);
    out.frames = t(tags::NUMBER_OF_FRAMES).trim().parse().unwrap_or(1).max(1);
    if let Some(px) = f.get(PIXEL_DATA) {
        out.pixel_truncated = px.truncated;
        let native = f.transfer_syntax == TS_IMPLICIT
            || f.transfer_syntax == super::raw::TS_EXPLICIT
            || f.transfer_syntax == super::raw::TS_BIG;
        if px.items.is_none() {
            out.pixel_declared = px.defined.map(u64::from);
        }
        if native && out.rows > 0 && out.cols > 0 && out.bits > 0 {
            out.pixel_expected = Some(
                out.rows as u64
                    * out.cols as u64
                    * out.frames as u64
                    * out.samples as u64
                    * out.bits as u64
                    / 8,
            );
        }
        if hash {
            let bytes = f.pixel_bytes();
            out.pixel_hash = bytes.as_ref().map(|b| hex(&Sha256::digest(b)));
            if native && px.items.is_none() {
                out.value_hash = bytes.and_then(|b| value_hash(&f, &out, &b));
            }
        }
    }
    // References, wherever they sit.
    for (path, e) in f.walk() {
        if path.is_empty() {
            continue;
        }
        let v = || e.text(&f.bytes);
        match e.tag {
            tags::REFERENCED_SOP_INSTANCE_UID => out.ref_sops.push(v()),
            tags::SERIES_INSTANCE_UID => out.ref_series.push(v()),
            tags::FRAME_OF_REFERENCE_UID | tags::REFERENCED_FRAME_OF_REFERENCE_UID => {
                out.ref_frames.push(v())
            }
            tags::CONTOUR_DATA => {
                let p = nums(&v());
                if p.len() >= 3 {
                    out.contour_z.push((p[2] * 100.0).round() / 100.0);
                }
            }
            _ => {}
        }
    }
    if out.sop_class == crate::sop::SOP_RTSTRUCT {
        let count = |tag| {
            f.get(tag)
                .and_then(|e| e.items.as_ref())
                .map_or(0, Vec::len)
        };
        out.rois = Some((
            count(tags::STRUCTURE_SET_ROI_SEQUENCE),
            count(tags::ROI_CONTOUR_SEQUENCE),
        ));
        // The frame's place: a FrameOfReferenceUID in the first item of
        // ReferencedFrameOfReferenceSequence, or (misplaced) a
        // ReferencedSOPInstanceUID there with no FrameOfReferenceUID.
        if let Some(items) = f
            .get(tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE)
            .and_then(|e| e.items.as_ref())
        {
            if let Some(first) = items.first() {
                let has = |tag| first.elements.iter().any(|e| e.tag == tag);
                if !has(tags::FRAME_OF_REFERENCE_UID) && has(tags::REFERENCED_SOP_INSTANCE_UID) {
                    out.frame_misplaced = true;
                    if let Some(e) = first
                        .elements
                        .iter()
                        .find(|e| e.tag == tags::REFERENCED_SOP_INSTANCE_UID)
                    {
                        let uid = e.text(&f.bytes);
                        out.ref_sops.retain(|s| *s != uid);
                        out.ref_frames.push(uid);
                    }
                }
            }
        }
    }
    for v in [&mut out.ref_sops, &mut out.ref_series, &mut out.ref_frames] {
        v.retain(|s| !s.is_empty());
        v.sort();
        v.dedup();
    }
    // An object's own series and frame are not references.
    let (own_series, own_frame) = (out.series.clone(), out.frame.clone());
    out.ref_series.retain(|s| *s != own_series);
    out.ref_frames.retain(|s| *s != own_frame);
    out
}

/// [`Facts::value_hash`] of a native image's pixel bytes.
fn value_hash(f: &RawFile, x: &Facts, bytes: &[u8]) -> Option<String> {
    if x.samples != 1 || x.frames != 1 || !matches!(x.bits, 8 | 16) {
        return None;
    }
    let n = x.rows as usize * x.cols as usize;
    let width = x.bits as usize / 8;
    if n == 0 || bytes.len() < n * width {
        return None;
    }
    let num = |tag, default: f64| {
        f.text(tag)
            .and_then(|t| t.split('\\').next().and_then(|v| v.trim().parse().ok()))
            .unwrap_or(default)
    };
    let slope = num(tags::RESCALE_SLOPE, 1.0);
    let intercept = num(tags::RESCALE_INTERCEPT, 0.0);
    let stored = match u16_value(f, tags::BITS_STORED) {
        0 => x.bits,
        b => b.min(x.bits),
    };
    let signed = u16_value(f, tags::PIXEL_REPRESENTATION) == 1;
    let big = f.transfer_syntax == super::raw::TS_BIG;
    let mut h = Sha256::new();
    h.update(x.rows.to_le_bytes());
    h.update(x.cols.to_le_bytes());
    let mask: u32 = if stored >= 32 {
        u32::MAX
    } else {
        (1u32 << stored) - 1
    };
    for i in 0..n {
        let raw: u32 = if width == 1 {
            bytes[i] as u32
        } else {
            let (a, b) = (bytes[2 * i], bytes[2 * i + 1]);
            u16::from_le_bytes(if big { [b, a] } else { [a, b] }) as u32
        };
        let v = raw & mask;
        let v: i64 = if signed && stored > 0 && v & (1 << (stored - 1)) != 0 {
            v as i64 - (1i64 << stored)
        } else {
            v as i64
        };
        let value = (v as f64 * slope + intercept).round() as i64;
        h.update((value.clamp(i32::MIN as i64, i32::MAX as i64) as i32).to_le_bytes());
    }
    Some(hex(&h.finalize()))
}

impl Facts {
    /// Whether the file is an image with pixels of the patient worth
    /// comparing by content: not a secondary capture or a placeholder
    /// (Philips' 8 × 8 respiratory-wave captures are the same in every
    /// series).
    pub fn twin_candidate(&self) -> bool {
        self.pixel_hash.is_some()
            && !crate::sop::is_secondary_capture(&self.sop_class)
            && (self.rows as u64 * self.cols as u64) > 64
    }

    /// The position along the slice normal, for an image with geometry.
    pub fn along_normal(&self) -> Option<f64> {
        let (p, o) = (self.position?, self.orientation?);
        let n = [
            o[1] * o[5] - o[2] * o[4],
            o[2] * o[3] - o[0] * o[5],
            o[0] * o[4] - o[1] * o[3],
        ];
        Some(p[0] * n[0] + p[1] * n[1] + p[2] * n[2])
    }
}
