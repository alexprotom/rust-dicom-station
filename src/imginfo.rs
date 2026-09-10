//! What the image itself is: geometry, sampling and acquisition, read back
//! out of the series' own headers.
//!
//! Everything downstream of loading - registration, resampling, DVH,
//! propagation, an ITV - assumes a regular lattice, and a physicist opening
//! two studies wants to know before starting whether they can be compared:
//! is the in-plane spacing the same, is the slice spacing the same as the
//! slice thickness (or is there a gap, or an overlap), is the spacing even
//! uniform, is the gantry tilted, do the two share a frame of reference.
//!
//! The reconstructed [`Volume`] answers the first half of that: it is what
//! the viewer actually shows, on a lattice the loader already regularized.
//! The second half - thickness, gaps, tilt, kV, kernel - only exists in the
//! headers, so this module reads them again. It reads headers only (never
//! pixels), in parallel, and the caller caches the result.

use std::path::Path;

use dicom_dictionary_std::tags;
use rayon::prelude::*;

use crate::geometry::Vec3;
use crate::loader::{f64_of, f64s_of, str_of, SeriesInfo};
use crate::volume::Volume;

/// One `label: value` line of the report, with an optional note that says
/// why the value deserves attention.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub label: String,
    pub value: String,
    /// Set when the value is one to look twice at (a gap between slices, a
    /// tilted gantry, non-uniform spacing). Shown as a warning.
    pub note: Option<String>,
}

impl Row {
    fn new(label: impl Into<String>, value: impl Into<String>) -> Row {
        Row {
            label: label.into(),
            value: value.into(),
            note: None,
        }
    }

    fn warn(mut self, note: impl Into<String>) -> Row {
        self.note = Some(note.into());
        self
    }
}

/// Everything the module reports about one image series, in sections.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImageInfo {
    /// The series this describes, as the tree names it.
    pub title: String,
    /// `(section name, rows)`, in display order.
    pub sections: Vec<(String, Vec<Row>)>,
    /// How many slice headers were read, and how many failed.
    pub read: usize,
    pub failed: usize,
}

impl ImageInfo {
    /// Every row of every section that carries a note: the short list a
    /// physicist has to act on.
    pub fn warnings(&self) -> Vec<&Row> {
        self.sections
            .iter()
            .flat_map(|(_, rows)| rows.iter())
            .filter(|r| r.note.is_some())
            .collect()
    }

    /// The whole report as plain text, for the clipboard.
    pub fn text(&self) -> String {
        let mut s = format!("{}\n", self.title);
        for (name, rows) in &self.sections {
            s.push_str(&format!("\n[{name}]\n"));
            for r in rows {
                s.push_str(&format!("{}: {}", r.label, r.value));
                if let Some(n) = &r.note {
                    s.push_str(&format!("   ! {n}"));
                }
                s.push('\n');
            }
        }
        s
    }
}

/// What one slice header contributes to the report.
#[derive(Clone, Debug, Default)]
struct SliceHead {
    proj: Option<f64>,
    thickness: Option<f64>,
    spacing_between: Option<f64>,
    pixel_spacing: Option<[f64; 2]>,
    tilt: Option<f64>,
    kvp: Option<f64>,
    exposure: Option<f64>,
    current: Option<f64>,
    ctdi: Option<f64>,
}

/// Format a millimetre value the way a physicist writes it: three decimals
/// while it matters, no trailing noise.
fn mm(v: f64) -> String {
    format!("{v:.3}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

/// A unit direction as the three cosines a DICOM header writes.
fn dir(v: Vec3) -> String {
    format!("({:.3}, {:.3}, {:.3})", v.x, v.y, v.z)
}

fn one_or_range(vals: &[f64], unit: &str) -> Option<(String, bool)> {
    let mut v: Vec<f64> = vals.to_vec();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let (lo, hi) = (v[0], v[v.len() - 1]);
    if (hi - lo).abs() < 1e-6 {
        Some((format!("{}{unit}", mm(lo)), false))
    } else {
        Some((format!("{} to {}{unit}", mm(lo), mm(hi)), true))
    }
}

/// Read one slice header. Failures are counted, never fatal: a series with
/// one unreadable file still deserves a report of the rest.
fn read_head(path: &Path, normal: Vec3) -> Option<SliceHead> {
    let obj = crate::dicomfile::open_header(path).ok()?;
    let ps = f64s_of(&obj, tags::PIXEL_SPACING)
        .filter(|v| v.len() >= 2)
        .map(|v| [v[1], v[0]]);
    Some(SliceHead {
        proj: f64s_of(&obj, tags::IMAGE_POSITION_PATIENT)
            .filter(|v| v.len() >= 3)
            .map(|v| Vec3::from_slice(&v).dot(normal)),
        thickness: f64_of(&obj, tags::SLICE_THICKNESS),
        spacing_between: f64_of(&obj, tags::SPACING_BETWEEN_SLICES),
        pixel_spacing: ps,
        tilt: f64_of(&obj, tags::GANTRY_DETECTOR_TILT),
        kvp: f64_of(&obj, tags::KVP),
        exposure: f64_of(&obj, tags::EXPOSURE),
        current: f64_of(&obj, tags::X_RAY_TUBE_CURRENT),
        ctdi: f64_of(&obj, tags::CTD_IVOL),
    })
}

/// The report for one series and the volume reconstructed from it.
///
/// `vol` is the lattice the viewer draws and every tool computes on; the
/// headers say what the scanner actually wrote, and the two disagreeing is
/// exactly what this is for.
pub fn describe(series: &SeriesInfo, vol: &Volume) -> ImageInfo {
    let normal = vol.normal;
    let heads: Vec<Option<SliceHead>> = series
        .files
        .par_iter()
        .map(|p| read_head(p, normal))
        .collect();
    let failed = heads.iter().filter(|h| h.is_none()).count();
    let heads: Vec<SliceHead> = heads.into_iter().flatten().collect();
    let read = heads.len();

    // One header carries the tags that do not vary per slice.
    let first = series.files.first().and_then(|p| {
        crate::dicomfile::open_header(p)
            .ok()
            .map(|o| (o, p.clone()))
    });
    let tag =
        |t: dicom_core::Tag| -> Option<String> { first.as_ref().and_then(|(o, _)| str_of(o, t)) };

    let mut sections: Vec<(String, Vec<Row>)> = Vec::new();

    // -- identity ---------------------------------------------------------
    let mut id = vec![
        Row::new(
            "Patient",
            if series.patient_id.is_empty() {
                series.patient_name.clone()
            } else {
                format!("{} ({})", series.patient_name, series.patient_id)
            },
        ),
        Row::new(
            "Study",
            if series.study_description.is_empty() {
                series.study_date.clone()
            } else {
                format!("{} · {}", series.study_description, series.study_date)
            },
        ),
        Row::new("Modality", &series.modality),
    ];
    if let Some(n) = series.series_number {
        id.push(Row::new("Series number", n.to_string()));
    }
    if let Some(v) = tag(tags::BODY_PART_EXAMINED) {
        id.push(Row::new("Body part", v));
    }
    if let Some(v) = tag(tags::PROTOCOL_NAME) {
        id.push(Row::new("Protocol", v));
    }
    if let Some(v) = tag(tags::PATIENT_POSITION) {
        id.push(Row::new("Patient position", v));
    }
    let acq = [
        tag(tags::ACQUISITION_DATE).or_else(|| tag(tags::SERIES_DATE)),
        tag(tags::ACQUISITION_TIME).or_else(|| tag(tags::SERIES_TIME)),
    ];
    if acq.iter().any(|v| v.is_some()) {
        id.push(Row::new(
            "Acquired",
            acq.iter()
                .filter_map(|v| v.clone())
                .collect::<Vec<_>>()
                .join(" "),
        ));
    }
    sections.push(("Series".to_string(), id));

    // -- sampling: the numbers the question is really about ---------------
    let [nx, ny, nz] = vol.dims;
    let [sx, sy, sz] = vol.spacing;
    let mut samp = vec![
        Row::new("Dimensions", format!("{nx} × {ny} × {nz} voxels")),
        {
            let r = Row::new(
                "Voxel spacing",
                format!("{} × {} × {} mm", mm(sx), mm(sy), mm(sz)),
            );
            if (sx - sy).abs() > 1e-3 {
                r.warn("The in-plane spacing is anisotropic")
            } else {
                r
            }
        },
        Row::new(
            "Field of view",
            format!(
                "{} × {} × {} mm",
                mm(nx as f64 * sx),
                mm(ny as f64 * sy),
                mm(nz as f64 * sz)
            ),
        ),
    ];

    // Slice count: files found, slices reconstructed, and whether the
    // loader had to drop any.
    let files = series.files.len();
    let r = Row::new(
        "Slices",
        if files == nz {
            format!("{nz}")
        } else {
            format!("{nz} of {files} files")
        },
    );
    samp.push(if files > nz {
        r.warn(format!(
            "{} file(s) of the series are not in the volume: duplicates at the same \
             position, a mismatched matrix, or a slice that would not decode",
            files - nz
        ))
    } else {
        r
    });

    // Slice thickness against slice spacing: gap, overlap, or neither.
    let thicknesses: Vec<f64> = heads.iter().filter_map(|h| h.thickness).collect();
    if let Some((text, varies)) = one_or_range(&thicknesses, " mm") {
        let r = Row::new("Slice thickness", text);
        samp.push(if varies {
            r.warn("The slices are not all the same thickness")
        } else {
            r
        });
    } else {
        samp.push(
            Row::new("Slice thickness", "not stated")
                .warn("The header carries no SliceThickness, so no gap or overlap can be checked"),
        );
    }
    if let Some(t) = thicknesses.first().copied() {
        let gap = sz - t;
        let r = Row::new(
            "Slice gap",
            if gap.abs() < 0.01 {
                "contiguous".to_string()
            } else if gap > 0.0 {
                format!("{} mm gap", mm(gap))
            } else {
                format!("{} mm overlap", mm(-gap))
            },
        );
        samp.push(if gap > 0.01 {
            r.warn(
                "The slices do not touch: everything between them is interpolated, and a \
                 structure's volume and a DVH are only as good as that",
            )
        } else if gap < -0.01 {
            r.warn("The slices overlap, so the reconstruction double-counts the overlap")
        } else {
            r
        });
    }

    // What the header claims the spacing is, when it says at all: a
    // SpacingBetweenSlices that disagrees with the positions is a sign the
    // series was assembled from more than one scan.
    let between: Vec<f64> = heads.iter().filter_map(|h| h.spacing_between).collect();
    if let Some(v) = between.first().copied() {
        let r = Row::new("Stated slice spacing", format!("{} mm", mm(v)));
        samp.push(if (v - sz).abs() > 0.01 {
            r.warn(format!(
                "The header says {} mm, the slice positions say {} mm",
                mm(v),
                mm(sz)
            ))
        } else {
            r
        });
    }

    // Spacing uniformity, measured from the positions themselves.
    let mut projs: Vec<f64> = heads.iter().filter_map(|h| h.proj).collect();
    projs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    projs.dedup_by(|a, b| (*a - *b).abs() < 0.01);
    if projs.len() > 2 {
        let diffs: Vec<f64> = projs.windows(2).map(|w| w[1] - w[0]).collect();
        let lo = diffs.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = diffs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let dev = (hi - lo).abs();
        let r = Row::new(
            "Slice positions",
            if dev < 0.01 {
                format!("even, every {} mm", mm(lo))
            } else {
                format!("{} to {} mm apart", mm(lo), mm(hi))
            },
        );
        samp.push(if dev >= 0.01 {
            r.warn(format!(
                "Uneven by {} mm. The volume is resampled onto one even lattice, so \
                 anything measured along the slice axis carries that error",
                mm(dev)
            ))
        } else {
            r
        });
    }

    // In-plane spacing straight from the headers, in case the slices
    // disagree with each other.
    let px: Vec<f64> = heads
        .iter()
        .filter_map(|h| h.pixel_spacing.map(|p| p[0]))
        .collect();
    let py: Vec<f64> = heads
        .iter()
        .filter_map(|h| h.pixel_spacing.map(|p| p[1]))
        .collect();
    if let (Some((a, va)), Some((b, vb))) = (one_or_range(&px, ""), one_or_range(&py, "")) {
        if va || vb {
            samp.push(
                Row::new("Header pixel spacing", format!("{a} × {b} mm"))
                    .warn("The slices do not all state the same pixel spacing"),
            );
        }
    }
    sections.push(("Sampling".to_string(), samp));

    // -- geometry ---------------------------------------------------------
    let mut geo = vec![
        Row::new(
            "Origin",
            format!(
                "({}, {}, {}) mm",
                mm(vol.origin.x),
                mm(vol.origin.y),
                mm(vol.origin.z)
            ),
        ),
        Row::new("Row direction", dir(vol.row_dir)),
        Row::new("Column direction", dir(vol.col_dir)),
        Row::new("Slice direction", dir(vol.normal)),
    ];
    // An axis-aligned volume is the ordinary case; anything else means
    // every reformat and every mask is resampled obliquely.
    let axis_aligned = [vol.row_dir, vol.col_dir, vol.normal].iter().all(|d| {
        let m = [d.x.abs(), d.y.abs(), d.z.abs()];
        m.iter().any(|v| (*v - 1.0).abs() < 1e-3)
    });
    if !axis_aligned {
        geo.push(
            Row::new("Axes", "oblique")
                .warn("The image axes are not the patient axes, so every reformat resamples"),
        );
    }
    let tilts: Vec<f64> = heads.iter().filter_map(|h| h.tilt).collect();
    if let Some(t) = tilts.iter().copied().find(|t| t.abs() > 0.01) {
        geo.push(
            Row::new("Gantry tilt", format!("{:.2}°", t))
                .warn("A tilted gantry shears the stack: check the reconstruction before using it"),
        );
    } else if !tilts.is_empty() {
        geo.push(Row::new("Gantry tilt", "0°"));
    }
    let r = Row::new(
        "Frame of reference",
        if vol.frame_of_reference_uid.is_empty() {
            "none".to_string()
        } else {
            vol.frame_of_reference_uid.clone()
        },
    );
    geo.push(if vol.frame_of_reference_uid.is_empty() {
        r.warn("Without one, nothing ties a structure set or a dose to this image")
    } else {
        r
    });
    geo.push(Row::new("Series UID", &series.uid));
    geo.push(Row::new("Study UID", &series.study_uid));
    sections.push(("Geometry".to_string(), geo));

    // -- acquisition ------------------------------------------------------
    let mut acq = Vec::new();
    let scanner = [tag(tags::MANUFACTURER), tag(tags::MANUFACTURER_MODEL_NAME)]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    if !scanner.is_empty() {
        acq.push(Row::new("Scanner", scanner));
    }
    if let Some(v) = tag(tags::STATION_NAME) {
        acq.push(Row::new("Station", v));
    }
    if let Some(v) = tag(tags::SOFTWARE_VERSIONS) {
        acq.push(Row::new("Software", v));
    }
    let kv: Vec<f64> = heads.iter().filter_map(|h| h.kvp).collect();
    if let Some((v, _)) = one_or_range(&kv, " kV") {
        acq.push(Row::new("Tube voltage", v));
    }
    let ma: Vec<f64> = heads.iter().filter_map(|h| h.current).collect();
    if let Some((v, _)) = one_or_range(&ma, " mA") {
        acq.push(Row::new("Tube current", v));
    }
    let ex: Vec<f64> = heads.iter().filter_map(|h| h.exposure).collect();
    if let Some((v, _)) = one_or_range(&ex, " mAs") {
        acq.push(Row::new("Exposure", v));
    }
    let ct: Vec<f64> = heads.iter().filter_map(|h| h.ctdi).collect();
    if let Some((v, _)) = one_or_range(&ct, " mGy") {
        acq.push(Row::new("CTDIvol", v));
    }
    if let Some(v) = tag(tags::CONVOLUTION_KERNEL) {
        acq.push(Row::new("Kernel", v));
    }
    if let Some(v) = tag(tags::SCAN_OPTIONS) {
        acq.push(Row::new("Scan options", v));
    }
    if let Some(v) = f64_of_first(&first, tags::RECONSTRUCTION_DIAMETER) {
        acq.push(Row::new("Recon diameter", format!("{} mm", mm(v))));
    }
    if !acq.is_empty() {
        sections.push(("Acquisition".to_string(), acq));
    }

    // -- pixels -----------------------------------------------------------
    let mut px_rows = vec![Row::new(
        "Value range",
        format!("{} to {}", vol.min_value, vol.max_value),
    )];
    let slope = f64_of_first(&first, tags::RESCALE_SLOPE);
    let inter = f64_of_first(&first, tags::RESCALE_INTERCEPT);
    if slope.is_some() || inter.is_some() {
        px_rows.push(Row::new(
            "Rescale",
            format!(
                "slope {} · intercept {}",
                mm(slope.unwrap_or(1.0)),
                mm(inter.unwrap_or(0.0))
            ),
        ));
    }
    if let Some(v) = tag(tags::RESCALE_TYPE) {
        px_rows.push(Row::new("Units", v));
    } else if series.modality == "CT" {
        px_rows.push(Row::new("Units", "HU (assumed)"));
    }
    if let Some(v) = tag(tags::PHOTOMETRIC_INTERPRETATION) {
        px_rows.push(Row::new("Photometric", v));
    }
    if let Some(v) = first
        .as_ref()
        .and_then(|(o, _)| crate::loader::i32_of(o, tags::BITS_STORED))
    {
        px_rows.push(Row::new("Bits stored", v.to_string()));
    }
    if let Some((_, p)) = &first {
        px_rows.push(Row::new("First file", p.display().to_string()));
    }
    sections.push(("Pixels".to_string(), px_rows));

    ImageInfo {
        title: if series.description.is_empty() {
            format!("{} ({} slices)", series.modality, nz)
        } else {
            format!(
                "{} · {} ({} slices)",
                series.modality, series.description, nz
            )
        },
        sections,
        read,
        failed,
    }
}

fn f64_of_first(
    first: &Option<(dicom_object::DefaultDicomObject, std::path::PathBuf)>,
    tag: dicom_core::Tag,
) -> Option<f64> {
    first.as_ref().and_then(|(o, _)| f64_of(o, tag))
}
