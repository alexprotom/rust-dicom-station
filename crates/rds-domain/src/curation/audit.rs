//! What is wrong with a folder of DICOM files, from the headers and the
//! element lengths alone (no pixel is decoded):
//!
//! * a file that does not read, or reads with a problem
//!   ([`super::raw::RawFile::problems`]: a length past its container, a
//!   transfer syntax that does not match the bytes);
//! * pixel data shorter than rows × columns × frames × samples × bits says,
//!   or cut off by the end of the file;
//! * along an image series: gaps (a step over 1.5 × the usual spacing),
//!   two slices at one position, more than one matrix size, more than one
//!   frame of reference, slices without an instance number;
//! * an RTSTRUCT: fewer StructureSetROISequence items than
//!   ROIContourSequence items, its frame of reference in (0008,1155),
//!   referenced series, images and frames that are neither in the folder
//!   nor in what the caller knows of the archive, contour planes that fall
//!   on no slice of the series it references;
//! * a registration whose frames or series lead nowhere;
//! * a structured report whose Modality is `DX` (the X-ray dose reports of
//!   one setup system), which the station routes by SOP class.
//!
//! Each finding has a severity: `broken` (the file cannot be used as it
//! is), `dangling-reference` (it is fine, but something it names is not
//! here), `warning`.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::Serialize;

use super::facts::{self, Facts};
use crate::progress::{Progress, CANCELLED};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Severity {
    Broken,
    DanglingReference,
    Warning,
}

#[derive(Clone, Debug, Serialize)]
pub struct Finding {
    pub severity: Severity,
    /// A short name of the check (`pixel-data-short`, `slice-gap`, ...).
    pub check: &'static str,
    /// What was found, in words: UIDs, counts, tags; never a patient's
    /// name or ID.
    pub detail: String,
    /// The series it is about, when it is about one.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub series: String,
    /// The files it is about (all of them, for a series finding: the caller
    /// shortens the list).
    pub files: Vec<PathBuf>,
}

/// What is known to exist outside the folder: the archive's SOP Instance
/// and Series Instance UIDs, and any frames of reference.
#[derive(Default)]
pub struct Known {
    pub sops: HashSet<String>,
    pub series: HashSet<String>,
    pub frames: HashSet<String>,
}

/// Findings by Study Instance UID (`""` for files that name none).
pub type Report = BTreeMap<String, Vec<Finding>>;

/// Read every file under `files` and check it.
/// `all_objects` checks the references of every non-image object (plans,
/// doses, RT images, records, segmentations) too, not only structure sets
/// and registrations: what `check_references` asks before an apply.
pub fn audit(
    files: &[PathBuf],
    known: &Known,
    all_objects: bool,
    p: &Progress,
) -> anyhow::Result<(Report, usize)> {
    let n = files.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let all: Vec<Facts> = files
        .par_iter()
        .map(|f| {
            let i = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i.is_multiple_of(256) {
                p.set(format!("Reading headers {i}/{n}"));
            }
            if p.cancelled() {
                return Facts::default();
            }
            facts::read(f, false)
        })
        .collect();
    if p.cancelled() {
        anyhow::bail!(CANCELLED);
    }
    Ok((check(&all, known, all_objects), all.len()))
}

/// The checks, on facts already read.
pub fn check(all: &[Facts], known: &Known, all_objects: bool) -> Report {
    let mut out: Report = BTreeMap::new();
    let mut add = |study: &str, f: Finding| out.entry(study.to_string()).or_default().push(f);
    // What the folder itself holds.
    let mut here = Known::default();
    for f in all {
        here.sops.insert(f.sop.clone());
        here.series.insert(f.series.clone());
        if !f.frame.is_empty() {
            here.frames.insert(f.frame.clone());
        }
    }
    let has = |set: fn(&Known) -> &HashSet<String>, uid: &str| {
        set(&here).contains(uid) || set(known).contains(uid)
    };
    let one = |sev, check, detail: String, f: &Facts| Finding {
        severity: sev,
        check,
        detail,
        series: f.series.clone(),
        files: vec![f.path.clone()],
    };

    // ---- per file --------------------------------------------------------
    for f in all {
        if let Some(why) = &f.unreadable {
            add("", one(Severity::Broken, "unreadable", why.clone(), f));
            continue;
        }
        for pr in &f.problems {
            add(&f.study, one(Severity::Broken, "parse", pr.clone(), f));
        }
        if f.pixel_truncated {
            add(
                &f.study,
                one(
                    Severity::Broken,
                    "pixel-data-cut-off",
                    "the pixel data runs past the end of the file".into(),
                    f,
                ),
            );
        } else if let (Some(d), Some(e)) = (f.pixel_declared, f.pixel_expected) {
            if d < e {
                add(
                    &f.study,
                    one(
                        Severity::Broken,
                        "pixel-data-short",
                        format!("{d} bytes of pixel data where the header asks for {e}"),
                        f,
                    ),
                );
            }
        }
        if crate::sop::is_structured_report(&f.sop_class) && f.modality.eq_ignore_ascii_case("DX") {
            add(
                &f.study,
                one(
                    Severity::Warning,
                    "sr-as-dx",
                    format!(
                        "a structured report ({}) whose Modality is DX",
                        crate::sop::name_or_uid(&f.sop_class)
                    ),
                    f,
                ),
            );
        }
        if f.sop_class == crate::sop::SOP_RTSTRUCT {
            if let Some((ss, rc)) = f.rois {
                if ss < rc {
                    add(
                        &f.study,
                        one(
                            Severity::Broken,
                            "rtstruct-roi-count",
                            format!("{ss} StructureSetROISequence items for {rc} ROIContourSequence items"),
                            f,
                        ),
                    );
                }
            }
            if f.frame_misplaced {
                add(
                    &f.study,
                    one(
                        Severity::Warning,
                        "rtstruct-frame-misplaced",
                        "the frame of reference sits in (0008,1155) of ReferencedFrameOfReferenceSequence".into(),
                        f,
                    ),
                );
            }
        }
        let rt_object = [
            crate::sop::SOP_RTPLAN,
            crate::sop::SOP_RTIONPLAN,
            crate::sop::SOP_RTIMAGE,
        ]
        .contains(&f.sop_class.as_str())
            || crate::sop::is_beams_record(&f.sop_class)
            || crate::sop::is_structured_report(&f.sop_class)
            || f.sop_class == crate::sop::SOP_RTDOSE
            || f.sop_class == crate::sop::SOP_SEG;
        if f.sop_class == crate::sop::SOP_RTSTRUCT
            || crate::sop::is_registration(&f.sop_class)
            || (all_objects && rt_object)
        {
            let series: Vec<&String> = f
                .ref_series
                .iter()
                .filter(|u| !has(|k| &k.series, u))
                .collect();
            let frames: Vec<&String> = f
                .ref_frames
                .iter()
                .filter(|u| !has(|k| &k.frames, u))
                .collect();
            // A structure set's images: counted, not listed (hundreds).
            let images = if f.sop_class == crate::sop::SOP_RTSTRUCT || all_objects {
                f.ref_sops.iter().filter(|u| !has(|k| &k.sops, u)).count()
            } else {
                0
            };
            if !series.is_empty() || !frames.is_empty() || images > 0 {
                let mut parts = Vec::new();
                if !series.is_empty() {
                    parts.push(format!("series {}", join(&series)));
                }
                if images > 0 {
                    parts.push(format!(
                        "{images} of {} referenced instances",
                        f.ref_sops.len()
                    ));
                }
                if !frames.is_empty() {
                    parts.push(format!("frame of reference {}", join(&frames)));
                }
                add(
                    &f.study,
                    one(
                        Severity::DanglingReference,
                        if f.sop_class == crate::sop::SOP_RTSTRUCT {
                            "rtstruct-reference"
                        } else if crate::sop::is_registration(&f.sop_class) {
                            "reg-reference"
                        } else {
                            "object-reference"
                        },
                        format!("not in the folder or the archive: {}", parts.join("; ")),
                        f,
                    ),
                );
            }
        }
    }

    // ---- per image series -----------------------------------------------
    let mut by_series: BTreeMap<&str, Vec<&Facts>> = BTreeMap::new();
    for f in all
        .iter()
        .filter(|f| f.unreadable.is_none() && f.position.is_some())
    {
        by_series.entry(f.series.as_str()).or_default().push(f);
    }
    // The slice positions of each series, for the contour check below.
    let mut slice_z: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    for (uid, files) in &by_series {
        let study = files[0].study.clone();
        let paths = || files.iter().map(|f| f.path.clone()).collect::<Vec<_>>();
        let finding = |sev, check, detail: String| Finding {
            severity: sev,
            check,
            detail,
            series: uid.to_string(),
            files: paths(),
        };
        let frames: BTreeSet<&str> = files
            .iter()
            .map(|f| f.frame.as_str())
            .filter(|s| !s.is_empty())
            .collect();
        if frames.len() > 1 {
            add(
                &study,
                finding(
                    Severity::Warning,
                    "series-frames",
                    format!("{} frames of reference in one series", frames.len()),
                ),
            );
        }
        let sizes: BTreeSet<(u32, u32)> = files.iter().map(|f| (f.rows, f.cols)).collect();
        if sizes.len() > 1 {
            add(
                &study,
                finding(
                    Severity::Warning,
                    "series-matrix",
                    format!("{} matrix sizes in one series: {sizes:?}", sizes.len()),
                ),
            );
        }
        let missing = files.iter().filter(|f| f.instance.is_none()).count();
        if missing > 0 {
            add(
                &study,
                finding(
                    Severity::Warning,
                    "instance-number",
                    format!(
                        "{missing} of {} slices have no instance number",
                        files.len()
                    ),
                ),
            );
        }
        // Positions along the normal (single-frame slices only).
        let mut z: Vec<f64> = files
            .iter()
            .filter(|f| f.frames <= 1)
            .filter_map(|f| f.along_normal())
            .collect();
        z.sort_by(f64::total_cmp);
        slice_z.insert(
            uid,
            files
                .iter()
                .filter_map(|f| f.position.map(|p| (p[2] * 100.0).round() / 100.0))
                .collect(),
        );
        if z.len() < 3 {
            continue;
        }
        let steps: Vec<f64> = z.windows(2).map(|w| w[1] - w[0]).collect();
        let doubles = steps.iter().filter(|s| s.abs() < 0.01).count();
        if doubles > 0 {
            add(
                &study,
                finding(
                    Severity::Warning,
                    "slice-duplicate",
                    format!("{doubles} slice positions occur twice"),
                ),
            );
        }
        let mut real: Vec<f64> = steps.iter().copied().filter(|s| s.abs() >= 0.01).collect();
        real.sort_by(f64::total_cmp);
        if let Some(&usual) = real.get(real.len() / 2) {
            let gaps: Vec<f64> = real.iter().copied().filter(|s| *s > usual * 1.5).collect();
            if !gaps.is_empty() {
                let widest = gaps.iter().copied().fold(0.0, f64::max);
                add(
                    &study,
                    finding(
                        Severity::Warning,
                        "slice-gap",
                        format!("{} gaps along the series (spacing {usual:.2} mm, widest {widest:.2} mm)", gaps.len()),
                    ),
                );
            }
        }
    }

    // ---- contours against the slices of their series ---------------------
    for f in all
        .iter()
        .filter(|f| f.sop_class == crate::sop::SOP_RTSTRUCT)
    {
        let Some(series) = f.ref_series.iter().find_map(|s| slice_z.get(s.as_str())) else {
            continue;
        };
        let planes: BTreeSet<i64> = f
            .contour_z
            .iter()
            .map(|z| (z * 100.0).round() as i64)
            .collect();
        let slices: BTreeSet<i64> = series.iter().map(|z| (z * 100.0).round() as i64).collect();
        let off = planes
            .iter()
            .filter(|z| !slices.iter().any(|s| (*s - **z).abs() <= 2))
            .count();
        if off > 0 {
            add(
                &f.study,
                one(
                    Severity::Warning,
                    "contour-planes",
                    format!(
                        "{off} of {} contour planes fall on no slice of the referenced series",
                        planes.len()
                    ),
                    f,
                ),
            );
        }
    }
    for v in out.values_mut() {
        v.sort_by(|a, b| (a.severity, a.check, &a.series).cmp(&(b.severity, b.check, &b.series)));
    }
    out
}

fn join(v: &[&String]) -> String {
    let shown: Vec<&str> = v.iter().take(4).map(|s| s.as_str()).collect();
    if v.len() > 4 {
        format!("{} and {} more", shown.join(", "), v.len() - 4)
    } else {
        shown.join(", ")
    }
}

/// Whether `p` lies under `dir`.
pub fn under(p: &Path, dir: &Path) -> bool {
    p.starts_with(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ct(series: &str, z: f64, n: Option<i64>) -> Facts {
        Facts {
            path: PathBuf::from(format!("{series}-{z}.dcm")),
            sop_class: crate::sop::SOP_CT.into(),
            sop: format!("{series}.{z}"),
            study: "st".into(),
            series: series.into(),
            frame: "for1".into(),
            instance: n,
            position: Some([0.0, 0.0, z]),
            orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            rows: 4,
            cols: 4,
            frames: 1,
            bits: 16,
            samples: 1,
            pixel_declared: Some(32),
            pixel_expected: Some(32),
            ..Default::default()
        }
    }

    #[test]
    fn gaps_doubles_short_pixels_and_dangling_references_are_found() {
        let mut all: Vec<Facts> = [0.0, 2.0, 4.0, 4.0, 6.0, 12.0, 14.0]
            .iter()
            .map(|z| ct("ct1", *z, Some(1)))
            .collect();
        all[1].pixel_declared = Some(16);
        all.push(Facts {
            path: "rs.dcm".into(),
            sop_class: crate::sop::SOP_RTSTRUCT.into(),
            sop: "rs".into(),
            study: "st".into(),
            series: "rs-series".into(),
            rois: Some((2, 3)),
            ref_series: vec!["ct1".into(), "planning-ct".into()],
            ref_sops: vec!["ct1.0".into(), "gone.1".into()],
            contour_z: vec![0.0, 2.0, 3.0],
            ..Default::default()
        });
        let r = check(&all, &Known::default(), false);
        let checks: Vec<&str> = r["st"].iter().map(|f| f.check).collect();
        for want in [
            "pixel-data-short",
            "rtstruct-roi-count",
            "rtstruct-reference",
            "slice-duplicate",
            "slice-gap",
            "contour-planes",
        ] {
            assert!(checks.contains(&want), "{want} in {checks:?}");
        }
        let dangling = r["st"]
            .iter()
            .find(|f| f.check == "rtstruct-reference")
            .unwrap();
        assert!(dangling.detail.contains("planning-ct") && dangling.detail.contains("1 of 2"));
        // Known to the archive: no longer dangling.
        let known = Known {
            series: ["planning-ct".to_string()].into(),
            sops: ["gone.1".to_string()].into(),
            ..Default::default()
        };
        assert!(!check(&all, &known, false)["st"]
            .iter()
            .any(|f| f.check == "rtstruct-reference"));
    }

    #[test]
    fn a_clean_series_has_nothing_to_say() {
        let all: Vec<Facts> = (0..10)
            .map(|i| ct("ct1", i as f64 * 3.0, Some(i)))
            .collect();
        assert!(check(&all, &Known::default(), false).is_empty());
    }
}
