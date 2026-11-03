//! Re-pointing a structure set at a CT series the archive already holds.
//!
//! The case: an export carries a structure set drawn on its own copy of a
//! planning CT, and that CT is a copy of one in the archive under other
//! UIDs (re-anonymized), or is missing altogether. The structure set is
//! made to refer to the archive's series instead: every image it names is
//! mapped to one of the target series' images, and its study, series and
//! frame-of-reference references follow. Two ways to map an image:
//!
//! * **pixel** - the image the structure set names is in the export, and
//!   the target series has its twin (the same pixel bytes at the same
//!   ImagePositionPatient, [`super::dupes`]);
//! * **z** - the target slice at the same position along the slice normal,
//!   after checking that the two series lie the same way (orientation, the
//!   in-plane origin and pixel spacing). When the named image is not in the
//!   export at all, the z of the contours drawn on it is used; an image
//!   no contour lies on (the series-level list names every slice, contoured
//!   or not) takes the z its place in that list gives - the list's
//!   contoured entries must lie on one straight line of index against z,
//!   or nothing is planned.
//!
//! An anonymizer that gave the images new UIDs and left the RT objects
//! pointing at the old ones leaves their plan and dose pointing at the old
//! structure set and plan too. `companions` (the export's RTPLAN and
//! RTDOSE files) are rewritten with the structure set: a plan's structure
//! set reference goes to this structure set, a dose's plan reference to
//! the one plan among them, and their study and frame of reference follow
//! the target's. ROIs whose frame of reference was left empty get the
//! target's.
//!
//! With `merge`, series of the export that shared the structure set's frame
//! of reference (the phases of a 4DCT) move with it into the target study
//! and frame - but only under the pixel mapping, where at least one series
//! of that frame is a verified twin of the target: a frame of reference is
//! never given to a series because of a merge alone.
//!
//! The result is a set of [`UidMaps`] and the files to rewrite with them;
//! [`super::remap::plan_rewrites`] runs the rewrites dry and refuses
//! if any old UID stays or any reference leads nowhere.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use dicom_dictionary_std::tags;

use super::facts::{self, Facts};
use super::raw::RawFile;
use super::remap::{contour_images, kind_of, UidKind, UidMaps};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mapping {
    Pixel,
    Z,
}

/// What a relink would do.
#[derive(Clone, Debug, Default)]
pub struct Relink {
    pub maps: UidMaps,
    /// The structure set, then the files of the merged series.
    pub files: Vec<PathBuf>,
    /// Images mapped, and how.
    pub by_pixels: usize,
    pub by_z: usize,
    pub by_contour_z: usize,
    /// Images no contour lies on, placed by the series-level list.
    pub by_list: usize,
    /// ROIs whose empty ReferencedFrameOfReferenceUID is filled.
    pub rois_framed: usize,
    /// Companions re-pointed: plans to the structure set, doses to the plan.
    pub plans_repointed: usize,
    pub doses_repointed: usize,
    /// How many of `files` are those plans and doses.
    pub companions: usize,
    /// Elements to set in every rewritten file ([`super::remap::Changes::set`]).
    pub set: Vec<(String, String)>,
    pub warnings: Vec<String>,
}

/// How far a list entry's z may lie off the line its neighbours give (mm).
const LIST_MM: f64 = 0.05;

/// The images the series-level list of a structure set names, in order:
/// ReferencedFrameOfReference ▶ RTReferencedStudy ▶ RTReferencedSeries ▶
/// ContourImage.
fn series_list(raw: &RawFile) -> Vec<String> {
    let want = [
        tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
        tags::RT_REFERENCED_STUDY_SEQUENCE,
        tags::RT_REFERENCED_SERIES_SEQUENCE,
        tags::CONTOUR_IMAGE_SEQUENCE,
    ];
    raw.walk()
        .into_iter()
        .filter(|(p, e)| *p == want && e.tag == tags::REFERENCED_SOP_INSTANCE_UID)
        .map(|(_, e)| e.text(&raw.bytes))
        .collect()
}

/// The Referenced SOP Instance UIDs directly in the items of a top-level
/// sequence.
fn referenced_in(raw: &RawFile, seq: dicom_core::Tag) -> Vec<String> {
    raw.walk()
        .into_iter()
        .filter(|(p, e)| *p == [seq] && e.tag == tags::REFERENCED_SOP_INSTANCE_UID)
        .map(|(_, e)| e.text(&raw.bytes))
        .filter(|v| !v.is_empty())
        .collect()
}

/// The z of every entry of `list`, from the entries whose z is known: the
/// least-squares line of index against z, when every known entry lies on
/// it. `None` with fewer than three known entries (two lie on a line
/// whatever the order), or when they do not.
fn list_z(list: &[String], known: &BTreeMap<String, f64>) -> Option<Vec<f64>> {
    let pts: Vec<(f64, f64)> = list
        .iter()
        .enumerate()
        .filter_map(|(i, s)| Some((i as f64, *known.get(s)?)))
        .collect();
    if pts.len() < 3 {
        return None;
    }
    let n = pts.len() as f64;
    let (mx, my) = (
        pts.iter().map(|p| p.0).sum::<f64>() / n,
        pts.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    if sxx == 0.0 {
        return None;
    }
    let b = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / sxx;
    let a = my - b * mx;
    if b.abs() < LIST_MM || pts.iter().any(|p| (a + b * p.0 - p.1).abs() > LIST_MM) {
        return None;
    }
    Some((0..list.len()).map(|i| a + b * i as f64).collect())
}

/// How near two positions must be to be one (mm).
const SAME_MM: f64 = 0.01;

fn pos_key(f: &Facts) -> Option<(String, String)> {
    let p = f.position?;
    Some((
        f.pixel_hash.clone()?,
        format!("{:.3}\\{:.3}\\{:.3}", p[0], p[1], p[2]),
    ))
}

/// Whether two images lie the same way: one orientation, one in-plane
/// origin (the position minus its component along the normal), one pixel
/// spacing.
fn same_plane(a: &Facts, b: &Facts) -> bool {
    let (Some(oa), Some(ob)) = (a.orientation, b.orientation) else {
        return false;
    };
    if oa.iter().zip(ob).any(|(x, y)| (x - y).abs() > 1e-4) {
        return false;
    }
    match (a.spacing, b.spacing) {
        (Some(sa), Some(sb)) if (sa[0] - sb[0]).abs() < 1e-4 && (sa[1] - sb[1]).abs() < 1e-4 => {}
        _ => return false,
    }
    let (Some(pa), Some(pb), Some(za), Some(zb)) =
        (a.position, b.position, a.along_normal(), b.along_normal())
    else {
        return false;
    };
    let n = [
        oa[1] * oa[5] - oa[2] * oa[4],
        oa[2] * oa[3] - oa[0] * oa[5],
        oa[0] * oa[4] - oa[1] * oa[3],
    ];
    (0..3).all(|i| ((pa[i] - za * n[i]) - (pb[i] - zb * n[i])).abs() < SAME_MM)
}

/// Plan a relink of the structure set `rs` onto the series `target`
/// (its files' facts, pixel hashes read for the pixel mapping).
/// `originals` are the files of the export the structure set came with,
/// with their hashes; `merge` the Series Instance UIDs among them to move
/// along; `companions` the export's plans and doses to re-point with it.
pub fn plan(
    rs: &Path,
    target: &[Facts],
    originals: &[Facts],
    mapping: Mapping,
    merge: &[String],
    companions: &[PathBuf],
) -> Result<Relink> {
    let raw = RawFile::read(rs)?;
    let me = facts::read(rs, false);
    if me.sop_class != crate::sop::SOP_RTSTRUCT {
        bail!("{} is not a structure set", rs.display());
    }
    let Some(first) = target.first() else {
        bail!("the target series has no files");
    };
    let series: BTreeSet<&str> = target.iter().map(|f| f.series.as_str()).collect();
    let studies: BTreeSet<&str> = target.iter().map(|f| f.study.as_str()).collect();
    let frames: BTreeSet<&str> = target.iter().map(|f| f.frame.as_str()).collect();
    if series.len() != 1 || studies.len() != 1 || frames.len() != 1 {
        bail!("the target files must be one series in one study and frame of reference");
    }
    if first.frame.is_empty() {
        bail!("the target series has no frame of reference");
    }
    let (t_series, t_study, t_frame) = (&first.series, &first.study, &first.frame);
    let target_sops: BTreeSet<&str> = target.iter().map(|f| f.sop.as_str()).collect();
    let by_pixels: HashMap<(String, String), &Facts> = target
        .iter()
        .filter_map(|f| Some((pos_key(f)?, f)))
        .collect();
    let originals_by_sop: HashMap<&str, &Facts> =
        originals.iter().map(|f| (f.sop.as_str(), f)).collect();

    // What the structure set names.
    let mut ref_sops: BTreeSet<String> = BTreeSet::new();
    let mut ref_series: BTreeSet<String> = BTreeSet::new();
    let mut ref_studies: BTreeSet<String> = BTreeSet::new();
    let mut ref_frames: BTreeSet<String> = BTreeSet::new();
    for (path, e) in raw.walk() {
        if e.items.is_some() {
            continue;
        }
        let v = e.text(&raw.bytes);
        if v.is_empty() {
            continue;
        }
        match (kind_of(&path, e.tag), e.tag) {
            (UidKind::Object, tags::REFERENCED_SOP_INSTANCE_UID) => {
                ref_sops.insert(v);
            }
            (UidKind::Object, tags::SERIES_INSTANCE_UID) => {
                ref_series.insert(v);
            }
            (UidKind::Study, _) => {
                ref_studies.insert(v);
            }
            (UidKind::Frame, _) => {
                ref_frames.insert(v);
            }
            _ => {}
        }
    }
    ref_studies.insert(me.study.clone());
    if !me.frame.is_empty() {
        ref_frames.insert(me.frame.clone());
    }
    let contour_z: BTreeMap<String, f64> = contour_images(&raw)
        .into_iter()
        .filter_map(|(sop, z)| Some((sop, z?)))
        .collect();

    // The z of the images named only in the series-level list, from the
    // place in it of those whose z is known (contours, or the export's
    // own image).
    let list = series_list(&raw);
    let mut known_z = contour_z.clone();
    for s in &list {
        if let Some(p) = originals_by_sop.get(s.as_str()).and_then(|o| o.position) {
            known_z.entry(s.clone()).or_insert(p[2]);
        }
    }
    let by_list: BTreeMap<&str, f64> = match (mapping, list_z(&list, &known_z)) {
        (Mapping::Z, Some(z)) => list.iter().map(String::as_str).zip(z).collect(),
        _ => BTreeMap::new(),
    };

    let mut out = Relink::default();
    let mut unresolved = Vec::new();
    let mut placed_by_list: BTreeSet<&str> = BTreeSet::new();
    // The series of the export whose images were mapped by their pixels.
    let mut twin_series: BTreeSet<String> = BTreeSet::new();
    for sop in &ref_sops {
        if target_sops.contains(sop.as_str()) {
            continue;
        }
        let orig = originals_by_sop.get(sop.as_str());
        let found: Option<&Facts> = match (mapping, orig) {
            (Mapping::Pixel, Some(o)) => pos_key(o).and_then(|k| by_pixels.get(&k).copied()),
            (Mapping::Z, Some(o)) => {
                let z = o.along_normal();
                target.iter().find(|t| {
                    same_plane(o, t)
                        && z.zip(t.along_normal())
                            .is_some_and(|(a, b)| (a - b).abs() < SAME_MM)
                })
            }
            (Mapping::Z, None) => match contour_z.get(sop) {
                // A patient-space z: the axial target slice at that z.
                Some(z) => target
                    .iter()
                    .find(|t| t.position.is_some_and(|p| (p[2] - z).abs() < SAME_MM)),
                // No contour on it: where the series-level list puts it.
                None => by_list.get(sop.as_str()).and_then(|z| {
                    let t = target
                        .iter()
                        .find(|t| t.position.is_some_and(|p| (p[2] - z).abs() < LIST_MM));
                    if t.is_some() {
                        placed_by_list.insert(sop.as_str());
                    }
                    t
                }),
            },
            (Mapping::Pixel, None) => None,
        };
        match found {
            Some(t) => {
                out.maps
                    .insert(UidKind::Object, sop, &t.sop)
                    .map_err(|_| anyhow::anyhow!("{sop} maps to two images"))?;
                match (mapping, orig) {
                    (Mapping::Pixel, Some(o)) => {
                        out.by_pixels += 1;
                        twin_series.insert(o.series.clone());
                    }
                    (_, Some(_)) => out.by_z += 1,
                    _ if placed_by_list.contains(sop.as_str()) => out.by_list += 1,
                    _ => out.by_contour_z += 1,
                }
            }
            None => unresolved.push(sop.clone()),
        }
    }
    if !unresolved.is_empty() {
        bail!(
            "{} of the {} images the structure set names have no counterpart in the target \
             series (the first: {}); nothing is planned",
            unresolved.len(),
            ref_sops.len(),
            unresolved[0]
        );
    }
    // The series it names: the image series of what was mapped, or - with
    // the CT missing - the one series it names.
    let mapped_series: BTreeSet<&str> = ref_sops
        .iter()
        .filter_map(|s| originals_by_sop.get(s.as_str()).map(|o| o.series.as_str()))
        .collect();
    let others: Vec<&String> = ref_series
        .iter()
        .filter(|s| *s != t_series && !mapped_series.contains(s.as_str()))
        .collect();
    for s in &mapped_series {
        let _ = out.maps.insert(UidKind::Object, s, t_series);
    }
    match others.as_slice() {
        [] => {}
        [one] if mapped_series.is_empty() => {
            let _ = out.maps.insert(UidKind::Object, one, t_series);
        }
        _ => bail!(
            "the structure set names {} series that are not the images it was mapped from \
             (the first: {}); nothing is planned",
            others.len(),
            others[0]
        ),
    }
    for s in &ref_studies {
        let _ = out.maps.insert(UidKind::Study, s, t_study);
    }
    for f in &ref_frames {
        let _ = out.maps.insert(UidKind::Frame, f, t_frame);
    }
    out.files.push(rs.to_path_buf());

    // ROIs with an empty frame of reference get the target's.
    let empty_frames = raw
        .walk()
        .into_iter()
        .filter(|(p, e)| {
            *p == [tags::STRUCTURE_SET_ROI_SEQUENCE]
                && e.tag == tags::REFERENCED_FRAME_OF_REFERENCE_UID
                && e.text(&raw.bytes).is_empty()
        })
        .count();
    if empty_frames > 0 {
        out.rois_framed = empty_frames;
        out.set.push(("30060024".to_string(), t_frame.to_string()));
    }

    // The export's plans and doses: the plan names this structure set, the
    // dose the plan, both in the target's study and frame.
    let comp: Vec<(PathBuf, RawFile, Facts)> = companions
        .iter()
        .filter_map(|p| Some((p.clone(), RawFile::read(p).ok()?, facts::read(p, false))))
        .filter(|(_, _, f)| {
            crate::sop::is_rt_plan(&f.sop_class) || f.sop_class == crate::sop::SOP_RTDOSE
        })
        .collect();
    let plans: Vec<&Facts> = comp
        .iter()
        .map(|c| &c.2)
        .filter(|f| crate::sop::is_rt_plan(&f.sop_class))
        .collect();
    for (path, craw, f) in &comp {
        if crate::sop::is_rt_plan(&f.sop_class) {
            for old in referenced_in(craw, tags::REFERENCED_STRUCTURE_SET_SEQUENCE) {
                if old != me.sop {
                    out.maps
                        .insert(UidKind::Object, &old, &me.sop)
                        .map_err(|_| anyhow::anyhow!("{old} is named as two structure sets"))?;
                    out.plans_repointed += 1;
                }
            }
        } else {
            for old in referenced_in(craw, tags::REFERENCED_RT_PLAN_SEQUENCE) {
                match plans.as_slice() {
                    [p] if old != p.sop => {
                        out.maps
                            .insert(UidKind::Object, &old, &p.sop)
                            .map_err(|_| anyhow::anyhow!("{old} is named as two plans"))?;
                        out.doses_repointed += 1;
                    }
                    [_] => {}
                    _ => out.warnings.push(format!(
                        "{}: its plan is left as it is ({} plans among the companions)",
                        path.display(),
                        plans.len()
                    )),
                }
            }
        }
        if !f.study.is_empty() {
            let _ = out.maps.insert(UidKind::Study, &f.study, t_study);
        }
        if !f.frame.is_empty() {
            let _ = out.maps.insert(UidKind::Frame, &f.frame, t_frame);
        }
        for fr in &f.ref_frames {
            let _ = out.maps.insert(UidKind::Frame, fr, t_frame);
        }
        out.files.push(path.clone());
        out.companions += 1;
    }

    // The series that move with it.
    for uid in merge {
        let files: Vec<&Facts> = originals.iter().filter(|f| f.series == *uid).collect();
        if files.is_empty() {
            bail!("series {uid} is not among the structure set's originals");
        }
        if mapping != Mapping::Pixel {
            bail!("series move into the target's frame of reference only under the pixel mapping");
        }
        if files.iter().any(|f| f.frame != me.frame) {
            bail!(
                "series {uid} is not in the structure set's frame of reference; \
                 it does not move with it"
            );
        }
        if twin_series.is_empty() {
            bail!("no series of that frame of reference is a verified twin of the target");
        }
        for f in &files {
            if out.maps.insert(UidKind::Study, &f.study, t_study).is_err() {
                bail!("series {uid} is in a study already mapped elsewhere");
            }
        }
        out.files.extend(files.iter().map(|f| f.path.clone()));
    }
    out.maps.resolve()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::raw::tests::{ex, item, part10};
    use super::super::raw::{encode_text, TS_EXPLICIT};
    use super::*;
    use dicom_core::{Tag, VR};

    fn ui(tag: Tag, v: &str) -> Vec<u8> {
        ex(tag, "UI", &encode_text(VR::UI, v))
    }

    /// A target slice at z.
    fn slice(sop: &str, z: f64) -> Facts {
        Facts {
            sop: sop.into(),
            series: "1.2.3".into(),
            study: "1.2".into(),
            frame: "1.2.9".into(),
            position: Some([0.0, 0.0, z]),
            orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            spacing: Some([1.0, 1.0]),
            ..Default::default()
        }
    }

    /// The 4DCT case: a structure set whose planning CT is in neither the
    /// export nor the archive, relinked by its contours' z.
    #[test]
    fn a_structure_set_without_its_ct_is_relinked_by_its_contours_z() {
        let mut d = Vec::new();
        d.extend(ui(tags::SOP_CLASS_UID, crate::sop::SOP_RTSTRUCT));
        d.extend(ui(tags::SOP_INSTANCE_UID, "9.9.1"));
        d.extend(ui(tags::STUDY_INSTANCE_UID, "9.0"));
        d.extend(ui(tags::SERIES_INSTANCE_UID, "9.9"));
        d.extend(ui(tags::FRAME_OF_REFERENCE_UID, "9.8"));
        let refd = |sop: &str| item(&ui(tags::REFERENCED_SOP_INSTANCE_UID, sop));
        let series_item = [
            ui(tags::SERIES_INSTANCE_UID, "9.5"),
            ex(
                tags::CONTOUR_IMAGE_SEQUENCE,
                "SQ",
                &[refd("9.5.1"), refd("9.5.2")].concat(),
            ),
        ]
        .concat();
        let study_item = [
            ui(tags::REFERENCED_SOP_INSTANCE_UID, "9.0"),
            ex(
                tags::RT_REFERENCED_SERIES_SEQUENCE,
                "SQ",
                &item(&series_item),
            ),
        ]
        .concat();
        let frame_item = [
            ui(tags::FRAME_OF_REFERENCE_UID, "9.8"),
            ex(tags::RT_REFERENCED_STUDY_SEQUENCE, "SQ", &item(&study_item)),
        ]
        .concat();
        d.extend(ex(
            tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
            "SQ",
            &item(&frame_item),
        ));
        let contour = |sop: &str, z: &str| {
            item(
                &[
                    ex(tags::CONTOUR_IMAGE_SEQUENCE, "SQ", &refd(sop)),
                    ex(
                        tags::CONTOUR_DATA,
                        "DS",
                        &encode_text(VR::DS, &format!("1\\2\\{z}\\3\\4\\{z}")),
                    ),
                ]
                .concat(),
            )
        };
        let roi = item(&ex(
            tags::CONTOUR_SEQUENCE,
            "SQ",
            &[contour("9.5.1", "5"), contour("9.5.2", "7.5")].concat(),
        ));
        d.extend(ex(tags::ROI_CONTOUR_SEQUENCE, "SQ", &roi));
        let dir = std::env::temp_dir().join("rds_relink_z");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("rs.dcm");
        std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
        let target = vec![
            slice("1.2.3.1", 2.5),
            slice("1.2.3.2", 5.0),
            slice("1.2.3.3", 7.5),
        ];
        let r = plan(&p, &target, &[], Mapping::Z, &[], &[]).unwrap();
        assert_eq!(r.by_contour_z, 2);
        assert_eq!(
            r.maps.objects.get("9.5.1").map(String::as_str),
            Some("1.2.3.2")
        );
        assert_eq!(
            r.maps.objects.get("9.5.2").map(String::as_str),
            Some("1.2.3.3")
        );
        assert_eq!(r.maps.objects.get("9.5").map(String::as_str), Some("1.2.3"));
        assert_eq!(r.maps.studies.get("9.0").map(String::as_str), Some("1.2"));
        assert_eq!(r.maps.frames.get("9.8").map(String::as_str), Some("1.2.9"));
        // Moving series along needs verified twins: not under z.
        assert!(plan(&p, &target, &[], Mapping::Z, &["9.7".into()], &[]).is_err());
        // A contour on no target slice refuses.
        let short = vec![slice("1.2.3.2", 5.0)];
        assert!(plan(&p, &short, &[], Mapping::Z, &[], &[]).is_err());
    }

    /// A structure set (and its plan and dose) an anonymizer left pointing
    /// at the images, structure set and plan of before: the series-level
    /// list names slices no contour lies on, one ROI's frame is empty.
    #[test]
    fn an_anonymizers_leftovers_are_relinked_whole() {
        let refd = |sop: &str| item(&ui(tags::REFERENCED_SOP_INSTANCE_UID, sop));
        let rs_file = |list: &[&str], name: &str| -> PathBuf {
            let mut d = Vec::new();
            d.extend(ui(tags::SOP_CLASS_UID, crate::sop::SOP_RTSTRUCT));
            d.extend(ui(tags::SOP_INSTANCE_UID, "9.9.1"));
            d.extend(ui(tags::STUDY_INSTANCE_UID, "9.0"));
            d.extend(ui(tags::SERIES_INSTANCE_UID, "9.9"));
            let images: Vec<u8> = list.iter().flat_map(|s| refd(s)).collect();
            let series_item = [
                ui(tags::SERIES_INSTANCE_UID, "9.5"),
                ex(tags::CONTOUR_IMAGE_SEQUENCE, "SQ", &images),
            ]
            .concat();
            let study_item = [
                ui(tags::REFERENCED_SOP_INSTANCE_UID, "9.0"),
                ex(
                    tags::RT_REFERENCED_SERIES_SEQUENCE,
                    "SQ",
                    &item(&series_item),
                ),
            ]
            .concat();
            let frame_item = [
                ui(tags::FRAME_OF_REFERENCE_UID, "9.8"),
                ex(tags::RT_REFERENCED_STUDY_SEQUENCE, "SQ", &item(&study_item)),
            ]
            .concat();
            d.extend(ex(
                tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE,
                "SQ",
                &item(&frame_item),
            ));
            // Two ROIs, the second with its frame left empty.
            let roi = |n: &str, frame: &str| {
                item(
                    &[
                        ex(tags::ROI_NUMBER, "IS", &encode_text(VR::IS, n)),
                        ui(tags::REFERENCED_FRAME_OF_REFERENCE_UID, frame),
                    ]
                    .concat(),
                )
            };
            d.extend(ex(
                tags::STRUCTURE_SET_ROI_SEQUENCE,
                "SQ",
                &[roi("1", "9.8"), roi("2", "")].concat(),
            ));
            let contour = |sop: &str, z: &str| {
                item(
                    &[
                        ex(tags::CONTOUR_IMAGE_SEQUENCE, "SQ", &refd(sop)),
                        ex(
                            tags::CONTOUR_DATA,
                            "DS",
                            &encode_text(VR::DS, &format!("1\\2\\{z}\\3\\4\\{z}")),
                        ),
                    ]
                    .concat(),
                )
            };
            let rois = item(&ex(
                tags::CONTOUR_SEQUENCE,
                "SQ",
                &[
                    contour("9.5.1", "5"),
                    contour("9.5.2", "7.5"),
                    contour("9.5.4", "10"),
                ]
                .concat(),
            ));
            d.extend(ex(tags::ROI_CONTOUR_SEQUENCE, "SQ", &rois));
            let dir = std::env::temp_dir().join("rds_relink_whole");
            std::fs::create_dir_all(&dir).unwrap();
            let p = dir.join(name);
            std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
            p
        };
        let dir = std::env::temp_dir().join("rds_relink_whole");
        let object = |name: &str, class: &str, sop: &str, seq: Tag, old: &str| -> PathBuf {
            let mut d = Vec::new();
            d.extend(ui(tags::SOP_CLASS_UID, class));
            d.extend(ui(tags::SOP_INSTANCE_UID, sop));
            d.extend(ui(tags::STUDY_INSTANCE_UID, "9.0"));
            d.extend(ui(tags::SERIES_INSTANCE_UID, &format!("{sop}.0")));
            d.extend(ui(tags::FRAME_OF_REFERENCE_UID, "9.8"));
            d.extend(ex(seq, "SQ", &refd(old)));
            let p = dir.join(name);
            std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
            p
        };
        // The list in slice order: two slices without a contour around the
        // two with one.
        let rs = rs_file(&["9.5.0", "9.5.1", "9.5.2", "9.5.4", "9.5.3"], "rs.dcm");
        let rp = object(
            "rp.dcm",
            crate::sop::SOP_RTIONPLAN,
            "9.7.1",
            tags::REFERENCED_STRUCTURE_SET_SEQUENCE,
            "8.8.1",
        );
        let rd = object(
            "rd.dcm",
            crate::sop::SOP_RTDOSE,
            "9.6.1",
            tags::REFERENCED_RT_PLAN_SEQUENCE,
            "8.7.1",
        );
        let target: Vec<Facts> = [2.5, 5.0, 7.5, 10.0, 12.5]
            .iter()
            .enumerate()
            .map(|(i, z)| slice(&format!("1.2.3.{}", i + 1), *z))
            .collect();
        let r = plan(
            &rs,
            &target,
            &[],
            Mapping::Z,
            &[],
            &[rp.clone(), rd.clone()],
        )
        .unwrap();
        assert_eq!((r.by_contour_z, r.by_list), (3, 2));
        let obj = |k: &str| r.maps.objects.get(k).map(String::as_str);
        assert_eq!(obj("9.5.0"), Some("1.2.3.1"));
        assert_eq!(obj("9.5.3"), Some("1.2.3.5"));
        assert_eq!(
            obj("8.8.1"),
            Some("9.9.1"),
            "the plan names this structure set"
        );
        assert_eq!(obj("8.7.1"), Some("9.7.1"), "the dose names the plan");
        assert_eq!(
            (r.plans_repointed, r.doses_repointed, r.companions),
            (1, 1, 2)
        );
        assert_eq!(r.rois_framed, 1);
        assert_eq!(r.files.len(), 3);
        // The rewrite fills the empty frame and leaves no old UID.
        let changes = super::super::remap::Changes {
            remap: true,
            set: r.set.clone(),
            ..Default::default()
        };
        let sim = super::super::remap::simulate(&rs, &changes, &r.maps).unwrap();
        let g = RawFile::parse(sim.bytes).unwrap();
        let frames: Vec<String> = g
            .walk()
            .into_iter()
            .filter(|(_, e)| e.tag == tags::REFERENCED_FRAME_OF_REFERENCE_UID)
            .map(|(_, e)| e.text(&g.bytes))
            .collect();
        assert_eq!(frames, ["1.2.9", "1.2.9"]);
        let sim = super::super::remap::simulate(&rp, &changes, &r.maps).unwrap();
        let g = RawFile::parse(sim.bytes).unwrap();
        assert_eq!(
            referenced_in(&g, tags::REFERENCED_STRUCTURE_SET_SEQUENCE),
            ["9.9.1"]
        );
        assert_eq!(g.text(tags::STUDY_INSTANCE_UID).as_deref(), Some("1.2"));

        // A list out of slice order says nothing about the slices between.
        let jumbled = rs_file(&["9.5.0", "9.5.2", "9.5.1", "9.5.4", "9.5.3"], "rs2.dcm");
        assert!(plan(&jumbled, &target, &[], Mapping::Z, &[], &[]).is_err());
    }
}
