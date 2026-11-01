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
//!   export at all, the z of the contours drawn on it is used.
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
    pub warnings: Vec<String>,
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
/// along.
pub fn plan(
    rs: &Path,
    target: &[Facts],
    originals: &[Facts],
    mapping: Mapping,
    merge: &[String],
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

    let mut out = Relink::default();
    let mut unresolved = Vec::new();
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
            (Mapping::Z, None) => contour_z.get(sop).and_then(|z| {
                // A patient-space z: the axial target slice at that z.
                target
                    .iter()
                    .find(|t| t.position.is_some_and(|p| (p[2] - z).abs() < SAME_MM))
            }),
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
        let r = plan(&p, &target, &[], Mapping::Z, &[]).unwrap();
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
        assert!(plan(&p, &target, &[], Mapping::Z, &["9.7".into()]).is_err());
        // A contour on no target slice refuses.
        let short = vec![slice("1.2.3.2", 5.0)];
        assert!(plan(&p, &short, &[], Mapping::Z, &[]).is_err());
    }
}
