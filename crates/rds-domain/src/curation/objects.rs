//! Which RT objects of an import the archive holds already, when the UIDs
//! say nothing.
//!
//! Structure sets, plans, doses, RT images, registrations, records and
//! reports have no pixels to hash (or, for a dose or an RT image, pixels
//! that are not enough), and a re-anonymizing export gave them new UIDs. So
//! they are compared by content, through four fingerprints tried in order:
//!
//! 1. **mapped** - every element but the object's own UIDs, its creation
//!    stamp, the patient module, private and group-length elements, the
//!    per-export DoseReferenceUID and the DRR links
//!    (ReferencedReferenceImageSequence, ReferencedSetupImageSequence: only
//!    the full export has them), with every UID it refers to put **through
//!    the UID map** first - so contours copied onto two 4D phases stay two
//!    objects, and a plan matches only once its structure set has (the
//!    objects go in the order structure sets, plans, the rest, each match
//!    adding to the map).
//! 2. **content** - as 1, but every UI value left out, and the sequences an
//!    anonymizer or a repair drops (ContourImageSequence, a registration's
//!    ReferencedSeriesSequence, ROIGenerationAlgorithm). Several candidates
//!    are told apart by their creation stamp.
//! 3. **stamp** - one object with this creation stamp (SOP class plus the
//!    first of InstanceCreation, StructureSet, RTPlan, Content, Treatment
//!    date and time) on each side, differing only in the fields a repair
//!    changes (ROIName, ROIGenerationAlgorithm, TreatmentTime).
//! 4. **core** - for a re-export without dates: the beam names, energies
//!    and spot maps of a plan, the ROI names and contour data of a
//!    structure set.
//!
//! Inside the import, objects with one mapped fingerprint are one object:
//! the largest file (the most complete export) is kept and the others map
//! to it. Every match adds `source SOP → kept SOP` (and the series) to the
//! map, which `remap_references` then applies to what refers to them.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use dicom_core::{Tag, VR};
use dicom_dictionary_std::tags;
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::journal::hex;
use super::raw::{Element, RawFile};
use super::remap::{kind_of, UidKind, UidMaps};
use crate::progress::{Progress, CANCELLED};

/// Whether a SOP class is one of the objects compared here.
pub fn is_rt_object(class: &str) -> bool {
    use crate::sop::*;
    matches!(
        class,
        SOP_RTSTRUCT | SOP_RTPLAN | SOP_RTIONPLAN | SOP_RTDOSE | SOP_RTIMAGE | SOP_SEG
    ) || is_beams_record(class)
        || is_registration(class)
        || is_structured_report(class)
}

/// 0: structure sets, 1: plans, 2: the rest - the order matches go in.
fn phase(class: &str) -> u8 {
    use crate::sop::*;
    match class {
        SOP_RTSTRUCT => 0,
        SOP_RTPLAN | SOP_RTIONPLAN => 1,
        _ => 2,
    }
}

const DRR_LINKS: [Tag; 2] = [
    tags::REFERENCED_REFERENCE_IMAGE_SEQUENCE,
    tags::REFERENCED_SETUP_IMAGE_SEQUENCE,
];
const SCAN_SPOT_METERSET_WEIGHTS: Tag = Tag(0x300A, 0x0396);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Level {
    Mapped,
    Content,
    Tolerant,
}

fn skipped(path: &[Tag], e: &Element, level: Level, is_reg: bool) -> bool {
    let t = e.tag;
    let top = path.is_empty();
    if t.group() % 2 == 1 || t.element() == 0 || t.group() == 0x0010 {
        return true;
    }
    if top
        && matches!(
            t,
            tags::SOP_INSTANCE_UID
                | tags::SERIES_INSTANCE_UID
                | tags::INSTANCE_CREATION_DATE
                | tags::INSTANCE_CREATION_TIME
        )
    {
        return true;
    }
    if t == tags::DOSE_REFERENCE_UID || DRR_LINKS.contains(&t) {
        return true;
    }
    if level == Level::Mapped {
        return false;
    }
    if e.vr == VR::UI
        || t == tags::CONTOUR_IMAGE_SEQUENCE
        || t == tags::ROI_GENERATION_ALGORITHM
        || (is_reg && t == tags::REFERENCED_SERIES_SEQUENCE)
    {
        return true;
    }
    level == Level::Tolerant && matches!(t, tags::ROI_NAME | tags::TREATMENT_TIME)
}

/// A fingerprint of `f` at `level`, UIDs through `maps` (mapped level).
fn fingerprint(f: &RawFile, level: Level, maps: &UidMaps) -> String {
    let is_reg = crate::sop::is_registration(&f.text(tags::SOP_CLASS_UID).unwrap_or_default());
    let mut h = Sha256::new();
    fn go(
        f: &RawFile,
        els: &[Element],
        path: &mut Vec<Tag>,
        h: &mut Sha256,
        level: Level,
        maps: &UidMaps,
        is_reg: bool,
    ) {
        for e in els {
            if skipped(path, e, level, is_reg) {
                continue;
            }
            h.update(e.tag.group().to_le_bytes());
            h.update(e.tag.element().to_le_bytes());
            match &e.items {
                Some(items) if e.tag != super::raw::PIXEL_DATA => {
                    h.update((items.len() as u32).to_le_bytes());
                    path.push(e.tag);
                    for it in items {
                        h.update(b"{");
                        go(f, &it.elements, path, h, level, maps, is_reg);
                        h.update(b"}");
                    }
                    path.pop();
                }
                Some(_) => {
                    if let Some(px) = f.pixel_bytes() {
                        h.update(&px);
                    }
                }
                None if e.vr == VR::UI => {
                    let v = e.text(&f.bytes);
                    let v = maps.remap(kind_of(path, e.tag), &v).unwrap_or(v);
                    h.update(v.as_bytes());
                }
                None => {
                    let v = e.value(&f.bytes);
                    // Text padding is not content.
                    let end = v
                        .iter()
                        .rposition(|b| *b != 0 && *b != b' ')
                        .map_or(0, |i| i + 1);
                    h.update(&v[..end]);
                }
            }
            h.update(b";");
        }
    }
    go(f, &f.elements, &mut Vec::new(), &mut h, level, maps, is_reg);
    hex(&h.finalize())
}

/// The creation stamp: SOP class and the first date and time it carries.
fn stamp(f: &RawFile) -> Option<String> {
    let t = |tag| f.text(tag).unwrap_or_default();
    [
        (tags::INSTANCE_CREATION_DATE, tags::INSTANCE_CREATION_TIME),
        (tags::STRUCTURE_SET_DATE, tags::STRUCTURE_SET_TIME),
        (tags::RT_PLAN_DATE, tags::RT_PLAN_TIME),
        (tags::CONTENT_DATE, tags::CONTENT_TIME),
        (tags::TREATMENT_DATE, tags::TREATMENT_TIME),
    ]
    .into_iter()
    .map(|(d, tm)| (t(d), t(tm)))
    .find(|(d, _)| !d.is_empty())
    .map(|(d, tm)| format!("{}|{d}|{tm}", t(tags::SOP_CLASS_UID)))
}

/// The core of a plan or a structure set: numbers rounded to 0.001.
fn core(f: &RawFile) -> Option<String> {
    let class = f.text(tags::SOP_CLASS_UID).unwrap_or_default();
    let wanted: &[Tag] = match phase(&class) {
        0 => &[tags::ROI_NAME, tags::CONTOUR_DATA],
        1 => &[
            tags::BEAM_NAME,
            tags::NOMINAL_BEAM_ENERGY,
            tags::SCAN_SPOT_POSITION_MAP,
            SCAN_SPOT_METERSET_WEIGHTS,
        ],
        _ => return None,
    };
    let mut h = Sha256::new();
    let mut any = false;
    for (_, e) in f.walk() {
        if !wanted.contains(&e.tag) || e.items.is_some() {
            continue;
        }
        any = true;
        h.update(e.tag.element().to_le_bytes());
        match e.vr {
            VR::DS | VR::IS => {
                for v in e.text(&f.bytes).split('\\') {
                    match v.trim().parse::<f64>() {
                        Ok(x) => h.update(format!("{:.3};", x).as_bytes()),
                        Err(_) => h.update(v.trim().as_bytes()),
                    }
                }
            }
            VR::FL => {
                for c in e.value(&f.bytes).as_chunks::<4>().0 {
                    let x = f32::from_le_bytes(*c);
                    h.update(format!("{:.3};", x).as_bytes());
                }
            }
            _ => h.update(e.text(&f.bytes).trim().as_bytes()),
        }
    }
    any.then(|| hex(&h.finalize()))
}

/// One object, read.
struct Obj {
    path: PathBuf,
    bytes: u64,
    sop: String,
    series: String,
    class: String,
    file: RawFile,
    content: String,
    tolerant: String,
    stamp: Option<String>,
    core: Option<String>,
}

fn read(path: &Path) -> Option<Obj> {
    let file = RawFile::read(path).ok()?;
    let class = file.text(tags::SOP_CLASS_UID)?;
    if !is_rt_object(&class) {
        return None;
    }
    let none = UidMaps::default();
    Some(Obj {
        path: path.to_path_buf(),
        bytes: file.bytes.len() as u64,
        sop: file.text(tags::SOP_INSTANCE_UID).unwrap_or_default(),
        series: file.text(tags::SERIES_INSTANCE_UID).unwrap_or_default(),
        content: fingerprint(&file, Level::Content, &none),
        tolerant: fingerprint(&file, Level::Tolerant, &none),
        stamp: stamp(&file),
        core: core(&file),
        class,
        file,
    })
}

/// How a source object was matched.
#[derive(Clone, Debug, Serialize)]
pub struct Match {
    pub path: PathBuf,
    pub sop: String,
    /// The object kept: in the archive, or another of the source.
    pub kept: PathBuf,
    pub kept_sop: String,
    /// `mapped`, `content`, `stamp`, `core`; `inside` for a copy within the
    /// source.
    pub how: &'static str,
}

#[derive(Debug, Default, Serialize)]
pub struct Dedupe {
    pub matches: Vec<Match>,
    /// Source objects per SOP class name: (objects, matched).
    pub by_class: BTreeMap<String, (usize, usize)>,
    /// The map, with every match added.
    #[serde(skip)]
    pub maps: UidMaps,
}

/// Compare the RT objects of `source` with those of `against`, starting
/// from `maps` (the image twins'). Only files [`is_rt_object`] are read.
pub fn dedupe(
    source: &[PathBuf],
    against: &[PathBuf],
    maps: UidMaps,
    p: &Progress,
) -> anyhow::Result<Dedupe> {
    p.set("Reading the objects");
    let src: Vec<Obj> = source.par_iter().filter_map(|p| read(p)).collect();
    let arc: Vec<Obj> = against.par_iter().filter_map(|p| read(p)).collect();
    if p.cancelled() {
        anyhow::bail!(CANCELLED);
    }
    let mut out = Dedupe {
        maps,
        ..Default::default()
    };
    for o in &src {
        out.by_class
            .entry(crate::sop::name_or_uid(&o.class))
            .or_default()
            .0 += 1;
    }
    let none = UidMaps::default();
    for ph in 0..3u8 {
        let s: Vec<&Obj> = src.iter().filter(|o| phase(&o.class) == ph).collect();
        let a: Vec<&Obj> = arc.iter().filter(|o| phase(&o.class) == ph).collect();
        // The archive's own UIDs are what the map leads to: no map.
        let a_mapped: HashMap<String, &Obj> = a
            .iter()
            .map(|o| (fingerprint(&o.file, Level::Mapped, &none), *o))
            .collect();
        let mut by_content: HashMap<&str, Vec<&Obj>> = HashMap::new();
        let mut by_stamp: HashMap<&str, Vec<&Obj>> = HashMap::new();
        let mut by_core: HashMap<&str, Vec<&Obj>> = HashMap::new();
        for o in &a {
            by_content.entry(o.content.as_str()).or_default().push(o);
            if let Some(st) = &o.stamp {
                by_stamp.entry(st.as_str()).or_default().push(o);
            }
            if let Some(c) = &o.core {
                by_core.entry(c.as_str()).or_default().push(o);
            }
        }
        let src_stamps: HashMap<&str, usize> =
            s.iter()
                .filter_map(|o| o.stamp.as_deref())
                .fold(HashMap::new(), |mut m, st| {
                    *m.entry(st).or_default() += 1;
                    m
                });
        let mapped: Vec<String> = s
            .par_iter()
            .map(|o| fingerprint(&o.file, Level::Mapped, &out.maps))
            .collect();
        let mut matched: Vec<(usize, &Obj, &'static str)> = Vec::new();
        for (i, o) in s.iter().enumerate() {
            let found: Option<(&Obj, &'static str)> = if let Some(t) = a_mapped.get(&mapped[i]) {
                Some((t, "mapped"))
            } else if let Some(list) = by_content.get(o.content.as_str()) {
                match list.as_slice() {
                    [one] => Some((one, "content")),
                    // Told apart by the stamp; still several: none for sure.
                    many => {
                        let same: Vec<&&Obj> = many
                            .iter()
                            .filter(|t| t.stamp.is_some() && t.stamp == o.stamp)
                            .collect();
                        (same.len() == 1).then(|| (*same[0], "content"))
                    }
                }
            } else if let Some(t) = o
                .stamp
                .as_deref()
                .filter(|st| src_stamps.get(st) == Some(&1))
                .and_then(|st| by_stamp.get(st))
                .and_then(|l| (l.len() == 1).then(|| l[0]))
                .filter(|t| t.tolerant == o.tolerant)
            {
                Some((t, "stamp"))
            } else if o.stamp.is_none() {
                o.core
                    .as_deref()
                    .and_then(|c| by_core.get(c))
                    .and_then(|l| (l.len() == 1).then(|| (l[0], "core")))
            } else {
                None
            };
            if let Some((t, how)) = found {
                matched.push((i, t, how));
            }
        }
        let mut done = vec![false; s.len()];
        for (i, t, how) in matched {
            let o = s[i];
            done[i] = true;
            let _ = out.maps.insert(UidKind::Object, &o.sop, &t.sop);
            let _ = out.maps.insert(UidKind::Object, &o.series, &t.series);
            out.by_class
                .entry(crate::sop::name_or_uid(&o.class))
                .or_default()
                .1 += 1;
            out.matches.push(Match {
                path: o.path.clone(),
                sop: o.sop.clone(),
                kept: t.path.clone(),
                kept_sop: t.sop.clone(),
                how,
            });
        }
        // Copies inside the source: one mapped fingerprint, the largest kept.
        let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, fp) in mapped.iter().enumerate().filter(|(i, _)| !done[*i]) {
            groups.entry(fp.as_str()).or_default().push(i);
        }
        let mut groups: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
        groups.sort();
        for mut g in groups {
            g.sort_by(|x, y| {
                s[*y]
                    .bytes
                    .cmp(&s[*x].bytes)
                    .then(s[*x].path.cmp(&s[*y].path))
            });
            let keep = s[g[0]];
            for &i in &g[1..] {
                let o = s[i];
                let _ = out.maps.insert(UidKind::Object, &o.sop, &keep.sop);
                out.by_class
                    .entry(crate::sop::name_or_uid(&o.class))
                    .or_default()
                    .1 += 1;
                out.matches.push(Match {
                    path: o.path.clone(),
                    sop: o.sop.clone(),
                    kept: keep.path.clone(),
                    kept_sop: keep.sop.clone(),
                    how: "inside",
                });
            }
        }
    }
    out.maps.resolve()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::raw::tests::{ex, item, part10};
    use super::super::raw::{encode_text, TS_EXPLICIT};
    use super::*;

    fn ui(tag: Tag, v: &str) -> Vec<u8> {
        ex(tag, "UI", &encode_text(VR::UI, v))
    }

    /// A structure set on `image`, made at `stamp`, with one contour.
    fn rs(
        dir: &std::path::Path,
        name: &str,
        sop: &str,
        image: &str,
        stamp: &str,
        z: &str,
    ) -> PathBuf {
        let mut d = Vec::new();
        d.extend(ex(tags::INSTANCE_CREATION_DATE, "DA", stamp.as_bytes()));
        d.extend(ui(tags::SOP_CLASS_UID, crate::sop::SOP_RTSTRUCT));
        d.extend(ui(tags::SOP_INSTANCE_UID, sop));
        d.extend(ex(tags::PATIENT_ID, "LO", b"P01-001 "));
        d.extend(ui(tags::SERIES_INSTANCE_UID, &format!("{sop}.9")));
        let img = item(&ui(tags::REFERENCED_SOP_INSTANCE_UID, image));
        let mut c = ex(tags::CONTOUR_IMAGE_SEQUENCE, "SQ", &img);
        c.extend(ex(
            tags::CONTOUR_DATA,
            "DS",
            &encode_text(VR::DS, &format!("0\\0\\{z}")),
        ));
        let contour = item(&c);
        let roi = item(&ex(tags::CONTOUR_SEQUENCE, "SQ", &contour));
        d.extend(ex(tags::ROI_CONTOUR_SEQUENCE, "SQ", &roi));
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
        p
    }

    #[test]
    fn rt_objects_match_through_the_map_by_content_and_inside_the_source() {
        let d = std::env::temp_dir().join("rds_objects");
        let _ = std::fs::remove_dir_all(&d);
        let (arc, src) = (d.join("archive"), d.join("import"));
        // The archive: one set on image 1.1 at z 5 (phase A), one on 2.1
        // at z 5 (phase B): the same contour on two phases.
        let a1 = rs(&arc, "a1", "1.9.1", "1.1", "20240101", "5");
        let a2 = rs(&arc, "a2", "1.9.2", "2.1", "20240101", "5");
        // The import: the phase-A set re-anonymized (its image 7.1 is a twin
        // of 1.1), twice (a plan folder's copy), and a set whose UIDs say
        // nothing and whose content matches no archive set.
        let s1 = rs(&src.join("full"), "s1", "7.9.1", "7.1", "20240101", "5");
        let s2 = rs(&src.join("plan"), "s2", "8.9.1", "7.1", "20240101", "5");
        let s3 = rs(&src.join("full"), "s3", "7.9.3", "7.3", "20240202", "9");
        let mut maps = UidMaps::default();
        maps.insert(UidKind::Object, "7.1", "1.1").unwrap();
        let r = dedupe(
            &[s1.clone(), s2.clone(), s3.clone()],
            &[a1.clone(), a2],
            maps,
            &Progress::default(),
        )
        .unwrap();
        let how = |p: &PathBuf| {
            r.matches
                .iter()
                .find(|m| m.path == *p)
                .map(|m| (m.how, m.kept_sop.clone()))
        };
        // Through the map, to the phase-A set and not the phase-B one,
        // although their content is the same but for the image.
        assert_eq!(how(&s1), Some(("mapped", "1.9.1".into())));
        assert_eq!(how(&s2), Some(("mapped", "1.9.1".into())));
        assert_eq!(how(&s3), None);
        assert_eq!(
            r.maps.objects.get("7.9.1").map(String::as_str),
            Some("1.9.1")
        );
    }
}
