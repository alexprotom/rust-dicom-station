//! Changing the UIDs an object refers to, by type.
//!
//! A re-anonymized export gives every object new UIDs; once its CT is
//! found to be a copy of one in the archive (pixel twins,
//! [`super::dupes`]), the structure sets, plans and doses of the export
//! must refer to the archive's CT instead. That is a rewrite of the UI
//! values they hold - but never by value alone: Philips writes one UID as
//! both the SeriesInstanceUID and the FrameOfReferenceUID of a 4D phase,
//! and a map from values would rewrite the wrong one. So there are three
//! maps ([`UidMaps`]), and every UI element is given to one by where it
//! sits ([`kind_of`]):
//!
//! * **objects** - Referenced SOP Instance UIDs, Series Instance UIDs
//!   inside a reference sequence, an SR's `UID`;
//! * **studies** - the Study Instance UID (the object's own, which moves
//!   it into the target study, and any inside a sequence), and the
//!   Referenced SOP Instance UID of an RT Referenced Study or Referenced
//!   Study item (whose class is a study's);
//! * **frames** - (Referenced / Related / Source) Frame of Reference UIDs,
//!   and RaySearch's misplaced one in (0008,1155) of a
//!   ReferencedFrameOfReferenceSequence item.
//!
//! The object's own SOP Instance UID and Series Instance UID (top level)
//! are never touched, and neither is anything in a private element.
//!
//! A rewrite is described by [`Changes`] - tags to set (at any depth),
//! tags to remove, the private groups to drop, and whether to apply the
//! manifest's maps - and [`simulate`] runs it in memory, so that a dry run
//! can check the result (no old UID left, every reference resolving, no
//! patient token left) before anything is written; the apply then insists
//! on the very bytes the dry run checked (their SHA-256).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Result};
use dicom_core::{Tag, VR};
use dicom_dictionary_std::tags;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::journal::{hex, parse_tag};
use super::raw::{self, encode_text, At, RawFile};

/// Retired, but written by old planning systems.
const RELATED_FRAME_OF_REFERENCE_UID: Tag = Tag(0x3006, 0x00C2);

/// UID → UID, by the type of what the UID names.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct UidMaps {
    pub objects: BTreeMap<String, String>,
    pub studies: BTreeMap<String, String>,
    pub frames: BTreeMap<String, String>,
}

/// What a UI element names, by where it sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UidKind {
    /// The object's own SOP / Series Instance UID: never remapped.
    Own,
    Object,
    Study,
    Frame,
    /// Any other UI: classes, transfer syntaxes, private UIDs.
    Other,
}

/// The kind of the UI element `tag` under the sequences `path`.
pub fn kind_of(path: &[Tag], tag: Tag) -> UidKind {
    if path.iter().any(|t| t.group() % 2 == 1) || tag.group() % 2 == 1 {
        return UidKind::Other;
    }
    let top = path.is_empty();
    match tag {
        tags::SOP_INSTANCE_UID | tags::SERIES_INSTANCE_UID if top => UidKind::Own,
        tags::SOP_INSTANCE_UID | tags::SERIES_INSTANCE_UID | tags::UID => UidKind::Object,
        tags::STUDY_INSTANCE_UID => UidKind::Study,
        tags::FRAME_OF_REFERENCE_UID
        | tags::REFERENCED_FRAME_OF_REFERENCE_UID
        | RELATED_FRAME_OF_REFERENCE_UID
        | tags::SOURCE_FRAME_OF_REFERENCE_UID => UidKind::Frame,
        tags::REFERENCED_SOP_INSTANCE_UID => match path.last() {
            Some(&tags::RT_REFERENCED_STUDY_SEQUENCE) | Some(&tags::REFERENCED_STUDY_SEQUENCE) => {
                UidKind::Study
            }
            Some(&tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE) => UidKind::Frame,
            _ => UidKind::Object,
        },
        _ => UidKind::Other,
    }
}

impl UidMaps {
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty() && self.studies.is_empty() && self.frames.is_empty()
    }

    pub fn len(&self) -> usize {
        self.objects.len() + self.studies.len() + self.frames.len()
    }

    pub fn map(&self, kind: UidKind) -> Option<&BTreeMap<String, String>> {
        match kind {
            UidKind::Object => Some(&self.objects),
            UidKind::Study => Some(&self.studies),
            UidKind::Frame => Some(&self.frames),
            UidKind::Own | UidKind::Other => None,
        }
    }

    fn map_mut(&mut self, kind: UidKind) -> Option<&mut BTreeMap<String, String>> {
        match kind {
            UidKind::Object => Some(&mut self.objects),
            UidKind::Study => Some(&mut self.studies),
            UidKind::Frame => Some(&mut self.frames),
            UidKind::Own | UidKind::Other => None,
        }
    }

    /// The new value of a UI element (backslash-separated values mapped one
    /// by one), or `None` when nothing in it is mapped.
    pub fn remap(&self, kind: UidKind, value: &str) -> Option<String> {
        let m = self.map(kind)?;
        let mut any = false;
        let out: Vec<&str> = value
            .split('\\')
            .map(|v| match m.get(v.trim()) {
                Some(new) => {
                    any = true;
                    new.as_str()
                }
                None => v.trim(),
            })
            .collect();
        any.then(|| out.join("\\"))
    }

    /// Add `from → to`; an entry that maps `from` elsewhere already is a
    /// conflict, returned as the target it had.
    pub fn insert(&mut self, kind: UidKind, from: &str, to: &str) -> Result<(), String> {
        if from == to || from.is_empty() || to.is_empty() {
            return Ok(());
        }
        let Some(m) = self.map_mut(kind) else {
            return Ok(());
        };
        match m.get(from) {
            Some(old) if old != to => Err(old.clone()),
            _ => {
                m.insert(from.to_string(), to.to_string());
                Ok(())
            }
        }
    }

    /// Follow chains (`a → b`, `b → c` gives `a → c`), and refuse cycles.
    pub fn resolve(&mut self) -> Result<()> {
        for m in [&mut self.objects, &mut self.studies, &mut self.frames] {
            let snapshot = m.clone();
            for (from, to) in m.iter_mut() {
                let mut seen = BTreeSet::from([from.clone()]);
                while let Some(next) = snapshot.get(to.as_str()) {
                    if !seen.insert(to.clone()) {
                        bail!("the UID map runs in a circle through {from}");
                    }
                    *to = next.clone();
                }
            }
        }
        Ok(())
    }

    /// Every map of `other` added to this one; conflicts are refused.
    pub fn merge(&mut self, other: &UidMaps) -> Result<()> {
        for (kind, m) in [
            (UidKind::Object, &other.objects),
            (UidKind::Study, &other.studies),
            (UidKind::Frame, &other.frames),
        ] {
            for (from, to) in m {
                if let Err(old) = self.insert(kind, from, to) {
                    bail!("{from} would map to both {old} and {to}");
                }
            }
        }
        Ok(())
    }
}

/// What a rewrite of one file does.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Changes {
    /// `(tag as GGGGEEEE, value)`: set wherever the tag is, in sequences
    /// too (an element that is not there is not added).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub set: Vec<(String, String)>,
    /// Tags to remove, at any depth (a sequence goes with its items).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub remove: Vec<String>,
    /// Drop every private element (odd groups).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub remove_private: bool,
    /// Apply the manifest's UID maps.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub remap: bool,
}

/// [`Changes`] with its tags parsed, ready to drive a rewrite.
pub struct Compiled<'a> {
    set: Vec<(Tag, String)>,
    remove: BTreeSet<Tag>,
    remove_private: bool,
    maps: Option<&'a UidMaps>,
}

impl Changes {
    pub fn is_empty(&self) -> bool {
        self.set.is_empty() && self.remove.is_empty() && !self.remove_private && !self.remap
    }

    pub fn compile<'a>(&self, maps: &'a UidMaps) -> Result<Compiled<'a>> {
        Ok(Compiled {
            set: self
                .set
                .iter()
                .map(|(t, v)| Ok((parse_tag(t)?, v.clone())))
                .collect::<Result<_>>()?,
            remove: self
                .remove
                .iter()
                .map(|t| parse_tag(t))
                .collect::<Result<_>>()?,
            remove_private: self.remove_private,
            maps: self.remap.then_some(maps),
        })
    }
}

impl Compiled<'_> {
    /// Whether the element `tag` under `path` goes.
    pub fn removes(&self, path: &[Tag], tag: Tag) -> bool {
        self.remove.contains(&tag) || (self.remove_private && tag.group() % 2 == 1)
            // An element inside a private sequence goes with it.
            || (self.remove_private && path.iter().any(|t| t.group() % 2 == 1))
    }

    /// The new value of a leaf, encoded, or `None` to keep it.
    pub fn edit(&self, at: &At) -> Option<Vec<u8>> {
        if let Some((_, v)) = self.set.iter().find(|(t, _)| *t == at.tag) {
            let new = encode_text(at.vr, v);
            return (new != at.value).then_some(new);
        }
        if at.vr == VR::UI {
            if let Some(maps) = self.maps {
                let new = maps.remap(kind_of(at.path, at.tag), &at.text())?;
                let new = encode_text(VR::UI, &new);
                return (new != at.value).then_some(new);
            }
        }
        None
    }
}

/// What a dry run of a rewrite gives: the file's SHA-256 now, the bytes it
/// would have, their SHA-256, and how many elements change.
pub struct Simulated {
    pub source: String,
    pub bytes: Vec<u8>,
    pub expect: String,
    pub changed: usize,
}

/// Run `changes` on a file in memory.
pub fn simulate(path: &Path, changes: &Changes, maps: &UidMaps) -> Result<Simulated> {
    let file = RawFile::read(path)?;
    let source = hex(&Sha256::digest(&file.bytes));
    let c = changes.compile(maps)?;
    let (bytes, changed) = raw::rewrite_with(&file, &|p, t| c.removes(p, t), &mut |at| c.edit(at))?;
    Ok(Simulated {
        source,
        expect: hex(&Sha256::digest(&bytes)),
        bytes,
        changed,
    })
}

/// Every UI value of a file that is a reference (not its own UIDs, nothing
/// private), with its kind.
pub fn references(f: &RawFile) -> Vec<(UidKind, String)> {
    let mut out = Vec::new();
    for (path, e) in f.walk() {
        if e.vr != VR::UI || e.items.is_some() {
            continue;
        }
        let kind = kind_of(&path, e.tag);
        if matches!(kind, UidKind::Object | UidKind::Study | UidKind::Frame) {
            for v in e.text(&f.bytes).split('\\') {
                let v = v.trim();
                if !v.is_empty() {
                    out.push((kind, v.to_string()));
                }
            }
        }
    }
    out
}

/// The UIDs a map would replace that are still in `f`, in the place of
/// their kind: what must be empty after a remap.
pub fn leftovers(f: &RawFile, maps: &UidMaps) -> Vec<(UidKind, String)> {
    let mut out: Vec<(UidKind, String)> = references(f)
        .into_iter()
        .filter(|(k, v)| maps.map(*k).is_some_and(|m| m.contains_key(v)))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// What a dry run of rewrites found.
#[derive(Debug, Default)]
pub struct Planned {
    /// One [`Op::Rewrite`](super::journal::Op::Rewrite) per file that
    /// changes, with the SHA-256 it has and will have.
    pub ops: Vec<super::journal::Op>,
    /// Files that would come out the same.
    pub unchanged: usize,
    /// Old UIDs a rewrite would leave in place: `(file, kind, uid)`.
    pub leftovers: Vec<(std::path::PathBuf, UidKind, String)>,
    /// References that would lead nowhere: `(kind, uid)`, once each.
    pub unresolved: BTreeSet<(UidKind, String)>,
    /// Files that cannot be rewritten, and why.
    pub refused: Vec<(std::path::PathBuf, String)>,
}

/// Run `changes` dry on every file: what would change, what old UIDs would
/// be left, and which references of the result `resolves` does not know
/// (the files themselves, once rewritten, always resolve: their own SOP,
/// series, study and frame).
pub fn plan_rewrites(
    files: &[std::path::PathBuf],
    changes: &Changes,
    maps: &UidMaps,
    resolves: &(dyn Fn(UidKind, &str) -> bool + Sync),
) -> Planned {
    use rayon::prelude::*;
    let results: Vec<_> = files
        .par_iter()
        .map(|f| {
            // A file that is not DICOM is no business of a rewrite.
            if !RawFile::read(f).is_ok_and(|r| r.is_dicom()) {
                return (f, None);
            }
            let r = simulate(f, changes, maps).and_then(|s| {
                let g = RawFile::parse(s.bytes.clone())?;
                Ok((s, g))
            });
            (f, Some(r))
        })
        .collect();
    // What the rewritten files are, for references among themselves.
    let mut own: BTreeSet<(UidKind, String)> = BTreeSet::new();
    for (_, r) in &results {
        if let Some(Ok((_, g))) = r {
            for (tag, kind) in [
                (tags::SOP_INSTANCE_UID, UidKind::Object),
                (tags::SERIES_INSTANCE_UID, UidKind::Object),
                (tags::STUDY_INSTANCE_UID, UidKind::Study),
                (tags::FRAME_OF_REFERENCE_UID, UidKind::Frame),
            ] {
                if let Some(v) = g.text(tag).filter(|v| !v.is_empty()) {
                    own.insert((kind, v));
                }
            }
        }
    }
    let mut out = Planned::default();
    for (f, r) in results {
        let (s, g) = match r {
            None => continue,
            Some(Ok(x)) => x,
            Some(Err(e)) => {
                out.refused.push((f.clone(), format!("{e:#}")));
                continue;
            }
        };
        for (k, v) in leftovers(&g, maps) {
            out.leftovers.push((f.clone(), k, v));
        }
        for (k, v) in references(&g) {
            if !own.contains(&(k, v.clone())) && !resolves(k, &v) {
                out.unresolved.insert((k, v));
            }
        }
        if s.changed == 0 {
            out.unchanged += 1;
            continue;
        }
        out.ops.push(super::journal::Op::Rewrite {
            path: f.clone(),
            changes: changes.clone(),
            source: Some(s.source),
            expect: Some(s.expect),
        });
    }
    out
}

/// The contours of a structure set and the images they name: for every
/// ContourSequence item, the Referenced SOP Instance UIDs of its
/// ContourImageSequence and the z of its first point.
pub fn contour_images(f: &RawFile) -> Vec<(String, Option<f64>)> {
    let mut out = Vec::new();
    let Some(roi_contours) = f
        .get(tags::ROI_CONTOUR_SEQUENCE)
        .and_then(|e| e.items.as_ref())
    else {
        return out;
    };
    for roi in roi_contours {
        let Some(contours) = roi
            .elements
            .iter()
            .find(|e| e.tag == tags::CONTOUR_SEQUENCE)
            .and_then(|e| e.items.as_ref())
        else {
            continue;
        };
        for c in contours {
            let z = c
                .elements
                .iter()
                .find(|e| e.tag == tags::CONTOUR_DATA)
                .and_then(|e| {
                    e.text(&f.bytes)
                        .split('\\')
                        .nth(2)
                        .and_then(|z| z.trim().parse::<f64>().ok())
                });
            let refs = c
                .elements
                .iter()
                .find(|e| e.tag == tags::CONTOUR_IMAGE_SEQUENCE)
                .and_then(|e| e.items.as_ref());
            for it in refs.into_iter().flatten() {
                for e in &it.elements {
                    if e.tag == tags::REFERENCED_SOP_INSTANCE_UID {
                        out.push((e.text(&f.bytes), z));
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::raw::tests::{ex, item, part10};
    use super::super::raw::TS_EXPLICIT;
    use super::*;

    fn ui(tag: Tag, v: &str) -> Vec<u8> {
        ex(tag, "UI", &encode_text(VR::UI, v))
    }

    fn sq(tag: Tag, items: &[Vec<u8>]) -> Vec<u8> {
        let body: Vec<u8> = items.iter().flat_map(|b| item(b)).collect();
        ex(tag, "SQ", &body)
    }

    /// A Philips-style object whose series UID is also its frame's, and a
    /// structure set pointing at it.
    #[test]
    fn a_uid_is_remapped_by_where_it_sits_not_by_its_value() {
        let mut d = Vec::new();
        d.extend(ui(tags::SOP_INSTANCE_UID, "9.1.1"));
        d.extend(ui(tags::STUDY_INSTANCE_UID, "9.0"));
        d.extend(ui(tags::SERIES_INSTANCE_UID, "9.9"));
        // The frame of reference has the series' value (Philips 4D).
        d.extend(ui(tags::FRAME_OF_REFERENCE_UID, "9.9"));
        d.extend(sq(
            tags::REFERENCED_SERIES_SEQUENCE,
            &[[ui(tags::SERIES_INSTANCE_UID, "9.9")].concat()],
        ));
        d.extend(sq(
            tags::RT_REFERENCED_STUDY_SEQUENCE,
            &[ui(tags::REFERENCED_SOP_INSTANCE_UID, "9.0")],
        ));
        d.extend(ex(Tag(0x0009, 0x1001), "UI", &encode_text(VR::UI, "9.9")));
        let dir = std::env::temp_dir().join("rds_remap_kind");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.dcm");
        std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
        let mut maps = UidMaps::default();
        maps.insert(UidKind::Frame, "9.9", "1.5").unwrap();
        maps.insert(UidKind::Study, "9.0", "1.0").unwrap();
        let changes = Changes {
            remap: true,
            ..Default::default()
        };
        let s = simulate(&p, &changes, &maps).unwrap();
        let g = RawFile::parse(s.bytes).unwrap();
        // Own series untouched, the frame remapped, the series reference
        // (an object) untouched since only the frame is in the map.
        assert_eq!(g.text(tags::SERIES_INSTANCE_UID).as_deref(), Some("9.9"));
        assert_eq!(g.text(tags::FRAME_OF_REFERENCE_UID).as_deref(), Some("1.5"));
        assert_eq!(g.text(tags::STUDY_INSTANCE_UID).as_deref(), Some("1.0"));
        let refs = references(&g);
        assert!(refs.contains(&(UidKind::Object, "9.9".into())));
        assert!(refs.contains(&(UidKind::Study, "1.0".into())));
        assert!(leftovers(&g, &maps).is_empty());
        // The private UI is left as it was.
        assert_eq!(g.text(Tag(0x0009, 0x1001)).as_deref(), Some("9.9"));
        assert_eq!(s.changed, 3);
    }

    #[test]
    fn maps_follow_chains_and_refuse_two_targets() {
        let mut m = UidMaps::default();
        m.insert(UidKind::Study, "a", "b").unwrap();
        m.insert(UidKind::Study, "b", "c").unwrap();
        assert_eq!(m.insert(UidKind::Study, "a", "d"), Err("b".into()));
        m.resolve().unwrap();
        assert_eq!(m.studies.get("a").map(String::as_str), Some("c"));
        m.insert(UidKind::Frame, "x", "y").unwrap();
        m.insert(UidKind::Frame, "y", "x").unwrap();
        assert!(m.resolve().is_err());
    }

    const OTHER_IDS: Tag = Tag(0x0010, 0x1000);

    #[test]
    fn removals_take_tags_anywhere_and_private_groups_whole() {
        let mut d = Vec::new();
        d.extend(ui(tags::SOP_INSTANCE_UID, "1.1"));
        d.extend(ex(OTHER_IDS, "LO", b"X1"));
        d.extend(sq(
            Tag(0x0029, 0x1010),
            &[ex(tags::PATIENT_NAME, "PN", b"A^B ")],
        ));
        d.extend(ex(tags::PATIENT_NAME, "PN", b"A^B "));
        let dir = std::env::temp_dir().join("rds_remap_remove");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.dcm");
        std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
        let changes = Changes {
            set: vec![("00100010".into(), "P01-001".into())],
            remove: vec!["00101000".into()],
            remove_private: true,
            remap: false,
        };
        let s = simulate(&p, &changes, &UidMaps::default()).unwrap();
        let g = RawFile::parse(s.bytes.clone()).unwrap();
        assert!(g.problems.is_empty());
        assert_eq!(g.text(tags::PATIENT_NAME).as_deref(), Some("P01-001"));
        assert!(g.get(OTHER_IDS).is_none());
        assert!(g.get(Tag(0x0029, 0x1010)).is_none());
        // And on disk, verified.
        let none = UidMaps::default();
        let c = changes.compile(&none).unwrap();
        let (n, sha) = raw::rewrite_file_with(
            &p,
            &|path, t| c.removes(path, t),
            &mut |at| c.edit(at),
            Some(&s.expect),
        )
        .unwrap();
        assert_eq!((n, sha), (s.changed, s.expect));
    }
}
