//! Which files of a source folder an archive holds already.
//!
//! Two ways a file can be a copy:
//!
//! * **exact** - the archive has a file with the same SOP Instance UID, and
//!   the two are element for element the same on their raw bytes, leaving
//!   out the tags the caller names (by default PatientName, PatientID and
//!   every group length: what a re-sort rewrites). The same UID with other
//!   content is reported as such, never as a copy.
//! * **pixel twin** - an export that re-anonymized its data gave every
//!   object new UIDs, so the UID says nothing; the pixels do. A twin has
//!   the same SHA-256 over its pixel bytes *and* the same
//!   ImagePositionPatient as a file of the archive. Secondary captures and
//!   placeholder images (64 pixels or fewer: Philips' respiratory-wave
//!   captures are one picture in every 4D series) are left out.
//!
//! Verdicts are per source series: `identical` (every file an exact copy),
//! `pixel-twin` (every file a twin, all of one archive series), `partial`
//! (some files copies, some not), `ambiguous` (twins spread over more than
//! one archive series: never merged on that basis), `new` (none). Twins
//! *inside* the source - the same CT re-exported with every plan - are
//! counted as well.
//!
//! Hashing an archive means reading all of it, so the hashes are kept in a
//! [`PixelIndex`] in the records folder, keyed by path, size and
//! modification time: a later run reads only what changed.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use dicom_core::Tag;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::facts::{self, Facts};
use super::raw::RawFile;
use crate::progress::{Progress, CANCELLED};

/// What the index remembers of one file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Indexed {
    pub bytes: u64,
    pub mtime: i64,
    pub sop: String,
    pub series: String,
    /// The Study Instance UID and Frame of Reference UID (for the UID maps
    /// a twin gives, [`twin_maps`]).
    #[serde(default)]
    pub study: String,
    #[serde(default)]
    pub frame: String,
    /// The SOP Class UID (for the RT objects of [`super::objects`]).
    #[serde(default)]
    pub class: String,
    pub hash: Option<String>,
    /// ImagePositionPatient, rounded to 0.001 mm, as text.
    pub position: Option<String>,
    /// Kept out of the pixel-twin logic ([`Facts::twin_candidate`]).
    pub candidate: bool,
}

/// Path → what it held when last read.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PixelIndex {
    /// [`INDEX_VERSION`]: an index of an older version is read again whole.
    #[serde(default)]
    pub version: u32,
    pub files: BTreeMap<String, Indexed>,
}

pub const INDEX_FILE: &str = "pixel-index.json";
/// 3: entries carry their study, frame of reference and SOP class.
pub const INDEX_VERSION: u32 = 3;

fn mtime(path: &Path) -> (u64, i64) {
    let m = std::fs::metadata(path).ok();
    let bytes = m.as_ref().map(|m| m.len()).unwrap_or(0);
    let t = m
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (bytes, t)
}

fn position_key(p: Option<[f64; 3]>) -> Option<String> {
    p.map(|p| format!("{:.3}\\{:.3}\\{:.3}", p[0], p[1], p[2]))
}

impl PixelIndex {
    pub fn load(records: &Path) -> PixelIndex {
        std::fs::read_to_string(records.join(INDEX_FILE))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .filter(|i: &PixelIndex| i.version == INDEX_VERSION)
            .unwrap_or(PixelIndex {
                version: INDEX_VERSION,
                files: BTreeMap::new(),
            })
    }

    /// Written whole, through a temporary file and a rename.
    pub fn save(&self, records: &Path) -> anyhow::Result<()> {
        std::fs::create_dir_all(records)?;
        let tmp = records.join(format!("{INDEX_FILE}.tmp"));
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, records.join(INDEX_FILE))?;
        Ok(())
    }

    /// The entries of `files`, reading (and hashing) only those the index
    /// does not know as they are now. Returns them, and how many were read.
    pub fn refresh(
        &mut self,
        files: &[PathBuf],
        p: &Progress,
    ) -> anyhow::Result<(Vec<(PathBuf, Indexed)>, usize)> {
        let stale: Vec<&PathBuf> = files
            .iter()
            .filter(|f| {
                let (b, t) = mtime(f);
                self.files
                    .get(&f.to_string_lossy().to_string())
                    .is_none_or(|e| e.bytes != b || e.mtime != t)
            })
            .collect();
        let n = stale.len();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let read: Vec<(String, Indexed)> = stale
            .par_iter()
            .map(|f| {
                let i = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i.is_multiple_of(256) {
                    p.set(format!("Hashing pixel data {i}/{n}"));
                }
                if p.cancelled() {
                    return (String::new(), Indexed::default());
                }
                let (bytes, mtime) = mtime(f);
                let x = facts::read(f, true);
                (
                    f.to_string_lossy().to_string(),
                    Indexed {
                        bytes,
                        mtime,
                        candidate: x.twin_candidate(),
                        sop: x.sop,
                        series: x.series,
                        study: x.study,
                        frame: x.frame,
                        class: x.sop_class,
                        hash: x.pixel_hash,
                        position: position_key(x.position),
                    },
                )
            })
            .collect();
        if p.cancelled() {
            anyhow::bail!(CANCELLED);
        }
        for (k, v) in read {
            self.files.insert(k, v);
        }
        let out = files
            .iter()
            .filter_map(|f| {
                self.files
                    .get(&f.to_string_lossy().to_string())
                    .map(|e| (f.clone(), e.clone()))
            })
            .collect();
        Ok((out, n))
    }
}

/// The tags left out of the exact comparison by default.
pub fn default_ignored() -> Vec<Tag> {
    vec![Tag(0x0010, 0x0010), Tag(0x0010, 0x0020)]
}

/// Whether two files are the same, element for element on their raw
/// bytes, but for `ignore` and every group length. The file meta group is
/// not compared (it names the program that wrote the file).
pub fn same_content(a: &Path, b: &Path, ignore: &[Tag]) -> anyhow::Result<bool> {
    let (x, y) = (RawFile::read(a)?, RawFile::read(b)?);
    let keep = |e: &&super::raw::Element| e.tag.element() != 0 && !ignore.contains(&e.tag);
    let xs: Vec<_> = x.elements.iter().filter(keep).collect();
    let ys: Vec<_> = y.elements.iter().filter(keep).collect();
    if xs.len() != ys.len() {
        return Ok(false);
    }
    // Values compared, not headers: the same element written in explicit
    // and in implicit VR is the same element.
    Ok(xs.iter().zip(&ys).all(|(p, q)| {
        p.tag == q.tag && x.bytes[p.value_start..p.end] == y.bytes[q.value_start..q.end]
    }))
}

/// One source series' verdict.
#[derive(Clone, Debug, Serialize)]
pub struct SeriesVerdict {
    pub series: String,
    pub modality: String,
    pub files: usize,
    /// `identical`, `pixel-twin`, `partial`, `ambiguous`, `new`.
    pub verdict: &'static str,
    pub exact: usize,
    pub twins: usize,
    /// Same SOP Instance UID as an archive file, other content.
    pub same_uid_other_content: usize,
    /// The archive series its copies are in, with how many each.
    pub matches: BTreeMap<String, usize>,
    /// Files whose pixels another file of the source has too.
    pub internal_twins: usize,
    /// The source files that are copies (exact or twin).
    #[serde(skip)]
    pub copies: Vec<PathBuf>,
    /// Each copy with the archive file it copies.
    #[serde(skip)]
    pub pairs: Vec<Pair>,
}

/// A source file and the archive file it is a copy of: their UIDs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pair {
    pub path: PathBuf,
    pub sop: String,
    pub series: String,
    pub study: String,
    pub frame: String,
    pub target_path: PathBuf,
    pub target: Indexed,
}

/// Any two files may be of one patient: the default of [`find`]'s
/// `same_patient`.
pub fn any_patient(_: &Path, _: &Path) -> bool {
    true
}

/// Find the copies of `source`'s files in `against`. `same_patient(source
/// file, archive file)` says whether the two can be of one patient (by
/// their folders): a pixel twin counts only then.
pub fn find(
    source: &[PathBuf],
    against: &[(PathBuf, Indexed)],
    ignore: &[Tag],
    same_patient: &(dyn Fn(&Path, &Path) -> bool + Sync),
    p: &Progress,
) -> anyhow::Result<Vec<SeriesVerdict>> {
    p.set("Reading the source");
    let src: Vec<Facts> = source.par_iter().map(|f| facts::read(f, true)).collect();
    if p.cancelled() {
        anyhow::bail!(CANCELLED);
    }
    Ok(find_in(&src, against, ignore, same_patient))
}

/// [`find`] on facts already read (with their pixel hashes).
pub fn find_in(
    src: &[Facts],
    against: &[(PathBuf, Indexed)],
    ignore: &[Tag],
    same_patient: &(dyn Fn(&Path, &Path) -> bool + Sync),
) -> Vec<SeriesVerdict> {
    let by_sop: HashMap<&str, &(PathBuf, Indexed)> =
        against.iter().map(|a| (a.1.sop.as_str(), a)).collect();
    let mut by_pixels: HashMap<(&str, &str), Vec<&(PathBuf, Indexed)>> = HashMap::new();
    for a in against.iter().filter(|a| a.1.candidate) {
        if let (Some(h), Some(pos)) = (&a.1.hash, &a.1.position) {
            by_pixels
                .entry((h.as_str(), pos.as_str()))
                .or_default()
                .push(a);
        }
    }
    // Twins inside the source: one pixel hash and position in two series.
    let mut inside: HashMap<(String, String), BTreeSet<String>> = HashMap::new();
    for f in src.iter().filter(|f| f.twin_candidate()) {
        if let (Some(h), Some(pos)) = (&f.pixel_hash, position_key(f.position)) {
            inside
                .entry((h.clone(), pos))
                .or_default()
                .insert(f.series.clone());
        }
    }
    // Per file: exact, same uid other content, twin of series, or new.
    enum Kind<'a> {
        Exact(&'a (PathBuf, Indexed)),
        OtherContent,
        Twin(&'a (PathBuf, Indexed)),
        /// Twins in this many archive series.
        TwinOfSeveral(usize),
        New,
    }
    let kinds: Vec<Kind> = src
        .par_iter()
        .map(|f| {
            if f.unreadable.is_some() {
                return Kind::New;
            }
            if let Some(a) = by_sop.get(f.sop.as_str()) {
                return match same_content(&f.path, &a.0, ignore) {
                    Ok(true) => Kind::Exact(a),
                    _ => Kind::OtherContent,
                };
            }
            if f.twin_candidate() {
                if let (Some(h), Some(pos)) = (&f.pixel_hash, position_key(f.position)) {
                    if let Some(list) = by_pixels.get(&(h.as_str(), pos.as_str())) {
                        let list: Vec<&&(PathBuf, Indexed)> = list
                            .iter()
                            .filter(|a| same_patient(&f.path, &a.0))
                            .collect();
                        let series: BTreeSet<&str> =
                            list.iter().map(|a| a.1.series.as_str()).collect();
                        match series.len() {
                            0 => {}
                            1 => return Kind::Twin(list[0]),
                            // Two archive series with these pixels at this
                            // position: a twin of each, of neither for sure.
                            n => return Kind::TwinOfSeveral(n),
                        }
                    }
                }
            }
            Kind::New
        })
        .collect();
    let mut by_series: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    // Files that are not DICOM are no series (an attachment is reported by
    // the tools that move it).
    for (i, f) in src
        .iter()
        .enumerate()
        .filter(|(_, f)| f.unreadable.is_none())
    {
        by_series.entry(f.series.as_str()).or_default().push(i);
    }
    let mut out = Vec::new();
    for (series, idx) in by_series {
        let mut v = SeriesVerdict {
            series: series.to_string(),
            modality: src[idx[0]].modality.clone(),
            files: idx.len(),
            verdict: "new",
            exact: 0,
            twins: 0,
            same_uid_other_content: 0,
            matches: BTreeMap::new(),
            internal_twins: 0,
            copies: Vec::new(),
            pairs: Vec::new(),
        };
        for &i in &idx {
            let f = &src[i];
            let pair = |a: &(PathBuf, Indexed)| Pair {
                path: f.path.clone(),
                sop: f.sop.clone(),
                series: f.series.clone(),
                study: f.study.clone(),
                frame: f.frame.clone(),
                target_path: a.0.clone(),
                target: a.1.clone(),
            };
            match &kinds[i] {
                Kind::Exact(a) => {
                    v.exact += 1;
                    *v.matches.entry(a.1.series.clone()).or_default() += 1;
                    v.copies.push(f.path.clone());
                    v.pairs.push(pair(a));
                }
                Kind::Twin(a) => {
                    v.twins += 1;
                    *v.matches.entry(a.1.series.clone()).or_default() += 1;
                    v.copies.push(f.path.clone());
                    v.pairs.push(pair(a));
                }
                Kind::TwinOfSeveral(n) => {
                    v.twins += 1;
                    *v.matches.entry(format!("{n} series")).or_default() += 1;
                    v.copies.push(f.path.clone());
                }
                Kind::OtherContent => v.same_uid_other_content += 1,
                Kind::New => {}
            }
            if let (Some(h), Some(pos)) = (&f.pixel_hash, position_key(f.position)) {
                if inside.get(&(h.clone(), pos)).is_some_and(|s| s.len() > 1) {
                    v.internal_twins += 1;
                }
            }
        }
        let copies = v.exact + v.twins;
        v.verdict = if copies == 0 {
            "new"
        } else if v.matches.len() > 1 || v.matches.keys().any(|k| k.ends_with(" series")) {
            "ambiguous"
        } else if v.exact == v.files {
            "identical"
        } else if copies == v.files {
            "pixel-twin"
        } else {
            "partial"
        };
        out.push(v);
    }
    out
}

/// A split found while mapping: one source study (or frame of reference)
/// whose copies lie in several archive ones.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Split {
    /// `study` or `frame`.
    pub kind: &'static str,
    pub source: String,
    pub targets: Vec<String>,
}

/// The UID maps the copies of whole series give: every copy's SOP, series,
/// study and frame of reference to its archive file's. Only series judged
/// `identical` or `pixel-twin` count (a partial or ambiguous one says
/// nothing for sure). A source study or frame of reference whose copies lie
/// in two archive studies or frames is not mapped: it is returned as a
/// [`Split`] - an earlier split of one study in the archive (for
/// `merge_studies` / `merge_frames`), or two patients.
pub fn twin_maps(verdicts: &[SeriesVerdict]) -> (super::remap::UidMaps, Vec<Split>) {
    use super::remap::{UidKind, UidMaps};
    let mut maps = UidMaps::default();
    let mut studies: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut frames: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for v in verdicts
        .iter()
        .filter(|v| matches!(v.verdict, "identical" | "pixel-twin"))
    {
        for p in &v.pairs {
            let _ = maps.insert(UidKind::Object, &p.sop, &p.target.sop);
            let _ = maps.insert(UidKind::Object, &p.series, &p.target.series);
            if !p.study.is_empty() && !p.target.study.is_empty() {
                studies.entry(&p.study).or_default().insert(&p.target.study);
            }
            if !p.frame.is_empty() && !p.target.frame.is_empty() {
                frames.entry(&p.frame).or_default().insert(&p.target.frame);
            }
        }
    }
    let mut splits = Vec::new();
    for (kind, name, m) in [
        (UidKind::Study, "study", &studies),
        (UidKind::Frame, "frame", &frames),
    ] {
        for (from, to) in m {
            if to.len() == 1 {
                let _ = maps.insert(kind, from, to.iter().next().unwrap());
            } else {
                splits.push(Split {
                    kind: name,
                    source: from.to_string(),
                    targets: to.iter().map(|s| s.to_string()).collect(),
                });
            }
        }
    }
    (maps, splits)
}

/// The copies inside one folder: files with one SOP Instance UID (the same
/// bytes, or versions of one object), and pixel twins across its series
/// (the same CT re-exported with every plan).
#[derive(Clone, Debug, Default, Serialize)]
pub struct Inside {
    /// Byte-identical repeats of a SOP Instance UID, beyond the first.
    pub same_sop_identical: Vec<PathBuf>,
    /// Versions of one SOP Instance UID with other content: all but the
    /// largest (the most complete).
    pub same_sop_versions: Vec<PathBuf>,
    /// How many SOP Instance UIDs have versions.
    pub versioned_sops: usize,
    /// Files of series whose every image is a twin of a file of a series
    /// kept instead (the canonical copy).
    pub twin_copies: Vec<PathBuf>,
    /// Source series judged copies of another source series: (copy, kept).
    pub twin_series: Vec<(String, String)>,
}

/// Rank a path by `prefer`: the index of the first word (lower case) its
/// text contains, words not found after all of them.
fn rank(path: &Path, prefer: &[String]) -> usize {
    let s = path.to_string_lossy().to_lowercase();
    prefer
        .iter()
        .position(|w| !w.is_empty() && s.contains(&w.to_lowercase()))
        .unwrap_or(prefer.len())
}

/// The default order of preference between internal copies: a full
/// export over a case copy over the copies in plan folders.
pub fn default_prefer() -> Vec<String> {
    vec!["full".into(), "case".into(), "plan".into()]
}

/// Find the copies inside `src`. Among pixel-twin series the one kept is
/// the best ranked by `prefer` ([`default_prefer`]), then the one with
/// more files, then by path.
pub fn inside(src: &[Facts], prefer: &[String]) -> Inside {
    let mut out = Inside::default();
    // One SOP UID in several files.
    let mut by_sop: BTreeMap<&str, Vec<&Facts>> = BTreeMap::new();
    for f in src.iter().filter(|f| !f.sop.is_empty()) {
        by_sop.entry(f.sop.as_str()).or_default().push(f);
    }
    for list in by_sop.values().filter(|l| l.len() > 1) {
        let mut list = list.clone();
        list.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.path.cmp(&b.path)));
        let keep = list[0];
        let mut versions = false;
        for f in &list[1..] {
            if same_content(&keep.path, &f.path, &[]).unwrap_or(false) {
                out.same_sop_identical.push(f.path.clone());
            } else {
                versions = true;
                out.same_sop_versions.push(f.path.clone());
            }
        }
        out.versioned_sops += versions as usize;
    }
    // Pixel twins across series: a series all of whose candidate images
    // have a twin in one other series.
    let mut series: BTreeMap<&str, Vec<&Facts>> = BTreeMap::new();
    for f in src.iter().filter(|f| f.twin_candidate()) {
        series.entry(f.series.as_str()).or_default().push(f);
    }
    type Key = (Option<String>, Option<String>);
    let key = |f: &Facts| -> Key { (f.pixel_hash.clone(), position_key(f.position)) };
    let mut holders: HashMap<Key, BTreeSet<&str>> = HashMap::new();
    for (s, files) in &series {
        for f in files {
            holders.entry(key(f)).or_default().insert(s);
        }
    }
    let first_path = |s: &str| {
        series[s]
            .iter()
            .map(|f| f.path.clone())
            .min()
            .unwrap_or_default()
    };
    // A series ranks by its best-ranked file (a full export's folder may
    // hold other folders too), then by size, then by path.
    let order = |s: &str| {
        let best = series[s]
            .iter()
            .map(|f| rank(&f.path, prefer))
            .min()
            .unwrap_or(prefer.len());
        (best, std::cmp::Reverse(series[s].len()), first_path(s))
    };
    for (s, files) in &series {
        // The other series that hold every image of this one.
        let mut common: Option<BTreeSet<&str>> = None;
        for f in files {
            let others: BTreeSet<&str> = holders[&key(f)]
                .iter()
                .copied()
                .filter(|o| o != s)
                .collect();
            common = Some(match common {
                None => others,
                Some(c) => c.intersection(&others).copied().collect(),
            });
        }
        // Kept instead: the best of those, when it is better than this one.
        let kept = common
            .unwrap_or_default()
            .into_iter()
            .filter(|o| series[o].len() >= files.len())
            .min_by_key(|o| order(o));
        if let Some(k) = kept.filter(|k| order(k) < order(s)) {
            out.twin_series.push((s.to_string(), k.to_string()));
            out.twin_copies.extend(files.iter().map(|f| f.path.clone()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::raw::tests::{ex, part10};
    use super::super::raw::{encode_text, TS_EXPLICIT};
    use super::*;
    use dicom_core::VR;

    fn ct(
        dir: &Path,
        name: &str,
        sop: &str,
        series: &str,
        z: f64,
        pixels: u8,
        pid: &str,
    ) -> PathBuf {
        let mut d = Vec::new();
        d.extend(ex(
            Tag(0x0008, 0x0016),
            "UI",
            &encode_text(VR::UI, crate::sop::SOP_CT),
        ));
        d.extend(ex(Tag(0x0008, 0x0018), "UI", &encode_text(VR::UI, sop)));
        d.extend(ex(Tag(0x0010, 0x0020), "LO", &encode_text(VR::LO, pid)));
        d.extend(ex(Tag(0x0020, 0x000E), "UI", &encode_text(VR::UI, series)));
        d.extend(ex(
            Tag(0x0020, 0x0032),
            "DS",
            &encode_text(VR::DS, &format!("0\\0\\{z}")),
        ));
        d.extend(ex(Tag(0x0028, 0x0010), "US", &16u16.to_le_bytes()));
        d.extend(ex(Tag(0x0028, 0x0011), "US", &16u16.to_le_bytes()));
        d.extend(ex(Tag(0x0028, 0x0100), "US", &8u16.to_le_bytes()));
        d.extend(ex(
            Tag(0x7FE0, 0x0010),
            "OB",
            &vec![pixels.wrapping_add(z as u8); 256],
        ));
        let p = dir.join(name);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
        p
    }

    #[test]
    fn identical_twins_partial_and_new_series_are_told_apart() {
        let d = std::env::temp_dir().join("rds_dupes");
        let _ = std::fs::remove_dir_all(&d);
        let (arc, src) = (d.join("archive"), d.join("import"));
        // The archive: series A (3 slices) and B (2 slices).
        let mut archive = Vec::new();
        for z in 0..3 {
            archive.push(ct(
                &arc,
                &format!("a{z}"),
                &format!("1.1.{z}"),
                "1.1",
                z as f64,
                10,
                "P01-001",
            ));
        }
        for z in 0..2 {
            archive.push(ct(
                &arc,
                &format!("b{z}"),
                &format!("1.2.{z}"),
                "1.2",
                z as f64,
                50,
                "P01-002",
            ));
        }
        // The import: A again under another ID (exact but for PatientID);
        // B re-anonymized (new UIDs, same pixels); C half new.
        let mut source = Vec::new();
        for z in 0..3 {
            source.push(ct(
                &src,
                &format!("a{z}"),
                &format!("1.1.{z}"),
                "1.1",
                z as f64,
                10,
                "other",
            ));
        }
        for z in 0..2 {
            source.push(ct(
                &src,
                &format!("b{z}"),
                &format!("9.2.{z}"),
                "9.2",
                z as f64,
                50,
                "x",
            ));
        }
        source.push(ct(&src, "c0", "9.3.0", "9.3", 0.0, 50, "x"));
        source.push(ct(&src, "c1", "9.3.1", "9.3", 1.0, 99, "x"));
        source.push(ct(&src, "n0", "9.4.0", "9.4", 0.0, 7, "x"));
        let mut index = PixelIndex::default();
        let (against, read) = index.refresh(&archive, &Progress::default()).unwrap();
        assert_eq!(read, 5);
        assert_eq!(
            index.refresh(&archive, &Progress::default()).unwrap().1,
            0,
            "read once"
        );
        let v = find(
            &source,
            &against,
            &default_ignored(),
            &any_patient,
            &Progress::default(),
        )
        .unwrap();
        let of = |s: &str| v.iter().find(|x| x.series == s).unwrap();
        assert_eq!(of("1.1").verdict, "identical");
        assert_eq!(of("1.1").exact, 3);
        assert_eq!(of("9.2").verdict, "pixel-twin");
        assert_eq!(of("9.2").matches.get("1.2"), Some(&2));
        assert_eq!(of("9.3").verdict, "partial");
        assert_eq!(of("9.4").verdict, "new");
        // c0 has b0's pixels at b0's position: a twin inside the source too.
        assert_eq!(of("9.3").internal_twins, 1);
        // PatientID counts when it is not left out.
        assert!(!same_content(&source[0], &archive[0], &[]).unwrap());
    }

    /// A RayStation-style export: the full export's CT re-exported in a plan
    /// folder under new UIDs, and one SOP written twice (once the same, once
    /// a fuller version).
    #[test]
    fn copies_inside_an_import_keep_the_full_export() {
        let d = std::env::temp_dir().join("rds_dupes_inside");
        let _ = std::fs::remove_dir_all(&d);
        let mut files = Vec::new();
        for z in 0..3 {
            files.push(ct(
                &d.join("full"),
                &format!("f{z}"),
                &format!("5.1.{z}"),
                "5.1",
                z as f64,
                30,
                "x",
            ));
            files.push(ct(
                &d.join("plan_1"),
                &format!("p{z}"),
                &format!("6.1.{z}"),
                "6.1",
                z as f64,
                30,
                "x",
            ));
        }
        // The same file twice, and a version of another with more in it.
        let twice = d.join("again/f0");
        std::fs::create_dir_all(twice.parent().unwrap()).unwrap();
        std::fs::copy(&files[0], &twice).unwrap();
        files.push(twice.clone());
        let fuller = ct(
            &d.join("again"),
            "f1",
            "5.1.1",
            "5.1",
            1.0,
            30,
            "x-longer-id",
        );
        files.push(fuller);
        let src: Vec<Facts> = files.iter().map(|f| facts::read(f, true)).collect();
        let r = inside(&src, &default_prefer());
        assert_eq!(r.twin_series, vec![("6.1".to_string(), "5.1".to_string())]);
        assert_eq!(r.twin_copies.len(), 3);
        assert!(r
            .twin_copies
            .iter()
            .all(|p| p.to_string_lossy().contains("plan_1")));
        assert_eq!(r.same_sop_identical.len(), 1);
        assert_eq!(r.same_sop_versions.len(), 1);
        assert_eq!(r.versioned_sops, 1);
    }
}
