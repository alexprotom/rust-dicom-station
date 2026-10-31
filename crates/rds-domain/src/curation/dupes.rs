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
    pub hash: Option<String>,
    /// ImagePositionPatient, rounded to 0.001 mm, as text.
    pub position: Option<String>,
    /// Kept out of the pixel-twin logic ([`Facts::twin_candidate`]).
    pub candidate: bool,
}

/// Path → what it held when last read.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PixelIndex {
    pub files: BTreeMap<String, Indexed>,
}

pub const INDEX_FILE: &str = "pixel-index.json";

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
            .unwrap_or_default()
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
}

/// Find the copies of `source`'s files in `against`.
pub fn find(
    source: &[PathBuf],
    against: &[(PathBuf, Indexed)],
    ignore: &[Tag],
    p: &Progress,
) -> anyhow::Result<Vec<SeriesVerdict>> {
    p.set("Reading the source");
    let src: Vec<Facts> = source.par_iter().map(|f| facts::read(f, true)).collect();
    if p.cancelled() {
        anyhow::bail!(CANCELLED);
    }
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
    p.set("Comparing");
    // Per file: exact, same uid other content, twin of series, or new.
    enum Kind {
        Exact(String),
        OtherContent,
        Twin(String),
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
                    Ok(true) => Kind::Exact(a.1.series.clone()),
                    _ => Kind::OtherContent,
                };
            }
            if f.twin_candidate() {
                if let (Some(h), Some(pos)) = (&f.pixel_hash, position_key(f.position)) {
                    if let Some(list) = by_pixels.get(&(h.as_str(), pos.as_str())) {
                        let series: BTreeSet<&str> =
                            list.iter().map(|a| a.1.series.as_str()).collect();
                        if let Some(s) = series.iter().next().filter(|_| series.len() == 1) {
                            return Kind::Twin(s.to_string());
                        }
                        // Two archive series with these pixels at this
                        // position: a twin of each, of neither for sure.
                        return Kind::Twin(format!("{} series", series.len()));
                    }
                }
            }
            Kind::New
        })
        .collect();
    let mut by_series: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, f) in src.iter().enumerate() {
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
        };
        for &i in &idx {
            let f = &src[i];
            match &kinds[i] {
                Kind::Exact(s) => {
                    v.exact += 1;
                    *v.matches.entry(s.clone()).or_default() += 1;
                    v.copies.push(f.path.clone());
                }
                Kind::Twin(s) => {
                    v.twins += 1;
                    *v.matches.entry(s.clone()).or_default() += 1;
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
    Ok(out)
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
        let v = find(&source, &against, &default_ignored(), &Progress::default()).unwrap();
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
}
