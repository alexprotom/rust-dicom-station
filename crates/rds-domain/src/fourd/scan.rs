//! Finding the 4D acquisitions of a study from its headers alone, before
//! anything is loaded: what the PACS catalog lists and the viewer groups.
//!
//! The description rule of the module above (a `30%` or `phase 3` in
//! every phase's SeriesDescription) finds the 4DCTs whose descriptions
//! survived. Half of what a clinical archive holds did not: an anonymizing
//! export (RayStation 10.1) strips the description, the protocol, the dates
//! and the scanner's private tags, and no scanner of the trial this was
//! written for fills the standard 4D attributes. What is left still says it,
//! once the series are put side by side:
//!
//! - **the phase**, first match wins: NominalPercentageOfRespiratoryPhase,
//!   TemporalPositionIdentifier, else a percentage in the description,
//!   ImageComments (Philips: ` 30% Amplitude`) or one of Philips' private
//!   texts `(00E1,1040)`, `(01E1,101E)`, `(01F1,1041)`, as `T=30%` (GE
//!   Advantage 4D), `30% Amplitude` / `30% AMP`, `phase 3` or `30% Ex|In`;
//! - **a reconstruction** of the set: `UnTagged` (the ungated series of the
//!   same scan), `AVG` / `Average` / `mean` / `AverageIntensity`, `MIP` /
//!   `MaximumIntensity`, `MinIP` / `MinimumIntensity`;
//! - **4D context**: a ProtocolName with `4D`, Philips' `(01F1,104E)`
//!   `Resp. 4D CT`, the GE `Advantage 4D` workstation.
//!
//! Series of a study are grouped by series-number block (`number / 100`:
//! a Philips scan numbers its phases 201, 202, ..., the next scan of the
//! day 301, ...), then by reconstruction (the standard one and the
//! Extended-FOV one, `DERIVED`, are two sets of one scan), then by
//! acquisition: every phase of a Philips scan carries the same
//! AcquisitionDate and AcquisitionTime (to the minute), which keeps a 4DCT
//! apart from the daily CTs sharing its block; a member stripped of its
//! dates joins the dated piece of the same geometry (Rows, first and last
//! slice position, slice count), or forms a piece of its own per geometry.
//! GE stamps every phase with its own time: pieces of one block that share
//! a date and whose phase labels are disjoint are one set.
//!
//! A set is accepted on
//!
//! | evidence | rule |
//! |---|---|
//! | `phase-labels` | three or more distinct phases |
//! | `header-4d` | three or more members with 4D context, or a same-geometry set of eight with one |
//! | `same-acquisition` | eight members, unique series numbers, one geometry and one acquisition time |
//! | `geometry-only` | eight members, unique series numbers, one geometry and nothing else: *probable* |
//!
//! and never when a series number repeats inside it (a course of daily
//! CTs). A ten-phase set whose known labels are each ten times their
//! position gets the missing ones by order, marked *inferred* (`~30%`);
//! any other unlabelled phase is `t1`, `t2`, ... A projection numbered
//! below 100 (a scanner's `0` series) joins the study's set of the same
//! slice range, and a respiratory trace (`For Series: 201-222`) the set
//! whose phases it names.
//!
//! The series no header rule claims go through the description rule of
//! the module above, so a dataset grouped before is grouped as it was.
//! Everything is a function of the inputs, sorted by Series Instance UID
//! first: the same series give the same sets, the same group ids and the
//! same names, in the catalog and in the viewer alike. The rules and their
//! thresholds reproduce `detect2.py`, the script they were proven with on
//! 1,724 series of the trial (35 sets, 30 scans in 25 studies).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::LazyLock;

use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::loader::{i32_of, items_of, str_of};

/// The private elements read: three that carry the phase as text, and the
/// one that says `Resp. 4D CT`. Read as text, never decoded otherwise; a
/// malformed Philips element beside them, `(01F1,1026)`, is never touched.
pub const PRIVATE_TAGS: [(u16, u16); 4] = [
    (0x00E1, 0x1040),
    (0x01E1, 0x101E),
    (0x01F1, 0x1041),
    (0x01F1, 0x104E),
];
/// Those of [`PRIVATE_TAGS`] that may name the phase, as [`tag_key`] spells them.
const PHASE_TAGS: [&str; 3] = ["00E1,1040", "01E1,101E", "01F1,1041"];

/// `(00E1,1040)` as the key a [`SeriesHead`] files it under: `00E1,1040`.
pub fn tag_key(group: u16, element: u16) -> String {
    format!("{group:04X},{element:04X}")
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// What the detection reads of a series' first file (the catalog keeps it
/// per CT and structure-set series, so that the rules run again without a
/// header read). Every field is empty when the header does not say.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SeriesHead {
    /// ImageType, its values joined by `\`.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub image_type: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub protocol: String,
    /// ImageComments.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub comments: String,
    /// DerivationDescription.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub derivation: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub acq_date: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub acq_time: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub series_date: String,
    #[serde(skip_serializing_if = "is_zero")]
    pub rows: u32,
    /// NominalPercentageOfRespiratoryPhase, as written.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub nominal: String,
    /// TemporalPositionIdentifier (else TemporalPositionIndex), as written.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub temporal: String,
    /// The [`PRIVATE_TAGS`] present, by [`tag_key`].
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub private: BTreeMap<String, String>,
    /// A structure set's referenced image series (Series Instance UIDs).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub refs: Vec<String>,
}

/// A private element's value as text: its bytes, whatever VR it came
/// with, up to the first NUL, trimmed. Over 120 bytes is no label.
fn private_text(obj: &InMemDicomObject, group: u16, element: u16) -> Option<String> {
    let e = obj.element(dicom_core::Tag(group, element)).ok()?;
    let bytes = e.value().to_bytes().ok()?;
    let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
    let s = String::from_utf8_lossy(&bytes[..end]).trim().to_string();
    (!s.is_empty() && s.len() < 120).then_some(s)
}

impl SeriesHead {
    /// Read from a file's header.
    pub fn read(obj: &InMemDicomObject) -> SeriesHead {
        let s = |t| str_of(obj, t).unwrap_or_default();
        let mut private = BTreeMap::new();
        for (g, e) in PRIVATE_TAGS {
            if let Some(v) = private_text(obj, g, e) {
                private.insert(tag_key(g, e), v);
            }
        }
        let mut refs: Vec<String> = Vec::new();
        for f in items_of(obj, tags::REFERENCED_FRAME_OF_REFERENCE_SEQUENCE).unwrap_or(&[]) {
            for st in items_of(f, tags::RT_REFERENCED_STUDY_SEQUENCE).unwrap_or(&[]) {
                for se in items_of(st, tags::RT_REFERENCED_SERIES_SEQUENCE).unwrap_or(&[]) {
                    if let Some(u) = str_of(se, tags::SERIES_INSTANCE_UID) {
                        if !refs.contains(&u) {
                            refs.push(u);
                        }
                    }
                }
            }
        }
        let temporal = {
            let t = s(tags::TEMPORAL_POSITION_IDENTIFIER);
            if t.is_empty() {
                s(tags::TEMPORAL_POSITION_INDEX)
            } else {
                t
            }
        };
        SeriesHead {
            image_type: s(tags::IMAGE_TYPE),
            model: s(tags::MANUFACTURER_MODEL_NAME),
            protocol: s(tags::PROTOCOL_NAME),
            comments: s(tags::IMAGE_COMMENTS),
            derivation: s(tags::DERIVATION_DESCRIPTION),
            acq_date: s(tags::ACQUISITION_DATE),
            acq_time: s(tags::ACQUISITION_TIME),
            series_date: s(tags::SERIES_DATE),
            rows: i32_of(obj, tags::ROWS).map_or(0, |r| r.max(0) as u32),
            nominal: s(tags::NOMINAL_PERCENTAGE_OF_RESPIRATORY_PHASE),
            temporal,
            private,
            refs,
        }
    }
}

/// A slice position as the detection compares it: z in hundredths of a
/// millimetre.
pub fn z_of(obj: &InMemDicomObject) -> Option<i64> {
    let v = crate::loader::f64s_of(obj, tags::IMAGE_POSITION_PATIENT)?;
    (v.len() == 3 && v[2].is_finite()).then(|| (v[2] * 100.0).round() as i64)
}

/// The slice positions of a series: first and last (hundredths of a
/// millimetre) and how many files are positioned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Z {
    pub min: i64,
    pub max: i64,
    pub n: usize,
}

impl Z {
    pub fn of(zs: impl IntoIterator<Item = i64>) -> Option<Z> {
        let mut out: Option<Z> = None;
        for z in zs {
            out = Some(match out {
                None => Z {
                    min: z,
                    max: z,
                    n: 1,
                },
                Some(o) => Z {
                    min: o.min.min(z),
                    max: o.max.max(z),
                    n: o.n + 1,
                },
            });
        }
        out
    }
}

/// One series as the detection sees it.
#[derive(Clone, Copy, Debug)]
pub struct Input<'a> {
    pub uid: &'a str,
    pub study: &'a str,
    pub modality: &'a str,
    pub number: Option<i64>,
    pub description: &'a str,
    /// What its first file says; `None`: only the description rule.
    pub head: Option<&'a SeriesHead>,
    /// TemporalPositionIdentifier, where the caller read it without a head.
    pub temporal: Option<i64>,
    pub z: Option<Z>,
    /// An image series (what the description rule groups).
    pub image: bool,
}

/// What a member is to its set.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum Kind {
    #[default]
    #[serde(rename = "phase")]
    Phase,
    /// The ungated reconstruction of the same scan.
    #[serde(rename = "untagged")]
    Untagged,
    #[serde(rename = "AVG")]
    Avg,
    #[serde(rename = "MIP")]
    Mip,
    #[serde(rename = "MinIP")]
    MinIp,
    /// The scanner's respiratory trace.
    #[serde(rename = "trace")]
    Trace,
}

impl Kind {
    pub fn word(self) -> &'static str {
        match self {
            Kind::Phase => "phase",
            Kind::Untagged => "untagged",
            Kind::Avg => "AVG",
            Kind::Mip => "MIP",
            Kind::MinIp => "MinIP",
            Kind::Trace => "trace",
        }
    }

    pub fn of_word(w: &str) -> Option<Kind> {
        [
            Kind::Phase,
            Kind::Untagged,
            Kind::Avg,
            Kind::Mip,
            Kind::MinIp,
            Kind::Trace,
        ]
        .into_iter()
        .find(|k| k.word().eq_ignore_ascii_case(w.trim()))
    }

    /// A reconstruction of the phases (not a phase, not the trace).
    pub fn is_recon(self) -> bool {
        matches!(self, Kind::Untagged | Kind::Avg | Kind::Mip | Kind::MinIp)
    }
}

/// Why a set was accepted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Evidence {
    #[default]
    #[serde(rename = "phase-labels")]
    PhaseLabels,
    #[serde(rename = "header-4d")]
    Header4d,
    #[serde(rename = "same-acquisition")]
    SameAcquisition,
    /// Probable only: shown as `4DCT?` until a person confirms it.
    #[serde(rename = "geometry-only")]
    GeometryOnly,
    /// The phases of the description rule.
    #[serde(rename = "description")]
    Description,
    /// Made or confirmed by hand.
    #[serde(rename = "custom")]
    Custom,
}

impl Evidence {
    pub fn word(self) -> &'static str {
        match self {
            Evidence::PhaseLabels => "phase-labels",
            Evidence::Header4d => "header-4d",
            Evidence::SameAcquisition => "same-acquisition",
            Evidence::GeometryOnly => "geometry-only",
            Evidence::Description => "description",
            Evidence::Custom => "custom",
        }
    }

    /// A set this evidence alone makes a 4DCT (all but `geometry-only`).
    pub fn confirmed(self) -> bool {
        self != Evidence::GeometryOnly
    }
}

/// One member of a set found.
#[derive(Clone, Debug, PartialEq)]
pub struct FoundMember {
    pub series: String,
    /// SeriesNumber; -1 when it has none.
    pub number: i64,
    pub kind: Kind,
    /// `30%`, `t3`, `AVG`: what it is called within the set.
    pub label: String,
    pub percent: Option<f64>,
    /// The phase was inferred from the series' place in the set.
    pub inferred: bool,
}

/// One 4D set found.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    /// `<study>/b2/20240610T1408` (`/efov` for the Extended-FOV set): the
    /// same for the same series wherever the detection runs.
    pub id: String,
    pub study: String,
    /// `4DCT 2024-06-10 (10 phases + 1)`; `4DCT?` while probable. Made of
    /// dates and counts only.
    pub name: String,
    /// The series-number block (-1 below 100, and for the description rule).
    pub block: i64,
    /// The Extended-FOV reconstruction of the scan.
    pub efov: bool,
    pub evidence: Evidence,
    /// `YYYYMMDD`, empty when no member says.
    pub date: String,
    /// The scanner model, as the first member names it.
    pub scanner: String,
    pub phases: usize,
    /// Slices of a phase.
    pub slices: usize,
    /// The scan the set is of, from 1, within its study: the standard and
    /// Extended-FOV sets of one block and date are one scan.
    pub scan: usize,
    /// Phases in order, then the reconstructions, then the trace.
    pub members: Vec<FoundMember>,
}

impl Found {
    pub fn confirmed(&self) -> bool {
        self.evidence.confirmed()
    }
}

// ---- the rules -----------------------------------------------------------------

/// A phase as read: a percentage, or a temporal position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum PhaseKey {
    Pct(i64),
    Index(i64),
}

impl PhaseKey {
    fn order(self) -> i64 {
        match self {
            PhaseKey::Pct(p) | PhaseKey::Index(p) => p,
        }
    }
}

static RE_T: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"T\s*=\s*(\d{1,3})\s*%").unwrap());
static RE_AMP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(\d{1,3})\s*%\s*(amplitude|amp)\b").unwrap());
static RE_PHASE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)phase[\s_]*(\d{1,3})").unwrap());
static RE_EXIN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(\d{1,3})\s*%\s*(ex|in)\b").unwrap());
static RE_MIP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)MaximumIntensity|\bMIP\b").unwrap());
static RE_MINIP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)MinimumIntensity|\bMinIP\b").unwrap());
static RE_AVG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)AverageIntensity|\bAVG\b|\bAverage\b|\bmean\b").unwrap());
static RE_UNTAGGED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)untagged").unwrap());

/// The phase of a series, first match wins (see the module doc).
fn phase_of(x: &Input, h: &SeriesHead) -> Option<PhaseKey> {
    if let Ok(p) = h.nominal.trim().parse::<f64>() {
        if p.is_finite() {
            return Some(PhaseKey::Pct(p.round() as i64));
        }
    }
    if let Ok(t) = h.temporal.trim().parse::<i64>() {
        return Some(PhaseKey::Index(t));
    }
    let mut texts: Vec<&str> = vec![x.description, h.comments.as_str()];
    for t in PHASE_TAGS {
        if let Some(v) = h.private.get(t) {
            texts.push(v);
        }
    }
    for s in texts {
        for re in [&*RE_T, &*RE_AMP, &*RE_PHASE, &*RE_EXIN] {
            if let Some(n) = re
                .captures(s)
                .and_then(|c| c.get(1))
                .and_then(|m| m.as_str().parse::<i64>().ok())
            {
                return Some(PhaseKey::Pct(n));
            }
        }
    }
    None
}

/// The reconstruction a series is, if it is one, from its description and
/// its first file.
pub fn recon_of(description: &str, h: &SeriesHead) -> Option<Kind> {
    let s = format!("{description} {} {}", h.comments, h.derivation);
    if RE_MIP.is_match(&s) {
        Some(Kind::Mip)
    } else if RE_MINIP.is_match(&s) {
        Some(Kind::MinIp)
    } else if RE_AVG.is_match(&s) {
        Some(Kind::Avg)
    } else if RE_UNTAGGED.is_match(&s) {
        Some(Kind::Untagged)
    } else {
        None
    }
}

/// Does the header say the scan is 4D (not which phase)?
pub fn has_context(h: &SeriesHead) -> bool {
    h.protocol.contains("4D")
        || h.private.values().any(|v| v.contains("4D"))
        || h.model.contains("Advantage 4D")
}

/// Rows and slice positions: what two phases of one scan share.
type Geom = (u32, Option<Z>);

/// A candidate series.
struct X<'a> {
    input: &'a Input<'a>,
    head: &'a SeriesHead,
    num: i64,
    phase: Option<PhaseKey>,
    recon: Option<Kind>,
    ctx: bool,
    efov: bool,
}

impl X<'_> {
    fn geom(&self) -> Geom {
        (self.head.rows, self.input.z)
    }

    /// AcquisitionDate and AcquisitionTime to the minute, when both are there.
    fn acq(&self) -> Option<(String, String)> {
        let d = self.head.acq_date.as_str();
        let t: String = self.head.acq_time.chars().take(4).collect();
        (!d.is_empty() && !t.is_empty()).then(|| (d.to_string(), t))
    }
}

/// How a block's series are split into acquisitions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum SubKey {
    Acq(String, String),
    Geom(Geom),
}

/// A map that remembers the order its keys came in.
struct Ordered<K, V> {
    keys: Vec<K>,
    at: HashMap<K, usize>,
    values: Vec<V>,
}

impl<K: Clone + Eq + std::hash::Hash, V: Default> Ordered<K, V> {
    fn new() -> Self {
        Ordered {
            keys: Vec::new(),
            at: HashMap::new(),
            values: Vec::new(),
        }
    }

    fn entry(&mut self, k: K) -> &mut V {
        let i = match self.at.get(&k) {
            Some(i) => *i,
            None => {
                self.at.insert(k.clone(), self.keys.len());
                self.keys.push(k);
                self.values.push(V::default());
                self.keys.len() - 1
            }
        };
        &mut self.values[i]
    }

    fn pairs(self) -> impl Iterator<Item = (K, V)> {
        self.keys.into_iter().zip(self.values)
    }
}

/// A set being put together.
struct Piece<'a> {
    study: &'a str,
    block: i64,
    efov: bool,
    key: SubKey,
    xs: Vec<&'a X<'a>>,
    evidence: Option<Evidence>,
    labs: BTreeSet<PhaseKey>,
    dates: BTreeSet<String>,
}

/// The most frequent value and how often it comes; ties go to the first.
fn most_common<T: Eq + std::hash::Hash + Clone>(
    it: impl IntoIterator<Item = T>,
) -> Option<(T, usize)> {
    let mut counts: Ordered<T, usize> = Ordered::new();
    for v in it {
        *counts.entry(v) += 1;
    }
    let mut best: Option<(T, usize)> = None;
    for (k, n) in counts.pairs() {
        if best.as_ref().is_none_or(|(_, b)| n > *b) {
            best = Some((k, n));
        }
    }
    best
}

fn evidence_of(xs: &[&X]) -> Option<Evidence> {
    let labels: BTreeSet<PhaseKey> = xs.iter().filter_map(|x| x.phase).collect();
    let ctx = xs.iter().filter(|x| x.ctx).count();
    if labels.len() >= 3 {
        return Some(Evidence::PhaseLabels);
    }
    if ctx >= 3 {
        return Some(Evidence::Header4d);
    }
    let nums: BTreeSet<i64> = xs.iter().map(|x| x.num).collect();
    let geo = most_common(xs.iter().map(|x| x.geom())).map_or(0, |(_, n)| n);
    if xs.len() >= 8 && nums.len() == xs.len() && geo >= 8 {
        if ctx > 0 {
            return Some(Evidence::Header4d);
        }
        let first = xs[0].acq();
        if first.is_some() && xs.iter().all(|x| x.acq() == first) {
            return Some(Evidence::SameAcquisition);
        }
        return Some(Evidence::GeometryOnly);
    }
    None
}

/// `20240610` as `2024-06-10`, `undated` for nothing.
fn show(date: &str) -> String {
    if date.is_empty() {
        "undated".into()
    } else {
        crate::archive::show_date(date)
    }
}

/// The name of a set: dates and counts, nothing from a header's text.
pub fn name_of(confirmed: bool, date: &str, efov: bool, phases: usize, extras: usize) -> String {
    format!(
        "4DCT{} {}{} ({phases} phases{})",
        if confirmed { "" } else { "?" },
        show(date),
        if efov { " EFOV" } else { "" },
        if extras > 0 {
            format!(" + {extras}")
        } else {
            String::new()
        }
    )
}

/// A short, stable hash of a text (FNV-1a), for the id of a set the
/// description rule found.
fn hash8(s: &str) -> String {
    let mut h: u32 = 0x811c_9dc5;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    format!("{h:08x}")
}

/// Find the 4D sets among `series` (see the module doc).
pub fn detect(series: &[Input]) -> Vec<Found> {
    let mut order: Vec<&Input> = series.iter().collect();
    order.sort_by(|a, b| a.uid.cmp(b.uid));
    order.dedup_by(|a, b| a.uid == b.uid);

    let xs: Vec<X> = order
        .iter()
        .filter_map(|x| {
            let h = x.head?;
            if x.modality != "CT"
                || h.image_type.contains("LOCALIZER")
                || x.z.is_none_or(|z| z.n < 2)
            {
                return None;
            }
            Some(X {
                input: x,
                head: h,
                num: x.number.unwrap_or(-1),
                phase: phase_of(x, h),
                recon: recon_of(x.description, h),
                ctx: has_context(h),
                efov: h.image_type.contains("DERIVED") || x.description.contains("Extended FOV"),
            })
        })
        .collect();

    // Study, block, reconstruction; the projections numbered below 100 also
    // wait to join a set of their study.
    let mut by_block: Ordered<(&str, i64, bool), Vec<&X>> = Ordered::new();
    let mut loose: Ordered<&str, Vec<&X>> = Ordered::new();
    for x in &xs {
        if matches!(x.recon, Some(Kind::Mip | Kind::Avg | Kind::MinIp)) && x.num < 100 {
            loose.entry(x.input.study).push(x);
        }
        let block = if x.num >= 100 { x.num / 100 } else { -1 };
        by_block.entry((x.input.study, block, x.efov)).push(x);
    }

    // A block into acquisitions: by date and time, the undated by geometry.
    let mut pieces: Vec<Piece> = Vec::new();
    for ((study, block, efov), members) in by_block.pairs() {
        let mut sub: Ordered<SubKey, Vec<&X>> = Ordered::new();
        let mut geo_of: HashMap<Geom, (String, String)> = HashMap::new();
        for x in &members {
            if let Some((d, t)) = x.acq() {
                sub.entry(SubKey::Acq(d.clone(), t.clone())).push(x);
                geo_of.entry(x.geom()).or_insert((d, t));
            }
        }
        for x in &members {
            if x.acq().is_none() {
                let key = match geo_of.get(&x.geom()) {
                    Some((d, t)) => SubKey::Acq(d.clone(), t.clone()),
                    None => SubKey::Geom(x.geom()),
                };
                sub.entry(key).push(x);
            }
        }
        for (key, mut v) in sub.pairs() {
            v.sort_by_key(|x| x.num);
            let evidence = evidence_of(&v);
            pieces.push(Piece {
                study,
                block,
                efov,
                key,
                labs: v.iter().filter_map(|x| x.phase).collect(),
                dates: v.iter().map(|x| x.head.acq_date.clone()).collect(),
                xs: v,
                evidence,
            });
        }
    }

    // GE: labelled pieces of one block, of one date, with disjoint phases.
    pieces.sort_by(|a, b| (a.study, a.block, !a.efov).cmp(&(b.study, b.block, !b.efov)));
    let mut sets: Vec<Piece> = Vec::new();
    for p in pieces {
        let into = sets.iter_mut().find(|m| {
            (m.study, m.block, m.efov) == (p.study, p.block, p.efov)
                && !p.labs.is_empty()
                && !m.labs.is_empty()
                && m.labs.is_disjoint(&p.labs)
                && p.dates == m.dates
        });
        match into {
            Some(m) => {
                m.xs.extend(p.xs);
                m.labs.extend(p.labs);
                m.evidence = m.evidence.or(p.evidence);
                m.xs.sort_by_key(|x| (x.phase.map_or(999, PhaseKey::order), x.num));
            }
            None => sets.push(p),
        }
    }
    for s in &mut sets {
        if s.labs.len() >= 3 && s.evidence.is_none() {
            s.evidence = Some(Evidence::PhaseLabels);
        }
        // A series number twice is a course of daily CTs, never a 4DCT.
        let nums: BTreeSet<i64> = s.xs.iter().map(|x| x.num).collect();
        if nums.len() < s.xs.len() {
            s.evidence = None;
        }
    }

    let mut out: Vec<Found> = Vec::new();
    let mut ids: BTreeSet<String> = BTreeSet::new();
    for s in sets.iter().filter(|s| s.evidence.is_some()) {
        let evidence = s.evidence.unwrap_or_default();
        let phases: Vec<&&X> = s.xs.iter().filter(|x| x.recon.is_none()).collect();
        let recons: Vec<&&X> = s.xs.iter().filter(|x| x.recon.is_some()).collect();
        if phases.is_empty() {
            continue;
        }
        // Ten phases whose known labels sit at ten times their place: the
        // others are inferred by order.
        let step = phases.len() == 10
            && phases
                .iter()
                .enumerate()
                .all(|(i, x)| x.phase.is_none_or(|p| p == PhaseKey::Pct(10 * i as i64)));
        let mut members: Vec<FoundMember> = Vec::new();
        for (i, x) in phases.iter().enumerate() {
            let (label, percent, inferred) = match x.phase {
                Some(PhaseKey::Pct(p)) => (format!("{p}%"), Some(p as f64), false),
                Some(PhaseKey::Index(t)) => (format!("t{t}"), None, false),
                None if step => (format!("{}%", 10 * i), Some(10.0 * i as f64), true),
                None => (format!("t{}", i + 1), None, false),
            };
            members.push(FoundMember {
                series: x.input.uid.to_string(),
                number: x.num,
                kind: Kind::Phase,
                label,
                percent,
                inferred,
            });
        }
        for x in &recons {
            let kind = x.recon.unwrap_or(Kind::Untagged);
            members.push(FoundMember {
                series: x.input.uid.to_string(),
                number: x.num,
                kind,
                label: kind.word().to_string(),
                percent: None,
                inferred: false,
            });
        }
        let date =
            s.xs.iter()
                .map(|x| {
                    if x.head.acq_date.is_empty() {
                        x.head.series_date.clone()
                    } else {
                        x.head.acq_date.clone()
                    }
                })
                .filter(|d| !d.is_empty())
                .min()
                .unwrap_or_default();
        let slices =
            most_common(phases.iter().map(|x| x.input.z.map_or(0, |z| z.n))).map_or(0, |(n, _)| n);
        let key = match &s.key {
            SubKey::Acq(d, t) => format!("{d}T{t}"),
            SubKey::Geom((rows, z)) => match z {
                Some(z) => format!("g{rows}x{}z{}_{}", z.n, z.min, z.max),
                None => format!("g{rows}"),
            },
        };
        let mut id = format!(
            "{}/b{}{}/{key}",
            s.study,
            s.block,
            if s.efov { "/efov" } else { "" }
        );
        if !ids.insert(id.clone()) {
            let mut n = 2;
            while !ids.insert(format!("{id}#{n}")) {
                n += 1;
            }
            id = format!("{id}#{n}");
        }
        out.push(Found {
            id,
            study: s.study.to_string(),
            name: String::new(),
            block: s.block,
            efov: s.efov,
            evidence,
            date,
            scanner: s.xs[0].head.model.clone(),
            phases: phases.len(),
            slices,
            scan: 0,
            members,
        });
    }

    // A projection numbered below 100 joins its study's set of the same
    // slice range (the first one when none has it).
    for (study, v) in loose.pairs() {
        for x in v {
            if out
                .iter()
                .any(|f| f.members.iter().any(|m| m.series == x.input.uid))
            {
                continue;
            }
            let range = x.input.z.map(|z| (z.min, z.max));
            let same = |f: &Found| {
                f.members.iter().any(|m| {
                    xs.iter()
                        .find(|y| y.input.uid == m.series)
                        .is_some_and(|y| y.input.z.map(|z| (z.min, z.max)) == range)
                })
            };
            let target = out
                .iter()
                .position(|f| f.study == study && same(f))
                .or_else(|| out.iter().position(|f| f.study == study));
            if let Some(t) = target {
                let kind = x.recon.unwrap_or(Kind::Avg);
                out[t].members.push(FoundMember {
                    series: x.input.uid.to_string(),
                    number: x.num,
                    kind,
                    label: kind.word().to_string(),
                    percent: None,
                    inferred: false,
                });
            }
        }
    }

    // The respiratory trace: a non-image series naming the phases' numbers.
    for x in order
        .iter()
        .filter(|x| !x.image && x.modality != "RTSTRUCT")
    {
        let Some((a, b)) = super::series_range(x.description) else {
            continue;
        };
        if let Some(f) = out.iter_mut().find(|f| {
            f.study == x.study
                && f.members
                    .iter()
                    .any(|m| m.kind == Kind::Phase && (a..=b).contains(&m.number))
        }) {
            f.members.push(FoundMember {
                series: x.uid.to_string(),
                number: x.number.unwrap_or(-1),
                kind: Kind::Trace,
                label: Kind::Trace.word().to_string(),
                percent: None,
                inferred: false,
            });
        }
    }

    // What no header rule claimed: the description rule, as before.
    let claimed: BTreeSet<&str> = out
        .iter()
        .flat_map(|f| f.members.iter().map(|m| m.series.as_str()))
        .collect();
    let rest: Vec<super::Desc> = order
        .iter()
        .filter(|x| x.image && !claimed.contains(x.uid))
        .map(|x| super::Desc {
            uid: x.uid,
            study: x.study,
            modality: x.modality,
            description: x.description,
            temporal: x
                .head
                .and_then(|h| h.temporal.trim().parse::<i64>().ok())
                .or(x.temporal),
        })
        .collect();
    let numbers: HashMap<&str, i64> = order
        .iter()
        .map(|x| (x.uid, x.number.unwrap_or(-1)))
        .collect();
    for (g, stem) in super::detect_by_description(&rest) {
        let phases = g.phase_members().len();
        out.push(Found {
            id: format!(
                "{}/d/{}",
                g.study_uid,
                hash8(&format!("{}|{stem}", g.members[0].series_uid))
            ),
            study: g.study_uid.clone(),
            name: g.name.clone(),
            block: -1,
            efov: false,
            evidence: Evidence::Description,
            date: String::new(),
            scanner: String::new(),
            phases,
            slices: 0,
            scan: 0,
            members: g
                .members
                .iter()
                .map(|m| FoundMember {
                    series: m.series_uid.clone(),
                    number: numbers.get(m.series_uid.as_str()).copied().unwrap_or(-1),
                    kind: m.role.kind(),
                    label: m.label.clone(),
                    percent: m.percent.map(f64::from),
                    inferred: false,
                })
                .collect(),
        });
    }

    number_scans(&mut out);
    for f in out
        .iter_mut()
        .filter(|f| f.evidence != Evidence::Description)
    {
        let extras = f.members.iter().filter(|m| m.kind.is_recon()).count();
        f.name = name_of(f.confirmed(), &f.date, f.efov, f.phases, extras);
    }
    out
}

/// Number the scans of each study: the sets of one block and date are one
/// scan (a standard and an Extended-FOV reconstruction), in the order the
/// sets come.
pub fn number_scans(sets: &mut [Found]) {
    let mut seen: BTreeMap<&str, Vec<(i64, String)>> = BTreeMap::new();
    let mut scans: Vec<usize> = Vec::with_capacity(sets.len());
    for f in sets.iter() {
        let v = seen.entry(f.study.as_str()).or_default();
        // The description rule's sets are a scan each.
        let key = if f.block < 0 {
            (i64::MIN + v.len() as i64, f.id.clone())
        } else {
            (f.block, f.date.clone())
        };
        let n = match v.iter().position(|k| *k == key) {
            Some(i) => i + 1,
            None => {
                v.push(key);
                v.len()
            }
        };
        scans.push(n);
    }
    for (f, n) in sets.iter_mut().zip(scans) {
        f.scan = n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One series of a made-up Philips export.
    struct S {
        uid: String,
        number: i64,
        desc: String,
        head: SeriesHead,
        z: Z,
    }

    fn philips(uid: &str, number: i64, acq: &str, rows: u32, z: (i64, i64, usize)) -> S {
        S {
            uid: uid.into(),
            number,
            desc: String::new(),
            head: SeriesHead {
                image_type: "ORIGINAL\\PRIMARY\\AXIAL".into(),
                model: "Brilliance Big Bore".into(),
                acq_date: if acq.is_empty() {
                    String::new()
                } else {
                    "20240610".into()
                },
                acq_time: acq.into(),
                rows,
                ..Default::default()
            },
            z: Z {
                min: z.0,
                max: z.1,
                n: z.2,
            },
        }
    }

    fn run(v: &[S]) -> Vec<Found> {
        let inputs: Vec<Input> = v
            .iter()
            .map(|s| Input {
                uid: &s.uid,
                study: "1.2",
                modality: "CT",
                number: Some(s.number),
                description: &s.desc,
                head: Some(&s.head),
                temporal: None,
                z: Some(s.z),
                image: true,
            })
            .collect();
        detect(&inputs)
    }

    #[test]
    fn phase_texts_are_read_the_way_each_scanner_writes_them() {
        let x = |desc: &str, comments: &str, private: Option<(&str, &str)>| {
            let mut h = SeriesHead {
                comments: comments.into(),
                ..Default::default()
            };
            if let Some((t, v)) = private {
                h.private.insert(t.into(), v.into());
            }
            let s = S {
                uid: "u".into(),
                number: 1,
                desc: desc.into(),
                head: h,
                z: Z {
                    min: 0,
                    max: 0,
                    n: 2,
                },
            };
            let i = Input {
                uid: &s.uid,
                study: "1",
                modality: "CT",
                number: None,
                description: &s.desc,
                head: Some(&s.head),
                temporal: None,
                z: None,
                image: true,
            };
            phase_of(&i, &s.head)
        };
        assert_eq!(x("T=30%,PR=40", "", None), Some(PhaseKey::Pct(30)));
        assert_eq!(x("", " 30% Amplitude", None), Some(PhaseKey::Pct(30)));
        assert_eq!(
            x("", "", Some(("01F1,1041", "60% AMP"))),
            Some(PhaseKey::Pct(60))
        );
        assert_eq!(x("Lung phase 3", "", None), Some(PhaseKey::Pct(3)));
        assert_eq!(x("CT 50 % Ex", "", None), Some(PhaseKey::Pct(50)));
        assert_eq!(x("Thorax 3 mm", "", None), None);
        let h = SeriesHead {
            nominal: "20.0".into(),
            ..Default::default()
        };
        let s = S {
            uid: "u".into(),
            number: 1,
            desc: String::new(),
            head: h,
            z: Z {
                min: 0,
                max: 0,
                n: 2,
            },
        };
        let i = Input {
            uid: &s.uid,
            study: "1",
            modality: "CT",
            number: None,
            description: "",
            head: Some(&s.head),
            temporal: None,
            z: None,
            image: true,
        };
        assert_eq!(
            phase_of(&i, &s.head),
            Some(PhaseKey::Pct(20)),
            "the standard attribute first"
        );
    }

    /// A Philips 4DCT exported with its descriptions stripped: ten series
    /// numbered 201-210 of one acquisition time, and in the same block the
    /// daily CTs of a course, every one numbered 201.
    #[test]
    fn a_stripped_export_is_found_by_its_acquisition_and_daily_cts_are_not() {
        let mut v: Vec<S> = (0..10)
            .map(|i| {
                philips(
                    &format!("1.{i:02}"),
                    201 + i,
                    "140847",
                    512,
                    (-10000, 5000, 120),
                )
            })
            .collect();
        for d in 0..12 {
            let mut s = philips(
                &format!("2.{d:02}"),
                201,
                &format!("09{d:02}00"),
                512,
                (-9000, 4000, 100),
            );
            s.head.acq_date = format!("202406{:02}", 11 + d);
            v.push(s);
        }
        let found = run(&v);
        assert_eq!(found.len(), 1, "{found:#?}");
        let f = &found[0];
        assert_eq!(f.evidence, Evidence::SameAcquisition);
        assert_eq!(f.phases, 10);
        assert_eq!(f.id, "1.2/b2/20240610T1408");
        assert_eq!(f.name, "4DCT 2024-06-10 (10 phases)");
        // Ten phases and no label: each one's by its place, inferred.
        assert!(f.members.iter().all(|m| m.inferred));
        assert_eq!(f.members[3].label, "30%");
    }

    /// GE Advantage 4D: a time per phase, `T=n%` in the description; the
    /// pieces of one date and disjoint phases are one set.
    #[test]
    fn ge_phases_with_their_own_times_make_one_set() {
        let v: Vec<S> = (0..10)
            .map(|i| {
                let mut s = philips(
                    &format!("3.{i:02}"),
                    300 + i,
                    &format!("1200{i:02}"),
                    512,
                    (0, 3000, 150),
                );
                s.head.model = "Advantage 4D".into();
                s.head.acq_time = format!("12{i:02}00");
                s.desc = format!("T={}%,PR=4", i * 10);
                s
            })
            .collect();
        let found = run(&v);
        assert_eq!(found.len(), 1, "{found:#?}");
        assert_eq!(found[0].evidence, Evidence::PhaseLabels);
        let labels: Vec<&str> = found[0].members.iter().map(|m| m.label.as_str()).collect();
        assert_eq!(
            labels,
            ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%"]
        );
    }

    /// The standard and Extended-FOV reconstructions of one scan: two sets,
    /// one scan; the percent only in a private tag; a phase whose label was
    /// stripped is inferred from its place; the untagged series joins.
    #[test]
    fn an_efov_pair_is_one_scan_and_a_missing_label_is_inferred() {
        let mut v: Vec<S> = Vec::new();
        for i in 0..10 {
            let mut std = philips(
                &format!("4.{i:02}"),
                202 + 2 * i,
                "140847",
                512,
                (0, 3000, 150),
            );
            if i != 3 {
                std.head
                    .private
                    .insert("00E1,1040".into(), format!("{}% Amplitude", i * 10));
            }
            let mut efov = philips(
                &format!("5.{i:02}"),
                201 + 2 * i,
                "140847",
                741,
                (0, 3000, 150),
            );
            efov.head.image_type = "DERIVED\\SECONDARY\\AXIAL".into();
            efov.desc = format!("EFOV {}% Amplitude", i * 10);
            v.push(std);
            v.push(efov);
        }
        let mut untagged = philips("6.1", 221, "140847", 512, (0, 3000, 150));
        untagged.desc = "Soft tissiu UnTagged".into();
        v.push(untagged);
        let found = run(&v);
        assert_eq!(found.len(), 2, "{found:#?}");
        assert!(found.iter().all(|f| f.scan == 1), "one scan");
        let std = found.iter().find(|f| !f.efov).unwrap();
        let m3 = &std.members[3];
        assert_eq!((m3.label.as_str(), m3.inferred), ("30%", true));
        assert!(std.members.iter().any(|m| m.kind == Kind::Untagged));
        assert_eq!(std.name, "4DCT 2024-06-10 (10 phases + 1)");
        assert!(found.iter().any(|f| f.efov && f.name.contains("EFOV")));
    }

    /// Reconstructions of a brain scan (and five series of three
    /// geometries) say nothing about breathing.
    #[test]
    fn reconstruction_variants_are_no_4dct() {
        let v: Vec<S> = [(201, 512, "O-MAR"), (202, 512, ""), (203, 768, "Bone")]
            .into_iter()
            .map(|(n, rows, d)| {
                let mut s = philips(&format!("7.{n}"), n, "101010", rows, (0, 2000, 80));
                s.desc = d.into();
                s
            })
            .collect();
        assert!(run(&v).is_empty());
        // Eight of one geometry and nothing else: probable, `4DCT?`.
        let v: Vec<S> = (0..8)
            .map(|i| philips(&format!("8.{i}"), 501 + i, "", 512, (0, 2000, 80)))
            .collect();
        let found = run(&v);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].evidence, Evidence::GeometryOnly);
        assert!(found[0].name.starts_with("4DCT? undated"));
    }
}
