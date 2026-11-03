//! DICOM files as bytes: every element with where it sits, and rewrites
//! that leave every element they do not change byte for byte as it was.
//!
//! The curation tools compare, hash and rewrite archive files, and a
//! reader that decodes a file into values and an encoder that writes them
//! back is the wrong tool for that: it normalises what it does not
//! understand (a private element with a bad length, a retired group length,
//! an odd padding byte) and the file that comes back is not the file that
//! went in. So this module never decodes a value it is not asked about. A
//! file is parsed into a tree of [`Element`]s that only point into its
//! bytes - tag, VR, where the header starts, where the value starts and
//! ends, the items of a sequence - and a rewrite ([`rewrite`]) copies every
//! element it leaves alone verbatim, re-encodes only the ones it replaces,
//! and fixes the lengths around them: a defined-length sequence or item that
//! holds a changed element, and a group length (gggg,0000) of a group that
//! changed size, are given their new length; undefined lengths stay
//! undefined.
//!
//! What it reads: a Part 10 file (preamble, `DICM`, the file meta group in
//! explicit VR little endian) or a naked data set (no preamble, the
//! encoding sniffed from the first element, as [`crate::dicomfile`] does),
//! in implicit or explicit VR little endian, with native or encapsulated
//! pixel data. Explicit VR big endian and deflated data sets are parsed for
//! reading where possible and refused for rewriting. In implicit VR a
//! sequence is known by the dictionary (or by an undefined length); a
//! private element of unknown VR is opaque bytes, which is all a rewrite
//! needs of it.
//!
//! A length that runs past its container (Philips' private (01F1,1026) with
//! a bad length is the known case) does not stop the read of anything
//! before it: the element is recorded as truncated, the rest of its
//! container is not parsed, and [`RawFile::problems`] says so.

use anyhow::{bail, Context, Result};
use dicom_core::dictionary::{DataDictionary, DataDictionaryEntry};
use dicom_core::{Tag, VR};
use dicom_dictionary_std::StandardDataDictionary;
use std::path::Path;

/// Implicit VR little endian.
pub const TS_IMPLICIT: &str = "1.2.840.10008.1.2";
/// Explicit VR little endian.
pub const TS_EXPLICIT: &str = "1.2.840.10008.1.2.1";
/// Deflated explicit VR little endian.
pub const TS_DEFLATED: &str = "1.2.840.10008.1.2.1.99";
/// Explicit VR big endian (retired).
pub const TS_BIG: &str = "1.2.840.10008.1.2.2";

pub const PIXEL_DATA: Tag = Tag(0x7FE0, 0x0010);
const ITEM: Tag = Tag(0xFFFE, 0xE000);
const ITEM_END: Tag = Tag(0xFFFE, 0xE00D);
const SEQ_END: Tag = Tag(0xFFFE, 0xE0DD);
const UNDEFINED: u32 = 0xFFFF_FFFF;

/// One element of the file, by position.
#[derive(Clone, Debug)]
pub struct Element {
    pub tag: Tag,
    /// The VR as written (explicit) or as the dictionary has it (implicit;
    /// `UN` for an element it does not know).
    pub vr: VR,
    /// Whether the VR is written in the file.
    pub explicit: bool,
    /// Where the element starts (its tag).
    pub start: usize,
    /// Where its value starts.
    pub value_start: usize,
    /// Where its value ends: for an undefined length, before the sequence
    /// delimiter.
    pub value_end: usize,
    /// Where the element ends, delimiter included.
    pub end: usize,
    /// The length as written; `None` for undefined.
    pub defined: Option<u32>,
    /// The items of a sequence, or the fragments of encapsulated pixel data
    /// (as items with no elements).
    pub items: Option<Vec<Item>>,
    /// The element's value runs past its container.
    pub truncated: bool,
}

/// One item of a sequence.
#[derive(Clone, Debug)]
pub struct Item {
    pub start: usize,
    pub value_start: usize,
    pub value_end: usize,
    pub end: usize,
    pub defined: Option<u32>,
    pub elements: Vec<Element>,
}

/// A whole file.
#[derive(Clone, Debug)]
pub struct RawFile {
    pub bytes: Vec<u8>,
    /// The file meta group (empty for a naked data set).
    pub meta: Vec<Element>,
    /// Where the data set starts.
    pub data_start: usize,
    /// The data set, top level.
    pub elements: Vec<Element>,
    /// The transfer syntax the data set is in.
    pub transfer_syntax: String,
    /// Whether the file has the 128-byte preamble and `DICM`.
    pub part10: bool,
    /// What did not parse, in words, by tag.
    pub problems: Vec<String>,
}

impl Element {
    /// The value's bytes.
    pub fn value<'a>(&self, bytes: &'a [u8]) -> &'a [u8] {
        &bytes[self.value_start.min(bytes.len())..self.value_end.min(bytes.len())]
    }

    /// The value as text: trailing spaces and NULs dropped.
    pub fn text(&self, bytes: &[u8]) -> String {
        String::from_utf8_lossy(self.value(bytes))
            .trim_end_matches(['\0', ' '])
            .trim_start()
            .to_string()
    }

    pub fn is_sequence(&self) -> bool {
        self.items.is_some() && self.tag != PIXEL_DATA
    }
}

fn u16_at(b: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes([*b.get(i)?, *b.get(i + 1)?]))
}

fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes([
        *b.get(i)?,
        *b.get(i + 1)?,
        *b.get(i + 2)?,
        *b.get(i + 3)?,
    ]))
}

/// Whether an explicit VR has the 4-byte length form.
fn long_form(vr: VR) -> bool {
    matches!(
        vr,
        VR::OB
            | VR::OD
            | VR::OF
            | VR::OL
            | VR::OV
            | VR::OW
            | VR::SQ
            | VR::UC
            | VR::UR
            | VR::UT
            | VR::UN
            | VR::SV
            | VR::UV
    )
}

fn vr_of_bytes(b: [u8; 2]) -> Option<VR> {
    std::str::from_utf8(&b).ok()?.parse::<VR>().ok()
}

/// The dictionary's VR of a tag, for implicit VR; `UN` when unknown.
fn dictionary_vr(tag: Tag) -> VR {
    if tag.element() == 0 {
        return VR::UL;
    }
    StandardDataDictionary
        .by_tag(tag)
        .map(|e| e.vr().relaxed())
        .unwrap_or(VR::UN)
}

struct Parser<'a> {
    b: &'a [u8],
    explicit: bool,
    problems: Vec<String>,
}

impl Parser<'_> {
    /// The elements from `at` to `end` (or, with `until_item_end`, to the
    /// item delimiter). Returns them and where parsing stopped (after the
    /// delimiter, if one was found).
    fn elements(
        &mut self,
        mut at: usize,
        end: usize,
        until_item_end: bool,
    ) -> (Vec<Element>, usize) {
        let mut out = Vec::new();
        while at + 8 <= end {
            let (Some(g), Some(e)) = (u16_at(self.b, at), u16_at(self.b, at + 2)) else {
                break;
            };
            let tag = Tag(g, e);
            if tag == ITEM_END {
                return (out, at + 8);
            }
            if tag == SEQ_END || tag == ITEM {
                self.problems
                    .push(format!("stray delimiter {tag} at byte {at}"));
                return (out, end);
            }
            match self.element(at, end, tag) {
                Some(el) => {
                    at = el.end;
                    let stop = el.truncated;
                    out.push(el);
                    if stop {
                        return (out, end);
                    }
                }
                None => {
                    self.problems
                        .push(format!("element {tag} at byte {at} has no complete header"));
                    return (out, end);
                }
            }
        }
        if until_item_end {
            self.problems
                .push(format!("an item ending at byte {at} has no delimiter"));
        }
        (out, at)
    }

    fn element(&mut self, start: usize, end: usize, tag: Tag) -> Option<Element> {
        let b = self.b;
        // Items and delimiters have no VR in any encoding; group 0002 is
        // always explicit.
        let explicit = self.explicit || tag.group() == 0x0002;
        let (vr, len, value_start) = if explicit {
            let vr = vr_of_bytes([*b.get(start + 4)?, *b.get(start + 5)?]).unwrap_or(VR::UN);
            if long_form(vr) {
                (vr, u32_at(b, start + 8)?, start + 12)
            } else {
                (vr, u16_at(b, start + 6)? as u32, start + 8)
            }
        } else {
            (dictionary_vr(tag), u32_at(b, start + 4)?, start + 8)
        };
        if value_start > end {
            return None;
        }
        let mut el = Element {
            tag,
            vr,
            explicit,
            start,
            value_start,
            value_end: value_start,
            end: value_start,
            defined: (len != UNDEFINED).then_some(len),
            items: None,
            truncated: false,
        };
        let is_seq = vr == VR::SQ || (len == UNDEFINED && tag != PIXEL_DATA);
        match el.defined {
            Some(n) => {
                let v_end = value_start + n as usize;
                if v_end > end {
                    self.problems.push(format!(
                        "element {tag} at byte {start}: its length {n} runs past its container"
                    ));
                    el.value_end = end;
                    el.end = end;
                    el.truncated = true;
                    return Some(el);
                }
                el.value_end = v_end;
                el.end = v_end;
                if is_seq {
                    el.items = Some(self.items(value_start, v_end, false));
                }
            }
            None => {
                // An undefined length: a sequence (or UN holding one), or
                // encapsulated pixel data; items up to the delimiter.
                let fragments = tag == PIXEL_DATA;
                // An undefined-length UN is a sequence written in implicit VR,
                // whatever the data set around it is in.
                let saved = self.explicit;
                if explicit && vr == VR::UN {
                    self.explicit = false;
                }
                let (items, stop, delim) = self.items_undefined(value_start, end, fragments);
                self.explicit = saved;
                el.items = Some(items);
                el.value_end = delim;
                el.end = stop;
                if fragments {
                    el.vr = if explicit { el.vr } else { VR::OB };
                } else if el.vr == VR::UN && !explicit {
                    el.vr = VR::SQ;
                }
            }
        }
        Some(el)
    }

    /// The items of a defined-length sequence.
    fn items(&mut self, mut at: usize, end: usize, fragments: bool) -> Vec<Item> {
        let mut out = Vec::new();
        while at + 8 <= end {
            let (Some(g), Some(e), Some(len)) = (
                u16_at(self.b, at),
                u16_at(self.b, at + 2),
                u32_at(self.b, at + 4),
            ) else {
                break;
            };
            if Tag(g, e) != ITEM {
                self.problems.push(format!(
                    "expected an item at byte {at}, found {}",
                    Tag(g, e)
                ));
                break;
            }
            let item = self.item(at, len, end, fragments);
            at = item.end;
            out.push(item);
        }
        out
    }

    /// The items of an undefined-length sequence: (items, where the
    /// element ends, where its value ends - at the delimiter).
    fn items_undefined(
        &mut self,
        mut at: usize,
        end: usize,
        fragments: bool,
    ) -> (Vec<Item>, usize, usize) {
        let mut out = Vec::new();
        while at + 8 <= end {
            let (Some(g), Some(e), Some(len)) = (
                u16_at(self.b, at),
                u16_at(self.b, at + 2),
                u32_at(self.b, at + 4),
            ) else {
                break;
            };
            match Tag(g, e) {
                SEQ_END => return (out, at + 8, at),
                ITEM => {
                    let item = self.item(at, len, end, fragments);
                    at = item.end;
                    out.push(item);
                }
                other => {
                    self.problems
                        .push(format!("expected an item at byte {at}, found {other}"));
                    return (out, end, end);
                }
            }
        }
        self.problems
            .push(format!("a sequence ending at byte {end} has no delimiter"));
        (out, end, end)
    }

    fn item(&mut self, start: usize, len: u32, end: usize, fragments: bool) -> Item {
        let value_start = start + 8;
        if len != UNDEFINED {
            let v_end = (value_start + len as usize).min(end);
            if value_start + len as usize > end {
                self.problems.push(format!(
                    "item at byte {start}: its length runs past its sequence"
                ));
            }
            let elements = if fragments {
                Vec::new()
            } else {
                self.elements(value_start, v_end, false).0
            };
            Item {
                start,
                value_start,
                value_end: v_end,
                end: v_end,
                defined: Some(len),
                elements,
            }
        } else {
            let (elements, stop) = self.elements(value_start, end, true);
            let value_end = stop.saturating_sub(8).max(value_start);
            Item {
                start,
                value_start,
                value_end,
                end: stop,
                defined: None,
                elements,
            }
        }
    }
}

/// Whether a data set starting at `at` is in explicit VR: the two bytes
/// after its first tag are a VR.
fn sniff_explicit(b: &[u8], at: usize) -> bool {
    matches!(
        (b.get(at + 4), b.get(at + 5)),
        (Some(x), Some(y)) if x.is_ascii_uppercase() && y.is_ascii_uppercase()
            && vr_of_bytes([*x, *y]).is_some()
    )
}

impl RawFile {
    pub fn read(path: &Path) -> Result<RawFile> {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        RawFile::parse(bytes)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<RawFile> {
        let part10 = bytes.len() >= 132 && &bytes[128..132] == b"DICM";
        let mut p = Parser {
            b: &bytes,
            explicit: true,
            problems: Vec::new(),
        };
        let (meta, data_start, ts) = if part10 {
            // The meta group: explicit VR LE, group 0002 only.
            let mut at = 132;
            let mut meta = Vec::new();
            while at + 8 <= bytes.len() && u16_at(&bytes, at) == Some(0x0002) {
                let tag = Tag(0x0002, u16_at(&bytes, at + 2).unwrap_or(0));
                match p.element(at, bytes.len(), tag) {
                    Some(el) => {
                        at = el.end;
                        meta.push(el);
                    }
                    None => bail!("the file meta group ends early"),
                }
            }
            let ts = meta
                .iter()
                .find(|e| e.tag == Tag(0x0002, 0x0010))
                .map(|e| e.text(&bytes))
                .unwrap_or_else(|| TS_EXPLICIT.to_string());
            (meta, at, ts)
        } else {
            if bytes.len() < 8 {
                bail!("not DICOM: {} bytes", bytes.len());
            }
            let ts = if sniff_explicit(&bytes, 0) {
                TS_EXPLICIT
            } else {
                TS_IMPLICIT
            };
            (Vec::new(), 0, ts.to_string())
        };
        if ts == TS_DEFLATED {
            return Ok(RawFile {
                problems: vec!["deflated data set: not parsed".into()],
                bytes,
                meta,
                data_start,
                elements: Vec::new(),
                transfer_syntax: ts,
                part10,
            });
        }
        // A file that says implicit but is written explicit (seen in an
        // export), or the other way round: the bytes decide.
        let explicit = if data_start + 8 > bytes.len() {
            ts != TS_IMPLICIT
        } else {
            sniff_explicit(&bytes, data_start)
        };
        if (ts == TS_IMPLICIT) == explicit && data_start < bytes.len() {
            p.problems.push(format!(
                "the transfer syntax says {} VR but the data set is written in {} VR",
                if ts == TS_IMPLICIT {
                    "implicit"
                } else {
                    "explicit"
                },
                if explicit { "explicit" } else { "implicit" }
            ));
        }
        p.explicit = explicit;
        let n = bytes.len();
        let (elements, _) = p.elements(data_start, n, false);
        let problems = p.problems;
        Ok(RawFile {
            bytes,
            meta,
            data_start,
            elements,
            transfer_syntax: ts,
            part10,
            problems,
        })
    }

    /// Whether this is a DICOM file at all: a data set with the `DICM`
    /// header, or one without that names an instance (bytes of a screenshot
    /// may parse, and name nothing).
    pub fn is_dicom(&self) -> bool {
        !self.elements.is_empty()
            && (self.part10
                || self
                    .text(Tag(0x0008, 0x0018))
                    .is_some_and(|s| !s.is_empty()))
    }

    /// Whether [`rewrite`] can work on this file.
    pub fn rewritable(&self) -> Result<()> {
        if self.transfer_syntax == TS_DEFLATED || self.transfer_syntax == TS_BIG {
            bail!("transfer syntax {} is not rewritten", self.transfer_syntax);
        }
        if let Some(p) = self.problems.first() {
            bail!("the file does not parse cleanly ({p}); it is not rewritten");
        }
        Ok(())
    }

    /// The top-level element `tag`.
    pub fn get(&self, tag: Tag) -> Option<&Element> {
        self.elements.iter().find(|e| e.tag == tag)
    }

    /// A top-level value as text.
    pub fn text(&self, tag: Tag) -> Option<String> {
        self.get(tag).map(|e| e.text(&self.bytes))
    }

    /// The bytes of the pixel data: the native value, or every fragment of
    /// encapsulated data in order (the offset table included).
    pub fn pixel_bytes(&self) -> Option<Vec<u8>> {
        let e = self.get(PIXEL_DATA)?;
        match &e.items {
            None => Some(e.value(&self.bytes).to_vec()),
            Some(items) => {
                let mut out = Vec::new();
                for it in items {
                    out.extend_from_slice(&self.bytes[it.value_start..it.value_end]);
                }
                Some(out)
            }
        }
    }

    /// Every element of the data set, nested ones included, with the tags
    /// of the sequences above it (outermost first).
    pub fn walk(&self) -> Vec<(Vec<Tag>, &Element)> {
        fn go<'a>(els: &'a [Element], path: &mut Vec<Tag>, out: &mut Vec<(Vec<Tag>, &'a Element)>) {
            for e in els {
                out.push((path.clone(), e));
                if let Some(items) = &e.items {
                    if e.tag == PIXEL_DATA {
                        continue;
                    }
                    path.push(e.tag);
                    for it in items {
                        go(&it.elements, path, out);
                    }
                    path.pop();
                }
            }
        }
        let mut out = Vec::new();
        go(&self.elements, &mut Vec::new(), &mut out);
        out
    }
}

/// Where an element is, for a rewrite's decision: its tag, VR, value, and
/// the sequences above it.
pub struct At<'a> {
    pub tag: Tag,
    pub vr: VR,
    pub value: &'a [u8],
    /// The tags of the sequences above it, outermost first.
    pub path: &'a [Tag],
}

impl At<'_> {
    pub fn text(&self) -> String {
        String::from_utf8_lossy(self.value)
            .trim_end_matches(['\0', ' '])
            .trim_start()
            .to_string()
    }
}

/// A value as it is written: padded to an even length, with a NUL for a
/// UID and a space for any other text.
pub fn encode_text(vr: VR, s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    if v.len() % 2 == 1 {
        v.push(if vr == VR::UI { 0 } else { b' ' });
    }
    v
}

struct Writer<'a> {
    src: &'a [u8],
    changed: usize,
    /// Elements that go, by position (the sequences above, the tag).
    remove: &'a dyn Fn(&[Tag], Tag) -> bool,
}

impl Writer<'_> {
    /// Re-encode `els` with `f` deciding new values.
    fn elements(
        &mut self,
        els: &[Element],
        path: &mut Vec<Tag>,
        f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        let mut out: Vec<u8> = Vec::new();
        // Where each group's length element was written, to fix it once
        // the group is complete.
        let mut group_len_at: Option<(u16, usize)> = None;
        let fix = |out: &mut Vec<u8>, gl: Option<(u16, usize)>, here: usize| {
            if let Some((_, pos)) = gl {
                // The group's length: from after this element to the end.
                let len = (here - (pos + 4)) as u32;
                out[pos..pos + 4].copy_from_slice(&len.to_le_bytes());
            }
        };
        for e in els {
            if let Some((g, _)) = group_len_at {
                if e.tag.group() != g {
                    let here = out.len();
                    fix(&mut out, group_len_at, here);
                    group_len_at = None;
                }
            }
            if (self.remove)(path, e.tag) {
                self.changed += 1;
                continue;
            }
            if e.tag.element() == 0 && e.defined == Some(4) && e.items.is_none() {
                // A group length: copied, its value fixed at the group's end.
                out.extend_from_slice(&self.src[e.start..e.end]);
                group_len_at = Some((e.tag.group(), out.len() - 4));
                continue;
            }
            out.extend(self.element(e, path, f)?);
        }
        let here = out.len();
        fix(&mut out, group_len_at, here);
        Ok(out)
    }

    fn header(e: &Element, len: u32) -> Result<Vec<u8>> {
        let mut h = Vec::with_capacity(12);
        h.extend_from_slice(&e.tag.group().to_le_bytes());
        h.extend_from_slice(&e.tag.element().to_le_bytes());
        if e.explicit {
            h.extend_from_slice(e.vr.to_string().as_bytes());
            if long_form(e.vr) {
                h.extend_from_slice(&[0, 0]);
                h.extend_from_slice(&len.to_le_bytes());
            } else {
                if len > u16::MAX as u32 {
                    bail!("a value of {len} bytes does not fit element {}", e.tag);
                }
                h.extend_from_slice(&(len as u16).to_le_bytes());
            }
        } else {
            h.extend_from_slice(&len.to_le_bytes());
        }
        Ok(h)
    }

    fn element(
        &mut self,
        e: &Element,
        path: &mut Vec<Tag>,
        f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
    ) -> Result<Vec<u8>> {
        if e.truncated {
            bail!("element {} is truncated", e.tag);
        }
        match &e.items {
            Some(items) if e.tag != PIXEL_DATA => {
                path.push(e.tag);
                let mut body = Vec::new();
                for it in items {
                    let inner = self.elements(&it.elements, path, f)?;
                    body.extend_from_slice(&ITEM.group().to_le_bytes());
                    body.extend_from_slice(&ITEM.element().to_le_bytes());
                    match it.defined {
                        Some(_) => body.extend_from_slice(&(inner.len() as u32).to_le_bytes()),
                        None => body.extend_from_slice(&UNDEFINED.to_le_bytes()),
                    }
                    body.extend(inner);
                    if it.defined.is_none() {
                        body.extend_from_slice(&[0xFE, 0xFF, 0x0D, 0xE0, 0, 0, 0, 0]);
                    }
                }
                path.pop();
                let mut out = match e.defined {
                    Some(_) => Self::header(e, body.len() as u32)?,
                    None => Self::header(e, UNDEFINED)?,
                };
                out.extend(body);
                if e.defined.is_none() {
                    out.extend_from_slice(&[0xFE, 0xFF, 0xDD, 0xE0, 0, 0, 0, 0]);
                }
                Ok(out)
            }
            // Encapsulated pixel data, and every leaf: the callback decides.
            _ => {
                let at = At {
                    tag: e.tag,
                    vr: e.vr,
                    value: e.value(self.src),
                    path,
                };
                match f(&at) {
                    Some(new) if e.items.is_none() => {
                        self.changed += 1;
                        let mut out = Self::header(e, new.len() as u32)?;
                        out.extend(new);
                        Ok(out)
                    }
                    Some(_) => bail!("encapsulated pixel data is not rewritten"),
                    None => Ok(self.src[e.start..e.end].to_vec()),
                }
            }
        }
    }
}

/// Removes nothing: for [`rewrite_with`] and its kin.
pub fn keep_all(_: &[Tag], _: Tag) -> bool {
    false
}

/// A rewrite of `file`: `f` sees every leaf element (nested ones included)
/// and returns its new value, already encoded ([`encode_text`]), or `None`
/// to keep it. Returns the new bytes and how many elements changed. The
/// preamble and the file meta group are copied as they are.
pub fn rewrite(
    file: &RawFile,
    f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
) -> Result<(Vec<u8>, usize)> {
    rewrite_with(file, &keep_all, f)
}

/// [`rewrite`], and the elements `remove` names (by the sequences above
/// them and their tag, at any depth: a sequence goes with its items) are
/// left out. A removed element counts as a change.
pub fn rewrite_with(
    file: &RawFile,
    remove: &dyn Fn(&[Tag], Tag) -> bool,
    f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
) -> Result<(Vec<u8>, usize)> {
    file.rewritable()?;
    let mut w = Writer {
        src: &file.bytes,
        changed: 0,
        remove,
    };
    let body = w.elements(&file.elements, &mut Vec::new(), f)?;
    let mut out = file.bytes[..file.data_start].to_vec();
    out.extend(body);
    Ok((out, w.changed))
}

/// Check a rewritten file against its original, element by element on the
/// raw bytes: the same elements in the same order, each leaf's value what
/// `f` asked for or else byte for byte the original's. The preamble and
/// meta group must be untouched.
pub fn verify(
    before: &RawFile,
    after: &RawFile,
    f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
) -> Result<()> {
    verify_with(before, after, &keep_all, f)
}

/// [`verify`] of a [`rewrite_with`]: the elements `remove` names must be
/// gone, and everything else as [`verify`] has it.
pub fn verify_with(
    before: &RawFile,
    after: &RawFile,
    remove: &dyn Fn(&[Tag], Tag) -> bool,
    f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
) -> Result<()> {
    if before.bytes[..before.data_start] != after.bytes[..after.data_start.min(after.bytes.len())] {
        bail!("the file meta group changed");
    }
    if !after.problems.is_empty() {
        bail!("the rewritten file does not parse: {}", after.problems[0]);
    }
    #[allow(clippy::too_many_arguments)]
    fn go(
        b: &RawFile,
        a: &RawFile,
        eb: &[Element],
        ea: &[Element],
        path: &mut Vec<Tag>,
        remove: &dyn Fn(&[Tag], Tag) -> bool,
        f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
    ) -> Result<()> {
        let eb: Vec<&Element> = eb.iter().filter(|e| !remove(path, e.tag)).collect();
        if eb.len() != ea.len() {
            bail!("{} elements became {}", eb.len(), ea.len());
        }
        for (x, y) in eb.into_iter().zip(ea) {
            if x.tag != y.tag || x.vr != y.vr {
                bail!("element {} became {}", x.tag, y.tag);
            }
            if x.tag.element() == 0 && x.items.is_none() {
                continue; // a group length, recomputed
            }
            match (&x.items, &y.items) {
                (Some(ib), Some(ia)) if x.tag != PIXEL_DATA => {
                    if ib.len() != ia.len() {
                        bail!("sequence {} lost or gained items", x.tag);
                    }
                    path.push(x.tag);
                    for (p, q) in ib.iter().zip(ia) {
                        go(b, a, &p.elements, &q.elements, path, remove, f)?;
                    }
                    path.pop();
                }
                (None, None) | (Some(_), Some(_)) => {
                    let at = At {
                        tag: x.tag,
                        vr: x.vr,
                        value: x.value(&b.bytes),
                        path,
                    };
                    let want = f(&at).unwrap_or_else(|| x.value(&b.bytes).to_vec());
                    let got: Vec<u8> = match &y.items {
                        Some(items) => items
                            .iter()
                            .flat_map(|it| a.bytes[it.start..it.end].to_vec())
                            .collect(),
                        None => y.value(&a.bytes).to_vec(),
                    };
                    let orig: Vec<u8> = match &x.items {
                        Some(items) => items
                            .iter()
                            .flat_map(|it| b.bytes[it.start..it.end].to_vec())
                            .collect(),
                        None => want.clone(),
                    };
                    if got != orig {
                        bail!("element {} does not hold what was written", x.tag);
                    }
                }
                _ => bail!("element {} changed kind", x.tag),
            }
        }
        Ok(())
    }
    go(
        before,
        after,
        &before.elements,
        &after.elements,
        &mut Vec::new(),
        remove,
        f,
    )
}

/// Rewrite a file on disk, verified: the new bytes go to `<file>.tmp`, are
/// read back from disk and [`verify`]d against the original, and only then
/// replace it (a rename on the same volume). Returns how many elements
/// changed; with none, the file is left alone.
pub fn rewrite_file(path: &Path, f: &mut dyn FnMut(&At) -> Option<Vec<u8>>) -> Result<usize> {
    rewrite_file_with(path, &keep_all, f, None).map(|(n, _)| n)
}

/// [`rewrite_file`] with removals ([`rewrite_with`]); with `expect`, the
/// new bytes must have that SHA-256 (what a dry run computed and checked)
/// or nothing is written. Returns how many elements changed and the new
/// file's SHA-256.
pub fn rewrite_file_with(
    path: &Path,
    remove: &dyn Fn(&[Tag], Tag) -> bool,
    f: &mut dyn FnMut(&At) -> Option<Vec<u8>>,
    expect: Option<&str>,
) -> Result<(usize, String)> {
    use sha2::{Digest, Sha256};
    let before = RawFile::read(path)?;
    let (bytes, changed) = rewrite_with(&before, remove, f)?;
    let sha = super::journal::hex(&Sha256::digest(&bytes));
    if changed == 0 {
        return Ok((0, sha));
    }
    if let Some(want) = expect {
        if want != sha {
            bail!(
                "{} would not come out as the dry run computed: was it changed since?",
                path.display()
            );
        }
    }
    let tmp = path.with_extension(match path.extension() {
        Some(e) => format!("{}.tmp", e.to_string_lossy()),
        None => "tmp".to_string(),
    });
    std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
    let check = RawFile::read(&tmp).and_then(|after| verify_with(&before, &after, remove, f));
    if let Err(e) = check {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.context(format!("{} was not replaced", path.display())));
    }
    std::fs::rename(&tmp, path).with_context(|| format!("replace {}", path.display()))?;
    Ok((changed, sha))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// An explicit-VR element.
    pub fn ex(tag: Tag, vr: &str, value: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&tag.group().to_le_bytes());
        out.extend_from_slice(&tag.element().to_le_bytes());
        out.extend_from_slice(vr.as_bytes());
        let v: VR = vr.parse().unwrap();
        if long_form(v) {
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        } else {
            out.extend_from_slice(&(value.len() as u16).to_le_bytes());
        }
        out.extend_from_slice(value);
        out
    }

    /// An implicit-VR element.
    pub fn im(tag: Tag, value: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&tag.group().to_le_bytes());
        out.extend_from_slice(&tag.element().to_le_bytes());
        out.extend_from_slice(&(value.len() as u32).to_le_bytes());
        out.extend_from_slice(value);
        out
    }

    /// An item of defined length holding `body`.
    pub fn item(body: &[u8]) -> Vec<u8> {
        let mut out = vec![0xFE, 0xFF, 0x00, 0xE0];
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        out
    }

    /// A Part 10 file of `ts` around `data`.
    pub fn part10(ts: &str, data: &[u8]) -> Vec<u8> {
        let mut meta = Vec::new();
        meta.extend(ex(Tag(0x0002, 0x0010), "UI", &encode_text(VR::UI, ts)));
        let mut out = vec![0u8; 128];
        out.extend_from_slice(b"DICM");
        out.extend(ex(
            Tag(0x0002, 0x0000),
            "UL",
            &(meta.len() as u32).to_le_bytes(),
        ));
        out.extend(meta);
        out.extend_from_slice(data);
        out
    }

    const NAME: Tag = Tag(0x0010, 0x0010);
    const PID: Tag = Tag(0x0010, 0x0020);
    const REF_SERIES: Tag = Tag(0x0008, 0x1115);
    const SERIES_UID: Tag = Tag(0x0020, 0x000E);

    /// Implicit VR with a group length over group 0010, a defined-length
    /// sequence holding a UID, an undefined-length one, and a private
    /// element whose length runs past nothing.
    fn implicit_file() -> Vec<u8> {
        let mut g10 = Vec::new();
        g10.extend(im(NAME, b"P01-039 "));
        g10.extend(im(PID, b"P01-039 "));
        let mut data = Vec::new();
        data.extend(im(Tag(0x0008, 0x0060), b"CT"));
        data.extend(im(Tag(0x0010, 0x0000), &(g10.len() as u32).to_le_bytes()));
        data.extend(g10);
        let inner = im(SERIES_UID, &encode_text(VR::UI, "1.2.3.4"));
        data.extend(im(REF_SERIES, &item(&inner)));
        // Undefined length, undefined item.
        let mut undef = Vec::new();
        undef.extend_from_slice(&[0x08, 0x00, 0x40, 0x11, 0xFF, 0xFF, 0xFF, 0xFF]);
        undef.extend_from_slice(&[0xFE, 0xFF, 0x00, 0xE0, 0xFF, 0xFF, 0xFF, 0xFF]);
        undef.extend(im(SERIES_UID, &encode_text(VR::UI, "1.2.3.4")));
        undef.extend_from_slice(&[0xFE, 0xFF, 0x0D, 0xE0, 0, 0, 0, 0]);
        undef.extend_from_slice(&[0xFE, 0xFF, 0xDD, 0xE0, 0, 0, 0, 0]);
        data.extend(undef);
        data.extend(im(Tag(0x01F1, 0x1026), &[1, 2, 3, 4, 5, 6]));
        data.extend(im(PIXEL_DATA, &[7, 8, 9, 10]));
        part10(TS_IMPLICIT, &data)
    }

    #[test]
    fn a_file_parses_into_positions_with_nested_sequences() {
        let f = RawFile::parse(implicit_file()).unwrap();
        assert!(f.problems.is_empty(), "{:?}", f.problems);
        assert_eq!(f.text(NAME).as_deref(), Some("P01-039"));
        assert_eq!(f.pixel_bytes().unwrap(), vec![7, 8, 9, 10]);
        let uids: Vec<String> = f
            .walk()
            .into_iter()
            .filter(|(_, e)| e.tag == SERIES_UID)
            .map(|(_, e)| e.text(&f.bytes))
            .collect();
        assert_eq!(uids, vec!["1.2.3.4", "1.2.3.4"]);
    }

    #[test]
    fn a_rewrite_keeps_every_other_byte_and_fixes_the_lengths_around_it() {
        let src = implicit_file();
        let f = RawFile::parse(src.clone()).unwrap();
        let mut edit = |at: &At| -> Option<Vec<u8>> {
            match at.tag {
                NAME => Some(encode_text(VR::PN, "P01-0390")),
                SERIES_UID => Some(encode_text(VR::UI, "1.2.3.44444")),
                _ => None,
            }
        };
        let (out, changed) = rewrite(&f, &mut edit).unwrap();
        assert_eq!(changed, 3);
        let g = RawFile::parse(out.clone()).unwrap();
        assert!(g.problems.is_empty(), "{:?}", g.problems);
        verify(&f, &g, &mut edit).unwrap();
        assert_eq!(g.text(NAME).as_deref(), Some("P01-0390"));
        // The group length follows the group.
        let gl = g.get(Tag(0x0010, 0x0000)).unwrap();
        let n = u32::from_le_bytes(gl.value(&g.bytes).try_into().unwrap());
        assert_eq!(
            n as usize,
            g.get(PID).unwrap().end - g.get(NAME).unwrap().start
        );
        // The private element and the pixels are byte for byte as they were.
        let p = |file: &RawFile| {
            file.get(Tag(0x01F1, 0x1026))
                .unwrap()
                .value(&file.bytes)
                .to_vec()
        };
        assert_eq!(p(&f), p(&g));
        // A rewrite that changes nothing is the same file.
        let (same, n) = rewrite(&f, &mut |_| None).unwrap();
        assert_eq!((same, n), (src, 0));
    }

    #[test]
    fn a_length_past_its_container_is_a_problem_not_a_failure() {
        let mut data = Vec::new();
        data.extend(ex(Tag(0x0008, 0x0060), "CS", b"CT"));
        // Claims 100 bytes, holds 4.
        let mut bad = ex(Tag(0x01F1, 0x1026), "UN", &[1, 2, 3, 4]);
        bad[8..12].copy_from_slice(&100u32.to_le_bytes());
        data.extend(bad);
        let f = RawFile::parse(part10(TS_EXPLICIT, &data)).unwrap();
        assert_eq!(f.text(Tag(0x0008, 0x0060)).as_deref(), Some("CT"));
        assert_eq!(f.problems.len(), 1);
        assert!(f.rewritable().is_err());
    }

    #[test]
    fn a_file_that_says_implicit_but_is_explicit_reads_by_its_bytes() {
        let mut data = Vec::new();
        data.extend(ex(Tag(0x0008, 0x0060), "CS", b"CT"));
        data.extend(ex(NAME, "PN", b"P01-019 "));
        let f = RawFile::parse(part10(TS_IMPLICIT, &data)).unwrap();
        assert_eq!(f.text(NAME).as_deref(), Some("P01-019"));
        assert_eq!(f.problems.len(), 1, "the mismatch is reported");
    }
}
