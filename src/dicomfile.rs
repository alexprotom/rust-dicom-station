//! Opening DICOM files, including the ones written without a file header.
//!
//! A conformant DICOM *file* starts with a 128-byte preamble, the magic word
//! `DICM`, and the File Meta group (0002) whose Transfer Syntax UID says how
//! the rest of the file is encoded. `dicom_object::open_file` insists on all
//! three, and for a file that has them that is exactly right.
//!
//! Plenty of real exports do not have them. What travels over the wire in a
//! DICOM association is the naked data set, and a scanner console or a
//! research export that writes that stream straight to disk produces a file
//! whose very first bytes are already the first element - most often
//! (0008,0005) *Specific Character Set* in implicit VR little endian. Such a
//! file is perfectly readable; it just never says in what encoding, because
//! the association negotiated that instead.
//!
//! `example_data_star/raw/.../CCT_RTSTR` is one of those exports: 318 CT
//! slices and a structure set, none of which the viewer would open, because
//! the scan rejected every file before looking at it and the load then failed
//! with "No DICOM files found".
//!
//! So this module tries the standard reader first and, only for a file with
//! no `DICM` magic at all, sniffs the encoding from the first element headers
//! and reads the naked data set. The file meta table is then synthesised from
//! what the data set says about itself, so the result is an ordinary
//! [`DefaultDicomObject`]: pixel decoding, the RT parsers, the anonymiser and
//! re-export all work on it without knowing where it came from.

use std::io::{Cursor, Read};
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use dicom_core::Tag;
use dicom_dictionary_std::tags;
use dicom_object::{DefaultDicomObject, FileMetaTableBuilder, InMemDicomObject, OpenFileOptions};
use dicom_transfer_syntax_registry::{TransferSyntaxIndex, TransferSyntaxRegistry};

/// Open a file for classification: everything before *Pixel Data*.
///
/// This is the header-only read the directory scan and the anonymiser's first
/// pass do, where the pixels are never looked at and reading them would be
/// most of the work.
pub fn open_header(path: &Path) -> Result<DefaultDicomObject> {
    open(path, Some(tags::PIXEL_DATA))
}

/// Open a file in full, pixels included.
pub fn open_full(path: &Path) -> Result<DefaultDicomObject> {
    open(path, None)
}

fn open(path: &Path, until: Option<Tag>) -> Result<DefaultDicomObject> {
    let opened = match until {
        Some(tag) => OpenFileOptions::new().read_until(tag).open_file(path),
        None => OpenFileOptions::new().open_file(path),
    };
    let err = match opened {
        Ok(obj) => return Ok(obj),
        Err(e) => e,
    };

    // The file claims a header and the standard reader still could not use
    // it. Guessing an encoding on top of that would replace a precise
    // complaint with a vague one, so let its own error stand.
    if has_file_header(path) {
        return Err(anyhow::Error::new(err).context(format!("open {}", path.display())));
    }
    open_bare(path, until)
}

/// Does the file start like a DICOM *file*, as opposed to a bare data set?
///
/// `DICM` sits after the 128-byte preamble. A few writers omit the preamble
/// but keep the magic word, so offset 0 counts as well.
fn has_file_header(path: &Path) -> bool {
    let Ok(mut f) = std::fs::File::open(path) else {
        return false;
    };
    let mut head = [0u8; 132];
    let mut got = 0usize;
    while got < head.len() {
        match f.read(&mut head[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(_) => return false,
        }
    }
    (got >= 132 && &head[128..132] == b"DICM") || (got >= 4 && &head[0..4] == b"DICM")
}

// ---------------------------------------------------------------------------
// Header-less data sets
// ---------------------------------------------------------------------------

/// The encodings a header-less data set can plausibly be in.
///
/// Implicit VR little endian is the default transfer syntax and by far the
/// most common way to find a naked data set on disk. Implicit VR *big* endian
/// has never existed, so a big-endian data set is necessarily explicit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Enc {
    ImplicitLe,
    ExplicitLe,
    ExplicitBe,
}

impl Enc {
    const ALL: [Enc; 3] = [Enc::ImplicitLe, Enc::ExplicitLe, Enc::ExplicitBe];

    fn uid(self) -> &'static str {
        match self {
            Enc::ImplicitLe => "1.2.840.10008.1.2",
            Enc::ExplicitLe => "1.2.840.10008.1.2.1",
            Enc::ExplicitBe => "1.2.840.10008.1.2.2",
        }
    }

    fn explicit(self) -> bool {
        !matches!(self, Enc::ImplicitLe)
    }

    fn u16(self, b: &[u8]) -> u16 {
        let v = [b[0], b[1]];
        if matches!(self, Enc::ExplicitBe) {
            u16::from_be_bytes(v)
        } else {
            u16::from_le_bytes(v)
        }
    }

    fn u32(self, b: &[u8]) -> u32 {
        let v = [b[0], b[1], b[2], b[3]];
        if matches!(self, Enc::ExplicitBe) {
            u32::from_be_bytes(v)
        } else {
            u32::from_le_bytes(v)
        }
    }
}

/// How far a shallow walk of the top-level element headers got.
struct Walk {
    /// Elements whose header parsed and whose value fitted in the buffer.
    elements: usize,
    /// Byte offset of the element we were asked to stop before, if reached.
    stop_at: Option<usize>,
}

/// VRs encoded with two reserved bytes and a 32-bit length.
const LONG_VRS: [&[u8; 2]; 11] = [
    b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"UC", b"UN", b"UR", b"UT",
];

/// Walk the top-level element headers of `bytes` without decoding any value.
///
/// Values are skipped by their declared length, so the walk is essentially
/// free, and it answers the two questions that matter here. Is `enc` the
/// encoding this data set is actually in - a wrong guess desynchronises
/// within an element or two and the lengths then run off the end of the
/// buffer or the tags stop ascending. And where does `stop` begin, so that
/// the pixels can be left unread.
///
/// An undefined length (a sequence, or encapsulated pixel data) ends the
/// walk: following one means parsing items, which is the real parser's job.
fn walk(bytes: &[u8], enc: Enc, stop: Option<Tag>) -> Walk {
    let mut at = 0usize;
    let mut elements = 0usize;
    let mut last: Option<Tag> = None;

    while at + 8 <= bytes.len() {
        let tag = Tag(enc.u16(&bytes[at..]), enc.u16(&bytes[at + 2..]));
        // Elements are stored in ascending tag order. Anything else means we
        // are no longer looking at element headers.
        if last.is_some_and(|prev| tag <= prev) {
            break;
        }
        last = Some(tag);
        if stop == Some(tag) {
            return Walk {
                elements,
                stop_at: Some(at),
            };
        }

        let (header, len) = if enc.explicit() {
            let vr = &bytes[at + 4..at + 6];
            if LONG_VRS.iter().any(|known| known.as_slice() == vr) {
                if at + 12 > bytes.len() {
                    break;
                }
                (12usize, enc.u32(&bytes[at + 8..]))
            } else if vr.iter().all(u8::is_ascii_uppercase) {
                (8usize, u32::from(enc.u16(&bytes[at + 6..])))
            } else {
                break;
            }
        } else {
            (8usize, enc.u32(&bytes[at + 4..]))
        };

        if len == u32::MAX {
            break; // undefined length
        }
        let Some(next) = at
            .checked_add(header)
            .and_then(|a| a.checked_add(len as usize))
        else {
            break;
        };
        if next > bytes.len() {
            break;
        }
        at = next;
        elements += 1;
    }

    Walk {
        elements,
        stop_at: None,
    }
}

/// Least number of elements a candidate encoding must walk to be believed.
const MIN_ELEMENTS: usize = 4;

/// Guess the encoding of a header-less data set from its first elements.
///
/// The data set is sorted by tag, so its first element is in one of the
/// identifying groups (0002 for a stray meta group, then 0008, 0010, 0018,
/// 0020). A group number that small has a zero high byte, which pins the byte
/// order to whichever reading yields it; implicit and explicit VR are then
/// told apart by whether bytes 4 and 5 spell a VR. Every candidate is
/// confirmed by [`walk`] rather than by that first guess alone, which is what
/// keeps a file that is not DICOM at all from being read as one.
fn sniff(bytes: &[u8]) -> Option<Enc> {
    Enc::ALL
        .into_iter()
        .filter(|enc| {
            bytes.len() >= 8 && {
                let group = enc.u16(&bytes[0..]);
                (0x0002..=0x0020).contains(&group) && group % 2 == 0
            }
        })
        .map(|enc| (enc, walk(bytes, enc, Some(tags::PIXEL_DATA))))
        .filter(|(_, w)| w.elements >= MIN_ELEMENTS)
        .max_by_key(|(_, w)| w.elements)
        .map(|(enc, _)| enc)
}

/// Read a data set that carries no file header of its own.
fn open_bare(path: &Path, until: Option<Tag>) -> Result<DefaultDicomObject> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let enc = sniff(&bytes).ok_or_else(|| {
        anyhow!(
            "{} has no DICOM file header and does not begin with a readable data set element",
            path.display()
        )
    })?;

    // Stopping before Pixel Data is what keeps the classification pass from
    // reading whole images. When the offset cannot be established (an
    // undefined-length sequence on the way, or no pixels at all, as in an
    // RTSTRUCT) the whole file is parsed, which is what the caller asked for
    // anyway, only slower.
    let end = until
        .and_then(|tag| walk(&bytes, enc, Some(tag)).stop_at)
        .unwrap_or(bytes.len());

    let ts = TransferSyntaxRegistry
        .get(enc.uid())
        .ok_or_else(|| anyhow!("transfer syntax {} is not in the registry", enc.uid()))?;
    let obj = InMemDicomObject::read_dataset_with_ts(Cursor::new(&bytes[..end]), ts).with_context(
        || {
            format!(
                "read {} as a header-less data set ({})",
                path.display(),
                enc.uid()
            )
        },
    )?;

    // `with_meta` fills Media Storage SOP Class / Instance UID from the data
    // set itself, so the synthesised header agrees with its contents.
    obj.with_meta(FileMetaTableBuilder::new().transfer_syntax(enc.uid()))
        .with_context(|| format!("build a file meta table for {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Implicit VR little endian element with a short text value.
    fn implicit(tag: Tag, value: &str) -> Vec<u8> {
        let mut v = value.as_bytes().to_vec();
        if v.len() % 2 == 1 {
            v.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&tag.0.to_le_bytes());
        out.extend_from_slice(&tag.1.to_le_bytes());
        out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        out.extend_from_slice(&v);
        out
    }

    /// Explicit VR little endian element with a short text value.
    fn explicit(tag: Tag, vr: &[u8; 2], value: &str) -> Vec<u8> {
        let mut v = value.as_bytes().to_vec();
        if v.len() % 2 == 1 {
            v.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&tag.0.to_le_bytes());
        out.extend_from_slice(&tag.1.to_le_bytes());
        out.extend_from_slice(vr);
        out.extend_from_slice(&(v.len() as u16).to_le_bytes());
        out.extend_from_slice(&v);
        out
    }

    fn implicit_ct_header() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend(implicit(tags::SPECIFIC_CHARACTER_SET, "ISO_IR 100"));
        b.extend(implicit(tags::IMAGE_TYPE, "ORIGINAL\\PRIMARY\\AXIAL"));
        b.extend(implicit(tags::SOP_CLASS_UID, "1.2.840.10008.5.1.4.1.1.2"));
        b.extend(implicit(tags::SOP_INSTANCE_UID, "1.2.3.4.5.6"));
        b.extend(implicit(tags::MODALITY, "CT"));
        b.extend(implicit(tags::PATIENT_NAME, "PHANTOM^RAW"));
        b
    }

    #[test]
    fn an_implicit_vr_data_set_is_recognised() {
        let b = implicit_ct_header();
        assert_eq!(sniff(&b), Some(Enc::ImplicitLe));
    }

    #[test]
    fn an_explicit_vr_data_set_is_recognised() {
        let mut b = Vec::new();
        b.extend(explicit(tags::SPECIFIC_CHARACTER_SET, b"CS", "ISO_IR 100"));
        b.extend(explicit(tags::IMAGE_TYPE, b"CS", "ORIGINAL\\PRIMARY"));
        b.extend(explicit(
            tags::SOP_CLASS_UID,
            b"UI",
            "1.2.840.10008.5.1.4.1.1.2",
        ));
        b.extend(explicit(tags::SOP_INSTANCE_UID, b"UI", "1.2.3.4.5.6"));
        b.extend(explicit(tags::MODALITY, b"CS", "CT"));
        assert_eq!(sniff(&b), Some(Enc::ExplicitLe));
    }

    #[test]
    fn the_walk_finds_where_the_pixels_start() {
        let mut b = implicit_ct_header();
        let start = b.len();
        // Pixel Data, 8 bytes of it.
        b.extend(implicit(tags::PIXEL_DATA, "abcdefgh"));
        let w = walk(&b, Enc::ImplicitLe, Some(tags::PIXEL_DATA));
        assert_eq!(w.stop_at, Some(start));
        assert!(w.elements >= MIN_ELEMENTS);
    }

    #[test]
    fn data_that_is_not_dicom_is_not_guessed_into_dicom() {
        assert_eq!(sniff(b"hello"), None);
        assert_eq!(
            sniff(&[0u8; 4096]),
            None,
            "a run of zeros is not a data set"
        );
        assert_eq!(
            sniff(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00"),
            None
        );
        // Random-looking bytes that happen to start with a small group.
        let mut junk = vec![0x08, 0x00, 0x05, 0x00];
        junk.extend_from_slice(&[0xAB; 64]);
        assert_eq!(sniff(&junk), None, "one plausible tag is not enough");
    }

    #[test]
    fn a_header_less_file_opens_and_keeps_its_transfer_syntax() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_headerless");
        std::fs::create_dir_all(&dir).expect("create the folder");
        let path = dir.join("bare.dcm");
        std::fs::write(&path, implicit_ct_header()).expect("write");

        assert!(!has_file_header(&path));
        let obj = open_full(&path).expect("a header-less data set opens");
        assert_eq!(obj.meta().transfer_syntax(), "1.2.840.10008.1.2");
        assert_eq!(
            obj.meta()
                .media_storage_sop_class_uid()
                .trim_end_matches('\0'),
            "1.2.840.10008.5.1.4.1.1.2",
            "the synthesised header agrees with the data set"
        );
        assert_eq!(
            crate::loader::str_of(&obj, tags::MODALITY).as_deref(),
            Some("CT")
        );
    }

    #[test]
    fn a_file_that_is_not_dicom_still_fails() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_headerless");
        std::fs::create_dir_all(&dir).expect("create the folder");
        let path = dir.join("not-dicom.txt");
        std::fs::write(&path, b"hello").expect("write");
        assert!(open_full(&path).is_err());
        assert!(open_header(&path).is_err());
    }
}
