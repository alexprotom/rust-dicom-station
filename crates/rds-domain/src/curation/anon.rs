//! Pseudonymizing files in place (or on their way to the quarantine), the
//! way the archive keeps its trial data: the patient's name and ID become
//! the patient's code, the birth date keeps its year (`YYYY0101`), the
//! dates of the course stay, the UIDs stay (the copy is merged with the
//! archive), private elements stay (a setup system's registrations need
//! them) unless asked, and the tags of [`REMOVED`] go. Staff names
//! (OperatorsName, ReviewerName) stay, as in the archive.
//!
//! What makes it safe is the **leftover scan**: every token of every name
//! and ID the patient's files carry (three characters or more; a number
//! only with nine digits or more, as national IDs have; not the words of
//! [`STOP`], which test and service accounts use) must be absent from every
//! text element of the result, nested ones and private text included
//! (binary values, UIDs and the pixel data aside). A dry run rewrites each
//! file in memory and scans it; a token anywhere refuses the plan, saying
//! where (by tag), never what.
//!
//! [`report`] is that scan on the files as they are: where identifiers
//! occur, by tag and count - what to know before choosing the options.

use std::collections::{BTreeMap, BTreeSet};

use dicom_core::{Tag, VR};
use dicom_dictionary_std::tags;

use super::raw::RawFile;

/// The tags a pseudonymization removes: other names and IDs, birth and
/// mother's birth names, address, telephone, the medical record locator,
/// the issuer of the ID, the physicians, the patient's comments. A dataset
/// that still holds any of them is not pseudonymized
/// ([`crate::mcp::phi`]'s policy reads the same list).
#[allow(deprecated)]
pub const REMOVED: &[(Tag, &str)] = &[
    (tags::OTHER_PATIENT_NAMES, "OtherPatientNames"),
    (tags::OTHER_PATIENT_I_DS, "OtherPatientIDs"),
    (tags::OTHER_PATIENT_I_DS_SEQUENCE, "OtherPatientIDsSequence"),
    (tags::PATIENT_BIRTH_NAME, "PatientBirthName"),
    (tags::PATIENT_MOTHER_BIRTH_NAME, "PatientMotherBirthName"),
    (tags::PATIENT_ADDRESS, "PatientAddress"),
    (tags::PATIENT_TELEPHONE_NUMBERS, "PatientTelephoneNumbers"),
    (tags::MEDICAL_RECORD_LOCATOR, "MedicalRecordLocator"),
    (tags::ISSUER_OF_PATIENT_ID, "IssuerOfPatientID"),
    (tags::REFERRING_PHYSICIAN_NAME, "ReferringPhysicianName"),
    (tags::PERFORMING_PHYSICIAN_NAME, "PerformingPhysicianName"),
    (tags::PHYSICIANS_OF_RECORD, "PhysiciansOfRecord"),
    (tags::PATIENT_COMMENTS, "PatientComments"),
];

/// Words that are not a patient's, however their accounts are named.
pub const STOP: &[&str] = &[
    "service",
    "test",
    "phantom",
    "qa",
    "sim",
    "patient",
    "anonymous",
    "anonymized",
    "unknown",
];

/// Lower-case alphanumeric runs of `s`.
fn words(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
}

/// The tokens of a patient's names and IDs that must not be left; `keep`
/// are the tokens of the code they get (which may stay).
pub fn tokens<'a>(values: impl IntoIterator<Item = &'a str>, keep: &str) -> BTreeSet<String> {
    let keep: BTreeSet<String> = words(keep).collect();
    let mut out = BTreeSet::new();
    for v in values {
        for w in words(v) {
            let digits = w.chars().all(|c| c.is_ascii_digit());
            let ok = if digits {
                w.len() >= 9
            } else {
                w.chars().count() >= 3
            };
            if ok && !keep.contains(&w) && !STOP.contains(&w.as_str()) {
                out.insert(w);
            }
        }
    }
    out
}

/// Whether a value is text worth scanning: a text VR, or a private or
/// unknown one whose bytes are all printable.
fn is_text(vr: VR, tag: Tag, value: &[u8]) -> bool {
    match vr {
        VR::AE
        | VR::AS
        | VR::CS
        | VR::DA
        | VR::DS
        | VR::DT
        | VR::IS
        | VR::LO
        | VR::LT
        | VR::PN
        | VR::SH
        | VR::ST
        | VR::TM
        | VR::UC
        | VR::UR
        | VR::UT => true,
        VR::UN | VR::OB if tag.group() % 2 == 1 || vr == VR::UN => {
            !value.is_empty()
                && value.iter().all(|b| {
                    *b == 0
                        || *b == b'\t'
                        || *b == b'\r'
                        || *b == b'\n'
                        || (32..127).contains(b)
                        || *b >= 160
                })
        }
        _ => false,
    }
}

/// Where the tag sits, as `GGGG,EEEE` joined by `>`.
fn place(path: &[Tag], tag: Tag) -> String {
    path.iter()
        .chain(std::iter::once(&tag))
        .map(|t| format!("{:04X},{:04X}", t.group(), t.element()))
        .collect::<Vec<_>>()
        .join(">")
}

/// The places in `f` whose text holds one of `tokens`.
pub fn scan(f: &RawFile, tokens: &BTreeSet<String>) -> Vec<String> {
    if tokens.is_empty() {
        return Vec::new();
    }
    let long_numbers: Vec<&String> = tokens
        .iter()
        .filter(|t| t.chars().all(|c| c.is_ascii_digit()))
        .collect();
    let mut out = Vec::new();
    for (path, e) in f.walk() {
        if e.items.is_some() || e.tag == super::raw::PIXEL_DATA {
            continue;
        }
        let v = e.value(&f.bytes);
        if !is_text(e.vr, e.tag, v) {
            continue;
        }
        let s = String::from_utf8_lossy(v);
        let hit = words(&s).any(|w| tokens.contains(&w))
            || long_numbers.iter().any(|n| s.contains(n.as_str()));
        if hit {
            out.push(place(&path, e.tag));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// The names and IDs a file carries: PatientName, PatientID, and the other
/// names and IDs (nested ones too).
#[allow(deprecated)]
pub fn identities(f: &RawFile) -> Vec<String> {
    let wanted = [
        tags::PATIENT_NAME,
        tags::PATIENT_ID,
        tags::OTHER_PATIENT_NAMES,
        tags::OTHER_PATIENT_I_DS,
        tags::PATIENT_BIRTH_NAME,
    ];
    f.walk()
        .into_iter()
        .filter(|(_, e)| wanted.contains(&e.tag) && e.items.is_none())
        .map(|(_, e)| e.text(&f.bytes))
        .filter(|v| !v.trim().is_empty())
        .collect()
}

/// `YYYY0101` of a birth date (empty stays empty).
pub fn year_only(date: &str) -> String {
    let d = date.trim();
    if d.len() >= 4 && d[..4].chars().all(|c| c.is_ascii_digit()) {
        format!("{}0101", &d[..4])
    } else {
        String::new()
    }
}

/// Where the identifiers of one patient's files occur: place → files.
#[derive(Debug, Default, serde::Serialize)]
pub struct Report {
    pub files: usize,
    /// Distinct names and IDs among its files (counted, not shown).
    pub identities: usize,
    pub tokens: usize,
    /// Tag places holding a token, with how many files.
    pub places: BTreeMap<String, usize>,
    /// [`REMOVED`] tags present, with how many files.
    pub removed_tags: BTreeMap<String, usize>,
    /// Birth dates that are not `YYYY0101`, and private elements.
    pub full_birth_dates: usize,
    pub files_with_private: usize,
}

/// The report over one patient's files.
pub fn report(files: &[RawFile], code: &str) -> Report {
    let ids: BTreeSet<String> = files.iter().flat_map(identities).collect();
    let toks = tokens(ids.iter().map(String::as_str), code);
    let mut r = Report {
        files: files.len(),
        identities: ids.len(),
        tokens: toks.len(),
        ..Default::default()
    };
    for f in files {
        // The name and ID themselves are expected; the rest is the news.
        for p in scan(f, &toks) {
            *r.places.entry(p).or_default() += 1;
        }
        for (t, name) in REMOVED {
            if f.walk().iter().any(|(_, e)| e.tag == *t) {
                *r.removed_tags.entry(name.to_string()).or_default() += 1;
            }
        }
        let bd = f.text(tags::PATIENT_BIRTH_DATE).unwrap_or_default();
        if !bd.is_empty() && year_only(&bd) != bd {
            r.full_birth_dates += 1;
        }
        if f.elements.iter().any(|e| e.tag.group() % 2 == 1) {
            r.files_with_private += 1;
        }
    }
    r
}

/// The [`super::remap::Changes`] that pseudonymize one file as `code`.
pub fn changes(f: &RawFile, code: &str, keep_private: bool) -> super::remap::Changes {
    let mut set = vec![
        ("00100010".to_string(), code.to_string()),
        ("00100020".to_string(), code.to_string()),
    ];
    let bd = f.text(tags::PATIENT_BIRTH_DATE).unwrap_or_default();
    if !bd.is_empty() && year_only(&bd) != bd {
        set.push(("00100030".to_string(), year_only(&bd)));
    }
    super::remap::Changes {
        set,
        remove: REMOVED
            .iter()
            .map(|(t, _)| format!("{:04X}{:04X}", t.group(), t.element()))
            .collect(),
        remove_private: !keep_private,
        remap: false,
    }
}

/// Patients whose files disagree on birth date or sex: per variant
/// (birth date, sex) how many files, the values withheld.
pub fn demographics(files: &[RawFile]) -> (usize, usize, Vec<usize>) {
    let mut by: BTreeMap<(String, String), usize> = BTreeMap::new();
    for f in files {
        let k = (
            f.text(tags::PATIENT_BIRTH_DATE).unwrap_or_default(),
            f.text(tags::PATIENT_SEX).unwrap_or_default(),
        );
        *by.entry(k).or_default() += 1;
    }
    let dates: BTreeSet<&String> = by.keys().map(|k| &k.0).collect();
    let sexes: BTreeSet<&String> = by.keys().map(|k| &k.1).collect();
    (dates.len(), sexes.len(), by.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::super::raw::tests::{ex, item, part10};
    use super::super::raw::TS_EXPLICIT;
    use super::super::remap::{simulate, UidMaps};
    use super::*;

    #[test]
    fn tokens_leave_short_numbers_stop_words_and_the_code() {
        let t = tokens(
            ["Cohen^David", "123456789", "1111", "service_qa", "P01-031"],
            "P01-031",
        );
        assert!(t.contains("cohen") && t.contains("david") && t.contains("123456789"));
        assert!(!t.contains("1111") && !t.contains("service") && !t.contains("qa"));
        assert!(!t.contains("p01") && !t.contains("031"));
        assert_eq!(year_only("19560317"), "19560101");
        assert_eq!(year_only(""), "");
    }

    #[test]
    fn a_name_left_in_a_nested_or_private_element_refuses() {
        let mut d = Vec::new();
        d.extend(ex(tags::PATIENT_NAME, "PN", b"Cohen^David "));
        d.extend(ex(tags::PATIENT_ID, "LO", b"123456789 "));
        d.extend(ex(tags::PATIENT_BIRTH_DATE, "DA", b"19560317"));
        d.extend(ex(tags::OTHER_PATIENT_NAMES, "PN", b"Kohen "));
        // A plan's free text names the patient, and so does a private note.
        let note = item(&ex(tags::OPERATORS_NAME, "PN", b"Smith^A "));
        d.extend(ex(
            Tag(0x300A, 0x00B0),
            "SQ",
            &item(&ex(tags::BEAM_DESCRIPTION, "ST", b"for Cohen ")),
        ));
        d.extend(ex(Tag(0x0008, 0x1072), "SQ", &note));
        d.extend(ex(Tag(0x0029, 0x1010), "UN", b"ID 123456789 "));
        let dir = std::env::temp_dir().join("rds_anon");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.dcm");
        std::fs::write(&p, part10(TS_EXPLICIT, &d)).unwrap();
        let f = RawFile::read(&p).unwrap();
        let ids = identities(&f);
        let toks = tokens(ids.iter().map(String::as_str), "P01-031");
        assert!(toks.contains("kohen"), "other names count");
        let r = report(std::slice::from_ref(&f), "P01-031");
        assert!(r.places.contains_key("300A,00B0>300A,00C3"));
        assert!(r.places.contains_key("0029,1010"));
        assert_eq!(r.removed_tags.get("OtherPatientNames"), Some(&1));
        assert_eq!(r.full_birth_dates, 1);
        // Pseudonymized, private dropped: the beam description still
        // names him.
        let c = changes(&f, "P01-031", false);
        let s = simulate(&p, &c, &UidMaps::default()).unwrap();
        let g = RawFile::parse(s.bytes).unwrap();
        assert_eq!(
            g.text(tags::PATIENT_BIRTH_DATE).as_deref(),
            Some("19560101")
        );
        assert_eq!(scan(&g, &toks), vec!["300A,00B0>300A,00C3".to_string()]);
        // Staff stay.
        assert!(scan(&g, &tokens(["Smith"], "")).len() == 1);
    }
}
