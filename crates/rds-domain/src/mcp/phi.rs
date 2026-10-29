//! The gate and the door: who a dataset names, and the one place every
//! string passes on its way out of the process.
//!
//! Whatever the server returns ends up in a language model's context, and
//! for most clients that context leaves the machine. So the rule is not
//! "redact patient names": it is that no tool *has* a patient name to
//! return, and that everything else that could carry one - error texts
//! quoting a file name, a series description a site typed a name into, a
//! folder named after the patient - is scrubbed by [`Redactor`] before it
//! becomes part of a frame.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rayon::prelude::*;
use serde_json::Value;

use crate::anonymize;
use crate::dicomfile;
use crate::loader::{self, LoadedStudy};

/// Longest DICOM free-text value a tool result carries. Descriptions and
/// structure names are data from the scanner, not prose; the cap keeps a
/// pathological one from becoming a page of the result.
pub const TEXT_CAP: usize = 64;

/// What the headers of a dataset say about the person in it.
pub struct PhiVerdict {
    /// Names of the identity tags that hold a value the anonymizer did not
    /// write. Empty means anonymized.
    pub tags: Vec<String>,
    /// How many files were read for the verdict.
    pub files_checked: usize,
    /// The values themselves. Never leave this struct except into a
    /// [`Redactor`]; nothing else needs them.
    values: Vec<String>,
}

impl PhiVerdict {
    pub fn is_anonymized(&self) -> bool {
        self.tags.is_empty()
    }

    /// The values, for scrubbing a study in memory. They must not travel
    /// further than that.
    pub fn values(&self) -> &[String] {
        &self.values
    }

    /// The sentence a refusal carries: which tags, never their values.
    pub fn describe(&self) -> String {
        if self.is_anonymized() {
            return format!("anonymized ({} files checked)", self.files_checked);
        }
        let shown: Vec<&str> = self.tags.iter().take(3).map(String::as_str).collect();
        let more = self.tags.len().saturating_sub(shown.len());
        let list = if more > 0 {
            format!("{} and {more} more", shown.join(", "))
        } else {
            shown.join(", ")
        };
        format!(
            "carries identifying data in {} tag{} ({list})",
            self.tags.len(),
            if self.tags.len() == 1 { "" } else { "s" }
        )
    }
}

/// The files worth reading for a verdict: the first file of every image
/// series (the patient tags do not change along a series) and every file of
/// the folder that belongs to no image series - the RT objects.
pub fn sample_files(study: &LoadedStudy, dir: Option<&Path>) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut in_series: BTreeSet<PathBuf> = BTreeSet::new();
    for s in &study.series {
        if let Some(f) = s.files.first() {
            out.push(f.clone());
        }
        in_series.extend(s.files.iter().cloned());
    }
    if let Some(dir) = dir {
        for e in walkdir::WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
        {
            let p = e.into_path();
            if !in_series.contains(&p) {
                out.push(p);
            }
        }
    }
    out
}

/// Read the identity tags of `files` and say what they hold.
///
/// A file that is not DICOM contributes nothing (a folder often holds a
/// readme). The tag list is the anonymizer's own, so the two never
/// disagree about what "identifying" means.
pub fn classify(files: &[PathBuf]) -> Result<PhiVerdict> {
    let tags = anonymize::identity_tags();
    let per_file: Vec<Vec<(usize, String)>> = files
        .par_iter()
        .map(|path| {
            let Ok(obj) = dicomfile::open_header(path) else {
                return Vec::new();
            };
            let mut found = Vec::new();
            for (i, (tag, _)) in tags.iter().enumerate() {
                if let Some(v) = loader::str_of(&obj, *tag) {
                    let v = v.trim().trim_end_matches('\0').trim().to_string();
                    if !anonymize::looks_anonymized(&v) {
                        found.push((i, v));
                    }
                }
            }
            found
        })
        .collect();
    let mut hit = vec![false; tags.len()];
    let mut values: BTreeSet<String> = BTreeSet::new();
    for f in per_file {
        for (i, v) in f {
            hit[i] = true;
            values.insert(v);
        }
    }
    Ok(PhiVerdict {
        tags: tags
            .iter()
            .zip(&hit)
            .filter(|(_, &h)| h)
            .map(|((_, name), _)| (*name).to_string())
            .collect(),
        files_checked: files.len(),
        values: values.into_iter().collect(),
    })
}

/// A string that has been through the redactor. The only constructor is
/// [`Redactor::text`], so a tool that formats a `String` straight into a
/// result does not compile: the door is a type, not a convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Public(String);

impl Public {
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn into_string(self) -> String {
        self.0
    }
}

/// One needle: the value as found and, for person names, its components.
///
/// Values of two characters are kept: a two-letter family name is a name.
/// What keeps such a needle from tearing words and UIDs apart is how it is
/// matched ([`SHORT_NEEDLE`]), not a higher floor.
fn needles_of(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    let v = value.trim();
    if v.len() >= 2 {
        out.push(v.to_string());
    }
    // A PN "DOE^JANE" shows up elsewhere as "Doe Jane" or as "Doe" alone.
    if v.contains('^') {
        out.push(v.replace('^', " ").trim().to_string());
        for part in v.split('^') {
            let part = part.trim();
            if part.len() >= 3 {
                out.push(part.to_string());
            }
        }
    }
    out
}

/// Every string that leaves the process passes through here.
#[derive(Default, Clone)]
pub struct Redactor {
    /// Identifying values seen in any open dataset, longest first. Where a
    /// full name and its components both match, the longest match at a
    /// position wins, so the full name is replaced whole.
    needles: Vec<String>,
    /// Root folders as `(prefix, label)`: a path under a root is reported
    /// relative to it, so a folder named after the patient never appears.
    roots: Vec<(String, String)>,
}

/// What an identifying value becomes.
pub const ALIAS: &str = "[redacted]";

impl Redactor {
    pub fn new() -> Redactor {
        Redactor::default()
    }

    /// Remember identifying values. Duplicates are dropped; the list is
    /// kept longest-first.
    pub fn add_values(&mut self, values: Vec<String>) {
        for v in values {
            for n in needles_of(&v) {
                if !self.needles.iter().any(|e| e.eq_ignore_ascii_case(&n)) {
                    self.needles.push(n);
                }
            }
        }
        self.needles
            .sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
    }

    /// Register a root folder under `label` (for example `root1`). Both the
    /// path as configured and its canonical form are recognised, with either
    /// kind of separator, with and without Windows' `\\?\` prefix (a
    /// canonical path on Windows carries it, and the label must take its
    /// place too, not follow it).
    pub fn add_root(&mut self, path: &Path, label: &str) {
        let mut forms: Vec<String> = Vec::new();
        let mut push = |p: &Path| {
            let s = p.to_string_lossy().to_string();
            let s = s.strip_prefix(r"\\?\").unwrap_or(&s).to_string();
            let verbatim = format!(r"\\?\{}", s.replace('/', "\\"));
            for f in [
                s.clone(),
                s.replace('\\', "/"),
                s.replace('/', "\\"),
                verbatim,
            ] {
                let f = f.trim_end_matches(['/', '\\']).to_string();
                if !f.is_empty() && !forms.contains(&f) {
                    forms.push(f);
                }
            }
        };
        push(path);
        if let Ok(c) = path.canonicalize() {
            push(&c);
        }
        for f in forms {
            self.roots.push((f, label.to_string()));
        }
        self.roots.sort_by_key(|r| std::cmp::Reverse(r.0.len()));
    }

    /// A path for the client: relative to the root it is under, or its
    /// file name alone when it is under none (an output path is under the
    /// output folder, which is not a root).
    pub fn path(&self, p: &Path) -> Public {
        self.text(&p.to_string_lossy())
    }

    /// Scrub one string: root folders become their labels and identifying
    /// values the alias, in one pass ([`replace_all`]).
    pub fn text(&self, s: &str) -> Public {
        let mut pats: Vec<Pattern> = Vec::with_capacity(self.roots.len() + self.needles.len());
        pats.extend(
            self.roots
                .iter()
                .map(|(prefix, label)| Pattern::root(prefix, label)),
        );
        pats.extend(self.needles.iter().map(|n| Pattern::value(n, ALIAS)));
        Public(replace_all(s, &pats))
    }

    /// Scrub every string inside a JSON value, keys included.
    pub fn json(&self, v: Value) -> Value {
        match v {
            Value::String(s) => Value::String(self.text(&s).0),
            Value::Array(a) => Value::Array(a.into_iter().map(|x| self.json(x)).collect()),
            Value::Object(m) => Value::Object(
                m.into_iter()
                    .map(|(k, x)| (self.text(&k).0, self.json(x)))
                    .collect(),
            ),
            other => other,
        }
    }

    /// Replace every one of `values` in `hay` by `with`, matched as
    /// [`Redactor::text`] matches its values, for callers scrubbing a study
    /// in memory before anything reads it.
    pub fn scrub_with(hay: &str, values: &[String], with: &str) -> String {
        let needles: Vec<String> = values.iter().flat_map(|v| needles_of(v)).collect();
        let pats: Vec<Pattern> = needles.iter().map(|n| Pattern::value(n, with)).collect();
        replace_all(hay, &pats)
    }

    /// How many identifying values are known (for `describe_session`).
    pub fn known_values(&self) -> usize {
        self.needles.len()
    }
}

/// Below this many characters a value matches only as a whole token: with
/// no ASCII letter or digit right before or after it. A short value is
/// mostly a fragment of ordinary words ("in" in "Supine", "IA" in
/// "AXIAL"); a long one is a name, and is still found glued into a folder
/// name (`Doe-Jane_2024`).
pub const SHORT_NEEDLE: usize = 5;

/// One thing [`replace_all`] looks for.
struct Pattern<'a> {
    /// ASCII lower case, compared with the lower-cased text.
    lower: String,
    with: &'a str,
    /// An identifying value, held to the token and number rules; a root
    /// folder is replaced wherever it is.
    value: bool,
    /// Under [`SHORT_NEEDLE`] characters.
    short: bool,
}

impl<'a> Pattern<'a> {
    fn root(prefix: &str, label: &'a str) -> Pattern<'a> {
        Pattern {
            lower: prefix.to_ascii_lowercase(),
            with: label,
            value: false,
            short: false,
        }
    }

    fn value(needle: &str, with: &'a str) -> Pattern<'a> {
        Pattern {
            lower: needle.to_ascii_lowercase(),
            with,
            value: true,
            short: needle.chars().count() < SHORT_NEEDLE,
        }
    }

    /// Whether this pattern, found at bytes `start..end` of `h`, counts.
    fn counts(&self, h: &[u8], start: usize, end: usize) -> bool {
        if !self.value {
            return true;
        }
        let before = start.checked_sub(1).map(|i| h[i]);
        let after = h.get(end).copied();
        if self.short
            && (before.is_some_and(|b| b.is_ascii_alphanumeric())
                || after.is_some_and(|b| b.is_ascii_alphanumeric()))
        {
            return false;
        }
        // Never strictly inside a number or a UID: a value that starts
        // (ends) with a digit or a dot does not count where the text goes
        // on with one, so a date or an ID is not found inside
        // `1.2.752.243.1.1.20261018145021580.3000`, while a whole one
        // still is (`id 123456,`, `CT_19650312.dcm`).
        let numeric = |b: u8| b.is_ascii_digit() || b == b'.';
        let first = self.lower.as_bytes()[0];
        let last = self.lower.as_bytes()[self.lower.len() - 1];
        let left_number = match before {
            Some(b) if b.is_ascii_digit() => true,
            Some(b'.') => start >= 2 && h[start - 2].is_ascii_digit(),
            _ => false,
        };
        let right_number = match after {
            Some(b) if b.is_ascii_digit() => true,
            Some(b'.') => h.get(end + 1).is_some_and(|b| b.is_ascii_digit()),
            _ => false,
        };
        !((numeric(first) && left_number) || (numeric(last) && right_number))
    }
}

/// Replace every pattern in `hay` in one pass, case-insensitively (ASCII):
/// at each position the longest pattern that matches there and
/// [counts](Pattern::counts) is replaced, and the scan goes on after it.
/// The replacement is never scanned again, so an alias is never cut into
/// by a value that happens to be part of it (`[re[redacted]cted]`), and
/// one value is never found inside another that has been replaced already.
fn replace_all(hay: &str, pats: &[Pattern]) -> String {
    let pats: Vec<&Pattern> = pats.iter().filter(|p| !p.lower.is_empty()).collect();
    if pats.is_empty() {
        return hay.to_string();
    }
    // Lower-casing ASCII changes no byte offsets, so positions in `lower`
    // are positions in `hay`; every needle is valid UTF-8, so a match
    // starts and ends on a character boundary.
    let lower = hay.to_ascii_lowercase();
    let h = lower.as_bytes();
    let mut out = String::with_capacity(hay.len());
    let (mut i, mut copied) = (0, 0);
    while i < h.len() {
        let best = pats
            .iter()
            .filter(|p| h[i..].starts_with(p.lower.as_bytes()) && p.counts(h, i, i + p.lower.len()))
            .max_by_key(|p| p.lower.len());
        match best {
            Some(p) => {
                out.push_str(&hay[copied..i]);
                out.push_str(p.with);
                i += p.lower.len();
                copied = i;
            }
            None => {
                // On to the next character, not the next byte.
                i += hay[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
    }
    out.push_str(&hay[copied..]);
    out
}

/// A DICOM free-text value fit for a result: control characters removed,
/// cut at [`TEXT_CAP`] characters.
pub fn clean_text(s: &str) -> String {
    let cleaned: String = s
        .trim_end_matches('\0')
        .chars()
        .filter(|c| !c.is_control())
        .collect();
    let cleaned = cleaned.trim();
    if cleaned.chars().count() <= TEXT_CAP {
        cleaned.to_string()
    } else {
        let mut cut: String = cleaned.chars().take(TEXT_CAP - 1).collect();
        cut.push('…');
        cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_and_their_parts_are_replaced_case_insensitively() {
        let mut r = Redactor::new();
        r.add_values(vec!["DOE^JANE".into(), "123456".into()]);
        let s = r.text("Patient Doe^Jane (doe jane), id 123456, Jane's CT");
        assert_eq!(
            s.as_str(),
            "Patient [redacted] ([redacted]), id [redacted], [redacted]'s CT"
        );
    }

    #[test]
    fn short_values_are_not_needles() {
        let mut r = Redactor::new();
        r.add_values(vec!["O".into(), "M".into()]);
        assert_eq!(r.text("MODALITY CT").as_str(), "MODALITY CT");
    }

    #[test]
    fn short_values_match_whole_tokens_only() {
        let mut r = Redactor::new();
        r.add_values(vec!["in".into(), "da".into(), "50".into(), "IA".into()]);
        for s in [
            "Supine",
            "modality",
            "[redacted]",
            "ORIGINAL / PRIMARY / AXIAL",
            "1.2.752.243.1.1.20261018145021580.3000.24234",
            "NoDate_Supine-CT",
        ] {
            assert_eq!(r.text(s).as_str(), s, "{s} is left alone");
        }
        // Standing alone, a two-letter value is still a value.
        assert_eq!(
            r.text("Patient DA here").as_str(),
            "Patient [redacted] here"
        );
        assert_eq!(r.text("da").as_str(), "[redacted]");
        assert_eq!(r.text("dose 50 Gy").as_str(), "dose [redacted] Gy");
    }

    #[test]
    fn long_values_are_found_inside_words_but_not_inside_numbers() {
        let mut r = Redactor::new();
        r.add_values(vec![
            "DOE^JANE".into(),
            "in".into(),
            "da".into(),
            "20240517".into(),
        ]);
        assert_eq!(
            r.text("Doe^Jane and DOE JANE").as_str(),
            "[redacted] and [redacted]"
        );
        assert_eq!(
            r.text("Doe-Jane_2024").as_str(),
            "[redacted]-[redacted]_2024"
        );
        // A date is a value on its own or in a file name, never as the
        // middle of a UID.
        assert_eq!(r.text("born 20240517").as_str(), "born [redacted]");
        assert_eq!(r.text("CT_20240517.dcm").as_str(), "CT_[redacted].dcm");
        let uid = "1.2.3.20240517.4";
        assert_eq!(r.text(uid).as_str(), uid);
        assert_eq!(r.text("1.2.320240517991").as_str(), "1.2.320240517991");
        // The alias is never scanned again, whatever it contains.
        assert_eq!(
            r.text("Jane in Doe^Jane").as_str(),
            "[redacted] [redacted] [redacted]"
        );
    }

    #[test]
    fn roots_and_values_are_replaced_in_one_pass() {
        let mut r = Redactor::new();
        r.add_root(Path::new(r"D:\PACS"), "archive");
        r.add_values(vec!["ch".into(), "Doe^Jane".into()]);
        assert_eq!(
            r.text(r"D:\PACS\Doe-Jane_CT\x.dcm").as_str(),
            r"archive\[redacted]-[redacted]_CT\x.dcm",
            "the label is not cut into by a value it contains"
        );
    }

    #[test]
    fn scrubbing_in_memory_follows_the_same_rules() {
        let values = vec!["DOE^JANE".to_string(), "in".to_string()];
        assert_eq!(
            Redactor::scrub_with("Supine Doe Jane in", &values, "anon_1"),
            "Supine anon_1 anon_1"
        );
    }

    #[test]
    fn roots_become_labels_with_either_separator() {
        let mut r = Redactor::new();
        r.add_root(Path::new(r"D:\studies\Rambam_patient_2"), "root1");
        assert_eq!(
            r.text(r"read D:\studies\Rambam_patient_2\4DCT\img.dcm")
                .as_str(),
            r"read root1\4DCT\img.dcm"
        );
        assert_eq!(
            r.text("read D:/studies/Rambam_patient_2/4DCT").as_str(),
            "read root1/4DCT"
        );
        // A canonical path on Windows: the label takes the prefix's place.
        assert_eq!(
            r.text(r"\\?\D:\studies\Rambam_patient_2\4DCT").as_str(),
            r"root1\4DCT"
        );
    }

    #[test]
    fn json_is_scrubbed_throughout() {
        let mut r = Redactor::new();
        r.add_values(vec!["Smith".into()]);
        let v = r.json(serde_json::json!({"a": ["x Smith y"], "Smith": {"b": "SMITH"}}));
        assert_eq!(
            v,
            serde_json::json!({"a": ["x [redacted] y"], "[redacted]": {"b": "[redacted]"}})
        );
    }

    #[test]
    fn free_text_is_capped_and_cleaned() {
        assert_eq!(clean_text("CT\u{0}\u{1} thorax\0"), "CT thorax");
        let long = "x".repeat(200);
        assert_eq!(clean_text(&long).chars().count(), TEXT_CAP);
    }

    #[test]
    fn the_anonymizers_own_output_passes() {
        assert!(anonymize::looks_anonymized(""));
        assert!(anonymize::looks_anonymized("anon_0a1b2c"));
        assert!(!anonymize::looks_anonymized("anon_patient"));
        assert!(!anonymize::looks_anonymized("DOE^JANE"));
    }
}
