//! CLIP's byte-pair tokenizer.
//!
//! Text prompts have to become the same token ids the text tower was trained
//! on, which means reproducing OpenAI's tokenizer exactly rather than
//! approximating it. It is a small, well-specified algorithm — the two data
//! files it needs (`vocab.json` and `merges.txt`) are fetched alongside the
//! weights, the same way the auto-segmentation module fetches `plans.json`
//! beside its checkpoint.
//!
//! Three details are easy to get wrong and all three are pinned by tests:
//!
//! * text is byte-level. Each pre-token's UTF-8 bytes are mapped into a
//!   printable-character alphabet before any merging, so non-ASCII input can
//!   never fail to tokenize;
//! * the pre-tokenizer splits **each digit separately** — `\p{N}` matches one
//!   character, so "2024" becomes four tokens, not one;
//! * the last symbol of every word carries a `</w>` suffix, which is what
//!   distinguishes a word-final fragment from a word-internal one.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;

/// `<|startoftext|>` in the published vocabulary.
pub const BOS: u32 = 49406;
/// `<|endoftext|>`, which is also the padding token.
pub const EOS: u32 = 49407;
/// Positions the text tower can attend over.
pub const MAX_TOKENS: usize = super::config::CLIP_MAX_POSITIONS;

/// The reversible byte-to-character alphabet GPT-2 and CLIP share.
///
/// Printable ASCII and Latin-1 map to themselves; the remaining 68 byte
/// values are lifted into the private-use range starting at U+0100, so every
/// byte has a distinct printable character and the mapping is invertible.
fn byte_encoder() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut used = [false; 256];
    for b in b'!'..=b'~' {
        out[b as usize] = b as char;
        used[b as usize] = true;
    }
    for b in 0xa1u16..=0xac {
        out[b as usize] = char::from_u32(b as u32).unwrap();
        used[b as usize] = true;
    }
    for b in 0xaeu16..=0xff {
        out[b as usize] = char::from_u32(b as u32).unwrap();
        used[b as usize] = true;
    }
    let mut n = 0u32;
    for (b, u) in used.iter().enumerate() {
        if !u {
            out[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    out
}

/// Collapse whitespace and lowercase, as CLIP's `whitespace_clean` does.
fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = true; // leading whitespace is dropped
    for c in text.chars() {
        if c.is_whitespace() {
            if !space {
                out.push(' ');
                space = true;
            }
        } else {
            out.extend(c.to_lowercase());
            space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// CLIP's pre-tokenizer, hand-rolled from its regex:
/// `'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`.
fn pre_tokenize(text: &str) -> Vec<String> {
    const CONTRACTIONS: [&str; 7] = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"];
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '\'' {
            let rest: String = chars[i..].iter().take(4).collect();
            if let Some(m) = CONTRACTIONS.iter().find(|m| rest.starts_with(**m)) {
                out.push((*m).to_string());
                i += m.chars().count();
                continue;
            }
        }
        if c.is_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        } else if c.is_numeric() {
            // one digit at a time — the regex class is not repeated
            out.push(c.to_string());
            i += 1;
        } else {
            let start = i;
            while i < chars.len()
                && !chars[i].is_whitespace()
                && !chars[i].is_alphabetic()
                && !chars[i].is_numeric()
            {
                i += 1;
            }
            out.push(chars[start..i].iter().collect());
        }
    }
    out
}

/// A loaded CLIP tokenizer.
pub struct Bpe {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    byte_enc: [char; 256],
    decoder: HashMap<u32, String>,
}

impl Bpe {
    /// Build from the contents of `vocab.json` and `merges.txt`.
    pub fn new(vocab_json: &str, merges_txt: &str) -> Result<Bpe> {
        let vocab: HashMap<String, u32> =
            serde_json::from_str(vocab_json).context("parse vocab.json")?;
        if vocab.len() != super::config::CLIP_VOCAB {
            bail!(
                "vocab.json has {} entries, expected {}",
                vocab.len(),
                super::config::CLIP_VOCAB
            );
        }
        let mut ranks = HashMap::new();
        for (i, line) in merges_txt
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with("#version"))
            .enumerate()
        {
            let mut it = line.split_whitespace();
            let (a, b) = (it.next(), it.next());
            match (a, b) {
                (Some(a), Some(b)) => {
                    ranks.insert((a.to_string(), b.to_string()), i as u32);
                }
                _ => bail!("merges.txt line {} is malformed: {line:?}", i + 1),
            }
        }
        if ranks.is_empty() {
            bail!("merges.txt contains no merges");
        }
        let decoder = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Ok(Bpe {
            vocab,
            ranks,
            byte_enc: byte_encoder(),
            decoder,
        })
    }

    /// Load from a directory containing `vocab.json` and `merges.txt`.
    pub fn from_dir(dir: &std::path::Path) -> Result<Bpe> {
        let v = std::fs::read_to_string(dir.join("vocab.json"))
            .with_context(|| format!("read {}", dir.join("vocab.json").display()))?;
        let m = std::fs::read_to_string(dir.join("merges.txt"))
            .with_context(|| format!("read {}", dir.join("merges.txt").display()))?;
        Bpe::new(&v, &m)
    }

    /// Merge one pre-token into vocabulary symbols.
    fn bpe(&self, token: &str) -> Vec<String> {
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if word.is_empty() {
            return word;
        }
        // the final symbol marks the end of a word
        let last = word.len() - 1;
        word[last] = format!("{}</w>", word[last]);
        loop {
            // the lowest-ranked adjacent pair present in the merge table
            let mut best: Option<(usize, u32)> = None;
            for i in 0..word.len().saturating_sub(1) {
                if let Some(r) = self.ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.is_none_or(|(_, br)| *r < br) {
                        best = Some((i, *r));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            // merge every non-overlapping occurrence of that pair
            let (pa, pb) = (word[i].clone(), word[i + 1].clone());
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == pa && word[i + 1] == pb {
                    merged.push(format!("{pa}{pb}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        word
    }

    /// Encode text to token ids, wrapped in the start and end markers and
    /// truncated to [`MAX_TOKENS`].
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![BOS];
        for pre in pre_tokenize(&clean(text)) {
            // byte-level: the pre-token's UTF-8 bytes through the alphabet
            let mapped: String = pre.bytes().map(|b| self.byte_enc[b as usize]).collect();
            for sym in self.bpe(&mapped) {
                if let Some(id) = self.vocab.get(&sym) {
                    ids.push(*id);
                }
            }
        }
        ids.truncate(MAX_TOKENS - 1);
        ids.push(EOS);
        ids
    }

    /// Decode ids back to text. Round-tripping `encode` is the strongest
    /// available check that the vocabulary and merges were loaded correctly.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        let inv: HashMap<char, u8> = self
            .byte_enc
            .iter()
            .enumerate()
            .map(|(b, c)| (*c, b as u8))
            .collect();
        for id in ids {
            if *id == BOS || *id == EOS {
                continue;
            }
            let Some(sym) = self.decoder.get(id) else {
                continue;
            };
            let sym = sym.replace("</w>", " ");
            for c in sym.chars() {
                if let Some(b) = inv.get(&c) {
                    bytes.push(*b);
                } else {
                    // the </w> replacement space is not in the alphabet
                    bytes.push(b' ');
                }
            }
        }
        String::from_utf8_lossy(&bytes).trim().to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

/// The prompt template the model was trained with. Every text prompt goes
/// through this before tokenization.
pub fn prompt_for(structure: &str) -> String {
    format!("A computerized tomography of a {structure}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_byte_alphabet_is_a_bijection() {
        let enc = byte_encoder();
        let mut seen = std::collections::HashSet::new();
        for c in enc {
            assert!(seen.insert(c), "character {c:?} used twice");
        }
        assert_eq!(seen.len(), 256);
        // printable ASCII maps to itself
        assert_eq!(enc[b'a' as usize], 'a');
        assert_eq!(enc[b'!' as usize], '!');
        assert_eq!(enc[b'~' as usize], '~');
        // space is not printable in the source range, so it is lifted
        assert_eq!(enc[b' ' as usize], 'Ġ');
        assert_eq!(enc[b'\n' as usize], 'Ċ');
    }

    #[test]
    fn cleaning_collapses_whitespace_and_lowercases() {
        assert_eq!(
            clean("  A  Computerized\tTOMOGRAPHY \n"),
            "a computerized tomography"
        );
        assert_eq!(clean(""), "");
        assert_eq!(clean("   "), "");
    }

    #[test]
    fn each_digit_is_its_own_pre_token() {
        // \p{N} matches one character, so numbers never merge at this stage.
        assert_eq!(pre_tokenize("t1 2024"), vec!["t", "1", "2", "0", "2", "4"]);
    }

    #[test]
    fn pre_tokenization_splits_words_punctuation_and_contractions() {
        assert_eq!(
            pre_tokenize("a computerized tomography of a liver."),
            vec!["a", "computerized", "tomography", "of", "a", "liver", "."]
        );
        assert_eq!(pre_tokenize("don't"), vec!["don", "'t"]);
        assert_eq!(
            pre_tokenize("it's a lung's"),
            vec!["it", "'s", "a", "lung", "'s"]
        );
        // runs of punctuation stay together, letters and punctuation separate
        assert_eq!(pre_tokenize("l4-l5!!"), vec!["l", "4", "-", "l", "5", "!!"]);
        // non-ASCII is still alphabetic
        assert_eq!(pre_tokenize("Ösophagus"), vec!["Ösophagus"]);
    }

    /// A miniature vocabulary, enough to exercise merging end to end.
    fn toy() -> Bpe {
        // symbols: individual letters, plus merges building up "ab" and "ab c"
        let mut vocab = serde_json::Map::new();
        let mut id = 0u32;
        let put = |v: &mut serde_json::Map<String, serde_json::Value>, s: &str, id: &mut u32| {
            v.insert(s.to_string(), serde_json::json!(*id));
            *id += 1;
        };
        for s in [
            "a", "b", "c", "a</w>", "b</w>", "c</w>", "ab", "ab</w>", "abc</w>", "Ġ", "Ġ</w>",
        ] {
            put(&mut vocab, s, &mut id);
        }
        let vocab_json = serde_json::to_string(&vocab).unwrap();
        // merges are ranked: "a b" first, then "ab c</w>"
        let merges = "#version: 0.2\na b\nab c</w>\n";
        // Bpe::new enforces the real vocabulary size, so build it directly.
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_json).unwrap();
        let mut ranks = HashMap::new();
        for (i, line) in merges.lines().skip(1).filter(|l| !l.is_empty()).enumerate() {
            let mut it = line.split_whitespace();
            ranks.insert(
                (
                    it.next().unwrap().to_string(),
                    it.next().unwrap().to_string(),
                ),
                i as u32,
            );
        }
        let decoder = vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Bpe {
            vocab,
            ranks,
            byte_enc: byte_encoder(),
            decoder,
        }
    }

    #[test]
    fn merging_applies_the_lowest_ranked_pair_first() {
        let t = toy();
        // "abc" -> a,b,c</w> -> (a b) -> ab,c</w> -> (ab c</w>) -> abc</w>
        assert_eq!(t.bpe("abc"), vec!["abc</w>"]);
        // "ab" -> a,b</w>; "a b</w>" is not a merge, so it stays split
        assert_eq!(t.bpe("ab"), vec!["a", "b</w>"]);
        // a single character is just itself, word-final
        assert_eq!(t.bpe("a"), vec!["a</w>"]);
        assert_eq!(t.bpe(""), Vec::<String>::new());
    }

    #[test]
    fn encoding_wraps_in_the_start_and_end_markers() {
        let t = toy();
        let ids = t.encode("abc");
        assert_eq!(ids.first(), Some(&BOS));
        assert_eq!(ids.last(), Some(&EOS));
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[1], t.vocab["abc</w>"]);
    }

    #[test]
    fn encoding_is_truncated_to_the_position_budget() {
        let t = toy();
        let long = "abc ".repeat(200);
        let ids = t.encode(&long);
        assert!(ids.len() <= MAX_TOKENS, "{} tokens", ids.len());
        assert_eq!(ids.last(), Some(&EOS), "the end marker survives truncation");
    }

    #[test]
    fn malformed_data_files_are_rejected() {
        assert!(Bpe::new("not json", "a b\n").is_err());
        // a vocabulary of the wrong size is not the published one
        assert!(Bpe::new(r#"{"a":0}"#, "a b\n").is_err());
    }

    #[test]
    fn the_prompt_template_is_the_trained_one() {
        assert_eq!(prompt_for("liver"), "A computerized tomography of a liver.");
        // and it survives cleaning into the form the tokenizer sees
        assert_eq!(
            clean(&prompt_for("Left Kidney")),
            "a computerized tomography of a left kidney."
        );
    }
}
