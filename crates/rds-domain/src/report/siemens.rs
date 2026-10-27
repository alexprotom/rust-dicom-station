//! The text of a Siemens scanner's report page.
//!
//! syngo writes its *Patientenprotokoll* and dose report pages as
//! Secondary Capture images whose pixels are blank: the text lives in an
//! overlay plane (the rendering) and, as text, in the private element
//! *SIEMENS MEDCOM OOG* (0029,xx10), a STEP (ISO 10303-21) file of
//! `CsaGraphicText` entities. Each entity carries its characters as a list
//! of codes, a reference to a `CsaGraDoubleVec3DArray` with its column
//! position, and the line it sits on. That is what this reads back: the
//! texts, by line and column, and from them the table the page draws (a
//! header line of column names, a unit line under it, rows of numbers
//! aligned under the columns, a label at the left of each row) and the
//! lines around it.

use std::collections::BTreeMap;

use dicom_core::Tag;
use dicom_object::InMemDicomObject;

use super::Table;

/// One text of the page: where it sits (`x` in page pixels, `line` as the
/// page numbers its lines) and what it says.
#[derive(Clone, Debug, PartialEq)]
pub struct TextItem {
    pub x: f64,
    pub line: i64,
    pub text: String,
}

/// The private creator the page's text is filed under.
const CREATOR: &str = "SIEMENS MEDCOM OOG";

/// The STEP text of the page, when the file carries one.
pub fn oog_text(obj: &InMemDicomObject) -> Option<String> {
    // The creator element (0029,00xx) names the block (0029,xx00-xxFF);
    // the text is element xx10 of it.
    for e in 0x10..=0xFFu16 {
        let creator = Tag(0x0029, e);
        let Ok(el) = obj.element(creator) else {
            continue;
        };
        let Ok(name) = el.to_str() else {
            continue;
        };
        if name.trim() != CREATOR {
            continue;
        }
        let data = Tag(0x0029, (e << 8) | 0x10);
        let bytes = obj.element(data).ok()?.to_bytes().ok()?;
        let text: String = bytes.iter().map(|&b| b as char).collect();
        return text.trim_start().starts_with("STEP;").then_some(text);
    }
    None
}

/// The texts of the page, from its STEP entities.
pub fn parse(step: &str) -> Vec<TextItem> {
    // `@id = CsaGraDoubleVec3DArray(0, (x, y, z));` gives the positions.
    let mut positions: BTreeMap<u64, f64> = BTreeMap::new();
    for line in step.lines() {
        let l = line.trim();
        let Some(rest) = l.strip_prefix('@') else {
            continue;
        };
        let Some((id, body)) = rest.split_once('=') else {
            continue;
        };
        let Ok(id) = id.trim().parse::<u64>() else {
            continue;
        };
        let body = body.trim();
        if let Some(args) = body.strip_prefix("CsaGraDoubleVec3DArray(") {
            // `0, (145.0, 0.0, 0.0)` or `0, ()`.
            if let Some(start) = args.find('(') {
                let inner = &args[start + 1..];
                let inner = inner.split(')').next().unwrap_or("");
                if let Some(x) = inner.split(',').next().and_then(|v| v.trim().parse().ok()) {
                    positions.insert(id, x);
                }
            }
        }
    }
    let mut out = Vec::new();
    for line in step.lines() {
        let l = line.trim();
        let Some(rest) = l.strip_prefix('@') else {
            continue;
        };
        let Some((_, body)) = rest.split_once('=') else {
            continue;
        };
        let body = body.trim();
        let Some(args) = body.strip_prefix("CsaGraphicText(") else {
            continue;
        };
        // The position: the first `#id` of the argument list.
        let x = args
            .split(',')
            .map(str::trim)
            .find_map(|a| a.strip_prefix('#'))
            .and_then(|id| id.parse::<u64>().ok())
            .and_then(|id| positions.get(&id).copied())
            .unwrap_or(0.0);
        // The characters: the first parenthesised list of integers; the
        // line: the third plain integer after it.
        let Some(open) = args.find('(') else {
            continue;
        };
        let Some(close) = args[open..].find(')') else {
            continue;
        };
        let codes = &args[open + 1..open + close];
        let text: String = codes
            .split(',')
            .filter_map(|c| c.trim().parse::<u32>().ok())
            .take_while(|&c| c != 0)
            .filter_map(char::from_u32)
            .collect();
        let after = &args[open + close + 1..];
        let line = after
            .split(',')
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .filter_map(|a| a.parse::<i64>().ok())
            .nth(2)
            .unwrap_or(0);
        if text.trim().is_empty() {
            continue;
        }
        out.push(TextItem { x, line, text });
    }
    out
}

/// Does this text read as a number (`110`, `0.06(L)`, `-3`)?
fn numeric(t: &str) -> bool {
    let t = t.trim();
    let mut c = t.chars();
    match c.next() {
        Some(d) if d.is_ascii_digit() => true,
        Some('-' | '+') => c.next().is_some_and(|d| d.is_ascii_digit()),
        _ => false,
    }
}

/// What the page says: its free lines, and the table it draws when it
/// draws one.
pub fn layout(items: &[TextItem]) -> (Vec<String>, Vec<Table>) {
    // By line, by column.
    let mut lines: BTreeMap<i64, Vec<&TextItem>> = BTreeMap::new();
    for it in items {
        lines.entry(it.line).or_default().push(it);
    }
    for v in lines.values_mut() {
        v.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    }
    let ordered: Vec<(i64, Vec<&TextItem>)> = lines.into_iter().collect();

    // The header: the first line of three or more words with no number,
    // followed somewhere by a line with numbers.
    let header_at = ordered.iter().position(|(_, v)| {
        v.len() >= 3
            && v.iter()
                .all(|t| !numeric(&t.text) && !t.text.contains('\n'))
    });
    // The free lines, by line number, so the header's words can join them
    // when no row follows the header after all.
    let mut notes: Vec<(i64, String)> = Vec::new();
    let mut tables = Vec::new();
    let mut columns: Vec<(f64, String)> = Vec::new();
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut unit_line_done = false;
    let mut header_lines: Vec<(i64, String)> = Vec::new();
    for (i, (ln, v)) in ordered.iter().enumerate() {
        let in_header = header_at == Some(i);
        let after_header = header_at.is_some_and(|h| i > h);
        let joined = || {
            v.iter()
                .map(|t| t.text.trim())
                .collect::<Vec<_>>()
                .join(" ")
        };
        if in_header {
            header_lines.push((*ln, joined()));
            // Words closer than a cell's width belong to one column
            // (`mAs`, `/`, `ref.`).
            for t in v {
                match columns.last_mut() {
                    Some((x, name)) if t.x - *x < 20.0 => {
                        name.push(' ');
                        name.push_str(t.text.trim());
                    }
                    _ => columns.push((t.x, t.text.trim().to_string())),
                }
            }
            continue;
        }
        let has_number = v.iter().any(|t| numeric(&t.text));
        if after_header && !unit_line_done && !has_number && !columns.is_empty() {
            // The unit line under the header, when its words sit under
            // columns: `mGy` under `CTDIvol`.
            let under: Vec<Option<usize>> = v.iter().map(|t| nearest(&columns, t.x)).collect();
            if !v.is_empty() && under.iter().all(|u| u.is_some()) && !v[0].text.contains('\n') {
                header_lines.push((*ln, joined()));
                for (t, u) in v.iter().zip(under) {
                    if let Some(c) = u {
                        columns[c].1.push(' ');
                        columns[c].1.push_str(t.text.trim());
                    }
                }
                unit_line_done = true;
                continue;
            }
            unit_line_done = true;
        }
        if after_header && has_number && !columns.is_empty() {
            // A row: the label at the left of the first column, the
            // numbers under theirs, several under one column joined.
            let first_x = columns[0].0;
            let mut row: Vec<String> = vec![String::new(); columns.len() + 1];
            for t in v {
                let text = t.text.trim();
                let cell = if t.x < first_x - 20.0 {
                    0
                } else {
                    nearest(&columns, t.x).map(|c| c + 1).unwrap_or(0)
                };
                if !row[cell].is_empty() {
                    row[cell].push(' ');
                }
                row[cell].push_str(text);
            }
            rows.push(row);
            continue;
        }
        // A free line: its texts in order; a text of several lines pairs
        // its lines with the next text's (`Station:` with the station).
        let parts: Vec<Vec<&str>> = v
            .iter()
            .map(|t| {
                t.text
                    .split(['\r', '\n'])
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .collect();
        let n = parts.iter().map(Vec::len).max().unwrap_or(0);
        for k in 0..n {
            let line: String = parts
                .iter()
                .filter_map(|p| p.get(k))
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !line.is_empty() {
                notes.push((*ln, line));
            }
        }
    }
    if rows.is_empty() {
        // No table after all: the words taken for a header were lines.
        notes.extend(header_lines);
        notes.sort_by_key(|(ln, _)| *ln);
    }
    let notes: Vec<String> = notes.into_iter().map(|(_, l)| l).collect();
    if !rows.is_empty() {
        let mut cols: Vec<String> = vec![String::new()];
        cols.extend(columns.into_iter().map(|(_, n)| n));
        // A label column nobody used is not drawn.
        if rows.iter().all(|r| r[0].is_empty()) {
            cols.remove(0);
            for r in &mut rows {
                r.remove(0);
            }
        }
        tables.push(Table {
            title: String::new(),
            columns: cols,
            rows,
        });
    }
    (notes, tables)
}

/// The column whose position is nearest to `x`, within a cell's reach.
fn nearest(columns: &[(f64, String)], x: f64) -> Option<usize> {
    columns
        .iter()
        .enumerate()
        .map(|(i, (cx, _))| (i, (cx - x).abs()))
        .filter(|(_, d)| *d < 60.0)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(id: u64, pos: u64, line: i64, s: &str) -> String {
        let codes: Vec<String> = s.bytes().map(|b| b.to_string()).collect();
        format!(
            "@{id} = CsaGraphicText(0, 0, 0, 3, 3.0, 0, #{pos}, 0, 1, 0, 0, #2252, $, $, $, $, 0, \
             0.0, 0.0, 0.0, 0, 0, 0, 0, 65535, 1.0, 1.0, 1.0, 0.0, 0, 0.299, 0.587, 0.114, 0, \
             65535, 0.0, 0.0, 0.0, 0.0, 0, 0.299, 0.587, 0.114, ({}, 0), 1, -1, {line}, 100, \
             -16, 0, 0, 0, 400, 0, 0, 0, 0, 3, 2, 1, 34, 77, 77, 0, 0, 1);",
            codes.join(", ")
        )
    }

    fn pos(id: u64, x: f64) -> String {
        format!("@{id} = CsaGraDoubleVec3DArray(0, ({x:.1}, 0.0, 0.0));")
    }

    /// A page as syngo writes one: the date, the station block, the totals,
    /// a two-line header, the patient position, three rows, the phantom.
    fn page() -> String {
        let mut s = String::from("STEP;\nHEADER;\nENDSEC;\nDATA;\n");
        let mut id = 10;
        let mut put = |x: f64, line: i64, t: &str| {
            s.push_str(&pos(id, x));
            s.push('\n');
            s.push_str(&text(id + 1, id, line, t));
            s.push('\n');
            id += 2;
        };
        put(0.0, 5, "  07-Mar-2025  08:31");
        put(
            0.0,
            7,
            "  Station:\r\n  Untersuchender Arzt:\r\n  Assistenz:",
        );
        put(145.0, 7, " SIHDCT1\r\n \r\n ");
        put(0.0, 11, "  Gesamt mAs 2817      Gesamt DLP 511 mGy*cm");
        for (x, t) in [
            (167.0, "Scan"),
            (197.0, "kV"),
            (232.0, "mAs"),
            (240.0, "/"),
            (245.0, "ref."),
            (370.0, "CTDIvol"),
            (425.0, "DLP"),
            (460.0, "TI"),
            (500.0, "cSL"),
        ] {
            put(x, 13, t);
        }
        for (x, t) in [
            (370.0, "mGy"),
            (425.0, "mGy*cm"),
            (460.0, "s"),
            (500.0, "mm"),
        ] {
            put(x, 14, t);
        }
        put(0.0, 16, "  Patientenposition H-SP");
        put(0.0, 17, "  Topogram");
        for (x, t) in [
            (167.0, "1"),
            (197.0, "110"),
            (370.0, "0.06(L)"),
            (425.0, "2.89"),
            (460.0, "5.6"),
            (500.0, "0.6"),
        ] {
            put(x, 17, t);
        }
        put(0.0, 18, "  Steinsuche");
        for (x, t) in [
            (167.0, "2"),
            (197.0, "110"),
            (232.0, "38"),
            (240.0, "/"),
            (270.0, "50"),
            (370.0, "2.80(L)"),
            (425.0, "119.11"),
            (460.0, "0.6"),
            (500.0, "1.2"),
        ] {
            put(x, 18, t);
        }
        put(500.0, 1023, "Phantom Type (L) 32cm (S) 16cm");
        s.push_str("ENDSEC;\nENDSTEP;\n");
        s
    }

    #[test]
    fn the_texts_come_back_with_their_column_and_line() {
        let items = parse(&page());
        let kv = items.iter().find(|t| t.text == "kV").expect("kV");
        assert_eq!(kv.x, 197.0);
        assert_eq!(kv.line, 13);
        let date = items
            .iter()
            .find(|t| t.text.contains("07-Mar-2025"))
            .unwrap();
        assert_eq!(date.line, 5);
    }

    #[test]
    fn the_page_lays_out_as_its_table_and_its_lines() {
        let (notes, tables) = layout(&parse(&page()));
        assert_eq!(tables.len(), 1, "{tables:?}");
        let t = &tables[0];
        assert_eq!(
            t.columns,
            vec![
                "",
                "Scan",
                "kV",
                "mAs / ref.",
                "CTDIvol mGy",
                "DLP mGy*cm",
                "TI s",
                "cSL mm"
            ]
        );
        assert_eq!(t.rows.len(), 2);
        assert_eq!(
            t.rows[0],
            vec!["Topogram", "1", "110", "", "0.06(L)", "2.89", "5.6", "0.6"]
        );
        assert_eq!(
            t.rows[1],
            vec![
                "Steinsuche",
                "2",
                "110",
                "38 / 50",
                "2.80(L)",
                "119.11",
                "0.6",
                "1.2"
            ]
        );
        assert_eq!(notes[0], "07-Mar-2025  08:31");
        assert_eq!(notes[1], "Station: SIHDCT1");
        assert_eq!(notes[2], "Untersuchender Arzt:");
        assert!(notes.iter().any(|n| n.starts_with("Gesamt mAs 2817")));
        assert!(notes.iter().any(|n| n == "Patientenposition H-SP"));
        assert!(notes.iter().any(|n| n.starts_with("Phantom Type")));
    }

    #[test]
    fn numbers_are_told_from_words() {
        assert!(numeric("110"));
        assert!(numeric("0.06(L)"));
        assert!(numeric("-3"));
        assert!(!numeric("kV"));
        assert!(!numeric("/"));
        assert!(!numeric("H-SP"));
    }
}
