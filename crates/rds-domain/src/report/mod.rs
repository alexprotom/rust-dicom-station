//! Reports: the objects of a study that are read rather than looked at.
//!
//! A CT study carries, beside its images, what the scanner wrote about
//! them: an *X-Ray Radiation Dose SR* (TID 10011) with every
//! acquisition's technique and dose indices, and on a Siemens scanner the
//! *Patientenprotokoll*, the same table as a page (a Secondary Capture
//! whose pixels are blank and whose text is an overlay plane, with the
//! text itself in a private element). Other Structured Reports come the
//! same way: a measurement report, a radiology report. None of them is an
//! image of the patient, and a pane that windows their pixels shows
//! nothing; what a person wants is the table.
//!
//! So a [`Report`] is a few facts, the lines and tables of its content,
//! and for a page the rendered page beside them. The SR content tree
//! ([`sr`]) is read whole; a template this knows ([`dose`]) becomes its
//! own table, any other is laid out as an indented two-column table, so no
//! SR is lost. A page's text ([`siemens`]) is laid out from the positions
//! of its words. An encapsulated PDF ([`document`]) is a report whose
//! content is its pages, drawn when first shown. The viewer shows a report
//! in the row's *Additional* pane as a table (a document as its pages),
//! the MCP server returns its tables (and never a name an SR item carries,
//! nor a document's title or bytes), and `describe_archive_study` says
//! which files are reports before anything opens them.

pub mod document;
pub mod dose;
pub mod siemens;
pub mod sr;

use std::path::Path;

use anyhow::{Context, Result};
use dicom_dictionary_std::tags;

use crate::extras::PlanarImage;
use crate::loader::str_of;

/// A table of a report: its columns and its rows of text.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Table {
    /// Empty for the one table of a report.
    pub title: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

/// What kind of report this is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReportKind {
    /// A Structured Report of a template nothing here interprets: shown
    /// as its content tree.
    #[default]
    Structured,
    /// An X-Ray Radiation Dose SR of a CT scanner (TID 10011).
    DoseReport,
    /// A scanner's page: a Secondary Capture whose text is read back.
    ProtocolPage,
    /// An encapsulated document (a PDF): shown as its pages.
    Document,
}

impl ReportKind {
    pub fn label(self) -> &'static str {
        match self {
            ReportKind::Structured => "structured report",
            ReportKind::DoseReport => "dose report",
            ReportKind::ProtocolPage => "protocol page",
            ReportKind::Document => "document",
        }
    }
}

/// One report of a study.
#[derive(Clone, Default)]
pub struct Report {
    pub label: String,
    pub kind: ReportKind,
    pub modality: String,
    /// The content date, `2025-03-07`; empty when not stated.
    pub date: String,
    /// Facts above the tables: the scanner, the totals, the procedure.
    pub info: Vec<(String, String)>,
    /// Free lines of a page.
    pub notes: Vec<String>,
    /// The tables of the content. Never a person's name: an SR's names
    /// stay in [`Report::tree`].
    pub tables: Vec<Table>,
    /// The SR content tree; empty for a page.
    pub tree: Vec<sr::Node>,
    /// The page as the scanner rendered it, for a page.
    pub page: Option<PlanarImage>,
    /// The document, for an encapsulated one.
    pub document: Option<document::Document>,
    pub file_name: String,
    /// Study Instance UID: the study the data tree files it under.
    pub study_uid: String,
}

impl Report {
    /// The SR tree as an indented two-column table, names withheld on
    /// request; empty when there is no tree.
    pub fn tree_table(&self, hide_names: bool) -> Option<Table> {
        if self.tree.is_empty() {
            return None;
        }
        let root = sr::Node {
            children: self.tree.clone(),
            ..Default::default()
        };
        let rows: Vec<Vec<String>> = root
            .walk()
            .into_iter()
            .map(|(depth, n)| {
                let name = if n.name.is_empty() {
                    n.value_type.clone()
                } else {
                    n.name.clone()
                };
                vec![
                    format!("{}{name}", "    ".repeat(depth)),
                    n.value.text(hide_names),
                ]
            })
            .collect();
        Some(Table {
            title: "Content".to_string(),
            columns: vec!["Item".to_string(), "Value".to_string()],
            rows,
        })
    }

    /// Everything shown: the tables, then the full content of an SR.
    pub fn all_tables(&self, hide_names: bool) -> Vec<Table> {
        let mut out = self.tables.clone();
        if let Some(t) = self.tree_table(hide_names) {
            out.push(t);
        }
        out
    }
}

/// Is this SOP class a Structured Report's? The SR storage classes share
/// the `1.2.840.10008.5.1.4.1.1.88.` prefix; the Key Object Selection,
/// encoded the same way, holds references and no content, and is not one
/// ([`crate::sop::is_structured_report`]).
pub fn is_sr_sop(sop_class: &str) -> bool {
    crate::sop::is_structured_report(sop_class)
}

/// Read a Structured Report file, or an encapsulated document, into a
/// report.
pub fn load_report(path: &Path) -> Result<Report> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open report {}", path.display()))?;
    if str_of(&obj, tags::SOP_CLASS_UID).as_deref() == Some(crate::sop::SOP_ENCAPSULATED_PDF) {
        return load_document(&obj, path);
    }
    let root = sr::read(&obj);
    let series = str_of(&obj, tags::SERIES_DESCRIPTION);
    let date = str_of(&obj, tags::CONTENT_DATE)
        .or_else(|| str_of(&obj, tags::STUDY_DATE))
        .map(|d| sr::show_date(&d))
        .unwrap_or_default();
    let mut r = Report {
        modality: str_of(&obj, tags::MODALITY).unwrap_or_else(|| "SR".into()),
        study_uid: str_of(&obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        date,
        file_name: path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default(),
        ..Default::default()
    };
    if dose::is_dose_report(&root) {
        r.kind = ReportKind::DoseReport;
        r.label = series.unwrap_or_else(|| "Dose report".into());
        r.info = dose::facts(&root);
        let t = dose::acquisitions(&root);
        if !t.rows.is_empty() {
            r.tables.push(t);
        }
    } else {
        r.kind = ReportKind::Structured;
        r.label = series
            .or_else(|| (!root.name.is_empty()).then(|| root.name.clone()))
            .unwrap_or_else(|| "Structured report".into());
        if !root.name.is_empty() {
            r.info.push(("Document".into(), root.name.clone()));
        }
        for (k, tag) in [
            ("Completion", tags::COMPLETION_FLAG),
            ("Verification", tags::VERIFICATION_FLAG),
        ] {
            if let Some(v) = str_of(&obj, tag) {
                r.info.push((k.into(), v));
            }
        }
    }
    r.tree = root.children;
    Ok(r)
}

/// An encapsulated document as a report: named after its series (the
/// title can carry identifiers, see [`document`]), its facts what the
/// header says about it, its content the document itself.
fn load_document(obj: &dicom_object::InMemDicomObject, path: &Path) -> Result<Report> {
    let Some(doc) = document::read(obj) else {
        anyhow::bail!("the encapsulated document of {} is empty", path.display());
    };
    let date = str_of(obj, tags::CONTENT_DATE)
        .or_else(|| str_of(obj, tags::STUDY_DATE))
        .map(|d| sr::show_date(&d))
        .unwrap_or_default();
    let mut info = vec![
        (
            "Type".to_string(),
            if doc.mime.is_empty() {
                "PDF".to_string()
            } else {
                doc.mime.clone()
            },
        ),
        ("Size".to_string(), show_size(doc.bytes.len())),
    ];
    for (k, tag) in [
        ("Completion", tags::COMPLETION_FLAG),
        ("Verification", tags::VERIFICATION_FLAG),
    ] {
        if let Some(v) = str_of(obj, tag) {
            info.push((k.into(), v));
        }
    }
    Ok(Report {
        label: str_of(obj, tags::SERIES_DESCRIPTION).unwrap_or_else(|| "PDF document".into()),
        kind: ReportKind::Document,
        modality: str_of(obj, tags::MODALITY).unwrap_or_else(|| "DOC".into()),
        date,
        info,
        study_uid: str_of(obj, tags::STUDY_INSTANCE_UID).unwrap_or_default(),
        file_name: path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default(),
        document: Some(doc),
        ..Default::default()
    })
}

/// A byte count as a person reads it: `812 B`, `34.1 kB`, `2.4 MB`.
fn show_size(n: usize) -> String {
    match n {
        0..1000 => format!("{n} B"),
        1000..1_000_000 => format!("{:.1} kB", n as f64 / 1e3),
        _ => format!("{:.1} MB", n as f64 / 1e6),
    }
}

/// What a single-image file turns out to be: an image, or a page whose
/// text makes it a report.
// One per file read, handed straight on: the size of the larger variant
// costs nothing worth a box.
#[allow(clippy::large_enum_variant)]
pub enum PageObject {
    Image(PlanarImage),
    Report(Report),
}

/// Read a single-image file: a radiograph or a secondary capture as the
/// image it is, a scanner's page as a report with the page beside it.
pub fn load_page(path: &Path) -> Result<PageObject> {
    let obj = crate::dicomfile::open_full(path)
        .with_context(|| format!("open planar image {}", path.display()))?;
    let image = crate::extras::planar_of(&obj, path)?;
    let Some(step) = siemens::oog_text(&obj) else {
        return Ok(PageObject::Image(image));
    };
    let items = siemens::parse(&step);
    if items.is_empty() {
        return Ok(PageObject::Image(image));
    }
    let (notes, tables) = siemens::layout(&items);
    let date = str_of(&obj, tags::CONTENT_DATE)
        .or_else(|| str_of(&obj, tags::STUDY_DATE))
        .map(|d| sr::show_date(&d))
        .unwrap_or_default();
    Ok(PageObject::Report(Report {
        label: image.label.clone(),
        kind: ReportKind::ProtocolPage,
        modality: image.modality.clone(),
        date,
        info: Vec::new(),
        notes,
        tables,
        tree: Vec::new(),
        file_name: path
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default(),
        study_uid: image.study_uid.clone(),
        page: Some(image),
        document: None,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sr_storage_classes_are_told_by_their_prefix() {
        assert!(is_sr_sop("1.2.840.10008.5.1.4.1.1.88.67"));
        assert!(is_sr_sop("1.2.840.10008.5.1.4.1.1.88.11"));
        assert!(!is_sr_sop("1.2.840.10008.5.1.4.1.1.7"));
        assert!(!is_sr_sop(""));
    }

    #[test]
    fn the_tree_table_indents_by_depth_and_withholds_names_on_request() {
        let r = Report {
            tree: vec![sr::Node {
                name: "Observer".into(),
                value_type: "CONTAINER".into(),
                children: vec![sr::Node {
                    name: "Person Observer Name".into(),
                    value_type: "PNAME".into(),
                    value: sr::Value::Name("HOUSE^GREGORY".into()),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        let shown = r.tree_table(false).unwrap();
        assert_eq!(shown.rows[0], vec!["Observer", ""]);
        assert_eq!(
            shown.rows[1],
            vec!["    Person Observer Name", "HOUSE GREGORY"]
        );
        let hidden = r.tree_table(true).unwrap();
        assert_eq!(hidden.rows[1][1], "(name)");
        assert!(Report::default().tree_table(false).is_none());
    }
}
