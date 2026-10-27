//! The content tree of a DICOM Structured Report, as it is: every content
//! item with its relationship, value type, concept name and value, nested
//! as the Content Sequence nests them.
//!
//! This is the one reader every SR template goes through. The report
//! interpreters ([`super::dose`]) walk this tree by concept code; the
//! generic view lays it out as an indented two-column table, so an SR of a
//! template nothing here knows still reads. Codes are the triple of the
//! Code Sequence Macro (value, scheme, meaning); a node is addressed by its
//! concept name's code value, which is stable where the meaning is a free
//! translation.

use dicom_dictionary_std::tags;
use dicom_object::InMemDicomObject;

use crate::loader::{items_of, str_of};

/// One content item.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    /// `CONTAINS`, `HAS OBS CONTEXT`, `HAS PROPERTIES`, ...; empty at the
    /// root.
    pub relationship: String,
    /// `CONTAINER`, `NUM`, `TEXT`, `CODE`, `UIDREF`, `DATETIME`, `PNAME`,
    /// ...
    pub value_type: String,
    /// The concept name's code value (`113830`), scheme (`DCM`) and
    /// meaning (`Mean CTDIvol`).
    pub code: String,
    pub scheme: String,
    pub name: String,
    pub value: Value,
    pub children: Vec<Node>,
}

/// What a content item holds.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum Value {
    /// A container, or a type nothing here reads (an image reference).
    #[default]
    None,
    Text(String),
    /// A measurement: the number as the file writes it, parsed, and the
    /// unit's meaning (`mGy`).
    Num {
        text: String,
        value: Option<f64>,
        unit: String,
    },
    /// A coded answer: its meaning and its code value.
    Code {
        meaning: String,
        code: String,
    },
    Uid(String),
    DateTime(String),
    Date(String),
    Time(String),
    /// A person's name. Shown by the viewer, never by the MCP server.
    Name(String),
}

impl Value {
    /// The value as a line of text; a name withheld reads `(name)`.
    pub fn text(&self, hide_names: bool) -> String {
        match self {
            Value::None => String::new(),
            Value::Text(t) | Value::Uid(t) | Value::Time(t) => t.clone(),
            Value::Num { text, unit, .. } => {
                if unit.is_empty() {
                    text.clone()
                } else {
                    format!("{text} {unit}")
                }
            }
            Value::Code { meaning, .. } => meaning.clone(),
            Value::DateTime(t) => show_datetime(t),
            Value::Date(t) => show_date(t),
            Value::Name(n) => {
                if hide_names {
                    "(name)".to_string()
                } else {
                    n.replace('^', " ").trim().to_string()
                }
            }
        }
    }

    /// The number of a `NUM` item.
    pub fn number(&self) -> Option<f64> {
        match self {
            Value::Num { value, .. } => *value,
            _ => None,
        }
    }
}

/// `20250307084137.488` as `2025-03-07 08:41:37`.
pub fn show_datetime(t: &str) -> String {
    let d: String = t.chars().filter(|c| c.is_ascii_digit()).collect();
    if d.len() >= 14 {
        format!(
            "{}-{}-{} {}:{}:{}",
            &d[..4],
            &d[4..6],
            &d[6..8],
            &d[8..10],
            &d[10..12],
            &d[12..14]
        )
    } else if d.len() >= 8 {
        show_date(&d[..8])
    } else {
        t.trim().to_string()
    }
}

/// `20250307` as `2025-03-07`.
pub fn show_date(t: &str) -> String {
    let d = t.trim();
    if d.len() == 8 && d.bytes().all(|b| b.is_ascii_digit()) {
        format!("{}-{}-{}", &d[..4], &d[4..6], &d[6..])
    } else {
        d.to_string()
    }
}

/// The first item of a code sequence as (value, scheme, meaning).
fn code_of(obj: &InMemDicomObject, tag: dicom_core::Tag) -> (String, String, String) {
    let Some(item) = items_of(obj, tag).and_then(|it| it.first()) else {
        return Default::default();
    };
    (
        str_of(item, tags::CODE_VALUE).unwrap_or_default(),
        str_of(item, tags::CODING_SCHEME_DESIGNATOR).unwrap_or_default(),
        str_of(item, tags::CODE_MEANING).unwrap_or_default(),
    )
}

/// One content item and, through its Content Sequence, everything under it.
fn node_of(item: &InMemDicomObject) -> Node {
    let (code, scheme, name) = code_of(item, tags::CONCEPT_NAME_CODE_SEQUENCE);
    let value_type = str_of(item, tags::VALUE_TYPE).unwrap_or_default();
    let value = match value_type.as_str() {
        "TEXT" => str_of(item, tags::TEXT_VALUE)
            .map(Value::Text)
            .unwrap_or_default(),
        "NUM" => items_of(item, tags::MEASURED_VALUE_SEQUENCE)
            .and_then(|it| it.first())
            .map(|m| {
                let text = str_of(m, tags::NUMERIC_VALUE).unwrap_or_default();
                let (_, _, unit) = code_of(m, tags::MEASUREMENT_UNITS_CODE_SEQUENCE);
                Value::Num {
                    value: text.trim().parse().ok(),
                    text,
                    unit,
                }
            })
            .unwrap_or_default(),
        "CODE" => {
            let (code, _, meaning) = code_of(item, tags::CONCEPT_CODE_SEQUENCE);
            Value::Code { meaning, code }
        }
        "UIDREF" => str_of(item, tags::UID).map(Value::Uid).unwrap_or_default(),
        "DATETIME" => str_of(item, tags::DATE_TIME)
            .map(Value::DateTime)
            .unwrap_or_default(),
        "DATE" => str_of(item, tags::DATE)
            .map(Value::Date)
            .unwrap_or_default(),
        "TIME" => str_of(item, tags::TIME)
            .map(Value::Time)
            .unwrap_or_default(),
        "PNAME" => str_of(item, tags::PERSON_NAME)
            .map(Value::Name)
            .unwrap_or_default(),
        _ => Value::None,
    };
    Node {
        relationship: str_of(item, tags::RELATIONSHIP_TYPE).unwrap_or_default(),
        value_type,
        code,
        scheme,
        name,
        value,
        children: items_of(item, tags::CONTENT_SEQUENCE)
            .map(|items| items.iter().map(node_of).collect())
            .unwrap_or_default(),
    }
}

/// The whole document: the root container (the dataset's own concept
/// name) with every item under it.
pub fn read(obj: &InMemDicomObject) -> Node {
    node_of(obj)
}

impl Node {
    /// The first item anywhere below (depth first) whose concept code is
    /// `code`.
    pub fn find(&self, code: &str) -> Option<&Node> {
        self.children.iter().find_map(|c| {
            if c.code == code {
                Some(c)
            } else {
                c.find(code)
            }
        })
    }

    /// The direct children whose concept code is `code`.
    pub fn children_coded<'a>(&'a self, code: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.children.iter().filter(move |c| c.code == code)
    }

    /// The value of the first item below with concept code `code`, as text.
    pub fn text_of(&self, code: &str, hide_names: bool) -> String {
        self.find(code)
            .map(|n| n.value.text(hide_names))
            .unwrap_or_default()
    }

    /// The number of the first `NUM` item below with concept code `code`,
    /// as the file writes it.
    pub fn num_text(&self, code: &str) -> String {
        self.find(code)
            .map(|n| match &n.value {
                Value::Num { text, .. } => text.clone(),
                v => v.text(true),
            })
            .unwrap_or_default()
    }

    /// The unit of the first `NUM` item below with concept code `code`.
    pub fn unit_of(&self, code: &str) -> String {
        self.find(code)
            .and_then(|n| match &n.value {
                Value::Num { unit, .. } => Some(unit.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Every item below, depth first, with its depth (the root's children
    /// at depth 0).
    pub fn walk(&self) -> Vec<(usize, &Node)> {
        fn go<'a>(n: &'a Node, depth: usize, out: &mut Vec<(usize, &'a Node)>) {
            for c in &n.children {
                out.push((depth, c));
                go(c, depth + 1, out);
            }
        }
        let mut out = Vec::new();
        go(self, 0, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_datetimes_read_as_people_write_them() {
        assert_eq!(show_date("20250307"), "2025-03-07");
        assert_eq!(
            show_datetime("20250307084137.488000"),
            "2025-03-07 08:41:37"
        );
        assert_eq!(show_datetime("20250307"), "2025-03-07");
        assert_eq!(show_date("yesterday"), "yesterday");
    }

    #[test]
    fn a_name_is_withheld_on_request_and_a_number_carries_its_unit() {
        let n = Value::Name("DOE^JANE".into());
        assert_eq!(n.text(false), "DOE JANE");
        assert_eq!(n.text(true), "(name)");
        let m = Value::Num {
            text: "2.8".into(),
            value: Some(2.8),
            unit: "mGy".into(),
        };
        assert_eq!(m.text(true), "2.8 mGy");
        assert_eq!(m.number(), Some(2.8));
    }
}
