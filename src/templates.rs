//! Structure templates: the names, types, colours and recipes a department
//! uses again on every patient.
//!
//! A template is not geometry. It is the *list*: what a head-and-neck case
//! is called here, which of them are targets, what colour the parotids are
//! drawn in, and which structures are derived from which others and how.
//! Applying one creates the empty structures in the right order with the
//! right properties, and the recipes come with them, so the derived ones
//! fill themselves in as soon as their operands exist.
//!
//! What a template deliberately does not do is run anything. The learned
//! engines and the generators are one click away and have their own
//! windows; a template that started three of them behind a planner's back
//! would be a worse tool, not a better one.
//!
//! One JSON file per template under `<data dir>/templates`, so a department
//! can put them in version control, mail them to each other, and read them
//! without this program.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::derived::{self, Derived};
use crate::rtstruct::{Roi, StructureSet};
use crate::settings;

/// One entry of a template.
#[derive(Clone, Debug, PartialEq)]
pub struct TemplateRoi {
    pub name: String,
    /// RT ROI Interpreted Type: `PTV`, `ORGAN`, `EXTERNAL`, …
    pub roi_type: String,
    pub color: [u8; 3],
    /// The derived recipe, exactly as it is stored in ROI Description
    /// (including its prefix), or empty for a structure that is drawn.
    pub recipe: String,
}

/// A named list of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    pub name: String,
    pub rois: Vec<TemplateRoi>,
}

/// Where templates live.
pub fn dir() -> PathBuf {
    settings::data_dir().join("templates")
}

/// A file name that cannot escape the directory or upset a file system.
fn file_name(name: &str) -> String {
    let safe: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{}.json", safe.trim())
}

pub fn path_of(name: &str) -> PathBuf {
    dir().join(file_name(name))
}

/// Every template on disk, by name, in alphabetical order.
pub fn list() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir()) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            if let Some(t) = std::fs::read_to_string(&p).ok().and_then(|s| decode(&s)) {
                out.push(t.name);
            }
        }
    }
    out.sort_by_key(|n| n.to_lowercase());
    out.dedup();
    out
}

pub fn load(name: &str) -> Result<Template> {
    let p = path_of(name);
    let text = std::fs::read_to_string(&p).with_context(|| format!("read {}", p.display()))?;
    decode(&text).with_context(|| format!("{} is not a structure template", p.display()))
}

pub fn save(t: &Template) -> Result<()> {
    if t.name.trim().is_empty() {
        bail!("a template needs a name");
    }
    let d = dir();
    std::fs::create_dir_all(&d).with_context(|| format!("create {}", d.display()))?;
    let p = path_of(&t.name);
    std::fs::write(&p, encode(t)).with_context(|| format!("write {}", p.display()))
}

pub fn delete(name: &str) -> Result<()> {
    let p = path_of(name);
    if !Path::new(&p).exists() {
        return Ok(());
    }
    std::fs::remove_file(&p).with_context(|| format!("remove {}", p.display()))
}

pub fn encode(t: &Template) -> String {
    let rois: Vec<serde_json::Value> = t
        .rois
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "type": r.roi_type,
                "color": r.color.to_vec(),
                "recipe": r.recipe,
            })
        })
        .collect();
    let v = serde_json::json!({
        "kind": "rds-structure-template",
        "v": 1,
        "name": t.name,
        "rois": rois,
    });
    serde_json::to_string_pretty(&v).unwrap_or_default()
}

pub fn decode(text: &str) -> Option<Template> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    if v.get("kind")?.as_str()? != "rds-structure-template" {
        return None;
    }
    let name = v.get("name")?.as_str()?.to_string();
    let mut rois = Vec::new();
    for r in v.get("rois")?.as_array()? {
        let color = r
            .get("color")
            .and_then(|c| c.as_array())
            .map(|a| {
                let mut out = [200u8, 200, 200];
                for (i, x) in a.iter().take(3).enumerate() {
                    out[i] = x.as_u64().unwrap_or(200).min(255) as u8;
                }
                out
            })
            .unwrap_or([200, 200, 200]);
        rois.push(TemplateRoi {
            name: r.get("name")?.as_str()?.to_string(),
            roi_type: r
                .get("type")
                .and_then(|x| x.as_str())
                .unwrap_or("ORGAN")
                .to_string(),
            color,
            recipe: r
                .get("recipe")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string(),
        });
    }
    Some(Template { name, rois })
}

/// Read a template off a structure set that is already the way somebody
/// wants it: every structure's name, type, colour and recipe, and nothing
/// about where any of them happen to be on this patient.
pub fn of_set(name: &str, ss: &StructureSet) -> Template {
    Template {
        name: name.to_string(),
        rois: ss
            .rois
            .iter()
            .map(|r| TemplateRoi {
                name: r.name.clone(),
                roi_type: r.roi_type.clone(),
                color: r.color,
                recipe: match derived::of_roi(r) {
                    // The stored recipe travels; the geometry fingerprint of
                    // *this* patient does not, so the applied structure
                    // starts out needing an update, which is the truth.
                    Some(d) => Derived {
                        hash: 0,
                        overridden: false,
                        ..d
                    }
                    .encode(),
                    None => String::new(),
                },
            })
            .collect(),
    }
}

/// The structures a template asks for that are not in the set yet, as
/// ready-made ROIs. Numbering continues from what is there.
///
/// Names already present are left alone rather than duplicated: applying a
/// template twice, or to a set an engine has already filled in, has to be
/// harmless.
pub fn missing_rois(t: &Template, ss: &StructureSet) -> Vec<Roi> {
    let have: Vec<String> = ss.rois.iter().map(|r| r.name.to_lowercase()).collect();
    let mut number = ss.rois.iter().map(|r| r.number).max().unwrap_or(0);
    let mut out = Vec::new();
    for r in &t.rois {
        if have.contains(&r.name.to_lowercase()) {
            continue;
        }
        number += 1;
        out.push(Roi {
            number,
            name: r.name.clone(),
            color: r.color,
            roi_type: r.roi_type.clone(),
            description: r.recipe.clone(),
            contours: Vec::new(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtstruct::Contour;
    use crate::structops::{BoolOp, Margin};

    fn roi(name: &str, description: &str) -> Roi {
        Roi {
            number: 1,
            name: name.into(),
            color: [10, 20, 30],
            roi_type: "ORGAN".into(),
            description: description.into(),
            contours: vec![Contour {
                points: vec![crate::geometry::Vec3::new(0.0, 0.0, 0.0)],
                geometric_type: "POINT".into(),
            }],
        }
    }

    fn set(rois: Vec<Roi>) -> StructureSet {
        StructureSet {
            label: "set".into(),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: String::new(),
            series_instance_uid: String::new(),
            study_uid: String::new(),
            referenced_series_uid: String::new(),
            file_name: String::new(),
            locked: false,
            rois,
        }
    }

    #[test]
    fn a_template_round_trips_through_its_file_format() {
        let t = Template {
            name: "Head and neck".into(),
            rois: vec![
                TemplateRoi {
                    name: "Parotid_L".into(),
                    roi_type: "ORGAN".into(),
                    color: [255, 0, 0],
                    recipe: String::new(),
                },
                TemplateRoi {
                    name: "PTV_high".into(),
                    roi_type: "PTV".into(),
                    color: [0, 0, 255],
                    recipe: "RDS-DERIVED:{}".into(),
                },
            ],
        };
        let back = decode(&encode(&t)).expect("decodes");
        assert_eq!(back, t);
        // Anything else on disk is not a template, and says so instead of
        // arriving as an empty one.
        assert!(decode("{\"kind\":\"something else\"}").is_none());
        assert!(decode("not json at all").is_none());
    }

    #[test]
    fn a_template_taken_off_a_set_keeps_the_recipe_and_drops_the_patient() {
        let d = Derived {
            expr: derived::Expr {
                op: BoolOp::Subtract,
                deps: vec![
                    derived::Dep {
                        name: "Body".into(),
                        margin: Margin::default(),
                    },
                    derived::Dep {
                        name: "PTV".into(),
                        margin: Margin::uniform(5.0),
                    },
                ],
                margin: Margin::default(),
                cleanup: Default::default(),
            },
            hash: 0x1234_5678_9abc_def0,
            overridden: true,
        };
        let ss = set(vec![roi("Ring", &d.encode()), roi("PTV", "")]);
        let t = of_set("Prostate", &ss);
        assert_eq!(t.rois.len(), 2);
        let back = Derived::decode(&t.rois[0].recipe).expect("the recipe travels");
        assert_eq!(back.expr, d.expr);
        // ... without this patient's geometry, so it arrives asking to be
        // computed rather than claiming to be up to date.
        assert_eq!(back.hash, 0);
        assert!(!back.overridden);
        assert!(t.rois[1].recipe.is_empty());
    }

    #[test]
    fn applying_a_template_adds_only_what_is_missing() {
        let t = Template {
            name: "T".into(),
            rois: vec![
                TemplateRoi {
                    name: "Body".into(),
                    roi_type: "EXTERNAL".into(),
                    color: [1, 2, 3],
                    recipe: String::new(),
                },
                TemplateRoi {
                    name: "PTV".into(),
                    roi_type: "PTV".into(),
                    color: [4, 5, 6],
                    recipe: "RDS-DERIVED:{}".into(),
                },
            ],
        };
        let ss = set(vec![roi("body", "")]);
        let missing = missing_rois(&t, &ss);
        assert_eq!(missing.len(), 1, "the existing Body is left alone");
        assert_eq!(missing[0].name, "PTV");
        assert_eq!(missing[0].roi_type, "PTV");
        assert_eq!(missing[0].description, "RDS-DERIVED:{}");
        assert_eq!(missing[0].number, 2);
        // Nothing to add to a set that already has everything.
        let full = set(vec![roi("Body", ""), roi("PTV", "")]);
        assert!(missing_rois(&t, &full).is_empty());
    }
}
