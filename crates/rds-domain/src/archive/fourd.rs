//! The 4D sets of the PACS, kept in its catalog: which studies hold a
//! 4DCT, how many scans, which series are the phases and which the
//! reconstructions - known before a study is loaded.
//!
//! [`index`] runs the header rules of [`crate::fourd::scan`] over what the
//! catalog keeps of every series (the head of a CT's first file, the slice
//! position of each of its files) after every write of the catalog, so the
//! sets always say what the series say; a scan that finds nothing new
//! writes nothing, since the same series give the same sets. What a person
//! decided wins over the rules: a group made, corrected or confirmed by
//! hand, or one dissolved (a tombstone), lives in
//! [`Catalog::fourd_custom`] under its group id and replaces any set found
//! that shares a series with it. The viewer, the PACS window and the MCP
//! tools all write there, under the catalog's lock, so that they are one
//! source.
//!
//! Beside the sets, [`index`] links a structure set drawn on one phase to
//! that phase (`structures`, with the phase's label), and lists the series
//! of a study that say "4D" with no set to go with - a scout taken for a
//! 4DCT whose phases never reached the PACS, an ungated reconstruction of a
//! scan exported without its phases.
//!
//! Before version 3 the sets were written into the catalog's labels by a
//! script (`4DCT 30%`, `· 4DCT×2`). [`migrate_labels`] compares each such
//! label with the set found once, clears the ones that agree (the
//! structure says it now), keeps and notes the others, and takes a label
//! that says `4DCT` where the rules only say probable as the person's
//! confirmation of that set.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use super::catalog::{self, Catalog};
use super::Archive;
use crate::fourd::scan::{self, Evidence, Found, FoundMember, Input, Kind, Z};

/// The modalities whose first file's head the catalog keeps.
pub const HEAD_MODALITIES: [&str; 2] = ["CT", "RTSTRUCT"];

/// What the description rule may group, as the viewer would.
const IMAGE_MODALITIES: [&str; 4] = ["CT", "MR", "PT", "NM"];

/// The journal of the label migration, beside the catalog.
pub const LABEL_JOURNAL: &str = ".rds-pacs-fourd-labels.jsonl";

fn no(b: &bool) -> bool {
    !*b
}

/// A series in a 4D set: the set's id and what the series is to it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SeriesFourD {
    pub group: String,
    /// `phase`, `untagged`, `AVG`, `MIP`, `MinIP`, `trace`; `structures`
    /// for a structure set drawn on one of the phases.
    pub role: String,
    /// `30%`, `t3`; for `structures`, the phase it was drawn on.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub phase: String,
    /// The phase was inferred from the series' place in the set.
    #[serde(skip_serializing_if = "no")]
    pub inferred: bool,
    /// `std` or `EFOV` (the Extended-FOV reconstruction).
    #[serde(skip_serializing_if = "String::is_empty")]
    pub variant: String,
}

impl SeriesFourD {
    /// `30%`, `~30%`, `AVG`: what the PACS window shows beside the series.
    pub fn show(&self) -> String {
        match self.role.as_str() {
            "phase" if self.inferred => format!("~{}", self.phase),
            "phase" => self.phase.clone(),
            "structures" => format!("structures on {}", self.phase),
            r => r.to_string(),
        }
    }
}

/// One 4D set of a study.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct StudyFourD {
    pub group: String,
    /// `4DCT 2024-06-10 (10 phases + 1)`.
    pub name: String,
    /// `YYYYMMDD`; empty when no member says.
    pub date: String,
    pub scanner: String,
    pub phases: usize,
    /// Slices of a phase.
    pub slices: usize,
    /// `untagged`, `AVG`, `MIP`, `MinIP`, as present.
    pub recons: Vec<String>,
    /// `std` or `EFOV`.
    pub variant: String,
    /// The scan the set is of, from 1, within the study.
    pub scan: usize,
    /// `phase-labels`, `header-4d`, `same-acquisition`, `geometry-only`,
    /// `description`, `custom`.
    pub evidence: String,
    /// False only for `geometry-only`, until a person confirms it.
    pub confirmed: bool,
    /// Made, corrected or confirmed by hand.
    pub custom: bool,
    /// Its series, the trace included.
    pub series: usize,
}

/// A series that says "4D" with no set to go with.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Orphan {
    pub series: String,
    pub number: String,
    /// `scout`, `untagged`, or `4D context`.
    pub what: String,
}

/// A group as a person made or decided it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomGroup {
    pub study: String,
    pub name: String,
    /// `[Series Instance UID, role, phase]`, phases in order.
    pub members: Vec<(String, String, String)>,
    /// Dissolved: the set the rules find with any of these members stays
    /// away.
    #[serde(skip_serializing_if = "no")]
    pub tombstone: bool,
    /// The series-number block, date and reconstruction of the set it was
    /// made from, so that scans are counted as before.
    pub block: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub date: String,
    #[serde(skip_serializing_if = "no")]
    pub efov: bool,
    /// When and why, for the journal of the catalog.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub note: String,
}

/// The files' slice positions of a series.
fn z_of(s: &catalog::SeriesRec) -> Option<Z> {
    Z::of(s.files.iter().filter_map(|f| f.5))
}

/// The sets of the catalog as the rules find them, before what a person
/// decided.
pub fn detect(c: &Catalog) -> Vec<Found> {
    let zs: Vec<Option<Z>> = c.series.values().map(z_of).collect();
    let inputs: Vec<Input> = c
        .series
        .iter()
        .zip(&zs)
        .map(|((uid, s), z)| Input {
            uid,
            study: &s.study,
            modality: &s.modality,
            number: s.number.trim().parse().ok(),
            description: &s.description,
            head: s.head.as_ref().filter(|_| s.modality == "CT"),
            temporal: None,
            z: *z,
            image: IMAGE_MODALITIES.contains(&s.modality.as_str()),
        })
        .collect();
    scan::detect(&inputs)
}

/// The sets with what a person decided put in: custom groups replace the
/// sets they share a series with, tombstones take theirs away.
pub fn sets(c: &Catalog) -> Vec<Found> {
    let mut found = detect(c);
    let custom = &c.fourd_custom;
    let mine: BTreeSet<&str> = custom
        .values()
        .flat_map(|g| g.members.iter().map(|m| m.0.as_str()))
        .collect();
    found.retain(|f| {
        !custom.contains_key(&f.id) && !f.members.iter().any(|m| mine.contains(m.series.as_str()))
    });
    for (id, g) in custom.iter().filter(|(_, g)| !g.tombstone) {
        let members: Vec<FoundMember> = g
            .members
            .iter()
            .filter(|m| c.series.contains_key(&m.0))
            .map(|(uid, role, phase)| FoundMember {
                series: uid.clone(),
                number: c.series[uid].number.trim().parse().unwrap_or(-1),
                kind: Kind::of_word(role).unwrap_or_default(),
                label: if phase.is_empty() {
                    role.clone()
                } else {
                    phase.clone()
                },
                percent: phase.trim_end_matches('%').parse().ok(),
                inferred: false,
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        let phases = members.iter().filter(|m| m.kind == Kind::Phase).count();
        let slices = members
            .iter()
            .find(|m| m.kind == Kind::Phase)
            .and_then(|m| z_of(&c.series[&m.series]))
            .map_or(0, |z| z.n);
        found.push(Found {
            id: id.clone(),
            study: g.study.clone(),
            name: g.name.clone(),
            block: g.block,
            efov: g.efov,
            evidence: Evidence::Custom,
            date: g.date.clone(),
            scanner: members
                .iter()
                .find_map(|m| c.series[&m.series].head.as_ref().map(|h| h.model.clone()))
                .unwrap_or_default(),
            phases,
            slices,
            scan: 0,
            members,
        });
    }
    scan::number_scans(&mut found);
    found
}

/// Find the sets again and write them into the series and studies, with
/// the structure sets drawn on a phase and the series left over.
pub fn index(c: &mut Catalog) {
    let found = sets(c);
    for s in c.series.values_mut() {
        s.fourd = None;
    }
    for st in c.studies.values_mut() {
        st.fourd.clear();
        st.fourd_orphans.clear();
    }
    let mut member_of: BTreeMap<String, (String, String)> = BTreeMap::new();
    for f in &found {
        let variant = if f.efov { "EFOV" } else { "std" };
        for m in &f.members {
            if let Some(s) = c.series.get_mut(&m.series) {
                s.fourd = Some(SeriesFourD {
                    group: f.id.clone(),
                    role: m.kind.word().to_string(),
                    phase: if m.kind == Kind::Phase {
                        m.label.clone()
                    } else {
                        String::new()
                    },
                    inferred: m.inferred,
                    variant: variant.to_string(),
                });
            }
            if m.kind == Kind::Phase {
                member_of.insert(m.series.clone(), (f.id.clone(), m.label.clone()));
            }
        }
        let mut recons: Vec<String> = Vec::new();
        for m in f.members.iter().filter(|m| m.kind.is_recon()) {
            let w = m.kind.word().to_string();
            if !recons.contains(&w) {
                recons.push(w);
            }
        }
        if let Some(st) = c.studies.get_mut(&f.study) {
            st.fourd.push(StudyFourD {
                group: f.id.clone(),
                name: f.name.clone(),
                date: f.date.clone(),
                scanner: f.scanner.clone(),
                phases: f.phases,
                slices: f.slices,
                recons,
                variant: variant.to_string(),
                scan: f.scan,
                evidence: f.evidence.word().to_string(),
                confirmed: f.evidence.confirmed(),
                custom: f.evidence == Evidence::Custom,
                series: f.members.len(),
            });
        }
    }
    // A structure set drawn on a phase: under that phase.
    for s in c.series.values_mut().filter(|s| s.modality == "RTSTRUCT") {
        let on = s
            .head
            .as_ref()
            .and_then(|h| h.refs.iter().find_map(|r| member_of.get(r)));
        if let Some((group, phase)) = on {
            s.fourd = Some(SeriesFourD {
                group: group.clone(),
                role: "structures".into(),
                phase: phase.clone(),
                ..Default::default()
            });
        }
    }
    // What says 4D with no set in its block.
    let blocks: BTreeSet<(String, i64)> = found
        .iter()
        .flat_map(|f| {
            f.members
                .iter()
                .map(move |m| (f.study.clone(), m.number.max(0) / 100))
        })
        .collect();
    let mut orphans: Vec<(String, Orphan)> = Vec::new();
    for (uid, s) in c
        .series
        .iter()
        .filter(|(_, s)| s.modality == "CT" && s.fourd.is_none())
    {
        let Some(h) = &s.head else { continue };
        let recon = scan::recon_of(&s.description, h);
        let what = if recon == Some(Kind::Untagged) {
            "untagged"
        } else if !scan::has_context(h) {
            continue;
        } else if h.image_type.contains("LOCALIZER") {
            "scout"
        } else {
            "4D context"
        };
        let number: i64 = s.number.trim().parse().unwrap_or(-1);
        if blocks.contains(&(s.study.clone(), number.max(0) / 100)) {
            continue;
        }
        orphans.push((
            s.study.clone(),
            Orphan {
                series: uid.clone(),
                number: s.number.trim().to_string(),
                what: what.into(),
            },
        ));
    }
    for (study, o) in orphans {
        if let Some(st) = c.studies.get_mut(&study) {
            st.fourd_orphans.push(o);
        }
    }
}

/// The label the 4D script wrote for a series of a set: `4DCT 30%`,
/// `4DCT ~30% EFOV`, `4DCT? untagged`. `None` for what it never tagged.
pub fn script_label(f: &SeriesFourD, confirmed: bool) -> Option<String> {
    let what = match f.role.as_str() {
        "phase" if f.inferred => format!("~{}", f.phase),
        "phase" => f.phase.clone(),
        "untagged" | "AVG" | "MIP" | "MinIP" => f.role.clone(),
        _ => return None,
    };
    Some(format!(
        "{} {what}{}",
        if confirmed { "4DCT" } else { "4DCT?" },
        if f.variant == "EFOV" { " EFOV" } else { "" }
    ))
}

/// Is this a label the 4D script wrote (`4DCT ...`, `4DCT? ...`)?
fn script_wrote(label: &str) -> bool {
    let rest = label
        .strip_prefix("4DCT?")
        .or_else(|| label.strip_prefix("4DCT"));
    rest.is_some_and(|r| r.is_empty() || r.starts_with(' '))
}

/// A study label with the script's ` · 4DCT...` suffix taken off; `None`
/// when it has none.
pub fn strip_study_suffix(label: &str) -> Option<String> {
    let at = label.rfind(" \u{b7} 4DCT")?;
    let tail = &label[at + " \u{b7} ".len()..];
    (!tail.contains('\u{b7}')).then(|| label[..at].to_string())
}

/// One line of the migration's journal.
#[derive(Clone, Debug, Serialize)]
pub struct LabelNote {
    /// `series` or `study`.
    pub kind: &'static str,
    pub uid: String,
    pub old: String,
    /// The label as it is now (empty: cleared).
    pub new: String,
    /// `cleared`, `cleared, confirmed`, `kept: ...`.
    pub what: String,
}

/// Compare the labels the 4D script wrote with the sets found, once, when
/// every CT of the catalog has its head (see the module doc). The sets must
/// have been found ([`index`]) first; a confirmation found here becomes a
/// custom group, which the next [`index`] puts in.
pub fn migrate_labels(c: &mut Catalog) -> Vec<LabelNote> {
    if c.fourd_migrated {
        return Vec::new();
    }
    if c.series
        .values()
        .any(|s| s.modality == "CT" && s.head.is_none())
    {
        return Vec::new();
    }
    let confirmed: BTreeMap<String, bool> = c
        .studies
        .values()
        .flat_map(|st| st.fourd.iter().map(|f| (f.group.clone(), f.confirmed)))
        .collect();
    let mut notes: Vec<LabelNote> = Vec::new();
    let mut confirm: BTreeSet<String> = BTreeSet::new();
    for (uid, s) in c.series.iter_mut().filter(|(_, s)| script_wrote(&s.label)) {
        let old = s.label.clone();
        let what = match &s.fourd {
            Some(f) => {
                let ok = confirmed.get(&f.group).copied().unwrap_or(true);
                if script_label(f, ok).as_deref() == Some(old.as_str()) {
                    "cleared".to_string()
                } else if !ok && script_label(f, true).as_deref() == Some(old.as_str()) {
                    confirm.insert(f.group.clone());
                    "cleared, confirmed".to_string()
                } else {
                    format!(
                        "kept: the set found says {}",
                        script_label(f, ok).unwrap_or_else(|| f.show())
                    )
                }
            }
            None => "kept: in no set found".to_string(),
        };
        if what.starts_with("cleared") {
            s.label.clear();
        }
        notes.push(LabelNote {
            kind: "series",
            uid: uid.clone(),
            old,
            new: s.label.clone(),
            what,
        });
    }
    for (uid, st) in c.studies.iter_mut() {
        let Some(stripped) = strip_study_suffix(&st.label) else {
            continue;
        };
        let old = st.label.clone();
        let what = if st.fourd.is_empty() {
            "kept: no set found in the study".to_string()
        } else {
            st.label = stripped;
            "cleared".to_string()
        };
        notes.push(LabelNote {
            kind: "study",
            uid: uid.clone(),
            old,
            new: st.label.clone(),
            what,
        });
    }
    for id in confirm {
        let _ = confirm_in(c, &id, true, "confirmed by the label the 4D script wrote");
    }
    c.fourd_migrated = true;
    notes
}

/// Append the notes to the journal beside the catalog.
pub fn journal(main: &Path, notes: &[LabelNote]) {
    if notes.is_empty() {
        return;
    }
    let (date, time) = crate::dicom_export::today();
    let mut text = String::new();
    for n in notes {
        if let Ok(mut v) = serde_json::to_value(n) {
            v["time"] = serde_json::json!(format!("{date}T{time}"));
            text.push_str(&v.to_string());
            text.push('\n');
        }
    }
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(main.join(LABEL_JOURNAL))
    {
        let _ = f.write_all(text.as_bytes());
    }
}

/// Confirm (`yes`) or reject a set in the catalog: a confirmed set becomes
/// a custom group of the same series, a rejected one a tombstone.
fn confirm_in(c: &mut Catalog, group: &str, yes: bool, note: &str) -> Result<String> {
    let found = sets(c);
    let Some(f) = found.iter().find(|f| f.id == group) else {
        bail!("no 4D group {group} in the PACS");
    };
    let members: Vec<(String, String, String)> = f
        .members
        .iter()
        .map(|m| {
            (
                m.series.clone(),
                m.kind.word().to_string(),
                if m.kind == Kind::Phase {
                    m.label.clone()
                } else {
                    String::new()
                },
            )
        })
        .collect();
    let extras = f.members.iter().filter(|m| m.kind.is_recon()).count();
    c.fourd_custom.insert(
        f.id.clone(),
        CustomGroup {
            study: f.study.clone(),
            name: if f.evidence == Evidence::Custom {
                f.name.clone()
            } else {
                scan::name_of(true, &f.date, f.efov, f.phases, extras)
            },
            members,
            tombstone: !yes,
            block: f.block,
            date: f.date.clone(),
            efov: f.efov,
            note: note.into(),
        },
    );
    Ok(f.id.clone())
}

/// One member of a group as a listing gives it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GroupMember {
    pub series: String,
    pub number: String,
    pub role: String,
    pub phase: String,
    pub inferred: bool,
    pub files: usize,
}

/// One group of a study with its members.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct GroupListing {
    #[serde(flatten)]
    pub set: StudyFourD,
    pub members: Vec<GroupMember>,
    /// Structure sets drawn on its phases.
    pub structures: Vec<GroupMember>,
}

/// The groups of a study as the catalog has them, with their members.
pub fn groups_of(c: &Catalog, study: &str) -> Vec<GroupListing> {
    let Some(st) = c.studies.get(study) else {
        return Vec::new();
    };
    st.fourd
        .iter()
        .map(|g| {
            let mut members: Vec<GroupMember> = Vec::new();
            let mut structures: Vec<GroupMember> = Vec::new();
            for (uid, s) in c.series.iter().filter(|(_, s)| s.study == study) {
                let Some(f) = s.fourd.as_ref().filter(|f| f.group == g.group) else {
                    continue;
                };
                let m = GroupMember {
                    series: uid.clone(),
                    number: s.number.trim().to_string(),
                    role: f.role.clone(),
                    phase: f.phase.clone(),
                    inferred: f.inferred,
                    files: s.files.len(),
                };
                if f.role == "structures" {
                    structures.push(m);
                } else {
                    members.push(m);
                }
            }
            // Phases in their order, then the reconstructions, the trace.
            let order: Vec<String> = sets_order(c, &g.group);
            members.sort_by_key(|m| {
                order
                    .iter()
                    .position(|u| *u == m.series)
                    .unwrap_or(usize::MAX)
            });
            GroupListing {
                set: g.clone(),
                members,
                structures,
            }
        })
        .collect()
}

/// The members of a group in the order the rules (or the person) put them.
fn sets_order(c: &Catalog, group: &str) -> Vec<String> {
    sets(c)
        .into_iter()
        .find(|f| f.id == group)
        .map(|f| f.members.into_iter().map(|m| m.series).collect())
        .unwrap_or_default()
}

impl Archive {
    /// The 4D groups of a study, from the catalog alone.
    pub fn fourd_groups(&self, study: &str) -> Vec<GroupListing> {
        groups_of(&catalog::load(self.root()), study)
    }

    /// Confirm (`yes`) or reject a group: kept in the catalog as the
    /// person's decision. Answers the group's id.
    pub fn confirm_4d(&self, group: &str, yes: bool) -> Result<String> {
        let group = group.to_string();
        catalog::update(self.root(), move |c| {
            confirm_in(c, &group, yes, if yes { "confirmed" } else { "rejected" })
        })
    }

    /// Make a group by hand of these series of one study (`[series, role,
    /// phase]`, phases in order; an empty role is a phase, an empty phase
    /// `t<n>`), named `name`: it replaces every set found that shares a
    /// series with it. Answers the new group's id.
    pub fn set_4d_group(
        &self,
        study: &str,
        members: &[(String, String, String)],
        name: &str,
    ) -> Result<String> {
        if members.is_empty() {
            bail!("a 4D group needs at least one series");
        }
        let study = study.to_string();
        let name = name.trim().to_string();
        let members = members.to_vec();
        catalog::update(self.root(), move |c| {
            set_group_in(c, &study, &members, &name)
        })
    }

    /// Forget what a person decided about a group: the rules decide again.
    pub fn reset_4d_group(&self, group: &str) -> Result<()> {
        let group = group.to_string();
        catalog::update(self.root(), move |c| {
            c.fourd_custom.remove(&group);
            Ok(())
        })
    }

    /// The files to load for a group: its phases and reconstructions (and
    /// trace), and the structure sets drawn on them.
    pub fn fourd_group_series(&self, group: &str) -> Vec<String> {
        let c = catalog::load(self.root());
        c.series
            .iter()
            .filter(|(_, s)| s.fourd.as_ref().is_some_and(|f| f.group == group))
            .map(|(uid, _)| uid.clone())
            .collect()
    }
}

/// Put a group made by hand into the catalog (see [`Archive::set_4d_group`]).
fn set_group_in(
    c: &mut Catalog,
    study: &str,
    members: &[(String, String, String)],
    name: &str,
) -> Result<String> {
    if members.is_empty() {
        bail!("a 4D group needs at least one series");
    }
    let mut out: Vec<(String, String, String)> = Vec::new();
    let mut n = 0;
    for (uid, role, phase) in members {
        let Some(s) = c.series.get(uid) else {
            bail!("the PACS holds no series {uid}");
        };
        if s.study != study {
            bail!("series {uid} is not of study {study}");
        }
        let kind = if role.trim().is_empty() {
            Kind::Phase
        } else {
            Kind::of_word(role).ok_or_else(|| {
                anyhow::anyhow!(
                    "role must be phase, untagged, AVG, MIP, MinIP or trace, not {role}"
                )
            })?
        };
        let phase = if kind == Kind::Phase {
            n += 1;
            if phase.trim().is_empty() {
                format!("t{n}")
            } else {
                phase.trim().to_string()
            }
        } else {
            String::new()
        };
        if out.iter().any(|m| m.0 == *uid) {
            continue;
        }
        out.push((uid.clone(), kind.word().to_string(), phase));
    }
    // The id of the set it replaces when it is one; a new one
    // otherwise.
    let found = sets(c);
    let first = &out[0].0;
    let id = found
        .iter()
        .find(|f| f.members.iter().any(|m| m.series == *first))
        .map(|f| f.id.clone())
        .unwrap_or_else(|| {
            let mut k = 1;
            while c.fourd_custom.contains_key(&format!("{study}/custom{k}")) {
                k += 1;
            }
            format!("{study}/custom{k}")
        });
    let phases = out.iter().filter(|m| m.1 == "phase").count();
    let date = out
        .iter()
        .filter_map(|m| c.series.get(&m.0))
        .filter_map(|s| s.datetime.get(..8))
        .min()
        .unwrap_or_default()
        .to_string();
    let block = c.series[first]
        .number
        .trim()
        .parse::<i64>()
        .map_or(-1, |n| if n >= 100 { n / 100 } else { -1 });
    // Every other group that held one of these series loses it.
    let mine: BTreeSet<String> = out.iter().map(|m| m.0.clone()).collect();
    for g in c.fourd_custom.values_mut() {
        g.members.retain(|m| !mine.contains(&m.0));
    }
    c.fourd_custom.insert(
        id.clone(),
        CustomGroup {
            study: study.to_string(),
            name: if name.is_empty() {
                scan::name_of(true, &date, false, phases, out.len() - phases)
            } else {
                name.to_string()
            },
            members: out,
            tombstone: false,
            block,
            date,
            efov: false,
            note: "made by hand".into(),
        },
    );
    Ok(id)
}

/// The viewer's 4D groups of a study loaded from this PACS with what was
/// decided about its sets in the catalog put in: a group made or confirmed
/// by hand, a set dissolved, with the series of it that are loaded.
pub fn apply_to_study(ar: &Archive, study: &mut crate::loader::LoadedStudy) {
    let c = catalog::load(ar.root());
    let loaded: BTreeSet<&str> = study.series.iter().map(|s| s.uid.as_str()).collect();
    let mut kept: Vec<crate::fourd::FourDGroup> = Vec::new();
    for (id, g) in &c.fourd_custom {
        // The trace and the structure sets are no image series of the
        // viewer: a decision counts where its series are loaded.
        if !g.members.iter().any(|m| loaded.contains(m.0.as_str())) {
            continue;
        }
        kept.push(crate::fourd::FourDGroup {
            id: id.clone(),
            name: g.name.clone(),
            study_uid: g.study.clone(),
            members: g
                .members
                .iter()
                .filter(|m| loaded.contains(m.0.as_str()))
                .filter_map(|(uid, role, phase)| {
                    let kind = Kind::of_word(role).unwrap_or_default();
                    Some(crate::fourd::Member {
                        series_uid: uid.clone(),
                        label: if phase.is_empty() {
                            kind.word().to_string()
                        } else {
                            phase.clone()
                        },
                        role: crate::fourd::Role::of_kind(kind)?,
                        percent: phase.trim_end_matches('%').parse().ok(),
                    })
                })
                .collect(),
            custom: !g.tombstone,
            dissolved: g.tombstone,
        });
    }
    if kept.is_empty() {
        return;
    }
    // What the viewer decided itself stays; the catalog's decisions come in
    // over what was detected.
    let mut existing: Vec<crate::fourd::FourDGroup> = study
        .fourd_groups
        .iter()
        .filter(|g| g.custom || g.dissolved)
        .cloned()
        .collect();
    existing.extend(kept);
    study.fourd_groups = crate::fourd::refresh(&existing, &study.series);
}

/// What the viewer decided about one study's groups, waiting to be kept.
struct Pending {
    root: std::path::PathBuf,
    groups: Vec<crate::fourd::FourDGroup>,
    /// Groups that had an id and are gone: the rules decide again.
    removed: BTreeSet<String>,
}

/// The decisions waiting, by study, and whether a writer is at them.
static PENDING: std::sync::Mutex<(bool, BTreeMap<String, Pending>)> =
    std::sync::Mutex::new((false, BTreeMap::new()));

/// Keep the viewer's 4D groups of `study` in this PACS's catalog, in the
/// background: the groups made or edited by hand, the sets dissolved,
/// `removed` the ids of groups that are gone. Nothing happens for a study
/// the catalog does not hold. Edits made quickly one after another are
/// written once, the last state winning.
pub fn keep_viewer_groups(
    ar: &Archive,
    study: &str,
    groups: Vec<crate::fourd::FourDGroup>,
    removed: Vec<String>,
) {
    let Ok(mut q) = PENDING.lock() else {
        return;
    };
    let e = q.1.entry(study.to_string()).or_insert_with(|| Pending {
        root: ar.root().to_path_buf(),
        groups: Vec::new(),
        removed: BTreeSet::new(),
    });
    e.groups = groups;
    e.removed.extend(removed);
    if q.0 {
        return;
    }
    q.0 = true;
    drop(q);
    std::thread::spawn(|| loop {
        let next = {
            let Ok(mut q) = PENDING.lock() else {
                return;
            };
            match q.1.pop_first() {
                Some(n) => n,
                None => {
                    q.0 = false;
                    return;
                }
            }
        };
        let (study, p) = next;
        let _ = keep_in(&p.root, &study, &p.groups, &p.removed);
    });
}

/// One study's viewer decisions into the catalog, in one write.
fn keep_in(
    root: &Path,
    study: &str,
    groups: &[crate::fourd::FourDGroup],
    removed: &BTreeSet<String>,
) -> Result<()> {
    let c = catalog::load(root);
    if !c.studies.contains_key(study) {
        return Ok(());
    }
    let groups = groups.to_vec();
    let removed = removed.clone();
    let study = study.to_string();
    catalog::update(root, move |c| {
        for id in &removed {
            c.fourd_custom.remove(id);
        }
        for g in &groups {
            if g.dissolved {
                if !g.id.is_empty() && sets(c).iter().any(|f| f.id == g.id) {
                    confirm_in(c, &g.id, false, "dissolved in the viewer")?;
                }
                continue;
            }
            if !g.custom {
                continue;
            }
            let members: Vec<(String, String, String)> = g
                .members
                .iter()
                .map(|m| {
                    (
                        m.series_uid.clone(),
                        m.role.kind().word().to_string(),
                        if m.role == crate::fourd::Role::Phase {
                            m.label.trim_start_matches('~').to_string()
                        } else {
                            String::new()
                        },
                    )
                })
                .collect();
            let all_here = members
                .iter()
                .all(|m| c.series.get(&m.0).is_some_and(|s| s.study == study));
            if !members.is_empty() && all_here {
                set_group_in(c, &study, &members, &g.name)?;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scripts_labels_are_recognised() {
        let f = SeriesFourD {
            group: "g".into(),
            role: "phase".into(),
            phase: "30%".into(),
            inferred: true,
            variant: "EFOV".into(),
        };
        assert_eq!(script_label(&f, true).as_deref(), Some("4DCT ~30% EFOV"));
        assert_eq!(script_label(&f, false).as_deref(), Some("4DCT? ~30% EFOV"));
        assert!(script_wrote("4DCT 30%") && script_wrote("4DCT? untagged") && script_wrote("4DCT"));
        assert!(!script_wrote("4DCTX") && !script_wrote("Planning CT"));
        assert_eq!(
            strip_study_suffix("2024-06-10 (Upright: #4) \u{b7} CT \u{b7} 4DCT\u{d7}2").as_deref(),
            Some("2024-06-10 (Upright: #4) \u{b7} CT")
        );
        assert_eq!(strip_study_suffix("2024-06-10 \u{b7} CT"), None);
    }
}
