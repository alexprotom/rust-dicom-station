//! Grouping image series into 4D sub-studies.
//!
//! A 4DCT arrives as one series per respiratory phase, usually with an
//! average (and sometimes a MIP) reconstruction beside them. DICOM stores
//! no node for the acquisition they belong to - the phase lives in the
//! series description ("Thorax 4D 30%") or, for enhanced exports, in
//! TemporalPositionIdentifier. This module reconstructs that node: it
//! recognises the phase series of a study, orders them, and files the
//! companion reconstructions with them, so the data tree can show one
//! "4D" group and the motion tools can iterate over its phases.
//!
//! Detection is a heuristic over headers, so every result can be corrected
//! by hand: groups built or edited in the tree are marked [`FourDGroup::
//! custom`] and are never replaced by re-detection.
//!
//! Members reference series by UID, not by index - series are renamed,
//! removed and moved between datasets, and a UID survives all of that
//! (an unresolvable UID simply drops out of the resolved view).

use crate::loader::SeriesInfo;

/// What a member series is within its group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// One respiratory (or cardiac) phase of the acquisition.
    Phase,
    /// The time-averaged reconstruction.
    Average,
    /// Maximum-intensity projection over the phases.
    Mip,
    /// Minimum-intensity projection over the phases.
    MinIp,
}

impl Role {
    /// Short tag shown after the member label in the tree.
    pub fn tag(self) -> &'static str {
        match self {
            Role::Phase => "",
            Role::Average => "AVG",
            Role::Mip => "MIP",
            Role::MinIp => "MinIP",
        }
    }
}

/// One series of a 4D group.
#[derive(Clone, Debug)]
pub struct Member {
    /// SeriesInstanceUID - the stable identity of the series.
    pub series_uid: String,
    /// What the member is called within the group: "0%", "50%", "t3", "AVG".
    pub label: String,
    pub role: Role,
    /// Respiratory phase in percent, when the description declared one.
    pub percent: Option<f32>,
}

/// A group of image series that form one 4D acquisition.
#[derive(Clone, Debug)]
pub struct FourDGroup {
    /// Name shown in the tree; renameable.
    pub name: String,
    /// Study the group belongs to (all members share it).
    pub study_uid: String,
    /// The members, phases first in temporal order, then the
    /// reconstructions (AVG, MIP, …).
    pub members: Vec<Member>,
    /// Built or edited by hand - re-detection must not replace it.
    pub custom: bool,
    /// Dissolved by hand. The group stays as a hidden tombstone so
    /// re-detection does not resurrect it; an explicit *Re-detect 4D
    /// groups* clears the tombstones.
    pub dissolved: bool,
}

impl FourDGroup {
    /// Indices of the members' series within `series`, in member order;
    /// `None` for a member whose series is gone.
    pub fn resolve(&self, series: &[SeriesInfo]) -> Vec<Option<usize>> {
        self.members
            .iter()
            .map(|m| series.iter().position(|s| s.uid == m.series_uid))
            .collect()
    }

    /// Positions (within `members`) of the phase members, in order.
    pub fn phase_members(&self) -> Vec<usize> {
        (0..self.members.len())
            .filter(|&i| self.members[i].role == Role::Phase)
            .collect()
    }

    /// The member the motion tools should use as the reference phase by
    /// default: the 0 % phase when there is one, else the first phase.
    pub fn default_reference(&self) -> Option<usize> {
        let phases = self.phase_members();
        phases
            .iter()
            .copied()
            .find(|&i| self.members[i].percent.is_some_and(|p| p.abs() < 0.01))
            .or_else(|| phases.first().copied())
    }

    /// `4D CT - Thorax (10 phases + 1)`, the default group name; the `+ 1`
    /// counts the AVG / MIP members.
    fn derive_name(modality: &str, stem: &str, n_phases: usize, extras: usize) -> String {
        // Leftover separators around the removed phase number ("4DCT_") are
        // not part of the name.
        let stem = stem.trim_matches(|c: char| {
            c.is_whitespace() || matches!(c, '_' | '-' | '—' | ':' | ',' | '.')
        });
        let what = if stem.is_empty() || stem.eq_ignore_ascii_case(&format!("4D {modality}")) {
            format!("4D {modality}")
        } else {
            format!("4D {modality} - {stem}")
        };
        if extras > 0 {
            format!("{what} ({n_phases} phases + {extras})")
        } else {
            format!("{what} ({n_phases} phases)")
        }
    }
}

/// The phase hint one description carries.
#[derive(Clone, PartialEq, Debug)]
enum Hint {
    /// A phase series: its number, the description with the number removed
    /// (the *template*, identifying which 4D set it belongs to - original
    /// case, whitespace collapsed; compared case-insensitively), and
    /// whether the number was written as a literal percent ("30%") rather
    /// than a keyword form ("phase_030") - which decides the label.
    Phase {
        number: f32,
        template: String,
        literal_pct: bool,
    },
    Average,
    Mip,
    MinIp,
    None,
}

impl Hint {
    /// The member label a phase hint yields: "30%" for a literal percent,
    /// "phase 3" for the keyword form.
    fn phase_label(number: f32, literal_pct: bool) -> String {
        let n = if number.fract().abs() < 1e-3 {
            format!("{}", number as i64)
        } else {
            format!("{number}")
        };
        if literal_pct {
            format!("{n}%")
        } else {
            format!("phase {n}")
        }
    }
}

/// Parse the phase hint out of a series description.
fn hint_of(desc: &str) -> Hint {
    let lower = desc.to_lowercase();
    // Projections and averages first: "MinIP" contains "mip" backwards
    // ordering would misfile it.
    for (needle, hint) in [
        ("minip", Hint::MinIp),
        ("min-ip", Hint::MinIp),
        ("min ip", Hint::MinIp),
        ("mip", Hint::Mip),
        ("average", Hint::Average),
        ("avg", Hint::Average),
        (" ave ", Hint::Average),
        ("mean", Hint::Average),
    ] {
        if lower.contains(needle) {
            return hint;
        }
    }
    // A number immediately followed by '%' (possibly with a space): the
    // respiratory phase. The description with the number removed is the
    // template that tells two 4D sets in one study apart. The scan runs on
    // the original bytes - indices into the lowercased copy would not be
    // valid slice positions of `desc` for non-ASCII descriptions.
    let bytes = desc.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b != b'%' {
            continue;
        }
        // Walk back over an optional space, then the digits.
        let mut j = i;
        if j > 0 && bytes[j - 1] == b' ' {
            j -= 1;
        }
        let end = j;
        while j > 0 && (bytes[j - 1].is_ascii_digit() || bytes[j - 1] == b'.') {
            j -= 1;
        }
        if j == end {
            continue; // '%' with no number before it
        }
        if let Ok(pct) = desc[j..end].parse::<f32>() {
            if (0.0..=100.0).contains(&pct) {
                let template = format!("{}{}", &desc[..j], &desc[i + 1..]);
                return Hint::Phase {
                    number: pct,
                    template: normalize_template(&template),
                    literal_pct: true,
                };
            }
        }
    }
    // The keyword form: "phase" followed by separators and a number
    // ("4DCT_phase_000", "Phase 3", "phase-50"). "Phase contrast" and
    // friends have no number there and fall through.
    let find_phase = || {
        bytes
            .windows(5)
            .position(|w| w.eq_ignore_ascii_case(b"phase"))
    };
    if let Some(p) = find_phase() {
        let tail = &bytes[p + 5..];
        let mut k = 0;
        while k < tail.len() && (tail[k] == b' ' || tail[k] == b'_' || tail[k] == b'-') {
            k += 1;
        }
        let start = k;
        while k < tail.len() && (tail[k].is_ascii_digit() || tail[k] == b'.') {
            k += 1;
        }
        if k > start {
            if let Ok(n) = desc[p + 5 + start..p + 5 + k].parse::<f32>() {
                if (0.0..=100.0).contains(&n) {
                    // Template: the description with "phase…<number>" removed.
                    let template = format!("{}{}", &desc[..p], &desc[p + 5 + k..]);
                    return Hint::Phase {
                        number: n,
                        template: normalize_template(&template),
                        literal_pct: false,
                    };
                }
            }
        }
    }
    Hint::None
}

/// Collapse runs of whitespace, keeping the original case (the template is
/// also what the group is named after).
fn normalize_template(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Recognise the 4D groups of a series list.
///
/// Series are bucketed by (study, modality); within a bucket, series whose
/// descriptions carry a percent phase are grouped by their description
/// template, and series with a TemporalPositionIdentifier but no percent
/// are grouped by identical description. A group needs at least three
/// phases - two series with "50%" in the name are more likely a coincidence
/// than an acquisition. Average / MIP / MinIP reconstructions of the bucket
/// are attached to its first group.
pub fn detect(series: &[SeriesInfo]) -> Vec<FourDGroup> {
    // (study_uid, modality) buckets, in first-seen order.
    let mut buckets: Vec<(String, String, Vec<usize>)> = Vec::new();
    for (i, s) in series.iter().enumerate() {
        match buckets
            .iter_mut()
            .find(|(st, m, _)| *st == s.study_uid && *m == s.modality)
        {
            Some((_, _, v)) => v.push(i),
            None => buckets.push((s.study_uid.clone(), s.modality.clone(), vec![i])),
        }
    }

    let mut out = Vec::new();
    for (study_uid, modality, idxs) in buckets {
        // Percent-tagged series by template, in first-seen order.
        // (template, written as a literal percent, [(series index, number)]).
        type Template = (String, bool, Vec<(usize, f32)>);
        let mut templates: Vec<Template> = Vec::new();
        // Temporal-index series by identical description.
        let mut temporal: Vec<(String, Vec<(usize, i64)>)> = Vec::new();
        let mut extras: Vec<(usize, Role)> = Vec::new();
        for &i in &idxs {
            let s = &series[i];
            match hint_of(&s.description) {
                Hint::Phase {
                    number,
                    template: tpl,
                    literal_pct,
                } => match templates
                    .iter_mut()
                    .find(|(t, _, _)| t.eq_ignore_ascii_case(&tpl))
                {
                    Some((_, _, v)) => v.push((i, number)),
                    None => templates.push((tpl, literal_pct, vec![(i, number)])),
                },
                Hint::Average => extras.push((i, Role::Average)),
                Hint::Mip => extras.push((i, Role::Mip)),
                Hint::MinIp => extras.push((i, Role::MinIp)),
                Hint::None => {
                    if let Some(t) = s.temporal_id {
                        let key = normalize_template(&s.description);
                        match temporal
                            .iter_mut()
                            .find(|(k, _)| k.eq_ignore_ascii_case(&key))
                        {
                            Some((_, v)) => v.push((i, t)),
                            None => temporal.push((key, vec![(i, t)])),
                        }
                    }
                }
            }
        }

        // Each group is kept with the stem its name was derived from, so
        // attaching the reconstructions can re-derive the name without
        // parsing it back apart (a stem may itself contain " - " or "(").
        let mut groups_here: Vec<(FourDGroup, String)> = Vec::new();
        for (tpl, literal_pct, mut members) in templates {
            if members.len() < 3 {
                continue;
            }
            members.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            let n = members.len();
            let members: Vec<Member> = members
                .into_iter()
                .map(|(i, pct)| Member {
                    series_uid: series[i].uid.clone(),
                    label: Hint::phase_label(pct, literal_pct),
                    role: Role::Phase,
                    percent: Some(pct),
                })
                .collect();
            groups_here.push((
                FourDGroup {
                    name: FourDGroup::derive_name(&modality, &tpl, n, 0),
                    study_uid: study_uid.clone(),
                    members,
                    custom: false,
                    dissolved: false,
                },
                tpl,
            ));
        }
        for (_, mut members) in temporal {
            if members.len() < 3 {
                continue;
            }
            members.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
            let n = members.len();
            let desc = series[members[0].0].description.clone();
            let members: Vec<Member> = members
                .into_iter()
                .map(|(i, t)| Member {
                    series_uid: series[i].uid.clone(),
                    label: format!("t{t}"),
                    role: Role::Phase,
                    percent: None,
                })
                .collect();
            let name = FourDGroup::derive_name(&modality, &desc, n, 0);
            groups_here.push((
                FourDGroup {
                    name,
                    study_uid: study_uid.clone(),
                    members,
                    custom: false,
                    dissolved: false,
                },
                desc,
            ));
        }

        // The bucket's reconstructions belong to its (usually only) group.
        if let (Some((g, stem)), false) = (groups_here.first_mut(), extras.is_empty()) {
            for (i, role) in &extras {
                g.members.push(Member {
                    series_uid: series[*i].uid.clone(),
                    label: role.tag().to_string(),
                    role: *role,
                    percent: None,
                });
            }
            let n_phases = g.phase_members().len();
            g.name = FourDGroup::derive_name(&modality, stem, n_phases, extras.len());
        }
        out.extend(groups_here.into_iter().map(|(g, _)| g));
    }
    out
}

/// A [`Member`] for one series added to a group by hand: the phase percent
/// is parsed from the description when there is one, otherwise the series
/// becomes phase `t<n>` (1-based position it will get in the group).
pub fn member_for(series: &SeriesInfo, position: usize) -> Member {
    match hint_of(&series.description) {
        Hint::Phase {
            number,
            literal_pct,
            ..
        } => Member {
            series_uid: series.uid.clone(),
            label: Hint::phase_label(number, literal_pct),
            role: Role::Phase,
            percent: Some(number),
        },
        Hint::Average => Member {
            series_uid: series.uid.clone(),
            label: Role::Average.tag().to_string(),
            role: Role::Average,
            percent: None,
        },
        Hint::Mip => Member {
            series_uid: series.uid.clone(),
            label: Role::Mip.tag().to_string(),
            role: Role::Mip,
            percent: None,
        },
        Hint::MinIp => Member {
            series_uid: series.uid.clone(),
            label: Role::MinIp.tag().to_string(),
            role: Role::MinIp,
            percent: None,
        },
        Hint::None => Member {
            series_uid: series.uid.clone(),
            label: format!("t{position}"),
            role: Role::Phase,
            percent: None,
        },
    }
}

/// Re-detect after the series list changed, keeping every custom group and
/// every custom edit: detected groups that share a member with a custom
/// group are dropped in its favour.
pub fn refresh(existing: &[FourDGroup], series: &[SeriesInfo]) -> Vec<FourDGroup> {
    // Custom groups and dissolved tombstones both survive re-detection and
    // both suppress a detected group that shares a member with them.
    let kept: Vec<FourDGroup> = existing
        .iter()
        .filter(|g| g.custom || g.dissolved)
        .cloned()
        .collect();
    let mut out = kept.clone();
    for g in detect(series) {
        let overlaps = kept.iter().any(|c| {
            c.members
                .iter()
                .any(|m| g.members.iter().any(|n| n.series_uid == m.series_uid))
        });
        if !overlaps {
            out.push(g);
        }
    }
    // A group whose members are all gone has nothing left to say.
    out.retain(|g| g.resolve(series).iter().any(|r| r.is_some()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(uid: &str, desc: &str, study: &str, modality: &str) -> SeriesInfo {
        SeriesInfo {
            uid: uid.into(),
            modality: modality.into(),
            description: desc.into(),
            patient_id: "P1".into(),
            patient_name: "Test^Patient".into(),
            study_uid: study.into(),
            study_date: String::new(),
            study_description: String::new(),
            series_number: None,
            temporal_id: None,
            files: Vec::new(),
        }
    }

    #[test]
    fn percent_hints_are_parsed_out_of_descriptions() {
        assert_eq!(
            hint_of("Thorax 4D 30%"),
            Hint::Phase {
                number: 30.0,
                template: "Thorax 4D".into(),
                literal_pct: true
            }
        );
        assert_eq!(
            hint_of("CT 0 % Ex"),
            Hint::Phase {
                number: 0.0,
                template: "CT Ex".into(),
                literal_pct: true
            },
            "space before the percent sign"
        );
        assert_eq!(
            hint_of("4DCT_phase_000"),
            Hint::Phase {
                number: 0.0,
                template: "4DCT_".into(),
                literal_pct: false
            },
            "keyword form without a percent sign"
        );
        assert_eq!(hint_of("Phase contrast MR"), Hint::None);
        assert_eq!(hint_of("4D MIP"), Hint::Mip);
        assert_eq!(hint_of("Untagged Average CT"), Hint::Average);
        assert_eq!(hint_of("MinIP recon"), Hint::MinIp);
        assert_eq!(hint_of("Cardiac CCT"), Hint::None);
        assert_eq!(hint_of("Contrast 300%"), Hint::None, "over 100 %");
    }

    #[test]
    fn a_ten_phase_study_with_avg_becomes_one_ordered_group() {
        let mut v: Vec<SeriesInfo> = (0..10)
            .map(|p| {
                series(
                    &format!("uid{p}"),
                    &format!("Thorax 4D {}%", p * 10),
                    "st1",
                    "CT",
                )
            })
            .collect();
        // Shuffle the arrival order and add the reconstructions.
        v.swap(0, 7);
        v.push(series("uidavg", "Thorax 4D AVG", "st1", "CT"));
        v.push(series("uidmip", "Thorax 4D MIP", "st1", "CT"));
        let groups = detect(&v);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.phase_members().len(), 10);
        assert_eq!(g.members.len(), 12);
        let labels: Vec<&str> = g.members.iter().map(|m| m.label.as_str()).collect();
        assert_eq!(
            labels,
            ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "AVG", "MIP"]
        );
        assert_eq!(g.default_reference(), Some(0));
        assert!(g.name.contains("10 phases"), "{}", g.name);
    }

    #[test]
    fn two_recons_of_the_same_phases_split_into_two_groups() {
        let mut v = Vec::new();
        for p in [0, 30, 60] {
            v.push(series(
                &format!("a{p}"),
                &format!("Lung 4D {p}%"),
                "st1",
                "CT",
            ));
            v.push(series(
                &format!("b{p}"),
                &format!("Lung 4D thin {p}%"),
                "st1",
                "CT",
            ));
        }
        let groups = detect(&v);
        assert_eq!(groups.len(), 2);
        assert!(groups.iter().all(|g| g.phase_members().len() == 3));
    }

    #[test]
    fn too_few_phases_or_other_studies_do_not_group() {
        let v = vec![
            series("u1", "CT 0%", "st1", "CT"),
            series("u2", "CT 50%", "st1", "CT"),
            series("u3", "CT 50%", "st2", "CT"), // other study
            series("u4", "Head CT", "st1", "CT"),
        ];
        assert!(detect(&v).is_empty());
    }

    #[test]
    fn temporal_position_identifiers_group_without_percent() {
        let mut v: Vec<SeriesInfo> = (0..4)
            .map(|t| {
                let mut s = series(&format!("u{t}"), "Dynamic MR", "st1", "MR");
                s.temporal_id = Some(4 - t); // reversed arrival order
                s
            })
            .collect();
        v.push(series("plain", "Localizer", "st1", "MR"));
        let groups = detect(&v);
        assert_eq!(groups.len(), 1);
        let labels: Vec<&str> = groups[0].members.iter().map(|m| m.label.as_str()).collect();
        assert_eq!(labels, ["t1", "t2", "t3", "t4"]);
    }

    #[test]
    fn a_dissolved_group_stays_dissolved_through_refresh() {
        let v: Vec<SeriesInfo> = (0..3)
            .map(|p| series(&format!("u{p}"), &format!("4D {}%", p * 10), "st1", "CT"))
            .collect();
        let mut groups = detect(&v);
        assert_eq!(groups.len(), 1);
        groups[0].dissolved = true;
        let refreshed = refresh(&groups, &v);
        assert_eq!(refreshed.len(), 1, "the tombstone suppresses re-detection");
        assert!(refreshed[0].dissolved);
        // Clearing the tombstone (an explicit re-detect) resurrects it.
        let cleared: Vec<FourDGroup> = refreshed.into_iter().filter(|g| !g.dissolved).collect();
        let redetected = refresh(&cleared, &v);
        assert_eq!(redetected.len(), 1);
        assert!(!redetected[0].dissolved);
    }

    #[test]
    fn refresh_keeps_custom_groups_and_drops_their_double_detection() {
        let v: Vec<SeriesInfo> = (0..3)
            .map(|p| series(&format!("u{p}"), &format!("4D {}%", p * 10), "st1", "CT"))
            .collect();
        let mut groups = detect(&v);
        assert_eq!(groups.len(), 1);
        groups[0].custom = true;
        groups[0].name = "my group".into();
        let refreshed = refresh(&groups, &v);
        assert_eq!(refreshed.len(), 1, "no duplicate of the custom group");
        assert_eq!(refreshed[0].name, "my group");
        // A custom group whose series all vanished is dropped.
        let refreshed = refresh(&groups, &[]);
        assert!(refreshed.is_empty());
    }

    #[test]
    fn members_resolve_by_uid_after_reordering() {
        let mut v: Vec<SeriesInfo> = (0..3)
            .map(|p| series(&format!("u{p}"), &format!("4D {}%", p * 10), "st1", "CT"))
            .collect();
        let groups = detect(&v);
        v.reverse();
        let r = groups[0].resolve(&v);
        assert_eq!(r, vec![Some(2), Some(1), Some(0)]);
    }
}
