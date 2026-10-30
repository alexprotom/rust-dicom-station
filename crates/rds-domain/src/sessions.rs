//! Treatment sessions: the radiographs, registrations and treatment records
//! of a course, put together by the day they were made and the beam they
//! were made for.
//!
//! A treatment day on a room with a setup imaging system leaves a trail of
//! objects behind it:
//!
//! * a daily CT, a registration that places it in the room (*imaging*) and
//!   one that maps it onto the planning CT (*CT to CT*);
//! * then per beam one or more **setup pairs** (two orthogonal radiographs
//!   taken together), the **treatment** registration that holds the setup
//!   correction, written when the last pair is approved, the **treatment
//!   records** of the delivery (more than one when it was interrupted) and
//!   a **verification pair** after the beam.
//!
//! The standard references would tie these together, but the setup systems
//! write few of them, so after what references there are the rules are by
//! time and by kind. Measured on a real course of 15 fractions (87 pairs,
//! 87 registrations, 47 records), they are:
//!
//! 1. **Pair.** Two projection images with the same SeriesNumber, the same
//!    InstanceNumber and the same acquisition date, acquired at most
//!    [`PAIR_SECONDS`] apart. Never keyed on the review time: one review
//!    approves every pair of a setup iteration at once.
//! 2. **Setup pairs to their treatment registration.** A pair reviewed
//!    0 to [`REVIEW_TO_REG_SECONDS`] before a treatment registration of the
//!    same day, and acquired before it, is a setup pair of that
//!    registration's step; the latest of them is the final position.
//! 3. **Record to its step.** A record belongs to the latest treatment
//!    registration of the same day before its TreatmentDate/Time. A record
//!    with none before it (a day with no setup imaging) belongs to the
//!    day's session only.
//! 4. **Verification pair.** A pair that is not a setup pair, taken after
//!    the last record of a step and before the next step's registration,
//!    verifies that step. Anything else of the day is a loose pair.
//! 5. **Which registration is which**, by what it references - never by the
//!    vendor's comment text: a plan loaded in the workspace makes it a
//!    *treatment* registration, two loaded image series a *CT to CT* one,
//!    one series or nothing an *imaging* one.
//! 6. **Session.** One per calendar date with at least one record or one
//!    treatment registration; numbered by date order ("3 of 16"). The daily
//!    images are the image series made that day, and the imaging and CT to
//!    CT registrations of that day are its own. Only a course with at least
//!    one treatment registration has sessions: records alone (a plain RT
//!    study's) stay the study's list, as they always were.
//! 7. **Report.** A report made on a session's day (by its series' date,
//!    [`crate::report::report_datetime`]) is the session's; one made at most
//!    [`PAIR_SECONDS`] from a pair of the session - the dose report a setup
//!    imaging system writes for each exposure - is that pair's.
//!
//! What the files get wrong, and is tolerated rather than corrected:
//!
//! * A registration's Referenced SOP Instance UID may hold a **Series**
//!   Instance UID, so references are matched against the loaded series as
//!   well as the plans; a registration whose own Frame of Reference UID is
//!   a plan's SOP Instance UID references that plan; the class a plan is
//!   referenced with (RT Plan for an RT Ion Plan) is ignored.
//! * RT Images may carry no frame of reference, plan, beam or geometry:
//!   time, SeriesNumber, InstanceNumber and the review are all there is. A
//!   plan an image or a record does name is honoured: it only links to a
//!   registration that references the same plan.
//! * Current Fraction Number may be 1 in every record, so a session is
//!   numbered by date order and never by it.
//! * A record's control-point times may be on another clock than its
//!   Treatment Time: only TreatmentDate/Time (else the series') is used.
//!
//! Everything here is pure and deterministic: every list is sorted by its
//! time stamp, then series number, then index, so the same folder gives the
//! same sessions whatever order the disk lists its files in. What no rule
//! links is left where it was; the data tree draws it in its study's flat
//! lists.

use std::collections::{BTreeSet, HashSet};

use crate::extras::{PlanarImage, PlanarKind, SpatialReg, TreatRecord};
use crate::loader::LoadedStudy;

/// The two images of an orthogonal pair are acquired at most this many
/// seconds apart (both in the same second in most pairs, one apart in some).
pub const PAIR_SECONDS: i64 = 2;
/// A treatment registration is written at most this many seconds after the
/// review of the setup pairs it was made from.
pub const REVIEW_TO_REG_SECONDS: i64 = 5;

/// Seconds since 1970-01-01 of a `YYYYMMDDHHMMSS` stamp
/// ([`crate::loader::datetime_of`]); `None` for an empty or malformed one.
pub fn seconds(stamp: &str) -> Option<i64> {
    if stamp.len() < 14 || !stamp.is_char_boundary(14) {
        return None;
    }
    let num = |a: usize, b: usize| stamp.get(a..b)?.parse::<i64>().ok();
    let (y, m, d) = (num(0, 4)?, num(4, 6)?, num(6, 8)?);
    let (hh, mm, ss) = (num(8, 10)?, num(10, 12)?, num(12, 14)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) || hh > 23 || mm > 59 || ss > 60 {
        return None;
    }
    // Days from the civil date (Howard Hinnant's algorithm).
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// The date of a stamp, `YYYYMMDD`; empty for an empty stamp.
pub fn day(stamp: &str) -> &str {
    stamp.get(..8).unwrap_or("")
}

/// `HH:MM:SS` of a stamp, for a row of the data tree.
pub fn clock(stamp: &str) -> String {
    match stamp.get(8..14) {
        Some(t) => format!("{}:{}:{}", &t[..2], &t[2..4], &t[4..]),
        None => String::new(),
    }
}

/// `YYYY-MM-DD` of a `YYYYMMDD` date.
pub fn date_label(date: &str) -> String {
    match (date.get(..4), date.get(4..6), date.get(6..8)) {
        (Some(y), Some(m), Some(d)) => format!("{y}-{m}-{d}"),
        _ => date.to_string(),
    }
}

/// Two radiographs taken together, or one whose partner is missing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Pair {
    /// Indices into `planar_images`, in acquisition order: two, or one.
    pub images: Vec<usize>,
    /// The earlier acquisition stamp of the two.
    pub acquired: String,
    /// The latest review stamp of the two, empty if neither was reviewed.
    pub reviewed: String,
    pub series_number: Option<i64>,
    /// Reports made with it (indices into `reports`): its dose report.
    pub reports: Vec<usize>,
}

/// What a registration is for, told by what it references (rule 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegKind {
    /// References a loaded plan: the setup correction of a beam.
    Treatment,
    /// References two loaded image series: a daily CT onto the planning CT.
    CtToCt,
    /// References one image series or nothing: the images in the room.
    Imaging,
}

/// One beam of a session: a treatment registration with the setup pairs it
/// was made from, the records of the delivery and the pair that verified it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    /// The beam of its records, else the beam its images reference, else
    /// "Beam".
    pub name: String,
    /// Index into `registrations`.
    pub treatment_reg: Option<usize>,
    /// Setup iterations in acquisition order; the last is the final one.
    pub setup: Vec<Pair>,
    /// Indices into `treat_records`, in delivery order.
    pub records: Vec<usize>,
    pub verification: Vec<Pair>,
    /// Every beam of every record terminated `NORMAL` (false with no record).
    pub all_normal: bool,
}

/// One treatment day.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Session {
    /// `YYYYMMDD`.
    pub date: String,
    /// 1-based, by date order among the sessions put together at once.
    pub number: usize,
    pub of: usize,
    /// Image series (indices into `series`) made that day: the daily CT.
    pub series: Vec<usize>,
    /// The day's imaging registrations (indices into `registrations`).
    pub image_regs: Vec<usize>,
    /// The day's CT to CT registrations.
    pub ct_regs: Vec<usize>,
    pub steps: Vec<Step>,
    /// Records with no treatment registration before them that day.
    pub loose_records: Vec<usize>,
    /// Pairs of the day that no step took.
    pub loose_pairs: Vec<Pair>,
    /// Reports of the day that no pair took (indices into `reports`).
    pub loose_reports: Vec<usize>,
}

impl Session {
    /// Every registration the session holds.
    pub fn regs(&self) -> impl Iterator<Item = usize> + '_ {
        self.image_regs
            .iter()
            .chain(&self.ct_regs)
            .copied()
            .chain(self.steps.iter().filter_map(|s| s.treatment_reg))
    }

    /// Every record the session holds.
    pub fn records(&self) -> impl Iterator<Item = usize> + '_ {
        self.steps
            .iter()
            .flat_map(|s| s.records.iter())
            .chain(&self.loose_records)
            .copied()
    }

    /// Every pair the session holds.
    pub fn pairs(&self) -> impl Iterator<Item = &Pair> + '_ {
        self.steps
            .iter()
            .flat_map(|s| s.setup.iter().chain(&s.verification))
            .chain(&self.loose_pairs)
    }

    /// Every report the session holds.
    pub fn reports(&self) -> impl Iterator<Item = usize> + '_ {
        self.pairs()
            .flat_map(|p| p.reports.iter())
            .chain(&self.loose_reports)
            .copied()
    }

    /// Every planar image the session holds.
    pub fn images(&self) -> impl Iterator<Item = usize> + '_ {
        self.pairs().flat_map(|p| p.images.iter().copied())
    }
}

/// Which objects of a workspace to put into sessions: the data tree asks
/// once per study, with what it files under that study.
#[derive(Clone, Debug, Default)]
pub struct Part {
    pub series: Vec<usize>,
    pub images: Vec<usize>,
    pub regs: Vec<usize>,
    pub records: Vec<usize>,
    pub reports: Vec<usize>,
}

impl Part {
    /// Everything the workspace holds.
    pub fn all(study: &LoadedStudy) -> Part {
        Part {
            series: (0..study.series.len()).collect(),
            images: (0..study.planar_images.len()).collect(),
            regs: (0..study.registrations.len()).collect(),
            records: (0..study.treat_records.len()).collect(),
            reports: (0..study.reports.len()).collect(),
        }
    }
}

/// The kind of a registration against what the workspace has loaded.
pub fn reg_kind(reg: &SpatialReg, study: &LoadedStudy) -> RegKind {
    let plans: HashSet<&str> = plan_uids(study);
    let series: HashSet<&str> = study.series.iter().map(|s| s.uid.as_str()).collect();
    kind_of(reg, &plans, &series)
}

fn plan_uids(study: &LoadedStudy) -> HashSet<&str> {
    study
        .plans
        .iter()
        .map(|p| p.sop_instance_uid.as_str())
        .filter(|u| !u.is_empty())
        .collect()
}

fn kind_of(reg: &SpatialReg, plans: &HashSet<&str>, series: &HashSet<&str>) -> RegKind {
    let names_plan = reg.references.iter().any(|u| plans.contains(u.as_str()))
        || plans.contains(reg.frame_of_reference_uid.as_str());
    if names_plan {
        return RegKind::Treatment;
    }
    let images = reg
        .references
        .iter()
        .filter(|u| series.contains(u.as_str()))
        .count();
    if images >= 2 {
        RegKind::CtToCt
    } else {
        RegKind::Imaging
    }
}

/// The orthogonal pairs among all the projection images (rule 1).
pub fn pairs(images: &[PlanarImage]) -> Vec<Pair> {
    pairs_of(images, &(0..images.len()).collect::<Vec<_>>())
}

/// The pairs among the images `which` names. Images made here (a DRR has
/// no acquisition time) and additional objects are never paired.
pub fn pairs_of(images: &[PlanarImage], which: &[usize]) -> Vec<Pair> {
    let mut cand: Vec<(usize, i64)> = which
        .iter()
        .filter_map(|&i| {
            let im = images.get(i)?;
            (im.kind == PlanarKind::Projection)
                .then(|| seconds(&im.acquired))
                .flatten()
                .map(|t| (i, t))
        })
        .collect();
    let key = |&(i, t): &(usize, i64)| {
        let im = &images[i];
        (im.series_number, im.instance_number, t, i)
    };
    cand.sort_by_key(key);
    let mut out = Vec::new();
    let mut used = vec![false; cand.len()];
    for a in 0..cand.len() {
        if used[a] {
            continue;
        }
        used[a] = true;
        let (ia, ta) = cand[a];
        let ka = &images[ia];
        let partner = (a + 1..cand.len()).find(|&b| {
            let (ib, tb) = cand[b];
            let im = &images[ib];
            !used[b]
                && im.series_number == ka.series_number
                && im.instance_number == ka.instance_number
                && day(&im.acquired) == day(&ka.acquired)
                && (tb - ta).abs() <= PAIR_SECONDS
        });
        let mut members = vec![ia];
        if let Some(b) = partner {
            used[b] = true;
            members.push(cand[b].0);
        }
        let reviewed = members
            .iter()
            .map(|&i| images[i].reviewed.as_str())
            .max()
            .unwrap_or("")
            .to_string();
        out.push(Pair {
            images: members,
            acquired: ka.acquired.clone(),
            reviewed,
            series_number: ka.series_number,
            reports: Vec::new(),
        });
    }
    out.sort_by(|p, q| {
        (&p.acquired, p.series_number, p.images[0]).cmp(&(
            &q.acquired,
            q.series_number,
            q.images[0],
        ))
    });
    out
}

/// The sessions of everything the workspace holds.
pub fn sessions(study: &LoadedStudy) -> Vec<Session> {
    sessions_in(study, &Part::all(study))
}

/// The sessions of the objects `part` names (rules 2 to 7).
pub fn sessions_in(study: &LoadedStudy, part: &Part) -> Vec<Session> {
    let plans = plan_uids(study);
    let series_uids: HashSet<&str> = study.series.iter().map(|s| s.uid.as_str()).collect();

    // Registrations and records with a time, in time order.
    let mut regs: Vec<(usize, i64, RegKind)> = part
        .regs
        .iter()
        .filter_map(|&i| {
            let r = study.registrations.get(i)?;
            Some((i, seconds(&r.datetime)?, kind_of(r, &plans, &series_uids)))
        })
        .collect();
    regs.sort_by_key(|&(i, t, _)| (t, study.registrations[i].series_number, i));
    let mut records: Vec<(usize, i64)> = part
        .records
        .iter()
        .filter_map(|&i| Some((i, seconds(&study.treat_records.get(i)?.datetime)?)))
        .collect();
    records.sort_by_key(|&(i, t)| (t, study.treat_records[i].series_number, i));

    if !regs.iter().any(|r| r.2 == RegKind::Treatment) {
        return Vec::new();
    }
    let reg_day = |i: usize| day(&study.registrations[i].datetime);
    let rec_day = |i: usize| day(&study.treat_records[i].datetime);
    let days: BTreeSet<&str> = regs
        .iter()
        .filter(|r| r.2 == RegKind::Treatment)
        .map(|r| reg_day(r.0))
        .chain(records.iter().map(|r| rec_day(r.0)))
        .collect();
    let of = days.len();
    let all_pairs = pairs_of(&study.planar_images, &part.images);

    let mut out = Vec::with_capacity(of);
    for (n, date) in days.into_iter().enumerate() {
        let treat: Vec<(usize, i64)> = regs
            .iter()
            .filter(|r| r.2 == RegKind::Treatment && reg_day(r.0) == date)
            .map(|r| (r.0, r.1))
            .collect();
        let mut steps: Vec<Step> = treat
            .iter()
            .map(|&(i, _)| Step {
                treatment_reg: Some(i),
                ..Default::default()
            })
            .collect();
        // A plan an object names must be one the registration references.
        let fits = |plan: &str, k: usize| {
            plan.is_empty()
                || study.registrations[treat[k].0]
                    .references
                    .iter()
                    .any(|u| u == plan)
        };
        let mut last_record: Vec<Option<i64>> = vec![None; steps.len()];

        let mut loose_records = Vec::new();
        for &(ri, t) in records.iter().filter(|r| rec_day(r.0) == date) {
            let plan = study.treat_records[ri].plan_uid.as_str();
            match (0..treat.len())
                .rev()
                .find(|&k| treat[k].1 <= t && fits(plan, k))
            {
                Some(k) => {
                    steps[k].records.push(ri);
                    last_record[k] = Some(t);
                }
                None => loose_records.push(ri),
            }
        }

        let mut loose_pairs = Vec::new();
        for pair in all_pairs.iter().filter(|p| day(&p.acquired) == date) {
            let Some(at) = seconds(&pair.acquired) else {
                continue;
            };
            let plan = pair
                .images
                .iter()
                .map(|&i| study.planar_images[i].plan_uid.as_str())
                .find(|p| !p.is_empty())
                .unwrap_or("");
            let setup = seconds(&pair.reviewed).and_then(|rv| {
                (0..treat.len()).find(|&k| {
                    let lag = treat[k].1 - rv;
                    (0..=REVIEW_TO_REG_SECONDS).contains(&lag) && at <= treat[k].1 && fits(plan, k)
                })
            });
            if let Some(k) = setup {
                steps[k].setup.push(pair.clone());
                continue;
            }
            // After the last record of the step whose registration came
            // last before it, and so before the next step's registration.
            let verifies = (0..treat.len())
                .rev()
                .find(|&k| treat[k].1 <= at)
                .filter(|&k| last_record[k].is_some_and(|lr| lr <= at));
            match verifies {
                Some(k) => steps[k].verification.push(pair.clone()),
                None => loose_pairs.push(pair.clone()),
            }
        }

        for step in &mut steps {
            step.all_normal = !step.records.is_empty()
                && step.records.iter().all(|&r| {
                    study.treat_records[r]
                        .beams
                        .iter()
                        .all(|b| b.termination_status == "NORMAL")
                });
            step.name = step_name(study, step);
        }

        let (mut image_regs, mut ct_regs) = (Vec::new(), Vec::new());
        for &(i, _, kind) in regs.iter().filter(|r| reg_day(r.0) == date) {
            match kind {
                RegKind::Imaging => image_regs.push(i),
                RegKind::CtToCt => ct_regs.push(i),
                RegKind::Treatment => {}
            }
        }
        let mut series: Vec<usize> = part
            .series
            .iter()
            .copied()
            .filter(|&s| {
                study
                    .series
                    .get(s)
                    .is_some_and(|se| day(&se.series_datetime) == date)
            })
            .collect();
        series.sort_by_key(|&s| (study.series[s].series_datetime.clone(), s));

        // The day's reports, each to the pair it was made with.
        let mut loose_reports = Vec::new();
        let mut day_reports: Vec<(usize, Option<i64>)> = part
            .reports
            .iter()
            .filter_map(|&r| {
                let rep = study.reports.get(r)?;
                (day(&rep.datetime) == date).then(|| (r, seconds(&rep.datetime)))
            })
            .collect();
        day_reports.sort_by_key(|&(r, t)| (t, r));
        for (r, t) in day_reports {
            let mut pairs: Vec<&mut Pair> = steps
                .iter_mut()
                .flat_map(|s| s.setup.iter_mut().chain(s.verification.iter_mut()))
                .chain(loose_pairs.iter_mut())
                .collect();
            let near = t.and_then(|t| {
                pairs
                    .iter()
                    .enumerate()
                    .filter_map(|(k, p)| Some((k, (seconds(&p.acquired)? - t).abs())))
                    .filter(|&(_, d)| d <= PAIR_SECONDS)
                    .min_by_key(|&(k, d)| (d, k))
                    .map(|(k, _)| k)
            });
            match near {
                Some(k) => pairs[k].reports.push(r),
                None => loose_reports.push(r),
            }
        }

        out.push(Session {
            date: date.to_string(),
            number: n + 1,
            of,
            series,
            image_regs,
            ct_regs,
            steps,
            loose_records,
            loose_pairs,
            loose_reports,
        });
    }
    out
}

/// The beam a step delivered: the beams of its first record, else the beam
/// one of its images references (named from the plan), else "Beam".
fn step_name(study: &LoadedStudy, step: &Step) -> String {
    if let Some(&r) = step.records.first() {
        let names: Vec<&str> = study.treat_records[r]
            .beams
            .iter()
            .map(|b| b.name.as_str())
            .collect();
        if !names.is_empty() {
            return names.join(" + ");
        }
    }
    let beam = step
        .setup
        .iter()
        .chain(&step.verification)
        .flat_map(|p| p.images.iter())
        .find_map(|&i| study.planar_images[i].referenced_beam);
    if let Some(number) = beam {
        let refs = step
            .treatment_reg
            .map(|r| study.registrations[r].references.as_slice())
            .unwrap_or(&[]);
        if let Some(b) = study
            .plans
            .iter()
            .filter(|p| refs.is_empty() || refs.contains(&p.sop_instance_uid))
            .flat_map(|p| p.beams.iter())
            .find(|b| b.number == number)
        {
            return b.name.clone();
        }
        return format!("Beam {number}");
    }
    "Beam".to_string()
}

/// A record's beams in one word each: `NORMAL` for all of them, else the
/// first other termination status.
pub fn record_status(rec: &TreatRecord) -> &str {
    rec.beams
        .iter()
        .map(|b| b.termination_status.as_str())
        .find(|s| *s != "NORMAL")
        .unwrap_or(if rec.beams.is_empty() { "" } else { "NORMAL" })
}

/// The setup correction a treatment registration holds, as
/// `[tx, ty, tz]` mm and `[rx, ry, rz]` degrees: the first matrix that is not
/// the identity, decomposed as [`crate::extras::matrix_to_rigid`] does
/// (`R = Rz·Ry·Rx`, the conventions of
/// [`crate::registration::RigidTransform`]). Zero when every matrix is the
/// identity; `None` without a matrix, or for one that is not rigid.
pub fn correction(reg: &SpatialReg) -> Option<[f64; 6]> {
    let item = reg
        .items
        .iter()
        .find(|it| !it.is_identity)
        .or(reg.items.first())?;
    if item.is_identity {
        return Some([0.0; 6]);
    }
    let p = crate::extras::matrix_to_rigid(&item.matrix, false)?.params();
    Some([
        p[3],
        p[4],
        p[5],
        p[0].to_degrees(),
        p[1].to_degrees(),
        p[2].to_degrees(),
    ])
}

/// A correction as one line: `Δ -0.8, 4.9, -4.4 mm · 0.50, 0.09, 0.33°`.
pub fn correction_text(c: &[f64; 6]) -> String {
    // A value that rounds to zero is written without its sign.
    let f = |v: f64, digits: usize| {
        let s = format!("{v:.digits$}");
        match s.strip_prefix('-') {
            Some(rest) if rest.chars().all(|ch| ch == '0' || ch == '.') => rest.to_string(),
            _ => s,
        }
    };
    format!(
        "Δ {}, {}, {} mm · {}, {}, {}°",
        f(c[0], 1),
        f(c[1], 1),
        f(c[2], 1),
        f(c[3], 2),
        f(c[4], 2),
        f(c[5], 2)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extras::{RecordBeam, RegMatrixItem};
    use crate::loader::SeriesInfo;
    use crate::rtplan::{BeamInfo, PlanInfo};

    const PLAN: &str = "9.9.1";
    const PLAN_CT: &str = "9.9.2";
    const DAILY_CT: &str = "9.9.3";

    /// A course with a planning CT, a daily CT on 2024-09-17 and a plan of
    /// two beams; the day's objects are added by each test.
    fn course() -> LoadedStudy {
        let series = |uid: &str, when: &str| SeriesInfo {
            uid: uid.into(),
            modality: "CT".into(),
            series_datetime: when.into(),
            ..Default::default()
        };
        let beam = |number: i32, name: &str| BeamInfo {
            number,
            name: name.into(),
            ..Default::default()
        };
        LoadedStudy {
            series: vec![
                series(PLAN_CT, "20240903115855"),
                series(DAILY_CT, "20240917103701"),
            ],
            plans: vec![PlanInfo {
                sop_instance_uid: PLAN.into(),
                beams: vec![beam(2, "LPO"), beam(3, "PA")],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn reg(when: &str, refs: &[&str]) -> SpatialReg {
        SpatialReg {
            datetime: when.into(),
            references: refs.iter().map(|s| s.to_string()).collect(),
            items: vec![RegMatrixItem {
                matrix: IDENTITY,
                is_identity: true,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn record(when: &str, beam: &str, status: &str) -> TreatRecord {
        TreatRecord {
            datetime: when.into(),
            plan_uid: PLAN.into(),
            beams: vec![RecordBeam {
                name: beam.into(),
                termination_status: status.into(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    /// The two images of a pair: same series and instance number.
    fn pair(study: &mut LoadedStudy, series: i64, instance: i64, at: [&str; 2], review: &str) {
        for when in at {
            study.planar_images.push(PlanarImage {
                kind: PlanarKind::Projection,
                series_number: Some(series),
                instance_number: Some(instance),
                acquired: when.into(),
                reviewed: review.into(),
                ..Default::default()
            });
        }
    }

    #[rustfmt::skip]
    const IDENTITY: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];

    /// 2024-09-17 as the setup system wrote it: CT, imaging and CT to CT
    /// registrations, two setup iterations for LPO, then PA.
    fn ordinary_day() -> LoadedStudy {
        let mut s = course();
        s.registrations = vec![
            reg("20240917104038", &[DAILY_CT]),
            reg("20240917104553", &[PLAN_CT, DAILY_CT]),
            reg("20240917105403", &[PLAN]),
            reg("20240917105933", &[PLAN]),
        ];
        s.treat_records = vec![
            record("20240917105418", "LPO", "NORMAL"),
            record("20240917105946", "PA", "NORMAL"),
        ];
        pair(
            &mut s,
            212,
            3,
            ["20240917105145", "20240917105146"],
            "20240917105402",
        );
        pair(
            &mut s,
            212,
            5,
            ["20240917105347", "20240917105347"],
            "20240917105402",
        );
        pair(
            &mut s,
            213,
            3,
            ["20240917105757", "20240917105757"],
            "20240917105802",
        );
        pair(
            &mut s,
            212,
            3,
            ["20240917105919", "20240917105919"],
            "20240917105932",
        );
        pair(
            &mut s,
            213,
            3,
            ["20240917110333", "20240917110333"],
            "20240917110338",
        );
        s
    }

    /// Every registration, record and image of the workspace is in at most
    /// one place of the sessions.
    fn each_once(study: &LoadedStudy, sessions: &[Session]) {
        let regs: Vec<usize> = sessions.iter().flat_map(|s| s.regs()).collect();
        let recs: Vec<usize> = sessions.iter().flat_map(|s| s.records()).collect();
        let imgs: Vec<usize> = sessions.iter().flat_map(|s| s.images()).collect();
        for (what, v) in [("reg", &regs), ("record", &recs), ("image", &imgs)] {
            let set: HashSet<usize> = v.iter().copied().collect();
            assert_eq!(set.len(), v.len(), "a {what} twice: {v:?}");
        }
        assert!(regs.iter().all(|&r| r < study.registrations.len()));
    }

    #[test]
    fn an_ordinary_day_is_one_session_of_two_steps() {
        let s = ordinary_day();
        let sessions = sessions(&s);
        assert_eq!(sessions.len(), 1);
        let d = &sessions[0];
        assert_eq!((d.date.as_str(), d.number, d.of), ("20240917", 1, 1));
        assert_eq!(d.series, vec![1], "the daily CT, not the planning CT");
        assert_eq!(d.image_regs, vec![0]);
        assert_eq!(d.ct_regs, vec![1]);
        assert_eq!(d.steps.len(), 2);
        let lpo = &d.steps[0];
        assert_eq!(lpo.name, "LPO");
        assert_eq!(lpo.treatment_reg, Some(2));
        assert_eq!(
            lpo.setup
                .iter()
                .map(|p| p.images.clone())
                .collect::<Vec<_>>(),
            vec![vec![0, 1], vec![2, 3]]
        );
        assert_eq!(lpo.records, vec![0]);
        assert_eq!(lpo.verification.len(), 1);
        assert_eq!(lpo.verification[0].images, vec![4, 5]);
        assert!(lpo.all_normal);
        let pa = &d.steps[1];
        assert_eq!(pa.name, "PA");
        assert_eq!(pa.treatment_reg, Some(3));
        assert_eq!(pa.setup[0].images, vec![6, 7]);
        assert_eq!(pa.records, vec![1]);
        assert_eq!(pa.verification[0].images, vec![8, 9]);
        assert!(d.loose_pairs.is_empty() && d.loose_records.is_empty());
        each_once(&s, &sessions);
        assert_eq!(sessions[0].regs().count(), 4);
        assert_eq!(sessions[0].images().count(), 10);
    }

    #[test]
    fn a_pair_one_second_apart_is_still_a_pair_and_three_seconds_is_not() {
        let mut s = course();
        pair(&mut s, 212, 3, ["20240917105145", "20240917105146"], "");
        pair(&mut s, 212, 7, ["20240917105145", "20240917105148"], "");
        let p = pairs(&s.planar_images);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0].images, vec![0, 1]);
        assert_eq!(p.iter().filter(|p| p.images.len() == 1).count(), 2);
        // A DRR (no acquisition time) and an additional object never pair.
        s.planar_images.push(PlanarImage::default());
        s.planar_images.push(PlanarImage {
            kind: PlanarKind::Additional,
            acquired: "20240917105145".into(),
            ..Default::default()
        });
        assert_eq!(pairs(&s.planar_images).len(), 3);
    }

    #[test]
    fn three_setup_iterations_sharing_one_review_go_to_one_step_in_order() {
        let mut s = course();
        s.registrations = vec![reg("20240917105403", &[PLAN])];
        s.treat_records = vec![record("20240917105418", "LPO", "NORMAL")];
        // Listed out of order: the sessions sort them.
        pair(
            &mut s,
            212,
            5,
            ["20240917105347", "20240917105347"],
            "20240917105402",
        );
        pair(
            &mut s,
            212,
            3,
            ["20240917105145", "20240917105145"],
            "20240917105402",
        );
        pair(
            &mut s,
            212,
            4,
            ["20240917105250", "20240917105250"],
            "20240917105402",
        );
        let d = &sessions(&s)[0];
        let firsts: Vec<usize> = d.steps[0].setup.iter().map(|p| p.images[0]).collect();
        assert_eq!(firsts, vec![2, 4, 0], "acquisition order, the final last");
    }

    #[test]
    fn an_interrupted_delivery_keeps_both_records_on_one_step() {
        let mut s = course();
        s.registrations = vec![reg("20240926101430", &[PLAN])];
        s.treat_records = vec![
            record("20240926101445", "LPO", "MACHINE"),
            record("20240926102010", "LPO", "NORMAL"),
        ];
        let d = &sessions(&s)[0];
        assert_eq!(d.steps.len(), 1);
        assert_eq!(d.steps[0].records, vec![0, 1]);
        assert!(!d.steps[0].all_normal);
        assert_eq!(record_status(&s.treat_records[0]), "MACHINE");
        assert_eq!(record_status(&s.treat_records[1]), "NORMAL");
    }

    #[test]
    fn a_records_only_day_is_a_session_of_loose_records() {
        let mut s = ordinary_day();
        s.treat_records.extend([
            record("20240912100000", "LPO", "NORMAL"),
            record("20240912100500", "PA", "MACHINE"),
            record("20240912100900", "PA", "NORMAL"),
        ]);
        let all = sessions(&s);
        assert_eq!(all.len(), 2);
        let first = &all[0];
        assert_eq!(
            (first.date.as_str(), first.number, first.of),
            ("20240912", 1, 2)
        );
        assert!(first.steps.is_empty());
        assert_eq!(first.loose_records, vec![2, 3, 4]);
        assert_eq!(all[1].number, 2);
        each_once(&s, &all);
    }

    #[test]
    fn an_imaging_registration_with_no_reference_or_an_unknown_one_is_the_days() {
        let mut s = ordinary_day();
        s.registrations.push(reg("20240917110500", &[]));
        s.registrations.push(reg("20240917110600", &["9.9.404"]));
        let d = &sessions(&s)[0];
        assert_eq!(d.image_regs, vec![0, 4, 5]);
        assert_eq!(reg_kind(&s.registrations[5], &s), RegKind::Imaging);
        assert_eq!(reg_kind(&s.registrations[1], &s), RegKind::CtToCt);
        assert_eq!(reg_kind(&s.registrations[2], &s), RegKind::Treatment);
        // The plan as the registration's own frame of reference names it too.
        let mut by_for = reg("20240917110700", &[]);
        by_for.frame_of_reference_uid = PLAN.into();
        assert_eq!(reg_kind(&by_for, &s), RegKind::Treatment);
    }

    #[test]
    fn a_second_block_gives_two_steps_of_the_same_beam() {
        let mut s = course();
        s.registrations = vec![
            reg("20240926092800", &[PLAN]),
            reg("20240926093500", &[DAILY_CT]),
            reg("20240926101420", &[PLAN]),
        ];
        s.treat_records = vec![
            record("20240926092830", "LPO", "MACHINE"),
            record("20240926101445", "LPO", "NORMAL"),
        ];
        pair(
            &mut s,
            212,
            3,
            ["20240926092745", "20240926092745"],
            "20240926092758",
        );
        pair(
            &mut s,
            212,
            3,
            ["20240926101413", "20240926101413"],
            "20240926101417",
        );
        let d = &sessions(&s)[0];
        assert_eq!(d.steps.len(), 2);
        assert!(d
            .steps
            .iter()
            .all(|st| st.name == "LPO" && st.setup.len() == 1));
        assert_eq!(d.steps[0].records, vec![0]);
        assert_eq!(d.steps[1].records, vec![1]);
        assert_eq!(d.image_regs, vec![1]);
        each_once(&s, std::slice::from_ref(d));
    }

    #[test]
    fn a_dose_report_goes_to_the_pair_it_was_made_with_else_to_its_day() {
        let mut s = ordinary_day();
        let report = |when: &str| crate::report::Report {
            kind: crate::report::ReportKind::DoseReport,
            datetime: when.into(),
            ..Default::default()
        };
        s.reports = vec![
            report("20240917105146"), // the first LPO setup pair, 1 s on
            report("20240917105758"), // the LPO verification pair
            report("20240917120000"), // nothing taken then: the day's
            report("20240916105146"), // another day: no session's
        ];
        let sessions = sessions(&s);
        let day = &sessions[0];
        let lpo = &day.steps[0];
        assert_eq!(lpo.setup[0].reports, vec![0]);
        assert!(lpo.setup[1].reports.is_empty());
        assert_eq!(lpo.verification[0].reports, vec![1]);
        assert_eq!(day.loose_reports, vec![2]);
        let all: Vec<usize> = day.reports().collect();
        assert_eq!(all, vec![0, 1, 2]);
    }

    #[test]
    fn a_day_of_imaging_registrations_only_is_no_session() {
        let mut s = course();
        s.registrations = vec![
            reg("20240903120217", &[]),
            reg("20240903120504", &[PLAN_CT]),
        ];
        assert!(sessions(&s).is_empty());
        // Records with no treatment registration anywhere: no course.
        s.treat_records = vec![record("20240912100000", "LPO", "NORMAL")];
        assert!(sessions(&s).is_empty());
    }

    #[test]
    fn a_plan_an_object_names_must_be_the_registrations() {
        let mut s = ordinary_day();
        s.treat_records[1].plan_uid = "9.9.other".into();
        let d = &sessions(&s)[0];
        // PA's record names another plan: no step of this one takes it.
        assert!(d.steps[1].records.is_empty());
        assert_eq!(d.loose_records, vec![1]);
        assert_eq!(d.steps[1].name, "Beam");
    }

    #[test]
    fn a_step_without_records_is_named_after_the_beam_its_images_reference() {
        let mut s = course();
        s.registrations = vec![reg("20240917105403", &[PLAN])];
        pair(
            &mut s,
            212,
            3,
            ["20240917105145", "20240917105145"],
            "20240917105402",
        );
        s.planar_images[0].referenced_beam = Some(3);
        assert_eq!(
            sessions(&s).len(),
            1,
            "a treatment registration makes the day"
        );
        assert_eq!(sessions(&s)[0].steps[0].name, "PA");
    }

    #[test]
    fn the_correction_is_the_translation_and_the_angles_of_the_matrix() {
        let (a, t) = (10f64.to_radians(), [-0.8, 4.9, -4.4]);
        let (c, s) = (a.cos(), a.sin());
        // A rotation of 10° about z, then the translation.
        #[rustfmt::skip]
        let m = [
            c,  -s,  0.0, t[0],
            s,   c,  0.0, t[1],
            0.0, 0.0, 1.0, t[2],
            0.0, 0.0, 0.0, 1.0,
        ];
        let mut r = reg("20240917105403", &[PLAN]);
        r.items.push(RegMatrixItem {
            matrix: m,
            ..Default::default()
        });
        let k = correction(&r).unwrap();
        for (got, want) in k.iter().zip([t[0], t[1], t[2], 0.0, 0.0, 10.0]) {
            assert!((got - want).abs() < 1e-9, "{k:?}");
        }
        assert_eq!(
            correction_text(&k),
            "Δ -0.8, 4.9, -4.4 mm · 0.00, 0.00, 10.00°"
        );
        assert_eq!(correction(&reg("20240917105403", &[])), Some([0.0; 6]));
        assert_eq!(correction(&SpatialReg::default()), None);
    }

    #[test]
    fn stamps_count_seconds_across_midnight_and_months() {
        let a = seconds("20240930235959").unwrap();
        assert_eq!(seconds("20241001000000").unwrap() - a, 1);
        assert_eq!(seconds("19700101000000"), Some(0));
        assert_eq!(seconds("2024093023595"), None);
        assert_eq!(seconds("20241301000000"), None);
        assert_eq!(clock("20240917105403"), "10:54:03");
        assert_eq!(date_label("20240917"), "2024-09-17");
    }
}
