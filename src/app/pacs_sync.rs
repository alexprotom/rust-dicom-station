//! *Save changes to the PACS*: the data tree's way to write back to this
//! station's PACS what was changed here - structures edited since they were
//! loaded or last saved, and the structure sets and segmentation series made
//! here.
//!
//! **The name.** *Save changes*, not *Sync*: it goes one way, from the
//! workspace into the PACS, and never brings anything back; and it writes
//! only what changed. *Sync* would promise both directions. It says *to the
//! PACS* as *File ▶ Save to the PACS* does, because it is that window.
//!
//! **One way to save.** The entry opens [`super::pacs_save`]'s window,
//! scoped to the row it was chosen on ([`SaveScope`]: a study, an image
//! series, one structure set or segmentation series), with the new and the
//! changed objects of that scope ticked and nothing else; the window looks
//! up the PACS on a job, files each object beside the series it was drawn on
//! ([`crate::archive::save::save_derived`], under the path limit, PHI kept
//! out of the file names), says what was saved and what failed, and keeps
//! the failures ticked. *Show the whole workspace* there lifts the scope.
//! The save path writes structure sets and segmentation series; the other
//! objects of a study (doses, plans, registrations, planar images) are what
//! the PACS was given and are not edited here.
//!
//! **When it shows.** Only on a study the PACS holds: [`Studies`], what the
//! catalog says of the workspace's studies, is looked up on a job the first
//! time a menu asks, and again when it is older than [`FRESH`] or the
//! workspace's study was replaced; while it is looked up the entry is there,
//! greyed, saying so. Greyed too when nothing in the scope is new or
//! changed ([`pending_of`]: the edit fingerprints of
//! [`super::slot::StudySlot::saved_marks`] against the objects as they are;
//! cached, since a large structure set takes milliseconds to fingerprint and
//! a menu is drawn every frame it is open).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use crate::dicom_export::{self, DerivedKind};

use super::pacs_save::{mark_of_seg, mark_of_set};
use super::*;

/// How long what the PACS was found to hold is taken as it is.
const FRESH: Duration = Duration::from_secs(30);

/// The objects of one workspace *Save to the PACS* is opened on, and what
/// they are, for the window's heading.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SaveScope {
    /// "study 2026-01-27", "series CT_1", "TotalSeg".
    pub label: String,
    pub objects: Vec<(DerivedKind, usize)>,
}

impl SaveScope {
    pub(super) fn holds(&self, kind: DerivedKind, index: usize) -> bool {
        self.objects.contains(&(kind, index))
    }
}

/// What the PACS was found to hold of one workspace's studies.
pub(super) struct Studies {
    /// The Study Instance UIDs asked about.
    asked: Vec<String>,
    /// Those the PACS holds.
    held: HashSet<String>,
    /// Whether the PACS has a catalog at all.
    catalog: bool,
    at: Instant,
}

pub(super) type StudiesDone = (usize, Studies);

/// The entry's state, read through `&self` while a menu is drawn.
#[derive(Default)]
pub(super) struct SyncState {
    studies: [Option<Studies>; MAX_WORKSPACES],
    /// Workspaces a menu asked about while nothing fresh was known.
    wanted: [bool; MAX_WORKSPACES],
    /// Studies a menu asked about that [`study_uids`] does not list (a row
    /// filed by an object it does not look at): asked about next time, so a
    /// menu never waits on a lookup that leaves its study out.
    extra: [Vec<String>; MAX_WORKSPACES],
    /// Per workspace: the objects new or changed, and what they were found
    /// for ([`ViewerApp::pending_key`]).
    pending: [Option<(u64, Objects)>; MAX_WORKSPACES],
}

/// Objects of a workspace, by kind and index.
type Objects = Vec<(DerivedKind, usize)>;

/// Whether the PACS holds a study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    Yes,
    No,
    /// Not looked up yet.
    Unknown,
}

/// The structure sets and segmentation series of `study` that are new or
/// changed against `marks` ([`super::slot::StudySlot::saved_marks`]): made
/// here or brought from elsewhere (no fingerprint), or edited since they
/// were loaded or saved (another one). Empty ones have nothing to save.
pub(super) fn pending_of(
    study: &LoadedStudy,
    marks: &HashMap<(DerivedKind, String), u64>,
) -> Vec<(DerivedKind, usize)> {
    let differs = |kind: DerivedKind, sop: &str, now: &dyn Fn() -> u64| {
        sop.is_empty()
            || marks
                .get(&(kind, sop.to_string()))
                .is_none_or(|then| *then != now())
    };
    let mut out = Vec::new();
    for (i, ss) in study.structure_sets.iter().enumerate() {
        if !ss.rois.is_empty()
            && differs(DerivedKind::StructureSet, &ss.sop_instance_uid, &|| {
                mark_of_set(ss)
            })
        {
            out.push((DerivedKind::StructureSet, i));
        }
    }
    for (i, ser) in study.seg_series.iter().enumerate() {
        if ser.segs.iter().any(|s| s.count > 0)
            && differs(DerivedKind::Segmentation, &ser.sop_instance_uid, &|| {
                mark_of_seg(ser)
            })
        {
            out.push((DerivedKind::Segmentation, i));
        }
    }
    out
}

/// Every study a workspace's tree can show a row for.
fn study_uids(study: &LoadedStudy) -> Vec<String> {
    let mut out: Vec<String> = study.series.iter().map(|s| s.study_uid.clone()).collect();
    for s in &study.structure_sets {
        out.push(dicom_export::derived_study_uid(
            study,
            &s.study_uid,
            &s.referenced_series_uid,
        ));
    }
    for s in &study.seg_series {
        out.push(dicom_export::derived_study_uid(
            study,
            &s.study_uid,
            &s.referenced_series_uid,
        ));
    }
    out.extend(study.doses.iter().map(|d| d.study_uid.clone()));
    out.extend(study.plans.iter().map(|p| p.study_uid.clone()));
    out.retain(|u| !u.is_empty());
    out.sort();
    out.dedup();
    out
}

impl ViewerApp {
    /// Whether this station's PACS holds study `uid` of workspace `slot`;
    /// asks for a lookup when what is known is stale or says nothing of it
    /// (the answer known so far stands meanwhile), and asks for a frame to
    /// start it in: a menu left open under a still pointer draws no other.
    fn pacs_holds(&self, ctx: &egui::Context, slot: usize, uid: &str) -> Held {
        if uid.is_empty() {
            return Held::No;
        }
        let mut st = self.pacs_sync.lock().unwrap_or_else(|e| e.into_inner());
        let known = st.studies[slot].as_ref().and_then(|s| {
            let fresh = s.at.elapsed() < FRESH;
            s.asked.iter().any(|a| a == uid).then(|| {
                let held = if s.catalog && s.held.contains(uid) {
                    Held::Yes
                } else {
                    Held::No
                };
                (held, fresh)
            })
        });
        match known {
            Some((held, true)) => held,
            other => {
                if other.is_none() && !st.extra[slot].iter().any(|u| u == uid) {
                    st.extra[slot].push(uid.to_string());
                }
                st.wanted[slot] = true;
                ctx.request_repaint();
                other.map_or(Held::Unknown, |(held, _)| held)
            }
        }
    }

    /// What the new-or-changed list of workspace `slot` depends on: every
    /// structure edit (`structures_gen`), the sets' and series' names,
    /// colours and sizes, the segments' generations, and the fingerprints
    /// kept for them. A colour picked in the tree moves no counter, so the
    /// cheap parts are hashed outright.
    fn pending_key(&self, slot: usize) -> u64 {
        let s = &self.slots[slot];
        let Some(study) = s.study.as_ref() else {
            return 0;
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.structures_gen(slot).hash(&mut h);
        for ss in &study.structure_sets {
            (&ss.label, &ss.sop_instance_uid, ss.rois.len()).hash(&mut h);
            for r in &ss.rois {
                (&r.name, r.color, r.number, &r.roi_type, r.contours.len()).hash(&mut h);
            }
        }
        for ser in &study.seg_series {
            (&ser.label, &ser.sop_instance_uid, ser.segs.len()).hash(&mut h);
            for g in &ser.segs {
                (&g.name, g.color, g.gen, g.count).hash(&mut h);
            }
        }
        // In any order: a map's.
        let marks = s.saved_marks.iter().fold(0u64, |acc, (k, v)| {
            let mut e = std::collections::hash_map::DefaultHasher::new();
            (k, v).hash(&mut e);
            acc ^ e.finish()
        });
        marks.hash(&mut h);
        h.finish() | 1
    }

    /// The new or changed objects of workspace `slot` ([`pending_of`]),
    /// from the cache while nothing they depend on moved.
    fn pending_objects(&self, slot: usize) -> Vec<(DerivedKind, usize)> {
        let key = self.pending_key(slot);
        {
            let st = self.pacs_sync.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((k, list)) = &st.pending[slot] {
                if *k == key {
                    return list.clone();
                }
            }
        }
        let s = &self.slots[slot];
        let list = s
            .study
            .as_ref()
            .map(|st| pending_of(st, &s.saved_marks))
            .unwrap_or_default();
        let mut st = self.pacs_sync.lock().unwrap_or_else(|e| e.into_inner());
        st.pending[slot] = Some((key, list.clone()));
        list
    }

    /// The *Save changes to the PACS* entry of a data-tree menu: for the
    /// objects of `scope`, under study `study_uid` of workspace `slot`.
    /// Nothing at all when the PACS does not hold the study; greyed while it
    /// is looked up and when nothing in the scope is new or changed. The
    /// scope to open the window on when chosen.
    pub(super) fn pacs_sync_entry(
        &self,
        ui: &mut egui::Ui,
        slot: usize,
        study_uid: &str,
        scope: SaveScope,
    ) -> Option<SaveScope> {
        const LABEL: &str = "📤 Save changes to the PACS";
        match self.pacs_holds(ui.ctx(), slot, study_uid) {
            Held::No => return None,
            Held::Unknown => {
                enabled_tip_button(ui, false, LABEL, "Looking up this station's PACS");
                return None;
            }
            Held::Yes => {}
        }
        let n = self
            .pending_objects(slot)
            .into_iter()
            .filter(|(k, i)| scope.holds(*k, *i))
            .count();
        if n == 0 {
            enabled_tip_button(
                ui,
                false,
                LABEL,
                "Nothing new or changed here since it was loaded or saved",
            );
            return None;
        }
        if enabled_tip_button(
            ui,
            true,
            format!("{LABEL} ({n})"),
            "Open Save to the PACS with what was made or changed here ticked",
        ) {
            ui.close();
            return Some(scope);
        }
        None
    }

    /// The scope of a study row: the structure sets and segmentation series
    /// the tree files under it.
    pub(super) fn study_scope(title: &str, structs: &[usize], segs: &[usize]) -> SaveScope {
        SaveScope {
            label: format!("study {title}"),
            objects: structs
                .iter()
                .map(|&i| (DerivedKind::StructureSet, i))
                .chain(segs.iter().map(|&i| (DerivedKind::Segmentation, i)))
                .collect(),
        }
    }

    /// The scope of an image series row: what was drawn on that series.
    pub(super) fn series_scope(study: &LoadedStudy, index: usize) -> Option<SaveScope> {
        let se = study.series.get(index)?;
        let on = |r: &str| r == se.uid;
        Some(SaveScope {
            label: format!(
                "series {}",
                if se.description.is_empty() {
                    "without a description"
                } else {
                    &se.description
                }
            ),
            objects: study
                .structure_sets
                .iter()
                .enumerate()
                .filter(|(_, s)| on(&s.referenced_series_uid))
                .map(|(i, _)| (DerivedKind::StructureSet, i))
                .chain(
                    study
                        .seg_series
                        .iter()
                        .enumerate()
                        .filter(|(_, s)| on(&s.referenced_series_uid))
                        .map(|(i, _)| (DerivedKind::Segmentation, i)),
                )
                .collect(),
        })
    }

    /// The study a set row is filed under, and the scope of that one set.
    pub(super) fn set_sync_scope(&self, here: SetRef) -> Option<(String, SaveScope)> {
        let study = self.slots[here.slot].study.as_ref()?;
        let (own, referenced, label, kind) = match here.kind {
            SetKind::Structures => {
                let s = study.structure_sets.get(here.idx)?;
                let label = if s.label.is_empty() {
                    &s.file_name
                } else {
                    &s.label
                };
                (
                    &s.study_uid,
                    &s.referenced_series_uid,
                    format!("structure set {label}"),
                    DerivedKind::StructureSet,
                )
            }
            SetKind::Segmentations => {
                let s = study.seg_series.get(here.idx)?;
                (
                    &s.study_uid,
                    &s.referenced_series_uid,
                    format!("segmentation series {}", s.label),
                    DerivedKind::Segmentation,
                )
            }
        };
        Some((
            dicom_export::derived_study_uid(study, own, referenced),
            SaveScope {
                label,
                objects: vec![(kind, here.idx)],
            },
        ))
    }

    /// Ask for the studies of the workspaces a menu wanted, one lookup at a
    /// time: [`crate::archive::Archive::save_targets`] on a job, never on the
    /// interface thread.
    pub(super) fn start_wanted_pacs_studies(&mut self) {
        if self.pacs_sync_job.is_some() {
            return;
        }
        let (slot, extra) = {
            let mut st = self.pacs_sync.lock().unwrap_or_else(|e| e.into_inner());
            let Some(slot) = (0..MAX_WORKSPACES).find(|&s| st.wanted[s]) else {
                return;
            };
            st.wanted[slot] = false;
            (slot, st.extra[slot].clone())
        };
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let mut asked = study_uids(study);
        asked.extend(extra);
        asked.sort();
        asked.dedup();
        let keys: Vec<(String, String)> =
            asked.iter().map(|u| (u.clone(), String::new())).collect();
        let archive = self.station_pacs();
        let progress = Arc::new(Progress::default());
        progress.set("Reading the PACS");
        self.pacs_sync_job = Some(Job::spawn(progress, move |_| {
            let t = archive.save_targets(&keys, &[]);
            let held = t
                .homes
                .iter()
                .filter(|(_, home)| home.is_some())
                .map(|((study, _), _)| study.clone())
                .collect();
            (
                slot,
                Studies {
                    asked,
                    held,
                    catalog: t.catalog,
                    at: Instant::now(),
                },
            )
        }));
    }

    /// A lookup landed. Dropped when the workspace's study changed
    /// meanwhile: the next menu asks again.
    pub(super) fn on_pacs_studies(&mut self, slot: usize, studies: Studies) {
        let now = self.slots[slot]
            .study
            .as_ref()
            .map(study_uids)
            .unwrap_or_default();
        let mut st = self.pacs_sync.lock().unwrap_or_else(|e| e.into_inner());
        if now.iter().all(|u| studies.asked.contains(u)) {
            st.studies[slot] = Some(studies);
        }
    }

    /// Workspace `slot` holds another study: nothing known of the last one
    /// stands.
    pub(super) fn forget_pacs_studies(&mut self, slot: usize) {
        let st = self.pacs_sync.get_mut().unwrap_or_else(|e| e.into_inner());
        st.studies[slot] = None;
        st.pending[slot] = None;
        st.extra[slot].clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dicomseg::SegSeries;
    use crate::rtstruct::{Roi, StructureSet};

    fn set(sop: &str, rois: usize) -> StructureSet {
        StructureSet {
            label: format!("set {sop}"),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: sop.into(),
            series_instance_uid: String::new(),
            study_uid: "S".into(),
            referenced_series_uid: String::new(),
            file_name: String::new(),
            locked: false,
            rois: (0..rois)
                .map(|n| Roi {
                    number: n as i32 + 1,
                    name: format!("R{n}"),
                    color: [1, 2, 3],
                    roi_type: String::new(),
                    description: String::new(),
                    contours: Vec::new(),
                })
                .collect(),
        }
    }

    /// Loaded and untouched: nothing. Edited, made here, brought from
    /// elsewhere: pending. Empty: nothing to save.
    #[test]
    fn what_was_made_or_changed_here_is_pending() {
        let grid = crate::volume::Grid {
            dims: [1, 1, 1],
            spacing: [1.0, 1.0, 1.0],
            origin: crate::geometry::Vec3::ZERO,
            row_dir: crate::geometry::Vec3::new(1.0, 0.0, 0.0),
            col_dir: crate::geometry::Vec3::new(0.0, 1.0, 0.0),
            normal: crate::geometry::Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        };
        let mut seg = SegSeries::new("Painted".into(), grid.clone(), String::new(), "S".into());
        seg.segs.push(crate::segmentation::Segmentation::from_mask(
            "Ball".into(),
            [1, 2, 3],
            [1, 1, 1],
            vec![1],
        ));
        let mut blank = SegSeries::new("Blank".into(), grid, String::new(), "S".into());
        blank
            .segs
            .push(crate::segmentation::Segmentation::from_mask(
                "None".into(),
                [1, 2, 3],
                [1, 1, 1],
                vec![0],
            ));
        let mut study = LoadedStudy {
            structure_sets: vec![
                set("1.1", 2), // loaded, untouched
                set("1.2", 2), // loaded, then edited
                set("", 1),    // made here
                set("9.9", 1), // no fingerprint: from elsewhere
                set("", 0),    // made here, empty
            ],
            seg_series: vec![seg, blank],
            ..Default::default()
        };
        let mut marks = HashMap::new();
        for ss in &study.structure_sets[..2] {
            marks.insert(
                (DerivedKind::StructureSet, ss.sop_instance_uid.clone()),
                mark_of_set(ss),
            );
        }
        study.structure_sets[1].rois[0].name = "renamed".into();
        assert_eq!(
            pending_of(&study, &marks),
            vec![
                (DerivedKind::StructureSet, 1),
                (DerivedKind::StructureSet, 2),
                (DerivedKind::StructureSet, 3),
                (DerivedKind::Segmentation, 0),
            ]
        );
    }

    #[test]
    fn a_series_row_scopes_what_was_drawn_on_it() {
        let mut a = set("1", 1);
        a.referenced_series_uid = "CT1".into();
        let mut b = set("2", 1);
        b.referenced_series_uid = "CT2".into();
        let study = LoadedStudy {
            series: vec![
                crate::loader::SeriesInfo {
                    uid: "CT1".into(),
                    description: "CT_1".into(),
                    ..Default::default()
                },
                crate::loader::SeriesInfo {
                    uid: "CT2".into(),
                    ..Default::default()
                },
            ],
            structure_sets: vec![a, b],
            ..Default::default()
        };
        let s = ViewerApp::series_scope(&study, 0).unwrap();
        assert_eq!(s.label, "series CT_1");
        assert_eq!(s.objects, vec![(DerivedKind::StructureSet, 0)]);
        assert!(s.holds(DerivedKind::StructureSet, 0));
        assert!(!s.holds(DerivedKind::StructureSet, 1));
        assert_eq!(
            ViewerApp::series_scope(&study, 1).unwrap().label,
            "series without a description"
        );
    }
}
