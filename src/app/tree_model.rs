//! The data tree as a model: the patient > study > modality > series layout
//! of a workspace (made once per change and cached), and the arrow keys'
//! walk over its levels.

use super::*;

/// The rows at `t`'s own level of the tree, in the order the tree draws
/// them - the rule behind the arrow keys, kept apart from the app so it can
/// be checked on a study alone.
fn tree_level_of(study: &LoadedStudy, slot: usize, t: &RenameTarget) -> Vec<RenameTarget> {
    let n_of = |n: usize| (0..n).collect::<Vec<_>>();
    match t {
        RenameTarget::Patient { .. } => tree_layout(study)
            .into_iter()
            .map(|p| RenameTarget::Patient { slot, key: p.key })
            .collect(),
        RenameTarget::Study { .. } => tree_layout(study)
            .into_iter()
            .flat_map(|p| p.studies)
            .map(|s| RenameTarget::Study { slot, uid: s.uid })
            .collect(),
        RenameTarget::Series { idx, .. } => {
            let series = |list: &[usize]| -> Vec<RenameTarget> {
                list.iter()
                    .map(|i| RenameTarget::Series { slot, idx: *i })
                    .collect()
            };
            for p in tree_layout(study) {
                for st in p.studies {
                    // A 4D group first: its phases are the level, and
                    // they are also the ones the tree does not repeat
                    // under the modality node.
                    for gi in &st.fourd {
                        let Some(g) = study.fourd_groups.get(*gi) else {
                            continue;
                        };
                        let members: Vec<usize> =
                            g.resolve(&study.series).into_iter().flatten().collect();
                        if members.contains(idx) {
                            return series(&members);
                        }
                    }
                    for (_, list) in &st.modalities {
                        if list.contains(idx) {
                            return series(list);
                        }
                    }
                }
            }
            Vec::new()
        }
        RenameTarget::FourD { .. } => n_of(study.fourd_groups.len())
            .into_iter()
            .map(|idx| RenameTarget::FourD { slot, idx })
            .collect(),
        RenameTarget::Set(r) => {
            let n = match r.kind {
                SetKind::Structures => study.structure_sets.len(),
                SetKind::Segmentations => study.seg_series.len(),
            };
            n_of(n)
                .into_iter()
                .map(|idx| {
                    RenameTarget::Set(SetRef {
                        slot,
                        kind: r.kind,
                        idx,
                    })
                })
                .collect()
        }
        RenameTarget::Item { set, .. } => {
            let n = match set.kind {
                SetKind::Structures => study
                    .structure_sets
                    .get(set.idx)
                    .map(|s| s.rois.len())
                    .unwrap_or(0),
                SetKind::Segmentations => study
                    .seg_series
                    .get(set.idx)
                    .map(|s| s.segs.len())
                    .unwrap_or(0),
            };
            n_of(n)
                .into_iter()
                .map(|idx| RenameTarget::Item { set: *set, idx })
                .collect()
        }
        RenameTarget::Dose { .. } => n_of(study.doses.len())
            .into_iter()
            .map(|idx| RenameTarget::Dose { slot, idx })
            .collect(),
        RenameTarget::Plan { .. } => n_of(study.plans.len())
            .into_iter()
            .map(|idx| RenameTarget::Plan { slot, idx })
            .collect(),
        // Planar images and reports sit under their study, the projections
        // and the additional objects in a list each: the level is the list
        // the row is in.
        RenameTarget::Planar { idx, .. } => tree_layout(study)
            .into_iter()
            .flat_map(|p| p.studies)
            .flat_map(|st| [st.planar, st.additional])
            .find(|list| list.contains(idx))
            .unwrap_or_default()
            .into_iter()
            .map(|idx| RenameTarget::Planar { slot, idx })
            .collect(),
        RenameTarget::Registration { .. } => n_of(study.registrations.len())
            .into_iter()
            .map(|idx| RenameTarget::Registration { slot, idx })
            .collect(),
        RenameTarget::Record { .. } => n_of(study.treat_records.len())
            .into_iter()
            .map(|idx| RenameTarget::Record { slot, idx })
            .collect(),
        RenameTarget::Report { idx, .. } => tree_layout(study)
            .into_iter()
            .flat_map(|p| p.studies)
            .map(|st| st.reports)
            .find(|list| list.contains(idx))
            .unwrap_or_default()
            .into_iter()
            .map(|idx| RenameTarget::Report { slot, idx })
            .collect(),
        RenameTarget::Setup { .. } => n_of(study.setups.len())
            .into_iter()
            .map(|idx| RenameTarget::Setup { slot, idx })
            .collect(),
    }
}

impl ViewerApp {
    // -- Arrow keys on the data tree ---------------------------------------

    /// The rows at the focused row's own level of the tree, in the order the
    /// tree draws them.
    ///
    /// "Its own level" is what the tree shows, not what the DICOM model
    /// says: an image series that belongs to a 4D group has the group's
    /// other phases beside it, because that is where the tree files it; one
    /// that does not has the other series of its study *and its modality*,
    /// which is the node it sits under. [`tree_layout`] is asked rather than
    /// the study directly, so the order the arrows follow is the order on
    /// screen even after a re-detect moves a series from one node to
    /// another.
    fn tree_level(&self, t: &RenameTarget) -> Vec<RenameTarget> {
        match self.slots[t.slot()].study.as_ref() {
            Some(study) => tree_level_of(study, t.slot(), t),
            None => Vec::new(),
        }
    }
    /// ↑ / ↓ on the data tree: move the focus one row up or down its own
    /// level, and show what it lands on.
    ///
    /// The ends do not wrap. Holding ↓ through the phases of a 4D group and
    /// stopping at the last one is the behaviour a slider has; coming back
    /// round to the first would make the count of presses meaningless.
    pub(super) fn tree_key_step(&mut self, delta: isize) {
        // A series switch is a load, and a load already in flight makes the
        // next one a no-op. Moving the focus anyway would walk it past rows
        // the views never showed, so the key does nothing until the volume
        // it last asked for is there.
        if self.loading.is_some() {
            return;
        }
        let Some(here) = self.tree_focus.clone() else {
            return;
        };
        let level = self.tree_level(&here);
        let Some(at) = level.iter().position(|x| *x == here) else {
            return;
        };
        let next = at as isize + delta;
        if next < 0 || next as usize >= level.len() {
            return;
        }
        let target = level[next as usize].clone();
        self.tree_focus = Some(target.clone());
        self.tree_show_focused(&target);
    }

    /// Display what a row points at, the way clicking it would.
    ///
    /// Only the rows that *have* something to display: stepping onto a
    /// patient or a study is a move through the tree, a planar image opens
    /// in its own pane rather than being selected, and stepping onto one
    /// structure of a set would otherwise toggle its visibility, which is
    /// what clicking it does and is not what an arrow key should do.
    fn tree_show_focused(&mut self, t: &RenameTarget) {
        match t {
            RenameTarget::Series { slot, idx } => {
                let (slot, idx) = (*slot, *idx);
                // Between the phases of the group on display the view is
                // kept, exactly as clicking a phase does: two phases are the
                // same patient a moment apart, and a crosshair that jumps
                // back to the middle slice hides the motion being looked at.
                match self
                    .fourd_phases(slot)
                    .and_then(|(_, idxs, _)| idxs.iter().position(|x| *x == idx))
                {
                    Some(phase) => self.goto_phase(slot, phase),
                    None => self.start_series_switch(slot, idx),
                }
            }
            RenameTarget::Set(r) => {
                let s = &mut self.slots[r.slot];
                match r.kind {
                    SetKind::Structures => {
                        s.active_structs = r.idx;
                        s.structs_shown = true;
                    }
                    SetKind::Segmentations => {
                        s.active_seg_series = r.idx;
                        s.active_seg = 0;
                        s.segs_shown = true;
                    }
                }
                self.settings_gen += 1;
            }
            RenameTarget::Dose { slot, idx } => {
                self.slots[*slot].active_dose = *idx;
                self.settings_gen += 1;
            }
            RenameTarget::Plan { slot, idx } => {
                self.slots[*slot].active_plan = *idx;
                self.settings_gen += 1;
            }
            _ => {}
        }
    }
}

/// One study node of the data tree: which of the workspace's objects belong
/// under it, as indices into the parallel arrays of [`LoadedStudy`].
#[derive(Default)]
pub(super) struct StudyNode {
    pub(super) uid: String,
    pub(super) title: String,
    /// Image series of this study, grouped by modality in first-seen order -
    /// the CT / MR / US level DICOM implies but does not store as a node.
    pub(super) modalities: Vec<(String, Vec<usize>)>,
    pub(super) structs: Vec<usize>,
    pub(super) segs: Vec<usize>,
    pub(super) doses: Vec<usize>,
    pub(super) plans: Vec<usize>,
    /// 4D groups of this study - indices into `LoadedStudy::fourd_groups`.
    pub(super) fourd: Vec<usize>,
    /// Planar images of the study: the projections (radiographs, RT
    /// images, DRRs) and the additional objects (secondary captures,
    /// pages), each a list of indices into `LoadedStudy::planar_images`.
    pub(super) planar: Vec<usize>,
    pub(super) additional: Vec<usize>,
    /// Its reports - indices into `LoadedStudy::reports`.
    pub(super) reports: Vec<usize>,
}

/// One patient node: the studies filed under them.
pub(super) struct PatientNode {
    pub(super) key: String,
    pub(super) title: String,
    pub(super) studies: Vec<StudyNode>,
}

/// Sort a workspace into the patient ▶ study ▶ category ▶ series tree.
///
/// Series carry the patient and study they belong to, so those two levels
/// fall straight out of them. The RT objects carry a StudyInstanceUID as
/// well, but not always a usable one - a set built in the application from a
/// series that had none, or a file written by a tool that left it blank. An
/// object whose study is not in the tree is therefore filed under the study
/// of the image series it references, and failing that under the first study
/// there is: a structure set that cannot be reached is worse than one shown
/// a level away from where its header claims it lives.
/// A study's line in the data tree: what the study is called, then its
/// StudyID, then its date in parentheses - each only when the header carries
/// it.
///
/// The word "study" is not in it. The node sits under a patient and above
/// that study's series, which is what says what it is; spending the line on
/// the word leaves less of it for the description, which is the part that
/// tells two studies of one patient apart. With neither a description nor a
/// StudyID there is nothing to say but which study it is, and `#n` says that
/// without the word.
pub(super) fn study_title(description: &str, study_id: &str, date: &str, n: usize) -> String {
    let mut title = String::new();
    if !description.is_empty() {
        title.push_str(description);
    }
    if !study_id.is_empty() {
        if !title.is_empty() {
            title.push_str(" · ");
        }
        title.push_str(study_id);
    }
    if title.is_empty() {
        title = format!("#{n}");
    }
    if !date.is_empty() {
        title.push_str(&format!(" ({})", dicom_date(date)));
    }
    title
}

/// `19980325` as `1998-03-25`; anything else as it stands, because a header
/// that does not hold a DICOM date is better shown than silently dropped.
fn dicom_date(da: &str) -> String {
    let da = da.trim();
    if da.len() == 8 && da.bytes().all(|b| b.is_ascii_digit()) {
        format!("{}-{}-{}", &da[..4], &da[4..6], &da[6..])
    } else {
        da.to_string()
    }
}

/// What [`tree_layout`] reads, cheaply: a layout kept under this key is the
/// layout of the study as long as the key holds. `settings_gen` moves with
/// every rename, removal and set action; the counts and the study's address
/// catch what arrives without it (a merge, a study swapped in).
fn tree_key(study: &LoadedStudy, settings_gen: u64) -> u64 {
    let mut h = mix(0x7EE5_7EE5, settings_gen);
    h = mix(h, study as *const LoadedStudy as u64);
    for n in [
        study.series.len(),
        study.structure_sets.len(),
        study.seg_series.len(),
        study.doses.len(),
        study.plans.len(),
        study.fourd_groups.len(),
        study.planar_images.len(),
        study.reports.len(),
    ] {
        h = mix(h, n as u64);
    }
    // A group edit does not move `settings_gen`, and one can reorder or
    // rename without changing a count: the groups themselves are part of
    // the key (a few dozen strings, not the series).
    use std::hash::{Hash, Hasher};
    let mut groups = std::collections::hash_map::DefaultHasher::new();
    for g in &study.fourd_groups {
        (&g.name, g.dissolved, g.custom).hash(&mut groups);
        for m in &g.members {
            (&m.series_uid, &m.label, m.percent.map(f32::to_bits)).hash(&mut groups);
            std::mem::discriminant(&m.role).hash(&mut groups);
        }
    }
    mix(h, groups.finish())
}

impl ViewerApp {
    /// [`tree_layout`] of a workspace, made once per change of the study
    /// rather than on every frame the tree is drawn: the titles, the 4D
    /// groups resolved against the series and the study index were all
    /// rebuilt per frame per workspace.
    pub(super) fn tree_layout_cached(&mut self, slot: usize) -> Option<Arc<Vec<PatientNode>>> {
        let study = self.slots[slot].study.as_ref()?;
        let key = tree_key(study, self.settings_gen);
        if let Some((k, layout)) = &self.tree_cache[slot] {
            if *k == key {
                return Some(layout.clone());
            }
        }
        let layout = Arc::new(tree_layout(study));
        self.tree_cache[slot] = Some((key, layout.clone()));
        Some(layout)
    }
}

pub(super) fn tree_layout(study: &LoadedStudy) -> Vec<PatientNode> {
    // Series filed under a 4D group render inside that node, not under
    // their modality - one series, one place in the tree.
    let mut grouped = vec![false; study.series.len()];
    for g in &study.fourd_groups {
        if g.dissolved {
            continue;
        }
        for r in g.resolve(&study.series).into_iter().flatten() {
            grouped[r] = true;
        }
    }
    // Patients and their studies, both in first-seen order.
    let mut patients: Vec<PatientNode> = Vec::new();
    for se in &study.series {
        let key = se.patient_key().to_string();
        if !patients.iter().any(|p| p.key == key) {
            let name = se.patient_name.replace('^', " ");
            let title = match (name.is_empty(), se.patient_id.is_empty()) {
                (true, true) => "Unknown patient".to_string(),
                (true, false) => format!("Patient {}", se.patient_id),
                (false, true) => name.clone(),
                (false, false) => format!("{} ({})", name, se.patient_id),
            };
            patients.push(PatientNode {
                key,
                title,
                studies: Vec::new(),
            });
        }
    }
    for (si, se) in study.series.iter().enumerate() {
        let p = patients
            .iter_mut()
            .find(|p| p.key == se.patient_key())
            .expect("every patient was collected above");
        let node = match p.studies.iter_mut().position(|s| s.uid == se.study_uid) {
            Some(i) => &mut p.studies[i],
            None => {
                let n = p.studies.len() + 1;
                let title = study_title(&se.study_description, &se.study_id, &se.study_date, n);
                p.studies.push(StudyNode {
                    uid: se.study_uid.clone(),
                    title,
                    ..Default::default()
                });
                p.studies.last_mut().expect("just pushed")
            }
        };
        if grouped[si] {
            continue;
        }
        let modality = if se.modality.is_empty() {
            "Other".to_string()
        } else {
            se.modality.clone()
        };
        match node.modalities.iter_mut().find(|(m, _)| *m == modality) {
            Some((_, v)) => v.push(si),
            None => node.modalities.push((modality, vec![si])),
        }
    }

    // File each 4D group under its study node (falling back to the study of
    // its first surviving series, then to the first study - same rule as
    // the RT objects below).
    for (gi, g) in study.fourd_groups.iter().enumerate() {
        if g.dissolved {
            continue;
        }
        let resolved = g.resolve(&study.series);
        let Some(first) = resolved.iter().flatten().next() else {
            continue; // nothing left of this group
        };
        let fallback = &study.series[*first].study_uid;
        let find = |uid: &str| -> Option<(usize, usize)> {
            patients.iter().enumerate().find_map(|(pi, p)| {
                p.studies
                    .iter()
                    .position(|st| st.uid == uid)
                    .map(|si| (pi, si))
            })
        };
        if let Some((pi, si)) = find(&g.study_uid).or_else(|| find(fallback)) {
            patients[pi].studies[si].fourd.push(gi);
        }
    }

    // Studies that no image series announced.
    //
    // A workspace does not have to contain images at all - a folder or a file
    // selection can hold nothing but a structure set, a plan, a dose grid or
    // a handful of RT images. Those objects carry their own Study Instance
    // UID, so a study node is made from it and filed under the patient the
    // workspace's own metadata names. Without this the tree would be empty and
    // the objects invisible, which is the one outcome worse than no images.
    {
        let known: std::collections::HashSet<String> = patients
            .iter()
            .flat_map(|p| p.studies.iter().map(|s| s.uid.clone()))
            .collect();
        let mut missing: Vec<String> = Vec::new();
        let mut orphans = false;
        let mut want = |uid: &str| {
            // A blank UID names no study, so it gets no node of its own: the
            // fallback below files those objects under the first study there
            // is, which is the long-standing rule.
            if uid.is_empty() {
                orphans = true;
            } else if !known.contains(uid) && !missing.iter().any(|u| u == uid) {
                missing.push(uid.to_string());
            }
        };
        for ss in &study.structure_sets {
            want(&ss.study_uid);
        }
        for sr in &study.seg_series {
            want(&sr.study_uid);
        }
        for d in &study.doses {
            want(&d.study_uid);
        }
        for pl in &study.plans {
            want(&pl.study_uid);
        }
        for img in &study.planar_images {
            want(&img.study_uid);
        }
        for r in &study.reports {
            want(&r.study_uid);
        }
        // …unless there is no first study either. Then one study node has to
        // exist for those objects to be reachable at all.
        if missing.is_empty() && patients.is_empty() && orphans {
            missing.push(String::new());
        }
        if !missing.is_empty() {
            let m = &study.meta;
            let name = m.patient_name.replace('^', " ");
            let key = if !m.patient_id.is_empty() {
                m.patient_id.clone()
            } else if !name.is_empty() {
                name.clone()
            } else {
                "?".to_string()
            };
            let title = match (name.is_empty(), m.patient_id.is_empty()) {
                (true, true) => "Unknown patient".to_string(),
                (true, false) => format!("Patient {}", m.patient_id),
                (false, true) => name.clone(),
                (false, false) => format!("{name} ({})", m.patient_id),
            };
            let pi = match patients.iter().position(|p| p.key == key) {
                Some(i) => i,
                None => {
                    patients.push(PatientNode {
                        key,
                        title,
                        studies: Vec::new(),
                    });
                    patients.len() - 1
                }
            };
            for uid in missing {
                let n = patients[pi].studies.len() + 1;
                let title = study_title(&m.study_description, &m.study_id, &m.study_date, n);
                patients[pi].studies.push(StudyNode {
                    uid,
                    title,
                    ..Default::default()
                });
            }
        }
    }

    // Where an RT object goes, by the rule in the doc comment above.
    let series_study = |uid: &str| -> Option<String> {
        study
            .series
            .iter()
            .find(|se| se.uid == uid)
            .map(|se| se.study_uid.clone())
    };
    // The (patient, study) address of every study node, so the placement
    // below can look one up without borrowing `patients` while it fills them.
    let index: Vec<(String, usize, usize)> = patients
        .iter()
        .enumerate()
        .flat_map(|(pi, p)| {
            p.studies
                .iter()
                .enumerate()
                .map(move |(si, s)| (s.uid.clone(), pi, si))
        })
        .collect();
    let place = |own: &str, referenced: &str| -> Option<(usize, usize)> {
        let find = |uid: &str| {
            index
                .iter()
                .find(|(u, _, _)| u == uid)
                .map(|(_, pi, si)| (*pi, *si))
        };
        find(own)
            .or_else(|| series_study(referenced).and_then(|u| find(&u)))
            .or_else(|| index.first().map(|(_, pi, si)| (*pi, *si)))
    };
    for (i, ss) in study.structure_sets.iter().enumerate() {
        if let Some((pi, si)) = place(&ss.study_uid, &ss.referenced_series_uid) {
            patients[pi].studies[si].structs.push(i);
        }
    }
    for (i, sr) in study.seg_series.iter().enumerate() {
        if let Some((pi, si)) = place(&sr.study_uid, &sr.referenced_series_uid) {
            patients[pi].studies[si].segs.push(i);
        }
    }
    for (i, d) in study.doses.iter().enumerate() {
        if let Some((pi, si)) = place(&d.study_uid, "") {
            patients[pi].studies[si].doses.push(i);
        }
    }
    for (i, p) in study.plans.iter().enumerate() {
        if let Some((pi, si)) = place(&p.study_uid, "") {
            patients[pi].studies[si].plans.push(i);
        }
    }
    // Planar images and reports carry their study too; a DRR is given the
    // study of the image it was made from when it is made.
    for (i, img) in study.planar_images.iter().enumerate() {
        if let Some((pi, si)) = place(&img.study_uid, "") {
            let node = &mut patients[pi].studies[si];
            match img.kind {
                crate::extras::PlanarKind::Projection => node.planar.push(i),
                crate::extras::PlanarKind::Additional => node.additional.push(i),
            }
        }
    }
    for (i, r) in study.reports.iter().enumerate() {
        if let Some((pi, si)) = place(&r.study_uid, "") {
            patients[pi].studies[si].reports.push(i);
        }
    }
    patients
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use crate::dicomseg::SegSeries;
    use crate::geometry::Vec3;
    use crate::rtstruct::StructureSet;

    #[test]
    fn a_study_line_is_its_name_its_id_and_its_date_and_never_the_word() {
        // Everything present.
        assert_eq!(
            study_title("4DFBCT+RTS", "1", "19980325", 1),
            "4DFBCT+RTS · 1 (1998-03-25)"
        );
        // The bundled data: a description and a StudyID, no date.
        assert_eq!(study_title("4DCBCT", "1", "", 2), "4DCBCT · 1");
        // Description alone, which is the common case.
        assert_eq!(
            study_title("Planning CT", "", "20240115", 1),
            "Planning CT (2024-01-15)"
        );
        assert_eq!(study_title("Planning CT", "", "", 1), "Planning CT");
        // No description: the StudyID carries the line on its own.
        assert_eq!(study_title("", "4711", "20240115", 3), "4711 (2024-01-15)");
        // Neither: which study it is, and still not the word "study".
        assert_eq!(study_title("", "", "", 2), "#2");
        assert_eq!(study_title("", "", "20240115", 2), "#2 (2024-01-15)");
        for t in [
            study_title("", "", "", 1),
            study_title("Planning CT", "1", "20240115", 1),
        ] {
            assert!(
                !t.to_ascii_lowercase().contains("study"),
                "the node is under a patient and above its series: {t}"
            );
        }
        // A header with something other than a DICOM date in it is shown as
        // it stands rather than sliced into nonsense.
        assert_eq!(study_title("CT", "", "1998", 1), "CT (1998)");
    }

    fn series(uid: &str, modality: &str, patient: &str, study: &str) -> loader::SeriesInfo {
        loader::SeriesInfo {
            uid: uid.into(),
            modality: modality.into(),
            description: format!("{uid} desc"),
            patient_id: patient.into(),
            patient_name: format!("{patient}^Name"),
            study_uid: study.into(),
            study_date: "20260827".into(),
            study_description: String::new(),
            study_id: String::new(),
            series_number: None,
            temporal_id: None,
            suv_bw: None,
            files: vec![std::path::PathBuf::from(format!("{uid}.dcm"))],
        }
    }

    fn structset(sop: &str, series_uid: &str, study: &str) -> StructureSet {
        StructureSet {
            label: sop.into(),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: sop.into(),
            series_instance_uid: String::new(),
            study_uid: study.into(),
            referenced_series_uid: series_uid.into(),
            file_name: String::new(),
            locked: false,
            rois: Vec::new(),
        }
    }

    fn study() -> LoadedStudy {
        let vol = Arc::new(Volume {
            data: vec![0],
            dims: [1, 1, 1],
            spacing: [1.0; 3],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        });
        LoadedStudy {
            meta: loader::PatientMeta::default(),
            series: vec![
                series("ct1", "CT", "P1", "st1"),
                series("ct2", "CT", "P1", "st1"),
                series("mr1", "MR", "P1", "st1"),
                series("ct3", "CT", "P1", "st2"),
                series("us1", "US", "P2", "st3"),
            ],
            active_series: 0,
            volume: vol.clone(),
            structure_sets: vec![
                structset("ss1", "ct1", "st1"),
                // No study of its own: it must follow the series it references.
                structset("ss2", "ct3", ""),
                // Neither: it must still be reachable, under the first study.
                structset("ss3", "", ""),
            ],
            seg_series: vec![SegSeries::new(
                "segs".into(),
                vol.grid(),
                "us1".into(),
                "st3".into(),
            )],
            doses: Vec::new(),
            plans: Vec::new(),
            planar_images: Vec::new(),
            registrations: Vec::new(),
            treat_records: Vec::new(),
            reports: Vec::new(),
            setups: Vec::new(),
            fourd_groups: Vec::new(),
            warnings: Vec::new(),
            default_window: (40.0, 400.0),
        }
    }

    /// Patients, their studies and the modality level DICOM implies but does
    /// not store, all in first-seen order.
    #[test]
    fn the_tree_nests_patient_study_modality_series() {
        let layout = tree_layout(&study());
        assert_eq!(layout.len(), 2, "two patients");
        assert_eq!(layout[0].key, "P1");
        assert_eq!(layout[0].studies.len(), 2);
        assert_eq!(layout[1].studies.len(), 1);

        let st1 = &layout[0].studies[0];
        let mods: Vec<&str> = st1.modalities.iter().map(|(m, _)| m.as_str()).collect();
        assert_eq!(mods, vec!["CT", "MR"], "one node per modality, first seen");
        assert_eq!(st1.modalities[0].1, vec![0, 1], "both CT series under CT");
        assert_eq!(st1.modalities[1].1, vec![2]);
        assert_eq!(layout[1].studies[0].modalities[0].0, "US");
    }

    #[test]
    fn the_arrows_walk_the_level_the_tree_draws() {
        let st = study();
        let series = |i: usize| RenameTarget::Series { slot: 0, idx: i };
        let level = |t: &RenameTarget| tree_level_of(&st, 0, t);

        // A CT of study st1: the other CT of that study, not the MR beside
        // it and not the CT of the other study.
        assert_eq!(level(&series(0)), vec![series(0), series(1)]);
        assert_eq!(level(&series(1)), vec![series(0), series(1)]);
        // The MR is alone in its own modality node.
        assert_eq!(level(&series(2)), vec![series(2)]);
        // A different study is a different level.
        assert_eq!(level(&series(3)), vec![series(3)]);
        // A different patient too.
        assert_eq!(level(&series(4)), vec![series(4)]);

        // Studies and patients walk their own levels.
        let studies = level(&RenameTarget::Study {
            slot: 0,
            uid: "st1".into(),
        });
        assert_eq!(
            studies.len(),
            3,
            "every study of the workspace: {studies:?}"
        );
        let patients = level(&RenameTarget::Patient {
            slot: 0,
            key: "P1".into(),
        });
        assert_eq!(patients.len(), 2);

        // Structure sets: the workspace's sets, in order.
        let sets = level(&RenameTarget::Set(SetRef {
            slot: 0,
            kind: SetKind::Structures,
            idx: 0,
        }));
        assert_eq!(sets.len(), st.structure_sets.len());

        // A row whose level has nothing else in it does not move, and a
        // level that is empty is not a panic.
        assert!(level(&RenameTarget::Dose { slot: 0, idx: 0 }).is_empty());
    }

    #[test]
    fn a_four_d_group_is_the_level_for_the_phases_it_holds() {
        use crate::fourd::{FourDGroup, Member, Role};
        let mut st = study();
        // Group the two CT series of st1 as a 4D acquisition. The tree files
        // them under the group and not under the CT node, so that is the
        // level the arrows have to follow.
        st.fourd_groups.push(FourDGroup {
            name: "4DCT (2 phases)".into(),
            study_uid: "st1".into(),
            members: vec![
                Member {
                    series_uid: "ct1".into(),
                    label: "0%".into(),
                    role: Role::Phase,
                    percent: Some(0.0),
                },
                Member {
                    series_uid: "ct2".into(),
                    label: "50%".into(),
                    role: Role::Phase,
                    percent: Some(50.0),
                },
            ],
            custom: false,
            dissolved: false,
        });
        let series = |i: usize| RenameTarget::Series { slot: 0, idx: i };
        assert_eq!(
            tree_level_of(&st, 0, &series(0)),
            vec![series(0), series(1)],
            "the group's phases, in member order"
        );
        // The MR is untouched by the grouping.
        assert_eq!(tree_level_of(&st, 0, &series(2)), vec![series(2)]);
    }

    /// An RT object with an incomplete StudyInstanceUID must still land
    /// somewhere reachable rather than disappearing from the tree.
    #[test]
    fn rt_objects_fall_back_to_their_series_then_to_the_first_study() {
        let layout = tree_layout(&study());
        assert_eq!(
            layout[0].studies[0].structs,
            vec![0, 2],
            "own study, then the orphan"
        );
        assert_eq!(
            layout[0].studies[1].structs,
            vec![1],
            "no study of its own - filed under the study of the series it references"
        );
        assert_eq!(
            layout[1].studies[0].segs,
            vec![0],
            "the segmentation series follows its own study"
        );
        let total: usize = layout
            .iter()
            .flat_map(|p| p.studies.iter())
            .map(|s| s.structs.len())
            .sum();
        assert_eq!(total, 3, "every structure set is reachable exactly once");
    }

    /// A report or an additional object belongs to a study, not to the
    /// workspace: it is filed under the study its header names, and one
    /// that names none under the first study, as an RT object would be.
    #[test]
    fn planar_images_and_reports_sit_under_their_study() {
        use crate::extras::{PlanarImage, PlanarKind};
        use crate::report::Report;
        let mut st = study();
        let img = |study_uid: &str, kind| PlanarImage {
            study_uid: study_uid.into(),
            kind,
            ..Default::default()
        };
        st.planar_images = vec![
            img("st1", PlanarKind::Projection),
            img("st3", PlanarKind::Additional),
            img("", PlanarKind::Projection),
        ];
        st.reports = vec![
            Report {
                study_uid: "st3".into(),
                ..Default::default()
            },
            Report {
                study_uid: "st1".into(),
                ..Default::default()
            },
        ];
        let layout = tree_layout(&st);
        let (st1, st2, st3) = (
            &layout[0].studies[0],
            &layout[0].studies[1],
            &layout[1].studies[0],
        );
        assert_eq!((&st1.planar, &st1.additional), (&vec![0, 2], &vec![]));
        assert!(st2.planar.is_empty() && st2.reports.is_empty());
        assert_eq!(
            st3.additional,
            vec![1],
            "a page under the other patient's study"
        );
        assert_eq!((&st1.reports, &st3.reports), (&vec![1], &vec![0]));
        // The arrows step through the list a row is in, not the workspace.
        let level = tree_level_of(&st, 0, &RenameTarget::Report { slot: 0, idx: 0 });
        assert_eq!(level, vec![RenameTarget::Report { slot: 0, idx: 0 }]);
    }

    /// A workspace can hold no image series at all - a folder of RT images, a
    /// structure set opened on its own. Its objects must still be in the
    /// tree, under a patient and a study, or there is no way to reach them.
    #[test]
    fn a_dataset_without_image_series_still_has_a_patient_and_a_study() {
        let mut st = study();
        st.series.clear();
        st.volume = Arc::new(Volume::empty());
        st.meta = loader::PatientMeta {
            patient_name: "Doe^John".into(),
            patient_id: "P9".into(),
            study_date: "20260901".into(),
            study_description: "Portal images".into(),
            study_id: String::new(),
        };
        let layout = tree_layout(&st);
        assert_eq!(
            layout.len(),
            1,
            "one patient, from the workspace's own tags"
        );
        assert_eq!(layout[0].title, "Doe John (P9)");
        // ss1 names st1, ss2 names none but references a series that is gone,
        // ss3 names none at all: two real studies plus the fallback.
        assert_eq!(
            layout[0].studies.len(),
            2,
            "one node per study the objects name: {:?}",
            layout[0].studies.iter().map(|s| &s.uid).collect::<Vec<_>>()
        );
        assert!(
            layout[0]
                .studies
                .iter()
                .any(|s| s.title.contains("2026-09-01")),
            "the study is dated from the workspace's metadata, readably"
        );
        let total: usize = layout[0].studies.iter().map(|s| s.structs.len()).sum();
        assert_eq!(total, 3, "every structure set is still reachable");
        assert_eq!(
            layout[0]
                .studies
                .iter()
                .map(|s| s.segs.len())
                .sum::<usize>(),
            1,
            "and so is the segmentation series"
        );
    }

    /// The degenerate case: objects that name no study whatsoever, in a
    /// workspace with no series to fall back to.
    #[test]
    fn objects_naming_no_study_at_all_still_get_one() {
        let mut st = study();
        st.series.clear();
        st.volume = Arc::new(Volume::empty());
        st.structure_sets = vec![structset("ss3", "", "")];
        st.seg_series.clear();
        let layout = tree_layout(&st);
        assert_eq!(layout.len(), 1);
        assert_eq!(layout[0].studies.len(), 1);
        assert_eq!(layout[0].studies[0].uid, "", "the catch-all study");
        assert_eq!(layout[0].studies[0].structs, vec![0]);
    }
}
