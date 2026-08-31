//! Copy / move / remove over the patient - study - series tree.
//!
//! Transfers work on boolean masks over the source study's parallel arrays,
//! so a selection at any level of the tree reduces to the same subset
//! operation, and reference chains (series -> RTSTRUCT -> RTDOSE -> RTPLAN)
//! are carried along with it.

use super::*;

impl ViewerApp {
    /// Empty a study slot completely (used by tree "move" actions).
    pub(super) fn tree_clear_slot(&mut self, slot: usize) {
        self.slots[slot] = StudySlot::empty();
        self.planar_windows.retain(|w| w.slot != slot);
        self.d3_windows.retain(|w| w.slot != slot);
        if self.maximized.map(|(s, _)| s == slot).unwrap_or(false) {
            self.maximized = None;
        }
        self.clear_registration();
        self.hovered_slot = 0;
    }

    // -- Data tree copy / move / remove actions ----------------------------
    pub(super) fn apply_tree_action(&mut self, action: TreeAction) {
        self.tree_transfer(action.from, &action.sel, action.op);
    }

    /// Series selection mask for a tree selection.
    pub(super) fn tree_sel_mask(study: &LoadedStudy, sel: &TreeSel) -> Vec<bool> {
        match sel {
            TreeSel::Patient(pid) => study
                .series
                .iter()
                .map(|s| s.patient_key() == pid)
                .collect(),
            TreeSel::Study(uid) => study.series.iter().map(|s| s.study_uid == *uid).collect(),
            TreeSel::Series(i) => (0..study.series.len()).map(|k| k == *i).collect(),
        }
    }

    /// Which RT objects the selected series carry along. A single series
    /// takes only its reference chain (RTSTRUCT drawn on it, plans made on
    /// those structure sets, doses computed for those plans); study/patient
    /// selections additionally take objects filed under the same studies.
    pub(super) fn subset_masks(
        study: &LoadedStudy,
        sel: &[bool],
        study_scope: bool,
        take_extras: bool,
    ) -> SubsetMasks {
        if take_extras {
            // Whole slot content: everything goes.
            return SubsetMasks {
                series: sel.to_vec(),
                structs: vec![true; study.structure_sets.len()],
                seg_series: vec![true; study.seg_series.len()],
                doses: vec![true; study.doses.len()],
                plans: vec![true; study.plans.len()],
                take_extras,
            };
        }
        let suids: Vec<&str> = study
            .series
            .iter()
            .zip(sel)
            .filter(|(_, k)| **k)
            .map(|(s, _)| s.uid.as_str())
            .collect();
        let stuids: Vec<&str> = study
            .series
            .iter()
            .zip(sel)
            .filter(|(_, k)| **k)
            .map(|(s, _)| s.study_uid.as_str())
            .filter(|u| !u.is_empty())
            .collect();
        let structs: Vec<bool> = study
            .structure_sets
            .iter()
            .map(|ss| {
                suids.contains(&ss.referenced_series_uid.as_str())
                    || (study_scope
                        && !ss.study_uid.is_empty()
                        && stuids.contains(&ss.study_uid.as_str()))
            })
            .collect();
        let seg_series: Vec<bool> = study
            .seg_series
            .iter()
            .map(|sr| {
                suids.contains(&sr.referenced_series_uid.as_str())
                    || (study_scope
                        && !sr.study_uid.is_empty()
                        && stuids.contains(&sr.study_uid.as_str()))
            })
            .collect();
        let struct_sops: Vec<&str> = study
            .structure_sets
            .iter()
            .zip(&structs)
            .filter(|(_, k)| **k)
            .map(|(s, _)| s.sop_instance_uid.as_str())
            .filter(|u| !u.is_empty())
            .collect();
        let plans: Vec<bool> = study
            .plans
            .iter()
            .map(|p| {
                struct_sops.contains(&p.referenced_structset_uid.as_str())
                    || (study_scope
                        && !p.study_uid.is_empty()
                        && stuids.contains(&p.study_uid.as_str()))
            })
            .collect();
        let plan_sops: Vec<&str> = study
            .plans
            .iter()
            .zip(&plans)
            .filter(|(_, k)| **k)
            .map(|(p, _)| p.sop_instance_uid.as_str())
            .filter(|u| !u.is_empty())
            .collect();
        let doses: Vec<bool> = study
            .doses
            .iter()
            .map(|d| {
                plan_sops.contains(&d.referenced_plan_uid.as_str())
                    || (study_scope
                        && !d.study_uid.is_empty()
                        && stuids.contains(&d.study_uid.as_str()))
            })
            .collect();
        SubsetMasks {
            series: sel.to_vec(),
            structs,
            seg_series,
            doses,
            plans,
            take_extras,
        }
    }

    /// Standalone copy of the selected subset. `activate` is the source
    /// series index to display; the volume is a placeholder (the source's
    /// current volume) that is correct exactly when `activate` is the
    /// source's active series.
    pub(super) fn build_subset(
        study: &LoadedStudy,
        masks: &SubsetMasks,
        activate: usize,
    ) -> LoadedStudy {
        let pick = |sel: &[bool], n: usize| -> Vec<usize> {
            (0..n)
                .filter(|&i| sel.get(i).copied().unwrap_or(false))
                .collect()
        };
        let series: Vec<loader::SeriesInfo> = pick(&masks.series, study.series.len())
            .iter()
            .map(|&i| study.series[i].clone())
            .collect();
        let sub_active = pick(&masks.series, study.series.len())
            .iter()
            .position(|&i| i == activate)
            .unwrap_or(0);
        let se = &study.series[activate];
        let meta = loader::PatientMeta {
            patient_name: if se.patient_name.is_empty() {
                study.meta.patient_name.clone()
            } else {
                se.patient_name.clone()
            },
            patient_id: if se.patient_id.is_empty() {
                study.meta.patient_id.clone()
            } else {
                se.patient_id.clone()
            },
            study_date: se.study_date.clone(),
            study_description: se.study_description.clone(),
        };
        LoadedStudy {
            meta,
            series,
            active_series: sub_active,
            volume: study.volume.clone(),
            structure_sets: pick(&masks.structs, study.structure_sets.len())
                .iter()
                .map(|&i| study.structure_sets[i].clone())
                .collect(),
            seg_series: pick(&masks.seg_series, study.seg_series.len())
                .iter()
                .map(|&i| study.seg_series[i].clone())
                .collect(),
            doses: pick(&masks.doses, study.doses.len())
                .iter()
                .map(|&i| study.doses[i].clone())
                .collect(),
            plans: pick(&masks.plans, study.plans.len())
                .iter()
                .map(|&i| study.plans[i].clone())
                .collect(),
            planar_images: if masks.take_extras {
                study.planar_images.clone()
            } else {
                Vec::new()
            },
            registrations: if masks.take_extras {
                study.registrations.clone()
            } else {
                Vec::new()
            },
            treat_records: if masks.take_extras {
                study.treat_records.clone()
            } else {
                Vec::new()
            },
            fourd_groups: study.fourd_groups.clone(),
            warnings: Vec::new(),
            default_window: study.default_window,
        }
    }

    /// Copy / move / remove a tree selection. Copy and move merge the
    /// selection (plus its linked RT objects) into the other dataset slot;
    /// move and remove then delete it from the source.
    pub(super) fn tree_transfer(&mut self, from: usize, sel: &TreeSel, op: TreeOp) {
        let Some(study) = self.slots[from].study.as_ref() else {
            return;
        };
        let sel_mask = Self::tree_sel_mask(study, sel);
        if !sel_mask.iter().any(|b| *b) {
            return;
        }
        let all_selected = sel_mask.iter().all(|b| *b);
        let study_scope = !matches!(sel, TreeSel::Series(_));
        let masks = Self::subset_masks(study, &sel_mask, study_scope, study_scope && all_selected);

        if op != TreeOp::Remove {
            // Choose the series the destination will display.
            let active = study.active_series;
            let activate = if sel_mask.get(active).copied().unwrap_or(false) {
                active
            } else {
                match (0..study.series.len())
                    .find(|&i| sel_mask[i] && !study.series[i].files.is_empty())
                {
                    Some(i) => i,
                    None => {
                        self.error = Some(
                            "The selected series exist only in memory (no source files) — \
                             they cannot be loaded as the displayed volume of the other slot"
                                .into(),
                        );
                        return;
                    }
                }
            };
            let sub = Self::build_subset(study, &masks, activate);
            let direct = (activate == active).then(|| (study.volume.clone(), study.default_window));
            let uid = study.series[activate].uid.clone();
            self.tree_insert(1 - from, sub, &uid, direct);
        }
        if op != TreeOp::Copy {
            if all_selected {
                self.tree_clear_slot(from);
            } else {
                self.remove_subset(from, &masks);
            }
        }
    }

    /// Merge a subset into a slot (or load it into an empty slot) and show
    /// the series with UID `activate_uid` there. `direct` carries the
    /// volume when it is already in memory (no file reload needed).
    pub(super) fn tree_insert(
        &mut self,
        to: usize,
        sub: LoadedStudy,
        activate_uid: &str,
        direct: Option<(Arc<Volume>, (f32, f32))>,
    ) {
        self.comparison = true;
        if self.slots[to].study.is_none() {
            let need_switch = direct.is_none();
            let idx = sub.active_series;
            self.on_study_loaded(to, sub);
            if need_switch {
                self.start_series_switch(to, idx);
            }
            return;
        }
        let idx = {
            let dest = self.slots[to].study.as_mut().unwrap();
            let notes = loader::merge_study(dest, sub);
            dest.warnings.extend(notes);
            dest.series.iter().position(|s| s.uid == activate_uid)
        };
        if let Some(idx) = idx {
            match direct {
                Some((vol, win)) => self.apply_new_volume(to, vol, win, idx),
                None => self.start_series_switch(to, idx),
            }
        }
        self.settings_gen += 1;
    }

    /// Delete the masked subset from a slot, keeping the displayed volume
    /// valid (switching to another file-backed series if the active one was
    /// removed, clearing the slot if nothing is left).
    pub(super) fn remove_subset(&mut self, slot: usize, masks: &SubsetMasks) {
        let mut reload: Option<usize> = None;
        let mut empty = false;
        {
            let s = &mut self.slots[slot];
            let Some(st) = s.study.as_mut() else { return };
            let active_uid = st.series.get(st.active_series).map(|se| se.uid.clone());
            let active_struct_uid = st
                .structure_sets
                .get(s.active_structs)
                .map(|ss| ss.sop_instance_uid.clone());
            let mut i = 0;
            st.series.retain(|_| {
                let k = !masks.series.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            let mut i = 0;
            st.structure_sets.retain(|_| {
                let k = !masks.structs.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            let mut i = 0;
            st.seg_series.retain(|_| {
                let k = !masks.seg_series.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            let mut i = 0;
            st.doses.retain(|_| {
                let k = !masks.doses.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            let mut i = 0;
            st.plans.retain(|_| {
                let k = !masks.plans.get(i).copied().unwrap_or(false);
                i += 1;
                k
            });
            if masks.take_extras {
                st.planar_images.clear();
                st.registrations.clear();
                st.treat_records.clear();
            }
            if st.series.is_empty() {
                empty = true;
            } else {
                match active_uid
                    .as_deref()
                    .and_then(|uid| st.series.iter().position(|se| se.uid == uid))
                {
                    Some(i) => st.active_series = i,
                    None => {
                        if let Some(i) = st.series.iter().position(|se| !se.files.is_empty()) {
                            st.active_series = i;
                            reload = Some(i);
                        } else {
                            st.active_series = 0;
                        }
                    }
                }
                // Re-locate the active structure set after pruning (indices may
                // have shifted, or the set itself may be gone); rebuild the
                // visibility list whenever the active set changed so it can
                // never be indexed with a stale length.
                let relocated = active_struct_uid.as_deref().and_then(|uid| {
                    st.structure_sets
                        .iter()
                        .position(|ss| ss.sop_instance_uid == uid)
                });
                match relocated {
                    Some(i) => s.active_structs = i,
                    None => {
                        s.active_structs = 0;
                        let n = st
                            .structure_sets
                            .first()
                            .map(|ss| ss.rois.len())
                            .unwrap_or(0);
                        s.roi_visible = vec![true; n];
                    }
                }
                if s.active_dose >= st.doses.len() {
                    s.active_dose = 0;
                }
                if s.active_seg_series >= st.seg_series.len() {
                    s.active_seg_series = 0;
                    s.active_seg = 0;
                }
            }
        }
        if empty {
            self.tree_clear_slot(slot);
            return;
        }
        if let Some(st) = self.slots[slot].study.as_mut() {
            // Groups follow the series they reference; removed series drop
            // out and a group left empty disappears.
            st.refresh_fourd();
        }
        if let Some(i) = reload {
            self.start_series_switch(slot, i);
        }
        self.settings_gen += 1;
    }
}

#[cfg(test)]
mod tree_tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::rtdose::DoseGrid;
    use crate::rtplan::PlanInfo;
    use crate::rtstruct::StructureSet;

    fn series(uid: &str, patient: &str, study: &str) -> loader::SeriesInfo {
        loader::SeriesInfo {
            uid: uid.into(),
            modality: "CT".into(),
            description: format!("{uid} desc"),
            patient_id: patient.into(),
            patient_name: format!("{patient}^Name"),
            study_uid: study.into(),
            study_date: "20260818".into(),
            study_description: String::new(),
            series_number: None,
            temporal_id: None,
            files: vec![std::path::PathBuf::from(format!("{uid}.dcm"))],
        }
    }

    fn structset(sop: &str, series_uid: &str, study: &str) -> StructureSet {
        StructureSet {
            label: sop.into(),
            frame_of_reference_uid: String::new(),
            sop_instance_uid: sop.into(),
            study_uid: study.into(),
            referenced_series_uid: series_uid.into(),
            file_name: format!("{sop}.dcm"),
            rois: Vec::new(),
        }
    }

    fn plan(sop: &str, structset_sop: &str, study: &str) -> PlanInfo {
        PlanInfo {
            label: sop.into(),
            name: String::new(),
            date: String::new(),
            plan_kind: "Ion".into(),
            n_fractions: None,
            target_prescription_dose: None,
            sop_instance_uid: sop.into(),
            study_uid: study.into(),
            referenced_structset_uid: structset_sop.into(),
            beams: Vec::new(),
        }
    }

    fn dose(plan_sop: &str, study: &str) -> DoseGrid {
        DoseGrid {
            data: vec![0.0],
            dims: [1, 1, 1],
            spacing: [1.0, 1.0],
            origin: Vec3::new(0.0, 0.0, 0.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            offsets: vec![0.0],
            units: "GY".into(),
            summation_type: "PLAN".into(),
            max_dose: 1.0,
            frame_of_reference_uid: String::new(),
            study_uid: study.into(),
            referenced_plan_uid: plan_sop.into(),
            label: plan_sop.into(),
        }
    }

    /// Two series, each with its own RTSTRUCT ▶ RTPLAN ▶ RTDOSE chain.
    fn two_chain_study() -> LoadedStudy {
        LoadedStudy {
            meta: loader::PatientMeta::default(),
            series: vec![series("se1", "P1", "st1"), series("se2", "P1", "st2")],
            active_series: 0,
            volume: Arc::new(Volume {
                data: vec![0],
                dims: [1, 1, 1],
                spacing: [1.0, 1.0, 1.0],
                origin: Vec3::new(0.0, 0.0, 0.0),
                row_dir: Vec3::new(1.0, 0.0, 0.0),
                col_dir: Vec3::new(0.0, 1.0, 0.0),
                normal: Vec3::new(0.0, 0.0, 1.0),
                frame_of_reference_uid: String::new(),
                min_value: 0,
                max_value: 0,
            }),
            structure_sets: vec![
                structset("ss1", "se1", "st1"),
                structset("ss2", "se2", "st2"),
            ],
            seg_series: Vec::new(),
            doses: vec![dose("pl1", "st1"), dose("pl2", "st2")],
            plans: vec![plan("pl1", "ss1", "st1"), plan("pl2", "ss2", "st2")],
            planar_images: Vec::new(),
            registrations: Vec::new(),
            treat_records: Vec::new(),
            fourd_groups: Vec::new(),
            warnings: Vec::new(),
            default_window: (40.0, 400.0),
        }
    }

    /// Selecting one series must take exactly its reference chain — the bug
    /// this guards against is "move series moved every series and RT object".
    #[test]
    fn series_selection_takes_only_linked_objects() {
        let study = two_chain_study();
        let sel = ViewerApp::tree_sel_mask(&study, &TreeSel::Series(0));
        assert_eq!(sel, vec![true, false]);
        let masks = ViewerApp::subset_masks(&study, &sel, false, false);
        assert_eq!(masks.series, vec![true, false]);
        assert_eq!(masks.structs, vec![true, false]);
        assert_eq!(masks.plans, vec![true, false]);
        assert_eq!(masks.doses, vec![true, false]);

        let sub = ViewerApp::build_subset(&study, &masks, 0);
        assert_eq!(sub.series.len(), 1);
        assert_eq!(sub.series[0].uid, "se1");
        assert_eq!(sub.structure_sets.len(), 1);
        assert_eq!(sub.structure_sets[0].sop_instance_uid, "ss1");
        assert_eq!(sub.plans.len(), 1);
        assert_eq!(sub.doses.len(), 1);
        assert_eq!(sub.doses[0].referenced_plan_uid, "pl1");
    }

    /// Study selection takes the chain plus same-study objects.
    #[test]
    fn study_selection_takes_study_objects() {
        let study = two_chain_study();
        let sel = ViewerApp::tree_sel_mask(&study, &TreeSel::Study("st2".into()));
        assert_eq!(sel, vec![false, true]);
        let masks = ViewerApp::subset_masks(&study, &sel, true, false);
        assert_eq!(masks.structs, vec![false, true]);
        assert_eq!(masks.plans, vec![false, true]);
        assert_eq!(masks.doses, vec![false, true]);
    }

    /// Patient selection over all series covers everything.
    #[test]
    fn patient_selection_covers_all() {
        let study = two_chain_study();
        let sel = ViewerApp::tree_sel_mask(&study, &TreeSel::Patient("P1".into()));
        assert_eq!(sel, vec![true, true]);
        let masks = ViewerApp::subset_masks(&study, &sel, true, true);
        assert!(masks.structs.iter().all(|b| *b));
        assert!(masks.take_extras);
    }

    /// merge_study skips series and RT objects that are already present.
    #[test]
    fn merge_dedupes_by_uid() {
        let mut dest = two_chain_study();
        let masks = ViewerApp::subset_masks(&dest, &[true, false], false, false);
        let sub = ViewerApp::build_subset(&dest, &masks, 0);
        let notes = loader::merge_study(&mut dest, sub);
        assert_eq!(dest.series.len(), 2, "duplicate series must not be added");
        assert_eq!(dest.structure_sets.len(), 2);
        assert_eq!(dest.plans.len(), 2);
        assert_eq!(dest.doses.len(), 2);
        assert!(!notes.is_empty(), "skipping duplicates should be reported");

        // A genuinely new series does get merged.
        let mut extra = two_chain_study();
        extra.series[0].uid = "se3".into();
        extra.series[0].study_uid = "st3".into();
        extra.structure_sets.clear();
        extra.plans.clear();
        extra.doses.clear();
        extra.series.truncate(1);
        loader::merge_study(&mut dest, extra);
        assert_eq!(dest.series.len(), 3);
    }
}
