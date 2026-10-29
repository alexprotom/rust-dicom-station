//! A 4D group's node in the data tree, and what its menu does to the
//! group (add, remove, shift, re-detect, dissolve, analyse).

use super::data_tree::{SeriesMenu, SeriesOut};
use super::*;

impl ViewerApp {
    /// One 4D group node: the ordered members, each row switching the
    /// displayed series like an ordinary series row.
    pub(super) fn fourd_node(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        pi: usize,
        si: usize,
        gi: usize,
    ) {
        let badges = self.series_badges(slot);
        let picked = self.picked_series(slot);
        let targets = self.copy_targets(slot);
        let open = self.open_slots();
        let (ctrl, shift) = ui.input(|i| (i.modifiers.command, i.modifiers.shift));
        let Some(study) = self.slots[slot].study.as_ref() else {
            return;
        };
        let Some(group) = study.fourd_groups.get(gi) else {
            return;
        };
        let title = format!("🎞 {}", group.name);
        let resolved = group.resolve(&study.series);
        // (member index, series index, row label, series UID) for every
        // surviving member.
        let rows: Vec<(usize, usize, String, String)> = group
            .members
            .iter()
            .enumerate()
            .zip(&resolved)
            .filter_map(|((mi, m), r)| {
                r.map(|sidx| {
                    let se = &study.series[sidx];
                    let tag = m.role.tag();
                    let ord = Self::name_ordinal(study, sidx);
                    let tick = if picked.contains(&sidx) { "✔ " } else { "" };
                    let label = if tag.is_empty() {
                        format!(
                            "{tick}{} - {}{ord} ({} sl.)",
                            m.label,
                            se.description,
                            se.files.len()
                        )
                    } else {
                        format!(
                            "{tick}{tag} - {}{ord} ({} sl.)",
                            se.description,
                            se.files.len()
                        )
                    };
                    (mi, sidx, label, se.uid.clone())
                })
            })
            .collect();
        let tips: Vec<String> = rows
            .iter()
            .map(|(_, sidx, _, _)| Self::series_tip(&study.series[*sidx]))
            .collect();
        let n_members = group.members.len();
        let active = study.active_series;

        let mut switch_to = None;
        let mut fourd: Option<FourDAction> = None;
        let mut rename = None;
        let mut focus = None;
        let mut act: Option<TreeAction> = None;
        let mut out = SeriesOut::default();
        let current = self.tree_focus.clone();
        let members: Vec<usize> = rows.iter().map(|(_, sidx, _, _)| *sidx).collect();
        let showing = rows.iter().any(|(_, sidx, _, _)| *sidx == active);
        let me_group = RenameTarget::FourD { slot, idx: gi };
        let group_focused = current.as_ref() == Some(&me_group);
        let resp = Self::wrapped_node_extra(
            ui,
            ("fourd", slot, pi, si, gi),
            showing,
            title,
            |ui| Self::keyboard_badge(ui, group_focused),
            |ui| {
                for ((mi, sidx, label, uid), tip) in rows.iter().zip(&tips) {
                    let me = RenameTarget::Series { slot, idx: *sidx };
                    let resp = Self::badged_series_row(
                        ui,
                        *sidx == active,
                        label.clone(),
                        uid,
                        &badges,
                        current.as_ref() == Some(&me),
                    );
                    Self::focus_row(ui, &resp, me, &current, &mut focus);
                    // As on any series row: Shift ticks a range, Ctrl one
                    // phase, a plain click displays it.
                    if resp.clicked() {
                        if shift {
                            out.range = Some(*sidx);
                        } else if ctrl {
                            out.pick = Some(uid.clone());
                        } else if *sidx != active {
                            switch_to = Some(*sidx);
                        }
                    }
                    let menu = SeriesMenu::new(slot, *sidx, &picked);
                    resp.on_hover_text(tip.as_str()).context_menu(|ui| {
                        if *sidx != active {
                            if ui.button("👁 Display").clicked() {
                                switch_to = Some(*sidx);
                                ui.close();
                            }
                            ui.separator();
                        }
                        if ui.button("⬆ Move up").clicked() {
                            fourd = Some(FourDAction::Shift {
                                slot,
                                group: gi,
                                member: *mi,
                                delta: -1,
                            });
                            ui.close();
                        }
                        if ui.button("⬇ Move down").clicked() {
                            fourd = Some(FourDAction::Shift {
                                slot,
                                group: gi,
                                member: *mi,
                                delta: 1,
                            });
                            ui.close();
                        }
                        ui.menu_button("Role", |ui| {
                            for (role, label) in [
                                (fourd::Role::Phase, "Phase"),
                                (fourd::Role::Average, "Average (AVG)"),
                                (fourd::Role::Mip, "MIP"),
                                (fourd::Role::MinIp, "MinIP"),
                            ] {
                                if ui.button(label).clicked() {
                                    fourd = Some(FourDAction::SetRole {
                                        slot,
                                        group: gi,
                                        member: *mi,
                                        role,
                                    });
                                    ui.close();
                                }
                            }
                        });
                        if ui.button("Take out of the group").clicked() {
                            fourd = Some(FourDAction::RemoveMember {
                                slot,
                                group: gi,
                                member: *mi,
                            });
                            ui.close();
                        }
                        ui.separator();
                        menu.common_entries(ui, &targets, &open, &mut out);
                    });
                }
                if rows.len() < n_members {
                    ui.weak(format!(
                        "{} member(s) whose series is gone",
                        n_members - rows.len()
                    ));
                }
            },
        );
        Self::focus_row(ui, &resp, me_group.clone(), &current, &mut focus);
        resp.context_menu(|ui| {
            if ui.button("📈 Motion / ITV analysis").clicked() {
                fourd = Some(FourDAction::Analyse { slot, group: gi });
                ui.close();
            }
            if ui.button("Re-detect 4D groups").clicked() {
                fourd = Some(FourDAction::Redetect { slot });
                ui.close();
            }
            ui.separator();
            if Self::rename_entry(ui) {
                rename = Some(me_group.clone());
                ui.close();
            }
            Self::transfer_menu(
                ui,
                &format!("the group's {} series", members.len()),
                &targets,
                &open,
                |to, op| {
                    act = Some(TreeAction {
                        from: slot,
                        to,
                        sel: TreeSel::Many(members.clone()),
                        op,
                    });
                },
            );
            ui.separator();
            // The key takes the group apart and keeps its series: of the
            // two removals, the one that cannot lose images.
            if ui
                .add(egui::Button::new("Dissolve group").shortcut_text("Delete"))
                .on_hover_text("Take the group apart; its series stay in the study")
                .clicked()
            {
                fourd = Some(FourDAction::Dissolve { slot, group: gi });
                ui.close();
            }
            if tip_button(
                ui,
                format!("🗑 Remove group and its {} series", members.len()),
                "Take the group's series - and the structure sets, plans and doses \
                 drawn on them - out of the workspace",
            ) {
                act = Some(TreeAction {
                    from: slot,
                    to: slot,
                    sel: TreeSel::Many(members.clone()),
                    op: TreeOp::Remove,
                });
                ui.close();
            }
        });
        resp.on_hover_text(format!(
            "4D group · {} series: the phases in temporal order, then the reconstructions",
            members.len()
        ));
        if fourd.is_some() {
            self.fourd_action = fourd;
        }
        if act.is_some() {
            self.tree_action = act;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if let Some(i) = switch_to {
            // Stepping between the phases of the group already on display
            // keeps the view: two phases are the same patient a moment
            // apart, and a crosshair that jumps back to the middle slice
            // every time hides the very motion one is looking for. Any
            // other member is an ordinary series switch.
            match self
                .fourd_phases(slot)
                .and_then(|(_, idxs, _)| idxs.iter().position(|x| *x == i))
            {
                Some(phase) => self.goto_phase(slot, phase),
                None => self.start_series_switch(slot, i),
            }
        }
        self.apply_series_out(slot, &members, out, focus);
    }

    /// Apply a deferred 4D-group edit from the tree's context menus.
    pub(super) fn apply_fourd_action(&mut self, act: FourDAction) {
        match act {
            FourDAction::Analyse { slot, group } => {
                self.open_motion_dialog(slot, Some(group));
                self.reveal_module(Module::Motion);
                return;
            }
            FourDAction::Redetect { slot } => {
                if let Some(study) = self.slots[slot].study.as_mut() {
                    // An explicit re-detect is the one action that clears
                    // dissolved tombstones - the user asked for detection.
                    study.fourd_groups.retain(|g| !g.dissolved);
                    study.refresh_fourd();
                }
                return;
            }
            _ => {}
        }
        let slot = match act {
            FourDAction::Add { slot, .. }
            | FourDAction::New { slot, .. }
            | FourDAction::NewFrom { slot, .. }
            | FourDAction::AddMany { slot, .. }
            | FourDAction::RemoveMember { slot, .. }
            | FourDAction::Shift { slot, .. }
            | FourDAction::SetRole { slot, .. }
            | FourDAction::Dissolve { slot, .. } => slot,
            _ => return,
        };
        let Some(study) = self.slots[slot].study.as_mut() else {
            return;
        };
        match act {
            FourDAction::Add { group, series, .. } => {
                let Some(se) = study.series.get(series) else {
                    return;
                };
                let member = fourd::member_for(se, {
                    study
                        .fourd_groups
                        .get(group)
                        .map(|g| g.phase_members().len() + 1)
                        .unwrap_or(1)
                });
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    if !g.members.iter().any(|m| m.series_uid == member.series_uid) {
                        g.members.push(member);
                        g.custom = true;
                    }
                }
            }
            FourDAction::New { series, .. } => {
                let Some(se) = study.series.get(series) else {
                    return;
                };
                let member = fourd::member_for(se, 1);
                let n = study.fourd_groups.len() + 1;
                study.fourd_groups.push(fourd::FourDGroup {
                    name: format!("4D group {n}"),
                    study_uid: se.study_uid.clone(),
                    members: vec![member],
                    custom: true,
                    dissolved: false,
                });
            }
            FourDAction::NewFrom { series, .. } => {
                let Some(group) = fourd::group_from(&study.series, &series) else {
                    return;
                };
                // A series belongs to one group: the ones just picked leave
                // whatever group they were in, and a group left empty by
                // that goes with them.
                let uids: Vec<&str> = group
                    .members
                    .iter()
                    .map(|m| m.series_uid.as_str())
                    .collect();
                for g in &mut study.fourd_groups {
                    let before = g.members.len();
                    g.members.retain(|m| !uids.contains(&m.series_uid.as_str()));
                    if g.members.len() != before {
                        g.custom = true;
                    }
                }
                study.fourd_groups.retain(|g| !g.members.is_empty());
                study.fourd_groups.push(group);
                // The tick marks have done their job.
                self.tree_picks.retain(|(s, _)| *s != slot);
            }
            FourDAction::AddMany { group, series, .. } => {
                let Some(g) = study.fourd_groups.get(group) else {
                    return;
                };
                let mut position = g.phase_members().len() + 1;
                let mut new: Vec<fourd::Member> = Vec::new();
                for &si in &series {
                    let Some(se) = study.series.get(si) else {
                        continue;
                    };
                    if g.members.iter().any(|m| m.series_uid == se.uid)
                        || new.iter().any(|m| m.series_uid == se.uid)
                    {
                        continue;
                    }
                    let m = fourd::member_for(se, position);
                    if m.role == fourd::Role::Phase {
                        position += 1;
                    }
                    new.push(m);
                }
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    g.members.extend(new);
                    fourd::sort_phases_by_percent(g);
                    g.custom = true;
                }
                self.tree_picks.retain(|(s, _)| *s != slot);
            }
            FourDAction::RemoveMember { group, member, .. } => {
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    if g.members.len() == 1 && member == 0 {
                        // Removing the last member dissolves the group; the
                        // member stays inside the tombstone so re-detection
                        // does not immediately rebuild what was taken apart.
                        g.dissolved = true;
                        g.custom = true;
                    } else if member < g.members.len() {
                        g.members.remove(member);
                        g.custom = true;
                    }
                }
            }
            FourDAction::Shift {
                group,
                member,
                delta,
                ..
            } => {
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    let to = member as isize + delta;
                    if to >= 0 && (to as usize) < g.members.len() {
                        g.members.swap(member, to as usize);
                        g.custom = true;
                    }
                }
            }
            FourDAction::SetRole {
                group,
                member,
                role,
                ..
            } => {
                if let Some(m) = study
                    .fourd_groups
                    .get_mut(group)
                    .and_then(|g| g.members.get_mut(member))
                {
                    m.role = role;
                    if role != fourd::Role::Phase {
                        m.label = role.tag().to_string();
                        m.percent = None;
                    } else if m.label.is_empty()
                        || m.label == "AVG"
                        || m.label == "MIP"
                        || m.label == "MinIP"
                    {
                        m.label = format!("t{}", member + 1);
                    }
                }
                if let Some(g) = study.fourd_groups.get_mut(group) {
                    g.custom = true;
                }
            }
            FourDAction::Dissolve { group, .. } if group < study.fourd_groups.len() => {
                // A custom group leaves nothing behind; an auto-detected one
                // leaves a hidden tombstone so re-detection (on the next
                // series change) does not resurrect it. *Re-detect 4D
                // groups* clears tombstones explicitly.
                if study.fourd_groups[group].custom {
                    study.fourd_groups.remove(group);
                } else {
                    study.fourd_groups[group].dissolved = true;
                }
            }
            _ => {}
        }
    }
}
