//! The data tree's structures section: the structure sets of a study,
//! their ROIs with visibility, colour and type, and their menus.

use super::panels::apply_tick;
use super::sets::TemplateAct;
use super::*;

impl ViewerApp {
    /// The RT structure sets filed under one study, then the ROIs of the
    /// active one.
    ///
    /// A ROI's check box is both its visibility and its selection, so *All* /
    /// *None* tick everything or nothing, Shift-click extends a range, and
    /// the action row works on whatever is ticked.
    pub(super) fn structures_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        pat: usize,
        stu: usize,
        which: &[usize],
        displayed: bool,
    ) {
        if self.slots[slot].study.is_none() {
            return;
        }
        // Sets can be listed, renamed, moved and deleted without an image
        // volume; only what would draw into one is held back - and only
        // under the study whose images are on screen, because everything
        // that draws draws on the displayed series and nothing else.
        let has_volume = self.slots[slot].has_volume() && displayed;
        // Rendering runs behind a shared borrow of `self` (the context menus
        // need to list the other workspace's series), so the one piece of
        // mutable state in the list is edited on a copy and written back.
        let mut vis = std::mem::take(&mut self.slots[slot].roi_visible);
        let structs_shown = self.slots[slot].structs_shown;
        let mut new_shown: Option<bool> = None;
        let mut new_active: Option<usize> = None;
        let mut set_act: Option<SetAction> = None;
        let mut item_act: Option<ItemAction> = None;
        let mut new_anchor: Option<(SetRef, usize)> = None;
        let mut new_color: Option<(usize, [u8; 3])> = None;
        let mut new_edit: Option<usize> = None;
        let mut open_editor: Option<usize> = None;
        let mut toggle_lock: Option<usize> = None;
        let mut template_act: Option<TemplateAct> = None;
        let mut poi_act: Option<(PoiAct, usize)> = None;
        let mut update_derived = false;
        let mut focus: Option<RenameTarget> = None;
        // Derived statuses are recomputed only when something changed; the
        // list reads them behind the shared borrow below.
        self.refresh_derived(slot);
        let shift = ui.input(|i| i.modifiers.shift);
        {
            let me = &*self;
            let current = &me.tree_focus;
            let edit_roi = me.edit_target(slot).map(|(_, r)| r);
            let study = me.slots[slot].study.as_ref().unwrap();
            let sets = &study.structure_sets;
            // The active set counts as belonging here only when this study
            // node actually holds it; otherwise this node shows its sets but
            // edits none of them.
            let active_set = which
                .contains(&me.slots[slot].active_structs)
                .then(|| me.slots[slot].active_structs);
            let n_rois = active_set
                .and_then(|i| sets.get(i))
                .map(|ss| ss.rois.len())
                .unwrap_or(0);
            if active_set.is_some() {
                vis.resize(n_rois, true);
            }
            let n_vis = vis.iter().filter(|v| **v).count();
            // The heading counts the sets this node holds; how many of a
            // set's structures are ticked belongs on the set's own row,
            // where it says which set it is talking about.
            let title = format!("RT structures ({})", which.len());
            let mut add_set = false;
            Self::wrapped_node_extra(
                ui,
                ("structs", slot, pat, stu),
                // Open under the study on screen, folded under the others
                // - unless there is nothing on screen to prefer.
                displayed || !me.slots[slot].has_volume(),
                title,
                |ui| {
                    if tip_widget(
                        ui,
                        has_volume,
                        egui::Button::new("+").small(),
                        if has_volume {
                            "New: an empty RT structure set, drawn on the displayed image \
                             series"
                        } else if displayed {
                            "This workspace has no image volume to draw on"
                        } else {
                            "Display this study's images first - a new set is drawn on \
                             the series on screen"
                        },
                    ) {
                        add_set = true;
                    }
                },
                |ui| {
                    for &i in which {
                        let Some(set) = sets.get(i) else { continue };
                        let here = SetRef {
                            slot,
                            kind: SetKind::Structures,
                            idx: i,
                        };
                        let name = if set.label.is_empty() {
                            &set.file_name
                        } else {
                            &set.label
                        };
                        let selected = active_set == Some(i);
                        let resp = Self::series_row(
                            ui,
                            selected,
                            structs_shown,
                            format!(
                                "{}{name} ({}){}",
                                if set.locked { "🔒 " } else { "" },
                                // Ticked of all, for the set whose list is
                                // open; a set that is not the active one has
                                // no tick boxes of its own to count, so it
                                // says how many structures it holds.
                                if selected {
                                    format!("{n_vis}/{}", set.rois.len())
                                } else {
                                    format!("{}", set.rois.len())
                                },
                                Self::series_suffix(study, &set.referenced_series_uid)
                            ),
                            current.as_ref() == Some(&RenameTarget::Set(here)),
                        );
                        Self::focus_row(ui, &resp, RenameTarget::Set(here), current, &mut focus);
                        // Clicking the shown set hides it; clicking any
                        // other makes it the shown one.
                        let mut toggle = resp.clicked();
                        let locked = set.locked;
                        let tip = Self::object_tip(
                            &set.sop_instance_uid,
                            if set.file_name.is_empty() {
                                "made here, not written to a file"
                            } else {
                                &set.file_name
                            },
                        );
                        let resp = resp.on_hover_text(tip);
                        resp.context_menu(|ui| {
                            if Self::show_entry(ui, selected && structs_shown) {
                                toggle = true;
                                ui.close();
                            }
                            ui.separator();
                            ui.menu_button("🏷 Template", |ui| {
                                if tip_button(
                                    ui,
                                    "Save this set as a template",
                                    "The names, types, colours and recipes - not the \
                                     geometry. Saved under the set's own label, in \
                                     the templates folder",
                                ) {
                                    template_act = Some(TemplateAct::Save(i));
                                    ui.close();
                                }
                                let names = crate::templates::list();
                                if names.is_empty() {
                                    ui.weak("no templates saved yet");
                                }
                                for name in names {
                                    if tip_button(
                                        ui,
                                        format!("Apply '{name}'"),
                                        "Create the structures of this template that \
                                         are not here yet, with their types, colours \
                                         and recipes. Nothing already in the set is \
                                         touched",
                                    ) {
                                        template_act = Some(TemplateAct::Apply(i, name.clone()));
                                        ui.close();
                                    }
                                }
                            });
                            if tip_button(
                                ui,
                                if locked { "🔓 Unlock" } else { "🔒 Lock" },
                                if locked {
                                    "Allow this set to be edited again"
                                } else {
                                    "Make this set read-only: no drawing, no new \
                                     structures, nothing removed. It is written out as \
                                 an approved structure set, and it is a guard \
                                 against a slip of the hand, not a signature - \
                                 anyone here can unlock it again"
                                },
                            ) {
                                toggle_lock = Some(i);
                                ui.close();
                            }
                            ui.separator();
                            me.set_context_menu(ui, here, &mut set_act)
                        });
                        if toggle {
                            if selected {
                                new_shown = Some(!structs_shown);
                            } else {
                                new_active = Some(i);
                                new_shown = Some(true);
                            }
                        }

                        // The structures of a set, and what acts on them, sit
                        // under *that* set - not at the end of the list, where
                        // ten sets put them a screen away from the row they
                        // belong to.
                        let Some(ss) = (if selected { sets.get(i) } else { None }) else {
                            continue;
                        };
                        let selection: Vec<usize> = vis
                            .iter()
                            .enumerate()
                            .filter(|(_, v)| **v)
                            .map(|(i, _)| i)
                            .collect();
                        ui.horizontal_wrapped(|ui| {
                            if small_tip_button(ui, "All", "Show and select every structure") {
                                vis.iter_mut().for_each(|v| *v = true);
                            }
                            if ui.small_button("None").clicked() {
                                vis.iter_mut().for_each(|v| *v = false);
                            }
                            if me.derived[slot]
                                .items
                                .iter()
                                .any(|s| matches!(s, Some(crate::derived::Status::NeedsUpdate)))
                                && ui
                                    .add_enabled(
                                        has_volume,
                                        egui::Button::new("Update derived").small(),
                                    )
                                    .on_hover_text(
                                        "Re-run the recipe of every derived structure here \
                                     whose operands have changed",
                                    )
                                    .clicked()
                            {
                                update_derived = true;
                            }
                            me.selection_buttons(ui, here, &selection, &mut item_act);
                        });
                        let anchor = me.tick_anchor.filter(|(r, _)| *r == here).map(|(_, i)| i);
                        for (i, roi) in ss.rois.iter().enumerate() {
                            ui.horizontal(|ui| {
                                let mut color = roi.color;
                                if color_swatch(ui, &mut color) {
                                    new_color = Some((i, color));
                                }
                                // Shown or not is the tick box; selected - the
                                // one the editor and the contour tools work
                                // on - is the name, as in the segmentation list.
                                let tick = ui.checkbox(&mut vis[i], "").on_hover_text(
                                    "Show / select this structure\nShift-click: tick or \
                                     untick the whole range from the last one",
                                );
                                if tick.clicked() {
                                    new_anchor =
                                        Some((here, apply_tick(&mut vis, i, shift, anchor)));
                                }
                                // Derived: green circle up to date, red square
                                // out of date, yellow triangle overridden. It
                                // goes *before* the name, because a name is as
                                // long as it likes and a narrow panel would
                                // push a marker at the end off the edge.
                                if let Some(st) = me.derived_status(slot, i) {
                                    let c = st.color();
                                    ui.label(egui::RichText::new(st.glyph()).color(theme::rgb(c)))
                                        .on_hover_text(format!(
                                            "Derived structure: {}\nThe recipe is in the \
                                             Structure editor",
                                            st.label()
                                        ));
                                }
                                // The name stays live under any study: renaming,
                                // removing and copying need no image volume. What
                                // draws - editing it, the crosshair - waits for one.
                                let resp = ui
                                    .add(
                                        egui::Button::selectable(
                                            edit_roi == Some(i),
                                            format!(
                                                "{}{}",
                                                roi.name,
                                                if roi.roi_type.is_empty() {
                                                    String::new()
                                                } else {
                                                    format!("  [{}]", roi.roi_type)
                                                }
                                            ),
                                        )
                                        .wrap(),
                                    )
                                    .on_hover_text(format!(
                                        "ROI {} · {} contour(s)",
                                        roi.number,
                                        roi.contours.len()
                                    ));
                                Self::keyboard_badge(
                                    ui,
                                    current.as_ref()
                                        == Some(&RenameTarget::Item { set: here, idx: i }),
                                );
                                Self::focus_row(
                                    ui,
                                    &resp,
                                    RenameTarget::Item { set: here, idx: i },
                                    current,
                                    &mut focus,
                                );
                                if resp.clicked() && has_volume {
                                    new_edit = Some(i);
                                }
                                resp.context_menu(|ui| {
                                    if enabled_tip_button(
                                        ui,
                                        has_volume,
                                        "📝 Edit in the Structure editor",
                                        "Select it and open the editor on it",
                                    ) {
                                        open_editor = Some(i);
                                        ui.close();
                                    }
                                    if enabled_tip_button(
                                        ui,
                                        has_volume,
                                        "⌖ Localize",
                                        "Put the crosshair on it: on the point, or on \
                                         the centre of gravity of the structure",
                                    ) {
                                        poi_act = Some((PoiAct::Localize, i));
                                        ui.close();
                                    }
                                    if roi.is_poi() {
                                        if enabled_tip_button(
                                            ui,
                                            has_volume,
                                            "Move to the crosshair",
                                            "Put the point where the three views cross",
                                        ) {
                                            poi_act = Some((PoiAct::MoveHere, i));
                                            ui.close();
                                        }
                                        if !roi.is_localization()
                                            && ui
                                                .button("Make the localization point")
                                                .on_hover_text(
                                                    "The one point per structure set the \
                                                     patient is lined up on; any other \
                                                     falls back to a plain marker",
                                                )
                                                .clicked()
                                        {
                                            poi_act = Some((PoiAct::Localization, i));
                                            ui.close();
                                        }
                                    } else if enabled_tip_button(
                                        ui,
                                        has_volume,
                                        "+ POI at its centre",
                                        "A new point of interest at this structure's \
                                         centre of gravity",
                                    ) {
                                        poi_act = Some((PoiAct::AtCentre, i));
                                        ui.close();
                                    }
                                    ui.separator();
                                    me.item_context_menu(
                                        ui,
                                        here,
                                        i,
                                        &roi.name,
                                        &selection,
                                        &mut item_act,
                                    )
                                });
                                if let Some(p) = roi.point() {
                                    ui.label(
                                        egui::RichText::new(if roi.is_localization() {
                                            "🎯"
                                        } else {
                                            "✱"
                                        })
                                        .weak(),
                                    )
                                    .on_hover_text(format!(
                                        "{}\n{:.1}, {:.1}, {:.1} mm",
                                        if roi.is_localization() {
                                            "The localization point"
                                        } else {
                                            "Point of interest"
                                        },
                                        p.x,
                                        p.y,
                                        p.z
                                    ));
                                }
                            });
                        }
                    }
                },
            );
            if add_set {
                set_act = Some(SetAction::New(SetRef {
                    slot,
                    kind: SetKind::Structures,
                    idx: SetRef::NEW,
                }));
            }
        }
        self.slots[slot].roi_visible = vis;
        if focus.is_some() {
            self.tree_focus = focus;
        }
        if let Some((i, color)) = new_color {
            let active = self.slots[slot].active_structs;
            if let Some(roi) = self.slots[slot].roi_mut(active, i) {
                roi.color = color;
            }
        }
        if let Some(on) = new_shown {
            self.slots[slot].structs_shown = on;
        }
        if new_anchor.is_some() {
            self.tick_anchor = new_anchor;
        }
        if let Some(i) = new_active {
            let s = &mut self.slots[slot];
            s.active_structs = i;
            let n = s
                .study
                .as_ref()
                .map(|st| st.structure_sets[i].rois.len())
                .unwrap_or(0);
            s.roi_visible = vec![true; n];
            // Another set means another list: nothing is being edited yet.
            s.active_roi = usize::MAX;
            self.drop_edit();
        }
        if let Some(i) = new_edit {
            self.slots[slot].active_roi = i;
            self.drop_edit();
        }
        if let Some(i) = open_editor {
            self.edit_roi_in_editor(slot, i);
        }
        if let Some(act) = template_act {
            self.apply_template_act(slot, act);
        }
        if let Some(i) = toggle_lock {
            if let Some(ss) = self.slots[slot]
                .study
                .as_mut()
                .and_then(|st| st.structure_sets.get_mut(i))
            {
                ss.locked = !ss.locked;
            }
            self.settings_gen += 1;
        }
        if let Some((act, i)) = poi_act {
            match act {
                PoiAct::Localize => {
                    self.localize_roi(slot, i);
                }
                PoiAct::MoveHere => {
                    self.move_poi_to_crosshair(slot, i);
                }
                PoiAct::Localization => self.set_localization(slot, i),
                PoiAct::AtCentre => {
                    if self.poi_at_structure(slot, i).is_none() {
                        self.notice =
                            Some("That structure has no geometry to take a centre of.".into());
                    }
                }
            }
        }
        if update_derived {
            self.update_all_derived(slot);
        }
        if set_act.is_some() {
            self.set_action = set_act;
        }
        if item_act.is_some() {
            self.item_action = item_act;
        }
    }
}
