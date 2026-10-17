//! The data tree's segmentations section: the segmentation series of a
//! study and their segments.

use super::panels::apply_tick;
use super::*;

impl ViewerApp {
    /// The segmentation series filed under one study, then the segments of
    /// the active one with the tools that edit them.
    pub(super) fn segmentation_section(
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
        // As in `structures_section`: a tool acts on the series being
        // displayed, so under any other study node it stays disabled rather
        // than running on images nobody pointed at.
        let has_volume = self.slots[slot].has_volume() && displayed;
        // Whichever engine is running on this slot: its glyph, message and
        // fraction, read before the section borrows anything.
        let running = self
            .running_tool(slot)
            .map(|(tool, p)| (tool.glyph, p.get(), p.frac()));
        let active_here = self.slots[slot]
            .seg_series_idx()
            .filter(|i| which.contains(i));
        // (name, colour, visible, cm³) of the active series' segments - the
        // editable columns live on this copy, see `structures_section`.
        let mut rows: Vec<(String, [u8; 3], bool, f64)> = {
            let s = &self.slots[slot];
            let spacing = s.spacing_or_unit();
            match active_here {
                Some(_) => s
                    .segs()
                    .iter()
                    .map(|g| (g.name.clone(), g.color, g.visible, g.volume_cm3(spacing)))
                    .collect(),
                None => Vec::new(),
            }
        };
        let before: Vec<([u8; 3], bool)> = rows.iter().map(|r| (r.1, r.2)).collect();
        let mut new_series = false;
        let mut cancel_tool = false;
        let mut set_all: Option<bool> = None;
        let mut new_active_series: Option<usize> = None;
        let segs_shown = self.slots[slot].segs_shown;
        let mut new_segs_shown: Option<bool> = None;
        let mut activate: Option<usize> = None;
        let mut set_act: Option<SetAction> = None;
        let mut item_act: Option<ItemAction> = None;
        let mut new_anchor: Option<(SetRef, usize)> = None;
        let mut focus: Option<RenameTarget> = None;
        let shift = ui.input(|i| i.modifiers.shift);
        {
            let me = &*self;
            let current = &me.tree_focus;
            let study = me.slots[slot].study.as_ref().unwrap();
            let series = &study.seg_series;
            let active_seg = me.slots[slot].active_seg;
            let n_vis = rows.iter().filter(|r| r.2).count();
            // As above: the heading counts the series, the row counts the
            // segments of the series it names.
            let title = format!("Segmentations ({})", which.len());
            let mut add_series = false;
            Self::wrapped_node_extra(
                ui,
                ("segs", slot, pat, stu),
                !which.is_empty(),
                title,
                |ui| {
                    if tip_widget(
                        ui,
                        has_volume,
                        egui::Button::new("+").small(),
                        if has_volume {
                            "New: an empty segmentation series, drawn on the displayed image \
                             series - exports as one DICOM SEG file"
                        } else if displayed {
                            "This workspace has no image volume to draw on"
                        } else {
                            "Display this study's images first - a new series is drawn on \
                             the series on screen"
                        },
                    ) {
                        add_series = true;
                    }
                },
                |ui| {
                    for &i in which {
                        let Some(sr) = series.get(i) else { continue };
                        let here = SetRef {
                            slot,
                            kind: SetKind::Segmentations,
                            idx: i,
                        };
                        let selected = active_here == Some(i);
                        let resp = Self::series_row(
                            ui,
                            selected,
                            segs_shown,
                            format!(
                                "{} ({}){}",
                                sr.label,
                                if selected {
                                    format!("{n_vis}/{}", sr.segs.len())
                                } else {
                                    format!("{}", sr.segs.len())
                                },
                                Self::series_suffix(study, &sr.referenced_series_uid)
                            ),
                            current.as_ref() == Some(&RenameTarget::Set(here)),
                        );
                        Self::focus_row(ui, &resp, RenameTarget::Set(here), current, &mut focus);
                        if resp.clicked() {
                            if selected {
                                new_segs_shown = Some(!segs_shown);
                            } else {
                                new_active_series = Some(i);
                                new_segs_shown = Some(true);
                            }
                        }
                        resp.context_menu(|ui| me.set_context_menu(ui, here, &mut set_act));
                        resp.on_hover_text(format!(
                            "{}\nclick: show or hide\nright-click: connect to another image \
                         series, copy / move to the other workspace, export as DICOM SEG, remove",
                            if sr.file_name.is_empty() {
                                "created here"
                            } else {
                                &sr.file_name
                            }
                        ));

                        // The segments of a series, and what acts on them, sit
                        // under *that* series rather than at the end of the
                        // list - ten phases of a 4D group otherwise put them a
                        // screen away from the row they belong to.
                        if !selected {
                            continue;
                        }
                        // Masks of a series drawn on another image series are
                        // on that series' lattice, so nothing here can index
                        // them.
                        if !study.has_volume() {
                            ui.weak(
                                "this workspace has no image volume - add the image series these \
                             segments were drawn on to see and edit them",
                            );
                            continue;
                        }
                        if sr.grid.dims != study.volume.dims {
                            ui.weak(
                                "drawn on another image series - display that series to see and \
                             edit these segments",
                            );
                            continue;
                        }
                        if let Some((glyph, msg, frac)) = &running {
                            ui.horizontal(|ui| {
                                ui.label(*glyph);
                                ui.add(
                                    egui::ProgressBar::new(*frac)
                                        .desired_width(120.0)
                                        .show_percentage(),
                                );
                                if ui.small_button("Cancel").clicked() {
                                    cancel_tool = true;
                                }
                            });
                            ui.weak(msg);
                        }
                        let selection: Vec<usize> = rows
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| r.2)
                            .map(|(i, _)| i)
                            .collect();
                        ui.horizontal_wrapped(|ui| {
                            if small_tip_button(ui, "All", "Show and select every segmentation") {
                                set_all = Some(true);
                            }
                            if ui.small_button("None").clicked() {
                                set_all = Some(false);
                            }
                            me.selection_buttons(ui, here, &selection, &mut item_act);
                        });
                        let anchor = me.tick_anchor.filter(|(r, _)| *r == here).map(|(_, i)| i);
                        // The check boxes are edited on `ticks` so a Shift-range
                        // can reach rows the loop has already drawn.
                        let mut ticks: Vec<bool> = rows.iter().map(|r| r.2).collect();
                        for (i, row) in rows.iter_mut().enumerate() {
                            let name = row.0.clone();
                            ui.horizontal(|ui| {
                                color_swatch(ui, &mut row.1);
                                let tick = ui.checkbox(&mut ticks[i], "").on_hover_text(
                                    "Show / select this segmentation\nShift-click: tick or \
                                 untick the whole range from the last one",
                                );
                                if tick.clicked() {
                                    new_anchor =
                                        Some((here, apply_tick(&mut ticks, i, shift, anchor)));
                                }
                                let resp = ui
                                    .add(
                                        egui::Button::selectable(i == active_seg, name.clone())
                                            .wrap(),
                                    )
                                    .on_hover_text(
                                        "Click to make this the segmentation the tools edit\n\
                                         F2: rename, Delete: remove",
                                    );
                                Self::focus_row(
                                    ui,
                                    &resp,
                                    RenameTarget::Item { set: here, idx: i },
                                    current,
                                    &mut focus,
                                );
                                if resp.clicked() {
                                    activate = Some(i);
                                }
                                resp.context_menu(|ui| {
                                    me.item_context_menu(
                                        ui,
                                        here,
                                        i,
                                        &name,
                                        &selection,
                                        &mut item_act,
                                    )
                                });
                                ui.weak(format!("{:.2} cm³", row.3));
                            });
                        }
                        for (row, on) in rows.iter_mut().zip(ticks) {
                            row.2 = on;
                        }
                    }
                },
            );
            if add_series {
                new_series = true;
            }
        }
        if let Some(v) = set_all {
            rows.iter_mut().for_each(|r| r.2 = v);
        }
        if new_anchor.is_some() {
            self.tick_anchor = new_anchor;
        }
        let edited: Vec<(usize, [u8; 3], bool)> = rows
            .iter()
            .enumerate()
            .zip(&before)
            .filter(|((_, r), b)| (r.1, r.2) != **b)
            .map(|((i, r), _)| (i, r.1, r.2))
            .collect();
        if !edited.is_empty() {
            if let Some(segs) = self.slots[slot].segs_mut() {
                for (i, color, visible) in edited {
                    if let Some(seg) = segs.get_mut(i) {
                        seg.color = color;
                        seg.visible = visible;
                    }
                }
            }
        }
        if new_series {
            self.new_set(slot, SetKind::Segmentations);
        }
        if focus.is_some() {
            self.tree_focus = focus;
        }
        if let Some(i) = new_active_series {
            let s = &mut self.slots[slot];
            s.active_seg_series = i;
            s.active_seg = 0;
            self.settings_gen += 1;
        }
        if let Some(on) = new_segs_shown {
            self.slots[slot].segs_shown = on;
        }
        if let Some(i) = activate {
            self.slots[slot].active_seg = i;
        }
        if cancel_tool {
            if let Some((_, p)) = self.running_tool(slot) {
                p.cancel();
            }
        }
        if set_act.is_some() {
            self.set_action = set_act;
        }
        if item_act.is_some() {
            self.item_action = item_act;
        }
    }
}
