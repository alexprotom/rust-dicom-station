//! The two side panels and their sections.
//!
//! The left panel is the data tree and nothing else: one section per kind of
//! loaded object - series, structures, segmentations, dose, plan, planar
//! images, registrations, records - under one node per workspace.
//!
//! The right panel holds the modules, each one turned on and off in the
//! Modules menu: image registration and image simulation (both in
//! `reg_panel.rs`) and structures propagation (`propagate_win.rs`).
//!
//! Both panels are built the same way, by [`ViewerApp::edge_panel`]: a thin
//! strip along the window edge carrying the show / hide arrow, and beside it
//! the panel itself, which can also be dragged shut and pulled back open.
//! F9 works the left one, F10 the right.

use super::*;

/// What tells one edge panel from the other.
struct EdgePanel {
    /// Keys the panel, its strip and its remembered width.
    id: &'static str,
    /// Which edge it lives on.
    left: bool,
    /// Width it opens at the first time.
    width: f32,
    /// What the hover text calls it.
    name: &'static str,
    /// The shortcut the hover text names.
    key: &'static str,
}

impl ViewerApp {
    /// One edge panel: the strip with the show / hide arrow, and the panel
    /// beside it.
    ///
    /// The strip stays there when the panel is gone - otherwise nothing on
    /// screen would say how to get it back, and only the shortcut would.
    ///
    /// `open` is the flag the caller keeps: `show_collapsible` wants it by
    /// reference, which the body's `&mut self` cannot share, so it travels
    /// in and out through the return value.
    fn edge_panel(
        ui: &mut egui::Ui,
        p: EdgePanel,
        open: bool,
        body: impl FnOnce(&mut egui::Ui),
    ) -> bool {
        let EdgePanel {
            id,
            left,
            width,
            name,
            key,
        } = p;
        let mut open = open;
        let strip = egui::Frame::new()
            .fill(ui.visuals().panel_fill)
            .inner_margin(egui::Margin::symmetric(1, 4));
        // The arrow always points the way the panel would move.
        let glyph = match (left, open) {
            (true, true) | (false, false) => "◀",
            _ => "▶",
        };
        let hint = if open {
            format!("Hide the {name} panel ({key})")
        } else {
            format!("Show the {name} panel ({key})")
        };
        let toggle = egui::Id::new((id, "toggle"));
        let strip_panel = if left {
            egui::Panel::left(toggle)
        } else {
            egui::Panel::right(toggle)
        };
        strip_panel
            .exact_size(22.0)
            .resizable(false)
            .frame(strip)
            .show(ui, |ui| {
                if ui
                    .add(
                        egui::Button::new(glyph)
                            .frame(false)
                            .min_size(egui::vec2(20.0, 22.0)),
                    )
                    .on_hover_text(hint)
                    .clicked()
                {
                    open = !open;
                }
            });
        let panel = if left {
            egui::Panel::left(egui::Id::new(id))
        } else {
            egui::Panel::right(egui::Id::new(id))
        };
        panel
            .resizable(true)
            .default_size(width)
            .show_collapsible(ui, &mut open, |ui| {
                egui::ScrollArea::vertical().show(ui, body);
            });
        open
    }

    // -- Left panel: the data tree ----------------------------------------
    pub(super) fn side_panel(&mut self, ui: &mut egui::Ui) {
        if !self.slots.iter().any(|s| s.study.is_some()) {
            return;
        }
        self.side_open = Self::edge_panel(
            ui,
            EdgePanel {
                id: "side",
                left: true,
                width: 280.0,
                name: "data tree",
                key: "F9",
            },
            self.side_open,
            |ui| {
                for slot in self.open_slots() {
                    if self.slots[slot].study.is_none() {
                        continue;
                    }
                    self.study_section(ui, slot);
                }
            },
        );
    }

    // -- Right panel: the modules -----------------------------------------
    /// Every module the Modules menu has turned on, one section each. The
    /// panel is not built at all while they are all off, so an untouched
    /// installation looks exactly as it did before there was one.
    pub(super) fn modules_panel(&mut self, ui: &mut egui::Ui) {
        /// The id salt of every module header (its title unless it sets
        /// one) and the settings word for it.
        const MODULE_HEADERS: [(&str, &str); 9] = [
            ("Image information", "information"),
            ("Playback", "playback"),
            ("Image registration", "registration"),
            ("Image simulation", "simulation"),
            ("Image generation", "generation"),
            ("Structure editor", "editor"),
            ("Structure auto tools", "auto"),
            ("module_propagate", "propagation"),
            ("Dose estimation", "dose"),
        ];
        if !self.slots.iter().any(|s| s.study.is_some()) {
            return;
        }
        if !self.any_module() {
            return;
        }
        self.tools.visible = false;
        // The MedSAM2 box is drawn in the views only while its section is
        // on screen; the section sets this again every frame it is drawn.
        self.medsam2.open = false;
        self.nni.open = false;
        self.right_open = Self::edge_panel(
            ui,
            EdgePanel {
                id: "modules",
                left: false,
                width: 320.0,
                name: "modules",
                key: "F10",
            },
            self.right_open,
            |ui| {
                // Every section starts folded. *Restore the last session*
                // unfolds the ones that were open at the last run; from
                // then on the list follows the headers and is written
                // with the settings.
                let ids: Vec<(egui::Id, &str)> = MODULE_HEADERS
                    .iter()
                    .map(|(salt, key)| (ui.make_persistent_id(*salt), *key))
                    .collect();
                if self.apply_modules_open {
                    self.apply_modules_open = false;
                    self.modules_tracked = true;
                    for (id, key) in &ids {
                        let mut st =
                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(),
                                *id,
                                false,
                            );
                        st.set_open(self.modules_open.iter().any(|k| k == key));
                        st.store(ui.ctx());
                    }
                }
                if self.module_info {
                    self.image_info_section(ui);
                }
                if self.module_play {
                    self.playback_section(ui);
                }
                if self.module_registration {
                    self.registration_section(ui);
                }
                if self.module_simulation {
                    self.simulation_section(ui);
                }
                if self.module_generation {
                    self.image_generation_section(ui);
                }
                if self.module_structures {
                    self.structures_editor_section(ui);
                }
                if self.module_auto {
                    self.auto_tools_section(ui);
                }
                if self.module_propagation {
                    self.propagate_section(ui);
                }
                if self.module_dose {
                    self.dose_est_section(ui);
                }
                let open: Vec<String> = ids
                    .iter()
                    .filter(|(id, _)| {
                        egui::collapsing_header::CollapsingState::load(ui.ctx(), *id)
                            .is_some_and(|s| s.is_open())
                    })
                    .map(|(_, key)| key.to_string())
                    .collect();
                // Until a section has been unfolded in this run (or the
                // last run's list applied), the list on disk is the last
                // run's and stays as it is for *Restore the last session*.
                if !open.is_empty() {
                    self.modules_tracked = true;
                }
                if self.modules_tracked && open != self.modules_open {
                    self.modules_open = open;
                    self.persist_settings();
                }
            },
        );
        // The dashed interpolation preview belongs to the Edit section;
        // with the section out of sight nothing says what it is.
        if !self.tools.visible && self.interp.is_some() {
            self.interp = None;
        }
    }

    /// A tree node whose title wraps over as many lines as it needs.
    ///
    /// [`egui::CollapsingHeader`] lays its title out on a single line and
    /// lets it run past the panel edge, so a long patient name, a long
    /// study description or a long ID would pin the panel open at that
    /// width. Only the module headers keep that one-line behaviour.
    ///
    /// Returns the title's own response, so the caller can hang the node's
    /// context menu on it.
    pub(super) fn wrapped_node<R>(
        ui: &mut egui::Ui,
        id_salt: impl std::hash::Hash + std::fmt::Debug,
        default_open: bool,
        title: impl Into<String>,
        body: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::Response {
        Self::wrapped_node_extra(ui, id_salt, default_open, title, |_| {}, body)
    }

    /// The same node with room for a widget or two after the title.
    ///
    /// What a section *makes* belongs on the section's own line: a "+" that
    /// adds a set reads as part of the heading, where a full-width button
    /// under it read as the first row of the list.
    ///
    /// `header_extra` and `body` are separate closures called at different
    /// times, so anything they both report has to travel in separate
    /// variables and be merged by the caller.
    pub(super) fn wrapped_node_extra<R>(
        ui: &mut egui::Ui,
        id_salt: impl std::hash::Hash + std::fmt::Debug,
        default_open: bool,
        title: impl Into<String>,
        header_extra: impl FnOnce(&mut egui::Ui),
        body: impl FnOnce(&mut egui::Ui) -> R,
    ) -> egui::Response {
        let id = ui.make_persistent_id(id_salt);
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            id,
            default_open,
        );
        let (_, header, _) = state
            .show_header(ui, |ui| {
                let resp = ui.add(
                    egui::Label::new(
                        egui::RichText::new(title.into()).text_style(egui::TextStyle::Button),
                    )
                    .wrap()
                    .sense(egui::Sense::click()),
                );
                header_extra(ui);
                resp
            })
            .body(body);
        let resp = header.inner;
        // Clicking the title opens and closes the node, as it does on a
        // standard header; the arrow keeps working on its own.
        if resp.clicked() {
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(),
                id,
                default_open,
            );
            state.toggle(ui);
            state.store(ui.ctx());
        }
        resp
    }

    /// Study transform simulator: apply a known rigid motion + optional
    /// Gaussian deformation to a study and generate the result into the
    /// other slot (the generated study is exportable via *File ▶ Export*).
    /// One RT-object series row of the data tree.
    ///
    /// The same shape as an image series row: a selectable label and nothing
    /// in front of it. The views draw one set of each kind at a time, so
    /// selection and visibility are one thing - clicking another row makes it
    /// the shown one, and clicking the shown row again hides it.
    ///
    /// The highlight therefore means *drawn in the views*, not *selected*: an
    /// active set that has been hidden goes back to looking like any other
    /// row, weakly. Nothing is lost by that, because which set is the active
    /// one is written underneath it - its structures, and the buttons that
    /// act on them, are listed under that row and no other.
    pub(super) fn series_row(
        ui: &mut egui::Ui,
        active: bool,
        shown: bool,
        title: String,
        focused: bool,
    ) -> egui::Response {
        let text = egui::RichText::new(title);
        let text = if active && !shown { text.weak() } else { text };
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.horizontal(|ui| {
                let resp = ui.add(egui::Button::selectable(active && shown, text).wrap());
                Self::keyboard_badge(ui, focused);
                resp
            })
            .inner
        })
        .inner
    }

    /// The badge on the row the arrow keys are standing on, drawn inline
    /// with the row's name exactly as RTS and SEG are.
    ///
    /// The outline [`focus_row`](Self::focus_row) draws says *this row* but
    /// not *what the keyboard will do here*; the glyph does, and it is the
    /// one mark on the tree that means "typing acts here".
    pub(super) fn keyboard_badge(ui: &mut egui::Ui, focused: bool) {
        if !focused {
            return;
        }
        ui.add(
            egui::Button::new(egui::RichText::new("⌨").small().strong())
                .small()
                .corner_radius(2.0)
                .sense(egui::Sense::hover()),
        )
        .on_hover_text(
            "The keyboard is on this row: ↑ and ↓ step to the row above and below at this \
             level, F2 renames it, Delete removes it",
        );
    }

    /// Note a click on a data-tree row - either button - as the row the
    /// keyboard acts on (F2 renames it, Delete removes it), and outline the
    /// row that has that focus.
    ///
    /// The rows are drawn behind a shared borrow of the study, so the focus
    /// the frame started with comes in as `current` and a new one goes out
    /// through `out`, to be written back once the borrow ends - the same
    /// two-step every other edit the tree makes goes through.
    pub(super) fn focus_row(
        ui: &egui::Ui,
        resp: &egui::Response,
        me: RenameTarget,
        current: &Option<RenameTarget>,
        out: &mut Option<RenameTarget>,
    ) {
        if current.as_ref() == Some(&me) {
            ui.painter().rect_stroke(
                resp.rect.expand(1.0),
                2.0,
                egui::Stroke::new(1.0, ui.visuals().weak_text_color()),
                egui::StrokeKind::Outside,
            );
        }
        if resp.clicked() || resp.secondary_clicked() {
            *out = Some(me);
        }
    }

    /// The image series a set is drawn on, as a tree suffix.
    /// An image-series row: the selectable label, and a small framed badge
    /// for each current selection that lands on this series.
    ///
    /// Used by the plain series rows and by the members of a 4D group, so a
    /// phase says what is drawn on it exactly as a lone series does.
    pub(super) fn badged_series_row(
        ui: &mut egui::Ui,
        selected: bool,
        label: String,
        uid: &str,
        badges: &[(&'static str, Option<String>); 4],
        focused: bool,
    ) -> egui::Response {
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.horizontal(|ui| {
                let resp = ui.add(egui::Button::selectable(selected, label).wrap());
                for (tag, what) in badges {
                    if what.as_deref() != Some(uid) {
                        continue;
                    }
                    ui.add(
                        egui::Button::new(egui::RichText::new(*tag).small().strong())
                            .small()
                            .corner_radius(2.0)
                            .sense(egui::Sense::hover()),
                    )
                    .on_hover_text(Self::badge_hint(tag));
                }
                // Last, so it sits at the end of the row whatever else the
                // selection has put on this series.
                Self::keyboard_badge(ui, focused);
                resp
            })
            .inner
        })
        .inner
    }

    /// Which of RTS / SEG / RTD / RTP the *selected* objects of a workspace
    /// sit on, as series UIDs.
    ///
    /// The badges follow the selection, not the study: they say what the
    /// structure set, segmentation series, dose and plan that are active
    /// right now were drawn on or computed for, so selecting another one
    /// moves the badge to another series. A dose reaches its images the
    /// long way round - through its plan, then that plan's structure set -
    /// and falls back to its frame of reference when a link of that chain
    /// was not loaded.
    pub(super) fn series_badges(&self, slot: usize) -> [(&'static str, Option<String>); 4] {
        let Some(study) = self.slots[slot].study.as_ref() else {
            return [("RTS", None), ("SEG", None), ("RTD", None), ("RTP", None)];
        };
        let set_series = |sop: &str| -> Option<String> {
            study
                .structure_sets
                .iter()
                .find(|ss| ss.sop_instance_uid == sop)
                .map(|ss| ss.referenced_series_uid.clone())
                .filter(|u| !u.is_empty())
        };
        let rts = study
            .structure_sets
            .get(self.slots[slot].active_structs)
            .map(|ss| ss.referenced_series_uid.clone())
            .filter(|u| !u.is_empty());
        let seg = self.slots[slot]
            .seg_series_idx()
            .and_then(|i| study.seg_series.get(i))
            .map(|ser| ser.referenced_series_uid.clone())
            .filter(|u| !u.is_empty());
        let plan = study.plans.get(self.slots[slot].active_plan);
        let rtp = plan.and_then(|p| set_series(&p.referenced_structset_uid));
        let rtd = study.doses.get(self.slots[slot].active_dose).and_then(|d| {
            study
                .plans
                .iter()
                .find(|p| p.sop_instance_uid == d.referenced_plan_uid)
                .and_then(|p| set_series(&p.referenced_structset_uid))
            // No plan loaded, or a plan whose structure set is not here:
            // nothing in the files says which images this grid was computed
            // on, so nothing is claimed.
        });
        [("RTS", rts), ("SEG", seg), ("RTD", rtd), ("RTP", rtp)]
    }

    /// What a badge on a series row stands for.
    fn badge_hint(tag: &str) -> &'static str {
        match tag {
            "RTS" => "The selected RT structure set was drawn on this series",
            "SEG" => "The selected segmentation series belongs to this series",
            "RTD" => "The selected dose was computed on this series",
            _ => "The selected plan was made on this series",
        }
    }

    /// "#N" for a series whose description another series here also carries,
    /// counting from 0 in tree order; nothing when the name is unique.
    ///
    /// Two series called *CT 4DCT* are two different acquisitions, and the
    /// tree is where one is picked: a row that cannot be told from its
    /// neighbour is a row that cannot be picked on purpose.
    pub(super) fn name_ordinal(study: &LoadedStudy, idx: usize) -> String {
        let Some(me) = study.series.get(idx) else {
            return String::new();
        };
        let same = |o: &loader::SeriesInfo| o.description == me.description;
        if study.series.iter().filter(|o| same(o)).count() < 2 {
            return String::new();
        }
        let n = study.series.iter().take(idx).filter(|o| same(o)).count();
        format!(" #{n}")
    }

    pub(super) fn series_suffix(study: &LoadedStudy, uid: &str) -> String {
        if uid.is_empty() {
            return " ▶ (any image of this frame)".to_string();
        }
        study
            .series
            .iter()
            .find(|se| se.uid == uid)
            .map(|se| {
                format!(
                    " ▶ {} {}",
                    se.modality,
                    if se.description.is_empty() {
                        "series"
                    } else {
                        &se.description
                    }
                )
            })
            .unwrap_or_else(|| " ▶ (image series not loaded)".to_string())
    }

    /// The *Copy to / Move to / Remove / Export* buttons that act on whatever
    /// is ticked, added inline so they share one row with *All* / *None* - a
    /// multi-item action should not have to be found by right-clicking
    /// exactly the right row.
    pub(super) fn selection_buttons(
        &self,
        ui: &mut egui::Ui,
        here: SetRef,
        selection: &[usize],
        item_act: &mut Option<ItemAction>,
    ) {
        let n = selection.len();
        let what = format!("{n} ticked {}", here.kind.item_name(n));
        ui.add_enabled_ui(n > 0, |ui| {
            ui.menu_button("Copy to", |ui| {
                if let Some(to) = self.destination_menu(ui, here) {
                    *item_act = Some(to.transfer(here, selection.to_vec(), true));
                }
            })
            .response
            .on_hover_text(format!("Copy the {what} into another series"));
            ui.menu_button("Move to", |ui| {
                if let Some(to) = self.destination_menu(ui, here) {
                    *item_act = Some(to.transfer(here, selection.to_vec(), false));
                }
            })
            .response
            .on_hover_text(format!("Move the {what} into another series"));
            if small_tip_button(ui, "🗑", format!("Remove the {what}")) {
                *item_act = Some(ItemAction::Remove {
                    from: here,
                    items: selection.to_vec(),
                });
            }
            if here.kind == SetKind::Segmentations
                && ui
                    .small_button("💾")
                    .on_hover_text(format!("Write the {what} as a DICOM SEG file of their own"))
                    .clicked()
            {
                *item_act = Some(ItemAction::ExportSeg {
                    from: here,
                    items: selection.to_vec(),
                });
            }
        });
        ui.weak(format!("{n} selected"));
    }
}

/// Apply a check-box click to a visibility/selection list, extending from
/// `anchor` when Shift is held, and return the new anchor.
///
/// egui has already toggled `vis[i]` by the time this runs, so the clicked
/// row's *new* value is what the span is filled with: tick one row and
/// Shift-tick a later one and everything between turns on; untick and
/// Shift-untick and it all turns off. Rows outside the span are never
/// touched - the box is a visibility toggle as much as a selection, and
/// silently hiding structures the user did not point at would be worse than
/// any convenience gained.
pub(super) fn apply_tick(vis: &mut [bool], i: usize, shift: bool, anchor: Option<usize>) -> usize {
    if let (true, Some(a)) = (shift, anchor) {
        if a < vis.len() && i < vis.len() {
            let v = vis[i];
            let (lo, hi) = if a <= i { (a, i) } else { (i, a) };
            vis[lo..=hi].iter_mut().for_each(|x| *x = v);
        }
    }
    i
}

#[cfg(test)]
mod tick_tests {
    use super::apply_tick;

    /// Without Shift a click is what egui already did to that one row.
    #[test]
    fn a_plain_click_touches_only_its_own_row() {
        let mut v = vec![false, true, false, false];
        assert_eq!(apply_tick(&mut v, 2, false, Some(0)), 2, "anchor moves");
        assert_eq!(v, vec![false, true, false, false]);
    }

    /// Shift fills the span with the clicked row's new value, in either
    /// direction, and leaves everything outside it alone.
    #[test]
    fn shift_fills_the_span_from_the_anchor() {
        let mut v = vec![false; 6];
        v[4] = true; // egui toggled the clicked row on
        assert_eq!(apply_tick(&mut v, 4, true, Some(1)), 4);
        assert_eq!(v, vec![false, true, true, true, true, false]);

        // Backwards, and unticking: the span follows the clicked row's value.
        let mut v = vec![true; 6];
        v[1] = false;
        apply_tick(&mut v, 1, true, Some(3));
        assert_eq!(v, vec![true, false, false, false, true, true]);
    }

    /// A stale anchor (the list shrank, or it belongs to nothing) must not
    /// panic or reach outside the list.
    #[test]
    fn a_stale_anchor_is_ignored() {
        let mut v = vec![false, true];
        assert_eq!(apply_tick(&mut v, 1, true, Some(9)), 1);
        assert_eq!(v, vec![false, true]);
        let mut v = vec![false, true];
        assert_eq!(apply_tick(&mut v, 1, true, None), 1);
        assert_eq!(v, vec![false, true]);
    }
}
