//! The data tree's planar images section (radiographs, RT images and DRRs)
//! and its *Additional* section (secondary captures, a scanner's protocol
//! and dose report pages, scanned documents: objects with pixels that are
//! no image of the patient), and the panes they are shown in.

use super::*;

use crate::extras::PlanarKind;

impl ViewerApp {
    /// Both sections: the projections, then the additional objects, each
    /// only when the workspace holds some.
    pub(super) fn planar_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        self.planar_kind_section(ui, slot, PlanarKind::Projection);
        self.planar_kind_section(ui, slot, PlanarKind::Additional);
    }

    /// The images of one kind, drawn like image series: a click shows the
    /// image in a pane of the row (a planar pane for a radiograph, the
    /// additional pane for the rest), and a `P1` / `P2` / `A` badge says
    /// which pane holds it.
    fn planar_kind_section(&mut self, ui: &mut egui::Ui, slot: usize, kind: PlanarKind) {
        let images = self.planar_images_of(slot, kind);
        let n = images.len();
        if n == 0 {
            return;
        }
        #[derive(Clone, Copy)]
        enum Act {
            Show(usize, Option<usize>),
            Stack(usize, Option<usize>),
            Take(usize, usize),
        }
        let mut act: Option<Act> = None;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        let current = self.tree_focus.clone();
        let panes: Vec<Vec<usize>> = images
            .iter()
            .map(|i| self.planar_panes_showing(slot, *i))
            .collect();
        let ctrl = ui.input(|i| i.modifiers.command);
        // Normally a side note beneath the tree; for a workspace with no image
        // volume these *are* the images, so the section opens itself (the
        // additional one only when there is no radiograph either).
        let sole_content = !self.slots[slot].has_volume()
            && (kind == PlanarKind::Projection
                || !self.slots[slot]
                    .study
                    .as_ref()
                    .is_some_and(|st| st.has_radiographs()));
        let (title, salt) = match kind {
            PlanarKind::Projection => (format!("Planar images ({n})"), "planar"),
            PlanarKind::Additional => (format!("Additional ({n})"), "additional"),
        };
        // The panes an image of this kind can be shown in, with their
        // badges: the two planar panes, or the additional one.
        let pane_choices: Vec<(usize, &str, PaneKind)> = match kind {
            PlanarKind::Projection => (0..settings::PLANAR_PANES)
                .map(|k| {
                    (
                        k,
                        if k == 0 { "P1" } else { "P2" },
                        PaneKind::Planar(k as u8),
                    )
                })
                .collect(),
            PlanarKind::Additional => {
                vec![(super::planar::ADDITIONAL_PANE, "A", PaneKind::Additional)]
            }
        };
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            egui::CollapsingHeader::new(title)
                .id_salt((salt, slot))
                .default_open(sole_content)
                .show(ui, |ui| {
                    for (row, &i) in images.iter().enumerate() {
                        let img = &study.planar_images[i];
                        let me = RenameTarget::Planar { slot, idx: i };
                        let in_panes = &panes[row];
                        let resp = ui
                            .scope(|ui| {
                                ui.spacing_mut().item_spacing.x = 3.0;
                                ui.horizontal(|ui| {
                                    let resp = ui.add(
                                        egui::Button::selectable(
                                            !in_panes.is_empty(),
                                            img.label.as_str(),
                                        )
                                        .wrap(),
                                    );
                                    for k in in_panes {
                                        let Some((_, badge, pane)) =
                                            pane_choices.iter().find(|(j, _, _)| j == k)
                                        else {
                                            continue;
                                        };
                                        ui.add(
                                            egui::Button::new(
                                                egui::RichText::new(*badge).small().strong(),
                                            )
                                            .small()
                                            .corner_radius(2.0)
                                            .sense(egui::Sense::hover()),
                                        )
                                        .on_hover_text(format!("Shown in {}", pane.label()));
                                    }
                                    Self::keyboard_badge(ui, current.as_ref() == Some(&me));
                                    resp
                                })
                                .inner
                            })
                            .inner;
                        Self::focus_row(ui, &resp, me.clone(), &current, &mut focus);
                        if resp.clicked() {
                            act = Some(if ctrl {
                                Act::Stack(i, None)
                            } else {
                                Act::Show(i, None)
                            });
                        }
                        let mut tip = format!("{} · {} × {} px", img.modality, img.cols, img.rows);
                        for (k, v) in &img.info {
                            tip.push_str(&format!("\n{k}: {v}"));
                        }
                        resp.on_hover_text(tip).context_menu(|ui| {
                            for (k, _, pane) in &pane_choices {
                                let label = pane.label();
                                if ui.button(format!("👁 Show in {label}")).clicked() {
                                    act = Some(Act::Show(i, Some(*k)));
                                    ui.close();
                                }
                            }
                            ui.separator();
                            for (k, _, pane) in &pane_choices {
                                let label = pane.label();
                                if in_panes.contains(k) {
                                    if ui.button(format!("➖ Take out of {label}")).clicked() {
                                        act = Some(Act::Take(i, *k));
                                        ui.close();
                                    }
                                } else if ui
                                    .button(format!("➕ Compare in {label}"))
                                    .on_hover_text("Put it on top of what the pane shows")
                                    .clicked()
                                {
                                    act = Some(Act::Stack(i, Some(*k)));
                                    ui.close();
                                }
                            }
                            ui.separator();
                            if ui.button("✏ Rename").clicked() {
                                rename = Some(RenameTarget::Planar { slot, idx: i });
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("🗑 Remove").clicked() {
                                remove = Some(ObjRef {
                                    slot,
                                    kind: ObjKind::Planar,
                                    idx: i,
                                });
                                ui.close();
                            }
                        });
                    }
                });
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if remove.is_some() {
            self.obj_remove = remove;
        }
        if focus.is_some() {
            self.tree_focus = focus;
        }
        let Some(a) = act else { return };
        match a {
            Act::Show(i, pane) => self.show_planar(slot, i, pane),
            Act::Stack(i, pane) => self.stack_planar(slot, i, pane),
            Act::Take(i, k) => self.take_planar_image(slot, k, i),
        }
        if !matches!(a, Act::Take(..)) {
            // Shown means on screen: a workspace put aside comes back, and a
            // maximized pane of another kind gives way to the row.
            self.shown[slot] = true;
            if self.maximized.is_some_and(|(s, k)| {
                s != slot || !matches!(k, PaneKind::Planar(_) | PaneKind::Additional)
            }) {
                self.maximized = None;
            }
        }
    }
}
