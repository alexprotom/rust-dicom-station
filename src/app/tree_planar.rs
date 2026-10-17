//! The data tree's planar images section: radiographs, RT images and DRRs,
//! and the panes they are shown in.

use super::*;

impl ViewerApp {
    /// DX / CR / RTIMAGE planar images, DRRs among them, drawn like image
    /// series: a click shows the image in a planar pane of the row, and a
    /// `P1` / `P2` badge says which pane holds it.
    pub(super) fn planar_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.planar_images.len())
            .unwrap_or(0);
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
        let panes: Vec<Vec<usize>> = (0..n).map(|i| self.planar_panes_showing(slot, i)).collect();
        let ctrl = ui.input(|i| i.modifiers.command);
        // Normally a side note beneath the tree; for a workspace with no image
        // volume these *are* the images, so the section opens itself.
        let sole_content = !self.slots[slot].has_volume();
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            egui::CollapsingHeader::new(format!("Planar images ({n})"))
                .id_salt(("planar", slot))
                .default_open(sole_content)
                .show(ui, |ui| {
                    for (i, img) in study.planar_images.iter().enumerate() {
                        let me = RenameTarget::Planar { slot, idx: i };
                        let in_panes = &panes[i];
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
                                        ui.add(
                                            egui::Button::new(
                                                egui::RichText::new(format!("P{}", k + 1))
                                                    .small()
                                                    .strong(),
                                            )
                                            .small()
                                            .corner_radius(2.0)
                                            .sense(egui::Sense::hover()),
                                        )
                                        .on_hover_text(
                                            format!(
                                                "Shown in {}",
                                                PaneKind::Planar(*k as u8).label()
                                            ),
                                        );
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
                            for k in 0..settings::PLANAR_PANES {
                                let label = PaneKind::Planar(k as u8).label();
                                if ui.button(format!("👁 Show in {label}")).clicked() {
                                    act = Some(Act::Show(i, Some(k)));
                                    ui.close();
                                }
                            }
                            ui.separator();
                            for k in 0..settings::PLANAR_PANES {
                                let label = PaneKind::Planar(k as u8).label();
                                if in_panes.contains(&k) {
                                    if ui.button(format!("➖ Take out of {label}")).clicked() {
                                        act = Some(Act::Take(i, k));
                                        ui.close();
                                    }
                                } else if ui
                                    .button(format!("➕ Compare in {label}"))
                                    .on_hover_text("Put it on top of what the pane shows")
                                    .clicked()
                                {
                                    act = Some(Act::Stack(i, Some(k)));
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
            if self
                .maximized
                .is_some_and(|(s, k)| s != slot || !matches!(k, PaneKind::Planar(_)))
            {
                self.maximized = None;
            }
        }
    }
}
