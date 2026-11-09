//! The data tree's *Planar structures* section: the workspace's planar
//! structure sets ([`crate::planar_structs`]) under the planar images of
//! their study, each set with its structures.
//!
//! A set row unfolds into its structures and carries the set's menu
//! (rename, lock, project structures into it, save it as a presentation
//! state, remove); a structure row has the tick that shows it, the colour
//! swatch, its name - a click makes it the structure the 2D contour tools
//! edit and puts one of its images on screen - and its own menu (rename,
//! carry it through a 2D/2D registration, remove). The row's tooltip names
//! each contour's image and where the contour came from.

use super::*;

use super::tree_model::StudyNode;
use crate::planar_structs::PoseChoice;

/// What a click in the section asks for, applied after the rows are drawn.
enum Act {
    Visible(usize, usize, bool),
    AllVisible(usize, bool),
    Color(usize, usize, [u8; 3]),
    Activate(usize, usize),
    Lock(usize, bool),
    RemoveSet(usize),
    RemoveStructure(usize, usize),
    Project(usize),
    Save(usize),
    Carry(usize, usize),
}

impl ViewerApp {
    /// The sets of one study node: those with a contour on one of its
    /// images, and those whose images are none of them loaded but whose
    /// study it is.
    fn planar_sets_of(&self, slot: usize, node: &StudyNode) -> Vec<usize> {
        let Some(study) = &self.slots[slot].study else {
            return Vec::new();
        };
        let uids: Vec<&str> = node
            .planar
            .iter()
            .filter_map(|&i| study.planar_images.get(i))
            .map(|p| p.sop_instance_uid.as_str())
            .filter(|u| !u.is_empty())
            .collect();
        study
            .planar_structs
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.images.iter().any(|u| uids.contains(&u.as_str()))
                    || (s.images.iter().all(|u| study.planar_by_uid(u).is_none())
                        && (s.study_uid == node.uid
                            || (s.study_uid.is_empty() && !node.planar.is_empty())))
            })
            .map(|(k, _)| k)
            .collect()
    }

    /// *Planar structures (n)* under one study's planar images.
    pub(super) fn planar_structs_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        (pi, si): (usize, usize),
        node: &StudyNode,
    ) {
        let sets = self.planar_sets_of(slot, node);
        if sets.is_empty() {
            return;
        }
        let mut act: Option<Act> = None;
        let mut rename: Option<RenameTarget> = None;
        let active = self.slots[slot].active_planar;
        {
            let Some(study) = &self.slots[slot].study else {
                return;
            };
            egui::CollapsingHeader::new(format!("Planar structures ({})", sets.len()))
                .id_salt(("planar_structs", slot, pi, si))
                .default_open(true)
                .show(ui, |ui| {
                    for &k in &sets {
                        let set = &study.planar_structs[k];
                        let missing = set.missing_images(study);
                        let title = format!(
                            "{}{} ({})",
                            if set.locked { "🔒 " } else { "" },
                            set.label,
                            set.structures.len()
                        );
                        let header = egui::CollapsingHeader::new(title)
                            .id_salt(("planar_set", slot, k, set_key_salt(set)))
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.small_button("All").clicked() {
                                        act = Some(Act::AllVisible(k, true));
                                    }
                                    if ui.small_button("None").clicked() {
                                        act = Some(Act::AllVisible(k, false));
                                    }
                                });
                                for (j, s) in set.structures.iter().enumerate() {
                                    ui.horizontal(|ui| {
                                        let mut color = s.color;
                                        if color_swatch(ui, &mut color) {
                                            act = Some(Act::Color(k, j, color));
                                        }
                                        let mut on = self.planar_visible(slot, set, &s.name);
                                        if ui.checkbox(&mut on, "").changed() {
                                            act = Some(Act::Visible(k, j, on));
                                        }
                                        let mut tip = String::new();
                                        for c in &s.contours {
                                            let img = study
                                                .planar_by_uid(&c.image_uid)
                                                .map(|i| study.planar_images[i].label.clone())
                                                .unwrap_or_else(|| "image not loaded".into());
                                            tip.push_str(&format!(
                                                "{img}: {} ring(s), {}\n",
                                                c.region.rings.len(),
                                                c.source.describe()
                                            ));
                                        }
                                        if !s.points.is_empty() {
                                            tip.push_str(&format!("{} point(s)\n", s.points.len()));
                                        }
                                        if tip.is_empty() {
                                            tip = "no contour yet".into();
                                        }
                                        let resp = ui
                                            .add(
                                                egui::Button::selectable(
                                                    active == Some((k, j)),
                                                    s.name.as_str(),
                                                )
                                                .wrap(),
                                            )
                                            .on_hover_text(tip.trim_end());
                                        if resp.clicked() {
                                            act = Some(Act::Activate(k, j));
                                        }
                                        resp.context_menu(|ui| {
                                            if Self::rename_entry(ui) {
                                                rename = Some(RenameTarget::PlanarStructure {
                                                    slot,
                                                    set: k,
                                                    idx: j,
                                                });
                                                ui.close();
                                            }
                                            if ui
                                                .button("⇥ Carry to...")
                                                .on_hover_text(
                                                    "Through a 2D/2D registration of the \
                                                     workspace, onto its fixed image",
                                                )
                                                .clicked()
                                            {
                                                act = Some(Act::Carry(k, j));
                                                ui.close();
                                            }
                                            ui.separator();
                                            if Self::remove_entry(ui, "🗑 Remove") {
                                                act = Some(Act::RemoveStructure(k, j));
                                                ui.close();
                                            }
                                        });
                                    });
                                }
                                if !missing.is_empty() {
                                    ui.weak(format!("{} image(s) not loaded", missing.len()));
                                }
                            });
                        header.header_response.context_menu(|ui| {
                            if Self::rename_entry(ui) {
                                rename = Some(RenameTarget::PlanarSet { slot, idx: k });
                                ui.close();
                            }
                            let lock = if set.locked {
                                "🔓 Unlock"
                            } else {
                                "🔒 Lock"
                            };
                            if ui.button(lock).clicked() {
                                act = Some(Act::Lock(k, !set.locked));
                                ui.close();
                            }
                            if ui
                                .add_enabled(!set.locked, egui::Button::new("⇩ Project structures"))
                                .on_hover_text("Project 3D structures into this set")
                                .clicked()
                            {
                                act = Some(Act::Project(k));
                                ui.close();
                            }
                            if ui
                                .button("💾 Save as presentation state")
                                .on_hover_text(
                                    "A Grayscale Softcopy Presentation State, with every \
                                     image made here as an RT Image beside it",
                                )
                                .clicked()
                            {
                                act = Some(Act::Save(k));
                                ui.close();
                            }
                            ui.separator();
                            if Self::remove_entry(ui, "🗑 Remove") {
                                act = Some(Act::RemoveSet(k));
                                ui.close();
                            }
                        });
                    }
                });
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        let Some(a) = act else { return };
        self.apply_planar_act(slot, a);
    }

    fn apply_planar_act(&mut self, slot: usize, a: Act) {
        match a {
            Act::Visible(k, j, on) => {
                let name = self.planar_structure_name(slot, k, j);
                if let Some(name) = name {
                    self.set_planar_visible(slot, k, &name, on);
                }
            }
            Act::AllVisible(k, on) => {
                let names: Vec<String> = self.slots[slot]
                    .study
                    .as_ref()
                    .and_then(|st| st.planar_structs.get(k))
                    .map(|s| s.structures.iter().map(|x| x.name.clone()).collect())
                    .unwrap_or_default();
                for n in names {
                    self.set_planar_visible(slot, k, &n, on);
                }
            }
            Act::Color(k, j, c) => {
                if let Some(s) = self.planar_structure_mut(slot, k, j) {
                    s.color = c;
                }
            }
            Act::Activate(k, j) => {
                self.slots[slot].active_planar = Some((k, j));
                // One of its images on screen, when none shows.
                let image = self.slots[slot].study.as_ref().and_then(|st| {
                    let s = st.planar_structs.get(k)?.structures.get(j)?;
                    let imgs: Vec<usize> = s
                        .contours
                        .iter()
                        .filter_map(|c| st.planar_by_uid(&c.image_uid))
                        .collect();
                    Some(imgs)
                });
                if let Some(imgs) = image {
                    let shown = imgs
                        .iter()
                        .any(|&i| !self.planar_panes_showing(slot, i).is_empty());
                    if let (false, Some(&i)) = (shown, imgs.first()) {
                        self.show_planar(slot, i, None);
                        self.planar_on_screen(slot);
                    }
                }
            }
            Act::Lock(k, on) => {
                if let Some(s) = self.slots[slot]
                    .study
                    .as_mut()
                    .and_then(|st| st.planar_structs.get_mut(k))
                {
                    s.locked = on;
                }
            }
            Act::RemoveSet(k) => {
                if let Some(st) = self.slots[slot].study.as_mut() {
                    if k < st.planar_structs.len() {
                        st.planar_structs.remove(k);
                    }
                }
                self.slots[slot].active_planar = match self.slots[slot].active_planar {
                    Some((s, _)) if s == k => None,
                    Some((s, j)) if s > k => Some((s - 1, j)),
                    a => a,
                };
                self.settings_gen += 1;
            }
            Act::RemoveStructure(k, j) => {
                if let Some(set) = self.slots[slot]
                    .study
                    .as_mut()
                    .and_then(|st| st.planar_structs.get_mut(k))
                {
                    if j < set.structures.len() && !set.locked {
                        set.structures.remove(j);
                        set.refresh_images();
                    }
                }
                self.slots[slot].active_planar = match self.slots[slot].active_planar {
                    Some((s, i)) if s == k && i == j => None,
                    Some((s, i)) if s == k && i > j => Some((s, i - 1)),
                    a => a,
                };
                self.settings_gen += 1;
            }
            Act::Project(k) => {
                let images: Vec<usize> = self.slots[slot]
                    .study
                    .as_ref()
                    .and_then(|st| {
                        let set = st.planar_structs.get(k)?;
                        Some(
                            set.images
                                .iter()
                                .filter_map(|u| st.planar_by_uid(u))
                                .collect(),
                        )
                    })
                    .unwrap_or_default();
                self.open_project_prompt(slot, &images, PoseChoice::AsIs, Some(k));
            }
            Act::Save(k) => self.save_planar_set(slot, k),
            Act::Carry(k, j) => self.open_carry(slot, k, j),
        }
    }

    pub(super) fn planar_structure_name(&self, slot: usize, k: usize, j: usize) -> Option<String> {
        self.slots[slot]
            .study
            .as_ref()?
            .planar_structs
            .get(k)?
            .structures
            .get(j)
            .map(|s| s.name.clone())
    }

    pub(super) fn planar_structure_mut(
        &mut self,
        slot: usize,
        k: usize,
        j: usize,
    ) -> Option<&mut crate::planar_structs::PlanarStructure> {
        self.slots[slot]
            .study
            .as_mut()?
            .planar_structs
            .get_mut(k)?
            .structures
            .get_mut(j)
    }
}

/// A salt that keeps a set's fold state when the sets before it go.
fn set_key_salt(set: &crate::planar_structs::PlanarStructureSet) -> String {
    super::planar_contours::set_key(set)
}

/// One row of *2D registrations*: the registration, its label and line,
/// the fixed image and the moved copy (indices into `planar_images`).
type RegRow = (usize, String, String, Option<usize>, Option<usize>);

impl ViewerApp {
    /// *2D registrations (n)* under one study's planar images: the 2D/2D
    /// registrations whose fixed image is one of them. A click shows the
    /// fixed image with the moved copy over it and makes it the 2D/2D
    /// section's current one.
    pub(super) fn planar_regs_section(
        &mut self,
        ui: &mut egui::Ui,
        slot: usize,
        (pi, si): (usize, usize),
        node: &StudyNode,
    ) {
        let rows: Vec<RegRow> = {
            let Some(study) = &self.slots[slot].study else {
                return;
            };
            study
                .planar_regs
                .iter()
                .enumerate()
                .filter_map(|(k, r)| {
                    let fi = study.planar_by_uid(&r.fixed_uid)?;
                    node.planar.contains(&fi).then(|| {
                        (
                            k,
                            r.label.clone(),
                            r.result.line(),
                            Some(fi),
                            r.moved_copy_uid
                                .as_ref()
                                .and_then(|u| study.planar_by_uid(u)),
                        )
                    })
                })
                .collect()
        };
        if rows.is_empty() {
            return;
        }
        let mut show = None;
        let mut remove = None;
        let mut rename = None;
        egui::CollapsingHeader::new(format!("2D registrations ({})", rows.len()))
            .id_salt(("planar_regs", slot, pi, si))
            .default_open(false)
            .show(ui, |ui| {
                for (k, label, line, fi, copy) in &rows {
                    let resp = ui
                        .add(egui::Button::selectable(false, label.as_str()).wrap())
                        .on_hover_text(line.as_str());
                    if resp.clicked() {
                        show = Some((*k, *fi, *copy));
                    }
                    resp.context_menu(|ui| {
                        if Self::rename_entry(ui) {
                            rename = Some(RenameTarget::PlanarReg { slot, idx: *k });
                            ui.close();
                        }
                        ui.separator();
                        if Self::remove_entry(ui, "🗑 Remove") {
                            remove = Some((*k, *copy));
                            ui.close();
                        }
                    });
                }
            });
        if rename.is_some() {
            self.rename_request = rename;
        }
        if let Some((k, fi, copy)) = show {
            if let Some(f) = fi {
                self.show_planar(slot, f, None);
                if let Some(c) = copy {
                    self.stack_planar(slot, c, None);
                }
                self.planar_on_screen(slot);
            }
            if let Some(st) = self.planar2d.reg.as_mut().filter(|st| st.slot == slot) {
                st.current = Some(k);
            }
        }
        if let Some((k, copy)) = remove {
            match copy {
                // The copy goes, and the registration with it.
                Some(c) => {
                    self.obj_remove = Some(ObjRef {
                        slot,
                        kind: ObjKind::Planar,
                        idx: c,
                    });
                }
                None => {
                    if let Some(st) = self.slots[slot].study.as_mut() {
                        if k < st.planar_regs.len() {
                            st.planar_regs.remove(k);
                        }
                    }
                    self.reg2d_removed(slot, k);
                }
            }
            self.settings_gen += 1;
        }
    }
}
