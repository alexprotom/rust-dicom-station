//! The data tree's registrations section: the spatial registrations and
//! deformation fields of a study.

use super::*;

impl ViewerApp {
    /// REG spatial registration objects: matrices + apply as active
    /// registration.
    pub(super) fn reg_objects_section(&mut self, ui: &mut egui::Ui, slot: usize) {
        let n = self.slots[slot]
            .study
            .as_ref()
            .map(|s| s.registrations.len())
            .unwrap_or(0);
        if n == 0 {
            return;
        }
        // The workspaces this one can be paired with: a transform read
        // from a file maps one frame of reference onto another, and both
        // ends have to be on screen with an image in them.
        let partners: Vec<usize> = self
            .volume_slot_list()
            .into_iter()
            .filter(|s| *s != slot)
            .collect();

        // (transform, fixed workspace, moving workspace) - with more than
        // two open, which pair a file's transform is meant for is a choice.
        let mut apply: Option<(registration::RigidTransform, usize, usize)> = None;
        let mut apply_grid: Option<(usize, usize, usize)> = None;
        let mut rename: Option<RenameTarget> = None;
        let mut remove: Option<ObjRef> = None;
        let mut focus: Option<RenameTarget> = None;
        {
            let study = self.slots[slot].study.as_ref().unwrap();
            // Frame-of-reference UID of every open workspace's volume, so
            // the hint can name the one a registration actually refers to.
            let fors: Vec<(usize, String)> = self
                .open_slots()
                .into_iter()
                .filter_map(|s| {
                    let uid = self.slots[s]
                        .study
                        .as_ref()?
                        .volume
                        .frame_of_reference_uid
                        .clone();
                    (!uid.is_empty()).then_some((s, uid))
                })
                .collect();
            let hint = |uid: &str| -> String {
                fors.iter()
                    .find(|(_, f)| f == uid)
                    .map(|(s, _)| format!(" (= {})", SLOT_NAMES[*s]))
                    .unwrap_or_default()
            };
            let mut invert = self.reg_apply_invert;
            let current = self.tree_focus.clone();
            egui::CollapsingHeader::new(format!("Spatial registrations ({n})"))
                .id_salt(("regobj", slot))
                .default_open(false)
                .show(ui, |ui| {
                    for (ri, reg) in study.registrations.iter().enumerate() {
                        let resp = ui
                            .label(
                                egui::RichText::new(format!(
                                    "{}{}",
                                    reg.label,
                                    if reg.deformable {
                                        "  [deformable: matrices only]"
                                    } else {
                                        ""
                                    }
                                ))
                                .strong(),
                            )
                            .on_hover_text(
                                "right-click: rename or remove this registration\nF2: rename, \
                                 Delete: remove",
                            );
                        Self::focus_row(
                            ui,
                            &resp,
                            RenameTarget::Registration { slot, idx: ri },
                            &current,
                            &mut focus,
                        );
                        resp.context_menu(|ui| {
                            if ui.button("✏ Rename").clicked() {
                                rename = Some(RenameTarget::Registration { slot, idx: ri });
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("🗑 Remove").clicked() {
                                remove = Some(ObjRef {
                                    slot,
                                    kind: ObjKind::Registration,
                                    idx: ri,
                                });
                                ui.close();
                            }
                        });
                        for (ii, item) in reg.items.iter().enumerate() {
                            if item.is_identity {
                                ui.weak(format!("· item {}: identity ({})", ii + 1, item.matrix_type));
                                continue;
                            }
                            let m = &item.matrix;
                            ui.weak(format!("· item {}: {}", ii + 1, item.matrix_type));
                            for r in 0..3 {
                                ui.monospace(format!(
                                    "  [{:7.3} {:7.3} {:7.3} {:8.2}]",
                                    m[r * 4],
                                    m[r * 4 + 1],
                                    m[r * 4 + 2],
                                    m[r * 4 + 3]
                                ));
                            }
                            // FoR hints against the loaded studies.
                            let src_hint = hint(&item.for_uid);
                            let dst_hint = hint(&reg.frame_of_reference_uid);
                            ui.weak(format!(
                                "  maps FoR …{}{} ▶ …{}{}",
                                tail(&item.for_uid),
                                src_hint,
                                tail(&reg.frame_of_reference_uid),
                                dst_hint
                            ));
                            match extras::matrix_to_rigid(m, invert) {
                                Some(rigid) => {
                                    let rp = rigid.params();
                                    ui.weak(format!(
                                        "  t = ({:.1}, {:.1}, {:.1}) mm  r = ({:.2}, {:.2}, {:.2})°{}",
                                        rp[3],
                                        rp[4],
                                        rp[5],
                                        rp[0].to_degrees(),
                                        rp[1].to_degrees(),
                                        rp[2].to_degrees(),
                                        if invert { "  (inverted)" } else { "" }
                                    ));
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut invert, "Invert")
                                            .on_hover_text(
                                                "Invert the matrix before applying (flip the mapping direction)",
                                            );
                                        // This workspace against each of
                                        // the others, both ways round.
                                        for other in &partners {
                                            for (fixed, moving) in
                                                [(slot, *other), (*other, slot)]
                                            {
                                                let label = format!(
                                                    "Apply as {} ▶ {}",
                                                    SLOT_NAMES[moving], SLOT_NAMES[fixed]
                                                );
                                                let tip = format!(
                                                    "Use this matrix as the transform mapping \
                                                     {} (fixed) coordinates into {} (moving)",
                                                    SLOT_NAMES[fixed], SLOT_NAMES[moving]
                                                );
                                                if ui
                                                    .add(egui::Button::new(label))
                                                    .on_hover_text(tip)
                                                    .clicked()
                                                {
                                                    if let Some(r2) =
                                                        extras::matrix_to_rigid(m, invert)
                                                    {
                                                        apply = Some((r2, fixed, moving));
                                                    }
                                                }
                                            }
                                        }
                                    });
                                }
                                None => {
                                    ui.weak("  (matrix is not a pure rigid transform - cannot apply)");
                                }
                            }
                        }
                        // A Deformable Spatial Registration's displacement
                        // lattice applies exactly as a matrix does.
                        if let Some(grid) = &reg.grid {
                            ui.weak(format!("· deformation grid: {}", grid.describe()));
                            let src_hint = fors
                                .iter()
                                .find(|(_, f)| *f == reg.grid_source_for_uid)
                                .map(|(s, _)| *s);
                            ui.horizontal(|ui| {
                                for (fixed, moving) in partners
                                    .iter()
                                    .flat_map(|o| [(slot, *o), (*o, slot)])
                                {
                                    let label = format!(
                                        "Apply grid as {} ▶ {}",
                                        SLOT_NAMES[moving], SLOT_NAMES[fixed]
                                    );
                                    let hint = match src_hint {
                                        Some(s) if s == fixed => {
                                            "The grid's own frame of reference matches this                                              workspace - this is the direction the file means"
                                        }
                                        Some(_) => {
                                            "The grid's frame of reference matches the *other*                                              workspace; applying it this way round inverts what                                              the file says"
                                        }
                                        None => {
                                            "Neither loaded workspace matches the grid's frame of                                              reference - check that this is the right pair"
                                        }
                                    };
                                    if enabled_tip_button(ui, true, label, hint) {
                                        apply_grid = Some((ri, fixed, moving));
                                    }
                                }
                            });
                        }
                        if ri + 1 < study.registrations.len() {
                            ui.separator();
                        }
                    }
                });
            self.reg_apply_invert = invert;
        }
        if rename.is_some() {
            self.rename_request = rename;
        }
        if focus.is_some() {
            self.tree_focus = focus;
        }
        if let Some((rigid, fixed_slot, moving_slot)) = apply {
            self.apply_external_transform_between(
                Transform3::rigid_only(rigid),
                registration::RegMethod::ElastixRigid,
                fixed_slot,
                moving_slot,
            );
        }
        if let Some((ri, fixed_slot, moving_slot)) = apply_grid {
            if let Some(field) = self.slots[slot]
                .study
                .as_ref()
                .and_then(|s| s.registrations.get(ri))
                .and_then(|r| r.grid.clone())
            {
                let center = field.origin;
                let transform = Transform3 {
                    rigid: registration::RigidTransform::identity(center),
                    warp: registration::Warp::Field(Arc::new(field)),
                    manual: None,
                };
                self.apply_external_transform_between(
                    transform,
                    registration::RegMethod::PlastimatchBSpline,
                    fixed_slot,
                    moving_slot,
                );
            }
        }
    }
}
